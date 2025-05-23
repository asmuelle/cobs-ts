import { Matrix } from './matrix';
import { LPSolver } from './solver';
import {
    Constraint,
    CobsOptions,
    CobsResult
} from './types';

export class Cobs {
    private order: number = 4;
    private knots: number[] = [];
    private readonly tol: number = 1e-10;

    private generateKnots(x: number[]): number[] {
        const n = x.length;
        const order = this.order; // Relies on this.order being set
        const knots: number[] = [];

        // Add order + 1 knots at the beginning
        const min = x[0];
        for (let i = 0; i <= order; i++) {
            knots.push(min);
        }

        // Add interior knots based on desired number of coefficients
        const numInteriorKnots = n - (order + 1);
        if (numInteriorKnots > 0) {
            const step = (x[n - 1] - x[0]) / (numInteriorKnots + 1);
            for (let i = 1; i <= numInteriorKnots; i++) {
                knots.push(min + i * step);
            }
        }

        // Add order + 1 knots at the end
        const max = x[n - 1];
        for (let i = 0; i <= order; i++) {
            knots.push(max);
        }

        return knots;
    }

    private createDesignMatrix(x: number[], order: number, knots: number[]): Matrix {
        const n = x.length;
        const numBasis = knots.length - order - 1;
        if (numBasis < 0) { 
             throw new Error('Insufficient knots for the given order to create basis functions.');
        }
        const design = Array(n).fill(0).map(() => Array(numBasis).fill(0));

        for (let i = 0; i < n; i++) {
            const xi = x[i];
            const span = this.findSpan(xi, order, knots);
            const basis = this.evaluateBasis(xi, order, span, knots);

            for (let j = 0; j <= order; j++) {
                if (span - order + j >= 0 && span - order + j < numBasis) {
                    design[i][span - order + j] = basis[j];
                }
            }
        }

        return new Matrix(design.map(row => row.map(value => Math.round(value * 1e12) / 1e12)));
    }

    private findSpan(x: number, order: number, knots: number[]): number {
        const n_deboor = knots.length - order - 2;
        
        if (x >= knots[n_deboor + 1]) { 
            return n_deboor;
        }
        if (x <= knots[order]) { 
            return order;
        }

        let low = order;
        let high = n_deboor + 1; 
        let mid = Math.floor((low + high) / 2);

        while (x < knots[mid] || x >= knots[mid + 1]) {
            if (x < knots[mid]) {
                high = mid;
            } else {
                low = mid;
            }
            mid = Math.floor((low + high) / 2);
        }

        return mid;
    }

    private evaluateBasis(x: number, order: number, span: number, knots: number[]): number[] {
        const basis = Array(order + 1).fill(0);
        const left = Array(order + 1).fill(0);
        const right = Array(order + 1).fill(0);

        basis[0] = 1;
        for (let j = 1; j <= order; j++) {
            left[j] = x - knots[span + 1 - j];
            right[j] = knots[span + j] - x;
            let saved = 0;

            for (let r = 0; r < j; r++) {
                const den = right[r + 1] + left[j - r];
                let temp = 0;
                if (den !== 0) { 
                    temp = basis[r] / den;
                }
                basis[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            basis[j] = saved;
        }

        return basis;
    }

    private createResult(coefficients: number[], x: number[], y: number[], tau?: number): CobsResult {
        const design = this.createDesignMatrix(x, this.order, this.knots);
        const error = this.calculateError(design, y, coefficients);

        const fitted = x.map(xi => {
            const basisMatrixRow = this.createDesignMatrix([xi], this.order, this.knots); 
            let result = 0;
            for (let j = 0; j < coefficients.length; j++) {
                if (j < basisMatrixRow.numCols) {
                    result += basisMatrixRow.get(0, j) * coefficients[j];
                }
            }
            return result;
        });

        const residuals = y.map((yi, i) => yi - fitted[i]);

        const evaluate = (xi: number): number => {
            const basisMatrixRow = this.createDesignMatrix([xi], this.order, this.knots);
            let result = 0;
            for (let j = 0; j < coefficients.length; j++) {
                 if (j < basisMatrixRow.numCols) {
                    result += basisMatrixRow.get(0, j) * coefficients[j];
                }
            }
            return result;
        };

        const evaluateSecondDerivative = (xi: number): number => {
            const h = 1e-6; 
            const fxph = evaluate(xi + h);
            const fx = evaluate(xi);
            const fxmh = evaluate(xi - h);
            return (fxph - 2 * fx + fxmh) / (h * h);
        };

        const pp = { evaluate, evaluateSecondDerivative };

        return {
            coefficients,
            error,
            fit: { 
                pp,
                coefficients,
                fitted,
                residuals
            },
            pp, 
            evaluate, 
            knots: this.knots,
            order: this.order,
            tau: tau 
        };
    }

    private calculateError(design: Matrix, y: number[], coefficients: number[]): number {
        let error = 0;
        for (let i = 0; i < y.length; i++) {
            let fitted_i = 0;
            for (let j = 0; j < coefficients.length; j++) {
                if (j < design.numCols) { 
                    fitted_i += design.get(i, j) * coefficients[j];
                }
            }
            error += Math.pow(y[i] - fitted_i, 2);
        }
        return error;
    }

    private solveConstrainedProblem(design: Matrix, y: number[], constraints: Constraint[]): { A: Matrix; b: number[] } {
        if (!constraints || constraints.length === 0) { 
            return {
                A: new Matrix({ rows: 0, cols: design.numCols, data: [], rowIndices: [], colIndices: [] }),
                b: []
            };
        }

        const constraintMatrices: Matrix[] = [];
        const constraintVectors: number[][] = [];

        for (const constraint of constraints) {
            let result: { A: Matrix; b: number[] } | null = null;

            switch (constraint.type) {
                case 'monotone':
                    if (constraint.increasing !== undefined) {
                        result = this.generateMonotonicityConstraints(design, constraint.increasing);
                    }
                    break;
                case 'convex':
                    if (constraint.convex !== undefined) { 
                         result = this.generateConvexityConstraints(design, constraint.convex);
                    } else { 
                         result = this.generateConvexityConstraints(design, true); 
                    }
                    break;
                case 'concave':
                    result = this.generateConvexityConstraints(design, false); // false for concave
                    break;
                case 'periodic':
                    result = this.generatePeriodicityConstraints(design);
                    break;
                case 'pointwise':
                    if (constraint.x !== undefined && constraint.y !== undefined && constraint.operator) {
                        result = this.generatePointwiseConstraints(design, constraint.x, constraint.y, constraint.operator);
                    }
                    break;
                default:
                    const unknownConstraint: any = constraint;
                    throw new Error(`Unsupported constraint type: ${unknownConstraint.type}`);
            }

            if (result) {
                constraintMatrices.push(result.A);
                constraintVectors.push(result.b);
            }
        }

        if (constraintMatrices.length === 0) { 
             return {
                A: new Matrix({ rows: 0, cols: design.numCols, data: [], rowIndices: [], colIndices: [] }),
                b: []
            };
        }

        const totalRows = constraintMatrices.reduce((sum, m) => sum + m.numRows, 0);
        const A = new Matrix({ rows: totalRows, cols: design.numCols, data: [], rowIndices: [], colIndices: [] });
        const b: number[] = [];

        let currentRow = 0;
        for (let i = 0; i < constraintMatrices.length; i++) {
            const matrix = constraintMatrices[i];
            const vector = constraintVectors[i];

            for (let row = 0; row < matrix.numRows; row++) {
                for (let col = 0; col < matrix.numCols; col++) {
                    const value = matrix.get(row, col);
                    if (Math.abs(value) > this.tol) {
                        A.set(currentRow, col, value);
                    }
                }
                b.push(vector[row]);
                currentRow++;
            }
        }

        return { A, b };
    }

    private generateMonotonicityConstraints(design: Matrix, increasing: boolean): { A: Matrix; b: number[] } {
        const n_coeffs = design.numCols; 
        const n_points = design.numRows; 
        
        const m = n_points - 1; 
        if (m <= 0) { 
             return { A: new Matrix({ rows: 0, cols: n_coeffs, data: [], rowIndices: [], colIndices: [] }), b: [] };
        }

        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = Array(m).fill(0); 

        for (let i = 0; i < m; i++) { 
            for (let j = 0; j < n_coeffs; j++) { 
                const diff = design.get(i + 1, j) - design.get(i, j); 
                if (Math.abs(diff) > this.tol) {
                    data.push(increasing ? -diff : diff);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
        }

        return {
            A: new Matrix({ rows: m, cols: n_coeffs, data, rowIndices, colIndices }),
            b
        };
    }

    private generateConvexityConstraints(design: Matrix, convex: boolean): { A: Matrix; b: number[] } {
        const n_coeffs = design.numCols;
        const n_points = design.numRows;
        
        const m = n_points - 2; 
        if (m <= 0) {
            return { A: new Matrix({ rows: 0, cols: n_coeffs, data: [], rowIndices: [], colIndices: [] }), b: [] };
        }
        
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = Array(m).fill(0); 

        for (let i = 0; i < m; i++) { 
            for (let j = 0; j < n_coeffs; j++) {
                const val_i = design.get(i, j);
                const val_i1 = design.get(i + 1, j);
                const val_i2 = design.get(i + 2, j);
                const second_diff_coeff = val_i2 - 2 * val_i1 + val_i; 
                
                if (Math.abs(second_diff_coeff) > this.tol) {
                    data.push(convex ? -second_diff_coeff : second_diff_coeff); 
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
        }

        return {
            A: new Matrix({ rows: m, cols: n_coeffs, data, rowIndices, colIndices }),
            b
        };
    }

    private generatePeriodicityConstraints(design: Matrix): { A: Matrix; b: number[] } {
        const n_coeffs = design.numCols;
       
        if (design.numRows < 2) { 
             return { A: new Matrix({ rows: 0, cols: n_coeffs, data: [], rowIndices: [], colIndices: [] }), b: [] };
        }
        
        const data: number[] = [];
        const rowIndices: number[] = []; 
        const colIndices: number[] = [];
        const b_vector: number[] = [0]; 

        if (design.numRows >= 2) { 
            for (let j = 0; j < n_coeffs; j++) {
                const first_diff_coeff = design.get(1, j) - design.get(0, j);
                const last_diff_coeff = design.get(design.numRows - 1, j) - design.get(design.numRows - 2, j);
                const constraint_coeff = first_diff_coeff - last_diff_coeff;
                if (Math.abs(constraint_coeff) > this.tol) {
                    data.push(constraint_coeff);
                    rowIndices.push(0);
                    colIndices.push(j);
                }
            }
        }
        
        return {
            A: new Matrix({ rows: (data.length > 0 ? 1:0) , cols: n_coeffs, data, rowIndices, colIndices }), 
            b: (data.length > 0 ? b_vector:[])
        };
    }

    private generatePointwiseConstraints(design: Matrix, x_val: number, y_val: number, operator: string): { A: Matrix; b: number[] } {
        const numBasis = design.numCols;
        // Ensure this.order and this.knots are correctly set and accessible
        // If createDesignMatrix is computationally expensive, consider optimizing if called frequently
        const basisAtXRow = this.createDesignMatrix([x_val], this.order, this.knots); 
        const basisCoeffsAtX: number[] = [];
        for (let j = 0; j < numBasis; j++) {
            // Ensure get method handles potential out-of-bounds if basisAtXRow is not as expected
            basisCoeffsAtX.push(basisAtXRow.get(0, j));
        }

        let A_data: number[] = [];
        const A_rowIndices: number[] = [];
        const A_colIndices: number[] = [];
        let b_vector: number[] = [];
        let numConstraintRows = 0;

        if (operator === '=') {
            numConstraintRows = 0; // Will be incremented if data is added
            let actualRowsAdded = 0;

            // Constraint 1: basisCoeffsAtX * coeffs <= y_val
            let row1HasData = false;
            for (let j = 0; j < numBasis; j++) {
                if (Math.abs(basisCoeffsAtX[j]) > this.tol) {
                    A_data.push(basisCoeffsAtX[j]);
                    A_rowIndices.push(actualRowsAdded); // Use actualRowsAdded for row index
                    A_colIndices.push(j);
                    row1HasData = true;
                }
            }
            if (row1HasData) {
                b_vector.push(y_val);
                actualRowsAdded++;
            }

            // Constraint 2: basisCoeffsAtX * coeffs >= y_val  => -basisCoeffsAtX * coeffs <= -y_val
            let row2HasData = false;
            for (let j = 0; j < numBasis; j++) {
                if (Math.abs(basisCoeffsAtX[j]) > this.tol) { // Check magnitude of original coefficient
                    A_data.push(-basisCoeffsAtX[j]);
                    A_rowIndices.push(actualRowsAdded); // Use actualRowsAdded for row index
                    A_colIndices.push(j);
                    row2HasData = true;
                }
            }
            if (row2HasData) {
                b_vector.push(-y_val);
                actualRowsAdded++;
            }
            numConstraintRows = actualRowsAdded;

        } else if (operator === '>=') {
            // basisCoeffsAtX * coeffs >= y_val => -basisCoeffsAtX * coeffs <= -y_val
            let rowHasData = false;
            for (let j = 0; j < numBasis; j++) {
                if (Math.abs(basisCoeffsAtX[j]) > this.tol) {
                    A_data.push(-basisCoeffsAtX[j]);
                    A_rowIndices.push(0);
                    A_colIndices.push(j);
                    rowHasData = true;
                }
            }
            if (rowHasData) {
                b_vector.push(-y_val);
                numConstraintRows = 1;
            } else {
                numConstraintRows = 0;
            }

        } else if (operator === '<=') {
            // basisCoeffsAtX * coeffs <= y_val
            let rowHasData = false;
            for (let j = 0; j < numBasis; j++) {
                if (Math.abs(basisCoeffsAtX[j]) > this.tol) {
                    A_data.push(basisCoeffsAtX[j]);
                    A_rowIndices.push(0);
                    A_colIndices.push(j);
                    rowHasData = true;
                }
            }
            if (rowHasData) {
                b_vector.push(y_val);
                numConstraintRows = 1;
            } else {
                numConstraintRows = 0;
            }
        } else {
            throw new Error(`Unsupported operator: ${operator}`);
        }
        
        // Only create a matrix if there are actual constraints
        if (numConstraintRows > 0) {
            const A = new Matrix({ rows: numConstraintRows, cols: numBasis, data: A_data, rowIndices: A_rowIndices, colIndices: A_colIndices });
            return { A, b: b_vector };
        } else {
            // Return an empty matrix and vector if no constraint data was generated
            return { 
                A: new Matrix({ rows: 0, cols: numBasis, data: [], rowIndices: [], colIndices: [] }), 
                b: [] 
            };
        }
    }

    fit(x: number[], y: number[], options: CobsOptions = {}): CobsResult {
        if (x.length !== y.length) {
            throw new Error('Input arrays x and y must have the same length');
        }

        if (x.length < 2) { 
            throw new Error('At least two data points are required');
        }
        
        const constraints = options.constraints || []; 

        if (options.order !== undefined) { 
            if (options.order < 1) {
                throw new Error('Order must be an integer greater than or equal to 1.');
            }
            this.order = options.order;
        } else {
            this.order = 4; // Default order if not specified
        }
        const tauValue = options.tau;
       
        if (options.knots) {
            if (options.knots.length < 2 * this.order) { 
                throw new Error(`Provided knots array is too short for order ${this.order}. Minimum length is ${2 * this.order}.`);
            }
            for (let i = 0; i < options.knots.length - 1; i++) {
                if (options.knots[i] > options.knots[i+1]) {
                    throw new Error('Provided knots array must be sorted non-decreasingly.');
                }
            }
            this.knots = options.knots;
        } else {
            if (x.length === 0) { 
                 throw new Error("Cannot generate knots for empty x array.");
            }
            this.knots = this.generateKnots(x); // this.order is now correctly set
        }

        const design = this.createDesignMatrix(x, this.order, this.knots);
        if (design.numRows === 0 || design.numCols === 0) {
            throw new Error('Design matrix is empty, cannot proceed with fit.');
        }

        let solverSolution: number[] | null = null; 

        if (constraints && constraints.length > 0) {
            try {
                const { A, b } = this.solveConstrainedProblem(design, y, constraints);
                
                if (A.numRows > 0) { // Only attempt LP if matrix A actually has constraint rows
                    // Create a 'c' vector for the objective function: minimize sum of coefficients
                    const objectiveCoeffs = Array(design.numCols).fill(1.0); 
                    const solver = new LPSolver(A, b, objectiveCoeffs); // Pass actual objective coefficients
                    solverSolution = solver.solve(); // Can be null if LP is infeasible
                } else {
                    // If constraints resulted in an empty A matrix, solution remains null, proceed to unconstrained.
                    solverSolution = null;
                }
            } catch (error: any) {
                if (error && error.message && 
                    (error.message.includes('Unsupported constraint type') || error.message.includes('Unsupported operator'))) {
                    throw error; // Re-throw specific errors
                }
                // For other unexpected errors, solution remains null, fall back to unconstrained.
                // Optional: console.warn('An unexpected error occurred during constrained fitting attempt:', error);
                solverSolution = null;
            }
        }

        let final_coeffs: number[];
        if (solverSolution) {
             // Ensure solution has correct number of coefficients before creating result
            if (solverSolution.length !== design.numCols) {
                // This case should ideally not happen if LPSolver is consistent with design.numCols
                console.warn("LPSolver solution length differs from design matrix columns. Using unconstrained fit.");
                const unconstrainedFallbackSolution = design.solve(y);
                final_coeffs = unconstrainedFallbackSolution.map(value => Math.round(value * 1e12) / 1e12);
            } else {
                final_coeffs = solverSolution.map(value => Math.round(value * 1e12) / 1e12);
            }
        } else {
            // Unconstrained fit:
            // 1. No constraints provided.
            // 2. Constraints resulted in empty A.
            // 3. LPSolver.solve() returned null (infeasible or not attempted).
            // 4. Unexpected error during constraint processing (not re-thrown).
            const unconstrainedSolution = design.solve(y);
            final_coeffs = unconstrainedSolution.map(value => Math.round(value * 1e12) / 1e12);
        }
        return this.createResult(final_coeffs, x, y, tauValue);
    }
}
