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
        const order = this.order;
        const knots: number[] = [];

        // Add order + 1 knots at the beginning
        const min = x[0];
        for (let i = 0; i <= order; i++) {
            knots.push(min);
        }

        // Add interior knots based on desired number of coefficients
        // For n data points, we want n coefficients
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
        const n = knots.length - order - 2;
        
        if (x >= knots[n + 1]) return n;
        if (x <= knots[order]) return order;

        let low = order;
        let high = n + 1;
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
                const temp = basis[r] / (right[r + 1] + left[j - r]);
                basis[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            basis[j] = saved;
        }

        return basis;
    }

    private createResult(coefficients: number[], x: number[], y: number[]): CobsResult {
        const design = this.createDesignMatrix(x, this.order, this.knots);
        const error = this.calculateError(design, y, coefficients);

        // Calculate fitted values and residuals
        const fitted = x.map(xi => {
            const basis = this.createDesignMatrix([xi], this.order, this.knots);
            let result = 0;
            for (let j = 0; j < coefficients.length; j++) {
                result += basis.get(0, j) * coefficients[j];
            }
            return result;
        });

        const residuals = y.map((yi, i) => yi - fitted[i]);

        const evaluate = (xi: number): number => {
            const basis = this.createDesignMatrix([xi], this.order, this.knots);
            let result = 0;
            for (let j = 0; j < coefficients.length; j++) {
                result += basis.get(0, j) * coefficients[j];
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
            order: this.order
        };
    }

    private calculateError(design: Matrix, y: number[], coefficients: number[]): number {
        let error = 0;
        for (let i = 0; i < y.length; i++) {
            let fitted = 0;
            for (let j = 0; j < coefficients.length; j++) {
                fitted += design.get(i, j) * coefficients[j];
            }
            error += Math.pow(y[i] - fitted, 2);
        }
        return error;
    }

    private solveConstrainedProblem(design: Matrix, y: number[], constraints: Constraint[]): { A: Matrix; b: number[] } {
        if (!constraints.length) {
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
                    }
                    break;
                case 'periodic':
                    result = this.generatePeriodicityConstraints(design);
                    break;
                case 'pointwise':
                    if (constraint.x !== undefined && constraint.y !== undefined && constraint.operator) {
                        result = this.generatePointwiseConstraints(design, constraint.x, constraint.y, constraint.operator);
                    }
                    break;
            }

            if (result) {
                constraintMatrices.push(result.A);
                constraintVectors.push(result.b);
            }
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
        const n = design.numCols;
        const m = design.numRows - 1;
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                const diff = design.get(i + 1, j) - design.get(i, j);
                if (Math.abs(diff) > this.tol) {
                    data.push(increasing ? -diff : diff);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
            b.push(0);
        }

        return {
            A: new Matrix({ rows: m, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private generateConvexityConstraints(design: Matrix, convex: boolean): { A: Matrix; b: number[] } {
        const n = design.numCols;
        const m = design.numRows - 2;  // We need at least 3 points for second derivative
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        // For each interior point
        for (let i = 0; i < m; i++) {
            // Get three consecutive points
            const p1 = design.getRow(i);
            const p2 = design.getRow(i + 1);
            const p3 = design.getRow(i + 2);

            // Calculate second derivative coefficients
            for (let j = 0; j < n; j++) {
                const coef = (p3[j] - p2[j]) - (p2[j] - p1[j]);  // Second difference
                if (Math.abs(coef) > this.tol) {
                    data.push(convex ? coef : -coef);  // Convex: second derivative >= 0
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
            b.push(0);
        }

        return {
            A: new Matrix({ rows: m, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private generatePeriodicityConstraints(design: Matrix): { A: Matrix; b: number[] } {
        const n = design.numCols;
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        for (let j = 0; j < n; j++) {
            const d1 = design.get(1, j) - design.get(0, j);
            const d2 = design.get(design.numRows - 1, j) - design.get(design.numRows - 2, j);
            if (Math.abs(d1 - d2) > this.tol) {
                data.push(d1 - d2);
                rowIndices.push(0);
                colIndices.push(j);
            }
        }
        b.push(0);

        return {
            A: new Matrix({ rows: 1, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private generatePointwiseConstraints(design: Matrix, x: number, y: number, operator: string): { A: Matrix; b: number[] } {
        const numBasis = design.numCols;
        const A = new Matrix({ rows: 1, cols: numBasis, data: [], rowIndices: [], colIndices: [] });
        const b = new Array(1).fill(0);

        const basis = this.createDesignMatrix([x], this.order, this.knots);
        for (let j = 0; j < numBasis; j++) {
            const value = basis.get(0, j);
            if (Math.abs(value) > this.tol) {
                switch (operator) {
                    case '=':
                        A.set(0, j, value);
                        b[0] = y;
                        break;
                    case '>=':
                        A.set(0, j, value);
                        b[0] = y;
                        break;
                    case '<=':
                        A.set(0, j, -value);
                        b[0] = -y;
                        break;
                    default:
                        throw new Error(`Unsupported operator: ${operator}`);
                }
            }
        }

        return { A, b };
    }

    fit(x: number[], y: number[], options: CobsOptions = {}): CobsResult {
        if (x.length !== y.length) {
            throw new Error('Input arrays x and y must have the same length');
        }

        if (x.length < 2) {
            throw new Error('At least two data points are required');
        }

        // Set default options
        const constraints = options.constraints || [];
        if (options.order !== undefined) {
            this.order = options.order;
        }

        // Generate knots if not provided
        this.knots = options.knots || this.generateKnots(x);

        // Create design matrix
        const design = this.createDesignMatrix(x, this.order, this.knots);

        try {
            // Try constrained fit first
            if (constraints.length > 0) {
                const { A, b } = this.solveConstrainedProblem(design, y, constraints);
                const solver = new LPSolver(A, b, []);
                const solution = solver.solve();

                if (solution) {
                    return this.createResult(solution.map(value => Math.round(value * 1e12) / 1e12), x, y);
                }
            }

            // Fall back to unconstrained fit
            const solution = design.solve(y);
            return this.createResult(solution.map(value => Math.round(value * 1e12) / 1e12), x, y);
        } catch (error) {
            // If everything fails, try simple least squares
            const solution = design.solve(y);
            return this.createResult(solution.map(value => Math.round(value * 1e12) / 1e12), x, y);
        }
    }
}
