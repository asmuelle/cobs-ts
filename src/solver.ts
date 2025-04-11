import { Matrix } from './matrix';

/**
 * Implementation of the Revised Simplex Method for sparse matrices
 */
export class LPSolver {
    private readonly tol = 1e-12;
    private readonly maxIter = 1000;

    private A: Matrix;
    private b: number[];
    private c: number[];
    private m: number;
    private n: number;

    constructor(A: Matrix, b: number[], c: number[]) {
        this.A = A;
        this.b = b;
        this.c = c;
        this.m = A.numRows;
        this.n = A.numCols;
    }

    solve(): number[] {
        // Initialize basis
        const basis = Array(this.m).fill(-1);
        const nonbasis = Array.from({ length: this.n }, (_, i) => i);

        // Add artificial variables if needed
        for (let i = 0; i < this.m; i++) {
            if (basis[i] === -1) {
                const col = this.findBasisColumn(i);
                if (col !== -1) {
                    basis[i] = col;
                    nonbasis.splice(nonbasis.indexOf(col), 1);
                } else {
                    // Add artificial variable
                    basis[i] = nonbasis.pop()!;
                }
            }
        }

        // Main simplex loop
        for (let iter = 0; iter < this.maxIter; iter++) {
            // Get basis inverse
            const Binv = this.getBasisInverse(basis);
            if (!Binv) {
                // If we can't get a basis inverse, return zero solution
                return Array(this.n).fill(0);
            }

            // Check feasibility
            const xB = this.multiplyMatrixVector(Binv, this.b);
            const feasible = xB.every(x => x >= -this.tol);
            if (!feasible) {
                // If solution is not feasible, return zero solution
                return Array(this.n).fill(0);
            }

            // Compute reduced costs
            const cB = basis.map(j => this.c[j]);
            const y = this.multiplyVectorMatrix(cB, Binv);
            
            // Find entering variable
            let enteringCol = -1;
            let minReducedCost = -this.tol;
            for (const j of nonbasis) {
                const aj = this.getColumn(j);
                const reducedCost = this.c[j] - this.dotProduct(y, aj);
                if (reducedCost < minReducedCost) {
                    minReducedCost = reducedCost;
                    enteringCol = j;
                }
            }

            // Optimality check
            if (enteringCol === -1) {
                // Optimal solution found
                const x = Array(this.n).fill(0);
                for (let i = 0; i < this.m; i++) {
                    if (basis[i] >= 0 && basis[i] < this.n) {
                        x[basis[i]] = Math.max(0, xB[i]);
                    }
                }
                return x;
            }

            // Compute step direction
            const d = this.multiplyMatrixVector(Binv, this.getColumn(enteringCol));

            // Find leaving variable
            let leavingRow = -1;
            let minRatio = Infinity;
            for (let i = 0; i < this.m; i++) {
                if (d[i] > this.tol) {
                    const ratio = xB[i] / d[i];
                    if (ratio < minRatio) {
                        minRatio = ratio;
                        leavingRow = i;
                    }
                }
            }

            if (leavingRow === -1) {
                // Problem is unbounded, return zero solution
                return Array(this.n).fill(0);
            }

            // Update basis
            const temp = basis[leavingRow];
            basis[leavingRow] = enteringCol;
            nonbasis[nonbasis.indexOf(enteringCol)] = temp;
        }

        // If max iterations reached, return zero solution
        return Array(this.n).fill(0);
    }

    private findBasisColumn(row: number): number {
        for (let j = 0; j < this.n; j++) {
            let isUnitVector = true;
            for (let i = 0; i < this.m; i++) {
                const value = this.A.get(i, j);
                if (i === row && Math.abs(value - 1) > this.tol) {
                    isUnitVector = false;
                    break;
                }
                if (i !== row && Math.abs(value) > this.tol) {
                    isUnitVector = false;
                    break;
                }
            }
            if (isUnitVector) {
                return j;
            }
        }
        return -1;
    }

    private getBasisInverse(basis: number[]): Matrix | null {
        const m = this.m;
        const B = Matrix.zeros(m, m);
        let nonZeroCount = 0;
        
        // Extract basis columns
        for (let i = 0; i < m; i++) {
            if (basis[i] >= 0 && basis[i] < this.n) {
                for (let j = 0; j < m; j++) {
                    const value = this.A.get(j, basis[i]);
                    if (Math.abs(value) > this.tol) {
                        nonZeroCount++;
                        B.set(j, i, value);
                    }
                }
            }
        }

        // If matrix is empty or nearly empty, return null
        if (nonZeroCount < m) {
            return null;
        }

        return B.inverse();
    }

    private getColumn(j: number): number[] {
        const col = Array(this.m).fill(0);
        for (let i = 0; i < this.m; i++) {
            col[i] = this.A.get(i, j);
        }
        return col;
    }

    private multiplyMatrixVector(A: Matrix, x: number[]): number[] {
        const result = Array(A.numRows).fill(0);
        for (let i = 0; i < A.numRows; i++) {
            for (let j = 0; j < A.numCols; j++) {
                result[i] += A.get(i, j) * x[j];
            }
        }
        return result;
    }

    private multiplyVectorMatrix(x: number[], A: Matrix): number[] {
        const result = Array(A.numCols).fill(0);
        for (let j = 0; j < A.numCols; j++) {
            for (let i = 0; i < A.numRows; i++) {
                result[j] += x[i] * A.get(i, j);
            }
        }
        return result;
    }

    private dotProduct(x: number[], y: number[]): number {
        return x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    }
}
