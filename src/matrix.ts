import { create, all } from 'mathjs';

const math = create(all);

export interface MatrixData {
    rows: number;
    cols: number;
    data: number[];
    rowIndices: number[];
    colIndices: number[];
}

export class Matrix {
    private dense: number[][];
    public rows: number;
    public cols: number;

    constructor(data: number[][] | MatrixData) {
        if (Array.isArray(data)) {
            // Initialize from 2D array
            if (!data || !data.length || !data[0] || !data[0].length) {
                throw new Error('Invalid matrix data: empty or malformed array');
            }

            this.dense = data.map(row => [...row]);
            this.rows = data.length;
            this.cols = data[0].length;

            // Ensure all rows have same length
            if (!data.every(row => row.length === this.cols)) {
                throw new Error('Invalid matrix data: rows have different lengths');
            }
        } else {
            // Initialize from sparse format
            this.rows = data.rows;
            this.cols = data.cols;
            this.dense = Array(data.rows).fill(0).map(() => Array(data.cols).fill(0));

            // Fill in non-zero elements
            for (let i = 0; i < data.data.length; i++) {
                this.dense[data.rowIndices[i]][data.colIndices[i]] = data.data[i];
            }
        }
    }

    get numRows(): number {
        return this.rows;
    }

    get numCols(): number {
        return this.cols;
    }

    get(i: number, j: number): number {
        if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
            throw new Error(`Invalid indices: (${i}, ${j})`);
        }
        return this.dense[i][j];
    }

    set(i: number, j: number, value: number): void {
        if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
            throw new Error(`Invalid indices: (${i}, ${j})`);
        }
        this.dense[i][j] = value;
    }

    multiply(other: Matrix | number[]): Matrix | number[] {
        if (Array.isArray(other)) {
            // Matrix-vector multiplication
            if (this.cols !== other.length) {
                throw new Error('Matrix dimensions do not match for multiplication');
            }
            const result = Array(this.rows).fill(0);
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    result[i] += this.get(i, j) * other[j];
                }
            }
            return result;
        } else {
            // Matrix-matrix multiplication
            if (this.cols !== other.numRows) {
                throw new Error('Matrix dimensions do not match for multiplication');
            }
            const result = new Matrix(Array(this.rows).fill(0).map(() => Array(other.numCols).fill(0)));
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < other.numCols; j++) {
                    for (let k = 0; k < this.cols; k++) {
                        result.dense[i][j] += this.dense[i][k] * other.dense[k][j];
                    }
                }
            }
            return result;
        }
    }

    copy(): Matrix {
        return new Matrix(this.dense.map(row => [...row]));
    }

    static identity(n: number): Matrix {
        const matrix = Array(n).fill(0).map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            matrix[i][i] = 1;
        }
        return new Matrix(matrix);
    }

    inverse(): Matrix | null {
        if (this.rows !== this.cols) {
            throw new Error('Matrix must be square to compute inverse');
        }

        const n = this.rows;
        const augmented = new Matrix(Array(n).fill(0).map((_, i) => 
            [...this.dense[i], ...Array(n).fill(0).map((_, j) => i === j ? 1 : 0)]
        ));

        // Gaussian elimination with partial pivoting
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxEl = Math.abs(augmented.get(i, i));
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                const absEl = Math.abs(augmented.get(k, i));
                if (absEl > maxEl) {
                    maxEl = absEl;
                    maxRow = k;
                }
            }

            // Check if matrix is singular
            if (maxEl < 1e-10) {
                return null;
            }

            // Swap maximum row with current row
            if (maxRow !== i) {
                for (let k = i; k < 2 * n; k++) {
                    const tmp = augmented.get(i, k);
                    augmented.set(i, k, augmented.get(maxRow, k));
                    augmented.set(maxRow, k, tmp);
                }
            }

            // Make all rows below this one 0 in current column
            for (let k = i + 1; k < n; k++) {
                const c = -augmented.get(k, i) / augmented.get(i, i);
                for (let j = i; j < 2 * n; j++) {
                    if (i === j) {
                        augmented.set(k, j, 0);
                    } else {
                        augmented.set(k, j, augmented.get(k, j) + c * augmented.get(i, j));
                    }
                }
            }
        }

        // Back substitution
        for (let i = n - 1; i >= 0; i--) {
            const c = 1 / augmented.get(i, i);
            for (let j = 0; j < n; j++) {
                augmented.set(i, j + n, augmented.get(i, j + n) * c);
            }
            for (let k = 0; k < i; k++) {
                for (let j = 0; j < n; j++) {
                    augmented.set(k, j + n, augmented.get(k, j + n) - augmented.get(k, i) * augmented.get(i, j + n));
                }
            }
        }

        // Extract right half
        const inv = new Matrix(Array(n).fill(0).map((_, i) => 
            Array(n).fill(0).map((_, j) => augmented.get(i, j + n))
        ));

        return inv;
    }

    solve(b: number[] | any): number[] {
        if (b.length !== this.rows) {
            throw new Error('Right-hand side vector must match matrix dimensions');
        }

        // For non-square matrices or if direct solve fails, use regularized least squares
        // Solve (A^T A + λI)x = A^T b
        const AT = this.transpose();
        const ATA = AT.multiply(this) as Matrix;

        // Add regularization
        const lambda = 1e-10;
        for (let i = 0; i < ATA.cols; i++) {
            ATA.set(i, i, ATA.get(i, i) + lambda);
        }

        const ATb = AT.multiply(b) as number[];
        const inv = ATA.inverse();

        if (!inv) {
            throw new Error('Matrix is still singular after regularization');
        }

        return inv.multiply(ATb) as number[];
    }

    getRow(i: number): number[] {
        if (i < 0 || i >= this.rows) {
            throw new Error(`Row index ${i} out of bounds [0, ${this.rows})`);
        }
        return [...this.dense[i]];
    }

    getColumn(j: number): number[] {
        if (j < 0 || j >= this.cols) {
            throw new Error(`Column index ${j} out of bounds [0, ${this.cols})`);
        }
        return this.dense.map(row => row[j]);
    }

    scale(factor: number): Matrix {
        const scaled = new Matrix(this.dense);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                scaled.dense[i][j] = this.dense[i][j] * factor;
            }
        }
        return scaled;
    }

    getBasisInverse(basis: number[]): Matrix | null {
        const m = this.rows;
        const B = new Matrix(Array(m).fill(0).map(() => Array(m).fill(0)));
        
        // Extract basis columns
        for (let i = 0; i < m; i++) {
            if (basis[i] >= 0) {
                for (let j = 0; j < m; j++) {
                    B.dense[j][i] = this.dense[j][basis[i]];
                }
            }
        }

        try {
            const inv = math.inv(B.toDense());
            return new Matrix(inv as number[][]);
        } catch (error) {
            return null;
        }
    }

    toDense(): number[][] {
        return this.dense;
    }

    static zeros(rows: number, cols: number): Matrix {
        return new Matrix(Array(rows).fill(0).map(() => Array(cols).fill(0)));
    }

    clone(): Matrix {
        return this.copy();
    }

    getMaxAbsValue(): number {
        return Math.max(...this.dense.flat().map(Math.abs));
    }

    transpose(): Matrix {
        const result = new Matrix(Array(this.cols).fill(0).map(() => Array(this.rows).fill(0)));
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.dense[j][i] = this.dense[i][j];
            }
        }
        return result;
    }
}
