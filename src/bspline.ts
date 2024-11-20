import { Matrix } from './matrix';

export class BSpline {
    private knots: number[];
    private degree: number;

    constructor(knots: number[], degree: number) {
        this.knots = knots;
        this.degree = degree;
    }

    get numCoefficients(): number {
        return this.knots.length - this.degree - 1;
    }

    getKnots(): number[] {
        return [...this.knots];
    }

    getDegree(): number {
        return this.degree;
    }

    evaluate(x: number): number[] {
        const n = this.numCoefficients;
        const result = new Array(n).fill(0);

        // Find the knot span containing x
        let span = this.findSpan(x);

        // Compute the basis functions
        const basis = this.computeBasisFunctions(span, x);

        // Fill in the non-zero basis functions
        for (let i = 0; i <= this.degree; i++) {
            result[span - this.degree + i] = basis[i];
        }

        return result;
    }

    evaluateDerivative(x: number): number[] {
        const n = this.numCoefficients;
        const result = new Array(n).fill(0);

        // Find the knot span containing x
        let span = this.findSpan(x);

        // Compute the derivative basis functions
        const basis = this.computeDerivativeBasisFunctions(span, x, 1);

        // Fill in the non-zero basis functions
        for (let i = 0; i <= this.degree; i++) {
            result[span - this.degree + i] = basis[i];
        }

        return result;
    }

    evaluateSecondDerivative(x: number): number[] {
        const n = this.numCoefficients;
        const result = new Array(n).fill(0);

        // Find the knot span containing x
        let span = this.findSpan(x);

        // Compute the second derivative basis functions
        const basis = this.computeDerivativeBasisFunctions(span, x, 2);

        // Fill in the non-zero basis functions
        for (let i = 0; i <= this.degree; i++) {
            result[span - this.degree + i] = basis[i];
        }

        return result;
    }

    private findSpan(x: number): number {
        const n = this.knots.length - this.degree - 2;

        if (x >= this.knots[n + 1]) {
            return n;
        }
        if (x <= this.knots[this.degree]) {
            return this.degree;
        }

        let low = this.degree;
        let high = n + 1;
        let mid = Math.floor((low + high) / 2);

        while (x < this.knots[mid] || x >= this.knots[mid + 1]) {
            if (x < this.knots[mid]) {
                high = mid;
            } else {
                low = mid;
            }
            mid = Math.floor((low + high) / 2);
        }

        return mid;
    }

    private computeBasisFunctions(span: number, x: number): number[] {
        const left = new Array(this.degree + 1).fill(0);
        const right = new Array(this.degree + 1).fill(0);
        const N = new Array(this.degree + 1).fill(0);

        N[0] = 1.0;
        for (let j = 1; j <= this.degree; j++) {
            left[j] = x - this.knots[span + 1 - j];
            right[j] = this.knots[span + j] - x;
            let saved = 0.0;

            for (let r = 0; r < j; r++) {
                const temp = N[r] / (right[r + 1] + left[j - r]);
                N[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            N[j] = saved;
        }

        return N;
    }

    private computeDerivativeBasisFunctions(span: number, x: number, deriv: number): number[] {
        const ndu = Array(this.degree + 1).fill(0).map(() => Array(this.degree + 1).fill(0));
        const left = new Array(this.degree + 1).fill(0);
        const right = new Array(this.degree + 1).fill(0);
        const ders = Array(deriv + 1).fill(0).map(() => Array(this.degree + 1).fill(0));

        ndu[0][0] = 1.0;
        for (let j = 1; j <= this.degree; j++) {
            left[j] = x - this.knots[span + 1 - j];
            right[j] = this.knots[span + j] - x;
            let saved = 0.0;

            for (let r = 0; r < j; r++) {
                ndu[j][r] = right[r + 1] + left[j - r];
                const temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        for (let j = 0; j <= this.degree; j++) {
            ders[0][j] = ndu[j][this.degree];
        }

        for (let r = 0; r <= this.degree; r++) {
            let s1 = 0;
            let s2 = 1;
            const a = Array(2).fill(0).map(() => Array(this.degree + 1).fill(0));
            a[0][0] = 1.0;

            for (let k = 1; k <= deriv; k++) {
                let d = 0.0;
                const rk = r - k;
                const pk = this.degree - k;

                if (r >= k) {
                    a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                    d = a[s2][0] * ndu[rk][pk];
                }

                const j1 = rk >= -1 ? 1 : -rk;
                const j2 = r - 1 <= pk ? k - 1 : this.degree - r;

                for (let j = j1; j <= j2; j++) {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                    d += a[s2][j] * ndu[rk + j][pk];
                }

                if (r <= pk) {
                    a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                    d += a[s2][k] * ndu[r][pk];
                }

                ders[k][r] = d;

                const j = s1;
                s1 = s2;
                s2 = j;
            }
        }

        let r = this.degree;
        for (let k = 1; k <= deriv; k++) {
            for (let j = 0; j <= this.degree; j++) {
                ders[k][j] *= r;
            }
            r *= (this.degree - k);
        }

        return ders[deriv];
    }

    public createDesignMatrix(x: number[]): Matrix {
        const n = this.knots.length - this.degree - 1;
        const m = x.length;
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];

        for (let i = 0; i < m; i++) {
            const basis = this.evaluate(x[i]);
            for (let j = 0; j < n; j++) {
                if (Math.abs(basis[j]) > 1e-10) {
                    data.push(basis[j]);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
        }

        return new Matrix({
            rows: m,
            cols: n,
            data,
            rowIndices,
            colIndices
        });
    }

    public createDerivativeMatrix(x: number[], order: number = 1): Matrix {
        const n = this.knots.length - this.degree - 1;
        const m = x.length;
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];

        // Add evaluation points between each x point to ensure proper derivatives
        const allX = [];
        for (let i = 0; i < m - 1; i++) {
            allX.push(x[i]);
            const mid = (x[i] + x[i + 1]) / 2;
            allX.push(mid);
        }
        allX.push(x[m - 1]);

        // Evaluate derivative at each x
        for (let i = 0; i < allX.length; i++) {
            const basis = this.evaluateDerivative(allX[i]);
            for (let j = 0; j < n; j++) {
                if (Math.abs(basis[j]) > 1e-10) {
                    data.push(basis[j]);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
        }

        return new Matrix({
            rows: allX.length,
            cols: n,
            data,
            rowIndices,
            colIndices
        });
    }
}
