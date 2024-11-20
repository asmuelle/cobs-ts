import { BSpline } from './bspline';

/**
 * Represents a polynomial piece of a spline
 */
export class Polynomial {
    private coefficients: number[];

    constructor(coefficients: number[]) {
        this.coefficients = [...coefficients];
    }

    /**
     * Evaluate polynomial at point x
     */
    evaluate(x: number): number {
        let result = 0;
        let xPower = 1;
        
        for (let i = 0; i < this.coefficients.length; i++) {
            result += this.coefficients[i] * xPower;
            xPower *= x;
        }
        
        return result;
    }

    /**
     * Get derivative polynomial
     */
    derivative(): Polynomial {
        if (this.coefficients.length <= 1) {
            return new Polynomial([0]);
        }
        
        const derivCoef = this.coefficients
            .slice(1)
            .map((c, i) => c * (i + 1));
        
        return new Polynomial(derivCoef);
    }

    /**
     * Get polynomial coefficients
     */
    getCoefficients(): number[] {
        return [...this.coefficients];
    }
}

/**
 * Represents a piecewise polynomial function
 */
export class PiecewisePolynomial {
    private pieces: Polynomial[];
    private breakpoints: number[];
    private degree: number;

    constructor(pieces: Polynomial[], breakpoints: number[], degree: number) {
        if (pieces.length !== breakpoints.length - 1) {
            throw new Error('Number of pieces must be equal to number of intervals');
        }
        
        this.pieces = pieces;
        this.breakpoints = breakpoints;
        this.degree = degree;
    }

    /**
     * Evaluate piecewise polynomial at point x
     */
    evaluate(x: number): number {
        // Find the appropriate piece
        let i = 0;
        while (i < this.breakpoints.length - 1 && x > this.breakpoints[i + 1]) {
            i++;
        }
        
        if (i >= this.pieces.length) {
            return this.pieces[this.pieces.length - 1].evaluate(x);
        }
        
        return this.pieces[i].evaluate(x - this.breakpoints[i]);
    }

    /**
     * Get derivative piecewise polynomial
     */
    derivative(): PiecewisePolynomial {
        const derivPieces = this.pieces.map(p => p.derivative());
        return new PiecewisePolynomial(derivPieces, this.breakpoints, this.degree - 1);
    }

    /**
     * Convert B-spline representation to piecewise polynomial
     */
    static fromBSpline(knots: number[], coefficients: number[], degree: number): PiecewisePolynomial {
        const n = knots.length - degree - 1;
        const pieces: Polynomial[] = [];
        const breakpoints: number[] = [];
        
        // Convert each B-spline segment to polynomial form
        for (let i = degree; i < n; i++) {
            const localKnots = knots.slice(i - degree, i + degree + 1);
            const localCoefs = coefficients.slice(i - degree, i + 1);
            
            // Convert B-spline segment to Bézier form
            const bezierCoefs = this.bsplineToBezier(localKnots, localCoefs, degree);
            
            // Convert Bézier form to power basis
            const powerCoefs = this.bezierToPower(bezierCoefs, knots[i], knots[i + 1]);
            
            pieces.push(new Polynomial(powerCoefs));
            breakpoints.push(knots[i]);
        }
        breakpoints.push(knots[n]);
        
        return new PiecewisePolynomial(pieces, breakpoints, degree);
    }

    /**
     * Convert B-spline coefficients to Bézier form for a segment
     */
    private static bsplineToBezier(
        knots: number[],
        coefficients: number[],
        degree: number
    ): number[] {
        const n = coefficients.length;
        const result = [...coefficients];
        
        for (let r = 1; r <= degree; r++) {
            for (let i = 0; i < n - r; i++) {
                const alpha = (knots[i + degree + 1] - knots[i + r]) /
                            (knots[i + degree + 1] - knots[i]);
                result[i] = alpha * result[i] + (1 - alpha) * result[i + 1];
            }
        }
        
        return result;
    }

    /**
     * Convert Bézier coefficients to power basis form
     */
    private static bezierToPower(
        bezierCoefs: number[],
        t0: number,
        t1: number
    ): number[] {
        const n = bezierCoefs.length - 1;
        const result = new Array(n + 1).fill(0);
        const dt = t1 - t0;
        
        // Pascal's triangle for binomial coefficients
        const pascal = Array(n + 1).fill(0).map(() => Array(n + 1).fill(0));
        pascal[0][0] = 1;
        for (let i = 1; i <= n; i++) {
            pascal[i][0] = 1;
            for (let j = 1; j <= i; j++) {
                pascal[i][j] = pascal[i - 1][j - 1] + pascal[i - 1][j];
            }
        }
        
        // Convert to power basis
        for (let i = 0; i <= n; i++) {
            for (let j = i; j <= n; j++) {
                const coef = pascal[j][i] * Math.pow(-t0, j - i) / Math.pow(dt, j);
                result[i] += coef * bezierCoefs[j];
            }
        }
        
        return result;
    }

    /**
     * Get breakpoints
     */
    getBreakpoints(): number[] {
        return [...this.breakpoints];
    }

    /**
     * Get polynomial pieces
     */
    getPieces(): Polynomial[] {
        return [...this.pieces];
    }

    /**
     * Get degree
     */
    getDegree(): number {
        return this.degree;
    }
}
