import { create, all } from 'mathjs';
import { BSpline } from './bspline';

const math = create(all);

export class PiecewisePolynomial {
    private knots: number[];
    private coefficients: number[];
    private degree: number;
    private bspline: BSpline;

    constructor(knots: number[], coefficients: number[], degree: number) {
        this.knots = knots;
        this.coefficients = coefficients;
        this.degree = degree;
        this.bspline = new BSpline(knots, degree);
    }

    evaluate(x: number): number {
        const basis = this.bspline.evaluate(x);
        return basis.reduce((sum, b, i) => sum + b * this.coefficients[i], 0);
    }

    evaluateDerivative(x: number): number {
        const basis = this.bspline.evaluateDerivative(x);
        return basis.reduce((sum, b, i) => sum + b * this.coefficients[i], 0);
    }

    evaluateSecondDerivative(x: number): number {
        const basis = this.bspline.evaluateSecondDerivative(x);
        return basis.reduce((sum, b, i) => sum + b * this.coefficients[i], 0);
    }

    getKnots(): number[] {
        return [...this.knots];
    }

    getCoefficients(): number[] {
        return [...this.coefficients];
    }

    getDegree(): number {
        return this.degree;
    }

    static fromBSpline(bspline: BSpline, coefficients: number[]): PiecewisePolynomial {
        return new PiecewisePolynomial(bspline.getKnots(), coefficients, bspline.getDegree());
    }
}
