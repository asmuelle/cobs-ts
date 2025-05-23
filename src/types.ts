export type ConstraintType = 'none' | 'increase' | 'decrease' | 'convex' | 'concave' | 'periodic';

export interface MonotonicityConstraint {
    type: 'monotone';
    increasing: boolean;
}

export interface ConvexityConstraint {
    type: 'convex';
    convex: boolean;
}

export interface PeriodicityConstraint {
    type: 'periodic';
}

export interface PointwiseConstraint {
    type: 'pointwise';
    x: number;
    y: number;
    operator: '=' | '<=' | '>=';
}

export interface Constraint {
    type: 'monotone' | 'convex' | 'concave' | 'periodic' | 'pointwise';
    increasing?: boolean;
    convex?: boolean;
    x?: number;
    y?: number;
    operator?: string;
}

export interface CobsOptions {
    degree?: number;
    numKnots?: number;
    lambda?: number;
    weights?: number[];
    maxiter?: number;
    ic?: 'aic' | 'bic' | 'sic';
    order?: number;
    knots?: number[];
    constraints?: Constraint[];
    tolerance?: number;
    tau?: number;
}

export interface PiecewisePolynomial {
    evaluate(x: number): number;
    evaluateDerivative(x: number): number;
    evaluateSecondDerivative(x: number): number;
    getKnots(): number[];
    getCoefficients(): number[];
    getDegree(): number;
}

export interface CobsFit {
    pp: PiecewisePolynomial;
    coefficients: number[];
    fitted: number[];
    residuals: number[];
    lambda: number;
    sic: number;
    icScore: number;
    fit: boolean;
}

export interface CobsResult {
    coefficients: number[];
    knots: number[];
    order: number;
    error: number;
    fit: {
        pp: {
            evaluate: (x: number) => number;
            evaluateSecondDerivative: (x: number) => number;
        };
        coefficients: number[];
        fitted: number[];
        residuals: number[];
    };
    pp: {
        evaluate: (x: number) => number;
        evaluateSecondDerivative: (x: number) => number;
    };
    evaluate: (x: number) => number;
    tau?: number;
    lambda?: number;
    sic?: number;
}
