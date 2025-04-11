import { Cobs } from '../cobs';

describe('Cobs', () => {
    describe('fit', () => {
        it('should fit a basic B-spline without constraints', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(5);
            expect(result.fit.residuals).toHaveLength(5);
        });

        it('should handle monotone increasing constraints', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 2, 4, 7, 11];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5,
                constraints: [{
                    type: 'monotone',
                    increasing: true
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);

            // Check monotonicity
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(x => result.fit.pp.evaluate(x));
            for (let i = 1; i < values.length; i++) {
                expect(Math.round(values[i] * 1000000) / 1000000).toBeGreaterThanOrEqual(Math.round(values[i - 1] * 1000000) / 1000000);
            }
        });

        it('should handle convexity constraints', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 2, 4, 7, 11];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5,
                constraints: [{
                    type: 'convex'
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);

            // Check convexity
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const secondDerivs = evalPoints.map(x => result.fit.pp.evaluateSecondDerivative(x));
            for (const d2 of secondDerivs) {
                expect(d2).toBeGreaterThanOrEqual(-1e-10);
            }
        });

        it('should handle periodic constraints', () => {
            const x = [0, 1, 2, 3, 4, 5, 6];
            const y = [0, 1, 0, -1, 0, 1, 0];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 7,
                constraints: [{
                    type: 'periodic'
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(7);

            // Check periodicity
            const start = result.fit.pp.evaluate(0);
            const end = result.fit.pp.evaluate(6);
            expect(Math.abs(start - end)).toBeLessThan(1e-10);
        });

        it('should handle pointwise constraints', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5,
                constraints: [{
                    type: 'pointwise',
                    x: 3,
                    y: 9,
                    operator: '='
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);

            // Check pointwise constraint
            const value = result.fit.pp.evaluate(3);
            expect(Math.round(value * 1000000) / 1000000).toBe(Math.round(9 * 1000000) / 1000000);
        });

        it('should handle multiple constraints', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 2, 4, 7, 11];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5,
                constraints: [
                    {
                        type: 'monotone',
                        increasing: true
                    },
                    {
                        type: 'convex'
                    },
                    {
                        type: 'pointwise',
                        x: 3,
                        y: 4,
                        operator: '='
                    }
                ]
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);

            // Check all constraints
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(x => result.fit.pp.evaluate(x));
            const secondDerivs = evalPoints.map(x => result.fit.pp.evaluateSecondDerivative(x));

            // Monotonicity
            for (let i = 1; i < values.length; i++) {
                expect(Math.round(values[i] * 1000000) / 1000000).toBeGreaterThanOrEqual(Math.round(values[i - 1] * 1000000) / 1000000);
            }

            // Convexity
            for (const d2 of secondDerivs) {
                expect(d2).toBeGreaterThanOrEqual(-1e-10);
            }

            // Pointwise
            const value = result.fit.pp.evaluate(3);
            expect(Math.round(value * 1000000) / 1000000).toBe(Math.round(4 * 1000000) / 1000000);
        });

        it('should handle numerical stability for small tolerances', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 1.000001, 1.000002, 1.000003, 1.000004];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(5);
            expect(result.fit.residuals).toHaveLength(5);

            // Check numerical stability
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(x => result.fit.pp.evaluate(x));
            for (let i = 1; i < values.length; i++) {
                expect(Math.abs(values[i] - values[i - 1])).toBeLessThan(1e-6);
            }
        });

        it('should handle edge cases with very small tolerances', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 1.0000001, 1.0000002, 1.0000003, 1.0000004];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {
                degree: 3,
                numKnots: 5
            });

            expect(result.fit).toBeDefined();
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(5);
            expect(result.fit.residuals).toHaveLength(5);

            // Check numerical stability
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(x => result.fit.pp.evaluate(x));
            for (let i = 1; i < values.length; i++) {
                expect(Math.abs(values[i] - values[i - 1])).toBeLessThan(1e-7);
            }
        });
    });
});
