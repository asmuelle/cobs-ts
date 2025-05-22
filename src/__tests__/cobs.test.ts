import { Cobs } from '../cobs';
import { CobsResult, Constraint } from '../types';

describe('Cobs', () => {
    describe('fit', () => {
        it('should fit a basic B-spline without constraints (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {}); // Uses default order 4 (degree 3)

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4); // Verify default order is used
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(5);
            expect(result.fit.residuals).toHaveLength(5);
        });

        it('should handle monotone increasing constraints (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 2, 4, 7, 11];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, { // Uses default order 4
                constraints: [{
                    type: 'monotone',
                    increasing: true
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(5);

            // Check monotonicity
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(xVal => result.fit.pp.evaluate(xVal));
            for (let i = 1; i < values.length; i++) {
                expect(Math.round(values[i] * 1000000) / 1000000).toBeGreaterThanOrEqual(Math.round(values[i - 1] * 1000000) / 1000000);
            }
        });

        it('should handle convexity constraints (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 2, 4, 7, 11];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, { // Uses default order 4
                constraints: [{
                    type: 'convex'
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(5);

            // Check convexity
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const secondDerivs = evalPoints.map(xVal => result.fit.pp.evaluateSecondDerivative(xVal));
            for (const d2 of secondDerivs) {
                expect(d2).toBeGreaterThanOrEqual(-1e-10);
            }
        });

        it('should handle periodic constraints (default order 4)', () => {
            const x = [0, 1, 2, 3, 4, 5, 6];
            const y = [0, 1, 0, -1, 0, 1, 0];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, { // Uses default order 4
                constraints: [{
                    type: 'periodic'
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(7);

            // Check periodicity
            const start = result.fit.pp.evaluate(0);
            const end = result.fit.pp.evaluate(6);
            expect(Math.abs(start - end)).toBeLessThan(1e-10);
        });

        it('should handle pointwise constraints (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, { // Uses default order 4
                constraints: [{
                    type: 'pointwise',
                    x: 3,
                    y: 9,
                    operator: '='
                }]
            });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(5);

            // Check pointwise constraint
            const value = result.fit.pp.evaluate(3);
            expect(Math.round(value * 1000000) / 1000000).toBe(Math.round(9 * 1000000) / 1000000);
        });

        it('should handle multiple constraints (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 2, 4, 7, 11];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, { // Uses default order 4
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
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(5);

            // Check all constraints
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(xVal => result.fit.pp.evaluate(xVal));
            const secondDerivs = evalPoints.map(xVal => result.fit.pp.evaluateSecondDerivative(xVal));

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

        it('should handle numerical stability for small tolerances (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 1.000001, 1.000002, 1.000003, 1.000004];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {}); // Uses default order 4

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(5);
            expect(result.fit.residuals).toHaveLength(5);

            // Check numerical stability
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(xVal => result.fit.pp.evaluate(xVal));
            for (let i = 1; i < values.length; i++) {
                expect(Math.abs(values[i] - values[i - 1])).toBeLessThan(1e-6);
            }
        });

        it('should handle edge cases with very small tolerances (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 1.0000001, 1.0000002, 1.0000003, 1.0000004];
            const cobs = new Cobs();
            const result = cobs.fit(x, y, {}); // Uses default order 4

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(5);
            expect(result.fit.residuals).toHaveLength(5);

            // Check numerical stability
            const evalPoints = [1.5, 2.5, 3.5, 4.5];
            const values = evalPoints.map(xVal => result.fit.pp.evaluate(xVal));
            for (let i = 1; i < values.length; i++) {
                expect(Math.abs(values[i] - values[i - 1])).toBeLessThan(1e-7);
            }
        });
        
        it('should handle conflicting constraints by finding a compromise (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 3, 5, 7, 9]; // Generally increasing data
            const cobs = new Cobs();
            const conflictingConstraints: Constraint[] = [
                { type: 'monotone', increasing: true },
                { type: 'pointwise', x: 3, y: 0, operator: '=' }
            ];
            
            let result: CobsResult | undefined;
            expect(() => {
                result = cobs.fit(x, y, { // Uses default order 4
                    constraints: conflictingConstraints
                });
            }).not.toThrow();

            expect(result).toBeDefined();
            if (result) expect(result.order).toBe(4); // Check default order if result is defined
            if (!result) return; // Guard for TypeScript

            expect(result.fit).toBeDefined();
            const pp = result.fit.pp;
            const valAtX3 = pp.evaluate(3);

            expect(valAtX3).toBeCloseTo(0, 5); 

            const valAt2_5 = pp.evaluate(2.5);
            expect(valAt2_5).toBeGreaterThan(valAtX3 - 1e-6); 
            const valAt1_5 = pp.evaluate(1.5);
            expect(valAt2_5).toBeGreaterThanOrEqual(valAt1_5 - 1e-6);

            const valAt3_5 = pp.evaluate(3.5);
            const valAt4_5 = pp.evaluate(4.5);
            expect(valAt3_5).toBeGreaterThanOrEqual(valAtX3 - 1e-6); 
            expect(valAt4_5).toBeGreaterThanOrEqual(valAt3_5 - 1e-6); 
        });

        it('should handle pointwise constraint with >= operator (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25]; 
            const cobs = new Cobs();
            const constraintValue = 8; 
            const result = cobs.fit(x, y, { // Uses default order 4
                constraints: [{ type: 'pointwise', x: 3, y: constraintValue, operator: '>=' }]
            });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            const valAtX3 = result.fit.pp.evaluate(3);
            expect(valAtX3).toBeGreaterThanOrEqual(constraintValue - 1e-6); 
        });

        it('should handle pointwise constraint with <= operator (default order 4)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25]; 
            const cobs = new Cobs();
            const constraintValue = 10; 
            const result = cobs.fit(x, y, { // Uses default order 4
                constraints: [{ type: 'pointwise', x: 3, y: constraintValue, operator: '<=' }]
            });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(4);
            const valAtX3 = result.fit.pp.evaluate(3);
            expect(valAtX3).toBeLessThanOrEqual(constraintValue + 1e-6); 
    });

        it('should fit a linear spline (order: 1)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [2, 4, 6, 8, 10]; 
            const cobs = new Cobs();
            const result: CobsResult = cobs.fit(x, y, { order: 1, numKnots: x.length });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(1); 
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(x.length);
            for (let i = 0; i < y.length; i++) {
                expect(result.fit.fitted[i]).toBeCloseTo(y[i], 1); 
            }
        });

        it('should fit a quadratic spline (order: 2)', () => {
            const x = [1, 2, 3, 4, 5];
            const y = [1, 4, 9, 16, 25]; 
            const cobs = new Cobs();
            const result: CobsResult = cobs.fit(x, y, { order: 2, numKnots: x.length });

            expect(result.fit).toBeDefined();
            expect(result.order).toBe(2);
            expect(result.fit.coefficients).toHaveLength(5);
            expect(result.fit.fitted).toHaveLength(x.length);
            for (let i = 0; i < y.length; i++) {
                expect(result.fit.fitted[i]).toBeCloseTo(y[i], 1);
            }
        });

        it('should fit a reasonable spline to noisy data (smoothing effect with specified knots)', () => {
            const numPoints = 25;
            const x = Array.from({ length: numPoints }, (_, i) => (i / (numPoints - 1)) * 10); 
            const y_ideal = x.map(xi => Math.sin(xi) + 0.5 * xi - 2); 

            const yRange = Math.max(...y_ideal) - Math.min(...y_ideal);
            const noise_amplitude = 0.25 * yRange; 
            
            const y_noisy = y_ideal.map(yi => yi + (Math.random() - 0.5) * noise_amplitude);

            const cobs = new Cobs();
            const fitOrder = 4; 
            const numBasisFunctionsTarget = 8;

            const totalKnotsNeeded = numBasisFunctionsTarget + fitOrder + 1; 
            const userKnots: number[] = [];
            const xMin = x[0];
            const xMax = x[x.length - 1];

            for (let i = 0; i <= fitOrder; i++) userKnots.push(xMin);
            const numDistinctInteriorKnots = totalKnotsNeeded - 2 * (fitOrder + 1); 
            if (numDistinctInteriorKnots > 0) {
                const interiorKnotStep = (xMax - xMin) / (numDistinctInteriorKnots + 1);
                for (let i = 1; i <= numDistinctInteriorKnots; i++) userKnots.push(xMin + i * interiorKnotStep);
            }
            for (let i = 0; i <= fitOrder; i++) userKnots.push(xMax);
            userKnots.sort((a,b) => a-b); 

            const result: CobsResult = cobs.fit(x, y_noisy, { order: fitOrder, knots: userKnots });

            expect(result.fit).toBeDefined();
            expect(result.knots).toEqual(userKnots); 
            expect(result.order).toBe(fitOrder);
            expect(result.fit.coefficients).toHaveLength(numBasisFunctionsTarget);
            expect(result.fit.fitted).toHaveLength(x.length); 

            let sse_fitted_ideal = 0;
            for (let i = 0; i < x.length; i++) {
                const val_at_xi = result.fit.pp.evaluate(x[i]);
                expect(val_at_xi).not.toBeNaN();
                sse_fitted_ideal += Math.pow(val_at_xi - y_ideal[i], 2);
            }

            let sse_noisy_ideal = 0;
            for (let i = 0; i < x.length; i++) {
                sse_noisy_ideal += Math.pow(y_noisy[i] - y_ideal[i], 2);
            }
            
            expect(sse_fitted_ideal).toBeLessThan(sse_noisy_ideal);
            expect(sse_fitted_ideal).toBeLessThan(sse_noisy_ideal * 0.5); 
        });

        // Moved describe blocks start here
        describe('Error Handling', () => {
            const x_valid = [1, 2, 3, 4, 5];
            const y_valid = [1, 4, 9, 16, 25];
            const cobs = new Cobs();
            // Note: defaultOptions.degree is a NO-OP as Cobs.ts uses default order 4 unless options.order is set.
            const defaultOptions = { degree: 3 }; 
    
            it('should throw an error for empty x and y arrays (less than 2 points)', () => {
                expect(() => {
                    cobs.fit([], [], defaultOptions);
                }).toThrowError('At least two data points are required');
            });
    
            it('should throw an error for single point x and y arrays (less than 2 points)', () => {
                expect(() => {
                    cobs.fit([1], [1], defaultOptions);
                }).toThrowError('At least two data points are required');
            });
    
            it('should throw an error for x and y arrays of different lengths', () => {
                expect(() => {
                    cobs.fit([1, 2], [1], defaultOptions);
                }).toThrowError('Input arrays x and y must have the same length');
            });
    
            it('should throw an error for invalid order option (e.g., 0)', () => {
                expect(() => {
                    cobs.fit(x_valid, y_valid, { ...defaultOptions, order: 0 });
                }).toThrowError();
            });
    
            it('should throw an error if an explicit insufficient knots array is provided (e.g., knots.length <= order)', () => {
                expect(() => {
                    const order = 4; 
                    cobs.fit(x_valid, y_valid, { order: order, knots: [1, 2, 3, 4] });
                }).toThrowError();
            });
    
            it('should throw an error for unsupported constraint type if strict handling is expected', () => {
                expect(() => {
                    cobs.fit(x_valid, y_valid, {
                        ...defaultOptions,
                        constraints: [{ type: 'unsupported_type_xyz' } as any]
                    });
                }).toThrowError();
            });
    
            it('should throw an error for invalid operator in pointwise constraint', () => {
                expect(() => {
                    cobs.fit(x_valid, y_valid, {
                        ...defaultOptions,
                        constraints: [{ type: 'pointwise', x: 3, y: 9, operator: '!=' as any }]
                    });
                }).toThrowError(/Unsupported operator: !=/);
            });
        });

        describe('Default Knot Generation Behavior (via fit)', () => {
            it('should generate default knots correctly when n > order + 1', () => {
                const x = [1, 2, 3, 4, 5, 6]; 
                const y = [1, 4, 9, 16, 25, 36];
                const cobs = new Cobs(); 
                const result = cobs.fit(x, y, {}); // Use default order 4
        
                const order = result.order; 
                expect(order).toBe(4);
        
                const expectedNumKnots = x.length + order + 1;
                expect(result.knots).toBeDefined();
                expect(result.knots).toHaveLength(expectedNumKnots); 
        
                for (let i = 0; i <= order; i++) {
                    expect(result.knots[i]).toBe(x[0]);
                }
        
                for (let i = 0; i <= order; i++) {
                    expect(result.knots[expectedNumKnots - 1 - i]).toBe(x[x.length - 1]);
                }
        
                const interiorKnots = result.knots.slice(order + 1, expectedNumKnots - (order + 1));
                expect(interiorKnots.length).toBe(expectedNumKnots - 2 * (order + 1)); 
                
                for (let i = 0; i < interiorKnots.length; i++) {
                    expect(interiorKnots[i]).toBeGreaterThanOrEqual(x[0]);
                    expect(interiorKnots[i]).toBeLessThanOrEqual(x[x.length - 1]);
                    if (i > 0) {
                        expect(interiorKnots[i]).toBeGreaterThanOrEqual(interiorKnots[i - 1]);
                    }
                }
            });
        
            it('should generate knots correctly with explicit order when n > order + 1', () => {
                const x = [10, 20, 30, 40, 50]; 
                const y = [1, 2, 3, 2, 1];
                const explicitOrder = 3; 
                const cobs = new Cobs();
                const result = cobs.fit(x, y, { order: explicitOrder });
        
                const order = result.order;
                expect(order).toBe(explicitOrder);
        
                const expectedNumKnots = x.length + order + 1;
                expect(result.knots).toBeDefined();
                expect(result.knots).toHaveLength(expectedNumKnots); 
        
                for (let i = 0; i <= order; i++) {
                    expect(result.knots[i]).toBe(x[0]);
                }
        
                for (let i = 0; i <= order; i++) {
                    expect(result.knots[expectedNumKnots - 1 - i]).toBe(x[x.length - 1]);
                }
        
                const numDistinctInteriorKnots = x.length - (order + 1); 
                const interiorKnots = result.knots.slice(order + 1, expectedNumKnots - (order + 1));
                expect(interiorKnots.length).toBe(numDistinctInteriorKnots);
                
                const expectedMinVal = x[0];
                const expectedMaxVal = x[x.length-1];
                for (let i = 0; i < interiorKnots.length; i++) {
                    expect(interiorKnots[i]).toBeGreaterThanOrEqual(expectedMinVal);
                    expect(interiorKnots[i]).toBeLessThanOrEqual(expectedMaxVal);
                    if (i > 0) {
                        expect(interiorKnots[i]).toBeGreaterThanOrEqual(interiorKnots[i - 1]);
                    }
                }
            });
        
            it('should generate knots correctly with few data points (n <= order + 1)', () => {
                const x = [5, 10, 15]; 
                const y = [2, 4, 6];
                const explicitOrder = 3; 
                const cobs = new Cobs();
                const result = cobs.fit(x, y, { order: explicitOrder });
        
                const order = result.order;
                expect(order).toBe(explicitOrder);
        
                const expectedNumKnots = 2 * (order + 1);
                expect(result.knots).toBeDefined();
                expect(result.knots).toHaveLength(expectedNumKnots); 
        
                for (let i = 0; i <= order; i++) {
                    expect(result.knots[i]).toBe(x[0]); 
                }
        
                for (let i = 0; i <= order; i++) {
                    expect(result.knots[expectedNumKnots - 1 - i]).toBe(x[x.length - 1]); 
                }
                
                const interiorKnots = result.knots.slice(order + 1, expectedNumKnots - (order + 1));
                expect(interiorKnots.length).toBe(0);
            });
        });

        describe('User-Provided Knots Behavior (via fit)', () => {
            it('should fit with valid user-provided knots', () => {
                const x = [1, 2, 3, 4, 5];
                const y = [1, 4, 9, 16, 25];
                const cobs = new Cobs();
                const userOrder = 3; 
                const userKnots = [1, 1, 1, 1, 3, 5, 5, 5, 5]; 
        
                const result: CobsResult = cobs.fit(x, y, { order: userOrder, knots: userKnots });
        
                expect(result.fit).toBeDefined();
                expect(result.knots).toEqual(userKnots);
                expect(result.order).toBe(userOrder);
        
                const expectedNumCoefficients = userKnots.length - userOrder - 1;
                expect(result.fit.coefficients).toHaveLength(expectedNumCoefficients);
                expect(result.fit.fitted).toHaveLength(x.length);
                result.fit.fitted.forEach(val => expect(val).not.toBeNaN());
        
                let sumSqResiduals = 0;
                for (let i = 0; i < y.length; i++) {
                    sumSqResiduals += Math.pow(result.fit.residuals[i], 2);
                }
                expect(sumSqResiduals).toBeLessThan(1e-5); 
            });
        
            it('should fit with non-uniformly spaced user-provided knots', () => {
                const x = [0, 1, 2, 3, 4, 5]; 
                const y = [0, 1, 0, -1, 0, 1];
                const cobs = new Cobs();
                const userOrder = 3; 
                const userKnots = [0, 0, 0, 0, 1.2, 2.7, 3.1, 5, 5, 5, 5]; 
        
                const result: CobsResult = cobs.fit(x, y, { order: userOrder, knots: userKnots });
        
                expect(result.fit).toBeDefined();
                expect(result.knots).toEqual(userKnots);
                expect(result.order).toBe(userOrder);
        
                const expectedNumCoefficients = userKnots.length - userOrder - 1; 
                expect(result.fit.coefficients).toHaveLength(expectedNumCoefficients);
                expect(result.fit.fitted).toHaveLength(x.length);
                result.fit.fitted.forEach(val => expect(val).not.toBeNaN());
            });
        
            it('should fit with user-provided coincident interior knots', () => {
                const x = [1, 2, 3, 4, 5]; 
                const y = [1, 2, 0, 4, 5]; 
                const cobs = new Cobs();
                const userOrder = 3; 
                const userKnots = [1, 1, 1, 1, 2.5, 2.5, 5, 5, 5, 5]; 
        
                const result: CobsResult = cobs.fit(x, y, { order: userOrder, knots: userKnots });
        
                expect(result.fit).toBeDefined();
                expect(result.knots).toEqual(userKnots);
                expect(result.order).toBe(userOrder);
        
                const expectedNumCoefficients = userKnots.length - userOrder - 1; 
                expect(result.fit.coefficients).toHaveLength(expectedNumCoefficients);
                expect(result.fit.fitted).toHaveLength(x.length);
                result.fit.fitted.forEach(val => expect(val).not.toBeNaN());
            });
        });

    }); // End of describe('fit')
}); // End of describe('Cobs')
