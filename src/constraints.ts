import { Matrix } from './matrix';
import { BSpline } from './bspline';
import { PointwiseConstraint, ConstraintType, Constraint } from './types';

export class Constraints {
    static generateConstraints(bspline: BSpline, constraints: Constraint[], x: number[]): { A: Matrix; b: number[] } {
        const allConstraints: { A: Matrix; b: number[] }[] = [];

        for (const constraint of constraints) {
            switch (constraint.type) {
                case 'monotone':
                    allConstraints.push(this.generateMonotoneConstraints(bspline, constraint.increasing||true, x));
                    break;
                case 'convex':
                    allConstraints.push(this.generateConvexityConstraints(bspline, x));
                    break;
                case 'periodic':
                    allConstraints.push(this.generatePeriodicityConstraints(bspline, x));
                    break;
                case 'pointwise':
                    allConstraints.push(this.generatePointwiseConstraints(bspline, constraint as PointwiseConstraint, x));
                    break;
            }
        }

        return this.combineConstraints(allConstraints);
    }

    private static generateMonotoneConstraints(bspline: BSpline, increasing: boolean, x: number[]): { A: Matrix; b: number[] } {
        const n = bspline.numCoefficients;
        const numPoints = 100;
        const xmin = Math.min(...x);
        const xmax = Math.max(...x);
        const h = (xmax - xmin) / (numPoints + 1);

        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        // Add constraints at multiple points
        for (let i = 0; i < numPoints; i++) {
            const x = xmin + (i + 1) * h;
            const basis = bspline.evaluateDerivative(x);

            for (let j = 0; j < n; j++) {
                if (Math.abs(basis[j]) > 1e-10) {
                    data.push(increasing ? basis[j] : -basis[j]);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
            b.push(0);
        }

        return {
            A: new Matrix({ rows: numPoints, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private static generateConvexityConstraints(bspline: BSpline, x: number[]): { A: Matrix; b: number[] } {
        const n = bspline.numCoefficients;
        const numPoints = 100;
        const xmin = Math.min(...x);
        const xmax = Math.max(...x);
        const h = (xmax - xmin) / (numPoints + 1);

        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        // Add constraints at multiple points
        for (let i = 0; i < numPoints; i++) {
            const x = xmin + (i + 1) * h;
            const basis = bspline.evaluateSecondDerivative(x);

            for (let j = 0; j < n; j++) {
                if (Math.abs(basis[j]) > 1e-10) {
                    data.push(basis[j]);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
            b.push(0);
        }

        return {
            A: new Matrix({ rows: numPoints, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private static generatePeriodicityConstraints(bspline: BSpline, x: number[]): { A: Matrix; b: number[] } {
        const n = bspline.numCoefficients;
        const xmin = Math.min(...x);
        const xmax = Math.max(...x);

        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        // Value constraint
        const basisStart = bspline.evaluate(xmin);
        const basisEnd = bspline.evaluate(xmax);

        for (let j = 0; j < n; j++) {
            if (Math.abs(basisStart[j]) > 1e-10) {
                data.push(basisStart[j]);
                rowIndices.push(0);
                colIndices.push(j);
            }
            if (Math.abs(basisEnd[j]) > 1e-10) {
                data.push(-basisEnd[j]);
                rowIndices.push(0);
                colIndices.push(j);
            }
        }
        b.push(0);

        // Derivative constraint
        const derivStart = bspline.evaluateDerivative(xmin);
        const derivEnd = bspline.evaluateDerivative(xmax);

        for (let j = 0; j < n; j++) {
            if (Math.abs(derivStart[j]) > 1e-10) {
                data.push(derivStart[j]);
                rowIndices.push(1);
                colIndices.push(j);
            }
            if (Math.abs(derivEnd[j]) > 1e-10) {
                data.push(-derivEnd[j]);
                rowIndices.push(1);
                colIndices.push(j);
            }
        }
        b.push(0);

        return {
            A: new Matrix({ rows: 2, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private static generatePointwiseConstraints(bspline: BSpline, constraint: PointwiseConstraint, x: number[]): { A: Matrix; b: number[] } {
        const n = bspline.numCoefficients;
        const basis = bspline.evaluate(constraint.x);

        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        // Add single equality constraint
        for (let j = 0; j < n; j++) {
            if (Math.abs(basis[j]) > 1e-10) {
                data.push(basis[j]);
                rowIndices.push(0);
                colIndices.push(j);
            }
        }
        b.push(constraint.y);

        return {
            A: new Matrix({ rows: 1, cols: n, data, rowIndices, colIndices }),
            b
        };
    }

    private static combineConstraints(constraints: { A: Matrix; b: number[] }[]): { A: Matrix; b: number[] } {
        if (constraints.length === 0) {
            return { A: new Matrix({ rows: 0, cols: 0, data: [], rowIndices: [], colIndices: [] }), b: [] };
        }

        const cols = constraints[0].A.cols;
        let totalRows = 0;
        for (const constraint of constraints) {
            totalRows += constraint.A.rows;
        }

        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];
        const b: number[] = [];

        let currentRow = 0;
        for (const constraint of constraints) {
            // Get the raw data from the matrix
            const matrixData = this.getMatrixData(constraint.A);
            for (let i = 0; i < matrixData.data.length; i++) {
                data.push(matrixData.data[i]);
                rowIndices.push(matrixData.rowIndices[i] + currentRow);
                colIndices.push(matrixData.colIndices[i]);
            }
            b.push(...constraint.b);
            currentRow += constraint.A.rows;
        }

        return {
            A: new Matrix({ rows: totalRows, cols, data, rowIndices, colIndices }),
            b
        };
    }

    private static getMatrixData(matrix: Matrix): { data: number[]; rowIndices: number[]; colIndices: number[] } {
        const data: number[] = [];
        const rowIndices: number[] = [];
        const colIndices: number[] = [];

        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                const value = matrix.get(i, j);
                if (Math.abs(value) > 1e-10) {
                    data.push(value);
                    rowIndices.push(i);
                    colIndices.push(j);
                }
            }
        }

        return { data, rowIndices, colIndices };
    }
}
