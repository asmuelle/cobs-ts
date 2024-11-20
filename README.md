# COBS-TS

A TypeScript implementation of Constrained B-Splines (COBS), based on the R package [cobs](https://cran.r-project.org/web/packages/cobs/).

## Description

COBS-TS provides qualitatively constrained (regression) smoothing splines via linear programming and sparse matrices. It supports various constraints including:

- Monotonicity (increasing/decreasing)
- Convexity/Concavity
- Periodicity
- Pointwise constraints

## Installation

```bash
npm install cobs-ts
```

## Usage

```typescript
import { Cobs } from 'cobs-ts';

// Example: Fit a monotone increasing spline
const x = [1, 2, 3, 4, 5];
const y = [1.1, 2.3, 2.9, 3.8, 4.2];

const result = Cobs.fit(x, y, {
    constraint: 'increase',
    degree: 2,
    tau: 0.5
});

console.log(result.fit.fitted);  // Fitted values
console.log(result.knots);       // Knot locations
console.log(result.coef);        // Spline coefficients
```

## Features

- B-spline basis functions of degree 1 or 2
- Multiple constraint types
- Automatic knot selection
- Information criterion (AIC, BIC, SIC) for model selection
- Sparse matrix implementation for efficient computation
- Support for weighted fitting
- Quantile regression capability

## API

### Cobs.fit(x: number[], y: number[], options?: CobsOptions): CobsResult

Main function for fitting constrained B-splines.

#### Options

- `constraint`: 'none' | 'increase' | 'decrease' | 'convex' | 'concave' | 'periodic'
- `weights`: number[] - Observation weights
- `knots`: number | number[] - Number of knots or explicit knot locations
- `degree`: 1 | 2 - Degree of B-spline basis
- `tau`: number - Quantile level (default: 0.5 for median regression)
- `lambda`: number | null - Smoothing parameter
- `ic`: 'AIC' | 'BIC' | 'SIC' - Information criterion for model selection
- `pointwise`: PointwiseConstraint[] - Additional pointwise constraints
- `maxiter`: number - Maximum number of iterations
- `nknots`: number - Number of knots if not explicitly specified
- `keep`: boolean - Whether to keep additional fitting information

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Run linter
npm run lint
```

## License

GPL-2.0-or-later (matching the original R package license)

## References

- Original R package by Pin T. Ng and Martin Maechler
- He, X. and Ng, P. (1999): "COBS: Qualitatively Constrained Smoothing via Linear Programming"; Computational Statistics, 14, 315-337.
