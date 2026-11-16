# scioxide-interp

Interpolation and approximation of functions from samples.
The crate builds a function from values at given points and evaluates it, its derivatives and its integrals.
It covers polynomial and rational interpolation, piecewise cubics, B-splines and NURBS, smoothing splines, rectilinear grids, scattered data in the plane and in higher dimensions, local regression and kriging, and splines on the sphere and along curves.
Values can be real or complex scalars, arrays or vectors, and every query states how to treat points outside the data.

## Main entry points

- Shared traits and options: `Interpolant`, `Univariate`, `Differentiable`, `TwiceDifferentiable`, `Integrable`, `ExtrapolationPolicy`, `SampleOptions`, `NodeSeparation`.
- Piecewise cubics: `CubicSpline`, `PchipInterpolator`, `AkimaInterpolator`.
- Polynomials: `BarycentricInterpolator`, `NewtonInterpolator`, `NevilleInterpolator`, `HermiteInterpolator`.
- Rational functions: `aaa`, `BarycentricRational`, `FloaterHormannInterpolator`, `BulirschStoerInterpolator`, `PadeApproximant`.
- B-splines and fits: `BSpline`, `BSpline::interpolate`, `BSplineBasis`, `Nurbs`, `cardinal_bspline`, `TensorSplineInterpolationPlan`, `PenalizedSplinePlan`.
- Grids: `BilinearInterpolator`, `BicubicInterpolator`.
- Scattered data: `interpolate_scattered`, `RbfInterpolator`, `NearestNeighborInterpolator`, `SimplexLinearInterpolator`, `CloughTocherInterpolator`, `NaturalNeighborInterpolator`.
- Local models and kriging: `LocalModelPlan`, `LocalRbfModel`, `LocalPolynomialModel`, `GaussianProcessModel`.
- Sphere and curves: `SphericalSplinePlan`, `fit_spherical_spline`, `fit_parametric_spline`.
- Many points at once: `Interpolant::evaluate_many`, `BatchEvaluator`.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-interp) for a quick start, a guide to choosing a method, and the rules for extrapolation, errors and accuracy.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
