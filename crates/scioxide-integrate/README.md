# scioxide-integrate

Numerical integration for Rust.
The crate computes definite integrals of real, complex and vector-valued functions in one or more dimensions: adaptive and fixed quadrature on finite and infinite intervals, integrals with singular or oscillatory weights, integrals of sampled data, cubature on boxes, simplices and the sphere, sparse grids and adaptive Monte Carlo.
It also sums infinite series with convergence acceleration and computes derivatives of integrals with respect to parameters, for `f32` and `f64`.

## Main entry points

- One call: `integrate`, `integrate_piecewise`, `integrate_fourier`, `integrate_unbounded`, `integrate_box`, `integrate_iterated`, `integrate_monte_carlo`, `trapezoid`, `simpson`, `cumulative_trapezoid`, `sum_series`.
- Adaptive quadrature: `AdaptiveIntegrator`, `Romberg`, `DoubleExponentialIntegrator`, `SubdivisionOptions`, `LevelOptions`.
- Fixed rules: `GaussianRulePlan::gauss_legendre`, `GaussianRulePlan::new`, `FixedRule`, `GaussKronrod`, `ClenshawCurtis`, `NewtonCotes`.
- Weights and oscillation: `WeightedQuadraturePlan`, `FourierTailPlan`, `OscillatoryQuadraturePlan`, `ContourPlan`.
- Sampled data: `SampledTrapezoidPlan`, `SampledSimpsonPlan`, `SampledRombergPlan`.
- Several dimensions: `CubaturePlan`, `ProductRule`, `IteratedDomainPlan`, `SparseGridPlan`, `SimplexRule`, `LebedevRule`, `SphereProductRule`.
- Monte Carlo: `MonteCarloIntegrator`, `VegasTrainer`, `MiserTrainer`.
- Series and derivatives: `SeriesSummation`, `QuadratureSensitivityPlan`.
- Results: `Integral`, `IntegrationReport`, `IntegrationTermination`.

One-shot integrals reject unsuccessful termination.
Configured forms retain reports and partial estimates; sampled workspaces come from their plans.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-integrate) for a quick start, a guide to choosing a method, and the rules for errors, termination and error estimates.
Third-party attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
