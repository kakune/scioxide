# scioxide-optimize

Numerical optimization for Rust.
The crate minimizes functions of real variables with or without derivatives, subject to bounds and nonlinear constraints.
It fits nonlinear and linear least-squares models, solves linear, quadratic and mixed-integer linear programs, checks claimed LP and QP solutions, and approximates quadratic assignment problems.
Global, multiobjective, proximal and stochastic methods cover problems that local smooth methods cannot, for `f32` and `f64`.

## Main entry points

- One call: `minimize`, `minimize_bounded`, `minimize_constrained`, `minimize_scalar`, `minimize_global`, `least_squares`, `curve_fit`, `nnls`, `bvls`, `quadratic_assignment`, `LpProblem::solve`, `QpProblem::solve`, `MipProblem::solve`.
- Problems: `ObjectiveFunction`, `ObjectiveFn`, `BoundedObjective`, `ConstrainedProblem`, `Constraints`, `LeastSquaresProblem`, `LpProblem`, `QpProblem`, `MipProblem`.
- Local minimization: `Bfgs`, `Lbfgs`, `LbfgsB`, `NonlinearCg`, `GradientDescent`, `NewtonOptimizer`, `TrustRegionOptimizer`, `NelderMead`, `Powell`, `ScalarMinimizer`.
- Nonlinear constraints: `Sqp`, `InteriorPoint`, `AugmentedLagrangian`, `PenaltyMethod`, `Cobyla`, `Cobyqa`.
- Least squares: `LeastSquares`, `Nnls`, `BoundedLinearLeastSquares`.
- Linear, quadratic and integer programming: `SimplexSolver`, `InteriorPointLp`, `QpSolver`, `BranchAndBound`, `ProgrammingVerifier`, `QapPlan`.
- Global search: `MultiStart`, `BasinHopping`, `DualAnnealing`, `DifferentialEvolution`, `CmaEs`, `Direct`, `Shgo`, `SurrogateSearch`.
- Composite, multiobjective and stochastic: `ProximalGradient`, `Fista`, `Admm`, `Nsga2`, `StochasticGradientDescent`, `StochasticTrustRegion`.
- Results: `OptimizeResult`, `OptimizeTermination`, `ProgrammingResult`, `MipResult`.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-optimize) for a quick start, a guide to choosing a method, and the rules for termination and errors.
Third-party attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
