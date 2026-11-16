# scioxide-differentiate

Numerical derivatives of functions that you can only evaluate.
The crate computes first and second derivatives, gradients, Jacobians, Hessians and Jacobian-vector products by fixed-step forward and central differences, by adaptive differences with Ridders' or Richardson extrapolation and an error estimate, by finite-difference weights on nodes you choose, and by the complex step.
Sparse Jacobians with a known pattern are recovered from a column coloring.
Solvers use `DerivativeStrategy` to choose between an analytic derivative, automatic differentiation and finite differences.

## Main entry points

- One-shot functions: `derivative`, `second_derivative`, `gradient`, `jacobian`, `hessian`, `jvp`, `sparse_jacobian`, `physical_difference_derivative`, `complex_step_derivative`.
- Method values: `FiniteDifference` (fixed steps, `DifferenceScheme`, `DifferenceStep`), `AdaptiveDifference` (error estimates, `AdaptiveDifferenceMethod`) and `ComplexStep`; their workspaces are sized by `DerivativeRequest` and `DerivativeShape`.
- Node-offset method values: `RelativeDifference`, `RelativeComplexStep`.
- Plans: `ColoredJacobianPlan`, `PhysicalDifferencePlan`, `ComplexStepPlan`.
- Finite-difference weights: `FiniteDifferenceStencil`.
- Results: `DifferentiationResult`, `DifferentiationReport`, `DifferentiationTermination`.
- Callbacks: closures, `Contextual`, `Parallel`.
- Solvers: `DerivativeStrategy`, `DerivativeCandidate`.

## Feature flags

- `std` (default): enables `std` in `scioxide-core`.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-differentiate) for a quick start, a guide to choosing a method, and the step, accuracy and error rules.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
