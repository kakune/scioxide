# scioxide-ode

Differential equations for Rust.
The crate integrates initial-value problems with explicit and implicit Runge–Kutta, Rosenbrock-W, Adams and BDF methods, with automatic stiffness switching, events and dense output.
It also solves index-1 differential-algebraic equations, delay equations with constant delays, Itô and Stratonovich stochastic equations, Hamiltonian systems with symplectic methods, and two-point boundary-value problems by collocation.
Forward sensitivities and adjoints give gradients with respect to parameters.

## Main entry points

- One-shot answers: `solve_ivp`, `solve_ivp_at`, `solve_dde`, `solve_sde`, `solve_hamiltonian`, `solve_bvp`.
- Initial-value problems: `OdeMethod`, `OdeOptions`, `OdeProblem`, `OdeState`, `OdeSystem`, `OdeFn`, `BuiltinOdeMethod`.
- Explicit methods: `DormandPrince5`, `DormandPrince853`, `Verner65`, `Rkf45`, `ClassicalRk4`, `Heun`, `ExplicitMidpoint`, `ForwardEuler`.
- Stiff methods: `RadauIia`, `Bdf`, `RosenbrockW`, `BackwardEuler`, `ImplicitTrapezoid`, `Sdirk`, `GaussRk`; multistep `AdamsBashforth`, `AdamsMoulton`; switching `AutoStiff`, `AdamsBdf`.
- Events and output: `EventFunction`, `OdeEventOptions`, `OdeOutputSchedule`, `OdeOutputSink`, `OdeTrajectory`.
- Other equations: `DaeBdf`, `DdePlan`, `SdeMethod`, `StormerVerlet`, `Rattle`, `BvpPlan`.
- Gradients and ensembles: `ForwardSensitivitySystem`, `OdeAdjointSolver`, `DaeAdjointSolver`, `solve_ensemble_into`.
- Results: `OdeSolveReport`, `OdeTermination`, and the report and termination types of each family.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-ode) for a quick start, a guide to choosing a method, and the rules for error control, events, termination and errors.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
The Hairer/Wanner DOP853 coefficient tables retain their original BSD-2-Clause terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
