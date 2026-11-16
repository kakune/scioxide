# scioxide-rootfind

Root finding for Rust.
The crate solves scalar equations `f(x) = 0` with bracketing and open methods, in real or complex arithmetic, and square nonlinear systems `F(x) = 0` with Newton, quasi-Newton, trust-region, fixed-point and Jacobian-free Newton–Krylov methods.
It follows solution curves by continuation, finds further roots by deflation, computes all roots of a polynomial, and returns indexed zeros of Bessel, Airy, Kelvin, error and Fresnel functions.

## Main entry points

- One-shot forms with default settings: `find_root`, `find_bracket`, `solve_system`, `fixed_point`, `polynomial_roots`, `special_zeros`.
- Scalar equations, bracketing: `Brent`, `Toms748`, `Chandrupatla`, `Brenth`, `Ridder`, `Illinois`, `RegulaFalsi`, `Bisection`, `find_bracket_with_options`.
- Scalar equations, open: `Newton`, `Halley`, `Secant`, `Muller`, `SafeguardedNewton`, `SafeguardedHalley`, `ComplexNewton`, `ComplexHalley`, `ComplexSecant`, `ComplexMuller`, `ScalarRootMethod`.
- Nonlinear systems: `NewtonSystem`, `Broyden`, `TrustRegionSystem`, `FixedPointSolver`, `NewtonKrylovSolver`.
- Solution curves and further roots: `Continuation`, `LinearHomotopy`, `DeflationSet`, `DeflatedProblem`.
- Polynomials and special functions: `polynomial_roots_with_options`, `polynomial_roots_into`, `special_zeros_with_options`, `special_zeros_into`.
- Problems and results: `ScalarFn`, `ResidualFn`, `RootResult`, `SystemRootResult`, `RootTermination`.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-rootfind) for a quick start, a guide to choosing a method, and the rules for termination and errors.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
