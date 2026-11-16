# scioxide-validated

Verified numerics for Rust.
Every result is an enclosure that provably contains the exact value, with all rounding errors accounted for.
The crate provides interval arithmetic with decorations, enclosures of the elementary functions, and real and complex affine arithmetic.
It proves that a square linear system is nonsingular and bounds the error of its solution, proves the existence, uniqueness or absence of roots of scalar functions, nonlinear systems and polynomials, and encloses definite integrals.
Approximate results from other SciOxide crates can be checked against the original problem.

## Main entry points

- One-shot proofs: `solve`, `roots`, `polynomial_roots`, `integrate`.
  Each returns a verified or complete result; an inconclusive proof is an error.
- Interval arithmetic: `DecoratedInterval`, `Decoration`, `ElementaryEnclosure`, `EnclosureStatus`.
- Affine arithmetic: `AffineContext`, `AffineForm`, `AffineOptions`, `AffineUnary`, `AffineBinary`.
- Interval extensions you provide: `IntervalScalarFunction`, `IntervalLinearOperator`, `IntervalResidualOperator`, `IntervalJacobianOperator`, `QuadratureOracle`.
- Linear systems: `LinearVerifier`, `DenseIntervalOperator`, `SparseIntervalOperator`, `LuIntervalInverse`, `LinearCertificate`.
- Roots: `ScalarRootVerifier`, `SystemRootVerifier`, `PolynomialRootPlan`, `RootClaim`, `RootIsolation`, `PolynomialRootCover`.
- Integrals: `QuadratureVerifier`, `QuadraturePiece`, `QuadratureAssumption`, `QuadratureCertificate`.
- Options and workspaces: `ValidatedOptions`, `ProofRequest`, `LinearWorkspace`, `RootWorkspace`, `QuadratureWorkspace`.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

The crate requires round-to-nearest-even with gradual underflow on an x86-64, x86 with SSE2, or AArch64 target.
See the [API documentation](https://docs.rs/scioxide-validated) for a quick start, a guide to choosing a method, and the rules for errors and accuracy.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
