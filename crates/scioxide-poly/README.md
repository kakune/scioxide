# scioxide-poly

Polynomials for Rust.
The crate does exact polynomial and rational-function algebra over a ring or field you supply, and floating-point polynomial work in an explicit basis: monomial, Newton, barycentric, Bernstein or a classical orthogonal family.
Each floating-point result reports a backward-error and conditioning estimate.
The crate also fits samples by least squares, computes approximate GCDs, resultants and discriminants, decomposes partial fractions, and builds the matrices whose eigenvalues are a polynomial's roots.

## Main entry points

- Exact algebra: `ExactPolynomial`, `ExactRationalFunction`, `exact_subresultants`, `exact_discriminant`, `ExactPolynomialProductTree`.
- Floating-point polynomials: `ApproxPolynomial` (`monomial`, `from_roots`, `newton`, `interpolate_newton`, `interpolate_barycentric`, `bernstein`, `orthogonal`, each with a `_with_options` form; `evaluate`, `evaluate_batch`, `integral`), `PolynomialApproximationPolicy`.
- Orthogonal bases: `OrthogonalFamily`, `OrthogonalBasisSpec`, `OrthogonalBasisPlan`, `JacobiRecurrence`.
- Fitting and design matrices: `ApproxPolynomial::fit_monomial`, `ApproxPolynomial::fit_orthogonal`, `PolynomialFitPlan`, `vandermonde`, `Vandermonde`.
- Repeated operations: `PolynomialOperationPlan`, `exact_multiply_into`, `exact_compose_into`, `exact_taylor_shift_into`.
- Common factors and elimination: `approximate_gcd`, `ApproxRationalFunction`, `approximate_resultant`, `approximate_discriminant`, `SylvesterElimination`, `SylvesterDiscriminant`.
- Partial fractions: `ApproxPartialFractionPlan`, `ExactPartialFractionPlan`.
- Roots: `polynomial_root_matrix_into`, `polynomial_root_diagnostics`, `polynomial_root_radii`.
  The roots themselves come from `scioxide-rootfind`.
- Geometry and several variables: `TensorBernstein`, `ApproxSparsePolynomial`, `ExactSparsePolynomial`.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-poly) for a quick start, a guide to choosing a method, and the rules for diagnostics and errors.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
