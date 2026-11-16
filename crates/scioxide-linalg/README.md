# scioxide-linalg

Dense and structured linear algebra for Rust.
The crate solves linear systems and least-squares problems with direct factorizations and iterative methods, computes eigenvalues, Schur forms and singular values, evaluates matrix functions and their actions on vectors, and solves Sylvester, Lyapunov and Riccati equations, for real and complex `f32` and `f64`.

`backend::lu::LuFactor` prepares dense partial-pivoted LU for repeated component solves without nested accounting, source retention or condition diagnostics.
Its caller owns solve admission and publication; capacity preparation separates allocation from factorization and substitution.

## Main entry points

- One-shot functions: `solve`, `least_squares`, `solve_triangular`, `solve_structured`, `matrix_function`, `solve_sylvester`, `solve_lyapunov`, `solve_riccati`.
- Matrices: `Matrix`, `Vector`, `MatrixView`, `MatrixViewMut`, `VectorView`.
- Direct solvers: `LuDecomposition`, `CholeskyDecomposition`, `LdlDecomposition`, `HouseholderDecomposition`, `HouseholderMinimumNormPlan`, `QrUpdateDecomposition`, `DirectSolveOptions`, `RankPolicy`.
- Singular values: `SvdValues`, `ThinSvdDecomposition`, `SvdDecomposition`, `GsvdDecomposition`, `CsdDecomposition`, `PolarDecomposition`.
- Structured systems: `BandedLuDecomposition`, `BandedCholeskyDecomposition`, `TridiagonalLuDecomposition`, `TriangularPlan`, `StructuredSolvePlan`.
- Eigenproblems: `HermitianEigenDecomposition`, `GeneralEigenDecomposition`, `SchurDecomposition`, `GeneralizedHermitianEigenDecomposition`, `GeneralizedSchurDecomposition`.
- Iterative solvers: `CgSolver`, `MinresSolver`, `GmresSolver`, `BicgstabSolver`, `LsqrSolver`, `LsmrSolver`, `IterativeRefinement`.
- Matrix functions and equations: `MatrixFunctionPlan`, `SylvesterPlan`, `LyapunovPlan`, `RiccatiPlan`, `SparseFunctionPlan`.
- Large operators: `ExponentialAction`, `PartialSpectrumSolver`, `LobpcgSolver`, `PartialSvdSolver`, `RandomizedLowRank`, `OneNormEstimator`.
- Subspaces: `orth`, `null_space`, `subspace_angles`, `orthogonal_procrustes`.

## Feature flags

- `std` (default): threads, clocks and runtime CPU feature detection.
  Without it the crate is `no_std` and needs `alloc`.
- `blas`: lets you inject an external BLAS implementation through `BlasProvider`.
  No BLAS library is linked.

See the [API documentation](https://docs.rs/scioxide-linalg) for a quick start, a guide to choosing a method, and the error and accuracy rules.
Third-party attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
