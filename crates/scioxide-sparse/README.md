# scioxide-sparse

Sparse matrices and sparse linear algebra for Rust.
The crate stores matrices in COO, CSR, CSC, BSR and DIA formats, solves sparse linear systems and least-squares problems with LU, Cholesky, `LDLᴴ` and QR factorizations, and builds preconditioners for Krylov methods: Jacobi, incomplete LU and Cholesky, algebraic multigrid and overlapping Schwarz.
It also computes fill-reducing orderings, Jacobian colorings and graph algorithms on sparsity patterns, for real and complex `f32` and `f64`.

## Main entry points

One-shot forms take the problem data and return the answer; the types after them configure and reuse the same computation.

- Construction and storage: `CsrMatrix::from_triplets`, `CsrMatrix::identity`; `SparseBuilder`, `CooMatrix`, `CsrMatrix`, `CscMatrix`, `BsrMatrix`, `DiaMatrix`, `SparseView`.
- Arithmetic and products: `CsrMatrix::apply`, `CsrMatrix::multiply`, `CsrMatrix::add`, `sparse_block_grid`, `sparse_diagonals`; `SparseProductPlan`, `sparse_dense_into`.
- Direct solvers: `solve`, `least_squares`, `SparseLu::new`, `SparseCholesky::new`, `SparseLdl::new`; `SparseLu`, `SparseCholesky`, `SparseLdl`, `SparseQr`, `SparseQrMinimumNormPlan`, `SparseTriangular`.
- Preconditioners: `JacobiPreconditioner::new`, `IncompleteFactor::ilu0`, `IncompleteFactor::ic0`, `AmgPreconditioner::new`; `IncompleteFactor` (ILU(k), ILUT, IC(0), ICT), `AmgHierarchy`, `SchwarzPlan`.
- Orderings and colorings: `approximate_minimum_degree`, `reverse_cuthill_mckee`, `distance_one_coloring`, `derivative_coloring`.
- Graphs: `connected_components`, `shortest_paths`, `maximum_bipartite_matching`, `maximum_flow`, `minimum_cost_assignment`, `laplacian`; `GraphPlan` (traversal, components, spanning forests, shortest paths), `MaxFlowPlan`, `AssignmentPlan`, `LaplacianPlan`.

## Feature flags

- `std` (default): threads and clocks from `scioxide-core`.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-sparse) for a quick start, a guide to choosing a method, and the error and accuracy rules.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
