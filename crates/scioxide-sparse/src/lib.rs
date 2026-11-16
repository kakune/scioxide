//! Sparse matrices, sparse direct and incomplete factorizations,
//! preconditioners, orderings and graph algorithms.
//!
//! The crate stores sparse matrices in coordinate (COO), compressed row (CSR),
//! compressed column (CSC), block row (BSR) and diagonal (DIA) formats, and
//! computes sparse–sparse and sparse–dense products. It solves sparse linear
//! systems and least-squares problems with LU, Cholesky, `LDLᴴ` and QR
//! factorizations, and it builds preconditioners for Krylov methods: Jacobi,
//! incomplete LU and Cholesky, smoothed-aggregation algebraic multigrid and
//! overlapping Schwarz. It also computes fill-reducing and bandwidth-reducing
//! orderings, graph colorings for sparse Jacobians, and classical graph
//! algorithms on the sparsity pattern. Real and complex `f32` and `f64` scalars
//! are supported.
//!
//! # Quick start
//!
//! ```
//! use scioxide_sparse::{CsrMatrix, SparseLu};
//!
//! // A = tridiag(−1, 2, −1) of order 3 from (row, column, value) triplets.
//! let (rows, cols) = ([0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]);
//! let a = CsrMatrix::from_triplets(3, 3, &rows, &cols, &[2.0_f64, -1., -1., 2., -1., -1., 2.])?;
//! // For b = (1, 0, 1), x = (1, 1, 1).
//! let x = scioxide_sparse::solve(&a, &[1.0, 0.0, 1.0])?;
//! assert!(x.iter().all(|xi| (xi - 1.0).abs() < 1e-14));
//! // Factor once to solve for several right-hand sides.
//! let lu = SparseLu::new(&a)?;
//! let (y, z) = (
//!     lu.solve(&[1.0, 0.0, 0.0])?,
//!     lu.solve_transpose(&[0.0, 0.0, 1.0])?,
//! );
//! assert!((y[0] - 0.75).abs() < 1e-14 && (z[2] - 0.75).abs() < 1e-14);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! Each row names the one-shot form first, then the type that configures or
//! reuses it.
//!
//! | Problem | Entry point | Method |
//! |---|---|---|
//! | Assemble a matrix from (row, column, value) triplets | [`CsrMatrix::from_triplets`], [`SparseBuilder`] | sort, then sum or reject duplicates |
//! | Store and apply a matrix | [`CsrMatrix::apply`]; [`CsrMatrix`], [`CscMatrix`], [`CooMatrix`], [`BsrMatrix`], [`DiaMatrix`] | `A x` and `Aᴴ x`; implements [`LinearOperator`](scioxide_core::LinearOperator) |
//! | Sparse–sparse product `A B` | [`CsrMatrix::multiply`]; [`SparseProductPlan`] | Gustavson's row-wise algorithm with a prepared pattern |
//! | Sums, differences, Hadamard products, block assembly | [`CsrMatrix::add`], [`CsrMatrix::subtract`], [`CsrMatrix::hadamard`], [`sparse_block_grid`], [`sparse_diagonals`] | entrywise on the union or intersection of the patterns |
//! | Sparse–dense product `op(A) X` | [`sparse_dense_into`] | any storage format, `A`, `Aᵀ` or `Aᴴ` |
//! | General square system | [`solve`]; [`SparseLu`] | left-looking LU with partial pivoting (Gilbert–Peierls) |
//! | Hermitian positive definite system | [`SparseCholesky::new`]; [`SparseCholesky`] | Cholesky with fill predicted by the elimination tree |
//! | Hermitian indefinite system, inertia | [`SparseLdl::new`]; [`SparseLdl`] | `LDLᴴ` with bounded rook pivoting and 1×1 or 2×2 pivots |
//! | Least squares, rank-deficient or underdetermined system | [`least_squares`]; [`SparseQr`], [`SparseQrMinimumNormPlan`] | Householder QR in a fixed column order; minimum-norm solve with a rank threshold |
//! | Triangular system | [`SparseTriangular`] | forward or back substitution |
//! | Preconditioner for a Krylov method | [`JacobiPreconditioner::new`], [`IncompleteFactor::ilu0`], [`IncompleteFactor::ic0`], [`AmgPreconditioner::new`]; [`IncompleteFactor`], [`AmgHierarchy`], [`SchwarzPlan`] | diagonal scaling; ILU(k), ILUT, IC(0), ICT; smoothed aggregation; additive, restricted additive or multiplicative Schwarz |
//! | Fill-reducing or bandwidth-reducing ordering | [`approximate_minimum_degree`], [`reverse_cuthill_mckee`] | approximate minimum degree; reverse Cuthill–McKee |
//! | Column groups for a sparse Jacobian | [`derivative_coloring`], [`distance_one_coloring`] | greedy coloring in ascending index order |
//! | Connected components, shortest paths, traversal, spanning forests | [`connected_components`], [`shortest_paths`]; [`GraphPlan`] | BFS, DFS, Kosaraju, Kruskal, Dijkstra, Bellman–Ford, Floyd–Warshall, Johnson |
//! | Maximum bipartite matching, maximum flow, assignment | [`maximum_bipartite_matching`], [`maximum_flow`], [`minimum_cost_assignment`]; [`MaxFlowPlan`], [`AssignmentPlan`] | Hopcroft–Karp; Dinic; shortest augmenting paths |
//! | Graph Laplacian | [`laplacian`]; [`LaplacianPlan`] | `D − A` or `D^(−1/2) (D − A) D^(−1/2)` as a sparse matrix |
//!
//! # Storage
//!
//! A [`CsrMatrix`] stores, for each row, the column indices of its entries in
//! strictly increasing order, so a row holds no duplicates; [`CscMatrix`] does the
//! same by columns, and [`CooMatrix`] keeps triplets sorted by row, then column.
//! Every stored value is finite. Explicitly stored zeros are kept as entries and
//! are part of the sparsity pattern. [`CsrMatrix::from_triplets`] accepts
//! triplets in any order, adds the values at a repeated coordinate exactly and
//! keeps every coordinate, cancelled sums included: the policies
//! `(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)`. [`SparseBuilder`]
//! takes the [`DuplicatePolicy`] and [`ExplicitZeroPolicy`] as arguments.
//! [`CsrMatrix::from_raw`] checks arrays that are already in this form.
//! [`BsrMatrix`] and [`DiaMatrix`] are created from their own arrays with
//! `from_raw` or converted from another format with `from_sparse`.
//!
//! The sparsity pattern of a matrix is fixed after construction. Its values can
//! change through `update_values`, which checks all new values before it writes
//! any of them.
//!
//! # Direct solvers
//!
//! A direct solve has three steps:
//!
//! 1. Symbolic analysis ([`SparseLuSymbolic`], [`SparseSymmetricSymbolic`],
//!    [`SparseQrSymbolic`]) reads only the sparsity pattern. It applies a
//!    [`SparseOrdering`], which defaults to approximate minimum degree, and
//!    checks the predicted storage against explicit limits such as
//!    [`SparseLuLimits`]. Their entry limits size the factor storage.
//! 2. Numerical factorization (`factor`) takes the analysis and the values. LU
//!    and `LDLᴴ` may permute rows further for stability, which can add fill; fill
//!    beyond the configured limit is a `ResourceLimit` error.
//! 3. Solves (`solve`, `solve_transpose`, `solve_adjoint`, `solve_into`,
//!    `solve_block_into`) apply the factor to one or more right-hand sides,
//!    with `A`, `Aᵀ` or `Aᴴ`.
//!
//! The one-shot forms [`solve`], [`least_squares`] and the `new` constructors
//! of [`SparseLu`], [`SparseCholesky`], [`SparseLdl`] and
//! [`SparseQrMinimumNormPlan`] run all three steps from the matrix alone, with a
//! fill-reducing ordering and no byte limit. They choose the entry limits by
//! the capacity rule: the storage starts from an estimate, doubles after each
//! `ResourceLimit` and stops at a bound that no factor of the matrix exceeds,
//! so a failed attempt only repeats work. Entry limits are read only as
//! capacities and in comparisons, so the result equals the configured
//! analysis and `factor` with any sufficient limits. Each `new` states its
//! estimate and bound.
//!
//! For a sequence of matrices with the same pattern, call `refactor` with a
//! workspace reserved once. It does not allocate, and if it fails the previous
//! factor stays usable.
//!
//! # Execution
//!
//! Operations have up to three levels of control:
//!
//! - One-shot functions and constructors take the problem data and return the
//!   answer, such as [`solve`], [`shortest_paths`] or [`AmgPreconditioner::new`].
//! - Owned forms of a prepared value, such as `factor`, `solve`,
//!   `solve_transpose`, `solve_adjoint`, `apply` and the graph plan methods,
//!   take no options, workspace or context. Each runs its configured sibling
//!   with the default [`SparseSolveOptions`] or [`GraphExecutionOptions`], a
//!   fresh workspace and
//!   [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native),
//!   and returns the same numbers.
//! - Configured forms (`factor_scoped`, `solve_into`, `solve_block_into`,
//!   `apply_into`, `*_scoped`) take the options, caller storage and a scope.
//!
//! The options select the [`Accuracy`](scioxide_core::Accuracy), the
//! [`ExecutionPolicy`](scioxide_core::ExecutionPolicy) (at most 64 workers) and the
//! iteration and evaluation limits; products use [`SparseProductOptions`]. The
//! scope is an [execution context](scioxide_core#execution-contexts) or a
//! [`SolveContext`](scioxide_core::SolveContext) of an enclosing solver, whose
//! limits the call also counts toward, as described in
//! [Solve accounting](scioxide_core#solve-accounting). Parallel solves split the
//! right-hand sides between workers; factorization itself is sequential.
//! Traversals, components and spanning forests are fixed-cost kernels and take
//! no scope.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: storage arrays are not in the form described under
//!   [Storage](#storage), a builder rejects a duplicate, an option is out of
//!   range, or the matrix passed to `refactor` has a different pattern from the
//!   analysis.
//! - `DimensionMismatch`: shapes do not agree, a matrix that must be square is
//!   not, or an output, scratch slice or workspace is too short or was reserved
//!   for another analysis.
//! - `OutOfBounds`: a triplet or vertex index lies outside the matrix or graph.
//! - `NonFinite`: an input value is NaN or infinite.
//! - `Singular`: LU meets a column without a nonzero pivot, a triangular or
//!   incomplete factor has a zero pivot, or a solve uses a singular `LDLᴴ` factor.
//! - `NumericalInstability`: a value overflows or a nonzero result underflows to
//!   zero, or Cholesky meets a pivot that is not positive.
//! - `ResourceLimit`: fill, active entries or storage exceed the configured
//!   limits.
//! - `ConvergenceFailure`: an operation exhausts the iteration limit in its
//!   options or scope.
//!
//! On error, outputs and existing factors are unchanged, as described in
//! [Output on error](scioxide_core#output-on-error).
//!
//! # Accuracy
//!
//! Products, substitutions and preconditioner applications use the accuracy
//! selected by [`Accuracy`](scioxide_core::Accuracy). `Standard` uses ordinary
//! floating-point arithmetic, rescaling to avoid overflow and underflow, and
//! satisfies the inner-product error bounds stated on
//! [`Accuracy`](scioxide_core::Accuracy). `Exact` accumulates each sum of
//! products exactly and rounds once. The error of a solve also depends on the
//! condition number of the matrix and on the growth of the factors.
//! Incomplete factors and multigrid hierarchies approximate the matrix on
//! purpose; they are meant as preconditioners, not as solvers.
//!
//! # Reusing plans and workspaces
//!
//! Analyses, factors and plans follow the
//! [shared rules](scioxide_core#plans-and-workspaces). Each method reports its
//! storage with `workspace_requirements`; matching, which has no plan, asks
//! [`MatchingWorkspace::workspace_requirements`]. A workspace reserved with
//! `try_reserve`, or a scratch slice wrapped by
//! [`SparseSolveWorkspace::borrowed`], makes repeated `*_into` and `*_scoped`
//! calls and `refactor` run without allocation. The byte limit given at analysis
//! covers the analysis, the factor and its workspace together. Each workspace
//! records the [`AttemptRecord`](scioxide_core::AttemptRecord) of its last call.
//!
//! # Feature flags
//!
//! - `std` (default): enables the standard-library capabilities of
//!   `scioxide-core`, such as threads and clocks. Without it the crate is `no_std`
//!   and needs only `alloc`; the API is the same.
//!
//! The [`backend`] module holds `LDLᴴ` pivot-block arithmetic shared with
//! `scioxide-linalg`. Most users do not need it.
//!
//! # References
//!
//! - T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
//! - Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
//! - J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time
//!   proportional to arithmetic operations", *SIAM J. Sci. Stat. Comput.* 9
//!   (1988), 862–874.
//! - P. R. Amestoy, T. A. Davis and I. S. Duff, "An approximate minimum degree
//!   ordering algorithm", *SIAM J. Matrix Anal. Appl.* 17 (1996), 886–905.
//! - C. Ashcraft, R. G. Grimes and J. G. Lewis, "Accurate symmetric indefinite
//!   linear equation solvers", *SIAM J. Matrix Anal. Appl.* 20 (1998), 513–561.
//! - P. Vaněk, J. Mandel and M. Brezina, "Algebraic multigrid by smoothed
//!   aggregation for second and fourth order elliptic problems", *Computing* 56
//!   (1996), 179–196.
//! - B. Smith, P. Bjørstad and W. Gropp, *Domain Decomposition*, Cambridge
//!   University Press, 1996.
//! - T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to
//!   Algorithms*, 3rd ed., MIT Press, 2009.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![allow(clippy::module_name_repetitions)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod build;
mod capacity;
mod compressed;
mod direct;
pub use direct::{least_squares, solve};
mod execution;
mod factor_arithmetic;
mod factor_solve;
pub use factor_solve::{SparseSolveOptions, SparseSolveRequirements, SparseSolveWorkspace};
mod helpers;
mod values;
pub use values::{SparseValueRequirements, SparseValueWorkspace};
mod lu;
mod preconditioner;
mod qr;
mod schwarz;
mod triangular;
pub use lu::{SparseLu, SparseLuLimits, SparseLuRequirements, SparseLuSymbolic, SparseLuWorkspace};
pub use qr::{
    SparseQr, SparseQrLimits, SparseQrMinimumNormPlan, SparseQrRequirements, SparseQrSolveReport,
    SparseQrSymbolic, SparseQrWorkspace,
};

pub use build::{CooMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
pub use compressed::{CscMatrix, CsrMatrix};
pub use preconditioner::{IdentityPreconditioner, JacobiPreconditioner};
pub use schwarz::{
    SchwarzKind, SchwarzLocalSolveView, SchwarzOperatorView, SchwarzOptions, SchwarzPlan,
    SchwarzSubdomainView,
};
pub use triangular::SparseTriangular;

mod ordering;
mod traversal;
pub use ordering::{
    SparseOrdering, approximate_minimum_degree, derivative_coloring, distance_one_coloring,
    reverse_cuthill_mckee,
};
pub use traversal::{SparseTraversal, SparseTraversalView, transpose_pattern};

mod storage;
pub use storage::{
    BsrMatrix, DiaMatrix, SparseEntries, SparseOperator, SparseStorageRequirements, SparseView,
};

mod arithmetic;
pub use arithmetic::{
    SparseArithmeticOptions, SparsePruning, sparse_block_diagonal, sparse_block_grid,
    sparse_diagonals,
};

mod product;
pub use product::{
    SparseOperation, SparseProductOptions, SparseProductPlan, SparseProductPlanOptions,
    SparseProductRequirements, SparseProductWorkspace, sparse_dense_into,
};

mod graph;
pub use graph::{
    AssignmentObjective, AssignmentOutcome, AssignmentPlan, AssignmentResiduals, AssignmentView,
    AssignmentWorkspace, AssignmentWorkspaceRequirements, BipartiteMatching, CapacityEdge,
    FlowView, FlowWorkspace, FlowWorkspaceRequirements, ForestEdge, FullAssignment,
    GraphComponents, GraphDirection, GraphExecutionOptions, GraphPathMethod, GraphPathResult,
    GraphPathView, GraphPlan, GraphTraversal, GraphWorkspace, GraphWorkspaceRequest,
    GraphWorkspaceRequirements, LaplacianDegree, LaplacianNormalization, LaplacianPlan,
    MatchingView, MatchingWorkspace, MatchingWorkspaceRequirements, MaxFlowPlan, MaximumFlow,
    MinimumSpanningForest, NegativeCycle, PathOutcome, connected_components, laplacian,
    maximum_bipartite_matching, maximum_bipartite_matching_scoped,
    maximum_bipartite_matching_with_options, maximum_flow, minimum_cost_assignment, shortest_paths,
};

pub mod backend;
mod hermitian;
pub use hermitian::{LdlInertia, LdlPivotBlock};
mod symmetric;
pub use symmetric::{
    SparseCholesky, SparseLdl, SparseSymmetricLimits, SparseSymmetricRequirements,
    SparseSymmetricSymbolic, SparseSymmetricWorkspace,
};

mod incomplete;
pub use incomplete::{
    IncompleteDiagnostics, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
    IncompletePivotPolicy, IncompleteRequirements, IncompleteSymbolic, IncompleteWorkspace,
};

mod amg;
pub use amg::{
    AmgCycle, AmgCycleKind, AmgCycleOptions, AmgDiagnostics, AmgHierarchy, AmgLevelView, AmgLimits,
    AmgPreconditioner, AmgSetupOptions, AmgSetupTermination, AmgSmootherView,
};
