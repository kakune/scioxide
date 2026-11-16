//! Dense and structured linear algebra, eigenproblems and iterative solvers.
//!
//! The crate solves linear systems and least-squares problems with direct
//! factorizations (LU, Cholesky, LDLᴴ, QR, SVD) and with Krylov and stationary
//! iterations. It computes eigenvalues, Schur forms and singular values of dense
//! matrices, and a few eigenpairs or singular triplets of large operators. It
//! also evaluates matrix functions such as `exp(A)` and their actions on vectors,
//! and solves Sylvester, Lyapunov and algebraic Riccati equations. Real and
//! complex `f32` and `f64` scalars are supported throughout.
//!
//! # Quick start
//!
//! ```
//! use scioxide_linalg::{LuDecomposition, Matrix, Vector, solve};
//!
//! // Solve A x = b. The exact solution is x = (2, 3, −1).
//! let a = Matrix::from_rows(&[[2.0_f64, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])?;
//! let x = solve(&a, &Vector::from_vec(vec![8.0, -11.0, -3.0]))?;
//! for (computed, exact) in x.as_slice().iter().zip([2.0, 3.0, -1.0]) {
//!     assert!((computed - exact).abs() < 1e-12);
//! }
//! // Factor once and reuse the factorization for more right-hand sides.
//! let lu = LuDecomposition::factor(&a)?;
//! let y = lu.solve(&Vector::from_vec(vec![2.0, -2.0, 1.0]))?;
//! assert!((y.as_slice()[0] - 1.0).abs() < 1e-12);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | Shortest call | Method |
//! |---|---|---|
//! | General square system | [`solve`], [`LuDecomposition::factor`] | blocked LU with partial pivoting, about `2n³/3` flops |
//! | Determinant, inverse, condition estimate | methods of [`LuDecomposition`], such as [`LuDecomposition::determinant`] | from the LU factors |
//! | Hermitian positive definite system | [`CholeskyDecomposition::factor`] | blocked Cholesky, about `n³/3` flops; [`CholeskyPivoting`] for semidefinite matrices |
//! | Hermitian indefinite system, inertia | [`LdlDecomposition::factor`] | blocked `LDLᴴ` with bounded rook pivoting |
//! | Least squares, minimum-norm solutions | [`least_squares`], [`HouseholderDecomposition::factor`], [`HouseholderMinimumNormPlan`] | Householder QR or LQ, optional column pivoting |
//! | Singular values, rank, pseudoinverse solves | [`SvdValues::compute`], [`ThinSvdDecomposition::compute`], [`SvdDecomposition::compute`] | Golub–Kahan bidiagonalization, then dqds (values) or QR and divide and conquer (vectors) |
//! | Adding or removing rows and columns of a QR factorization | [`QrUpdateDecomposition::factor`] | plane rotations, `O(m² + mn)` per change |
//! | GSVD, CS decomposition, polar decomposition | [`GsvdDecomposition::factor`], [`CsdDecomposition::factor`], [`PolarDecomposition::factor`] | SVD and Jacobi rotations |
//! | Triangular system | [`solve_triangular`], [`TriangularPlan`] | forward or back substitution, `n²` flops |
//! | Banded or tridiagonal system | [`BandedLuDecomposition::factor`], [`BandedCholeskyDecomposition::factor`], [`TridiagonalLuDecomposition::factor`], [`CyclicTridiagonalPlan`] | band elimination, linear in `n` for a fixed bandwidth |
//! | Toeplitz, Hankel or circulant system | [`solve_structured`], [`StructuredSolvePlan`] | Levinson recursion, Gohberg–Kailath–Olshevsky elimination, DFT diagonalization |
//! | Hermitian eigenproblem | [`HermitianEigenDecomposition::decompose`] | tridiagonal reduction, then QL/QR, divide and conquer, or bisection and inverse iteration |
//! | General eigenproblem, Schur form | [`GeneralEigenDecomposition::decompose`], [`SchurDecomposition::decompose`] | balancing, Hessenberg reduction, then Francis double-shift QR (real input) or single-shift QR (complex input) |
//! | Generalized eigenproblem `A v = λ B v` | [`GeneralizedHermitianEigenDecomposition::decompose`], [`GeneralizedSchurDecomposition::decompose`] | Cholesky reduction (Hermitian-definite), QZ (general) |
//! | Large sparse or matrix-free system | `solve(a, b)` of [`CgSolver`], [`MinresSolver`], [`GmresSolver`], [`BicgstabSolver`], [`LsqrSolver`], [`LsmrSolver`] and others | Krylov subspace methods; see [Iterative solves](#iterative-solves) |
//! | Matrix function `f(A)` | [`matrix_function`], [`MatrixFunctionPlan`] | scaling and squaring with Padé approximants (`exp`), Schur-based methods (others) |
//! | Action `exp(tA) b` or `f(A) b` | [`ExponentialAction`], [`FunctionAction`] | truncated Taylor series with scaling (`exp`), restarted Arnoldi approximation (other functions); products with `A` only |
//! | A few eigenpairs or singular triplets | [`PartialSpectrumSolver`], [`LobpcgSolver`], [`PartialSvdSolver`] | Krylov–Schur and thick-restart Lanczos, LOBPCG, restarted Golub–Kahan–Lanczos bidiagonalization; products with `A` (and `Aᴴ` for singular triplets) |
//! | Low-rank approximation, norm estimation | [`RandomizedLowRank`], [`OneNormEstimator`] | randomized range finder with your sketch; Hager–Higham one-norm estimation from products with `A` and `Aᴴ` |
//! | Norms and inner products | [`MatrixView::norm`], [`VectorView::norm2`], [`VectorView::dotc_left`], [`operator_norm`] | overflow-free scaled sums |
//! | Sylvester, Lyapunov, Riccati equations | [`solve_sylvester`], [`solve_lyapunov`], [`solve_riccati`] | Bartels–Stewart; ordered QZ of the extended pencil |
//! | Refining a solution in higher precision | [`IterativeRefinement`] | mixed-precision iterative refinement |
//! | Orthonormal bases, principal angles, Procrustes | [`orth`], [`null_space`], [`subspace_angles`], [`orthogonal_procrustes`] | SVD |
//!
//! # Matrices and views
//!
//! [`Matrix`] and [`Vector`] own contiguous storage. Matrices are stored in
//! row-major order. [`MatrixView`], [`MatrixViewMut`], [`VectorView`] and
//! [`VectorViewMut`] borrow slices with element strides, so a submatrix, a
//! transposed layout or a column needs no copy. A mutable view rejects layouts in
//! which two entries share storage. Functions accept anything that converts into
//! a view, such as `&Matrix` or `&[S]`.
//!
//! Inputs are checked for NaN and infinity before any work starts. To check a
//! matrix once and reuse it, convert it with [`MatrixView::try_finite`] into a
//! [`FiniteMatrixView`], which also serves as a matrix-free
//! [`LinearOperator`](scioxide_core::LinearOperator) for the iterative solvers.
//!
//! # Execution
//!
//! An operation has up to four forms, named by their suffix:
//!
//! - The plain form, such as [`LuDecomposition::factor`] or
//!   [`VectorView::norm2`], takes the problem data and nothing else. It uses
//!   the default options, allocates its result and runs sequentially on
//!   [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
//!   A plain solve returns the solution and reports a rejection as an error.
//! - The `*_with_options` form, such as [`LuDecomposition::factor_with_options`],
//!   is the configured owned form: it takes the options as its last argument,
//!   so it can select [`Accuracy::Exact`] or a byte limit, and a solve returns
//!   the complete result with its report. It also runs sequentially, and that
//!   context has no clock, so a time limit is rejected with
//!   [`CapabilityUnavailable`](scioxide_core::SciOxideErrorKind::CapabilityUnavailable).
//! - The `*_into` form writes into an output and a workspace that you
//!   provide, and takes an [execution context](scioxide_core#execution-contexts)
//!   or a [`LinalgExecutionContext`]. It can run in parallel, observe
//!   cancellation and deadlines, and use an injected BLAS provider.
//! - An operation whose result is a value and not an array, such as
//!   [`VectorView::norm2_scoped`] or [`LuDecomposition::condition_estimate_scoped`],
//!   has a `*_scoped` form instead, which takes the options, the workspace if
//!   there is one, and the context last, and returns the value.
//!
//! [`LinalgExecutionOptions`] sets the execution policy, the time and byte limits,
//! the iteration budget and the product [`Accuracy`].
//! Option types of each method family embed one value of it through
//! `with_execution_options`. A workspace over the byte limit is a
//! `ResourceLimit` error, a time limit on an owned form is
//! `CapabilityUnavailable`, and a dense iteration that exhausts its budget is
//! a `ConvergenceFailure`.
//!
//! # Direct solves
//!
//! A factorization fails only for a structural reason, such as an exact zero
//! pivot in LU, or when an entry of a factor overflows. Whether a solution is
//! acceptable is decided separately when you solve, by the [`RankPolicy`] and
//! the optional minimum reciprocal condition number in [`DirectSolveOptions`].
//! The default options use [`RankPolicy::Exact`], which rejects only an
//! exactly singular factor, and set no condition threshold. A residual
//! requested with [`DirectSolveOptions::with_residual`] is reported but does
//! not reject the solution. Methods with a residual tolerance in their own
//! options, such as [`StructuredSolvePlan`] and the matrix equation plans,
//! reject a solution whose residual exceeds it.
//!
//! A plain `solve` returns the solution and turns a rejection into an error
//! ([`DirectSolveTermination::require_accepted`]): `Singular` for a
//! rank-deficient factor and `NumericalInstability` otherwise. A
//! `solve_with_options` or `solve_into` call that these checks reject is not
//! an error: it returns `Ok` with a [`DirectSolveTermination`] other than
//! `Accepted` and without a solution, and a `*_into` solve then leaves its
//! output unchanged. The report keeps the rank and condition diagnostics of
//! the rejected candidate, and its residual when a residual test rejected it.
//!
//! # Iterative solves
//!
//! Iterative solvers, partial eigensolvers and refinement stop with a
//! termination status: [`LinearSolveTermination`], [`SpectralTermination`] or
//! [`RefinementTermination`]. Convergence is tested on the residual of the
//! original system, recomputed from the returned iterate.
//!
//! The plain `solve(a, b)` of a single-vector solver starts from zero without
//! preconditioning and returns the solution; a solve that ends without
//! meeting the tolerance is a `ConvergenceFailure`. `solve_with_options` takes
//! a preconditioner and a start vector, and `solve_with_options` and
//! `solve_into` return the status: when the iteration budget, the evaluation
//! budget, cancellation or a deadline ends a solve after a usable iterate,
//! they return `Ok` with an unsuccessful status and the iterate with the
//! smallest verified residual. Budgets count as described in
//! [Solve accounting](scioxide_core#solve-accounting). The report marks the
//! returned iterate as
//! [`PartialResultTrust::ResidualChecked`](scioxide_core::PartialResultTrust::ResidualChecked)
//! only when the solve succeeded, and as `Admissible` otherwise.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `DimensionMismatch`: shapes do not agree, a matrix that must be square is
//!   not, or an output or workspace is too small.
//! - `NonFinite`: an input matrix or vector contains NaN or infinity, or an
//!   operator or preconditioner callback returned one.
//! - `InvalidInput`: an argument is out of range, contradicts another
//!   argument, or selects an unsupported combination, such as a matrix that is
//!   not exactly Hermitian under [`HermitianInput::Exact`] (`Inconsistent`).
//! - `Singular`: a factorization met an exact zero pivot.
//! - `DomainError`: a matrix lacks a required mathematical property, such as
//!   positive definiteness for Cholesky.
//! - `NumericalInstability`: a result such as a determinant or an eigenvector
//!   is not representable in the scalar type.
//! - `ConvergenceFailure`: a dense eigenvalue or SVD iteration exhausted its
//!   budget, or an iterative method stopped before producing a usable iterate.
//! - `ResourceLimit`, `AllocationFailure`: a workspace exceeds the byte limit or
//!   cannot grow.
//!
//! On error, outputs are unchanged as described in
//! [Output on error](scioxide_core#output-on-error). Numerical rejection of a
//! configured solve is not an error (see [Direct solves](#direct-solves)).
//!
//! # Accuracy
//!
//! The dense factorizations are normwise backward stable. For example, the
//! computed LU factors satisfy `P(A + ΔA) = L U` with
//! `‖ΔA‖∞ ≤ c n ε ‖|L| |U|‖∞`, where `ε` is the machine epsilon of the scalar type
//! and `c` is a modest constant (Higham, 2002, chapter 9). The forward error of a
//! solution is about the condition number times the backward error. To measure
//! both, request a residual with [`DirectSolveOptions::with_residual`] and a
//! condition estimate with [`DirectSolveOptions::with_minimum_reciprocal_condition`].
//!
//! Matrix products use the accuracy selected by
//! [`Accuracy`]. `Standard` uses ordinary floating-point
//! arithmetic with power-of-two rescaling to avoid overflow. Its results can
//! differ between native SIMD and portable kernels within the usual
//! inner-product error bound, but for a fixed kernel they do not depend on the
//! number of threads. `Exact` accumulates each inner product exactly and rounds
//! once, so its results are the same on every platform.
//!
//! # Reusing plans and workspaces
//!
//! Plans and workspaces follow the [shared rules](scioxide_core#plans-and-workspaces).
//! Dense factorizations, eigensolvers, matrix functions and matrix equations
//! use one [`LinalgWorkspace`]. Iterative solvers use an [`IterativeWorkspace`],
//! and operator actions, Krylov recycling, refinement and sparse matrix
//! functions have their own workspace types. Each method reports what it needs
//! with `workspace_requirements(request)`. After `try_reserve`, repeated
//! `*_into` calls of the same size do not allocate.
//!
//! ```
//! use scioxide_core::Capacity;
//! use scioxide_linalg::{LinalgExecutionOptions, LuDecomposition, LuWorkspaceRequest};
//!
//! let request = LuWorkspaceRequest::factor(4, &LinalgExecutionOptions::new());
//! let requirements = LuDecomposition::<f64>::workspace_requirements(&request)?;
//! let workspace = LuDecomposition::<f64>::workspace(&request)?;
//! assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Feature flags
//!
//! - `std` (default): enables standard-library execution capabilities from
//!   `scioxide-core`, such as threads, clocks and runtime CPU feature detection.
//!   Without it the crate is `no_std` and needs only `alloc`; every numerical
//!   API stays available.
//! - `blas`: adds the `BlasProvider` trait and related types, through which
//!   you can supply an external BLAS implementation for dense products and
//!   triangular solves. The crate links no BLAS library.
//!
//! Sparse matrices and preconditioners come from `scioxide-sparse`. The
//! [`backend`] module holds lower-level kernels for other SciOxide crates.
//!
//! # References
//!
//! - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
//!   Hopkins University Press, 2013.
//! - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//!   SIAM, 2002.
//! - N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.
//! - Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
//! - J. W. Demmel, *Applied Numerical Linear Algebra*, SIAM, 1997.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg_attr(all(test, not(feature = "std")), macro_use)]
extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod actions;
mod analysis;
/// Lower-level kernels used by other SciOxide crates.
///
/// Unlike the rest of the crate, these kernels may modify their scratch buffers
/// on error, and they apply no rank policy and report no diagnostics.
pub mod backend;
#[cfg(feature = "blas")]
mod blas;
mod checked;
mod context;
mod decompose;
mod dense;
mod diagnostics;
mod eigen;
mod equations;
mod estimate;
mod functions;
mod iterative;

mod kernel;
mod least_squares_trust_region;
mod nullspace;
mod one_shot;
mod options;
pub use least_squares_trust_region::LeastSquaresTrustRegionReport;
mod reduction;
mod refinement;
mod reflector;
mod result;
mod rotation;
mod secular;
mod solve_support;
mod sparse_functions;
mod structured;
mod subspace;
mod traits;
mod workspace;

pub use actions::{
    ExponentialAction, ExponentialActionAccepted, ExponentialActionReport,
    ExponentialActionRequirements, ExponentialActionWorkspace, FunctionAction,
    FunctionActionAccepted, FunctionActionReport, FunctionActionRequirements,
    FunctionActionWorkspace, InterpolativeBuffer, LobpcgBuffer, LobpcgOperator, LobpcgReport,
    LobpcgRequirements, LobpcgSelection, LobpcgSolver, LobpcgWorkspace, LowRankReport,
    OneNormEstimator, OneNormReport, OneNormRequirements, OneNormWorkspace, OperatorStructure,
    PartialSpectrumBuffer, PartialSpectrumReport, PartialSpectrumRequirements,
    PartialSpectrumSolver, PartialSpectrumWorkspace, PartialSvdBuffer, PartialSvdReport,
    PartialSvdRequirements, PartialSvdSolver, PartialSvdWorkspace, PhiFunction, RandomizedLowRank,
    RandomizedLowRankRequirements, RandomizedLowRankWorkspace, RandomizedSvdBuffer,
    SpectralTermination, SpectrumOperator, SpectrumSelection,
};

pub use analysis::{
    ConditionEstimate, DirectSolveOptions, DirectSolveReport, DirectSolveResult,
    DirectSolveTermination, EquilibrationMetadata, RankPolicy, RankReport,
    ReciprocalConditionThreshold,
};

#[cfg(feature = "blas")]
pub use blas::{
    BlasBandMatrix, BlasCall, BlasCapabilities, BlasDispatch, BlasFallbackPolicy, BlasIntegerWidth,
    BlasLayout, BlasMatrix, BlasMatrixMut, BlasOperation, BlasPackedMatrix, BlasPackedMatrixMut,
    BlasProvider, BlasThreading, BlasTransform, BlasUlp, BlasVector, BlasVectorMut,
};

#[cfg(feature = "blas")]
pub(crate) use blas::{BlasStructuredMatrix, BlasStructuredMatrixMut};

pub use context::LinalgExecutionContext;
use scioxide_core::Accuracy;

pub(crate) use context::KernelContext;

pub use decompose::{
    CholeskyDecomposition, CholeskyDefinitenessReport, CholeskyInverseOperator, CholeskyPivoting,
    CholeskyWorkspaceRequest, CsdDecomposition, CsdReport, CsdWorkspaceRequest,
    GeneralizedSingularValue, GsvdDecomposition, GsvdOptions, GsvdReport, GsvdWorkspaceRequest,
    HermitianInput, HermitianProjectionPolicy, HermitianProjectionReport, HouseholderDecomposition,
    HouseholderMinimumNormPlan, HouseholderOrientation, HouseholderPivoting,
    HouseholderWorkspaceRequest, LdlDecomposition, LdlInverseOperator, LdlWorkspaceRequest,
    LuDecomposition, LuInverseOperator, LuWorkspaceRequest, PolarDecomposition, PolarOptions,
    PolarReport, PolarSide, PolarWorkspaceRequest, QrModification, QrRankDecision, QrRankSource,
    QrUpdateDecomposition, QrUpdateOptions, QrUpdateReport, QrUpdateWorkspaceRequest,
    RqDecomposition, RqWorkspaceRequest, SignLogDeterminant, SvdDecomposition, SvdValues,
    SvdWorkspaceRequest, ThinSvdDecomposition, TriangularRankWorkspace,
};

pub use dense::{
    CopyPolicy, FiniteMatrixView, Matrix, MatrixView, MatrixViewMut, Vector, VectorView,
    VectorViewMut,
};

pub(crate) use dense::FiniteVectorView;

pub use eigen::{
    EigenIndexRange, EigenReport, EigenvalueAccuracy, EigenvalueOrder, GeneralEigenBalanceMetadata,
    GeneralEigenBalancing, GeneralEigenDecomposition, GeneralEigenOptions,
    GeneralEigenWorkspaceRequest, GeneralEigenvaluesBuffer, GeneralEigenvaluesResult,
    GeneralEigenvectorSelection, GeneralizedEigenvalue, GeneralizedHermitianEigenDecomposition,
    GeneralizedHermitianEigenWorkspaceRequest, GeneralizedHermitianEigenvaluesBuffer,
    GeneralizedHermitianEigenvaluesResult, GeneralizedSchurDecomposition,
    GeneralizedSchurReorderReport, GeneralizedSchurReport, GeneralizedSchurWorkspaceRequest,
    HermitianEigenDecomposition, HermitianEigenFirstComponents, HermitianEigenOptions,
    HermitianEigenWorkspaceRequest, HermitianEigenvaluesBuffer, HermitianEigenvaluesResult,
    SchurBlock, SchurDecomposition, SchurOptions, SchurReorderOptions, SchurReorderReport,
    SchurWorkspaceRequest,
};

pub use equations::{
    LyapunovKind, LyapunovPlan, MatrixEquationOptions, MatrixEquationReport, MatrixEquationResult,
    RiccatiBalancing, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem, RiccatiReport,
    RiccatiResult, SylvesterPlan, SylvesterSign,
};

pub use functions::{
    AnalyticMatrixFunction, MatrixFunction, MatrixFunctionConditionReport, MatrixFunctionOptions,
    MatrixFunctionPlan, MatrixFunctionReport, MatrixFunctionSettings,
    MatrixFunctionWorkspaceRequest,
};

pub use iterative::{
    BicgSolver, BicgstabSolver, BlockGcrotSolver, BlockGmresSolver, BlockKrylovReport,
    BlockKrylovResult, BlockLgmresSolver, CgSolver, CgsSolver, GaussSeidelSolver, GcrotSolver,
    GmresSolver, IterativeAccepted, IterativeProgress, IterativeRequirements, IterativeWorkspace,
    IterativeWorkspaceRequest, JacobiSolver, KrylovColumns, KrylovRecycleRequirements,
    KrylovRecycleSpace, KrylovRecycleWorkspace, KrylovRecycleWorkspaceRequest, LgmresSolver,
    LsmrSolver, LsqrSolver, MinresSolver, Preconditioning, QmrSolver, RelaxationWeight,
    RichardsonSolver, SorSolver, SorWeight, StationaryMatrix, TfqmrSolver,
};

pub use kernel::matrix::MatrixSide;

pub use kernel::modified_rotation::{
    ModifiedRotation, ModifiedRotationFlag, ModifiedRotationResult, modified_givens_rotation,
    modified_givens_rotation_scoped,
};

pub use kernel::vector::{
    GivensRotation, GivensRotationResult, givens_rotation, givens_rotation_scoped,
};

pub use nullspace::{
    BorderedSystem, BorderedSystemDiagnostics, BorderedSystemOperator, BorderedSystemOptions,
    BorderedSystemWorkspaceRequest, CompatibilityPolicy,
};

pub use one_shot::{
    least_squares, matrix_function, solve, solve_lyapunov, solve_riccati, solve_structured,
    solve_sylvester, solve_triangular,
};

pub use options::{Estimation, LinalgExecutionOptions};

pub use refinement::{
    BackwardError, IterativeRefinement, RefinementCriterion, RefinementFallback,
    RefinementFallbackReason, RefinementOptions, RefinementProblem, RefinementReport,
    RefinementRequirements, RefinementSystem, RefinementTermination, RefinementWorkspace,
    RefinementWorkspaceRequest,
};

pub use result::{
    IterativeSolveReport, IterativeSolveResult, KrylovDiagnostics, LinearSolveBreakdown,
    LinearSolveTermination,
};

pub use sparse_functions::{
    SparseFunctionOptions, SparseFunctionPlan, SparseMaterialization, SparseMaterializationReport,
    SparseMaterializationRequirements, SparseMaterializationWorkspace, SparsePowerOptions,
    SparsePowerPlan,
};

pub use structured::{
    BandedCholeskyDecomposition, BandedLuDecomposition, BandedMatrix, BandedMatrixView,
    BlockDiagonalOperator, CompanionKind, CompositeSolveOptions, CompositeSolveReport,
    ConvolutionMode, CyclicTridiagonalPlan, DftDirection, DftNormalization, HelmertKind,
    HermitianBandMatrix, HermitianBandMatrixView, KroneckerOperator, MatrixForm, MatrixNorm,
    OperatorWorkspaceRequest, PackedKind, PackedMatrix, PackedMatrixView, PackedMatrixViewMut,
    PascalKind, StructuredMatrix, StructuredMatrixView, StructuredOperation,
    StructuredSolveOptions, StructuredSolvePlan, StructuredSolveReport, StructuredWorkspaceRequest,
    TriangularPlan, TridiagonalLuDecomposition, TridiagonalMatrix, TridiagonalMatrixView,
    apply_operator_matrix_into, khatri_rao, khatri_rao_into, khatri_rao_with_options,
    materialize_operator_into, operator_norm, operator_norm_scoped, operator_norm_with_options,
};

pub use subspace::{
    ProcrustesAlignment, ProcrustesReport, ProcrustesUniqueness, ProcrustesWorkspaceRequest,
    SubspaceAngles, SubspaceAnglesReport, SubspaceAnglesWorkspaceRequest, SubspaceBasis,
    SubspaceBasisWorkspaceRequest, SubspaceReport, null_space, null_space_into, orth, orth_into,
    orthogonal_procrustes, orthogonal_procrustes_into, subspace_angles, subspace_angles_into,
};

pub use traits::IterativeLinearSolver;

pub use workspace::{
    DenseWorkspaceRequest, LinalgRequirements, LinalgWorkspace, LinalgWorkspaceRequest,
};

pub(crate) use workspace::LinalgSlices;

// Unit-test allocation assertions in every module rely on this tracker.

#[cfg(test)]
#[path = "../tests/support/bidiagonal.rs"]
mod bidiagonal_fixtures;
