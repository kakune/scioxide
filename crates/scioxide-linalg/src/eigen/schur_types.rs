//! Schur blocks, options, results and workspace requests.

use super::schur::decompose_lengths;
use super::support::direct_requirements;
use crate::ConditionEstimate;
use crate::{Estimation, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix};
use alloc::vec::Vec;
use scioxide_core::{
    Budget, Complex, NumericScalar, RealScalar, ReliableEstimate, SciOxideError, Tolerance,
};

/// A diagonal block of a Schur form: 1×1, or a real 2×2 block holding a complex conjugate pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurBlock {
    /// A 1×1 block holding one real or complex eigenvalue.
    OneByOne {
        /// Index of the block's row and column.
        start: usize,
    },
    /// A real 2×2 block holding two complex conjugate eigenvalues.
    ///
    /// It occurs only for real input, and cannot be split by a reordering or
    /// an invariant subspace.
    RealTwoByTwo {
        /// Index of the block's first row and column.
        start: usize,
    },
}
impl SchurBlock {
    /// Returns the index of the block's first row and column.
    pub const fn start(self) -> usize {
        match self {
            Self::OneByOne { start } | Self::RealTwoByTwo { start } => start,
        }
    }
    /// Returns the order of the block, 1 or 2.
    pub const fn size(self) -> usize {
        match self {
            Self::OneByOne { .. } => 1,
            Self::RealTwoByTwo { .. } => 2,
        }
    }
    pub(super) const fn sized(start: usize, size: usize) -> Self {
        if size == 2 {
            Self::RealTwoByTwo { start }
        } else {
            Self::OneByOne { start }
        }
    }
}

/// Options for the Schur decomposition and the generalized Schur (QZ) decomposition.
///
/// The default uses the deflation tolerance [`Tolerance::from_machine_epsilon`]
/// and the default [`LinalgExecutionOptions`]. Without an explicit iteration
/// budget, QR or QZ iteration may take `30 max(n, 1)` steps for order `n`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_linalg::SchurOptions;
/// let options = SchurOptions::<f64>::new().with_execution_options(
///     scioxide_linalg::LinalgExecutionOptions::new()
///         .with_iteration_budget(Budget::ONE.saturating_add(9)),
/// );
/// assert_eq!(options.execution_options().iteration_budget().unwrap().get(), 10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SchurOptions<R: RealScalar> {
    pub(super) execution: LinalgExecutionOptions,
    pub(super) tolerance: Tolerance<R>,
}
impl<R: RealScalar> Default for SchurOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> SchurOptions<R> {
    /// Sets the execution policy, product accuracy, limits and iteration budget.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Creates the default options.
    pub fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            tolerance: Tolerance::from_machine_epsilon(),
        }
    }
    /// Sets the tolerance below which subdiagonal entries are treated as zero
    /// during QR or QZ iteration.
    ///
    /// It applies to the input scaled so that its largest entry magnitude is 1.
    pub const fn with_deflation_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }
    /// Returns the deflation tolerance.
    pub const fn deflation_tolerance(self) -> Tolerance<R> {
        self.tolerance
    }
    /// Explicit maximum QR/QZ work units; `None` selects `30 * max(order, 1)`.
    pub(crate) const fn iteration_budget(self) -> Option<Budget> {
        self.execution.iteration_budget()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SchurJob {
    Decompose,
    Reorder(Estimation),
}

/// Workspace request for a Schur decomposition or reordering of a given order.
///
/// [`Self::new`] covers [`SchurDecomposition::decompose_into`] and
/// [`Self::reorder`] covers [`SchurDecomposition::reorder_in_place`]. Combine
/// the requirements of both with [`LinalgRequirements::max`] to size one
/// workspace for both.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{SchurDecomposition, SchurOptions, SchurWorkspaceRequest};
/// let request = SchurWorkspaceRequest::new(4, &SchurOptions::<f64>::new());
/// let requirements = SchurDecomposition::<f64>::workspace_requirements(&request)?;
/// let workspace = SchurDecomposition::<f64>::workspace(&request)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SchurWorkspaceRequest {
    pub(super) order: usize,
    pub(super) job: SchurJob,
    pub(super) execution: LinalgExecutionOptions,
}
impl SchurWorkspaceRequest {
    /// Creates a request for the decomposition of an `order × order` matrix.
    pub const fn new<R: RealScalar>(order: usize, options: &SchurOptions<R>) -> Self {
        Self {
            order,
            job: SchurJob::Decompose,
            execution: options.execution,
        }
    }
    /// Creates a request for reordering a decomposition of order `order`.
    ///
    /// Scratch for condition estimates is included only with
    /// [`Estimation::Estimate`].
    pub const fn reorder(order: usize, options: &super::SchurReorderOptions) -> Self {
        Self {
            order,
            job: SchurJob::Reorder(options.condition_estimates()),
            execution: options.execution_options(),
        }
    }
}

/// Schur decomposition `A = Q T Qᴴ` of a square matrix.
///
/// `Q` is unitary, or orthogonal for real input. For complex input `T` is
/// upper triangular. For real input `T` is real and upper quasi-triangular:
/// each complex conjugate pair of eigenvalues occupies a 2×2 diagonal block in
/// the standardized form `[[a, b], [c, a]]` with `b c < 0`, whose eigenvalues
/// are `a ± i √(|b| |c|)`. [`Self::blocks`] lists the diagonal blocks and
/// [`Self::eigenvalues`] the eigenvalues in block order.
///
/// The eigenvalues appear in the order the QR iteration leaves them. Use
/// [`Self::reorder`] to move selected eigenvalues to the leading block and
/// [`Self::invariant_subspace`] to read the corresponding invariant subspace.
///
/// Compute it with [`Self::decompose`].
#[derive(Debug, Clone)]
pub struct SchurDecomposition<S: NumericScalar> {
    pub(super) report: Option<super::EigenReport>,
    pub(super) t: Matrix<S>,
    pub(super) q: Matrix<S>,
    pub(super) values: Vec<Complex<S::Real>>,
    pub(super) blocks: Vec<SchurBlock>,
}

impl<S: NumericScalar> Default for SchurDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> SchurDecomposition<S> {
    /// Creates an empty decomposition for use with [`Self::decompose_into`].
    pub const fn empty() -> Self {
        Self {
            report: None,
            t: Matrix::empty(),
            q: Matrix::empty(),
            values: Vec::new(),
            blocks: Vec::new(),
        }
    }
    /// Returns the report of the last successful decomposition, or `None` before one.
    pub const fn report(&self) -> Option<super::EigenReport> {
        self.report
    }
    /// Returns the Schur form `T`: triangular, or quasi-triangular for real input.
    pub const fn t(&self) -> &Matrix<S> {
        &self.t
    }
    /// Returns the Schur vectors `Q`.
    pub const fn q(&self) -> &Matrix<S> {
        &self.q
    }
    /// Returns the eigenvalues in block order, with the positive imaginary
    /// part first in each real 2×2 block.
    pub fn eigenvalues(&self) -> &[Complex<S::Real>] {
        &self.values
    }
    /// Returns the diagonal blocks of `T` in order.
    pub fn blocks(&self) -> &[SchurBlock] {
        &self.blocks
    }
    /// Returns the workspace requirements of a decomposition or reordering.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &SchurWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let lengths = match request.job {
            SchurJob::Decompose => {
                decompose_lengths::<S>(request.order, request.execution.execution_policy())?
            }
            SchurJob::Reorder(estimates) => {
                super::schur_reorder::reorder_lengths::<S>(request.order, estimates)?
            }
        };
        direct_requirements::<S>(lengths, request.execution.byte_limit())
    }
    /// Allocates a workspace for a decomposition or reordering.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace(
        request: &SchurWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

/// Options for reordering a Schur or generalized Schur decomposition.
///
/// The default skips condition estimates ([`Estimation::Skip`]) and uses the
/// default [`LinalgExecutionOptions`]. Without an explicit iteration budget, a
/// reordering may take 30 000 work units: block swaps plus the Sylvester
/// operator applications of the condition estimates.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Estimation, SchurReorderOptions};
/// let options = SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate);
/// assert_eq!(options.condition_estimates(), Estimation::Estimate);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SchurReorderOptions {
    pub(super) execution: LinalgExecutionOptions,
    pub(super) estimates: Estimation,
}
impl Default for SchurReorderOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SchurReorderOptions {
    /// Sets the execution policy, limits and iteration budget.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Creates the default options.
    pub const fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            estimates: Estimation::Skip,
        }
    }
    /// Sets whether the reordering estimates the condition of the selected
    /// cluster and the separation of the two diagonal blocks.
    pub const fn with_condition_estimates(mut self, estimates: Estimation) -> Self {
        self.estimates = estimates;
        self
    }
    /// Returns whether condition estimates are computed.
    pub const fn condition_estimates(self) -> Estimation {
        self.estimates
    }
    /// Maximum number of work units.
    pub(crate) const fn iteration_budget(self) -> Budget {
        match self.execution.iteration_budget() {
            Some(limit) => limit,
            None => scioxide_core::Budget::ONE.saturating_add(29999),
        }
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }
}

/// Diagnostics of a Schur reordering.
///
/// After reordering, `T = [[T₁₁, T₁₂], [0, T₂₂]]`, where `T₁₁` holds the
/// selected eigenvalues. The condition numbers are estimates, not bounds.
///
/// # References
///
/// Z. Bai, J. W. Demmel and A. McKenney, "On computing condition numbers for
/// the nonsymmetric eigenproblem", *ACM Trans. Math. Softw.* 19, 1993.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchurReorderReport<R: RealScalar> {
    pub(super) selected: usize,
    pub(super) swaps: usize,
    pub(super) iterations: usize,
    pub(super) condition: Option<ConditionEstimate<R>>,
    pub(super) separation: Option<ReliableEstimate<R>>,
}
impl<R: RealScalar> SchurReorderReport<R> {
    /// Returns the order `m` of `T₁₁`, the number of selected eigenvalues.
    pub const fn selected_dimension(self) -> usize {
        self.selected
    }
    /// Returns the number of adjacent block swaps performed.
    pub const fn swaps(self) -> usize {
        self.swaps
    }
    /// Returns the work units used: attempted swaps plus Sylvester operator applications.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the reciprocal condition number `1 / √(1 + ‖R‖_F²)` of the
    /// selected cluster of eigenvalues, or `None` without [`Estimation::Estimate`].
    ///
    /// `R` solves the Sylvester equation `T₁₁ R − R T₂₂ = T₁₂`. The value lies
    /// in `[0, 1]`; small values mean an ill-conditioned cluster.
    pub const fn cluster_condition(self) -> Option<ConditionEstimate<R>> {
        self.condition
    }
    /// Returns an estimate of the separation `sep(T₁₁, T₂₂)`, in the units of
    /// `A`, or `None` without [`Estimation::Estimate`].
    ///
    /// The separation is `1 / ‖S⁻¹‖` for the Sylvester operator
    /// `S(R) = T₁₁ R − R T₂₂`, estimated with a 1-norm estimator. Small values
    /// mean an ill-conditioned invariant subspace.
    pub const fn separation(self) -> Option<ReliableEstimate<R>> {
        self.separation
    }
}
