//! Options, results and workspace requests of Hermitian eigenproblems.

use core::num::NonZeroUsize;

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, Tolerance};

use crate::dense::Matrix;
use crate::workspace::Packing;
use crate::{KernelContext, LinalgRequirements, LinalgWorkspace};

use super::types::{EigenvalueOrder, reserve};
use alloc::vec::Vec;

/// Half-open range `start..end` of eigenvalue indices in ascending order.
///
/// Index `0` is the algebraically smallest eigenvalue. The end is checked
/// against the matrix order when a problem is solved, so one range can be used
/// for problems of different orders.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::EigenIndexRange;
/// let range = EigenIndexRange::new(1, 3)?;
/// assert_eq!((range.start(), range.end()), (1, 3));
/// assert!(EigenIndexRange::new(3, 1).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct EigenIndexRange {
    start: usize,
    end: usize,
}

impl EigenIndexRange {
    /// Creates the index range `start..end`; an empty range is valid.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `start > end`.
    pub fn new(start: usize, end: usize) -> Result<Self, SciOxideError> {
        if start > end {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("start"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "index range must be ordered",
            ));
        }
        Ok(Self { start, end })
    }

    /// Returns the first selected index.
    pub const fn start(self) -> usize {
        self.start
    }

    /// Returns one past the last selected index.
    pub const fn end(self) -> usize {
        self.end
    }
}

/// Accuracy target for the eigenvalues of the tridiagonal problem.
///
/// For a dense matrix, the reduction to tridiagonal form is backward stable
/// only in norm, so [`Self::Relative`] does not give relative accuracy for the
/// small eigenvalues of an arbitrary dense matrix.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{EigenvalueAccuracy, HermitianEigenOptions};
/// let options =
///     HermitianEigenOptions::<f64>::default().with_eigenvalue_accuracy(EigenvalueAccuracy::Relative);
/// assert_eq!(options.eigenvalue_accuracy(), EigenvalueAccuracy::Relative);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenvalueAccuracy {
    /// Converges relative to the magnitude of the whole spectrum. The complete
    /// spectrum uses QL/QR iteration or divide and conquer.
    Absolute,
    /// Converges relative to each eigenvalue's own magnitude, using Sturm
    /// bisection and inverse iteration. This keeps the relative accuracy of
    /// small eigenvalues of graded tridiagonal matrices.
    Relative,
}

/// Options for Hermitian and real symmetric tridiagonal eigenproblems.
///
/// The default uses the deflation tolerance
/// [`Tolerance::from_machine_epsilon`], ascending order, the complete spectrum,
/// [`EigenvalueAccuracy::Absolute`] and the default
/// [`LinalgExecutionOptions`](crate::LinalgExecutionOptions).
///
/// # Iteration budget
///
/// The iteration budget of the execution options bounds all QL/QR sweeps,
/// secular-equation steps, bisection steps and inverse-iteration steps of one
/// call. Called with a [`SolveContext`](scioxide_core::SolveContext), these
/// iterations appear in its `counts().nested()` and do not use the context's
/// own iteration budget. With `k = max(n, 1)` for order `n`, the default
/// budget is:
///
/// | Job | Default budget |
/// |---|---|
/// | complete spectrum: eigenvalues only, or eigenvectors with `n ≤ 32` | `30 k` |
/// | complete spectrum: eigenvectors with `n > 32` (divide and conquer) | `30 k (1 + ⌈log₂ ⌈k / 24⌉⌉)` |
/// | index range or [`EigenvalueAccuracy::Relative`] | `k (E_max − E_min + p + 32)` |
///
/// Here `E_max` and `E_min` are the binary exponents of the largest finite
/// and the smallest subnormal number, and `p` is the significand precision:
/// 2182 per eigenvalue for `f64` and 332 for `f32`. This covers bisection
/// across the whole floating-point range followed by inverse iteration. An
/// explicit budget replaces the default. A default that overflows `usize` is
/// a `SizeOverflow` error.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_linalg::{EigenIndexRange, EigenvalueOrder, HermitianEigenOptions};
/// let options = HermitianEigenOptions::<f64>::default()
///     .with_ordering(EigenvalueOrder::Descending)
///     .with_index_range(EigenIndexRange::new(0, 2)?)
///     .with_execution_options(
///         scioxide_linalg::LinalgExecutionOptions::new()
///             .with_iteration_budget(Budget::ONE.saturating_add(999)),
///     );
/// assert_eq!(options.execution_options().iteration_budget().unwrap().get(), 1000);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct HermitianEigenOptions<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    tolerance: Tolerance<R>,
    ordering: EigenvalueOrder,
    indices: Option<EigenIndexRange>,
    accuracy: EigenvalueAccuracy,
}

impl<R: RealScalar> HermitianEigenOptions<R> {
    /// Sets the execution policy, product accuracy, limits and iteration budget.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the tolerance below which off-diagonal entries of the tridiagonal
    /// matrix are treated as zero.
    ///
    /// The tolerance applies to the matrix scaled so that its largest entry
    /// magnitude is 1.
    pub const fn with_deflation_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the order of the returned eigenvalues and eigenvectors.
    pub const fn with_ordering(mut self, ordering: EigenvalueOrder) -> Self {
        self.ordering = ordering;
        self
    }

    /// Selects the eigenpairs whose ascending indices lie in `range`.
    ///
    /// The output order of [`Self::with_ordering`] is applied after selection.
    /// A range that ends beyond the matrix order is an `InvalidInput` error when
    /// the problem is solved. Generalized Hermitian problems require the
    /// complete spectrum.
    pub const fn with_index_range(mut self, range: EigenIndexRange) -> Self {
        self.indices = Some(range);
        self
    }

    /// Sets the accuracy target for the tridiagonal eigenvalues.
    pub const fn with_eigenvalue_accuracy(mut self, accuracy: EigenvalueAccuracy) -> Self {
        self.accuracy = accuracy;
        self
    }

    /// Returns the selected index range, or `None` for the complete spectrum.
    pub const fn index_range(self) -> Option<EigenIndexRange> {
        self.indices
    }

    /// Returns the accuracy target for the tridiagonal eigenvalues.
    pub const fn eigenvalue_accuracy(self) -> EigenvalueAccuracy {
        self.accuracy
    }

    /// Returns the deflation tolerance, in units of the largest entry magnitude.
    pub const fn deflation_tolerance(self) -> Tolerance<R> {
        self.tolerance
    }

    /// Explicit iteration budget of one call; `None` selects the default.
    pub(crate) const fn iteration_budget(self) -> Option<Budget> {
        self.execution.iteration_budget()
    }

    /// Returns the order of the returned eigenvalues.
    pub const fn ordering(self) -> EigenvalueOrder {
        self.ordering
    }

    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Whether bisection and inverse iteration replace the full-spectrum path.
    pub(super) const fn selects(self) -> bool {
        self.indices.is_some() || matches!(self.accuracy, EigenvalueAccuracy::Relative)
    }

    pub(super) fn resolve_indices(
        self,
        order: usize,
    ) -> Result<core::ops::Range<usize>, SciOxideError> {
        let range = self
            .indices
            .map_or(0..order, |range| range.start..range.end);
        if range.end > order {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "selected eigenvalue index exceeds matrix order",
            ));
        }
        Ok(range)
    }
}

impl<R: RealScalar> Default for HermitianEigenOptions<R> {
    fn default() -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            tolerance: Tolerance::from_machine_epsilon(),
            ordering: EigenvalueOrder::Ascending,
            indices: None,
            accuracy: EigenvalueAccuracy::Absolute,
        }
    }
}

/// Eigenvalues of a symmetric tridiagonal matrix and the first components of its eigenvectors.
///
/// Component `j` of [`Self::first_components`] is the first entry of the unit
/// eigenvector for eigenvalue `j`. Its sign is unspecified. For a repeated
/// eigenvalue the components are not unique, but the sum of their squares over
/// the eigenspace is. Storage is O(n).
///
/// These are the quantities of the Golub–Welsch algorithm. For the Jacobi
/// matrix of a family of orthogonal polynomials, the eigenvalues are the nodes
/// of the Gauss quadrature rule and the weights are the squared first
/// components times the zeroth moment `μ₀`.
///
/// # References
///
/// G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules",
/// *Math. Comp.* 23, 1969.
#[derive(Debug, Clone)]
#[must_use]
pub struct HermitianEigenFirstComponents<R: RealScalar> {
    pub(super) eigenvalues: Vec<R>,
    pub(super) first_components: Vec<R>,
    pub(super) report: Option<super::EigenReport>,
}

impl<R: RealScalar> HermitianEigenFirstComponents<R> {
    /// Creates an empty result for use with [`Self::tridiagonal_into`].
    pub const fn empty() -> Self {
        Self {
            eigenvalues: Vec::new(),
            first_components: Vec::new(),
            report: None,
        }
    }

    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[R] {
        &self.eigenvalues
    }

    /// Returns the first eigenvector components, paired with the eigenvalues.
    pub fn first_components(&self) -> &[R] {
        &self.first_components
    }

    /// Returns the report of the last successful computation, or `None` before one.
    pub const fn report(&self) -> Option<super::EigenReport> {
        self.report
    }

    /// Returns the eigenvalues and first components without copying.
    pub fn into_parts(self) -> (Vec<R>, Vec<R>) {
        (self.eigenvalues, self.first_components)
    }

    pub(super) fn reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        reserve(
            &mut self.eigenvalues,
            order,
            scioxide_core::quantity!("output.eigenvalues"),
        )?;
        reserve(
            &mut self.first_components,
            order,
            scioxide_core::quantity!("output.first_components"),
        )
    }
}

impl<R: RealScalar> Default for HermitianEigenFirstComponents<R> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<R: RealScalar + NumericScalar<Real = R>> HermitianEigenFirstComponents<R> {
    /// Returns the workspace requirements of [`Self::tridiagonal_into`].
    ///
    /// The request must come from [`HermitianEigenWorkspaceRequest::tridiagonal`]
    /// with the complete spectrum and [`EigenvalueAccuracy::Absolute`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the request describes a dense matrix, an index range
    ///   or [`EigenvalueAccuracy::Relative`].
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &HermitianEigenWorkspaceRequest<R>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        super::hermitian::first_components::requirements(request)
    }

    /// Allocates a workspace for [`Self::tridiagonal_into`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace(
        request: &HermitianEigenWorkspaceRequest<R>,
    ) -> Result<LinalgWorkspace<'static, R>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

/// Reusable output for the eigenvalues of a Hermitian or symmetric tridiagonal matrix.
#[derive(Debug, Clone)]
#[must_use]
pub struct HermitianEigenvaluesBuffer<R: RealScalar> {
    pub(super) eigenvalues: Vec<R>,
    pub(super) projection: Option<crate::HermitianProjectionReport<R>>,
}

impl<R: RealScalar> HermitianEigenvaluesBuffer<R> {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            eigenvalues: Vec::new(),
            projection: None,
        }
    }

    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[R] {
        &self.eigenvalues
    }

    /// Returns the change made by projecting the input, or `None` unless the
    /// input was [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn projection_report(&self) -> Option<&crate::HermitianProjectionReport<R>> {
        self.projection.as_ref()
    }

    /// Returns the eigenvalues without copying.
    pub fn into_eigenvalues(self) -> Vec<R> {
        self.eigenvalues
    }

    pub(super) fn reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        reserve(
            &mut self.eigenvalues,
            order,
            scioxide_core::quantity!("output.eigenvalues"),
        )
    }
}

impl<R: RealScalar> Default for HermitianEigenvaluesBuffer<R> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Eigenvalues and orthonormal eigenvectors `A = V Λ Vᴴ` of a Hermitian matrix.
///
/// The eigenvalues are real and follow the [`EigenvalueOrder`] of the options.
/// Column `j` of [`Self::eigenvectors`] is the eigenvector of eigenvalue `j`.
/// Each column has unit 2-norm and is multiplied by a unit-modulus factor so
/// that its component of largest magnitude (the last one, on ties) is real and
/// positive. For a real matrix this fixes the sign of each column. With an
/// index range selecting `k` eigenpairs, `V` is `n × k`.
///
/// Compute it with [`Self::decompose`] or [`Self::tridiagonal`].
#[derive(Debug, Clone)]
#[must_use]
pub struct HermitianEigenDecomposition<S: NumericScalar> {
    pub(super) report: Option<super::EigenReport>,
    pub(super) eigenvalues: Vec<S::Real>,
    pub(super) projection: Option<crate::HermitianProjectionReport<S::Real>>,
    pub(super) eigenvectors: Matrix<S>,
}

impl<S: NumericScalar> HermitianEigenDecomposition<S> {
    /// Creates an empty result for use with the `_into` methods.
    pub const fn empty() -> Self {
        Self {
            report: None,
            eigenvalues: Vec::new(),
            projection: None,
            eigenvectors: Matrix::empty(),
        }
    }
    /// Returns the report of the last successful computation, or `None` before one.
    pub const fn report(&self) -> Option<super::EigenReport> {
        self.report
    }

    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[S::Real] {
        &self.eigenvalues
    }

    /// Returns the change made by projecting the input, or `None` unless the
    /// input was [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn projection_report(&self) -> Option<&crate::HermitianProjectionReport<S::Real>> {
        self.projection.as_ref()
    }

    /// Returns the eigenvector matrix `V`, one eigenvector per column.
    pub const fn eigenvectors(&self) -> &Matrix<S> {
        &self.eigenvectors
    }

    /// Returns the eigenvalues and the eigenvector matrix without copying.
    pub fn into_parts(self) -> (Vec<S::Real>, Matrix<S>) {
        (self.eigenvalues, self.eigenvectors)
    }

    pub(crate) fn action_storage(
        &self,
        plan: &mut crate::workspace::accounting::StoragePlan,
        order: usize,
    ) -> Result<(), SciOxideError> {
        plan.buffer(&self.eigenvalues, order)?;
        plan.matrix(&self.eigenvectors, order, order)
    }

    /// Grow output storage for an `order × count` job without changing its value.
    pub(crate) fn reserve(&mut self, order: usize, count: usize) -> Result<(), SciOxideError> {
        reserve(
            &mut self.eigenvalues,
            count,
            scioxide_core::quantity!("output.eigenvalues"),
        )?;
        self.eigenvectors.try_reserve_shape(order, count)
    }
}

impl<S: NumericScalar> Default for HermitianEigenDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Workspace request for a Hermitian eigenproblem of a given order and options.
///
/// [`Self::new`] describes a dense `n × n` matrix and [`Self::tridiagonal`] a
/// real symmetric tridiagonal one. Pass the request to `workspace_requirements`
/// or `workspace` of the output type: [`HermitianEigenvaluesBuffer`] for
/// eigenvalues, [`HermitianEigenDecomposition`] for eigenvectors, or
/// [`HermitianEigenFirstComponents`] for first components. Generalized
/// problems use
/// [`GeneralizedHermitianEigenWorkspaceRequest`](crate::GeneralizedHermitianEigenWorkspaceRequest).
/// The order, options and output type select the method, so the requirements
/// do not depend on the processor that runs the computation.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{
///     HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
/// };
/// let request = HermitianEigenWorkspaceRequest::new(4, &HermitianEigenOptions::default());
/// let requirements = HermitianEigenDecomposition::<f64>::workspace_requirements(&request)?;
/// let workspace = HermitianEigenDecomposition::<f64>::workspace(&request)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// let tridiagonal =
///     HermitianEigenWorkspaceRequest::tridiagonal(4, &HermitianEigenOptions::default());
/// assert!(
///     HermitianEigenDecomposition::<f64>::workspace_requirements(&tridiagonal)?.scalar_len()?
///         <= requirements.scalar_len()?
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct HermitianEigenWorkspaceRequest<R: RealScalar> {
    pub(super) order: usize,
    pub(super) options: HermitianEigenOptions<R>,
    pub(super) workers: NonZeroUsize,
    pub(super) packing: Packing,
    pub(super) storage: HermitianStorage,
}

/// Matrix storage of a Hermitian eigenjob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HermitianStorage {
    Dense,
    Tridiagonal,
}

impl<R: RealScalar> HermitianEigenWorkspaceRequest<R> {
    /// Creates a request for a dense `order × order` Hermitian matrix.
    pub const fn new(order: usize, options: &HermitianEigenOptions<R>) -> Self {
        Self {
            order,
            options: *options,
            workers: NonZeroUsize::MAX,
            packing: Packing::Native,
            storage: HermitianStorage::Dense,
        }
    }

    /// Creates a request for a real symmetric tridiagonal matrix of order
    /// `order`, as solved by the `tridiagonal_into` methods.
    pub const fn tridiagonal(order: usize, options: &HermitianEigenOptions<R>) -> Self {
        Self {
            storage: HermitianStorage::Tridiagonal,
            ..Self::new(order, options)
        }
    }

    /// Limits the number of parallel workers the workspace is sized for.
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.workers = workers;
        self
    }

    /// Includes packing storage for an injected BLAS provider.
    pub const fn with_provider_packing(mut self) -> Self {
        self.packing = Packing::Provider;
        self
    }

    /// Sets the worker bound and provider packing from `context`.
    pub fn for_context(self, context: &crate::LinalgExecutionContext<'_, '_>) -> Self {
        self.for_kernels(context.kernel())
    }
    pub(crate) fn for_kernels(mut self, context: KernelContext<'_>) -> Self {
        self.workers = context.execution().worker_budget();
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self = self.with_provider_packing();
        }
        self
    }

    /// Dense-product scratch for `rows × shared` by `shared × cols` products.
    ///
    /// The byte limit is applied once to the combined job requirements, so
    /// limit failures report the job's own operation.
    pub(super) fn product<S: NumericScalar<Real = R>>(
        &self,
        rows: usize,
        shared: usize,
        cols: usize,
        structured: bool,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let options = crate::LinalgExecutionOptions::new()
            .with_execution_policy(self.options.execution_options().execution_policy());
        let request = if structured {
            crate::DenseWorkspaceRequest::structured(rows, shared, cols, &options)
        } else {
            crate::DenseWorkspaceRequest::product(rows, shared, cols, &options)
        };
        let mut request = request
            .with_product_accuracy(self.options.execution_options().accuracy())
            .with_worker_bound(self.workers);
        if matches!(self.packing, Packing::Provider) {
            request = request.with_provider_packing();
        }
        crate::kernel::product_requirements::<S>(&request)
    }
}

impl<R: RealScalar> HermitianEigenvaluesBuffer<R> {
    /// Returns the workspace requirements for eigenvalues of a matrix with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        request: &HermitianEigenWorkspaceRequest<R>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        super::hermitian::requirements::<S>(request, super::hermitian::Job::Values)
    }

    /// Allocates a workspace for eigenvalues of a matrix with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace<S: NumericScalar<Real = R>>(
        request: &HermitianEigenWorkspaceRequest<R>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}

impl<S: NumericScalar> HermitianEigenDecomposition<S> {
    /// Returns the workspace requirements for eigenvalues and eigenvectors.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &HermitianEigenWorkspaceRequest<S::Real>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        super::hermitian::requirements::<S>(request, super::hermitian::Job::Vectors)
    }

    /// Allocates a workspace for eigenvalues and eigenvectors.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace(
        request: &HermitianEigenWorkspaceRequest<S::Real>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

impl<S: NumericScalar> crate::workspace::accounting::StorageBytes
    for HermitianEigenDecomposition<S>
{
    fn storage_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        crate::workspace::accounting::sum([
            self.eigenvalues.storage_bytes()?,
            self.eigenvectors.storage_bytes()?,
        ])
    }
}

/// Eigenvalues of a Hermitian or symmetric tridiagonal matrix, with the iteration report.
///
/// Compute it with [`Self::compute`] or [`Self::tridiagonal`].
#[derive(Debug, Clone)]
#[must_use]
pub struct HermitianEigenvaluesResult<R: RealScalar> {
    pub(super) output: HermitianEigenvaluesBuffer<R>,
    pub(super) report: super::EigenReport,
}
impl<R: RealScalar> HermitianEigenvaluesResult<R> {
    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[R] {
        self.output.eigenvalues()
    }
    /// Returns the iteration report.
    pub const fn report(&self) -> super::EigenReport {
        self.report
    }
    /// Returns the eigenvalues without copying.
    pub fn into_eigenvalues(self) -> Vec<R> {
        self.output.eigenvalues
    }
    /// Splits the result into a reusable buffer and the report.
    pub fn into_parts(self) -> (HermitianEigenvaluesBuffer<R>, super::EigenReport) {
        (self.output, self.report)
    }
    /// Returns the change made by projecting the input, or `None` unless the
    /// input was [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn projection_report(&self) -> Option<&crate::HermitianProjectionReport<R>> {
        self.output.projection_report()
    }
}
