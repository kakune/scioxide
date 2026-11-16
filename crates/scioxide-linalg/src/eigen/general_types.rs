//! Options, results and workspace requests of general eigenproblems.

use super::general::{values_lengths, vectors_lengths};
use super::support::direct_requirements;
use crate::{LinalgRequirements, LinalgWorkspace};
use scioxide_core::Complex;
use scioxide_core::SciOxideError;
use scioxide_core::{Budget, NumericScalar, RealScalar, Tolerance};

use crate::ConditionEstimate;
use crate::LinalgExecutionOptions;
use crate::dense::Matrix;

use super::types::EigenvalueOrder;
use alloc::vec::Vec;

/// Balancing applied to a general matrix before its eigenvalues are computed.
///
/// Balancing is the similarity `B = D⁻¹ Pᵀ A P D`, where `P` is a permutation
/// and `D` is diagonal with power-of-two entries. It does not change the
/// eigenvalues and introduces no rounding error. The permutation isolates
/// eigenvalues that can be read from the diagonal. The scaling equalizes the
/// 2-norms of each row and its column in the remaining window, including the
/// diagonal entry, so diagonally dominant rows stay unscaled and a
/// well-conditioned problem is not made worse. Eigenvectors are transformed
/// back to the coordinates of `A`.
///
/// # References
///
/// - B. N. Parlett and C. Reinsch, "Balancing a matrix for calculation of
///   eigenvalues and eigenvectors", *Numer. Math.* 13, 1969.
/// - R. James, J. Langou and B. R. Lowery, "On matrix balancing and eigenvector
///   computation", 2014.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum GeneralEigenBalancing {
    /// Do not permute or scale the matrix.
    None,
    /// Isolate decoupled rows and columns with similarity permutations.
    Permute,
    /// Scale the whole matrix with powers of two.
    Scale,
    /// Permute, then scale the remaining window with powers of two.
    #[default]
    PermuteAndScale,
}

/// Eigenvectors computed by a [`GeneralEigenDecomposition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum GeneralEigenvectorSelection {
    /// Left eigenvectors `yᴴ A = λ yᴴ` only.
    Left,
    /// Right eigenvectors `A x = λ x` only.
    #[default]
    Right,
    /// Left and right eigenvectors, which also give eigenvalue condition estimates.
    Both,
}

impl GeneralEigenvectorSelection {
    pub(super) const fn includes_left(self) -> bool {
        matches!(self, Self::Left | Self::Both)
    }

    pub(super) const fn includes_right(self) -> bool {
        matches!(self, Self::Right | Self::Both)
    }
}

/// Balancing similarity `B = D⁻¹ Pᵀ A P D` used by a general eigendecomposition.
///
/// See [`GeneralEigenBalancing`] for the definition.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct GeneralEigenBalanceMetadata<R: RealScalar> {
    pub(super) strategy: GeneralEigenBalancing,
    pub(super) scale_factors: Vec<R>,
    pub(super) permutation: Vec<usize>,
}

impl<R: RealScalar> GeneralEigenBalanceMetadata<R> {
    const fn empty() -> Self {
        Self {
            strategy: GeneralEigenBalancing::None,
            scale_factors: Vec::new(),
            permutation: Vec::new(),
        }
    }

    /// Returns the requested balancing.
    pub const fn strategy(&self) -> GeneralEigenBalancing {
        self.strategy
    }

    /// Returns the diagonal of `D`, indexed in balanced coordinates.
    ///
    /// The factors are positive powers of two. Indices isolated by the
    /// permutation have factor 1. The scaling is limited so that no nonzero
    /// entry of the matrix becomes zero and the factors and their reciprocals
    /// stay in the normal floating-point range.
    pub fn scale_factors(&self) -> &[R] {
        &self.scale_factors
    }

    /// Returns the permutation `P` as a map from balanced to original indices.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
}

/// Options for general (non-Hermitian) eigenproblems.
///
/// The default uses deflation and residual tolerances from
/// [`Tolerance::from_machine_epsilon`], [`GeneralEigenBalancing::PermuteAndScale`],
/// [`EigenvalueOrder::Ascending`], right eigenvectors only, and the default
/// [`LinalgExecutionOptions`]. Without an explicit iteration budget, QR
/// iteration may take `30 max(n, 1)` steps for order `n`.
///
/// The matrix is scaled by its largest entry magnitude, balanced, and scaled
/// again before QR iteration. The deflation tolerance applies to this scaled
/// matrix. The pivot protection in eigenvector computation and the
/// linear-independence check of eigenvectors depend only on the precision,
/// order and scale of the matrix, not on either tolerance. Independence is
/// checked in balanced coordinates and residuals in the coordinates of `A`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ExecutionPolicy};
/// use scioxide_linalg::{GeneralEigenOptions, GeneralEigenvectorSelection};
/// let options = GeneralEigenOptions::<f64>::new()
///     .with_eigenvectors(GeneralEigenvectorSelection::Both)
///     .with_execution_options(
///         scioxide_linalg::LinalgExecutionOptions::new()
///             .with_iteration_budget(Budget::ONE.saturating_add(99))
///             .with_execution_policy(ExecutionPolicy::Sequential),
///     );
/// assert_eq!(options.eigenvectors(), GeneralEigenvectorSelection::Both);
/// assert_eq!(options.execution_options().iteration_budget().unwrap().get(), 100);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct GeneralEigenOptions<R: RealScalar> {
    execution: LinalgExecutionOptions,
    tolerance: Tolerance<R>,
    residual_tolerance: scioxide_core::ResidualTolerance<R>,
    balancing: GeneralEigenBalancing,
    ordering: EigenvalueOrder,
    eigenvectors: GeneralEigenvectorSelection,
}

impl<R: RealScalar> GeneralEigenOptions<R> {
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
            residual_tolerance: scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::from_machine_epsilon(),
            ),
            balancing: GeneralEigenBalancing::PermuteAndScale,
            ordering: EigenvalueOrder::Ascending,
            eigenvectors: GeneralEigenvectorSelection::Right,
        }
    }

    /// Sets the tolerance below which subdiagonal entries are treated as zero
    /// during QR iteration.
    ///
    /// It applies to the balanced matrix scaled so that its largest entry
    /// magnitude is 1.
    pub const fn with_deflation_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the tolerance for accepting eigenvectors.
    ///
    /// Each relative residual (see [`GeneralEigenDecomposition::right_residuals`])
    /// must not exceed `32 max(n, 1) tolerance.threshold(1)`. Otherwise the
    /// decomposition fails with `NumericalInstability`.
    pub const fn with_residual_tolerance(
        mut self,
        tolerance: scioxide_core::ResidualTolerance<R>,
    ) -> Self {
        self.residual_tolerance = tolerance;
        self
    }

    /// Returns the residual tolerance.
    pub const fn residual_tolerance(self) -> scioxide_core::ResidualTolerance<R> {
        self.residual_tolerance
    }

    /// Sets the balancing applied before QR iteration.
    pub const fn with_balancing(mut self, balancing: GeneralEigenBalancing) -> Self {
        self.balancing = balancing;
        self
    }

    /// Sets the order of the returned eigenvalues and eigenvectors.
    pub const fn with_ordering(mut self, ordering: EigenvalueOrder) -> Self {
        self.ordering = ordering;
        self
    }

    /// Sets which eigenvectors [`GeneralEigenDecomposition`] computes.
    pub const fn with_eigenvectors(mut self, eigenvectors: GeneralEigenvectorSelection) -> Self {
        self.eigenvectors = eigenvectors;
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

    /// Explicit QR step budget; `None` selects `30 * max(order, 1)` steps.
    pub(crate) const fn iteration_budget(self) -> Option<Budget> {
        self.execution.iteration_budget()
    }

    /// Returns the balancing.
    pub const fn balancing(self) -> GeneralEigenBalancing {
        self.balancing
    }

    /// Returns the order of the returned eigenvalues.
    pub const fn ordering(self) -> EigenvalueOrder {
        self.ordering
    }

    /// Returns which eigenvectors are computed.
    pub const fn eigenvectors(self) -> GeneralEigenvectorSelection {
        self.eigenvectors
    }
}

impl<R: RealScalar> Default for GeneralEigenOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Workspace request for a general eigenproblem of a given order and options.
///
/// Pass it to `workspace_requirements` or `workspace` of
/// [`GeneralEigenvaluesBuffer`] for eigenvalues or [`GeneralEigenDecomposition`]
/// for eigenvectors. The scalar type `S` of those methods selects real or
/// complex arithmetic.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{GeneralEigenOptions, GeneralEigenWorkspaceRequest, GeneralEigenvaluesBuffer};
/// let request = GeneralEigenWorkspaceRequest::new(4, &GeneralEigenOptions::<f64>::new());
/// let requirements = GeneralEigenvaluesBuffer::<f64>::workspace_requirements::<f64>(&request)?;
/// let workspace = GeneralEigenvaluesBuffer::<f64>::workspace::<f64>(&request)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeneralEigenWorkspaceRequest {
    pub(super) order: usize,
    pub(super) eigenvectors: GeneralEigenvectorSelection,
    pub(super) execution: LinalgExecutionOptions,
}

impl GeneralEigenWorkspaceRequest {
    /// Creates a request for an `order × order` matrix.
    pub const fn new<R: RealScalar>(order: usize, options: &GeneralEigenOptions<R>) -> Self {
        Self {
            order,
            eigenvectors: options.eigenvectors,
            execution: options.execution,
        }
    }
}

/// Reusable output for the eigenvalues of a general matrix.
#[derive(Debug, Clone)]
#[must_use]
pub struct GeneralEigenvaluesBuffer<R: RealScalar> {
    pub(super) eigenvalues: Vec<Complex<R>>,
}

impl<R: RealScalar> GeneralEigenvaluesBuffer<R> {
    /// Returns the bytes of eigenvalue storage held, including capacity from earlier calls.
    ///
    /// Inline fields and allocator overhead are not counted.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn retained_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        scioxide_core::backend::checked_mul(
            self.eigenvalues.capacity(),
            core::mem::size_of::<Complex<R>>(),
            scioxide_core::quantity!("retained_bytes"),
        )
    }

    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            eigenvalues: Vec::new(),
        }
    }

    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[Complex<R>] {
        &self.eigenvalues
    }

    /// Returns the eigenvalues without copying.
    pub fn into_eigenvalues(self) -> Vec<Complex<R>> {
        self.eigenvalues
    }
}

impl<R: RealScalar> Default for GeneralEigenvaluesBuffer<R> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Eigenvalues and left or right eigenvectors of a general square matrix.
///
/// Right eigenvectors satisfy `A x = λ x` and left eigenvectors `yᴴ A = λ yᴴ`.
/// The eigenvalues follow the [`EigenvalueOrder`] of the options, and column
/// `j` of each eigenvector matrix belongs to eigenvalue `j`. Eigenvectors are
/// complex even for a real matrix. Each has unit 2-norm, and its component of
/// largest magnitude (the last one, on ties) is real and positive. For a real
/// matrix, the eigenvectors of a complex conjugate pair of eigenvalues are
/// complex conjugates.
///
/// Compute it with [`Self::decompose`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GeneralEigenDecomposition<R: RealScalar>
where
    Complex<R>: NumericScalar<Real = R>,
{
    pub(super) report: Option<super::EigenReport>,
    pub(super) eigenvalues: Vec<Complex<R>>,
    pub(super) selection: GeneralEigenvectorSelection,
    pub(super) left_eigenvectors: Matrix<Complex<R>>,
    pub(super) right_eigenvectors: Matrix<Complex<R>>,
    pub(super) balancing: GeneralEigenBalanceMetadata<R>,
    pub(super) left_residuals: Vec<R>,
    pub(super) right_residuals: Vec<R>,
    pub(super) eigenvalue_conditions: Vec<ConditionEstimate<R>>,
    pub(super) eigenvector_conditions: Vec<ConditionEstimate<R>>,
}

impl<R: RealScalar> GeneralEigenDecomposition<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Creates an empty result for use with [`Self::decompose_into`].
    pub const fn empty() -> Self {
        Self {
            report: None,
            eigenvalues: Vec::new(),
            selection: GeneralEigenvectorSelection::Right,
            left_eigenvectors: Matrix::empty(),
            right_eigenvectors: Matrix::empty(),
            balancing: GeneralEigenBalanceMetadata::empty(),
            left_residuals: Vec::new(),
            right_residuals: Vec::new(),
            eigenvalue_conditions: Vec::new(),
            eigenvector_conditions: Vec::new(),
        }
    }
    /// Returns the report of the last successful computation, or `None` before one.
    pub const fn report(&self) -> Option<super::EigenReport> {
        self.report
    }

    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[Complex<R>] {
        &self.eigenvalues
    }

    /// Returns which eigenvectors this result holds.
    pub const fn eigenvector_selection(&self) -> GeneralEigenvectorSelection {
        self.selection
    }

    /// Returns the left eigenvectors as columns, or `None` if they were not requested.
    pub fn left_eigenvectors(&self) -> Option<&Matrix<Complex<R>>> {
        self.selection
            .includes_left()
            .then_some(&self.left_eigenvectors)
    }

    /// Returns the right eigenvectors as columns, or `None` if they were not requested.
    pub fn right_eigenvectors(&self) -> Option<&Matrix<Complex<R>>> {
        self.selection
            .includes_right()
            .then_some(&self.right_eigenvectors)
    }

    /// Returns the balancing similarity applied to the input.
    pub const fn balancing(&self) -> &GeneralEigenBalanceMetadata<R> {
        &self.balancing
    }

    /// Returns the relative residual of each left eigenvector, or `None` if
    /// they were not requested.
    ///
    /// The residual of `y` is `‖Aᴴ y − conj(λ) y‖∞ / ((‖Aᴴ‖∞ + |λ|) ‖y‖∞)`,
    /// with `‖Aᴴ‖∞ = ‖A‖₁`, computed for the scaled matrix.
    pub fn left_residuals(&self) -> Option<&[R]> {
        self.selection
            .includes_left()
            .then_some(self.left_residuals.as_slice())
    }

    /// Returns the relative residual of each right eigenvector, or `None` if
    /// they were not requested.
    ///
    /// The residual of `x` is `‖A x − λ x‖∞ / ((‖A‖∞ + |λ|) ‖x‖∞)`, computed for
    /// the scaled matrix, so it does not change when `A` is multiplied by a
    /// constant.
    pub fn right_residuals(&self) -> Option<&[R]> {
        self.selection
            .includes_right()
            .then_some(self.right_residuals.as_slice())
    }

    /// Returns the reciprocal condition number `|yᴴ x| / (‖x‖₂ ‖y‖₂)` of each
    /// eigenvalue, or `None` unless both sides were requested.
    ///
    /// Small values mean an ill-conditioned eigenvalue (Wilkinson's `s(λ)`).
    pub fn eigenvalue_condition_estimates(&self) -> Option<&[ConditionEstimate<R>]> {
        (self.selection == GeneralEigenvectorSelection::Both)
            .then_some(self.eigenvalue_conditions.as_slice())
    }

    /// Returns a heuristic reciprocal condition estimate `min(1, gap / ‖A‖∞)` of each eigenvector.
    ///
    /// `gap` is the distance to the nearest other eigenvalue and `‖A‖∞` the norm
    /// of the unbalanced input, so a constant multiple of `A` gives the same
    /// estimates. A 1×1 matrix has estimate 1 and a repeated eigenvalue has
    /// estimate 0.
    pub fn eigenvector_condition_estimates(&self) -> &[ConditionEstimate<R>] {
        &self.eigenvector_conditions
    }
}

impl<R: RealScalar> Default for GeneralEigenDecomposition<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn default() -> Self {
        Self::empty()
    }
}

/// Eigenvalues of a general square matrix, with the iteration report.
///
/// Compute it with [`Self::compute`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GeneralEigenvaluesResult<R: RealScalar> {
    pub(super) output: GeneralEigenvaluesBuffer<R>,
    pub(super) report: super::EigenReport,
}
impl<R: RealScalar> GeneralEigenvaluesResult<R> {
    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[Complex<R>] {
        self.output.eigenvalues()
    }
    /// Returns the iteration report.
    pub const fn report(&self) -> super::EigenReport {
        self.report
    }
    /// Returns the eigenvalues without copying.
    pub fn into_eigenvalues(self) -> Vec<Complex<R>> {
        self.output.eigenvalues
    }
    /// Splits the result into a reusable buffer and the report.
    pub fn into_parts(self) -> (GeneralEigenvaluesBuffer<R>, super::EigenReport) {
        (self.output, self.report)
    }
}

impl<R: RealScalar> GeneralEigenvaluesBuffer<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Returns the workspace requirements for the eigenvalues of a matrix with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        request: &GeneralEigenWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        direct_requirements::<Complex<R>>(
            values_lengths::<S>(request.order, request.execution.execution_policy())?,
            request.execution.byte_limit(),
        )
    }
    /// Allocates a workspace for the eigenvalues of a matrix with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace<S: NumericScalar<Real = R>>(
        request: &GeneralEigenWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, Complex<R>>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}

impl<R: RealScalar> GeneralEigenDecomposition<R>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Returns the workspace requirements for an eigendecomposition with scalar type `S`.
    ///
    /// Each requested side of eigenvectors adds one `n × n` matrix.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        request: &GeneralEigenWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        direct_requirements::<Complex<R>>(
            vectors_lengths::<S>(
                request.order,
                request.eigenvectors,
                request.execution.execution_policy(),
            )?,
            request.execution.byte_limit(),
        )
    }
    /// Allocates a workspace for an eigendecomposition with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace<S: NumericScalar<Real = R>>(
        request: &GeneralEigenWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, Complex<R>>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}
