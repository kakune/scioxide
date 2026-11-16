//! Results and workspace requests of Hermitian-definite eigenproblems.

use super::generalized::generalized_requirements;
use super::hermitian::Job;
use crate::{
    HermitianEigenOptions, HermitianEigenWorkspaceRequest, HermitianProjectionReport,
    KernelContext, LinalgRequirements, LinalgWorkspace, Matrix,
};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

/// Reusable output for the eigenvalues of a Hermitian-definite pair `A v = λ B v`.
#[derive(Debug, Clone)]
#[must_use]
pub struct GeneralizedHermitianEigenvaluesBuffer<R: RealScalar> {
    pub(super) eigenvalues: Vec<R>,
    pub(super) projections: [Option<HermitianProjectionReport<R>>; 2],
}

/// Eigenvalues and `B`-orthonormal eigenvectors of a Hermitian-definite pair `A v = λ B v`.
///
/// `A` is Hermitian and `B` Hermitian positive definite. The eigenvalues are
/// real and follow the [`EigenvalueOrder`](crate::EigenvalueOrder) of the
/// options. Column `j` of [`Self::eigenvectors`] belongs to eigenvalue `j`,
/// and the columns satisfy `Vᴴ B V = I` to working accuracy.
///
/// Compute it with [`Self::decompose`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GeneralizedHermitianEigenDecomposition<S: NumericScalar> {
    pub(super) report: Option<super::EigenReport>,
    pub(super) eigenvalues: Vec<S::Real>,
    pub(super) eigenvectors: Matrix<S>,
    pub(super) residuals: Vec<S::Real>,
    pub(super) projections: [Option<HermitianProjectionReport<S::Real>>; 2],
}

/// Eigenvalues of a Hermitian-definite pair, with the iteration report.
///
/// Compute it with [`Self::compute`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GeneralizedHermitianEigenvaluesResult<R: RealScalar> {
    pub(super) output: GeneralizedHermitianEigenvaluesBuffer<R>,
    pub(super) report: super::EigenReport,
}

impl<R: RealScalar> Default for GeneralizedHermitianEigenvaluesBuffer<R> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<R: RealScalar> GeneralizedHermitianEigenvaluesBuffer<R> {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            eigenvalues: Vec::new(),
            projections: [None; 2],
        }
    }
    /// Returns the change made by projecting `A`, or `None` unless `A` was
    /// [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn a_projection_report(&self) -> Option<&HermitianProjectionReport<R>> {
        self.projections[0].as_ref()
    }
    /// Returns the change made by projecting `B`, or `None` unless `B` was
    /// [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn b_projection_report(&self) -> Option<&HermitianProjectionReport<R>> {
        self.projections[1].as_ref()
    }
    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[R] {
        &self.eigenvalues
    }
    /// Returns the workspace requirements for eigenvalues of a pair with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the options select an index range.
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        request: &GeneralizedHermitianEigenWorkspaceRequest<R>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        generalized_requirements::<S>(request, Job::Values)
    }
    /// Allocates a workspace for eigenvalues of a pair with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace<S: NumericScalar<Real = R>>(
        request: &GeneralizedHermitianEigenWorkspaceRequest<R>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}

impl<S: NumericScalar> Default for GeneralizedHermitianEigenDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> GeneralizedHermitianEigenDecomposition<S> {
    /// Creates an empty result for use with [`Self::decompose_into`].
    pub const fn empty() -> Self {
        Self {
            report: None,
            eigenvalues: Vec::new(),
            projections: [None; 2],
            eigenvectors: Matrix::empty(),
            residuals: Vec::new(),
        }
    }
    /// Returns the report of the last successful computation, or `None` before one.
    pub const fn report(&self) -> Option<super::EigenReport> {
        self.report
    }
    /// Returns the change made by projecting `A`, or `None` unless `A` was
    /// [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn a_projection_report(&self) -> Option<&HermitianProjectionReport<S::Real>> {
        self.projections[0].as_ref()
    }
    /// Returns the change made by projecting `B`, or `None` unless `B` was
    /// [`HermitianInput::Projected`](crate::HermitianInput::Projected).
    pub const fn b_projection_report(&self) -> Option<&HermitianProjectionReport<S::Real>> {
        self.projections[1].as_ref()
    }
    /// Returns the eigenvalues in the requested order.
    pub fn eigenvalues(&self) -> &[S::Real] {
        &self.eigenvalues
    }
    /// Returns the eigenvector matrix `V`, with `Vᴴ B V = I` to working accuracy.
    pub const fn eigenvectors(&self) -> &Matrix<S> {
        &self.eigenvectors
    }
    /// Returns the scaled residual of each eigenpair.
    ///
    /// For eigenpair `(λ, v)` the residual is
    ///
    /// ```text
    /// ‖A v − λ B v‖∞ / (n (‖A‖max + |λ| ‖B‖max) ‖v‖max),
    /// ```
    ///
    /// where `‖·‖max` is the largest entry magnitude and `n` the order. It is
    /// computed for `A` and `B` as defined by the input interpretations and
    /// does not change when `A` or `B` is multiplied by a constant.
    pub fn residuals(&self) -> &[S::Real] {
        &self.residuals
    }
    /// Returns the workspace requirements for eigenvalues and eigenvectors.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the options select an index range.
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &GeneralizedHermitianEigenWorkspaceRequest<S::Real>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        generalized_requirements::<S>(request, Job::Vectors)
    }
    /// Allocates a workspace for eigenvalues and eigenvectors.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot be reserved.
    pub fn workspace(
        request: &GeneralizedHermitianEigenWorkspaceRequest<S::Real>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

impl<R: RealScalar> GeneralizedHermitianEigenvaluesResult<R> {
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
    pub fn into_parts(self) -> (GeneralizedHermitianEigenvaluesBuffer<R>, super::EigenReport) {
        (self.output, self.report)
    }
    /// Returns the change made by projecting `A`, or `None` if `A` was not projected.
    pub const fn a_projection_report(&self) -> Option<&crate::HermitianProjectionReport<R>> {
        self.output.a_projection_report()
    }
    /// Returns the change made by projecting `B`, or `None` if `B` was not projected.
    pub const fn b_projection_report(&self) -> Option<&crate::HermitianProjectionReport<R>> {
        self.output.b_projection_report()
    }
}

/// Workspace request for a Hermitian-definite eigenproblem of a given order and options.
///
/// Pass it to `workspace_requirements` or `workspace` of
/// [`GeneralizedHermitianEigenvaluesBuffer`] or
/// [`GeneralizedHermitianEigenDecomposition`]. It cannot be used for a
/// standard Hermitian problem:
///
/// ```compile_fail
/// use scioxide_linalg::{
///     GeneralizedHermitianEigenWorkspaceRequest, HermitianEigenOptions, HermitianEigenvaluesBuffer,
/// };
/// let request =
///     GeneralizedHermitianEigenWorkspaceRequest::new(3, &HermitianEigenOptions::<f64>::default());
/// HermitianEigenvaluesBuffer::<f64>::workspace::<f64>(&request).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct GeneralizedHermitianEigenWorkspaceRequest<R: RealScalar> {
    pub(super) dense: HermitianEigenWorkspaceRequest<R>,
}
impl<R: RealScalar> GeneralizedHermitianEigenWorkspaceRequest<R> {
    /// Creates a request for a pair of `order × order` matrices.
    pub const fn new(order: usize, options: &HermitianEigenOptions<R>) -> Self {
        Self {
            dense: HermitianEigenWorkspaceRequest::new(order, options),
        }
    }
    /// Limits the number of parallel workers the workspace is sized for.
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.dense = self.dense.with_worker_bound(workers);
        self
    }
    /// Includes packing storage for an injected BLAS provider.
    pub const fn with_provider_packing(mut self) -> Self {
        self.dense = self.dense.with_provider_packing();
        self
    }
    /// Sets the worker bound and provider packing from `context`.
    pub fn for_context(self, context: &crate::LinalgExecutionContext<'_, '_>) -> Self {
        self.for_kernels(context.kernel())
    }
    pub(crate) fn for_kernels(mut self, context: KernelContext<'_>) -> Self {
        self.dense = self.dense.for_kernels(context);
        self
    }
}
