//! Generalized eigenvalues, results and workspace requests of the QZ decomposition.

use super::qz::{decompose_lengths, round_complex};
use super::support::direct_requirements;
use super::{SchurBlock, SchurOptions};
use crate::ConditionEstimate;
use crate::{Estimation, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix};
use alloc::vec::Vec;
use scioxide_core::{
    Complex, NumericScalar, RealScalar, ReliableEstimate, ScaledValue, SciOxideError,
};

/// Generalized eigenvalue `λ = α / β` of a pencil `A − λ B`, in homogeneous form `(α, β)`.
///
/// `β` is real and nonnegative, and `β = 0` is an infinite eigenvalue, stored
/// as `(1, 0)`. A finite eigenvalue is stored as the diagonal pair of the
/// generalized Schur form, rotated so that `β > 0`. If that pair is not
/// representable, `α` and `β` are multiplied by a common power of two. The
/// quotient `α / β` can overflow or underflow even when `(α, β)` is
/// representable, so it is computed only on request by [`Self::ratio`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedEigenvalue<R: RealScalar> {
    pub(super) alpha: Complex<R>,
    pub(super) beta: R,
}
impl<R: RealScalar> GeneralizedEigenvalue<R> {
    /// Returns the numerator `α`.
    pub const fn alpha(self) -> Complex<R> {
        self.alpha
    }
    /// Returns the denominator `β ≥ 0`; zero means an infinite eigenvalue.
    pub const fn beta(self) -> R {
        self.beta
    }
    /// Returns `true` if `β = 0`.
    pub fn is_infinite(self) -> bool {
        self.beta == R::ZERO
    }
    /// Returns `true` if `β > 0`, even when `α / β` overflows.
    pub fn is_finite(self) -> bool {
        !self.is_infinite()
    }
    /// Returns the eigenvalue `λ = α / β` as a complex number.
    ///
    /// # Errors
    ///
    /// - `DomainError` if the eigenvalue is infinite (`β = 0`).
    /// - `NumericalInstability` if the quotient overflows, or if a nonzero
    ///   component underflows to zero.
    pub fn ratio(self) -> Result<Complex<R>, SciOxideError> {
        round_complex(
            ScaledValue::from_value(self.alpha)?.div(ScaledValue::from_value(
                scioxide_core::Complex::new(self.beta, <_ as scioxide_core::AlgebraicScalar>::ZERO),
            )?)?,
        )
    }
    /// Rebuild a value stored by `qz_values` as `(alpha.re, alpha.im, beta)`.
    pub(crate) fn from_parts(parts: &[R]) -> Self {
        Self {
            alpha: Complex::new(parts[0], parts[1]),
            beta: parts[2],
        }
    }
    pub(super) const fn zero() -> Self {
        Self {
            alpha: Complex::new(R::ZERO, R::ZERO),
            beta: R::ONE,
        }
    }
}

/// Iteration counts of a successful generalized Schur decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneralizedSchurReport {
    pub(super) counts: scioxide_core::SolveCounts,
    pub(super) infinite: usize,
}
impl GeneralizedSchurReport {
    /// Returns the counts of this decomposition and of its nested solves.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of QZ sweeps, including the chases that deflate an
    /// infinite eigenvalue.
    pub const fn iterations(self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns the number of infinite eigenvalues: the diagonal entries of `T`
    /// that are exactly zero.
    ///
    /// A nonzero diagonal entry is never counted, even if it is tiny.
    pub const fn infinite_eigenvalues(self) -> usize {
        self.infinite
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PencilJob {
    Decompose,
    Reorder(Estimation),
}

/// Workspace request for a generalized Schur decomposition or reordering of a given order.
///
/// [`Self::new`] covers [`GeneralizedSchurDecomposition::decompose_into`] and
/// [`Self::reorder`] covers [`GeneralizedSchurDecomposition::reorder_in_place`].
/// Combine the requirements of both with [`LinalgRequirements::max`] to size
/// one workspace for both.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedSchurWorkspaceRequest {
    pub(super) order: usize,
    pub(super) job: PencilJob,
    pub(super) execution: LinalgExecutionOptions,
}
impl GeneralizedSchurWorkspaceRequest {
    /// Creates a request for the decomposition of an `order × order` pencil.
    pub const fn new<R: RealScalar>(order: usize, options: &SchurOptions<R>) -> Self {
        Self {
            order,
            job: PencilJob::Decompose,
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
            job: PencilJob::Reorder(options.condition_estimates()),
            execution: options.execution_options(),
        }
    }
}

/// Generalized Schur (QZ) decomposition `A = Q S Zᴴ`, `B = Q T Zᴴ` of a square pencil `A − λ B`.
///
/// `Q` and `Z` are unitary, or orthogonal for real input, and `T` is upper
/// triangular. For complex input `S` is upper triangular. For real input `S`
/// is upper quasi-triangular, with a 2×2 diagonal block for each complex
/// conjugate pair of eigenvalues. A 1×1 block gives the eigenvalue
/// `λⱼ = Sⱼⱼ / Tⱼⱼ`. All eigenvalues are returned in homogeneous form
/// `(α, β)` (see [`GeneralizedEigenvalue`]), so an infinite eigenvalue
/// (`Tⱼⱼ = 0`) needs no special case.
///
/// Compute it with [`Self::decompose`].
#[derive(Debug, Clone)]
pub struct GeneralizedSchurDecomposition<S: NumericScalar> {
    pub(super) s: Matrix<S>,
    pub(super) t: Matrix<S>,
    pub(super) q: Matrix<S>,
    pub(super) z: Matrix<S>,
    pub(super) values: Vec<GeneralizedEigenvalue<S::Real>>,
    pub(super) blocks: Vec<SchurBlock>,
    pub(super) report: Option<GeneralizedSchurReport>,
}

impl<S: NumericScalar> Default for GeneralizedSchurDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> GeneralizedSchurDecomposition<S> {
    /// Creates an empty decomposition for use with [`Self::decompose_into`].
    pub const fn empty() -> Self {
        Self {
            s: Matrix::empty(),
            t: Matrix::empty(),
            q: Matrix::empty(),
            z: Matrix::empty(),
            values: Vec::new(),
            blocks: Vec::new(),
            report: None,
        }
    }
    /// Returns `S = Qᴴ A Z`, in the units of `A`.
    pub const fn s(&self) -> &Matrix<S> {
        &self.s
    }
    /// Returns `T = Qᴴ B Z`, in the units of `B`.
    pub const fn t(&self) -> &Matrix<S> {
        &self.t
    }
    /// Returns the left transformation `Q`.
    pub const fn q(&self) -> &Matrix<S> {
        &self.q
    }
    /// Returns the right transformation `Z`.
    pub const fn z(&self) -> &Matrix<S> {
        &self.z
    }
    /// Returns the eigenvalues in block order, with the positive imaginary
    /// part first in each real 2×2 block.
    ///
    /// For real input, the two eigenvalues of a 2×2 block have exactly
    /// conjugate numerators and equal denominators, also after reordering.
    pub fn eigenvalues(&self) -> &[GeneralizedEigenvalue<S::Real>] {
        &self.values
    }
    /// Returns the diagonal blocks of `S` in order.
    pub fn blocks(&self) -> &[SchurBlock] {
        &self.blocks
    }
    /// Returns the report of the last successful decomposition, or `None` before one.
    pub const fn report(&self) -> Option<GeneralizedSchurReport> {
        self.report
    }
    /// Returns the workspace requirements of a decomposition or reordering.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the scratch size overflows `usize`.
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &GeneralizedSchurWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let lengths = match request.job {
            PencilJob::Decompose => {
                decompose_lengths::<S>(request.order, request.execution.execution_policy())?
            }
            PencilJob::Reorder(estimates) => {
                super::qz_reorder::reorder_lengths(request.order, estimates)?
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
        request: &GeneralizedSchurWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

/// Diagnostics of a generalized Schur reordering.
///
/// After reordering, `S = [[S₁₁, S₁₂], [0, S₂₂]]` and
/// `T = [[T₁₁, T₁₂], [0, T₂₂]]`, where `(S₁₁, T₁₁)` holds the selected
/// eigenvalues. The condition numbers are estimates, not bounds.
///
/// # References
///
/// B. Kågström and P. Poromaa, "Computing eigenspaces with specified
/// eigenvalues of a regular matrix pair (A, B) and condition estimation:
/// theory, algorithms and software", *Numer. Algorithms* 12, 1996.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedSchurReorderReport<R: RealScalar> {
    pub(super) selected: usize,
    pub(super) swaps: usize,
    pub(super) iterations: usize,
    pub(super) left: Option<ConditionEstimate<R>>,
    pub(super) right: Option<ConditionEstimate<R>>,
    pub(super) separation: Option<[ReliableEstimate<R>; 2]>,
}
impl<R: RealScalar> GeneralizedSchurReorderReport<R> {
    /// Returns the order `m` of `S₁₁`, the number of selected eigenvalues.
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
    /// Returns the reciprocal norm of the spectral projector onto the left
    /// deflating subspace, in `[0, 1]`, or `None` without [`Estimation::Estimate`].
    pub const fn left_condition(self) -> Option<ConditionEstimate<R>> {
        self.left
    }
    /// Returns the reciprocal norm of the spectral projector onto the right
    /// deflating subspace, in `[0, 1]`, or `None` without [`Estimation::Estimate`].
    pub const fn right_condition(self) -> Option<ConditionEstimate<R>> {
        self.right
    }
    /// Returns estimates of the separations of the two diagonal blocks, in the
    /// units of the pencil, or `None` without [`Estimation::Estimate`].
    ///
    /// Each entry is `1 / ‖K⁻¹‖₁`, estimated with a 1-norm estimator, for the
    /// coupled Sylvester operator `K` of the pencil and for the operator with
    /// the roles of the two blocks reversed. Small values mean ill-conditioned
    /// deflating subspaces.
    pub const fn separation(self) -> Option<[ReliableEstimate<R>; 2]> {
        self.separation
    }
}
