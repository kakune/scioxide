//! Independent numerical LP/QP certificate verification in original coordinates.
//!
//! Verification checks supplied certificates; it never solves for replacement multipliers.
//! Accepted reports are floating-point, tolerance-based checks, not interval proofs.
pub(crate) mod candidate;
pub(crate) mod kernel;
mod problem;
use alloc::vec::Vec;
pub use problem::{LpVerificationProblem, QpVerificationProblem};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{
    HermitianEigenOptions, HermitianEigenWorkspaceRequest, HermitianEigenvaluesBuffer,
    LinalgWorkspace, Matrix,
};
mod verifier;
pub use verifier::{
    ProgrammingClaim, ProgrammingVerifier, VerificationOptions, VerificationProblem,
};

/// Borrows the multipliers of an LP or QP in original coordinates.
///
/// Multipliers of equality rows are free, those of `≤` rows are nonnegative and
/// those of `≥` rows are nonpositive. Lower- and upper-bound multipliers are
/// nonnegative and must be zero where the bound is absent.
#[derive(Debug, Clone, Copy)]
pub struct DualCertificate<'a, S> {
    /// One multiplier per constraint row.
    pub rows: &'a [S],
    /// One lower-bound multiplier per variable.
    pub lower: &'a [S],
    /// One upper-bound multiplier per variable.
    pub upper: &'a [S],
}
/// Borrows a certificate for [`ProgrammingVerifier`].
#[derive(Debug, Clone, Copy)]
pub enum ProgrammingCertificate<'a, S> {
    /// Optimality: a primal point and its multipliers.
    Optimality {
        /// Primal point.
        point: &'a [S],
        /// Multipliers.
        dual: DualCertificate<'a, S>,
    },
    /// Infeasibility: Farkas multipliers, rescaled by a power of two before the check.
    Infeasibility(DualCertificate<'a, S>),
    /// Unboundedness: a feasible point and a nonzero recession direction, rescaled before the check.
    Unboundedness {
        /// Feasible point.
        point: &'a [S],
        /// Recession direction.
        direction: &'a [S],
    },
}
/// Gives the outcome of a [`ProgrammingVerifier`] check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Every check the certificate requires passed.
    Accepted,
    /// At least one numerical check failed.
    Rejected,
    /// The KKT conditions hold, but convexity of the QP could not be checked, so
    /// the point is not known to be a global minimizer.
    KktOnly,
    /// A limit, cancellation or deadline stopped the check before it finished.
    Incomplete,
}
/// Holds the residuals and outcome of a [`ProgrammingVerifier`] check.
///
/// The residuals are computed in floating point in original units; they are not
/// rigorous enclosures.
#[derive(Debug, Clone, Copy)]
pub struct VerificationReport<S: RealScalar> {
    pub(crate) status: VerificationStatus,
    pub(crate) primal_violation: S,
    pub(crate) bound_violation: S,
    pub(crate) dual_violation: S,
    pub(crate) stationarity: S,
    pub(crate) complementarity: S,
    pub(crate) primal_objective: Option<S>,
    pub(crate) objective_gap: Option<S>,
    pub(crate) dual_bound: Option<S>,
    pub(crate) witness_margin: Option<S>,
    pub(crate) convexity_checked: bool,
    pub(crate) termination: Option<crate::OptimizeTermination>,
}
impl<S: RealScalar> Default for VerificationReport<S> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<S: RealScalar> VerificationReport<S> {
    /// Creates a `Rejected` report, for use as `output` of [`ProgrammingVerifier::verify_into`].
    pub const fn empty() -> Self {
        Self {
            status: VerificationStatus::Rejected,
            primal_violation: S::ZERO,
            bound_violation: S::ZERO,
            dual_violation: S::ZERO,
            stationarity: S::ZERO,
            complementarity: S::ZERO,
            primal_objective: None,
            objective_gap: None,
            dual_bound: None,
            witness_margin: None,
            convexity_checked: false,
            termination: None,
        }
    }
    /// Returns whether [`Self::status`] is [`VerificationStatus::Accepted`].
    pub const fn is_accepted(&self) -> bool {
        matches!(self.status, VerificationStatus::Accepted)
    }
    /// Returns the outcome of the check.
    pub const fn status(&self) -> VerificationStatus {
        self.status
    }
    /// Returns the largest constraint-row violation of the supplied point.
    pub const fn primal_violation(&self) -> S {
        self.primal_violation
    }
    /// Returns the largest variable-bound violation of the supplied point.
    pub const fn bound_violation(&self) -> S {
        self.bound_violation
    }
    /// Returns the largest multiplier sign violation.
    pub const fn dual_violation(&self) -> S {
        self.dual_violation
    }
    /// Returns the largest component of the Lagrangian gradient, or of the Farkas combination `Aᵀy`.
    pub const fn stationarity(&self) -> S {
        self.stationarity
    }
    /// Returns the largest product of a multiplier and its slack, for an optimality certificate.
    pub const fn complementarity(&self) -> S {
        self.complementarity
    }
    /// Returns the objective recomputed at the supplied point, if a point was supplied.
    pub const fn primal_objective(&self) -> Option<S> {
        self.primal_objective
    }
    /// Returns the difference between the primal objective and the dual objective, if computed.
    pub const fn objective_gap(&self) -> Option<S> {
        self.objective_gap
    }
    /// Returns a lower bound on the optimal objective from the multipliers, if the
    /// signs, convexity and residuals allow one.
    pub const fn dual_bound(&self) -> Option<S> {
        self.dual_bound
    }
    /// Returns the margin of an infeasibility or unboundedness certificate: the
    /// rescaled Farkas contradiction or the objective descent along the direction.
    pub const fn witness_margin(&self) -> Option<S> {
        self.witness_margin
    }
    /// Returns whether convexity was checked; always true for an LP.
    pub const fn convexity_checked(&self) -> bool {
        self.convexity_checked
    }
    /// Returns the reason the check stopped, for an `Incomplete` report.
    pub const fn termination(&self) -> Option<crate::OptimizeTermination> {
        self.termination
    }
    /// Original-scale residual summary; no rigorous optimality gap is inferred.
    pub(crate) fn diagnostics(&self) -> crate::OptimalityDiagnostics<S> {
        crate::OptimalityDiagnostics {
            primal_feasibility: Some(self.primal_violation),
            bound_violation: Some(self.bound_violation),
            dual_feasibility: Some(self.dual_violation),
            stationarity: Some(self.stationarity),
            stationarity_norm: scioxide_core::NormSelection::Infinity,
            complementarity: Some(self.complementarity),
            ..crate::OptimalityDiagnostics::unavailable()
        }
    }
}

/// Holds the reusable storage of [`ProgrammingVerifier`].
///
/// [`ProgrammingVerifier::workspace`] sizes every buffer for one problem,
/// including the dense Cholesky and eigenvalue storage for the convexity check
/// of an explicit QP Hessian. Checks with that workspace, including the first,
/// do not allocate.
pub struct VerificationWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    pub(super) shell: crate::attempt::Shell<()>,
    pub(super) rows: Vec<S>,
    pub(super) row_dual: Vec<S>,
    pub(super) gradient: Vec<S>,
    pub(super) transpose: Vec<S>,
    pub(super) direction: Vec<S>,
    pub(super) lower: Vec<S>,
    pub(super) upper: Vec<S>,
    pub(super) scratch: Vec<S>,
    pub(super) matrix: Option<Matrix<S>>,
    pub(super) cholesky: scioxide_linalg::CholeskyDecomposition<S>,
    pub(super) cholesky_workspace: LinalgWorkspace<'static, S>,
    pub(super) eigenvalues: HermitianEigenvaluesBuffer<S>,
    pub(super) eigen_workspace: LinalgWorkspace<'static, S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> VerificationWorkspace<S> {
    /// Returns the record of the most recent check.
    pub const fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.shell.last_attempt()
    }
    /// Allocate exact LP product-buffer lengths.
    /// # Errors
    /// Returns size, allocation, or operator workspace-query errors.
    pub(crate) fn for_lp(problem: &LpVerificationProblem<'_, S>) -> Result<Self, SciOxideError> {
        Self::prepare(problem, None)
    }
    /// Allocate QP buffers, including dense convexity scratch for explicit coefficients.
    /// # Errors
    /// Returns size, allocation, or operator workspace-query errors.
    pub(crate) fn for_qp(problem: &QpVerificationProblem<'_, S>) -> Result<Self, SciOxideError> {
        Self::prepare(problem.linear(), Some(problem))
    }
    fn prepare(
        p: &LpVerificationProblem<'_, S>,
        q: Option<&QpVerificationProblem<'_, S>>,
    ) -> Result<Self, SciOxideError> {
        use scioxide_core::backend::try_filled;
        let mut scratch = 0;
        for op in [p.operator, q.map(|q| q.hessian())].into_iter().flatten() {
            scratch = scratch
                .max(op.apply_workspace_len()?)
                .max(op.apply_adjoint_workspace_len()?);
        }
        let n = p.dimension();
        let m = p.constraints_len();
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let mut eigenvalues = HermitianEigenvaluesBuffer::empty();
        let mut eigen_workspace = LinalgWorkspace::new();
        let mut cholesky = scioxide_linalg::CholeskyDecomposition::empty();
        let mut cholesky_workspace = LinalgWorkspace::new();
        let matrix = if q.is_some_and(|q| q.inspectable()) {
            eigen_workspace = HermitianEigenvaluesBuffer::<S>::workspace::<S>(
                &HermitianEigenWorkspaceRequest::new(n, &HermitianEigenOptions::default()),
            )?;
            // Reserve dense fallback storage, but initialize the eigenvalue
            // output through a diagonal job instead of reducing a dense identity.
            let diagonal = try_filled(
                n,
                <S as scioxide_core::AlgebraicScalar>::ONE,
                scioxide_core::quantity!("verification_eigen_diagonal"),
            )?;
            let off_diagonal = try_filled(
                n.saturating_sub(1),
                zero,
                scioxide_core::quantity!("verification_eigen_off_diagonal"),
            )?;
            HermitianEigenvaluesBuffer::tridiagonal_into(
                &diagonal,
                &off_diagonal,
                &HermitianEigenOptions::default(),
                &mut eigenvalues,
                &mut eigen_workspace,
                scioxide_core::ExecutionContext::sequential(),
            )?;
            cholesky.try_reserve(n)?;
            cholesky_workspace.try_reserve(
                &scioxide_linalg::CholeskyDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::CholeskyWorkspaceRequest::factor(
                        n,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
            let matrix = Matrix::zeros(n, n)?;
            Some(matrix)
        } else {
            None
        };
        Ok(Self {
            cholesky,
            cholesky_workspace,
            shell: crate::attempt::Shell::new(),
            rows: try_filled(m, zero, scioxide_core::quantity!("verification_rows"))?,
            row_dual: try_filled(m, zero, scioxide_core::quantity!("verification_dual_rows"))?,
            gradient: try_filled(n, zero, scioxide_core::quantity!("verification_gradient"))?,
            transpose: try_filled(n, zero, scioxide_core::quantity!("verification_transpose"))?,
            direction: try_filled(n, zero, scioxide_core::quantity!("verification_direction"))?,
            lower: try_filled(
                n,
                zero,
                scioxide_core::quantity!("verification_lower_multipliers"),
            )?,
            upper: try_filled(
                n,
                zero,
                scioxide_core::quantity!("verification_upper_multipliers"),
            )?,
            scratch: try_filled(
                scratch,
                zero,
                scioxide_core::quantity!("verification_operator_scratch"),
            )?,
            matrix,
            eigenvalues,
            eigen_workspace,
        })
    }
}
pub(super) use crate::validation::finite_input as finite;
pub(super) use crate::validation::invalid;
pub(super) use crate::validation::shape;

/// Owns the multipliers of an LP or QP certificate.
#[derive(Debug, Clone)]
pub struct OwnedDualCertificate<S> {
    pub(crate) rows: Vec<S>,
    pub(crate) lower: Vec<S>,
    pub(crate) upper: Vec<S>,
}
impl<S> OwnedDualCertificate<S> {
    /// Creates a certificate from row, lower-bound and upper-bound multipliers.
    ///
    /// Lengths, finiteness and signs are checked when the certificate is verified.
    pub fn new(rows: Vec<S>, lower: Vec<S>, upper: Vec<S>) -> Self {
        Self { rows, lower, upper }
    }
    /// Borrows the multipliers for [`ProgrammingVerifier`].
    pub fn as_certificate(&self) -> DualCertificate<'_, S> {
        DualCertificate {
            rows: &self.rows,
            lower: &self.lower,
            upper: &self.upper,
        }
    }
}
impl<S: Copy> OwnedDualCertificate<S> {
    /// Fallibly copy every multiplier without changing the source.
    pub(crate) fn try_clone(&self) -> Result<Self, SciOxideError> {
        use scioxide_core::backend::try_copy_slice as copy;
        Ok(Self {
            rows: copy(&self.rows, scioxide_core::quantity!("certificate_rows"))?,
            lower: copy(&self.lower, scioxide_core::quantity!("certificate_lower"))?,
            upper: copy(&self.upper, scioxide_core::quantity!("certificate_upper"))?,
        })
    }
}
/// Owns an LP or QP certificate without its point.
///
/// [`ProgrammingResult`](crate::ProgrammingResult) stores certificates in this
/// form, and [`ProgrammingResult::from_candidate`](crate::ProgrammingResult::from_candidate)
/// accepts one from an external solver. [`Self::borrow`] pairs it with its point
/// for [`ProgrammingVerifier`].
#[derive(Debug, Clone)]
pub enum OwnedProgrammingCertificate<S> {
    /// Multipliers of an optimum; the point is supplied separately.
    Optimality(OwnedDualCertificate<S>),
    /// Farkas multipliers proving infeasibility.
    Infeasibility(OwnedDualCertificate<S>),
    /// Recession direction proving unboundedness; the feasible point is supplied separately.
    Unboundedness(Vec<S>),
}
impl<S> OwnedProgrammingCertificate<S> {
    /// Borrows the certificate together with `point`.
    pub fn borrow<'a>(&'a self, point: &'a [S]) -> ProgrammingCertificate<'a, S> {
        match self {
            Self::Optimality(d) => ProgrammingCertificate::Optimality {
                point,
                dual: d.as_certificate(),
            },
            Self::Infeasibility(d) => ProgrammingCertificate::Infeasibility(d.as_certificate()),
            Self::Unboundedness(direction) => {
                ProgrammingCertificate::Unboundedness { point, direction }
            }
        }
    }
}
impl<S: Copy> OwnedProgrammingCertificate<S> {
    /// Fallibly copy the certificate without changing the source.
    pub(crate) fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(match self {
            Self::Optimality(dual) => Self::Optimality(dual.try_clone()?),
            Self::Infeasibility(dual) => Self::Infeasibility(dual.try_clone()?),
            Self::Unboundedness(direction) => {
                Self::Unboundedness(scioxide_core::backend::try_copy_slice(
                    direction,
                    scioxide_core::quantity!("certificate_direction"),
                )?)
            }
        })
    }
}
