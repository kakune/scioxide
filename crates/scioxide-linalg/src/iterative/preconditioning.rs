//! Preconditioners passed to an iterative solve.

use scioxide_core::{AdjointOperator, Dimensions, NumericScalar, Preconditioner, SciOxideError};

/// Preconditioner `M⁻¹` for one iterative solve.
///
/// The value is passed to each solve call, so one solver can be used with
/// different preconditioners. The provider applies the inverse `M⁻¹`, not `M`.
/// Each solver applies it on its natural side: CG and MINRES precondition
/// symmetrically, Richardson from the left, and the nonsymmetric Krylov solvers
/// and the block solvers from the right. LSQR, LSMR and the Jacobi, Gauss–Seidel
/// and SOR solvers take no preconditioner.
///
/// CG and MINRES require `M⁻¹` to be Hermitian positive definite; a violation
/// that the solver detects stops it with
/// [`Breakdown(Preconditioner)`](crate::LinearSolveBreakdown::Preconditioner).
/// Preconditioned MINRES converges only on the true residual, because its
/// least-squares objective uses the norm induced by `M⁻¹`. GMRES, LGMRES, GCROT
/// and their block forms store each preconditioned direction and therefore
/// accept an `M⁻¹` that changes between applications (flexible
/// preconditioning). The other solvers require a fixed linear `M⁻¹` during a solve.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ResidualTolerance};
/// use scioxide_linalg::{CgSolver, Matrix, Preconditioning, Vector};
/// use scioxide_sparse::IdentityPreconditioner;
/// let a = Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 3.0])?;
/// let inverse = IdentityPreconditioner::new(2);
/// let solver = CgSolver::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(10)?);
/// let b = Vector::from_vec(vec![1.0, 2.0]);
/// let result = solver.solve_with_options(
///     &a.view().try_finite()?,
///     &b,
///     Preconditioning::Inverse(&inverse),
///     None,
/// )?;
/// assert!(result.is_success());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub enum Preconditioning<'a, S: NumericScalar> {
    /// No preconditioning, `M⁻¹ = I`.
    None,
    /// The action of `M⁻¹`.
    Inverse(&'a dyn Preconditioner<S>),
    /// The action of `M⁻¹` together with its adjoint `M⁻ᴴ`.
    ///
    /// [`BicgSolver`](crate::BicgSolver) and [`QmrSolver`](crate::QmrSolver) require
    /// this form when preconditioned. The other solvers use only the forward action.
    InverseAdjoint(&'a dyn AdjointOperator<S>),
}

impl<S: NumericScalar> core::fmt::Debug for Preconditioning<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::None => "None",
            Self::Inverse(_) => "Inverse",
            Self::InverseAdjoint(_) => "InverseAdjoint",
        })
    }
}

impl<S: NumericScalar> Preconditioning<'_, S> {
    pub(crate) fn validate_dimension(self, n: usize) -> Result<(), SciOxideError> {
        let (rows, cols) = match self {
            Self::None => (n, n),
            Self::Inverse(p) => (p.dimension(), p.dimension()),
            Self::InverseAdjoint(p) => (p.rows(), p.cols()),
        };
        if rows == n && cols == n {
            Ok(())
        } else {
            Err(SciOxideError::dimension_mismatch(
                Dimensions::two(n, n),
                Dimensions::two(rows, cols),
            ))
        }
    }

    pub(crate) fn workspace_len(self) -> Result<usize, SciOxideError> {
        match self {
            Self::None => Ok(0),
            Self::Inverse(p) => p.precondition_workspace_len(),
            Self::InverseAdjoint(p) => Ok(p
                .apply_workspace_len()?
                .max(p.apply_adjoint_workspace_len()?)),
        }
    }

    pub(super) fn adjoint(self, n: usize) -> Result<(), SciOxideError> {
        match self {
            Self::None | Self::InverseAdjoint(_) => self.validate_dimension(n),
            Self::Inverse(_) => Err(SciOxideError::invalid_input(
                scioxide_core::argument!("preconditioner"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "adjoint inverse capability required",
            )),
        }
    }
}
