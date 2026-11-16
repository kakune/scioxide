//! Workspaces of the bounded linear least-squares solvers. Only preparation allocates.
use super::*;

use scioxide_core::backend::try_filled;
use scioxide_linalg::{
    HouseholderDecomposition, LinalgWorkspace, Matrix, ThinSvdDecomposition, Vector,
};
macro_rules! linear_workspace {
    ($(#[$meta:meta])* $workspace:ident) => {
        $(#[$meta])*
        pub struct $workspace<S: RealScalar + NumericScalar<Real = S>> {
            pub(super) storage: Storage<S>,
            pub(super) shell: crate::attempt::Shell<crate::MinimizerProgress<S>>,
        }
        impl<S: RealScalar + NumericScalar<Real = S>> $workspace<S> {
            /// Returns the record of the last call.
            pub const fn last_attempt(
                &self,
            ) -> Option<&scioxide_core::AttemptRecord<crate::MinimizerProgress<S>>> {
                self.shell.last_attempt()
            }
            /// Returns the accepted feasible point, its gradient and progress from the
            /// last call, if it accepted any.
            pub fn last_accepted(&self) -> Option<crate::MinimizerAccepted<'_, S>> {
                crate::attempt::last_accepted(&self.shell, &self.storage)
            }
        }
    };
}
linear_workspace!(
    /// Holds the active set, QR factors and operator scratch of an [`Nnls`]
    /// solve, and a record of the last call.
    NnlsWorkspace
);
linear_workspace!(
    /// Holds the active set, QR factors and operator scratch of a
    /// [`BoundedLinearLeastSquares`] solve, and a record of the last call.
    BoundedLinearLeastSquaresWorkspace
);
pub(super) struct Dense<S: RealScalar + NumericScalar<Real = S>> {
    pub active: super::active_qr::ActiveQr<S>,
    pub rank: scioxide_linalg::TriangularRankWorkspace<S>,
    pub matrix: Matrix<S>,
    pub rhs: Vector<S>,
    pub solution: Vector<S>,
    pub qr: HouseholderDecomposition<S>,
    pub qw: LinalgWorkspace<'static, S>,
    pub svd: ThinSvdDecomposition<S>,
    pub sw: LinalgWorkspace<'static, S>,
}
pub(super) struct Storage<S: RealScalar + NumericScalar<Real = S>> {
    pub accepted: LinearLeastSquaresResult<S>,
    pub trial: LinearLeastSquaresResult<S>,
    pub passive: Vec<bool>,
    pub blocked: Vec<bool>,
    pub step: Vec<S>,
    pub candidate: Vec<S>,
    pub provider: Vec<S>,
    pub map: Vec<usize>,
    pub dense: Option<Dense<S>>,
    pub lsmr: Option<scioxide_linalg::IterativeWorkspace<S>>,
    pub rhs: Vector<S>,
    pub solution: Vector<S>,
    pub rank: Option<RankReport<S>>,
    pub nnls: bool,
    pub observed: bool,
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::attempt::Accepted<S> for Storage<S> {
    fn accepted(&self) -> Option<crate::attempt::AcceptedState<'_, S>> {
        self.observed.then_some(crate::attempt::AcceptedState {
            point: &self.accepted.x,
            value: self.accepted.objective,
            gradient: Some(&self.accepted.gradient),
            iterations: self.accepted.iterations,
        })
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Storage<S> {
    pub(super) fn new(
        p: &LinearLeastSquaresProblem<'_, S>,
        o: &LinearLeastSquaresOptions<S>,
        nnls: bool,
    ) -> Result<Self, SciOxideError> {
        let (m, n) = (p.rows(), p.cols());
        let rows = m;
        let provider = p
            .operator()
            .apply_workspace_len()?
            .max(p.operator().apply_adjoint_workspace_len()?);
        let mut w = Self {
            accepted: LinearLeastSquaresResult::sized(m, n)?,
            trial: LinearLeastSquaresResult::sized(m, n)?,
            passive: try_filled(n, false, scioxide_core::quantity!("workspace.passive"))?,
            blocked: try_filled(n, false, scioxide_core::quantity!("workspace.blocked"))?,
            step: try_filled(
                n,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("workspace.step"),
            )?,
            candidate: try_filled(
                n,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("workspace.candidate"),
            )?,
            provider: try_filled(
                provider,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("workspace.provider"),
            )?,
            map: try_filled(n, 0, scioxide_core::quantity!("workspace.map"))?,
            dense: None,
            lsmr: None,
            rhs: Vector::zeros(rows)?,
            solution: Vector::zeros(n)?,
            rank: None,
            nnls,
            observed: false,
        };
        if p.dense().is_some() && m > 0 && n > 0 {
            let mut d = Dense {
                active: super::active_qr::ActiveQr::new(m, n)?,
                rank: scioxide_linalg::TriangularRankWorkspace::new(m, n)?,
                matrix: Matrix::zeros(rows, n)?,
                rhs: Vector::zeros(rows)?,
                solution: Vector::zeros(n)?,
                qr: HouseholderDecomposition::empty(),
                qw: LinalgWorkspace::new(),
                svd: ThinSvdDecomposition::empty(),
                sw: LinalgWorkspace::new(),
            };
            // Prepare factor outputs as well as scratch at maximum shape; subsequent
            // compact passive systems retain these capacities.
            scioxide_linalg::HouseholderDecomposition::factor_into(
                &d.matrix,
                scioxide_linalg::HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
                &mut d.qr,
                &mut d.qw,
                scioxide_core::ExecutionContext::sequential(),
            )?;
            d.qw.try_reserve(
                &scioxide_linalg::HouseholderDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                        rows,
                        rows,
                        scioxide_linalg::HouseholderOrientation::Qr,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
            scioxide_linalg::ThinSvdDecomposition::compute_into(
                &d.matrix,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut d.svd,
                &mut d.sw,
                scioxide_core::ExecutionContext::sequential(),
            )?;
            // Wide rank diagnostics use additional candidate U/V storage.
            d.sw.try_reserve(
                &scioxide_linalg::ThinSvdDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::SvdWorkspaceRequest::new(
                        m,
                        n,
                        &(scioxide_linalg::LinalgExecutionOptions::new()),
                    ),
                )?,
            )?;
            w.dense = Some(d);
        } else if m > 0 && n > 0 {
            let op = super::linear::Model {
                p,
                passive: &w.passive,
            };
            w.lsmr = Some(
                super::linear::solver(o)?
                    .workspace(&scioxide_linalg::IterativeWorkspaceRequest::adjoint(&op)?)?,
            );
        }
        Ok(w)
    }
}
