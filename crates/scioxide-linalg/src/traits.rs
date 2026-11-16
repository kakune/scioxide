//! The object-safe trait shared by the single-vector iterative solvers.

// Keep the object-safe signatures and argument forwarding identical for every
// scalar solver. Numerical execution and owned allocation live in the inherent API.
macro_rules! impl_iterative_solver {
    ($solver:ident, $plan:ident, $algorithm:expr) => {
        impl<S: scioxide_core::NumericScalar> crate::IterativeLinearSolver<S> for $solver<S> {
            fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
                let $plan = self;
                $algorithm
            }

            fn workspace_requirements(
                &self,
                request: &crate::IterativeWorkspaceRequest,
            ) -> core::result::Result<crate::IterativeRequirements<S>, scioxide_core::SciOxideError>
            {
                Self::workspace_requirements(self, request)
            }

            fn workspace(
                &self,
                request: &crate::IterativeWorkspaceRequest,
            ) -> core::result::Result<crate::IterativeWorkspace<S>, scioxide_core::SciOxideError>
            {
                Self::workspace(self, request)
            }

            fn solve(
                &self,
                a: &dyn scioxide_core::LinearOperator<S>,
                b: crate::VectorView<'_, S>,
            ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
                Self::solve(self, a, b)
            }

            fn solve_with_options(
                &self,
                a: &dyn scioxide_core::LinearOperator<S>,
                b: crate::VectorView<'_, S>,
                preconditioning: crate::Preconditioning<'_, S>,
                x0: Option<crate::VectorView<'_, S>>,
            ) -> core::result::Result<crate::IterativeSolveResult<S>, scioxide_core::SciOxideError>
            {
                Self::solve_with_options(self, a, b, preconditioning, x0)
            }

            fn solve_into(
                &self,
                a: &dyn scioxide_core::LinearOperator<S>,
                b: crate::VectorView<'_, S>,
                preconditioning: crate::Preconditioning<'_, S>,
                x: crate::VectorViewMut<'_, S>,
                workspace: &mut crate::IterativeWorkspace<S>,
                context: crate::LinalgExecutionContext<'_, '_>,
            ) -> core::result::Result<
                crate::IterativeSolveReport<S::Real>,
                scioxide_core::SciOxideError,
            > {
                Self::solve_into(self, a, b, preconditioning, x, workspace, context)
            }
        }
    };
}
pub(crate) use impl_iterative_solver;

use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, ExecutionContext, LinearOperator, NumericScalar,
    SciOxideError,
};

use crate::result::{IterativeSolveReport, IterativeSolveResult};
use crate::{
    IterativeRequirements, IterativeWorkspace, IterativeWorkspaceRequest, LinalgExecutionContext,
    Preconditioning, VectorView, VectorViewMut,
};

/// Object-safe interface of iterative solvers for a square system `A x = b`.
///
/// `A` is any [`LinearOperator`] and the preconditioner is passed per call as
/// a [`Preconditioning`] value; each solver documents on which side it applies
/// it. Use [`Self::solve`] for a one-off solve from zero without
/// preconditioning that returns the solution, [`Self::solve_with_options`] for
/// a preconditioner, a start vector and the report, and [`Self::solve_into`]
/// with an [`IterativeWorkspace`] to reuse storage. Code
/// that selects a solver at run time can hold a `&dyn IterativeLinearSolver<S>`.
///
/// Implemented by [`CgSolver`](crate::CgSolver), [`GmresSolver`](crate::GmresSolver),
/// [`BicgstabSolver`](crate::BicgstabSolver), [`MinresSolver`](crate::MinresSolver),
/// [`RichardsonSolver`](crate::RichardsonSolver), [`CgsSolver`](crate::CgsSolver),
/// [`TfqmrSolver`](crate::TfqmrSolver), [`LgmresSolver`](crate::LgmresSolver) and
/// [`GcrotSolver`](crate::GcrotSolver). The other solvers need more than a
/// `LinearOperator` and offer only inherent methods: [`BicgSolver`](crate::BicgSolver)
/// and [`QmrSolver`](crate::QmrSolver) apply `Aᴴ`, the Jacobi, Gauss–Seidel and SOR
/// solvers read matrix entries, [`LsqrSolver`](crate::LsqrSolver) and
/// [`LsmrSolver`](crate::LsmrSolver) accept rectangular least-squares problems, and
/// the `Block*` solvers iterate on several right-hand sides together.
///
/// An implementation outside this crate builds its report with
/// [`IterativeSolveReport::new`]. It does not have to update the workspace's
/// attempt record.
pub trait IterativeLinearSolver<S: NumericScalar> {
    /// Returns the identifier of the solver's algorithm; its method name is, for example, `CG`.
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId;

    /// Returns `Ok` if `report` has a successful termination.
    ///
    /// # Errors
    ///
    /// - `ExecutionStopped` with the original reason if the solve was cancelled
    ///   or reached its deadline.
    /// - `ConvergenceFailure` (`NestedSolver`) for any other unsuccessful
    ///   termination. Its diagnostics name this solver and carry the iteration
    ///   count and residual norm of `report`.
    fn require_success(&self, report: &IterativeSolveReport<S::Real>) -> Result<(), SciOxideError> {
        report.require_success(self.algorithm_id())
    }

    /// Solves `A x = b` in place, starting from the values in `x`.
    ///
    /// Pass an `ExecutionContext` for a standalone solve, or a solve context to
    /// run as a nested solve whose operator and preconditioner calls count
    /// toward the parent's limits (see [Solve accounting](scioxide_core#solve-accounting)).
    /// Stops on the iteration limit, the evaluation limit, cancellation or a
    /// deadline return `Ok` with an unsuccessful termination, as described in
    /// [Iterative solves](crate#iterative-solves).
    ///
    /// # Errors
    ///
    /// The solvers of this crate return:
    ///
    /// - `DimensionMismatch` if `a` is not square, or if `b`, `x` or the
    ///   preconditioner does not match its dimension.
    /// - `NonFinite` if `b` or `x` contains NaN or infinity, or if the operator
    ///   or preconditioner returns one.
    /// - `NumericalInstability` if the recurrence overflows.
    /// - `ConvergenceFailure` (`EvaluationLimit`) if the evaluation limit is
    ///   reached before the initial residual is computed.
    /// - Errors returned by the operator or preconditioner, unchanged.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    fn solve_into(
        &self,
        a: &dyn LinearOperator<S>,
        b: VectorView<'_, S>,
        preconditioning: Preconditioning<'_, S>,
        x: VectorViewMut<'_, S>,
        workspace: &mut IterativeWorkspace<S>,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<S::Real>, SciOxideError>;

    /// Returns the storage that [`Self::solve_into`] and [`Self::solve_block_into`] need.
    ///
    /// The default implementation reserves only the columns staged by
    /// [`Self::solve_block_into`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the solver's byte limit.
    fn workspace_requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>, SciOxideError> {
        IterativeRequirements::staging_only(request)
    }

    /// Creates a workspace reserved for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if memory cannot be reserved.
    fn workspace(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeWorkspace<S>, SciOxideError> {
        let mut workspace = IterativeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
    ///
    /// Runs as described in [Execution](crate#execution) for owned forms. A
    /// solve that ends without meeting the tolerance is an error.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_with_options`].
    /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge;
    ///   its diagnostics name the solver and carry the iteration count and the
    ///   residual norm.
    fn solve(
        &self,
        a: &dyn LinearOperator<S>,
        b: VectorView<'_, S>,
    ) -> Result<crate::Vector<S>, SciOxideError> {
        self.solve_with_options(a, b, Preconditioning::None, None)?
            .into_converged_solution()
    }

    /// Solves `A x = b` from the initial guess `x0`, or from zero when `x0` is
    /// `None`, and returns the iterate with its report.
    ///
    /// Runs as described in [Execution](crate#execution) for owned forms.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    fn solve_with_options(
        &self,
        a: &dyn LinearOperator<S>,
        b: VectorView<'_, S>,
        preconditioning: Preconditioning<'_, S>,
        x0: Option<VectorView<'_, S>>,
    ) -> Result<IterativeSolveResult<S>, SciOxideError> {
        IterativeWorkspace::solve_owned(a.cols(), x0, self.algorithm_id(), |x, workspace| {
            self.solve_into(
                a,
                b,
                preconditioning,
                x,
                workspace,
                ExecutionContext::sequential_native().into(),
            )
        })
    }

    /// Solves `A X = B` for several right-hand sides, one column at a time.
    ///
    /// `b` and `x` store one column per right-hand side; the columns of `x` are
    /// the initial guesses. Each column is an independent call of
    /// [`Self::solve_into`], so this is not a block Krylov method; use the
    /// `Block*` solvers to iterate on all columns together. A workspace reserved
    /// for a request with `with_right_hand_sides(k)` solves `k` columns without
    /// allocating. `x` is written only after every column converges. After a
    /// failure, [`IterativeWorkspace::last_attempt`] describes the failing column.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_into`] for any column.
    /// - `DimensionMismatch` if `b` or `x` does not have `a.cols()` rows and the
    ///   same number of columns.
    /// - `ExecutionStopped` with the original reason if a column is cancelled
    ///   or reaches its deadline.
    /// - `ConvergenceFailure` (`NestedSolver`) for the first column that does not
    ///   converge.
    fn solve_block_into(
        &self,
        a: &dyn LinearOperator<S>,
        b: BlockVectorView<'_, S>,
        preconditioning: Preconditioning<'_, S>,
        mut x: BlockVectorViewMut<'_, S>,
        workspace: &mut IterativeWorkspace<S>,
        mut context: LinalgExecutionContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        // Until a column starts its epoch, the record is a preflight rejection.
        workspace.reject();
        let (n, columns) = (a.cols(), b.right_hand_sides());
        crate::checked::square(a.rows(), n)?;
        crate::checked::expect_shape([n, columns], [b.rows(), columns])?;
        crate::checked::expect_shape([n, columns], [x.rows(), x.right_hand_sides()])?;
        let request = IterativeWorkspaceRequest::operator(a)?
            .with_preconditioner(preconditioning)?
            .with_right_hand_sides(columns);
        workspace.try_reserve(&self.workspace_requirements(&request)?)?;
        let mut staging = workspace.take_block();
        staging.clear();
        staging.extend_from_slice(x.as_mut_slice());
        let outcome = staging
            .chunks_exact_mut(n.max(1))
            .take(columns)
            .enumerate()
            .try_for_each(|(column, solution)| {
                self.solve_into(
                    a,
                    VectorView::from_slice(b.column(column)?),
                    preconditioning,
                    VectorViewMut::from_slice(solution),
                    workspace,
                    context.reborrow(),
                )?
                .require_success(self.algorithm_id())
            });
        if outcome.is_ok() {
            x.as_mut_slice().copy_from_slice(&staging);
        }
        workspace.restore_block(staging);
        outcome
    }
}
