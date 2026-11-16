//! Restarted flexible GMRES.
use super::arnoldi::{Method, Options};
use super::flexible::{requirements, solve_scalar};
use crate::Preconditioning;
use crate::{
    IterativeLinearSolver, IterativeRequirements, IterativeSolveReport, IterativeSolveResult,
    IterativeWorkspace, IterativeWorkspaceRequest, LinalgExecutionContext, VectorView,
    VectorViewMut,
};
use core::fmt;
use scioxide_core::{
    Budget, ExecutionContext, LinearOperator, NumericScalar, RankThreshold, ResidualTolerance,
    SciOxideError,
};

/// Restarted flexible GMRES for general square systems.
///
/// `A` may be nonsymmetric or indefinite. In each cycle of at most `m` steps
/// (default 30, see [`Self::with_restart`]), GMRES builds an orthonormal Arnoldi
/// basis `v₁, v₂, …`, stores the preconditioned directions `zⱼ = M⁻¹ vⱼ`, and
/// minimizes `‖b − A x‖₂` over `x₀ + span{z₁, …, zⱼ}`. The images `A zⱼ` are
/// orthonormalized as they are computed, so the least-squares update needs no
/// Hessenberg matrix. Because each `zⱼ` is stored, `M⁻¹` may change between
/// applications (right flexible preconditioning).
///
/// Step `j` of a cycle costs one product with `A`, one application of `M⁻¹`
/// and `O(j n)` orthogonalization work. The workspace holds about `3m` vectors
/// of length `n`. `max_iterations` counts steps over all cycles.
///
/// The solve converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`, with `‖b‖₂`
/// replaced by one when `b = 0`. The true residual is recomputed at the end of
/// each cycle, every 50 steps, and whenever the residual estimate meets the
/// tolerance. If the Krylov space stops growing before convergence, the solve
/// stops with [`Breakdown(ArnoldiInvariant)`](crate::LinearSolveBreakdown::ArnoldiInvariant).
#[derive(Clone)]
#[doc = r#"

# Examples

```
use scioxide_core::{Budget, ResidualTolerance};
use scioxide_linalg::*;
let a = Matrix::<f64>::identity(2)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let tolerance = ResidualTolerance::new(1e-10, 0.0)?;
let solver = GmresSolver::new(tolerance, Budget::try_from(20_usize)?);
let result = solver.solve(&a.view().try_finite()?, &rhs)?;
assert!((result.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.

# References

- Y. Saad and M. H. Schultz, "GMRES: A generalized minimal residual algorithm for
  solving nonsymmetric linear systems", *SIAM J. Sci. Stat. Comput.* 7(3), 1986.
- Y. Saad, "A flexible inner-outer preconditioned GMRES algorithm", *SIAM J. Sci.
  Comput.* 14(2), 1993."#]
pub struct GmresSolver<S: NumericScalar> {
    options: Options<S::Real>,
}
impl<S: NumericScalar> fmt::Debug for GmresSolver<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GmresSolver")
            .field("options", &self.options)
            .finish()
    }
}
impl<S: NumericScalar> GmresSolver<S> {
    /// Returns the execution options.
    pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
        self.options.execution
    }

    /// Sets the execution options: policy, product accuracy, time limit, byte
    /// limit and iteration budget.
    ///
    /// An iteration budget in `execution` caps `max_iterations`. The accuracy
    /// setting applies to dense operator products; the recurrence safeguards
    /// always use `Standard` accuracy.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.options.execution = execution;
        self
    }

    /// Creates a GMRES solver with residual tolerance `tolerance` and at most
    /// `max_iterations` steps in total.
    ///
    /// The restart length is 30, both parts of the rank threshold are the
    /// machine epsilon, and no history is recorded.
    pub fn new(tolerance: ResidualTolerance<S::Real>, max_iterations: Budget) -> Self {
        Self {
            options: Options::new(Method::Gmres, tolerance, max_iterations),
        }
    }
    /// Sets the restart length `m`, the number of steps per cycle.
    ///
    /// Memory grows linearly with `m` and orthogonalization cost quadratically.
    pub const fn with_restart(mut self, restart: Budget) -> Self {
        self.options.restart = restart;
        self
    }

    /// Sets the threshold for dropping dependent directions.
    ///
    /// A new basis vector or image is dropped when orthogonalization reduces its
    /// norm to at most `atol + rtol` times its original norm. By default both
    /// `atol` and `rtol` are the machine epsilon.
    pub const fn with_rank_threshold(mut self, rank: RankThreshold<S::Real>) -> Self {
        self.options.rank = rank;
        self
    }

    /// Enables or disables the residual history.
    ///
    /// The history holds the residual estimate after each step. The entries for
    /// the initial guess, cycle ends and the returned iterate are true residuals.
    pub const fn with_history(mut self, record: bool) -> Self {
        self.options.history = record;
        self
    }
    /// Returns the storage a solve of `request` needs, including preconditioner scratch.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the execution options.
    pub fn workspace_requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>, SciOxideError> {
        requirements(&self.options, request, 1, Some(request))
    }
    /// Creates a workspace reserved for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn workspace(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeWorkspace<S>, SciOxideError> {
        let mut workspace = IterativeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
    ///
    /// A solve that ends without meeting the tolerance, at the iteration or
    /// evaluation limit or by a breakdown, is an error. Use
    /// [`solve_with_options`](Self::solve_with_options) for a preconditioner, a
    /// start vector or the iterate of an unsuccessful solve.
    ///
    /// # Errors
    ///
    /// - The errors of [`solve_with_options`](Self::solve_with_options).
    /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge; its
    ///   diagnostics name the solver and carry the iteration count and the
    ///   residual norm.
    pub fn solve<'b>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
    ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
        self.solve_with_options(a, b, crate::Preconditioning::None, None)?
            .into_converged_solution()
    }
    /// Solves `A x = b` from the initial guess `x0`, or from zero when `x0` is `None`, and returns
    /// the iterate with its report.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn solve_with_options<'b>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
        preconditioning: Preconditioning<'_, S>,
        x0: Option<VectorView<'_, S>>,
    ) -> Result<IterativeSolveResult<S>, SciOxideError> {
        let b = b.into();
        IterativeWorkspace::solve_owned(a.cols(), x0, self.algorithm_id(), |x, workspace| {
            self.solve_into(
                a,
                b,
                preconditioning,
                x,
                workspace,
                ExecutionContext::sequential_native(),
            )
        })
    }
    /// Solves `A x = b` in place, starting from the values in `x`.
    ///
    /// # Errors
    ///
    /// The errors listed on
    /// [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
        preconditioning: Preconditioning<'_, S>,
        x: VectorViewMut<'_, S>,
        workspace: &mut IterativeWorkspace<S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<IterativeSolveReport<S::Real>, SciOxideError> {
        solve_scalar(
            &self.options,
            a,
            b.into(),
            preconditioning,
            None,
            x,
            workspace,
            context.into(),
        )
    }
}
crate::traits::impl_iterative_solver!(GmresSolver, plan, plan.options.method.algorithm_id(false));

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use scioxide_core::Preconditioner;
    use std::cell::Cell;

    use super::*;
    use crate::Vector;
    use crate::dense::Matrix;
    use crate::iterative::safety::tests::true_residual;
    use scioxide_core::Complex;
    use scioxide_sparse::{CsrMatrix, JacobiPreconditioner};

    use super::super::test_support::{FailingPreconditioner, max_iter_behavior};

    struct ZeroPreconditioner;

    impl Preconditioner<f64> for ZeroPreconditioner {
        fn dimension(&self) -> usize {
            2
        }

        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }

        fn precondition_into(
            &self,
            _input: &[f64],
            output: &mut [f64],
            _workspace: &mut [f64],
        ) -> Result<(), SciOxideError> {
            output.fill(0.0);
            Ok(())
        }
    }

    struct DriftingOperator {
        calls: Cell<usize>,
    }

    impl LinearOperator<f64> for DriftingOperator {
        fn rows(&self) -> usize {
            1
        }

        fn cols(&self) -> usize {
            1
        }

        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(1)
        }

        fn apply_into(
            &self,
            input: &[f64],
            output: &mut [f64],
            workspace: &mut [f64],
        ) -> Result<(), SciOxideError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            workspace[0] = if call == 1 { input[0] } else { 2.0 * input[0] };
            output.copy_from_slice(workspace);
            Ok(())
        }
    }

    fn budget(value: usize) -> Budget {
        Budget::try_from(value).unwrap()
    }

    fn solver(tolerance: f64, max_iterations: usize) -> GmresSolver<f64> {
        GmresSolver::new(
            ResidualTolerance::new(tolerance, 0.0).unwrap(),
            budget(max_iterations),
        )
    }

    /// Helper: build a dense matrix from flat data.
    fn mat_op(data: &[f64], rows: usize, cols: usize) -> Matrix<f64> {
        Matrix::from_vec(rows, cols, data.to_vec()).unwrap()
    }

    #[test]
    fn test_solve_nonsymmetric() {
        // A = [[2, 1], [0, 3]], b = [5, 9] => x = [1, 3]
        let a = mat_op(&[2.0, 1.0, 0.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![5.0, 9.0]);
        let solver = solver(1e-10, 100);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!((result.solution().view().get(0).unwrap() - 1.0).abs() < 1e-8);
        assert!((result.solution().view().get(1).unwrap() - 3.0).abs() < 1e-8);
    }

    #[test]
    fn complex_unitary_givens_reproducer_converges() {
        let imaginary = Complex::new(0.0_f64, 1.0);
        let a = Matrix::from_vec(
            2,
            2,
            vec![
                imaginary,
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
            ],
        )
        .unwrap();
        let b = Vector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let solver = GmresSolver::<Complex<f64>>::new(
            ResidualTolerance::new(1.0e-12, 0.0).unwrap(),
            budget(8),
        )
        .with_restart(budget(2));

        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() <= 1.0e-12);
        assert!((result.solution().as_slice()[0] + imaginary).norm() <= 1.0e-12);
        assert!((result.solution().as_slice()[1] - imaginary).norm() <= 1.0e-12);
    }

    #[test]
    fn deterministic_complex_residual_fixtures_converge() {
        for hermitian in [true, false] {
            let (matrix, right_hand_side) =
                crate::iterative::test_support::deterministic_complex_system(hermitian);
            let solver = GmresSolver::<Complex<f64>>::new(
                ResidualTolerance::new(1.0e-11, 0.0).unwrap(),
                budget(32),
            )
            .with_restart(budget(4));

            let result = solver
                .solve_with_options(
                    &matrix.view().try_finite().unwrap(),
                    &right_hand_side,
                    Preconditioning::None,
                    None,
                )
                .unwrap();
            let (_, residual) = true_residual(
                &matrix.view().try_finite().unwrap(),
                &right_hand_side,
                result.solution(),
            )
            .unwrap();
            assert!(result.is_success());
            assert!(
                residual <= 1.0e-11,
                "Hermitian={hermitian}, residual={residual}"
            );
        }
    }

    #[test]
    fn test_solve_identity() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let b = Vector::<f64>::from_vec(vec![7.0, 8.0, 9.0]);
        let solver = solver(1e-10, 100);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        for i in 0..3 {
            assert!(
                (result.solution().view().get(i).unwrap() - b.view().get(i).unwrap()).abs() < 1e-8
            );
        }
    }

    #[test]
    fn test_convergence() {
        let a = mat_op(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 200);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() < 1e-10);
        let (_, true_norm) =
            true_residual(&a.view().try_finite().unwrap(), &b, result.solution()).unwrap();
        assert_eq!(
            result.report().diagnostic_residual_norm().unwrap(),
            true_norm
        );
    }

    #[test]
    fn test_restart_behavior() {
        let a = mat_op(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 200).with_restart(budget(2));
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_with_preconditioner() {
        let a_data = [4.0, 1.0, 1.0, 3.0];
        let a = mat_op(&a_data, 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let a_mat = mat_op(&a_data, 2, 2);
        let sparse = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            a_mat.as_slice().to_vec(),
        )
        .unwrap();
        let pc = JacobiPreconditioner::new(&sparse).unwrap();
        let solver = solver(1e-10, 200);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&pc),
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() < 1e-10);
    }

    #[test]
    fn test_history_recording() {
        let a = mat_op(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 200).with_history(true);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.history().is_some());
        let hist = result.history().unwrap();
        assert_eq!(hist.len(), result.report().iterations() + 1);
        assert_eq!(
            hist.last().copied(),
            Some(result.report().diagnostic_residual_norm().unwrap())
        );
    }

    #[test]
    fn workspace_reuses_storage_and_borrows_history() {
        let a = mat_op(&[2.0, 1.0, 0.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![5.0, 9.0]);
        let solver = solver(1e-10, 20).with_restart(budget(2)).with_history(true);
        let mut workspace = IterativeWorkspace::new();
        let mut x = Vector::from_vec(vec![0.0, 0.0]);

        {
            let report = solver
                .solve_into(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    x.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(
                workspace.last_accepted().unwrap().history().unwrap().len(),
                report.iterations() + 1
            );
            assert_eq!(
                workspace.last_accepted().unwrap().history().unwrap().last(),
                Some(&report.diagnostic_residual_norm().unwrap())
            );
        }

        x.view_mut().fill(0.0);
        let report = solver
            .solve_into(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-8);
        assert!((x.view().get(1).unwrap() - 3.0).abs() < 1e-8);
    }

    #[test]
    fn smaller_reservation_is_replaced_by_the_solve() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let a = a.view().try_finite().unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        let request = IterativeWorkspaceRequest::operator(&a).unwrap();
        let mut workspace = solver(1e-10, 1)
            .with_restart(budget(1))
            .workspace(&request)
            .unwrap();
        let solver = solver(1e-10, 4).with_restart(budget(2)).with_history(true);
        let requirements = solver.workspace_requirements(&request).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            scioxide_core::Capacity::NeedsGrowth
        );
        let mut x = Vector::from_vec(vec![7.0, 8.0]);
        let report = solver
            .solve_into(
                &a,
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            scioxide_core::Capacity::Sufficient
        );
    }

    #[test]
    fn workspace_preconditioner_error_leaves_caller_solution_unchanged() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 2);
        let mut workspace = IterativeWorkspace::new();
        let mut x = Vector::from_vec(vec![7.0, 8.0]);

        assert!(matches!(
            (solver.solve_into(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&FailingPreconditioner),
                x.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(x.as_slice(), &[7.0, 8.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = mat_op(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let solver = solver(1e-10, 100);
        assert!(matches!(
            (solver.solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_max_iter_behavior() {
        let _ = max_iter_behavior(&solver(1e-15, 1).with_restart(budget(1)));
    }

    #[test]
    fn test_restart_budget_rejects_zero() {
        assert!(matches!(
            (Budget::try_from(0)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_restart_workspace_overflow_is_structural() {
        let a = Matrix::<f64>::identity(1).unwrap();
        let b = Vector::<f64>::from_vec(vec![1.0]);
        let solver = solver(0.0, usize::MAX).with_restart(budget(usize::MAX));
        assert!(
            matches!(solver.solve_with_options(&a.view().try_finite().unwrap(), &b, Preconditioning::None, None), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow { context, .. } if context == scioxide_core::quantity!("gmres_hessenberg_leading_dimension")))
        );
    }

    #[test]
    fn test_iteration_budget_rejects_zero() {
        assert!(matches!(
            (Budget::try_from(0)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn singular_consistent_system_converges_by_happy_breakdown() {
        let a = mat_op(&[1.0, 0.0, 0.0, 0.0], 2, 2);
        let b = Vector::from_vec(vec![1.0, 0.0]);
        let result = solver(0.0, 4)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());
        assert_eq!(result.report().iterations(), 1);
        assert_eq!(result.report().diagnostic_residual_norm().unwrap(), 0.0);
    }

    #[test]
    fn singular_inconsistent_system_reports_breakdown() {
        let a = mat_op(&[1.0, 0.0, 0.0, 0.0], 2, 2);
        let b = Vector::from_vec(vec![0.0, 1.0]);

        let result = solver(0.0, 4)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(matches!(
            result.report().termination(),
            crate::LinearSolveTermination::Breakdown(_)
        ));
        assert_eq!(result.report().true_residual().unwrap().raw_norm(), 1.0);
    }

    #[test]
    fn scale_aware_hessenberg_checks_accept_small_system() {
        let a = mat_op(&[1.0e-100], 1, 1);
        let b = Vector::from_vec(vec![1.0e-100]);
        let result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());
        assert!((result.solution().view().get(0).unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn zero_rhs_and_exact_guess_converge_without_preconditioning() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let zero = Vector::from_vec(vec![0.0, 0.0]);
        let zero_result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &zero,
                Preconditioning::Inverse(&FailingPreconditioner),
                None,
            )
            .unwrap();
        assert_eq!(zero_result.report().iterations(), 0);

        let exact = Vector::from_vec(vec![2.0, -3.0]);
        let exact_result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &exact,
                Preconditioning::Inverse(&FailingPreconditioner),
                Some(exact.view()),
            )
            .unwrap();
        assert_eq!(exact_result.report().iterations(), 0);
    }

    #[test]
    fn true_residual_restarts_after_false_projected_convergence() {
        let operator = DriftingOperator {
            calls: Cell::new(0),
        };
        let b = Vector::from_vec(vec![1.0]);
        let result = solver(0.0, 2)
            .solve_with_options(&operator, &b, Preconditioning::None, None)
            .unwrap();

        assert!(result.is_success());
        assert!((result.solution().view().get(0).unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(result.report().diagnostic_residual_norm().unwrap(), 0.0);
    }

    #[test]
    fn injected_zero_preconditioner_reports_breakdown() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let b = Vector::from_vec(vec![1.0, 0.0]);
        let result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&ZeroPreconditioner),
                None,
            )
            .unwrap();
        assert!(matches!(
            result.report().termination(),
            crate::LinearSolveTermination::Breakdown(_)
        ));
    }

    #[test]
    fn preconditioner_error_context_is_preserved() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let b = Vector::from_vec(vec![1.0, 0.0]);
        let error = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&FailingPreconditioner),
                None,
            )
            .unwrap_err();

        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, reason, detail, .. } if argument == scioxide_core::argument!("injected_preconditioner") && reason == scioxide_core::InvalidInputReason::UnsupportedConfiguration && detail == "injected preconditioner failure")
        );
    }
}
