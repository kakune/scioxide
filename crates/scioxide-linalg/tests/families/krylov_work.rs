//! Deterministic recurrence work, finite-input routing and history contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AdjointOperator, Budget, EvaluationRole, ExecutionContext, GradientTolerance, LinearOperator,
    ResidualTolerance, SciOxideError,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgstabSolver, CgSolver, FiniteMatrixView, GmresSolver, IterativeLinearSolver,
    IterativeWorkspace, LsqrSolver, Matrix, MinresSolver, Vector,
};
use std::cell::Cell;

struct Counted<'a> {
    matrix: FiniteMatrixView<'a, f64>,
    forward: Cell<usize>,
    adjoint: Cell<usize>,
    checked: Cell<usize>,
    stop_at: Cell<Option<usize>>,
}

impl<'a> Counted<'a> {
    fn new(matrix: &'a Matrix<f64>) -> Self {
        Self {
            matrix: matrix.view().try_finite().unwrap(),
            forward: Cell::new(0),
            adjoint: Cell::new(0),
            checked: Cell::new(0),
            stop_at: Cell::new(None),
        }
    }
}

impl LinearOperator<f64> for Counted<'_> {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }
    fn cols(&self) -> usize {
        self.matrix.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_workspace_len()
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<(), SciOxideError> {
        self.checked.set(self.checked.get() + 1);
        self.matrix.apply_into(x, y, w)
    }
    fn apply_finite_into_scoped(
        &self,
        x: &[f64],
        maximum: f64,
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        assert!(x.iter().all(|v| v.is_finite() && v.abs() <= maximum));
        self.forward.set(self.forward.get() + 1);
        if self.stop_at.get() == Some(self.forward.get() + self.adjoint.get()) {
            return Err(SciOxideError::execution_stopped(
                scioxide_core::ExecutionStopReason::DeadlineReached,
            ));
        }
        self.matrix.apply_finite_into_scoped(x, maximum, y, w, c)
    }
}

impl AdjointOperator<f64> for Counted<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.checked.set(self.checked.get() + 1);
        self.matrix.apply_adjoint_into(x, y, w)
    }
    fn apply_adjoint_finite_into_scoped(
        &self,
        x: &[f64],
        maximum: f64,
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        assert!(x.iter().all(|v| v.is_finite() && v.abs() <= maximum));
        self.adjoint.set(self.adjoint.get() + 1);
        if self.stop_at.get() == Some(self.forward.get() + self.adjoint.get()) {
            return Err(SciOxideError::execution_stopped(
                scioxide_core::ExecutionStopReason::DeadlineReached,
            ));
        }
        self.matrix
            .apply_adjoint_finite_into_scoped(x, maximum, y, w, c)
    }
}

#[test]
fn fixed_spd_counts_include_only_recurrence_and_verified_checkpoints() {
    let matrix = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 4.]).unwrap();
    let b = [1.; 3];
    let tolerance = ResidualTolerance::new(0., 1e-8).unwrap();
    let limit = Budget::try_from(30).unwrap();
    for history in [false, true] {
        let cg = CgSolver::new(tolerance, limit).with_history(history);
        let gmres = GmresSolver::new(tolerance, limit).with_history(history);
        let bicgstab = BicgstabSolver::new(tolerance, limit).with_history(history);
        let minres = MinresSolver::new(tolerance, limit).with_history(history);
        let solvers: [(&dyn IterativeLinearSolver<f64>, usize); 4] =
            [(&cg, 5), (&gmres, 5), (&bicgstab, 7), (&minres, 7)];
        for (solver, applications) in solvers {
            let a = Counted::new(&matrix);
            let mut x = Vector::zeros(3).unwrap();
            let mut workspace = IterativeWorkspace::new();
            let report = solver
                .solve_into(
                    &a,
                    b.as_slice().into(),
                    Preconditioning::None,
                    x.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential().into(),
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(report.iterations(), 3);
            assert_eq!(a.forward.get(), applications);
            assert_eq!(a.adjoint.get(), 0);
            assert_eq!(a.checked.get(), 0);
            assert_eq!(
                workspace
                    .last_attempt()
                    .unwrap()
                    .counts()
                    .own()
                    .provider(EvaluationRole::LinearOperator),
                applications
            );
            for (&actual, expected) in x.as_slice().iter().zip([1., 0.5, 0.25]) {
                assert!((actual - expected).abs() < 1e-8);
            }
            if history {
                let samples = workspace.last_accepted().unwrap().history().unwrap();
                assert_eq!(samples.len(), 4);
                assert_eq!(samples.last().copied(), report.diagnostic_residual_norm());
            }
        }
    }
}

#[test]
fn fixed_inconsistent_least_squares_reuses_initial_adjoint_and_checks_final_residuals() {
    let matrix = Matrix::from_vec(
        6,
        3,
        vec![
            1., 0., 0., 0., 2., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    )
    .unwrap();
    for history in [false, true] {
        let a = Counted::new(&matrix);
        let solver = LsqrSolver::new(
            ResidualTolerance::new(0., 1e-8).unwrap(),
            Budget::try_from(30).unwrap(),
        )
        .with_normal_tolerance(GradientTolerance::new(0., 1e-8).unwrap())
        .with_history(history);
        let result = solver.solve_with_options(&a, &[1.; 6][..], None).unwrap();
        let report = result.report();
        assert!(result.is_success());
        assert_eq!(report.iterations(), 3);
        assert_eq!(
            (a.forward.get(), a.adjoint.get(), a.checked.get()),
            (5, 5, 0)
        );

        assert!((report.true_residual().unwrap().raw_norm() - 3_f64.sqrt()).abs() < 1e-12);
        assert!(
            report
                .krylov()
                .unwrap()
                .normal_residual()
                .unwrap()
                .raw_norm()
                < 1e-8
        );
        if history {
            assert_eq!(result.history().unwrap().len(), 4);
            assert_eq!(
                result.history().unwrap().last().copied(),
                report.diagnostic_residual_norm()
            );
        }
    }
}

#[test]
fn nonfinite_inputs_fail_before_finite_operator_hooks() {
    let matrix = Matrix::from_vec(1, 1, vec![1.]).unwrap();
    let a = Counted::new(&matrix);
    let solver = CgSolver::new(ResidualTolerance::default(), Budget::ONE);
    for (b, guess) in [(f64::NAN, 0.), (1., f64::INFINITY)] {
        let mut x = Vector::from_vec(vec![guess]);
        assert!(
            solver
                .solve_into(
                    &a,
                    &[b][..],
                    Preconditioning::None,
                    x.view_mut(),
                    &mut IterativeWorkspace::new(),
                    ExecutionContext::sequential(),
                )
                .is_err()
        );
        assert_eq!(x.as_slice()[0].to_bits(), guess.to_bits());
        assert_eq!(a.forward.get(), 0);
    }
}

#[test]
fn short_recurrences_restart_after_false_projected_convergence() {
    struct PerturbedProduct(Cell<usize>);
    impl LinearOperator<f64> for PerturbedProduct {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            let call = self.0.get() + 1;
            self.0.set(call);
            // Initial verification uses the identity; disturb only the first
            // recurrence application to simulate a false projected residual.
            y[0] = if call == 3 { 2. * x[0] } else { x[0] };
            Ok(())
        }
    }
    impl AdjointOperator<f64> for PerturbedProduct {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            w: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.apply_into(x, y, w)
        }
    }
    let tolerance = ResidualTolerance::new(0., 1e-12).unwrap();
    let budget = Budget::try_from(10).unwrap();
    let a = PerturbedProduct(Cell::new(0));
    let minres = MinresSolver::new(tolerance, budget)
        .with_history(true)
        .solve_with_options(&a, &[2.][..], Preconditioning::None, None)
        .unwrap();
    a.0.set(0);
    let lsqr = LsqrSolver::new(tolerance, budget)
        .with_history(true)
        .solve_with_options(&a, &[2.][..], None)
        .unwrap();
    for result in [minres, lsqr] {
        assert!(result.is_success());
        assert_eq!(result.report().iterations(), 2);
        assert_eq!(result.report().krylov().unwrap().restarts(), 1);
        assert!((result.solution().as_slice()[0] - 2.).abs() < 1e-12);
        assert!(result.report().true_residual().unwrap().raw_norm() < 1e-12);
        assert_eq!(
            result.history().unwrap().last().copied(),
            result.report().diagnostic_residual_norm()
        );
    }
}

#[test]
fn dense_finite_adjoint_tiles_and_scaled_fallback_preserve_range_and_shape_checks() {
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "adjoint_tiles", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    // Three output tiles, including a partial tail; an independent integer result.
    let values: Vec<_> = (0..3)
        .flat_map(|row| (0..129).map(move |col| (row + col % 7) as f64))
        .collect();
    let matrix = Matrix::from_vec(3, 129, values).unwrap();
    let a = matrix.view().try_finite().unwrap();
    let mut out = [9.; 129];
    let mut scratch = [0.; 129];
    a.apply_adjoint_finite_into_scoped(
        &[1., -2., 4.],
        4.,
        &mut out,
        &mut scratch,
        &mut scioxide_core::OperatorContext::new(&mut context),
    )
    .and_then(scioxide_core::Invocation::require)
    .unwrap();
    for (i, &value) in out.iter().enumerate() {
        assert!((value - (3 * (i % 7) + 6) as f64).abs() < 1e-12);
    }
    for maximum in [-1., f64::NAN, f64::INFINITY] {
        let before = out;
        assert!(
            a.apply_adjoint_finite_into_scoped(
                &[1., -2., 4.],
                maximum,
                &mut out,
                &mut scratch,
                &mut scioxide_core::OperatorContext::new(&mut context)
            )
            .is_err()
        );
        assert_eq!(out, before);
    }
    let before = out;
    assert!(
        a.apply_adjoint_finite_into_scoped(
            &[1., 2.],
            2.,
            &mut out,
            &mut scratch,
            &mut scioxide_core::OperatorContext::new(&mut context)
        )
        .is_err()
    );
    assert_eq!(out, before);
    assert!(
        a.apply_adjoint_finite_into_scoped(
            &[1., 2., 3.],
            3.,
            &mut out,
            &mut [],
            &mut scioxide_core::OperatorContext::new(&mut context)
        )
        .is_err()
    );
    assert_eq!(out, before);
    // Overflowing individual products cancel in the Standard scaled frame.
    let mut values = vec![f64::MAX; 129];
    values.extend_from_slice(&[-f64::MAX; 129]);
    let matrix = Matrix::from_vec(2, 129, values).unwrap();
    matrix
        .view()
        .try_finite()
        .unwrap()
        .apply_adjoint_finite_into_scoped(
            &[2., 2.],
            2.,
            &mut out,
            &mut scratch,
            &mut scioxide_core::OperatorContext::new(&mut context),
        )
        .and_then(scioxide_core::Invocation::require)
        .unwrap();
    assert_eq!(out, [0.; 129]);
    // A representable subnormal result survives the scaled fallback.
    let matrix = Matrix::from_vec(2, 129, vec![f64::MIN_POSITIVE; 258]).unwrap();
    matrix
        .view()
        .try_finite()
        .unwrap()
        .apply_adjoint_finite_into_scoped(
            &[0.125, 0.125],
            0.125,
            &mut out,
            &mut scratch,
            &mut scioxide_core::OperatorContext::new(&mut context),
        )
        .and_then(scioxide_core::Invocation::require)
        .unwrap();
    assert!(out.iter().all(|&x| x == f64::MIN_POSITIVE * 0.25));
}

#[test]
fn every_recurrence_charges_only_its_kernel_and_checkpoints_with_reused_storage() {
    use scioxide_linalg::*;
    // Keep every method above its rounding floor through the 65-step budget.
    let n = 180;
    let mut values = vec![0.; n * n];
    for i in 0..n {
        values[i * n + i] = (i + 1) as f64;
    }
    let matrix = Matrix::from_vec(n, n, values).unwrap();
    let b = vec![1.; n];
    let tolerance = ResidualTolerance::new(0., 0.).unwrap();
    let limit = Budget::try_from(65).unwrap();
    macro_rules! check {
        (@inverse) => { Preconditioning::None };
        (@inverse $p:expr) => { $p };
        ($solver:expr, $forward:expr, $adjoint:expr, [$($preconditioning:expr)?], $preconditioner:expr) => {{
            let solver = $solver.with_history(true);
            let a = Counted::new(&matrix);
            let mut x = Vector::zeros(n).unwrap();
            let mut workspace = IterativeWorkspace::new();
            let request = IterativeWorkspaceRequest::adjoint(&a)
                .unwrap()
                .with_preconditioner::<f64>(check!(@inverse $($preconditioning)?))
                .unwrap();
            workspace
                .try_reserve(&solver.workspace_requirements(&request).unwrap())
                .unwrap();
            let report = { let mut evaluate_case = || {
                    solver
                        .solve_into(
                            &a,
                            b.as_slice(),
                            $($preconditioning,)?
                            x.view_mut(),
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap()
                }; evaluate_case() };
            assert_eq!(
                report.termination(),
                LinearSolveTermination::IterationLimit,
                "{}: {report:?}",
                stringify!($solver)
            );
            assert_eq!(report.iterations(), 65);
            assert_eq!(a.forward.get(), $forward, "{} forward", stringify!($solver));
            assert_eq!(a.adjoint.get(), $adjoint, "{} adjoint", stringify!($solver));
            let counts = workspace.last_attempt().unwrap().counts().own();
            assert_eq!(
                counts.provider(EvaluationRole::LinearOperator),
                $forward + $adjoint
            );
            assert_eq!(
                counts.provider(EvaluationRole::Preconditioner),
                $preconditioner
            );
            assert_eq!(
                workspace.last_accepted().unwrap().history().unwrap().len(),
                66
            );
            assert_eq!(workspace.last_accepted().unwrap().progress().restarts(), 0);
            // Stop during final verification, after a completed periodic checkpoint.
            x.as_mut_slice().fill(0.);
            a.forward.set(0);
            a.adjoint.set(0);
            a.stop_at.set(Some($forward + $adjoint));
            let stopped = { let mut evaluate_case = || {
                    solver
                        .solve_into(
                            &a,
                            b.as_slice(),
                            $($preconditioning,)?
                            x.view_mut(),
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap()
                }; evaluate_case() };
            assert_eq!(
                stopped.termination(),
                LinearSolveTermination::ExecutionStopped(
                    scioxide_core::ExecutionStopReason::DeadlineReached
                )
            );
            let true_norm = x
                .as_slice()
                .iter()
                .enumerate()
                .fold(0_f64, |r, (i, x)| r.hypot(1. - (i + 1) as f64 * x));
            assert!((stopped.diagnostic_residual_norm().unwrap() - true_norm).abs() <= 1e-12);
            assert_eq!(
                workspace
                    .last_accepted()
                    .unwrap()
                    .history()
                    .unwrap()
                    .last()
                    .copied(),
                stopped.diagnostic_residual_norm()
            );
        }};
    }
    check!(
        CgSolver::new(tolerance, limit),
        68,
        0,
        [Preconditioning::None],
        0
    );
    check!(
        BicgstabSolver::new(tolerance, limit),
        133,
        0,
        [Preconditioning::None],
        0
    );
    check!(
        BicgSolver::new(tolerance, limit),
        68,
        65,
        [Preconditioning::None],
        0
    );
    check!(
        CgsSolver::new(tolerance, limit),
        133,
        0,
        [Preconditioning::None],
        0
    );
    check!(
        QmrSolver::new(tolerance, limit),
        68,
        65,
        [Preconditioning::None],
        0
    );
    check!(
        TfqmrSolver::new(tolerance, limit),
        68,
        0,
        [Preconditioning::None],
        0
    );
    check!(
        MinresSolver::new(tolerance, limit),
        71,
        0,
        [Preconditioning::None],
        0
    );
    check!(LsqrSolver::new(tolerance, limit), 68, 68, [], 0);
    check!(LsmrSolver::new(tolerance, limit), 68, 68, [], 0);
    let restart = Budget::try_from(100).unwrap();
    check!(
        GmresSolver::new(tolerance, limit).with_restart(restart),
        68,
        0,
        [Preconditioning::None],
        0
    );
    check!(
        LgmresSolver::new(tolerance, limit).with_restart(restart),
        68,
        0,
        [Preconditioning::None],
        0
    );
    check!(
        GcrotSolver::new(tolerance, limit).with_restart(restart),
        68,
        0,
        [Preconditioning::None],
        0
    );
    struct Identity;
    impl scioxide_core::Preconditioner<f64> for Identity {
        fn dimension(&self) -> usize {
            180
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y.copy_from_slice(x);
            Ok(())
        }
    }
    check!(
        MinresSolver::new(tolerance, limit),
        71,
        0,
        [Preconditioning::Inverse(&Identity)],
        66
    );
}

#[test]
fn variable_inverse_uses_shared_verification_at_restarts_and_periodic_checkpoints() {
    use scioxide_core::Preconditioner;
    use scioxide_linalg::{
        GcrotSolver, IterativeWorkspaceRequest, LgmresSolver, LinearSolveTermination,
    };
    struct Changing {
        calls: Cell<usize>,
        n: usize,
    }
    impl Preconditioner<f64> for Changing {
        fn dimension(&self) -> usize {
            self.n
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            input: &[f64],
            output: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            for (i, (out, value)) in output.iter_mut().zip(input).enumerate() {
                *out = value / (1. + ((call + i) % 3) as f64);
            }
            Ok(())
        }
    }
    let n = 120;
    let mut values = vec![0.; n * n];
    for i in 0..n {
        values[i * n + i] = (i + 1) as f64;
    }
    let matrix = Matrix::from_vec(n, n, values).unwrap();
    let tolerance = ResidualTolerance::new(0., 0.).unwrap();
    let limit = Budget::try_from(65).unwrap();
    for (restart, forward, restarts) in [(100, 68, 0), (30, 70, 2)] {
        let restart = Budget::try_from(restart).unwrap();
        let mut baseline = None;
        for history in [false, true] {
            let solvers: [&dyn IterativeLinearSolver<f64>; 3] = [
                &GmresSolver::new(tolerance, limit)
                    .with_restart(restart)
                    .with_history(history),
                &LgmresSolver::new(tolerance, limit)
                    .with_restart(restart)
                    .with_history(history),
                &GcrotSolver::new(tolerance, limit)
                    .with_restart(restart)
                    .with_history(history),
            ];
            for solver in solvers {
                let a = Counted::new(&matrix);
                let inverse = Changing {
                    calls: Cell::new(0),
                    n,
                };
                let p = Preconditioning::Inverse(&inverse);
                let request = IterativeWorkspaceRequest::operator(&a)
                    .unwrap()
                    .with_preconditioner(p)
                    .unwrap();
                let mut workspace = solver.workspace(&request).unwrap();
                let mut x = Vector::zeros(n).unwrap();
                let b = vec![1.; n];
                let report = solver
                    .solve_into(
                        &a,
                        b.as_slice().into(),
                        p,
                        x.view_mut(),
                        &mut workspace,
                        ExecutionContext::sequential().into(),
                    )
                    .unwrap();
                assert_eq!(report.termination(), LinearSolveTermination::IterationLimit);
                assert_eq!(report.iterations(), 65);
                assert_eq!(a.forward.get(), forward);
                assert_eq!(inverse.calls.get(), 65);
                let counts = workspace.last_attempt().unwrap().counts().own();
                assert_eq!(counts.provider(EvaluationRole::LinearOperator), forward);
                assert_eq!(counts.provider(EvaluationRole::Preconditioner), 65);
                assert_eq!(report.krylov().unwrap().restarts(), restarts);
                let residual = x
                    .as_slice()
                    .iter()
                    .enumerate()
                    .fold(0_f64, |r, (i, x)| r.hypot(1. - (i + 1) as f64 * x));
                assert!((report.true_residual().unwrap().raw_norm() - residual).abs() < 1e-12);
                if let Some(previous) = &baseline {
                    let previous: &Vec<f64> = previous;
                    assert!(
                        x.as_slice()
                            .iter()
                            .zip(previous)
                            .all(|(x, y)| (x - y).abs() < 1e-12)
                    );
                } else {
                    baseline = Some(x.as_slice().to_vec());
                }
            }
        }
    }
}
