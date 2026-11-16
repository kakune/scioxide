//! Exact provider budgets, poisoned storage, representation parity and matrix-free memory.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;

use scioxide_core::{
    AdjointOperator, BlockVectorView, BlockVectorViewMut, Budget, BudgetKind, EvaluationRole,
    ExecutionContext, LinearOperator, ProblemToken, SciOxideError, SolveBudgets, SolveIdentity,
    SolveState,
};
use scioxide_linalg::{
    ExponentialAction, Matrix, MatrixEquationOptions, PartialSpectrumBuffer, PartialSpectrumSolver,
    SpectralTermination, SylvesterPlan, SylvesterSign,
};
use scioxide_sparse::ExplicitZeroPolicy;
struct Diagonal {
    n: usize,
    calls: Cell<usize>,
    adjoints: Cell<usize>,
    fail: bool,
}
impl LinearOperator<f64> for Diagonal {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        for (o, &v) in y.iter_mut().zip(x) {
            *o = if self.fail { f64::NAN } else { -v };
        }
        Ok(())
    }
}
impl AdjointOperator<f64> for Diagonal {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.adjoints.set(self.adjoints.get() + 1);
        self.apply_into(x, y, scratch)
    }
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(479),
        scioxide_core::algorithm_id!("linalg", "matrix-method-resources", 1),
        0,
        0,
    )
}
#[test]
fn exact_action_budgets_and_callback_rollback() {
    let a = Diagonal {
        n: 3,
        calls: Cell::new(0),
        adjoints: Cell::new(0),
        fail: false,
    };
    let options = ExponentialAction::new(1.);
    let p = options;
    let mut w = p.workspace(&a, 1).unwrap();
    for limit in 1..12 {
        a.calls.set(0);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut out = [42.; 3];
        let report = p
            .apply_into(
                &a,
                scioxide_linalg::PhiFunction::phi(0, 1.),
                BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
                BlockVectorViewMut::new(&mut out, 3, 1).unwrap(),
                &mut w,
                &mut context,
            )
            .unwrap();
        // The action runs in a child epoch that charges the parent's shared budget.
        assert_eq!(
            a.calls.get(),
            context
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator)
        );
        assert!(a.calls.get() <= limit);
        assert_eq!(report.applications(), a.calls.get());
        if report.termination() != SpectralTermination::Converged {
            assert!(w.last_accepted().unwrap().progress()[0] < 1.);
        }
    }
    let bad = Diagonal {
        n: 3,
        calls: Cell::new(0),
        adjoints: Cell::new(0),
        fail: true,
    };
    let mut out = [42.; 3];
    assert!(
        p.apply_into(
            &bad,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, 3, 1).unwrap(),
            &mut w,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [42.; 3]);
}

#[test]
fn poisoned_equation_output_and_cross_plan_workspace() {
    let a = Matrix::identity(2).unwrap();
    let p = SylvesterPlan::new(
        a.view(),
        a.view(),
        SylvesterSign::Plus,
        MatrixEquationOptions::new(),
    )
    .unwrap();
    let mut w = p.workspace(()).unwrap();
    let mut out = Matrix::from_vec(2, 2, vec![f64::NAN; 4]).unwrap();
    p.solve_into(
        a.view(),
        out.view_mut(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out.as_slice(), &[0.5, 0., 0., 0.5]);
    let a = Matrix::<f64>::identity(4).unwrap();
    let p = PartialSpectrumSolver::new(1);
    let mut w = p.workspace(&a.view().try_finite().unwrap()).unwrap();
    let other = PartialSpectrumSolver::new(2);
    let mut result = PartialSpectrumBuffer::new(4, 2).unwrap();
    assert!(
        other
            .solve_into(
                &a.view().try_finite().unwrap(),
                [1.; 4].as_slice(),
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
}
#[test]
fn dense_sparse_view_parity() {
    let a = Matrix::from_vec(3, 3, vec![-1., 0., 0., 0., -2., 0., 0., 0., -3.]).unwrap();
    let csr = a.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let csc = csr.to_csc().unwrap();
    let av = a.view();
    let cv = csr.view();
    for op in [
        &a.view().try_finite().unwrap() as &dyn AdjointOperator<f64>,
        &av.try_finite().unwrap(),
        &csr,
        &csc,
        &cv,
    ] {
        let p = ExponentialAction::new(3.);
        let mut w = p.workspace(op, 1).unwrap();
        let mut out = [0.; 3];
        p.apply_into(
            op,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, 3, 1).unwrap(),
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - (-(i as f64 + 1.)).exp()).abs() < 1e-8);
        }
        #[cfg(feature = "std")]
        {
            let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::new(&pool), identity(), &mut budgets)
                .unwrap();
            let mut parallel = [0.; 3];
            p.apply_into(
                op,
                scioxide_linalg::PhiFunction::phi(0, 1.),
                BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
                BlockVectorViewMut::new(&mut parallel, 3, 1).unwrap(),
                &mut w,
                &mut context,
            )
            .unwrap();
            assert_eq!(out, parallel);
        }
    }
}
#[test]
fn exponential_caps_are_validated_before_execution() {
    let op = Diagonal {
        n: 1,
        calls: Cell::new(0),
        adjoints: Cell::new(0),
        fail: false,
    };
    for (terms, phi) in [(8, 20), (55, 200), (56, 0), (55, 56)] {
        let options = ExponentialAction::<f64>::new(1.0)
            .with_terms(Budget::try_from(terms).unwrap())
            .with_max_phi(phi);
        assert_eq!(
            options.workspace_requirements(&op, 1).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }
    let plan = ExponentialAction::new(1.0).with_max_phi(55);
    let mut workspace = plan.workspace(&op, 1).unwrap();
    let mut output = [0.0];
    let report = plan
        .apply_into(
            &op,
            scioxide_linalg::PhiFunction::phi(55, 0.0),
            BlockVectorView::new(&[1.0], 1, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 1, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    let expected = 1.0 / (1..=55).map(f64::from).product::<f64>();
    assert!((output[0] / expected - 1.0).abs() < 1e-13);
    assert_eq!(op.calls.get(), 0);
}

#[test]
fn partial_svd_and_spectrum_share_restart_capacity_boundaries() {
    use scioxide_linalg::PartialSvdSolver;
    for (dimension, count, basis, accepted) in [
        (8, 3, 3, false),
        (8, 3, 4, false),
        (8, 3, 5, true),
        (8, 8, 8, true),
        (8, 7, 8, true),
        (8, 9, 10, false),
        (0, 0, 1, true),
        (1, 1, 1, true),
        (8, 0, 2, true),
    ] {
        let basis = Budget::try_from(basis).unwrap();
        let square = Matrix::<f64>::zeros(dimension, dimension).unwrap();
        let spectrum = PartialSpectrumSolver::<f64>::new(count)
            .with_basis(basis)
            .workspace_requirements(&square.view().try_finite().unwrap());
        assert_eq!(
            spectrum.is_ok(),
            accepted,
            "spectrum {dimension}/{count}/{basis:?}"
        );
        for (rows, cols) in [
            (dimension, dimension),
            (dimension, dimension + 3),
            (dimension + 3, dimension),
        ] {
            let a = Matrix::<f64>::zeros(rows, cols).unwrap();
            let svd = PartialSvdSolver::<f64>::new(count)
                .with_basis(basis)
                .workspace_requirements(&a.view().try_finite().unwrap());
            assert_eq!(svd.is_ok(), accepted, "SVD {rows}x{cols}/{count}/{basis:?}");
        }
    }
}

#[test]
fn child_epochs_record_action_iterations_as_nested_work() {
    let a = Diagonal {
        n: 3,
        calls: Cell::new(0),
        adjoints: Cell::new(0),
        fail: false,
    };
    let p = ExponentialAction::new(1.);
    let mut w = p.workspace(&a, 1).unwrap();
    let mut out = [0.; 3];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = p
        .apply_into(
            &a,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, 3, 1).unwrap(),
            &mut w,
            &mut context,
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), scioxide_core::AttemptOutcome::Succeeded);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        attempt.counts().own().iterations()
    );
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        a.calls.get()
    );
}

#[test]
fn exponential_power_estimates_count_adjoint_calls_in_the_shared_budget() {
    let a = Diagonal {
        n: 8,
        calls: Cell::new(0),
        adjoints: Cell::new(0),
        fail: false,
    };
    // A deliberately loose upper bound forces power estimation on an operator
    // large enough to use the adjoint estimator instead of column enumeration.
    let plan = ExponentialAction::new(1000.);
    let mut workspace = plan.workspace(&a, 1).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut out = [42.; 8];
    let report = plan
        .apply_into(
            &a,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 8], 8, 1).unwrap(),
            BlockVectorViewMut::new(&mut out, 8, 1).unwrap(),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert!(a.adjoints.get() > 0);
    assert_eq!(report.applications(), a.calls.get());
    assert_eq!(
        a.calls.get(),
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator)
    );
    assert!(out.iter().all(|x| (x - (-1.0_f64).exp()).abs() < 1e-8));
}
