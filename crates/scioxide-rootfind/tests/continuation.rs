//! Public continuation, homotopy and deflation behavior.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
use core::cell::Cell;
use scioxide_core::{
    Budget, Complex, EvaluationRole, ExecutionContext, JacobianOperator, LinearOperator, NdView,
    NonFiniteOrigin, NumericScalar, PartialResultTrust, ProblemToken, RealScalar, ResidualOperator,
    ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveState,
    StepTolerance,
};
use scioxide_linalg::{LinearSolveTermination, Vector};
use scioxide_rootfind::{
    Continuation, ContinuationDiagnostic, ContinuationDiagnosticProvider, ContinuationOptions,
    ContinuationPath, ContinuationProblem, ContinuationState, DeflatedProblem, DeflationOptions,
    DeflationSet, LinearHomotopy, NewtonKrylovOptions, NonlinearLinearSolver, RealifiedProblem,
    ResidualFn, RootTermination,
};

/// Accepted augmented points of `state`, one per history entry.
fn points<R: RealScalar>(state: &ContinuationState<R>) -> Vec<Vec<R>> {
    (0..state.history().len())
        .map(|i| state.history_point(i).unwrap().to_vec())
        .collect()
}

fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn options<R: RealScalar>() -> ContinuationOptions<R> {
    ContinuationOptions::new(r(0.1), r(1e-6), r(0.15), budget(300))
        .unwrap()
        .with_residual_tolerance(ResidualTolerance::new(R::EPSILON * r(100.), R::ZERO).unwrap())
}
struct Fold;
impl<R: RealScalar> ResidualOperator<R> for Fold {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[R], y: &mut [R], _: &mut [R]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0] - x[1];
        Ok(())
    }
}
impl<R: RealScalar> JacobianOperator<R> for Fold {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[R], v: &[R], y: &mut [R], _: &mut [R]) -> Result<(), SciOxideError> {
        y[0] = r::<R>(2.) * x[0] * v[0] - v[1];
        Ok(())
    }
}
fn natural<R: RealScalar>() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![r(2.), r(0.5)]), options()).unwrap();
    let path = continuation.state(&p, &[R::ONE, R::ONE]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert_eq!(path.termination(), RootTermination::Complete);
    assert_eq!(path.point()[1], r(0.5));
    assert!((path.point()[0] - r::<R>(0.5).sqrt()).abs() < r::<R>(1000.) * R::EPSILON);
    for i in 0..path.history().len() {
        let x = path.history_point(i).unwrap();
        assert!((x[0] * x[0] - x[1]).abs() < r::<R>(200.) * R::EPSILON);
    }
}
#[test]
fn natural_paths_both_precisions_and_reversal() {
    natural::<f32>();
    natural::<f64>();
}
fn fold<R: RealScalar>() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let continuation = Continuation::new(
        ContinuationPath::PseudoArclength(r(3.)),
        options().with_dense_diagnostics(r(1e-4)),
    )
    .unwrap();
    let path = continuation
        .state(&p, &[R::ONE, R::ONE])
        .unwrap()
        .with_initial_tangent(&[-R::ONE, -r::<R>(2.)])
        .unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert_eq!(path.termination(), RootTermination::Complete);
    assert!(path.point()[0] < R::ZERO);
    assert!(path.history().iter().any(|h| h.turning_point_candidate()));
    assert!(
        path.history()
            .iter()
            .all(|h| h.diagnostic().unwrap().augmented_rank() == Some(1))
    );
}
#[test]
fn pseudo_arclength_crosses_fold_in_both_precisions() {
    fold::<f32>();
    fold::<f64>();
}
struct Pitchfork;
impl ResidualOperator<f64> for Pitchfork {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = x[0] * (x[0] * x[0] - x[1]);
        Ok(())
    }
}
impl JacobianOperator<f64> for Pitchfork {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        y[0] = (3. * x[0] * x[0] - x[1]) * v[0] - x[0] * v[1];
        Ok(())
    }
}
#[test]
fn pitchfork_reports_signed_branch_candidate_without_a_fold() {
    let p = ContinuationProblem::new(&Pitchfork).unwrap();
    let continuation = Continuation::new(
        ContinuationPath::Parameters(vec![0.5]),
        options().with_dense_diagnostics(1e-8),
    )
    .unwrap();
    let path = continuation.state(&p, &[0., -0.51]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert!(path.is_success());
    assert!(
        path.history()
            .iter()
            .any(|h| h.diagnostic().unwrap().branch_point_candidate())
    );
    assert!(!path.history().iter().any(|h| h.turning_point_candidate()));
}
#[test]
fn checkpoint_restores_exact_path_and_work() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::PseudoArclength(0.9), options()).unwrap();
    let uninterrupted = continuation.state(&problem, &[1.0, 1.0]).unwrap();
    let uninterrupted = continuation.follow(&problem, uninterrupted).unwrap();
    let mut split = continuation.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = continuation.workspace(&problem).unwrap();
    for _ in 0..2 {
        continuation
            .advance(
                &problem,
                &mut split,
                |_, _| Ok(()),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
    }
    let checkpoint = split.checkpoint().unwrap();
    let mut restored = continuation.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = continuation.workspace(&problem).unwrap();
    continuation
        .resume(
            &problem,
            &checkpoint,
            &mut restored,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(
        workspace.last_attempt().unwrap().kind(),
        scioxide_core::AttemptKind::Resume
    );
    let restored = continuation.follow(&problem, restored).unwrap();
    assert_eq!(restored.point(), uninterrupted.point());
    assert_eq!(points(&restored), points(&uninterrupted));
    assert_eq!(restored.counts(), uninterrupted.counts());
    let wrong = Continuation::new(ContinuationPath::PseudoArclength(1.0), options()).unwrap();
    let mut wrong_state = wrong.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = wrong.workspace(&problem).unwrap();
    let before = checkpoint.metadata();
    assert!(
        wrong
            .resume(
                &problem,
                &checkpoint,
                &mut wrong_state,
                |_, _| Ok(()),
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(wrong_state.point(), &[1.0, 1.0]);
    assert_eq!(checkpoint.metadata(), before);
    assert_eq!(
        workspace.last_attempt().unwrap().kind(),
        scioxide_core::AttemptKind::Resume
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    assert!(workspace.last_accepted().is_none());
}
#[test]
fn exact_budget_stop_preserves_accepted_history() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let continuation = Continuation::new(
        ContinuationPath::Parameters(vec![2.]),
        options().with_budgets(
            SolveBudgets::unbounded().with_limit(scioxide_core::BudgetKind::Proposal, budget(1)),
        ),
    )
    .unwrap();
    let mut path = continuation.state(&p, &[1., 1.]).unwrap();
    let mut w = continuation.workspace(&p).unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(scioxide_core::BudgetKind::Proposal, budget(1));
    let mut c = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "continuation-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    continuation
        .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
        .unwrap();
    // One outer proposal admits its whole corrector. Fold's independent
    // equation is x² - lambda = 0, with the unchanged configured tolerance.
    assert_eq!(
        continuation
            .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .unwrap(),
        RootTermination::Advanced
    );
    assert!(path.point()[1] > 1.0);
    assert!((path.point()[0] * path.point()[0] - path.point()[1]).abs() <= 100.0 * f64::EPSILON);
    assert_eq!(w.last_accepted(), Some(path.point()));
    let before = path.point().to_vec();
    assert_eq!(
        continuation
            .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .unwrap(),
        RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal)
    );
    assert_eq!(path.point(), before);
    assert_eq!(path.history().len(), 2);
    // The stopped attempt accepted nothing; earlier progress is not its own.
    assert!(w.last_accepted().is_none());
    assert!(w.last_attempt().unwrap().progress().is_none());
}
#[test]
fn advancing_a_complete_path_records_no_new_progress() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    let state = continuation.state(&p, &[1., 1.]).unwrap();
    let mut state = continuation.follow(&p, state).unwrap();
    assert_eq!(state.termination(), RootTermination::Complete);
    let history = state.history().len();
    let mut w = continuation.workspace(&p).unwrap();
    let report = continuation
        .advance(
            &p,
            &mut state,
            |_, _| Ok(()),
            &mut w,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), RootTermination::Complete);
    assert!(report.observation().is_none());
    assert_eq!(state.history().len(), history);
    assert!(w.last_accepted().is_none());
    assert!(w.last_attempt().unwrap().progress().is_none());
}
struct Square<T> {
    constant: T,
}
impl<T: NumericScalar> ResidualOperator<T> for Square<T> {
    fn input_dimension(&self) -> usize {
        1
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[T], y: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0] - self.constant;
        Ok(())
    }
}
impl<T: NumericScalar> JacobianOperator<T> for Square<T> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[T], v: &[T], y: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        y[0] = (T::ONE + T::ONE) * x[0] * v[0];
        Ok(())
    }
}
#[test]
fn homotopy_reaches_original_target() {
    let start = Square { constant: 1. };
    let target = Square { constant: 4. };
    let h = LinearHomotopy::new(&start, &target).unwrap();
    let p = ContinuationProblem::new(&h).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![1.]), options()).unwrap();
    let path = continuation.state(&p, &[1., 0.]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert!(path.is_success());
    assert_eq!(path.point()[1], 1.);
    assert!((path.point()[0] - 2.).abs() < 1e-10);
}
fn deflation<R: RealScalar>() {
    let p = Square { constant: R::ONE };
    let mut set = DeflationSet::new(
        1,
        budget(2),
        DeflationOptions::default().with_residual_tolerance(
            ResidualTolerance::new(r::<R>(100.) * R::EPSILON, R::ZERO).unwrap(),
        ),
    )
    .unwrap();
    set.insert(&p, &[R::ONE]).unwrap();
    assert!(set.insert(&p, &[R::ONE]).is_err());
    assert!(set.insert(&p, &[r(2.)]).is_err());
    assert_eq!(set.len(), 1);
    let d = DeflatedProblem::new(
        &p,
        &set,
        NewtonKrylovOptions::default()
            .with_step_tolerance(StepTolerance::new(R::EPSILON, R::EPSILON).unwrap())
            .with_residual_tolerance(
                ResidualTolerance::new(r::<R>(100.) * R::EPSILON, R::ZERO).unwrap(),
            ),
    )
    .unwrap();

    let result = d.solve(&[r(0.2)]).unwrap();
    let raw = result.residual_report().unwrap();
    assert!(result.is_success(), "{:?}", result.termination());
    assert!((result.point()[0] + R::ONE).abs() < r::<R>(1000.) * R::EPSILON);
    assert!(raw.raw_norm() < r::<R>(200.) * R::EPSILON);
    assert!(d.solve(&[R::ONE]).is_err());
}
#[test]
fn explicit_deflation_finds_distinct_real_solutions() {
    deflation::<f32>();
    deflation::<f64>();
}
fn complex<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let original = Square {
        constant: Complex::new(-R::ONE, R::ZERO),
    };
    let real = RealifiedProblem::square(&original).unwrap();
    let mut set = DeflationSet::new(2, budget(2), DeflationOptions::default()).unwrap();
    set.insert(&real, &[R::ZERO, R::ONE]).unwrap();
    let deflated = DeflatedProblem::new(
        &real,
        &set,
        NewtonKrylovOptions::default()
            .with_step_tolerance(StepTolerance::new(R::EPSILON, R::EPSILON).unwrap())
            .with_residual_tolerance(
                ResidualTolerance::new(r::<R>(100.) * R::EPSILON, R::ZERO).unwrap(),
            ),
    )
    .unwrap();

    let result = deflated.solve(&[r(0.1), r(-0.5)]).unwrap();
    let raw = result.residual_report().unwrap();
    assert!(result.is_success(), "{:?}", result.termination());
    assert!(result.point()[0].abs() < r::<R>(1000.) * R::EPSILON);
    assert!((result.point()[1] + R::ONE).abs() < r::<R>(1000.) * R::EPSILON);
    assert!(raw.raw_norm() < r::<R>(200.) * R::EPSILON);
}
#[test]
fn complex_deflation_uses_real_linear_derivative() {
    complex::<f32>();
    complex::<f64>();
}
#[test]
fn deflated_jvp_matches_independent_difference() {
    let p = Square { constant: 1. };
    let mut set = DeflationSet::new(1, budget(2), DeflationOptions::default()).unwrap();
    set.insert(&p, &[1.]).unwrap();
    let d = DeflatedProblem::new(&p, &set, NewtonKrylovOptions::default()).unwrap();
    let mut s = vec![0.0; d.residual_workspace_len().unwrap()];
    let mut j = [0.];
    d.jvp_into(&[0.3], &[0.7], &mut j, &mut s).unwrap();
    let mut a = [0.];
    let mut b = [0.];
    d.residual_into(&[0.3 + 0.7e-5], &mut a, &mut s).unwrap();
    d.residual_into(&[0.3 - 0.7e-5], &mut b, &mut s).unwrap();
    assert!((j[0] - (a[0] - b[0]) / 2e-5).abs() < 1e-8);
}

struct ComplexLine;
impl<R: RealScalar> ResidualOperator<Complex<R>> for ComplexLine
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[Complex<R>],
        y: &mut [Complex<R>],
        _: &mut [Complex<R>],
    ) -> Result<(), SciOxideError> {
        y[0] = x[0] - Complex::new(R::ONE, r(2.)) * x[1];
        Ok(())
    }
}
impl<R: RealScalar> JacobianOperator<Complex<R>> for ComplexLine
where
    Complex<R>: NumericScalar<Real = R>,
{
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        _: &[Complex<R>],
        v: &[Complex<R>],
        y: &mut [Complex<R>],
        _: &mut [Complex<R>],
    ) -> Result<(), SciOxideError> {
        y[0] = v[0] - Complex::new(R::ONE, r(2.)) * v[1];
        Ok(())
    }
}
fn complex_path<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let adapter = RealifiedProblem::with_real_parameter(&ComplexLine).unwrap();
    let p = ContinuationProblem::new(&adapter).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![R::ONE]), options()).unwrap();
    let path = continuation.state(&p, &[R::ZERO; 3]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert!(path.is_success());
    for (&actual, expected) in path.point().iter().zip([R::ONE, r(2.), R::ONE]) {
        assert!((actual - expected).abs() < r::<R>(200.) * R::EPSILON);
    }
    let mut bad = [r(9.)];
    assert!(adapter.residual_into(&[], &mut bad, &mut []).is_err());
    assert_eq!(bad, [r(9.)]);
}
#[test]
fn complex_continuation_both_precisions() {
    complex_path::<f32>();
    complex_path::<f64>();
}
#[test]
fn strided_seed_output_and_explicit_storage_reuse() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let shape = [2];
    let stride = [2];
    let data = [1., 99., 1.];
    let view = NdView::new(
        &data,
        scioxide_core::StridedLayout::new(&shape, &stride).unwrap(),
    )
    .unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    let path = ContinuationState::from_view(&continuation, &p, view).unwrap();
    let mut path = continuation.follow(&p, path).unwrap();
    let expected = path.point().to_vec();
    let counts = path.counts();
    let mut out = [9., 42., 9.];
    path.copy_point_into(
        scioxide_core::NdViewMut::new(
            &mut out,
            scioxide_core::StridedLayout::new(&shape, &stride).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out, [expected[0], 42., expected[1]]);
    path.reset(&[1., 1.], None).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert_eq!(path.point(), expected);
    assert_eq!(path.counts(), counts);
}
#[test]
fn oversized_steps_reject_without_publishing_trials() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let options = ContinuationOptions::new(2., 1e-5, 2., budget(300))
        .unwrap()
        .with_residual_tolerance(ResidualTolerance::new(1e-9, 0.).unwrap())
        .with_corrector(
            NewtonKrylovOptions::default()
                .with_max_iterations(budget(2))
                .with_step_tolerance(StepTolerance::new(1e-15, 0.).unwrap())
                .with_residual_tolerance(ResidualTolerance::new(1e-9, 0.).unwrap()),
        );
    let continuation = Continuation::new(ContinuationPath::Parameters(vec![2.]), options).unwrap();
    let path = continuation.state(&p, &[1., 1.]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert!(path.is_success(), "{:?}", path.termination());
    assert!(path.history().iter().any(|h| h.rejected_steps() > 0));
    for point in points(&path) {
        assert!((point[0] * point[0] - point[1]).abs() < 1e-9);
    }
}
struct Failing {
    fail: Cell<bool>,
    calls: Cell<usize>,
}
impl ResidualOperator<f64> for Failing {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        y[0] = 99.;
        if self.fail.get() {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "injected callback failure",
            ))
        } else {
            y[0] = x[0] * x[0] - x[1];
            Ok(())
        }
    }
}
impl JacobianOperator<f64> for Failing {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        s: &mut [f64],
    ) -> Result<(), SciOxideError> {
        Fold.jvp_into(x, v, y, s)
    }
}
#[test]
fn failed_provider_preserves_history_and_workspace_can_continue() {
    let f = Failing {
        fail: Cell::new(false),
        calls: Cell::new(0),
    };
    let p = ContinuationProblem::new(&f).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    let mut path = continuation.state(&p, &[1., 1.]).unwrap();
    let mut w = continuation.workspace(&p).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut c = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "continuation-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    continuation
        .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
        .unwrap();
    let before = points(&path);
    f.fail.set(true);
    assert!(
        continuation
            .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .is_err()
    );
    assert_eq!(points(&path), before);
    assert_eq!(
        c.counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Residual),
        f.calls.get()
    );
    f.fail.set(false);
    assert_eq!(
        continuation
            .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .unwrap(),
        RootTermination::Advanced
    );
}
#[test]
fn evaluation_limits_include_nested_homotopy_work() {
    let a = Square { constant: 1. };
    let b = Square { constant: 4. };
    let homotopy = LinearHomotopy::new(&a, &b).unwrap();
    let p = ContinuationProblem::new(&homotopy).unwrap();
    let mut tested = 0;
    for limit in 1..100 {
        let continuation =
            Continuation::new(ContinuationPath::Parameters(vec![1.]), options()).unwrap();
        let mut path = continuation.state(&p, &[1., 0.]).unwrap();
        let mut w = continuation.workspace(&p).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(scioxide_core::BudgetKind::Evaluation, budget(limit));
        let mut c = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(0),
                    scioxide_core::algorithm_id!("test", "continuation-parent", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        if continuation
            .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
            .is_ok()
        {
            let point = path.point().to_vec();
            let outcome = continuation
                .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
                .map(|report| report.termination());
            assert!(outcome.is_ok(), "limit={limit}: {outcome:?}");
            if path.termination()
                == RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
            {
                tested += 1;
                assert_eq!(path.point(), point);
            }
        }
        let used = [
            EvaluationRole::Residual,
            EvaluationRole::LinearOperator,
            EvaluationRole::LinearSolve,
            EvaluationRole::Preconditioner,
        ]
        .iter()
        .map(|&r| c.counts().own().provider(r))
        .sum::<usize>();
        assert!(used <= limit);
    }
    assert!(tested > 10);
}
#[cfg(feature = "std")]
#[test]
fn injected_executor_and_cancellation_preserve_contracts() {
    use core::num::NonZeroUsize;
    use scioxide_core::{CancellationToken, ThreadExecutor};
    let pool = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let p = ContinuationProblem::new(&Fold).unwrap();
    let make = || {
        Continuation::new(ContinuationPath::PseudoArclength(0.4), options())
            .and_then(|continuation| {
                let state = continuation.state(&p, &[1., 1.])?;
                Ok((continuation, state))
            })
            .unwrap()
    };
    let (sequential_continuation, sequential) = make();
    let sequential = sequential_continuation.follow(&p, sequential).unwrap();
    let (parallel_continuation, mut parallel) = make();
    let mut w = parallel_continuation.workspace(&p).unwrap();
    let cancel = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut c = state
        .begin_solve(
            ExecutionContext::new(&pool).with_cancellation(&cancel),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "continuation-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    parallel_continuation
        .advance(&p, &mut parallel, |_, _| Ok(()), &mut w, &mut c)
        .unwrap();
    while parallel.termination() == RootTermination::Advanced {
        parallel_continuation
            .advance(&p, &mut parallel, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .unwrap();
    }
    assert_eq!(parallel.point(), sequential.point());
    assert_eq!(parallel.counts(), sequential.counts());
    drop(c);
    parallel.reset(&[1., 1.], None).unwrap();
    let mut c = state
        .begin_solve(
            ExecutionContext::new(&pool).with_cancellation(&cancel),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "continuation-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    parallel_continuation
        .advance(&p, &mut parallel, |_, _| Ok(()), &mut w, &mut c)
        .unwrap();
    cancel.cancel();
    assert_eq!(
        parallel_continuation
            .advance(&p, &mut parallel, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .unwrap(),
        RootTermination::Cancelled
    );
    assert_eq!(parallel.point(), &[1., 1.]);
}
#[test]
fn invalid_and_exhausted_resources_are_explicit() {
    assert!(ContinuationOptions::new(0., 1e-4, 1., budget(3)).is_err());
    assert!(ContinuationOptions::new(1., 2., 3., budget(3)).is_err());
    let p = ContinuationProblem::new(&Fold).unwrap();
    assert!(
        Continuation::new(ContinuationPath::Parameters(vec![f64::NAN]), options())
            .and_then(|continuation| {
                let state = continuation.state(&p, &[1., 1.])?;
                Ok((continuation, state))
            })
            .is_err()
    );
    let continuation = Continuation::new(
        ContinuationPath::Parameters(vec![2.]),
        ContinuationOptions::new(0.1, 0.01, 0.1, budget(2)).unwrap(),
    )
    .unwrap();
    let path = continuation.state(&p, &[1., 1.]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert_eq!(path.termination(), RootTermination::HistoryLimit);
    assert_eq!(path.history().len(), 2);
    let p = Square { constant: 1. };
    let mut set = DeflationSet::new(1, budget(1), DeflationOptions::default()).unwrap();
    set.insert(&p, &[1.]).unwrap();
    assert!(set.insert(&p, &[-1.]).is_err());
    assert_eq!(set.len(), 1);
}

#[test]
fn dense_sparse_views_and_operator_paths_agree_near_singularity() {
    use scioxide_linalg::Matrix;
    use scioxide_sparse::ExplicitZeroPolicy;
    struct Map<'a>(&'a dyn LinearOperator<f64>);
    impl ResidualOperator<f64> for Map<'_> {
        fn input_dimension(&self) -> usize {
            4
        }
        fn output_dimension(&self) -> usize {
            3
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            self.0.apply_workspace_len()
        }
        fn residual_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.0.apply_into(x, y, s)
        }
    }
    impl JacobianOperator<f64> for Map<'_> {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            self.0.apply_workspace_len()
        }
        fn jvp_into(
            &self,
            _: &[f64],
            v: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.0.apply_into(v, y, s)
        }
    }
    let matrix = Matrix::from_vec(
        3,
        4,
        vec![2., 0., 0., -2., 0., 1e-4, 0., -2e-4, 0., 0., 3., -9.],
    )
    .unwrap();
    let view = matrix.view();
    let csr = matrix.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let csc = csr.to_csc().unwrap();
    let borrowed = csr.view();
    for operator in [
        &matrix.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &view.try_finite().unwrap(),
        &csr,
        &csc,
        &borrowed,
    ] {
        let map = Map(operator);
        let p = ContinuationProblem::new(&map).unwrap();
        let continuation =
            Continuation::new(ContinuationPath::Parameters(vec![1.]), options()).unwrap();
        let path = continuation.state(&p, &[0.; 4]).unwrap();
        let path = continuation.follow(&p, path).unwrap();
        assert!(path.is_success());
        for (a, b) in path.point().iter().zip([1., 2., 3., 1.]) {
            assert!((a - b).abs() < 1e-8);
        }
        assert!(path.history().iter().all(|h| h.residual_norm() < 1e-10));
    }
}
#[test]
fn injected_linear_solver_and_diagnostic_are_used_and_checked() {
    use scioxide_linalg::IterativeSolveReport;
    struct Direct(Cell<usize>);
    impl NonlinearLinearSolver<f64> for Direct {
        fn solve_into(
            &self,
            a: &scioxide_rootfind::NonlinearLinearization<'_, f64>,
            b: &Vector<f64>,
            x: &mut Vector<f64>,
            _: ResidualTolerance<f64>,
            context: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
            let scope = scioxide_core::SolveScope::from(context);
            let a = a.operator();
            let mut state = SolveState::new();
            let identity = SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "correction", 1),
                a.cols() as u128,
                0,
            );
            scope.run(&mut state, identity, SolveBudgets::unbounded(), |c| {
                let mut first = [0.; 2];
                let mut second = [0.; 2];
                assert!(
                    c.record_provider(EvaluationRole::LinearOperator)?
                        .is_granted()
                );
                a.apply_into_scoped(
                    &[1., 0.],
                    &mut first,
                    &mut [],
                    &mut scioxide_core::OperatorContext::new(c),
                )
                .and_then(scioxide_core::Invocation::require)?;
                assert!(
                    c.record_provider(EvaluationRole::LinearOperator)?
                        .is_granted()
                );
                a.apply_into_scoped(
                    &[0., 1.],
                    &mut second,
                    &mut [],
                    &mut scioxide_core::OperatorContext::new(c),
                )
                .and_then(scioxide_core::Invocation::require)?;
                self.0.set(self.0.get() + 1);
                let det = first[0] * second[1] - second[0] * first[1];
                let rhs = b.as_slice();
                x.as_mut_slice()[0] = (second[1] * rhs[0] - second[0] * rhs[1]) / det;
                x.as_mut_slice()[1] = (first[0] * rhs[1] - first[1] * rhs[0]) / det;
                Ok(IterativeSolveReport::new(
                    LinearSolveTermination::Converged,
                    0,
                    None,
                    PartialResultTrust::DiagnosticOnly,
                ))
            })
        }
    }
    struct Diagnostic(Cell<usize>);
    impl ContinuationDiagnosticProvider<f64> for Diagnostic {
        fn evaluate(
            &self,
            _: &dyn JacobianOperator<f64>,
            _: &[f64],
            _: &[f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<ContinuationDiagnostic<f64>, SciOxideError> {
            self.0.set(self.0.get() + 1);
            ContinuationDiagnostic::new(Some(1), None, None, false)
        }
    }
    let direct = Direct(Cell::new(0));
    let diagnostic = Diagnostic(Cell::new(0));
    let p = ContinuationProblem::new(&Fold)
        .unwrap()
        .with_linear_solver(&direct)
        .with_diagnostic(&diagnostic);
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    let path = continuation.state(&p, &[1., 1.]).unwrap();
    let path = continuation.follow(&p, path).unwrap();
    assert!(path.is_success());
    assert!(direct.0.get() > path.history().len());
    assert_eq!(diagnostic.0.get(), path.history().len());
    assert_eq!(
        path.counts().provider(EvaluationRole::Event),
        diagnostic.0.get()
    );
}
#[test]
fn deflation_multiple_exclusions_and_small_shift_cannot_fake_success() {
    struct Cubic;
    impl ResidualOperator<f64> for Cubic {
        fn input_dimension(&self) -> usize {
            1
        }
        fn output_dimension(&self) -> usize {
            1
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y[0] = x[0] * (x[0] - 1.) * (x[0] + 1.);
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Cubic {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            x: &[f64],
            v: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y[0] = (3. * x[0] * x[0] - 1.) * v[0];
            Ok(())
        }
    }
    let mut set = DeflationSet::new(1, budget(3), DeflationOptions::default()).unwrap();
    set.insert(&Cubic, &[0.]).unwrap();
    set.insert(&Cubic, &[1.]).unwrap();
    let d = DeflatedProblem::new(&Cubic, &set, NewtonKrylovOptions::default()).unwrap();
    let result = d.solve(&[-0.5]).unwrap();
    let raw = result.residual_report().unwrap();
    assert!(result.is_success());
    assert!((result.point()[0] + 1.).abs() < 1e-7);
    assert!(raw.raw_norm() < 1e-7);
    let constant = Square { constant: 1. };
    let mut set = DeflationSet::new(
        1,
        budget(1),
        DeflationOptions::default().with_shifted_power(4, 1e-30),
    )
    .unwrap();
    set.insert(&constant, &[1.]).unwrap();
    let d = DeflatedProblem::new(
        &constant,
        &set,
        NewtonKrylovOptions::default()
            .with_residual_tolerance(ResidualTolerance::new(1e-8, 0.).unwrap()),
    )
    .unwrap();
    let result = d.solve(&[1e8]).unwrap();
    let raw = result.residual_report().unwrap();
    assert!(!result.is_success());
    assert_eq!(result.termination(), RootTermination::Unverified);
    assert_eq!(
        result.inner_termination(),
        RootTermination::ResidualTolerance
    );
    assert!(result.inner_residual_report().unwrap().raw_norm() < 1e-8);
    assert!(raw.raw_norm() > 1e15);
}

#[test]
fn nonfinite_input_and_callback_provenance_are_distinct() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    let result = continuation.state(&p, &[f64::NAN, 1.]);
    assert!(matches!(
        result.err().unwrap().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    struct Nonfinite;
    impl ResidualOperator<f64> for Nonfinite {
        fn input_dimension(&self) -> usize {
            2
        }
        fn output_dimension(&self) -> usize {
            1
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = f64::NAN;
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Nonfinite {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            x: &[f64],
            v: &[f64],
            out: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            Fold.jvp_into(x, v, out, s)
        }
    }
    let p = ContinuationProblem::new(&Nonfinite).unwrap();
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    let mut path = continuation.state(&p, &[1., 1.]).unwrap();
    let mut workspace = continuation.workspace(&p).unwrap();
    assert!(
        continuation
            .advance(
                &p,
                &mut path,
                |_, _| Ok(()),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .is_err()
    );
    assert!(path.history().is_empty());
    assert_eq!(path.point(), &[1., 1.]);
    let failure = continuation.follow(&p, path).unwrap_err();
    assert!(matches!(
        failure.error().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));
    assert!(failure.progress().is_none());
}

struct ScaledProblem<P> {
    provider: P,
    scale: f64,
}
impl<P: ResidualOperator<f64>> ResidualOperator<f64> for ScaledProblem<P> {
    fn input_dimension(&self) -> usize {
        self.provider.input_dimension()
    }
    fn output_dimension(&self) -> usize {
        self.provider.output_dimension()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider.residual_workspace_len()
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], s: &mut [f64]) -> Result<(), SciOxideError> {
        self.provider.residual_into(x, y, s)?;
        y.iter_mut().for_each(|v| *v *= self.scale);
        Ok(())
    }
}
impl<P: JacobianOperator<f64>> JacobianOperator<f64> for ScaledProblem<P> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider.jvp_workspace_len()
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        s: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.provider.jvp_into(x, v, y, s)?;
        y.iter_mut().for_each(|v| *v *= self.scale);
        Ok(())
    }
}
#[test]
fn continuation_defaults_correct_small_scale_seed_before_acceptance() {
    let scaled = ScaledProblem {
        provider: Fold,
        scale: 1e-9,
    };
    let p = ContinuationProblem::new(&scaled).unwrap();
    let opts = ContinuationOptions::new(0.1, 1e-6, 0.15, budget(30)).unwrap();
    let continuation = Continuation::new(ContinuationPath::Parameters(vec![1.1]), opts).unwrap();
    let mut path = continuation.state(&p, &[0.5, 1.0]).unwrap();
    let mut w = continuation.workspace(&p).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut c = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "continuation-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    continuation
        .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
        .unwrap();
    assert!((path.point()[0] - 1.0).abs() < 1e-7);
    while path.termination() == RootTermination::Advanced {
        continuation
            .advance(&p, &mut path, |_, _| Ok(()), &mut w, &mut c)
            .map(|report| report.termination())
            .unwrap();
    }
    assert!(path.is_success());
    assert!((path.point()[0] * path.point()[0] - 1.1).abs() < 1e-7);
}
#[test]
fn deflation_insertion_requires_explicit_evidence_for_approximate_roots() {
    let p = ScaledProblem {
        provider: Square { constant: 1.0 },
        scale: 1e-9,
    };
    let mut set = DeflationSet::new(1, budget(2), DeflationOptions::default()).unwrap();
    assert!(set.insert(&p, &[0.0]).is_err());
    assert!(set.insert(&p, &[1.0 + 1e-10]).is_err());
    assert!(set.is_empty());
    set.insert(&p, &[1.0]).unwrap();
    let mut explicit = DeflationSet::new(
        1,
        budget(2),
        DeflationOptions::default().with_residual_scale(1e-9),
    )
    .unwrap();
    assert!(explicit.insert(&p, &[0.0]).is_err());
    explicit.insert(&p, &[1.0 + 1e-10]).unwrap();
    let mut absolute = DeflationSet::new(
        1,
        budget(1),
        DeflationOptions::default()
            .with_residual_tolerance(ResidualTolerance::new(1e-18, 0.0).unwrap()),
    )
    .unwrap();
    absolute.insert(&p, &[1.0 + 1e-10]).unwrap();
}

#[test]
fn deflation_insertion_records_every_attempt_through_the_shared_shell() {
    use scioxide_core::AttemptOutcome;
    let p = Square { constant: 1.0 };
    let mut set = DeflationSet::new(1, budget(2), DeflationOptions::default()).unwrap();
    let mut workspace = set.workspace(&p).unwrap();
    let insert = |set: &mut DeflationSet<f64>, root: &[f64], workspace: &mut _| {
        set.insert_in_place(&p, root, workspace, ExecutionContext::sequential_native())
    };
    // Shape rejection happens before any epoch.
    assert!(insert(&mut set, &[1.0, 2.0], &mut workspace).is_err());
    let record = workspace.last_attempt().unwrap();
    assert_eq!(record.outcome(), AttemptOutcome::PreflightRejected);
    assert!(record.epoch().is_none() && record.progress().is_none());
    // A residual rejection inside the epoch fails without progress or insertion.
    assert!(insert(&mut set, &[0.5], &mut workspace).is_err());
    let record = workspace.last_attempt().unwrap();
    assert_eq!(record.outcome(), AttemptOutcome::Failed);
    assert!(record.epoch().is_some() && record.progress().is_none());
    assert_eq!(
        record.counts().total().unwrap().total_providers().unwrap(),
        1
    );
    assert!(workspace.last_accepted().is_none() && set.is_empty());
    // An accepted root succeeds with its original residual report.
    insert(&mut set, &[1.0], &mut workspace).unwrap();
    let record = workspace.last_attempt().unwrap();
    assert_eq!(record.outcome(), AttemptOutcome::Succeeded);
    assert!(record.epoch().is_some());
    assert_eq!(record.progress().unwrap().raw_norm(), 0.0);
    assert_eq!(workspace.last_accepted().unwrap(), &[1.0]);
    assert_eq!(set.len(), 1);
}

/// Advances `state` with one workspace while the termination is `Advanced`.
fn advance_to_end<R: RealScalar>(
    continuation: &Continuation<R>,
    problem: &ContinuationProblem<'_, R>,
    mut state: ContinuationState<R>,
) -> ContinuationState<R> {
    let mut workspace = continuation.workspace(problem).unwrap();
    while continuation
        .advance(
            problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap()
        .termination()
        == RootTermination::Advanced
    {}
    state
}

fn follow_matches_advance<R: RealScalar>() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    let natural =
        Continuation::new(ContinuationPath::Parameters(vec![r(2.), r(0.5)]), options()).unwrap();
    let arclength = Continuation::new(ContinuationPath::PseudoArclength(r(3.)), options()).unwrap();
    for (continuation, tangent) in [
        (natural, [R::ZERO, R::ONE]),
        (arclength, [-R::ONE, -r::<R>(2.)]),
    ] {
        let seed = || {
            continuation
                .state(&p, &[R::ONE, R::ONE])
                .unwrap()
                .with_initial_tangent(&tangent)
                .unwrap()
        };
        let expected = advance_to_end(&continuation, &p, seed());
        let followed = continuation.follow(&p, seed()).unwrap();
        assert_eq!(followed.termination(), RootTermination::Complete);
        assert_eq!(followed.termination(), expected.termination());
        assert!(followed.history().len() > 2);
        assert_eq!(points(&followed), points(&expected));
        assert_eq!(followed.tangent(), expected.tangent());
        assert_eq!(followed.residual(), expected.residual());
        assert_eq!(followed.accounting(), expected.accounting());
    }
}
#[test]
fn follow_reproduces_the_advance_loop_point_for_point() {
    follow_matches_advance::<f32>();
    follow_matches_advance::<f64>();
}

#[test]
fn follow_returns_unsuccessful_terminations_as_values() {
    let p = ContinuationProblem::new(&Fold).unwrap();
    // Two points fill the history long before λ = 2.
    let short = Continuation::new(
        ContinuationPath::Parameters(vec![2.]),
        ContinuationOptions::new(0.1, 0.01, 0.1, budget(2)).unwrap(),
    )
    .unwrap();
    let path = short
        .follow(&p, short.state(&p, &[1., 1.]).unwrap())
        .unwrap();
    assert_eq!(path.termination(), RootTermination::HistoryLimit);
    assert!(!path.is_success());
    assert_eq!(path.history().len(), 2);
    // Steps of 1e-30 are rejected down to the minimum step after the seed.
    let tiny = Continuation::new(
        ContinuationPath::PseudoArclength(1.),
        ContinuationOptions::new(1e-30, 1e-31, 1e-30, budget(3)).unwrap(),
    )
    .unwrap();
    let path = tiny.follow(&p, tiny.state(&p, &[1., 1.]).unwrap()).unwrap();
    assert_eq!(path.termination(), RootTermination::StepLimit);
    assert_eq!(path.history().len(), 1);
}

/// `F(x, λ) = x² − λ`, whose residual fails for `λ > limit`.
fn guarded(limit: f64) -> impl JacobianOperator<f64> {
    ResidualFn::with_parameter(1, move |y: &[f64], out: &mut [f64]| {
        if y[1] > limit {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "injected callback failure",
            ));
        }
        out[0] = y[0] * y[0] - y[1];
        Ok(())
    })
    .with_jvp(|y: &[f64], v: &[f64], out: &mut [f64]| {
        out[0] = 2. * y[0] * v[0] - v[1];
        Ok(())
    })
}

fn injected(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::Inconsistent,
            ..
        }
    ) && error.to_string().contains("injected callback failure")
}

#[test]
fn follow_returns_accepted_points_with_a_callback_error() {
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![2.]), options()).unwrap();
    // The seed is corrected at λ = 1; the first step predicts beyond λ = 1.05.
    let after_seed = guarded(1.05);
    let p = ContinuationProblem::new(&after_seed).unwrap();
    let failure = continuation
        .follow(&p, continuation.state(&p, &[1., 1.]).unwrap())
        .unwrap_err();
    assert!(injected(failure.error()), "{:?}", failure.error());
    let progress = failure.progress().unwrap();
    assert_eq!(progress.history().len(), 1);
    let point = progress.history_point(0).unwrap();
    assert!((point[0] - 1.).abs() < 1e-12 && (point[1] - 1.).abs() < 1e-12);
    assert_eq!(progress.point(), point);
    assert!(injected(&SciOxideError::from(failure)));
    // Without an accepted point the failure carries no progress.
    let before_seed = guarded(0.5);
    let p = ContinuationProblem::new(&before_seed).unwrap();
    let failure = continuation
        .follow(&p, continuation.state(&p, &[1., 1.]).unwrap())
        .unwrap_err();
    assert!(injected(failure.error()), "{:?}", failure.error());
    assert!(failure.progress().is_none());
}
