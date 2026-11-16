//! Cross-family lifecycle, resource classification and nested-stop regressions.
use core::cell::Cell;
use scioxide_core::{
    Admission, Budget, BudgetKind, ConvergenceReason, EvaluationRole, ExecutionContext, Invocation,
    JacobianOperator, OperatorContext, PartialResultTrust, ResidualOperator, ResidualTolerance,
    SciOxideError, SciOxideErrorView, SolveBudgets, SolveContext, SolveState,
};
use scioxide_linalg::{IterativeSolveReport, LinearSolveTermination, Vector};
use scioxide_rootfind::*;

#[test]
fn reserved_outputs_have_no_numerical_termination_or_residual() {
    let mut dense = SystemRootResult::<f64>::empty();
    dense.try_reserve(2).unwrap();
    let fixed = FixedPointResult::<f64>::new(2).unwrap();
    let newton = NewtonKrylovResult::<f64>::new(2, &NewtonKrylovOptions::default()).unwrap();
    let deflated = DeflatedRootResult::<f64>::new(2, &NewtonKrylovOptions::default()).unwrap();
    for (termination, norm, success, iterations) in [
        (
            dense.termination(),
            dense.residual_norm(),
            dense.is_success(),
            dense.iterations(),
        ),
        (
            fixed.termination(),
            fixed.residual_norm(),
            fixed.is_success(),
            fixed.iterations(),
        ),
        (
            newton.termination(),
            newton.residual_norm(),
            newton.is_success(),
            newton.iterations(),
        ),
        (
            deflated.termination(),
            deflated.residual_norm(),
            deflated.is_success(),
            deflated.iterations(),
        ),
    ] {
        assert_eq!(termination, RootTermination::NotStarted);
        assert_eq!(norm, None);
        assert!(!success);
        assert_eq!(iterations, 0);
    }
    assert_eq!(fixed.scaled_residual_norm(), None);
}

#[test]
fn scoped_bracket_exhaustion_preserves_the_budget_kind() {
    for (kind, expected) in [
        (BudgetKind::Iteration, ConvergenceReason::IterationLimit),
        (
            BudgetKind::Proposal,
            ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
        ),
    ] {
        let options = BracketSearchOptions::<f64>::default()
            .with_budgets(SolveBudgets::unbounded().with_limit(kind, Budget::ONE));
        let error = find_bracket_scoped(
            &ScalarFn::new(|_: f64| Ok(1.0)),
            scioxide_rootfind::BracketSearchInput::Guess(0.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &options,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::ConvergenceFailure { reason, diagnostics, .. }
            if reason == expected && diagnostics.budget().is_some_and(|r| r.kind() == kind && r.consumed() == 1))
        );
    }
}

struct ObserveStop(Cell<bool>);
impl NonlinearLinearSolver<f64> for ObserveStop {
    fn solve_into(
        &self,
        operator: &scioxide_rootfind::NonlinearLinearization<'_, f64>,
        _: &Vector<f64>,
        output: &mut Vector<f64>,
        _: ResidualTolerance<f64>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        let operator = operator.operator();
        scope.run(
            &mut SolveState::new(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "observe-stop", 1),
                1,
                0,
            ),
            SolveBudgets::unbounded(),
            |context| {
                assert!(
                    context
                        .record_provider(EvaluationRole::LinearOperator)?
                        .is_granted()
                );
                let outcome = operator.apply_into_scoped(
                    &[1.0],
                    output.as_mut_slice(),
                    &mut [],
                    &mut OperatorContext::new(context),
                )?;
                assert!(matches!(outcome, Invocation::BudgetExhausted(report)
                if report.kind() == BudgetKind::Evaluation && report.consumed() == 3));
                self.0.set(true);
                Ok(IterativeSolveReport::new(
                    LinearSolveTermination::EvaluationLimit,
                    0,
                    None,
                    PartialResultTrust::DiagnosticOnly,
                ))
            },
        )
    }
}

#[test]
fn numerical_jvp_denial_is_visible_to_injected_solvers_without_a_fake_cancellation() {
    let calls = Cell::new(0);
    let residual = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        calls.set(calls.get() + 1);
        out[0] = x[0] * x[0] - 2.0;
        Ok(())
    });
    let observer = ObserveStop(Cell::new(false));
    let problem = NewtonKrylovProblem::numerical(&residual).with_linear_solver(&observer);
    let plan = NewtonKrylovSolver::new(NewtonKrylovOptions::default().with_budgets(
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap()),
    ))
    .unwrap();
    let result = plan.solve(&problem, &[1.0]).unwrap();
    assert!(observer.0.get());
    assert_eq!(
        result.termination(),
        RootTermination::WorkLimit(BudgetKind::Evaluation)
    );
    assert!(!result.is_success());
    assert_eq!(calls.get(), 1);
    assert_eq!(result.point(), &[1.0]);
    assert_eq!(result.residual(), &[-1.0]);
    assert!((result.residual_norm().unwrap() - 1.0).abs() < f64::EPSILON);
    assert_eq!(result.counts().total_providers().unwrap(), 2);
    assert_eq!(
        result
            .accounting()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        1
    );
    assert_eq!(
        result
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap(),
        3
    );
}

#[test]
fn complex_special_zero_exhaustion_reports_work_instead_of_refinement_iterations() {
    // The first eight starts are tabulated and can finish in one paired sample.
    let options = SpecialZeroOptions::<f64>::default();
    let mut workspace = SpecialZeroWorkspace::workspace(1, &options).unwrap();
    let mut output = [None];
    let mut parent_state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut parent = parent_state
        .begin_solve(
            ExecutionContext::sequential(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "complex-zero-stop", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let report = special_zeros_into(
        SpecialZeroFamily::Erf,
        SpecialZeroBranch::UPPER,
        9,
        &options,
        &mut output,
        &mut workspace,
        &mut parent,
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        RootTermination::WorkLimit(BudgetKind::Evaluation)
    );
    assert_eq!(report.completed(), 0);
    assert!(report.candidate().is_some());
    assert!(output[0].is_none());
}

#[test]
fn continuation_stops_before_initialization_and_after_progress_keep_the_real_reason() {
    struct Line;
    impl ResidualOperator<f64> for Line {
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
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = x[0] - x[1];
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Line {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            _: &[f64],
            v: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = v[0] - v[1];
            Ok(())
        }
    }
    let problem = ContinuationProblem::new(&Line).unwrap();
    let options = ContinuationOptions::new(0.1, 1e-6, 0.2, Budget::try_from(16).unwrap()).unwrap();
    let limited = Continuation::new(
        ContinuationPath::Parameters(vec![1.0]),
        options.clone().with_budgets(
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE),
        ),
    )
    .unwrap();
    let mut state = limited.state(&problem, &[0.0, 0.0]).unwrap();
    let mut workspace = limited.workspace(&problem).unwrap();
    let error = limited
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
    assert_eq!(state.termination(), RootTermination::NotStarted);
    assert!(state.history().is_empty());
    let plan = Continuation::new(ContinuationPath::Parameters(vec![1.0]), options).unwrap();
    let mut state = plan.state(&problem, &[0.0, 0.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let mut parent_state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(BudgetKind::Evaluation, Budget::try_from(1024).unwrap());
    let mut parent = parent_state
        .begin_solve(
            ExecutionContext::sequential(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "continuation-stop", 1),
                2,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let first = plan
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(first.termination(), RootTermination::Advanced);
    let point = state.point().to_vec();
    let residual = state.residual().to_vec();
    let used = parent.counts().total().unwrap().total_providers().unwrap();
    assert!(matches!(
        parent
            .record_providers(EvaluationRole::Event, 1024 - used)
            .unwrap(),
        Admission::Granted(())
    ));
    let report = plan
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        RootTermination::WorkLimit(BudgetKind::Evaluation)
    );
    assert_eq!(state.point(), point);
    assert_eq!(state.residual(), residual);
    assert_eq!(state.history().len(), 1);
    assert_eq!(state.residual_norm(), Some(0.0));
    assert!(report.observation().is_none());
}

#[test]
fn a_provider_error_at_the_last_allowed_evaluation_is_not_a_budget_stop() {
    let calls = Cell::new(0);
    let residual = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        calls.set(calls.get() + 1);
        if calls.get() > 1 {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::EvaluationLimit,
                scioxide_core::ConvergenceDiagnostics::new(),
                "provider sentinel",
            ));
        }
        out[0] = x[0] - 2.0;
        Ok(())
    });
    let observer = ObserveStop(Cell::new(false));
    let problem = NewtonKrylovProblem::numerical(&residual).with_linear_solver(&observer);
    let plan = NewtonKrylovSolver::new(NewtonKrylovOptions::default().with_budgets(
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(4).unwrap()),
    ))
    .unwrap();
    let error = plan.solve(&problem, &[1.0]).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure { reason: ConvergenceReason::EvaluationLimit, diagnostics, detail }
        if diagnostics.budget().is_none() && detail == "provider sentinel")
    );
    assert_eq!(calls.get(), 2);
    assert!(!observer.0.get());
}

struct InnerIterationLimit;
impl NonlinearLinearSolver<f64> for InnerIterationLimit {
    fn solve_into(
        &self,
        _: &scioxide_rootfind::NonlinearLinearization<'_, f64>,
        _: &Vector<f64>,
        _: &mut Vector<f64>,
        _: ResidualTolerance<f64>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        scope.run(
            &mut SolveState::new(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "inner-limit", 1),
                1,
                0,
            ),
            SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
            |context| {
                assert!(context.record_iteration()?.is_granted());
                assert!(!context.record_iteration()?.is_granted());
                Ok(IterativeSolveReport::new(
                    LinearSolveTermination::IterationLimit,
                    1,
                    None,
                    PartialResultTrust::DiagnosticOnly,
                ))
            },
        )
    }
}

#[test]
fn a_numerical_iteration_ceiling_is_distinct_from_an_earlier_work_allowance() {
    let equation = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        out[0] = x[0] - 2.0;
        Ok(())
    });
    let problem =
        NewtonKrylovProblem::numerical(&equation).with_linear_solver(&InnerIterationLimit);
    for (ceiling, expected) in [
        (1, RootTermination::IterationLimit),
        (2, RootTermination::WorkLimit(BudgetKind::Iteration)),
    ] {
        let options = NewtonKrylovOptions::default()
            .with_max_iterations(Budget::try_from(ceiling).unwrap())
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
        let result = NewtonKrylovSolver::new(options)
            .unwrap()
            .solve(&problem, &[1.0])
            .unwrap();
        assert_eq!(result.termination(), expected);
        assert_eq!(result.iterations(), 1);
        assert_eq!(result.accounting().nested().iterations(), 1);
        assert!(!result.is_success());
    }
}
