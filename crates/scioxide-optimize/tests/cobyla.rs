#![allow(clippy::panic)]

use scioxide_core::{
    Bounds, Budget, BudgetKind, EvaluationRole, ExecutionContext, ProblemToken, RealScalar,
    SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    Cobyla, CobylaOptions, ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFn,
    ObjectiveFunction, OptimizeResult, OptimizeTermination,
};

/// Minimize `objective` subject to `constraints` with the owned form.
fn solve<S: RealScalar + scioxide_core::NumericScalar<Real = S>>(
    solver: &Cobyla<S>,
    objective: &dyn ObjectiveFunction<S>,
    constraints: &Constraints<S>,
    x0: &[S],
) -> Result<OptimizeResult<S>, SciOxideError> {
    solver.minimize(&ConstrainedProblem::new(objective, constraints)?, x0)
}

macro_rules! cases {
    ($name:ident, $s:ty, $error:expr) => {
        #[test]
        fn $name() -> Result<(), SciOxideError> {
            let objective = ObjectiveFn::new(1, |x: &[$s]| Ok((x[0] - 2.0).powi(2)));
            let free = Constraints::new(1)?;
            let solver = Cobyla::<$s>::default();
            let result = solve(&solver, &objective, &free, &[0.0])?;
            assert!(result.is_success(), "{result:?}");
            assert!((result.x()[0] - 2.0).abs() < $error);
            assert_eq!(result.stationarity(), None);
            assert!(cobyla_of(&result).poised());

            let lower =
                Constraints::new(1)?.inequality(ConstraintFn::new(1, |x: &[$s]| Ok(1.0 - x[0])))?;
            let result = solve(&solver, &objective, &lower, &[0.0])?;
            assert!(result.is_success(), "{result:?}");
            assert!((result.x()[0] - 2.0).abs() < $error);
            assert!(result.primal_feasibility().unwrap() <= $error);

            let equality =
                Constraints::new(1)?.equality(ConstraintFn::new(1, |x: &[$s]| Ok(x[0] - 1.0)))?;
            let result = solve(&solver, &objective, &equality, &[0.0])?;
            assert!(result.is_success(), "{result:?}");
            assert!((result.x()[0] - 1.0).abs() < $error);

            let fixed = Bounds::new(vec![(Some(1.0), Some(1.0))])?;
            let constraints = Constraints::new(1)?.bounds(fixed)?;
            let result = solve(&solver, &objective, &constraints, &[0.0])?;
            assert_eq!(result.termination(), OptimizeTermination::Converged);
            assert_eq!(result.x(), &[1.0]);

            let narrow =
                Constraints::new(1)?.bounds(Bounds::new(vec![(Some(1.0), Some(1.001))])?)?;
            let result = solve(&solver, &objective, &narrow, &[0.0])?;
            assert!(result.is_success(), "{result:?}");
            assert!((result.x()[0] - 1.001).abs() < $error);

            let nonsmooth = ObjectiveFn::new(1, |x: &[$s]| Ok((x[0] - 0.3).abs()));
            let result = solve(&solver, &nonsmooth, &free, &[0.8])?;
            assert_eq!(
                result.termination(),
                if <$s>::MANTISSA_DIGITS == 24 {
                    OptimizeTermination::Converged
                } else {
                    OptimizeTermination::Stagnation
                }
            );
            assert!((result.x()[0] - 0.3).abs() < $error);
            Ok(())
        }
    };
}

cases!(cobyla_f64, f64, 0.02);
cases!(cobyla_f32, f32, 0.03);

#[test]
fn two_dimensional_active_set_and_original_feasibility() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
    let constraints = Constraints::new(2)?
        .inequality(ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1])))?
        .bounds(Bounds::new(vec![
            (Some(0.0), Some(1.0)),
            (Some(0.0), Some(1.0)),
        ])?)?;
    let result = solve(&Cobyla::default(), &objective, &constraints, &[0.0, 0.0])?;
    assert!(result.is_success(), "{result:?}");
    assert!(result.primal_feasibility().unwrap() <= 1e-4);
    assert!((result.x()[0] - 0.5).abs() <= 0.001, "{result:?}");
    assert!((result.x()[1] - 0.5).abs() <= 0.001, "{result:?}");
    // Independent analytic KKT check: grad f = lambda (1,1) at the active row.
    assert!(
        (2.0 * result.x()[0] - 2.0 * result.x()[1]).abs() <= 0.001,
        "{result:?}"
    );
    assert!(cobyla_of(&result).model_error().unwrap().is_finite());
    assert!(cobyla_of(&result).model_stationarity().unwrap() < 10.0);
    Ok(())
}

#[test]
fn f32_two_dimensional_active_set() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f32]| Ok(x[0] * x[0] + x[1] * x[1]));
    let constraints = Constraints::new(2)?
        .inequality(ConstraintFn::new(2, |x: &[f32]| Ok(1.0 - x[0] - x[1])))?
        .bounds(Bounds::new(vec![
            (Some(0.0), Some(1.0)),
            (Some(0.0), Some(1.0)),
        ])?)?;
    let result = solve(&Cobyla::default(), &objective, &constraints, &[0.0, 0.0])?;
    assert!(result.is_success(), "{result:?}");
    assert!(result.primal_feasibility().unwrap_or(f32::INFINITY) <= 5e-4);
    assert!((result.x()[0] - 0.5).abs() <= 0.02);
    assert!((result.x()[1] - 0.5).abs() <= 0.02);
    Ok(())
}

#[test]
fn mixed_fixed_coordinates_keep_a_full_rank_free_model() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    });
    let bounds = Bounds::new(vec![(Some(1.0), Some(1.0)), (Some(0.0), Some(3.0))])?;
    let constraints = Constraints::new(2)?.bounds(bounds)?;
    let result = solve(&Cobyla::default(), &objective, &constraints, &[0.0, 0.0])?;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(result.x()[0], 1.0);
    assert!((result.x()[1] - 2.0).abs() < 0.02);
    assert!(cobyla_of(&result).poised());
    Ok(())
}

#[test]
fn injected_budget_stops_after_admitted_center() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 2.0).powi(2)));
    let constraints = Constraints::new(1)?;
    let problem = ConstrainedProblem::new(&objective, &constraints)?;
    let solver = Cobyla::<f64>::default();
    let mut workspace = solver.workspace(&problem)?;
    let mut result = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2)?);
    let mut parent = state.begin_solve(
        ExecutionContext::sequential(),
        SolveIdentity::new(
            ProblemToken::new(658),
            scioxide_core::algorithm_id!("test", "CobylaParent", 1),
            1,
            0,
        ),
        &mut budgets,
    )?;
    solver.minimize_into(&problem, &[0.0], &mut result, &mut workspace, &mut parent)?;
    assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(result.x(), &[0.0]);
    assert_eq!(result.counts().own().provider(EvaluationRole::Objective), 1);
    assert_eq!(
        parent.counts().total()?.provider(EvaluationRole::Objective),
        1
    );
    Ok(())
}

#[test]
fn malformed_configuration_fails_before_callbacks() -> Result<(), SciOxideError> {
    assert!(Cobyla::new(CobylaOptions::<f64>::default().with_radii(0.0, 1e-4)).is_err());
    assert!(Cobyla::new(CobylaOptions::<f64>::default().with_radii(1e-4, 1.0)).is_err());
    let objective = ObjectiveFn::new(1, |_x: &[f64]| Ok(0.0));
    let constraints = Constraints::<f64>::new(2)?;
    assert!(ConstrainedProblem::new(&objective, &constraints).is_err());
    Ok(())
}

#[test]
fn budget_and_callback_failure() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 2.0).powi(2)));
    let free = Constraints::new(1)?;
    let limited = Cobyla::new(CobylaOptions::default().with_max_evaluations(Budget::try_from(1)?))?;
    let result = solve(&limited, &objective, &free, &[0.0])?;
    assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(result.x(), &[0.0]);
    assert_eq!(result.fun_evals(), 1);

    let failing = ObjectiveFn::new(1, |x: &[f64]| {
        if x[0] > 0.0 {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "trial failure",
            ))
        } else {
            Ok(1.0)
        }
    });
    let problem = ConstrainedProblem::new(&failing, &free)?;
    let solver = Cobyla::<f64>::default();
    let mut workspace = solver.workspace(&problem)?;
    let mut output = OptimizeResult::empty();
    assert!(
        solver
            .minimize_into(
                &problem,
                &[0.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output.termination(), OptimizeTermination::NotStarted);
    Ok(())
}

fn cobyla_of<S: scioxide_core::RealScalar>(
    result: &scioxide_optimize::OptimizeResult<S>,
) -> scioxide_optimize::CobylaDiagnostics<S> {
    match result.method_diagnostics() {
        scioxide_optimize::OptimizeDiagnostics::Cobyla(diagnostics) => diagnostics,
        other => panic!("unexpected method diagnostics {other:?}"),
    }
}
