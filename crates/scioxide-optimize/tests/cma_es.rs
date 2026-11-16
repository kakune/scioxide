//! Independent CMA-ES accuracy, resource, restart, and replay tests.
#![allow(clippy::unwrap_used)]

#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;

use scioxide_core::{
    AttemptOutcome, Bounds, Budget, BudgetKind, ExecutionContext, ExecutionPolicy, ProblemToken,
    ResidualTolerance, SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    BorrowedStartDesign, BoundedObjective, CmaEs, CmaEsOptions, CmaEsRestartPolicy,
    LocalSolverFactory, ObjectiveFn, OptimizeResult, OptimizeTermination, PowellLocalFactory,
    SearchLocalSolver,
};

fn parent(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        2,
        0,
    )
}

fn options<S: scioxide_core::RealScalar>(generations: usize) -> CmaEsOptions<S> {
    CmaEsOptions::default()
        .with_population_size(Budget::try_from(12).unwrap())
        .with_max_generations(Budget::try_from(generations).unwrap())
        .with_fitness_tolerance(ResidualTolerance::new(S::ZERO, S::ZERO).unwrap())
}

fn solver<S>(generations: usize) -> CmaEs<S>
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
{
    CmaEs::new(options(generations)).unwrap()
}

fn unit_box<S: scioxide_core::RealScalar>() -> Bounds<S> {
    Bounds::new(vec![(Some(-S::ONE), Some(S::ONE)); 2]).unwrap()
}

fn rotated<S: scioxide_core::RealScalar>(x: &[S]) -> S {
    let a = x[0] - S::try_from_f64_lossy(0.25).unwrap();
    let b = x[1] + S::try_from_f64_lossy(0.4).unwrap();
    let u = a + b;
    let v = a - b;
    u * u + S::try_from_f64_lossy(1000.).unwrap() * v * v
}

fn independent_error(x: &[f64]) -> f64 {
    let a = x[0] - 0.25;
    let b = x[1] + 0.4;
    1001. * (a * a + b * b) - 1998. * a * b
}

#[test]
fn rotated_ill_conditioned_quadratic_both_precisions() {
    let f64_objective = ObjectiveFn::new(2, |x: &[f64]| Ok(rotated(x)));
    let f32_objective = ObjectiveFn::new(2, |x: &[f32]| Ok(rotated(x)));
    let (bounds64, bounds32) = (unit_box::<f64>(), unit_box::<f32>());
    let result64 = solver::<f64>(400)
        .minimize(
            &BoundedObjective::new(&f64_objective, &bounds64).unwrap(),
            &[0.8, 0.8],
        )
        .unwrap();
    let result32 = solver::<f32>(400)
        .minimize(
            &BoundedObjective::new(&f32_objective, &bounds32).unwrap(),
            &[0.8, 0.8],
        )
        .unwrap();
    let x32 = result32
        .x()
        .iter()
        .map(|&x| f64::from(x))
        .collect::<Vec<_>>();
    for (value, tolerance) in [
        (independent_error(result64.x()), 1e-8),
        (independent_error(&x32), 1e-3),
    ] {
        assert!(
            value < tolerance,
            "objective error {value} exceeds {tolerance}"
        );
    }
}

#[test]
fn deterministic_replay_and_complete_generation_reports() {
    let solver = solver::<f64>(50);
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(rotated(x)));
    let bounds = unit_box::<f64>();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[0.8, 0.8],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let first = (output.x().to_vec(), output.fun(), output.fun_evals());
    assert_eq!(work.generations().len(), output.iterations());
    assert!(
        work.generations()
            .iter()
            .all(|report| report.covariance_rank() == 2
                && report.covariance_log_condition().is_finite()
                && report.step_size().is_finite())
    );
    solver
        .minimize_into(
            &problem,
            &[0.8, 0.8],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        first,
        (output.x().to_vec(), output.fun(), output.fun_evals())
    );
    assert!(!output.is_success());
    let attempt = work.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(work.last_accepted().unwrap().point(), output.x());
    // The owned form replays the prepared sequential path.
    assert_eq!(solver.minimize(&problem, &[0.8, 0.8]).unwrap().x(), first.0);
}

#[test]
fn partitioned_policy_replays_sequential_population_order() {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(rotated(x)));
    let bounds = unit_box::<f64>();
    let expected = solver::<f64>(50)
        .minimize(
            &BoundedObjective::new(&objective, &bounds).unwrap(),
            &[0.8, 0.8],
        )
        .unwrap();
    let solver = CmaEs::new(options(50).with_execution(ExecutionPolicy::Auto)).unwrap();
    #[cfg(feature = "std")]
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(core::num::NonZeroUsize::new(2).unwrap());
    let problem = BoundedObjective::new_sync(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut result = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[0.8, 0.8],
            &mut result,
            &mut work,
            ExecutionContext::new(&pool),
        )
        .unwrap();
    assert_eq!(result.x(), expected.x());
    assert_eq!(result.fun(), expected.fun());
    assert_eq!(result.fun_evals(), expected.fun_evals());
}

#[test]
fn callback_failure_preserves_result_and_replays_the_seed() {
    let solver = solver::<f64>(10);
    let good = ObjectiveFn::new(2, |x: &[f64]| Ok(rotated(x)));
    let bounds = unit_box::<f64>();
    let good = BoundedObjective::new(&good, &bounds).unwrap();
    let mut output = solver.minimize(&good, &[0.8, 0.8]).unwrap();
    let old = (output.x().to_vec(), output.fun());
    let calls = core::cell::Cell::new(0);
    let failing = ObjectiveFn::new(2, |_: &[f64]| {
        calls.set(calls.get() + 1);
        if calls.get() == 2 {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "injected failure",
            ))
        } else {
            Ok(1.)
        }
    });
    let failing = BoundedObjective::new(&failing, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    assert!(
        solver
            .minimize_into(
                &failing,
                &[0.8, 0.8],
                &mut output,
                &mut work,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!((output.x().to_vec(), output.fun()), old);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    // The configured seed restarts the stream, so the next call replays the first.
    solver
        .minimize_into(
            &good,
            &[0.8, 0.8],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!((output.x().to_vec(), output.fun()), old);
}

#[test]
fn injected_restart_design_finds_other_basin() {
    let points = [0.7, 0.0];
    let solver = CmaEs::new(
        CmaEsOptions::default()
            .with_population_size(Budget::try_from(8).unwrap())
            .with_max_generations(Budget::try_from(10).unwrap())
            .with_restart(CmaEsRestartPolicy::IncreasingPopulation { max_restarts: 1 })
            .with_fitness_tolerance(ResidualTolerance::new(100., 0.).unwrap())
            .with_seed(7),
    )
    .unwrap()
    .with_design(BorrowedStartDesign::new(&points, 2).unwrap());
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        let left = (x[0] + 0.7).powi(2) + x[1].powi(2) + 0.2;
        let right = (x[0] - 0.7).powi(2) + x[1].powi(2);
        Ok(left.min(right))
    });
    let bounds = unit_box::<f64>();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[-0.7, 0.0],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(work.restarts(), 1);
    assert!(output.fun() < 1e-12);
    assert_eq!(
        output.termination(),
        OptimizeTermination::PopulationStagnation
    );
}

#[test]
fn evaluation_budget_returns_best_observed_start() {
    let solver = solver::<f64>(10);
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(rotated(x)));
    let bounds = unit_box::<f64>();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(1).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(8), &mut budgets)
        .unwrap();
    solver
        .minimize_into(&problem, &[0.8, 0.8], &mut output, &mut work, &mut context)
        .unwrap();
    assert_eq!(output.x(), &[0.8, 0.8]);
    assert_eq!(output.fun_evals(), 1);
    assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
}

#[test]
fn sample_budget_never_advances_past_allowance() {
    let solver = solver::<f64>(10);
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(rotated(x)));
    let bounds = unit_box::<f64>();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Sample, Budget::try_from(1).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(14), &mut budgets)
        .unwrap();
    solver
        .minimize_into(&problem, &[0.8, 0.8], &mut output, &mut work, &mut context)
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::SampleLimit);
    assert_eq!(output.fun_evals(), 1);
    assert_eq!(context.budget_report(BudgetKind::Sample).consumed(), 1);
}

#[test]
fn optional_local_refinement_reuses_existing_solver_boundary() {
    let solver = solver::<f64>(10).with_local(PowellLocalFactory::default());
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
    let bounds = unit_box::<f64>();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(13), &mut budgets)
        .unwrap();
    solver
        .minimize_into(&problem, &[0.8, 0.8], &mut output, &mut work, &mut context)
        .unwrap();
    assert!(output.fun() < 1e-6);
    assert!(work.local_error().is_none());
    assert!(!output.is_success());
    assert_eq!(work.last_accepted().unwrap().point(), output.x());
    assert_eq!(
        output.counts().total().unwrap(),
        context.counts().total().unwrap()
    );
}

#[derive(Debug)]
struct Malformed;

struct MalformedLocal;

impl LocalSolverFactory<f64> for Malformed {
    type Solver = MalformedLocal;
    fn prepare(&self, _: usize) -> Result<MalformedLocal, SciOxideError> {
        Ok(MalformedLocal)
    }
}

impl SearchLocalSolver<f64> for MalformedLocal {
    fn minimize_into(
        &mut self,
        _: &BoundedObjective<'_, f64>,
        _: &[f64],
        _: u64,
        _: &mut OptimizeResult<f64>,
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}

#[test]
fn malformed_local_result_does_not_replace_previous_output() {
    let solver = solver::<f64>(10);
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
    let bounds = unit_box::<f64>();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut output = solver.minimize(&problem, &[0.8, 0.8]).unwrap();
    let old = (output.x().to_vec(), output.fun(), output.termination());
    let solver = solver.with_local(Malformed);
    let mut work = solver.workspace(2).unwrap();
    assert!(
        solver
            .minimize_into(
                &problem,
                &[0.8, 0.8],
                &mut output,
                &mut work,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(
        (output.x().to_vec(), output.fun(), output.termination()),
        old
    );
}
