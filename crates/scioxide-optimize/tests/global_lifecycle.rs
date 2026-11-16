//! Prepared lifecycle contracts of the global searches: success, preflight
//! rejection with unchanged output, budget stops, nested budgets charged to a
//! parent context, allocation-free prepared calls, execution policies selected
//! by options, and thread-safe immutable plans.

#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Bounds, Budget, BudgetKind, EvaluationRole, ExecutionContext, ExecutionPolicy,
    PartialResultTrust, ProblemToken, SciOxideError, SciOxideErrorKind, SolveBudgets,
    SolveIdentity, SolveState,
};
use scioxide_interp::{
    Covariance, CovarianceKernel, LocalModelOptions, RbfKernel, RbfRegularization,
};
use scioxide_optimize::{
    BasinHopping, BasinHoppingOptions, BoundedObjective, CmaEs, CmaEsOptions,
    DifferentialEvolution, DifferentialEvolutionOptions, Direct, DirectOptions, DirectResult,
    DualAnnealing, DualAnnealingOptions, GaussianAcquisition, GlobalSearchOptions,
    GlobalSearchResult, KrigingSurrogateFitter, MultiStart, ObjectiveFn, OptimizeResult,
    OptimizeTermination, PowellLocalFactory, RbfSurrogateFitter, Shgo, ShgoOptions, ShgoResult,
    SurrogateSearch, SurrogateSearchOptions, UniformBasinProposal, UniformStartDesign,
    UpdatingMode,
};
use scioxide_spatial::GeometryOptions;

fn parent(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "GlobalParent", 1),
        2,
        0,
    )
}

fn bowl(x: &[f64]) -> Result<f64, SciOxideError> {
    Ok((x[0] - 0.5).powi(2) + 4.0 * (x[1] + 0.25).powi(2))
}

fn unit_box() -> Bounds<f64> {
    Bounds::new(vec![(Some(-1.0), Some(1.0)); 2]).unwrap()
}

fn unbounded() -> Bounds<f64> {
    Bounds::new(vec![(None, Some(1.0)), (Some(-1.0), Some(1.0))]).unwrap()
}

fn starts(count: usize) -> GlobalSearchOptions<f64> {
    GlobalSearchOptions::default().with_starts(Budget::try_from(count).unwrap())
}

#[test]
fn multistart_success_rejection_budget_and_nesting() {
    let plan =
        MultiStart::new(starts(3), UniformStartDesign, PowellLocalFactory::default()).unwrap();
    let objective = ObjectiveFn::new(2, bowl);
    let bounds = unit_box();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut workspace = plan.workspace(2).unwrap();
    let mut output = GlobalSearchResult::empty();
    let sequential = ExecutionContext::sequential;

    plan.minimize_into(&problem, &mut output, &mut workspace, sequential())
        .unwrap();
    let best = output.best().unwrap();
    assert!(best.fun() < 1e-8);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert!(attempt.epoch().is_some());
    assert_eq!(attempt.counts(), output.counts());
    let accepted = workspace.last_accepted().unwrap();
    assert_eq!(accepted.point(), best.x());
    assert_eq!(accepted.progress().value(), best.fun());
    assert_eq!(accepted.progress().iterations(), 3);
    assert_eq!(accepted.progress().trust(), PartialResultTrust::Admissible);

    // Preflight rejection leaves the output and the accepted state untouched.
    let before = (best.x().to_vec(), output.counts());
    let open = unbounded();
    let error = plan
        .minimize_into(
            &BoundedObjective::new(&objective, &open).unwrap(),
            &mut output,
            &mut workspace,
            sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.epoch().is_none());
    assert!(workspace.last_accepted().is_none());
    assert_eq!(
        (output.best().unwrap().x().to_vec(), output.counts()),
        before
    );

    // A parent's evaluation budget stops the child search with its first observation.
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(5).unwrap());
    let mut context = state
        .begin_solve(sequential(), parent(1), &mut budgets)
        .unwrap();
    plan.minimize_into(&problem, &mut output, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
    assert!(output.best().is_some());
    let charged = context.counts();
    assert_eq!(charged.own().provider(EvaluationRole::Objective), 0);
    assert_eq!(charged.nested().provider(EvaluationRole::Objective), 5);
    assert_eq!(charged.total().unwrap(), output.counts().total().unwrap());
    // The exhausted ancestor denies the next child's first observation.
    plan.minimize_into(&problem, &mut output, &mut workspace, &mut context)
        .unwrap();
    assert!(output.best().is_none());
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
}

#[test]
fn prepared_global_calls_execute() {
    let objective = ObjectiveFn::new(2, bowl);
    let bounds = unit_box();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let sequential = ExecutionContext::sequential;

    let multistart =
        MultiStart::new(starts(3), UniformStartDesign, PowellLocalFactory::default()).unwrap();
    let mut workspace = multistart.workspace(2).unwrap();
    let mut output = GlobalSearchResult::empty();
    // The first publish reserves the caller's output.
    multistart
        .minimize_into(&problem, &mut output, &mut workspace, sequential())
        .unwrap();
    for _ in 0..2 {
        multistart
            .minimize_into(&problem, &mut output, &mut workspace, sequential())
            .unwrap();
    }

    let basin = BasinHopping::new(
        BasinHoppingOptions::new(starts(3), 0.5).unwrap(),
        UniformBasinProposal::new(0.2).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let mut workspace = basin.workspace(2).unwrap();
    basin
        .minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            sequential(),
        )
        .unwrap();
    let anneal = DualAnnealing::new(
        DualAnnealingOptions::new(starts(3)),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let mut workspace = anneal.workspace(2).unwrap();
    anneal
        .minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            sequential(),
        )
        .unwrap();

    let mut result = OptimizeResult::empty();
    result.try_reserve(2).unwrap();
    let evolution = DifferentialEvolution::new(
        DifferentialEvolutionOptions::default()
            .with_max_generations(Budget::try_from(5).unwrap())
            .with_updating(UpdatingMode::Deferred),
    )
    .unwrap();
    let mut workspace = evolution.workspace(2).unwrap();
    for _ in 0..2 {
        evolution
            .minimize_into(
                &problem,
                &[0.0, 0.0],
                &mut result,
                &mut workspace,
                sequential(),
            )
            .unwrap();
    }
    let cma =
        CmaEs::new(CmaEsOptions::default().with_max_generations(Budget::try_from(5).unwrap()))
            .unwrap();
    let mut workspace = cma.workspace(2).unwrap();
    for _ in 0..2 {
        cma.minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut result,
            &mut workspace,
            sequential(),
        )
        .unwrap();
    }
    let refined =
        CmaEs::new(CmaEsOptions::default().with_max_generations(Budget::try_from(5).unwrap()))
            .unwrap()
            .with_local(PowellLocalFactory::default());
    let mut workspace = refined.workspace(2).unwrap();
    refined
        .minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut result,
            &mut workspace,
            sequential(),
        )
        .unwrap();
}

#[test]
fn population_searches_follow_the_attempt_contract() {
    let objective = ObjectiveFn::new(2, bowl);
    let bounds = unit_box();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let evolution = DifferentialEvolution::new(DifferentialEvolutionOptions::default()).unwrap();
    let cma = CmaEs::new(CmaEsOptions::default()).unwrap();
    macro_rules! check {
        ($solver:expr, $name:literal) => {{
            let solver = $solver;
            let mut workspace = solver.workspace(2).unwrap();
            let mut output = OptimizeResult::empty();
            solver
                .minimize_into(
                    &problem,
                    &[0.0, 0.0],
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(output.fun() < 1e-8, $name);
            assert!(!output.is_success(), $name);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Stopped, $name);
            assert_eq!(attempt.counts(), output.counts(), $name);
            assert_eq!(
                workspace.last_accepted().unwrap().point(),
                output.x(),
                $name
            );

            // An `x0` outside the box is rejected before any callback.
            let saved = (output.x().to_vec(), output.fun());
            assert!(
                solver
                    .minimize_into(
                        &problem,
                        &[2.0, 0.0],
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential()
                    )
                    .is_err()
            );
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected, $name);
            assert!(workspace.last_accepted().is_none(), $name);
            assert_eq!((output.x().to_vec(), output.fun()), saved, $name);

            // A nested evaluation budget publishes the best observed member.
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), parent(2), &mut budgets)
                .unwrap();
            solver
                .minimize_into(
                    &problem,
                    &[0.0, 0.0],
                    &mut output,
                    &mut workspace,
                    &mut context,
                )
                .unwrap();
            assert_eq!(
                output.termination(),
                OptimizeTermination::EvaluationLimit,
                $name
            );
            // Batched generations are admitted whole, so fewer calls may run.
            assert!((1..=3).contains(&output.fun_evals()), $name);
            assert_eq!(
                context
                    .counts()
                    .nested()
                    .provider(EvaluationRole::Objective),
                output.fun_evals(),
                $name
            );
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Stopped, $name);
            assert_eq!(
                workspace.last_accepted().unwrap().progress().value(),
                output.fun(),
                $name
            );
        }};
    }
    check!(evolution, "differential evolution");
    check!(cma, "CMA-ES");
}

#[test]
fn sampled_landscapes_publish_atomically() {
    let objective = ObjectiveFn::new(2, bowl);
    let bounds = unit_box();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let failing = ObjectiveFn::new(2, |_: &[f64]| Ok(f64::NAN));
    let failing = BoundedObjective::new(&failing, &bounds).unwrap();
    let sequential = ExecutionContext::sequential;

    let direct = Direct::new(DirectOptions::new(
        Budget::try_from(60).unwrap(),
        Budget::try_from(121).unwrap(),
        Budget::try_from(1 << 20).unwrap(),
    ))
    .unwrap();
    let mut workspace = direct.workspace(2).unwrap();
    let mut output = DirectResult::empty();
    direct
        .minimize_into(&problem, &mut output, &mut workspace, sequential())
        .unwrap();
    let best = output.best_sample().unwrap().1;
    assert!(best < 1e-2);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert!(
        direct
            .minimize_into(&failing, &mut output, &mut workspace, sequential())
            .is_err()
    );
    assert_eq!(output.best_sample().unwrap().1, best);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );

    let shgo = Shgo::new(
        ShgoOptions::new(Budget::try_from(8).unwrap(), GeometryOptions::new()),
        scioxide_random::QmcPlan::sobol(2).unwrap(),
        PowellLocalFactory::default(),
    );
    let mut workspace = shgo.workspace(2).unwrap();
    let mut output = ShgoResult::empty();
    shgo.minimize_into(&problem, &mut output, &mut workspace, sequential())
        .unwrap();
    let best = output.best_observed().unwrap().1;
    assert!(best < 1e-8);
    let open = unbounded();
    assert!(
        shgo.minimize_into(
            &BoundedObjective::new(&objective, &open).unwrap(),
            &mut output,
            &mut workspace,
            sequential(),
        )
        .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(output.best_observed().unwrap().1, best);
}

#[test]
fn execution_policies_in_options_replay_the_sequential_path() {
    let objective = ObjectiveFn::new(2, bowl);
    let bounds = unit_box();
    let local = BoundedObjective::new(&objective, &bounds).unwrap();
    let shared = BoundedObjective::new_sync(&objective, &bounds).unwrap();
    let parallel = ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap());

    let sequential = MultiStart::new(starts(4), UniformStartDesign, PowellLocalFactory::default())
        .unwrap()
        .minimize(&local)
        .unwrap();
    let partitioned = MultiStart::new(
        starts(4).with_execution(parallel),
        UniformStartDesign,
        PowellLocalFactory::default(),
    )
    .unwrap();
    assert!(partitioned.minimize(&local).is_err());
    let replay = partitioned.minimize(&shared).unwrap();
    for (left, right) in sequential.starts().iter().zip(replay.starts()) {
        assert_eq!(left.start(), right.start());
        assert_eq!(
            left.candidate().map(OptimizeResult::x),
            right.candidate().map(OptimizeResult::x)
        );
    }

    let deferred = DifferentialEvolutionOptions::default()
        .with_max_generations(Budget::try_from(20).unwrap())
        .with_updating(UpdatingMode::Deferred);
    let expected = DifferentialEvolution::new(deferred.clone())
        .unwrap()
        .minimize(&local, &[0.0, 0.0])
        .unwrap();
    let actual = DifferentialEvolution::new(deferred.with_execution(parallel))
        .unwrap()
        .minimize(&shared, &[0.0, 0.0])
        .unwrap();
    assert_eq!(expected.x(), actual.x());
    assert_eq!(expected.fun_evals(), actual.fun_evals());

    let options = CmaEsOptions::default().with_max_generations(Budget::try_from(20).unwrap());
    let expected = CmaEs::new(options.clone())
        .unwrap()
        .minimize(&local, &[0.0, 0.0])
        .unwrap();
    let actual = CmaEs::new(options.with_execution(parallel))
        .unwrap()
        .minimize(&shared, &[0.0, 0.0])
        .unwrap();
    assert_eq!(expected.x(), actual.x());

    // DIRECT and the chains are sequential by construction.
    let direct = DirectOptions::<f64>::new(Budget::ONE, Budget::ONE, Budget::ONE);
    assert!(Direct::new(direct.with_execution(parallel)).is_err());
    assert!(
        DualAnnealing::new(
            DualAnnealingOptions::new(starts(2).with_execution(parallel)),
            PowellLocalFactory::default(),
        )
        .is_err()
    );
}

type Feasibility = fn(&[f64]) -> Result<bool, SciOxideError>;

#[test]
fn plans_are_send_sync_and_clone() {
    fn plan<T: Send + Sync + Clone>(_: &T) {}
    let factory = PowellLocalFactory::<f64>::default();
    plan(&MultiStart::new(starts(2), UniformStartDesign, factory.clone()).unwrap());
    plan(
        &MultiStart::new(
            starts(2),
            scioxide_random::QmcPlan::halton(2).unwrap(),
            factory.clone(),
        )
        .unwrap(),
    );
    plan(
        &BasinHopping::new(
            BasinHoppingOptions::new(starts(2), 1.0).unwrap(),
            UniformBasinProposal::new(0.2).unwrap(),
            factory.clone(),
        )
        .unwrap(),
    );
    plan(&DualAnnealing::new(DualAnnealingOptions::new(starts(2)), factory.clone()).unwrap());
    plan(&Direct::<f64>::new(DirectOptions::new(Budget::ONE, Budget::ONE, Budget::ONE)).unwrap());
    plan(&Shgo::new(
        ShgoOptions::new(Budget::ONE, GeometryOptions::new()),
        scioxide_random::QmcPlan::sobol(2).unwrap(),
        factory.clone(),
    ));
    plan(&DifferentialEvolution::<f64>::new(DifferentialEvolutionOptions::default()).unwrap());
    let cma = CmaEs::<f64>::new(CmaEsOptions::default()).unwrap();
    plan(&cma);
    plan(&cma.with_local(factory.clone()));
    let feasible: Feasibility = |_| Ok(true);
    let kriging = KrigingSurrogateFitter::new(
        Covariance::new(CovarianceKernel::SquaredExponential, 1.0, [0.8]).unwrap(),
        LocalModelOptions::default(),
    );
    plan(
        &SurrogateSearch::<f64, 1, _, _, _, _, _>::new(
            SurrogateSearchOptions::default(),
            kriging,
            GaussianAcquisition::ExpectedImprovement { margin: 0.0 },
            feasible,
            UniformStartDesign,
            factory.clone(),
        )
        .unwrap(),
    );
    let rbf = RbfSurrogateFitter::new(
        RbfKernel::Gaussian(1.0),
        RbfRegularization::Diagonal(0.01),
        LocalModelOptions::default(),
    );
    plan(
        &SurrogateSearch::<f64, 1, _, _, _, _, _>::new(
            SurrogateSearchOptions::default(),
            rbf,
            GaussianAcquisition::LowerConfidenceBound { weight: 1.0 },
            feasible,
            UniformStartDesign,
            factory,
        )
        .unwrap(),
    );
}
