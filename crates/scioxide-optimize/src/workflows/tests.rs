use super::*;
use crate::{BorrowedStartDesign, ConstraintFn, Constraints, FiniteBounds};
use alloc::{vec, vec::Vec};
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use scioxide_core::{
    AttemptKind, AttemptOutcome, Bounds, Budget, BudgetKind, Capacity, Complex, EvaluationRole,
    ExecutionContext, ExecutionPolicy, FeasibilityTolerance, GradientTolerance, ProblemToken,
    SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};

fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn parent_identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    )
}
fn error_kind(result: Result<(), SciOxideError>) -> SciOxideErrorKind {
    result.unwrap_err().kind()
}
fn assert_rejected_resume<P>(record: Option<&scioxide_core::AttemptRecord<P>>) {
    let record = record.unwrap();
    assert_eq!(record.kind(), AttemptKind::Resume);
    assert_eq!(record.outcome(), AttemptOutcome::PreflightRejected);
    assert!(record.progress().is_none());
    assert!(record.epoch().is_none());
}

/// Delegating provider that counts callbacks and can override the noise model.
struct Counted<'a, T: ?Sized> {
    inner: &'a T,
    noise: Option<NoiseModel>,
    calls: AtomicUsize,
}
impl<'a, T: ?Sized> Counted<'a, T> {
    fn new(inner: &'a T, noise: Option<NoiseModel>) -> Self {
        Self {
            inner,
            noise,
            calls: AtomicUsize::new(0),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}
impl<S: NumericScalar, T: SampledObjective<S> + ?Sized> SampledObjective<S> for Counted<'_, T> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn objectives(&self) -> usize {
        self.inner.objectives()
    }
    fn noise_model(&self) -> NoiseModel {
        self.noise.unwrap_or_else(|| self.inner.noise_model())
    }
    fn sample_into(&self, x: &[S], sample: u64, out: &mut [S::Real]) -> Result<(), SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.sample_into(x, sample, out)
    }
}
impl<S: NumericScalar, T: SampledGradient<S> + ?Sized> SampledGradient<S> for Counted<'_, T> {
    fn gradient_sample_into(
        &self,
        x: &[S],
        sample: u64,
        out: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.gradient_sample_into(x, sample, out)
    }
}
/// Philox observations under another configuration token, with a call counter.
struct CountingSampler {
    token: u128,
    calls: AtomicUsize,
}
impl CountingSampler {
    fn new(token: u128) -> Self {
        Self {
            token,
            calls: AtomicUsize::new(0),
        }
    }
}
impl OptimizationSampler for CountingSampler {
    fn configuration_token(&self) -> u128 {
        self.token
    }
    fn sample(&self, key: OptimizationSampleKey) -> Result<u64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        PhiloxSampler.sample(key)
    }
}

struct Schaffer;
impl<R: RealScalar + NumericScalar<Real = R>> SampledObjective<R> for Schaffer {
    fn dimension(&self) -> usize {
        1
    }
    fn objectives(&self) -> usize {
        2
    }
    fn noise_model(&self) -> NoiseModel {
        NoiseModel::Deterministic
    }
    fn sample_into(&self, x: &[R], _: u64, out: &mut [R]) -> Result<(), SciOxideError> {
        out[0] = x[0] * x[0];
        let d = x[0] - number::<R>(2.)?;
        out[1] = d * d;
        Ok(())
    }
}
/// Schaffer's problem with a constant third objective.
struct Three;
impl SampledObjective<f64> for Three {
    fn dimension(&self) -> usize {
        1
    }
    fn objectives(&self) -> usize {
        3
    }
    fn noise_model(&self) -> NoiseModel {
        NoiseModel::Deterministic
    }
    fn sample_into(&self, x: &[f64], token: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
        Schaffer.sample_into(x, token, &mut out[..2])?;
        out[2] = 7.;
        Ok(())
    }
}
fn bounds<R: RealScalar>() -> FiniteBounds<R> {
    FiniteBounds::new(vec![(number(-1.).unwrap(), number(3.).unwrap())]).unwrap()
}
fn nsga<R: RealScalar + NumericScalar<Real = R>>(generations: usize) -> Nsga2<R> {
    Nsga2::new(Nsga2Options::new(32, 24, budget(generations))).unwrap()
}
fn points(archive: &ParetoArchive<f64>) -> Vec<f64> {
    archive.entries().map(|e| e.point()[0]).collect()
}
fn population(result: &Nsga2Result<f64>) -> Vec<f64> {
    result.population().map(|e| e.point()[0]).collect()
}

#[test]
fn exact_dominance_ties_and_invalid_shapes() {
    assert_eq!(
        pareto_dominance(&[1., 2.], &[1., 3.]).unwrap(),
        ParetoDominance::Dominates
    );
    assert_eq!(
        pareto_dominance(&[1., 3.], &[1., 2.]).unwrap(),
        ParetoDominance::Dominated
    );
    assert_eq!(
        pareto_dominance(&[0., 2.], &[-0., 2.]).unwrap(),
        ParetoDominance::Equal
    );
    assert_eq!(
        pareto_dominance(&[0., 3.], &[1., 2.]).unwrap(),
        ParetoDominance::Incomparable
    );
    assert!(pareto_dominance::<f64>(&[], &[]).is_err());
    assert!(pareto_dominance(&[1.], &[1., 2.]).is_err());
    assert!(pareto_dominance(&[f64::NAN], &[0.]).is_err());
}
#[test]
fn archive_pruning_scaling_constraints_and_hypervolume() {
    let mut a = ParetoArchive::new(1, 2, 3).unwrap();
    let mut scaled = ParetoArchive::new(1, 2, 3).unwrap();
    for i in 0..9 {
        let x = i as f64 / 8.;
        a.insert(&[x], &[x, 1. - x], &[]).unwrap();
        scaled
            .insert(&[x], &[x * 8., (1. - x) * 0.25], &[])
            .unwrap();
    }
    assert_eq!(a.len(), 3);
    assert_eq!(a.pruned(), 6);
    assert!(a.verify().unwrap());
    assert_eq!(
        a.entries().map(|e| e.id()).collect::<Vec<_>>(),
        scaled.entries().map(|e| e.id()).collect::<Vec<_>>()
    );
    let before = a
        .entries()
        .map(|e| e.objectives().to_vec())
        .collect::<Vec<_>>();
    assert!(!a.insert(&[3.], &[2., 2.], &[]).unwrap());
    assert!(!a.insert(&[0.], &[0., 1.], &[]).unwrap());
    assert!(a.insert(&[0.], &[f64::NAN, 1.], &[]).is_err());
    assert_eq!(
        before,
        a.entries()
            .map(|e| e.objectives().to_vec())
            .collect::<Vec<_>>()
    );
    let mut h = ParetoArchive::new(1, 2, 3).unwrap();
    for p in [[0., 1.], [0.5, 0.5], [1., 0.]] {
        h.insert(&[p[0]], &p, &[]).unwrap();
    }
    assert!((h.hypervolume_2d([2., 2.]).unwrap() - 3.25_f64).abs() < 1e-14);
    assert!(h.hypervolume_2d([0., 0.]).is_err());
    let mut c = ParetoArchive::new(1, 2, 3)
        .unwrap()
        .with_constraints(1, 1, FeasibilityTolerance::new(1e-6, 0.).unwrap())
        .unwrap();
    assert!(!c.insert(&[0.], &[0., 0.], &[1e-3, -1.]).unwrap());
    assert!(c.insert(&[0.], &[0., 0.], &[1e-7, -1.]).unwrap());
    assert!(c.verify().unwrap());
    let mut constant = ParetoArchive::new(1, 2, 1).unwrap();
    assert!(constant.insert(&[0.], &[1., 1.], &[]).unwrap());
    assert!(!constant.insert(&[1.], &[1., 1.], &[]).unwrap());
    assert_eq!(constant.entries().next().unwrap().id(), 0);
}
fn schaffer<R: RealScalar + NumericScalar<Real = R>>() {
    let b = bounds::<R>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let result = nsga::<R>(40).solve(&problem).unwrap();
    assert!(result.archive().verify().unwrap());
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    assert_eq!(result.generations(), 40);
    let mut x: Vec<_> = result
        .archive()
        .entries()
        .map(|e| e.point()[0].try_to_f64_exact().unwrap())
        .collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(x.len() >= 20);
    assert!(x[0].abs() < 0.05, "{x:?}");
    assert!((x[x.len() - 1] - 2.).abs() < 0.05, "{x:?}");
    assert!(x.windows(2).all(|w| w[1] - w[0] < 0.25), "{x:?}");
    assert!(x.iter().all(|&x| (-0.05..=2.05).contains(&x)));
    let again = nsga::<R>(40).solve(&problem).unwrap();
    assert_eq!(
        result
            .archive()
            .entries()
            .map(|e| e.point()[0])
            .collect::<Vec<_>>(),
        again
            .archive()
            .entries()
            .map(|e| e.point()[0])
            .collect::<Vec<_>>()
    );
}
#[test]
fn analytic_schaffer_front_f32_f64() {
    schaffer::<f32>();
    schaffer::<f64>();
}
#[test]
fn feasible_first_bounds_equalities_and_all_infeasible() {
    let constraints = Constraints::new(1)
        .unwrap()
        .inequality(ConstraintFn::new(1, |x: &[f64]| Ok(0.5 - x[0])))
        .unwrap()
        .bounds(Bounds::new(vec![(None, Some(1.5))]).unwrap())
        .unwrap();
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let solver = nsga::<f64>(20);
    let result = solver
        .solve(&problem.with_constraints(&constraints).unwrap())
        .unwrap();
    assert!(result.archive().verify().unwrap());
    assert!(!result.archive().is_empty());
    assert!(
        result
            .archive()
            .entries()
            .all(|e| (0.5 - 1e-7..=1.5 + 1e-7).contains(&e.point()[0]))
    );
    let impossible = Constraints::new(1)
        .unwrap()
        .inequality(ConstraintFn::new(1, |_: &[f64]| Ok(1.)))
        .unwrap();
    let result = solver
        .solve(&problem.with_constraints(&impossible).unwrap())
        .unwrap();
    assert!(result.archive().is_empty());
    assert_eq!(result.least_violating().unwrap().id(), 0);
    assert!(result.least_violating().unwrap().violation() > 0.9);
    let equality = Constraints::new(1)
        .unwrap()
        .equality(ConstraintFn::new(1, |x: &[f64]| Ok(x[0] - 1.)))
        .unwrap();
    let starts = vec![1.; 32];
    let solver = solver.with_start_design(BorrowedStartDesign::new(&starts, 1).unwrap());
    let result = solver
        .solve(&problem.with_constraints(&equality).unwrap())
        .unwrap();
    assert!(result.archive().verify().unwrap());
    assert!(result.archive().entries().any(|e| e.point() == [1.]));
}
struct Quadratic<S: NumericScalar> {
    center: S,
}
impl<S: NumericScalar> crate::SmoothGradient<S> for Quadratic<S> {
    fn dimension(&self) -> usize {
        1
    }
    fn value(&self, x: &[S]) -> Result<S::Real, SciOxideError> {
        Ok((x[0] - self.center).norm_sqr() * S::Real::HALF)
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        out[0] = x[0] - self.center;
        Ok(())
    }
}
fn stochastic_exact<S: NumericScalar>(center: S, tolerance: S::Real)
where
    S::Real: NumericScalar<Real = S::Real>,
{
    let q = Quadratic { center };
    let p = DeterministicGradient(&q);
    let problem = StochasticProblem::new(&p);
    let tol =
        GradientTolerance::new(tolerance, <_ as scioxide_core::AlgebraicScalar>::ZERO).unwrap();
    let sgd = StochasticGradientDescent::new(
        SgdOptions::new(
            SgdStep::Constant(number::<S::Real>(0.5).unwrap()),
            budget(100),
            Budget::ONE,
        )
        .with_tolerance(tol),
    )
    .unwrap();
    let result = sgd.minimize(&problem, &[S::ZERO]).unwrap();
    assert!(result.is_success());
    assert!((result.x()[0] - center).abs() <= tolerance);
    let trust = StochasticTrustRegion::new(
        StochasticTrustOptions::new(budget(200), ReplicationOptions::fixed(Budget::ONE))
            .with_tolerance(tol),
    )
    .unwrap();
    let result = trust.minimize(&problem, &[S::ZERO]).unwrap();
    assert!(result.is_success(), "{:?}", result.termination());
    assert!((result.x()[0] - center).abs() <= tolerance);
    assert!((result.gradient().unwrap()[0] - (result.x()[0] - center)).abs() <= tolerance);
}
#[test]
fn analytic_real_and_complex_stochastic_solvers() {
    stochastic_exact::<f32>(3., 1e-4);
    stochastic_exact::<f64>(3., 1e-9);
    stochastic_exact::<Complex<f32>>(Complex { re: 3., im: 4. }, 2e-4);
    stochastic_exact::<Complex<f64>>(Complex { re: 3., im: 4. }, 1e-9);
}
struct Noisy;
impl SampledObjective<f64> for Noisy {
    fn dimension(&self) -> usize {
        1
    }
    fn objectives(&self) -> usize {
        1
    }
    fn sample_into(&self, x: &[f64], token: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = 0.5 * x[0] * x[0] + (scioxide_random::decode_u64::<f64>(token) - 0.5) * 0.1;
        Ok(())
    }
}
impl SampledGradient<f64> for Noisy {
    fn gradient_sample_into(
        &self,
        x: &[f64],
        token: u64,
        out: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = x[0] + (scioxide_random::decode_u64::<f64>(token) - 0.5) * 0.1;
        Ok(())
    }
}
#[test]
fn statistical_stopping_and_noise_limit_are_not_deterministic_success() {
    let problem = StochasticProblem::new(&Noisy);
    let sgd = StochasticGradientDescent::new(
        SgdOptions::new(
            SgdStep::Diminishing {
                initial: 0.5,
                decay: 0.02,
                power: 1.,
            },
            budget(200),
            budget(32),
        )
        .with_tolerance(GradientTolerance::new(0.02, 0.).unwrap()),
    )
    .unwrap();
    let result = sgd.minimize(&problem, &[1.]).unwrap();
    assert_eq!(
        result.termination(),
        OptimizeTermination::EstimatedStationarity
    );
    assert!(!result.is_success());
    assert!(result.x()[0].abs() < 0.02);
    assert!(result.standard_error().unwrap() > 0.);
    let trust = StochasticTrustRegion::new(
        StochasticTrustOptions::new(
            budget(100),
            ReplicationOptions::adaptive(budget(2), budget(8)).unwrap(),
        )
        .with_radii(1e-12, 1e-4, 1.)
        .with_tolerance(GradientTolerance::new(1e-12, 0.).unwrap()),
    )
    .unwrap();
    let result = trust.minimize(&problem, &[1.]).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::NoiseLimit);
    assert_eq!(result.gradient_replicates(), Some(8));
    assert_eq!(result.x(), &[1.]);
    assert!(!result.is_success());
}
#[test]
fn independent_noise_moments_and_replication_calibration() {
    let mut estimates = scioxide_stats::OnlineMoments::new(scioxide_stats::MomentOrder::Second);
    let mut expected_variance = 0.;
    for seed in 0..128 {
        let mut batch = sampling::Batch::<f64>::new(1, 32).unwrap();
        let mut cursor = 0;
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        assert_eq!(
            batch
                .extend(
                    32,
                    NoiseModel::Independent,
                    seed,
                    1,
                    &mut cursor,
                    &PhiloxSampler,
                    EvaluationRole::Objective,
                    ExecutionPolicy::Sequential,
                    &mut context,
                    &|token, out| Noisy.sample_into(&[0.], token, out),
                )
                .unwrap(),
            None
        );
        estimates.observe(batch.mean[0]).unwrap();
        expected_variance += batch.error[0].powi(2);
        assert_eq!(cursor, 32);
        assert_eq!(context.counts().own().provider(EvaluationRole::Sample), 32);
        assert_eq!(
            context.counts().own().provider(EvaluationRole::Objective),
            32
        );
    }
    let snapshot = estimates.snapshot();
    let analytic = 0.1_f64.powi(2) / 12. / 32.;
    assert!((expected_variance / 128. / analytic - 1.).abs() < 0.15);
    assert!((snapshot.sample_variance().unwrap().unwrap() / analytic - 1.).abs() < 0.4);
    assert!(snapshot.mean().unwrap().unwrap().abs() < 6. * (analytic / 128.).sqrt());
}
#[test]
fn prepared_solves_and_archive_insertion_execute() {
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let solver = nsga::<f64>(2);
    let mut out = solver.result(&problem).unwrap();
    let mut work = solver.workspace(&problem).unwrap();
    let requirements = solver.workspace_requirements(&problem).unwrap();
    assert_eq!(work.check(&requirements).unwrap(), Capacity::Sufficient);
    assert_eq!(
        Nsga2Workspace::<f64>::new().check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    solver
        .solve_into(
            &problem,
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out.generations(), 2);
    assert!(!out.archive().is_empty());
    let mut archive = ParetoArchive::new(1, 2, 4).unwrap();
    for i in 0..9 {
        let x = i as f64 / 8.;
        archive.insert(&[x], &[x, 1. - x], &[]).unwrap();
    }
    assert_eq!(archive.len(), 4);

    let q = Quadratic { center: 3. };
    let p = DeterministicGradient(&q);
    let problem = StochasticProblem::new(&p);
    let solver = StochasticGradientDescent::new(SgdOptions::new(
        SgdStep::Constant(0.5),
        budget(40),
        budget(2),
    ))
    .unwrap();
    let mut out = StochasticResult::empty();
    out.try_reserve(1).unwrap();
    let mut work = solver.workspace::<f64>(1).unwrap();
    assert_eq!(
        work.check(&solver.workspace_requirements::<f64>(1).unwrap())
            .unwrap(),
        Capacity::Sufficient
    );
    solver
        .minimize_into(
            &problem,
            &[0.],
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(out.is_success());
    let solver = StochasticTrustRegion::new(StochasticTrustOptions::new(
        budget(40),
        ReplicationOptions::adaptive(budget(2), budget(16)).unwrap(),
    ))
    .unwrap();
    let mut work = solver.workspace::<f64>(1).unwrap();
    solver
        .minimize_into(
            &problem,
            &[0.],
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(out.is_success());
}

#[test]
fn exact_budget_admission_and_nested_parent_limits() {
    let solver = StochasticGradientDescent::new(SgdOptions::new(
        SgdStep::Constant(0.25),
        budget(8),
        budget(4),
    ))
    .unwrap();
    let problem = StochasticProblem::new(&Noisy);
    // Shared parent allowances stop the child at exact batch admission.
    for (kind, limit, expected) in [
        (BudgetKind::Sample, 9, OptimizeTermination::SampleLimit),
        (
            BudgetKind::Evaluation,
            17,
            OptimizeTermination::EvaluationLimit,
        ),
    ] {
        let mut result = StochasticResult::empty();
        let mut work = solver.workspace::<f64>(1).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, budget(limit));
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(&problem, &[1.], &mut result, &mut work, &mut parent)
            .unwrap();
        assert_eq!(result.termination(), expected);
        assert!(parent.budget_report(kind).consumed() <= limit);
        assert_eq!(result.x(), &[1.]);
        assert_eq!(result.sample_cursor(), 8);
        assert_eq!(result.counts().own().provider(EvaluationRole::Objective), 4);
        assert_eq!(result.counts().own().provider(EvaluationRole::Gradient), 4);
        assert_eq!(parent.counts().own().total_providers().unwrap(), 0);
        assert_eq!(
            parent.counts().nested().provider(EvaluationRole::Objective),
            4
        );
        let record = work.last_attempt().unwrap();
        assert_eq!(record.outcome(), AttemptOutcome::Stopped);
        assert_eq!(record.progress().unwrap().iterations(), 0);
        assert_eq!(work.last_accepted(), Some(&[1.][..]));
    }
    // The configured iteration total is the epoch's own allowance.
    let capped = StochasticGradientDescent::new(SgdOptions::new(
        SgdStep::Constant(0.25),
        budget(1),
        budget(4),
    ))
    .unwrap();
    let result = capped.minimize(&problem, &[1.]).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(result.iterations(), 1);
    // Iteration and proposal allowances are scope-local: a parent's limits never stop a child.
    let expected = solver.minimize(&problem, &[1.]).unwrap();
    assert_eq!(expected.termination(), OptimizeTermination::IterationLimit);
    for kind in [BudgetKind::Iteration, BudgetKind::Proposal] {
        let mut result = StochasticResult::empty();
        let mut work = solver.workspace::<f64>(1).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::ONE);
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(&problem, &[1.], &mut result, &mut work, &mut parent)
            .unwrap();
        assert_eq!(result.iterations(), 8);
        assert_eq!(result.x(), expected.x());
        assert_eq!(result.sample_cursor(), expected.sample_cursor());
        assert_eq!(parent.budget_report(kind).consumed(), 0);
    }
    // Unaffordable initial batches invoke no sampler or objective and keep the output.
    let mut result = expected.clone();
    let mut work = solver.workspace::<f64>(1).unwrap();
    let counted = Counted::new(&Noisy, None);
    let sampler = CountingSampler::new(PhiloxSampler.configuration_token());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(3));
    let mut parent = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(),
            &mut budgets,
        )
        .unwrap();
    assert!(
        solver
            .minimize_into(
                &StochasticProblem::new(&counted).with_sampler(&sampler),
                &[2.],
                &mut result,
                &mut work,
                &mut parent,
            )
            .is_err()
    );
    assert_eq!(counted.calls(), 0);
    assert_eq!(sampler.calls.load(Ordering::Relaxed), 0);
    assert_eq!(result.x(), expected.x());
    assert_eq!(result.sample_cursor(), expected.sample_cursor());
    assert_eq!(result.termination(), expected.termination());
    assert_eq!(parent.counts().own().total_providers().unwrap(), 0);
    assert_eq!(parent.counts().nested().total_providers().unwrap(), 0);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert!(work.last_accepted().is_none());
    assert_eq!(
        work.checkpoint().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn nsga_checkpoints_resume_at_completed_generations_and_reject_mismatches() {
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let solver = nsga::<f64>(5);
    let expected = solver.solve(&problem).unwrap();
    assert_eq!(expected.generations(), 5);
    // Earlier generations do not depend on the total, so a shorter run marks the
    // sample ordinal at which the third generation begins.
    let boundary = nsga::<f64>(2).solve(&problem).unwrap().sample_cursor();

    let mut workspace = solver.workspace(&problem).unwrap();
    assert_eq!(
        workspace.checkpoint().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let mut output = solver.result(&problem).unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(boundary as usize));
    {
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .solve_into(&problem, &mut output, &mut workspace, &mut parent)
            .unwrap();
        assert_eq!(
            parent.budget_report(BudgetKind::Sample).consumed(),
            boundary as usize
        );
    }
    // A budget stop after the initial population publishes an unsuccessful result.
    assert_eq!(output.termination(), OptimizeTermination::SampleLimit);
    assert!(!output.termination().is_success());
    assert_eq!(output.generations(), 2);
    assert_eq!(output.sample_cursor(), boundary);
    let record = workspace.last_attempt().unwrap();
    assert_eq!(record.kind(), AttemptKind::Call);
    assert_eq!(record.outcome(), AttemptOutcome::Stopped);
    assert_eq!(record.progress().unwrap().generations(), 2);
    assert_eq!(
        record.progress().unwrap().archived(),
        output.archive().len()
    );
    assert_eq!(
        points(workspace.last_accepted().unwrap().archive()),
        points(output.archive())
    );
    let checkpoint = workspace.checkpoint().unwrap();
    assert_eq!(checkpoint.generations(), 2);
    assert_eq!(checkpoint.sample_cursor(), boundary);
    assert_eq!(points(checkpoint.archive()), points(output.archive()));

    // Resuming under a fresh unbounded root replays the uninterrupted search exactly.
    let mut resumed = solver.result(&problem).unwrap();
    let mut fresh = solver.workspace(&problem).unwrap();
    solver
        .resume_into(
            &problem,
            &checkpoint,
            &mut resumed,
            &mut fresh,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        resumed.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    assert_eq!(resumed.generations(), expected.generations());
    assert_eq!(points(resumed.archive()), points(expected.archive()));
    assert_eq!(
        resumed.archive().crowding_distances(),
        expected.archive().crowding_distances()
    );
    assert_eq!(population(&resumed), population(&expected));
    assert_eq!(resumed.sample_cursor(), expected.sample_cursor());
    let record = fresh.last_attempt().unwrap();
    assert_eq!(record.kind(), AttemptKind::Resume);
    assert_eq!(record.progress().unwrap().generations(), 5);
    assert_eq!(checkpoint.generations(), 2);

    // Mismatched resumes are rejected before any callback and change nothing.
    let metadata = checkpoint.metadata();
    let archived = points(checkpoint.archive());
    let published = (
        points(output.archive()),
        population(&output),
        output.sample_cursor(),
        output.termination(),
    );
    let check_unchanged = |output: &Nsga2Result<f64>| {
        assert_eq!(checkpoint.metadata(), metadata);
        assert_eq!(points(checkpoint.archive()), archived);
        assert_eq!(checkpoint.sample_cursor(), boundary);
        assert_eq!(checkpoint.generations(), 2);
        assert_eq!(
            (
                points(output.archive()),
                population(output),
                output.sample_cursor(),
                output.termination(),
            ),
            published
        );
    };
    let counted = Counted::new(&Schaffer, None);
    let reseeded = Nsga2::new(Nsga2Options::new(32, 24, budget(5)).with_seed(7)).unwrap();
    assert_eq!(
        error_kind(reseeded.resume_into(
            &Nsga2Problem::new(&counted, &b).unwrap(),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_rejected_resume(workspace.last_attempt());
    assert!(workspace.last_accepted().is_none());
    check_unchanged(&output);
    let sampler = CountingSampler::new(7);
    assert_eq!(
        error_kind(
            solver.resume_into(
                &Nsga2Problem::new(&counted, &b)
                    .unwrap()
                    .with_sampler(&sampler),
                &checkpoint,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
        ),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(sampler.calls.load(Ordering::Relaxed), 0);
    assert_rejected_resume(workspace.last_attempt());
    check_unchanged(&output);
    let noisy = Counted::new(&Schaffer, Some(NoiseModel::Independent));
    assert_eq!(
        error_kind(solver.resume_into(
            &Nsga2Problem::new(&noisy, &b).unwrap(),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(noisy.calls(), 0);
    assert_rejected_resume(workspace.last_attempt());
    check_unchanged(&output);
    let three = Counted::new(&Three, None);
    assert_eq!(
        error_kind(solver.resume_into(
            &Nsga2Problem::new(&three, &b).unwrap(),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(three.calls(), 0);
    assert_rejected_resume(workspace.last_attempt());
    check_unchanged(&output);
    assert_eq!(counted.calls(), 0);
    assert_eq!(
        workspace.checkpoint().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );

    // Initialized points commit individually and resume mid-initialization exactly.
    let mut partial = solver.result(&problem).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(40));
    {
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .solve_into(&problem, &mut partial, &mut workspace, &mut parent)
            .unwrap();
    }
    assert_eq!(partial.termination(), OptimizeTermination::SampleLimit);
    assert_eq!(partial.generations(), 0);
    assert_eq!(partial.population().count(), 20);
    let checkpoint = workspace.checkpoint().unwrap();
    solver
        .resume_into(
            &problem,
            &checkpoint,
            &mut partial,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(points(partial.archive()), points(expected.archive()));
    assert_eq!(population(&partial), population(&expected));
    assert_eq!(partial.sample_cursor(), expected.sample_cursor());
}

#[test]
fn sgd_checkpoints_resume_at_completed_updates_and_reject_mismatches() {
    let options = SgdOptions::new(SgdStep::Constant(0.1), budget(8), budget(4))
        .with_tolerance(GradientTolerance::new(1e-12, 0.).unwrap());
    let solver = StochasticGradientDescent::new(options).unwrap();
    let problem = StochasticProblem::new(&Noisy);
    let expected = solver.minimize(&problem, &[1.]).unwrap();
    assert_eq!(expected.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(expected.iterations(), 8);

    let mut workspace = solver.workspace::<f64>(1).unwrap();
    assert_eq!(
        workspace.checkpoint().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    // Four initial samples, then four gradient and four candidate samples per update.
    let boundary = 4 + 3 * 8;
    let mut output = StochasticResult::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(boundary));
    {
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(&problem, &[1.], &mut output, &mut workspace, &mut parent)
            .unwrap();
        assert_eq!(
            parent.budget_report(BudgetKind::Sample).consumed(),
            boundary
        );
    }
    assert_eq!(output.termination(), OptimizeTermination::SampleLimit);
    assert!(!output.is_success());
    assert_eq!(output.iterations(), 3);
    assert_eq!(output.sample_cursor(), boundary as u64);
    let record = workspace.last_attempt().unwrap();
    assert_eq!(record.kind(), AttemptKind::Call);
    assert_eq!(record.outcome(), AttemptOutcome::Stopped);
    assert_eq!(record.progress().unwrap().iterations(), 3);
    assert_eq!(record.progress().unwrap().value(), output.value().unwrap());
    assert_eq!(workspace.last_accepted(), Some(output.x()));
    let checkpoint = workspace.checkpoint().unwrap();
    assert_eq!(checkpoint.x(), output.x());
    assert_eq!(checkpoint.value(), output.value());
    assert_eq!(checkpoint.iterations(), 3);
    assert_eq!(checkpoint.sample_cursor(), boundary as u64);

    let mut resumed = StochasticResult::empty();
    let mut fresh = solver.workspace::<f64>(1).unwrap();
    solver
        .resume_into(
            &problem,
            &checkpoint,
            &mut resumed,
            &mut fresh,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(resumed.termination(), expected.termination());
    assert_eq!(resumed.x(), expected.x());
    assert_eq!(resumed.value(), expected.value());
    assert_eq!(resumed.gradient(), expected.gradient());
    assert_eq!(resumed.iterations(), expected.iterations());
    assert_eq!(resumed.sample_cursor(), expected.sample_cursor());
    assert_eq!(fresh.last_attempt().unwrap().kind(), AttemptKind::Resume);
    assert_eq!(checkpoint.iterations(), 3);

    // Mismatched resumes are rejected before any callback and change nothing.
    let metadata = checkpoint.metadata();
    let published = output.clone();
    let check_unchanged = |output: &StochasticResult<f64>| {
        assert_eq!(checkpoint.metadata(), metadata);
        assert_eq!(checkpoint.x(), published.x());
        assert_eq!(checkpoint.sample_cursor(), boundary as u64);
        assert_eq!(output.x(), published.x());
        assert_eq!(output.value(), published.value());
        assert_eq!(output.sample_cursor(), published.sample_cursor());
        assert_eq!(output.termination(), published.termination());
    };
    let counted = Counted::new(&Noisy, None);
    let reseeded = StochasticGradientDescent::new(options.with_seed(7)).unwrap();
    assert_eq!(
        error_kind(reseeded.resume_into(
            &StochasticProblem::new(&counted),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_rejected_resume(workspace.last_attempt());
    assert!(workspace.last_accepted().is_none());
    check_unchanged(&output);
    let sampler = CountingSampler::new(7);
    assert_eq!(
        error_kind(solver.resume_into(
            &StochasticProblem::new(&counted).with_sampler(&sampler),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(sampler.calls.load(Ordering::Relaxed), 0);
    assert_rejected_resume(workspace.last_attempt());
    check_unchanged(&output);
    let deterministic = Counted::new(&Noisy, Some(NoiseModel::Deterministic));
    assert_eq!(
        error_kind(solver.resume_into(
            &StochasticProblem::new(&deterministic),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(deterministic.calls(), 0);
    assert_rejected_resume(workspace.last_attempt());
    check_unchanged(&output);
    let q = Quadratic { center: 0. };
    let p = DeterministicGradient(&q);
    let paired = Paired(&p);
    let pair = Counted::new(&paired, None);
    assert_eq!(
        error_kind(solver.resume_into(
            &StochasticProblem::new(&pair),
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(pair.calls(), 0);
    assert_rejected_resume(workspace.last_attempt());
    check_unchanged(&output);
    assert_eq!(counted.calls(), 0);
    assert_eq!(
        workspace.checkpoint().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );

    // A stop inside an update discards its partial batches without consuming the
    // update allowance; the sample cursor is never rewound.
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(4 + 8 + 4 + 1));
    {
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(&problem, &[1.], &mut output, &mut workspace, &mut parent)
            .unwrap();
    }
    assert_eq!(output.termination(), OptimizeTermination::SampleLimit);
    assert_eq!(output.iterations(), 1);
    assert_eq!(output.sample_cursor(), 16);
    let checkpoint = workspace.checkpoint().unwrap();
    solver
        .resume_into(
            &problem,
            &checkpoint,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(output.iterations(), 8);
    assert_eq!(output.sample_cursor(), expected.sample_cursor() + 4);
    assert!((output.x()[0] - expected.x()[0]).abs() < 0.05);
}

/// Two-coordinate view of a one-coordinate provider, for shape mismatches.
struct Paired<'a, F: ?Sized>(&'a F);
impl<F: SampledObjective<f64> + ?Sized> SampledObjective<f64> for Paired<'_, F> {
    fn dimension(&self) -> usize {
        2
    }
    fn objectives(&self) -> usize {
        1
    }
    fn noise_model(&self) -> NoiseModel {
        self.0.noise_model()
    }
    fn sample_into(&self, x: &[f64], token: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
        self.0.sample_into(&x[..1], token, out)
    }
}
impl<F: SampledGradient<f64> + ?Sized> SampledGradient<f64> for Paired<'_, F> {
    fn gradient_sample_into(
        &self,
        x: &[f64],
        token: u64,
        out: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[1] = 0.;
        self.0.gradient_sample_into(&x[..1], token, &mut out[..1])
    }
}

#[test]
fn trust_checkpoint_resumes_at_a_completed_update() {
    let q = Quadratic { center: 3. };
    let p = DeterministicGradient(&q);
    let problem = StochasticProblem::new(&p);
    let solver = StochasticTrustRegion::new(
        StochasticTrustOptions::new(budget(10), ReplicationOptions::fixed(Budget::ONE))
            .with_radii(1e-12, 0.1, 0.1)
            .with_tolerance(GradientTolerance::new(1e-12, 0.).unwrap()),
    )
    .unwrap();
    let expected = solver.minimize(&problem, &[0.]).unwrap();
    assert_eq!(expected.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(expected.iterations(), 10);
    assert!((expected.x()[0] - 1.).abs() < 1e-12);
    // One initial value, then gradient, base and trial observations per update.
    let boundary = 1 + 3 * 4;
    let mut output = StochasticResult::empty();
    let mut workspace = solver.workspace::<f64>(1).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(boundary));
    {
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(&problem, &[0.], &mut output, &mut workspace, &mut parent)
            .unwrap();
    }
    assert_eq!(output.termination(), OptimizeTermination::SampleLimit);
    assert_eq!(output.iterations(), 4);
    assert!((output.x()[0] - 0.4).abs() < 1e-12);
    let checkpoint = workspace.checkpoint().unwrap();
    let mut resumed = StochasticResult::empty();
    solver
        .resume_into(
            &problem,
            &checkpoint,
            &mut resumed,
            &mut solver.workspace::<f64>(1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(resumed.termination(), expected.termination());
    assert_eq!(resumed.x(), expected.x());
    assert_eq!(resumed.value(), expected.value());
    assert_eq!(resumed.trust_radius(), expected.trust_radius());
    assert_eq!(resumed.iterations(), expected.iterations());
    assert_eq!(resumed.sample_cursor(), expected.sample_cursor());
    // A checkpoint of the other method is rejected.
    let sgd = StochasticGradientDescent::new(SgdOptions::new(
        SgdStep::Constant(0.1),
        budget(10),
        Budget::ONE,
    ))
    .unwrap();
    let mut sgd_workspace = sgd.workspace::<f64>(1).unwrap();
    assert_eq!(
        error_kind(sgd.resume_into(
            &problem,
            &checkpoint,
            &mut resumed,
            &mut sgd_workspace,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_rejected_resume(sgd_workspace.last_attempt());
    assert_eq!(resumed.x(), expected.x());
}

#[test]
fn incomplete_generation_retains_the_complete_parent_population() {
    let starts: Vec<_> = (0..32).map(|i| i as f64 / 16.).collect();
    let solver = nsga::<f64>(3).with_start_design(BorrowedStartDesign::new(&starts, 1).unwrap());
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let mut result = solver.result(&problem).unwrap();
    let mut work = solver.workspace(&problem).unwrap();
    let mut state = SolveState::new();
    // Borrowed starts draw nothing: 32 objective samples, then two variation draws.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(34));
    let mut parent = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(),
            &mut budgets,
        )
        .unwrap();
    solver
        .solve_into(&problem, &mut result, &mut work, &mut parent)
        .unwrap();
    assert_eq!(result.generations(), 0);
    assert_eq!(result.termination(), OptimizeTermination::SampleLimit);
    assert_eq!(population(&result), starts);
    assert_eq!(result.sample_cursor(), 34);
    assert!(result.archive().verify().unwrap());
    assert!(result.archive().entries().all(|e| e.id() < 32));
    assert_eq!(parent.budget_report(BudgetKind::Sample).consumed(), 34);
}

#[test]
fn noise_vector_outputs_retain_means_variances_and_replication_counts() {
    struct Alternating;
    impl OptimizationSampler for Alternating {
        fn configuration_token(&self) -> u128 {
            1
        }
        fn sample(&self, key: OptimizationSampleKey) -> Result<u64, SciOxideError> {
            if key.role == 0 {
                PhiloxSampler.sample(key)
            } else {
                Ok(key.ordinal % 2)
            }
        }
    }
    struct VectorNoise;
    impl SampledObjective<f64> for VectorNoise {
        fn dimension(&self) -> usize {
            1
        }
        fn objectives(&self) -> usize {
            2
        }
        fn sample_into(&self, x: &[f64], token: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
            let noise = if token == 0 { -0.25 } else { 0.25 };
            out[0] = x[0] * x[0] + noise;
            out[1] = (x[0] - 2.).powi(2) + 2. * noise;
            Ok(())
        }
    }
    let solver = Nsga2::new(Nsga2Options::new(8, 8, budget(2)).with_replicates(budget(4))).unwrap();
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&VectorNoise, &b)
        .unwrap()
        .with_sampler(&Alternating);
    let mut result = solver.result(&problem).unwrap();
    let mut work = solver.workspace(&problem).unwrap();
    solver
        .solve_into(
            &problem,
            &mut result,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(result.archive().verify().unwrap());
    assert!(!result.archive().is_empty());
    for e in result.archive().entries() {
        let x = e.point()[0];
        assert!((e.objectives()[0] - x * x).abs() < 1e-14);
        assert!((e.objectives()[1] - (x - 2.).powi(2)).abs() < 1e-14);
        assert_eq!(e.replicates(), 4);
        assert!((e.sample_variances().unwrap()[0] - 1. / 12.).abs() < 1e-14);
        assert!((e.sample_variances().unwrap()[1] - 4. / 12.).abs() < 1e-14);
        assert!((e.standard_errors().unwrap()[0] - (1. / 48_f64).sqrt()).abs() < 1e-14);
    }
    assert_eq!(
        result.counts().own().provider(EvaluationRole::Objective),
        8 * 3 * 4
    );
}

#[test]
fn cancellation_and_deadline_do_not_publish_incomplete_observations() {
    use core::sync::atomic::AtomicU64;
    use scioxide_core::{CancellationToken, Clock, MonotonicInstant};
    struct Timer(AtomicU64);
    impl Clock for Timer {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_nanos(self.0.load(Ordering::Relaxed)),
            ))
        }
    }
    struct Interrupt<'a> {
        token: &'a CancellationToken,
        timer: &'a Timer,
        deadline: bool,
    }
    impl SampledObjective<f64> for Interrupt<'_> {
        fn dimension(&self) -> usize {
            1
        }
        fn objectives(&self) -> usize {
            1
        }
        fn noise_model(&self) -> NoiseModel {
            NoiseModel::Deterministic
        }
        fn sample_into(&self, x: &[f64], _: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = 0.5 * x[0] * x[0];
            if x[0] != 1. {
                if self.deadline {
                    self.timer.0.store(100, Ordering::Relaxed);
                } else {
                    self.token.cancel();
                }
            }
            Ok(())
        }
    }
    impl SampledGradient<f64> for Interrupt<'_> {
        fn gradient_sample_into(
            &self,
            x: &[f64],
            _: u64,
            out: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = x[0];
            Ok(())
        }
    }
    for deadline in [false, true] {
        let token = CancellationToken::new();
        let timer = Timer(AtomicU64::new(0));
        let timer_binding = scioxide_core::ClockBinding::new(&timer);
        let interrupt = Interrupt {
            token: &token,
            timer: &timer,
            deadline,
        };
        let solver = StochasticGradientDescent::new(
            SgdOptions::new(SgdStep::Constant(0.5), budget(10), budget(4))
                .with_time_limit(Duration::from_nanos(50)),
        )
        .unwrap();
        let mut result = StochasticResult::empty();
        let mut work = solver.workspace::<f64>(1).unwrap();
        let execution = ExecutionContext::sequential()
            .with_cancellation(&token)
            .with_clock(&timer_binding)
            .unwrap();
        solver
            .minimize_into(
                &StochasticProblem::new(&interrupt),
                &[1.],
                &mut result,
                &mut work,
                execution,
            )
            .unwrap();
        assert_eq!(result.x(), &[1.]);
        assert_eq!(result.value(), Some(0.5));
        assert_eq!(result.iterations(), 0);
        assert_eq!(
            result.termination(),
            if deadline {
                OptimizeTermination::TimeLimit
            } else {
                OptimizeTermination::Cancelled
            }
        );
        // All slots were admitted before dispatch; an interrupted batch cannot partially commit.
        assert_eq!(result.counts().own().provider(EvaluationRole::Objective), 8);
        assert_eq!(
            work.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
        assert_eq!(work.checkpoint().unwrap().x(), &[1.]);
    }
}

#[test]
fn time_limits_without_a_clock_are_rejected_before_callbacks() {
    let counted = Counted::new(&Noisy, None);
    let problem = StochasticProblem::new(&counted);
    let solver = StochasticGradientDescent::new(
        SgdOptions::new(SgdStep::Constant(0.5), budget(10), budget(2))
            .with_time_limit(Duration::from_secs(1)),
    )
    .unwrap();
    assert_eq!(
        solver.minimize(&problem, &[1.]).unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let mut output = StochasticResult::empty();
    let mut work = solver.workspace::<f64>(1).unwrap();
    assert_eq!(
        error_kind(solver.minimize_into(
            &problem,
            &[1.],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let record = work.last_attempt().unwrap();
    assert_eq!(record.kind(), AttemptKind::Call);
    assert_eq!(record.outcome(), AttemptOutcome::PreflightRejected);
    assert!(output.x().is_empty());
    let trust = StochasticTrustRegion::new(
        StochasticTrustOptions::new(
            budget(10),
            ReplicationOptions::adaptive(budget(2), budget(4)).unwrap(),
        )
        .with_time_limit(Duration::from_secs(1)),
    )
    .unwrap();
    assert_eq!(
        trust.minimize(&problem, &[1.]).unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(counted.calls(), 0);

    let schaffer = Counted::new(&Schaffer, None);
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&schaffer, &b).unwrap();
    let solver =
        Nsga2::new(Nsga2Options::new(8, 8, budget(2)).with_time_limit(Duration::from_secs(1)))
            .unwrap();
    assert_eq!(
        solver.solve(&problem).unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    let mut output = solver.result(&problem).unwrap();
    let mut work = solver.workspace(&problem).unwrap();
    assert_eq!(
        error_kind(solver.solve_into(
            &problem,
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(output.termination(), OptimizeTermination::NotStarted);
    assert_eq!(schaffer.calls(), 0);
}

#[test]
fn plan_constructors_validate_the_whole_configuration() {
    let invalid = |result: Result<(), SciOxideError>| {
        assert_eq!(error_kind(result), SciOxideErrorKind::InvalidInput);
    };
    let ok = |options: Nsga2Options<f64>| Nsga2::new(options).map(|_| ());
    invalid(ok(Nsga2Options::new(3, 8, budget(2))));
    invalid(ok(Nsga2Options::new(8, 0, budget(2))));
    let base = Nsga2Options::new(8, 8, budget(2));
    invalid(ok(base.clone().with_variation(1.5, 15., 0.1, 20.)));
    invalid(ok(base.clone().with_variation(0.9, 15., -0.1, 20.)));
    invalid(ok(base.clone().with_variation(f64::NAN, 15., 0.1, 20.)));
    invalid(ok(base.clone().with_variation(0.9, 0., 0.1, 20.)));
    invalid(ok(base.clone().with_variation(0.9, 15., 0.1, f64::NAN)));
    ok(base.clone().with_variation(0., 15., 1., 20.)).unwrap();
    let solver = Nsga2::new(base).unwrap();
    assert_eq!(solver.options().population(), 8);
    let b = bounds::<f64>();
    let wide = FiniteBounds::new(vec![(0., 1.), (0., 1.)]).unwrap();
    assert!(Nsga2Problem::new(&Schaffer, &wide).is_err());
    let constraints = Constraints::<f64>::new(2).unwrap();
    assert!(
        Nsga2Problem::new(&Schaffer, &b)
            .unwrap()
            .with_constraints(&constraints)
            .is_err()
    );
    let starts = [0.5; 4];
    let short = solver.with_start_design(BorrowedStartDesign::new(&starts, 1).unwrap());
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    assert!(short.workspace_requirements(&problem).is_err());
    assert!(short.solve(&problem).is_err());

    let sgd = |options: SgdOptions<f64>| StochasticGradientDescent::new(options).map(|_| ());
    invalid(sgd(SgdOptions::new(
        SgdStep::Constant(-1.),
        budget(3),
        budget(1),
    )));
    invalid(sgd(SgdOptions::new(
        SgdStep::Constant(f64::INFINITY),
        budget(3),
        budget(1),
    )));
    for (initial, decay, power) in [
        (0.5, 0.1, 0.5),
        (0.5, 0.1, 1.5),
        (0.5, 0., 1.),
        (0., 0.1, 1.),
    ] {
        invalid(sgd(SgdOptions::new(
            SgdStep::Diminishing {
                initial,
                decay,
                power,
            },
            budget(3),
            budget(1),
        )));
    }
    invalid(sgd(SgdOptions::new(
        SgdStep::Constant(0.1),
        budget(3),
        budget(1),
    )
    .with_error_multiplier(0.)));
    sgd(SgdOptions::new(
        SgdStep::Diminishing {
            initial: 0.5,
            decay: 0.1,
            power: 1.,
        },
        budget(3),
        budget(1),
    ))
    .unwrap();

    let trust =
        |options: StochasticTrustOptions<f64>| StochasticTrustRegion::new(options).map(|_| ());
    let base = StochasticTrustOptions::new(budget(3), ReplicationOptions::fixed(budget(2)));
    trust(base).unwrap();
    invalid(trust(base.with_radii(1., 0.5, 2.)));
    invalid(trust(base.with_radii(0.1, 1., 0.5)));
    invalid(trust(base.with_radii(-1., 1., 2.)));
    invalid(trust(base.with_radii(0.1, 1., f64::INFINITY)));
    invalid(trust(base.with_model_policy(1., 1e-3, 2., 0.25)));
    invalid(trust(base.with_model_policy(0., 1e-3, 2., 0.25)));
    invalid(trust(base.with_model_policy(0.1, 1e-3, 1., 0.25)));
    invalid(trust(base.with_model_policy(0.1, 0., 2., 0.25)));
    invalid(trust(base.with_model_policy(0.1, 1e-3, 2., 0.)));
    invalid(trust(StochasticTrustOptions::new(
        budget(3),
        ReplicationOptions::fixed(budget(2)).with_error_multiplier(f64::NAN),
    )));
    assert_eq!(
        ReplicationOptions::<f64>::adaptive(budget(4), budget(2))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let replication = ReplicationOptions::<f64>::adaptive(budget(2), budget(8))
        .unwrap()
        .with_error_multiplier(3.);
    assert_eq!(
        (
            replication.minimum(),
            replication.maximum(),
            replication.error_multiplier()
        ),
        (2, 8, 3.)
    );
}

#[test]
fn invalid_inputs_and_failed_providers_preserve_the_result() {
    struct Broken;
    impl SampledObjective<f64> for Broken {
        fn dimension(&self) -> usize {
            1
        }
        fn objectives(&self) -> usize {
            1
        }
        fn sample_into(&self, _: &[f64], _: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = 0.;
            Ok(())
        }
    }
    impl SampledGradient<f64> for Broken {
        fn gradient_sample_into(
            &self,
            _: &[f64],
            _: u64,
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
    }
    struct FailedSampler;
    impl OptimizationSampler for FailedSampler {
        fn configuration_token(&self) -> u128 {
            99
        }
        fn sample(&self, _: OptimizationSampleKey) -> Result<u64, SciOxideError> {
            Err(invalid(
                scioxide_core::argument!("key"),
                "injected sampler failure",
            ))
        }
    }
    let solver = StochasticGradientDescent::new(SgdOptions::new(
        SgdStep::Constant(0.5),
        budget(3),
        budget(2),
    ))
    .unwrap();
    let problem = StochasticProblem::new(&Noisy);
    let mut result = solver.minimize(&problem, &[1.]).unwrap();
    let before = result.clone();
    let unchanged = |result: &StochasticResult<f64>| {
        assert_eq!(result.x(), before.x());
        assert_eq!(result.value(), before.value());
        assert_eq!(result.sample_cursor(), before.sample_cursor());
        assert_eq!(result.termination(), before.termination());
    };
    let mut work = solver.workspace::<f64>(1).unwrap();
    // The unwritten gradient is rejected after the initial value was accepted.
    assert!(
        solver
            .minimize_into(
                &StochasticProblem::new(&Broken),
                &[1.],
                &mut result,
                &mut work,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    unchanged(&result);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert_eq!(work.last_accepted(), Some(&[1.][..]));
    for x0 in [&[f64::NAN][..], &[], &[1., 2.]] {
        assert!(
            solver
                .minimize_into(
                    &problem,
                    x0,
                    &mut result,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .is_err()
        );
        unchanged(&result);
        assert_eq!(
            work.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
        assert!(work.last_accepted().is_none());
    }
    assert!(
        solver
            .minimize_into(
                &problem.with_sampler(&FailedSampler),
                &[1.],
                &mut result,
                &mut work,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    unchanged(&result);
    let record = work.last_attempt().unwrap();
    assert_eq!(record.outcome(), AttemptOutcome::Failed);
    assert_eq!(record.counts().own().provider(EvaluationRole::Sample), 2);
    assert!(solver.workspace::<f64>(0).is_err());
    assert!(solver.workspace_requirements::<f64>(0).is_err());
    // Noisy trust regions need two replicates to estimate errors.
    let trust = StochasticTrustRegion::new(StochasticTrustOptions::new(
        budget(10),
        ReplicationOptions::fixed(Budget::ONE),
    ))
    .unwrap();
    assert!(trust.minimize(&problem, &[1.]).is_err());
    assert!(ParetoArchive::<f64>::new(usize::MAX, 2, 8).is_err());
}

#[cfg(feature = "std")]
#[test]
fn executor_replays_samples_and_selection_independent_of_workers() {
    let options = SgdOptions::new(SgdStep::Constant(0.2), budget(5), budget(16));
    let problem = StochasticProblem::new(&Noisy);
    let expected = StochasticGradientDescent::new(options)
        .unwrap()
        .minimize(&problem, &[1.])
        .unwrap();
    let b = bounds::<f64>();
    let ns_problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let ns_options = Nsga2Options::new(8, 8, budget(3)).with_replicates(budget(8));
    let ns_expected = Nsga2::new(ns_options.clone())
        .unwrap()
        .solve(&ns_problem)
        .unwrap();
    for workers in [2, 3] {
        let workers = core::num::NonZeroUsize::new(workers).unwrap();
        let pool = scioxide_core::ThreadExecutor::new(workers);
        let solver = StochasticGradientDescent::new(
            options.with_execution(ExecutionPolicy::Parallel(workers)),
        )
        .unwrap();
        let mut out = StochasticResult::empty();
        let mut work = solver.workspace::<f64>(1).unwrap();
        solver
            .minimize_into(
                &problem,
                &[1.],
                &mut out,
                &mut work,
                ExecutionContext::new(&pool),
            )
            .unwrap();
        assert_eq!(out.x(), expected.x());
        assert_eq!(out.value(), expected.value());
        assert_eq!(out.gradient(), expected.gradient());
        assert_eq!(out.counts(), expected.counts());
        assert_eq!(out.sample_cursor(), expected.sample_cursor());
        let ns = Nsga2::new(
            ns_options
                .clone()
                .with_execution(ExecutionPolicy::Parallel(workers)),
        )
        .unwrap();
        let mut out = ns.result(&ns_problem).unwrap();
        let mut work = ns.workspace(&ns_problem).unwrap();
        ns.solve_into(
            &ns_problem,
            &mut out,
            &mut work,
            ExecutionContext::new(&pool),
        )
        .unwrap();
        assert_eq!(
            out.archive()
                .entries()
                .map(|e| e.objectives().to_vec())
                .collect::<Vec<_>>(),
            ns_expected
                .archive()
                .entries()
                .map(|e| e.objectives().to_vec())
                .collect::<Vec<_>>()
        );
        assert_eq!(out.counts(), ns_expected.counts());
        assert_eq!(out.sample_cursor(), ns_expected.sample_cursor());
    }
}

#[test]
fn adaptive_trust_replication_resolves_an_ambiguous_acceptance() {
    // Artificial balanced observations make the variance and the decision independently exact.
    struct Balanced;
    impl OptimizationSampler for Balanced {
        fn configuration_token(&self) -> u128 {
            8
        }
        fn sample(&self, key: OptimizationSampleKey) -> Result<u64, SciOxideError> {
            Ok(key.ordinal % 2)
        }
    }
    struct Model;
    impl SampledObjective<f64> for Model {
        fn dimension(&self) -> usize {
            1
        }
        fn objectives(&self) -> usize {
            1
        }
        fn sample_into(&self, x: &[f64], token: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = 0.5 * x[0] * x[0] + if token == 0 { -1. } else { 1. };
            Ok(())
        }
    }
    impl SampledGradient<f64> for Model {
        fn gradient_sample_into(
            &self,
            x: &[f64],
            _: u64,
            out: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = x[0];
            Ok(())
        }
    }
    let solver = StochasticTrustRegion::new(
        StochasticTrustOptions::new(
            budget(1),
            ReplicationOptions::adaptive(budget(2), budget(16)).unwrap(),
        )
        .with_tolerance(GradientTolerance::new(1e-12, 0.).unwrap()),
    )
    .unwrap();
    let mut result = StochasticResult::empty();
    let mut work = solver.workspace::<f64>(1).unwrap();
    solver
        .minimize_into(
            &StochasticProblem::new(&Model).with_sampler(&Balanced),
            &[2.],
            &mut result,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(result.x(), &[1.]);
    assert_eq!(result.value(), Some(0.5));
    assert_eq!(result.value_replicates(), 8);
    assert_eq!(result.trust_radius(), 2.);
    assert_eq!(
        result.counts().own().provider(EvaluationRole::Objective),
        18
    ); // 2 initial + 8 base + 8 trial
    assert_eq!(result.counts().own().provider(EvaluationRole::Gradient), 4);
}

#[test]
fn dense_view_sparse_and_operator_gradient_providers_share_the_engine() {
    use scioxide_core::LinearOperator;
    use scioxide_linalg::Matrix;
    use scioxide_sparse::CsrMatrix;
    struct OperatorModel<'a, O: ?Sized>(&'a O);
    impl<O: LinearOperator<f64> + Sync + ?Sized> crate::SmoothGradient<f64> for OperatorModel<'_, O> {
        fn dimension(&self) -> usize {
            2
        }
        fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            let mut y = [0.; 2];
            let mut scratch = [0.; 2];
            self.0
                .apply_into(x, &mut y, &mut scratch[..self.0.apply_workspace_len()?])?;
            Ok(0.5 * (x[0] * y[0] + x[1] * y[1]) - 2. * x[0] + 8. * x[1])
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            let mut scratch = [0.; 2];
            self.0
                .apply_into(x, out, &mut scratch[..self.0.apply_workspace_len()?])?;
            out[0] -= 2.;
            out[1] += 8.;
            Ok(())
        }
    }
    struct Diagonal;
    impl LinearOperator<f64> for Diagonal {
        fn rows(&self) -> usize {
            2
        }
        fn cols(&self) -> usize {
            2
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = 2. * x[0];
            out[1] = 4. * x[1];
            Ok(())
        }
    }
    let dense = Matrix::from_vec(2, 2, vec![2., 0., 0., 4.]).unwrap();
    let view = dense.view();
    let sparse = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2., 4.]).unwrap();
    let operators: [&(dyn LinearOperator<f64> + Sync); 4] = [
        &dense.view().try_finite().unwrap(),
        &view.try_finite().unwrap(),
        &sparse,
        &Diagonal,
    ];
    let sgd = StochasticGradientDescent::new(
        SgdOptions::new(SgdStep::Constant(0.2), budget(100), Budget::ONE)
            .with_tolerance(GradientTolerance::new(1e-7, 0.).unwrap()),
    )
    .unwrap();
    let trust = StochasticTrustRegion::new(
        StochasticTrustOptions::new(budget(300), ReplicationOptions::fixed(Budget::ONE))
            .with_tolerance(GradientTolerance::new(1e-6, 0.).unwrap()),
    )
    .unwrap();
    for operator in operators {
        let model = OperatorModel(operator);
        let objective = DeterministicGradient(&model);
        let problem = StochasticProblem::new(&objective);
        // Independent symmetric finite differences check the injected gradient convention.
        let point = [0.3, -0.7];
        let mut gradient = [0.; 2];
        crate::SmoothGradient::gradient_into(&model, &point, &mut gradient).unwrap();
        for j in 0..2 {
            let mut left = point;
            let mut right = point;
            left[j] -= 1e-5;
            right[j] += 1e-5;
            let difference = (crate::SmoothGradient::value(&model, &right).unwrap()
                - crate::SmoothGradient::value(&model, &left).unwrap())
                / 2e-5;
            assert!((difference - gradient[j]).abs() < 1e-9);
        }
        let result = sgd.minimize(&problem, &[0., 0.]).unwrap();
        assert!(result.is_success());
        assert!((result.x()[0] - 1.).abs() < 1e-7);
        assert!((result.x()[1] + 2.).abs() < 1e-7);
        let result = trust.minimize(&problem, &[0., 0.]).unwrap();
        assert!(result.is_success(), "{:?}", result.termination());
        assert!((result.x()[0] - 1.).abs() < 1e-6);
        assert!((result.x()[1] + 2.).abs() < 1e-6);
    }
}

#[test]
fn executor_failure_and_nsga_callback_failure_do_not_commit_output() {
    struct FailedExecutor;
    impl scioxide_core::Executor for FailedExecutor {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::MIN
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: scioxide_core::ExecutionPlan,
            _: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            Err(SciOxideError::execution_failure(
                None,
                scioxide_core::ExecutionFailureReason::ProviderFailure,
                "injected failure",
            ))
        }
    }
    let solver = StochasticGradientDescent::new(
        SgdOptions::new(SgdStep::Constant(0.2), budget(3), budget(4))
            .with_execution(ExecutionPolicy::Auto),
    )
    .unwrap();
    let problem = StochasticProblem::new(&Noisy);
    let mut result = solver.minimize(&problem, &[1.]).unwrap();
    let before = result.clone();
    let mut work = solver.workspace::<f64>(1).unwrap();
    assert!(
        solver
            .minimize_into(
                &problem,
                &[2.],
                &mut result,
                &mut work,
                ExecutionContext::new(&FailedExecutor),
            )
            .is_err()
    );
    assert_eq!(result.x(), before.x());
    assert_eq!(result.value(), before.value());
    assert_eq!(result.sample_cursor(), before.sample_cursor());
    let record = work.last_attempt().unwrap();
    assert_eq!(record.outcome(), AttemptOutcome::Failed);
    assert_eq!(record.counts().own().provider(EvaluationRole::Objective), 4);
    assert!(work.last_accepted().is_none());

    struct Bad;
    impl SampledObjective<f64> for Bad {
        fn dimension(&self) -> usize {
            1
        }
        fn objectives(&self) -> usize {
            2
        }
        fn sample_into(&self, _: &[f64], _: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = 0.;
            Ok(())
        }
    }
    let solver = nsga::<f64>(2);
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let mut result = solver.solve(&problem).unwrap();
    let before = points(result.archive());
    let mut work = solver.workspace(&problem).unwrap();
    assert!(
        solver
            .solve_into(
                &Nsga2Problem::new(&Bad, &b).unwrap(),
                &mut result,
                &mut work,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(before, points(result.archive()));
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert!(work.last_accepted().is_none());
}

#[test]
fn constant_third_objective_and_empty_constraint_workspace_preflight() {
    let solver = nsga::<f64>(4);
    let b = bounds::<f64>();
    let problem = Nsga2Problem::new(&Schaffer, &b).unwrap();
    let two = solver.solve(&problem).unwrap();
    let three = solver
        .solve(&Nsga2Problem::new(&Three, &b).unwrap())
        .unwrap();
    assert_eq!(points(two.archive()), points(three.archive()));
    assert_eq!(
        two.archive().crowding_distances(),
        three.archive().crowding_distances()
    );
    assert!(three.archive().hypervolume_2d([5., 5.]).is_err());
    // A result of another objective count is rejected before any callback.
    let counted = Counted::new(&Three, None);
    let mut output = solver.result(&problem).unwrap();
    let mut work = solver.workspace(&problem).unwrap();
    assert_eq!(
        error_kind(solver.solve_into(
            &Nsga2Problem::new(&counted, &b).unwrap(),
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(counted.calls(), 0);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    // Empty constraints prepare their own workspace and leave the search unchanged.
    let constraints = Constraints::new(1).unwrap();
    let constrained = problem.with_constraints(&constraints).unwrap();
    let mut work = solver.workspace(&problem).unwrap();
    solver
        .solve_into(
            &constrained,
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(points(output.archive()), points(two.archive()));
    assert_eq!(output.sample_cursor(), two.sample_cursor());
}

#[test]
fn hypervolume_rounds_complete_thin_rectangles_without_width_overflow() {
    let mut a = ParetoArchive::new(1, 2, 1).unwrap();
    a.insert(&[0.], &[-f64::MAX, 0.], &[]).unwrap();
    assert_eq!(
        a.hypervolume_2d([f64::MAX, f64::MIN_POSITIVE]).unwrap(),
        4. * (2. - f64::EPSILON)
    );
    assert_eq!(a.hypervolume_2d([f64::MAX, 0.]).unwrap(), 0.);
    assert!(a.hypervolume_2d([f64::MAX, 1.]).is_err());
    let mut a = ParetoArchive::new(1, 2, 1).unwrap();
    a.insert(&[0_f32], &[-f32::MAX, 0.], &[]).unwrap();
    assert_eq!(
        a.hypervolume_2d([f32::MAX, f32::MIN_POSITIVE]).unwrap(),
        4. * (2. - f32::EPSILON)
    );
    assert_eq!(a.hypervolume_2d([f32::MAX, 0.]).unwrap(), 0.);
}
