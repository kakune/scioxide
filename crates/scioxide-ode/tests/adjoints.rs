#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_ad::CheckpointSchedule;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Capacity, Complex, ErrorTolerance, ExecutionContext,
    NormSelection, NumericScalar, ProblemToken, RealScalar, SciOxideError, SciOxideErrorKind,
    SolveBudgets, SolveContext,
};
use scioxide_linalg::{MatrixView, MatrixViewMut, Vector};
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::*;

struct Exponential<S: RealScalar> {
    parameter: [S; 1],
}
impl<T: NumericScalar> OdeSystem<T::Real, T> for Exponential<T::Real> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: T::Real,
        y: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = T::from_real(self.parameter[0]) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: T::Real,
        _: &[T],
        output: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.set(0, 0, T::from_real(self.parameter[0]))?;
        Ok(Some(()))
    }
}
impl<T: NumericScalar> OdeAdjointSystem<T> for Exponential<T::Real> {
    fn parameters(&self) -> &[T::Real] {
        &self.parameter
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: T::Real,
        y: &[T],
        seed: &[T],
        state: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state[0] = T::from_real(self.parameter[0]) * seed[0];
        parameters[0] = (seed[0].conj() * y[0]).real_component();
        Ok(())
    }
}
struct Energy(f64);
impl<T: NumericScalar> AdjointObjective<T> for Energy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: T::Real,
        y: &[T],
        _: &[T::Real],
        state: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        let c = T::Real::try_from_f64_lossy(if part == AdjointObjectivePart::Running {
            self.0
        } else {
            1.
        })?;
        state[0] = T::from_real(c) * y[0];
        parameters.fill(T::Real::ZERO);
        Ok(AdjointObjectiveValue {
            value: c * y[0].norm_sqr() * T::Real::HALF,
            time_derivative: T::Real::ZERO,
        })
    }
}
/// Common options with the structural limits shared by these tests.
fn common<S: RealScalar>(mode: AdjointMode, step: S) -> AdjointOptions<S> {
    AdjointOptions::new(mode, step)
        .unwrap()
        .with_tape_capacity(128)
        .with_max_replay_steps(20000)
        .with_max_reverse_steps(10000)
        .with_max_workspace_bytes(2 * 1024 * 1024)
}
/// ODE adjoint options with the shared structural limits.
fn options<S: RealScalar>(mode: AdjointMode, step: S) -> OdeAdjointOptions<S> {
    OdeAdjointOptions::new(common(mode, step))
}
/// The progress of the workspace's last attempt, which must have failed.
fn failed<T: NumericScalar>(workspace: &OdeAdjointWorkspace<T>) -> AdjointReport<T::Real> {
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    *attempt.progress().unwrap()
}
fn exponential<T: NumericScalar>(y0: T, tolerance: f64) {
    let model = Exponential {
        parameter: [T::Real::try_from_f64_lossy(-0.7).unwrap()],
    };
    let objective = Energy(0.3);
    let t = 0.6_f64;
    let p = -0.7_f64;
    let exp = (2. * p * t).exp();
    let yscale = exp + 0.3 * (exp - 1.) / (2. * p);
    let pscale = t * exp + 0.3 * (2. * p * t * exp - exp + 1.) / (4. * p * p);
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        for checkpoint in [
            CheckpointPolicy::StoreAll,
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(128, 2).unwrap()),
        ] {
            let error = T::Real::try_from_f64_lossy(tolerance * 0.01).unwrap();
            let error = ErrorTolerance::scalar(error, error, NormSelection::Infinity).unwrap();
            let options = OdeAdjointOptions::new(
                common(mode, T::Real::try_from_f64_lossy(0.1).unwrap())
                    .with_continuous_error(error.clone())
                    .with_quadrature_error(error.clone())
                    .with_checkpoint_policy(checkpoint),
            )
            .with_primal_error(error);
            let plan = OdeAdjointSolver::new(1, 1, options).unwrap();
            let result = plan
                .solve(&scioxide_ode::OdeAdjointProblem::new(
                    &model,
                    &objective,
                    &[y0],
                    (T::Real::ZERO, T::Real::try_from_f64_lossy(t).unwrap()),
                ))
                .unwrap();
            assert_eq!(
                result.report().termination(),
                Some(AdjointTermination::ReachedEnd)
            );
            assert!(
                (result.initial_gradient()[0]
                    - T::from_real(T::Real::try_from_f64_lossy(yscale).unwrap()) * y0)
                    .abs()
                    .into_f64()
                    < tolerance,
                "{:?} {:?}",
                mode,
                result
            );
            assert!(
                (result.parameter_gradient()[0].into_f64() - y0.norm_sqr().into_f64() * pscale)
                    .abs()
                    < tolerance * 3.,
                "{:?} {:?}",
                mode,
                result
            );
            assert_eq!(
                result.report().accepted_steps(),
                result.report().reverse_steps()
            );
        }
    }
}
#[test]
fn analytic_terminal_and_running_real_and_complex() {
    exponential(1.25_f64, 2e-8);
    exponential(Complex::new(1.25_f64, -0.5), 3e-8);
    exponential(1.25_f32, 5e-4);
    exponential(Complex::new(1.25_f32, -0.5), 8e-4);
}

#[test]
fn discrete_dp5_matches_independent_stability_polynomial() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let plan = OdeAdjointSolver::new(
        1,
        1,
        OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.125).with_fixed_step(true)),
    )
    .unwrap();
    let result = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    let z = 0.4_f64 * 0.125;
    let r = 1.
        + z
        + z.powi(2) / 2.
        + z.powi(3) / 6.
        + z.powi(4) / 24.
        + z.powi(5) / 120.
        + z.powi(6) / 600.;
    let dr = 1. + z + z.powi(2) / 2. + z.powi(3) / 6. + z.powi(4) / 24. + z.powi(5) / 100.;
    assert!((result.initial_gradient()[0] - r.powi(8)).abs() < 3e-14);
    assert!((result.parameter_gradient()[0] - 0.5 * dr * r.powi(7)).abs() < 3e-14);
    assert_eq!(result.report().replayed_steps(), 0);
}

/// Counts pullbacks and those whose seed is identically zero.
struct CountedExponential {
    model: Exponential<f64>,
    pullbacks: core::cell::Cell<usize>,
    zero_seeds: core::cell::Cell<usize>,
}
impl OdeSystem<f64> for CountedExponential {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.model.rhs(t, y, out, cx)
    }
}
impl OdeAdjointSystem<f64> for CountedExponential {
    fn parameters(&self) -> &[f64] {
        &self.model.parameter
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        t: f64,
        y: &[f64],
        seed: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        work: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.pullbacks.set(self.pullbacks.get() + 1);
        if seed.iter().all(|&value| value == 0.) {
            self.zero_seeds.set(self.zero_seeds.get() + 1);
        }
        self.model.pullback(t, y, seed, state, parameters, work, cx)
    }
}

#[test]
fn discrete_dp5_skips_the_zero_seeded_last_stage_of_full_steps() {
    let model = CountedExponential {
        model: Exponential { parameter: [0.4] },
        pullbacks: core::cell::Cell::new(0),
        zero_seeds: core::cell::Cell::new(0),
    };
    for objective in [Energy(0.), Energy(0.3)] {
        model.pullbacks.set(0);
        model.zero_seeds.set(0);
        let plan = OdeAdjointSolver::new(
            1,
            1,
            OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.125).with_fixed_step(true)),
        )
        .unwrap();
        let result = plan
            .solve(&OdeAdjointProblem::new(
                &model,
                &objective,
                &[1.],
                (0., 0.5),
            ))
            .unwrap();
        let steps = result.report().reverse_steps();
        assert_eq!((steps, result.report().replayed_steps()), (4, 0));
        // b₇ = 0, so every full step pulls back stages one to six only.
        assert_eq!(model.pullbacks.get(), 6 * steps);
        assert_eq!(model.zero_seeds.get(), 0);
        if objective.0 == 0. {
            let z = 0.4_f64 * 0.125;
            let r = 1.
                + z
                + z.powi(2) / 2.
                + z.powi(3) / 6.
                + z.powi(4) / 24.
                + z.powi(5) / 120.
                + z.powi(6) / 600.;
            let dr = 1. + z + z.powi(2) / 2. + z.powi(3) / 6. + z.powi(4) / 24. + z.powi(5) / 100.;
            assert!((result.initial_gradient()[0] - r.powi(8)).abs() < 3e-14);
            assert!((result.parameter_gradient()[0] - 0.5 * dr * r.powi(7)).abs() < 3e-14);
        }
    }
}

struct Level;
impl AdjointTerminalEvent<f64> for Level {
    fn unique_branch(&self) -> bool {
        true
    }
    fn transversality_threshold(&self) -> f64 {
        1e-8
    }
    fn value(&self, _: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(y[0] - 2.)
    }
    fn gradient(
        &self,
        _: f64,
        _: &[f64],
        y: &mut [f64],
        p: &mut [f64],
    ) -> Result<f64, SciOxideError> {
        y[0] = 1.;
        p[0] = 0.;
        Ok(0.)
    }
}
struct TerminalTime;
impl AdjointObjective<f64> for TerminalTime {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        time: f64,
        _: &[f64],
        _: &[f64],
        y: &mut [f64],
        p: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        y.fill(0.);
        p.fill(0.);
        let c = if part == AdjointObjectivePart::Terminal {
            1.
        } else {
            0.
        };
        Ok(AdjointObjectiveValue {
            value: c * time,
            time_derivative: c,
        })
    }
}
#[test]
fn transverse_terminal_event_includes_time_derivative() {
    let model = Exponential {
        parameter: [0.7_f64],
    };
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let error = ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap();
        let options = OdeAdjointOptions::new(
            common(mode, 0.1)
                .with_continuous_error(error.clone())
                .with_checkpoint_policy(CheckpointPolicy::Explicit(
                    CheckpointSchedule::binomial(128, 2).unwrap(),
                )),
        )
        .with_primal_error(error);
        let plan = OdeAdjointSolver::new(1, 1, options).unwrap();
        let result = plan
            .solve(
                &scioxide_ode::OdeAdjointProblem::new(&model, &TerminalTime, &[1.], (0., 2.))
                    .with_terminal_event(Some(&Level)),
            )
            .unwrap();
        assert_eq!(
            result.report().termination(),
            Some(AdjointTermination::TerminalEvent)
        );
        assert!((result.report().final_time() - 2_f64.ln() / 0.7).abs() < 1e-9);
        assert!(
            (result.initial_gradient()[0] + 1. / 0.7).abs() < 1e-8,
            "{result:?}"
        );
        assert!(
            (result.parameter_gradient()[0] + 2_f64.ln() / 0.7_f64.powi(2)).abs() < 1e-8,
            "{result:?}"
        );
    }
}

#[test]
fn prepared_terminal_event_adjoints_execute() {
    let model = Exponential {
        parameter: [0.7_f64],
    };
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let error = ErrorTolerance::scalar(1e-10, 1e-10, NormSelection::Infinity).unwrap();
        let plan = OdeAdjointSolver::new(
            1,
            1,
            OdeAdjointOptions::new(
                common(mode, 0.1)
                    .with_continuous_error(error.clone())
                    .with_checkpoint_policy(CheckpointPolicy::StoreAll),
            )
            .with_primal_error(error),
        )
        .unwrap();
        let problem = OdeAdjointProblem::new(&model, &TerminalTime, &[1.], (0., 2.))
            .with_terminal_event(Some(&Level));
        let mut workspace = plan.workspace(&problem).unwrap();
        let (mut y, mut p) = ([0.], [0.]);
        for _ in 0..2 {
            let report = plan
                .solve_into(
                    &problem,
                    &mut y,
                    &mut p,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                Some(AdjointTermination::TerminalEvent)
            );
            // The event time ln(2)/0.7 has gradients -1/0.7 and -ln(2)/0.7².
            assert!((y[0] + 1. / 0.7).abs() < 1e-7, "{mode:?} {y:?}");
            assert!((p[0] + 2_f64.ln() / 0.49).abs() < 1e-7, "{mode:?} {p:?}");
        }
    }
}

fn identity() -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        ProblemToken::new(624),
        scioxide_core::algorithm_id!("test", "adjoints", 1),
        0,
        0,
    )
}

#[test]
fn checkpoint_bytes_work_boundaries_and_prepared_allocations() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let objective = Energy(0.3);
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let base = common(mode, 0.125).with_fixed_step(true);
        let primal = ErrorTolerance::try_vector(
            &[f64::EPSILON.sqrt()],
            f64::EPSILON.sqrt(),
            NormSelection::Infinity,
        )
        .unwrap();
        let adjoint = |common: AdjointOptions<f64>| {
            OdeAdjointOptions::new(common).with_primal_error(primal.clone())
        };
        let full = OdeAdjointSolver::<f64>::new(
            1,
            1,
            adjoint(
                base.clone()
                    .with_checkpoint_policy(CheckpointPolicy::StoreAll),
            ),
        )
        .unwrap();
        let bytes = full
            .workspace_requirements(&scioxide_ode::OdeAdjointProblem::new(
                &model,
                &objective,
                &[],
                (
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                ),
            ))
            .unwrap()
            .required_bytes();
        assert!(
            OdeAdjointSolver::<f64>::new(
                1,
                1,
                adjoint(
                    base.clone()
                        .with_max_workspace_bytes(bytes)
                        .with_checkpoint_policy(CheckpointPolicy::StoreAll)
                )
            )
            .is_ok()
        );
        assert!(
            OdeAdjointSolver::<f64>::new(
                1,
                1,
                adjoint(
                    base.clone()
                        .with_max_workspace_bytes(bytes - 1)
                        .with_checkpoint_policy(CheckpointPolicy::StoreAll)
                )
            )
            .is_err()
        );
        let bounded = OdeAdjointSolver::<f64>::new(
            1,
            1,
            adjoint(
                base.clone()
                    .with_max_workspace_bytes(bytes - 1)
                    .with_checkpoint_policy(CheckpointPolicy::MemoryBudget),
            ),
        )
        .unwrap();
        assert!(
            bounded
                .workspace_requirements(&scioxide_ode::OdeAdjointProblem::new(
                    &model,
                    &objective,
                    &[],
                    (
                        <_ as scioxide_core::AlgebraicScalar>::ZERO,
                        <_ as scioxide_core::AlgebraicScalar>::ZERO
                    )
                ))
                .unwrap()
                .required_bytes()
                < bytes
        );
        for plan in [full, bounded] {
            let mut workspace = plan
                .workspace(&scioxide_ode::OdeAdjointProblem::new(
                    &model,
                    &objective,
                    &[1.],
                    (0., 0.5),
                ))
                .unwrap();
            let mut y = [99.; 3];
            let mut p = [98.; 3];
            for _ in 0..2 {
                let report = plan
                    .solve_into(
                        &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 0.5)),
                        &mut y[..1],
                        &mut p[..1],
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(report.is_success());
                assert_eq!(&y[1..], &[99., 99.]);
                assert_eq!(&p[1..], &[98., 98.]);
            }
        }
        assert!(
            OdeAdjointSolver::<f64>::new(
                1,
                1,
                adjoint(
                    base.with_max_replay_steps(0)
                        .with_max_workspace_bytes(bytes - 1)
                        .with_checkpoint_policy(CheckpointPolicy::MemoryBudget)
                )
            )
            .is_err()
        );
    }
}

#[test]
fn replay_limits_are_decided_at_construction() {
    let model = Exponential { parameter: [-0.7] };
    let schedule = CheckpointSchedule::binomial(8, 1).unwrap();
    let options = |limit: usize| {
        OdeAdjointOptions::new(
            AdjointOptions::new(AdjointMode::Discrete, 0.0625)
                .unwrap()
                .with_fixed_step(true)
                .with_tape_capacity(8)
                .with_max_replay_steps(limit)
                .with_max_reverse_steps(10000)
                .with_max_workspace_bytes(1 << 20)
                .with_checkpoint_policy(CheckpointPolicy::Explicit(schedule.clone())),
        )
    };
    let error =
        OdeAdjointSolver::<f64>::new(1, 1, options(schedule.forward_steps() - 1)).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    // A schedule admitted at the step capacity stays within the limit for every
    // realized step count, so replay never stops early.
    let plan = OdeAdjointSolver::<f64>::new(1, 1, options(schedule.forward_steps())).unwrap();
    for steps in 1..=8 {
        let end = 0.0625 * f64::from(steps);
        let problem = OdeAdjointProblem::new(&model, &Energy(0.), &[1.], (0., end));
        let mut workspace = plan.workspace(&problem).unwrap();
        let (mut y, mut p) = ([0.], [0.]);
        let report = plan
            .solve_into(
                &problem,
                &mut y,
                &mut p,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(report.accepted_steps(), steps as usize);
        assert_eq!(
            report.replayed_steps(),
            schedule.forward_steps_for(steps as usize).unwrap()
        );
        assert!(report.replayed_steps() <= schedule.forward_steps());
        // d(y(T)²/2)/dy₀ = y₀ e^{2pT} up to the DP5 discretization error.
        assert!((y[0] - (-1.4 * end).exp()).abs() < 1e-6, "{steps}: {y:?}");
    }
}

#[test]
fn explicit_boundaries_and_failures_preserve_outputs() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let objective = Energy(0.);
    let base = common(AdjointMode::Discrete, 0.125).with_fixed_step(true);
    let schedule =
        || CheckpointPolicy::Explicit(CheckpointSchedule::explicit(128, &[0, 1, 3]).unwrap());
    let plan = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(base.clone().with_checkpoint_policy(schedule())),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    let mut y = [12.];
    let mut p = [13.];
    // An ancestor evaluation budget is shared with the adjoint's child epoch.
    let mut epoch = scioxide_core::SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = epoch
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let error = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 0.5)),
            &mut y,
            &mut p,
            &mut workspace,
            &mut context,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(
        failed(&workspace).termination(),
        Some(AdjointTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert_eq!(y, [12.]);
    assert_eq!(p, [13.]);
    assert_eq!(context.budget_report(BudgetKind::Evaluation).consumed(), 1);
    // The adjoint's own evaluation budget.
    let limited = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(
            base.clone()
                .with_checkpoint_policy(schedule())
                .with_max_evaluations(Budget::ONE),
        ),
    )
    .unwrap();
    let mut limited_workspace = limited
        .workspace(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    let error = limited
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 0.5)),
            &mut y,
            &mut p,
            &mut limited_workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    let progress = failed(&limited_workspace);
    assert_eq!(
        progress.termination(),
        Some(AdjointTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert_eq!(progress.counts().total_providers().unwrap(), 1);
    assert_eq!(y, [12.]);
    assert_eq!(p, [13.]);
    let report = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 0.5)),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.replayed_steps(), 5);
    assert!(y[0] > 1.);
    // A workspace prepared for a differently bounded plan is replaced before the solve.
    let small = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(
            base.with_tape_capacity(1)
                .with_checkpoint_policy(CheckpointPolicy::StoreAll),
        ),
    )
    .unwrap();
    let mut small_workspace = small
        .workspace(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[],
            (
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ),
        ))
        .unwrap();
    let requirements = plan
        .workspace_requirements(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    assert_eq!(
        small_workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let before = (y, p);
    let replaced = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 0.5)),
            &mut y,
            &mut p,
            &mut small_workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(replaced.replayed_steps(), 5);
    assert_eq!((y, p), before);
    assert_eq!(
        small_workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
}

#[test]
fn backward_euler_reverses_the_accepted_full_step() {
    let model = Exponential {
        parameter: [-8_f64],
    };
    let residual = OdeAdjointResidual::<f64, _>::new(&model).unwrap();
    let dae = DaeOptions::new().with_numerics(
        scioxide_ode::DaeNumericalOptions::new()
            .with_differential_residual(scioxide_core::ResidualTolerance::new(1e-12, 0.).unwrap()),
    );
    let plan = DaeAdjointSolver::backward_euler(
        residual.dimension(),
        residual.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(
            (OdeAdjointOptions::new(
                common(AdjointMode::Discrete, 0.125)
                    .with_fixed_step(true)
                    .with_checkpoint_policy(CheckpointPolicy::Explicit(
                        CheckpointSchedule::binomial(128, 2).unwrap(),
                    )),
            ))
            .common()
            .clone(),
            (dae).numerics().clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let result = plan
        .solve(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.),
            &[1.],
            &[-8.],
            (0., 0.5),
        ))
        .unwrap();
    let r = 0.5_f64;
    assert!((result.initial_gradient()[0] - r.powi(8)).abs() < 1e-14);
    assert!((result.parameter_gradient()[0] - 0.5 * r.powi(9)).abs() < 1e-14);
}

#[test]
fn zero_duration_and_backward_time_reports() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let objective = Energy(0.);
    let plan = OdeAdjointSolver::<f64>::new(1, 1, options(AdjointMode::Discrete, 0.125)).unwrap();
    let zero = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[2.],
            (1., 1.),
        ))
        .unwrap();
    assert_eq!(zero.initial_gradient(), [2.]);
    assert_eq!(zero.parameter_gradient(), [0.]);
    assert_eq!(zero.report().accepted_steps(), 0);
    let coarse = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[2.],
            (1., 0.),
        ))
        .unwrap();
    assert!((coarse.initial_gradient()[0] - 2. * (-0.8_f64).exp()).abs() < 1e-7);
    assert!((coarse.parameter_gradient()[0] + 4. * (-0.8_f64).exp()).abs() < 2e-7);
    let fine_plan = OdeAdjointSolver::new(
        1,
        1,
        options(AdjointMode::Discrete, 0.0625).with_primal_error(
            ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap(),
        ),
    )
    .unwrap();
    let fine = fine_plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[2.],
            (1., 0.),
        ))
        .unwrap();
    // The refined gradients agree with the coarse ones.
    let difference = fine
        .initial_gradient()
        .iter()
        .zip(coarse.initial_gradient())
        .chain(
            fine.parameter_gradient()
                .iter()
                .zip(coarse.parameter_gradient()),
        )
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(difference < 2e-7);
}

struct ChangedReplay {
    calls: core::cell::Cell<usize>,
    parameters: [f64; 1],
}
impl OdeSystem<f64> for ChangedReplay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        out[0] = self.parameters[0] * y[0] + if calls >= 28 { 0.1 } else { 0. };
        Ok(())
    }
}
impl OdeAdjointSystem<f64> for ChangedReplay {
    fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        y: &[f64],
        seed: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state[0] = self.parameters[0] * seed[0];
        parameters[0] = y[0] * seed[0];
        Ok(())
    }
}
#[test]
fn changed_replay_is_detected_without_publishing_gradients() {
    let model = ChangedReplay {
        calls: core::cell::Cell::new(0),
        parameters: [0.4],
    };
    let plan = OdeAdjointSolver::new(
        1,
        1,
        OdeAdjointOptions::new(
            common(AdjointMode::Discrete, 0.125)
                .with_fixed_step(true)
                .with_checkpoint_policy(CheckpointPolicy::Explicit(
                    CheckpointSchedule::binomial(128, 1).unwrap(),
                )),
        ),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    let mut y = [12.];
    let mut p = [13.];
    let error = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &Energy(0.), &[1.], (0., 0.5)),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(
        failed(&workspace).termination(),
        Some(AdjointTermination::ReplayMismatch)
    );
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ReplayMismatch,
            ..
        }
    ));
    assert_eq!((y, p), ([12.], [13.]));
}

#[test]
fn cancelled_execution_and_initial_event_are_unavailable() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let plan = OdeAdjointSolver::<f64>::new(1, 1, options(AdjointMode::Discrete, 0.125)).unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    let mut y = [12.];
    let mut p = [13.];
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let error = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &Energy(0.), &[1.], (0., 0.5)),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert!(matches!(
        failed(&workspace).termination(),
        Some(AdjointTermination::ExecutionStopped(_))
    ));
    assert_eq!((y, p), ([12.], [13.]));
    let failure = plan
        .solve(
            &scioxide_ode::OdeAdjointProblem::new(&model, &Energy(0.), &[2.], (0., 0.5))
                .with_terminal_event(Some(&Level)),
        )
        .unwrap_err();
    assert_eq!(
        failure.progress().unwrap().termination(),
        Some(AdjointTermination::EventDerivativeUnavailable)
    );
}

struct AdModel {
    parameter: [f64; 1],
    tape: core::cell::RefCell<scioxide_ad::Tape<f64>>,
}
impl OdeSystem<f64> for AdModel {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = self.parameter[0] * y[0];
        Ok(())
    }
}
impl OdeAdjointSystem<f64> for AdModel {
    fn parameters(&self) -> &[f64] {
        &self.parameter
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        y: &[f64],
        seed: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mut tape = self.tape.borrow_mut();
        let mut point = tape.linearize(
            |x, f| {
                f.push(x[0] * x[1]);
                Ok(())
            },
            &[y[0], self.parameter[0]],
            context.execution(),
        )?;
        let mut result = [0.; 2];
        point.vjp_into(seed, &mut result, context.execution())?;
        state[0] = result[0];
        parameters[0] = result[1];
        Ok(())
    }
}
#[test]
fn supplied_ad_vjp_composes_with_checkpoint_replay() {
    let model = AdModel {
        parameter: [0.4],
        tape: core::cell::RefCell::new(scioxide_ad::Tape::new(scioxide_ad::TapeOptions::new())),
    };
    let plan = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.125).with_checkpoint_policy(
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(128, 1).unwrap()),
        )),
    )
    .unwrap();
    let result = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    assert!((result.initial_gradient()[0] - 0.4_f64.exp()).abs() < 1e-7);
    assert!((result.parameter_gradient()[0] - 0.5 * 0.4_f64.exp()).abs() < 1e-7);
}

struct TangentModel;
impl OdeSensitivitySystem<f64> for TangentModel {
    fn dimension(&self) -> usize {
        1
    }
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        p: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = p[0] * y[0];
        Ok(())
    }
    fn tangent_rhs(
        &self,
        _: f64,
        y: &[f64],
        p: &[f64],
        dy: &[f64],
        dp: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = p[0] * dy[0] + dp[0] * y[0];
        Ok(())
    }
}
#[test]
fn adjoint_agrees_with_existing_forward_sensitivity_lift() {
    let mut sensitivity = scioxide_ode::ForwardSensitivityWorkspace::new();
    let system = scioxide_ode::ForwardSensitivitySystem::bind(
        &TangentModel,
        &[0.4],
        MatrixView::from_slice(1, 1, &[1.]).unwrap(),
        &mut sensitivity,
    )
    .unwrap();
    let solver = (
        DormandPrince5::new(),
        OdeOptions::new().with_error_tolerance(
            ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap(),
        ),
    );
    let solution = solver
        .0
        .solve(
            &OdeProblem::new(&system, 0.5),
            &solver.1,
            OdeState::new(0., Vector::from_vec(vec![1., 0.])),
        )
        .unwrap();
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let plan = OdeAdjointSolver::<f64>::new(1, 1, options(AdjointMode::Discrete, 0.125)).unwrap();
    let reverse = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    let state = solution.state().values();
    assert!((reverse.parameter_gradient()[0] - state[0] * state[1]).abs() < 1e-7);
}

#[test]
fn fixed_mesh_directional_dot_matches_centered_objective_difference() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let objective = Energy(0.3);
    let plan = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.125).with_fixed_step(true)),
    )
    .unwrap();
    let result = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &objective,
            &[1.2],
            (0., 0.5),
        ))
        .unwrap();
    let h = 1e-5;
    let values = [-1., 1.].map(|sign| {
        plan.solve(&scioxide_ode::OdeAdjointProblem::new(
            &Exponential {
                parameter: [0.4 + sign * h * 0.3],
            },
            &objective,
            &[1.2 - sign * h * 0.7],
            (0., 0.5),
        ))
        .unwrap()
        .report()
        .objective()
        .unwrap()
    });
    let difference = (values[1] - values[0]) / (2. * h);
    assert!(
        (difference - (-0.7 * result.initial_gradient()[0] + 0.3 * result.parameter_gradient()[0]))
            .abs()
            < 1e-9
    );
}

#[test]
fn event_running_cost_includes_moving_limit_and_rejects_flat_root() {
    let model = Exponential {
        parameter: [0.7_f64],
    };
    let objective = Energy(0.3);
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let error = ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap();
        let plan = OdeAdjointSolver::<f64>::new(
            1,
            1,
            OdeAdjointOptions::new(
                common(mode, 0.125)
                    .with_continuous_error(error.clone())
                    .with_quadrature_error(error.clone()),
            )
            .with_primal_error(error),
        )
        .unwrap();
        let result = plan
            .solve(
                &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 2.))
                    .with_terminal_event(Some(&Level)),
            )
            .unwrap();
        assert!(
            (result.initial_gradient()[0] + 0.3 / 1.4).abs() < 2e-8,
            "{result:?}"
        );
        assert!(
            (result.parameter_gradient()[0] + 0.9 / (4. * 0.7_f64.powi(2))).abs() < 2e-8,
            "{result:?}"
        );
        struct Flat;
        impl AdjointTerminalEvent<f64> for Flat {
            fn unique_branch(&self) -> bool {
                true
            }
            fn transversality_threshold(&self) -> f64 {
                1e-8
            }
            fn value(&self, _: f64, y: &[f64]) -> Result<f64, SciOxideError> {
                Ok((y[0] - 2.).powi(3))
            }
            fn gradient(
                &self,
                _: f64,
                y: &[f64],
                g: &mut [f64],
                p: &mut [f64],
            ) -> Result<f64, SciOxideError> {
                g[0] = 3. * (y[0] - 2.).powi(2);
                p.fill(0.);
                Ok(0.)
            }
        }
        let failure = plan
            .solve(
                &scioxide_ode::OdeAdjointProblem::new(&model, &objective, &[1.], (0., 2.))
                    .with_terminal_event(Some(&Flat)),
            )
            .unwrap_err();
        assert_eq!(
            failure.progress().unwrap().termination(),
            Some(AdjointTermination::EventDerivativeUnavailable)
        );
    }
}

struct ConjugateFlow;
impl OdeSystem<f64, Complex<f64>> for ConjugateFlow {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[Complex<f64>],
        out: &mut [Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[0].conj();
        Ok(())
    }
}
impl OdeAdjointSystem<Complex<f64>> for ConjugateFlow {
    fn parameters(&self) -> &[f64] {
        &[]
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        _: &[Complex<f64>],
        seed: &[Complex<f64>],
        state: &mut [Complex<f64>],
        _: &mut [f64],
        _: &mut [Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state[0] = seed[0].conj();
        Ok(())
    }
}
#[test]
fn real_linear_complex_pullback_and_empty_parameter_gradient() {
    for mode in [AdjointMode::Discrete, AdjointMode::Continuous] {
        let error = ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap();
        let plan = OdeAdjointSolver::new(
            ConjugateFlow.dimension(),
            ConjugateFlow.parameter_dimension(),
            OdeAdjointOptions::new(common(mode, 0.0625).with_continuous_error(error.clone()))
                .with_primal_error(error),
        )
        .unwrap();
        let result = plan
            .solve(&scioxide_ode::OdeAdjointProblem::new(
                &ConjugateFlow,
                &Energy(0.),
                &[Complex::new(1., 2.)],
                (0., 0.5),
            ))
            .unwrap();
        assert!(result.parameter_gradient().is_empty());
        assert!(
            (result.initial_gradient()[0] - Complex::new(1_f64.exp(), 2. * (-1_f64).exp())).abs()
                < 1e-9
        );
    }
}

struct ChangedParameter(core::cell::Cell<usize>);
impl OdeSystem<f64> for ChangedParameter {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.0.set(1);
        out[0] = self.parameters()[0] * y[0];
        Ok(())
    }
}
impl OdeAdjointSystem<f64> for ChangedParameter {
    fn parameters(&self) -> &[f64] {
        if self.0.get() == 0 { &[0.4] } else { &[0.8] }
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        y: &[f64],
        seed: &[f64],
        g: &mut [f64],
        p: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        g[0] = self.parameters()[0] * seed[0];
        p[0] = y[0] * seed[0];
        Ok(())
    }
}
#[test]
fn parameter_changes_are_rejected_even_when_the_primal_state_is_constant() {
    let model = ChangedParameter(core::cell::Cell::new(0));
    let plan = OdeAdjointSolver::new(
        1,
        1,
        OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.125).with_fixed_step(true)),
    )
    .unwrap();
    let failure = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[0.],
            (0., 0.5),
        ))
        .unwrap_err();
    assert_eq!(
        failure.progress().unwrap().termination(),
        Some(AdjointTermination::ReplayMismatch)
    );
    assert!(matches!(
        failure.error().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ReplayMismatch,
            ..
        }
    ));
}
struct ChangedCost(core::cell::Cell<bool>);
impl AdjointObjective<f64> for ChangedCost {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        y: &[f64],
        _: &[f64],
        g: &mut [f64],
        p: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        p.fill(0.);
        if part == AdjointObjectivePart::Terminal {
            self.0.set(true);
            g[0] = y[0];
            return Ok(AdjointObjectiveValue {
                value: y[0] * y[0] * 0.5,
                time_derivative: 0.,
            });
        }
        let scale = if self.0.get() { 2. } else { 1. };
        g[0] = scale;
        Ok(AdjointObjectiveValue {
            value: scale * y[0],
            time_derivative: 0.,
        })
    }
}
#[test]
fn changed_running_cost_is_part_of_exact_replay_validation() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let cost = ChangedCost(core::cell::Cell::new(false));
    let plan = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(
            common(AdjointMode::Discrete, 0.125)
                .with_fixed_step(true)
                .with_checkpoint_policy(CheckpointPolicy::Explicit(
                    CheckpointSchedule::binomial(128, 1).unwrap(),
                )),
        ),
    )
    .unwrap();
    let failure = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &cost,
            &[1.],
            (0., 0.5),
        ))
        .unwrap_err();
    assert_eq!(
        failure.progress().unwrap().termination(),
        Some(AdjointTermination::ReplayMismatch)
    );
    assert!(matches!(
        failure.error().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ReplayMismatch,
            ..
        }
    ));
}

#[test]
fn event_localization_has_its_own_iteration_scope() {
    let model = Exponential {
        parameter: [0.7_f64],
    };
    let plan = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(common(AdjointMode::Discrete, 1.5).with_fixed_step(true)),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(
            &scioxide_ode::OdeAdjointProblem::new(&model, &TerminalTime, &[1.], (0., 2.))
                .with_terminal_event(Some(&Level)),
        )
        .unwrap();
    let mut epoch = scioxide_core::SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = epoch
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &TerminalTime, &[1.], (0., 2.))
                .with_terminal_event(Some(&Level)),
            &mut [0.],
            &mut [0.],
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(AdjointTermination::TerminalEvent)
    );
    assert!(report.final_time() > 0.0 && report.final_time() < 1.5);
    assert_eq!(report.counts().iterations(), 0);
    assert!(report.accounting().nested().iterations() > 1);
    assert_eq!(context.budget_report(BudgetKind::Iteration).consumed(), 0);
}

struct LimitedFlow {
    limit: f64,
    calls: core::cell::Cell<usize>,
}
impl OdeSystem<f64> for LimitedFlow {
    fn dimension(&self) -> usize {
        1
    }
    fn step_limit(
        &self,
        _: f64,
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<f64>, SciOxideError> {
        Ok(Some(self.limit))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = 0.4 * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, 0.4)?;
        Ok(Some(()))
    }
}
impl OdeAdjointSystem<f64> for LimitedFlow {
    fn parameters(&self) -> &[f64] {
        &[0.4]
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        y: &[f64],
        seed: &[f64],
        g: &mut [f64],
        p: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        g[0] = 0.4 * seed[0];
        p[0] = y[0] * seed[0];
        Ok(())
    }
}
#[test]
fn system_step_limits_are_preserved_in_explicit_and_residual_adapters() {
    let model = LimitedFlow {
        limit: 0.0625,
        calls: core::cell::Cell::new(0),
    };
    let options = OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.5).with_fixed_step(true));
    let plan = OdeAdjointSolver::new(1, 1, options.clone()).unwrap();
    let result = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &Energy(0.),
            &[1.],
            (0., 0.5),
        ))
        .unwrap();
    assert_eq!(result.report().accepted_steps(), 8);
    let residual = OdeAdjointResidual::new(&model).unwrap();
    let implicit = DaeAdjointSolver::backward_euler(
        residual.dimension(),
        residual.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(
            (options).common().clone(),
            (DaeOptions::new()).numerics().clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let result = implicit
        .solve(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.),
            &[1.],
            &[0.4],
            (0., 0.5),
        ))
        .unwrap();
    assert_eq!(result.report().accepted_steps(), 8);
    for limit in [0., -1., f64::NAN, f64::INFINITY] {
        let invalid = LimitedFlow {
            limit,
            calls: core::cell::Cell::new(0),
        };
        assert!(
            plan.solve(&scioxide_ode::OdeAdjointProblem::new(
                &invalid,
                &Energy(0.),
                &[1.],
                (0., 0.5)
            ))
            .is_err()
        );
        assert_eq!(invalid.calls.get(), 0);
    }
}

struct SharedTimeEvent {
    root: f64,
    tangent: bool,
}
impl SharedTimeEvent {
    fn value_at(&self, time: f64) -> f64 {
        let distance = time - self.root;
        if self.tangent {
            distance * distance
        } else {
            distance
        }
    }
    fn slope_at(&self, time: f64) -> f64 {
        if self.tangent {
            2. * (time - self.root)
        } else {
            1.
        }
    }
}
impl AdjointTerminalEvent<f64> for SharedTimeEvent {
    fn unique_branch(&self) -> bool {
        true
    }
    fn transversality_threshold(&self) -> f64 {
        1e-6
    }
    fn value(&self, time: f64, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.value_at(time))
    }
    fn gradient(
        &self,
        time: f64,
        _: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
    ) -> Result<f64, SciOxideError> {
        state.fill(0.);
        parameters.fill(0.);
        Ok(self.slope_at(time))
    }
}
impl EventFunction<f64> for SharedTimeEvent {
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        true
    }
    fn eval(&self, time: f64, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.value_at(time))
    }
    fn has_total_derivative(&self) -> bool {
        true
    }
    fn total_derivative(&self, time: f64, _: &[f64], _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.slope_at(time))
    }
}

#[test]
fn shared_detector_handles_boundary_crossings_and_tangency() {
    let model = Exponential { parameter: [0.] };
    for tangent in [false, true] {
        let root = if tangent {
            0.37
        } else {
            f64::from_bits(0.25_f64.to_bits() + 1)
        };
        let base =
            OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.25).with_fixed_step(true));
        let event_options = base
            .event_options()
            .with_search(EventSearchPolicy::DerivativeAssisted(Budget::ONE));
        let options = base.with_event_options(event_options);
        let event = SharedTimeEvent { root, tangent };
        let events: [&dyn EventFunction<f64>; 1] = [&event];
        let forward = (
            DormandPrince5::new(),
            OdeOptions::new()
                .with_initial_step(0.25)
                .with_maximum_step(0.25)
                .with_event_options(event_options),
        );
        let solution = forward
            .0
            .solve(
                &OdeProblem::new(&model, 1.).with_events(&events),
                &forward.1,
                OdeState::new(0., Vector::from_vec(vec![1.])),
            )
            .unwrap();
        let report = solution.report();
        assert_eq!(report.termination(), Some(OdeTermination::TerminalEvent));
        assert!((report.final_time() - root).abs() < 1e-13);
        if tangent {
            assert_eq!(
                solution.trajectory().events()[0].occurrence(),
                EventOccurrence::Tangency
            );
        }
        let plan = OdeAdjointSolver::new(1, 1, options).unwrap();
        let mut workspace = plan
            .workspace(
                &scioxide_ode::OdeAdjointProblem::new(&model, &TerminalTime, &[1.], (0., 1.))
                    .with_terminal_event(Some(&event)),
            )
            .unwrap();
        let mut gy = [17.];
        let mut gp = [19.];
        let result = plan.solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &TerminalTime, &[1.], (0., 1.))
                .with_terminal_event(Some(&event)),
            &mut gy,
            &mut gp,
            &mut workspace,
            ExecutionContext::sequential(),
        );
        if tangent {
            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
            let progress = failed(&workspace);
            assert_eq!(
                progress.termination(),
                Some(AdjointTermination::EventDerivativeUnavailable)
            );
            assert!((progress.final_time() - report.final_time()).abs() < 1e-13);
            assert_eq!(gy, [17.]);
            assert_eq!(gp, [19.]);
        } else {
            let result = result.unwrap();
            assert_eq!(
                result.termination(),
                Some(AdjointTermination::TerminalEvent)
            );
            assert!((result.final_time() - report.final_time()).abs() < 1e-13);
            assert!(gy[0].abs() < 1e-13);
            assert!(gp[0].abs() < 1e-13);
        }
    }
}

struct BoundaryReplay {
    replay: core::cell::Cell<bool>,
}
impl OdeSystem<f64> for BoundaryReplay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        time: f64,
        _: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        output[0] = if self.replay.get() && time <= 0.5 {
            64. * time * (time - 0.25) * (time - 0.5)
        } else {
            0.
        };
        Ok(())
    }
}
impl OdeAdjointSystem<f64> for BoundaryReplay {
    fn parameters(&self) -> &[f64] {
        &[]
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        state: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state.fill(0.);
        Ok(())
    }
}
impl AdjointObjective<f64> for BoundaryReplay {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        _: &[f64],
        _: &[f64],
        state: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        if part == AdjointObjectivePart::Terminal {
            self.replay.set(true);
        }
        state.fill(0.);
        Ok(AdjointObjectiveValue {
            value: 0.,
            time_derivative: 0.,
        })
    }
}

#[test]
fn nonexact_replay_checks_retained_boundaries_even_when_final_state_matches() {
    let model = BoundaryReplay {
        replay: core::cell::Cell::new(true),
    };
    let forward = (
        DormandPrince5::new(),
        OdeOptions::new()
            .with_initial_step(0.25)
            .with_minimum_step(0.25)
            .with_maximum_step(0.25),
    );
    let changed = forward
        .0
        .solve(
            &OdeProblem::new(&model, 1.),
            &forward.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        )
        .unwrap();
    assert_eq!(
        changed.report().termination(),
        Some(OdeTermination::ReachedEnd)
    );
    assert!((changed.state().values()[0] - 1.).abs() < 1e-14);
    model.replay.set(false);
    let plan = OdeAdjointSolver::new(
        1,
        0,
        OdeAdjointOptions::new(
            common(AdjointMode::Discrete, 0.25)
                .with_fixed_step(true)
                .with_replay_error(
                    ErrorTolerance::scalar(1e-10, 0., NormSelection::Infinity).unwrap(),
                )
                .with_tape_capacity(4)
                .with_checkpoint_policy(CheckpointPolicy::Explicit(
                    CheckpointSchedule::explicit(4, &[0, 1, 2]).unwrap(),
                )),
        ),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &model,
            &[1.],
            (0., 1.),
        ))
        .unwrap();
    let mut gradient = [17.];
    let error = plan
        .solve_into(
            &scioxide_ode::OdeAdjointProblem::new(&model, &model, &[1.], (0., 1.)),
            &mut gradient,
            &mut [],
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ReplayMismatch,
            ..
        }
    ));
    let progress = failed(&workspace);
    assert_eq!(
        progress.termination(),
        Some(AdjointTermination::ReplayMismatch)
    );
    assert!(progress.errors().replay() > 1e-6);
    assert_eq!(gradient, [17.]);
}

trait ContinuousError<S: RealScalar> {
    fn with_continuous_error(self, error: ErrorTolerance<S>) -> Self;
}
impl<S: RealScalar> ContinuousError<S> for AdjointOptions<S> {
    fn with_continuous_error(self, error: ErrorTolerance<S>) -> Self {
        if self.mode() == AdjointMode::Continuous {
            self.with_adjoint_error(error)
        } else {
            self
        }
    }
}

#[test]
fn forward_attempt_and_reverse_limits_are_distinct_from_tape_capacity() {
    let model = Exponential {
        parameter: [0.4_f64],
    };
    let objective = Energy(0.);
    let problem = OdeAdjointProblem::new(&model, &objective, &[1.], (0., 0.5));
    for (configuration, termination, accepted) in [
        (
            OdeAdjointOptions::new(
                common(AdjointMode::Discrete, 0.125)
                    .with_fixed_step(true)
                    .with_max_steps(Budget::ONE),
            ),
            AdjointTermination::BudgetLimit(BudgetKind::Proposal),
            1,
        ),
        (
            OdeAdjointOptions::new(
                common(AdjointMode::Discrete, 0.125)
                    .with_fixed_step(true)
                    .with_max_reverse_steps(0),
            ),
            AdjointTermination::BudgetLimit(BudgetKind::Iteration),
            4,
        ),
    ] {
        let plan = OdeAdjointSolver::<f64>::new(1, 1, configuration).unwrap();
        let mut workspace = plan.workspace(&problem).unwrap();
        let (mut initial, mut parameter) = ([12.], [13.]);
        assert!(
            plan.solve_into(
                &problem,
                &mut initial,
                &mut parameter,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        let report = failed(&workspace);
        assert_eq!(report.termination(), Some(termination));
        assert_eq!(report.accepted_steps(), accepted);
        assert_eq!(report.rejected_steps(), 0);
        assert_eq!(
            report.trust(),
            scioxide_core::PartialResultTrust::Admissible
        );
        assert!(report.last_step_error_estimate().is_some());
        assert_eq!((initial, parameter), ([12.], [13.]));
    }
}

#[test]
fn adjoint_construction_rejects_inapplicable_formulation_controls() {
    let error = ErrorTolerance::scalar(1e-6, 1e-6, NormSelection::Infinity).unwrap();
    let common = AdjointOptions::new(AdjointMode::Discrete, 0.125)
        .unwrap()
        .with_fixed_step(true);
    assert!(
        OdeAdjointSolver::<f64>::new(1, 1, common.clone().with_adjoint_error(error.clone()))
            .is_err()
    );
    assert!(
        OdeAdjointSolver::<f64>::new(
            1,
            1,
            OdeAdjointOptions::new(common.clone()).with_interpolation_defect_sampling(true)
        )
        .is_err()
    );
    let numerics = DaeNumericalOptions::new();
    assert!(
        DaeAdjointSolver::<f64>::backward_euler(
            1,
            1,
            OdeStageBackend::Dense,
            DaeAdjointOptions::new(
                common.clone().with_adjoint_error(error.clone()),
                numerics.clone()
            )
            .unwrap()
        )
        .is_err()
    );
    assert!(
        DaeAdjointSolver::<f64>::trapezoid(
            1,
            1,
            OdeStageBackend::Dense,
            DaeAdjointOptions::new(common.with_quadrature_error(error), numerics).unwrap()
        )
        .is_err()
    );
}

struct ScratchModel {
    model: Exponential<f64>,
    required: usize,
}
impl OdeSystem<f64> for ScratchModel {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.model.rhs(time, state, output, context)
    }
}
impl OdeAdjointSystem<f64> for ScratchModel {
    fn parameters(&self) -> &[f64] {
        &self.model.parameter
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback_workspace_len(&self) -> usize {
        self.required
    }
    fn pullback(
        &self,
        time: f64,
        state: &[f64],
        seed: &[f64],
        state_gradient: &mut [f64],
        parameter_gradient: &mut [f64],
        scratch: &mut [f64],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if scratch.len() < self.required {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.required),
                scioxide_core::Dimensions::one(scratch.len()),
            ));
        }
        self.model.pullback(
            time,
            state,
            seed,
            state_gradient,
            parameter_gradient,
            scratch,
            context,
        )
    }
}

#[test]
fn problem_preparation_admits_callback_scratch_and_reuses_it_with_reused_storage() {
    let small = ScratchModel {
        model: Exponential { parameter: [0.4] },
        required: 0,
    };
    let larger = ScratchModel {
        model: Exponential { parameter: [0.4] },
        required: 32,
    };
    let objective = Energy(0.);
    let small_problem = OdeAdjointProblem::new(&small, &objective, &[1.], (0., 0.5));
    let problem = OdeAdjointProblem::new(&larger, &objective, &[1.], (0., 0.5));
    let plan = OdeAdjointSolver::<f64>::new(
        1,
        1,
        OdeAdjointOptions::new(common(AdjointMode::Discrete, 0.125).with_fixed_step(true)),
    )
    .unwrap();
    let mut workspace = plan.workspace(&small_problem).unwrap();
    let requirements = plan.workspace_requirements(&problem).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let (mut initial, mut parameter) = ([0.], [0.]);
    let _ = plan
        .solve_into(
            &problem,
            &mut initial,
            &mut parameter,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let _ = plan
        .solve_into(
            &problem,
            &mut initial,
            &mut parameter,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!((initial[0] - 0.4_f64.exp()).abs() < 1e-7);
    assert!((parameter[0] - 0.5 * 0.4_f64.exp()).abs() < 1e-7);
}
