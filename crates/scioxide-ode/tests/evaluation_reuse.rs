//! Callback evaluation counts and cache lifecycle across repeated solves.
#![allow(clippy::unwrap_used, clippy::too_many_arguments)]

use core::cell::Cell;
use scioxide_ad::CheckpointSchedule;
use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection,
    SciOxideError, SolveContext,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::*;

struct Decay {
    calls: Cell<usize>,
    parameters: [f64; 1],
}
impl Decay {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            parameters: [-1.0],
        }
    }
}
impl OdeSystem<f64> for Decay {
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(0.);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        state: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = self.parameters[0] * state[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, self.parameters[0])?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

fn tolerance(value: f64) -> ErrorTolerance<f64> {
    ErrorTolerance::scalar(value, value, NormSelection::Infinity).unwrap()
}

/// Integrate `problem` from `state` on `workspace` under sequential native execution.
fn run<M: OdeMethod<f64> + Clone, Sys: OdeSystem<f64> + ?Sized>(
    plan: &(M, OdeOptions<f64>),
    problem: &OdeProblem<'_, f64, Sys>,
    state: &mut OdeState<f64>,
    workspace: &mut OdeWorkspace<f64, M>,
) -> OdeSolveReport<f64> {
    plan.0
        .advance(
            problem,
            &plan.1,
            state,
            &mut (),
            workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap()
}

fn counted_solve<M: OdeMethod<f64> + Clone + core::fmt::Debug>(
    method: M,
    dense: bool,
    force_rejection: bool,
) -> OdeSolveReport<f64> {
    let system = Decay::new();
    let h = if force_rejection { 1.0 } else { 1.0 / 128.0 };
    let end = if force_rejection { 1.0 } else { 0.125 };
    let mut options = OdeOptions::new()
        .with_initial_step(h)
        .with_maximum_step(h)
        .with_error_tolerance(tolerance(if force_rejection { 1e-8 } else { 1e-5 }));
    if dense {
        options = options.with_output_schedule(OdeOutputSchedule::times(vec![h / 2.0, end]));
    }
    let plan = (method.clone(), options);
    let problem = OdeProblem::new(&system, end);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
    let report = run(&plan, &problem, &mut state, &mut workspace);
    assert!(report.is_success(), "{:?}", report.termination());
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        system.calls.get()
    );
    assert!(
        (state.values()[0] - (-end).exp()).abs() < 1e-3,
        "{method:?} end={end} actual={} expected={}",
        state.values()[0],
        (-end).exp()
    );
    if force_rejection {
        assert!(report.rejected_steps() > 0);
    }
    report
}

fn attempts(report: &OdeSolveReport<f64>) -> usize {
    report.accepted_steps() + report.rejected_steps()
}

/// `evaluations` per attempt and `extension` more for the one accepted step whose
/// interpolant the first requested time reads.
fn stages<M: OdeMethod<f64> + Clone + core::fmt::Debug>(
    method: M,
    dense: bool,
    rejection: bool,
    evaluations: usize,
    extension: usize,
) {
    let report = counted_solve(method, dense, rejection);
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        1 + evaluations * attempts(&report) + extension
    );
}

/// `count − 1` later stages per attempt and `f(tₙ, yₙ)` once per accepted base: the
/// endpoint derivative of the one step whose interpolant is read becomes the next base
/// derivative. Returns the RHS count.
fn per_base<M: OdeMethod<f64> + Clone + core::fmt::Debug>(
    method: M,
    dense: bool,
    rejection: bool,
    count: usize,
) -> usize {
    let report = counted_solve(method, dense, rejection);
    let calls = report.counts().provider(EvaluationRole::RightHandSide);
    assert_eq!(
        calls,
        report.accepted_steps() + (count - 1) * attempts(&report)
    );
    calls
}

#[test]
fn explicit_and_rosenbrock_counts_include_only_recomputed_stages() {
    for rejection in [false, true] {
        stages(DormandPrince5::new(), false, rejection, 6, 0);
        stages(DormandPrince853::new(), false, rejection, 12, 0);
        // The three extension stages wait for an accepted step whose interpolant is read.
        stages(DormandPrince853::new(), true, rejection, 12, 3);
        // The Hermite endpoint derivative waits for an accepted step whose interpolant is
        // read, so requested output costs nothing per rejected attempt.
        assert_eq!(
            per_base(Rkf45::new(), true, rejection, 6),
            per_base(Rkf45::new(), false, rejection, 6)
        );
        assert_eq!(
            per_base(Verner65::new(), true, rejection, 8),
            per_base(Verner65::new(), false, rejection, 8)
        );
        for (method, count) in [(RosenbrockMethod::Ros2, 2), (RosenbrockMethod::Ros34Pw2, 4)] {
            per_base(RosenbrockW::new(method).unwrap(), true, rejection, count);
        }
    }
}

#[test]
fn automatic_initial_steps_share_the_initial_derivative() {
    // The heuristic evaluates f(t0, y0) and one trial derivative; the first step
    // reuses f(t0, y0) as its base derivative.
    fn solve<M: OdeMethod<f64> + Clone>(method: M) -> (usize, OdeSolveReport<f64>) {
        let system = Decay::new();
        let options = OdeOptions::new().with_error_tolerance(tolerance(1e-7));
        let plan = (method, options);
        let problem = OdeProblem::new(&system, 2.0);
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
        let report = run(&plan, &problem, &mut state, &mut workspace);
        assert!(report.is_success());
        assert!((state.values()[0] - (-2.0_f64).exp()).abs() < 1e-5);
        assert_eq!(
            report.counts().provider(EvaluationRole::RightHandSide),
            system.calls.get()
        );
        (system.calls.get(), report)
    }
    let (calls, report) = solve(DormandPrince5::new());
    assert_eq!(calls, 2 + 6 * attempts(&report));
    let (calls, report) = solve(DormandPrince853::new());
    assert_eq!(calls, 2 + 12 * attempts(&report));
    // Without FSAL every accepted base is evaluated once; f(t0, y0) is the first.
    let (calls, report) = solve(Rkf45::new());
    assert_eq!(calls, 1 + report.accepted_steps() + 5 * attempts(&report));
    let (calls, report) =
        solve(AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5)).unwrap());
    assert_eq!(calls, 2 + 2 * attempts(&report));
    // Rosenbrock evaluates each accepted base once and its s − 1 later stages.
    let (calls, report) = solve(RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap());
    assert_eq!(calls, 1 + report.accepted_steps() + 3 * attempts(&report));
}

/// Counts right-hand-side calls at the initial point `(0, [1])`.
struct AtStart {
    system: Decay,
    calls: Cell<usize>,
}
impl OdeSystem<f64> for AtStart {
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.time_derivative(time, state, output, cx)
    }
    fn rhs(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if time == 0.0 && state == [1.0] {
            self.calls.set(self.calls.get() + 1);
        }
        self.system.rhs(time, state, output, cx)
    }
    fn jacobian(
        &self,
        time: f64,
        state: &[f64],
        output: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.jacobian(time, state, output, cx)
    }
}

#[test]
fn implicit_methods_take_the_initial_derivative_from_the_step_heuristic() {
    fn calls_at_start<M: OdeMethod<f64> + Clone>(method: M) -> usize {
        let system = AtStart {
            system: Decay::new(),
            calls: Cell::new(0),
        };
        let options = OdeOptions::new().with_error_tolerance(tolerance(1e-7));
        let plan = (method, options);
        let problem = OdeProblem::new(&system, 2.0);
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
        let report = run(&plan, &problem, &mut state, &mut workspace);
        assert!(report.is_success());
        assert!((state.values()[0] - (-2.0_f64).exp()).abs() < 1e-5);
        system.calls.get()
    }
    assert_eq!(calls_at_start(RadauIia::new()), 1);
    assert_eq!(
        calls_at_start(RosenbrockW::new(RosenbrockMethod::Ros2).unwrap()),
        1
    );
    assert_eq!(
        calls_at_start(RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap()),
        1
    );
}

#[test]
fn adams_pece_counts_distinguish_high_order_startup() {
    for order in 1..=5 {
        let report = counted_solve(AdamsMoulton::new(order).unwrap(), true, false);
        let startup_stages = if order <= 4 { 4 } else { 6 };
        assert_eq!(
            report.counts().provider(EvaluationRole::RightHandSide),
            1 + startup_stages * order + 2 * (report.accepted_steps() - order)
        );
        let report = counted_solve(
            AdamsMoulton::with_order_policy(MultistepOrder::Automatic(order)).unwrap(),
            true,
            false,
        );
        assert_eq!(
            report.counts().provider(EvaluationRole::RightHandSide),
            1 + 2 * attempts(&report)
        );
    }
    stages(
        AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        true,
        true,
        2,
        0,
    );
}

/// A rejected trial reuses only the accepted initial derivative (`warm` evaluations),
/// later accepted bases cost `rebase` evaluations (zero where the endpoint derivative
/// is retained), and a new epoch starts `cold` even at the cached endpoint.
fn invalidation<M: OdeMethod<f64> + Clone>(method: M, cold: usize, warm: usize, rebase: usize) {
    let system = Decay::new();
    // Requested output selects the dense-output stages; a requested time at a step
    // end reads the endpoint state, so no deferred extension is built.
    let options = OdeOptions::new()
        .with_initial_step(1.0)
        .with_maximum_step(1.0)
        .with_error_tolerance(tolerance(1e-8))
        .with_output_schedule(OdeOutputSchedule::times(vec![1.0]));
    let plan = (method, options);
    let forward = OdeProblem::new(&system, 1.0);
    let mut workspace = plan.0.workspace(&forward, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
    let report = run(&plan, &forward, &mut state, &mut workspace);
    assert!(report.is_success());
    assert!(report.rejected_steps() > 0);
    let forward_calls = system.calls.get();
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        forward_calls
    );
    assert_eq!(
        forward_calls,
        cold + warm * (attempts(&report) - 1) + rebase * (report.accepted_steps() - 1)
    );
    // A fresh epoch from the last accepted endpoint recomputes its initial derivative.
    let backward = OdeProblem::new(&system, 0.0);
    let report = run(&plan, &backward, &mut state, &mut workspace);
    assert!(report.is_success());
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        system.calls.get() - forward_calls
    );
    assert_eq!(
        system.calls.get() - forward_calls,
        cold + warm * (attempts(&report) - 1) + rebase * (report.accepted_steps() - 1)
    );
    assert!((state.values()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn rejected_endpoints_are_never_reused_and_cache_keys_are_checked() {
    invalidation(DormandPrince5::new(), 7, 6, 0);
    invalidation(DormandPrince853::new(), 13, 12, 0);
    // The requested span end reads no Hermite interpolant, so each later accepted
    // base evaluates its own derivative.
    invalidation(Rkf45::new(), 6, 5, 1);
    invalidation(Verner65::new(), 8, 7, 1);
    invalidation(RosenbrockW::new(RosenbrockMethod::Ros2).unwrap(), 2, 1, 1);
}

struct FailedStage {
    system: Decay,
    armed: Cell<bool>,
    /// Calls made up to and including the failed trial evaluation.
    at_failure: Cell<usize>,
}
impl OdeSystem<f64> for FailedStage {
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.time_derivative(time, state, output, cx)
    }
    fn rhs(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.system.rhs(time, state, output, cx)?;
        if time > 0.0 && self.armed.replace(false) {
            self.at_failure.set(self.system.calls.get());
            output[0] = f64::NAN;
        }
        Ok(())
    }
    fn jacobian(
        &self,
        time: f64,
        state: &[f64],
        output: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.system.jacobian(time, state, output, cx)
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

fn retry_failed_stage<M: OdeMethod<f64> + Clone>(method: M, expected: usize) {
    let system = FailedStage {
        system: Decay::new(),
        armed: Cell::new(true),
        at_failure: Cell::new(0),
    };
    // The nonfinite trial is rejected and retried at half the step, which the loose
    // tolerance accepts; the step budget stops integration after that retry.
    let options = OdeOptions::new()
        .with_error_tolerance(tolerance(1e-2))
        .with_initial_step(0.125)
        .with_output_schedule(OdeOutputSchedule::times(vec![0.0625]))
        .with_max_steps(Budget::try_from(2).unwrap());
    let plan = (method, options);
    let problem = OdeProblem::new(&system, 0.125);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
    let report = run(&plan, &problem, &mut state, &mut workspace);
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!((report.rejected_steps(), report.accepted_steps()), (1, 1));
    assert_eq!(state.time(), 0.0625);
    // The retry matches a clean step of the same size from the same initial state.
    let clean = Decay::new();
    let clean_plan = (plan.0.clone(), plan.1.clone().with_initial_step(0.0625));
    let clean_problem = OdeProblem::new(&clean, 0.0625);
    let mut clean_state = OdeState::from_slice(0.0, &[1.0]).unwrap();
    let mut clean_workspace = clean_plan
        .0
        .workspace(&clean_problem, &clean_plan.1)
        .unwrap();
    assert!(
        run(
            &clean_plan,
            &clean_problem,
            &mut clean_state,
            &mut clean_workspace
        )
        .is_success()
    );
    assert_eq!(state.values(), clean_state.values());
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        system.system.calls.get()
    );
    assert_eq!(
        system.system.calls.get() - system.at_failure.get(),
        expected
    );
}

#[test]
fn failed_stages_preserve_only_the_valid_initial_derivative() {
    retry_failed_stage(DormandPrince5::new(), 6);
    retry_failed_stage(DormandPrince853::new(), 12);
    retry_failed_stage(Rkf45::new(), 5);
    retry_failed_stage(Verner65::new(), 7);
    retry_failed_stage(RosenbrockW::new(RosenbrockMethod::Ros2).unwrap(), 1);
}

impl OdeAdjointSystem<f64> for Decay {
    fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        state: &[f64],
        seed: &[f64],
        state_gradient: &mut [f64],
        parameter_gradient: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state_gradient[0] = self.parameters[0] * seed[0];
        parameter_gradient[0] = state[0] * seed[0];
        Ok(())
    }
}
struct Cost {
    calls: Cell<usize>,
    schedule_calls: Cell<usize>,
}
impl AdjointObjective<f64> for Cost {
    fn observation_times(&self) -> &[f64] {
        self.schedule_calls.set(self.schedule_calls.get() + 1);
        &[]
    }
    fn evaluate(
        &self,
        _: AdjointObjectivePart,
        _: f64,
        state: &[f64],
        _: &[f64],
        state_gradient: &mut [f64],
        parameter_gradient: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        state_gradient[0] = state[0];
        parameter_gradient.fill(0.0);
        Ok(AdjointObjectiveValue {
            value: 0.5 * state[0] * state[0],
            time_derivative: 0.0,
        })
    }
}

fn adjoint(
    mode: AdjointMode,
    diagnostics: bool,
    checkpoint: CheckpointPolicy,
) -> (AdjointResult<f64>, usize) {
    let model = Decay::new();
    let cost = Cost {
        calls: Cell::new(0),
        schedule_calls: Cell::new(0),
    };
    let options = scioxide_ode::OdeAdjointOptions::new(
        AdjointOptions::new(mode, 0.03125)
            .unwrap()
            .with_fixed_step(true)
            .with_checkpoint_policy(checkpoint)
            .with_tape_capacity(16)
            .with_max_replay_steps(1000)
            .with_max_reverse_steps(1000)
            .with_max_workspace_bytes(1024 * 1024),
    )
    .with_interpolation_defect_sampling(diagnostics);
    let plan =
        OdeAdjointSolver::new(model.dimension(), model.parameter_dimension(), options).unwrap();
    let result = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &cost,
            &[1.0],
            (0.0, 0.125),
        ))
        .unwrap();
    let report = result.report();
    assert!(report.is_success());
    let total = report.accounting().total().unwrap();
    assert_eq!(
        total.provider(EvaluationRole::RightHandSide),
        model.calls.get()
    );
    assert_eq!(total.provider(EvaluationRole::Objective), cost.calls.get());
    assert_eq!(cost.schedule_calls.get(), 3); // preparation, entry, publication
    (result, cost.calls.get())
}

fn rhs_evaluations(result: &AdjointResult<f64>) -> usize {
    result
        .report()
        .accounting()
        .total()
        .unwrap()
        .provider(EvaluationRole::RightHandSide)
}

#[test]
fn discrete_adjoint_reuses_forward_rhs_and_recorded_cost_gradients() {
    let (full, costs) = adjoint(AdjointMode::Discrete, false, CheckpointPolicy::StoreAll);
    let forward = rhs_evaluations(&full);
    assert_eq!(forward, 1 + 6 * full.report().accepted_steps());
    assert_eq!(costs, forward + 1); // terminal cost only during reversal
    let (replay, _) = adjoint(
        AdjointMode::Discrete,
        false,
        CheckpointPolicy::Explicit(CheckpointSchedule::binomial(16, 2).unwrap()),
    );
    assert!(replay.report().replayed_steps() > 0);
    assert!((full.initial_gradient()[0] - replay.initial_gradient()[0]).abs() < 1e-13);
    assert!((full.parameter_gradient()[0] - replay.parameter_gradient()[0]).abs() < 1e-13);
    let expected = 0.5 + 0.5 * (-0.25_f64).exp();
    assert!((full.initial_gradient()[0] - expected).abs() < 1e-9);
}

#[test]
fn continuous_interpolation_diagnostics_are_opt_in() {
    let (plain, _) = adjoint(AdjointMode::Continuous, false, CheckpointPolicy::StoreAll);
    let (sampled, _) = adjoint(AdjointMode::Continuous, true, CheckpointPolicy::StoreAll);
    assert!(plain.report().errors().interpolation_defect().is_none());
    assert!(sampled.report().errors().interpolation_defect().is_some());
    assert!(rhs_evaluations(&sampled) > rhs_evaluations(&plain));
    assert!((plain.initial_gradient()[0] - sampled.initial_gradient()[0]).abs() < 1e-14);
    assert!((plain.parameter_gradient()[0] - sampled.parameter_gradient()[0]).abs() < 1e-14);
}
