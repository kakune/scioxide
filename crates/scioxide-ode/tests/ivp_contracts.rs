//! Contracts of `OdeMethod::advance`: allocation-free prepared execution,
//! preflight atomicity, failure progress, budget stops, epochs and child scopes.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, ErrorTolerance, EvaluationCounts, EvaluationRole,
    ExecutionContext, NormSelection, PartialResultTrust, ProblemToken, SciOxideError,
    SciOxideErrorKind, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    DormandPrince5, EventDirection, EventFunction, EventSearchPolicy, OdeEvent, OdeEventOptions,
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeOutputSink, OdeProblem, OdeState, OdeSystem,
    OdeTermination, OdeWorkspace,
};

#[test]
fn controller_options_are_immutable_and_each_drive_restarts_history() {
    use scioxide_ode::{OdeStepControl, ProportionalIntegralController};
    for control in [
        OdeStepControl::MethodDefault,
        OdeStepControl::ProportionalIntegral(
            ProportionalIntegralController::new(0.9, 5., 0.2, 1., 0.).unwrap(),
        ),
        OdeStepControl::ProportionalIntegral(ProportionalIntegralController::default()),
        OdeStepControl::Radau,
    ] {
        let options = OdeOptions::new()
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-9, 1e-9, NormSelection::Infinity).unwrap(),
            )
            .with_initial_step(1.)
            .with_step_control(control.clone());
        let plan = (DormandPrince5::new(), options.clone());
        let problem = OdeProblem::new(&Oscillator, 1.);
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let mut previous = None;
        for _ in 0..2 {
            let mut state = OdeState::from_slice(0., &[1., 0.]).unwrap();
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
            assert!(report.rejected_steps() > 0);
            assert!((state.values()[0] - 1_f64.cos()).abs() < 1e-8);
            assert!((state.values()[1] + 1_f64.sin()).abs() < 1e-8);
            let observed = (
                state.values().to_vec(),
                report.accepted_steps(),
                report.rejected_steps(),
            );
            if let Some(previous) = &previous {
                assert_eq!(&observed, previous);
            }
            previous = Some(observed);
            assert_eq!(&plan.1, &options);
            assert_eq!(plan.1.step_control(), &control);
        }
    }
}

/// `y0' = y1, y1' = -y0`, so `y = (cos t, -sin t)` from `(1, 0)`.
struct Oscillator;

impl OdeSystem<f64> for Oscillator {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[1];
        out[1] = -y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        jacobian.set(0, 0, 0.0)?;
        jacobian.set(0, 1, 1.0)?;
        jacobian.set(1, 0, -1.0)?;
        jacobian.set(1, 1, 0.0)?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

/// Counts every sample and event and keeps the last sample; never allocates.
#[derive(Default)]
struct Recorder {
    samples: usize,
    requested: usize,
    events: usize,
    terminal: usize,
    event_times: [f64; 4],
    last_time: f64,
    last_state: [f64; 2],
}

impl OdeOutputSink<f64> for Recorder {
    fn emit(&mut self, time: f64, state: &[f64]) -> Result<(), SciOxideError> {
        self.samples += 1;
        self.last_time = time;
        self.last_state[..state.len()].copy_from_slice(state);
        Ok(())
    }
    fn emit_requested(
        &mut self,
        index: usize,
        time: f64,
        state: &[f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(index, self.requested);
        self.requested += 1;
        self.emit(time, state)
    }
    fn event(&mut self, event: OdeEvent<'_, f64>) -> Result<(), SciOxideError> {
        if let Some(slot) = self.event_times.get_mut(self.events) {
            *slot = event.time();
        }
        self.events += 1;
        self.terminal += usize::from(event.is_terminal());
        Ok(())
    }
}

/// Counts RHS and event calls of a unit-rate system.
struct Counted<'a> {
    calls: &'a core::cell::Cell<usize>,
}

impl OdeSystem<f64> for Counted<'_> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = 1.0;
        Ok(())
    }
}

struct CountedEvent<'a> {
    calls: &'a core::cell::Cell<usize>,
}

impl EventFunction<f64> for CountedEvent<'_> {
    fn eval(&self, _: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        Ok(y[0] - 0.5)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

#[test]
fn preflight_rejection_leaves_the_state_and_runs_no_callback() {
    let rhs_calls = core::cell::Cell::new(0);
    let event_calls = core::cell::Cell::new(0);
    let system = Counted { calls: &rhs_calls };
    let event = CountedEvent {
        calls: &event_calls,
    };
    let events: [&dyn EventFunction<f64>; 1] = [&event];
    let plan = (DormandPrince5::new(), OdeOptions::new());
    let mut workspace = OdeWorkspace::new();
    let cases: [(f64, f64, Vec<f64>, SciOxideErrorKind); 4] = [
        (0.0, 1.0, vec![f64::NAN], SciOxideErrorKind::NonFinite),
        (
            0.0,
            1.0,
            vec![0.0, 1.0],
            SciOxideErrorKind::DimensionMismatch,
        ),
        (1.0, 1.0, vec![0.0], SciOxideErrorKind::InvalidInput),
        (f64::INFINITY, 1.0, vec![0.0], SciOxideErrorKind::NonFinite),
    ];
    for (start, end, values, kind) in cases {
        let problem = OdeProblem::new(&system, end).with_events(&events);
        let mut state = OdeState::new(start, Vector::from_vec(values.clone()));
        let error = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(state.time().to_bits(), start.to_bits());
        assert!(
            state
                .values()
                .iter()
                .zip(&values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert!(attempt.epoch().is_none());
        assert!(attempt.progress().is_none());
        assert_eq!(attempt.counts().own(), EvaluationCounts::new());
        assert_eq!((rhs_calls.get(), event_calls.get()), (0, 0));
    }
    // An unsupported feature is a preflight rejection too: derivative-assisted search
    // needs every event's total derivative.
    let assisted = (
        DormandPrince5::new(),
        OdeOptions::new().with_event_options(OdeEventOptions::new().with_search(
            EventSearchPolicy::DerivativeAssisted(Budget::try_from(4).unwrap()),
        )),
    );
    let problem = OdeProblem::new(&system, 1.0).with_events(&events);
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
    let error = assisted
        .0
        .advance(
            &problem,
            &assisted.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!((state.time(), state.values()), (0.0, &[0.0][..]));
    assert_eq!((rhs_calls.get(), event_calls.get()), (0, 0));
}

/// Unit-rate growth in every component whose RHS fails beyond `limit`.
struct FailsAfter {
    limit: f64,
    dimension: usize,
}

impl OdeSystem<f64> for FailsAfter {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn rhs(
        &self,
        t: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if t > self.limit {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("t"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "injected right-hand-side failure",
            ));
        }
        out.fill(1.0);
        Ok(())
    }
}

fn failing_plan() -> (DormandPrince5<f64>, OdeOptions<f64>) {
    (
        DormandPrince5::new(),
        OdeOptions::new()
            .with_initial_step(0.05)
            .with_maximum_step(0.1)
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
    )
}

const FAILING: FailsAfter = FailsAfter {
    limit: 0.5,
    dimension: 2,
};

#[test]
fn mid_solve_failure_keeps_the_last_accepted_state_and_records_its_progress() {
    let plan = failing_plan();
    let problem = OdeProblem::new(&FAILING, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0, 0.0]));
    let mut sink = Recorder::default();
    let error = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    // The last emitted accepted endpoint is the caller's state.
    assert!(sink.samples > 2);
    assert!(state.time() > 0.0 && state.time() <= 0.5);
    assert_eq!(state.time(), sink.last_time);
    assert_eq!(state.values(), &sink.last_state);
    assert!((state.values()[0] - state.time()).abs() < 1e-12);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert!(attempt.epoch().is_some());
    let progress = attempt.progress().unwrap();
    assert_eq!(progress.time(), state.time());
    assert_eq!(progress.state(), state.values());
    assert_eq!(progress.accepted_steps(), sink.samples - 1);
    assert_eq!(progress.trust(), PartialResultTrust::Admissible);
}

#[test]
fn owned_solve_failure_carries_the_last_accepted_progress() {
    let plan = failing_plan();
    let problem = OdeProblem::new(&FAILING, 1.0);
    let mut reference = OdeState::new(0.0, Vector::from_vec(vec![0.0, 0.0]));
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    assert!(
        plan.0
            .advance(
                &problem,
                &plan.1,
                &mut reference,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    let failure = plan
        .0
        .solve(
            &problem,
            &plan.1,
            OdeState::new(0.0, Vector::from_vec(vec![0.0, 0.0])),
        )
        .unwrap_err();
    assert_eq!(failure.error().kind(), SciOxideErrorKind::InvalidInput);
    let progress = failure.progress().unwrap();
    assert!(progress.time() > 0.0);
    assert_eq!(progress.time(), reference.time());
    assert_eq!(progress.state(), reference.values());
    assert_eq!(progress.trust(), PartialResultTrust::Admissible);
    let resumed = progress.clone().into_state();
    assert_eq!(resumed, reference);
}

#[test]
fn budget_stops_succeed_with_stopped_attempts_and_epochs_increase() {
    let bounded = (
        DormandPrince5::new(),
        OdeOptions::new().with_max_steps(Budget::try_from(3).unwrap()),
    );
    let unbounded = (DormandPrince5::new(), OdeOptions::new());
    let problem = OdeProblem::new(&Oscillator, 5.0);
    let mut workspace = bounded.0.workspace(&problem, &bounded.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0, 0.0]));
    let mut epochs = Vec::new();
    let mut record = |workspace: &OdeWorkspace<f64, DormandPrince5<f64>>| {
        epochs.push(workspace.last_attempt().unwrap().epoch().unwrap().get());
    };

    let report = bounded
        .0
        .advance(
            &problem,
            &bounded.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert!(!report.is_success());
    assert_eq!(report.counts().proposals(), 3);
    assert_eq!(report.final_time(), state.time());
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.progress().unwrap().time(), state.time());
    assert_eq!(attempt.progress().unwrap().state(), state.values());
    record(&workspace);

    // Continuing from the stopped state is a fresh epoch on the same workspace.
    let report = unbounded
        .0
        .advance(
            &problem,
            &unbounded.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    assert!((state.values()[0] - 5.0_f64.cos()).abs() < 1e-2);
    record(&workspace);

    // A rejected preflight does not consume an epoch.
    assert!(
        unbounded
            .0
            .advance(
                &problem,
                &unbounded.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(workspace.last_attempt().unwrap().epoch().is_none());

    for _ in 0..2 {
        state.reset(0.0, &[1.0, 0.0]).unwrap();
        let report = bounded
            .0
            .advance(
                &problem,
                &bounded.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.counts().proposals(), 3);
        record(&workspace);
    }
    let failing = failing_plan();
    state.reset(0.0, &[0.0, 0.0]).unwrap();
    assert!(
        failing
            .0
            .advance(
                &OdeProblem::new(&FAILING, 1.0),
                &failing.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    record(&workspace);
    assert!(
        epochs.windows(2).all(|pair| pair[0] < pair[1]),
        "{epochs:?}"
    );
}

fn rhs_count(counts: EvaluationCounts) -> usize {
    counts.provider(EvaluationRole::RightHandSide)
}

#[test]
fn child_epoch_advances_charge_the_parent() {
    let plan = (DormandPrince5::new(), OdeOptions::new());
    let problem = OdeProblem::new(&Oscillator, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0, 0.0]));
    let mut parent_state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(2301),
        scioxide_core::algorithm_id!("test", "parent", 1),
        2,
        0,
    );
    let mut parent = parent_state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let first = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert!(first.is_success());
    assert!(rhs_count(first.counts()) > 0);
    assert_eq!(rhs_count(parent.counts().own()), 0);
    assert_eq!(
        rhs_count(parent.counts().nested()),
        rhs_count(first.counts())
    );
    let first_epoch = workspace.last_attempt().unwrap().epoch().unwrap();

    state.reset(0.0, &[1.0, 0.0]).unwrap();
    let second = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        rhs_count(parent.counts().nested()),
        rhs_count(first.counts()) + rhs_count(second.counts())
    );
    assert_eq!(
        rhs_count(parent.counts().total().unwrap()),
        rhs_count(first.counts()) + rhs_count(second.counts())
    );
    assert!(workspace.last_attempt().unwrap().epoch().unwrap() > first_epoch);

    // A shared parent allowance bounds every child epoch.
    let mut limited_state = SolveState::new();
    let mut limited = SolveBudgets::unbounded().with_limit(
        BudgetKind::Evaluation,
        Budget::try_from(rhs_count(first.counts()) / 2).unwrap(),
    );
    let mut parent = limited_state
        .begin_solve(ExecutionContext::sequential(), identity, &mut limited)
        .unwrap();
    state.reset(0.0, &[1.0, 0.0]).unwrap();
    let stopped = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        stopped.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert!(rhs_count(parent.counts().nested()) <= rhs_count(first.counts()) / 2);
}

/// Metadata is invariant throughout an epoch; queries after the first RHS expose
/// accidental per-step preflight work.
struct PreparedMetadata {
    mass: bool,
    started: core::cell::Cell<bool>,
    late_queries: core::cell::Cell<usize>,
}
impl PreparedMetadata {
    fn query(&self) {
        if self.started.get() {
            self.late_queries.set(self.late_queries.get() + 1);
        }
    }
}
impl OdeSystem<f64> for PreparedMetadata {
    fn dimension(&self) -> usize {
        self.query();
        1
    }
    fn mass_matrix_revision(&self) -> Option<u64> {
        self.query();
        self.mass.then_some(0)
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
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.started.set(true);
        output[0] = -state[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        output: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.set(0, 0, -1.0)?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn explicit_and_rosenbrock_steps_do_not_repeat_metadata_queries() {
    use scioxide_ode::{
        BuiltinOdeMethod, ClassicalRk4, DormandPrince853, Rkf45, RosenbrockMethod, RosenbrockW,
    };
    let methods: [BuiltinOdeMethod<f64>; 5] = [
        ClassicalRk4::new().into(),
        Rkf45::new().into(),
        DormandPrince5::new().into(),
        DormandPrince853::new().into(),
        RosenbrockW::new(RosenbrockMethod::Ros2).unwrap().into(),
    ];
    for method in methods {
        let system = PreparedMetadata {
            mass: false,
            started: core::cell::Cell::new(false),
            late_queries: core::cell::Cell::new(0),
        };
        let plan = (
            method,
            OdeOptions::new()
                .with_initial_step(0.01)
                .with_maximum_step(0.01),
        );
        let problem = OdeProblem::new(&system, 0.1);
        let result = plan
            .0
            .solve(
                &problem,
                &plan.1,
                OdeState::from_slice(0.0, &[1.0]).unwrap(),
            )
            .unwrap();
        assert!(result.report().is_success());
        assert!(result.report().accepted_steps() >= 10);
        assert!((result.state().values()[0] - (-0.1_f64).exp()).abs() < 1e-3);
        assert_eq!(system.late_queries.get(), 0);
    }
}

#[test]
fn removed_step_feature_checks_reject_at_preflight() {
    use scioxide_ode::{BuiltinOdeMethod, ForwardEuler, RosenbrockMethod, RosenbrockW};

    for absolute in [vec![], vec![1e-6, 1e-6]] {
        let options = OdeOptions::new().with_error_tolerance(
            ErrorTolerance::try_vector(&absolute, 1e-6, NormSelection::Infinity).unwrap(),
        );
        let error = (DormandPrince5::<f64>::new())
            .validate(&(options), 1)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    }
    let error = (ForwardEuler::<f64>::new())
        .validate(
            &(OdeOptions::new().with_output_schedule(OdeOutputSchedule::times(vec![0.5]))),
            1,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);

    let methods: [BuiltinOdeMethod<f64>; 2] = [
        DormandPrince5::new().into(),
        RosenbrockW::new(RosenbrockMethod::Ros2).unwrap().into(),
    ];
    for method in methods {
        let system = PreparedMetadata {
            mass: true,
            started: core::cell::Cell::new(false),
            late_queries: core::cell::Cell::new(0),
        };
        let plan = (method, OdeOptions::new());
        let problem = OdeProblem::new(&system, 1.0);
        let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
        let mut workspace = OdeWorkspace::new();
        let error = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        assert!(!system.started.get());
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.values(), &[1.0]);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
}

type Stopped = (
    Result<Option<OdeTermination>, SciOxideError>,
    Option<AttemptOutcome>,
);

/// Integrate to `t = 1` under an evaluation budget and return the termination and the
/// recorded attempt outcome.
fn budget_stop<M, Sys>(method: M, system: &Sys, initial: &[f64], budget: usize) -> Stopped
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64> + ?Sized,
{
    let options = OdeOptions::new()
        .with_initial_step(0.01)
        .with_max_evaluations(Budget::try_from(budget).unwrap());
    let plan = (method, options);
    let problem = OdeProblem::new(system, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0.0, initial).unwrap();
    let result = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .map(|report| report.termination());
    let outcome = workspace.last_attempt().map(|attempt| attempt.outcome());
    (result, outcome)
}

fn assert_evaluation_stops(label: &str, run: impl Fn(usize) -> Stopped) {
    for budget in 1..=24 {
        let (result, outcome) = run(budget);
        assert_eq!(
            result.map_err(|error| error.kind()),
            Ok(Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))),
            "{label}, budget {budget}"
        );
        assert_eq!(
            outcome,
            Some(AttemptOutcome::Stopped),
            "{label}, budget {budget}"
        );
    }
}

struct Growth;
impl scioxide_ode::OdeSensitivitySystem<f64> for Growth {
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
fn every_evaluation_budget_stops_a_sensitivity_integration() {
    assert_evaluation_stops("DP5 forward sensitivity", |budget| {
        let mut workspace = scioxide_ode::ForwardSensitivityWorkspace::new();
        let direction = [1.0];
        let directions = scioxide_linalg::MatrixView::from_slice(1, 1, &direction).unwrap();
        let system = scioxide_ode::ForwardSensitivitySystem::bind(
            &Growth,
            &[0.5],
            directions,
            &mut workspace,
        )
        .unwrap();
        budget_stop(DormandPrince5::new(), &system, &[1.0, 0.0], budget)
    });
}

#[test]
fn every_evaluation_budget_stops_matrix_free_stages() {
    let system = scioxide_ode::OdeFn::new(2, |_: f64, y: &[f64], f: &mut [f64]| {
        f[0] = -y[0] + 0.1 * y[1] * y[1];
        f[1] = -2.0 * y[1];
        Ok(())
    })
    .unwrap();
    let krylov = scioxide_ode::OdeStageBackend::Krylov {
        iterations: Budget::try_from(8).unwrap(),
        restart: Budget::try_from(8).unwrap(),
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    assert_evaluation_stops("backward Euler with Krylov stages", |budget| {
        let method = scioxide_ode::BackwardEuler::<f64>::new().with_backend(krylov.clone());
        budget_stop(method, &system, &[1.0, 1.0], budget)
    });
    // Radau IIA applies the real Jacobian to the real and imaginary parts of its
    // complex stage and admits the second product separately.
    assert_evaluation_stops("Radau IIA complex stage with Krylov", |budget| {
        let method = scioxide_ode::RadauIia::<f64>::new().with_backend(krylov.clone());
        budget_stop(method, &system, &[1.0, 1.0], budget)
    });
}

#[test]
fn options_are_validated_per_call_and_reuse_one_workspace() {
    let system = scioxide_ode::OdeFn::new(1, |_: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -y[0];
        Ok(())
    })
    .unwrap();
    let problem = OdeProblem::new(&system, 1.0);
    let method = DormandPrince5::new();
    let options = OdeOptions::default();
    let mut workspace = method.workspace(&problem, &options).unwrap();
    let mut state = OdeState::from_slice(0.0, &[1.0]).unwrap();
    let initial = state.clone();
    assert!(
        method
            .advance(
                &problem,
                &options.clone().with_initial_step(0.0),
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(state, initial);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    for options in [
        options.clone().with_initial_step(0.01),
        options.with_initial_step(0.05),
    ] {
        state.reset(0.0, &[1.0]).unwrap();
        assert!(
            method
                .advance(
                    &problem,
                    &options,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential_native()
                )
                .unwrap()
                .is_success()
        );
        assert!((state.values()[0] - (-1.0_f64).exp()).abs() < 1e-3);
    }
}
