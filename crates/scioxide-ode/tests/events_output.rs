//! Analytic event and exact requested-output coverage.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, EvaluationRole, ExecutionContext, NumericScalar,
    RealScalar, ResidualTolerance, SciOxideError, StepTolerance,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, EventDirection, EventEvidence, EventFunction, EventInfo, EventOccurrence,
    EventSearchPolicy, OdeEvent, OdeEventOptions, OdeMethod, OdeOptions, OdeOutputSchedule,
    OdeOutputSink, OdeProblem, OdeSolveReport, OdeState, OdeSystem, OdeTermination, OdeTrajectory,
    OdeWorkspace, RadauIia,
};

fn s<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}
struct Unit;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Unit {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        _: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = S::ONE;
        Ok(())
    }
}
struct Polynomial<S> {
    a: S,
    b: S,
    offset: S,
    terminal: bool,
    direction: EventDirection,
}
impl<S: RealScalar + NumericScalar<Real = S>> EventFunction<S> for Polynomial<S> {
    fn eval(&self, _: S, y: &[S]) -> Result<S, SciOxideError> {
        Ok((y[0] - self.a) * (y[0] - self.b) + self.offset)
    }
    fn direction(&self) -> EventDirection {
        self.direction
    }
    fn is_terminal(&self) -> bool {
        self.terminal
    }
    fn has_total_derivative(&self) -> bool {
        true
    }
    fn total_derivative(&self, _: S, y: &[S], slope: &[S]) -> Result<S, SciOxideError> {
        Ok(((y[0] - self.a) + (y[0] - self.b)) * slope[0])
    }
}
fn polynomial<S: RealScalar>(a: f64, b: f64, offset: f64) -> Polynomial<S> {
    Polynomial {
        a: s(a),
        b: s(b),
        offset: s(offset),
        terminal: false,
        direction: EventDirection::Either,
    }
}
/// Unit-step options: the first step spans the whole unit interval.
fn steps<S: RealScalar + NumericScalar<Real = S>>() -> OdeOptions<S> {
    OdeOptions::new()
        .with_initial_step(S::ONE)
        .with_maximum_step(S::ONE)
}
fn dp5<S: RealScalar + NumericScalar<Real = S>>(
    options: OdeOptions<S>,
) -> (DormandPrince5<S>, OdeOptions<S>) {
    (DormandPrince5::new(), options)
}
fn options<S: RealScalar>(policy: EventSearchPolicy) -> OdeEventOptions<S> {
    OdeEventOptions::new()
        .with_search(policy)
        .with_localization(
            StepTolerance::new(s(2e-6), S::ZERO).unwrap(),
            Budget::try_from(64).unwrap(),
        )
        .with_contact_tolerance(ResidualTolerance::new(s(1e-8), S::ZERO).unwrap(), S::ONE)
}
/// Advance `y' = 1, y(t0) = t0` over `span`; the state holds the last accepted point.
fn run<S, M, K>(
    plan: &(M, OdeOptions<S>),
    events: &[&dyn EventFunction<S>],
    span: (S, S),
    sink: &mut K,
    workspace: &mut OdeWorkspace<S, M>,
) -> (Result<OdeSolveReport<S>, SciOxideError>, OdeState<S>)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
    K: OdeOutputSink<S>,
{
    let problem = OdeProblem::new(&Unit, span.1).with_events(events);
    let mut state = OdeState::new(span.0, Vector::from_vec(vec![span.0]));
    let result = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        sink,
        workspace,
        ExecutionContext::sequential(),
    );
    (result, state)
}
/// Progress time recorded by the last failed or stopped attempt.
fn progress_time<S, M>(workspace: &OdeWorkspace<S, M>) -> S
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S>,
{
    workspace.last_attempt().unwrap().progress().unwrap().time()
}
fn search_cases<S: RealScalar + NumericScalar<Real = S>>() {
    for backwards in [false, true] {
        let span = if backwards {
            (S::ONE, S::ZERO)
        } else {
            (S::ZERO, S::ONE)
        };
        for policy in [
            EventSearchPolicy::Subdivide(Budget::try_from(8).unwrap()),
            EventSearchPolicy::DerivativeAssisted(Budget::ONE),
        ] {
            let plan = dp5::<S>(steps().with_event_options(options(policy)));
            let event = polynomial::<S>(0.23, 0.71, 0.0);
            let mut trajectory = OdeTrajectory::new();
            let (report, _) = run(
                &plan,
                &[&event],
                span,
                &mut trajectory,
                &mut OdeWorkspace::new(),
            );
            assert_eq!(report.unwrap().events(), 2);
            let events = trajectory.events();
            assert_eq!(events.len(), 2);
            let expected = if backwards {
                [0.71, 0.23]
            } else {
                [0.23, 0.71]
            };
            for (event, time) in events.iter().zip(expected) {
                assert!((event.time() - s(time)).abs() < s(4e-6));
                assert!(matches!(event.evidence(), EventEvidence::Crossing { .. }));
            }
            assert_eq!(events[0].occurrence(), EventOccurrence::Falling);
            assert_eq!(events[1].occurrence(), EventOccurrence::Rising);
        }
        for (offset, expected) in [(0.0, 1), (1e-4, 0)] {
            let plan = dp5::<S>(steps().with_event_options(options(
                EventSearchPolicy::DerivativeAssisted(Budget::try_from(4).unwrap()),
            )));
            let event = polynomial::<S>(0.37, 0.37, offset);
            let mut trajectory = OdeTrajectory::new();
            let (report, _) = run(
                &plan,
                &[&event],
                span,
                &mut trajectory,
                &mut OdeWorkspace::new(),
            );
            assert_eq!(report.unwrap().events(), expected);
            let events = trajectory.events();
            assert_eq!(events.len(), expected);
            if expected == 1 {
                assert_eq!(events[0].occurrence(), EventOccurrence::Tangency);
                assert!((events[0].time() - s(0.37)).abs() < s(4e-6));
                assert!(matches!(
                    events[0].evidence(),
                    EventEvidence::Contact { .. }
                ));
            }
        }
    }
    // No false completeness claim when endpoint sampling misses a double crossing.
    let event = polynomial::<S>(0.23, 0.71, 0.0);
    let mut trajectory = OdeTrajectory::new();
    let (report, _) = run(
        &dp5::<S>(steps()),
        &[&event],
        (S::ZERO, S::ONE),
        &mut trajectory,
        &mut OdeWorkspace::new(),
    );
    assert_eq!(report.unwrap().events(), 0);
    assert!(trajectory.events().is_empty());
}
#[test]
fn analytic_search_f32() {
    search_cases::<f32>();
}
#[test]
fn analytic_search_f64() {
    search_cases::<f64>();
}

struct Indexed<S: RealScalar + NumericScalar<Real = S>> {
    requested: Vec<(usize, S, S)>,
    extras: Vec<S>,
    events: Vec<EventInfo<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for Indexed<S> {
    fn default() -> Self {
        Self {
            requested: Vec::new(),
            extras: Vec::new(),
            events: Vec::new(),
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeOutputSink<S> for Indexed<S> {
    fn emit(&mut self, t: S, _: &[S]) -> Result<(), SciOxideError> {
        self.extras.push(t);
        Ok(())
    }
    fn emit_requested(&mut self, i: usize, t: S, y: &[S]) -> Result<(), SciOxideError> {
        self.requested.push((i, t, y[0]));
        Ok(())
    }
    fn event(&mut self, event: OdeEvent<'_, S>) -> Result<(), SciOxideError> {
        self.events.push(EventInfo::from_event(event)?);
        Ok(())
    }
}
struct TimeRoot<S> {
    root: S,
    terminal: bool,
}
impl<S: RealScalar + NumericScalar<Real = S>> EventFunction<S> for TimeRoot<S> {
    fn eval(&self, t: S, _: &[S]) -> Result<S, SciOxideError> {
        Ok(t - self.root)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        self.terminal
    }
}
fn output_cases<S: RealScalar + NumericScalar<Real = S>>() {
    for backwards in [false, true] {
        let span = if backwards {
            (S::ONE, S::ZERO)
        } else {
            (S::ZERO, S::ONE)
        };
        for terminal in [None, Some(span.0), Some(s(0.5))] {
            let mut times = vec![S::ZERO, S::ZERO, s(0.25), s(0.5), s(0.5), S::ONE, S::ONE];
            if backwards {
                times.reverse();
            }
            let plan =
                dp5::<S>(steps().with_output_schedule(OdeOutputSchedule::times(times.clone())));
            let root = terminal.map(|root| TimeRoot {
                root,
                terminal: true,
            });
            let events: Vec<&dyn EventFunction<S>> = root
                .iter()
                .map(|event| event as &dyn EventFunction<S>)
                .collect();
            let mut sink = Indexed::default();
            let (report, _) = run(&plan, &events, span, &mut sink, &mut OdeWorkspace::new());
            let report = report.unwrap();
            let expected: Vec<_> = times
                .iter()
                .enumerate()
                .filter(|(_, time)| {
                    if backwards {
                        **time >= report.final_time()
                    } else {
                        **time <= report.final_time()
                    }
                })
                .collect();
            assert_eq!(sink.requested.len(), expected.len());
            assert!(sink.extras.is_empty());
            for ((index, time, state), (expected_index, expected_time)) in
                sink.requested.iter().zip(expected)
            {
                assert_eq!(*index, expected_index);
                assert_eq!(*time, *expected_time);
                assert!((*state - *time).abs() < s(2e-6));
            }
        }
        for (initial, final_state) in [(false, false), (true, false), (false, true), (true, true)] {
            let plan = dp5::<S>(steps().with_output_schedule(OdeOutputSchedule::Times {
                times: vec![],
                initial,
                final_state,
            }));
            let mut sink = Indexed::default();
            let (report, _) = run(&plan, &[], span, &mut sink, &mut OdeWorkspace::new());
            assert!(report.unwrap().is_success());
            assert!(sink.requested.is_empty());
            assert_eq!(
                sink.extras.len(),
                usize::from(initial) + usize::from(final_state)
            );
        }
    }
}
#[test]
fn output_mapping_f32() {
    output_cases::<f32>();
}
#[test]
fn output_mapping_f64() {
    output_cases::<f64>();
}

fn radau_events<S: RealScalar + NumericScalar<Real = S>>() {
    for backwards in [false, true] {
        for terminal in [false, true] {
            let span = if backwards {
                (S::ONE, S::ZERO)
            } else {
                (S::ZERO, S::ONE)
            };
            let mut event = polynomial::<S>(0.23, 0.71, 0.0);
            event.terminal = terminal;
            let mut times = vec![s(0.1), s(0.2), s(0.5), s(0.8), s(0.9)];
            if backwards {
                times.reverse();
            }
            let plan = (
                RadauIia::new(),
                OdeOptions::new()
                    .with_initial_step(S::ONE)
                    .with_event_options(options(EventSearchPolicy::DerivativeAssisted(
                        Budget::try_from(8).unwrap(),
                    )))
                    .with_output_schedule(OdeOutputSchedule::times(times.clone())),
            );
            let mut workspace = OdeWorkspace::new();
            // Repeat the epoch to exercise cleanup after a truncated terminal segment.
            for _ in 0..2 {
                let mut sink = Indexed::default();
                let (report, state) = run(&plan, &[&event], span, &mut sink, &mut workspace);
                let report = report.unwrap();
                assert_eq!(
                    report.termination(),
                    if terminal {
                        Some(OdeTermination::TerminalEvent)
                    } else {
                        Some(OdeTermination::ReachedEnd)
                    }
                );
                assert_eq!(report.events(), if terminal { 1 } else { 2 });
                assert_eq!(sink.events.len(), report.events());
                let root = if backwards { s(0.71) } else { s(0.23) };
                assert!((sink.events[0].time() - root).abs() < s(4e-6));
                let expected_time = if terminal { root } else { span.1 };
                assert!((report.final_time() - expected_time).abs() < s(4e-6));
                assert_eq!(state.time(), report.final_time());
                assert!((state.values()[0] - expected_time).abs() < s(4e-6));
                let expected: Vec<_> = times
                    .iter()
                    .copied()
                    .filter(|&time| {
                        !terminal
                            || if backwards {
                                time >= root
                            } else {
                                time <= root
                            }
                    })
                    .collect();
                assert_eq!(sink.requested.len(), expected.len());
                for ((_, time, value), expected) in sink.requested.iter().zip(expected) {
                    assert_eq!(*time, expected);
                    assert!((*value - expected).abs() < s(4e-6));
                }
            }
        }
    }
}

#[test]
fn radau_shared_events_and_sampled_output_f32() {
    radau_events::<f32>();
}
#[test]
fn radau_shared_events_and_sampled_output_f64() {
    radau_events::<f64>();
}

#[test]
fn terminal_ties_and_reused_fsal_state() {
    let plan = dp5::<f64>(steps().with_output_schedule(OdeOutputSchedule::FinalOnly));
    let roots = [false, true, false, true].map(|terminal| TimeRoot {
        root: 0.5,
        terminal,
    });
    let events: Vec<&dyn EventFunction<f64>> = roots
        .iter()
        .map(|event| event as &dyn EventFunction<f64>)
        .collect();
    let mut workspace = OdeWorkspace::new();
    for _ in 0..2 {
        let mut sink = OdeTrajectory::new();
        let (report, state) = run(&plan, &events, (0.0, 1.0), &mut sink, &mut workspace);
        let report = report.unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::TerminalEvent));
        assert_eq!(report.terminal_event(), Some(1));
        assert_eq!(
            sink.events()
                .iter()
                .map(|event| event.index())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(report.final_time(), 0.5);
        assert!((state.values()[0] - 0.5).abs() < 1e-14);
        assert_eq!(sink.times(), [0.5]);
    }
}

#[test]
fn failures_do_not_commit_candidate_state() {
    let plan = dp5::<f64>(
        steps()
            .with_event_options(options(EventSearchPolicy::Subdivide(
                Budget::try_from(16).unwrap(),
            )))
            .with_max_evaluations(Budget::try_from(10).unwrap()),
    );
    let event = TimeRoot {
        root: 0.37,
        terminal: false,
    };
    let mut workspace = OdeWorkspace::new();
    let (report, state) = run(
        &plan,
        &[&event],
        (0.0, 1.0),
        &mut OdeTrajectory::new(),
        &mut workspace,
    );
    let report = report.unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert_eq!(report.final_time(), 0.0);
    assert_eq!(state.time(), 0.0);
    assert_eq!(report.counts().total_providers().unwrap(), 10);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let plan = dp5::<f64>(
        steps().with_event_options(
            options(EventSearchPolicy::Endpoints)
                .with_localization(StepTolerance::new(1e-15, 0.0).unwrap(), Budget::ONE),
        ),
    );
    // A linear root is solved exactly by the first interpolation.
    let event = Polynomial {
        terminal: true,
        ..polynomial::<f64>(0.37, -1.0, 0.0)
    };
    let mut workspace = OdeWorkspace::new();
    let (failure, state) = run(
        &plan,
        &[&event],
        (0.0, 1.0),
        &mut OdeTrajectory::new(),
        &mut workspace,
    );
    assert!(matches!(
        failure.unwrap_err().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(state.time(), 0.0);
    assert_eq!(state.values(), [0.0]);
    assert_eq!(progress_time(&workspace), 0.0);
}

#[test]
fn localization_budget_denial_preserves_the_accepted_state_and_exact_counts() {
    // Seven DP5 RHS calls and two discovery calls precede localization. Stop
    // before its first endpoint, second endpoint, or first interpolated point.
    for limit in [9, 10, 11] {
        let plan = dp5::<f64>(steps().with_max_evaluations(Budget::try_from(limit).unwrap()));
        let event = TimeRoot {
            root: 0.37,
            terminal: true,
        };
        let mut workspace = OdeWorkspace::new();
        let mut trajectory = OdeTrajectory::new();
        let (report, state) = run(
            &plan,
            &[&event],
            (0.0, 1.0),
            &mut trajectory,
            &mut workspace,
        );
        let report = report.unwrap();
        assert_eq!(
            report.termination(),
            Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
        );
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.values(), [0.0]);
        assert!(trajectory.events().is_empty());
        let total = report.accounting().total().unwrap();
        assert_eq!(total.total_providers().unwrap(), limit);
        assert_eq!(total.provider(EvaluationRole::Event), limit - 7);
        assert_eq!(total.provider(EvaluationRole::Function), 0);
        assert_eq!(report.event_search().unwrap().evaluations(), limit - 7);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
    }
}

#[derive(Clone, Copy)]
enum LocalizationFailure {
    Error,
    Nonfinite,
    Cancellation,
}

struct FailingLocalization {
    failure: LocalizationFailure,
    calls: core::cell::Cell<usize>,
    cancellation: scioxide_core::CancellationToken,
}

impl EventFunction<f64> for FailingLocalization {
    fn eval(&self, t: f64, _: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        if t > 0.0 && t < 1.0 {
            match self.failure {
                LocalizationFailure::Error => {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("event"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "localization callback failure",
                    ));
                }
                LocalizationFailure::Nonfinite => return Ok(f64::NAN),
                LocalizationFailure::Cancellation => self.cancellation.cancel(),
            }
        }
        Ok(t * t - 0.37)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        true
    }
}

#[test]
fn localization_failures_and_cancellation_are_counted_once_without_committing() {
    for failure in [
        LocalizationFailure::Error,
        LocalizationFailure::Nonfinite,
        LocalizationFailure::Cancellation,
    ] {
        let event = FailingLocalization {
            failure,
            calls: core::cell::Cell::new(0),
            cancellation: scioxide_core::CancellationToken::new(),
        };
        let plan = dp5::<f64>(steps());
        let events: [&dyn EventFunction<f64>; 1] = [&event];
        let problem = OdeProblem::new(&Unit, 1.0).with_events(&events);
        let mut state = OdeState::from_slice(0.0, &[0.0]).unwrap();
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let result = plan.0.advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&event.cancellation),
        );
        match failure {
            LocalizationFailure::Error => assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::InvalidInput
            ),
            LocalizationFailure::Nonfinite => assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::NonFinite
            ),
            LocalizationFailure::Cancellation => {
                let report = result.unwrap();
                assert_eq!(
                    report.termination(),
                    Some(OdeTermination::ExecutionStopped(
                        scioxide_core::ExecutionStopReason::Cancelled
                    ))
                );
                assert_eq!(
                    report.event_search().unwrap().evaluations(),
                    event.calls.get()
                );
            }
        }
        let total = workspace.last_attempt().unwrap().counts().total().unwrap();
        assert_eq!(total.provider(EvaluationRole::Event), event.calls.get());
        assert_eq!(total.provider(EvaluationRole::Function), 0);
        assert!(event.calls.get() > 4);
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.values(), [0.0]);
    }
}

#[test]
fn default_localization_uses_scalar_precision_for_f32() {
    let options = OdeEventOptions::<f32>::new();
    assert_eq!(options.localization().absolute(), 4.0 * f32::EPSILON);
    assert_eq!(options.localization().relative(), 4.0 * f32::EPSILON);
    assert_eq!(options.max_iterations().get(), 64);
    let plan = dp5::<f32>(steps());
    let event = polynomial::<f32>(0.37, -1.0, 0.0);
    let mut trajectory = OdeTrajectory::new();
    let (report, _) = run(
        &plan,
        &[&event],
        (0.0, 1.0),
        &mut trajectory,
        &mut OdeWorkspace::new(),
    );
    assert!(report.unwrap().is_success());
    assert_eq!(trajectory.events().len(), 1);
    assert!((trajectory.events()[0].time() - 0.37).abs() <= 4.0 * f32::EPSILON);
}

#[test]
fn missing_derivative_is_rejected_before_callbacks() {
    let plan = dp5::<f64>(
        steps().with_event_options(options(EventSearchPolicy::DerivativeAssisted(Budget::ONE))),
    );
    let event = TimeRoot {
        root: 0.5,
        terminal: false,
    };
    let mut workspace = OdeWorkspace::new();
    let (failure, _) = run(
        &plan,
        &[&event],
        (0.0, 1.0),
        &mut OdeTrajectory::new(),
        &mut workspace,
    );
    assert!(matches!(
        failure.unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert_eq!(attempt.counts().own().total_providers().unwrap(), 0);
}

#[test]
fn sink_failure_and_retry_preserve_commit_boundary() {
    let plan = dp5::<f64>(steps().with_output_schedule(OdeOutputSchedule::times(vec![0.25, 0.5])));
    let event = TimeRoot {
        root: 0.5,
        terminal: true,
    };
    let mut workspace = OdeWorkspace::new();
    let mut calls = 0;
    let mut sink = |_: f64, _: &[f64]| {
        calls += 1;
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("sink"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "injected failure",
        ))
    };
    let (failure, state) = run(&plan, &[&event], (0.0, 1.0), &mut sink, &mut workspace);
    assert!(failure.is_err());
    assert_eq!(calls, 1);
    assert_eq!(state.time(), 0.0);
    assert_eq!(progress_time(&workspace), 0.0);
    let mut output = OdeTrajectory::new();
    let (report, state) = run(&plan, &[&event], (0.0, 1.0), &mut output, &mut workspace);
    assert_eq!(report.unwrap().final_time(), 0.5);
    assert_eq!(state.time(), 0.5);
    assert_eq!(output.times(), [0.25, 0.5]);
}

#[test]
fn tangency_filters_and_sampled_stationary_endpoints() {
    for direction in [
        EventDirection::Rising,
        EventDirection::Falling,
        EventDirection::Either,
    ] {
        let mut event = polynomial::<f64>(0.5, 0.5, 0.0);
        event.direction = direction;
        let plan = dp5::<f64>(steps().with_event_options(options(
            EventSearchPolicy::DerivativeAssisted(Budget::try_from(8).unwrap()),
        )));
        let mut trajectory = OdeTrajectory::new();
        let (report, _) = run(
            &plan,
            &[&event],
            (0.0, 1.0),
            &mut trajectory,
            &mut OdeWorkspace::new(),
        );
        let expected = usize::from(direction == EventDirection::Either);
        assert_eq!(report.unwrap().events(), expected);
        assert_eq!(trajectory.events().len(), expected);
    }
}

#[test]
fn terminal_coincidence_group_is_not_transitive() {
    let plan = dp5::<f64>(
        steps().with_event_options(OdeEventOptions::new().with_localization(
            StepTolerance::new(0.02, 0.0).unwrap(),
            Budget::try_from(64).unwrap(),
        )),
    );
    // Linear roots are exact interpolation samples; the third is close to the second,
    // but outside the first terminal anchor's tolerance.
    let roots = [(0.5, true), (0.515625, false), (0.53125, true)]
        .map(|(root, terminal)| TimeRoot { root, terminal });
    let events: Vec<&dyn EventFunction<f64>> = roots
        .iter()
        .map(|event| event as &dyn EventFunction<f64>)
        .collect();
    let mut trajectory = OdeTrajectory::new();
    let (report, _) = run(
        &plan,
        &events,
        (0.0, 1.0),
        &mut trajectory,
        &mut OdeWorkspace::new(),
    );
    assert_eq!(report.unwrap().final_time(), 0.5);
    assert_eq!(
        trajectory
            .events()
            .iter()
            .map(|event| event.index())
            .collect::<Vec<_>>(),
        [0, 1]
    );
}

/// `-1` before `at` and `1` from it on: a sign change between adjacent times.
struct Jump {
    at: f64,
}
impl EventFunction<f64> for Jump {
    fn eval(&self, t: f64, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(if t < self.at { -1.0 } else { 1.0 })
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

#[test]
fn subdivision_merges_cells_of_rounding_sized_steps() {
    let policies = [
        EventSearchPolicy::Endpoints,
        EventSearchPolicy::Subdivide(Budget::try_from(2).unwrap()),
        EventSearchPolicy::Subdivide(Budget::try_from(8).unwrap()),
    ];
    for policy in policies {
        // Steps of one and three units in the last place, with and without a crossing.
        for ulps in [1, 3] {
            let start = 1.0_f64;
            let end = f64::from_bits(start.to_bits() + ulps);
            let plan = dp5::<f64>(steps().with_event_options(options(policy)));
            let quiet = TimeRoot {
                root: 0.5,
                terminal: false,
            };
            let jump = Jump {
                at: f64::from_bits(start.to_bits() + 1),
            };
            let mut trajectory = OdeTrajectory::new();
            let (report, state) = run(
                &plan,
                &[&quiet, &jump],
                (start, end),
                &mut trajectory,
                &mut OdeWorkspace::new(),
            );
            assert_eq!(
                report.unwrap().termination(),
                Some(OdeTermination::ReachedEnd)
            );
            assert_eq!(state.time(), end);
            let events = trajectory.events();
            assert_eq!(events.len(), 1, "{policy:?} {ulps}");
            assert_eq!(events[0].index(), 1);
            assert_eq!(events[0].occurrence(), EventOccurrence::Rising);
            assert!(events[0].time() >= start && events[0].time() <= jump.at);
        }
        // Ten sums of 0.1 fall a few units in the last place short of 1; the tenth
        // step absorbs that remainder instead of leaving a rounding-sized step.
        let plan = dp5::<f64>(
            OdeOptions::new()
                .with_initial_step(0.1)
                .with_maximum_step(0.1)
                .with_event_options(options(policy))
                .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
        );
        let event = TimeRoot {
            root: 0.55,
            terminal: false,
        };
        let mut trajectory = OdeTrajectory::new();
        let (report, state) = run(
            &plan,
            &[&event],
            (0.0, 1.0),
            &mut trajectory,
            &mut OdeWorkspace::new(),
        );
        assert_eq!(
            report.unwrap().termination(),
            Some(OdeTermination::ReachedEnd)
        );
        assert_eq!(state.time(), 1.0);
        let times = trajectory.times();
        assert_eq!(times.len(), 11, "{times:?}");
        assert!(
            (times[10] - times[9] - 0.1).abs() <= 4.0 * f64::EPSILON,
            "{times:?}"
        );
        let events = trajectory.events();
        assert_eq!(events.len(), 1);
        assert!((events[0].time() - 0.55).abs() < 4e-6);
    }
}

struct BadDerivative;
impl EventFunction<f64> for BadDerivative {
    fn eval(&self, t: f64, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(t + 1.0)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
    fn has_total_derivative(&self) -> bool {
        true
    }
    fn total_derivative(&self, _: f64, _: &[f64], _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(f64::NAN)
    }
}
#[test]
fn derivative_nonfinite_is_a_counted_callback_failure() {
    let plan = dp5::<f64>(
        steps().with_event_options(options(EventSearchPolicy::DerivativeAssisted(Budget::ONE))),
    );
    let mut workspace = OdeWorkspace::new();
    let (failure, state) = run(
        &plan,
        &[&BadDerivative],
        (0.0, 1.0),
        &mut OdeTrajectory::new(),
        &mut workspace,
    );
    assert!(matches!(
        failure.unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(state.time(), 0.0);
    assert_eq!(progress_time(&workspace), 0.0);
    // The initial and endpoint values and the failing derivative are all metered
    // event evaluations.
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.counts().own().provider(EvaluationRole::Event), 3);
}

#[test]
fn output_failure_preserves_earlier_committed_events() {
    /// Records events and fails on samples after the first committed step.
    #[derive(Default)]
    struct FailingAfterFirstStep {
        events: Vec<EventInfo<f64>>,
    }
    impl OdeOutputSink<f64> for FailingAfterFirstStep {
        fn emit(&mut self, t: f64, _: &[f64]) -> Result<(), SciOxideError> {
            if t > 0.25 {
                Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("sink"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "failure after first commit",
                ))
            } else {
                Ok(())
            }
        }
        fn event(&mut self, event: OdeEvent<'_, f64>) -> Result<(), SciOxideError> {
            self.events.push(EventInfo::from_event(event)?);
            Ok(())
        }
    }
    let plan = dp5::<f64>(
        OdeOptions::new()
            .with_initial_step(0.25)
            .with_maximum_step(0.25)
            .with_output_schedule(OdeOutputSchedule::times(vec![0.125, 0.375])),
    );
    let event = TimeRoot {
        root: 0.125,
        terminal: false,
    };
    let mut sink = FailingAfterFirstStep::default();
    let mut workspace = OdeWorkspace::new();
    let (failure, state) = run(&plan, &[&event], (0.0, 1.0), &mut sink, &mut workspace);
    assert!(failure.is_err());
    assert_eq!(state.time(), 0.25);
    assert_eq!(progress_time(&workspace), 0.25);
    assert_eq!(sink.events.len(), 1);
    assert_eq!(sink.events[0].time(), 0.125);
}

struct DirectedTimeRoot {
    root: TimeRoot<f64>,
    direction: EventDirection,
}
impl EventFunction<f64> for DirectedTimeRoot {
    fn eval(&self, time: f64, state: &[f64]) -> Result<f64, SciOxideError> {
        self.root.eval(time, state)
    }
    fn direction(&self) -> EventDirection {
        self.direction
    }
    fn is_terminal(&self) -> bool {
        self.root.terminal
    }
}

fn endpoint_event_cases<M: OdeMethod<f64> + Clone>(method: M) {
    for backwards in [false, true] {
        let span = if backwards { (1., 0.) } else { (0., 1.) };
        for terminal in [false, true] {
            let event = TimeRoot {
                root: span.1,
                terminal,
            };
            let plan = (method.clone(), steps());
            let mut trajectory = OdeTrajectory::new();
            let (report, state) = run(
                &plan,
                &[&event],
                span,
                &mut trajectory,
                &mut OdeWorkspace::new(),
            );
            let report = report.unwrap();
            assert_eq!(
                report.termination(),
                Some(if terminal {
                    OdeTermination::TerminalEvent
                } else {
                    OdeTermination::ReachedEnd
                })
            );
            assert_eq!(report.events(), 1);
            assert_eq!(trajectory.events().len(), 1);
            assert_eq!(trajectory.events()[0].time(), span.1);
            assert_eq!(report.final_time(), span.1);
            assert_eq!(state.time(), span.1);
            assert!((state.values()[0] - span.1).abs() < 1e-12);
        }
        let event = DirectedTimeRoot {
            root: TimeRoot {
                root: 0.5,
                terminal: true,
            },
            direction: if backwards {
                EventDirection::Rising
            } else {
                EventDirection::Falling
            },
        };
        let plan = (method.clone(), steps());
        let mut trajectory = OdeTrajectory::new();
        let (report, state) = run(
            &plan,
            &[&event],
            span,
            &mut trajectory,
            &mut OdeWorkspace::new(),
        );
        let report = report.unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(report.events(), 0);
        assert!(trajectory.events().is_empty());
        assert_eq!(state.time(), span.1);
        assert!((state.values()[0] - span.1).abs() < 1e-12);
    }
}
#[test]
fn dp5_endpoint_roots_and_excluded_terminal_direction() {
    endpoint_event_cases(DormandPrince5::new());
}
#[test]
fn radau_endpoint_roots_and_excluded_terminal_direction() {
    endpoint_event_cases(RadauIia::new());
}

/// `y² − c`: a simple root at `√c` with no exactly zero floating-point sample nearby.
struct SquareRoot(f64);
impl EventFunction<f64> for SquareRoot {
    fn eval(&self, _: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(y[0] * y[0] - self.0)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

#[test]
fn zero_localization_tolerance_resolves_adjacent_times_with_bounded_work() {
    let localization = OdeEventOptions::new().with_localization(
        StepTolerance::new(0.0, 0.0).unwrap(),
        Budget::try_from(64).unwrap(),
    );
    let root = 0.5_f64.sqrt();
    for span in [(0.0, 1.0), (1.0, 0.0)] {
        let plan = dp5::<f64>(steps().with_event_options(localization));
        let mut trajectory = OdeTrajectory::new();
        let (report, _) = run(
            &plan,
            &[&SquareRoot(0.5)],
            span,
            &mut trajectory,
            &mut OdeWorkspace::new(),
        );
        let report = report.unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(trajectory.events().len(), 1);
        let event = &trajectory.events()[0];
        // Nothing finer than adjacent times is representable.
        let evidence = event.evidence();
        assert!(
            matches!(evidence, EventEvidence::Crossing { bracket, .. }
                if bracket.lower().next_up() == bracket.upper()),
            "{evidence:?}"
        );
        assert!((event.time() - root).abs() <= f64::EPSILON);
        // One value at the start and one per accepted step; the rest localizes.
        let search = report.event_search().unwrap().evaluations();
        let localization_calls = search - 1 - report.accepted_steps();
        assert!(localization_calls <= 16, "{localization_calls} calls");
    }
}

/// Run the unit-slope system from `y(0) = 0` over `[0, 1]` in one step under `policy`.
fn from_initial_zero(event: &Polynomial<f64>, policy: EventSearchPolicy) -> Vec<EventInfo<f64>> {
    let plan = dp5::<f64>(steps().with_event_options(options(policy)));
    let mut trajectory = OdeTrajectory::new();
    let (report, _) = run(
        &plan,
        &[event],
        (0.0, 1.0),
        &mut trajectory,
        &mut OdeWorkspace::new(),
    );
    let report = report.unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(report.accepted_steps(), 1);
    trajectory.events().to_vec()
}

#[test]
fn initial_zero_takes_its_sign_from_the_next_nonzero_sample() {
    // g = y (y - 0.6) leaves zero downwards and rises through zero at 0.6.
    let event = polynomial::<f64>(0.0, 0.6, 0.0);
    // The endpoint sample alone shows no sign change after the initial zero.
    assert!(from_initial_zero(&event, EventSearchPolicy::Endpoints).is_empty());
    for policy in [
        EventSearchPolicy::Subdivide(Budget::try_from(8).unwrap()),
        EventSearchPolicy::DerivativeAssisted(Budget::ONE),
    ] {
        let events = from_initial_zero(&event, policy);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].occurrence(), EventOccurrence::Rising);
        assert!((events[0].time() - 0.6).abs() < 4e-6);
    }
}
