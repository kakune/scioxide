//! Independent analytic method-of-steps and prepared-resource coverage.
#![allow(clippy::unwrap_used, clippy::panic)]
use core::cell::Cell;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, ErrorTolerance, EvaluationRole, ExecutionContext,
    NormSelection, NumericScalar, PartialResultTrust, RealScalar, SciOxideError,
};
use scioxide_linalg::{MatrixView, Vector};
use scioxide_ode::{
    DdeHistoryBreak, DdeOptions, DdePlan, DdeProblem, DdeTermination, DdeWorkspaceRequest,
    EventDirection, EventFunction, OdeOptions, OdeOutputSchedule, OdeOutputSink, OdeState,
    OdeTrajectory,
};

fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn plan<S: RealScalar + NumericScalar<Real = S>>(
    end: f64,
    delays: &[f64],
    capacity: usize,
    ode: OdeOptions<S>,
) -> DdePlan<S> {
    DdePlan::new(
        1,
        (s(0.0), s(end)),
        &(delays.iter().copied().map(s).collect::<Vec<_>>()),
        &[],
        DdeOptions::new()
            .with_history_capacity(Budget::try_from(capacity).unwrap())
            .with_schedule_capacity(Budget::try_from(4096).unwrap())
            .with_ode_options(ode),
    )
    .unwrap()
}
fn options<S: RealScalar>(h: f64, tol: f64) -> OdeOptions<S> {
    OdeOptions::new()
        .with_initial_step(s(h))
        .with_maximum_step(s(h))
        .with_error_tolerance(
            ErrorTolerance::scalar(s(tol), s(tol), NormSelection::Infinity).unwrap(),
        )
}
fn limits(capacity: usize, schedule: usize) -> DdeOptions<f64> {
    DdeOptions::new()
        .with_history_capacity(Budget::try_from(capacity).unwrap())
        .with_schedule_capacity(Budget::try_from(schedule).unwrap())
}
/// A placeholder state; every advance overwrites it with the initial history.
fn placeholder<S: RealScalar + NumericScalar<Real = S>>(dimension: usize) -> OdeState<S> {
    OdeState::new(S::ZERO, Vector::zeros(dimension).unwrap())
}
fn order_one(time: f64) -> DdeHistoryBreak<f64> {
    DdeHistoryBreak::first_derivative(time)
}
// y'=y(t-1), history=1. Integrating each unit interval gives the finite series
// y(t)=1+sum_{k>=1} max(t-k+1,0)^k/k!, independently of any RK coefficients.
fn polynomial(t: f64) -> f64 {
    let mut value = 1.0;
    let mut factorial = 1.0;
    for k in 1..=8 {
        factorial *= f64::from(k);
        value += (t - f64::from(k) + 1.0).max(0.0).powi(k) / factorial;
    }
    value
}
fn polynomial_case<S: RealScalar + NumericScalar<Real = S>>(tol: f64) {
    let history = |_: S, out: &mut [S]| {
        out.fill(s(1.0));
        Ok(())
    };
    let rhs = |_: S, _: &[S], delayed: MatrixView<'_, S>, out: &mut [S]| {
        out[0] = delayed.get(0, 0)?;
        Ok(())
    };
    let times: Vec<S> = (0..=80).map(|i| s(f64::from(i) / 16.0)).collect();
    let plan = plan(
        5.0,
        &[1.0],
        128,
        options(0.25, tol).with_output_schedule(OdeOutputSchedule::times(times.clone())),
    );
    let problem = DdeProblem::new(&rhs, &history);
    let mut workspace = plan.workspace(DdeWorkspaceRequest::new(0)).unwrap();
    let mut state = placeholder(1);
    for _ in 0..2 {
        let mut trajectory = OdeTrajectory::new();
        let report = plan
            .advance(
                &problem,
                &mut state,
                &mut trajectory,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(DdeTermination::ReachedEnd));
        assert_eq!(trajectory.times(), times);
        for (&time, state) in trajectory.times().iter().zip(trajectory.states()) {
            let expected = polynomial(time.try_to_f64_exact().unwrap());
            assert!(
                (state.as_slice()[0] - s(expected)).abs() < s::<S>(tol * 200.0),
                "time={time:?} state={state:?} expected={expected}"
            );
        }
        assert!(report.counts().provider(EvaluationRole::Function) > 1);
        assert!(report.peak_history_segments() < 128);
    }
}
#[test]
fn plans_report_their_validated_options() {
    let validated = limits(64, 32).with_ode_options(options(0.25, 1e-8));
    let plan = DdePlan::new(1, (0.0, 4.0), &[1.0], &[], validated.clone()).unwrap();
    assert_eq!(plan.options(), &validated);
    assert_eq!(
        plan.options().history_capacity(),
        Budget::try_from(64).unwrap()
    );
}
#[test]
fn requirements_report_the_delay_storage_bytes() {
    let double = plan::<f64>(4.0, &[1.0, 0.5], 64, options(0.25, 1e-8));
    let bytes = |segments| {
        double
            .workspace_requirements(DdeWorkspaceRequest::new(0).with_history_segments(segments))
            .unwrap()
            .required_bytes()
    };
    // One state, two delays: six scalars per history slot plus two delayed values.
    assert_eq!(bytes(0), (6 + 2) * 8);
    assert_eq!(bytes(10), (11 * 6 + 2) * 8);
    assert_eq!(bytes(64) - bytes(63), 6 * 8);
    let single = plan::<f32>(4.0, &[1.0], 64, options(0.25, 1e-4));
    let requirements = single
        .workspace_requirements(DdeWorkspaceRequest::new(0).with_history_segments(3))
        .unwrap();
    assert_eq!(requirements.required_bytes(), (4 * 6 + 1) * 4);
}
#[test]
fn polynomial_f32() {
    polynomial_case::<f32>(2e-6);
}
#[test]
fn polynomial_f64() {
    polynomial_case::<f64>(1e-10);
}

fn exponential_error<S: RealScalar + NumericScalar<Real = S>>(tol: f64) -> (f64, usize) {
    // y=exp(t), with two genuine delayed contributions, delays 1 and 1/2.
    let history = |time: S, out: &mut [S]| {
        out[0] = time.exp();
        Ok(())
    };
    let rhs = |_: S, _: &[S], delayed: MatrixView<'_, S>, out: &mut [S]| {
        out[0] = (s::<S>(1.0).exp() * delayed.get(0, 0)?
            + s::<S>(0.5).exp() * delayed.get(1, 0)?)
            * s(0.5);
        Ok(())
    };
    let solution = plan(4.0, &[1.0, 0.5], 2048, options(0.5, tol))
        .solve(&DdeProblem::new(&rhs, &history))
        .unwrap();
    assert!(solution.report().is_success());
    (
        (solution.state().values()[0] / s::<S>(4.0).exp() - s(1.0))
            .abs()
            .try_to_f64_exact()
            .unwrap(),
        solution.report().counts().rejected(),
    )
}
#[test]
fn exponential_refinement() {
    let loose = exponential_error::<f64>(1e-4).0;
    let (tight, rejected) = exponential_error::<f64>(1e-10);
    assert!(tight < 2e-9, "{tight}");
    assert!(tight < loose / 100.0, "loose={loose} tight={tight}");
    assert!(rejected > 0);
    assert!(exponential_error::<f32>(2e-6).0 < 5e-5);
}

#[test]
fn scheduling_seeds_and_exact_landing() {
    let plan = DdePlan::new(
        1,
        (0.0_f64, 2.0),
        &[1.0, 0.5],
        &[order_one(-0.75)],
        limits(100, 100).with_ode_options(
            options(1.0, 1e-9).with_output_schedule(OdeOutputSchedule::AcceptedSteps),
        ),
    )
    .unwrap();
    assert_eq!(
        plan.discontinuities(),
        &[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    );
    let breaks = plan.discontinuities().to_vec();
    // History has a kink at -3/4. The RHS sees it at 1/4 and integrates it.
    let history = |t: f64, out: &mut [f64]| {
        out[0] = (t + 0.75).abs();
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], delayed: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = delayed.get(0, 0)?;
        Ok(())
    };
    let solution = plan.solve(&DdeProblem::new(&rhs, &history)).unwrap();
    assert!(solution.report().is_success());
    let trajectory = solution.trajectory();
    for time in breaks {
        assert!(trajectory.times().contains(&time));
    }
    let index = trajectory.times().iter().position(|t| *t == 0.25).unwrap();
    assert!((trajectory.states()[index].as_slice()[0] - 0.78125).abs() < 1e-10);
}

#[test]
fn duplicates_coupled_and_zero_dimension() {
    let history = |t: f64, out: &mut [f64]| {
        out.copy_from_slice(&[t, 2.0 * t]);
        Ok(())
    };
    let rhs = |t: f64, y: &[f64], z: MatrixView<'_, f64>, out: &mut [f64]| {
        assert_eq!(z.get(0, 0)?, z.get(1, 0)?);
        assert!((z.get(0, 1)? - 2.0 * (t - 0.5)).abs() < 1e-10);
        out[0] = z.get(0, 1)? - 2.0 * y[0] + 2.0;
        out[1] = 2.0 * out[0];
        Ok(())
    };
    let p = DdePlan::new(
        2,
        (0.0, 3.0),
        &[0.5, 0.5],
        &[],
        limits(8, 20).with_ode_options(options(0.25, 1e-8)),
    )
    .unwrap();
    let solution = p.solve(&DdeProblem::new(&rhs, &history)).unwrap();
    assert!(solution.report().is_success());
    assert!((solution.state().values()[1] - 6.0).abs() < 1e-10);
    let p = DdePlan::new(0, (0.0, 1.0), &[0.5], &[], limits(8, 8)).unwrap();
    let h = |_: f64, out: &mut [f64]| {
        assert!(out.is_empty());
        Ok(())
    };
    let f = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        assert!(out.is_empty());
        Ok(())
    };
    assert!(
        p.solve(&DdeProblem::new(&f, &h))
            .unwrap()
            .report()
            .is_success()
    );
}

#[test]
fn plan_rejections() {
    for delays in [
        vec![],
        vec![0.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
    ] {
        assert!(matches!(
            (DdePlan::new(1, (0.0, 1.0), &(delays), &[], limits(1, 1)))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
                | scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    assert!(DdePlan::new(1, (1.0, 0.0), &[1.0], &[], limits(1, 1)).is_err());
    assert!(DdePlan::new(1, (0.0, 3.0), &[1.0], &[], limits(1, 1)).is_err());
    assert!(DdePlan::new(1, (0.0, 1.0), &[1.0], &[order_one(-2.0)], limits(1, 10)).is_err());
    assert!(DdePlan::new(1, (1e20, 2e20), &[1.0], &[], limits(1, 10)).is_err());
    assert!(DdePlan::new(usize::MAX, (0.0, 1.0), &[1.0, 1.0], &[], limits(1, 10)).is_err());
}

#[test]
fn storage_exhaustion_and_budget_admission() {
    let hc = Cell::new(0);
    let rc = Cell::new(0);
    let history = |_: f64, out: &mut [f64]| {
        hc.set(hc.get() + 1);
        out[0] = 1.0;
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        rc.set(rc.get() + 1);
        out[0] = 1.0;
        Ok(())
    };
    let solution = plan(2.0, &[1.0], 1, options(0.25, 1e-8))
        .solve(&DdeProblem::new(&rhs, &history))
        .unwrap();
    assert_eq!(
        solution.report().termination(),
        Some(DdeTermination::HistoryCapacity)
    );
    assert_eq!(solution.report().final_time(), 0.25);
    assert_eq!(solution.state().values(), &[1.25]);
    for limit in [1, 2, 5, 12] {
        hc.set(0);
        rc.set(0);
        let budget = OdeOptions::new().with_max_evaluations(Budget::try_from(limit).unwrap());
        let solution = plan(2.0, &[1.0], 32, budget)
            .solve(&DdeProblem::new(&rhs, &history))
            .unwrap();
        let report = solution.report();
        assert_eq!(
            report.termination(),
            Some(DdeTermination::BudgetLimit(BudgetKind::Evaluation))
        );
        assert_eq!(hc.get() + rc.get(), limit);
        assert_eq!(report.counts().provider(EvaluationRole::Function), hc.get());
        assert_eq!(
            report.counts().provider(EvaluationRole::RightHandSide),
            rc.get()
        );
    }
}

#[test]
fn preflight_and_callback_failure_atomicity() {
    let calls = Cell::new(0);
    let history = |_: f64, out: &mut [f64]| {
        calls.set(calls.get() + 1);
        out[0] = 1.0;
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = f64::NAN;
        Ok(())
    };
    let outside = plan(
        2.0,
        &[1.0],
        32,
        OdeOptions::new().with_output_schedule(OdeOutputSchedule::times(vec![3.0])),
    );
    let mut workspace = outside.workspace(DdeWorkspaceRequest::new(0)).unwrap();
    let mut state = OdeState::new(7.0, Vector::from_vec(vec![5.0]));
    assert!(
        outside
            .advance(
                &DdeProblem::new(&rhs, &history),
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    assert_eq!((state.time(), state.values()), (7.0, &[5.0][..]));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let failing = plan(2.0, &[1.0], 32, OdeOptions::new());
    let failure = failing.solve(&DdeProblem::new(&rhs, &history)).unwrap_err();
    assert!(matches!(
        (failure.error()).view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(failure.progress().unwrap().time(), 0.0);
    assert_eq!(failure.progress().unwrap().state(), &[1.0]);
    // An incomplete initial history leaves no accepted state.
    let bad_history = |_: f64, _: &mut [f64]| Ok(());
    let failure = failing
        .solve(&DdeProblem::new(&rhs, &bad_history))
        .unwrap_err();
    assert!(matches!(
        (failure.error()).view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert!(failure.progress().is_none_or(|progress| {
        progress.accepted_steps() == 0 && progress.trust() == PartialResultTrust::DiagnosticOnly
    }));
}

struct Stop;
impl EventFunction<f64> for Stop {
    fn eval(&self, _: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(y[0] - 1.375)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Rising
    }
    fn is_terminal(&self) -> bool {
        true
    }
}
#[derive(Default)]
struct Indexed(Vec<(usize, f64)>);
impl OdeOutputSink<f64> for Indexed {
    fn emit(&mut self, _: f64, _: &[f64]) -> Result<(), SciOxideError> {
        panic!("no implicit samples")
    }
    fn emit_requested(&mut self, index: usize, time: f64, _: &[f64]) -> Result<(), SciOxideError> {
        self.0.push((index, time));
        Ok(())
    }
}
#[test]
fn terminal_requested_duplicates() {
    let history = |_: f64, out: &mut [f64]| {
        out[0] = 1.0;
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = 1.0;
        Ok(())
    };
    let plan = plan(
        2.0,
        &[1.0],
        32,
        OdeOptions::new()
            .with_output_schedule(OdeOutputSchedule::times(vec![0.0, 0.25, 0.25, 0.5])),
    );
    let events: [&dyn EventFunction<f64>; 1] = [&Stop];
    let problem = DdeProblem::new(&rhs, &history).with_events(&events);
    let mut sink = Indexed::default();
    let report = plan
        .advance(
            &problem,
            &mut placeholder(1),
            &mut sink,
            &mut plan
                .workspace(DdeWorkspaceRequest::new(events.len()))
                .unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(DdeTermination::TerminalEvent));
    assert!((report.final_time() - 0.375).abs() < 1e-6);
    assert_eq!(report.terminal_event(), Some(0));
    assert_eq!(sink.0, vec![(0, 0.0), (1, 0.25), (2, 0.25)]);
}

#[test]
fn cancellation_and_sink_failure_preserve_committed_state() {
    use scioxide_core::CancellationToken;
    let cancellation = CancellationToken::new();
    let history = |_: f64, out: &mut [f64]| {
        out[0] = 1.0;
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = 1.0;
        Ok(())
    };
    let problem = DdeProblem::new(&rhs, &history);
    let plan = plan(
        2.0,
        &[1.0],
        32,
        options(0.25, 1e-8).with_output_schedule(OdeOutputSchedule::AcceptedSteps),
    );
    let mut workspace = plan.workspace(DdeWorkspaceRequest::new(0)).unwrap();
    let mut state = placeholder(1);
    let mut sink = |time: f64, _: &[f64]| {
        if time == 0.5 {
            cancellation.cancel();
        }
        Ok(())
    };
    let report = plan
        .advance(
            &problem,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        report.termination(),
        Some(DdeTermination::ExecutionStopped(_))
    ));
    assert_eq!(report.final_time(), 0.5);
    assert_eq!(state.time(), 0.5);
    let mut sink = |time: f64, _: &[f64]| {
        if time == 0.5 {
            Err(SciOxideError::allocation_failure(
                scioxide_core::quantity!("injected_sink_failure"),
                1,
            ))
        } else {
            Ok(())
        }
    };
    assert!(
        plan.advance(
            &problem,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!((state.time(), state.values()), (0.25, &[1.25][..]));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().time(), 0.25);
    assert_eq!(attempt.progress().unwrap().state(), &[1.25]);
    let report = plan
        .advance(
            &problem,
            &mut state,
            &mut |_: f64, _: &[f64]| Ok(()),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(state.values(), &[3.0]);
    // Cancellation already present before history initialization invokes no provider.
    let calls = Cell::new(0);
    let history = |_: f64, out: &mut [f64]| {
        calls.set(calls.get() + 1);
        out[0] = 1.0;
        Ok(())
    };
    let stopped = plan.advance(
        &DdeProblem::new(&rhs, &history),
        &mut placeholder(1),
        &mut |_: f64, _: &[f64]| Ok(()),
        &mut workspace,
        ExecutionContext::sequential().with_cancellation(&cancellation),
    );
    assert!(stopped.is_err());
    assert_eq!(calls.get(), 0);
    let progress = workspace.last_attempt().unwrap().progress();
    assert!(progress.is_none_or(|progress| progress.accepted_steps() == 0));
}

#[test]
fn history_failure_during_proposal_is_metered_and_not_committed() {
    let hc = Cell::new(0);
    let rc = Cell::new(0);
    let history = |_: f64, out: &mut [f64]| {
        hc.set(hc.get() + 1);
        if hc.get() == 4 {
            out[0] = f64::INFINITY;
        } else {
            out[0] = 1.0;
        }
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        rc.set(rc.get() + 1);
        out[0] = 1.0;
        Ok(())
    };
    let plan = plan(2.0, &[1.0], 32, OdeOptions::new());
    let mut workspace = plan.workspace(DdeWorkspaceRequest::new(0)).unwrap();
    let mut state = placeholder(1);
    let error = plan
        .advance(
            &DdeProblem::new(&rhs, &history),
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(state.time(), 0.0);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.progress().unwrap().time(), 0.0);
    assert_eq!(attempt.counts().own().provider(EvaluationRole::Function), 4);
    assert_eq!(
        attempt
            .counts()
            .own()
            .provider(EvaluationRole::RightHandSide),
        rc.get()
    );
}

#[test]
fn small_delays_and_rounding_boundaries() {
    for (start, end, delay) in [
        (0.0, 1e-8, 1e-9),
        (1.0, 1.0 + 1e-12, 1e-13),
        (-2.0, -1.0, 0.1),
    ] {
        let p = DdePlan::new(1, (start, end), &[delay], &[], limits(128, 1024)).unwrap();
        let history = |t: f64, out: &mut [f64]| {
            out[0] = t;
            Ok(())
        };
        let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
            out[0] = 1.0;
            Ok(())
        };
        let solution = p.solve(&DdeProblem::new(&rhs, &history)).unwrap();
        assert!(solution.report().is_success());
        assert!((solution.state().values()[0] - end).abs() < 1e-14);
    }
}

#[test]
fn delay_interpolation_refines_at_smooth_interior_points() {
    let history = |t: f64, out: &mut [f64]| {
        out[0] = t.exp();
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], z: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = core::f64::consts::E * z.get(0, 0)?;
        Ok(())
    };
    let errors: Vec<f64> = [0.25, 0.125, 0.0625]
        .into_iter()
        .map(|step| {
            let plan = plan(
                3.0,
                &[1.0],
                256,
                options(step, 0.1).with_output_schedule(OdeOutputSchedule::times(
                    (64..=192).map(|i| f64::from(i) / 64.0).collect(),
                )),
            );
            let mut maximum_error = 0.0_f64;
            let report = plan
                .advance(
                    &DdeProblem::new(&rhs, &history),
                    &mut placeholder(1),
                    &mut |time: f64, y: &[f64]| {
                        maximum_error = maximum_error.max((y[0] / time.exp() - 1.0).abs());
                        Ok(())
                    },
                    &mut plan.workspace(DdeWorkspaceRequest::new(0)).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            maximum_error
        })
        .collect();
    assert!(errors[1] < errors[0] / 8.0, "{errors:?}");
    assert!(errors[2] < errors[1] / 8.0, "{errors:?}");
}

#[test]
fn injected_executor_and_tolerance_shape_preflight() {
    use scioxide_core::SequentialExecutor;
    let calls = Cell::new(0);
    let history = |_: f64, out: &mut [f64]| {
        calls.set(calls.get() + 1);
        out[0] = 1.0;
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = 1.0;
        Ok(())
    };
    // A tolerance of another shape is rejected while the plan is prepared.
    let tolerance =
        ErrorTolerance::try_vector(&[1e-8, 1e-8], 1e-8, NormSelection::Infinity).unwrap();
    let error = DdePlan::new(
        1,
        (0.0, 2.0),
        &[1.0],
        &[],
        limits(32, 4096).with_ode_options(OdeOptions::new().with_error_tolerance(tolerance)),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(calls.get(), 0);
    let plan = plan(2.0, &[1.0], 32, OdeOptions::new());
    let executor = SequentialExecutor;
    let mut state = placeholder(1);
    let report = plan
        .advance(
            &DdeProblem::new(&rhs, &history),
            &mut state,
            &mut |_: f64, _: &[f64]| Ok(()),
            &mut plan.workspace(DdeWorkspaceRequest::new(0)).unwrap(),
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((state.values()[0] - 3.0).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn incomplete_rhs_is_rejected_instead_of_inheriting_scratch() {
    let history = |_: f64, out: &mut [f64]| {
        out.fill(1.0);
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, _: &mut [f64]| Ok(());
    let failure = plan(2.0, &[1.0], 32, OdeOptions::new())
        .solve(&DdeProblem::new(&rhs, &history))
        .unwrap_err();
    assert!(matches!(
        (failure.error()).view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(failure.progress().unwrap().state(), &[1.0]);
}

fn decimal_probe(delays: &[f64], tolerance: f64, maximum_step: f64) -> (usize, Vec<f64>) {
    let history = |_: f64, out: &mut [f64]| {
        out[0] = 1.;
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], z: MatrixView<'_, f64>, out: &mut [f64]| {
        let mut sum = 0.;
        for j in 0..delays.len() {
            sum += z.get(j, 0)?;
        }
        out[0] = -sum / f64::from(u32::try_from(delays.len()).unwrap());
        Ok(())
    };
    let solution = plan(
        20.,
        delays,
        4096,
        options(maximum_step, tolerance)
            .with_initial_step(1e-3)
            .with_output_schedule(OdeOutputSchedule::times(
                (0..=80).map(|i| f64::from(i) / 4.).collect(),
            )),
    )
    .solve(&DdeProblem::new(&rhs, &history))
    .unwrap();
    assert!(solution.report().is_success());
    (
        solution.report().counts().accepted(),
        solution
            .trajectory()
            .states()
            .iter()
            .map(|v| v.as_slice()[0])
            .collect(),
    )
}
#[test]
fn decimal_delays_have_bounded_schedules_and_work() {
    let cases: [&[f64]; 3] = [&[1., 0.25], &[1., 0.1], &[1., 0.3, 0.7]];
    let schedule = |end, delays: &[f64]| plan::<f64>(end, delays, 4096, OdeOptions::new());
    let binary = schedule(20., cases[0]).discontinuities().len();
    let mut steps = Vec::new();
    for delays in cases {
        let forward = schedule(20., delays);
        assert!(forward.discontinuities().len() <= 2 * binary);
        for pair in forward.discontinuities().windows(2) {
            assert!(pair[1] - pair[0] > 8. * f64::EPSILON * pair[0].abs().max(pair[1].abs()));
        }
        let reversed = schedule(20., &delays.iter().rev().copied().collect::<Vec<_>>());
        assert_eq!(forward.discontinuities(), reversed.discontinuities());
        let narrow = plan::<f32>(20., delays, 4096, OdeOptions::new());
        assert!(narrow.discontinuities().len() <= 2 * binary);
        for pair in narrow.discontinuities().windows(2) {
            assert!(pair[1] - pair[0] > 8. * f32::EPSILON * pair[0].abs().max(pair[1].abs()));
        }
        let (accepted, values) = decimal_probe(delays, 1e-8, 1.);
        let (_, reference) = decimal_probe(delays, 1e-11, 1.);
        for (actual, expected) in values.iter().zip(reference) {
            assert!(
                (actual - expected).abs() <= 3e-7 * (1. + expected.abs()),
                "{delays:?}: {actual} {expected}"
            );
        }
        eprintln!(
            "delays={delays:?} breaks={} accepted={accepted}",
            forward.discontinuities().len()
        );
        steps.push(accepted);
    }
    assert!(steps[1] <= 300, "{steps:?}");
    // A causal method-of-steps solve needs >=200 steps for delay 0.1. Apply
    // the same maximum step to both rows when comparing scheduler work.
    let binary = decimal_probe(cases[0], 1e-8, 0.1).0;
    let decimal = decimal_probe(cases[1], 1e-8, 0.1).0;
    eprintln!("common maximum step 0.1: binary={binary}, decimal={decimal}");
    assert!(decimal <= 2 * binary);
}
#[test]
fn history_derivative_orders_bound_propagation() {
    let make = |order| {
        DdePlan::new(
            1,
            (0., 20.),
            &[1.],
            &[DdeHistoryBreak::new(-0.5, order)],
            limits(1024, 32),
        )
    };
    assert!(make(0).is_err());
    assert_eq!(make(4).unwrap().discontinuities(), &[0.5, 1., 2., 3., 4.]);
    assert_eq!(make(6).unwrap().discontinuities(), &[1., 2., 3., 4.]);
    let long = plan::<f32>(1e4, &[1., 0.1], 4096, OdeOptions::new());
    assert!(
        long.discontinuities().len()
            <= 2 * plan::<f32>(20., &[1., 0.25], 4096, OdeOptions::new())
                .discontinuities()
                .len()
    );
}

#[test]
fn schedule_capacity_is_a_resource_limit_on_the_merged_schedule() {
    let delays = || vec![1.0, 0.25];
    let exact = DdePlan::new(1, (0.0, 20.0), &(delays()), &[], limits(64, 4096))
        .unwrap()
        .discontinuities()
        .len();
    // The span start is a scheduled seed in addition to the published breaks.
    let required = exact + 1;
    let fits = DdePlan::new(1, (0.0, 20.0), &(delays()), &[], limits(64, required)).unwrap();
    assert_eq!(fits.discontinuities().len(), exact);
    for capacity in [1, required / 2, required - 1] {
        let error =
            DdePlan::new(1, (0.0, 20.0), &(delays()), &[], limits(64, capacity)).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { limit, required, .. }
                if limit == capacity && required == capacity + 1
        ));
    }
}

#[test]
fn rounded_break_at_the_span_end_uses_the_endpoint() {
    let schedule = plan::<f64>(0.3, &[0.1], 32, OdeOptions::new());
    assert_eq!(schedule.discontinuities().last(), Some(&0.3));
    assert!(schedule.discontinuities().iter().all(|&t| t <= 0.3));
}

#[test]
fn explicit_history_reservation_covers_the_first_solve() {
    use scioxide_core::Capacity;
    use scioxide_ode::DdeWorkspace;

    let plan = plan(4.0, &[1.0], 1_000_000, options(0.125, 1e-8));
    let request = DdeWorkspaceRequest::new(0).with_history_segments(32);
    let requirements = plan.workspace_requirements(request).unwrap();
    let mut workspace = DdeWorkspace::new();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let history = |_: f64, out: &mut [f64]| {
        out.fill(1.0);
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], delayed: MatrixView<'_, f64>, out: &mut [f64]| {
        out[0] = delayed.get(0, 0)?;
        Ok(())
    };
    let mut state = placeholder(1);
    for _ in 0..2 {
        let report = plan
            .advance(
                &DdeProblem::new(&rhs, &history),
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!(report.peak_history_segments() <= request.history_segments());
        assert!((state.values()[0] - polynomial(4.0)).abs() < 1e-6);
    }
    assert!(
        plan.workspace_requirements(request.with_history_segments(1_000_001))
            .is_err()
    );
    assert!(
        plan.workspace_requirements(request.with_history_segments(usize::MAX))
            .is_err()
    );
}

#[test]
fn retained_reservations_do_not_override_a_smaller_plan_limit() {
    let large = plan(2.0, &[1.0], 64, options(0.25, 1e-8));
    let mut workspace = large
        .workspace(DdeWorkspaceRequest::new(0).with_history_segments(64))
        .unwrap();
    let small = plan(2.0, &[1.0], 3, options(0.25, 1e-8));
    let history = |_: f64, out: &mut [f64]| {
        out.fill(1.0);
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out.fill(1.0);
        Ok(())
    };
    let mut state = placeholder(1);
    let report = small
        .advance(
            &DdeProblem::new(&rhs, &history),
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(DdeTermination::HistoryCapacity));
    assert_eq!(report.peak_history_segments(), 3);
    assert_eq!(state.time(), 0.75);
    assert!((state.values()[0] - 1.75).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn history_reservation_tracks_dimension_changes_and_checks_overflow() {
    use scioxide_core::{Capacity, SciOxideErrorView};

    let request = DdeWorkspaceRequest::new(0).with_history_segments(8);
    let first = plan(2.0, &[1.0], 16, options(0.25, 1e-8));
    let mut workspace = first.workspace(request).unwrap();
    let second = DdePlan::new(
        2,
        (0.0, 2.0),
        &[1.0],
        &[],
        limits(16, 64).with_ode_options(options(0.25, 1e-8)),
    )
    .unwrap();
    let requirements = second.workspace_requirements(request).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    let history = |_: f64, out: &mut [f64]| {
        out.fill(1.0);
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out.copy_from_slice(&[1.0, 2.0]);
        Ok(())
    };
    let mut state = placeholder(2);
    let report = second
        .advance(
            &DdeProblem::new(&rhs, &history),
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((state.values()[0] - 3.0).abs() <= 16.0 * f64::EPSILON);
    assert!((state.values()[1] - 5.0).abs() <= 16.0 * f64::EPSILON);

    let maximum = plan::<f64>(2.0, &[1.0], usize::MAX, options(0.25, 1e-8));
    assert!(maximum.workspace(DdeWorkspaceRequest::new(0)).is_ok());
    for count in [usize::MAX, usize::MAX / 2] {
        assert!(matches!(
            maximum
                .workspace_requirements(request.with_history_segments(count))
                .unwrap_err()
                .view(),
            SciOxideErrorView::SizeOverflow { .. }
        ));
    }
}

#[test]
fn scheduled_discontinuity_recomputes_the_base_derivative() {
    let at_break = Cell::new(0);
    let history = |_: f64, output: &mut [f64]| {
        output.fill(1.0);
        Ok(())
    };
    let rhs = |time: f64, _: &[f64], _: MatrixView<'_, f64>, output: &mut [f64]| {
        if time == 0.5 {
            at_break.set(at_break.get() + 1);
        }
        output.fill(1.0);
        Ok(())
    };
    let plan = plan(1.0, &[0.5], 32, options(0.25, 1e-8).with_maximum_step(0.25));
    let solution = plan.solve(&DdeProblem::new(&rhs, &history)).unwrap();
    assert!(solution.report().is_success());
    // DP5 evaluates two endpoint stages before the break and a fresh base after it.
    assert_eq!(at_break.get(), 3);
    assert!((solution.state().values()[0] - 2.0).abs() < 1e-12);
}
