//! Reusable workspace lifecycle and output-boundary regressions.
//!
//! The method workspace shape-failure recovery is covered by the crate's internal driver
//! tests.

#![allow(clippy::unwrap_used)]

use std::cell::Cell;

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, CancellationToken, Capacity, ErrorTolerance,
    ExecutionContext, NormSelection, SciOxideError, SciOxideErrorKind, SolveContext,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    ClassicalRk4, DormandPrince5, OdeFn, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem,
    OdeState, OdeSystem, OdeTermination, OdeTrajectory, OdeWorkspace,
};

struct UnitRate<'a> {
    calls: &'a Cell<usize>,
}

impl OdeSystem<f64> for UnitRate<'_> {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        derivative[0] = 1.0;
        Ok(())
    }
}

fn rk4(options: OdeOptions<f64>) -> (ClassicalRk4<f64>, OdeOptions<f64>) {
    (ClassicalRk4::new(), options)
}

#[test]
fn reused_workspace_serves_independent_epochs() {
    let calls = Cell::new(0);
    let system = UnitRate { calls: &calls };
    let plan = rk4(OdeOptions::new()
        .with_initial_step(0.25)
        .with_output_schedule(OdeOutputSchedule::FinalOnly));
    let problem = OdeProblem::new(&system, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();

    for initial in [0.0, 2.0] {
        let mut output = OdeTrajectory::new();
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![initial]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert!((state.values()[0] - (initial + 1.0)).abs() < 1e-12);
        assert_eq!(output.times(), [1.0]);
        assert_eq!(report.accepted_steps(), 4);
    }
    assert_eq!(calls.get(), 32);
}

#[test]
fn proposal_denial_leaves_the_iteration_counter_at_the_last_admitted_step() {
    for (proposals, final_time) in [(1, 0.25_f64), (2, 0.5)] {
        let calls = Cell::new(0);
        let system = UnitRate { calls: &calls };
        let plan = rk4(OdeOptions::new()
            .with_initial_step(0.25)
            .with_max_steps(Budget::try_from(proposals).unwrap()));
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&system, 1.0),
                &plan.1,
                OdeState::new(0.0, Vector::from_vec(vec![0.0])),
            )
            .unwrap();
        let report = solution.report();
        assert_eq!(
            report.termination(),
            Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
        );
        assert_eq!(report.counts().iterations(), proposals);
        assert_eq!(report.counts().proposals(), proposals);
        assert_eq!(report.counts().iterations(), proposals);
        assert_eq!(report.counts().proposals(), proposals);
        assert_eq!(calls.get(), 4 * proposals);
        assert!((report.final_time() - final_time).abs() <= f64::EPSILON);
        assert!((solution.state().values()[0] - final_time).abs() <= f64::EPSILON);
    }
}

#[test]
fn dimension_failure_precedes_callbacks() {
    let calls = Cell::new(0);
    let system = OdeFn::new(2, |_: f64, _: &[f64], dy: &mut [f64]| {
        calls.set(calls.get() + 1);
        dy.fill(1.0);
        Ok(())
    })
    .unwrap();
    let plan = (ClassicalRk4::new(), OdeOptions::new());
    let problem = OdeProblem::new(&system, 1.0);
    let mut workspace = OdeWorkspace::new();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
    let error = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut OdeTrajectory::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.progress().is_none());
    assert_eq!(calls.get(), 0);
    assert_eq!((state.time(), state.values()), (0.0, &[0.0][..]));
    let failure = plan
        .0
        .solve(
            &problem,
            &plan.1,
            OdeState::new(0.0, Vector::from_vec(vec![0.0])),
        )
        .unwrap_err();
    assert!(failure.progress().is_none());
    assert_eq!(calls.get(), 0);
}

#[test]
fn output_failure_retains_the_last_accepted_state() {
    let calls = Cell::new(0);
    let system = UnitRate { calls: &calls };
    let plan = rk4(OdeOptions::new().with_initial_step(0.25));
    let problem = OdeProblem::new(&system, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
    let mut sink = |_time: f64, _state: &[f64]| {
        Err(SciOxideError::allocation_failure(
            scioxide_core::quantity!("injected_output"),
            1,
        ))
    };
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
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
    assert_eq!(state.time(), 1.0);
    assert!((state.values()[0] - 1.0).abs() < 1e-12);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert_eq!(progress.time(), 1.0);
    assert!((progress.state()[0] - 1.0).abs() < 1e-12);
}

struct CancellingRate<'a> {
    cancellation: &'a CancellationToken,
}

impl OdeSystem<f64> for CancellingRate<'_> {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = 1.0;
        self.cancellation.cancel();
        Ok(())
    }
}

#[test]
fn cancellation_returns_the_last_accepted_state() {
    let cancellation = CancellationToken::new();
    let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
    let plan = rk4(OdeOptions::new().with_initial_step(0.25));
    let system = CancellingRate {
        cancellation: &cancellation,
    };
    let problem = OdeProblem::new(&system, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
    let mut output = OdeTrajectory::new();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut output,
            &mut workspace,
            execution,
        )
        .unwrap();

    assert!(matches!(
        report.termination(),
        Some(OdeTermination::ExecutionStopped(_))
    ));
    assert_eq!(report.final_time(), 0.25);
    assert_eq!(state.time(), 0.25);
    assert!((state.values()[0] - 0.25).abs() < 1e-12);
    assert_eq!(output.times(), &[0.25]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
}

struct FailingRate<'a> {
    calls: &'a Cell<usize>,
}

impl OdeSystem<f64> for FailingRate<'_> {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        if calls >= 4 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("derivative"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "callback failed after one accepted step",
            ));
        }
        derivative[0] = 1.0;
        Ok(())
    }
}

#[test]
fn callback_failure_returns_the_last_accepted_state() {
    let calls = Cell::new(0);
    let system = FailingRate { calls: &calls };
    let plan = rk4(OdeOptions::new().with_initial_step(0.25));
    let failure = plan
        .0
        .solve(
            &OdeProblem::new(&system, 1.0),
            &plan.1,
            OdeState::new(0.0, Vector::from_vec(vec![0.0])),
        )
        .unwrap_err();

    assert!(matches!(
        failure.error().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let partial = failure.progress().unwrap();
    assert_eq!(partial.time(), 0.25);
    assert!((partial.state()[0] - 0.25).abs() < 1e-12);
}

#[test]
fn prepared_workspace_satisfies_the_plan_requirements() {
    let calls = Cell::new(0);
    let system = UnitRate { calls: &calls };
    let plan = rk4(OdeOptions::new());
    let problem = OdeProblem::new(&system, 1.0);
    let requirements = plan.0.workspace_requirements(&problem, &plan.1).unwrap();
    assert_eq!(requirements.dimension(), 1);
    assert_eq!(
        OdeWorkspace::new().check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    assert!(workspace.last_attempt().is_none());
    assert_eq!(calls.get(), 0);
}

#[test]
fn closure_nonfinite_trial_output_recovers_but_accepted_output_fails() {
    let calls = Cell::new(0);
    let system = OdeFn::new(1, |_: f64, state: &[f64], output: &mut [f64]| {
        calls.set(calls.get() + 1);
        output[0] = if state[0] < 0.0 {
            f64::INFINITY
        } else {
            -state[0]
        };
        Ok(())
    })
    .unwrap();
    let plan = (
        DormandPrince5::new(),
        OdeOptions::new()
            .with_initial_step(4.0)
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity).unwrap(),
            ),
    );
    let problem = OdeProblem::new(&system, 4.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
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
    assert!(report.counts().rejected() > 0);
    assert!((state.values()[0] - (-4.0_f64).exp()).abs() < 1e-5);
    calls.set(0);
    state.reset(0.0, &[-1.0]).unwrap();
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
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    assert_eq!(calls.get(), 1);
    assert_eq!(state.values(), &[-1.0]);
}
