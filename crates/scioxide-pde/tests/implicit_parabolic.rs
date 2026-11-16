#![allow(clippy::unwrap_used)]

#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use std::f64::consts::PI;
use std::sync::Arc;

use mesh_fixture::uniform;
use scioxide_core::{AttemptOutcome, CancellationToken, ExecutionContext, SciOxideError};
use scioxide_pde::{
    FieldData, FieldPoint, ParabolicOptions, ParabolicProblem, ParabolicScheme, PdeOutputSchedule,
    PdeOutputSink, PdeSolveReport, PdeState, PdeTermination, PdeTrajectory, PdeWorkspace,
};
use support::{Discretization, Heat, ParabolicPde, Plan1d, homogeneous_dirichlet, source};

/// `u_t = u_xx`, `u(x, 0) = sin(πx)`; optionally cancels beyond the first step.
#[derive(Clone, Default)]
struct SineHeat {
    cancellation: Option<Arc<CancellationToken>>,
}

impl ParabolicPde<f64> for SineHeat {
    fn diffusion(&self, _position: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }

    fn source(&self, _position: &[f64], time: f64) -> Result<f64, SciOxideError> {
        if let Some(cancellation) = &self.cancellation
            && time > 0.0001
        {
            cancellation.cancel();
        }
        Ok(0.0)
    }

    fn initial_condition(&self, position: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * position[0]).sin())
    }
}

/// One advance to `t = 0.1` on 81 FDM nodes; the state and workspace are
/// returned with the outcome so failures can be inspected.
fn run<Sink: PdeOutputSink<f64> + ?Sized>(
    options: ParabolicOptions<f64>,
    problem: &SineHeat,
    sink: &mut Sink,
    execution: ExecutionContext<'_>,
) -> (
    Result<PdeSolveReport<f64>, SciOxideError>,
    PdeState<f64>,
    PdeWorkspace<f64>,
) {
    let mesh = uniform(0.0, 1.0, 81);
    let heat = Heat(problem.clone());
    let plan = Discretization::Fdm
        .plan(&mesh, &heat, &homogeneous_dirichlet())
        .unwrap();
    let initial = FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok((PI * p.point[0]).sin()));
    let source = source(&heat);
    let Plan1d::Fdm(fdm) = &plan else {
        unreachable!("an FDM plan was requested")
    };
    let parabolic = options;
    let problem = ParabolicProblem::new(fdm, &source, 0.1).unwrap();
    let mut state = problem.initial_state(&initial, &parabolic).unwrap();
    let mut workspace = problem.workspace().unwrap();
    let outcome = problem.advance(&parabolic, &mut state, sink, &mut workspace, execution);
    (outcome, state, workspace)
}

fn options(scheme: ParabolicScheme) -> ParabolicOptions<f64> {
    ParabolicOptions::new().with_scheme(scheme).with_step(0.005)
}

#[test]
fn implicit_schemes_reach_the_requested_time_with_finite_states() {
    for scheme in [
        ParabolicScheme::BackwardEuler,
        ParabolicScheme::CrankNicolson,
    ] {
        let mut output = PdeTrajectory::new();
        let (report, state, _) = run(
            options(scheme),
            &SineHeat::default(),
            &mut output,
            ExecutionContext::sequential(),
        );
        let report = report.unwrap();
        assert_eq!(report.termination(), PdeTermination::Completed);
        assert_eq!(report.final_time(), 0.1);
        assert_eq!(state.time(), 0.1);
        assert_eq!(output.times(), [0.1]);
        assert!(output.states()[0].iter().all(|value| value.is_finite()));
    }
}

#[test]
fn accepted_step_trajectory_is_caller_selected() {
    let mut output = PdeTrajectory::new();
    let (report, _, _) = run(
        options(ParabolicScheme::CrankNicolson).with_output_schedule(PdeOutputSchedule::TimeSteps),
        &SineHeat::default(),
        &mut output,
        ExecutionContext::sequential(),
    );
    let report = report.unwrap();
    assert_eq!(output.times().first().copied(), Some(0.0));
    assert_eq!(output.times().last().copied(), Some(report.final_time()));
    assert_eq!(
        output.times().len(),
        report.diagnostics().num_accepted_steps() + 1
    );
}

#[test]
fn injected_cancellation_returns_the_initial_partial_state() {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
    let mut output = PdeTrajectory::new();
    let (outcome, state, workspace) = run(
        options(ParabolicScheme::BackwardEuler),
        &SineHeat::default(),
        &mut output,
        execution,
    );
    let error = outcome.unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    // No step was accepted: the attempt carries no progress, the state stays at
    // the initial point and nothing was emitted.
    assert!(workspace.last_accepted().is_none());
    assert_eq!(state.time(), 0.0);
    assert!(output.times().is_empty());
}

#[test]
fn cancellation_after_a_step_returns_the_last_accepted_state() {
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.0001);
    let cancellation = Arc::new(CancellationToken::new());
    let problem = SineHeat {
        cancellation: Some(Arc::clone(&cancellation)),
    };
    let execution = ExecutionContext::sequential().with_cancellation(&*cancellation);
    let mut output = PdeTrajectory::new();
    let (report, state, workspace) = run(options, &problem, &mut output, execution);
    let report = report.unwrap();

    assert!(matches!(
        report.termination(),
        PdeTermination::ExecutionStopped(_)
    ));
    assert!(report.final_time() > 0.0);
    assert!(report.final_time() < 0.1);
    assert_eq!(state.time(), report.final_time());
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    // Final-only output still emits the last accepted field.
    assert_eq!(output.times(), &[report.final_time()]);
    assert!(output.states()[0].iter().all(|value| value.is_finite()));
}

#[test]
fn output_failure_is_typed_and_retains_the_completed_state() {
    let mut sink = |_observation: scioxide_pde::PdeObservation<'_, f64, f64>| {
        Err(SciOxideError::allocation_failure(
            scioxide_core::quantity!("injected_pde_output"),
            1,
        ))
    };
    let (outcome, state, workspace) = run(
        options(ParabolicScheme::BackwardEuler),
        &SineHeat::default(),
        &mut sink,
        ExecutionContext::sequential(),
    );
    assert!(matches!(
        outcome.unwrap_err().view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().time(), 0.1);
    assert_eq!(state.time(), 0.1);
}
