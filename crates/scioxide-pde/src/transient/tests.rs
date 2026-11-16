//! Lifecycle tests of transient plans: resume validation, preflight and phases.
#![allow(clippy::unwrap_used)]

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{AttemptOutcome, SciOxideErrorView};

use super::*;
use crate::FieldPoint;
use crate::fdm::StructuredFdmPlan;
use crate::support::{Discretization, End, Ends, Heat, HyperbolicPde, ParabolicPde, Plan1d};

const ARGUMENT: scioxide_core::Argument = scioxide_core::argument!("fixture");

/// The first heat and wave mode `sin(πx)` on `[0, 1]`.
#[derive(Clone)]
struct Mode;
impl ParabolicPde<f64> for Mode {
    fn diffusion(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((core::f64::consts::PI * x[0]).sin())
    }
}
impl HyperbolicPde<f64> for Mode {
    fn wave_speed(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        <Self as ParabolicPde<f64>>::initial_condition(self, x)
    }
    fn initial_velocity(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}

/// Unit-diffusion FDM plan of `nodes` nodes on `[0, 1]` with Dirichlet `ends`.
fn fdm_with(nodes: usize, ends: &Ends<f64>) -> StructuredFdmPlan<f64, f64, 1> {
    let mesh = crate::mesh_fixture::uniform(0.0, 1.0, nodes);
    match Discretization::Fdm.plan(&mesh, &Heat(Mode), ends).unwrap() {
        Plan1d::Fdm(plan) => plan,
        _ => unreachable!(),
    }
}

/// Homogeneous Dirichlet FDM plan; every call is a new mesh revision.
fn fdm(nodes: usize) -> StructuredFdmPlan<f64, f64, 1> {
    fdm_with(nodes, &crate::support::homogeneous_dirichlet())
}

/// `make()` on the heap; the temporary lives in this short frame only. Workspaces
/// hold the ODE workspace inline, and unoptimized test frames must stay small beside
/// the integrator's own frames.
fn boxed<T>(make: impl FnOnce() -> T) -> Box<T> {
    Box::new(make())
}

fn initial() -> FieldData<f64, f64, 1> {
    crate::support::initial(&Mode)
}

/// Zero source data counting its evaluations in `count`.
fn counting_source(count: &Arc<AtomicUsize>) -> FieldData<f64, f64, 1> {
    let count = Arc::clone(count);
    FieldData::time_dependent(move |_: FieldPoint<f64, 1>, _| {
        count.fetch_add(1, Ordering::Relaxed);
        Ok(0.0)
    })
}

fn backward_euler() -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::BackwardEuler)
        .with_step(0.01)
}

fn crank_nicolson(step: f64) -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::CrankNicolson)
        .with_step(step)
}

/// Crank--Nicolson with one interval of backward-Euler startup.
fn rannacher(step: f64) -> ParabolicOptions<f64> {
    crank_nicolson(step).with_rannacher_startup(Budget::ONE)
}

fn dopri() -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.005)
}

/// Every observable part of a state except its mesh token.
fn snapshot(state: &PdeState<f64>) -> (f64, Vec<f64>, Option<f64>, MethodOfLinesOrder) {
    (
        state.time(),
        state.coordinates().to_vec(),
        state.startup_end(),
        state.order(),
    )
}

fn assert_invalid_input(error: &SciOxideError) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { .. }),
        "{error}"
    );
}

/// The resume or advance attempt run by `attempt` must be a preflight rejection of
/// `kind`: an invalid-input error, no sink call, no source evaluation and neither
/// `state` nor `checkpoint` changed.
fn assert_rejected(
    kind: AttemptKind,
    evaluations: &AtomicUsize,
    checkpoint: Option<&PdeCheckpoint<f64>>,
    state: &mut PdeState<f64>,
    attempt: impl FnOnce(
        &mut PdeState<f64>,
        &mut dyn PdeOutputSink<f64>,
        &mut PdeWorkspace<f64>,
    ) -> Result<PdeSolveReport<f64>, SciOxideError>,
) {
    let saved_checkpoint = checkpoint.map(|c| snapshot(c.state()));
    let saved_state = snapshot(state);
    let token = state.mesh_token().clone();
    let before = evaluations.load(Ordering::Relaxed);
    let mut calls = 0;
    let mut workspace = Box::<PdeWorkspace<f64>>::default();
    let error = attempt(
        state,
        &mut |_: PdeObservation<'_, f64, f64>| {
            calls += 1;
            Ok(())
        },
        &mut workspace,
    )
    .unwrap_err();
    assert_invalid_input(&error);
    let record = workspace.last_attempt().unwrap();
    assert_eq!(record.kind(), kind);
    assert_eq!(record.outcome(), AttemptOutcome::PreflightRejected);
    assert!(record.epoch().is_none() && record.progress().is_none());
    assert!(workspace.last_accepted().is_none());
    assert_eq!(calls, 0);
    assert_eq!(evaluations.load(Ordering::Relaxed), before);
    assert_eq!(snapshot(state), saved_state);
    assert!(state.mesh_token().is_compatible_with(&token));
    assert_eq!(checkpoint.map(|c| snapshot(c.state())), saved_checkpoint);
}

#[test]
fn rejected_resumes_record_a_resume_and_change_nothing() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let source = counting_source(&evaluations);
    let initial = initial();
    let plan = fdm(9);
    let cx = ExecutionContext::sequential_native;

    // A completed backward-Euler checkpoint at t = 0.03.
    let transient = backward_euler();
    let problem = ParabolicProblem::new(&plan, &source, 0.03).unwrap();
    let mut state = problem.initial_state(&initial, &transient).unwrap();
    let mut workspace = boxed(|| problem.workspace().unwrap());
    let report = problem
        .advance(&transient, &mut state, &mut (), &mut workspace, cx())
        .unwrap();
    assert!(report.is_success());
    let checkpoint = PdeCheckpoint::new(&state).unwrap();
    assert_eq!(checkpoint.time(), 0.03);
    assert!(evaluations.load(Ordering::Relaxed) > 0);

    // Another mesh revision of an identical plan.
    let twin = fdm(9);
    let other = ParabolicProblem::new(&twin, &source, 0.05).unwrap();
    let mut target = other.initial_state(&initial, &transient).unwrap();
    assert_rejected(
        AttemptKind::Resume,
        &evaluations,
        Some(&checkpoint),
        &mut target,
        |state, sink, ws| other.resume(&transient, &checkpoint, state, sink, ws, cx()),
    );

    // A first-order checkpoint into a second-order plan on the same mesh.
    let wave = HyperbolicOptions::new().with_initial_step(0.01);
    let wave_problem = HyperbolicProblem::new(&plan, &source, 0.05).unwrap();
    let mut target = wave_problem
        .initial_state(&initial, &FieldData::zero())
        .unwrap();
    assert_rejected(
        AttemptKind::Resume,
        &evaluations,
        Some(&checkpoint),
        &mut target,
        |state, sink, ws| wave_problem.resume(&wave, &checkpoint, state, sink, ws, cx()),
    );

    let later = ParabolicProblem::new(&plan, &source, 0.05).unwrap();
    // A checkpoint beyond the end of the problem.
    let short = ParabolicProblem::new(&plan, &source, 0.02).unwrap();
    let mut target = short.initial_state(&initial, &transient).unwrap();
    assert_rejected(
        AttemptKind::Resume,
        &evaluations,
        Some(&checkpoint),
        &mut target,
        |state, sink, ws| short.resume(&transient, &checkpoint, state, sink, ws, cx()),
    );

    // A checkpoint inside Rannacher startup (h = 0.02, startup to 0.02) stopped by
    // its step budget after one half step.
    let startup = rannacher(0.02).with_step_budget(Budget::ONE);
    let mut state = later.initial_state(&initial, &startup).unwrap();
    let report = later
        .advance(&startup, &mut state, &mut (), &mut workspace, cx())
        .unwrap();
    assert_eq!(report.termination(), PdeTermination::TimeStepLimit);
    let pending = PdeCheckpoint::new(&state).unwrap();
    assert_eq!(pending.time(), 0.01);
    assert_eq!(pending.state().startup_end(), Some(0.02));
    // Resumed by a plan with another step (startup to 0.01) or without startup.
    for options in [rannacher(0.01), crank_nicolson(0.02)] {
        let mismatched = options;
        let mut target = later.initial_state(&initial, &mismatched).unwrap();
        assert_rejected(
            AttemptKind::Resume,
            &evaluations,
            Some(&pending),
            &mut target,
            |state, sink, ws| later.resume(&mismatched, &pending, state, sink, ws, cx()),
        );
    }
    // A state prepared without startup is rejected by a startup plan while the
    // startup interval lies ahead, and accepted once it has passed.
    let mut plain = later
        .initial_state(&initial, &crank_nicolson(0.02))
        .unwrap();
    assert_rejected(
        AttemptKind::Call,
        &evaluations,
        None,
        &mut plain,
        |state, sink, ws| later.advance(&rannacher(0.02), state, sink, ws, cx()),
    );
    let first_step = crank_nicolson(0.02).with_step_budget(Budget::ONE);
    let report = later
        .advance(&first_step, &mut plain, &mut (), &mut workspace, cx())
        .unwrap();
    assert_eq!(report.termination(), PdeTermination::TimeStepLimit);
    assert_eq!(plain.time(), 0.02);
    let report = later
        .advance(&rannacher(0.02), &mut plain, &mut (), &mut workspace, cx())
        .unwrap();
    assert!(report.is_success() && plain.time() == 0.05);
    // The matching plan resumes the startup phase and completes.
    let matching = rannacher(0.02);
    let mut target = later.initial_state(&initial, &matching).unwrap();
    let mut resumed = Box::<PdeWorkspace<f64>>::default();
    let report = later
        .resume(
            &matching,
            &pending,
            &mut target,
            &mut (),
            &mut resumed,
            cx(),
        )
        .unwrap();
    assert!(report.is_success() && target.time() == 0.05);
    let record = resumed.last_attempt().unwrap();
    assert_eq!(
        (record.kind(), record.outcome()),
        (AttemptKind::Resume, AttemptOutcome::Succeeded)
    );
}

#[test]
fn rejected_advances_record_a_call_and_change_nothing() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let source = counting_source(&evaluations);
    let initial = initial();
    let (plan, twin) = (fdm(9), fdm(9));
    let problem = ParabolicProblem::new(&plan, &source, 0.02).unwrap();
    let cx = ExecutionContext::sequential_native;

    // A requested time beyond the end of the problem.
    let late = dopri().with_output_schedule(PdeOutputSchedule::Times(vec![0.01, 0.03]));
    let mut state = problem.initial_state(&initial, &late).unwrap();
    assert_rejected(
        AttemptKind::Call,
        &evaluations,
        None,
        &mut state,
        |state, sink, ws| problem.advance(&late, state, sink, ws, cx()),
    );

    // A state of another mesh revision.
    let transient = backward_euler();
    let other = ParabolicProblem::new(&twin, &source, 0.02).unwrap();
    let mut state = other.initial_state(&initial, &transient).unwrap();
    assert_rejected(
        AttemptKind::Call,
        &evaluations,
        None,
        &mut state,
        |state, sink, ws| problem.advance(&transient, state, sink, ws, cx()),
    );
    assert_eq!(evaluations.load(Ordering::Relaxed), 0);
}

#[test]
fn failure_after_accepted_steps_keeps_the_last_accepted_point() {
    let plan = fdm(9);
    let (source, initial) = (FieldData::zero(), initial());
    let problem = ParabolicProblem::new(&plan, &source, 0.04).unwrap();
    let transient = backward_euler().with_output_schedule(PdeOutputSchedule::TimeSteps);
    let mut state = problem.initial_state(&initial, &transient).unwrap();
    let mut workspace = boxed(|| problem.workspace().unwrap());
    // The sink rejects the fourth sample, at t = 0.03: the step to 0.03 is not accepted.
    let mut times = Vec::new();
    let error = problem
        .advance(
            &transient,
            &mut state,
            &mut |sample: PdeObservation<'_, f64, f64>| {
                times.push(sample.time());
                if times.len() == 4 {
                    Err(crate::validation::invalid(ARGUMENT, "sink stop"))
                } else {
                    Ok(())
                }
            },
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert_invalid_input(&error);
    assert_eq!(times, [0.0, 0.01, 0.02, 0.03]);
    assert_eq!(state.time(), 0.02);
    let record = workspace.last_attempt().unwrap();
    assert_eq!(
        (record.kind(), record.outcome()),
        (AttemptKind::Call, AttemptOutcome::Failed)
    );
    let progress = record.progress().unwrap();
    assert_eq!(progress.time(), 0.02);
    assert_eq!(progress.state().coordinates(), state.coordinates());
    assert_eq!(progress.steps().accepted(), 2);
    assert_eq!(workspace.last_accepted().unwrap().time(), 0.02);

    // The owned form returns the same progress with its failure; its internal
    // trajectory cannot fail, so a source rejecting t > 0.02 stops the same step.
    let failing = FieldData::time_dependent(|_: FieldPoint<f64, 1>, t: f64| {
        if t > 0.025 {
            Err(crate::validation::invalid(ARGUMENT, "source stop"))
        } else {
            Ok(0.0)
        }
    });
    let failure = ParabolicProblem::new(&plan, &failing, 0.04)
        .unwrap()
        .solve_with_options(&initial, &transient)
        .unwrap_err();
    assert_invalid_input(failure.error());
    let owned = failure.progress().unwrap();
    assert_eq!(owned.time(), progress.time());
    assert_eq!(owned.state().coordinates(), progress.state().coordinates());
    assert_eq!(owned.steps().accepted(), progress.steps().accepted());
}

#[test]
fn continuation_to_a_later_end_continues_from_the_state() {
    let plan = fdm(9);
    let (source, initial) = (FieldData::zero(), initial());
    let first = ParabolicProblem::new(&plan, &source, 0.04).unwrap();
    let later = ParabolicProblem::new(&plan, &source, 0.05).unwrap();
    let transient = rannacher(0.02).with_output_schedule(PdeOutputSchedule::TimeSteps);
    let cx = ExecutionContext::sequential_native;
    let mut state = first.initial_state(&initial, &transient).unwrap();
    let mut workspace = boxed(|| first.workspace().unwrap());
    let mut trajectory = PdeTrajectory::new();
    let report = first
        .advance(
            &transient,
            &mut state,
            &mut trajectory,
            &mut workspace,
            cx(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(trajectory.times(), [0.0, 0.01, 0.02, 0.04]);
    assert_eq!((state.time(), state.startup_end()), (0.04, None));

    // Advancing a state already at the end completes without a step or a sample.
    let mut idle = PdeTrajectory::new();
    let report = first
        .advance(&transient, &mut state, &mut idle, &mut workspace, cx())
        .unwrap();
    assert!(report.is_success());
    assert_eq!(report.diagnostics().num_time_steps(), 0);
    assert!(idle.times().is_empty());
    assert_eq!(state.time(), 0.04);

    // A later end continues with the main method only.
    let mut continued = PdeTrajectory::new();
    let report = later
        .advance(&transient, &mut state, &mut continued, &mut workspace, cx())
        .unwrap();
    assert!(report.is_success());
    assert_eq!(report.final_time(), 0.05);
    assert_eq!(report.diagnostics().num_accepted_steps(), 1);
    assert_eq!(continued.times(), [0.04, 0.05]);
    // The steps match one advance to the later end.
    let single = later.solve_with_options(&initial, &transient).unwrap();
    assert_eq!(single.trajectory().times(), [0.0, 0.01, 0.02, 0.04, 0.05]);
    for (a, b) in state.coordinates().iter().zip(single.state().coordinates()) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} {b}");
    }
    for (a, b) in continued.states()[1].iter().zip(single.values()) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} {b}");
    }
}

#[test]
fn continued_requested_times_keep_global_indices_and_duplicates() {
    let plan = fdm(9);
    let (source, initial) = (FieldData::zero(), initial());
    let cx = ExecutionContext::sequential_native;
    let early = dopri().with_output_schedule(PdeOutputSchedule::Times(vec![0.0, 0.01, 0.01]));
    let first = ParabolicProblem::new(&plan, &source, 0.02).unwrap();
    let mut state = first.initial_state(&initial, &early).unwrap();
    let mut workspace = boxed(|| first.workspace().unwrap());
    let mut seen = Vec::new();
    let report = first
        .advance(
            &early,
            &mut state,
            &mut |sample: PdeObservation<'_, f64, f64>| {
                seen.push((sample.request_index(), sample.time()));
                Ok(())
            },
            &mut workspace,
            cx(),
        )
        .unwrap();
    assert!(report.is_success() && state.time() == 0.02);
    assert_eq!(seen, [(Some(0), 0.0), (Some(1), 0.01), (Some(2), 0.01)]);

    // The full request list skips the times before the state and keeps indices.
    let full = dopri().with_output_schedule(PdeOutputSchedule::Times(vec![
        0.0, 0.01, 0.01, 0.03, 0.03, 0.05,
    ]));
    let later = ParabolicProblem::new(&plan, &source, 0.05).unwrap();
    let mut seen = Vec::new();
    let report = later
        .advance(
            &full,
            &mut state,
            &mut |sample: PdeObservation<'_, f64, f64>| {
                assert_eq!(sample.values().len(), 9);
                seen.push((sample.request_index(), sample.time()));
                Ok(())
            },
            &mut workspace,
            cx(),
        )
        .unwrap();
    assert!(report.is_success() && state.time() == 0.05);
    assert_eq!(seen, [(Some(3), 0.03), (Some(4), 0.03), (Some(5), 0.05)]);
}

#[test]
fn fully_constrained_problems_emit_boundary_data_in_order() {
    // Both nodes carry u = x (1 + t): the field is [0, 1 + t] at every time.
    let moving = || End::dirichlet(crate::support::moving(|x, t| Ok(x * (1.0 + t))));
    let plan = fdm_with(2, &Ends::new(moving(), moving()));
    let source = FieldData::zero();
    let problem = ParabolicProblem::new(&plan, &source, 0.02).unwrap();
    for (schedule, expected) in [
        (
            PdeOutputSchedule::Times(vec![0.0, 0.01, 0.01, 0.02]),
            vec![
                (Some(0), 0.0),
                (Some(1), 0.01),
                (Some(2), 0.01),
                (Some(3), 0.02),
            ],
        ),
        (
            PdeOutputSchedule::TimeSteps,
            vec![(None, 0.0), (None, 0.02)],
        ),
        (PdeOutputSchedule::FinalOnly, vec![(None, 0.02)]),
    ] {
        let transient = backward_euler().with_output_schedule(schedule);
        let mut state = problem
            .initial_state(&FieldData::zero(), &transient)
            .unwrap();
        assert!(state.coordinates().is_empty());
        let mut seen = Vec::new();
        let report = problem
            .advance(
                &transient,
                &mut state,
                &mut |sample: PdeObservation<'_, f64, f64>| {
                    let t = sample.time();
                    assert_eq!(sample.values(), [0.0, 1.0 + t]);
                    seen.push((sample.request_index(), t));
                    Ok(())
                },
                &mut Box::<PdeWorkspace<f64>>::default(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(report.final_time(), 0.02);
        assert_eq!(report.diagnostics().num_time_steps(), 0);
        assert_eq!(state.time(), 0.02);
        assert_eq!(seen, expected);
    }
}

#[test]
fn one_shot_engine_preserves_progress_on_unsuccessful_termination() {
    let fdm = fdm(9);
    let source = FieldData::zero();
    let problem = PreparedProblem::new(&fdm, &source, 0.1).unwrap();
    let options = backward_euler().with_step_budget(Budget::ONE);
    let core = Transient {
        options: &options,
        order: MethodOfLinesOrder::First,
    };
    let failure = solve_owned(
        &core,
        &problem,
        core.initial_state(&problem, &initial(), None),
        true,
    )
    .unwrap_err();
    assert!(matches!(
        failure.error().view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit,
            ..
        }
    ));
    assert!((failure.progress().unwrap().state().time() - 0.01).abs() < 1e-15);
}
