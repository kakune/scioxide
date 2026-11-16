//! End-to-end outcome classification for both adjoint families.
#![allow(clippy::unwrap_used)]
use super::Policies as AdjointOptions;
use super::dae::PreparedDaeAdjoint as DaeAdjointSolver;
use super::ode::PreparedOdeAdjoint as OdeAdjointSolver;
use super::*;
use crate::OdeMethod;
use crate::{DaeOptions, DaePartition, OdeStageBackend, OdeSystem};
use scioxide_ad::CheckpointSchedule;
use scioxide_core::{
    AttemptOutcome, CancellationToken, ConvergenceReason, ExecutionContext, RelativeTolerance,
    SciOxideErrorKind, SciOxideErrorView,
};
/// The default transposed linear-equation defect of `DaeAdjointOptions`.
fn transpose_defect<S: RealScalar>() -> RelativeTolerance<S> {
    RelativeTolerance::new(scalar::<S>(64.).unwrap() * S::EPSILON).unwrap()
}
struct Model<'a, S: RealScalar, const N: usize = 1> {
    p: [S; N],
    cancel: Option<&'a CancellationToken>,
}
impl<S: RealScalar + NumericScalar<Real = S>, const N: usize> OdeSystem<S> for Model<'_, S, N> {
    fn dimension(&self) -> usize {
        N
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        j: &mut scioxide_linalg::MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..N {
            for column in 0..N {
                j.set(
                    row,
                    column,
                    if row == column { self.p[row] } else { S::ZERO },
                )?;
            }
        }
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if let Some(token) = self.cancel {
            token.cancel();
        }
        for (out, (&p, &y)) in out.iter_mut().zip(self.p.iter().zip(y)) {
            *out = p * y;
        }
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, const N: usize> OdeAdjointSystem<S>
    for Model<'_, S, N>
{
    fn parameters(&self) -> &[S] {
        &self.p
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: S,
        y: &[S],
        seed: &[S],
        state: &mut [S],
        parameters: &mut [S],
        _: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for i in 0..N {
            state[i] = self.p[i] * seed[i];
            parameters[i] = y[i] * seed[i];
        }
        Ok(())
    }
}
struct Energy(bool);
impl<S: RealScalar + NumericScalar<Real = S>> AdjointObjective<S> for Energy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: S,
        y: &[S],
        _: &[S],
        state: &mut [S],
        parameters: &mut [S],
        _: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<S>, SciOxideError> {
        let terminal = part == AdjointObjectivePart::Terminal || self.0;
        for (out, &y) in state.iter_mut().zip(y) {
            *out = if terminal { y } else { S::ZERO };
        }
        parameters.fill(S::ZERO);
        Ok(AdjointObjectiveValue {
            value: if terminal {
                y.iter().fold(S::ZERO, |sum, &y| sum + y * y * S::HALF)
            } else {
                S::ZERO
            },
            time_derivative: S::ZERO,
        })
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Case {
    Accepted,
    Reverse,
    ReversePartial,
    ReverseRejection,
    Underflow,
    ProjectionCancelled,
    ProjectionIteration,
    Evaluation,
}
// Parent proposal budgets are scope-local and never reach the adjoint's own epoch, so
// proposal denial has no configuration to exercise here.
fn outcomes<S: RealScalar + NumericScalar<Real = S>>() {
    for dae in [false, true] {
        for case in [
            Case::Accepted,
            Case::Reverse,
            Case::ReversePartial,
            Case::ReverseRejection,
            Case::Underflow,
            Case::ProjectionCancelled,
            Case::ProjectionIteration,
            Case::Evaluation,
        ] {
            if !dae && matches!(case, Case::ProjectionCancelled | Case::ProjectionIteration) {
                continue;
            }
            let cancel = CancellationToken::new();
            let model = Model {
                p: [scalar::<S>(-0.7).unwrap()],
                cancel: (case == Case::ProjectionCancelled).then_some(&cancel),
            };
            let mode = if matches!(
                case,
                Case::Reverse | Case::ReversePartial | Case::ReverseRejection
            ) {
                AdjointMode::Continuous
            } else {
                AdjointMode::Discrete
            };
            let mut options = AdjointOptions::new(mode, scalar::<S>(0.125).unwrap())
                .unwrap()
                .with_fixed_step(true)
                .with_tape_capacity(if case == Case::Accepted { 2 } else { 8 })
                .with_max_replay_steps(1000)
                .with_max_reverse_steps(match case {
                    Case::Reverse => 0,
                    Case::ReversePartial => 2,
                    Case::ReverseRejection => 1,
                    _ => 100000,
                })
                .with_max_workspace_bytes(16 * 1024 * 1024);
            if matches!(case, Case::ReversePartial | Case::ReverseRejection) {
                let tolerance = if case == Case::ReversePartial {
                    S::ONE
                } else {
                    S::EPSILON * S::EPSILON
                };
                let error =
                    ErrorTolerance::scalar(tolerance, tolerance, NormSelection::Infinity).unwrap();
                options = options
                    .with_adjoint_error(error.clone())
                    .with_quadrature_error(error);
            }
            let interval = if case == Case::Underflow {
                options =
                    options.with_initial_step(S::EPSILON / S::try_from_f64_lossy(4.).unwrap());
                (S::ONE, S::TWO)
            } else {
                (S::ZERO, S::HALF)
            };
            if case == Case::Evaluation {
                options = options.with_max_evaluations(Budget::ONE);
            }
            let scope = ExecutionContext::sequential().with_cancellation(&cancel);
            let mut gy = [S::TEN];
            let mut gp = [S::TEN];
            let (error, attempt) = if dae {
                let residual = OdeAdjointResidual::<S, _>::new(&model).unwrap();
                let mut dae_options = crate::DaeNumericalOptions::new();
                let mut derivative = model.p;
                if case == Case::ProjectionIteration {
                    dae_options = dae_options
                        .with_initialization(crate::DaeInitialization::Project)
                        .with_newton_limit(Budget::ONE);
                    derivative[0] = S::ZERO;
                }
                let plan = DaeAdjointSolver::new(
                    dae_shape(&residual, &Energy(false)),
                    DaePartition::new(&[true], &[true]).unwrap(),
                    OdeStageBackend::Dense,
                    dae_options,
                    options,
                    transpose_defect(),
                )
                .unwrap();
                let mut workspace = plan.workspace().unwrap();
                let error = plan
                    .solve_into(
                        &residual,
                        &Energy(false),
                        &[S::ONE],
                        &derivative,
                        interval,
                        &mut gy,
                        &mut gp,
                        &mut workspace,
                        scope,
                    )
                    .unwrap_err();
                (error, workspace.last_attempt().cloned().unwrap())
            } else {
                let plan =
                    OdeAdjointSolver::new(ode_shape(&model, &Energy(false)), options).unwrap();
                let mut workspace = plan.workspace().unwrap();
                let error = plan
                    .solve_into(
                        &model,
                        &Energy(false),
                        &[S::ONE],
                        interval,
                        None,
                        &mut gy,
                        &mut gp,
                        &mut workspace,
                        scope,
                    )
                    .unwrap_err();
                (error, workspace.last_attempt().cloned().unwrap())
            };
            let (termination, reason) = match case {
                Case::Accepted
                | Case::Reverse
                | Case::ReversePartial
                | Case::ReverseRejection
                | Case::ProjectionIteration => (
                    AdjointTermination::BudgetLimit(BudgetKind::Iteration),
                    Some(ConvergenceReason::IterationLimit),
                ),
                Case::Underflow => (
                    AdjointTermination::StepSizeLimit,
                    Some(ConvergenceReason::StepSizeLimit),
                ),
                Case::ProjectionCancelled => (
                    AdjointTermination::ExecutionStopped(
                        scioxide_core::ExecutionStopReason::Cancelled,
                    ),
                    None,
                ),
                Case::Evaluation => (
                    AdjointTermination::BudgetLimit(BudgetKind::Evaluation),
                    Some(ConvergenceReason::EvaluationLimit),
                ),
            };
            assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
            let progress = attempt.progress().unwrap();
            if case == Case::ReversePartial {
                assert_eq!(progress.reverse_steps(), 2);
            }
            if case == Case::ReverseRejection {
                assert_eq!(progress.reverse_steps(), 0);
                assert_eq!(progress.accounting().own().rejected(), 1);
            }
            assert_eq!(
                progress.termination(),
                Some(termination),
                "dae={dae} {case:?}: {error:?} {progress:?}"
            );
            assert!(!progress.is_success());
            if let Some(expected) = reason {
                assert_eq!(
                    error.kind(),
                    SciOxideErrorKind::ConvergenceFailure,
                    "dae={dae} {case:?}: {error:?}"
                );
                assert!(
                    matches!(error.view(),SciOxideErrorView::ConvergenceFailure {reason,..} if reason==expected),
                    "dae={dae} {case:?}: {error:?}"
                );
            } else {
                assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
            }
            assert_eq!(gy, [S::TEN]);
            assert_eq!(gp, [S::TEN]);
        }
    }
}
#[test]
fn f64_adjoint_outcomes() {
    outcomes::<f64>();
}
#[test]
fn f32_adjoint_outcomes() {
    outcomes::<f32>();
}

fn bdf_recurrence<S: RealScalar + NumericScalar<Real = S>>() {
    let model = Model {
        p: [scalar::<S>(-0.7).unwrap()],
        cancel: None,
    };
    let residual = OdeAdjointResidual::<S, _>::new(&model).unwrap();
    let options = AdjointOptions::new(AdjointMode::Discrete, scalar::<S>(0.125).unwrap())
        .unwrap()
        .with_tape_capacity(512)
        .with_max_replay_steps(1000000)
        .with_max_reverse_steps(100000)
        .with_max_workspace_bytes(16 * 1024 * 1024);
    let dae_options = DaeOptions::new();
    for running in [false, true] {
        let mut expected = None;
        for checkpoints in [
            CheckpointPolicy::StoreAll,
            CheckpointPolicy::Explicit(CheckpointSchedule::binomial(512, 2).unwrap()),
        ] {
            let plan = DaeAdjointSolver::new(
                dae_shape(&residual, &Energy(running)),
                DaePartition::new(&[true], &[true]).unwrap(),
                OdeStageBackend::Dense,
                (dae_options.clone()).numerics().clone(),
                options.clone().with_checkpoint_policy(checkpoints),
                transpose_defect(),
            )
            .unwrap();
            let mut workspace = plan.workspace().unwrap();
            let mut gy = [S::ZERO];
            let mut gp = [S::ZERO];
            let report = plan
                .solve_into(
                    &residual,
                    &Energy(running),
                    &[S::ONE],
                    &model.p,
                    (S::ZERO, S::HALF),
                    &mut gy,
                    &mut gp,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            let tape = &workspace.storage.as_ref().unwrap().tape;
            let mut states = alloc::vec![S::ONE];
            let mut tangent = alloc::vec![S::ZERO];
            let mut maximum = 1;
            let mut running_y = S::ZERO;
            let mut running_p = S::ZERO;
            for (index, step) in tape.steps.iter().take(report.accepted_steps).enumerate() {
                let geometry = step.metadata;
                let q = geometry.order;
                maximum = maximum.max(q);
                let mut alpha = [S::ZERO; 6];
                // Independently differentiate each Lagrange cardinal polynomial at x=1.
                alpha[0] = (1..=q).fold(S::ZERO, |sum, k| {
                    sum + S::ONE / (S::ONE - geometry.nodes[k])
                });
                for (j, coefficient) in alpha.iter_mut().enumerate().take(q + 1).skip(1) {
                    *coefficient = S::ONE / (geometry.nodes[j] - S::ONE);
                    for k in 1..=q {
                        if k != j {
                            *coefficient = *coefficient * (S::ONE - geometry.nodes[k])
                                / (geometry.nodes[j] - geometry.nodes[k]);
                        }
                    }
                }
                let denominator = alpha[0] - step.width * model.p[0];
                let y = -(1..=q).fold(S::ZERO, |sum, j| sum + alpha[j] * states[index + 1 - j])
                    / denominator;
                let dy = (step.width * y
                    - (1..=q).fold(S::ZERO, |sum, j| sum + alpha[j] * tangent[index + 1 - j]))
                    / denominator;
                states.push(y);
                tangent.push(dy);
                if running {
                    for j in 0..q {
                        let mut polynomial = [S::ZERO; 6];
                        polynomial[0] = S::ONE;
                        let mut degree = 0;
                        let mut denominator = S::ONE;
                        for k in 0..q {
                            if k != j {
                                for l in (1..=degree + 1).rev() {
                                    polynomial[l] =
                                        polynomial[l - 1] - geometry.nodes[k] * polynomial[l];
                                }
                                polynomial[0] = -geometry.nodes[k] * polynomial[0];
                                denominator *= geometry.nodes[j] - geometry.nodes[k];
                                degree += 1;
                            }
                        }
                        let weight = polynomial
                            .iter()
                            .enumerate()
                            .take(q)
                            .fold(S::ZERO, |sum, (k, &v)| {
                                sum + v / scalar::<S>((k + 1) as f64).unwrap()
                            })
                            / denominator
                            * step.width;
                        running_y += weight * states[index + 1 - j] * states[index + 1 - j];
                        running_p += weight * states[index + 1 - j] * tangent[index + 1 - j];
                    }
                }
            }
            assert!(maximum >= 3, "maximum order {maximum}");
            let y = states[report.accepted_steps];
            let dy = tangent[report.accepted_steps];
            let tolerance = scalar::<S>(512.).unwrap() * S::EPSILON;
            assert!(
                NumericScalar::abs(gy[0] - y * y - running_y) < tolerance,
                "{report:?}, gy={gy:?}, y={y:?}"
            );
            assert!(
                NumericScalar::abs(gp[0] - y * dy - running_p) < tolerance,
                "{report:?}, gp={gp:?}, dy={dy:?}"
            );
            if let Some((old_y, old_p)) = expected {
                assert_eq!(gy[0], old_y);
                assert_eq!(gp[0], old_p);
            } else {
                expected = Some((gy[0], gp[0]));
            }
            if !running && plan.tape.schedule.is_none() {
                let forward = (
                    crate::DaeBdf::new(
                        DaePartition::new(&[true], &[true]).unwrap(),
                        OdeStageBackend::Dense,
                    )
                    .unwrap(),
                    dae_options
                        .clone()
                        .with_initial_step(options.initial_step()),
                );
                let mut state = crate::DaeState::from_slices(S::ZERO, &[S::ONE], &model.p).unwrap();
                let mut forward_workspace = forward.0.workspace().unwrap();
                let forward_report = forward
                    .0
                    .advance(
                        &crate::DaeProblem::new(&residual, S::HALF),
                        &forward.1,
                        &mut state,
                        &mut (),
                        &mut forward_workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(forward_report.is_success());
                // Ordinary solves retain cross-step chord factors; recording
                // starts each proposal fresh for exact checkpoint replay. Their
                // meshes and rounding may differ within the configured accuracy.
                let recorded = tape.frames[(report.accepted_steps - 1) * plan.tape.frame_len + 1];
                let threshold = dae_options
                    .numerics()
                    .differential_error()
                    .threshold(0, 1, NumericScalar::abs(recorded))
                    .unwrap();
                assert!(NumericScalar::abs(state.values()[0] - recorded) <= S::TEN * threshold);
            }
        }
    }
}

#[test]
fn f64_multistep_adjoint_matches_independent_recurrence_and_forward_solver() {
    bdf_recurrence::<f64>();
}
#[test]
fn f32_multistep_adjoint_matches_independent_recurrence_and_forward_solver() {
    bdf_recurrence::<f32>();
}

#[test]
fn ode_checkpoint_workflow_still_fits_64_kib() {
    let model = Model {
        p: [-0.7_f64; 4],
        cancel: None,
    };
    let plan = OdeAdjointSolver::new(
        ode_shape(&model, &Energy(false)),
        AdjointOptions::new(AdjointMode::Discrete, 0.125)
            .unwrap()
            .with_tape_capacity(128)
            .with_max_replay_steps(20000)
            .with_max_reverse_steps(10000)
            .with_max_workspace_bytes(64 * 1024)
            .with_checkpoint_policy(CheckpointPolicy::MemoryBudget),
    )
    .unwrap();
    let result = plan
        .solve(&model, &Energy(false), &[1.; 4], (0., 0.5), None)
        .unwrap();
    for (&initial, &parameter) in result
        .initial_gradient()
        .iter()
        .zip(result.parameter_gradient())
    {
        assert!((initial - (-0.7_f64).exp()).abs() < 1e-7);
        assert!((parameter - 0.5 * (-0.7_f64).exp()).abs() < 1e-7);
    }
}

fn ode_forward_parity<S: RealScalar + NumericScalar<Real = S>>() {
    let model = Model {
        p: [
            scalar::<S>(-0.7).unwrap(),
            scalar::<S>(-3.).unwrap(),
            S::ONE,
            scalar::<S>(0.05).unwrap(),
        ],
        cancel: None,
    };
    for interval in [(S::ZERO, S::ONE), (S::ONE, S::ZERO)] {
        for norm in [
            scioxide_core::NormSelection::Infinity,
            scioxide_core::NormSelection::RootMeanSquare,
        ] {
            let options = AdjointOptions::new(AdjointMode::Discrete, S::HALF)
                .unwrap()
                .with_primal_error(
                    scioxide_core::ErrorTolerance::scalar(S::SQRT_EPSILON, S::SQRT_EPSILON, norm)
                        .unwrap(),
                )
                .with_tape_capacity(512)
                .with_max_replay_steps(10000)
                .with_max_reverse_steps(10000)
                .with_max_workspace_bytes(1024 * 1024);
            let plan =
                OdeAdjointSolver::new(ode_shape(&model, &Energy(false)), options.clone()).unwrap();
            let mut workspace = plan.workspace().unwrap();
            let report = plan
                .solve_into(
                    &model,
                    &Energy(false),
                    &[S::ONE; 4],
                    interval,
                    None,
                    &mut [S::ZERO; 4],
                    &mut [S::ZERO; 4],
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            let tape = &workspace.storage.as_ref().unwrap().tape;
            let forward = (
                crate::DormandPrince5::new(),
                crate::OdeOptions::new()
                    .with_initial_step(options.initial_step())
                    .with_error_tolerance(options.primal_error.clone())
                    .with_output_schedule(crate::OdeOutputSchedule::AcceptedSteps),
            );
            let problem = crate::OdeProblem::new(&model, interval.1);
            let mut forward_workspace = forward.0.workspace(&problem, &forward.1).unwrap();
            let mut state = crate::OdeState::new(
                interval.0,
                scioxide_linalg::Vector::from_vec(alloc::vec![S::ONE; 4]),
            );
            let mut index = 0;
            let forward_report = forward
                .0
                .advance(
                    &problem,
                    &forward.1,
                    &mut state,
                    &mut |time: S, state: &[S]| {
                        if index > 0 {
                            assert_eq!(time, tape.steps[index - 1].endpoint);
                            let expected = if index == report.accepted_steps {
                                &tape.final_state[..]
                            } else {
                                &tape.frames
                                    [index * plan.tape.frame_len..index * plan.tape.frame_len + 4]
                            };
                            for (&actual, &expected) in state.iter().zip(expected) {
                                assert!(
                                    NumericScalar::abs(actual - expected)
                                        <= scalar::<S>(4.).unwrap()
                                            * S::EPSILON
                                            * NumericScalar::abs(expected).max(S::ONE)
                                );
                            }
                        }
                        index += 1;
                        Ok::<(), SciOxideError>(())
                    },
                    &mut forward_workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(
                forward_report.termination(),
                Some(crate::OdeTermination::ReachedEnd)
            );
            assert_eq!(index - 1, report.accepted_steps);
        }
    }
}

#[test]
fn f64_dp5_recording_matches_forward_accepted_trajectory() {
    ode_forward_parity::<f64>();
}
#[test]
fn f32_dp5_recording_matches_forward_accepted_trajectory() {
    ode_forward_parity::<f32>();
}

struct CountedDiagonal {
    model: Model<'static, f64, 4>,
    pullbacks: core::cell::Cell<usize>,
}
impl OdeSystem<f64> for CountedDiagonal {
    fn dimension(&self) -> usize {
        4
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
impl OdeAdjointSystem<f64> for CountedDiagonal {
    fn parameters(&self) -> &[f64] {
        &self.model.p
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
        self.model.pullback(t, y, seed, state, parameters, work, cx)
    }
}

#[test]
fn continuous_dp5_pullback_work_and_gradient() {
    let model = CountedDiagonal {
        model: Model {
            p: [-0.7; 4],
            cancel: None,
        },
        pullbacks: core::cell::Cell::new(0),
    };
    let options = AdjointOptions::new(AdjointMode::Continuous, 0.01)
        .unwrap()
        .with_tape_capacity(1024)
        .with_max_reverse_steps(10000)
        .with_max_workspace_bytes(8 * 1024 * 1024);
    let plan = OdeAdjointSolver::new(ode_shape(&model, &Energy(false)), options).unwrap();
    let result = plan
        .solve(&model, &Energy(false), &[1.; 4], (0., 5.), None)
        .unwrap();
    // One initial derivative, then six pullbacks per accepted DP5 substep.
    let counts = result.report().accounting().own();
    let reverse_accepted = counts.accepted() - result.report().accepted_steps();
    assert_eq!(counts.rejected(), 0);
    assert_eq!(model.pullbacks.get(), 1 + 6 * reverse_accepted);
    // Reference gradients from the same recorded trajectory with uncached stages.
    for &value in result.initial_gradient() {
        assert!((value / 0.0009118821358350784 - 1.).abs() <= 1e-12);
    }
    for &value in result.parameter_gradient() {
        assert!((value / 0.00455941034031335 - 1.).abs() <= 1e-12);
    }
}

#[test]
fn continuous_dp5_rejections_retain_the_base_derivative() {
    let model = CountedDiagonal {
        model: Model {
            p: [-0.7; 4],
            cancel: None,
        },
        pullbacks: core::cell::Cell::new(0),
    };
    let error = ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap();
    let options = AdjointOptions::new(AdjointMode::Continuous, 1.)
        .unwrap()
        .with_fixed_step(true)
        .with_tape_capacity(4)
        .with_max_reverse_steps(10000)
        .with_adjoint_error(error.clone())
        .with_quadrature_error(error)
        .with_max_workspace_bytes(1024 * 1024);
    let plan = OdeAdjointSolver::new(ode_shape(&model, &Energy(false)), options).unwrap();
    let result = plan
        .solve(&model, &Energy(false), &[1.; 4], (0., 1.), None)
        .unwrap();
    let counts = result.report().accounting().own();
    let accepted = counts.accepted() - result.report().accepted_steps();
    assert!(counts.rejected() > 0);
    assert_eq!(
        model.pullbacks.get(),
        1 + 6 * (accepted + counts.rejected())
    );
    let expected = (result.report().objective().unwrap() / 2.).sqrt() * (-0.7_f64).exp();
    for &value in result.initial_gradient() {
        assert!((value - expected).abs() < 1e-10);
    }
}

#[test]
fn continuous_dp5_repeated_workspaces_restart_in_both_directions() {
    let model = CountedDiagonal {
        model: Model {
            p: [-0.7; 4],
            cancel: None,
        },
        pullbacks: core::cell::Cell::new(0),
    };
    for policy in [
        CheckpointPolicy::StoreAll,
        CheckpointPolicy::Explicit(CheckpointSchedule::binomial(64, 2).unwrap()),
    ] {
        let options = AdjointOptions::new(AdjointMode::Continuous, 0.125)
            .unwrap()
            .with_tape_capacity(64)
            .with_max_reverse_steps(10000)
            .with_max_replay_steps(10000)
            .with_max_workspace_bytes(1024 * 1024)
            .with_checkpoint_policy(policy);
        let plan = OdeAdjointSolver::new(ode_shape(&model, &Energy(false)), options).unwrap();
        let mut workspace = plan.workspace().unwrap();
        for interval in [(0., 1.), (1., 0.), (0., 1.)] {
            let reference = plan
                .solve(&model, &Energy(false), &[1.; 4], interval, None)
                .unwrap();
            let (mut initial, mut parameters) = ([0.; 4], [0.; 4]);
            let report = plan
                .solve_into(
                    &model,
                    &Energy(false),
                    &[1.; 4],
                    interval,
                    None,
                    &mut initial,
                    &mut parameters,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            for (&actual, &expected) in initial.iter().chain(&parameters).zip(
                reference
                    .initial_gradient()
                    .iter()
                    .chain(reference.parameter_gradient()),
            ) {
                assert!((actual / expected - 1.).abs() <= 1e-12);
            }
        }
    }
}

struct FailedReverse<'a> {
    model: Model<'static, f64>,
    calls: core::cell::Cell<usize>,
    fail_at: usize,
    cancel: Option<&'a CancellationToken>,
}
impl OdeSystem<f64> for FailedReverse<'_> {
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
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        out: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.model.jacobian(t, y, out, cx)
    }
}
impl OdeAdjointSystem<f64> for FailedReverse<'_> {
    fn parameters(&self) -> &[f64] {
        &self.model.p
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
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.fail_at {
            if let Some(cancel) = self.cancel {
                cancel.cancel();
            } else {
                return Err(convergence(
                    ConvergenceReason::Divergence,
                    "pullback failed",
                ));
            }
        }
        self.model.pullback(t, y, seed, state, parameters, work, cx)
    }
}

#[test]
fn continuous_reverse_provider_errors_and_cancellation_do_not_retry() {
    for dae in [false, true] {
        for cancelled in [false, true] {
            let token = CancellationToken::new();
            let model = FailedReverse {
                model: Model {
                    p: [-0.7],
                    cancel: None,
                },
                calls: core::cell::Cell::new(0),
                // Residual adjoints first reduce the terminal cotangent.
                fail_at: if dae { 2 } else { 1 },
                cancel: cancelled.then_some(&token),
            };
            let options = AdjointOptions::new(AdjointMode::Continuous, 0.125)
                .unwrap()
                .with_fixed_step(true)
                .with_tape_capacity(8)
                .with_max_reverse_steps(10000)
                .with_max_workspace_bytes(1024 * 1024);
            let scope = ExecutionContext::sequential().with_cancellation(&token);
            let (mut initial, mut parameters) = ([10.], [11.]);
            let (error, report) = if dae {
                let residual = OdeAdjointResidual::new(&model).unwrap();
                let plan = DaeAdjointSolver::backward_euler(
                    dae_shape(&residual, &Energy(false)),
                    OdeStageBackend::Dense,
                    (DaeOptions::new()).numerics().clone(),
                    options,
                    transpose_defect(),
                )
                .unwrap();
                let mut workspace = plan.workspace().unwrap();
                let error = plan
                    .solve_into(
                        &residual,
                        &Energy(false),
                        &[1.],
                        &[-0.7],
                        (0., 0.5),
                        &mut initial,
                        &mut parameters,
                        &mut workspace,
                        scope,
                    )
                    .unwrap_err();
                (
                    error,
                    *workspace.last_attempt().unwrap().progress().unwrap(),
                )
            } else {
                let plan =
                    OdeAdjointSolver::new(ode_shape(&model, &Energy(false)), options).unwrap();
                let mut workspace = plan.workspace().unwrap();
                let error = plan
                    .solve_into(
                        &model,
                        &Energy(false),
                        &[1.],
                        (0., 0.5),
                        None,
                        &mut initial,
                        &mut parameters,
                        &mut workspace,
                        scope,
                    )
                    .unwrap_err();
                (
                    error,
                    *workspace.last_attempt().unwrap().progress().unwrap(),
                )
            };
            assert_eq!(model.calls.get(), model.fail_at);
            assert_eq!(initial, [10.]);
            assert_eq!(parameters, [11.]);
            assert_eq!(report.reverse_steps(), 0);
            assert_eq!(report.counts().rejected(), 0);
            if cancelled {
                assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
                assert_eq!(
                    report.termination(),
                    Some(AdjointTermination::ExecutionStopped(
                        scioxide_core::ExecutionStopReason::Cancelled
                    ))
                );
            } else {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::ConvergenceFailure {
                        reason: ConvergenceReason::Divergence,
                        ..
                    }
                ));
                assert_eq!(report.termination(), None);
            }
        }
    }
}

fn ode_shape<'a, T: NumericScalar>(
    system: &impl OdeAdjointSystem<T>,
    objective: &'a impl AdjointObjective<T>,
) -> Shape<'a, T::Real> {
    Shape {
        dimension: system.dimension(),
        parameters: system.parameter_dimension(),
        provider_scratch: system.pullback_workspace_len(),
        objective_scratch: objective.workspace_len(),
        observations: objective.observation_times(),
    }
}
fn dae_shape<'a, T: NumericScalar>(
    system: &impl DaeAdjointSystem<T>,
    objective: &'a impl AdjointObjective<T>,
) -> Shape<'a, T::Real> {
    Shape {
        dimension: system.dimension(),
        parameters: system.parameter_dimension(),
        provider_scratch: system.pullback_workspace_len(),
        objective_scratch: objective.workspace_len(),
        observations: objective.observation_times(),
    }
}
