//! Transactional executor and solve-budget contracts for geometric operations.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, ExecutionContext, ExecutionPlan, ExecutionPolicy,
    Executor, NdView, PartitionFn, ProblemToken, SciOxideError, SequentialExecutor, SolveBudgets,
    SolveIdentity, SolveState, StridedLayout,
};
use scioxide_spatial::*;
struct Threads;
impl Executor for Threads {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, &|partition, (), scope| {
            std::thread::scope(|threads| {
                threads
                    .spawn(|| operation(partition, (), scope))
                    .join()
                    .unwrap()
            })
        })
    }
}
#[test]
fn strided_batch_executor_and_atomic_failure() {
    let data: Vec<f64> = (0..1800).map(|i| i as f64 / 1000.0).collect();
    let shape = [300, 3];
    let strides = [6, 2];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let r = Rotation::from_rotation_vector([0.2, 0.3, -0.4]).unwrap();
    let t = RigidTransform::new(r, [1.0, 2.0, 3.0]).unwrap();
    let mut output = vec![[0.0; 3]; 300];
    let mut scratch = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
        300,
        TransformOptions::new(),
    ))
    .unwrap();
    t.apply_batch_into(
        &points,
        TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut output,
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let expected = output.clone();
    t.apply_batch_into(
        &points,
        TransformOptions::new().with_execution_policy(ExecutionPolicy::Auto),
        &mut output,
        &mut scratch,
        ExecutionContext::new(&Threads),
    )
    .unwrap();
    assert_eq!(output, expected);
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        t.apply_batch_into(
            &points,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Auto),
            &mut output,
            &mut scratch,
            ExecutionContext::new(&Threads).with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(output, expected);
    let huge = [f64::MAX; 9];
    let shape = [3, 3];
    let strides = [3, 1];
    let huge =
        PointSet::new(NdView::new(&huge, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let overflow = RigidTransform::new(Rotation::identity(), [f64::MAX; 3]).unwrap();
    let mut out = [[42.0; 3]; 3];
    let mut stage =
        TransformWorkspace::workspace(TransformWorkspaceRequest::apply(3, TransformOptions::new()))
            .unwrap();
    assert!(
        overflow
            .apply_batch_into(
                &huge,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut out,
                &mut stage,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(out, [[42.0; 3]; 3]);
}
#[test]
fn spline_obeys_local_iterations_and_nested_accounting() {
    let rotations = [
        [0.1, 0.2, 0.3],
        [0.4, -0.1, 0.6],
        [-0.7, 0.6, 0.9],
        [0.8, 0.2, -0.1],
    ]
    .map(|r| Rotation::from_rotation_vector(r).unwrap());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(569),
        scioxide_core::algorithm_id!("spatial", "rotation-spline", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let accepted = context.accepted_state();
    let result = RotationSpline::build(
        &[0.0, 0.7, 2.0, 3.0],
        &rotations,
        RotationSplineOptions::default().with_max_iterations(1),
        &mut context,
    )
    .unwrap();
    assert_eq!(
        result.termination,
        RotationSplineTermination::IterationLimit
    );
    assert_eq!(result.iterations, 1);
    assert!(result.spline.is_none());
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
    assert_eq!(context.accepted_state(), accepted);
    let token = CancellationToken::new();
    token.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity,
            &mut budgets,
        )
        .unwrap();
    assert!(
        RotationSpline::build(
            &[0.0, 0.7, 2.0, 3.0],
            &rotations,
            RotationSplineOptions::default(),
            &mut context
        )
        .is_err()
    );
    assert_eq!(context.counts().own().iterations(), 0);
}

struct CancelAfter<'a>(&'a CancellationToken);
impl Executor for CancelAfter<'_> {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, operation)?;
        self.0.cancel();
        Ok(())
    }
}
struct Missing;
impl Executor for Missing {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        _: ExecutionContext<'_>,
        _: ExecutionPlan,
        _: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}

#[test]
fn euler_and_spline_batches_parallel_and_transactional() {
    use scioxide_core::Tolerance;
    let r = Rotation::from_rotation_vector([0.2_f64, 0.3, -0.4]).unwrap();
    let sequence = EulerSequence::new(
        [RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
        EulerFrame::Extrinsic,
    )
    .unwrap();
    let spline = RotationSpline::build(
        &[0.0, 1.0],
        &[Rotation::identity(), r],
        RotationSplineOptions::default(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
    .spline
    .unwrap();
    let mut angles = [EulerAngles {
        angles: [42.0; 3],
        gimbal_lock: false,
    }; 300];
    let mut angle_stage = TransformWorkspace::workspace(TransformWorkspaceRequest::euler(
        300,
        TransformOptions::new(),
    ))
    .unwrap();
    let mut samples = [r; 300];
    let mut sample_stage = TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
        300,
        TransformOptions::new(),
    ))
    .unwrap();
    let times: Vec<_> = (0..300).map(|i| (i % 100) as f64 / 99.0).collect();
    Rotation::as_euler_batch_into(
        &[r; 300],
        sequence,
        Tolerance::from_sqrt_epsilon(),
        TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut angles,
        &mut angle_stage,
        ExecutionContext::sequential(),
    )
    .unwrap();
    spline
        .evaluate_rotations_into(
            &times,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut samples,
            &mut sample_stage,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let expected_angles = angles;
    let expected_samples = samples;
    Rotation::as_euler_batch_into(
        &[r; 300],
        sequence,
        Tolerance::from_sqrt_epsilon(),
        TransformOptions::new()
            .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap())),
        &mut angles,
        &mut angle_stage,
        ExecutionContext::new(&Threads),
    )
    .unwrap();
    spline
        .evaluate_rotations_into(
            &times,
            TransformOptions::new()
                .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap())),
            &mut samples,
            &mut sample_stage,
            ExecutionContext::new(&Threads),
        )
        .unwrap();
    assert_eq!(angles, expected_angles);
    assert_eq!(samples, expected_samples);
    for mode in 0..3 {
        // Exercise an executor failure, pre-cancellation, and cancellation after
        // successful staging but before the caller's output is published.
        let token = CancellationToken::new();
        let executor: &dyn Executor = if mode == 0 {
            &Missing
        } else {
            &CancelAfter(&token)
        };
        if mode == 1 {
            token.cancel();
        }
        let context = ExecutionContext::new(executor).with_cancellation(&token);
        angles.fill(EulerAngles {
            angles: [42.0; 3],
            gimbal_lock: false,
        });
        let before = angles;
        assert!(
            Rotation::as_euler_batch_into(
                &[r; 300],
                sequence,
                Tolerance::from_sqrt_epsilon(),
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
                    NonZeroUsize::new(4).unwrap()
                )),
                &mut angles,
                &mut angle_stage,
                context
            )
            .is_err()
        );
        assert_eq!(angles, before);
        let token = CancellationToken::new();
        let executor: &dyn Executor = if mode == 0 {
            &Missing
        } else {
            &CancelAfter(&token)
        };
        if mode == 1 {
            token.cancel();
        }
        let context = ExecutionContext::new(executor).with_cancellation(&token);
        samples.fill(r);
        assert!(
            spline
                .evaluate_rotations_into(
                    &times,
                    TransformOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
                        NonZeroUsize::new(4).unwrap()
                    )),
                    &mut samples,
                    &mut sample_stage,
                    context
                )
                .is_err()
        );
        assert_eq!(samples, [r; 300]);
    }
}

#[test]
fn application_rejects_executor_failure_and_late_cancellation() {
    let data = [0.25_f64; 900];
    let shape = [300, 3];
    let strides = [3, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let r = Rotation::from_rotation_vector([0.2, 0.3, -0.4]).unwrap();
    for policy in [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
    ] {
        for missing in [false, true] {
            let token = CancellationToken::new();
            let executor: &dyn Executor = if missing {
                &Missing
            } else {
                &CancelAfter(&token)
            };
            let mut output = [[42.0; 3]; 300];
            assert!(
                r.apply_batch_into(
                    &points,
                    TransformOptions::new().with_execution_policy(policy),
                    &mut output,
                    &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                        300,
                        TransformOptions::new()
                    ))
                    .unwrap(),
                    ExecutionContext::new(executor).with_cancellation(&token)
                )
                .is_err()
            );
            assert_eq!(output, [[42.0; 3]; 300]);
        }
    }
}

struct CancellingClock<'a> {
    reads: core::sync::atomic::AtomicUsize,
    token: &'a CancellationToken,
}
impl scioxide_core::Clock for CancellingClock<'_> {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        if self
            .reads
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
            >= 8
        {
            self.token.cancel();
        }
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::ZERO,
        ))
    }
}
#[test]
fn application_polls_between_blocks_before_publishing() {
    let data = [0.25_f64; 12288];
    let shape = [4096, 3];
    let strides = [3, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let token = CancellationToken::new();
    let clock = CancellingClock {
        reads: core::sync::atomic::AtomicUsize::new(0),
        token: &token,
    };
    let binding = scioxide_core::ClockBinding::new(&clock);
    let context = ExecutionContext::sequential()
        .with_cancellation(&token)
        .with_deadline(scioxide_core::Deadline::at(
            &binding,
            scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(1),
            ),
        ))
        .unwrap();
    let mut output = [[42.0; 3]; 4096];
    let mut staging = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
        4096,
        TransformOptions::new(),
    ))
    .unwrap();
    let error = Rotation::identity()
        .apply_batch_into(
            &points,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut staging,
            context,
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    assert!(clock.reads.load(core::sync::atomic::Ordering::Relaxed) >= 9);
    assert_eq!(output, [[42.0; 3]; 4096]);
}

#[test]
fn slerp_batches_match_scalar_under_policies_and_publish_atomically() {
    let r = Rotation::from_rotation_vector([0.2_f64, 0.3, -0.4]).unwrap();
    let slerp = Slerp::new(&[0.0, 1.0], &[Rotation::identity(), r]).unwrap();
    let times: Vec<_> = (0..300).map(|i| (i % 100) as f64 / 99.0).collect();
    let expected: Vec<_> = times
        .iter()
        .map(|&time| slerp.evaluate(time).unwrap())
        .collect();
    let mut workspace = TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
        times.len(),
        TransformOptions::new(),
    ))
    .unwrap();
    for policy in [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Auto,
        ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
    ] {
        let options = TransformOptions::new().with_execution_policy(policy);
        let mut output = vec![r; times.len()];
        slerp
            .evaluate_rotations_into(
                &times,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::new(&Threads),
            )
            .unwrap();
        assert_eq!(output, expected);
        for mode in 0..3 {
            let token = CancellationToken::new();
            let executor: &dyn Executor = if mode == 0 {
                &Missing
            } else {
                &CancelAfter(&token)
            };
            if mode == 1 {
                token.cancel();
            }
            output.fill(r);
            assert!(
                slerp
                    .evaluate_rotations_into(
                        &times,
                        options,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::new(executor).with_cancellation(&token)
                    )
                    .is_err()
            );
            assert_eq!(output, vec![r; times.len()]);
        }
        // Reusing staging after a failed dispatch is supported.
        slerp
            .evaluate_rotations_into(
                &times,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::new(&Threads),
            )
            .unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn nested_preparation_preserves_parent_state_and_local_allowance() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(1122),
        scioxide_core::algorithm_id!("test", "transform-parent", 1),
        0,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(parent.record_iteration().unwrap().is_granted());
    let accepted = parent.accepted_state();
    let rotations = [
        [0.1, 0.2, 0.3],
        [0.4, -0.1, 0.6],
        [-0.7, 0.6, 0.9],
        [0.8, 0.2, -0.1],
    ]
    .map(|v| Rotation::from_rotation_vector(v).unwrap());
    let spline = RotationSpline::build(
        &[0.0, 0.7, 2.0, 3.0],
        &rotations,
        RotationSplineOptions::new(),
        scioxide_core::SolveScope::from(&mut parent),
    )
    .unwrap();
    assert_eq!(spline.termination, RotationSplineTermination::Converged);
    assert!(spline.iterations > 1);
    assert_eq!(parent.counts().nested().iterations(), spline.iterations);
    assert_eq!(parent.counts().own().iterations(), 1);
    assert_eq!(parent.accepted_state(), accepted);
    let data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let shape = [4, 3];
    let strides = [3, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let before = parent.counts().nested().iterations();
    let fit = estimate_rigid_transform_scoped(
        &points,
        &points,
        None,
        RigidEstimationOptions::new(),
        &mut parent,
    )
    .unwrap();
    assert!(fit.rms_residual < 1e-12);
    assert!(parent.counts().nested().iterations() > before + 1);
    assert_eq!(parent.counts().own().iterations(), 1);
    assert_eq!(parent.accepted_state(), accepted);
    let before = parent.counts();
    assert!(
        RotationSpline::build(
            &[0.0, 0.0],
            &rotations[..2],
            RotationSplineOptions::new(),
            &mut parent
        )
        .is_err()
    );
    assert_eq!(parent.counts(), before);
    let mut workspace =
        TransformWorkspace::workspace(TransformWorkspaceRequest::apply(4, TransformOptions::new()))
            .unwrap();
    let mut output = [[42.0; 3]; 4];
    Rotation::identity()
        .apply_batch_into(
            &points,
            TransformOptions::new(),
            &mut output,
            &mut workspace,
            SpatialExecutionContext::from(&mut parent),
        )
        .unwrap();
    assert_eq!(parent.counts(), before);
    assert_eq!(parent.accepted_state(), accepted);
}

#[test]
fn estimation_obeys_svd_limits_and_cancellation() {
    let data: [f64; 12] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let shape = [4, 3];
    let strides = [3, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let error = estimate_rigid_transform_scoped(
        &points,
        &points,
        None,
        RigidEstimationOptions::new(),
        ExecutionContext::sequential().with_cancellation(&token),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    let options = RigidEstimationOptions::new().with_svd(
        scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
    );
    assert_eq!(
        estimate_rigid_transform_scoped(
            &points,
            &points,
            None,
            options,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let options = RigidEstimationOptions::new().with_svd(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_time_limit(core::time::Duration::from_secs(1)),
    );
    assert_eq!(
        estimate_rigid_transform_scoped(
            &points,
            &points,
            None,
            options,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::CapabilityUnavailable
    );
}

struct ExpiredClock;
impl scioxide_core::Clock for ExpiredClock {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_secs(2),
        ))
    }
}

#[test]
fn expired_deadlines_reject_empty_batches_too() {
    let binding = scioxide_core::ClockBinding::new(&ExpiredClock);
    let context = ExecutionContext::sequential()
        .with_deadline(scioxide_core::Deadline::at(
            &binding,
            scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(1),
            ),
        ))
        .unwrap();
    let r = Rotation::<f64>::identity();
    let slerp = Slerp::new(&[0.0, 1.0], &[r; 2]).unwrap();
    let spline = RotationSpline::build(
        &[0.0, 1.0],
        &[r; 2],
        RotationSplineOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap()
    .spline
    .unwrap();
    let mut workspace = TransformWorkspace::new();
    let options = TransformOptions::new();
    let shape = [0, 3];
    let strides = [3, 1];
    let points =
        PointSet::new(NdView::new(&[], StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let sequence = EulerSequence::new(
        [RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
        EulerFrame::Extrinsic,
    )
    .unwrap();
    for error in [
        r.apply_batch_into(&points, options, &mut [], &mut workspace, context)
            .unwrap_err(),
        RigidTransform::identity()
            .apply_batch_into(&points, options, &mut [], &mut workspace, context)
            .unwrap_err(),
        Rotation::as_euler_batch_into(
            &[],
            sequence,
            scioxide_core::Tolerance::from_sqrt_epsilon(),
            options,
            &mut [],
            &mut workspace,
            context,
        )
        .unwrap_err(),
        slerp
            .evaluate_rotations_into(&[], options, &mut [], &mut workspace, context)
            .unwrap_err(),
        spline
            .evaluate_rotations_into(&[], options, &mut [], &mut workspace, context)
            .unwrap_err(),
        RotationSpline::build(&[0.0, 1.0], &[r; 2], RotationSplineOptions::new(), context)
            .unwrap_err(),
    ] {
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ExecutionStopped
        );
    }
}
