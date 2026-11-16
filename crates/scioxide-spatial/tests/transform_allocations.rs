//! Prepared rotation/interpolation queries allocate no storage.
#![allow(clippy::unwrap_used)]

use scioxide_spatial::*;

#[test]
fn prepared_geometry_allocates_nothing() {
    let r = Rotation::from_rotation_vector([0.2_f64, 0.3, 0.4]).unwrap();
    let rotations = [Rotation::identity(), r, r.inverse()];
    let times = [0.0, 1.0, 2.0];
    let slerp = Slerp::new(&times, &rotations).unwrap();
    let spline = RotationSpline::build(
        &times,
        &rotations,
        RotationSplineOptions::default(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
    .spline
    .unwrap();
    let t = RigidTransform::new(r, [1.0, 2.0, 3.0]).unwrap();
    {
        std::hint::black_box(r.compose(r).unwrap().apply([1.0, 2.0, 3.0]).unwrap());
        std::hint::black_box(t.inverse().unwrap().apply([1.0, 2.0, 3.0]).unwrap());
        std::hint::black_box(slerp.evaluate(0.5).unwrap());
        std::hint::black_box(spline.evaluate(1.5).unwrap());
    };
}

#[test]
fn sequential_transform_batches_execute() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, NdView, StridedLayout, Tolerance};
    let r = Rotation::from_rotation_vector([0.2_f64, 0.3, 0.4]).unwrap();
    let t = RigidTransform::new(r, [1.0, 2.0, 3.0]).unwrap();
    let data = [0.25; 900];
    let shape = [300, 3];
    let strides = [3, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let mut output = [[0.0; 3]; 300];
    let mut staging = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
        300,
        TransformOptions::new(),
    ))
    .unwrap();
    let rotations = [r; 300];
    let sequence = EulerSequence::new(
        [RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
        EulerFrame::Extrinsic,
    )
    .unwrap();
    let mut angles = [EulerAngles {
        angles: [0.0; 3],
        gimbal_lock: false,
    }; 300];
    let mut angle_staging = TransformWorkspace::workspace(TransformWorkspaceRequest::euler(
        300,
        TransformOptions::new(),
    ))
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
    let mut samples = [Rotation::identity(); 300];
    let mut sample_staging = TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
        300,
        TransformOptions::new(),
    ))
    .unwrap();
    let slerp = Slerp::new(&[0.0, 1.0], &[Rotation::identity(), r]).unwrap();
    {
        r.apply_batch_into(
            &points,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        t.apply_batch_into(
            &points,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        Rotation::as_euler_batch_into(
            &rotations,
            sequence,
            Tolerance::from_sqrt_epsilon(),
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut angles,
            &mut angle_staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
        slerp
            .evaluate_rotations_into(
                &[0.5; 300],
                TransformOptions::new(),
                &mut samples,
                &mut sample_staging,
                ExecutionContext::sequential(),
            )
            .unwrap();
        spline
            .evaluate_rotations_into(
                &[0.5; 300],
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut samples,
                &mut sample_staging,
                ExecutionContext::sequential(),
            )
            .unwrap();
    };
}
