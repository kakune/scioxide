//! Batch geometry preserves scalar conventions, layout proofs and atomic output.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NdView, NonFiniteOrigin, RealScalar, SciOxideErrorView,
    StridedLayout, Tolerance,
};
use scioxide_spatial::*;

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn close<S: RealScalar>(left: S, right: S, tolerance: f64) {
    assert!(
        (left.into_f64() - right.into_f64()).abs() <= tolerance * (1.0 + right.into_f64().abs())
    );
}
fn points<'a, 'b, S: RealScalar>(
    data: &'a [S],
    shape: &'b [usize; 2],
    strides: &'b [usize; 2],
) -> PointSet<'a, S> {
    PointSet::new(NdView::new(data, StridedLayout::new_broadcast(shape, strides).unwrap()).unwrap())
        .unwrap()
}

fn application<S: RealScalar>() {
    let rotation = Rotation::from_rotation_vector([0.2, -0.4, 0.7].map(scalar::<S>)).unwrap();
    let transform = RigidTransform::new(rotation, [0.5, -1.0, 2.0].map(scalar::<S>)).unwrap();
    // Padding is deliberately non-finite: only logical coordinates participate.
    for strides in [[3, 1], [7, 2], [5, 1], [0, 1], [1, 0]] {
        let shape = [5, 3];
        let mut data = vec![S::NAN; 40];
        for i in 0..5 {
            for j in 0..3 {
                data[i * strides[0] + j * strides[1]] = scalar((i + j) as f64 / 8.0);
            }
        }
        let input = points(&data, &shape, &strides);
        for translate in [false, true] {
            let mut out = [[S::ZERO; 3]; 5];
            let mut scratch = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                6,
                TransformOptions::new(),
            ))
            .unwrap();
            if translate {
                transform
                    .apply_batch_into(
                        &input,
                        TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                        &mut out,
                        &mut scratch,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            } else {
                rotation
                    .apply_batch_into(
                        &input,
                        TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                        &mut out,
                        &mut scratch,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            }
            for (i, actual) in out.into_iter().enumerate() {
                let p = core::array::from_fn(|j| data[i * strides[0] + j * strides[1]]);
                let expected = if translate {
                    transform.apply(p)
                } else {
                    rotation.apply(p)
                }
                .unwrap();
                for (a, b) in actual.into_iter().zip(expected) {
                    close(a, b, 16.0 * S::EPSILON.into_f64());
                }
            }
        }
    }
    let shape = [0, 3];
    let strides = [3, 1];
    let input = points(&[], &shape, &strides);
    rotation
        .apply_batch_into(
            &input,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut [],
            &mut TransformWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    let shape = [1, 3];
    for magnitude in [
        S::MAX / scalar::<S>(2.0),
        S::MIN_POSITIVE / scalar::<S>(8.0),
    ] {
        let data = [magnitude, -magnitude, S::ZERO];
        let input = points(&data, &shape, &strides);
        let quarter = Rotation::from_quaternion([S::ZERO, S::ZERO, S::ONE, S::ONE]).unwrap();
        let expected = quarter.apply(data).unwrap();
        let mut out = [[S::ZERO; 3]];
        quarter
            .apply_batch_into(
                &input,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut out,
                &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                    1,
                    TransformOptions::new(),
                ))
                .unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        for (a, b) in out[0].into_iter().zip(expected) {
            close(a / magnitude, b / magnitude, 8.0 * S::EPSILON.into_f64());
        }
    }
}
#[test]
fn application_f32() {
    application::<f32>();
}
#[test]
fn application_f64() {
    application::<f64>();
}

#[test]
fn invalid_coordinates_shapes_and_late_overflow_preserve_output() {
    let r = Rotation::identity();
    let shape = [2, 3];
    let strides = [3, 1];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, bad];
        let mut out = [[42.0; 3]; 2];
        let error = PointSet::new(
            NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .and_then(|p| {
            r.apply_batch_into(
                &p,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut out,
                &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                    2,
                    TransformOptions::new(),
                ))
                .unwrap(),
                ExecutionContext::sequential(),
            )
        })
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
        assert_eq!(out, [[42.0; 3]; 2]);
    }
    let input = points(&[0.0; 6], &shape, &strides);
    let mut output = [[42.0; 3]; 2];
    assert!(
        r.apply_batch_into(
            &input,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                1,
                TransformOptions::new()
            ))
            .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(
        r.apply_batch_into(
            &input,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output[..1],
            &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                2,
                TransformOptions::new()
            ))
            .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, [[42.0; 3]; 2]);
    let shape = [300, 3];
    let mut data = vec![0.0; 900];
    data[899] = f64::MAX;
    let input = points(&data, &shape, &strides);
    let transform = RigidTransform::new(r, [0.0, 0.0, f64::MAX]).unwrap();
    let mut output = [[42.0; 3]; 300];
    let error = transform
        .apply_batch_into(
            &input,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                300,
                TransformOptions::new(),
            ))
            .unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability { .. }
    ));
    assert_eq!(output, [[42.0; 3]; 300]);
    let wrong_shape = [2, 2];
    let wrong = points(&[0.0; 6], &wrong_shape, &strides);
    assert!(
        r.apply_batch_into(
            &wrong,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output[..2],
            &mut TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
                2,
                TransformOptions::new()
            ))
            .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, [[42.0; 3]; 300]);
}

fn euler<S: RealScalar>() {
    use RotationAxis::*;
    let tolerance =
        Tolerance::new(scalar::<S>(8.0) * S::EPSILON, scalar::<S>(8.0) * S::EPSILON).unwrap();
    let zero = EulerAngles {
        angles: [S::ZERO; 3],
        gimbal_lock: false,
    };
    for a in [X, Y, Z] {
        for b in [X, Y, Z] {
            for c in [X, Y, Z] {
                if a == b || b == c {
                    continue;
                }
                for frame in [EulerFrame::Intrinsic, EulerFrame::Extrinsic] {
                    let seq = EulerSequence::new([a, b, c], frame).unwrap();
                    let centers = if a == c {
                        [S::ZERO, S::PI]
                    } else {
                        [-S::PI * S::HALF, S::PI * S::HALF]
                    };
                    for center in centers {
                        for delta in [
                            S::ZERO,
                            S::EPSILON,
                            -S::EPSILON,
                            S::EPSILON.sqrt(),
                            -S::EPSILON.sqrt(),
                        ] {
                            let r = Rotation::from_euler(
                                seq,
                                [scalar(0.3), center + delta, scalar(-0.8)],
                            )
                            .unwrap();
                            let mut output = [zero; 3];
                            Rotation::as_euler_batch_into(
                                &[r; 3],
                                seq,
                                tolerance,
                                TransformOptions::new()
                                    .with_execution_policy(ExecutionPolicy::Sequential),
                                &mut output,
                                &mut TransformWorkspace::workspace(
                                    TransformWorkspaceRequest::euler(3, TransformOptions::new()),
                                )
                                .unwrap(),
                                ExecutionContext::sequential(),
                            )
                            .unwrap();
                            for actual in output {
                                let expected = r.as_euler(seq, tolerance).unwrap();
                                assert_eq!(actual.gimbal_lock, expected.gimbal_lock);
                                assert_eq!(actual.gimbal_lock, delta.abs() <= S::EPSILON);
                                for (x, y) in actual.angles.into_iter().zip(expected.angles) {
                                    close(x, y, 8.0 * S::EPSILON.into_f64());
                                }
                                if actual.gimbal_lock {
                                    assert_eq!(actual.angles[2], S::ZERO);
                                }
                                let reconstructed =
                                    Rotation::from_euler(seq, actual.angles).unwrap();
                                let error = r
                                    .inverse()
                                    .compose(reconstructed)
                                    .unwrap()
                                    .as_axis_angle()
                                    .1;
                                assert!(error <= scalar::<S>(16.0) * S::EPSILON.sqrt());
                            }
                        }
                    }
                }
            }
        }
    }
    let seq = EulerSequence::new([X, Y, Z], EulerFrame::Extrinsic).unwrap();
    Rotation::<S>::as_euler_batch_into(
        &[],
        seq,
        tolerance,
        TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut [],
        &mut TransformWorkspace::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut output = [zero];
    for (count, capacity) in [(2, 1), (1, 0)] {
        let scratch = &mut TransformWorkspace::workspace(TransformWorkspaceRequest::euler(
            capacity,
            TransformOptions::new(),
        ))
        .unwrap();
        assert!(
            Rotation::<S>::as_euler_batch_into(
                &[Rotation::identity(); 2][..count],
                seq,
                tolerance,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut output,
                scratch,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, [zero]);
    }
    let overflow = Tolerance::new(S::MAX, S::MAX).unwrap();
    assert!(
        Rotation::as_euler_batch_into(
            &[Rotation::identity()],
            seq,
            overflow,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut TransformWorkspace::workspace(TransformWorkspaceRequest::euler(
                1,
                TransformOptions::new()
            ))
            .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, [zero]);
}
#[test]
fn near_lock_euler_f32() {
    euler::<f32>();
}
#[test]
fn near_lock_euler_f64() {
    euler::<f64>();
}

fn spline<S: RealScalar>() {
    let rotations = [[0.0, 0.0, 0.0], [0.3, -0.2, 0.4], [-0.2, 0.3, 0.1]]
        .map(|r| Rotation::from_rotation_vector(r.map(scalar::<S>)).unwrap());
    let knots = [0.0, 0.7, 2.0].map(scalar::<S>);
    let spline = RotationSpline::with_options(&knots, &rotations, RotationSplineOptions::default())
        .unwrap()
        .spline
        .unwrap();
    let queries = [2.0, 0.5, 0.0, 1.5, 0.7, 0.5].map(scalar::<S>);
    let mut output = [Rotation::identity(); 6];
    let mut staging = TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
        6,
        TransformOptions::new(),
    ))
    .unwrap();
    spline
        .evaluate_rotations_into(
            &queries,
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut staging,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output[0], rotations[2]);
    assert_eq!(output[2], rotations[0]);
    assert_eq!(output[4], rotations[1]);
    for (r, t) in output.into_iter().zip(queries) {
        let error = r
            .inverse()
            .compose(spline.evaluate(t).unwrap().rotation)
            .unwrap()
            .as_axis_angle()
            .1;
        assert!(error <= scalar::<S>(16.0) * S::EPSILON);
    }
    let before = output;
    for bad in [S::NAN, S::INFINITY, -S::INFINITY, scalar(-0.1), scalar(2.1)] {
        let mut queries = queries;
        queries[5] = bad;
        let error = spline
            .evaluate_rotations_into(
                &queries,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut output,
                &mut staging,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        if !bad.is_finite() {
            assert!(matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        assert_eq!(output, before);
    }
    assert!(
        spline
            .evaluate_rotations_into(
                &queries,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut output,
                &mut TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(
                    5,
                    TransformOptions::new()
                ))
                .unwrap(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(
        spline
            .evaluate_rotations_into(
                &queries[..5],
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut output,
                &mut staging,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, before);
    spline
        .evaluate_rotations_into(
            &[],
            TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut [],
            &mut TransformWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
}
#[test]
fn spline_batch_f32() {
    spline::<f32>();
}
#[test]
fn spline_batch_f64() {
    spline::<f64>();
}

#[test]
fn kabsch_preserves_degenerate_error_classification() {
    let shape = [4, 3];
    let strides = [3, 1];
    for data in [
        [0.0; 12],
        [2.0; 12],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    ] {
        let p = points(&data, &shape, &strides);
        for weights in [None, Some(&[1.0; 4][..])] {
            let error = estimate_rigid_transform_scoped(
                &p,
                &p,
                weights,
                RigidEstimationOptions::default(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err();
            assert!(
                matches!(error.view(), SciOxideErrorView::InvalidInput { argument,  .. }
                if argument == scioxide_core::argument!("source"))
            );
        }
    }
    let tetrahedron = [
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
    ];
    let reflected = tetrahedron.map(|x| -x);
    let a = points(&tetrahedron, &shape, &strides);
    let b = points(&reflected, &shape, &strides);
    let error = estimate_rigid_transform(&a, &b).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { detail, .. } if detail == "reflected alignment has nonunique optimum")
    );
    for weights in [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0; 4]] {
        assert!(matches!(
            estimate_rigid_transform_scoped(
                &a,
                &a,
                Some(&weights),
                RigidEstimationOptions::default(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap_err()
            .view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = estimate_rigid_transform_scoped(
            &a,
            &b,
            Some(&[1.0, 1.0, 1.0, bad]),
            RigidEstimationOptions::default(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}

#[test]
fn kabsch_layouts_weights_and_zero_weight_outliers() {
    let source = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [f64::MAX; 3],
    ];
    let shape = [5, 3];
    for left_stride in [[3, 1], [7, 2]] {
        for right_stride in [[3, 1], [7, 2]] {
            let mut left = [f64::NAN; 40];
            let mut right = left;
            for (i, p) in source.into_iter().enumerate() {
                let target = if i == 4 {
                    [-f64::MAX; 3]
                } else {
                    [-p[1] + 5.0, p[0] - 2.0, p[2] + 1.0]
                };
                for j in 0..3 {
                    left[i * left_stride[0] + j * left_stride[1]] = p[j];
                    right[i * right_stride[0] + j * right_stride[1]] = target[j];
                }
            }
            let a = points(&left, &shape, &left_stride);
            let b = points(&right, &shape, &right_stride);
            for weights in [[1.0, 2.0, 3.0, 4.0, 0.0], [1e300, 2e300, 3e300, 4e300, 0.0]] {
                let fit = estimate_rigid_transform_scoped(
                    &a,
                    &b,
                    Some(&weights),
                    RigidEstimationOptions::default(),
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
                assert_eq!(fit.rank, 3);
                assert!(fit.rms_residual < 1e-13);
                for (actual, expected) in fit
                    .transform
                    .translation()
                    .into_iter()
                    .zip([5.0, -2.0, 1.0])
                {
                    close(actual, expected, 1e-14);
                }
            }
            let shape = [4, 3];
            let a = points(&left, &shape, &left_stride);
            let b = points(&right, &shape, &right_stride);
            let fit = estimate_rigid_transform(&a, &b).unwrap();
            assert!(fit.rms_residual < 1e-13);
        }
    }
}

#[test]
fn kabsch_retains_residuals_whose_squares_underflow() {
    let source = [
        1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
    ];
    let mut target = source;
    target[1] = 1e-200;
    target[4] = 1e-200;
    let shape = [6, 3];
    let strides = [3, 1];
    let fit = estimate_rigid_transform(
        &points(&source, &shape, &strides),
        &points(&target, &shape, &strides),
    )
    .unwrap();
    let independent = source
        .chunks_exact(3)
        .zip(target.chunks_exact(3))
        .map(|(a, b)| {
            let p = fit.transform.apply([a[0], a[1], a[2]]).unwrap();
            p.into_iter()
                .zip(b)
                .map(|(x, y)| ((x - y) / 1e-200).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
        / 6.0_f64.sqrt();
    assert!(fit.rms_residual > 0.0 && fit.rms_residual < 1e-190);
    close(fit.rms_residual / 1e-200, independent, 1e-14);
}

fn slerp_batch<S: RealScalar>() {
    let identity = Rotation::identity();
    let half_turn = Rotation::from_quaternion([S::ZERO, S::ZERO, S::ONE, S::ZERO]).unwrap();
    let turn = Rotation::from_rotation_vector([0.3, -0.2, 0.7].map(scalar::<S>)).unwrap();
    let knots = [0.0, 1.0, 2.0].map(scalar::<S>);
    let queries = [2.0, 0.5, 0.0, 1.5, 1.0, 0.5].map(scalar::<S>);
    let options = TransformOptions::new();
    let mut workspace =
        TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(queries.len(), options))
            .unwrap();
    for rotations in [
        [identity; 3],
        [identity, half_turn, identity],
        [turn, half_turn, turn.inverse()],
    ] {
        let slerp = Slerp::new(&knots, &rotations).unwrap();
        let mut output = [identity; 6];
        slerp
            .evaluate_rotations_into(
                &queries,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(output[0], rotations[2]);
        assert_eq!(output[2], rotations[0]);
        assert_eq!(output[4], rotations[1]);
        for (rotation, time) in output.into_iter().zip(queries) {
            let error = rotation
                .inverse()
                .compose(slerp.evaluate(time).unwrap())
                .unwrap()
                .as_axis_angle()
                .1;
            assert!(error <= scalar::<S>(16.0) * S::EPSILON);
        }
        let before = output;
        for invalid in [S::NAN, S::INFINITY, -S::INFINITY, scalar(-0.1), scalar(2.1)] {
            let mut bad = queries;
            bad[5] = invalid;
            assert!(
                slerp
                    .evaluate_rotations_into(
                        &bad,
                        options,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential()
                    )
                    .is_err()
            );
            assert_eq!(output, before);
        }
        for short_workspace in [true, false] {
            let mut short =
                TransformWorkspace::workspace(TransformWorkspaceRequest::rotations(5, options))
                    .unwrap();
            let (queries, workspace) = if short_workspace {
                (&queries[..], &mut short)
            } else {
                (&queries[..5], &mut workspace)
            };
            let error = slerp
                .evaluate_rotations_into(
                    queries,
                    options,
                    &mut output,
                    workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(output, before);
        }
        slerp
            .evaluate_rotations_into(
                &[],
                options,
                &mut [],
                &mut TransformWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
    }
}
#[test]
fn slerp_batch_f32() {
    slerp_batch::<f32>();
}
#[test]
fn slerp_batch_f64() {
    slerp_batch::<f64>();
}
