#![allow(clippy::unwrap_used)]
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NdView, RealScalar, StridedLayout, Tolerance,
};
use scioxide_spatial::*;
fn value<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn cases<S: RealScalar>() {
    let r = Rotation::from_axis_angle(
        [value::<S>(1.0), value::<S>(2.0), value::<S>(3.0)],
        value::<S>(0.7),
    )
    .unwrap();
    let transform =
        RigidTransform::new(r, [value::<S>(3.0), value::<S>(-2.0), value::<S>(1.0)]).unwrap();
    let tolerance =
        Tolerance::new(value::<S>(20.0) * S::EPSILON, value::<S>(20.0) * S::EPSILON).unwrap();
    let rt = RigidTransform::from_matrix(transform.as_matrix(), tolerance).unwrap();
    let identity = rt.compose(transform.inverse().unwrap()).unwrap();
    assert!(
        identity
            .translation()
            .iter()
            .all(|x| scioxide_core::NumericScalar::abs(*x) < value::<S>(50.0) * S::EPSILON)
    );
    let shape = [5, 3];
    let strides = [3, 1];
    for planar in [true, false] {
        let data = [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            2.0,
            3.0,
            if planar { 0.0 } else { 1.0 },
            -1.0,
            1.0,
            0.0,
        ]
        .map(value::<S>);
        let source = PointSet::new(
            NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        let mut out = [[S::ZERO; 3]; 5];
        let mut staging = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(
            out.len(),
            TransformOptions::new(),
        ))
        .unwrap();
        transform
            .apply_batch_into(
                &source,
                TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut out,
                &mut staging,
                ExecutionContext::sequential(),
            )
            .unwrap();
        let target_data: Vec<S> = out.into_iter().flatten().collect();
        let target = PointSet::new(
            NdView::new(&target_data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        for weights in [None, Some([1.0, 2.0, 3.0, 4.0, 5.0].map(value::<S>))] {
            let result = estimate_rigid_transform_scoped(
                &source,
                &target,
                weights.as_ref().map(|x| x.as_slice()),
                RigidEstimationOptions::default(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(result.rank, if planar { 2 } else { 3 });
            assert!(
                result.rms_residual < value::<S>(100.0) * S::EPSILON,
                "{}",
                result.rms_residual.into_f64()
            );
            assert!(
                result
                    .transform
                    .rotation()
                    .inverse()
                    .compose(r)
                    .unwrap()
                    .as_axis_angle()
                    .1
                    < value::<S>(100.0) * S::EPSILON
            );
        }
        assert!(
            estimate_rigid_transform_scoped(
                &source,
                &target,
                Some(&[S::ZERO; 5]),
                RigidEstimationOptions::default(),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        let before = out;
        assert!(
            transform
                .apply_batch_into(
                    &source,
                    TransformOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                    &mut out,
                    &mut TransformWorkspace::new(),
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(out, before);
    }
}
#[test]
fn f32_rigid() {
    cases::<f32>()
}
#[test]
fn f64_rigid() {
    cases::<f64>()
}

#[test]
fn alignment_shape_errors_use_dimension_mismatch() {
    use scioxide_core::{InvalidInputReason, SciOxideErrorView};
    fn with_points<T>(
        count: usize,
        dimension: usize,
        run: impl FnOnce(&PointSet<'_, f64>) -> T,
    ) -> T {
        let data = vec![0.; count * dimension];
        let shape = [count, dimension];
        let strides = [dimension, 1];
        let points = PointSet::new(
            NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        run(&points)
    }
    for (source, target, weights, expected, actual) in [
        ((3, 2), (3, 3), None, 3, 2),
        ((3, 3), (3, 4), None, 3, 4),
        ((3, 3), (4, 3), None, 3, 4),
        ((3, 3), (3, 3), Some(2), 3, 2),
    ] {
        with_points(source.0, source.1, |a| {
            with_points(target.0, target.1, |b| {
                let weights = weights.map(|n| vec![1.; n]);
                let error = estimate_rigid_transform_scoped(
                    a,
                    b,
                    weights.as_deref(),
                    RigidEstimationOptions::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
                assert!(
                    matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected: e, actual: a }
                if e == [expected] && a == [actual]),
                    "{error}"
                );
            })
        });
    }
    with_points(2, 3, |a| {
        let error = estimate_rigid_transform(a, a).unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { argument, reason: InvalidInputReason::OutOfRange, .. }
            if argument == scioxide_core::argument!("source"))
        );
    });
}

#[test]
fn alignment_scale_degeneracy_and_noise() {
    for magnitude in [1e-150_f64, 1.0, 1e150] {
        let source =
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0].map(|x| x * magnitude);
        // Independent quarter-turn about Z and translation, without using Rotation::apply.
        let target: Vec<_> = source
            .chunks_exact(3)
            .flat_map(|p| {
                [
                    -p[1] + 2.0 * magnitude,
                    p[0] - magnitude,
                    p[2] + 3.0 * magnitude,
                ]
            })
            .collect();
        let shape = [4, 3];
        let strides = [3, 1];
        let a = PointSet::new(
            NdView::new(&source, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        let b = PointSet::new(
            NdView::new(&target, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        let fit = estimate_rigid_transform(&a, &b).unwrap();
        assert!(fit.rms_residual / magnitude < 1e-13);
        assert!((fit.transform.rotation().apply([1.0, 0.0, 0.0]).unwrap()[1] - 1.0).abs() < 1e-14);
    }
    let data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let shape = [3, 3];
    let strides = [3, 1];
    let a =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    assert!(estimate_rigid_transform(&a, &a).is_err());
    let negative = estimate_rigid_transform_scoped(
        &a,
        &a,
        Some(&[1.0, -1.0, 1.0]),
        RigidEstimationOptions::default(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: _view_argument,
        reason: _view_reason,
        detail: _view_detail,
    } = negative.view()
    else {
        unreachable!("unexpected error kind")
    };

    assert_eq!(_view_argument, scioxide_core::argument!("weights"));
    assert!(
        estimate_rigid_transform_scoped(
            &a,
            &a,
            Some(&[1.0, f64::NAN, 1.0]),
            RigidEstimationOptions::default(),
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
}

#[test]
fn f32_alignment_with_large_translation_remains_identifiable() {
    let data = [
        0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0,
    ];
    let shape = [4, 3];
    let strides = [3, 1];
    let target: Vec<_> = data
        .chunks_exact(3)
        .flat_map(|p| [p[0] + 1000.0, p[1] - 1000.0, p[2] + 1000.0])
        .collect();
    let a =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let b =
        PointSet::new(NdView::new(&target, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let fit = estimate_rigid_transform(&a, &b).unwrap();
    assert_eq!(fit.rank, 3);
    assert!(fit.rms_residual < 0.001);
}

#[test]
fn translation_can_cancel_an_overflowing_rotated_component() {
    let rotation =
        Rotation::from_axis_angle([0.0_f64, 0.0, 1.0], core::f64::consts::FRAC_PI_4).unwrap();
    let transform = RigidTransform::new(rotation, [0.0, -1e308, 0.0]).unwrap();
    let point = [1.3e308, 1.3e308, 0.0];
    let value = transform.apply(point).unwrap();
    let points = scioxide_spatial::PointSet::from_points(core::slice::from_ref(&point)).unwrap();
    let options = scioxide_spatial::TransformOptions::new();
    let mut workspace = scioxide_spatial::TransformWorkspace::workspace(
        scioxide_spatial::TransformWorkspaceRequest::apply(1, options),
    )
    .unwrap();
    let mut output = [[0.0; 3]];
    transform
        .apply_batch_into(
            &points,
            options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output[0], value);
    let matrix = rotation.as_matrix();
    let expected = ((matrix[1][0] + matrix[1][1]) * 1.3 - 1.0) * 1e308;
    assert!((value[1] / expected - 1.0).abs() < 1e-14);
}
