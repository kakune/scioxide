//! Time scaling and independent circumcenter identities.
#![allow(clippy::unwrap_used)]
use scioxide_core::{ExecutionContext, NdView, RealScalar, ResidualTolerance, StridedLayout};
use scioxide_spatial::{Delaunay, PointSet, Rotation, RotationSpline, RotationSplineOptions};

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

fn identities<S: RealScalar>() {
    let mut reference = Vec::<Vec<f64>>::new();
    let times = [0., 0.7, 2., 3., 5.].map(scalar::<S>);
    let rotations = [
        [0.1, 0.2, 0.3],
        [0.4, -0.1, 0.6],
        [-0.7, 0.6, 0.9],
        [0.8, 0.2, -0.1],
        [0.4, 1., 0.6],
    ]
    .map(|r| Rotation::from_rotation_vector(r.map(scalar::<S>)).unwrap());
    for scale in [1., 0.0001, 10000.] {
        let times = times.map(|t| t * scalar::<S>(scale));
        let result = RotationSpline::build(
            &times,
            &rotations,
            RotationSplineOptions::new()
                .with_max_iterations(64)
                .with_continuity_tolerance(
                    ResidualTolerance::new(S::ZERO, scalar::<S>(64.) * S::EPSILON).unwrap(),
                ),
            ExecutionContext::sequential(),
        )
        .unwrap();

        let spline = result.spline.unwrap();
        for (index, time) in [0.2, 1., 2.4, 4.].into_iter().enumerate() {
            let sample = spline.evaluate(scalar::<S>(time * scale)).unwrap();
            let values: Vec<_> = sample
                .rotation
                .as_matrix()
                .into_iter()
                .flatten()
                .map(RealScalar::into_f64)
                .chain(sample.angular_velocity.map(|v| v.into_f64() * scale))
                .chain(
                    sample
                        .angular_acceleration
                        .map(|v| v.into_f64() * scale * scale),
                )
                .collect();
            if scale == 1.0 {
                reference.push(values);
            } else {
                for (actual, expected) in values.iter().zip(&reference[index]) {
                    assert!(
                        (actual - expected).abs()
                            < 4096.0 * S::EPSILON.into_f64() * expected.abs().max(1.0),
                        "scale={scale} actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    let coordinates = [0., 0., 1., 0., 1., 1., 0., 1., 0.25, 0.5, 1., 1.].map(scalar::<S>);
    let shape = [6, 2];
    let strides = [2, 1];
    let points = PointSet::new(
        NdView::new(&coordinates, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
    )
    .unwrap();
    let delaunay = Delaunay::new(&points).unwrap();
    for i in 0..delaunay.simplex_count() {
        let mut center = [S::ZERO; 2];
        delaunay.circumcenter_into(i, &mut center).unwrap();
        let mut radius = None::<f64>;
        for &vertex in delaunay.simplex(i).unwrap() {
            let row = delaunay.representatives()[vertex];
            let squared = (0..2)
                .map(|axis| {
                    let delta = coordinates[2 * row + axis].into_f64() - center[axis].into_f64();
                    delta * delta
                })
                .sum::<f64>();
            if let Some(expected) = radius {
                assert!(
                    (squared - expected).abs() < 128.0 * S::EPSILON.into_f64() * expected.max(1.0)
                );
            }
            radius = Some(squared);
        }
    }
}

#[test]
fn numerical_identities() {
    identities::<f32>();
    identities::<f64>();
}
