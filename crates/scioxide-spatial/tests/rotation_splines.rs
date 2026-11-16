#![allow(clippy::unwrap_used)]
use scioxide_core::{RealScalar, ResidualTolerance};
use scioxide_spatial::*;
fn value<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn distance<S: RealScalar>(a: Rotation<S>, b: Rotation<S>) -> f64 {
    a.inverse().compose(b).unwrap().as_axis_angle().1.into_f64()
}
fn cases<S: RealScalar>() {
    let times = [0.0, 0.7, 2.0, 3.0, 5.0].map(value::<S>);
    let rotations = [
        [0.1, 0.2, 0.3],
        [0.4, -0.1, 0.6],
        [-0.7, 0.6, 0.9],
        [0.8, 0.2, -0.1],
        [0.4, 1.0, 0.6],
    ]
    .map(|r| Rotation::from_rotation_vector(r.map(value::<S>)).unwrap());
    let options = RotationSplineOptions::new()
        .with_max_iterations(64)
        .with_continuity_tolerance(
            ResidualTolerance::new(value::<S>(64.0) * S::EPSILON, value::<S>(64.0) * S::EPSILON)
                .unwrap(),
        );
    let result = RotationSpline::with_options(&times, &rotations, options).unwrap();
    assert_eq!(
        result.termination,
        RotationSplineTermination::Converged,
        "residual {}",
        result.continuity_residual.into_f64()
    );
    let spline = result.spline.unwrap();
    let slerp = Slerp::new(&times, &rotations).unwrap();
    for (&time, &r) in times.iter().zip(&rotations) {
        assert_eq!(spline.evaluate(time).unwrap().rotation, r);
        assert_eq!(slerp.evaluate(time).unwrap(), r);
    }
    let delta = if S::EPSILON.into_f64() > 1e-10 {
        0.001
    } else {
        1e-5
    };
    for &time in &times[1..times.len() - 1] {
        let left = spline.evaluate(time - value::<S>(delta)).unwrap();
        let right = spline.evaluate(time + value::<S>(delta)).unwrap();
        for i in 0..3 {
            assert!(
                (left.angular_velocity[i].into_f64() - right.angular_velocity[i].into_f64()).abs()
                    < 30.0 * delta
            );
            assert!(
                (left.angular_acceleration[i].into_f64()
                    - right.angular_acceleration[i].into_f64())
                .abs()
                    < 100.0 * delta
            );
        }
    }
    for t in [0.2, 1.0, 2.4, 4.0] {
        let mid = spline.evaluate(value::<S>(t)).unwrap();
        let left = spline.evaluate(value::<S>(t - delta)).unwrap();
        let right = spline.evaluate(value::<S>(t + delta)).unwrap();
        let velocity = left
            .rotation
            .inverse()
            .compose(right.rotation)
            .unwrap()
            .as_rotation_vector();
        for (i, velocity) in velocity.iter().enumerate() {
            assert!(
                (velocity.into_f64() / (2.0 * delta) - mid.angular_velocity[i].into_f64()).abs()
                    < 10.0 * delta
            );
            assert!(
                ((right.angular_velocity[i].into_f64() - left.angular_velocity[i].into_f64())
                    / (2.0 * delta)
                    - mid.angular_acceleration[i].into_f64())
                .abs()
                    < 10.0 * delta
            );
        }
    }
    let result =
        RotationSpline::with_options(&times, &rotations, options.with_max_iterations(0)).unwrap();
    assert_eq!(
        result.termination,
        RotationSplineTermination::IterationLimit
    );
    assert!(result.spline.is_none());
    assert!(spline.evaluate(value::<S>(-0.1)).is_err());
    assert!(slerp.evaluate(S::NAN).is_err());
    assert!(Slerp::new(&[value::<S>(0.0); 2], &rotations[..2]).is_err());
    let two = RotationSpline::with_options(&times[..2], &rotations[..2], options)
        .unwrap()
        .spline
        .unwrap();
    let arc = Slerp::new(&times[..2], &rotations[..2]).unwrap();
    for fraction in [0.0, 0.2, 0.5, 0.8, 1.0] {
        let t = times[1] * value::<S>(fraction);
        let sample = two.evaluate(t).unwrap();
        assert!(
            distance(sample.rotation, arc.evaluate(t).unwrap()) < 100.0 * S::EPSILON.into_f64()
        );
        assert!(
            sample
                .angular_acceleration
                .iter()
                .all(|x| x.into_f64().abs() < 100.0 * S::EPSILON.into_f64())
        );
    }
}
#[test]
fn f64_splines() {
    cases::<f64>()
}
#[test]
fn f32_splines() {
    cases::<f32>()
}

#[test]
fn repeated_and_half_turn_orientations() {
    for angle in [
        0.0,
        1e-14,
        core::f64::consts::PI - 1e-12,
        core::f64::consts::PI,
    ] {
        let r = Rotation::from_axis_angle([0.0, 1.0, 0.0], angle).unwrap();
        let opposite = Rotation::from_quaternion(r.as_quaternion().map(|x| -x)).unwrap();
        let times = [0.0, 1.0, 2.0];
        let rotations = [Rotation::identity(), r, opposite];
        let arc = Slerp::new(&times, &rotations).unwrap();
        assert!((arc.evaluate(0.25).unwrap().as_axis_angle().1 - angle * 0.25).abs() < 1e-14);
        assert!(distance(arc.evaluate(1.5).unwrap(), r) < 1e-14);
        let spline =
            RotationSpline::with_options(&times, &rotations, RotationSplineOptions::default())
                .unwrap()
                .spline
                .unwrap();
        assert!(distance(spline.evaluate(1.0).unwrap().rotation, r) < 1e-14);
    }
}

// Continuity acceptance scales with the data, not with the time unit. Constant
// rates have almost zero acceleration, so only the magnitude of the terms
// that cancel bounds the rounding noise of the jump.
fn constant_rate<S: RealScalar>(dt: f64, rate: f64) -> RotationSplineResult<S> {
    let times: Vec<S> = (0..6).map(|i| value::<S>(f64::from(i) * dt)).collect();
    let rotations: Vec<_> = times
        .iter()
        .map(|&t| {
            Rotation::from_axis_angle([S::ZERO, S::ZERO, S::ONE], t * value::<S>(rate)).unwrap()
        })
        .collect();
    RotationSpline::with_options(&times, &rotations, RotationSplineOptions::new()).unwrap()
}
#[test]
fn continuity_acceptance_is_invariant_to_time_units() {
    for (dt, rate) in [(1e-3, 5.0), (1e-2, 5.0), (1e-3, 0.5), (1.0, 0.5)] {
        let result = constant_rate::<f32>(dt, rate);
        assert_eq!(
            result.termination,
            RotationSplineTermination::Converged,
            "f32 dt={dt}"
        );
        let spline = result.spline.unwrap();
        let sample = spline.evaluate(value::<f32>(2.5 * dt)).unwrap();
        let speed = sample
            .angular_velocity
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            (f64::from(speed) / rate - 1.0).abs() < 1e-3,
            "f32 dt={dt}: {speed}"
        );
    }
    for (dt, rate) in [(1e-9, 5.0), (1e-6, 5.0), (1.0, 0.5), (1e6, 1e-6)] {
        let result = constant_rate::<f64>(dt, rate);
        assert_eq!(
            result.termination,
            RotationSplineTermination::Converged,
            "f64 dt={dt}"
        );
        let sample = result.spline.unwrap().evaluate(2.5 * dt).unwrap();
        let speed = sample
            .angular_velocity
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!((speed / rate - 1.0).abs() < 1e-9, "f64 dt={dt}: {speed}");
    }
}
#[test]
fn knot_length_mismatch_is_a_dimension_error() {
    let error = RotationSpline::with_options(
        &[0.0_f64, 1.0, 2.0],
        &[Rotation::identity(); 2],
        RotationSplineOptions::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    let error = Slerp::new(&[0.0_f64, 1.0, 2.0], &[Rotation::identity(); 2]).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    let error = Slerp::new(&[0.0_f64], &[Rotation::identity()]).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
}
#[test]
fn unrepresentable_orthogonality_residual_is_invalid_input() {
    let tolerance = scioxide_core::Tolerance::from_sqrt_epsilon();
    for matrix in [
        [[1e200_f64, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[1., 0., 0.], [0., f64::MAX, f64::MAX], [0., 0., 1.]],
    ] {
        let error = Rotation::from_matrix(matrix, tolerance).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
    let error = Rotation::from_matrix(
        [[3e19_f32, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        scioxide_core::Tolerance::from_sqrt_epsilon(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
}
