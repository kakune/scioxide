#![allow(clippy::unwrap_used)]
use scioxide_core::{RealScalar, Tolerance};
use scioxide_spatial::{EulerFrame, EulerSequence, Rotation, RotationAxis};
fn value<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn tolerance<S: RealScalar>() -> Tolerance<S> {
    Tolerance::new(value::<S>(8.0) * S::EPSILON, value::<S>(8.0) * S::EPSILON).unwrap()
}
fn close<S: RealScalar>(a: Rotation<S>, b: Rotation<S>) {
    let error = a.inverse().compose(b).unwrap().as_axis_angle().1.into_f64();
    assert!(
        error < 200.0 * S::EPSILON.into_f64(),
        "rotation error {error}"
    );
}
fn cases<S: RealScalar>() {
    use RotationAxis::*;
    for a in [X, Y, Z] {
        for b in [X, Y, Z] {
            for c in [X, Y, Z] {
                if a == b || b == c {
                    continue;
                }
                for frame in [EulerFrame::Intrinsic, EulerFrame::Extrinsic] {
                    let seq = EulerSequence::new([a, b, c], frame).unwrap();
                    for middle in [
                        -2.0,
                        -1.0,
                        0.0,
                        0.7,
                        1.0,
                        2.0,
                        core::f64::consts::PI,
                        core::f64::consts::FRAC_PI_2,
                        -core::f64::consts::FRAC_PI_2,
                    ] {
                        let r = Rotation::from_euler(
                            seq,
                            [value::<S>(0.3), value::<S>(middle), value::<S>(-0.8)],
                        )
                        .unwrap();
                        let e = r.as_euler(seq, tolerance()).unwrap();
                        let singular = if a == c {
                            middle == 0.0 || middle == core::f64::consts::PI
                        } else {
                            middle.abs() == core::f64::consts::FRAC_PI_2
                        };
                        assert_eq!(e.gimbal_lock, singular);
                        close(r, Rotation::from_euler(seq, e.angles).unwrap());
                        if e.gimbal_lock {
                            assert_eq!(e.angles[2], S::ZERO);
                        }
                        close(
                            r,
                            Rotation::from_matrix(r.as_matrix(), tolerance()).unwrap(),
                        );
                        close(
                            r,
                            Rotation::from_quaternion(r.as_quaternion().map(|x| -x)).unwrap(),
                        );
                        close(
                            r,
                            Rotation::from_rotation_vector(r.as_rotation_vector()).unwrap(),
                        );
                        close(r.compose(r.inverse()).unwrap(), Rotation::identity());
                    }
                }
            }
        }
    }
    for angle in [0.0, 1e-20, 1e-8, 0.2, core::f64::consts::PI] {
        let r = Rotation::from_axis_angle(
            [value::<S>(1.0), value::<S>(2.0), value::<S>(3.0)],
            value::<S>(angle),
        )
        .unwrap();
        close(
            r,
            Rotation::from_rotation_vector(r.as_rotation_vector()).unwrap(),
        );
        let v = r
            .apply([value::<S>(1.0), value::<S>(0.0), value::<S>(0.0)])
            .unwrap();
        assert!(
            (v.iter().map(|&x| x * x).sum::<S>() - S::ONE).abs() < value::<S>(20.0) * S::EPSILON
        );
    }
    assert!(Rotation::<S>::from_quaternion([S::ZERO; 4]).is_err());
    assert!(Rotation::<S>::from_quaternion([S::NAN; 4]).is_err());
    assert!(
        Rotation::from_matrix(
            [
                [S::ONE, S::ZERO, S::ZERO],
                [S::ZERO, S::ONE, S::ZERO],
                [S::ZERO, S::ZERO, -S::ONE]
            ],
            tolerance()
        )
        .is_err()
    );
    let r = Rotation::from_quaternion([S::MAX; 4]).unwrap();
    close(r, Rotation::from_quaternion([S::ONE; 4]).unwrap());
}
#[test]
fn f64_rotations() {
    cases::<f64>();
}
#[test]
fn f32_rotations() {
    cases::<f32>();
}

#[test]
fn failures_preserve_input_and_overflow_provenance() {
    use scioxide_core::NonFiniteOrigin;
    assert!(
        matches ! (& (Rotation :: < f64 > :: from_quaternion ([f64 :: NAN ; 4])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { origin : NonFiniteOrigin :: Input , .. }))
    );
    assert!(
        matches ! (& (Rotation :: < f64 > :: from_quaternion ([0.0 ; 4])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    assert!(
        matches ! (& (Rotation :: identity () . apply ([f64 :: INFINITY , 0.0 , 0.0])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { origin : NonFiniteOrigin :: Input , .. }))
    );
    assert!(
        matches ! (& (Rotation :: from_rotation_vector ([f64 :: MAX ; 3])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { .. }))
    );
    assert!(
        matches ! (& (Rotation :: from_matrix ([[f64 :: MAX ; 3] ; 3] , Tolerance :: from_sqrt_epsilon ())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
}

#[test]
fn identity_preserves_independent_coordinate_ranges() {
    let point = [1e308_f64, 1e-100, 0.0];
    assert_eq!(Rotation::identity().apply(point).unwrap(), point);
    let point = [f32::MAX, 1e-10, 0.0];
    assert_eq!(Rotation::identity().apply(point).unwrap(), point);
}
