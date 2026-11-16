//! Contracts of the shared defining-sum and numerical comparison helpers.
#[path = "support/numerics.rs"]
mod numerics;

use numerics::{
    assert_close, assert_complex_close, assert_complex_l2_close, assert_real_close, direct_dft,
    direct_sum,
};
use scioxide_core::Complex;

#[test]
fn defining_sums_preserve_sign_weights_and_unnormalized_inverse() {
    let input = [
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let expected = [
        Complex::new(1.0, 0.0),
        Complex::new(0.0, -1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, 1.0),
    ];
    let forward = direct_dft(&input, false);
    assert_complex_close(&forward, &expected, Complex::norm, |_| 1e-14);
    let recovered = direct_dft(&forward, true);
    assert_complex_close(&recovered, &input.map(|z| z * 4.0), Complex::norm, |_| {
        1e-14
    });
    let weighted = direct_sum(
        &[Complex::new(3.0_f32, -2.0), Complex::new(7.0, 1.0)],
        |j| (0.0, if j == 0 { 2.0 } else { 0.0 }),
    );
    assert_eq!(weighted, Complex::new(6.0, -4.0));
}

#[test]
fn comparison_accepts_exact_and_boundary_errors_and_rejects_invalid_bounds() {
    assert_close(0.0, 0.0, "exact");
    assert_close(0.25, 0.25, "boundary");
    for (error, bound) in [
        (0.5, 0.25),
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (0.0, f64::NAN),
        (0.0, f64::INFINITY),
        (-1.0, 1.0),
        (0.0, -1.0),
    ] {
        assert!(std::panic::catch_unwind(|| assert_close(error, bound, "invalid")).is_err());
    }
}

#[test]
fn slice_adapters_reject_missing_outputs_and_nonfinite_components() {
    assert!(
        std::panic::catch_unwind(|| assert_real_close(&[1.0], &[] as &[f64], |_| 1.0)).is_err()
    );
    assert!(
        std::panic::catch_unwind(|| assert_complex_close(
            &[Complex::new(1.0, 0.0)],
            &[] as &[Complex<f64>],
            Complex::norm,
            |_| 1.0
        ))
        .is_err()
    );
    for value in [f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| assert_complex_close(
                &[Complex::new(value, 0.0)],
                &[Complex::new(0.0, 0.0)],
                numerics::component_max,
                |_| 1.0
            ))
            .is_err()
        );
    }
}

#[test]
fn relative_l2_comparison_preserves_width_scale_and_zero_reference() {
    let expected = [Complex::new(4.0_f64, 0.0)];
    let actual = [Complex::new(4.5_f32, 0.0)];
    assert_complex_l2_close(&actual, &expected, 0.125);
    assert_complex_l2_close(
        &[Complex::new(0.0_f32, 0.0)],
        &[Complex::new(0.0_f64, 0.0)],
        0.0,
    );
    assert!(
        std::panic::catch_unwind(|| assert_complex_l2_close(&actual, &expected, 0.0625)).is_err()
    );
    for bound in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| assert_complex_l2_close(&expected, &expected, bound))
                .is_err()
        );
    }
    for value in [1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| assert_complex_l2_close(
                &[Complex::new(value, 0.0)],
                &[Complex::new(0.0, 0.0)],
                1.0
            ))
            .is_err()
        );
        if !value.is_finite() {
            assert!(
                std::panic::catch_unwind(|| assert_complex_l2_close(
                    &[Complex::new(0.0, 0.0)],
                    &[Complex::new(value, 0.0)],
                    1.0
                ))
                .is_err()
            );
        }
    }
    assert!(
        std::panic::catch_unwind(|| assert_complex_l2_close(&actual, &[] as &[Complex<f64>], 1.0))
            .is_err()
    );
}
