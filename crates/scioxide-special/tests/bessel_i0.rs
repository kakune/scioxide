#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, RealScalar};
use scioxide_special::{BesselOptions, bessel_i_scaled, bessel_i_scaled_with_options};

fn references<S: RealScalar>() {
    // mpmath 1.3.0, 90 digits; the two largest arguments use the asymptotic
    // limit 1/sqrt(2*pi*x), whose correction is below binary64 rounding.
    for (x, expected) in [
        (0.0, 1.0),
        (1e-300, 1.0),
        (0.5, 0.645_035_270_449_150_1),
        (7.999_999_999_999_999, 0.143_431_781_856_850_32),
        (8.0, 0.143_431_781_856_850_3),
        (8.000_000_000_000_002, 0.143_431_781_856_850_3),
        (8.6, 0.138_164_247_398_068_48),
        (20.0, 0.089_780_311_884_826_02),
        (1000.0, 0.012_617_240_455_891_257),
    ] {
        let tolerance = 8.0 * S::EPSILON.into_f64() * expected;
        let original_x = x;
        let x = S::try_from_f64_lossy(x).unwrap();
        for argument in [x, -x] {
            let actual = bessel_i_scaled(S::ZERO, argument).unwrap().into_f64();
            assert!(
                (actual - expected).abs() <= tolerance,
                "x={argument:?}: {actual}"
            );
            let report =
                bessel_i_scaled_with_options(S::ZERO, argument, BesselOptions::default()).unwrap();
            let estimate = report.value().unwrap();
            if x.into_f64() == original_x {
                assert!((actual - expected).abs() <= estimate.absolute_error().into_f64());
            }
            assert_eq!(report.try_value().unwrap().into_f64(), actual);
        }
    }
}

#[test]
fn scaled_order_zero_f32_and_f64() {
    references::<f32>();
    references::<f64>();
    for (x, expected) in [
        (1e100, 3.989_422_804_014_327e-51),
        (f64::MAX, 2.975_447_459_315_899_5e-155),
    ] {
        let actual = bessel_i_scaled(0.0, x).unwrap();
        assert!((actual / expected - 1.0).abs() <= 8.0 * f64::EPSILON);
    }
}

#[test]
fn order_zero_validation_and_work_accounting() {
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(bessel_i_scaled(0.0, x).is_err());
        assert!(bessel_i_scaled(x, 1.0).is_err());
    }
    for x in [0.5, 8.0, 8.0_f64.next_up(), 1000.0] {
        let report = bessel_i_scaled_with_options(0.0, x, BesselOptions::default()).unwrap();
        assert!(report.termination().is_success());
        assert!(report.counts().iterations() >= 25);
        let options = BesselOptions::default().with_iteration_budget(Budget::ONE);
        assert!(bessel_i_scaled_with_options(0.0, x, options).is_err());
    }
}
