//! High-precision regression grids for the gamma-beta-cdfs work specialized kernels.

use scioxide_special::{beta, beta_inc, gamma, gamma_p, gamma_q};

#[test]
fn unit_shape_beta_preserves_overflow_representations() {
    use scioxide_core::{NumericalIssue, RealScalar, SciOxideErrorView};
    use scioxide_special::{ScalarRepresentation, beta_with_options};

    fn check<S: RealScalar>() {
        for (a, b) in [(S::ONE, S::MIN_SUBNORMAL), (S::MIN_SUBNORMAL, S::ONE)] {
            let report = beta_with_options(a, b, Default::default()).unwrap();
            assert_eq!(
                report.scalar_representation(),
                ScalarRepresentation::Overflow
            );
            assert!(report.termination().is_success());
            assert!(report.value().is_none());
            assert!(report.scaled().is_some());
            let logarithm = report.logarithmic().unwrap();
            let actual = logarithm.value().log_magnitude().unwrap();
            let expected = -S::MIN_SUBNORMAL.ln();
            assert!((actual - expected).abs() <= logarithm.absolute_error());
            for error in [beta(a, b).unwrap_err(), report.try_value().unwrap_err()] {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ));
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn unit_shape_beta_reports_positive_subnormal_rounding_error() {
    use scioxide_core::RealScalar;
    use scioxide_special::beta_with_options;

    fn check<S: RealScalar>() {
        for (a, b) in [(S::ONE, S::MAX), (S::MAX, S::ONE)] {
            let report = beta_with_options(a, b, Default::default()).unwrap();
            let value = report.value().unwrap();
            assert!(*value.value() > S::ZERO && *value.value() < S::MIN_POSITIVE);
            assert!(value.absolute_error() >= S::MIN_SUBNORMAL);
            assert!(report.scaled().unwrap().absolute_error() > S::ZERO);
            assert_eq!(beta(a, b).unwrap(), report.try_value().unwrap());
        }
    }
    check::<f32>();
    check::<f64>();
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
    );
}

#[test]
fn positive_integer_and_half_integer_gamma_lattices() {
    assert_eq!(gamma(1.0_f64).unwrap(), 1.0);
    assert_eq!(gamma(5.0_f64).unwrap(), 24.0);
    assert_eq!(gamma(10.0_f32).unwrap(), 362_880.0);
    // Correct rounding of sqrt(real pi), from the 160/240-digit oracle.
    // Taking sqrt of the already-rounded f64 PI loses one ULP here.
    assert_close(gamma(0.5_f64).unwrap(), 1.772_453_850_905_516, 0.0);
    assert_close(gamma(4.5_f64).unwrap(), 11.631_728_396_567_448, 2e-14);
}

#[test]
fn small_integer_and_half_integer_beta_lattices() {
    assert_close(beta(5.0_f64, 5.0).unwrap(), 1.0 / 630.0, 2e-18);
    assert_close(beta(0.5_f64, 0.5).unwrap(), std::f64::consts::PI, 0.0);
    assert_close(beta(0.5_f64, 3.5).unwrap(), 0.981_747_704_246_810_3, 2e-15);
    assert!((beta(5.0_f32, 5.0).unwrap() - 1.0 / 630.0).abs() <= 2e-9);
}

#[test]
fn incomplete_gamma_regions_match_mpmath_160_digit_oracles() {
    let cases = [
        (2.3, 0.4, 0.034_412_031_982_617_9),
        (2.3, 8.0, 0.994_985_876_608_115_7),
        (5.0, 5.0, 0.559_506_714_934_787_6),
        (4.5, 6.0, 0.786_690_694_916_583_5),
        (100.0, 100.0, 0.513_298_798_279_148_7),
        (1_000.0, 950.0, 0.055_054_686_230_738_03),
        (1_000.0, 1_050.0, 0.941_328_888_622_681_9),
        (10_000.0, 10_000.0, 0.501_329_808_339_955_2),
    ];

    cases.into_iter().for_each(|(a, x, expected_p)| {
        let p = gamma_p(a, x).unwrap();
        let q = gamma_q(a, x).unwrap();
        assert_close(p, expected_p, 2e-12);
        assert_close(q, 1.0 - expected_p, 2e-12);
        assert_close(p + q, 1.0, 2e-15);
    });
}

#[test]
fn incomplete_beta_regions_match_mpmath_160_digit_oracles() {
    let cases = [
        (5.0, 5.0, 0.5, 0.5),
        (15.0, 6.0, 0.7, 0.416_370_829_447_481_2),
        (2.3, 4.7, 0.2, 0.249_581_471_733_733),
        (2.3, 4.7, 0.8, 0.996_508_098_475_118_4),
        (0.5, 3.5, 0.4, 0.932_416_706_176_63),
        (100.0, 120.0, 0.45, 0.447_801_230_147_707_6),
    ];

    cases.into_iter().for_each(|(a, b, x, expected)| {
        assert_close(beta_inc(a, b, x).unwrap(), expected, 2e-12);
    });
}

#[test]
fn regularized_functions_preserve_symmetry_and_monotonicity() {
    [0.01, 0.1, 0.3, 0.5, 0.8, 0.99].into_iter().for_each(|x| {
        let direct = beta_inc(2.3_f64, 4.7, x).unwrap();
        let reflected = beta_inc(4.7_f64, 2.3, 1.0 - x).unwrap();
        assert_close(direct + reflected, 1.0, 3e-14);
    });

    let gamma_values = [0.01, 0.1, 1.0, 5.0, 20.0].map(|x| gamma_p(3.5_f64, x).unwrap());
    assert!(gamma_values.windows(2).all(|pair| pair[0] <= pair[1]));

    let beta_values = [0.01, 0.1, 0.3, 0.7, 0.99].map(|x| beta_inc(2.3_f64, 4.7, x).unwrap());
    assert!(beta_values.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn f32_specialized_regions_stay_within_published_budgets() {
    let p = gamma_p(100.0_f32, 100.0).unwrap();
    assert!((p - 0.513_298_8).abs() <= 5e-6);
    let beta_value = beta_inc(2.3_f32, 4.7, 0.2).unwrap();
    assert!((beta_value - 0.249_581_47).abs() <= 5e-6);
    assert!((p + gamma_q(100.0_f32, 100.0).unwrap() - 1.0).abs() <= 2e-6);
}
