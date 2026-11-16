//! Direct identities and range boundaries for continuous laws and combinators.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{ExtendedInterval, NumericalIssue, RealScalar, SciOxideErrorView};
use scioxide_random::*;

#[test]
fn student_density_retains_heavy_tails_after_coordinate_overflow() {
    let law = StudentT::new(0.1_f64).unwrap();
    let x = 1e200;
    let log = law.log_pdf(x).unwrap();
    assert!(log.is_finite() && log.exp() > 0.0);
    assert!((law.pdf(x).unwrap() / log.exp() - 1.0).abs() < 1e-12);
    let law = StudentT::new(0.1_f32).unwrap();
    assert!(law.log_pdf(f32::MAX).unwrap().is_finite());
    let law = JohnsonSU::new(1.5e154_f64, 1.0).unwrap();
    assert!((law.log_pdf(0.0).unwrap() / -1.125e308 - 1.0).abs() < 1e-14);
}

#[test]
fn extreme_value_inverse_scales_exponential_before_rounding() {
    let law = GeneralizedExtremeValue::new(-1e308_f64).unwrap();
    let p = 0.9999999999999999_f64;
    // This interior probability genuinely overflows, and retains a numerical error.
    assert!(law.quantile(p).is_err());
    for result in [
        TukeyLambda::new(f64::from_bits(1)).map(|_| ()),
        Kappa4::new(1.0, f64::from_bits(1)).map(|_| ()),
    ] {
        assert!(matches!(
            result.unwrap_err().view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }
    let law = GeneralizedExtremeValue::new(1e308_f64).unwrap();
    // isf(p) with -ln(1-p) near one avoids a need for subnormal p spacing near exp(-1).
    assert!(law.quantile(0.5).is_err()); // rounds to the finite support endpoint
}

#[test]
fn gamma_and_beta_log_densities_preserve_scaled_coordinates() {
    for shape in [1.0_f64, 2.0] {
        let law = Gamma::new(shape, 1e300).unwrap();
        let expected = (shape - 1.0) * (-600.0 * 10.0_f64.ln()) - 300.0 * 10.0_f64.ln();
        assert!((law.log_pdf(1e-300).unwrap() - expected).abs() < 1e-11);
    }
    for shape in [1e20_f64, 1e200, 1e308] {
        let law = Beta::new(shape, shape).unwrap();
        // Stirling's formula: f(1/2) = 2 sqrt(a/pi) * (1 + O(1/a)).
        let expected = core::f64::consts::LN_2 + 0.5 * (shape.ln() - core::f64::consts::PI.ln());
        assert!((law.log_pdf(0.5).unwrap() - expected).abs() < 1e-11);
        assert!((law.pdf(0.5).unwrap() / expected.exp() - 1.0).abs() < 1e-11);
    }
}

#[test]
fn generalized_exponential_retains_small_hazard_increments() {
    let law = GeneralizedExponential::new(1.0_f64, 1e30, 1e-10).unwrap();
    let x = 1e-10;
    // Integral = a*x + b*c*x^2/2 + O(b*c^2*x^3), whose remainder is < 2e-21.
    let expected = -(x + 0.5);
    assert!((law.log_sf(x).unwrap() - expected).abs() < 1e-13);
    let law = GeneralizedExponential::new(1e-200_f64, 1e300, 1e-200).unwrap();
    let expected = -5e-301;
    assert!((law.log_sf(1e-200).unwrap() / expected - 1.0).abs() < 1e-12);
}

#[test]
fn inverse_gaussian_extremes_preserve_positive_draws_and_central_tails() {
    let law = InverseGaussian::new(1e300_f64).unwrap();
    let mut rng = SplitMix64::from_seed(7);
    for _ in 0..32 {
        let x = law.sample(&mut rng).unwrap();
        assert!(x.is_finite() && x > 0.0);
    }
    let law = InverseGaussian::new(1e-20_f64).unwrap();
    assert!((law.cdf(1e-20).unwrap() - 0.5).abs() < 1e-9);
    assert!((law.sf(1e-20).unwrap() - 0.5).abs() < 1e-9);
}

#[test]
fn normal_and_cauchy_densities_keep_representable_scaled_tails() {
    let normal = Normal::new(0.0_f64, 1e-300).unwrap();
    let x = 40e-300;
    let expected = normal.log_pdf(x).unwrap().exp();
    assert!(expected > 0.0);
    assert!((normal.pdf(x).unwrap() / expected - 1.0).abs() < 1e-12);
    let normal = Normal::new(0.0_f64, 1e-310).unwrap();
    let x = 10e-310;
    let expected = normal.log_pdf(x).unwrap().exp();
    assert!((normal.pdf(x).unwrap() / expected - 1.0).abs() < 1e-12);
    let cauchy = Cauchy::new(0.0_f64, 1e-310).unwrap();
    let expected = 1e-310 / core::f64::consts::PI;
    assert!((cauchy.pdf(1.0).unwrap() / expected - 1.0).abs() < 1e-12);
    assert!((cauchy.cdf(-1.0).unwrap() / expected - 1.0).abs() < 1e-12);
}

#[test]
fn gaussian_and_cauchy_affine_inverses_cancel_before_rounding() {
    let normal = Normal::new(-f64::MAX, f64::MAX).unwrap();
    let expected = Normal::standard().quantile(0.9).unwrap() - 1.0;
    assert!((normal.quantile(0.9).unwrap() / f64::MAX - expected).abs() < 1e-14);
    let cauchy = Cauchy::new(-f64::MAX, f64::MAX).unwrap();
    let expected = Cauchy::new(0.0, 1.0).unwrap().quantile(0.8).unwrap() - 1.0;
    assert!((cauchy.quantile(0.8).unwrap() / f64::MAX - expected).abs() < 1e-14);
    assert!((cauchy.isf(0.2).unwrap() / f64::MAX - expected).abs() < 1e-14);
}

#[test]
fn lognormal_moments_combine_scale_before_final_rounding() {
    let sigma = 2.0_f64.powi(512);
    let mu = -2.0_f64.powi(1023);
    assert_eq!(LogNormal::new(mu, sigma).unwrap().mean().unwrap(), 1.0);
    let law = LogNormal::new(500.0_f64, 1e-200).unwrap();
    let expected = (1000.0 + 2.0 * 1e-200_f64.ln()).exp();
    assert!((law.variance().unwrap() / expected - 1.0).abs() < 1e-12);
}

#[test]
fn elementary_law_scaling_preserves_finite_results() {
    let law = Gompertz::new(1e-300_f64).unwrap();
    let expected = -(720.0 + 1e-300_f64.ln()).exp();
    assert!((law.log_sf(720.0).unwrap() / expected - 1.0).abs() < 1e-12);
    let tiny = f64::from_bits(1);
    let law = Gompertz::new(tiny).unwrap();
    let expected = core::f64::consts::LN_2.ln() - tiny.ln();
    assert!((law.quantile(0.5).unwrap() - expected).abs() < 1e-12);
    assert!((law.isf(0.5).unwrap() - expected).abs() < 1e-12);
    let law = Logistic::new(-f64::MAX, f64::MAX).unwrap();
    assert!((law.quantile(0.8).unwrap() / f64::MAX - (4.0_f64.ln() - 1.0)).abs() < 1e-14);
    let law = Logistic::new(f64::MAX, f64::MAX).unwrap();
    assert!((law.isf(0.8).unwrap() / f64::MAX + (4.0_f64.ln() - 1.0)).abs() < 1e-14);
    let law = HyperbolicSecant::<f64>::new();
    let x = f64::MAX * 0.5;
    let expected = -core::f64::consts::FRAC_PI_2 * x;
    assert!((law.log_sf(x).unwrap() / expected - 1.0).abs() < 1e-14);
    assert!((law.log_pdf(x).unwrap() / expected - 1.0).abs() < 1e-14);
}

#[test]
fn transformed_densities_and_quantiles_keep_logarithmic_range() {
    let law = JohnsonSU::new(0.0_f64, 1.0).unwrap();
    let x = 1e200_f64;
    let z = x.asinh();
    let expected = -0.5 * z * z - 0.5 * core::f64::consts::TAU.ln() - x.ln();
    assert!((law.log_pdf(x).unwrap() / expected - 1.0).abs() < 1e-13);
    assert_eq!(JohnsonSU::new(0.0_f64, 0.01).unwrap().mean().unwrap(), 0.0);
    let law = TukeyLambda::new(100.0_f64).unwrap();
    let expected = (0.25_f64.powi(100) - 0.75_f64.powi(100)) / 100.0;
    let actual = law.quantile(0.25).unwrap();
    assert!((actual / expected - 1.0).abs() < 1e-13);
    assert!((law.cdf(actual).unwrap() - 0.25).abs() < 1e-13);
    let law = ExponentiatedWeibull::new(1.0_f64, 1.0).unwrap();
    assert!((law.log_sf(1000.0).unwrap() + 1000.0).abs() < 1e-10);
    let law = ExponentiatedWeibull::new(1e308_f64, 1.0).unwrap();
    let x = law.isf(1e-300).unwrap();
    assert!((x - 608.0 * 10.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn generalized_pareto_inversion_cancels_scale_before_exponentiation() {
    let law = GeneralizedPareto::new(1e308_f64).unwrap();
    let p = 7.2e-306;
    let x = law.quantile(p).unwrap();
    assert!(x.is_finite() && x > 1.0);
    assert!((law.cdf(x).unwrap() / p - 1.0).abs() < 1e-13);
    let law = GeneralizedPareto::new(1e38_f32).unwrap();
    let p = 1e-36;
    let x = law.quantile(p).unwrap();
    assert!(x.is_finite() && x > 1.0);
    assert!((law.cdf(x).unwrap() / p - 1.0).abs() < 2e-6);
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual:e} != {expected:e}"
    );
}
/// Asserts that two results are the same floating-point value, bit for bit.
fn same<S: RealScalar>(actual: S, expected: S) {
    assert_eq!(
        actual.into_f64().to_bits(),
        expected.into_f64().to_bits(),
        "{} != {}",
        actual.into_f64(),
        expected.into_f64()
    );
}
fn overflow(error: scioxide_core::SciOxideError) {
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ),
        "{error}"
    );
}

fn order_statistics<S: RealScalar>(count: usize, tolerance: f64) {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let maximum =
        OrderStatistic::new(Uniform::new(S::ZERO, S::ONE).unwrap(), count, count).unwrap();
    let minimum = OrderStatistic::new(Uniform::new(S::ZERO, S::ONE).unwrap(), 1, count).unwrap();
    let x = c(0.7);
    let expected = count as f64 * x.into_f64().ln();
    close(maximum.log_cdf(x).unwrap().into_f64(), expected, tolerance);
    close(
        minimum.log_sf(S::ONE - x).unwrap().into_f64(),
        expected,
        tolerance,
    );
    assert_eq!(maximum.cdf(x).unwrap(), S::ZERO);
    for (law, upper, point) in [(&maximum, false, x), (&minimum, true, S::ONE - x)] {
        let logp = if upper {
            law.log_sf(point)
        } else {
            law.log_cdf(point)
        }
        .unwrap();
        let report =
            invert_log_probability_with_options(law, logp, upper, &InversionOptions::default())
                .unwrap();
        assert!(report.root().is_success());
        close(report.value().into_f64(), point.into_f64(), tolerance);
        same(
            invert_log_probability(law, logp, upper).unwrap(),
            report.value(),
        );
    }
    let p = c(if S::MANTISSA_DIGITS == 24 {
        1e-30
    } else {
        1e-300
    });
    let low = maximum.quantile(p).unwrap();
    let high = minimum.isf(p).unwrap();
    close(
        low.into_f64(),
        (p.into_f64().ln() / count as f64).exp(),
        tolerance,
    );
    close(
        high.into_f64(),
        -(p.into_f64().ln() / count as f64).exp_m1(),
        tolerance,
    );
    // Interior ranks exercise both direct incomplete-beta tails as well.
    let median =
        OrderStatistic::new(Uniform::new(S::ZERO, S::ONE).unwrap(), count / 2, count).unwrap();
    for x in [0.1, 0.9] {
        assert!(median.log_cdf(c(x)).unwrap().is_finite());
        assert!(median.log_sf(c(x)).unwrap().is_finite());
    }
}
#[test]
fn order_statistic_deep_log_tails_and_inverses() {
    order_statistics::<f64>(3000, 8e-8);
    order_statistics::<f32>(300, 8e-4);
}

fn logarithmic_transforms<S: RealScalar>(far: f64, tolerance: f64) {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let exponential = Transformed::log(Exponential::new(S::ONE).unwrap()).unwrap();
    let y = c(-far);
    close(exponential.log_cdf(y).unwrap().into_f64(), -far, tolerance);
    close(exponential.log_pdf(y).unwrap().into_f64(), -far, tolerance);
    assert_eq!(exponential.cdf(y).unwrap(), S::ZERO);
    assert_eq!(exponential.pdf(y).unwrap(), S::ZERO);
    assert!(exponential.contains(y).unwrap());
    assert!(exponential.contains(c(far)).unwrap());
    assert!(!exponential.contains(S::INFINITY).unwrap());
    assert_eq!(exponential.cdf(c(far)).unwrap(), S::ONE);
    assert_eq!(exponential.sf(c(far)).unwrap(), S::ZERO);
    assert_eq!(exponential.pdf(c(far)).unwrap(), S::ZERO);
    overflow(exponential.log_sf(c(far)).unwrap_err());
    let report =
        invert_log_probability_with_options(&exponential, y, false, &InversionOptions::default())
            .unwrap();
    assert!(report.root().is_success());
    close(report.value().into_f64(), -far, tolerance);
    same(
        invert_log_probability(&exponential, y, false).unwrap(),
        report.value(),
    );
    let small = c(if S::MANTISSA_DIGITS == 24 {
        1e-30
    } else {
        1e-300
    });
    let x = exponential.isf(small).unwrap();
    close(x.into_f64(), (-small.into_f64().ln()).ln(), tolerance);
    close(
        exponential.log_sf(x).unwrap().into_f64(),
        small.into_f64().ln(),
        tolerance,
    );

    let gamma = Transformed::log(Gamma::new(S::TWO, S::ONE).unwrap()).unwrap();
    let lowest = c(if S::MANTISSA_DIGITS == 24 {
        -f64::from(f32::MAX)
    } else {
        -f64::MAX
    });
    assert_eq!(gamma.log_sf(lowest).unwrap(), S::ZERO);
    assert_eq!(gamma.sf(lowest).unwrap(), S::ONE);
    assert_eq!(gamma.pdf(lowest).unwrap(), S::ZERO);
    overflow(gamma.log_cdf(lowest).unwrap_err());
    close(
        gamma.log_cdf(y).unwrap().into_f64(),
        -2.0 * far - core::f64::consts::LN_2,
        tolerance,
    );
    close(gamma.log_pdf(y).unwrap().into_f64(), -2.0 * far, tolerance);
    let options = InversionOptions::default();
    let report =
        invert_log_probability_with_options(&gamma, c(-2.0 * far), false, &options).unwrap();
    assert!(report.root().is_success());
    same(
        invert_log_probability(&gamma, c(-2.0 * far), false).unwrap(),
        report.value(),
    );
    close(
        report.value().into_f64(),
        -far + 0.5 * core::f64::consts::LN_2,
        tolerance,
    );

    let normal = Normal::new(S::ZERO, S::ONE).unwrap();
    let lognormal = Transformed::log(LogNormal::new(S::ZERO, S::ONE).unwrap()).unwrap();
    let exp = Transformed::exp(&normal).unwrap();
    let recovered = Transformed::log(&exp).unwrap();
    for x in [-far, far] {
        let x = c(x);
        close(
            lognormal.log_pdf(x).unwrap().into_f64(),
            normal.log_pdf(x).unwrap().into_f64(),
            tolerance,
        );
        close(
            recovered.log_pdf(x).unwrap().into_f64(),
            normal.log_pdf(x).unwrap().into_f64(),
            tolerance,
        );
        close(
            lognormal.log_cdf(x).unwrap().into_f64(),
            normal.log_cdf(x).unwrap().into_f64(),
            tolerance,
        );
        close(
            recovered.log_sf(x).unwrap().into_f64(),
            normal.log_sf(x).unwrap().into_f64(),
            tolerance,
        );
    }
    let histogram = Histogram::from_masses(&[S::ZERO, S::ONE], &[S::ONE]).unwrap();
    let loghist = Transformed::log(&histogram).unwrap();
    close(loghist.log_cdf(y).unwrap().into_f64(), -far, tolerance);
    close(loghist.log_pdf(y).unwrap().into_f64(), -far, tolerance);
    assert!(loghist.contains(y).unwrap());
    assert!(!loghist.contains(S::ONE).unwrap());
    assert!(
        !loghist
            .support_in(ExtendedInterval::new(Some(y), Some(y + S::ONE)).unwrap())
            .unwrap()
            .is_empty()
    );

    for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
        assert!(exponential.log_cdf(bad).is_err());
        assert!(exponential.log_pdf(bad).is_err());
    }
}
#[test]
fn log_transforms_beyond_exponential_range() {
    logarithmic_transforms::<f64>(800.0, 8e-8);
    logarithmic_transforms::<f32>(120.0, 8e-4);
}

#[test]
fn log_transforms_preserve_scale_before_exponentiation() {
    let gamma = Transformed::log(Gamma::new(2.0_f64, 1e300).unwrap()).unwrap();
    let exponential = Transformed::log(Exponential::new(1e-300_f64).unwrap()).unwrap();
    let y = 800.0;
    let z = (y - 1e300_f64.ln()).exp();
    close(exponential.log_sf(y).unwrap(), -z, 8e-8);
    close(gamma.log_sf(y).unwrap(), -z + z.ln_1p(), 8e-8);
    let generalized = Transformed::log(GeneralizedGamma::new(2.0_f64, -1.0).unwrap()).unwrap();
    close(
        generalized.log_sf(y).unwrap(),
        -1600.0 - core::f64::consts::LN_2,
        8e-8,
    );
}

#[test]
fn interior_endpoints_are_typed_errors_and_limits_are_exact() {
    let law = Trapezoid::new(0.2_f64, 0.8).unwrap();
    assert_eq!(law.isf(0.0).unwrap(), 1.0);
    assert_eq!(law.isf(1.0).unwrap(), 0.0);
    assert!(matches!(
        law.isf(1e-300).unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Underflow,
            ..
        }
    ));
    for shape in [0.5_f64, 1.0, 2.0, 3.0] {
        let law = TukeyLambda::new(shape).unwrap();
        let expected = if shape < 1.0 {
            0.0
        } else if shape == 1.0 {
            0.5
        } else {
            1.0
        };
        close(law.pdf(1.0 / shape).unwrap(), expected, 1e-14);
        close(law.pdf(-1.0 / shape).unwrap(), expected, 1e-14);
    }
}

#[test]
fn pearson_near_zero_skew_keeps_tails_and_inverses_consistent() {
    for skew in [-2e-5_f64, -1.6e-5, -1e-5, 0.0, 1e-5, 1.6e-5, 2e-5] {
        let law = Pearson3::new(skew).unwrap();
        let narrow = Pearson3::new(skew as f32).unwrap();
        assert_eq!(law.quantile(0.0).unwrap(), law.isf(1.0).unwrap());
        assert_eq!(law.quantile(1.0).unwrap(), law.isf(0.0).unwrap());
        for p in [1e-12_f64, 1e-4, 0.25, 0.5] {
            for upper in [false, true] {
                let x = if upper { law.isf(p) } else { law.quantile(p) }.unwrap();
                let log = if upper { law.log_sf(x) } else { law.log_cdf(x) }.unwrap();
                assert!(
                    (log - p.ln()).abs() < 8e-8,
                    "skew={skew} p={p} upper={upper} log={log}"
                );
                let q = p as f32;
                let x = if upper {
                    narrow.isf(q)
                } else {
                    narrow.quantile(q)
                }
                .unwrap();
                let log = if upper {
                    narrow.log_sf(x)
                } else {
                    narrow.log_cdf(x)
                }
                .unwrap();
                assert!(
                    (f64::from(log) - f64::from(q).ln()).abs()
                        < 8.0 * f64::from(f32::EPSILON).sqrt()
                );
            }
        }
    }
}

#[test]
fn logarithmic_capability_is_object_safe_and_preserves_tiny_gamma_shape() {
    let base = Gamma::new(1e-20_f64, 1.0).unwrap();
    let provider: &dyn LogCoordinateLaw<f64> = &base;
    let law = Transformed::log(provider).unwrap();
    // ln Gamma(1+a) = -EulerGamma*a + O(a^2), and exp(-800) is negligible.
    let expected = (1e-20_f64 * (800.0 - 0.577_215_664_901_532_9)).ln();
    close(law.log_sf(-800.0).unwrap(), expected, 1e-14);
    assert!(law.sf(-800.0).unwrap() > 0.0);
    assert_eq!(law.cdf(800.0).unwrap(), 1.0);
    assert_eq!(law.sf(800.0).unwrap(), 0.0);
    assert_eq!(law.pdf(800.0).unwrap(), 0.0);
}

#[test]
fn finite_law_coordinates_do_not_become_special_function_inputs() {
    let mielke = Mielke::new(1.0_f64, f64::MAX).unwrap();
    close(mielke.pdf(1e-10).unwrap(), 1.0, 8e-8);
    // For x<1 the correction to F(x)=x is smaller than exp(-1e308).
    close(mielke.log_cdf(1e-10).unwrap(), 1e-10_f64.ln(), 8e-8);
    close(mielke.log_sf(1e-10).unwrap(), (-1e-10_f64).ln_1p(), 8e-8);
    for skew in [-0.01_f64, 0.01] {
        let law = Pearson3::new(skew).unwrap();
        let x = skew.signum() * f64::MAX;
        assert_eq!(law.cdf(x).unwrap(), if skew > 0.0 { 1.0 } else { 0.0 });
        assert_eq!(law.sf(x).unwrap(), if skew > 0.0 { 0.0 } else { 1.0 });
        assert_eq!(law.pdf(x).unwrap(), 0.0);
        let large = if skew > 0.0 {
            law.log_cdf(x)
        } else {
            law.log_sf(x)
        };
        assert_eq!(large.unwrap(), 0.0);
        let small = if skew > 0.0 {
            law.log_sf(x)
        } else {
            law.log_cdf(x)
        };
        overflow(small.unwrap_err());
    }
}
