//! Range and work contracts complement the strict independent ULP oracle.
#![allow(clippy::unwrap_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, NumericalIssue, SciOxideError};
use scioxide_special::{
    BetaOptions, GammaOptions, ScalarRepresentation, beta, beta_with_options, binomial,
    binomial_with_options, gamma_upper, gamma_upper_with_options, ln_beta_with_options,
    ln_gamma_with_options, polygamma, polygamma_with_options,
};

#[test]
fn conditioned_kernels_do_not_allocate() {
    for (a, b) in [
        (0.25, f64::from_bits(0x4065_a558_6444_33fd)),
        (0.5, 3.5),
        (2.5, 0.5),
        (2.0, 3.0),
        (2.3, 4.7),
        (0.01, 32.5),
        (32.0, 256.0),
    ] {
        (|| {
            scioxide_special::ln_beta(a, b)?;
            scioxide_special::beta(a, b)?;
            scioxide_special::beta_inc(a, b, 0.2)?;
            scioxide_special::beta_inc(a, b, 0.001)?;
            scioxide_special::beta_inc_inv(a, b, 0.2)?;
            scioxide_special::gamma_p_inv(a, 0.2)?;
            scioxide_special::polygamma(2, a)?;
            Ok::<_, SciOxideError>(())
        })()
        .unwrap();
    }
}

#[test]
fn endpoint_series_respects_work_limits_and_reflection() {
    for (a, b, x) in [
        (0.125_f64, 0.25, 0.5),
        (2.3, 4.7, 0.05),
        (0.01, 32.5, 0.001),
        (2.5, 0.5, 0.999),
    ] {
        let options = BetaOptions::default();
        let full = scioxide_special::beta_inc_with_options(a, b, x, options).unwrap();
        let steps = full.counts().iterations();
        assert!(steps > 1);
        let limited = options.with_iteration_budget(Budget::try_from(steps - 1).unwrap());
        assert!(scioxide_special::beta_inc_with_options(a, b, x, limited).is_err());
        let admitted = options.with_iteration_budget(Budget::try_from(steps).unwrap());
        let value = scioxide_special::beta_inc(a, b, x).unwrap();
        assert_eq!(
            *scioxide_special::beta_inc_with_options(a, b, x, admitted)
                .unwrap()
                .value()
                .unwrap()
                .value(),
            value
        );
        let reflected = scioxide_special::beta_inc(b, a, 1.0 - x).unwrap();
        assert!((value + reflected - 1.0).abs() <= 32.0 * f64::EPSILON);
    }
}

#[test]
fn gamma_power_factor_preserves_subnormal_inputs() {
    let minimum = f64::from_bits(1);
    for x in [minimum, 2.0 * minimum, f64::MIN_POSITIVE] {
        let p = scioxide_special::gamma_p(1.0, x).unwrap();
        assert!((p - x).abs() <= 4.0 * minimum);
        assert_eq!(scioxide_special::gamma_q(1.0, x).unwrap(), 1.0);
        assert_eq!(scioxide_special::gamma_p(2.0, x).unwrap(), 0.0);
    }
}

#[test]
fn log_beta_resolves_quadratic_cancellation_at_unit_shapes() {
    // ln B(1-d,1+d) = zeta(2)d² + O(d⁴). Here the omitted term is
    // below 0.01 output ULP; the stored zeta(2) is independently rounded.
    for delta in [2.0_f64.powi(-30), 2.0_f64.powi(-40), f64::EPSILON] {
        let expected = 1.644_934_066_848_226_4 * delta * delta;
        let actual = scioxide_special::ln_beta(1.0 - delta, 1.0 + delta).unwrap();
        assert!((actual - expected).abs() <= 4.0 * (expected.next_up() - expected));
    }
}

fn issue(error: SciOxideError, expected: NumericalIssue) {
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { issue, .. } if issue == expected
        ),
        "{error:?}; expected {expected:?}"
    );
}

#[test]
fn range_classifications_survive_compensated_arithmetic() {
    issue(
        polygamma(1, f64::from_bits(1)).unwrap_err(),
        NumericalIssue::Overflow,
    );
    // ψ⁽⁴⁾(10¹⁰⁰) ≈ -6·10⁻⁴⁰⁰ rounds to a negative zero.
    let tail = polygamma(4, 1e100_f64).unwrap();
    assert_eq!(tail.to_bits(), (-0.0_f64).to_bits());
    issue(
        binomial::<f64>(2048, 1024).unwrap_err(),
        NumericalIssue::Overflow,
    );
    assert_eq!(beta(32.0_f32, 1000.0).unwrap().to_bits(), 0);
    let upper = gamma_upper_with_options(0.125_f32, 100.0, GammaOptions::default()).unwrap();
    assert_eq!(upper.scalar_representation(), ScalarRepresentation::Finite);
    assert_eq!(upper.try_value().unwrap().to_bits(), 0);
    assert!(f64::from(upper.value().unwrap().absolute_error()) >= 1e-46);
    assert!(
        upper
            .logarithmic()
            .unwrap()
            .value()
            .log_magnitude()
            .unwrap()
            .is_finite()
    );
    assert_eq!(gamma_upper(0.125_f32, 100.0).unwrap().to_bits(), 0);
    let upper = gamma_upper_with_options(200.0_f64, 1.0, GammaOptions::default()).unwrap();
    assert_eq!(
        upper.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(
        upper
            .logarithmic()
            .unwrap()
            .value()
            .log_magnitude()
            .unwrap()
            .is_finite()
    );
    issue(
        gamma_upper(200.0_f64, 1.0).unwrap_err(),
        NumericalIssue::Overflow,
    );
    let tiny = f64::from_bits(1);
    let complete = beta_with_options(tiny, tiny, BetaOptions::default()).unwrap();
    assert_eq!(
        complete.scalar_representation(),
        ScalarRepresentation::Overflow
    );
}

#[test]
fn fixed_kernel_admission_matches_the_declared_charge() {
    let gamma =
        |work| GammaOptions::default().with_iteration_budget(Budget::try_from(work).unwrap());
    let beta_options =
        |work| BetaOptions::default().with_iteration_budget(Budget::try_from(work).unwrap());
    assert!(scioxide_special::gamma_with_options(100.5_f64, gamma(17)).is_err());
    assert_eq!(
        scioxide_special::gamma_with_options(100.5_f64, gamma(18))
            .unwrap()
            .counts()
            .iterations(),
        18
    );
    assert!(ln_gamma_with_options(-2.5_f64, gamma(72)).is_err());
    assert_eq!(
        ln_gamma_with_options(-2.5_f64, gamma(73))
            .unwrap()
            .counts()
            .iterations(),
        73
    );
    assert!(ln_beta_with_options(0.5_f64, 2.75, beta_options(255)).is_err());
    assert_eq!(
        ln_beta_with_options(0.5_f64, 2.75, beta_options(256))
            .unwrap()
            .counts()
            .iterations(),
        256
    );
    assert!(beta_with_options(5.0_f64, 5.0, beta_options(38)).is_err());
    assert_eq!(
        beta_with_options(5.0_f64, 5.0, beta_options(39))
            .unwrap()
            .counts()
            .iterations(),
        39
    );
    let _ = publication::exhausted_counts(
        &binomial_with_options::<f64>(
            256,
            128,
            scioxide_special::CombinatorialOptions::new(Budget::try_from(38).unwrap()),
        )
        .unwrap_err(),
    );
    assert_eq!(
        binomial_with_options::<f64>(
            256,
            128,
            scioxide_special::CombinatorialOptions::new(Budget::try_from(39).unwrap())
        )
        .unwrap()
        .counts()
        .iterations(),
        39
    );
    assert!(gamma_upper_with_options(0.125_f64, 0.5, gamma(1)).is_err());
}

#[test]
fn large_order_and_negative_continuation_remain_bounded() {
    assert!(polygamma(400, 100.0_f64).unwrap().is_finite());
    let required = polygamma_with_options(400, 100.0_f64, GammaOptions::default())
        .unwrap()
        .counts()
        .iterations();
    assert!(
        polygamma_with_options(
            400,
            100.0_f64,
            GammaOptions::default().with_iteration_budget(Budget::try_from(required - 1).unwrap())
        )
        .is_err()
    );
    assert!(polygamma(4, -100000.5_f64).unwrap().is_finite());
    let value = polygamma(5, -100000.5_f64).unwrap();
    assert!((value - 15382.227097204871).abs() <= 4.0 * f64::EPSILON * value.abs());
    assert!(polygamma(2, -2.0_f64).is_err());
}

#[test]
fn negative_polygamma_references_cover_values_and_reported_errors() {
    use scioxide_core::RealScalar;

    fn check<S: RealScalar>(scalar: &str, decode: impl Fn(&str) -> S) {
        let mut rows = 0;
        for row in include_str!("data/polygamma_negative.tsv")
            .lines()
            .filter(|row| !row.starts_with('#'))
        {
            let fields: Vec<_> = row.split_whitespace().collect();
            if fields[0] != scalar {
                continue;
            }
            let n = fields[1].parse().unwrap();
            let x = decode(fields[2]);
            let expected = decode(fields[3]);
            let scale: f64 = fields[4].parse().unwrap();
            if expected.is_infinite() {
                let error = polygamma_with_options(n, x, GammaOptions::default())
                    .and_then(|report| report.try_value())
                    .unwrap_err();
                issue(error, NumericalIssue::Overflow);
                continue;
            }
            let report = polygamma_with_options(n, x, GammaOptions::default()).unwrap();
            let actual = report.try_value().unwrap();
            let spacing = (expected.abs().next_after(S::INFINITY) - expected.abs()).into_f64();
            let difference = (actual - expected).abs().into_f64();
            assert!(
                difference <= if scalar == "f64" { 4.0 } else { 1.0 } * spacing,
                "{row}: {actual:?}"
            );
            let error = report.value().unwrap().absolute_error().into_f64();
            assert!(difference <= error + spacing, "{row}: error coverage");
            let minimum = S::ZERO.next_after(S::ONE).into_f64();
            let cap = 4.0 * (S::EPSILON.into_f64() * scale).max(minimum);
            assert!(
                error <= cap * (1.0 + 8.0 * f64::EPSILON),
                "{row}: error cap {error}"
            );
            assert_eq!(
                polygamma(n, x).unwrap().into_f64().to_bits(),
                actual.into_f64().to_bits(),
                "{row}"
            );
            rows += 1;
        }
        assert!(rows > 200);
    }
    check("f64", |bits| {
        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
    });
    check("f32", |bits| {
        f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
    });
}

#[test]
fn negative_polygamma_admits_exact_work_with_reused_storage() {
    for (n, x) in [
        (5, -100000.5_f64),
        (20, -100000.25),
        (20, -1.25),
        (21, -0.25),
        (22, (-0.5_f64).next_down()),
        (64, -0.25),
        (64, -0.5),
        (64, (-0.5_f64).next_up()),
    ] {
        let complete = polygamma_with_options(n, x, GammaOptions::default()).unwrap();
        let required = complete.counts().iterations();
        assert!(required < 2048);
        for limit in [1, required - 1, required] {
            let result = polygamma_with_options(
                n,
                x,
                GammaOptions::default().with_iteration_budget(Budget::try_from(limit).unwrap()),
            );
            if limit == required {
                assert_eq!(result.unwrap(), complete);
            } else {
                assert_eq!(
                    publication::exhausted_counts(&result.unwrap_err()).iterations(),
                    limit
                );
            }
        }
    }
    for x in [0.0, -1.0, -100000000.0] {
        assert_eq!(
            polygamma(20, x).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DomainError
        );
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            polygamma(20, x).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }
    issue(
        polygamma(200, -1.25_f64).unwrap_err(),
        NumericalIssue::Overflow,
    );
    // ψ⁽⁴⁰⁰⁾(-10⁸-½) = ψ⁽⁴⁰⁰⁾(10⁸+3/2) = -400!·ζ(401, 10⁸+3/2) rounds to -0.
    assert_eq!(
        polygamma(400, -100000000.5_f64).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    issue(
        polygamma(16385, -100000.5_f64).unwrap_err(),
        NumericalIssue::Overflow,
    );
}

#[test]
fn negative_polygamma_fits_a_64_kib_stack() {
    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            for n in [1, 5, 64] {
                assert!(polygamma(n, -0.5_f64).unwrap().is_finite());
            }
            issue(
                polygamma(1000, -0.5_f64).unwrap_err(),
                NumericalIssue::Overflow,
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn large_polygamma_orders_follow_the_budget_and_scalar_range() {
    let small = GammaOptions::default().with_iteration_budget(Budget::ONE);
    let error = polygamma_with_options(20_000, 2.5_f64, small).unwrap_err();
    assert_eq!(publication::exhausted_counts(&error).iterations(), 1);
    let ample =
        GammaOptions::default().with_iteration_budget(Budget::try_from(100_000_000).unwrap());
    issue(
        polygamma_with_options(20_000, 2.5_f64, ample).unwrap_err(),
        NumericalIssue::Overflow,
    );
    assert!(matches!(
        polygamma(usize::MAX, 2.5_f64).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn integer_beta_tails_preserve_subnormal_results() {
    use scioxide_core::RealScalar;
    fn check<S: RealScalar>(scalar: &str, decode: impl Fn(&str) -> S) {
        for row in include_str!("data/beta_integer_tails.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = row.split_whitespace().collect();
            if fields[1] != scalar {
                continue;
            }
            let args: Vec<_> = fields[3]
                .split(',')
                .map(|arg| decode(arg.strip_prefix("f:").unwrap()))
                .collect();
            let (a, b, x) = (args[0], args[1], args[2]);
            let expected = decode(fields[4]);
            let (plain, report) = (|| {
                if fields[2] == "beta_inc" {
                    Ok::<_, SciOxideError>((
                        scioxide_special::beta_inc(a, b, x)?,
                        scioxide_special::beta_inc_with_options(a, b, x, BetaOptions::default())?,
                    ))
                } else {
                    Ok((
                        scioxide_special::beta_inc_q(a, b, x)?,
                        scioxide_special::beta_inc_q_with_options(a, b, x, BetaOptions::default())?,
                    ))
                }
            })()
            .unwrap();
            let actual = report.try_value().unwrap();
            let spacing = expected.next_after(S::INFINITY) - expected;
            let ulps = S::try_from_f64_lossy(if scalar == "f64" { 32.0 } else { 1.0 }).unwrap();
            assert!(
                (actual - expected).abs() <= ulps * spacing,
                "{row}: {actual:?}"
            );
            assert_eq!(actual, plain, "{row}");
            assert!(
                (actual - expected).abs() <= report.value().unwrap().absolute_error() + spacing,
                "{row}: error coverage"
            );
        }
    }
    check("f64", |bits| {
        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
    });
    check("f32", |bits| {
        f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
    });
}
