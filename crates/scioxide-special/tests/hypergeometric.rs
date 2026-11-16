//! Independent certified hypergeometric envelopes and public resource contracts.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, Complex, NumericScalar, RealScalar};
use scioxide_special::*;

fn identity() -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(71),
        scioxide_core::algorithm_id!("hypergeometric", "scoped_test", 1),
        0,
        0,
    )
}

#[test]
fn prepared_and_scoped_values_execute() {
    use scioxide_core::{ExecutionContext, SolveBudgets, SolveState};
    let plan = HypergeometricFunction::pfq(
        &[0.5, 0.75],
        &[1.5],
        false,
        HypergeometricOptions::default(),
    )
    .unwrap();
    let mut scratch = plan.workspace().unwrap();
    let result = plan
        .evaluate(0.3, HypergeometricDerivative::Numerator(0))
        .unwrap();
    assert!(result.termination().is_success());
    let reused = plan
        .evaluate_scoped(
            0.3,
            HypergeometricDerivative::Numerator(0),
            &mut scratch,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(reused, result);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let scoped = plan
        .evaluate_scoped(
            0.3,
            HypergeometricDerivative::Numerator(0),
            &mut scratch,
            &mut context,
        )
        .unwrap();
    assert_eq!(scoped, result);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        scoped.counts().iterations()
    );
}

#[test]
fn negative_ray_and_large_parameter_witnesses() {
    let options = HypergeometricOptions::default();
    for (b, z, reference) in [
        (-19.802211175572197, -318.995659244806, 1651515.2879527784),
        (-17.069635694660732, -129.98018120346, 10172.891365657824),
        (-29.02540336164347, -148.69546826899585, 342.9402891081915),
        (1.5, -1e7, -7.983750671202225e-5),
    ] {
        let result = hyp0f1_with_options(b, z, options).unwrap();
        let value = result.value().unwrap();
        assert!(
            (value.value() - reference).abs() < 1e-12 * reference.abs(),
            "{b} {z}: {value:?}"
        );
        assert!(
            (value.value() - reference).abs()
                <= value.absolute_error() + f64::EPSILON * reference.abs()
        );
        assert_eq!(
            value.reliability(),
            scioxide_core::EstimateReliability::StatisticallyCalibrated
        );
    }
    for (a, b, z, reference) in [
        (
            -59.89853467789528,
            43.61942446668316,
            124.02672634092988,
            -2.062377650605285e-8,
        ),
        (
            58.44983284865394,
            -51.556347933025016,
            -79.68783521348287,
            3.812637712924816e14,
        ),
        (
            -55.69152478791907,
            46.96735055804905,
            149.46902513878484,
            -2.251113338883704e-5,
        ),
        (
            58.554839471781605,
            -31.101251890567163,
            -118.32271243736997,
            1775.3039370392157,
        ),
        (
            -38.65691735818022,
            38.564559037049435,
            57.822138295981176,
            2.7012427367240515e-11,
        ),
    ] {
        let result = hyp1f1_with_options(a, b, z, options).unwrap();
        let actual = result.value().unwrap();
        assert!(
            (actual.value() - reference).abs() < 1e-12 * reference.abs(),
            "{a} {b} {z}: {actual:?}"
        );
        assert!(
            (actual.value() - reference).abs()
                <= actual.absolute_error() + f64::EPSILON * reference.abs()
        );
    }
    let result = hyp2f1_with_options(0.5, 0.5, 1.5, -1e200, options).unwrap();
    let reference = 2.309_516_564_799_645e-98;
    assert!((result.try_value().unwrap() - reference).abs() < 1e-13 * reference);
}

#[test]
fn negative_ray_2f0_bounds_cover_large_parameters() {
    // mpmath 1.4.1 at 50 digits. Large |ab z| amplifies the rounding of
    // -1/z and of 1+a-b in the U connection.
    for (a, b, z, reference) in [
        (200.0, 200.0, -1e-5, 0.670_856_066_198_247_2),
        (1e3, 1e3, -1e-12, 0.999_999_000_000_501),
        (1e5, 1e5, -1e-15, 0.999_990_000_050_000_8),
        (1e8, 1e8, -1e-20, 0.999_900_004_999_833_4),
        (
            -1_988.252_768_905_773_3,
            14.617_847_449_100_601,
            -1.711_692_897_834_417_8e-5,
            1.658_771_521_990_772_6,
        ),
        (
            9_810.076_394_699_62,
            -0.175_575_784_704_228_2,
            -3.570_589_522_019_738_5e-6,
            1.006_063_063_873_921_6,
        ),
    ] {
        let result = hyp2f0_with_options(a, b, z, HypergeometricOptions::default()).unwrap();
        let value = result.value().unwrap();
        assert!(
            (value.value() - reference).abs() <= value.absolute_error() + f64::EPSILON * reference,
            "{a} {b} {z}: {value:?}"
        );
    }
}

#[test]
fn negative_ray_2f0_accepts_subnormal_arguments() {
    for z in [-f64::from_bits(1), -1e-320, -1e-310] {
        let result = hyp2f0_with_options(1.0, -0.5, z, HypergeometricOptions::default()).unwrap();
        let value = result.value().unwrap();
        assert_eq!(*value.value(), 1.0);
        assert!(value.absolute_error() >= 0.5 * z.abs());
        let complex = hyp2f0(
            Complex::new(1.0, 0.0),
            Complex::new(-0.5, 0.0),
            Complex::new(z, 0.0),
        )
        .unwrap();
        assert_eq!(complex, Complex::new(1.0, 0.0));
    }
    // Parameters too large for the truncated series fail instead of guessing.
    assert!(hyp2f0(1e150, 1e150, -1e-320).is_err());
}

#[test]
fn resolved_paths_preserve_allocation_and_shared_budgets() {
    use scioxide_core::{ExecutionContext, SolveBudgets, SolveState};
    for (a, b, z) in [
        (&[][..], &[-19.802211175572197][..], -318.995659244806),
        (&[][..], &[0.5][..], -1e300),
        (
            &[-59.89853467789528][..],
            &[43.61942446668316][..],
            124.02672634092988,
        ),
        (&[0.5, 0.5][..], &[1.5][..], -1e200),
    ] {
        let plan =
            HypergeometricFunction::pfq(a, b, false, HypergeometricOptions::default()).unwrap();
        let mut scratch = plan.workspace().unwrap();
        let result = plan.evaluate(z, HypergeometricDerivative::None).unwrap();
        let used = result.counts().iterations();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let scoped = plan
            .evaluate_scoped(
                z,
                HypergeometricDerivative::None,
                &mut scratch,
                &mut context,
            )
            .unwrap();
        assert_eq!(result, scoped);
        assert_eq!(context.counts().nested().iterations(), used);
        let short = HypergeometricFunction::pfq(
            a,
            b,
            false,
            HypergeometricOptions::new(Budget::try_from(used - 1).unwrap()),
        )
        .unwrap();
        assert!(short.evaluate(z, HypergeometricDerivative::None,).is_err());
        let exact = HypergeometricFunction::pfq(
            a,
            b,
            false,
            HypergeometricOptions::new(Budget::try_from(used).unwrap()),
        )
        .unwrap();
        assert_eq!(
            exact.evaluate(z, HypergeometricDerivative::None,).unwrap(),
            result
        );
    }
}

#[test]
fn unresolved_parameter_directions_are_numerical_failures() {
    for (a, b, z, direction) in [
        (
            &[][..],
            -19.802211175572197,
            -318.995659244806,
            HypergeometricDerivative::Denominator(0),
        ),
        (
            &[-59.89853467789528][..],
            43.61942446668316,
            124.02672634092988,
            HypergeometricDerivative::Numerator(0),
        ),
    ] {
        let plan =
            HypergeometricFunction::pfq(a, &[b], false, HypergeometricOptions::default()).unwrap();
        let error = plan.evaluate(z, direction).unwrap_err();
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::LossOfSignificance,
                    ..
                }
            ),
            "{error}"
        );
    }
}

#[test]
fn signed_zero_selects_complex_cut_sides() {
    let r = |x| Complex::new(x, 0.0);
    for family in [HypergeometricFamily::U, HypergeometricFamily::Pfq] {
        let (upper, lower) = if family == HypergeometricFamily::U {
            (
                hyperu(r(0.5), r(1.25), Complex::new(-2., 0.)).unwrap(),
                hyperu(r(0.5), r(1.25), Complex::new(-2., -0.)).unwrap(),
            )
        } else {
            (
                hyp2f1(r(0.5), r(0.75), r(1.5), Complex::new(2., 0.)).unwrap(),
                hyp2f1(r(0.5), r(0.75), r(1.5), Complex::new(2., -0.)).unwrap(),
            )
        };
        assert!((upper - lower.conj()).norm() < 1e-10, "{upper:?} {lower:?}");
        assert!(upper.im.abs() > 0.01);
    }
}

#[test]
fn transformed_cut_lips_and_strict_accuracy_are_explicit() {
    let r = |x| Complex::new(x, 0.0);
    let upper = hyp2f1(r(0.5), r(0.75), r(0.5), Complex::new(2., 0.)).unwrap();
    let lower = hyp2f1(r(0.5), r(0.75), r(0.5), Complex::new(2., -0.)).unwrap();
    assert!((upper - lower.conj()).norm() < 1e-12);
    assert!(upper.im > 0.0);
    let upper = hyp2f0(r(0.5), r(1.25), Complex::new(0.5, 0.)).unwrap();
    let lower = hyp2f0(r(0.5), r(1.25), Complex::new(0.5, -0.)).unwrap();
    assert!((upper - lower.conj()).norm() < 1e-10, "{upper:?} {lower:?}");
    assert_eq!(
        hyp1f1_with_options(1., 1., 1., HypergeometricOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    assert_eq!(hyperu(0., 0., 0.).unwrap(), 1.);
    assert!(hyp2f1(0.5, 0.75, 2.5, 1.).is_ok());
}

fn complex(text: &str) -> Complex<f64> {
    let (a, b) = text.split_once(',').unwrap();
    Complex::new(a.parse().unwrap(), b.parse().unwrap())
}
fn params(text: &str) -> Vec<Complex<f64>> {
    if text == "-" {
        vec![]
    } else {
        text.split(';').map(complex).collect()
    }
}
fn selector(text: &str) -> HypergeometricDerivative {
    match text {
        "none" => HypergeometricDerivative::None,
        "z" => HypergeometricDerivative::Argument,
        "a0" => HypergeometricDerivative::Numerator(0),
        "b0" => HypergeometricDerivative::Denominator(0),
        _ => panic!("invalid selector"),
    }
}

fn fixtures<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut failures = Vec::new();
    for line in include_str!("data/hypergeometric.tsv")
        .lines()
        .filter(|x| !x.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0].parse::<u32>().unwrap() != if S::MANTISSA_DIGITS == 24 { 32 } else { 64 } {
            continue;
        }
        let convert = |z: Complex<f64>| {
            Complex::new(
                S::try_from_f64_lossy(z.re).unwrap(),
                S::try_from_f64_lossy(z.im).unwrap(),
            )
        };
        let a: Vec<_> = params(f[4]).into_iter().map(convert).collect();
        let b: Vec<_> = params(f[5]).into_iter().map(convert).collect();
        let z = convert(complex(f[6]));
        let options = HypergeometricOptions::default();
        let plan = if f[1] == "u" {
            HypergeometricFunction::u(a[0], b[0], options)
        } else {
            HypergeometricFunction::pfq(&a, &b, f[2] == "1", options)
        }
        .unwrap();
        let result = plan.evaluate(z, selector(f[3]));
        let r = match result {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{line}: {e:?}"));
                continue;
            }
        };
        if !r.termination().is_success() {
            failures.push(format!(
                "{line}: {:?}, value {:?}",
                r.termination(),
                r.value()
            ));
            continue;
        }
        let value = if f[3] == "none" {
            r.value()
        } else {
            r.derivative()
        };
        let actual = if let Some(value) = value {
            [
                value.value().re.try_to_f64_exact().unwrap(),
                value.value().im.try_to_f64_exact().unwrap(),
            ]
        } else if f[3] == "none" {
            let log = r.logarithmic().unwrap().value();
            let magnitude = log
                .log_magnitude()
                .unwrap()
                .try_to_f64_exact()
                .unwrap()
                .exp();
            [
                log.scaled_phase()
                    .round()
                    .unwrap()
                    .re
                    .try_to_f64_exact()
                    .unwrap()
                    * magnitude,
                log.scaled_phase()
                    .round()
                    .unwrap()
                    .im
                    .try_to_f64_exact()
                    .unwrap()
                    * magnitude,
            ]
        } else {
            failures.push(format!("{line}: missing derivative"));
            continue;
        };
        let bounds: Vec<f64> = f[7..].iter().map(|x| x.parse().unwrap()).collect();
        let scale = bounds.iter().map(|x| x.abs()).fold(0., f64::max);
        let tolerance = if S::MANTISSA_DIGITS == 24 {
            2e-5_f64.max(1e-3 * scale)
        } else {
            5e-12_f64.max(5e-9 * scale)
        };
        for k in 0..2 {
            if actual[k] < bounds[2 * k] - tolerance || actual[k] > bounds[2 * k + 1] + tolerance {
                failures.push(format!(
                    "{line}: got {actual:?}, error estimate {:?}",
                    value.map(|v| v.absolute_error())
                ));
                break;
            }
        }
        // The reported error reaches the certified enclosure, up to rounding.
        if let Some(value) = value {
            let gap = |k: usize| {
                (bounds[2 * k] - actual[k])
                    .max(actual[k] - bounds[2 * k + 1])
                    .max(0.0)
            };
            let reported = value.absolute_error().try_to_f64_exact().unwrap();
            let rounding =
                2.0 * S::EPSILON.try_to_f64_exact().unwrap() * actual[0].hypot(actual[1]);
            if gap(0).hypot(gap(1)) > reported + rounding {
                failures.push(format!(
                    "{line}: got {actual:?}, reported error {reported:e}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} fixture failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
#[test]
fn certified_f64() {
    fixtures::<f64>();
}
#[test]
fn certified_f32() {
    fixtures::<f32>();
}

#[test]
fn budgets_poles_and_domains() {
    assert!(hyp_pfq(&[0.5, 0.75, 1.25], &[2.], 0.0).is_err());
    let finite = HypergeometricFunction::pfq(
        &[-50.0_f64, 0.5, 0.75],
        &[],
        false,
        HypergeometricOptions::default(),
    )
    .unwrap();
    let finite_result = finite
        .evaluate(1e-6, HypergeometricDerivative::None)
        .unwrap();
    assert_eq!(finite_result.algorithm().method(), "terminating_polynomial");
    assert_eq!(finite_result.counts().iterations(), 200);

    // Positive large denominators make the entire tail smaller than one ULP.
    assert_eq!(hyp0f1(1e100_f64, 1.).unwrap(), 1.);
    assert_eq!(hyp1f1(1., 1e100_f64, 1.).unwrap(), 1.);
    assert_eq!(hyp0f1(1e30_f32, 1.).unwrap(), 1.);

    let polynomial = HypergeometricFunction::pfq(
        &[-1.0_f64],
        &[-2.0],
        false,
        HypergeometricOptions::default(),
    )
    .unwrap();
    let b_derivative = polynomial
        .evaluate(0.3, HypergeometricDerivative::Denominator(0))
        .unwrap();
    assert!((b_derivative.derivative().unwrap().value() - 0.3 / 4.0).abs() < 1e-15);
    {
        let outcome = &(polynomial.evaluate(0.3, HypergeometricDerivative::Numerator(0)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Pole,
                ..
            }
        ));
    }

    let r = hyp1f1_with_options(
        0.5,
        1.5,
        5.,
        HypergeometricOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    assert!(hyp_pfq(&[0.5, 0.75, 1.25], &[2.], 0.2).is_err());
    assert!(hyp_pfq(&[0.5, 0.75, 1.25], &[2., 3.], 1.).is_err());
    assert!(hyp1f1(1., 0., 0.).is_err());
    assert!(hyp1f1(-2., -3., 2.).is_ok());
    assert!(hyp1f1(-2., -2., 2.).is_err());
    assert!(hyp2f1_regularized(0.5, 0.75, 0., 0.3).is_ok());
}

#[test]
fn scaled_u_and_kummer_survive_scalar_range() {
    let options = HypergeometricOptions::default();
    let wide = hyp1f1_with_options(1.0_f64, 1., 3e9, options).unwrap();
    assert!(wide.termination().is_success());
    assert!(
        wide.scaled()
            .unwrap()
            .value()
            .try_projective()
            .unwrap()
            .exponent()
            > i64::from(i32::MAX)
    );

    let r = hyperu_with_options(100.0_f64, 101., 1e20, HypergeometricOptions::default()).unwrap();
    assert!(r.termination().is_success());
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Finite);
    assert_eq!(r.try_value().unwrap().to_bits(), 0.0_f64.to_bits());
    assert!(r.scaled().is_some());
    assert!(
        (r.logarithmic().unwrap().value().log_magnitude().unwrap() + 100.0 * 1e20_f64.ln()).abs()
            < 1e-10
    );
    let r = hyp1f1_with_options(1., 1., 1000., HypergeometricOptions::default()).unwrap();
    assert!(r.termination().is_success());
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(r.scaled().is_some());
}

#[test]
fn compensated_real_derivatives_match_independent_references() {
    for line in include_str!("data/hypergeometric_derivatives.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let f: Vec<_> = line.split('\t').collect();
        let scalar = f[0];
        let index: usize = f[2].parse().unwrap();
        let from_bits = |text: &str| {
            let bits = u64::from_str_radix(text, 16).unwrap();
            if scalar == "f64" {
                f64::from_bits(bits)
            } else {
                f64::from(f32::from_bits(bits as u32))
            }
        };
        let args: Vec<_> = f[3].split(',').map(|x| from_bits(&x[2..])).collect();
        let n = args.len();
        let selector = if index == n - 1 {
            HypergeometricDerivative::Argument
        } else if index == n - 2 {
            HypergeometricDerivative::Denominator(0)
        } else {
            HypergeometricDerivative::Numerator(index)
        };
        let reference = from_bits(f[4]);
        macro_rules! evaluate {
            ($scalar:ty) => {{
                let args: Vec<_> = args.iter().map(|&x| x as $scalar).collect();
                let plan = HypergeometricFunction::pfq(
                    &args[..n - 2],
                    &args[n - 2..n - 1],
                    false,
                    HypergeometricOptions::default(),
                )
                .unwrap();
                let result = plan.evaluate(args[n - 1], selector).unwrap();
                assert!(result.termination().is_success(), "{line}: {result:?}");
                let derivative = result.derivative().unwrap();
                let actual = f64::from(*derivative.value());
                let tolerance = 256.0 * f64::from(<$scalar>::EPSILON) * reference.abs();
                assert!(
                    (actual - reference).abs() <= tolerance,
                    "{line}: {actual} vs {reference}"
                );
                assert!(derivative.absolute_error().is_finite());
            }};
        }
        if scalar == "f64" {
            evaluate!(f64);
        } else {
            evaluate!(f32);
        }
    }
}

#[test]
fn compensated_paths_share_exact_work_limits_and_cancellation() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ExecutionContext, SolveBudgets, SolveState,
    };
    for (a, b, z) in [
        (&[][..], 1.25, -256.),
        (&[0.5][..], 1.5, 5.),
        (&[2.][..], 3., -256.),
        (&[0.25, 1.75][..], 2.5, -256.),
        (&[0.5, 0.5][..], 1.5, 1. - 2.0_f64.powi(-40)),
    ] {
        let plan =
            HypergeometricFunction::pfq(a, &[b], false, HypergeometricOptions::default()).unwrap();
        let mut scratch = plan.workspace().unwrap();
        let complete = plan
            .evaluate(z, HypergeometricDerivative::Argument)
            .unwrap();
        assert!(complete.termination().is_success());
        let reused = plan
            .evaluate_scoped(
                z,
                HypergeometricDerivative::Argument,
                &mut scratch,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(reused, complete);
        let limited = HypergeometricFunction::pfq(
            a,
            &[b],
            false,
            HypergeometricOptions::default().with_iteration_budget(Budget::try_from(17).unwrap()),
        )
        .unwrap();
        let result = limited
            .evaluate(z, HypergeometricDerivative::Argument)
            .unwrap_err();
        let result_counts = publication::exhausted_counts(&result);
        assert_eq!(result_counts.iterations(), 17);
        // A parent limit is scope-local; the plan budget bounds the child epoch.
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(17).unwrap());
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let result = limited
            .evaluate_scoped(
                z,
                HypergeometricDerivative::Argument,
                &mut scratch,
                &mut context,
            )
            .unwrap_err();
        let result_counts = publication::exhausted_counts(&result);
        assert_eq!(result_counts.iterations(), 17);
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(context.counts().nested().iterations(), 17);
        let unlimited = plan
            .evaluate_scoped(
                z,
                HypergeometricDerivative::Argument,
                &mut scratch,
                &mut context,
            )
            .unwrap();
        assert_eq!(unlimited, complete);

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&cancellation),
                identity(),
                &mut budgets,
            )
            .unwrap();
        assert!(matches!(
            plan.evaluate_scoped(
                z,
                HypergeometricDerivative::None,
                &mut scratch,
                &mut context
            )
            .unwrap_err()
            .view(),
            scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
        ));
        assert_eq!(context.counts().total().unwrap().iterations(), 0);
    }
}

#[test]
fn exact_origin_survives_an_unrepresentable_derivative() {
    let b = f64::from_bits(1);
    assert_eq!(hyp0f1(b, 0.).unwrap(), 1.);
    let result = hyp0f1_with_options(b, 0., HypergeometricOptions::default()).unwrap();
    assert!(result.termination().is_success());
    assert_eq!(*result.value().unwrap().value(), 1.);
    assert!(result.derivative().is_none());
}

#[test]
fn gauss_endpoint_preserves_parameter_derivatives_when_argument_derivative_diverges() {
    let plan = HypergeometricFunction::pfq(
        &[0.5_f64, 0.5],
        &[2.],
        false,
        HypergeometricOptions::default(),
    )
    .unwrap();
    // Gauss's formula gives F=4/pi. The derivatives follow independently
    // from psi(1), psi(3/2), psi(2) and cancel Euler's constant exactly.
    let value = 4.0 / core::f64::consts::PI;
    for (selector, derivative) in [
        (
            HypergeometricDerivative::Numerator(0),
            value * (2.0 - 2.0 * core::f64::consts::LN_2),
        ),
        (
            HypergeometricDerivative::Numerator(1),
            value * (2.0 - 2.0 * core::f64::consts::LN_2),
        ),
        (
            HypergeometricDerivative::Denominator(0),
            value * (4.0 * core::f64::consts::LN_2 - 3.0),
        ),
    ] {
        let result = plan.evaluate(1., selector).unwrap();
        assert!(result.termination().is_success());
        assert!((result.value().unwrap().value() - value).abs() <= 16.0 * f64::EPSILON * value);
        assert!(
            (result.derivative().unwrap().value() - derivative).abs()
                <= 16.0 * f64::EPSILON * derivative.abs()
        );
    }
    {
        let outcome = &(plan.evaluate(1., HypergeometricDerivative::Argument));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::OutsideDomain,
                ..
            }
        ));
    }
}

#[test]
fn complex_stable_paths_preserve_allocation_and_exact_work_limits() {
    use scioxide_core::ExecutionContext;
    let r = |x| Complex::new(x, 0.0);
    for (a, b, z) in [
        (&[r(0.5)][..], r(1.5), Complex::new(1.0, 20.0)),
        (&[r(0.5)][..], r(1.5), Complex::new(1.0, 200.0)),
        (&[r(0.5), r(0.75)][..], r(1.5), Complex::new(1.0, 1e-12)),
        (&[r(0.5), r(0.75)][..], r(1.25), Complex::new(1.0, 1e-12)),
    ] {
        let plan = HypergeometricFunction::pfq(a, &[b], false, Default::default()).unwrap();
        let mut scratch = plan.workspace().unwrap();
        let complete = plan
            .evaluate(z, HypergeometricDerivative::Argument)
            .unwrap();
        assert!(complete.termination().is_success());
        let reused = plan
            .evaluate_scoped(
                z,
                HypergeometricDerivative::Argument,
                &mut scratch,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(reused, complete);
        let count = complete.counts().iterations();
        let exact = HypergeometricFunction::pfq(
            a,
            &[b],
            false,
            HypergeometricOptions::default()
                .with_iteration_budget(Budget::try_from(count).unwrap()),
        )
        .unwrap();
        assert_eq!(
            complete,
            exact
                .evaluate(z, HypergeometricDerivative::Argument)
                .unwrap()
        );
        let limited = HypergeometricFunction::pfq(
            a,
            &[b],
            false,
            HypergeometricOptions::default()
                .with_iteration_budget(Budget::try_from(count - 1).unwrap()),
        )
        .unwrap();
        let error = limited
            .evaluate(z, HypergeometricDerivative::Argument)
            .unwrap_err();
        assert_eq!(
            publication::exhausted_counts(&error).iterations(),
            count - 1
        );
    }
}

/// Checks the plan-less `ₚF_q` forms against `hyp_pfq` and the stored function on one row.
fn plan_less_pfq_row<T: NumericScalar>(a: &[T], b: &[T], z: T) {
    use scioxide_core::{ExecutionContext, SolveBudgets, SolveState};
    let options = HypergeometricOptions::default();
    let plain = hyp_pfq(a, b, z);
    let detailed = hyp_pfq_with_options(a, b, z, options);
    let stored = HypergeometricFunction::pfq(a, b, false, options)
        .and_then(|f| f.evaluate(z, HypergeometricDerivative::None));
    let scoped = hyp_pfq_scoped(a, b, z, options, ExecutionContext::sequential());
    match (&plain, &detailed) {
        (Ok(value), Ok(result)) => {
            let projected = result.try_value().unwrap();
            assert_eq!(
                (
                    value.real_component().into_f64().to_bits(),
                    value.imaginary_component().into_f64().to_bits()
                ),
                (
                    projected.real_component().into_f64().to_bits(),
                    projected.imaginary_component().into_f64().to_bits()
                ),
                "{a:?} {b:?} {z:?}"
            );
            assert!(result.derivative().is_none());
            assert_eq!(stored.as_ref().unwrap(), result);
            assert_eq!(scoped.as_ref().unwrap(), result);
        }
        (Err(plain), Err(detailed)) => {
            assert_eq!(plain.kind(), detailed.kind());
            assert_eq!(plain.kind(), stored.unwrap_err().kind());
            assert_eq!(plain.kind(), scoped.unwrap_err().kind());
            return;
        }
        (Err(plain), Ok(result)) => {
            assert_eq!(
                format!("{plain:?}"),
                format!("{:?}", result.try_value().unwrap_err())
            );
            return;
        }
        (Ok(_), Err(error)) => panic!("{a:?} {b:?} {z:?}: {error}"),
    }
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let nested = hyp_pfq_scoped(a, b, z, options, &mut context).unwrap();
    assert_eq!(&nested, detailed.as_ref().unwrap());
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        nested.counts().iterations()
    );
}

fn plan_less_pfq<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let width = if S::MANTISSA_DIGITS == 24 { 32 } else { 64 };
    let mut rows = 0;
    for line in include_str!("data/hypergeometric.tsv")
        .lines()
        .filter(|x| !x.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0].parse::<u32>().unwrap() != width || f[1..4] != ["pfq", "0", "none"] {
            continue;
        }
        let convert = |z: Complex<f64>| {
            Complex::new(
                S::try_from_f64_lossy(z.re).unwrap(),
                S::try_from_f64_lossy(z.im).unwrap(),
            )
        };
        let a: Vec<_> = params(f[4]).into_iter().map(convert).collect();
        let b: Vec<_> = params(f[5]).into_iter().map(convert).collect();
        let z = convert(complex(f[6]));
        plan_less_pfq_row(&a, &b, z);
        if a.iter().chain(&b).chain([&z]).all(|x| x.im == S::ZERO) {
            let real = |x: &[Complex<S>]| x.iter().map(|x| x.re).collect::<Vec<_>>();
            plan_less_pfq_row(&real(&a), &real(&b), z.re);
        }
        rows += 1;
    }
    assert!(rows > 50, "{rows} pFq rows");
}

#[test]
fn plan_less_pfq_forms_match_the_plain_value_and_the_stored_function() {
    plan_less_pfq::<f64>();
    plan_less_pfq::<f32>();
}

#[test]
fn plan_less_pfq_budgets_poles_and_cancellation() {
    use scioxide_core::{CancellationToken, ExecutionContext, SolveBudgets, SolveState};
    let (a, b) = ([0.5_f64, 0.75, 1.25], [1.5, 2.0]);
    let error =
        hyp_pfq_with_options(&a, &b, 0.5, HypergeometricOptions::new(Budget::ONE)).unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ),
        "{error}"
    );
    for error in [
        hyp_pfq_with_options(&[1.0_f64], &[-2.0], 0.5, Default::default()).unwrap_err(),
        hyp_pfq_scoped(
            &[1.0_f64],
            &[-2.0],
            0.5,
            Default::default(),
            ExecutionContext::sequential(),
        )
        .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Pole,
                ..
            }
        ));
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let error = hyp_pfq_scoped(&a, &b, 0.5, Default::default(), &mut context).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
}
