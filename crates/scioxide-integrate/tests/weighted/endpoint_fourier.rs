//! Regressions for endpoint offsets, Fourier tails and moment-limited refinement.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, ErrorTolerance, Interval, NormSelection, RealScalar};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, DoubleExponentialDomain, DoubleExponentialIntegrator,
    DoubleExponentialPoint, FourierDomain, FourierTailOptions, FourierTailPlan, FourierWeight,
    IntegrationTermination, LevelOptions, SubdivisionOptions, WeightedQuadratureOptions,
    WeightedQuadraturePlan,
};

fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn tolerance<S: RealScalar>(t: f64) -> ErrorTolerance<S> {
    ErrorTolerance::scalar(s(t), s(t), NormSelection::Infinity).unwrap()
}

fn endpoint_cases<S: RealScalar + Default>(t: f64) {
    for a in [0.0, 1.0] {
        for upper in [false, true] {
            for exponent in [0.5, 0.9, 0.99] {
                let a = s::<S>(a);
                let b = a + S::ONE;
                let de = DoubleExponentialIntegrator::new(
                    DoubleExponentialDomain::Finite {
                        interval: Interval::new(a, b).unwrap(),
                        singular_points: &[],
                    },
                    LevelOptions::<S>::default().with_tolerance(tolerance(t)),
                )
                .unwrap();
                let endpoint = if upper { b } else { a };
                let exponent = s::<S>(exponent);
                let expected = 1.0 / (1.0 - exponent.into_f64());
                let (value, report) = de
                    .integrate(|point: DoubleExponentialPoint<S>| {
                        let distance = match point.nearest_endpoint() {
                            Some((location, distance)) if location == endpoint => distance,
                            _ => (point.coordinate() - endpoint).abs(),
                        };
                        assert!(distance > S::ZERO);
                        Ok(distance.powf(-exponent))
                    })
                    .unwrap();
                let actual = (value.into_f64() - expected).abs();
                let rounding = 8.0 * S::EPSILON.into_f64() * expected;
                assert!(
                    report
                        .error_estimate()
                        .is_some_and(|estimate| actual <= estimate.into_f64() + rounding),
                    "a={a:?} upper={upper} exponent={exponent:?} value={value:?} error={actual} {report:?}"
                );
                if report.is_success() {
                    assert!(
                        actual <= t * (1.0 + expected) + rounding,
                        "{value:?} {report:?}"
                    );
                }
                if S::EPSILON.into_f64() < 1e-10 && exponent == s(0.9) {
                    assert!(report.is_success(), "{value:?} {report:?}");
                }
            }
        }
    }
}

#[test]
fn endpoint_singularities_binary64() {
    endpoint_cases::<f64>(1e-10);
}
#[test]
fn endpoint_singularities_binary32() {
    endpoint_cases::<f32>(1e-4);
}

fn fourier_cases<S: RealScalar + Default>(t: f64) {
    let options = FourierTailOptions::new(
        SubdivisionOptions::default()
            .with_tolerance(tolerance::<S>(t))
            .with_max_subdivisions(Budget::try_from(512).unwrap()),
    );
    for weight in [FourierWeight::Sine, FourierWeight::Cosine] {
        let plan = FourierTailPlan::new(
            FourierDomain::Upper { lower: S::ZERO },
            S::ONE,
            weight,
            options.clone(),
        )
        .unwrap();
        for case in 0..4 {
            let (value, report) = plan
                .integrate(|x: S| {
                    Ok(match case {
                        0 => S::ONE + s::<S>(10.0) * (-x).exp(),
                        1 => S::ONE,
                        2 => x.sqrt(),
                        _ => S::TWO + x.cos(),
                    })
                })
                .unwrap();
            assert!(!report.is_success(), "case={case} {value:?} {report:?}");
        }
    }
    for (case, weight, expected) in [
        (0, FourierWeight::Sine, 0.5),
        (
            1,
            FourierWeight::Cosine,
            core::f64::consts::FRAC_PI_2 / core::f64::consts::E,
        ),
        (2, FourierWeight::Sine, core::f64::consts::FRAC_PI_2.sqrt()),
    ] {
        let plan = FourierTailPlan::new(
            FourierDomain::Upper { lower: S::ZERO },
            S::ONE,
            weight,
            options.clone(),
        )
        .unwrap();
        let (value, report) = plan
            .integrate(|x: S| {
                Ok(match case {
                    0 => (-x).exp(),
                    1 => S::ONE / (S::ONE + x * x),
                    _ => S::ONE / x.sqrt(),
                })
            })
            .unwrap();
        assert!(report.is_success(), "case={case} {value:?} {report:?}");
        assert!(
            (value.into_f64() - expected).abs() <= t * (1.0 + expected),
            "case={case} {value:?} {report:?}"
        );
    }
}
#[test]
fn fourier_tails_binary64() {
    fourier_cases::<f64>(1e-8);
}
#[test]
fn fourier_tails_binary32() {
    fourier_cases::<f32>(1e-4);
}

#[test]
fn weak_fourier_regions_refine_beyond_prepared_moments() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let amplitude = |x: f64| Ok(1.0 / ((x - 0.3).powi(2) + 1e-8));
    let (reference, _) = AdaptiveIntegrator::new(
        AdaptiveRule::GaussKronrod,
        SubdivisionOptions::default().with_tolerance(tolerance(1e-12)),
    )
    .unwrap()
    .integrate(|x: f64| Ok((50.0 * x).cos() * amplitude(x)?), interval)
    .unwrap();
    for depth in [6, 10] {
        let plan = WeightedQuadraturePlan::fourier(
            interval,
            &[],
            50.0,
            FourierWeight::Cosine,
            WeightedQuadratureOptions::default()
                .with_moment_levels(Budget::try_from(depth).unwrap()),
        )
        .unwrap();
        let (value, report) = plan.integrate(&amplitude).unwrap();
        assert!(report.is_success(), "depth={depth} {value} {report:?}");
        assert!((value - reference).abs() <= report.error_estimate().unwrap());
    }
    let amplitude = |x: f64| Ok(1.0 / (1.0 + 1000.0 * (x - 0.4).powi(2)));
    let (reference, _) = AdaptiveIntegrator::new(
        AdaptiveRule::GaussKronrod,
        SubdivisionOptions::default().with_tolerance(tolerance(1e-12)),
    )
    .unwrap()
    .integrate(|x: f64| Ok((20.0 * x).cos() * amplitude(x)?), interval)
    .unwrap();
    let plan = WeightedQuadraturePlan::fourier(
        interval,
        &[],
        20.0,
        FourierWeight::Cosine,
        WeightedQuadratureOptions::default().with_moment_levels(Budget::ONE),
    )
    .unwrap();
    let (value, report) = plan.integrate(&amplitude).unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::RefinementLimit
    );
    assert!(report.error_estimate().unwrap() >= (value - reference).abs());
}

#[test]
fn blocked_fourier_region_retains_error_while_other_regions_refine() {
    let boundary = 0.99;
    let options = WeightedQuadratureOptions::new(
        SubdivisionOptions::default()
            .with_tolerance(ErrorTolerance::scalar(1e-10, 0.0, NormSelection::Infinity).unwrap())
            .with_max_subdivisions(Budget::try_from(128).unwrap()),
    )
    .with_moment_levels(Budget::ONE);
    let amplitude = |x: f64| {
        Ok(if x < boundary {
            1e6 / (1.0 + 1000.0 * (x - 0.4).powi(2))
        } else {
            1.0 / (1.0 + ((x - 0.995) / 1e-5).powi(2))
        })
    };
    let strong = WeightedQuadraturePlan::fourier(
        Interval::new(0.0, boundary).unwrap(),
        &[],
        100.0,
        FourierWeight::Cosine,
        options.clone(),
    )
    .unwrap();
    let (_, blocked) = strong.integrate(&amplitude).unwrap();
    assert_eq!(
        blocked.termination(),
        IntegrationTermination::RefinementLimit
    );
    let plan = WeightedQuadraturePlan::fourier(
        Interval::new(0.0, 1.0).unwrap(),
        &[boundary],
        100.0,
        FourierWeight::Cosine,
        options,
    )
    .unwrap();
    let (_, report) = plan.integrate(&amplitude).unwrap();
    assert!(!report.is_success());
    assert!(report.evaluations() > 40, "{report:?}");
    assert!(report.error_estimate().unwrap() >= blocked.error_estimate().unwrap());
}

#[test]
fn endpoint_metadata_survives_rounded_coordinates_and_segment_boundaries() {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    let de = DoubleExponentialIntegrator::<f64>::new(
        DoubleExponentialDomain::Finite {
            interval: Interval::new(1.0, 2.0).unwrap(),
            singular_points: &[1.5],
        },
        LevelOptions::default(),
    )
    .unwrap();
    let rounded = AtomicBool::new(false);
    let (value, report) = de
        .integrate(|point: DoubleExponentialPoint<f64>| {
            let (endpoint, distance) = point.nearest_endpoint().unwrap();
            assert!(distance > 0.0);
            if point.coordinate() == endpoint {
                rounded.store(true, Relaxed);
            }
            let distance = if endpoint == 1.5 {
                distance
            } else {
                (point.coordinate() - 1.5).abs()
            };
            Ok(1.0 / distance.sqrt())
        })
        .unwrap();
    assert!(rounded.load(Relaxed));
    assert!(report.is_success());
    assert!((value - 2.0 * 2.0_f64.sqrt()).abs() < 1e-8);
    let (_, report) =
        DoubleExponentialIntegrator::new(DoubleExponentialDomain::RealLine, de.options().clone())
            .unwrap()
            .integrate(|point: DoubleExponentialPoint<f64>| {
                assert_eq!(point.nearest_endpoint(), None);
                Ok((-point.coordinate().powi(2)).exp())
            })
            .unwrap();
    assert!(report.is_success());
    let (value, report) = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::HalfLine { lower: 1.0 },
        de.options().clone(),
    )
    .unwrap()
    .integrate(|point: DoubleExponentialPoint<f64>| {
        let (endpoint, distance) = point.nearest_endpoint().unwrap();
        assert_eq!(endpoint, 1.0);
        Ok((-distance).exp() / distance.sqrt())
    })
    .unwrap();
    assert!(report.is_success(), "{report:?}");
    assert!((value - core::f64::consts::PI.sqrt()).abs() < 1e-8);
}

#[test]
fn each_fourier_component_and_direction_must_converge() {
    let make = |domain| {
        FourierTailPlan::new(
            domain,
            1.0,
            FourierWeight::Sine,
            FourierTailOptions::default(),
        )
        .unwrap()
    };
    let (_, report) = make(FourierDomain::Upper { lower: 0.0 })
        .integrate(|x: f64| Ok([(-x).exp(), 1.0 + 10.0 * (-x).exp()]))
        .unwrap();
    assert!(!report.is_success());
    let (_, report) = make(FourierDomain::Lower { upper: 0.0 })
        .integrate(|x: f64| Ok(1.0 + 10.0 * x.exp()))
        .unwrap();
    assert!(!report.is_success());
    let (_, report) = make(FourierDomain::RealLine)
        .integrate(|x: f64| {
            Ok(if x >= 0.0 {
                (-x).exp()
            } else {
                1.0 + 10.0 * x.exp()
            })
        })
        .unwrap();
    assert!(!report.is_success());
}

/// Amplitudes `L + g(x)` with `g` decreasing to zero: cycle masses and amplitude
/// maxima decrease, but towards `L != 0`, so their alternating partial sums only
/// have an Abel mean. A cosine cycle even integrates the constant part to zero.
fn nonzero_limit_cases<S: RealScalar + Default>() {
    let options = FourierTailOptions::<S>::default();
    let amplitudes: [fn(S) -> S; 3] = [
        |x| S::ONE + S::ONE / (S::ONE + x.abs()),
        |x| S::ONE + (-x.abs() / s::<S>(10.0)).exp(),
        |x| S::ONE + S::ONE / (S::ONE + x.abs()).sqrt(),
    ];
    for weight in [FourierWeight::Sine, FourierWeight::Cosine] {
        for (index, amplitude) in amplitudes.iter().enumerate() {
            for omega in [0.5, 1.0, 3.0, 7.5] {
                if index != 0 && omega != 1.0 {
                    continue;
                }
                let domains = [
                    FourierDomain::Upper { lower: S::ZERO },
                    FourierDomain::Lower { upper: S::ZERO },
                    FourierDomain::RealLine,
                ];
                for domain in domains.into_iter().take(if omega == 1.0 { 3 } else { 1 }) {
                    let plan =
                        FourierTailPlan::new(domain, s(omega), weight, options.clone()).unwrap();
                    let (value, report) = plan.integrate(|x: S| Ok(amplitude(x))).unwrap();
                    assert!(
                        !report.is_success(),
                        "{weight:?} amplitude {index} omega {omega} {domain:?}: {value:?} {report:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn fourier_tails_tending_to_a_nonzero_limit_never_succeed_binary64() {
    nonzero_limit_cases::<f64>();
}
#[test]
fn fourier_tails_tending_to_a_nonzero_limit_never_succeed_binary32() {
    nonzero_limit_cases::<f32>();
}

type TailCase<S> = (f64, f64, FourierWeight, fn(S) -> S, f64);

/// Convergent algebraic, logarithmic and geometric tails keep succeeding with
/// their error inside the estimate (references: closed forms and mpmath).
fn decaying_tail_cases<S: RealScalar + Default>(near_origin_only: bool) {
    let half_pi_root = core::f64::consts::FRAC_PI_2.sqrt();
    let cases: [TailCase<S>; 9] = [
        (
            0.0,
            1.0,
            FourierWeight::Sine,
            |x| S::ONE / (S::ONE + x),
            0.621_449_624_235_813_4,
        ),
        (
            0.0,
            1.0,
            FourierWeight::Cosine,
            |x| S::ONE / (S::ONE + x),
            0.343_377_961_556_427,
        ),
        (
            0.0,
            1.0,
            FourierWeight::Sine,
            |x| S::ONE / x.sqrt(),
            half_pi_root,
        ),
        // sqrt(pi/2) (pi/2 - gamma - 2 ln 2), from d/ds Gamma(s) sin(pi s/2) at s = 1/2.
        (
            0.0,
            1.0,
            FourierWeight::Sine,
            |x| x.ln() / x.sqrt(),
            -0.492_193_631_158_073_35,
        ),
        (0.0, 10.0, FourierWeight::Sine, |x| (-x).exp(), 10.0 / 101.0),
        (0.0, -1.0, FourierWeight::Sine, |x| (-x).exp(), -0.5),
        (
            0.0,
            1e-3,
            FourierWeight::Sine,
            |x| (-x).exp(),
            1e-3 / (1.0 + 1e-6),
        ),
        (
            0.0,
            1.0,
            FourierWeight::Sine,
            |x| S::ONE / ((S::ONE + x) * (S::ONE + x)),
            0.343_377_961_556_427,
        ),
        // pi/2 - Si(10^6).
        (
            1e6,
            1.0,
            FourierWeight::Sine,
            |x| S::ONE / x,
            9.367_517_775_377_691e-7,
        ),
    ];
    for (lower, omega, weight, amplitude, expected) in cases {
        if near_origin_only && lower != 0.0 {
            continue;
        }
        let plan = FourierTailPlan::new(
            FourierDomain::Upper { lower: s(lower) },
            s(omega),
            weight,
            FourierTailOptions::default(),
        )
        .unwrap();
        let (value, report) = plan.integrate(|x: S| Ok(amplitude(x))).unwrap();
        let error = (value.into_f64() - expected).abs();
        let rounding = 64.0 * S::EPSILON.into_f64() * expected.abs();
        assert!(
            report.is_success(),
            "lower {lower} omega {omega}: {value:?} {report:?}"
        );
        assert!(
            error <= report.error_estimate().unwrap().into_f64() + rounding,
            "lower {lower} omega {omega}: {value:?} error {error} {report:?}"
        );
    }
}

#[test]
fn decaying_fourier_tails_keep_succeeding_binary64() {
    decaying_tail_cases::<f64>(false);
}
#[test]
fn decaying_fourier_tails_keep_succeeding_binary32() {
    // A cycle 10^6 from the origin carries a phase error near one in binary32.
    decaying_tail_cases::<f32>(true);
}

/// Exact `∫ H(x - 0.3) (1 + x) sin(omega x) dx` over [0, 1].
fn jump_reference(omega: f64) -> f64 {
    let primitive =
        |x: f64| -(1.0 + x) * (omega * x).cos() / omega + (omega * x).sin() / (omega * omega);
    primitive(1.0) - primitive(0.3)
}

#[test]
fn interior_jumps_and_cusps_stay_visible_at_high_frequency() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let jump = |x: f64| Ok(if x < 0.3 { 0.0 } else { 1.0 + x });
    for omega in [1e4, 1e5, 1e6] {
        let plan = WeightedQuadraturePlan::fourier(
            interval,
            &[],
            omega,
            FourierWeight::Sine,
            WeightedQuadratureOptions::default(),
        )
        .unwrap();
        let (value, report) = plan.integrate(&jump).unwrap();
        let expected = jump_reference(omega);
        let error = (value - expected).abs();
        assert!(report.evaluations() > 25, "omega {omega}: {report:?}");
        assert!(
            error <= report.error_estimate().unwrap(),
            "omega {omega}: {value} versus {expected} {report:?}"
        );
        if report.is_success() {
            assert!(error <= 1.49e-8 * (1.0 + expected.abs()), "{report:?}");
        }
    }
    // mpmath reference, split at the cusp and at every period.
    let plan = WeightedQuadraturePlan::fourier(
        interval,
        &[],
        1e5,
        FourierWeight::Sine,
        WeightedQuadratureOptions::new(
            SubdivisionOptions::default()
                .with_tolerance(ErrorTolerance::scalar(0.0, 1e-6, NormSelection::Infinity).unwrap())
                .with_max_subdivisions(Budget::try_from(200).unwrap()),
        )
        .with_moment_levels(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let expected = 1.387_029_236_864_869e-5;
    let (value, report) = plan.integrate(|x: f64| Ok((x - 0.3).abs().sqrt())).unwrap();
    assert!(report.is_success(), "{value} {report:?}");
    assert!(
        (value - expected).abs() <= report.error_estimate().unwrap(),
        "{value} {report:?}"
    );
    assert!(
        (value - expected).abs() <= 2e-6 * expected,
        "{value} {report:?}"
    );
}

#[test]
fn resolved_smooth_amplitudes_keep_one_moment_region() {
    fn check<S: RealScalar + Default>() {
        for omega in [50.0, 1e3, 1e5] {
            let plan = WeightedQuadraturePlan::fourier(
                Interval::new(S::ZERO, S::ONE).unwrap(),
                &[],
                s::<S>(omega),
                FourierWeight::Cosine,
                WeightedQuadratureOptions::default(),
            )
            .unwrap();
            let (value, report) = plan.integrate(|x: S| Ok(x.exp())).unwrap();
            let w = s::<S>(omega).into_f64();
            let e = core::f64::consts::E;
            let expected = (e * (w * w.sin() + w.cos()) - 1.0) / (1.0 + w * w);
            assert!(report.is_success(), "{report:?}");
            assert_eq!(report.evaluations(), 25, "omega {omega}: {report:?}");
            let rounding = 64.0 * S::EPSILON.into_f64() * (1.0 + w * S::EPSILON.into_f64());
            assert!(
                (value.into_f64() - expected).abs()
                    <= report.error_estimate().unwrap().into_f64() + rounding,
                "omega {omega}: {value:?} versus {expected} {report:?}"
            );
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn poles_beside_endpoints_use_principal_value_moments() {
    fn check<S: RealScalar + Default>(poles: &[(f64, f64)]) {
        let interval = Interval::new(S::ZERO, S::ONE).unwrap();
        for &(pole, exponential) in poles {
            let pole = s::<S>(pole);
            let plan = WeightedQuadraturePlan::principal_value(
                interval,
                pole,
                WeightedQuadratureOptions::default(),
            )
            .unwrap();
            // PV ∫₀¹ dx/(x - c) = ln|1 - c| - ln|c| at the exact scalar pole.
            let c = pole.into_f64();
            let expected = (1.0 - c).abs().ln() - c.abs().ln();
            let (value, report) = plan.integrate(|_: S| Ok(S::ONE)).unwrap();
            assert!(report.is_success(), "pole {c:e}: {value:?} {report:?}");
            assert_eq!(report.evaluations(), 25, "pole {c:e}: {report:?}");
            let rounding = 64.0 * S::EPSILON.into_f64() * expected.abs();
            assert!(
                (value.into_f64() - expected).abs() <= rounding,
                "pole {c:e}: {value:?} versus {expected}"
            );
            if exponential.is_finite() {
                let (value, report) = plan.integrate(|x: S| Ok(x.exp())).unwrap();
                assert!(report.is_success(), "pole {c:e}: {value:?} {report:?}");
                let error = (value.into_f64() - exponential).abs();
                assert!(
                    error <= report.error_estimate().unwrap().into_f64() + rounding,
                    "pole {c:e}: {value:?} {report:?}"
                );
            }
        }
    }
    // mpmath references for exp(x) at the exact binary64 poles.
    check::<f64>(&[
        (1e-300, 692.093_430_049_668_1),
        (5e-324, f64::NAN),
        (1.0 - f64::EPSILON / 2.0, f64::NAN),
        (-1e-12, f64::NAN),
        (1.0 + 1e-12, -72.943_278_739_850_04),
    ]);
    check::<f32>(&[
        (1e-30, f64::NAN),
        (1e-45, f64::NAN),
        (1.0 + 1.0 / 1_048_576.0, f64::NAN),
    ]);
}

type EndpointCase<'a> = (f64, f64, f64, &'a [f64], f64);

#[test]
fn large_endpoint_exponents_and_underflowing_regions_stay_representable() {
    fn check<S: RealScalar + Default>(cases: &[EndpointCase<'_>]) {
        let relative = s::<S>(1e-10).max(s::<S>(64.0) * S::EPSILON);
        let options = WeightedQuadratureOptions::new(SubdivisionOptions::default().with_tolerance(
            ErrorTolerance::scalar(S::ZERO, relative, NormSelection::Infinity).unwrap(),
        ));
        for &(alpha, beta, upper, points, expected) in cases {
            let points: Vec<S> = points.iter().map(|&p| s(p)).collect();
            let plan = WeightedQuadraturePlan::endpoint(
                Interval::new(S::ZERO, s(upper)).unwrap(),
                &points,
                scioxide_integrate::EndpointWeight::new(s(alpha), s(beta), false, false).unwrap(),
                options.clone(),
            )
            .unwrap();
            let (value, report) = plan.integrate(|_: S| Ok(S::ONE)).unwrap();
            let error = (value.into_f64() - expected).abs();
            let rounding = 64.0 * S::EPSILON.into_f64() * expected;
            assert!(
                report.is_success(),
                "alpha {alpha} beta {beta} upper {upper}: {value:?} {report:?}"
            );
            assert!(
                error <= report.error_estimate().unwrap().into_f64() + rounding,
                "alpha {alpha} beta {beta} upper {upper}: {value:?} {report:?}"
            );
            assert!(
                error <= 2.0 * relative.into_f64() * expected + rounding,
                "alpha {alpha} beta {beta} upper {upper}: {value:?} versus {expected}"
            );
        }
    }
    // (b - a)^(alpha + beta + 1) B(alpha + 1, beta + 1) for a unit amplitude.
    check::<f64>(&[
        (600.0, 0.0, 1.0, &[], 1.0 / 601.0),
        (500.0, 0.0, 0.5, &[], 0.5_f64.powi(501) / 501.0),
        (0.0, 600.0, 1.0, &[], 1.0 / 601.0),
        (0.0, 600.0, 0.5, &[], 0.5_f64.powi(601) / 601.0),
        (3.0, 0.0, 1.0, &[1e-100], 0.25),
        (1022.0, 0.0, 2.0, &[], 2.0_f64.powi(1023) / 1023.0),
    ]);
    check::<f32>(&[
        (100.0, 0.0, 1.0, &[], 1.0 / 101.0),
        (3.0, 0.0, 1.0, &[1e-20], 0.25),
        (126.0, 0.0, 2.0, &[], 2.0_f64.powi(127) / 127.0),
    ]);
}
