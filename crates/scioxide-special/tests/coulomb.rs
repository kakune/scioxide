//! Independent ball-arithmetic references and physical identities.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
#[path = "support/scaled.rs"]
mod scaled_reference;
use scioxide_core::{AlgebraicScalar, Budget, Complex, NumericScalar, RealScalar};
use scioxide_special::*;

#[test]
fn plain_waves_project_all_components() {
    fn check<T: NumericScalar>(z: T) {
        let l = T::Real::ZERO;
        let eta = T::Real::ZERO;
        let values = coulomb_wave(l, eta, z).unwrap();
        let report = coulomb_wave_with_options(l, eta, z, CoulombOptions::default()).unwrap();
        for (actual, expected) in
            values
                .into_iter()
                .zip([report.f(), report.g(), report.f_prime(), report.g_prime()])
        {
            assert!(
                (actual - expected.try_value().unwrap()).abs()
                    <= T::Real::EPSILON * (T::Real::ONE + actual.abs())
            );
        }
    }
    check(1.0_f32);
    check(1.0_f64);
    check(Complex::new(1.0_f32, 0.25));
    check(Complex::new(1.0_f64, 0.25));
    let [f, g, fp, gp] = coulomb_wave(0.0, 0.0, 1.0_f64).unwrap();
    for (actual, expected) in [f, g, fp, gp].into_iter().zip([
        1.0_f64.sin(),
        1.0_f64.cos(),
        1.0_f64.cos(),
        -1.0_f64.sin(),
    ]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let report = coulomb_wave_with_options(0.0, 300.0, 1.0_f64, CoulombOptions::default()).unwrap();
    assert!(report.is_success());
    assert!(report.g().scaled().is_some());
    assert!(coulomb_wave(0.0, 300.0, 1.0_f64).is_err());
    for (actual, expected) in coulomb_wave(0.0, 0.0, 0.0_f64)
        .unwrap()
        .into_iter()
        .zip([0.0, 1.0, 1.0, 0.0])
    {
        assert!((actual - expected).abs() < 1e-14);
    }
    assert!(coulomb_wave(-0.5, 0.0, 1.0_f64).is_err());
}

fn oracle<R: RealScalar>(width: u32)
where
    Complex<R>: NumericScalar<Real = R>,
{
    let mut cases = 0;
    for line in include_str!("data/coulomb.tsv")
        .lines()
        .chain(include_str!("data/coulomb_complex.tsv").lines())
        .filter(|s| !s.starts_with('#') && !s.is_empty())
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[1].parse::<u32>().unwrap() != width {
            continue;
        }
        let x = |i: usize| fields[i].parse::<f64>().unwrap();
        let r = |i| R::try_from_f64_lossy(x(i)).unwrap();
        let z = Complex::new(r(4), r(5));
        let options = CoulombOptions::default();
        if fields[0] == "c" {
            let result = coulomb_normalization_with_options(r(2), r(3), options).unwrap();
            assert!(result.termination().is_success());
            let s = result.scaled().unwrap().value();
            let expected = x(6);
            let exponent = fields[8].parse::<i64>().unwrap();
            let actual = s
                .try_projective()
                .unwrap()
                .significand()
                .try_to_f64_exact()
                .unwrap()
                * 2.0_f64.powi((s.try_projective().unwrap().exponent() - exponent) as i32);
            assert!(
                (actual - expected).abs() <= expected.abs() * if width == 64 { 5e-9 } else { 1e-3 },
                "{line}: {actual}"
            );
            let estimate = result.scaled().unwrap();
            let bound = scaled_reference::error_at_exponent(
                estimate.value(),
                estimate.absolute_error().try_to_f64_exact().unwrap(),
                exponent,
            );
            assert!(
                (actual - expected).abs() <= bound + f64::EPSILON * expected.abs(),
                "normalization bound: {line}: {actual}, bound {bound}"
            );
            if let Some(value) = result.value() {
                let expected = expected.scalbn(exponent as i32);
                let actual = value.value().try_to_f64_exact().unwrap();
                assert!(
                    (actual - expected).abs()
                        <= value.absolute_error().try_to_f64_exact().unwrap()
                            + f64::EPSILON * expected.abs(),
                    "ordinary normalization: {line}"
                );
            }
            // Exercise recurrence normalization against the same independent fixture.
            if x(2) >= 0.0 && x(2) <= 7.0 && x(2).fract() == 0.0 {
                let mut output = [SpecialFunctionResult::<R>::zero(); 8];
                let mut staging = output;
                coulomb_normalization_sequence_into(
                    R::ZERO,
                    r(3),
                    options,
                    &mut output,
                    &mut staging,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
                let report = output[x(2) as usize];
                let estimate = report.scaled().unwrap();
                let actual = scaled_reference::at_exponent(estimate.value(), exponent).re;
                let bound = scaled_reference::error_at_exponent(
                    estimate.value(),
                    estimate.absolute_error().try_to_f64_exact().unwrap(),
                    exponent,
                );
                assert!(
                    (actual - expected).abs() <= bound + f64::EPSILON * expected.abs(),
                    "normalization recurrence bound: {line}: {actual}, bound {bound}"
                );
            }
            cases += 1;
            continue;
        }
        if fields[0] == "h" || fields[0] == "hp" {
            let report =
                hydrogenic_radial_with_options(x(2) as u32, x(3) as u32, r(9), z, options).unwrap();
            assert!(report.termination().is_success(), "{line}: {report:?}");
            let exponent = fields[8].parse::<i32>().unwrap();
            let expected = Complex::new(x(6), x(7));
            let relative = if width == 64 { 5e-9 } else { 1e-3 };
            if fields[0] == "hp" {
                if let Some(d) = report.derivative() {
                    let d = d.value();
                    let actual = Complex::new(
                        d.re.try_to_f64_exact().unwrap(),
                        d.im.try_to_f64_exact().unwrap(),
                    );
                    let reference = expected * Complex::new(2.0_f64.powi(exponent), 0.0);
                    assert!(
                        (actual - reference).norm() <= relative * reference.norm().max(1e-12),
                        "{line}: {report:?}"
                    );
                } else {
                    assert!(
                        exponent < if width == 64 { -1073 } else { -148 },
                        "{line}: missing derivative"
                    );
                }
            } else {
                let s = report.scaled().unwrap().value();
                let actual = scaled_reference::at_exponent(s, i64::from(exponent));
                assert!(
                    (actual - expected).norm() <= relative * expected.norm().max(1e-12),
                    "{line}: {report:?}, actual {actual:?}"
                );
            }
            cases += 1;
            continue;
        }
        let report = match fields[0] {
            "f" => coulomb_f_with_options(r(2), r(3), z, options),
            "g" => coulomb_g_with_options(r(2), r(3), z, options),
            "fp" => coulomb_f_prime_with_options(r(2), r(3), z, options),
            "gp" => coulomb_g_prime_with_options(r(2), r(3), z, options),
            _ => panic!("unknown fixture"),
        }
        .unwrap_or_else(|e| panic!("{line}: {e:?}"));
        assert!(report.termination().is_success(), "{line}: {report:?}");
        let scaled = report.scaled().unwrap().value();
        let exponent = fields[8].parse::<i64>().unwrap();
        let actual = scaled_reference::at_exponent(scaled, exponent);
        let expected = Complex::new(x(6), x(7));
        let relative = if width == 64 { 5e-9 } else { 1e-3 };
        assert!(
            (actual - expected).norm() <= relative * expected.norm(),
            "{line}: actual {actual:?}, expected {expected:?}, report {report:?}"
        );
        let estimate = report.scaled().unwrap();
        let bound = scaled_reference::error_at_exponent(
            estimate.value(),
            estimate.absolute_error().try_to_f64_exact().unwrap(),
            exponent,
        );
        assert!(
            (actual - expected).norm() <= bound + f64::EPSILON * expected.norm(),
            "wave bound: {line}: actual {actual:?}, bound {bound}, report {report:?}"
        );
        if let Some(value) = report.value() {
            let actual = Complex::new(value.value().re.into_f64(), value.value().im.into_f64());
            let expected = Complex::new(
                expected.re.scalbn(exponent as i32),
                expected.im.scalbn(exponent as i32),
            );
            assert!(
                (actual - expected).norm()
                    <= value.absolute_error().into_f64() + f64::EPSILON * expected.norm(),
                "ordinary wave bound: {line}"
            );
        }
        if x(5) == 0.0 && x(4) > 0.0 {
            let ordinary = match fields[0] {
                "f" => coulomb_f_with_options(r(2), r(3), r(4), options),
                "g" => coulomb_g_with_options(r(2), r(3), r(4), options),
                "fp" => coulomb_f_prime_with_options(r(2), r(3), r(4), options),
                _ => coulomb_g_prime_with_options(r(2), r(3), r(4), options),
            }
            .unwrap();
            assert_eq!(ordinary.termination(), report.termination());
        }
        cases += 1;
    }
    assert!(cases >= 12236, "incomplete oracle fixture: {cases}");
}
#[test]
fn independent_f64() {
    oracle::<f64>(64);
}
#[test]
fn independent_f32() {
    oracle::<f32>(32);
}

#[test]
fn wronskians_recurrences_and_zero_charge() {
    for (l, eta, x) in [
        (0.0, 0.0, 2.0),
        (0.0, 1.0, 0.3),
        (2.5, -3.0, 2.0),
        (10.0, 20.0, 43.0),
        (3.0, 1.0, 100.0),
    ] {
        let r = coulomb_wave_with_options(l, eta, x, scioxide_special::CoulombOptions::default())
            .unwrap();
        assert!(r.is_success(), "{r:?}");
        let f = r.f().try_value().unwrap();
        let g = r.g().try_value().unwrap();
        let fp = r.f_prime().try_value().unwrap();
        let gp = r.g_prime().try_value().unwrap();
        assert!(
            (fp * g - f * gp - 1.0_f64).abs() < 1e-9,
            "{l} {eta} {x}: {}",
            fp * g - f * gp
        );
        let next = coulomb_f(l + 1.0, eta, x).unwrap();
        let expected =
            ((l + 1.0) / x + eta / (l + 1.0)) * f - ((l + 1.0_f64).hypot(eta) / (l + 1.0)) * next;
        assert!((fp - expected).abs() < 1e-10);
    }
    for x in [0.001_f64, 1.0, 10.0, 1e12] {
        let r = coulomb_wave_with_options(0.0, 0.0, x, scioxide_special::CoulombOptions::default())
            .unwrap();
        assert!((r.f().try_value().unwrap() - x.sin()).abs() < 1e-10);
        assert!((r.g().try_value().unwrap() - x.cos()).abs() < 1e-10);
    }
}
#[test]
fn hydrogenic_limits_and_nodes() {
    for r in [0.0_f64, 0.1, 1.0, 2.0, 10.0] {
        let one = hydrogenic_radial_with_options(1, 0, 1.0, r, CoulombOptions::default()).unwrap();
        assert!((one.try_value().unwrap() - 2.0 * (-r).exp()).abs() < 1e-12);
        assert!((one.derivative().unwrap().value() + 2.0 * (-r).exp()).abs() < 1e-12);
        let two = hydrogenic_radial(2, 0, 1.0, r).unwrap();
        assert!((two - (2.0 - r) * (-r / 2.0).exp() / (2.0 * 2.0_f64.sqrt())).abs() < 1e-12);
        let p = hydrogenic_radial(2, 1, 1.0, r).unwrap();
        assert!((p - r * (-r / 2.0).exp() / (2.0 * 6.0_f64.sqrt())).abs() < 1e-12);
    }
}
#[test]
fn domains_budget_and_range() {
    assert!(coulomb_f(-0.5, 0.0, 1.0_f64).is_err());
    assert!(coulomb_f(0.0, f64::NAN, 1.0).is_err());
    assert!(
        (coulomb_g(0.0, 1.0, 0.0_f64).unwrap() * coulomb_normalization(0.0, 1.0).unwrap() - 1.0)
            .abs()
            < 1e-13
    );
    assert!(hydrogenic_radial(0, 0, 1.0, 1.0_f64).is_err());
    let r = coulomb_f_with_options(
        2.0,
        3.0,
        1.0_f64,
        CoulombOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    let r = coulomb_normalization_with_options(0.0, 300.0, CoulombOptions::default()).unwrap();
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Finite);
    assert_eq!(r.try_value().unwrap(), 0.0);
    assert!(r.try_value().unwrap().is_sign_positive());
    assert!(r.scaled().is_some() && r.logarithmic().is_some());
}

#[test]
fn origin_limits_cut_lips_and_complex_wronskian() {
    let options = CoulombOptions::default();
    let c = coulomb_normalization(0.0_f64, 1.0).unwrap();
    assert_eq!(coulomb_f(0.25, 1.0, 0.0_f64).unwrap(), 0.0);
    assert!((coulomb_f_prime(0.0, 1.0, 0.0_f64).unwrap() - c).abs() < 1e-14);
    assert!((coulomb_spherical_f(0.0, 1.0, 0.0_f64).unwrap() - c).abs() < 1e-14);
    assert!(
        coulomb_f_with_options(-0.25, 1.0, 0.0_f64, options)
            .unwrap()
            .derivative()
            .is_none()
    );
    assert!(coulomb_f_prime(-0.25, 1.0, 0.0_f64).is_err());
    assert!(coulomb_spherical_f(-0.25, 1.0, 0.0_f64).is_err());
    assert_eq!(coulomb_g_prime(0.0, 0.0, 0.0_f64).unwrap(), 0.0);
    for (l, eta, reason) in [
        (0.0, 1.0, scioxide_core::DomainReason::Pole),
        (1.0, 0.0, scioxide_core::DomainReason::Pole),
        (-0.25, 0.0, scioxide_core::DomainReason::OutsideDomain),
    ] {
        assert!(matches!(
            coulomb_g_prime(l, eta, 0.0_f64).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DomainError { reason: actual, .. } if actual == reason
        ));
    }
    for z in [
        Complex::new(-2.0_f64, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(-2.0, 0.2),
    ] {
        let upper =
            coulomb_wave_with_options(0.25, 1.0, z, scioxide_special::CoulombOptions::default())
                .unwrap();
        let lower = coulomb_wave_with_options(
            0.25,
            1.0,
            z.conj(),
            scioxide_special::CoulombOptions::default(),
        )
        .unwrap();
        for (a, b) in [
            (upper.f(), lower.f()),
            (upper.g(), lower.g()),
            (upper.f_prime(), lower.f_prime()),
            (upper.g_prime(), lower.g_prime()),
        ] {
            assert!((a.try_value().unwrap().conj() - b.try_value().unwrap()).norm() < 1e-11);
        }
        let w = upper.f_prime().try_value().unwrap() * upper.g().try_value().unwrap()
            - upper.f().try_value().unwrap() * upper.g_prime().try_value().unwrap();
        assert!((w - Complex::new(1.0, 0.0)).norm() < 1e-9);
    }
}

#[test]
fn hydrogenic_integral_normalization_and_largest_phase() {
    for (n, l, charge, limit) in [(1, 0, 1.0_f64, 40.0), (2, 1, 3.0, 40.0)] {
        let steps = 4096;
        let h = limit / steps as f64;
        let integral = (0..=steps)
            .map(|k| {
                let r = k as f64 * h;
                let value = hydrogenic_radial(n, l, charge, r).unwrap();
                let weight = if k == 0 || k == steps {
                    1.0
                } else if k % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                weight * value * value * r * r
            })
            .sum::<f64>()
            * h
            / 3.0;
        assert!((integral - 1.0).abs() < 1e-8, "{n} {l}: {integral}");
    }
    let r = coulomb_wave_with_options(
        0.0,
        0.0,
        1e308_f64,
        scioxide_special::CoulombOptions::default(),
    )
    .unwrap();
    assert!((r.f().try_value().unwrap() - 1e308_f64.sin()).abs() < 1e-12);
    assert!((r.g().try_value().unwrap() - 1e308_f64.cos()).abs() < 1e-12);
}

#[test]
fn issue_point_matches_high_precision_reference() {
    // mpmath at 80 and 120 digits rounds identically at this exact binary64 input.
    // 0x1.7bb71150a7e8ap-1, 0x1.445a31018fbbap-1, 0x1.42b60df3387e2p+0
    let [l, eta, x] = [
        0x3fe7_bb71_150a_7e8a_u64,
        0x3fe4_45a3_1018_fbba,
        0x3ff4_2b60_df33_87e2,
    ]
    .map(f64::from_bits);
    let expected = [
        0.29445960605841537,
        1.6836970574149153,
        0.4296505412878571,
        -0.939343265514107,
    ];
    let plain = coulomb_wave(l, eta, x).unwrap();
    let report = coulomb_wave_with_options(l, eta, x, CoulombOptions::default()).unwrap();
    for ((actual, component), reference) in plain
        .into_iter()
        .zip([report.f(), report.g(), report.f_prime(), report.g_prime()])
        .zip(expected)
    {
        assert_eq!(actual, component.try_value().unwrap());
        assert!((actual - reference).abs() <= 4.0 * f64::EPSILON * reference.abs());
        assert!(component.value().unwrap().absolute_error() >= (actual - reference).abs());
    }
}

#[test]
fn turning_anchor_and_series_radius_boundaries() {
    fn waves(l: f64, eta: f64, x: f64) -> [f64; 4] {
        let plain = coulomb_wave(l, eta, x).unwrap();
        let report = coulomb_wave_with_options(l, eta, x, CoulombOptions::default()).unwrap();
        let detailed = [report.f(), report.g(), report.f_prime(), report.g_prime()]
            .map(|component| component.try_value().unwrap());
        assert_eq!(plain, detailed, "{l} {eta} {x}: plain/detailed projection");
        let [f, g, fp, gp] = plain;
        assert!(
            (fp * g - f * gp - 1.0).abs() <= 1e-13 * (fp * g).abs().max((f * gp).abs()),
            "{l} {eta} {x}: Wronskian"
        );
        let next = coulomb_f(l + 1.0, eta, x).unwrap();
        let expected =
            ((l + 1.0) / x + eta / (l + 1.0)) * f - ((l + 1.0).hypot(eta) / (l + 1.0)) * next;
        assert!(
            (fp - expected).abs() <= 1e-12 * fp.abs().max(f.abs()),
            "{l} {eta} {x}: order recurrence"
        );
        plain
    }
    fn continuous(l: f64, eta: f64, x: f64) {
        let [below, at, above] = [x.next_down(), x, x.next_up()].map(|x| waves(l, eta, x));
        let scale = at.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for side in [below, above] {
            for (a, b) in side.iter().zip(at) {
                assert!(
                    (a - b).abs() <= 64.0 * f64::EPSILON * scale,
                    "{l} {eta} {x}"
                );
            }
        }
    }
    for l in [-0.25, 0.0, 0.74, 2.9] {
        for eta in [-1.0, -0.3, 0.0, 0.63, 1.0, 20.0] {
            // Steed's fractions run at x from one unit beyond the turning point.
            continuous(l, eta, eta + f64::hypot(eta, l + 0.5) + 1.0);
            // Below it, F's Frobenius series starts at x up to this radius.
            let radius =
                8.0 * core::f64::consts::SQRT_2 / ((eta.abs() + 8.0).sqrt() + eta.abs().sqrt());
            continuous(l, eta, radius * radius);
            for x in [1.0, 1.26, 2.5, 7.99] {
                waves(l, eta, x);
            }
        }
    }
}

/// SplitMix64 samples in [0, 1); the seed is fixed, with no platform entropy.
fn unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut bits = *state;
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((bits ^ (bits >> 31)) >> 11) as f64 * 2.0_f64.powi(-53)
}

/// Default-budget waves with a Wronskian check and tight covering estimates.
fn default_budget_waves(l: f64, eta: f64, x: f64) -> usize {
    let report = coulomb_wave_with_options(l, eta, x, CoulombOptions::default())
        .unwrap_or_else(|error| panic!("{l} {eta} {x}: {error:?}"));
    assert!(report.is_success(), "{l} {eta} {x}");
    let components = [report.f(), report.g(), report.f_prime(), report.g_prime()];
    let [f, g, fp, gp] = components.map(|component| component.try_value().unwrap());
    let scale = (fp * g).abs().max((f * gp).abs());
    assert!(
        (fp * g - f * gp - 1.0).abs() <= 1e-13 * scale,
        "{l} {eta} {x}: Wronskian"
    );
    for component in components {
        let error = component.value().unwrap().absolute_error();
        // Estimates stay within a few binary64 units of the wave amplitude.
        assert!(
            error <= 64.0 * f64::EPSILON * scale.sqrt(),
            "{l} {eta} {x}: estimate {error}"
        );
    }
    report.f().counts().iterations()
}

#[test]
fn outgoing_series_is_admitted_by_its_own_tails() {
    // Beyond the former (l²+η²)/x ≤ 16 gate: these exhausted the default
    // budget in Steed fractions whose work grows with x.
    for (l, eta, x) in [
        (1000.0, 1000.0, 1.2e5),
        (0.0, 1000.0, 7e4),
        (1500.0, 0.0, 2e5),
        (1e4, 1e4, 4e6),
        (3000.0, -3000.0, 3.1e5),
    ] {
        assert!(default_budget_waves(l, eta, x) < 1024, "{l} {eta} {x}");
    }
}

#[test]
fn seeded_default_budget_sweep() {
    // l ≤ 1300 and |η| ≤ 1300 keep hypot(l, η) below 1900: either the outgoing
    // series is accepted or Steed's fractions finish within the default budget.
    // Box corners on both sides of series acceptance; Steed is costliest at
    // attractive charges just below it.
    for (l, eta) in [
        (1300.0, 1300.0),
        (1300.0, -1300.0),
        (0.0, -1300.0),
        (1300.0, 0.0),
    ] {
        for ratio in [58.0, 61.0] {
            default_budget_waves(l, eta, (l * l + eta * eta) / ratio);
        }
    }
    let mut state = 1677;
    for _ in 0..256 {
        let l = 1300.0 * unit(&mut state);
        let eta = 1300.0 * (2.0 * unit(&mut state) - 1.0);
        let x = 2e4 * 5e3_f64.powf(unit(&mut state));
        default_budget_waves(l, eta, x);
    }
}

#[test]
fn steed_work_beyond_the_default_box_is_reported() {
    // Below series acceptance, (l²+η²)/x > 59, Steed's work grows like x + |η|.
    let (l, eta, x) = (3000.0, 3000.0, 1e5);
    let error = coulomb_wave_with_options(l, eta, x, CoulombOptions::default()).unwrap_err();
    assert_eq!(publication::exhausted_counts(&error).iterations(), 65_536);
    let options =
        CoulombOptions::default().with_iteration_budget(Budget::try_from(1 << 17).unwrap());
    let report = coulomb_wave_with_options(l, eta, x, options).unwrap();
    assert!(report.is_success());
    let [f, g, fp, gp] = [report.f(), report.g(), report.f_prime(), report.g_prime()]
        .map(|component| component.try_value().unwrap());
    assert!((fp * g - f * gp - 1.0).abs() <= 1e-13);
}
