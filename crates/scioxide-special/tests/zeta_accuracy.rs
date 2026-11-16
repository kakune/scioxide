//! Differential, continuity, work-bound, and allocation tests for zeta-eta work.

#![allow(clippy::panic, clippy::unwrap_used)]

use scioxide_special::{dirichlet_eta, hurwitz_zeta, riemann_zeta};

const ORACLE: &str = include_str!("data/zeta_oracle.tsv");

#[derive(Clone, Copy)]
enum Function {
    Riemann,
    Eta,
    Hurwitz,
}

struct Case {
    scalar: &'static str,
    function: Function,
    arguments: Vec<u64>,
    expected: u64,
}

fn cases() -> impl Iterator<Item = Case> {
    ORACLE
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split('\t');
            let scalar = fields.next().unwrap();
            let function = match fields.next().unwrap() {
                "riemann_zeta" => Function::Riemann,
                "dirichlet_eta" => Function::Eta,
                "hurwitz_zeta" => Function::Hurwitz,
                name => panic!("unknown zeta oracle function: {name}"),
            };
            let arguments = fields
                .next()
                .unwrap()
                .split(',')
                .map(|bits| u64::from_str_radix(bits, 16).unwrap())
                .collect();
            let expected = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
            Case {
                scalar,
                function,
                arguments,
                expected,
            }
        })
}

fn assert_accuracy(actual: f64, expected: f64, relative: f64, absolute: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= absolute.max(relative * expected.abs()),
        "{label}: {actual} versus {expected}"
    );
}

fn ordered_f64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}

fn ordered_f32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 0 {
        bits | (1 << 31)
    } else {
        !bits
    }
}

#[test]
fn generated_f64_grid_meets_family_budget() {
    for case in cases().filter(|case| case.scalar == "f64") {
        let argument = |index: usize| f64::from_bits(case.arguments[index]);
        let actual = match case.function {
            Function::Riemann => riemann_zeta(argument(0)).unwrap(),
            Function::Eta => dirichlet_eta(argument(0)).unwrap(),
            Function::Hurwitz => hurwitz_zeta(argument(0), argument(1)).unwrap(),
        };
        let expected = f64::from_bits(case.expected);
        if matches!(case.function, Function::Riemann | Function::Eta) {
            assert!(
                ordered_f64(actual).abs_diff(ordered_f64(expected)) <= 64,
                "zeta/eta s={}: actual={actual:e}, expected={expected:e}",
                argument(0)
            );
            continue;
        }
        assert!(
            ordered_f64(actual).abs_diff(ordered_f64(expected)) <= 64,
            "Hurwitz {:?}: {actual} versus {expected}",
            case.arguments
        );
    }
}

#[test]
fn generated_f32_grid_meets_family_budget() {
    for case in cases().filter(|case| case.scalar == "f32") {
        let argument = |index: usize| f32::from_bits(u32::try_from(case.arguments[index]).unwrap());
        let actual = match case.function {
            Function::Riemann => riemann_zeta(argument(0)).unwrap(),
            Function::Eta => dirichlet_eta(argument(0)).unwrap(),
            Function::Hurwitz => hurwitz_zeta(argument(0), argument(1)).unwrap(),
        };
        let expected = f32::from_bits(u32::try_from(case.expected).unwrap());
        assert!(
            ordered_f32(actual).abs_diff(ordered_f32(expected)) <= 64,
            "zeta family {:?}: {actual} versus {expected}",
            case.arguments
        );
    }
}

#[test]
fn primitive_cutovers_are_continuous() {
    for boundary in [-1.0_f64, 0.0, 2.0, 4.0, 8.0, 16.0, 24.0] {
        let center = riemann_zeta(boundary).unwrap();
        let below = riemann_zeta(boundary.next_down()).unwrap();
        let above = riemann_zeta(boundary.next_up()).unwrap();
        let tolerance = 5.0e-11 * center.abs().max(1.0);
        assert!(
            (below - center).abs() <= tolerance,
            "lower discontinuity at {boundary}"
        );
        assert!(
            (above - center).abs() <= tolerance,
            "upper discontinuity at {boundary}"
        );
    }
    for boundary in [-1.0_f64, 0.0, 1.0, 2.0] {
        let center = dirichlet_eta(boundary).unwrap();
        let below = dirichlet_eta(boundary.next_down()).unwrap();
        let above = dirichlet_eta(boundary.next_up()).unwrap();
        assert!((below - center).abs() <= 5.0e-11);
        assert!((above - center).abs() <= 5.0e-11);
    }
}

#[test]
fn hurwitz_shift_and_riemann_identities_hold_on_a_grid() {
    for (s, a) in [(1.125_f64, 0.25), (2.0, 1.0), (2.5, 1.5), (8.0, 64.0)] {
        let value = hurwitz_zeta(s, a).unwrap();
        let shifted = hurwitz_zeta(s, a + 1.0).unwrap() + a.powf(-s);
        assert_accuracy(value, shifted, 5.0e-11, 5.0e-12, "Hurwitz shift");
    }
    for s in [1.125_f64, 2.0, 2.5, 8.0, 32.0] {
        assert_accuracy(
            hurwitz_zeta(s, 1.0).unwrap(),
            riemann_zeta(s).unwrap(),
            128.0 * f64::EPSILON,
            8.0 * f64::from_bits(1),
            "Hurwitz/Riemann identity",
        );
    }
}

#[test]
fn every_public_zeta_family_path_is_allocation_free() {
    for s in [
        -100.5_f64,
        -31.5,
        -16.0_f64.next_down(),
        -16.0_f64.next_up(),
        -4.0,
        -3.0,
        -1.0_f64.next_down(),
        -1.0,
        -1.0e-12,
        0.0,
        0.75,
        1.5,
        2.0,
        3.0,
        6.0,
        12.0,
        20.0,
    ] {
        let zeta = std::hint::black_box(riemann_zeta(s));
        let eta = std::hint::black_box(dirichlet_eta(s));
        assert!(zeta.is_ok());
        assert!(eta.is_ok());
    }
    for s in [-7.5_f32, -2.0, -1.0e-5, 0.0, 0.75, 1.5, 3.0, 12.0, 20.0] {
        let zeta = riemann_zeta(s);
        let eta = dirichlet_eta(s);
        assert!(zeta.is_ok());
        assert!(eta.is_ok());
    }
    for (s, a) in [(1.125_f64, 0.25), (2.0, 1.0), (2.5, 64.0), (32.0, 0.25)] {
        let result = std::hint::black_box(hurwitz_zeta(s, a));
        assert!(result.is_ok());
    }
    let errors = [
        riemann_zeta(f64::NAN),
        riemann_zeta(1.0_f64),
        riemann_zeta(-1_000.5_f64),
        riemann_zeta(-280.5_f64),
        dirichlet_eta(f64::INFINITY),
        dirichlet_eta(-1_000.5_f64),
        dirichlet_eta(-260.5_f64),
        hurwitz_zeta(1.0_f64, -1.0),
        hurwitz_zeta(2.0_f64, 0.0),
    ];
    assert!(errors.iter().all(Result::is_err));
}

#[test]
fn detailed_results_expose_stable_branches_and_work_bounds() {
    let central = scioxide_special::riemann_zeta_with_options(
        2.0_f64,
        scioxide_special::ZetaOptions::default(),
    )
    .unwrap();
    assert_eq!(central.algorithm().method(), "primitive_central");
    assert_eq!(central.algorithm().family(), "riemann_zeta");
    assert_eq!(central.algorithm().version(), 1);
    assert!(central.counts().iterations() <= 24);
    let eta = scioxide_special::dirichlet_eta_with_options(
        0.75_f64,
        scioxide_special::ZetaOptions::default(),
    )
    .unwrap();
    assert_eq!(eta.algorithm().method(), "primitive_central");
    let hurwitz = scioxide_special::hurwitz_zeta_with_options(
        2.5_f64,
        0.25,
        scioxide_special::ZetaOptions::default(),
    )
    .unwrap();
    assert_eq!(
        hurwitz.algorithm().method(),
        "hurwitz_scaled_euler_maclaurin"
    );
    assert_eq!(hurwitz.algorithm().family(), "hurwitz_zeta");
    assert!((12..=24).contains(&hurwitz.counts().iterations()));
}

#[test]
fn hurwitz_continuation_regressions_return_covered_values() {
    use scioxide_special::{ZetaOptions, hurwitz_zeta_with_options};
    for (s, a, reference) in [
        (-10.5_f64, 0.5, -0.011_138_425_697_785_333),
        (-3.5, 0.9, 7.606_520_077_228_851e-4),
        (-0.75, 0.25, 0.041_300_913_660_409_69),
        (-20.5, 0.5, 108.217_402_082_367_61),
        (-0.5, 0.656_936_321_399_175, 2.536e-17),
    ] {
        let report = hurwitz_zeta_with_options(s, a, ZetaOptions::default()).unwrap();
        let estimate = report.value().unwrap();
        let error = (estimate.value() - reference).abs();
        assert!(
            error <= estimate.absolute_error(),
            "s={s}, a={a}: {estimate:?}"
        );
        assert!(
            error <= 2e-13 * reference.abs() + 2e-16,
            "s={s}, a={a}: {estimate:?}"
        );
        assert_eq!(
            estimate.reliability(),
            scioxide_core::EstimateReliability::ModelBased
        );
        assert_eq!(
            hurwitz_zeta(s, a).unwrap().to_bits(),
            estimate.value().to_bits()
        );
        hurwitz_zeta(s, a).unwrap();
    }
    let a = 0.9_f32;
    let reference = hurwitz_zeta(-3.5_f64, f64::from(a)).unwrap();
    assert!((f64::from(hurwitz_zeta(-3.5_f32, a).unwrap()) - reference).abs() <= 1e-10);
    for s in [-60.0, -20.5, -10.5, -3.5, -0.5, 0.0, 0.75, 2.0] {
        let z = scioxide_core::Complex::new(s, 0.0);
        let one = scioxide_core::Complex::new(1.0, 0.0);
        assert_eq!(hurwitz_zeta(z, one).unwrap(), riemann_zeta(z).unwrap());
    }
}

#[test]
fn hurwitz_large_shifts_and_symmetry_zeros_remain_representable() {
    for a in [1e20_f64, 1e100, 1e300] {
        let value = hurwitz_zeta(0.5, a).unwrap();
        let reference = -2.0 * a.sqrt();
        assert!((value - reference).abs() <= 4e-15 * reference.abs());
    }
    // a^(3/2) itself overflows, but its coefficient -2/3 makes zeta finite.
    let a = 4e205_f64;
    let reference = (-2.0 / 3.0 * a) * a.sqrt();
    let value = hurwitz_zeta(-0.5, a).unwrap();
    assert!((value - reference).abs() <= 4e-15 * reference.abs());
    for s in [-8.0_f64, -20.0, -60.0] {
        assert_eq!(hurwitz_zeta(s, 0.5).unwrap(), 0.0);
        assert_eq!(hurwitz_zeta(s, 1.0).unwrap(), 0.0);
    }
    for s in [-8.0_f64.next_down(), -8.0, -8.0_f64.next_up()] {
        let (a, shifted) = (0.9, 1.9);
        let value = hurwitz_zeta(s, a).unwrap();
        let recurrence = hurwitz_zeta(s, shifted).unwrap() + a.powf(-s);
        assert!((value - recurrence).abs() <= 2e-14);
    }
}

#[test]
fn hurwitz_continuation_exhaustion_retains_admitted_work() {
    for (s, a) in [(-0.5_f64, 0.9), (-10.5, 0.5), (-60.0, 1000.0)] {
        let result = scioxide_special::hurwitz_zeta_with_options(
            s,
            a,
            scioxide_special::ZetaOptions::new(scioxide_core::Budget::ONE),
        );
        match result.unwrap_err().view() {
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                ..
            } => {
                assert_eq!(reason, scioxide_core::ConvergenceReason::IterationLimit);
                assert_eq!(diagnostics.counts().own().iterations(), 1);
            }
            error => panic!("unexpected exhaustion: {error:?}"),
        }
    }
}

#[test]
fn complex_hurwitz_and_unit_lerch_share_real_continuation() {
    use scioxide_core::Complex;
    use scioxide_special::{ZetaOptions, hurwitz_zeta_with_options, lerch_phi};
    let one = Complex::new(1.0, 0.0);
    for s in [-20.5_f64, -5.5, -2.5, -5.0, -2.0, 0.0] {
        for a in [0.25, 0.5, 1.5, 3.0, 10.0] {
            let (cs, ca) = (Complex::new(s, 0.0), Complex::new(a, 0.0));
            let expected = hurwitz_zeta(s, a).unwrap();
            let report = hurwitz_zeta_with_options(cs, ca, ZetaOptions::default()).unwrap();
            let estimate = report.value().unwrap();
            assert!((estimate.value().re - expected).abs() <= estimate.absolute_error());
            assert_eq!(estimate.value().im, 0.0);
            assert_eq!(lerch_phi(one, cs, ca).unwrap(), *estimate.value());
            assert_eq!(lerch_phi(1.0, s, a).unwrap(), expected);
            hurwitz_zeta(cs, ca).unwrap();
            lerch_phi(one, cs, ca).unwrap();
        }
    }
}

#[test]
fn complex_hurwitz_shift_and_conjugation_preserve_continuation() {
    use scioxide_core::Complex;
    for s in [
        Complex::new(-20.5, 1.0),
        Complex::new(-5.5, 0.25),
        Complex::new(-2.5, -1.0),
    ] {
        for a in [
            Complex::new(0.25, 0.0),
            Complex::new(1.5, 0.25),
            Complex::new(10.0, -0.25),
        ] {
            let value = hurwitz_zeta(s, a).unwrap();
            let shifted = hurwitz_zeta(s, a + Complex::new(1.0, 0.0)).unwrap();
            let term = (-s * a.ln()).exp();
            assert!(
                (value - shifted - term).norm()
                    <= 2e-12 * (value.norm() + shifted.norm() + term.norm())
            );
            assert!(
                (value - hurwitz_zeta(s.conj(), a.conj()).unwrap().conj()).norm()
                    <= 2e-13 * value.norm()
            );
        }
    }
}

#[test]
fn complex_continuation_exhaustion_retains_shared_work() {
    use scioxide_core::{Budget, Complex, ConvergenceReason, SciOxideErrorView};
    use scioxide_special::{ZetaOptions, hurwitz_zeta_with_options};
    for (s, a) in [
        (Complex::new(-20.5, 0.0), Complex::new(10.0, 0.0)),
        (Complex::new(-2.5, 1.0), Complex::new(10.0, 0.0)),
        (Complex::new(-5.5, 0.25), Complex::new(1.25, 0.25)),
    ] {
        let error = hurwitz_zeta_with_options(s, a, ZetaOptions::new(Budget::ONE)).unwrap_err();
        match error.view() {
            SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                ..
            } => {
                assert_eq!(reason, ConvergenceReason::IterationLimit);
                assert_eq!(diagnostics.counts().own().iterations(), 1);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[test]
fn unit_lerch_preserves_polynomial_continuation_at_zero_shift() {
    use scioxide_core::Complex;
    use scioxide_special::lerch_phi;
    for (s, expected) in [(0.0_f64, 0.5), (-1.0, -1.0 / 12.0), (-2.0, 0.0)] {
        let result = lerch_phi(
            Complex::new(1.0, 0.0),
            Complex::new(s, 0.0),
            Complex::new(0.0, 0.0),
        )
        .unwrap();
        assert!((result.re - expected).abs() <= 16.0 * f64::EPSILON);
        assert_eq!(result.im, 0.0);
        assert!((lerch_phi(1.0_f64, s, 0.0).unwrap() - expected).abs() <= 16.0 * f64::EPSILON);
    }
}
