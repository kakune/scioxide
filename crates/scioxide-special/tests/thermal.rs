#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar};
use scioxide_special::*;

fn evaluate<S: RealScalar>(
    kind: u32,
    order: f64,
    x: S,
    b: S,
    budget: usize,
) -> Result<SpecialFunctionResult<S>, scioxide_core::SciOxideError> {
    let work = Budget::try_from(budget).unwrap();
    if kind == 0 {
        let options = DebyeOptions::new(work);
        match order as u32 {
            1 => debye_1_with_options(x, options),
            2 => debye_2_with_options(x, options),
            3 => debye_3_with_options(x, options),
            4 => debye_4_with_options(x, options),
            5 => debye_5_with_options(x, options),
            _ => debye_6_with_options(x, options),
        }
    } else {
        let options = FermiDiracOptions::new(work);
        if kind == 2 {
            fermi_dirac_inc_0_with_options(x, b, options)
        } else if order == -0.5 {
            fermi_dirac_mhalf_with_options(x, options)
        } else if order == 0.5 {
            fermi_dirac_half_with_options(x, options)
        } else if order == 1.5 {
            fermi_dirac_3half_with_options(x, options)
        } else {
            fermi_dirac_int_with_options(order as i32, x, options)
        }
    }
}

fn check<S: RealScalar>(kind: u32, order: f64, x: S, b: S, expected: S, log_expected: S) {
    let r = evaluate(kind, order, x, b, 65536).unwrap();
    let f32 = S::MANTISSA_DIGITS == 24;
    let relative = if f32 { 2e-4 } else { 5e-11 };
    let absolute = if f32 { 5e-6 } else { 5e-13 };
    let value = expected.into_f64();
    assert!(
        r.termination().is_success(),
        "kind={kind} j={order} x={x:?}: {r:?}"
    );
    if let Some(v) = r.value() {
        let actual = v.value().into_f64();
        let error = (actual - value).abs();
        if expected.is_finite() && expected.abs() < S::MIN_POSITIVE {
            let distance = if f32 {
                u64::from((actual as f32).to_bits().abs_diff((value as f32).to_bits()))
            } else {
                actual.to_bits().abs_diff(value.to_bits())
            };
            // Treat neutral zero signs alike; nonzero subnormal signs matter.
            if value != 0.0 {
                assert_eq!(actual.is_sign_negative(), value.is_sign_negative());
                assert!(
                    distance <= if f32 { 16 } else { 8 },
                    "subnormal kind={kind} j={order} x={x:?}: distance={distance}"
                );
            } else {
                assert_eq!(actual, 0.0, "underflow kind={kind} j={order} x={x:?}");
            }
        }
        if value.is_finite() {
            assert!(
                error <= absolute + relative * value.abs(),
                "kind={kind} j={order} x={x:?}: {actual} vs {value}, {r:?}"
            );
        }
        if value.is_finite() {
            let ulp = expected.next_after(S::INFINITY).into_f64() - value;
            assert!(
                error <= v.absolute_error().into_f64() + ulp.abs(),
                "uncovered error kind={kind} j={order} x={x:?}: {actual} vs {value}, {r:?}"
            );
        }
    } else {
        assert!(
            !expected.is_finite()
                || r.scalar_representation() == ScalarRepresentation::Overflow
                || !r.termination().is_success()
        );
    }
    if let Some(l) = r.logarithmic()
        && let Some(actual) = l.value().log_magnitude()
        && (value != 0.0 || log_expected != S::ZERO)
    {
        let error = (actual.into_f64() - log_expected.into_f64()).abs();
        let rounding =
            (log_expected.next_after(S::INFINITY).into_f64() - log_expected.into_f64()).abs();
        assert!(
            error <= l.absolute_error().into_f64() + rounding,
            "uncovered log kind={kind} j={order} x={x:?}: {actual:?} vs {log_expected:?}, {r:?}"
        );
    }
    if let Some(s) = r.scaled() {
        let m = s.value().try_projective().unwrap().significand().abs();
        assert!(m == S::ZERO || (m >= S::ONE && m < S::ONE + S::ONE));
        if m != S::ZERO {
            let implied_log = m.into_f64().ln()
                + s.value().try_projective().unwrap().exponent() as f64 * core::f64::consts::LN_2;
            let error = (implied_log - log_expected.into_f64()).abs();
            let rounding =
                (log_expected.next_after(S::INFINITY).into_f64() - log_expected.into_f64()).abs();
            let allowance =
                s.absolute_error().into_f64() + 8.0 * f64::EPSILON * implied_log.abs() + rounding;
            assert!(
                error <= allowance,
                "scaled kind={kind} j={order} x={x:?}: {r:?}"
            );
        }
    }
    assert!(r.counts().iterations() <= 65536);
}

#[test]
fn independent_oracles() {
    for line in include_str!("data/thermal_oracle.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let width: u32 = fields[0].parse().unwrap();
        let kind = fields[1].parse().unwrap();
        let order = fields[2].parse().unwrap();
        let bits: Vec<_> = fields[3..]
            .iter()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        if width == 32 {
            check(
                kind,
                order,
                f32::from_bits(bits[0] as u32),
                f32::from_bits(bits[1] as u32),
                f32::from_bits(bits[2] as u32),
                f32::from_bits(bits[3] as u32),
            );
        } else {
            check(
                kind,
                order,
                f64::from_bits(bits[0]),
                f64::from_bits(bits[1]),
                f64::from_bits(bits[2]),
                f64::from_bits(bits[3]),
            );
        }
    }
}

#[test]
fn public_forms_domains_and_identities() {
    for x in [0.0, 0.1, 1.0, 4.0, 100.0] {
        let plain = [
            debye_1(x),
            debye_2(x),
            debye_3(x),
            debye_4(x),
            debye_5(x),
            debye_6(x),
        ];
        for (i, value) in plain.into_iter().enumerate() {
            assert_eq!(
                value.unwrap(),
                evaluate(0, (i + 1) as f64, x, 0.0, 65536)
                    .unwrap()
                    .try_value()
                    .unwrap()
            );
        }
    }
    for x in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        for (j, value) in [
            (-1, fermi_dirac_m1(x)),
            (0, fermi_dirac_0(x)),
            (1, fermi_dirac_1(x)),
            (2, fermi_dirac_2(x)),
        ] {
            assert_eq!(value.unwrap(), fermi_dirac_int(j, x).unwrap());
        }
        assert_eq!(
            fermi_dirac_inc_0(x, 0.0).unwrap(),
            fermi_dirac_0(x).unwrap()
        );
        assert!(fermi_dirac_mhalf(x).unwrap() > 0.0);
        assert!(fermi_dirac_half(x).unwrap() > 0.0);
        assert!(fermi_dirac_3half(x).unwrap() > 0.0);
    }
    assert_eq!(debye_6(-0.0).unwrap(), 1.0);
    assert_eq!(fermi_dirac_m1(0.0).unwrap(), 0.5);
    assert_eq!(fermi_dirac_int(-3, 0.0).unwrap(), 0.0);
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(debye_1(x).is_err());
        assert!(fermi_dirac_half(x).is_err());
        assert!(fermi_dirac_inc_0(0.0, x).is_err());
    }
    assert!(debye_1(-1.0).is_err());
    assert!(fermi_dirac_int(-102, 0.0).is_err());
    assert!(fermi_dirac_int(i32::MIN, 0.0).is_err());
    assert!(fermi_dirac_inc_0(0.0, -1.0).is_err());
    // F'_j=F_(j-1), and x D'_n=n[x/(exp(x)-1)-D_n].
    let h: f64 = 1e-4;
    for x in [0.1, 2.0, 10.0, 40.0] {
        for j in [-1, 0, 1, 2, 3, 10] {
            let derivative = (fermi_dirac_int(j, x + h).unwrap()
                - fermi_dirac_int(j, x - h).unwrap())
                / (2.0 * h);
            let target = fermi_dirac_int(j - 1, x).unwrap();
            assert!((derivative - target).abs() < 1e-6 * target.abs().max(1.0));
        }
        let derivative = (debye_3(x + h).unwrap() - debye_3(x - h).unwrap()) / (2.0 * h);
        let target = 3.0 * (x / x.exp_m1() - debye_3(x).unwrap()) / x;
        assert!((derivative - target).abs() < 1e-8);
    }
}

#[test]
fn budgets_tolerance_and_extremes() {
    for (kind, j, x) in [
        (0, 3.0, 1.0),
        (0, 3.0, 10.0),
        (1, 1.5, 1.0),
        (1, 1.5, 100.0),
        (1, 10.0, 3.0),
        (1, -101.0, 1.0),
    ] {
        assert!(publication::budget_counts(evaluate(kind, j, x, 0.0, 1)).iterations() <= 1);
    }
    assert_eq!(
        debye_3_with_options(1.0, DebyeOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    let r = fermi_dirac_half_with_options(f64::MAX, FermiDiracOptions::default()).unwrap();
    assert!(r.value().is_none());
    assert!(r.logarithmic().is_some());
    assert!(r.scaled().is_some());
    let r = fermi_dirac_half_with_options(-1000.0, FermiDiracOptions::default()).unwrap();
    assert_eq!(r.try_value().unwrap(), 0.0);
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Finite);
    assert!(r.logarithmic().is_some());
    assert_eq!(fermi_dirac_inc_0(-f64::MAX, f64::MAX).unwrap(), 0.0);
    let limit =
        fermi_dirac_inc_0_with_options(-f64::MAX, f64::MAX, FermiDiracOptions::default()).unwrap();
    assert_eq!(limit.termination(), SpecialFunctionTermination::Limit);
    assert_eq!(limit.scalar_representation(), ScalarRepresentation::Finite);
    assert!(limit.scaled().is_none() && limit.logarithmic().is_none());
    let remote = fermi_dirac_half_with_options(-1e15_f64, FermiDiracOptions::default()).unwrap();
    assert!(remote.scaled().is_none());
    assert!(remote.logarithmic().is_some());
    // Very large orders with a small argument finish from a bounded polynomial
    // tail; a large order/argument pair must respect the caller's work limit.
    assert!((fermi_dirac_int(i32::MAX, 0.1).unwrap() - 0.1_f64.exp()).abs() < 1e-12);
    let r = fermi_dirac_int_with_options(
        i32::MAX,
        1e9,
        FermiDiracOptions::default().with_iteration_budget(Budget::try_from(16).unwrap()),
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 16);
}

#[test]
fn adjacent_regime_inputs_and_monotonicity() {
    for boundary in [-1.0_f64, 1.0, 4.0, 10.0, 30.0] {
        let left = boundary.next_after(f64::NEG_INFINITY);
        let right = boundary.next_after(f64::INFINITY);
        for j in [-0.5, 0.5, 1.5, 1.0, 2.0, 3.0, 100.0] {
            let a = evaluate(1, j, left, 0.0, 65536)
                .unwrap()
                .try_value()
                .unwrap();
            let b = evaluate(1, j, right, 0.0, 65536)
                .unwrap()
                .try_value()
                .unwrap();
            assert!(
                (a - b).abs() < 1e-10 * a.abs().max(1.0),
                "j={j}, boundary={boundary}"
            );
        }
        if boundary > 0.0 {
            let a = debye_6(left).unwrap();
            let b = debye_6(right).unwrap();
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn binary32_projection_boundary() {
    for j in 1..=20 {
        let power = f64::from(j + 1);
        let threshold =
            (f64::from(f32::MAX) * gamma(f64::from(j + 2)).unwrap()).powf(1.0 / power) as f32;
        for x in [
            threshold.next_after(0.0),
            threshold,
            threshold.next_after(f32::INFINITY),
        ] {
            let wide = fermi_dirac_int_with_options(j, f64::from(x), FermiDiracOptions::default())
                .unwrap();
            let value = *wide.value().unwrap().value();
            let expected = value as f32;
            let narrow = fermi_dirac_int_with_options(j, x, FermiDiracOptions::default()).unwrap();
            if expected.is_finite() {
                let actual = *narrow.value().unwrap().value();
                assert_eq!(actual, expected, "order={j}, x={x}");
            } else {
                assert_eq!(
                    narrow.scalar_representation(),
                    ScalarRepresentation::Overflow
                );
                assert!(narrow.value().is_none());
            }
        }
    }
    let r = debye_1_with_options(0.0_f64, DebyeOptions::default()).unwrap();
    assert_eq!(r.value().unwrap().absolute_error(), 0.0);
}
