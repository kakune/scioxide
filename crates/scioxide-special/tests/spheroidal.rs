#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::{RadialKind, SpectralOptions, SpheroidalKind as Kind, SpheroidalPlan};

fn independent_width<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/spheroidal_mpmath.tsv")
        .lines()
        .chain(include_str!("data/spheroidal_large.tsv").lines())
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        let number = |k: usize| S::try_from_f64_lossy(f[k].parse().unwrap()).unwrap();
        let kind = if f[0] == "pro" {
            Kind::Prolate
        } else {
            Kind::Oblate
        };
        let p = SpheroidalPlan::new(
            kind,
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            number(3),
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        let z = Complex::new(number(4), number(5));
        let r = if f[6] == "angular" {
            p.angular(z)
        } else {
            p.radial(
                z,
                match f[6] {
                    "r1" => RadialKind::First,
                    "r2" => RadialKind::Second,
                    "r3" => RadialKind::Third,
                    "r4" => RadialKind::Fourth,
                    _ => panic!("unknown reference operation"),
                },
            )
        }
        .unwrap();
        let value = r
            .try_value()
            .unwrap_or_else(|e| panic!("{row}: {e:?} {r:?}"));
        let derivative = *r.derivative().unwrap().value();
        let bound = if S::EPSILON.try_to_f64_exact().unwrap() > 1e-10 {
            2e-5
        } else {
            2e-9
        };
        for (a, b) in [
            (value, Complex::new(number(7), number(8))),
            (derivative, Complex::new(number(9), number(10))),
        ] {
            let error = (a - b).norm().try_to_f64_exact().unwrap();
            assert!(
                error < bound * (1.0 + b.norm().try_to_f64_exact().unwrap()),
                "{row}: value={value:?} derivative={derivative:?} error={error}"
            );
        }
    }
}

#[test]
fn independent_f64() {
    independent_width::<f64>();
}
#[test]
fn independent_f32() {
    independent_width::<f32>();
}

#[test]
fn larger_parameters_and_radial_connections() {
    for kind in [Kind::Prolate, Kind::Oblate] {
        for (m, n, c) in [(0, 0, 10.0_f64), (3, 8, 10.0), (0, 0, 100.0), (4, 12, 30.0)] {
            let p = SpheroidalPlan::new(
                kind,
                m,
                n,
                c,
                scioxide_special::SpectralNormalization::Standard,
                SpectralOptions::default(),
            )
            .unwrap();
            let negative = SpheroidalPlan::new(
                kind,
                m,
                n,
                -c,
                scioxide_special::SpectralNormalization::Standard,
                SpectralOptions::default(),
            )
            .unwrap();
            assert_eq!(
                p.characteristic_value().unwrap(),
                negative.characteristic_value().unwrap()
            );
            for z in [Complex::new(2.0, 0.0), Complex::new(2.0, 0.125)] {
                let a = p.radial(z, RadialKind::First).unwrap();
                let b = p.radial(z, RadialKind::Second).unwrap();
                let av = a
                    .try_value()
                    .unwrap_or_else(|e| panic!("{kind:?} {m} {n} {c} {z:?}: {e:?} {a:?}"));
                let bv = b
                    .try_value()
                    .unwrap_or_else(|e| panic!("{kind:?} {m} {n} {c} {z:?}: {e:?} {b:?}"));
                let w =
                    av * *b.derivative().unwrap().value() - bv * *a.derivative().unwrap().value();
                let expected = Complex::new(1.0 / c, 0.0)
                    / (z * z + Complex::new(if kind == Kind::Prolate { -1.0 } else { 1.0 }, 0.0));
                let conditioning = av.norm() * b.derivative().unwrap().value().norm()
                    + bv.norm() * a.derivative().unwrap().value().norm();
                assert!(
                    (w - expected).norm()
                        < 1e-8 * expected.norm() + 128.0 * f64::EPSILON * conditioning,
                    "{kind:?} {m} {n} {c} {z:?}: w={w:?} expected={expected:?}"
                );
                for (k, s) in [(RadialKind::Third, 1.0), (RadialKind::Fourth, -1.0)] {
                    let r = p.radial(z, k).unwrap().try_value().unwrap();
                    assert!(
                        (r - av - Complex::new(0.0, s) * bv).norm()
                            < 1e-9 * (1.0 + av.norm() + bv.norm())
                    );
                }
            }
        }
    }
}

#[test]
fn complex_continuation_and_cut_sides() {
    for kind in [Kind::Prolate, Kind::Oblate] {
        let p = SpheroidalPlan::new(
            kind,
            1,
            2,
            1.0_f64,
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        for z in [
            Complex::new(0.25, 0.125),
            Complex::new(-0.5, 0.125),
            Complex::new(-2.0, 0.125),
        ] {
            let a = p.radial(z, RadialKind::First).unwrap();
            let b = p.radial(z, RadialKind::Second).unwrap();
            let av = a.try_value().unwrap();
            let bv = b
                .try_value()
                .unwrap_or_else(|e| panic!("{kind:?} {z:?}: {e:?} {b:?}"));
            let w = av * *b.derivative().unwrap().value() - bv * *a.derivative().unwrap().value();
            let expected = Complex::new(1.0, 0.0)
                / (z * z + Complex::new(if kind == Kind::Prolate { -1.0 } else { 1.0 }, 0.0));
            assert!((w - expected).norm() < 1e-8 * (1.0 + expected.norm()));
            for (k, v) in [(RadialKind::First, av), (RadialKind::Second, bv)] {
                let lower = p.radial(z.conj(), k).unwrap().try_value().unwrap();
                assert!(
                    (v - lower.conj()).norm() < 1e-9 * (1.0 + v.norm()),
                    "{kind:?} {z:?} {k:?}: {v:?} lower={lower:?}"
                );
            }
        }
        if kind == Kind::Prolate {
            for k in [RadialKind::First, RadialKind::Second] {
                let upper = p
                    .radial(Complex::new(0.25, 0.0), k)
                    .unwrap()
                    .try_value()
                    .unwrap();
                let lower = p
                    .radial(Complex::new(0.25, -0.0), k)
                    .unwrap()
                    .try_value()
                    .unwrap();
                assert!((upper - lower.conj()).norm() < 1e-9 * (1.0 + upper.norm()));
                assert!(upper.im.abs() > 1e-4);
            }
        }
    }
}

#[test]
fn orthogonality_and_scaled_large_order_normalization() {
    use scioxide_special::{ScalarRepresentation, SpectralNormalization};
    for kind in [Kind::Prolate, Kind::Oblate] {
        let options = SpectralOptions::default();
        let a = SpheroidalPlan::new(kind, 1, 1, 2.0_f64, SpectralNormalization::UnitL2, options)
            .unwrap();
        let b = SpheroidalPlan::new(kind, 1, 3, 2.0_f64, SpectralNormalization::UnitL2, options)
            .unwrap();
        let integral = (0..4096)
            .map(|k| {
                let x = -1.0 + (k as f64 + 0.5) / 2048.0;
                a.angular(x).unwrap().try_value().unwrap()
                    * b.angular(x).unwrap().try_value().unwrap()
            })
            .sum::<f64>()
            / 2048.0;
        assert!(integral.abs() < 5e-7, "{kind:?} inner product {integral}");
    }
    let p = SpheroidalPlan::new(
        Kind::Prolate,
        200,
        200,
        0.0_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let r = p.angular(0.0_f64).unwrap();
    assert!(r.termination().is_success());
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(r.value().is_none());
    assert!(
        r.scaled()
            .unwrap()
            .value()
            .try_projective()
            .unwrap()
            .exponent()
            > 1024
    );
    // P_m^m(0)=(-1)^m(2m-1)!!, independently accumulated in log space.
    let logarithm = (1..=200).map(|k| ((2 * k - 1) as f64).ln()).sum::<f64>();
    assert!((r.logarithmic().unwrap().value().log_magnitude().unwrap() - logarithm).abs() < 1e-10);
}

#[test]
fn regular_focal_limits_and_zero_parameter() {
    for m in 0..=4 {
        for n in m..=m + 2 {
            let p = SpheroidalPlan::new(
                Kind::Prolate,
                m,
                n,
                1.0_f64,
                scioxide_special::SpectralNormalization::Standard,
                SpectralOptions::default(),
            )
            .unwrap();
            for x in [-1.0_f64, 1.0] {
                let r = p.radial(x, RadialKind::First).unwrap();
                let at = r.try_value().unwrap();
                let near = p
                    .radial(x * (1.0 + 1e-10), RadialKind::First)
                    .unwrap()
                    .try_value()
                    .unwrap();
                assert!(
                    (near - at).norm() < 1e-5,
                    "m={m} n={n} x={x} eta={} at={at:?} near={near:?}",
                    p.radial_normalization_point()
                );
                assert_eq!(r.derivative().is_none(), m == 1);
                if m > 0 {
                    assert_eq!(at, Complex::new(0.0, 0.0));
                }
                assert!(p.radial(x, RadialKind::Second).is_err());
            }
            let p = SpheroidalPlan::new(
                Kind::Prolate,
                m,
                n,
                0.0_f64,
                scioxide_special::SpectralNormalization::Standard,
                SpectralOptions::default(),
            )
            .unwrap();
            let actual = p.angular(0.25_f64).unwrap().try_value().unwrap();
            let expected =
                scioxide_special::legendre_p_assoc(n as usize, m as i32, 0.25_f64).unwrap();
            assert!((actual - expected).abs() < 1e-12 * (1.0 + expected.abs()));
        }
    }
}

#[test]
fn angular_equation_normalization_and_tiny_imaginary_part() {
    use scioxide_special::SpectralNormalization;
    for kind in [Kind::Prolate, Kind::Oblate] {
        let p = SpheroidalPlan::new(
            kind,
            1,
            3,
            2.0_f64,
            SpectralNormalization::UnitL2,
            SpectralOptions::default(),
        )
        .unwrap();
        let z = Complex::new(0.25, 0.125);
        let r = p.angular(z).unwrap();
        let v = r.try_value().unwrap();
        let d = *r.derivative().unwrap().value();
        let h = Complex::new(1e-5, 0.0);
        let second = (*p.angular(z + h).unwrap().derivative().unwrap().value()
            - *p.angular(z - h).unwrap().derivative().unwrap().value())
            / (2.0 * h.re);
        let potential = Complex::new(p.characteristic_value().unwrap(), 0.0)
            - z * z * if kind == Kind::Prolate { 4.0 } else { -4.0 }
            - Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - z * z);
        assert!(
            ((Complex::new(1.0, 0.0) - z * z) * second - z * 2.0 * d + potential * v).norm() < 1e-7
        );
        let integral = (0..4096)
            .map(|k| {
                p.angular(-1.0 + (k as f64 + 0.5) / 2048.0)
                    .unwrap()
                    .try_value()
                    .unwrap()
                    .powi(2)
            })
            .sum::<f64>()
            / 2048.0;
        assert!((integral - 1.0).abs() < 1e-6);
        for x in [-0.25, 0.25, 0.75] {
            let h = 1e-100;
            let d = *p.angular(x).unwrap().derivative().unwrap().value();
            let complex = p.angular(Complex::new(x, h)).unwrap().try_value().unwrap();
            assert!(
                (complex.im / h - d).abs() < 1e-10,
                "{kind:?} x={x} complex={complex:?} d={d}"
            );
        }
    }
}

#[test]
fn exterior_references() {
    for row in include_str!("data/spheroidal_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let v: Vec<_> = row.split_whitespace().collect();
        let kind = if v[0] == "pro" {
            Kind::Prolate
        } else {
            Kind::Oblate
        };
        let p = SpheroidalPlan::new(
            kind,
            v[1].parse().unwrap(),
            v[2].parse().unwrap(),
            v[3].parse::<f64>().unwrap(),
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        let x: f64 = v[4].parse().unwrap();
        let characteristic = v[5].parse::<f64>().unwrap();
        assert!(
            (p.characteristic_value().unwrap() - characteristic).abs()
                < 1e-12 * (1.0 + characteristic.abs()),
            "{row}"
        );
        for (k, offset) in [(RadialKind::First, 6), (RadialKind::Second, 8)] {
            let r = p.radial(x, k).unwrap();
            let value = r.try_value().unwrap();
            let derivative = *r.derivative().unwrap().value();
            for (a, b) in [
                (value, v[offset].parse::<f64>().unwrap()),
                (derivative, v[offset + 1].parse::<f64>().unwrap()),
            ] {
                assert!(
                    (a.re - b).abs() < 2e-9 * (1.0 + b.abs()) && a.im.abs() < 1e-10,
                    "{row} {k:?} eta={} got={a:?} expected={b} report={r:?}",
                    p.radial_normalization_point()
                );
            }
        }
    }
}

#[test]
fn interior_wronskians() {
    for kind in [Kind::Prolate, Kind::Oblate] {
        for (m, n, c) in [
            (0, 0, 1.0_f64),
            (0, 1, 1.0),
            (1, 1, 1.0),
            (1, 2, 2.0),
            (2, 4, 0.5),
        ] {
            let p = SpheroidalPlan::new(
                kind,
                m,
                n,
                c,
                scioxide_special::SpectralNormalization::Standard,
                SpectralOptions::default(),
            )
            .unwrap();
            for x in if kind == Kind::Prolate {
                [1.1, 1.5]
            } else {
                [0.0, 0.5]
            } {
                let a = p.radial(x, RadialKind::First).unwrap();
                let b = p.radial(x, RadialKind::Second).unwrap();
                let av = a.try_value().unwrap();
                let bv = b
                    .try_value()
                    .unwrap_or_else(|e| panic!("{kind:?} m={m} n={n} x={x}: {e:?} {b:?}"));
                let w =
                    av * *b.derivative().unwrap().value() - bv * *a.derivative().unwrap().value();
                let expected = 1.0 / (c * (x * x + if kind == Kind::Prolate { -1.0 } else { 1.0 }));
                assert!(
                    (w.re - expected).abs() < 1e-8 * expected.abs() && w.im.abs() < 1e-9,
                    "{kind:?} m={m} n={n} c={c} x={x} eta={} w={w:?} expected={expected}",
                    p.radial_normalization_point()
                );
            }
        }
    }
}

#[test]
fn real_angular_reports_cover_independent_fixtures() {
    for row in include_str!("data/spheroidal_mpmath.tsv")
        .lines()
        .chain(include_str!("data/spheroidal_large.tsv").lines())
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields[6] != "angular" || fields[5].parse::<f64>().unwrap() != 0.0 {
            continue;
        }
        let number = |i: usize| fields[i].parse::<f64>().unwrap();
        let kind = if fields[0] == "pro" {
            Kind::Prolate
        } else {
            Kind::Oblate
        };
        let plan = SpheroidalPlan::new(
            kind,
            fields[1].parse().unwrap(),
            fields[2].parse().unwrap(),
            number(3),
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        let report = plan.angular(number(4)).unwrap();
        assert!(report.termination().is_success());
        for (actual, expected) in [
            (report.value().unwrap(), number(7)),
            (report.derivative().unwrap(), number(9)),
        ] {
            assert!(
                (actual.value() - expected).abs()
                    <= actual.absolute_error() + f64::EPSILON * expected.abs(),
                "{row}: {actual:?} != {expected}"
            );
        }
    }
}

#[test]
fn real_and_complex_recurrences_agree_at_focal_endpoints_and_interior() {
    use scioxide_special::SpectralNormalization;
    for kind in [Kind::Prolate, Kind::Oblate] {
        for normalization in [
            SpectralNormalization::Standard,
            SpectralNormalization::UnitL2,
        ] {
            for m in 0..=4 {
                for c in [0.0_f64, -1.25] {
                    let plan = SpheroidalPlan::new(
                        kind,
                        m,
                        m + 3,
                        c,
                        normalization,
                        SpectralOptions::default(),
                    )
                    .unwrap();
                    let narrow = SpheroidalPlan::new(
                        kind,
                        m,
                        m + 3,
                        c as f32,
                        normalization,
                        SpectralOptions::default(),
                    )
                    .unwrap();
                    for x in [
                        -1.0_f64,
                        -0.999999999999,
                        -0.5,
                        0.0,
                        0.5,
                        0.999999999999,
                        1.0,
                    ] {
                        let real = plan.angular(x).unwrap();
                        let complex = plan.angular(Complex::new(x, 0.0)).unwrap();
                        assert!(real.termination().is_success());
                        assert_eq!(real.derivative().is_none(), m == 1 && x.abs() == 1.0);
                        assert_eq!(complex.derivative().is_none(), real.derivative().is_none());
                        for (r, z) in [
                            (real.value(), complex.value()),
                            (real.derivative(), complex.derivative()),
                        ] {
                            if let (Some(r), Some(z)) = (r, z) {
                                assert!(
                                    (r.value() - z.value().re).abs()
                                        <= 8.0 * f64::EPSILON * r.value().abs().max(1.0)
                                );
                                assert_eq!(z.value().im, 0.0);
                            }
                        }
                        // Compare at the same representable binary32 coordinate.
                        let reference = plan.angular(f64::from(x as f32)).unwrap();
                        let actual = narrow.angular(x as f32).unwrap();
                        assert_eq!(
                            actual.derivative().is_none(),
                            reference.derivative().is_none()
                        );
                        for (a, b) in [
                            (actual.value(), reference.value()),
                            (actual.derivative(), reference.derivative()),
                        ] {
                            if let (Some(a), Some(b)) = (a, b) {
                                assert!(
                                    (f64::from(*a.value()) - b.value()).abs()
                                        <= 8.0 * f64::from(f32::EPSILON) * b.value().abs().max(1.0)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn bits<V: NumericScalar>(value: V) -> (u64, u64) {
    (
        value.real_component().into_f64().to_bits(),
        value.imaginary_component().into_f64().to_bits(),
    )
}

/// Each one-shot returns the bits of the plan with the standard normalization and default
/// options, for both kinds and every radial kind.
fn one_shots_match_standard_plan<T: NumericScalar>(angular: T, radial: T)
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    use scioxide_special::{
        SpectralNormalization, spheroidal_angular, spheroidal_characteristic_value,
        spheroidal_radial,
    };
    for kind in [Kind::Prolate, Kind::Oblate] {
        for parameter in [1.5, -2.5, 0.0] {
            let c = T::Real::try_from_f64_lossy(parameter).unwrap();
            for (m, n) in [(0, 0), (0, 3), (1, 2), (2, 4)] {
                let plan = SpheroidalPlan::new(
                    kind,
                    m,
                    n,
                    c,
                    SpectralNormalization::Standard,
                    Default::default(),
                )
                .unwrap();
                let case = format!("{kind:?} m={m} n={n} c={c:?}");
                assert_eq!(
                    bits(spheroidal_characteristic_value(kind, m, n, c).unwrap()),
                    bits(plan.characteristic_value().unwrap()),
                    "{case}"
                );
                assert_eq!(
                    bits(spheroidal_angular(kind, m, n, c, angular).unwrap()),
                    bits(plan.angular(angular).unwrap().try_value().unwrap()),
                    "{case} {angular:?}"
                );
                let kinds: &[RadialKind] = if parameter == 0.0 {
                    &[RadialKind::First]
                } else {
                    &[
                        RadialKind::First,
                        RadialKind::Second,
                        RadialKind::Third,
                        RadialKind::Fourth,
                    ]
                };
                for &radial_kind in kinds {
                    assert_eq!(
                        bits(spheroidal_radial(kind, m, n, c, radial, radial_kind).unwrap()),
                        bits(
                            plan.radial(radial, radial_kind)
                                .unwrap()
                                .try_value()
                                .unwrap()
                        ),
                        "{case} {radial:?} {radial_kind:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn one_shots_equal_the_standard_plan_bit_for_bit() {
    one_shots_match_standard_plan(0.3_f32, 1.5_f32);
    one_shots_match_standard_plan(0.3_f64, 1.5_f64);
    one_shots_match_standard_plan(Complex::new(0.3_f64, 0.2), Complex::new(1.5_f64, 0.25));
}

#[test]
fn one_shots_reduce_to_ferrers_functions_at_zero_size() {
    use scioxide_special::{
        spheroidal_angular, spheroidal_characteristic_value, spheroidal_radial,
    };
    for kind in [Kind::Prolate, Kind::Oblate] {
        for (m, n) in [(0, 0), (0, 2), (1, 1), (1, 3), (3, 5)] {
            let lambda = spheroidal_characteristic_value(kind, m, n, 0.0_f64).unwrap();
            assert!((lambda - f64::from(n * (n + 1))).abs() < 1e-12);
            let r = spheroidal_radial(kind, m, n, 0.0, 1.5_f64, RadialKind::First).unwrap();
            let expected = if n == 0 { 1.0 } else { 0.0 };
            assert!((r - Complex::new(expected, 0.0)).norm() < 1e-12, "R_{m}{n}");
        }
        for x in [-0.75_f64, 0.0, 0.5] {
            let root = (1.0 - x * x).sqrt();
            // Ferrers functions with the Condon–Shortley phase (DLMF 14.7).
            for (m, n, expected) in [
                (0, 1, x),
                (0, 2, (3.0 * x * x - 1.0) / 2.0),
                (1, 1, -root),
                (1, 2, -3.0 * x * root),
                (2, 2, 3.0 * root * root),
            ] {
                let s = spheroidal_angular(kind, m, n, 0.0, x).unwrap();
                assert!((s - expected).abs() < 1e-12, "S_{m}{n}({x}) = {s}");
            }
        }
        let x = Complex::new(0.5_f64, 0.25);
        let s = spheroidal_angular(kind, 0, 2, 0.0, x).unwrap();
        assert!((s - (x * x * 3.0 - Complex::new(1.0, 0.0)) / 2.0).norm() < 1e-12);
    }
}

#[test]
fn one_shots_report_parameter_and_coordinate_errors() {
    use scioxide_core::{DomainReason, InvalidInputReason, SciOxideErrorKind, SciOxideErrorView};
    use scioxide_special::{
        spheroidal_angular, spheroidal_characteristic_value, spheroidal_radial,
    };
    let kind = Kind::Prolate;
    for error in [
        spheroidal_characteristic_value(kind, 2, 1, 1.0_f64).unwrap_err(),
        spheroidal_angular(kind, 2, 1, 1.0, 0.5_f64).unwrap_err(),
        spheroidal_radial(kind, 2, 1, 1.0, 1.5_f64, RadialKind::First).unwrap_err(),
    ] {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::InvalidInput {
                    argument,
                    reason: InvalidInputReason::OutOfRange,
                    ..
                } if argument == scioxide_core::argument!("n")
            ),
            "{error}"
        );
    }
    let nan = f64::NAN;
    for error in [
        spheroidal_characteristic_value(kind, 0, 2, nan).unwrap_err(),
        spheroidal_angular(kind, 0, 2, nan, 0.5).unwrap_err(),
        spheroidal_angular(kind, 0, 2, 1.0, nan).unwrap_err(),
        spheroidal_angular(kind, 0, 2, 1.0, Complex::new(0.5, nan)).unwrap_err(),
        spheroidal_radial(kind, 0, 2, nan, 1.5, RadialKind::First).unwrap_err(),
        spheroidal_radial(kind, 0, 2, 1.0, f64::INFINITY, RadialKind::First).unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite, "{error}");
    }
    for x in [1.0_f64.next_up(), -1.5] {
        let error = spheroidal_angular(kind, 0, 2, 1.0, x).unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::DomainError {
                    argument,
                    reason: DomainReason::OutsideDomain,
                    ..
                } if argument == scioxide_core::argument!("x")
            ),
            "{error}"
        );
    }
    assert!(spheroidal_angular(kind, 0, 2, 1.0, 1.0_f64).is_ok());
    for (c, x, argument) in [
        (0.0, 1.5, scioxide_core::argument!("c")),
        (1.0, 1.0, scioxide_core::argument!("x")),
    ] {
        let error = spheroidal_radial(kind, 0, 0, c, x, RadialKind::Second).unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::DomainError {
                    argument: actual,
                    reason: DomainReason::Pole,
                    ..
                } if actual == argument
            ),
            "{error}"
        );
    }
}
