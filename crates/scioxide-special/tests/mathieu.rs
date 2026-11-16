#![allow(clippy::unwrap_used)]
use scioxide_core::Complex;
use scioxide_special::{
    MathieuParity as Parity, MathieuPlan, SpectralNormalization, SpectralOptions,
};
#[test]
fn characteristic_references_and_zero_parameter() {
    // SciPy 1.18.0 real characteristic values; independent residuals below.
    for (order, parity, reference) in [
        (0, Parity::Even, -0.45513860410741364),
        (1, Parity::Even, 1.8591080725143634),
        (1, Parity::Odd, -0.11024881699209521),
        (2, Parity::Even, 4.371300982735086),
        (2, Parity::Odd, 3.917024772998471),
    ] {
        let p = MathieuPlan::new(
            order,
            parity,
            1.0_f64,
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        assert!((p.characteristic_value().unwrap() - reference).abs() < 2e-12);
        assert!(p.diagnostics().eigen_residual() < 1e-11);
        assert!(p.diagnostics().refinements() > 0);
    }
    for n in 0..6 {
        let p = MathieuPlan::new(
            n,
            Parity::Even,
            0.0_f64,
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        assert_eq!(p.characteristic_value().unwrap(), f64::from(n * n));
        let actual = p.angular(0.25_f64).unwrap().try_value().unwrap();
        let expected = if n == 0 {
            core::f64::consts::FRAC_1_SQRT_2
        } else {
            (f64::from(n) * 0.25).cos()
        };
        assert!((actual - expected).abs() < 1e-14);
    }
}
#[test]
fn complex_conjugation_ode_and_normalization() {
    let p = MathieuPlan::new(
        3,
        Parity::Even,
        2.0_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let z = Complex::new(0.5, 0.125);
    let r = p.angular(z).unwrap();
    let a = r.try_value().unwrap();
    let b = p.angular(z.conj()).unwrap().try_value().unwrap();
    assert!((a - b.conj()).norm() < 1e-12);
    let h = 1e-4;
    let plus = p.angular(z + Complex::new(h, 0.0)).unwrap();
    let minus = p.angular(z - Complex::new(h, 0.0)).unwrap();
    let second =
        (*plus.derivative().unwrap().value() - *minus.derivative().unwrap().value()) / (2.0 * h);
    let expected = (Complex::new(4.0, 0.0) * (z * 2.0).cos()
        - Complex::new(p.characteristic_value().unwrap(), 0.0))
        * a;
    assert!((second - expected).norm() < 1e-6);
    let options = SpectralOptions::default();
    let unit = MathieuPlan::new(
        3,
        Parity::Even,
        2.0_f64,
        SpectralNormalization::UnitL2,
        options,
    )
    .unwrap();
    let sum = (0..512)
        .map(|j| {
            let v = unit
                .angular((j as f64 + 0.5) * core::f64::consts::TAU / 512.0)
                .unwrap()
                .try_value()
                .unwrap();
            v * v
        })
        .sum::<f64>()
        * core::f64::consts::TAU
        / 512.0;
    assert!((sum - 1.0).abs() < 1e-11);
}
#[test]
fn radial_references_and_wronskians() {
    use scioxide_special::RadialKind;

    let p = MathieuPlan::new(
        0,
        Parity::Even,
        1.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let r = p.radial(0.5_f64, RadialKind::First).unwrap();
    let value = r.try_value().unwrap();
    assert!(
        (value.re - 0.4651064067337987).abs() < 5e-11 && value.im.abs() < 5e-12,
        "{r:?}"
    );
    assert!((r.derivative().unwrap().value().re + 0.8381261561246052).abs() < 5e-10);
    for parity in [Parity::Even, Parity::Odd] {
        for order in [1, 2, 3, 6] {
            let p = MathieuPlan::new(
                order,
                parity,
                2.0_f64,
                scioxide_special::SpectralNormalization::Standard,
                SpectralOptions::default(),
            )
            .unwrap();
            for x in [-0.5, 0.0, 0.5, 2.0] {
                let a = p.radial(x, RadialKind::First).unwrap();
                let b = p.radial(x, RadialKind::Second).unwrap();
                let av = a.try_value().unwrap();
                let bv = b.try_value().unwrap();
                let wronskian =
                    av * *b.derivative().unwrap().value() - *a.derivative().unwrap().value() * bv;
                assert!(
                    (wronskian.re - 2.0 / core::f64::consts::PI).abs() < 1e-8,
                    "n={order} {parity:?} x={x}: {wronskian:?}"
                );
                let third = p.radial(x, RadialKind::Third).unwrap().try_value().unwrap();
                assert!(
                    (third - av - Complex::new(0.0, 1.0) * bv).norm() < 1e-10 * (1.0 + bv.norm())
                );
            }
        }
    }
}

fn reference_width<S: scioxide_core::RealScalar>(width: &str)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_special::RadialKind;
    fn reference_calculation<S: scioxide_core::RealScalar>(
        p: &MathieuPlan<S>,
        mode: &str,
        z: Complex<S>,
    ) -> Result<(Complex<S>, Complex<S>), scioxide_core::SciOxideError>
    where
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        if mode == "char" {
            return Ok((
                Complex::new(p.characteristic_value()?, S::ZERO),
                Complex::new(S::ZERO, S::ZERO),
            ));
        }
        let r = if mode == "angular" {
            p.angular(z)?
        } else {
            p.radial(
                z,
                if mode == "r1" {
                    RadialKind::First
                } else {
                    RadialKind::Second
                },
            )?
        };
        let value = r.try_value()?;
        let derivative = r
            .derivative()
            .ok_or_else(|| {
                scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("fixture"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing derivative",
                )
            })?
            .value();
        Ok((value, *derivative))
    }
    let mut last = String::new();
    let mut plan = None;
    let mut failures = Vec::new();
    let mut count = 0;
    for line in include_str!("data/mathieu.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[1] != width {
            continue;
        }
        count += 1;
        let key = f[2..5].join(" ");
        let number = |i: usize| S::try_from_f64_lossy(f[i].parse::<f64>().unwrap()).unwrap();
        if key != last {
            plan = Some(
                MathieuPlan::new(
                    f[3].parse().unwrap(),
                    if f[2] == "ce" {
                        Parity::Even
                    } else {
                        Parity::Odd
                    },
                    number(4),
                    scioxide_special::SpectralNormalization::Standard,
                    SpectralOptions::default(),
                )
                .unwrap(),
            );
            last = key;
        }
        let p = plan.as_ref().unwrap();
        let expected = Complex::new(number(8), number(9));
        let expected_derivative = Complex::new(number(10), number(11));
        let z = Complex::new(number(5), number(6));
        let calculation = reference_calculation(p, f[7], z);
        match calculation {
            Err(error) => failures.push(format!("{line}: {error:?}")),
            Ok((value, derivative)) => {
                let (rel, abs) = if width == "64" {
                    (5e-9_f64, 5e-12_f64)
                } else {
                    (1e-3, 2e-5)
                };
                for (actual, expected) in [
                    (value.re, expected.re),
                    (value.im, expected.im),
                    (derivative.re, expected_derivative.re),
                    (derivative.im, expected_derivative.im),
                ] {
                    let error = (actual - expected).abs().try_to_f64_exact().unwrap();
                    let bound = abs.max(rel * expected.abs().try_to_f64_exact().unwrap());
                    if error > bound {
                        failures.push(format!("{line}: {value:?}, derivative {derivative:?}, error {error}, bound {bound}"));
                    }
                }
            }
        }
    }
    assert!(count > 100);
    assert!(
        failures.is_empty(),
        "{}\n{}",
        failures.len(),
        failures.join("\n")
    );
}
#[test]
fn independent_and_pinned_f64() {
    reference_width::<f64>("64");
}
#[test]
fn independent_and_pinned_f32() {
    reference_width::<f32>("32");
}
#[test]
fn tiny_imaginary_coordinate_retains_the_analytic_extension() {
    let p = MathieuPlan::new(
        2,
        Parity::Even,
        1.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let x = 0.5_f64;
    let h = 1e-100;
    let real = p.angular(x).unwrap();
    let complex = p.angular(Complex::new(x, h)).unwrap().try_value().unwrap();
    let derivative = *real.derivative().unwrap().value();
    assert!((complex.im / h - derivative).abs() < 1e-10);
}

#[test]
fn same_parity_modes_are_orthogonal() {
    let a = MathieuPlan::new(
        0,
        Parity::Even,
        1.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let b = MathieuPlan::new(
        2,
        Parity::Even,
        1.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let product = (0..512)
        .map(|k| {
            let x = (k as f64 + 0.5) * core::f64::consts::TAU / 512.0;
            a.angular(x).unwrap().try_value().unwrap() * b.angular(x).unwrap().try_value().unwrap()
        })
        .sum::<f64>()
        * core::f64::consts::TAU
        / 512.0;
    assert!(product.abs() < 1e-11);
}

fn recessive_width<S: scioxide_core::RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    for row in include_str!("data/mathieu_recessive.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        let scalar = |k: usize| S::try_from_f64_lossy(fields[k].parse().unwrap()).unwrap();
        let p = MathieuPlan::new(
            fields[1].parse().unwrap(),
            if fields[0] == "ce" {
                Parity::Even
            } else {
                Parity::Odd
            },
            scalar(2),
            scioxide_special::SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        for kind in [
            scioxide_special::RadialKind::Third,
            scioxide_special::RadialKind::Fourth,
        ] {
            let sign = if kind == scioxide_special::RadialKind::Third {
                S::ONE
            } else {
                -S::ONE
            };
            let r = p
                .radial(Complex::new(scalar(3), scalar(4) * sign), kind)
                .unwrap();
            let value = r.try_value().unwrap();
            let derivative = *r.derivative().unwrap().value();
            let reference = Complex::new(scalar(5), scalar(6) * sign);
            let reference_derivative = Complex::new(scalar(7), scalar(8) * sign);
            let bound = if S::EPSILON.try_to_f64_exact().unwrap() > 1e-10 {
                2e-6
            } else {
                2e-10
            };
            for (actual, expected) in [(value, reference), (derivative, reference_derivative)] {
                let error = ((actual - expected).norm() / expected.norm())
                    .try_to_f64_exact()
                    .unwrap();
                assert!(
                    error < bound,
                    "{row} {kind:?}: {actual:?} expected={expected:?} error={error}"
                );
            }
        }
    }
}
#[test]
fn recessive_outgoing_and_incoming_f64() {
    recessive_width::<f64>();
}
#[test]
fn recessive_outgoing_and_incoming_f32() {
    recessive_width::<f32>();
}

#[test]
fn selected_ground_mode_resolves_small_characteristics() {
    // a_0(q) = -q^2/2 + 7q^4/128 + O(q^6). Spectrum-wide
    // bisection tolerances would lose these values against the high modes.
    for q in [1e-6_f64, -1e-6, 1e-10, -1e-10] {
        let plan = MathieuPlan::new(
            0,
            Parity::Even,
            q,
            SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        let expected = -0.5 * q * q + 7.0 / 128.0 * q.powi(4);
        let actual = plan.characteristic_value().unwrap();
        assert!(
            (actual - expected).abs() <= 16.0 * f64::EPSILON * expected.abs(),
            "q={q}: {actual} != {expected}"
        );
    }
}

#[test]
fn fourier_recurrence_preserves_large_angles_and_zero_crossings() {
    for normalization in [
        SpectralNormalization::Standard,
        SpectralNormalization::UnitL2,
    ] {
        let scale = if normalization == SpectralNormalization::Standard {
            1.0
        } else {
            1.0 / core::f64::consts::PI.sqrt()
        };
        for parity in [Parity::Even, Parity::Odd] {
            for order in [1, 2, 31, 254] {
                let plan = MathieuPlan::new(
                    order,
                    parity,
                    0.0_f64,
                    normalization,
                    SpectralOptions::default(),
                )
                .unwrap();
                for x in [
                    0.0_f64,
                    -0.0,
                    0.5 * core::f64::consts::PI,
                    core::f64::consts::PI / f64::from(order),
                    1e100,
                    f64::MAX,
                    -f64::MAX,
                ] {
                    // Integer powering is an independent q=0 identity; forming
                    // order*x would overflow or discard reduction bits.
                    let (sin, cos) = x.sin_cos();
                    let phase = Complex::new(cos, sin).powu(order);
                    let (value, derivative) = if parity == Parity::Even {
                        (phase.re, -f64::from(order) * phase.im)
                    } else {
                        (phase.im, f64::from(order) * phase.re)
                    };
                    let report = plan.angular(x).unwrap();
                    assert!(report.termination().is_success());
                    for (actual, reference) in [
                        (report.value().unwrap(), value * scale),
                        (report.derivative().unwrap(), derivative * scale),
                    ] {
                        assert!(
                            (actual.value() - reference).abs() <= actual.absolute_error(),
                            "{normalization:?} {parity:?} {order} {x}: {actual:?} != {reference}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fourier_error_estimates_cover_independent_angular_fixtures() {
    for row in include_str!("data/mathieu.tsv")
        .lines()
        .filter(|line| line.starts_with("mpmath 64"))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[7] != "angular" {
            continue;
        }
        let number = |i: usize| f[i].parse::<f64>().unwrap();
        let plan = MathieuPlan::new(
            f[3].parse().unwrap(),
            if f[2] == "ce" {
                Parity::Even
            } else {
                Parity::Odd
            },
            number(4),
            SpectralNormalization::Standard,
            SpectralOptions::default(),
        )
        .unwrap();
        let report = plan.angular(Complex::new(number(5), number(6))).unwrap();
        assert!(report.termination().is_success());
        for (actual, reference) in [
            (report.value().unwrap(), Complex::new(number(8), number(9))),
            (
                report.derivative().unwrap(),
                Complex::new(number(10), number(11)),
            ),
        ] {
            assert!(
                (*actual.value() - reference).norm()
                    <= actual.absolute_error() + f64::EPSILON * reference.norm(),
                "{row}: {actual:?} != {reference:?}"
            );
        }
    }
}

fn bits<V: scioxide_core::NumericScalar>(value: V) -> (u64, u64) {
    use scioxide_core::RealScalar;
    (
        value.real_component().into_f64().to_bits(),
        value.imaginary_component().into_f64().to_bits(),
    )
}

/// Each one-shot returns the bits of the plan with the standard normalization and default
/// options, for both parities and every radial kind.
fn one_shots_match_standard_plan<T: scioxide_core::NumericScalar>(z: T)
where
    Complex<T::Real>: scioxide_core::NumericScalar<Real = T::Real>,
{
    use scioxide_core::RealScalar;
    use scioxide_special::{
        RadialKind, mathieu_a, mathieu_b, mathieu_ce, mathieu_mc, mathieu_ms, mathieu_se,
    };
    for parameter in [1.5, -2.0, 0.0] {
        let q = T::Real::try_from_f64_lossy(parameter).unwrap();
        for (order, parity) in [
            (0, Parity::Even),
            (1, Parity::Even),
            (4, Parity::Even),
            (1, Parity::Odd),
            (2, Parity::Odd),
            (5, Parity::Odd),
        ] {
            let plan = MathieuPlan::new(
                order,
                parity,
                q,
                SpectralNormalization::Standard,
                Default::default(),
            )
            .unwrap();
            let (characteristic, angular) = match parity {
                Parity::Even => (mathieu_a(order, q), mathieu_ce(order, q, z)),
                Parity::Odd => (mathieu_b(order, q), mathieu_se(order, q, z)),
            };
            let case = format!("order={order} {parity:?} q={q:?} z={z:?}");
            assert_eq!(
                bits(characteristic.unwrap()),
                bits(plan.characteristic_value().unwrap()),
                "{case}"
            );
            assert_eq!(
                bits(angular.unwrap()),
                bits(plan.angular(z).unwrap().try_value().unwrap()),
                "{case}"
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
            for &kind in kinds {
                let radial = match parity {
                    Parity::Even => mathieu_mc(order, q, z, kind),
                    Parity::Odd => mathieu_ms(order, q, z, kind),
                };
                assert_eq!(
                    bits(radial.unwrap()),
                    bits(plan.radial(z, kind).unwrap().try_value().unwrap()),
                    "{case} {kind:?}"
                );
            }
        }
    }
}

#[test]
fn one_shots_equal_the_standard_plan_bit_for_bit() {
    one_shots_match_standard_plan(0.3_f32);
    one_shots_match_standard_plan(0.3_f64);
    one_shots_match_standard_plan(Complex::new(0.3_f64, 0.2));
}

#[test]
fn one_shots_reduce_to_trigonometric_functions_at_zero_parameter() {
    use scioxide_special::{
        RadialKind, mathieu_a, mathieu_b, mathieu_ce, mathieu_mc, mathieu_ms, mathieu_se,
    };
    for n in 0..6_u32 {
        let square = f64::from(n * n);
        assert!((mathieu_a(n, 0.0_f64).unwrap() - square).abs() < 1e-12);
        for z in [-1.25_f64, 0.0, 0.5, 2.0] {
            let nz = f64::from(n) * z;
            let ce = mathieu_ce(n, 0.0, z).unwrap();
            let expected = if n == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                nz.cos()
            };
            assert!((ce - expected).abs() < 1e-12, "ce_{n}({z})");
            let mc = mathieu_mc(n, 0.0, z, RadialKind::First).unwrap();
            let expected = if n == 0 { 1.0 } else { 0.0 };
            assert!(
                (mc - Complex::new(expected, 0.0)).norm() < 1e-12,
                "Mc_{n}({z})"
            );
            if n > 0 {
                assert!((mathieu_se(n, 0.0, z).unwrap() - nz.sin()).abs() < 1e-12);
                assert!(mathieu_ms(n, 0.0, z, RadialKind::First).unwrap().norm() < 1e-12);
            }
        }
        if n > 0 {
            assert!((mathieu_b(n, 0.0_f64).unwrap() - square).abs() < 1e-12);
        }
    }
    let z = Complex::new(0.5_f64, 0.25);
    assert!((mathieu_ce(2, 0.0, z).unwrap() - (z * 2.0).cos()).norm() < 1e-12);
    assert!((mathieu_se(2, 0.0, z).unwrap() - (z * 2.0).sin()).norm() < 1e-12);
}

#[test]
fn one_shots_report_parameter_and_coordinate_errors() {
    use scioxide_core::{DomainReason, InvalidInputReason, SciOxideErrorKind, SciOxideErrorView};
    use scioxide_special::{
        RadialKind, mathieu_a, mathieu_b, mathieu_ce, mathieu_mc, mathieu_ms, mathieu_se,
    };
    for error in [
        mathieu_b(0, 1.0_f64).unwrap_err(),
        mathieu_se(0, 1.0, 0.5_f64).unwrap_err(),
        mathieu_ms(0, 1.0, 0.5_f64, RadialKind::First).unwrap_err(),
    ] {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::InvalidInput {
                    argument,
                    reason: InvalidInputReason::OutOfRange,
                    ..
                } if argument == scioxide_core::argument!("order")
            ),
            "{error}"
        );
    }
    let nan = f64::NAN;
    for error in [
        mathieu_a(2, nan).unwrap_err(),
        mathieu_b(2, nan).unwrap_err(),
        mathieu_ce(2, nan, 0.5).unwrap_err(),
        mathieu_ce(2, 1.0, nan).unwrap_err(),
        mathieu_se(2, 1.0, Complex::new(0.5, nan)).unwrap_err(),
        mathieu_mc(2, nan, 0.5, RadialKind::First).unwrap_err(),
        mathieu_mc(2, 1.0, Complex::new(nan, 0.0), RadialKind::First).unwrap_err(),
        mathieu_ms(2, 1.0, f64::INFINITY, RadialKind::Second).unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite, "{error}");
    }
    for error in [
        mathieu_mc(0, 0.0, 0.5_f64, RadialKind::Second).unwrap_err(),
        mathieu_ms(1, 0.0, 0.5_f64, RadialKind::Third).unwrap_err(),
    ] {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::DomainError {
                    argument,
                    reason: DomainReason::Pole,
                    ..
                } if argument == scioxide_core::argument!("q")
            ),
            "{error}"
        );
    }
}
