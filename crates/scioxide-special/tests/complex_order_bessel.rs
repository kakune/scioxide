//! Real/nonreal orders, independent branch references and analytic argument derivatives.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
#[path = "support/scaled.rs"]
mod scaled_reference;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn references<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/complex_order_bessel.tsv")
        .lines()
        .chain(include_str!("data/complex_order_bessel_real.tsv").lines())
        .chain(include_str!("data/complex_order_bessel_bands.tsv").lines())
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let n = |i: usize| f[i].parse::<f64>().unwrap();
        let c = |i| {
            Complex::new(
                S::try_from_f64_lossy(n(i)).unwrap(),
                S::try_from_f64_lossy(n(i + 1)).unwrap(),
            )
        };
        let nu = c(2);
        let z = c(4);
        let options = BesselOptions::default();
        let r = match f[1] {
            "j" => bessel_j_with_options(nu, z, options),
            "y" => bessel_y_with_options(nu, z, options),
            "i" => bessel_i_with_options(nu, z, options),
            "k" => bessel_k_with_options(nu, z, options),
            "h1" => hankel_1_with_options(nu, z, options),
            "h2" => hankel_2_with_options(nu, z, options),
            "j_scaled" => bessel_j_scaled_with_options(nu, z, options),
            "y_scaled" => bessel_y_scaled_with_options(nu, z, options),
            "i_scaled" => bessel_i_scaled_with_options(nu, z, options),
            "k_scaled" => bessel_k_scaled_with_options(nu, z, options),
            "h1_scaled" => hankel_1_scaled_with_options(nu, z, options),
            _ => hankel_2_scaled_with_options(nu, z, options),
        }
        .unwrap_or_else(|error| panic!("{row}: {error:?}"));
        assert!(r.termination().is_success(), "{row}: {r:?}");
        let s = r.scaled().unwrap().value();
        let actual = scaled_reference::at_exponent(s, f[8].parse().unwrap());
        let expected = Complex::new(n(6), n(7));
        // The reported scaled error covers the reference, allowing its rounding.
        let scaled = r.scaled().unwrap();
        let reported = scaled_reference::error_at_exponent(
            scaled.value(),
            scaled.absolute_error().into_f64(),
            f[8].parse().unwrap(),
        );
        let precision = if width == "32" {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        assert!(
            (actual - expected).norm() <= reported + 2.0 * precision * expected.norm(),
            "{row}: actual={actual:?}, expected={expected:?}, reported scaled error {reported:e}"
        );
        let tolerance = if width == "32" { 3e-6 } else { 2e-10 };
        assert!(
            (actual - expected).norm() < tolerance * expected.norm(),
            "{row}: {actual:?} expected={expected:?}"
        );
        let significand = Complex::new(n(9), n(10));
        let expected = significand * 2.0_f64.powi(f[11].parse().unwrap());
        // Outside the destination's finite and subnormal range no derivative
        // is published.
        let (largest, smallest) = if width == "32" {
            (f64::from(f32::MAX), f64::from(f32::from_bits(1)))
        } else {
            (f64::MAX, f64::from_bits(1))
        };
        if !expected.is_finite()
            || expected.re.abs().max(expected.im.abs()) > largest
            || (expected.norm() < smallest && significand != Complex::new(0.0, 0.0))
        {
            assert!(
                r.derivative().is_none(),
                "{row}: derivative outside scalar range"
            );
            continue;
        }
        let d = *r.derivative().unwrap().value();
        let actual = Complex::new(
            d.re.try_to_f64_exact().unwrap(),
            d.im.try_to_f64_exact().unwrap(),
        );
        let derivative_error = r.derivative().unwrap().absolute_error().into_f64();
        assert!(
            (actual - expected).norm() <= derivative_error + 2.0 * precision * expected.norm(),
            "{row}: derivative {actual:?} expected={expected:?}, reported error {derivative_error:e}"
        );
        // At derivative zeros compare against the rounding floor of the
        // analytic recurrence as well as the reference-relative threshold.
        let coordinate = Complex::new(n(4), n(5)).norm();
        let order = Complex::new(n(2), n(3)).norm();
        let value_norm = Complex::new(n(6), n(7)).norm() * 2.0_f64.powi(f[8].parse().unwrap());
        let rounding = 64.0 * f64::EPSILON * (coordinate + order / coordinate) * value_norm;
        assert!(
            (actual - expected).norm() < tolerance * expected.norm() + rounding,
            "{row}: derivative {actual:?} expected={expected:?}"
        );
    }
}
#[test]
fn independent_f64() {
    references::<f64>("64");
}
#[test]
fn independent_f32() {
    references::<f32>("32");
}

#[test]
fn complex_order_connections_and_derivatives() {
    let nu = Complex::new(0.5, 0.25);
    let z = Complex::new(2.0, 1.0);
    let j = bessel_j(nu, z).unwrap();
    let y = bessel_y(nu, z).unwrap();
    let i = Complex::new(0.0, 1.0);
    assert!((hankel_1(nu, z).unwrap() - j - i * y).norm() < 1e-10);
    assert!((hankel_2(nu, z).unwrap() - j + i * y).norm() < 1e-10);
    assert!((bessel_j(nu.conj(), z.conj()).unwrap() - j.conj()).norm() < 1e-12);
    assert!((bessel_k(-nu, z).unwrap() - bessel_k(nu, z).unwrap()).norm() < 1e-12);
    let lower = bessel_j(nu - Complex::new(1.0, 0.0), z).unwrap();
    let upper = bessel_j(nu + Complex::new(1.0, 0.0), z).unwrap();
    assert!((lower + upper - nu * j / z * 2.0).norm() < 1e-12);
    let report = bessel_j_with_options(nu, z, BesselOptions::default()).unwrap();
    assert!(
        (bessel_j_derivative(nu, z, 1).unwrap() - *report.derivative().unwrap().value()).norm()
            < 1e-12
    );
    assert!(
        (bessel_j_derivative(nu, z, 3).unwrap()
            - Complex::new(0.38457481914219895, 0.06393101458410719))
        .norm()
            < 2e-12
    );
    assert!(
        (bessel_k_derivative(nu, z, 3).unwrap()
            - Complex::new(-0.009642445084548694, 0.24591762790304176))
        .norm()
            < 2e-10
    );
}

#[test]
fn actual_work_scoped_limits_and_cancellation() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(502),
        scioxide_core::algorithm_id!("complex-order-Bessel", "test", 1),
        0,
        0,
    );
    let nu = Complex::new(0.5, 0.25);
    let z = Complex::new(2.0, 1.0);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let r = bessel_k_scoped(nu, z, BesselOptions::default(), &mut context).unwrap();
    assert!(r.termination().is_success());
    assert_eq!(
        r,
        bessel_k_with_options(nu, z, BesselOptions::default()).unwrap()
    );
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        r.counts().iterations(),
        context.counts().nested().iterations()
    );
    // A parent iteration limit is scope-local; the options budget bounds the call.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let unlimited = bessel_k_scoped(nu, z, BesselOptions::default(), &mut context).unwrap();
    assert!(unlimited.termination().is_success());
    let r = bessel_k_scoped(
        nu,
        z,
        BesselOptions::default().with_iteration_budget(Budget::ONE),
        &mut context,
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    let cancelled = scioxide_core::CancellationToken::new();
    cancelled.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancelled),
            identity,
            &mut budgets,
        )
        .unwrap();
    assert!(matches!(
        bessel_k_scoped(nu, z, BesselOptions::default(), &mut context)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
}

#[test]
fn origin_limits_and_direct_large_scaling() {
    let origin = Complex::new(0.0, 0.0);
    let r =
        bessel_j_with_options(Complex::new(0.5, 0.25), origin, BesselOptions::default()).unwrap();
    assert_eq!(r.try_value().unwrap(), origin);
    assert!(r.derivative().is_none());
    let r =
        bessel_i_with_options(Complex::new(1.5, 0.25), origin, BesselOptions::default()).unwrap();
    assert_eq!(*r.derivative().unwrap().value(), origin);
    assert!(bessel_j(Complex::new(0.0, 0.25), origin).is_err());
    assert_eq!(
        bessel_j_derivative(Complex::new(3.5, 0.25), origin, 3).unwrap(),
        origin
    );
    assert!(bessel_j_derivative(Complex::new(2.5, 0.25), origin, 3).is_err());
    let value = bessel_k_scaled(Complex::new(0.5, 0.25), Complex::new(1e100, 2.0)).unwrap();
    assert!(
        (value.im / value.re / (-0.875e-100) - 1.0).abs() < 1e-12,
        "{value:?}"
    );
    let nu = Complex::new(0.3, 0.25);
    let tiny = Complex::new(1e-250, 0.0);
    let k = bessel_k_with_options(nu, tiny, BesselOptions::default()).unwrap();
    assert!(k.try_value().unwrap().is_finite());
    assert!(k.derivative().is_none());
    let shifted =
        bessel_k_with_options(nu + Complex::new(2.0, 0.0), tiny, BesselOptions::default()).unwrap();
    assert!(shifted.termination().is_success());
    assert_eq!(
        shifted.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(shifted.scaled().is_some());
}

/// Order phases stay exact beside a large argument's phase: reflected scaled
/// Hankel waves satisfy DLMF 10.4.6 within their reported errors, and large
/// orders across the left half-plane inside the turning circle complete.
#[test]
fn large_argument_phases_and_turning_circle_reflections() {
    let options = BesselOptions::default();
    for (nu, z) in [
        (
            Complex::new(347.94442683042075, 0.0),
            Complex::new(2362344.0700518424, -4364147.953834023),
        ),
        (
            Complex::new(606.6524665638443, 1.2400554513650666),
            Complex::new(2885730.1011710404, -2435531.8985127755),
        ),
    ] {
        for (first, sign) in [(true, 1.0), (false, -1.0)] {
            let wave = |order| {
                if first {
                    hankel_1_scaled_with_options(order, z, options)
                } else {
                    hankel_2_scaled_with_options(order, z, options)
                }
                .unwrap()
            };
            let (positive, negative) = (wave(nu), wave(-nu));
            // exp(i sign pi nu), reduced exactly as the library does.
            let (sine, cosine) = scioxide_core::sin_cos_pi(sign * nu.re).unwrap();
            let factor = Complex::new(cosine, sine) * (-sign * std::f64::consts::PI * nu.im).exp();
            let expected = factor * positive.try_value().unwrap();
            let actual = negative.try_value().unwrap();
            let allowance = negative.value().unwrap().absolute_error()
                + factor.norm() * positive.value().unwrap().absolute_error();
            assert!(
                (actual - expected).norm() <= allowance
                    && (actual - expected).norm() < 1e-14 * expected.norm(),
                "{nu:?} {z:?}: {actual:?} {expected:?} {allowance:e}"
            );
        }
    }
    for (nu, z) in [
        (
            Complex::new(3547.0356855569316, 0.0),
            Complex::new(1363.8233344416083, -224.99690545412616),
        ),
        (
            Complex::new(3712.9949307830548, 0.0),
            Complex::new(-36.817093046397815, 2909.9754875681674),
        ),
    ] {
        for r in [
            bessel_y_with_options(nu, z, options),
            bessel_k_scaled_with_options(nu, z, options),
            hankel_1_scaled_with_options(nu, z, options),
        ] {
            let r = r.unwrap();
            assert!(r.termination().is_success(), "{nu:?} {z:?}: {r:?}");
            assert!(r.counts().iterations() < 1024);
        }
    }
}

#[test]
fn issue_1151_reproducers_complete_without_budget_increases() {
    use scioxide_core::Budget;
    for (kind, nu, z) in [
        (
            "y",
            Complex::new(0.6808015341978835_f64, 0.0),
            Complex::new(-0.027820334366123602, -0.037785140057554095),
        ),
        (
            "y",
            Complex::new(0.9648608751426924, 0.0),
            Complex::new(0.00341676610021134, -0.01564577103300104),
        ),
        (
            "j",
            Complex::new(-0.7680911945332545, 2.8333967024379394),
            Complex::new(3.9659514854688083, -25.469114458584826),
        ),
        (
            "j",
            Complex::new(0.7722156400262525, -2.5974158856735086),
            Complex::new(-14.597601035722716, -2.090269471992919),
        ),
    ] {
        let evaluate = |options| match kind {
            "j" => bessel_j_with_options(nu, z, options),
            _ => bessel_y_with_options(nu, z, options),
        };
        let report = evaluate(BesselOptions::default()).unwrap();
        assert!(
            report.termination().is_success(),
            "{kind} {nu:?} {z:?}: {report:?}"
        );
        assert!(report.counts().iterations() < if kind == "y" { 1024 } else { 4096 });
        assert_eq!(
            report,
            evaluate(
                BesselOptions::default()
                    .with_iteration_budget(Budget::try_from(16_777_216).unwrap())
            )
            .unwrap()
        );
        let plain = (if kind == "j" {
            bessel_j(nu, z)
        } else {
            bessel_y(nu, z)
        })
        .unwrap();
        let value = report.try_value().unwrap();
        assert_eq!(
            [plain.re.to_bits(), plain.im.to_bits()],
            [value.re.to_bits(), value.im.to_bits()]
        );
        let limited =
            evaluate(BesselOptions::default().with_iteration_budget(Budget::ONE)).unwrap_err();
        let limited_counts = publication::exhausted_counts(&limited);
        assert_eq!(limited_counts.iterations(), 1);
    }
}

#[test]
fn adjacent_solutions_and_scoped_forms_across_regimes() {
    use scioxide_core::{
        Budget, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    for (nu, z) in [
        (
            Complex::new(1.0_f64.next_up(), 0.0),
            Complex::new(0.5, -4.0),
        ),
        (Complex::new(2.3, 0.0), Complex::new(8.0, 4.0)),
        (Complex::new(-2.3, 0.0), Complex::new(8.0, 4.0)),
        (Complex::new(-2.0, 0.0), Complex::new(0.001, -0.002)),
        (
            Complex::new(3.0_f64.next_down(), 0.0),
            Complex::new(-0.001, -0.002),
        ),
        (Complex::new(4.75, 0.0), Complex::new(0.1, -0.02)),
        (Complex::new(1.0_f64, 1e-12), Complex::new(0.01, -0.02)),
        (Complex::new(-2.0, 1e-8), Complex::new(-0.05, -0.0)),
        (Complex::new(0.5, 3.0), Complex::new(12.0, -8.0)),
        (Complex::new(2.0, 0.5), Complex::new(70.0, 20.0)),
    ] {
        let j = bessel_j(nu, z).unwrap();
        let y = bessel_y(nu, z).unwrap();
        let i = Complex::new(0.0, 1.0);
        for name in ["j", "y", "i", "k", "h1", "h2"] {
            let detailed = |options| match name {
                "j" => bessel_j_with_options(nu, z, options),
                "y" => bessel_y_with_options(nu, z, options),
                "i" => bessel_i_with_options(nu, z, options),
                "k" => bessel_k_with_options(nu, z, options),
                "h1" => hankel_1_with_options(nu, z, options),
                _ => hankel_2_with_options(nu, z, options),
            };
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    SolveIdentity::new(
                        ProblemToken::new(1151),
                        scioxide_core::algorithm_id!("Bessel", "test", 1),
                        0,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            let report = match name {
                "j" => bessel_j_scoped(nu, z, BesselOptions::default(), &mut context),
                "y" => bessel_y_scoped(nu, z, BesselOptions::default(), &mut context),
                "i" => bessel_i_scoped(nu, z, BesselOptions::default(), &mut context),
                "k" => bessel_k_scoped(nu, z, BesselOptions::default(), &mut context),
                "h1" => hankel_1_scoped(nu, z, BesselOptions::default(), &mut context),
                _ => hankel_2_scoped(nu, z, BesselOptions::default(), &mut context),
            }
            .unwrap();
            assert!(report.termination().is_success());
            assert_eq!(report, detailed(BesselOptions::default()).unwrap());
            assert_eq!(
                report.counts().iterations(),
                context.counts().nested().iterations()
            );
            assert_eq!(context.counts().own().iterations(), 0);
            let plain = match name {
                "j" => bessel_j(nu, z),
                "y" => bessel_y(nu, z),
                "i" => bessel_i(nu, z),
                "k" => bessel_k(nu, z),
                "h1" => hankel_1(nu, z),
                _ => hankel_2(nu, z),
            }
            .unwrap();
            let value = report.try_value().unwrap();
            assert_eq!(
                [plain.re.to_bits(), plain.im.to_bits()],
                [value.re.to_bits(), value.im.to_bits()]
            );
            for limit in [1, 2, 8, 16] {
                let limited = detailed(
                    BesselOptions::default()
                        .with_iteration_budget(Budget::try_from(limit).unwrap()),
                );
                if limit < report.counts().iterations() {
                    assert!(
                        publication::exhausted_counts(&limited.unwrap_err()).iterations() <= limit
                    );
                } else {
                    assert!(limited.unwrap().termination().is_success());
                }
            }
            let cancelled = scioxide_core::CancellationToken::new();
            cancelled.cancel();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential().with_cancellation(&cancelled),
                    SolveIdentity::new(
                        ProblemToken::new(1303),
                        scioxide_core::algorithm_id!("Bessel", "test", 1),
                        0,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            let stopped = match name {
                "j" => bessel_j_scoped(nu, z, BesselOptions::default(), &mut context),
                "y" => bessel_y_scoped(nu, z, BesselOptions::default(), &mut context),
                "i" => bessel_i_scoped(nu, z, BesselOptions::default(), &mut context),
                "k" => bessel_k_scoped(nu, z, BesselOptions::default(), &mut context),
                "h1" => hankel_1_scoped(nu, z, BesselOptions::default(), &mut context),
                _ => hankel_2_scoped(nu, z, BesselOptions::default(), &mut context),
            };
            assert!(matches!(
                stopped.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
            ));
        }
        for (actual, expected) in [
            (hankel_1(nu, z).unwrap(), j + i * y),
            (hankel_2(nu, z).unwrap(), j - i * y),
        ] {
            assert!((actual - expected).norm() <= 2e-10 * (j.norm() + y.norm()));
        }
        let ip = bessel_i_with_options(nu, z, BesselOptions::default()).unwrap();
        let kp = bessel_k_with_options(nu, z, BesselOptions::default()).unwrap();
        let iv = ip.try_value().unwrap();
        let kv = kp.try_value().unwrap();
        let id = *ip.derivative().unwrap().value();
        let kd = *kp.derivative().unwrap().value();
        let residual = kv * id - kd * iv - Complex::new(1.0, 0.0) / z;
        assert!(
            residual.norm()
                <= 2e-10 * (kv.norm() * id.norm() + kd.norm() * iv.norm() + 1.0 / z.norm())
        );
        assert!((bessel_k(-nu, z).unwrap() - kv).norm() <= 2e-12 * kv.norm());
        let conjugate = bessel_j(nu.conj(), z.conj()).unwrap();
        assert!((conjugate - j.conj()).norm() <= 2e-10 * j.norm());
    }
}
