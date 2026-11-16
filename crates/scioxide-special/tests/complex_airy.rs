//! Complex Airy continuation, derivatives and explicit scaling.
#![allow(clippy::unwrap_used)]
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
    for row in include_str!("data/airy_complex.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let n = |i: usize| f[i].parse::<f64>().unwrap();
        let z = Complex::new(
            S::try_from_f64_lossy(n(2)).unwrap(),
            S::try_from_f64_lossy(n(3)).unwrap(),
        );
        let r = match f[1] {
            "ai" => airy_ai_with_options(z, AiryOptions::default()),
            "bi" => airy_bi_with_options(z, AiryOptions::default()),
            "ai_scaled" => airy_ai_scaled_with_options(z, AiryOptions::default()),
            _ => airy_bi_scaled_with_options(z, AiryOptions::default()),
        }
        .unwrap();
        assert!(r.termination().is_success(), "{row}: {r:?}");
        let s = r.scaled().unwrap().value();
        let actual = scaled_reference::at_exponent(s, f[6].parse().unwrap());
        let expected = Complex::new(n(4), n(5));
        // The reported scaled error covers the reference, allowing its rounding.
        let scaled = r.scaled().unwrap();
        let reported = scaled_reference::error_at_exponent(
            scaled.value(),
            scaled.absolute_error().into_f64(),
            f[6].parse().unwrap(),
        );
        let precision = if width == "32" {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        assert!(
            (actual - expected).norm() <= reported + 2.0 * precision * expected.norm(),
            "{row}: reported scaled error {reported:e}"
        );
        let tolerance = if width == "32" { 3e-6 } else { 2e-11 };
        assert!(
            (actual - expected).norm() < tolerance * expected.norm(),
            "{row}: {actual:?} expected={expected:?}"
        );
        let exponent = f[9].parse::<i32>().unwrap();
        let part = |i| {
            scioxide_core::checked_scalbn(n(i), exponent)
                .ok()
                .and_then(|v| S::try_from_f64_lossy(v).ok())
        };
        let expected = part(7)
            .zip(part(8))
            .map(|(re, im)| Complex::new(re, im))
            .filter(|z| *z != Complex::new(S::ZERO, S::ZERO));
        if let Some(expected) = expected {
            let actual = *r.derivative().unwrap().value();
            assert!(
                ((actual - expected).norm() / expected.norm())
                    .try_to_f64_exact()
                    .unwrap()
                    < tolerance,
                "{row}: derivative={actual:?} expected={expected:?}"
            );
        } else {
            assert!(r.derivative().is_none(), "{row}: {r:?}");
        }
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
fn large_scaled_real_seed() {
    let r = airy_bi_scaled_with_options(1e300_f64, AiryOptions::default()).unwrap();
    assert!(r.termination().is_success(), "{r:?}");
}

#[test]
fn identities_cut_lips_and_small_imaginary_components() {
    let omega = Complex::new(-0.5, 3.0_f64.sqrt() / 2.0);
    for z in [
        Complex::new(0.5, 0.25),
        Complex::new(-3.0, 1.0),
        Complex::new(5.0, -0.5),
    ] {
        let ai = airy_ai_with_options(z, AiryOptions::default()).unwrap();
        let bi = airy_bi_with_options(z, AiryOptions::default()).unwrap();
        let a = ai.try_value().unwrap();
        let b = bi.try_value().unwrap();
        let wronskian =
            a * *bi.derivative().unwrap().value() - *ai.derivative().unwrap().value() * b;
        assert!((wronskian - Complex::new(1.0 / core::f64::consts::PI, 0.0)).norm() < 2e-11);
        assert!((airy_ai(z.conj()).unwrap() - a.conj()).norm() < 2e-12 * (1.0 + a.norm()));
        assert!((airy_bi(z.conj()).unwrap() - b.conj()).norm() < 2e-12 * (1.0 + b.norm()));
        let connection = a
            + omega * airy_ai(omega * z).unwrap()
            + omega.conj() * airy_ai(omega.conj() * z).unwrap();
        assert!(connection.norm() < 2e-11 * (1.0 + a.norm()));
    }
    for x in [-1e100, -20.0, -5.0, 5.0, 20.0] {
        let h = 1e-100;
        let real = airy_ai_with_options(x, AiryOptions::default()).unwrap();
        let value = airy_ai(Complex::new(x, h)).unwrap();
        let derivative = *real.derivative().unwrap().value();
        assert!(
            (value.im / h / derivative - 1.0).abs() < 1e-10,
            "x={x} {value:?}"
        );
    }
    for x in [-5.0, -20.0, -1e100] {
        let upper = airy_ai_scaled(Complex::new(x, 0.0)).unwrap();
        let lower = airy_ai_scaled(Complex::new(x, -0.0)).unwrap();
        assert!((upper.conj() - lower).norm() < 1e-13 * upper.norm());
        assert_eq!(
            airy_ai(Complex::new(x, 0.0)).unwrap().re,
            airy_ai(x).unwrap()
        );
    }
    let x = 1e100_f64;
    let h = 1e-52;
    let value = airy_ai_scaled(Complex::new(x, h)).unwrap();
    assert!((value.im / (-0.25 * value.re * h / x) - 1.0).abs() < 1e-12);
}

#[test]
fn work_limits_cancellation_and_reused_storage() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(502),
        scioxide_core::algorithm_id!("complex-Airy", "test", 1),
        0,
        0,
    );
    for z in [
        Complex::new(0.5, 0.25),
        Complex::new(5.0, 0.5),
        Complex::new(-60.0, 0.5),
        Complex::new(-1e100, 1e-52),
    ] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let r = airy_ai_scoped(z, AiryOptions::default(), &mut context).unwrap();
        assert!(r.termination().is_success());
        assert_eq!(r, airy_ai_with_options(z, AiryOptions::default()).unwrap());
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(
            r.counts().iterations(),
            context.counts().nested().iterations()
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        // The parent limit is scope-local; the options budget bounds the child epoch.
        let r = airy_ai_scoped(
            z,
            AiryOptions::default().with_iteration_budget(Budget::ONE),
            &mut context,
        )
        .unwrap_err();
        let _ = publication::exhausted_counts(&r);
        assert_eq!(context.counts().nested().iterations(), 1);
    }
    assert_eq!(
        airy_ai_with_options(Complex::new(0.5, 0.25), AiryOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    assert!(airy_ai(Complex::new(f64::INFINITY, 0.0)).is_err());
    assert!(airy_bi(Complex::new(0.0, f64::NAN)).is_err());
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
    let mut context = state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap();
    assert!(matches!(
        airy_ai_scoped(Complex::new(5.0, 0.5), AiryOptions::default(), &mut context)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
}
