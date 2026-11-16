//! Independent complex values/derivatives, continuation and work contracts.
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
    for line in include_str!("data/complex_error.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let number = |i: usize| f[i].parse::<f64>().unwrap();
        let z = Complex::new(
            S::try_from_f64_lossy(number(2)).unwrap(),
            S::try_from_f64_lossy(number(3)).unwrap(),
        );
        let r = match f[1] {
            "erf" => erf_with_options(z, ErrorFunctionOptions::default()),
            "erfc" => erfc_with_options(z, ErrorFunctionOptions::default()),
            "erfcx" => erfcx_with_options(z, ErrorFunctionOptions::default()),
            "erfi" => erfi_with_options(z, ErrorFunctionOptions::default()),
            "log_erfc" => log_erfc_with_options(z, ErrorFunctionOptions::default()),
            _ => dawson_with_options(z, ErrorFunctionOptions::default()),
        }
        .unwrap();
        assert!(r.termination().is_success(), "{line}: {r:?}");
        let actual = r.scaled().unwrap().value();
        let actual = scaled_reference::at_exponent(actual, f[6].parse().unwrap());
        let normalization = if f[1] == "erfi_scaled" {
            core::f64::consts::PI.sqrt() * 0.5
        } else {
            1.0
        };
        let reference = Complex::new(number(4), number(5)).scale(normalization);
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
            (actual - reference).norm() <= reported + 2.0 * precision * reference.norm(),
            "{line}: reported scaled error {reported:e}"
        );
        let tolerance = if width == "32" { 2e-6 } else { 2e-12 };
        assert!(
            (actual - reference).norm() < tolerance,
            "{line}: {actual:?} expected={reference:?}"
        );
        let exponent = f[9].parse::<i32>().unwrap();
        let project = |i: usize| {
            scioxide_core::checked_scalbn(number(i) * normalization, exponent)
                .ok()
                .and_then(|v| S::try_from_f64_lossy(v).ok())
        };
        let expected = project(7)
            .zip(project(8))
            .map(|(re, im)| Complex::new(re, im))
            .filter(|v| v.is_finite() && *v != Complex::new(S::ZERO, S::ZERO));
        if let Some(expected) = expected {
            let d = *r.derivative().unwrap().value();
            let d = Complex::new(
                d.re.try_to_f64_exact().unwrap(),
                d.im.try_to_f64_exact().unwrap(),
            );
            let expected = Complex::new(
                expected.re.try_to_f64_exact().unwrap(),
                expected.im.try_to_f64_exact().unwrap(),
            );
            assert!(
                (d - expected).norm() / expected.norm() < tolerance,
                "{line}: derivative={d:?} expected={expected:?}"
            );
        } else {
            assert!(
                r.derivative().is_none(),
                "{line}: unexpected derivative {r:?}"
            );
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
fn complements_rotations_and_near_axis_components() {
    for z in [
        Complex::new(0.25, 0.5),
        Complex::new(-2.0, 1.0),
        Complex::new(5.0, -2.0),
    ] {
        let erf_value = erf(z).unwrap();
        let erfc_value = erfc(z).unwrap();
        assert!(
            (erf_value + erfc_value - Complex::new(1.0, 0.0)).norm()
                < 1e-12 * (1.0 + erf_value.norm())
        );
        assert!(
            (erf(z.conj()).unwrap() - erf_value.conj()).norm() < 1e-12 * (1.0 + erf_value.norm())
        );
        assert!((erf(-z).unwrap() + erf_value).norm() < 1e-12 * (1.0 + erf_value.norm()));
        let rotated = Complex::new(0.0, -1.0) * erf(Complex::new(0.0, 1.0) * z).unwrap();
        assert!((erfi(z).unwrap() - rotated).norm() < 1e-12 * (1.0 + rotated.norm()));
    }
    for x in [-20.0, -0.5, 0.5, 20.0] {
        let h = 1e-100;
        let r = erfcx_with_options(Complex::new(x, 0.0), ErrorFunctionOptions::default()).unwrap();
        let z = erfcx(Complex::new(x, h)).unwrap();
        let d = r.derivative().unwrap().value().re;
        assert!(
            (z.im / h - d).abs() < 1e-10 * (1.0 + d.abs()),
            "x={x}: {z:?}, derivative={d}"
        );
    }
}

#[test]
fn logarithmic_value_survives_unavailable_binary_scale_and_derivative() {
    let r = erfcx_with_options(
        Complex::new(-1e154_f64, 0.0),
        ErrorFunctionOptions::default(),
    )
    .unwrap();
    assert!(r.value().is_none());
    assert!(r.scaled().is_none());
    assert!(r.derivative().is_none());
    let log = r.logarithmic().unwrap().value();
    assert_eq!(log.scaled_phase().round().unwrap(), Complex::new(1.0, 0.0));
    assert!((log.log_magnitude().unwrap() / 1e308 - 1.0).abs() < 1e-14);
    let r = log_erfc_with_options(
        Complex::new(1e154_f64, 0.0),
        ErrorFunctionOptions::default(),
    )
    .unwrap();
    assert!((r.try_value().unwrap().re / (-1e308) - 1.0).abs() < 1e-14);
    assert!((r.derivative().unwrap().value().re / (-2e154) - 1.0).abs() < 1e-14);
    assert!(
        (log_erfc(Complex::new(1e-100_f64, 0.0)).unwrap().re
            / (-core::f64::consts::FRAC_2_SQRT_PI * 1e-100)
            - 1.0)
            .abs()
            < 1e-14
    );
}

#[test]
fn real_forward_migration_signed_zeros_and_work_contracts() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(502),
        scioxide_core::algorithm_id!("complex-error", "test", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    // The parent limit is scope-local; the options budget bounds the child epoch.
    let r = erf_scoped(
        Complex::new(0.25, 0.5),
        ErrorFunctionOptions::default().with_iteration_budget(Budget::ONE),
        &mut context,
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&r);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
    assert!(
        erf_scoped(
            Complex::new(0.25, 0.5),
            ErrorFunctionOptions::default(),
            &mut context,
        )
        .unwrap()
        .termination()
        .is_success()
    );
    assert_eq!(
        erf_with_options(0.75_f64, ErrorFunctionOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
    for sign in [1.0_f64, -1.0] {
        let z = Complex::new(0.0_f64.copysign(sign), -0.0);
        for value in [erf(z).unwrap(), erfi(z).unwrap(), dawson(z).unwrap()] {
            assert_eq!(value.re.to_bits(), z.re.to_bits());
            assert_eq!(value.im.to_bits(), z.im.to_bits());
        }
        assert_eq!(erfc(z).unwrap().im.to_bits(), 0.0_f64.to_bits());
    }
    assert_eq!(erf(f64::MAX).unwrap(), 1.0);
    assert_eq!(erfc(f64::MAX).unwrap(), 0.0);
    assert!(erfcx(f64::MAX).unwrap() > 0.0);
    assert!(dawson(f64::MAX).unwrap() > 0.0);
    assert_eq!(
        erf_with_options(0.75_f64, ErrorFunctionOptions::default())
            .unwrap()
            .counts()
            .iterations(),
        1
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let actual = dawson_scoped(
        Complex::new(20.0_f64, 0.0),
        ErrorFunctionOptions::default(),
        &mut context,
    )
    .unwrap();
    assert_eq!(
        actual,
        dawson_with_options(Complex::new(20.0_f64, 0.0), ErrorFunctionOptions::default()).unwrap()
    );
    assert_eq!(
        actual.counts().iterations(),
        context.counts().nested().iterations()
    );
}

#[test]
fn reserved_complex_values_and_failure_paths_execute() {
    use scioxide_core::{
        Budget, CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let z = Complex::new(0.25_f64, 0.5);
    erf(z).unwrap();
    erfcx(z).unwrap();
    dawson(z).unwrap();
    let options = ErrorFunctionOptions::default().with_iteration_budget(Budget::ONE);
    let r = erfc_with_options(z, options).unwrap_err();
    let _ = publication::exhausted_counts(&r);
    let token = CancellationToken::new();
    token.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(502),
        scioxide_core::algorithm_id!("complex-error", "test", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity,
            &mut budgets,
        )
        .unwrap();
    assert!(erf_scoped(z, ErrorFunctionOptions::default(), &mut context).is_err());
    assert_eq!(context.counts().total().unwrap().iterations(), 0);
    assert!(erf(Complex::new(f64::NAN, 0.0)).is_err());
}

#[test]
fn real_erfi_uses_fixed_cost_dawson_in_both_precisions() {
    fn check<S: scioxide_core::RealScalar>() {
        let options = ErrorFunctionOptions::new(scioxide_core::Budget::ONE);
        let factor = S::try_from_f64_lossy(core::f64::consts::FRAC_2_SQRT_PI).unwrap();
        for input in [
            -32.0, -16.0, -1.0, -0.75, -0.0, 0.0, 0.5, 0.75, 1.0, 16.0, 32.0,
        ] {
            let x = S::try_from_f64_lossy(input).unwrap();
            let expected = dawson(x).unwrap();
            let report = dawson_with_options(x, options).unwrap();
            let actual = report.try_value().unwrap();
            assert!((actual - expected).abs() <= S::EPSILON * expected.abs() * S::TWO);
            assert_eq!(report.counts().iterations(), 1);
            assert_eq!(actual, dawson(x).unwrap());
            let unscaled = erfi_with_options(x, options).unwrap();
            assert_eq!(unscaled.counts().iterations(), 1);
            if input.abs() <= 1.0 {
                let expected = factor * actual * (x * x).exp();
                assert!(
                    (unscaled.try_value().unwrap() - expected).abs()
                        <= S::EPSILON * expected.abs() * S::try_from_f64_lossy(4.0).unwrap()
                );
            }
        }
        for x in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
            assert_eq!(
                erfi(x).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::NonFinite
            );
            assert_eq!(
                dawson(x).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::NonFinite
            );
        }
    }
    check::<f32>();
    check::<f64>();
    for x in [
        -0.0_f64,
        0.0,
        f64::from_bits(1),
        0.75,
        1.0_f64.next_up(),
        16.0_f64.next_up(),
        f64::MAX,
    ] {
        assert_eq!(
            dawson(x).unwrap().to_bits(),
            dawson_with_options(x, ErrorFunctionOptions::default())
                .unwrap()
                .try_value()
                .unwrap()
                .to_bits()
        );
    }
    let overflow = erfi_with_options(30.0_f64, ErrorFunctionOptions::default()).unwrap();
    assert_eq!(
        overflow.scalar_representation(),
        scioxide_special::ScalarRepresentation::Overflow
    );
    assert!(overflow.scaled().is_some());
    assert!(overflow.logarithmic().is_some());
    assert!(overflow.try_value().is_err());
}
