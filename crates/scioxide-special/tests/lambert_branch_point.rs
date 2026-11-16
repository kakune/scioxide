#![allow(clippy::unwrap_used, clippy::panic)]

#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{
    Budget, Complex, ConvergenceReason, NumericScalar, RealScalar, SciOxideErrorView,
};
use scioxide_special::*;

fn oracle<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut real_cases = 0;
    let mut complex_cases = 0;
    for line in include_str!("data/lambert_branch_point.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0] != width {
            continue;
        }
        let number = |i: usize| {
            let bits = u64::from_str_radix(fields[i], 16).unwrap();
            let value = if width == "64" {
                f64::from_bits(bits)
            } else {
                f64::from(f32::from_bits(u32::try_from(bits).unwrap()))
            };
            S::try_from_f64_exact(value).unwrap()
        };
        let z = Complex::new(number(3), number(4));
        let expected = Complex::new(number(5), number(6));
        let options =
            LambertOptions::default().with_iteration_budget(Budget::try_from(64).unwrap());
        if matches!(fields[1], "w0" | "wm1") {
            real_cases += 1;
            let result = if fields[1] == "w0" {
                lambert_w0_with_options(z.re, options)
            } else {
                lambert_wm1_with_options(z.re, options)
            }
            .unwrap_or_else(|error| panic!("{line}: {error}"));
            assert!(result.termination().is_success(), "{line}: {result:?}");
            let actual = result.try_value().unwrap();
            assert!(expected.im == S::ZERO, "non-real oracle: {line}");
            let x = z.re.try_to_f64_exact().unwrap();
            if (-1.0 / core::f64::consts::E..=-0.3).contains(&x) {
                let bits = |value: S| {
                    let value = value.try_to_f64_exact().unwrap();
                    if width == "64" {
                        value.to_bits()
                    } else {
                        u64::from((value as f32).to_bits())
                    }
                };
                assert!(
                    actual.is_sign_negative() && bits(actual).abs_diff(bits(expected.re)) <= 4,
                    "{line}: actual={actual:?}"
                );
            } else {
                let tolerance = if width == "64" { 2e-12 } else { 2e-5 };
                assert!(
                    (actual - expected.re).abs().try_to_f64_exact().unwrap()
                        <= tolerance * expected.re.abs().try_to_f64_exact().unwrap(),
                    "{line}: actual={actual:?}"
                );
            }
        } else {
            complex_cases += 1;
            let options = options.with_iteration_budget(Budget::try_from(32).unwrap());
            let result = if fields[1] == "omega" {
                wright_omega_with_options(z, options)
            } else {
                lambert_w_complex_with_options(z, fields[2].parse().unwrap(), options)
            }
            .unwrap_or_else(|error| panic!("{line}: {error}"));
            assert!(result.termination().is_success(), "{line}: {result:?}");
            let actual = result.try_value().unwrap();
            let (relative, absolute): (f64, f64) = if width == "64" {
                (2e-12, 2e-14)
            } else {
                (2e-5, 2e-6)
            };
            let error = result.value().unwrap().absolute_error().into_f64();
            let difference = (actual.re.into_f64() - expected.re.into_f64())
                .hypot(actual.im.into_f64() - expected.im.into_f64());
            let half_spacing = |x: S| (x.abs() - x.abs().next_after(S::ZERO)).into_f64() * 0.5;
            let rounding = half_spacing(expected.re).hypot(half_spacing(expected.im));
            assert!(
                difference <= error + rounding,
                "{line}: difference={difference:e}, reported={error:e}, rounding={rounding:e}"
            );
            for (actual, expected) in [(actual.re, expected.re), (actual.im, expected.im)] {
                assert!(
                    (actual - expected).abs().try_to_f64_exact().unwrap()
                        <= absolute.max(relative * expected.abs().try_to_f64_exact().unwrap()),
                    "{line}: actual={actual:?}"
                );
            }
        }
    }
    assert!(real_cases >= if width == "64" { 20000 } else { 2000 });
    assert!(complex_cases >= 8000);
}

#[test]
fn mpmath_f64_branch_point_accuracy() {
    oracle::<f64>("64");
}

#[test]
fn mpmath_f32_branch_point_accuracy() {
    oracle::<f32>("32");
}

#[test]
fn reported_failures_converge_with_a_small_budget() {
    let options = LambertOptions::default().with_iteration_budget(Budget::try_from(32).unwrap());
    for (x, expected) in [
        (-0.36535822409447277, -0.8872619354075873),
        (-0.36713092845171563, -0.9375267040317816),
    ] {
        let result = lambert_w0_with_options(x, options).unwrap();
        assert!(result.termination().is_success());
        assert!((result.try_value().unwrap() - expected).abs() <= 4.0 * f64::EPSILON);
        assert!((lambert_w0(x).unwrap() - expected).abs() <= 4.0 * f64::EPSILON);
    }
    let x = -0.35846202554621825;
    let expected = -1.2453457637582213;
    let result = lambert_wm1_with_options(x, options).unwrap();
    assert!(result.termination().is_success());
    assert!((result.try_value().unwrap() - expected).abs() <= 4.0 * f64::EPSILON);
    assert!((lambert_wm1(x).unwrap() - expected).abs() <= 4.0 * f64::EPSILON);
}

#[test]
fn omega_unwinding_residual_roundoff() {
    let result = wright_omega_with_options(
        Complex::new(0.9, 0.2),
        LambertOptions::default().with_iteration_budget(Budget::try_from(32).unwrap()),
    );
    assert!(result.is_ok(), "{result:?}");
    let result = result.unwrap();
    assert!(result.termination().is_success(), "{result:?}");
}

fn omega_pi_multiples<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = LambertOptions::default().with_iteration_budget(Budget::try_from(64).unwrap());
    let pi = S::PI;
    for m in (-1000..=1000_i32).filter(|&m| m != 0) {
        let center = S::try_from_f64_exact(f64::from(m)).unwrap() * pi;
        for im in [
            center.next_after(S::NEG_INFINITY),
            center,
            center.next_after(S::INFINITY),
        ] {
            for re in [-5.0_f64, -2.0, -0.5, 0.0, 2.0] {
                let z = Complex::new(S::try_from_f64_exact(re).unwrap(), im);
                let report = wright_omega_with_options(z, options)
                    .unwrap_or_else(|e| panic!("m={m} z={z:?}: {e}"));
                assert!(
                    report.termination().is_success(),
                    "m={m} z={z:?}: {report:?}"
                );
                let val = report.value().unwrap();
                let w = *val.value();
                // Evaluate the residual in binary64 so binary32 log rounding does
                // not hide a wrong-sheet solution or dominate its bound.
                let wide = |v: Complex<S>| Complex::new(v.re.into_f64(), v.im.into_f64());
                let w = wide(w);
                let residual = (w + w.ln() - wide(z)).norm();
                let bound = val.absolute_error().into_f64();
                assert!(
                    residual <= bound,
                    "m={m} z={z:?} w={w:?}: residual={residual:e} > bound={bound:e}"
                );
            }
        }
    }
}

#[test]
fn omega_pi_multiples_and_neighbours_f64() {
    omega_pi_multiples::<f64>();
}

#[test]
fn omega_pi_multiples_and_neighbours_f32() {
    omega_pi_multiples::<f32>();
}

fn endpoints<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let endpoint = S::try_from_f64_lossy(-1.0 / core::f64::consts::E).unwrap();
    for branch in [lambert_w0::<S>, lambert_wm1::<S>] {
        assert_eq!(branch(endpoint).unwrap(), -S::ONE);
        assert!(matches!(
            branch(endpoint.next_after(-S::ONE)).unwrap_err().view(),
            SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::OutsideDomain,
                ..
            }
        ));
        assert!(branch(endpoint.next_after(S::ZERO)).is_ok());
    }
}

#[test]
fn rounded_endpoint_and_both_neighbours() {
    endpoints::<f64>();
    endpoints::<f32>();
}

#[test]
fn exhaustion_retains_finite_diagnostics_and_never_reports_overflow() {
    for limit in [1, 9, 10] {
        let options =
            LambertOptions::default().with_iteration_budget(Budget::try_from(limit).unwrap());
        for result in [
            lambert_w0_with_options(-0.3_f64, options),
            lambert_wm1_with_options(-0.3_f64, options),
        ] {
            if limit < 10 {
                assert_eq!(
                    publication::exhausted_counts(&result.unwrap_err()).iterations(),
                    limit
                );
                continue;
            }
            let result = result.unwrap();
            assert_eq!(
                result.termination(),
                SpecialFunctionTermination::IterationLimit
            );
            assert_eq!(result.counts().iterations(), limit);
            assert_eq!(result.value().is_some(), limit == 10);
            if let Some(value) = result.value() {
                assert!(value.value().is_finite());
                assert!(value.absolute_error().is_finite() && value.absolute_error() < 1.0);
            }
            match result.try_value().unwrap_err().view() {
                SciOxideErrorView::ConvergenceFailure {
                    reason,
                    diagnostics,
                    ..
                } => {
                    assert_eq!(reason, ConvergenceReason::IterationLimit);
                    assert_eq!(diagnostics.counts().own().iterations(), limit);
                    assert!(diagnostics.algorithm().is_none());
                    assert!(!result.algorithm().method().is_empty());
                }
                error => panic!("wrong exhaustion classification: {error:?}"),
            }
        }
    }
    let options = LambertOptions::default().with_iteration_budget(Budget::ONE);
    for (result, has_value) in [
        (
            lambert_w_complex_with_options(Complex::new(3.0, 1.0), 0, options),
            true,
        ),
        (
            wright_omega_with_options(Complex::new(3.0, 1.0), options),
            false,
        ),
        (
            wright_omega_with_options(Complex::new(-1.001, core::f64::consts::PI), options),
            false,
        ),
    ] {
        if !has_value {
            assert_eq!(
                publication::exhausted_counts(&result.unwrap_err()).iterations(),
                1
            );
            continue;
        }
        let result = result.unwrap();
        assert_eq!(
            result.termination(),
            SpecialFunctionTermination::IterationLimit
        );
        assert!(result.value().is_some());
        assert!(matches!(
            result.try_value().unwrap_err().view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                ..
            }
        ));
    }
}

#[test]
fn finite_values_do_not_require_representable_derivatives() {
    let x = -f32::from_bits(1);
    let result = lambert_wm1_with_options(x, LambertOptions::default()).unwrap();
    assert!(result.termination().is_success());
    assert!(result.derivative().is_none());
    assert!(result.try_value().unwrap().is_finite());
}

#[test]
fn branch_point_real_lips_keep_signed_zero() {
    let x = -0.365;
    for lip in [0.0_f64, -0.0] {
        let w = lambert_w_complex(Complex::new(x, lip), 0).unwrap();
        assert_eq!(w.im, 0.0);
        assert_eq!(w.im.is_sign_negative(), lip.is_sign_negative());
        let branch = if lip.is_sign_negative() { 1 } else { -1 };
        let w = lambert_w_complex(Complex::new(x, lip), branch).unwrap();
        assert_eq!(w.im, 0.0);
        assert_eq!(w.im.is_sign_negative(), !lip.is_sign_negative());
    }
}

#[test]
fn branch_point_work_obeys_scoped_limits_and_cancellation() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(1038),
        scioxide_core::algorithm_id!("test", "lambert_branch_point", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    // The parent limit is scope-local; the options budget bounds the child epoch.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let result = lambert_w0_scoped(
        -0.3,
        LambertOptions::default().with_iteration_budget(Budget::try_from(10).unwrap()),
        &mut context,
    )
    .unwrap();
    assert_eq!(
        result.termination(),
        SpecialFunctionTermination::IterationLimit
    );
    assert_eq!(result.counts().iterations(), 10);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 10);
    assert!(result.value().is_some());

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    assert!(matches!(
        lambert_w0_scoped(-0.365, LambertOptions::default(), &mut context)
            .unwrap_err()
            .view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert!(matches!(
        wright_omega_scoped(
            Complex::new(-1.001, core::f64::consts::PI),
            LambertOptions::default(),
            &mut context,
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(context.counts().total().unwrap().iterations(), 0);
}

#[test]
fn branch_point_paths_have_derivatives_and_do_not_allocate() {
    for z in [
        Complex::new(-0.36535822409447277, 0.0),
        Complex::new(-0.367, 0.001),
        Complex::new(-0.8321771468338024, -0.48448649200437593),
        Complex::new(-0.6010574767537059, -0.27357633062386477),
        Complex::new(-0.4376499173904599, -0.06434168299230306),
        Complex::new(-1.257635599398756, 0.059970603383333376),
    ] {
        for branch in [0, -1, 1] {
            let result =
                lambert_w_complex_with_options(z, branch, LambertOptions::default()).unwrap();
            assert!(result.termination().is_success());
            let w = *result.value().unwrap().value();
            let derivative = *result.derivative().unwrap().value();
            assert!((derivative * z * (Complex::new(1.0, 0.0) + w) - w).norm() < 2e-12);
        }
    }
    for sign in [-1.0, 1.0] {
        let z = Complex::new(-1.001, sign * (core::f64::consts::PI + 0.001));
        let result = wright_omega_with_options(z, LambertOptions::default()).unwrap();
        assert!(result.termination().is_success());
        let w = *result.value().unwrap().value();
        let derivative = *result.derivative().unwrap().value();
        assert!((derivative * (Complex::new(1.0, 0.0) + w) - w).norm() < 2e-12);
    }
    assert_eq!(
        lambert_w0_with_options(-0.36535822409447277, LambertOptions::default())
            .unwrap()
            .termination(),
        SpecialFunctionTermination::Converged
    );
}
