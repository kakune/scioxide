//! Accuracy envelopes, publication and work contracts for the real kernels.
#![allow(clippy::unwrap_used, clippy::panic)]

#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, Complex, NumericScalar, RealScalar, SciOxideError};
use scioxide_special::*;
use std::collections::BTreeMap;

const DATA: &str = include_str!("data/real_lambert_polylog.tsv");

fn pair<S: RealScalar>(name: &str, x: S) -> Result<(S, SpecialFunctionResult<S>), SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    Ok(match name {
        "lambert_w0" => (
            lambert_w0(x)?,
            lambert_w0_with_options(x, LambertOptions::default())?,
        ),
        "lambert_wm1" => (
            lambert_wm1(x)?,
            lambert_wm1_with_options(x, LambertOptions::default())?,
        ),
        "wright_omega" => (
            wright_omega(x)?,
            wright_omega_with_options(x, LambertOptions::default())?,
        ),
        "dilog" => (dilog(x)?, dilog_with_options(x, PolylogOptions::default())?),
        "spence" => (
            spence(x)?,
            spence_with_options(x, PolylogOptions::default())?,
        ),
        _ => panic!("unknown real function {name}"),
    })
}

fn oracle<S: RealScalar>(scalar: &str, polylog: bool)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let decode = |hex: &str| {
        let bits = u64::from_str_radix(hex, 16).unwrap();
        S::try_from_f64_exact(if scalar == "f64" {
            f64::from_bits(bits)
        } else {
            f64::from(f32::from_bits(bits as u32))
        })
        .unwrap()
    };
    let mut envelopes = BTreeMap::<&str, Vec<f64>>::new();
    for line in DATA.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[1] != scalar {
            continue;
        }
        let name = fields[2];
        if matches!(name, "dilog" | "spence") != polylog {
            continue;
        }
        let x = decode(fields[3].strip_prefix("f:").unwrap());
        let expected = decode(fields[4]);
        let (plain, detailed) = pair(name, x).unwrap_or_else(|error| panic!("{line}: {error}"));
        let actual = detailed.try_value().unwrap();
        assert!(
            plain == actual && plain.is_sign_negative() == actual.is_sign_negative(),
            "{line}"
        );
        let unit = (if expected.abs() == S::MAX {
            expected.abs() - expected.abs().next_after(S::ZERO)
        } else {
            expected.abs().next_after(S::INFINITY) - expected.abs()
        })
        .try_to_f64_exact()
        .unwrap();
        let difference = (actual - expected).try_to_f64_exact().unwrap();
        let offset: f64 = fields[6].parse().unwrap();
        let ulps = (difference / unit - offset).abs();
        let error = (difference - offset * unit).abs();
        let bound = detailed
            .value()
            .unwrap()
            .absolute_error()
            .try_to_f64_exact()
            .unwrap();
        assert!(error <= bound, "{line}: error={error:e}, bound={bound:e}");
        if fields[0] == "measured" {
            envelopes.entry(name).or_default().push(ulps);
        }
        if scalar == "f64"
            && name == "dilog"
            && (x.try_to_f64_exact().unwrap() - 12.595170369845016).abs() < 1e-10
        {
            assert!(error <= 5e-16, "dilog zero: {line}, error={error:e}");
        }
    }
    for (name, mut values) in envelopes {
        assert!(values.len() >= if scalar == "f64" { 4096 } else { 1000 });
        values.sort_by(f64::total_cmp);
        let max = *values.last().unwrap();
        let p99 = values[(values.len() * 99).div_ceil(100) - 1];
        eprintln!("{scalar} {name}: max={max:.4} ULP, p99={p99:.4} ULP");
        let target = if scalar == "f32" {
            1.0
        } else if name == "spence" {
            2.8
        } else {
            3.0
        };
        if name == "dilog" && scalar == "f64" {
            assert!(p99 <= 31.0, "{name}: p99={p99}");
        } else {
            assert!(max <= target, "{name}: max={max}, target={target}");
        }
    }
}

#[test]
fn real_f64_envelopes_and_reports() {
    oracle::<f64>("f64", false);
}

#[test]
fn real_f32_envelopes_and_reports() {
    oracle::<f32>("f32", false);
}

#[test]
fn real_values_and_reports_do_not_allocate() {
    for (name, x) in [
        ("lambert_w0", -0.365),
        ("lambert_wm1", -0.1),
        ("wright_omega", -1000.0),
        ("dilog", 12.595170369845016),
        ("spence", 1.001),
    ] {
        let _ = pair(name, x).unwrap();
    }
}

#[test]
fn real_derivatives_and_small_budgets() {
    for (name, x) in [
        ("lambert_w0", 3.0_f64),
        ("lambert_wm1", -0.1),
        ("wright_omega", 3.0),
        ("dilog", 1.5),
        ("spence", 0.5),
    ] {
        let (w, report) = pair(name, x).unwrap();
        let expected = match name {
            "lambert_w0" | "lambert_wm1" => w / x / (1.0 + w),
            "wright_omega" => w / (1.0 + w),
            "dilog" => -(1.0 - x).abs().ln() / x,
            "spence" => x.ln() / (1.0 - x),
            _ => unreachable!(),
        };
        assert!(
            (report.derivative().unwrap().value() - expected).abs()
                <= 8.0 * f64::EPSILON * expected.abs()
        );
    }
    let options = LambertOptions::default().with_iteration_budget(Budget::ONE);
    for report in [
        lambert_w0_with_options(3.0, options),
        lambert_wm1_with_options(-0.1, options),
        wright_omega_with_options(3.0, options),
    ] {
        let report = report.unwrap();
        assert_eq!(
            report.termination(),
            SpecialFunctionTermination::IterationLimit
        );
        assert_eq!(report.counts().iterations(), 1);
        assert!(report.value().is_some());
        assert!(report.try_value().is_err());
    }
    for limit in [1, 5, 13] {
        let options =
            PolylogOptions::default().with_iteration_budget(Budget::try_from(limit).unwrap());
        for report in [
            dilog_with_options(0.5, options),
            dilog_with_options(12.6, options),
            spence_with_options(0.5, options),
        ] {
            let report = report.unwrap_err();
            let report_counts = publication::exhausted_counts(&report);
            assert_eq!(report_counts.iterations(), limit);
        }
    }
}

#[test]
fn real_omega_scope_observes_cancellation_and_counts() {
    use scioxide_core::{
        CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(1047),
        scioxide_core::algorithm_id!("test", "real_omega", 1),
        0,
        0,
    );
    let cancellation = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    let options = LambertOptions::default().with_iteration_budget(Budget::ONE);
    let result = wright_omega_scoped(3.0, options, &mut context).unwrap();
    assert_eq!(
        result.termination(),
        SpecialFunctionTermination::IterationLimit
    );
    assert!(result.value().is_some());
    assert_eq!(context.counts().nested().iterations(), 1);
    assert_eq!(context.counts().own().iterations(), 0);
    cancellation.cancel();
    assert!(matches!(
        wright_omega_scoped(-1000.0, options, &mut context)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(context.counts().nested().iterations(), 1);
}

#[test]
fn real_dilog_f64_envelopes_and_reports() {
    oracle::<f64>("f64", true);
}
#[test]
fn real_dilog_f32_envelopes_and_reports() {
    oracle::<f32>("f32", true);
}
