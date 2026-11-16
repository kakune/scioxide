//! QAGS/QAGP numerical acceptance and prepared-execution contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, Complex, ErrorTolerance, ExecutionContext,
    ExecutionStopReason, Interval, NormSelection, SciOxideError,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, EndpointWeight, ErrorReliability, IntegrationTermination,
    SubdivisionOptions, WeightedQuadratureOptions, WeightedQuadraturePlan,
};

fn unit() -> Interval<f64> {
    Interval::new(0.0, 1.0).unwrap()
}
fn plan() -> AdaptiveIntegrator<f64> {
    AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
        .unwrap()
        .with_extrapolation(true)
        .unwrap()
}

type Integrand = fn(f64) -> f64;

#[test]
fn singular_endpoints_meet_qags_accuracy_and_callback_targets() {
    let cases: [(Integrand, f64, usize); 4] = [
        (|x| x.sqrt().recip(), 2.0, 231),
        (f64::ln, -1.0, 231),
        (|x| x.sqrt() * x.ln(), -4.0 / 9.0, 315),
        (|x| x.powf(-0.9), 10.0, 231),
    ];
    let solver = plan();
    let mut workspace = solver.workspace(1).unwrap();
    for (function, expected, qags_evaluations) in cases {
        let mut value = 0.0;
        let report = solver
            .integrate_into(
                |x| Ok(function(x)),
                unit(),
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        eprintln!(
            "expected={expected}, value={value}, termination={}, evaluations={}, error={:?}",
            report.termination(),
            report.evaluations(),
            report.error_estimate()
        );
        assert!(report.is_success());
        let tolerance = f64::EPSILON.sqrt() * expected.abs().max(1.0);
        assert!((value - expected).abs() <= tolerance);
        assert!(2 * report.evaluations() <= 3 * qags_evaluations);
        let error = workspace.component_errors().unwrap()[0];
        assert_eq!(
            error.extrapolation().unwrap().reliability(),
            ErrorReliability::Estimated
        );
        assert_eq!(error.total(), report.error_estimate());
        assert!(error.total().unwrap() <= tolerance);
    }
}

#[test]
fn breakpoint_singularities_and_unequal_roots_converge() {
    for split in [0.25_f64, 0.5, 0.9] {
        let solver = WeightedQuadraturePlan::breakpoints(
            unit(),
            &[split],
            WeightedQuadratureOptions::default(),
        )
        .unwrap()
        .with_extrapolation(true)
        .unwrap();
        let expected = 2.0 * (split.sqrt() + (1.0 - split).sqrt());
        let (value, report) = solver
            .integrate(|x: f64| {
                assert_ne!(x, split);
                Ok((x - split).abs().sqrt().recip())
            })
            .unwrap();
        assert!(report.is_success(), "{split}: {report:?}");
        assert!((value - expected).abs() < expected * f64::EPSILON.sqrt());
        assert!(report.error_components().unwrap().extrapolation().is_some());
    }
}

#[test]
fn two_singular_endpoints_and_power_log_sequences_converge() {
    let (value, report) = plan()
        .integrate(|x: f64| Ok((x * (1.0 - x)).sqrt().recip()), unit())
        .unwrap();
    assert!(report.is_success(), "{report:?}");
    assert!((value - core::f64::consts::PI).abs() < 5e-8);
    for exponent in [-0.95_f64, -0.75, -0.25, 0.25, 0.75, 1.25] {
        let (value, report) = plan()
            .integrate(|x: f64| Ok(x.powf(exponent) * x.ln()), unit())
            .unwrap();
        let expected = -1.0 / (exponent + 1.0).powi(2);
        assert!(report.is_success(), "{exponent}: {report:?}");
        assert!((value - expected).abs() < f64::EPSILON.sqrt() * expected.abs().max(1.0));
    }
}

#[test]
fn single_root_entrypoints_share_the_extrapolated_schedule() {
    let options = SubdivisionOptions::default();
    let adaptive = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone())
        .unwrap()
        .with_extrapolation(true)
        .unwrap();
    let weighted =
        WeightedQuadraturePlan::breakpoints(unit(), &[], WeightedQuadratureOptions::new(options))
            .unwrap()
            .with_extrapolation(true)
            .unwrap();
    let function = |x: f64| Ok([x.powf(-0.9), x.sqrt() * x.ln()]);
    let (expected, report) = adaptive.integrate(&function, unit()).unwrap();
    let (actual, other) = weighted.integrate(&function).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(other.termination(), report.termination());
    assert_eq!(other.evaluations(), report.evaluations());
    assert_eq!(other.error_estimate(), report.error_estimate());
}

#[test]
fn components_share_refinement_but_keep_independent_epsilon_evidence() {
    for norm in [NormSelection::Infinity, NormSelection::L2] {
        let options = SubdivisionOptions::default()
            .with_tolerance(ErrorTolerance::try_vector(&[1e-9, 1e-8, 1e-10], 1e-9, norm).unwrap());
        let solver = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options)
            .unwrap()
            .with_extrapolation(true)
            .unwrap();
        let mut values = [0.0; 3];
        let mut workspace = solver.workspace(3).unwrap();
        let report = solver
            .integrate_into(
                |x: f64| Ok([x.powf(-0.9), x.ln(), x * x]),
                unit(),
                &mut values,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.is_success(), "{report:?}");
        let errors: Vec<_> = workspace
            .component_errors()
            .unwrap()
            .iter()
            .map(|parts| parts.total().unwrap())
            .collect();
        let scales = values.map(f64::abs);
        assert!(
            solver
                .options()
                .tolerance()
                .normalized_norm(&errors, &scales)
                .unwrap()
                <= 1.0
        );
        for (actual, expected) in values.into_iter().zip([10.0, -1.0, 1.0 / 3.0]) {
            assert!((actual - expected).abs() < 1e-8);
        }
    }
    let (value, report) = plan()
        .integrate(|x: f64| Ok(Complex::new(x.powf(-0.9), x.ln())), unit())
        .unwrap();
    assert!(report.is_success());
    assert!((value.re - 10.0).abs() < 1e-7 && (value.im + 1.0).abs() < 1e-8);
    let solver =
        AdaptiveIntegrator::<f32>::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .with_extrapolation(true)
            .unwrap();
    let (value, report) = solver
        .integrate(|x: f32| Ok(x.powf(-0.9)), Interval::new(0.0, 1.0).unwrap())
        .unwrap();
    assert!(report.is_success());
    assert!((value - 10.0).abs() < 0.004);
}

#[test]
fn smooth_peaked_oscillatory_and_cancelling_cases_retain_accuracy() {
    let cases: [(Integrand, f64); 4] = [
        (|x| x.exp(), core::f64::consts::E - 1.0),
        (
            |x| 1.0 / (1.0 + 100.0 * (x - 0.3).powi(2)),
            (7.0_f64.atan() + 3.0_f64.atan()) / 10.0,
        ),
        (|x| (100.0 * x).sin(), (1.0 - 100.0_f64.cos()) / 100.0),
        (|x| (100.0 * (x - 0.5)).sin(), 0.0),
    ];
    for (function, expected) in cases {
        for enabled in [false, true] {
            let solver = plan().with_extrapolation(enabled).unwrap();
            let (value, report) = solver.integrate(|x| Ok(function(x)), unit()).unwrap();
            assert!(report.is_success(), "enabled={enabled}: {report:?}");
            assert!((value - expected).abs() <= f64::EPSILON.sqrt() * expected.abs().max(1.0));
        }
    }
    let (value, report) = plan()
        .integrate(|x: f64| Ok(x.sqrt().recip() - 2.0), unit())
        .unwrap();
    assert!(report.is_success());
    assert!(value.abs() < f64::EPSILON.sqrt());
}

#[test]
fn a_zero_epsilon_limit_cannot_satisfy_relative_only_error_control() {
    let solver = AdaptiveIntegrator::new(
        AdaptiveRule::GaussKronrod,
        SubdivisionOptions::default()
            .with_tolerance(ErrorTolerance::scalar(0.0, 1e-8, NormSelection::Infinity).unwrap()),
    )
    .unwrap()
    .with_extrapolation(true)
    .unwrap();
    let (value, report) = solver
        .integrate(|x: f64| Ok(x.sqrt().recip() - 2.0), unit())
        .unwrap();
    assert!(!report.is_success());
    assert!(value.is_finite());
    assert!(report.error_estimate().unwrap() > 0.0);
}

#[test]
fn divergent_integrands_never_claim_success() {
    for exponent in [-1.0_f64, -1.1, -2.0] {
        let (_, report) = plan()
            .integrate(|x: f64| Ok(x.powf(exponent)), unit())
            .unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::ProbablyDivergent,
            "{exponent}: {report:?}"
        );
    }
}

#[test]
fn budgets_and_atomic_errors_keep_the_correct_estimate() {
    for (limit, calls) in [(15, 15), (44, 15), (45, 45), (74, 45)] {
        let solver = AdaptiveIntegrator::new(
            AdaptiveRule::GaussKronrod,
            SubdivisionOptions::default()
                .with_max_evaluations(Some(Budget::try_from(limit).unwrap())),
        )
        .unwrap()
        .with_extrapolation(true)
        .unwrap();
        let (_, report) = solver.integrate(|x: f64| Ok(x.powf(-0.9)), unit()).unwrap();
        assert_eq!(report.evaluations(), calls);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(report.error_components().unwrap().extrapolation().is_none());
    }
    let solver = plan();
    let mut workspace = solver.workspace(1).unwrap();
    let mut output = 42.0;
    let error = solver.integrate_into(
        |x: f64| -> Result<f64, SciOxideError> {
            if x < 0.001 {
                Ok(f64::NAN)
            } else {
                Ok(x.powf(-0.9))
            }
        },
        unit(),
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    );
    assert!(error.is_err());
    assert_eq!(output, 42.0);
    assert!(workspace.component_errors().is_none());
}

#[test]
fn cancellation_zero_domains_and_unachievable_tolerances_preserve_contracts() {
    let solver = plan();
    let mut workspace = solver.workspace(1).unwrap();
    let mut value = 42.0;
    let token = CancellationToken::new();
    let report = solver
        .integrate_into(
            |x: f64| {
                token.cancel();
                Ok(x.powf(-0.9))
            },
            unit(),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native().with_cancellation(&token),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 15);
    assert_eq!(
        report.termination(),
        IntegrationTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    assert!(value.is_finite() && value != 42.0);
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let report = solver
        .integrate_into(
            |_: f64| -> Result<f64, SciOxideError> {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(1.0)
            },
            Interval::new(0.5, 0.5).unwrap(),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(value, 0.0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(
        report.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert!(workspace.component_errors().is_none());

    let exact = AdaptiveIntegrator::new(
        AdaptiveRule::GaussKronrod,
        SubdivisionOptions::default()
            .with_tolerance(ErrorTolerance::scalar(1e-30, 1e-30, NormSelection::Infinity).unwrap()),
    )
    .unwrap()
    .with_extrapolation(true)
    .unwrap();
    let (estimate, report) = exact.integrate(|x: f64| Ok(x.powf(-0.9)), unit()).unwrap();
    assert!(!report.is_success());
    assert!(estimate.is_finite());
    assert!(report.error_estimate().unwrap() > 0.0);
    let tiny = AdaptiveIntegrator::new(
        AdaptiveRule::GaussKronrod,
        SubdivisionOptions::default().with_max_evaluations(Some(Budget::try_from(14).unwrap())),
    )
    .unwrap()
    .with_extrapolation(true)
    .unwrap();
    value = 42.0;
    assert!(
        tiny.integrate_into(
            |x: f64| Ok(x.powf(-0.9)),
            unit(),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(value, 42.0);
}

#[test]
fn policy_rejects_specialized_weights_and_defaults_to_disabled() {
    assert!(
        !AdaptiveIntegrator::<f64>::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .extrapolation_enabled()
    );
    let breakpoint =
        WeightedQuadraturePlan::breakpoints(unit(), &[], WeightedQuadratureOptions::default())
            .unwrap();
    assert!(!breakpoint.extrapolation_enabled());
    assert!(
        breakpoint
            .with_extrapolation(true)
            .unwrap()
            .extrapolation_enabled()
    );
    let weighted = WeightedQuadraturePlan::endpoint(
        unit(),
        &[],
        EndpointWeight::new(-0.5, 0.0, false, false).unwrap(),
        WeightedQuadratureOptions::default(),
    )
    .unwrap();
    assert!(weighted.with_extrapolation(true).is_err());
}
