//! Adaptive integration methods demonstration for scioxide-integrate.
//!
//! Demonstrates adaptive quadrature with automatic error control:
//!
//! 1. **Adaptive Gauss-Kronrod** — Smooth function under global error control
//! 2. **Adaptive Simpson** — Reusable stencils and a caller-owned workspace
//! 3. **Peaked integrand** — Tight tolerance on a sharply peaked function
//! 4. **Tolerance comparison** — How evaluation count scales with precision
//!
//! Run: `cargo run --example adaptive_methods -p scioxide-integrate`

#![allow(
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::uninlined_format_args
)]

use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, Interval, NormSelection, SciOxideError,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, IntegrationTermination, SubdivisionOptions,
};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-6;

/// Subdivision options with equal absolute and relative tolerances.
fn options(tolerance: f64) -> Result<SubdivisionOptions<f64>, SciOxideError> {
    Ok(SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(
            tolerance,
            tolerance,
            NormSelection::Infinity,
        )?)
        .with_max_subdivisions(Budget::try_from(200)?))
}

// ---------------------------------------------------------------------------
// Part 1: Adaptive Gauss-Kronrod on smooth functions
// ---------------------------------------------------------------------------

fn part1_adaptive_gauss_kronrod() -> Result<(), SciOxideError> {
    println!("=== Part 1: Adaptive Gauss-Kronrod ===");
    println!();

    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(x.sin()) };
    let expected = 2.0; // ∫₀^π sin(x) dx = 2
    let interval = Interval::new(0.0, std::f64::consts::PI)?;
    let agk = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())?;
    let (value, report) = agk.integrate(&f, interval)?;
    let err = (value - expected).abs();

    println!("  ∫₀^π sin(x) dx = 2");
    println!(
        "  value = {:.12}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        report.error_estimate().unwrap_or(f64::NAN),
        report.evaluations(),
        report.termination()
    );
    println!("  |actual error| = {err:.2e}");

    assert!(report.is_success(), "AGK should converge on sin(x)");
    assert_eq!(
        report.termination(),
        IntegrationTermination::ToleranceReached
    );
    assert!(
        report
            .error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
    );
    assert!(err < TOL, "AGK result should be accurate");
    assert_eq!(
        report.evaluations(),
        15,
        "smooth G7K15 should need one rule"
    );

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Adaptive Simpson
// ---------------------------------------------------------------------------

fn part2_adaptive_simpson() -> Result<(), SciOxideError> {
    println!("=== Part 2: Adaptive Simpson ===");
    println!();

    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(x.sin()) };
    let expected = 2.0;

    let asim = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options(1e-8)?)?;
    // A caller-owned workspace for one real component can be reused across calls.
    let mut workspace = asim.workspace(1)?;
    let mut value = 0.0;
    let report = asim.integrate_into(
        &f,
        Interval::new(0.0, std::f64::consts::PI)?,
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let err = (value - expected).abs();

    println!("  ∫₀^π sin(x) dx = 2");
    println!(
        "  value = {:.12}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        report.error_estimate().unwrap_or(f64::NAN),
        report.evaluations(),
        report.termination()
    );
    println!("  |actual error| = {err:.2e}");

    assert!(
        report.is_success(),
        "Adaptive Simpson should converge on sin(x)"
    );
    assert_eq!(
        report.termination(),
        IntegrationTermination::ToleranceReached
    );
    assert!(
        report
            .error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
    );
    assert!(
        workspace
            .component_errors()
            .is_some_and(|errors| errors.len() == 1)
    );
    assert!(err < TOL, "Adaptive Simpson result should be accurate");
    // Nine root nodes and one witness, eight per bisection, four complete depths.
    assert!(report.evaluations() >= 130);
    assert_eq!((report.evaluations() - 10) % 8, 0);

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Peaked integrand with tight tolerance
// ---------------------------------------------------------------------------

fn part3_peaked_integrand() -> Result<(), SciOxideError> {
    println!("=== Part 3: Peaked integrand ===");
    println!();

    // ∫₀¹ 1/(1+100(x−0.5)²) dx
    // Substitution: u = 10(x−0.5), du = 10 dx
    // = (1/10) ∫₋₅⁵ 1/(1+u²) du = (1/10)(atan(5) − atan(−5)) = 2·atan(5)/10
    let f = |x: f64| -> Result<f64, SciOxideError> {
        let d = x - 0.5;
        Ok(1.0 / (1.0 + 100.0 * d * d))
    };
    let expected = 5.0_f64.atan() / 5.0;

    let agk = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options(1e-10)?)?;
    let (value, report) = agk.integrate(&f, Interval::new(0.0, 1.0)?)?;
    let err = (value - expected).abs();

    println!("  ∫₀¹ 1/(1+100(x−0.5)²) dx = atan(5)/5 ≈ {expected:.10}");
    println!(
        "  value = {:.12}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        report.error_estimate().unwrap_or(f64::NAN),
        report.evaluations(),
        report.termination()
    );
    println!("  |actual error| = {err:.2e}");

    assert!(
        report.is_success(),
        "AGK should converge on the peaked integrand"
    );
    assert_eq!(
        report.termination(),
        IntegrationTermination::ToleranceReached
    );
    assert!(
        report
            .error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
    );
    assert!(err < 1e-8, "AGK should handle peaked integrand");
    assert_eq!((report.evaluations() - 15) % 30, 0);

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Tolerance comparison
// ---------------------------------------------------------------------------

fn part4_tolerance_comparison() -> Result<(), SciOxideError> {
    println!("=== Part 4: Tolerance comparison ===");
    println!();

    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(x.sin()) };
    let expected = 2.0;

    let tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12];

    println!("  ∫₀^π sin(x) dx = 2, varying tolerance:");
    println!(
        "  {:<12} {:<16} {:<12} {:<8}",
        "tol", "value", "|error|", "evals"
    );
    println!("  {}", "-".repeat(52));

    let interval = Interval::new(0.0, std::f64::consts::PI)?;
    for &tol_val in &tolerances {
        let agk = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options(tol_val)?)?;
        let (value, report) = agk.integrate(&f, interval)?;
        let err = (value - expected).abs();
        assert_eq!(
            report.termination(),
            IntegrationTermination::ToleranceReached,
            "AGK should reach tolerance {tol_val:e}"
        );
        assert!(value.is_finite());
        assert!(
            report
                .error_estimate()
                .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
        );
        assert!(err <= f64::max(tol_val, 1e-14));
        assert_eq!((report.evaluations() - 15) % 30, 0);
        println!(
            "  {tol_val:<12.0e} {:<16.12} {err:<12.2e} {}",
            value,
            report.evaluations()
        );
    }

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-integrate Adaptive Methods");
    println!("==================================");
    println!();

    part1_adaptive_gauss_kronrod()?;
    part2_adaptive_simpson()?;
    part3_peaked_integrand()?;
    part4_tolerance_comparison()?;

    println!("All examples completed successfully.");
    Ok(())
}
