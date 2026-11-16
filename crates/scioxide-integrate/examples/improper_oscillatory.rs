//! Improper and oscillatory integration demonstration for scioxide-integrate.
//!
//! Demonstrates integration on difficult domains:
//!
//! 1. **Tanh-sinh (finite, singular endpoints)** — ∫₀¹ 1/√x dx = 2
//! 2. **Exp-sinh (semi-infinite)** — ∫₀^∞ exp(−x) dx = 1
//! 3. **Sinh-sinh (doubly infinite)** — ∫₋∞^∞ exp(−x²) dx = √π
//! 4. **Interior singularity** — ∫₀¹ 1/√|x−0.5| dx via prepared breakpoints
//! 5. **Oscillatory integration** — ∫₀^{2π} sin(100x) dx = 0
//!
//! Run: `cargo run --example improper_oscillatory -p scioxide-integrate`

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
    DoubleExponentialDomain, DoubleExponentialIntegrator, FourierWeight, IntegrationReport,
    IntegrationTermination, LevelOptions, SubdivisionOptions, WeightedQuadraturePlan,
};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-4;

fn measured(
    result: &(f64, scioxide_integrate::IntegrationReport<f64>),
) -> Result<f64, SciOxideError> {
    Ok(result.0)
}

/// Double-exponential integrator with absolute and relative tolerances of 1e-8.
fn double_exponential(
    domain: DoubleExponentialDomain<'_, f64>,
) -> Result<DoubleExponentialIntegrator<f64>, SciOxideError> {
    let tolerance = ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity)?;
    DoubleExponentialIntegrator::new(domain, LevelOptions::default().with_tolerance(tolerance))
}

/// Print and check a double-exponential estimate against its exact value.
fn check_double_exponential(value: f64, expected: f64, report: &IntegrationReport<f64>) -> f64 {
    let err = (value - expected).abs();
    println!(
        "  value = {:.10}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        report.error_estimate().unwrap_or(f64::NAN),
        report.evaluations(),
        report.termination()
    );
    println!("  |actual error| = {err:.2e}");
    // Tail truncation is heuristic: success means estimated, not certified, error control.
    assert_eq!(
        report.termination(),
        IntegrationTermination::HeuristicToleranceReached
    );
    println!("Tail diagnostics: {:?}", report.error_components());
    assert!(
        report
            .error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
    );
    err
}

// ---------------------------------------------------------------------------
// Part 1: Tanh-sinh (endpoint singularity)
// ---------------------------------------------------------------------------

fn part1_tanh_sinh() -> Result<(), SciOxideError> {
    println!("=== Part 1: Tanh-sinh (endpoint singularity) ===");
    println!();

    // ∫₀¹ 1/√x dx = 2
    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(1.0 / x.sqrt()) };
    let expected = 2.0;
    let integrator = double_exponential(DoubleExponentialDomain::Finite {
        interval: Interval::new(0.0, 1.0)?,
        singular_points: &[],
    })?;
    // A caller-owned workspace for one real component and no interior singular points.
    let mut workspace = integrator.workspace(1)?;
    let mut value = 0.0;
    let report = integrator.integrate_into(
        |point: scioxide_integrate::DoubleExponentialPoint<_>| f(point.coordinate()),
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;

    println!("  ∫₀¹ 1/√x dx = 2");
    let err = check_double_exponential(value, expected, &report);
    assert!(
        workspace
            .component_errors()
            .is_some_and(|errors| errors.len() == 1)
    );
    assert!(err < TOL, "tanh-sinh should handle 1/√x");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Exp-sinh (semi-infinite)
// ---------------------------------------------------------------------------

fn part2_exp_sinh() -> Result<(), SciOxideError> {
    println!("=== Part 2: Exp-sinh (semi-infinite) ===");
    println!();

    // ∫₀^∞ exp(−x) dx = 1
    let f = |x: f64| -> Result<f64, SciOxideError> { Ok((-x).exp()) };
    let expected = 1.0;
    let (value, report) = double_exponential(DoubleExponentialDomain::HalfLine { lower: 0.0 })?
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| f(point.coordinate()))?;

    println!("  ∫₀^∞ exp(−x) dx = 1");
    let err = check_double_exponential(value, expected, &report);
    assert!(
        report.evaluations() <= 2_222,
        "exp-sinh callback regression"
    );
    assert!(err < TOL, "exp-sinh should handle exp(-x)");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Sinh-sinh (doubly infinite)
// ---------------------------------------------------------------------------

fn part3_sinh_sinh() -> Result<(), SciOxideError> {
    println!("=== Part 3: Sinh-sinh (doubly infinite) ===");
    println!();

    // ∫₋∞^∞ exp(−x²) dx = √π
    let f = |x: f64| -> Result<f64, SciOxideError> { Ok((-x * x).exp()) };
    let expected = std::f64::consts::PI.sqrt();
    let (value, report) = double_exponential(DoubleExponentialDomain::RealLine)?
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| f(point.coordinate()))?;

    println!("  ∫₋∞^∞ exp(−x²) dx = √π ≈ {expected:.10}");
    let err = check_double_exponential(value, expected, &report);
    assert!(err < TOL, "sinh-sinh should handle Gaussian integral");

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Interior singularity
// ---------------------------------------------------------------------------

fn part4_interior_singularity() -> Result<(), SciOxideError> {
    println!("=== Part 4: Interior singularity ===");
    println!();

    // ∫₀¹ 1/√|x−0.5| dx = 2(√0.5 + √0.5) = 2√2 ≈ 2.828...
    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(1.0 / (x - 0.5).abs().sqrt()) };
    let expected = 2.0 * std::f64::consts::SQRT_2;
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-6, 1e-6, NormSelection::Infinity)?)
        .with_max_subdivisions(Budget::try_from(256)?);
    let plan = WeightedQuadraturePlan::breakpoints(
        Interval::new(0.0, 1.0)?,
        &[0.5],
        scioxide_integrate::WeightedQuadratureOptions::new(options),
    )?;
    let res = plan.integrate(&f)?;
    let value = measured(&res)?;
    let err = (value - expected).abs();

    println!("  ∫₀¹ 1/√|x−0.5| dx = 2√2 ≈ {expected:.10}");
    println!(
        "  value = {:.10}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        res.1.error_estimate().unwrap_or(f64::NAN),
        res.1.evaluations(),
        res.1.termination()
    );
    println!("  |actual error| = {err:.2e}");
    assert!(
        res.1.is_success(),
        "prepared breakpoints should meet tolerance"
    );
    println!("Tail diagnostics: {:?}", res.1.error_components());
    assert!(
        res.1
            .error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
    );
    assert!(
        err < TOL,
        "prepared breakpoints should handle interior singularity"
    );

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Oscillatory integration
// ---------------------------------------------------------------------------

fn part5_oscillatory() -> Result<(), SciOxideError> {
    println!("=== Part 5: Oscillatory integration ===");
    println!();

    // ∫₀^{2π} sin(100x) dx = [-cos(100x)/100]₀^{2π} = 0
    let f = |_: f64| -> Result<f64, SciOxideError> { Ok(1.0) };
    let expected = 0.0;
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-6, 1e-6, NormSelection::Infinity)?)
        .with_max_subdivisions(Budget::try_from(500)?);

    let two_pi = 2.0 * std::f64::consts::PI;
    let plan = WeightedQuadraturePlan::fourier(
        Interval::new(0.0, two_pi)?,
        &[],
        100.0,
        FourierWeight::Sine,
        scioxide_integrate::WeightedQuadratureOptions::new(options)
            .with_moment_levels(Budget::try_from(16)?),
    )?;
    let res = plan.integrate(&f)?;
    let value = measured(&res)?;
    let err = (value - expected).abs();

    assert!(
        res.1.is_success(),
        "oscillatory integration should converge"
    );
    assert_eq!(
        res.1.termination(),
        IntegrationTermination::ToleranceReached
    );
    assert!(
        res.1
            .error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate >= 0.0)
    );
    assert_eq!(
        res.1.evaluations(),
        25,
        "one prepared moment rule integrates the constant amplitude"
    );

    println!("  ∫₀^{{2π}} sin(100x) dx = 0");
    println!(
        "  value = {:.10}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        res.1.error_estimate().unwrap_or(f64::NAN),
        res.1.evaluations(),
        res.1.termination()
    );
    println!("  |actual error| = {err:.2e}");
    assert!(err < 1e-10, "oscillatory integration should give near-zero");

    println!();
    println!("  Part 5 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-integrate Improper & Oscillatory Integration");
    println!("====================================================");
    println!();

    part1_tanh_sinh()?;
    part2_exp_sinh()?;
    part3_sinh_sinh()?;
    part4_interior_singularity()?;
    part5_oscillatory()?;

    println!("All examples completed successfully.");
    Ok(())
}
