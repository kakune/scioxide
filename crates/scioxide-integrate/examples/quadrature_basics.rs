//! Basic quadrature rule demonstration for scioxide-integrate.
//!
//! Demonstrates fixed-order 1D integration rules and their accuracy:
//!
//! 1. **Newton-Cotes basic rules** — Midpoint, Trapezoidal, Simpson, Simpson 3/8, Boole
//! 2. **Composite Newton-Cotes** — Composite trapezoidal and Simpson with varying panels
//! 3. **Gauss-Legendre** — Varying number of nodes on polynomial and smooth integrands
//! 4. **Gauss-Kronrod** — All five GK pairs with embedded error estimates
//! 5. **Clenshaw-Curtis and Romberg** — High-accuracy fixed-order and adaptive rules
//!
//! Run: `cargo run --example quadrature_basics -p scioxide-integrate`

#![allow(
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::unnecessary_wraps,
    clippy::uninlined_format_args
)]

use scioxide_core::{Budget, ErrorTolerance, Interval, NormSelection, SciOxideError};
use scioxide_integrate::{
    ClenshawCurtis, FixedRule, GaussKronrod, GaussKronrodRule, GaussianRulePlan,
    IntegrationTermination, LevelOptions, NewtonCotes, NewtonCotesEndpoints, Romberg,
};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Part 1: Newton-Cotes basic rules
// ---------------------------------------------------------------------------

fn part1_newton_cotes_basic() -> Result<(), SciOxideError> {
    println!("=== Part 1: Newton-Cotes basic rules ===");
    println!();

    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(x * x) };
    let interval = Interval::new(0.0, 1.0)?;
    let expected = 1.0 / 3.0; // ∫₀¹ x² dx = 1/3

    // (name, stencil nodes, endpoint convention, single-panel value)
    let rules = [
        ("Midpoint", 1, NewtonCotesEndpoints::Open, 0.25),
        ("Trapezoidal", 2, NewtonCotesEndpoints::Closed, 0.5),
        ("Simpson", 3, NewtonCotesEndpoints::Closed, expected),
        ("Simpson 3/8", 4, NewtonCotesEndpoints::Closed, expected),
        ("Boole", 5, NewtonCotesEndpoints::Closed, expected),
    ];

    for &(name, nodes, endpoints, rule_expected) in &rules {
        let nc = NewtonCotes::new(nodes, endpoints, Budget::ONE)?;
        let (value, report) = nc.integrate(&f, interval)?;
        let err = (value - expected).abs();
        println!("  {name:<14}  value = {:.10}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(value.is_finite());
        assert!((value - rule_expected).abs() < 1e-14);
        assert_eq!(report.evaluations(), nodes);
        // Simpson and higher are exact for degree-2 polynomials.
        assert!(nc.exactness_degree() < 2 || err < 1e-14);
    }

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Composite Newton-Cotes
// ---------------------------------------------------------------------------

fn part2_composite_newton_cotes() -> Result<(), SciOxideError> {
    println!("=== Part 2: Composite Newton-Cotes ===");
    println!();

    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(x.sin()) };
    let interval = Interval::new(0.0, std::f64::consts::PI)?;
    let expected = 2.0; // ∫₀^π sin(x) dx = 2

    let panel_counts = [10, 100, 1000];

    println!("  Composite Trapezoidal:");
    let mut previous_error = f64::INFINITY;
    for &panels in &panel_counts {
        let nc = NewtonCotes::new(2, NewtonCotesEndpoints::Closed, Budget::try_from(panels)?)?;
        let (value, report) = nc.integrate(&f, interval)?;
        let err = (value - expected).abs();
        println!(
            "    panels={panels:<5}  value = {:.10}  |error| = {err:.2e}",
            value
        );
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(value.is_finite());
        assert_eq!(report.evaluations(), panels + 1);
        assert!(err < previous_error);
        previous_error = err;
    }

    println!("  Composite Simpson:");
    let mut previous_error = f64::INFINITY;
    for &panels in &panel_counts {
        let nc = NewtonCotes::new(3, NewtonCotesEndpoints::Closed, Budget::try_from(panels)?)?;
        let (value, report) = nc.integrate(&f, interval)?;
        let err = (value - expected).abs();
        println!(
            "    panels={panels:<5}  value = {:.10}  |error| = {err:.2e}",
            value
        );
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(value.is_finite());
        assert_eq!(report.evaluations(), 2 * panels + 1);
        assert!(err < previous_error);
        previous_error = err;
    }

    // Composite Simpson with 1000 panels should be very accurate
    let nc = NewtonCotes::new(3, NewtonCotesEndpoints::Closed, Budget::try_from(1000)?)?;
    let (value, _) = nc.integrate(&f, interval)?;
    assert!(
        (value - expected).abs() < 1e-12,
        "Composite Simpson with 1000 panels"
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Gauss-Legendre
// ---------------------------------------------------------------------------

fn part3_gauss_legendre() -> Result<(), SciOxideError> {
    println!("=== Part 3: Gauss-Legendre ===");
    println!();

    let interval = Interval::new(0.0, 1.0)?;

    // ∫₀¹ exp(x) dx = e − 1
    let f_exp = |x: f64| -> Result<f64, SciOxideError> { Ok(x.exp()) };
    let expected_exp = std::f64::consts::E - 1.0;

    println!("  ∫₀¹ exp(x) dx = e−1 ≈ {expected_exp:.10}");
    for &(n, tolerance) in &[(5, 1e-12), (10, 1e-14), (20, 1e-14), (50, 1e-14)] {
        let gl = GaussianRulePlan::gauss_legendre(n)?;
        let (value, report) = gl.integrate(&f_exp, interval)?;
        let err = (value - expected_exp).abs();
        println!("    n={n:<3}  value = {:.12}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(err < tolerance);
        assert_eq!(report.evaluations(), n);
    }

    // ∫₀¹ x⁴ dx = 1/5
    let f_x4 = |x: f64| -> Result<f64, SciOxideError> { Ok(x.powi(4)) };
    let expected_x4 = 0.2;

    println!();
    println!("  ∫₀¹ x⁴ dx = 1/5");
    for &n in &[3, 5, 10] {
        let gl = GaussianRulePlan::gauss_legendre(n)?;
        let (value, report) = gl.integrate(&f_x4, interval)?;
        let err = (value - expected_x4).abs();
        println!("    n={n:<3}  value = {:.12}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(err < 1e-14);
        assert_eq!(report.evaluations(), n);
    }

    // GL with n=3 should exactly integrate degree-5 polynomials, so x⁴ is exact
    let (value, _) = GaussianRulePlan::gauss_legendre(3)?.integrate(&f_x4, interval)?;
    assert!(
        (value - expected_x4).abs() < 1e-14,
        "GL(3) exact for degree ≤ 5"
    );

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Gauss-Kronrod pairs with error estimates
// ---------------------------------------------------------------------------

fn part4_gauss_kronrod() -> Result<(), SciOxideError> {
    println!("=== Part 4: Gauss-Kronrod pairs ===");
    println!();

    // ∫₀¹ exp(−x²) dx ≈ 0.746824132812427
    let f = |x: f64| -> Result<f64, SciOxideError> { Ok((-x * x).exp()) };
    let expected = 0.746_824_132_812_427_1;
    let interval = Interval::new(0.0, 1.0)?;

    let pairs = [
        ("G7K15", GaussKronrodRule::G7K15, 15_usize),
        ("G10K21", GaussKronrodRule::G10K21, 21),
        ("G15K31", GaussKronrodRule::G15K31, 31),
        ("G20K41", GaussKronrodRule::G20K41, 41),
        ("G25K51", GaussKronrodRule::G25K51, 51),
    ];

    for (name, pair, evaluations) in &pairs {
        let gk = GaussKronrod::new(*pair)?;
        let (value, report) = gk.integrate(&f, interval)?;
        let actual_err = (value - expected).abs();
        println!(
            "  {name:<7}  value = {:.12}  err_est = {:.2e}  |actual| = {actual_err:.2e}  evals = {}",
            value,
            report.error_estimate().unwrap_or(f64::NAN),
            report.evaluations()
        );
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(value.is_finite());
        assert!(actual_err < 1e-13, "{name} failed the Gaussian oracle");
        assert!(
            report
                .error_estimate()
                .is_some_and(|error| error.is_finite() && error >= 0.0)
        );
        assert_eq!(report.evaluations(), *evaluations);
    }

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Clenshaw-Curtis and Romberg
// ---------------------------------------------------------------------------

fn part5_clenshaw_curtis_romberg() -> Result<(), SciOxideError> {
    println!("=== Part 5: Clenshaw-Curtis and Romberg ===");
    println!();

    // Runge function: ∫₋₁¹ 1/(1+25x²) dx = 2·atan(5)/5
    let f = |x: f64| -> Result<f64, SciOxideError> { Ok(1.0 / (1.0 + 25.0 * x * x)) };
    let expected = 2.0 * 5.0_f64.atan() / 5.0;
    let interval = Interval::new(-1.0, 1.0)?;

    println!("  Runge function: ∫₋₁¹ 1/(1+25x²) dx = 2·atan(5)/5 ≈ {expected:.10}");

    // Clenshaw-Curtis with varying n
    println!();
    println!("  Clenshaw-Curtis:");
    let mut previous_error = f64::INFINITY;
    for &(n, tolerance) in &[(8, 0.06), (16, 0.003), (32, 5e-6), (64, 5e-11)] {
        let cc = ClenshawCurtis::new(n)?;
        let (value, report) = cc.integrate(&f, interval)?;
        let err = (value - expected).abs();
        println!("    n={n:<3}  value = {:.10}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(value.is_finite());
        assert_eq!(report.evaluations(), n + 1);
        assert!(err < tolerance && err < previous_error);
        previous_error = err;
    }

    // Romberg
    println!();
    println!("  Romberg:");
    let options = LevelOptions::default()
        .with_tolerance(ErrorTolerance::scalar(
            1e-12,
            1e-12,
            NormSelection::Infinity,
        )?)
        .with_max_levels(Budget::try_from(30)?);
    let rom = Romberg::new(options)?;
    let (value, report) = rom.integrate(&f, interval)?;
    println!(
        "    value = {:.12}  err_est = {:.2e}  evals = {}  termination = {}",
        value,
        report.error_estimate().unwrap_or(f64::NAN),
        report.evaluations(),
        report.termination()
    );
    assert!(
        (value - expected).abs() < TOL,
        "Romberg should converge for Runge function"
    );
    assert_eq!(
        report.termination(),
        IntegrationTermination::ToleranceReached
    );
    assert!(report.error_estimate().is_some_and(f64::is_finite));

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
    println!("scioxide-integrate Quadrature Basics");
    println!("===================================");
    println!();

    part1_newton_cotes_basic()?;
    part2_composite_newton_cotes()?;
    part3_gauss_legendre()?;
    part4_gauss_kronrod()?;
    part5_clenshaw_curtis_romberg()?;

    println!("All examples completed successfully.");
    Ok(())
}
