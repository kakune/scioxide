//! Weighted and multidimensional integration demonstration for scioxide-integrate.
//!
//! Demonstrates Gauss quadrature with built-in weight functions and multi-dim methods:
//!
//! 1. **Gauss-Chebyshev** — Weight function 1/√(1−x²), ∫₋₁¹ 1/√(1−x²) dx = π
//! 2. **Gauss-Hermite and Laguerre** — Probabilist's Hermite and exponential weight
//! 3. **Gauss-Jacobi** — Weighted integral with (1−t)^α·(1+t)^β
//! 4. **Product rule (2D tensor product)** — ∫₀¹∫₀¹ xy dxdy = 1/4
//! 5. **Cubature and Monte Carlo (3D)** — ∫ exp(−(x²+y²+z²)) on [−3,3]³
//!
//! Run: `cargo run --example weighted_and_multidim -p scioxide-integrate`

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
    CubaturePlan, GaussianRulePlan, GaussianSupport, IntegrationTermination, MonteCarloIntegrator,
    MonteCarloOptions, ProductRule, RandomStreams, SubdivisionOptions,
};
use scioxide_poly::OrthogonalFamily;
use scioxide_random::{Philox4x64, SeedableRng};
use scioxide_stats::ConfidenceProbability;

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-4;

fn measured(result: &(f64, scioxide_integrate::IntegrationReport<f64>)) -> f64 {
    result.0
}
/// Product xy integrand for two dimensions.
fn product_xy(x: &[f64]) -> Result<f64, SciOxideError> {
    Ok(x[0] * x[1])
}
/// Gaussian density amplitude in three dimensions.
fn gaussian_3d(x: &[f64]) -> Result<f64, SciOxideError> {
    let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    Ok((-r2).exp())
}

fn part1_gauss_chebyshev() -> Result<(), SciOxideError> {
    println!("=== Part 1: Gauss-Chebyshev ===");
    println!();

    // GaussChebyshev computes ∫₋₁¹ f(x)/√(1−x²) dx
    // With f(x) = 1, this gives ∫₋₁¹ 1/√(1−x²) dx = π
    let f = |_x: f64| -> Result<f64, SciOxideError> { Ok(1.0) };
    let expected = std::f64::consts::PI;
    let support = GaussianSupport::Interval(Interval::new(-1.0, 1.0)?);

    println!("  ∫₋₁¹ 1/√(1−x²) dx = π ≈ {expected:.10}");
    for &n in &[5, 10, 20] {
        let gc = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, n)?;
        let (value, report) = gc.integrate_weighted(&f, support)?;
        let err = (value - expected).abs();
        println!("    n={n:<3}  value = {:.12}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(err < 1e-12);
        assert_eq!(report.evaluations(), n);
    }

    // Should be exact for constant f
    let gc20 = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 20)?;
    let (value, _) = gc20.integrate_weighted(&f, support)?;
    assert!(
        (value - expected).abs() < 1e-10,
        "Gauss-Chebyshev exact for constant function"
    );

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Gauss-Hermite and Laguerre
// ---------------------------------------------------------------------------

fn part2_hermite_laguerre() -> Result<(), SciOxideError> {
    println!("=== Part 2: Gauss-Hermite and Laguerre ===");
    println!();

    // Gauss-Hermite computes ∫_{-∞}^{∞} f(x) exp(-x²/2) dx
    // With f(x) = x², this gives E[X²] of N(0,1) · √(2π) = √(2π)
    let f_x2 = |x: f64| -> Result<f64, SciOxideError> { Ok(x * x) };
    let expected_hermite = (2.0 * std::f64::consts::PI).sqrt();
    println!("  Gauss-Hermite: ∫ x² exp(−x²/2) dx = √(2π) ≈ {expected_hermite:.10}");
    for &n in &[5, 10, 20] {
        let gh = GaussianRulePlan::new(OrthogonalFamily::HermiteHe, n)?;
        let (value, report) = gh.integrate_weighted(&f_x2, GaussianSupport::RealLine)?;
        let err = (value - expected_hermite).abs();
        println!("    n={n:<3}  value = {:.12}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(err < 1e-12);
        assert_eq!(report.evaluations(), n);
    }

    // Gauss-Laguerre computes ∫_a^∞ f(x) exp(−(x−a)) dx
    // With a=0, f(x) = x: ∫₀^∞ x·exp(−x) dx = 1
    let f_x = |x: f64| -> Result<f64, SciOxideError> { Ok(x) };
    let expected_laguerre = 1.0;

    println!();
    println!("  Gauss-Laguerre: ∫₀^∞ x·exp(−x) dx = 1");
    for &n in &[5, 10, 20] {
        let gl = GaussianRulePlan::new(OrthogonalFamily::Laguerre, n)?;
        let (value, report) = gl.integrate_weighted(&f_x, GaussianSupport::HalfLine(0.0))?;
        let err = (value - expected_laguerre).abs();
        println!("    n={n:<3}  value = {:.12}  |error| = {err:.2e}", value);
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(err < 1e-12);
        assert_eq!(report.evaluations(), n);
    }

    let gl10 = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 10)?;
    let (value, _) = gl10.integrate_weighted(&f_x, GaussianSupport::HalfLine(0.0))?;
    assert!(
        (value - expected_laguerre).abs() < TOL,
        "Gauss-Laguerre should converge for x·exp(-x)"
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Gauss-Jacobi
// ---------------------------------------------------------------------------

fn part3_gauss_jacobi() -> Result<(), SciOxideError> {
    println!("=== Part 3: Gauss-Jacobi ===");
    println!();

    // Gauss-Jacobi computes ∫ f(x)·(1−t)^α·(1+t)^β dt on [−1,1]
    // With f(x) = 1, α = β = 0.5:
    // ∫₋₁¹ (1−t)^0.5 · (1+t)^0.5 dt = ∫₋₁¹ (1−t²)^0.5 dt = π/2
    let f = |_x: f64| -> Result<f64, SciOxideError> { Ok(1.0) };
    let second_moment = |x: f64| -> Result<f64, SciOxideError> { Ok(x * x) };
    let expected = std::f64::consts::PI / 2.0;
    let expected_second_moment = std::f64::consts::PI / 8.0;
    let support = GaussianSupport::Interval(Interval::new(-1.0, 1.0)?);
    let semicircle = OrthogonalFamily::Jacobi {
        alpha: 0.5,
        beta: 0.5,
    };

    println!("  Gauss-Jacobi (α=0.5, β=0.5): ∫₋₁¹ √(1−t²) dt = π/2 ≈ {expected:.10}");
    for &n in &[5, 10, 20] {
        let gj = GaussianRulePlan::new(semicircle, n)?;
        let (value, report) = gj.integrate_weighted(&f, support)?;
        let err = (value - expected).abs();
        println!("    n={n:<3}  value = {:.12}  |error| = {err:.2e}", value);
        let (moment, moment_report) = gj.integrate_weighted(&second_moment, support)?;
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert!(err < 1e-12);
        assert!((moment - expected_second_moment).abs() < 1e-12);
        assert_eq!(report.evaluations(), n);
        assert_eq!(moment_report.evaluations(), n);
    }

    let gj20 = GaussianRulePlan::new(semicircle, 20)?;
    let (value, _) = gj20.integrate_weighted(&f, support)?;
    assert!(
        (value - expected).abs() < TOL,
        "Gauss-Jacobi should converge for constant with (1−t²)^0.5 weight"
    );

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Product rule (2D)
// ---------------------------------------------------------------------------

fn part4_product_rule() -> Result<(), SciOxideError> {
    println!("=== Part 4: Product rule (2D) ===");
    println!();

    // ∫₀¹∫₀¹ xy dxdy = 1/4
    let f = product_xy;
    let expected = 0.25;
    let bounds = [Interval::new(0.0, 1.0)?, Interval::new(0.0, 1.0)?];
    let gl = GaussianRulePlan::gauss_legendre(5)?;
    let pr = ProductRule::new(&gl, scioxide_integrate::ProductOptions::default());
    let res = pr.integrate(&f, &bounds)?;
    let value = measured(&res);
    let err = (value - expected).abs();

    println!("  ∫₀¹∫₀¹ xy dxdy = 1/4");
    println!("  value = {:.12}  evals = {}", value, res.1.evaluations());
    println!("  |actual error| = {err:.2e}");
    assert_eq!(
        res.1.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    assert_eq!(res.1.evaluations(), 25);
    assert!(err < 1e-12, "ProductRule should be exact for xy");

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Cubature and Monte Carlo (3D)
// ---------------------------------------------------------------------------

fn part5_cubature_monte_carlo() -> Result<(), SciOxideError> {
    println!("=== Part 5: Cubature and Monte Carlo (3D) ===");
    println!();

    // 80-digit independent reference: (sqrt(pi) * erf(3))^3.
    let f = gaussian_3d;
    let truncated_gaussian = 5.567_958_983_584_81_f64;
    let bounds = [
        Interval::new(-3.0, 3.0)?,
        Interval::new(-3.0, 3.0)?,
        Interval::new(-3.0, 3.0)?,
    ];

    // Cubature
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-6, 1e-6, NormSelection::Infinity)?)
        .with_max_subdivisions(Budget::try_from(500)?)
        .with_max_evaluations(Some(Budget::try_from(100_000)?));
    let cub = CubaturePlan::new(&bounds, options)?;
    let res_cub = cub.integrate(&f)?;
    let cubature_value = measured(&res_cub);
    let err_cub = (cubature_value - truncated_gaussian).abs();

    println!("  ∫ exp(−r²) on [−3,3]³ ≈ {truncated_gaussian:.10}");
    println!(
        "  Cubature:    value = {:.10}  err_est = {:.2e}  evals = {}  termination = {}",
        cubature_value,
        res_cub.1.error_estimate().unwrap_or(f64::NAN),
        res_cub.1.evaluations(),
        res_cub.1.termination()
    );
    println!("               |actual error| = {err_cub:.2e}");
    assert_eq!(
        res_cub.1.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Subdivision)
    );
    assert!(!res_cub.1.is_success());
    assert_eq!(res_cub.1.evaluations(), 33 + 66 * 500);
    assert!(cubature_value.is_finite());
    assert!(
        res_cub
            .1
            .error_estimate()
            .is_some_and(|error| error.is_finite() && error >= 0.0)
    );
    assert!(
        err_cub < 1e-5,
        "limited Cubature result should retain an accurate full-domain estimate"
    );

    // Monte Carlo on an explicitly selected random stream
    let mc_options = MonteCarloOptions::default()
        .with_samples(Budget::try_from(100_000)?)
        .with_confidence_probability(ConfidenceProbability::new(0.975)?);
    let mc = MonteCarloIntegrator::new(&bounds, mc_options)?;
    let mut workspace = mc.workspace(1, ExecutionContext::sequential_native())?;
    let root = Philox4x64::from_seed(42);
    let mut monte_carlo_value = 0.0;
    let res_mc = mc.integrate_into(
        |x: &[f64]| f(x),
        RandomStreams::new(&root, 0),
        &mut monte_carlo_value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let err_mc = (monte_carlo_value - truncated_gaussian).abs();
    let statistics = workspace.statistics()[0]
        .as_ref()
        .ok_or_else(|| SciOxideError::external("example expected Monte Carlo statistics"))?;

    println!(
        "  Monte Carlo: value = {:.10}  std_err = {:.2e}  evals = {}",
        monte_carlo_value,
        statistics.standard_error(),
        res_mc.evaluations()
    );
    println!("               |actual error| = {err_mc:.2e}");
    assert_eq!(
        res_mc.termination(),
        IntegrationTermination::SampleBudgetCompleted
    );
    assert_eq!(res_mc.evaluations(), 100_000);
    assert_eq!(statistics.samples(), 100_000);
    assert!(statistics.standard_error().is_finite() && statistics.standard_error() > 0.0);
    assert!(
        err_mc <= 4.0 * statistics.standard_error(),
        "Monte Carlo result should agree with the oracle within four standard errors"
    );

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
    println!("scioxide-integrate Weighted & Multidimensional Integration");
    println!("=========================================================");
    println!();

    part1_gauss_chebyshev()?;
    part2_hermite_laguerre()?;
    part3_gauss_jacobi()?;
    part4_product_rule()?;
    part5_cubature_monte_carlo()?;

    println!("All examples completed successfully.");
    Ok(())
}
