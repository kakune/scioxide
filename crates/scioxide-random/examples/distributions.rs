//! Probability distribution demonstration for scioxide-random.
//!
//! Demonstrates continuous, discrete, and multivariate distributions:
//!
//! 1. **Continuous basics** — Normal, Uniform, Exponential: sample, pdf, cdf, quantile, mean, variance
//! 2. **More continuous** — Gamma, Beta, ChiSquared, StudentT, Cauchy, LogNormal
//! 3. **Discrete** — Poisson, Binomial: sample, pmf, cdf
//! 4. **Multivariate normal** — 2D sampling and mean verification
//! 5. **Statistical convergence** — law of large numbers for Normal(0,1)
//!
//! Run: `cargo run --example distributions -p scioxide-random`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::unnecessary_wraps
)]

use scioxide_core::SciOxideError;
use scioxide_linalg::{Matrix, Vector};
use scioxide_random::{
    Beta, Binomial, Cauchy, Cdf, ChiSquared, CountDistribution, Density, Distribution, Exponential,
    Gamma, LogNormal, Mass, Mean, MultivariateNormal, Normal, Poisson, Quantile, SeedableRng,
    SplitMix64, StudentT, Uniform, Variance,
};

/// Tolerance for exact mathematical identities.
const TOL: f64 = 1e-10;

/// Tolerance for statistical convergence tests.
const STAT_TOL: f64 = 0.05;

fn usize_to_f64(value: usize, name: &str) -> Result<f64, SciOxideError> {
    let as_u32 = u32::try_from(value).map_err(|_| {
        SciOxideError::external(format!("{name} is too large for precise f64 conversion"))
    })?;
    Ok(f64::from(as_u32))
}

// ---------------------------------------------------------------------------
// Part 1: Continuous basics — Normal, Uniform, Exponential
// ---------------------------------------------------------------------------

fn part1_continuous_basics() -> Result<(), SciOxideError> {
    println!("=== Part 1: Continuous basics ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);

    // Uniform(0, 1)
    let uniform = Uniform::new(0.0_f64, 1.0)?;
    let u_sample = uniform.sample(&mut rng)?;
    println!("  Uniform(0,1) sample: {u_sample}");
    assert!(
        (0.0..1.0).contains(&u_sample),
        "Uniform sample out of range"
    );

    let u_pdf = uniform.pdf(0.5)?;
    let u_cdf = uniform.cdf(0.5)?;
    let u_q = uniform.quantile(0.75)?;
    let u_mean = uniform.mean()?;
    let u_var = uniform.variance()?;
    println!("  pdf(0.5) = {u_pdf}, cdf(0.5) = {u_cdf}, quantile(0.75) = {u_q}");
    println!("  mean = {u_mean}, variance = {u_var}");
    assert!((u_pdf - 1.0).abs() < TOL, "Uniform(0,1) pdf should be 1");
    assert!(
        (u_cdf - 0.5).abs() < TOL,
        "Uniform(0,1) cdf(0.5) should be 0.5"
    );
    assert!(
        (u_q - 0.75).abs() < TOL,
        "Uniform(0,1) quantile(0.75) should be 0.75"
    );
    assert!(
        (u_mean - 0.5).abs() < TOL,
        "Uniform(0,1) mean should be 0.5"
    );

    // Normal(0, 1)
    let normal = Normal::new(0.0_f64, 1.0)?;
    let n_sample = normal.sample(&mut rng)?;
    println!("  Normal(0,1) sample: {n_sample}");

    let n_pdf = normal.pdf(0.0)?;
    let n_cdf = normal.cdf(0.0)?;
    let n_q = normal.quantile(0.5)?;
    let n_mean = normal.mean()?;
    let n_var = normal.variance()?;
    println!("  pdf(0) = {n_pdf:.6}, cdf(0) = {n_cdf}, quantile(0.5) = {n_q}");
    println!("  mean = {n_mean}, variance = {n_var}");
    let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    assert!(
        (n_pdf - expected_pdf).abs() < TOL,
        "Normal(0,1) pdf(0) = 1/sqrt(2pi)"
    );
    assert!(
        (n_cdf - 0.5).abs() < TOL,
        "Normal(0,1) cdf(0) should be 0.5"
    );
    assert!((n_q).abs() < TOL, "Normal(0,1) quantile(0.5) should be 0");
    assert!((n_mean).abs() < TOL);
    assert!((n_var - 1.0).abs() < TOL);

    // Exponential(1)
    let expo = Exponential::new(1.0_f64)?;
    let e_sample = expo.sample(&mut rng)?;
    println!("  Exponential(1) sample: {e_sample}");
    assert!(e_sample >= 0.0, "Exponential sample must be non-negative");

    let e_pdf = expo.pdf(1.0)?;
    let e_cdf = expo.cdf(1.0)?;
    let e_q = expo.quantile(0.5)?;
    let e_mean = expo.mean()?;
    let e_var = expo.variance()?;
    println!("  pdf(1) = {e_pdf:.6}, cdf(1) = {e_cdf:.6}, quantile(0.5) = {e_q:.6}");
    println!("  mean = {e_mean}, variance = {e_var}");
    let expected_e_pdf = (-1.0_f64).exp();
    assert!((e_pdf - expected_e_pdf).abs() < TOL, "Exp(1) pdf(1) = e^-1");
    let expected_e_cdf = 1.0 - (-1.0_f64).exp();
    assert!(
        (e_cdf - expected_e_cdf).abs() < TOL,
        "Exp(1) cdf(1) = 1-e^-1"
    );
    assert!((e_mean - 1.0).abs() < TOL, "Exp(1) mean = 1");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: More continuous distributions
// ---------------------------------------------------------------------------

fn part2_more_continuous() -> Result<(), SciOxideError> {
    println!("=== Part 2: More continuous distributions ===");
    println!();

    let mut rng = SplitMix64::from_seed(123);

    // Gamma(2, 1) — mean=2, var=2
    let gamma = Gamma::new(2.0_f64, 1.0)?;
    let samples = gamma.sample_n(&mut rng, 10_000)?;
    let sample_mean = samples.iter().copied().sum::<f64>() / samples.len() as f64;
    let g_mean = gamma.mean()?;
    let g_var = gamma.variance()?;
    println!("  Gamma(2,1): mean={g_mean}, var={g_var}, sample_mean={sample_mean:.3}");
    assert!((g_mean - 2.0).abs() < TOL, "Gamma(2,1) mean = 2");
    assert!((g_var - 2.0).abs() < TOL, "Gamma(2,1) var = 2");
    assert!(
        (sample_mean - 2.0).abs() < STAT_TOL,
        "Sample mean should be near 2"
    );

    // Beta(2, 5) — mean=2/7
    let beta = Beta::new(2.0_f64, 5.0)?;
    let b_mean = beta.mean()?;
    let b_var = beta.variance()?;
    println!("  Beta(2,5): mean={b_mean:.6}, var={b_var:.6}");
    assert!((b_mean - 2.0 / 7.0).abs() < TOL, "Beta(2,5) mean = 2/7");

    // ChiSquared(3) — mean=3, var=6
    let chi2 = ChiSquared::new(3.0_f64)?;
    let c_mean = chi2.mean()?;
    let c_var = chi2.variance()?;
    println!("  ChiSquared(3): mean={c_mean}, var={c_var}");
    assert!((c_mean - 3.0).abs() < TOL, "ChiSq(3) mean = 3");
    assert!((c_var - 6.0).abs() < TOL, "ChiSq(3) var = 6");

    // StudentT(5) — mean=0, var=5/3
    let student = StudentT::new(5.0_f64)?;
    let s_mean = student.mean()?;
    let s_var = student.variance()?;
    println!("  StudentT(5): mean={s_mean}, var={s_var:.6}");
    assert!(s_mean.abs() < TOL, "StudentT(5) mean = 0");
    assert!((s_var - 5.0 / 3.0).abs() < TOL, "StudentT(5) var = 5/3");

    // Cauchy(0, 1) — mean and variance are undefined
    let cauchy = Cauchy::new(0.0_f64, 1.0)?;
    let c_sample = cauchy.sample(&mut rng)?;
    println!("  Cauchy(0,1) sample: {c_sample:.6} (mean/variance undefined)");

    // LogNormal(0, 1) — mean=e^0.5, var=(e-1)*e
    let lognorm = LogNormal::new(0.0_f64, 1.0)?;
    let l_mean = lognorm.mean()?;
    let l_var = lognorm.variance()?;
    let expected_l_mean = (0.5_f64).exp();
    let expected_l_var = 1.0_f64.exp_m1() * 1.0_f64.exp();
    println!("  LogNormal(0,1): mean={l_mean:.6}, var={l_var:.6}");
    assert!(
        (l_mean - expected_l_mean).abs() < TOL,
        "LogNormal(0,1) mean = e^0.5"
    );
    assert!(
        (l_var - expected_l_var).abs() < TOL,
        "LogNormal(0,1) var = (e-1)*e"
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Discrete distributions
// ---------------------------------------------------------------------------

fn part3_discrete() -> Result<(), SciOxideError> {
    println!("=== Part 3: Discrete distributions ===");
    println!();

    let mut rng = SplitMix64::from_seed(77);

    // Poisson(5)
    let poisson = Poisson::new(5.0_f64)?;
    let p_sample = poisson.sample(&mut rng)?;
    println!("  Poisson(5) sample: {p_sample}");

    let p_pmf_0 = poisson.pmf(0)?;
    let p_pmf_5 = poisson.pmf(5)?;
    let p_cdf_5 = poisson.cdf(5)?;
    println!("  pmf(0) = {p_pmf_0:.6}, pmf(5) = {p_pmf_5:.6}, cdf(5) = {p_cdf_5:.6}");
    // P(X=0) = e^-5
    let expected_pmf_0 = (-5.0_f64).exp();
    assert!(
        (p_pmf_0 - expected_pmf_0).abs() < TOL,
        "Poisson(5) pmf(0) = e^-5"
    );
    assert!(p_cdf_5 > 0.5, "Poisson(5) cdf(5) should exceed 0.5");
    assert!(p_cdf_5 < 1.0, "Poisson(5) cdf(5) should be less than 1");

    // Binomial(20, 0.5)
    let binom = Binomial::new(20, 0.5_f64)?;
    let b_sample = binom.sample(&mut rng)?;
    println!("  Binomial(20, 0.5) sample: {b_sample}");
    assert!(b_sample <= 20, "Binomial sample must not exceed n");

    let b_pmf_10 = binom.pmf(10)?;
    let b_cdf_10 = binom.cdf(10)?;
    println!("  pmf(10) = {b_pmf_10:.6}, cdf(10) = {b_cdf_10:.6}");
    // C(20,10) * 0.5^20 = 184756 / 1048576 ≈ 0.17620
    assert!(
        b_pmf_10 > 0.17,
        "Binomial(20,0.5) pmf(10) should be around 0.176"
    );
    assert!(
        b_pmf_10 < 0.18,
        "Binomial(20,0.5) pmf(10) should be around 0.176"
    );
    assert!(
        (b_cdf_10 - 0.5).abs() < 0.12,
        "Binomial(20,0.5) cdf(10) should be near 0.5"
    );

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Multivariate normal
// ---------------------------------------------------------------------------

fn part4_multivariate_normal() -> Result<(), SciOxideError> {
    println!("=== Part 4: Multivariate normal ===");
    println!();

    let mut rng = SplitMix64::from_seed(99);

    let mean = Vector::<f64>::from_vec(vec![1.0, 2.0]);
    let cov = Matrix::<f64>::from_vec(2, 2, vec![1.0, 0.5, 0.5, 2.0])?;

    let mvn = MultivariateNormal::new(mean.as_slice(), &cov)?;
    println!("  MultivariateNormal dim = {}", mvn.dimension());
    assert_eq!(mvn.dimension(), 2);

    // Single sample
    let s = mvn.sample(&mut rng)?;
    println!("  Single sample: [{:.4}, {:.4}]", s[0], s[1]);

    // Row-major batch of n samples — compute empirical mean
    let n = 50_000_usize;
    let samples = mvn.sample_n(&mut rng, n)?;
    let mut sum_0 = 0.0_f64;
    let mut sum_1 = 0.0_f64;
    for row in samples.chunks_exact(mvn.sample_width()) {
        sum_0 += row[0];
        sum_1 += row[1];
    }
    let n_f = usize_to_f64(n, "n")?;
    let emp_mean_0 = sum_0 / n_f;
    let emp_mean_1 = sum_1 / n_f;
    println!("  Empirical mean ({n} samples): [{emp_mean_0:.4}, {emp_mean_1:.4}]");
    assert!(
        (emp_mean_0 - 1.0).abs() < STAT_TOL,
        "Empirical mean[0] should be near 1"
    );
    assert!(
        (emp_mean_1 - 2.0).abs() < STAT_TOL,
        "Empirical mean[1] should be near 2"
    );

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Statistical convergence — law of large numbers
// ---------------------------------------------------------------------------

fn part5_convergence() -> Result<(), SciOxideError> {
    println!("=== Part 5: Statistical convergence ===");
    println!();

    let mut rng = SplitMix64::from_seed(12_345);
    let normal = Normal::new(0.0_f64, 1.0)?;

    let n = 100_000;
    let samples = normal.sample_n(&mut rng, n)?;

    let mean = samples.iter().copied().sum::<f64>() / n as f64;
    let variance = samples
        .iter()
        .copied()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n as f64;

    println!("  Normal(0,1) — {n} samples:");
    println!("  Sample mean     = {mean:.6} (expected: 0)");
    println!("  Sample variance = {variance:.6} (expected: 1)");

    assert!(mean.abs() < STAT_TOL, "Sample mean should be near 0");
    assert!(
        (variance - 1.0).abs() < STAT_TOL,
        "Sample variance should be near 1"
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
    println!("scioxide-random Distribution Examples");
    println!("====================================");
    println!();

    part1_continuous_basics()?;
    part2_more_continuous()?;
    part3_discrete()?;
    part4_multivariate_normal()?;
    part5_convergence()?;

    println!("All examples completed successfully.");
    Ok(())
}
