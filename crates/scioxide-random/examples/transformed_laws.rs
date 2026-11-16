//! Canonical transformed-law constructions; run with
//! `cargo run -p scioxide-random --example transformed_laws`.
use scioxide_core::SciOxideError;
use scioxide_random::{
    Cdf, DoubleParetoLogNormal, GeneralizedHalfLogistic, LogSurvival, Normal, PowerNormal,
    Quantile, Transformed, Truncated,
};

fn main() -> Result<(), SciOxideError> {
    let folded = Transformed::abs(Normal::new(1.0_f64, 1.0)?)?;
    let truncated = Truncated::new(Normal::new(0.0_f64, 1.0)?, 8.0, 9.0)?;
    let powered = PowerNormal::new(2.0_f64)?;
    let bounded = GeneralizedHalfLogistic::new(0.5_f64)?;
    let heavy = DoubleParetoLogNormal::new(0.0_f64, 1.0, 3.0, 2.0)?;
    println!("folded normal CDF(1): {}", folded.cdf(1.0)?);
    println!("truncated normal median: {}", truncated.quantile(0.5)?);
    println!("power normal log SF(2): {}", powered.log_sf(2.0)?);
    println!(
        "generalized half logistic median: {}",
        bounded.quantile(0.5)?
    );
    println!("double Pareto lognormal CDF(1): {}", heavy.cdf(1.0)?);
    Ok(())
}
