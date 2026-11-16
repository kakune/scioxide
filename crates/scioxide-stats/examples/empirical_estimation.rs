//! Empirical quantiles, histograms and Gaussian kernel density estimation.
use scioxide_stats::{GaussianKde, histogram, quantile};

fn main() -> Result<(), scioxide_core::SciOxideError> {
    let samples = [0.1_f64, 0.25, 0.5, 0.75, 0.9];
    let bins = histogram(&samples, 4)?;
    let density = GaussianKde::new(&samples)?.evaluate(&[0.25, 0.5, 0.75])?;
    println!("edges={:?}; counts={:?}", bins.edges(), bins.counts());
    println!("median={}; density={density:?}", quantile(&samples, 0.5)?);
    Ok(())
}
