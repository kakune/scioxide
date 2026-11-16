//! Capability composition, normalized histograms, and direct upper-tail inversion.
use scioxide_core::SciOxideError;
use scioxide_random::{
    Cdf, ContinuousLaw, Histogram, InverseSurvival, LogSurvival, Logistic, Mixture, Normal,
    Quantile, Truncated,
};
fn main() -> Result<(), SciOxideError> {
    let normal = Normal::new(0.0_f64, 1.0)?;
    let x = normal.isf(1e-100)?;
    println!(
        "upper-tail quantile {x}, log probability {}",
        normal.log_sf(x)?
    );
    let tail = Truncated::new(&normal, 10.0, 11.0)?;
    println!("conditional median {}", tail.quantile(0.5)?);
    let histogram = Histogram::from_masses(&[0.0, 1.0, 2.0, 3.0], &[1.0, 0.0, 1.0])?;
    println!("gap CDF {}", histogram.cdf(1.5)?);
    let logistic = Logistic::new(0.0, 1.0)?;
    let components: Vec<&dyn ContinuousLaw<f64>> = vec![&normal, &logistic];
    let mixture = Mixture::new(components, &[1.0, 2.0])?;
    println!("mixture upper quantile {}", mixture.isf(1e-20)?);
    Ok(())
}
