//! Sample a Gaussian constrained to an affine line.
use scioxide_core::{SciOxideError, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_random::{MultivariateNormal, SeedableRng, SplitMix64};

fn main() -> Result<(), SciOxideError> {
    // X = (1, -2) + (Z, 2Z), so covariance has rank one.
    let covariance = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 2.0, 4.0])?;
    let law = MultivariateNormal::new(&[1.0, -2.0], &covariance)?;
    let mut rng = SplitMix64::from_seed(42);
    let sample = law.sample(&mut rng)?;
    let tolerance = Tolerance::new(1.0e-12, 1.0e-12)?;
    println!("rank: {}, sample: {:?}", law.rank(), sample.as_slice());
    println!(
        "on support: {}",
        law.contains_support(sample.as_slice(), tolerance)?
    );
    println!(
        "log pseudo-density: {}",
        law.log_pseudo_density(sample.as_slice(), tolerance)?
    );
    Ok(())
}
