//! Analytic log-Gamma retains phase turns on explicitly selected cut lips.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{GammaOptions, gamma_with_options, ln_gamma};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let z = Complex::new(-3.5_f64, -0.0);
    println!("lower-lip log-Gamma: {:?}", ln_gamma(z)?);
    let large = gamma_with_options(Complex::new(200.0_f64, 2.0), GammaOptions::default())?;
    println!(
        "large Gamma: {:?}; scaled {:?}",
        large.scalar_representation(),
        large.scaled()
    );
    Ok(())
}
