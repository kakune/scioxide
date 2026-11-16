//! Stable tails beyond ordinary floating-point probability range.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{
    ProbabilityOptions, faddeeva_with_options, gamma_q_inv_log, normal_inv_log_cdf, normal_log_cdf,
    owens_t, voigt_profile,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let log_p = -1000.0_f64;
    let x = normal_inv_log_cdf(log_p)?;
    println!("normal quantile={x}, log CDF={}", normal_log_cdf(x)?);
    println!("gamma upper quantile={}", gamma_q_inv_log(2.5, log_p)?);
    let r = faddeeva_with_options(Complex::new(1., 0.5), ProbabilityOptions::default())?;
    println!(
        "w={:?}, branch={}, work={}",
        r.try_value()?,
        r.algorithm().method(),
        r.counts().iterations()
    );
    println!(
        "Voigt={}, Owen T={}",
        voigt_profile(1., 0.8, 0.2)?,
        owens_t(1., 0.5)?
    );
    Ok(())
}
