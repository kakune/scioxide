//! Scaled special integrals and consecutive orders.
use scioxide_core::{Budget, SciOxideError};
use scioxide_special::{
    IntegralOptions, expint_en_scaled, expint_en_with_options, fresnel_c, fresnel_s,
    logarithmic_integral_offset,
};

pub(crate) fn main() -> Result<(), SciOxideError> {
    println!("exp(1000) E1(1000) = {}", expint_en_scaled(1, 1000.0_f64)?);
    println!(
        "Fresnel C(2), S(2) = {}, {}",
        fresnel_c(2.0_f64)?,
        fresnel_s(2.0_f64)?
    );
    println!(
        "Li(2 + 1e-8) = {}",
        logarithmic_integral_offset(2.0_f64 + 1e-8)?
    );
    let options =
        IntegralOptions::default().with_iteration_budget(Budget::ONE.saturating_add(2047));
    for n in 1..=4 {
        let value = expint_en_with_options(n, 2.0_f64, options)?;
        println!(
            "E{}(2): {:?}, work {}",
            n,
            value.value(),
            value.counts().iterations()
        );
    }
    Ok(())
}
