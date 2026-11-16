//! Entire values and explicit Airy scaling on the negative-real cut.
use scioxide_core::Complex;
use scioxide_special::{AiryOptions, airy_ai_scaled_with_options, airy_bi};
pub(crate) fn main() -> Result<(), scioxide_core::SciOxideError> {
    let upper = airy_ai_scaled_with_options(Complex::new(-20.0_f64, 0.0), AiryOptions::default())?;
    let lower = airy_ai_scaled_with_options(Complex::new(-20.0_f64, -0.0), AiryOptions::default())?;
    println!(
        "scaled Ai upper/lower: {:?} {:?}",
        upper.value(),
        lower.value()
    );
    println!("entire Bi: {:?}", airy_bi(Complex::new(0.5_f64, 0.25))?);
    Ok(())
}
