//! Complex parameters and explicit analytic cut limits, using m=k².
use scioxide_core::Complex;
use scioxide_special::{elliptic_k, elliptic_pi, jacobi_sn};
pub(crate) fn main() -> Result<(), scioxide_core::SciOxideError> {
    let m = Complex::new(0.5_f64, 0.25);
    println!(
        "K: {:?}; Pi: {:?}",
        elliptic_k(m)?,
        elliptic_pi(Complex::new(2.0, 0.5), m)?
    );
    println!("sn: {:?}", jacobi_sn(Complex::new(0.5, 0.25), m)?);
    let upper = elliptic_k(Complex::new(2.0_f64, 0.0))?;
    let lower = elliptic_k(Complex::new(2.0_f64, -0.0))?;
    println!("K cut lips: {upper:?}, {lower:?}");
    Ok(())
}
