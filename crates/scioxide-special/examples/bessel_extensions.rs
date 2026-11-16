//! Budgeted scalar families and complex values.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{
    BesselOptions, ParabolicCylinderOptions, bessel_i_scaled_with_options, bessel_j_derivative,
    bessel_j_with_options, bessel_k_log, kelvin, legendre_h3d, parabolic_d_with_options,
    riccati_jn, struve, struve_h0_tail_integral,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let options = BesselOptions::default();
    let complex = bessel_j_with_options(2.5, Complex::new(2.0_f64, 1.0), options)?;
    println!(
        "J_2.5(2+i) = {}; work = {}",
        complex.try_value()?,
        complex.counts().iterations()
    );
    println!("J_0'(2) = {}", bessel_j_derivative(0.0_f64, 2.0, 1)?);
    println!("log K_0.5(1000) = {}", bessel_k_log(0.5_f64, 1000.0)?);
    println!(
        "H3d_2 = {}; Riccati j_2 = {}",
        legendre_h3d(2, 1.0_f64, 2.0)?,
        riccati_jn(2, 2.0_f64)?
    );
    println!(
        "Struve H_0(2) = {}; weighted tail = {}",
        struve(0.0_f64, 2.0)?,
        struve_h0_tail_integral(2.0_f64)?
    );
    println!("Kelvin [ber, bei, ker, kei] = {:?}", kelvin(2.0_f64)?);
    let cylinder = parabolic_d_with_options(0.5_f64, 2.0, ParabolicCylinderOptions::default())?;
    println!(
        "D_0.5(2) = {}; derivative = {:?}",
        cylinder.try_value()?,
        cylinder.derivative()
    );
    for order in 0..8 {
        let result = bessel_i_scaled_with_options(order as f64, 20.0, options)?;
        println!("scaled I_{order}(20) = {}", result.try_value()?);
    }
    Ok(())
}
