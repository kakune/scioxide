//! Nonreal cylindrical order with values and derivatives.
use scioxide_core::Complex;
use scioxide_special::{BesselOptions, bessel_j_with_options, hankel_1_scaled_with_options};
pub(crate) fn main() -> Result<(), scioxide_core::SciOxideError> {
    let nu = Complex::new(0.5_f64, 0.25);
    let z = Complex::new(2.0_f64, 1.0);
    let wave = hankel_1_scaled_with_options(nu, z, BesselOptions::default())?;
    println!(
        "scaled outgoing wave: {:?}; original derivative with same scaling: {:?}",
        wave.value(),
        wave.derivative()
    );
    let values = (0..3)
        .map(|offset| {
            bessel_j_with_options(
                nu + Complex::new(offset as f64, 0.0),
                z,
                BesselOptions::default(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    println!("consecutive complex orders: {values:?}");
    Ok(())
}
