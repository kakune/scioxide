//! Scalar spherical value and derivative evaluation.
use scioxide_core::SciOxideError;
use scioxide_special::{
    SphericalBesselOptions, spherical_in_scaled_with_options, spherical_jn_with_options,
};

pub(crate) fn main() -> Result<(), SciOxideError> {
    let options = SphericalBesselOptions::default();
    for order in 0..8 {
        let result = spherical_jn_with_options(order, 2.0_f64, options)?;
        println!(
            "j_{order}(2) = {}, derivative = {:?}",
            result.try_value()?,
            result.derivative()
        );
    }
    let scaled = spherical_in_scaled_with_options(20, 1000.0_f64, options)?;
    println!("exp(-1000) i_20(1000) = {}", scaled.try_value()?);
    Ok(())
}
