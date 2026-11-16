//! Stored generalized hypergeometric parameters and analytic parameter sensitivity.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{
    HypergeometricDerivative, HypergeometricFunction, HypergeometricOptions, hyperu,
};

pub(crate) fn main() -> Result<(), SciOxideError> {
    let function = HypergeometricFunction::pfq(
        &[0.5_f64, 0.75, 1.25],
        &[1.5, 2.0],
        false,
        HypergeometricOptions::default(),
    )?;
    let result = function.evaluate(0.5, HypergeometricDerivative::Numerator(0))?;
    println!(
        "3F2 = {}; d/da = {:?}",
        result.try_value()?,
        result.derivative()
    );
    let value = hyperu(0.5, 1.25, Complex::new(-2.0_f64, -0.0))?;
    println!("U on the lower cut lip = {value:?}");
    Ok(())
}
