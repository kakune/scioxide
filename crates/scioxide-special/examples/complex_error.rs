//! Entire complex error functions and retained logarithmic/scaled tails.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{ErrorFunctionOptions, dawson_with_options, erfc_with_options, erfcx};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let z = Complex::new(2.0_f64, 1.0);
    println!("erfcx({z:?}) = {:?}", erfcx(z)?);
    let tail = erfc_with_options(Complex::new(30.0, 0.0), ErrorFunctionOptions::default())?;
    println!(
        "ordinary {:?}; binary {:?}; logarithmic {:?}",
        tail.value(),
        tail.scaled(),
        tail.logarithmic()
    );
    let scaled = dawson_with_options(Complex::new(20.0, 0.125), ErrorFunctionOptions::default())?;
    println!(
        "scaled erfi {:?}; derivative {:?}",
        scaled.try_value()?,
        scaled.derivative()
    );
    Ok(())
}
