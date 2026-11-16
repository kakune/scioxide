//! Prepared values, derivatives and weighted normalization.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{EllipsoidalHarmonicPlan, EllipsoidalOptions, EllipsoidalSigns};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let options = EllipsoidalOptions::default()
        .with_relative_tolerance(scioxide_core::RelativeTolerance::new(1e-8_f64)?);
    let harmonic = EllipsoidalHarmonicPlan::new(0.5, 1.0, 4, 1, options)?;
    let e = harmonic.first_kind(1.5, EllipsoidalSigns::default())?;
    let f = harmonic.second_kind(Complex::new(1.5, 0.25))?;
    println!("E={}, derivative={:?}", e.try_value()?, e.derivative());
    println!(
        "F={:?}, normalization={}",
        f.try_value()?,
        harmonic.normalization()?.try_value()?
    );
    Ok(())
}
