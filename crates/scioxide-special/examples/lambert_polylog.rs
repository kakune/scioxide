//! Explicit sheets, complex orders and caller-budgeted continuation.
use scioxide_core::{Budget, Complex, SciOxideError};
use scioxide_special::{
    LambertOptions, LerchOptions, lambert_w_complex_with_options, lerch_phi_with_options, spence,
    wright_omega,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let w =
        lambert_w_complex_with_options(Complex::new(-0.1_f64, 0.0), -1, LambertOptions::default())?;
    println!(
        "W-1(-0.1) = {:?}; derivative = {:?}",
        w.try_value()?,
        w.derivative()
    );
    let options = LerchOptions::default().with_iteration_budget(Budget::try_from(10000)?);
    let phi = lerch_phi_with_options(
        Complex::new(2., 1.),
        Complex::new(2., 0.5),
        Complex::new(1., 0.25),
        options,
    )?;
    println!(
        "Phi = {:?}; work = {}; status = {:?}",
        phi.value(),
        phi.counts().iterations(),
        phi.termination()
    );
    println!(
        "omega(1000) = {}; Spence(0.5) = {}",
        wright_omega(1000.)?,
        spence(0.5)?
    );
    Ok(())
}
