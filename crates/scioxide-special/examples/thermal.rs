//! Thermal-integral representations and work diagnostics.
use scioxide_core::SciOxideError;
use scioxide_special::{
    DebyeOptions, FermiDiracOptions, debye_3_with_options, fermi_dirac_half_with_options,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let heat_capacity_factor = debye_3_with_options(4.0_f64, DebyeOptions::default())?;
    let degenerate_gas = fermi_dirac_half_with_options(100.0_f64, FermiDiracOptions::default())?;
    for (name, result) in [
        ("Debye D3", heat_capacity_factor),
        ("normalized F1/2", degenerate_gas),
    ] {
        println!(
            "{name}: value={:?}, log={:?}, work={}, termination={:?}",
            result.value(),
            result.logarithmic(),
            result.counts().iterations(),
            result.termination()
        );
    }
    Ok(())
}
