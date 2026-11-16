//! Evaluate Mathieu functions once each, then prepare a plan for many points.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{
    MathieuParity, MathieuPlan, RadialKind, SpectralNormalization, SpectralOptions, mathieu_a,
    mathieu_ce, mathieu_mc,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    let a = mathieu_a(2, 1.0_f64)?;
    let ce = mathieu_ce(2, 1.0, Complex::new(0.5, 0.125))?;
    let mc = mathieu_mc(2, 1.0, 0.5_f64, RadialKind::Third)?;
    println!("a_2(1) = {a}; ce_2 = {ce:?}; Mc_2^(3) = {mc:?}");
    let plan = MathieuPlan::new(
        2,
        MathieuParity::Even,
        1.0_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )?;
    for z in [0.0_f64, 0.5, 1.0] {
        println!("ce_2({z}) = {}", plan.angular(z)?.try_value()?);
    }
    println!("{:?}", plan.diagnostics());
    Ok(())
}
