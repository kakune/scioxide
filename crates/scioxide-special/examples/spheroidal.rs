//! Evaluate oblate spheroidal functions once each, then prepare a plan for many points.
use scioxide_core::{Complex, SciOxideError};
use scioxide_special::{
    RadialKind, SpectralNormalization, SpectralOptions, SpheroidalKind, SpheroidalPlan,
    spheroidal_angular, spheroidal_characteristic_value, spheroidal_radial,
};

pub(crate) fn main() -> Result<(), SciOxideError> {
    let kind = SpheroidalKind::Oblate;
    let cv = spheroidal_characteristic_value(kind, 1, 2, 2.0_f64)?;
    let angular = spheroidal_angular(kind, 1, 2, 2.0, Complex::new(0.25, 0.125))?;
    let outgoing = spheroidal_radial(kind, 1, 2, 2.0, 0.5_f64, RadialKind::Third)?;
    println!("cv = {cv}; angular = {angular:?}; outgoing = {outgoing:?}");
    let plan = SpheroidalPlan::new(
        kind,
        1,
        2,
        2.0_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )?;
    for x in [-0.5_f64, 0.0, 0.5] {
        println!("S_12({x}) = {}", plan.angular(x)?.try_value()?);
    }
    println!("{:?}", plan.diagnostics());
    Ok(())
}
