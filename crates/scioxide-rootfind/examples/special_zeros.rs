//! Indexed roots, resource reports and supplied-zero Hankel composition.
use scioxide_core::Tolerance;
use scioxide_rootfind::{
    SpecialZeroBranch, SpecialZeroFamily, SpecialZeroOptions, special_zeros,
    special_zeros_with_options,
};
fn main() -> Result<(), scioxide_core::SciOxideError> {
    let zeros = special_zeros(SpecialZeroFamily::BesselJ(0.0_f64), 9)?;
    let plan = scioxide_fft::DiscreteHankelPlan::new(
        &zeros,
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-11, 1e-11)?)
            .with_limits(scioxide_fft::HankelLimits::default()),
    )?;
    println!("{} roots; {} radial nodes", zeros.len(), plan.nodes().len());
    let complex = special_zeros_with_options(
        SpecialZeroFamily::Erf,
        SpecialZeroBranch::UPPER,
        1,
        4,
        &SpecialZeroOptions::<f64>::default(),
    )?;
    println!(
        "{:?} including nested refinement",
        complex.report().accounting().total()?
    );
    for zero in complex.roots() {
        println!(
            "erf zero {} = {:?}; estimated coordinate error {}",
            zero.index(),
            zero.point(),
            zero.estimated_error()
        );
    }
    Ok(())
}
