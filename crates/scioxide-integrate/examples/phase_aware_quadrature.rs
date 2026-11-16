//! Explicit nonlinear phase, stationary neighborhoods and complex exponential output.
use scioxide_core::{Complex, ErrorTolerance, Interval, NormSelection, SciOxideError};
use scioxide_integrate::{
    FourierWeight, OscillatoryGeometry, OscillatoryMethod, OscillatoryPhase,
    OscillatoryPhaseInverse, OscillatoryQuadraturePlan, OscillatoryRegion, SubdivisionOptions,
};
/// `g(x) = x²`, evaluated with its derivative in one call and inverted on `[0, 1]`.
struct Square;
impl OscillatoryPhase<f64> for Square {
    fn evaluate(&self, x: f64) -> Result<(f64, f64), SciOxideError> {
        Ok((x * x, 2.0 * x))
    }
    fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<f64>> {
        Some(self)
    }
}
impl OscillatoryPhaseInverse<f64> for Square {
    fn invert(&self, _: Interval<f64>, phase: f64) -> Result<f64, SciOxideError> {
        Ok(phase.sqrt())
    }
}
fn main() -> Result<(), SciOxideError> {
    let options = SubdivisionOptions::default().with_tolerance(ErrorTolerance::scalar(
        2e-9,
        0.0,
        NormSelection::Infinity,
    )?);
    let domain = Interval::new(0.0, 1.0)?;
    let regions = [
        OscillatoryRegion::stationary(Interval::new(0.0, 0.25)?, Interval::new(0.0, 0.5)?)?,
        OscillatoryRegion::nonstationary(Interval::new(0.25, 0.5)?, Interval::new(0.5, 1.0)?)?,
        OscillatoryRegion::nonstationary(Interval::new(0.5, 1.0)?, Interval::new(1.0, 2.0)?)?,
    ];
    let phase = OscillatoryGeometry::new(domain, 1000.0, &regions)?;
    let cosine = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        phase.clone(),
        FourierWeight::Cosine,
        scioxide_integrate::OscillatoryOptions::new(options.clone()),
    )?
    .integrate(|_| Ok(1.0), &Square)?;
    let sine = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        phase,
        FourierWeight::Sine,
        scioxide_integrate::OscillatoryOptions::new(options),
    )?
    .integrate(|_| Ok(1.0), &Square)?;
    println!("Levin cosine: {cosine:?}\nFilon sine: {sine:?}");
    println!(
        "integral exp(i*1000*x²) dx = {:?}",
        Complex::new(cosine.0, sine.0)
    );
    assert!(cosine.1.is_success() && sine.1.is_success());
    Ok(())
}
