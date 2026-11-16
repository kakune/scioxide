//! Prepared weighted amplitudes and explicitly parameterized contours.
use scioxide_core::{Budget, Complex, ExecutionContext, Interval, SciOxideError};
use scioxide_integrate::*;
fn main() -> Result<(), SciOxideError> {
    let options = SubdivisionOptions::<f64>::default();
    let interval = Interval::new(0.0, 1.0)?;
    let endpoint = WeightedQuadraturePlan::endpoint(
        interval,
        &[],
        EndpointWeight::new(-0.5, 0.0, true, false)?,
        scioxide_integrate::WeightedQuadratureOptions::new(options.clone()),
    )?;
    println!(
        "Integral log(x)/sqrt(x), expected -4: {:?}",
        endpoint.integrate(|_: f64| Ok(1.0))?
    );
    let cpv = WeightedQuadraturePlan::principal_value(
        interval,
        0.3,
        scioxide_integrate::WeightedQuadratureOptions::new(options.clone()),
    )?;
    println!(
        "PV integral x/(x-0.3): {:?}",
        cpv.integrate(|x: f64| Ok(x))?
    );
    let fourier = FourierTailPlan::new(
        FourierDomain::Upper { lower: 0.0 },
        1.0,
        FourierWeight::Sine,
        scioxide_integrate::FourierTailOptions::new(options.clone())
            .with_max_cycles(Budget::try_from(128)?)
            .with_moment_levels(Budget::try_from(16)?),
    )?;
    println!(
        "Integral sin(x)/(1+x): {:?}",
        fourier.integrate(|x: f64| Ok(1.0 / (1.0 + x)))?
    );
    let contour = ContourPlan::new(
        Interval::new(0.0, 2.0 * core::f64::consts::PI)?,
        &[],
        options,
    )?;
    let mut workspace = contour.workspace()?;
    let mut output = Complex::new(0.0, 0.0);
    let report = contour.integrate_into(
        |z: Complex<f64>| Ok(z.conj()),
        |t: f64| {
            Ok((
                Complex::new(t.cos(), t.sin()),
                Complex::new(-t.sin(), t.cos()),
            ))
        },
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("Unit-circle integral conjugate(z) dz, expected 2*pi*i: {output:?}; {report:?}");
    Ok(())
}
