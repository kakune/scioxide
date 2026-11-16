//! Native holomorphic complex dynamics and an explicit nonholomorphic real block.
use scioxide_core::{Complex, ErrorTolerance, NormSelection, SciOxideError};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{GaussRk, OdeFn, OdeMethod, OdeOptions, OdeProblem, OdeState};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rotation = OdeFn::new(1, |_: f64, y: &[Complex<f64>], out: &mut [Complex<f64>]| {
        out[0] = Complex::new(0., 1.) * y[0];
        Ok(())
    })?
    .with_jacobian(
        |_: f64, _: &[Complex<f64>], out: &mut MatrixViewMut<'_, Complex<f64>>| {
            out.set(0, 0, Complex::new(0., 1.))
        },
    );
    let options = OdeOptions::new()
        .with_initial_step(0.01)
        .with_error_tolerance(ErrorTolerance::scalar(
            1e-12,
            1e-12,
            NormSelection::Infinity,
        )?);
    let native = (GaussRk::<f64, Complex<f64>>::new(), options.clone());
    let solution = native
        .0
        .solve(
            &OdeProblem::new(&rotation, 1.),
            &native.1,
            OdeState::new(0., Vector::from_vec(vec![Complex::new(1., 0.)])),
        )
        .map_err(SciOxideError::from)?;
    println!("native z'=i z: {:?}", solution.state().values());
    // z'=conj(z) is not holomorphic. Pack [Re(z), Im(z)], so x'=x, y'=-y.
    let conjugation = OdeFn::new(
        2,
        |_: f64, y: &[f64], out: &mut [f64]| -> Result<(), SciOxideError> {
            out[0] = y[0];
            out[1] = -y[1];
            Ok(())
        },
    )?
    .with_jacobian(|_: f64, _: &[f64], out: &mut MatrixViewMut<'_, f64>| {
        out.set(0, 0, 1.)?;
        out.set(0, 1, 0.)?;
        out.set(1, 0, 0.)?;
        out.set(1, 1, -1.)
    });
    let real = (GaussRk::new(), options);
    let solution = real
        .0
        .solve(
            &OdeProblem::new(&conjugation, 1.),
            &real.1,
            OdeState::new(0., Vector::from_vec(vec![1., 1.])),
        )
        .map_err(SciOxideError::from)?;
    println!(
        "real-block z'=conj(z): {:?}; exact [e, 1/e]",
        solution.state().values()
    );
    Ok(())
}
