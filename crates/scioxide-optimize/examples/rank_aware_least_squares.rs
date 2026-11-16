//! Robust and complex fits using explicit least-squares policies.
use scioxide_core::{Complex, SciOxideError};
use scioxide_optimize::{
    ComplexLeastSquares, LeastSquares, LeastSquaresFn, LeastSquaresGlobalization, LeastSquaresLoss,
    LeastSquaresOptions,
};
fn main() -> Result<(), SciOxideError> {
    let problem = LeastSquaresFn::new(1, 5, |x: &[f64], output| {
        output.copy_from_slice(&[x[0] - 1.0, x[0] - 1.0, x[0] - 1.0, x[0] - 1.0, x[0] - 100.0]);
        Ok(())
    })?;
    let options = LeastSquaresOptions::default()
        .with_condition_estimation(scioxide_linalg::Estimation::Estimate)
        .with_loss(LeastSquaresLoss::Huber, 1.0);
    let result = LeastSquares::new(options)?.solve(&problem, &[0.0])?;
    println!(
        "Huber location: {:?}, cost {}, termination {:?}",
        result.x(),
        result.cost()?,
        result.termination()
    );

    let mut parameters = [Complex::new(0.0, 0.0)];
    let mut residual = [Complex::new(0.0, 0.0)];
    let conjugate_fit = ComplexLeastSquares::new(
        |x: &[Complex<f64>], r: &mut [Complex<f64>]| {
            r[0] = Complex::new(x[0].re - 2.0, -x[0].im - 3.0);
            Ok(())
        },
        &mut parameters,
        &mut residual,
    )?;
    let gauss_newton = LeastSquares::new(
        LeastSquaresOptions::default()
            .with_condition_estimation(scioxide_linalg::Estimation::Estimate)
            .with_globalization(LeastSquaresGlobalization::gauss_newton()),
    )?;
    let result = gauss_newton.solve(&conjugate_fit, &[0.0, 0.0])?;
    println!(
        "Conjugate fit [real, imaginary]: {:?}; rank {:?}",
        result.x(),
        result.rank()
    );
    Ok(())
}
