//! Complex scalar solvers with independent residual verification.
//!
//! Run: `cargo run --example scalar_complex -p scioxide-rootfind`
use scioxide_core::{Complex, ResidualTolerance, SciOxideError, Tolerance};
use scioxide_rootfind::{
    ComplexMuller, ComplexNewton, ComplexRootSolver, ComplexScalarFn, ComplexScalarFunction,
    ScalarRootOptions,
};

fn main() -> Result<(), SciOxideError> {
    // z² + 1 has no real root; Muller crosses into the complex plane from real starts.
    let function = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)))
        .with_derivative(|z| Ok(z + z));
    let options = ScalarRootOptions::default().with_residual_tolerance(
        ResidualTolerance::from_tolerance(Tolerance::new(1e-12, 0.0)?),
    );
    let muller = ComplexMuller.solve(
        &function,
        (
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ),
        &options,
    )?;
    let newton = ComplexNewton.solve(&function, Complex::new(0.5, 0.5), &options)?;
    for (name, result) in [("ComplexMuller", muller), ("ComplexNewton", newton)] {
        println!(
            "{name}: root = {}, termination = {}",
            result.point(),
            result.termination()
        );
        assert!(result.is_success());
        assert!(function.eval(result.point())?.norm() <= 1e-12);
    }
    Ok(())
}
