//! Convert, discretize, simulate, and place poles for a low-order LTI model.
use scioxide_core::{Complex, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_signal::{DiscretizationMethod, LtiTime, StateSpace};

fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(2, 2, vec![0.0_f64, 1.0, -2.0, -3.0])?;
    let b = Matrix::from_vec(2, 1, vec![0.0, 1.0])?;
    let c = Matrix::from_vec(1, 2, vec![1.0, 0.0])?;
    let d = Matrix::zeros(1, 1)?;
    let system = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view())?;

    let transfer = system.to_transfer(0, 0)?;
    let discrete = system.discretize(0.1, DiscretizationMethod::ZeroOrderHold)?;
    let times = [0.0, 0.1, 0.2, 0.3, 0.4];
    let response = discrete.step_response(&times)?;
    // An error reports a placement that misses the residual tolerance.
    let placed = system.place_poles(&[Complex::new(-4.0, 0.0), Complex::new(-5.0, 0.0)])?;
    println!(
        "denominator={:?}, final step={}, gain={:?}, pole residual={}",
        transfer.denominator(),
        response.values()[response.values().len() - 1],
        placed.gain().as_slice(),
        placed.maximum_residual(),
    );
    Ok(())
}
