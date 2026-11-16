//! Automatic NDF order/step control with shared tolerances and dense output.
use scioxide_core::{ErrorTolerance, NormSelection, SciOxideError};
use scioxide_linalg::Vector;
use scioxide_ode::{
    Bdf, BdfFormula, MultistepOrder, OdeFn, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem,
    OdeState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = OdeFn::new(
        1,
        |_: f64, y: &[f64], output: &mut [f64]| -> Result<(), SciOxideError> {
            output[0] = -y[0];
            Ok(())
        },
    )?;
    let method =
        Bdf::with_order_policy(MultistepOrder::Automatic(5))?.with_formula(BdfFormula::Ndf);
    let options = OdeOptions::new()
        .with_initial_step(0.1)
        .with_error_tolerance(ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity)?)
        .with_output_schedule(OdeOutputSchedule::times(vec![0.0, 0.2, 0.5, 1.0]));
    let plan = (method, options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&system, 1.0),
            &plan.1,
            OdeState::new(0.0, Vector::from_vec(vec![1.0])),
        )
        .map_err(SciOxideError::from)?;
    assert!(solution.report().is_success());
    let samples = solution.trajectory();
    for (time, state) in samples.times().iter().zip(samples.states()) {
        println!("t={time:.2}, y={:.9}", state.as_slice()[0]);
    }
    Ok(())
}
