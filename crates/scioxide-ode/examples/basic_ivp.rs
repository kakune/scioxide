//! Solve a scalar initial-value problem with a configured method.
//!
//! Run with `cargo run -p scioxide-ode --example basic_ivp`.

use std::error::Error;

use scioxide_core::{EvaluationRole, SciOxideError, SolveContext};
use scioxide_linalg::Vector;
use scioxide_ode::{
    ClassicalRk4, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeSystem,
};

struct ExponentialDecay;

impl OdeSystem<f64> for ExponentialDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -2.0 * state[0];
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = OdeOptions::new()
        .with_initial_step(0.01)
        .with_output_schedule(OdeOutputSchedule::AcceptedSteps);
    let method = ClassicalRk4::new();
    let problem = OdeProblem::new(&ExponentialDecay, 2.0);
    let solution = method
        .solve(
            &problem,
            &options,
            OdeState::new(0.0, Vector::from_vec(vec![1.0])),
        )
        .map_err(SciOxideError::from)?;
    let report = solution.report();

    println!("termination: {:?}", report.termination());
    println!("final time: {}", report.final_time());
    println!("numerical: {}", solution.state().values()[0]);
    println!("exact: {}", (-4.0_f64).exp());
    println!("emitted states: {}", solution.trajectory().times().len());
    println!(
        "RHS evaluations: {}",
        report.counts().provider(EvaluationRole::RightHandSide)
    );
    Ok(())
}
