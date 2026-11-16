//! Compare adaptive methods using role-specific error tolerances.
//!
//! Run with `cargo run -p scioxide-ode --example adaptive_solvers`.

use std::error::Error;

use scioxide_core::{
    ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, SciOxideError, SolveContext,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, DormandPrince853, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem,
    OdeState, OdeSystem, Verner65,
};

struct Oscillator;

impl OdeSystem<f64> for Oscillator {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = state[1];
        derivative[1] = -state[0];
        Ok(())
    }
}

fn run<M>(name: &str, method: M) -> Result<(), Box<dyn Error>>
where
    M: OdeMethod<f64> + Clone,
{
    // The driver alone scales raw local errors, including distinct component atols.
    let tolerance =
        ErrorTolerance::try_vector(&[1e-10, 1e-9], 1e-8, NormSelection::RootMeanSquare)?;
    // Requested samples exercise the method's dense output; DOP853 uses its
    // seventh-order extension and charges all four additional RHS evaluations.
    let options = OdeOptions::new()
        .with_error_tolerance(tolerance)
        .with_output_schedule(OdeOutputSchedule::times(
            (1..=80).map(|i| f64::from(i) * 0.25).collect(),
        ));
    let plan = (method, options);
    let problem = OdeProblem::new(&Oscillator, 20.0);
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0, 0.0]));
    let report = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut (),
        &mut plan.0.workspace(&problem, &plan.1)?,
        ExecutionContext::sequential_native(),
    )?;
    let energy = state
        .values()
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    println!(
        "{name:>20}: steps={}, evaluations={}, squared radius={energy:.12}",
        report.accepted_steps(),
        report.counts().provider(EvaluationRole::RightHandSide)
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run("Dormand-Prince 5", DormandPrince5::new())?;
    run("Dormand-Prince 8(5,3)", DormandPrince853::new())?;
    run("Verner 6(5)", Verner65::new())?;
    Ok(())
}
