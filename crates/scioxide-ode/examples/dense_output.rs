//! Requested-time samples from the quartic continuous extension of one DP5 step.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, OdeFn, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState,
};
fn main() -> Result<(), SciOxideError> {
    let decay = OdeFn::new(1, |_: f64, y: &[f64], output: &mut [f64]| {
        output[0] = -y[0];
        Ok(())
    })?;
    // A single accepted step of 0.1; every sample interpolates inside it.
    let options = OdeOptions::new()
        .with_initial_step(0.1)
        .with_output_schedule(OdeOutputSchedule::times(vec![0.025, 0.05, 0.075]));
    let plan = (DormandPrince5::new(), options);
    let problem = OdeProblem::new(&decay, 0.1);
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let report = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut |t: f64, y: &[f64]| {
            println!("t={t}: y={}, exact={}", y[0], (-t).exp());
            Ok(())
        },
        &mut plan.0.workspace(&problem, &plan.1)?,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "{} accepted step(s), y(0.1)={}",
        report.accepted_steps(),
        state.values()[0]
    );
    Ok(())
}
