//! Method-of-steps solution of y'(t)=y(t-1), y(t)=1 for t<=0.
use scioxide_core::{Budget, ExecutionContext, SciOxideError};
use scioxide_linalg::{MatrixView, Vector};
use scioxide_ode::{DdeOptions, DdePlan, DdeProblem, DdeWorkspaceRequest, OdeState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let history = |_: f64, output: &mut [f64]| {
        output.fill(1.0);
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], delayed: MatrixView<'_, f64>, output: &mut [f64]| {
        output[0] = delayed.get(0, 0)?;
        Ok::<(), SciOxideError>(())
    };
    let options = DdeOptions::new()
        .with_history_capacity(Budget::try_from(128)?)
        .with_schedule_capacity(Budget::try_from(64)?);
    let plan = DdePlan::new(1, (0.0, 3.0), &[1.0], &[], options)?;
    // The state is overwritten with the initial history at the span start.
    let mut state = OdeState::new(0.0, Vector::zeros(1)?);
    let report = plan.advance(
        &DdeProblem::new(&rhs, &history),
        &mut state,
        &mut |t: f64, y: &[f64]| {
            println!("y({t}) = {}", y[0]);
            Ok(())
        },
        &mut plan.workspace(DdeWorkspaceRequest::new(0))?,
        ExecutionContext::sequential_native(),
    )?;
    println!("{:?}; exact y(3) = 37/6", report.termination());
    Ok(())
}
