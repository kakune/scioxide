//! Integrate to an adjacent finite time, and inspect a step-size stop.
use scioxide_core::SciOxideError;
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, OdeFn, OdeMethod, OdeOptions, OdeProblem, OdeState, OdeTermination,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let system = OdeFn::new(1, |_: f64, _: &[f64], derivative: &mut [f64]| {
        derivative.fill(0.0);
        Ok::<(), SciOxideError>(())
    })?;
    let start = 1.0_f64;
    let end = start.next_up();
    let problem = OdeProblem::new(&system, end);
    let initial = || OdeState::new(start, Vector::from_vec(vec![1.0]));
    let plan = (DormandPrince5::new(), OdeOptions::new());
    let solution = plan
        .0
        .solve(&problem, &plan.1, initial())
        .map_err(SciOxideError::from)?;
    assert_eq!(
        solution.report().termination(),
        Some(OdeTermination::ReachedEnd)
    );
    assert_eq!(solution.report().final_time(), end);
    let options = OdeOptions::new().with_maximum_step((end - start) / 4.0);
    let plan = (DormandPrince5::new(), options);
    let solution = plan
        .0
        .solve(&problem, &plan.1, initial())
        .map_err(SciOxideError::from)?;
    let report = solution.report();
    assert_eq!(report.termination(), Some(OdeTermination::StepSizeLimit));
    println!("{:?} at {}", report.termination(), report.final_time());
    Ok(())
}
