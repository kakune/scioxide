//! Locate a terminal event with dense output from an adaptive solver.
//!
//! Run with `cargo run -p scioxide-ode --example events_advanced`.

use std::error::Error;

use scioxide_core::{Budget, SciOxideError, SolveContext};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, EventDirection, EventFunction, EventSearchPolicy, OdeEventOptions, OdeMethod,
    OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeSystem,
};

struct UnitVelocity;

impl OdeSystem<f64> for UnitVelocity {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = 1.0;
        Ok(())
    }
}

struct ReachesThree;

impl EventFunction<f64> for ReachesThree {
    fn eval(&self, _time: f64, state: &[f64]) -> Result<f64, SciOxideError> {
        Ok(state[0] - 3.0)
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Rising
    }

    fn is_terminal(&self) -> bool {
        true
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = OdeOptions::new()
        .with_event_options(
            OdeEventOptions::new().with_search(EventSearchPolicy::Subdivide(Budget::try_from(4)?)),
        )
        .with_output_schedule(OdeOutputSchedule::times(vec![0.0, 1.0, 1.0, 2.0, 4.0]));
    let plan = (DormandPrince5::new(), options);
    let problem = OdeProblem::new(&UnitVelocity, 10.0).with_events(&[&ReachesThree]);
    let solution = plan
        .0
        .solve(
            &problem,
            &plan.1,
            OdeState::new(0.0, Vector::from_vec(vec![0.0])),
        )
        .map_err(SciOxideError::from)?;
    let (report, trajectory) = (solution.report(), solution.trajectory());

    println!("termination: {:?}", report.termination());
    println!("qualified search: {:?}", report.event_search());
    println!(
        "requested samples (duplicates preserved): {:?}",
        trajectory.times()
    );
    for event in trajectory.events() {
        println!(
            "event {} at t={:.12}, state={:.12}",
            event.index(),
            event.time(),
            event.state()[0]
        );
    }
    println!("streamed states: {}", trajectory.times().len());
    Ok(())
}
