//! Search a rotated quadratic using prepared CMA-ES with one optional restart.
use scioxide_core::{Bounds, Budget, ExecutionContext};
use scioxide_optimize::{
    BoundedObjective, CmaEs, CmaEsOptions, CmaEsRestartPolicy, ObjectiveFn, OptimizeResult,
};

fn main() -> Result<(), scioxide_core::SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        let a = x[0] - 0.25;
        let b = x[1] + 0.4;
        Ok((a + b).powi(2) + 1000. * (a - b).powi(2))
    });
    let bounds = Bounds::new(vec![(Some(-1.), Some(1.)); 2])?;
    let problem = BoundedObjective::new(&objective, &bounds)?;
    let options = CmaEsOptions::default()
        .with_population_size(Budget::try_from(12)?)
        .with_max_generations(Budget::try_from(250)?)
        .with_restart(CmaEsRestartPolicy::IncreasingPopulation { max_restarts: 1 });
    let solver = CmaEs::new(options)?;
    let mut work = solver.workspace(2)?;
    let mut result = OptimizeResult::empty();
    solver.minimize_into(
        &problem,
        &[0.8, 0.8],
        &mut result,
        &mut work,
        ExecutionContext::sequential(),
    )?;
    println!(
        "best={:?}, objective={}, stop={}, generations={}, restarts={}",
        result.x(),
        result.fun(),
        result.termination(),
        work.generations().len(),
        work.restarts(),
    );
    Ok(())
}
