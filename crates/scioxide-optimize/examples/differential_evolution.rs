//! Explicit DE boundary/update policies and completed-generation diagnostics.
use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
use scioxide_optimize::{
    BoundedObjective, DifferentialEvolution, DifferentialEvolutionBoundary,
    DifferentialEvolutionOptions, ObjectiveFn, OptimizeResult, UpdatingMode,
};

fn main() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = Bounds::new(vec![(Some(-5.), Some(5.)); 2])?;
    let problem = BoundedObjective::new(&objective, &bounds)?;
    let options = DifferentialEvolutionOptions::default()
        .with_boundary(DifferentialEvolutionBoundary::Reflect)
        .with_updating(UpdatingMode::Deferred);
    let solver = DifferentialEvolution::new(options)?;
    let mut workspace = solver.workspace(2)?;
    let mut output = OptimizeResult::empty();
    solver.minimize_into(
        &problem,
        &[2., 3.],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!(
        "x={:?}, f={}, termination={:?}",
        output.x(),
        output.fun(),
        output.termination()
    );
    if let Some(last) = workspace.generations().last() {
        println!(
            "generation {}: {} accepted, {} rejected, {} repaired coordinates",
            last.generation(),
            last.accepted(),
            last.rejected(),
            last.repaired_coordinates()
        );
    }
    Ok(())
}
