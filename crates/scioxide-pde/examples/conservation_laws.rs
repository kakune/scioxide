//! Sod's tube: conserved cell averages, HLLC/MUSCL and admissibility-checked Heun steps.
use scioxide_core::ExecutionContext;
use scioxide_ode::{Heun, OdeMethod, OdeOptions, OdeProblem, OdeState};
use scioxide_pde::{
    ConservationBoundary, ConservationLimits, ConservationOptions, ConservationReconstruction,
    Hllc, IdealGasEuler, StructuredConservationPlan,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CellField, CoordinateAxis, Grid1d, SegmentMesh,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells = 200;
    let mesh = SegmentMesh::new(Grid1d::new(
        [CoordinateAxis::uniform(0., 1., cells + 1)?],
        [AxisTopology::Bounded],
    )?)?;
    let boundaries = [AxisSide::Start, AxisSide::End]
        .into_iter()
        .map(|side| {
            Ok((
                BoundarySet::axis_side(&mesh, 0, side)?,
                ConservationBoundary::Transmissive,
            ))
        })
        .collect::<Result<Vec<_>, scioxide_core::SciOxideError>>()?;
    let plan = StructuredConservationPlan::<_, 1, 3>::with_options(
        mesh,
        boundaries,
        ConservationOptions::new()
            .with_reconstruction(ConservationReconstruction::MonotonizedCentral)
            .with_limits(ConservationLimits::new(600, 201, 1_000_000)),
    )?;
    let gas = IdealGasEuler::new(1.4)?;
    let left = gas.conserved(1., [0.], 1.)?;
    let right = gas.conserved(0.125, [0.], 0.1)?;
    let initial = CellField::from_fn(plan.mesh(), |i| if i < cells / 2 { left } else { right })?;
    let mut workspace = plan.workspace()?;
    let system = plan.bind(&gas, &Hllc, &mut workspace)?;
    // The system rejects inadmissible stage inputs and accepted candidates itself.
    let ode = (Heun::new(), OdeOptions::new().with_initial_step(0.0002));
    let problem = OdeProblem::new(&system, 0.2);
    let mut state = OdeState::new(0., plan.initial_state(&gas, initial.view())?);
    let report = ode.0.advance(
        &problem,
        &ode.1,
        &mut state,
        &mut (),
        &mut ode.0.workspace(&problem, &ode.1)?,
        ExecutionContext::sequential(),
    )?;
    let field = plan.field(&gas, state.values())?;
    let diagnostics = system.report(
        report.final_time(),
        state.values(),
        ExecutionContext::sequential(),
    )?;
    println!(
        "{:?} at t={}: {} cells, CFL step {:?}, {} limited cells, {} HLLC fallbacks",
        report.termination(),
        report.final_time(),
        field.len(),
        diagnostics.maximum_step(),
        diagnostics.limited_cells(),
        diagnostics.hlle_fallback_faces()
    );
    Ok(())
}
