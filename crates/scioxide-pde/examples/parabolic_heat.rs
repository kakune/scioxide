//! Solve the one-dimensional heat equation with caller-selected output,
//! stopping on a step budget and resuming from the committed checkpoint.
//!
//! Run with `cargo run -p scioxide-pde --example parabolic_heat`.

use std::error::Error;
use std::f64::consts::PI;

use scioxide_core::{Budget, ErrorTolerance, ExecutionContext, NormSelection, SciOxideError};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, FieldData, FieldPoint, ParabolicOptions, ParabolicProblem,
    ParabolicScheme, PdeCheckpoint, PdeObservation, PdeOutputSchedule, SpatialDiscretization,
    SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan,
};
use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};

struct Unit;
impl StructuredFdmCoefficients<f64, f64, 1> for Unit {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mesh = SegmentMesh::new(Grid1d::new(
        [CoordinateAxis::uniform(0.0, 1.0, 81)?],
        [AxisTopology::Bounded],
    )?)?;
    let side = |side| BoundarySet::axis_side(&mesh, 0, side);
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::dirichlet(side(AxisSide::Start)?, BoundaryData::zero()),
            BoundaryCondition::dirichlet(side(AxisSide::End)?, BoundaryData::zero()),
        ],
    )?;
    let plan = StructuredFdmPlan::with_options(
        mesh,
        &Unit,
        &boundaries,
        SpatialLimits::new(1 << 10, 1 << 14, 1 << 20),
    )?;
    let source = FieldData::zero();
    let initial = FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok((PI * p.point[0]).sin()));
    let problem = ParabolicProblem::new(&plan, &source, 0.1)?;
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.002)
        .with_error_tolerance(ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity)?)
        .with_output_schedule(PdeOutputSchedule::Times(vec![0.0, 0.025, 0.025, 0.1]));
    let mut emitted = 0;
    let mut last = Vec::new();
    let mut observe = |sample: PdeObservation<'_, f64, f64>| {
        assert_eq!(sample.values().len(), plan.unknowns());
        println!(
            "request {:?} at t={}",
            sample.request_index(),
            sample.time()
        );
        emitted += 1;
        last = sample.values().to_vec();
        Ok(())
    };
    let budgeted = options.clone().with_step_budget(Budget::try_from(1)?);
    let mut state = problem.initial_state(&initial, &budgeted)?;
    let mut workspace = problem.workspace()?;
    let first = problem.advance(
        &budgeted,
        &mut state,
        &mut observe,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("budgeted termination: {}", first.termination());
    let checkpoint = PdeCheckpoint::new(&state)?;
    println!("checkpoint time: {}", checkpoint.time());
    let report = problem.resume(
        &(options),
        &checkpoint,
        &mut state,
        &mut observe,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;

    println!("termination: {}", report.termination());
    println!(
        "accepted steps: {}",
        report.diagnostics().num_accepted_steps()
    );
    println!("emitted requested states: {emitted}");
    println!(
        "final midpoint: {} (exact {})",
        last[40],
        (-PI * PI * 0.1).exp()
    );
    Ok(())
}
