//! Integrate a one-dimensional wave equation and stream accepted time steps.
//!
//! The plan's diffusion is the squared wave speed; `u = cos(pi t) sin(pi x)`.
//!
//! Run with `cargo run -p scioxide-pde --example hyperbolic_wave`.

use std::error::Error;
use std::f64::consts::PI;

use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, FieldData, FieldPoint, HyperbolicOptions, HyperbolicProblem,
    PdeOutputSchedule, SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients,
    StructuredFdmPlan,
};
use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};

/// Unit wave speed, supplied as `c^2`.
struct UnitSpeed;
impl StructuredFdmCoefficients<f64, f64, 1> for UnitSpeed {
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
        &UnitSpeed,
        &boundaries,
        SpatialLimits::new(1 << 10, 1 << 14, 1 << 20),
    )?;
    let displacement =
        FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok((PI * p.point[0]).sin()));
    let source = FieldData::zero();
    let options = HyperbolicOptions::new()
        .with_initial_step(0.001)
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let solution = HyperbolicProblem::new(&plan, &source, 0.5)?
        .solve_with_options(&displacement, &FieldData::zero(), &(options))
        .map_err(SciOxideError::from)?;
    let report = solution.report();

    println!("termination: {}", report.termination());
    println!(
        "accepted steps: {}",
        report.diagnostics().num_accepted_steps()
    );
    println!("emitted states: {}", solution.trajectory().times().len());
    println!(
        "final displacement midpoint: {:.3e} (exact 0)",
        solution.values()[40]
    );
    Ok(())
}
