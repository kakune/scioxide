//! Solve `-u'' = pi^2 sin(pi x)` with zero Dirichlet boundaries.
//!
//! The structured finite-difference plan is second order on this uniform axis,
//! so the nodal error is close to `(pi h)^2 / 12`.
//!
//! Run with `cargo run -p scioxide-pde --example elliptic_poisson`.

use std::error::Error;
use std::f64::consts::PI;

use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, EllipticOptions, EllipticProblem, FieldData, FieldPoint,
    SpatialDiscretization, SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients,
    StructuredFdmPlan,
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
        [CoordinateAxis::uniform(0.0, 1.0, 101)?],
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
        SpatialLimits::new(1 << 16, 1 << 20, 1 << 26),
    )?;
    let source =
        FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok(PI * PI * (PI * p.point[0]).sin()));
    let solution =
        EllipticProblem::new(&plan, &source).solve_with_options(&(EllipticOptions::new()))?;
    let report = solution.report();
    let mut error = 0.0_f64;
    for (node, value) in solution.values().iter().enumerate() {
        error = error.max((value - (PI * plan.node_point(node)?[0]).sin()).abs());
    }
    let h = 0.01;
    println!("termination: {}", report.termination());
    println!(
        "maximum error: {error:.6e} (leading term {:.6e})",
        (PI * h).powi(2) / 12.0
    );
    println!("residual: {:?}", report.diagnostics().residual_norm());
    println!("unknowns: {}", plan.unknowns());
    Ok(())
}
