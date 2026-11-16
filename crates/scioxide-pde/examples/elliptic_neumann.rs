//! Solve a compatible pure-Neumann problem with an explicit physical mean.
//!
//! `u'' = 0` with outward conormal flux `-1` at `x = 0` and `+1` at `x = 1`
//! determines `u = x + c` up to a constant; the weighted-mean gauge selects
//! the mean 2, so `u = x + 1.5`.
use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, EllipticGauge, EllipticOptions, EllipticProblem, FieldData,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = SegmentMesh::new(Grid1d::new(
        [CoordinateAxis::uniform(0.0, 1.0, 17)?],
        [AxisTopology::Bounded],
    )?)?;
    let side = |side| BoundarySet::axis_side(&mesh, 0, side);
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::natural(side(AxisSide::Start)?, BoundaryData::constant(-1.0)),
            BoundaryCondition::natural(side(AxisSide::End)?, BoundaryData::constant(1.0)),
        ],
    )?;
    let plan = StructuredFdmPlan::with_options(
        mesh,
        &Unit,
        &boundaries,
        SpatialLimits::new(1 << 10, 1 << 14, 1 << 20),
    )?;
    println!("constant mode known: {}", plan.has_constant_modes());
    let options = EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(2.0));
    let source = FieldData::zero();
    let solution = EllipticProblem::new(&plan, &source).solve_with_options(&(options))?;
    let report = solution.report();
    let mut error = 0.0_f64;
    for (node, u) in solution.values().iter().enumerate() {
        error = error.max((u - plan.node_point(node)?[0] - 1.5).abs());
    }
    println!("u(x) = x + 1.5; termination: {:?}", report.termination());
    println!("maximum error: {error:.3e}");
    println!(
        "weighted-mean gauge defect: {:?}",
        report.diagnostics().nullspace_gauge_defect()
    );
    Ok(())
}
