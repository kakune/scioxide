//! Inspect shared mesh, boundary, plan and option contracts.
//!
//! Natural data is the outward conormal flux `k du/dn` and Robin data is
//! `k du/dn + alpha u = g`; periodicity belongs to mesh topology.
//!
//! Run with `cargo run -p scioxide-pde --example grid_and_options`.

use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, EllipticOptions, SpatialDiscretization, SpatialLimits,
    StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan,
};
use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};

struct Unit;
impl StructuredFdmCoefficients<f64, f64, 1> for Unit {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}

fn mesh(topology: AxisTopology) -> Result<SegmentMesh<f64>, SciOxideError> {
    SegmentMesh::new(Grid1d::new(
        [CoordinateAxis::uniform(0.0, 1.0, 11)?],
        [topology],
    )?)
}

fn main() -> Result<(), SciOxideError> {
    let grid = mesh(AxisTopology::Bounded)?;
    let axis = grid.grid().axis(0)?;
    println!(
        "nodes: {:?}",
        axis.coordinates().collect::<Result<Vec<_>, _>>()?
    );
    println!("spacing: {:?}", axis.coordinate(1)? - axis.start());

    // u = 1 at x = 0 and du/dn + u/2 = 3/2 at x = 1.
    let boundaries = StructuredBoundaries::new(
        &grid,
        1,
        vec![
            BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&grid, 0, AxisSide::Start)?,
                BoundaryData::constant(1.0),
            ),
            BoundaryCondition::robin(
                BoundarySet::axis_side(&grid, 0, AxisSide::End)?,
                0.5,
                BoundaryData::constant(1.5),
            ),
        ],
    )?;
    for condition in boundaries.conditions() {
        println!(
            "{:?} on {} face(s), {:?}",
            condition.kind(),
            condition.region().len(),
            condition.data().time_dependence()
        );
    }

    let plan = StructuredFdmPlan::with_options(
        grid,
        &Unit,
        &boundaries,
        SpatialLimits::new(64, 1024, 1 << 20),
    )?;
    println!("unknowns: {}", plan.unknowns());
    println!(
        "constrained: {:?}",
        plan.constrained()
            .iter()
            .enumerate()
            .filter_map(|(i, &fixed)| fixed.then_some(i))
            .collect::<Vec<_>>()
    );
    println!("lumped mass: {:?}", plan.lumped_mass());
    println!("constant modes: {}", plan.has_constant_modes());

    // A periodic mesh has no exterior faces, so it takes no conditions.
    let periodic = mesh(AxisTopology::Periodic)?;
    let seamless = StructuredBoundaries::<f64, f64, 1>::new(&periodic, 1, Vec::new())?;
    let ring = StructuredFdmPlan::with_options(
        periodic,
        &Unit,
        &seamless,
        SpatialLimits::new(64, 1024, 1 << 20),
    )?;
    println!(
        "periodic unknowns: {}, constant modes: {}",
        ring.unknowns(),
        ring.has_constant_modes()
    );

    let options = EllipticOptions::<f64>::new();
    println!(
        "linear iteration budget: {}",
        options.max_linear_iterations()
    );
    Ok(())
}
