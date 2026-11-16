//! Recover oriented fluxes from a cell field and inspect conservation.
//!
//! `u = x + 1` solves `-u'' = 0` with the Robin law `du/dn + (2/3) u = -1/3`
//! at `x = 0` (natural data is the outward conormal flux) and `u = 2` at
//! `x = 1`, so every face flux `-du/dx` is `-1`.
use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, EllipticOptions, EllipticProblem, FieldData, FvmCoefficients,
    SpatialLimits, StructuredBoundaries, StructuredFvmPlan,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, Cell, CoordinateAxis, FieldView, Grid1d, SegmentMesh,
};

struct Unit;
impl FvmCoefficients<f64, f64, 1> for Unit {
    fn diffusion(&self, _: [f64; 1]) -> Result<[[f64; 1]; 1], SciOxideError> {
        Ok([[1.0]])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = SegmentMesh::new(Grid1d::new(
        [CoordinateAxis::explicit(vec![0.0, 0.125, 0.5, 1.0])?],
        [AxisTopology::Bounded],
    )?)?;
    let side = |side| BoundarySet::axis_side(&mesh, 0, side);
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::robin(
                side(AxisSide::Start)?,
                2.0 / 3.0,
                BoundaryData::constant(-1.0 / 3.0),
            ),
            BoundaryCondition::dirichlet(side(AxisSide::End)?, BoundaryData::constant(2.0)),
        ],
    )?;
    let plan = StructuredFvmPlan::with_options(
        mesh.clone(),
        &Unit,
        &boundaries,
        SpatialLimits::new(64, 4096, 1 << 20),
    )?;
    let source = FieldData::zero();
    let solution =
        EllipticProblem::new(&plan, &source).solve_with_options(&(EllipticOptions::new()))?;
    if !solution.report().is_success() {
        return Err("elliptic solve did not converge".into());
    }
    let cells = FieldView::<_, Cell, 1>::new(&mesh, solution.values())?;
    let flux = plan.face_fluxes::<0>(cells, 0.0)?;
    println!("Cell averages: {:?}", solution.values());
    println!(
        "Fluxes toward increasing x (exact: -1): {:?}",
        flux.values()
    );
    for (cell, pair) in flux.values().windows(2).enumerate() {
        println!("Cell {cell} integrated balance: {}", pair[1] - pair[0]);
    }
    Ok(())
}
