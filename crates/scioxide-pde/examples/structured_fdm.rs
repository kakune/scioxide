//! Prepare a 2D conservative stencil and check the integrated equations of a
//! known solution against its load.
use scioxide_core::{LinearOperator, SciOxideError};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, FieldData, FieldPoint, SpatialDiscretization, SpatialLimits,
    StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan,
};
use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid2d, QuadMesh};

struct Poisson;
impl StructuredFdmCoefficients<f64, f64, 2> for Poisson {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}

fn main() -> Result<(), SciOxideError> {
    let mesh = QuadMesh::new(Grid2d::new(
        [
            CoordinateAxis::uniform(0.0, 1.0, 17)?,
            CoordinateAxis::uniform(0.0, 1.0, 17)?,
        ],
        [AxisTopology::Bounded; 2],
    )?)?;
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&mesh, axis, side)?,
                BoundaryData::zero(),
            ));
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions)?;
    let plan = StructuredFdmPlan::with_options(
        mesh,
        &Poisson,
        &boundaries,
        SpatialLimits::new(1_000, 10_000, 1_000_000),
    )?;
    // u = x(1-x)y(1-y) solves -Δu = 2(x(1-x) + y(1-y)) with zero boundary values.
    let source = FieldData::time_invariant(|p: FieldPoint<f64, 2>| {
        let [x, y] = p.point;
        Ok(2.0 * (x * (1.0 - x) + y * (1.0 - y)))
    });
    let rhs = plan.load(&source, 0.0)?;
    let mut state = Vec::with_capacity(plan.unknowns());
    for node in 0..plan.unknowns() {
        let [x, y] = plan.node_point(node)?;
        state.push(x * (1.0 - x) * y * (1.0 - y));
    }
    let csr = plan.assemble_csr()?;
    let mut actual = vec![0.0; plan.unknowns()];
    let mut workspace = actual.clone();
    plan.apply_into(&state, &mut actual, &mut workspace)?;
    let residual = actual
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "unknowns={}, nonzeros={}, max integrated residual={residual:e}",
        plan.unknowns(),
        csr.nnz()
    );
    Ok(())
}
