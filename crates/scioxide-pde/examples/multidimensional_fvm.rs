//! Prepare conservative cell-centered finite volumes on two mesh families.
//!
//! The affine field `1 + x + y` solves `-div(K grad u) = 0` for the constant
//! anisotropic tensor `K`, and Dirichlet data reproduce it on every face.
use scioxide_core::SciOxideError;
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryPoint, EllipticOptions, EllipticProblem, FieldData,
    FvmCoefficients, SimplexFvmPlan, SpatialDiscretization, SpatialLimits, StructuredBoundaries,
    StructuredFvmPlan, TaggedBoundaries,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CellField, CellMesh, CoordinateAxis, Point,
    SimplexBoundaryTags, StructuredGrid, TriangleMesh,
};

struct Diffusion;
impl FvmCoefficients<f64, f64, 2> for Diffusion {
    fn diffusion(&self, _: [f64; 2]) -> Result<[[f64; 2]; 2], SciOxideError> {
        Ok([[1.0, 0.25], [0.25, 2.0]])
    }
}

fn affine() -> BoundaryData<f64, f64, 2> {
    BoundaryData::time_invariant(|p: BoundaryPoint<f64, 2>| Ok(1.0 + p.point[0] + p.point[1]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = StructuredGrid::new(
        [
            CoordinateAxis::uniform(0.0, 1.0, 3)?,
            CoordinateAxis::uniform(0.0, 1.0, 3)?,
        ],
        [AxisTopology::Bounded; 2],
    )?;
    let structured = CellMesh::new(grid)?;
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&structured, axis, side)?,
                affine(),
            ));
        }
    }
    let boundaries = StructuredBoundaries::new(&structured, 1, conditions)?;
    let limits = SpatialLimits::new(100, 10_000, 1_000_000);
    let structured_plan =
        StructuredFvmPlan::with_options(structured, &Diffusion, &boundaries, limits)?;
    let source = FieldData::zero();
    let structured_rhs = structured_plan.load(&source, 0.0)?;
    let structured_matrix = structured_plan.assemble_csr()?;
    let solution = EllipticProblem::new(&structured_plan, &source)
        .solve_with_options(&(EllipticOptions::new()))?;
    let mut error = 0.0_f64;
    for (cell, value) in solution.values().iter().enumerate() {
        let [x, y] = structured_plan.dof_point(cell)?.point;
        error = error.max((value - (1.0 + x + y)).abs());
    }

    let simplex = TriangleMesh::new(
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
            .into_iter()
            .map(Point::new)
            .collect::<Result<Vec<_>, _>>()?,
        vec![[0, 1, 2], [0, 2, 3]],
    )?;
    let tags = SimplexBoundaryTags::new(
        &simplex,
        simplex.boundary_faces().map(|face| (face, 1)).collect(),
    )?;
    let simplex_boundaries = TaggedBoundaries::new(
        &simplex,
        &tags,
        1,
        vec![BoundaryCondition::dirichlet(1, affine())],
    )?;
    let simplex_plan =
        SimplexFvmPlan::with_options(simplex.clone(), &Diffusion, &simplex_boundaries, limits)?;
    let simplex_cells = CellField::from_fn(&simplex, |_| 1.5)?;
    let simplex_fluxes = simplex_plan.face_fluxes(simplex_cells.view(), 0.0)?;
    println!(
        "structured {} cells, {} CSR entries, {} loads, affine error {error:.1e}; simplex {} faces",
        structured_plan.unknowns(),
        structured_matrix.nnz(),
        structured_rhs.len(),
        simplex_fluxes.len()
    );
    Ok(())
}
