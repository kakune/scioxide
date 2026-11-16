//! Curved quadratic triangle forms, shared sparse solve and typed output fields.
use scioxide_core::{Budget, ResidualTolerance, SciOxideError};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{CgSolver, Vector};
use scioxide_pde::{
    SimplexFemGeometry, SimplexFemOptions, SimplexFemOrder, SimplexFemPlan, SpatialDiscretization,
};
#[path = "support/simplex_fem_problem.rs"]
mod problem;
#[path = "support/simplex_fem.rs"]
mod support;

fn main() -> Result<(), SciOxideError> {
    let base = support::triangle_grid::<f64>(3)?;
    let (mesh, edge_coordinates) = support::curved(&base)?;
    let boundaries = problem::affine_boundaries(&mesh, 2)?;
    let options = SimplexFemOptions::new().with_limits(problem::limits());
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Quadratic(edge_coordinates.view()),
        SimplexFemOrder::P2,
        &problem::Affine(2),
        &boundaries,
        options,
    )?;
    // Symmetric Dirichlet elimination keeps the coupled operator SPD.
    let matrix = plan.assemble_csr()?;
    let rhs = Vector::from_vec(plan.load(&problem::affine_source::<2>(2), 0.0)?);
    let solver = CgSolver::new(
        ResidualTolerance::new(1e-12, 1e-12)?,
        Budget::try_from(1000)?,
    );
    let mut workspace = scioxide_linalg::IterativeWorkspace::new();
    let mut solution = Vector::zeros(plan.unknowns())?;
    let report = solver.solve_into(
        &matrix,
        &rhs,
        Preconditioning::None,
        solution.view_mut(),
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    let fields = plan.fields(solution.as_slice())?;
    println!(
        "{:?}: {} unknowns, {} nonzeros; {} vertex fields and {} edge fields",
        report.termination(),
        plan.unknowns(),
        matrix.nnz(),
        fields.vertices().len(),
        fields.edges().len()
    );
    let mut value = [0.; 2];
    let mut gradient = [[0.; 2]; 2];
    plan.evaluate_into(
        0,
        [0.2, 0.3],
        solution.as_slice(),
        &mut value,
        &mut gradient,
    )?;
    println!("Interior values {value:?}, physical gradients {gradient:?}");
    Ok(())
}
