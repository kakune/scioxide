//! Mixed linear LDL solve and matrix-free finite-strain Newton-Krylov solve.
use scioxide_core::{ResidualOperator, ResidualTolerance, SciOxideError, StepTolerance};
use scioxide_linalg::{LdlDecomposition, Matrix, Vector};
use scioxide_pde::{
    HyperelasticityPlan, IsotropicElasticMaterial, LinearElasticityPlan, SimplexFemGeometry,
};
use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver};
#[path = "support/simplex_fem.rs"]
mod meshes;
#[path = "support/elasticity.rs"]
mod problem;

fn main() -> Result<(), SciOxideError> {
    let mesh = meshes::triangle_grid::<f64>(2)?;
    let boundary = problem::boundaries(&mesh)?;
    let linear = LinearElasticityPlan::new(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(1., 100_000.)?,
        &boundary,
        &|_| Ok([0.; 2]),
        problem::options::<f64, 2>(true),
    )?;
    // This small example explicitly selects dense LDL; the plan never densifies.
    let csr = linear.assemble_csr()?;
    let mut dense = Matrix::zeros(linear.unknowns(), linear.unknowns())?;
    csr.to_dense_into(dense.as_mut_slice())?;
    let factor = LdlDecomposition::factor(&dense, scioxide_linalg::HermitianInput::Exact)?;
    let x = factor.solve(&Vector::from_vec(linear.rhs().to_vec()))?;
    let stress = linear.stress(0, [0.2, 0.2], x.as_slice())?;
    println!(
        "Linear mixed: {} unknowns, inertia {:?}, shear stress {:.8}",
        linear.unknowns(),
        factor.inertia(scioxide_linalg::RankPolicy::Exact)?,
        stress.cauchy()[0][1]
    );
    let nonlinear = HyperelasticityPlan::new(
        mesh,
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(1., 100.)?,
        &boundary,
        &|_| Ok([0.; 2]),
        problem::options::<f64, 2>(true),
    )?;
    let settings = NewtonKrylovOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(1e-10, 1e-10)?)
        .with_step_tolerance(StepTolerance::new(1e-13, 1e-13)?);
    let result = NewtonKrylovSolver::new(settings).and_then(|root_plan| {
        root_plan.solve(
            &NewtonKrylovProblem::with_jacobian(&nonlinear),
            &vec![0.; nonlinear.unknowns()],
        )
    })?;
    let mut residual = vec![0.; nonlinear.unknowns()];
    let mut work = vec![0.; nonlinear.residual_workspace_len()?];
    nonlinear.residual_into(result.point(), &mut residual, &mut work)?;
    let defect = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    println!(
        "Neo-Hookean: {:?}, {} Newton steps, independently evaluated residual {:.3e}, energy {:?}",
        result.termination(),
        result.iterations(),
        defect,
        nonlinear.energy(result.point())?
    );
    if !result.is_success() || defect > 1e-8 {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::NestedSolver,
            scioxide_core::ConvergenceDiagnostics::new(),
            "nonlinear equilibrium was not verified",
        ));
    }
    let fields = nonlinear.fields(result.point())?;
    println!(
        "Displacement components {}, pressure vertices {}",
        fields.displacement().vertices().len(),
        fields.pressure().map_or(0, |p| p.len())
    );
    Ok(())
}
