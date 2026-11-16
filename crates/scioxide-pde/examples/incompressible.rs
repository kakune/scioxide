//! Conservative mixed flow with caller-prepared Schwarz/pressure preconditioning.
use scioxide_core::{Budget, ResidualOperator, ResidualTolerance, SciOxideError, StepTolerance};
use scioxide_linalg::{LdlDecomposition, Matrix};
use scioxide_pde::{NavierStokesPlan, SimplexFemGeometry, StokesPlan};
use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver};
use scioxide_sparse::{SchwarzKind, SchwarzLocalSolveView, SchwarzPlan, SchwarzSubdomainView};
#[path = "support/simplex_fem.rs"]
mod meshes;
#[path = "support/incompressible.rs"]
mod problem;

fn main() -> Result<(), SciOxideError> {
    let mesh = meshes::triangle_grid::<f64>(4)?;
    let boundary = problem::boundaries(&mesh, 0.04)?;
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &boundary,
        &|_| Ok([0.3, 0.]),
        problem::options(),
    )?;
    let matrix = p.assemble_csr()?;
    let velocity = p.velocity_unknowns();
    let pressure = p.mesh().vertex_count();
    // Geometry-derived overlap belongs to the PDE consumer, not Schwarz.
    let mut left = Vec::new();
    let mut right = Vec::new();
    for i in 0..velocity {
        let x = p.node_point(i % p.velocity_nodes())?[0];
        if x <= 0.625 {
            left.push(i);
        }
        if x >= 0.375 {
            right.push(i);
        }
    }
    let local = |indices: &[usize]| -> Result<LdlDecomposition<f64>, SciOxideError> {
        let n = indices.len();
        let mut a = Matrix::zeros(n, n)?;
        for (i, &row) in indices.iter().enumerate() {
            for (j, &col) in indices.iter().enumerate() {
                a.as_mut_slice()[i * n + j] = matrix.get(row, col)?;
            }
        }
        LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact)
    };
    let a = local(&left)?;
    let b = local(&right)?;
    let ai = a.inverse_operator();
    let bi = b.inverse_operator();
    let lw = left
        .iter()
        .map(|i| {
            if right.binary_search(i).is_ok() {
                0.5
            } else {
                1.
            }
        })
        .collect::<Vec<_>>();
    let rw = right
        .iter()
        .map(|i| {
            if left.binary_search(i).is_ok() {
                0.5
            } else {
                1.
            }
        })
        .collect::<Vec<_>>();
    let mut pm = Matrix::zeros(pressure, pressure)?;
    for cell in 0..p.mesh().cell_count() {
        let vertices = p.mesh().cell_vertices(cell)?;
        let mass = p.element_pressure_mass(cell)?;
        for (i, &row) in vertices.iter().enumerate() {
            for (j, &col) in vertices.iter().enumerate() {
                pm.as_mut_slice()[row * pressure + col] += mass[i * 3 + j];
            }
        }
    }
    let pf = LdlDecomposition::factor(&pm, scioxide_linalg::HermitianInput::Exact)?;
    let pi = pf.inverse_operator();
    let pressure_map = (velocity..velocity + pressure).collect::<Vec<_>>();
    let pw = vec![1.; pressure];
    let gauge_map = (velocity + pressure..p.unknowns()).collect::<Vec<_>>();
    let gw = vec![1.; gauge_map.len()];
    let gauge = Matrix::<f64>::identity(gauge_map.len())?;
    let domains = [
        SchwarzSubdomainView {
            indices: &left,
            weights: Some(&lw),
            solve: SchwarzLocalSolveView::InverseOperator(&ai),
        },
        SchwarzSubdomainView {
            indices: &right,
            weights: Some(&rw),
            solve: SchwarzLocalSolveView::InverseOperator(&bi),
        },
        SchwarzSubdomainView {
            indices: &pressure_map,
            weights: Some(&pw),
            solve: SchwarzLocalSolveView::InverseOperator(&pi),
        },
        SchwarzSubdomainView {
            indices: &gauge_map,
            weights: Some(&gw),
            solve: SchwarzLocalSolveView::InverseOperator(&gauge.view().try_finite()?),
        },
    ];
    let preconditioner = SchwarzPlan::new(p.unknowns(), SchwarzKind::Additive, &domains, None)?;
    let p = NavierStokesPlan::new(p);
    let options = NewtonKrylovOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(1e-11, 1e-11)?)
        .with_step_tolerance(StepTolerance::new(1e-12, 1e-12)?)
        .with_krylov(Budget::try_from(1000)?, Budget::try_from(80)?);
    let result = NewtonKrylovSolver::new(options).and_then(|root_plan| {
        root_plan.solve(
            &NewtonKrylovProblem::with_jacobian(&p).with_preconditioner(&preconditioner),
            &vec![0.; p.unknowns()],
        )
    })?;
    let mut residual = vec![0.; p.unknowns()];
    // The residual scratch also covers the Stokes operator scratch of `diagnostics`.
    let mut scratch = vec![0.; p.residual_workspace_len()?];
    p.residual_into(result.point(), &mut residual, &mut scratch)?;
    let norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    let diagnostics = p
        .stokes()
        .diagnostics(result.point(), 1., 0.001, &mut scratch)?;
    println!(
        "{:?}; {} Newton iterations; residual {norm:e}; {:?}",
        result.termination(),
        result.iterations(),
        diagnostics
    );
    if !result.is_success() || norm > 1e-9 {
        return Err(SciOxideError::external("equilibrium not verified"));
    }
    Ok(())
}
