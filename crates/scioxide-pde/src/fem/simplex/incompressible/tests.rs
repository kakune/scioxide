#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use alloc::boxed::Box;
use alloc::vec;
use scioxide_core::{Complex, JacobianOperator, ResidualOperator};
use scioxide_linalg::{LdlDecomposition, Matrix, Vector};
use scioxide_ode::OdeSystem;
#[path = "analytic_checks.rs"]
mod analytic_checks;
#[path = "../../../../examples/support/incompressible.rs"]
mod problem;
#[path = "time_tests.rs"]
mod time_tests;

fn positive_viscous_forms<const D: usize>(input: SimplexMesh<f64, D>) {
    for curved in [false, true] {
        let (warped, edges) = super::super::meshes::curved(&input).unwrap();
        let mesh = if curved { warped } else { input.clone() };
        let boundaries = problem::boundaries(&mesh, 0.).unwrap();
        let plan = StokesPlan::with_options(
            mesh,
            if curved {
                SimplexFemGeometry::Quadratic(edges.view())
            } else {
                SimplexFemGeometry::Affine
            },
            1.,
            1.,
            &boundaries,
            &|_| Ok([0.; D]),
            problem::options(),
        )
        .unwrap();
        assert!(
            plan.samples
                .volume
                .iter()
                .chain(&plan.samples.facets)
                .all(|s| s.weight > 0.)
        );
        let local = plan.local();
        let l = plan.space.order.nodes(D);
        let velocity = D * l;
        for (cell, matrix) in plan.matrices.chunks_exact(local * local).enumerate() {
            let mut block = vec![0.; velocity * velocity];
            for (row, entries) in block.chunks_exact_mut(velocity).enumerate() {
                entries.copy_from_slice(&matrix[row * local..row * local + velocity]);
            }
            super::super::tests::assert_psd(&block, velocity);
            super::super::tests::assert_psd(&plan.masses[cell * l * l..(cell + 1) * l * l], l);
        }
    }
}

#[test]
fn positive_quadrature_preserves_stokes_viscous_and_mass_blocks() {
    positive_viscous_forms(super::super::meshes::triangle_grid::<f64>(1).unwrap());
    positive_viscous_forms(super::super::meshes::tetrahedron_grid::<f64>(1).unwrap());
}

fn dense_solve<V: NumericScalar>(a: &CsrMatrix<V>, rhs: &[V]) -> Vec<V> {
    let mut matrix = Matrix::zeros(a.rows(), a.cols()).unwrap();
    a.to_dense_into(matrix.as_mut_slice()).unwrap();
    LdlDecomposition::factor(&matrix, scioxide_linalg::HermitianInput::Exact)
        .unwrap()
        .solve(&Vector::from_vec(rhs.to_vec()))
        .unwrap()
        .as_slice()
        .to_vec()
}
fn patch<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: SimplexMesh<R, D>,
    geometry: SimplexFemGeometry<'_, R, D>,
    amplitude: V,
    pressure: V,
    tol: R,
) {
    let b = problem::boundaries(&mesh, amplitude).unwrap();
    let p = StokesPlan::with_options(
        mesh,
        geometry,
        R::ONE,
        R::ONE,
        &b,
        &|_| {
            Ok(core::array::from_fn(|a| {
                if a == 0 { pressure } else { V::ZERO }
            }))
        },
        problem::options(),
    )
    .unwrap();
    let x = problem::state(&p, amplitude, pressure).unwrap();
    let mut y = vec![V::ZERO; p.unknowns()];
    let mut work = vec![V::ZERO; LinearOperator::apply_workspace_len(&p).unwrap()];
    p.apply_into(&x, &mut y, &mut work).unwrap();
    let defect = y
        .iter()
        .zip(p.rhs())
        .map(|(&a, &b)| (a - b).abs())
        .fold(R::ZERO, |a, b| a.max(b));
    assert!(defect < tol, "patch defect {defect:?}");
    let solved = dense_solve(&p.assemble_csr().unwrap(), p.rhs());
    let error = solved
        .iter()
        .zip(&x)
        .map(|(&a, &b)| (a - b).abs())
        .fold(R::ZERO, |a, b| a.max(b));
    assert!(error < tol.scalbn(5), "solve error {error:?}");
    let report = p
        .diagnostics(
            &solved,
            R::ONE,
            R::try_from_f64_lossy(0.01).unwrap(),
            &mut work,
        )
        .unwrap();
    assert!(report.weak_divergence < tol);
    assert!(report.divergence_l2 < tol.scalbn(5));
    assert!(report.boundary_flux < tol);
    assert!(report.gauge_defect < tol);
    let fields = p.fields(&x).unwrap();
    let vertices = fields
        .velocity()
        .vertices()
        .iter()
        .map(|v| v.view())
        .collect::<Vec<_>>();
    let edges = fields
        .velocity()
        .edges()
        .iter()
        .map(|v| v.view())
        .collect::<Vec<_>>();
    p.pack_fields_into(
        IncompressibleFieldView::new(
            SimplexFemFieldView::new(&vertices, &edges),
            fields.pressure().view(),
        ),
        &mut y,
    )
    .unwrap();
    assert_eq!(x, y);
}
#[test]
fn stokes_shear_pressure_all_scalars_and_dimensions() {
    patch(
        super::super::meshes::triangle_grid::<f64>(2).unwrap(),
        SimplexFemGeometry::Affine,
        0.04,
        0.3,
        1e-10,
    );
    patch(
        super::super::meshes::tetrahedron_grid::<f64>(2).unwrap(),
        SimplexFemGeometry::Affine,
        0.04,
        0.3,
        1e-9,
    );
    patch::<f32, f32, 2>(
        super::super::meshes::triangle_grid::<f32>(2).unwrap(),
        SimplexFemGeometry::Affine,
        0.04,
        0.3,
        2e-5,
    );
    patch::<f32, f32, 3>(
        super::super::meshes::tetrahedron_grid::<f32>(2).unwrap(),
        SimplexFemGeometry::Affine,
        0.04,
        0.3,
        5e-5,
    );
    patch(
        super::super::meshes::triangle_grid::<f64>(2).unwrap(),
        SimplexFemGeometry::Affine,
        Complex::new(0.04, 0.02),
        Complex::new(0.3, -0.1),
        1e-10,
    );
    patch(
        super::super::meshes::tetrahedron_grid::<f64>(2).unwrap(),
        SimplexFemGeometry::Affine,
        Complex::new(0.04, 0.02),
        Complex::new(0.3, -0.1),
        1e-9,
    );
    patch::<f32, Complex<f32>, 2>(
        super::super::meshes::triangle_grid::<f32>(2).unwrap(),
        SimplexFemGeometry::Affine,
        Complex::new(0.04, 0.02),
        Complex::new(0.3, -0.1),
        2e-5,
    );
    patch::<f32, Complex<f32>, 3>(
        super::super::meshes::tetrahedron_grid::<f32>(2).unwrap(),
        SimplexFemGeometry::Affine,
        Complex::new(0.04, 0.02),
        Complex::new(0.3, -0.1),
        5e-5,
    );
}
fn plan() -> StokesPlan<f64, f64, 2> {
    let mesh = super::super::meshes::triangle_grid(2).unwrap();
    let b = problem::boundaries(&mesh, 0.04).unwrap();
    StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &b,
        &|_| Ok([0.3, 0.]),
        problem::options(),
    )
    .unwrap()
}
#[test]
fn conservative_nonlinear_residual_tangent_adjoint_and_shear_solve() {
    let p = NavierStokesPlan::new(plan());
    let exact = problem::state(p.stokes(), 0.04, 0.3).unwrap();
    let n = p.unknowns();
    let mut work = vec![0.; ResidualOperator::residual_workspace_len(&p).unwrap()];
    let mut r = vec![0.; n];
    p.residual_into(&exact, &mut r, &mut work).unwrap();
    assert!(r.iter().all(|v| v.abs() < 1e-11));
    let state = exact
        .iter()
        .enumerate()
        .map(|(i, &v)| v + 0.02 * (i as f64).sin())
        .collect::<Vec<_>>();
    let d = (0..n).map(|i| (i as f64 + 0.3).cos()).collect::<Vec<_>>();
    let mut j = vec![0.; n];
    p.jvp_into(&state, &d, &mut j, &mut work).unwrap();
    let h = 1e-6;
    let plus = state
        .iter()
        .zip(&d)
        .map(|(&x, &d)| x + h * d)
        .collect::<Vec<_>>();
    let minus = state
        .iter()
        .zip(&d)
        .map(|(&x, &d)| x - h * d)
        .collect::<Vec<_>>();
    let mut rm = vec![0.; n];
    p.residual_into(&plus, &mut r, &mut work).unwrap();
    p.residual_into(&minus, &mut rm, &mut work).unwrap();
    assert!(
        r.iter()
            .zip(&rm)
            .zip(&j)
            .all(|((&a, &b), &j)| ((a - b) / (2. * h) - j).abs() < 1e-8)
    );
    let csr = p.assemble_tangent_csr(&state).unwrap();
    let mut cw = vec![0.; csr.apply_workspace_len().unwrap()];
    csr.apply_into(&d, &mut r, &mut cw).unwrap();
    assert!(r.iter().zip(&j).all(|(&a, &b)| (a - b).abs() < 1e-11));
    crate::PdeTangent::new(&p, &state)
        .unwrap()
        .apply_adjoint_into(&state, &mut r, &mut work)
        .unwrap();
    let lhs = state.iter().zip(&j).map(|(&a, &b)| a * b).sum::<f64>();
    let rhs = d.iter().zip(&r).map(|(&a, &b)| a * b).sum::<f64>();
    assert!((lhs - rhs).abs() < 1e-10);
    let options = scioxide_rootfind::NewtonKrylovOptions::default()
        .with_residual_tolerance(scioxide_core::ResidualTolerance::new(1e-10, 1e-10).unwrap())
        .with_step_tolerance(scioxide_core::StepTolerance::new(1e-12, 1e-12).unwrap());
    let solved = scioxide_rootfind::NewtonKrylovSolver::new(options)
        .and_then(|root_plan| {
            root_plan.solve(
                &scioxide_rootfind::NewtonKrylovProblem::with_jacobian(&p),
                &vec![0.; n],
            )
        })
        .unwrap();
    assert!(solved.is_success(), "{:?}", solved.termination());
    assert!(
        solved
            .point()
            .iter()
            .zip(&exact)
            .all(|(&a, &b)| (a - b).abs() < 1e-7)
    );
}
#[test]
fn constrained_time_recovers_stationary_pressure_and_projects_explicitly() {
    let p = NavierStokesPlan::new(plan());
    let exact = problem::state(p.stokes(), 0.04, 0.3).unwrap();
    let solver = scioxide_linalg::GmresSolver::new(
        scioxide_core::ResidualTolerance::new(1e-12, 1e-12).unwrap(),
        Budget::try_from(500).unwrap(),
    )
    .with_restart(Budget::try_from(p.unknowns()).unwrap());
    let mut workspace =
        IncompressibleTimeWorkspace::new(p.stokes(), &IncompressibleSolveOptions::new())
            .unwrap()
            .with_linear_solver(Box::new(solver));
    let system = IncompressibleSystem::navier_stokes(&p, &mut workspace, None, 0.01).unwrap();
    let y = system
        .free_velocity_indices()
        .iter()
        .map(|&i| exact[i])
        .collect::<Vec<_>>();
    let mut out = vec![0.; system.dimension()];
    system.validate_velocity(&y).unwrap();
    ephemeral(|ctx| system.rhs(0., &y, &mut out, ctx)).unwrap();
    assert!(out.iter().all(|v| v.abs() < 1e-8));
    let mut recovered = vec![0.; p.unknowns()];
    ephemeral(|ctx| system.recover_into(0., &y, &mut recovered, ctx)).unwrap();
    assert!(
        recovered
            .iter()
            .zip(&exact)
            .all(|(&a, &b)| (a - b).abs() < 1e-8)
    );
    let noisy = y
        .iter()
        .enumerate()
        .map(|(i, &v)| v + 0.01 * (i as f64).cos())
        .collect::<Vec<_>>();
    assert!(system.validate_velocity(&noisy).is_err());
    ephemeral(|ctx| system.project_velocity_into(&noisy, &mut out, ctx)).unwrap();
    system.validate_velocity(&out).unwrap();
}

mod preconditioner_tests;

mod warm_starts;

#[test]
fn automatic_closed_gauges_and_owned_solve_match_configured_forms() {
    let mesh = super::super::meshes::triangle_grid::<f64>(2).unwrap();
    let boundaries = problem::boundaries(&mesh, 0.04).unwrap();
    let auto = StokesPlan::new(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &boundaries,
        &|_| Ok([0.3, 0.]),
    )
    .unwrap();
    let configured = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &boundaries,
        &|_| Ok([0.3, 0.]),
        IncompressibleOptions::new().with_gauges(vec![EllipticGauge::WeightedMean(0.)]),
    )
    .unwrap();
    assert_eq!(
        auto.assemble_csr().unwrap().values(),
        configured.assemble_csr().unwrap().values()
    );
    assert_eq!(auto.rhs(), configured.rhs());
    let result = auto
        .solve_with_options(&IncompressibleSolveOptions::default())
        .unwrap();
    assert_eq!(
        auto.solve().unwrap(),
        result.into_converged_solution().unwrap().into_vec()
    );
}

#[test]
fn owned_nonlinear_flow_maps_unsuccessful_termination() {
    use super::solve::require_nonlinear_success;
    use scioxide_core::{
        BudgetKind, ConvergenceReason, EvaluationCounts, ExecutionStopReason, SciOxideErrorView,
    };
    use scioxide_rootfind::RootTermination;
    let counts = EvaluationCounts::from_iterations(3);
    assert!(require_nonlinear_success(RootTermination::ResidualTolerance, counts).is_ok());
    for (stop, reason) in [
        (
            RootTermination::IterationLimit,
            ConvergenceReason::IterationLimit,
        ),
        (
            RootTermination::WorkLimit(BudgetKind::Evaluation),
            ConvergenceReason::ResourceLimit(BudgetKind::Evaluation),
        ),
        (RootTermination::Stagnation, ConvergenceReason::Stagnation),
        (
            RootTermination::LinearFailure(scioxide_linalg::LinearSolveTermination::IterationLimit),
            ConvergenceReason::NestedSolver,
        ),
        (
            RootTermination::GlobalizationFailure,
            ConvergenceReason::NoSuccessfulCandidate,
        ),
    ] {
        let error = require_nonlinear_success(stop, counts).unwrap_err();
        let SciOxideErrorView::ConvergenceFailure {
            reason: actual,
            diagnostics,
            ..
        } = error.view()
        else {
            unreachable!()
        };
        assert_eq!(actual, reason);
        assert_eq!(diagnostics.counts().own(), counts);
    }
    for (stop, reason) in [
        (RootTermination::Cancelled, ExecutionStopReason::Cancelled),
        (
            RootTermination::TimeLimit,
            ExecutionStopReason::DeadlineReached,
        ),
    ] {
        assert_eq!(
            require_nonlinear_success(stop, counts).unwrap_err(),
            SciOxideError::execution_stopped(reason)
        );
    }
}
