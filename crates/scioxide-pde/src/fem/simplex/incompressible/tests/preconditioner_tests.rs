use super::*;
use scioxide_core::{Preconditioner, ProblemToken, SolveBudgets, SolveIdentity, SolveState};

fn scalar_solve<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: SimplexMesh<R, D>,
    amplitude: V,
) {
    let pressure = amplitude + V::ONE;
    let boundary = problem::boundaries(&mesh, amplitude).unwrap();
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        R::ONE,
        R::ONE,
        &boundary,
        &|_| {
            Ok(core::array::from_fn(|a| {
                if a == 0 { pressure } else { V::ZERO }
            }))
        },
        problem::options(),
    )
    .unwrap();
    let defaults = p
        .solve_with_options(&IncompressibleSolveOptions::default())
        .unwrap();
    assert!(defaults.is_success(), "{:?}", defaults.report());
    // Forward pressure errors amplify residual error. Check the default residual
    // contract, then request a precision-derived tighter tolerance for field accuracy.
    let requested = IncompressibleSolveOptions::<R>::default().residual();
    assert!(
        defaults.report().diagnostic_residual_norm().unwrap()
            <= requested
                .threshold(p.rhs().iter().fold(R::ZERO, |s, v| s.hypot(v.abs())))
                .unwrap()
    );
    let tighter = R::EPSILON.scalbn(5);
    let result = p
        .solve_with_options(
            &IncompressibleSolveOptions::default()
                .with_residual(scioxide_core::ResidualTolerance::new(tighter, tighter).unwrap()),
        )
        .unwrap();
    assert!(result.is_success(), "{:?}", result.report());
    let exact = problem::state(&p, amplitude, pressure).unwrap();
    let error = result
        .solution()
        .as_slice()
        .iter()
        .zip(&exact)
        .map(|(&a, &b)| (a - b).abs().into_f64())
        .fold(0., f64::max);
    let tolerance = if R::EPSILON.into_f64() > 1e-10 {
        0.03
    } else {
        2e-5
    };
    assert!(error < tolerance, "scalar solve error={error}");
}
#[test]
fn library_defaults_support_all_scalars_and_injected_nonlinear_corrections() {
    use scioxide_core::Complex;
    scalar_solve(
        crate::fem::simplex::meshes::triangle_grid::<f32>(2).unwrap(),
        0.04f32,
    );
    scalar_solve(
        crate::fem::simplex::meshes::triangle_grid::<f64>(2).unwrap(),
        0.04f64,
    );
    scalar_solve(
        crate::fem::simplex::meshes::tetrahedron_grid::<f32>(2).unwrap(),
        Complex::new(0.04f32, 0.02),
    );
    scalar_solve(
        crate::fem::simplex::meshes::tetrahedron_grid::<f64>(2).unwrap(),
        Complex::new(0.04f64, 0.02),
    );
    let p = NavierStokesPlan::new(plan());
    let solver = scioxide_linalg::GmresSolver::new(
        scioxide_core::ResidualTolerance::new(1e-14, 1e-14).unwrap(),
        Budget::try_from(1000).unwrap(),
    )
    .with_restart(Budget::try_from(80).unwrap());
    let options = IncompressibleSolveOptions::new()
        .with_nonlinear(scioxide_rootfind::NewtonKrylovOptions::default())
        .with_preconditioner(IncompressiblePreconditionerOptions::default());
    let mut workspace = p
        .workspace(&options)
        .unwrap()
        .with_linear_solver(Box::new(solver));
    let mut result =
        scioxide_rootfind::NewtonKrylovResult::new(p.unknowns(), options.nonlinear()).unwrap();
    p.solve_into(
        &vec![0.; p.unknowns()],
        &mut result,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(result.is_success(), "{:?}", result.termination());
}

#[test]
fn library_setup_limits_and_cancelled_application_are_structured_and_atomic() {
    let p = plan();
    let options = IncompressiblePreconditionerOptions::new().with_max_entries(0);
    let error = IncompressiblePreconditioner::new(&p, IncompressibleSaddle::Viscous, options)
        .err()
        .unwrap();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    let options = IncompressiblePreconditionerOptions::new()
        .with_factor_limits(scioxide_sparse::SparseSymmetricLimits::new(0, 0, 1));
    assert!(IncompressiblePreconditioner::new(&p, IncompressibleSaddle::Viscous, options).is_err());
    let inverse = IncompressiblePreconditioner::new(
        &p,
        IncompressibleSaddle::Viscous,
        IncompressiblePreconditionerOptions::default(),
    )
    .unwrap();
    let mut out = vec![42.; p.unknowns()];
    let mut work = vec![0.; inverse.precondition_workspace_len().unwrap()];
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "inverse", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            id,
            &mut budgets,
        )
        .unwrap();
    assert!(
        inverse
            .precondition_into_scoped(
                &vec![1.; p.unknowns()],
                &mut out,
                &mut work,
                &mut scioxide_core::OperatorContext::new(&mut ctx)
            )
            .is_err()
    );
    assert!(out.iter().all(|&v| v == 42.));
}
