use super::*;
use scioxide_linalg::GeneralizedHermitianEigenDecomposition;
use scioxide_ode::{Heun, OdeMethod, OdeOptions, OdeProblem, OdeSolution, OdeState};
use std::println;

#[test]
fn nonlinear_forms_cover_both_precisions_and_dimensions() {
    fn check<R: RealScalar, const D: usize>(mesh: SimplexMesh<R, D>) {
        let r = |v| R::try_from_f64_lossy(v).unwrap();
        let boundary = problem::boundaries(&mesh, r(0.04)).unwrap();
        let p = NavierStokesPlan::new(
            StokesPlan::with_options(
                mesh,
                SimplexFemGeometry::Affine,
                R::ONE,
                R::ONE,
                &boundary,
                &|_| Ok([R::ZERO; D]),
                problem::options(),
            )
            .unwrap(),
        );
        let exact = problem::state(p.stokes(), r(0.04), R::ZERO).unwrap();
        let x = exact
            .iter()
            .enumerate()
            .map(|(i, &v)| v + r(0.02 * (i as f64).sin()))
            .collect::<Vec<_>>();
        let d = (0..x.len())
            .map(|i| r(0.01 * (i as f64).cos()))
            .collect::<Vec<_>>();
        let h = r(1. / 256.);
        let plus = x
            .iter()
            .zip(&d)
            .map(|(&x, &d)| x + h * d)
            .collect::<Vec<_>>();
        let minus = x
            .iter()
            .zip(&d)
            .map(|(&x, &d)| x - h * d)
            .collect::<Vec<_>>();
        let mut a = vec![R::ZERO; x.len()];
        let mut b = a.clone();
        let mut j = a.clone();
        let mut work = vec![R::ZERO; ResidualOperator::residual_workspace_len(&p).unwrap()];
        p.residual_into(&plus, &mut a, &mut work).unwrap();
        p.residual_into(&minus, &mut b, &mut work).unwrap();
        p.jvp_into(&x, &d, &mut j, &mut work).unwrap();
        let error = a
            .iter()
            .zip(&b)
            .zip(&j)
            .map(|((&a, &b), &j)| ((a - b) / (h + h) - j).abs().into_f64())
            .fold(0., f64::max);
        assert!(
            error
                < if R::EPSILON.into_f64() > 1e-10 {
                    3e-5
                } else {
                    1e-11
                },
            "D={D}, derivative error={error}"
        );
    }
    check(super::super::super::meshes::triangle_grid::<f32>(2).unwrap());
    check(super::super::super::meshes::tetrahedron_grid::<f32>(2).unwrap());
    check(super::super::super::meshes::triangle_grid::<f64>(2).unwrap());
    check(super::super::super::meshes::tetrahedron_grid::<f64>(2).unwrap());
}

/// Time workspace whose saddles use `solver` with the library mass inverse.
fn injected<R: RealScalar, const D: usize>(
    plan: &StokesPlan<R, R, D>,
    solver: &scioxide_linalg::GmresSolver<R>,
) -> IncompressibleTimeWorkspace<R>
where
    IncompressiblePreconditioner<R>: Send,
    scioxide_linalg::GmresSolver<R>: Send + 'static,
{
    IncompressibleTimeWorkspace::new(plan, &IncompressibleSolveOptions::new())
        .unwrap()
        .with_linear_solver(Box::new(solver.clone()))
}

/// Fixed-step Heun from `y` at time zero to `end`.
fn heun(
    system: &IncompressibleSystem<'_, f64, 2>,
    y: &[f64],
    end: f64,
    h: f64,
) -> OdeSolution<f64> {
    Heun::new()
        .solve(
            &OdeProblem::new(system, end),
            &(OdeOptions::new().with_initial_step(h)),
            OdeState::new(0., Vector::from_vec(y.to_vec())),
        )
        .unwrap()
}

#[test]
fn nonlinear_time_evolution_satisfies_momentum_and_continuity() {
    let p = NavierStokesPlan::new(plan());
    let exact = problem::state(p.stokes(), 0.04, 0.3).unwrap();
    let solver = scioxide_linalg::GmresSolver::new(
        scioxide_core::ResidualTolerance::new(1e-12, 1e-12).unwrap(),
        Budget::try_from(500).unwrap(),
    )
    .with_restart(Budget::try_from(p.unknowns()).unwrap());
    let mut workspace = injected(p.stokes(), &solver);
    let system = IncompressibleSystem::navier_stokes(&p, &mut workspace, None, 0.0001).unwrap();
    let initial = system
        .free_velocity_indices()
        .iter()
        .enumerate()
        .map(|(k, &i)| exact[i] + 0.002 * (k as f64).cos())
        .collect::<Vec<_>>();
    let mut y = vec![0.; initial.len()];
    ephemeral(|ctx| system.project_velocity_into(&initial, &mut y, ctx)).unwrap();
    let mut acceleration = vec![0.; y.len()];
    ephemeral(|ctx| system.rhs(0., &y, &mut acceleration, ctx)).unwrap();
    assert!(acceleration.iter().any(|a| a.abs() > 1e-3));
    let mut state = vec![0.; p.unknowns()];
    ephemeral(|ctx| system.recover_into(0., &y, &mut state, ctx)).unwrap();
    let mut residual = vec![0.; p.unknowns()];
    let mut work = vec![0.; ResidualOperator::residual_workspace_len(&p).unwrap()];
    p.residual_into(&state, &mut residual, &mut work).unwrap();
    let mut full_acceleration = vec![0.; p.unknowns()];
    for (&i, &a) in system.free_velocity_indices().iter().zip(&acceleration) {
        full_acceleration[i] = a;
    }
    let mut mass = vec![0.; p.unknowns()];
    p.stokes()
        .mass_system()
        .apply_into(&full_acceleration, &mut mass, &mut work)
        .unwrap();
    for &i in system.free_velocity_indices() {
        assert!((mass[i] + residual[i]).abs() < 1e-9);
    }
    let solution = heun(&system, &y, 0.0004, 0.0001);
    assert_eq!(solution.report().final_time(), 0.0004);
    system.validate_velocity(solution.state().values()).unwrap();
    assert!(
        solution
            .state()
            .values()
            .iter()
            .zip(&y)
            .any(|(&a, &b)| (a - b).abs() > 1e-6)
    );
    let limited = scioxide_linalg::GmresSolver::new(
        scioxide_core::ResidualTolerance::new(1e-12, 1e-12).unwrap(),
        Budget::ONE,
    );
    let mut limited_workspace = injected(p.stokes(), &limited);
    let limited_system =
        IncompressibleSystem::navier_stokes(&p, &mut limited_workspace, None, 0.0001).unwrap();
    acceleration.fill(7.);
    let error = ephemeral(|ctx| limited_system.rhs(0., &y, &mut acceleration, ctx)).unwrap_err();
    assert!(
        matches ! (& (error) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ConvergenceFailure { reason : scioxide_core :: ConvergenceReason :: IterationLimit , .. }))
    );
    assert!(acceleration.iter().all(|&a| a == 7.));
}

// Independent dense constrained eigenproblem supplies an exact semidiscrete
// velocity/pressure exponential, separate from the production Krylov/ODE path.
fn mode(p: &StokesPlan<f64, f64, 2>) -> (Vec<f64>, Vec<f64>, f64) {
    let mass = p.mass_system().assemble_csr().unwrap();
    let nv = p.free.len();
    let n = p.unknowns();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for &col in &p.free {
        let mut rhs = vec![0.; n];
        for &row in &p.free {
            rhs[row] = mass.get(row, col).unwrap();
        }
        let x = dense_solve(&mass, &rhs);
        let mut v = p.free.iter().map(|&i| x[i]).collect::<Vec<_>>();
        for _ in 0..2 {
            for q in &basis {
                let dot = q.iter().zip(&v).map(|(&a, &b)| a * b).sum::<f64>();
                for (a, &b) in v.iter_mut().zip(q) {
                    *a -= dot * b;
                }
            }
        }
        let norm = v.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
            basis.push(v);
        }
    }
    assert_eq!(basis.len(), nv - (p.mesh().vertex_count() - 1));
    let k = basis.len();
    let a = p.assemble_csr().unwrap();
    let mut ar = Matrix::<f64>::zeros(k, k).unwrap();
    let mut mr = Matrix::<f64>::zeros(k, k).unwrap();
    for i in 0..k {
        for j in 0..k {
            for (row, &r) in p.free.iter().enumerate() {
                for (col, &c) in p.free.iter().enumerate() {
                    ar.as_mut_slice()[i * k + j] +=
                        basis[i][row] * a.get(r, c).unwrap() * basis[j][col];
                    mr.as_mut_slice()[i * k + j] +=
                        basis[i][row] * mass.get(r, c).unwrap() * basis[j][col];
                }
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            for matrix in [&mut ar, &mut mr] {
                let value =
                    0.5 * (matrix.view().get(i, j).unwrap() + matrix.view().get(j, i).unwrap());
                matrix.as_mut_slice()[i * k + j] = value;
                matrix.as_mut_slice()[j * k + i] = value;
            }
        }
    }
    let eigen = GeneralizedHermitianEigenDecomposition::decompose(
        &ar,
        &mr,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let lambda = eigen.eigenvalues()[0];
    let velocity = (0..nv)
        .map(|i| {
            0.01 * (0..k)
                .map(|j| basis[j][i] * eigen.eigenvectors().view().get(j, 0).unwrap())
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut full = vec![0.; n];
    for (&i, &v) in p.free.iter().zip(&velocity) {
        full[i] = v;
    }
    let mut rhs = vec![0.; n];
    for &i in &p.free {
        rhs[i] = -(0..n).map(|j| a.get(i, j).unwrap() * full[j]).sum::<f64>();
    }
    let mixed = dense_solve(&mass, &rhs);
    for (&i, &v) in p.free.iter().zip(&velocity) {
        assert!((mixed[i] + lambda * v).abs() < 1e-9);
    }
    (velocity, mixed, lambda)
}
#[test]
fn transient_velocity_and_endpoint_pressure_have_second_order_accuracy() {
    let mesh = super::super::super::meshes::triangle_grid::<f64>(2).unwrap();
    let b = problem::boundaries(&mesh, 0.).unwrap();
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &b,
        &|_| Ok([0.; 2]),
        problem::options(),
    )
    .unwrap();
    let (y, pressure, lambda) = mode(&p);
    let solver = scioxide_linalg::GmresSolver::new(
        scioxide_core::ResidualTolerance::new(1e-13, 1e-13).unwrap(),
        Budget::try_from(500).unwrap(),
    )
    .with_restart(Budget::try_from(p.unknowns()).unwrap());
    let end = 1. / 512.;
    let mut errors = Vec::new();
    for steps in [4, 8, 16] {
        let h = end / f64::from(steps);
        let mut work = injected(&p, &solver);
        let system = IncompressibleSystem::stokes(&p, &mut work, None, h).unwrap();
        system.validate_velocity(&y).unwrap();
        let solution = heun(&system, &y, end, h);
        assert_eq!(solution.report().final_time(), end);
        let amplification = (-lambda * end).exp();
        let discrete = (1. - lambda * h + 0.5 * (lambda * h).powi(2)).powi(steps);
        let actual = solution.state().values();
        let e = actual
            .iter()
            .zip(&y)
            .map(|(&a, &b)| (a - amplification * b).abs())
            .fold(0., f64::max);
        assert!(
            actual
                .iter()
                .zip(&y)
                .all(|(&a, &b)| (a - discrete * b).abs() < 1e-10)
        );
        let mut recovered = vec![0.; p.unknowns()];
        ephemeral(|ctx| system.recover_into(end, actual, &mut recovered, ctx)).unwrap();
        let pe = recovered[p.velocity_unknowns()..]
            .iter()
            .zip(&pressure[p.velocity_unknowns()..])
            .map(|(&a, &b)| (a - amplification * b).abs())
            .fold(0., f64::max);
        let mut scratch = vec![0.; LinearOperator::apply_workspace_len(&p).unwrap()];
        let diagnostic = p.diagnostics(&recovered, 1., h, &mut scratch).unwrap();
        assert!(diagnostic.weak_divergence < 1e-10);
        assert!(diagnostic.boundary_flux < 1e-12);
        println!(
            "transient steps={steps}, lambda={lambda}, velocity_error={e}, pressure_error={pe}, Courant={}, diffusion={}",
            diagnostic.courant, diagnostic.diffusion_number
        );
        errors.push((e, pe));
    }
    for pair in errors.windows(2) {
        assert!((pair[0].0 / pair[1].0).log2() > 1.9);
        assert!((pair[0].1 / pair[1].1).log2() > 1.9);
    }
}

#[test]
fn default_transient_flow_bounds_divergence_for_one_thousand_steps() {
    for nonlinear in [false, true] {
        let p = NavierStokesPlan::new(plan());
        let tolerance = scioxide_core::ResidualTolerance::new(1e-11, 1e-11).unwrap();
        let options = IncompressibleSolveOptions::new().with_residual(tolerance);
        let mut work = IncompressibleTimeWorkspace::new(p.stokes(), &options).unwrap();
        let h = 1. / 4096.;
        let forcing = |x: [f64; 2], t: f64| Ok([0.3 * (20. * t).sin() * x[1], 0.]);
        let system = if nonlinear {
            IncompressibleSystem::navier_stokes(&p, &mut work, Some(&forcing), h)
        } else {
            IncompressibleSystem::stokes(p.stokes(), &mut work, Some(&forcing), h)
        }
        .unwrap();
        let exact = problem::state(p.stokes(), 0.04, 0.3).unwrap();
        let initial = system
            .free_velocity_indices()
            .iter()
            .enumerate()
            .map(|(j, &i)| exact[i] + 0.003 * (j as f64).cos())
            .collect::<Vec<_>>();
        let mut projected = vec![0.; initial.len()];
        ephemeral(|ctx| system.project_velocity_into(&initial, &mut projected, ctx)).unwrap();
        let mut full = vec![0.; p.unknowns()];
        let mut residual = full.clone();
        let mut scratch = vec![0.; LinearOperator::apply_workspace_len(p.stokes()).unwrap()];
        let mut steps = 0;
        let mut maximum = 0.0_f64;
        let plan = (
            Heun::new(),
            OdeOptions::new()
                .with_initial_step(h)
                .with_output_schedule(scioxide_ode::OdeOutputSchedule::AcceptedSteps),
        );
        let ode_problem = OdeProblem::new(&system, 1000. * h);
        let mut state = OdeState::new(0., Vector::from_vec(projected));
        let report = plan
            .0
            .advance(
                &ode_problem,
                &plan.1,
                &mut state,
                &mut |time: f64, state: &[f64]| {
                    full.fill(0.);
                    for (i, value) in full
                        .iter_mut()
                        .enumerate()
                        .take(p.stokes().velocity_unknowns())
                    {
                        *value = p.stokes().prescribed(i).unwrap_or(0.);
                    }
                    for (&i, &value) in system.free_velocity_indices().iter().zip(state) {
                        full[i] = value;
                    }
                    p.stokes().apply_into(&full, &mut residual, &mut scratch)?;
                    let defect = residual[p.stokes().velocity_unknowns()..p.stokes().physical()]
                        .iter()
                        .zip(
                            &p.stokes().rhs()
                                [p.stokes().velocity_unknowns()..p.stokes().physical()],
                        )
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    maximum = maximum.max(defect);
                    let scale = full.iter().map(|v| v * v).sum::<f64>().sqrt();
                    // Heun's restoration amplification is <=1/2 at h=tau. Residual
                    // forcing contributes O(tau*tolerance); native arithmetic adds its floor.
                    let bound = 8. * h * (tolerance.absolute() + tolerance.relative())
                        + 512. * f64::EPSILON * (1. + scale);
                    assert!(
                        defect <= bound,
                        "nonlinear={nonlinear}, t={time}, defect={defect}, bound={bound}"
                    );
                    if time > 0. {
                        steps += 1;
                    }
                    Ok(())
                },
                &mut plan.0.workspace(&ode_problem, &plan.1).unwrap(),
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(steps, 1000);
        assert_eq!(report.final_time(), 1000. * h);
        std::println!("1000-step flow nonlinear={nonlinear}, max divergence={maximum}");
    }
}

#[test]
fn automatic_flow_limits_follow_viscous_and_convective_scaling() {
    let mut limits = Vec::new();
    for n in [2, 4, 8] {
        let mesh = super::super::super::meshes::triangle_grid::<f64>(n).unwrap();
        let b = problem::boundaries(&mesh, 0.).unwrap();
        let p = StokesPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            1.,
            1.,
            &b,
            &|_| Ok([0.; 2]),
            problem::options(),
        )
        .unwrap();
        let mut work =
            IncompressibleTimeWorkspace::new(&p, &IncompressibleSolveOptions::new()).unwrap();
        let system = IncompressibleSystem::stokes(&p, &mut work, None, 1000.).unwrap();
        limits.push(
            ephemeral(|ctx| system.step_limit(0., &vec![0.; system.dimension()], ctx))
                .unwrap()
                .unwrap(),
        );
    }
    for pair in limits.windows(2) {
        assert!((pair[1] / pair[0] - 0.25).abs() < 1e-12);
    }
    let mesh = super::super::super::meshes::triangle_grid::<f64>(2).unwrap();
    let b = problem::boundaries(&mesh, 0.).unwrap();
    let p = NavierStokesPlan::new(
        StokesPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            1.,
            1e-6,
            &b,
            &|_| Ok([0.; 2]),
            problem::options(),
        )
        .unwrap(),
    );
    let mut work =
        IncompressibleTimeWorkspace::new(p.stokes(), &IncompressibleSolveOptions::new()).unwrap();
    let system = IncompressibleSystem::navier_stokes(&p, &mut work, None, 1000.).unwrap();
    let limit = |speed| {
        ephemeral(|ctx| system.step_limit(0., &vec![speed; system.dimension()], ctx))
            .unwrap()
            .unwrap()
    };
    assert!((limit(20.) / limit(10.) - 0.5).abs() < 1e-12);
    let mut sentinel = vec![17.; system.dimension()];
    assert!(ephemeral(|ctx| system.rhs(0., &[f64::NAN], &mut sentinel, ctx)).is_err());
    assert!(sentinel.iter().all(|&v| v == 17.));
}
