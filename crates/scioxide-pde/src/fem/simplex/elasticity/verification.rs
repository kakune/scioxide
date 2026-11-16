use super::*;

#[cfg(feature = "std")]
use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState};

#[cfg(feature = "std")]
#[test]
fn executor_replay_matches_sequential_and_preserves_failure_output() {
    use scioxide_core::ThreadExecutor;
    let workers = core::num::NonZeroUsize::new(3).unwrap();
    let pool = ThreadExecutor::new(workers);
    let execution = ExecutionContext::new(&pool);
    let mesh = meshes::triangle_grid::<f64>(2).unwrap();
    let bc = natural(&mesh);
    let material = IsotropicElasticMaterial::new(1., 10.).unwrap();
    let sequential = HyperelasticityPlan::new(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        material,
        &bc,
        &|_| Ok([0.; 2]),
        options::<f64, 2>(true),
    )
    .unwrap();
    let opt = options::<f64, 2>(true).with_fem(
        SimplexFemOptions::new()
            .with_limits(limits())
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(632),
        scioxide_core::algorithm_id!("test", "elasticity", 1),
        0,
        0,
    );
    let mut ctx = state.begin_solve(execution, id, &mut budgets).unwrap();
    let parallel = HyperelasticityPlan::build(
        mesh,
        SimplexFemGeometry::Affine,
        material,
        &bc,
        &|_| Ok([0.; 2]),
        opt,
        &mut ctx,
    )
    .unwrap();
    let x = (0..parallel.unknowns())
        .map(|i| 0.001 * (i % 5) as f64)
        .collect::<Vec<_>>();
    let mut a = x.clone();
    let mut b = x.clone();
    let mut scratch = vec![0.; ResidualOperator::residual_workspace_len(&parallel).unwrap()];
    sequential.residual_into(&x, &mut a, &mut scratch).unwrap();
    parallel
        .residual_into_scoped(&x, &mut b, &mut scratch, &mut ctx)
        .unwrap();
    assert_eq!(a, b);
    sequential.jvp_into(&x, &x, &mut a, &mut scratch).unwrap();
    parallel
        .jvp_into_scoped(&x, &x, &mut b, &mut scratch, &mut ctx)
        .unwrap();
    assert_eq!(a, b);
}

fn prescribed<R: RealScalar, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    value: fn([R; D], usize) -> R,
) -> TaggedBoundaries<R, R, D> {
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    TaggedBoundaries::new(
        mesh,
        &tags,
        D,
        (0..D)
            .map(|component| {
                crate::BoundaryCondition::dirichlet(
                    0,
                    crate::boundary::point_normal(move |x, _| Ok(value(x, component))),
                )
                .for_component(component)
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn shared_newton_krylov_solves_mixed_neo_hookean_shear() {
    use scioxide_core::ResidualTolerance;
    use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver};
    for mixed in [false, true] {
        let mesh = meshes::triangle_grid::<f64>(2).unwrap();
        let boundary = prescribed(&mesh, |x, a| if a == 0 { 0.04 * x[1] } else { 0. });
        let plan = HyperelasticityPlan::new(
            mesh,
            SimplexFemGeometry::Affine,
            IsotropicElasticMaterial::new(1., 100.).unwrap(),
            &boundary,
            &|_| Ok([0.; 2]),
            options::<f64, 2>(mixed),
        )
        .unwrap();
        let initial = vec![0.; plan.unknowns()];
        let problem = NewtonKrylovProblem::with_jacobian(&plan);
        let settings = NewtonKrylovOptions::default()
            .with_residual_tolerance(ResidualTolerance::new(1e-10, 1e-10).unwrap())
            .with_step_tolerance(scioxide_core::StepTolerance::new(1e-13, 1e-13).unwrap());
        let result = NewtonKrylovSolver::new((settings).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &initial))
            .unwrap();
        assert!(
            result.is_success(),
            "mixed={mixed} {:?} {:?}",
            result.termination(),
            result.residual_report()
        );
        let x = result.point();
        for a in 0..2 {
            for i in 0..plan.displacement_nodes() {
                let expected = if a == 0 {
                    0.04 * plan.node_point(i).unwrap()[1]
                } else {
                    0.
                };
                assert!((x[a * plan.displacement_nodes() + i] - expected).abs() < 2e-8);
            }
        }
        if mixed {
            assert!(norm(&x[2 * plan.displacement_nodes()..]) < 2e-7);
        }
        let mut residual = initial.clone();
        let mut work = vec![0.; ResidualOperator::residual_workspace_len(&plan).unwrap()];
        plan.residual_into(x, &mut residual, &mut work).unwrap();
        assert!(norm(&residual) < 1e-9);
        assert!(
            plan.rigid_modes(Tolerance::new(1e-10, 1e-10).unwrap(), 3)
                .unwrap()
                .nullity()
                == 0
        );
    }
}

#[test]
fn small_strain_limit_and_homogeneous_volume_oracle() {
    let mesh = meshes::tetrahedron_grid::<f64>(1).unwrap();
    let bc = natural(&mesh);
    let material = IsotropicElasticMaterial::new(2., 7.).unwrap();
    for mixed in [false, true] {
        let linear = LinearElasticityPlan::new(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            material,
            &bc,
            &|_| Ok([0.; 3]),
            options::<f64, 3>(mixed),
        )
        .unwrap();
        let nonlinear = HyperelasticityPlan::new(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            material,
            &bc,
            &|_| Ok([0.; 3]),
            options::<f64, 3>(mixed),
        )
        .unwrap();
        let zero = vec![0.; linear.unknowns()];
        let tangent = dense(&nonlinear.assemble_tangent_csr(&zero).unwrap());
        let hooke = dense(&linear.assemble_csr().unwrap());
        assert!(
            norm(
                &tangent
                    .as_slice()
                    .iter()
                    .zip(hooke.as_slice())
                    .map(|(&a, &b)| a - b)
                    .collect::<Vec<_>>()
            ) < 5e-13
        );
        let stretch = 1.25_f64;
        let jac = stretch.powi(3);
        let mut state = zero;
        for a in 0..3 {
            for i in 0..nonlinear.displacement_nodes() {
                state[a * nonlinear.displacement_nodes() + i] =
                    (stretch - 1.) * nonlinear.node_point(i).unwrap()[a];
            }
        }
        if mixed {
            state[3 * nonlinear.displacement_nodes()..].fill(7. * (jac - 1.));
        }
        let expected = 3.5 * (jac - 1.).powi(2);
        assert!((nonlinear.energy(&state).unwrap().internal - expected).abs() < 2e-12);
        let stress = nonlinear.stress(0, [0.1; 3], &state).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (stress.cauchy[i][j] - if i == j { 7. * (jac - 1.) } else { 0. }).abs() < 2e-12
                );
            }
        }
    }
}

#[test]
fn disconnected_bodies_and_partial_supports_keep_only_surviving_modes() {
    use scioxide_spatial::{Point, TriangleMesh};
    let mesh = TriangleMesh::new(
        [[0., 0.], [1., 0.], [0., 1.], [3., 0.], [4., 0.], [3., 1.]]
            .into_iter()
            .map(|x| Point::new(x).unwrap())
            .collect(),
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    for supported in [false, true] {
        let tags = scioxide_spatial::SimplexBoundaryTags::new(
            &mesh,
            mesh.boundary_faces().map(|f| (f, 0)).collect(),
        )
        .unwrap();
        let bc = TaggedBoundaries::new(
            &mesh,
            &tags,
            2,
            (0..2)
                .map(|component| {
                    if supported && component == 0 {
                        crate::BoundaryCondition::dirichlet(
                            0,
                            crate::boundary::point_normal(|_, _| Ok(0.)),
                        )
                        .for_component(component)
                    } else {
                        crate::BoundaryCondition::natural(
                            0,
                            crate::boundary::point_normal(|_, _| Ok(0.)),
                        )
                        .for_component(component)
                    }
                })
                .collect(),
        )
        .unwrap();
        let plan = LinearElasticityPlan::new(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            IsotropicElasticMaterial::new(1., 10.).unwrap(),
            &bc,
            &|_| Ok([0.; 2]),
            options::<f64, 2>(true),
        )
        .unwrap();
        assert!(
            plan.rigid_modes(Tolerance::from_machine_epsilon(), 5)
                .is_err()
        );
        let modes = plan
            .rigid_modes(Tolerance::new(1e-10, 1e-10).unwrap(), 6)
            .unwrap();
        assert_eq!(modes.nullity(), if supported { 2 } else { 6 });
        let mut out = vec![0.; plan.unknowns()];
        let mut work = vec![0.; LinearOperator::apply_workspace_len(&plan).unwrap()];
        for k in 0..modes.nullity() {
            plan.apply_into(
                modes.basis().unwrap().column(k).unwrap(),
                &mut out,
                &mut work,
            )
            .unwrap();
            assert!(norm(&out) < 1e-12);
        }
    }
}
