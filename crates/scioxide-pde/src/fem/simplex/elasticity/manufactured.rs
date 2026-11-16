use super::*;
use scioxide_core::ResidualTolerance;
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{MinresSolver, Vector};

fn solve<R: RealScalar, const D: usize>(plan: &LinearElasticityPlan<R, R, D>) -> Vec<R> {
    let matrix = plan.assemble_csr().unwrap();
    // Mixed f32 systems need headroom above the MINRES rounding floor.
    let tolerance = r::<R>(2e-13).max(R::EPSILON * r(64.));
    // Keep the physical operator check independent of the solver tolerance.
    let physical_error_bound = if R::EPSILON.into_f64() > 1e-10 {
        5e-6
    } else {
        1e-11
    };
    let mut diagonal =
        SparseBuilder::with_capacity(plan.unknowns(), plan.unknowns(), plan.unknowns()).unwrap();
    for i in 0..plan.unknowns() {
        let mut d = matrix.get(i, i).unwrap().abs();
        if i >= D * plan.displacement_nodes()
            && plan.space.kinematics != ElasticityKinematics::PlaneStress
        {
            d = d * plan.space.material.bulk() / plan.space.material.shear();
        }
        diagonal.push(i, i, d).unwrap();
    }
    let diagonal = diagonal
        .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
        .unwrap();
    let preconditioner = scioxide_sparse::JacobiPreconditioner::new(&diagonal).unwrap();
    let residual_tolerance = ResidualTolerance::new(tolerance, tolerance).unwrap();
    let solver = MinresSolver::new(residual_tolerance, Budget::try_from(12000).unwrap())
        .with_normal_tolerance(
            scioxide_core::GradientTolerance::new(R::ZERO, R::EPSILON * R::EPSILON).unwrap(),
        );
    let mut workspace = scioxide_linalg::IterativeWorkspace::new();
    let mut solution = Vector::zeros(plan.unknowns()).unwrap();
    let rhs = Vector::from_vec(plan.rhs().to_vec());
    let report = solver
        .solve_into(
            &matrix,
            &rhs,
            Preconditioning::Inverse(&preconditioner),
            solution.view_mut(),
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success(), "MINRES {report:?}");
    let residual = report.true_residual().unwrap();
    assert!(
        residual_tolerance
            .is_satisfied(residual.raw_norm(), residual.scaling().denominator())
            .unwrap()
    );
    let mut check = vec![R::ZERO; plan.unknowns()];
    let mut work = vec![R::ZERO; LinearOperator::apply_workspace_len(plan).unwrap()];
    plan.apply_into(solution.as_slice(), &mut check, &mut work)
        .unwrap();
    let error = check
        .iter()
        .zip(plan.rhs())
        .fold(R::ZERO, |s, (&a, &b)| s.max((a - b).abs()));
    assert!(
        error.into_f64() < physical_error_bound,
        "physical error {:?}, bound {:?}",
        error,
        physical_error_bound
    );
    solution.as_slice().to_vec()
}
fn exponential<const D: usize>(x: [f64; D], a: usize) -> f64 {
    0.01 * (a + 1) as f64 * (0.5 * x.iter().sum::<f64>()).exp()
}
fn smooth_error<const D: usize>(
    mesh: SimplexMesh<f64, D>,
    order: SimplexFemOrder,
    mixed: bool,
    curved: bool,
    plane: ElasticityKinematics,
) -> (f64, f64, f64) {
    let (mapped, edges) = meshes::curved(&mesh).unwrap();
    let mesh = if curved { mapped } else { mesh };
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let bc = TaggedBoundaries::new(
        &mesh,
        &tags,
        D,
        (0..D)
            .map(|component| {
                crate::BoundaryCondition::dirichlet(
                    0,
                    crate::boundary::point_normal(move |x, _| Ok(exponential(x, component))),
                )
                .for_component(component)
            })
            .collect(),
    )
    .unwrap();
    let mu = 2.;
    let bulk = 7.;
    let lambda = bulk - 2. * mu / 3.;
    let effective = if plane == ElasticityKinematics::PlaneStress {
        2. * mu * lambda / (lambda + 2. * mu)
    } else {
        lambda
    };
    let body = |x| {
        Ok(core::array::from_fn(|a| {
            -(mu * D as f64 * exponential(x, a)
                + (effective + mu) * (0..D).map(|b| exponential(x, b)).sum::<f64>())
                * 0.25
        }))
    };
    let mut opt = formulated::<f64, D>(if mixed {
        ElasticityFormulation::MixedP2P1
    } else {
        ElasticityFormulation::Displacement(order)
    });
    opt.kinematics = plane;
    let plan = LinearElasticityPlan::new(
        mesh,
        if curved {
            SimplexFemGeometry::Quadratic(edges.view())
        } else {
            SimplexFemGeometry::Affine
        },
        IsotropicElasticMaterial::new(mu, bulk).unwrap(),
        &bc,
        &body,
        opt,
    )
    .unwrap();
    let solution = solve(&plan);
    let rule = SimplexRule::new(
        D,
        if curved { 4 } else { 3 },
        scioxide_integrate::GeometricOptions::default()
            .with_max_nodes(Budget::try_from(512).unwrap()),
    )
    .unwrap();
    let mut l2 = 0.;
    let mut stress_error = 0.;
    let mut pressure_error = 0.;
    let mut values = [0.; D];
    let mut gradients = [[0.; D]; D];
    for cell in 0..plan.mesh().cell_count() {
        for (bary, &weight) in rule
            .barycentric_nodes()
            .chunks_exact(D + 1)
            .zip(rule.weights())
        {
            let point = core::array::from_fn(|j| bary[j + 1]);
            let map = plan.space.space.map_point(cell, point).unwrap();
            let weight = weight * map.determinant;
            plan.space
                .space
                .evaluate_into(
                    cell,
                    point,
                    &solution[..D * plan.displacement_nodes()],
                    &mut values,
                    &mut gradients,
                )
                .unwrap();
            let exact: [f64; D] = core::array::from_fn(|a| exponential(map.point, a));
            let trace = 0.5 * exact.iter().sum::<f64>();
            let stress = plan.stress(cell, point, &solution).unwrap();
            for a in 0..D {
                l2 += weight * (values[a] - exact[a]).powi(2);
                for b in 0..D {
                    let expected = mu * 0.5 * (exact[a] + exact[b])
                        + if a == b { effective * trace } else { 0. };
                    stress_error += weight * (stress.cauchy[a][b] - expected).powi(2);
                }
            }
            let p = bulk
                * trace
                * if plane == ElasticityKinematics::PlaneStress {
                    2. * mu / (lambda + 2. * mu)
                } else {
                    1.
                };
            pressure_error += weight * (stress.pressure - p).powi(2);
        }
    }
    (l2.sqrt(), stress_error.sqrt(), pressure_error.sqrt())
}
fn refinement<const D: usize>(
    mesh: impl Fn(usize) -> SimplexMesh<f64, D>,
    sizes: [usize; 3],
    plane: ElasticityKinematics,
) {
    for (order, mixed, curved) in [
        (SimplexFemOrder::P1, false, false),
        (SimplexFemOrder::P2, false, false),
        (SimplexFemOrder::P2, true, false),
        (SimplexFemOrder::P2, true, true),
    ] {
        let errors = sizes.map(|n| smooth_error(mesh(n), order, mixed, curved, plane));
        let u = (errors[1].0 / errors[2].0).log2();
        let s = (errors[1].1 / errors[2].1).log2();
        #[cfg(feature = "std")]
        std::eprintln!(
            "elasticity D={D} {plane:?} {order:?} mixed={mixed} curved={curved}: {errors:?}; displacement={u:.4} stress={s:.4}"
        );
        assert!(errors[0].0 > errors[1].0 && errors[1].0 > errors[2].0);
        assert!(u > order.degree() as f64 + 0.75, "displacement rate {u}");
        assert!(s > order.degree() as f64 - 0.25, "stress rate {s}");
    }
}
#[test]
fn manufactured_linear_displacement_stress_and_pressure_orders_2d() {
    for plane in [
        ElasticityKinematics::PlaneStress,
        ElasticityKinematics::PlaneStrain,
    ] {
        refinement(|n| meshes::triangle_grid(n).unwrap(), [2, 4, 8], plane);
    }
}
#[test]
fn manufactured_linear_displacement_stress_and_pressure_orders_3d() {
    refinement(
        |n| meshes::tetrahedron_grid(n).unwrap(),
        [1, 2, 4],
        ElasticityKinematics::ThreeDimensional,
    );
}

// Divergence-free displacement from the polynomial streamfunction X(x) X(y),
// X(t)=t²(1-t)². Boundary displacement and net boundary flux are exactly zero.
fn polynomial<R: RealScalar>(x: R) -> [R; 4] {
    [
        x * x * (R::ONE - x).powi(2),
        r::<R>(2.) * x - r::<R>(6.) * x * x + r::<R>(4.) * x * x * x,
        r::<R>(2.) - r::<R>(12.) * x + r::<R>(12.) * x * x,
        -r::<R>(12.) + r::<R>(24.) * x,
    ]
}
fn locking<R: RealScalar>(n: usize, bulk: f64) -> f64 {
    let mesh = meshes::triangle_grid::<R>(n).unwrap();
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
                crate::BoundaryCondition::dirichlet(
                    0,
                    crate::boundary::point_normal(|_, _| Ok(R::ZERO)),
                )
                .for_component(component)
            })
            .collect(),
    )
    .unwrap();
    let body = |p: [R; 2]| {
        let x = polynomial(p[0]);
        let y = polynomial(p[1]);
        Ok([-(x[2] * y[1] + x[0] * y[3]), x[3] * y[0] + x[1] * y[2]])
    };
    let plan = LinearElasticityPlan::new(
        mesh,
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(R::ONE, r(bulk)).unwrap(),
        &bc,
        &body,
        options::<R, 2>(true),
    )
    .unwrap();
    let solution = solve(&plan);
    let rule = SimplexRule::new(
        2,
        4,
        scioxide_integrate::GeometricOptions::default()
            .with_max_nodes(Budget::try_from(256).unwrap()),
    )
    .unwrap();
    let mut error = R::ZERO;
    let mut values = [R::ZERO; 2];
    let mut gradients = [[R::ZERO; 2]; 2];
    for cell in 0..plan.mesh().cell_count() {
        for (bary, &weight) in rule.barycentric_nodes().chunks_exact(3).zip(rule.weights()) {
            let point = [bary[1], bary[2]];
            let map = plan.space.space.map_point(cell, point).unwrap();
            let x = polynomial(map.point[0]);
            let y = polynomial(map.point[1]);
            let exact = [x[0] * y[1], -x[1] * y[0]];
            plan.space
                .space
                .evaluate_into(
                    cell,
                    point,
                    &solution[..2 * plan.displacement_nodes()],
                    &mut values,
                    &mut gradients,
                )
                .unwrap();
            error += weight
                * map.determinant
                * ((values[0] - exact[0]).powi(2) + (values[1] - exact[1]).powi(2));
        }
    }
    error.sqrt().into_f64()
}
#[test]
fn mixed_near_incompressible_error_is_uniform_in_bulk_modulus() {
    for bulk in [10., 100., 10_000., 1_000_000.] {
        let coarse = locking::<f64>(4, bulk);
        let fine = locking::<f64>(8, bulk);
        let rate = (coarse / fine).log2();
        #[cfg(feature = "std")]
        std::eprintln!("mixed locking f64 bulk={bulk}: {coarse:e} -> {fine:e}, order {rate:.4}");
        assert!(rate > 2.75);
        assert!(fine < 1e-4);
    }
    for bulk in [10., 100., 10_000.] {
        let fine = locking::<f32>(8, bulk);
        #[cfg(feature = "std")]
        std::eprintln!("mixed locking f32 bulk={bulk}: {fine:e}");
        assert!(fine < 1e-4);
    }
}

fn neo_error<const D: usize>(
    mesh: SimplexMesh<f64, D>,
    order: SimplexFemOrder,
    mixed: bool,
) -> (f64, f64) {
    use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver};
    let value = |x: [f64; D], a: usize| {
        if a == 0 {
            0.02 * (0.5 * x[0]).exp()
        } else {
            0.
        }
    };
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let bc = TaggedBoundaries::new(
        &mesh,
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
    .unwrap();
    // Independent diagonal-stretch formula: P11=2mu/3(s^(1/3)-s^(-5/3))+kappa(s-1).
    let mu = 2.;
    let bulk = 7.;
    let body = |x: [f64; D]| {
        let s = 1. + 0.01 * (0.5 * x[0]).exp();
        let mut f = [0.; D];
        f[0] = -(2. * mu / 9. * (s.powf(-2. / 3.) + 5. * s.powf(-8. / 3.)) + bulk)
            * 0.005
            * (0.5 * x[0]).exp();
        Ok(f)
    };
    let opt = formulated::<f64, D>(if mixed {
        ElasticityFormulation::MixedP2P1
    } else {
        ElasticityFormulation::Displacement(order)
    });
    let plan = HyperelasticityPlan::new(
        mesh,
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(mu, bulk).unwrap(),
        &bc,
        &body,
        opt,
    )
    .unwrap();
    let mut initial = vec![0.; plan.unknowns()];
    for a in 0..D {
        for i in 0..plan.displacement_nodes() {
            initial[a * plan.displacement_nodes() + i] = value(plan.node_point(i).unwrap(), a);
        }
    }
    if mixed {
        for i in 0..plan.mesh().vertex_count() {
            initial[D * plan.displacement_nodes() + i] =
                bulk * 0.01 * (0.5 * plan.node_point(i).unwrap()[0]).exp();
        }
    }
    let settings = NewtonKrylovOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(1e-11, 1e-11).unwrap())
        .with_step_tolerance(scioxide_core::StepTolerance::new(1e-14, 1e-14).unwrap())
        .with_krylov(
            Budget::try_from(2000).unwrap(),
            Budget::try_from(80).unwrap(),
        );
    // Frozen Jacobi of the initial tangent; unpreconditioned GMRES stalls on the
    // mixed P2/P1 saddle-point Jacobian.
    let tangent = plan.assemble_tangent_csr(&initial).unwrap();
    let preconditioner = scioxide_sparse::JacobiPreconditioner::new(&tangent).unwrap();
    let result = NewtonKrylovSolver::new(settings)
        .and_then(|root_plan| {
            root_plan.solve(
                &NewtonKrylovProblem::with_jacobian(&plan).with_preconditioner(&preconditioner),
                &initial,
            )
        })
        .unwrap();
    assert!(
        result.is_success(),
        "neo D={D} {order:?} mixed={mixed}: {:?} {:?}",
        result.termination(),
        result.residual_report()
    );
    let solution = result.point();
    let rule = SimplexRule::new(
        D,
        3,
        scioxide_integrate::GeometricOptions::default()
            .with_max_nodes(Budget::try_from(256).unwrap()),
    )
    .unwrap();
    let mut l2 = 0.;
    let mut stress_error = 0.;
    let mut values = [0.; D];
    let mut gradients = [[0.; D]; D];
    for cell in 0..plan.mesh().cell_count() {
        for (bary, &w) in rule
            .barycentric_nodes()
            .chunks_exact(D + 1)
            .zip(rule.weights())
        {
            let point = core::array::from_fn(|j| bary[j + 1]);
            let map = plan.space.space.map_point(cell, point).unwrap();
            let w = w * map.determinant;
            let s = 1. + 0.01 * (0.5 * map.point[0]).exp();
            let iso = mu * s.powf(-2. / 3.);
            let expected = [
                iso * (s - (s * s + 2.) / (3. * s)) + bulk * (s - 1.),
                iso * (1. - (s * s + 2.) / 3.) + bulk * (s - 1.) * s,
            ];
            plan.space
                .space
                .evaluate_into(
                    cell,
                    point,
                    &solution[..D * plan.displacement_nodes()],
                    &mut values,
                    &mut gradients,
                )
                .unwrap();
            let stress = plan.stress(cell, point, solution).unwrap();
            for a in 0..D {
                l2 += w * (values[a] - value(map.point, a)).powi(2);
                for b in 0..D {
                    stress_error += w
                        * (stress.first_piola[a][b]
                            - if a == b {
                                expected[usize::from(a != 0)]
                            } else {
                                0.
                            })
                        .powi(2);
                }
            }
        }
    }
    (l2.sqrt(), stress_error.sqrt())
}
fn neo_refinement<const D: usize>(mesh: impl Fn(usize) -> SimplexMesh<f64, D>) {
    for (order, mixed) in [
        (SimplexFemOrder::P1, false),
        (SimplexFemOrder::P2, false),
        (SimplexFemOrder::P2, true),
    ] {
        let errors = [1, 2, 4].map(|n| neo_error(mesh(n), order, mixed));
        let u = (errors[1].0 / errors[2].0).log2();
        let s = (errors[1].1 / errors[2].1).log2();
        #[cfg(feature = "std")]
        std::eprintln!(
            "neo-Hookean D={D} {order:?} mixed={mixed}: {errors:?}, displacement={u:.4}, Piola={s:.4}"
        );
        assert!(
            u > order.degree() as f64 + 0.75,
            "neo displacement rate {u}"
        );
        assert!(s > order.degree() as f64 - 0.25, "neo Piola rate {s}");
    }
}
#[test]
fn manufactured_neo_hookean_displacement_and_stress_orders_2d() {
    neo_refinement(|n| meshes::triangle_grid(n).unwrap());
}
#[test]
fn manufactured_neo_hookean_displacement_and_stress_orders_3d() {
    neo_refinement(|n| meshes::tetrahedron_grid(n).unwrap());
}
