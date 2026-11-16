use super::*;
use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting};
use scioxide_spatial::{Point, TriangleMesh};
use std::println;

fn zeros<const D: usize>(mesh: &SimplexMesh<f64, D>) -> TaggedBoundaries<f64, f64, D> {
    problem::boundaries(mesh, 0.).unwrap()
}
fn solve<const D: usize>(p: &StokesPlan<f64, f64, D>) -> Vec<f64> {
    let options = IncompressibleSolveOptions::new()
        .with_residual(scioxide_core::ResidualTolerance::new(1e-12, 1e-12).unwrap());
    let result = p.solve_with_options(&options).unwrap();
    assert!(result.is_success(), "{:?}", result.report());
    result.solution().as_slice().to_vec()
}
fn inf_sup<const D: usize>(mesh: SimplexMesh<f64, D>) -> f64 {
    let b = zeros(&mesh);
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &b,
        &|_| Ok([0.; D]),
        problem::options(),
    )
    .unwrap();
    let csr = p.assemble_csr().unwrap();
    let nv = p.free.len();
    let np = p.mesh().vertex_count();
    let mut a = Matrix::<f64>::zeros(nv, nv).unwrap();
    for (i, &row) in p.free.iter().enumerate() {
        for (j, &col) in p.free.iter().enumerate() {
            a.as_mut_slice()[i * nv + j] = csr.get(row, col).unwrap();
        }
    }
    let factor = CholeskyDecomposition::factor(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        CholeskyPivoting::None,
    )
    .unwrap();
    let mut b = vec![0.; np * nv];
    for i in 0..np {
        for (j, &col) in p.free.iter().enumerate() {
            b[i * nv + j] = csr.get(p.velocity_unknowns() + i, col).unwrap();
        }
    }
    let mut schur = Matrix::<f64>::zeros(np, np).unwrap();
    for j in 0..np {
        let column = factor
            .solve(&Vector::from_vec(b[j * nv..(j + 1) * nv].to_vec()))
            .unwrap();
        for i in 0..np {
            schur.as_mut_slice()[i * np + j] = b[i * nv..(i + 1) * nv]
                .iter()
                .zip(column.as_slice())
                .map(|(&a, &b)| a * b)
                .sum();
        }
    }
    // Independent P1 mass: integral lambda_i lambda_j = volume*(1+delta_ij)/((D+1)(D+2)).
    let mut mass = Matrix::<f64>::zeros(np, np).unwrap();
    let mut weights = vec![0.; np];
    for cell in 0..p.mesh().cell_count() {
        let volume = p.mesh().cell_measure(cell).unwrap();
        let vertices = p.mesh().cell_vertices(cell).unwrap();
        for &i in vertices {
            weights[i] += volume / (D + 1) as f64;
            for &j in vertices {
                mass.as_mut_slice()[i * np + j] +=
                    volume * if i == j { 2. } else { 1. } / ((D + 1) * (D + 2)) as f64;
            }
        }
    }
    let n = np - 1;
    let mut s = Matrix::<f64>::zeros(n, n).unwrap();
    let mut m = Matrix::<f64>::zeros(n, n).unwrap();
    for i in 0..n {
        for j in 0..n {
            let wi = weights[i] / weights[n];
            let wj = weights[j] / weights[n];
            for (target, source) in [(&mut s, &schur), (&mut m, &mass)] {
                target.as_mut_slice()[i * n + j] = source.view().get(i, j).unwrap()
                    - wi * source.view().get(n, j).unwrap()
                    - wj * source.view().get(i, n).unwrap()
                    + wi * wj * source.view().get(n, n).unwrap();
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            let v = (s.view().get(i, j).unwrap() + s.view().get(j, i).unwrap()) * 0.5;
            s.as_mut_slice()[i * n + j] = v;
            s.as_mut_slice()[j * n + i] = v;
        }
    }
    let values = scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute(
        &s,
        &m,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let beta = values
        .eigenvalues()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .sqrt();
    println!(
        "inf-sup D={D}, cells={}, beta={beta:.8}",
        p.mesh().cell_count()
    );
    assert!(beta > 0.02);
    beta
}
#[test]
fn pressure_inf_sup_remains_bounded_under_refinement() {
    let a = inf_sup(super::super::super::meshes::triangle_grid(2).unwrap());
    let b = inf_sup(super::super::super::meshes::triangle_grid(4).unwrap());
    assert!(b > 0.5 * a);
    let a = inf_sup(super::super::super::meshes::tetrahedron_grid(2).unwrap());
    let b = inf_sup(super::super::super::meshes::tetrahedron_grid(3).unwrap());
    assert!(b > 0.5 * a);
}

// f and its first three exact polynomial derivatives. u=curl(0,0,prod f).
fn f(x: f64) -> [f64; 4] {
    [
        x * x * (1. - x) * (1. - x),
        2. * x - 6. * x * x + 4. * x * x * x,
        2. - 12. * x + 12. * x * x,
        -12. + 24. * x,
    ]
}
fn manufactured<const D: usize>(x: [f64; D]) -> ([f64; D], [[f64; D]; D], [f64; D], f64) {
    let a = f(x[0]);
    let b = f(x[1]);
    let c = if D == 3 { f(x[2]) } else { [1., 0., 0., 0.] };
    let mut u = [0.; D];
    u[0] = a[0] * b[1] * c[0];
    u[1] = -a[1] * b[0] * c[0];
    let mut g = [[0.; D]; D];
    g[0][0] = a[1] * b[1] * c[0];
    g[0][1] = a[0] * b[2] * c[0];
    g[1][0] = -a[2] * b[0] * c[0];
    g[1][1] = -a[1] * b[1] * c[0];
    if D == 3 {
        g[0][2] = a[0] * b[1] * c[1];
        g[1][2] = -a[1] * b[0] * c[1];
    }
    let mut lap = [0.; D];
    lap[0] = a[2] * b[1] * c[0] + a[0] * b[3] * c[0] + a[0] * b[1] * c[2];
    lap[1] = -a[3] * b[0] * c[0] - a[1] * b[2] * c[0] - a[1] * b[0] * c[2];
    (u, g, lap, x[0] - 0.5)
}
fn manufactured_error<const D: usize>(
    mesh: SimplexMesh<f64, D>,
    nonlinear: bool,
    mu: f64,
) -> (f64, f64, f64) {
    let b = zeros(&mesh);
    let body = |x| {
        let (u, g, lap, _) = manufactured(x);
        Ok(core::array::from_fn(|a| {
            -mu * lap[a]
                + if a == 0 { 1. } else { 0. }
                + if nonlinear {
                    (0..D).map(|j| u[j] * g[a][j]).sum()
                } else {
                    0.
                }
        }))
    };
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        mu,
        &b,
        &body,
        problem::options(),
    )
    .unwrap();
    let solution = if nonlinear {
        let preconditioner = IncompressiblePreconditioner::new(
            &p,
            IncompressibleSaddle::Viscous,
            IncompressiblePreconditionerOptions::default(),
        )
        .unwrap();
        let ns = NavierStokesPlan::new(p);
        let settings = scioxide_rootfind::NewtonKrylovOptions::default()
            .with_residual_tolerance(scioxide_core::ResidualTolerance::new(1e-11, 1e-11).unwrap())
            .with_step_tolerance(scioxide_core::StepTolerance::new(1e-12, 1e-12).unwrap())
            .with_krylov(
                Budget::try_from(4000).unwrap(),
                Budget::try_from(100).unwrap(),
            );
        let result = scioxide_rootfind::NewtonKrylovSolver::new(settings)
            .and_then(|root_plan| {
                root_plan.solve(
                    &scioxide_rootfind::NewtonKrylovProblem::with_jacobian(&ns)
                        .with_preconditioner(&preconditioner),
                    &vec![0.; ns.unknowns()],
                )
            })
            .unwrap();
        assert!(result.is_success(), "{:?}", result.termination());
        (ns.stokes, result.point().to_vec())
    } else {
        let x = solve(&p);
        (p, x)
    };
    let (p, x) = solution;
    let rule = scioxide_integrate::SimplexRule::new(
        D,
        6,
        scioxide_integrate::GeometricOptions::default()
            .with_max_nodes(Budget::try_from(1000).unwrap()),
    )
    .unwrap();
    let mut error = [0.; 3];
    for cell in 0..p.mesh().cell_count() {
        for (bary, &weight) in rule
            .barycentric_nodes()
            .chunks_exact(D + 1)
            .zip(rule.weights())
        {
            let (s, r) = p
                .sample_at(cell, core::array::from_fn(|a| bary[a + 1]), weight)
                .unwrap();
            let (u, g) = p.values_with(&s, &r.solution.values, &x, false).unwrap();
            let (ue, ge, _, pe) = manufactured(s.point);
            for a in 0..D {
                error[0] += s.weight * (u[a] - ue[a]).powi(2);
                for j in 0..D {
                    error[1] += s.weight * (g[a][j] - ge[a][j]).powi(2);
                }
            }
            let pressure = (0..=D)
                .map(|i| r.linear[i] * x[p.velocity_unknowns() + p.space.dofs(cell)[i]])
                .sum::<f64>();
            error[2] += s.weight * (pressure - pe).powi(2);
        }
    }
    let mut work = vec![0.; LinearOperator::apply_workspace_len(&p).unwrap()];
    let d = p.diagnostics(&x, 1., 0.001, &mut work).unwrap();
    assert!(d.weak_divergence < 1e-9);
    assert!(d.boundary_flux < 1e-12);
    assert!(d.gauge_defect < 1e-9);
    println!(
        "manufactured D={D}, nonlinear={nonlinear}, mu={mu}, cells={}, L2={}, H1={}, pressure={}, div={}, Re={}",
        p.mesh().cell_count(),
        error[0].sqrt(),
        error[1].sqrt(),
        error[2].sqrt(),
        d.divergence_l2,
        d.reynolds
    );
    (error[0].sqrt(), error[1].sqrt(), error[2].sqrt())
}
#[test]
fn manufactured_stokes_velocity_pressure_and_divergence_refinement() {
    let a = manufactured_error(
        super::super::super::meshes::triangle_grid(4).unwrap(),
        false,
        1.,
    );
    let b = manufactured_error(
        super::super::super::meshes::triangle_grid(8).unwrap(),
        false,
        1.,
    );
    for (coarse, fine, minimum) in [(a.0, b.0, 2.75), (a.1, b.1, 1.75), (a.2, b.2, 1.75)] {
        assert!((coarse / fine).log2() > minimum);
    }
    let a = manufactured_error(
        super::super::super::meshes::tetrahedron_grid(2).unwrap(),
        false,
        1.,
    );
    let b = manufactured_error(
        super::super::super::meshes::tetrahedron_grid(4).unwrap(),
        false,
        1.,
    );
    for (coarse, fine, minimum) in [(a.0, b.0, 2.0), (a.1, b.1, 1.5), (a.2, b.2, 1.5)] {
        assert!((coarse / fine).log2() > minimum);
    }
}
#[test]
fn manufactured_navier_stokes_reynolds_envelope() {
    for mu in [1., 0.1, 0.01] {
        let a = manufactured_error(
            super::super::super::meshes::triangle_grid(4).unwrap(),
            true,
            mu,
        );
        let b = manufactured_error(
            super::super::super::meshes::triangle_grid(8).unwrap(),
            true,
            mu,
        );
        assert!((a.0 / b.0).log2() > 2.5);
        assert!((a.1 / b.1).log2() > 1.5);
        assert!(b.0 < 1e-4);
    }
    let a = manufactured_error(
        super::super::super::meshes::tetrahedron_grid(2).unwrap(),
        true,
        0.1,
    );
    let b = manufactured_error(
        super::super::super::meshes::tetrahedron_grid(4).unwrap(),
        true,
        0.1,
    );
    assert!((a.0 / b.0).log2() > 2.0);
    assert!((a.1 / b.1).log2() > 1.5);
    assert!(b.0 < 3e-5);
}

#[test]
fn p2_p1_element_moments_are_independent_of_production_quadrature() {
    // Reference triangle: analytic barycentric monomials, not production basis/cubature.
    let mesh = TriangleMesh::new(
        vec![
            Point::new([0., 0.]).unwrap(),
            Point::new([1., 0.]).unwrap(),
            Point::new([0., 1.]).unwrap(),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let boundary = zeros(&mesh);
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        2.,
        3.,
        &boundary,
        &|_| Ok([0.; 2]),
        problem::options(),
    )
    .unwrap();
    type Polynomial = Vec<(f64, [usize; 3])>;
    fn moment(powers: [usize; 3]) -> f64 {
        fn factorial(n: usize) -> f64 {
            (1..=n).fold(1., |s, k| s * k as f64)
        }
        powers.iter().map(|&n| factorial(n)).product::<f64>()
            / factorial(2 + powers.iter().sum::<usize>())
    }
    fn integral(a: &Polynomial, b: &Polynomial) -> f64 {
        a.iter()
            .flat_map(|(x, p)| {
                b.iter()
                    .map(move |(y, q)| x * y * moment(core::array::from_fn(|i| p[i] + q[i])))
            })
            .sum()
    }
    let gradients = [[-1., -1.], [1., 0.], [0., 1.]];
    let mut basis: Vec<Polynomial> = Vec::new();
    let mut derivative: Vec<[Polynomial; 2]> = Vec::new();
    for i in 0..3 {
        let mut one = [0; 3];
        one[i] = 1;
        let mut two = one;
        two[i] = 2;
        basis.push(vec![(2., two), (-1., one)]);
        derivative.push(core::array::from_fn(|a| {
            vec![(4. * gradients[i][a], one), (-gradients[i][a], [0; 3])]
        }));
    }
    for i in 0..3 {
        for j in i + 1..3 {
            let mut powers = [0; 3];
            powers[i] = 1;
            powers[j] = 1;
            basis.push(vec![(4., powers)]);
            let mut ei = [0; 3];
            ei[i] = 1;
            let mut ej = [0; 3];
            ej[j] = 1;
            derivative.push(core::array::from_fn(|a| {
                vec![(4. * gradients[i][a], ej), (4. * gradients[j][a], ei)]
            }));
        }
    }
    let matrix = p.element_matrix(0).unwrap();
    let mass = p.element_velocity_mass(0).unwrap();
    let pressure = p.element_pressure_mass(0).unwrap();
    let m = 15;
    for i in 0..6 {
        for j in 0..6 {
            assert!((mass[i * 6 + j] - 2. * integral(&basis[i], &basis[j])).abs() < 1e-14);
            for a in 0..2 {
                for b in 0..2 {
                    let value = 3.
                        * (if a == b {
                            (0..2)
                                .map(|k| integral(&derivative[i][k], &derivative[j][k]))
                                .sum()
                        } else {
                            0.
                        } + integral(&derivative[i][b], &derivative[j][a]));
                    assert!((matrix[(a * 6 + i) * m + b * 6 + j] - value).abs() < 1e-13);
                }
            }
        }
        for j in 0..3 {
            let mut powers = [0; 3];
            powers[j] = 1;
            for a in 0..2 {
                assert!(
                    (matrix[(a * 6 + i) * m + 12 + j]
                        + integral(&derivative[i][a], &vec![(1., powers)]))
                    .abs()
                        < 1e-14
                );
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            assert!((pressure[i * 3 + j] - if i == j { 1. / 12. } else { 1. / 24. }).abs() < 1e-14);
        }
    }
}
#[test]
fn curved_shear_and_traction_anchored_pressure() {
    let mesh = super::super::super::meshes::triangle_grid::<f64>(2).unwrap();
    let (mesh, edges) = super::super::super::meshes::curved(&mesh).unwrap();
    patch(
        mesh,
        SimplexFemGeometry::Quadratic(edges.view()),
        0.04,
        0.,
        1e-8,
    );
    let mesh = super::super::super::meshes::tetrahedron_grid::<f64>(2).unwrap();
    let (mesh, edges) = super::super::super::meshes::curved(&mesh).unwrap();
    patch(
        mesh,
        SimplexFemGeometry::Quadratic(edges.view()),
        0.04,
        0.,
        1e-7,
    );
    let mesh = super::super::super::meshes::triangle_grid::<f64>(2).unwrap();
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces()
            .map(|face| {
                let right = mesh
                    .face_vertices(face)
                    .unwrap()
                    .iter()
                    .all(|&v| mesh.points()[v].as_array()[0] == 1.);
                (face, u32::from(right))
            })
            .collect(),
    )
    .unwrap();
    let mut conditions = Vec::new();
    for a in 0..2 {
        conditions.push(
            crate::BoundaryCondition::dirichlet(
                0,
                crate::boundary::point_normal(move |x, _| {
                    Ok(if a == 0 { 0.04 * x[1] } else { 0. })
                }),
            )
            .for_component(a),
        );
        conditions.push(
            crate::BoundaryCondition::natural(
                1,
                crate::boundary::point_normal(move |x, n| {
                    Ok(-0.3 * (x[0] - 0.5) * n[a] + 0.04 * n[1 - a])
                }),
            )
            .for_component(a),
        );
    }
    let b = TaggedBoundaries::new(&mesh, &tags, 2, conditions).unwrap();
    let mut o = problem::options();
    o.gauges[0] = EllipticGauge::Unspecified;
    let automatic = StokesPlan::new(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &b,
        &|_| Ok([0.3, 0.]),
    )
    .unwrap();
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &b,
        &|_| Ok([0.3, 0.]),
        o,
    )
    .unwrap();
    assert_eq!(
        automatic.assemble_csr().unwrap().values(),
        p.assemble_csr().unwrap().values()
    );
    assert_eq!(automatic.rhs(), p.rhs());
    let exact = problem::state(&p, 0.04, 0.3).unwrap();
    let solved = solve(&p);
    assert!(solved.iter().zip(exact).all(|(&a, b)| (a - b).abs() < 1e-9));
    // The outflow has nonzero advective momentum flux. Its facet term must
    // cancel the integrated volume term for this exact parallel shear.
    let ns = NavierStokesPlan::new(p);
    let mut residual = vec![0.; ns.unknowns()];
    let mut work = vec![0.; ResidualOperator::residual_workspace_len(&ns).unwrap()];
    ns.residual_into(&solved, &mut residual, &mut work).unwrap();
    assert!(residual.iter().all(|r| r.abs() < 1e-10));
    let direction = (0..solved.len())
        .map(|i| 0.01 * (i as f64).sin())
        .collect::<Vec<_>>();
    let h = 1. / 256.;
    let plus = solved
        .iter()
        .zip(&direction)
        .map(|(&x, &v)| x + h * v)
        .collect::<Vec<_>>();
    let minus = solved
        .iter()
        .zip(&direction)
        .map(|(&x, &v)| x - h * v)
        .collect::<Vec<_>>();
    let mut other = residual.clone();
    let mut product = residual.clone();
    ns.residual_into(&plus, &mut residual, &mut work).unwrap();
    ns.residual_into(&minus, &mut other, &mut work).unwrap();
    ns.jvp_into(&solved, &direction, &mut product, &mut work)
        .unwrap();
    assert!(
        residual
            .iter()
            .zip(&other)
            .zip(&product)
            .all(|((&a, &b), &j)| ((a - b) / (2. * h) - j).abs() < 1e-10)
    );
}
#[test]
fn incompatible_flux_and_disconnected_pressure_gauges() {
    let first = super::super::super::meshes::triangle_grid::<f64>(2).unwrap();
    let nv = first.vertex_count();
    let mut points = first.points().to_vec();
    points.extend(
        first
            .points()
            .iter()
            .map(|p| Point::new([p.as_array()[0] + 2., p.as_array()[1]]).unwrap()),
    );
    let mut cells = first
        .cells()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>();
    cells.extend(first.cells().map(|v| [v[0] + nv, v[1] + nv, v[2] + nv]));
    let mesh = TriangleMesh::new(points, cells).unwrap();
    let b = zeros(&mesh);
    let mut o = problem::options();
    o.gauges = vec![
        EllipticGauge::WeightedMean(2.),
        EllipticGauge::WeightedMean(-3.),
    ];
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &b,
        &|_| Ok([0.; 2]),
        o,
    )
    .unwrap();
    let x = solve(&p);
    for v in 0..2 * nv {
        assert!((x[p.velocity_unknowns() + v] - if v < nv { 2. } else { -3. }).abs() < 1e-10);
    }
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &first,
        first.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let b = TaggedBoundaries::new(
        &first,
        &tags,
        2,
        (0..2)
            .map(|a| {
                crate::BoundaryCondition::dirichlet(
                    0,
                    crate::boundary::point_normal(move |x, _| Ok(x[a])),
                )
                .for_component(a)
            })
            .collect(),
    )
    .unwrap();
    assert!(
        StokesPlan::with_options(
            first,
            SimplexFemGeometry::Affine,
            1.,
            1.,
            &b,
            &|_| Ok([0.; 2]),
            problem::options()
        )
        .is_err()
    );
}

fn library_refinement_iterations<const D: usize>(mesh: SimplexMesh<f64, D>) -> usize {
    let boundaries = zeros(&mesh);
    let p = StokesPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        1.,
        1.,
        &boundaries,
        &|x| {
            let (_, _, lap, _) = manufactured(x);
            Ok(core::array::from_fn(|a| {
                -lap[a] + if a == 0 { 1. } else { 0. }
            }))
        },
        problem::options(),
    )
    .unwrap();
    let inverse = IncompressiblePreconditioner::new(
        &p,
        IncompressibleSaddle::Viscous,
        IncompressiblePreconditionerOptions::default(),
    )
    .unwrap();
    let norm = p.rhs().iter().map(|v| v * v).sum::<f64>().sqrt();
    let rhs = Vector::from_vec(p.rhs().iter().map(|v| v / norm).collect());
    let solver = scioxide_linalg::GmresSolver::new(
        scioxide_core::ResidualTolerance::new(1e-11, 1e-9).unwrap(),
        Budget::try_from(1000).unwrap(),
    )
    .with_restart(Budget::try_from(80).unwrap());
    let result = solver
        .solve_with_options(
            &p,
            &rhs,
            scioxide_linalg::Preconditioning::Inverse(&inverse),
            None,
        )
        .unwrap();
    println!(
        "library refinement D={D}, cells={}, report={:?}",
        p.mesh().cell_count(),
        result.report()
    );
    assert!(result.is_success());
    let mut residual = vec![0.; p.unknowns()];
    p.apply_into(
        result.solution().as_slice(),
        &mut residual,
        &mut vec![0.; LinearOperator::apply_workspace_len(&p).unwrap()],
    )
    .unwrap();
    let defect = residual
        .iter()
        .zip(rhs.as_slice())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(defect <= 1.01e-9);
    result.report().iterations()
}

#[test]
fn library_stokes_iterations_are_bounded_under_uniform_refinement() {
    for counts in [
        [2, 4, 8, 16]
            .map(|n| {
                library_refinement_iterations(
                    super::super::super::meshes::triangle_grid(n).unwrap(),
                )
            })
            .to_vec(),
        [2, 4, 8]
            .map(|n| {
                library_refinement_iterations(
                    super::super::super::meshes::tetrahedron_grid(n).unwrap(),
                )
            })
            .to_vec(),
    ] {
        let minimum = counts.iter().min().unwrap();
        let maximum = counts.iter().max().unwrap();
        // The coarsest 3D pressure space is tiny; allow sixteen extra
        // iterations across the complete sequence, and an absolute ceiling.
        assert!(
            *maximum <= minimum + 16 && *maximum <= 32,
            "iteration counts: {counts:?}"
        );
    }
}

#[test]
fn nullspace_validation_does_not_hide_a_local_defect_under_refinement() {
    for n in [2, 4, 8] {
        let mesh = super::super::super::meshes::triangle_grid::<f64>(n).unwrap();
        let b = zeros(&mesh);
        let mut options = problem::options();
        options.compatibility = Tolerance::new(0., 1e-8).unwrap();
        let mut p = StokesPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            1.,
            1.,
            &b,
            &|_| Ok([0.; 2]),
            options,
        )
        .unwrap();
        let local = p.local();
        let velocity = 2 * p.space.order.nodes(2);
        let mut changed = false;
        for cell in 0..p.mesh().cell_count() {
            if let Some(row) =
                (0..velocity).find(|&row| p.prescribed(p.global(cell, row)).is_none())
            {
                let delta = 1e-6 * (p.mesh().vertex_count() as f64).sqrt();
                p.matrices[cell * local * local + row * local + velocity] += delta;
                p.matrices[cell * local * local + velocity * local + row] += delta;
                changed = true;
                break;
            }
        }
        assert!(changed);
        assert!(ephemeral(|ctx| p.validate_nullspaces(ctx)).is_err());
    }
}
