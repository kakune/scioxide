#![allow(clippy::unwrap_used, clippy::panic)]
use super::super::meshes;
use super::*;
use alloc::vec;
use scioxide_core::Complex;
use scioxide_linalg::{
    BorderedSystem, BorderedSystemWorkspaceRequest, CompatibilityPolicy, Vector,
};
#[path = "manufactured.rs"]
mod manufactured;
#[path = "verification.rs"]
mod verification;

fn limits() -> SimplexFemLimits {
    SimplexFemLimits::new(100_000, 40_000_000, 256, 1_000_000_000)
}
fn formulated<R: RealScalar, const D: usize>(
    formulation: ElasticityFormulation,
) -> ElasticityOptions<R> {
    ElasticityOptions::new(
        if D == 2 {
            ElasticityKinematics::PlaneStrain
        } else {
            ElasticityKinematics::ThreeDimensional
        },
        formulation,
    )
    .with_fem(SimplexFemOptions::new().with_limits(limits()))
}
fn options<R: RealScalar, const D: usize>(mixed: bool) -> ElasticityOptions<R> {
    formulated::<R, D>(if mixed {
        ElasticityFormulation::MixedP2P1
    } else {
        ElasticityFormulation::Displacement(SimplexFemOrder::P2)
    })
}
fn natural<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: &SimplexMesh<R, D>,
) -> TaggedBoundaries<R, V, D> {
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    TaggedBoundaries::new(
        mesh,
        &tags,
        D,
        (0..if mesh.is_empty() { 0 } else { D })
            .map(|component| {
                crate::BoundaryCondition::natural(
                    0,
                    crate::boundary::point_normal(|_, _| Ok(V::ZERO)),
                )
                .for_component(component)
            })
            .collect(),
    )
    .unwrap()
}
fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn norm<V: NumericScalar>(x: &[V]) -> f64 {
    x.iter().fold(0.0_f64, |a, v| a.max(v.abs().into_f64()))
}

fn positive_elastic_forms<const D: usize>(input: SimplexMesh<f64, D>) {
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        for curved in [false, true] {
            if curved && order == SimplexFemOrder::P1 {
                continue;
            }
            let (warped, edges) = meshes::curved(&input).unwrap();
            let mesh = if curved { warped } else { input.clone() };
            let boundaries = natural::<f64, f64, D>(&mesh);
            let options = formulated::<f64, D>(ElasticityFormulation::Displacement(order));
            let plan = LinearElasticityPlan::new(
                mesh,
                if curved {
                    SimplexFemGeometry::Quadratic(edges.view())
                } else {
                    SimplexFemGeometry::Affine
                },
                IsotropicElasticMaterial::new(2., 7.).unwrap(),
                &boundaries,
                &|_| Ok([0.; D]),
                options,
            )
            .unwrap();
            let csr = plan.assemble_csr().unwrap();
            let mut dense = vec![0.; csr.rows() * csr.rows()];
            csr.to_dense_into(&mut dense).unwrap();
            super::super::tests::assert_psd(&dense, csr.rows());
            assert!(volume_samples(&plan.space).iter().all(|s| s.weight > 0.));
        }
    }
}

#[test]
fn positive_quadrature_preserves_elasticity_semidefiniteness() {
    positive_elastic_forms(meshes::triangle_grid::<f64>(1).unwrap());
    positive_elastic_forms(meshes::tetrahedron_grid::<f64>(1).unwrap());
}

fn patch<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: SimplexMesh<R, D>,
    plane: ElasticityKinematics,
    mixed: bool,
    curved: bool,
) {
    let material = IsotropicElasticMaterial::new(r(2.), r(7.)).unwrap();
    let (warped, edges) = meshes::curved(&mesh).unwrap();
    let mesh = if curved { warped } else { mesh };
    let amplitude = V::from_components(R::ONE, r(0.25));
    let h: [[R; 3]; 3] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            if i < D && j < D {
                r((1 + i + 2 * j) as f64 / 32.)
            } else {
                R::ZERO
            }
        })
    });
    let mut strain: [[R; 3]; 3] =
        core::array::from_fn(|i| core::array::from_fn(|j| (h[i][j] + h[j][i]) * R::HALF));
    let lambda = r::<R>(7. - 4. / 3.);
    if plane == ElasticityKinematics::PlaneStress {
        strain[2][2] = -lambda * (strain[0][0] + strain[1][1]) / (lambda + r(4.));
    }
    let trace = (0..3).fold(R::ZERO, |s, i| s + strain[i][i]);
    let sigma: [[R; 3]; 3] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            r::<R>(4.) * strain[i][j] + if i == j { lambda * trace } else { R::ZERO }
        })
    });
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let boundary = TaggedBoundaries::new(
        &mesh,
        &tags,
        D,
        (0..D)
            .map(|component| {
                crate::BoundaryCondition::natural(
                    0,
                    crate::boundary::point_normal(move |_, n| {
                        Ok(amplitude
                            * V::from_real(
                                (0..D).fold(R::ZERO, |s, j| s + sigma[component][j] * n[j]),
                            ))
                    }),
                )
                .for_component(component)
            })
            .collect(),
    )
    .unwrap();
    let mut opts = options::<R, D>(mixed);
    opts.kinematics = plane;
    let plan = LinearElasticityPlan::new(
        mesh,
        if curved {
            SimplexFemGeometry::Quadratic(edges.view())
        } else {
            SimplexFemGeometry::Affine
        },
        material,
        &boundary,
        &|_| Ok([V::ZERO; D]),
        opts,
    )
    .unwrap();
    let mut x = vec![V::ZERO; plan.unknowns()];
    for a in 0..D {
        for i in 0..plan.displacement_nodes() {
            let p = plan.node_point(i).unwrap();
            x[a * plan.displacement_nodes() + i] = amplitude
                * V::from_real(r::<R>(0.1) + (0..D).fold(R::ZERO, |s, j| s + h[a][j] * p[j]));
        }
    }
    if mixed {
        x[D * plan.displacement_nodes()..].fill(amplitude * V::from_real(r::<R>(7.) * trace));
    }
    let mut y = x.clone();
    let mut work = vec![V::ZERO; LinearOperator::apply_workspace_len(&plan).unwrap()];
    plan.apply_into(&x, &mut y, &mut work).unwrap();
    let error = norm(
        &y.iter()
            .zip(plan.rhs())
            .map(|(&a, &b)| a - b)
            .collect::<Vec<_>>(),
    );
    assert!(
        error < R::EPSILON.into_f64() * 4000.,
        "patch D={D} mixed={mixed} curved={curved}: {error:e}"
    );
    let s = plan.stress(0, [r(0.125); D], &x).unwrap();
    for (actual_row, expected_row) in s.cauchy.iter().zip(&sigma) {
        for (&actual, &expected) in actual_row.iter().zip(expected_row) {
            assert!(
                (actual - amplitude * V::from_real(expected))
                    .abs()
                    .into_f64()
                    < R::EPSILON.into_f64() * 2000.
            );
        }
    }
    let csr = plan.assemble_csr().unwrap();
    let dense = dense(&csr);
    let product = dense.view().mul_vec(&Vector::from_vec(x.clone())).unwrap();
    assert!(
        norm(
            &product
                .as_slice()
                .iter()
                .zip(&y)
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>()
        ) < R::EPSILON.into_f64() * 5000.
    );
    let modes = plan
        .rigid_modes(Tolerance::new(r(1e-5), r(1e-5)).unwrap(), 6)
        .unwrap();
    assert_eq!(modes.nullity(), D * (D + 1) / 2);
    for k in 0..modes.nullity() {
        plan.apply_into(modes.basis().unwrap().column(k).unwrap(), &mut y, &mut work)
            .unwrap();
        assert!(norm(&y) < R::EPSILON.into_f64() * 5000.);
    }
    let energy = plan.energy(&x).unwrap();
    let density = (0..3)
        .flat_map(|i| (0..3).map(move |j| strain[i][j] * sigma[i][j] * R::HALF))
        .fold(R::ZERO, |s, x| s + x)
        * amplitude.norm_sqr();
    let volume = volume_samples(&plan.space)
        .iter()
        .fold(R::ZERO, |s, q| s + q.weight);
    assert!((energy.internal - density * volume).abs().into_f64() < R::EPSILON.into_f64() * 10000.);
}
#[test]
fn linear_patches_all_scalars_dimensions_reductions_and_geometry() {
    for mixed in [false, true] {
        for curved in [false, true] {
            for plane in [
                ElasticityKinematics::PlaneStrain,
                ElasticityKinematics::PlaneStress,
            ] {
                patch::<f64, f64, 2>(meshes::triangle_grid(1).unwrap(), plane, mixed, curved);
                patch::<f32, f32, 2>(meshes::triangle_grid(1).unwrap(), plane, mixed, curved);
                patch::<f64, Complex<f64>, 2>(
                    meshes::triangle_grid(1).unwrap(),
                    plane,
                    mixed,
                    curved,
                );
                patch::<f32, Complex<f32>, 2>(
                    meshes::triangle_grid(1).unwrap(),
                    plane,
                    mixed,
                    curved,
                );
            }
            patch::<f64, f64, 3>(
                meshes::tetrahedron_grid(1).unwrap(),
                ElasticityKinematics::ThreeDimensional,
                mixed,
                curved,
            );
            patch::<f32, Complex<f32>, 3>(
                meshes::tetrahedron_grid(1).unwrap(),
                ElasticityKinematics::ThreeDimensional,
                mixed,
                curved,
            );
            patch::<f32, f32, 3>(
                meshes::tetrahedron_grid(1).unwrap(),
                ElasticityKinematics::ThreeDimensional,
                mixed,
                curved,
            );
            patch::<f64, Complex<f64>, 3>(
                meshes::tetrahedron_grid(1).unwrap(),
                ElasticityKinematics::ThreeDimensional,
                mixed,
                curved,
            );
        }
    }
}

fn nonlinear<R: RealScalar, const D: usize>(
    mesh: SimplexMesh<R, D>,
    plane: ElasticityKinematics,
    mixed: bool,
    curved: bool,
) {
    let (mapped, edges) = meshes::curved(&mesh).unwrap();
    let mesh = if curved { mapped } else { mesh };
    let boundary = natural(&mesh);
    let mut opts = options::<R, D>(mixed);
    opts.kinematics = plane;
    let plan = HyperelasticityPlan::new(
        mesh,
        if curved {
            SimplexFemGeometry::Quadratic(edges.view())
        } else {
            SimplexFemGeometry::Affine
        },
        IsotropicElasticMaterial::new(r(2.), r(7.)).unwrap(),
        &boundary,
        &|_| Ok([R::ZERO; D]),
        opts,
    )
    .unwrap();
    let n = plan.unknowns();
    let mut x = vec![R::ZERO; n];
    for a in 0..D {
        for i in 0..plan.displacement_nodes() {
            let p = plan.node_point(i).unwrap();
            x[a * plan.displacement_nodes() + i] = r::<R>(0.015) * (R::ONE + p[a] * p[(a + 1) % D]);
        }
    }
    if mixed {
        x[D * plan.displacement_nodes()..].fill(r(0.2));
    }
    let v = (0..n)
        .map(|i| r::<R>(((i % 7) as f64 - 3.) / 100.))
        .collect::<Vec<_>>();
    let mut out = x.clone();
    let mut jv = x.clone();
    let mut work = vec![R::ZERO; ResidualOperator::residual_workspace_len(&plan).unwrap()];
    plan.residual_into(&x, &mut out, &mut work).unwrap();
    plan.jvp_into(&x, &v, &mut jv, &mut work).unwrap();
    let h: R = if R::EPSILON.into_f64() > 1e-10 {
        r(0.003)
    } else {
        r(0.00001)
    };
    let plus = x
        .iter()
        .zip(&v)
        .map(|(&x, &v)| x + h * v)
        .collect::<Vec<_>>();
    let minus = x
        .iter()
        .zip(&v)
        .map(|(&x, &v)| x - h * v)
        .collect::<Vec<_>>();
    let mut rp = x.clone();
    let mut rm = x.clone();
    plan.residual_into(&plus, &mut rp, &mut work).unwrap();
    plan.residual_into(&minus, &mut rm, &mut work).unwrap();
    let tolerance = if R::EPSILON.into_f64() > 1e-10 {
        0.004
    } else {
        2e-7
    };
    let error = rp
        .iter()
        .zip(&rm)
        .zip(&jv)
        .fold(0.0_f64, |s, ((&p, &m), &j)| {
            s.max(((p - m) / (h + h) - j).abs().into_f64())
        });
    assert!(
        error < tolerance,
        "JVP {plane:?} mixed={mixed} D={D}: {error:e}"
    );
    let energy_diff =
        (plan.energy(&plus).unwrap().potential - plan.energy(&minus).unwrap().potential) / (h + h);
    let work_dot = out.iter().zip(&v).fold(R::ZERO, |s, (&a, &b)| s + a * b);
    assert!(
        (energy_diff - work_dot).abs().into_f64() < tolerance,
        "energy {plane:?} {energy_diff:?} {work_dot:?}"
    );
    let matrix = dense(&plan.assemble_tangent_csr(&x).unwrap());
    let mv = matrix.view().mul_vec(&Vector::from_vec(v)).unwrap();
    assert!(
        norm(
            &mv.as_slice()
                .iter()
                .zip(&jv)
                .map(|(&a, &b)| a - b)
                .collect::<Vec<_>>()
        ) < R::EPSILON.into_f64() * 5000.
    );
    // Identity has zero stress; a proper rigid rotation has zero stored energy.
    x.fill(R::ZERO);
    let angle = r::<R>(0.3);
    for i in 0..plan.displacement_nodes() {
        let p = plan.node_point(i).unwrap();
        x[i] = (angle.cos() - R::ONE) * p[0] - angle.sin() * p[1];
        x[plan.displacement_nodes() + i] = angle.sin() * p[0] + (angle.cos() - R::ONE) * p[1];
    }
    assert!(plan.energy(&x).unwrap().internal.abs().into_f64() < R::EPSILON.into_f64() * 5000.);
    plan.residual_into(&x, &mut out, &mut work).unwrap();
    assert!(norm(&out) < R::EPSILON.into_f64() * 5000.);
}
#[test]
fn neo_hookean_energy_tangents_objectivity_both_precisions() {
    for curved in [false, true] {
        for mixed in [false, true] {
            for plane in [
                ElasticityKinematics::PlaneStress,
                ElasticityKinematics::PlaneStrain,
            ] {
                nonlinear::<f64, 2>(meshes::triangle_grid(1).unwrap(), plane, mixed, curved);
                nonlinear::<f32, 2>(meshes::triangle_grid(1).unwrap(), plane, mixed, curved);
            }
            nonlinear::<f64, 3>(
                meshes::tetrahedron_grid(1).unwrap(),
                ElasticityKinematics::ThreeDimensional,
                mixed,
                curved,
            );
            nonlinear::<f32, 3>(
                meshes::tetrahedron_grid(1).unwrap(),
                ElasticityKinematics::ThreeDimensional,
                mixed,
                curved,
            );
        }
    }
}

#[test]
fn independent_p1_matrix_and_mixed_pressure_moments() {
    let mesh = scioxide_spatial::TriangleMesh::new(
        vec![
            scioxide_spatial::Point::new([0., 0.]).unwrap(),
            scioxide_spatial::Point::new([1., 0.]).unwrap(),
            scioxide_spatial::Point::new([0., 1.]).unwrap(),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let bc = natural(&mesh);
    let m = IsotropicElasticMaterial::new(2., 7.).unwrap();
    let opt = formulated::<f64, 2>(ElasticityFormulation::Displacement(SimplexFemOrder::P1));
    let plan = LinearElasticityPlan::new(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        m,
        &bc,
        &|_| Ok([0.; 2]),
        opt,
    )
    .unwrap();
    let matrix = plan.element_matrix(0).unwrap();
    let g = [[-1., -1.], [1., 0.], [0., 1.]];
    for a in 0..2 {
        for b in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    let expected = 0.5
                        * (if a == b {
                            2. * (g[i][0] * g[j][0] + g[i][1] * g[j][1])
                        } else {
                            0.
                        } + 2. * g[i][b] * g[j][a]
                            + (7. - 4. / 3.) * g[i][a] * g[j][b]);
                    assert!((matrix[(a * 3 + i) * 6 + b * 3 + j] - expected).abs() < 2e-13);
                }
            }
        }
    }
    let mixed = LinearElasticityPlan::new(
        mesh,
        SimplexFemGeometry::Affine,
        m,
        &bc,
        &|_| Ok([0.; 2]),
        options::<f64, 2>(true),
    )
    .unwrap();
    let matrix = mixed.element_matrix(0).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = -if i == j { 1. / 12. } else { 1. / 24. } / 7.;
            assert!((matrix[(12 + i) * 15 + 12 + j] - expected).abs() < 1e-15);
        }
    }
}

#[test]
fn rigid_modes_and_incompatible_loads_use_shared_contract() {
    let mesh = meshes::triangle_grid::<f64>(1).unwrap();
    let bc = natural(&mesh);
    let plan = LinearElasticityPlan::new(
        mesh,
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(1., 10.).unwrap(),
        &bc,
        &|_| Ok([1., 0.]),
        options::<f64, 2>(true),
    )
    .unwrap();
    let modes = plan
        .rigid_modes(Tolerance::new(1e-10, 1e-10).unwrap(), 3)
        .unwrap();
    let nullspace = BorderedSystem::new(
        &plan,
        modes.basis().unwrap(),
        modes.basis().unwrap(),
        100.,
        &scioxide_linalg::BorderedSystemOptions::new()
            .with_validation_tolerance(Tolerance::new(1e-9, 1e-9).unwrap())
            .with_compatibility_tolerance(Tolerance::new(1e-9, 1e-9).unwrap()),
    )
    .unwrap();
    let n = plan.unknowns();
    let mut out = vec![0.; n + 3];
    let mut work = nullspace
        .workspace(BorderedSystemWorkspaceRequest::PrepareRhs)
        .unwrap();
    assert!(
        nullspace
            .prepare_rhs_into(
                scioxide_linalg::VectorView::from_slice(plan.load()),
                scioxide_linalg::VectorView::from_slice(&[0.; 3]),
                CompatibilityPolicy::RequireCompatible,
                scioxide_linalg::VectorViewMut::from_slice(&mut out),
                &mut work,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
}

fn dense<V: NumericScalar>(csr: &CsrMatrix<V>) -> scioxide_linalg::Matrix<V> {
    let mut m = scioxide_linalg::Matrix::zeros(csr.rows(), csr.cols()).unwrap();
    csr.to_dense_into(m.as_mut_slice()).unwrap();
    m
}

#[test]
fn discontinuous_displacement_uses_shared_dirichlet_precedence() {
    let mesh = meshes::triangle_grid::<f64>(2).unwrap();
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces()
            .map(|face| (face, face as u32))
            .collect(),
    )
    .unwrap();
    let boundary = TaggedBoundaries::new(
        &mesh,
        &tags,
        2,
        mesh.boundary_faces()
            .flat_map(|face| {
                (0..2).map(move |component| {
                    crate::BoundaryCondition::dirichlet(
                        face as u32,
                        crate::boundary::point_normal(move |_, _| Ok((face + component) as f64)),
                    )
                    .for_component(component)
                })
            })
            .collect(),
    )
    .unwrap();
    let material = IsotropicElasticMaterial::new(1., 10.).unwrap();
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        let config = formulated::<f64, 2>(ElasticityFormulation::Displacement(order));
        assert!(
            LinearElasticityPlan::new(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                material,
                &boundary,
                &|_| Ok([0.; 2]),
                config
            )
            .is_err()
        );
    }
    let boundary =
        boundary.with_dirichlet_conflict_policy(crate::DirichletConflictPolicy::LastConditionWins);
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        let config = formulated::<f64, 2>(ElasticityFormulation::Displacement(order));
        let p = LinearElasticityPlan::new(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            material,
            &boundary,
            &|_| Ok([0.; 2]),
            config,
        )
        .unwrap();
        for vertex in 0..mesh.vertex_count() {
            let winner = mesh
                .boundary_faces()
                .filter(|&f| mesh.face_vertices(f).unwrap().contains(&vertex))
                .max();
            for component in 0..2 {
                assert_eq!(
                    p.space.prescribed(component * p.space.space.nodes + vertex),
                    winner.map(|face| (face + component) as f64)
                );
            }
        }
    }
}

/// Mapped volume samples of every cell of a prepared elasticity space.
fn volume_samples<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &Space<R, V, D>,
) -> Vec<Sample<R, D>> {
    (0..space.space.cell_count())
        .flat_map(|cell| {
            space
                .space
                .volume_nodes()
                .map(move |reference| space.space.sample(cell, reference).unwrap())
        })
        .collect()
}
