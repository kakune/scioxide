#![allow(clippy::unwrap_used, clippy::panic)]
use super::*;
use crate::*;
use core::sync::atomic::{AtomicUsize, Ordering};

fn limits() -> SimplexFemLimits {
    SimplexFemLimits::new(1_000_000, 10_000_000, 256, usize::MAX)
}
fn boundary<const D: usize>(
    mesh: &SimplexMesh<f64, D>,
    components: usize,
    fixed: bool,
) -> TaggedBoundaries<f64, f64, D> {
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let condition = if fixed {
        BoundaryCondition::dirichlet(0, BoundaryData::zero())
    } else {
        BoundaryCondition::natural(0, BoundaryData::zero())
    };
    TaggedBoundaries::new(mesh, &tags, components, vec![condition]).unwrap()
}
struct Constant {
    declared: bool,
    calls: AtomicUsize,
}
impl<R: RealScalar, const D: usize> SimplexFemCoefficients<R, R, D> for Constant {
    fn is_spatially_constant(&self) -> bool {
        self.declared
    }
    fn evaluate(&self, _: [R; D], k: &mut [R], c: &mut [R]) -> Result<(), SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let n = k.len().isqrt();
        for (i, value) in k.iter_mut().enumerate() {
            *value = if i / n == i % n {
                R::TWO
            } else {
                R::ONE.scalbn(-3)
            };
        }
        c.fill(R::HALF);
        Ok(())
    }
}
fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            (a - b).abs() <= 2e-12 * a.abs().max(b.abs()).max(1.0),
            "{a} != {b}"
        );
    }
}

#[test]
fn constant_affine_moments_match_sampled_forms_and_admit_once() {
    fn check<const D: usize>(mesh: SimplexMesh<f64, D>) {
        for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
            let bc = boundary(&mesh, 1, false);
            let coefficients = Constant {
                declared: true,
                calls: AtomicUsize::new(0),
            };
            let plan = SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                order,
                &coefficients,
                &bc,
                SimplexFemOptions::new().with_limits(limits()),
            )
            .unwrap();
            assert_eq!(coefficients.calls.load(Ordering::Relaxed), 1);
            let sampled = Constant {
                declared: false,
                calls: AtomicUsize::new(0),
            };
            let reference = SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                order,
                &sampled,
                &bc,
                SimplexFemOptions::new().with_limits(limits()),
            )
            .unwrap();
            assert!(sampled.calls.load(Ordering::Relaxed) > 1);
            for cell in 0..mesh.cell_count() {
                close(
                    plan.element_matrix(cell).unwrap(),
                    reference.element_matrix(cell).unwrap(),
                );
                close(
                    plan.element_mass(cell).unwrap(),
                    reference.element_mass(cell).unwrap(),
                );
            }
        }
    }
    check(meshes::triangle_grid(2).unwrap());
    check(meshes::tetrahedron_grid(1).unwrap());
    struct Incomplete;
    impl SimplexFemCoefficients<f64, f64, 2> for Incomplete {
        fn is_spatially_constant(&self) -> bool {
            true
        }
        fn evaluate(&self, _: [f64; 2], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
            k[0] = 1.0;
            c.fill(0.0);
            Ok(())
        }
    }
    let mesh = meshes::triangle_grid(1).unwrap();
    let bc = boundary(&mesh, 1, false);
    assert!(
        SimplexFemPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P1,
            &Incomplete,
            &bc,
            SimplexFemOptions::new().with_limits(limits())
        )
        .is_err()
    );
}

#[test]
fn geometry_range_fallback_and_affine_forms_agree() {
    for scale in [2.0_f64.powi(-530), 2.0_f64.powi(500)] {
        let basis = ReferenceBasis::<f64, 2>::new(SimplexFemOrder::P1).unwrap();
        let map = reference::map_coordinates(
            &[[0.0; 2], [scale, 0.0], [0.0, scale]],
            0.0,
            &basis.sample([0.25; 2]).unwrap(),
            false,
        )
        .unwrap();
        assert!(map.determinant > 0.0 && map.determinant.is_finite());
        assert!((map.inverse_jacobian[0][0] * scale - 1.0).abs() < 4.0 * f64::EPSILON);
        assert!((map.point[0] / scale - 0.25).abs() < f64::EPSILON);
        let mesh = scioxide_spatial::TriangleMesh::new(
            vec![
                scioxide_spatial::Point::new([0.0; 2]).unwrap(),
                scioxide_spatial::Point::new([scale, 0.0]).unwrap(),
                scioxide_spatial::Point::new([0.0, scale]).unwrap(),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let bc = boundary(&mesh, 1, false);
        let make = |declared| {
            SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                SimplexFemOrder::P1,
                &Constant {
                    declared,
                    calls: AtomicUsize::new(0),
                },
                &bc,
                SimplexFemOptions::new().with_limits(limits()),
            )
            .unwrap()
        };
        close(
            make(true).element_matrix(0).unwrap(),
            make(false).element_matrix(0).unwrap(),
        );
    }
}

#[test]
fn constant_f32_and_complex_forms_match_sampling() {
    let mesh = meshes::tetrahedron_grid::<f32>(1).unwrap();
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let bc = TaggedBoundaries::new(
        &mesh,
        &tags,
        2,
        vec![BoundaryCondition::natural(0, BoundaryData::zero())],
    )
    .unwrap();
    let make = |declared| {
        SimplexFemPlan::with_options(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &Constant {
                declared,
                calls: AtomicUsize::new(0),
            },
            &bc,
            SimplexFemOptions::new().with_limits(limits()),
        )
        .unwrap()
    };
    let a = make(true);
    let b = make(false);
    for cell in 0..mesh.cell_count() {
        for (&a, &b) in a
            .element_matrix(cell)
            .unwrap()
            .iter()
            .zip(b.element_matrix(cell).unwrap())
        {
            assert!((a - b).abs() < 4e-6 * a.abs().max(b.abs()).max(1.0));
        }
    }
    struct ComplexConstant(bool);
    impl SimplexFemCoefficients<f64, scioxide_core::Complex64, 2> for ComplexConstant {
        fn is_spatially_constant(&self) -> bool {
            self.0
        }
        fn evaluate(
            &self,
            _: [f64; 2],
            k: &mut [scioxide_core::Complex64],
            c: &mut [scioxide_core::Complex64],
        ) -> Result<(), SciOxideError> {
            use scioxide_core::Complex64 as C;
            k.copy_from_slice(&[
                C::new(2.0, 0.0),
                C::new(0.2, 0.3),
                C::new(0.2, -0.3),
                C::new(3.0, 0.0),
            ]);
            c.fill(C::new(0.5, 0.25));
            Ok(())
        }
    }
    let mesh = meshes::triangle_grid::<f64>(2).unwrap();
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, 0)).collect(),
    )
    .unwrap();
    let bc = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![BoundaryCondition::natural(0, BoundaryData::zero())],
    )
    .unwrap();
    let make = |constant| {
        SimplexFemPlan::with_options(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &ComplexConstant(constant),
            &bc,
            SimplexFemOptions::new().with_limits(limits()),
        )
        .unwrap()
    };
    let a = make(true).assemble_csr().unwrap();
    let b = make(false).assemble_csr().unwrap();
    assert_eq!(a.structural_pattern(), b.structural_pattern());
    for (&a, &b) in a.values().iter().zip(b.values()) {
        assert!((a - b).abs() < 2e-12);
    }
}

#[cfg(feature = "std")]
#[test]
fn parallel_prepared_tangents_and_linear_assembly_match_sequential() {
    let workers = core::num::NonZeroUsize::new(3).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let execution = ExecutionContext::new(&pool);
    let mesh = meshes::triangle_grid(2).unwrap();
    let bc = boundary(&mesh, 2, true);
    let options = ElasticityOptions::new(
        ElasticityKinematics::PlaneStress,
        ElasticityFormulation::MixedP2P1,
    )
    .with_fem(
        SimplexFemOptions::new()
            .with_limits(limits())
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    );
    let plan = HyperelasticityPlan::build(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(1.0, 10.0).unwrap(),
        &bc,
        &|_| Ok([0.0; 2]),
        options.clone(),
        execution,
    )
    .unwrap();
    let x = vec![0.001; plan.unknowns()];
    let mut a = plan.linearization_workspace().unwrap();
    let mut b = plan.linearization_workspace().unwrap();
    let expected = plan
        .linearize(&x, &mut a, ExecutionContext::sequential_native())
        .unwrap();
    let actual = plan.linearize(&x, &mut b, execution).unwrap();
    close(expected.matrix().values(), actual.matrix().values());
    let linear = LinearElasticityPlan::build(
        mesh,
        SimplexFemGeometry::Affine,
        IsotropicElasticMaterial::new(1.0, 10.0).unwrap(),
        &bc,
        &|_| Ok([0.0; 2]),
        options,
        execution,
    )
    .unwrap();
    let mut matrix = linear.assemble_csr().unwrap();
    let mut workspace = linear.assembly_workspace().unwrap();
    let expected = matrix.values().to_vec();
    linear
        .assemble_csr_into(&mut matrix, &mut workspace, execution)
        .unwrap();
    close(matrix.values(), &expected);
    let mesh = meshes::triangle_grid(2).unwrap();
    let bc = boundary(&mesh, 2, true);
    let plan = NavierStokesPlan::new(
        StokesPlan::build(
            mesh,
            SimplexFemGeometry::Affine,
            1.0,
            1.0,
            &bc,
            &|_| Ok([0.0; 2]),
            IncompressibleOptions::new()
                .with_fem(
                    SimplexFemOptions::new()
                        .with_limits(limits())
                        .with_execution_policy(ExecutionPolicy::Parallel(workers)),
                )
                .with_gauges(vec![EllipticGauge::Orthogonal]),
            execution,
        )
        .unwrap(),
    );
    let x = vec![0.001; plan.unknowns()];
    let mut a = plan.linearization_workspace().unwrap();
    let mut b = plan.linearization_workspace().unwrap();
    let expected = plan
        .linearize(&x, &mut a, ExecutionContext::sequential_native())
        .unwrap();
    let actual = plan.linearize(&x, &mut b, execution).unwrap();
    close(expected.matrix().values(), actual.matrix().values());
}

#[test]
fn small_and_block_spd_admission_preserves_the_rounding_contract() {
    for n in 1..=6 {
        for scale in [f64::from_bits(1), 1.0, f64::MAX] {
            let mut tensor = vec![0.0; n * n];
            for i in 0..n {
                tensor[i * n + i] = scale;
            }
            let mut scratch = vec![f64::NAN; n * n];
            crate::validation::positive_tensor(&mut tensor, &mut scratch, n).unwrap();
            tensor[0] = -scale;
            assert!(crate::validation::positive_tensor(&mut tensor, &mut scratch, n).is_err());
        }
    }
    let mut tensor = vec![1.0, 1.0, 1.0, 1.0];
    assert!(crate::validation::positive_tensor(&mut tensor, &mut [0.0; 4], 2).is_err());
}

#[test]
fn default_quadrature_admits_quadratic_tetrahedral_geometry() {
    let mesh = meshes::tetrahedron_grid::<f64>(1).unwrap();
    let (mesh, edges) = meshes::curved(&mesh).unwrap();
    let boundaries = boundary(&mesh, 1, true);
    let coefficients = crate::ConstantDiffusion::new(1.);
    let plain = SimplexFemPlan::new(
        mesh.clone(),
        SimplexFemGeometry::Quadratic(edges.view()),
        SimplexFemOrder::P2,
        &coefficients,
        &boundaries,
    )
    .unwrap();
    let configured = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Quadratic(edges.view()),
        SimplexFemOrder::P2,
        &coefficients,
        &boundaries,
        SimplexFemOptions::default(),
    )
    .unwrap();
    assert_eq!(
        plain.assemble_csr().unwrap().values(),
        configured.assemble_csr().unwrap().values()
    );
}
