#![allow(clippy::unwrap_used, clippy::panic)]

use super::meshes;
use super::*;
use crate::SpatialDiscretization;
use crate::boundary::{
    BoundaryCondition, DirichletConflictPolicy, FieldData, FieldPoint, point_normal,
};
use scioxide_spatial::SimplexBoundaryTags;

/// Source data from a pointwise per-component callback.
fn source<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>(
    f: impl Fn([R; D], usize) -> V + Send + Sync + 'static,
) -> FieldData<R, V, D> {
    FieldData::time_invariant(move |p: FieldPoint<R, D>| Ok(f(p.point, p.component)))
}
#[cfg(feature = "std")]
use scioxide_core::AlgebraicScalar;
use scioxide_linalg::Preconditioning;
use scioxide_spatial::{Point, TetrahedronMesh, TriangleMesh};

const TEST_ARGUMENT: scioxide_core::Argument = scioxide_core::argument!("fixture");

/// Apply `operator` inside one epoch of `execution`, as an enclosing solver would.
fn apply_scoped<V: NumericScalar>(
    operator: &dyn LinearOperator<V>,
    input: &[V],
    output: &mut [V],
    workspace: &mut [V],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    use scioxide_core::{
        OperatorContext, ProblemToken, SolveBudgets, SolveIdentity, SolveScope, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "apply", 1),
        0,
        0,
    );
    SolveScope::from(execution).run(
        &mut SolveState::new(),
        identity,
        SolveBudgets::unbounded(),
        |context| {
            operator
                .apply_into_scoped(input, output, workspace, &mut OperatorContext::new(context))?
                .require()
        },
    )
}

fn limits() -> SimplexFemLimits {
    SimplexFemLimits::new(100_000, 30_000_000, 256, 1_000_000_000)
}

pub(super) fn assert_psd(values: &[f64], n: usize) {
    let scale = values.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
    for row in 0..n {
        for col in 0..row {
            assert!((values[row * n + col] - values[col * n + row]).abs() < 1e-12 * scale.max(1.));
        }
    }
    let matrix = scioxide_linalg::Matrix::from_vec(n, n, values.to_vec()).unwrap();
    let eigen = scioxide_linalg::HermitianEigenvaluesResult::compute(
        &matrix,
        scioxide_linalg::HermitianInput::Triangle(scioxide_core::Triangle::Lower),
    )
    .unwrap();
    assert!(
        eigen
            .eigenvalues()
            .iter()
            .all(|&v| v >= -2048. * f64::EPSILON * scale.max(1.)),
        "eigenvalues {:?}",
        eigen.eigenvalues()
    );
}

struct PositiveBump;
impl<const D: usize> SimplexFemCoefficients<f64, f64, D> for PositiveBump {
    fn evaluate(&self, point: [f64; D], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
        let center = 1. / (D + 3) as f64;
        let distance: f64 = point.iter().map(|x| (x - center).powi(2)).sum();
        let value = 1. + 10000. * (-distance / 0.03_f64.powi(2)).exp();
        k.fill(0.);
        for i in 0..D {
            k[i * D + i] = value;
        }
        c.fill(value);
        Ok(())
    }
}

fn bump_psd<const D: usize>(input: SimplexMesh<f64, D>) {
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        for curved in [false, true] {
            let (warped, edges) = meshes::curved(&input).unwrap();
            let mesh = if curved { warped } else { input.clone() };
            let boundary = natural::<f64, f64, D>(&mesh, 1);
            let geometry = if curved {
                SimplexFemGeometry::Quadratic(edges.view())
            } else {
                SimplexFemGeometry::Affine
            };
            let plan = SimplexFemPlan::with_options(
                mesh,
                geometry,
                order,
                &PositiveBump,
                &boundary,
                SimplexFemOptions::new().with_limits(limits()),
            )
            .unwrap();
            for cell in 0..plan.mesh().cell_count() {
                assert_psd(plan.element_matrix(cell).unwrap(), order.nodes(D));
                assert_psd(plan.element_mass(cell).unwrap(), order.nodes(D));
            }
            let n = plan.unknowns();
            let mut dense = vec![0.; n * n];
            plan.assemble_csr()
                .unwrap()
                .to_dense_into(&mut dense)
                .unwrap();
            assert_psd(&dense, n);
            assert!(
                plan.space
                    .options
                    .volume_rule()
                    .unwrap()
                    .weights()
                    .iter()
                    .all(|&w| w > 0.)
            );
            assert!(
                plan.space
                    .options
                    .facet_rule()
                    .unwrap()
                    .weights()
                    .iter()
                    .all(|&w| w > 0.)
            );
            if !curved {
                assert_eq!(
                    plan.space.options.volume_rule().unwrap().node_count(),
                    match (D, order) {
                        (2, SimplexFemOrder::P1) => 3,
                        (2, _) => 6,
                        (3, SimplexFemOrder::P1) => 4,
                        _ => 11,
                    }
                );
            }
        }
    }
}

#[test]
fn localized_positive_coefficients_have_psd_elements_and_global_forms() {
    bump_psd(triangle::<f64>());
    bump_psd(tetrahedron::<f64>());
    bump_psd(meshes::triangle_grid::<f64>(1).unwrap());
    bump_psd(meshes::tetrahedron_grid::<f64>(1).unwrap());
}

#[test]
fn rotated_and_rounding_asymmetric_diffusion_is_admitted_and_projected() {
    struct Rotated(f64);
    impl SimplexFemCoefficients<f64, f64, 2> for Rotated {
        fn evaluate(&self, _: [f64; 2], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
            let (s, t) = self.0.sin_cos();
            k.copy_from_slice(&[
                t * t + 10. * s * s,
                -9. * s * t,
                -9. * s * t + 8. * f64::EPSILON,
                s * s + 10. * t * t,
            ]);
            c.fill(0.);
            Ok(())
        }
    }
    for angle in (0..=64).map(|i| i as f64 * 0.1) {
        let mesh = triangle::<f64>();
        let boundaries = natural::<f64, f64, 2>(&mesh, 1);
        let plan = SimplexFemPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &Rotated(angle),
            &boundaries,
            SimplexFemOptions::new().with_limits(limits()),
        )
        .unwrap();
        assert_psd(plan.element_matrix(0).unwrap(), 6);
    }
    for mut tensor in [[1., 2., 2., 1.], [1., 0.1, 0.2, 1.], [1., 0., 0., 0.]] {
        assert!(crate::validation::positive_tensor(&mut tensor, &mut [0.; 4], 2).is_err());
    }
}
struct Identity;
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFemCoefficients<R, V, D>
    for Identity
{
    fn evaluate(&self, _: [R; D], k: &mut [V], c: &mut [V]) -> Result<(), SciOxideError> {
        let n = c.len().isqrt() * D;
        k.fill(V::ZERO);
        for i in 0..n {
            k[i * n + i] = V::ONE;
        }
        c.fill(V::ZERO);
        Ok(())
    }
}
fn triangle<R: RealScalar>() -> TriangleMesh<R> {
    TriangleMesh::new(
        vec![
            Point::new([R::ZERO, R::ZERO]).unwrap(),
            Point::new([R::ONE, R::ZERO]).unwrap(),
            Point::new([R::ZERO, R::ONE]).unwrap(),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap()
}
fn tetrahedron<R: RealScalar>() -> TetrahedronMesh<R> {
    TetrahedronMesh::new(
        vec![
            Point::new([R::ZERO, R::ZERO, R::ZERO]).unwrap(),
            Point::new([R::ONE, R::ZERO, R::ZERO]).unwrap(),
            Point::new([R::ZERO, R::ONE, R::ZERO]).unwrap(),
            Point::new([R::ZERO, R::ZERO, R::ONE]).unwrap(),
        ],
        vec![[0, 1, 2, 3]],
    )
    .unwrap()
}
fn natural<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    components: usize,
) -> TaggedBoundaries<R, V, D> {
    let tags =
        SimplexBoundaryTags::new(mesh, mesh.boundary_faces().map(|f| (f, 0)).collect()).unwrap();
    TaggedBoundaries::new(
        mesh,
        &tags,
        components,
        (0..if mesh.is_empty() { 0 } else { components })
            .map(|component| {
                BoundaryCondition::natural(0, point_normal(|_, _| Ok(V::ZERO)))
                    .for_component(component)
            })
            .collect(),
    )
    .unwrap()
}
fn check_basis<R: RealScalar, const D: usize>(mesh: SimplexMesh<R, D>) {
    let tolerance = R::EPSILON.into_f64() * 2048.;
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        let boundary = natural::<R, R, D>(&mesh, 1);
        let plan = SimplexFemPlan::with_options(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            order,
            &Identity,
            &boundary,
            SimplexFemOptions::new().with_limits(limits()),
        )
        .unwrap();
        let local = order.nodes(D);
        let mut values = vec![R::ZERO; local];
        let mut gradients = vec![[R::ZERO; D]; local];
        for i in 0..local {
            plan.basis_into(reference::reference_node(i), &mut values, &mut gradients)
                .unwrap();
            for (j, &v) in values.iter().enumerate() {
                assert!((v.into_f64() - if i == j { 1. } else { 0. }).abs() < tolerance);
            }
        }
        let point = [R::try_from_f64_lossy(0.17).unwrap(); D];
        plan.basis_into(point, &mut values, &mut gradients).unwrap();
        assert!((values.iter().map(|x| x.into_f64()).sum::<f64>() - 1.).abs() < tolerance);
        for axis in 0..D {
            assert!(
                gradients
                    .iter()
                    .map(|x| x[axis].into_f64())
                    .sum::<f64>()
                    .abs()
                    < tolerance
            );
        }
        let mass = plan.element_mass(0).unwrap();
        let measure = if D == 2 { 0.5 } else { 1. / 6. };
        assert!((mass.iter().map(|x| x.into_f64()).sum::<f64>() - measure).abs() < tolerance);
        let mut out = vec![R::ZERO; plan.unknowns()];
        let mut scratch = out.clone();
        plan.apply_into(&vec![R::ONE; plan.unknowns()], &mut out, &mut scratch)
            .unwrap();
        assert!(out.iter().all(|v| v.abs().into_f64() < tolerance));
        if order == SimplexFemOrder::P1 {
            for i in 0..local {
                for j in 0..local {
                    let expected =
                        measure * (if i == j { 2. } else { 1. }) / ((D + 1) * (D + 2)) as f64;
                    assert!((mass[i * local + j].into_f64() - expected).abs() < tolerance);
                    let dot = if i == 0 && j == 0 {
                        D as f64
                    } else if i == 0 || j == 0 {
                        -1.
                    } else if i == j {
                        1.
                    } else {
                        0.
                    };
                    assert!(
                        (plan.element_matrix(0).unwrap()[i * local + j].into_f64() - measure * dot)
                            .abs()
                            < tolerance
                    );
                }
            }
        }
    }
}
#[test]
fn bases_and_moments_both_precisions_and_dimensions() {
    check_basis(triangle::<f64>());
    check_basis(triangle::<f32>());
    check_basis(tetrahedron::<f64>());
    check_basis(tetrahedron::<f32>());
}

#[test]
fn quadratic_geometry_matches_analytic_map_and_physical_gradient() {
    let mesh = triangle::<f64>();
    // F(r,s) = (r, s + r*s/2). Positive J with determinant 1+r/2.
    let edge = UnstructuredEdgeField::from_fn(&mesh, |e| {
        let [a, b] = *mesh.edge_vertices(e).unwrap();
        let a = mesh.points()[a].as_array();
        let b = mesh.points()[b].as_array();
        let r = (a[0] + b[0]) * 0.5;
        let s = (a[1] + b[1]) * 0.5;
        [r, s + r * s * 0.5]
    })
    .unwrap();
    let boundary = natural::<f64, f64, 2>(&mesh, 1);
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Quadratic(edge.view()),
        SimplexFemOrder::P2,
        &Identity,
        &boundary,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let map = plan.map(0, [0.2, 0.3]).unwrap();
    assert!((map.point[1] - 0.33).abs() < 1e-14);
    assert!((map.determinant - 1.1).abs() < 1e-14);
    assert!((map.jacobian[1][0] - 0.15).abs() < 1e-14);
    assert!((map.jacobian[1][1] - 1.1).abs() < 1e-14);
    assert!((plan.element_mass(0).unwrap().iter().sum::<f64>() - 7. / 12.).abs() < 1e-13);
    let input = (0..plan.unknowns())
        .map(|i| {
            let x = plan.node_point(i).unwrap();
            2. + 3. * x[0] - x[1]
        })
        .collect::<Vec<_>>();
    let mut values = [0.];
    let mut gradients = [[0.; 2]];
    plan.evaluate_into(0, [0.2, 0.3], &input, &mut values, &mut gradients)
        .unwrap();
    assert!((values[0] - 2.27).abs() < 1e-13);
    assert!((gradients[0][0] - 3.).abs() < 1e-13);
    assert!((gradients[0][1] + 1.).abs() < 1e-13);
}

// Independent barycentric beta-integral oracle, using exact small integer
// coefficient/exponent lists rather than production polynomial/cubature code.
fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}
type Term = ([usize; 4], f64);
fn moment<const D: usize>(a: &[usize; 4]) -> f64 {
    a[..D + 1].iter().map(|&p| factorial(p)).product::<f64>()
        / factorial(D + a[..D + 1].iter().sum::<usize>())
}
fn basis_terms<const D: usize>() -> Vec<Vec<Term>> {
    let mut basis = Vec::new();
    for i in 0..=D {
        let mut a = [0; 4];
        a[i] = 2;
        let mut b = [0; 4];
        b[i] = 1;
        basis.push(vec![(a, 2.), (b, -1.)]);
    }
    for i in 0..=D {
        for j in i + 1..=D {
            let mut a = [0; 4];
            a[i] = 1;
            a[j] = 1;
            basis.push(vec![(a, 4.)]);
        }
    }
    basis
}
fn derivative_terms(terms: &[Term], axis: usize) -> Vec<Term> {
    terms
        .iter()
        .flat_map(|&(powers, c)| {
            [0, axis + 1].into_iter().filter_map(move |variable| {
                if powers[variable] == 0 {
                    None
                } else {
                    let mut next = powers;
                    next[variable] -= 1;
                    Some((
                        next,
                        c * powers[variable] as f64 * if variable == 0 { -1. } else { 1. },
                    ))
                }
            })
        })
        .collect()
}
fn integral_product<const D: usize>(a: &[Term], b: &[Term]) -> f64 {
    a.iter()
        .flat_map(|(p, c)| {
            b.iter()
                .map(move |(q, d)| c * d * moment::<D>(&core::array::from_fn(|i| p[i] + q[i])))
        })
        .sum()
}
fn p2_oracle<const D: usize>(mesh: SimplexMesh<f64, D>) {
    let boundaries = natural::<f64, f64, D>(&mesh, 1);
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Identity,
        &boundaries,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let basis = basis_terms::<D>();
    let n = basis.len();
    for (i, a) in basis.iter().enumerate() {
        for (j, b) in basis.iter().enumerate() {
            assert!(
                (plan.element_mass(0).unwrap()[i * n + j] - integral_product::<D>(a, b)).abs()
                    < 3e-14
            );
            let stiffness = (0..D)
                .map(|axis| {
                    integral_product::<D>(&derivative_terms(a, axis), &derivative_terms(b, axis))
                })
                .sum::<f64>();
            assert!((plan.element_matrix(0).unwrap()[i * n + j] - stiffness).abs() < 4e-14);
        }
    }
}
#[test]
fn p2_mass_and_stiffness_match_independent_beta_integrals() {
    p2_oracle(triangle());
    p2_oracle(tetrahedron());
}

fn scalar<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn coupling<R: RealScalar, V: NumericScalar<Real = R>>(a: usize, b: usize) -> V {
    if a == b {
        V::from_real(scalar(2. + a as f64))
    } else {
        V::from_components(scalar(0.2), scalar(if a < b { 0.1 } else { -0.1 }))
    }
}
fn amplitude<R: RealScalar, V: NumericScalar<Real = R>>(component: usize) -> V {
    V::from_components(scalar(1. + component as f64), scalar(0.3))
}
fn patch_value<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    point: [R; D],
    component: usize,
    quadratic: bool,
) -> V {
    let value = point.iter().enumerate().fold(R::ONE, |s, (axis, &x)| {
        s + scalar::<R>((axis + 1) as f64) * x + if quadratic { x * x } else { R::ZERO }
    });
    amplitude::<R, V>(component) * V::from_real(value)
}
fn tensor<R: RealScalar, V: NumericScalar<Real = R>>(
    a: usize,
    axis: usize,
    b: usize,
    direction: usize,
) -> V {
    coupling::<R, V>(a, b) * V::from_real(scalar(if axis == direction { 1. } else { 0.15 }))
}
fn flux<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    point: [R; D],
    normal: [R; D],
    a: usize,
    components: usize,
    quadratic: bool,
) -> V {
    let mut value = V::ZERO;
    for b in 0..components {
        for (axis, &normal_component) in normal.iter().enumerate() {
            for (direction, &coordinate) in point.iter().enumerate() {
                let grad = scalar::<R>((direction + 1) as f64)
                    + if quadratic {
                        scalar::<R>(2.) * coordinate
                    } else {
                        R::ZERO
                    };
                value = value
                    + tensor::<R, V>(a, axis, b, direction)
                        * amplitude::<R, V>(b)
                        * V::from_real(normal_component * grad);
            }
        }
    }
    value
}
struct Patch;
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFemCoefficients<R, V, D>
    for Patch
{
    fn evaluate(&self, _: [R; D], k: &mut [V], c: &mut [V]) -> Result<(), SciOxideError> {
        let components = c.len().isqrt();
        let kd = components * D;
        for a in 0..components {
            for b in 0..components {
                c[a * components + b] = coupling::<R, V>(a, b) * V::from_real(scalar(0.25));
                for axis in 0..D {
                    for direction in 0..D {
                        k[(a * D + axis) * kd + b * D + direction] = tensor(a, axis, b, direction);
                    }
                }
            }
        }
        Ok(())
    }
}
fn patch_source<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
>(
    components: usize,
    quadratic: bool,
) -> FieldData<R, V, D> {
    source(move |point, a| {
        let mut value = V::ZERO;
        for b in 0..components {
            value = value
                + coupling::<R, V>(a, b)
                    * V::from_real(scalar(0.25))
                    * patch_value::<R, V, D>(point, b, quadratic);
            if quadratic {
                value = value
                    - coupling::<R, V>(a, b)
                        * amplitude::<R, V>(b)
                        * V::from_real(scalar(2. * D as f64));
            }
        }
        value
    })
}
fn patch_boundaries<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
>(
    mesh: &SimplexMesh<R, D>,
    components: usize,
    quadratic: bool,
) -> TaggedBoundaries<R, V, D> {
    let tags = SimplexBoundaryTags::new(
        mesh,
        mesh.boundary_faces()
            .map(|face| {
                let v = mesh.face_vertices(face).unwrap();
                let tag = if v.iter().all(|&i| mesh.points()[i].as_array()[0] == R::ZERO) {
                    0
                } else if v.iter().all(|&i| mesh.points()[i].as_array()[0] == R::ONE) {
                    1
                } else {
                    2
                };
                (face, tag)
            })
            .collect(),
    )
    .unwrap();
    let mut conditions = Vec::new();
    for component in 0..components {
        conditions.push(
            BoundaryCondition::dirichlet(
                0,
                point_normal(move |x, _| Ok(patch_value(x, component, quadratic))),
            )
            .for_component(component),
        );
        conditions.push(
            BoundaryCondition::robin(
                1,
                V::ONE,
                point_normal(move |x, n| {
                    Ok(flux::<R, V, D>(x, n, component, components, quadratic)
                        + patch_value::<R, V, D>(x, component, quadratic))
                }),
            )
            .for_component(component),
        );
        conditions.push(
            BoundaryCondition::natural(
                2,
                point_normal(move |x, n| Ok(flux(x, n, component, components, quadratic))),
            )
            .for_component(component),
        );
    }
    TaggedBoundaries::new(mesh, &tags, components, conditions).unwrap()
}
fn patch_case<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>(
    base: SimplexMesh<R, D>,
) {
    for curved in [false, true] {
        for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
            let (mesh, edge) = meshes::curved(&base).unwrap();
            let mesh = if curved { mesh } else { base.clone() };
            let quadratic = order == SimplexFemOrder::P2 && !curved;
            // Physical linear functions pull back to degree two on quadratic maps;
            // P1 curved maps are separately covered by convergence below.
            if curved && order == SimplexFemOrder::P1 {
                continue;
            }
            let boundary = patch_boundaries::<R, V, D>(&mesh, 2, quadratic);
            let plan = SimplexFemPlan::with_options(
                mesh,
                if curved {
                    SimplexFemGeometry::Quadratic(edge.view())
                } else {
                    SimplexFemGeometry::Affine
                },
                order,
                &Patch,
                &boundary,
                SimplexFemOptions::new().with_limits(limits()),
            )
            .unwrap();
            let input = (0..plan.unknowns())
                .map(|i| {
                    patch_value::<R, V, D>(
                        plan.node_point(i % plan.space.nodes).unwrap(),
                        i / plan.space.nodes,
                        quadratic,
                    )
                })
                .collect::<Vec<_>>();
            let mut output = vec![V::ZERO; input.len()];
            let mut scratch = output.clone();
            plan.apply_into(&input, &mut output, &mut scratch).unwrap();
            let rhs = plan.load(&patch_source(2, quadratic), R::ZERO).unwrap();
            let scale = rhs.iter().fold(R::ONE, |s, v| s.max(v.abs())).into_f64();
            let tol = R::EPSILON.into_f64() * scale * 8192.;
            let defect = output
                .iter()
                .zip(&rhs)
                .map(|(&a, &b)| (a - b).abs().into_f64())
                .fold(0., f64::max);
            assert!(
                defect < tol,
                "D={D} {order:?} curved={curved} patch defect {defect:e}, tolerance {tol:e}"
            );
            let csr = plan.assemble_csr().unwrap();
            let mut dense = vec![V::ZERO; input.len() * input.len()];
            csr.to_dense_into(&mut dense).unwrap();
            for (i, row) in dense.chunks_exact(input.len()).enumerate() {
                let value = row
                    .iter()
                    .zip(&input)
                    .fold(V::ZERO, |s, (&a, &b)| s + a * b);
                assert!((value - output[i]).abs().into_f64() < tol);
                for (j, &a) in row.iter().enumerate() {
                    assert!((a - dense[j * input.len() + i].conj()).abs().into_f64() < tol);
                }
            }
            let energy = input
                .iter()
                .zip(&output)
                .fold(V::ZERO, |s, (&x, &y)| s + x.conj() * y);
            assert!(energy.real_component() > R::ZERO);
            assert!(energy.imaginary_component().abs().into_f64() < tol * input.len() as f64);
            let owned = plan.fields(&input).unwrap();
            let vertices = owned
                .vertices()
                .iter()
                .map(|f| f.view())
                .collect::<Vec<_>>();
            let edges = owned.edges().iter().map(|f| f.view()).collect::<Vec<_>>();
            let mut packed = vec![V::ZERO; input.len()];
            plan.pack_fields_into(SimplexFemFieldView::new(&vertices, &edges), &mut packed)
                .unwrap();
            assert_eq!(packed, input);
            let applied = plan
                .apply_fields(SimplexFemFieldView::new(&vertices, &edges))
                .unwrap();
            let vv = applied
                .vertices()
                .iter()
                .map(|f| f.view())
                .collect::<Vec<_>>();
            let ee = applied.edges().iter().map(|f| f.view()).collect::<Vec<_>>();
            plan.pack_fields_into(SimplexFemFieldView::new(&vv, &ee), &mut packed)
                .unwrap();
            assert_eq!(packed, output);
        }
    }
}
#[test]
fn anisotropic_coupled_mixed_patches_all_scalar_domains() {
    patch_case::<f64, f64, 2>(meshes::triangle_grid(2).unwrap());
    patch_case::<f64, scioxide_core::Complex64, 2>(meshes::triangle_grid(2).unwrap());
    patch_case::<f32, f32, 2>(meshes::triangle_grid(2).unwrap());
    patch_case::<f32, scioxide_core::Complex32, 2>(meshes::triangle_grid(2).unwrap());
    patch_case::<f64, f64, 3>(meshes::tetrahedron_grid(1).unwrap());
    patch_case::<f64, scioxide_core::Complex64, 3>(meshes::tetrahedron_grid(1).unwrap());
    patch_case::<f32, f32, 3>(meshes::tetrahedron_grid(1).unwrap());
    patch_case::<f32, scioxide_core::Complex32, 3>(meshes::tetrahedron_grid(1).unwrap());
}

struct Smooth;
fn smooth<const D: usize>(x: [f64; D]) -> f64 {
    (x.iter().sum::<f64>() * 0.5).exp()
}
impl<const D: usize> SimplexFemCoefficients<f64, f64, D> for Smooth {
    fn evaluate(&self, _: [f64; D], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
        for i in 0..D {
            for j in 0..D {
                k[i * D + j] = if i == j { 1. } else { 0.15 };
            }
        }
        c[0] = 1.;
        Ok(())
    }
}
fn smooth_error<const D: usize>(
    base: SimplexMesh<f64, D>,
    order: SimplexFemOrder,
    curved: bool,
) -> (f64, f64) {
    let (mapped, edges) = meshes::curved(&base).unwrap();
    let mesh = if curved { mapped } else { base };
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect()).unwrap();
    let boundaries = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![BoundaryCondition::dirichlet(0, point_normal(|x, _| Ok(smooth(x)))).for_component(0)],
    )
    .unwrap();
    let plan = SimplexFemPlan::with_options(
        mesh,
        if curved {
            SimplexFemGeometry::Quadratic(edges.view())
        } else {
            SimplexFemGeometry::Affine
        },
        order,
        &Smooth,
        &boundaries,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let solver = scioxide_linalg::CgSolver::new(
        scioxide_core::ResidualTolerance::new(1e-13, 1e-13).unwrap(),
        Budget::try_from(5000).unwrap(),
    );
    let mut workspace = scioxide_linalg::IterativeWorkspace::new();
    let mut solution = scioxide_linalg::Vector::zeros(plan.unknowns()).unwrap();
    let rhs = scioxide_linalg::Vector::from_vec(
        plan.load(
            &source(|x: [f64; D], _| {
                (1. - 0.25 * (D as f64 + 0.15 * (D * (D - 1)) as f64)) * smooth(x)
            }),
            0.,
        )
        .unwrap(),
    );
    let report = solver
        .solve_into(
            &plan,
            &rhs,
            Preconditioning::None,
            solution.view_mut(),
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success(), "CG {:?}", report.termination());
    // A higher-degree error rule than assembly; analytic values and gradients
    // are the accuracy oracle, not nodal superconvergence or solver residuals.
    let rule = SimplexRule::new(
        D,
        if curved { 4 } else { 3 },
        scioxide_integrate::GeometricOptions::default()
            .with_max_nodes(Budget::try_from(512).unwrap()),
    )
    .unwrap();
    let mut values = [0.];
    let mut gradients = [[0.; D]];
    let mut l2 = 0.;
    let mut energy = 0.;
    for cell in 0..plan.mesh().cell_count() {
        for (bary, &w) in rule
            .barycentric_nodes()
            .chunks_exact(D + 1)
            .zip(rule.weights())
        {
            let reference = core::array::from_fn(|i| bary[i + 1]);
            let map = plan.map(cell, reference).unwrap();
            plan.evaluate_into(
                cell,
                reference,
                solution.as_slice(),
                &mut values,
                &mut gradients,
            )
            .unwrap();
            let exact = smooth(map.point);
            let error = values[0] - exact;
            let weight = w * map.determinant;
            l2 += weight * error * error;
            for i in 0..D {
                for j in 0..D {
                    energy += weight
                        * (gradients[0][i] - 0.5 * exact)
                        * (gradients[0][j] - 0.5 * exact)
                        * if i == j { 1. } else { 0.15 };
                }
            }
        }
    }
    (l2.sqrt(), energy.sqrt())
}
fn refinement<const D: usize>(mesh: impl Fn(usize) -> SimplexMesh<f64, D>, sizes: [usize; 3]) {
    for curved in [false, true] {
        for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
            let errors = sizes.map(|n| smooth_error(mesh(n), order, curved));
            let l2_rate = (errors[1].0 / errors[2].0).log2();
            let energy_rate = (errors[1].1 / errors[2].1).log2();
            std::eprintln!(
                "simplex FEM D={D} {order:?} curved={curved}, sizes={sizes:?}, errors={errors:?}, L2 order={l2_rate:.4}, energy order={energy_rate:.4}"
            );
            assert!(errors[0].0 > errors[1].0 && errors[0].1 > errors[1].1);
            assert!(l2_rate > order.degree() as f64 + 0.75, "L2 order {l2_rate}");
            assert!(
                energy_rate > order.degree() as f64 - 0.25,
                "energy order {energy_rate}"
            );
        }
    }
}
#[test]
fn manufactured_2d_orders_on_affine_and_curved_meshes() {
    refinement(|n| meshes::triangle_grid(n).unwrap(), [2, 4, 8]);
}
#[test]
fn manufactured_3d_orders_on_affine_and_curved_meshes() {
    refinement(|n| meshes::tetrahedron_grid(n).unwrap(), [1, 2, 4]);
}

#[test]
fn curved_tetrahedral_jacobian_volume_and_shared_edge_dofs() {
    let base = tetrahedron::<f64>();
    let (mesh, edges) = meshes::curved(&base).unwrap();
    let boundaries = natural::<f64, f64, 3>(&mesh, 1);
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Quadratic(edges.view()),
        SimplexFemOrder::P2,
        &Identity,
        &boundaries,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let map = plan.map(0, [0.2, 0.3, 0.1]).unwrap();
    assert!((map.determinant - 1.21).abs() < 1e-14);
    assert!((map.jacobian[1][0] - 0.15).abs() < 1e-14);
    assert!((map.jacobian[2][0] - 0.05).abs() < 1e-14);
    assert!((plan.element_mass(0).unwrap().iter().sum::<f64>() - 17. / 80.).abs() < 3e-14);
    let mesh = meshes::triangle_grid::<f64>(1).unwrap();
    let boundary = natural::<f64, f64, 2>(&mesh, 1);
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Identity,
        &boundary,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let shared = plan
        .cell_dofs(0)
        .unwrap()
        .iter()
        .filter(|i| plan.cell_dofs(1).unwrap().contains(i))
        .count();
    assert_eq!(shared, 3);
}

#[test]
fn failures_reject_before_commit_and_preserve_work_accounting() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let mesh = triangle::<f64>();
    let boundary = natural::<f64, f64, 2>(&mesh, 1);
    let plan = SimplexFemPlan::with_options(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Identity,
        &boundary,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let mut input = vec![1.; plan.unknowns()];
    let mut output = vec![37.; plan.unknowns()];
    let mut scratch = vec![0.; LinearOperator::apply_workspace_len(&plan).unwrap()];
    input[0] = f64::NAN;
    assert!(plan.apply_into(&input, &mut output, &mut scratch).is_err());
    assert!(output.iter().all(|&x| x == 37.));
    input[0] = 1.;
    assert!(plan.apply_into(&input, &mut output, &mut []).is_err());
    assert!(plan.map(0, [0.9, 0.9]).is_err());
    assert!(plan.map(5, [0.1, 0.1]).is_err());
    let mut values = [37.];
    let mut gradients = [[37.; 2]];
    assert!(
        plan.evaluate_into(0, [-0.1, 0.], &input, &mut values, &mut gradients)
            .is_err()
    );
    assert_eq!(values, [37.]);
    assert_eq!(gradients, [[37.; 2]]);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let execution = ExecutionContext::sequential().with_cancellation(&cancel);
    assert!(apply_scoped(&plan, &input, &mut output, &mut scratch, execution).is_err());
    assert!(output.iter().all(|&x| x == 37.));
    for case in 0..5 {
        let options = SimplexFemOptions::new().with_limits(limits());
        let options = match case {
            0 => options.with_limits(SimplexFemLimits::new(1, 30_000_000, 256, 1_000_000_000)),
            1 => options.with_limits(SimplexFemLimits::new(100_000, 30_000_000, 256, 1)),
            2 => options.with_limits(SimplexFemLimits::new(100_000, 1, 256, 1_000_000_000)),
            3 => options.with_volume_rule(
                SimplexRule::new(
                    2,
                    0,
                    scioxide_integrate::GeometricOptions::default()
                        .with_max_nodes(Budget::try_from(1).unwrap()),
                )
                .unwrap(),
            ),
            _ => options.with_minimum_reciprocal_condition(f64::NAN),
        };
        assert!(
            SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                SimplexFemOrder::P2,
                &Identity,
                &boundary,
                options
            )
            .is_err()
        );
    }
    let alien = triangle::<f64>();
    let field = VertexField::new(&alien, vec![1.; 3]).unwrap();
    let edge = UnstructuredEdgeField::new(&mesh, vec![1.; 3]).unwrap();
    assert!(
        plan.pack_fields_into(
            SimplexFemFieldView::new(&[field.view()], &[edge.view()]),
            &mut output
        )
        .is_err()
    );
    assert!(output.iter().all(|&x| x == 37.));
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
    let id = SolveIdentity::new(
        ProblemToken::new(673),
        scioxide_core::algorithm_id!("test", "fem", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    assert!(
        SimplexFemPlan::build(
            mesh,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &Identity,
            &boundary,
            SimplexFemOptions::new().with_limits(limits()),
            &mut context
        )
        .is_err()
    );
    assert_eq!(context.counts().own().total_providers().unwrap(), 0);
    assert_eq!(
        context.counts().total().unwrap().total_providers().unwrap(),
        2
    );
    assert_eq!(context.budget_report(BudgetKind::Evaluation).consumed(), 2);
}

#[test]
fn coefficient_boundary_and_invalid_curved_geometry_failures() {
    struct Bad(usize);
    impl SimplexFemCoefficients<f64, f64, 2> for Bad {
        fn evaluate(&self, x: [f64; 2], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
            Identity.evaluate(x, k, c)?;
            match self.0 {
                0 => k.fill(0.),
                1 => k[0] = -1.,
                2 => k[1] = 0.5,
                3 => c[0] = f64::NAN,
                _ => {
                    return Err(invalid(TEST_ARGUMENT, "deliberate coefficient failure"));
                }
            }
            Ok(())
        }
    }
    let mesh = triangle::<f64>();
    let boundary = natural::<f64, f64, 2>(&mesh, 1);
    for i in 0..5 {
        assert!(
            SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                SimplexFemOrder::P2,
                &Bad(i),
                &boundary,
                SimplexFemOptions::new().with_limits(limits())
            )
            .is_err()
        );
    }
    let wrong = UnstructuredEdgeField::from_fn(&mesh, |_| [-2., -2.]).unwrap();
    assert!(
        SimplexFemPlan::with_options(
            mesh.clone(),
            SimplexFemGeometry::Quadratic(wrong.view()),
            SimplexFemOrder::P2,
            &Identity,
            &boundary,
            SimplexFemOptions::new().with_limits(limits())
        )
        .is_err()
    );
    let tags = SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, f as u32)).collect(),
    )
    .unwrap();
    assert!(TaggedBoundaries::<f64, f64, 2>::new(&mesh, &tags, 1, vec![]).is_err());
    let conditions = mesh
        .boundary_faces()
        .map(|face| {
            BoundaryCondition::dirichlet(face as u32, point_normal(move |_, _| Ok(face as f64)))
                .for_component(0)
        })
        .collect();
    let conflict = TaggedBoundaries::new(&mesh, &tags, 1, conditions).unwrap();
    assert!(
        SimplexFemPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &Identity,
            &conflict,
            SimplexFemOptions::new().with_limits(limits())
        )
        .is_err()
    );
}

#[cfg(feature = "std")]
#[test]
fn executor_preparation_and_application_match_sequential() {
    use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState, ThreadExecutor};
    let mesh = meshes::triangle_grid::<f64>(3).unwrap();
    let boundary = patch_boundaries::<f64, scioxide_core::Complex64, 2>(&mesh, 2, true);
    let seq = SimplexFemPlan::with_options(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Patch,
        &boundary,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let workers = core::num::NonZeroUsize::new(3).unwrap();
    let pool = ThreadExecutor::new(workers);
    let execution = ExecutionContext::new(&pool);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(673),
        scioxide_core::algorithm_id!("test", "fem", 1),
        0,
        0,
    );
    let mut context = state.begin_solve(execution, id, &mut budgets).unwrap();
    let options = SimplexFemOptions::new()
        .with_limits(limits())
        .with_execution_policy(ExecutionPolicy::Parallel(workers));
    let parallel = SimplexFemPlan::build(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Patch,
        &boundary,
        options,
        &mut context,
    )
    .unwrap();
    let data = patch_source::<f64, scioxide_core::Complex64, 2>(2, true);
    assert_eq!(
        seq.load(&data, 0.).unwrap(),
        parallel.load(&data, 0.).unwrap()
    );
    for cell in 0..seq.space.cell_count() {
        assert_eq!(
            seq.element_matrix(cell).unwrap(),
            parallel.element_matrix(cell).unwrap()
        );
    }
    let input = (0..seq.unknowns())
        .map(|i| scioxide_core::Complex64::new(i as f64 * 0.1, -0.3))
        .collect::<Vec<_>>();
    let mut a = vec![scioxide_core::Complex64::ZERO; input.len()];
    let mut b = a.clone();
    let mut scratch = a.clone();
    seq.apply_into(&input, &mut a, &mut scratch).unwrap();
    apply_scoped(&parallel, &input, &mut b, &mut scratch, execution).unwrap();
    assert_eq!(a, b);
}

#[test]
fn variable_coefficients_preserve_polynomial_exactness() {
    struct Variable;
    impl SimplexFemCoefficients<f64, f64, 2> for Variable {
        fn evaluate(&self, x: [f64; 2], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
            k.copy_from_slice(&[1. + x[0], 0., 0., 1. + x[0]]);
            c[0] = 1. + x[1];
            Ok(())
        }
    }
    let mesh = triangle::<f64>();
    let boundary = natural::<f64, f64, 2>(&mesh, 1);
    let basis = basis_terms::<2>();
    let one = [([0; 4], 1.)];
    let multiply = |a: &[Term], b: &[Term]| {
        a.iter()
            .flat_map(|(p, c)| {
                b.iter()
                    .map(move |(q, d)| (core::array::from_fn(|i| p[i] + q[i]), c * d))
            })
            .collect::<Vec<_>>()
    };
    let k = [([0; 4], 1.), ([0, 1, 0, 0], 1.)];
    let c = [([0; 4], 1.), ([0, 0, 1, 0], 1.)];
    let f = [([0; 4], 1.), ([0, 2, 0, 0], 1.)];
    for index in [2, 4] {
        let options = SimplexFemOptions::new()
            .with_limits(limits())
            .with_volume_rule(
                SimplexRule::new(
                    2,
                    index,
                    scioxide_integrate::GeometricOptions::default()
                        .with_max_nodes(Budget::try_from(256).unwrap()),
                )
                .unwrap(),
            );
        let plan = SimplexFemPlan::with_options(
            mesh.clone(),
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &Variable,
            &boundary,
            options,
        )
        .unwrap();
        let load = plan
            .load(&source(|x: [f64; 2], _| 1. + x[0] * x[0]), 0.)
            .unwrap();
        for (i, a) in basis.iter().enumerate() {
            assert!(
                (load[plan.space.dofs(0)[i]] - integral_product::<2>(&multiply(a, &f), &one)).abs()
                    < 1e-13
            );
            for (j, b) in basis.iter().enumerate() {
                let expected = integral_product::<2>(&multiply(a, &c), b)
                    + (0..2)
                        .map(|axis| {
                            integral_product::<2>(
                                &multiply(&derivative_terms(a, axis), &k),
                                &derivative_terms(b, axis),
                            )
                        })
                        .sum::<f64>();
                assert!((plan.element_matrix(0).unwrap()[i * 6 + j] - expected).abs() < 2e-13);
            }
        }
    }
}

fn balance<const D: usize>(mesh: SimplexMesh<f64, D>) {
    struct Balance;
    impl<const D: usize> SimplexFemCoefficients<f64, f64, D> for Balance {
        fn evaluate(&self, x: [f64; D], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
            Identity.evaluate(x, k, c)?;
            Ok(())
        }
    }
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect()).unwrap();
    let boundaries = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![
            BoundaryCondition::natural(
                0,
                point_normal(|x, n| Ok(2. * x.iter().zip(n).map(|(&x, n)| x * n).sum::<f64>())),
            )
            .for_component(0),
        ],
    )
    .unwrap();
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Balance,
        &boundaries,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    let rhs = plan.load(&FieldData::constant(-2. * D as f64), 0.).unwrap();
    assert!(rhs.iter().sum::<f64>().abs() < 2e-13);
    let input = (0..plan.unknowns())
        .map(|i| plan.node_point(i).unwrap().iter().map(|x| x * x).sum())
        .collect::<Vec<f64>>();
    let mut out = vec![0.; input.len()];
    let mut work = out.clone();
    plan.apply_into(&input, &mut out, &mut work).unwrap();
    assert!(out.iter().zip(&rhs).all(|(&a, &b)| (a - b).abs() < 2e-13));
    plan.apply_into(&vec![1.; input.len()], &mut out, &mut work)
        .unwrap();
    assert!(out.iter().all(|v| v.abs() < 2e-13));
}
#[test]
fn pure_neumann_global_balance_and_constant_mode() {
    balance(meshes::triangle_grid(2).unwrap());
    balance(meshes::tetrahedron_grid(1).unwrap());
}

#[test]
fn empty_meshes_and_geometry_conditioning() {
    let mesh = TriangleMesh::<f64>::new(vec![], vec![]).unwrap();
    let boundary = natural::<f64, f64, 2>(&mesh, 1);
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Identity,
        &boundary,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    assert_eq!(plan.unknowns(), 0);
    plan.apply_into(&[], &mut [], &mut []).unwrap();
    assert_eq!(plan.assemble_csr().unwrap().nnz(), 0);
    let skinny = TriangleMesh::new(
        vec![
            Point::new([0., 0.]).unwrap(),
            Point::new([1., 0.]).unwrap(),
            Point::new([0., 1e-16]).unwrap(),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let boundary = natural::<f64, f64, 2>(&skinny, 1);
    assert!(
        SimplexFemPlan::with_options(
            skinny,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P1,
            &Identity,
            &boundary,
            SimplexFemOptions::new().with_limits(limits())
        )
        .is_err()
    );
}

#[test]
fn dirichlet_precedence_is_explicit_for_p1_and_p2() {
    let mesh = meshes::triangle_grid::<f64>(2).unwrap();
    let tags = SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, f as u32)).collect(),
    )
    .unwrap();
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        for reverse in [false, true] {
            let mut faces = mesh.boundary_faces().collect::<Vec<_>>();
            if reverse {
                faces.reverse();
            }
            let conditions = faces
                .iter()
                .map(|&face| {
                    BoundaryCondition::dirichlet(
                        face as u32,
                        point_normal(move |_, _| Ok(face as f64 + 1.)),
                    )
                    .for_component(0)
                })
                .collect();
            let boundaries = TaggedBoundaries::new(&mesh, &tags, 1, conditions).unwrap();
            assert!(
                SimplexFemPlan::with_options(
                    mesh.clone(),
                    SimplexFemGeometry::Affine,
                    order,
                    &Identity,
                    &boundaries,
                    SimplexFemOptions::new().with_limits(limits())
                )
                .is_err()
            );
            let boundaries = boundaries
                .with_dirichlet_conflict_policy(DirichletConflictPolicy::LastConditionWins);
            let plan = SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Affine,
                order,
                &Identity,
                &boundaries,
                SimplexFemOptions::new().with_limits(limits()),
            )
            .unwrap();
            for vertex in 0..mesh.vertex_count() {
                let expected = faces
                    .iter()
                    .rev()
                    .find(|&&face| mesh.face_vertices(face).unwrap().contains(&vertex))
                    .map(|&face| face as f64 + 1.);
                assert_eq!(prescribed(&plan)[vertex], expected);
            }
            // Edge midpoint traces retain their own condition; shared vertices use the global winner.
            if order == SimplexFemOrder::P2 {
                for edge in 0..mesh.edge_count() {
                    let vertices = mesh.edge_vertices(edge).unwrap();
                    let expected = faces
                        .iter()
                        .find(|&&face| {
                            let fv = mesh.face_vertices(face).unwrap();
                            fv.contains(&vertices[0]) && fv.contains(&vertices[1])
                        })
                        .map(|&face| face as f64 + 1.);
                    assert_eq!(prescribed(&plan)[mesh.vertex_count() + edge], expected);
                }
            }
            #[cfg(feature = "std")]
            {
                let workers = core::num::NonZeroUsize::new(3).unwrap();
                let pool = scioxide_core::ThreadExecutor::new(workers);
                let mut state = scioxide_core::SolveState::new();
                let mut budgets = scioxide_core::SolveBudgets::unbounded();
                let id = scioxide_core::SolveIdentity::new(
                    scioxide_core::ProblemToken::new(0),
                    scioxide_core::algorithm_id!("test", "precedence", 1),
                    0,
                    0,
                );
                let mut ctx = state
                    .begin_solve(ExecutionContext::new(&pool), id, &mut budgets)
                    .unwrap();
                let options = SimplexFemOptions::new()
                    .with_limits(limits())
                    .with_execution_policy(ExecutionPolicy::Parallel(workers));
                let parallel = SimplexFemPlan::build(
                    mesh.clone(),
                    SimplexFemGeometry::Affine,
                    order,
                    &Identity,
                    &boundaries,
                    options,
                    &mut ctx,
                )
                .unwrap();
                assert_eq!(prescribed(&parallel), prescribed(&plan));
                assert_eq!(
                    parallel.load(&FieldData::zero(), 0.).unwrap(),
                    plan.load(&FieldData::zero(), 0.).unwrap()
                );
            }
        }
    }
}

#[test]
fn dirichlet_normal_ties_follow_canonical_face_order_and_validate_losers() {
    let mesh = meshes::triangle_grid::<f64>(1).unwrap();
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|f| (f, 0)).collect()).unwrap();
    let boundaries = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![
            BoundaryCondition::dirichlet(
                0,
                point_normal(|_, normal| Ok(normal[0] + 2. * normal[1])),
            )
            .for_component(0),
        ],
    )
    .unwrap()
    .with_dirichlet_conflict_policy(DirichletConflictPolicy::LastConditionWins);
    let plan = SimplexFemPlan::with_options(
        mesh.clone(),
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P2,
        &Identity,
        &boundaries,
        SimplexFemOptions::new().with_limits(limits()),
    )
    .unwrap();
    for vertex in 0..mesh.vertex_count() {
        let face = mesh
            .boundary_faces()
            .filter(|&f| mesh.face_vertices(f).unwrap().contains(&vertex))
            .max()
            .unwrap();
        let vertices = mesh.face_vertices(face).unwrap();
        let a = mesh.points()[vertices[0]].as_array();
        let b = mesh.points()[vertices[1]].as_array();
        let normal = if a[0] == b[0] {
            [if a[0] == 0. { -1. } else { 1. }, 0.]
        } else {
            [0., if a[1] == 0. { -1. } else { 1. }]
        };
        assert!((prescribed(&plan)[vertex].unwrap() - normal[0] - 2. * normal[1]).abs() < 1e-14);
    }
    let tags = SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|f| (f, f as u32)).collect(),
    )
    .unwrap();
    let first = mesh.boundary_faces().next().unwrap();
    let boundaries = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        mesh.boundary_faces()
            .map(|face| {
                BoundaryCondition::dirichlet(
                    face as u32,
                    point_normal(move |_, _| Ok(if face == first { f64::NAN } else { 0. })),
                )
                .for_component(0)
            })
            .collect(),
    )
    .unwrap()
    .with_dirichlet_conflict_policy(DirichletConflictPolicy::LastConditionWins);
    assert!(
        SimplexFemPlan::with_options(
            mesh,
            SimplexFemGeometry::Affine,
            SimplexFemOrder::P2,
            &Identity,
            &boundaries,
            SimplexFemOptions::new().with_limits(limits())
        )
        .is_err()
    );
}

/// Prescribed values of the constrained unknowns of a scalar plan at time zero.
fn prescribed<V: NumericScalar<Real = f64>, const D: usize>(
    plan: &SimplexFemPlan<f64, V, D>,
) -> Vec<Option<V>> {
    let load = plan.load(&FieldData::zero(), 0.).unwrap();
    plan.constrained
        .iter()
        .zip(load)
        .map(|(&fixed, value)| fixed.then_some(value))
        .collect()
}

#[test]
fn space_components_are_labeled_by_smallest_node() {
    use scioxide_spatial::{IntervalMesh, Point};
    let points = [0., 1., 2., 3., 4.]
        .map(|x| Point::new([x]))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    // Components {0, 2, 4} and {1, 3}; the first cell belongs to the second.
    let mesh = IntervalMesh::new(points, vec![[1, 3], [2, 4], [0, 2]]).unwrap();
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        let space = crate::attempt::sequential(
            scioxide_core::algorithm_id!("test", "space", 1),
            |context| {
                SimplexSpace::prepare(
                    mesh.clone(),
                    SimplexFemGeometry::Affine,
                    order,
                    SimplexFemOptions::new().with_limits(limits()),
                    (2, 2),
                    context,
                )
            },
        )
        .unwrap();
        let components = space.components().unwrap();
        assert_eq!(components.count(), 2);
        assert_eq!(&components.labels()[..5], &[0, 1, 0, 1, 0]);
        // Edge nodes join the component of their cell.
        for cell in 0..space.cell_count() {
            let dofs = space.dofs(cell);
            assert!(
                dofs.iter()
                    .all(|&n| components.labels()[n] == components.labels()[dofs[0]])
            );
        }
    }
}
