use super::*;
use crate::{BoundaryCondition, BoundaryData, DirichletConflictPolicy};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Complex, ExecutionContext, OperatorContext, ProblemToken, SolveBudgets, SolveIdentity,
    SolveScope, SolveState,
};
use scioxide_spatial::{AxisSide, BoundarySet, CoordinateAxis, StructuredGrid};

fn limits() -> SpatialLimits {
    SpatialLimits::new(200_000, 4_000_000, 400_000_000)
}

fn mesh<R: RealScalar, const D: usize>(size: usize, periodic_x: bool) -> CellMesh<R, D> {
    let one = R::ONE;
    let axes = core::array::from_fn(|_| CoordinateAxis::uniform(R::ZERO, one, size).unwrap());
    let topology = core::array::from_fn(|axis| {
        if axis == 0 && periodic_x {
            AxisTopology::Periodic
        } else {
            AxisTopology::Bounded
        }
    });
    CellMesh::new(StructuredGrid::new(axes, topology).unwrap()).unwrap()
}

fn graded_mesh<R: RealScalar, const D: usize>(size: usize) -> CellMesh<R, D> {
    let fifth = R::try_from_f64_lossy(0.2).unwrap();
    let divisor = R::ONE + fifth;
    let axes = core::array::from_fn(|_| {
        let coordinates = (0..size)
            .map(|index| {
                let t = R::try_from_usize_exact(index).unwrap()
                    / R::try_from_usize_exact(size - 1).unwrap();
                (t + fifth * t * t) / divisor
            })
            .collect();
        CoordinateAxis::explicit(coordinates).unwrap()
    });
    CellMesh::new(StructuredGrid::new(axes, [AxisTopology::Bounded; D]).unwrap()).unwrap()
}

fn dirichlet_boundaries<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
>(
    mesh: &CellMesh<R, D>,
    components: usize,
    data: BoundaryData<R, V, D>,
) -> StructuredBoundaries<R, V, D> {
    let mut conditions = Vec::new();
    for axis in 0..D {
        if mesh.grid().axis_topology(axis).unwrap() == AxisTopology::Periodic {
            continue;
        }
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::dirichlet(
                BoundarySet::axis_side(mesh, axis, side).unwrap(),
                data.clone(),
            ));
        }
    }
    StructuredBoundaries::new(mesh, components, conditions).unwrap()
}

fn zero_boundaries<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
>(
    mesh: &CellMesh<R, D>,
    components: usize,
) -> StructuredBoundaries<R, V, D> {
    dirichlet_boundaries(mesh, components, BoundaryData::zero())
}

/// Nodal source data sampled from `f(point)`.
fn nodal<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>(
    f: impl Fn([R; D], usize) -> V + Send + Sync + 'static,
) -> FieldData<R, V, D> {
    FieldData::time_invariant(move |p: FieldPoint<R, D>| Ok(f(p.point, p.component)))
}

struct Diffusion;
impl<R: RealScalar, const D: usize> StructuredFdmCoefficients<R, R, D> for Diffusion {
    fn diffusion(&self, axis: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
        Ok(if axis == 0 { R::TWO } else { R::ONE })
    }
    fn advection(&self, _: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
        Ok(R::ZERO)
    }
    fn reaction(&self, _: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
        Ok(R::ZERO)
    }
}

fn sine<R: RealScalar, const D: usize>(point: [R; D], periodic_x: bool) -> R {
    let mut value = R::ONE;
    for (axis, x) in point.into_iter().enumerate() {
        let frequency = if axis == 0 && periodic_x {
            R::TAU
        } else {
            R::PI
        };
        value *= (frequency * x).sin();
    }
    value
}

fn sine_defect<R: RealScalar, const D: usize>(size: usize, periodic_x: bool, graded: bool) -> f64 {
    let mesh = if graded {
        graded_mesh::<R, D>(size)
    } else {
        mesh::<R, D>(size, periodic_x)
    };
    let boundaries = zero_boundaries::<R, R, D>(&mesh, 1);
    let plan =
        StructuredFdmPlan::with_options(mesh.clone(), &Diffusion, &boundaries, limits()).unwrap();
    let full = mesh.vertex_extent().element_count().unwrap();
    let mut values = Vec::new();
    let mut eigenvalue = R::ZERO;
    for axis in 0..D {
        let frequency = if axis == 0 && periodic_x {
            R::TAU
        } else {
            R::PI
        };
        let k = if axis == 0 { R::TWO } else { R::ONE };
        eigenvalue += k * frequency * frequency;
    }
    for linear in 0..full {
        let point = *mesh
            .point(mesh.vertex_extent().delinearize(linear).unwrap())
            .unwrap()
            .as_array();
        values.push(sine(point, periodic_x));
    }
    let rhs = plan
        .load(
            &nodal(move |point, _| eigenvalue * sine(point, periodic_x)),
            R::ZERO,
        )
        .unwrap();
    let state_field = VertexField::new(&mesh, values).unwrap();
    let mut input = Vec::new();
    for node in 0..plan.points.len() {
        input.push(sine(plan.node_point(node).unwrap(), periodic_x));
    }
    let mut output = vec![R::ZERO; input.len()];
    let mut workspace = vec![R::ZERO; input.len()];
    plan.apply_into(&input, &mut output, &mut workspace)
        .unwrap();
    let fields = plan.apply_fields(&[state_field.view()]).unwrap();
    assert_eq!(fields[0].len(), full);
    let mut error = 0.0_f64;
    for node in 0..input.len() {
        if !plan.operator.constrained[node] {
            let defect = ((output[node] - rhs[node]) / plan.volumes[node])
                .into_f64()
                .abs();
            error = error.max(defect);
        }
    }
    error
}

#[test]
fn manufactured_order_affine_2d_3d_and_periodic() {
    for (dimension, periodic) in [(2, false), (3, false), (2, true)] {
        let errors = if dimension == 2 {
            [
                sine_defect::<f64, 2>(9, periodic, false),
                sine_defect::<f64, 2>(17, periodic, false),
            ]
        } else {
            [
                sine_defect::<f64, 3>(7, periodic, false),
                sine_defect::<f64, 3>(13, periodic, false),
            ]
        };
        assert!(
            errors[0] / errors[1] > 3.2,
            "dimension={dimension} periodic={periodic} {errors:?}"
        );
    }
    let errors = [
        sine_defect::<f32, 2>(9, false, false),
        sine_defect::<f32, 2>(17, false, false),
    ];
    assert!(errors[0] / errors[1] > 3.0, "f32 {errors:?}");
    let errors = [
        sine_defect::<f32, 3>(7, false, false),
        sine_defect::<f32, 3>(13, false, false),
    ];
    assert!(errors[0] / errors[1] > 3.0, "f32 3D {errors:?}");
    let errors = [
        sine_defect::<f64, 2>(9, false, true),
        sine_defect::<f64, 2>(17, false, true),
    ];
    assert!(errors[0] / errors[1] > 3.0, "graded 2D {errors:?}");
    let errors = [
        sine_defect::<f64, 3>(7, false, true),
        sine_defect::<f64, 3>(13, false, true),
    ];
    assert!(errors[0] / errors[1] > 3.0, "graded 3D {errors:?}");
}

fn quadratic<const D: usize>(point: [f64; D]) -> f64 {
    point.into_iter().map(|x| x * x).sum()
}

#[test]
fn mixed_boundaries_and_conservation() {
    for dimension in [2, 3] {
        if dimension == 2 {
            quadratic_case::<2>();
        } else {
            quadratic_case::<3>();
        }
    }
}

fn quadratic_case<const D: usize>() {
    let mesh = mesh::<f64, D>(9, false);
    let mut conditions = Vec::new();
    for axis in 0..D {
        for side in [AxisSide::Start, AxisSide::End] {
            let region = BoundarySet::axis_side(&mesh, axis, side).unwrap();
            if axis == 0 && side == AxisSide::Start {
                conditions.push(BoundaryCondition::dirichlet(
                    region,
                    BoundaryData::time_invariant(|p: crate::BoundaryPoint<f64, D>| {
                        Ok(quadratic(p.point))
                    }),
                ));
            } else if axis == 0 && side == AxisSide::End {
                // du/dn + (2/3) u = (2q + 6)/3 is 2u + 3 du/dn = 2q + 6 at x = 1.
                conditions.push(BoundaryCondition::robin(
                    region,
                    2.0 / 3.0,
                    BoundaryData::time_invariant(|p: crate::BoundaryPoint<f64, D>| {
                        Ok((2.0 * quadratic(p.point) + 6.0) / 3.0)
                    }),
                ));
            } else {
                // Unit diffusion: the conormal flux is the outward derivative.
                conditions.push(BoundaryCondition::natural(
                    region,
                    BoundaryData::time_invariant(move |p: crate::BoundaryPoint<f64, D>| {
                        Ok(2.0 * p.point[axis] * p.normal[axis])
                    }),
                ));
            }
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    struct Unit;
    impl<const D: usize> StructuredFdmCoefficients<f64, f64, D> for Unit {
        fn diffusion(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn advection(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn reaction(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
    }
    let plan = StructuredFdmPlan::with_options(mesh, &Unit, &boundaries, limits()).unwrap();
    let rhs = plan
        .load(&FieldData::constant(-((2 * D) as f64)), 0.0)
        .unwrap();
    let input: Vec<_> = plan.points.iter().copied().map(quadratic).collect();
    let mut result = vec![0.0; input.len()];
    let mut workspace = result.clone();
    plan.apply_into(&input, &mut result, &mut workspace)
        .unwrap();
    for (actual, expected) in result.iter().zip(rhs) {
        assert!((actual - expected).abs() < 2e-13, "{actual} != {expected}");
    }
    let csr = plan.assemble_csr().unwrap();
    let mut sparse_result = vec![0.0; input.len()];
    let mut sparse_workspace = vec![0.0; csr.apply_workspace_len().unwrap()];
    csr.apply_into(&input, &mut sparse_result, &mut sparse_workspace)
        .unwrap();
    for (actual, expected) in sparse_result.iter().zip(result) {
        assert!((actual - expected).abs() < 2e-13);
    }
}

#[test]
fn pure_neumann_conserves_and_checks_compatibility() {
    let mesh = mesh::<f64, 2>(11, false);
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::natural(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                BoundaryData::time_invariant(move |p: crate::BoundaryPoint<f64, 2>| {
                    Ok(2.0 * p.point[axis] * p.normal[axis])
                }),
            ));
        }
    }
    struct Unit;
    impl StructuredFdmCoefficients<f64, f64, 2> for Unit {
        fn diffusion(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn advection(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn reaction(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFdmPlan::with_options(mesh, &Unit, &boundaries, limits()).unwrap();
    assert!(plan.has_constant_modes());
    let input: Vec<_> = (0..plan.rows()).map(|index| (index % 17) as f64).collect();
    let mut output = vec![0.0; plan.rows()];
    let mut workspace = output.clone();
    plan.apply_into(&input, &mut output, &mut workspace)
        .unwrap();
    assert!(output.iter().sum::<f64>().abs() < 1e-11);
    let mut rhs = plan.load(&FieldData::constant(-4.0), 0.0).unwrap();
    assert!(rhs.iter().sum::<f64>().abs() < 1e-12);
    plan.check_constant_mode_compatibility(&rhs, Tolerance::from_sqrt_epsilon())
        .unwrap();
    rhs[0] += 0.1;
    assert!(
        plan.check_constant_mode_compatibility(&rhs, Tolerance::from_sqrt_epsilon())
            .is_err()
    );
}

#[test]
fn centered_advection_and_sparse_memory_scale() {
    struct Transport;
    impl StructuredFdmCoefficients<f64, f64, 2> for Transport {
        fn diffusion(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn advection(&self, axis: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(if axis == 0 { 1.5 } else { -0.5 })
        }
        fn reaction(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
    }
    let mut previous_nnz = None;
    for size in [9, 17] {
        let mesh = mesh::<f64, 2>(size, false);
        let boundaries = dirichlet_boundaries(
            &mesh,
            1,
            BoundaryData::time_invariant(|p: crate::BoundaryPoint<f64, 2>| {
                Ok(p.point[0] + 2.0 * p.point[1])
            }),
        );
        let plan =
            StructuredFdmPlan::with_options(mesh, &Transport, &boundaries, limits()).unwrap();
        let csr = plan.assemble_csr().unwrap();
        assert!(csr.nnz() <= plan.rows() * 5);
        if let Some(previous) = previous_nnz {
            assert!(csr.nnz() > previous);
        }
        previous_nnz = Some(csr.nnz());
        let input: Vec<_> = plan
            .points
            .iter()
            .map(|point| point[0] + 2.0 * point[1])
            .collect();
        let mut result = vec![0.0; plan.rows()];
        let mut workspace = result.clone();
        plan.apply_into(&input, &mut result, &mut workspace)
            .unwrap();
        // Eliminated free rows: A u = f vol − L g with f = v·∇u = 0.5.
        let load = plan.load(&FieldData::constant(0.5), 0.0).unwrap();
        for (node, (actual, expected)) in result.iter().zip(&load).enumerate() {
            assert!((actual - expected).abs() < 3e-14, "{node}");
            let (i, j) = (node % size, node / size);
            if (2..size - 2).contains(&i) && (2..size - 2).contains(&j) {
                assert!((actual - 0.5 * plan.volumes[node]).abs() < 3e-14);
            }
        }
    }
}

#[test]
fn coupled_complex_fields_and_resource_rejection() {
    let mesh = mesh::<f64, 2>(5, false);
    let boundaries = zero_boundaries::<f64, Complex<f64>, 2>(&mesh, 2);
    struct Coupled;
    impl StructuredFdmCoefficients<f64, Complex<f64>, 2> for Coupled {
        fn diffusion(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn advection(&self, axis: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(if axis == 0 { 0.25 } else { -0.5 })
        }
        fn reaction(
            &self,
            row: usize,
            col: usize,
            _: [f64; 2],
        ) -> Result<Complex<f64>, SciOxideError> {
            Ok(if row == col {
                Complex::new(2.0, 0.0)
            } else {
                Complex::new(0.5, -0.25)
            })
        }
    }
    assert!(
        StructuredFdmPlan::with_options(
            mesh.clone(),
            &Coupled,
            &boundaries,
            SpatialLimits::new(1, 1, 1)
        )
        .is_err()
    );
    let plan = StructuredFdmPlan::with_options(mesh, &Coupled, &boundaries, limits()).unwrap();
    let input = vec![Complex::new(1.0, 0.5); plan.rows()];
    let mut actual = vec![Complex::new(0.0, 0.0); input.len()];
    let mut workspace = actual.clone();
    plan.apply_into(&input, &mut actual, &mut workspace)
        .unwrap();
    let csr = plan.assemble_csr().unwrap();
    let mut expected = workspace.clone();
    csr.apply_into(&input, &mut expected, &mut workspace)
        .unwrap();
    for (left, right) in actual.into_iter().zip(expected) {
        assert!((left - right).abs() < 1e-12);
    }
}

fn complex_quadratic<R: RealScalar, const D: usize>(point: [R; D]) -> R {
    point
        .into_iter()
        .map(|x| x * x)
        .fold(R::ZERO, |sum, x| sum + x)
}

fn complex_accuracy<R: RealScalar, const D: usize>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let mesh = mesh::<R, D>(7, false);
    let one = R::ONE;
    let half = R::HALF;
    let amplitudes = [Complex::new(one, one), Complex::new(one + one, -half)];
    let coupling = Complex::new(half, -half);
    let boundaries = dirichlet_boundaries(
        &mesh,
        2,
        BoundaryData::time_invariant(move |p: crate::BoundaryPoint<R, D>| {
            Ok(amplitudes[p.component] * Complex::from_real(complex_quadratic(p.point)))
        }),
    );
    struct Model<R: RealScalar> {
        coupling: Complex<R>,
    }
    impl<R: RealScalar, const D: usize> StructuredFdmCoefficients<R, Complex<R>, D> for Model<R>
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        fn diffusion(&self, _: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
            Ok(R::ONE)
        }
        fn advection(&self, _: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
            Ok(R::ZERO)
        }
        fn reaction(
            &self,
            row: usize,
            column: usize,
            _: [R; D],
        ) -> Result<Complex<R>, SciOxideError> {
            Ok(if row == column {
                Complex::from_real(R::ONE)
            } else {
                self.coupling
            })
        }
    }
    let plan =
        StructuredFdmPlan::with_options(mesh, &Model { coupling }, &boundaries, limits()).unwrap();
    let laplacian = R::try_from_usize_exact(2 * D).unwrap();
    let rhs = plan
        .load(
            &nodal(move |point, row| {
                let q = Complex::from_real(complex_quadratic(point));
                -amplitudes[row] * Complex::from_real(laplacian)
                    + amplitudes[row] * q
                    + coupling * amplitudes[1 - row] * q
            }),
            R::ZERO,
        )
        .unwrap();
    let input: Vec<_> = amplitudes
        .iter()
        .flat_map(|&amplitude| {
            plan.points
                .iter()
                .map(move |&point| amplitude * Complex::from_real(complex_quadratic(point)))
        })
        .collect();
    let mut output = vec![Complex::ZERO; input.len()];
    let mut workspace = output.clone();
    plan.apply_into(&input, &mut output, &mut workspace)
        .unwrap();
    let tolerance = if R::EPSILON.into_f64() > 1e-10 {
        3e-5
    } else {
        2e-13
    };
    for (actual, expected) in output.iter().zip(rhs) {
        assert!((*actual - expected).abs().into_f64() < tolerance);
    }
}

#[test]
fn coupled_complex_manufactured_both_precisions() {
    complex_accuracy::<f32, 2>();
    complex_accuracy::<f64, 2>();
    complex_accuracy::<f32, 3>();
    complex_accuracy::<f64, 3>();
}

#[test]
fn invalid_mesh_and_nonfinite_input_are_transactional() {
    let mesh = mesh::<f64, 2>(5, false);
    let boundaries = zero_boundaries::<f64, f64, 2>(&mesh, 1);
    struct Counting<'a>(&'a core::cell::Cell<usize>);
    impl StructuredFdmCoefficients<f64, f64, 2> for Counting<'_> {
        fn diffusion(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            self.0.set(self.0.get() + 1);
            Ok(1.0)
        }
        fn advection(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn reaction(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
    }
    let calls = core::cell::Cell::new(0);
    assert!(
        StructuredFdmPlan::with_options(
            mesh.clone(),
            &Counting(&calls),
            &boundaries,
            SpatialLimits::new(1, 1, 1),
        )
        .is_err()
    );
    assert_eq!(calls.get(), 0);
    let plan = StructuredFdmPlan::with_options(mesh, &Diffusion, &boundaries, limits()).unwrap();
    let different = self::mesh::<f64, 2>(5, false);
    let foreign = VertexField::from_fn(&different, |_| 0.0).unwrap();
    assert!(plan.apply_fields(&[foreign.view()]).is_err());
    let mut input = vec![0.0; plan.rows()];
    input[0] = f64::NAN;
    let mut output = vec![7.0; plan.rows()];
    let mut workspace = vec![0.0; plan.rows()];
    assert!(
        plan.apply_into(&input, &mut output, &mut workspace)
            .is_err()
    );
    assert!(output.iter().all(|&value| value == 7.0));
    input[0] = 0.0;
    let cancelled = scioxide_core::CancellationToken::new();
    cancelled.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&cancelled);
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "apply", 1),
        0,
        0,
    );
    assert!(
        SolveScope::from(context)
            .run(
                &mut SolveState::new(),
                identity,
                SolveBudgets::unbounded(),
                |cx| {
                    LinearOperator::apply_into_scoped(
                        &plan,
                        &input,
                        &mut output,
                        &mut workspace,
                        &mut OperatorContext::new(cx),
                    )?
                    .require()
                }
            )
            .is_err()
    );
    assert!(output.iter().all(|&value| value == 7.0));
}

#[test]
fn boundary_structure_and_corner_conflicts_are_rejected() {
    let mesh = mesh::<f64, 2>(5, false);
    let zero = || BoundaryData::<f64, f64, 2>::zero();
    let incomplete = vec![BoundaryCondition::dirichlet(
        BoundarySet::axis_side(&mesh, 0, AxisSide::Start).unwrap(),
        zero(),
    )];
    assert!(StructuredBoundaries::new(&mesh, 1, incomplete).is_err());
    let face = mesh
        .axis_boundary_faces(0, AxisSide::Start)
        .unwrap()
        .next()
        .unwrap();
    let partial = BoundarySet::new(&mesh, vec![face]).unwrap();
    let mut conditions = vec![BoundaryCondition::dirichlet(partial, zero())];
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                zero(),
            ));
        }
    }
    assert!(StructuredBoundaries::new(&mesh, 1, conditions).is_err());
    let conflicting = || {
        let mut conditions = Vec::new();
        for axis in 0..2 {
            for side in [AxisSide::Start, AxisSide::End] {
                conditions.push(BoundaryCondition::dirichlet(
                    BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                    BoundaryData::constant(if axis == 1 && side == AxisSide::Start {
                        1.0
                    } else {
                        0.0
                    }),
                ));
            }
        }
        StructuredBoundaries::new(&mesh, 1, conditions).unwrap()
    };
    // Conflicts appear when data is evaluated, at every time.
    let plan = StructuredFdmPlan::with_options(mesh.clone(), &Diffusion, &conflicting(), limits())
        .unwrap();
    assert!(plan.load(&FieldData::zero(), 0.0).is_err());
    // The later condition (axis 1, start) wins its corners when requested.
    let boundaries =
        conflicting().with_dirichlet_conflict_policy(DirichletConflictPolicy::LastConditionWins);
    let plan =
        StructuredFdmPlan::with_options(mesh.clone(), &Diffusion, &boundaries, limits()).unwrap();
    let load = plan.load(&FieldData::zero(), 0.0).unwrap();
    assert_eq!(load[0], 1.0);
    assert_eq!(load[4], 1.0);
    assert_eq!(load[20], 0.0);
}

#[test]
fn one_dimensional_plan_is_tridiagonal_and_lumped() {
    let mesh = mesh::<f64, 1>(5, false);
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&mesh, 0, AxisSide::Start).unwrap(),
                BoundaryData::constant(1.0),
            ),
            BoundaryCondition::robin(
                BoundarySet::axis_side(&mesh, 0, AxisSide::End).unwrap(),
                2.0,
                BoundaryData::constant(3.0),
            ),
        ],
    )
    .unwrap();
    let plan = StructuredFdmPlan::with_options(mesh, &Diffusion, &boundaries, limits()).unwrap();
    let csr = plan.assemble_csr().unwrap();
    // k = 2 on axis 0, h = 1/4: conductance 8, symmetric elimination of node 0.
    assert_eq!(csr.get(0, 0).unwrap(), 1.0);
    assert_eq!(csr.get(0, 1).unwrap(), 0.0);
    assert_eq!(csr.get(1, 0).unwrap(), 0.0);
    assert_eq!(csr.get(1, 1).unwrap(), 16.0);
    assert_eq!(csr.get(4, 4).unwrap(), 8.0 + 2.0);
    for (row, column, value) in csr.iter() {
        assert_eq!(csr.get(column, row).unwrap(), value);
    }
    let load = plan.load(&FieldData::constant(4.0), 0.0).unwrap();
    assert_eq!(load, vec![1.0, 1.0 + 8.0, 1.0, 1.0, 0.5 + 3.0]);
    assert_eq!(plan.lumped_mass(), &[0.125, 0.25, 0.25, 0.25, 0.125]);
}
