use super::*;
use crate::boundary::structured_face_at;
use crate::{BoundaryCondition, BoundaryData};
use scioxide_core::{
    Cancellation, Complex32, Complex64, OperatorContext, ProblemToken, SolveBudgets, SolveIdentity,
    SolveScope, SolveState,
};
use scioxide_linalg::{LuDecomposition, Matrix, Vector};
use scioxide_spatial::{
    AxisSide, CoordinateAxis, Point, StructuredGrid, TetrahedronMesh, TriangleMesh,
};
use scioxide_spatial::{BoundarySet, SimplexBoundaryTags};

fn limits() -> SpatialLimits {
    SpatialLimits::new(100_000, 20_000_000, 1_000_000_000)
}

fn data<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>(
    f: impl Fn([R; D]) -> Result<V, SciOxideError> + Send + Sync + 'static,
) -> BoundaryData<R, V, D> {
    BoundaryData::time_invariant(move |p: BoundaryPoint<R, D>| f(p.point))
}

fn dirichlet<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
    Region,
>(
    region: Region,
    f: impl Fn([R; D]) -> Result<V, SciOxideError> + Send + Sync + 'static,
) -> BoundaryCondition<R, V, D, Region> {
    BoundaryCondition::dirichlet(region, data(f))
}

fn neumann<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
    Region,
>(
    region: Region,
    f: impl Fn([R; D]) -> Result<V, SciOxideError> + Send + Sync + 'static,
) -> BoundaryCondition<R, V, D, Region> {
    BoundaryCondition::natural(region, data(f))
}

/// `alpha u + beta K∇u·n = g` as `K∇u·n + (alpha/beta) u = g/beta`.
fn robin<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
    Region,
>(
    region: Region,
    alpha: R,
    beta: R,
    f: impl Fn([R; D]) -> Result<V, SciOxideError> + Send + Sync + 'static,
) -> BoundaryCondition<R, V, D, Region> {
    BoundaryCondition::robin(
        region,
        V::from_real(alpha / beta),
        data(move |point| Ok(f(point)? / V::from_real(beta))),
    )
}

fn field<R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static, const D: usize>(
    f: impl Fn([R; D]) -> V + Send + Sync + 'static,
) -> FieldData<R, V, D> {
    FieldData::time_invariant(move |p: FieldPoint<R, D>| Ok(f(p.point)))
}

struct ConstantTensor<R>(R);
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> FvmCoefficients<R, V, D>
    for ConstantTensor<R>
{
    fn diffusion(&self, _: [R; D]) -> Result<[[R; D]; D], SciOxideError> {
        Ok(core::array::from_fn(|row| {
            core::array::from_fn(|column| if row == column { R::ONE } else { self.0 })
        }))
    }
    fn reaction(&self, _: [R; D]) -> Result<V, SciOxideError> {
        Ok(V::ZERO)
    }
}

fn structured_mesh<R: RealScalar, const D: usize>(sites: usize) -> CellMesh<R, D> {
    let end = R::ONE;
    let axes = core::array::from_fn(|_| CoordinateAxis::uniform(R::ZERO, end, sites).unwrap());
    CellMesh::new(StructuredGrid::new(axes, [AxisTopology::Bounded; D]).unwrap()).unwrap()
}

fn affine<R: RealScalar, const D: usize>(point: [R; D]) -> R {
    point
        .into_iter()
        .fold(R::ONE, |sum, coordinate| sum + coordinate)
}

fn structured_centroid<R: RealScalar, const D: usize>(
    mesh: &CellMesh<R, D>,
    cell: usize,
) -> [R; D] {
    let index = mesh.cell_extent().delinearize(cell).unwrap();
    core::array::from_fn(|axis| {
        let grid_axis = mesh.grid().axis(axis).unwrap();
        let i = index.as_array()[axis];
        (grid_axis.coordinate(i).unwrap() + grid_axis.coordinate(i + 1).unwrap()) * R::HALF
    })
}

fn structured_affine_boundaries<R: RealScalar, const D: usize>(
    mesh: &CellMesh<R, D>,
) -> StructuredBoundaries<R, R, D> {
    let mut conditions = Vec::new();
    for axis in 0..D {
        for side in [AxisSide::Start, AxisSide::End] {
            let region = BoundarySet::axis_side(mesh, axis, side).unwrap();
            let flux_scale = R::ONE
                + R::try_from_f64_lossy(0.125).unwrap() * R::try_from_usize_exact(D - 1).unwrap();
            let derivative = if side == AxisSide::Start {
                -flux_scale
            } else {
                flux_scale
            };
            if axis == 0 && side == AxisSide::Start {
                conditions.push(dirichlet(region, |point| Ok(affine(point))));
            } else if axis == 0 && side == AxisSide::End {
                conditions.push(robin(region, R::ONE, R::ONE, move |point| {
                    Ok(affine(point) + derivative)
                }));
            } else {
                conditions.push(neumann(region, move |_| Ok(derivative)));
            }
        }
    }
    StructuredBoundaries::new(mesh, 1, conditions).unwrap()
}

fn check_structured_affine<R: RealScalar, const D: usize>() {
    let mesh = structured_mesh::<R, D>(5);
    let boundaries = structured_affine_boundaries(&mesh);
    let coupling = R::try_from_f64_lossy(0.125).unwrap();
    let plan = StructuredFvmPlan::with_options(
        mesh.clone(),
        &ConstantTensor(coupling),
        &boundaries,
        limits(),
    )
    .unwrap();
    let values =
        CellField::from_fn(&mesh, |cell| affine(structured_centroid(&mesh, cell))).unwrap();
    let rhs = plan.load(&FieldData::zero(), R::ZERO).unwrap();
    let mut output = vec![R::ZERO; plan.unknowns()];
    let mut workspace = output.clone();
    plan.apply_into(values.values(), &mut output, &mut workspace)
        .unwrap();
    let tolerance = R::EPSILON.into_f64() * 1000.0;
    let defect = output
        .iter()
        .zip(&rhs)
        .map(|(&a, &b)| (a - b).abs().into_f64())
        .fold(0.0_f64, f64::max);
    assert!(defect <= tolerance, "affine integrated defect {defect:e}");
    let csr = plan.assemble_csr().unwrap();
    assert!(csr.nnz() <= plan.entry_bound());
    let mut dense = vec![R::ZERO; plan.unknowns() * plan.unknowns()];
    csr.to_dense_into(&mut dense).unwrap();
    for row in 0..plan.unknowns() {
        let expected = dense[row * plan.unknowns()..(row + 1) * plan.unknowns()]
            .iter()
            .zip(values.values())
            .fold(R::ZERO, |sum, (&a, &u)| sum + a * u);
        assert!((expected - output[row]).abs().into_f64() <= tolerance);
    }
    let result = plan.apply_field(values.view()).unwrap();
    assert!(result.is_compatible_with(&mesh));
    let mut flux = vec![R::ZERO; mesh.topology_sizes().faces()];
    plan.face_fluxes_into(
        values.view(),
        R::ZERO,
        &mut flux,
        &mut plan.workspace().unwrap(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let net = flux.iter().enumerate().fold(R::ZERO, |sum, (id, &q)| {
        let face = structured_face_at(&mesh, id).unwrap();
        if mesh.face_cells(face).unwrap().len() != 1 {
            sum
        } else if face.index().as_array()[face.normal_axis()] == 0 {
            sum - q
        } else {
            sum + q
        }
    });
    assert!(net.abs().into_f64() <= tolerance);
    let geometry = structured_geometry(&mesh).unwrap();
    for cell in 0..plan.unknowns() {
        let outward = geometry.cell_faces[cell]
            .iter()
            .map(|&face| {
                if geometry.faces[face].left == Some(cell) {
                    flux[face]
                } else {
                    -flux[face]
                }
            })
            .fold(R::ZERO, |sum, value| sum + value);
        assert!(outward.abs().into_f64() <= tolerance);
    }
}

#[test]
fn structured_affine_anisotropic_mixed_f32_f64_2d_3d() {
    check_structured_affine::<f32, 2>();
    check_structured_affine::<f64, 2>();
    check_structured_affine::<f32, 3>();
    check_structured_affine::<f64, 3>();
}

fn triangle_mesh() -> TriangleMesh<f64> {
    let points = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        .into_iter()
        .map(|coordinates| Point::new(coordinates).unwrap())
        .collect();
    TriangleMesh::new(points, vec![[0, 1, 2], [0, 2, 3]]).unwrap()
}

fn tetrahedron_mesh() -> TetrahedronMesh<f64> {
    let points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        .into_iter()
        .map(|coordinates| Point::new(coordinates).unwrap())
        .collect();
    TetrahedronMesh::new(points, vec![[0, 1, 2, 3]]).unwrap()
}

fn check_simplex_affine<const D: usize>(mesh: SimplexMesh<f64, D>) {
    let geometry = simplex_geometry(&mesh).unwrap();
    let entries: Vec<_> = mesh
        .boundary_faces()
        .map(|face| (face, face as u32))
        .collect();
    let tags = SimplexBoundaryTags::new(&mesh, entries).unwrap();
    let mut conditions = Vec::new();
    for face in mesh.boundary_faces() {
        let data = &geometry.faces[face];
        if face % 3 == 0 {
            conditions.push(dirichlet(face as u32, |point| Ok(affine(point))));
        } else if face % 3 == 1 {
            let derivative: f64 =
                data.normal.into_iter().sum::<f64>() * (1. + 0.125 * (D - 1) as f64);
            conditions.push(neumann(face as u32, move |_| Ok(derivative)));
        } else {
            let derivative: f64 =
                data.normal.into_iter().sum::<f64>() * (1. + 0.125 * (D - 1) as f64);
            conditions.push(robin(face as u32, 1., 1., move |point| {
                Ok(affine(point) + derivative)
            }));
        }
    }
    let boundaries = TaggedBoundaries::new(&mesh, &tags, 1, conditions).unwrap();
    let plan =
        SimplexFvmPlan::with_options(mesh.clone(), &ConstantTensor(0.125), &boundaries, limits())
            .unwrap();
    let values = CellField::from_fn(&mesh, |cell| affine(geometry.centers[cell])).unwrap();
    let rhs = plan.load(&FieldData::zero(), 0.).unwrap();
    let mut output = vec![0.; plan.unknowns()];
    let mut scratch = output.clone();
    plan.apply_into(values.values(), &mut output, &mut scratch)
        .unwrap();
    for (&a, &b) in output.iter().zip(&rhs) {
        assert!((a - b).abs() < 1e-11);
    }
    let flux = plan.face_fluxes(values.view(), 0.).unwrap();
    assert!(flux.is_compatible_with(&mesh));
    assert_eq!(flux.len(), mesh.face_count());
    let sum: f64 = mesh.boundary_faces().map(|face| flux.values()[face]).sum();
    assert!(sum.abs() < 1e-11);
}

#[test]
fn simplex_affine_anisotropic_mixed_2d_3d() {
    check_simplex_affine(triangle_mesh());
    check_simplex_affine(tetrahedron_mesh());
}

#[test]
fn complex_operator_and_field_identity() {
    let mesh = structured_mesh::<f64, 2>(3);
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                |_| Ok(Complex64::new(0., 0.)),
            ));
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFvmPlan::with_options(
        mesh.clone(),
        &ConstantTensor(0.125),
        &boundaries,
        limits(),
    )
    .unwrap();
    let values = CellField::from_fn(&mesh, |cell| Complex64::new(cell as f64, 1.)).unwrap();
    let mut output = vec![Complex64::new(0., 0.); plan.unknowns()];
    let mut scratch = output.clone();
    plan.apply_into(values.values(), &mut output, &mut scratch)
        .unwrap();
    let csr = plan.assemble_csr().unwrap();
    let mut dense = vec![Complex64::new(0., 0.); plan.unknowns() * plan.unknowns()];
    csr.to_dense_into(&mut dense).unwrap();
    for row in 0..plan.unknowns() {
        let expected = dense[row * plan.unknowns()..(row + 1) * plan.unknowns()]
            .iter()
            .zip(values.values())
            .fold(Complex64::new(0., 0.), |sum, (&a, &u)| sum + a * u);
        assert!((expected - output[row]).abs() < 1e-12);
    }
    let other = structured_mesh::<f64, 2>(3);
    let foreign = CellField::from_fn(&other, |_| Complex64::new(0., 0.)).unwrap();
    assert!(plan.apply_field(foreign.view()).is_err());
    assert!(plan.face_fluxes::<0>(values.view(), 0.).is_ok());
}

#[test]
fn rejected_shapes_limits_and_transactional_output() {
    let mesh = structured_mesh::<f64, 2>(3);
    let boundaries = structured_affine_boundaries(&mesh);
    assert!(
        StructuredFvmPlan::with_options(
            mesh.clone(),
            &ConstantTensor(0.125),
            &boundaries,
            SpatialLimits::new(1, 1000, 100000)
        )
        .is_err()
    );
    let plan = StructuredFvmPlan::with_options(
        mesh.clone(),
        &ConstantTensor(0.125),
        &boundaries,
        limits(),
    )
    .unwrap();
    let input = vec![0.; plan.unknowns()];
    let mut output = vec![7.; plan.unknowns()];
    let mut scratch = vec![0.; plan.unknowns()];
    assert!(
        plan.apply_into(&input[..1], &mut output, &mut scratch)
            .is_err()
    );
    assert!(output.iter().all(|&value| value == 7.));
    let values = CellField::new(&mesh, input).unwrap();
    let mut flux = vec![7.; mesh.topology_sizes().faces()];
    let mut flux_workspace = plan.workspace().unwrap();
    assert!(
        plan.face_fluxes_into(
            values.view(),
            0.,
            &mut flux[..1],
            &mut flux_workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(flux[0], 7.);
}

struct Cancelled;
impl Cancellation for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

struct InvalidTensor;
impl FvmCoefficients<f64, f64, 2> for InvalidTensor {
    fn diffusion(&self, _: [f64; 2]) -> Result<[[f64; 2]; 2], SciOxideError> {
        Ok([[1., 2.], [2., 1.]])
    }
    fn reaction(&self, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(0.)
    }
}

#[test]
fn rejects_invalid_tensor_and_cancelled_apply_without_committing() {
    let mesh = structured_mesh::<f64, 2>(3);
    let boundaries = structured_affine_boundaries(&mesh);
    assert!(
        StructuredFvmPlan::with_options(mesh.clone(), &InvalidTensor, &boundaries, limits())
            .is_err()
    );
    let plan = StructuredFvmPlan::with_options(mesh, &ConstantTensor(0.125), &boundaries, limits())
        .unwrap();
    let input = vec![1.; plan.unknowns()];
    let mut output = vec![7.; plan.unknowns()];
    let mut workspace = vec![0.; plan.unknowns()];
    let context = ExecutionContext::sequential().with_cancellation(&Cancelled);
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
    assert!(output.iter().all(|&value| value == 7.));
}

#[test]
fn pure_neumann_mode_and_compatibility() {
    let mesh = structured_mesh::<f64, 2>(4);
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(neumann(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                |_| Ok(0.),
            ));
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFvmPlan::with_options(mesh, &ConstantTensor(0.125), &boundaries, limits())
        .unwrap();
    assert!(plan.has_constant_modes());
    let rhs = plan.load(&FieldData::zero(), 0.).unwrap();
    plan.check_constant_mode_compatibility(&rhs, Tolerance::from_machine_epsilon())
        .unwrap();
    let mut bad_rhs = rhs;
    bad_rhs[0] = 1.;
    assert!(
        plan.check_constant_mode_compatibility(&bad_rhs, Tolerance::from_machine_epsilon())
            .is_err()
    );
    let input = vec![1.; plan.unknowns()];
    let mut output = vec![0.; plan.unknowns()];
    let mut workspace = output.clone();
    plan.apply_into(&input, &mut output, &mut workspace)
        .unwrap();
    assert!(output.iter().all(|value| value.abs() < 1e-12));
}

fn nullspace_load(matrix: CsrMatrix<f64>) -> Vec<f64> {
    let n = matrix.rows();
    let mut columns = vec![0.; n];
    let mut load = vec![0.; n];
    let scale = matrix.values().iter().fold(0.0_f64, |s, a| s.max(a.abs()));
    for (row, b) in load.iter_mut().enumerate() {
        let mut sum = 0.;
        for index in matrix.row_ptr()[row]..matrix.row_ptr()[row + 1] {
            let col = matrix.col_indices()[index];
            let value = matrix.values()[index];
            sum += value;
            columns[col] += value;
            *b += value * ((col * 17 % 97) as f64 / 53.);
        }
        assert!(sum.abs() < 512. * f64::EPSILON * scale.max(1.));
    }
    assert!(
        columns
            .iter()
            .all(|v| v.abs() < 512. * f64::EPSILON * scale.max(1.))
    );
    load
}

#[test]
fn anisotropic_neumann_conserves_columns_and_accepts_operator_loads() {
    for coupling in [0., 0.5, 0.9] {
        for count in [8, 32] {
            let mesh = structured_mesh::<f64, 2>(count + 1);
            let mesh_ref = &mesh;
            let conditions = (0..2)
                .flat_map(|axis| {
                    [AxisSide::Start, AxisSide::End].map(move |side| {
                        neumann(
                            BoundarySet::axis_side(mesh_ref, axis, side).unwrap(),
                            |_| Ok(0.),
                        )
                    })
                })
                .collect();
            let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
            let plan = StructuredFvmPlan::with_options(
                mesh,
                &ConstantTensor(coupling),
                &boundaries,
                limits(),
            )
            .unwrap();
            let mut load = nullspace_load(plan.assemble_csr().unwrap());
            plan.check_constant_mode_compatibility(&load, Tolerance::new(1e-10, 1e-12).unwrap())
                .unwrap();
            load[0] += 1.;
            assert!(
                plan.check_constant_mode_compatibility(
                    &load,
                    Tolerance::new(1e-10, 1e-12).unwrap()
                )
                .is_err()
            );
        }
        let mesh = crate::fem::simplex::meshes::triangle_grid::<f64>(4).unwrap();
        let tags =
            SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|id| (id, 0)).collect())
                .unwrap();
        let boundaries =
            TaggedBoundaries::new(&mesh, &tags, 1, vec![neumann(0, |_| Ok(0.))]).unwrap();
        let plan =
            SimplexFvmPlan::with_options(mesh, &ConstantTensor(coupling), &boundaries, limits())
                .unwrap();
        let mut load = nullspace_load(plan.assemble_csr().unwrap());
        plan.check_constant_mode_compatibility(&load, Tolerance::new(1e-10, 1e-12).unwrap())
            .unwrap();
        load[0] += 1.;
        assert!(
            plan.check_constant_mode_compatibility(&load, Tolerance::new(1e-10, 1e-12).unwrap())
                .is_err()
        );
    }
}

struct JumpTensor;
impl FvmCoefficients<f64, f64, 2> for JumpTensor {
    fn diffusion(&self, point: [f64; 2]) -> Result<[[f64; 2]; 2], SciOxideError> {
        if point[0] == 0.5 {
            return Err(invalid(
                scioxide_core::argument!("point"),
                "material is undefined on the interface",
            ));
        }
        let k = if point[0] < 0.5 { 1. } else { 100. };
        Ok([[k, 0.], [0., k]])
    }
    fn reaction(&self, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(0.)
    }
}
fn jump_profile(point: [f64; 2]) -> f64 {
    let s = if point[0] < 0.5 {
        point[0]
    } else {
        0.5 + (point[0] - 0.5) / 100.
    };
    s * (core::f64::consts::PI * point[1]).sin()
}
fn jump_error(cells: usize) -> f64 {
    let mesh = structured_mesh::<f64, 2>(cells + 1);
    let mesh_ref = &mesh;
    let conditions = (0..2)
        .flat_map(|axis| {
            [AxisSide::Start, AxisSide::End].map(move |side| {
                dirichlet(BoundarySet::axis_side(mesh_ref, axis, side).unwrap(), |x| {
                    Ok(jump_profile(x))
                })
            })
        })
        .collect();
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan =
        StructuredFvmPlan::with_options(mesh.clone(), &JumpTensor, &boundaries, limits()).unwrap();
    let source = field(|x: [f64; 2]| {
        let k = if x[0] < 0.5 { 1. } else { 100. };
        k * core::f64::consts::PI.powi(2) * jump_profile(x)
    });
    let matrix = plan.assemble_csr().unwrap();
    let n = plan.unknowns();
    let mut dense = vec![0.; n * n];
    matrix.to_dense_into(&mut dense).unwrap();
    let factor = LuDecomposition::factor(&Matrix::from_vec(n, n, dense).unwrap()).unwrap();
    let solution = factor
        .solve(&Vector::from_vec(plan.load(&source, 0.).unwrap()))
        .unwrap();
    solution
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &u)| (u - jump_profile(structured_centroid(&mesh, i))).abs())
        .fold(0., f64::max)
}

#[test]
fn face_aligned_2d_material_jump_has_second_order_accuracy() {
    let errors = [4, 8, 16].map(jump_error);
    for pair in errors.windows(2) {
        assert!((pair[0] / pair[1]).log2() > 1.7, "jump errors {errors:?}");
    }
}

#[test]
fn skew_anisotropy_rejects_nonpositive_half_cell_conductance() {
    let mesh = TriangleMesh::new(
        vec![
            Point::new([0., 0.]).unwrap(),
            Point::new([10., 0.]).unwrap(),
            Point::new([0., 1.]).unwrap(),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|id| (id, 0)).collect()).unwrap();
    let boundaries =
        TaggedBoundaries::new(&mesh, &tags, 1, vec![dirichlet(0, |_| Ok(0.))]).unwrap();
    let failure = SimplexFvmPlan::with_options(mesh, &ConstantTensor(0.9), &boundaries, limits());
    assert!(matches!(failure, Err(ref e) if matches!(e.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { issue: scioxide_core::NumericalIssue::Breakdown, .. })));
}

#[test]
fn complex32_operator_is_available() {
    let mesh = structured_mesh::<f32, 2>(3);
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                |_| Ok(Complex32::new(0., 0.)),
            ));
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFvmPlan::with_options(
        mesh.clone(),
        &ConstantTensor(0.125),
        &boundaries,
        limits(),
    )
    .unwrap();
    let values = CellField::from_fn(&mesh, |_| Complex32::new(1., 1.)).unwrap();
    assert!(plan.apply_field(values.view()).is_ok());
    assert!(plan.assemble_csr().is_ok());
}

#[test]
fn boundary_coverage_and_mesh_revisions_are_checked() {
    let mesh = structured_mesh::<f64, 2>(3);
    let incomplete = StructuredBoundaries::new(
        &mesh,
        1,
        vec![dirichlet(
            BoundarySet::axis_side(&mesh, 0, AxisSide::Start).unwrap(),
            |_| Ok(0.),
        )],
    );
    assert!(incomplete.is_err());
    let simplex = triangle_mesh();
    let first = simplex.boundary_faces().next().unwrap();
    let incomplete_tags = SimplexBoundaryTags::new(&simplex, vec![(first, 7)]).unwrap();
    assert!(
        TaggedBoundaries::new(
            &simplex,
            &incomplete_tags,
            1,
            vec![dirichlet(7, |_| Ok(0.))],
        )
        .is_err()
    );
    let complete_tags = SimplexBoundaryTags::new(
        &simplex,
        simplex.boundary_faces().map(|face| (face, 7)).collect(),
    )
    .unwrap();
    let boundaries =
        TaggedBoundaries::new(&simplex, &complete_tags, 1, vec![dirichlet(7, |_| Ok(0.))]).unwrap();
    let changed = simplex
        .with_points(
            simplex
                .points()
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let mut coordinates = *point.as_array();
                    if index == 2 {
                        coordinates[0] = 1.1;
                    }
                    Point::new(coordinates).unwrap()
                })
                .collect(),
        )
        .unwrap();
    assert!(
        SimplexFvmPlan::with_options(changed, &ConstantTensor(0.125), &boundaries, limits())
            .is_err()
    );
}

fn profile<const D: usize>(point: [f64; D]) -> f64 {
    point.into_iter().map(|x| x * (1. - x)).product()
}

fn manufactured_source<const D: usize>(point: [f64; D]) -> f64 {
    let mut source = 0.;
    for row in 0..D {
        for column in 0..D {
            let transverse: f64 = point
                .iter()
                .enumerate()
                .filter(|(axis, _)| *axis != row && *axis != column)
                .map(|(_, &x)| x * (1. - x))
                .product();
            if row == column {
                source += 2. * transverse;
            } else {
                source -= 0.125 * (1. - 2. * point[row]) * (1. - 2. * point[column]) * transverse;
            }
        }
    }
    source
}

fn solve_error<const D: usize>(matrix: CsrMatrix<f64>, rhs: Vec<f64>, centers: &[[f64; D]]) -> f64 {
    let n = rhs.len();
    let mut dense = vec![0.; n * n];
    matrix.to_dense_into(&mut dense).unwrap();
    let matrix = Matrix::from_vec(n, n, dense).unwrap();
    let factor = LuDecomposition::factor(&matrix).unwrap();
    let solution = factor.solve(&Vector::from_vec(rhs)).unwrap();
    solution
        .as_slice()
        .iter()
        .zip(centers)
        .map(|(&u, &point)| (u - profile(point)).abs())
        .fold(0., f64::max)
}

fn mapped_structured<const D: usize>(cells: usize) -> CellMesh<f64, D> {
    let axes = core::array::from_fn(|_| {
        CoordinateAxis::explicit(
            (0..=cells)
                .map(|i| {
                    let x = i as f64 / cells as f64;
                    x + 0.12 * x * (1. - x)
                })
                .collect(),
        )
        .unwrap()
    });
    CellMesh::new(StructuredGrid::new(axes, [AxisTopology::Bounded; D]).unwrap()).unwrap()
}

fn structured_error<const D: usize>(cells: usize) -> f64 {
    let mesh = mapped_structured::<D>(cells);
    let mut conditions = Vec::new();
    for axis in 0..D {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                |_| Ok(0.),
            ));
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFvmPlan::with_options(
        mesh.clone(),
        &ConstantTensor(0.125),
        &boundaries,
        limits(),
    )
    .unwrap();
    let centers: Vec<_> = (0..plan.unknowns())
        .map(|cell| structured_centroid(&mesh, cell))
        .collect();
    let rhs = plan.load(&field(manufactured_source::<D>), 0.).unwrap();
    let csr = plan.assemble_csr().unwrap();
    assert!(csr.nnz() <= 40 * plan.unknowns());
    solve_error(csr, rhs, &centers)
}

fn triangular_grid(cells: usize) -> TriangleMesh<f64> {
    let mut points = Vec::new();
    for y in 0..=cells {
        for x in 0..=cells {
            let x = x as f64 / cells as f64;
            let y = y as f64 / cells as f64;
            points.push(Point::new([x + 0.08 * x * (1. - x) * y, y]).unwrap());
        }
    }
    let mut triangles = Vec::new();
    for y in 0..cells {
        for x in 0..cells {
            let a = y * (cells + 1) + x;
            let b = a + 1;
            let d = a + cells + 1;
            let c = d + 1;
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
    }
    TriangleMesh::new(points, triangles).unwrap()
}

fn tetrahedral_grid(cells: usize) -> TetrahedronMesh<f64> {
    let mut points = Vec::new();
    for z in 0..=cells {
        for y in 0..=cells {
            for x in 0..=cells {
                let x = x as f64 / cells as f64;
                let y = y as f64 / cells as f64;
                let z = z as f64 / cells as f64;
                points.push(Point::new([x + 0.06 * x * (1. - x) * y, y, z]).unwrap());
            }
        }
    }
    let index = |x: usize, y: usize, z: usize| (z * (cells + 1) + y) * (cells + 1) + x;
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut tetrahedra = Vec::new();
    for z in 0..cells {
        for y in 0..cells {
            for x in 0..cells {
                for (permutation_id, permutation) in permutations.iter().enumerate() {
                    let mut site = [x, y, z];
                    let a = index(site[0], site[1], site[2]);
                    site[permutation[0]] += 1;
                    let b = index(site[0], site[1], site[2]);
                    site[permutation[1]] += 1;
                    let c = index(site[0], site[1], site[2]);
                    site[permutation[2]] += 1;
                    let d = index(site[0], site[1], site[2]);
                    tetrahedra.push(if matches!(permutation_id, 0 | 3 | 4) {
                        [a, b, c, d]
                    } else {
                        [a, c, b, d]
                    });
                }
            }
        }
    }
    TetrahedronMesh::new(points, tetrahedra).unwrap()
}

fn simplex_error<const D: usize>(mesh: SimplexMesh<f64, D>) -> f64 {
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|face| (face, 1)).collect())
            .unwrap();
    let boundaries =
        TaggedBoundaries::new(&mesh, &tags, 1, vec![dirichlet(1, |_| Ok(0.))]).unwrap();
    let geometry = simplex_geometry(&mesh).unwrap();
    let plan =
        SimplexFvmPlan::with_options(mesh, &ConstantTensor(0.125), &boundaries, limits()).unwrap();
    let rhs = plan.load(&field(manufactured_source::<D>), 0.).unwrap();
    let csr = plan.assemble_csr().unwrap();
    assert!(csr.nnz() <= 40 * plan.unknowns());
    solve_error(csr, rhs, &geometry.centers)
}

#[test]
fn manufactured_solution_refines_on_mapped_and_simplex_meshes() {
    let structured_2d = (structured_error::<2>(6), structured_error::<2>(12));
    let structured_3d = (structured_error::<3>(3), structured_error::<3>(5));
    let simplex_2d = (
        simplex_error(triangular_grid(4)),
        simplex_error(triangular_grid(8)),
    );
    let simplex_3d = (
        simplex_error(tetrahedral_grid(2)),
        simplex_error(tetrahedral_grid(3)),
    );
    for (name, (coarse, fine), max_ratio) in [
        ("structured 2D", structured_2d, 0.55),
        ("structured 3D", structured_3d, 0.70),
        ("simplex 2D", simplex_2d, 0.55),
        ("simplex 3D", simplex_3d, 0.70),
    ] {
        assert!(fine < coarse * max_ratio, "{name}: {coarse:e} -> {fine:e}");
    }
}
