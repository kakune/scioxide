//! One boundary vocabulary and natural-data convention across FDM, FVM and FEM.
#![allow(clippy::unwrap_used)]
#[path = "../examples/support/simplex_fem.rs"]
mod meshes;

use scioxide_core::{Budget, ExecutionContext, ResidualTolerance, SciOxideError};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryPoint, DirichletConflictPolicy, EllipticOptions,
    EllipticProblem, FieldData, FieldPoint, FvmCoefficients, SimplexFemCoefficients,
    SimplexFemGeometry, SimplexFemLimits, SimplexFemOptions, SimplexFemOrder, SimplexFemPlan,
    SpatialDiscretization, SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients,
    StructuredFdmPlan, StructuredFvmPlan, TaggedBoundaries,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CellMesh, CoordinateAxis, QuadMesh, SimplexBoundaryTags,
    StructuredGrid,
};

/// Diagonal anisotropic diffusion `K = diag(1, 4)` with unit reaction.
const K: [f64; 2] = [1., 4.];
const ALPHA: f64 = 2.;

fn exact(x: [f64; 2]) -> f64 {
    (1.3 * x[0]).sin() * (0.7 * x[1]).cos() + x[0] * x[1]
}
fn gradient(x: [f64; 2]) -> [f64; 2] {
    [
        1.3 * (1.3 * x[0]).cos() * (0.7 * x[1]).cos() + x[1],
        -0.7 * (1.3 * x[0]).sin() * (0.7 * x[1]).sin() + x[0],
    ]
}
fn source(x: [f64; 2]) -> f64 {
    // -div(K grad u) + u for u = sin(1.3x)cos(0.7y) + xy.
    let s = (1.3 * x[0]).sin() * (0.7 * x[1]).cos();
    K[0] * 1.69 * s + K[1] * 0.49 * s + exact(x)
}
/// Robin data `n·K∇u + α u` of the exact solution, valid for every family.
fn robin_data() -> BoundaryData<f64, f64, 2> {
    BoundaryData::time_invariant(|p: BoundaryPoint<f64, 2>| {
        let g = gradient(p.point);
        Ok(p.normal[0] * K[0] * g[0] + p.normal[1] * K[1] * g[1] + ALPHA * exact(p.point))
    })
}

struct Anisotropic;
impl StructuredFdmCoefficients<f64, f64, 2> for Anisotropic {
    fn diffusion(&self, axis: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(K[axis])
    }
    fn reaction(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(1.)
    }
}
impl FvmCoefficients<f64, f64, 2> for Anisotropic {
    fn diffusion(&self, _: [f64; 2]) -> Result<[[f64; 2]; 2], SciOxideError> {
        Ok([[K[0], 0.], [0., K[1]]])
    }
    fn reaction(&self, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(1.)
    }
}
impl SimplexFemCoefficients<f64, f64, 2> for Anisotropic {
    fn evaluate(&self, _: [f64; 2], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
        k.copy_from_slice(&[K[0], 0., 0., K[1]]);
        c[0] = 1.;
        Ok(())
    }
}

fn square(n: usize) -> QuadMesh<f64> {
    let axis = || CoordinateAxis::uniform(0., 1., n + 1).unwrap();
    CellMesh::new(StructuredGrid::new([axis(), axis()], [AxisTopology::Bounded; 2]).unwrap())
        .unwrap()
}

fn sides(mesh: &QuadMesh<f64>) -> StructuredBoundaries<f64, f64, 2> {
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::robin(
                BoundarySet::axis_side(mesh, axis, side).unwrap(),
                ALPHA,
                robin_data(),
            ));
        }
    }
    StructuredBoundaries::new(mesh, 1, conditions).unwrap()
}

fn limits() -> SpatialLimits {
    SpatialLimits::new(1 << 20, 1 << 26, usize::MAX)
}

fn solve<P: SpatialDiscretization<f64, f64, 2>>(plan: &P) -> Vec<f64> {
    let source = FieldData::time_invariant(|p: FieldPoint<f64, 2>| Ok(source(p.point)));
    let (values, report) = EllipticProblem::new(plan, &source)
        .solve_with_options(&(EllipticOptions::new()))
        .unwrap()
        .into_parts();
    assert!(report.is_success());
    values
}

/// Lumped-mass weighted L² norm of `values - reference` over the unknowns.
fn l2<P: SpatialDiscretization<f64, f64, 2>>(
    plan: &P,
    values: &[f64],
    reference: impl Fn(usize) -> f64,
) -> f64 {
    plan.lumped_mass()
        .iter()
        .zip(values)
        .enumerate()
        .map(|(i, (m, u))| m * (u - reference(i)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn error<P: SpatialDiscretization<f64, f64, 2>>(plan: &P, values: &[f64]) -> f64 {
    l2(plan, values, |i| exact(plan.dof_point(i).unwrap().point))
}

/// L² errors of FDM, FVM and FEM, plus the L² distance between the FDM and FEM
/// nodal solutions.
fn errors(n: usize) -> ([f64; 3], f64) {
    let mesh = square(n);
    let boundaries = sides(&mesh);
    let fdm =
        StructuredFdmPlan::with_options(mesh.clone(), &Anisotropic, &boundaries, limits()).unwrap();
    let fvm = StructuredFvmPlan::with_options(mesh, &Anisotropic, &boundaries, limits()).unwrap();
    let triangles = meshes::triangle_grid::<f64>(n).unwrap();
    let tags = SimplexBoundaryTags::new(
        &triangles,
        triangles.boundary_faces().map(|face| (face, 0)).collect(),
    )
    .unwrap();
    let tagged = TaggedBoundaries::new(
        &triangles,
        &tags,
        1,
        vec![BoundaryCondition::robin(0, ALPHA, robin_data())],
    )
    .unwrap();
    let fem = SimplexFemPlan::with_options(
        triangles,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P1,
        &Anisotropic,
        &tagged,
        SimplexFemOptions::new().with_limits(SimplexFemLimits::new(
            1 << 20,
            1 << 26,
            64,
            usize::MAX,
        )),
    )
    .unwrap();
    let (u_fdm, u_fvm, u_fem) = (solve(&fdm), solve(&fvm), solve(&fem));
    let difference = l2(&fdm, &u_fdm, |i| u_fem[i]);
    (
        [
            error(&fdm, &u_fdm),
            error(&fvm, &u_fvm),
            error(&fem, &u_fem),
        ],
        difference,
    )
}

#[test]
fn anisotropic_robin_problem_converges_to_one_solution_in_every_family() {
    let (coarse, coarse_difference) = errors(8);
    let (fine, fine_difference) = errors(16);
    for (name, (c, f)) in ["FDM", "FVM", "FEM"].iter().zip(coarse.iter().zip(fine)) {
        assert!(f < 5e-4, "{name}: {f:e}");
        assert!(c / f > 3.6, "{name}: {c:e} -> {f:e}");
    }
    assert!(
        coarse_difference / fine_difference > 3.6,
        "{coarse_difference:e} -> {fine_difference:e}"
    );
}

#[test]
fn structured_fdm_dirichlet_diffusion_is_symmetric_and_solves_with_cg() {
    let mesh = square(12);
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                BoundaryData::time_invariant(|p: BoundaryPoint<f64, 2>| Ok(1. + p.point[0])),
            ));
        }
    }
    struct Pure;
    impl StructuredFdmCoefficients<f64, f64, 2> for Pure {
        fn diffusion(&self, axis: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(K[axis])
        }
    }
    let boundaries = StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFdmPlan::with_options(mesh, &Pure, &boundaries, limits()).unwrap();
    let csr = plan.assemble_csr().unwrap();
    for (row, column, value) in csr.iter() {
        assert_eq!(csr.get(column, row).unwrap(), value, "({row}, {column})");
    }
    let cg = scioxide_linalg::CgSolver::new(
        ResidualTolerance::new(1e-12, 1e-12).unwrap(),
        Budget::try_from(1000).unwrap(),
    );
    let elliptic = EllipticOptions::new();
    let mut workspace = scioxide_pde::EllipticWorkspace::new().with_linear_solver(Box::new(cg));
    let mut values = vec![0.0; plan.unknowns()];
    let report = EllipticProblem::new(&plan, &FieldData::zero())
        .solve_into(
            &elliptic,
            &mut values,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success(), "{:?}", report.termination());
    assert!(report.diagnostics().num_linear_iterations() > 0);
    // The affine harmonic field is reproduced exactly.
    for (i, &u) in values.iter().enumerate() {
        let x = plan.node_point(i).unwrap();
        assert!((u - 1. - x[0]).abs() < 1e-10);
    }
}

/// Squared wave speed whose assembled entries overflow in every family.
struct Overflowing;
impl StructuredFdmCoefficients<f64, f64, 1> for Overflowing {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(f64::MAX)
    }
}
impl FvmCoefficients<f64, f64, 1> for Overflowing {
    fn diffusion(&self, _: [f64; 1]) -> Result<[[f64; 1]; 1], SciOxideError> {
        Ok([[f64::MAX]])
    }
}
impl SimplexFemCoefficients<f64, f64, 1> for Overflowing {
    fn evaluate(&self, _: [f64; 1], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
        k[0] = f64::MAX;
        c[0] = 0.;
        Ok(())
    }
}

#[test]
fn wave_speed_overflow_has_one_classification() {
    let segment = scioxide_spatial::SegmentMesh::new(
        StructuredGrid::new(
            [CoordinateAxis::uniform(0., 1., 9).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    let side = |s| BoundarySet::axis_side(&segment, 0, s).unwrap();
    let structured = StructuredBoundaries::new(
        &segment,
        1,
        vec![
            BoundaryCondition::dirichlet(side(AxisSide::Start), BoundaryData::zero()),
            BoundaryCondition::natural(side(AxisSide::End), BoundaryData::zero()),
        ],
    )
    .unwrap();
    let intervals = scioxide_spatial::IntervalMesh::new(
        (0..9)
            .map(|i| scioxide_spatial::Point::new([f64::from(i) / 8.]).unwrap())
            .collect(),
        (0..8).map(|i| [i, i + 1]).collect(),
    )
    .unwrap();
    let tags = SimplexBoundaryTags::new(
        &intervals,
        intervals
            .boundary_faces()
            .map(|face| {
                (
                    face,
                    u32::from(intervals.face_vertices(face).unwrap()[0] == 8),
                )
            })
            .collect(),
    )
    .unwrap();
    let tagged = TaggedBoundaries::new(
        &intervals,
        &tags,
        1,
        vec![
            BoundaryCondition::dirichlet(0, BoundaryData::zero()),
            BoundaryCondition::natural(1, BoundaryData::zero()),
        ],
    )
    .unwrap();
    let overflow = |error: SciOxideError| {
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        )
    };
    let fdm = StructuredFdmPlan::with_options(segment.clone(), &Overflowing, &structured, limits());
    assert!(overflow(fdm.err().unwrap()));
    let fvm = StructuredFvmPlan::with_options(segment, &Overflowing, &structured, limits());
    assert!(overflow(fvm.err().unwrap()));
    let fem = SimplexFemPlan::with_options(
        intervals,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P1,
        &Overflowing,
        &tagged,
        SimplexFemOptions::new().with_limits(SimplexFemLimits::new(100, 1000, 16, usize::MAX)),
    );
    assert!(overflow(fem.err().unwrap()));
}

#[test]
fn one_validation_path_rejects_invalid_lists_for_every_region_family() {
    let mesh = square(2);
    let side = |axis, side| BoundarySet::axis_side(&mesh, axis, side).unwrap();
    let all = |count: usize| -> Vec<BoundaryCondition<f64, f64, 2, BoundarySet<2>>> {
        (0..2)
            .flat_map(|axis| [AxisSide::Start, AxisSide::End].map(move |s| (axis, s)))
            .take(count)
            .map(|(axis, s)| BoundaryCondition::natural(side(axis, s), BoundaryData::zero()))
            .collect()
    };
    let invalid = |result: Result<StructuredBoundaries<f64, f64, 2>, SciOxideError>| {
        assert!(result.is_err());
    };
    // Missing, duplicate, zero-component, out-of-range component and non-finite Robin.
    invalid(StructuredBoundaries::new(&mesh, 1, all(3)));
    let mut duplicate = all(4);
    duplicate.push(BoundaryCondition::dirichlet(
        side(0, AxisSide::Start),
        BoundaryData::zero(),
    ));
    invalid(StructuredBoundaries::new(&mesh, 1, duplicate));
    invalid(StructuredBoundaries::new(&mesh, 0, all(4)));
    let mut component = all(4);
    component[0] =
        BoundaryCondition::natural(side(0, AxisSide::Start), BoundaryData::zero()).for_component(1);
    invalid(StructuredBoundaries::new(&mesh, 1, component));
    let mut robin = all(4);
    robin[0] = BoundaryCondition::robin(side(0, AxisSide::Start), f64::NAN, BoundaryData::zero());
    invalid(StructuredBoundaries::new(&mesh, 1, robin));
    // A foreign mesh revision and a periodic seam face.
    let other = square(2);
    invalid(StructuredBoundaries::new(&other, 1, all(4)));
    let periodic = CellMesh::new(
        StructuredGrid::new(
            [
                CoordinateAxis::uniform(0., 1., 3).unwrap(),
                CoordinateAxis::uniform(0., 1., 3).unwrap(),
            ],
            [AxisTopology::Periodic, AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    let seam = BoundarySet::axis_side(&periodic, 0, AxisSide::Start).unwrap();
    invalid(StructuredBoundaries::new(
        &periodic,
        1,
        vec![BoundaryCondition::natural(seam, BoundaryData::zero())],
    ));
    // Per-component conditions combine across components.
    let mut split = all(4);
    split[0] =
        BoundaryCondition::natural(side(0, AxisSide::Start), BoundaryData::zero()).for_component(0);
    assert!(StructuredBoundaries::new(&mesh, 2, split.clone()).is_err());
    split.push(
        BoundaryCondition::dirichlet(side(0, AxisSide::Start), BoundaryData::zero())
            .for_component(1),
    );
    assert!(StructuredBoundaries::new(&mesh, 2, split).is_ok());
    let mut split = all(4)
        .into_iter()
        .map(|c| c.for_component(0))
        .collect::<Vec<_>>();
    split.extend(all(4).into_iter().map(|c| c.for_component(1)));
    assert!(StructuredBoundaries::new(&mesh, 2, split).is_ok());
    // Tagged faces: unused tags and untagged exterior faces are rejected.
    let triangles = meshes::triangle_grid::<f64>(1).unwrap();
    let tags = SimplexBoundaryTags::new(
        &triangles,
        triangles.boundary_faces().map(|face| (face, 3)).collect(),
    )
    .unwrap();
    let zero = || BoundaryData::<f64, f64, 2>::zero();
    assert!(
        TaggedBoundaries::new(
            &triangles,
            &tags,
            1,
            vec![
                BoundaryCondition::natural(3, zero()),
                BoundaryCondition::natural(4, zero())
            ]
        )
        .is_err()
    );
    let partial = SimplexBoundaryTags::new(
        &triangles,
        triangles
            .boundary_faces()
            .take(1)
            .map(|face| (face, 3))
            .collect(),
    )
    .unwrap();
    assert!(
        TaggedBoundaries::new(
            &triangles,
            &partial,
            1,
            vec![BoundaryCondition::natural(3, zero())]
        )
        .is_err()
    );
    let bound = TaggedBoundaries::new(
        &triangles,
        &tags,
        1,
        vec![BoundaryCondition::natural(3, zero())],
    )
    .unwrap()
    .with_dirichlet_conflict_policy(DirichletConflictPolicy::LastConditionWins);
    assert_eq!(
        bound.conflict_policy(),
        DirichletConflictPolicy::LastConditionWins
    );
}

#[test]
fn structured_fdm_corner_follows_the_shared_conflict_policy() {
    struct Unit;
    impl StructuredFdmCoefficients<f64, f64, 2> for Unit {
        fn diffusion(&self, _: usize, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
            Ok(1.)
        }
    }
    let mesh = square(2);
    let side = |axis, s| BoundarySet::axis_side(&mesh, axis, s).unwrap();
    // The x = 0 and y = 0 sides meet at vertex 0 with different values.
    let conditions = |first: f64, second: f64| {
        vec![
            BoundaryCondition::dirichlet(side(0, AxisSide::Start), BoundaryData::constant(first)),
            BoundaryCondition::dirichlet(side(1, AxisSide::Start), BoundaryData::constant(second)),
            BoundaryCondition::natural(side(0, AxisSide::End), BoundaryData::zero()),
            BoundaryCondition::natural(side(1, AxisSide::End), BoundaryData::zero()),
        ]
    };
    let corner = |first, second, policy| {
        let boundaries = StructuredBoundaries::new(&mesh, 1, conditions(first, second))
            .unwrap()
            .with_dirichlet_conflict_policy(policy);
        StructuredFdmPlan::with_options(mesh.clone(), &Unit, &boundaries, limits())
            .and_then(|plan| plan.load(&FieldData::zero(), 0.))
            .map(|load| load[0])
    };
    assert!(corner(0., 1., DirichletConflictPolicy::Reject).is_err());
    assert_eq!(corner(1., 1., DirichletConflictPolicy::Reject).unwrap(), 1.);
    assert_eq!(
        corner(0., 1., DirichletConflictPolicy::LastConditionWins).unwrap(),
        1.
    );
    assert_eq!(
        corner(1., 0., DirichletConflictPolicy::LastConditionWins).unwrap(),
        0.
    );
}
