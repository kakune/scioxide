use super::*;
use crate::fem::simplex::meshes;
use crate::*;
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CellMesh, CoordinateAxis, StructuredGrid,
};

struct Diffusion;
impl<const D: usize> StructuredFdmCoefficients<f64, f64, D> for Diffusion {
    fn diffusion(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(1.)
    }
    fn advection(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(0.)
    }
    fn reaction(&self, _: usize, _: usize, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(0.)
    }
}
impl<const D: usize> FvmCoefficients<f64, f64, D> for Diffusion {
    fn diffusion(&self, _: [f64; D]) -> Result<[[f64; D]; D], SciOxideError> {
        Ok(core::array::from_fn(|i| {
            core::array::from_fn(|j| if i == j { 1. } else { 0. })
        }))
    }
    fn reaction(&self, _: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(0.)
    }
}
impl<const D: usize> SimplexFemCoefficients<f64, f64, D> for Diffusion {
    fn evaluate(
        &self,
        _: [f64; D],
        diffusion: &mut [f64],
        reaction: &mut [f64],
    ) -> Result<(), SciOxideError> {
        diffusion.fill(0.);
        for i in 0..D {
            diffusion[i * D + i] = 1.;
        }
        reaction.fill(0.);
        Ok(())
    }
}
struct Reduced<'a> {
    operator: &'a dyn LinearOperator<f64>,
    free: &'a [usize],
}
impl LinearOperator<f64> for Reduced<'_> {
    fn rows(&self) -> usize {
        self.free.len()
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(2 * self.operator.rows() + self.operator.apply_workspace_len()?)
    }
    fn apply_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let n = self.operator.rows();
        let (full, tail) = scratch.split_at_mut(n);
        let (applied, work) = tail.split_at_mut(n);
        full.fill(0.);
        for (&i, &v) in self.free.iter().zip(x) {
            full[i] = v;
        }
        self.operator.apply_into(full, applied, work)?;
        for (out, &i) in out.iter_mut().zip(self.free) {
            *out = applied[i];
        }
        Ok(())
    }
}
fn solve<const D: usize>(
    operator: &dyn LinearOperator<f64>,
    matrix: &CsrMatrix<f64>,
    mass: &[f64],
    points: &[[f64; D]],
    free: &[usize],
) -> f64 {
    let reduced = Reduced { operator, free };
    let mut positions = vec![usize::MAX; operator.rows()];
    for (i, &k) in free.iter().enumerate() {
        positions[k] = i;
    }
    let mut coo = scioxide_sparse::SparseBuilder::new(free.len(), free.len());
    for (row, &i) in free.iter().enumerate() {
        for k in matrix.row_ptr()[i]..matrix.row_ptr()[i + 1] {
            let col = positions[matrix.col_indices()[k]];
            if col != usize::MAX {
                coo.push(row, col, matrix.values()[k]).unwrap();
            }
        }
    }
    let csr = coo
        .finish_csr(
            scioxide_sparse::DuplicatePolicy::Sum,
            scioxide_sparse::ExplicitZeroPolicy::Drop,
        )
        .unwrap();
    let mass: Vec<_> = free.iter().map(|&i| mass[i]).collect();
    let q: Vec<_> = free
        .iter()
        .map(|&i| {
            points[i]
                .iter()
                .map(|&x| (core::f64::consts::PI * x).sin())
                .product::<f64>()
        })
        .collect();
    let target: Vec<_> = q
        .iter()
        .map(|x| 0.8 * x)
        .chain(q.iter().map(|x| 0.5 * x))
        .collect();
    let laplace: Vec<_> = target
        .iter()
        .map(|x| D as f64 * core::f64::consts::PI.powi(2) * x)
        .collect();
    let mut provider = ReactionDiffusion::new(&reduced, &mass, &target, &laplace).unwrap();
    provider.assembled = Some(&csr);
    let fields = PdeFieldLayout::new(&[free.len(), free.len()]).unwrap();
    let pattern = block_pattern(&csr);
    let opts = options()
        .with_residual_tolerance(ResidualTolerance::new(1e-10, 0.).unwrap())
        .with_krylov(
            scioxide_core::Budget::try_from(300).unwrap(),
            scioxide_core::Budget::try_from(30).unwrap(),
        );
    let mut answers = Vec::new();
    for mode in [
        PdeJacobianMode::Sparse(pattern),
        PdeJacobianMode::Operator {
            operator_scratch: 0,
        },
    ] {
        let plan_fields = fields.clone();
        let plan_mode = mode;
        let mut work = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
            .workspace()
            .unwrap();
        let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
            .bind(&mut work)
            .unwrap();
        let result = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| {
                root_plan.solve(
                    &NewtonKrylovProblem::with_jacobian(&residual),
                    &vec![0.; target.len()],
                )
            })
            .unwrap();
        assert!(result.is_success(), "{:?}", result.termination());
        let mut independent = vec![0.; target.len()];
        context(|cx| provider.rhs(0., result.point(), &mut independent, cx)).unwrap();
        assert!(independent.iter().all(|r| r.abs() < 1e-9));
        answers.push(result.point().to_vec());
    }
    assert!(
        answers[0]
            .iter()
            .zip(&answers[1])
            .all(|(a, b)| (a - b).abs() < 1e-8)
    );
    answers[0]
        .iter()
        .zip(&target)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
fn grid<const D: usize>(n: usize) -> CellMesh<f64, D> {
    CellMesh::new(
        StructuredGrid::new(
            core::array::from_fn(|_| CoordinateAxis::uniform(0., 1., n + 1).unwrap()),
            [AxisTopology::Bounded; D],
        )
        .unwrap(),
    )
    .unwrap()
}
fn fdm<const D: usize>(n: usize) -> f64 {
    let mesh = grid::<D>(n);
    let mut conditions = Vec::new();
    for axis in 0..D {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(crate::BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                crate::BoundaryData::zero(),
            ));
        }
    }
    let boundaries = crate::StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFdmPlan::with_options(
        mesh,
        &Diffusion,
        &boundaries,
        crate::SpatialLimits::new(10000, 1000000, 100000000),
    )
    .unwrap();
    let points: Vec<_> = (0..crate::SpatialDiscretization::unknowns(&plan))
        .map(|i| plan.node_point(i).unwrap())
        .collect();
    let free: Vec<_> = points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.iter().all(|&x| x > 0. && x < 1.).then_some(i))
        .collect();
    solve(
        &plan,
        &plan.assemble_csr().unwrap(),
        plan.volumes(),
        &points,
        &free,
    )
}
fn fvm(n: usize) -> f64 {
    let mesh = grid::<2>(n);
    let mut conditions = Vec::new();
    for axis in 0..2 {
        for side in [AxisSide::Start, AxisSide::End] {
            conditions.push(crate::BoundaryCondition::dirichlet(
                BoundarySet::axis_side(&mesh, axis, side).unwrap(),
                crate::BoundaryData::zero(),
            ));
        }
    }
    let boundaries = crate::StructuredBoundaries::new(&mesh, 1, conditions).unwrap();
    let plan = StructuredFvmPlan::with_options(
        mesh,
        &Diffusion,
        &boundaries,
        crate::SpatialLimits::new(10000, 1000000, 100000000),
    )
    .unwrap();
    let points: Vec<_> = (0..n * n)
        .map(|i| {
            [
                (i % n) as f64 / n as f64 + 0.5 / n as f64,
                (i / n) as f64 / n as f64 + 0.5 / n as f64,
            ]
        })
        .collect();
    solve(
        &plan,
        &plan.assemble_csr().unwrap(),
        plan.volumes(),
        &points,
        &(0..n * n).collect::<Vec<_>>(),
    )
}
fn fem(n: usize) -> f64 {
    let mesh = meshes::triangle_grid::<f64>(n).unwrap();
    let tags = scioxide_spatial::SimplexBoundaryTags::new(
        &mesh,
        mesh.boundary_faces().map(|i| (i, 0)).collect(),
    )
    .unwrap();
    let b = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![
            crate::BoundaryCondition::dirichlet(0, crate::boundary::point_normal(|_, _| Ok(0.)))
                .for_component(0),
        ],
    )
    .unwrap();
    let plan = SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P1,
        &Diffusion,
        &b,
        SimplexFemOptions::new().with_limits(SimplexFemLimits::new(10000, 1000000, 128, 100000000)),
    )
    .unwrap();
    let points: Vec<_> = (0..plan.unknowns())
        .map(|i| plan.node_point(i).unwrap())
        .collect();
    let free: Vec<_> = crate::SpatialDiscretization::constrained(&plan)
        .iter()
        .enumerate()
        .filter_map(|(i, &fixed)| (!fixed).then_some(i))
        .collect();
    let mass = plan.assemble_mass_csr().unwrap();
    let mut lumped = vec![0.; plan.unknowns()];
    let mut work = lumped.clone();
    mass.apply_into(&vec![1.; plan.unknowns()], &mut lumped, &mut work)
        .unwrap();
    solve(
        &plan,
        &plan.assemble_csr().unwrap(),
        &lumped,
        &points,
        &free,
    )
}
#[test]
fn manufactured_refinement_fdm_fvm_and_lumped_simplex_fem() {
    for (name, run) in [
        ("fdm", fdm::<2> as fn(usize) -> f64),
        ("fvm", fvm),
        ("fem", fem),
    ] {
        let coarse = run(4);
        let fine = run(8);
        std::println!(
            "{name}: coarse={coarse:e}, fine={fine:e}, order={}",
            (coarse / fine).log2()
        );
        assert!(coarse / fine > 3.2);
        assert!(fine < 0.012);
    }
    let coarse = fdm::<3>(3);
    let fine = fdm::<3>(6);
    assert!(coarse / fine > 2.5);
    assert!(fine < 0.025);
}
