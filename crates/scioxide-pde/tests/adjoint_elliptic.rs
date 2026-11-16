//! Steady adjoints of the crate's own FDM, FVM and FEM plans against central
//! finite differences of the crate's own solves.
#![allow(clippy::unwrap_used)]
#[path = "support/fd.rs"]
mod fd;
#[path = "../examples/support/simplex_fem.rs"]
mod meshes;

use scioxide_ad::{Dual, Real};
use scioxide_core::{
    ExecutionContext, ResidualTolerance, SciOxideError, SciOxideErrorKind, SolveContext,
};
use scioxide_linalg::VectorViewMut;
use scioxide_ode::{AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryPoint, DifferentiableDiscretization, EllipticOptions,
    EllipticProblem, FieldData, FieldPoint, FvmCoefficients, PdeConditionTangent,
    PdeEllipticAdjointPlan, PdeParameterTangent, PdeParameterization, PdeSteadyAdjointOptions,
    SimplexFemCoefficients, SimplexFemGeometry, SimplexFemLimits, SimplexFemOptions,
    SimplexFemOrder, SimplexFemPlan, SimplexFvmPlan, SpatialDiscretization, SpatialLimits,
    StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan, StructuredFvmPlan,
    TaggedBoundaries,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CellMesh, CoordinateAxis, Grid1d, SegmentMesh,
    SimplexBoundaryTags, SimplexMesh, StructuredGrid, TriangleMesh,
};

/// Parameters: diffusion scale κ, source amplitude s, Dirichlet amplitude g,
/// Robin coefficient α and Robin data amplitude r.
const P0: [f64; 5] = [1.3, 0.8, 0.4, 2.0, 0.6];
const NAMES: [&str; 5] = ["kappa", "source", "dirichlet", "robin", "robin data"];

/// `k(x) = κ (1 + x₀/2)` (FDM, FEM) or `K = κ [[1 + x₀/2, 0.2], [0.2, 1]]` (FVM).
/// With `κ = 1` it is the exact coefficient tangent `∂k/∂κ`.
struct Diffusion(f64);
impl<const D: usize> StructuredFdmCoefficients<f64, f64, D> for Diffusion {
    fn diffusion(&self, _: usize, _: usize, x: [f64; D]) -> Result<f64, SciOxideError> {
        Ok(self.0 * (1.0 + 0.5 * x[0]))
    }
}
impl FvmCoefficients<f64, f64, 2> for Diffusion {
    fn diffusion(&self, x: [f64; 2]) -> Result<[[f64; 2]; 2], SciOxideError> {
        Ok([
            [self.0 * (1.0 + 0.5 * x[0]), 0.2 * self.0],
            [0.2 * self.0, self.0],
        ])
    }
}
impl SimplexFemCoefficients<f64, f64, 2> for Diffusion {
    fn evaluate(&self, x: [f64; 2], k: &mut [f64], c: &mut [f64]) -> Result<(), SciOxideError> {
        let value = self.0 * (1.0 + 0.5 * x[0]);
        k.copy_from_slice(&[value, 0.0, 0.0, value]);
        c[0] = 0.0;
        Ok(())
    }
}

fn source_data<const D: usize>(p: &[f64]) -> FieldData<f64, f64, D> {
    let s = p[1];
    FieldData::time_invariant(move |q: FieldPoint<f64, D>| Ok(s * shape_f(q.point)))
}
fn shape_f<const D: usize>(x: [f64; D]) -> f64 {
    1.0 + x[0] * x[D - 1]
}
fn dirichlet_shape<const D: usize>(x: [f64; D]) -> f64 {
    1.0 + x[D - 1]
}
fn robin_shape<const D: usize>(x: [f64; D]) -> f64 {
    1.0 + x[0] * x[D - 1] + x[0]
}
fn dirichlet<const D: usize>(amplitude: f64) -> BoundaryData<f64, f64, D> {
    BoundaryData::time_invariant(move |q: BoundaryPoint<f64, D>| {
        Ok(amplitude * dirichlet_shape(q.point))
    })
}
fn robin<const D: usize>(amplitude: f64) -> BoundaryData<f64, f64, D> {
    BoundaryData::time_invariant(move |q: BoundaryPoint<f64, D>| {
        Ok(amplitude * robin_shape(q.point))
    })
}

/// Derivative data of every parameter for `conditions` conditions whose first is
/// the Dirichlet condition and the rest Robin.
fn tangents<'a, const D: usize, C: ?Sized>(
    unit: &'a C,
    conditions: usize,
) -> Vec<PdeParameterTangent<'a, f64, f64, D, C>> {
    let list = |first: PdeConditionTangent<f64, f64, D>,
                rest: &dyn Fn() -> PdeConditionTangent<f64, f64, D>| {
        let mut list = vec![first];
        list.extend((1..conditions).map(|_| rest()));
        list
    };
    vec![
        PdeParameterTangent::new().with_coefficients(unit),
        PdeParameterTangent::new().with_source(source_data(&[0.0, 1.0])),
        PdeParameterTangent::new().with_boundaries(list(
            PdeConditionTangent::new(dirichlet(1.0)),
            &PdeConditionTangent::zero,
        )),
        PdeParameterTangent::new().with_boundaries(list(PdeConditionTangent::zero(), &|| {
            PdeConditionTangent::zero().with_robin_coefficient(1.0)
        })),
        PdeParameterTangent::new().with_boundaries(list(PdeConditionTangent::zero(), &|| {
            PdeConditionTangent::new(robin(1.0))
        })),
    ]
}

/// `J(u, p) = ½ Σ m_i (u_i − 0.3)² + 0.05 κ²` with lumped-mass weights.
struct Objective(Vec<f64>);
impl Objective {
    fn value(&self, u: &[f64], p: &[f64]) -> f64 {
        0.5 * self
            .0
            .iter()
            .zip(u)
            .map(|(m, u)| m * (u - 0.3).powi(2))
            .sum::<f64>()
            + 0.05 * p[0] * p[0]
    }
}
impl AdjointObjective<f64> for Objective {
    fn evaluate(
        &self,
        _: AdjointObjectivePart,
        _: f64,
        u: &[f64],
        p: &[f64],
        gu: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        for ((g, m), u) in gu.iter_mut().zip(&self.0).zip(u) {
            *g = m * (u - 0.3);
        }
        gp.fill(0.0);
        gp[0] = 0.1 * p[0];
        Ok(AdjointObjectiveValue {
            value: self.value(u, p),
            time_derivative: 0.0,
        })
    }
}

fn options() -> PdeSteadyAdjointOptions<f64> {
    let tolerance = ResidualTolerance::new(1e-11, 1e-11).unwrap();
    PdeSteadyAdjointOptions::new()
        .with_primal_tolerance(tolerance)
        .with_linear_tolerance(tolerance)
}

fn solve<P: SpatialDiscretization<f64, f64, D>, const D: usize>(
    plan: &P,
    source: &FieldData<f64, f64, D>,
) -> Vec<f64> {
    let solution = EllipticProblem::new(plan, source)
        .solve_with_options(&(EllipticOptions::new()))
        .unwrap();
    assert!(solution.report().is_success());
    solution.into_parts().0
}

/// Residual `A u − b` of the plan prepared at `p`.
fn residual<P: SpatialDiscretization<f64, f64, D>, const D: usize>(
    plan: &P,
    source: &FieldData<f64, f64, D>,
    u: &[f64],
) -> Vec<f64> {
    let mut applied = vec![0.0; u.len()];
    let mut scratch = vec![0.0; plan.apply_workspace_len().unwrap()];
    plan.apply_into(u, &mut applied, &mut scratch).unwrap();
    let load = plan.load(source, 0.0).unwrap();
    applied.iter().zip(load).map(|(a, b)| a - b).collect()
}

/// Every parameter's gradient and residual tangent against central differences.
fn check<P, const D: usize>(
    name: &str,
    build: impl Fn(&[f64]) -> P,
    parameterize: impl Fn(&P) -> PdeParameterization<f64, f64, D>,
) where
    P: DifferentiableDiscretization<f64, f64, D>,
{
    let plan = build(&P0);
    let source = source_data::<D>(&P0);
    let problem = EllipticProblem::new(&plan, &source);
    let solution = solve(&plan, &source);
    let objective = Objective(plan.lumped_mass().to_vec());
    let parameterization = parameterize(&plan);
    assert_eq!(parameterization.values(), &P0);
    let adjoint =
        PdeEllipticAdjointPlan::new(&problem, &parameterization, &objective, options()).unwrap();
    let result = adjoint
        .solve(&problem, &parameterization, &solution, &objective)
        .unwrap();
    assert!(result.report().derivative_residual() <= 1e-9);
    let n = solution.len();
    // A state off the equilibrium: the tangent must not depend on it being one.
    let state: Vec<f64> = (0..n)
        .map(|i| solution[i] + 0.01 * (i as f64).sin())
        .collect();
    let mut tangent = vec![0.0; n];
    let mut scratch = vec![0.0; parameterization.scratch_len()];
    for j in 0..P0.len() {
        let perturbed = |x: f64| {
            let mut p = P0;
            p[j] = x;
            p
        };
        let (estimate, tolerance) = fd::central(
            |x| {
                let p = perturbed(x);
                let plan = build(&p);
                objective.value(&solve(&plan, &source_data::<D>(&p)), &p)
            },
            P0[j],
            1e-13,
        );
        let gradient = result.parameter_gradient()[j];
        assert!(
            (gradient - estimate).abs() <= tolerance.max(1e-10 * estimate.abs()),
            "{name} {}: adjoint {gradient:e}, finite difference {estimate:e} ± {tolerance:e}",
            NAMES[j]
        );
        parameterization
            .residual_tangent_into(j, &state, 0.0, &mut tangent, &mut scratch)
            .unwrap();
        let differences = fd::central_each(
            |x| {
                let p = perturbed(x);
                residual(&build(&p), &source_data::<D>(&p), &state)
            },
            P0[j],
            1e-14,
        );
        for (i, (&actual, (estimate, tolerance))) in tangent.iter().zip(differences).enumerate() {
            assert!(
                (actual - estimate).abs() <= tolerance + 1e-9,
                "{name} {} row {i}: tangent {actual:e}, finite difference {estimate:e}",
                NAMES[j]
            );
        }
    }
    // Forward sensitivity of one direction against the finite-difference solution.
    let direction = [0.3, -0.2, 0.5, 0.1, 0.7];
    let mut du = vec![0.0; n];
    let mut workspace = adjoint.workspace().unwrap();
    adjoint
        .sensitivity_into(
            &problem,
            &parameterization,
            &solution,
            &direction,
            &mut VectorViewMut::from_slice(&mut du),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let step = 1e-6;
    let shifted = |sign: f64| {
        let p: Vec<f64> = P0
            .iter()
            .zip(direction)
            .map(|(p, d)| p + sign * step * d)
            .collect();
        solve(&build(&p), &source_data::<D>(&p))
    };
    let (plus, minus) = (shifted(1.0), shifted(-1.0));
    for i in 0..n {
        let estimate = (plus[i] - minus[i]) / (2.0 * step);
        assert!(
            (du[i] - estimate).abs() <= 1e-6 * (1.0 + estimate.abs()),
            "{name} sensitivity {i}: {:e} vs {estimate:e}",
            du[i]
        );
    }
}

fn segment(n: usize) -> SegmentMesh<f64> {
    SegmentMesh::new(
        Grid1d::new(
            [CoordinateAxis::uniform(0.0, 1.0, n).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap()
}

fn square(n: usize) -> scioxide_spatial::QuadMesh<f64> {
    let axis = || CoordinateAxis::uniform(0.0, 1.0, n + 1).unwrap();
    CellMesh::new(StructuredGrid::new([axis(), axis()], [AxisTopology::Bounded; 2]).unwrap())
        .unwrap()
}

/// Dirichlet on the start of axis zero, Robin on every other side.
fn sides<const D: usize>(mesh: &CellMesh<f64, D>, p: &[f64]) -> StructuredBoundaries<f64, f64, D> {
    let mut conditions = vec![BoundaryCondition::dirichlet(
        BoundarySet::axis_side(mesh, 0, AxisSide::Start).unwrap(),
        dirichlet(p[2]),
    )];
    for axis in 0..D {
        for side in [AxisSide::Start, AxisSide::End] {
            if axis == 0 && side == AxisSide::Start {
                continue;
            }
            conditions.push(BoundaryCondition::robin(
                BoundarySet::axis_side(mesh, axis, side).unwrap(),
                p[3],
                robin(p[4]),
            ));
        }
    }
    StructuredBoundaries::new(mesh, 1, conditions).unwrap()
}

fn limits() -> SpatialLimits {
    SpatialLimits::new(1 << 20, 1 << 24, usize::MAX)
}

/// Tag faces on `x = 0` with 0 and every other exterior face with 1.
fn tagged(mesh: &TriangleMesh<f64>, p: &[f64]) -> TaggedBoundaries<f64, f64, 2> {
    let tags = SimplexBoundaryTags::new(
        mesh,
        mesh.boundary_faces()
            .map(|face| {
                let left = mesh
                    .face_vertices(face)
                    .unwrap()
                    .iter()
                    .all(|&v| mesh.points()[v].as_array()[0] == 0.0);
                (face, u32::from(!left))
            })
            .collect(),
    )
    .unwrap();
    TaggedBoundaries::new(
        mesh,
        &tags,
        1,
        vec![
            BoundaryCondition::dirichlet(0, dirichlet(p[2])),
            BoundaryCondition::robin(1, p[3], robin(p[4])),
        ],
    )
    .unwrap()
}

fn fem_options() -> SimplexFemOptions<f64> {
    SimplexFemOptions::new().with_limits(SimplexFemLimits::new(1 << 16, 1 << 22, 64, usize::MAX))
}

#[test]
fn structured_fdm_1d_gradients_match_finite_differences() {
    let unit = Diffusion(1.0);
    check(
        "FDM 1D",
        |p| {
            let mesh = segment(17);
            let boundaries = sides(&mesh, p);
            StructuredFdmPlan::with_options(mesh, &Diffusion(p[0]), &boundaries, limits()).unwrap()
        },
        |plan| {
            plan.parameterize(
                &P0,
                &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
            )
            .unwrap()
        },
    );
}

#[test]
fn structured_fdm_2d_gradients_match_finite_differences() {
    let unit = Diffusion(1.0);
    check(
        "FDM 2D",
        |p| {
            let mesh = square(6);
            let boundaries = sides(&mesh, p);
            StructuredFdmPlan::with_options(mesh, &Diffusion(p[0]), &boundaries, limits()).unwrap()
        },
        |plan| {
            plan.parameterize(
                &P0,
                &tangents::<2, dyn StructuredFdmCoefficients<f64, f64, 2>>(&unit, 4),
            )
            .unwrap()
        },
    );
}

#[test]
fn structured_fvm_gradients_match_finite_differences() {
    let unit = Diffusion(1.0);
    check(
        "structured FVM",
        |p| {
            let mesh = square(6);
            let boundaries = sides(&mesh, p);
            StructuredFvmPlan::with_options(mesh, &Diffusion(p[0]), &boundaries, limits()).unwrap()
        },
        |plan| {
            plan.parameterize(
                &P0,
                &tangents::<2, dyn FvmCoefficients<f64, f64, 2>>(&unit, 4),
            )
            .unwrap()
        },
    );
}

#[test]
fn simplex_fvm_gradients_match_finite_differences() {
    let unit = Diffusion(1.0);
    check(
        "simplex FVM",
        |p| {
            let mesh = meshes::triangle_grid::<f64>(5).unwrap();
            let boundaries = tagged(&mesh, p);
            SimplexFvmPlan::with_options(mesh, &Diffusion(p[0]), &boundaries, limits()).unwrap()
        },
        |plan| {
            plan.parameterize(
                &P0,
                &tangents::<2, dyn FvmCoefficients<f64, f64, 2>>(&unit, 2),
            )
            .unwrap()
        },
    );
}

#[test]
fn simplex_fem_gradients_match_finite_differences() {
    let unit = Diffusion(1.0);
    for order in [SimplexFemOrder::P1, SimplexFemOrder::P2] {
        check(
            &format!("FEM {order:?}"),
            |p| {
                let mesh = meshes::triangle_grid::<f64>(4).unwrap();
                let boundaries = tagged(&mesh, p);
                SimplexFemPlan::with_options(
                    mesh,
                    SimplexFemGeometry::Affine,
                    order,
                    &Diffusion(p[0]),
                    &boundaries,
                    fem_options(),
                )
                .unwrap()
            },
            |plan| {
                plan.parameterize(
                    &P0,
                    &tangents::<2, dyn SimplexFemCoefficients<f64, f64, 2>>(&unit, 2),
                )
                .unwrap()
            },
        );
    }
}

#[test]
fn curved_p2_fem_gradients_match_finite_differences() {
    let unit = Diffusion(1.0);
    let base: SimplexMesh<f64, 2> = meshes::triangle_grid::<f64>(3).unwrap();
    let (mesh, edges) = meshes::curved(&base).unwrap();
    check(
        "curved FEM P2",
        |p| {
            let boundaries = tagged(&mesh, p);
            SimplexFemPlan::with_options(
                mesh.clone(),
                SimplexFemGeometry::Quadratic(edges.view()),
                SimplexFemOrder::P2,
                &Diffusion(p[0]),
                &boundaries,
                fem_options(),
            )
            .unwrap()
        },
        |plan| {
            plan.parameterize(
                &P0,
                &tangents::<2, dyn SimplexFemCoefficients<f64, f64, 2>>(&unit, 2),
            )
            .unwrap()
        },
    );
}

/// `k(x; κ) = exp(κ x)`, whose tangent a caller evaluates with forward-mode AD.
struct Exponential(f64);
impl StructuredFdmCoefficients<f64, f64, 1> for Exponential {
    fn diffusion(&self, _: usize, _: usize, x: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok((self.0 * x[0]).exp())
    }
}
struct ExponentialTangent(f64);
impl StructuredFdmCoefficients<f64, f64, 1> for ExponentialTangent {
    fn diffusion(&self, _: usize, _: usize, x: [f64; 1]) -> Result<f64, SciOxideError> {
        let k = (Dual::variable(self.0) * x[0]).exp();
        Ok(k.derivative())
    }
}

#[test]
fn ad_coefficient_tangents_give_finite_difference_gradients() {
    let kappa = 0.7;
    let build = |kappa: f64| {
        let mesh = segment(21);
        let boundaries = sides(&mesh, &P0);
        StructuredFdmPlan::with_options(mesh, &Exponential(kappa), &boundaries, limits()).unwrap()
    };
    let plan = build(kappa);
    let source = source_data::<1>(&P0);
    let problem = EllipticProblem::new(&plan, &source);
    let objective = Objective(plan.lumped_mass().to_vec());
    let tangent = ExponentialTangent(kappa);
    let parameterization = plan
        .parameterize(
            &[kappa],
            &[PdeParameterTangent::<f64, f64, 1, dyn StructuredFdmCoefficients<f64, f64, 1>>::new()
                .with_coefficients(&tangent)],
        )
        .unwrap();
    let adjoint =
        PdeEllipticAdjointPlan::new(&problem, &parameterization, &objective, options()).unwrap();
    let gradient = adjoint
        .solve(
            &problem,
            &parameterization,
            &solve(&plan, &source),
            &objective,
        )
        .unwrap()
        .parameter_gradient()[0];
    // The objective's κ term reads `parameters[0]`, which is κ here as well.
    let (estimate, tolerance) = fd::central(
        |x| objective.value(&solve(&build(x), &source), &[x]),
        kappa,
        1e-13,
    );
    assert!(
        (gradient - estimate).abs() <= tolerance,
        "{gradient:e} vs {estimate:e}"
    );
}

fn fdm_problem() -> (StructuredFdmPlan<f64, f64, 1>, FieldData<f64, f64, 1>) {
    let mesh = segment(9);
    let boundaries = sides(&mesh, &P0);
    (
        StructuredFdmPlan::with_options(mesh, &Diffusion(P0[0]), &boundaries, limits()).unwrap(),
        source_data::<1>(&P0),
    )
}

#[test]
fn invalid_tangents_are_rejected() {
    let (plan, _) = fdm_problem();
    let unit = Diffusion(1.0);
    type Tangent<'a> =
        PdeParameterTangent<'a, f64, f64, 1, dyn StructuredFdmCoefficients<f64, f64, 1>>;
    // Count mismatch between values and tangents.
    let error = plan
        .parameterize(&[1.0, 2.0], &[Tangent::new()])
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    // Boundary tangents must list every condition.
    let error = plan
        .parameterize(
            &[1.0],
            &[Tangent::new().with_boundaries(vec![PdeConditionTangent::zero()])],
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    // A Robin derivative on the Dirichlet condition.
    let error = plan
        .parameterize(
            &[1.0],
            &[Tangent::new().with_boundaries(vec![
                PdeConditionTangent::zero().with_robin_coefficient(1.0),
                PdeConditionTangent::zero(),
            ])],
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    // Non-finite parameter values.
    let error = plan
        .parameterize(&[f64::NAN], &[Tangent::new().with_coefficients(&unit)])
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    // Signed coefficient tangents are accepted (primal plans reject negative diffusion).
    let negative = Diffusion(-1.0);
    assert!(
        plan.parameterize(&[1.0], &[Tangent::new().with_coefficients(&negative)])
            .is_ok()
    );
    // No parameters is a valid, empty parameterization.
    assert!(plan.parameterize(&[], &[]).unwrap().is_empty());
}

#[test]
fn adjoint_plan_rejects_incompatible_problems_and_preserves_outputs() {
    let (plan, source) = fdm_problem();
    let unit = Diffusion(1.0);
    let problem = EllipticProblem::new(&plan, &source);
    let solution = solve(&plan, &source);
    let objective = Objective(plan.lumped_mass().to_vec());
    let parameterization = plan
        .parameterize(
            &P0,
            &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
        )
        .unwrap();
    let adjoint =
        PdeEllipticAdjointPlan::new(&problem, &parameterization, &objective, options()).unwrap();
    let mut workspace = adjoint.workspace().unwrap();
    let (mut lambda, mut gradient) = (vec![7.0; solution.len()], vec![7.0; P0.len()]);
    let mut run = |solution: &[f64],
                   parameterization: &PdeParameterization<f64, f64, 1>,
                   lambda: &mut [f64],
                   gradient: &mut [f64]| {
        adjoint.solve_into(
            &problem,
            parameterization,
            solution,
            &objective,
            &mut VectorViewMut::from_slice(lambda),
            &mut VectorViewMut::from_slice(gradient),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
    };
    // A state that is not an equilibrium fails the primal check transactionally.
    let mut wrong = solution.clone();
    wrong[3] += 1e-3;
    assert_eq!(
        run(&wrong, &parameterization, &mut lambda, &mut gradient)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    // A wrong solution length.
    assert!(
        run(
            &solution[1..],
            &parameterization,
            &mut lambda,
            &mut gradient
        )
        .is_err()
    );
    // A parameterization of a different plan size.
    let mesh = segment(11);
    let boundaries = sides(&mesh, &P0);
    let other =
        StructuredFdmPlan::with_options(mesh, &Diffusion(1.0), &boundaries, limits()).unwrap();
    let foreign = other.parameterize(&[], &[]).unwrap();
    assert!(run(&solution, &foreign, &mut lambda, &mut gradient).is_err());
    assert!(lambda.iter().chain(&gradient).all(|&v| v == 7.0));
    run(&solution, &parameterization, &mut lambda, &mut gradient).unwrap();
    assert!(gradient.iter().all(|&v| v != 7.0));
    // Byte limits cover the stored operator and factor.
    let tight = options().with_max_workspace_bytes(64);
    assert!(PdeEllipticAdjointPlan::new(&problem, &parameterization, &objective, tight).is_err());
}

#[test]
fn constant_mode_plans_are_rejected() {
    let mesh = segment(9);
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::natural(
                BoundarySet::axis_side(&mesh, 0, AxisSide::Start).unwrap(),
                BoundaryData::zero(),
            ),
            BoundaryCondition::natural(
                BoundarySet::axis_side(&mesh, 0, AxisSide::End).unwrap(),
                BoundaryData::zero(),
            ),
        ],
    )
    .unwrap();
    let plan =
        StructuredFdmPlan::with_options(mesh, &Diffusion(1.0), &boundaries, limits()).unwrap();
    let source = FieldData::zero();
    let parameterization = plan.parameterize(&[], &[]).unwrap();
    let objective = Objective(plan.lumped_mass().to_vec());
    let error = PdeEllipticAdjointPlan::new(
        &EllipticProblem::new(&plan, &source),
        &parameterization,
        &objective,
        options(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
}

#[test]
fn prepared_workspace_solves_do_not_allocate() {
    let (plan, source) = fdm_problem();
    let unit = Diffusion(1.0);
    let problem = EllipticProblem::new(&plan, &source);
    let solution = solve(&plan, &source);
    let objective = Objective(plan.lumped_mass().to_vec());
    let parameterization = plan
        .parameterize(
            &P0,
            &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
        )
        .unwrap();
    let adjoint =
        PdeEllipticAdjointPlan::new(&problem, &parameterization, &objective, options()).unwrap();
    let mut workspace = adjoint.workspace().unwrap();
    let (mut lambda, mut gradient) = (vec![0.0; solution.len()], vec![0.0; P0.len()]);
    let mut run = || {
        adjoint
            .solve_into(
                &problem,
                &parameterization,
                &solution,
                &objective,
                &mut VectorViewMut::from_slice(&mut lambda),
                &mut VectorViewMut::from_slice(&mut gradient),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap()
    };
    run();
    (run)();
}
