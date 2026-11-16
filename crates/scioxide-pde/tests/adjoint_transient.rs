//! Transient adjoints of the method-of-lines adapter over the crate's own plans,
//! against central finite differences of the crate's own `ParabolicProblem` solves.
#![allow(clippy::unwrap_used)]
#[path = "support/fd.rs"]
mod fd;
#[path = "../examples/support/simplex_fem.rs"]
mod meshes;

use scioxide_ad::CheckpointSchedule;
use scioxide_core::{ExecutionContext, SciOxideError, SciOxideErrorKind, SolveContext};
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::{
    AdjointMode, AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue, AdjointOptions,
    DaeOptions, OdeAdjointSystem, OdeSystem,
};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, BoundaryPoint, CflPolicy, DifferentiableDiscretization,
    FieldData, FieldPoint, ParabolicOptions, ParabolicProblem, ParabolicScheme,
    PdeConditionTangent, PdeParameterTangent, PdeParameterization, PdeTransientAdjointPlan,
    SimplexFemCoefficients, SimplexFemGeometry, SimplexFemLimits, SimplexFemOptions,
    SimplexFemOrder, SimplexFemPlan, SpatialDiscretization, SpatialLimits, StructuredBoundaries,
    StructuredFdmCoefficients, StructuredFdmPlan, TaggedBoundaries,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh, SimplexBoundaryTags,
};

/// Parameters: diffusion scale κ, source amplitude s and Dirichlet amplitude g.
const P0: [f64; 3] = [1.2, 0.9, 0.5];
const NAMES: [&str; 3] = ["kappa", "source", "dirichlet"];
const END: f64 = 0.25;
const STEP: f64 = 1.0 / 64.0;

/// `k(x) = κ (1 + x₀/2)`; with `κ = 1` it is the tangent `∂k/∂κ`.
struct Diffusion(f64);
impl StructuredFdmCoefficients<f64, f64, 1> for Diffusion {
    fn diffusion(&self, _: usize, _: usize, x: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(self.0 * (1.0 + 0.5 * x[0]))
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

/// Time-dependent source `s (1 + x₀)(1 + t)`.
fn source<const D: usize>(s: f64) -> FieldData<f64, f64, D> {
    FieldData::time_dependent(move |q: FieldPoint<f64, D>, t| {
        Ok(s * (1.0 + q.point[0]) * (1.0 + t))
    })
}
/// Time-dependent Dirichlet data `g (1 + x_last)(1 + 2t)`.
fn dirichlet<const D: usize>(g: f64) -> BoundaryData<f64, f64, D> {
    BoundaryData::time_dependent(move |q: BoundaryPoint<f64, D>, t| {
        Ok(g * (1.0 + q.point[D - 1]) * (1.0 + 2.0 * t))
    })
}
fn initial<const D: usize>() -> FieldData<f64, f64, D> {
    FieldData::time_invariant(|q: FieldPoint<f64, D>| Ok((3.0 * q.point[0]).sin()))
}

/// Derivative data of κ, s and g; the Dirichlet condition comes first.
fn tangents<'a, const D: usize, C: ?Sized>(
    unit: &'a C,
    conditions: usize,
) -> Vec<PdeParameterTangent<'a, f64, f64, D, C>> {
    let mut boundary = vec![PdeConditionTangent::new(dirichlet(1.0))];
    boundary.extend((1..conditions).map(|_| PdeConditionTangent::zero()));
    vec![
        PdeParameterTangent::new().with_coefficients(unit),
        PdeParameterTangent::new().with_source(source(1.0)),
        PdeParameterTangent::new().with_boundaries(boundary),
    ]
}

/// Terminal `½ Σ m u²` plus, optionally, a running `∫ Σ m u dt`, an observation
/// `Σ m u(t_o)` and a direct `0.1 κ²` term; all over the free unknowns.
struct Objective {
    mass: Vec<f64>,
    running: bool,
    observations: Vec<f64>,
}
impl Objective {
    fn terminal(mass: &[f64]) -> Self {
        Self {
            mass: mass.to_vec(),
            running: false,
            observations: Vec::new(),
        }
    }
    fn value(&self, u: &[f64], p: &[f64]) -> f64 {
        0.5 * self.mass.iter().zip(u).map(|(m, u)| m * u * u).sum::<f64>() + 0.1 * p[0] * p[0]
    }
}
impl AdjointObjective<f64> for Objective {
    fn observation_times(&self) -> &[f64] {
        &self.observations
    }
    fn observation(
        &self,
        _: usize,
        u: &[f64],
        _: &[f64],
        gu: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<f64, SciOxideError> {
        gu.copy_from_slice(&self.mass);
        gp.fill(0.0);
        Ok(self.mass.iter().zip(u).map(|(m, u)| m * u).sum())
    }
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        u: &[f64],
        p: &[f64],
        gu: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        gp.fill(0.0);
        let value = match part {
            AdjointObjectivePart::Terminal => {
                for ((g, m), u) in gu.iter_mut().zip(&self.mass).zip(u) {
                    *g = m * u;
                }
                gp[0] = 0.2 * p[0];
                self.value(u, p)
            }
            AdjointObjectivePart::Running if self.running => {
                gu.copy_from_slice(&self.mass);
                self.mass.iter().zip(u).map(|(m, u)| m * u).sum()
            }
            AdjointObjectivePart::Running => {
                gu.fill(0.0);
                0.0
            }
        };
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: 0.0,
        })
    }
}

fn options(step: f64) -> AdjointOptions<f64> {
    AdjointOptions::new(AdjointMode::Discrete, step)
        .unwrap()
        .with_fixed_step(true)
        .with_tape_capacity(4096)
        .with_max_replay_steps(1 << 16)
        .with_max_reverse_steps(1 << 16)
}

fn fdm(p: &[f64]) -> StructuredFdmPlan<f64, f64, 1> {
    let mesh = SegmentMesh::new(
        Grid1d::new(
            [CoordinateAxis::uniform(0.0, 1.0, 17).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    let side = |s| BoundarySet::axis_side(&mesh, 0, s).unwrap();
    let boundaries = StructuredBoundaries::new(
        &mesh,
        1,
        vec![
            BoundaryCondition::dirichlet(side(AxisSide::Start), dirichlet(p[2])),
            BoundaryCondition::robin(side(AxisSide::End), 1.5, BoundaryData::constant(0.2)),
        ],
    )
    .unwrap();
    StructuredFdmPlan::with_options(
        mesh,
        &Diffusion(p[0]),
        &boundaries,
        SpatialLimits::new(1 << 12, 1 << 16, usize::MAX),
    )
    .unwrap()
}

fn fem(p: &[f64]) -> SimplexFemPlan<f64, f64, 2> {
    let mesh = meshes::triangle_grid::<f64>(4).unwrap();
    let tags = SimplexBoundaryTags::new(
        &mesh,
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
    let boundaries = TaggedBoundaries::new(
        &mesh,
        &tags,
        1,
        vec![
            BoundaryCondition::dirichlet(0, dirichlet(p[2])),
            BoundaryCondition::natural(1, BoundaryData::constant(0.1)),
        ],
    )
    .unwrap();
    SimplexFemPlan::with_options(
        mesh,
        SimplexFemGeometry::Affine,
        SimplexFemOrder::P1,
        &Diffusion(p[0]),
        &boundaries,
        SimplexFemOptions::new().with_limits(SimplexFemLimits::new(
            1 << 12,
            1 << 18,
            64,
            usize::MAX,
        )),
    )
    .unwrap()
}

/// Free final state of the crate's own fixed-step parabolic solve.
fn parabolic<P: SpatialDiscretization<f64, f64, D>, const D: usize>(
    plan: &P,
    s: f64,
    scheme: ParabolicScheme,
) -> Vec<f64> {
    let source = source::<D>(s);
    let problem = ParabolicProblem::new(plan, &source, END).unwrap();
    let solution = problem
        .solve_with_options(
            &initial(),
            &(ParabolicOptions::new().with_scheme(scheme).with_step(STEP)),
        )
        .unwrap();
    solution.state().coordinates().to_vec()
}

/// Constructor of an implicit transient adjoint on the 1D FDM problem.
type Implicit = fn(
    &ParabolicProblem<'_, f64, StructuredFdmPlan<f64, f64, 1>, 1>,
    &PdeParameterization<f64, f64, 1>,
    &Objective,
    Option<scioxide_ode::OdeStageBackend>,
    scioxide_ode::DaeAdjointOptions<f64>,
) -> Result<PdeTransientAdjointPlan<f64>, SciOxideError>;

fn implicit(scheme: ParabolicScheme) -> Implicit {
    match scheme {
        ParabolicScheme::CrankNicolson => PdeTransientAdjointPlan::crank_nicolson,
        _ => PdeTransientAdjointPlan::backward_euler,
    }
}

/// Gradients of the terminal objective against finite differences of
/// `ParabolicProblem` solves for one family.
fn terminal_gradients<P, const D: usize>(
    name: &str,
    build: impl Fn(&[f64]) -> P,
    parameterize: impl Fn(&P) -> PdeParameterization<f64, f64, D>,
) where
    P: DifferentiableDiscretization<f64, f64, D>,
{
    for scheme in [
        ParabolicScheme::BackwardEuler,
        ParabolicScheme::CrankNicolson,
    ] {
        let plan = build(&P0);
        let data = source::<D>(P0[1]);
        let problem = ParabolicProblem::new(&plan, &data, END).unwrap();
        let parameterization = parameterize(&plan);
        let objective = Objective::terminal(problem.system().free_mass());
        let prepare = match scheme {
            ParabolicScheme::CrankNicolson => PdeTransientAdjointPlan::crank_nicolson,
            _ => PdeTransientAdjointPlan::backward_euler,
        };
        let adjoint = prepare(
            &problem,
            &parameterization,
            &objective,
            None,
            scioxide_ode::DaeAdjointOptions::new(
                options(STEP),
                (DaeOptions::new())
                    .numerics()
                    .clone()
                    .with_initialization(scioxide_ode::DaeInitialization::Project),
            )
            .unwrap(),
        )
        .unwrap();
        let start = problem.system().initial_state(&initial()).unwrap();
        let result = adjoint
            .solve(&problem, &parameterization, &objective, &start)
            .unwrap();
        assert_eq!(result.report().accepted_steps(), 16);
        // The adjoint's own forward matches the crate's parabolic solve.
        let forward = objective.value(&parabolic(&plan, P0[1], scheme), &P0);
        let recorded = result.report().objective().unwrap();
        assert!(
            (recorded - forward).abs() <= 1e-10 * forward.abs(),
            "{name} {scheme:?}: {recorded:e} vs {forward:e}"
        );
        for j in 0..P0.len() {
            let (estimate, tolerance) = fd::central(
                |x| {
                    let mut p = P0;
                    p[j] = x;
                    objective.value(&parabolic(&build(&p), p[1], scheme), &p)
                },
                P0[j],
                1e-12,
            );
            let gradient = result.parameter_gradient()[j];
            assert!(
                (gradient - estimate).abs() <= tolerance,
                "{name} {scheme:?} {}: adjoint {gradient:e}, finite difference {estimate:e} ± {tolerance:e}",
                NAMES[j]
            );
        }
        // The initial cotangent against a perturbed initial state.
        let index = start.len() / 2;
        let (estimate, tolerance) = fd::central(
            |x| {
                let mut shifted = start.clone();
                shifted[index] = x;
                let mut gradient = vec![0.0; P0.len()];
                let mut initial_gradient = vec![0.0; start.len()];
                let mut workspace = adjoint.workspace().unwrap();
                adjoint
                    .solve_into(
                        &problem,
                        &parameterization,
                        &objective,
                        &shifted,
                        &mut initial_gradient,
                        &mut gradient,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap()
                    .objective()
                    .unwrap()
            },
            start[index],
            1e-12,
        );
        let actual = result.initial_gradient()[index];
        assert!(
            (actual - estimate).abs() <= tolerance,
            "{name} {scheme:?} initial: {actual:e} vs {estimate:e}"
        );
    }
}

#[test]
fn fdm_backward_euler_and_crank_nicolson_match_parabolic_finite_differences() {
    let unit = Diffusion(1.0);
    terminal_gradients("FDM", fdm, |plan| {
        plan.parameterize(
            &P0,
            &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
        )
        .unwrap()
    });
}

#[test]
fn fem_backward_euler_and_crank_nicolson_match_parabolic_finite_differences() {
    let unit = Diffusion(1.0);
    terminal_gradients("FEM P1", fem, |plan| {
        plan.parameterize(
            &P0,
            &tangents::<2, dyn SimplexFemCoefficients<f64, f64, 2>>(&unit, 2),
        )
        .unwrap()
    });
}

/// Objective of the adjoint plan's own forward recording at `p`.
fn recorded(
    p: &[f64],
    objective: &Objective,
    prepare: impl Fn(
        &ParabolicProblem<'_, f64, StructuredFdmPlan<f64, f64, 1>, 1>,
        &PdeParameterization<f64, f64, 1>,
    ) -> PdeTransientAdjointPlan<f64>,
) -> (f64, Vec<f64>) {
    let unit = Diffusion(1.0);
    let plan = fdm(p);
    let data = source::<1>(p[1]);
    let problem = ParabolicProblem::new(&plan, &data, END).unwrap();
    let parameterization = plan
        .parameterize(
            p,
            &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
        )
        .unwrap();
    let adjoint = prepare(&problem, &parameterization);
    let start = problem.system().initial_state(&initial()).unwrap();
    let result = adjoint
        .solve(&problem, &parameterization, objective, &start)
        .unwrap();
    (
        result.report().objective().unwrap(),
        result.parameter_gradient().to_vec(),
    )
}

fn check_recorded(
    name: &str,
    objective: &Objective,
    prepare: impl Fn(
        &ParabolicProblem<'_, f64, StructuredFdmPlan<f64, f64, 1>, 1>,
        &PdeParameterization<f64, f64, 1>,
    ) -> PdeTransientAdjointPlan<f64>,
) {
    let (_, gradient) = recorded(&P0, objective, &prepare);
    for j in 0..P0.len() {
        let (estimate, tolerance) = fd::central(
            |x| {
                let mut p = P0;
                p[j] = x;
                recorded(&p, objective, &prepare).0
            },
            P0[j],
            1e-12,
        );
        assert!(
            (gradient[j] - estimate).abs() <= tolerance,
            "{name} {}: adjoint {:e}, finite difference {estimate:e} ± {tolerance:e}",
            NAMES[j],
            gradient[j]
        );
    }
}

#[test]
fn running_costs_and_observations_match_recorded_objective_differences() {
    let free = ParabolicProblem::new(&fdm(&P0), &source::<1>(P0[1]), END)
        .unwrap()
        .system()
        .free_mass()
        .to_vec();
    let objective = Objective {
        mass: free,
        running: true,
        observations: vec![0.1, 0.2],
    };
    for scheme in [
        ParabolicScheme::BackwardEuler,
        ParabolicScheme::CrankNicolson,
    ] {
        check_recorded(
            &format!("{scheme:?}"),
            &objective,
            |problem, parameterization| {
                implicit(scheme)(
                    problem,
                    parameterization,
                    &objective,
                    None,
                    scioxide_ode::DaeAdjointOptions::new(
                        options(STEP),
                        (DaeOptions::new())
                            .numerics()
                            .clone()
                            .with_initialization(scioxide_ode::DaeInitialization::Project),
                    )
                    .unwrap(),
                )
                .unwrap()
            },
        );
    }
}

#[test]
fn dopri5_matches_recorded_objective_differences() {
    let free = ParabolicProblem::new(&fdm(&P0), &source::<1>(P0[1]), END)
        .unwrap()
        .system()
        .free_mass()
        .to_vec();
    let objective = Objective::terminal(&free);
    let step = 1.0 / 2048.0;
    check_recorded("DP5", &objective, |problem, parameterization| {
        PdeTransientAdjointPlan::dopri5(
            problem,
            parameterization,
            &objective,
            CflPolicy::MaximumStep(step),
            options(step),
        )
        .unwrap()
    });
}

#[test]
fn checkpoint_policies_give_identical_gradients() {
    let free = ParabolicProblem::new(&fdm(&P0), &source::<1>(P0[1]), END)
        .unwrap()
        .system()
        .free_mass()
        .to_vec();
    let objective = Objective::terminal(&free);
    let gradients: Vec<_> = [
        CheckpointPolicy::StoreAll,
        CheckpointPolicy::Explicit(CheckpointSchedule::binomial(16, 2).unwrap()),
    ]
    .into_iter()
    .map(|policy| {
        recorded(&P0, &objective, |problem, parameterization| {
            PdeTransientAdjointPlan::crank_nicolson(
                problem,
                parameterization,
                &objective,
                None,
                scioxide_ode::DaeAdjointOptions::new(
                    options(STEP)
                        .with_tape_capacity(16)
                        .with_checkpoint_policy(policy.clone()),
                    (DaeOptions::new())
                        .numerics()
                        .clone()
                        .with_initialization(scioxide_ode::DaeInitialization::Project),
                )
                .unwrap(),
            )
            .unwrap()
        })
        .1
    })
    .collect();
    assert_eq!(gradients[0], gradients[1]);
}

#[test]
fn memory_budget_selects_a_schedule_within_the_whole_plan_limit() {
    let free = ParabolicProblem::new(&fdm(&P0), &source::<1>(P0[1]), END)
        .unwrap()
        .system()
        .free_mass()
        .to_vec();
    let objective = Objective::terminal(&free);
    let bytes = std::cell::Cell::new(0);
    let run = |policy: CheckpointPolicy, limit: usize| {
        recorded(&P0, &objective, |problem, parameterization| {
            let plan = PdeTransientAdjointPlan::crank_nicolson(
                problem,
                parameterization,
                &objective,
                None,
                scioxide_ode::DaeAdjointOptions::new(
                    options(STEP)
                        .with_tape_capacity(16)
                        .with_checkpoint_policy(policy.clone())
                        .with_max_workspace_bytes(limit),
                    (DaeOptions::new())
                        .numerics()
                        .clone()
                        .with_initialization(scioxide_ode::DaeInitialization::Project),
                )
                .unwrap(),
            )
            .unwrap();
            bytes.set(plan.workspace_bytes());
            plan
        })
        .1
    };
    let stored = run(CheckpointPolicy::StoreAll, usize::MAX);
    let store_all = bytes.get();
    // One byte below storing every step forces a binomial schedule for the whole plan.
    let budgeted = run(CheckpointPolicy::MemoryBudget, store_all - 1);
    assert!(bytes.get() < store_all);
    assert_eq!(stored, budgeted);
}

#[test]
fn invalid_configurations_fail_and_preserve_outputs() {
    let unit = Diffusion(1.0);
    let plan = fdm(&P0);
    let data = source::<1>(P0[1]);
    let problem = ParabolicProblem::new(&plan, &data, END).unwrap();
    let parameterization = plan
        .parameterize(
            &P0,
            &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
        )
        .unwrap();
    let objective = Objective::terminal(problem.system().free_mass());
    let dae = || DaeOptions::new();
    let error =
        |result: Result<PdeTransientAdjointPlan<f64>, SciOxideError>| result.err().unwrap().kind();
    // Implicit schemes require projection rather than discarding this control.
    assert_eq!(
        error(PdeTransientAdjointPlan::backward_euler(
            &problem,
            &parameterization,
            &objective,
            None,
            scioxide_ode::DaeAdjointOptions::new(
                options(STEP),
                scioxide_ode::DaeNumericalOptions::new()
            )
            .unwrap(),
        )),
        SciOxideErrorKind::InvalidInput,
    );
    // Continuous adjoints and adaptive implicit steps are rejected.
    let continuous = AdjointOptions::new(AdjointMode::Continuous, STEP).unwrap();
    assert_eq!(
        error(PdeTransientAdjointPlan::backward_euler(
            &problem,
            &parameterization,
            &objective,
            None,
            scioxide_ode::DaeAdjointOptions::new(
                continuous.clone(),
                (dae())
                    .numerics()
                    .clone()
                    .with_initialization(scioxide_ode::DaeInitialization::Project)
            )
            .unwrap()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error(PdeTransientAdjointPlan::dopri5(
            &problem,
            &parameterization,
            &objective,
            CflPolicy::Automatic,
            continuous
        )),
        SciOxideErrorKind::InvalidInput
    );
    let adaptive = AdjointOptions::new(AdjointMode::Discrete, STEP).unwrap();
    assert_eq!(
        error(PdeTransientAdjointPlan::crank_nicolson(
            &problem,
            &parameterization,
            &objective,
            None,
            scioxide_ode::DaeAdjointOptions::new(
                adaptive,
                (dae())
                    .numerics()
                    .clone()
                    .with_initialization(scioxide_ode::DaeInitialization::Project)
            )
            .unwrap()
        )),
        SciOxideErrorKind::InvalidInput
    );
    // A parameterization of another plan size.
    let other = SegmentMesh::new(
        Grid1d::new(
            [CoordinateAxis::uniform(0.0, 1.0, 9).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    let side = |s| BoundarySet::axis_side(&other, 0, s).unwrap();
    let boundaries = StructuredBoundaries::new(
        &other,
        1,
        vec![
            BoundaryCondition::dirichlet(side(AxisSide::Start), BoundaryData::zero()),
            BoundaryCondition::natural(side(AxisSide::End), BoundaryData::zero()),
        ],
    )
    .unwrap();
    let small = StructuredFdmPlan::with_options(
        other,
        &Diffusion(1.0),
        &boundaries,
        SpatialLimits::new(64, 1024, usize::MAX),
    )
    .unwrap();
    let foreign = small.parameterize(&[], &[]).unwrap();
    assert!(
        PdeTransientAdjointPlan::backward_euler(
            &problem,
            &foreign,
            &objective,
            None,
            scioxide_ode::DaeAdjointOptions::new(
                options(STEP),
                (dae())
                    .numerics()
                    .clone()
                    .with_initialization(scioxide_ode::DaeInitialization::Project)
            )
            .unwrap()
        )
        .is_err()
    );
    // A workspace of another scheme and too few steps fail transactionally.
    let euler = PdeTransientAdjointPlan::backward_euler(
        &problem,
        &parameterization,
        &objective,
        None,
        scioxide_ode::DaeAdjointOptions::new(
            options(STEP).with_tape_capacity(4),
            (dae())
                .numerics()
                .clone()
                .with_initialization(scioxide_ode::DaeInitialization::Project),
        )
        .unwrap(),
    )
    .unwrap();
    let explicit = PdeTransientAdjointPlan::dopri5(
        &problem,
        &parameterization,
        &objective,
        CflPolicy::MaximumStep(STEP),
        options(STEP),
    )
    .unwrap();
    let start = problem.system().initial_state(&initial()).unwrap();
    let (mut initial_gradient, mut gradient) = (vec![5.0; start.len()], vec![5.0; P0.len()]);
    for (plan, mut workspace) in [
        (&euler, explicit.workspace().unwrap()),
        (&euler, euler.workspace().unwrap()),
    ] {
        assert!(
            plan.solve_into(
                &problem,
                &parameterization,
                &objective,
                &start,
                &mut initial_gradient,
                &mut gradient,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .is_err()
        );
    }
    assert!(initial_gradient.iter().chain(&gradient).all(|&v| v == 5.0));
    assert_eq!(euler.scheme(), ParabolicScheme::BackwardEuler);
}

#[test]
fn semidiscrete_pullback_is_the_transpose_of_finite_difference_dynamics() {
    let unit = Diffusion(1.0);
    let plan = fdm(&P0);
    let data = source::<1>(P0[1]);
    let problem = ParabolicProblem::new(&plan, &data, END).unwrap();
    let parameterization = plan
        .parameterize(
            &P0,
            &tangents::<1, dyn StructuredFdmCoefficients<f64, f64, 1>>(&unit, 2),
        )
        .unwrap();
    let system = problem.system().adjoint(&parameterization).unwrap();
    assert_eq!(system.parameters(), &P0);
    let m = system.dimension();
    let state: Vec<f64> = (0..m).map(|i| (i as f64 * 0.37).cos()).collect();
    let seed: Vec<f64> = (0..m).map(|i| 1.0 + 0.1 * i as f64).collect();
    let (mut state_output, mut parameter_output) = (vec![0.0; m], vec![0.0; P0.len()]);
    let mut scratch = vec![0.0; system.pullback_workspace_len()];
    let time = 0.1;
    scioxide_core::SolveState::new()
        .begin_solve(
            ExecutionContext::sequential_native(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(1),
                scioxide_core::algorithm_id!("pde", "pullback-test", 1),
                0,
                0,
            ),
            &mut scioxide_core::SolveBudgets::unbounded(),
        )
        .map(|mut context| {
            system
                .pullback(
                    time,
                    &state,
                    &seed,
                    &mut state_output,
                    &mut parameter_output,
                    &mut scratch,
                    &mut context,
                )
                .unwrap();
        })
        .unwrap();
    // f(t, u; p) through freshly prepared plans.
    let rhs = |p: &[f64], u: &[f64]| {
        let plan = fdm(p);
        let data = source::<1>(p[1]);
        let problem = ParabolicProblem::new(&plan, &data, END).unwrap();
        let mut out = vec![0.0; m];
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential_native(),
                scioxide_core::SolveIdentity::new(
                    scioxide_core::ProblemToken::new(1),
                    scioxide_core::algorithm_id!("pde", "pullback-test", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        problem
            .system()
            .rhs(time, u, &mut out, &mut context)
            .unwrap();
        out
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    for j in 0..P0.len() {
        let (estimate, tolerance) = fd::central(
            |x| {
                let mut p = P0;
                p[j] = x;
                dot(&seed, &rhs(&p, &state))
            },
            P0[j],
            1e-14,
        );
        assert!(
            (parameter_output[j] - estimate).abs() <= tolerance,
            "{}",
            NAMES[j]
        );
    }
    let direction: Vec<f64> = (0..m).map(|i| (i as f64).sin()).collect();
    let shifted = |sign: f64| {
        let u: Vec<f64> = state
            .iter()
            .zip(&direction)
            .map(|(u, d)| u + sign * d)
            .collect();
        rhs(&P0, &u)
    };
    let jvp: Vec<f64> = shifted(1.0)
        .iter()
        .zip(shifted(-1.0))
        .map(|(a, b)| (a - b) / 2.0)
        .collect();
    assert!((dot(&seed, &jvp) - dot(&state_output, &direction)).abs() <= 1e-10);
}
