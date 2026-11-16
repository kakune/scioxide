//! Cross-discretization model and lifecycle contracts.

#![allow(clippy::unwrap_used)]

#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use mesh_fixture::explicit;
use scioxide_spatial::SegmentMesh;

use std::f64::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use scioxide_core::{
    AttemptOutcome, ErrorTolerance, ExecutionContext, NonFiniteOrigin, NormSelection,
    SciOxideError, SciOxideErrorView, Tolerance,
};
use scioxide_pde::{
    EllipticOptions, EllipticProblem, EllipticSolution, FieldData, FieldPoint, HyperbolicOptions,
    HyperbolicProblem, ParabolicOptions, ParabolicProblem, ParabolicScheme, PdeOutputSchedule,
    PdeSolution, PdeTrajectory, StructuredFdmCoefficients, StructuredFdmPlan,
};
use support::{
    DISCRETIZATIONS, Discretization, EllipticPde, End, Ends, Heat, HyperbolicPde, ParabolicPde,
    Plan1d, Steady, Wave, homogeneous_dirichlet, initial, moving, source, value, wave_initial,
};

fn grid(nodes: usize) -> SegmentMesh<f64> {
    mesh_fixture::uniform(0.0, 1.0, nodes)
}

/// Steady solve of `problem` with default elliptic options.
fn elliptic<P: EllipticPde<f64> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    grid: &SegmentMesh<f64>,
    problem: &P,
    boundaries: &Ends<f64>,
) -> Result<EllipticSolution<f64>, SciOxideError> {
    let steady = Steady(problem.clone());
    let plan = discretization.plan(grid, &steady, boundaries)?;
    let source = source(&steady);
    let solver = EllipticOptions::new();
    with_plan!(&plan, |p| EllipticProblem::new(p, &source)
        .solve_with_options(&solver))
}

/// A prepared plan (to restore full fields) and its owned transient solution.
type Solved = (Plan1d<f64>, PdeSolution<f64>);

/// Heat solve of `problem` to `end`; returns the plan to restore full fields.
fn parabolic<P: ParabolicPde<f64> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    grid: &SegmentMesh<f64>,
    problem: &P,
    boundaries: &Ends<f64>,
    options: &ParabolicOptions<f64>,
    end: f64,
) -> Result<Solved, SciOxideError> {
    let heat = Heat(problem.clone());
    let plan = discretization.plan(grid, &heat, boundaries)?;
    let source = source(&heat);
    let solver = options
        .clone()
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let solution = with_plan!(&plan, |p| ParabolicProblem::new(p, &source, end)?
        .solve_with_options(&initial(problem), &solver))?;
    Ok((plan, solution))
}

/// Wave solve of `problem` to `end` from `initial_step` with accepted-step output
/// and a square-root-epsilon local error tolerance.
fn hyperbolic<P: HyperbolicPde<f64> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    grid: &SegmentMesh<f64>,
    problem: &P,
    boundaries: &Ends<f64>,
    end: f64,
    initial_step: f64,
) -> Result<Solved, SciOxideError> {
    let wave = Wave(problem.clone());
    let plan = discretization.plan(grid, &wave, boundaries)?;
    let source = source(&wave);
    let (displacement, velocity) = wave_initial(problem);
    let solver = HyperbolicOptions::new()
        .with_initial_step(initial_step)
        .with_error_tolerance(ErrorTolerance::from_tolerance(
            Tolerance::from_sqrt_epsilon(),
            NormSelection::RootMeanSquare,
        ))
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let solution = with_plan!(&plan, |p| HyperbolicProblem::new(p, &source, end)?
        .solve_with_options(&displacement, &velocity, &solver))?;
    Ok((plan, solution))
}

/// Emitted samples as full fields in coordinate order.
fn snapshots((plan, solution): &Solved) -> Vec<(f64, Vec<f64>)> {
    let trajectory = solution.trajectory();
    trajectory
        .times()
        .iter()
        .zip(trajectory.states())
        .map(|(&time, state)| (time, plan.full(state).unwrap()))
        .collect()
}

fn assert_domain_error(error: SciOxideError) {
    assert!(
        matches!(error.view(), SciOxideErrorView::DomainError { .. }),
        "expected DomainError, got {error}"
    );
}

fn assert_callback_non_finite(error: &SciOxideError) {
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ),
        "expected callback NonFinite, got {error}"
    );
}

#[derive(Clone)]
struct NegativeElliptic;

impl EllipticPde<f64> for NegativeElliptic {
    fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(-1.0)
    }

    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn source(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}

#[derive(Clone)]
struct ConstantCoefficientProblem {
    coefficient: f64,
}

impl ParabolicPde<f64> for ConstantCoefficientProblem {
    fn diffusion(&self, _x: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(self.coefficient)
    }

    fn source(&self, _x: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * x[0]).sin())
    }
}

/// End time of every heat solve.
const FINAL_TIME: f64 = 0.001;

fn explicit_options() -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.0001)
}

fn implicit_options(scheme: ParabolicScheme) -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(scheme)
        .with_step(0.0001)
}

fn options(scheme: ParabolicScheme) -> ParabolicOptions<f64> {
    if scheme == ParabolicScheme::ExplicitDopri5 {
        explicit_options()
    } else {
        implicit_options(scheme)
    }
}

const SCHEMES: [ParabolicScheme; 3] = [
    ParabolicScheme::ExplicitDopri5,
    ParabolicScheme::BackwardEuler,
    ParabolicScheme::CrankNicolson,
];

/// FDM admits nonnegative diffusion (a negative value is a domain error);
/// FEM and FVM require a positive-definite diffusion tensor (invalid input).
fn assert_inadmissible_diffusion(_: Discretization, error: SciOxideError) {
    assert_domain_error(error);
}

#[test]
fn negative_diffusion_is_rejected_by_every_route() {
    let grid = grid(17);
    let boundaries = homogeneous_dirichlet();

    for discretization in DISCRETIZATIONS {
        assert_inadmissible_diffusion(
            discretization,
            elliptic(discretization, &grid, &NegativeElliptic, &boundaries)
                .map(|_| ())
                .unwrap_err(),
        );
    }

    let problem = ConstantCoefficientProblem { coefficient: -1.0 };
    for scheme in SCHEMES {
        for discretization in DISCRETIZATIONS {
            assert_inadmissible_diffusion(
                discretization,
                parabolic(
                    discretization,
                    &grid,
                    &problem,
                    &boundaries,
                    &self::options(scheme),
                    FINAL_TIME,
                )
                .map(|_| ())
                .unwrap_err(),
            );
        }
    }
}

#[test]
fn zero_diffusion_is_valid_for_explicit_and_implicit_fdm() {
    let grid = grid(17);
    let boundaries = homogeneous_dirichlet();
    let problem = ConstantCoefficientProblem { coefficient: 0.0 };

    for scheme in SCHEMES {
        let options = options(scheme);
        let (_, solution) = parabolic(
            Discretization::Fdm,
            &grid,
            &problem,
            &boundaries,
            &options,
            FINAL_TIME,
        )
        .unwrap();
        assert!(solution.report().is_success());
        for discretization in [Discretization::Fem, Discretization::Fvm] {
            assert_inadmissible_diffusion(
                discretization,
                parabolic(
                    discretization,
                    &grid,
                    &problem,
                    &boundaries,
                    &options,
                    FINAL_TIME,
                )
                .map(|_| ())
                .unwrap_err(),
            );
        }
    }
}

#[derive(Clone)]
struct VariableDiffusionHeat;

impl ParabolicPde<f64> for VariableDiffusionHeat {
    fn diffusion(&self, x: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(1.0 + x[0])
    }

    fn source(&self, _x: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * x[0]).sin())
    }
}

fn max_solution_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f64::max)
}

#[test]
fn explicit_and_implicit_routes_use_the_same_spatial_operator() {
    let grid = grid(33);
    let boundaries = homogeneous_dirichlet();
    let explicit = explicit_options();
    let implicit = implicit_options(ParabolicScheme::CrankNicolson);

    for discretization in DISCRETIZATIONS {
        let final_field = |options| {
            let (plan, solution) = parabolic(
                discretization,
                &grid,
                &VariableDiffusionHeat,
                &boundaries,
                options,
                FINAL_TIME,
            )
            .unwrap();
            assert_eq!(solution.state().time(), FINAL_TIME);
            plan.full(solution.values()).unwrap()
        };
        assert!(
            max_solution_difference(&final_field(&explicit), &final_field(&implicit)) < 2.0e-6,
            "{}",
            discretization.name()
        );
    }
}

#[derive(Clone)]
struct ManufacturedVariableWave;

impl HyperbolicPde<f64> for ManufacturedVariableWave {
    fn wave_speed(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0 + x[0])
    }

    fn source(&self, x: &[f64], time: f64) -> Result<f64, SciOxideError> {
        let sin = (PI * x[0]).sin();
        let cos = (PI * x[0]).cos();
        let speed = 1.0 + x[0];
        Ok((speed * speed * PI * PI * sin - 2.0 * speed * PI * cos - sin) * time.cos())
    }

    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * x[0]).sin())
    }

    fn initial_velocity(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}

fn manufactured_wave_error(time: f64, values: &[f64], coordinates: &[f64]) -> f64 {
    assert_eq!(values.len(), coordinates.len());
    values
        .iter()
        .zip(coordinates)
        .map(|(&value, &x)| (value - (PI * x).sin() * time.cos()).abs())
        .fold(0.0, f64::max)
}

#[test]
fn variable_speed_wave_is_the_same_divergence_equation_for_all_solvers() {
    let grid = grid(65);
    let boundaries = homogeneous_dirichlet();
    for (discretization, bound) in DISCRETIZATIONS.into_iter().zip([2.0e-3, 2.0e-3, 3.0e-3]) {
        let (plan, solution) = hyperbolic(
            discretization,
            &grid,
            &ManufacturedVariableWave,
            &boundaries,
            0.02,
            0.001,
        )
        .unwrap();
        let time = solution.state().time();
        let values = plan.full(solution.values()).unwrap();
        assert!(
            manufactured_wave_error(time, &values, &discretization.coordinates(&grid)) < bound,
            "{}",
            discretization.name()
        );
    }
}

#[derive(Clone)]
struct InvalidWave {
    speed: f64,
    displacement: f64,
    velocity: f64,
}

impl HyperbolicPde<f64> for InvalidWave {
    fn wave_speed(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.speed)
    }

    fn source(&self, _x: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn initial_condition(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.displacement)
    }

    fn initial_velocity(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.velocity)
    }
}

fn hyperbolic_runs(
    problem: &InvalidWave,
    boundaries: &Ends<f64>,
    final_time: f64,
) -> [Result<Solved, SciOxideError>; 3] {
    let grid = grid(17);
    DISCRETIZATIONS.map(|discretization| {
        hyperbolic(
            discretization,
            &grid,
            problem,
            boundaries,
            final_time,
            0.0001,
        )
    })
}

#[test]
fn wave_speed_is_validated_and_fixed_boundaries_override_initial_data() {
    // The squared speed is the plan diffusion: a non-finite speed is rejected.
    for result in hyperbolic_runs(
        &InvalidWave {
            speed: f64::NAN,
            displacement: 0.0,
            velocity: 0.0,
        },
        &homogeneous_dirichlet(),
        0.001,
    ) {
        assert!(matches!(
            result.map(|_| ()).unwrap_err().view(),
            SciOxideErrorView::NonFinite { .. }
        ));
    }

    // Initial data is sampled at free unknowns only; fixed values always come
    // from the Dirichlet data, including the initial observation.
    for invalid in [
        InvalidWave {
            speed: 1.0,
            displacement: 1.0,
            velocity: 0.0,
        },
        InvalidWave {
            speed: 1.0,
            displacement: 0.0,
            velocity: 1.0,
        },
    ] {
        for (discretization, result) in DISCRETIZATIONS.into_iter().zip(hyperbolic_runs(
            &invalid,
            &homogeneous_dirichlet(),
            0.001,
        )) {
            let solution = result.unwrap();
            assert!(solution.1.report().is_success());
            if discretization != Discretization::Fvm {
                let samples = snapshots(&solution);
                assert!(!samples.is_empty());
                for (_, values) in &samples {
                    assert_eq!(values[0], 0.0);
                    assert_eq!(values[values.len() - 1], 0.0);
                }
            }
        }
    }
}

#[test]
fn hyperbolic_supports_time_dependent_dirichlet_data() {
    // u = t solves u_tt = u_xx with u(x, 0) = 0, u_t(x, 0) = 1 and u = t on both ends.
    let end = || End::dirichlet(moving(|_, time| Ok(time)));
    let boundaries = Ends::new(end(), end());
    let wave = InvalidWave {
        speed: 1.0,
        displacement: 0.0,
        velocity: 1.0,
    };
    for (discretization, result) in
        DISCRETIZATIONS
            .into_iter()
            .zip(hyperbolic_runs(&wave, &boundaries, 0.1))
    {
        let solution = result.unwrap();
        let samples = snapshots(&solution);
        let times = samples.iter().map(|(time, _)| *time).collect::<Vec<_>>();
        let fields = samples
            .iter()
            .map(|(_, values)| values.as_slice())
            .collect::<Vec<_>>();
        support::assert_valid_solution(
            discretization.name(),
            solution.1.report(),
            &times,
            &fields,
            discretization.coordinates(&grid(17)).len(),
        );
        for (time, values) in &samples {
            for &value in values {
                assert!(
                    (value - time).abs() < 1e-10,
                    "{}: {value} at t={time}",
                    discretization.name(),
                );
            }
        }
    }
}

/// Negative diffusion that must never be sampled before boundary binding fails.
struct NegativeCoefficients;
impl StructuredFdmCoefficients<f64, f64, 1> for NegativeCoefficients {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(-1.0)
    }
}

#[test]
fn boundary_sets_are_bound_to_their_exact_grid() {
    let original = explicit(vec![0.0, 0.25, 1.0]);
    let other = explicit(vec![0.0, 0.5, 1.0]);
    let boundaries = homogeneous_dirichlet().structured(&original).unwrap();
    let plan = StructuredFdmPlan::with_options(
        other,
        &NegativeCoefficients,
        &boundaries,
        support::limits(),
    );
    assert!(matches!(
        plan.map(|_| ()).unwrap_err().view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn non_finite_boundary_values_are_typed_callback_failures() {
    // Coefficients are sampled during plan preparation; boundary data is
    // evaluated by the solve.
    let grid = grid(17);
    let boundaries = Ends::new(
        End::dirichlet(value(|_| Ok(f64::NAN))),
        End::dirichlet(value(|_| Ok(0.0))),
    );
    #[derive(Clone)]
    struct Laplace;
    impl EllipticPde<f64> for Laplace {
        fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn source(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
    }
    for discretization in DISCRETIZATIONS {
        let error = elliptic(discretization, &grid, &Laplace, &boundaries)
            .map(|_| ())
            .unwrap_err();
        assert_callback_non_finite(&error);
    }
}

#[derive(Clone)]
struct FailOnceSource {
    failed: Arc<AtomicBool>,
}

impl ParabolicPde<f64> for FailOnceSource {
    fn diffusion(&self, _x: &[f64], _time: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }

    fn source(&self, _x: &[f64], time: f64) -> Result<f64, SciOxideError> {
        if time > 0.0 && !self.failed.swap(true, Ordering::Relaxed) {
            Ok(f64::NAN)
        } else {
            Ok(0.0)
        }
    }

    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * x[0]).sin())
    }
}

/// Advance one state, sink and workspace twice against an FDM plan of
/// [`FailOnceSource`]: the failure keeps the initial state, and the retry succeeds.
fn fail_once(options: ParabolicOptions<f64>) {
    let problem = Heat(FailOnceSource {
        failed: Arc::new(AtomicBool::new(false)),
    });
    let plan = Discretization::Fdm
        .plan(&grid(17), &problem, &homogeneous_dirichlet())
        .unwrap();
    let source = support::source(&problem);
    let initial = FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok((PI * p.point[0]).sin()));
    let solver = options;
    let Plan1d::Fdm(fdm) = &plan else {
        unreachable!("an FDM plan was requested")
    };
    {
        let problem = ParabolicProblem::new(fdm, &source, FINAL_TIME).unwrap();
        let mut state = problem.initial_state(&initial, &solver).unwrap();
        let start = state.coordinates().to_vec();
        let mut workspace = problem.workspace().unwrap();
        let mut output = PdeTrajectory::new();
        let error = problem
            .advance(
                &solver,
                &mut state,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_callback_non_finite(&error);
        // No step was accepted: the recorded progress and the state keep the initial
        // point, and nothing was emitted.
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        let progress = attempt.progress().unwrap();
        assert_eq!(progress.time(), 0.0);
        assert_eq!(progress.steps().accepted(), 0);
        assert_eq!(progress.state().coordinates(), start.as_slice());
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.coordinates(), start.as_slice());
        assert!(state.coordinates().iter().all(|value| value.is_finite()));
        assert!(output.times().is_empty());

        let report = problem
            .advance(
                &solver,
                &mut state,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(state.time(), FINAL_TIME);
        assert_eq!(output.times(), [FINAL_TIME]);
        assert!(output.states()[0].iter().all(|value| value.is_finite()));
    }
}

#[test]
fn failed_solve_object_preserves_output_and_can_be_retried() {
    fail_once(implicit_options(ParabolicScheme::BackwardEuler));
}

#[test]
fn explicit_callback_failure_retains_the_last_accepted_state() {
    fail_once(explicit_options());
}

#[test]
fn extreme_cell_and_mass_arithmetic_returns_typed_errors() {
    let tiny = f64::from_bits(1);
    let grid = explicit(vec![0.0, tiny, tiny + tiny]);
    let problem = ConstantCoefficientProblem { coefficient: 1.0 };
    let options = implicit_options(ParabolicScheme::BackwardEuler);

    for discretization in [Discretization::Fem, Discretization::Fvm] {
        assert!(
            parabolic(
                discretization,
                &grid,
                &problem,
                &homogeneous_dirichlet(),
                &options,
                FINAL_TIME,
            )
            .is_err()
        );
    }
}
