//! Transient solves with natural, Robin and time-dependent boundary data, and
//! mass- and integrator-aware automatic step limits.
#![allow(clippy::unwrap_used)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use scioxide_core::{ErrorTolerance, NormSelection, SciOxideError, Tolerance};
use scioxide_linalg::Vector;
use scioxide_ode::{DormandPrince5, OdeMethod, OdeOptions, OdeProblem, OdeState};
use scioxide_pde::{
    CflPolicy, FieldData, FieldPoint, HyperbolicOptions, HyperbolicProblem, MethodOfLinesOrder,
    ParabolicOptions, ParabolicProblem, ParabolicScheme, PdeSolveReport, SemidiscreteSystem,
};
use scioxide_spatial::SegmentMesh;
use support::{
    DISCRETIZATIONS, Discretization, End, Ends, Heat, HyperbolicPde, ParabolicPde, Plan1d, Wave,
    initial, moving, source, value, wave_initial,
};

const T_HEAT: f64 = 0.1;
const ALPHA: f64 = 2.0;

fn mesh(cells: usize) -> SegmentMesh<f64> {
    mesh_fixture::uniform(0.0, 1.0, cells + 1)
}

/// `u = e^{-t} sin(x + 1/2)` solves `u_t = u_xx` without a source.
fn heat_exact(x: f64, t: f64) -> f64 {
    (-t).exp() * (x + 0.5).sin()
}

#[derive(Clone)]
struct Decay;
impl ParabolicPde<f64> for Decay {
    fn diffusion(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(heat_exact(x[0], 0.0))
    }
}

/// Natural data `n·k u_x` on the left and Robin data `n·k u_x + α u` on the
/// right, both time-dependent.
fn heat_ends() -> Ends<f64> {
    Ends::new(
        End::natural(moving(|_, t: f64| Ok(-(-t).exp() * 0.5_f64.cos()))),
        End::robin(
            ALPHA,
            moving(|_, t: f64| Ok((-t).exp() * (1.5_f64.cos() + ALPHA * 1.5_f64.sin()))),
        ),
    )
}

fn heat(
    discretization: Discretization,
    cells: usize,
    options: &ParabolicOptions<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mesh = mesh(cells);
    let problem = Heat(Decay);
    let plan = discretization.plan(&mesh, &problem, &heat_ends()).unwrap();
    let data = source(&problem);
    let parabolic = options.clone();
    let solution = with_plan!(&plan, |p| ParabolicProblem::new(p, &data, T_HEAT)
        .unwrap()
        .solve_with_options(&initial(&Decay), &parabolic))
    .unwrap();
    assert!(solution.report().is_success());
    let values = plan.full(solution.values()).unwrap();
    (values, discretization.coordinates(&mesh))
}

fn tight_dopri5() -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(1e-4)
        .with_error_tolerance(
            ErrorTolerance::scalar(1e-11, 1e-11, NormSelection::Infinity).unwrap(),
        )
}

fn fixed(scheme: ParabolicScheme, step: f64) -> ParabolicOptions<f64> {
    ParabolicOptions::new().with_scheme(scheme).with_step(step)
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn heat_error(discretization: Discretization, cells: usize) -> f64 {
    let (values, x) = heat(discretization, cells, &tight_dopri5());
    let exact: Vec<_> = x.iter().map(|&x| heat_exact(x, T_HEAT)).collect();
    max_difference(&values, &exact)
}

#[test]
fn heat_with_natural_and_robin_ends_converges_in_space_under_dopri5() {
    for discretization in DISCRETIZATIONS {
        let (coarse, fine) = (
            heat_error(discretization, 16),
            heat_error(discretization, 32),
        );
        assert!(fine < 1e-3, "{discretization:?}: {fine:e}");
        assert!(
            coarse / fine > 3.5,
            "{discretization:?}: {coarse:e} -> {fine:e}"
        );
    }
}

#[test]
fn heat_with_natural_and_robin_ends_converges_in_time_under_be_and_cn() {
    for discretization in DISCRETIZATIONS {
        // The same semi-discretization integrated to near round-off isolates
        // the time-stepping error.
        let (reference, _) = heat(discretization, 32, &tight_dopri5());
        for (scheme, order) in [
            (ParabolicScheme::BackwardEuler, 1),
            (ParabolicScheme::CrankNicolson, 2),
        ] {
            let error = |step| {
                max_difference(
                    &heat(discretization, 32, &fixed(scheme, step)).0,
                    &reference,
                )
            };
            // Steps with λ_max Δt ≈ 10 damp the stiff Crank–Nicolson modes, so
            // the smooth modes set the observed order.
            let (coarse, fine) = (error(0.0025), error(0.00125));
            let ratio = coarse / fine;
            let expected = f64::from(1 << order);
            assert!(
                ratio > 0.9 * expected && ratio < 1.1 * expected,
                "{discretization:?} {scheme:?}: {coarse:e} -> {fine:e}"
            );
        }
    }
}

const T_WAVE: f64 = 0.5;

/// Standing wave `sin(πx/2) cos(πt/2)`: fixed at `x = 0`, free at `x = 1`.
#[derive(Clone)]
struct Standing;
/// Travelling wave `sin(2(x - t))` with moving Dirichlet and Robin data.
#[derive(Clone)]
struct Travelling;

impl HyperbolicPde<f64> for Standing {
    fn wave_speed(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((core::f64::consts::FRAC_PI_2 * x[0]).sin())
    }
    fn initial_velocity(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}
impl HyperbolicPde<f64> for Travelling {
    fn wave_speed(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((2.0 * x[0]).sin())
    }
    fn initial_velocity(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(-2.0 * (2.0 * x[0]).cos())
    }
}

fn wave_error<P: HyperbolicPde<f64> + Clone + Send + Sync + 'static>(
    discretization: Discretization,
    cells: usize,
    problem: &P,
    ends: &Ends<f64>,
    exact: impl Fn(f64) -> f64,
) -> f64 {
    let mesh = mesh(cells);
    let wave = Wave(problem.clone());
    let plan = discretization.plan(&mesh, &wave, ends).unwrap();
    let data = source(&wave);
    let (displacement, velocity) = wave_initial(problem);
    let hyperbolic = HyperbolicOptions::new()
        .with_initial_step(1e-3)
        .with_error_tolerance(ErrorTolerance::from_tolerance(
            Tolerance::new(1e-11, 1e-11).unwrap(),
            NormSelection::RootMeanSquare,
        ));
    let solution = with_plan!(&plan, |p| HyperbolicProblem::new(p, &data, T_WAVE)
        .unwrap()
        .solve_with_options(&displacement, &velocity, &hyperbolic))
    .unwrap();
    assert!(solution.report().is_success());
    plan.full(solution.values())
        .unwrap()
        .iter()
        .zip(&discretization.coordinates(&mesh))
        .map(|(u, &x)| (u - exact(x)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn wave_with_free_and_robin_ends_converges_in_space() {
    let free = Ends::new(
        End::dirichlet(value(|_| Ok(0.0))),
        End::natural(value(|_| Ok(0.0))),
    );
    let standing = |x: f64| {
        (core::f64::consts::FRAC_PI_2 * x).sin() * (core::f64::consts::FRAC_PI_2 * T_WAVE).cos()
    };
    let moving_ends = Ends::new(
        End::dirichlet(moving(|_, t: f64| Ok((-2.0 * t).sin()))),
        End::robin(
            1.0,
            moving(|_, t: f64| {
                let phase = 2.0 * (1.0 - t);
                Ok(2.0 * phase.cos() + phase.sin())
            }),
        ),
    );
    let travelling = |x: f64| (2.0 * (x - T_WAVE)).sin();
    for discretization in DISCRETIZATIONS {
        let cases: [(f64, f64, &str); 2] = [
            (
                wave_error(discretization, 16, &Standing, &free, standing),
                wave_error(discretization, 32, &Standing, &free, standing),
                "free end",
            ),
            (
                wave_error(discretization, 16, &Travelling, &moving_ends, travelling),
                wave_error(discretization, 32, &Travelling, &moving_ends, travelling),
                "moving Dirichlet and Robin ends",
            ),
        ];
        for (coarse, fine, name) in cases {
            assert!(fine < 2e-3, "{discretization:?} {name}: {fine:e}");
            assert!(
                coarse / fine > 3.5,
                "{discretization:?} {name}: {coarse:e} -> {fine:e}"
            );
        }
    }
}

#[test]
fn lumped_mass_fem_through_the_adapter_takes_dopri5_real_axis_limited_steps() {
    const CELLS: usize = 32;
    const T: f64 = 0.05;
    let zero = Ends::new(
        End::dirichlet(value(|_| Ok(0.0))),
        End::dirichlet(value(|_| Ok(0.0))),
    );
    let plan = Discretization::Fem
        .plan(&mesh(CELLS), &support::Heat(Decay), &zero)
        .unwrap();
    let Plan1d::Fem(fem) = &plan else {
        unreachable!()
    };
    let source = FieldData::zero();
    let semi = SemidiscreteSystem::new(fem, &source).unwrap();
    let solver = DormandPrince5::<f64>::new();
    let adapter = semi
        .method_of_lines(MethodOfLinesOrder::First)
        .unwrap()
        .with_step_limit(CflPolicy::Automatic, solver.linear_stability())
        .unwrap();
    let mut work = adapter.workspace().unwrap();
    let system = adapter.bind_with_fields(&semi, &semi, &mut work).unwrap();
    // The smooth mode plus a small highest-frequency perturbation, which an
    // unstable step would amplify.
    let initial = semi
        .initial_state(&FieldData::time_invariant(|p: FieldPoint<f64, 1>| {
            let node = (p.point[0] * CELLS as f64).round();
            Ok((core::f64::consts::PI * p.point[0]).sin()
                + 1e-3 * if node % 2.0 == 0.0 { 1.0 } else { -1.0 })
        }))
        .unwrap();
    let mut state = vec![0.0; initial.len()];
    adapter.encode_into(&initial, &mut state).unwrap();
    let options = OdeOptions::new()
        .with_error_tolerance(
            adapter
                .error_tolerance(
                    &ErrorTolerance::scalar(1e-3, 1e-3, NormSelection::Infinity).unwrap(),
                )
                .unwrap(),
        )
        .with_initial_step(1.0);
    let ode = (solver, options);
    let problem = OdeProblem::new(&system, T);
    let mut state = OdeState::new(0.0, Vector::from_vec(state));
    let raw = ode
        .0
        .advance(
            &problem,
            &ode.1,
            &mut state,
            &mut (),
            &mut ode.0.workspace(&problem, &ode.1).unwrap(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    let report = PdeSolveReport::try_from(&raw).unwrap();
    assert!(report.is_success(), "{:?}", report.termination());
    // Lumped P1 mass gives M⁻¹K = h⁻² tridiag(-1, 2, -1) on the free nodes.
    let h = 1.0 / CELLS as f64;
    let mode = |k: usize| {
        4.0 / (h * h)
            * (k as f64 * core::f64::consts::PI / (2.0 * CELLS as f64))
                .sin()
                .powi(2)
    };
    let real_axis_steps = T * mode(CELLS - 1) / 3.3066;
    let accepted = report.diagnostics().num_accepted_steps() as f64;
    assert!(
        accepted >= real_axis_steps && accepted <= 1.15 * real_axis_steps,
        "{accepted} accepted steps against {real_axis_steps} real-axis-limited steps"
    );
    let decay = (-mode(1) * T).exp();
    // The bound fields reconstruct the full nodal field, Dirichlet ends included.
    assert_eq!(system.field_len(), CELLS + 1);
    let mut field = vec![f64::NAN; system.field_len()];
    system
        .reconstruct_into(state.time(), state.values(), &mut field)
        .unwrap();
    assert_eq!(report.final_time(), state.time());
    for (node, &u) in field.iter().enumerate() {
        let x = node as f64 * h;
        assert!(
            (u - decay * (core::f64::consts::PI * x).sin()).abs() < 1e-3,
            "x = {x}: {u}"
        );
    }
}
