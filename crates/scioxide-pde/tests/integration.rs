//! Integration tests for scioxide-pde.
//!
//! Tests cover: elliptic (Poisson), parabolic (heat), hyperbolic (wave),
//! boundary conditions, grid construction, the FDM/FEM/FVM plans, error
//! paths, and f32 support.

#![allow(clippy::unwrap_used)]

#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use std::f64::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mesh_fixture::{endpoint, nodes, periodic};
use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, Interval, LinearOperator, NormSelection,
    PartialResultTrust, ResidualTolerance, SciOxideError, SciOxideErrorView, Tolerance,
};
use scioxide_linalg::{GmresSolver, IterativeLinearSolver, Preconditioning};
use scioxide_pde::{
    BoundaryCondition, BoundaryData, EllipticGauge, EllipticOptions, EllipticProblem, FieldData,
    HyperbolicOptions, HyperbolicProblem, ParabolicOptions, ParabolicProblem, ParabolicScheme,
    PdeOutputSchedule, PdeSolution, PdeSolveReport, PdeTermination, SpatialDiscretization,
    SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan,
};
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CoordinateAxis, SegmentMesh, StructuredGrid,
};
use support::Discretization::{Fdm, Fem, Fvm};
use support::{
    DISCRETIZATIONS, Discretization, EllipticPde, End, Ends, Heat, HyperbolicPde, ParabolicPde,
    Steady, TestScalar, Wave, homogeneous_dirichlet, initial, max_abs_error, moving, source, value,
    wave_initial,
};

/// Assert that `result` failed with an error whose view matches `pattern`.
macro_rules! assert_error {
    ($result:expr, $pattern:pat $(if $guard:expr)?) => {{
        let result = $result;
        let error = result.as_ref().err();
        assert!(
            error.is_some_and(|error| matches!(error.view(), $pattern $(if $guard)?)),
            "unexpected outcome: {:?}",
            error.map(ToString::to_string)
        );
    }};
}

// ============================================================================
// Test PDE problems
// ============================================================================

/// Poisson 1D: -u'' = pi^2 sin(pi x), u(0) = u(1) = 0.
/// Exact solution: u(x) = sin(pi x).
#[derive(Clone)]
struct Poisson1D;

impl<S: TestScalar> EllipticPde<S> for Poisson1D {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
    fn source(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::PI * S::PI * (S::PI * x[0]).sin())
    }
}

#[derive(Clone)]
struct CountingElliptic {
    calls: Arc<AtomicUsize>,
}

impl EllipticPde<f64> for CountingElliptic {
    fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0)
    }

    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(0.0)
    }

    fn source(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(0.0)
    }
}

#[derive(Clone)]
struct DomainFailingElliptic;

impl EllipticPde<f64> for DomainFailingElliptic {
    fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Err(SciOxideError::domain_error(
            scioxide_core::argument!("position"),
            scioxide_core::DomainReason::OutsideDomain,
            "test domain failure",
        ))
    }

    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn source(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
}

/// Variable-coefficient elliptic: -d/dx(k(x) du/dx) + c(x) u = f(x).
/// k(x) = 1 + x, c(x) = 1, f(x) chosen so u(x) = sin(pi x).
#[derive(Clone)]
struct VariableCoeffElliptic;

impl EllipticPde<f64> for VariableCoeffElliptic {
    fn diffusion(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0 + x[0])
    }
    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        // -d/dx((1+x) u') + u = -pi cos(pi x) + (1+x) pi^2 sin(pi x) + sin(pi x)
        let sin_val = (PI * x[0]).sin();
        let cos_val = (PI * x[0]).cos();
        Ok(-PI * cos_val + (1.0 + x[0]) * PI * PI * sin_val + sin_val)
    }
}

/// Homogeneous elliptic problem whose exact solutions are linear functions.
#[derive(Clone)]
struct ConstantCoefficientProblem<S> {
    scale: S,
}

impl<S: TestScalar> EllipticPde<S> for ConstantCoefficientProblem<S> {
    fn diffusion(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(self.scale)
    }

    fn reaction(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }

    fn source(&self, _x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}

const LAPLACIAN: ConstantCoefficientProblem<f64> = ConstantCoefficientProblem { scale: 1.0 };

/// Poisson with reaction: -u'' + u = f(x), u(0) = u(1) = 0.
/// Choose f so u(x) = sin(pi x): f = (1 + pi^2) sin(pi x).
#[derive(Clone)]
struct PoissonWithReaction;

impl EllipticPde<f64> for PoissonWithReaction {
    fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((1.0 + PI * PI) * (PI * x[0]).sin())
    }
}

/// Compatible pure-Neumann problem with an exact zero-mean solution.
///
/// `-u'' = sin(2*pi*x)` and outward fluxes are `-1/(2*pi)` on the left
/// and `1/(2*pi)` on the right. The exact solution is
/// `sin(2*pi*x)/(4*pi^2)`.
#[derive(Clone)]
struct CompatiblePureNeumann;

impl EllipticPde<f64> for CompatiblePureNeumann {
    fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }

    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn source(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((2.0 * PI * x[0]).sin())
    }
}

#[derive(Clone)]
struct ConstantReactionSource {
    reaction: f64,
    source: f64,
}

impl EllipticPde<f64> for ConstantReactionSource {
    fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }

    fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.reaction)
    }

    fn source(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.source)
    }
}

/// Heat equation 1D: u_t = u_xx, u(x,0) = sin(pi x), u(0,t) = u(1,t) = 0.
/// Exact: u(x,t) = exp(-pi^2 t) sin(pi x).
#[derive(Clone)]
struct HeatEquation;

impl ParabolicPde<f64> for HeatEquation {
    fn diffusion(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * x[0]).sin())
    }
}

/// Heat eigenmode `exp(-mu^2 t) * mode(mu x + phase)` with unit diffusion.
#[derive(Clone)]
struct HeatMode {
    mu: f64,
    phase: f64,
}

impl HeatMode {
    fn exact(&self, x: f64, t: f64) -> f64 {
        (-self.mu * self.mu * t).exp() * (self.mu * x + self.phase).sin()
    }
}

impl ParabolicPde<f64> for HeatMode {
    fn diffusion(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn source(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.exact(x[0], 0.0))
    }
}

#[derive(Clone)]
struct CountingParabolic {
    calls: Arc<AtomicUsize>,
}

impl ParabolicPde<f64> for CountingParabolic {
    fn diffusion(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0)
    }

    fn source(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(0.0)
    }

    fn initial_condition(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(0.0)
    }
}

/// Wave equation 1D: u_tt = c^2 u_xx, u(x,0) = sin(pi x), v(x,0) = v0(x).
/// With `drift = 0` the exact solution is u(x,t) = cos(c pi t) sin(pi x); a
/// nonzero `drift` adds the linear wave `drift * t * (1 - 2x)`.
#[derive(Clone)]
struct WaveEquation {
    wave_speed: f64,
    drift: f64,
}

const WAVE: WaveEquation = WaveEquation {
    wave_speed: 1.0,
    drift: 0.0,
};

impl WaveEquation {
    fn exact(&self, x: f64, t: f64) -> f64 {
        (self.wave_speed * PI * t).cos() * (PI * x).sin() + self.drift * t * (1.0 - 2.0 * x)
    }
}

impl HyperbolicPde<f64> for WaveEquation {
    fn wave_speed(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.wave_speed)
    }
    fn source(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((PI * x[0]).sin())
    }
    fn initial_velocity(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.drift * (1.0 - 2.0 * x[0]))
    }
}

/// Iterative linear solver that returns a usable-looking iterate without convergence.
struct NonConvergingLinearSolver;

impl IterativeLinearSolver<f64> for NonConvergingLinearSolver {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "NonConvergingLinearSolver", 1)
    }

    fn solve_into(
        &self,
        _a: &dyn LinearOperator<f64>,
        b: scioxide_linalg::VectorView<'_, f64>,
        _preconditioning: Preconditioning<'_, f64>,
        mut x: scioxide_linalg::VectorViewMut<'_, f64>,
        _workspace: &mut scioxide_linalg::IterativeWorkspace<f64>,
        _context: scioxide_linalg::LinalgExecutionContext<'_, '_>,
    ) -> Result<scioxide_linalg::IterativeSolveReport<f64>, SciOxideError> {
        b.copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (x).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(scioxide_linalg::IterativeSolveReport::new(
            scioxide_linalg::LinearSolveTermination::IterationLimit,
            7,
            Some(scioxide_core::ResidualReport::unscaled(
                0.25,
                scioxide_core::NormSelection::L2,
            )?),
            scioxide_core::PartialResultTrust::ResidualChecked,
        ))
    }
}

/// Iterative linear solver that reports a structural input failure.
struct StructurallyFailingLinearSolver;

impl IterativeLinearSolver<f64> for StructurallyFailingLinearSolver {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "StructurallyFailingLinearSolver", 1)
    }

    fn solve_into(
        &self,
        _a: &dyn LinearOperator<f64>,
        _b: scioxide_linalg::VectorView<'_, f64>,
        _preconditioning: Preconditioning<'_, f64>,
        _x: scioxide_linalg::VectorViewMut<'_, f64>,
        _workspace: &mut scioxide_linalg::IterativeWorkspace<f64>,
        _context: scioxide_linalg::LinalgExecutionContext<'_, '_>,
    ) -> Result<scioxide_linalg::IterativeSolveReport<f64>, SciOxideError> {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(2),
            scioxide_core::Dimensions::one(1),
        ))
    }
}

/// Unit diffusion sampled directly by a structured FDM plan.
struct UnitDiffusion;

impl StructuredFdmCoefficients<f64, f64, 1> for UnitDiffusion {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}

impl scioxide_pde::FvmCoefficients<f64, f64, 1> for UnitDiffusion {
    fn diffusion(&self, _: [f64; 1]) -> Result<[[f64; 1]; 1], SciOxideError> {
        Ok([[1.0]])
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn uniform(nodes: usize) -> SegmentMesh<f64> {
    mesh_fixture::uniform(0.0, 1.0, nodes)
}

fn uniform_f32(nodes: usize) -> SegmentMesh<f32> {
    mesh_fixture::uniform(0.0_f32, 1.0, nodes)
}

/// Bounded segment mesh over one coordinate axis.
fn bounded(axis: CoordinateAxis<f64>) -> Result<SegmentMesh<f64>, SciOxideError> {
    SegmentMesh::new(StructuredGrid::new([axis], [AxisTopology::Bounded])?)
}

fn constant<S: TestScalar>(data: S) -> BoundaryData<S, S, 1> {
    value(move |_| Ok(data))
}

fn dirichlet<S: TestScalar>(left: S, right: S) -> Ends<S> {
    Ends::new(
        End::dirichlet(constant(left)),
        End::dirichlet(constant(right)),
    )
}

/// Time-varying Dirichlet data `u(0,t) = t`, `u(1,t) = -t`.
fn time_varying_dirichlet() -> Ends<f64> {
    Ends::new(
        End::dirichlet(moving(|_, t| Ok(t))),
        End::dirichlet(moving(|_, t: f64| Ok(-t))),
    )
}

fn pure_neumann(left: f64, right: f64) -> Ends<f64> {
    Ends::new(End::natural(constant(left)), End::natural(constant(right)))
}

/// Solve `problem` with an [`EllipticProblem`], through the owned form or, with an
/// injected solver, `solve_into`. Returns the field in coordinate order and the report.
fn steady<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
    options: EllipticOptions<S>,
    solver: Option<Box<dyn IterativeLinearSolver<S> + Send>>,
) -> Result<(Vec<S>, PdeSolveReport<S>), SciOxideError> {
    let problem = Steady(problem.clone());
    let plan = discretization.plan(mesh, &problem, ends)?;
    let source = source(&problem);
    let elliptic = options;
    let (values, report) = match solver {
        None => with_plan!(&plan, |p| EllipticProblem::new(p, &source)
            .solve_with_options(&elliptic))?
        .into_parts(),
        Some(solver) => {
            let mut values = vec![S::zero_value(); with_plan!(&plan, |p| p.unknowns())];
            let mut workspace = scioxide_pde::EllipticWorkspace::new().with_linear_solver(solver);
            let report = with_plan!(&plan, |p| EllipticProblem::new(p, &source).solve_into(
                &elliptic,
                &mut values,
                &mut workspace,
                ExecutionContext::sequential_native()
            ))?;
            (values, report)
        }
    };
    Ok((plan.full(&values)?, report))
}

/// Integrate a heat problem with a [`ParabolicProblem`] emitting every accepted
/// step. Returns the solution and every emitted field in coordinate order.
fn heat<P: ParabolicPde<f64> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    mesh: &SegmentMesh<f64>,
    problem: &P,
    ends: &Ends<f64>,
    options: ParabolicOptions<f64>,
    end: f64,
) -> Result<(PdeSolution<f64>, Vec<Vec<f64>>), SciOxideError> {
    let heat = Heat(problem.clone());
    let plan = discretization.plan(mesh, &heat, ends)?;
    let source = source(&heat);
    let parabolic = options.with_output_schedule(PdeOutputSchedule::TimeSteps);
    let solution = with_plan!(&plan, |p| ParabolicProblem::new(p, &source, end)?
        .solve_with_options(&initial(problem), &parabolic))?;
    let fields = solution
        .trajectory()
        .states()
        .iter()
        .map(|state| plan.full(state))
        .collect::<Result<_, _>>()?;
    Ok((solution, fields))
}

/// Hyperbolic options with root-mean-square local error control `tolerance`.
fn wave_options(initial_step: f64, tolerance: Tolerance<f64>) -> HyperbolicOptions<f64> {
    HyperbolicOptions::new()
        .with_initial_step(initial_step)
        .with_error_tolerance(ErrorTolerance::from_tolerance(
            tolerance,
            NormSelection::RootMeanSquare,
        ))
}

/// Integrate a wave problem with a [`HyperbolicProblem`] emitting every accepted
/// step. Returns the solution and every emitted field in coordinate order.
fn wave(
    discretization: Discretization,
    mesh: &SegmentMesh<f64>,
    problem: &WaveEquation,
    ends: &Ends<f64>,
    options: HyperbolicOptions<f64>,
    end: f64,
) -> Result<(PdeSolution<f64>, Vec<Vec<f64>>), SciOxideError> {
    let wave = Wave(problem.clone());
    let plan = discretization.plan(mesh, &wave, ends)?;
    let source = source(&wave);
    let (displacement, velocity) = wave_initial(problem);
    let hyperbolic = options.with_output_schedule(PdeOutputSchedule::TimeSteps);
    let solution = with_plan!(&plan, |p| HyperbolicProblem::new(p, &source, end)?
        .solve_with_options(&displacement, &velocity, &hyperbolic))?;
    let fields = solution
        .trajectory()
        .states()
        .iter()
        .map(|state| plan.full(state))
        .collect::<Result<_, _>>()?;
    Ok((solution, fields))
}

fn max_error(
    discretization: Discretization,
    mesh: &SegmentMesh<f64>,
    values: &[f64],
    exact: impl Fn(f64) -> f64,
) -> f64 {
    max_abs_error(values, &discretization.coordinates(mesh), exact)
}

/// Last emitted time and field of a transient run.
fn last<'a>(solution: &PdeSolution<f64>, fields: &'a [Vec<f64>]) -> (f64, &'a [f64]) {
    (
        *solution.trajectory().times().last().unwrap(),
        fields.last().unwrap(),
    )
}

fn assert_converged_elliptic_reporting(report: &PdeSolveReport<f64>) {
    let stats = report.diagnostics();
    assert_eq!(report.termination(), PdeTermination::Completed);
    assert!(report.is_success());
    assert!(report.termination().is_success());
    assert_eq!(stats.num_assemblies(), 1);
    assert_eq!(stats.num_linear_solves(), 1);
    assert_eq!(stats.num_time_steps(), 0);
    assert_eq!(stats.num_accepted_steps(), 0);
    assert_eq!(stats.num_rejected_steps(), 0);
    assert_eq!(stats.num_rhs_evaluations(), 0);
    assert_eq!(stats.num_jacobian_evaluations(), 0);
    assert_eq!(stats.num_nonlinear_iterations(), 0);
    assert!(stats.residual_norm().is_some_and(f64::is_finite));
    assert_eq!(stats.termination(), PdeTermination::Completed);
    assert!(stats.method().is_some_and(|method| !method.is_empty()));
}

fn assert_completed_time_reporting(solution: &PdeSolution<f64>) {
    let report = solution.report();
    assert_eq!(report.termination(), PdeTermination::Completed);
    assert!(report.is_success());
    assert!(report.termination().is_success());
    let stats = report.diagnostics();
    assert!(stats.num_time_steps() > 0);
    assert_eq!(
        stats.num_time_steps(),
        stats.num_accepted_steps() + stats.num_rejected_steps()
    );
    assert_eq!(
        solution.trajectory().times().len(),
        stats.num_accepted_steps() + 1
    );
    assert_eq!(stats.num_assemblies(), 0);
    assert_eq!(stats.num_linear_solves(), 0);
    assert!(stats.num_rhs_evaluations() > 0);
    // Automatic explicit step limits evaluate the constant Jacobian once.
    assert_eq!(stats.num_jacobian_evaluations(), 1);
    assert_eq!(stats.num_nonlinear_iterations(), 0);
    assert!(
        stats
            .last_step_error_estimate()
            .is_some_and(|estimate| estimate.is_finite() && estimate <= 1.0)
    );
    assert!(stats.method().is_some_and(|method| !method.is_empty()));
}

fn explicit_parabolic_options(initial_step: f64) -> ParabolicOptions<f64> {
    ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(initial_step)
}

fn gmres(restart: Option<usize>) -> Option<Box<dyn IterativeLinearSolver<f64> + Send>> {
    let solver = GmresSolver::new(
        ResidualTolerance::new(1e-12, 0.0).unwrap(),
        Budget::try_from(5000).unwrap(),
    );
    Some(Box::new(match restart {
        Some(restart) => solver.with_restart(Budget::try_from(restart).unwrap()),
        None => solver,
    }))
}

// ============================================================================
// Elliptic tests
// ============================================================================

#[test]
fn test_elliptic_poisson_dirichlet() {
    // Second-order schemes on 101 points (h = 0.01) give O(h^2) ~ 1e-4.
    let mesh = uniform(101);
    for discretization in DISCRETIZATIONS {
        let (values, report) = steady(
            discretization,
            &mesh,
            &Poisson1D,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        assert_converged_elliptic_reporting(&report);
        // FVM has N-1 cells for N nodes.
        let unknowns = if discretization == Fvm { 100 } else { 101 };
        assert_eq!(values.len(), unknowns);
        let max_err = max_error(discretization, &mesh, &values, |x| (PI * x).sin());
        assert!(
            max_err < 1e-3,
            "{} Poisson max error too large: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_elliptic_poisson_reaction_term() {
    let mesh = uniform(101);
    for discretization in [Fdm, Fem] {
        let (values, _) = steady(
            discretization,
            &mesh,
            &PoissonWithReaction,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        let max_err = max_error(discretization, &mesh, &values, |x| (PI * x).sin());
        assert!(
            max_err < 1e-3,
            "{} Poisson+reaction max error too large: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_elliptic_variable_coefficient() {
    // Tight GMRES on 201 points gives O(h^2) ~ 2.5e-5.
    let mesh = uniform(201);
    for discretization in [Fdm, Fem] {
        let (values, _) = steady(
            discretization,
            &mesh,
            &VariableCoeffElliptic,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            gmres(Some(201)),
        )
        .unwrap();
        let max_err = max_error(discretization, &mesh, &values, |x| (PI * x).sin());
        assert!(
            max_err < 5e-3,
            "{} variable coefficient max error too large: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_elliptic_neumann_dirichlet() {
    // -u'' = 0 on [0, 1], outward flux du/dn(0) = -1, u(1) = 1. Exact: u(x) = x.
    let ends = Ends::new(End::natural(constant(-1.0)), End::dirichlet(constant(1.0)));
    let mesh = uniform(51);
    for discretization in DISCRETIZATIONS {
        let (values, _) = steady(
            discretization,
            &mesh,
            &LAPLACIAN,
            &ends,
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        let max_err = max_error(discretization, &mesh, &values, |x| x);
        assert!(
            max_err < 1e-6,
            "{} Neumann+Dirichlet max error too large: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_elliptic_robin_bc() {
    // -u'' = 0, Robin at left: du/dn + u = 1, Dirichlet at right: u(1) = 1.
    // Exact: u(x) = 1.
    let ends = Ends::new(
        End::robin(1.0, constant(1.0)),
        End::dirichlet(constant(1.0)),
    );
    let mesh = uniform(51);
    for discretization in DISCRETIZATIONS {
        let (values, _) = steady(
            discretization,
            &mesh,
            &LAPLACIAN,
            &ends,
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        let max_err = max_error(discretization, &mesh, &values, |_| 1.0);
        assert!(
            max_err < 1e-6,
            "{} Robin BC max error too large: {max_err}",
            discretization.name()
        );
    }
}

// ============================================================================
// Parabolic tests
// ============================================================================

#[test]
fn test_parabolic_heat_equation() {
    let opts = explicit_parabolic_options(0.001);
    let mesh = uniform(51);
    for discretization in DISCRETIZATIONS {
        let (sol, fields) = heat(
            discretization,
            &mesh,
            &HeatEquation,
            &homogeneous_dirichlet(),
            opts.clone(),
            0.05,
        )
        .unwrap();

        assert_completed_time_reporting(&sol);
        assert!(fields.len() > 1, "should have multiple time steps");
        assert!(sol.report().diagnostics().num_time_steps() > 1);
        let (t_final, values) = last(&sol, &fields);
        if discretization == Fdm {
            assert_eq!(values.len(), 51);
        }

        let max_err = max_error(discretization, &mesh, values, |x| {
            (-PI * PI * t_final).exp() * (PI * x).sin()
        });
        assert!(
            max_err < 5e-3,
            "{} heat equation max error at t={t_final}: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_parabolic_with_source_term() {
    // u_t = u_xx + pi^2 sin(pi x), u(x,0) = 0, u(0,t) = u(1,t) = 0.
    // Exact at steady state: u(x) -> sin(pi x).
    #[derive(Clone)]
    struct HeatWithSource;
    impl ParabolicPde<f64> for HeatWithSource {
        fn diffusion(&self, _x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn source(&self, x: &[f64], _t: f64) -> Result<f64, SciOxideError> {
            Ok(PI * PI * (PI * x[0]).sin())
        }
        fn initial_condition(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
    }

    let opts = explicit_parabolic_options(0.005);
    let mesh = uniform(51);
    let (sol, fields) = heat(
        Fdm,
        &mesh,
        &HeatWithSource,
        &homogeneous_dirichlet(),
        opts,
        0.5,
    )
    .unwrap();

    // At t=0.5, should be close to steady state
    let max_err = max_error(Fdm, &mesh, last(&sol, &fields).1, |x| (PI * x).sin());
    assert!(max_err < 0.02, "Heat+source max error too large: {max_err}");
}

#[test]
fn test_parabolic_natural_and_robin_boundaries_are_supported() {
    // Formerly unsupported: FDM Neumann u_x(0) = 0 with u(1) = 0 has the mode
    // cos(pi x / 2); FEM Robin -u_x(0) + u(0) = 0 with u(1) = 0 has the mode
    // sin(mu (1 - x)) with tan(mu) = -mu.
    let mu = 2.028_757_838_110_434;
    let cases = [
        (
            Fdm,
            End::natural(constant(0.0)),
            HeatMode {
                mu: PI / 2.0,
                phase: PI / 2.0,
            },
        ),
        (
            Fem,
            End::robin(1.0, constant(0.0)),
            HeatMode { mu: -mu, phase: mu },
        ),
    ];
    let options = explicit_parabolic_options(0.01);
    let mesh = uniform(21);
    for (discretization, left, problem) in cases {
        let ends = Ends::new(left, End::dirichlet(constant(0.0)));
        let (sol, fields) =
            heat(discretization, &mesh, &problem, &ends, options.clone(), 0.1).unwrap();
        assert_completed_time_reporting(&sol);
        let (t_final, values) = last(&sol, &fields);
        assert_eq!(t_final, 0.1);
        let max_err = max_error(discretization, &mesh, values, |x| problem.exact(x, t_final));
        assert!(
            max_err < 1e-3,
            "{} natural/Robin heat max error: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_parabolic_periodic_topology() {
    // Periodic heat is now supported by FDM: the mode sin(2 pi x) decays.
    let problem = HeatMode {
        mu: 2.0 * PI,
        phase: 0.0,
    };
    let mesh = periodic(&uniform(21));
    let options = explicit_parabolic_options(0.01);
    let (sol, fields) = heat(
        Fdm,
        &mesh,
        &problem,
        &Ends::periodic(),
        options.clone(),
        0.1,
    )
    .unwrap();
    assert_completed_time_reporting(&sol);
    let (t_final, values) = last(&sol, &fields);
    assert_eq!(values.len(), 21);
    assert_eq!(values[0], values[20]);
    let max_err = max_error(Fdm, &mesh, values, |x| problem.exact(x, t_final));
    assert!(max_err < 1e-3, "FDM periodic heat max error: {max_err}");

    // Structured finite volumes remain bounded-only.
    assert_error!(
        heat(Fvm, &mesh, &problem, &Ends::periodic(), options, 0.1),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn test_parabolic_time_varying_dirichlet_is_supported() {
    for (discretization, final_time, step) in
        [(Fdm, 0.1, 0.01), (Fem, 0.1, 0.01), (Fvm, 0.01, 0.001)]
    {
        let options = explicit_parabolic_options(step);
        let (sol, _) = heat(
            discretization,
            &uniform(21),
            &HeatEquation,
            &time_varying_dirichlet(),
            options,
            final_time,
        )
        .unwrap();
        assert!(sol.report().is_success(), "{}", discretization.name());
    }
}

#[test]
fn test_fvm_parabolic_right_dirichlet_evaluates_right_boundary() {
    let ends = Ends::new(
        End::dirichlet(constant(0.0)),
        End::dirichlet(value(|x: f64| {
            if (x - 1.0).abs() > 1e-12 {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("input"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "right boundary function called with non-right x",
                ));
            }
            Ok(0.0)
        })),
    );
    let options = explicit_parabolic_options(0.001);
    let result = heat(Fvm, &uniform(21), &HeatEquation, &ends, options, 0.01);
    assert!(
        result.is_ok(),
        "right boundary function should only see x=1"
    );
}

#[test]
fn test_parabolic_natural_boundary_evaluates_problem_and_boundary_callbacks() {
    // Formerly rejected before callbacks; Neumann data is now integrated.
    let problem_calls = Arc::new(AtomicUsize::new(0));
    let boundary_calls = Arc::new(AtomicUsize::new(0));
    let problem = CountingParabolic {
        calls: Arc::clone(&problem_calls),
    };
    let counted = |calls: &Arc<AtomicUsize>| -> BoundaryData<f64, f64, 1> {
        let calls = Arc::clone(calls);
        value(move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(0.0)
        })
    };
    let ends = Ends::new(
        End::natural(counted(&boundary_calls)),
        End::dirichlet(counted(&boundary_calls)),
    );
    let options = explicit_parabolic_options(0.01);
    let (sol, fields) = heat(Fdm, &uniform(5), &problem, &ends, options, 0.1).unwrap();
    assert!(sol.report().is_success());
    assert!(last(&sol, &fields).1.iter().all(|&u| u == 0.0));
    assert!(problem_calls.load(Ordering::Relaxed) > 0);
    assert!(boundary_calls.load(Ordering::Relaxed) > 0);
}

// ============================================================================
// Hyperbolic tests
// ============================================================================

#[test]
fn test_hyperbolic_wave_equation() {
    let mesh = uniform(51);
    for discretization in DISCRETIZATIONS {
        let options = wave_options(0.001, Tolerance::from_sqrt_epsilon());
        let (sol, fields) = wave(
            discretization,
            &mesh,
            &WAVE,
            &homogeneous_dirichlet(),
            options,
            0.1,
        )
        .unwrap();

        assert_completed_time_reporting(&sol);
        assert!(fields.len() > 1);

        let (t_final, values) = last(&sol, &fields);
        let max_err = max_error(discretization, &mesh, values, |x| WAVE.exact(x, t_final));
        assert!(
            max_err < 0.01,
            "{} wave equation max error at t={t_final}: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_hyperbolic_stability() {
    // For the wave equation with Dirichlet BCs, verify the solution stays bounded.
    // Exact: u(x,t) = cos(pi t) sin(pi x), so max|u| = 1 for all time.
    let options = wave_options(0.001, Tolerance::new(1e-10, 1e-10).unwrap());
    let mesh = uniform(101);
    let (sol, fields) = wave(Fdm, &mesh, &WAVE, &homogeneous_dirichlet(), options, 0.5).unwrap();

    // The analytical maximum is 1.0 (from sin(pi x)).
    let max_disp = fields
        .iter()
        .flatten()
        .map(|&u| u.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_disp < 1.1,
        "Maximum displacement exceeds initial amplitude: {max_disp}"
    );

    // At the midpoint time index the solution approximately matches the exact one.
    let mid = fields.len() / 2;
    let t_mid = sol.trajectory().times()[mid];
    let max_mid_err = max_error(Fdm, &mesh, &fields[mid], |x| WAVE.exact(x, t_mid));
    assert!(
        max_mid_err < 0.05,
        "Wave equation mid-time error too large at t={t_mid}: {max_mid_err}"
    );
}

#[test]
fn test_hyperbolic_time_varying_dirichlet_is_supported() {
    // Formerly unsupported. Boundary data u(0,t) = t, u(1,t) = -t with the
    // compatible initial velocity 1 - 2x gives cos(pi t) sin(pi x) + t (1 - 2x).
    let problem = WaveEquation {
        wave_speed: 1.0,
        drift: 1.0,
    };
    let mesh = uniform(21);
    for discretization in DISCRETIZATIONS {
        let options = wave_options(0.01, Tolerance::from_sqrt_epsilon());
        let (sol, fields) = wave(
            discretization,
            &mesh,
            &problem,
            &time_varying_dirichlet(),
            options,
            0.1,
        )
        .unwrap();
        assert_completed_time_reporting(&sol);
        let (t_final, values) = last(&sol, &fields);
        assert_eq!(t_final, 0.1);
        let max_err = max_error(discretization, &mesh, values, |x| problem.exact(x, t_final));
        assert!(
            max_err < 1e-3,
            "{} moving-boundary wave max error: {max_err}",
            discretization.name()
        );
    }
}

// ============================================================================
// Grid tests
// ============================================================================

#[test]
fn test_grid_uniform() {
    let grid = bounded(CoordinateAxis::uniform(0.0, 1.0, 11).unwrap()).unwrap();
    let axis = grid.grid().axis(0).unwrap();
    assert_eq!(
        Interval::new(axis.start(), axis.end()).unwrap(),
        Interval::new(0.0, 1.0).unwrap()
    );
    assert_eq!(grid.topology_sizes().vertices(), 11);

    let nodes = nodes(&grid);
    assert_eq!(nodes.len(), 11);
    assert!((nodes[0] - 0.0_f64).abs() < 1e-15);
    assert!((nodes[10] - 1.0_f64).abs() < 1e-14);
    assert!((axis.coordinate(1).unwrap() - axis.start() - 0.1_f64).abs() < 1e-14);
}

#[test]
fn test_grid_from_nodes_derives_interval() {
    let grid = bounded(CoordinateAxis::explicit(vec![0.0, 0.25, 0.5, 0.75, 1.0]).unwrap()).unwrap();
    let axis = grid.grid().axis(0).unwrap();
    assert_eq!(
        Interval::new(axis.start(), axis.end()).unwrap(),
        Interval::new(0.0, 1.0).unwrap()
    );
    assert_eq!(grid.topology_sizes().vertices(), 5);
}

#[test]
fn test_grid_rejects_too_few_nodes() {
    assert_error!(
        CoordinateAxis::uniform(0.0, 1.0, 1).and_then(bounded),
        SciOxideErrorView::InvalidInput { .. }
    );
    assert_error!(
        CoordinateAxis::explicit(vec![0.0]).and_then(bounded),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn test_grid_rejects_degenerate_interval() {
    assert_error!(
        CoordinateAxis::uniform(1.0_f64, 1.0, 10).and_then(bounded),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn test_grid_rejects_unordered_or_duplicate_nodes() {
    for nodes in [vec![0.0, 0.5, 0.3, 1.0], vec![0.0, 0.5, 0.5, 1.0]] {
        assert_error!(
            CoordinateAxis::explicit(nodes).and_then(bounded),
            SciOxideErrorView::InvalidInput { .. }
        );
    }
}

#[test]
fn test_grid_rejects_non_finite_nodes() {
    assert_error!(
        CoordinateAxis::explicit(vec![0.0, f64::NAN, 1.0]).and_then(bounded),
        SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    );
}

#[test]
fn uniform_grid_rejects_extreme_or_duplicate_derived_nodes() {
    let lower = 1.0_f64;
    let upper = f64::from_bits(lower.to_bits() + 1);
    let two_nodes = bounded(CoordinateAxis::uniform(lower, upper, 2).unwrap()).unwrap();
    assert_eq!(nodes(&two_nodes), &[lower, upper]);
    assert_error!(
        CoordinateAxis::uniform(lower, upper, 3).and_then(bounded),
        SciOxideErrorView::InvalidInput { .. }
    );
    // Every plan classifies the unrepresentable spacing of an extreme two-node
    // mesh as overflow.
    for mesh in [
        mesh_fixture::uniform(-f64::MAX, f64::MAX, 2),
        mesh_fixture::explicit(vec![-f64::MAX, f64::MAX]),
    ] {
        let plan = |d: Discretization| d.plan(&mesh, &Steady(LAPLACIAN), &homogeneous_dirichlet());
        for discretization in DISCRETIZATIONS {
            assert_error!(
                plan(discretization),
                SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            );
        }
    }
}

#[test]
fn plan_preflights_storage_against_spatial_limits() {
    let mesh = uniform(1001);
    let boundaries = homogeneous_dirichlet().structured(&mesh).unwrap();
    for limits in [
        SpatialLimits::new(1000, usize::MAX, usize::MAX),
        SpatialLimits::new(usize::MAX, 1000, usize::MAX),
        SpatialLimits::new(usize::MAX, usize::MAX, 1000),
    ] {
        assert_error!(
            StructuredFdmPlan::with_options(mesh.clone(), &UnitDiffusion, &boundaries, limits),
            SciOxideErrorView::InvalidInput { .. }
        );
    }
    let limits = SpatialLimits::new(1001, usize::MAX, usize::MAX);
    assert!(StructuredFdmPlan::with_options(mesh, &UnitDiffusion, &boundaries, limits).is_ok());
    // A compact uniform mesh with 2^50 nodes binds its two endpoint conditions
    // in constant work; every plan then rejects it during storage preflight.
    let huge = mesh_fixture::uniform(0.0, 1.0, 1 << 50);
    let boundaries = homogeneous_dirichlet().structured(&huge).unwrap();
    assert_error!(
        StructuredFdmPlan::with_options(
            huge.clone(),
            &UnitDiffusion,
            &boundaries,
            SpatialLimits::new(1 << 20, 1 << 24, 1 << 30)
        ),
        SciOxideErrorView::InvalidInput { .. }
    );
    assert_error!(
        scioxide_pde::StructuredFvmPlan::with_options(
            huge,
            &UnitDiffusion,
            &boundaries,
            SpatialLimits::new(1 << 20, 1 << 24, 1 << 30)
        ),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn core_interval_rejects_invalid_grid_domains() {
    assert_error!(
        Interval::new(1.0_f64, 0.0),
        SciOxideErrorView::InvalidInput { .. }
    );
    assert_error!(
        Interval::new(0.0, f64::NAN),
        SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    );
}

// ============================================================================
// Boundary condition tests
// ============================================================================

fn structured(
    grid: &SegmentMesh<f64>,
    conditions: Vec<BoundaryCondition<f64, f64, 1, BoundarySet<1>>>,
) -> Result<StructuredBoundaries<f64, f64, 1>, SciOxideError> {
    StructuredBoundaries::new(grid, 1, conditions)
}

#[test]
fn test_boundary_regions_left_and_right_validate() {
    let grid = uniform(10);
    let conditions = homogeneous_dirichlet().structured(&grid).unwrap();
    assert_eq!(conditions.conditions().len(), 2);
}

#[test]
fn test_periodic_boundary_uses_grid_topology() {
    let grid = periodic(&uniform(10));
    let conditions = Ends::<f64>::periodic().structured(&grid).unwrap();
    assert!(conditions.conditions().is_empty());
}

#[test]
fn test_boundary_conditions_reject_empty_topology() {
    assert_error!(
        structured(&uniform(10), Vec::new()),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn test_boundary_conditions_require_both_nonperiodic_endpoints() {
    let grid = uniform(10);
    for side in [AxisSide::Start, AxisSide::End] {
        assert_error!(
            structured(
                &grid,
                vec![BoundaryCondition::dirichlet(
                    endpoint(&grid, side),
                    BoundaryData::zero()
                )],
            ),
            SciOxideErrorView::InvalidInput { .. }
        );
    }
}

#[test]
fn two_interface_fvm_retains_one_cell_on_original_mesh() {
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = CountingElliptic {
        calls: Arc::clone(&calls),
    };
    let mesh = uniform(2);
    let (values, _) = steady(
        Fvm,
        &mesh,
        &problem,
        &homogeneous_dirichlet(),
        EllipticOptions::new(),
        None,
    )
    .unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(mesh.topology_sizes().vertices(), 2);
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn test_boundary_conditions_reject_mixed_periodic_topology() {
    let grid = periodic(&uniform(10));
    let result = BoundarySet::axis_side(&grid, 0, AxisSide::Start).and_then(|region| {
        structured(
            &grid,
            vec![BoundaryCondition::dirichlet(region, BoundaryData::zero())],
        )
    });
    assert_error!(result, SciOxideErrorView::InvalidInput { .. });
}

#[test]
fn test_boundary_condition_debug() {
    let grid = uniform(3);
    let bc = BoundaryCondition::<f64, f64, 1, _>::dirichlet(
        endpoint(&grid, AxisSide::Start),
        BoundaryData::zero(),
    );
    let debug_str = format!("{bc:?}");
    assert!(debug_str.contains("Dirichlet"));
    assert!(debug_str.contains("BoundarySet"));
}

#[test]
fn test_boundary_conditions_reject_duplicate_dirichlet_constraints() {
    let grid = uniform(11);
    let left = || endpoint(&grid, AxisSide::Start);
    assert_error!(
        structured(
            &grid,
            vec![
                BoundaryCondition::dirichlet(left(), BoundaryData::constant(1.0)),
                BoundaryCondition::dirichlet(left(), BoundaryData::constant(1.0)),
            ],
        ),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn test_boundary_conditions_reject_conflicting_kinds() {
    let grid = uniform(11);
    let left = || endpoint(&grid, AxisSide::Start);
    assert_error!(
        structured(
            &grid,
            vec![
                BoundaryCondition::dirichlet(left(), BoundaryData::constant(1.0)),
                BoundaryCondition::natural(left(), BoundaryData::zero()),
            ],
        ),
        SciOxideErrorView::InvalidInput { .. }
    );
}

#[test]
fn test_fem_interval_mesh_without_endpoint_conditions_fails_before_callbacks() {
    // Interval simplex meshes have no periodic identification, so a periodic
    // request leaves both endpoints without conditions.
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = CountingElliptic {
        calls: Arc::clone(&calls),
    };
    let result = steady(
        Fem,
        &periodic(&uniform(5)),
        &problem,
        &Ends::periodic(),
        EllipticOptions::new(),
        None,
    );
    assert_error!(result, SciOxideErrorView::InvalidInput { .. });
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

// ============================================================================
// Dirichlet lifting and constrained systems
// ============================================================================

#[test]
fn test_fem_elliptic_nonzero_dirichlet_lifting() {
    let mesh = uniform(21);
    let (values, _) = steady(
        Fem,
        &mesh,
        &LAPLACIAN,
        &dirichlet(1.0, 3.0),
        EllipticOptions::new(),
        None,
    )
    .unwrap();
    let max_err = max_error(Fem, &mesh, &values, |x| 1.0 + 2.0 * x);
    assert!(max_err < 1e-8, "nonzero Dirichlet error: {max_err}");
}

#[test]
fn test_fem_elliptic_strongly_scaled_nonzero_dirichlet() {
    let (values_f64, _) = steady(
        Fem,
        &uniform(3),
        &ConstantCoefficientProblem {
            scale: 2.0_f64.powi(60),
        },
        &dirichlet(2.0, 5.0),
        EllipticOptions::new(),
        None,
    )
    .unwrap();
    assert_eq!(values_f64, vec![2.0, 3.5, 5.0]);

    let (values_f32, _) = steady(
        Fem,
        &uniform_f32(3),
        &ConstantCoefficientProblem {
            scale: 2.0_f32.powi(30),
        },
        &dirichlet(2.0_f32, 5.0),
        EllipticOptions::new(),
        None,
    )
    .unwrap();
    assert_eq!(values_f32, vec![2.0, 3.5, 5.0]);
}

#[test]
fn test_fem_fully_constrained_grid_solves_unit_rows() {
    // Every unknown is a strongly imposed value, so the eliminated operator is
    // the identity and the direct solve reproduces the data.
    let (values, report) = steady(
        Fem,
        &uniform(2),
        &LAPLACIAN,
        &dirichlet(2.0, 5.0),
        EllipticOptions::new(),
        None,
    )
    .unwrap();
    assert_eq!(values, vec![2.0, 5.0]);
    assert_eq!(report.termination(), PdeTermination::Completed);
    assert_eq!(report.diagnostics().num_assemblies(), 1);
}

// ============================================================================
// Pure Neumann and nullspace tests
// ============================================================================

#[test]
fn test_compatible_pure_neumann_uses_weighted_mean_zero_gauge() {
    let derivative = 1.0 / (2.0 * PI);
    let ends = pure_neumann(-derivative, derivative);
    let mesh = uniform(51);
    for discretization in DISCRETIZATIONS {
        let name = discretization.name();
        let (values, report) = steady(
            discretization,
            &mesh,
            &CompatiblePureNeumann,
            &ends,
            EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(0.0)),
            None,
        )
        .unwrap();
        assert_eq!(report.termination(), PdeTermination::Completed, "{name}");
        assert!(report.diagnostics().nullspace().is_some(), "{name}");

        let max_err = max_error(discretization, &mesh, &values, |x| {
            (2.0 * PI * x).sin() / (4.0 * PI * PI)
        });
        assert!(max_err < 3e-3, "{name} max error: {max_err}");

        let coordinates = discretization.coordinates(&mesh);
        let weighted_mean = if discretization == Fvm {
            values.iter().sum::<f64>() / values.len() as f64
        } else {
            let mut integral = 0.0;
            let mut total_weight = 0.0;
            for (x, u) in coordinates.windows(2).zip(values.windows(2)) {
                let width = x[1] - x[0];
                integral += width * (u[0] + u[1]) / 2.0;
                total_weight += width;
            }
            integral / total_weight
        };
        assert!(weighted_mean.abs() < 1e-8, "{name} mean: {weighted_mean}");
    }
}

#[test]
fn test_incompatible_pure_neumann_is_a_domain_error() {
    let problem = ConstantReactionSource {
        reaction: 0.0,
        source: 1.0,
    };
    for discretization in DISCRETIZATIONS {
        let result = steady(
            discretization,
            &uniform(21),
            &problem,
            &pure_neumann(0.0, 0.0),
            EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(0.0)),
            None,
        );
        assert_error!(
            result,
            SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Undefined,
                ..
            }
        );
    }
}

#[test]
fn test_reaction_removes_pure_neumann_nullspace() {
    let problem = ConstantReactionSource {
        reaction: 1.0,
        source: 1.0,
    };
    let mesh = uniform(21);
    for discretization in DISCRETIZATIONS {
        let (values, _) = steady(
            discretization,
            &mesh,
            &problem,
            &pure_neumann(0.0, 0.0),
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        let max_err = max_error(discretization, &mesh, &values, |_| 1.0);
        assert!(
            max_err < 1e-6,
            "{} max error: {max_err}",
            discretization.name()
        );
    }
}

// ============================================================================
// Linear solver injection
// ============================================================================

#[test]
fn test_linear_solver_injection() {
    let mesh = uniform(51);
    for discretization in DISCRETIZATIONS {
        let (values, _) = steady(
            discretization,
            &mesh,
            &Poisson1D,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            gmres(None),
        )
        .unwrap();
        let max_err = max_error(discretization, &mesh, &values, |x| (PI * x).sin());
        assert!(
            max_err < 1e-3,
            "{} injected solver max error: {max_err}",
            discretization.name()
        );
    }
}

#[test]
fn test_elliptic_solvers_preserve_unconverged_linear_iterates() {
    for discretization in DISCRETIZATIONS {
        let (values, report) = steady(
            discretization,
            &uniform(11),
            &Poisson1D,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            Some(Box::new(NonConvergingLinearSolver)),
        )
        .unwrap();

        let stats = report.diagnostics();
        assert_eq!(report.termination(), PdeTermination::LinearIterationLimit);
        assert!(!report.is_success());
        assert!(!report.termination().is_success());
        assert_eq!(report.trust(), PartialResultTrust::DiagnosticOnly);
        assert_eq!(stats.residual_norm(), Some(0.25));
        assert!(!values.is_empty());
        assert_eq!(stats.num_assemblies(), 1);
        assert_eq!(stats.num_linear_solves(), 1);
        assert_eq!(stats.termination(), PdeTermination::LinearIterationLimit);
        assert_eq!(stats.num_linear_iterations(), 7);
        assert_eq!(stats.residual_history(), None);
        assert_eq!(stats.method(), Some("NonConvergingLinearSolver"));
    }
}

#[test]
fn test_elliptic_solver_preserves_injected_structural_failure() {
    let result = steady(
        Fdm,
        &uniform(11),
        &Poisson1D,
        &homogeneous_dirichlet(),
        EllipticOptions::new(),
        Some(Box::new(StructurallyFailingLinearSolver)),
    );
    assert_error!(
        result,
        SciOxideErrorView::DimensionMismatch { expected, actual, .. }
            if expected == [2].as_slice() && actual == [1].as_slice()
    );
}

// ============================================================================
// f32 support
// ============================================================================

#[test]
fn test_elliptic_f32() {
    // f32 has lower precision; allow a larger tolerance.
    let pi = std::f32::consts::PI;
    let mesh = uniform_f32(51);
    for discretization in DISCRETIZATIONS {
        let (values, _) = steady(
            discretization,
            &mesh,
            &Poisson1D,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        let max_err = values
            .iter()
            .zip(discretization.coordinates(&mesh))
            .map(|(&u, x)| (u - (pi * x).sin()).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 0.01,
            "{} f32 Poisson max error: {max_err}",
            discretization.name()
        );
    }
}

// ============================================================================
// Error path tests
// ============================================================================

#[test]
fn test_parabolic_options_reject_invalid_time_params() {
    // The step belongs to the options and the end time to the problem.
    let mesh = uniform(5);
    let boundaries = homogeneous_dirichlet().structured(&mesh).unwrap();
    let fdm = StructuredFdmPlan::with_options(mesh, &UnitDiffusion, &boundaries, support::limits())
        .unwrap();
    let source = FieldData::zero();
    let problem = ParabolicProblem::new(&fdm, &source, 0.1).unwrap();
    assert_error!(
        problem.initial_state(&FieldData::zero(), &explicit_parabolic_options(-0.01)),
        SciOxideErrorView::InvalidInput { .. }
    );
    assert_error!(
        ParabolicProblem::new(&fdm, &source, -1.0),
        SciOxideErrorView::InvalidInput { .. }
    );
    assert_error!(
        ParabolicProblem::new(&fdm, &source, f64::NAN),
        SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    );
}

#[test]
fn test_hyperbolic_invalid_time_params() {
    let options = wave_options(0.01, Tolerance::from_sqrt_epsilon());
    let result = wave(
        Fdm,
        &uniform(10),
        &WAVE,
        &homogeneous_dirichlet(),
        options,
        -1.0,
    );
    assert_error!(result, SciOxideErrorView::InvalidInput { .. });
}

#[test]
fn test_supported_callback_domain_error_is_preserved() {
    let result = steady(
        Fdm,
        &uniform(5),
        &DomainFailingElliptic,
        &homogeneous_dirichlet(),
        EllipticOptions::new(),
        None,
    );
    assert_error!(result, SciOxideErrorView::DomainError { .. });
}

#[test]
fn test_pde_stats_population() {
    let (values, report) = steady(
        Fdm,
        &uniform(21),
        &Poisson1D,
        &homogeneous_dirichlet(),
        EllipticOptions::new(),
        None,
    )
    .unwrap();

    let stats = report.diagnostics();
    assert_eq!(values.len(), 21);
    assert_eq!(stats.num_assemblies(), 1);
    assert_eq!(stats.num_linear_solves(), 1);
    assert_eq!(stats.num_time_steps(), 0); // Elliptic = steady state
    assert_eq!(stats.num_accepted_steps(), 0);
    assert_eq!(stats.num_rejected_steps(), 0);
    assert_eq!(stats.num_rhs_evaluations(), 0);
}

#[test]
fn test_pde_options_default() {
    let opts = EllipticOptions::<f64>::new();
    assert_eq!(opts.max_linear_iterations().get(), 1000);
    assert!(format!("{opts:?}").contains("EllipticOptions"));
    assert_eq!(opts.max_linear_iterations().get(), 1000);
}

#[test]
fn test_non_uniform_grid_elliptic() {
    // Very coarse grid, but it should still capture the right shape.
    let grid = mesh_fixture::explicit(vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]);
    for discretization in [Fdm, Fem] {
        let (values, _) = steady(
            discretization,
            &grid,
            &Poisson1D,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            None,
        )
        .unwrap();
        let max_err = max_error(discretization, &grid, &values, |x| (PI * x).sin());
        assert!(
            max_err < 0.2,
            "{} non-uniform grid max error: {max_err}",
            discretization.name()
        );
    }
}

// ============================================================================
// Periodic BC tests
// ============================================================================

#[test]
fn test_fdm_periodic_elliptic() {
    // -u'' = sin(x) on [0, 2pi] with periodic topology; exact u(x) = sin(x).
    #[derive(Clone)]
    struct PeriodicPoisson;
    impl EllipticPde<f64> for PeriodicPoisson {
        fn diffusion(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn reaction(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn source(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0].sin())
        }
    }

    let grid = periodic(&mesh_fixture::uniform(0.0, 2.0 * PI, 101));
    let opts = EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(0.0));
    let (values, _) = steady(Fdm, &grid, &PeriodicPoisson, &Ends::periodic(), opts, None).unwrap();

    // Periodic topology allows an additive constant; compare shapes.
    let nodes = Fdm.coordinates(&grid);
    let quarter = nodes.len() / 4; // value at pi/2
    let offset = values[quarter] - nodes[quarter].sin();
    let max_err = max_error(Fdm, &grid, &values, |x| x.sin() + offset);
    assert!(max_err < 0.05, "FDM periodic max error: {max_err}");
}

// ============================================================================
// Cross-solver consistency test
// ============================================================================

#[test]
fn test_fdm_fem_consistency_poisson() {
    // Both FDM and FEM should produce similar results on the same grid.
    let solve = |discretization: Discretization| {
        steady(
            discretization,
            &uniform(51),
            &Poisson1D,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            None,
        )
        .unwrap()
        .0
    };
    let (fdm, fem) = (solve(Fdm), solve(Fem));
    let max_diff = fdm
        .iter()
        .zip(&fem)
        .map(|(&u_fdm, &u_fem)| (u_fdm - u_fem).abs())
        .fold(0.0_f64, f64::max);

    // Both are O(h^2), should agree to within discretization error
    assert!(max_diff < 1e-3, "FDM vs FEM disagreement: {max_diff}");
}
