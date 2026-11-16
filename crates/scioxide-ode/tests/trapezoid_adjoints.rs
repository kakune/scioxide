#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Fixed-step discrete implicit-trapezoid adjoints on the residual DAE kernel.
use core::cell::Cell;
use scioxide_ad::CheckpointSchedule;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Complex, ConvergenceReason, ErrorTolerance,
    ExecutionContext, NormSelection, NumericScalar, RealScalar, ResidualTolerance, SciOxideError,
    SciOxideErrorView, SolveContext, StructuralPattern, StructuralPatternView,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;

/// `y' = p y`.
struct Growth<S: RealScalar> {
    parameter: [S; 1],
}
impl<T: NumericScalar> OdeSystem<T::Real, T> for Growth<T::Real> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: T::Real,
        y: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = T::from_real(self.parameter[0]) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: T::Real,
        _: &[T],
        output: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.set(0, 0, T::from_real(self.parameter[0]))?;
        Ok(Some(()))
    }
}
impl<T: NumericScalar> OdeAdjointSystem<T> for Growth<T::Real> {
    fn parameters(&self) -> &[T::Real] {
        &self.parameter
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: T::Real,
        y: &[T],
        seed: &[T],
        state: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state[0] = T::from_real(self.parameter[0]) * seed[0];
        parameters[0] = (seed[0].conj() * y[0]).real_component();
        Ok(())
    }
}

/// Terminal `|y|^2 / 2` plus the running cost `c |y|^2 / 2`.
struct Energy(f64);
impl<T: NumericScalar> AdjointObjective<T> for Energy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: T::Real,
        y: &[T],
        _: &[T::Real],
        state: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        let c = T::Real::try_from_f64_lossy(if part == AdjointObjectivePart::Running {
            self.0
        } else {
            1.
        })?;
        state[0] = T::from_real(c) * y[0];
        parameters.fill(T::Real::ZERO);
        Ok(AdjointObjectiveValue {
            value: c * y[0].norm_sqr() * T::Real::HALF,
            time_derivative: T::Real::ZERO,
        })
    }
}

/// Closed form of the trapezoid recurrence `y_{k+1} = r y_k` with
/// `r = (1 + h p / 2) / (1 - h p / 2)` for `Energy(c)` over `steps` steps:
/// `(J, dJ/dy0 / y0, dJ/dp)` with `|y0|^2 = norm`.
fn closed_form(norm: f64, p: f64, h: f64, steps: i32, c: f64) -> (f64, f64, f64) {
    let a = h * p / 2.;
    let r = (1. + a) / (1. - a);
    let mut bracket = r.powi(2 * steps) / 2.;
    let mut slope = f64::from(steps) * r.powi(2 * steps - 1);
    for k in 0..steps {
        bracket += h * c / 4. * (r.powi(2 * k) + r.powi(2 * k + 2));
        slope += h * c / 4.
            * (f64::from(2 * k) * r.powi(2 * k - 1) + f64::from(2 * k + 2) * r.powi(2 * k + 1));
    }
    (
        norm * bracket,
        2. * bracket,
        norm * slope * h / ((1. - a) * (1. - a)),
    )
}

fn options(step: f64) -> AdjointOptions<f64> {
    AdjointOptions::new(AdjointMode::Discrete, step)
        .unwrap()
        .with_fixed_step(true)
        .with_tape_capacity(64)
        .with_max_replay_steps(20000)
        .with_max_workspace_bytes(4 * 1024 * 1024)
}
fn tight<S: RealScalar>() -> DaeOptions<S> {
    let tolerance = S::try_from_f64_lossy(1e-13).unwrap();
    DaeOptions::new().with_numerics(
        scioxide_ode::DaeNumericalOptions::new()
            .with_differential_error(
                ErrorTolerance::scalar(tolerance, tolerance, NormSelection::Infinity).unwrap(),
            )
            .with_differential_residual(ResidualTolerance::new(tolerance, S::ZERO).unwrap()),
    )
}
fn failed(workspace: &DaeAdjointWorkspace<f64>) -> AdjointReport<f64> {
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    *attempt.progress().unwrap()
}

#[test]
fn linear_growth_matches_the_closed_form_recurrence() {
    for (interval, steps) in [((0., 0.5), 4), ((0.5, 0.), 4)] {
        let h = (interval.1 - interval.0) / f64::from(steps);
        let model = Growth { parameter: [-0.7] };
        let residual = OdeAdjointResidual::<f64, _>::new(&model).unwrap();
        for c in [0., 0.3] {
            let plan = DaeAdjointSolver::trapezoid(
                residual.dimension(),
                residual.parameter_dimension(),
                OdeStageBackend::Dense,
                scioxide_ode::DaeAdjointOptions::new(options(0.125), (tight()).numerics().clone())
                    .unwrap(),
            )
            .unwrap();
            let y0 = 1.2;
            let result = plan
                .solve(&scioxide_ode::DaeAdjointProblem::new(
                    &residual,
                    &Energy(c),
                    &[y0],
                    &[-0.7 * y0],
                    interval,
                ))
                .unwrap();
            let (objective, factor, parameter) = closed_form(y0 * y0, -0.7, h, steps, c);
            let report = result.report();
            assert_eq!(report.termination(), Some(AdjointTermination::ReachedEnd));
            assert_eq!(report.accepted_steps(), 4);
            assert!((report.objective().unwrap() - objective).abs() < 1e-14 * objective);
            assert!(
                (result.initial_gradient()[0] - factor * y0).abs() < 1e-13,
                "{interval:?} {c}: {result:?}"
            );
            assert!(
                (result.parameter_gradient()[0] - parameter).abs() < 1e-13,
                "{interval:?} {c}: {result:?} {parameter}"
            );
            // The trapezoid formula has no primal or quadrature error estimate.
            assert_eq!(report.errors().primal(), None);
            assert_eq!(report.errors().quadrature(), None);
        }
    }
}

#[test]
fn complex_state_with_real_parameter_matches_the_closed_form() {
    let model = Growth { parameter: [-0.7] };
    let residual = OdeAdjointResidual::<Complex<f64>, _>::new(&model).unwrap();
    let plan = DaeAdjointSolver::trapezoid(
        residual.dimension(),
        residual.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(options(0.125), (tight()).numerics().clone()).unwrap(),
    )
    .unwrap();
    let y0 = Complex::new(1.1, 0.4);
    let result = plan
        .solve(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.3),
            &[y0],
            &[y0 * Complex::new(-0.7, 0.)],
            (0., 0.5),
        ))
        .unwrap();
    let (objective, factor, parameter) = closed_form(y0.norm_sqr(), -0.7, 0.125, 4, 0.3);
    assert!((result.report().objective().unwrap() - objective).abs() < 1e-14);
    assert!((result.initial_gradient()[0] - y0 * Complex::new(factor, 0.)).abs() < 1e-13);
    assert!((result.parameter_gradient()[0] - parameter).abs() < 1e-13);
}

#[test]
fn zero_duration_interval_returns_the_terminal_gradient() {
    let model = Growth { parameter: [-0.7] };
    let residual = OdeAdjointResidual::<f64, _>::new(&model).unwrap();
    let plan = DaeAdjointSolver::trapezoid(
        residual.dimension(),
        residual.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(options(0.125), (tight()).numerics().clone()).unwrap(),
    )
    .unwrap();
    let result = plan
        .solve(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.3),
            &[2.],
            &[-1.4],
            (1., 1.),
        ))
        .unwrap();
    assert_eq!(result.report().accepted_steps(), 0);
    assert_eq!(result.initial_gradient(), [2.]);
    assert_eq!(result.parameter_gradient(), [0.]);
    assert_eq!(result.report().objective(), Some(2.));
}

/// `y0' = -p0 y0 + y1^2`, `y1' = -p1 y1 + t y0`.
struct Coupled {
    parameters: [f64; 2],
}
impl Coupled {
    fn rate(&self, t: f64, y: &[f64]) -> [f64; 2] {
        let [a, b] = self.parameters;
        [-a * y[0] + y[1] * y[1], -b * y[1] + t * y[0]]
    }
    fn entries(&self, t: f64, y: &[f64]) -> [f64; 4] {
        let [a, b] = self.parameters;
        [-a, 2. * y[1], t, -b]
    }
}
impl OdeSystem<f64> for Coupled {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&self.rate(t, y));
        Ok(())
    }
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        output: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (index, value) in self.entries(t, y).into_iter().enumerate() {
            output.set(index / 2, index % 2, value)?;
        }
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        t: f64,
        y: &[f64],
        direction: &[f64],
        output: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let j = self.entries(t, y);
        output[0] = j[0] * direction[0] + j[1] * direction[1];
        output[1] = j[2] * direction[0] + j[3] * direction[1];
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        t: f64,
        y: &[f64],
        _: StructuralPatternView<'_>,
        values: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        values.copy_from_slice(&self.entries(t, y));
        Ok(Some(()))
    }
}
impl OdeAdjointSystem<f64> for Coupled {
    fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        t: f64,
        y: &[f64],
        seed: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let j = self.entries(t, y);
        state[0] = j[0] * seed[0] + j[2] * seed[1];
        state[1] = j[1] * seed[0] + j[3] * seed[1];
        parameters[0] = -y[0] * seed[0];
        parameters[1] = -y[1] * seed[1];
        Ok(())
    }
}

/// Time-dependent parameter-dependent running cost, two observations (one off the
/// step mesh) and a nonlinear terminal cost.
struct Tracking {
    times: [f64; 2],
}
impl AdjointObjective<f64> for Tracking {
    fn observation_times(&self) -> &[f64] {
        &self.times
    }
    fn observation(
        &self,
        i: usize,
        y: &[f64],
        p: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<f64, SciOxideError> {
        let w = (i + 1) as f64;
        gy[0] = w * y[1] + 0.5 * p[0];
        gy[1] = w * y[0];
        gp[0] = 0.5 * y[0];
        gp[1] = 0.;
        Ok(w * y[0] * y[1] + 0.5 * p[0] * y[0])
    }
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        t: f64,
        y: &[f64],
        p: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        Ok(if part == AdjointObjectivePart::Terminal {
            gy[0] = 2. * y[0] + y[1];
            gy[1] = y[0];
            gp.fill(0.);
            AdjointObjectiveValue {
                value: y[0] * y[0] + y[0] * y[1],
                time_derivative: 0.,
            }
        } else {
            let energy = (y[0] * y[0] + 0.5 * y[1] * y[1]) / 2.;
            gy[0] = (1. + t) * y[0];
            gy[1] = (1. + t) * 0.5 * y[1] + 0.5 * p[1] * y[1];
            gp[0] = 0.;
            gp[1] = 0.25 * y[1] * y[1];
            AdjointObjectiveValue {
                value: (1. + t) * energy + 0.25 * p[1] * y[1] * y[1],
                time_derivative: energy,
            }
        })
    }
}

const TRACKING: Tracking = Tracking { times: [0.3, 0.5] };
const INTERVAL: (f64, f64) = (0., 0.75);

fn coupled_plan(
    model: &Coupled,
    backend: OdeStageBackend,
    policy: CheckpointPolicy,
    bytes: usize,
) -> DaeAdjointSolver<f64> {
    DaeAdjointSolver::trapezoid(
        model.dimension(),
        model.parameter_dimension(),
        backend,
        scioxide_ode::DaeAdjointOptions::new(
            options(0.125)
                .with_tape_capacity(16)
                .with_checkpoint_policy(policy)
                .with_max_workspace_bytes(bytes),
            (tight()).numerics().clone(),
        )
        .unwrap(),
    )
    .unwrap()
}
fn coupled_solve(
    plan: &DaeAdjointSolver<f64>,
    model: &Coupled,
    y0: [f64; 2],
) -> AdjointResult<f64> {
    let residual = OdeAdjointResidual::new(model).unwrap();
    plan.solve(&scioxide_ode::DaeAdjointProblem::new(
        &residual,
        &TRACKING,
        &y0,
        &model.rate(0., &y0),
        INTERVAL,
    ))
    .unwrap()
}
const Y0: [f64; 2] = [0.9, -0.4];
const P: [f64; 2] = [0.8, 0.3];
fn dense() -> DaeAdjointSolver<f64> {
    coupled_plan(
        &Coupled { parameters: P },
        OdeStageBackend::Dense,
        CheckpointPolicy::StoreAll,
        4 * 1024 * 1024,
    )
}

#[test]
fn directional_derivative_matches_centered_objective_difference() {
    let plan = dense();
    let result = coupled_solve(&plan, &Coupled { parameters: P }, Y0);
    // 0.125, 0.25, the off-mesh observation 0.3, 0.425, 0.5, 0.625, 0.75.
    assert_eq!(result.report().accepted_steps(), 7);
    let (dy, dp) = ([0.3, -0.7], [0.4, 0.2]);
    let epsilon = 1e-5;
    let values = [-1., 1.].map(|sign| {
        let shift = |x: [f64; 2], d: [f64; 2]| [0, 1].map(|i| x[i] + sign * epsilon * d[i]);
        coupled_solve(
            &plan,
            &Coupled {
                parameters: shift(P, dp),
            },
            shift(Y0, dy),
        )
        .report()
        .objective()
        .unwrap()
    });
    let difference = (values[1] - values[0]) / (2. * epsilon);
    let directional = (0..2)
        .map(|i| result.initial_gradient()[i] * dy[i] + result.parameter_gradient()[i] * dp[i])
        .sum::<f64>();
    // The central difference error is O(epsilon^2) plus roundoff O(eps / epsilon).
    assert!(
        (difference - directional).abs() < 1e-8,
        "{difference} {directional}"
    );
}

#[test]
fn checkpoint_policies_give_identical_gradients() {
    let model = Coupled { parameters: P };
    let reference = coupled_solve(&dense(), &model, Y0);
    assert_eq!(reference.report().replayed_steps(), 0);
    let residual = OdeAdjointResidual::new(&model).unwrap();
    let stored = dense()
        .workspace_requirements(&DaeAdjointProblem::new(
            &residual,
            &TRACKING,
            &Y0,
            &[],
            INTERVAL,
        ))
        .unwrap()
        .required_bytes();
    for policy in [
        CheckpointPolicy::MemoryBudget,
        CheckpointPolicy::Explicit(CheckpointSchedule::binomial(16, 2).unwrap()),
    ] {
        let plan = coupled_plan(&model, OdeStageBackend::Dense, policy, stored - 1);
        let result = coupled_solve(&plan, &model, Y0);
        assert!(result.report().replayed_steps() > 0);
        assert_eq!(result.initial_gradient(), reference.initial_gradient());
        assert_eq!(result.parameter_gradient(), reference.parameter_gradient());
        assert_eq!(result.report().objective(), reference.report().objective());
    }
}

#[test]
fn dense_sparse_and_krylov_backends_agree() {
    let model = Coupled { parameters: P };
    let reference = coupled_solve(&dense(), &model, Y0);
    for backend in [
        OdeStageBackend::Sparse {
            pattern: StructuralPattern::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1]).unwrap(),
            limits: SparseLuLimits::new(64, 128 * 1024),
        },
        OdeStageBackend::Krylov {
            iterations: Budget::try_from(16).unwrap(),
            restart: Budget::try_from(2).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        },
    ] {
        let plan = coupled_plan(
            &model,
            backend.clone(),
            CheckpointPolicy::StoreAll,
            4 * 1024 * 1024,
        );
        let result = coupled_solve(&plan, &model, Y0);
        for (a, b) in result
            .initial_gradient()
            .iter()
            .chain(result.parameter_gradient())
            .zip(
                reference
                    .initial_gradient()
                    .iter()
                    .chain(reference.parameter_gradient()),
            )
        {
            assert!((a - b).abs() < 1e-10, "{backend:?}: {result:?}");
        }
    }
}

/// Terminal cost `y[index]` alone.
struct Component(usize);
impl AdjointObjective<f64> for Component {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: f64,
        y: &[f64],
        _: &[f64],
        gy: &mut [f64],
        gp: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
        gy.fill(0.);
        gp.fill(0.);
        let terminal = part == AdjointObjectivePart::Terminal;
        if terminal {
            gy[self.0] = 1.;
        }
        Ok(AdjointObjectiveValue {
            value: if terminal { y[self.0] } else { 0. },
            time_derivative: 0.,
        })
    }
}

#[test]
fn forward_state_agrees_with_the_implicit_trapezoid_stepper() {
    let model = Coupled { parameters: P };
    let residual = OdeAdjointResidual::new(&model).unwrap();
    let stepper = ImplicitTrapezoid::<f64>::new()
        .solve(
            &OdeProblem::new(&model, 0.5),
            &(OdeOptions::new()
                .with_initial_step(0.125)
                .with_error_tolerance(
                    ErrorTolerance::scalar(1e-13, 1e-13, NormSelection::Infinity).unwrap(),
                )),
            OdeState::new(0., Vector::from_vec(Y0.to_vec())),
        )
        .unwrap();
    assert_eq!(stepper.report().accepted_steps(), 4);
    for (index, &expected) in stepper.state().values().iter().enumerate() {
        let plan = DaeAdjointSolver::trapezoid(
            residual.dimension(),
            residual.parameter_dimension(),
            OdeStageBackend::Dense,
            scioxide_ode::DaeAdjointOptions::new(options(0.125), (tight()).numerics().clone())
                .unwrap(),
        )
        .unwrap();
        let result = plan
            .solve(&scioxide_ode::DaeAdjointProblem::new(
                &residual,
                &Component(index),
                &Y0,
                &model.rate(0., &Y0),
                (0., 0.5),
            ))
            .unwrap();
        let value = result.report().objective().unwrap();
        assert!(
            (value - expected).abs() < 1e-11,
            "{index}: {value} {expected}"
        );
    }
}

#[test]
fn continuous_mode_and_adaptive_steps_are_rejected() {
    let model = Growth { parameter: [-0.7] };
    let residual = OdeAdjointResidual::<f64, _>::new(&model).unwrap();
    for (options, name) in [
        (
            AdjointOptions::new(AdjointMode::Continuous, 0.125)
                .unwrap()
                .with_fixed_step(true),
            scioxide_core::argument!("options.mode"),
        ),
        (
            AdjointOptions::new(AdjointMode::Discrete, 0.125).unwrap(),
            scioxide_core::argument!("options.fixed_step"),
        ),
    ] {
        let error = DaeAdjointSolver::<f64>::trapezoid(
            residual.dimension(),
            residual.parameter_dimension(),
            OdeStageBackend::Dense,
            scioxide_ode::DaeAdjointOptions::new(options, (tight()).numerics().clone()).unwrap(),
        )
        .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument == name),
            "{error:?}"
        );
    }
}

#[test]
fn exhausted_step_capacity_leaves_outputs_unchanged() {
    let model = Growth { parameter: [-0.7] };
    let residual = OdeAdjointResidual::<f64, _>::new(&model).unwrap();
    let plan = DaeAdjointSolver::trapezoid(
        residual.dimension(),
        residual.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(
            options(0.125).with_tape_capacity(3),
            (tight()).numerics().clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.3),
            &[1.],
            &[-0.7],
            (0., 0.5),
        ))
        .unwrap();
    let (mut y, mut p) = ([12.], [13.]);
    let error = plan
        .solve_into(
            &scioxide_ode::DaeAdjointProblem::new(
                &residual,
                &Energy(0.3),
                &[1.],
                &[-0.7],
                (0., 0.5),
            ),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            ..
        }
    ));
    assert_eq!(
        failed(&workspace).termination(),
        Some(AdjointTermination::BudgetLimit(BudgetKind::Iteration))
    );
    assert_eq!((y, p), ([12.], [13.]));
}

/// `y' = p y`, shifted by a constant once `limit` right-hand sides were evaluated.
struct Drifting {
    calls: Cell<usize>,
    limit: usize,
    parameters: [f64; 1],
}
impl OdeSystem<f64> for Drifting {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        out[0] = self.parameters[0] * y[0] + if calls >= self.limit { 1e-3 } else { 0. };
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        output: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.set(0, 0, self.parameters[0])?;
        Ok(Some(()))
    }
}
impl OdeAdjointSystem<f64> for Drifting {
    fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: f64,
        y: &[f64],
        seed: &[f64],
        state: &mut [f64],
        parameters: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        state[0] = self.parameters[0] * seed[0];
        parameters[0] = y[0] * seed[0];
        Ok(())
    }
}

#[test]
fn changed_replay_is_detected_without_publishing_gradients() {
    let drifting = |limit| Drifting {
        calls: Cell::new(0),
        limit,
        parameters: [-0.7],
    };
    let plan = |model: &Drifting, policy| {
        DaeAdjointSolver::trapezoid(
            model.dimension(),
            model.parameter_dimension(),
            OdeStageBackend::Dense,
            scioxide_ode::DaeAdjointOptions::new(
                options(0.125).with_checkpoint_policy(policy),
                (tight()).numerics().clone(),
            )
            .unwrap(),
        )
        .unwrap()
    };
    // Count the right-hand sides of one recording, which StoreAll never replays.
    let counting = drifting(usize::MAX);
    let residual = OdeAdjointResidual::new(&counting).unwrap();
    let recorded = plan(&counting, CheckpointPolicy::StoreAll)
        .solve(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.),
            &[1.],
            &[-0.7],
            (0., 0.5),
        ))
        .unwrap();
    assert_eq!(recorded.report().replayed_steps(), 0);
    let model = drifting(counting.calls.get());
    let residual = OdeAdjointResidual::new(&model).unwrap();
    let plan = plan(
        &model,
        CheckpointPolicy::Explicit(CheckpointSchedule::binomial(64, 1).unwrap()),
    );
    let mut workspace = plan
        .workspace(&scioxide_ode::DaeAdjointProblem::new(
            &residual,
            &Energy(0.),
            &[1.],
            &[-0.7],
            (0., 0.5),
        ))
        .unwrap();
    let (mut y, mut p) = ([12.], [13.]);
    let error = plan
        .solve_into(
            &scioxide_ode::DaeAdjointProblem::new(
                &residual,
                &Energy(0.),
                &[1.],
                &[-0.7],
                (0., 0.5),
            ),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::ReplayMismatch,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        failed(&workspace).termination(),
        Some(AdjointTermination::ReplayMismatch)
    );
    assert_eq!((y, p), ([12.], [13.]));
}

#[test]
fn prepared_trapezoid_solves_execute() {
    let plan = dense();
    let model = Coupled { parameters: P };
    let residual = OdeAdjointResidual::new(&model).unwrap();
    let rate = model.rate(0., &Y0);
    let mut workspace = plan
        .workspace(&scioxide_ode::DaeAdjointProblem::new(
            &residual, &TRACKING, &Y0, &rate, INTERVAL,
        ))
        .unwrap();
    let (mut y, mut p) = ([0.; 2], [0.; 2]);
    let report = plan
        .solve_into(
            &scioxide_ode::DaeAdjointProblem::new(&residual, &TRACKING, &Y0, &rate, INTERVAL),
            &mut y,
            &mut p,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let reference = coupled_solve(&plan, &model, Y0);
    assert_eq!(report.objective(), reference.report().objective());
    assert_eq!(y, reference.initial_gradient());
    assert_eq!(p, reference.parameter_gradient());
}
