//! Nonlinear, mass, output, tolerance and geometric tests for real-time complex ODEs.
#![allow(clippy::unwrap_used)]
use core::marker::PhantomData;
use scioxide_core::*;
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

struct Riccati<S> {
    analytic: bool,
    mass: bool,
    scalar: PhantomData<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S, Complex<S>> for Riccati<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        y: &[Complex<S>],
        out: &mut [Complex<S>],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mass = Complex::new(s::<S>(1.), s::<S>(if self.mass { 1. } else { 0. }));
        out[0] = -mass * y[0] * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        y: &[Complex<S>],
        out: &mut MatrixViewMut<'_, Complex<S>>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if !self.analytic {
            return Ok(None);
        }
        let mass = Complex::new(s::<S>(-2.), s::<S>(if self.mass { -2. } else { 0. }));
        out.set(0, 0, mass * y[0])?;
        Ok(Some(()))
    }
    fn mass_matrix_revision(&self) -> Option<u64> {
        self.mass.then_some(1)
    }
    fn mass_matrix(
        &self,
        out: &mut MatrixViewMut<'_, Complex<S>>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.set(0, 0, Complex::new(s::<S>(1.), s::<S>(1.)))
    }
}
/// Integrate `[0, 1]` from `(0, y0)`; returns the report and the final values.
fn integrate<S, M, P>(
    method: M,
    options: OdeOptions<S>,
    problem: &P,
    initial: Vector<Complex<S>>,
) -> (OdeSolveReport<S>, Vector<Complex<S>>)
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    M: OdeMethod<S, Complex<S>> + Clone,
    P: OdeSystem<S, Complex<S>>,
{
    let plan = (method, options);
    let mut state = OdeState::new(S::ZERO, initial);
    let report = plan
        .0
        .advance(
            &OdeProblem::new(problem, S::ONE),
            &plan.1,
            &mut state,
            &mut (),
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    (report, state.into_parts().1)
}
/// Fixed-step methods (including fixed-order BDF) take `1/steps` steps. Radau IIA adapts
/// from that first step under the same tolerance, which also governs its stage solves.
fn nonlinear<S, M>(solver: M, mass: bool, analytic: bool, steps: usize, bound: f64)
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    M: OdeMethod<S, Complex<S>> + Clone + Into<BuiltinOdeMethod<S, Complex<S>>>,
{
    let name = solver.clone().into().name();
    let problem = Riccati {
        analytic,
        mass,
        scalar: PhantomData,
    };
    let initial = Complex::new(s::<S>(0.5), s::<S>(0.2));
    let tol = ErrorTolerance::scalar(
        s::<S>(if S::EPSILON > s(1e-10) { 1e-6 } else { 1e-11 }),
        s(0.),
        NormSelection::Infinity,
    )
    .unwrap();
    let h = s::<S>(1.0 / steps as f64);
    let options = OdeOptions::new()
        .with_error_tolerance(tol)
        .with_initial_step(h);
    let (_, y) = integrate(solver, options, &problem, Vector::from_vec(vec![initial]));
    let exact = initial / (Complex::new(s(1.), s(0.)) + initial);
    assert!(
        (y.as_slice()[0] - exact).abs() < s::<S>(bound),
        "{name} nonlinear error {:?}, mass={mass}, analytic={analytic}",
        (y.as_slice()[0] - exact).abs()
    );
}
fn nonlinear_domains<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    for mass in [false, true] {
        for analytic in [false, true] {
            nonlinear::<S, _>(RadauIia::new(), mass, analytic, 20, 0.000005);
            for mode in [NewtonMode::Simplified, NewtonMode::Full] {
                nonlinear::<S, _>(
                    BackwardEuler::new().with_newton_mode(mode),
                    mass,
                    analytic,
                    100,
                    0.001,
                );
                nonlinear::<S, _>(
                    ImplicitTrapezoid::new().with_newton_mode(mode),
                    mass,
                    analytic,
                    40,
                    0.00005,
                );
                nonlinear::<S, _>(
                    GaussRk::new().with_newton_mode(mode),
                    mass,
                    analytic,
                    20,
                    0.000005,
                );
                nonlinear::<S, _>(
                    Sdirk::new().with_newton_mode(mode),
                    mass,
                    analytic,
                    40,
                    0.00001,
                );
                nonlinear::<S, _>(
                    Bdf::new(3).unwrap().with_newton_mode(mode),
                    mass,
                    analytic,
                    100,
                    0.00025,
                );
            }
        }
    }
}
#[test]
fn nonlinear_and_complex_mass_f32() {
    nonlinear_domains::<f32>();
}
#[test]
fn nonlinear_and_complex_mass_f64() {
    nonlinear_domains::<f64>();
}

struct Rotation;
impl<S: RealScalar, T: NumericScalar<Real = S>> OdeSystem<S, T> for Rotation {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        y: &[T],
        out: &mut [T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = T::from_components(-y[0].imaginary_component(), y[0].real_component());
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, T::from_components(S::ZERO, S::ONE))?;
        Ok(Some(()))
    }
}
fn output<S, M>(solver: M)
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    M: OdeMethod<S, Complex<S>> + Clone,
{
    let tol = if S::EPSILON > s::<S>(1e-10) {
        s::<S>(1e-4)
    } else {
        s::<S>(1e-8)
    };
    let times = vec![s::<S>(0.125), s(0.125), s(0.333), s(0.731)];
    let options = OdeOptions::new()
        .with_initial_step(s(0.02))
        .with_maximum_step(s(0.02))
        .with_error_tolerance(
            ErrorTolerance::scalar(tol, tol, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_output_schedule(OdeOutputSchedule::times(times.clone()));
    let plan = (solver, options);
    let mut trajectory = OdeTrajectory::new();
    let mut state = OdeState::new(s(0.), Vector::from_vec(vec![Complex::new(s(1.), s(0.))]));
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&Rotation, s(0.731)),
            &plan.1,
            &mut state,
            &mut trajectory,
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(trajectory.times(), &times);
    assert_eq!(trajectory.states()[0], trajectory.states()[1]);
    for (&t, y) in trajectory.times().iter().zip(trajectory.states()) {
        assert!((y.as_slice()[0] - Complex::new(t.cos(), t.sin())).abs() < s::<S>(0.0003));
    }
}
fn output_domains<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    output::<S, _>(Rkf45::new());
    output::<S, _>(Verner65::new());
    output::<S, _>(DormandPrince853::new());
    output::<S, _>(AdamsBashforth::new(4).unwrap());
    output::<S, _>(AdamsMoulton::new(4).unwrap());
    output::<S, _>(Bdf::new(3).unwrap());
}
#[test]
fn complex_dense_requested_times_f32() {
    output_domains::<f32>();
}
#[test]
fn complex_dense_requested_times_f64() {
    output_domains::<f64>();
}

#[test]
fn callback_and_inner_failure_do_not_publish_complex_proposals() {
    let initial = Vector::from_vec(vec![Complex::new(1_f64, 2.)]);
    let broken = OdeFn::new(1, |_: f64, _: &[Complex<f64>], out: &mut [Complex<f64>]| {
        out[0] = Complex::new(0., f64::INFINITY);
        Ok(())
    })
    .unwrap();
    let plan = (
        BackwardEuler::<f64, Complex<f64>>::new(),
        OdeOptions::new().with_initial_step(0.1),
    );
    let mut y = OdeState::new(0., initial.clone());
    let mut w = OdeWorkspace::new();
    assert!(
        plan.0
            .advance(
                &OdeProblem::new(&broken, 0.1),
                &plan.1,
                &mut y,
                &mut (),
                &mut w,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(y.values(), initial.as_slice());
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    // One Newton iteration cannot meet the tiny tolerance. Fixed-order BDF is a
    // fixed-step method, so the failure is returned without publishing the proposal.
    let problem = Riccati {
        analytic: true,
        mass: false,
        scalar: PhantomData,
    };
    let tiny = ErrorTolerance::scalar(1e-14, 0., NormSelection::Infinity).unwrap();
    let plan = (
        Bdf::<f64, Complex<f64>>::new(3)
            .unwrap()
            .with_max_iterations(Budget::ONE),
        OdeOptions::new()
            .with_error_tolerance(tiny)
            .with_initial_step(0.5),
    );
    let mut w = OdeWorkspace::new();
    let error = plan
        .0
        .advance(
            &OdeProblem::new(&problem, 0.5),
            &plan.1,
            &mut y,
            &mut (),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(y.time(), 0.);
    assert_eq!(y.values(), initial.as_slice());
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().state(), initial.as_slice());
}

struct Oscillator;
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> OdeSystem<S, T>
    for Oscillator
{
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        y: &[T],
        out: &mut [T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[1];
        out[1] = -y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, T::ZERO)?;
        out.set(0, 1, T::ONE)?;
        out.set(1, 0, -T::ONE)?;
        out.set(1, 1, T::ZERO)?;
        Ok(Some(()))
    }
}
fn geometric<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let initial = Vector::from_vec(vec![
        Complex::new(s::<S>(1.), s(2.)),
        Complex::new(s(0.3), s(-0.2)),
    ]);
    let energy = |y: &[Complex<S>]| (y[0].norm_sqr() + y[1].norm_sqr()) / s::<S>(2.);
    let tol = ErrorTolerance::scalar(
        s::<S>(if S::EPSILON > s(1e-10) { 1e-5 } else { 1e-12 }),
        s(0.),
        NormSelection::Infinity,
    )
    .unwrap();
    let plan = (
        GaussRk::new(),
        OdeOptions::new()
            .with_error_tolerance(tol)
            .with_initial_step(s(0.1))
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
    );
    let mut w = OdeWorkspace::new();
    let initial_energy = energy(initial.as_slice());
    let mut maximum = S::ZERO;
    let end = (0..1000).fold(S::ZERO, |time, _| time + s(0.1));
    let mut y = OdeState::new(S::ZERO, initial.clone());
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&Oscillator, end),
            &plan.1,
            &mut y,
            &mut |_: S, values: &[Complex<S>]| {
                maximum = maximum.max((energy(values) - initial_energy).abs());
                Ok(())
            },
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    let bound = s::<S>(if S::EPSILON > s(1e-10) { 0.001 } else { 1e-9 });
    assert!(maximum < bound);
    // Integrate back to the start with the same step magnitude.
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&Oscillator, S::ZERO),
            &plan.1,
            &mut y,
            &mut (),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    for (&a, &b) in y.values().iter().zip(initial.as_slice()) {
        assert!((a - b).abs() < bound);
    }
    // Independent determinant of the canonical 2x2 step map: M^T J M = J.
    let mut columns = Vec::new();
    for basis in [[1., 0.], [0., 1.]] {
        let mut y = OdeState::new(
            S::ZERO,
            Vector::from_vec(basis.map(|v| Complex::new(s::<S>(v), s(0.))).to_vec()),
        );
        let report = plan
            .0
            .advance(
                &OdeProblem::new(&Oscillator, s(0.1)),
                &plan.1,
                &mut y,
                &mut (),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.accepted_steps(), 1);
        columns.push(y.into_parts().1);
    }
    let determinant = columns[0].as_slice()[0].re * columns[1].as_slice()[1].re
        - columns[1].as_slice()[0].re * columns[0].as_slice()[1].re;
    assert!((determinant - s::<S>(1.)).abs() < s::<S>(20.) * S::EPSILON);
}
#[test]
fn complex_gauss_geometric_f32() {
    geometric::<f32>();
}
#[test]
fn complex_gauss_geometric_f64() {
    geometric::<f64>();
}

struct ScalarLinearProvider {
    wrong: bool,
}
impl StageLinearSolver<f64, Complex<f64>> for ScalarLinearProvider {
    fn solve_into(
        &self,
        operator: &dyn LinearOperator<Complex<f64>>,
        _: &[Complex<f64>],
        rhs: &[Complex<f64>],
        out: &mut [Complex<f64>],
        accuracy: StageAccuracy<'_, f64, Complex<f64>>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        accuracy.admit_provider(EvaluationRole::LinearOperator, cx)?;
        let mut applied = [Complex::new(0., 0.)];
        operator
            .apply_into_scoped(
                &[Complex::new(1., 0.)],
                &mut applied,
                &mut [Complex::new(0., 0.)],
                &mut OperatorContext::new(cx),
            )?
            .require()?;
        out[0] = if self.wrong {
            Complex::new(0., 0.)
        } else {
            rhs[0] / applied[0]
        };
        Ok(())
    }
}
#[test]
fn injected_complex_linear_corrections_are_independently_verified() {
    for wrong in [false, true] {
        let provider = ScalarLinearProvider { wrong };
        let plan = (
            BackwardEuler::new(),
            OdeOptions::new().with_initial_step(0.1),
        );
        let initial = Complex::new(1., 0.2);
        let mut y = OdeState::new(0., Vector::from_vec(vec![initial]));
        let result = plan.0.advance(
            &OdeProblem::new(&Rotation, 0.1)
                .with_stage_solver(scioxide_ode::OdeStageSolver::Linear(&provider)),
            &plan.1,
            &mut y,
            &mut (),
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        );
        if wrong {
            assert!(result.is_err());
            assert_eq!(y.values(), &[initial]);
        } else {
            assert!(result.unwrap().is_success());
            assert!((y.values()[0] - initial / Complex::new(1., -0.1)).abs() < 1e-12);
        }
    }
}
#[test]
fn complex_budget_and_cancellation_preserve_committed_state() {
    let plan = (
        BackwardEuler::<f64, Complex<f64>>::new(),
        OdeOptions::new()
            .with_initial_step(0.01)
            .with_max_steps(Budget::ONE),
    );
    let problem = OdeProblem::new(&Rotation, 1.);
    let initial = Vector::from_vec(vec![Complex::new(1., 0.)]);
    let mut w = OdeWorkspace::new();
    let mut y = OdeState::new(0., initial.clone());
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut y,
            &mut (),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(!report.is_success());
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!(report.final_time(), 0.01);
    assert_eq!(y.time(), 0.01);
    assert!(y.values()[0].has_finite_components());
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
    let mut y = OdeState::new(0., initial.clone());
    let report = plan
        .0
        .advance(&problem, &plan.1, &mut y, &mut (), &mut w, execution)
        .unwrap();
    assert!(matches!(
        report.termination(),
        Some(OdeTermination::ExecutionStopped(_))
    ));
    assert_eq!(y.values(), initial.as_slice());
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Stopped);
}
/// One ensemble member: its caller-owned state and workspace.
struct Member<S: RealScalar + NumericScalar<Real = S>>
where
    Complex<S>: NumericScalar<Real = S>,
{
    state: OdeState<S, Complex<S>>,
    workspace: OdeWorkspace<S, Bdf<S, Complex<S>>, Complex<S>>,
}
fn ensemble<S: RealScalar + NumericScalar<Real = S>>(
    parallel: bool,
) -> (Vec<Vector<Complex<S>>>, EvaluationCounts)
where
    Complex<S>: NumericScalar<Real = S>,
{
    use core::num::NonZeroUsize;
    let solver = (
        Bdf::<S, Complex<S>>::new(3).unwrap(),
        OdeOptions::new()
            .with_initial_step(s(0.005))
            .with_maximum_step(s(0.005)),
    );
    let policy = if parallel {
        ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap())
    } else {
        ExecutionPolicy::Sequential
    };
    let plan = policy;
    let parameters = EnsembleParameters::Shared(&[]);
    let mut w = {
        let mut workspace = scioxide_ode::EnsembleWorkspace::new();
        workspace
            .try_reserve(&scioxide_ode::EnsembleRequirements::new(3, parameters).unwrap())
            .unwrap();
        workspace
    };
    let mut members: Vec<Member<S>> = (0..3)
        .map(|_| Member {
            state: OdeState::new(s(0.), Vector::from_vec(vec![Complex::new(s(1.), s(0.))])),
            workspace: OdeWorkspace::new(),
        })
        .collect();
    let mut output: Vec<Option<Result<OdeSolveReport<S>, SciOxideError>>> =
        (0..3).map(|_| None).collect();
    #[cfg(feature = "std")]
    let executor: Box<dyn Executor> = if parallel {
        Box::new(ThreadExecutor::new(NonZeroUsize::new(2).unwrap()))
    } else {
        Box::new(SequentialExecutor)
    };
    #[cfg(not(feature = "std"))]
    let executor: Box<dyn Executor> = Box::new(SequentialExecutor);
    // Problems borrow non-`Sync` event and stage-solver slices; build one per member.
    let run = |_: usize, _: &[S], member: &mut Member<S>, context: &mut SolveContext<'_, '_>| {
        solver.0.advance(
            &OdeProblem::new(&Rotation, s(0.203)),
            &solver.1,
            &mut member.state,
            &mut (),
            &mut member.workspace,
            context,
        )
    };
    let report = scioxide_ode::solve_ensemble_into(
        parameters,
        &run,
        plan,
        &mut members,
        &mut output,
        &mut w,
        ExecutionContext::new(executor.as_ref()),
    )
    .unwrap();
    assert_eq!(report.successful(), 3);
    let exact = Complex::new(s::<S>(0.203).cos(), s::<S>(0.203).sin());
    for outcome in output {
        let report = outcome.unwrap().unwrap();
        assert!(report.is_success());
        assert_eq!(report.final_time(), s(0.203));
    }
    let states = members
        .into_iter()
        .map(|member| {
            assert!((member.state.values()[0] - exact).abs() < s::<S>(0.00005));
            member.state.into_parts().1
        })
        .collect();
    (states, report.accounting().total().unwrap())
}
#[test]
fn complex_executor_ensemble_f32() {
    assert_eq!(ensemble::<f32>(false), ensemble::<f32>(true));
}
#[test]
fn complex_executor_ensemble_f64() {
    assert_eq!(ensemble::<f64>(false), ensemble::<f64>(true));
}

struct StiffManufactured<S>(PhantomData<S>);
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S, Complex<S>> for StiffManufactured<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: S,
        y: &[Complex<S>],
        out: &mut [Complex<S>],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let exact = Complex::new(t.cos(), t.sin());
        out[0] =
            Complex::new(s::<S>(-1000.), s(50.)) * (y[0] - exact) + Complex::new(-t.sin(), t.cos());
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[Complex<S>],
        out: &mut MatrixViewMut<'_, Complex<S>>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, Complex::new(s(-1000.), s(50.)))?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
/// Fixed-step methods (including fixed-order BDF) take fifty steps of 0.01; Radau IIA
/// adapts from that first step under the same tolerance.
fn stiff<S, M>(solver: M, bound: f64)
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    M: OdeMethod<S, Complex<S>> + Clone + Into<BuiltinOdeMethod<S, Complex<S>>>,
{
    let name = solver.clone().into().name();
    let h = s::<S>(0.01);
    let t = (0..50).fold(S::ZERO, |t, _| t + h);
    let tol = ErrorTolerance::scalar(
        s::<S>(if S::EPSILON > s(1e-10) { 1e-5 } else { 1e-11 }),
        s(0.),
        NormSelection::Infinity,
    )
    .unwrap();
    let plan = (
        solver,
        OdeOptions::new()
            .with_error_tolerance(tol)
            .with_initial_step(h),
    );
    let mut y = OdeState::new(
        S::ZERO,
        Vector::from_vec(vec![Complex::new(s::<S>(1.), s(0.))]),
    );
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&StiffManufactured(PhantomData), t),
            &plan.1,
            &mut y,
            &mut (),
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!(
        (y.values()[0] - Complex::new(t.cos(), t.sin())).abs() < s::<S>(bound),
        "{name} stiff manufactured error {:?}",
        y.values()[0] - Complex::new(t.cos(), t.sin())
    );
}
fn stiff_domains<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    stiff::<S, _>(BackwardEuler::new(), 5e-5);
    stiff::<S, _>(ImplicitTrapezoid::new(), 1e-5);
    stiff::<S, _>(GaussRk::new(), 1e-5);
    stiff::<S, _>(RadauIia::new(), 1e-5);
    stiff::<S, _>(Sdirk::new(), 1e-5);
    stiff::<S, _>(Bdf::new(3).unwrap(), 1e-5);
}
#[test]
fn stiff_manufactured_f32() {
    stiff_domains::<f32>();
}
#[test]
fn stiff_manufactured_f64() {
    stiff_domains::<f64>();
}
