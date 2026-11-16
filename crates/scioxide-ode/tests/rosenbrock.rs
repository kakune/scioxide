//! Independent manufactured solutions and backend/lifecycle coverage.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Budget, Complex, ErrorTolerance, EvaluationRole, ExecutionContext, LinearOperator,
    NormSelection, NumericScalar, RealScalar, SciOxideError, SolveContext, StructuralPattern,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeStageBackend, OdeState,
    OdeStepControl, OdeSystem, OdeTermination, OdeWorkspace, ProportionalIntegralController,
    RosenbrockMethod, RosenbrockW,
};
use scioxide_sparse::SparseLuLimits;

struct Decay<S> {
    rate: S,
    approximate: S,
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> LinearOperator<T>
    for Decay<S>
{
    fn rows(&self) -> usize {
        1
    }
    fn cols(&self) -> usize {
        1
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[T], y: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        y[0] = T::from_real(self.rate * self.approximate) * x[0];
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> OdeSystem<S, T>
    for Decay<S>
{
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        _: S,
        _: &[T],
        output: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(T::ZERO);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: S,
        y: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = T::from_real(self.rate) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[T],
        j: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, T::from_real(self.rate * self.approximate))?;
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: S,
        _: &[T],
        _: scioxide_core::StructuralPatternView<'_>,
        values: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        values[0] = T::from_real(self.rate * self.approximate);
        Ok(Some(()))
    }
    fn jacobian_operator(
        &self,
        _: S,
        _: &[T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn LinearOperator<T>>, SciOxideError> {
        Ok(Some(self))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
fn backends() -> Vec<OdeStageBackend> {
    vec![
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern: StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
            limits: SparseLuLimits::new(1, 8192),
        },
        OdeStageBackend::Krylov {
            iterations: Budget::try_from(20).unwrap(),
            restart: Budget::try_from(4).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        },
    ]
}

/// Exactly `h`-sized accepted steps: a loose tolerance accepts every step and a
/// unit-factor controller never changes the step. The scalar linear stages are solved
/// exactly by every backend, whatever the tolerance.
fn fixed<S: RealScalar + NumericScalar<Real = S>>(h: S) -> OdeOptions<S> {
    OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(S::ONE, S::ONE, NormSelection::Infinity).unwrap(),
        )
        .with_initial_step(h)
        .with_step_control(OdeStepControl::ProportionalIntegral(
            ProportionalIntegralController::new(S::ONE, S::ONE, S::ONE, S::ONE, S::ZERO).unwrap(),
        ))
}

fn integrate<S, T>(
    method: RosenbrockMethod,
    backend: OdeStageBackend,
    steps: u8,
    approximate: S,
) -> T
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let f = |v| S::try_from_f64_lossy(v).unwrap();
    let system = Decay {
        rate: f(-1.),
        approximate,
    };
    let solver = RosenbrockW::<S, T>::new(method)
        .unwrap()
        .with_backend(backend);
    let plan = (solver, fixed(f(1. / f64::from(steps))));
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&system, f(1.)),
            &plan.1,
            OdeState::new(
                f(0.),
                Vector::from_vec(vec![T::from_components(f(1.), f(0.5))]),
            ),
        )
        .unwrap();
    assert_eq!(
        solution.report().termination(),
        Some(OdeTermination::ReachedEnd)
    );
    assert_eq!(solution.report().accepted_steps(), usize::from(steps));
    assert_eq!(solution.report().rejected_steps(), 0);
    solution.state().values()[0]
}
#[test]
fn independently_known_order_with_approximate_jacobians() {
    for method in [RosenbrockMethod::Ros2, RosenbrockMethod::Ros34Pw2] {
        for approximate in [1., 0.7] {
            let exact = (-1.0_f64).exp();
            let coarse = (integrate::<f64, f64>(method, OdeStageBackend::Dense, 32, approximate)
                - exact)
                .abs();
            let fine = (integrate::<f64, f64>(method, OdeStageBackend::Dense, 64, approximate)
                - exact)
                .abs();
            let minimum = if method == RosenbrockMethod::Ros2 {
                3.5
            } else {
                7.0
            };
            assert!(
                coarse / fine > minimum,
                "{method:?} approximate={approximate}: {coarse}/{fine}"
            );
        }
    }
}
fn parity<S, T>()
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let f = |v| S::try_from_f64_lossy(v).unwrap();
    for method in [RosenbrockMethod::Ros2, RosenbrockMethod::Ros34Pw2] {
        let reference = integrate::<S, T>(method, OdeStageBackend::Dense, 32, f(1.));
        for backend in backends() {
            let value = integrate::<S, T>(method, backend, 32, f(1.));
            assert!((value - reference).abs() < f(100.) * S::EPSILON);
            let exact = T::from_components(f(-1.).exp(), f(0.5) * f(-1.).exp());
            assert!((value - exact).abs() < f(0.001));
        }
    }
}
#[test]
fn f32_backends() {
    parity::<f32, f32>();
}
#[test]
fn f64_backends() {
    parity::<f64, f64>();
}
#[test]
fn complex32_backends() {
    parity::<f32, Complex<f32>>();
}
#[test]
fn complex64_backends() {
    parity::<f64, Complex<f64>>();
}

struct Polynomial;
impl OdeSystem<f64> for Polynomial {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -10. * (y[0] - t * t) + 2. * t;
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -10.)?;
        Ok(Some(()))
    }
    fn time_derivative(
        &self,
        t: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out[0] = 20. * t + 2.;
        Ok(Some(()))
    }
}
#[test]
fn nonautonomous_polynomial_and_dense_endpoints() {
    for method in [RosenbrockMethod::Ros2, RosenbrockMethod::Ros34Pw2] {
        // One accepted step of 0.01: the loose tolerance only gates acceptance, since
        // dense linearly implicit stages have no inner tolerance.
        let plan = (
            RosenbrockW::new(method).unwrap(),
            fixed(0.01).with_output_schedule(OdeOutputSchedule::times(vec![0., 0.01])),
        );
        let mut samples = Vec::new();
        let mut sink = |t: f64, y: &[f64]| {
            samples.push((t, y[0]));
            Ok(())
        };
        let problem = OdeProblem::new(&Polynomial, 0.01);
        let mut state = OdeState::new(0., Vector::from_vec(vec![0.]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut sink,
                &mut plan.0.workspace(&problem, &plan.1).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.counts().proposals(), 1);
        let y = state.values()[0];
        assert_eq!(samples, [(0., 0.), (0.01, y)]);
        assert!((y - 0.0001).abs() < 2e-5);
        assert_eq!(report.counts().provider(EvaluationRole::Factorization), 1);
    }
}
#[test]
fn invalid_pattern_and_missing_required_analytic_time_derivative_are_errors() {
    let decay = Decay {
        rate: -1.,
        approximate: 1.,
    };
    let problem = OdeProblem::new(&decay, 0.1);
    let invalid = OdeStageBackend::Sparse {
        pattern: StructuralPattern::new(1, 1, vec![0, 0], vec![]).unwrap(),
        limits: SparseLuLimits::new(1, 8192),
    };
    let plan = (
        RosenbrockW::<f64>::new(RosenbrockMethod::Ros2)
            .unwrap()
            .with_backend(invalid),
        OdeOptions::new().with_initial_step(0.1),
    );
    assert!(plan.0.workspace(&problem, &plan.1).is_err());
    let mut workspace = OdeWorkspace::new();
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    assert!(
        plan.0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    assert_eq!(state.values(), [1.]);

    let plan = (
        RosenbrockW::<f64>::new(RosenbrockMethod::Ros2)
            .unwrap()
            .with_derivative_strategy(scioxide_differentiate::DerivativeStrategy::Analytic),
        OdeOptions::new().with_initial_step(0.1),
    );
    let missing_time = scioxide_ode::OdeFn::new(1, |_: f64, y: &[f64], out: &mut [f64]| {
        out[0] = -y[0];
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: f64, _: &[f64], out: &mut MatrixViewMut<'_, f64>| out.set(0, 0, -1.));
    let problem = OdeProblem::new(&missing_time, 0.1);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    assert!(
        plan.0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(state.time(), 0.);
    assert_eq!(state.values(), [1.]);
}

struct Exchange;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Exchange {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -S::try_from_f64_lossy(100.)? * y[0] + S::try_from_f64_lossy(2.)? * y[1];
        out[1] = -out[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        j: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, S::try_from_f64_lossy(-100.)?)?;
        j.set(0, 1, S::try_from_f64_lossy(2.)?)?;
        j.set(1, 0, S::try_from_f64_lossy(100.)?)?;
        j.set(1, 1, S::try_from_f64_lossy(-2.)?)?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
fn conservation<S: RealScalar + NumericScalar<Real = S>>() {
    let f = |v| S::try_from_f64_lossy(v).unwrap();
    let tol = S::SQRT_EPSILON;
    for method in [RosenbrockMethod::Ros2, RosenbrockMethod::Ros34Pw2] {
        let plan = (
            RosenbrockW::<S>::new(method).unwrap(),
            OdeOptions::new().with_error_tolerance(
                ErrorTolerance::scalar(tol, tol, NormSelection::Infinity).unwrap(),
            ),
        );
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&Exchange, f(0.1)),
                &plan.1,
                OdeState::new(f(0.), Vector::from_vec(vec![f(1.), f(0.)])),
            )
            .unwrap();
        assert!(solution.report().is_success());
        let y = solution.state().values();
        assert!((y[0] + y[1] - f(1.)).abs() < f(100.) * S::EPSILON);
        let exact = f(2. / 102.) + f(100. / 102.) * f(-10.2).exp();
        assert!((y[0] - exact).abs() < f(20.) * tol);
    }
}
#[test]
fn conservation_f32() {
    conservation::<f32>();
}
#[test]
fn conservation_f64() {
    conservation::<f64>();
}

/// y(t) = exp(t); no derivative providers at all.
struct Nonautonomous;
impl OdeSystem<f64> for Nonautonomous {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -2. * y[0] + 3. * t.exp();
        Ok(())
    }
}
#[test]
fn differenced_time_derivatives_preserve_declared_orders() {
    use scioxide_differentiate::DerivativeStrategy;
    for method in [RosenbrockMethod::Ros2, RosenbrockMethod::Ros34Pw2] {
        for strategy in [
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            DerivativeStrategy::Forward,
            DerivativeStrategy::Central,
        ] {
            let error = |steps: u32| {
                let plan = (
                    RosenbrockW::new(method)
                        .unwrap()
                        .with_derivative_strategy(strategy),
                    fixed(1. / f64::from(steps)),
                );
                let solution = plan
                    .0
                    .solve(
                        &OdeProblem::new(&Nonautonomous, 1.),
                        &plan.1,
                        OdeState::new(0., Vector::from_vec(vec![1.])),
                    )
                    .unwrap();
                assert_eq!(
                    solution.report().termination(),
                    Some(OdeTermination::ReachedEnd)
                );
                (solution.state().values()[0] - 1_f64.exp()).abs()
            };
            let ratio = error(32) / error(64);
            assert!(
                ratio
                    > if method == RosenbrockMethod::Ros2 {
                        3.5
                    } else {
                        7.0
                    },
                "{method:?} {strategy:?}: ratio={ratio}"
            );
        }
    }
}

struct TimeProbe {
    rhs: core::cell::Cell<usize>,
    time: core::cell::Cell<usize>,
    fail: bool,
    autonomous: bool,
}
impl OdeSystem<f64> for TimeProbe {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.rhs.set(self.rhs.get() + 1);
        out[0] = -y[0] + if self.autonomous { 0. } else { t };
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, -1.)?;
        Ok(Some(()))
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.time.set(self.time.get() + 1);
        if self.fail {
            return Err(SciOxideError::singular(None));
        }
        out[0] = if self.autonomous { 0. } else { 1. };
        Ok(Some(()))
    }
}
#[test]
fn time_strategy_respects_provider_precedence_errors_and_autonomy() {
    use scioxide_differentiate::DerivativeStrategy;
    for (strategy, autonomous, fail) in [
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            false,
            false,
        ),
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            false,
            true,
        ),
        (DerivativeStrategy::Forward, false, true),
        (DerivativeStrategy::Central, false, true),
        (DerivativeStrategy::Analytic, true, false),
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            true,
            false,
        ),
    ] {
        let system = TimeProbe {
            rhs: core::cell::Cell::new(0),
            time: core::cell::Cell::new(0),
            fail,
            autonomous,
        };
        let method = RosenbrockW::new(RosenbrockMethod::Ros2)
            .unwrap()
            .with_derivative_strategy(strategy);
        let plan = (method, fixed(0.01));
        let result = plan.0.solve(
            &OdeProblem::new(&system, 0.01),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        );
        let analytic = matches!(
            strategy,
            DerivativeStrategy::Analytic | DerivativeStrategy::AnalyticOrAutomaticOrForward
        );
        assert_eq!(system.time.get(), usize::from(analytic));
        if fail && analytic {
            assert_eq!(result.unwrap_err().error(), &SciOxideError::singular(None));
        } else {
            let solution = result.unwrap();
            assert_eq!(
                solution.report().termination(),
                Some(OdeTermination::ReachedEnd)
            );
            if autonomous {
                assert_eq!(
                    system.rhs.get(),
                    2,
                    "zero analytic time derivatives need no RHS probes"
                );
                // ROS2's local error here is approximately 1.32 h³ at h = 0.01.
                assert!((solution.state().values()[0] - (-0.01_f64).exp()).abs() < 2e-6);
            }
            assert_eq!(
                solution
                    .report()
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::RightHandSide),
                system.rhs.get()
            );
        }
    }
}

/// `y′ = −50(y − cos t)` with an analytic Jacobian, an optional analytic `∂f/∂t`, and
/// counts of every right-hand-side, Jacobian and time-derivative call.
struct Relaxation {
    analytic_time: bool,
    rhs: core::cell::Cell<usize>,
    jacobians: core::cell::Cell<usize>,
    time: core::cell::Cell<usize>,
}
impl OdeSystem<f64> for Relaxation {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.rhs.set(self.rhs.get() + 1);
        out[0] = -50. * (y[0] - t.cos());
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.jacobians.set(self.jacobians.get() + 1);
        out.set(0, 0, -50.)?;
        Ok(Some(()))
    }
    fn time_derivative(
        &self,
        t: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.time.set(self.time.get() + 1);
        if !self.analytic_time {
            return Ok(None);
        }
        out[0] = -50. * t.sin();
        Ok(Some(()))
    }
}

/// Rejected attempts reuse `f(tₙ, yₙ)` and `∂f/∂t` at their base, and no stage
/// multiplies the Jacobian by a zero stage combination.
#[test]
fn retries_reuse_base_derivatives_and_skip_zero_stage_products() {
    for (method, stages) in [(RosenbrockMethod::Ros2, 2), (RosenbrockMethod::Ros34Pw2, 4)] {
        for analytic_time in [true, false] {
            let system = Relaxation {
                analytic_time,
                rhs: core::cell::Cell::new(0),
                jacobians: core::cell::Cell::new(0),
                time: core::cell::Cell::new(0),
            };
            let options = OdeOptions::new()
                .with_error_tolerance(
                    ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::RootMeanSquare).unwrap(),
                )
                .with_initial_step(0.5);
            let plan = (RosenbrockW::new(method).unwrap(), options);
            let solution = plan
                .0
                .solve(
                    &OdeProblem::new(&system, 2.),
                    &plan.1,
                    OdeState::new(0., Vector::from_vec(vec![0.])),
                )
                .unwrap();
            let report = solution.report();
            assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
            let (accepted, rejected) = (report.accepted_steps(), report.rejected_steps());
            assert!(
                rejected > 0,
                "{method:?}: the large first step must be rejected"
            );
            let attempts = accepted + rejected;
            let counts = report.accounting().total().unwrap();
            // The first stage never multiplies J; from y(0) = 0 no later combination is zero.
            assert_eq!(
                counts.provider(EvaluationRole::LinearOperator),
                (stages - 1) * attempts
            );
            // One base derivative per accepted base, `s − 1` stage derivatives per
            // attempt, and one forward time difference per accepted base without ∂f/∂t.
            let differences = if analytic_time { 0 } else { accepted };
            assert_eq!(
                system.rhs.get(),
                accepted + (stages - 1) * attempts + differences
            );
            // An unavailable ∂f/∂t is asked once per epoch; every call is admitted
            // as Jacobian work.
            assert_eq!(system.time.get(), if analytic_time { accepted } else { 1 });
            assert_eq!(
                counts.provider(EvaluationRole::Jacobian),
                system.jacobians.get() + system.time.get()
            );
            // Exact solution: y = (2500 cos t + 50 sin t − 2500 e^{−50t})/2501.
            let exact =
                (2500. * 2_f64.cos() + 50. * 2_f64.sin() - 2500. * (-100_f64).exp()) / 2501.;
            assert!((solution.state().values()[0] - exact).abs() < 1e-5);
        }
    }
}
