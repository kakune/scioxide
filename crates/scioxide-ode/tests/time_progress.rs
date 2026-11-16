//! Independent constant/affine solutions and IEEE endpoint regressions.
#![allow(clippy::unwrap_used)]
use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Vector;
use scioxide_ode::*;

struct Constant<S> {
    start: S,
    end: S,
    derivative: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Constant<S> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: S,
        _: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        assert!(
            t >= self.start.min(self.end) && t <= self.start.max(self.end),
            "callback outside span: {t:?}"
        );
        out.fill(self.derivative);
        Ok(())
    }
}
/// Integrate `y' = derivative` over `span` from `initial`; returns the report and state.
fn integrate<S, M, Sys>(
    method: M,
    system: &Sys,
    span: (S, S),
    initial: S,
    options: OdeOptions<S>,
) -> (OdeSolveReport<S>, OdeState<S>)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
    Sys: OdeSystem<S>,
{
    let plan = (method, options);
    let mut state = OdeState::new(span.0, Vector::from_vec(vec![initial]));
    let report = plan
        .0
        .advance(
            &OdeProblem::new(system, span.1),
            &plan.1,
            &mut state,
            &mut (),
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(state.time(), report.final_time());
    (report, state)
}
fn run<S, M>(method: M, span: (S, S), options: OdeOptions<S>) -> (OdeSolveReport<S>, OdeState<S>)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    integrate(
        method,
        &Constant {
            start: span.0,
            end: span.1,
            derivative: <_ as scioxide_core::AlgebraicScalar>::ZERO,
        },
        span,
        <_ as scioxide_core::AlgebraicScalar>::ONE,
        options,
    )
}
fn endpoints<S, M>(make: impl Fn() -> M)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    let tiny = S::ZERO.next_after(<_ as scioxide_core::AlgebraicScalar>::ONE);
    let huge = S::MAX;
    let spans = [
        (<_ as scioxide_core::AlgebraicScalar>::ZERO, tiny),
        (-tiny, tiny),
        (
            S::ONE,
            S::ONE.next_after(<_ as scioxide_core::AlgebraicScalar>::ZERO),
        ),
        (
            huge.next_after(<_ as scioxide_core::AlgebraicScalar>::ZERO),
            huge,
        ),
        (
            S::try_from_f64_lossy(1e-20).unwrap(),
            S::try_from_f64_lossy(2e-20).unwrap(),
        ),
    ];
    for (a, b) in spans {
        for span in [(a, b), (b, a), (-a, -b), (-b, -a)] {
            for explicit in [false, true] {
                let options = if explicit {
                    OdeOptions::new().with_initial_step(NumericScalar::abs(span.1 - span.0))
                } else {
                    OdeOptions::new()
                };
                let (result, state) = run(make(), span, options);
                assert_eq!(result.termination(), Some(OdeTermination::ReachedEnd));
                assert_eq!(result.final_time(), span.1);
                assert_eq!(state.values(), &[S::ONE]);
            }
        }
    }
}
fn families<S: RealScalar + NumericScalar<Real = S>>() {
    endpoints(ForwardEuler::<S>::new);
    endpoints(ExplicitMidpoint::<S>::new);
    endpoints(Heun::<S>::new);
    endpoints(ClassicalRk4::<S>::new);
    endpoints(DormandPrince5::<S>::new);
    endpoints(DormandPrince853::<S>::new);
    endpoints(Rkf45::<S>::new);
    endpoints(Verner65::<S>::new);
    endpoints(BackwardEuler::<S>::new);
    endpoints(ImplicitTrapezoid::<S>::new);
    endpoints(GaussRk::<S>::new);
    endpoints(RadauIia::<S>::new);
    endpoints(Sdirk::<S>::new);
    endpoints(|| AdamsBashforth::<S>::new(3).unwrap());
    endpoints(|| AdamsMoulton::<S>::new(3).unwrap());
    endpoints(|| Bdf::<S>::new(3).unwrap());
    endpoints(|| AdamsMoulton::<S>::new(4).unwrap());
}
#[test]
fn endpoints_f32() {
    families::<f32>();
}
#[test]
fn endpoints_f64() {
    families::<f64>();
}

fn bounds<S: RealScalar + NumericScalar<Real = S>>() {
    let start = S::ONE;
    let end = start.next_after(start + start);
    let gap = end - start;
    for span in [(start, end), (end, start)] {
        let options = OdeOptions::new()
            .with_initial_step(gap + gap)
            .with_minimum_step(gap + gap);
        let (report, _) = run(DormandPrince5::new(), span, options);
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(report.final_time(), span.1);
        // 3/4 ULP rounds upward, but must not exceed an explicit maximum.
        let maximum = gap * S::try_from_f64_lossy(0.75).unwrap();
        for explicit in [false, true] {
            let options = OdeOptions::new().with_maximum_step(maximum);
            let options = if explicit {
                options.with_initial_step(maximum)
            } else {
                options
            };
            let (report, state) = run(DormandPrince5::new(), span, options);
            assert_eq!(report.termination(), Some(OdeTermination::StepSizeLimit));
            assert_eq!(report.final_time(), span.0);
            assert_eq!(state.values(), &[S::ONE]);
            assert_eq!(report.counts().accepted(), 0);
        }
    }
}
#[test]
fn bounds_f32() {
    bounds::<f32>();
}
#[test]
fn bounds_f64() {
    bounds::<f64>();
}

/// Derivative alternating between one and two by call, so every embedded error
/// estimate is nonzero and exceeds a subnormal tolerance even across one ulp.
struct Alternating {
    calls: std::cell::Cell<usize>,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Alternating {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        _: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        out.fill(if call.is_multiple_of(2) {
            S::ONE
        } else {
            S::ONE + S::ONE
        });
        Ok(())
    }
}
fn retries<S: RealScalar + NumericScalar<Real = S>>() {
    use scioxide_core::{ErrorTolerance, NormSelection};
    let start = S::ONE;
    let end = start.next_after(start + start);
    let gap = end - start;
    let tiny = S::ZERO.next_after(<_ as scioxide_core::AlgebraicScalar>::ONE);
    // The rejection retries at 3/4 of the one-ulp step, which rounds back onto an
    // endpoint: the driver ends with a step-size limit instead of spinning.
    let reject = ProportionalIntegralController::new(
        S::try_from_f64_lossy(0.9).unwrap(),
        S::try_from_f64_lossy(5.0).unwrap(),
        S::try_from_f64_lossy(0.75).unwrap(),
        S::ONE,
        S::ZERO,
    )
    .unwrap();
    for span in [(start, end), (end, start)] {
        let options = OdeOptions::new()
            .with_error_tolerance(
                ErrorTolerance::scalar(tiny, S::ZERO, NormSelection::Infinity).unwrap(),
            )
            .with_initial_step(gap)
            .with_step_control(OdeStepControl::ProportionalIntegral(reject.clone()));
        let system = Alternating {
            calls: std::cell::Cell::new(0),
        };
        let (result, state) = integrate(
            DormandPrince5::<S>::new(),
            &system,
            span,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
            options,
        );
        assert_eq!(result.termination(), Some(OdeTermination::StepSizeLimit));
        assert_eq!(result.final_time(), span.0);
        assert_eq!(state.values(), &[S::ZERO]);
        assert_eq!(result.counts().rejected(), 1);
        assert_eq!(result.counts().proposals(), 1);
        assert!(result.last_step_error_estimate().unwrap() > S::ONE);
    }
}
#[test]
fn retries_f32() {
    retries::<f32>();
}
#[test]
fn retries_f64() {
    retries::<f64>();
}

fn affine<S: RealScalar + NumericScalar<Real = S>>() {
    let small = S::try_from_f64_lossy(1e-20).unwrap();
    for span in [
        (<_ as scioxide_core::AlgebraicScalar>::ZERO, small),
        (small, <_ as scioxide_core::AlgebraicScalar>::ZERO),
    ] {
        let system = Constant {
            start: span.0,
            end: span.1,
            derivative: <_ as scioxide_core::AlgebraicScalar>::ONE,
        };
        let (report, state) = integrate(
            DormandPrince5::<S>::new(),
            &system,
            span,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
            OdeOptions::new(),
        );
        let exact = span.1 - span.0; // Integral of 1, independent of RK tables.
        let error = NumericScalar::abs(state.values()[0] - exact);
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert!(error <= small * S::EPSILON * S::try_from_f64_lossy(8.0).unwrap());
    }
}
#[test]
fn affine_f32() {
    affine::<f32>();
}
#[test]
fn affine_f64() {
    affine::<f64>();
}

fn setters<S: RealScalar + NumericScalar<Real = S>>() {
    use scioxide_core::{ErrorTolerance, NormSelection};
    // Setters are infallible; the plan validates the whole configuration.
    let validate = |options: OdeOptions<S>| DormandPrince5::<S>::new().validate(&options, 1);
    let a = S::ONE;
    let b = a + a;
    let lower_first = OdeOptions::new().with_minimum_step(a).with_maximum_step(b);
    let upper_first = OdeOptions::new().with_maximum_step(b).with_minimum_step(a);
    assert_eq!(lower_first, upper_first);
    assert!(validate(lower_first).is_ok());
    assert!(validate(OdeOptions::new().with_minimum_step(b).with_maximum_step(a)).is_err());
    assert!(validate(OdeOptions::new().with_maximum_step(a).with_minimum_step(b)).is_err());
    for value in [
        <_ as scioxide_core::AlgebraicScalar>::ZERO,
        -a,
        S::INFINITY,
        S::NAN,
    ] {
        assert!(validate(OdeOptions::new().with_initial_step(value)).is_err());
        assert!(validate(OdeOptions::new().with_minimum_step(value)).is_err());
        assert!(validate(OdeOptions::new().with_maximum_step(value)).is_err());
    }
    // Component-count compatibility belongs to plan validation. Assignment is
    // infallible, even when the configuration cannot solve this problem shape.
    let tolerance = ErrorTolerance::try_vector(&[a, a], a, NormSelection::Infinity).unwrap();
    let options = OdeOptions::new().with_error_tolerance(tolerance);
    assert!(matches!(
        validate(options.clone()).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!((DormandPrince5::<S>::new()).validate(&(options), 2).is_ok());
}
#[test]
fn setters_f32() {
    setters::<f32>();
}
#[test]
fn setters_f64() {
    setters::<f64>();
}

#[test]
fn clipped_subtraction_must_round_trip_before_publishing_an_endpoint() {
    for end in [1e-20, -1e-20] {
        for sign in [1.0, -1.0] {
            let span = (sign, sign * end);
            let options = OdeOptions::new().with_initial_step(2.0);
            let (report, _) = run(DormandPrince5::new(), span, options);
            assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
            assert_eq!(report.final_time(), span.1);
            assert!(report.counts().accepted() >= 2);
        }
    }
}

#[test]
fn dde_preserves_step_size_termination_from_the_shared_driver() {
    use scioxide_core::Budget;
    use scioxide_linalg::MatrixView;
    let start = 1.0_f64;
    let end = start.next_up();
    let plan = DdePlan::new(
        1,
        (start, end),
        &[1.0],
        &[],
        DdeOptions::new()
            .with_history_capacity(Budget::ONE)
            .with_schedule_capacity(Budget::try_from(8).unwrap())
            .with_ode_options(OdeOptions::new().with_initial_step((end - start) / 4.0)),
    )
    .unwrap();
    let history = |_: f64, out: &mut [f64]| {
        out.fill(1.0);
        Ok(())
    };
    let rhs = |_: f64, _: &[f64], _: MatrixView<'_, f64>, out: &mut [f64]| {
        out.fill(0.0);
        Ok(())
    };
    let solution = plan.solve(&DdeProblem::new(&rhs, &history)).unwrap();
    assert_eq!(
        solution.report().termination(),
        Some(DdeTermination::StepSizeLimit)
    );
    assert_eq!(solution.report().final_time(), start);
    assert_eq!(solution.state().time(), start);
}

#[test]
fn requested_adjacent_endpoint_and_execution_budgets_agree() {
    use scioxide_core::{Budget, BudgetKind};
    let start = f64::MAX.next_down();
    let end = f64::MAX;
    let options = OdeOptions::new()
        .with_max_steps(Budget::ONE)
        .with_output_schedule(OdeOutputSchedule::times(vec![end, end]));
    let plan = (DormandPrince5::new(), options.clone());
    let mut output = OdeTrajectory::new();
    let mut state = OdeState::new(start, Vector::from_vec(vec![1.0]));
    let report = plan
        .0
        .advance(
            &OdeProblem::new(
                &Constant {
                    start,
                    end,
                    derivative: 0.0,
                },
                end,
            ),
            &plan.1,
            &mut state,
            &mut output,
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(output.times(), &[end, end]);
    assert_eq!(report.counts().proposals(), 1);
    let (report, state) = integrate(
        DormandPrince5::new(),
        &Constant {
            start: 0.0,
            end: 1.0,
            derivative: 0.0,
        },
        (0.0, 1.0),
        1.0,
        options
            .with_output_schedule(OdeOutputSchedule::FinalOnly)
            .with_initial_step(0.25),
    );
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!(report.final_time(), 0.25);
    assert_eq!(state.values(), &[1.0]);
}
