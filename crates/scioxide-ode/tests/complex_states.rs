//! Native complex states retain real time, event values, and component scales.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, ErrorTolerance, ExecutionContext, NormSelection, NumericScalar, RealScalar,
    SciOxideError,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, EventDirection, EventFunction, OdeMethod, OdeOptions, OdeOutputSchedule,
    OdeProblem, OdeState, OdeSystem, OdeTermination, OdeTrajectory, OdeWorkspace,
};

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
}

struct Stop;
impl<S: RealScalar, T: NumericScalar<Real = S>> EventFunction<S, T> for Stop {
    fn eval(&self, t: S, _: &[T]) -> Result<S, SciOxideError> {
        Ok(t - S::try_from_f64_lossy(0.75)?)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Rising
    }
    fn is_terminal(&self) -> bool {
        true
    }
}

fn rotation<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let f = |v| S::try_from_f64_lossy(v).unwrap();
    let tolerance = S::SQRT_EPSILON * f(0.1);
    let options = OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(tolerance, tolerance, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_output_schedule(OdeOutputSchedule::times(vec![f(0.2), f(0.4), f(0.6)]));
    let plan = (DormandPrince5::<S, Complex<S>>::new(), options);
    let events: [&dyn EventFunction<S, Complex<S>>; 1] = [&Stop];
    let problem = OdeProblem::new(&Rotation, f(1.0)).with_events(&events);
    let mut state = OdeState::new(f(0.0), Vector::from_vec(vec![Complex::new(f(1.0), f(0.0))]));
    let mut output = OdeTrajectory::new();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut output,
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::TerminalEvent));
    assert_eq!(output.events().len(), 1);
    assert_eq!(output.events()[0].time(), report.final_time());
    assert!((report.final_time() - f(0.75)).abs() < f(20.0) * S::SQRT_EPSILON);
    for (&t, y) in output
        .times()
        .iter()
        .zip(output.states())
        .map(|(t, y)| (t, y.as_slice()))
        .chain(core::iter::once((&report.final_time(), state.values())))
    {
        assert!((y[0].re - t.cos()).abs() < f(30.0) * tolerance);
        assert!((y[0].im - t.sin()).abs() < f(30.0) * tolerance);
    }
    assert_eq!(output.times().len(), 3);
}

#[test]
fn complex_f32_rotation_and_terminal_event() {
    rotation::<f32>();
}
#[test]
fn complex_f64_rotation_and_terminal_event() {
    rotation::<f64>();
}
