//! Localization accuracy and work on the actual DP5 continuous extensions.

use alloc::{rc::Rc, vec::Vec};
use core::cell::{Cell, RefCell};

use scioxide_core::{
    ErrorTolerance, EvaluationRole, ExecutionContext, Interval, NormSelection, RealScalar,
    SciOxideError, SolveContext, StepTolerance,
};
use scioxide_linalg::Vector;
use scioxide_rootfind::{Bisection, ScalarFn, ScalarRootOptions, ScalarRootSolver};

use super::Decay;
use crate::ivp::events::EventStorage;
use crate::result::StepResult;
use crate::stepper::testing::with_context;
use crate::stepper::{MethodInfo, StepRequest, StepTrace, Stepper};
use crate::{
    DormandPrince5, EventDirection, EventFunction, OdeEventOptions, OdeMethod, OdeOptions,
    OdeProblem, OdeState, OdeSystem, OdeTermination, OdeTrajectory,
};

#[derive(Clone, Copy)]
enum ProbeEvent {
    Time,
    Kepler,
}

impl EventFunction<f64> for ProbeEvent {
    fn eval(&self, t: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(match self {
            Self::Time => t - 0.7,
            Self::Kepler => y[1],
        })
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Time)
    }
}

struct CountedEvent {
    event: ProbeEvent,
    calls: Cell<usize>,
}

impl EventFunction<f64> for CountedEvent {
    fn eval(&self, t: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        self.event.eval(t, y)
    }
    fn direction(&self) -> EventDirection {
        self.event.direction()
    }
    fn is_terminal(&self) -> bool {
        self.event.is_terminal()
    }
}

struct Kepler;
impl OdeSystem<f64> for Kepler {
    fn dimension(&self) -> usize {
        4
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let r2 = y[0] * y[0] + y[1] * y[1];
        let inverse = 1.0 / (r2 * r2.sqrt());
        out.copy_from_slice(&[y[2], y[3], -y[0] * inverse, -y[1] * inverse]);
        Ok(())
    }
}

/// Inspect the original borrowed segment before the driver consumes it. Reference
/// solves use independent contexts and never charge the integration's accounting.
#[derive(Clone)]
struct CheckedDp5 {
    method: DormandPrince5<f64>,
    event: ProbeEvent,
    references: Rc<RefCell<Vec<f64>>>,
}

fn within_four_ulps(actual: f64, expected: f64) -> bool {
    let ulp = expected.next_after(f64::INFINITY) - expected;
    (actual - expected).abs() <= 4.0 * ulp
}

impl CheckedDp5 {
    fn check(
        &self,
        start: f64,
        end: f64,
        dense: &crate::dense_output::DenseOutputStorage<f64>,
    ) -> Result<(), SciOxideError> {
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        let dimension = dense.dimension();
        dense.eval_into(start, &mut left[..dimension])?;
        dense.eval_into(end, &mut right[..dimension])?;
        let old = self.event.eval(start, &left[..dimension])?;
        let new = self.event.eval(end, &right[..dimension])?;
        if old == 0.0 || new == 0.0 || old.is_sign_positive() == new.is_sign_positive() {
            return Ok(());
        }
        let mut storage = EventStorage::default();
        storage.reserve(1, 1, dimension)?;
        let trace = StepTrace::default();
        with_context(|cx| {
            storage.init(&[&self.event], start, &left[..dimension], &trace, cx)?;
            assert!(storage.endpoint(
                &[&self.event],
                OdeEventOptions::new(),
                end,
                &right[..dimension],
                &trace,
                cx,
            )?);
            storage.detect(
                &[&self.event],
                OdeEventOptions::new(),
                start,
                end,
                dense,
                &trace,
                cx,
            )?;
            assert_eq!(storage.records.len(), 1);
            let localization_calls = storage.evaluations - 2;
            assert!(
                localization_calls <= 12,
                "{localization_calls} calls on [{start}, {end}]"
            );
            assert_eq!(
                cx.counts().total()?.provider(EvaluationRole::Event),
                storage.evaluations
            );
            assert_eq!(cx.counts().total()?.provider(EvaluationRole::Function), 0);
            Ok(())
        })?;
        let state = RefCell::new([0.0; 4]);
        let function = ScalarFn::new(|t| {
            let mut state = state.borrow_mut();
            dense.eval_into(t, &mut state[..dimension])?;
            self.event.eval(t, &state[..dimension])
        });
        // Refine beyond the requested 4ε tolerance to avoid reference error
        // consuming the four-ulp agreement allowance.
        let reference = Bisection.solve(
            &function,
            Interval::new(start, end)?,
            &ScalarRootOptions::default()
                .with_step_tolerance(StepTolerance::new(0.0, f64::EPSILON)?),
        )?;
        assert!(reference.is_success());
        let actual = storage.records[0].time;
        assert!(
            within_four_ulps(actual, reference.point()),
            "{actual} versus {}",
            reference.point()
        );
        self.references.borrow_mut().push(reference.point());
        Ok(())
    }
}

impl Stepper<f64, f64> for CheckedDp5 {
    type Workspace = <DormandPrince5<f64> as Stepper<f64, f64>>::Workspace;
    fn info(&self) -> MethodInfo {
        self.method.info()
    }
    fn fits(&self, n: usize, provided: bool, w: &Self::Workspace) -> bool {
        self.method.fits(n, provided, w)
    }
    fn reserve(
        &self,
        n: usize,
        provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.method.reserve(n, provided, w)
    }
    fn begin(
        &self,
        n: usize,
        provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.method.begin(n, provided, w)
    }
    fn accept(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        self.method.accept(w)
    }
    fn discard(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        self.method.discard(w)
    }
    fn step<'w, System: OdeSystem<f64> + ?Sized>(
        &self,
        system: &System,
        t: f64,
        y: &mut Vector<f64>,
        h: f64,
        request: StepRequest<'_, f64, f64>,
        w: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, f64>, SciOxideError> {
        let result = self.method.step(system, t, y, h, request, w, cx)?;
        if let Some(dense) = result.dense_output {
            self.check(t, t + h, dense)?;
        }
        Ok(result)
    }
}

#[test]
fn decay_and_kepler_events_resolve_the_same_interpolant_with_bounded_work() {
    for tolerance in [1e-6, 1e-10] {
        for probe in [ProbeEvent::Time, ProbeEvent::Kepler] {
            let (system, initial, end): (&dyn OdeSystem<f64>, &[f64], f64) = match probe {
                ProbeEvent::Time => (&Decay(1), &[1.0], 1.0),
                ProbeEvent::Kepler => (&Kepler, &[0.5, 0.0, 0.0, 3.0_f64.sqrt()], 20.0),
            };
            let event = CountedEvent {
                event: probe,
                calls: Cell::new(0),
            };
            let references = Rc::new(RefCell::new(Vec::new()));
            let method = CheckedDp5 {
                method: DormandPrince5::new(),
                event: probe,
                references: Rc::clone(&references),
            };
            let options = OdeOptions::new()
                .with_initial_step(1e-3)
                .with_error_tolerance(
                    ErrorTolerance::scalar(tolerance, tolerance, NormSelection::Infinity).unwrap(),
                );
            let plan = (method, options);
            let events: [&dyn EventFunction<f64>; 1] = [&event];
            let problem = OdeProblem::new(system, end).with_events(&events);
            let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
            let mut state = OdeState::from_slice(0.0, initial).unwrap();
            let mut trajectory = OdeTrajectory::new();
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut trajectory,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(
                report.event_search().unwrap().evaluations(),
                event.calls.get()
            );
            assert_eq!(
                report
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Event),
                event.calls.get()
            );
            assert_eq!(
                report
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Function),
                0
            );
            match probe {
                ProbeEvent::Time => {
                    assert_eq!(report.termination(), Some(OdeTermination::TerminalEvent));
                    assert!(within_four_ulps(state.time(), 0.7));
                    assert_eq!(trajectory.events().len(), 1);
                }
                ProbeEvent::Kepler => {
                    // The exact zero at the start is not an event.
                    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
                    assert_eq!(trajectory.events().len(), 6);
                }
            }
            for event in trajectory.events() {
                assert!(
                    references
                        .borrow()
                        .iter()
                        .any(|&root| within_four_ulps(event.time(), root))
                );
            }
        }
    }
}
