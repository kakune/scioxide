//! Dynamic system step limits through the public plan API.

use alloc::vec::Vec;
use core::cell::Cell;

use scioxide_core::{AttemptOutcome, ExecutionContext, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

use crate::{
    DormandPrince5, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeSystem,
    OdeWorkspace,
};

struct Limited {
    cap: f64,
    calls: Cell<usize>,
}

impl OdeSystem<f64> for Limited {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = 1.;
        Ok(())
    }
    fn step_limit(
        &self,
        t: f64,
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<f64>, SciOxideError> {
        Ok(Some(if t < 0.1 { self.cap } else { self.cap / 2. }))
    }
}

#[test]
fn dynamic_cap_limits_first_step_growth_and_backward_steps() {
    for span in [(0., 0.2), (0.2, 0.)] {
        let system = Limited {
            cap: 0.04,
            calls: Cell::new(0),
        };
        let plan = (
            DormandPrince5::new(),
            OdeOptions::new()
                .with_initial_step(1.)
                .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
        );
        let problem = OdeProblem::new(&system, span.1);
        let mut state = OdeState::new(span.0, Vector::from_vec(vec![span.0]));
        let mut previous = span.0;
        let mut sink = |t: f64, _: &[f64]| {
            let cap = if previous < 0.1 { 0.04 } else { 0.02 };
            assert!((t - previous).abs() <= cap + 1e-15);
            previous = t;
            Ok::<(), SciOxideError>(())
        };
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut sink,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!((state.values()[0] - span.1).abs() < 1e-14);
        assert_eq!(report.accepted_steps(), report.counts().accepted());
    }
}

#[test]
fn invalid_cap_never_invokes_a_stage() {
    for cap in [0., -1., f64::INFINITY, f64::NAN] {
        let system = Limited {
            cap,
            calls: Cell::new(0),
        };
        let plan = (
            DormandPrince5::new(),
            OdeOptions::new().with_initial_step(0.1),
        );
        let problem = OdeProblem::new(&system, 1.);
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let mut state = OdeState::new(0., Vector::from_vec(vec![0.]));
        assert!(
            plan.0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(system.calls.get(), 0);
        assert_eq!(state.time(), 0.);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        assert_eq!(attempt.progress().unwrap().time(), 0.);
    }
}

struct Combined {
    rate: f64,
    cap: Option<f64>,
    base_calls: Cell<usize>,
    stage_calls: Cell<usize>,
    incomplete: bool,
}
impl OdeSystem<f64> for Combined {
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
        self.stage_calls.set(self.stage_calls.get() + 1);
        out[0] = self.rate * y[0];
        Ok(())
    }
    fn rhs_with_step_limit(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<crate::OdeRhsStepLimit<f64>, SciOxideError> {
        cx.record_provider(scioxide_core::EvaluationRole::RightHandSide)?
            .require()?;
        self.base_calls.set(self.base_calls.get() + 1);
        if !self.incomplete {
            out[0] = self.rate * y[0];
        }
        Ok(crate::OdeRhsStepLimit::Evaluated {
            maximum_step: self.cap,
        })
    }
    fn admits(
        &self,
        _: f64,
        y: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        Ok(y[0] >= 0.)
    }
}
fn combined(rate: f64, cap: Option<f64>) -> Combined {
    Combined {
        rate,
        cap,
        base_calls: Cell::new(0),
        stage_calls: Cell::new(0),
        incomplete: false,
    }
}

#[test]
fn combined_rhs_counts_stages_and_resets_between_epochs() {
    use crate::{BuiltinOdeMethod, ClassicalRk4, ForwardEuler, Heun};
    use scioxide_core::EvaluationRole;
    for (method, stages) in [
        (BuiltinOdeMethod::from(ForwardEuler::new()), 1),
        (Heun::new().into(), 2),
        (ClassicalRk4::new().into(), 4),
    ] {
        let system = combined(-1., None);
        let h: f64 = 0.125;
        let factor = match stages {
            1 => 1. - h,
            2 => 1. - h + h * h / 2.,
            _ => 1. - h + h * h / 2. - h.powi(3) / 6. + h.powi(4) / 24.,
        };
        let plan = (method, OdeOptions::new().with_initial_step(0.125));
        let problem = OdeProblem::new(&system, 0.25);
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        for initial in [1., 2.] {
            let mut state = OdeState::new(0., Vector::from_vec(vec![initial]));
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(report.accepted_steps(), 2);
            assert_eq!(
                report.counts().provider(EvaluationRole::RightHandSide),
                2 * stages
            );
            assert_eq!(report.counts().provider(EvaluationRole::Function), 0);
            assert!((state.values()[0] - initial * factor * factor).abs() < 1e-14);
        }
        assert_eq!(system.base_calls.get(), 4);
        assert_eq!(system.stage_calls.get(), 4 * (stages - 1));
    }
}

#[test]
fn combined_rhs_reuses_only_the_accepted_base_on_rejection() {
    let system = combined(-10., Some(0.25));
    let plan = (
        crate::Heun::new(),
        OdeOptions::new().with_initial_step(0.25),
    );
    let problem = OdeProblem::new(&system, 0.25);
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!(report.rejected_steps() >= 2);
    assert_eq!(system.base_calls.get(), report.accepted_steps());
    assert_eq!(system.stage_calls.get(), report.accepted_steps());
    assert!(report.counts().cache_hits() >= 2);
    assert_eq!(
        report
            .counts()
            .provider(scioxide_core::EvaluationRole::RightHandSide),
        2 * report.accepted_steps()
    );
    assert!(state.values()[0] > 0.);
}

#[test]
fn combined_rhs_validates_limits_output_and_admission_before_publication() {
    for (cap, incomplete) in [(Some(0.), false), (Some(f64::NAN), false), (None, true)] {
        let mut system = combined(-1., cap);
        system.incomplete = incomplete;
        let plan = (
            crate::Heun::new(),
            OdeOptions::new().with_initial_step(0.125),
        );
        let problem = OdeProblem::new(&system, 0.25);
        let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
        assert!(
            plan.0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut plan.0.workspace(&problem, &plan.1).unwrap(),
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(system.stage_calls.get(), 0);
        assert_eq!(state.time(), 0.);
        assert_eq!(state.values(), [1.]);
    }
    let budget = 1;
    let system = combined(-1., Some(0.125));
    let plan = (
        crate::Heun::new(),
        OdeOptions::new()
            .with_initial_step(0.125)
            .with_max_evaluations(scioxide_core::Budget::try_from(budget).unwrap()),
    );
    let problem = OdeProblem::new(&system, 0.25);
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut plan.0.workspace(&problem, &plan.1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(crate::OdeTermination::BudgetLimit(
            scioxide_core::BudgetKind::Evaluation
        ))
    );
    assert_eq!(system.base_calls.get(), budget);
    assert_eq!(system.stage_calls.get(), 0);
    assert_eq!(state.time(), 0.);
    assert_eq!(state.values(), [1.]);
}

#[test]
fn fixed_steps_absorb_the_rounding_of_accumulated_times() {
    // Ten sums of 0.01 give 0.09999999999999999, one unit in the last place below 0.1.
    for (end, step, steps) in [
        (0.1, 0.01, 10),
        (1.0, 0.1, 10),
        (0.7, 0.07, 10),
        (0.25, 0.1, 3),
    ] {
        let system = Limited {
            cap: 10.0,
            calls: Cell::new(0),
        };
        let options = OdeOptions::new()
            .with_initial_step(step)
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps);
        let mut times = Vec::new();
        let mut sink = |t: f64, _: &[f64]| {
            times.push(t);
            Ok::<(), SciOxideError>(())
        };
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
        let report = crate::ClassicalRk4::new()
            .advance(
                &OdeProblem::new(&system, end),
                &options,
                &mut state,
                &mut sink,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(report.accepted_steps(), steps);
        assert_eq!(state.time(), end);
        // The last step absorbs the rounding; only a genuine remainder is shorter.
        let last = times[times.len() - 1] - times[times.len() - 2];
        let expected = if end == 0.25 { 0.05 } else { step };
        assert!((last - expected).abs() <= 1e-15);
    }
}
