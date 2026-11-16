//! Driver retry, rejection, cleanup, validation and epoch contracts observed through
//! crate-private steppers.

use alloc::rc::Rc;
use core::cell::{Cell, RefCell};

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, ConvergenceDiagnostics, ConvergenceReason, ErrorTolerance,
    ExecutionContext, NormSelection, NumericScalar, RealScalar, SciOxideError, SciOxideErrorKind,
    SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};

use super::Decay;
use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::workspace::MultistepWorkspace;
use crate::{
    Bdf, MultistepOrder, OdeMethod, OdeOptions, OdeProblem, OdeSolveReport, OdeState, OdeSystem,
    OdeTermination, OdeWorkspace,
};

fn info(name: &'static str, adaptive: bool) -> MethodInfo {
    MethodInfo {
        name,
        algorithm: scioxide_core::algorithm_id!("ode", "test-stepper", 1),
        order: 1,
        initial_step_order: 1,
        controller: DefaultController::ProportionalIntegral,
        dense_output: false,
        events: false,
        adaptive,
        mass_matrices: false,
        injectable: false,
    }
}

/// Advance `system` over `span` from `initial` on `workspace`.
fn advance_on<M, Sys>(
    method: &M,
    options: OdeOptions<f64>,
    system: &Sys,
    span: (f64, f64),
    initial: &[f64],
    workspace: &mut OdeWorkspace<f64, M>,
) -> (Result<OdeSolveReport<f64>, SciOxideError>, OdeState<f64>)
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64> + ?Sized,
{
    let plan = (method.clone(), options);
    let problem = OdeProblem::new(system, span.1);
    let mut state = OdeState::from_slice(span.0, initial).unwrap();
    let result = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut (),
        workspace,
        ExecutionContext::sequential(),
    );
    (result, state)
}

fn advance<M: OdeMethod<f64> + Clone>(
    method: &M,
    options: OdeOptions<f64>,
    span: (f64, f64),
    initial: &[f64],
) -> (Result<OdeSolveReport<f64>, SciOxideError>, OdeState<f64>) {
    let system = Decay(initial.len());
    advance_on(
        method,
        options,
        &system,
        span,
        initial,
        &mut OdeWorkspace::new(),
    )
}

/// Rejects its first attempt (error estimate or stage failure), then accepts exactly.
#[derive(Clone)]
struct Trial {
    attempts: Rc<RefCell<alloc::vec::Vec<f64>>>,
    fail: bool,
    cleanup_fails: bool,
    stage_failure: bool,
}

impl Stepper<f64, f64> for Trial {
    type Workspace = [f64; 2];
    fn info(&self) -> MethodInfo {
        info("trial", true)
    }
    fn fits(&self, _: usize, _: bool, _: &Self::Workspace) -> bool {
        true
    }
    fn reserve(&self, _: usize, _: bool, _: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn begin(&self, _: usize, _: bool, _: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn discard(&self, _: &mut Self::Workspace) -> Result<(), SciOxideError> {
        if self.cleanup_fails {
            Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(1),
                scioxide_core::Dimensions::one(2),
            ))
        } else {
            Ok(())
        }
    }
    fn step<'w, System: OdeSystem<f64> + ?Sized>(
        &self,
        _: &System,
        _: f64,
        _: &mut Vector<f64>,
        h: f64,
        _: StepRequest<'_, f64, f64>,
        w: &'w mut Self::Workspace,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, f64>, SciOxideError> {
        self.attempts.borrow_mut().push(h);
        if self.fail {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("trial"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "original failure",
            ));
        }
        let first = self.attempts.borrow().len() == 1;
        if first && self.stage_failure {
            return Err(SciOxideError::singular(None));
        }
        w.fill(if first { 4. } else { 0. });
        StepResult::new(Some(w), None)
    }
}

fn trial(fail: bool, cleanup_fails: bool, stage_failure: bool) -> Trial {
    Trial {
        attempts: Rc::new(RefCell::new(alloc::vec::Vec::new())),
        fail,
        cleanup_fails,
        stage_failure,
    }
}

#[test]
fn rejection_caps_the_retry_and_following_proposal_in_both_directions() {
    for backward in [false, true] {
        for stage_failure in [false, true] {
            let method = trial(false, false, stage_failure);
            let span = if backward { (1., 0.) } else { (0., 1.) };
            let options = OdeOptions::new()
                .with_initial_step(0.1)
                .with_error_tolerance(
                    ErrorTolerance::scalar(1., 0., NormSelection::Infinity).unwrap(),
                );
            let (report, _) = advance(&method, options, span, &[0., 1.]);
            assert_eq!(
                report.unwrap().termination(),
                Some(OdeTermination::ReachedEnd)
            );
            let attempts = method.attempts.borrow();
            assert!(attempts[1].abs() < attempts[0].abs());
            assert!(
                attempts[2].abs() <= attempts[1].abs() * (1. + 8. * f64::EPSILON),
                "{attempts:?}"
            );
        }
    }
}

#[test]
fn cleanup_failures_never_mask_integration_errors() {
    let options = || OdeOptions::new().with_initial_step(0.1);
    // Without a rejection no proposal is discarded: success needs no cleanup.
    let method = trial(false, true, true);
    method.attempts.borrow_mut().push(0.);
    let (report, state) = advance(&method, options(), (0., 0.1), &[0., 1.]);
    assert_eq!(
        report.unwrap().termination(),
        Some(OdeTermination::ReachedEnd)
    );
    assert_eq!(state.time(), 0.1);
    // A failing discard of a rejected proposal is the failure of the integration.
    for (fail, kind) in [
        (false, SciOxideErrorKind::DimensionMismatch),
        (true, SciOxideErrorKind::InvalidInput),
    ] {
        let method = trial(fail, true, false);
        let system = Decay(2);
        let mut workspace = OdeWorkspace::new();
        let (report, state) = advance_on(
            &method,
            options(),
            &system,
            (0., 0.1),
            &[0., 1.],
            &mut workspace,
        );
        assert_eq!(report.unwrap_err().kind(), kind);
        assert_eq!(state.time(), 0.);
        assert_eq!(state.values(), &[0., 1.]);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        assert_eq!(attempt.progress().unwrap().time(), 0.);
    }
}

#[derive(Clone, Copy)]
enum InvalidStateKind {
    WrongShape,
    NonFinite,
}

#[derive(Clone)]
struct InvalidStateSolver {
    kind: InvalidStateKind,
}

impl Stepper<f64, f64> for InvalidStateSolver {
    type Workspace = ();
    fn info(&self) -> MethodInfo {
        info("InvalidStateSolver", false)
    }
    fn fits(&self, _: usize, _: bool, (): &()) -> bool {
        true
    }
    fn reserve(&self, _: usize, _: bool, (): &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn begin(&self, _: usize, _: bool, (): &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn step<'w, System: OdeSystem<f64> + ?Sized>(
        &self,
        _: &System,
        _: f64,
        state: &mut Vector<f64>,
        _: f64,
        _: StepRequest<'_, f64, f64>,
        _: &'w mut (),
        _: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, f64>, SciOxideError> {
        *state = match self.kind {
            InvalidStateKind::WrongShape => Vector::from_vec(vec![0.0, 1.0]),
            InvalidStateKind::NonFinite => Vector::from_vec(vec![f64::NAN]),
        };
        StepResult::new(None, None)
    }
}

#[test]
fn malformed_method_workspaces_are_rejected() {
    for kind in [InvalidStateKind::WrongShape, InvalidStateKind::NonFinite] {
        let (result, state) = advance(
            &InvalidStateSolver { kind },
            OdeOptions::new().with_initial_step(1.0),
            (0.0, 1.0),
            &[0.0],
        );
        assert!(result.is_err_and(|error| matches!(
            error.kind(),
            SciOxideErrorKind::DimensionMismatch | SciOxideErrorKind::NonFinite
        )));
        assert_eq!((state.time(), state.values()), (0.0, &[0.0][..]));
    }
}

#[derive(Clone, Copy)]
enum InvalidErrorKind {
    Infinite,
    NonFinite,
}

#[derive(Clone)]
struct InvalidErrorSolver {
    kind: InvalidErrorKind,
}

impl Stepper<f64, f64> for InvalidErrorSolver {
    type Workspace = ();
    fn info(&self) -> MethodInfo {
        info("InvalidErrorSolver", true)
    }
    fn fits(&self, _: usize, _: bool, (): &()) -> bool {
        true
    }
    fn reserve(&self, _: usize, _: bool, (): &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn begin(&self, _: usize, _: bool, (): &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn step<'w, System: OdeSystem<f64> + ?Sized>(
        &self,
        _: &System,
        _: f64,
        state: &mut Vector<f64>,
        dt: f64,
        _: StepRequest<'_, f64, f64>,
        _: &'w mut (),
        _: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, f64>, SciOxideError> {
        state.as_mut_slice()[0] += dt;
        StepResult::new(
            Some(match self.kind {
                InvalidErrorKind::Infinite => &[f64::INFINITY],
                InvalidErrorKind::NonFinite => &[f64::NAN],
            }),
            None,
        )
    }
}

#[test]
fn invalid_custom_error_has_empty_prepared_and_owned_parity() {
    for kind in [InvalidErrorKind::Infinite, InvalidErrorKind::NonFinite] {
        let method = InvalidErrorSolver { kind };
        let options = || OdeOptions::new().with_initial_step(1.0);
        let (empty, _) = advance(&method, options(), (0.0, 1.0), &[0.0]);
        let empty = empty.unwrap_err();

        let plan = (method.clone(), options());
        let system = Decay(1);
        let problem = OdeProblem::new(&system, 1.0);
        let mut prepared = plan.0.workspace(&problem, &plan.1).unwrap();
        let (workspace_error, _) = advance_on(
            &method,
            options(),
            &system,
            (0.0, 1.0),
            &[0.0],
            &mut prepared,
        );
        let workspace_error = workspace_error.unwrap_err();

        let owned = plan
            .0
            .solve(
                &problem,
                &plan.1,
                OdeState::from_slice(0.0, &[0.0]).unwrap(),
            )
            .unwrap_err();

        assert_eq!(empty.kind(), workspace_error.kind());
        assert_eq!(empty.kind(), owned.error().kind());
        assert_eq!(owned.progress().unwrap().state(), &[0.0]);
    }
}

#[derive(Clone)]
struct RepairableShape {
    first: Cell<bool>,
}

impl Stepper<f64, f64> for RepairableShape {
    type Workspace = ();
    fn info(&self) -> MethodInfo {
        info("RepairableShape", false)
    }
    fn fits(&self, _: usize, _: bool, (): &()) -> bool {
        true
    }
    fn reserve(&self, _: usize, _: bool, (): &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn begin(&self, _: usize, _: bool, (): &mut ()) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn step<'w, System: OdeSystem<f64> + ?Sized>(
        &self,
        _: &System,
        _: f64,
        state: &mut Vector<f64>,
        step: f64,
        _: StepRequest<'_, f64, f64>,
        _: &'w mut (),
        _: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, f64>, SciOxideError> {
        if self.first.replace(false) {
            *state = Vector::from_vec(vec![0.0; 2]);
        } else {
            state.as_mut_slice()[0] += step;
        }
        StepResult::new(None, None)
    }
}

#[test]
fn a_method_workspace_shape_failure_does_not_corrupt_the_next_epoch() {
    let plan = (
        RepairableShape {
            first: Cell::new(true),
        },
        OdeOptions::new().with_initial_step(1.0),
    );
    let system = Decay(1);
    let problem = OdeProblem::new(&system, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0.0, &[0.0]).unwrap();
    let failure = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(failure.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(state.values(), &[0.0]);
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
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!((state.values()[0] - 1.0).abs() < f64::EPSILON);
}

/// Reports fixed signed component errors without changing the state.
#[derive(Clone)]
struct RawErrors<S> {
    errors: [S; 2],
}

impl<S: RealScalar + NumericScalar<Real = S>> Stepper<S, S> for RawErrors<S> {
    type Workspace = Option<[S; 2]>;
    fn info(&self) -> MethodInfo {
        info("RawErrors", true)
    }
    fn fits(&self, _: usize, _: bool, _: &Self::Workspace) -> bool {
        true
    }
    fn reserve(&self, _: usize, _: bool, _: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn begin(&self, _: usize, _: bool, _: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn step<'w, System: OdeSystem<S> + ?Sized>(
        &self,
        _: &System,
        _: S,
        _: &mut Vector<S>,
        _: S,
        _: StepRequest<'_, S, S>,
        w: &'w mut Self::Workspace,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S>, SciOxideError> {
        StepResult::new(Some(w.insert(self.errors)), None)
    }
}

struct Constant;

impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Constant {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        _: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(S::ZERO);
        Ok(())
    }
}

fn component_norms<S: RealScalar + NumericScalar<Real = S>>() {
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    for (norm, expected) in [
        (NormSelection::L1, 0.75),
        (NormSelection::L2, 0.3125_f64.sqrt()),
        (NormSelection::Infinity, 0.5),
        (NormSelection::RootMeanSquare, 0.15625_f64.sqrt()),
    ] {
        let options = OdeOptions::new()
            .with_initial_step(scalar(0.25))
            .with_error_tolerance(
                ErrorTolerance::try_vector(&[scalar(1.0), scalar(10.0)], scalar(0.5), norm)
                    .unwrap(),
            );
        let plan = (
            RawErrors {
                errors: [scalar(-1.0), scalar(5.0)],
            },
            options,
        );
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&Constant, scalar(0.25)),
                &plan.1,
                OdeState::from_slice(scalar(0.0), &[scalar(2.0), scalar(20.0)]).unwrap(),
            )
            .unwrap();
        assert!(
            NumericScalar::abs(
                solution.report().last_step_error_estimate().unwrap() - scalar(expected)
            ) <= scalar(8.0) * S::EPSILON
        );
    }
}

#[test]
fn component_error_norms_f32() {
    component_norms::<f32>();
}

#[test]
fn component_error_norms_f64() {
    component_norms::<f64>();
}

/// `y' = -1000 y` with its analytic Jacobian.
struct Stiff;

impl OdeSystem<f64> for Stiff {
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
        out[0] = -1000. * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -1000.)?;
        Ok(Some(()))
    }
}

/// Fails its first (or every) trial with a numerical stage failure, otherwise BDF.
#[derive(Clone)]
struct FailTrial {
    inner: Bdf<f64>,
    reason: Option<ConvergenceReason>,
    minimum: bool,
    failed: Cell<bool>,
}

fn fail_trial(reason: Option<ConvergenceReason>, minimum: bool) -> FailTrial {
    FailTrial {
        inner: Bdf::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        reason,
        minimum,
        failed: Cell::new(false),
    }
}

impl Stepper<f64, f64> for FailTrial {
    type Workspace = MultistepWorkspace<f64>;
    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "FailTrial",
            algorithm: scioxide_core::algorithm_id!("ode", "recovery-test", 1),
            ..Stepper::<f64, f64>::info(&self.inner)
        }
    }
    fn fits(&self, n: usize, provided: bool, w: &Self::Workspace) -> bool {
        Stepper::<f64, f64>::fits(&self.inner, n, provided, w)
    }
    fn reserve(
        &self,
        n: usize,
        provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        Stepper::<f64, f64>::reserve(&self.inner, n, provided, w)
    }
    fn begin(
        &self,
        n: usize,
        provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        Stepper::<f64, f64>::begin(&self.inner, n, provided, w)
    }
    fn accept(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Stepper::<f64, f64>::accept(&self.inner, w)
    }
    fn discard(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Stepper::<f64, f64>::discard(&self.inner, w)
    }
    fn step<'w, System: OdeSystem<f64> + ?Sized>(
        &self,
        sys: &System,
        t: f64,
        y: &mut Vector<f64>,
        h: f64,
        request: StepRequest<'_, f64, f64>,
        w: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, f64>, SciOxideError> {
        if self.minimum || !self.failed.replace(true) {
            // Failed proposals may dirty their private candidate, never accepted output.
            y.as_mut_slice()[0] = 123.;
            return Err(self.reason.map_or_else(
                || SciOxideError::singular(None),
                |reason| {
                    SciOxideError::convergence_failure(
                        reason,
                        ConvergenceDiagnostics::new(),
                        "injected trial failure",
                    )
                },
            ));
        }
        self.inner.step(sys, t, y, h, request, w, cx)
    }
}

#[test]
fn numerical_trial_failures_reject_and_continue() {
    for reason in [
        None,
        Some(ConvergenceReason::IterationLimit),
        Some(ConvergenceReason::Divergence),
        Some(ConvergenceReason::Stagnation),
        Some(ConvergenceReason::NestedSolver),
    ] {
        let options = OdeOptions::new()
            .with_initial_step(0.001)
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-10, 1e-6, NormSelection::RootMeanSquare).unwrap(),
            );
        let (report, state) = advance_on(
            &fail_trial(reason, false),
            options,
            &Stiff,
            (0., 0.01),
            &[1.],
            &mut OdeWorkspace::new(),
        );
        let report = report.unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert!(report.counts().rejected() >= 1);
        assert_eq!(
            report.counts().proposals(),
            report.counts().accepted() + report.counts().rejected()
        );
        assert!((state.values()[0] - (-10_f64).exp()).abs() < 1e-8);
    }
}

#[test]
fn minimum_step_retains_the_accepted_state() {
    let options = OdeOptions::new()
        .with_initial_step(0.1)
        .with_minimum_step(0.025);
    let (report, state) = advance_on(
        &fail_trial(Some(ConvergenceReason::Divergence), true),
        options,
        &Stiff,
        (0., 1.),
        &[1.],
        &mut OdeWorkspace::new(),
    );
    let report = report.unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::StepSizeLimit));
    assert_eq!(report.final_time(), 0.);
    assert_eq!(state.values(), &[1.]);
    assert_eq!(report.counts().rejected(), 3);
}

#[test]
fn failed_trials_exhaust_the_proposal_budget() {
    let options = OdeOptions::new()
        .with_initial_step(0.1)
        .with_max_steps(Budget::try_from(3).unwrap());
    let (report, state) = advance_on(
        &fail_trial(Some(ConvergenceReason::Stagnation), true),
        options,
        &Stiff,
        (0., 1.),
        &[1.],
        &mut OdeWorkspace::new(),
    );
    let report = report.unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!(report.counts().proposals(), 3);
    assert_eq!(report.counts().rejected(), 3);
    assert_eq!(state.values(), &[1.]);
}
