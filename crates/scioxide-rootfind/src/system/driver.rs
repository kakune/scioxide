//! Phases shared by the Newton, Broyden and trust-region kernels.
//!
//! [`drive`] runs every dense kernel: it loads the initial guess, evaluates and
//! tests the initial residual ([`begin`]), runs the method's iteration, writes
//! the final [`Done`] state, and finally ([`conclude`]) records accepted
//! progress and turns a resource stop after an accepted point into an
//! unsuccessful result.
//! Newton and Broyden also share their full-step trial ([`full_step`]) and their
//! acceptance tests ([`settle`]).
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, Tolerance};

use super::options::SystemRootOptions;
use super::{
    SystemWorkspace, accept_trial, evaluate_system_into, finish_solve, prepare_solve, residual_ok,
};
use crate::context::Session;
use crate::nonlinear_math::{StepRounding, affine_into, difference, norm};
use crate::residual::DenseJacobian;
use crate::result::{RootTermination, SystemRootResult};

/// Terminal state of an iteration: the accepted residual norm and its termination.
pub(crate) struct Done<S> {
    pub(crate) norm: S,
    pub(crate) termination: RootTermination,
}

impl<S> Done<S> {
    pub(crate) const fn new(norm: S, termination: RootTermination) -> Self {
        Self { norm, termination }
    }
}

/// Fixed controls of one attempt, known once the initial residual is assessed.
pub(crate) struct Start<S: RealScalar> {
    /// Accepted initial residual norm.
    pub(crate) norm: S,
    /// Residual reference scale.
    pub(crate) scale: S,
    /// Residual acceptance tolerance.
    pub(crate) tolerance: Tolerance<S>,
    /// Whether the step tolerance is active.
    pub(crate) step_test: bool,
}

/// Run one dense attempt around the method's `iterate` phase.
///
/// `iterate` starts after an unconverged initial residual and returns the
/// final state. Writing the result, recording progress and converting stops
/// are common to all methods.
///
/// # Errors
/// Returns callback, numerical and accounting errors. A resource stop after an
/// accepted point writes an unsuccessful result instead.
pub(crate) fn drive<S, F>(
    f: &F,
    initial: &[S],
    opts: &SystemRootOptions<'_, S>,
    result: &mut SystemRootResult<S>,
    workspace: &mut SystemWorkspace<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
    iterate: impl FnOnce(
        &mut SystemWorkspace<S>,
        &Session<'_, '_, '_>,
        Start<S>,
    ) -> Result<Done<S>, SciOxideError>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    prepare_solve(initial, &mut workspace.state);
    let session = Session::new(context).with_policy(opts.policy);
    let mut scale = opts.residual_scale().unwrap_or(S::ONE);
    let outcome = begin(f, opts, workspace, &session, &mut scale)
        .and_then(|start| {
            if residual_ok(&start.tolerance, start.norm, start.scale)? {
                Ok(Done::new(start.norm, RootTermination::ResidualTolerance))
            } else {
                iterate(workspace, &session, start)
            }
        })
        .and_then(|done| {
            finish_solve(
                result,
                &mut workspace.state,
                done.norm,
                done.termination,
                &session,
                scale,
            )
        });
    conclude(outcome, result, workspace, &session, scale)
}

/// Evaluates the initial point and fixes the residual reference `scale`.
fn begin<S, F>(
    f: &F,
    opts: &SystemRootOptions<'_, S>,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    scale: &mut S,
) -> Result<Start<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    evaluate_system_into(
        f,
        &workspace.state.current_point,
        &mut workspace.state.current_residual,
        &mut workspace.finite_difference_residual,
        &mut workspace.state.scratch,
        session,
    )?;
    let norm = norm(workspace.state.current_residual.as_slice())?;
    *scale = opts.residual_scale().unwrap_or(norm);
    session.assess()?;
    let step = opts.step_tolerance.tolerance();
    Ok(Start {
        norm,
        scale: *scale,
        tolerance: opts.residual_tolerance.tolerance(),
        step_test: step.absolute() != S::ZERO || step.relative() != S::ZERO,
    })
}

/// Records accepted progress and writes a resource stop after an accepted point
/// as an unsuccessful result.
fn conclude<S: RealScalar + NumericScalar<Real = S>>(
    outcome: Result<(), SciOxideError>,
    result: &mut SystemRootResult<S>,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    scale: S,
) -> Result<(), SciOxideError> {
    if session.accepted.get() {
        let norm = norm(workspace.state.current_residual.as_slice())?;
        workspace.state.accepted = Some(crate::context::residual(norm, scale)?);
    }
    match (outcome, session.stopped.get(), session.accepted.get()) {
        (Err(_), Some(termination), true) => {
            let norm = norm(workspace.state.current_residual.as_slice())?;
            finish_solve(
                result,
                &mut workspace.state,
                norm,
                termination,
                session,
                scale,
            )
        }
        (outcome, _, _) => outcome,
    }
}

/// Stagnation-test scale `max(‖x‖, 1)` of the accepted point, or one when inactive.
pub(crate) fn point_scale<S: RealScalar + NumericScalar<Real = S>>(
    start: &Start<S>,
    workspace: &SystemWorkspace<S>,
) -> Result<S, SciOxideError> {
    Ok(if start.step_test {
        norm(workspace.state.current_point.as_slice())?.max(S::ONE)
    } else {
        S::ONE
    })
}

/// Evaluate the undamped trial `x + d`; `false` when it equals the accepted point.
pub(crate) fn full_step<S, F>(
    f: &F,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    affine_into(
        workspace.state.trial_point.as_mut_slice(),
        workspace.state.current_point.as_slice(),
        S::ONE,
        workspace.direction.as_slice(),
        StepRounding::Fused,
    )?;
    if workspace.state.trial_point.as_slice() == workspace.state.current_point.as_slice() {
        return Ok(false);
    }
    evaluate_system_into(
        f,
        &workspace.state.trial_point,
        &mut workspace.state.trial_residual,
        &mut workspace.line_search_residual,
        &mut workspace.state.scratch,
        session,
    )?;
    Ok(true)
}

/// Store the checked step `trial − x` and return its norm.
pub(crate) fn step_into<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
) -> Result<S, SciOxideError> {
    for ((step, &trial), &point) in workspace
        .step
        .as_mut_slice()
        .iter_mut()
        .zip(workspace.state.trial_point.as_slice())
        .zip(workspace.state.current_point.as_slice())
    {
        *step = difference(trial, point)?;
    }
    norm(workspace.step.as_slice())
}

/// Outcome of the shared acceptance tests of an evaluated trial.
pub(crate) enum Settled<S> {
    /// The trial was accepted and the attempt terminates.
    Done(Done<S>),
    /// The trial passed no terminal test; its residual norm.
    Continue(S),
}

/// Accept a trial that meets the residual tolerance, or that stagnates without
/// residual decrease (when the step test is active); otherwise continue.
pub(crate) fn settle<S: RealScalar + NumericScalar<Real = S>>(
    opts: &SystemRootOptions<'_, S>,
    start: &Start<S>,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    norm_before: S,
    point_scale: S,
    step_norm: S,
) -> Result<Settled<S>, SciOxideError> {
    let trial_norm = norm(workspace.state.trial_residual.as_slice())?;
    let termination = if residual_ok(&start.tolerance, trial_norm, start.scale)? {
        RootTermination::ResidualTolerance
    } else if start.step_test
        && opts.step_tolerance.is_satisfied(step_norm, point_scale)?
        && trial_norm >= norm_before
    {
        RootTermination::Stagnation
    } else {
        return Ok(Settled::Continue(trial_norm));
    };
    accept_trial(&mut workspace.state, session)?;
    Ok(Settled::Done(Done::new(trial_norm, termination)))
}
