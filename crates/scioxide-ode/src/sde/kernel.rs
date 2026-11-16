//! The SDE stepping loop: work accounting, refinement traversal and explicit stages.

use super::workspace::{Buffers, StepScratch};
use super::{SdeMethod, SdeProblem, SdeReport, SdeRun, SdeSystem};
use crate::safety::checkpoint as check_stop;
use crate::safety::{invalid, stage_affine, state_difference};
use crate::termination::Stop;
use crate::termination::{Fault, Work, admit, record};
use crate::{OdeOutputSink, OdeState};
use scioxide_core::backend::checked_add;
use scioxide_core::{
    EvaluationRole, NonFiniteOrigin, NumericScalar, PartialResultTrust, RealScalar, SciOxideError,
    SolveContext,
};
use scioxide_core::{checked_difference, checked_sum};
use scioxide_linalg::MatrixView;

type RunResult<T> = Result<T, Fault>;

/// Integrate every coarse interval, writing each accepted point into `state`; `Some`
/// is a step-size stop of refinement.
pub(super) fn run<S, T, K>(
    plan: &SdeRun<'_, S>,
    problem: &SdeProblem<'_, T>,
    state: &mut OdeState<S, T>,
    sink: &mut K,
    buffers: &mut Buffers<T>,
    report: &mut SdeReport<S>,
    cx: &mut SolveContext<'_, '_>,
) -> RunResult<Option<Stop>>
where
    S: RealScalar,
    T: NumericScalar<Real = S>,
    K: OdeOutputSink<S, T> + ?Sized,
{
    let system = problem.system();
    let increments = problem.increments();
    let tolerance = plan.options.refinement();
    buffers.shape(plan.dimension, plan.noise_dimension);
    let Buffers {
        full,
        half,
        candidate,
        errors,
        scales,
        pending,
        scratch,
    } = buffers;
    let (time, values) = state.parts_mut();
    cx.assess_accepted(PartialResultTrust::Admissible)?;
    check_stop(cx)?;
    for pair in plan.boundaries.intervals() {
        pending.push((pair[0], pair[1]));
        while let Some((left, right)) = pending.pop() {
            check_stop(cx)?;
            if tolerance.is_some() && right - left < 2 {
                return Ok(Some(Stop::StepSize));
            }
            record(cx, Work::Step)?;
            let accepted = values.as_slice();
            step(
                plan, system, increments, left, right, accepted, full, scratch, cx,
            )?;
            let proposal = if let Some(tolerance) = tolerance {
                let mid = usize::midpoint(left, right);
                step(
                    plan, system, increments, left, mid, accepted, half, scratch, cx,
                )?;
                step(
                    plan, system, increments, mid, right, half, candidate, scratch, cx,
                )?;
                for (i, ((error, scale), (&fine, &coarse))) in errors
                    .iter_mut()
                    .zip(scales.iter_mut())
                    .zip(candidate.iter().zip(full.iter()))
                    .enumerate()
                {
                    *error = state_difference(fine, coarse)?.abs();
                    *scale = fine.abs().max(accepted[i].abs());
                }
                let error = tolerance.normalized_norm(errors, scales)?;
                report.last_step_error_estimate = Some(error);
                if error > S::ONE {
                    cx.record_rejected()?;
                    // The prepared capacity is a proven bound for this depth-first traversal.
                    if pending.len() + 2 > plan.stack_capacity {
                        return Err(invalid(
                            scioxide_core::argument!("self"),
                            "SDE refinement stack capacity exceeded",
                        )
                        .into());
                    }
                    pending.push((mid, right));
                    pending.push((left, mid));
                    continue;
                }
                &*candidate
            } else {
                &*full
            };
            check_stop(cx)?;
            let steps = checked_add(report.steps, 1, scioxide_core::quantity!("accepted_steps"))?;
            cx.commit_accepted(PartialResultTrust::Admissible)?;
            values.as_mut_slice().copy_from_slice(proposal);
            *time = plan.times[right];
            report.final_time = *time;
            report.grid_index = right;
            report.steps = steps;
            sink.emit(*time, values.as_slice())?;
        }
    }
    Ok(None)
}

// Recompute from immutable leaves with the same midpoint reduction tree. No
// prefix subtraction or path cache can lose small increments against a large path.
fn increment<S: RealScalar>(
    values: MatrixView<'_, S>,
    left: usize,
    right: usize,
    column: usize,
) -> Result<S, SciOxideError> {
    if right - left == 1 {
        return values.get(left, column);
    }
    let mid = usize::midpoint(left, right);
    checked_sum(&[
        increment(values, left, mid, column)?,
        increment(values, mid, right, column)?,
    ])
}

#[allow(
    clippy::too_many_arguments,
    reason = "explicit stage inputs and storage"
)]
fn step<S: RealScalar, T: NumericScalar<Real = S>>(
    plan: &SdeRun<'_, S>,
    system: &dyn SdeSystem<T>,
    increments: MatrixView<'_, S>,
    left: usize,
    right: usize,
    state: &[T],
    output: &mut [T],
    scratch: &mut StepScratch<T>,
    cx: &mut SolveContext<'_, '_>,
) -> RunResult<()> {
    if state.is_empty() {
        return Ok(());
    }
    let time = plan.times[left];
    let end = plan.times[right];
    let h = checked_difference(end, time)?;
    for (j, out) in scratch.increment.iter_mut().enumerate() {
        *out = increment(increments, left, right, j)?;
    }
    coefficients(
        system,
        time,
        state,
        &scratch.increment,
        &mut scratch.drift,
        &mut scratch.diffusion,
        cx,
    )?;
    for (i, out) in output.iter_mut().enumerate() {
        *out = stage_affine(
            state[i],
            S::ONE,
            [(h, scratch.drift[i]), (S::ONE, scratch.diffusion[i])],
        )?;
    }
    if plan.method == SdeMethod::StochasticHeun {
        scratch.predictor.copy_from_slice(output);
        coefficients(
            system,
            end,
            &scratch.predictor,
            &scratch.increment,
            &mut scratch.next_drift,
            &mut scratch.next_diffusion,
            cx,
        )?;
        let half = S::HALF;
        for (i, out) in output.iter_mut().enumerate() {
            // Put 1/2 outside the extended-range sum so subnormal h is retained.
            *out = stage_affine(
                state[i],
                half,
                [
                    (h, scratch.drift[i]),
                    (h, scratch.next_drift[i]),
                    (S::ONE, scratch.diffusion[i]),
                    (S::ONE, scratch.next_diffusion[i]),
                ],
            )?;
        }
    }
    crate::safety::finite(output, NonFiniteOrigin::Intermediate)?;
    Ok(())
}

fn coefficients<T: NumericScalar>(
    system: &dyn SdeSystem<T>,
    time: T::Real,
    state: &[T],
    increment: &[T::Real],
    drift: &mut [T],
    diffusion: &mut [T],
    cx: &mut SolveContext<'_, '_>,
) -> RunResult<()> {
    drift.fill(T::from_real(T::Real::NAN));
    admit(cx, EvaluationRole::RightHandSide)?;
    system.drift(time, state, drift, cx)?;
    crate::safety::finite(drift, NonFiniteOrigin::Callback)?;
    if increment.is_empty() {
        diffusion.fill(T::ZERO);
    } else {
        diffusion.fill(T::from_real(T::Real::NAN));
        admit(cx, EvaluationRole::LinearOperator)?;
        system.diffusion_product(time, state, increment, diffusion, cx)?;
        crate::safety::finite(diffusion, NonFiniteOrigin::Callback)?;
    }
    Ok(())
}
