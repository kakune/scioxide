//! Checkpointed loops recorded as one call statement.

use alloc::sync::Arc;
use alloc::vec::Vec;

use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

use super::call::{Call, Rule};
use super::solve::common;
use super::{Tape, TapeOptions, TapeValue, Var, word};
use crate::real::sealed::{Loop, Token};
use crate::rules::Class;
use crate::{CheckpointAction, CheckpointSchedule, Dual, LoopBody, Real};

/// A loop body with its number type erased, so a call statement can keep it.
pub(crate) trait ErasedLoop<V: TapeValue>: Send + Sync {
    /// Advances `state` by step `index` on values.
    fn advance(&self, index: usize, state: &mut [V], parameters: &[V])
    -> Result<(), SciOxideError>;
    /// Advances `state` by step `index` on duals.
    fn advance_dual(
        &self,
        index: usize,
        state: &mut [Dual<V>],
        parameters: &[Dual<V>],
    ) -> Result<(), SciOxideError>;
    /// Replaces `adjoint` by the pullback of step `index` at `state` and adds the
    /// parameter part to `parameter_adjoint`, recording the step on `tape`.
    #[allow(clippy::too_many_arguments)]
    fn pull_step(
        &self,
        tape: &mut Tape<V>,
        index: usize,
        point: &mut [V],
        gradient: &mut [V],
        adjoint: &mut [V],
        parameter_adjoint: &mut [V],
    ) -> Result<(), SciOxideError>;
}

impl<S, V, B> ErasedLoop<V> for B
where
    S: RealScalar,
    V: TapeValue<Scalar = S>,
    B: LoopBody<S> + Send + Sync + 'static,
{
    fn advance(
        &self,
        index: usize,
        state: &mut [V],
        parameters: &[V],
    ) -> Result<(), SciOxideError> {
        self.step(index, state, parameters)
    }

    fn advance_dual(
        &self,
        index: usize,
        state: &mut [Dual<V>],
        parameters: &[Dual<V>],
    ) -> Result<(), SciOxideError> {
        self.step(index, state, parameters)
    }

    fn pull_step(
        &self,
        tape: &mut Tape<V>,
        index: usize,
        point: &mut [V],
        gradient: &mut [V],
        adjoint: &mut [V],
        parameter_adjoint: &mut [V],
    ) -> Result<(), SciOxideError> {
        let n = adjoint.len();
        let cx = ExecutionContext::sequential();
        let mut linear = tape.linearize(
            |x, y| {
                let (state, parameters) = x.split_at(n);
                y.extend_from_slice(state);
                self.step(index, y, parameters)
            },
            point,
            cx,
        )?;
        linear.vjp_into(adjoint, gradient, cx)?;
        adjoint.copy_from_slice(&gradient[..n]);
        for (target, &g) in parameter_adjoint.iter_mut().zip(&gradient[n..]) {
            *target += g;
        }
        Ok(())
    }
}

fn buffer<T: Clone>(len: usize, fill: T) -> Result<Vec<T>, SciOxideError> {
    scioxide_core::backend::try_filled(len, fill, scioxide_core::quantity!("loop_scratch"))
}

/// Adds the pullback of the whole loop: `(s̄₀, p̄) += Jᵀ·s̄_N`.
pub(crate) fn pullback<V: TapeValue>(
    body: &Arc<dyn ErasedLoop<V>>,
    steps: usize,
    schedule: &CheckpointSchedule,
    inputs: &[V],
    output_adjoints: &[V],
    input_adjoints: &mut [V],
) -> Result<(), SciOxideError> {
    let mut tape = Tape::new(TapeOptions::new());
    pullback_on(
        &mut tape,
        body,
        steps,
        schedule,
        inputs,
        output_adjoints,
        input_adjoints,
    )
}

/// [`pullback`] with the inner tape that records one step at a time.
fn pullback_on<V: TapeValue>(
    tape: &mut Tape<V>,
    body: &Arc<dyn ErasedLoop<V>>,
    steps: usize,
    schedule: &CheckpointSchedule,
    inputs: &[V],
    output_adjoints: &[V],
    input_adjoints: &mut [V],
) -> Result<(), SciOxideError> {
    let n = output_adjoints.len();
    let q = inputs.len() - n;
    let (initial, parameters) = inputs.split_at(n);
    let zero = V::zero();
    let snapshot_len = scioxide_core::backend::checked_mul(
        schedule.checkpoint_slots() + 1,
        n,
        scioxide_core::quantity!("loop_scratch"),
    )?;
    let mut snapshots = buffer(snapshot_len, zero)?;
    snapshots[..n].copy_from_slice(initial);
    let mut control = buffer(schedule.control_slots(), 0_u64)?;
    let mut current = buffer(n, zero)?;
    let mut before = buffer(n, zero)?;
    let mut adjoint = buffer(n, zero)?;
    adjoint.copy_from_slice(output_adjoints);
    let mut parameter_adjoint = buffer(q, zero)?;
    let mut point = buffer(n + q, zero)?;
    let mut gradient = buffer(n + q, zero)?;
    // An explicit schedule restores the states at its boundaries but emits no `Save`:
    // its consumer stores them on the forward pass. The loop ran on values when it
    // was recorded, so store them here with one pass up to the last boundary.
    if let Some(boundaries) = schedule.boundaries() {
        current.copy_from_slice(initial);
        let mut reached = 0;
        for (slot, &boundary) in boundaries.iter().enumerate().skip(1) {
            for index in reached..boundary {
                body.advance(index, &mut current, parameters)?;
            }
            reached = boundary;
            snapshots[slot * n..(slot + 1) * n].copy_from_slice(&current);
        }
    }
    schedule.visit(steps, &mut control, |action| {
        match action {
            CheckpointAction::Restore { slot, .. } => {
                current.copy_from_slice(&snapshots[slot * n..(slot + 1) * n]);
            }
            CheckpointAction::Advance { start, end } => {
                for index in start..end {
                    if index + 1 == end {
                        before.copy_from_slice(&current);
                    }
                    body.advance(index, &mut current, parameters)?;
                }
            }
            CheckpointAction::Save { slot, .. } => {
                snapshots[slot * n..(slot + 1) * n].copy_from_slice(&current);
            }
            CheckpointAction::Reverse { step } => {
                point[..n].copy_from_slice(&before);
                point[n..].copy_from_slice(parameters);
                body.pull_step(
                    tape,
                    step,
                    &mut point,
                    &mut gradient,
                    &mut adjoint,
                    &mut parameter_adjoint,
                )?;
            }
        }
        Ok(())
    })?;
    let (state_adjoint, rest) = input_adjoints.split_at_mut(n);
    for (target, &a) in state_adjoint.iter_mut().zip(&adjoint) {
        *target += a;
    }
    for (target, &a) in rest.iter_mut().zip(&parameter_adjoint) {
        *target += a;
    }
    Ok(())
}

/// Writes the tangent of the final state by running the loop on duals.
pub(crate) fn pushforward<V: TapeValue>(
    body: &Arc<dyn ErasedLoop<V>>,
    steps: usize,
    inputs: &[V],
    input_tangents: &[V],
    output_tangents: &mut [V],
) -> Result<(), SciOxideError> {
    let n = output_tangents.len();
    let mut duals = buffer(inputs.len(), Dual::new(V::zero(), V::zero()))?;
    for (target, (&v, &t)) in duals.iter_mut().zip(inputs.iter().zip(input_tangents)) {
        *target = Dual::new(v, t);
    }
    let (state, parameters) = duals.split_at_mut(n);
    for index in 0..steps {
        body.advance_dual(index, state, parameters)?;
    }
    for (target, value) in output_tangents.iter_mut().zip(state.iter()) {
        *target = value.derivative();
    }
    Ok(())
}

impl<S: RealScalar, V: TapeValue<Scalar = S>> Loop<S> for Var<'_, V> {
    fn run_loop<B: LoopBody<S> + Send + Sync + 'static>(
        _: Token,
        body: &Arc<B>,
        steps: usize,
        state: &mut [Self],
        parameters: &[Self],
        schedule: &CheckpointSchedule,
    ) -> Result<(), SciOxideError> {
        let (n, q) = (state.len(), parameters.len());
        let mut values = buffer(n, V::zero())?;
        for (target, variable) in values.iter_mut().zip(state.iter()) {
            *target = variable.value();
        }
        let mut initial = buffer(n, V::zero())?;
        initial.copy_from_slice(&values);
        let mut parameter_values = buffer(q, V::zero())?;
        for (target, parameter) in parameter_values.iter_mut().zip(parameters) {
            *target = parameter.value();
        }
        for index in 0..steps {
            body.step(index, &mut values, &parameter_values)?;
        }
        let Ok(recorder) = common(state.iter().chain(parameters).copied()) else {
            state.fill(Var::constant(S::NAN));
            return Ok(());
        };
        let Some(r) = recorder else {
            for (target, &value) in state.iter_mut().zip(&values) {
                *target = Var::lift(value);
            }
            return Ok(());
        };
        let saved = r.full_position();
        initial
            .iter()
            .chain(&parameter_values)
            .chain(&values)
            .for_each(|&v| r.push_full(v));
        let erased: Arc<dyn ErasedLoop<V>> = Arc::clone(body) as Arc<dyn ErasedLoop<V>>;
        let index = r.push_call(Call {
            rule: Rule::Loop {
                body: erased,
                steps,
                schedule: schedule.try_clone()?,
            },
            inputs: n + q,
            outputs: n,
            saved,
        });
        let arguments = state.iter().chain(parameters).map(|v| (v.slot(), None));
        r.extended(Some(index), Class::Nonlinear, n + q, arguments);
        for (target, &value) in state.iter_mut().zip(&values) {
            let slot = r.push_word(word::CALL_OUTPUT);
            *target = Var::active(value, slot, r);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// `s ← s·(1 + p)` per step.
    struct Growth;

    impl LoopBody for Growth {
        fn step<R: Real>(&self, _: usize, s: &mut [R], p: &[R]) -> Result<(), SciOxideError> {
            s[0] *= p[0] + 1.0;
            s[1] += s[0] * p[0];
            Ok(())
        }
    }

    #[test]
    fn the_inner_tape_holds_one_step() {
        let body: Arc<dyn ErasedLoop<f64>> = Arc::new(Growth);
        let mut one = Tape::new(TapeOptions::new());
        let schedule = CheckpointSchedule::binomial(1, 0).unwrap();
        let mut adjoints = [0.0; 3];
        pullback_on(
            &mut one,
            &body,
            1,
            &schedule,
            &[1.0, 0.0, 0.1],
            &[1.0, 1.0],
            &mut adjoints,
        )
        .unwrap();
        for (steps, slots) in [(40, 1), (40, 2), (40, 5)] {
            let schedule = CheckpointSchedule::binomial(steps, slots).unwrap();
            let mut inner = Tape::new(TapeOptions::new());
            let mut adjoints = [0.0; 3];
            pullback_on(
                &mut inner,
                &body,
                steps,
                &schedule,
                &[1.0, 0.0, 0.1],
                &[1.0, 1.0],
                &mut adjoints,
            )
            .unwrap();
            assert_eq!(inner.requirements(), one.requirements());
            assert_eq!(inner.report().words(), one.report().words());
            // d s₀ / d s₀(0) = 1.1^steps.
            assert!(
                (adjoints[0] - 1.1_f64.powi(steps as i32) - adjoints_of_sum(steps)).abs()
                    < 1e-9 * adjoints[0]
            );
        }
    }

    /// The contribution of s₁ to ∂(s₀ + s₁)/∂s₀(0): Σₖ 0.1·1.1ᵏ⁺¹ for k < steps.
    fn adjoints_of_sum(steps: usize) -> f64 {
        (0..steps).map(|k| 0.1 * 1.1_f64.powi(k as i32 + 1)).sum()
    }
}
