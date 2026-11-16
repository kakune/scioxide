//! Loops reversed with checkpoints.

use alloc::sync::Arc;

use scioxide_core::{RealScalar, SciOxideError};

use crate::real::sealed::{self, Token};
use crate::{CheckpointSchedule, Dual, Lanes, Real};

/// One step `state ← step(index, state, parameters)` of a loop, written once for
/// every number type.
pub trait LoopBody<S: RealScalar = f64> {
    /// Advances `state` by step `index`.
    ///
    /// # Errors
    ///
    /// Any error of the step, returned unchanged.
    fn step<R: Real<S>>(
        &self,
        index: usize,
        state: &mut [R],
        parameters: &[R],
    ) -> Result<(), SciOxideError>;
}

/// Runs `steps` steps of `body` on `state` with checkpointed reverse mode.
///
/// Every number type but [`Var`](crate::Var) runs the plain loop in a copy of the
/// state, using `O(state.len())` scratch. The state is published only on success,
/// so a callback error leaves every component unchanged. On a tape the loop
/// runs on the values and records one call statement from the initial state and
/// the parameters to the final state; the tape keeps the `Arc`. Its pullback
/// follows `schedule` ([`CheckpointSchedule::visit`]): restores, advances and saves
/// replay steps on values between snapshots, and each reversed step is recorded on
/// an inner tape and swept, so memory is the snapshots plus one step of tape. For
/// an explicit schedule the pullback first advances from the initial state to the
/// last boundary and stores the state at each boundary, `boundaries.last()`
/// advances beyond [`CheckpointSchedule::forward_steps`]. Its pushforward runs the
/// loop on duals. A plain Rust loop is the store-all form.
///
/// The body runs on values when the loop is recorded, so a tape counts no kinks
/// inside it and [`KinkPolicy::Error`](crate::KinkPolicy::Error) does not see them.
///
/// # Errors
///
/// - `DimensionMismatch` if `schedule.steps() != steps`.
/// - `SizeOverflow` if the snapshot storage size overflows `usize`.
/// - `AllocationFailure` if the loop's scratch or retained schedule cannot be allocated.
/// - The errors of `body`, unchanged.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use scioxide_ad::{CheckpointSchedule, LoopBody, Real, checkpointed_loop, grad};
/// use scioxide_core::SciOxideError;
///
/// /// s ← s·(1 + p).
/// struct Growth;
/// impl LoopBody for Growth {
///     fn step<R: Real>(&self, _: usize, s: &mut [R], p: &[R]) -> Result<(), SciOxideError> {
///         s[0] *= p[0] + 1.0;
///         Ok(())
///     }
/// }
/// let (body, schedule) = (Arc::new(Growth), CheckpointSchedule::binomial(10, 2)?);
/// let g = grad(|x| { let mut s = [x[0]]; checkpointed_loop(&body, 10, &mut s, &x[1..], &schedule)?; Ok(s[0]) }, &[1.0, 0.1])?;
/// assert!((g.as_slice()[0] - 1.1_f64.powi(10)).abs() < 1e-13);
/// assert!((g.as_slice()[1] - 10.0 * 1.1_f64.powi(9)).abs() < 1e-12);
/// # Ok::<(), SciOxideError>(())
/// ```
pub fn checkpointed_loop<S, R, B>(
    body: &Arc<B>,
    steps: usize,
    state: &mut [R],
    parameters: &[R],
    schedule: &CheckpointSchedule,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    R: Real<S>,
    B: LoopBody<S> + Send + Sync + 'static,
{
    crate::support::check_len(schedule.steps(), steps)?;
    R::run_loop(Token(()), body, steps, state, parameters, schedule)
}

impl<S: RealScalar> sealed::Loop<S> for S {}
impl<S: RealScalar, const W: usize> sealed::Loop<S> for Lanes<S, W> {}
impl<S: RealScalar, V: Real<S>> sealed::Loop<S> for Dual<V> {}
impl<S: RealScalar, V: Real<S>, const N: usize> sealed::Loop<S> for crate::MultiDual<V, N> {}
impl<S: RealScalar, V: Real<S>, const N: usize> sealed::Loop<S> for crate::Jet<V, N> {}
