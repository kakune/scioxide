//! Checked step schedules for reversing a chain of steps with checkpoints.

use alloc::vec::Vec;
use scioxide_core::SciOxideError;

/// Describes when to save, restore and recompute states while reversing a
/// chain of steps.
///
/// A checkpoint stores the forward state needed to restart from a step
/// boundary. One working frame serves the reverse step, and the initial state
/// is stored separately. Two kinds of schedule exist:
///
/// - [`Self::binomial`]: binomial checkpointing for a chain of equal-cost
///   steps with a fixed number of snapshot slots. It minimizes the number of
///   forward advances (Griewank and Walther, 2000).
/// - [`Self::explicit`]: snapshots at step boundaries you choose.
///
/// [`Self::visit`] runs a step schedule without recursion or allocation.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{CheckpointAction, CheckpointSchedule};
/// for schedule in [
///     CheckpointSchedule::binomial(4, 1)?,
///     CheckpointSchedule::explicit(4, &[0, 2])?,
/// ] {
///     let mut control = vec![0; schedule.control_slots()];
///     let mut reversed = Vec::new();
///     schedule.visit(4, &mut control, |action| {
///         if let CheckpointAction::Reverse { step } = action {
///             reversed.push(step);
///         }
///         Ok(())
///     })?;
///     assert_eq!(reversed, [3, 2, 1, 0]);
/// }
/// // Each two-step segment is replayed for both of its steps: 2·(1 + 2) = 6 advances.
/// assert_eq!(CheckpointSchedule::explicit(4, &[0, 2])?.forward_steps(), 6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. Griewank and A. Walther, "Algorithm 799: revolve: an implementation of
/// checkpointing for the reverse or adjoint mode of computational
/// differentiation", *ACM Trans. Math. Softw.* 26(1), 2000.
#[derive(Debug, Clone)]
pub struct CheckpointSchedule {
    steps: usize,
    slots: usize,
    boundaries: Option<Vec<usize>>,
    forward_steps: usize,
}

/// Lists one action of a reverse schedule, as passed to [`CheckpointSchedule::visit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointAction {
    /// Restore a named restart state at the given step boundary.
    Restore {
        /// Checkpoint slot; zero denotes the initial state.
        slot: usize,
        /// Boundary preceding this step.
        step: usize,
    },
    /// Advance the primal through the half-open step range.
    Advance {
        /// First step.
        start: usize,
        /// Exclusive end.
        end: usize,
    },
    /// Save the current restart state after advancing to this boundary.
    Save {
        /// Validated checkpoint slot.
        slot: usize,
        /// Current step boundary.
        step: usize,
    },
    /// Apply the reverse rule of the most recently advanced step.
    Reverse {
        /// Step whose frame is current.
        step: usize,
    },
}

impl CheckpointSchedule {
    /// Builds a binomial schedule that reverses `steps` equal-cost steps with
    /// `slots` snapshots and the fewest forward advances.
    ///
    /// `slots` excludes the initial state and the working frame. Slots beyond
    /// `steps − 1` cannot be used and are ignored.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `slots` exceeds 64.
    /// - `SizeOverflow` if the number of forward advances overflows `usize`.
    pub fn binomial(steps: usize, slots: usize) -> Result<Self, SciOxideError> {
        if slots > 64 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("slots"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "at most 64 checkpoint slots are supported",
            ));
        }
        // More snapshots than steps - 1 can never be used (the root already holds step
        // 0), and clamping keeps `cost` and `visit` minimal. The 64-slot cap bounds the
        // controller's frame stack.
        let slots = slots.min(steps.saturating_sub(1));
        Ok(Self {
            steps,
            slots,
            boundaries: None,
            forward_steps: cost(steps, slots)?,
        })
    }

    /// Builds a schedule that restarts from the states at `boundaries`.
    ///
    /// `boundaries` must start at zero, be strictly increasing and lie below
    /// `steps` (for `steps = 0` it must be `[0]`). The consumer stores the state at
    /// each boundary during its forward pass ([`visit`](Self::visit) emits no
    /// `Save` for an explicit schedule). Each reversed step restores the nearest
    /// boundary at or before it and advances to the step, so a segment of length
    /// `m` costs `m(m + 1)/2` advances; [`forward_steps`](Self::forward_steps)
    /// returns the total, which a caller with a work limit compares with it.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `boundaries` breaks the rules above.
    /// - `SizeOverflow` if the advance count overflows `usize`.
    /// - `AllocationFailure` if the boundary list cannot be stored.
    pub fn explicit(steps: usize, boundaries: &[usize]) -> Result<Self, SciOxideError> {
        if (steps == 0 && boundaries != [0])
            || boundaries.first() != Some(&0)
            || boundaries.windows(2).any(|p| p[0] >= p[1])
            || boundaries.iter().skip(1).any(|&b| b >= steps)
        {
            return Err(inconsistent(scioxide_core::argument!("boundaries")));
        }
        // Each segment is reversed by restoring its start and re-advancing to every
        // step in turn, i.e. len·(len + 1)/2 advances (`cost(len, 0)`).
        let mut work = 0_usize;
        for (&start, end) in boundaries.iter().zip(
            boundaries
                .iter()
                .copied()
                .skip(1)
                .chain(core::iter::once(steps)),
        ) {
            work = work.checked_add(cost(end - start, 0)?).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("checkpoint_replay_work"))
            })?;
        }
        let mut owned = Vec::new();
        owned.try_reserve_exact(boundaries.len()).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("checkpoint_boundaries"),
                boundaries.len(),
            )
        })?;
        owned.extend_from_slice(boundaries);
        Ok(Self {
            steps,
            slots: boundaries.len() - 1,
            boundaries: Some(owned),
            forward_steps: work,
        })
    }

    /// Copies the explicit boundaries fallibly when a tape retains the schedule.
    pub(crate) fn try_clone(&self) -> Result<Self, SciOxideError> {
        let boundaries = self
            .boundaries
            .as_deref()
            .map(|values| {
                scioxide_core::backend::try_copy_slice(
                    values,
                    scioxide_core::quantity!("checkpoint_boundaries"),
                )
            })
            .transpose()?;
        Ok(Self {
            boundaries,
            ..*self
        })
    }

    /// Returns the maximum number of steps.
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Returns the number of snapshot slots, excluding the initial state and
    /// the working frame.
    pub const fn checkpoint_slots(&self) -> usize {
        self.slots
    }
    /// Returns the number of forward step advances for reversing the full
    /// chain.
    ///
    /// The count excludes the forward evaluation that precedes the reversal and,
    /// for an explicit schedule, the advances that store the boundary states.
    pub const fn forward_steps(&self) -> usize {
        self.forward_steps
    }
    /// Returns the number of forward step advances for reversing a chain of
    /// `realized_steps` steps.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `realized_steps` exceeds
    ///   [`Self::steps`].
    /// - `SizeOverflow` if the advance count overflows `usize`.
    pub fn forward_steps_for(&self, realized_steps: usize) -> Result<usize, SciOxideError> {
        self.validate_realized(realized_steps)?;
        let realized = realized_steps;
        if let Some(boundaries) = &self.boundaries {
            boundaries
                .iter()
                .copied()
                .take_while(|&b| b < realized)
                .enumerate()
                .try_fold(0_usize, |work, (i, start)| {
                    let end = boundaries
                        .get(i + 1)
                        .copied()
                        .unwrap_or(realized)
                        .min(realized);
                    work.checked_add(cost(end - start, 0)?).ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!(
                            "explicit_realized_replay_work"
                        ))
                    })
                })
        } else {
            cost(realized, self.slots)
        }
    }

    /// Returns the explicit restart boundaries, or `None` for a binomial
    /// schedule.
    pub fn boundaries(&self) -> Option<&[usize]> {
        self.boundaries.as_deref()
    }
    /// Returns the length of the `control` buffer that [`Self::visit`] needs:
    /// five words per stack frame, `5 · (min(slots, 64) + 1)` in total.
    pub fn control_slots(&self) -> usize {
        (self.slots.min(64) + 1) * 5
    }

    fn validate_realized(&self, realized_steps: usize) -> Result<(), SciOxideError> {
        if realized_steps > self.steps {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("realized_steps"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "realized steps exceed the schedule maximum",
            ));
        }
        Ok(())
    }

    /// Calls `visitor` with each [`CheckpointAction`] needed to reverse
    /// `realized_steps` steps, in order.
    ///
    /// The steps are reversed from last to first. `control` is scratch for the
    /// binomial controller and must hold at least [`Self::control_slots`]
    /// words. The call does not allocate. Arguments are checked before the
    /// first callback.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `realized_steps` exceeds
    ///   [`Self::steps`].
    /// - `DimensionMismatch` if `control` is too short.
    /// - Any error returned by `visitor`, unchanged.
    pub fn visit(
        &self,
        realized_steps: usize,
        control: &mut [u64],
        mut visitor: impl FnMut(CheckpointAction) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        self.validate_realized(realized_steps)?;
        if control.len() < self.control_slots() {
            return Err(SciOxideError::dimension_mismatch(
                self.control_slots(),
                control.len(),
            ));
        }
        // Explicit segments: for each step in reverse, restore the snapshot at the
        // start of its segment and advance up to and including the step so its
        // residuals are live for `Reverse`.
        if let Some(boundaries) = &self.boundaries {
            for step in (0..realized_steps).rev() {
                let slot = boundaries.partition_point(|&b| b <= step) - 1;
                let start = boundaries[slot];
                visitor(CheckpointAction::Restore { slot, step: start })?;
                visitor(CheckpointAction::Advance {
                    start,
                    end: step + 1,
                })?;
                visitor(CheckpointAction::Reverse { step })?;
            }
            return Ok(());
        }
        // Binomial (Revolve-style) reversal without recursion or allocation. The
        // caller-provided `control` buffer is used as an explicit stack of five-word
        // frames. A frame reverses steps [start, start + length) given a snapshot of
        // `start` in `root` and `spare` further snapshot slots. Phase 0 advances to a
        // split point, saves it in root + 1 and pushes the suffix (one fewer spare
        // slot); once the suffix has been reversed the frame is rewritten in place to
        // cover only the prefix. Every push uses one slot, so the depth is at most
        // slots + 1, which is exactly `control_slots`.
        // Frames are [start, length, spare slots, root slot, phase/split].
        let mut depth = 1;
        control[..5].copy_from_slice(&[0, realized_steps as u64, self.slots as u64, 0, 0]);
        while depth != 0 {
            let offset = (depth - 1) * 5;
            let (start, n, slots, root, phase) = (
                control[offset] as usize,
                control[offset + 1] as usize,
                control[offset + 2] as usize,
                control[offset + 3] as usize,
                control[offset + 4] as usize,
            );
            if n == 0 {
                depth -= 1;
                continue;
            }
            // No spare slot: reverse the frame quadratically from its root snapshot.
            if slots == 0 || n == 1 {
                for step in (start..start + n).rev() {
                    visitor(CheckpointAction::Restore {
                        slot: root,
                        step: start,
                    })?;
                    visitor(CheckpointAction::Advance {
                        start,
                        end: step + 1,
                    })?;
                    visitor(CheckpointAction::Reverse { step })?;
                }
                depth -= 1;
            } else if phase == 0 {
                let split = split(n, slots);
                visitor(CheckpointAction::Restore {
                    slot: root,
                    step: start,
                })?;
                visitor(CheckpointAction::Advance {
                    start,
                    end: start + split,
                })?;
                visitor(CheckpointAction::Save {
                    slot: root + 1,
                    step: start + split,
                })?;
                control[offset + 4] = split as u64;
                control[depth * 5..depth * 5 + 5].copy_from_slice(&[
                    (start + split) as u64,
                    (n - split) as u64,
                    (slots - 1) as u64,
                    (root + 1) as u64,
                    0,
                ]);
                depth += 1;
            } else {
                // Tail-replace the prefix frame; depth depends on slots, not trip count.
                control[offset + 1] = phase as u64;
                control[offset + 4] = 0;
            }
        }
        Ok(())
    }
}

fn inconsistent(argument: scioxide_core::Argument) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        "checkpoint schedule arguments are inconsistent",
    )
}

// Exact C(n, k) by the multiplicative formula; every intermediate
// result·(n-k+i)/i is itself a binomial coefficient, so the division is exact.
// Overflow saturates to u128::MAX, which callers only compare against a usize
// bound, so saturation is harmless.
fn binomial(n: u128, k: u128) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1_u128;
    for i in 1..=k {
        let Some(next) = result.checked_mul(n - k + i) else {
            return u128::MAX;
        };
        result = next / i;
    }
    result
}

fn level(n: usize, slots: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    // Repetition number t: the smallest t with C(c + t, c) >= n, where
    // c = slots + 1 snapshots counting the root. By Griewank's binomial bound,
    // c snapshots at repetition number t can reverse up to C(c + t, c) steps.
    // The binomial is monotone in t, so bisection over [0, n - 1] suffices.
    let (mut low, mut high) = (0, n - 1);
    while low < high {
        let mid = low + (high - low) / 2;
        if binomial(slots as u128 + mid as u128 + 1, slots as u128 + 1) >= n as u128 {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

fn cost(n: usize, slots: usize) -> Result<usize, SciOxideError> {
    // Total advances of the binomial schedule: (t + 1)·n - C(c + t, c + 1) with c =
    // slots + 1. This is Griewank-Walther's minimal forward-step count plus one
    // advance per step, because `visit` re-advances each step immediately before
    // reversing it to recreate its residuals.
    let t = level(n, slots);
    let a = (t as u128 + 1) * n as u128;
    let b = binomial(slots as u128 + t as u128 + 1, slots as u128 + 2);
    let count = a.checked_sub(b).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("binomial_checkpoint_work"))
    })?;
    usize::try_from(count).map_err(|_| {
        SciOxideError::size_overflow(scioxide_core::quantity!("binomial_checkpoint_work"))
    })
}

fn split(n: usize, slots: usize) -> usize {
    // Choose the first snapshot position: the smallest prefix length whose
    // repetition level (+1 for the sweep across it) reaches the level of the suffix
    // reversed with one fewer spare slot. This balances the two sub-schedules as in
    // Revolve; the predicate is monotone in `mid`, so bisection finds the boundary.
    let (mut low, mut high) = (1, n - 1);
    while low < high {
        let mid = low + (high - low) / 2;
        if 1 + level(mid + 1, slots) >= level(n - mid, slots - 1) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedule_matches_an_independent_dynamic_program() -> Result<(), SciOxideError> {
        let mut optimum = [[0_usize; 6]; 25];
        for n in 1..25 {
            optimum[n][0] = n * (n + 1) / 2;
            for slots in 1..6 {
                optimum[n][slots] = if n == 1 {
                    1
                } else {
                    (1..n)
                        .map(|k| k + optimum[k][slots] + optimum[n - k][slots - 1])
                        .min()
                        .unwrap()
                };
                let schedule = CheckpointSchedule::binomial(n, slots)?;
                assert_eq!(
                    schedule.forward_steps(),
                    optimum[n][slots],
                    "n={n}, s={slots}"
                );
                let mut controls = [0; 330];
                let mut forward = 0;
                let mut next_reverse = n;
                schedule.visit(n, &mut controls, |action| {
                    match action {
                        CheckpointAction::Advance { start, end } => forward += end - start,
                        CheckpointAction::Reverse { step } => {
                            next_reverse -= 1;
                            assert_eq!(step, next_reverse);
                        }
                        _ => {}
                    }
                    Ok(())
                })?;
                assert_eq!(forward, optimum[n][slots]);
                assert_eq!(next_reverse, 0);
            }
        }
        Ok(())
    }

    #[test]
    fn explicit_schedules_count_segment_advances_and_pass_visitor_errors()
    -> Result<(), SciOxideError> {
        // Each segment of length L is reversed with L(L + 1)/2 advances.
        assert_eq!(
            CheckpointSchedule::explicit(8, &[0, 4])?.forward_steps(),
            20
        );
        assert_eq!(CheckpointSchedule::explicit(4, &[0, 2])?.forward_steps(), 6);
        assert_eq!(CheckpointSchedule::explicit(4, &[0])?.forward_steps(), 10);
        let schedule = CheckpointSchedule::explicit(6, &[0, 2, 5])?;
        let mut controls = alloc::vec![0; schedule.control_slots()];
        let (mut forward, mut reversed, mut saves) = (0, Vec::new(), 0);
        schedule.visit(6, &mut controls, |action| {
            match action {
                CheckpointAction::Advance { start, end } => forward += end - start,
                CheckpointAction::Reverse { step } => reversed.push(step),
                CheckpointAction::Save { .. } => saves += 1,
                _ => {}
            }
            Ok(())
        })?;
        assert_eq!(forward, schedule.forward_steps());
        assert_eq!(reversed, [5, 4, 3, 2, 1, 0]);
        assert_eq!(saves, 0);
        for boundaries in [&[][..], &[1], &[0, 0], &[0, 6]] {
            assert!(CheckpointSchedule::explicit(6, boundaries).is_err());
        }
        // A visitor error comes back unchanged.
        let error = SciOxideError::out_of_bounds(scioxide_core::quantity!("probe"), 7, 3);
        let returned = schedule
            .visit(6, &mut controls, |_| Err(error.clone()))
            .unwrap_err();
        assert_eq!(returned, error);
        Ok(())
    }
}
