//! Wynn ε-algorithm on the bisection totals, per component, with a bounded
//! table and error estimates for the extrapolated values.
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{RealScalar, SciOxideError};

use crate::series::kernel::WynnEstimate;

// QUADPACK bounds its epsilon sequence to fifty entries. Incremental
// diagonals bound each observation to fifty recurrence steps.
const HISTORY: usize = 50;

#[derive(Clone, Copy)]
pub(crate) struct Candidate<S: RealScalar> {
    pub(crate) value: S,
    pub(crate) error: S,
    pub(crate) larger: S,
    pub(crate) floor: S,
}

impl<S: RealScalar> Candidate<S> {
    pub(crate) fn total(self) -> Result<S, SciOxideError> {
        crate::kernel::checked_intermediate(self.error + self.larger.max(self.floor))
    }
}

#[derive(Clone)]
struct Sequence<S: RealScalar> {
    diagonal: [S; HISTORY],
    valid: usize,
    recent: [S; 3],
    agreements: usize,
    candidate: Option<Candidate<S>>,
}

impl<S: RealScalar> Sequence<S> {
    /// Advances the newest ε diagonal. For order k, the preceding diagonal
    /// supplies ε_(k−2) and ε_(k−1), so no old columns are rebuilt:
    /// ε_k = old_(k−2) + 1/(new_(k−1) − old_(k−1)).
    fn transform(&mut self, value: S) -> Result<WynnEstimate<S>, SciOxideError> {
        let limit = self.valid.min(HISTORY - 1);
        let mut older = S::ZERO;
        let mut old = self.diagonal[0];
        let mut current = value;
        let mut estimate = WynnEstimate {
            value,
            change: S::ZERO,
            accelerated: false,
        };
        self.diagonal[0] = value;
        self.valid = 1;
        let ten = S::try_from_f64_lossy(10.0)?;
        let hundred = S::try_from_f64_lossy(100.0)?;
        for order in 1..=limit {
            let difference = current - old;
            let floor = ten * S::EPSILON * current.abs().max(old.abs());
            if !difference.is_finite() || difference.abs() <= floor || difference == S::ZERO {
                break;
            }
            let next = older + S::ONE / difference;
            if !next.is_finite() {
                break;
            }
            let previous = self.diagonal[order];
            self.diagonal[order] = next;
            self.valid = order + 1;
            older = old;
            old = previous;
            current = next;
            if order % 2 == 0 {
                estimate.change = if estimate.accelerated {
                    (next - estimate.value).abs()
                } else {
                    hundred * S::EPSILON * next.abs()
                };
                estimate.value = next;
                estimate.accelerated = true;
            }
        }
        Ok(estimate)
    }
}

/// Bounded ε table with the error estimate from the last three extrapolated
/// values (Piessens et al., 1983), for bisections and cycles.
pub(crate) struct EpsilonTable<S: RealScalar> {
    components: Vec<Sequence<S>>,
    length: usize,
}

impl<S: RealScalar> EpsilonTable<S> {
    pub(crate) fn new(components: usize) -> Result<Self, SciOxideError> {
        let quantity = scioxide_core::quantity!("epsilon_history");
        Ok(Self {
            components: try_filled(
                components,
                Sequence {
                    diagonal: [S::ZERO; HISTORY],
                    valid: 0,
                    recent: [S::ZERO; 3],
                    agreements: 0,
                    candidate: None,
                },
                quantity,
            )?,
            length: 0,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.length = 0;
        for lane in &mut self.components {
            lane.agreements = 0;
            lane.candidate = None;
            lane.valid = 0;
        }
    }

    pub(crate) fn candidate(&self, c: usize) -> Option<Candidate<S>> {
        self.components[c].candidate
    }

    /// A candidate needs three preceding extrapolated values; a singular table
    /// supplies no candidate. When the table is full, its older half is dropped
    /// and the count of preceding values restarts.
    pub(crate) fn observe(
        &mut self,
        values: &[S],
        floors: impl Fn(usize) -> S,
    ) -> Result<(), SciOxideError> {
        if self.length == HISTORY {
            for lane in &mut self.components {
                lane.agreements = 0;
                lane.candidate = None;
                // Order k depends only on the last k+1 inputs. Keeping
                // the low diagonal orders retains the recent half-window
                // exactly, without rebuilding any old tableau columns.
                lane.valid = lane.valid.min(HISTORY / 2);
            }
            self.length = HISTORY / 2;
        }
        for (c, &value) in values.iter().enumerate() {
            let lane = &mut self.components[c];
            let estimate = lane.transform(value)?;
            if !estimate.accelerated {
                lane.agreements = 0;
                continue;
            }
            let value = estimate.value;
            let floor = floors(c).max(S::try_from_f64_lossy(100.0)? * S::EPSILON * value.abs());
            if lane.agreements >= 3 {
                let change = lane.recent.iter().try_fold(S::ZERO, |sum, &previous| {
                    crate::kernel::checked_intermediate(sum + (value - previous).abs())
                })?;
                let error = change.max(estimate.change).max(floor);
                if lane.candidate.is_none_or(|old| error < old.error) {
                    lane.candidate = Some(Candidate {
                        value,
                        error,
                        larger: S::ZERO,
                        floor,
                    });
                }
            }
            lane.recent.rotate_left(1);
            lane.recent[2] = value;
            lane.agreements += 1;
        }
        self.length += 1;
        Ok(())
    }
}

#[derive(Clone)]
struct Component<S: RealScalar> {
    candidate: Option<Candidate<S>>,
    previous: S,
    delta: S,
    growth: usize,
}

pub(super) struct Extrapolation<S: RealScalar> {
    table: EpsilonTable<S>,
    components: Vec<Component<S>>,
    stalled: usize,
}

impl<S: RealScalar> Extrapolation<S> {
    pub(super) fn new(components: usize) -> Result<Self, SciOxideError> {
        let quantity = scioxide_core::quantity!("epsilon_diagnostics");
        Ok(Self {
            table: EpsilonTable::new(components)?,
            components: try_filled(
                components,
                Component {
                    candidate: None,
                    previous: S::ZERO,
                    delta: S::ZERO,
                    growth: 0,
                },
                quantity,
            )?,
            stalled: 0,
        })
    }

    pub(super) fn reset(&mut self) {
        self.table.reset();
        for lane in &mut self.components {
            lane.candidate = None;
            lane.delta = S::ZERO;
            lane.growth = 0;
        }
        self.stalled = 0;
    }

    pub(super) fn candidate(&self, c: usize) -> Option<Candidate<S>> {
        self.components[c].candidate
    }

    pub(super) fn observe(
        &mut self,
        values: &[S],
        parts: &[[S; 4]],
        larger: &[S],
    ) -> Result<bool, SciOxideError> {
        let first = self.table.length == 0;
        let rollover = self.table.length == HISTORY;
        self.table.observe(values, |c| parts[c][2])?;
        let mut divergent = false;
        let mut improved = false;
        for (c, &value) in values.iter().enumerate() {
            let lane = &mut self.components[c];
            if rollover {
                lane.candidate = None;
            }
            if !first {
                let delta = crate::kernel::checked_intermediate(value - lane.previous)?;
                let floor = S::try_from_f64_lossy(100.0)? * S::EPSILON * value.abs();
                if delta.abs() > floor
                    && lane.delta != S::ZERO
                    && (delta > S::ZERO) == (lane.delta > S::ZERO)
                    && delta.abs() >= S::try_from_f64_lossy(0.99)? * lane.delta.abs()
                {
                    lane.growth += 1;
                } else {
                    lane.growth = 0;
                }
                lane.delta = delta;
                divergent |= lane.growth >= 6
                    && lane.candidate.is_none()
                    && self.table.components[c].agreements == 0;
            }
            lane.previous = value;
            let Some(mut next) = self.table.candidate(c) else {
                continue;
            };
            next.larger = larger[c];
            next.floor = next.floor.max(parts[c][2]);
            next.error = next.error.max(next.floor);
            let mass = parts[c][2] / (S::try_from_f64_lossy(50.0)? * S::EPSILON);
            let cancellation =
                next.value.abs().max(value.abs()) <= S::try_from_f64_lossy(0.01)? * mass;
            let inconsistent = !cancellation
                && value != S::ZERO
                && (next.value / value < S::try_from_f64_lossy(0.01)?
                    || next.value / value > S::try_from_f64_lossy(100.0)?);
            let total = next.total()?;
            let better = match lane.candidate {
                Some(old) => total < old.total()?,
                None => true,
            };
            let direct =
                crate::kernel::checked_intermediate(parts[c][1].max(parts[c][2]) + parts[c][3])?;
            divergent |= inconsistent && better && total < direct;
            if !inconsistent && better {
                lane.candidate = Some(next);
                improved = true;
            }
        }
        self.stalled = if improved { 0 } else { self.stalled + 1 };
        Ok(divergent)
    }

    pub(super) fn stagnated(&self, direct: &[S]) -> Result<bool, SciOxideError> {
        let fraction = S::try_from_f64_lossy(0.001)?;
        Ok(self.stalled > 5
            && self.components.iter().zip(direct).any(|(lane, &raw)| {
                lane.candidate
                    .is_some_and(|best| best.error < fraction * raw)
            }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn incremental_diagonals_match_the_batch_recurrence() {
        fn check<S: RealScalar>() {
            let mut table = EpsilonTable::<S>::new(1).unwrap();
            let mut current = [S::ZERO; HISTORY];
            let mut previous = [S::ZERO; HISTORY];
            let values = [0.0, 0.5, 0.75, 0.85, 0.91, 0.95]
                .map(|value| S::try_from_f64_lossy(value).unwrap());
            for (index, &value) in values.iter().enumerate() {
                let streamed = table.components[0].transform(value).unwrap();
                let batch = crate::series::kernel::wynn_epsilon(
                    &values[..index + 1],
                    &mut current,
                    &mut previous,
                )
                .unwrap();
                assert_eq!(streamed.accelerated, batch.accelerated);
                let allowance = S::try_from_f64_lossy(32.0).unwrap()
                    * S::EPSILON
                    * batch.value.abs().max(S::ONE);
                assert!((streamed.value - batch.value).abs() <= allowance);
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn first_even_column_and_singular_tables_do_not_prove_convergence() {
        let mut state = Extrapolation::<f64>::new(1).unwrap();
        for value in [0.0, 0.5, 0.75, 0.875, 0.9375] {
            state
                .observe(&[value], &[[value, 1.0, 1e-14, 0.0]], &[0.0])
                .unwrap();
            assert!(state.candidate(0).is_none());
        }
        state
            .observe(&[0.96875], &[[0.96875, 1.0, 1e-14, 0.0]], &[0.0])
            .unwrap();
        assert!((state.candidate(0).unwrap().value - 1.0).abs() < 1e-14);
        assert!(state.candidate(0).unwrap().error >= 1e-14);
        state.reset();
        for _ in 0..8 {
            state
                .observe(&[1.0], &[[1.0, 1.0, 1e-14, 0.0]], &[0.0])
                .unwrap();
            assert!(state.candidate(0).is_none());
        }
    }

    #[test]
    fn bounded_history_keeps_a_recent_window_and_restarts_agreement() {
        let mut state = Extrapolation::<f64>::new(1).unwrap();
        for index in 0..HISTORY {
            let value = 1.0 / (index as f64 + 1.0);
            state
                .observe(&[value], &[[value, 1.0, 1e-14, 0.0]], &[0.0])
                .unwrap();
        }
        let value = 1.0 / 51.0;
        state
            .observe(&[value], &[[value, 1.0, 1e-14, 0.0]], &[0.0])
            .unwrap();
        assert_eq!(state.table.length, HISTORY / 2 + 1);
        assert!(state.candidate(0).is_none());
        assert!((state.table.components[0].diagonal[0] - 1.0 / 51.0).abs() < 1e-15);
        assert!(state.table.components[0].valid <= HISTORY / 2 + 1);
    }
}
