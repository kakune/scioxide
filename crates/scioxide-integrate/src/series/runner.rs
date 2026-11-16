//! One call: term steps, batched evaluation under a parallel policy, and output.

use super::double::Wide;
use super::kernel::{levin_u, wynn_epsilon};
use super::state::Candidate;
use super::workspace::SeriesScratch;
use super::{
    EulerMaclaurinProvider, SeriesError, SeriesReport, SeriesState, SeriesSummation,
    SeriesTermination, SeriesTerms, TERM_BATCH, checked_callback_number,
};
use core::sync::atomic::{AtomicU32, Ordering};
use scioxide_core::{
    AlgebraicScalar, BudgetKind, EvaluationRole, NumericScalar, PartialResultTrust, RealScalar,
    SciOxideError, SolveContext,
};

// One bit per prefetched row records its successful sampling.
const _: () = assert!(TERM_BATCH <= 32);

/// Consecutive steps with a new estimate whose error meets the tolerance
/// before a heuristic stop. Each step's error estimate spans two changes
/// between estimates, so a stop needs three consecutive small changes; one
/// agreement can be accidental.
const STABLE_STEPS: usize = 2;

/// Relative size below which a term cannot change its rounded partial sum by
/// more than a few units in the last place; such rows carry no information.
const NEGLIGIBLE_TERM: f64 = 10.0;

/// Relative arithmetic floor of every accelerated or partial-sum estimate.
const ROUNDOFF: f64 = 64.0;

/// Multiple of the floor below which candidate changes are rounding noise.
const NOISE: f64 = 4.0;

/// The candidate transform of a step, chosen once from the problem/method pairing.
#[derive(Clone, Copy)]
pub(super) enum Transform<'a, T: NumericScalar> {
    Wynn,
    Levin,
    Euler(&'a dyn EulerMaclaurinProvider<T>),
}

/// Mutable borrows of one call. `sink` sees the state after every added term.
pub(super) struct Runner<'a, 's, 'c, T: NumericScalar> {
    pub(super) plan: &'a SeriesSummation<T>,
    pub(super) state: &'a mut SeriesState<T>,
    pub(super) scratch: &'a mut SeriesScratch<T>,
    pub(super) output: &'a mut [T],
    pub(super) errors: &'a mut [SeriesError<T::Real>],
    pub(super) context: &'a mut SolveContext<'s, 'c>,
    pub(super) sink: &'a mut dyn FnMut(&SeriesState<T>) -> Result<(), SciOxideError>,
}

impl<T: NumericScalar> Runner<'_, '_, '_, T> {
    /// Sample at most 32 independent terms at once, then replay in index order.
    /// An early stop discards at most 31 already charged callbacks. A failed
    /// batch first accepts the rows before its first unsampled index, exactly
    /// as sequential execution would, and then returns the failure.
    pub(super) fn prefetch(
        &mut self,
        provider: &dyn SeriesTerms<T>,
        transform: Transform<'_, T>,
        batch: &mut [T],
        quota: usize,
    ) -> Result<SeriesReport, SciOxideError> {
        let max_terms = self.plan.options().max_terms().get();
        let width = self.state.width;
        let mut remaining = quota;
        let initial_calls = self
            .context
            .counts()
            .own()
            .provider(EvaluationRole::Function);
        loop {
            let mut count = remaining
                .min(max_terms.saturating_sub(self.state.count))
                .min(TERM_BATCH);
            for kind in [
                BudgetKind::Evaluation,
                BudgetKind::Iteration,
                BudgetKind::Proposal,
            ] {
                if let Some(available) = self.context.effective_budget_report(kind).remaining() {
                    count = count.min(available);
                }
            }
            if count == 0 || width == 0 || self.context.stop_reason()?.is_some() {
                // The sequential admission path owns stop precedence, including
                // quota/term boundaries and stops before the first estimate.
                let mut report = self.run(provider, transform, remaining, false)?;
                report.provider_calls = self
                    .context
                    .counts()
                    .own()
                    .provider(EvaluationRole::Function)
                    - initial_calls;
                return Ok(report);
            }
            let first = self.plan.first + self.state.count;
            let rows = &mut batch[..count * width];
            let sampled = AtomicU32::new(0);
            let outcome = crate::evaluation::sample_rows(
                count,
                width,
                rows,
                self.plan.options().execution_policy(),
                self.context,
                &|index, row| {
                    row.fill(T::from_components(T::Real::NAN, T::Real::NAN));
                    provider.term(first + index, row)?;
                    row.iter()
                        .try_for_each(|&value| checked_callback_number(value).map(drop))?;
                    sampled.fetch_or(1 << index, Ordering::Relaxed);
                    Ok(())
                },
            );
            let usable = match outcome {
                Ok(()) => count,
                Err(_) => (!sampled.load(Ordering::Relaxed)).trailing_zeros() as usize,
            };
            let rows = &*rows;
            let before = self.state.count;
            let mut report = self.run(
                &|index, row: &mut [T]| {
                    let offset = (index - first) * width;
                    row.copy_from_slice(&rows[offset..offset + width]);
                    Ok(())
                },
                transform,
                usable,
                true,
            )?;
            remaining -= self.state.count - before;
            report.provider_calls = self
                .context
                .counts()
                .own()
                .provider(EvaluationRole::Function)
                - initial_calls;
            // A stop inside the usable prefix precedes the failure, as it would sequentially.
            if report.termination != SeriesTermination::CallLimit {
                return Ok(report);
            }
            outcome?;
            if remaining == 0 {
                return Ok(report);
            }
        }
    }

    /// Accept at most `quota` terms in index order. Prefetched replays have
    /// already been charged by their sampler.
    pub(super) fn run(
        &mut self,
        provider: &dyn SeriesTerms<T>,
        transform: Transform<'_, T>,
        quota: usize,
        precounted: bool,
    ) -> Result<SeriesReport, SciOxideError> {
        let width = self.state.width;
        if width == 0 {
            self.context
                .commit_accepted(PartialResultTrust::Admissible)?;
            (self.sink)(self.state)?;
            return Ok(SeriesReport {
                termination: SeriesTermination::ExactEmpty,
                terms: 0,
                provider_calls: 0,
                accepted_state: self.context.accepted_state(),
            });
        }
        let options = self.plan.options();
        let max_terms = options.max_terms().get();
        let tolerance = options.tolerance();
        let floors = Floors::new()?;
        let mut calls = 0;
        let mut termination = SeriesTermination::CallLimit;
        for _ in 0..quota {
            if self.state.count >= max_terms {
                termination = SeriesTermination::TermLimit;
                break;
            }
            // Euler-Maclaurin needs the term, the tail integral, `order` odd derivatives and
            // one even-derivative bound per step.
            let needed = match transform {
                Transform::Euler(_) => self.state.order + 3,
                _ if precounted => 0,
                _ => 1,
            };
            if let Some(stop) = crate::workspace::admit(self.context, needed, false)? {
                termination = match stop {
                    crate::IntegrationTermination::BudgetLimit(kind) => {
                        SeriesTermination::ContextBudgetLimit(kind)
                    }
                    crate::IntegrationTermination::ExecutionStopped(reason) => {
                        SeriesTermination::ExecutionStopped(reason)
                    }
                    _ => {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Breakdown,
                            "invalid series admission status",
                        ));
                    }
                };
                break;
            }
            if !precounted {
                charge(self.context, EvaluationRole::Function)?;
                calls += 1;
            }
            let row = self.state.count;
            let index = self.plan.first + row;
            let offset = self.state.offset(row);
            let terms = &mut self.state.terms[offset..offset + width];
            terms.fill(T::from_components(T::Real::NAN, T::Real::NAN));
            provider.term(index, terms)?;
            for &value in terms.iter() {
                checked_callback_number(value)?;
            }
            // Real and imaginary parts are accumulated separately with exact range-safe sums,
            // in place. Everything else derived from the new term is staged in the scratch
            // buffers and committed only after the whole step succeeded; a failure before
            // then withdraws the term from the exact sums again, so a provider or arithmetic
            // error leaves the state at the previous term and the summation can resume. The
            // new row overwrites only the ring slot that left the window.
            shift_sums(self.state, offset, false)?;
            let (within, fresh) =
                match self.stage(transform, &floors, tolerance, index, offset, &mut calls) {
                    Ok(staged) => staged,
                    Err(error) => {
                        shift_sums(self.state, offset, true)?;
                        return Err(error);
                    }
                };
            self.commit(transform, offset, within, fresh);
            self.context
                .commit_accepted(PartialResultTrust::Admissible)?;
            (self.sink)(self.state)?;
            if let Some(stop) = self.stop(transform, within) {
                termination = stop;
                break;
            }
        }
        if termination == SeriesTermination::CallLimit && self.state.count >= max_terms {
            termination = SeriesTermination::TermLimit;
        }
        Ok(self.publish(termination, calls))
    }

    /// Stage the candidates of the row at ring offset `offset`, whose term is
    /// already in the exact sums, and report whether the step is within
    /// tolerance and whether any component has a new estimate.
    fn stage(
        &mut self,
        transform: Transform<'_, T>,
        floors: &Floors<T::Real>,
        tolerance: &scioxide_core::ErrorTolerance<T::Real>,
        index: usize,
        offset: usize,
        calls: &mut usize,
    ) -> Result<(bool, bool), SciOxideError> {
        let width = self.state.width;
        for c in 0..width {
            let sums = &self.state.accumulators[c];
            let sum = T::from_components(sums[0].finish()?, sums[1].finish()?);
            self.state.partials[offset + c] = checked_callback_number(sum)?;
        }
        let scratch = &mut *self.scratch;
        match transform {
            Transform::Euler(smooth) => {
                *calls +=
                    super::euler::candidate(smooth, index, self.state, scratch, self.context)?;
                for c in 0..width {
                    bounded(self.state, scratch, c)?;
                }
            }
            Transform::Wynn | Transform::Levin => {
                let levin = matches!(transform, Transform::Levin);
                for c in 0..width {
                    accelerate(levin, floors, self.state, scratch, c)?;
                }
            }
        }
        // A step is within tolerance when the latest evidence of every component meets
        // it; components without new information keep their previous candidate.
        let mut available = true;
        let mut fresh = false;
        for c in 0..width {
            let latest = if scratch.fresh[c] {
                &scratch.staged[c]
            } else {
                &self.state.latest[c]
            };
            fresh |= scratch.fresh[c];
            available &= latest.total.is_finite();
            scratch.norm_errors[c] = if latest.total.is_finite() {
                latest.total
            } else {
                T::Real::ZERO
            };
            scratch.norm_scales[c] = latest.value.abs();
        }
        let within =
            available && tolerance.is_satisfied(&scratch.norm_errors, &scratch.norm_scales)?;
        Ok((within, fresh))
    }

    /// Commit the staged step whose row starts at ring offset `offset`: latest,
    /// best and current estimates and the stability counter.
    fn commit(&mut self, transform: Transform<'_, T>, offset: usize, within: bool, fresh: bool) {
        let state = &mut *self.state;
        let scratch = &mut *self.scratch;
        let euler = matches!(transform, Transform::Euler(_));
        for c in 0..state.width {
            // The new partial sum competes for the best only when no fresh candidate is
            // at least as good; ties prefer the newer estimate.
            let best = &mut state.best[c];
            let total = scratch.remainder_totals[c];
            if !euler && total <= best.total {
                *best = Candidate {
                    value: state.partials[offset + c],
                    total,
                    floor: total,
                    errors: scratch.remainders[c],
                    seen: true,
                    ..Candidate::unknown()
                };
            }
            if scratch.fresh[c] {
                let staged = &scratch.staged[c];
                state.latest[c] = *staged;
                if staged.total <= best.total {
                    *best = *staged;
                }
            }
            // The newest candidate is published unless it cannot beat the best estimate.
            let latest = &state.latest[c];
            let published = if latest.seen && latest.floor <= best.total {
                latest
            } else {
                best
            };
            state.estimate[c] = published.value;
            state.errors[c] = published.errors;
        }
        state.count += 1;
        if !euler && fresh {
            state.stable = if within { state.stable + 1 } else { 0 };
        }
    }

    /// Termination after a committed step, if any.
    fn stop(&self, transform: Transform<'_, T>, within: bool) -> Option<SeriesTermination> {
        if let Transform::Euler(_) = transform {
            within.then_some(SeriesTermination::BoundedTailReached)
        } else {
            (self.state.stable >= STABLE_STEPS)
                .then_some(SeriesTermination::HeuristicToleranceReached)
        }
    }

    fn publish(&mut self, termination: SeriesTermination, calls: usize) -> SeriesReport {
        let measured = self.state.count > 0;
        if measured {
            self.output.copy_from_slice(&self.state.estimate);
            self.errors.copy_from_slice(&self.state.errors);
        }
        SeriesReport {
            termination,
            terms: self.state.count,
            provider_calls: calls,
            accepted_state: measured.then(|| self.context.accepted_state()).flatten(),
        }
    }
}

/// Add the term row at ring offset `offset` to the exact running sums, or
/// withdraw it again; both restore the exact sums of the other prefix. An
/// update that fails reverts the parts it already applied. Sums of finite
/// scalars stay within the accumulators' capacity, so neither direction fails
/// in practice.
fn shift_sums<T: NumericScalar>(
    state: &mut SeriesState<T>,
    offset: usize,
    withdraw: bool,
) -> Result<(), SciOxideError> {
    let part = |state: &SeriesState<T>, index: usize, negate: bool| {
        let term = state.terms[offset + index / 2];
        let value = if index.is_multiple_of(2) {
            term.real_component()
        } else {
            term.imaginary_component()
        };
        if negate { -value } else { value }
    };
    for index in 0..2 * state.width {
        let value = part(state, index, withdraw);
        if let Err(error) = state.accumulators[index / 2][index % 2].add(value) {
            for applied in (0..index).rev() {
                let value = part(state, applied, !withdraw);
                state.accumulators[applied / 2][applied % 2].add(value)?;
            }
            return Err(error);
        }
    }
    Ok(())
}

/// Records one provider call that the budget check already allowed.
pub(super) fn charge(
    context: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
) -> Result<(), SciOxideError> {
    if context.record_provider(role)?.is_granted() {
        Ok(())
    } else {
        Err(crate::kernel::invalid(
            scioxide_core::argument!("context"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "admitted series provider was unexpectedly denied",
        ))
    }
}

/// Working-precision thresholds of the accelerated steps, formed once per advance.
struct Floors<R: RealScalar> {
    /// Relative size of a term below which its row carries no information.
    negligible: R,
    /// Relative arithmetic floor of an estimate.
    roundoff: R,
    /// Multiple of the floor below which changes are rounding noise.
    noise: R,
    /// Partial-sum exponents beyond which Wynn's input is normalized.
    band: i32,
}

impl<R: RealScalar> Floors<R> {
    fn new() -> Result<Self, SciOxideError> {
        Ok(Self {
            negligible: R::try_from_f64_lossy(NEGLIGIBLE_TERM)? * R::EPSILON,
            roundoff: R::try_from_f64_lossy(ROUNDOFF)? * R::EPSILON,
            noise: R::try_from_f64_lossy(NOISE)?,
            band: scioxide_core::backend::standard::exponent(R::MAX)? / 4,
        })
    }
}

/// Stage an Euler–Maclaurin candidate: its error is the sum of the provider
/// error, the conditional tail bound and the arithmetic floor.
fn bounded<T: NumericScalar>(
    state: &SeriesState<T>,
    scratch: &mut SeriesScratch<T>,
    c: usize,
) -> Result<(), SciOxideError> {
    let value = scratch.staged[c].value;
    let errors = &mut scratch.errors[c];
    errors.roundoff = errors
        .roundoff
        .max(T::Real::try_from_f64_lossy(ROUNDOFF)? * T::Real::EPSILON * value.abs());
    let total = errors.provider.unwrap_or(T::Real::ZERO)
        + errors.tail_bound.unwrap_or(T::Real::ZERO)
        + errors.roundoff;
    let previous = state.latest[c];
    let total = if total.is_finite() {
        total
    } else {
        T::Real::INFINITY
    };
    scratch.staged[c] = Candidate {
        value,
        change: if previous.seen {
            (value - previous.value).abs()
        } else {
            T::Real::INFINITY
        },
        evidence: T::Real::INFINITY,
        total,
        // A bound is its own floor: a larger bound never replaces a smaller one.
        floor: total,
        errors: *errors,
        seen: true,
    };
    scratch.fresh[c] = true;
    Ok(())
}

/// Stage component `c`'s accelerated candidate from the window ending at the
/// new row. Wynn and Levin read at most the last `order + 1` partial sums and
/// terms, which the state's rings store.
///
/// Rows whose term is zero or cannot change its rounded partial sum carry no
/// information and are omitted from the transform; a new such row keeps the
/// previous candidate, so repeated inputs never fake agreement. A full window
/// of such rows is stationary: the newest partial sum with the largest window
/// term as its change. The error estimate is the largest of the transform's internal
/// change and the last two changes between candidates, discounted above the
/// rounding noise by its geometric tail (unknown without contraction), and never below the
/// arithmetic floor: `64 * epsilon` times the larger of the candidate and the
/// window's partial sums, and for Levin the amplified half-ulp rounding of its
/// terms (see [`levin_u`]). Levin forms its partial-sum differences exactly
/// from the window's terms, receiving each run of omitted terms as one exact
/// double-word sum. Wynn's recurrence commutes with exact power-of-two
/// scaling, so its input is normalized only when the partial sums' binary
/// exponent leaves a quarter of the exponent range, which keeps reciprocal
/// differences in range.
///
/// The new partial sum is also staged with its remainder estimate, the largest
/// `(n + 1) |a_n|` of a full window (Levin's remainder model), floored likewise.
fn accelerate<T: NumericScalar>(
    levin: bool,
    floors: &Floors<T::Real>,
    state: &SeriesState<T>,
    scratch: &mut SeriesScratch<T>,
    c: usize,
) -> Result<(), SciOxideError> {
    let n = state.count + 1;
    let window = state.window();
    let full = n >= window;
    let start = n - n.min(window);
    let mut informative = 0;
    let mut scale = T::Real::ZERO;
    let mut largest = T::Real::ZERO;
    let mut remainder = T::Real::ZERO;
    let mut newest = false;
    let mut sum = T::ZERO;
    let mut gap = Wide::ZERO;
    for (row, offset) in (start..n).zip(state.offsets(start, n - start)) {
        let term = state.terms[offset + c];
        let partial = state.partials[offset + c];
        let point = state.first + row + 1;
        let size = term.abs();
        let magnitude = partial.abs();
        scale = scale.max(magnitude);
        largest = largest.max(size);
        sum = partial;
        // A partial window says nothing about the terms that follow it.
        if full {
            #[allow(clippy::cast_precision_loss, reason = "a magnitude estimate")]
            let model = T::Real::try_from_f64_lossy(point as f64)? * size;
            remainder = remainder.max(model);
        }
        newest = term != T::ZERO && size > floors.negligible * magnitude;
        if newest {
            scratch.history[informative] = partial;
            if levin {
                scratch.term_history[informative] = term;
                scratch.points[informative] = point;
                scratch.gaps[informative] = gap;
                gap = Wide::ZERO;
            }
            informative += 1;
        } else if levin {
            gap = gap.add_value(term);
        }
    }
    let partial_floor = floors.roundoff * scale;
    let remainder = Some(remainder).filter(|r| full && r.is_finite());
    scratch.remainders[c] = SeriesError {
        remainder,
        roundoff: partial_floor,
        ..SeriesError::default()
    };
    scratch.remainder_totals[c] =
        remainder.map_or(T::Real::INFINITY, |remainder| remainder.max(partial_floor));
    let zero = T::Real::ZERO;
    let estimate = if !newest {
        if informative > 0 || !full {
            scratch.fresh[c] = false;
            return Ok(());
        }
        Some((sum, largest, zero))
    } else if levin {
        levin_u(
            &scratch.term_history[..informative],
            &scratch.gaps[..informative],
            &scratch.history[..informative],
            &scratch.points[..informative],
            &mut scratch.magnitudes,
        )?
        .map(|estimate| (estimate.value, zero, estimate.roundoff))
    } else {
        let exponent = scioxide_core::backend::standard::exponent(scale)?;
        let shift = if exponent.abs() > floors.band {
            exponent
        } else {
            0
        };
        if shift != 0 {
            for value in &mut scratch.history[..informative] {
                *value = scioxide_core::backend::standard::scale(*value, -shift);
            }
        }
        let estimate = wynn_epsilon(
            &scratch.history[..informative],
            &mut scratch.tableau_a,
            &mut scratch.tableau_b,
        )?;
        let value = scioxide_core::backend::standard::scale(estimate.value, shift);
        let change = if shift == 0 {
            estimate.change
        } else {
            estimate.change.scalbn(shift)
        };
        (estimate.accelerated && value.has_finite_components() && change.is_finite())
            .then_some((value, change, zero))
    };
    let previous = &state.latest[c];
    let value = estimate.map_or(sum, |(value, ..)| value);
    let change = if previous.seen {
        (value - previous.value).abs()
    } else {
        T::Real::INFINITY
    };
    let roundoff = floors.roundoff * value.abs().max(scale);
    let noise = floors.noise * roundoff;
    let (raw, roundoff) = match estimate {
        Some((_, internal, rounding)) => (
            internal.max(change).max(previous.change),
            roundoff.max(rounding),
        ),
        None => (T::Real::INFINITY, roundoff),
    };
    // Changes above the floor must contract to be credible: a geometric tail with ratio
    // rho adds rho / (1 - rho) times the latest change. Changes within a few working
    // floors (including Wynn's nominal first-column change) or within Levin's amplified
    // term rounding are rounding noise.
    let contraction = raw / previous.evidence;
    let evidence = if raw <= noise.max(roundoff) {
        raw
    } else if contraction < T::Real::ONE {
        raw / (T::Real::ONE - contraction)
    } else {
        T::Real::INFINITY
    };
    let evidence = Some(evidence).filter(|e| e.is_finite());
    // Without a transform the candidate is the partial sum: its remainder evidence
    // bounds what it can achieve, but it never stops a sum.
    let (floor, errors) = if estimate.is_some() {
        (
            roundoff,
            SeriesError {
                acceleration: evidence,
                roundoff,
                ..SeriesError::default()
            },
        )
    } else {
        (scratch.remainder_totals[c], scratch.remainders[c])
    };
    scratch.staged[c] = Candidate {
        value,
        change,
        evidence: raw,
        total: evidence.map_or(T::Real::INFINITY, |evidence| evidence.max(roundoff)),
        floor,
        errors,
        seen: true,
    };
    scratch.fresh[c] = true;
    Ok(())
}
