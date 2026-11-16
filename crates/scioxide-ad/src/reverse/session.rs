//! Recording sessions for drivers that sweep while they record.
//!
//! A batch records a shared prefix once, then for every group records the item,
//! sweeps down to the mark, reads the input adjoints and rewinds. The session
//! keeps the recorder alive across those steps and tracks the peak column
//! lengths, so an overflow grows the tape and the driver runs again.

use alloc::vec::Vec;

use scioxide_core::{DomainReason, ExecutionContext, InvalidInputReason, SciOxideError};

use super::recorder::{Counts, MIXED, Recorder, WORD_LIMIT};
use super::tape::Tape;
use super::word;
use super::{TapeValue, Var, sweep};
use crate::KinkPolicy;

/// One recording driven step by step.
pub(crate) struct Session<'r, V: TapeValue> {
    recorder: &'r Recorder<'r, V>,
    work: &'r mut [V],
    call_work: &'r mut [V],
    peak: Counts,
    overflowed: bool,
}

impl<'r, V: TapeValue> Session<'r, V> {
    /// Records an independent input.
    #[inline]
    pub(crate) fn input(&self, value: V) -> Var<'r, V> {
        self.recorder.input(value)
    }

    /// The current column lengths.
    pub(crate) fn mark(&self) -> Counts {
        self.recorder.counts()
    }

    /// Resets the columns to `mark`.
    pub(crate) fn rewind(&self, mark: Counts) {
        self.recorder.rewind(mark);
    }

    /// Counts kink hits only in the first `lanes` lanes.
    pub(crate) fn set_valid_lanes(&self, lanes: usize) {
        self.recorder.set_valid_lanes(lanes);
    }

    /// Records the peak lengths and returns whether a statement did not fit; the
    /// driver must stop and return `None`.
    pub(crate) fn overflowed(&mut self) -> bool {
        let counts = self.recorder.counts();
        self.peak = Counts {
            words: self.peak.words.max(counts.words),
            full: self.peak.full.max(counts.full),
            scalars: self.peak.scalars.max(counts.scalars),
            calls: self.peak.calls.max(counts.calls),
            scratch: self.peak.scratch.max(counts.scratch),
        };
        self.overflowed |= self.recorder.overflowed();
        self.overflowed
    }

    /// Adds `seed` to the adjoint of `output`.
    pub(crate) fn seed(&mut self, output: Var<'r, V>, seed: V) {
        match output.recorder() {
            Some(r) if core::ptr::eq(r, self.recorder) => {
                if let Some(target) = self.work.get_mut(output.slot() as usize) {
                    *target += seed;
                }
            }
            Some(r) => {
                r.fault(MIXED);
                self.recorder.fault(MIXED);
            }
            None => {}
        }
    }

    /// Sweeps the statements recorded after `stop` in reverse.
    pub(crate) fn sweep(
        &mut self,
        stop: Counts,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let end = self.recorder.counts();
        sweep::reverse(
            self.recorder,
            end,
            stop.words,
            self.work,
            self.call_work,
            context,
        )
    }

    /// Sweeps the statements recorded between `stop` and `end` in reverse.
    pub(crate) fn sweep_between(
        &mut self,
        end: Counts,
        stop: Counts,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        sweep::reverse(
            self.recorder,
            end,
            stop.words,
            self.work,
            self.call_work,
            context,
        )
    }

    /// Returns and clears the adjoint of an input.
    pub(crate) fn take_adjoint(&mut self, input: Var<'r, V>) -> V {
        match input.recorder() {
            Some(r) if core::ptr::eq(r, self.recorder) => self
                .work
                .get_mut(input.slot() as usize)
                .map_or_else(V::zero, |value| core::mem::replace(value, V::zero())),
            _ => V::zero(),
        }
    }

    /// The kink hits counted so far.
    pub(crate) fn kinks(&self) -> u64 {
        self.recorder.kinks()
    }
}

/// The variable vectors a session lends to its driver.
pub(crate) type Lent<'r, V> = (Vec<Var<'r, V>>, Vec<Var<'r, V>>);

impl<V: TapeValue> Tape<V> {
    /// Runs `body` on a fresh recording; when a statement did not fit, grows the
    /// tape to the peak lengths and runs it once more.
    ///
    /// `body` returns `Ok(None)` after [`Session::overflowed`] returned true. It
    /// receives two empty variable vectors with retained capacity.
    pub(crate) fn session<T, F>(
        &mut self,
        variables: usize,
        mut body: F,
    ) -> Result<T, SciOxideError>
    where
        F: for<'r> FnMut(&mut Session<'r, V>, &mut Lent<'r, V>) -> Result<Option<T>, SciOxideError>,
    {
        self.reserve_variables(variables)?;
        for attempt in 0..2 {
            if !self.clean {
                self.work.fill(V::zero());
            }
            // Dirty while the body seeds and sweeps, also if it panics.
            self.clean = false;
            let Self {
                words,
                full,
                scalars,
                calls,
                scratch,
                work,
                call_work,
                inputs,
                outputs,
                options,
                ..
            } = self;
            let recorder = Recorder::new(words, full, scalars, calls, scratch, options.word_limit);
            recorder.push_word(word::INPUT);
            let mut session = Session {
                recorder: &recorder,
                work,
                call_work,
                peak: Counts::default(),
                overflowed: false,
            };
            let mut lent = (inputs.take(), outputs.take());
            let outcome = body(&mut session, &mut lent);
            let overflowed = session.overflowed();
            let peak = session.peak;
            let kinks = session.kinks();
            let faults = recorder.faults();
            if let Some(constant) = session.work.first_mut() {
                *constant = V::zero();
            }
            let (first, second) = lent;
            inputs.keep(first);
            outputs.keep(second);
            if overflowed {
                // An error of the body is its own, even from an attempt that did not
                // fit: return it without running the body again.
                outcome.as_ref().map_err(Clone::clone)?;
                if attempt == 0 {
                    self.grow_to(peak, variables)?;
                    continue;
                }
                break;
            }
            // A completed attempt leaves every adjoint zero.
            self.clean = outcome.is_ok();
            self.report.kinks = kinks;
            let value = outcome?;
            if faults & WORD_LIMIT != 0 {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("tape_words"),
                    peak.words,
                    self.options.word_limit,
                ));
            }
            if faults & MIXED != 0 {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("function"),
                    InvalidInputReason::Inconsistent,
                    "an operation mixed values of two recordings",
                ));
            }
            if self.options.kink_policy() == KinkPolicy::Error && kinks != 0 {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("point"),
                    DomainReason::Undefined,
                    "an active argument sits on a kink",
                ));
            }
            if let Some(value) = value {
                return Ok(value);
            }
            break;
        }
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("function"),
            InvalidInputReason::Inconsistent,
            "the function recorded more on its second run than on its first",
        ))
    }
}
