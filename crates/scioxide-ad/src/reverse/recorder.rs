//! The recorder a transform builds on its stack for one recording.
//!
//! Its columns are `&[Cell<_>]` views of the tape's vectors, so `Var` operators can
//! append statements through a shared reference. A statement that does not fit is
//! not stored, but the counters keep counting; the transform then grows the
//! vectors to the counted requirement and records again.

use core::cell::Cell;

use super::call::Call;
use super::word::{self, Kind};
use super::{TapeValue, Var};
use crate::rules::Class;

/// Operands from two recorders met in one operation.
pub(crate) const MIXED: u8 = 1;
/// The recording exceeded the word limit.
pub(crate) const WORD_LIMIT: u8 = 2;

/// Lengths of the columns of a recording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    pub(crate) words: usize,
    pub(crate) full: usize,
    pub(crate) scalars: usize,
    pub(crate) calls: usize,
    pub(crate) scratch: usize,
}

/// Read access to the columns of a recording, through plain slices or cells.
pub(crate) trait Columns<V: TapeValue> {
    fn word(&self, index: usize) -> u64;
    fn full(&self, index: usize) -> V;
    fn scalar(&self, index: usize) -> V::Scalar;
    /// Runs `f` with call `index`.
    fn with_call<R>(&self, index: usize, f: impl FnOnce(Option<&Call<V>>) -> R) -> R;
}

/// The columns of a completed recording.
pub(crate) struct Frozen<'a, V: TapeValue> {
    pub(crate) words: &'a [u64],
    pub(crate) full: &'a [V],
    pub(crate) scalars: &'a [V::Scalar],
    pub(crate) calls: &'a [Option<Call<V>>],
}

impl<V: TapeValue> Columns<V> for Frozen<'_, V> {
    #[inline(always)]
    fn word(&self, index: usize) -> u64 {
        self.words.get(index).copied().unwrap_or(word::INPUT)
    }
    #[inline(always)]
    fn full(&self, index: usize) -> V {
        self.full
            .get(index)
            .copied()
            .unwrap_or_else(|| V::constant(<V::Scalar as scioxide_core::RealScalar>::NAN))
    }
    #[inline(always)]
    fn scalar(&self, index: usize) -> V::Scalar {
        self.scalars
            .get(index)
            .copied()
            .unwrap_or(<V::Scalar as scioxide_core::RealScalar>::NAN)
    }
    #[inline]
    fn with_call<R>(&self, index: usize, f: impl FnOnce(Option<&Call<V>>) -> R) -> R {
        f(self.calls.get(index).and_then(Option::as_ref))
    }
}

/// The recorder of one recording.
pub(crate) struct Recorder<'r, V: TapeValue> {
    words: &'r [Cell<u64>],
    full: &'r [Cell<V>],
    scalars: &'r [Cell<V::Scalar>],
    calls: &'r [Cell<Option<Call<V>>>],
    scratch: Cell<Option<&'r mut [V]>>,
    word_count: Cell<usize>,
    full_count: Cell<usize>,
    scalar_count: Cell<usize>,
    call_count: Cell<usize>,
    scratch_need: Cell<usize>,
    kinks: Cell<u64>,
    valid_lanes: Cell<usize>,
    faults: Cell<u8>,
    word_limit: usize,
}

impl<'r, V: TapeValue> Recorder<'r, V> {
    pub(crate) fn new(
        words: &'r mut [u64],
        full: &'r mut [V],
        scalars: &'r mut [V::Scalar],
        calls: &'r mut [Option<Call<V>>],
        scratch: &'r mut [V],
        word_limit: usize,
    ) -> Self {
        Self {
            words: Cell::from_mut(words).as_slice_of_cells(),
            full: Cell::from_mut(full).as_slice_of_cells(),
            scalars: Cell::from_mut(scalars).as_slice_of_cells(),
            calls: Cell::from_mut(calls).as_slice_of_cells(),
            scratch: Cell::new(Some(scratch)),
            word_count: Cell::new(0),
            full_count: Cell::new(0),
            scalar_count: Cell::new(0),
            call_count: Cell::new(0),
            scratch_need: Cell::new(0),
            kinks: Cell::new(0),
            valid_lanes: Cell::new(usize::MAX),
            faults: Cell::new(0),
            word_limit,
        }
    }

    /// Appends a statement word and returns its slot.
    #[inline(always)]
    pub(crate) fn push_word(&self, word: u64) -> u32 {
        let index = self.word_count.get();
        self.word_count.set(index + 1);
        if let Some(cell) = self.words.get(index) {
            cell.set(word);
        }
        if index >= self.word_limit {
            self.fault(WORD_LIMIT);
            return 0;
        }
        index as u32
    }

    #[inline(always)]
    pub(crate) fn push_full(&self, value: V) {
        let index = self.full_count.get();
        self.full_count.set(index + 1);
        if let Some(cell) = self.full.get(index) {
            cell.set(value);
        }
    }

    #[inline(always)]
    pub(crate) fn push_scalar(&self, value: V::Scalar) {
        let index = self.scalar_count.get();
        self.scalar_count.set(index + 1);
        if let Some(cell) = self.scalars.get(index) {
            cell.set(value);
        }
    }

    /// Appends a call table entry and returns its index.
    pub(crate) fn push_call(&self, call: Call<V>) -> u32 {
        // Products stage the values and adjoints of every call in the call scratch.
        let need = call.inputs + call.outputs;
        self.scratch_need.set(self.scratch_need.get().max(need));
        let index = self.call_count.get();
        self.call_count.set(index + 1);
        if let Some(cell) = self.calls.get(index) {
            cell.set(Some(call));
        }
        index.min(u32::MAX as usize) as u32
    }

    /// The full partial column position of the next push.
    #[inline(always)]
    pub(crate) fn full_position(&self) -> usize {
        self.full_count.get()
    }

    /// Runs `f` with evaluation scratch of `len` values, or with `None` when the
    /// tape's scratch is too short (the need is recorded for the next attempt).
    pub(crate) fn with_scratch<R>(&self, len: usize, f: impl FnOnce(Option<&mut [V]>) -> R) -> R {
        self.scratch_need.set(self.scratch_need.get().max(len));
        let taken = self.scratch.take();
        match taken {
            Some(scratch) if scratch.len() >= len => {
                let result = f(Some(&mut scratch[..len]));
                self.scratch.set(Some(scratch));
                result
            }
            other => {
                self.scratch.set(other);
                f(None)
            }
        }
    }

    #[inline(always)]
    pub(crate) fn fault(&self, fault: u8) {
        self.faults.set(self.faults.get() | fault);
    }

    #[inline(always)]
    pub(crate) fn faults(&self) -> u8 {
        self.faults.get()
    }

    #[inline(always)]
    pub(crate) fn add_kinks(&self, hits: u64) {
        self.kinks.set(self.kinks.get() + hits);
    }

    pub(crate) fn kinks(&self) -> u64 {
        self.kinks.get()
    }

    /// The number of leading lanes whose kink hits are counted.
    #[inline(always)]
    pub(crate) fn valid_lanes(&self) -> usize {
        self.valid_lanes.get()
    }

    /// Counts kink hits only in the first `lanes` lanes (padded lanes repeat an
    /// item and are not counted).
    pub(crate) fn set_valid_lanes(&self, lanes: usize) {
        self.valid_lanes.set(lanes);
    }

    /// Resets every column to `mark`; kink counts are kept.
    pub(crate) fn rewind(&self, mark: Counts) {
        self.word_count.set(mark.words);
        self.full_count.set(mark.full);
        self.scalar_count.set(mark.scalars);
        self.call_count.set(mark.calls);
    }

    /// The column lengths counted so far.
    pub(crate) fn counts(&self) -> Counts {
        Counts {
            words: self.word_count.get(),
            full: self.full_count.get(),
            scalars: self.scalar_count.get(),
            calls: self.call_count.get(),
            scratch: self.scratch_need.get(),
        }
    }

    /// Whether some statement did not fit.
    pub(crate) fn overflowed(&self) -> bool {
        let counts = self.counts();
        counts.words > self.words.len()
            || counts.full > self.full.len()
            || counts.scalars > self.scalars.len()
            || counts.calls > self.calls.len()
            || self.scratch_need.get() > self.scratch_len()
    }

    fn scratch_len(&self) -> usize {
        let scratch = self.scratch.take();
        let len = scratch.as_ref().map_or(0, |s| s.len());
        self.scratch.set(scratch);
        len
    }

    /// Records an independent input and returns it.
    #[inline]
    pub(crate) fn input(&'r self, value: V) -> Var<'r, V> {
        let slot = self.push_word(word::INPUT);
        Var::active(value, slot, self)
    }

    /// Records a unary statement.
    #[inline(always)]
    pub(crate) fn unary(&self, kind: Kind, class: Class, slot: u32) -> u32 {
        self.push_word(word::unary(kind, class, slot))
    }

    /// Records a binary statement.
    #[inline(always)]
    pub(crate) fn binary(&self, kinds: [Kind; 2], class: Class, slots: [u32; 2]) -> u32 {
        self.push_word(word::binary(kinds, class, slots))
    }

    /// Records an extended statement with `count` arguments given as (slot, full
    /// partial) pairs.
    pub(crate) fn extended(
        &self,
        call: Option<u32>,
        class: Class,
        count: usize,
        arguments: impl Iterator<Item = (u32, Option<V>)>,
    ) -> u32 {
        let count = count.min(word::MAX_WORDS - 1) as u32;
        let header = word::header(call, class, count);
        self.push_word(header);
        for (slot, partial) in arguments {
            let kind = match partial {
                Some(partial) => {
                    self.push_full(partial);
                    Kind::Full
                }
                None => Kind::One,
            };
            self.push_word(word::argument(kind, slot));
        }
        self.push_word(header)
    }
}

impl<V: TapeValue> Columns<V> for Recorder<'_, V> {
    #[inline(always)]
    fn word(&self, index: usize) -> u64 {
        self.words.get(index).map_or(word::INPUT, Cell::get)
    }
    #[inline(always)]
    fn full(&self, index: usize) -> V {
        self.full.get(index).map_or_else(
            || V::constant(<V::Scalar as scioxide_core::RealScalar>::NAN),
            Cell::get,
        )
    }
    #[inline(always)]
    fn scalar(&self, index: usize) -> V::Scalar {
        self.scalars
            .get(index)
            .map_or(<V::Scalar as scioxide_core::RealScalar>::NAN, Cell::get)
    }
    fn with_call<R>(&self, index: usize, f: impl FnOnce(Option<&Call<V>>) -> R) -> R {
        match self.calls.get(index) {
            Some(cell) => {
                let call = cell.take();
                let result = f(call.as_ref());
                cell.set(call);
                result
            }
            None => f(None),
        }
    }
}
