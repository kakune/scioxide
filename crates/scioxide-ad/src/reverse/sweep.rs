//! Reverse and tangent sweeps over a recording.
//!
//! A sweep is one loop over statement words. Every update is an absolute-zero
//! product ([`Real::azmul`](crate::Real::azmul)), so an infinite partial times a zero
//! adjoint or tangent contributes zero. Both sweeps call `context.check()` every
//! 1,024 words.

use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

use super::TapeValue;
use super::recorder::{Columns, Counts};
use super::word::{Kind, Word};
use crate::Mask as _;

/// Words between cancellation checks.
const CHECK_INTERVAL: usize = 1024;

/// Polls inside extended statements, which can contain more than 1,024 words.
#[inline]
fn check_arguments(index: usize, context: ExecutionContext<'_>) -> Result<(), SciOxideError> {
    if index.is_multiple_of(CHECK_INTERVAL) {
        context.check()?;
    }
    Ok(())
}

#[inline(always)]
fn take<V: TapeValue>(work: &mut [V], slot: usize) -> V {
    work.get_mut(slot)
        .map_or_else(V::zero, |value| core::mem::replace(value, V::zero()))
}

/// A partial popped or read from the columns.
#[derive(Clone, Copy)]
enum Partial<V: TapeValue> {
    Full(V),
    Scalar(V::Scalar),
    One,
    MinusOne,
}

impl<V: TapeValue> Partial<V> {
    /// Returns `partial · value` with absolute-zero products.
    #[inline(always)]
    fn apply(self, value: V) -> V {
        match self {
            Self::Full(partial) => partial.azmul(value),
            Self::Scalar(partial) => V::constant(partial).azmul(value),
            Self::One => value,
            Self::MinusOne => -value,
        }
    }
}

/// Cursors into the partial columns.
struct Cursor {
    full: usize,
    scalars: usize,
}

impl Cursor {
    /// Pops the partial of `kind` and returns `partial · adjoint`.
    #[inline(always)]
    fn pull<V: TapeValue, C: Columns<V>>(&mut self, columns: &C, kind: Kind, adjoint: V) -> V {
        match kind {
            Kind::Full => {
                self.full = self.full.wrapping_sub(1);
                columns.full(self.full).azmul(adjoint)
            }
            Kind::Scalar => {
                self.scalars = self.scalars.wrapping_sub(1);
                V::constant(columns.scalar(self.scalars)).azmul(adjoint)
            }
            Kind::One => adjoint,
            Kind::MinusOne => -adjoint,
        }
    }

    #[inline(always)]
    fn pop<V: TapeValue, C: Columns<V>>(&mut self, columns: &C, kind: Kind) -> Partial<V> {
        match kind {
            Kind::Full => {
                self.full = self.full.wrapping_sub(1);
                Partial::Full(columns.full(self.full))
            }
            Kind::Scalar => {
                self.scalars = self.scalars.wrapping_sub(1);
                Partial::Scalar(columns.scalar(self.scalars))
            }
            Kind::One => Partial::One,
            Kind::MinusOne => Partial::MinusOne,
        }
    }

    #[inline(always)]
    fn next<V: TapeValue, C: Columns<V>>(&mut self, columns: &C, kind: Kind) -> Partial<V> {
        match kind {
            Kind::Full => {
                let partial = columns.full(self.full);
                self.full += 1;
                Partial::Full(partial)
            }
            Kind::Scalar => {
                let partial = columns.scalar(self.scalars);
                self.scalars += 1;
                Partial::Scalar(partial)
            }
            Kind::One => Partial::One,
            Kind::MinusOne => Partial::MinusOne,
        }
    }
}

#[inline(always)]
fn accumulate<V: TapeValue>(work: &mut [V], slot: usize, partial: Partial<V>, adjoint: V) {
    add(work, slot, partial.apply(adjoint));
}

#[inline(always)]
fn add<V: TapeValue>(work: &mut [V], slot: usize, update: V) {
    if let Some(target) = work.get_mut(slot) {
        *target += update;
    }
}

#[inline(always)]
fn read<V: TapeValue>(work: &[V], slot: usize) -> V {
    work.get(slot).copied().unwrap_or_else(V::zero)
}

fn short_scratch() -> SciOxideError {
    SciOxideError::accounting_invariant("call scratch shorter than the recording needs", None)
}

/// Sweeps the statements of words `stop..end.words` in reverse, adding the
/// adjoint of each statement to its arguments and clearing it.
///
/// Input words are skipped and keep their adjoints, so the caller reads and
/// clears them afterwards.
pub(crate) fn reverse<V: TapeValue, C: Columns<V>>(
    columns: &C,
    end: Counts,
    stop: usize,
    work: &mut [V],
    call_work: &mut [V],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    let mut cursor = Cursor {
        full: end.full,
        scalars: end.scalars,
    };
    let mut index = end.words;
    let mut since_check = 0;
    while index > stop {
        index -= 1;
        since_check += 1;
        if since_check >= CHECK_INTERVAL {
            since_check = 0;
            context.check()?;
        }
        let word = Word(columns.word(index));
        match word.arity() {
            0 => {}
            1 => {
                let adjoint = take(work, index);
                let update = cursor.pull(columns, word.kind0(), adjoint);
                add(work, word.slot0(), update);
            }
            2 => {
                let adjoint = take(work, index);
                let right = cursor.pull(columns, word.kind1(), adjoint);
                let left = cursor.pull(columns, word.kind0(), adjoint);
                add(work, word.slot1(), right);
                add(work, word.slot0(), left);
            }
            _ => {
                let count = word.count();
                // The argument words count toward the cancellation interval.
                since_check += count;
                let first = index.saturating_sub(count);
                if word.is_call() {
                    let saved = columns.with_call(word.call(), |call| {
                        let Some(call) = call else {
                            return Err(SciOxideError::accounting_invariant(
                                "missing call entry",
                                None,
                            ));
                        };
                        let (inputs, outputs) = (call.inputs, call.outputs);
                        let needed = 2 * (inputs + outputs);
                        if call_work.len() < needed {
                            return Err(short_scratch());
                        }
                        let (values, rest) = call_work.split_at_mut(inputs + outputs);
                        let (output_adjoints, rest) = rest.split_at_mut(outputs);
                        let input_adjoints = &mut rest[..inputs];
                        for (k, value) in values.iter_mut().enumerate() {
                            check_arguments(k, context)?;
                            *value = columns.full(call.saved + k);
                        }
                        for (k, adjoint) in output_adjoints.iter_mut().enumerate() {
                            check_arguments(k, context)?;
                            *adjoint = take(work, index + 1 + k);
                        }
                        // A pullback is linear in the output adjoints: with all of them
                        // exactly zero it contributes nothing, so it is not run.
                        if output_adjoints
                            .iter()
                            .all(|&a| crate::support::is_zero(a).all())
                        {
                            return Ok(call.saved_len());
                        }
                        input_adjoints.fill(V::zero());
                        let (input_values, output_values) = values.split_at(inputs);
                        call.pullback(
                            input_values,
                            output_values,
                            output_adjoints,
                            input_adjoints,
                        )?;
                        for (k, &adjoint) in input_adjoints.iter().enumerate() {
                            check_arguments(k, context)?;
                            let argument = Word(columns.word(first + k));
                            if let Some(target) = work.get_mut(argument.slot0()) {
                                *target += adjoint;
                            }
                        }
                        Ok(call.saved_len())
                    })?;
                    cursor.full = cursor.full.wrapping_sub(saved);
                } else {
                    let adjoint = take(work, index);
                    for k in (0..count).rev() {
                        check_arguments(k, context)?;
                        let argument = Word(columns.word(first + k));
                        let partial = cursor.pop(columns, argument.kind0());
                        accumulate(work, argument.slot0(), partial, adjoint);
                    }
                }
                index = first.saturating_sub(1);
            }
        }
    }
    context.check()
}

/// Sweeps the statements of words `start.words..end` forward, writing the tangent
/// of each statement. The caller seeds the inputs and the constant slot.
pub(crate) fn tangent<V: TapeValue, C: Columns<V>>(
    columns: &C,
    start: Counts,
    end: usize,
    work: &mut [V],
    call_work: &mut [V],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    let mut cursor = Cursor {
        full: start.full,
        scalars: start.scalars,
    };
    let mut index = start.words;
    let mut since_check = 0;
    while index < end {
        since_check += 1;
        if since_check >= CHECK_INTERVAL {
            since_check = 0;
            context.check()?;
        }
        let word = Word(columns.word(index));
        match word.arity() {
            0 => {}
            1 => {
                let partial = cursor.next(columns, word.kind0());
                let value = partial.apply(read(work, word.slot0()));
                if let Some(target) = work.get_mut(index) {
                    *target = value;
                }
            }
            2 => {
                let left = cursor.next(columns, word.kind0());
                let right = cursor.next(columns, word.kind1());
                let value =
                    left.apply(read(work, word.slot0())) + right.apply(read(work, word.slot1()));
                if let Some(target) = work.get_mut(index) {
                    *target = value;
                }
            }
            _ => {
                let count = word.count();
                since_check += count;
                let copy = index + count + 1;
                if word.is_call() {
                    let saved = columns.with_call(word.call(), |call| {
                        let Some(call) = call else {
                            return Err(SciOxideError::accounting_invariant(
                                "missing call entry",
                                None,
                            ));
                        };
                        let (inputs, outputs) = (call.inputs, call.outputs);
                        if call_work.len() < 2 * (inputs + outputs) {
                            return Err(short_scratch());
                        }
                        let (values, rest) = call_work.split_at_mut(inputs + outputs);
                        let (input_tangents, rest) = rest.split_at_mut(inputs);
                        let output_tangents = &mut rest[..outputs];
                        for (k, value) in values.iter_mut().enumerate() {
                            check_arguments(k, context)?;
                            *value = columns.full(call.saved + k);
                        }
                        for (k, tangent) in input_tangents.iter_mut().enumerate() {
                            check_arguments(k, context)?;
                            *tangent = read(work, Word(columns.word(index + 1 + k)).slot0());
                        }
                        let (input_values, output_values) = values.split_at(inputs);
                        call.pushforward(
                            input_values,
                            output_values,
                            input_tangents,
                            output_tangents,
                        )?;
                        for (k, &tangent) in output_tangents.iter().enumerate() {
                            check_arguments(k, context)?;
                            if let Some(target) = work.get_mut(copy + 1 + k) {
                                *target = tangent;
                            }
                        }
                        Ok(call.saved_len())
                    })?;
                    cursor.full += saved;
                } else {
                    let mut sum = V::constant(<V::Scalar as RealScalar>::NEG_ZERO);
                    for k in 0..count {
                        check_arguments(k, context)?;
                        let argument = Word(columns.word(index + 1 + k));
                        let partial = cursor.next(columns, argument.kind0());
                        sum += partial.apply(read(work, argument.slot0()));
                    }
                    if let Some(target) = work.get_mut(copy) {
                        *target = sum;
                    }
                }
                index = copy;
            }
        }
        index += 1;
    }
    context.check()
}
