//! The statement word of a tape.
//!
//! One `u64` per statement; the slot of a statement is the index of its word.
//!
//! | Bits | Field |
//! |---|---|
//! | 0–1 | arity: 0 input, 1 unary, 2 binary, 3 extended |
//! | 2–3, 4–5 | kind of argument 0 and 1 ([`Kind`]) |
//! | 6–7 | class ([`Class`]) |
//! | 8–35, 36–63 | slot of argument 0 and 1 |
//!
//! An extended statement is a header (arity 3; bit 2 set for a call; bits 8–35:
//! argument count; bits 36–63: call index), its argument words (kind in bits 2–3,
//! slot in bits 8–35) and a copy of the header, whose index is the statement slot.
//! A call statement is followed by its output words: arity 0 with bit 6 set.

use crate::rules::Class;

/// The largest number of words, and one more than the largest slot.
pub(crate) const MAX_WORDS: usize = 1 << 28;
const SLOT_MASK: u64 = (1 << 28) - 1;

/// How an argument's partial is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The next value of the full partial column.
    Full = 0,
    /// The next value of the scalar partial column.
    Scalar = 1,
    /// The partial is 1.
    One = 2,
    /// The partial is −1.
    MinusOne = 3,
}

impl Kind {
    #[inline(always)]
    const fn decode(bits: u64) -> Self {
        match bits & 3 {
            0 => Self::Full,
            1 => Self::Scalar,
            2 => Self::One,
            _ => Self::MinusOne,
        }
    }
}

const fn class_bits(class: Class) -> u64 {
    match class {
        Class::Nonlinear => 0,
        Class::Linear => 1,
        Class::Bilinear => 2,
    }
}

/// The word of an independent input or of the constant slot 0.
pub(crate) const INPUT: u64 = 0;
/// The word of a call output.
pub(crate) const CALL_OUTPUT: u64 = 1 << 6;

#[inline(always)]
pub(crate) const fn unary(kind: Kind, class: Class, slot: u32) -> u64 {
    1 | (kind as u64) << 2 | class_bits(class) << 6 | (slot as u64) << 8
}

#[inline(always)]
pub(crate) const fn binary(kinds: [Kind; 2], class: Class, slots: [u32; 2]) -> u64 {
    2 | (kinds[0] as u64) << 2
        | (kinds[1] as u64) << 4
        | class_bits(class) << 6
        | (slots[0] as u64) << 8
        | (slots[1] as u64) << 36
}

#[inline(always)]
pub(crate) const fn header(call: Option<u32>, class: Class, count: u32) -> u64 {
    let (flag, index) = match call {
        Some(index) => (1, index as u64),
        None => (0, 0),
    };
    3 | flag << 2 | class_bits(class) << 6 | (count as u64) << 8 | index << 36
}

#[inline(always)]
pub(crate) const fn argument(kind: Kind, slot: u32) -> u64 {
    3 | (kind as u64) << 2 | (slot as u64) << 8
}

/// A decoded statement word.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Word(pub(crate) u64);

impl Word {
    #[inline(always)]
    pub(crate) const fn arity(self) -> u64 {
        self.0 & 3
    }
    #[inline(always)]
    pub(crate) const fn kind0(self) -> Kind {
        Kind::decode(self.0 >> 2)
    }
    #[inline(always)]
    pub(crate) const fn kind1(self) -> Kind {
        Kind::decode(self.0 >> 4)
    }
    #[inline(always)]
    pub(crate) const fn class(self) -> Class {
        match (self.0 >> 6) & 3 {
            0 => Class::Nonlinear,
            1 => Class::Linear,
            _ => Class::Bilinear,
        }
    }
    #[inline(always)]
    pub(crate) const fn slot0(self) -> usize {
        ((self.0 >> 8) & SLOT_MASK) as usize
    }
    #[inline(always)]
    pub(crate) const fn slot1(self) -> usize {
        (self.0 >> 36) as usize
    }
    /// Whether an extended header belongs to a call.
    #[inline(always)]
    pub(crate) const fn is_call(self) -> bool {
        (self.0 >> 2) & 1 == 1
    }
    /// Whether an arity-0 word is a call output.
    #[inline(always)]
    pub(crate) const fn is_call_output(self) -> bool {
        self.0 & CALL_OUTPUT != 0
    }
    /// The argument count of an extended header.
    #[inline(always)]
    pub(crate) const fn count(self) -> usize {
        self.slot0()
    }
    /// The call index of a call header.
    #[inline(always)]
    pub(crate) const fn call(self) -> usize {
        self.slot1()
    }
}
