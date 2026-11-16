//! Bounded signed multiword integers for exact angular-momentum sums.
//! The caller reserves the storage; no general-purpose public integer API is exposed.
use crate::support::Work;
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::{Scaled, SciOxideError, backend::try_filled};
/// Reserved-capacity error reported for the `scratch` workspace.
fn capacity() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("scratch"),
        scioxide_core::InvalidInputReason::OutOfRange,
        "reserved angular integer capacity exhausted",
    )
}
#[derive(Debug)]
pub(super) struct Integer {
    words: Vec<u32>,
    used: usize,
    pub negative: bool,
}
impl Integer {
    /// Reserve `limbs` words.
    pub(super) fn new(limbs: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            words: try_filled(limbs, 0, scioxide_core::quantity!("integer_limbs"))?,
            used: 0,
            negative: false,
        })
    }
    pub(super) fn clear(&mut self) {
        self.used = 0;
        self.negative = false;
    }
    pub(super) fn is_zero(&self) -> bool {
        self.used == 0
    }
    pub(super) fn one(&mut self) -> Result<(), SciOxideError> {
        if self.words.is_empty() {
            return Err(capacity());
        }
        self.words[0] = 1;
        self.used = 1;
        self.negative = false;
        Ok(())
    }
    pub(super) fn copy_from(&mut self, other: &Self) -> Result<(), SciOxideError> {
        if other.used > self.words.len() {
            return Err(capacity());
        }
        self.words[..other.used].copy_from_slice(&other.words[..other.used]);
        self.used = other.used;
        self.negative = other.negative;
        Ok(())
    }
    fn trim(&mut self) {
        while self.used > 0 && self.words[self.used - 1] == 0 {
            self.used -= 1;
        }
        if self.used == 0 {
            self.negative = false;
        }
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.used.cmp(&other.used).then_with(|| {
            self.words[..self.used]
                .iter()
                .rev()
                .cmp(other.words[..other.used].iter().rev())
        })
    }
    pub(super) fn mul_small(
        &mut self,
        n: u32,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        if n == 0 {
            self.clear();
            return Ok(());
        }
        let mut carry = 0_u64;
        for word in &mut self.words[..self.used] {
            work.try_step()?;
            let product = u64::from(*word) * u64::from(n) + carry;
            *word = product as u32;
            carry = product >> 32;
        }
        if carry != 0 {
            if self.used == self.words.len() {
                return Err(capacity());
            }
            self.words[self.used] = carry as u32;
            self.used += 1;
        }
        Ok(())
    }
    pub(super) fn product(
        a: &Self,
        b: &Self,
        out: &mut Self,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        out.clear();
        if a.is_zero() || b.is_zero() {
            return Ok(());
        }
        let length = a.used.checked_add(b.used).ok_or_else(capacity)?;
        if length > out.words.len() {
            return Err(capacity());
        }
        out.words[..length].fill(0);
        for (i, &x) in a.words[..a.used].iter().enumerate() {
            let mut carry = 0_u64;
            for (j, &y) in b.words[..b.used].iter().enumerate() {
                work.try_step()?;
                let product = u64::from(x) * u64::from(y) + u64::from(out.words[i + j]) + carry;
                out.words[i + j] = product as u32;
                carry = product >> 32;
            }
            out.words[i + b.used] = carry as u32;
        }
        out.used = length;
        out.negative = a.negative ^ b.negative;
        out.trim();
        Ok(())
    }
    pub(super) fn add(
        &mut self,
        other: &Self,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        if other.is_zero() {
            return Ok(());
        }
        if self.is_zero() {
            return self.copy_from(other);
        }
        let old = self.used;
        let count = old.max(other.used);
        if count > self.words.len() {
            return Err(capacity());
        }
        if self.negative == other.negative {
            let mut carry = 0_u64;
            for i in 0..count {
                work.try_step()?;
                let a = if i < old { self.words[i] } else { 0 };
                let b = if i < other.used { other.words[i] } else { 0 };
                let total = u64::from(a) + u64::from(b) + carry;
                self.words[i] = total as u32;
                carry = total >> 32;
            }
            self.used = count;
            if carry != 0 {
                if count == self.words.len() {
                    return Err(capacity());
                }
                self.words[count] = carry as u32;
                self.used += 1;
            }
        } else {
            let reverse = self.compare(other) == Ordering::Less;
            let mut borrow = 0_i64;
            for i in 0..count {
                work.try_step()?;
                let a = i64::from(if i < old { self.words[i] } else { 0 });
                let b = i64::from(if i < other.used { other.words[i] } else { 0 });
                let difference = if reverse {
                    b - a - borrow
                } else {
                    a - b - borrow
                };
                self.words[i] = difference as u32;
                borrow = i64::from(difference < 0);
            }
            if reverse {
                self.negative = other.negative;
            }
            self.used = count;
            self.trim();
        }
        Ok(())
    }
    pub(super) fn scaled(&self) -> Result<Scaled<f64>, SciOxideError> {
        if self.is_zero() {
            return Ok(Scaled::ZERO);
        }
        let start = self.used.saturating_sub(3);
        let top = self.words[start..self.used]
            .iter()
            .rev()
            .fold(0.0, |a, &b| a * 4_294_967_296.0 + f64::from(b));
        let exponent = i64::try_from(start)
            .ok()
            .and_then(|n| n.checked_mul(32))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("binary_exponent"))
            })?;
        Scaled::from_value(if self.negative { -top } else { top })?.shifted(exponent)
    }
}
