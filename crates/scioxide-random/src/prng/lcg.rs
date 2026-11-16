//! 64-bit linear congruential generator with Knuth's constants.

use scioxide_core::SciOxideError;

use crate::util::TransitionCounter;

use crate::state::{GeneratorAlgorithm as Algorithm, Snapshot, StateEngine, StreamIdentity};
use crate::traits::{BitGenerator, SeedableRng, SkipAhead};

/// Knuth's multiplier for the 64-bit LCG.
const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// Knuth's increment for the 64-bit LCG.
const INCREMENT: u64 = 1_442_695_040_888_963_407;

/// Linear congruential generator modulo 2⁶⁴ with Knuth's constants and period 2⁶⁴.
///
/// The recurrence is
///
/// ```text
/// xₙ₊₁ = (a xₙ + c) mod 2⁶⁴,   a = 6364136223846793005,   c = 1442695040888963407,
/// ```
///
/// and each output is the new state `xₙ₊₁`. The seed is the initial state `x₀`. Bit `k` of the
/// state has period `2^(k+1)`, so the low bits are strongly patterned.
/// [`next_u32`](crate::BitGenerator::next_u32) returns the high 32 bits.
///
/// Each output costs one multiply-add. [`SkipAhead`](crate::SkipAhead) jumps `n` steps in
/// `O(log n)` by squaring the affine map. Drawing never fails.
///
/// # References
///
/// D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed., Addison-Wesley, 1997,
/// §3.2.1.
///
/// # Examples
///
/// ```
/// use scioxide_random::{BitGenerator, Lcg64, SeedableRng, SerializableState, SkipAhead};
/// let mut rng = Lcg64::from_seed(7);
/// let mut ahead = rng.clone();
/// ahead.skip_to(50)?; // O(log n) fast-forward
/// for _ in 0..50 {
///     rng.next_u64()?;
/// }
/// let checkpoint = rng.export_state()?;
/// let word = ahead.next_u64()?;
/// assert_eq!(rng.next_u64()?, word);
/// assert_eq!(Lcg64::import_state(&checkpoint)?.next_u64()?, word);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Lcg64 {
    state: u64,
    origin: u64,
    counter: TransitionCounter,
}

impl Lcg64 {
    /// Returns the state after `n` steps, by binary exponentiation of the affine map.
    const fn advance(state: u64, n: u64) -> u64 {
        let mut cur_mult: u64 = MULTIPLIER;
        let mut cur_plus: u64 = INCREMENT;
        let mut acc_mult: u64 = 1;
        let mut acc_plus: u64 = 0;
        let mut delta = n;

        // Same affine-map squaring as `Pcg64::advance`, modulo 2^64.
        while delta > 0 {
            if delta & 1 != 0 {
                acc_mult = acc_mult.wrapping_mul(cur_mult);
                acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            }
            cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
            delta >>= 1;
        }

        acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
    }
}

impl BitGenerator for Lcg64 {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    #[inline]
    fn position(&self) -> Option<u128> {
        self.counter.position()
    }
    fn stream_identity(&self) -> Option<StreamIdentity<'_>> {
        Some(self.identity())
    }

    #[inline]
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        self.counter.increment();
        Ok(self.state)
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        // In a power-of-two-modulus LCG, bit k has period only 2^(k+1), so the
        // high half is used; the low half would be badly patterned.
        Ok((self.next_u64()? >> 32) as u32)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
        crate::util::fill_bytes_with(dest, 8, || self.next_u64())
    }
}

impl SeedableRng for Lcg64 {
    fn from_seed(seed: u64) -> Self {
        Self {
            state: seed,
            origin: seed,
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
}

impl SkipAhead for Lcg64 {
    fn skip_ahead(&mut self, n: u128) -> Result<(), SciOxideError> {
        // The period is 2^64, so only `n mod 2^64` changes the state; the
        // counter still records the full u128 advance.
        #[allow(clippy::cast_possible_truncation)]
        let n64 = n as u64;
        self.state = Self::advance(self.state, n64);
        self.counter = self.counter.advanced(n);
        Ok(())
    }
    // Every word or float read is one native transition.
    #[inline]
    fn read_transition_bound(&self, _: crate::traits::private::Token) -> Option<u128> {
        Some(1)
    }
}

impl Lcg64 {
    fn identity(&self) -> StreamIdentity<'static> {
        StreamIdentity::new(Algorithm::Lcg64, [self.origin, 0, 0, 0, 0, 0], None)
    }
}
impl StateEngine for Lcg64 {
    const ALGORITHM: Algorithm = Algorithm::Lcg64;
    const WORDS: usize = 1;
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            identity: self.identity(),
            words: [self.state, 0, 0, 0, 0, 0],
            parameter: 0,
            counter: self.counter.position(),
        }
    }
    fn from_snapshot(snapshot: Snapshot) -> Result<Self, SciOxideError> {
        Ok(Self {
            state: snapshot.words[0],
            origin: snapshot.identity.origin[0],
            counter: TransitionCounter::from_position(snapshot.counter),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Rng, SerializableState};

    #[test]
    fn test_different_seeds() {
        let mut rng1 = Lcg64::from_seed(1);
        let mut rng2 = Lcg64::from_seed(2);
        assert_ne!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_float_range_f64() {
        let mut rng = Lcg64::from_seed(123);
        for _ in 0..1000 {
            let v: f64 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_fill_bytes() {
        let mut rng = Lcg64::from_seed(0);
        let mut buf = [0u8; 20];
        rng.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_skip_ahead() {
        let mut rng1 = Lcg64::from_seed(99);
        for _ in 0..100 {
            rng1.next_u64().unwrap();
        }

        let mut rng2 = Lcg64::from_seed(99);
        rng2.skip_ahead(100).unwrap();

        assert_eq!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_skip_to() {
        let mut rng1 = Lcg64::from_seed(7);
        for _ in 0..50 {
            rng1.next_u64().unwrap();
        }

        let mut rng2 = Lcg64::from_seed(7);
        rng2.skip_to(50).unwrap();

        assert_eq!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn position_overflow_becomes_untracked_without_stopping_the_stream() {
        let mut rng = Lcg64::from_seed(7);
        let initial_state = rng.state;
        rng.skip_ahead(u128::MAX).unwrap();
        rng.skip_ahead(1).unwrap();
        assert_eq!(rng.state, initial_state);
        assert_eq!(rng.position(), None);
        rng.skip_ahead(1).unwrap();
        let _ = rng.next_u64().unwrap();
        assert_eq!(rng.position(), None);
        let error = rng.skip_to(0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        assert_eq!(
            Lcg64::import_state(&rng.export_state().unwrap())
                .unwrap()
                .position(),
            None
        );
    }

    #[test]
    fn test_skip_to_backward_errors() {
        let mut rng = Lcg64::from_seed(7);
        rng.skip_ahead(10).unwrap();
        let error = rng.skip_to(5).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
    }

    #[test]
    fn test_position() {
        let mut rng = Lcg64::from_seed(0);
        assert_eq!(rng.position(), Some(0));
        rng.next_u64().unwrap();
        assert_eq!(rng.position(), Some(1));
    }

    #[test]
    fn test_serializable_roundtrip() {
        let mut rng = Lcg64::from_seed(42);
        rng.next_u64().unwrap();
        let state = rng.export_state().unwrap();
        let mut rng2 = Lcg64::import_state(&state).unwrap();
        assert_eq!(rng.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_import_state_invalid_length() {
        let error = Lcg64::import_state(&[0; 10]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_zero_state_payload_is_valid() {
        let rng = Lcg64::from_seed(0);
        let state = rng.export_state().unwrap();
        let mut restored = Lcg64::import_state(&state).unwrap();
        assert_eq!(
            restored.next_u64().unwrap(),
            Lcg64::from_seed(0).next_u64().unwrap()
        );
    }

    #[test]
    fn test_next_u32_uses_upper_bits() {
        let mut rng = Lcg64::from_seed(42);
        let v64 = rng.next_u64().unwrap();
        let mut rng2 = Lcg64::from_seed(42);
        let v32 = rng2.next_u32().unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let expected = (v64 >> 32) as u32;
        assert_eq!(v32, expected);
    }
}
