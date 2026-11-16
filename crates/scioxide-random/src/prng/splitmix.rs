//! The `SplitMix64` generator.

use scioxide_core::SciOxideError;

use crate::util::TransitionCounter;

use crate::state::{GeneratorAlgorithm as Algorithm, Snapshot, StateEngine, StreamIdentity};
use crate::traits::{BitGenerator, SeedableRng, SkipAhead};

use crate::util::{SPLITMIX64_GAMMA as GOLDEN, splitmix64_mix};

/// `SplitMix64` generator with 64-bit output and period 2⁶⁴.
///
/// The state is a Weyl sequence `sₙ = s₀ + n γ mod 2⁶⁴` with `γ = 0x9e3779b97f4a7c15`, and the
/// output is a fixed bijective mix of `sₙ` (two xorshift-multiply rounds and a final xorshift).
/// The seed is `s₀`. The method is Steele, Lea and Flood's SplittableRandom with the 64-bit mix
/// used by Vigna.
/// [`next_u32`](crate::BitGenerator::next_u32) returns the low 32 bits of a 64-bit output.
///
/// Each output costs O(1), and [`SkipAhead`](crate::SkipAhead) costs O(1) for any distance. The
/// generator is suited to expanding one seed into the state of another generator, which
/// [`seed_sequence`](crate::seed_sequence) and the xoshiro seeders do. Drawing never fails.
///
/// # References
///
/// - G. L. Steele Jr., D. Lea and C. H. Flood, "Fast splittable pseudorandom number
///   generators", *Proc. OOPSLA 2014*, ACM, 2014, 453–472.
/// - S. Vigna, "Further scramblings of Marsaglia's xorshift generators", *J. Comput. Appl.
///   Math.* 315, 2017, 175–181.
///
/// # Examples
///
/// ```
/// use scioxide_random::{BitGenerator, SeedableRng, SerializableState, SkipAhead, SplitMix64};
/// let mut rng = SplitMix64::from_seed(42);
/// let mut ahead = SplitMix64::from_seed(42);
/// ahead.skip_ahead(3)?; // O(1)
/// for _ in 0..3 {
///     rng.next_u64()?;
/// }
/// let mut checkpoint = [0_u8; SplitMix64::STATE_BYTES]; // a fixed-size buffer
/// rng.export_state_into(&mut checkpoint)?;
/// let word = ahead.next_u64()?;
/// assert_eq!(rng.next_u64()?, word);
/// assert_eq!(SplitMix64::import_state(&checkpoint)?.next_u64()?, word);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SplitMix64 {
    state: u64,
    origin: u64,
    counter: TransitionCounter,
}

impl BitGenerator for SplitMix64 {
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
        Ok(self.next_word())
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(self.next_u64()? as u32)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
        crate::util::fill_bytes_with(dest, 8, || self.next_u64())
    }
}

impl SeedableRng for SplitMix64 {
    fn from_seed(seed: u64) -> Self {
        Self {
            state: seed,
            origin: seed,
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
}

impl SkipAhead for SplitMix64 {
    fn skip_ahead(&mut self, n: u128) -> Result<(), SciOxideError> {
        // The state is a Weyl sequence `state_k = state_0 + k * GOLDEN
        // (mod 2^64)` and the output is a stateless mix of it, so skipping is
        // O(1). Truncating `n` to u64 is exact because the period is 2^64 and
        // only `n mod 2^64` affects the state; the counter still records the
        // full u128 advance.
        #[allow(clippy::cast_possible_truncation)]
        let n64 = n as u64;
        self.state = self.state.wrapping_add(n64.wrapping_mul(GOLDEN));
        self.counter = self.counter.advanced(n);
        Ok(())
    }
    // Every word or float read is one native transition.
    #[inline]
    fn read_transition_bound(&self, _: crate::traits::private::Token) -> Option<u128> {
        Some(1)
    }
}

impl SplitMix64 {
    #[inline]
    pub(super) fn next_word(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN);
        self.counter.increment();
        splitmix64_mix(self.state)
    }

    fn identity(&self) -> StreamIdentity<'static> {
        StreamIdentity::new(Algorithm::SplitMix64, [self.origin, 0, 0, 0, 0, 0], None)
    }
}
impl StateEngine for SplitMix64 {
    const ALGORITHM: Algorithm = Algorithm::SplitMix64;
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
    fn test_next_u32() {
        let mut rng = SplitMix64::from_seed(0);
        let v = rng.next_u32().unwrap();
        // Should be the lower 32 bits of the first u64
        let mut rng2 = SplitMix64::from_seed(0);
        let v64 = rng2.next_u64().unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let expected = v64 as u32;
        assert_eq!(v, expected);
    }

    #[test]
    fn test_next_float_range_f64() {
        let mut rng = SplitMix64::from_seed(123);
        for _ in 0..1000 {
            let v: f64 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_next_float_range_f32() {
        let mut rng = SplitMix64::from_seed(456);
        for _ in 0..1000 {
            let v: f32 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_fill_bytes() {
        let mut rng = SplitMix64::from_seed(0);
        let mut buf = [0u8; 20];
        rng.fill_bytes(&mut buf).unwrap();
        // Not all zeros after filling
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_fill_bytes_small() {
        let mut rng = SplitMix64::from_seed(0);
        let mut buf = [0u8; 3];
        rng.fill_bytes(&mut buf).unwrap();
        // Should not panic and should fill partial
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_skip_ahead() {
        let mut rng1 = SplitMix64::from_seed(99);
        for _ in 0..100 {
            rng1.next_u64().unwrap();
        }

        let mut rng2 = SplitMix64::from_seed(99);
        rng2.skip_ahead(100).unwrap();

        assert_eq!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_skip_to() {
        let mut rng1 = SplitMix64::from_seed(7);
        for _ in 0..50 {
            rng1.next_u64().unwrap();
        }

        let mut rng2 = SplitMix64::from_seed(7);
        rng2.skip_to(50).unwrap();

        assert_eq!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn position_overflow_becomes_untracked_without_stopping_the_stream() {
        let mut rng = SplitMix64::from_seed(7);
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
            SplitMix64::import_state(&rng.export_state().unwrap())
                .unwrap()
                .position(),
            None
        );
    }

    #[test]
    fn test_skip_to_backward_errors() {
        let mut rng = SplitMix64::from_seed(7);
        rng.skip_ahead(10).unwrap();
        let error = rng.skip_to(5).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
    }

    #[test]
    fn test_position() {
        let mut rng = SplitMix64::from_seed(0);
        assert_eq!(rng.position(), Some(0));
        rng.next_u64().unwrap();
        assert_eq!(rng.position(), Some(1));
        rng.next_u64().unwrap();
        assert_eq!(rng.position(), Some(2));
    }

    #[test]
    fn test_serializable_state_roundtrip() {
        let mut rng = SplitMix64::from_seed(42);
        rng.next_u64().unwrap();
        rng.next_u64().unwrap();

        let state = rng.export_state().unwrap();
        let mut rng2 = SplitMix64::import_state(&state).unwrap();

        assert_eq!(rng.next_u64().unwrap(), rng2.next_u64().unwrap());
        assert_eq!(rng.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_import_state_invalid_length() {
        let error = SplitMix64::import_state(&[0; 10]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_zero_state_payload_is_valid() {
        let rng = SplitMix64::from_seed(0);
        let state = rng.export_state().unwrap();
        let mut restored = SplitMix64::import_state(&state).unwrap();
        assert_eq!(
            restored.next_u64().unwrap(),
            SplitMix64::from_seed(0).next_u64().unwrap()
        );
    }

    #[test]
    fn test_different_seeds_different_output() {
        let mut rng1 = SplitMix64::from_seed(1);
        let mut rng2 = SplitMix64::from_seed(2);
        assert_ne!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }
}
