//! PCG64 (PCG-XSL-RR): a 128-bit linear congruential generator with a permuted
//! 64-bit output.

use scioxide_core::SciOxideError;

use crate::util::TransitionCounter;

use crate::state::{self, GeneratorAlgorithm as Algorithm, Snapshot, StateEngine, StreamIdentity};
use crate::traits::{BitGenerator, SeedableRng, SkipAhead, SplittableRng};

/// Default multiplier for the 128-bit LCG.
const MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// Default increment (must be odd).
const DEFAULT_INCREMENT: u128 = 0x5851_F42D_4C95_7F2D_1405_7B7E_F767_814F;

/// PCG64 generator (PCG-XSL-RR 128/64) with 64-bit output and period 2¹²⁸ per stream.
///
/// The state follows the linear congruential recurrence
///
/// ```text
/// sₙ₊₁ = (a sₙ + c) mod 2¹²⁸,
/// ```
///
/// with O'Neill's 128-bit multiplier `a` and an odd increment `c` that selects the stream. Each
/// output is the XOR of the high and low 64-bit halves of `sₙ₊₁`, rotated right by the top six
/// bits of `sₙ₊₁` (O'Neill 2014). [`next_u32`](crate::BitGenerator::next_u32) returns the low 32
/// bits of a 64-bit output.
///
/// Each output costs one 128-bit multiply-add. [`SkipAhead`](crate::SkipAhead) jumps `n` steps in
/// `O(log n)` (Brown 1994). [`SplittableRng::split`](crate::SplittableRng::split) with stream id
/// `i` uses the increment `c = 2i + 1`, and
/// [`substream`](crate::SplittableRng::substream) `(i, j)` starts `j · 2⁶⁴` steps into stream
/// `i`. Both are derived from the seeded starting state, not from the current position. Drawing
/// never fails.
///
/// # References
///
/// - M. E. O'Neill, "PCG: a family of simple fast space-efficient statistically good algorithms
///   for random number generation", Technical Report HMC-CS-2014-0905, Harvey Mudd College, 2014.
/// - F. B. Brown, "Random number generation with arbitrary strides", *Trans. Am. Nucl. Soc.* 71,
///   1994, 202–203.
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     BitGenerator, Pcg64, SeedableRng, SerializableState, SkipAhead, SplittableRng,
/// };
/// let base = Pcg64::from_seed(42);
/// let mut stream = base.split(7)?;
/// assert_eq!(stream.stream_identity().and_then(|id| id.selectors()), Some((7, 0)));
/// stream.skip_ahead(1_000)?;
/// let checkpoint = stream.export_state()?;
/// let word = stream.next_u64()?;
/// let mut restored = Pcg64::import_state(&checkpoint)?;
/// assert_eq!(restored.next_u64()?, word);
/// assert_eq!(restored.position(), Some(1_001));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Pcg64 {
    state: u128,
    increment: u128,
    origin: u128,
    selectors: Option<(u64, u64)>,
    counter: TransitionCounter,
}

impl Pcg64 {
    /// Step the internal state forward by one LCG step.
    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
        self.counter.increment();
    }

    /// Apply the PCG-XSL-RR output permutation to the current state.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    const fn output(&self) -> u64 {
        // XSL-RR: fold the 128-bit state to 64 bits by XOR-ing its halves,
        // then rotate by the top 6 bits. The high bits of an LCG state are the
        // best-mixed, so they choose the rotation while the weak low bits are
        // hidden by the fold.
        let state = self.state;
        let rot = (state >> 122) as u32;
        let xsl = ((state >> 64) ^ state) as u64;
        xsl.rotate_right(rot)
    }

    /// LCG fast-forward via modular exponentiation in u128.
    ///
    /// Computes the state after `n` steps without iterating:
    /// `state' = multiplier^n * state + (multiplier^n - 1) / (multiplier - 1) * increment`
    const fn advance(state: u128, increment: u128, n: u128) -> u128 {
        let mut cur_mult: u128 = MULTIPLIER;
        let mut cur_plus: u128 = increment;
        let mut acc_mult: u128 = 1;
        let mut acc_plus: u128 = 0;
        let mut delta = n;

        // Brown's (1994) binary decomposition. Invariant: (cur_mult, cur_plus)
        // is the affine map for 2^k steps and (acc_mult, acc_plus) is the map
        // for the low k bits of `n` already consumed. Squaring an affine map
        // x -> a x + c gives x -> a^2 x + (a + 1) c, and everything wraps
        // modulo 2^128 exactly as the step itself does, so no division by
        // `multiplier - 1` is needed.
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

impl BitGenerator for Pcg64 {
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
        // Advance, then output from the new state, as the 128-bit reference PCG does.
        self.step();
        Ok(self.output())
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

impl SeedableRng for Pcg64 {
    fn from_seed(seed: u64) -> Self {
        // Reference `pcg_setseq` seeding: from state 0, one step yields the
        // increment, the seed is added, and one more step mixes it in.
        let initial = DEFAULT_INCREMENT.wrapping_add(u128::from(seed));
        let state = Self::advance(initial, DEFAULT_INCREMENT, 1);
        Self {
            state,
            origin: state,
            increment: DEFAULT_INCREMENT,
            selectors: None,
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
}

impl SkipAhead for Pcg64 {
    #[inline]
    fn skip_ahead(&mut self, n: u128) -> Result<(), SciOxideError> {
        self.state = Self::advance(self.state, self.increment, n);
        self.counter = self.counter.advanced(n);
        Ok(())
    }
    // Every word or float read is one native transition.
    #[inline]
    fn read_transition_bound(&self, _: crate::traits::private::Token) -> Option<u128> {
        Some(1)
    }
}

impl SplittableRng for Pcg64 {
    #[inline]
    fn split(&self, stream_id: u64) -> Result<Self, SciOxideError> {
        self.substream(stream_id, 0)
    }
    #[inline]
    fn substream(&self, stream_id: u64, subseq_id: u64) -> Result<Self, SciOxideError> {
        // Each odd increment selects a distinct full-period LCG sequence, so
        // the stream id is encoded as `2 id + 1`. Subsequences are then carved
        // out of that stream in 2^64-step blocks. Both are addressed from the
        // shared origin so derivation is independent of this handle's position.
        let increment = (u128::from(stream_id) << 1) | 1;
        let start = Self::advance(self.origin, increment, 1);
        let state = Self::advance(start, increment, u128::from(subseq_id) << 64);
        Ok(Self {
            state,
            origin: self.origin,
            increment,
            selectors: Some((stream_id, subseq_id)),
            counter: TransitionCounter::from_position(Some(0)),
        })
    }
}

impl Pcg64 {
    #[allow(clippy::cast_possible_truncation)]
    fn identity(&self) -> StreamIdentity<'static> {
        StreamIdentity::new(
            Algorithm::Pcg64,
            [self.origin as u64, (self.origin >> 64) as u64, 0, 0, 0, 0],
            self.selectors,
        )
    }
}
impl StateEngine for Pcg64 {
    const ALGORITHM: Algorithm = Algorithm::Pcg64;
    const WORDS: usize = 2;
    const PARAMETER_BYTES: usize = 16;
    const HAS_STREAMS: bool = true;
    #[allow(clippy::cast_possible_truncation)]
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            identity: self.identity(),
            words: [self.state as u64, (self.state >> 64) as u64, 0, 0, 0, 0],
            parameter: self.increment,
            counter: self.counter.position(),
        }
    }
    fn from_snapshot(snapshot: Snapshot) -> Result<Self, SciOxideError> {
        // The increment is fully determined by the stream selector, so a
        // snapshot carrying any other increment would silently describe a
        // different sequence than its identity claims.
        let expected = snapshot
            .identity
            .selectors
            .map_or(DEFAULT_INCREMENT, |(id, _)| (u128::from(id) << 1) | 1);
        if snapshot.parameter != expected {
            return Err(state::malformed(
                "PCG increment must match immutable stream selection",
                scioxide_core::argument!("bytes"),
            ));
        }
        Ok(Self {
            state: u128::from(snapshot.words[0]) | (u128::from(snapshot.words[1]) << 64),
            origin: u128::from(snapshot.identity.origin[0])
                | (u128::from(snapshot.identity.origin[1]) << 64),
            increment: snapshot.parameter,
            selectors: snapshot.identity.selectors,
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
        let mut rng1 = Pcg64::from_seed(1);
        let mut rng2 = Pcg64::from_seed(2);
        assert_ne!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_float_range_f64() {
        let mut rng = Pcg64::from_seed(123);
        for _ in 0..1000 {
            let v: f64 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_fill_bytes() {
        let mut rng = Pcg64::from_seed(0);
        let mut buf = [0u8; 20];
        rng.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_skip_ahead() {
        let mut rng1 = Pcg64::from_seed(99);
        for _ in 0..100 {
            rng1.next_u64().unwrap();
        }

        let mut rng2 = Pcg64::from_seed(99);
        rng2.skip_ahead(100).unwrap();

        assert_eq!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_skip_to() {
        let mut rng1 = Pcg64::from_seed(7);
        for _ in 0..50 {
            rng1.next_u64().unwrap();
        }

        let mut rng2 = Pcg64::from_seed(7);
        rng2.skip_to(50).unwrap();

        assert_eq!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_skip_to_backward_errors() {
        let mut rng = Pcg64::from_seed(7);
        rng.skip_ahead(10).unwrap();
        let error = rng.skip_to(5).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
    }

    #[test]
    fn test_position_tracks_steps() {
        let mut rng = Pcg64::from_seed(42);
        assert_eq!(rng.position(), Some(0));
        rng.next_u64().unwrap();
        assert_eq!(rng.position(), Some(1));
        rng.skip_ahead(9).unwrap();
        assert_eq!(rng.position(), Some(10));
    }

    #[test]
    fn position_overflow_becomes_untracked_without_stopping_the_stream() {
        let mut rng = Pcg64::from_seed(7);
        rng.counter = TransitionCounter::from_position(Some(u128::MAX));
        let expected = {
            let mut clone = rng.clone();
            clone.state = Pcg64::advance(clone.state, clone.increment, 2);
            clone.state
        };
        rng.skip_ahead(1).unwrap();
        rng.skip_ahead(1).unwrap();
        assert_eq!(rng.state, expected);
        assert_eq!(rng.position(), None);
        assert!(rng.skip_to(0).is_err());
        assert_eq!(
            Pcg64::import_state(&rng.export_state().unwrap())
                .unwrap()
                .position(),
            None
        );
    }

    #[test]
    fn test_split_different_streams() {
        let base = Pcg64::from_seed(42);
        let mut stream0 = base.split(0).unwrap();
        let mut stream1 = base.split(1).unwrap();
        // Different streams should produce different output
        assert_ne!(stream0.next_u64().unwrap(), stream1.next_u64().unwrap());
    }

    #[test]
    fn test_split_deterministic() {
        let base = Pcg64::from_seed(42);
        let mut s1a = base.split(5).unwrap();
        let mut s1b = base.split(5).unwrap();
        assert_eq!(s1a.next_u64().unwrap(), s1b.next_u64().unwrap());
    }

    #[test]
    fn test_serializable_roundtrip() {
        let mut rng = Pcg64::from_seed(42);
        rng.next_u64().unwrap();
        let state = rng.export_state().unwrap();
        let mut rng2 = Pcg64::import_state(&state).unwrap();
        assert_eq!(rng.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_import_state_invalid_length() {
        let error = Pcg64::import_state(&[0; 10]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let error = Pcg64::import_state(&[0; 32]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_import_state_rejects_even_increment() {
        let mut state = Pcg64::from_seed(42).export_state().unwrap();
        state[21..37].fill(0);
        let error = Pcg64::import_state(&state).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Malformed,
                ..
            }
        ));
    }
}
