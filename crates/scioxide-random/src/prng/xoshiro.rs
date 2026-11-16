//! The xoshiro256** and xoshiro128** generators.

use scioxide_core::SciOxideError;

use crate::util::TransitionCounter;

use crate::prng::SplitMix64;
use crate::state::{self, GeneratorAlgorithm as Algorithm, Snapshot, StateEngine, StreamIdentity};
use crate::traits::{BitGenerator, JumpableRng, SeedableRng};

// ---------------------------------------------------------------------------
// Xoshiro256**
// ---------------------------------------------------------------------------

/// xoshiro256** generator with 64-bit output and period 2²⁵⁶ − 1.
///
/// The state is four 64-bit words updated by a linear xor/shift/rotate map over GF(2) with
/// shift 17 and rotation 45. The output applies the `**` scrambler `rotl(5 s₁, 7) · 9` to the
/// second state word before the update (Blackman and Vigna 2021).
/// [`next_u32`](crate::BitGenerator::next_u32) returns the low 32 bits of a 64-bit output.
///
/// Each output costs O(1). [`from_seed`](crate::SeedableRng::from_seed) fills the state with
/// four outputs of [`SplitMix64`](crate::SplitMix64) seeded with the seed. For parallel work,
/// [`jump`](crate::JumpableRng::jump) advances by 2¹²⁸ steps and
/// [`long_jump`](crate::JumpableRng::long_jump) by 2¹⁹² steps, each in 256 generator steps. The
/// position after a jump does not fit in `u128`, so [`position`](crate::BitGenerator::position)
/// returns `None`. Drawing never fails.
///
/// # References
///
/// D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators", *ACM Trans.
/// Math. Softw.* 47(4), 2021, article 36.
///
/// # Examples
///
/// ```
/// use scioxide_random::{BitGenerator, JumpableRng, SeedableRng, SerializableState};
/// use scioxide_random::Xoshiro256StarStar;
/// let mut worker = Xoshiro256StarStar::from_seed(42);
/// worker.jump()?;
/// assert_eq!(worker.position(), None); // 2^128 exceeds the tracked range
/// let checkpoint = worker.export_state()?;
/// let word = worker.next_u64()?;
/// worker.restore_state(&checkpoint)?;
/// assert_eq!(worker.next_u64()?, word);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
    origin: [u64; 4],
    counter: TransitionCounter,
}

impl Xoshiro256StarStar {
    #[inline]
    pub(super) fn next_word(&mut self) -> u64 {
        // The `**` scrambler (rotl(s1 * 5, 7) * 9) reads the pre-update state
        // as in the reference implementation, so output matches published
        // test vectors. Shift 17 and rotation 45 are the published xoshiro256
        // linear-engine parameters.
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        self.counter.increment();
        result
    }

    /// Fills the state from a single `u64` with `SplitMix64`, replacing an all-zero state.
    fn seed_state(seed: u64) -> [u64; 4] {
        let mut sm = SplitMix64::from_seed(seed);
        let mut s = [0u64; 4];
        for val in &mut s {
            *val = sm.next_word();
        }
        // The linear engine maps the all-zero state to itself, so that state
        // would emit zeros forever. SplitMix64 makes it essentially
        // unreachable, but the replacement keeps the invariant unconditional.
        if s == [0; 4] {
            s[0] = 1;
        }
        s
    }

    /// Advances the state by applying a jump polynomial.
    fn apply_jump(&mut self, poly: &[u64; 4]) {
        // The engine is linear over GF(2), so advancing by J steps equals
        // applying the jump polynomial p(T) = sum_i c_i T^i of the transition
        // map T. It is evaluated by stepping the engine once per coefficient
        // and XOR-accumulating every state whose coefficient bit is set. The
        // transient `next_word` calls also bump the counter; callers
        // overwrite it afterwards.
        let mut s = [0u64; 4];
        for &jump_word in poly {
            for bit in 0..64 {
                if (jump_word >> bit) & 1 != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_word();
            }
        }
        self.s = s;
    }
}

impl BitGenerator for Xoshiro256StarStar {
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

impl SeedableRng for Xoshiro256StarStar {
    fn from_seed(seed: u64) -> Self {
        let s = Self::seed_state(seed);
        Self {
            s,
            origin: s,
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
}

impl JumpableRng for Xoshiro256StarStar {
    /// Advances the generator by 2¹²⁸ steps. The position becomes unknown (`None`).
    fn jump(&mut self) -> Result<(), SciOxideError> {
        // Published jump polynomial for xoshiro256**
        self.apply_jump(&[
            0x180e_c6d3_3cfd_0aba,
            0xd5a6_1266_f0c9_392c,
            0xa958_2618_e03f_c9aa,
            0x39ab_dc45_29b1_661c,
        ]);
        // A 2^128-step jump cannot be represented by the u128 position
        // counter, so the position becomes unknown rather than wrapping.
        self.counter = TransitionCounter::from_position(None);
        Ok(())
    }

    /// Advances the generator by 2¹⁹² steps. The position becomes unknown (`None`).
    fn long_jump(&mut self) -> Result<(), SciOxideError> {
        self.apply_jump(&[
            0x76e1_5d3e_fefd_cbbf,
            0xc500_4e44_1c52_2fb3,
            0x7771_0069_854e_e241,
            0x3910_9bb0_2acb_e635,
        ]);
        self.counter = TransitionCounter::from_position(None);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Xoshiro128**
// ---------------------------------------------------------------------------

/// xoshiro128** generator with 32-bit output and period 2¹²⁸ − 1.
///
/// The state is four 32-bit words updated by a linear xor/shift/rotate map over GF(2) with
/// shift 9 and rotation 11. The output applies the `**` scrambler `rotl(5 s₁, 7) · 9` to the
/// second state word before the update (Blackman and Vigna 2021).
/// [`next_u64`](crate::BitGenerator::next_u64) joins two outputs, the first in the low half.
///
/// Each output costs O(1). [`from_seed`](crate::SeedableRng::from_seed) fills the state with two
/// outputs of [`SplitMix64`](crate::SplitMix64) seeded with the seed. For parallel work,
/// [`jump`](crate::JumpableRng::jump) advances by 2⁶⁴ steps and
/// [`long_jump`](crate::JumpableRng::long_jump) by 2⁹⁶ steps, each in 128 generator steps.
/// Drawing never fails.
///
/// # References
///
/// D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators", *ACM Trans.
/// Math. Softw.* 47(4), 2021, article 36.
///
/// # Examples
///
/// ```
/// use scioxide_random::{BitGenerator, JumpableRng, SeedableRng, SerializableState};
/// use scioxide_random::Xoshiro128StarStar;
/// let mut worker = Xoshiro128StarStar::from_seed(42);
/// worker.jump()?;
/// assert_eq!(worker.position(), Some(1 << 64));
/// let checkpoint = worker.export_state()?;
/// let word = worker.next_u32()?; // one native 32-bit word
/// let mut replay = Xoshiro128StarStar::import_state(&checkpoint)?;
/// assert_eq!(replay.next_u32()?, word);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Xoshiro128StarStar {
    s: [u32; 4],
    origin: [u32; 4],
    counter: TransitionCounter,
}

impl Xoshiro128StarStar {
    #[inline]
    pub(super) fn next_word(&mut self) -> u32 {
        // Same `**` scrambler as the 256-bit variant with the xoshiro128
        // engine parameters (shift 9, rotation 11).
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        self.counter.increment();
        result
    }

    /// Fills the state from a single `u64` with `SplitMix64`, replacing an all-zero state.
    fn seed_state(seed: u64) -> [u32; 4] {
        let mut sm = SplitMix64::from_seed(seed);
        let v0 = sm.next_word();
        let v1 = sm.next_word();
        #[allow(clippy::cast_possible_truncation)]
        let s = [v0 as u32, (v0 >> 32) as u32, v1 as u32, (v1 >> 32) as u32];
        if s == [0; 4] { [1, 0, 0, 0] } else { s }
    }

    /// Advances the state by applying a jump polynomial.
    fn apply_jump(&mut self, poly: &[u32; 4]) {
        let mut s = [0u32; 4];
        for &jump_word in poly {
            for bit in 0..32 {
                if (jump_word >> bit) & 1 != 0 {
                    s[0] ^= self.s[0];
                    s[1] ^= self.s[1];
                    s[2] ^= self.s[2];
                    s[3] ^= self.s[3];
                }
                self.next_word();
            }
        }
        self.s = s;
    }
}

impl BitGenerator for Xoshiro128StarStar {
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
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(self.next_word())
    }

    #[inline]
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        let lo = u64::from(self.next_u32()?);
        let hi = u64::from(self.next_u32()?);
        Ok((hi << 32) | lo)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
        crate::util::fill_bytes_with(dest, 4, || self.next_u32().map(u64::from))
    }
}

impl SeedableRng for Xoshiro128StarStar {
    fn from_seed(seed: u64) -> Self {
        let s = Self::seed_state(seed);
        Self {
            s,
            origin: s,
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
}

impl JumpableRng for Xoshiro128StarStar {
    /// Advances the generator by 2⁶⁴ steps.
    fn jump(&mut self) -> Result<(), SciOxideError> {
        // `apply_jump` steps the engine 128 times and so perturbs the counter;
        // the logical position is rebuilt from the value captured here. Unlike
        // the 256-bit variant, the jumped position still fits in u128.
        let before = self.counter;
        self.apply_jump(&[0x8764_000b, 0xf542_d2d3, 0x6fa0_35c3, 0x77f2_db5b]);
        self.counter = before.advanced(1_u128 << 64);
        Ok(())
    }

    /// Advances the generator by 2⁹⁶ steps.
    fn long_jump(&mut self) -> Result<(), SciOxideError> {
        let before = self.counter;
        self.apply_jump(&[0xb523_952e, 0x0b6f_099f, 0xccf5_a0ef, 0x1c58_0662]);
        self.counter = before.advanced(1_u128 << 96);
        Ok(())
    }
}

macro_rules! xoshiro_state {
    ($engine:ident, $word:ty) => {
        impl $engine {
            fn identity(&self) -> StreamIdentity<'static> {
                let [a, b, c, d] = self.origin.map(u64::from);
                StreamIdentity::new(Algorithm::$engine, [a, b, c, d, 0, 0], None)
            }
        }
        impl StateEngine for $engine {
            const ALGORITHM: Algorithm = Algorithm::$engine;
            const WORDS: usize = 4;
            const WORD_BYTES: usize = core::mem::size_of::<$word>();
            fn snapshot(&self) -> Snapshot {
                let [a, b, c, d] = self.s.map(u64::from);
                Snapshot {
                    identity: self.identity(),
                    words: [a, b, c, d, 0, 0],
                    parameter: 0,
                    counter: self.counter.position(),
                }
            }
            fn from_snapshot(snapshot: Snapshot) -> Result<Self, SciOxideError> {
                // Narrow words have already been decoded at their native width.
                #[allow(clippy::cast_possible_truncation)]
                let s = core::array::from_fn(|i| snapshot.words[i] as $word);
                #[allow(clippy::cast_possible_truncation)]
                let origin = core::array::from_fn(|i| snapshot.identity.origin[i] as $word);
                // Never repair an absorbing zero state during restoration.
                if s == [0; 4] || origin == [0; 4] {
                    return Err(state::malformed(
                        "Xoshiro current state and origin must not be all zero",
                        scioxide_core::argument!("bytes"),
                    ));
                }
                Ok(Self {
                    s,
                    origin,
                    counter: TransitionCounter::from_position(snapshot.counter),
                })
            }
        }
    };
}
xoshiro_state!(Xoshiro256StarStar, u64);
xoshiro_state!(Xoshiro128StarStar, u32);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Rng, SerializableState};

    // ---- Xoshiro256StarStar ----

    #[test]
    fn test_xoshiro256_different_seeds() {
        let mut rng1 = Xoshiro256StarStar::from_seed(1);
        let mut rng2 = Xoshiro256StarStar::from_seed(2);
        assert_ne!(rng1.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_xoshiro256_float_range_f64() {
        let mut rng = Xoshiro256StarStar::from_seed(123);
        for _ in 0..1000 {
            let v: f64 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_xoshiro256_float_range_f32() {
        let mut rng = Xoshiro256StarStar::from_seed(456);
        for _ in 0..1000 {
            let v: f32 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_xoshiro256_fill_bytes() {
        let mut rng = Xoshiro256StarStar::from_seed(0);
        let mut buf = [0u8; 20];
        rng.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_xoshiro256_serializable_roundtrip() {
        let mut rng = Xoshiro256StarStar::from_seed(42);
        rng.next_u64().unwrap();
        let state = rng.export_state().unwrap();
        let mut rng2 = Xoshiro256StarStar::import_state(&state).unwrap();
        assert_eq!(rng.next_u64().unwrap(), rng2.next_u64().unwrap());
    }

    #[test]
    fn test_xoshiro256_import_state_invalid_length() {
        let error = Xoshiro256StarStar::import_state(&[0; 10]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_xoshiro256_import_state_all_zero() {
        let mut state = Xoshiro256StarStar::from_seed(42).export_state().unwrap();
        state[crate::state::PREFIX_BYTES..].fill(0);
        let error = Xoshiro256StarStar::import_state(&state).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    // ---- Xoshiro128StarStar ----

    #[test]
    fn test_xoshiro128_different_seeds() {
        let mut rng1 = Xoshiro128StarStar::from_seed(1);
        let mut rng2 = Xoshiro128StarStar::from_seed(2);
        assert_ne!(rng1.next_u32().unwrap(), rng2.next_u32().unwrap());
    }

    #[test]
    fn test_xoshiro128_float_range_f64() {
        let mut rng = Xoshiro128StarStar::from_seed(123);
        for _ in 0..1000 {
            let v: f64 = rng.next_float().unwrap();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_xoshiro128_next_u64() {
        let mut rng = Xoshiro128StarStar::from_seed(42);
        let v = rng.next_u64().unwrap();
        // Should combine two u32s
        let mut rng2 = Xoshiro128StarStar::from_seed(42);
        let lo = u64::from(rng2.next_u32().unwrap());
        let hi = u64::from(rng2.next_u32().unwrap());
        assert_eq!(v, (hi << 32) | lo);
    }

    #[test]
    fn test_xoshiro128_fill_bytes() {
        let mut rng = Xoshiro128StarStar::from_seed(0);
        let mut buf = [0u8; 15];
        rng.fill_bytes(&mut buf).unwrap();
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_xoshiro128_serializable_roundtrip() {
        let mut rng = Xoshiro128StarStar::from_seed(42);
        rng.next_u32().unwrap();
        let state = rng.export_state().unwrap();
        let mut rng2 = Xoshiro128StarStar::import_state(&state).unwrap();
        assert_eq!(rng.next_u32().unwrap(), rng2.next_u32().unwrap());
    }

    #[test]
    fn test_xoshiro128_import_state_invalid_length() {
        let error = Xoshiro128StarStar::import_state(&[0; 10]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_xoshiro128_import_state_all_zero() {
        let mut state = Xoshiro128StarStar::from_seed(42).export_state().unwrap();
        state[crate::state::PREFIX_BYTES..].fill(0);
        let error = Xoshiro128StarStar::import_state(&state).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
}
