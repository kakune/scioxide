//! Float decoding and checked batch draws over scalar-independent bit engines.

use scioxide_core::{RealScalar, SciOxideError};

use crate::{
    BitGenerator, JumpableRng, Lcg64, Mrg32k3a, Pcg64, Rng, SkipAhead, SplitMix64,
    Xoshiro128StarStar, Xoshiro256StarStar,
};

/// Fills `output` with uniform floats in `[0, 1)` and counts the draws in `attempts`.
///
/// `attempts` is increased by one for every call to the generator, including a
/// call that fails. It is cumulative, and it must have room for the whole slice
/// before the first draw. Values from a custom [`Rng`] are checked as described
/// under [uniform variates](crate#uniform-variates). A generator failure stops
/// the fill and leaves the values drawn so far in `output`. An invalid value is
/// reported only after the whole slice is drawn, so a later generator failure
/// takes precedence. On error, `output` may hold unchecked values.
///
/// ```
/// use scioxide_random::{fill_uniforms, Philox4x64, SeedableRng};
/// let mut rng = Philox4x64::from_seed(7);
/// let mut values = [0.0_f64; 4];
/// let mut attempts = 0;
/// fill_uniforms(&mut rng, &mut values, &mut attempts)?;
/// assert_eq!(attempts, 4);
/// assert!(values.iter().all(|&u| (0.0..1.0).contains(&u)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// - `SizeOverflow` before any draw if `attempts + output.len()` overflows `usize`.
/// - Any error returned by the generator, such as an exhausted [`BoundedSubstream`](crate::BoundedSubstream).
/// - `NonFinite` or `InvalidInput` for a non-finite or out-of-range value from a custom [`Rng`].
#[inline]
pub fn fill_uniforms<S: RealScalar, R: Rng<S> + ?Sized>(
    rng: &mut R,
    output: &mut [S],
    attempts: &mut usize,
) -> Result<(), SciOxideError> {
    attempts
        .checked_add(output.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("uniform_draws")))?;
    rng.fill_checked(output, attempts, crate::traits::private::Token)
}

/// Converts a 64-bit word to a float in `[0, 1)` from its 24 (`f32`) or 53 (`f64`) high bits.
///
/// The result is `⌊w / 2^(64−p)⌋ · 2^(−p)` for word `w` and precision `p`, exactly.
/// It is uniform only if the word is, so pass a full-width uniform word such as
/// [`BitGenerator::next_uniform_u64`]; native outputs may have a restricted range.
///
/// ```
/// use scioxide_random::{BitGenerator, SeedableRng, SplitMix64, decode_u64};
/// assert_eq!(decode_u64::<f64>(1 << 63), 0.5); // dyadic values decode exactly
/// assert!(decode_u64::<f32>(u64::MAX) < 1.0);
/// let mut rng = SplitMix64::from_seed(42);
/// let value: f64 = decode_u64(rng.next_uniform_u64()?);
/// assert!((0.0..1.0).contains(&value));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn decode_u64<S: RealScalar>(word: u64) -> S {
    S::from_u64_half_open01(word)
}

/// Converts a 32-bit word to a float in `[0, 1)` from its 24 high bits (`f32`) or all 32 bits (`f64`).
///
/// The result is uniform only if the word is, so pass a full-width uniform word
/// such as [`BitGenerator::next_uniform_u32`].
///
/// ```
/// use scioxide_random::{BitGenerator, SeedableRng, Xoshiro128StarStar, decode_u32};
/// assert_eq!(decode_u32::<f64>(1 << 30), 0.25); // dyadic values decode exactly
/// assert!(decode_u32::<f32>(u32::MAX) < 1.0);
/// let mut rng = Xoshiro128StarStar::from_seed(42);
/// let value: f32 = decode_u32(rng.next_uniform_u32()?);
/// assert!((0.0..1.0).contains(&value));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn decode_u32<S: RealScalar>(word: u32) -> S {
    S::from_u32_half_open01(word)
}

/// Built-in float decoding. Every implementation returns a finite value in
/// `[0, 1)` by construction, which lets the engines' [`Rng`] implementations
/// skip provider validation.
pub(crate) trait FloatSource<S: RealScalar>: BitGenerator {
    fn draw_float(&mut self) -> Result<S, scioxide_core::SciOxideError>;

    /// The caller has checked that `attempts` has room for the whole output.
    #[inline]
    fn fill_floats(&mut self, output: &mut [S], attempts: &mut usize) -> Result<(), SciOxideError> {
        for value in output {
            *attempts += 1;
            *value = self.draw_float()?;
        }
        Ok(())
    }
}

macro_rules! word_decoder {
    ($engine:ty, $word:ident, $decode:ident) => {
        impl<S: RealScalar> FloatSource<S> for $engine {
            #[inline]
            fn draw_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
                self.$word().map($decode)
            }
        }
    };
}
word_decoder!(SplitMix64, next_u64, decode_u64);
word_decoder!(Lcg64, next_u64, decode_u64);
word_decoder!(Pcg64, next_u64, decode_u64);
word_decoder!(Xoshiro256StarStar, next_u64, decode_u64);
word_decoder!(Xoshiro128StarStar, next_u32, decode_u32);

impl FloatSource<f64> for Mrg32k3a {
    #[inline]
    fn draw_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
        self.next_u32()
            // RngStreams normalization `word / (m1 + 1)`. Native words lie in
            // [1, m1], so the result is strictly inside (0, 1) and matches
            // published RngStreams uniforms bit-for-bit.
            .map(|word| f64::from(word) * (1.0 / 4_294_967_088.0))
    }
}
// The RngStreams constant would round the largest words to 1.0 in f32, so f32
// falls back to the generic top-24-bit decoder, which is always below 1.
impl FloatSource<f32> for Mrg32k3a {
    #[inline]
    fn draw_float(&mut self) -> Result<f32, scioxide_core::SciOxideError> {
        self.next_u32().map(decode_u32)
    }
}

/// Implements [`Rng`] for a built-in engine through its [`FloatSource`] decoder.
macro_rules! decoded_rng {
    ([$($generics:tt)*] $scalar:ty, $engine:ty) => {
        impl $($generics)* Rng<$scalar> for $engine {
            #[inline]
            fn next_float(&mut self) -> Result<$scalar, SciOxideError> {
                FloatSource::<$scalar>::draw_float(self)
            }
            #[inline]
            fn next_open_float(&mut self) -> Result<$scalar, SciOxideError> {
                crate::util::open_probability(crate::util::MAX_SAMPLING_ATTEMPTS, || {
                    FloatSource::<$scalar>::draw_float(self)
                })
            }
            // Built-in decoders prove the half-open range, so the checked
            // boundary is the decoder itself.
            #[inline]
            fn checked_open_float(
                &mut self,
                _: crate::traits::private::Token,
            ) -> Result<$scalar, SciOxideError> {
                self.next_open_float()
            }
            #[inline]
            fn checked_float(
                &mut self,
                _: scioxide_core::Argument,
                _: crate::traits::private::Token,
            ) -> Result<$scalar, SciOxideError> {
                FloatSource::<$scalar>::draw_float(self)
            }
            // A dedicated producer loop keeps engine-buffer state local without
            // inlining a counter round into the consumer's numerical kernel.
            #[inline(never)]
            fn fill_checked(
                &mut self,
                output: &mut [$scalar],
                attempts: &mut usize,
                _: crate::traits::private::Token,
            ) -> Result<(), SciOxideError> {
                FloatSource::<$scalar>::fill_floats(self, output, attempts)
            }
        }
    };
}
macro_rules! engine_rng {
    ($($engine:ty),+ $(,)?) => {$(
        decoded_rng!([<S: RealScalar>] S, $engine);
    )+};
}
macro_rules! modular_rng {
    ($($engine:ty),+ $(,)?) => {$(
        decoded_rng!([] f64, $engine);
        decoded_rng!([] f32, $engine);
    )+};
}
engine_rng!(
    SplitMix64,
    Lcg64,
    Pcg64,
    Xoshiro256StarStar,
    Xoshiro128StarStar,
    crate::Mt19937,
    crate::Ranlux,
    crate::Ranlxs,
    crate::Ranlxd,
    crate::Taus2,
    crate::Taus113,
    crate::Gfsr4,
    crate::Philox4x64,
    crate::Threefry4x64,
);
modular_rng!(Mrg32k3a, crate::Mrg, crate::Cmrg);

impl<B: BitGenerator + ?Sized> BitGenerator for &mut B {
    #[inline]
    fn position(&self) -> Option<u128> {
        (**self).position()
    }
    fn stream_identity(&self) -> Option<crate::StreamIdentity<'_>> {
        (**self).stream_identity()
    }
    #[inline]
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        (**self).next_u32()
    }
    #[inline]
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        (**self).next_u64()
    }
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        (**self).next_uniform_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        (**self).next_uniform_u64()
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        (**self).fill_bytes(output)?;
        Ok(())
    }
}
impl<B: SkipAhead + ?Sized> SkipAhead for &mut B {
    fn skip_ahead(&mut self, n: u128) -> Result<(), SciOxideError> {
        (**self).skip_ahead(n)
    }
    fn skip_to(&mut self, index: u128) -> Result<(), SciOxideError> {
        (**self).skip_to(index)
    }
    #[inline]
    fn read_transition_bound(&self, token: crate::traits::private::Token) -> Option<u128> {
        (**self).read_transition_bound(token)
    }
}
impl<B: JumpableRng + ?Sized> JumpableRng for &mut B {
    fn jump(&mut self) -> Result<(), SciOxideError> {
        (**self).jump()
    }
    fn long_jump(&mut self) -> Result<(), SciOxideError> {
        (**self).long_jump()
    }
}

/// Binary64 midpoints on the 52-bit grid, preserving count-sampler streams.
/// This deterministic decoder needs one uniform word and no rejection.
pub(crate) fn open_midpoint<R: BitGenerator + ?Sized>(rng: &mut R) -> Result<f64, SciOxideError> {
    Ok(((rng.next_uniform_u64()? >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0))
}

/// Native-width half-open decoding with bounded zero rejection. This keeps
/// the test-statistic sampler's 24/53-bit grid and word accounting.
pub(crate) fn open_word_probability<S: RealScalar, R: BitGenerator>(
    rng: &mut R,
    limit: scioxide_core::Budget,
) -> Result<S, SciOxideError> {
    crate::util::open_probability(limit.get().min(crate::util::MAX_SAMPLING_ATTEMPTS), || {
        Ok(decode_u64(rng.next_uniform_u64()?))
    })
}

/// Deterministic QMC endpoint squeeze; it neither draws nor rejects a point.
pub(crate) fn open_coordinate(u: f64) -> f64 {
    u * (1.0 - f64::EPSILON) + 0.5 * f64::EPSILON
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::{
        Distribution, Gamma, GumbelRight, Logistic, SeedableRng, SerializableState, Uniform,
        WeightedSampler,
    };

    #[test]
    fn counted_batches_match_scalar_decoding_and_borrowed_state() {
        fn check<S: RealScalar, B: Rng<S> + Clone + SerializableState>(engine: B) {
            let mut batch = engine.clone();
            let mut scalar = engine;
            let mut output = [S::ZERO; 33];
            let mut attempts = 7;
            let mut expected_attempts = 7;
            for size in [0, 1, 3, 4, 9, 33] {
                let mut borrowed = &mut batch;
                fill_uniforms(&mut borrowed, &mut output[..size], &mut attempts).unwrap();
                expected_attempts += size;
                assert_eq!(attempts, expected_attempts);
                for &value in &output[..size] {
                    let expected: S = scalar.next_float().unwrap();
                    assert_eq!(value.into_f64().to_bits(), expected.into_f64().to_bits());
                }
                assert_eq!(batch.position(), scalar.position());
                let mut batch_state = alloc::vec![0; B::STATE_BYTES];
                let mut scalar_state = alloc::vec![0; B::STATE_BYTES];
                batch.export_state_into(&mut batch_state).unwrap();
                scalar.export_state_into(&mut scalar_state).unwrap();
                assert_eq!(batch_state, scalar_state);
            }
        }
        macro_rules! engine {
            ($engine:ty) => {{
                check::<f32, _>(<$engine>::from_seed(7));
                check::<f64, _>(<$engine>::from_seed(7));
            }};
        }
        engine!(SplitMix64);
        engine!(Pcg64);
        engine!(Mrg32k3a);
        engine!(Xoshiro128StarStar);
        engine!(crate::Philox4x64);
        engine!(crate::Threefry4x64);
    }

    #[test]
    fn counted_batches_preserve_partial_failure_and_preflight_overflow() {
        let mut rng = crate::Philox4x64::from_seed(7);
        let mut output = [-1.0_f64; 4];
        let mut attempts = usize::MAX;
        assert!(fill_uniforms(&mut rng, &mut output, &mut attempts).is_err());
        assert_eq!(attempts, usize::MAX);
        assert_eq!(rng.position(), Some(0));
        assert_eq!(output, [-1.0; 4]);
        rng.skip_ahead((1_u128 << 66) - 2).unwrap();
        let mut scalar = rng.clone();
        attempts = 0;
        assert!(fill_uniforms(&mut rng, &mut output, &mut attempts).is_err());
        assert_eq!(attempts, 3);
        assert_eq!(rng.position(), Some(1_u128 << 66));
        for &value in &output[..2] {
            let expected: f64 = scalar.next_float().unwrap();
            assert_eq!(value.to_bits(), expected.to_bits());
        }
        assert_eq!(output[2..], [-1.0; 2]);
        // The final aligned block succeeds completely; the next request records
        // one failing attempt and leaves its output untouched.
        let mut aligned = crate::Philox4x64::from_seed(7);
        aligned.skip_ahead((1_u128 << 66) - 4).unwrap();
        let mut scalar = aligned.clone();
        attempts = 0;
        fill_uniforms(&mut aligned, &mut output, &mut attempts).unwrap();
        assert_eq!(attempts, 4);
        assert_eq!(aligned.position(), Some(1_u128 << 66));
        for &value in &output {
            let expected: f64 = scalar.next_float().unwrap();
            assert_eq!(value.to_bits(), expected.to_bits());
        }
        let before = output;
        assert!(fill_uniforms(&mut aligned, &mut output, &mut attempts).is_err());
        assert_eq!(attempts, 5);
        assert_eq!(output, before);
        let mut invalid = One(SplitMix64::from_seed(7));
        attempts = 0;
        assert!(fill_uniforms(&mut invalid, &mut output, &mut attempts).is_err());
        assert_eq!(attempts, 4);
        assert_eq!(output, [1.0; 4]);
    }

    /// A custom provider: only the raw protocol, so every value is validated.
    struct Custom<R>(R);

    impl<R: BitGenerator> BitGenerator for Custom<R> {
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            self.0.next_u32()
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.0.next_u64()
        }
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.0.next_uniform_u32()
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.0.next_uniform_u64()
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
            self.0.fill_bytes(dest)
        }
    }
    impl<R: Rng<f64>> Rng<f64> for Custom<R> {
        fn next_float(&mut self) -> Result<f64, SciOxideError> {
            self.0.next_float()
        }
    }

    /// A custom provider whose floats are always the invalid endpoint one.
    struct One(SplitMix64);

    impl BitGenerator for One {
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            self.0.next_u32()
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.0.next_u64()
        }
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.0.next_uniform_u32()
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.0.next_uniform_u64()
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
            self.0.fill_bytes(dest)
        }
    }
    impl Rng<f64> for One {
        fn next_float(&mut self) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
    }

    #[test]
    fn built_in_decoders_skip_only_redundant_validation() {
        fn draws<R: Rng<f64>>(rng: &mut R) -> Vec<u64> {
            let uniform = Uniform::new(-1.0, 2.0).unwrap();
            let gamma = Gamma::new(0.4, 1.5).unwrap();
            let logistic = Logistic::new(1.0, 2.0).unwrap();
            let weighted = WeightedSampler::new(&[1.0, 2.0, 3.0]).unwrap();
            (0..256)
                .flat_map(|_| {
                    [
                        uniform.sample(rng).unwrap().to_bits(),
                        gamma.sample(rng).unwrap().to_bits(),
                        logistic.sample(rng).unwrap().to_bits(),
                        GumbelRight::new().sample(rng).unwrap().to_bits(),
                        weighted.sample_index(rng).unwrap() as u64,
                        rng.next_open_float().unwrap().to_bits(),
                    ]
                })
                .collect()
        }
        let built_in = draws(&mut SplitMix64::from_seed(1345));
        let custom = draws(&mut Custom(SplitMix64::from_seed(1345)));
        assert_eq!(built_in, custom);
    }

    #[test]
    fn custom_providers_remain_validated() {
        let invalid = |error: SciOxideError| {
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        };
        let mut rng = One(SplitMix64::from_seed(1345));
        invalid(
            Uniform::new(0.0, 1.0)
                .unwrap()
                .sample(&mut rng)
                .unwrap_err(),
        );
        invalid(Gamma::new(2.0, 1.0).unwrap().sample(&mut rng).unwrap_err());
        invalid(
            Logistic::new(0.0, 1.0)
                .unwrap()
                .sample(&mut rng)
                .unwrap_err(),
        );
        invalid(GumbelRight::new().sample(&mut rng).unwrap_err());
        invalid(
            WeightedSampler::new(&[1.0, 2.0])
                .unwrap()
                .sample_index(&mut rng)
                .unwrap_err(),
        );
        invalid(rng.next_open_float().unwrap_err());
    }

    #[test]
    fn owned_and_borrowed_decoders_share_raw_state_without_cached_words() {
        let mut raw = SplitMix64::from_seed(42);
        let mut expected = raw.clone();
        let expected_f32 = decode_u64::<f32>(expected.next_u64().unwrap());
        let expected_f64 = decode_u64::<f64>(expected.next_u64().unwrap());
        {
            let borrowed = &mut raw;
            let first: f32 = borrowed.next_float().unwrap();
            let second: f64 = borrowed.next_float().unwrap();
            assert_eq!(first.to_bits(), expected_f32.to_bits());
            assert_eq!(second.to_bits(), expected_f64.to_bits());
        }
        assert_eq!(raw.next_u64().unwrap(), expected.next_u64().unwrap());
        assert_eq!(raw.next_u32().unwrap(), expected.next_u32().unwrap());
    }
}
