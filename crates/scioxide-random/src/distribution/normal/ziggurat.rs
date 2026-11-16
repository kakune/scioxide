//! Typed, stateless Normal Ziggurat sampling leaves.
//!
//! The `f32` and `f64` tables in `ziggurat_tables.rs` were derived independently
//! from the 256-layer equal-area construction for `f(x) = exp(-x²/2)` at 100
//! decimal digits. With cutoff `R` and layer area `V = R f(R) + ∫_R^∞ f`, the
//! boundaries follow `x[i] = sqrt(-2 ln(V / x[i+1] + f(x[i+1])))` from
//! `x[255] = R`. Values are rounded to the target type only after that
//! construction, and are stored as exact IEEE bit patterns.
//!
//! One full-width uniform word supplies independent fields. The low eight bits
//! select the layer, bit eight selects the sign, and the next 23 (`f32`) or 52 (`f64`) bits
//! give the rectangle magnitude. The exponential tail uses two open-interval
//! uniforms within the common attempt budget.

#![allow(clippy::cast_precision_loss)]

#[cfg(test)]
use alloc::vec::Vec;
#[cfg(test)]
use scioxide_core::NonFiniteOrigin;
use scioxide_core::{RealScalar, SciOxideError};

use super::ziggurat_tables::{F_F32, F_F64, K_F32, K_F64, R_F32, R_F64, X_F32, X_F64};
use crate::traits::Rng;
use crate::util::{MAX_SAMPLING_ATTEMPTS, validated_unit_float};
// One uniform word is split into disjoint bit fields so the common fast-accept path
// costs exactly one generator call: bits 0..8 select one of the 256 equal-area
// layers, bit 8 carries the sign, and the bits from 9 upward supply the mantissa
// of the horizontal coordinate. Disjoint fields keep layer, sign and magnitude
// independent, which the symmetric two-sided construction relies on.
const LAYER_MASK: u32 = 0xff;
const SIGN_MASK: u32 = 0x100;
const F32_MAGNITUDE_MASK: u32 = (1 << 23) - 1;
const F64_MAGNITUDE_MASK: u64 = (1 << 52) - 1;
const F32_ONE_BITS: u32 = 1.0_f32.to_bits();
const F64_ONE_BITS: u64 = 1.0_f64.to_bits();

#[inline(always)]
fn typed_uniform_f32<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<f32, SciOxideError> {
    f32::try_from(validated_unit_float(rng, scioxide_core::argument!("rng"))?.value())
}

#[inline(always)]
fn typed_uniform_f64<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<f64, SciOxideError> {
    f64::try_from(validated_unit_float(rng, scioxide_core::argument!("rng"))?.value())
}

// The sign is applied by flipping the IEEE sign bit rather than by a branch on
// negation; this is exact and keeps the fast path branch-free.
#[inline(always)]
fn signed_f32(value: f32, negative: bool) -> f32 {
    f32::from_bits(value.to_bits() ^ (u32::from(negative) << 31))
}

#[inline(always)]
fn signed_f64(value: f64, negative: bool) -> f64 {
    f64::from_bits(value.to_bits() ^ (u64::from(negative) << 63))
}

#[cold]
fn tail_f32<S: RealScalar, R: Rng<S>>(
    rng: &mut R,
    negative: bool,
) -> Result<Option<f32>, SciOxideError> {
    // Marsaglia's tail method for |x| > R: propose x = R + E1 / R with
    // E1, E2 ~ Exp(1) and accept when 2 E2 > (E1 / R)^2. This samples the
    // normal tail exactly without an inverse-CDF evaluation. Zero uniforms are
    // rejected because `ln(0)` would yield an infinite candidate; `None` makes
    // the caller restart with a fresh word instead of failing.
    let radial_uniform = typed_uniform_f32(rng)?;
    let acceptance_uniform = typed_uniform_f32(rng)?;
    if radial_uniform <= 0.0 || acceptance_uniform <= 0.0 {
        return Ok(None);
    }

    let radial = -radial_uniform.ln() / R_F32;
    let acceptance = -acceptance_uniform.ln();
    let candidate = R_F32 + radial;
    Ok(
        (candidate.is_finite() && 2.0 * acceptance > radial * radial)
            .then(|| signed_f32(candidate, negative)),
    )
}

#[cold]
fn tail_f64<S: RealScalar, R: Rng<S>>(
    rng: &mut R,
    negative: bool,
) -> Result<Option<f64>, SciOxideError> {
    let radial_uniform = typed_uniform_f64(rng)?;
    let acceptance_uniform = typed_uniform_f64(rng)?;
    if radial_uniform <= 0.0 || acceptance_uniform <= 0.0 {
        return Ok(None);
    }

    let radial = -radial_uniform.ln() / R_F64;
    let acceptance = -acceptance_uniform.ln();
    let candidate = R_F64 + radial;
    Ok(
        (candidate.is_finite() && 2.0 * acceptance > radial * radial)
            .then(|| signed_f64(candidate, negative)),
    )
}

// The first word's inner-rectangle test is the hot path; the cold remainder
// re-runs that first word through the complete three-stage loop. Keeping the
// wedge and tail out of line keeps every caller's register footprint small.
#[inline(always)]
pub(super) fn sample_f32<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<f32, SciOxideError> {
    let word = rng.next_uniform_u32()?;
    let layer = (word & LAYER_MASK) as usize;
    let magnitude = (word >> 9) & F32_MAGNITUDE_MASK;
    if magnitude < K_F32[layer] {
        let unit = f32::from_bits(F32_ONE_BITS | magnitude) - 1.0;
        return Ok(signed_f32(unit * X_F32[layer], word & SIGN_MASK != 0));
    }
    rejection_f32(rng, word)
}

#[cold]
#[inline(never)]
fn rejection_f32<S: RealScalar, R: Rng<S>>(rng: &mut R, first: u32) -> Result<f32, SciOxideError> {
    let mut word = first;
    for attempt in 0..MAX_SAMPLING_ATTEMPTS {
        if attempt > 0 {
            word = rng.next_uniform_u32()?;
        }
        let layer = (word & LAYER_MASK) as usize;
        let negative = word & SIGN_MASK != 0;
        let magnitude = (word >> 9) & F32_MAGNITUDE_MASK;
        // OR-ing the mantissa into the bit pattern of 1.0 gives a float in
        // [1, 2); subtracting 1 yields a uniform on [0, 1) with a full 23-bit
        // grid and no integer-to-float rounding.
        let unit = f32::from_bits(F32_ONE_BITS | magnitude) - 1.0;
        let candidate = unit * X_F32[layer];

        // `K[layer]` is the fraction of the layer width that lies entirely
        // under the density, pre-scaled to mantissa units. The integer compare
        // therefore accepts the inner rectangle without evaluating `exp`; this
        // is the dominant fast path. `K[1]` is zero because the top cap has
        // no inner rectangle.
        if magnitude < K_F32[layer] {
            return Ok(signed_f32(candidate, negative));
        }
        // Layer 0 is the base strip whose width `V / f(R)` covers both
        // [0, R) and, by area, the unbounded tail. The part below R was
        // accepted above, so anything else must come from the tail sampler.
        if layer == 0 {
            if let Some(tail) = tail_f32(rng, negative)? {
                return Ok(tail);
            }
            continue;
        }

        // Wedge test: the candidate lies between the inner rectangle and the
        // layer's outer edge, so draw a height uniformly between the layer's
        // lower bound `f(x[layer])` and upper bound `f(x[layer - 1])` and
        // accept under the exact density.
        let uniform = typed_uniform_f32(rng)?;
        let vertical = F_F32[layer] + uniform * (F_F32[layer - 1] - F_F32[layer]);
        if vertical < (-0.5 * candidate * candidate).exp() {
            return Ok(signed_f32(candidate, negative));
        }
    }
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[inline(always)]
pub(super) fn sample_f64<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<f64, SciOxideError> {
    let word = rng.next_uniform_u64()?;
    let layer = (word as u32 & LAYER_MASK) as usize;
    let magnitude = (word >> 9) & F64_MAGNITUDE_MASK;
    if magnitude < K_F64[layer] {
        let unit = f64::from_bits(F64_ONE_BITS | magnitude) - 1.0;
        let negative = word & u64::from(SIGN_MASK) != 0;
        return Ok(signed_f64(unit * X_F64[layer], negative));
    }
    rejection_f64(rng, word)
}

#[cold]
#[inline(never)]
fn rejection_f64<S: RealScalar, R: Rng<S>>(rng: &mut R, first: u64) -> Result<f64, SciOxideError> {
    let mut word = first;
    for attempt in 0..MAX_SAMPLING_ATTEMPTS {
        if attempt > 0 {
            word = rng.next_uniform_u64()?;
        }
        // Same bit-field split and three-stage test as `rejection_f32`; the
        // wider word supplies a 52-bit mantissa so f64 samples have full
        // double-precision resolution rather than being widened f32 values.
        let layer = (word as u32 & LAYER_MASK) as usize;
        let negative = word & u64::from(SIGN_MASK) != 0;
        let magnitude = (word >> 9) & F64_MAGNITUDE_MASK;
        let unit = f64::from_bits(F64_ONE_BITS | magnitude) - 1.0;
        let candidate = unit * X_F64[layer];

        if magnitude < K_F64[layer] {
            return Ok(signed_f64(candidate, negative));
        }
        if layer == 0 {
            if let Some(tail) = tail_f64(rng, negative)? {
                return Ok(tail);
            }
            continue;
        }

        let uniform = typed_uniform_f64(rng)?;
        let vertical = F_F64[layer] + uniform * (F_F64[layer - 1] - F_F64[layer]);
        if vertical < (-0.5 * candidate * candidate).exp() {
            return Ok(signed_f64(candidate, negative));
        }
    }
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::traits::BitGenerator;

    struct ScriptedRng<S> {
        words: std::vec::IntoIter<u64>,
        uniforms: std::vec::IntoIter<S>,
        raw_draws: usize,
        uniform_draws: usize,
    }

    impl<S> ScriptedRng<S> {
        fn new(words: Vec<u64>, uniforms: Vec<S>) -> Self {
            Self {
                words: words.into_iter(),
                uniforms: uniforms.into_iter(),
                raw_draws: 0,
                uniform_draws: 0,
            }
        }
    }

    impl<S> BitGenerator for ScriptedRng<S> {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.raw_draws += 1;
            Ok(self.words.next().unwrap() as u32)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.raw_draws += 1;
            Ok(self.words.next().unwrap())
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl<S: RealScalar> Rng<S> for ScriptedRng<S> {
        fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
            self.uniform_draws += 1;
            Ok(self.uniforms.next().unwrap())
        }
    }

    const fn word_f32(layer: u32, negative: bool, magnitude: u32) -> u64 {
        let sign = if negative { SIGN_MASK } else { 0 };
        (((magnitude & F32_MAGNITUDE_MASK) << 9) | sign | (layer & LAYER_MASK)) as u64
    }

    const fn word_f64(layer: u64, negative: bool, magnitude: u64) -> u64 {
        let sign = if negative { SIGN_MASK as u64 } else { 0 };
        ((magnitude & F64_MAGNITUDE_MASK) << 9) | sign | (layer & LAYER_MASK as u64)
    }

    #[test]
    fn zero_words_fast_accept_without_uniforms() {
        let mut rng_f32 = ScriptedRng::<f32>::new(vec![0], vec![]);
        assert_eq!(
            sample_f32(&mut rng_f32).unwrap().to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!((rng_f32.raw_draws, rng_f32.uniform_draws), (1, 0));

        let mut rng_f64 = ScriptedRng::<f64>::new(vec![0], vec![]);
        assert_eq!(
            sample_f64(&mut rng_f64).unwrap().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!((rng_f64.raw_draws, rng_f64.uniform_draws), (1, 0));
    }

    #[test]
    fn maximum_words_take_finite_negative_density_path() {
        let mut rng_f32 = ScriptedRng::new(vec![u64::from(u32::MAX)], vec![0.0_f32]);
        let sample_f32 = sample_f32(&mut rng_f32).unwrap();
        assert!(sample_f32.is_finite() && sample_f32.is_sign_negative());
        assert_eq!((rng_f32.raw_draws, rng_f32.uniform_draws), (1, 1));

        let mut rng_f64 = ScriptedRng::new(vec![u64::MAX], vec![0.0_f64]);
        let sample_f64 = sample_f64(&mut rng_f64).unwrap();
        assert!(sample_f64.is_finite() && sample_f64.is_sign_negative());
        assert_eq!((rng_f64.raw_draws, rng_f64.uniform_draws), (1, 1));
    }

    #[test]
    fn malformed_density_uniform_is_rejected_at_the_provider_boundary() {
        let mut rng = ScriptedRng::new(vec![u64::MAX], vec![f64::NAN]);
        let error = sample_f64(&mut rng).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ));
        assert_eq!((rng.raw_draws, rng.uniform_draws), (1, 1));
    }

    #[test]
    fn density_rejection_restarts_with_a_fresh_raw_word() {
        let reject_f32 = word_f32(1, false, F32_MAGNITUDE_MASK);
        let mut rng_f32 = ScriptedRng::new(vec![reject_f32, 0], vec![0.999_999_94_f32]);
        assert_eq!(sample_f32(&mut rng_f32).unwrap(), 0.0);
        assert_eq!((rng_f32.raw_draws, rng_f32.uniform_draws), (2, 1));

        let reject_f64 = word_f64(1, false, F64_MAGNITUDE_MASK);
        let mut rng_f64 = ScriptedRng::new(
            vec![reject_f64, 0],
            vec![f64::from_bits(1.0_f64.to_bits() - 1)],
        );
        assert_eq!(sample_f64(&mut rng_f64).unwrap(), 0.0);
        assert_eq!((rng_f64.raw_draws, rng_f64.uniform_draws), (2, 1));
    }

    #[test]
    fn slow_density_path_accepts_from_the_half_open_endpoint() {
        let layer = 255;
        let mut rng_f32 = ScriptedRng::new(
            vec![word_f32(layer, false, K_F32[layer as usize])],
            vec![0.0_f32],
        );
        assert!(sample_f32(&mut rng_f32).unwrap().is_finite());

        let mut rng_f64 = ScriptedRng::new(
            vec![word_f64(layer.into(), false, K_F64[layer as usize])],
            vec![0.0_f64],
        );
        assert!(sample_f64(&mut rng_f64).unwrap().is_finite());
    }

    #[test]
    fn tail_rejection_and_acceptance_share_the_candidate_budget() {
        let tail_f32 = word_f32(0, true, K_F32[0]);
        let mut rng_f32 = ScriptedRng::new(vec![tail_f32, tail_f32], vec![0.01_f32, 0.9, 0.9, 0.1]);
        assert!(sample_f32(&mut rng_f32).unwrap() < -R_F32);
        assert_eq!((rng_f32.raw_draws, rng_f32.uniform_draws), (2, 4));

        let tail_f64 = word_f64(0, true, K_F64[0]);
        let mut rng_f64 = ScriptedRng::new(vec![tail_f64, tail_f64], vec![0.01_f64, 0.9, 0.9, 0.1]);
        assert!(sample_f64(&mut rng_f64).unwrap() < -R_F64);
        assert_eq!((rng_f64.raw_draws, rng_f64.uniform_draws), (2, 4));
    }

    #[test]
    fn zero_tail_uniform_is_rejected_without_unbounded_redraw() {
        let mut rng_f32 =
            ScriptedRng::new(vec![word_f32(0, false, K_F32[0]), 0], vec![0.0_f32, 0.5]);
        assert_eq!(sample_f32(&mut rng_f32).unwrap(), 0.0);
        assert_eq!((rng_f32.raw_draws, rng_f32.uniform_draws), (2, 2));

        let mut rng_f64 =
            ScriptedRng::new(vec![word_f64(0, false, K_F64[0]), 0], vec![0.0_f64, 0.5]);
        assert_eq!(sample_f64(&mut rng_f64).unwrap(), 0.0);
        assert_eq!((rng_f64.raw_draws, rng_f64.uniform_draws), (2, 2));
    }

    #[test]
    fn repeated_density_rejection_reports_structured_exhaustion() {
        let words_f32 = vec![word_f32(1, false, F32_MAGNITUDE_MASK); MAX_SAMPLING_ATTEMPTS];
        let uniforms_f32 = vec![0.999_999_94_f32; MAX_SAMPLING_ATTEMPTS];
        let mut rng_f32 = ScriptedRng::new(words_f32, uniforms_f32);
        let error = sample_f32(&mut rng_f32).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(
            (rng_f32.raw_draws, rng_f32.uniform_draws),
            (MAX_SAMPLING_ATTEMPTS, MAX_SAMPLING_ATTEMPTS)
        );

        let words_f64 = vec![word_f64(1, false, F64_MAGNITUDE_MASK); MAX_SAMPLING_ATTEMPTS];
        let uniforms_f64 = vec![f64::from_bits(1.0_f64.to_bits() - 1); MAX_SAMPLING_ATTEMPTS];
        let mut rng_f64 = ScriptedRng::new(words_f64, uniforms_f64);
        let error = sample_f64(&mut rng_f64).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(
            (rng_f64.raw_draws, rng_f64.uniform_draws),
            (MAX_SAMPLING_ATTEMPTS, MAX_SAMPLING_ATTEMPTS)
        );
    }
}
