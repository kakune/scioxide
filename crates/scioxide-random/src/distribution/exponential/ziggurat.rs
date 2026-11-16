//! Typed, stateless exponential Ziggurat sampling leaves.
//!
//! The `f32` and `f64` tables in `ziggurat_tables.rs` were derived independently
//! from the 256-layer equal-area construction for `f(x) = exp(-x)` at 120
//! decimal digits. With cutoff `R` and layer area `V = R f(R) + ∫_R^∞ f =
//! (R + 1) exp(-R)`, the boundaries follow `x[i] = -ln(V / x[i+1] + f(x[i+1]))`
//! from `x[255] = R`, and `R` is the root for which the top layer closes at
//! `f(0) = 1`. Entry 0 holds the base-strip width `V / f(R)`. `F[i] = f(x[i])`
//! with `F[0] = f(0)`, and `K[i]` is `x[i-1] / x[i]` (`R / x[0]` for the base
//! strip, zero for the top cap) scaled to mantissa units and rounded down.
//! Values are rounded to the target type only after that construction, and are
//! stored as exact IEEE bit patterns.
//!
//! One full-width uniform word supplies independent fields. The low eight bits
//! select the layer and the next 23 (`f32`) or 52 (`f64`) bits give the
//! rectangle magnitude. By memorylessness the tail beyond `R` is `R - ln U`,
//! drawn from one open-interval uniform within the common attempt budget.

#![allow(clippy::cast_precision_loss)]

use scioxide_core::{RealScalar, SciOxideError};

use super::ziggurat_tables::{F_F32, F_F64, K_F32, K_F64, R_F32, R_F64, X_F32, X_F64};
use crate::traits::Rng;
use crate::util::{MAX_SAMPLING_ATTEMPTS, validated_unit_float};

const LAYER_MASK: u32 = 0xff;
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

// A zero uniform would make `ln` infinite; `None` restarts with a fresh word.
#[cold]
fn tail_f32<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<Option<f32>, SciOxideError> {
    let uniform = typed_uniform_f32(rng)?;
    Ok((uniform > 0.0).then(|| R_F32 - uniform.ln()))
}

#[cold]
fn tail_f64<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<Option<f64>, SciOxideError> {
    let uniform = typed_uniform_f64(rng)?;
    Ok((uniform > 0.0).then(|| R_F64 - uniform.ln()))
}

// The first word's inner-rectangle test is the hot path; the cold remainder
// re-runs that first word through the complete three-stage loop, keeping the
// wedge and tail out of every caller's register footprint.
#[inline(always)]
pub(super) fn sample_f32<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<f32, SciOxideError> {
    let word = rng.next_uniform_u32()?;
    let layer = (word & LAYER_MASK) as usize;
    let magnitude = (word >> 8) & F32_MAGNITUDE_MASK;
    if magnitude < K_F32[layer] {
        return Ok((f32::from_bits(F32_ONE_BITS | magnitude) - 1.0) * X_F32[layer]);
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
        let magnitude = (word >> 8) & F32_MAGNITUDE_MASK;
        // OR-ing the mantissa into the bit pattern of 1.0 gives a float in
        // [1, 2); subtracting 1 yields a uniform on [0, 1) with a full 23-bit
        // grid and no integer-to-float rounding.
        let unit = f32::from_bits(F32_ONE_BITS | magnitude) - 1.0;
        let candidate = unit * X_F32[layer];

        // The integer compare accepts the inner rectangle, which lies entirely
        // under the density, without evaluating `exp`.
        if magnitude < K_F32[layer] {
            return Ok(candidate);
        }
        // The base strip's remainder, by area, is the unbounded tail.
        if layer == 0 {
            if let Some(tail) = tail_f32(rng)? {
                return Ok(tail);
            }
            continue;
        }
        // Wedge test between the layer's lower bound `f(x[layer])` and upper
        // bound `f(x[layer - 1])`, accepted under the exact density.
        let uniform = typed_uniform_f32(rng)?;
        let vertical = F_F32[layer] + uniform * (F_F32[layer - 1] - F_F32[layer]);
        if vertical < (-candidate).exp() {
            return Ok(candidate);
        }
    }
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[inline(always)]
pub(super) fn sample_f64<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<f64, SciOxideError> {
    let word = rng.next_uniform_u64()?;
    #[allow(clippy::cast_possible_truncation)]
    let layer = (word as u32 & LAYER_MASK) as usize;
    let magnitude = (word >> 8) & F64_MAGNITUDE_MASK;
    if magnitude < K_F64[layer] {
        return Ok((f64::from_bits(F64_ONE_BITS | magnitude) - 1.0) * X_F64[layer]);
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
        // Same bit-field split and three-stage test as `rejection_f32` with a
        // 52-bit mantissa for full double-precision resolution.
        #[allow(clippy::cast_possible_truncation)]
        let layer = (word as u32 & LAYER_MASK) as usize;
        let magnitude = (word >> 8) & F64_MAGNITUDE_MASK;
        let unit = f64::from_bits(F64_ONE_BITS | magnitude) - 1.0;
        let candidate = unit * X_F64[layer];

        if magnitude < K_F64[layer] {
            return Ok(candidate);
        }
        if layer == 0 {
            if let Some(tail) = tail_f64(rng)? {
                return Ok(tail);
            }
            continue;
        }
        let uniform = typed_uniform_f64(rng)?;
        let vertical = F_F64[layer] + uniform * (F_F64[layer - 1] - F_F64[layer]);
        if vertical < (-candidate).exp() {
            return Ok(candidate);
        }
    }
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::traits::BitGenerator;

    struct ScriptedRng<S> {
        words: alloc::vec::IntoIter<u64>,
        uniforms: alloc::vec::IntoIter<S>,
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
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u32()
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            self.raw_draws += 1;
            Ok(self.words.next().unwrap() as u32)
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.raw_draws += 1;
            Ok(self.words.next().unwrap())
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl<S: RealScalar> Rng<S> for ScriptedRng<S> {
        fn next_float(&mut self) -> Result<S, SciOxideError> {
            self.uniform_draws += 1;
            Ok(self.uniforms.next().unwrap())
        }
    }

    const fn word_f32(layer: u32, magnitude: u32) -> u64 {
        (((magnitude & F32_MAGNITUDE_MASK) << 8) | (layer & LAYER_MASK)) as u64
    }

    const fn word_f64(layer: u64, magnitude: u64) -> u64 {
        ((magnitude & F64_MAGNITUDE_MASK) << 8) | (layer & LAYER_MASK as u64)
    }

    #[test]
    fn tables_satisfy_the_equal_area_construction() {
        let area = (R_F64 + 1.0) * (-R_F64).exp();
        let close = |actual: f64, expected: f64, tolerance: f64| {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs(),
                "{actual} vs {expected}"
            );
        };
        assert_eq!(X_F64[255], R_F64);
        close(X_F64[0] * F_F64[255], area, 1e-14);
        close(X_F64[1] * (1.0 - F_F64[1]), area, 1e-12);
        assert_eq!(F_F64[0], 1.0);
        for layer in 1..256 {
            close(F_F64[layer], (-X_F64[layer]).exp(), 4.0 * f64::EPSILON);
            close(
                f64::from(X_F32[layer]),
                X_F64[layer],
                f64::from(f32::EPSILON),
            );
            close(
                f64::from(F_F32[layer]),
                F_F64[layer],
                f64::from(f32::EPSILON),
            );
            if layer >= 2 {
                assert!(X_F64[layer - 1] < X_F64[layer]);
                close(
                    X_F64[layer] * (F_F64[layer - 1] - F_F64[layer]),
                    area,
                    1e-11,
                );
                let ratio = X_F64[layer - 1] / X_F64[layer];
                close(K_F64[layer] as f64, ratio * 2_f64.powi(52), 1e-15);
                close(f64::from(K_F32[layer]), ratio * 2_f64.powi(23), 1e-6);
            }
        }
        assert_eq!((K_F64[1], K_F32[1]), (0, 0));
        close(K_F64[0] as f64, R_F64 / X_F64[0] * 2_f64.powi(52), 1e-15);
    }

    #[test]
    fn zero_words_fast_accept_without_uniforms() {
        let mut rng_f32 = ScriptedRng::<f32>::new(alloc::vec![0], alloc::vec![]);
        assert_eq!(sample_f32(&mut rng_f32).unwrap().to_bits(), 0);
        assert_eq!((rng_f32.raw_draws, rng_f32.uniform_draws), (1, 0));
        let mut rng_f64 = ScriptedRng::<f64>::new(alloc::vec![0], alloc::vec![]);
        assert_eq!(sample_f64(&mut rng_f64).unwrap().to_bits(), 0);
        assert_eq!((rng_f64.raw_draws, rng_f64.uniform_draws), (1, 0));
    }

    #[test]
    fn base_strip_overflow_takes_the_memoryless_tail() {
        let mut rng = ScriptedRng::new(
            alloc::vec![word_f64(0, F64_MAGNITUDE_MASK)],
            alloc::vec![0.25],
        );
        assert_eq!(sample_f64(&mut rng).unwrap(), R_F64 - 0.25_f64.ln());
        assert_eq!((rng.raw_draws, rng.uniform_draws), (1, 1));
        let mut rng = ScriptedRng::new(
            alloc::vec![word_f32(0, F32_MAGNITUDE_MASK)],
            alloc::vec![0.25_f32],
        );
        assert_eq!(sample_f32(&mut rng).unwrap(), R_F32 - 0.25_f32.ln());
    }

    #[test]
    fn zero_tail_uniform_restarts_with_a_fresh_word() {
        let mut rng = ScriptedRng::new(
            alloc::vec![word_f64(0, F64_MAGNITUDE_MASK), 0],
            alloc::vec![0.0],
        );
        assert_eq!(sample_f64(&mut rng).unwrap(), 0.0);
        assert_eq!((rng.raw_draws, rng.uniform_draws), (2, 1));
    }

    #[test]
    fn wedge_accepts_under_and_rejects_over_the_density() {
        // The top cap has no inner rectangle, so even its midpoint takes the
        // wedge test; the lowest height lies under the density.
        let mut accept = ScriptedRng::new(alloc::vec![word_f64(1, 1 << 51)], alloc::vec![0.0]);
        assert_eq!(sample_f64(&mut accept).unwrap(), X_F64[1] / 2.0);
        assert_eq!((accept.raw_draws, accept.uniform_draws), (1, 1));

        let mut reject = ScriptedRng::new(
            alloc::vec![word_f64(1, F64_MAGNITUDE_MASK), 0],
            alloc::vec![1.0 - f64::EPSILON / 2.0],
        );
        assert_eq!(sample_f64(&mut reject).unwrap(), 0.0);
        assert_eq!((reject.raw_draws, reject.uniform_draws), (2, 1));
    }

    #[test]
    fn invalid_wedge_uniform_is_rejected_at_the_provider_boundary() {
        let mut rng = ScriptedRng::new(
            alloc::vec![word_f64(1, F64_MAGNITUDE_MASK)],
            alloc::vec![1.0],
        );
        assert_eq!(
            sample_f64(&mut rng).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }

    #[test]
    fn repeated_rejection_reports_structured_exhaustion() {
        let attempts = MAX_SAMPLING_ATTEMPTS;
        let mut rng = ScriptedRng::new(
            alloc::vec![word_f64(1, F64_MAGNITUDE_MASK); attempts],
            alloc::vec![1.0 - f64::EPSILON / 2.0; attempts],
        );
        let error = sample_f64(&mut rng).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!((rng.raw_draws, rng.uniform_draws), (attempts, attempts));
    }
}
