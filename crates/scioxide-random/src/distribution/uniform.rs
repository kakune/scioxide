//! Continuous uniform distribution on `[low, high)`.

#[cfg(test)]
use alloc::vec::Vec;

#[cfg(test)]
use scioxide_core::{InvalidInputReason, NonFiniteOrigin};
use scioxide_core::{NumericalIssue, RealScalar, SciOxideError};

use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};
use crate::util::validated_unit_float;

/// Continuous uniform distribution on `[low, high)`.
///
/// ```text
/// f(x) = 1 / (b − a),   a ≤ x < b,
/// ```
///
/// with `a = low` and `b = high`. Samples and quantiles are computed as
/// `(1 − u)a + ub`, which does not overflow even when `b − a` does, and are
/// kept below `b`. So every sample lies in `[a, b)`, a sample is `a` when `a`
/// and `b` are adjacent floats, and a quantile at `p < 1` is below `b`. The
/// quantile at `p = 1` is `b`. The density and CDF rescale a width `b − a` that
/// overflows. The mean is the correctly rounded midpoint, including subnormal
/// endpoints. A variance outside the range of `S` is a `NumericalInstability` error.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Density, Distribution, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Uniform, Variance};
/// let law = Uniform::new(2.0_f64, 6.0)?;
/// assert_eq!((law.pdf(3.0)?, law.cdf(3.0)?), (0.25, 0.25));
/// assert!((law.quantile(0.25)? - 3.0).abs() < 1e-15);
/// assert!((law.mean()? - 4.0).abs() < 1e-15 && (law.variance()? - 4.0 / 3.0).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(9);
/// let x = law.sample(&mut rng)?;
/// assert!((2.0..6.0).contains(&x));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Uniform<S> {
    low: S,
    high: S,
}

impl<S: RealScalar> Uniform<S> {
    /// Creates a uniform distribution on `[low, high)`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `low` or `high` is NaN or infinite.
    /// - `DomainError` if `low ≥ high`.
    pub fn new(low: S, high: S) -> Result<Self, SciOxideError> {
        crate::law::argument(low, "low")?;
        crate::law::argument(high, "high")?;
        if low >= high {
            return Err(crate::law::domain(
                scioxide_core::argument!("high"),
                "lower bound must precede upper bound",
            ));
        }
        Ok(Self { low, high })
    }
}

#[inline]
fn convex_interpolation<S: RealScalar>(low: S, high: S, weight: S) -> S {
    if weight == S::ZERO {
        return low;
    }
    if weight == S::ONE {
        return high;
    }

    // w * high + (1 - w) * low instead of low + w * (high - low): the width
    // `high - low` may overflow for finite endpoints of opposite sign, while
    // the convex form stays finite and hits both endpoints exactly. `mul_add`
    // removes one rounding step.
    let complement = S::ONE - weight;
    weight.mul_add(high, complement * low)
}

#[cold]
fn overflow(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::Overflow, detail)
}

#[cold]
fn support_breakdown() -> SciOxideError {
    SciOxideError::numerical_instability(
        NumericalIssue::Breakdown,
        "endpoint interpolation produced a value outside the required support",
    )
}

#[inline]
fn half_open_interpolation<S: RealScalar>(low: S, high: S, weight: S) -> Result<S, SciOxideError> {
    // Rounding can land a weight just below 1 exactly on `high`; the half-open
    // contract [low, high) is restored by stepping to the predecessor. When
    // the interval is a single ulp wide, `max(low)` keeps the result in range.
    let candidate = convex_interpolation(low, high, weight);
    let sample = if candidate >= high {
        high.next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY)
            .max(low)
    } else {
        candidate
    };

    if !sample.is_finite() {
        return Err(overflow(
            "finite endpoint interpolation produced a non-finite result",
        ));
    }
    if sample < low || sample >= high {
        return Err(support_breakdown());
    }
    Ok(sample)
}

#[inline]
fn scaled_width<S: RealScalar>(low: S, high: S) -> (S, S) {
    // Overflow-safe width: divide both endpoints by the larger magnitude
    // first, so the normalized width is at most 2 and the true width is
    // `scale * normalized_width` without ever being formed.
    let scale = low.abs().max(high.abs());
    debug_assert!(scale > S::ZERO);
    let normalized_width = high / scale - low / scale;
    (scale, normalized_width)
}

#[inline]
fn reciprocal_width<S: RealScalar>(low: S, high: S) -> S {
    let width = high - low;
    if width.is_finite() {
        S::ONE / width
    } else {
        let (scale, normalized_width) = scaled_width(low, high);
        (S::ONE / scale) / normalized_width
    }
}

#[inline]
fn normalized_position<S: RealScalar>(low: S, high: S, value: S) -> S {
    let width = high - low;
    if width.is_finite() {
        (value - low) / width
    } else {
        let (scale, normalized_width) = scaled_width(low, high);
        (value / scale - low / scale) / normalized_width
    }
}

impl<S: RealScalar> Distribution<S> for Uniform<S> {
    #[inline]
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let unit = validated_unit_float(rng, scioxide_core::argument!("rng"))?;
        half_open_interpolation(self.low, self.high, unit)
    }
}

impl<S: RealScalar> Density<S> for Uniform<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x >= self.low && x < self.high {
            let density = reciprocal_width(self.low, self.high);
            if !density.is_finite() {
                return Err(overflow(
                    "the uniform density exceeds the finite scalar range",
                ));
            }
            if density <= S::ZERO {
                return Err(SciOxideError::numerical_instability(
                    NumericalIssue::Underflow,
                    "the uniform density was lost during finite-width scaling",
                ));
            }
            Ok(density)
        } else {
            Ok(S::ZERO)
        }
    }
}

impl<S: RealScalar> Cdf<S> for Uniform<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x < self.low {
            Ok(S::ZERO)
        } else if x >= self.high {
            Ok(S::ONE)
        } else {
            let position = normalized_position(self.low, self.high, x);
            if !position.is_finite() {
                return Err(overflow(
                    "finite endpoint normalization produced a non-finite result",
                ));
            }
            Ok(position.clamp(S::ZERO, S::ONE))
        }
    }
}

impl<S: RealScalar> Quantile<S> for Uniform<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ONE {
            return Ok(self.high);
        }
        half_open_interpolation(self.low, self.high, p)
    }
}

impl<S: RealScalar> Mean<S> for Uniform<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        scioxide_core::checked_midpoint(self.low, self.high)
    }
}

impl<S: RealScalar> Variance<S> for Uniform<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let width = self.high - self.low;
        if !width.is_finite() {
            return Err(overflow(
                "the endpoint width exceeds the finite scalar range",
            ));
        }

        let twelve = S::try_from_f64_exact(12.0)?;
        let scaled_width = width / twelve.sqrt();
        let variance = scaled_width * scaled_width;
        if !variance.is_finite() {
            return Err(overflow(
                "the uniform variance exceeds the finite scalar range",
            ));
        }
        if variance == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Underflow,
                "positive uniform variance rounded to zero",
            ));
        }
        Ok(variance)
    }
}

impl<S: RealScalar> crate::Support<S> for Uniform<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(self.low), Some(self.high))
    }
}

fn log_width<S: RealScalar>(a: S, b: S) -> Result<S, SciOxideError> {
    let width = b - a;
    if width.is_finite() {
        Ok(width.ln())
    } else {
        let h = S::HALF;
        Ok((b * h - a * h).ln() - h.ln())
    }
}
impl<S: RealScalar> Uniform<S> {
    fn log_tail(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= self.low {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if x >= self.high {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        // Each tail is either its own width ratio or the complement of the
        // other side. The smaller ratio is always formed directly and the
        // larger one through `log1mexp`, so neither tail loses precision.
        let left = log_width(self.low, x)?;
        let right = log_width(x, self.high)?;
        let total = log_width(self.low, self.high)?;
        if upper {
            Ok(if right < left {
                right - total
            } else {
                crate::law::log1mexp(left - total)?
            })
        } else {
            Ok(if left < right {
                left - total
            } else {
                crate::law::log1mexp(right - total)?
            })
        }
    }
}
impl<S: RealScalar> crate::Survival<S> for Uniform<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_tail(x, true)?.exp())
    }
}
impl<S: RealScalar> crate::LogCdf<S> for Uniform<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_tail(x, false)
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for Uniform<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_tail(x, true)
    }
}
impl<S: RealScalar> crate::LogDensity<S> for Uniform<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        Ok(if x < self.low || x >= self.high {
            S::NEG_INFINITY
        } else {
            -log_width(self.low, self.high)?
        })
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for Uniform<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        Ok(convex_interpolation(self.high, self.low, p))
    }
}
impl<S: RealScalar> crate::Entropy<S> for Uniform<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        log_width(self.low, self.high)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::{BitGenerator, SeedableRng};
    use crate::workspace::SamplingWorkspace;

    struct ScriptedRng<S> {
        values: std::vec::IntoIter<S>,
        fallback: S,
        draws: usize,
    }

    impl<S> ScriptedRng<S> {
        fn new(values: Vec<S>, fallback: S) -> Self {
            Self {
                values: values.into_iter(),
                fallback,
                draws: 0,
            }
        }
    }

    impl<S: Copy> BitGenerator for ScriptedRng<S> {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl<S: RealScalar> Rng<S> for ScriptedRng<S> {
        fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
            self.draws += 1;
            Ok(self.values.next().unwrap_or(self.fallback))
        }
    }

    fn assert_represented_contract<S: RealScalar>(cases: &[(S, S)]) {
        let weights = [
            S::ZERO,
            S::try_from_f64_exact(0.25).unwrap(),
            S::HALF,
            S::try_from_f64_exact(0.75).unwrap(),
            S::ONE.next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY),
        ];

        cases.iter().copied().for_each(|(low, high)| {
            let dist = Uniform::new(low, high).unwrap();
            weights.into_iter().for_each(|weight| {
                let mut rng = ScriptedRng::new(vec![weight], weight);
                let sample = dist.sample(&mut rng).unwrap();
                assert!(sample.is_finite());
                assert!(sample >= low && sample < high);

                let quantile = dist.quantile(weight).unwrap();
                assert!(quantile.is_finite());
                assert!(quantile >= low && quantile < high);
            });

            assert_eq!(dist.cdf(low).unwrap(), S::ZERO);
            assert_eq!(dist.cdf(high).unwrap(), S::ONE);
            assert_eq!(dist.quantile(S::ONE).unwrap(), high);

            let mean = dist.mean().unwrap();
            assert!(mean.is_finite());
            assert!(mean >= low && mean <= high);

            assert!(match dist.pdf(low) {
                Ok(density) => density.is_finite() && density > S::ZERO,
                Err(ref error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability {
                            issue: NumericalIssue::Overflow | NumericalIssue::Underflow,
                            ..
                        }
                    ) =>
                {
                    true
                }
                Err(_) => false,
            });
            assert!(match dist.variance() {
                Ok(variance) => variance.is_finite() && variance >= S::ZERO,
                Err(ref error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability {
                            issue: NumericalIssue::Overflow | NumericalIssue::Underflow,
                            ..
                        }
                    ) =>
                {
                    true
                }
                Err(_) => false,
            });
        });
    }

    #[test]
    fn test_new_valid() {
        assert!(Uniform::<f64>::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn test_new_invalid_low_ge_high() {
        let error = Uniform::<f64>::new(1.0, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_equal_bounds() {
        let error = Uniform::<f64>::new(1.0, 1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_signed_zero_equal_bounds_are_rejected() {
        let error = Uniform::<f64>::new(-0.0, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        let error = Uniform::<f32>::new(-0.0, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_bounds() {
        let error = Uniform::<f64>::new(f64::NEG_INFINITY, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = Uniform::<f64>::new(0.0, f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_in_range() {
        let dist = Uniform::new(2.0_f64, 5.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!((2.0..5.0).contains(&x));
        }
    }

    #[test]
    fn test_malformed_provider_never_becomes_a_uniform_sample() {
        let dist = Uniform::new(-2.0_f64, 3.0).unwrap();

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut rng = ScriptedRng::new(vec![value], value);
            let error = dist.sample(&mut rng).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Callback,
                    ..
                }
            ));
        }

        for value in [-f64::MIN_POSITIVE, 1.0, 1.0 + f64::EPSILON] {
            let mut rng = ScriptedRng::new(vec![value], value);
            let error = dist.sample(&mut rng).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::OutOfRange,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_sample_n_into_preserves_output_and_rng_advancement() {
        let dist = Uniform::new(2.0_f64, 6.0).unwrap();
        let mut rng = ScriptedRng::new(vec![0.25, f64::NAN, 0.75], 0.75);
        let mut output = [-99.0; 3];
        let mut workspace = SamplingWorkspace::new();

        let error = dist
            .sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ));
        assert_eq!(output, [-99.0; 3]);
        assert_eq!(rng.draws, 2);
    }

    #[test]
    fn test_f32_represented_support_across_endpoint_classes() {
        let subnormal = f32::from_bits(1);
        assert_represented_contract(&[
            (-0.0, subnormal),
            (-subnormal, -0.0),
            (subnormal, f32::from_bits(4)),
            (1.0, 1.0_f32.next_up()),
            (-1.0, (-1.0_f32).next_up()),
            (f32::MAX / 2.0, f32::MAX),
            (-f32::MAX, -f32::MAX / 2.0),
            (-f32::MAX, f32::MAX),
        ]);
    }

    #[test]
    fn test_f64_represented_support_across_endpoint_classes() {
        let subnormal = f64::from_bits(1);
        assert_represented_contract(&[
            (-0.0, subnormal),
            (-subnormal, -0.0),
            (subnormal, f64::from_bits(4)),
            (1.0, 1.0_f64.next_up()),
            (-1.0, (-1.0_f64).next_up()),
            (f64::MAX / 2.0, f64::MAX),
            (-f64::MAX, -f64::MAX / 2.0),
            (-f64::MAX, f64::MAX),
        ]);
    }

    #[test]
    fn test_adjacent_endpoints_always_sample_low() {
        let low_f32 = 1.0_f32;
        let high_f32 = low_f32.next_up();
        let dist_f32 = Uniform::new(low_f32, high_f32).unwrap();
        let mut rng_f32 = ScriptedRng::new(vec![1.0_f32.next_down()], 0.5);
        assert_eq!(dist_f32.sample(&mut rng_f32).unwrap(), low_f32);
        assert_eq!(dist_f32.quantile(1.0_f32.next_down()).unwrap(), low_f32);

        let low_f64 = 1.0_f64;
        let high_f64 = low_f64.next_up();
        let dist_f64 = Uniform::new(low_f64, high_f64).unwrap();
        let mut rng_f64 = ScriptedRng::new(vec![1.0_f64.next_down()], 0.5);
        assert_eq!(dist_f64.sample(&mut rng_f64).unwrap(), low_f64);
        assert_eq!(dist_f64.quantile(1.0_f64.next_down()).unwrap(), low_f64);
    }

    #[test]
    fn test_zero_endpoint_preserves_its_sign_at_zero_weight() {
        let high_f32 = f32::from_bits(1);
        let dist_f32 = Uniform::new(-0.0_f32, high_f32).unwrap();
        let mut rng_f32 = ScriptedRng::new(vec![-0.0], 0.0);
        assert_eq!(
            dist_f32.sample(&mut rng_f32).unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            dist_f32.quantile(-0.0).unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );

        let high_f64 = f64::from_bits(1);
        let dist_f64 = Uniform::new(-0.0_f64, high_f64).unwrap();
        let mut rng_f64 = ScriptedRng::new(vec![-0.0], 0.0);
        assert_eq!(
            dist_f64.sample(&mut rng_f64).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            dist_f64.quantile(-0.0).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn test_pdf_inside() {
        let dist = Uniform::new(0.0_f64, 4.0).unwrap();
        let p = dist.pdf(2.0).unwrap();
        assert!((p - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_pdf_outside() {
        let dist = Uniform::new(0.0_f64, 1.0).unwrap();
        assert!((dist.pdf(-0.1).unwrap()).abs() < 1e-12);
        assert!((dist.pdf(1.1).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_pdf_and_cdf_reject_non_finite_queries() {
        let dist = Uniform::new(0.0_f64, 1.0).unwrap();
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .for_each(|value| {
                let error = dist.pdf(value).unwrap_err();
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
                let error = dist.cdf(value).unwrap_err();
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
            });
    }

    #[test]
    fn test_overflowing_width_statistics_are_meaningful_for_f32() {
        let dist = Uniform::new(-f32::MAX, f32::MAX).unwrap();
        let density = dist.pdf(0.0).unwrap();
        assert!(density.is_finite() && density > 0.0);
        assert!((dist.cdf(0.0).unwrap() - 0.5).abs() <= f32::EPSILON);
        assert_eq!(dist.mean().unwrap(), 0.0);
        let error = dist.variance().unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn test_overflowing_width_statistics_are_meaningful_for_f64() {
        let dist = Uniform::new(-f64::MAX, f64::MAX).unwrap();
        let density = dist.pdf(0.0).unwrap();
        assert!(density.is_finite() && density > 0.0);
        assert!((dist.cdf(0.0).unwrap() - 0.5).abs() <= f64::EPSILON);
        assert_eq!(dist.mean().unwrap(), 0.0);
        let error = dist.variance().unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn test_subnormal_width_density_reports_typed_overflow() {
        let f32_dist = Uniform::new(0.0_f32, f32::from_bits(1)).unwrap();
        let error = f32_dist.pdf(0.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        let error = f32_dist.variance().unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            }
        ));

        let f64_dist = Uniform::new(0.0_f64, f64::from_bits(1)).unwrap();
        let error = f64_dist.pdf(0.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        let error = f64_dist.variance().unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            }
        ));
    }

    #[test]
    fn test_cdf_values() {
        let dist = Uniform::new(0.0_f64, 2.0).unwrap();
        assert!((dist.cdf(-1.0).unwrap()).abs() < 1e-12);
        assert!((dist.cdf(1.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((dist.cdf(3.0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_values() {
        let dist = Uniform::new(2.0_f64, 6.0).unwrap();
        assert!((dist.quantile(0.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((dist.quantile(0.5).unwrap() - 4.0).abs() < 1e-12);
        assert!((dist.quantile(1.0).unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_domain_error() {
        let dist = Uniform::new(0.0_f64, 1.0).unwrap();
        let error = dist.quantile(-0.1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = Uniform::new(0.0_f64, 1.0).unwrap();
        let error = dist.quantile(f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_mean() {
        let dist = Uniform::new(2.0_f64, 8.0).unwrap();
        assert!((dist.mean().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn subnormal_midpoints_round_once() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            Uniform::new(tiny, 2.0 * tiny).unwrap().mean().unwrap(),
            2.0 * tiny
        );
        assert_eq!(
            Uniform::new(-2.0 * tiny, -tiny).unwrap().mean().unwrap(),
            -2.0 * tiny
        );
        let tiny = f32::from_bits(1);
        assert_eq!(
            Uniform::new(tiny, 2.0 * tiny).unwrap().mean().unwrap(),
            2.0 * tiny
        );
        assert_eq!(
            Uniform::new(-2.0 * tiny, -tiny).unwrap().mean().unwrap(),
            -2.0 * tiny
        );
    }

    #[test]
    fn test_variance() {
        let dist = Uniform::new(0.0_f64, 12.0).unwrap();
        assert!((dist.variance().unwrap() - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_f32() {
        let dist = Uniform::new(0.0_f32, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let x: f32 = dist.sample(&mut rng).unwrap();
        assert!((0.0..1.0).contains(&x));
    }
}
