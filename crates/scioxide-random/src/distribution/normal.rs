//! Normal (Gaussian) distribution.
//!
//! [`Distribution::sample`] and [`Distribution::sample_n_into`] use the Ziggurat
//! method with separate tables for `f32` and `f64`. The CDF and survival
//! function use the scaled complementary error function `erfcx`.

#[cfg(test)]
use alloc::vec::Vec;
#[cfg(test)]
use scioxide_core::ConvergenceReason;
use scioxide_core::backend::{ScalarKind, checked_final_rounding};
use scioxide_core::checked_float_product;
#[cfg(test)]
use scioxide_core::checked_sum_of_products;
use scioxide_core::{RealScalar, ScalarConversionReason, SciOxideError};

use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};

pub(super) mod related;
pub use related::{
    Alpha, DoubleParetoLogNormal, ExponentiallyModifiedNormal, FatigueLife, GeneralizedNormal,
    PowerNormal,
};

mod ziggurat;
mod ziggurat_tables;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SamplingKernel {
    F32Ziggurat,
    F64Ziggurat,
}

#[inline(always)]
pub(super) fn sampling_kernel<S: RealScalar>() -> Result<SamplingKernel, SciOxideError> {
    match S::kind() {
        ScalarKind::F32 => Ok(SamplingKernel::F32Ziggurat),
        ScalarKind::F64 => Ok(SamplingKernel::F64Ziggurat),
        ScalarKind::Complex32 | ScalarKind::Complex64 => Err(dispatch_error::<S, S>()),
    }
}

#[inline(always)]
fn dispatch_error<T: scioxide_core::backend::Scalar, S: scioxide_core::backend::Scalar>()
-> SciOxideError {
    SciOxideError::scalar_conversion(
        <T as scioxide_core::backend::Scalar>::kind().into(),
        <S as scioxide_core::backend::Scalar>::kind().into(),
        ScalarConversionReason::Unsupported,
    )
}

pub(crate) fn standard_normal_quantile<S: RealScalar>(probability: S) -> Result<S, SciOxideError> {
    crate::law::argument(probability, "p")?;
    if probability <= S::ZERO || probability >= S::ONE {
        return Err(crate::law::domain(
            scioxide_core::argument!("probability"),
            "standard normal interior quantile requires probability in (0, 1)",
        ));
    }
    scioxide_special::normal_quantile(probability)
}

#[inline]
fn standardized<S: RealScalar>(x: S, mean: S, scale: S) -> S {
    // `x - mean` overflows when both are huge with opposite signs; dividing
    // first keeps the z-score finite in that case, at the cost of one extra
    // rounding that only matters on this rare path.
    let difference = x - mean;
    if difference.is_finite() {
        difference / scale
    } else {
        x / scale - mean / scale
    }
}

/// Normal (Gaussian) distribution with mean `μ` and standard deviation `σ`.
///
/// ```text
/// f(x) = exp(−(x − μ)² / (2σ²)) / (σ √(2π)),   x ∈ ℝ, σ > 0.
/// ```
///
/// Sampling uses the Ziggurat method with 256 layers and separate tables for
/// `f32` and `f64`. A sample is `μ + σz` for a standard normal draw `z`,
/// computed with one fused multiply-add and therefore rounded once. An exact
/// zero is `+0`. A result outside the range of `S` is a `NumericalInstability`
/// error, and so is an inverse survival value outside that range.
///
/// The tail on the side of `z = (x − μ)/σ` away from the mean is
/// `Q(|z|) = ½ erfcx(|z|/√2) exp(−z²/2)`, with `erfcx` the scaled complementary
/// error function and `z²/2` split into two `f64` words before the exponential.
/// The CDF and survival function return this tail directly or `1 − Q(|z|)` on
/// the other side. The quantiles at `p = 0` and `p = 1` are `−∞` and
/// `+∞`. Every other quantile is finite.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Density, Distribution, Mean, Normal, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Variance};
/// let law = Normal::new(1.0_f64, 2.0)?;
/// assert_eq!((law.mean()?, law.variance()?), (1.0, 4.0));
/// assert_eq!(law.cdf(1.0)?, 0.5);
/// // f(μ) = 1 / (σ √(2π)) with σ = 2.
/// assert!((law.pdf(1.0)? - 0.199_471_140_200_716_34).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(1);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. Marsaglia and W. W. Tsang, "The Ziggurat method for generating random
/// variables", *Journal of Statistical Software* 5(8), 2000.
#[derive(Debug, Clone)]
#[must_use]
pub struct Normal<S> {
    mean: S,
    std_dev: S,
    /// `1 / std_dev` when it is exact (a power of two), so standardizing
    /// multiplies with the same result as dividing.
    inverse: Option<S>,
}

impl<S: RealScalar> Normal<S> {
    /// Creates a normal distribution with mean `mean` and standard deviation
    /// `std_dev`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `mean` or `std_dev` is NaN or infinite.
    /// - `DomainError` if `std_dev ≤ 0`.
    pub fn new(mean: S, std_dev: S) -> Result<Self, SciOxideError> {
        crate::law::argument(mean, "mean")?;
        crate::law::positive(std_dev, scioxide_core::argument!("std_dev"))?;
        let bits = std_dev.into_f64().to_bits();
        let inverse = S::ONE / std_dev;
        let exact = bits & ((1_u64 << 52) - 1) == 0 && bits >> 52 != 0 && inverse.is_normal();
        Ok(Self {
            mean,
            std_dev,
            inverse: exact.then_some(inverse),
        })
    }

    /// Returns the standard normal law, with mean 0 and standard deviation 1.
    ///
    /// It equals `Normal::new(0, 1)` and cannot fail.
    ///
    /// ```
    /// use scioxide_random::{Distribution, Normal, Quantile, SeedableRng, SplitMix64};
    /// let law = Normal::<f64>::standard();
    /// // Φ⁻¹(0.975) = 1.959963984540054
    /// assert!((law.quantile(0.975)? - 1.959_963_984_540_054).abs() < 1e-12);
    /// assert!(law.sample(&mut SplitMix64::from_seed(42))?.is_finite());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn standard() -> Self {
        Self {
            mean: S::ZERO,
            std_dev: S::ONE,
            inverse: Some(S::ONE),
        }
    }

    /// The standardized coordinate `(x - mean) / std_dev`.
    #[inline(always)]
    fn z(&self, x: S) -> S {
        match self.inverse {
            Some(inverse) if (x - self.mean).is_finite() => (x - self.mean) * inverse,
            _ => standardized(x, self.mean, self.std_dev),
        }
    }

    #[inline(always)]
    fn affine_finite_standard(&self, standard: S) -> Result<S, SciOxideError> {
        debug_assert!(
            standard.is_finite(),
            "typed Ziggurat leaves return finite values"
        );
        // The identity transform is exact without the fused step.
        let value = if self.mean == S::ZERO && self.std_dev == S::ONE {
            standard
        } else {
            // One rounding of the exact `σ z + μ`: the correctly rounded affine
            // value, identical to an exact sum of products.
            self.std_dev.mul_add(standard, self.mean)
        };
        // Exact cancellation yields the canonical positive zero.
        checked_final_rounding(
            if value == S::ZERO { S::ZERO } else { value },
            "normal affine value exceeds finite scalar range",
        )
    }
}

/// One standard normal draw from the width-specific Ziggurat kernel. The
/// value is finite; a zero may carry either sign.
#[inline(always)]
pub(crate) fn standard_normal<S: RealScalar, R: Rng<S>>(rng: &mut R) -> Result<S, SciOxideError> {
    match sampling_kernel::<S>()? {
        SamplingKernel::F32Ziggurat => S::from_value(scioxide_core::backend::ScalarValue::F32(
            ziggurat::sample_f32(rng)?,
        )),
        SamplingKernel::F64Ziggurat => S::from_value(scioxide_core::backend::ScalarValue::F64(
            ziggurat::sample_f64(rng)?,
        )),
    }
}

impl<S: RealScalar> Distribution<S> for Normal<S> {
    #[inline(always)]
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        // The kernel is chosen by scalar width so f32 draws use the 32-bit
        // tables (one `next_uniform_u32` per fast path) rather than widening f64
        // samples; both kernels only return finite values.
        self.affine_finite_standard(standard_normal(rng)?)
    }
}

impl<S: RealScalar> Density<S> for Normal<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let two = S::TWO;
        let sqrt_2pi = S::try_from_f64_lossy(core::f64::consts::TAU.sqrt())?;
        let z = self.z(x);
        if !z.is_finite() {
            return Ok(S::ZERO);
        }
        let exponent = -(z * z) / two;
        // Divide in two steps instead of `std_dev * sqrt(2 pi)` so a tiny
        // standard deviation does not underflow the product before inversion.
        let normalizer = (S::ONE / self.std_dev) / sqrt_2pi;
        let numerator = exponent.exp();
        if !normalizer.is_finite() || !numerator.is_normal() {
            return checked_final_rounding(
                crate::LogDensity::log_pdf(self, x)?.exp(),
                "normal density exceeds the finite scalar range",
            );
        }
        checked_final_rounding(
            numerator * normalizer,
            "normal density exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Cdf<S> for Normal<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let standardized = self.z(x);
        if !standardized.is_finite() {
            return Ok(if x < self.mean { S::ZERO } else { S::ONE });
        }
        let probability = scioxide_special::normal_cdf(standardized)?;
        if probability.is_finite() && probability >= S::ZERO && probability <= S::ONE {
            Ok(probability)
        } else {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "normal CDF left the represented probability interval",
            ))
        }
    }
}

impl<S: RealScalar> Quantile<S> for Normal<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        let z = standard_normal_quantile(p)?;
        self.affine_finite_standard(z)
    }
}

impl<S: RealScalar> Mean<S> for Normal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.mean)
    }
}

impl<S: RealScalar> Variance<S> for Normal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let variance = checked_float_product(self.std_dev, self.std_dev)?;
        if variance == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "positive normal variance rounded to zero",
            ));
        }
        Ok(variance)
    }
}

impl<S: RealScalar> crate::Support<S> for Normal<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(None, None)
    }
}

impl<S: RealScalar> crate::Survival<S> for Normal<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let z = self.z(x);
        if !z.is_finite() {
            return Ok(if x < self.mean { S::ONE } else { S::ZERO });
        }
        scioxide_special::normal_sf(z)
    }
}
impl<S: RealScalar> crate::LogCdf<S> for Normal<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let z = self.z(x);
        if !z.is_finite() {
            if x < self.mean {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "normal log tail exceeds the finite scalar range",
                ));
            }
            return Ok(S::ZERO);
        }
        scioxide_special::normal_log_cdf(z)
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for Normal<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let z = self.z(x);
        if !z.is_finite() {
            if x >= self.mean {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "normal log tail exceeds the finite scalar range",
                ));
            }
            return Ok(S::ZERO);
        }
        scioxide_special::normal_log_sf(z)
    }
}
impl<S: RealScalar> crate::LogDensity<S> for Normal<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let z = self.z(x);
        Ok(-z * (z / S::TWO)
            - self.std_dev.ln()
            - S::try_from_f64_lossy(0.5 * crate::distribution::special::log_two_pi::<f64>())?)
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for Normal<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        // `normal_isf` evaluates the upper tail directly, so small p keeps
        // full relative accuracy instead of going through `quantile(1 - p)`.
        self.affine_finite_standard(scioxide_special::normal_isf(p)?)
    }
}
impl<S: RealScalar> crate::Entropy<S> for Normal<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        crate::law::finite(
            self.std_dev.ln()
                + S::try_from_f64_lossy(
                    0.5 * (1.0 + crate::distribution::special::log_two_pi::<f64>()),
                )?,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::{BitGenerator, SeedableRng};
    use crate::workspace::SamplingWorkspace;

    #[test]
    fn standard_parameters_preserve_exact_bits_including_signed_zero() {
        macro_rules! check {
            ($scalar:ty) => {{
                let normal = Normal::new(0.0 as $scalar, 1.0).unwrap();
                for value in [
                    -<$scalar>::MAX,
                    -<$scalar>::MIN_POSITIVE,
                    -<$scalar>::from_bits(1),
                    -0.0,
                    0.0,
                    <$scalar>::from_bits(1),
                    <$scalar>::MIN_POSITIVE,
                    0.7,
                    <$scalar>::MAX,
                ] {
                    let expected = checked_sum_of_products(&[1.0, 1.0], &[0.0, value]).unwrap();
                    assert_eq!(
                        normal.affine_finite_standard(value).unwrap().to_bits(),
                        expected.to_bits()
                    );
                }
            }};
        }
        check!(f32);
        check!(f64);
    }

    #[test]
    fn general_affine_is_the_correctly_rounded_exact_sum_of_products() {
        macro_rules! check {
            ($scalar:ty) => {{
                let scales: [$scalar; 6] =
                    [3.0, 0.1, 1.0 / 3.0, 2.0, <$scalar>::MIN_POSITIVE, 1e30];
                let means: [$scalar; 7] = [2.0, -0.0, 0.0, -1.0 / 7.0, 1e-30, -5e3, 0.3];
                let values: [$scalar; 9] = [
                    -0.0,
                    0.0,
                    0.7,
                    -1.234_567,
                    1.0 / 3.0,
                    <$scalar>::from_bits(1),
                    -2.5,
                    6.0,
                    -0.3,
                ];
                for &scale in &scales {
                    for &mean in &means {
                        let normal = Normal::new(mean, scale).unwrap();
                        for &value in &values {
                            let actual = normal.affine_finite_standard(value);
                            let expected = checked_sum_of_products(&[1.0, scale], &[mean, value])
                                .map(|x| if x == 0.0 { 0.0 } else { x });
                            match (actual, expected) {
                                (Ok(actual), Ok(expected)) => {
                                    assert_eq!(
                                        actual.to_bits(),
                                        expected.to_bits(),
                                        "{mean} {scale} {value}"
                                    )
                                }
                                (Err(actual), Err(expected)) => {
                                    assert_eq!(actual.kind(), expected.kind())
                                }
                                (actual, expected) => panic!("{actual:?} vs {expected:?}"),
                            }
                        }
                    }
                }
                // Exact cancellation, including of signed zeros, is canonical +0.
                let normal = Normal::new(-0.0 as $scalar, 2.0).unwrap();
                assert_eq!(normal.affine_finite_standard(-0.0).unwrap().to_bits(), 0);
                let normal = Normal::new(-1.0 as $scalar, 0.5).unwrap();
                assert_eq!(normal.affine_finite_standard(2.0).unwrap().to_bits(), 0);
            }};
        }
        check!(f32);
        check!(f64);
    }

    struct FirstThenRejectRng<S> {
        raw_draws: usize,
        rejection_word: u64,
        rejection_uniform: S,
    }

    impl<S> BitGenerator for FirstThenRejectRng<S> {
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
            Ok(if self.raw_draws == 1 {
                0
            } else {
                self.rejection_word as u32
            })
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.raw_draws += 1;
            Ok(if self.raw_draws == 1 {
                0
            } else {
                self.rejection_word
            })
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl<S: RealScalar> Rng<S> for FirstThenRejectRng<S> {
        fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
            Ok(self.rejection_uniform)
        }
    }

    #[test]
    fn test_new_valid() {
        assert!(Normal::<f64>::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn test_new_zero_std_dev() {
        let error = Normal::<f64>::new(0.0, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_negative_std_dev() {
        let error = Normal::<f64>::new(0.0, -1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_params() {
        let error = Normal::<f64>::new(f64::NAN, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = Normal::<f64>::new(0.0, f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_mean_approx() {
        let dist = Normal::new(3.0_f64, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample(&mut rng).unwrap()).sum();
        let sample_mean = sum / f64::from(n);
        assert!((sample_mean - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_sample_n_into_matches_continuous_scalar_sampling() {
        let dist = Normal::new(1.0_f64, 2.0).unwrap();
        [1, 2, 5, 8].into_iter().for_each(|length| {
            let mut expected_rng = SplitMix64::from_seed(41);
            let expected = (0..length)
                .map(|_| dist.sample(&mut expected_rng).unwrap())
                .collect::<Vec<_>>();

            let mut actual_rng = SplitMix64::from_seed(41);
            let mut output = vec![f64::NAN; length];
            let mut workspace = SamplingWorkspace::new();
            dist.sample_n_into(
                &mut actual_rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap();

            assert_eq!(output, expected);
            assert_eq!(actual_rng.position(), expected_rng.position());
        });
    }

    #[test]
    fn test_empty_sample_n_into_does_not_advance_rng() {
        fn assert_empty<S: RealScalar>() {
            let dist = Normal::new(S::ZERO, S::ONE).unwrap();
            let mut rng = SplitMix64::from_seed(53);
            let position = rng.position();
            let mut workspace = SamplingWorkspace::new();

            dist.sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut [],
                &mut workspace,
            )
            .unwrap();

            assert_eq!(rng.position(), position);
        }

        assert_empty::<f32>();
        assert_empty::<f64>();
    }

    #[test]
    fn test_sample_n_matches_sample_n_into_for_same_call_boundary() {
        fn assert_lengths<S: RealScalar>() {
            let dist = Normal::new(S::ZERO, S::ONE).unwrap();
            [0, 1, 2, 7, 8].into_iter().for_each(|length| {
                let mut allocating_rng = SplitMix64::from_seed(67);
                let allocating = dist.sample_n(&mut allocating_rng, length).unwrap();

                let mut buffered_rng = SplitMix64::from_seed(67);
                let mut buffered = vec![S::NAN; length];
                let mut workspace = SamplingWorkspace::new();
                dist.sample_n_into(
                    &mut buffered_rng,
                    crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                    &mut buffered,
                    &mut workspace,
                )
                .unwrap();

                assert_eq!(allocating, buffered);
                assert_eq!(allocating_rng.position(), buffered_rng.position());
            });
        }

        assert_lengths::<f32>();
        assert_lengths::<f64>();
    }

    #[test]
    fn test_same_method_sequences_are_deterministic_for_both_precisions() {
        fn assert_deterministic<S: RealScalar>() {
            let dist = Normal::new(S::ZERO, S::ONE).unwrap();
            let mut first_rng = SplitMix64::from_seed(71);
            let mut second_rng = SplitMix64::from_seed(71);
            let first = dist.sample_n(&mut first_rng, 128).unwrap();
            let second = dist.sample_n(&mut second_rng, 128).unwrap();

            assert_eq!(first, second);
            assert_eq!(first_rng.position(), second_rng.position());
        }

        assert_deterministic::<f32>();
        assert_deterministic::<f64>();
    }

    #[test]
    fn test_sample_n_into_preserves_caller_output_on_exhaustion() {
        fn assert_prefix<S: RealScalar>(rejection_word: u64, rejection_uniform: S) {
            let dist = Normal::new(S::ZERO, S::ONE).unwrap();
            let mut rng = FirstThenRejectRng {
                raw_draws: 0,
                rejection_word,
                rejection_uniform,
            };
            let mut output = [S::NAN, S::NAN];
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
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::IterationLimit,
                    ..
                }
            ));
            assert!(output.iter().all(|value| value.is_nan()));
        }

        assert_prefix::<f32>(u64::from(((1_u32 << 23) - 1) << 9 | 1), 0.999_999_94);
        assert_prefix::<f64>(
            (((1_u64 << 52) - 1) << 9) | 1,
            f64::from_bits(1.0_f64.to_bits() - 1),
        );
    }

    #[test]
    fn test_unrepresentable_affine_output_is_rejected_before_writing() {
        fn assert_overflow<S: RealScalar>() {
            let standard = Normal::new(S::ZERO, S::ONE).unwrap();
            let seed = (0_u64..10_000)
                .find(|seed| {
                    let mut rng = SplitMix64::from_seed(*seed);
                    standard.sample(&mut rng).unwrap() > S::ONE
                })
                .unwrap();
            let dist = Normal::new(S::MAX, S::MAX).unwrap();
            let mut rng = SplitMix64::from_seed(seed);
            let mut output = [S::ZERO];
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
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ));
            assert_eq!(output, [S::ZERO]);
        }

        assert_overflow::<f32>();
        assert_overflow::<f64>();
    }

    #[test]
    fn test_sample_n_reports_storage_failures_before_sampling() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(79);
        let position = rng.position();

        for (n, kind) in [
            (usize::MAX, scioxide_core::SciOxideErrorKind::SizeOverflow),
            (
                isize::MAX as usize / core::mem::size_of::<f64>() + 1,
                scioxide_core::SciOxideErrorKind::AllocationFailure,
            ),
        ] {
            assert_eq!(dist.sample_n(&mut rng, n).unwrap_err().kind(), kind);
            assert_eq!(rng.position(), position);
        }
    }

    #[test]
    fn test_normal_is_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<Normal<f32>>();
        assert_send_sync::<Normal<f64>>();
    }

    #[test]
    fn test_sampling_kernel_selects_the_sealed_primitive() {
        assert_eq!(
            sampling_kernel::<f32>().unwrap(),
            SamplingKernel::F32Ziggurat
        );
        assert_eq!(
            sampling_kernel::<f64>().unwrap(),
            SamplingKernel::F64Ziggurat
        );
    }

    #[test]
    fn test_pdf_standard_at_zero() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((dist.pdf(0.0).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_standard_at_zero() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        assert!((dist.cdf(0.0).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_standard_central_and_tail_oracles_f64() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        let cases = [
            (-10.0, 7.619_853_024_160_527e-24),
            (-8.0, 6.220_960_574_271_784e-16),
            (-5.0, 2.866_515_718_791_933e-7),
            (-1.0, 0.158_655_253_931_457_07),
            (0.5, 0.691_462_461_274_013_1),
            (1.0, 0.841_344_746_068_542_9),
            (5.0, 0.999_999_713_348_428_1),
            (8.0, 0.999_999_999_999_999_3),
        ];

        cases.into_iter().for_each(|(x, expected)| {
            let actual = dist.cdf(x).unwrap();
            let error = (actual - expected).abs();
            let tolerance = if expected < 1e-12 {
                2e-14 * expected
            } else {
                (2e-14 * expected).max(2e-16)
            };
            assert!(error <= tolerance, "cdf({x})={actual}, expected {expected}");
        });
        assert!(dist.cdf(-10.0).unwrap() > 0.0);
    }

    #[test]
    fn test_cdf_standard_lower_tail_f32_does_not_cancel() {
        let dist = Normal::new(0.0_f32, 1.0).unwrap();
        let expected = 6.220_960_4e-16_f32;
        let actual = dist.cdf(-8.0).unwrap();
        assert!(actual > 0.0);
        assert!(((actual - expected) / expected).abs() < 2e-6);
    }

    #[test]
    fn test_cdf_symmetry_monotonicity_and_finite_limits() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        let values = [-8.0, -5.0, -1.0, 0.0, 1.0, 5.0, 8.0].map(|x| dist.cdf(x).unwrap());

        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        [0.25_f64, 1.0, 3.0, 5.0].into_iter().for_each(|x| {
            let complement = dist.cdf(-x).unwrap() + dist.cdf(x).unwrap();
            assert!((complement - 1.0).abs() <= 2e-16);
        });
        assert_eq!(dist.cdf(-f64::MAX).unwrap(), 0.0);
        assert_eq!(dist.cdf(f64::MAX).unwrap(), 1.0);
    }

    #[test]
    fn test_cdf_rejects_non_finite_inputs() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .for_each(|x| {
                let error = dist.cdf(x).unwrap_err();
                assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
            });
    }

    #[test]
    fn test_cdf_quantile_roundtrip_across_regions() {
        let dist = Normal::new(3.0_f64, 2.0).unwrap();
        [1e-12_f64, 1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6]
            .into_iter()
            .for_each(|probability| {
                let x = dist.quantile(probability).unwrap();
                let roundtrip = dist.cdf(x).unwrap();
                let tolerance = 4e-14_f64.max(probability * 4e-14);
                assert!(
                    (roundtrip - probability).abs() <= tolerance,
                    "cdf(quantile({probability}))={roundtrip}"
                );
            });
    }

    #[test]
    fn test_quantile_median() {
        let dist = Normal::new(5.0_f64, 2.0).unwrap();
        assert!((dist.quantile(0.5).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_endpoints_are_domain_infinities() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn minimum_positive_interior_quantiles_remain_finite() {
        let f64_dist = Normal::new(0.0_f64, 1.0).unwrap();
        assert!(f64_dist.quantile(f64::from_bits(1)).unwrap().is_finite());
        assert!(f64_dist.quantile(1.0_f64.next_down()).unwrap().is_finite());

        let f32_dist = Normal::new(0.0_f32, 1.0).unwrap();
        assert!(f32_dist.quantile(f32::from_bits(1)).unwrap().is_finite());
        assert!(f32_dist.quantile(1.0_f32.next_down()).unwrap().is_finite());
    }

    #[test]
    fn extreme_scale_density_preserves_representable_subnormal() {
        let density = Normal::new(0.0_f64, f64::MAX).unwrap().pdf(0.0).unwrap();
        assert!(density.is_subnormal());
        assert!(density > 0.0);
    }

    #[test]
    fn test_quantile_domain_error() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
        let error = dist.quantile(-0.1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = Normal::new(0.0_f64, 1.0).unwrap();
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
    fn test_mean_and_variance() {
        let dist = Normal::new(2.0_f64, 3.0).unwrap();
        assert!((dist.mean().unwrap() - 2.0).abs() < 1e-12);
        assert!((dist.variance().unwrap() - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_f32() {
        let dist = Normal::new(0.0_f32, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let x: f32 = dist.sample(&mut rng).unwrap();
        assert!(x.is_finite());
    }

    #[test]
    fn interior_normal_quantile_half() {
        // probit(0.5) = 0
        let n: f64 = standard_normal_quantile(0.5).unwrap();
        assert!(n.abs() < 1e-10, "probit(0.5) should be 0, got {n}");
    }

    #[test]
    fn interior_normal_quantile_symmetry() {
        let a: f64 = standard_normal_quantile(0.25).unwrap();
        let b: f64 = standard_normal_quantile(0.75).unwrap();
        assert!((a + b).abs() < 1e-10, "probit should be antisymmetric");
    }

    #[test]
    fn interior_normal_quantile_rejects_endpoints_and_non_finite_values() {
        for value in [0.0_f64, 1.0, -0.5, 1.5] {
            let error = standard_normal_quantile(value).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        }
        let error = standard_normal_quantile(f64::NAN).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }
}
