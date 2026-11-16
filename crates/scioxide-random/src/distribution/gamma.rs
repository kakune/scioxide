//! Gamma and generalized gamma-related laws.

use crate::{InverseSurvival, LogCdf, LogDensity, LogSurvival};
use scioxide_core::backend::checked_final_rounding;
use scioxide_core::checked_float_product;
use scioxide_core::{RealScalar, SciOxideError};

use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};
use crate::util::MAX_SAMPLING_ATTEMPTS;

use crate::distribution::formula::Formula;
use crate::distribution::symmetric;

use crate::distribution::formula::formula_law;

use crate::law::constant as number;

use crate::law::positive;

use crate::distribution::formula::positive_support;
use crate::law::{argument, finite, log1mexp};

/// ln Gamma(1+a) without losing a when 1+a rounds to one.
pub(super) fn log_gamma_one_plus<S: RealScalar>(a: S) -> Result<S, SciOxideError> {
    if a < S::SQRT_EPSILON {
        // Taylor remainder is O(a^3); the quadratic coefficient is zeta(2)/2.
        return Ok(
            a * (number::<S>(-0.577_215_664_901_532_9)? + a * S::PI * S::PI / number::<S>(12.0)?)
        );
    }
    scioxide_special::ln_gamma(a + S::ONE)
}

/// Gamma log tails from ln(x), including x below the representable range.
pub(super) fn gamma_log_coordinate<S: RealScalar>(
    a: S,
    log_x: S,
    upper: bool,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let x = log_x.exp();
    if x == S::INFINITY {
        return if upper {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "gamma log tail exceeds the finite scalar range",
            ))
        } else {
            Ok(S::ZERO)
        };
    }
    if log_x < S::EPSILON.ln() {
        // Lower gamma series: the omitted relative term is <= x/(a+1).
        let log_p = a * log_x - log_gamma_one_plus(a)?;
        return if upper {
            log1mexp(log_p)
        } else {
            finite(log_p)
        };
    }
    if upper {
        crate::law::special(left, |budget| {
            scioxide_special::gamma_log_q_with_options(
                a,
                x,
                scioxide_special::GammaOptions::new(budget),
            )
        })
    } else {
        crate::law::special(left, |budget| {
            scioxide_special::gamma_log_p_with_options(
                a,
                x,
                scioxide_special::GammaOptions::new(budget),
            )
        })
    }
}

/// Prepared standard-gamma sampler: Marsaglia–Tsang constants for the shape,
/// boosted by one below unity (Ahrens–Dieter / Stuart).
///
/// Construction validates a positive finite shape and evaluates only a square
/// root and a reciprocal, so laws prepare it once and draws whose shape varies
/// build it without special functions. Each attempt draws one standard normal
/// and one open uniform; the boost adds one open uniform per accepted draw.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GammaKernel<S> {
    shape: S,
    /// Marsaglia–Tsang `d = α' - 1/3` for the boosted shape `α' >= 1`.
    d: S,
    /// Marsaglia–Tsang `c = 1 / sqrt(9 d)`.
    c: S,
    /// Squeeze coefficient `0.0331` in the sampling precision.
    squeeze: S,
}

impl<S: RealScalar> GammaKernel<S> {
    /// Prepare the sampler for a positive finite `shape`.
    pub(crate) fn new(shape: S) -> Result<Self, SciOxideError> {
        crate::law::positive(shape, scioxide_core::argument!("shape"))?;
        let boosted = if shape < S::ONE {
            shape + S::ONE
        } else {
            shape
        };
        let one_third = S::try_from_f64_lossy(1.0 / 3.0)?;
        let d = boosted - one_third;
        Ok(Self {
            shape,
            d,
            c: one_third / d.sqrt(),
            squeeze: S::try_from_f64_lossy(0.0331)?,
        })
    }

    fn boosted(&self) -> bool {
        self.shape < S::ONE
    }

    /// One standard (unit-scale) gamma variate.
    #[inline]
    pub(crate) fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let x = self.base(rng)?;
        if self.boosted() {
            // Ahrens-Dieter boost: Gamma(α) = Gamma(α+1) * U^(1/α)
            // (Stuart's theorem). Marsaglia-Tsang needs α >= 1, so small shapes
            // sample the boosted law and scale by an independent power of U.
            let u: S = crate::util::validated_open_float(rng)?;
            checked_float_product(x, u.powf(S::ONE / self.shape))
        } else {
            Ok(x)
        }
    }

    /// Logarithm of one standard gamma variate, finite even when the variate
    /// itself would underflow.
    #[inline]
    pub(crate) fn sample_log<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        // Same boost as `sample`, carried out in log space: for tiny shapes
        // U^(1/alpha) underflows to zero long before its logarithm
        // ln(U) / alpha leaves the finite range. Dirichlet sampling only
        // needs the logarithm.
        let mut value = self.base(rng)?.ln();
        if self.boosted() {
            value += crate::util::validated_open_float(rng)?.ln() / self.shape;
        }
        crate::law::finite(value)
    }

    /// Logarithm of a power, scaling the boost before dividing by its shape.
    /// Widening also keeps `f32` logarithms below its ordinary value range.
    pub(crate) fn sample_log_power<R: Rng<S>>(
        &self,
        rng: &mut R,
        power: f64,
    ) -> Result<f64, SciOxideError> {
        let base = self.base(rng)?.into_f64().ln() * power;
        Ok(if self.boosted() {
            base + crate::util::validated_open_float(rng)?.into_f64().ln()
                * (power / self.shape.into_f64())
        } else {
            base
        })
    }

    /// Logarithm of an independent Gamma ratio. Scale boost terms before
    /// subtraction, so equal tiny-shape draws can cancel without inf - inf.
    pub(crate) fn sample_log_ratio<R: Rng<S>>(
        &self,
        other: &Self,
        rng: &mut R,
    ) -> Result<f64, SciOxideError> {
        let first = self.base(rng)?.into_f64().ln();
        let first_boost = if self.boosted() {
            crate::util::validated_open_float(rng)?.into_f64().ln()
        } else {
            0.0
        };
        let second = other.base(rng)?.into_f64().ln();
        let second_boost = if other.boosted() {
            crate::util::validated_open_float(rng)?.into_f64().ln()
        } else {
            0.0
        };
        let scale = self.shape.into_f64().min(other.shape.into_f64()).min(1.0);
        let boost = first_boost * (scale / self.shape.into_f64())
            - second_boost * (scale / other.shape.into_f64());
        // Infinite log odds denote a probability that rounds to an endpoint.
        Ok(first - second + boost / scale)
    }

    /// Marsaglia–Tsang method for Gamma(α' ≥ 1, 1).
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    #[inline]
    fn base<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        // Marsaglia & Tsang (2000): for alpha >= 1, with d = alpha - 1/3 and
        // c = 1 / sqrt(9 d), X = d (1 + c Z)^3 for standard normal Z has a
        // density very close to Gamma(alpha), so a simple rejection step on
        // (Z, U) corrects it exactly.
        let (d, c) = (self.d, self.c);
        for _ in 0..MAX_SAMPLING_ATTEMPTS {
            let x = crate::distribution::normal::standard_normal(rng)?;

            // The transform is only monotone for 1 + c Z > 0; outside that range
            // the target density is zero, so such Z are rejected outright.
            let v = S::ONE + c * x;
            if v <= S::ZERO {
                continue;
            }
            let v = v * v * v;
            let u: S = crate::util::validated_open_float(rng)?;
            let x2 = x * x;

            // Squeeze test: most iterations accept here. 1 - 0.0331 Z^4 is the
            // published cheap lower bound of the acceptance ratio, avoiding logs.
            if u < S::ONE - self.squeeze * x2 * x2 {
                return Ok(d * v);
            }
            // Full test: ln U < Z^2 / 2 + d (1 - V + ln V) is the exact log
            // acceptance ratio of the transformed density against the normal.
            if u.ln() < S::HALF * x2 + d * (S::ONE - v + v.ln()) {
                return Ok(d * v);
            }
        }

        Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
    }
}

/// Gamma distribution with shape `α` and scale `θ`.
///
/// ```text
/// f(x) = x^(α−1) e^(−x/θ) / (Γ(α) θ^α),   x > 0, α > 0, θ > 0.
/// ```
///
/// The mean is `αθ` and the variance `αθ²`. At `x = 0` the density is `+∞`,
/// `1/θ` or `0` for `α < 1`, `α = 1` or `α > 1`. The CDF and survival function
/// are the regularized incomplete gamma functions `P(α, x/θ)` and `Q(α, x/θ)`
/// (DLMF 8.2.4). The quantile at `p = 1` is `+∞`. If `ln Γ(α)` is not
/// representable, only the operations that need it fail.
///
/// Sampling uses the Marsaglia–Tsang method for `α ≥ 1`. Each candidate takes
/// one standard normal and one uniform draw. For `α < 1` a sample is
/// `G U^(1/α)` with `G ~ Gamma(α + 1, 1)` and `U` uniform on `(0, 1)`.
/// For `α < 1`, the boost and scale are combined in logarithms before rounding.
/// A positive sample rounding to zero or infinity returns numerical `Underflow`
/// or `Overflow`.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Gamma, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = Gamma::new(2.0_f64, 3.0)?;
/// assert_eq!((law.mean()?, law.variance()?), (6.0, 18.0));
/// // For α = 2: F(x) = 1 − (1 + x/θ) e^(−x/θ), so F(3) = 1 − 2/e.
/// assert!((law.cdf(3.0)? - (1.0 - 2.0 / core::f64::consts::E)).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(13);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. Marsaglia and W. W. Tsang, "A simple method for generating gamma
///   variables", *ACM Trans. Math. Softw.* 26(3), 2000, 363–372.
/// - NIST DLMF, §8.2, <https://dlmf.nist.gov/8.2>.
#[derive(Debug, Clone)]
#[must_use]
pub struct Gamma<S: RealScalar> {
    options: crate::LawOptions<S>,
    shape: S,
    scale: S,
    log_scale: S,
    log_gamma: Result<S, SciOxideError>,
    /// `ln Gamma(1 + shape)` for small log-coordinate tails, prepared once.
    log_gamma_one_plus: Result<S, SciOxideError>,
    kernel: GammaKernel<S>,
}

impl<S: RealScalar> Gamma<S> {
    /// Creates a gamma distribution with shape `shape` and scale `scale`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` or `scale` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0` or `scale ≤ 0`.
    pub fn new(shape: S, scale: S) -> Result<Self, SciOxideError> {
        Self::with_options(shape, scale, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(
        shape: S,
        scale: S,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let kernel = GammaKernel::new(shape)?;
        crate::law::positive(scale, scioxide_core::argument!("scale"))?;
        Ok(Self {
            options,
            shape,
            scale,
            log_scale: scale.ln(),
            log_gamma: scioxide_special::ln_gamma(shape),
            log_gamma_one_plus: log_gamma_one_plus(shape),
            kernel,
        })
    }

    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let z = x / self.scale;
        if z == S::INFINITY {
            return Ok(S::ZERO);
        }
        if z == S::ZERO {
            return gamma_log_coordinate(self.shape, x.ln() - self.log_scale, false, left);
        }
        crate::law::special(left, |budget| {
            scioxide_special::gamma_log_p_with_options(
                self.shape,
                z,
                scioxide_special::GammaOptions::new(budget),
            )
        })
    }

    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::ZERO);
        }
        let z = x / self.scale;
        if z == S::INFINITY {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "gamma log tail exceeds the finite scalar range",
            ));
        }
        if z == S::ZERO {
            return gamma_log_coordinate(self.shape, x.ln() - self.log_scale, true, left);
        }
        crate::law::special(left, |budget| {
            scioxide_special::gamma_log_q_with_options(
                self.shape,
                z,
                scioxide_special::GammaOptions::new(budget),
            )
        })
    }

    fn quantile_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        let x = crate::law::special(left, |budget| {
            scioxide_special::gamma_p_inv_with_options(
                self.shape,
                p,
                scioxide_special::GammaOptions::new(budget),
            )
        })?;
        checked_float_product(self.scale, x)
    }

    fn isf_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        // Invert the regularized upper incomplete gamma directly so upper-tail
        // quantiles do not lose precision to `1 - p`.
        crate::law::finite(
            self.scale
                * crate::law::special(left, |budget| {
                    scioxide_special::gamma_q_inv_with_options(
                        self.shape,
                        p,
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?,
        )
    }
}

impl<S: RealScalar> Distribution<S> for Gamma<S> {
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    #[inline]
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        if self.shape < S::ONE {
            crate::law::positive_draw(
                self.kernel.sample_log_power(rng, 1.0)? + self.scale.into_f64().ln(),
            )
        } else {
            crate::law::positive_result(checked_float_product(
                self.scale,
                self.kernel.sample(rng)?,
            )?)
        }
    }
}

impl<S: RealScalar> Density<S> for Gamma<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x < S::ZERO {
            return Ok(S::ZERO);
        }
        if x == S::ZERO {
            return if self.shape < S::ONE {
                Ok(S::INFINITY)
            } else if self.shape == S::ONE {
                checked_final_rounding(
                    S::ONE / self.scale,
                    "gamma endpoint density exceeds the finite scalar range",
                )
            } else {
                Ok(S::ZERO)
            };
        }
        // pdf = x^(α-1) * exp(-x/β) / (β^α * Γ(α))
        // In log space: (α-1)*ln(x) - x/β - α*ln(β) - ln_gamma(α)
        let log_pdf = self.log_pdf(x)?;
        checked_final_rounding(
            log_pdf.exp(),
            "gamma density exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Cdf<S> for Gamma<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(crate::LogCdf::log_cdf(self, x)?.exp())
    }
}

impl<S: RealScalar> Quantile<S> for Gamma<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        self.quantile_work(p, &mut self.options.work_budget().get())
    }

    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.quantile_work(p, left)
        })
    }
}

impl<S: RealScalar> Mean<S> for Gamma<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        let mean = checked_float_product(self.shape, self.scale)?;
        if mean == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "positive gamma mean rounded to zero",
            ));
        }
        Ok(mean)
    }
}

impl<S: RealScalar> Variance<S> for Gamma<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let scaled_shape = checked_float_product(self.shape, self.scale)?;
        let variance = checked_float_product(scaled_shape, self.scale)?;
        if variance == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "positive gamma variance rounded to zero",
            ));
        }
        Ok(variance)
    }
}

impl<S: RealScalar> crate::Support<S> for Gamma<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(S::ZERO), None)
    }
}

impl<S: RealScalar> crate::LogCdf<S> for Gamma<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_cdf_work(x, &mut self.options.work_budget().get())
    }

    fn log_cdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_cdf_work(x, left)
        })
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for Gamma<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_sf_work(x, &mut self.options.work_budget().get())
    }

    fn log_sf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_sf_work(x, left)
        })
    }
}
impl<S: RealScalar> crate::Survival<S> for Gamma<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x > S::ZERO && x / self.scale == S::INFINITY {
            return Ok(S::ZERO);
        }
        Ok(crate::LogSurvival::log_sf(self, x)?.exp())
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for Gamma<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        self.isf_work(p, &mut self.options.work_budget().get())
    }

    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.isf_work(p, left)
        })
    }
}
impl<S: RealScalar> crate::LogDensity<S> for Gamma<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(self.pdf(x)?.ln());
        }
        let z = x / self.scale;
        if z == S::INFINITY {
            return Ok(S::NEG_INFINITY);
        }
        let log_z = if z == S::ZERO {
            x.ln() - self.log_scale
        } else {
            z.ln()
        };
        crate::law::checked_log_density(
            (self.shape - S::ONE) * log_z - z - self.log_gamma.clone()? - self.log_scale,
        )
    }
}
impl<S: RealScalar> crate::Entropy<S> for Gamma<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        S::try_from_f64_lossy(crate::law::finite(
            super::moment_math::gamma_entropy(self.shape.into_f64())? + self.scale.into_f64().ln(),
        )?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::{BitGenerator, SeedableRng};

    #[test]
    fn pearson3_f32_draws_near_zero_skew_keep_full_resolution() {
        for skew in [2e-5_f32, 1e-4, 1e-3, -1.5e-3] {
            let law = Pearson3::new(skew).unwrap();
            let mut rng = SplitMix64::from_seed(8);
            let mut samples = law.sample_n(&mut rng, 2000).unwrap();
            let mean = samples.iter().map(|&x| f64::from(x)).sum::<f64>() / 2000.0;
            // Five standard errors of a unit-variance mean.
            assert!(mean.abs() < 5.0 / 2000.0_f64.sqrt(), "skew = {skew}");
            samples.sort_by(f32::total_cmp);
            samples.dedup();
            assert!(
                samples.len() >= 1990,
                "skew = {skew}: {} distinct",
                samples.len()
            );
        }
    }

    #[derive(Default)]
    struct CountingRng {
        raw_draws: usize,
        uniform_draws: usize,
    }

    impl BitGenerator for CountingRng {
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
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.raw_draws += 1;
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl Rng<f64> for CountingRng {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            self.uniform_draws += 1;
            Ok(0.5)
        }
    }

    struct RejectingRng {
        normal_draws: usize,
        uniform_draws: usize,
    }

    impl BitGenerator for RejectingRng {
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
            self.normal_draws += 1;
            // Layer 255 at half its rectangle width is accepted by the normal
            // kernel but rejected by Gamma(1) when its uniform is near one.
            Ok((1_u64 << 60) | 255)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl Rng<f64> for RejectingRng {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            self.uniform_draws += 1;
            Ok(1.0_f64.next_down())
        }
    }

    #[test]
    fn test_new_valid() {
        assert!(Gamma::<f64>::new(2.0, 1.0).is_ok());
    }

    #[test]
    fn test_new_invalid_shape() {
        let error = Gamma::<f64>::new(0.0, 1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_invalid_scale() {
        let error = Gamma::<f64>::new(1.0, -1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_params() {
        let error = Gamma::<f64>::new(f64::NAN, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = Gamma::<f64>::new(1.0, f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_positive() {
        let dist = Gamma::new(2.0_f64, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_marsaglia_tsang_uses_the_default_normal_primitive() {
        let dist = Gamma::new(2.0_f64, 1.0).unwrap();
        let mut rng = CountingRng::default();

        assert!(dist.sample(&mut rng).unwrap() > 0.0);
        assert_eq!(rng.raw_draws, 1);
        assert_eq!(rng.uniform_draws, 1);
    }

    #[test]
    #[allow(clippy::panic)]
    fn marsaglia_tsang_rejection_is_bounded_exactly() {
        let mut rng = RejectingRng {
            normal_draws: 0,
            uniform_draws: 0,
        };
        let error = GammaKernel::new(1.0_f64)
            .unwrap()
            .base(&mut rng)
            .unwrap_err();
        let scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit,
            diagnostics,
            ..
        } = error.view()
        else {
            panic!("unexpected error: {error:?}")
        };
        assert_eq!(
            diagnostics.counts().own().iterations(),
            MAX_SAMPLING_ATTEMPTS
        );
        assert_eq!(rng.normal_draws, MAX_SAMPLING_ATTEMPTS);
        assert_eq!(rng.uniform_draws, MAX_SAMPLING_ATTEMPTS);
    }

    #[test]
    fn test_sample_small_shape() {
        // α < 1 uses Ahrens-Dieter boost
        let dist = Gamma::new(0.5_f64, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x > 0.0);
        }
    }

    #[test]
    fn scale_recovers_a_boost_that_underflows_on_its_own() {
        let shape = 0.0009_f64;
        let law = Gamma::new(shape, 1e300).unwrap();
        let mut rng = CountingRng::default();
        let expected =
            ((shape + 1.0 - 1.0 / 3.0).ln() + 0.5_f64.ln() / shape + 1e300_f64.ln()).exp();
        assert!((law.sample(&mut rng).unwrap() / expected - 1.0).abs() < 1e-12);
        assert_eq!((rng.raw_draws, rng.uniform_draws), (1, 2));
    }

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn test_sample_mean_approx() {
        let alpha = 3.0_f64;
        let beta = 2.0_f64;
        let dist = Gamma::new(alpha, beta).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample(&mut rng).unwrap()).sum();
        let sample_mean = sum / f64::from(n);
        assert!(
            (sample_mean - alpha * beta).abs() < 0.5,
            "sample_mean={sample_mean}"
        );
    }

    #[test]
    fn test_pdf_exponential_case() {
        // Gamma(1, 1) = Exponential(1)
        let dist = Gamma::new(1.0_f64, 1.0).unwrap();
        let x = 1.0;
        let expected = (-1.0_f64).exp();
        assert!((dist.pdf(x).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn endpoint_limits_are_explicit() {
        assert_eq!(
            Gamma::new(0.5_f64, 2.0).unwrap().pdf(0.0).unwrap(),
            f64::INFINITY
        );
        assert_eq!(Gamma::new(1.0_f64, 2.0).unwrap().pdf(0.0).unwrap(), 0.5);
        assert_eq!(Gamma::new(2.0_f64, 2.0).unwrap().pdf(0.0).unwrap(), 0.0);

        let dist = Gamma::new(2.0_f64, 3.0).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), 0.0);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_cdf_exponential_case() {
        let dist = Gamma::new(1.0_f64, 1.0).unwrap();
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((dist.cdf(1.0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = Gamma::new(2.0_f64, 3.0).unwrap();
        let p = 0.75;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-8);
    }

    #[test]
    fn test_quantile_domain_error() {
        let dist = Gamma::new(1.0_f64, 1.0).unwrap();
        let error = dist.quantile(-0.1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = Gamma::new(1.0_f64, 1.0).unwrap();
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
        let dist = Gamma::new(3.0_f64, 2.0).unwrap();
        assert!((dist.mean().unwrap() - 6.0).abs() < 1e-12);
        assert!((dist.variance().unwrap() - 12.0).abs() < 1e-12);
    }
}

/// Double gamma distribution with shape `α`: `±G` with equal probability, where
/// `G ~ Gamma(α, 1)`.
///
/// ```text
/// f(x) = |x|^(α−1) e^(−|x|) / (2 Γ(α)),   x ∈ ℝ, α > 0.
/// ```
///
/// The mean is 0 and the variance `α(α + 1)`. Shape 1 is the standard Laplace
/// distribution. Shift and scale the law with [`Affine`](crate::Affine).
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, DoubleGamma, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Variance};
/// let law = DoubleGamma::new(1.0_f64)?;
/// assert_eq!((law.mean()?, law.variance()?), (0.0, 2.0));
/// assert!((law.cdf(-1.0)? - 0.5 / core::f64::consts::E).abs() < 1e-16);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(14);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct DoubleGamma<S: RealScalar> {
    options: crate::LawOptions<S>,
    shape: S,
    gamma: super::Gamma<S>,
}
impl<S: RealScalar> DoubleGamma<S> {
    /// Creates a double gamma distribution with shape `shape` and unit scale.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        Self::with_options(shape, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(shape: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self {
            options,
            shape,
            gamma: super::Gamma::with_options(shape, S::ONE, options)?,
        })
    }
}
impl<S: RealScalar> Formula<S> for DoubleGamma<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn center(&self) -> Option<S> {
        Some(S::ZERO)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.gamma.log_pdf(x.abs())? - S::LN_2)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        symmetric::log_tail(x, upper, |y| {
            crate::law::scoped(left, |context| self.gamma.log_sf_scoped(y, context))
        })
    }
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        symmetric::inverse(p, upper, |tail| {
            crate::law::scoped(left, |context| self.gamma.isf_scoped(tail, context))
        })
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        symmetric::draw(rng, |rng| self.gamma.sample(rng))
    }
}
formula_law!(DoubleGamma);
impl<S: RealScalar> Mean<S> for DoubleGamma<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for DoubleGamma<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(self.shape * (self.shape + S::ONE))
    }
}

/// Generalized gamma distribution (Stacy) with shape parameters `a` and `c`.
///
/// ```text
/// f(x) = |c| x^(ac−1) exp(−x^c) / Γ(a),   x > 0, a > 0, c ≠ 0.
/// ```
///
/// Equivalently `X^c ~ Gamma(a, 1)`. The law has unit scale. Scale it with
/// [`Affine`](crate::Affine). The moment `E[X^r]` exists when `a + r/c > 0`.
///
/// Sampling draws `ln G` for `G ~ Gamma(a, 1)` and returns `exp(ln G / c)`,
/// rounded once. A sample that underflows to zero or overflows is a
/// `NumericalInstability` error.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, GeneralizedGamma, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support};
/// let law = GeneralizedGamma::new(1.0_f64, 2.0)?;
/// assert!((law.cdf(1.0)? - (1.0 - (-1.0_f64).exp())).abs() < 1e-15);
/// assert!((law.mean()? - core::f64::consts::PI.sqrt() / 2.0).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(15);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedGamma<S: RealScalar> {
    options: crate::LawOptions<S>,
    a: S,
    c: S,
    gamma: GammaKernel<S>,
    /// `ln Gamma(a)` and `ln Gamma(1 + a)`, prepared once.
    log_gamma: Result<S, SciOxideError>,
    log_gamma_one_plus: Result<S, SciOxideError>,
}
impl<S: RealScalar> GeneralizedGamma<S> {
    /// Creates a generalized gamma distribution with shapes `a` and `c`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` or `c` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0` or `c = 0`.
    pub fn new(a: S, c: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, c, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(a: S, c: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(a, scioxide_core::argument!("a"))?;
        argument(c, "c")?;
        if c == S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("c"),
                "generalized gamma exponent must be nonzero",
            ));
        }
        Ok(Self {
            options,
            a,
            c,
            gamma: GammaKernel::new(a)?,
            log_gamma: scioxide_special::ln_gamma(a),
            log_gamma_one_plus: log_gamma_one_plus(a),
        })
    }
    fn coordinate_probability(&self, y: S, upper: bool) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let log_z = self.c * y;
        let gamma_upper = upper == (self.c > S::ZERO);
        if log_z.exp() == S::INFINITY {
            return Ok(if gamma_upper { S::ZERO } else { S::ONE });
        }
        if log_z < S::EPSILON.ln() {
            let log_p = self.a * log_z - self.log_gamma_one_plus.clone()?;
            return Ok(if gamma_upper {
                -log_p.exp_m1()
            } else {
                log_p.exp()
            });
        }
        Ok(gamma_log_coordinate(
            self.a,
            log_z,
            gamma_upper,
            &mut self.options.work_budget().get(),
        )?
        .exp())
    }
    fn raw_moment(&self, r: S) -> Result<S, SciOxideError> {
        let a = self.a.into_f64();
        let shift = r.into_f64() / self.c.into_f64();
        if a + shift <= 0.0 {
            return Err(crate::law::undefined_moment(
                "generalized gamma moment does not exist",
            ));
        }
        S::try_from_f64_lossy(finite(super::special::log_gamma_ratio(a, shift)?.exp())?)
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedGamma<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        positive_support()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            return Ok(if self.c < S::ZERO {
                S::NEG_INFINITY
            } else if self.a * self.c < S::ONE {
                S::INFINITY
            } else if self.a * self.c == S::ONE {
                self.c.abs().ln() - self.log_gamma.clone()?
            } else {
                S::NEG_INFINITY
            });
        }
        Ok(self.c.abs().ln() + (self.a * self.c - S::ONE) * x.ln()
            - x.powf(self.c)
            - self.log_gamma.clone()?)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        gamma_log_coordinate(self.a, self.c * x.ln(), upper == (self.c > S::ZERO), left)
    }
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if x > S::ZERO && (self.c * x.ln()).exp() == S::INFINITY {
            return Ok(if upper == (self.c > S::ZERO) {
                S::ZERO
            } else {
                S::ONE
            });
        }
        Ok(self.log_probability(x, upper)?.exp())
    }
    fn log_probability_with_work(
        &self,
        x: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        // Both directions are evaluated directly. Complementing here would
        // unnecessarily request an opposite logarithm outside the scalar range.
        crate::law::log_probability(self.tail_with_work(x, upper, left)?)
    }

    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if upper {
            if p == S::ZERO {
                return Ok(S::INFINITY);
            }
            if p == S::ONE {
                return Ok(S::ZERO);
            }
            let z = if self.c > S::ZERO {
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_q_inv_with_options(
                        self.a,
                        p,
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?
            } else {
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_p_inv_with_options(
                        self.a,
                        p,
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?
            };
            finite(z.powf(S::ONE / self.c))
        } else {
            if p == S::ZERO {
                return Ok(S::ZERO);
            }
            if p == S::ONE {
                return Ok(S::INFINITY);
            }
            let z = if self.c > S::ZERO {
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_p_inv_with_options(
                        self.a,
                        p,
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?
            } else {
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_q_inv_with_options(
                        self.a,
                        p,
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?
            };
            finite(z.powf(S::ONE / self.c))
        }
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        crate::law::positive_draw(self.gamma.sample_log_power(rng, 1.0 / self.c.into_f64())?)
    }
}
formula_law!(GeneralizedGamma);
impl<S: RealScalar> Mean<S> for GeneralizedGamma<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(S::ONE)
    }
}
impl<S: RealScalar> Variance<S> for GeneralizedGamma<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        super::moment_math::gamma_variance(self.a.into_f64(), self.c.into_f64().recip())
    }
}

/// Standardized Pearson type III distribution with skewness `γ`.
///
/// The law is `(G − β²)/β` with `β = 2/γ` and `G ~ Gamma(β², 1)`, and the
/// standard normal distribution at `γ = 0`. It has mean 0, variance 1 and
/// skewness `γ`. Shift and scale it with [`Affine`](crate::Affine).
///
/// For `|γ| ≥ 0.002`, probabilities and quantiles use the regularized
/// incomplete gamma function and draws use the gamma representation. For
/// `0 < |γ| < 0.002`, where `β²` is large and the centering cancels, they use a
/// second-order Edgeworth expansion, an endpoint expansion or quadrature of the
/// density in `f64`, and draws invert the CDF.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Mean, Pearson3, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = Pearson3::new(2.0_f64)?; // Exponential(1) - 1 on [-1, ∞)
/// assert_eq!((law.mean()?, law.variance()?), (0.0, 1.0));
/// assert!((law.cdf(0.0)? - (1.0 - (-1.0_f64).exp())).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(16);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Pearson3<S: RealScalar> {
    options: crate::LawOptions<S>,
    skew: S,
    normal: super::Normal<S>,
    gamma: Option<crate::Affine<super::Gamma<S>, S>>,
}
impl<S: RealScalar> Pearson3<S> {
    /// Creates a standardized Pearson type III distribution with skewness
    /// `skew`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `skew` is NaN or infinite.
    pub fn new(skew: S) -> Result<Self, SciOxideError> {
        Self::with_options(skew, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(skew: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(skew, "skew")?;
        // A gamma draw G ~ Gamma(4/skew^2) centered as (G - beta^2)/beta has
        // absolute resolution about 2 eps/|skew|. Below |skew| = 0.002, as in
        // the other kernels, draws invert the CDF evaluated in f64 instead.
        let gamma = if skew.into_f64().abs() < 0.002 {
            None
        } else {
            let beta = S::TWO / skew;
            let a = beta * beta;
            Some(crate::Affine::new(
                super::Gamma::with_options(a, S::ONE, options)?,
                -beta,
                S::ONE / beta,
            )?)
        };
        Ok(Self {
            options,
            skew,
            normal: super::Normal::standard(),
            gamma,
        })
    }
}
impl<S: RealScalar> Pearson3<S> {
    fn expansion(&self, x: f64) -> bool {
        // The expansion parameter includes tail conditioning (Mills ratio).
        self.skew.into_f64().abs() < 0.002
            && self.skew.into_f64().abs() * (1.0 + x.abs()).powi(3) <= 1e-3
    }
    fn expanded_tail(&self, x: f64, upper: bool) -> Result<f64, SciOxideError> {
        let skew = self.skew.into_f64();
        let x2 = x * x;
        let h2 = x2 - 1.0;
        let h3 = x * (x2 - 3.0);
        let h5 = x * (x2 * x2 - 10.0 * x2 + 15.0);
        let correction = skew * h2 / 6.0 + skew * skew * (h3 / 16.0 + h5 / 72.0);
        let base = if upper {
            scioxide_special::normal_log_sf(x)?
        } else {
            scioxide_special::normal_log_cdf(x)?
        };
        let log_phi = -0.5 * x * x - 0.5 * core::f64::consts::TAU.ln();
        let delta = (if upper { correction } else { -correction }) * (log_phi - base).exp();
        crate::law::log_probability(base + delta.ln_1p())
    }
    // Centered Gamma log density. Eliminating ln Gamma(a)'s leading Stirling
    // terms analytically avoids subtracting O(a ln a) values near a=4/skew^2.
    // At |skew|<0.002, the first omitted Stirling term is below 1e-45.
    fn centered_density(&self, x: f64) -> Result<f64, SciOxideError> {
        let skew = self.skew.into_f64();
        let delta = 0.5 * skew * x;
        if delta <= -1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        let deviance = if delta.abs() < 0.25 {
            // (delta-ln(1+delta))/delta^2, including the exact delta=0 limit.
            let ratio = (0..32)
                .rev()
                .fold(0.0, |value, k| 1.0 / (k as f64 + 2.0) - delta * value);
            x * (x * ratio)
        } else {
            // delta is separated from zero, so this subtraction is conditioned.
            let beta = 2.0 / skew;
            (beta * (delta - delta.ln_1p())) * beta
        };
        let inverse_a = (0.5 * skew).powi(2);
        let stirling = inverse_a
            * (1.0 / 12.0 - inverse_a.powi(2) * (1.0 / 360.0 - inverse_a.powi(2) / 1260.0));
        Ok(-deviance - delta.ln_1p() - 0.5 * core::f64::consts::TAU.ln() - stirling)
    }
    fn near_zero_tail(&self, x: f64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        let small_upper = x >= 0.0;
        let small = if self.expansion(x) {
            self.expanded_tail(x, small_upper)?
        } else if x.abs() > 128.0 {
            // Endpoint Laplace expansion of the positive density integral.
            // The relative remainder is O(3/x^4 + |skew|/|x|^3), < 2e-8 here.
            let skew = self.skew.into_f64();
            let delta = 0.5 * skew * x;
            let density = self.centered_density(x)?;
            if density == f64::NEG_INFINITY {
                return Ok(if upper == small_upper { density } else { 0.0 });
            }
            let denominator = x + 0.5 * skew;
            let correction = -(1.0 - (0.5 * skew).powi(2)) * (1.0 / denominator).powi(2);
            density - denominator.abs().ln() + delta.ln_1p() + correction.ln_1p()
        } else {
            crate::distribution::quadrature::tail(
                |y, _| self.centered_density(y),
                x,
                small_upper,
                -0.5 * self.skew.into_f64(),
                1.0,
                self.options.widened()?,
                left,
            )?
        };
        if upper == small_upper {
            Ok(small)
        } else {
            log1mexp(small)
        }
    }
    fn near_zero_inverse(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let p = p.into_f64();
        let (log_p, upper) = if p > 0.5 {
            ((-p).ln_1p(), !upper)
        } else {
            (p.ln(), upper)
        };
        let z = if upper {
            scioxide_special::normal_isf(log_p.exp())?
        } else {
            scioxide_special::normal_quantile(log_p.exp())?
        };
        let skew = self.skew.into_f64();
        let guess = z + skew * (z * z - 1.0) / 6.0 + skew * skew * z * (z * z - 7.0) / 144.0;
        let edge = -2.0 / skew;
        let bounds = if skew > 0.0 {
            (edge, f64::INFINITY)
        } else {
            (f64::NEG_INFINITY, edge)
        };
        number(crate::inversion::invert_smooth_remaining(
            log_p,
            upper,
            guess,
            bounds,
            &crate::inversion::smooth_options(self.work_budget(), log_p)?,
            |x, upper, left| self.near_zero_tail(x, upper, left),
            |x, _| self.centered_density(x),
            left,
        )?)
    }
    fn centered_gamma_tail(
        &self,
        x: f64,
        upper: bool,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        let beta = 2.0 / self.skew.into_f64();
        let a = beta * beta;
        let z = a + beta * x;
        if z == f64::INFINITY {
            return if upper == (beta > 0.0) {
                Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "Pearson III log tail exceeds the finite scalar range",
                ))
            } else {
                Ok(0.0)
            };
        }
        if z <= 0.0 {
            return Ok(if upper == (beta > 0.0) {
                0.0
            } else {
                f64::NEG_INFINITY
            });
        }
        if upper == (beta > 0.0) {
            crate::law::special(left, |budget| {
                scioxide_special::gamma_log_q_with_options(
                    a,
                    z,
                    scioxide_special::GammaOptions::new(budget),
                )
            })
        } else {
            crate::law::special(left, |budget| {
                scioxide_special::gamma_log_p_with_options(
                    a,
                    z,
                    scioxide_special::GammaOptions::new(budget),
                )
            })
        }
    }
}
impl<S: RealScalar> Formula<S> for Pearson3<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        if self.skew == S::ZERO {
            return (None, None);
        }
        let edge = -S::TWO / self.skew;
        if !edge.is_finite() {
            return (None, None);
        }
        if self.skew > S::ZERO {
            (Some(edge), None)
        } else {
            (None, Some(edge))
        }
    }
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if self.skew.into_f64().abs() >= 0.002 {
            let beta = 2.0 / self.skew.into_f64();
            if beta * beta + beta * x.into_f64() == f64::INFINITY {
                return Ok(if upper == (beta > 0.0) {
                    S::ZERO
                } else {
                    S::ONE
                });
            }
        }
        Ok(self.log_probability(x, upper)?.exp())
    }
    fn log_probability_with_work(
        &self,
        x: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        // The tail kernels already retain their direct complementary forms.
        crate::law::log_probability(self.tail_with_work(x, upper, left)?)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if self.skew == S::ZERO {
            return self.normal.log_pdf(x);
        }
        let y = x.into_f64();
        if self.skew.into_f64().abs() < 0.002 {
            return number(self.centered_density(y)?);
        }
        let beta = 2.0 / self.skew.into_f64();
        if beta * beta + beta * y == f64::INFINITY {
            return Ok(S::NEG_INFINITY);
        }
        let g = crate::Affine::new(
            Gamma::with_options(beta * beta, 1.0, self.options.widened()?)?,
            -beta,
            1.0 / beta,
        )?;
        number(g.log_pdf(y)?)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if self.skew == S::ZERO {
            return if upper {
                self.normal.log_sf(x)
            } else {
                self.normal.log_cdf(x)
            };
        }
        let y = x.into_f64();
        let (low, high) = self.bounds();
        if low.is_some_and(|a| x <= a) {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if high.is_some_and(|a| x >= a) {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        number(if self.skew.into_f64().abs() < 0.002 {
            self.near_zero_tail(y, upper, left)?
        } else {
            self.centered_gamma_tail(y, upper, left)?
        })
    }

    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if upper {
            if p == S::ZERO || p == S::ONE {
                let (low, high) = self.bounds();
                return Ok(if p != S::ZERO {
                    low.unwrap_or(S::NEG_INFINITY)
                } else {
                    high.unwrap_or(S::INFINITY)
                });
            }
            if self.skew != S::ZERO && self.skew.into_f64().abs() < 0.002 {
                return self.near_zero_inverse(p, true, left);
            }
            if self.gamma.is_some() {
                let beta = 2.0 / self.skew.into_f64();
                let g = crate::Affine::new(
                    Gamma::with_options(beta * beta, 1.0, self.options.widened()?)?,
                    -beta,
                    1.0 / beta,
                )?;
                number(crate::law::scoped(left, |context| {
                    g.isf_scoped(p.into_f64(), context)
                })?)
            } else {
                // Only skew = 0 has no gamma form outside the near-zero branch.
                self.normal.isf(p)
            }
        } else {
            if p == S::ZERO || p == S::ONE {
                let (low, high) = self.bounds();
                return Ok(if p == S::ZERO {
                    low.unwrap_or(S::NEG_INFINITY)
                } else {
                    high.unwrap_or(S::INFINITY)
                });
            }
            if self.skew != S::ZERO && self.skew.into_f64().abs() < 0.002 {
                return self.near_zero_inverse(p, false, left);
            }
            if self.gamma.is_some() {
                let beta = 2.0 / self.skew.into_f64();
                let g = crate::Affine::new(
                    Gamma::with_options(beta * beta, 1.0, self.options.widened()?)?,
                    -beta,
                    1.0 / beta,
                )?;
                number(crate::law::scoped(left, |context| {
                    g.quantile_scoped(p.into_f64(), context)
                })?)
            } else {
                // Only skew = 0 has no gamma form outside the near-zero branch.
                self.normal.quantile(p)
            }
        }
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if let Some(g) = &self.gamma {
            g.sample(rng)
        } else if self.skew == S::ZERO {
            self.normal.sample(rng)
        } else {
            self.inverse_with_remaining(crate::util::validated_open_float(rng)?, false, left)
        }
    }
}
formula_law!(Pearson3);
impl<S: RealScalar> Mean<S> for Pearson3<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for Pearson3<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        Ok(S::ONE)
    }
}

impl<S: RealScalar> crate::LogCoordinateLaw<S> for Gamma<S> {
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let z = y - self.log_scale;
        if z.exp() == S::INFINITY {
            return Ok(S::ONE);
        }
        if z < S::EPSILON.ln() {
            return Ok((self.shape * z - self.log_gamma_one_plus.clone()?).exp());
        }
        Ok(self.log_cdf_at_log(y)?.exp())
    }
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let z = y - self.log_scale;
        if z.exp() == S::INFINITY {
            return Ok(S::ZERO);
        }
        if z < S::EPSILON.ln() {
            let log_p = self.shape * z - self.log_gamma_one_plus.clone()?;
            return Ok(-log_p.exp_m1());
        }
        Ok(self.log_sf_at_log(y)?.exp())
    }
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let z = y - self.log_scale;
        if z.exp() == S::INFINITY {
            return Ok(S::ZERO);
        }
        finite((self.shape * z - z.exp() - self.log_gamma.clone()?).exp())
    }
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        gamma_log_coordinate(
            self.shape,
            y - self.log_scale,
            false,
            &mut self.options.work_budget().get(),
        )
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        gamma_log_coordinate(
            self.shape,
            y - self.log_scale,
            true,
            &mut self.options.work_budget().get(),
        )
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let log_z = y - self.log_scale;
        finite(self.shape * log_z - log_z.exp() - self.log_gamma.clone()?)
    }

    fn log_cdf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        crate::law::with_work(context, self.options.work_budget(), |left| {
            gamma_log_coordinate(self.shape, y - self.log_scale, false, left)
        })
    }

    fn log_sf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        crate::law::with_work(context, self.options.work_budget(), |left| {
            gamma_log_coordinate(self.shape, y - self.log_scale, true, left)
        })
    }
}
impl<S: RealScalar> crate::LogCoordinateLaw<S> for GeneralizedGamma<S> {
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.coordinate_probability(y, false)
    }
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.coordinate_probability(y, true)
    }
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let z = self.c * y;
        if z.exp() == S::INFINITY {
            return Ok(S::ZERO);
        }
        finite((self.c.abs().ln() + self.a * z - z.exp() - self.log_gamma.clone()?).exp())
    }
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        gamma_log_coordinate(
            self.a,
            self.c * y,
            self.c < S::ZERO,
            &mut self.options.work_budget().get(),
        )
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        gamma_log_coordinate(
            self.a,
            self.c * y,
            self.c > S::ZERO,
            &mut self.options.work_budget().get(),
        )
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        let log_z = self.c * y;
        finite(self.c.abs().ln() + self.a * log_z - log_z.exp() - self.log_gamma.clone()?)
    }

    fn log_cdf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        crate::law::with_work(context, self.options.work_budget(), |left| {
            gamma_log_coordinate(self.a, self.c * y, self.c < S::ZERO, left)
        })
    }

    fn log_sf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        argument(y, "y")?;
        crate::law::with_work(context, self.options.work_budget(), |left| {
            gamma_log_coordinate(self.a, self.c * y, self.c > S::ZERO, left)
        })
    }
}
