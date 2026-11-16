//! Cauchy (Lorentz) distribution.

use scioxide_core::backend::checked_final_rounding;
use scioxide_core::{RealScalar, SciOxideError};

use crate::traits::{Cdf, Density, Distribution, Quantile, Rng};

/// Cauchy (Lorentz) distribution with location `x₀` and scale `γ`.
///
/// ```text
/// f(x) = 1 / (πγ (1 + ((x − x₀)/γ)²)),   x ∈ ℝ, γ > 0.
/// ```
///
/// Sampling inverts the CDF: a sample is `x₀ + γ tan(π(U − 1/2))` with `U`
/// uniform on `(0, 1)`. The quantiles at `p = 0` and `p = 1` are `−∞` and
/// `+∞`. Every other quantile is finite. The law has no mean or variance, so
/// it does not implement [`Mean`](crate::Mean) or [`Variance`](crate::Variance).
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cauchy, Cdf, Density, Distribution, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = Cauchy::new(1.0_f64, 2.0)?;
/// assert!((law.pdf(1.0)? - 0.5 / core::f64::consts::PI).abs() < 1e-16);
/// assert!((law.cdf(3.0)? - 0.75).abs() < 1e-15 && (law.sf(3.0)? - 0.25).abs() < 1e-15);
/// assert!((law.quantile(0.75)? - 3.0).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(23);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Cauchy<S> {
    location: S,
    scale: S,
}

impl<S: RealScalar> Cauchy<S> {
    /// Creates a Cauchy distribution with location `location` and scale
    /// `scale`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `location` or `scale` is NaN or infinite.
    /// - `DomainError` if `scale ≤ 0`.
    pub fn new(location: S, scale: S) -> Result<Self, SciOxideError> {
        crate::law::argument(location, "location")?;
        crate::law::positive(scale, scioxide_core::argument!("scale"))?;
        Ok(Self { location, scale })
    }
}

impl<S: RealScalar> Distribution<S> for Cauchy<S> {
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let u: S = crate::util::validated_open_float(rng)?;
        self.interior_quantile(u)
    }
}

impl<S: RealScalar> Cauchy<S> {
    fn interior_quantile(&self, probability: S) -> Result<S, SciOxideError> {
        let half = S::HALF;
        if probability == half {
            return Ok(self.location);
        }
        // Q(p) = loc + scale tan(pi (p - 1/2)) = loc - scale / tan(pi p).
        // Using the reflected form on each side keeps the argument of `tan`
        // in (0, pi/2), where it is small near the tails and loses nothing to
        // the rounding of `p - 1/2`.
        let pi = S::PI;
        let (tangent, scale) = if probability < half {
            ((pi * probability).tan(), -self.scale)
        } else {
            ((pi * (S::ONE - probability)).tan(), self.scale)
        };
        self.shifted_cotangent(tangent, scale)
    }

    fn shifted_cotangent(&self, tangent: S, scale: S) -> Result<S, SciOxideError> {
        let mut numerator = scioxide_core::RangeAccumulator::new();
        numerator.add_product(self.location, tangent)?;
        numerator.add(scale)?;
        numerator.finish_quotient(tangent)
    }
}

impl<S: RealScalar> Density<S> for Cauchy<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let pi = S::PI;
        let difference = x - self.location;
        let z = if difference.is_finite() {
            difference / self.scale
        } else {
            x / self.scale - self.location / self.scale
        };
        let normalizer = (S::ONE / self.scale) / pi;
        // 1 / (1 + z^2) rewritten as z^-2 / (1 + z^-2) for |z| > 1 so z^2
        // cannot overflow to infinity and flush a representable density.
        let density = if z.abs() > S::ONE {
            let inverse = S::ONE / z;
            let inverse_squared = inverse * inverse;
            normalizer * inverse_squared / (S::ONE + inverse_squared)
        } else {
            normalizer / (S::ONE + z * z)
        };
        checked_final_rounding(
            if density == S::ZERO || !density.is_finite() {
                crate::LogDensity::log_pdf(self, x)?.exp()
            } else {
                density
            },
            "Cauchy density exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Cdf<S> for Cauchy<S> {
    // Keep the branch-heavy tail projection out of mixed-sign bulk callers.
    #[inline(never)]
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let pi = S::PI;
        let half = S::HALF;
        let difference = x - self.location;
        let z = if difference.is_finite() {
            difference / self.scale
        } else {
            x / self.scale - self.location / self.scale
        };
        // F(z) = 1/2 + atan(z)/pi. For z < 0 this equals atan(-1/z)/pi, which
        // keeps small lower-tail probabilities accurate instead of forming
        // them as 1/2 minus nearly 1/2.
        if !z.is_finite() {
            return Ok(self.log_tail(x, false)?.exp());
        }
        let probability = if z < S::ZERO {
            (-S::ONE / z).atan() / pi
        } else if z > S::ZERO {
            S::ONE - (S::ONE / z).atan() / pi
        } else {
            half
        };
        // The positive reciprocal's atan lies in [0, pi/2], so these two
        // branches already preserve [0,1]. Check the arithmetic projection.
        checked_final_rounding(probability, "Cauchy CDF exceeds the finite scalar range")
    }
}

impl<S: RealScalar> Quantile<S> for Cauchy<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        self.interior_quantile(p)
    }
}

impl<S: RealScalar> crate::Support<S> for Cauchy<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(None, None)
    }
}

impl<S: RealScalar> Cauchy<S> {
    fn log_distance(&self, x: S) -> Result<S, SciOxideError> {
        // ln|x - loc|, halving both operands first if the difference overflows.
        let d = x - self.location;
        if d.is_finite() {
            Ok(d.abs().ln())
        } else {
            let h = S::HALF;
            Ok((x * h - self.location * h).abs().ln() - h.ln())
        }
    }
    fn log_tail(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x == self.location {
            return Ok(-S::LN_2);
        }
        // The far tail has probability atan(scale / |x - loc|) / pi; `ratio` is
        // the logarithm of that argument, and the near side is its complement.
        let ratio = self.scale.ln() - self.log_distance(x)?;
        let log_pi = S::try_from_f64_lossy(core::f64::consts::PI.ln())?;
        // atan(r)/r differs from one by O(r^2); retain tiny tails in log space.
        let small = if ratio < S::EPSILON.ln() {
            ratio - log_pi
        } else {
            ratio.exp().atan().ln() - log_pi
        };
        Ok(if (x > self.location) == upper {
            small
        } else {
            crate::law::log1mexp(small)?
        })
    }
}
impl<S: RealScalar> crate::LogDensity<S> for Cauchy<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let z = self.log_distance(x)? - self.scale.ln();
        let l = crate::law::logadd(S::ZERO, S::TWO * z)?;
        Ok(-self.scale.ln() - S::try_from_f64_lossy(core::f64::consts::PI.ln())? - l)
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for Cauchy<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_tail(x, true)
    }
}
impl<S: RealScalar> crate::LogCdf<S> for Cauchy<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_tail(x, false)
    }
}
impl<S: RealScalar> crate::Survival<S> for Cauchy<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(crate::LogSurvival::log_sf(self, x)?.exp())
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for Cauchy<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        let half = S::HALF;
        if p == half {
            return Ok(self.location);
        }
        let pi = S::PI;
        let (tangent, scale) = if p < half {
            ((pi * p).tan(), self.scale)
        } else {
            ((pi * (S::ONE - p)).tan(), -self.scale)
        };
        self.shifted_cotangent(tangent, scale)
    }
}
impl<S: RealScalar> crate::Entropy<S> for Cauchy<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        Ok(self.scale.ln() + S::try_from_f64_lossy((4.0 * core::f64::consts::PI).ln())?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::SeedableRng;

    #[test]
    fn test_new_valid() {
        assert!(Cauchy::<f64>::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn test_new_invalid_scale() {
        let error = Cauchy::<f64>::new(0.0, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_params() {
        let error = Cauchy::<f64>::new(f64::NAN, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = Cauchy::<f64>::new(0.0, f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_finite() {
        let dist = Cauchy::new(0.0_f64, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        // Cauchy has heavy tails but samples should be finite
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x.is_finite());
        }
    }

    #[test]
    fn test_pdf_at_location() {
        let dist = Cauchy::new(0.0_f64, 1.0).unwrap();
        let expected = 1.0 / std::f64::consts::PI;
        assert!((dist.pdf(0.0).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_pdf_symmetric() {
        let dist = Cauchy::new(2.0_f64, 1.0).unwrap();
        let p1 = dist.pdf(3.0).unwrap();
        let p2 = dist.pdf(1.0).unwrap();
        assert!((p1 - p2).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_at_location() {
        let dist = Cauchy::new(0.0_f64, 1.0).unwrap();
        assert!((dist.cdf(0.0).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_median() {
        let dist = Cauchy::new(3.0_f64, 2.0).unwrap();
        assert!((dist.quantile(0.5).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn quantile_endpoints_are_exact_support_limits() {
        let dist = Cauchy::new(3.0_f64, 2.0).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = Cauchy::new(1.0_f64, 2.0).unwrap();
        let p = 0.75;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = Cauchy::new(0.0_f64, 1.0).unwrap();
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
    fn finite_extreme_tails_and_non_finite_queries_are_consistent() {
        let dist = Cauchy::new(0.0_f64, 1.0).unwrap();
        assert_eq!(dist.pdf(f64::MAX).unwrap(), 0.0);
        let lower_tail = dist.cdf(-f64::MAX).unwrap();
        assert!(lower_tail.is_subnormal() && lower_tail > 0.0);
        assert_eq!(dist.cdf(f64::MAX).unwrap(), 1.0);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = dist.pdf(value).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
            let error = dist.cdf(value).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        }

        let wide = Cauchy::new(0.0_f64, f64::MAX).unwrap();
        let density = wide.pdf(0.0).unwrap();
        assert!(density.is_subnormal() && density > 0.0);
    }
}
