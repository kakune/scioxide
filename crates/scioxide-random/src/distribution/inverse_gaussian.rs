//! Inverse Gaussian and reciprocal inverse Gaussian distributions.
use crate::distribution::formula::formula_law;
use crate::distribution::formula::positive_support;
use crate::distribution::formula::{Formula, Inverse};
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{finite, logadd, logdiff};
use crate::{Distribution, InverseSurvival, LogCdf, LogDensity, LogSurvival, Quantile};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Inverse Gaussian (Wald) distribution on `x > 0` with mean `μ > 0` and unit shape.
///
/// ```text
/// f(x) = exp(−(x − μ)² / (2μ²x)) / √(2π x³).
/// ```
///
/// Scale it with [`Affine`](crate::Affine). Draws use the transformation with
/// multiple roots of Michael, Schucany and Haas: one normal and one uniform
/// variate per draw.
///
/// # References
///
/// J. R. Michael, W. R. Schucany and R. W. Haas, "Generating random variates
/// using transformations with multiple roots", *Amer. Statist.* 30(2), 1976,
/// 88–90.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, InverseGaussian, Mean, Quantile, SeedableRng, SplitMix64,
///     Variance,
/// };
/// let law = InverseGaussian::<f64>::new(2.0)?;
/// assert_eq!(law.mean()?, 2.0);
/// assert!((law.variance()? - 8.0).abs() < 1e-15);
/// let x = law.quantile(0.4)?;
/// assert!((law.cdf(x)? - 0.4).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(63);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct InverseGaussian<S: RealScalar> {
    options: crate::LawOptions<S>,
    mu: S,
    normal: super::Normal<S>,
}
impl<S: RealScalar> InverseGaussian<S> {
    /// Creates an inverse Gaussian distribution with mean `μ = mu`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `mu` is NaN or infinite.
    /// - `DomainError` if `mu ≤ 0`.
    pub fn new(mu: S) -> Result<Self, SciOxideError> {
        Self::with_options(mu, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(mu: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(mu, scioxide_core::argument!("mu"))?;
        Ok(Self {
            options,
            mu,
            normal: super::Normal::standard(),
        })
    }
    // Unit-shape inverse Gaussian: F(x) = Phi(a) + e^{2/mu} Phi(b) with
    // a = (x/mu - 1)/sqrt(x) and b = -(x/mu + 1)/sqrt(x). The e^{2/mu} factor
    // is folded into the log so it cannot overflow for small mu.
    fn parts(&self, x: S) -> Result<(f64, f64), SciOxideError> {
        let (x, mu) = (x.into_f64(), self.mu.into_f64());
        let r = x.sqrt();
        let ratio = x / mu;
        let a = if ratio.is_finite() {
            (ratio - 1.0) / r
        } else {
            r / mu - 1.0 / r
        };
        let b = r / mu + 1.0 / r;
        // b^2-a^2 = 4/mu. Cancel the exponential analytically before
        // evaluating the scaled complementary error function.
        let log_b = if b.is_infinite() {
            f64::NEG_INFINITY
        } else {
            -0.5 * a * a + scioxide_special::erfcx(b * core::f64::consts::FRAC_1_SQRT_2)?.ln()
                - core::f64::consts::LN_2
        };
        Ok((a, log_b))
    }
}
impl<S: RealScalar> Formula<S> for InverseGaussian<S> {
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
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let z = (x / self.mu - S::ONE) / x.sqrt();
        Ok(-crate::distribution::special::log_two_pi::<S>() * S::HALF
            - number::<S>(1.5)? * x.ln()
            - z * z * S::HALF)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            // S(x) = Phi(-a) - e^{2/mu} Phi(b): the upper tail is taken directly
            // from `log_sf` rather than as 1 - F, and the difference in log space.
            let (a, b) = self.parts(x)?;
            let tail = super::special::normal_log(a, true)?;
            let value = logdiff(tail, b, scioxide_core::argument!("self"))?;
            if value == f64::NEG_INFINITY && tail.is_finite() {
                return Err(super::moment_math::cancellation());
            }
            number(value)
        } else {
            if x <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            let (a, b) = self.parts(x)?;
            number(logadd(super::special::normal_log(a, false)?, b)?)
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        let z = self.normal.sample(rng)?;
        let y = z.into_f64().powi(2);
        let m = self.mu.into_f64();
        // Michael, Schucany & Haas (1976) with unit shape. The smaller root
        // mu (1 + v/2 - sqrt(v (4 + v)) / 2), v = mu Z^2, is rationalized to
        // 2 mu / (2 + v + sqrt(v (4 + v))) so large v does not cancel. The
        // root is then chosen against its reciprocal partner mu^2 / x with
        // probability mu / (mu + x).
        let v = m * y;
        let x = if v.is_finite() {
            m / (1.0 + 0.5 * v + v.sqrt() * (1.0 + 0.25 * v).sqrt())
        } else {
            // The relative correction is O(1/v), below binary64 rounding.
            y.recip()
        };
        let u = crate::util::validated_open_float(rng)?.into_f64();
        let value = if u <= 1.0 / (1.0 + x / m) {
            x
        } else {
            m / (x / m)
        };
        scioxide_core::Scaled::from_value(finite(value)?)?.try_value_to()
    }
}
formula_law!(InverseGaussian);
impl<S: RealScalar> Mean<S> for InverseGaussian<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.mu)
    }
}
impl<S: RealScalar> Variance<S> for InverseGaussian<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        crate::law::positive_result(self.mu * self.mu * self.mu)
    }
}

/// Reciprocal inverse Gaussian distribution: the law of `1/X` for `X` from
/// [`InverseGaussian`] with mean `μ` and unit shape.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, Quantile, ReciprocalInverseGaussian, SeedableRng,
///     SplitMix64, Variance,
/// };
/// let law = ReciprocalInverseGaussian::<f64>::new(2.0)?;
/// assert!((law.mean()? - 1.5).abs() < 1e-15);
/// assert!((law.variance()? - 2.5).abs() < 1e-15);
/// let x = law.quantile(0.4)?;
/// assert!((law.cdf(x)? - 0.4).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(64);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ReciprocalInverseGaussian<S: RealScalar> {
    base: InverseGaussian<S>,
}
impl<S: RealScalar> ReciprocalInverseGaussian<S> {
    /// Creates the law of `1/X` for `X` inverse Gaussian with mean `μ = mu`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `mu` is NaN or infinite.
    /// - `DomainError` if `mu ≤ 0`.
    pub fn new(mu: S) -> Result<Self, SciOxideError> {
        Self::with_options(mu, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(mu: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            base: InverseGaussian::with_options(mu, options)?,
        })
    }
}
impl<S: RealScalar> Formula<S> for ReciprocalInverseGaussian<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        positive_support()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            Ok(S::NEG_INFINITY)
        } else {
            Ok(self.base.log_pdf(S::ONE / x)? - number::<S>(2.)? * x.ln())
        }
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                Ok(S::ZERO)
            } else {
                crate::law::scoped(left, |context| {
                    self.base.log_cdf_scoped(S::ONE / x, context)
                })
            }
        } else {
            if x <= S::ZERO {
                Ok(S::NEG_INFINITY)
            } else {
                crate::law::scoped(left, |context| self.base.log_sf_scoped(S::ONE / x, context))
            }
        }
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
            finite(
                S::ONE / crate::law::scoped(left, |context| self.base.quantile_scoped(p, context))?,
            )
        } else {
            if p == S::ZERO {
                return Ok(S::ZERO);
            }
            if p == S::ONE {
                return Ok(S::INFINITY);
            }
            finite(S::ONE / crate::law::scoped(left, |context| self.base.isf_scoped(p, context))?)
        }
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.base.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.base.byte_limit()
    }
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.base.tolerance()
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        finite(S::ONE / self.base.sample(rng)?)
    }
}
formula_law!(ReciprocalInverseGaussian);
impl<S: RealScalar> Mean<S> for ReciprocalInverseGaussian<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(S::ONE / self.base.mu + S::ONE)
    }
}
impl<S: RealScalar> Variance<S> for ReciprocalInverseGaussian<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(S::ONE / self.base.mu + number::<S>(2.)?)
    }
}
