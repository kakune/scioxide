//! Logistic, generalized logistic and hyperbolic secant distributions.
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::law::constant as number;
use crate::law::log1mexp;
use crate::law::positive;
use crate::law::{argument, constant, finite, probability};
use crate::{
    Cdf, Density, Distribution, Entropy, InverseSurvival, LogCdf, LogDensity, LogSurvival, Mean,
    Quantile, Rng, Support, Survival, Variance,
};
use scioxide_core::{ExtendedInterval, RealScalar, SciOxideError};

/// Logistic distribution with location `μ` and scale `s > 0`.
///
/// `F(x) = 1 / (1 + exp(−(x − μ)/s))` on the real line. Draws and quantiles use
/// inversion, `μ + s ln(p / (1 − p))`.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Logistic, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Variance};
/// let law = Logistic::new(1.0_f64, 2.0)?;
/// assert_eq!(law.mean()?, 1.0);
/// assert!((law.variance()? - 4.0 * core::f64::consts::PI.powi(2) / 3.0).abs() < 1e-14);
/// assert!((law.cdf(1.0 + 2.0 * 3.0_f64.ln())? - 0.75).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(25);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Logistic<S> {
    location: S,
    scale: S,
}
impl<S: RealScalar> Logistic<S> {
    /// Creates a logistic distribution with location `μ` and scale `s`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `location` or `scale` is NaN or infinite.
    /// - `DomainError` if `scale ≤ 0`.
    pub fn new(location: S, scale: S) -> Result<Self, SciOxideError> {
        argument(location, "location")?;
        argument(scale, "scale")?;
        if scale <= S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("scale"),
                "logistic scale must be positive",
            ));
        }
        Ok(Self { location, scale })
    }
    fn z(&self, x: S) -> S {
        let d = x - self.location;
        if d.is_finite() {
            d / self.scale
        } else {
            x / self.scale - self.location / self.scale
        }
    }
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let z = if upper { -self.z(x) } else { self.z(x) };
        // Finite coordinates may standardize beyond the scalar range.
        if z == S::INFINITY {
            Ok(S::ONE)
        } else if z == S::NEG_INFINITY {
            Ok(S::ZERO)
        } else {
            scioxide_core::sigmoid(z)
        }
    }
}
impl<S: RealScalar> Support<S> for Logistic<S> {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        Ok(ExtendedInterval::entire())
    }
}
impl<S: RealScalar> LogCdf<S> for Logistic<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        Ok(-scioxide_core::softplus(-self.z(x))?)
    }
}
impl<S: RealScalar> LogSurvival<S> for Logistic<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        Ok(-scioxide_core::softplus(self.z(x))?)
    }
}
impl<S: RealScalar> Cdf<S> for Logistic<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.probability(x, false)
    }
}
impl<S: RealScalar> Survival<S> for Logistic<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        self.probability(x, true)
    }
}
impl<S: RealScalar> LogDensity<S> for Logistic<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        // The density is symmetric, so evaluate at -|z| where e^{-|z|} <= 1
        // cannot overflow: log f = -|z| - 2 ln(1 + e^{-|z|}) - ln s.
        let z = self.z(x).abs();
        Ok(-z - constant::<S>(2.0)? * (-z).exp().ln_1p() - self.scale.ln())
    }
}
impl<S: RealScalar> Density<S> for Logistic<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        finite(self.log_pdf(x)?.exp())
    }
}
/// `ln(p / (1 - p))` for `0 < p < 1` with one logarithm. On `[1/4, 3/4]` the
/// numerator `2p - 1` (or `1 - 2p`) and `1 - p` are exact by Sterbenz's lemma,
/// so `ln_1p` of their quotient keeps the relative accuracy of a small result;
/// outside it `|logit| > ln 3` bounds the error of the rounded quotient.
fn logit<S: RealScalar>(p: S) -> S {
    let quarter = S::HALF * S::HALF;
    if p >= S::HALF {
        if p <= S::ONE - quarter {
            ((p + p - S::ONE) / (S::ONE - p)).ln_1p()
        } else {
            (p / (S::ONE - p)).ln()
        }
    } else if p >= quarter {
        -((S::ONE - (p + p)) / p).ln_1p()
    } else {
        (p / (S::ONE - p)).ln()
    }
}

impl<S: RealScalar> Quantile<S> for Logistic<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        finite(self.scale.mul_add(logit(p), self.location))
    }
}
impl<S: RealScalar> InverseSurvival<S> for Logistic<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        finite((-self.scale).mul_add(logit(p), self.location))
    }
}
impl<S: RealScalar> Distribution<S> for Logistic<S> {
    /// Draws `μ + s ln(u / (1 − u))` from an open uniform `u`.
    #[inline]
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let u = crate::util::validated_open_float(rng)?;
        finite(self.scale.mul_add(logit(u), self.location))
    }
}
impl<S: RealScalar> Mean<S> for Logistic<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.location)
    }
}
impl<S: RealScalar> Variance<S> for Logistic<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        super::moment_math::variance(
            2.0 * self.scale.into_f64().ln() + (core::f64::consts::PI.powi(2) / 3.0).ln(),
        )
    }
}
impl<S: RealScalar> Entropy<S> for Logistic<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        Ok(self.scale.ln() + constant(2.0)?)
    }
}

/// Type I generalized logistic distribution with shape `a > 0`.
///
/// `F(x) = (1 + exp(−x))^(−a)` on the real line. Shape one is the standard
/// logistic law. Shift and scale it with [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{Cdf, Distribution, GeneralizedLogistic, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Variance};
/// let law = GeneralizedLogistic::new(2.0_f64)?;
/// assert!((law.cdf(0.0)? - 0.25).abs() < 1e-16);
/// assert!((law.mean()? - 1.0).abs() < 1e-14); // ψ(2) - ψ(1)
/// assert!((law.variance()? - (core::f64::consts::PI.powi(2) / 3.0 - 1.0)).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(26);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedLogistic<S: RealScalar> {
    shape: S,
}
impl<S: RealScalar> GeneralizedLogistic<S> {
    /// Creates a type I generalized logistic distribution with shape `a`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self { shape })
    }
    fn inverse_log_cdf(&self, log_p: S) -> Result<S, SciOxideError> {
        let rate = -log_p / self.shape;
        if rate < S::MIN_POSITIVE {
            // The quotient can vanish while its logarithm is representable.
            return finite(self.shape.ln() - (-log_p).ln());
        }
        finite(-crate::law::log_expm1(rate)?)
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedLogistic<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.shape.ln() - x - (self.shape + S::ONE) * scioxide_core::softplus(-x)?)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            // Form a*softplus(-x) before complementing. The leading term is
            // valid only when that entire rate vanishes, including its shape.
            let log_rate = self.shape.ln() + crate::law::log_softplus(-x)?;
            let rate = log_rate.exp();
            if rate == S::ZERO {
                Ok(log_rate)
            } else {
                log1mexp(-rate)
            }
        } else {
            Ok(-self.shape * scioxide_core::softplus(-x)?)
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        self.inverse_log_cdf(p.ln())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        self.inverse_log_cdf((-p).ln_1p())
    }
}
formula_law!(GeneralizedLogistic);
impl<S: RealScalar> Mean<S> for GeneralizedLogistic<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(scioxide_special::digamma(self.shape)? - scioxide_special::digamma(S::ONE)?)
    }
}
impl<S: RealScalar> Variance<S> for GeneralizedLogistic<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(
            scioxide_special::polygamma(1, self.shape)? + scioxide_special::polygamma(1, S::ONE)?,
        )
    }
}

/// Generalized half logistic distribution on `[0, 1/c]` with shape `c > 0`.
///
/// `F(x) = (1 − t) / (1 + t)` with `t = (1 − cx)^(1/c)`. The quantile and
/// inverse survival function are closed-form: a survival probability `p` gives
/// `t = p / (2 − p)`. [`Mean`](crate::Mean) and [`Variance`](crate::Variance)
/// are not implemented.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, GeneralizedHalfLogistic, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support};
/// let law = GeneralizedHalfLogistic::new(1.0_f64)?; // F(x) = x / (2 - x)
/// assert!((law.cdf(0.5)? - 1.0 / 3.0).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(27);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedHalfLogistic<S: RealScalar> {
    c: S,
}
impl<S: RealScalar> GeneralizedHalfLogistic<S> {
    /// Creates a generalized half logistic distribution with shape `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` is NaN or infinite.
    /// - `DomainError` if `c ≤ 0`.
    /// - `NumericalInstability` if the upper support end `1/c` overflows.
    pub fn new(c: S) -> Result<Self, SciOxideError> {
        positive(c, scioxide_core::argument!("c"))?;
        crate::law::positive_result(S::ONE / c)?;
        Ok(Self { c })
    }
    fn log_t(&self, x: S) -> S {
        (-self.c * x).ln_1p() / self.c
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedHalfLogistic<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), Some(S::ONE / self.c))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO || x > S::ONE / self.c {
            return Ok(S::NEG_INFINITY);
        }
        let lt = self.log_t(x);
        if lt == S::NEG_INFINITY {
            return Ok(if self.c > S::ONE {
                S::INFINITY
            } else if self.c == S::ONE {
                number::<S>(2.)?.ln()
            } else {
                S::NEG_INFINITY
            });
        }
        Ok(number::<S>(2.)?.ln() + (S::ONE - self.c) * lt - number::<S>(2.)? * lt.exp().ln_1p())
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            if x >= S::ONE / self.c {
                return Ok(S::NEG_INFINITY);
            }
            let lt = self.log_t(x);
            Ok(number::<S>(2.)?.ln() + lt - lt.exp().ln_1p())
        } else {
            if x <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            if x >= S::ONE / self.c {
                return Ok(S::ZERO);
            }
            let lt = self.log_t(x);
            Ok(log1mexp(lt)? - lt.exp().ln_1p())
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            return Ok(S::ONE / self.c);
        }
        finite(-(self.c * ((-p).ln_1p() - p.ln_1p())).exp_m1() / self.c)
    }

    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ONE / self.c);
        }
        finite(-(self.c * (p.ln() - (S::TWO - p).ln())).exp_m1() / self.c)
    }
}
formula_law!(GeneralizedHalfLogistic);

/// Standard hyperbolic secant distribution.
///
/// `F(x) = (2/π) atan(exp(πx/2))` on the real line. The law has mean zero and
/// unit variance. Shift and scale it with [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{Cdf, Distribution, HyperbolicSecant, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Variance};
/// let law = HyperbolicSecant::<f64>::new();
/// assert_eq!((law.mean()?, law.variance()?), (0.0, 1.0));
/// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(28);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicSecant<S: RealScalar>(core::marker::PhantomData<S>);
impl<S: RealScalar> HyperbolicSecant<S> {
    /// Creates the standard hyperbolic secant distribution.
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}
impl<S: RealScalar> Default for HyperbolicSecant<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> Formula<S> for HyperbolicSecant<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let z = (S::PI * S::HALF) * x.abs();
        Ok(-z - (-z).exp().powi(2).ln_1p())
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        let x = if upper { -x } else { x };
        if x > S::ZERO {
            return log1mexp(self.tail_with_work(x, true, _left)?);
        }
        // F(x) = (2 / pi) atan(e^{pi x / 2}). In the far lower tail
        // atan(z) = z (1 - z^2 / 3 + ...) equals z to working precision, so the
        // log CDF is linear in x and never needs e^{pi x / 2} itself.
        let log_z = (S::PI * S::HALF) * x;
        if log_z < S::EPSILON.ln() * S::HALF {
            Ok(S::LN_2 - S::PI.ln() + log_z)
        } else {
            let z = log_z.exp();
            Ok((number::<S>(2.)? / S::PI * z.atan()).ln())
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        let t = S::PI * if p <= S::HALF { p } else { S::ONE - p } * S::HALF;
        let y = number::<S>(2.)? / S::PI * t.tan().ln();
        finite(if p <= S::HALF { y } else { -y })
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        Ok(-self.inverse(p)?)
    }
}
formula_law!(HyperbolicSecant);
impl<S: RealScalar> Mean<S> for HyperbolicSecant<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for HyperbolicSecant<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        Ok(S::ONE)
    }
}
