//! Weibull, double Weibull, inverse Weibull and exponentiated Weibull distributions.
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::distribution::formula::positive_support;
use crate::distribution::symmetric;
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{finite, log1mexp};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Weibull distribution on `x ≥ 0` with shape `c > 0` and unit scale.
///
/// The survival function is `S(x) = exp(−x^c)`. A draw is `E^(1/c)` for a
/// standard exponential `E`. Shift and scale the law with
/// [`Affine`](crate::Affine). For example,
/// `Affine::new(WeibullMin::new(2.0)?, 0.0, 2.0_f64.sqrt())` is the standard
/// Rayleigh law.
///
/// ```
/// use scioxide_random::{Distribution, Mean, Quantile, SeedableRng, SplitMix64};
/// use scioxide_random::{Support, Survival, Variance, WeibullMin};
/// let law = WeibullMin::new(2.0_f64)?;
/// assert!((law.sf(1.5)? - (-2.25_f64).exp()).abs() < 1e-16);
/// assert!((law.mean()? - core::f64::consts::PI.sqrt() / 2.0).abs() < 1e-15);
/// assert!((law.variance()? - (1.0 - core::f64::consts::FRAC_PI_4)).abs() < 1e-15);
/// assert!((law.sf(law.quantile(0.3)?)? - 0.7).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(33);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WeibullMin<S: RealScalar> {
    shape: S,
    exponent: S,
}
impl<S: RealScalar> WeibullMin<S> {
    /// Creates a Weibull distribution with shape `c = shape`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self {
            shape,
            exponent: S::ONE / shape,
        })
    }
}
impl<S: RealScalar> Formula<S> for WeibullMin<S> {
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        let log = self.tail_with_work(x, true, &mut self.work_budget().get())?;
        // Subtraction also keeps the exact lower-support limit at +0.
        Ok(if upper {
            log.exp()
        } else {
            S::ZERO - log.exp_m1()
        })
    }
    fn bounds(&self) -> (Option<S>, Option<S>) {
        positive_support()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            return Ok(if self.shape < S::ONE {
                S::INFINITY
            } else if self.shape == S::ONE {
                S::ZERO
            } else {
                S::NEG_INFINITY
            });
        }
        Ok(self.shape.ln() + (self.shape - S::ONE) * x.ln() - x.powf(self.shape))
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(if x <= S::ZERO {
                S::ZERO
            } else {
                -x.powf(self.shape)
            })
        } else {
            Ok(if x <= S::ZERO {
                S::NEG_INFINITY
            } else {
                log1mexp(-x.powf(self.shape))?
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            Ok(S::INFINITY)
        } else {
            finite((-(-p).ln_1p()).powf(self.exponent))
        }
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            Ok(S::INFINITY)
        } else {
            finite((-p.ln()).powf(self.exponent))
        }
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        Ok(super::exponential::standard_exponential(rng)?.powf(self.exponent))
    }
}
formula_law!(WeibullMin);
impl<S: RealScalar> Mean<S> for WeibullMin<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        scioxide_special::gamma(S::ONE + S::ONE / self.shape)
    }
}
impl<S: RealScalar> Variance<S> for WeibullMin<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        super::moment_math::gamma_variance(1.0, self.shape.into_f64().recip())
    }
}

/// Double Weibull distribution with shape `c > 0`.
///
/// The density is `f(x) = (c/2) |x|^(c−1) exp(−|x|^c)` on the real line. Shape
/// one is the standard Laplace law. Shift and scale it with
/// [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{Cdf, Distribution, DoubleWeibull, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Variance};
/// let law = DoubleWeibull::new(2.0_f64)?;
/// assert_eq!((law.mean()?, law.variance()?), (0.0, 1.0)); // Γ(1 + 2/c)
/// assert!((law.cdf(-1.0)? - 0.5 / core::f64::consts::E).abs() < 1e-16);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(34);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DoubleWeibull<S: RealScalar> {
    magnitude: WeibullMin<S>,
}
impl<S: RealScalar> DoubleWeibull<S> {
    /// Creates a double Weibull distribution with shape `c = shape`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            magnitude: WeibullMin::new(shape)?,
        })
    }
}
impl<S: RealScalar> Formula<S> for DoubleWeibull<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn center(&self) -> Option<S> {
        Some(S::ZERO)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.magnitude.density_log(x.abs())? - S::LN_2)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        symmetric::log_tail(x, upper, |y| self.magnitude.tail_with_work(y, true, left))
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        symmetric::inverse(p, false, |tail| self.magnitude.inverse_upper(tail))
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        symmetric::inverse(p, true, |tail| self.magnitude.inverse_upper(tail))
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        symmetric::draw(rng, |rng| self.magnitude.draw_with_remaining(rng, left))
    }
}
formula_law!(DoubleWeibull);
impl<S: RealScalar> Mean<S> for DoubleWeibull<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for DoubleWeibull<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        scioxide_special::gamma(S::ONE + number::<S>(2.)? / self.magnitude.shape)
    }
}

/// Inverse Weibull (Fréchet) distribution on `x > 0` with shape `c > 0` and unit scale.
///
/// `F(x) = exp(−x^(−c))`. The moment of order `r` exists for `r < c`. Shift and
/// scale the law with [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{Cdf, Distribution, InverseWeibull, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support};
/// let law = InverseWeibull::new(3.0_f64)?;
/// assert!((law.cdf(1.0)? - (-1.0_f64).exp()).abs() < 1e-16);
/// assert!((law.mean()? - 1.354_117_939_426_400_4).abs() < 1e-14); // Γ(1 - 1/c)
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(35);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct InverseWeibull<S: RealScalar> {
    c: S,
}
impl<S: RealScalar> InverseWeibull<S> {
    /// Creates an inverse Weibull distribution with shape `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` is NaN or infinite.
    /// - `DomainError` if `c ≤ 0`.
    pub fn new(c: S) -> Result<Self, SciOxideError> {
        positive(c, scioxide_core::argument!("c"))?;
        Ok(Self { c })
    }
    fn raw_moment(&self, r: S) -> Result<S, SciOxideError> {
        if r >= self.c {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            scioxide_special::gamma(S::ONE - r / self.c)
        }
    }
}
impl<S: RealScalar> Formula<S> for InverseWeibull<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            Ok(S::NEG_INFINITY)
        } else {
            Ok(self.c.ln() - (self.c + S::ONE) * x.ln() - (-self.c * x.ln()).exp())
        }
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            let log_t = -self.c * x.ln();
            if log_t < S::EPSILON.ln() {
                Ok(log_t)
            } else {
                log1mexp(-log_t.exp())
            }
        } else {
            if x <= S::ZERO {
                Ok(S::NEG_INFINITY)
            } else {
                Ok(-(-self.c * x.ln()).exp())
            }
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            Ok(S::ZERO)
        } else if p == S::ONE {
            Ok(S::INFINITY)
        } else {
            finite((-(-p.ln()).ln() / self.c).exp())
        }
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            Ok(S::ZERO)
        } else if p == S::ZERO {
            Ok(S::INFINITY)
        } else {
            finite((-(-(-p).ln_1p()).ln() / self.c).exp())
        }
    }
}
formula_law!(InverseWeibull);
impl<S: RealScalar> Mean<S> for InverseWeibull<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(S::ONE)
    }
}
impl<S: RealScalar> Variance<S> for InverseWeibull<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        super::moment_math::gamma_variance(1.0, -self.c.into_f64().recip())
    }
}

/// Exponentiated Weibull distribution on `x ≥ 0` with exponent `a > 0` and shape `c > 0`.
///
/// `F(x) = (1 − exp(−x^c))^a`. The law has unit scale; scale it with
/// [`Affine`](crate::Affine). The quantile and inverse survival function are
/// closed-form, evaluated from logarithmic probabilities. [`Mean`](crate::Mean)
/// and [`Variance`](crate::Variance) are not implemented.
///
/// # References
///
/// G. S. Mudholkar and D. K. Srivastava, "Exponentiated Weibull family for
/// analyzing bathtub failure-rate data", *IEEE Trans. Reliab.* 42(2), 1993,
/// 299–302.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, ExponentiatedWeibull, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support};
/// let law = ExponentiatedWeibull::new(2.0_f64, 1.0)?;
/// assert!((law.cdf(2.0_f64.ln())? - 0.25).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(36);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExponentiatedWeibull<S: RealScalar> {
    a: S,
    c: S,
}
impl<S: RealScalar> ExponentiatedWeibull<S> {
    // ln(-ln(F(x))), with q = x^c. When q is tiny, ln(1-exp(-q)) = ln(q)
    // to working precision; combine a*c before materializing either power.
    fn log_rate(&self, log_x: f64) -> Result<f64, SciOxideError> {
        let log_q = self.c.into_f64() * log_x;
        Ok(self.a.into_f64().ln()
            + if log_q < f64::EPSILON.ln() {
                self.c.into_f64().ln() + (-log_x).ln()
            } else {
                log_negative_log_complement(log_q)?
            })
    }

    fn inverse_log_cdf(&self, log_p: f64) -> Result<S, SciOxideError> {
        let log_rate = (-log_p).ln() - self.a.into_f64().ln();
        let log_x = if log_rate > (-f64::EPSILON.ln()).ln() {
            // ln(-ln(1-exp(-rate))) = -rate to relative epsilon.
            -(log_rate - self.c.into_f64().ln()).exp()
        } else {
            log_negative_log_complement(log_rate)? / self.c.into_f64()
        };
        crate::law::positive_draw(log_x)
    }

    /// Creates an exponentiated Weibull distribution with exponent `a` and shape `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` or `c` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0` or `c ≤ 0`.
    pub fn new(a: S, c: S) -> Result<Self, SciOxideError> {
        positive(a, scioxide_core::argument!("a"))?;
        positive(c, scioxide_core::argument!("c"))?;
        Ok(Self { a, c })
    }
}
impl<S: RealScalar> Formula<S> for ExponentiatedWeibull<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            let power = self.a * self.c;
            return Ok(if power < S::ONE {
                S::INFINITY
            } else if power > S::ONE {
                S::NEG_INFINITY
            } else {
                self.a.ln() + self.c.ln()
            });
        }
        let log_x = x.into_f64().ln();
        let log_q = self.c.into_f64() * log_x;
        let q = log_q.exp();
        if q.is_infinite() {
            return Ok(S::NEG_INFINITY);
        }
        let correction = if log_q < f64::EPSILON.ln() {
            0.0
        } else {
            log_q - log1mexp(-q)?
        };
        number(
            self.a.into_f64().ln() + self.c.into_f64().ln() - log_x + correction
                - q
                - self.log_rate(log_x)?.exp(),
        )
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        let log_rate = self.log_rate(x.into_f64().ln())?;
        number(if upper {
            if log_rate < f64::EPSILON.ln() {
                log_rate
            } else {
                log1mexp(-log_rate.exp())?
            }
        } else {
            -log_rate.exp()
        })
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        self.inverse_log_cdf(p.into_f64().ln())
    }

    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        self.inverse_log_cdf((-p.into_f64()).ln_1p())
    }
}
formula_law!(ExponentiatedWeibull);

/// ln(-ln(1-exp(-exp(log_t)))); the limiting branches avoid losing either tail.
fn log_negative_log_complement(log_t: f64) -> Result<f64, SciOxideError> {
    Ok(if log_t < f64::EPSILON.ln() {
        (-log_t).ln()
    } else if log_t > (-f64::EPSILON.ln()).ln() {
        -log_t.exp()
    } else {
        (-log1mexp(-log_t.exp())?).ln()
    })
}
