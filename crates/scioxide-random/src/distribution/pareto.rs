//! Pareto, generalized Pareto and truncated Pareto distributions.
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{argument, finite, log1mexp, logadd};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

fn log1p_positive_product<S: RealScalar>(a: S, x: S) -> S {
    let y = a * x;
    if y.is_finite() {
        y.ln_1p()
    } else {
        // The correction log(1+1/(a*x)) is below rounding once a*x overflows.
        a.ln() + x.ln()
    }
}

/// Pareto type I distribution on `x ≥ 1` with shape `b > 0`.
///
/// The survival function is `S(x) = x^(−b)`. A draw is `exp(E/b)` for a
/// standard exponential `E`. The mean exists for `b > 1` and the variance for
/// `b > 2`. The law has unit scale; shift and scale it with
/// [`Affine`](crate::Affine). For example, `Affine::new(law, -1.0, 1.0)` is the
/// Lomax (Pareto type II) law on `x ≥ 0`.
///
/// ```
/// use scioxide_random::{Affine, Distribution, Mean, Pareto, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Survival, Variance};
/// let law = Pareto::new(3.0_f64)?;
/// assert!((law.sf(2.0)? - 0.125).abs() < 1e-16);
/// assert!((law.mean()? - 1.5).abs() < 1e-15 && (law.variance()? - 0.75).abs() < 1e-15);
/// assert!((law.sf(law.quantile(0.3)?)? - 0.7).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(30);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// let lomax = Affine::new(law, -1.0, 1.0)?;
/// assert!((lomax.mean()? - 0.5).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Pareto<S: RealScalar> {
    b: S,
}
impl<S: RealScalar> Pareto<S> {
    /// Creates a Pareto distribution with shape `b`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `b` is NaN or infinite.
    /// - `DomainError` if `b ≤ 0`.
    pub fn new(b: S) -> Result<Self, SciOxideError> {
        positive(b, scioxide_core::argument!("b"))?;
        Ok(Self { b })
    }
}
impl<S: RealScalar> Formula<S> for Pareto<S> {
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let log = self.tail_with_work(x, true, &mut self.work_budget().get())?;
        // Subtraction also keeps the exact lower-support limit at +0.
        Ok(if upper {
            log.exp()
        } else {
            S::ZERO - log.exp_m1()
        })
    }
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ONE), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ONE {
            Ok(S::NEG_INFINITY)
        } else {
            Ok(self.b.ln() - (self.b + S::ONE) * x.ln())
        }
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ONE {
                Ok(S::ZERO)
            } else {
                Ok(-self.b * x.ln())
            }
        } else {
            if x <= S::ONE {
                Ok(S::NEG_INFINITY)
            } else {
                log1mexp(self.tail_with_work(x, true, _left)?)
            }
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            Ok(S::INFINITY)
        } else {
            finite((-(-p).ln_1p() / self.b).exp())
        }
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            Ok(S::INFINITY)
        } else {
            finite((-p.ln() / self.b).exp())
        }
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        Ok((super::exponential::standard_exponential(rng)? / self.b).exp())
    }
}
formula_law!(Pareto);
impl<S: RealScalar> Mean<S> for Pareto<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.b <= S::ONE {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            finite(self.b / (self.b - S::ONE))
        }
    }
}
impl<S: RealScalar> Variance<S> for Pareto<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.b <= number::<S>(2.)? {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            let b = self.b.into_f64();
            super::moment_math::variance(b.ln() - 2.0 * (b - 1.0).ln() - (b - 2.0).ln())
        }
    }
}

/// Generalized Pareto distribution with shape `c`, zero location and unit scale.
///
/// The survival function is `S(x) = (1 + cx)^(−1/c)` on `x ≥ 0`, and
/// `S(x) = exp(−x)` for `c = 0`. For `c < 0` the support is `[0, −1/c]`. The
/// mean exists for `c < 1` and the variance for `c < 1/2`. Shift and scale the
/// law with [`Affine`](crate::Affine).
///
/// # References
///
/// J. Pickands III, "Statistical inference using extreme order statistics",
/// *Ann. Statist.* 3(1), 1975, 119–131.
///
/// ```
/// use scioxide_random::{Distribution, GeneralizedPareto, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Survival, Variance};
/// let law = GeneralizedPareto::new(0.25_f64)?;
/// assert!((law.sf(1.0)? - 1.25_f64.powi(-4)).abs() < 1e-15);
/// assert!((law.mean()? - 4.0 / 3.0).abs() < 1e-15);
/// assert!((law.variance()? - 1.0 / (0.75 * 0.75 * 0.5)).abs() < 1e-14);
/// assert!((law.sf(law.quantile(0.3)?)? - 0.7).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(31);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedPareto<S: RealScalar> {
    c: S,
}
impl<S: RealScalar> GeneralizedPareto<S> {
    /// Creates a generalized Pareto distribution with shape `c`.
    ///
    /// `c = 0` gives the standard exponential law.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if `c < 0` and the upper support end `−1/c` overflows.
    pub fn new(c: S) -> Result<Self, SciOxideError> {
        argument(c, "c")?;
        if c < S::ZERO {
            finite(-S::ONE / c)?;
        }
        Ok(Self { c })
    }
    fn exponent(&self, x: S) -> S {
        let y = self.c * x;
        if y == S::ZERO {
            x
        } else if y.is_finite() {
            (y.ln_1p() / y) * x
        } else {
            log1p_positive_product(self.c, x) / self.c
        }
    }
    fn inverse_exponent(&self, t: S) -> S {
        crate::law::expm1_ratio(self.c, t)
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedPareto<S> {
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let log = self.tail_with_work(x, true, &mut self.work_budget().get())?;
        Ok(if upper { log.exp() } else { -log.exp_m1() })
    }
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (
            Some(S::ZERO),
            if self.c < S::ZERO {
                Some(-S::ONE / self.c)
            } else {
                None
            },
        )
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO || (self.c < S::ZERO && x > -S::ONE / self.c) {
            return Ok(S::NEG_INFINITY);
        }
        if self.c == S::ZERO {
            return Ok(-x);
        }
        if self.c < S::ZERO && x == -S::ONE / self.c {
            return Ok(if self.c < -S::ONE {
                S::INFINITY
            } else if self.c == -S::ONE {
                S::ZERO
            } else {
                S::NEG_INFINITY
            });
        }
        Ok(-(S::ONE + self.c) * self.exponent(x))
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            if self.c < S::ZERO && x >= -S::ONE / self.c {
                return Ok(S::NEG_INFINITY);
            }
            Ok(if self.c == S::ZERO {
                -x
            } else {
                -self.exponent(x)
            })
        } else {
            if x <= S::ZERO {
                Ok(S::NEG_INFINITY)
            } else {
                log1mexp(self.tail_with_work(x, true, _left)?)
            }
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(self.bounds().1.unwrap_or(S::INFINITY));
        }
        finite(if self.c == S::ZERO {
            -(-p).ln_1p()
        } else {
            self.inverse_exponent(-(-p).ln_1p())
        })
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        if p == S::ZERO {
            return Ok(self.bounds().1.unwrap_or(S::INFINITY));
        }
        finite(if self.c == S::ZERO {
            -p.ln()
        } else {
            self.inverse_exponent(-p.ln())
        })
    }
}
formula_law!(GeneralizedPareto);
impl<S: RealScalar> Mean<S> for GeneralizedPareto<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.c >= S::ONE {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            finite(S::ONE / (S::ONE - self.c))
        }
    }
}
impl<S: RealScalar> Variance<S> for GeneralizedPareto<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.c >= S::HALF {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            let c = self.c.into_f64();
            // ln(1-2c) = ln(2) + ln(1/2-c) avoids overflowing 2c.
            super::moment_math::variance(
                -2.0 * (-c).ln_1p() - core::f64::consts::LN_2 - (0.5 - c).ln(),
            )
        }
    }
}

/// Truncated Pareto distribution on `[1, c]` with power `b ≠ 0` and upper end `c > 1`.
///
/// The density is `f(x) ∝ x^(−b−1)` on `[1, c]`, and `b` may be negative. The
/// law has unit scale; scale it with [`Affine`](crate::Affine).
/// [`Mean`](crate::Mean) and [`Variance`](crate::Variance) are not implemented.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Quantile, SeedableRng, SplitMix64};
/// use scioxide_random::{Support, TruncatedPareto};
/// let law = TruncatedPareto::new(2.0_f64, 2.0)?;
/// // F(x) = (1 - x^-2) / (1 - 2^-2)
/// assert!((law.cdf(2.0_f64.sqrt())? - 2.0 / 3.0).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(32);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TruncatedPareto<S: RealScalar> {
    b: S,
    c: S,
    log_c: S,
    log_norm: S,
}
impl<S: RealScalar> TruncatedPareto<S> {
    /// Creates a truncated Pareto distribution with power `b` and upper end `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `b` or `c` is NaN or infinite.
    /// - `DomainError` if `b = 0` or `c ≤ 1`.
    /// - `NumericalInstability` (`Overflow`) if `|b| ln c` overflows.
    pub fn new(b: S, c: S) -> Result<Self, SciOxideError> {
        crate::law::argument(b, "b")?;
        crate::law::argument(c, "c")?;
        if b == S::ZERO || c <= S::ONE {
            return Err(crate::law::domain(
                if b == S::ZERO {
                    scioxide_core::argument!("b")
                } else {
                    scioxide_core::argument!("c")
                },
                "truncated Pareto requires b != 0 and c > 1",
            ));
        }
        let log_c = c.ln();
        let width = b.abs() * log_c;
        if !width.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "truncated Pareto normalization is not representable",
            ));
        }
        let log_norm = if b > S::ZERO {
            log1mexp(-width)?
        } else {
            width + log1mexp(-width)?
        };
        Ok(Self {
            b,
            c,
            log_c,
            log_norm,
        })
    }
}
impl<S: RealScalar> Formula<S> for TruncatedPareto<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ONE), Some(self.c))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ONE || x > self.c {
            return Ok(S::NEG_INFINITY);
        }
        Ok(self.b.abs().ln() - (self.b + S::ONE) * x.ln() - self.log_norm)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ONE {
                return Ok(S::ZERO);
            }
            if x >= self.c {
                return Ok(S::NEG_INFINITY);
            }
            let delta = self.log_c - x.ln();
            Ok(if self.b > S::ZERO {
                -self.b * x.ln() + log1mexp(-self.b * delta)? - self.log_norm
            } else {
                -self.b * self.log_c + log1mexp(self.b * delta)? - self.log_norm
            })
        } else {
            if x <= S::ONE {
                return Ok(S::NEG_INFINITY);
            }
            if x >= self.c {
                return Ok(S::ZERO);
            }
            let y = x.ln();
            Ok(if self.b > S::ZERO {
                log1mexp(-self.b * y)? - self.log_norm
            } else {
                -self.b * y + log1mexp(self.b * y)? - self.log_norm
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ONE);
        }
        if p == S::ONE {
            return Ok(self.c);
        }
        let y = if self.b > S::ZERO {
            (p * (-self.b * self.log_c).exp_m1()).ln_1p() / -self.b
        } else {
            let k = -self.b;
            logadd((-p).ln_1p(), p.ln() + k * self.log_c)? / k
        };
        finite(y.exp())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(self.c);
        }
        if p == S::ONE {
            return Ok(S::ONE);
        }
        let y = if self.b > S::ZERO {
            logadd(p.ln(), (-p).ln_1p() - self.b * self.log_c)? / -self.b
        } else {
            let k = -self.b;
            logadd((-p).ln_1p() + k * self.log_c, p.ln())? / k
        };
        finite(y.exp())
    }
}
formula_law!(TruncatedPareto);
