//! Gumbel, generalized extreme value, kappa and Gompertz distributions.
use crate::distribution::LawOptions;
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::distribution::moment_math::quantile_abs_moment;
use crate::law::constant as number;
use crate::law::{argument, finite, log1mexp};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Standard Gumbel distribution (maximum extreme value type I).
///
/// `F(x) = exp(−exp(−x))` on the real line. Shift and scale it with
/// [`Affine`](crate::Affine). Scale `−1` gives the left-skewed Gumbel law of
/// minima, with `F(x) = 1 − exp(−exp(x))`.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, GumbelRight, Mean, Quantile, SeedableRng, SplitMix64,
/// };
/// let law = GumbelRight::<f64>::new();
/// let x = law.quantile(0.3)?;
/// assert!((law.cdf(x)? - 0.3).abs() < 1e-14);
/// assert!((law.mean()? - 0.577_215_664_901_532_9).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(11);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GumbelRight<S: RealScalar>(core::marker::PhantomData<S>);
impl<S: RealScalar> GumbelRight<S> {
    /// Creates the standard Gumbel distribution.
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}
impl<S: RealScalar> Default for GumbelRight<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> Formula<S> for GumbelRight<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(-x - (-x).exp())
    }
    // exp(-e^-x) and its complement -expm1(-e^-x) directly, without the
    // logarithmic tail round trip.
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let t = (-x).exp();
        Ok(if upper { -(-t).exp_m1() } else { (-t).exp() })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            // 1 - exp(-e^-x) ~ e^-x once e^-x < eps; the leading term avoids
            // complementing a CDF that has rounded to one.
            if x > -S::EPSILON.ln() {
                Ok(-x)
            } else {
                log1mexp(self.tail_with_work(x, false, _left)?)
            }
        } else {
            Ok(-(-x).exp())
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        finite(-(-p.ln()).ln())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        finite(-(-(-p).ln_1p()).ln())
    }
    // Inversion of an open uniform; both logarithms stay finite.

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        Ok(-(-crate::util::validated_open_float(rng)?.ln()).ln())
    }
}
formula_law!(GumbelRight);
impl<S: RealScalar> Mean<S> for GumbelRight<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(0.5772156649015329)
    }
}
impl<S: RealScalar> Variance<S> for GumbelRight<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let p = S::PI;
        finite(p * p / number::<S>(6.)?)
    }
}

/// Generalized extreme value distribution with shape `c`, zero location and unit scale.
///
/// ```text
/// F(x) = exp(−(1 − cx)^(1/c)),   1 − cx > 0,
/// ```
///
/// and `F(x) = exp(−exp(−x))` (the Gumbel law) for `c = 0`. For `c > 0` the
/// support is bounded above at `1/c`, and for `c < 0` it is bounded below at
/// `1/c`. With `ξ = −c` this is the other common form,
/// `F(x) = exp(−(1 + ξx)^(−1/ξ))`. Shift and scale the law with
/// [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, GeneralizedExtremeValue, Quantile, SeedableRng, SplitMix64,
///     Support,
/// };
/// let law = GeneralizedExtremeValue::<f64>::new(0.5)?;
/// assert!(law.contains(2.0)? && !law.contains(2.5)?);
/// let x = law.quantile(0.9)?;
/// assert!((law.cdf(x)? - 0.9).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(12);
/// assert!(law.sample(&mut rng)? <= 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedExtremeValue<S: RealScalar> {
    c: S,
}
impl<S: RealScalar> GeneralizedExtremeValue<S> {
    /// Creates a generalized extreme value distribution with shape `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if `c ≠ 0` and the support end `1/c` overflows.
    pub fn new(c: S) -> Result<Self, SciOxideError> {
        argument(c, "c")?;
        if c != S::ZERO {
            finite(S::ONE / c)?;
        }
        Ok(Self { c })
    }
    fn log_base(&self, x: S) -> S {
        let y = -self.c * x;
        if y.is_finite() {
            y.ln_1p()
        } else {
            self.c.abs().ln() + x.abs().ln()
        }
    }
    fn log_t(&self, x: S) -> S {
        if self.c == S::ZERO {
            -x
        } else {
            self.log_base(x) / self.c
        }
    }
    fn t(&self, x: S) -> S {
        self.log_t(x).exp()
    }
    fn inverse_log_t(&self, l: S) -> S {
        -crate::law::expm1_ratio(self.c, l)
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedExtremeValue<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        if self.c > S::ZERO {
            (None, Some(S::ONE / self.c))
        } else if self.c < S::ZERO {
            (Some(S::ONE / self.c), None)
        } else {
            (None, None)
        }
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let z = S::ONE - self.c * x;
        if self.c > S::ZERO && z <= S::ZERO {
            return Ok(if z == S::ZERO && self.c > S::ONE {
                S::INFINITY
            } else if z == S::ZERO && self.c == S::ONE {
                S::ZERO
            } else {
                S::NEG_INFINITY
            });
        }
        if self.c < S::ZERO && z <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if self.c == S::ZERO {
            return Ok(-x - (-x).exp());
        }
        Ok(self.log_t(x) - self.log_base(x) - self.t(x))
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            let z = S::ONE - self.c * x;
            if self.c > S::ZERO && z <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            if self.c < S::ZERO && z <= S::ZERO {
                return Ok(S::ZERO);
            }
            let log_t = self.log_t(x);
            if log_t < S::EPSILON.ln() {
                Ok(log_t)
            } else {
                log1mexp(-log_t.exp())
            }
        } else {
            let z = S::ONE - self.c * x;
            if self.c > S::ZERO && z <= S::ZERO {
                return Ok(S::ZERO);
            }
            if self.c < S::ZERO && z <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            Ok(-self.t(x))
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(self.bounds().0.unwrap_or(S::NEG_INFINITY));
        }
        if p == S::ONE {
            return Ok(self.bounds().1.unwrap_or(S::INFINITY));
        }
        let l = (-p.ln()).ln();
        finite(if self.c == S::ZERO {
            -l
        } else {
            self.inverse_log_t(l)
        })
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            return Ok(self.bounds().0.unwrap_or(S::NEG_INFINITY));
        }
        if p == S::ZERO {
            return Ok(self.bounds().1.unwrap_or(S::INFINITY));
        }
        let l = (-(-p).ln_1p()).ln();
        finite(if self.c == S::ZERO {
            -l
        } else {
            self.inverse_log_t(l)
        })
    }
}
formula_law!(GeneralizedExtremeValue);
impl<S: RealScalar> Mean<S> for GeneralizedExtremeValue<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.c <= -S::ONE {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        if self.c.abs() < S::SQRT_EPSILON {
            let euler = 0.5772156649015329_f64;
            return number(
                euler
                    - (0.5 * euler * euler + core::f64::consts::PI.powi(2) / 12.0)
                        * self.c.into_f64(),
            );
        }
        number(finite(
            -super::special::log_gamma_ratio(1.0, self.c.into_f64())?.exp_m1() / self.c.into_f64(),
        )?)
    }
}
impl<S: RealScalar> Variance<S> for GeneralizedExtremeValue<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.c <= -S::HALF {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        if self.c == S::ZERO {
            let p = number::<S>(core::f64::consts::PI)?;
            return finite(p * p / number::<S>(6.)?);
        }
        let c = self.c.into_f64();
        super::moment_math::variance(
            super::moment_math::log_gamma_variance::<S>(1.0, c)? - 2.0 * c.abs().ln(),
        )
    }
}

/// Kappa distribution of Hosking with shapes `h` and `k`, zero location and unit scale.
///
/// ```text
/// F(x) = (1 − h (1 − kx)^(1/k))^(1/h),
/// ```
///
/// with the limits `(1 − kx)^(1/k) → exp(−x)` as `k → 0` and
/// `(1 − h t)^(1/h) → exp(−t)` as `h → 0`. It contains the generalized
/// extreme value (`h = 0`), generalized logistic (`h = −1`) and generalized
/// Pareto (`h = 1`) laws. The mean and variance are computed by quadrature of
/// the quantile function, within the [work limits](crate#work-limits).
///
/// # References
///
/// J. R. M. Hosking, "The four-parameter kappa distribution", *IBM J. Res.
/// Develop.* 38(3), 1994, 251–258.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Kappa4, Mean, SeedableRng, SplitMix64};
/// // h = 1, k = 0 is the standard exponential law.
/// let law = Kappa4::<f64>::new(1.0, 0.0)?;
/// assert!((law.cdf(1.0)? - (1.0 - (-1.0_f64).exp())).abs() < 1e-14);
/// assert!((law.mean()? - 1.0).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(13);
/// assert!(law.sample(&mut rng)? >= 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Kappa4<S: RealScalar> {
    options: LawOptions<S>,
    h: S,
    k: S,
}
impl<S: RealScalar> Kappa4<S> {
    /// Creates a kappa distribution with shapes `h` and `k`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `h` or `k` is NaN or infinite.
    /// - `NumericalInstability` if a finite support endpoint cannot be represented.
    pub fn new(h: S, k: S) -> Result<Self, SciOxideError> {
        Self::with_options(h, k, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(h: S, k: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(h, "h")?;
        argument(k, "k")?;
        let law = Self { options, h, k };
        let (low, high) = law.bounds();
        for endpoint in [low, high].into_iter().flatten() {
            finite(endpoint)?;
        }
        Ok(law)
    }
    fn moment_exists(&self, order: u32) -> Result<(), SciOxideError> {
        let r = number::<S>(order as f64)?;
        if self.k < S::ZERO && r >= -S::ONE / self.k
            || self.h < S::ZERO && self.k > S::ZERO && r >= -S::ONE / (self.h * self.k)
        {
            return Err(crate::law::undefined_moment("Kappa4 moment does not exist"));
        }
        Ok(())
    }
    fn raw_moment(&self, order: u32, left: &mut usize) -> Result<S, SciOxideError> {
        self.moment_exists(order)?;
        // Split the quantile integral at F(0) so each piece integrates |Q|^r
        // from a probability endpoint with its own sign, then recombine with
        // the parity of the order.
        let split = self
            .tail_with_work(S::ZERO, false, &mut self.work_budget().get())?
            .exp();
        let negative = quantile_abs_moment(self, order, S::ZERO, split, self.options, left)?;
        let positive = quantile_abs_moment(self, order, split, S::ONE, self.options, left)?;
        finite(if order.is_multiple_of(2) {
            positive + negative
        } else {
            positive - negative
        })
    }

    fn endpoint(&self) -> S {
        -crate::law::expm1_ratio(self.k, -self.h.ln())
    }
    // Hosking's four-parameter kappa: F(x) = (1 - h t(x))^(1/h) with
    // t(x) = (1 - k x)^(1/k). The k = 0 and h = 0 cases are the analytic
    // limits (exponential and Gumbel-type), and `ln_1p`/`expm1` keep the
    // expressions accurate as h or k approach those limits.
    fn t(&self, x: S) -> S {
        if self.k == S::ZERO {
            (-x).exp()
        } else {
            ((-self.k * x).ln_1p() / self.k).exp()
        }
    }
    fn log_cdf_inside(&self, x: S) -> S {
        let t = self.t(x);
        if self.h == S::ZERO {
            -t
        } else {
            (-self.h * t).ln_1p() / self.h
        }
    }
    fn quantile_logp(&self, l: S) -> Result<S, SciOxideError> {
        let log_t = if self.h == S::ZERO {
            (-l).ln()
        } else if self.h < S::ZERO {
            crate::law::log_expm1(self.h * l)? - (-self.h).ln()
        } else {
            log1mexp(self.h * l)? - self.h.ln()
        };
        finite(-crate::law::expm1_ratio(self.k, log_t))
    }
}
impl<S: RealScalar> Formula<S> for Kappa4<S> {
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
        let low = if self.h > S::ZERO {
            Some(self.endpoint())
        } else if self.k < S::ZERO {
            Some(S::ONE / self.k)
        } else {
            None
        };
        let high = if self.k > S::ZERO {
            Some(S::ONE / self.k)
        } else {
            None
        };
        (low, high)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let (low, high) = self.bounds();
        if low.is_some_and(|a| x < a) || high.is_some_and(|b| x > b) {
            return Ok(S::NEG_INFINITY);
        }
        if high == Some(x) {
            return Ok(if self.k < S::ONE {
                S::NEG_INFINITY
            } else if self.k == S::ONE {
                S::ZERO
            } else {
                S::INFINITY
            });
        }
        if low == Some(x) {
            if self.h > S::ZERO {
                return Ok(if self.h < S::ONE {
                    S::NEG_INFINITY
                } else if self.h > S::ONE {
                    S::INFINITY
                } else if self.k == S::ZERO {
                    -x
                } else {
                    (S::ONE / self.k - S::ONE) * (-self.k * x).ln_1p()
                });
            }
            if self.h == S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            let power = S::ONE / (self.h * self.k) - S::ONE;
            return Ok(if power > S::ZERO {
                S::NEG_INFINITY
            } else if power < S::ZERO {
                S::INFINITY
            } else {
                (S::ONE / self.h - S::ONE) * (-self.h).ln()
            });
        }
        let t = self.t(x);
        let lead = if self.k == S::ZERO {
            -x
        } else {
            (S::ONE / self.k - S::ONE) * (-self.k * x).ln_1p()
        };
        Ok(lead
            + if self.h == S::ZERO {
                -t
            } else if self.h == S::ONE {
                S::ZERO
            } else {
                (S::ONE / self.h - S::ONE) * (-self.h * t).ln_1p()
            })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            let (low, high) = self.bounds();
            if low.is_some_and(|a| x <= a) {
                return Ok(S::ZERO);
            }
            if high.is_some_and(|b| x >= b) {
                return Ok(S::NEG_INFINITY);
            }
            let log_t = if self.k == S::ZERO {
                -x
            } else {
                (-self.k * x).ln_1p() / self.k
            };
            // Both t and |h|*t must be small for the first-order tail.
            if log_t < S::EPSILON.ln()
                && (self.h == S::ZERO || self.h.abs().ln() + log_t < S::EPSILON.ln())
            {
                Ok(log_t)
            } else {
                log1mexp(self.tail_with_work(x, false, _left)?)
            }
        } else {
            let (low, high) = self.bounds();
            Ok(if low.is_some_and(|a| x <= a) {
                S::NEG_INFINITY
            } else if high.is_some_and(|b| x >= b) {
                S::ZERO
            } else {
                self.log_cdf_inside(x)
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        let (low, high) = self.bounds();
        if p == S::ZERO {
            return Ok(low.unwrap_or(S::NEG_INFINITY));
        }
        if p == S::ONE {
            return Ok(high.unwrap_or(S::INFINITY));
        }
        self.quantile_logp(p.ln())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        let (low, high) = self.bounds();
        if p == S::ZERO {
            return Ok(high.unwrap_or(S::INFINITY));
        }
        if p == S::ONE {
            return Ok(low.unwrap_or(S::NEG_INFINITY));
        }
        // Both inverses share the log-probability form, so an upper-tail
        // request is converted with ln(1 - p) computed by `ln_1p`.
        self.quantile_logp((-p).ln_1p())
    }
}
formula_law!(Kappa4);
impl<S: RealScalar> Mean<S> for Kappa4<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.mean_work(&mut self.options.work_budget().get())
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.mean_work(left)
        })
    }
}
impl<S: RealScalar> Kappa4<S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        self.raw_moment(1, left)
    }
}

impl<S: RealScalar> Variance<S> for Kappa4<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        self.variance_work(&mut self.options.work_budget().get())
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.variance_work(left)
        })
    }
}
impl<S: RealScalar> Kappa4<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let mean = self.raw_moment(1, left)?;
        crate::distribution::moment_math::variance_from_moments(self.raw_moment(2, left)?, mean)
    }
}

mod gompertz;
pub use gompertz::Gompertz;
