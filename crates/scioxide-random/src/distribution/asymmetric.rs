//! Skewed distributions in standard form.
use crate::distribution::LawOptions;
use crate::distribution::beta::BetaKernel;
use crate::distribution::formula::{Formula, Inverse, formula_law};
use crate::distribution::normal::standard_normal;
use crate::law::{argument, constant as c, finite, log1mexp, logadd, probability};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Crystal Ball distribution with join point `−β` (`β > 0`) and tail exponent `m > 1`.
///
/// The density is a standard normal core for `x > −β`, joined continuously
/// with a continuous derivative to a power-law left tail:
///
/// ```text
/// f(x) ∝ exp(−x²/2)                           for x > −β,
/// f(x) ∝ (m/β)^m exp(−β²/2) (m/β − β − x)^(−m)   for x ≤ −β.
/// ```
///
/// The moment of order `k` exists only for `m > k + 1`.
///
/// ```
/// use scioxide_random::{
///     Cdf, CrystalBall, Distribution, Mean, Quantile, SeedableRng, SplitMix64,
/// };
/// let law = CrystalBall::<f64>::new(2.0, 3.0)?;
/// let x = law.quantile(0.01)?;
/// assert!(x < -2.0 && (law.cdf(x)? - 0.01).abs() < 1e-14);
/// assert!(law.mean()?.is_finite());
/// let mut rng = SplitMix64::from_seed(41);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct CrystalBall<S> {
    beta: S,
    m: S,
    log_tail: S,
    log_norm: S,
}
impl<S: RealScalar> CrystalBall<S> {
    /// Creates a Crystal Ball distribution with join point `−beta` and exponent `m`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `beta` or `m` is NaN or infinite.
    /// - `DomainError` if `beta ≤ 0` or `m ≤ 1`.
    /// - `NumericalInstability` if the normalizing constant is outside the range of `S`.
    pub fn new(beta: S, m: S) -> Result<Self, SciOxideError> {
        argument(beta, "beta")?;
        argument(m, "m")?;
        if beta <= S::ZERO || m <= S::ONE {
            return Err(crate::law::domain(
                if beta <= S::ZERO {
                    scioxide_core::argument!("beta")
                } else {
                    scioxide_core::argument!("m")
                },
                "CrystalBall requires beta > 0 and m > 1",
            ));
        }
        // Unnormalized mass split at the join x = -beta: the power-law tail
        // integrates to (m / beta) / (m - 1) e^{-beta^2/2} and the Gaussian core
        // to sqrt(2 pi) Phi(beta). Both are kept as logs so the normalizer is
        // safe for large beta or m close to 1.
        let log_tail = m.ln() - beta.ln() - (m - S::ONE).ln() - beta * beta * S::HALF;
        let log_core = crate::distribution::special::log_two_pi::<S>() * S::HALF
            + scioxide_special::normal_log_cdf(beta)?;
        let log_norm = finite(logadd(log_tail, log_core)?)?;
        Ok(Self {
            beta,
            m,
            log_tail,
            log_norm,
        })
    }
    fn log_join_ratio(&self, x: S) -> Result<S, SciOxideError> {
        let d = -x - self.beta;
        let ratio = (d / self.m) * self.beta;
        Ok(if ratio.is_finite() {
            ratio.ln_1p()
        } else {
            logadd(S::ZERO, d.ln() + self.beta.ln() - self.m.ln())?
        })
    }
    fn tail(&self, x: S) -> Result<S, SciOxideError> {
        // Ratio to the join avoids enormous powers and cancellation in m/beta-beta-x.
        Ok(self.log_tail - self.log_norm + (S::ONE - self.m) * self.log_join_ratio(x)?)
    }
    fn invert(&self, p: S, upper: bool) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                S::NEG_INFINITY
            } else {
                S::INFINITY
            });
        }
        // Invert piecewise: probabilities below the tail mass come from the
        // closed-form power-law CDF, the rest from the Gaussian core via its
        // log survival function so upper quantiles keep their precision.
        let lp = if upper { (-p).ln_1p() } else { p.ln() };
        if lp <= self.log_tail - self.log_norm {
            finite(
                -self.beta
                    - self.m / self.beta
                        * ((lp - self.log_tail + self.log_norm) / (S::ONE - self.m)).exp_m1(),
            )
        } else {
            let ls = if upper { p.ln() } else { (-p).ln_1p() };
            scioxide_special::normal_inv_log_sf(
                ls + self.log_norm - crate::distribution::special::log_two_pi::<S>() * S::HALF,
            )
        }
    }
    fn raw_moment(&self, order: usize) -> Result<S, SciOxideError> {
        let k = S::try_from_usize_exact(order)?;
        if self.m <= k + S::ONE {
            return Err(crate::law::undefined_moment(
                "CrystalBall moment does not exist",
            ));
        }
        let r = self.m / self.beta;
        let e = (-self.beta * self.beta * S::HALF - self.log_norm).exp();
        let tail0 = (self.log_tail - self.log_norm).exp();
        let t1 = r / (self.m - c(2.)?);
        if order == 1 {
            return finite(-tail0 * (self.beta + t1) + e);
        }
        let t2 = c::<S>(2.)? * r / (self.m - c(2.)?) * r / (self.m - c(3.)?);
        finite(
            tail0 * (self.beta * self.beta + c::<S>(2.)? * self.beta * t1 + t2) + (S::ONE - tail0)
                - self.beta * e,
        )
    }
}
impl<S: RealScalar> Formula<S> for CrystalBall<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(if x <= -self.beta {
            -self.beta * self.beta * S::HALF - self.log_norm - self.m * self.log_join_ratio(x)?
        } else {
            -x * x * S::HALF - self.log_norm
        })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if x <= -self.beta {
            let lower = self.tail(x)?;
            return if upper { log1mexp(lower) } else { Ok(lower) };
        }
        let survival = scioxide_special::normal_log_sf(x)?
            + crate::distribution::special::log_two_pi::<S>() * S::HALF
            - self.log_norm;
        if upper {
            Ok(survival)
        } else {
            log1mexp(survival)
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        self.invert(p, false)
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        self.invert(p, true)
    }
}
formula_law!(CrystalBall);
impl<S: RealScalar> Mean<S> for CrystalBall<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(1)
    }
}
impl<S: RealScalar> Variance<S> for CrystalBall<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let m = self.mean()?;
        crate::distribution::moment_math::variance_from_moments(self.raw_moment(2)?, m)
    }
}

/// Skewed Cauchy distribution with skewness `−1 < a < 1`.
///
/// The density is `f(x) = 1 / (π (1 + x²/(1 + a sgn x)²))`: half-Cauchy pieces
/// with scale `1 − a` on the left and `1 + a` on the right. The mean and
/// variance do not exist.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Quantile, SeedableRng, SkewCauchy, SplitMix64};
/// let law = SkewCauchy::<f64>::new(0.5)?;
/// assert!((law.cdf(0.0)? - 0.25).abs() < 1e-15);
/// let x = law.quantile(0.9)?;
/// assert!((law.cdf(x)? - 0.9).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SkewCauchy<S> {
    a: S,
}
impl<S: RealScalar> SkewCauchy<S> {
    /// Creates a skewed Cauchy distribution with skewness `a`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` is NaN or infinite.
    /// - `DomainError` if `|a| ≥ 1`.
    pub fn new(a: S) -> Result<Self, SciOxideError> {
        argument(a, "a")?;
        if a.abs() >= S::ONE {
            return Err(crate::law::domain(
                scioxide_core::argument!("a"),
                "SkewCauchy requires |a| < 1",
            ));
        }
        Ok(Self { a })
    }
    fn small_tail(&self, x: S, a: S) -> S {
        let scale = S::ONE + a;
        if x == S::ZERO {
            return (scale * S::HALF).ln();
        }
        // atan(scale/x) rather than pi/2-atan(x/scale).
        let ratio = scale / x;
        let log_angle = if ratio < S::SQRT_EPSILON {
            scale.ln() - x.ln()
        } else {
            ratio.atan().ln()
        };
        scale.ln() + log_angle - S::PI.ln()
    }
    fn tails(&self, x: S) -> Result<(S, S), SciOxideError> {
        argument(x, "x")?;
        let a = if x < S::ZERO { -self.a } else { self.a };
        let magnitude = x.abs();
        let scale = S::ONE + a;
        let two = S::TWO;
        let (lower, upper) = if a > S::ZERO && magnitude <= scale {
            // The lower tail is a sum of the small half-mass and positive central mass.
            // Computing it from a rounded near-one survival would erase this probability.
            let lower = logadd(
                (-a).ln_1p() - two.ln(),
                scale.ln() + (magnitude / scale).atan().ln() - S::PI.ln(),
            )?;
            (lower, log1mexp(lower)?)
        } else {
            let upper = self.small_tail(magnitude, a);
            (log1mexp(upper)?, upper)
        };
        Ok(if x < S::ZERO {
            (upper, lower)
        } else {
            (lower, upper)
        })
    }
    fn invert(&self, p: S, upper: bool) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                S::NEG_INFINITY
            } else {
                S::INFINITY
            });
        }
        // The law is two half-Cauchy pieces with scales 1 - a (left) and
        // 1 + a (right); P(X < 0) = (1 - a) / 2. Each piece is inverted with the
        // reflected cotangent form, and upper-tail requests mirror a.
        let a = if upper { -self.a } else { self.a };
        let left = (S::ONE - a) * S::HALF;
        let x = if p <= left {
            -(S::ONE - a) / (S::PI * (p / (S::ONE - a))).tan()
        } else {
            (S::ONE + a) / (S::PI * ((S::ONE - p) / (S::ONE + a))).tan()
        };
        finite(if upper { -x } else { x })
    }
}
impl<S: RealScalar> Formula<S> for SkewCauchy<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let scale = if x < S::ZERO {
            S::ONE - self.a
        } else {
            S::ONE + self.a
        };
        Ok(-S::PI.ln() - c::<S>(2.)? * (scale.hypot(x).ln() - scale.ln()))
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        let (lower, survival) = self.tails(x)?;
        Ok(if upper { survival } else { lower })
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        self.invert(p, false)
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        self.invert(p, true)
    }
}
formula_law!(SkewCauchy);

/// Jones–Faddy skew-t distribution with shapes `a > 0` and `b > 0`.
///
/// With `r = a + b`,
///
/// ```text
/// f(x) = (1 + x/√(r + x²))^(a+1/2) (1 − x/√(r + x²))^(b+1/2) / (2^(r−1) B(a, b) √r),
/// ```
///
/// where `B` is the beta function. `U = (1 + X/√(r + X²))/2` has a
/// `Beta(a, b)` distribution, and `a = b` gives Student's t with `2a` degrees
/// of freedom. The mean exists for `a, b > 1/2` and the variance for `a, b > 1`.
///
/// # References
///
/// M. C. Jones and M. J. Faddy, "A skew extension of the t-distribution, with
/// applications", *J. R. Stat. Soc. B* 65(1), 2003, 159–174.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, JonesFaddySkewT, Quantile, SeedableRng, SplitMix64, Variance,
/// };
/// // a = b = 2 is Student's t with four degrees of freedom.
/// let law = JonesFaddySkewT::<f64>::new(2.0, 2.0)?;
/// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-15);
/// let x = law.quantile(0.95)?;
/// assert!((law.cdf(x)? - 0.95).abs() < 1e-13);
/// assert!((law.variance()? - 2.0).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(43);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct JonesFaddySkewT<S: RealScalar> {
    options: crate::LawOptions<S>,
    a: S,
    b: S,
    root: S,
    log_norm: S,
    beta: BetaKernel<S>,
}
impl<S: RealScalar> JonesFaddySkewT<S> {
    /// Creates a Jones–Faddy skew-t distribution with shapes `a` and `b`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` or `b` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0` or `b ≤ 0`.
    /// - `NumericalInstability` if `a + b` or the normalizing constant is outside the range of `S`.
    pub fn new(a: S, b: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, b, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(a: S, b: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(a, "a")?;
        argument(b, "b")?;
        if a <= S::ZERO || b <= S::ZERO {
            return Err(crate::law::domain(
                if a <= S::ZERO {
                    scioxide_core::argument!("a")
                } else {
                    scioxide_core::argument!("b")
                },
                "JonesFaddySkewT requires positive shapes",
            ));
        }
        let sum = finite(a + b)?;
        let root = sum.sqrt();
        let log_norm = finite(
            (sum - S::ONE) * c::<S>(2.)?.ln() + scioxide_special::ln_beta(a, b)? + root.ln(),
        )?;
        Ok(Self {
            options,
            a,
            b,
            root,
            log_norm,
            beta: BetaKernel::new(a, b)?,
        })
    }
    // Jones-Faddy: U = (1 + x / sqrt(r + x^2)) / 2 ~ Beta(a, b) with r = a + b.
    // Returns (ln U, ln(1 - U)). The coordinate that is near zero is formed
    // from r / (h (h + |x|)) instead of 1 - (large one), so far tails keep
    // their relative precision.
    fn log_coordinates(&self, x: S) -> (S, S) {
        let h = self.root.hypot(x);
        let two = S::TWO;
        let large = (S::ONE + x.abs() / h).ln() - two.ln();
        let small = -two.ln() + two * (self.root.ln() - h.ln()) - (S::ONE + x.abs() / h).ln();
        if x < S::ZERO {
            (small, large)
        } else {
            (large, small)
        }
    }
    fn invert(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                S::NEG_INFINITY
            } else {
                S::INFINITY
            });
        }
        // The upper tail of X is the lower tail of -X, whose beta parameters
        // are swapped.
        let (a, b) = if upper {
            (self.b, self.a)
        } else {
            (self.a, self.b)
        };
        let u = crate::law::special(left, |budget| {
            scioxide_special::beta_inc_inv_log_with_options(
                a,
                b,
                p.ln(),
                scioxide_special::BetaOptions::new(budget),
            )
        })?;
        // Above one half, invert the complement 1 - u directly instead of
        // rounding it from u.
        let x = if u <= S::HALF {
            self.root * (c::<S>(2.)? * u - S::ONE) / (c::<S>(2.)? * u.sqrt() * (S::ONE - u).sqrt())
        } else {
            let v = crate::law::special(left, |budget| {
                scioxide_special::beta_inc_inv_log_with_options(
                    b,
                    a,
                    (-p).ln_1p(),
                    scioxide_special::BetaOptions::new(budget),
                )
            })?;
            self.root * (S::ONE - c::<S>(2.)? * v) / (c::<S>(2.)? * v.sqrt() * (S::ONE - v).sqrt())
        };
        finite(if upper { -x } else { x })
    }

    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let (u, v) = self.log_coordinates(x);
        if u < v {
            beta_log_coordinate(self.a, self.b, u, left)
        } else {
            log1mexp(beta_log_coordinate(self.b, self.a, v, left)?)
        }
    }

    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let (u, v) = self.log_coordinates(x);
        if v < u {
            beta_log_coordinate(self.b, self.a, v, left)
        } else {
            log1mexp(beta_log_coordinate(self.a, self.b, u, left)?)
        }
    }
}
impl<S: RealScalar> Formula<S> for JonesFaddySkewT<S> {
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
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        let (u, v) = self.log_coordinates(x);
        let l2 = c::<S>(2.)?.ln();
        Ok((self.a + S::HALF) * (u + l2) + (self.b + S::HALF) * (v + l2) - self.log_norm)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            self.log_sf_work(x, left)
        } else {
            self.log_cdf_work(x, left)
        }
    }
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        self.invert(p, upper, left)
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        // Inverse of the Beta coordinate map: x = sqrt(r) (2u - 1) / (2 sqrt(u (1 - u))).
        let u = self.beta.sample(rng)?;
        finite(
            self.root * (c::<S>(2.)? * u - S::ONE) / (c::<S>(2.)? * u.sqrt() * (S::ONE - u).sqrt()),
        )
    }
}
formula_law!(JonesFaddySkewT);
impl<S: RealScalar> Mean<S> for JonesFaddySkewT<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.a <= S::HALF || self.b <= S::HALF {
            return Err(crate::law::undefined_moment(
                "Jones-Faddy mean does not exist",
            ));
        }
        // ln B(a-1/2, b-1/2) - ln B(a, b) as three log-gamma ratios.
        let (a, b) = (self.a.into_f64(), self.b.into_f64());
        let log_ratio = crate::distribution::special::log_gamma_ratio(a, -0.5)?
            + crate::distribution::special::log_gamma_ratio(b, -0.5)?
            - crate::distribution::special::log_gamma_ratio(a + b, -1.0)?;
        finite(
            self.root * S::HALF * (self.a - self.b) / (self.a + self.b - S::ONE)
                * c::<S>(log_ratio.exp())?,
        )
    }
}
impl<S: RealScalar> Variance<S> for JonesFaddySkewT<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.a <= S::ONE || self.b <= S::ONE {
            return Err(crate::law::undefined_moment(
                "Jones-Faddy variance does not exist",
            ));
        }
        let mean = self.mean()?;
        let d = self.a - self.b;
        crate::distribution::moment_math::variance_from_moments(
            (self.a + self.b) * (d * d + self.a + self.b - c(2.)?)
                / (c::<S>(4.)? * (self.a - S::ONE) * (self.b - S::ONE)),
            mean,
        )
    }
}

/// Skew-normal distribution with shape `a`.
///
/// The density is `f(x) = 2 φ(x) Φ(a x)`, where `φ` and `Φ` are the standard
/// normal density and CDF. `a = 0` gives the standard normal law. Shift and
/// scale it with [`Affine`](crate::Affine).
///
/// # References
///
/// A. Azzalini, "A class of distributions which includes the normal ones",
/// *Scand. J. Statist.* 12(2), 1985, 171–178.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, Quantile, SeedableRng, SkewNormal, SplitMix64,
/// };
/// let law = SkewNormal::<f64>::new(3.0)?;
/// let x = law.quantile(0.2)?;
/// assert!((law.cdf(x)? - 0.2).abs() < 1e-12);
/// let delta = 3.0 / 10.0_f64.sqrt();
/// assert!((law.mean()? - delta * (2.0 / core::f64::consts::PI).sqrt()).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(44);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SkewNormal<S: RealScalar> {
    a: S,
    options: LawOptions<S>,
}
impl<S: RealScalar> SkewNormal<S> {
    /// Creates a skew-normal distribution with shape `a`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` is NaN or infinite.
    pub fn new(a: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(a: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(a, "a")?;
        Ok(Self { a, options })
    }
    fn tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let (x, a) = if upper { (x, self.a) } else { (-x, -self.a) };
        if a == S::ZERO {
            return crate::law::special(left, |budget| {
                scioxide_special::normal_log_sf_with_options(
                    x,
                    scioxide_special::ProbabilityOptions::default().with_iteration_budget(budget),
                )
            });
        }
        if x < S::ZERO {
            let reflected = Self {
                a: -a,
                options: self.options,
            };
            return log1mexp(reflected.tail(-x, true, left)?);
        }
        // Closed form S(x) = Phi(-x) + 2 T(x, a) with Owen's T. It is used for
        // a > 0 and moderate x, where both terms are positive and no
        // cancellation occurs; beyond x = 8 or for a < 0 (where the terms
        // nearly cancel) the tail is integrated directly from the density.
        if a > S::ZERO && x < c(8.)? {
            let n = crate::law::special(left, |budget| {
                scioxide_special::normal_log_sf_with_options(
                    x,
                    scioxide_special::ProbabilityOptions::default().with_iteration_budget(budget),
                )
            })?;
            let t = crate::law::special(left, |budget| {
                scioxide_special::owens_t_with_options(
                    x,
                    a,
                    scioxide_special::ProbabilityOptions::default().with_iteration_budget(budget),
                )
            })?;
            return crate::law::log_probability(logadd(n, (c::<S>(2.)? * t).ln())?);
        }
        let law = Self {
            a,
            options: self.options,
        };
        // For a < 0 the right tail decays like exp(-(1 + a^2) x^2 / 2), so the
        // quadrature panel width shrinks to its natural scale 1 / sqrt(1 + a^2).
        let width = if a < S::ZERO {
            S::ONE / S::ONE.hypot(a)
        } else {
            S::ONE
        };
        crate::distribution::quadrature::tail(
            |z, left| law.log_pdf_work(z, left),
            x,
            true,
            S::ZERO,
            width,
            self.options,
            left,
        )
    }

    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let z = self.a * x;
        let ln = if z == S::INFINITY {
            S::ZERO
        } else if z == S::NEG_INFINITY {
            S::NEG_INFINITY
        } else {
            crate::law::special(left, |budget| {
                scioxide_special::normal_log_cdf_with_options(
                    z,
                    scioxide_special::ProbabilityOptions::default().with_iteration_budget(budget),
                )
            })?
        };
        Ok(c::<S>(2.)?.ln()
            - crate::distribution::special::log_two_pi::<S>() * S::HALF
            - x * x * S::HALF
            + ln)
    }
}

impl<S: RealScalar> Formula<S> for SkewNormal<S> {
    const INVERSE: Inverse = Inverse::Lattice;
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
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        self.log_pdf_work(x, &mut self.options.work_budget().get())
    }
    fn density_with_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_pdf_work(x, left)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        self.tail(x, upper, left)
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        // Azzalini's representation: delta |Z0| + sqrt(1 - delta^2) Z1 with
        // delta = a / sqrt(1 + a^2) is skew-normal(a); no rejection is needed.
        let h = S::ONE.hypot(self.a);
        let first = standard_normal::<S, R>(rng)?.abs();
        finite((self.a / h) * first + standard_normal::<S, R>(rng)? / h)
    }
}
formula_law!(SkewNormal);
impl<S: RealScalar> Mean<S> for SkewNormal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.a / S::ONE.hypot(self.a) * (c::<S>(2.)? / S::PI).sqrt())
    }
}
impl<S: RealScalar> Variance<S> for SkewNormal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let m = self.mean()?;
        Ok(S::ONE - m * m)
    }
}

fn beta_log_coordinate<S: RealScalar>(
    a: S,
    b: S,
    u: S,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    // The leading term avoids evaluating `beta_inc` at an underflowed x.
    if let Some(value) =
        crate::distribution::special::beta_leading_log(a, b, u, || scioxide_special::ln_beta(a, b))?
    {
        return Ok(value);
    }
    crate::law::special(left, |budget| {
        scioxide_special::beta_inc_log_with_options(
            a,
            b,
            u.exp(),
            scioxide_special::BetaOptions::new(budget),
        )
    })
}
