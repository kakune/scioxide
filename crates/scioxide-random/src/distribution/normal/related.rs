//! Laws derived from the normal distribution.

use crate::distribution::formula::formula_law;
use crate::distribution::formula::positive_support;
use crate::distribution::formula::{Formula, Inverse};
use crate::distribution::special::normal_log_density;
use crate::distribution::symmetric;
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{finite, log1mexp, logadd, logdiff};
use crate::{Distribution, LogCdf, LogDensity, LogSurvival, Quantile};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Alpha distribution on `x > 0` with shape `a > 0`.
///
/// `X = 1 / (a − Z)` for a standard normal `Z` conditioned on `Z < a`, so
/// `F(x) = Φ(a − 1/x) / Φ(a)`, where `Φ` is the standard normal CDF. The mean
/// and variance do not exist. The law has unit scale; scale it with
/// [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{Alpha, Cdf, Density, Distribution, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support};
/// let law = Alpha::new(2.0_f64)?;
/// // F(1/a) = Φ(0) / Φ(a).
/// assert!((law.cdf(0.5)? - 0.5 / 0.977_249_868_051_820_8).abs() < 1e-8);
/// assert!(law.pdf(0.5)? > 0.0 && law.pdf(-1.0)? == 0.0);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(2);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Alpha<S: RealScalar> {
    options: crate::LawOptions<S>,
    shape: S,
    normal: super::Normal<S>,
    log_norm: S,
}
impl<S: RealScalar> Alpha<S> {
    /// Creates an alpha distribution with shape `a`.
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
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(shape: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(shape, scioxide_core::argument!("shape"))?;
        // X = 1 / (a - Z) conditioned on Z < a, so every probability carries
        // the normalizer Phi(a), cached here in log form.
        let normal = super::Normal::standard();
        let log_norm = normal.log_cdf(shape)?;
        Ok(Self {
            options,
            shape,
            normal,
            log_norm,
        })
    }
}
impl<S: RealScalar> Formula<S> for Alpha<S> {
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
            Ok(S::NEG_INFINITY)
        } else {
            Ok(self.normal.log_pdf(self.shape - S::ONE / x)?
                - number::<S>(2.)? * x.ln()
                - self.log_norm)
        }
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            // With h = 1/x, Taylor expansion of phi about a gives
            // Phi(a) - Phi(a - h) = phi(a) h B with
            // ln B = a h/2 + (a^2/24 - 1/6) h^2 - a h^3/24 + O(h^4 (1 + a^4)),
            // which avoids subtracting nearly equal CDFs. Past this point
            // h < sqrt(eps), so the omitted terms are below eps relative for
            // |a| <= 100 in f32 and f64.
            if x > S::ONE / S::SQRT_EPSILON {
                let h = x.recip();
                let a = self.shape;
                let correction = a * h / number::<S>(2.)?
                    + (a * a / number::<S>(24.)? - S::ONE / number::<S>(6.)?) * h * h;
                return Ok(self.normal.log_pdf(a)? - x.ln() + correction - self.log_norm);
            }
            Ok(crate::law::logdiff(
                self.log_norm,
                self.normal.log_cdf(self.shape - S::ONE / x)?,
                scioxide_core::argument!("self"),
            )? - self.log_norm)
        } else {
            if x <= S::ZERO {
                Ok(S::NEG_INFINITY)
            } else {
                Ok(self.normal.log_cdf(self.shape - S::ONE / x)? - self.log_norm)
            }
        }
    }
    // Closed forms on the cancellation-free side: F(x) = p gives
    // x = 1 / (a - inverse Phi(p Phi(a))), with the product kept in log space.
    // Each side above one half inverts the exact complement on the other.
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        self.inverse_with_remaining(p, false, &mut self.work_budget().get())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        self.inverse_with_remaining(p, true, &mut self.work_budget().get())
    }
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        crate::law::charge(left)?;
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                S::ZERO
            } else {
                S::INFINITY
            });
        }
        let (tail, upper) = if p > S::HALF {
            (S::ONE - p, !upper)
        } else {
            (p, upper)
        };
        if upper {
            self.upper_inverse(tail, left)
        } else {
            self.lower_inverse(tail)
        }
    }
}
impl<S: RealScalar> Alpha<S> {
    fn lower_inverse(&self, p: S) -> Result<S, SciOxideError> {
        let z = scioxide_special::normal_inv_log_cdf(p.ln() + self.log_norm)?;
        finite(S::ONE / (self.shape - z))
    }
    /// `S(x) = (Phi(a) - Phi(a - 1/x)) / Phi(a) = q`: the closed form cancels for
    /// small `q`, so an unresolved difference uses the first-order
    /// `1/x ~ q Phi(a) / phi(a)` to seed Newton in `ln x`.
    fn upper_inverse(&self, q: S, left: &mut usize) -> Result<S, SciOxideError> {
        let delta =
            self.shape - scioxide_special::normal_inv_log_cdf(self.log_norm + (-q).ln_1p())?;
        let target = q.ln();
        let mut t = if delta > S::SQRT_EPSILON * self.shape.max(S::ONE) {
            -delta.ln()
        } else {
            self.normal.log_pdf(self.shape)? - self.log_norm - target
        };
        let mut iterations = 0;
        for _ in 0..16 {
            crate::law::charge(left)?;
            iterations += 1;
            let x = finite(t.exp())?;
            let log_tail = self.tail_with_work(x, true, left)?;
            let residual = log_tail - target;
            if residual.abs() <= self.options.tolerance().relative() * target.abs().max(S::ONE) {
                return finite(x);
            }
            let slope = -(t + self.density_log(x)? - log_tail).exp();
            let step = residual / slope;
            if !step.is_finite() {
                break;
            }
            t -= step;
            if step.abs() <= S::TWO * S::EPSILON * t.abs().max(S::ONE) {
                break;
            }
        }
        Err(crate::law::iteration_limit(iterations))
    }
}
formula_law!(Alpha);

/// Fatigue-life (Birnbaum–Saunders) distribution on `x > 0` with shape `c > 0`.
///
/// `F(x) = Φ((√x − 1/√x) / c)`, where `Φ` is the standard normal CDF, so the
/// median is exactly one. The law has unit scale; scale it with
/// [`Affine`](crate::Affine).
///
/// # References
///
/// Z. W. Birnbaum and S. C. Saunders, "A new family of life distributions",
/// *J. Appl. Probab.* 6(2), 1969, 319–327.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, FatigueLife, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = FatigueLife::new(0.5_f64)?;
/// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-8);
/// assert!((law.mean()? - 1.125).abs() < 1e-15); // 1 + c²/2
/// assert!((law.variance()? - 0.25 * (1.0 + 1.25 * 0.25)).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(3);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct FatigueLife<S: RealScalar> {
    shape: S,
    normal: super::Normal<S>,
}
impl<S: RealScalar> FatigueLife<S> {
    /// Creates a fatigue-life distribution with shape `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self {
            shape,
            normal: super::Normal::standard(),
        })
    }
    fn z(&self, x: S) -> S {
        (x.sqrt() - S::ONE / x.sqrt()) / self.shape
    }
}
impl<S: RealScalar> Formula<S> for FatigueLife<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        positive_support()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let z = self.z(x);
        Ok(x.ln_1p()
            - (number::<S>(2.)? * self.shape).ln()
            - crate::distribution::special::log_two_pi::<S>() * S::HALF
            - number::<S>(3.)? * S::HALF * x.ln()
            - z * z * S::HALF)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                Ok(S::ZERO)
            } else {
                self.normal.log_sf(self.z(x))
            }
        } else {
            if x <= S::ZERO {
                Ok(S::NEG_INFINITY)
            } else {
                self.normal.log_cdf(self.z(x))
            }
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        // Solve sqrt(x) - 1/sqrt(x) = c z for y = sqrt(x): y = w + sqrt(w^2 + 1)
        // with w = c z / 2. For w < 0 the sum cancels, so the equivalent
        // 1 / (sqrt(w^2 + 1) - w) is used; `hypot` avoids overflow of w^2.
        let z = self.normal.quantile(p)? * self.shape * S::HALF;
        let root = z.hypot(S::ONE);
        let y = if z < S::ZERO {
            S::ONE / (root - z)
        } else {
            root + z
        };
        finite(y * y)
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        let z = -self.normal.quantile(p)? * self.shape * S::HALF;
        let root = z.hypot(S::ONE);
        let y = if z < S::ZERO {
            S::ONE / (root - z)
        } else {
            root + z
        };
        finite(y * y)
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        let z = self.normal.sample(rng)? * self.shape * S::HALF;
        let root = z.hypot(S::ONE);
        let y = if z < S::ZERO {
            S::ONE / (root - z)
        } else {
            root + z
        };
        finite(y * y)
    }
}
formula_law!(FatigueLife);
impl<S: RealScalar> Mean<S> for FatigueLife<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(S::ONE + self.shape * self.shape * S::HALF)
    }
}
impl<S: RealScalar> Variance<S> for FatigueLife<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let c2 = self.shape * self.shape;
        finite(c2 * (S::ONE + number::<S>(1.25)? * c2))
    }
}

/// Symmetric generalized normal (exponential power) distribution with exponent `β > 0`.
///
/// The density is `f(x) = β exp(−|x|^β) / (2 Γ(1/β))` on the real line.
/// `β = 1` is the standard Laplace law and `β = 2` a normal law with variance
/// `1/2`. The law has zero location and unit scale; shift and scale it with
/// [`Affine`](crate::Affine).
///
/// Draws are `±G^(1/β)` with `G ~ Gamma(1/β)`, computed in logarithms. A
/// nonzero draw whose magnitude is outside the range of `S` is a
/// `NumericalInstability` error (`Underflow` or `Overflow`). The quantile at
/// `p = 1/2` is exactly zero.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, GeneralizedNormal, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Variance};
/// let law = GeneralizedNormal::new(2.0_f64)?;
/// assert_eq!(law.mean()?, 0.0);
/// assert!((law.variance()? - 0.5).abs() < 1e-15);
/// assert_eq!(law.quantile(0.5)?, 0.0);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(4);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedNormal<S: RealScalar> {
    options: crate::LawOptions<S>,
    shape: S,
    gamma: super::super::gamma::GammaKernel<S>,
    /// `ln Gamma(1/beta)` for the density, prepared once.
    log_gamma: Result<S, SciOxideError>,
    /// `ln Gamma(1 + 1/beta)` in binary64 for the inverse, prepared once.
    log_gamma_one_plus: Result<f64, SciOxideError>,
}
impl<S: RealScalar> GeneralizedNormal<S> {
    /// Creates a generalized normal distribution with exponent `β = shape`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    /// - `NumericalInstability` (`Overflow`) if `1/shape` is not finite.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        Self::with_options(shape, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(shape: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(shape, scioxide_core::argument!("shape"))?;
        let inverse = finite(S::ONE / shape)?;
        Ok(Self {
            options,
            shape,
            gamma: super::super::gamma::GammaKernel::new(inverse)?,
            log_gamma: scioxide_special::ln_gamma(inverse),
            log_gamma_one_plus: crate::distribution::gamma::log_gamma_one_plus(
                1.0 / shape.into_f64(),
            ),
        })
    }
    // Binary64 logarithmic tail of ±|Z|^(1/beta), Z ~ Gamma(1/beta): the
    // magnitude survives beyond y with probability Q(1/beta, y^beta).
    fn log_tail(&self, x: f64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        let beta = self.shape.into_f64();
        symmetric::log_tail(x, upper, |y| {
            let log_y = beta * y.ln();
            // Beyond the binary64 coordinate range the survival rounds to zero.
            if log_y > f64::MAX.ln() {
                Ok(f64::NEG_INFINITY)
            } else {
                crate::distribution::gamma::gamma_log_coordinate(1.0 / beta, log_y, true, left)
            }
        })
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedNormal<S> {
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
        Ok(self.shape.ln() - S::LN_2 - self.log_gamma.clone()? - x.abs().powf(self.shape))
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        number(self.log_tail(x.into_f64(), upper, _left)?)
    }
    fn probability(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        // Form the probability before narrowing a large negative binary64
        // logarithm; its rounded binary32 zero is still representable.
        number(
            self.log_tail(x.into_f64(), upper, &mut self.options.work_budget().get())?
                .exp(),
        )
    }
    fn center(&self) -> Option<S> {
        Some(S::ZERO)
    }
    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        symmetric::inverse(p, upper, |tail| {
            let beta = self.shape.into_f64();
            let log_p = (-tail.into_f64()).ln_1p();
            let log_norm = self.log_gamma_one_plus.clone()?;
            let log_y = beta * (log_p + log_norm);
            let log_magnitude = if log_y < f64::EPSILON.ln() {
                log_p + log_norm
            } else {
                crate::law::special(left, |budget| {
                    scioxide_special::gamma_q_inv_with_options(
                        1.0 / beta,
                        tail.into_f64(),
                        scioxide_special::GammaOptions::new(budget),
                    )
                })?
                .ln()
                    / beta
            };
            crate::law::positive_draw::<S>(log_magnitude)
        })
    }
    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        symmetric::draw(rng, |rng| {
            crate::law::positive_draw::<S>(
                self.gamma
                    .sample_log_power(rng, 1.0 / self.shape.into_f64())?,
            )
        })
    }
}
formula_law!(GeneralizedNormal);
impl<S: RealScalar> Mean<S> for GeneralizedNormal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> Variance<S> for GeneralizedNormal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let base = 1.0 / self.shape.into_f64();
        finite(number::<S>(
            crate::distribution::special::log_gamma_ratio(base, 2.0 * base)?.exp(),
        )?)
    }
}

/// Exponentially modified normal distribution with shape `k > 0`.
///
/// `X = Z + E`, where `Z` is standard normal and `E` is an independent
/// exponential variable with mean `k`. Shift and scale it with
/// [`Affine`](crate::Affine). Quantiles are computed by numerical inversion of
/// the CDF.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, ExponentiallyModifiedNormal, Mean};
/// use scioxide_random::{Quantile, SeedableRng, SplitMix64, Survival, Variance};
/// let law = ExponentiallyModifiedNormal::new(2.0_f64)?;
/// assert_eq!((law.mean()?, law.variance()?), (2.0, 5.0));
/// assert!((law.cdf(1.0)? + law.sf(1.0)? - 1.0).abs() < 1e-8);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(5);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExponentiallyModifiedNormal<S: RealScalar> {
    options: crate::LawOptions<S>,
    k: S,
}
impl<S: RealScalar> ExponentiallyModifiedNormal<S> {
    /// Creates an exponentially modified normal distribution with exponential mean `k`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `k` is NaN or infinite.
    /// - `DomainError` if `k ≤ 0`.
    /// - `NumericalInstability` (`Overflow`) if `1/k` overflows.
    pub fn new(k: S) -> Result<Self, SciOxideError> {
        Self::with_options(k, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(k: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(k, scioxide_core::argument!("k"))?;
        if !(S::ONE / k).is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "exponential normal rate is not representable",
            ));
        }
        Ok(Self { options, k })
    }
    fn log_component(&self, x: S) -> Result<S, SciOxideError> {
        let inv = S::ONE / self.k;
        let shift = inv * S::HALF;
        Ok(-x * inv + inv * shift + scioxide_special::normal_log_cdf(x - inv)?)
    }
}
impl<S: RealScalar> Formula<S> for ExponentiallyModifiedNormal<S> {
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
        Ok(self.log_component(x)? - self.k.ln())
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            crate::law::log_probability(logadd(
                scioxide_special::normal_log_sf(x)?,
                self.log_component(x)?,
            )?)
        } else {
            logdiff(
                scioxide_special::normal_log_cdf(x)?,
                self.log_component(x)?,
                scioxide_core::argument!("self"),
            )
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        let z = super::Normal::standard().sample(rng)?;
        finite(z + self.k * super::super::exponential::standard_exponential::<S, R>(rng)?)
    }
}
formula_law!(ExponentiallyModifiedNormal);
impl<S: RealScalar> Mean<S> for ExponentiallyModifiedNormal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.k)
    }
}
impl<S: RealScalar> Variance<S> for ExponentiallyModifiedNormal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(S::ONE + self.k * self.k)
    }
}

/// Power-normal distribution with power `c > 0`.
///
/// The survival function is `S(x) = (1 − Φ(x))^c`, where `Φ` is the standard
/// normal CDF. Quantiles and inverse survival values come from the normal
/// quantile in closed form, through the smaller tail. They switch to
/// logarithmic inversion only where the normal tail probability underflows.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, PowerNormal, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = PowerNormal::new(2.0_f64)?;
/// assert!((law.sf(0.0)? - 0.25).abs() < 1e-15); // normal_sf(0)^2
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(6);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PowerNormal<S: RealScalar> {
    options: crate::LawOptions<S>,
    c: S,
}
impl<S: RealScalar> PowerNormal<S> {
    /// Creates a power-normal distribution with power `c`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` is NaN or infinite.
    /// - `DomainError` if `c ≤ 0`.
    pub fn new(c: S) -> Result<Self, SciOxideError> {
        Self::with_options(c, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(c: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(c, scioxide_core::argument!("c"))?;
        Ok(Self { options, c })
    }
}
impl<S: RealScalar> PowerNormal<S> {
    fn inverse_tail(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                S::NEG_INFINITY
            } else {
                S::INFINITY
            });
        }
        let log_sf = if upper { p.ln() } else { (-p).ln_1p() } / self.c;
        let (probability, reflect) = if log_sf > -S::LN_2 {
            (-log_sf.exp_m1(), false)
        } else {
            (log_sf.exp(), true)
        };
        if probability == S::ZERO {
            // Only probability underflow needs the logarithmic inverse driver.
            return crate::distribution::formula::numerical_inverse(
                self,
                p,
                upper,
                Some(self.options.tolerance()),
                left,
            );
        }
        let x = scioxide_special::normal_quantile(probability)?;
        finite(if reflect { -x } else { x })
    }
}
impl<S: RealScalar> Formula<S> for PowerNormal<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.c.ln()
            + normal_log_density(x)
            + (self.c - S::ONE) * scioxide_special::normal_log_sf(x)?)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(self.c * scioxide_special::normal_log_sf(x)?)
        } else {
            log1mexp(self.tail_with_work(x, true, _left)?)
        }
    }

    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if upper {
            self.inverse_tail(p, true, left)
        } else {
            self.inverse_tail(p, false, left)
        }
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }
}
formula_law!(PowerNormal);

/// Double Pareto lognormal distribution on `x > 0`.
///
/// `X = exp(μ + σZ + E₁ − E₂)`, where `Z` is standard normal and `E₁`, `E₂` are
/// independent exponential variables with rates `α > 0` and `β > 0`. The mean
/// exists only for `α > 1`. Quantiles are computed by numerical inversion of
/// the CDF.
///
/// # References
///
/// W. J. Reed and M. Jorgensen, "The double Pareto-lognormal distribution: a
/// new parametric model for size distributions", *Comm. Statist. Theory
/// Methods* 33(8), 2004, 1733–1753.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, DoubleParetoLogNormal, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support};
/// let law = DoubleParetoLogNormal::new(0.0_f64, 0.5, 3.0, 2.0)?;
/// // E[X] = exp(μ + σ²/2) α/(α-1) β/(β+1).
/// let mean = 0.125_f64.exp() * 1.5 * (2.0 / 3.0);
/// assert!((law.mean()? - mean).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(7);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DoubleParetoLogNormal<S: RealScalar> {
    options: crate::LawOptions<S>,
    mu: S,
    sigma: S,
    alpha: S,
    beta: S,
}
impl<S: RealScalar> DoubleParetoLogNormal<S> {
    /// Creates a double Pareto lognormal distribution with parameters `μ = mu`,
    /// `σ = sigma`, `α = alpha` and `β = beta`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if any parameter is NaN or infinite.
    /// - `DomainError` if `sigma`, `alpha` or `beta` is not positive.
    /// - `NumericalInstability` (`Overflow`) if `alpha + beta` overflows.
    pub fn new(mu: S, sigma: S, alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Self::with_options(mu, sigma, alpha, beta, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        mu: S,
        sigma: S,
        alpha: S,
        beta: S,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::argument(mu, "mu")?;
        positive(sigma, scioxide_core::argument!("sigma"))?;
        positive(alpha, scioxide_core::argument!("alpha"))?;
        positive(beta, scioxide_core::argument!("beta"))?;
        if !(alpha + beta).is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "double Pareto lognormal rates exceed scalar range",
            ));
        }
        Ok(Self {
            options,
            mu,
            sigma,
            alpha,
            beta,
        })
    }
    fn components(&self, x: S) -> Result<(S, S, S), SciOxideError> {
        let d = x.ln() - self.mu;
        let z = d / self.sigma;
        let asig = self.alpha * self.sigma;
        let bsig = self.beta * self.sigma;
        let pos =
            -self.alpha * d + S::HALF * asig * asig + scioxide_special::normal_log_cdf(z - asig)?;
        let neg =
            self.beta * d + S::HALF * bsig * bsig + scioxide_special::normal_log_sf(z + bsig)?;
        Ok((z, pos, neg))
    }
    fn log_moment(&self, r: S) -> Result<S, SciOxideError> {
        if r >= self.alpha {
            return Err(crate::law::undefined_moment(
                "double Pareto lognormal moment does not exist",
            ));
        }
        Ok(
            r * self.mu + S::HALF * (r * self.sigma) * (r * self.sigma) + self.alpha.ln()
                - (self.alpha - r).ln()
                + self.beta.ln()
                - (self.beta + r).ln(),
        )
    }
}
impl<S: RealScalar> Formula<S> for DoubleParetoLogNormal<S> {
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
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let (_, pos, neg) = self.components(x)?;
        Ok(
            self.alpha.ln() + self.beta.ln() - (self.alpha + self.beta).ln() + logadd(pos, neg)?
                - x.ln(),
        )
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            let (z, pos, neg) = self.components(x)?;
            let phi = scioxide_special::normal_log_sf(z)?;
            let positive = logadd(phi, pos)?;
            let negative = logdiff(phi, neg, scioxide_core::argument!("self"))?;
            let total = (self.alpha + self.beta).ln();
            crate::law::log_probability(logadd(
                self.beta.ln() - total + positive,
                self.alpha.ln() - total + negative,
            )?)
        } else {
            if x <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            let (z, pos, neg) = self.components(x)?;
            let phi = scioxide_special::normal_log_cdf(z)?;
            let positive = logdiff(phi, pos, scioxide_core::argument!("self"))?;
            let negative = logadd(phi, neg)?;
            let total = (self.alpha + self.beta).ln();
            crate::law::log_probability(logadd(
                self.beta.ln() - total + positive,
                self.alpha.ln() - total + negative,
            )?)
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        __left: &mut usize,
    ) -> Result<S, SciOxideError> {
        let z = super::Normal::new(self.mu, self.sigma)?.sample(rng)?;
        let y = z + super::super::exponential::standard_exponential::<S, R>(rng)? / self.alpha
            - super::super::exponential::standard_exponential::<S, R>(rng)? / self.beta;
        let x = finite(y.exp())?;
        if x == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "double Pareto lognormal sample underflowed",
            ));
        }
        Ok(x)
    }
}
formula_law!(DoubleParetoLogNormal);
impl<S: RealScalar> Mean<S> for DoubleParetoLogNormal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(self.log_moment(S::ONE)?.exp())
    }
}
impl<S: RealScalar> Variance<S> for DoubleParetoLogNormal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let second = self.log_moment(number::<S>(2.)?)?;
        let twice_mean = number::<S>(2.)? * self.log_moment(S::ONE)?;
        finite(logdiff(second, twice_mean, scioxide_core::argument!("self"))?.exp())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // Reference: ln S(x) = ln φ(a) − ln x + ln B, with
    // ln B = a h/2 + (a²/24 − 1/6) h² − a h³/24 for h = 1/x.
    fn reference_log_sf(a: f64, x: f64) -> f64 {
        let normal = super::super::Normal::<f64>::standard();
        let h = 1.0 / x;
        normal.log_pdf(a).unwrap() - x.ln() + a * h / 2.0 + (a * a / 24.0 - 1.0 / 6.0) * h * h
            - a * h * h * h / 24.0
            - normal.log_cdf(a).unwrap()
    }

    #[test]
    fn generalized_normal_tiny_shape_overflows() {
        for shape in [1e-310_f64, f64::from_bits(1)] {
            let error = GeneralizedNormal::new(shape).unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::NumericalInstability,
                "{error}"
            );
        }
        assert!(GeneralizedNormal::new(1e-300_f64).is_ok());
    }

    #[test]
    fn alpha_far_upper_tail_keeps_the_shape_correction() {
        for (a, x) in [(1.0, 1e9), (50.0, 1e8), (0.5, 1e12), (100.0, 7e7)] {
            let law = Alpha::new(a).unwrap();
            let got = law.log_sf(x).unwrap();
            let expected = reference_log_sf(a, x);
            assert!(
                (got - expected).abs() <= 1e-14 * expected.abs(),
                "a = {a}, x = {x}: {got} vs {expected}"
            );
        }
    }
}
