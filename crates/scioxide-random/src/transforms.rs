//! Affine maps, truncation and order statistics of continuous laws.
use crate::law::{argument, finite, invalid, logdiff};
use crate::{
    Cdf, Density, Distribution, Entropy, InverseSurvival, LogCdf, LogDensity, LogSurvival, Mean,
    Quantile, Rng, Support, Survival, Variance,
};
use scioxide_core::{ExtendedInterval, RealScalar, SciOxideError};

/// Continuous law that evaluates its density, tails and support, as a dyn-compatible trait.
///
/// It combines [`Density`], [`LogDensity`], [`Cdf`], [`Survival`], [`LogCdf`], [`LogSurvival`]
/// and [`Support`], and every type with those traits implements it. Combinators such as
/// [`Mixture`](crate::Mixture) can therefore hold different laws as `&dyn ContinuousLaw<S>`.
/// [`Distribution`](crate::Distribution) is not dyn compatible, so such a combination can be
/// evaluated but not sampled. To sample, use components of one concrete type:
///
/// ```compile_fail
/// use scioxide_random::{
///     ContinuousLaw, Distribution, Exponential, Mixture, SeedableRng, SplitMix64,
///     Uniform,
/// };
/// let (uniform, exponential) = (Uniform::new(0.0_f64, 1.0)?, Exponential::new(1.0)?);
/// let parts: Vec<&dyn ContinuousLaw<f64>> = vec![&uniform, &exponential];
/// let law = Mixture::new(parts, &[1.0, 1.0])?;
/// law.sample(&mut SplitMix64::from_seed(1))?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Evaluation works:
///
/// ```
/// use scioxide_random::{Cdf, ContinuousLaw, Exponential, Mixture, Uniform};
/// let (uniform, exponential) = (Uniform::new(0.0_f64, 1.0)?, Exponential::new(1.0)?);
/// let parts: Vec<&dyn ContinuousLaw<f64>> = vec![&uniform, &exponential];
/// let law = Mixture::new(parts, &[1.0, 1.0])?;
/// assert!((law.cdf(1.0)? - (2.0 - (-1.0_f64).exp()) / 2.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait ContinuousLaw<S: RealScalar>:
    Density<S> + LogDensity<S> + Cdf<S> + Survival<S> + LogCdf<S> + LogSurvival<S> + Support<S>
{
}
impl<
    S: RealScalar,
    D: Density<S>
        + LogDensity<S>
        + Cdf<S>
        + Survival<S>
        + LogCdf<S>
        + LogSurvival<S>
        + Support<S>
        + ?Sized,
> ContinuousLaw<S> for D
{
}

/// Law of `Y = μ + σX` for a law of `X`, a location `μ` and a nonzero scale `σ`.
///
/// The density is `f((y − μ)/σ) / |σ|`, and for `σ > 0` the distribution function is
/// `F((y − μ)/σ)`. A negative `σ` reflects the law, which swaps the lower and upper tails and the
/// quantile and inverse survival functions. The mean is `μ + σ E[X]`, the variance `σ² Var X`
/// and the entropy `H(X) + ln |σ|`. Each operation is available when the base law provides it,
/// and costs one evaluation of the base law. Tails are taken from the base law's matching tail,
/// so they keep its relative accuracy.
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     Affine, Cdf, Distribution, GumbelRight, Mean, Normal, Quantile, SeedableRng,
///     SplitMix64, Variance,
/// };
/// let law = Affine::new(Normal::new(0.0_f64, 1.0)?, 1.0, 2.0)?;
/// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-15);
/// assert!((law.quantile(0.5)? - 1.0).abs() < 1e-15);
/// assert!((law.variance()? - 4.0).abs() < 1e-15);
/// // Scale -1 reflects the right Gumbel law into the left-skew one.
/// let gumbel_l = Affine::new(GumbelRight::<f64>::new(), 0.0, -1.0)?;
/// assert!((gumbel_l.mean()? + 0.577_215_664_901_532_9).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(81);
/// assert!(gumbel_l.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Affine<D, S> {
    law: D,
    location: S,
    scale: S,
}
impl<D, S: RealScalar> Affine<D, S> {
    /// Creates the law of `location + scale · X` for `X` with law `law`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `location` or `scale` is NaN or infinite.
    /// - `DomainError` if `scale` is zero.
    pub fn new(law: D, location: S, scale: S) -> Result<Self, SciOxideError> {
        argument(location, "location")?;
        argument(scale, "scale")?;
        if scale == S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("scale"),
                "affine scale must be nonzero",
            ));
        }
        Ok(Self {
            law,
            location,
            scale,
        })
    }
    fn inverse_point(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let d = x - self.location;
        finite(if d.is_finite() {
            d / self.scale
        } else {
            x / self.scale - self.location / self.scale
        })
    }
    fn point(&self, x: S) -> Result<S, SciOxideError> {
        if x.is_infinite() {
            Ok(if self.scale > S::ZERO { x } else { -x })
        } else {
            finite(self.scale.mul_add(x, self.location))
        }
    }
}
impl<S: RealScalar, D: Support<S>> Support<S> for Affine<D, S> {
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        let Some((a, b)) = self.support()?.intersection(&interval).bounds() else {
            return Ok(ExtendedInterval::empty());
        };
        let a = a.map(|x| self.inverse_point(x)).transpose()?;
        let b = b.map(|x| self.inverse_point(x)).transpose()?;
        let input = if self.scale > S::ZERO {
            ExtendedInterval::new(a, b)?
        } else {
            ExtendedInterval::new(b, a)?
        };
        let Some((a, b)) = self.law.support_in(input)?.bounds() else {
            return Ok(ExtendedInterval::empty());
        };
        let a = a.map(|x| self.point(x)).transpose()?;
        let b = b.map(|x| self.point(x)).transpose()?;
        if self.scale > S::ZERO {
            ExtendedInterval::new(a, b)
        } else {
            ExtendedInterval::new(b, a)
        }
    }

    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        let (a, b) = self
            .law
            .support()?
            .bounds()
            .ok_or_else(|| invalid(scioxide_core::argument!("self"), "empty affine support"))?;
        let a = a.map(|x| self.point(x)).transpose()?;
        let b = b.map(|x| self.point(x)).transpose()?;
        if self.scale > S::ZERO {
            ExtendedInterval::new(a, b)
        } else {
            ExtendedInterval::new(b, a)
        }
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        self.law.contains(self.inverse_point(x)?)
    }
}
impl<S: RealScalar, D: LogDensity<S>> LogDensity<S> for Affine<D, S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.law.log_pdf(self.inverse_point(x)?)? - self.scale.abs().ln())
    }

    fn log_pdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        Ok(self.law.log_pdf_scoped(self.inverse_point(x)?, context)? - self.scale.abs().ln())
    }
}
impl<S: RealScalar, D: Density<S>> Density<S> for Affine<D, S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        let value = self.law.pdf(self.inverse_point(x)?)?;
        if value == S::INFINITY {
            Ok(value)
        } else {
            finite(value / self.scale.abs())
        }
    }
}
impl<S: RealScalar, D: Cdf<S> + Survival<S>> Cdf<S> for Affine<D, S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        // A negative scale reverses order, so the lower tail of the result is
        // the upper tail of the base law (and vice versa); using the base
        // law's own tail function preserves its small-probability accuracy.
        let z = self.inverse_point(x)?;
        if self.scale > S::ZERO {
            self.law.cdf(z)
        } else {
            self.law.sf(z)
        }
    }
}
impl<S: RealScalar, D: Cdf<S> + Survival<S>> Survival<S> for Affine<D, S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        let z = self.inverse_point(x)?;
        if self.scale > S::ZERO {
            self.law.sf(z)
        } else {
            self.law.cdf(z)
        }
    }
}
impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> LogCdf<S> for Affine<D, S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        let z = self.inverse_point(x)?;
        if self.scale > S::ZERO {
            self.law.log_cdf(z)
        } else {
            self.law.log_sf(z)
        }
    }

    fn log_cdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        let z = self.inverse_point(x)?;
        if self.scale > S::ZERO {
            self.law.log_cdf_scoped(z, context)
        } else {
            self.law.log_sf_scoped(z, context)
        }
    }
}
impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> LogSurvival<S> for Affine<D, S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        let z = self.inverse_point(x)?;
        if self.scale > S::ZERO {
            self.law.log_sf(z)
        } else {
            self.law.log_cdf(z)
        }
    }

    fn log_sf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        let z = self.inverse_point(x)?;
        if self.scale > S::ZERO {
            self.law.log_sf_scoped(z, context)
        } else {
            self.law.log_cdf_scoped(z, context)
        }
    }
}
impl<S: RealScalar, D: Quantile<S> + InverseSurvival<S>> Quantile<S> for Affine<D, S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        self.point(if self.scale > S::ZERO {
            self.law.quantile(p)?
        } else {
            self.law.isf(p)?
        })
    }

    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.point(if self.scale > S::ZERO {
            self.law.quantile_scoped(p, context)?
        } else {
            self.law.isf_scoped(p, context)?
        })
    }
}
impl<S: RealScalar, D: Quantile<S> + InverseSurvival<S>> InverseSurvival<S> for Affine<D, S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        self.point(if self.scale > S::ZERO {
            self.law.isf(p)?
        } else {
            self.law.quantile(p)?
        })
    }

    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.point(if self.scale > S::ZERO {
            self.law.isf_scoped(p, context)?
        } else {
            self.law.quantile_scoped(p, context)?
        })
    }
}
impl<S: RealScalar, D: Distribution<S>> Distribution<S> for Affine<D, S> {
    fn batch_byte_limit(&self, token: crate::traits::private::Token) -> usize {
        self.law.batch_byte_limit(token)
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        finite(self.scale.mul_add(self.law.sample(rng)?, self.location))
    }

    fn sample_scoped<R: Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        finite(
            self.scale
                .mul_add(self.law.sample_scoped(rng, context)?, self.location),
        )
    }
}
impl<S: RealScalar, D: Mean<S>> Mean<S> for Affine<D, S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(self.scale.mul_add(self.law.mean()?, self.location))
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        finite(
            self.scale
                .mul_add(self.law.mean_scoped(context)?, self.location),
        )
    }
}
impl<S: RealScalar, D: Variance<S>> Variance<S> for Affine<D, S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(self.law.variance()? * self.scale * self.scale)
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        finite(self.law.variance_scoped(context)? * self.scale * self.scale)
    }
}
impl<S: RealScalar, D: Entropy<S>> Entropy<S> for Affine<D, S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        finite(self.law.entropy()? + self.scale.abs().ln())
    }
    fn entropy_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        finite(self.law.entropy_scoped(context)? + self.scale.abs().ln())
    }
}

/// Law of `X` conditioned on `a < X ≤ b`, for a continuous law of `X` and finite `a < b`.
///
/// With `Z = P(a < X ≤ b)`, the density is `f(x)/Z` and the distribution function
/// `(F(x) − F(a))/Z` on `[a, b]`. `Z` and the tails are computed in log form from the smaller
/// base tail, so truncation far into a tail, where `Z` is below the smallest normal number,
/// still gives accurate results. Quantiles map `p` to the base law's quantile or inverse
/// survival function, whichever works with the smaller probability. When that probability
/// rounds away, a numerical inversion in log coordinates takes over. Sampling inverts a uniform
/// and needs the base law to implement [`Quantile`] and [`InverseSurvival`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     Cdf, Density, Distribution, Normal, SeedableRng, SplitMix64, Truncated,
/// };
/// let normal = Normal::new(0.0_f64, 1.0)?;
/// let law = Truncated::new(normal.clone(), -1.0, 1.0)?;
/// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-15);
/// let mass = normal.cdf(1.0)? - normal.cdf(-1.0)?;
/// assert!((law.pdf(0.0)? - normal.pdf(0.0)? / mass).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(82);
/// assert!(law.sample(&mut rng)?.abs() <= 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Truncated<D, S: RealScalar> {
    options: crate::LawOptions<S>,
    law: D,
    low: S,
    high: S,
    log_mass: S,
    /// `ln F(low)` and `ln S(low)` of the base law.
    log_lower: S,
    low_survival: S,
    /// `ln F(high)` and `ln S(high)` of the base law.
    high_cdf: S,
    log_upper: S,
}
impl<S: RealScalar, D: ContinuousLaw<S>> Truncated<D, S> {
    /// Creates the law of `X` conditioned on `low < X ≤ high`.
    ///
    /// The support is the part of the base support inside `[low, high]`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `low` or `high` is NaN or infinite.
    /// - `DomainError` if `low ≥ high`.
    /// - `InvalidInput` if `P(low < X ≤ high)` is zero or not representable even in log form,
    ///   or the truncated support is empty.
    /// - `ConvergenceFailure` if evaluating the base tails exhausts the work limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{Normal, Quantile, Truncated};
    /// let law = Truncated::new(Normal::new(0.0_f64, 1.0)?, 8.0, 9.0)?;
    /// assert!((8.0..=9.0).contains(&law.quantile(0.5)?));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(law: D, low: S, high: S) -> Result<Self, SciOxideError> {
        Self::with_options(law, low, high, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        law: D,
        low: S,
        high: S,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(low, "low")?;
        argument(high, "high")?;
        if low >= high {
            return Err(crate::law::domain(
                scioxide_core::argument!("high"),
                "truncation bounds must be increasing",
            ));
        }
        // The retained mass is cached in log form so truncation far into a tail
        // (mass below the smallest normal float) still normalizes correctly.
        let mut left = options.work_budget().get();
        let log_mass = crate::law::interval_with_work(&law, low, high, &mut left)?;
        if !log_mass.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("law"),
                "truncation requires resolvable positive probability",
            ));
        }
        let hull = law.support_in(ExtendedInterval::new(Some(low), Some(high))?)?;
        let (a, b) = hull
            .bounds()
            .ok_or_else(|| invalid(scioxide_core::argument!("law"), "empty truncation support"))?;
        let log_lower = crate::law::scoped(&mut left, |context| law.log_cdf_scoped(low, context))?;
        let low_survival =
            crate::law::scoped(&mut left, |context| law.log_sf_scoped(low, context))?;
        let high_cdf = crate::law::scoped(&mut left, |context| law.log_cdf_scoped(high, context))?;
        let log_upper = crate::law::scoped(&mut left, |context| law.log_sf_scoped(high, context))?;
        Ok(Truncated {
            options,
            law,
            low: a
                .ok_or_else(|| invalid(scioxide_core::argument!("low"), "unbounded truncation"))?,
            high: b
                .ok_or_else(|| invalid(scioxide_core::argument!("high"), "unbounded truncation"))?,
            log_mass,
            log_lower,
            low_survival,
            high_cdf,
            log_upper,
        })
    }
}

impl<S: RealScalar, D: Support<S>> Support<S> for Truncated<D, S> {
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.law.support_in(self.support()?.intersection(&interval))
    }

    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        ExtendedInterval::new(Some(self.low), Some(self.high))
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        Ok(x >= self.low && x <= self.high && self.law.contains(x)?)
    }
}
impl<S: RealScalar, D: LogDensity<S>> LogDensity<S> for Truncated<D, S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_pdf_work(x, &mut self.options.work_budget().get())
    }

    fn log_pdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_pdf_work(x, left)
        })
    }
}
impl<S: RealScalar, D: LogDensity<S>> Truncated<D, S> {
    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        Ok(if x < self.low || x > self.high {
            S::NEG_INFINITY
        } else {
            crate::law::scoped(left, |context| self.law.log_pdf_scoped(x, context))? - self.log_mass
        })
    }
}

impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> LogCdf<S> for Truncated<D, S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.tail_work(x, false, &mut self.options.work_budget().get())
    }

    fn log_cdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.tail_work(x, false, left)
        })
    }
}
impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> Truncated<D, S> {
    fn tail_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if x <= self.low || x >= self.high {
            return Ok(if (x <= self.low) == upper {
                S::ZERO
            } else {
                S::NEG_INFINITY
            });
        }
        Ok(crate::law::smaller_tail_first(upper, upper, |upper| {
            Ok(self.inner_interval(x, upper, left)? - self.log_mass)
        })?
        .0)
    }
    /// `ln P(low < X <= x)` (`upper`: `ln P(x < X <= high)`) of the base law,
    /// subtracting on the side where probabilities are small like
    /// [`crate::law::interval_with_work`], with the bound's tails cached.
    fn inner_interval(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let lc = |left: &mut usize| {
            crate::law::scoped(left, |context| self.law.log_cdf_scoped(x, context))
        };
        let ls = |left: &mut usize| {
            crate::law::scoped(left, |context| self.law.log_sf_scoped(x, context))
        };
        let law = scioxide_core::argument!("law");
        if upper {
            if self.high_cdf < -S::ONE {
                logdiff(self.high_cdf, lc(left)?, law)
            } else {
                logdiff(ls(left)?, self.log_upper, law)
            }
        } else {
            let cdf = lc(left)?;
            if cdf < -S::ONE {
                logdiff(cdf, self.log_lower, law)
            } else {
                logdiff(self.low_survival, ls(left)?, law)
            }
        }
    }
}

impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> LogSurvival<S> for Truncated<D, S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.tail_work(x, true, &mut self.options.work_budget().get())
    }

    fn log_sf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.tail_work(x, true, left)
        })
    }
}

/// Law of the `k`-th smallest of `n` independent draws from a continuous law.
///
/// With `F` and `f` the distribution function and density of the base law, the `k`-th order
/// statistic `X₍ₖ₎` has
///
/// ```text
/// P(X₍ₖ₎ ≤ x) = I_{F(x)}(k, n − k + 1),
/// f₍ₖ₎(x) = f(x) F(x)^(k−1) (1 − F(x))^(n−k) / B(k, n − k + 1),
/// ```
///
/// where `I` is the regularized incomplete beta function and `B` the beta function (David and
/// Nagaraja 2003, §2.1). The incomplete beta function is evaluated from whichever base tail is
/// smaller, using `I_y(a, b) = 1 − I_{1−y}(b, a)`, so both tails keep their relative accuracy.
/// Quantiles and samples invert the distribution function numerically, so the base law needs
/// no sampler. The support is that of the base law.
/// A singular density limit is preserved when its factors determine it;
/// indeterminate products at a support endpoint return numerical `Breakdown`.
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, OrderStatistic, SeedableRng, SplitMix64, Uniform,
/// };
/// // The maximum of three standard uniforms has CDF x^3.
/// let law = OrderStatistic::new(Uniform::new(0.0_f64, 1.0)?, 3, 3)?;
/// assert!((law.cdf(0.5)? - 0.125).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(83);
/// assert!((0.0..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. A. David and H. N. Nagaraja, *Order Statistics*, 3rd ed., Wiley, 2003.
#[derive(Debug, Clone)]
pub struct OrderStatistic<D, S: RealScalar> {
    options: crate::LawOptions<S>,
    law: D,
    a: S,
    b: S,
}
impl<S: RealScalar, D: ContinuousLaw<S>> OrderStatistic<D, S> {
    /// Creates the law of the `rank`-th smallest of `count` draws from `law`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `rank` is outside `1..=count`.
    /// - `InvalidInput` if `rank` or `count − rank + 1` is not exactly representable in `S`,
    ///   or the base support is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{OrderStatistic, Quantile, Uniform};
    /// let median = OrderStatistic::new(Uniform::new(0.0_f64, 1.0)?, 2, 3)?;
    /// assert!((median.quantile(0.5)? - 0.5).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(law: D, rank: usize, count: usize) -> Result<Self, SciOxideError> {
        Self::with_options(law, rank, count, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        law: D,
        rank: usize,
        count: usize,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if rank == 0 || rank > count {
            return Err(crate::law::domain(
                scioxide_core::argument!("rank"),
                "order rank must be in 1..=count",
            ));
        }
        let a = S::try_from_usize_exact(rank)?;
        let b = S::try_from_usize_exact(count - rank + 1)?;
        if law.support()?.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("law"),
                "empty order-statistic support",
            ));
        }
        Ok(OrderStatistic { options, law, a, b })
    }
}

impl<S: RealScalar, D: Support<S>> Support<S> for OrderStatistic<D, S> {
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.law.support_in(interval)
    }

    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.law.support()
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        self.law.contains(x)
    }
}
impl<S: RealScalar, D: LogDensity<S> + LogCdf<S> + LogSurvival<S>> LogDensity<S>
    for OrderStatistic<D, S>
{
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_pdf_work(x, &mut self.options.work_budget().get())
    }

    fn log_pdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_pdf_work(x, left)
        })
    }
}
impl<S: RealScalar, D: LogDensity<S> + LogCdf<S> + LogSurvival<S>> OrderStatistic<D, S> {
    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        let f = crate::law::scoped(left, |context| self.law.log_pdf_scoped(x, context))?;
        if f == S::NEG_INFINITY {
            return Ok(f);
        }
        let lower = if self.a == S::ONE {
            S::ZERO
        } else {
            (self.a - S::ONE)
                * crate::law::scoped(left, |context| self.law.log_cdf_scoped(x, context))?
        };
        let upper = if self.b == S::ONE {
            S::ZERO
        } else {
            (self.b - S::ONE)
                * crate::law::scoped(left, |context| self.law.log_sf_scoped(x, context))?
        };
        crate::law::checked_log_density(
            f + lower + upper - scioxide_special::ln_beta(self.a, self.b)?,
        )
    }
}

impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> OrderStatistic<D, S> {
    fn log_tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        // F_(k)(x) = I_{F(x)}(k, n - k + 1). Evaluate from whichever base tail
        // is smaller, using the reflection I_y(a, b) = 1 - I_{1-y}(b, a), so the
        // incomplete-beta argument never sits next to 1.
        let lc = crate::law::scoped(left, |context| self.law.log_cdf_scoped(x, context))?;
        let ls = crate::law::scoped(left, |context| self.law.log_sf_scoped(x, context))?;
        // For tiny endpoints, use the leading incomplete-beta series term;
        // omitted correction is bounded by (a+b)*min(F,SF).
        let tiny = S::EPSILON / (self.a + self.b);
        let (a, b, l, complement) = if lc < ls {
            (self.a, self.b, lc, upper)
        } else {
            (self.b, self.a, ls, !upper)
        };
        if l < tiny.ln() {
            let result = a * l - a.ln() - scioxide_special::ln_beta(a, b)?;
            if complement {
                crate::law::log1mexp(result)
            } else {
                Ok(result)
            }
        } else if complement {
            crate::law::special(left, |budget| {
                scioxide_special::beta_inc_log_q_with_options(
                    a,
                    b,
                    l.exp(),
                    scioxide_special::BetaOptions::new(budget),
                )
            })
        } else {
            crate::law::special(left, |budget| {
                scioxide_special::beta_inc_log_with_options(
                    a,
                    b,
                    l.exp(),
                    scioxide_special::BetaOptions::new(budget),
                )
            })
        }
    }
}
impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> LogCdf<S> for OrderStatistic<D, S> {
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
impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> OrderStatistic<D, S> {
    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_tail(x, false, left)
    }
}

impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> LogSurvival<S> for OrderStatistic<D, S> {
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
impl<S: RealScalar, D: LogCdf<S> + LogSurvival<S>> OrderStatistic<D, S> {
    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.log_tail(x, true, left)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> Density<S> for OrderStatistic<D, S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::density_from_log(self.log_pdf(x)?)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Cdf<S> for OrderStatistic<D, S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Survival<S> for OrderStatistic<D, S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Quantile<S> for OrderStatistic<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S>> OrderStatistic<D, S> {
    fn quantile_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::inversion::inverse_remaining(
            crate::Support::support(self)?,
            p,
            false,
            &crate::inversion::law_options(self.options.work_budget())?,
            None,
            left,
            |x, upper, left| {
                crate::law::scoped(left, |context| {
                    if upper {
                        self.log_sf_scoped(x, context)
                    } else {
                        self.log_cdf_scoped(x, context)
                    }
                })
            },
        )
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> InverseSurvival<S> for OrderStatistic<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S>> OrderStatistic<D, S> {
    fn isf_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::inversion::inverse_remaining(
            crate::Support::support(self)?,
            p,
            true,
            &crate::inversion::law_options(self.options.work_budget())?,
            None,
            left,
            |x, upper, left| {
                crate::law::scoped(left, |context| {
                    if upper {
                        self.log_sf_scoped(x, context)
                    } else {
                        self.log_cdf_scoped(x, context)
                    }
                })
            },
        )
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> Distribution<S> for OrderStatistic<D, S> {
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.quantile(crate::util::validated_open_float(rng)?)
    }

    fn sample_scoped<R: Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.quantile_scoped(crate::util::validated_open_float(rng)?, context)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> Density<S> for Truncated<D, S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::density_from_log(self.log_pdf(x)?)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Cdf<S> for Truncated<D, S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Survival<S> for Truncated<D, S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> Truncated<D, S> {
    fn invert(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO || p == S::ONE {
            return Ok(if (p == S::ZERO) != upper {
                self.low
            } else {
                self.high
            });
        }
        let (log_p, log_q) = (p.ln(), (-p).ln_1p());
        let fraction = if upper { log_q } else { log_p };
        // Determine the smaller tail without evaluating a near-one log sum:
        // its cancellation certificate can be much costlier than the inverse.
        let use_upper = if self.log_lower >= -S::LN_2 {
            true
        } else if self.log_upper >= -S::LN_2 {
            false
        } else {
            let lower = self.log_lower.exp() + (self.log_mass + fraction).exp();
            lower > S::HALF || (lower == S::HALF && upper)
        };
        let boundary = if use_upper {
            self.log_upper
        } else {
            self.log_lower
        };
        let fraction = if use_upper == upper { log_p } else { log_q };
        let target = crate::law::logadd(boundary, self.log_mass + fraction)?;
        let probability = target.exp();
        // A representable base probability avoids all conditional-CDF iteration.
        // Tiny/narrow intervals can lose the target even in the shorter tail.
        if probability < S::MIN_POSITIVE || target == boundary || probability == boundary.exp() {
            return crate::inversion::inverse_remaining(
                crate::Support::support(self)?,
                p,
                upper,
                &crate::inversion::law_options(self.options.work_budget())?,
                None,
                left,
                |x, upper, left| {
                    crate::law::scoped(left, |context| {
                        if upper {
                            self.log_sf_scoped(x, context)
                        } else {
                            self.log_cdf_scoped(x, context)
                        }
                    })
                },
            );
        }
        let mut x = if !use_upper {
            crate::law::scoped(left, |context| {
                self.law.quantile_scoped(probability, context)
            })?
        } else {
            crate::law::scoped(left, |context| self.law.isf_scoped(probability, context))?
        };
        if x < self.low || x > self.high {
            return crate::inversion::inverse_remaining(
                crate::Support::support(self)?,
                p,
                upper,
                &crate::inversion::law_options(self.options.work_budget())?,
                None,
                left,
                |x, upper, left| {
                    crate::law::scoped(left, |context| {
                        if upper {
                            self.log_sf_scoped(x, context)
                        } else {
                            self.log_cdf_scoped(x, context)
                        }
                    })
                },
            );
        }
        // Switching inverse direction can select the opposite end of a flat
        // CDF segment. Clip the positive-probability support on the requested
        // side to recover its endpoint without iterating over the CDF.
        if use_upper != upper {
            // Base inverse coordinates have the shared four-ULP boundary
            // allowance; inspect beyond that rounding neighborhood.
            let toward = if upper { S::INFINITY } else { S::NEG_INFINITY };
            let neighbor = (0..4).fold(x, |value, _| value.next_after(toward));
            if neighbor > self.low
                && neighbor < self.high
                && crate::law::scoped(left, |context| self.law.log_pdf_scoped(neighbor, context))?
                    == S::NEG_INFINITY
            {
                let interval = if upper {
                    ExtendedInterval::new(Some(neighbor), Some(self.high))?
                } else {
                    ExtendedInterval::new(Some(self.low), Some(neighbor))?
                };
                if let Some((low, high)) = self.law.support_in(interval)?.bounds() {
                    x = if upper { low } else { high }.unwrap_or(x);
                }
            }
        }
        finite(x)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> Quantile<S>
    for Truncated<D, S>
{
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
impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> Truncated<D, S> {
    fn quantile_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.invert(p, false, left)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> InverseSurvival<S>
    for Truncated<D, S>
{
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
impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> Truncated<D, S> {
    fn isf_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.invert(p, true, left)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> Distribution<S>
    for Truncated<D, S>
{
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.quantile(crate::util::validated_open_float(rng)?)
    }

    fn sample_scoped<R: Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.quantile_scoped(crate::util::validated_open_float(rng)?, context)
    }
}
