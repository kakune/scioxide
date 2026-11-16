//! Laws of `|X|`, `exp(X)` and `ln(X)` for a continuous law of `X`.
use crate::law::{argument, finite, invalid, logadd};
use crate::{
    Cdf, ContinuousLaw, Density, Distribution, InverseSurvival, LogCdf, LogDensity, LogSurvival,
    Quantile, Rng, Support, Survival,
};
use scioxide_core::{ExtendedInterval, RealScalar, SciOxideError};
#[derive(Debug, Clone, Copy)]
enum Transform {
    Absolute,
    Exponential,
}
/// Law of `|X|` or `exp(X)` for a continuous law of `X` with density `f` and distribution
/// function `F`.
///
/// [`Self::abs`] gives the folded law of `Y = |X|`, with density `f(y) + f(−y)` for `y ≥ 0`,
/// upper tail `P(X < −y) + P(X > y)` and lower tail `P(−y ≤ X ≤ y)`. [`Self::exp`] gives the law
/// of `Y = exp(X)`, with density `f(ln y)/y` and distribution function `F(ln y)` for `y > 0`.
/// The density and tails cost a few evaluations of the base law. When the lower tail of `|X|`
/// is below `√ε`, it is integrated by quadrature to keep its relative accuracy. Quantiles are
/// found by numerical inversion of the tails, and samples transform a sample of `X`.
///
/// [`Self::log`] gives the law of `ln(X)` as a [`LogTransformed`]. The marker
/// [`ExponentialTransform`] lets the law of `exp(X)` evaluate in log coordinates, so
/// `ln(exp(X))` uses the base law directly. Moments and entropy are not provided, because they
/// need more than the density and tails of the base law. Operations that iterate take the
/// [work limits](crate#work-limits) of the [`LawOptions`](crate::LawOptions).
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     Cdf, Density, Distribution, Normal, Quantile, SeedableRng, SplitMix64,
///     Transformed,
/// };
/// // exp of a standard normal is the standard log-normal law.
/// let law = Transformed::exp(Normal::new(0.0_f64, 1.0)?)?;
/// assert!((law.pdf(1.0)? - 1.0 / (2.0 * core::f64::consts::PI).sqrt()).abs() < 1e-15);
/// let x = law.quantile(0.3)?;
/// assert!((law.cdf(x)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(85);
/// assert!(law.sample(&mut rng)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Transformed<D, S: RealScalar, T = ()> {
    options: crate::LawOptions<S>,
    law: D,
    kind: Transform,
    support: ExtendedInterval<S>,
    marker: core::marker::PhantomData<T>,
}
impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
    fn absolute_log_mass(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        let value = crate::law::interval_with_work(&self.law, -x, x, left)?;
        if value >= self.options.tolerance().relative().ln() {
            return Ok(value);
        }
        // Below sqrt(EPSILON) mass, subtracting nearly equal base tails loses
        // that relative accuracy. Integrate the positive density on a scaled
        // interval to the same tolerance, within the enclosing query's remaining allowance.
        Ok(x.ln()
            + crate::distribution::log_integral(
                -S::ONE,
                S::ONE,
                self.options.tolerance().relative(),
                &mut |t| {
                    crate::law::charge(left)?;
                    crate::law::scoped(left, |context| self.law.log_pdf_scoped(t * x, context))
                },
                None,
            )?)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Transformed<D, S> {
    /// Returns the law of `|X|`, combining the preimages `±y`.
    ///
    /// The support is the image of the support of `X`. For `X` on `[1, 3]` it is `[1, 3]`, and
    /// for `X` on `[−3, 1]` it is `[0, 3]`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::abs_with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the support is empty.
    /// - The error of the base law's support.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{Normal, Support, Transformed};
    /// let folded = Transformed::abs(Normal::new(0.0_f64, 1.0)?)?;
    /// assert!(!folded.contains(-1.0)?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn abs(law: D) -> Result<Self, SciOxideError> {
        Self::abs_with_options(law, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::abs`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::abs`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn abs_with_options(law: D, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        prepare(
            law,
            options,
            Transform::Absolute,
            scioxide_core::argument!("law"),
        )
    }
    /// Returns the law of `exp(X)`, with density `f(ln y)/y`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::exp_with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the support is empty.
    /// - `NumericalInstability` if `exp` of a finite support endpoint overflows.
    /// - The error of the base law's support.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{Cdf, Normal, Transformed};
    /// let law = Transformed::exp(Normal::new(0.0_f64, 1.0)?)?;
    /// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn exp(law: D) -> Result<Transformed<D, S, ExponentialTransform>, SciOxideError> {
        Self::exp_with_options(law, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::exp`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::exp`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn exp_with_options(
        law: D,
        options: crate::LawOptions<S>,
    ) -> Result<Transformed<D, S, ExponentialTransform>, SciOxideError> {
        prepare(
            law,
            options,
            Transform::Exponential,
            scioxide_core::argument!("law"),
        )
    }
    /// Returns the law of `ln(X)` for a law with support in `[0, ∞)`.
    ///
    /// The base law must implement [`LogCoordinateLaw`], which evaluates it at `x = exp(y)`
    /// without forming `exp(y)`. The result is therefore accurate where `exp(y)` would
    /// underflow or overflow. A law without that trait does not compile:
    ///
    /// ```compile_fail
    /// use scioxide_random::{Normal, Transformed};
    /// // Normal does not implement LogCoordinateLaw.
    /// let _ = Transformed::log(Normal::new(0.0_f64, 1.0).unwrap());
    /// ```
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::log_with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the support extends below zero or the logarithmic support is empty.
    /// - The error of the base law's support.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{Cdf, LogNormal, Transformed};
    /// let law = Transformed::log(LogNormal::new(0.0_f64, 1.0)?)?;
    /// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn log(law: D) -> Result<LogTransformed<D, S>, SciOxideError>
    where
        D: LogCoordinateLaw<S>,
    {
        Self::log_with_options(law, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::log`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::log`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn log_with_options(
        law: D,
        options: crate::LawOptions<S>,
    ) -> Result<LogTransformed<D, S>, SciOxideError>
    where
        D: LogCoordinateLaw<S>,
    {
        let support = law.log_support_in(ExtendedInterval::new(None, None)?)?;
        if support.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("law"),
                "empty logarithmic support",
            ));
        }
        options.validate()?;
        Ok(LogTransformed {
            options,
            law,
            support,
        })
    }
}
fn prepare<S: RealScalar, D: ContinuousLaw<S>, T>(
    law: D,
    options: crate::LawOptions<S>,
    kind: Transform,
    argument: scioxide_core::Argument,
) -> Result<Transformed<D, S, T>, SciOxideError> {
    options.validate()?;
    let support = map_support(law.support()?, kind)?;
    if support.is_empty() {
        return Err(invalid(argument, "empty transformed support"));
    }
    // The endpoint image of the base support can be looser than the true
    // support (e.g. |X| for a law on [-3, 1] reaches 0 only if 0 is inside).
    // Recomputing through `support_in` tightens it once so later queries can
    // use the cached interval.
    let prepared = Transformed {
        options,
        law,
        kind,
        support,
        marker: core::marker::PhantomData,
    };
    let support = prepared.support_in(support)?;
    Ok(Transformed {
        support,
        ..prepared
    })
}
fn map_support<S: RealScalar>(
    interval: ExtendedInterval<S>,
    kind: Transform,
) -> Result<ExtendedInterval<S>, SciOxideError> {
    let Some((a, b)) = interval.bounds() else {
        return Ok(ExtendedInterval::empty());
    };
    let (low, high) = match kind {
        Transform::Absolute => {
            let low = match (a, b) {
                (Some(a), _) if a > S::ZERO => a,
                (_, Some(b)) if b < S::ZERO => -b,
                _ => S::ZERO,
            };
            (
                Some(low),
                match (a, b) {
                    (Some(a), Some(b)) => Some(a.abs().max(b.abs())),
                    _ => None,
                },
            )
        }
        Transform::Exponential => (
            Some(match a {
                Some(a) => positive_exp(a)?,
                None => S::ZERO,
            }),
            b.map(|x| positive_exp(x)).transpose()?,
        ),
    };
    ExtendedInterval::new(low, high)
}
fn positive_exp<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let y = finite(x.exp())?;
    if y == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "positive inverse image is unrepresentable",
        ));
    }
    Ok(y)
}
impl<S: RealScalar, D: ContinuousLaw<S>, T> Support<S> for Transformed<D, S, T> {
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        let clipped = self.support.intersection(&interval);
        let Some((a, b)) = clipped.bounds() else {
            return Ok(clipped);
        };
        if a.is_some() && a == b {
            return Ok(ExtendedInterval::empty());
        }
        let input = match self.kind {
            Transform::Absolute => {
                let lower = a.unwrap_or(S::ZERO);
                let positive = self
                    .law
                    .support_in(ExtendedInterval::new(Some(lower), b)?)?;
                let negative = self
                    .law
                    .support_in(ExtendedInterval::new(b.map(|x| -x), Some(-lower))?)?;
                return Ok(map_support(positive, self.kind)?
                    .hull(&map_support(negative, self.kind)?)
                    .intersection(&clipped));
            }
            Transform::Exponential => ExtendedInterval::new(
                a.filter(|x| *x > S::ZERO).map(|x| x.ln()),
                b.map(|x| x.ln()),
            )?,
        };
        Ok(map_support(self.law.support_in(input)?, self.kind)?.intersection(&clipped))
    }

    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        Ok(self.support)
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        if !self.support.contains(x) {
            return Ok(false);
        }
        match self.kind {
            Transform::Absolute => Ok(self.law.contains(x)? || self.law.contains(-x)?),
            Transform::Exponential => {
                if x == S::ZERO {
                    Ok(true)
                } else {
                    self.law.contains(x.ln())
                }
            }
        }
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>, T> LogDensity<S> for Transformed<D, S, T> {
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
impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        match self.kind {
            Transform::Absolute => {
                if x < S::ZERO {
                    Ok(S::NEG_INFINITY)
                } else {
                    let positive =
                        crate::law::checked_log_density(crate::law::scoped(left, |context| {
                            self.law.log_pdf_scoped(x, context)
                        })?)?;
                    let negative =
                        crate::law::checked_log_density(crate::law::scoped(left, |context| {
                            self.law.log_pdf_scoped(-x, context)
                        })?)?;
                    if positive == S::INFINITY || negative == S::INFINITY {
                        Ok(S::INFINITY)
                    } else {
                        logadd(positive, negative)
                    }
                }
            }
            Transform::Exponential => {
                if x <= S::ZERO {
                    Ok(S::NEG_INFINITY)
                } else {
                    Ok(crate::law::scoped(left, |context| {
                        self.law.log_pdf_scoped(x.ln(), context)
                    })? - x.ln())
                }
            }
        }
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>, T> LogCdf<S> for Transformed<D, S, T> {
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
impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        match self.kind {
            Transform::Absolute => {
                // P(|X| <= x) = P(-x <= X <= x), taken from the side with the
                // smaller probabilities to avoid cancellation.
                if x <= S::ZERO {
                    Ok(S::NEG_INFINITY)
                } else {
                    self.absolute_tail(x, false, left)
                }
            }
            Transform::Exponential => {
                if x <= S::ZERO {
                    Ok(S::NEG_INFINITY)
                } else {
                    crate::law::scoped(left, |context| self.law.log_cdf_scoped(x.ln(), context))
                }
            }
        }
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
    // Tails of |X| at x > 0: the lower tail is P(-x <= X <= x) and the upper
    // tail P(X < -x) + P(X > x); a near-one tail comes from its complement.
    fn absolute_tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        Ok(crate::law::smaller_tail_first(upper, upper, |upper| {
            if upper {
                logadd(
                    crate::law::scoped(left, |context| self.law.log_cdf_scoped(-x, context))?,
                    crate::law::scoped(left, |context| self.law.log_sf_scoped(x, context))?,
                )
            } else {
                self.absolute_log_mass(x, left)
            }
        })?
        .0)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>, T> LogSurvival<S> for Transformed<D, S, T> {
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
impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        match self.kind {
            Transform::Absolute => {
                if x <= S::ZERO {
                    Ok(S::ZERO)
                } else {
                    self.absolute_tail(x, true, left)
                }
            }
            Transform::Exponential => {
                if x <= S::ZERO {
                    Ok(S::ZERO)
                } else {
                    crate::law::scoped(left, |context| self.law.log_sf_scoped(x.ln(), context))
                }
            }
        }
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>, T> Density<S> for Transformed<D, S, T> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::density_from_log(self.log_pdf(x)?)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>, T> Cdf<S> for Transformed<D, S, T> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>, T> Survival<S> for Transformed<D, S, T> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>, T> Quantile<S> for Transformed<D, S, T> {
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
impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
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

impl<S: RealScalar, D: ContinuousLaw<S>, T> InverseSurvival<S> for Transformed<D, S, T> {
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
impl<S: RealScalar, D: ContinuousLaw<S>, T> Transformed<D, S, T> {
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

impl<S: RealScalar, D: ContinuousLaw<S> + Distribution<S>, T> Distribution<S>
    for Transformed<D, S, T>
{
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.draw(rng, &mut self.options.work_budget().get())
    }

    fn sample_scoped<R: Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.draw(rng, left)
        })
    }
}
impl<S: RealScalar, D: ContinuousLaw<S> + Distribution<S>, T> Transformed<D, S, T> {
    fn draw<R: Rng<S>>(&self, rng: &mut R, left: &mut usize) -> Result<S, SciOxideError> {
        let x = crate::law::scoped(left, |context| self.law.sample_scoped(rng, context))?;
        finite(match self.kind {
            Transform::Absolute => x.abs(),
            Transform::Exponential => positive_exp(x)?,
        })
    }
}

/// Type marker of the law of `exp(X)` returned by [`Transformed::exp`].
///
/// It makes that law a [`LogCoordinateLaw`] that evaluates the law of `X` directly.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialTransform;

/// Law on `[0, ∞)` that can be evaluated at `x = exp(y)` given `y = ln x`.
///
/// [`Transformed::log`] needs this trait. Implementations compute the tails `ln F(exp(y))` and
/// `ln(1 − F(exp(y)))` and the log density of `ln X`, `y + ln f(exp(y))`, without requiring
/// `exp(y)` to be representable. Plain probabilities may round to 0 or 1. If a log probability
/// is too negative to represent, the methods return `NumericalInstability` (`Overflow`), not
/// an input error or a false zero. The default support methods assume a single interval in
/// `[0, ∞)`. Laws whose support has gaps override both.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Exponential, LogCoordinateLaw};
/// let law = Exponential::new(1.0_f64)?;
/// assert!((law.log_cdf_at_log(-800.0)? + 800.0).abs() < 1e-12);
/// assert!((law.log_pushforward_pdf(-800.0)? + 800.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait LogCoordinateLaw<S: RealScalar>: ContinuousLaw<S> {
    /// Returns `ln F(exp(y))`, accurate deep in the lower tail.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `y` is NaN or infinite.
    /// - `NumericalInstability` if the result is not representable.
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError>;
    /// Computes [`Self::log_cdf_at_log`] within the work limit of `context`.
    ///
    /// The default counts one evaluation. Laws that iterate override it.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::log_cdf_at_log`].
    /// - `ConvergenceFailure` if the work limit of `context` is exhausted.
    fn log_cdf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.log_cdf_at_log(y))
    }

    /// Returns `ln(1 − F(exp(y)))`, accurate deep in the upper tail.
    ///
    /// # Errors
    ///
    /// - The conditions of [`Self::log_cdf_at_log`].
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError>;
    /// Computes [`Self::log_sf_at_log`] within the work limit of `context`.
    ///
    /// The default counts one evaluation. Laws that iterate override it.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::log_sf_at_log`].
    /// - `ConvergenceFailure` if the work limit of `context` is exhausted.
    fn log_sf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.log_sf_at_log(y))
    }

    /// Returns the log density of `ln X` at `y`: `y + ln f(exp(y))`, including the Jacobian.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `y` is NaN or infinite.
    /// - `NumericalInstability` if the result is not representable.
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError>;
    /// Computes [`Self::log_pushforward_pdf`] within the work limit of `context`.
    ///
    /// The default counts one evaluation. Laws that iterate override it.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::log_pushforward_pdf`].
    /// - `ConvergenceFailure` if the work limit of `context` is exhausted.
    fn log_pushforward_pdf_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.log_pushforward_pdf(y))
    }

    /// Returns `F(exp(y))`.
    ///
    /// Override the default if the log probability can overflow.
    ///
    /// # Errors
    ///
    /// - The conditions of [`Self::log_cdf_at_log`].
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf_at_log(y)?.exp())
    }
    /// Returns `1 − F(exp(y))`.
    ///
    /// # Errors
    ///
    /// - The conditions of [`Self::log_sf_at_log`].
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf_at_log(y)?.exp())
    }
    /// Returns the density of `ln X` at `y`, `exp(y) f(exp(y))`.
    ///
    /// # Errors
    ///
    /// - The conditions of [`Self::log_pushforward_pdf`].
    /// - `NumericalInstability` if the density overflows.
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::density_from_log(self.log_pushforward_pdf(y)?)
    }
    /// Returns the smallest interval within `interval` that contains the support of `ln X`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the support of `X` extends below zero.
    /// - The error of the law's support.
    fn log_support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        let support = logarithmic_support(self.support()?)?.intersection(&interval);
        Ok(
            if support.bounds().is_some_and(|(a, b)| a.is_some() && a == b) {
                ExtendedInterval::empty()
            } else {
                support
            },
        )
    }
    /// Returns whether `exp(y)` is in the support, without forming `exp(y)`.
    ///
    /// A NaN or infinite `y` gives `false`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the support of `X` extends below zero.
    /// - The error of the law's support.
    fn contains_at_log(&self, y: S) -> Result<bool, SciOxideError> {
        Ok(y.is_finite() && logarithmic_support(self.support()?)?.contains(y))
    }
}

pub(crate) fn logarithmic_support<S: RealScalar>(
    support: ExtendedInterval<S>,
) -> Result<ExtendedInterval<S>, SciOxideError> {
    let Some((a, b)) = support.bounds() else {
        return Ok(support);
    };
    let a = a.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("law"),
            "log transform requires nonnegative support",
        )
    })?;
    if a < S::ZERO || b.is_some_and(|b| b <= S::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("law"),
            "log transform requires positive interior support",
        ));
    }
    ExtendedInterval::new(
        if a == S::ZERO { None } else { Some(a.ln()) },
        b.map(|b| b.ln()),
    )
}

impl<S: RealScalar, D: LogCoordinateLaw<S> + ?Sized> LogCoordinateLaw<S> for &D {
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        (**self).log_cdf_at_log(y)
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        (**self).log_sf_at_log(y)
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        (**self).log_pushforward_pdf(y)
    }
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        (**self).cdf_at_log(y)
    }
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        (**self).sf_at_log(y)
    }
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        (**self).pushforward_pdf(y)
    }
    fn log_support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        (**self).log_support_in(interval)
    }
    fn contains_at_log(&self, y: S) -> Result<bool, SciOxideError> {
        (**self).contains_at_log(y)
    }

    fn log_cdf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::log_cdf_at_log_scoped(self, y, context)
    }

    fn log_sf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::log_sf_at_log_scoped(self, y, context)
    }

    fn log_pushforward_pdf_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::log_pushforward_pdf_scoped(self, y, context)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> LogCoordinateLaw<S>
    for Transformed<D, S, ExponentialTransform>
{
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.law.log_cdf(y)
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.law.log_sf(y)
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        self.law.log_pdf(y)
    }
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.law.cdf(y)
    }
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.law.sf(y)
    }
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        self.law.pdf(y)
    }
    fn log_support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.law.support_in(interval)
    }
    fn contains_at_log(&self, y: S) -> Result<bool, SciOxideError> {
        self.law.contains(y)
    }

    fn log_cdf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.law.log_cdf_scoped(y, context)
    }

    fn log_sf_at_log_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.law.log_sf_scoped(y, context)
    }

    fn log_pushforward_pdf_scoped(
        &self,
        y: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        self.law.log_pdf_scoped(y, context)
    }
}

/// Law of `ln(X)` for a law of `X` on `[0, ∞)`, returned by [`Transformed::log`].
///
/// The density, tails and support come from the [`LogCoordinateLaw`] methods of the base law,
/// so they stay accurate where `exp(y)` is not representable. Quantiles are found by numerical
/// inversion of the tails within the [work limits](crate#work-limits). Sampling takes the
/// logarithm of a sample of `X` and needs the base law to implement [`Distribution`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{Exponential, LogCdf, Transformed};
/// let law = Transformed::log(Exponential::new(1.0_f64)?)?;
/// assert!((law.log_cdf(-800.0)? + 800.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LogTransformed<D, S: RealScalar> {
    options: crate::LawOptions<S>,
    law: D,
    support: ExtendedInterval<S>,
}

impl<S: RealScalar, D: LogCoordinateLaw<S>> Support<S> for LogTransformed<D, S> {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        Ok(self.support)
    }
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.law
            .log_support_in(interval.intersection(&self.support))
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        self.law.contains_at_log(x)
    }
}
impl<S: RealScalar, D: LogCoordinateLaw<S>> LogCdf<S> for LogTransformed<D, S> {
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
impl<S: RealScalar, D: LogCoordinateLaw<S>> LogTransformed<D, S> {
    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        crate::law::scoped(left, |context| self.law.log_cdf_at_log_scoped(x, context))
    }
}

impl<S: RealScalar, D: LogCoordinateLaw<S>> LogSurvival<S> for LogTransformed<D, S> {
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
impl<S: RealScalar, D: LogCoordinateLaw<S>> LogTransformed<D, S> {
    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        crate::law::scoped(left, |context| self.law.log_sf_at_log_scoped(x, context))
    }
}

impl<S: RealScalar, D: LogCoordinateLaw<S>> LogDensity<S> for LogTransformed<D, S> {
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
impl<S: RealScalar, D: LogCoordinateLaw<S>> LogTransformed<D, S> {
    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        crate::law::scoped(left, |context| {
            self.law.log_pushforward_pdf_scoped(x, context)
        })
    }
}

impl<S: RealScalar, D: LogCoordinateLaw<S>> Cdf<S> for LogTransformed<D, S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        self.law.cdf_at_log(x)
    }
}
impl<S: RealScalar, D: LogCoordinateLaw<S>> Survival<S> for LogTransformed<D, S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        self.law.sf_at_log(x)
    }
}
impl<S: RealScalar, D: LogCoordinateLaw<S>> Density<S> for LogTransformed<D, S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        self.law.pushforward_pdf(x)
    }
}
impl<S: RealScalar, D: LogCoordinateLaw<S>> Quantile<S> for LogTransformed<D, S> {
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
impl<S: RealScalar, D: LogCoordinateLaw<S>> LogTransformed<D, S> {
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

impl<S: RealScalar, D: LogCoordinateLaw<S>> InverseSurvival<S> for LogTransformed<D, S> {
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
impl<S: RealScalar, D: LogCoordinateLaw<S>> LogTransformed<D, S> {
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

impl<S: RealScalar, D: LogCoordinateLaw<S> + Distribution<S>> Distribution<S>
    for LogTransformed<D, S>
{
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.draw(rng, &mut self.options.work_budget().get())
    }

    fn sample_scoped<R: Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.draw(rng, left)
        })
    }
}
impl<S: RealScalar, D: LogCoordinateLaw<S> + Distribution<S>> LogTransformed<D, S> {
    fn draw<R: Rng<S>>(&self, rng: &mut R, left: &mut usize) -> Result<S, SciOxideError> {
        finite(crate::law::scoped(left, |context| self.law.sample_scoped(rng, context))?.ln())
    }
}
