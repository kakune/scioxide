//! Finite mixtures of continuous laws.
use crate::law::{argument, invalid, logadd};
use crate::{
    Cdf, ContinuousLaw, Density, Distribution, InverseSurvival, LogCdf, LogDensity, LogSurvival,
    Mean, Quantile, Rng, Support, Survival, Variance, WeightedSampler,
};
use alloc::vec::Vec;
use scioxide_core::{ExtendedInterval, RealScalar, Scaled, ScaledAccumulator, SciOxideError};
/// Finite mixture of continuous laws with weights `w₁, …, wₙ`.
///
/// With `pᵢ = wᵢ / Σⱼ wⱼ`, the mixture has density `Σᵢ pᵢ fᵢ(x)` and distribution function
/// `Σᵢ pᵢ Fᵢ(x)`. Both are summed in log form, and each tail is summed from the components'
/// own tails, so small tail probabilities keep their relative accuracy. The mean is
/// `Σᵢ pᵢ μᵢ`, and the variance follows from the law of total variance. Each query costs
/// O(n) component evaluations. Weighted moments combine scales before final rounding,
/// so a deviation need not be squared in the scalar range before applying its weight.
/// Quantiles are found by numerical inversion within the
/// [work limits](crate#work-limits). A sample picks a component with Walker's alias method in
/// O(1) and draws from it. Entropy is not provided, because it needs numerical integration.
/// Where a component's density is infinite at an endpoint, so is the mixture's.
///
/// Components of one concrete type can be sampled. For laws of different types, use
/// `D = &dyn ContinuousLaw<S>`. Such a mixture can be evaluated but not sampled, because
/// [`Distribution`] is not dyn compatible (see [`ContinuousLaw`]).
///
/// # Examples
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, Mixture, Normal, Quantile, SeedableRng, SplitMix64,
/// };
/// let parts = vec![Normal::new(-1.0_f64, 1.0)?, Normal::new(1.0, 1.0)?];
/// let law = Mixture::new(parts, &[1.0, 1.0])?;
/// // An equal mixture symmetric about 0 has median 0 and mean 0.
/// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-15);
/// assert!(law.mean()?.abs() < 1e-15);
/// let x = law.quantile(0.9)?;
/// assert!((law.cdf(x)? - 0.9).abs() < 1e-12);
/// let mut rng = SplitMix64::from_seed(84);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. J. Walker, "An efficient method for generating discrete random variables with general
/// distributions", *ACM Trans. Math. Softw.* 3(3), 1977, 253–256.
#[derive(Debug, Clone)]
pub struct Mixture<D, S: RealScalar> {
    options: crate::LawOptions<S>,
    components: Vec<D>,
    log_weights: Vec<S>,
    sampler: WeightedSampler<S>,
}
impl<S: RealScalar, D: ContinuousLaw<S>> Mixture<D, S> {
    /// Creates the mixture of `components` with nonnegative weights `weights`.
    ///
    /// The weights need not sum to one. Components with zero weight are ignored.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `components` is empty, the lengths differ, a component with positive
    ///   weight has empty support.
    /// - The weight errors of [`WeightedSampler::new`]: a negative or non-finite weight, or no
    ///   positive weight.
    /// - `ConvergenceFailure` if there are more components than the work limit.
    /// - `ResourceLimit` or `AllocationFailure` if storage exceeds the byte limit or cannot be
    ///   allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{Distribution, Mixture, Normal, SeedableRng, SplitMix64};
    /// let parts = vec![Normal::new(-2.0_f64, 1.0)?, Normal::new(3.0, 1.0)?];
    /// let law = Mixture::new(parts, &[1.0, 2.0])?;
    /// let mut rng = SplitMix64::from_seed(42);
    /// assert!(law.sample(&mut rng)?.is_finite());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(components: Vec<D>, weights: &[S]) -> Result<Self, SciOxideError> {
        Self::with_options(components, weights, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        components: Vec<D>,
        weights: &[S],
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if components.is_empty() || components.len() != weights.len() {
            return Err(invalid(
                scioxide_core::argument!("weights"),
                "mixture requires one weight per component",
            ));
        }
        options.admit::<S>(components.len())?;
        options.admit::<usize>(components.len())?;
        if components.len() > options.work_budget().get() {
            return Err(crate::law::exhausted());
        }
        let (sampler, log_weights) = WeightedSampler::with_log_weights(weights)?;
        for (law, &weight) in components.iter().zip(weights) {
            if weight > S::ZERO && law.support()?.is_empty() {
                return Err(invalid(
                    scioxide_core::argument!("components"),
                    "empty mixture component",
                ));
            }
        }
        Ok(Self {
            options,
            components,
            log_weights,
            sampler,
        })
    }
    fn sum_log(
        &self,
        x: S,
        left: &mut usize,
        mut query: impl FnMut(&D, S, &mut usize) -> Result<S, SciOxideError>,
    ) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        self.components
            .iter()
            .zip(&self.log_weights)
            .filter(|(_, w)| w.is_finite())
            .try_fold(S::NEG_INFINITY, |sum, (law, w)| {
                crate::law::charge(left)?;
                let term = *w + query(law, x, left)?;
                // Core log sums reject positive infinity; a mixture density can
                // inherit this exact endpoint limit. Tail callers validate their
                // result separately, so this cannot admit an infinite probability.
                if term == S::INFINITY || (sum == S::INFINITY && !term.is_nan()) {
                    Ok(S::INFINITY)
                } else {
                    logadd(sum, term)
                }
            })
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Support<S> for Mixture<D, S> {
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.components
            .iter()
            .zip(&self.log_weights)
            .filter(|(_, w)| w.is_finite())
            .try_fold(ExtendedInterval::empty(), |h, (law, _)| {
                Ok(h.hull(&law.support_in(interval)?))
            })
    }

    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        self.components
            .iter()
            .zip(&self.log_weights)
            .filter(|(_, w)| w.is_finite())
            .try_fold(ExtendedInterval::empty(), |h, (law, _)| {
                Ok(h.hull(&law.support()?))
            })
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        for (law, w) in self.components.iter().zip(&self.log_weights) {
            if w.is_finite() && law.contains(x)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> LogDensity<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S>> Mixture<D, S> {
    fn log_pdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        self.sum_log(x, left, |law, x, left| {
            crate::law::scoped(left, |context| law.log_pdf_scoped(x, context))
        })
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> LogCdf<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S>> Mixture<D, S> {
    fn tail_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        Ok(crate::law::smaller_tail_first(upper, upper, |upper| {
            self.sum_log(x, left, |law, x, left| {
                crate::law::scoped(left, |context| {
                    if upper {
                        law.log_sf_scoped(x, context)
                    } else {
                        law.log_cdf_scoped(x, context)
                    }
                })
            })
        })?
        .0)
    }
}

impl<S: RealScalar, D: ContinuousLaw<S>> LogSurvival<S> for Mixture<D, S> {
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

impl<S: RealScalar, D: ContinuousLaw<S>> Density<S> for Mixture<D, S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        let value = self.log_pdf(x)?;
        crate::law::density_from_log(value)
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Cdf<S> for Mixture<D, S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Survival<S> for Mixture<D, S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf(x)?.exp())
    }
}
impl<S: RealScalar, D: ContinuousLaw<S>> Quantile<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S>> Mixture<D, S> {
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

impl<S: RealScalar, D: ContinuousLaw<S>> InverseSurvival<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S>> Mixture<D, S> {
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

impl<S: RealScalar, D: ContinuousLaw<S> + Distribution<S>> Distribution<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S> + Distribution<S>> Mixture<D, S> {
    fn draw<R: Rng<S>>(&self, rng: &mut R, left: &mut usize) -> Result<S, SciOxideError> {
        let mut rng = crate::distribution::words::Words::forwarding(
            rng,
            self.options.sample_word_budget().get(),
        );
        let index = self.sampler.sample_index(&mut rng)?;
        crate::law::scoped(left, |context| {
            self.components[index].sample_scoped(&mut rng, context)
        })
    }
}

impl<S: RealScalar, D: ContinuousLaw<S> + Mean<S>> Mean<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S> + Mean<S>> Mixture<D, S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        self.mean_scaled_work(left)?.try_value_to()
    }

    fn mean_scaled_work(&self, left: &mut usize) -> Result<Scaled<f64>, SciOxideError> {
        let mut sum = ScaledAccumulator::ZERO;
        for (law, weight) in self
            .components
            .iter()
            .zip(&self.log_weights)
            .filter(|(_, w)| w.is_finite())
        {
            let mean = crate::law::scoped(left, |context| law.mean_scoped(context))?;
            sum.add_product(
                Scaled::from_log(1.0, weight.into_f64())?,
                Scaled::from_value(mean.into_f64())?,
            )?;
        }
        sum.total()
    }
}

impl<S: RealScalar, D: ContinuousLaw<S> + Mean<S> + Variance<S>> Variance<S> for Mixture<D, S> {
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
impl<S: RealScalar, D: ContinuousLaw<S> + Mean<S> + Variance<S>> Mixture<D, S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        // Law of total variance: sum w_i (Var_i + (mu_i - mu)^2). Using
        // deviations from the mixture mean avoids E[X^2] - mu^2 cancellation.
        let mean = self.mean_scaled_work(left)?;
        let mut sum = ScaledAccumulator::ZERO;
        for (law, weight) in self
            .components
            .iter()
            .zip(&self.log_weights)
            .filter(|(_, w)| w.is_finite())
        {
            let component_mean = crate::law::scoped(left, |context| law.mean_scoped(context))?;
            let deviation = Scaled::from_value(component_mean.into_f64())?.sub(mean)?;
            let variance = crate::law::scoped(left, |context| law.variance_scoped(context))?;
            let total = Scaled::from_value(variance.into_f64())?.add(deviation.square()?)?;
            sum.add_product(Scaled::from_log(1.0, weight.into_f64())?, total)?;
        }
        sum.total()?.try_value_to()
    }
}
