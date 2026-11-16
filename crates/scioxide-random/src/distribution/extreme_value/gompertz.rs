//! Gompertz distribution.
use crate::distribution::LawOptions;
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::distribution::formula::positive_support;
use crate::distribution::moment_math::quantile_mean;
use crate::distribution::moment_math::quantile_variance;
use crate::law::positive;
use crate::law::{finite, log1mexp};
use crate::{Mean, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Gompertz distribution on `x ≥ 0` with shape `c > 0` and unit scale.
///
/// The survival function is `S(x) = exp(−c (e^x − 1))`. It is a left-skewed
/// Gumbel law conditioned on `x ≥ 0`. The mean and variance are computed by
/// quadrature, within the [work limits](crate#work-limits).
///
/// ```
/// use scioxide_random::{
///     Distribution, Gompertz, Mean, Quantile, SeedableRng, SplitMix64, Survival,
/// };
/// let law = Gompertz::<f64>::new(2.0)?;
/// assert!((law.sf(1.0)? - (-2.0 * (1.0_f64.exp() - 1.0)).exp()).abs() < 1e-14);
/// let x = law.quantile(0.25)?;
/// assert!((law.sf(x)? - 0.75).abs() < 1e-14);
/// assert!(law.mean()? > 0.0);
/// let mut rng = SplitMix64::from_seed(14);
/// assert!(law.sample(&mut rng)? >= 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Gompertz<S: RealScalar> {
    options: LawOptions<S>,
    shape: S,
}
impl<S: RealScalar> Gompertz<S> {
    fn rate(&self, x: S) -> Result<S, SciOxideError> {
        let exponential = x.exp_m1();
        Ok(if exponential.is_finite() {
            self.shape * exponential
        } else {
            (self.shape.ln() + crate::law::log_expm1(x)?).exp()
        })
    }

    fn inverse_rate(&self, rate: S) -> Result<S, SciOxideError> {
        let quotient = rate / self.shape;
        finite(if quotient.is_finite() {
            quotient.ln_1p()
        } else {
            // The quotient exceeds the scalar range; ln(1+q) = ln(q) to working precision.
            rate.ln() - self.shape.ln()
        })
    }

    /// Creates a Gompertz distribution with shape `c = shape`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        Self::with_options(shape, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(shape: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self { options, shape })
    }
}
impl<S: RealScalar> Formula<S> for Gompertz<S> {
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
        Ok(if x < S::ZERO {
            S::NEG_INFINITY
        } else {
            self.shape.ln() + x - self.rate(x)?
        })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(if x <= S::ZERO {
                S::ZERO
            } else {
                -self.rate(x)?
            })
        } else {
            Ok(if x <= S::ZERO {
                S::NEG_INFINITY
            } else {
                // S(x) = exp(-c (e^x - 1)); `expm1` keeps small-x tails exact.
                log1mexp(-self.rate(x)?)?
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            Ok(S::INFINITY)
        } else {
            self.inverse_rate(-(-p).ln_1p())
        }
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            Ok(S::INFINITY)
        } else {
            self.inverse_rate(-p.ln())
        }
    }
}
formula_law!(Gompertz);
impl<S: RealScalar> Mean<S> for Gompertz<S> {
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
impl<S: RealScalar> Gompertz<S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let options = self.options;
        quantile_mean(self, options, left)
    }
}

impl<S: RealScalar> Variance<S> for Gompertz<S> {
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
impl<S: RealScalar> Gompertz<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let options = self.options;
        quantile_variance(self, options, left)
    }
}
