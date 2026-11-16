//! Relativistic Breit–Wigner distribution.
use crate::distribution::LawOptions;
use crate::distribution::formula::formula_law;
use crate::distribution::formula::{Formula, Inverse};
use crate::distribution::moment_math::checked_variance;
use crate::distribution::quadrature::positive_integral;
use crate::law::constant as number;
use crate::law::{finite, log1mexp, logadd};
use crate::{Mean, Variance};
use scioxide_core::{Budget, RealScalar, SciOxideError};

/// Relativistic Breit–Wigner distribution on `x ≥ 0` with shape `ρ > 0` and unit scale.
///
/// The density is `f(x) ∝ 1 / ((x² − ρ²)² + ρ²)`. Tail probabilities are
/// computed by quadrature, within the [work limits](crate#work-limits).
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, Quantile, RelativisticBreitWigner, SeedableRng,
///     SplitMix64,
/// };
/// let law = RelativisticBreitWigner::<f64>::new(1.0)?;
/// let x = law.quantile(0.5)?;
/// assert!((law.cdf(x)? - 0.5).abs() < 1e-8);
/// assert!(law.mean()? > 0.0);
/// let mut rng = SplitMix64::from_seed(61);
/// assert!(law.sample(&mut rng)? >= 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RelativisticBreitWigner<S: RealScalar> {
    rho: S,
    mode_scale: S,
    log_k: S,
    q: S,
    options: LawOptions<S>,
}
impl<S: RealScalar> RelativisticBreitWigner<S> {
    /// Creates a relativistic Breit–Wigner distribution with shape `ρ = rho`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `rho` is NaN or infinite.
    /// - `DomainError` if `rho ≤ 0`.
    /// - `NumericalInstability` if the normalizing constant is outside the range of `S`.
    pub fn new(rho: S) -> Result<Self, SciOxideError> {
        Self::with_options(rho, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(rho: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::positive(rho, scioxide_core::argument!("rho"))?;
        let q = finite(rho * rho.hypot(S::ONE))?;
        let mode_scale = finite(((q + rho * rho) * S::HALF).sqrt())?;
        let v = rho / (number::<S>(2.)? * mode_scale);
        let log_k = (number::<S>(4.)? / number::<S>(core::f64::consts::PI)?).ln() + q.ln() + v.ln();
        crate::law::positive_result(q)?;
        crate::law::positive_result(mode_scale)?;
        crate::law::positive_result(v)?;
        finite(log_k)?;
        Ok(Self {
            rho,
            mode_scale,
            log_k,
            q,
            options,
        })
    }
    /// Returns the options the law was created with.
    pub const fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn log_denominator(&self, x: S) -> Result<S, SciOxideError> {
        let two = S::TWO;
        let a = (x - self.rho).abs();
        let b = x + self.rho;
        logadd(two * (a.ln() + b.ln()), two * self.rho.ln())
    }
    fn direct_tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if upper && x / self.mode_scale > number::<S>(1e6)? {
            // The next relative term is O((mode_scale/x)^2). This also avoids a
            // subnormal inverse coordinate for the farthest represented x.
            return Ok(self.log_k - number::<S>(3.)?.ln() - number::<S>(3.)? * x.ln());
        }
        let tol = self.options.tolerance.relative();
        if upper {
            let end = S::ONE / x;
            let f = |y: S| {
                if y == S::ZERO {
                    return Ok(S::NEG_INFINITY);
                }
                let z = self.rho * y;
                let a = (S::ONE - z * z).abs();
                let logd = logadd(
                    number::<S>(2.)? * a.ln(),
                    number::<S>(2.)? * self.rho.ln() + number::<S>(4.)? * y.ln(),
                )?;
                Ok(self.log_k + number::<S>(2.)? * y.ln() - logd)
            };
            positive_integral(S::ZERO, end, tol, left, &f, None)
        } else {
            positive_integral(S::ZERO, x, tol, left, &|y| self.density_log(y), None)
        }
    }
}
impl<S: RealScalar> Formula<S> for RelativisticBreitWigner<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> Budget {
        self.options.work
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        let first = x > self.mode_scale;
        let value = self.direct_tail(x, first, left)?;
        Ok(if first == upper {
            value
        } else {
            log1mexp(value)?
        })
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            Ok(S::NEG_INFINITY)
        } else if x / self.rho > number::<S>(2.)? {
            let y = self.rho / x;
            let z = self.rho / x / x;
            let a = S::ONE - y * y;
            Ok(self.log_k - number::<S>(4.)? * x.ln() - (a * a + z * z).ln())
        } else {
            Ok(self.log_k - self.log_denominator(x)?)
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;
}
formula_law!(RelativisticBreitWigner);
impl<S: RealScalar> Mean<S> for RelativisticBreitWigner<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(
            self.log_k.exp() / (number::<S>(2.)? * self.rho)
                * (number::<S>(core::f64::consts::FRAC_PI_2)? + self.rho.atan()),
        )
    }
}
impl<S: RealScalar> Variance<S> for RelativisticBreitWigner<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let m = self.mean()?;
        checked_variance(self.q - m * m)
    }
}
