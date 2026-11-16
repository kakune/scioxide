//! Normal-inverse-gamma distribution.
use super::*;
use crate::distribution::LawOptions;

/// Normal-inverse-gamma distribution of a pair `(x, v)` with location `μ`,
/// precision factor `λ > 0`, shape `α > 0` and scale `β > 0`.
///
/// The variance `v` is inverse-gamma with shape `α` and scale `β`, and given
/// `v` the location `x` is normal with mean `μ` and variance `v / λ`. The density is
///
/// ```text
/// p(x, v) = √λ / √(2π v) · β^α / Γ(α) · v^(−α−1) · exp(−(2β + λ (x − μ)²) / (2v))
/// ```
///
/// Samples are `[x, v]`.
///
/// ```
/// use scioxide_random::{LogDensity, Mean, NormalInverseGamma, SeedableRng, SplitMix64};
/// let law = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0)?;
/// // At (0, 1): beta^alpha / Gamma(alpha) * exp(-beta) / sqrt(2 pi) = 4 exp(-2) / sqrt(2 pi).
/// let expected = 4.0_f64.ln() - 2.0 - 0.5 * (2.0 * core::f64::consts::PI).ln();
/// assert!((law.log_pdf((0.0, 1.0))? - expected).abs() < 1e-12);
/// assert!((law.mean()?[1] - 1.0).abs() < 1e-12);
/// let draw = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(draw[0].is_finite() && draw[1] > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct NormalInverseGamma<S: RealScalar> {
    mu: f64,
    lambda: f64,
    alpha: f64,
    beta: f64,
    gamma: crate::distribution::gamma::GammaKernel<f64>,
    log_gamma: Result<f64, SciOxideError>,
    options: LawOptions<S>,
}
impl<S: RealScalar> NormalInverseGamma<S> {
    /// Creates the law with parameters `μ = mu`, `λ = lambda`, `α = alpha` and `β = beta`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `DomainError` if `lambda`, `alpha` or `beta` is not positive.
    /// - `ResourceLimit` if the byte limit is below the bytes the parameters need.
    pub fn new(mu: S, lambda: S, alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Self::with_options(mu, lambda, alpha, beta, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options(
        mu: S,
        lambda: S,
        alpha: S,
        beta: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::argument(mu, "mu")?;
        options.admit::<f64>(2)?;
        Ok(Self {
            mu: mu.into_f64(),
            lambda: positive(lambda.into_f64(), scioxide_core::argument!("lambda"))?,
            alpha: positive(alpha.into_f64(), scioxide_core::argument!("alpha"))?,
            beta: positive(beta.into_f64(), scioxide_core::argument!("beta"))?,
            gamma: crate::distribution::gamma::GammaKernel::new(alpha.into_f64())?,
            log_gamma: scioxide_special::ln_gamma(alpha.into_f64()),
            options,
        })
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
}
impl<S: RealScalar> LogDensity<S, (S, S)> for NormalInverseGamma<S> {
    fn log_pdf(&self, (x, v): (S, S)) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        crate::law::argument(v, "v")?;
        let v = v.into_f64();
        if v <= 0.0 {
            return Ok(S::NEG_INFINITY);
        }
        let delta = finite(x.into_f64() - self.mu)?;
        // lambda (x - mu)^2 / (2 v) assembled in logs so that neither the
        // square nor the division overflows before the final exponent.
        let quadratic = if delta == 0.0 {
            0.0
        } else {
            finite(
                (self.lambda.ln() + 2.0 * delta.abs().ln() - core::f64::consts::LN_2 - v.ln())
                    .exp(),
            )?
        };
        log_terms(&[
            0.5 * (self.lambda.ln() - crate::distribution::special::log_two_pi::<f64>()),
            self.alpha * self.beta.ln(),
            -self.log_gamma.clone()?,
            -(self.alpha + 1.5) * v.ln(),
            -self.beta / v,
            -quadratic,
        ])
    }
}
impl<S: RealScalar> Density<S, (S, S)> for NormalInverseGamma<S> {
    fn pdf(&self, x: (S, S)) -> Result<S, SciOxideError> {
        number(self.log_pdf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for NormalInverseGamma<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        if self.alpha <= 1.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        converted([self.mu, self.beta / (self.alpha - 1.0)])
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for NormalInverseGamma<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        if self.alpha <= 2.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        converted([
            self.beta / (self.alpha - 1.0) / self.lambda,
            (self.beta / (self.alpha - 1.0)).powi(2) / (self.alpha - 2.0),
        ])
    }
}
impl<S: RealScalar> RowLaw<S, S> for NormalInverseGamma<S> {
    fn width(&self) -> usize {
        2
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        _: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        // sigma^2 = beta / G with G ~ Gamma(alpha), then x ~ N(mu, sigma^2 / lambda).
        // Working with ln sigma^2 keeps both steps finite for extreme alpha/beta.
        let logvariance = self.beta.ln() - self.gamma.sample_log(&mut words)?;
        let variance = crate::law::positive_draw::<S>(logvariance)?;
        let location = finite(
            self.mu
                + Normal::<f64>::standard().sample(&mut words)?
                    * (0.5 * (logvariance - self.lambda.ln())).exp(),
        )?;
        out[0] = number(location)?;
        out[1] = variance;
        Ok(())
    }
}
vector_sampling!(NormalInverseGamma, S, S::ZERO);
