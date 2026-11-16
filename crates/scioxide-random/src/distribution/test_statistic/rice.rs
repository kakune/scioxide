//! Rice magnitudes of shifted two-dimensional normal samples.
use super::math::{Query, Work, boundary, number};
use super::{
    NoncentralChi, NoncentralChiSquared, Statistic, TestStatisticResult, inverse_result, mixture,
    statistic_law,
};
use crate::distribution::LawOptions;
use crate::distribution::formula::{Formula, formula_law};
use crate::distribution::normal::standard_normal;
use crate::law::{argument, finite};
use crate::{Mean, Rng, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Rice distribution with shape `b ≥ 0` and unit scale: the length of a two-dimensional normal
/// vector with mean `(b, 0)`.
///
/// `X = √((b + Z₁)² + Z₂²)` with independent standard normal `Z₁, Z₂`. The
/// density is
///
/// ```text
/// f(x) = x exp(−(x² + b²)/2) I₀(b x),   x ≥ 0
/// ```
///
/// where `I₀` is the modified Bessel function of the first kind. `X²` is
/// noncentral chi-squared with 2 degrees of freedom and noncentrality `b²`,
/// and quantiles are square roots of its quantiles. `b = 0` is the Rayleigh
/// law. Use [`Affine`](crate::Affine) for another scale.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Mean, Quantile, Rice, SeedableRng};
/// use scioxide_random::{SplitMix64, Support};
/// let rayleigh = Rice::new(0.0_f64)?;
/// assert!((rayleigh.cdf(1.0)? - (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
/// assert!((rayleigh.mean()? - core::f64::consts::FRAC_PI_2.sqrt()).abs() < 1e-12);
/// let law = Rice::new(1.5_f64)?;
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(8);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Rice<S: RealScalar> {
    b: f64,
    base: NoncentralChiSquared<S>,
    options: LawOptions<S>,
}
impl<S: RealScalar> Rice<S> {
    /// Creates a Rice law with shape `b` and unit scale.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape < 0`.
    /// - `NumericalInstability` if `shape²` overflows.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        Self::with_options(shape, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(shape: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(shape, "shape")?;
        if shape < S::ZERO {
            return Err(crate::law::domain(
                scioxide_core::argument!("shape"),
                "Rice shape must be nonnegative",
            ));
        }
        let b = shape.into_f64();
        let lambda = finite(b * b)?;
        Ok(Self {
            b,
            base: NoncentralChiSquared {
                nu: 2.0,
                lambda,
                draw: NoncentralChi::new(2.0, lambda),
                central: options
                    .widened()
                    .and_then(|widened| crate::ChiSquared::with_options(2.0, widened)),
                options,
            },
            options,
        })
    }
    fn first(&self, w: &mut Work) -> Result<f64, SciOxideError> {
        if self.b >= 32.0 {
            w.charge(12)?;
            let (correction, _) = rice_moment_correction(self.b);
            return finite(self.b + self.b * correction);
        }
        // E[R] as a Poisson(b^2/2) mixture of chi_{2 + 2j} means,
        // sqrt(2) Gamma(j + 3/2) / Gamma(j + 1). The tolerance is tightened with
        // b^2 because the variance 2 + b^2 - E[R]^2 subtracts E[R]^2 ~ b^2.
        w.tolerance = w
            .tolerance
            .min((1e-10 / (1.0 + self.b * self.b)).max(64.0 * f64::EPSILON));
        Ok(mixture(
            self.b * self.b / 2.0,
            |j, _| {
                Ok(0.5 * (2.0 * (j + 1.0)).ln()
                    + crate::distribution::special::log_gamma_ratio_remainder(j + 1.0, 0.5)?)
            },
            w,
        )?
        .exp())
    }
}
impl<S: RealScalar> Statistic<S> for Rice<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn median(&self) -> f64 {
        (self.b * self.b + 2. * core::f64::consts::LN_2).sqrt()
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        if let Some(v) = boundary(x, 0.0, f64::INFINITY, q) {
            return Ok(v);
        }
        // Rice(b) is the square root of a noncentral chi^2 with 2 degrees and
        // noncentrality b^2; the density gains the Jacobian 2x.
        Ok(self.base.kernel(finite(x * x)?, q, w)?
            + if q == Query::Density {
                core::f64::consts::LN_2 + x.ln()
            } else {
                0.0
            })
    }
    fn solve(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        let p = p.into_f64();
        inverse_result(self.base.inverse_value(p, upper, left)?.sqrt(), p, true)
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let mut r = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let x = standard_normal::<f64, _>(&mut r)? + self.b;
        number(finite(x.hypot(standard_normal::<f64, _>(&mut r)?))?)
    }
}
statistic_law!(Rice, report);
impl<S: RealScalar> Rice<S> {
    fn second(&self, w: &mut Work) -> Result<S, SciOxideError> {
        if self.b >= 32.0 {
            w.charge(12)?;
            return number(rice_moment_correction(self.b).1);
        }
        let m = self.first(w)?;
        let v = 2.0 + self.b * self.b - m * m;
        number(crate::distribution::moment_math::positive_variance(v)?)
    }
}
impl<S: RealScalar> Mean<S> for Rice<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(self.first(&mut Work::new(self.options))?)
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            Work::within(self.options, left, |w| number(self.first(w)?))
        })
    }
}
impl<S: RealScalar> Variance<S> for Rice<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        self.second(&mut Work::new(self.options))
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            Work::within(self.options, left, |w| self.second(w))
        })
    }
}

// Large-argument expansion of the Rice half moment (Laguerre/Bessel expansion).
// b>=32 and twelve terms place the first omitted term below double precision.
fn rice_moment_correction(b: f64) -> (f64, f64) {
    let t = (1.0 / b) / b;
    let mut term = 0.5 * t;
    let mut correction = term;
    let mut remainder = 0.0;
    for n in 2..=12 {
        term *= f64::from(2 * n - 3).powi(2) * t / f64::from(2 * n);
        correction += term;
        remainder += term / t;
    }
    (
        correction,
        1.0 - 2.0 * remainder - correction * (correction / t),
    )
}
