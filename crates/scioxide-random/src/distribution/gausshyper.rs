//! Gauss hypergeometric distribution.
use crate::distribution::LawOptions;
use crate::distribution::formula::formula_law;
use crate::distribution::formula::{Formula, Inverse};
use crate::distribution::quadrature::positive_integral;
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{argument, finite, logadd};
use crate::{Mean, Variance};
use scioxide_core::{Budget, RealScalar, SciOxideError};

/// Gauss hypergeometric distribution on `[0, 1]` with shapes `a, b > 0`, exponent `c` and `z > −1`.
///
/// The density is `f(x) ∝ x^(a−1) (1 − x)^(b−1) (1 + zx)^(−c)`. The
/// normalizing constant is a Gauss hypergeometric function `₂F₁` (DLMF 15.6.1).
/// The normalizing constant, the CDF and the moments are computed by
/// quadrature, within the [work limits](crate#work-limits).
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, GaussHypergeometric, Mean, Quantile, SeedableRng, SplitMix64,
/// };
/// // c = 0 reduces to Beta(2, 3).
/// let law = GaussHypergeometric::<f64>::new(2.0, 3.0, 0.0, 0.5)?;
/// assert!((law.mean()? - 0.4).abs() < 1e-10);
/// let x = law.quantile(0.5)?;
/// assert!((law.cdf(x)? - 0.5).abs() < 1e-8);
/// let mut rng = SplitMix64::from_seed(62);
/// assert!((0.0..=1.0).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GaussHypergeometric<S: RealScalar> {
    a: S,
    b: S,
    c: S,
    z: S,
    log_norm: S,
    options: LawOptions<S>,
}
impl<S: RealScalar> GaussHypergeometric<S> {
    /// Creates a Gauss hypergeometric distribution with parameters `a`, `b`, `c` and `z`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `DomainError` if `a ≤ 0`, `b ≤ 0` or `z ≤ −1`.
    /// - `NumericalInstability` if the normalizing constant is outside the range of `S`.
    /// - `ConvergenceFailure` if the quadrature for the normalizing constant
    ///   exhausts the work limit.
    pub fn new(a: S, b: S, c: S, z: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, b, c, z, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(
        a: S,
        b: S,
        c: S,
        z: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(a, scioxide_core::argument!("a"))?;
        positive(b, scioxide_core::argument!("b"))?;
        argument(c, "c")?;
        argument(z, "z")?;
        if z <= -S::ONE {
            return Err(crate::law::domain(
                scioxide_core::argument!("z"),
                "gausshyper requires z > -1",
            ));
        }
        let mut law = Self {
            a,
            b,
            c,
            z,
            log_norm: S::ZERO,
            options,
        };
        // The normalizing constant is the same integral as the CDF, evaluated
        // from each endpoint to 1/2, so each half can use the substitution
        // that removes its own endpoint singularity.
        let mut left = options.work.get();
        law.log_norm = logadd(
            law.integral(S::HALF, false, 0, &mut left)?,
            law.integral(S::HALF, true, 0, &mut left)?,
        )?;
        law.log_norm = finite(law.log_norm)?;
        Ok(law)
    }
    fn integral(
        &self,
        x: S,
        upper: bool,
        moment: u32,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if x == S::ZERO && !upper || x == S::ONE && upper {
            return Ok(S::NEG_INFINITY);
        }
        // For the rth raw moment use t = x u^(4/(a+r)), so
        // t^(a+r-1) dt = 4 x^(a+r) u^3/(a+r) du. Retaining a's
        // substitution for r>0 would hide the moment mass when a is tiny.
        // The cubic zero smooths the remaining endpoint derivatives as well as
        // removing the density singularity. The upper piece mirrors this with
        // 1-t = (1-x) u^(4/b). This is important when one shared inverse budget
        // must cover many tail evaluations.
        let m = number::<S>(moment as f64)?;
        let lower_shape = self.a + m;
        let four = S::TWO * S::TWO;
        let f = |u: S| {
            let log_u = u.ln();
            let t = if upper {
                S::ONE - (S::ONE - x) * (four * log_u / self.b).exp()
            } else {
                x * (four * log_u / lower_shape).exp()
            };
            let base = if upper {
                self.b * (S::ONE - x).ln() - self.b.ln() + (self.a - S::ONE + m) * t.ln()
            } else {
                lower_shape * x.ln() - lower_shape.ln() + (self.b - S::ONE) * (-t).ln_1p()
            };
            Ok(base + four.ln() + (four - S::ONE) * log_u - self.c * (self.z * t).ln_1p())
        };
        // Reserve accuracy for normalization and two raw moments before their
        // subtraction; highly concentrated cases still fail the variance guard.
        positive_integral(
            S::ZERO,
            S::ONE,
            (self.options.tolerance.relative() / number::<S>(128.0)?)
                .max(number::<S>(64.0)? * S::EPSILON),
            left,
            &f,
            None,
        )
    }
    fn tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if x >= S::ONE {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        Ok(self.integral(x, upper, 0, left)? - self.log_norm)
    }
    fn raw_moment(&self, n: u32) -> Result<S, SciOxideError> {
        self.raw_moment_with_work(n, &mut self.options.work.get())
    }
    fn raw_moment_with_work(&self, n: u32, left: &mut usize) -> Result<S, SciOxideError> {
        let x = S::HALF;
        finite(
            (logadd(
                self.integral(x, false, n, left)?,
                self.integral(x, true, n, left)?,
            )? - self.log_norm)
                .exp(),
        )
    }
}
impl<S: RealScalar> Formula<S> for GaussHypergeometric<S> {
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
        self.tail(x, upper, left)
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), Some(S::ONE))
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO || x > S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            return Ok(if self.a < S::ONE {
                S::INFINITY
            } else if self.a > S::ONE {
                S::NEG_INFINITY
            } else {
                -self.log_norm
            });
        }
        if x == S::ONE {
            return Ok(if self.b < S::ONE {
                S::INFINITY
            } else if self.b > S::ONE {
                S::NEG_INFINITY
            } else {
                -self.c * self.z.ln_1p() - self.log_norm
            });
        }
        Ok(
            (self.a - S::ONE) * x.ln() + (self.b - S::ONE) * (-x).ln_1p()
                - self.c * (self.z * x).ln_1p()
                - self.log_norm,
        )
    }

    const INVERSE: Inverse = Inverse::Smooth;
}
formula_law!(GaussHypergeometric);
impl<S: RealScalar> Mean<S> for GaussHypergeometric<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(1)
    }
}
impl<S: RealScalar> Variance<S> for GaussHypergeometric<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let mut left = self.options.work.get();
        let mean = self.raw_moment_with_work(1, &mut left)?;
        let second = self.raw_moment_with_work(2, &mut left)?;
        let tolerance = self.options.tolerance.relative();
        let error = tolerance / number::<S>(128.0)?
            + number::<S>(64.0)? * S::EPSILON
            + number::<S>(8.0)? * S::EPSILON * self.log_norm.abs();
        crate::distribution::moment_math::variance_from_moments_with_accuracy(
            second, mean, error, tolerance,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Cdf, Density, Survival};
    #[test]
    fn uniform_reduction() {
        let law = GaussHypergeometric::<f64>::new(1., 1., 0., 0.).unwrap();
        assert!((law.cdf(0.25).unwrap() - 0.25).abs() < 1e-12);
        assert!((law.sf(0.25).unwrap() - 0.75).abs() < 1e-12);
        assert!((law.pdf(0.25).unwrap() - 1.).abs() < 1e-12);
        assert!((law.mean().unwrap() - 0.5).abs() < 1e-12);
    }
}
