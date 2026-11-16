//! Asymmetric Laplace distribution.
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::law::positive;
use crate::law::{finite, log1mexp};
use crate::{Mean, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Asymmetric Laplace distribution with asymmetry `κ > 0`.
///
/// The density is `f(x) = κ/(1 + κ²) exp(−κx)` for `x ≥ 0` and
/// `f(x) = κ/(1 + κ²) exp(x/κ)` for `x < 0`. `κ = 1` is the standard Laplace
/// law. Shift and scale it with [`Affine`](crate::Affine).
///
/// ```
/// use scioxide_random::{AsymmetricLaplace, Cdf, Distribution, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Variance};
/// let law = AsymmetricLaplace::new(2.0_f64)?;
/// assert!((law.cdf(0.0)? - 0.8).abs() < 1e-15); // κ²/(1+κ²)
/// assert_eq!((law.mean()?, law.variance()?), (-1.5, 4.25));
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(29);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AsymmetricLaplace<S: RealScalar> {
    kappa: S,
}
impl<S: RealScalar> AsymmetricLaplace<S> {
    /// Creates an asymmetric Laplace distribution with asymmetry `κ = kappa`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `kappa` is NaN or infinite.
    /// - `DomainError` if `kappa ≤ 0`.
    pub fn new(kappa: S) -> Result<Self, SciOxideError> {
        positive(kappa, scioxide_core::argument!("kappa"))?;
        Ok(Self { kappa })
    }
    // P(X < 0) = kappa^2 / (1 + kappa^2) and P(X >= 0) = 1 / (1 + kappa^2),
    // formed in log space as t - softplus(t) and -softplus(t) with
    // t = 2 ln kappa, so extreme asymmetry neither overflows nor rounds a
    // side's mass to zero.
    fn log_negative_mass(&self) -> Result<S, SciOxideError> {
        let t = self.kappa.ln() * (S::TWO);
        Ok(t - scioxide_core::softplus(t)?)
    }
    fn log_positive_mass(&self) -> Result<S, SciOxideError> {
        Ok(-scioxide_core::softplus(self.kappa.ln() * (S::TWO))?)
    }
}
impl<S: RealScalar> Formula<S> for AsymmetricLaplace<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.kappa.ln()
            + self.log_positive_mass()?
            + if x < S::ZERO {
                x / self.kappa
            } else {
                -self.kappa * x
            })
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(if x >= S::ZERO {
                self.log_positive_mass()? - self.kappa * x
            } else {
                log1mexp(self.log_negative_mass()? + x / self.kappa)?
            })
        } else {
            Ok(if x < S::ZERO {
                self.log_negative_mass()? + x / self.kappa
            } else {
                log1mexp(self.log_positive_mass()? - self.kappa * x)?
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        let m = self.log_negative_mass()?.exp();
        finite(if p < m {
            self.kappa * (p.ln() - self.log_negative_mass()?)
        } else {
            -(log1mexp(p.ln())? - self.log_positive_mass()?) / self.kappa
        })
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        let m = self.log_positive_mass()?.exp();
        finite(if p < m {
            (self.log_positive_mass()? - p.ln()) / self.kappa
        } else {
            self.kappa * (log1mexp(p.ln())? - self.log_negative_mass()?)
        })
    }
}
formula_law!(AsymmetricLaplace);
impl<S: RealScalar> Mean<S> for AsymmetricLaplace<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        finite(S::ONE / self.kappa - self.kappa)
    }
}
impl<S: RealScalar> Variance<S> for AsymmetricLaplace<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        finite(S::ONE / (self.kappa * self.kappa) + self.kappa * self.kappa)
    }
}
