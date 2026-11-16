//! Moyal distribution.
use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::law::constant as number;
use crate::law::{finite, log1mexp};
use crate::{Mean, Variance};
use scioxide_core::{RealScalar, SciOxideError};

/// Standard Moyal distribution on the real line.
///
/// The density is `f(x) = exp(−(x + e^(−x))/2) / √(2π)`. Shift and scale it
/// with [`Affine`](crate::Affine).
///
/// # References
///
/// J. E. Moyal, "Theory of ionization fluctuations", *Phil. Mag.* 46(374),
/// 1955, 263–280.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, Moyal, Quantile, SeedableRng, SplitMix64, Variance,
/// };
/// let law = Moyal::<f64>::new();
/// let x = law.quantile(0.4)?;
/// assert!((law.cdf(x)? - 0.4).abs() < 1e-14);
/// assert!((law.mean()? - (0.577_215_664_901_532_9 + 2.0_f64.ln())).abs() < 1e-15);
/// assert!((law.variance()? - core::f64::consts::PI.powi(2) / 2.0).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(23);
/// assert!(law.sample(&mut rng)?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Moyal<S: RealScalar>(core::marker::PhantomData<S>);
impl<S: RealScalar> Moyal<S> {
    /// Creates the standard Moyal distribution.
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}
impl<S: RealScalar> Default for Moyal<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> Formula<S> for Moyal<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (None, None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        Ok(-(x + (-x).exp()) * S::HALF - crate::distribution::special::log_two_pi::<S>() * S::HALF)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x > -S::EPSILON.ln() {
                // erf(t) ~ 2t/sqrt(pi); form its logarithm before t underflows.
                return Ok(-x * S::HALF
                    + (number::<S>(2.)? / number::<S>(core::f64::consts::PI)?).ln() * S::HALF);
            }
            let t = (-x * S::HALF).exp() / number::<S>(2.)?.sqrt();
            if t < S::ONE {
                Ok(scioxide_special::erf(t)?.ln())
            } else {
                log1mexp(self.tail_with_work(x, false, _left)?)
            }
        } else {
            let t = (-x * S::HALF).exp() / number::<S>(2.)?.sqrt();
            if t > S::ONE {
                Ok(scioxide_special::erfcx(t)?.ln() - t * t)
            } else {
                Ok(scioxide_special::erfc(t)?.ln())
            }
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        let t = scioxide_special::erfcinv(p)?;
        finite(-(number::<S>(2.)? * t * t).ln())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p < S::SQRT_EPSILON {
            return finite(
                -number::<S>(2.)? * p.ln()
                    + (number::<S>(2.)? / number::<S>(core::f64::consts::PI)?).ln(),
            );
        }
        let t = scioxide_special::erfinv(p)?;
        finite(-(number::<S>(2.)? * t * t).ln())
    }
}
formula_law!(Moyal);
impl<S: RealScalar> Mean<S> for Moyal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(0.5772156649015329 + core::f64::consts::LN_2)
    }
}
impl<S: RealScalar> Variance<S> for Moyal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let p = number::<S>(core::f64::consts::PI)?;
        finite(p * p * S::HALF)
    }
}
