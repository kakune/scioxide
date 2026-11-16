//! Analytic densities shared only by examples and tests.
use scioxide_core::{ExtendedInterval, RealScalar, SciOxideError};
use scioxide_random::*;
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    pub offset: f64,
}
impl<S: RealScalar> Support<S> for Gaussian {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        ExtendedInterval::new(None, None)
    }
}
impl<S: RealScalar> LogDensity<S> for Gaussian {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        S::try_from_f64_lossy(self.offset - 0.5 * x.into_f64() * x.into_f64())
    }
}
impl<S: RealScalar> UserDensity<S> for Gaussian {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> ConcaveDensity<S> for Gaussian {
    fn log_density_derivative(&self, x: S) -> Result<S, SciOxideError> {
        Ok(-x)
    }
    fn validate_concavity(&self, _: DensityTransform) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> BoundedDensity<S> for Gaussian {
    fn ratio_uniforms_bounds(&self) -> Result<RatioUniformsBounds<S>, SciOxideError> {
        let scale = (self.offset * 0.5).exp();
        Ok(RatioUniformsBounds {
            shift: S::ZERO,
            u_max: S::try_from_f64_lossy(scale * (1.0 + 4.0 * S::EPSILON.into_f64()))?,
            v_min: S::try_from_f64_lossy(-scale)?,
            v_max: S::try_from_f64_lossy(scale)?,
        })
    }
}
pub struct Unit;
impl<S: RealScalar> Support<S> for Unit {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        ExtendedInterval::new(Some(S::ZERO), Some(S::ONE))
    }
}
impl<S: RealScalar> LogDensity<S> for Unit {
    fn log_pdf(&self, _: S) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}
impl<S: RealScalar> UserDensity<S> for Unit {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> ConcaveDensity<S> for Unit {
    fn log_density_derivative(&self, _: S) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
    fn validate_concavity(&self, _: DensityTransform) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<S: RealScalar> LogCdf<S> for Unit {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(if x <= S::ZERO {
            S::NEG_INFINITY
        } else if x >= S::ONE {
            S::ZERO
        } else {
            x.ln()
        })
    }
}
impl<S: RealScalar> LogSurvival<S> for Unit {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(if x <= S::ZERO {
            S::ZERO
        } else if x >= S::ONE {
            S::NEG_INFINITY
        } else {
            (-x).ln_1p()
        })
    }
}
