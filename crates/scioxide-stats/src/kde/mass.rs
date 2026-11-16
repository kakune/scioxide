//! Closed-form integrals of the KDE over boxes, against another KDE and against a Gaussian.
use super::kernel::Kernel;
use super::{KdeBandwidth, KdeOptions, RawGaussianKde};
use crate::ObservationData;
use crate::error_helpers::invalid;
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::logsumexp;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

impl<S: RealScalar> RawGaussianKde<S> {
    /// Integrates the density over the box `[lower, upper]`, intersected with the
    /// support, from normal CDF differences. Bounds may be infinite. Requires a
    /// diagonal kernel covariance.
    ///
    /// # Errors
    /// Fails on NaN or reversed bounds, a dimension mismatch, or a kernel with
    /// correlations.
    pub(crate) fn integrate_box(&self, lower: &[S], upper: &[S]) -> Result<S, SciOxideError> {
        let d = self.dimensions;
        if lower.len() != d
            || upper.len() != d
            || lower
                .iter()
                .zip(upper)
                .any(|(a, b)| a.is_nan() || b.is_nan() || a > b)
        {
            return Err(invalid(
                scioxide_core::argument!("lower"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid KDE integration bounds",
            ));
        }
        if !self.kernel.diagonal {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "analytic box integration requires diagonal kernels",
            ));
        }
        let mut result = f64::NEG_INFINITY;
        for (i, weight) in self.weights.iter().enumerate() {
            let mut term = weight.log_mass;
            for j in 0..d {
                let (mut lo, mut hi) = (lower[j], upper[j]);
                if let Some(support) = &self.support {
                    lo = lo.max(support[j].lower());
                    hi = hi.min(support[j].upper());
                }
                if hi <= lo {
                    return Ok(S::ZERO);
                }
                let mass =
                    normal_interval_mass(lo, hi, self.centers[i * d + j], self.kernel.scales[j])?;
                term += if self.support.is_some() {
                    mass.log_ratio(self.interval_masses[i * d + j])?
                } else {
                    mass.log_value()?
                };
            }
            result = logsumexp(&[result, term])?;
        }
        S::try_from_f64_lossy(result.exp().min(1.))
    }
    /// Computes `∫ f(x) g(x) dx` for two KDEs without support bounds. Factors
    /// the sum of the kernel covariances once, then sums over all pairs of
    /// centers. Allocates.
    ///
    /// # Errors
    /// Fails if either KDE has a support, the dimensions differ, allocation
    /// fails, or the result cannot be represented in `S`.
    pub(crate) fn integrate_kde(&self, other: &Self) -> Result<S, SciOxideError> {
        if self.dimensions != other.dimensions || self.support.is_some() || other.support.is_some()
        {
            return Err(invalid(
                scioxide_core::argument!("other"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "KDE overlap requires matching unbounded dimensions",
            ));
        }
        let d = self.dimensions;
        let mut cov = try_with_capacity(d * d, scioxide_core::quantity!("kde_overlap"))?;
        for (a, b) in self.kernel.covariance.iter().zip(&other.kernel.covariance) {
            cov.push(a.add(*b)?);
        }
        let kernel = Kernel::new(cov, d, scioxide_core::argument!("other"))?;
        let mut scratch = try_filled(d, 0., scioxide_core::quantity!("kde_scratch"))?;
        let mut result = f64::NEG_INFINITY;
        for (i, a) in self.weights.iter().enumerate() {
            for (j, b) in other.weights.iter().enumerate() {
                result = logsumexp(&[
                    result,
                    a.log_mass
                        + b.log_mass
                        + kernel.log_value(
                            |k| {
                                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
                                    self.centers[i * d + k],
                                ))?
                                .sub(Scaled::<f64>::from_value(
                                    scioxide_core::RealScalar::into_f64(other.centers[j * d + k]),
                                )?)
                            },
                            &mut scratch,
                        )?,
                ])?;
            }
        }
        S::try_from_f64_lossy(result.exp())
    }
    /// Computes `∫ f(x) φ(x; μ, Σ) dx` for a KDE `f` without support bounds and
    /// a Gaussian density `φ` with mean `μ` and covariance `Σ`.
    ///
    /// # Errors
    /// Fails if the KDE has a support, the mean or covariance is invalid or not
    /// positive definite, or allocation fails.
    pub(crate) fn integrate_gaussian(
        &self,
        mean: &[S],
        covariance: &[S],
    ) -> Result<S, SciOxideError> {
        let d = self.dimensions;
        if mean.len() != d {
            return Err(invalid(
                scioxide_core::argument!("mean"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Gaussian mean dimension mismatch",
            ));
        }
        let shape = [1, d];
        let strides = [d, 1];
        let view = ObservationData::new(
            mean,
            scioxide_core::StridedLayout::new(&shape, &strides)?,
            0,
            Some(1),
        )?;
        let other = Self::prepare(
            view,
            0,
            KdeOptions {
                bandwidth: KdeBandwidth::Covariance(covariance),
                ..KdeOptions::default()
            },
            ExecutionContext::sequential(),
        )?;
        self.integrate_kde(&other)
    }
}
fn standardized<S: RealScalar>(x: S, center: S, scale: Scaled<f64>) -> Result<f64, SciOxideError> {
    if !x.is_finite() {
        return Ok(x.into_f64());
    }
    let value = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?
        .sub(Scaled::<f64>::from_value(
            scioxide_core::RealScalar::into_f64(center),
        )?)?
        .div(scale)?;
    Ok(value
        .round_to::<f64>()
        .unwrap_or(if value.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }))
}
#[derive(Clone, Copy)]
pub(super) struct NormalMass {
    pub(super) width: Scaled<f64>,
    pub(super) log_average: f64,
}
impl NormalMass {
    pub(super) fn log_value(self) -> Result<f64, SciOxideError> {
        Ok(self.width.ln_abs()? + self.log_average)
    }
    fn log_ratio(self, denominator: Self) -> Result<f64, SciOxideError> {
        Ok(
            self.width.div(denominator.width)?.ln_abs()? + self.log_average
                - denominator.log_average,
        )
    }
}
pub(super) fn normal_interval_mass<S: RealScalar>(
    low: S,
    high: S,
    center: S,
    scale: Scaled<f64>,
) -> Result<NormalMass, SciOxideError> {
    if high <= low {
        return Ok(NormalMass {
            width: Scaled::<f64>::ONE,
            log_average: f64::NEG_INFINITY,
        });
    }
    if low.is_finite() && high.is_finite() {
        let width = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(high))?
            .sub(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(low),
            )?)?
            .div(scale)?;
        let middle = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(low))?
            .sub(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(center),
            )?)?
            .div(scale)?
            .add(width.mul(Scaled::<f64>::from_value(0.5)?)?)?;
        if let Ok(middle) = middle.round_to::<f64>()
            && width
                .abs_cmp(Scaled::<f64>::from_value(0.1 / (1. + middle.abs()))?)
                .is_lt()
        {
            // A narrow interval is integrated in log space: its width need not
            // be representable as a scalar probability. Gaussian quadrature
            // avoids both CDF cancellation and PDF underflow in the tails.
            const X: [f64; 4] = [
                0.1834346424956498,
                0.525532409916329,
                0.7966664774136267,
                0.9602898564975363,
            ];
            const W: [f64; 4] = [
                0.362683783378362,
                0.3137066458778873,
                0.2223810344533745,
                0.1012285362903763,
            ];
            let half = width.round_to::<f64>()? * 0.5;
            let mut terms = [0.0; 8];
            for ((pair, x), w) in terms.chunks_exact_mut(2).zip(X).zip(W) {
                let h = half * x;
                pair[0] = w.ln() - 0.5 * (middle - h).powi(2);
                pair[1] = w.ln() - 0.5 * (middle + h).powi(2);
            }
            let sum = logsumexp(&terms)?;
            return Ok(NormalMass {
                width,
                log_average: -core::f64::consts::LN_2 - 0.5 * (2. * core::f64::consts::PI).ln()
                    + sum,
            });
        }
    }
    let low = standardized(low, center, scale)?;
    let high = standardized(high, center, scale)?;
    let tail = |x: f64| {
        if x == f64::INFINITY {
            Ok(0.0)
        } else if x == f64::NEG_INFINITY {
            Ok(1.0)
        } else {
            scioxide_special::erfc(x / core::f64::consts::SQRT_2).map(|value| value * 0.5)
        }
    };
    let probability = if low >= 0. {
        tail(low)? - tail(high)?
    } else if high <= 0. {
        tail(-high)? - tail(-low)?
    } else {
        1. - tail(-low)? - tail(high)?
    };
    Ok(NormalMass {
        width: Scaled::<f64>::ONE,
        log_average: probability.ln(),
    })
}
