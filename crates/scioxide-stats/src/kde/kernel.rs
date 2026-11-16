//! Cholesky-factored Gaussian kernels and compensated sums over kernel centers.
use crate::arithmetic::Compensated;
use crate::error_helpers::invalid;
use alloc::vec::Vec;
use scioxide_core::Argument;
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, Matrix};

pub(super) struct Kernel {
    pub(super) covariance: Vec<Scaled<f64>>,
    pub(super) scales: Vec<Scaled<f64>>,
    lower: Vec<f64>,
    inverse_scales: Vec<f64>,
    pub(super) log_normalization: f64,
    pub(super) diagonal: bool,
}
impl Kernel {
    pub(super) fn new(
        covariance: Vec<Scaled<f64>>,
        d: usize,
        argument: Argument,
    ) -> Result<Self, SciOxideError> {
        let mut scales = try_with_capacity(d, scioxide_core::quantity!("kde_scales"))?;
        for i in 0..d {
            let v = covariance[i * d + i];
            if v.is_zero() || v.is_sign_negative() {
                return Err(invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "KDE covariance must be positive definite",
                ));
            }
            scales.push(v.sqrt()?);
        }
        let mut normalized = try_filled(d * d, 0., scioxide_core::quantity!("kde_normalized"))?;
        let mut diagonal = true;
        for i in 0..d {
            for j in 0..d {
                if covariance[i * d + j] != covariance[j * d + i] {
                    return Err(invalid(
                        argument,
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "KDE covariance must be symmetric",
                    ));
                }
                if i != j && !covariance[i * d + j].is_zero() {
                    diagonal = false;
                }
                normalized[i * d + j] = covariance[i * d + j]
                    .div(scales[i])?
                    .div(scales[j])?
                    .round_to()?;
            }
        }
        let matrix = Matrix::from_vec(d, d, normalized)?;
        let factor = CholeskyDecomposition::factor(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            CholeskyPivoting::None,
        )?;
        let mut lower = try_with_capacity(d * d, scioxide_core::quantity!("kde_cholesky"))?;
        lower.extend_from_slice(factor.l_matrix().as_slice());
        let log_normalization = -(d as f64) * 0.5 * (2. * core::f64::consts::PI).ln()
            - scales
                .iter()
                .try_fold(0.0, |sum, s| s.ln_abs().map(|v| sum + v))?
            - (0..d).map(|i| lower[i * d + i].ln()).sum::<f64>();
        let mut inverse_scales =
            try_with_capacity(d, scioxide_core::quantity!("kde_inverse_scales"))?;
        for scale in &scales {
            // Zero marks a scale outside the ordinary kernel's range. The
            // original scaled factor remains available for every query.
            let inverse = Scaled::<f64>::ONE.div(*scale)?.round_to::<f64>();
            inverse_scales.push(inverse.ok().filter(|x| x.is_normal()).unwrap_or(0.));
        }
        Ok(Self {
            covariance,
            scales,
            lower,
            inverse_scales,
            log_normalization,
            diagonal,
        })
    }
    /// Returns half the squared Mahalanobis distance in `f64`, or `None` if the
    /// result would lose range or suffer cancellation, so that the scaled path is used.
    #[inline]
    pub(super) fn half_distance<S: RealScalar>(
        &self,
        point: &[S],
        center: &[S],
        scratch: &mut [f64],
    ) -> Option<f64> {
        let d = self.scales.len();
        if d == 1 {
            if self.inverse_scales[0] == 0. {
                return None;
            }
            let delta = point[0].into_f64() - center[0].into_f64();
            let z = ordinary_product(delta, self.inverse_scales[0])?;
            return ordinary_product(z * 0.5, z);
        }
        let mut distance = Compensated::<f64>::zero();
        for i in 0..d {
            if self.inverse_scales[i] == 0. {
                return None;
            }
            let delta = point[i].into_f64() - center[i].into_f64();
            let mut z = ordinary_product(delta, self.inverse_scales[i])?;
            if !self.diagonal {
                let mut value = Compensated::<f64>::zero().added(z).ok()?;
                let mut magnitude = z.abs();
                for (j, &previous) in scratch[..i].iter().enumerate() {
                    let term = ordinary_product(self.lower[i * d + j], previous)?;
                    value = value.added(-term).ok()?;
                    magnitude += term.abs();
                }
                z = value.total().ok()?;
                // A precision-scaled cancellation guard accounts for the
                // rounded products even though the reduction is compensated.
                if !magnitude.is_finite()
                    || (magnitude > 0. && z.abs() <= 8. * d as f64 * f64::EPSILON * magnitude)
                {
                    return None;
                }
                z /= self.lower[i * d + i];
                if !z.is_finite() || (z != 0. && !z.is_normal()) {
                    return None;
                }
                scratch[i] = z;
            }
            distance = distance.added(ordinary_product(z * 0.5, z)?).ok()?;
        }
        distance.total().ok()
    }
    pub(super) fn log_value(
        &self,
        delta: impl Fn(usize) -> Result<Scaled<f64>, SciOxideError>,
        scratch: &mut [f64],
    ) -> Result<f64, SciOxideError> {
        let d = self.scales.len();
        let mut q = Compensated::<Scaled<f64>>::zero();
        for i in 0..d {
            let mut value = delta(i)?.div(self.scales[i])?;
            for (j, previous) in scratch[..i].iter().enumerate() {
                value = value.sub(
                    Scaled::<f64>::from_value(self.lower[i * d + j])?
                        .mul(Scaled::<f64>::from_value(*previous)?)?,
                )?;
            }
            let value = value.div(Scaled::<f64>::from_value(self.lower[i * d + i])?)?;
            let Ok(rounded) = value.round_to::<f64>() else {
                return Ok(f64::NEG_INFINITY);
            };
            scratch[i] = rounded;
            q = q.added(value.square()?)?;
        }
        Ok(q.total()?
            .mul(Scaled::<f64>::from_value(0.5)?)?
            .round_to::<f64>()
            .map_or(f64::NEG_INFINITY, |half_q| self.log_normalization - half_q))
    }
}

/// Returns `a · b` if it is a normal number (or an exact zero), and `None` otherwise.
#[inline]
fn ordinary_product(a: f64, b: f64) -> Option<f64> {
    let value = a * b;
    (a.is_finite() && b.is_finite() && (value.is_normal() || (value == 0. && (a == 0. || b == 0.))))
        .then_some(value)
}

pub(super) struct KernelWeight {
    pub(super) log_mass: f64,
    pub(super) log_truncation: f64,
    pub(super) log_peak: f64,
    pub(super) peak: f64,
}

/// Range of one-dimensional query points for which every exponential and
/// weighted term is a normal `f64`, with a factor-two margin. Very small squared
/// distances may round to zero, which gives a Gaussian factor of one.
pub(super) struct OrdinaryInterval {
    lower: f64,
    upper: f64,
    pub(super) inverse: f64,
    half_distance_limit: f64,
}
impl OrdinaryInterval {
    pub(super) fn new<S: RealScalar>(
        centers: &[S],
        kernel: &Kernel,
        weights: &[KernelWeight],
    ) -> Option<Self> {
        if kernel.scales.len() != 1
            || kernel.inverse_scales[0] == 0.
            || weights.iter().any(|weight| !weight.peak.is_normal())
        {
            return None;
        }
        let lower = centers
            .iter()
            .map(|x| x.into_f64())
            .fold(f64::INFINITY, f64::min);
        let upper = centers
            .iter()
            .map(|x| x.into_f64())
            .fold(f64::NEG_INFINITY, f64::max);
        let minimum_log_peak = weights
            .iter()
            .map(|w| w.log_peak)
            .fold(f64::INFINITY, f64::min);
        let log_minimum = f64::MIN_POSITIVE.ln();
        Some(Self {
            lower,
            upper,
            inverse: kernel.inverse_scales[0],
            half_distance_limit: (-log_minimum).min(minimum_log_peak - log_minimum)
                - core::f64::consts::LN_2,
        })
    }
    pub(super) fn contains(&self, point: f64) -> bool {
        let bound = ((point - self.lower) * self.inverse)
            .abs()
            .max(((point - self.upper) * self.inverse).abs());
        let half_distance = (0.5 * bound) * bound;
        half_distance.is_finite() && half_distance <= self.half_distance_limit
    }
}

#[derive(Clone, Copy)]
pub(super) struct QuerySum {
    sum: Compensated<f64>,
    maximum: f64,
}
impl QuerySum {
    pub(super) fn zero() -> Self {
        Self {
            sum: Compensated::zero(),
            maximum: f64::NEG_INFINITY,
        }
    }
    #[inline]
    pub(super) fn add<const LOG: bool>(
        &mut self,
        half_distance: f64,
        weight: &KernelWeight,
    ) -> Option<()> {
        if !LOG {
            let exponential = (-half_distance).exp();
            if !weight.peak.is_normal() || !exponential.is_normal() {
                return None;
            }
            self.sum = self
                .sum
                .added(ordinary_product(weight.peak, exponential)?)
                .ok()?;
            return Some(());
        }
        self.add_bounded::<LOG>(half_distance, weight)
    }
    #[inline]
    pub(super) fn add_bounded<const LOG: bool>(
        &mut self,
        half_distance: f64,
        weight: &KernelWeight,
    ) -> Option<()> {
        let term = if LOG {
            let value = weight.log_peak - half_distance;
            if !value.is_finite() {
                return None;
            }
            if self.maximum == f64::NEG_INFINITY {
                self.maximum = value;
                1.
            } else if value > self.maximum {
                let scale = (self.maximum - value).exp();
                let [a, b] = self.sum.parts();
                self.sum = Compensated::zero()
                    .added(a * scale)
                    .ok()?
                    .added(b * scale)
                    .ok()?;
                self.maximum = value;
                1.
            } else {
                (value - self.maximum).exp()
            }
        } else {
            weight.peak * (-half_distance).exp()
        };
        self.sum = self.sum.added(term).ok()?;
        Some(())
    }
    pub(super) fn finish<const LOG: bool>(self) -> Option<f64> {
        let sum = self.sum.total().ok()?;
        if LOG {
            let result = self.maximum + sum.ln();
            result.is_finite().then_some(result)
        } else {
            sum.is_normal().then_some(sum)
        }
    }
}
