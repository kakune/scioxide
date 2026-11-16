//! Positive sums and integrals with explicit work limits.
use super::*;
use crate::distribution::LawOptions;
pub(super) use crate::law::positive;
pub(in crate::distribution) use crate::law::{constant as number, exhausted};
/// Requested density or logarithmic tail.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::distribution) enum Query {
    Density,
    Lower,
    Upper,
}
pub(in crate::distribution) fn boundary(x: f64, low: f64, high: f64, q: Query) -> Option<f64> {
    if x <= low {
        Some(if q == Query::Upper {
            0.0
        } else {
            f64::NEG_INFINITY
        })
    } else if x >= high {
        Some(if q == Query::Lower {
            0.0
        } else {
            f64::NEG_INFINITY
        })
    } else {
        None
    }
}
/// Work ledger of one law evaluation.
pub(in crate::distribution) struct Work {
    pub(in crate::distribution) used: usize,
    pub(in crate::distribution) limit: usize,
    pub(in crate::distribution) tolerance: f64,
    pub(in crate::distribution) error: f64,
}
impl Work {
    pub(in crate::distribution) fn new<S: RealScalar>(o: LawOptions<S>) -> Self {
        let tol = o.tolerance.relative().into_f64();
        // The internal quadrature target is capped at sqrt(eps) even for looser
        // user tolerances, because these integrals feed further log-space
        // arithmetic (complements, inversions) that would amplify a coarse
        // result. `error` starts at the unavoidable rounding floor.
        Self {
            used: 0,
            limit: o.work.get(),
            tolerance: tol.min(f64::SQRT_EPSILON),
            error: 64.0 * f64::EPSILON,
        }
    }
    pub(in crate::distribution) fn charge(&mut self, n: usize) -> Result<(), SciOxideError> {
        if n > self.limit - self.used {
            return Err(crate::law::iteration_limit(self.used));
        }
        self.used += n;
        Ok(())
    }

    /// Run `evaluate` on a ledger limited to the caller's remaining work,
    /// then return the unused allowance to `left`.
    pub(in crate::distribution) fn within<S: RealScalar, T>(
        options: LawOptions<S>,
        left: &mut usize,
        evaluate: impl FnOnce(&mut Self) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut work = Self::new(options);
        work.limit = *left;
        let result = evaluate(&mut work);
        *left = work.limit - work.used;
        result
    }

    pub(in crate::distribution) fn kernel<T>(
        &mut self,
        evaluate: impl FnOnce(&mut usize) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let available = self.limit - self.used;
        let mut left = available;
        let result = evaluate(&mut left);
        self.charge(available - left)?;
        result
    }
    pub(in crate::distribution) fn special(
        &mut self,
        evaluate: impl FnOnce(
            Budget,
        )
            -> Result<scioxide_special::SpecialFunctionResult<f64>, SciOxideError>,
    ) -> Result<f64, SciOxideError> {
        self.kernel(|left| crate::law::special(left, evaluate))
    }
}
/// Tail report whose absolute error also covers final scalar rounding.
pub(in crate::distribution) fn tail_result<S: RealScalar>(
    value: f64,
    scale: f64,
    work: &Work,
) -> Result<TestStatisticResult<S>, SciOxideError> {
    Ok(TestStatisticResult {
        log_probability: number(value)?,
        absolute_error: number(scale * work.error + value.exp() * S::EPSILON.into_f64())?,
        evaluations: work.used,
    })
}
pub(super) fn integral(
    mut f: impl FnMut(f64, &mut Work) -> Result<f64, SciOxideError>,
    a: f64,
    b: f64,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    let tolerance = w.tolerance;
    let value = crate::distribution::quadrature::log_integral(
        a,
        b,
        tolerance,
        &mut |x| {
            w.charge(1)?;
            f(x, w)
        },
        None,
    )?;
    w.error = w.error.max(tolerance);
    Ok(value)
}
/// Whole-real-line log integral. Locate the integrand peak before mapping;
/// this protects concentrated and remote tails against missed quadrature nodes.
pub(super) fn real_integral(
    mut f: impl FnMut(f64, &mut Work) -> Result<f64, SciOxideError>,
    center: f64,
    width: f64,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    // Whole-line integral of a unimodal log-integrand. (1) Walk with doubling
    // steps until the peak is bracketed, (2) refine the mode by ternary
    // search, (3) take the local Gaussian width from the curvature, and
    // (4) integrate the two mapped half-lines, which
    // puts quadrature nodes where the mass is and compresses the tails.
    let mut mode = center;
    let mut h = width;
    w.charge(1)?;
    let mut peak = f(mode, w)?;
    for _ in 0..64 {
        w.charge(2)?;
        let left = f(mode - h, w)?;
        let right = f(mode + h, w)?;
        if peak.is_finite() && left <= peak && right <= peak {
            break;
        }
        if peak == f64::NEG_INFINITY && left == peak && right == peak {
            h *= 2.0;
            continue;
        }
        if left > right {
            mode -= h;
            peak = left;
        } else {
            mode += h;
            peak = right;
        }
        h *= 2.0;
    }
    let mut a = mode - h;
    let mut b = mode + h;
    for _ in 0..40 {
        let l = a + (b - a) / 3.0;
        let r = b - (b - a) / 3.0;
        w.charge(2)?;
        if f(l, w)? > f(r, w)? {
            b = r;
        } else {
            a = l;
        }
    }
    mode = (a + b) * 0.5;
    w.charge(3)?;
    peak = f(mode, w)?;
    if !peak.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "integrand peak could not be resolved",
        ));
    }
    let h = width * 0.01;
    let curvature = (f(mode - h, w)? + f(mode + h, w)? - 2.0 * peak) / (h * h);
    let scale = if curvature.is_finite() && curvature < 0.0 {
        (-1.0 / curvature).sqrt()
    } else {
        width
    };
    let tolerance = w.tolerance;
    let mut evaluate = |x| {
        w.charge(1)?;
        f(x, w)
    };
    let lower =
        crate::distribution::quadrature::half_line(mode, -1., scale, tolerance, &mut evaluate)?;
    let upper =
        crate::distribution::quadrature::half_line(mode, 1., scale, tolerance, &mut evaluate)?;
    w.error = w.error.max(tolerance);
    logadd(lower, upper)
}

use crate::distribution::special::beta_leading_log;
pub(super) use crate::distribution::special::{
    normal_log, normal_log_density as log_phi, normal_log_interval as log_interval,
};
/// Incomplete beta evaluated from logarithmic coordinates, preserving tiny tails.
pub(super) fn beta_log(
    a: f64,
    b: f64,
    lx: f64,
    ly: f64,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    // ln I_x(a, b) given ln x and ln(1 - x) separately (so x near 1 is not
    // rounded): the leading series term for tiny x, the reflected complement
    // I_x(a, b) = 1 - I_{1-x}(b, a) when 1 - x < 1/2, otherwise the library.
    if let Some(value) = beta_leading_log(a, b, lx, || scioxide_special::ln_beta(a, b))? {
        return Ok(value);
    }
    if ly < -core::f64::consts::LN_2 {
        return log1mexp(beta_log(b, a, ly, lx, w)?);
    }
    w.special(|budget| {
        scioxide_special::beta_inc_log_with_options(
            a,
            b,
            lx.exp(),
            scioxide_special::BetaOptions::new(budget),
        )
    })
}

/// Wilson–Hilferty median of a (noncentral) chi-square with mean `mean` and
/// variance `2 * spread`. It only orders tail evaluations.
pub(super) fn chi_square_center(mean: f64, spread: f64) -> f64 {
    let cube = 1. - 2. * spread / (9. * mean * mean);
    if cube > 0. {
        mean * cube * cube * cube
    } else {
        mean
    }
}
