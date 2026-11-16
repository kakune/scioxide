//! Real Li_2 and Spence by reflection/inversion (DLMF 25.12.4, 25.12.6).
//!
//! Reduction puts Li_2's argument in [-1, 1/2]. With w = -log(1-x),
//! integrating w/(exp(w)-1) gives w - w²/4 + Σ B_2k*w^(2k+1)/(2k+1)!.
//! Here |w| <= log(2); thirteen Bernoulli terms leave less than 8e-29.

use crate::support::TranscendentalEstimate as Estimate;
use crate::support::Work;
use crate::support::wide::Wide as D;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "required when elementary methods use the no_std scalar trait"
)]
use scioxide_core::RealScalar;
use scioxide_core::{SciOxideError, algorithm_component};

use crate::support::constants::ZETA_2 as ZETA_TWO;

const BERNOULLI_INTEGRAL: [f64; 13] = {
    let mut coefficients = [0.0; 13];
    let mut j = 0;
    while j < coefficients.len() {
        coefficients[j] =
            crate::support::gamma_series::BERNOULLI_OVER_FACTORIAL[j].hi / (2 * j + 3) as f64;
        j += 1;
    }
    coefficients
};

fn bernoulli(w: f64, work: &mut Work<'_, '_, '_>) -> Result<f64, SciOxideError> {
    let square = w * w;
    let mut polynomial: f64 = 0.0;
    for &coefficient in BERNOULLI_INTEGRAL.iter().rev() {
        work.try_step()?;
        polynomial = scioxide_core::fast_mul_add(polynomial, square, coefficient);
    }
    Ok(scioxide_core::fast_mul_add(
        square,
        scioxide_core::fast_mul_add(w, polynomial, -0.25),
        w,
    ))
}

/// Reconstruct in two terms: the constants and square often nearly cancel.
fn invert(base: D, logarithm: D, positive: bool) -> (f64, f64) {
    let constant = if positive {
        ZETA_TWO.mul_f64(2.0)
    } else {
        -ZETA_TWO
    };
    let square = (logarithm * logarithm).mul_f64(0.5);
    let value = (constant - square - base).value();
    (
        value,
        16.0 * f64::EPSILON * (constant.value().abs() + square.value().abs() + base.value().abs()),
    )
}

/// The defining series at 1/x <= 1/12, in two terms. Its geometric tail is
/// below 1e-33; this keeps output ulps through the inversion's real zero.
fn inverse_series(x: f64, work: &mut Work<'_, '_, '_>) -> Result<D, SciOxideError> {
    let y = D::from(1.0) / D::from(x);
    let mut power = y;
    let mut sum = y;
    for k in 2..=32 {
        work.try_step()?;
        power = power * y;
        let term = power.div_f64(f64::from(k * k));
        sum = sum + term;
        if term.value().abs() <= 1e-33 {
            break;
        }
    }
    Ok(sum)
}

pub(super) fn kernel(
    x: f64,
    spence: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate<f64>, SciOxideError> {
    let (value, error) = if (spence && x == 1.0) || (!spence && x == 0.0) {
        (if spence { 0.0 } else { x }, 0.0)
    } else if (spence && x == 0.0) || (!spence && x == 1.0) {
        (ZETA_TWO.hi, f64::EPSILON * ZETA_TWO.hi)
    } else {
        work.try_step()?;
        if spence {
            if x < 0.5 {
                let base = bernoulli(-(-x).ln_1p(), work)?;
                let correction = D::from(x.ln()) * D::from((-x).ln_1p());
                let value = (ZETA_TWO - correction - D::from(base)).value();
                (value, 16.0 * f64::EPSILON * ZETA_TWO.hi)
            } else if x <= 2.0 {
                // ln(x) retains the original input, including both sides of 1.
                let value = bernoulli(-x.ln(), work)?;
                (value, 8.0 * f64::EPSILON * value.abs())
            } else {
                let inverse = 1.0 / (x - 1.0);
                let base = bernoulli(-inverse.ln_1p(), work)?;
                invert(D::from(base), D::from((x - 1.0).ln()), false)
            }
        } else if x < -1.0 {
            let base = bernoulli(-(-1.0 / x).ln_1p(), work)?;
            invert(D::from(base), D::from((-x).ln()), false)
        } else if x <= 0.5 {
            let value = bernoulli(-(-x).ln_1p(), work)?;
            (value, 8.0 * f64::EPSILON * value.abs())
        } else if x < 2.0 {
            let base = bernoulli(-x.ln(), work)?;
            let correction = D::from(x.ln()) * D::from((1.0 - x).abs().ln());
            let value = (ZETA_TWO - correction - D::from(base)).value();
            (
                value,
                16.0 * f64::EPSILON * (value.abs() + base.abs() + correction.value().abs()),
            )
        } else {
            let base = if (12.0..=13.0).contains(&x) {
                inverse_series(x, work)?
            } else {
                D::from(bernoulli(-(-1.0 / x).ln_1p(), work)?)
            };
            // Around the real zero, binary64 logarithm rounding dominates the
            // cancelling reconstruction. The existing two-term logarithm
            // keeps the absolute error small without slowing remote inputs.
            let logarithm = if (8.0..=24.0).contains(&x) {
                D::from(x).ln()
            } else {
                D::from(x.ln())
            };
            invert(base, logarithm, true)
        }
    };
    let mut e = Estimate::new(
        value,
        error,
        if spence {
            algorithm_component!("real_spence_bernoulli")
        } else {
            algorithm_component!("real_dilog_bernoulli")
        },
    );
    let derivative = if spence {
        if x == 1.0 { -1.0 } else { x.ln() / (1.0 - x) }
    } else if x == 0.0 {
        1.0
    } else if x < 1.0 {
        -(-x).ln_1p() / x
    } else {
        -(x - 1.0).ln() / x
    };
    e.derivative = derivative.is_finite().then_some(derivative);
    Ok(e)
}
