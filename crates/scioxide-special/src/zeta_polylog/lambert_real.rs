//! Scalar Halley kernels for the real branches; DLMF 4.13.
//!
//! Near -1/e, refine u = W + 1 using the compensated branch coordinate and
//! series shared with the complex kernel. Away from it, exponential and
//! logarithmic residuals avoid forming W*exp(W). Values and diagnostics stay
//! real throughout evaluation and publication.

use super::{PUISEUX_COEFFICIENTS, refine};
use crate::support::TranscendentalEstimate as Estimate;
use crate::support::Work;
use core::f64::consts::E;
#[cfg(test)]
use scioxide_core::ConvergenceReason;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "required when elementary methods use the no_std scalar trait"
)]
use scioxide_core::RealScalar;
use scioxide_core::{SciOxideError, algorithm_component};

fn branch(
    q: f64,
    lower: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate<f64>, SciOxideError> {
    let p = (2.0 * q).sqrt() * if lower { -1.0 } else { 1.0 };
    let mut u: f64 = 0.0;
    for &coefficient in PUISEUX_COEFFICIENTS.iter().rev() {
        work.try_step()?;
        u = u.mul_add(p, coefficient);
    }
    refine(
        u * p,
        -1.0,
        None,
        algorithm_component!("real_branch_point_halley"),
        |u| super::branch_residual(u, q, |u| u.exp()),
        work,
    )
}

pub(super) fn lambert(
    x: f64,
    lower: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate<f64>, SciOxideError> {
    let mut e = if !lower && x.abs() < 1e-8 {
        // The omitted -8*x^4/3 term is below one binary64 rounding unit.
        work.try_step()?;
        let value = x.mul_add(x.mul_add(1.5 * x, -x), x);
        Estimate::new(
            value,
            4.0 * f64::EPSILON * x.abs(),
            algorithm_component!("real_lambert_small"),
        )
    } else {
        let q = if x < 0.0 {
            x.mul_add(E, 1.0) + x * 1.4456468917292502e-16
        } else {
            1.0
        };
        if q <= 0.2 {
            branch(q, lower, work)?
        } else if lower {
            let l = (-x).ln();
            refine(
                l - (-l).ln(),
                0.0,
                None,
                algorithm_component!("real_lambert_log_halley"),
                |w| {
                    let logarithm = (-w).ln();
                    (
                        (w - l) + logarithm,
                        1.0 + 1.0 / w,
                        -1.0 / w / w,
                        8.0 * f64::EPSILON * (w.abs() + l.abs() + logarithm.abs()),
                    )
                },
                work,
            )?
        } else {
            let initial = if x < 0.5 {
                x * (1.0 - x + 1.5 * x * x)
            } else if x < 2.0 {
                0.5671432904097838 + 0.36 * (x - 1.0)
            } else {
                let l = x.ln();
                l - l.ln()
            };
            refine(
                initial,
                0.0,
                None,
                algorithm_component!("real_lambert_halley"),
                |w| {
                    let t = x * (-w).exp();
                    (w - t, 1.0 + t, -t, 8.0 * f64::EPSILON * (w.abs() + t.abs()))
                },
                work,
            )?
        }
    };
    let w = e.value;
    e.derivative = Some(if x == 0.0 { 1.0 } else { w / x / (1.0 + w) });
    e.derivative_error = if x == 0.0 {
        0.0
    } else {
        e.error / x.abs() / (1.0 + w).powi(2)
    };
    Ok(e)
}

pub(super) fn omega(x: f64, work: &mut Work<'_, '_, '_>) -> Result<Estimate<f64>, SciOxideError> {
    if x == 1.0 {
        let mut e = Estimate::new(1.0, 0.0, algorithm_component!("identity"));
        e.derivative = Some(0.5);
        return Ok(e);
    }
    if x <= -36.0 {
        work.try_step()?;
        // W(exp(x)) = exp(x) - exp(2x) + ..., with exp(x) < 2*epsilon.
        let value = x.exp();
        let mut e = Estimate::new(
            value,
            (8.0 * f64::EPSILON * value).max(f64::from_bits(1)),
            algorithm_component!("real_omega_small"),
        );
        e.derivative = Some(value);
        e.logarithm = Some(x);
        return Ok(e);
    }
    let mut e = if x < 0.0 {
        let t = x.exp();
        refine(
            t / (1.0 + t),
            0.0,
            None,
            algorithm_component!("real_omega_exp_halley"),
            |w| {
                let t = t * (-w).exp();
                (w - t, 1.0 + t, -t, 8.0 * f64::EPSILON * (w + t))
            },
            work,
        )?
    } else {
        let initial = if x <= 1.0 { 0.5 + 0.5 * x } else { x - x.ln() };
        refine(
            initial,
            0.0,
            None,
            algorithm_component!("real_omega_log_halley"),
            |w| {
                let logarithm = w.ln();
                (
                    (w - x) + logarithm,
                    1.0 + 1.0 / w,
                    -1.0 / w / w,
                    8.0 * f64::EPSILON * w
                        + 8.0 * f64::EPSILON * x
                        + 8.0 * f64::EPSILON * logarithm.abs(),
                )
            },
            work,
        )?
    };
    let w = e.value;
    e.derivative = Some(w / (1.0 + w));
    e.derivative_error = e.error / (1.0 + w).powi(2);
    Ok(e)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn real_cycles_require_a_small_residual() {
        for accurate in [false, true] {
            let mut work = Work::new(32);
            let result = refine(
                1.0,
                0.0,
                None,
                algorithm_component!("real_cycle_test"),
                |w| {
                    let step = if accurate {
                        if w == 1.0 {
                            -f64::EPSILON
                        } else {
                            f64::EPSILON
                        }
                    } else {
                        2.0 * w
                    };
                    (step, 1.0, 0.0, f64::EPSILON)
                },
                &mut work,
            );
            assert_eq!(work.used, 2);
            if accurate {
                let estimate = result.unwrap();
                assert!((estimate.value - 1.0).abs() <= f64::EPSILON);
                assert!(estimate.error.is_finite());
            } else {
                assert!(matches!(
                    result.err().unwrap().view(),
                    scioxide_core::SciOxideErrorView::ConvergenceFailure {
                        reason: ConvergenceReason::Stagnation,
                        ..
                    }
                ));
            }
        }
    }
}
