//! Complete and incomplete elliptic integrals.
//!
//! This module provides the five standard elliptic integrals using
//! numerically stable algorithms:
//!
//! - [`crate::elliptic_k`] — complete elliptic integral of the first kind `K(m)`
//! - [`crate::elliptic_e`] — complete elliptic integral of the second kind `E(m)`
//! - [`crate::elliptic_f`] — incomplete elliptic integral of the first kind `F(phi|m)`
//! - [`crate::elliptic_e_inc`] — incomplete elliptic integral of the second kind `E(phi|m)`
//! - [`crate::elliptic_pi`] — complete elliptic integral of the third kind `Pi(n|m)`
//!
//! All functions use the parameter convention where `m = k^2` (the
//! *parameter*, not the *modulus* `k`).
//!
//! # Algorithms
//!
//! - **Fixed-cost polynomial/logarithmic approximations** for complete `K` and
//!   `E` on `m < 1` (using the imaginary-modulus transformation on `m < 0`).
//! - **Carlson symmetric forms** (`R_F`, `R_D`, `R_C`, and `R_J`) for
//!   incomplete integrals and complete `Pi`.
//!
//! Incomplete `E` and complete `Pi` share their `R_F` duplication state with
//! `R_D` and `R_J`, respectively.
//! A single amplitude reducer handles signs, quadrants, and complete periods.

use crate::support::Work;
use crate::support::Z;
use crate::support::{check_finite, require_finite_param};
use crate::util::{is_exact, is_exact_zero};
use scioxide_core::{AlgorithmComponent, RealScalar, SciOxideError, algorithm_component};

/// Maximum number of Carlson duplication iterations.
#[cfg(test)]
const MAX_ITER: usize = 100;

#[derive(Clone, Copy, Debug)]
struct CarlsonPair<S> {
    rf: S,
    companion: S,
    iterations: usize,
}

#[derive(Clone, Copy, Debug)]
struct KernelEvaluation<S> {
    value: S,
    iterations: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EllipticEvaluation<S> {
    pub(crate) value: S,
    pub(crate) iterations: usize,
    pub(crate) branch_id: AlgorithmComponent,
}

impl<S> EllipticEvaluation<S> {
    /// Whether the branch ran widened `f64` Carlson duplication.
    pub(crate) fn widened(&self) -> bool {
        matches!(
            self.branch_id.as_str(),
            "elliptic_f.carlson_rf" | "elliptic_e_inc.carlson_rf_rd" | "elliptic_pi.carlson_rf_rj"
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct ReducedAmplitude<S> {
    sign: S,
    local_sign: S,
    reduced: S,
    complete_periods: S,
}

// Test adapter for the work-accounted kernel below.
#[cfg(test)]
fn elliptic_k_evaluate<S: RealScalar>(m: S) -> Result<EllipticEvaluation<S>, SciOxideError> {
    elliptic_k_evaluate_with_work(m, &mut Work::elliptic(MAX_ITER, f64::EPSILON, None))
}

#[inline]
pub(crate) fn elliptic_k_evaluate_with_work<S: RealScalar>(
    m: S,
    work: &mut Work,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    work.check()?;
    complete_k(m)
}
/// Checked fixed-cost complete elliptic kernel, without work accounting or diagnostics.
#[inline]
pub(crate) fn complete_k<S: RealScalar>(m: S) -> Result<EllipticEvaluation<S>, SciOxideError> {
    validate_domain_k(m)?;
    if is_exact(m, S::ONE) {
        return Ok(EllipticEvaluation {
            value: S::INFINITY,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_k.exact_m_one"),
        });
    }

    Ok(EllipticEvaluation {
        value: S::try_from_f64_lossy(super::complete::k(m.into_f64()))?,
        iterations: 0,
        branch_id: algorithm_component!("complete_minimax"),
    })
}

// Test adapter for the work-accounted kernel below.
#[cfg(test)]
fn elliptic_e_evaluate<S: RealScalar>(m: S) -> Result<EllipticEvaluation<S>, SciOxideError> {
    elliptic_e_evaluate_with_work(m, &mut Work::elliptic(MAX_ITER, f64::EPSILON, None))
}

#[inline]
pub(crate) fn elliptic_e_evaluate_with_work<S: RealScalar>(
    m: S,
    work: &mut Work,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    work.check()?;
    complete_e(m)
}
/// Checked fixed-cost complete elliptic kernel, without work accounting or diagnostics.
#[inline]
pub(crate) fn complete_e<S: RealScalar>(m: S) -> Result<EllipticEvaluation<S>, SciOxideError> {
    validate_domain_e(m)?;

    // E(1) = 1 exactly; avoid the symmetric-form singular endpoint.
    if is_exact(m, S::ONE) {
        return Ok(EllipticEvaluation {
            value: S::ONE,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_e.exact_m_one"),
        });
    }

    Ok(EllipticEvaluation {
        value: S::try_from_f64_lossy(super::complete::e(m.into_f64()))?,
        iterations: 0,
        branch_id: algorithm_component!("complete_minimax"),
    })
}

// Test adapter for the work-accounted kernel below.
#[cfg(test)]
fn elliptic_f_evaluate<S: RealScalar>(
    phi: S,
    m: S,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    elliptic_f_evaluate_with_work(phi, m, &mut Work::elliptic(MAX_ITER, f64::EPSILON, None))
}

pub(crate) fn elliptic_f_evaluate_with_work<S: RealScalar>(
    phi: S,
    m: S,
    work: &mut Work,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    require_finite_param(phi, "elliptic parameter amplitude must be finite")?;
    validate_domain_incomplete(m)?;

    if is_exact(m, S::ONE) {
        let half_pi = S::PI * S::HALF;
        if is_exact(phi, half_pi) {
            return Ok(EllipticEvaluation {
                value: S::INFINITY,
                iterations: 0,
                branch_id: algorithm_component!("elliptic_f.exact_positive_pole"),
            });
        }
        if is_exact(phi, -half_pi) {
            return Ok(EllipticEvaluation {
                value: S::NEG_INFINITY,
                iterations: 0,
                branch_id: algorithm_component!("elliptic_f.exact_negative_pole"),
            });
        }
        if phi.abs() > half_pi {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("amplitude"),
                scioxide_core::DomainReason::Undefined,
                "the integration path crosses a singularity for m = 1",
            ));
        }
        return Ok(EllipticEvaluation {
            value: check_finite(phi.sin().atanh())?,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_f.exact_m_one"),
        });
    }

    if is_exact_zero(phi) {
        return Ok(EllipticEvaluation {
            value: phi,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_f.exact_phi_zero"),
        });
    }
    if is_exact_zero(m) {
        return Ok(EllipticEvaluation {
            value: phi,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_f.exact_m_zero"),
        });
    }

    let reduced = reduce_amplitude(phi);
    let half_pi = S::PI * S::HALF;
    let needs_complete = reduced.complete_periods > S::ZERO || is_exact(reduced.reduced, half_pi);
    let complete = if needs_complete {
        Some(elliptic_k_evaluate_with_work(m, work)?)
    } else {
        None
    };
    let local = if is_exact(reduced.reduced, half_pi) {
        KernelEvaluation {
            value: complete.map_or(S::ZERO, |evaluation| evaluation.value),
            iterations: 0,
        }
    } else {
        carlson_f(reduced.reduced, m, work)?
    };
    let period_value =
        complete.map_or(S::ZERO, |evaluation| evaluation.value) * reduced.complete_periods;
    let value = check_finite(reduced.sign * reduced.local_sign.mul_add(local.value, period_value))?;
    let iterations = local.iterations + complete.map_or(0, |evaluation| evaluation.iterations);
    let branch_id = if reduced.complete_periods > S::ZERO {
        algorithm_component!("elliptic_f.period_reduction")
    } else if is_exact(reduced.reduced, half_pi) {
        algorithm_component!("elliptic_f.complete_endpoint")
    } else {
        algorithm_component!("elliptic_f.carlson_rf")
    };
    Ok(EllipticEvaluation {
        value,
        iterations,
        branch_id,
    })
}

// Test adapter for the work-accounted kernel below.
#[cfg(test)]
fn elliptic_e_inc_evaluate<S: RealScalar>(
    phi: S,
    m: S,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    elliptic_e_inc_evaluate_with_work(phi, m, &mut Work::elliptic(MAX_ITER, f64::EPSILON, None))
}

pub(crate) fn elliptic_e_inc_evaluate_with_work<S: RealScalar>(
    phi: S,
    m: S,
    work: &mut Work,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    require_finite_param(phi, "elliptic parameter amplitude must be finite")?;
    validate_domain_incomplete(m)?;

    if is_exact_zero(phi) {
        return Ok(EllipticEvaluation {
            value: phi,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_e_inc.exact_phi_zero"),
        });
    }
    if is_exact_zero(m) {
        return Ok(EllipticEvaluation {
            value: phi,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_e_inc.exact_m_zero"),
        });
    }

    let reduced = reduce_amplitude(phi);
    let half_pi = S::PI * S::HALF;
    let needs_complete = reduced.complete_periods > S::ZERO || is_exact(reduced.reduced, half_pi);
    let complete = if needs_complete {
        Some(elliptic_e_evaluate_with_work(m, work)?)
    } else {
        None
    };
    let local = if is_exact(reduced.reduced, half_pi) {
        KernelEvaluation {
            value: complete.map_or(S::ZERO, |evaluation| evaluation.value),
            iterations: 0,
        }
    } else if is_exact(m, S::ONE) {
        KernelEvaluation {
            value: reduced.reduced.sin(),
            iterations: 0,
        }
    } else {
        carlson_e(reduced.reduced, m, work)?
    };
    let period_value =
        complete.map_or(S::ZERO, |evaluation| evaluation.value) * reduced.complete_periods;
    let value = check_finite(reduced.sign * reduced.local_sign.mul_add(local.value, period_value))?;
    let iterations = local.iterations + complete.map_or(0, |evaluation| evaluation.iterations);
    let branch_id = if reduced.complete_periods > S::ZERO {
        algorithm_component!("elliptic_e_inc.period_reduction")
    } else if is_exact(reduced.reduced, half_pi) {
        algorithm_component!("elliptic_e_inc.complete_endpoint")
    } else if is_exact(m, S::ONE) {
        algorithm_component!("elliptic_e_inc.exact_m_one")
    } else {
        algorithm_component!("elliptic_e_inc.carlson_rf_rd")
    };
    Ok(EllipticEvaluation {
        value,
        iterations,
        branch_id,
    })
}

// Test adapter for the work-accounted kernel below.
#[cfg(test)]
fn elliptic_pi_evaluate<S: RealScalar>(n: S, m: S) -> Result<EllipticEvaluation<S>, SciOxideError> {
    elliptic_pi_evaluate_with_work(n, m, &mut Work::elliptic(MAX_ITER, f64::EPSILON, None))
}

pub(crate) fn elliptic_pi_evaluate_with_work<S: RealScalar>(
    n: S,
    m: S,
    work: &mut Work,
) -> Result<EllipticEvaluation<S>, SciOxideError> {
    require_finite_param(n, "elliptic parameter characteristic must be finite")?;
    validate_domain_k(m)?;

    if n > S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("characteristic"),
            scioxide_core::DomainReason::Undefined,
            "n > 1 places a pole inside the integration interval",
        ));
    }
    if is_exact(n, S::ONE) || is_exact(m, S::ONE) {
        return Ok(EllipticEvaluation {
            value: S::INFINITY,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_pi.exact_pole"),
        });
    }

    // Pi(0 | m) = K(m)
    if is_exact_zero(n) {
        let complete = elliptic_k_evaluate_with_work(m, work)?;
        return Ok(EllipticEvaluation {
            value: complete.value,
            iterations: complete.iterations,
            branch_id: algorithm_component!("elliptic_pi.n_zero"),
        });
    }

    let one = S::ONE;
    let two = S::TWO;
    let parameter_complement = one - m;
    let characteristic_complement = one - n;

    if n < S::ZERO {
        let inverse_root = S::ONE / (-n).sqrt();
        if inverse_root <= <S as RealScalar>::EPSILON * <S as RealScalar>::EPSILON {
            return Ok(EllipticEvaluation {
                value: check_finite(S::PI * S::HALF * inverse_root)?,
                iterations: 0,
                branch_id: algorithm_component!("elliptic_pi.extreme_negative_asymptotic"),
            });
        }
    }

    if is_exact_zero(m) {
        return Ok(EllipticEvaluation {
            value: check_finite(S::PI / (two * characteristic_complement.sqrt()))?,
            iterations: 0,
            branch_id: algorithm_component!("elliptic_pi.exact_m_zero"),
        });
    }
    if is_exact(n, m) {
        let complete = elliptic_e_evaluate_with_work(m, work)?;
        return Ok(EllipticEvaluation {
            value: check_finite(complete.value / parameter_complement)?,
            iterations: complete.iterations,
            branch_id: algorithm_component!("elliptic_pi.equal_parameters"),
        });
    }

    let combination = carlson_rf_rj_combination(
        S::ZERO,
        parameter_complement,
        one,
        characteristic_complement,
        n,
        work,
    )?;
    Ok(EllipticEvaluation {
        value: combination.value,
        iterations: combination.iterations,
        branch_id: algorithm_component!("elliptic_pi.carlson_rf_rj"),
    })
}

// ---------------------------------------------------------------------------
// Domain validation helpers
// ---------------------------------------------------------------------------

/// Validate `m` for `K(m)` and `Pi(n|m)`: requires `m <= 1`.
fn validate_domain_k<S: RealScalar>(m: S) -> Result<(), SciOxideError> {
    require_finite_param(m, "elliptic parameter parameter must be finite")?;
    if m > S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("parameter"),
            scioxide_core::DomainReason::OutsideDomain,
            "the real branch requires m <= 1",
        ));
    }
    Ok(())
}

/// Validate `m` for `E(m)`: requires `m <= 1`.
fn validate_domain_e<S: RealScalar>(m: S) -> Result<(), SciOxideError> {
    require_finite_param(m, "elliptic parameter parameter must be finite")?;
    if m > S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("parameter"),
            scioxide_core::DomainReason::OutsideDomain,
            "real elliptic parameter must satisfy m <= 1",
        ));
    }
    Ok(())
}

/// Validate `m` for incomplete integrals: requires `m <= 1`.
fn validate_domain_incomplete<S: RealScalar>(m: S) -> Result<(), SciOxideError> {
    require_finite_param(m, "elliptic parameter parameter must be finite")?;
    if m > S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("parameter"),
            scioxide_core::DomainReason::OutsideDomain,
            "real elliptic parameter must satisfy m <= 1",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Amplitude reduction
// ---------------------------------------------------------------------------

/// Reduce a signed amplitude to `[0, pi/2]` and keep its complete-period
/// contribution as a scalar multiplier. No elliptic kernel is entered here.
fn reduce_amplitude<S: RealScalar>(phi: S) -> ReducedAmplitude<S> {
    let (sign, phi) = if phi < S::ZERO {
        (-S::ONE, -phi)
    } else {
        (S::ONE, phi)
    };
    let half_pi = S::PI * S::HALF;
    if phi <= half_pi {
        return ReducedAmplitude {
            sign,
            local_sign: S::ONE,
            reduced: phi,
            complete_periods: S::ZERO,
        };
    }
    let pi = S::PI;
    let n_half = (phi / pi).floor();
    let phi_red = phi - n_half * pi;
    let (phi_final, local_sign, extra_periods) = if phi_red > half_pi {
        (pi - phi_red, -S::ONE, n_half + S::ONE)
    } else {
        (phi_red, S::ONE, n_half)
    };
    ReducedAmplitude {
        sign,
        local_sign,
        reduced: phi_final,
        complete_periods: extra_periods * (S::TWO),
    }
}

// ---------------------------------------------------------------------------
// Carlson symmetric forms for incomplete integrals
// ---------------------------------------------------------------------------

/// Compute `F(phi | m)` via Carlson's `R_F`.
///
/// `F(phi | m) = sin(phi) * R_F(cos^2(phi), 1 - m*sin^2(phi), 1)`
fn carlson_f<S: RealScalar>(
    phi: S,
    m: S,
    work: &mut Work,
) -> Result<KernelEvaluation<S>, SciOxideError> {
    let one = S::ONE;

    let sin_phi = phi.sin();
    let cos_phi = phi.cos();

    let x = cos_phi * cos_phi;
    let y = one - m * sin_phi * sin_phi;
    let z = one;

    let rf = carlson_rf_with_work(x, y, z, work)?;
    Ok(KernelEvaluation {
        value: check_finite(sin_phi * rf.value)?,
        iterations: rf.iterations,
    })
}

/// Compute `E(phi | m)` via Carlson's `R_F` and `R_D`.
///
/// `E(phi | m) = sin(phi) * R_F(x, y, 1) - (m/3) * sin^3(phi) * R_D(x, y, 1)`
///
/// where `x = cos^2(phi)`, `y = 1 - m*sin^2(phi)`.
fn carlson_e<S: RealScalar>(
    phi: S,
    m: S,
    work: &mut Work,
) -> Result<KernelEvaluation<S>, SciOxideError> {
    let one = S::ONE;
    let three = one + one + one;

    let sin_phi = phi.sin();
    let cos_phi = phi.cos();

    let x = cos_phi * cos_phi;
    let y = one - m * sin_phi * sin_phi;
    let z = one;

    let pair = carlson_rf_rd_with_work(x, y, z, work)?;
    let sin3 = sin_phi * sin_phi * sin_phi;
    let result = (-(m / three) * sin3).mul_add(pair.companion, sin_phi * pair.rf);
    Ok(KernelEvaluation {
        value: check_finite(result)?,
        iterations: pair.iterations,
    })
}

// Legendre reductions share the real Carlson dispatch and extended-range fallback.
fn scaled_real<S: RealScalar>(value: S) -> Result<crate::support::Z, SciOxideError> {
    crate::support::Z::from_value(scioxide_core::Complex::new(value.try_to_f64_exact()?, 0.0))
}
fn project_scaled<S: RealScalar>(value: crate::support::Z) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(value.real().round()?)
}
fn carlson_rf_with_work<S: RealScalar>(
    x: S,
    y: S,
    z: S,
    work: &mut Work,
) -> Result<KernelEvaluation<S>, SciOxideError> {
    use crate::elliptic_family::elliptic_kernel::{Companion, duplication};
    let start = work.used;
    let a = [
        x.try_to_f64_exact()?,
        y.try_to_f64_exact()?,
        z.try_to_f64_exact()?,
    ];
    if crate::elliptic_family::elliptic_real::ordinary(&a) {
        let pair = crate::elliptic_family::elliptic_real::duplication(
            a[0],
            a[1],
            a[2],
            Companion::None,
            work,
        )?;
        return Ok(KernelEvaluation {
            value: S::try_from_f64_lossy(pair.rf)?,
            iterations: work.used - start,
        });
    }
    let pair = duplication(
        scaled_real(x)?,
        scaled_real(y)?,
        scaled_real(z)?,
        Companion::None,
        work,
    )?;
    Ok(KernelEvaluation {
        value: project_scaled(pair.rf.value)?,
        iterations: work.used - start,
    })
}
fn carlson_rf_rd_with_work<S: RealScalar>(
    x: S,
    y: S,
    z: S,
    work: &mut Work,
) -> Result<CarlsonPair<S>, SciOxideError> {
    use crate::elliptic_family::elliptic_kernel::{Companion, duplication};
    let start = work.used;
    let a = [
        x.try_to_f64_exact()?,
        y.try_to_f64_exact()?,
        z.try_to_f64_exact()?,
    ];
    if crate::elliptic_family::elliptic_real::ordinary(&a) {
        let pair = crate::elliptic_family::elliptic_real::duplication(
            a[0],
            a[1],
            a[2],
            Companion::Rd,
            work,
        )?;
        return Ok(CarlsonPair {
            rf: S::try_from_f64_lossy(pair.rf)?,
            companion: S::try_from_f64_lossy(pair.companion)?,
            iterations: work.used - start,
        });
    }
    let pair = duplication(
        scaled_real(x)?,
        scaled_real(y)?,
        scaled_real(z)?,
        Companion::Rd,
        work,
    )?;
    Ok(CarlsonPair {
        rf: project_scaled(pair.rf.value)?,
        companion: project_scaled(pair.companion.value)?,
        iterations: work.used - start,
    })
}
fn carlson_rf_rj_combination<S: RealScalar>(
    x: S,
    y: S,
    z: S,
    p: S,
    n: S,
    work: &mut Work,
) -> Result<KernelEvaluation<S>, SciOxideError> {
    use crate::elliptic_family::elliptic_kernel::{Companion, duplication};
    let start = work.used;
    let a = [
        x.try_to_f64_exact()?,
        y.try_to_f64_exact()?,
        z.try_to_f64_exact()?,
        p.try_to_f64_exact()?,
    ];
    let coefficient = n.try_to_f64_exact()? / 3.0;
    if crate::elliptic_family::elliptic_real::ordinary(&a)
        && crate::elliptic_family::elliptic_real::ordinary(&[coefficient])
    {
        let pair = crate::elliptic_family::elliptic_real::duplication(
            a[0],
            a[1],
            a[2],
            Companion::Rj(a[3]),
            work,
        )?;
        return Ok(KernelEvaluation {
            value: S::try_from_f64_lossy(coefficient.mul_add(pair.companion, pair.rf))?,
            iterations: work.used - start,
        });
    }
    let pair = duplication(
        scaled_real(x)?,
        scaled_real(y)?,
        scaled_real(z)?,
        Companion::Rj(scaled_real(p)?),
        work,
    )?;
    let correction = pair
        .companion
        .value
        .mul(scaled_real(n)?)?
        .mul(Z::from_value(scioxide_core::Complex::new(1.0 / 3.0, 0.0))?)?;
    Ok(KernelEvaluation {
        value: project_scaled(pair.rf.value.add(correction)?)?,
        iterations: work.used - start,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "elliptic_tests.rs"]
mod tests;
