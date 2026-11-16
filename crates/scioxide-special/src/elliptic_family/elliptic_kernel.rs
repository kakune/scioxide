//! Carlson duplication, extended-range estimates and work accounting.
//!
//! Carlson (1995), DLMF 19.36: fifth-degree corrections use the same truncation
//! target in bounded complex and extended-range coordinates. Right-half-plane
//! inputs with every nonzero Cartesian component in `[2^-100, 2^100]` use
//! ordinary complex arithmetic; other inputs keep component-preserving scales.
//! Nested RC checks its own range and shares the outer iteration allowance.
use crate::ValueWithError;
use crate::result::{Evaluation, Forms, Publish};
use crate::support::ELLIPTIC_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::{C, ONE, ZERO, real};
use crate::support::{EllipticEstimate as Estimate, Z};
use crate::support::{is_real, widen_numeric as widen};
use crate::{ScalarRepresentation, SpecialFunctionTermination};
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::scaled::factor_common_scale;
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, EstimateReliability, EvaluationCounts, LogValue,
    NumericScalar, RealScalar, ScaledValue, SciOxideError, algorithm_component,
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Family {
    Rf,
    Rd,
    Rj,
    Rc,
    Rg,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Component {
    Sn,
    Cn,
    Dn,
    Amplitude,
}

pub(crate) fn principal_amplitude(sn: C, cn: C) -> Result<C, SciOxideError> {
    if sn.norm() < 0.5 && cn.re > 0.0 {
        let delta = -sn * sn / (cn + ONE) + C::new(0.0, 1.0) * sn;
        return Ok(C::new(0.0, -1.0) * scioxide_core::xlog1py(ONE, delta)?);
    }
    let plus = cn + C::new(0.0, 1.0) * sn;
    let minus = cn - C::new(0.0, 1.0) * sn;
    // (cn+i sn)(cn-i sn)=1: take the larger term to avoid cancellation.
    if plus.norm() >= minus.norm() {
        Ok(C::new(0.0, -1.0) * plus.ln())
    } else {
        Ok(C::new(0.0, 1.0) * minus.ln())
    }
}
/// Narrow a result.
pub(crate) fn narrow_real_branch<T: NumericScalar>(x: C) -> Result<T, SciOxideError> {
    if is_real::<T>() && x.im != 0.0 {
        return Err(crate::support::instability(
            "computed elliptic real projection has a significant imaginary component",
        ));
    }
    crate::support::narrow(x)
}

/// Loosest duplication target. Duplication converges geometrically and feeds
/// quadrature integrands, so looser quadrature requests keep this truncation.
const DUPLICATION_TARGET: f64 = 1e-12;

/// Fixed accuracy target of the budget-only elliptic, Carlson and Jacobi
/// family: `1e-12` for `f64` and `1e-5` for `f32`. It
/// sets the adaptive quadrature target; duplication is capped at `1e-12`.
pub(crate) fn family_target<R: RealScalar>() -> f64 {
    if R::MANTISSA_DIGITS <= f32::MANTISSA_DIGITS {
        f64::from(1e-5_f32)
    } else {
        1e-12
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Companion<A = Z> {
    None,
    Rd,
    Rj(A),
}
pub(crate) struct Pair {
    pub rf: Estimate,
    pub companion: Estimate,
}
struct KernelPair {
    rf: Z,
    companion: Z,
    error: f64,
}
pub(crate) fn duplication(
    x: Z,
    y: Z,
    z: Z,
    kind: Companion,
    work: &mut Work<'_, '_, '_>,
) -> Result<Pair, SciOxideError> {
    let pair = duplicate(x, y, z, kind, true, work)?;
    Ok(Pair {
        rf: Estimate::new(pair.rf, pair.error),
        companion: Estimate::new(pair.companion, pair.error),
    })
}
/// Evaluate only the companion; combined Legendre/RG callers use `duplication`.
pub(crate) fn companion(
    x: Z,
    y: Z,
    z: Z,
    kind: Companion,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let pair = duplicate(x, y, z, kind, false, work)?;
    Ok(Estimate::new(pair.companion, pair.error))
}

/// Guard every Cartesian component before projecting out of extended range.
/// The right half-plane avoids cancellation in the real parts of the means
/// and duplication sums. At most cubic RJ products of these inputs have over
/// 700 exponent bits of headroom; nested RC arguments are checked separately.
fn ordinary_component(x: scioxide_core::Scaled<f64>) -> bool {
    x.is_zero()
        || ((-100..100).contains(&x.exponent())
            || x.exponent() == 100 && x.significand().abs() == 1.0)
}
fn ordinary_scaled(a: [Z; 4]) -> bool {
    a.into_iter().all(|z| {
        !z.real().is_sign_negative()
            && ordinary_component(z.real())
            && ordinary_component(z.imaginary())
    })
}
fn ordinary(a: [C; 4]) -> bool {
    const LOW: f64 = f64::from_bits((1023 - 100) << 52);
    const HIGH: f64 = f64::from_bits((1023 + 100) << 52);
    a.into_iter().all(|z| {
        z.re >= 0.0
            && [z.re, z.im]
                .into_iter()
                .all(|x| x == 0.0 || (LOW..=HIGH).contains(&x.abs()))
    })
}
fn duplicate(
    x: Z,
    y: Z,
    z: Z,
    kind: Companion,
    rf_needed: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<KernelPair, SciOxideError> {
    let p = match kind {
        Companion::Rj(p) => p,
        _ => z,
    };
    if ordinary_scaled([x, y, z, p]) {
        let kind = match kind {
            Companion::None => Companion::None,
            Companion::Rd => Companion::Rd,
            Companion::Rj(p) => Companion::Rj(p.round()?),
        };
        let pair = duplicate_ordinary(x.round()?, y.round()?, z.round()?, kind, rf_needed, work)?;
        return Ok(KernelPair {
            rf: Z::from_value(pair.0)?,
            companion: Z::from_value(pair.1)?,
            error: pair.2,
        });
    }
    duplicate_scaled(x, y, z, kind, rf_needed, work)
}
fn threshold(work: &Work<'_, '_, '_>) -> f64 {
    // Fifth-degree expansions: the neglected sixth-degree term is O(delta^6).
    (work.target.clamp(f64::EPSILON * 0.125, DUPLICATION_TARGET) * 0.01).powf(1.0 / 6.0)
}
fn scale(z: Z, a: f64) -> Result<Z, SciOxideError> {
    z.mul(Z::from_value(real(a))?)
}
fn deviation(mean: Z, value: Z) -> Result<C, SciOxideError> {
    if mean.is_zero() {
        Ok(ZERO)
    } else {
        mean.sub(value)?.div(mean)?.round()
    }
}
use crate::support::norm;
fn correction_rf(x: C, y: C, z: C) -> C {
    let e2 = x * y - z * z;
    let e3 = x * y * z;
    ONE - e2 * real(0.1) + e3 * real(1.0 / 14.0) + e2 * e2 * real(1.0 / 24.0)
        - e2 * e3 * real(3.0 / 44.0)
}
fn correction_companion(rd: bool, cx: C, cy: C, cz: C, cp: C) -> C {
    if rd {
        let e2 = cx * cy - cz * cz * real(6.0);
        let e3 = (cx * cy * real(3.0) - cz * cz * real(8.0)) * cz;
        let e4 = (cx * cy - cz * cz) * cz * cz * real(3.0);
        let e5 = cx * cy * cz * cz * cz;
        ONE - e2 * real(3.0 / 14.0) + e3 * real(1.0 / 6.0) + e2 * e2 * real(9.0 / 88.0)
            - e4 * real(3.0 / 22.0)
            - e2 * e3 * real(9.0 / 52.0)
            + e5 * real(3.0 / 26.0)
    } else {
        let ea = cx * (cy + cz) + cy * cz;
        let eb = cx * cy * cz;
        let ec = cp * cp;
        let ed = ea - ec * real(3.0);
        let ee = eb + cp * (ea - ec) * real(2.0);
        ONE + ed * (-real(3.0 / 14.0) + ed * real(9.0 / 88.0) - ee * real(9.0 / 52.0))
            + eb * (real(1.0 / 6.0) + cp * (-real(3.0 / 11.0) + cp * real(3.0 / 26.0)))
            + cp * ea * (real(1.0 / 3.0) - cp * real(3.0 / 22.0))
            - cp * ec * real(1.0 / 3.0)
    }
}
/// One duplication state, optionally accumulating the third-kind companion.
fn duplicate_scaled(
    mut x: Z,
    mut y: Z,
    mut z: Z,
    kind: Companion,
    rf_needed: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<KernelPair, SciOxideError> {
    let start = work.used;
    let mut p = match kind {
        Companion::Rj(p) => p,
        _ => z,
    };
    let mut sum = Z::ZERO;
    let mut factor = Z::from_value(ONE)?;
    let threshold = threshold(work);
    loop {
        work.try_step()?;
        let mean = if rf_needed {
            scale(x.add(y)?.add(z)?, 1.0 / 3.0)?
        } else {
            Z::ZERO
        };
        let dx = deviation(mean, x)?;
        let dy = deviation(mean, y)?;
        let dz = deviation(mean, z)?;
        let mut delta = if !rf_needed {
            0.0
        } else if mean.is_zero() {
            f64::INFINITY
        } else {
            norm(dx).max(norm(dy)).max(norm(dz))
        };
        let cm = match kind {
            Companion::None => mean,
            Companion::Rd => scale(x.add(y)?.add(scale(z, 3.0)?)?, 0.2)?,
            Companion::Rj(_) => scale(x.add(y)?.add(z)?.add(scale(p, 2.0)?)?, 0.2)?,
        };
        if cm.is_zero() {
            delta = f64::INFINITY;
        }
        let (cx, cy, cz, cp) = if matches!(kind, Companion::None) {
            (ZERO, ZERO, ZERO, ZERO)
        } else {
            let cx = deviation(cm, x)?;
            let cy = deviation(cm, y)?;
            let cz = deviation(cm, z)?;
            let cp = deviation(cm, p)?;
            delta = delta
                .max(norm(cx))
                .max(norm(cy))
                .max(norm(cz))
                .max(norm(cp));
            (cx, cy, cz, cp)
        };
        if delta < threshold {
            let rf = if rf_needed {
                Z::from_value(correction_rf(dx, dy, dz))?.div(mean.sqrt()?)?
            } else {
                Z::ZERO
            };
            let companion = if matches!(kind, Companion::None) {
                Z::ZERO
            } else {
                let correction =
                    correction_companion(matches!(kind, Companion::Rd), cx, cy, cz, cp);
                scale(sum, 3.0)?.add(
                    factor
                        .mul(Z::from_value(correction)?)?
                        .div(cm.mul(cm.sqrt()?)?)?,
                )?
            };
            let error = 8.0 * (work.used - start + 1) as f64 * f64::EPSILON + 4.0 * delta.powi(6);
            return Ok(KernelPair {
                rf,
                companion,
                error,
            });
        }
        let sx = x.sqrt()?;
        let sy = y.sqrt()?;
        let sz = z.sqrt()?;
        let lambda = sx.mul(sy.add(sz)?)?.add(sy.mul(sz)?)?;
        match kind {
            Companion::None => {}
            Companion::Rd => {
                sum = sum.add(factor.div(sz.mul(z.add(lambda)?)?)?)?;
            }
            Companion::Rj(_) => {
                let alpha = p.mul(sx.add(sy)?.add(sz)?)?.add(sx.mul(sy)?.mul(sz)?)?;
                let pl = p.add(lambda)?;
                let beta = p.mul(pl)?.mul(pl)?;
                let rc = duplicate(alpha.mul(alpha)?, beta, beta, Companion::None, true, work)?.rf;
                sum = sum.add(factor.mul(rc)?)?;
                p = scale(pl, 0.25)?;
            }
        }
        factor = scale(factor, 0.25)?;
        x = scale(x.add(lambda)?, 0.25)?;
        y = scale(y.add(lambda)?, 0.25)?;
        z = scale(z.add(lambda)?, 0.25)?;
        if matches!(kind, Companion::Rd) {
            p = z;
        }
    }
}

/// Duplication in bounded IEEE coordinates. Inputs have passed `ordinary` or
/// `ordinary_scaled`; no nonzero Cartesian component is lost on entry. Nested
/// RC calls select their representation before counting any of their work.
fn duplicate_ordinary(
    mut x: C,
    mut y: C,
    mut z: C,
    kind: Companion<C>,
    rf_needed: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<(C, C, f64), SciOxideError> {
    let start = work.used;
    let mut p = match kind {
        Companion::Rj(p) => p,
        _ => z,
    };
    // Equality includes the signed imaginary zero used by principal roots.
    let repeated = y.re.to_bits() == z.re.to_bits() && y.im.to_bits() == z.im.to_bits();
    let mut sum = ZERO;
    let mut sum_error = ZERO;
    let mut factor = 1.0;
    let threshold = threshold(work);
    loop {
        work.try_step()?;
        let total = x + y + z;
        let mean = total / 3.0;
        let (dx, dy, dz) = if rf_needed && mean != ZERO {
            let dx = (mean - x) / mean;
            let dy = (mean - y) / mean;
            (dx, dy, if repeated { dy } else { (mean - z) / mean })
        } else {
            (ZERO, ZERO, ZERO)
        };
        let mut delta = if !rf_needed {
            0.0
        } else if mean == ZERO {
            f64::INFINITY
        } else {
            norm(dx).max(norm(dy)).max(norm(dz))
        };
        let cm = match kind {
            Companion::None => mean,
            Companion::Rd => (x + y + z * 3.0) * 0.2,
            Companion::Rj(_) => (total + p * 2.0) * 0.2,
        };
        let (cx, cy, cz, cp) = if matches!(kind, Companion::None) || cm == ZERO {
            (ZERO, ZERO, ZERO, ZERO)
        } else {
            let cx = (cm - x) / cm;
            let cy = (cm - y) / cm;
            let cz = if repeated { cy } else { (cm - z) / cm };
            let cp = if matches!(kind, Companion::Rd) {
                cz
            } else {
                (cm - p) / cm
            };
            delta = delta
                .max(norm(cx))
                .max(norm(cy))
                .max(norm(cz))
                .max(norm(cp));
            (cx, cy, cz, cp)
        };
        if cm == ZERO {
            delta = f64::INFINITY;
        }
        if delta < threshold {
            let rf = if rf_needed {
                correction_rf(dx, dy, dz) / mean.sqrt()
            } else {
                ZERO
            };
            let companion = if matches!(kind, Companion::None) {
                ZERO
            } else {
                let correction =
                    correction_companion(matches!(kind, Companion::Rd), cx, cy, cz, cp);
                (sum + sum_error) * 3.0 + correction * factor / (cm * cm.sqrt())
            };
            if !rf.is_finite() || !companion.is_finite() {
                return Err(ERRORS.unstable());
            }
            return Ok((
                rf,
                companion,
                8.0 * (work.used - start + 1) as f64 * f64::EPSILON + 4.0 * delta.powi(6),
            ));
        }
        let sx = x.sqrt();
        let sy = y.sqrt();
        let sz = if repeated { sy } else { z.sqrt() };
        let lambda = sx * (sy + sz) + sy * sz;
        let term = match kind {
            Companion::None => ZERO,
            Companion::Rd => real(factor) / (sz * (z + lambda)),
            Companion::Rj(_) => {
                let alpha = p * (sx + sy + sz) + sx * sy * sz;
                let pl = p + lambda;
                let beta = p * pl * pl;
                let square = alpha * alpha;
                let rc = if ordinary([square, beta, beta, beta]) {
                    duplicate_ordinary(square, beta, beta, Companion::None, true, work)?.0
                } else {
                    let a = Z::from_value(square)?;
                    let b = Z::from_value(beta)?;
                    duplicate_scaled(a, b, b, Companion::None, true, work)?
                        .rf
                        .round()?
                };
                p = pl * 0.25;
                rc * factor
            }
        };
        // Componentwise Neumaier accumulation retains small RD/RJ terms.
        let next = sum + term;
        sum_error += C::new(
            if sum.re.abs() >= term.re.abs() {
                (sum.re - next.re) + term.re
            } else {
                (term.re - next.re) + sum.re
            },
            if sum.im.abs() >= term.im.abs() {
                (sum.im - next.im) + term.im
            } else {
                (term.im - next.im) + sum.im
            },
        );
        sum = next;
        factor *= 0.25;
        x = (x + lambda) * 0.25;
        y = (y + lambda) * 0.25;
        z = if repeated { y } else { (z + lambda) * 0.25 };
        if matches!(kind, Companion::Rd) {
            p = z;
        }
        if ![x, y, z, p, sum, sum_error].into_iter().all(C::is_finite) {
            return Err(ERRORS.unstable());
        }
    }
}

/// Evaluation of an elliptic, Carlson or Jacobi function.
pub(crate) type EllipticEvaluation<T> = Evaluation<T, EllipticForm<T>>;

/// A normalized estimate `value·2^exponent` with its relative and absolute errors.
#[derive(Clone, Copy)]
pub(crate) enum EllipticForm<T> {
    Real {
        value: f64,
        error: f64,
    },
    Scaled {
        e: Estimate,
        value: T,
        mantissa: C,
        exponent: i64,
        relative: f64,
    },
}

pub(crate) fn report<T: NumericScalar>(
    estimate: Result<Estimate, SciOxideError>,
    work: &Work<'_, '_, '_>,
    method: AlgorithmComponent,
) -> Result<EllipticEvaluation<T>, SciOxideError> {
    let algorithm = AlgorithmId::from_components(work.family, method, 1);
    if work.exhausted() {
        return Err(crate::result::exhausted_work(work.used));
    }
    let e = estimate?;
    let (parts, mut exponent) = factor_common_scale([e.value.real(), e.value.imaginary()])?;
    let mut mantissa = C::new(parts[0].round()?, parts[1].round()?);
    if mantissa.norm() >= 1.0 {
        mantissa *= real(0.5);
        exponent += 1;
    }
    // Quantize before normalization: rounding f32 can carry into the exponent.
    let mut value = narrow_real_branch::<T>(mantissa)?;
    if value.abs() >= T::Real::ONE {
        value = value * T::from_real(T::Real::HALF);
        exponent += 1;
        mantissa *= real(0.5);
    }
    let rounding = (widen(value, "elliptic parameter value must be finite")? - mantissa).norm();
    let relative = e.relative
        + if mantissa == ZERO {
            0.0
        } else {
            rounding / mantissa.norm()
        };
    let power = i32::try_from(exponent).map_err(|_| ERRORS.unstable())?;
    // Componentwise scalbn avoids an overflowing intermediate 2^exponent.
    let projected = T::from_components(
        value.real_component().scalbn(power),
        value.imaginary_component().scalbn(power),
    );
    let representation = if projected.has_finite_components() {
        ScalarRepresentation::Finite
    } else {
        ScalarRepresentation::Overflow
    };
    let termination = if e.exact && rounding == 0.0 {
        SpecialFunctionTermination::Exact
    } else {
        SpecialFunctionTermination::Converged
    };
    Ok(Evaluation::from_parts(
        (representation == ScalarRepresentation::Finite).then_some(projected),
        representation,
        termination,
        EvaluationCounts::from_iterations(work.used),
        algorithm,
        Some(EllipticForm::Scaled {
            e,
            value,
            mantissa,
            exponent,
            relative,
        }),
    ))
}

/// Carry a finite real kernel result without normalizing it through scaled
/// complex arithmetic. Out-of-destination-range results keep the scaled forms.
pub(crate) fn report_real<T: NumericScalar>(
    estimate: Result<(f64, f64), SciOxideError>,
    work: &Work<'_, '_, '_>,
    method: AlgorithmComponent,
    exact: bool,
) -> Result<EllipticEvaluation<T>, SciOxideError> {
    let algorithm = AlgorithmId::from_components(work.family, method, 1);
    if work.exhausted() {
        return Err(crate::result::exhausted_work(work.used));
    }
    let (value, error) = estimate?;
    if !value.is_finite() || !error.is_finite() {
        return Err(ERRORS.unstable());
    }
    // A rounded-away uncertainty is not an exact mathematical identity.
    let error = if exact {
        error
    } else {
        error.max(f64::from_bits(1))
    };
    let narrowed = T::Real::try_from_f64_lossy(value).unwrap_or(T::Real::INFINITY);
    if !narrowed.is_finite() || (narrowed == T::Real::ZERO && value != 0.0) {
        let mut e = Estimate::new(Z::from_value(real(value))?, 0.0);
        e.absolute = error;
        return report(Ok(e), work, method);
    }
    Ok(Evaluation::from_parts(
        Some(T::from_real(narrowed)),
        ScalarRepresentation::Finite,
        if exact {
            SpecialFunctionTermination::Exact
        } else {
            SpecialFunctionTermination::Converged
        },
        EvaluationCounts::from_iterations(work.used),
        algorithm,
        Some(EllipticForm::Real { value, error }),
    ))
}

impl<T: NumericScalar> Publish<T> for EllipticForm<T> {
    fn publish(self, ordinary: Option<T>) -> Result<Forms<T>, SciOxideError> {
        if let Self::Real { value, error } = self {
            let mut e = Estimate::new(Z::from_value(real(value))?, 0.0);
            e.absolute = error;
            let work = Work::elliptic(0, 0.0, None);
            let evaluation = report::<T>(Ok(e), &work, algorithm_component!("real_publication"))?;
            return evaluation
                .form()
                .copied()
                .ok_or_else(|| ERRORS.unstable())?
                .publish(ordinary);
        }
        let Self::Scaled {
            e,
            value,
            mantissa,
            exponent,
            relative,
        } = self
        else {
            unreachable!()
        };
        let scaled = ScaledValue::from_parts(value, exponent)?;
        let power = i32::try_from(exponent).map_err(|_| ERRORS.unstable())?;
        let max = T::Real::MAX.try_to_f64_exact()?;
        let bound = crate::result::outward::<T::Real>;
        let subnormal_projection = ordinary.is_some_and(|projected| {
            (value.real_component() != T::Real::ZERO
                && projected.real_component().abs() < T::Real::MIN_POSITIVE)
                || (value.imaginary_component() != T::Real::ZERO
                    && projected.imaginary_component().abs() < T::Real::MIN_POSITIVE)
        });
        let projection_error = match ordinary {
            Some(projected) if subnormal_projection => {
                let ordinary = widen(projected, "elliptic parameter projected must be finite")?;
                let difference = (C::new(ordinary.re.scalbn(-power), ordinary.im.scalbn(-power))
                    - widen(value, "elliptic parameter value must be finite")?)
                .norm();
                if difference == 0.0 {
                    0.0
                } else {
                    difference.scalbn(power).max(f64::from_bits(1))
                }
            }
            _ => 0.0,
        };
        let absolute = e.absolute + relative * mantissa.norm().scalbn(power) + projection_error;
        let reliability = EstimateReliability::ModelBased;
        let ordinary = ordinary
            .map(|projected| {
                let magnitude = projected.abs();
                let norm = if magnitude.is_finite() {
                    magnitude.try_to_f64_exact()?
                } else {
                    max
                };
                ValueWithError::new(
                    projected,
                    bound(absolute)?,
                    Some(bound(
                        relative
                            + if norm > 0.0 {
                                projection_error / norm
                            } else {
                                0.0
                            },
                    )?),
                    reliability,
                )
            })
            .transpose()?;
        let magnitude = scaled.scaled_magnitude().try_to_f64_exact()?;
        // An absolute part beyond the scaled coordinate range leaves the scaled
        // and logarithmic forms without a meaningful estimate.
        let scaled_error = scaled
            .scale_absolute_error(bound(e.absolute)?)
            .ok()
            .map(|absolute| {
                absolute
                    .try_to_f64_exact()
                    .map(|absolute| relative * magnitude + absolute)
            })
            .transpose()?;
        // The binary64 log magnitude and its rounding when stored in `T`: a few
        // binary64 roundings of ln, the exponent product and their sum, then
        // the destination rounding.
        let (logarithm, log_rounding) = if value == T::ZERO {
            (LogValue::from_value(T::ZERO)?, 0.0)
        } else {
            let wide =
                value.abs().try_to_f64_exact()?.ln() + exponent as f64 * core::f64::consts::LN_2;
            let stored = T::Real::try_from_f64_lossy(wide)?;
            (
                LogValue::from_log(value / T::from_real(value.abs()), stored)?,
                (stored.into_f64() - wide).abs() + 2.0 * f64::EPSILON * wide.abs(),
            )
        };
        let log_error = scaled_error.and_then(|error| {
            if magnitude == 0.0 {
                (error == 0.0).then_some(0.0)
            } else {
                crate::result::log_error(error / magnitude, 0.0).map(|error| error + log_rounding)
            }
        });
        Ok(Forms {
            value: ordinary,
            scaled: scaled_error
                .map(|error| {
                    ValueWithError::new(scaled, bound(error)?, Some(bound(relative)?), reliability)
                })
                .transpose()?,
            logarithmic: log_error
                .map(|error| ValueWithError::new(logarithm, bound(error)?, None, reliability))
                .transpose()?,
            derivative: None,
        })
    }
}

/// Extended-range exp(z);
pub(crate) fn scaled_exp(z: C) -> Result<Z, SciOxideError> {
    let magnitude = scioxide_core::Scaled::from_log(1.0, z.re)?;
    let (sin, cos) = z.im.sin_cos();
    Ok(Z::from_components(
        magnitude.mul(scioxide_core::Scaled::from_value(cos)?)?,
        magnitude.mul(scioxide_core::Scaled::from_value(sin)?)?,
    ))
}

/// DLMF 19.12: the two-term endpoint expansions, with a caller-supplied
/// logarithm L=log(4)-log(1-m)/2 (principal complex log for complex parameters).
pub(crate) fn endpoint_series<T: NumericScalar>(delta: T, log: T) -> Result<(T, T), SciOxideError> {
    let c = |x| T::Real::try_from_f64_lossy(x).map(T::from_real);
    let square = delta * delta;
    Ok((
        log + delta * c(0.25)? * (log - T::ONE) + square * c(9.0 / 64.0)? * (log - c(7.0 / 6.0)?),
        T::ONE
            + delta * c(0.5)? * (log - c(0.5)?)
            + square * c(3.0 / 16.0)? * (log - c(13.0 / 12.0)?),
    ))
}
