//! Positive logarithmic G7/K15 quadrature and half-line mappings.
//! Callers supply log integrands; these drivers own stopping and accounting.
use crate::distribution::LawOptions;
use crate::law::{charge, constant, constant as c, exhausted, finite, logadd};
use scioxide_core::{RealScalar, SciOxideError};

/// Logarithmic integrands may have exact zero mass, but no indeterminate or infinite density.
fn evaluate_log<S: RealScalar>(
    f: &mut impl FnMut(S) -> Result<S, SciOxideError>,
    x: S,
) -> Result<S, SciOxideError> {
    let value = f(x)?;
    if value.is_nan() || value == S::INFINITY {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "invalid positive log integrand",
        ));
    }
    Ok(value)
}

/// One adaptive log-space G7/K15 driver. The callback charges its complete work,
/// including nested integrals, before evaluation. The root allowance is one eighth
/// of the relative tolerance; children split that absolute allowance equally.
/// Discrepancies below 64 working-precision epsilons are rounding noise. The
/// 64-level stack bound is a numerical guard, independent of the evaluation budget.
/// The estimate is not a certified error bound.
pub(crate) fn log_integral<S: RealScalar>(
    a: S,
    b: S,
    tolerance: S,
    f: &mut impl FnMut(S) -> Result<S, SciOxideError>,
    floor: Option<S>,
) -> Result<S, SciOxideError> {
    if !tolerance.is_finite() || tolerance < constant::<S>(64.)? * S::EPSILON {
        return Err(crate::law::invalid(
            scioxide_core::argument!("options.tolerance"),
            "quadrature tolerance is below attainable precision",
        ));
    }
    if a == b {
        return Ok(S::NEG_INFINITY);
    }
    if !a.is_finite() || !b.is_finite() || a > b {
        return Err(crate::law::invalid(
            scioxide_core::argument!("interval"),
            "invalid quadrature interval",
        ));
    }
    adaptive_interval(a, b, tolerance, f, floor, 64)
}

fn adaptive_interval<S: RealScalar>(
    a: S,
    b: S,
    tolerance: S,
    f: &mut impl FnMut(S) -> Result<S, SciOxideError>,
    floor: Option<S>,
    depth: usize,
) -> Result<S, SciOxideError> {
    let (k, g) = rule(a, b, |x| evaluate_log(f, x))?;
    if k == S::NEG_INFINITY {
        return Ok(k);
    }
    let half = b * S::HALF - a * S::HALF;
    let value = finite(k + half.ln())?;
    let error = (g - k).exp_m1().abs();
    let floor = floor.unwrap_or(value + (tolerance * constant::<S>(0.125)?).ln());
    if error <= constant::<S>(64.)? * S::EPSILON || value + error.ln() <= floor {
        return Ok(value);
    }
    let mid = a * S::HALF + b * S::HALF;
    if depth == 0 || mid == a || mid == b {
        return Err(exhausted());
    }
    let child = Some(floor - S::LN_2);
    logadd(
        adaptive_interval(a, mid, tolerance, f, child, depth - 1)?,
        adaptive_interval(mid, b, tolerance, f, child, depth - 1)?,
    )
}

/// Charge a scalar kernel against the caller's remaining operation budget.
pub(crate) fn positive_integral<S: RealScalar>(
    a: S,
    b: S,
    tolerance: S,
    left: &mut usize,
    f: &impl Fn(S) -> Result<S, SciOxideError>,
    floor: Option<S>,
) -> Result<S, SciOxideError> {
    log_integral(
        a,
        b,
        tolerance,
        &mut |x| {
            charge(left)?;
            f(x)
        },
        floor,
    )
}

/// Map [0,1) to a half-line, preserving the log Jacobian and zero mass beyond
/// the coordinate range. Peak discovery belongs to the integrand's caller.
pub(crate) fn half_line<S: RealScalar>(
    start: S,
    sign: S,
    scale: S,
    tolerance: S,
    f: &mut impl FnMut(S) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    if !scale.is_finite() || scale <= S::ZERO {
        return Err(exhausted());
    }
    log_integral(
        S::ZERO,
        S::ONE,
        tolerance,
        &mut |t| {
            let q = S::ONE - t;
            let x = start + sign * scale * t / q;
            if !x.is_finite() {
                return Ok(S::NEG_INFINITY);
            }
            Ok(f(x)? + scale.ln() - S::TWO * q.ln())
        },
        None,
    )
}

/// Integrate a positive density in its natural coordinate from x toward one infinity.
/// Split explicitly around the mode so a narrow peak cannot disappear between nodes.
pub(super) fn tail<S: RealScalar>(
    f: impl Fn(S, &mut usize) -> Result<S, SciOxideError>,
    x: S,
    upper: bool,
    mode: S,
    width: S,
    options: LawOptions<S>,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let tol = options.tolerance.relative();
    // Tail integral of a log-density f from x outward. First integrate over
    // whichever of the fixed panels [mode - w, mode, mode + w] lie beyond x,
    // so the bulk near the mode is resolved on finite intervals; the
    // remaining semi-infinite part is then mapped onto [0, 1).
    let sign = if upper { S::ONE } else { -S::ONE };
    let mut start = x;
    let mut total = S::NEG_INFINITY;
    let points = if upper {
        [mode - width, mode, mode + width]
    } else {
        [mode + width, mode, mode - width]
    };
    for end in points {
        if sign * (end - start) > S::ZERO {
            let (a, b) = if upper { (start, end) } else { (end, start) };
            total = logadd(
                total,
                log_integral(
                    a,
                    b,
                    tol,
                    &mut |z| {
                        charge(left)?;
                        f(z, left)
                    },
                    None,
                )?,
            )?;
            start = end;
        }
    }
    let h = width * c::<S>(0.01)?;
    let first = evaluate_log(
        &mut |z| {
            charge(left)?;
            f(z, left)
        },
        start,
    )?;
    let next = evaluate_log(
        &mut |z| {
            charge(left)?;
            f(z, left)
        },
        start + sign * h,
    )?;
    // Match the mapping scale to a steep local tail, without enlarging it past width.
    let scale = if next < first {
        width.min(h / (first - next))
    } else {
        width
    };
    if !scale.is_finite() || scale <= S::ZERO {
        return Err(exhausted());
    }
    total = logadd(
        total,
        half_line(start, sign, scale, tol, &mut |z| {
            charge(left)?;
            f(z, left)
        })?,
    )?;
    crate::law::log_probability(total)
}

const X: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.20778495500789847,
    0.,
];
const K: [f64; 8] = [
    0.022935322010529225,
    0.06309209262997855,
    0.10479001032225018,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
];
const G: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927664,
    0.3818300505051189,
    0.4179591836734694,
];
/// Return unscaled logarithmic Kronrod/Gauss sums. The caller owns work charging,
/// adaptive subdivision, and multiplication by the interval half-width.
pub(super) fn rule<S: RealScalar>(
    a: S,
    b: S,
    mut evaluate: impl FnMut(S) -> Result<S, SciOxideError>,
) -> Result<(S, S), SciOxideError> {
    // Nodes and weights are the standard G7/K15 pair on [-1, 1]; every
    // other Kronrod abscissa (odd `i`) is a Gauss node, so both sums reuse the
    // same 15 evaluations. Sums use core logarithmic reductions so integrands
    // spanning hundreds of orders of magnitude never underflow. The midpoint
    // is formed as `a/2 + b/2` so it cannot overflow for huge endpoints.
    let mid = a * S::HALF + b * S::HALF;
    let half = b * S::HALF - a * S::HALF;
    let center = evaluate(mid)?;
    let mut kron = [S::NEG_INFINITY; 15];
    let mut gauss = [S::NEG_INFINITY; 7];
    kron[14] = center + constant::<S>(K[7])?.ln();
    gauss[6] = center + constant::<S>(G[3])?.ln();
    for (i, &node) in X[..7].iter().enumerate() {
        let dx = half * constant(node)?;
        let low = evaluate(mid - dx)?;
        let high = evaluate(mid + dx)?;
        let weight = constant::<S>(K[i])?.ln();
        kron[2 * i] = low + weight;
        kron[2 * i + 1] = high + weight;
        if i % 2 == 1 {
            let weight = constant::<S>(G[i / 2])?.ln();
            gauss[i - 1] = low + weight;
            gauss[i] = high + weight;
        }
    }
    // Reduce each panel once with core's checked logarithmic sum. Repeated
    // pairwise certified log additions would redo the reduction at every node.
    Ok((
        scioxide_core::logsumexp(&kron)?,
        scioxide_core::logsumexp(&gauss)?,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[test]
    fn quadrature_charges_before_calling_and_preserves_log_zero() {
        for budget in [14, 15] {
            let calls = Cell::new(0);
            let mut left = budget;
            let result = positive_integral(
                0_f64,
                1.,
                1e-8,
                &mut left,
                &|_| {
                    calls.set(calls.get() + 1);
                    Ok(0.)
                },
                None,
            );
            assert_eq!(calls.get(), budget);
            assert_eq!(left, 0);
            if budget == 15 {
                assert!(result.unwrap().abs() < 1e-14);
            } else {
                assert_eq!(
                    result.unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::ConvergenceFailure
                );
            }
        }
        assert_eq!(
            positive_integral(0., 1., 1e-8, &mut 15, &|_| Ok(f64::NEG_INFINITY), None).unwrap(),
            f64::NEG_INFINITY
        );
        assert!(positive_integral(0., 1., 1e-8, &mut 15, &|_| Ok(f64::NAN), None).is_err());
        assert!(positive_integral(0., 1., 1e-8, &mut 15, &|_| Ok(f64::INFINITY), None).is_err());
    }

    #[test]
    fn finite_and_half_line_integrals_share_accuracy() {
        let exact = (1. - (-2_f64).exp()).ln();
        let actual = positive_integral(0., 2., 1e-8, &mut 4096, &|x| Ok(-x), None).unwrap();
        assert!((actual - exact).abs() < 1e-12);
        for sign in [-1., 1.] {
            let mut left = 4096;
            let actual = half_line(3., sign, 0.2, 1e-8, &mut |x: f64| {
                charge(&mut left)?;
                Ok(-(x - 3.).abs() / 0.2 - 0.2_f64.ln())
            })
            .unwrap();
            assert!(actual.abs() < 1e-8);
            assert!(left < 4096);
        }
    }
}
