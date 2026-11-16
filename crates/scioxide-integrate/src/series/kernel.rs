//! Sequence transformations: Wynn's epsilon algorithm and Levin's u transform.
//! Wynn's recurrence is also used by adaptive extrapolation; callers choose and
//! scale its input.

use super::double::{Double, Wide};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

/// First even epsilon column beyond the unaccelerated partial sums.
pub(super) const FIRST_ACCELERATED_ORDER: usize = 2;

pub(crate) struct WynnEstimate<T: NumericScalar> {
    pub(crate) value: T,
    pub(crate) change: T::Real,
    pub(crate) accelerated: bool,
}

/// Wynn's epsilon algorithm on partial sums. At a zero or rounding-level
/// difference, the table stops and the last usable even column is returned.
pub(crate) fn wynn_epsilon<T: NumericScalar>(
    history: &[T],
    current: &mut [T],
    previous: &mut [T],
) -> Result<WynnEstimate<T>, SciOxideError> {
    let n = history.len();
    if n == 0 || current.len() < n || previous.len() < n {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "history and tableau storage must have equal nonzero capacity",
        ));
    }
    let mut current = &mut current[..n];
    let mut previous = &mut previous[..n];
    // Wynn's epsilon algorithm eps_(k+1)^(i) = eps_(k-1)^(i+1) + 1/(eps_k^(i+1) - eps_k^(i)),
    // with eps_(-1) = 0 and eps_0 = partial sums, stored as two columns only: `current` is
    // column k and `previous` column k-1. Entry i of the new column overwrites
    // `previous[i]` after `previous[i + 1]` has been read, then the columns swap roles.
    // Only even columns approximate the limit (odd ones are auxiliary reciprocals); the
    // newest even-column entry is the estimate and its change between even orders is the
    // extrapolation error.
    current[..n].copy_from_slice(history);
    previous[..n].fill(T::ZERO);
    let mut value = history[n - 1];
    let mut change = T::Real::ZERO;
    let mut accelerated = false;
    let ten = T::Real::try_from_f64_lossy(10.0)?;
    let hundred = T::Real::try_from_f64_lossy(100.0)?;
    for order in 1..n {
        let count = n - order;
        let mut valid = true;
        for i in 0..count {
            // A difference at roundoff level would turn noise into a huge reciprocal; the
            // table is truncated at the last trustworthy column instead.
            let difference = current[i + 1] - current[i];
            let floor = ten * T::Real::EPSILON * current[i + 1].abs().max(current[i].abs());
            if difference.abs() <= floor || difference == T::ZERO {
                valid = false;
                break;
            }
            let next = previous[i + 1] + T::ONE / difference;
            if !next.has_finite_components() {
                valid = false;
                break;
            }
            previous[i] = next;
        }
        if !valid {
            break;
        }
        if order % FIRST_ACCELERATED_ORDER == 0 {
            let candidate = previous[count - 1];
            // The first even column has no predecessor to compare with, so it reports a
            // nominal roundoff-sized change rather than claiming zero error.
            change = if accelerated {
                (candidate - value).abs()
            } else {
                hundred * T::Real::EPSILON * candidate.abs()
            };
            value = candidate;
            accelerated = true;
        }
        // Only the shrinking active prefix is read by the next order.
        // Swap column borrows rather than every stored tableau element.
        core::mem::swap(&mut current, &mut previous);
    }
    Ok(WynnEstimate {
        value,
        change,
        accelerated,
    })
}

/// Levin-u estimate and its rounding scale.
pub(super) struct LevinEstimate<T: NumericScalar> {
    pub(super) value: T,
    /// First-order effect of half-ulp perturbations of every term on the
    /// estimate: `epsilon / 2` times the Euclidean norm of the weighted
    /// remainders `w_j (L - S_j)` over the magnitude of the weight sum.
    pub(super) roundoff: T::Real,
}

fn overflow() -> SciOxideError {
    SciOxideError::size_overflow(scioxide_core::quantity!("levin_u_points"))
}

/// Significant bits of a positive integer, excluding trailing zeros.
fn significant_bits(value: u128) -> u32 {
    u128::BITS - value.leading_zeros() - value.trailing_zeros()
}

/// `base^exponent` by squaring; exact while the result fits the significand.
fn power<R: RealScalar>(base: R, mut exponent: u32) -> R {
    let mut result = R::ONE;
    let mut factor = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result *= factor;
        }
        exponent >>= 1;
        if exponent > 0 {
            factor = factor * factor;
        }
    }
    result
}

/// Levin's u transform (`beta = 1`) of the informative rows `terms` with
/// rounded partial sums `partials` at strictly increasing one-based term
/// coordinates `points`. `gaps[j]` is the exact sum of the omitted terms
/// between rows `j - 1` and `j` (zero without omitted rows; `gaps[0]` is
/// unused), so every partial-sum difference is an exact sum of terms.
///
/// The remainder model is `omega_j = x_j * a_j`. With consecutive points this
/// is the classical transform; otherwise the `k`-th finite difference becomes
/// the divided difference over the given points, so rows without information
/// (zero terms) may be omitted. The weight sum cancels by the conditioning,
/// which would amplify every working-precision weight rounding into the
/// estimate, so weights, their sums and the correction to the newest partial
/// sum are double words. Each weight is an exact integer coordinate factor
/// (formed in working precision when its significant bits fit, otherwise in
/// double words) over its term; terms and coordinates are normalized by powers
/// of two near the newest ones, which keeps the error-free products of a
/// window in range unless its term ratios approach the exponent range. What
/// remains is the amplified rounding of the terms themselves, reported as
/// `roundoff`. A nonfinite weight, a weight sum at roundoff level relative to
/// the weight mass, or a nonfinite estimate yields no candidate. `magnitudes`
/// is scratch of at least `terms.len()` values.
pub(super) fn levin_u<T: NumericScalar>(
    terms: &[T],
    gaps: &[Wide<T>],
    partials: &[T],
    points: &[usize],
    magnitudes: &mut [T::Real],
) -> Result<Option<LevinEstimate<T>>, SciOxideError> {
    let n = terms.len();
    if n < 2 || gaps.len() != n || partials.len() != n || points.len() != n || magnitudes.len() < n
    {
        return Ok(None);
    }
    // L = sum_j c_j x_j^(k-2) S_j / a_j / sum_j c_j x_j^(k-2) / a_j with the divided-difference
    // coefficients c_j = 1 / prod_(l != j) (x_j - x_l). Over the span R = x_k - x_0, the
    // integers R! c_j = (-1)^(k-j) C(R, x_j - x_0) prod_(m missing) |x_j - m| are exact; the
    // missing coordinates are those of omitted rows. Order one uses x_j^(-1) ~ x_(1-j). The
    // correction L - S_k is sum_j w_j D_j / sum_j w_j with D_j = S_j - S_k = -sum_(t > j) G_t,
    // where G_t is the term of row t plus the gap before it, so sum_j w_j D_j is
    // -sum_t G_t W_t over the prefix weight sums W_t = sum_(j < t) w_j.
    let order = n - 1;
    let first = points[0];
    let span = points[order] - first;
    let sparse = span > order;
    let exponent = |value: T::Real| scioxide_core::backend::standard::exponent(value);
    let unit = T::Real::ONE.scalbn(-exponent(T::Real::try_from_usize_exact(points[order])?)?);
    let limit = exponent(T::Real::MAX)?;
    let reference = exponent(scioxide_core::backend::standard::magnitude(terms[order]))?
        .clamp(1 - limit, limit - 2);
    let (scale, inverse) = (
        T::Real::ONE.scalbn(-reference),
        T::Real::ONE.scalbn(reference),
    );
    let scaled = |value: T| {
        T::from_components(
            value.real_component() * scale,
            value.imaginary_component() * scale,
        )
    };
    let degree = u32::try_from(order.saturating_sub(2)).map_err(|_| overflow())?;
    let mut binomial = 1_u64;
    let mut offset = 0;
    let mut mass = T::Real::ZERO;
    let mut denominator = Wide::ZERO;
    let mut numerator = Wide::ZERO;
    for (j, magnitude) in magnitudes[..n].iter_mut().enumerate() {
        let point = points[j];
        while offset < point - first {
            let factor = u64::try_from(span - offset).map_err(|_| overflow())?;
            binomial = binomial.checked_mul(factor).ok_or_else(overflow)? / (offset as u64 + 1);
            offset += 1;
        }
        let mut coefficient = u128::from(binomial);
        if sparse {
            for pair in points.windows(2) {
                for missing in pair[0] + 1..pair[1] {
                    coefficient = coefficient
                        .checked_mul(point.abs_diff(missing) as u128)
                        .ok_or_else(overflow)?;
                }
            }
        }
        let (coordinate, degree) = if order >= 2 {
            (point, degree)
        } else {
            (points[1 - j], 1)
        };
        // Exact: the coordinate is an integer and the unit a power of two.
        let base = T::Real::try_from_usize_exact(coordinate)? * unit;
        let bits = significant_bits(coordinate as u128)
            .saturating_mul(degree)
            .saturating_add(significant_bits(coefficient));
        let factor = if bits <= T::Real::MANTISSA_DIGITS {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                reason = "the coefficient fits the significand exactly"
            )]
            let coefficient = T::Real::try_from_f64_lossy(coefficient as u64 as f64)?;
            Double::new(coefficient * power(base, degree))
        } else {
            Double::integer(coefficient).mul(Double::power(base, degree))
        };
        let factor = if (order - j) % 2 == 1 {
            factor.neg()
        } else {
            factor
        };
        // A nonfinite weight reaches the mass and the sums, which are checked below.
        let term = scaled(terms[j]);
        let weight = Wide::quotient(factor, term);
        if j > 0 {
            let step = if sparse {
                Wide::of(term).add(gaps[j].scaled(scale))
            } else {
                Wide::of(term)
            };
            numerator = numerator.accumulate_product(step.neg(), denominator);
        }
        denominator = denominator.accumulate(weight);
        *magnitude = weight.leading().abs();
        mass += *magnitude;
    }
    let (denominator, numerator) = (denominator.renormalized(), numerator.renormalized());
    let size = denominator.leading().abs();
    let floor = T::Real::try_from_f64_lossy(32.0)? * T::Real::EPSILON;
    if !mass.is_finite() || !size.is_finite() || size <= floor * mass || !numerator.is_finite() {
        return Ok(None);
    }
    let correction = numerator.div(denominator);
    let newest = partials[order];
    let value = Wide::of(newest).add(correction.scaled(inverse)).value();
    if !value.has_finite_components() {
        return Ok(None);
    }
    // Euclidean norm of w_j (L - S_j) = w_j (correction - D_j), scaled against overflow by
    // the power of two of its largest entry.
    let leading = correction.leading();
    let mut largest = T::Real::ZERO;
    for (magnitude, &partial) in magnitudes[..n].iter_mut().zip(partials) {
        *magnitude *= (leading - scaled(partial - newest)).abs();
        largest = largest.max(*magnitude);
    }
    let norm = if largest.is_finite() {
        let top = exponent(largest)?.max(3 - limit);
        let shrink = T::Real::ONE.scalbn(-top);
        let squares = magnitudes[..n]
            .iter()
            .fold(T::Real::ZERO, |sum, &remainder| {
                let ratio = remainder * shrink;
                sum + ratio * ratio
            });
        squares.sqrt().scalbn(top)
    } else {
        largest
    };
    Ok(Some(LevinEstimate {
        value,
        roundoff: T::Real::HALF * T::Real::EPSILON * norm * inverse / size,
    }))
}
