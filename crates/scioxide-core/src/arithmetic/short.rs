//! Correctly rounded short sums of products, with a fallback signal for exact accumulation.
use super::{HARDWARE_FMA, two_sum};
use crate::RealScalar;

// The transforms require destination-format intermediates, excluding x87's
// excess precision just as the portable fused-operation implementation does.
const EFT_ARITHMETIC: bool = cfg!(any(not(target_arch = "x86"), target_feature = "sse2"));

/// Returns `a b + c d` correctly rounded, or `None` if correct rounding cannot
/// be proved cheaply.
///
/// The products are formed error-free (with a hardware FMA, or by Dekker
/// splitting otherwise). `None` is returned when the range or rounding check is
/// inconclusive, including for non-finite inputs; the caller must then use exact
/// accumulation such as [`checked_sum_of_products`](crate::checked_sum_of_products).
/// An exact zero is `+0`.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::certified_sum_two_products;
///
/// assert_eq!(certified_sum_two_products(1.5_f64, 2.5, 3.5, 4.5), Some(19.5));
/// ```
#[inline]
pub fn certified_sum_two_products<S: RealScalar>(a: S, b: S, c: S, d: S) -> Option<S> {
    if HARDWARE_FMA {
        certified_sum_two_products_with(a, b, c, d, |a, b, c| a.mul_add(b, c))
    } else {
        contraction(a, b, c, d, product_residual)
    }
}

/// Returns `Σ aᵢ bᵢ` over at most four pairs, correctly rounded, or `None` if
/// correct rounding cannot be proved cheaply.
///
/// The products are formed error-free and summed with compensation; the result
/// is returned only if an enclosure of the exact sum lies inside one rounding
/// interval. `None` is returned for more than four pairs, non-finite factors,
/// values outside the guarded range and results near a rounding midpoint; the
/// caller must then use exact accumulation. An empty input or exact zero gives
/// `+0`.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::certified_short_sum_products;
///
/// let terms = [(1.5_f64, 2.5), (3.5, 4.5), (0.25, 0.5)];
/// assert_eq!(certified_short_sum_products(&terms), Some(19.625));
/// ```
#[inline(always)]
pub fn certified_short_sum_products<S: RealScalar>(terms: &[(S, S)]) -> Option<S> {
    if !EFT_ARITHMETIC || terms.len() > 4 {
        return None;
    }
    if let [(a, b), (c, d)] = terms {
        return certified_sum_two_products(*a, *b, *c, *d);
    }
    if let [a, b, c, d] = terms {
        return four_products([*a, *b, *c, *d]);
    }
    let mut high = S::ZERO;
    let mut low = S::ZERO;
    let mut mass = S::ZERO;
    for &(a, b) in terms {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        if a == S::ZERO || b == S::ZERO {
            continue;
        }
        if !admitted(a) || !admitted(b) {
            return None;
        }
        let p = a * b;
        let ep = product_residual(a, b, p);
        let (sum, e) = two_sum(high, p);
        high = sum;
        low = (low + e) + ep;
        mass = (mass + e.abs()) + ep.abs();
    }
    rounded_expansion(high, low, mass)
}

/// Independent products and a balanced TwoSum tree shorten the dependency
/// chain of four-column fits while keeping every exact expansion component.
#[inline(always)]
fn four_products<S: RealScalar>(terms: [(S, S); 4]) -> Option<S> {
    if !terms
        .iter()
        .all(|&(a, b)| (a == S::ZERO || admitted(a)) && (b == S::ZERO || admitted(b)))
    {
        return None;
    }
    let products = terms.map(|(a, b)| a * b);
    let residuals =
        core::array::from_fn::<_, 4, _>(|i| product_residual(terms[i].0, terms[i].1, products[i]));
    let (left, le) = two_sum(products[0], products[1]);
    let (right, re) = two_sum(products[2], products[3]);
    let (high, error) = two_sum(left, right);
    let errors = [
        residuals[0],
        residuals[1],
        residuals[2],
        residuals[3],
        le,
        re,
        error,
    ];
    let low = errors.into_iter().fold(S::ZERO, |sum, x| sum + x);
    let mass = errors.into_iter().fold(S::ZERO, |sum, x| sum + x.abs());
    rounded_expansion(high, low, mass)
}

#[inline(always)]
fn rounded_expansion<S: RealScalar>(high: S, low: S, mass: S) -> Option<S> {
    if mass == S::ZERO {
        return Some(if high == S::ZERO { S::ZERO } else { high });
    }
    let (rounded, tail) = two_sum(high, low);
    // At most eight additions accumulate the low parts. Their rounding error
    // is <= gamma_8 times their exact absolute mass. 16*EPSILON times the
    // rounded mass exceeds this including its own reduction's rounding. The
    // admitted product quantum keeps every nonzero bound normal.
    let four = S::TWO * S::TWO;
    let error = mass * (S::EPSILON * (four * four));
    if rounded.is_normal() {
        let magnitude = rounded.abs();
        let margin = (magnitude - magnitude.next_after(S::ZERO)) * S::HALF;
        if tail.abs() + error < margin {
            return Some(rounded);
        }
    }
    None
}

#[inline(always)]
fn admitted<S: RealScalar>(value: S) -> bool {
    let (lower, upper) = if S::MANTISSA_DIGITS == 24 {
        (5.960_464_477_539_063e-8, 16_777_216.0)
    } else {
        (6.223_015_277_861_142e-61, 1.606_938_044_258_990_3e60)
    };
    let value = value.into_f64().abs();
    value >= lower && value <= upper
}

/// Returns the rounded product `a b` and its rounding error (TwoProduct).
///
/// With [`HARDWARE_FMA`](crate::HARDWARE_FMA) the error is one inline fused
/// operation. Otherwise it uses Dekker/Veltkamp splitting (Ogita, Rump and
/// Oishi 2005, Algorithm 3.3), never a library or software FMA. If both
/// operands are finite with magnitude at most `2^(MAX_EXP − MANTISSA_DIGITS)`
/// and the product is zero or at least `2^(MIN_EXP + 2 MANTISSA_DIGITS)` in
/// magnitude, then `product + residual` equals `a b` exactly. The function does
/// not check these conditions.
///
/// # Examples
///
/// ```
/// let (product, residual) =
///     scioxide_core::backend::two_product(1.0_f64 + f64::EPSILON, 1.0 - f64::EPSILON);
/// assert_eq!(product, 1.0);
/// assert_eq!(residual, -f64::EPSILON * f64::EPSILON);
/// ```
#[inline(always)]
pub fn two_product<S: RealScalar>(a: S, b: S) -> (S, S) {
    let product = a * b;
    (product, product_residual(a, b, product))
}

#[inline(always)]
fn product_residual<S: RealScalar>(a: S, b: S, product: S) -> S {
    if a.abs() == S::ONE || b.abs() == S::ONE {
        return S::ZERO;
    }
    if HARDWARE_FMA {
        return a.mul_add(b, -product);
    }
    // Dekker/Veltkamp TwoProduct, Ogita–Rump–Oishi (2005), Alg. 3.3.
    // Admission keeps every split and product away from the exponent limits.
    let splitter = S::from_f64_ieee(if S::MANTISSA_DIGITS == 24 {
        4097.0
    } else {
        134_217_729.0
    });
    let split = |x| {
        let t = splitter * x;
        let high = t - (t - x);
        (high, x - high)
    };
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    al * bl - (((product - ah * bh) - al * bh) - ah * bl)
}

/// Returns `Σ aᵢ bᵢ cᵢ` over at most eight triples, correctly rounded, or `None`
/// if correct rounding cannot be proved cheaply.
///
/// Each triple product becomes four exact terms by two TwoProduct steps, and a
/// compensated sum checks the final rounding. `None` is returned for more than
/// eight triples, non-finite factors, values outside the guarded range and
/// results near a rounding midpoint; the caller must then use exact
/// accumulation. An empty input or exact cancellation gives `+0`.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::certified_short_sum_triples;
///
/// assert_eq!(certified_short_sum_triples(&[(1.5_f64, 2.5, 3.5)]), Some(13.125));
/// ```
#[inline]
pub fn certified_short_sum_triples<S: RealScalar>(terms: &[(S, S, S)]) -> Option<S> {
    if !EFT_ARITHMETIC || terms.len() > 8 {
        return None;
    }
    let (lower, upper) = if S::MANTISSA_DIGITS == 24 {
        (0.000_244_140_625, 4096.0) // 2^±12: lowest triple bit >= 2^-105.
    } else {
        (6.223_015_277_861_142e-61, 1.606_938_044_258_990_3e60) // >= 2^-756.
    };
    let mut high = S::ZERO;
    let mut low = S::ZERO;
    let mut mass = S::ZERO;
    for &(a, b, c) in terms {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return None;
        }
        if a == S::ZERO || b == S::ZERO || c == S::ZERO {
            continue;
        }
        if ![a, b, c].into_iter().all(|x| {
            let x = x.into_f64().abs();
            x >= lower && x <= upper
        }) {
            return None;
        }
        let p = a * b;
        let ep = product_residual(a, b, p);
        let r = p * c;
        let er = product_residual(p, c, r);
        let q = ep * c;
        let eq = if ep == S::ZERO {
            S::ZERO
        } else {
            product_residual(ep, c, q)
        };
        // Exact identity: a*b*c = r + er + q + eq. The range guard
        // includes the *lowest* product bit, not just rounded high parts.
        for term in [r, er, q, eq] {
            if term == S::ZERO {
                continue;
            }
            let (sum, error) = two_sum(high, term);
            high = sum;
            low += error;
            mass += error.abs();
        }
    }
    if mass == S::ZERO {
        return Some(if high == S::ZERO { S::ZERO } else { high });
    }
    let (rounded, tail) = two_sum(high, low);
    // At most 32 low-part additions, with gamma_32 error. 128*EPSILON
    // times the rounded absolute mass also covers that mass's reduction error.
    // The minimum admitted quantum keeps every nonzero bound normal.
    let error = mass * (S::EPSILON * S::from_f64_ieee(128.0));
    if rounded.is_normal() {
        let magnitude = rounded.abs();
        let margin = (magnitude - magnitude.next_after(S::ZERO)) * S::HALF;
        if tail.abs() + error < margin {
            return Some(rounded);
        }
    }
    None
}

/// Returns `a b + c d` correctly rounded, like [`certified_sum_two_products`],
/// using a fused multiply-add that you supply.
///
/// `fused(x, y, z)` must compute `x y + z` with one rounding, as IEEE 754 fused
/// multiply-add does, for finite inputs. The range and rounding checks are the
/// same as in [`certified_sum_two_products`]. `None` means the caller must use
/// exact accumulation.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::certified_sum_two_products_with;
/// let value = certified_sum_two_products_with(1.5_f64, 2.5, 3.5, 4.5, f64::mul_add);
/// assert_eq!(value, Some(19.5));
/// ```
#[inline]
pub fn certified_sum_two_products_with<S: RealScalar>(
    a: S,
    b: S,
    c: S,
    d: S,
    fused: impl Fn(S, S, S) -> S,
) -> Option<S> {
    contraction(a, b, c, d, |a, b, p| fused(a, b, -p))
}

#[inline(always)]
fn contraction<S: RealScalar>(
    a: S,
    b: S,
    c: S,
    d: S,
    residual: impl Fn(S, S, S) -> S,
) -> Option<S> {
    if !EFT_ARITHMETIC {
        return None;
    }
    // Binary32: product low bits >= 2^-94; binary64: >= 2^-504.
    // The upper bounds also leave room for splitting and every TwoSum
    // subtraction. Powers of two and precision are exact in both formats.
    let (lower, upper) = if S::MANTISSA_DIGITS == 24 {
        (5.960_464_477_539_063e-8, 16_777_216.0)
    } else {
        (6.223_015_277_861_142e-61, 1.606_938_044_258_990_3e60)
    };
    if ![a, b, c, d].into_iter().all(|x| {
        let x = x.into_f64().abs();
        x >= lower && x <= upper
    }) {
        return None;
    }
    let p = a * b;
    let q = c * d;
    let (s, e) = two_sum(p, q);
    let ep = residual(a, b, p);
    let eq = residual(c, d, q);
    let correction = (e + ep) + eq;
    let candidate = s + correction;
    if candidate.is_normal() && s.abs() >= correction.abs() {
        // FastTwoSum is exact under this magnitude guard. The two low-part
        // additions have error at most gamma_2 times their exact absolute mass.
        // 4*EPSILON times its rounded mass exceeds that bound, including the
        // mass reduction's own rounding. Admission keeps nonzero bounds normal.
        let tail = correction - (candidate - s);
        let error = ((e.abs() + ep.abs()) + eq.abs()) * (S::EPSILON * (S::TWO * S::TWO));
        let magnitude = candidate.abs();
        let margin = (magnitude - magnitude.next_after(S::ZERO)) * S::HALF;
        // margin is an exact power of two and no greater than either half-ULP.
        // A rounded nonnegative bound strictly below it proves the exact bound
        // is below it as well. Midpoints proceed to the full expansion/fallback.
        if tail.abs() + error < margin {
            return Some(candidate);
        }
    }
    let (u, v) = two_sum(ep, eq);
    let (w, z) = two_sum(e, u);
    let (rounded, tail) = two_sum(s, w);
    // The exact contraction is rounded + tail + z + v. In particular, this
    // equality has lost no low bits through product residual underflow.
    if tail == S::ZERO && z == -v {
        return Some(if rounded == S::ZERO { S::ZERO } else { rounded });
    }
    let bound = (z.abs() + v.abs()).next_after(S::MAX);
    super::certified_rounding_cell(rounded, tail, bound)
}
