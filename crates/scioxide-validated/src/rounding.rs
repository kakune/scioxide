//! Error-free residual signs determine directed endpoints in a checked environment.
//! Extreme products use at most 106 significand bits, never a wide accumulator.

use core::cmp::Ordering;
use scioxide_core::backend::{two_product, two_sum};
use scioxide_core::{FloatDecomposition, HARDWARE_FMA, RealScalar};

/// Compare two finite exact products, including values outside scalar range.
/// Equal leading exponents permit left-normalizing both integer significands
/// independently: no alignment can discard low bits or shift by 128.
#[inline]
pub(crate) fn compare_products<S: RealScalar>(a: S, b: S, c: S, d: S) -> Ordering {
    fn parts<S: RealScalar>(x: S) -> (u64, i32, bool) {
        let FloatDecomposition::Finite {
            significand,
            exponent,
            negative,
        } = x.decompose()
        else {
            unreachable!("product comparison requires finite operands")
        };
        (significand, i32::from(exponent), negative)
    }
    let product = |a, b| {
        let (a, ae, an) = parts(a);
        let (b, be, bn) = parts(b);
        let magnitude = u128::from(a) * u128::from(b);
        (magnitude, ae + be, magnitude != 0 && an != bn)
    };
    let (left, le, ln) = product(a, b);
    let (right, re, rn) = product(c, d);
    if ln != rn {
        return if ln {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let magnitude = if left == 0 || right == 0 {
        left.cmp(&right)
    } else {
        let lz = left.leading_zeros();
        let rz = right.leading_zeros();
        (le - lz as i32)
            .cmp(&(re - rz as i32))
            .then_with(|| (left << lz).cmp(&(right << rz)))
    };
    if ln { magnitude.reverse() } else { magnitude }
}

#[inline]
fn sign<S: RealScalar>(x: S) -> Ordering {
    if x > S::ZERO {
        Ordering::Greater
    } else if x < S::ZERO {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

/// FMA rounds only once, so a nonzero residual always has the exact sign.
/// At scale >= MIN_POSITIVE * 2^(p+2), even the least bit of a product
/// of two p-bit significands is representable. The two-bit margin covers
/// binade crossings and the quotient candidate's multiplication by its divisor.
#[inline]
fn fused_sign<S: RealScalar>(residual: S, scale: S) -> Option<Ordering> {
    if residual != S::ZERO || scale.abs() >= S::MIN_POSITIVE.scalbn(S::MANTISSA_DIGITS as i32 + 2) {
        Some(sign(residual))
    } else {
        None
    }
}

/// Accept split products only away from underflow and overflow. Rounded high parts
/// can overflow their product even when the unsplit product is finite.
#[inline]
fn split_admitted<S: RealScalar>(a: S, b: S, product: S) -> bool {
    let p = S::MANTISSA_DIGITS as i32;
    let ceiling = S::MAX.scalbn(-p);
    product.is_finite()
        && product.abs() <= S::MAX.scalbn(-1)
        && a.abs() <= ceiling
        && b.abs() <= ceiling
        && product.abs() > S::MIN_POSITIVE.scalbn(2 * p + 1)
}

#[inline]
fn product_sign<S: RealScalar>(a: S, b: S, candidate: S, fused: bool) -> Ordering {
    if fused {
        if let Some(sign) = fused_sign(a.mul_add(b, -candidate), candidate) {
            return sign;
        }
    } else if split_admitted(a, b, candidate) {
        return sign(two_product(a, b).1);
    }
    compare_products(a, b, candidate, S::ONE)
}

#[inline]
fn difference_product_sign<S: RealScalar>(a: S, b: S, c: S, fused: bool) -> Ordering {
    if fused {
        if let Some(sign) = fused_sign((-b).mul_add(c, a), a) {
            return sign;
        }
    } else {
        let product = b * c;
        if b.is_normal() && split_admitted(b, c, product) {
            let (_, residual) = two_product(b, c);
            // b*c approximates a within a rounding error. Both are in the
            // same sign frame and within a factor two: Sterbenz subtraction
            // is exact. Compare rather than round the final tiny difference.
            let difference = a - product;
            return if difference < residual {
                Ordering::Less
            } else if difference > residual {
                Ordering::Greater
            } else {
                Ordering::Equal
            };
        }
    }
    compare_products(a, S::ONE, b, c)
}

/// Compare an exact product with a finite candidate of the same sign.
/// Conversion to binary64 is exact for both supported scalar precisions.
/// Returns whether the candidate must move in the requested direction.
#[inline(always)]
fn product_requires_step<S: RealScalar>(a: S, b: S, candidate: S, upward: bool) -> bool {
    let [c_bits, a_bits, b_bits] = [candidate, a, b].map(|x| x.into_f64().to_bits());
    let exponent = |bits: u64| ((bits >> 52) & 0x7ff) as i32;
    let [ce, ae, be] = [c_bits, a_bits, b_bits].map(exponent);
    // Binary64's exponent bias is 1023 and its fraction has 52 bits. Aligning
    // the candidate with the exact product of the two integer significands
    // therefore requires ce + 1023 + 52 - ae - be bits. Shifts of 52 or 53
    // keep both magnitudes within 106 bits; other exponents use the general path.
    let shift = ce + 1075 - ae - be;
    if ce != 0 && ae != 0 && be != 0 && (52..=53).contains(&shift) {
        let significand = |bits: u64| (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
        let left = u128::from(significand(c_bits)) << shift;
        let right = u128::from(significand(a_bits)) * u128::from(significand(b_bits));
        let increasing_magnitude = upward ^ ((a_bits ^ b_bits) >> 63 != 0);
        (left != right) & ((left < right) == increasing_magnitude)
    } else {
        let sign = compare_products(a, b, candidate, S::ONE);
        (sign != Ordering::Equal) & ((sign == Ordering::Greater) == upward)
    }
}

/// Exact extreme corner operands. Sign dispatch resolves every case except
/// two zero-straddling factors. Only those compare two contenders per side.
/// Zero times an infinite limit is zero, matching endpoint arithmetic.
pub(crate) type ProductCorners<S> = ((S, S), (S, S));
#[inline]
pub(crate) fn product_corners<S: RealScalar>(a: S, b: S, c: S, d: S) -> ProductCorners<S> {
    let zero = S::ZERO;
    if a >= zero {
        if c >= zero {
            ((a, c), (b, d))
        } else if d <= zero {
            ((b, c), (a, d))
        } else {
            ((b, c), (b, d))
        }
    } else if b <= zero {
        if c >= zero {
            ((a, d), (b, c))
        } else if d <= zero {
            ((b, d), (a, c))
        } else {
            ((a, d), (a, c))
        }
    } else if c >= zero {
        ((a, d), (b, d))
    } else if d <= zero {
        ((b, c), (a, c))
    } else {
        let compare = |(a, b): (S, S), (c, d): (S, S)| {
            let product = |a: S, b: S| if a == zero || b == zero { zero } else { a * b };
            let left = product(a, b);
            let right = product(c, d);
            if left < right {
                Ordering::Less
            } else if left > right {
                Ordering::Greater
            } else if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() {
                compare_products(a, b, c, d)
            } else {
                Ordering::Equal
            }
        };
        let low = if compare((a, d), (b, c)).is_gt() {
            (b, c)
        } else {
            (a, d)
        };
        let high = if compare((a, c), (b, d)).is_lt() {
            (b, d)
        } else {
            (a, c)
        };
        (low, high)
    }
}

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Operation {
    Add,
    Multiply,
    Divide,
    Sqrt,
}

/// Directed rounding of `op` on finite operands.
#[inline]
pub(crate) fn directed<S: RealScalar>(a: S, b: S, op: Operation, upward: bool) -> S {
    directed_with(a, b, op, upward, HARDWARE_FMA)
}

/// The explicit strategy lets tests exercise non-FMA correction on FMA hosts.
#[inline]
fn directed_with<S: RealScalar>(a: S, b: S, op: Operation, upward: bool, fused: bool) -> S {
    let candidate = match op {
        Operation::Add => a + b,
        Operation::Multiply => a * b,
        Operation::Divide => a / b,
        Operation::Sqrt => a.sqrt(),
    };
    round_candidate(a, b, candidate, op, upward, fused)
}

/// Correct an already evaluated nearest-even candidate using the exact residual sign.
#[inline(always)]
fn round_candidate<S: RealScalar>(
    a: S,
    b: S,
    candidate: S,
    op: Operation,
    upward: bool,
    fused: bool,
) -> S {
    // A finite-input overflow lies strictly beyond MAX. Its directed endpoints
    // are MAX and infinity (or their negative counterparts), never two infinities.
    if candidate.is_infinite() {
        return if candidate > S::ZERO {
            if upward { S::INFINITY } else { S::MAX }
        } else if upward {
            -S::MAX
        } else {
            -S::INFINITY
        };
    }
    if matches!(op, Operation::Divide) && !fused {
        let step = product_requires_step(candidate, b, a, !upward ^ b.is_sign_negative());
        let adjacent = candidate.next_after(if upward { S::INFINITY } else { -S::INFINITY });
        // Select between prepared values without branching on the residual sign.
        return [candidate, adjacent][usize::from(step)];
    }
    let mut sign = match op {
        Operation::Add => {
            // Ordering by magnitude prevents overflowing subtraction
            // intermediates even when opposite signs cancel near MAX.
            let (a, b) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
            sign(two_sum(a, b).1)
        }
        Operation::Multiply => product_sign(a, b, candidate, fused),
        Operation::Divide => difference_product_sign(a, candidate, b, fused),
        Operation::Sqrt => difference_product_sign(a, candidate, candidate, fused),
    };
    if matches!(op, Operation::Divide) && b < S::ZERO {
        sign = sign.reverse();
    }
    if upward && sign == Ordering::Greater {
        candidate.next_after(S::INFINITY)
    } else if !upward && sign == Ordering::Less {
        candidate.next_after(-S::INFINITY)
    } else {
        candidate
    }
}

/// Square nonnegative magnitude bounds without general product corner dispatch.
#[inline(always)]
pub(crate) fn square_bounds<S: RealScalar>(lo: S, hi: S) -> (S, S) {
    let square = |x: S, upward: bool| {
        if x == S::ZERO || x.is_infinite() {
            x
        } else {
            let candidate = x * x;
            if !HARDWARE_FMA && candidate.is_finite() {
                let step = product_requires_step(x, x, candidate, upward);
                let adjacent =
                    candidate.next_after(if upward { S::INFINITY } else { -S::INFINITY });
                [candidate, adjacent][usize::from(step)]
            } else {
                round_candidate(x, x, candidate, Operation::Multiply, upward, HARDWARE_FMA)
            }
        }
    };
    (square(lo, false), square(hi, true))
}

/// Evaluate both quotient candidates before their independent residual corrections.
#[inline(always)]
pub(crate) fn quotient_bounds<S: RealScalar>(lo: S, lo_divisor: S, hi: S, hi_divisor: S) -> (S, S) {
    if lo.is_finite()
        && lo_divisor.is_finite()
        && hi.is_finite()
        && hi_divisor.is_finite()
        && lo_divisor != S::ZERO
        && hi_divisor != S::ZERO
    {
        let lower = lo / lo_divisor;
        let upper = hi / hi_divisor;
        (
            round_candidate(
                lo,
                lo_divisor,
                lower,
                Operation::Divide,
                false,
                HARDWARE_FMA,
            ),
            round_candidate(hi, hi_divisor, upper, Operation::Divide, true, HARDWARE_FMA),
        )
    } else {
        (
            endpoint(lo, lo_divisor, Operation::Divide, false),
            endpoint(hi, hi_divisor, Operation::Divide, true),
        )
    }
}

/// Directed endpoint of `op`.
/// Infinity here denotes a limit endpoint, not a member of a real set.
#[inline]
pub(crate) fn endpoint<S: RealScalar>(a: S, b: S, op: Operation, up: bool) -> S {
    let zero = S::ZERO;
    match op {
        Operation::Multiply if a == zero || b == zero => zero,
        Operation::Multiply if a.is_infinite() || b.is_infinite() => {
            if (a < zero) ^ (b < zero) {
                -S::INFINITY
            } else {
                S::INFINITY
            }
        }
        Operation::Divide if a == zero || b.is_infinite() => zero,
        Operation::Divide if b == zero || a.is_infinite() => {
            if a.is_sign_negative() ^ b.is_sign_negative() {
                -S::INFINITY
            } else {
                S::INFINITY
            }
        }
        Operation::Add if a.is_infinite() || b.is_infinite() => a + b,
        Operation::Sqrt if a.is_infinite() => a,
        _ => directed(a, b, op, up),
    }
}
