//! Exact fixed-limb multiplication by the shared truncated 2/pi constant. No rounded large
//! integer subtraction is used to recover a small phase.
use super::{
    approx::{A, polynomial_tail},
    constants::{FACTORIAL, PI},
    kernel::{B, SMALL, enclosure, point, range, tiny_odd},
};
use scioxide_core::{RealScalar, SciOxideError, backend::TWO_OVER_PI_LIMBS};

// 66 table limbs plus three significand/carry limbs; base 2^24.
const LIMBS: usize = 69;
const MASK: u128 = (1 << 24) - 1;
fn bit(a: &[u32; LIMBS], n: i32) -> u32 {
    if n < 0 || n >= (24 * LIMBS) as i32 {
        0
    } else {
        (a[n as usize / 24] >> (n as usize % 24)) & 1
    }
}
// Retain 106 leading bits of the exact fixed-limb fraction. Its remaining
// tail is at most 2^(bottom-shift); scaling underflow costs at most two eta.
fn fraction(a: &[u32; LIMBS], shift: i32) -> A {
    let top = (0..shift.min((24 * LIMBS) as i32))
        .rev()
        .find(|&i| bit(a, i) != 0);
    let Some(top) = top else {
        return A::ZERO;
    };
    // Integer bit indices have no signed zero.
    let middle = (top - 52).max(0);
    let bottom = (middle - 53).max(0);
    let word = |lo, hi| {
        (lo..hi)
            .rev()
            .fold(0_u64, |n, i| (n << 1) | u64::from(bit(a, i)))
    };
    let high = RealScalar::scalbn(word(middle, top + 1) as f64, middle - shift);
    let low = RealScalar::scalbn(word(bottom, middle) as f64, bottom - shift);
    let error = RealScalar::scalbn(1.0, bottom - shift).max(f64::from_bits(1));
    A::point(high)
        .add(A::point(low))
        .with_tail(error + 2.0 * f64::from_bits(1))
}

/// q mod 4, remainder enclosure in [-pi/4,pi/4] apart from rounding at
/// boundaries. x = (q+4*k)*pi/2 + remainder for some exact integer k.
fn reduce(x: f64) -> Result<(i32, A), SciOxideError> {
    if x.abs() <= core::f64::consts::FRAC_PI_4 {
        return Ok((0, A::point(x)));
    }
    if x < 0.0 {
        let (q, r) = reduce(-x)?;
        return Ok(((-q).rem_euclid(4), r.neg()));
    }
    // This range stays at least 0.07 away from pi/2, so the split
    // constant's absolute tail is also a tiny relative remainder error.
    if x <= 1.5 {
        return Ok((1, A::point(x).sub(PI.scale(-1))));
    }
    let scioxide_core::FloatDecomposition::Finite {
        significand: m,
        exponent: e,
        ..
    } = RealScalar::decompose(x)
    else {
        return Err(scioxide_core::SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "finite scalar required",
        ));
    };
    let mut a = [0_u32; LIMBS];
    let mut carry = 0_u128;
    for (out, &digit) in a.iter_mut().zip(TWO_OVER_PI_LIMBS.iter().rev()) {
        let product = u128::from(digit as u32) * u128::from(m) + carry;
        *out = (product & MASK) as u32;
        carry = product >> 24;
    }
    for out in &mut a[66..] {
        *out = (carry & MASK) as u32;
        carry >>= 24;
    }
    let shift = 1584 - i32::from(e);
    let q = (bit(&a, shift) + 2 * bit(&a, shift + 1)) as i32;
    let round_up = bit(&a, shift - 1) != 0;
    // Remove integer bits before taking a fractional two's complement.
    for (i, digit) in a.iter_mut().enumerate() {
        let kept = (shift - 24 * i as i32).clamp(0, 24) as u32;
        *digit &= (1_u32 << kept) - 1;
    }
    let fraction = if round_up {
        // 2^shift - (A mod 2^shift), exactly, without cancellation.
        let mut carry = 1_u32;
        for (i, digit) in a.iter_mut().enumerate() {
            let kept = (shift - 24 * i as i32).clamp(0, 24) as u32;
            let mask = (1_u32 << kept) - 1;
            let total = ((!*digit) & mask) + carry;
            *digit = total & mask;
            carry = total >> kept;
        }
        fraction(&a, shift).neg()
    } else {
        fraction(&a, shift)
    };
    // 0 < x*(2/pi - C*2^-1584) < 2^(ilogb(x)+1-1584).
    // A minimum subnormal remains an upper bound if scalbn underflows.
    let exponent = scioxide_core::backend::standard::exponent(x)? - 1584;
    let tail = RealScalar::scalbn(1.0, exponent).max(f64::from_bits(1));
    let remainder = fraction.with_tail(tail).mul(PI.scale(-1));
    Ok(((q + i32::from(round_up)) % 4, remainder))
}
/// Fixed sine/cosine polynomials on |r| < 0.8. The alternating
/// degree-19 sine tail is < |r|*2^-70; the degree-20 cosine tail < 2^-74.
fn reduced(r: A, cosine: bool) -> A {
    let z = r.mul(r).neg();
    let last = if cosine { 20 } else { 19 };
    // The tail starts at z³. Its absolute mass is < 2^-12 for sine,
    // < 2^-9 for cosine. gamma_16, including coefficient rounding,
    // gives respectively < 2^-60 and < 2^-57. Their derivatives are
    // < 1e-5 and < 1e-4. Leading coefficients retain both words.
    let tail = polynomial_tail(
        z,
        (3..=last / 2)
            .rev()
            .map(|n| FACTORIAL[2 * n + usize::from(!cosine)]),
        if cosine {
            6.938893903907228e-18
        } else {
            8.673617379884035e-19
        },
        if cosine { 1e-4 } else { 1e-5 },
    );
    let inner = z.mul_add(tail, FACTORIAL[4 + usize::from(!cosine)]);
    let middle = z.mul_add(inner, FACTORIAL[2 + usize::from(!cosine)]);
    let p = z.mul_add(middle, A::ONE);
    if cosine {
        p.with_tail(5.293955920339377e-23)
    } else {
        p.mul(r).with_tail(r.mass() * 8.470329472543003e-22)
    }
}
fn approximation(q: i32, r: A, cosine: bool) -> A {
    let use_cosine = cosine ^ (q % 2 != 0);
    let result = reduced(r, use_cosine);
    let negative = if cosine { q == 1 || q == 2 } else { q >= 2 };
    if negative { result.neg() } else { result }
}
/// One trigonometric endpoint; the unused polynomial is never evaluated.
pub(super) fn value(x: f64, cosine: bool) -> Result<B, SciOxideError> {
    if x.abs() < SMALL {
        if cosine {
            return if x == 0.0 {
                point(1.0)
            } else {
                range(RealScalar::next_after(1.0, 0.0), 1.0)
            };
        }
        return tiny_odd(x, x < 0.0);
    }
    let (q, r) = reduce(x)?;
    let result = enclosure(approximation(q, r, cosine))?;
    Ok(result.intersection(&range(-1.0, 1.0)?))
}
/// Tangent keeps both polynomial low words through the quotient.
pub(super) fn tangent(x: f64) -> Result<B, SciOxideError> {
    if x.abs() < SMALL {
        return tiny_odd(x, x > 0.0);
    }
    let (q, r) = reduce(x)?;
    enclosure(approximation(q, r, false).div(approximation(q, r, true)))
}

/// Determine which pi/2 lattice points may occur in a finite interval less
/// than a full turn wide. Enumerating a fixed nine positions covers endpoint
/// quadrant uncertainty and every interior extremum/pole. The booleans are
/// (possibly present, certainly present) for each residue class.
pub(super) fn crossings(a: f64, b: f64) -> Result<[(bool, bool); 4], SciOxideError> {
    let (q, r) = reduce(a)?;
    let width = A::point(b).sub(A::point(a));
    let half_pi = PI.scale(-1);
    let mut result = [(false, false); 4];
    for j in -1..=7 {
        let distance = half_pi.mul(A::point(f64::from(j))).sub(r);
        let (dlo, dhi) = distance.endpoints();
        let (wlo, whi) = width.endpoints();
        if dhi >= 0.0 && dlo <= whi {
            let entry = &mut result[(q + j).rem_euclid(4) as usize];
            entry.0 = true;
            entry.1 |= dlo >= 0.0 && dhi <= wlo;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_conversion_encloses_rounding_up_to_minimum_normal() {
        let mut limbs = [0_u32; LIMBS];
        limbs[0] = (1 << 24) - 1;
        limbs[1] = (1 << 24) - 1;
        limbs[2] = 31;
        // (2^53-1)*2^-1075 is the exact midpoint between the largest
        // subnormal and the smallest normal. Nearest-even rounds upward.

        let (lo, hi) = fraction(&limbs, 1075).endpoints();
        assert!(lo <= f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1));
        assert!(hi >= f64::MIN_POSITIVE);
    }
}
