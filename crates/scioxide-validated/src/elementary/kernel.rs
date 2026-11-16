//! Certified point approximations. Reduction, fixed-polynomial remainders and
//! reconstruction keep their absolute errors until final outward rounding.
use super::{
    approx::{A, polynomial_tail},
    constants::{EXP_TABLE, FACTORIAL, LN2, LN10, ODD_RECIPROCAL, PI},
};
use crate::{DecoratedInterval as D, Decoration};
use core::cmp::Ordering;
use scioxide_core::{RangeAccumulator, RealScalar, SciOxideError, backend::standard::exponent};

pub(super) type B = D<f64>;
#[inline]
pub(super) fn range(lo: f64, hi: f64) -> Result<B, SciOxideError> {
    B::result(lo, hi, Decoration::Com)
}
pub(super) fn point(x: f64) -> Result<B, SciOxideError> {
    range(x, x)
}
#[inline]
pub(super) fn bounds(x: B) -> Result<(f64, f64), SciOxideError> {
    x.bounds()
        .ok_or_else(|| crate::shell::breakdown("empty internal enclosure"))
}
#[inline]
pub(super) fn enclosure(x: A) -> Result<B, SciOxideError> {
    let (lo, hi) = x.endpoints();
    range(lo, hi)
}
pub(super) fn pi() -> Result<B, SciOxideError> {
    enclosure(PI)
}
const EXP_FINITE: f64 = f64::from_bits(0x40862e42fefa39ef);
const EXP10_FINITE: f64 = f64::from_bits(0x40734413509f79fe);
const HYPERBOLIC_FINITE: f64 = f64::from_bits(0x408633ce8fb9f87d);
pub(super) const SMALL: f64 = 7.450580596923828e-9; // 2^-27
const TINY: f64 = 5.551115123125783e-17; // 2^-54
const ETA: f64 = f64::from_bits(1);

/// Scale endpoints directly. The reverse scaling compares the exact input
/// with the rounded subnormal image, without a second uncertain reduction.
fn scale_endpoint(x: f64, exponent: i32, up: bool) -> f64 {
    let y = RealScalar::scalbn(x, exponent);
    if y.is_infinite() {
        return if y > 0.0 {
            if up { y } else { f64::MAX }
        } else if up {
            -f64::MAX
        } else {
            y
        };
    }
    if exponent < 0 && y.abs() <= f64::MIN_POSITIVE {
        let back = RealScalar::scalbn(y, -exponent);
        if up && back < x {
            return RealScalar::next_after(y, f64::INFINITY);
        }
        if !up && back > x {
            return RealScalar::next_after(y, f64::NEG_INFINITY);
        }
    }
    y
}
pub(super) fn scale(x: B, exponent: i32) -> Result<B, SciOxideError> {
    let (lo, hi) = bounds(x)?;
    range(
        scale_endpoint(lo, exponent, false),
        scale_endpoint(hi, exponent, true),
    )
}
fn scaled(x: A, exponent: i32) -> Result<B, SciOxideError> {
    scale(enclosure(x)?, exponent)
}
fn cap_finite(x: B) -> Result<B, SciOxideError> {
    let (lo, hi) = bounds(x)?;
    range(lo, hi.min(f64::MAX))
}
fn map_monotone(x: B, f: fn(f64) -> Result<B, SciOxideError>) -> Result<B, SciOxideError> {
    let (a, b) = bounds(x)?;
    range(bounds(f(a)?)?.0, bounds(f(b)?)?.1)
}
pub(super) fn exp_interval(x: B) -> Result<B, SciOxideError> {
    map_monotone(x, exp)
}

/// x = (32*k+j)*ln(2)/32 + r, |r| < 0.011. Degree nine has
/// |remainder| <= exp(0.011)*0.011^10/10! < 2^-86. The looser
/// 2^-84 bound also covers the tiny uncertainty in the selected remainder.
fn exponential(x: A) -> (A, i32) {
    let q = RealScalar::round(x.hi * (32.0 * core::f64::consts::LOG2_E)) as i32;
    let r = x.sub(LN2.mul(A::point(f64::from(q) / 32.0)));
    // The scalar tail starts at r². Its mass is < 0.51 and gamma_16
    // times that mass, including coefficient rounding, is < 2^-48.
    // The derivative is < 1. Leading 1+r retains both words.
    let tail = polynomial_tail(
        r,
        FACTORIAL[2..=9].iter().rev().copied(),
        3.552713678800501e-15,
        1.0,
    );
    let p = r.mul_add(r.mul(tail), A::ONE.add(r));
    (
        p.with_tail(5.169878828456423e-26)
            .mul(EXP_TABLE[q.rem_euclid(32) as usize]),
        q.div_euclid(32),
    )
}
pub(super) fn exp(x: f64) -> Result<B, SciOxideError> {
    if x == f64::NEG_INFINITY {
        return point(0.0);
    }
    if x > EXP_FINITE {
        return range(f64::MAX, f64::INFINITY);
    }
    if x < -746.0 {
        return range(0.0, ETA);
    }
    if x == 0.0 {
        return point(1.0);
    }
    if x.abs() < TINY {
        return if x > 0.0 {
            range(1.0, RealScalar::next_after(1.0, f64::INFINITY))
        } else {
            range(RealScalar::next_after(1.0, 0.0), 1.0)
        };
    }
    let (a, k) = exponential(A::point(x));
    cap_finite(scaled(a, k)?)
}
fn expm1_approx(x: A) -> A {
    if x.hi.abs() <= 0.5 {
        let mut p = FACTORIAL[20];
        for &c in FACTORIAL[1..20].iter().rev() {
            p = p.mul(x).add(c);
        }
        // Tail divided by |x| is at most 0.5^20/21!/(1-0.5/22) < 2^-84.
        p.mul(x).with_tail(x.mass() * 5.169878828456423e-26)
    } else {
        let (a, k) = exponential(x);
        a.scale(k).sub(A::ONE)
    }
}
pub(super) fn expm1(x: f64) -> Result<B, SciOxideError> {
    if x == 0.0 {
        return point(x);
    }
    if x.abs() < TINY {
        return range(x, RealScalar::next_after(x, f64::INFINITY));
    }
    if x < -40.0 {
        return range(-1.0, RealScalar::next_after(-1.0, 0.0));
    }
    if x > EXP_FINITE {
        return range(f64::MAX, f64::INFINITY);
    }
    if x > 40.0 {
        let (a, k) = exponential(A::point(x));
        // One is represented in the normalized reconstruction, not discarded.
        return cap_finite(scaled(a.sub(A::point(RealScalar::scalbn(1.0, -k))), k)?);
    }
    enclosure(expm1_approx(A::point(x)))
}
pub(super) fn exp2(x: f64) -> Result<B, SciOxideError> {
    if x == f64::NEG_INFINITY {
        return point(0.0);
    }
    if x >= 1024.0 {
        return range(f64::MAX, f64::INFINITY);
    }
    if x < -1075.0 {
        return range(0.0, ETA);
    }
    let k = RealScalar::round(x) as i32;
    // TwoSum retains the subtraction residual, including inexact 1+x.
    let r = A::point(x).sub(A::point(f64::from(k)));
    if r.hi == 0.0 && r.lo == 0.0 {
        return scaled(A::ONE, k);
    }
    let (a, j) = exponential(r.mul(LN2));
    scaled(a, k + j)
}
pub(super) fn exp10(x: f64) -> Result<B, SciOxideError> {
    if x == f64::NEG_INFINITY {
        return point(0.0);
    }
    if x > EXP10_FINITE {
        return range(f64::MAX, f64::INFINITY);
    }
    if x < -324.0 {
        return range(0.0, ETA);
    }
    if x == 0.0 {
        return point(1.0);
    }
    let (a, k) = exponential(A::point(x).mul(LN10));
    cap_finite(scaled(a, k)?)
}

/// 2*atanh(t), |t| <= 0.172. The omitted tail is bounded by
/// 2*|t|^49/(49*(1-t²)) < |t|*2^-120.
fn log_reduced(t: A) -> A {
    let t2 = t.mul(t);
    let mut p = ODD_RECIPROCAL[23];
    for &c in ODD_RECIPROCAL[..23].iter().rev() {
        p = p.mul(t2).add(c);
    }
    p.mul(t).scale(1).with_tail(t.mass() * 7.52316384526264e-37)
}
fn logarithm(x: A) -> Result<A, SciOxideError> {
    let mut k = exponent(x.hi)? - 1;
    let mut m = x.scale(-k);
    if m.hi > core::f64::consts::SQRT_2 {
        k += 1;
        m = m.scale(-1);
    }
    let t = m.sub(A::ONE).div(m.add(A::ONE));
    Ok(log_reduced(t).add(LN2.mul(A::point(f64::from(k)))))
}
pub(super) fn ln(x: f64) -> Result<B, SciOxideError> {
    if x == 0.0 {
        return range(f64::NEG_INFINITY, -f64::MAX);
    }
    if x == f64::INFINITY {
        return range(f64::MAX, f64::INFINITY);
    }
    if x == 1.0 {
        return point(0.0);
    }
    enclosure(logarithm(A::point(x))?)
}
fn log1p_approx(x: A) -> Result<A, SciOxideError> {
    if x.hi.abs() <= 0.25 {
        Ok(log_reduced(x.div(A::point(2.0).add(x))))
    } else {
        logarithm(A::ONE.add(x))
    }
}
pub(super) fn log1p(x: f64) -> Result<B, SciOxideError> {
    if x == f64::INFINITY {
        return range(f64::MAX, f64::INFINITY);
    }
    if x == -1.0 {
        return range(f64::NEG_INFINITY, -f64::MAX);
    }
    if x == 0.0 {
        return point(x);
    }
    if x.abs() < TINY {
        return range(RealScalar::next_after(x, f64::NEG_INFINITY), x);
    }
    if x > 1.0e100 {
        return enclosure(logarithm(A::point(x))?.with_tail(2.0 / x));
    }
    enclosure(log1p_approx(A::point(x))?)
}
pub(super) fn log2(x: f64) -> Result<B, SciOxideError> {
    if x == 0.0 || !x.is_finite() {
        return ln(x);
    }
    if x == 1.0 {
        return point(0.0);
    }
    enclosure(logarithm(A::point(x))?.div(LN2))
}
pub(super) fn log10(x: f64) -> Result<B, SciOxideError> {
    if x == 0.0 || !x.is_finite() {
        return ln(x);
    }
    if x == 1.0 {
        return point(0.0);
    }
    enclosure(logarithm(A::point(x))?.div(LN10))
}

/// Alternating arctangent polynomial on |t| <= sqrt(2)-1. Its
/// degree-95 remainder is <= |t|^97/97 < |t|*2^-120.
fn atan_reduced(t: A) -> A {
    let z = t.mul(t).neg();
    let mut p = ODD_RECIPROCAL[47];
    for &c in ODD_RECIPROCAL[..47].iter().rev() {
        p = p.mul(z).add(c);
    }
    p.mul(t).with_tail(t.mass() * 7.52316384526264e-37)
}
fn arctangent(x: A) -> A {
    if x.hi < 0.0 {
        return arctangent(x.neg()).neg();
    }
    if x.hi > 1.0 {
        return PI.scale(-1).sub(arctangent(A::ONE.div(x)));
    }
    if x.hi > 0.41421356237309503 {
        PI.scale(-2)
            .add(atan_reduced(x.sub(A::ONE).div(x.add(A::ONE))))
    } else {
        atan_reduced(x)
    }
}
pub(super) fn tiny_odd(x: f64, above: bool) -> Result<B, SciOxideError> {
    if x == 0.0 {
        return point(x);
    }
    let neighbor = RealScalar::next_after(
        x,
        if above {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        },
    );
    range(x.min(neighbor), x.max(neighbor))
}
pub(super) fn atan(x: f64) -> Result<B, SciOxideError> {
    if x.abs() < TINY {
        return tiny_odd(x, x < 0.0);
    }
    if x.abs() > 1.152921504606847e18 {
        let a = PI.scale(-1).with_tail(2.0 / x.abs());
        return enclosure(if x < 0.0 { a.neg() } else { a });
    }
    enclosure(arctangent(A::point(x)))
}
pub(super) fn asin(x: f64) -> Result<B, SciOxideError> {
    if x.abs() < TINY {
        return tiny_odd(x, x > 0.0);
    }
    if x.abs() == 1.0 {
        return enclosure(if x < 0.0 {
            PI.scale(-1).neg()
        } else {
            PI.scale(-1)
        });
    }
    let p = A::point(x.abs());
    let d = A::ONE.sub(p).mul(A::ONE.add(p)).sqrt();
    let a = arctangent(p.div(d));
    enclosure(if x < 0.0 { a.neg() } else { a })
}
pub(super) fn acos(x: f64) -> Result<B, SciOxideError> {
    if x == 1.0 {
        return point(0.0);
    }
    let p = A::point(x.abs());
    let a = if x.abs() == 1.0 {
        A::ZERO
    } else {
        arctangent(A::ONE.sub(p).div(A::ONE.add(p)).sqrt()).scale(1)
    };
    enclosure(if x < 0.0 { PI.sub(a) } else { a })
}
pub(super) fn sinh(x: f64) -> Result<B, SciOxideError> {
    if x < 0.0 {
        return sinh(-x)?.neg_admitted();
    }
    if x < TINY {
        return tiny_odd(x, true);
    }
    if x > HYPERBOLIC_FINITE {
        return range(f64::MAX, f64::INFINITY);
    }
    if x > 40.0 {
        let (a, k) = exponential(A::point(x));
        return cap_finite(scaled(
            a.with_tail(a.mass() * 7.703719777548943e-34),
            k - 1,
        )?);
    }
    let u = expm1_approx(A::point(x));
    enclosure(u.add(u.div(A::ONE.add(u))).scale(-1))
}
pub(super) fn cosh(x: f64) -> Result<B, SciOxideError> {
    let x = x.abs();
    if x == 0.0 {
        return point(1.0);
    }
    if x < SMALL {
        return range(1.0, RealScalar::next_after(1.0, f64::INFINITY));
    }
    if x > HYPERBOLIC_FINITE {
        return range(f64::MAX, f64::INFINITY);
    }
    let (a, k) = exponential(A::point(x));
    if x > 40.0 {
        return cap_finite(scaled(
            a.with_tail(a.mass() * 7.703719777548943e-34),
            k - 1,
        )?);
    }
    let e = a.scale(k);
    enclosure(e.add(A::ONE.div(e)).scale(-1))
}
pub(super) fn tanh(x: f64) -> Result<B, SciOxideError> {
    if x < 0.0 {
        return tanh(-x)?.neg_admitted();
    }
    if x < TINY {
        return tiny_odd(x, false);
    }
    if x > 20.0 {
        return range(RealScalar::next_after(1.0, 0.0), 1.0);
    }
    let u = expm1_approx(A::point(2.0 * x));
    enclosure(u.div(u.add(A::point(2.0))))
}
pub(super) fn asinh(x: f64) -> Result<B, SciOxideError> {
    if x < 0.0 {
        return asinh(-x)?.neg_admitted();
    }
    if x == f64::INFINITY {
        return range(f64::MAX, f64::INFINITY);
    }
    if x < TINY {
        return tiny_odd(x, false);
    }
    if x > 1073741824.0 {
        return enclosure(
            logarithm(A::point(x))?
                .add(LN2)
                .with_tail(8.673617379884035e-19),
        );
    }
    let p = A::point(x);
    let square = p.mul(p);
    enclosure(log1p_approx(
        p.add(square.div(A::ONE.add(A::ONE.add(square).sqrt()))),
    )?)
}
pub(super) fn acosh(x: f64) -> Result<B, SciOxideError> {
    if x == f64::INFINITY {
        return range(f64::MAX, f64::INFINITY);
    }
    if x == 1.0 {
        return point(0.0);
    }
    if x > 1073741824.0 {
        return enclosure(
            logarithm(A::point(x))?
                .add(LN2)
                .with_tail(8.673617379884035e-19),
        );
    }
    let p = A::point(x);
    let u = p.sub(A::ONE);
    enclosure(log1p_approx(u.add(u.mul(p.add(A::ONE)).sqrt()))?)
}
pub(super) fn atanh(x: f64) -> Result<B, SciOxideError> {
    if x < 0.0 {
        return atanh(-x)?.neg_admitted();
    }
    if x == 1.0 {
        return range(f64::MAX, f64::INFINITY);
    }
    if x < TINY {
        return tiny_odd(x, true);
    }
    enclosure(
        log1p_approx(A::point(x))?
            .sub(log1p_approx(A::point(-x))?)
            .scale(-1),
    )
}
pub(super) fn cbrt(x: f64) -> Result<B, SciOxideError> {
    if x < 0.0 {
        return cbrt(-x)?.neg_admitted();
    }
    if x == 0.0 {
        return point(x);
    }
    if x == f64::INFINITY {
        return range(f64::MAX, f64::INFINITY);
    }
    let mut q = RealScalar::cbrt(x);
    // The ordinary candidate is untrusted. Exact cubic residuals certify the
    // bracket; stepping also handles any permissible point-kernel error.
    let residual = |q| -> Result<Ordering, SciOxideError> {
        let mut r = RangeAccumulator::new();
        r.add(x)?;
        r.subtract_triple_product(q, q, q)?;
        Ok(r.sign())
    };
    let sign = residual(q)?;
    if sign == Ordering::Equal {
        return point(q);
    }
    let up = sign == Ordering::Greater;
    // A four-ULP scalar envelope spans at most eight neighbors across a
    // binade boundary. A doubled allowance detects candidate breakdown.
    for _ in 0..16 {
        let next = RealScalar::next_after(q, if up { f64::INFINITY } else { 0.0 });
        let side = residual(next)?;
        if side == Ordering::Equal {
            return point(next);
        }
        if side != sign {
            return range(q.min(next), q.max(next));
        }
        q = next;
    }
    Err(SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "cube-root candidate could not be bracketed within its qualified neighborhood",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_kernel_enclosure_is_a_breakdown() {
        assert!(matches!(
            bounds(B::empty()).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Breakdown,
                ..
            }
        ));
    }

    #[test]
    fn tiny_odd_uses_canonical_zero_bounds_and_subnormal_neighbors() {
        for x in [0.0, -0.0] {
            for above in [false, true] {
                let (lo, hi) = bounds(tiny_odd(x, above).unwrap()).unwrap();
                assert_eq!(lo.to_bits(), (-0.0_f64).to_bits());
                assert_eq!(hi.to_bits(), 0.0_f64.to_bits());
            }
        }
        for x in [f64::from_bits(1), -f64::from_bits(1)] {
            for above in [false, true] {
                let (lo, hi) = bounds(tiny_odd(x, above).unwrap()).unwrap();
                let neighbor = RealScalar::next_after(
                    x,
                    if above {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    },
                );
                let expected_lo = x.min(neighbor);
                let expected_hi = x.max(neighbor);
                assert_eq!(
                    lo.to_bits(),
                    if expected_lo == 0.0 {
                        (-0.0_f64).to_bits()
                    } else {
                        expected_lo.to_bits()
                    }
                );
                assert_eq!(
                    hi.to_bits(),
                    if expected_hi == 0.0 {
                        0.0_f64.to_bits()
                    } else {
                        expected_hi.to_bits()
                    }
                );
            }
        }
        for x in [
            SMALL.next_after(0.0),
            SMALL,
            SMALL.next_after(f64::INFINITY),
        ] {
            for value in [x, -x] {
                let sine = super::super::phase::value(value, false).unwrap();
                let tangent = super::super::phase::tangent(value).unwrap();
                assert!(sine.contains(value.sin()));
                assert!(tangent.contains(value.tan()));
            }
        }
    }
}
