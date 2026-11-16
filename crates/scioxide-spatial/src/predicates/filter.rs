//! Shewchuk (1996 predicates.c, public domain), first-stage A filters.
//!
//! Preserve the reference evaluation order and unit roundoff (epsilon / 2).
//! Only zero or coordinates in [2^-100, 2^100] enter these formulas. All
//! coordinates are then multiples of 2^-152; nonzero degree-five arithmetic
//! is a multiple of 2^-760, and magnitudes stay below 2^520. Even multiplying
//! the permanent by the error coefficient stays normal. Thus the relative
//! error proofs encounter neither underflow nor overflow. Others fall back.

use super::PredicateSign;
use scioxide_core::RealScalar;

const U: f64 = f64::EPSILON / 2.0;

// The exponent envelope also proves finiteness. Normal interior coordinates
// need one exponent comparison; both signed zeros and the upper endpoint
// are admitted separately. The upper exponent otherwise exceeds the envelope.
#[inline]
pub(crate) fn in_range(x: f64) -> bool {
    in_envelope(x, 100)
}

// Fixed predicates and N-D recurrence have different proven exponent bounds.
#[inline]
pub(super) fn in_envelope(x: f64, exponent_bound: u32) -> bool {
    let magnitude = x.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exponent = (magnitude >> 52) as u32;
    exponent.wrapping_sub(1023 - exponent_bound) < 2 * exponent_bound
        || magnitude == 0
        || magnitude == u64::from(1023 + exponent_bound) << 52
}

#[inline]
pub(super) fn eligible<S: RealScalar, const D: usize, const N: usize>(
    points: [&[S; D]; N],
) -> bool {
    points
        .into_iter()
        .all(|point| point.iter().all(|&x| in_range(x.into_f64())))
}

// Value and unsigned permanent for a two-by-two xy minor.
#[inline]
fn cross<const D: usize>(a: [f64; D], b: [f64; D]) -> (f64, f64) {
    let left = a[0] * b[1];
    let right = b[0] * a[1];
    (left - right, left.abs() + right.abs())
}

#[inline]
fn lift(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1]) + p[2] * p[2]
}

// The following arithmetic kernels require the original coordinates to be
// admitted by `in_range`. Translating by the last point removes the column
// of ones; rounded differences are already covered by the reference A bounds.
#[inline]
pub(super) fn orient2d<S: RealScalar>(a: &[S; 2], b: &[S; 2], c: &[S; 2]) -> Option<PredicateSign> {
    let a = core::array::from_fn::<_, 2, _>(|i| a[i].into_f64() - c[i].into_f64());
    let b = core::array::from_fn::<_, 2, _>(|i| b[i].into_f64() - c[i].into_f64());
    let (det, permanent) = cross(a, b);
    certify(det, permanent, (3.0 + 16.0 * U) * U)
}

#[inline]
pub(super) fn orient3d<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
) -> Option<PredicateSign> {
    let a = core::array::from_fn::<_, 3, _>(|i| a[i].into_f64() - d[i].into_f64());
    let b = core::array::from_fn::<_, 3, _>(|i| b[i].into_f64() - d[i].into_f64());
    let c = core::array::from_fn::<_, 3, _>(|i| c[i].into_f64() - d[i].into_f64());
    let (bc, bcp) = cross(b, c);
    let (ca, cap) = cross(c, a);
    let (ab, abp) = cross(a, b);
    let det = (a[2] * bc + b[2] * ca) + c[2] * ab;
    let permanent = (bcp * a[2].abs() + cap * b[2].abs()) + abp * c[2].abs();
    certify(det, permanent, (7.0 + 56.0 * U) * U)
}

#[inline]
pub(super) fn incircle<S: RealScalar>(
    a: &[S; 2],
    b: &[S; 2],
    c: &[S; 2],
    d: &[S; 2],
) -> Option<PredicateSign> {
    let a = core::array::from_fn::<_, 2, _>(|i| a[i].into_f64() - d[i].into_f64());
    let b = core::array::from_fn::<_, 2, _>(|i| b[i].into_f64() - d[i].into_f64());
    let c = core::array::from_fn::<_, 2, _>(|i| c[i].into_f64() - d[i].into_f64());
    let (bc, bcp) = cross(b, c);
    let (ca, cap) = cross(c, a);
    let (ab, abp) = cross(a, b);
    let al = a[0] * a[0] + a[1] * a[1];
    let bl = b[0] * b[0] + b[1] * b[1];
    let cl = c[0] * c[0] + c[1] * c[1];
    let det = (al * bc + bl * ca) + cl * ab;
    let permanent = (bcp * al + cap * bl) + abp * cl;
    certify(det, permanent, (10.0 + 96.0 * U) * U)
}

#[inline]
pub(super) fn insphere<S: RealScalar>(
    a: &[S; 3],
    b: &[S; 3],
    c: &[S; 3],
    d: &[S; 3],
    e: &[S; 3],
) -> Option<PredicateSign> {
    let a = core::array::from_fn(|i| a[i].into_f64() - e[i].into_f64());
    let b = core::array::from_fn(|i| b[i].into_f64() - e[i].into_f64());
    let c = core::array::from_fn(|i| c[i].into_f64() - e[i].into_f64());
    let d = core::array::from_fn(|i| d[i].into_f64() - e[i].into_f64());
    let (det, permanent, bound) = sphere(a, b, c, d);
    certify(det, permanent, bound)
}

#[inline]
fn certify(det: f64, permanent: f64, bound: f64) -> Option<PredicateSign> {
    // Strict certification excludes Zero: degeneracy is always decided exactly.
    let error = bound * permanent;
    if det > error {
        Some(PredicateSign::Positive)
    } else if -det > error {
        Some(PredicateSign::Negative)
    } else {
        None
    }
}

// Cofactor expansion of the translated 4x4 insphere determinant along the
// lifted column. The six xy minors are shared among the four 3x3 cofactors
// (abc, bcd, cda, dab), exactly as in the reference insphere.
#[inline]
fn sphere(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> (f64, f64, f64) {
    let (ab, abp) = cross(a, b);
    let (bc, bcp) = cross(b, c);
    let (cd, cdp) = cross(c, d);
    let (da, dap) = cross(d, a);
    let (ac, acp) = cross(a, c);
    let (bd, bdp) = cross(b, d);
    let abc = (a[2] * bc - b[2] * ac) + c[2] * ab;
    let bcd = (b[2] * cd - c[2] * bd) + d[2] * bc;
    let cda = (c[2] * da + d[2] * ac) + a[2] * cd;
    let dab = (d[2] * ab + a[2] * bd) + b[2] * da;
    let [al, bl, cl, dl] = [a, b, c, d].map(lift);
    let det = (dl * abc - cl * dab) + (bl * cda - al * bcd);
    let [az, bz, cz, dz] = [a[2], b[2], c[2], d[2]].map(f64::abs);
    let permanent = (((cdp * bz + bdp * cz) + bcp * dz) * al
        + ((dap * cz + acp * dz) + cdp * az) * bl)
        + ((abp * dz + bdp * az) + dap * bz) * cl
        + ((bcp * az + acp * bz) + abp * cz) * dl;
    (det, permanent, (16.0 + 224.0 * U) * U)
}
