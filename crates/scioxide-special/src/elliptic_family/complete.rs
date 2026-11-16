// Complete K/E on `m < 1`, using Cephes polynomials in `x = 1 - m` and the
// imaginary-modulus transformation on `m < 0`.
//
// Coefficients from Stephen L. Moshier's ellpk/ellpe, as distributed by SciPy
// xsf (ellpk.h / ellpe.h). See THIRD_PARTY_NOTICES.md for attribution.
// Fused Horner evaluation keeps the f64 error within two ulps on the tested
// range; f32 callers widen before forming the complement.

#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "required when elementary methods use the no_std scalar trait"
)]
use scioxide_core::RealScalar as _;

use crate::support::polynomial::{PolynomialPolicy, polynomial};

const K_P: [f64; 11] = [
    0.00013798286460627325,
    0.002280257240058756,
    0.007974040132204152,
    0.00985821379021226,
    0.006874896874499499,
    0.006189010336376876,
    0.008790782739527438,
    0.014938044891680526,
    0.0308851465246712,
    0.09657359028116902,
    1.3862943611198906,
];

const K_Q: [f64; 11] = [
    2.940789550485985e-05,
    0.0009141847238659173,
    0.005940583037531678,
    0.01548505166497624,
    0.023908960271592488,
    0.030120471522760404,
    0.037377431417382326,
    0.04882803475709983,
    0.07031249969639575,
    0.12499999999987083,
    0.5,
];

const E_P: [f64; 11] = [
    0.0001535525773010133,
    0.0025088849216360204,
    0.008687868165658896,
    0.010735094905607619,
    0.007773954925167871,
    0.007583952894135147,
    0.011568843681057412,
    0.021831799601555724,
    0.05680519456178606,
    0.44314718056099084,
    1.0,
];

const E_Q: [f64; 10] = [
    3.2795489857648585e-05,
    0.0010096279267935672,
    0.006506094899769275,
    0.016886216399331133,
    0.026176974245449364,
    0.03348339048882249,
    0.04271809265189315,
    0.0585936634471101,
    0.09374999971976443,
    0.2499999999998883,
];

#[inline]
fn k_complement(x: f64, neg_ln_x: f64) -> f64 {
    scioxide_core::fast_mul_add(
        neg_ln_x,
        polynomial(x, &K_Q, PolynomialPolicy::FusedDescending),
        polynomial(x, &K_P, PolynomialPolicy::FusedDescending),
    )
}

#[inline]
fn e_complement(x: f64, neg_ln_x: f64) -> f64 {
    scioxide_core::fast_mul_add(
        neg_ln_x,
        x * polynomial(x, &E_Q, PolynomialPolicy::FusedDescending),
        polynomial(x, &E_P, PolynomialPolicy::FusedDescending),
    )
}

#[inline]
fn negative_modulus_parts(m: f64) -> (f64, f64, f64) {
    const SCALE_DOWN: f64 = f64::from_bits(0x1ff0_0000_0000_0000); // 2^-512
    const SCALE_UP: f64 = f64::from_bits(0x4ff0_0000_0000_0000); // 2^256
    const THRESHOLD: f64 = -f64::from_bits(0x5ff0_0000_0000_0000); // -2^512
    if m < THRESHOLD {
        let c_scaled = (-m) * SCALE_DOWN;
        let root_c = c_scaled.sqrt() * SCALE_UP;
        let x = SCALE_DOWN / c_scaled;
        let neg_ln_x =
            scioxide_core::fast_mul_add(512.0_f64, core::f64::consts::LN_2, c_scaled.ln());
        (x, neg_ln_x, root_c)
    } else {
        let c = 1.0 - m;
        (1.0 / c, (-m).ln_1p(), c.sqrt())
    }
}

/// First kind; caller has validated `m < 1`.
#[inline]
pub(super) fn k(m: f64) -> f64 {
    if m == 0.0 {
        return core::f64::consts::FRAC_PI_2;
    }
    if m < 0.0 {
        let (x, neg_ln_x, root_c) = negative_modulus_parts(m);
        return k_complement(x, neg_ln_x) / root_c;
    }
    let x = 1.0 - m;
    k_complement(x, -x.ln())
}

/// Second kind; caller has validated `m < 1`.
#[inline]
pub(super) fn e(m: f64) -> f64 {
    if m == 0.0 {
        return core::f64::consts::FRAC_PI_2;
    }
    if m < 0.0 {
        let (x, neg_ln_x, root_c) = negative_modulus_parts(m);
        return e_complement(x, neg_ln_x) * root_c;
    }
    let x = 1.0 - m;
    e_complement(x, -x.ln())
}
