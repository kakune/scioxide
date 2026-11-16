//! Logarithm kernels: `ln` and `ln_1p`.

use super::exp::{LN2_HI, LN2_LO};
use super::{MANTISSA, exponent, select};

// fdlibm `log` polynomial coefficients.
const LG1: f64 = f64::from_bits(0x3fe5_5555_5555_5593);
const LG2: f64 = f64::from_bits(0x3fd9_9999_9997_fa04);
const LG3: f64 = f64::from_bits(0x3fd2_4924_9422_9359);
const LG4: f64 = f64::from_bits(0x3fcc_71c5_1d8e_78af);
const LG5: f64 = f64::from_bits(0x3fc7_4664_96cb_03de);
const LG6: f64 = f64::from_bits(0x3fc3_9a09_d078_c69f);
const LG7: f64 = f64::from_bits(0x3fc2_f112_df3e_5244);

/// Adding this to the bits of a positive normal `x = 2ᵉ·m`, `m ∈ [1, 2)`,
/// carries into the exponent when `m` reaches `√2` truncated to 20 fraction
/// bits; masking the result and adding [`REDUCED_BASE`] then gives a
/// significand in `[√½, √2)` up to that truncation.
pub(super) const REDUCTION_OFFSET: u64 = 0x3ff0_0000_0000_0000 - 0x3fe6_a09e_0000_0000;
pub(super) const REDUCED_BASE: u64 = 0x3fe6_a09e_0000_0000;

/// Splits a positive normal `x` into `k` and `m ∈ [√½, √2)` with `x = 2ᵏ·m`.
#[inline(always)]
pub(super) fn reduce(x: f64) -> (f64, f64) {
    let bits = x.to_bits().wrapping_add(REDUCTION_OFFSET);
    (
        exponent(bits),
        f64::from_bits((bits & MANTISSA) + REDUCED_BASE),
    )
}

/// `k ln 2 + ln(1 + f) + c` for `1 + f ∈ [√½, √2)` and a small correction `c`:
/// `s = f/(2 + f)` and fdlibm's even polynomial in `s`.
#[inline(always)]
fn log_kernel(f: f64, k: f64, c: f64) -> f64 {
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    s * (hfsq + r) + (k * LN2_LO + c) - hfsq + f + k * LN2_HI
}

/// `ln x` for positive normal finite `x`.
#[inline(always)]
pub(crate) fn ln(x: f64) -> f64 {
    let (k, m) = reduce(x);
    log_kernel(m - 1.0, k, 0.0)
}

/// `ln(1 + x)` for finite `x > −1` (fdlibm `log1p` with every case selected).
#[inline(always)]
pub(crate) fn ln_1p(x: f64) -> f64 {
    let u = 1.0 + x;
    let bits = u.to_bits().wrapping_add(REDUCTION_OFFSET);
    let k = exponent(bits);
    // ln(1 + x) − ln(u), from the rounding error of u = 1 + x.
    let correction = select(k >= 2.0, 1.0 - (u - x), x - (u - 1.0)) / u;
    let reduced = f64::from_bits((bits & MANTISSA) + REDUCED_BASE) - 1.0;
    // √½ ≤ 1 + x < √2, as fdlibm's high-word tests: 1 + x needs no reduction.
    let direct =
        (x > -f64::from_bits(0x3fd2_bec5_0000_0000)) & (x < f64::from_bits(0x3fda_827a_0000_0000));
    let value = log_kernel(
        select(direct, x, reduced),
        select(direct, 0.0, k),
        select(direct | (k >= 54.0), 0.0, correction),
    );
    // |x| < 2⁻⁵³ rounds to x, which keeps signed zeros and subnormals.
    select(x.abs() < f64::from_bits(0x3ca0_0000_0000_0000), x, value)
}
