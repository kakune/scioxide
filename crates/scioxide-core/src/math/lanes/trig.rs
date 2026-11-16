//! Trigonometric kernels: `sin` and `cos` for `|x| ≤ 2²⁰`.

use super::{blend, mask, round_to_integer};
use core::f64::consts::FRAC_2_PI;

/// Largest magnitude evaluated; larger arguments need Payne–Hanek reduction.
pub(crate) const LIMIT: f64 = 1_048_576.0;

// fdlibm splits of π/2: PIO2_1, PIO2_2 and PIO2_3 have 33 significant bits, so
// their products with |n| < 2²⁰ are exact, and PIO2_3T is the remainder.
const PIO2_1: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
const PIO2_2: f64 = f64::from_bits(0x3dd0_b461_1a60_0000);
const PIO2_3: f64 = f64::from_bits(0x3ba3_198a_2e00_0000);
const PIO2_3T: f64 = f64::from_bits(0x397b_839a_2520_49c1);

// fdlibm `__kernel_sin` and `__kernel_cos` coefficients.
const S1: f64 = f64::from_bits(0xbfc5_5555_5555_5549);
const S2: f64 = f64::from_bits(0x3f81_1111_1110_f8a6);
const S3: f64 = f64::from_bits(0xbf2a_01a0_19c1_61d5);
const S4: f64 = f64::from_bits(0x3ec7_1de3_57b1_fe7d);
const S5: f64 = f64::from_bits(0xbe5a_e5e6_8a2b_9ceb);
const S6: f64 = f64::from_bits(0x3de5_d93a_5acf_d57c);
const C1: f64 = f64::from_bits(0x3fa5_5555_5555_554c);
const C2: f64 = f64::from_bits(0xbf56_c16c_16c1_5177);
const C3: f64 = f64::from_bits(0x3efa_01a0_19cb_1590);
const C4: f64 = f64::from_bits(0xbe92_7e4f_809c_52ad);
const C5: f64 = f64::from_bits(0x3e21_ee9e_bdb4_b1c4);
const C6: f64 = f64::from_bits(0xbda8_fae9_be88_38d4);

/// Returns `(y0, y1, n)` with `y0 + y1 = x − n·π/2` to about 2⁻¹⁵⁰ relative
/// to `π/2`, `|y0 + y1| ≤ π/4` and `n` in two's-complement bits.
///
/// Three-constant Cody–Waite reduction: `r₁ = x − n·PIO2_1` is exact, the
/// rounding errors of the two further subtractions are recovered exactly by
/// fast two-sums for `|x| ≤ 2²⁰`, and they join the last constant's product.
#[inline(always)]
fn reduce(x: f64) -> (f64, f64, u64) {
    let (n, bits) = round_to_integer(x * FRAC_2_PI);
    let r1 = x - n * PIO2_1;
    let p2 = n * PIO2_2;
    let r2 = r1 - p2;
    let p3 = n * PIO2_3;
    let r3 = r2 - p3;
    let w = n * PIO2_3T - (((r1 - r2) - p2) + ((r2 - r3) - p3));
    let y0 = r3 - w;
    (y0, (r3 - y0) - w, bits)
}

/// fdlibm `__kernel_sin(x, y, 1)` for `|x + y| ≤ π/4`.
#[inline(always)]
fn sin_kernel(x: f64, y: f64) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    x - ((z * (0.5 * y - v * r) - y) - v * S1)
}

/// fdlibm `__kernel_cos(x, y)` for `|x + y| ≤ π/4`.
#[inline(always)]
fn cos_kernel(x: f64, y: f64) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

/// Selects `±sin_kernel` or `±cos_kernel` for quadrant `q` (mod 4) of
/// `sin(y + q·π/2)`.
#[inline(always)]
fn quadrant(y0: f64, y1: f64, q: u64) -> f64 {
    let value = blend(mask(q & 1 == 1), cos_kernel(y0, y1), sin_kernel(y0, y1));
    f64::from_bits(value.to_bits() ^ ((q & 2) << 62))
}

/// `sin x` for `|x| ≤ LIMIT`.
#[inline(always)]
pub(crate) fn sin(x: f64) -> f64 {
    let (y0, y1, n) = reduce(x);
    quadrant(y0, y1, n)
}

/// `cos x` for `|x| ≤ LIMIT`, as `sin(x + π/2)`.
#[inline(always)]
pub(crate) fn cos(x: f64) -> f64 {
    let (y0, y1, n) = reduce(x);
    quadrant(y0, y1, n.wrapping_add(1))
}
