//! Branch-free binary64 kernels behind the `lanes_*` routes of the registry.
//!
//! A kernel is a straight-line sequence of IEEE operations and bit operations
//! on one `f64`: no table lookup, no fused multiply-add, and special values by
//! selection. Every lane runs the same sequence, so a loop over `[f64; W]`
//! vectorizes for the enabled target features and each result is independent
//! of `W` and of FMA support. The registry's lane routes recompute lanes
//! outside a kernel's domain with the scalar route, or for `exp` beyond `±708`
//! with the kernel's full-range form.

pub(crate) mod exp;
pub(crate) mod log;
pub(crate) mod pow;
pub(crate) mod trig;

const SIGN: u64 = 1 << 63;
const MANTISSA: u64 = (1 << 52) - 1;
/// `1.5 · 2⁵²`: for `|v| < 2⁵¹`, `v + SHIFT` is `SHIFT + k` with `k` the
/// nearest integer to `v` (ties to even), and its bits are `SHIFT`'s plus `k`.
const SHIFT: f64 = 6_755_399_441_055_744.0;
/// `2²⁷ + 1`, Veltkamp's splitting constant for binary64.
const SPLITTER: f64 = 134_217_729.0;

#[inline(always)]
fn select(condition: bool, yes: f64, no: f64) -> f64 {
    if condition { yes } else { no }
}

/// All ones if `condition`, else zero.
#[inline(always)]
fn mask(condition: bool) -> u64 {
    0_u64.wrapping_sub(u64::from(condition))
}

/// Selects by a [`mask`]. Sign flips and selections between a value and its
/// negation use this bit form, which vectorizes more reliably than [`select`].
#[inline(always)]
fn blend(mask: u64, yes: f64, no: f64) -> f64 {
    f64::from_bits((yes.to_bits() & mask) | (no.to_bits() & !mask))
}

/// Rounds `value` (`|value| < 2⁵¹`) to the nearest integer `k`, returned as an
/// `f64` and as two's-complement bits.
#[inline(always)]
fn round_to_integer(value: f64) -> (f64, u64) {
    let shifted = value + SHIFT;
    (
        shifted - SHIFT,
        shifted.to_bits().wrapping_sub(SHIFT.to_bits()),
    )
}

/// Multiplies `value` by `2ᵏ` for `|k| ≤ 2044` as `value·2^⌊k/2⌋·2^⌈k/2⌉`.
/// For `value` near one the first product is exact and normal, so a
/// subnormal result is rounded once.
#[inline(always)]
fn scale(value: f64, k: u64) -> f64 {
    let first = (k.wrapping_add(2048) >> 1).wrapping_sub(1);
    let second = k.wrapping_add(2046).wrapping_sub(first);
    value * f64::from_bits(first << 52) * f64::from_bits(second << 52)
}

/// Returns the biased exponent field of `bits` minus 1023, exactly, for a
/// clear sign bit.
#[inline(always)]
fn exponent(bits: u64) -> f64 {
    const BASE: u64 = 0x4330_0000_0000_0000;
    f64::from_bits(BASE | (bits >> 52)) - (f64::from_bits(BASE) + 1023.0)
}

/// Splits `value` into a high part of at most 26 significant bits and an
/// exact remainder (Veltkamp).
#[inline(always)]
fn split(value: f64) -> (f64, f64) {
    let scaled = SPLITTER * value;
    let high = scaled - (scaled - value);
    (high, value - high)
}

/// Returns `a·b` rounded and its exact rounding error (Dekker), without FMA.
#[inline(always)]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    let (a_high, a_low) = split(a);
    let (b_high, b_low) = split(b);
    let error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low;
    (product, error)
}

/// Returns `a + b` rounded and its exact rounding error for `|a| ≥ |b|` or
/// `a = 0` (Dekker's fast two-sum).
#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    (sum, b - (sum - a))
}
