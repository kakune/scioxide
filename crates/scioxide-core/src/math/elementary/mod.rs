//! Bounded portable elementary kernels, adapted from fdlibm/musl via libm 0.2.16.
//! Only the `f64` round-to-nearest-even paths are kept. `f32` inputs widen exactly
//! and round on return; their independent fixture envelope includes that rounding.
// Coefficients retain the published decimal spellings and evaluation order.

// The helpers below reproduce the fdlibm vocabulary (32-bit high/low word
// access, sqrt, floor, scalbn) on top of SciOxide's own primitives, so that
// ported kernels keep their upstream structure and constants unchanged.
// Upstream expressions wrapped in `core::hint::black_box` exist only to
// raise IEEE inexact or underflow flags. Rust exposes no floating-point
// environment, so they have no observable effect here and are kept for
// line-by-line fidelity with the reference.
use super::scalbn_f64 as scalbn;
#[cfg(not(feature = "std"))]
fn sqrt(x: f64) -> f64 {
    crate::RealScalar::sqrt(x)
}
#[cfg(not(feature = "std"))]
fn fabs(x: f64) -> f64 {
    x.abs()
}
#[cfg(not(feature = "std"))]
fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}
#[cfg(not(feature = "std"))]
fn with_set_high_word(x: f64, hi: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff) | (u64::from(hi) << 32))
}
#[cfg(not(feature = "std"))]
fn with_set_low_word(x: f64, lo: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | u64::from(lo))
}
#[cfg(not(feature = "std"))]
fn get_low_word(x: f64) -> u32 {
    x.to_bits() as u32
}
#[cfg(not(feature = "std"))]
fn combine_words(hi: u32, lo: u32) -> f64 {
    f64::from_bits((u64::from(hi) << 32) | u64::from(lo))
}
fn floor(x: f64) -> f64 {
    crate::RealScalar::floor(x)
}
#[cfg(not(feature = "std"))]
mod sincos;
#[cfg(not(feature = "std"))]
pub(super) use sincos::sincos;
#[cfg(not(feature = "std"))]
mod tan;
#[cfg(not(feature = "std"))]
pub(super) use tan::tan;
#[cfg(not(feature = "std"))]
mod rem_pio2;
#[cfg(not(feature = "std"))]
pub(super) use rem_pio2::rem_pio2;
mod rem_pio2_large;
pub(crate) use rem_pio2_large::IPIO2;
pub(super) use rem_pio2_large::rem_pio2_large;
pub(super) use rem_pio2_large::rem_pio2_large_precise;
mod k_sin;
pub(super) use k_sin::k_sin;
mod k_cos;
pub(super) use k_cos::k_cos;
#[cfg(not(feature = "std"))]
mod k_tan;
#[cfg(not(feature = "std"))]
pub(super) use k_tan::k_tan;
mod exp;
pub(super) use exp::exp;
mod exp2;
#[cfg(not(feature = "std"))]
pub(super) use exp2::exp2;
pub(super) use exp2::positive_exp;
#[cfg(not(feature = "std"))]
mod expm1;
#[cfg(not(feature = "std"))]
pub(super) use expm1::expm1;
#[cfg(not(feature = "std"))]
mod log;
#[cfg(not(feature = "std"))]
pub(super) use log::log;
mod log1p;
pub(super) use log1p::log1p;
#[cfg(not(feature = "std"))]
mod log2;
#[cfg(not(feature = "std"))]
pub(super) use log2::log2;
#[cfg(not(feature = "std"))]
mod log10;
#[cfg(not(feature = "std"))]
pub(super) use log10::log10;
#[cfg(not(feature = "std"))]
mod pow;
#[cfg(not(feature = "std"))]
pub(super) use pow::pow;
#[cfg(not(feature = "std"))]
mod hypot;
#[cfg(not(feature = "std"))]
pub(super) use hypot::hypot;
#[cfg(not(feature = "std"))]
mod asin;
#[cfg(not(feature = "std"))]
pub(super) use asin::asin;
#[cfg(not(feature = "std"))]
mod acos;
#[cfg(not(feature = "std"))]
pub(super) use acos::acos;
#[cfg(not(feature = "std"))]
mod atan;
#[cfg(not(feature = "std"))]
pub(super) use atan::atan;
#[cfg(not(feature = "std"))]
mod atan2;
#[cfg(not(feature = "std"))]
pub(super) use atan2::atan2;
#[cfg(not(feature = "std"))]
mod sinh;
#[cfg(not(feature = "std"))]
pub(super) use sinh::sinh;
#[cfg(not(feature = "std"))]
mod cosh;
#[cfg(not(feature = "std"))]
pub(super) use cosh::cosh;
#[cfg(not(feature = "std"))]
mod tanh;
#[cfg(not(feature = "std"))]
pub(super) use tanh::tanh;
mod asinh;
pub(super) use asinh::asinh;
#[cfg(not(feature = "std"))]
mod acosh;
#[cfg(not(feature = "std"))]
pub(super) use acosh::acosh;
mod atanh;
pub(super) use atanh::atanh;
#[cfg(not(feature = "std"))]
mod expo2;
#[cfg(not(feature = "std"))]
pub(super) use expo2::expo2;
#[cfg(not(feature = "std"))]
mod k_expo2;
#[cfg(not(feature = "std"))]
pub(super) use k_expo2::k_expo2;
