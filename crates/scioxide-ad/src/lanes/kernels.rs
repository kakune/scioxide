//! Routes the eight kernel functions of [`Lanes`](super::Lanes) to the lane kernels
//! of `scioxide_core::backend`.
//!
//! Lanes are widened to `f64`, evaluated by the lane kernel and rounded back, so
//! `f32` lanes are evaluated in `f64`; for `f64` the conversions vanish.

use scioxide_core::RealScalar;
use scioxide_core::backend;

macro_rules! unary_kernel {
    ($($name:ident => $kernel:ident),*) => {
        $(
            #[inline(always)]
            pub(super) fn $name<S: RealScalar, const W: usize>(x: [S; W]) -> [S; W] {
                backend::$kernel(x.map(S::into_f64)).map(S::from_f64_ieee)
            }
        )*
    };
}

unary_kernel!(
    exp => lanes_exp,
    exp_m1 => lanes_exp_m1,
    ln => lanes_ln,
    ln_1p => lanes_ln_1p,
    sin => lanes_sin,
    cos => lanes_cos,
    tanh => lanes_tanh
);

#[inline(always)]
pub(super) fn powf<S: RealScalar, const W: usize>(base: [S; W], exponent: [S; W]) -> [S; W] {
    backend::lanes_powf(base.map(S::into_f64), exponent.map(S::into_f64)).map(S::from_f64_ieee)
}
