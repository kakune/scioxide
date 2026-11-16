//! Zeta, polylogarithm, Lambert, Lerch, and Clausen kernels.

pub(crate) mod clausen;
mod clausen_coefficients;
mod clausen_generated;
pub(crate) mod complex_zeta;
pub(crate) mod lambert;
mod lerch;
pub(crate) mod negative;
pub(crate) mod polylog;
mod transcendental;
pub(crate) mod zeta;

pub(crate) mod api;
pub(crate) mod real_api;
