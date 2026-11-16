//! Elliptic, Carlson, and Jacobi kernels and continuations.

pub(crate) mod carlson;
mod complete;
pub(crate) mod complex_elliptic;
mod complex_jacobi;
pub(crate) mod elliptic;
pub(crate) mod elliptic_kernel;
pub(crate) mod elliptic_quadrature;
mod elliptic_real;
pub(crate) mod jacobi;

pub(crate) mod api;
pub(crate) mod real_api;
