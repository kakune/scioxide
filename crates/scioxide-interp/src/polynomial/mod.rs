//! Global polynomial interpolants on increasing real nodes.
//!
//! Barycentric, Newton, Neville and Hermite interpolation, Bulirsch–Stoer
//! diagonal rational interpolation, and the antiderivative of a global
//! polynomial. Each interpolant stores values node by node
//! (`[node][component]`) and evaluates through [`crate::Interpolant`] with a
//! [`crate::QueryWorkspace`].

pub(crate) mod antiderivative;
pub(crate) mod barycentric;
pub(crate) mod bulirsch_stoer;
pub(crate) mod hermite;
pub(crate) mod neville;
pub(crate) mod newton;
mod tableau;

pub use antiderivative::PreparedPolynomialAntiderivative;
pub use barycentric::BarycentricInterpolator;
pub use bulirsch_stoer::BulirschStoerInterpolator;
pub use hermite::HermiteInterpolator;
pub use neville::NevilleInterpolator;
pub use newton::NewtonInterpolator;
