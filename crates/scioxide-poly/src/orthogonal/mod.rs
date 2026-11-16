//! Classical orthogonal families, prepared bases and shared recurrence arithmetic.
mod admission;
mod basis;
mod calculus;
pub(crate) mod coefficients;
mod conversion;
mod family;
pub(crate) mod recurrence;
pub(crate) mod zeros;
pub use basis::{BasisNormalization, OrthogonalBasis, OrthogonalBasisPlan, OrthogonalBasisSpec};
pub use family::OrthogonalFamily;
pub use zeros::JacobiRecurrence;
#[cfg(test)]
mod calculus_tests;
#[cfg(test)]
mod chebyshev_tests;
#[cfg(test)]
mod hermite_tests;
#[cfg(test)]
mod jacobi_tests;
#[cfg(test)]
mod laguerre_tests;
#[cfg(test)]
mod legendre_tests;

pub use calculus::{OrthogonalAntiderivative, OrthogonalDerivative};
pub use conversion::OrthogonalTransform;
