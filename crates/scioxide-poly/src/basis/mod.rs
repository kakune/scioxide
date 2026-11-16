//! Floating-point polynomial bases, conversions between them and their diagnostics.
use crate::PolynomialScratchRequirements;
use crate::util::{
    coefficient_scale, empty_error, exact_len, finite_input, minimum_len, round_complex,
    rounding_error, scaled_complex, validate_slice,
};
use scioxide_core::ExecutionContext;
use scioxide_core::backend::try_copy_slice;
use scioxide_core::backend::{checked_add, checked_mul};

use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::AlgebraicScalar;

use crate::backend::polynomial_basis::{
    ValidatedNodes, numeric_divided_differences, weights_for_nodes,
};
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Argument, InvalidInputReason, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_core::{Scaled, ScaledValue};
use scioxide_linalg::RankPolicy;

mod barycentric;
mod bernstein;
pub(crate) mod casteljau;
mod conversion;
mod monomial;
mod newton;
mod policy;
mod polynomial;
mod root_matrix;
mod root_product;
#[cfg(test)]
mod tests;
mod validation;

pub use barycentric::BarycentricBasis;
pub use bernstein::{BernsteinBasis, BernsteinBounds};
pub(crate) use bernstein::{bernstein_derivative_coefficients_into, control_bounds};
pub(crate) use conversion::conversion_result;
use conversion::{compose_scaled_affine, project_coefficients};
pub use monomial::MonomialBasis;
use monomial::evaluate_monomial;
pub use newton::NewtonBasis;
use policy::scale_ratio;
pub(crate) use policy::{EvaluationEnvelope, absolute_envelope};
pub use policy::{
    PolynomialApproximationPolicy, PolynomialCoefficientReport, PolynomialConditionReport,
    PolynomialConversion, PolynomialEvaluation,
};
pub(crate) use policy::{
    condition_report, condition_report_from_scale, rank_threshold, scaled_condition_ratio,
};
pub use polynomial::ApproxPolynomial;
pub use root_matrix::{PolynomialRootBasis, PolynomialRootBasisRef, polynomial_root_matrix_into};
use root_product::{
    balanced_coefficients_from_roots, balanced_product, leja_order, round_monic_product,
    scaled_factor,
};
pub(crate) use validation::interval;
use validation::*;

/// Gives read access to the stored coordinates of a polynomial basis.
///
/// The trait is sealed: it is implemented by [`MonomialBasis`],
/// [`NewtonBasis`], [`BarycentricBasis`], [`BernsteinBasis`] and
/// [`OrthogonalBasis`](crate::OrthogonalBasis), and cannot be implemented
/// outside this crate:
///
/// ```compile_fail
/// use scioxide_poly::PolynomialBasis;
/// #[derive(Clone, PartialEq)]
/// struct ExternalBasis;
/// impl PolynomialBasis<f64> for ExternalBasis {
///     fn coefficients(&self) -> &[f64] {
///         &[1.0_f64]
///     }
/// }
/// ```
pub trait PolynomialBasis<C: NumericScalar>: private::Prepared<C> + Clone + PartialEq {
    /// Returns the stored coordinates.
    ///
    /// These are the ascending monomial, Newton or orthogonal coefficients, the
    /// Bernstein control coefficients, or the barycentric nodal values in node
    /// order. Call it on [`ApproxPolynomial::basis`] with this trait in scope.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0])?;
    /// assert_eq!(PolynomialBasis::coefficients(p.basis()).len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn coefficients(&self) -> &[C];

    /// Returns the structural degree, the number of stored coordinates minus one.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0])?;
    /// assert_eq!(PolynomialBasis::degree(p.basis()), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn degree(&self) -> usize {
        self.coefficients().len().saturating_sub(1)
    }
}

pub(crate) mod private {
    use super::*;
    pub trait Prepared<C: NumericScalar> {
        /// Explicitly classify degree-bearing coordinates without pruning.
        fn effective_degree(
            &self,
            tolerance: scioxide_core::Tolerance<C::Real>,
        ) -> Result<usize, SciOxideError>
        where
            Self: PolynomialBasis<C>,
        {
            let coefficients = self.coefficients();
            let threshold = tolerance.threshold(coefficient_scale(coefficients))?;
            Ok(coefficients
                .iter()
                .rposition(|c| c.abs() > threshold)
                .unwrap_or(0))
        }

        /// Minimum scalar scratch for evaluation and basis-native root conditioning.
        fn scratch_len(&self) -> usize {
            0
        }

        /// Evaluate value, derivative, and the absolute evaluation envelope.
        fn evaluate_prepared(
            &self,
            point: C,
            scratch: &mut [C],
        ) -> Result<(C, C, EvaluationEnvelope<C::Real>), SciOxideError>;

        /// Basis-native sensitivity using the representation's evaluation envelope.
        fn root_condition_prepared(
            &self,
            point: C,
            scratch: &mut [C],
        ) -> Result<crate::RootConditionReport<C::Real>, SciOxideError> {
            let (value, derivative, envelope) = self.evaluate_prepared(point, scratch)?;
            let magnitude = scaled_complex(value)?.magnitude()?;
            crate::root_diagnostics::condition_from_scaled(
                envelope.scaled(magnitude)?,
                magnitude,
                scaled_complex(derivative)?.magnitude()?,
                scaled_complex(point)?.magnitude()?,
            )
        }

        /// Materialize ascending monomial coefficients for an explicit conversion.
        fn materialize_monomial(&self) -> Result<Vec<C>, SciOxideError>;
    }
}
