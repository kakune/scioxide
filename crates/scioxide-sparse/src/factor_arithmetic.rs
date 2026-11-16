//! Immutable arithmetic selection for sparse numerical epochs.
use crate::helpers::{self, Projection};
use core::cmp::Ordering;
use scioxide_core::backend::standard::{product_needs_retry, sum_products};
use scioxide_core::{
    Accuracy, AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

#[derive(Clone, Copy)]
pub(crate) struct Arithmetic(pub(crate) Accuracy);

impl Arithmetic {
    #[inline]
    pub(crate) fn subtract_product<S: NumericScalar>(
        self,
        a: S,
        b: S,
        c: S,
    ) -> Result<S, SciOxideError> {
        if self.0 == Accuracy::Standard {
            let product = b * c;
            let result = a - product;
            if !product_needs_retry(b, c, product) && result.has_finite_components() {
                return Ok(result);
            }
        }
        helpers::subtract_product(a, b, c)
    }

    /// Keep each block entry's original reduction boundary in Exact.
    #[inline]
    pub(crate) fn subtract_products<S: NumericScalar>(
        self,
        prior: S,
        products: impl Iterator<Item = (S, S)> + Clone,
    ) -> Result<S, SciOxideError> {
        sum_products(
            core::iter::once((prior, S::ONE)).chain(products.map(|(a, b)| (-a, b))),
            self.0,
        )
    }

    #[inline]
    pub(crate) fn divide<S: NumericScalar>(
        self,
        a: S,
        b: S,
        policy: Projection,
    ) -> Result<S, SciOxideError> {
        if b == S::ZERO {
            return Err(SciOxideError::singular(None));
        }
        if self.0 == Accuracy::Standard {
            let result = a / b;
            let real = result.real_component();
            let imaginary = result.imaginary_component();
            // Tiny results and possibly disappearing complex components replay
            // the original operands at the existing explicit projection boundary.
            let real_safe = real.abs() >= S::Real::MIN_POSITIVE || a == S::ZERO;
            let imaginary_safe = S::IS_REAL
                || imaginary.abs() >= S::Real::MIN_POSITIVE
                || (a.imaginary_component() == S::Real::ZERO
                    && b.imaginary_component() == S::Real::ZERO);
            if result.has_finite_components() && real_safe && imaginary_safe {
                return Ok(result);
            }
        }
        helpers::divide(a, b, policy)
    }

    #[inline]
    pub(crate) fn compare_magnitude<S: NumericScalar>(
        self,
        a: S,
        b: S,
    ) -> Result<Ordering, SciOxideError> {
        if self.0 == Accuracy::Standard {
            let (a, b) = (a.abs(), b.abs());
            if a.is_finite() && b.is_finite() {
                return Ok(if a > b {
                    Ordering::Greater
                } else if a < b {
                    Ordering::Less
                } else {
                    Ordering::Equal
                });
            }
        }
        Ok(helpers::scaled(a)?
            .magnitude()?
            .abs_cmp(helpers::scaled(b)?.magnitude()?))
    }

    /// Cholesky retries the complete diagonal recurrence before taking its root.
    pub(crate) fn cholesky_diagonal<S: NumericScalar>(
        self,
        prior: S::Real,
        entries: impl Iterator<Item = S> + Clone,
    ) -> Result<S, SciOxideError> {
        if self.0 == Accuracy::Standard {
            let mut diagonal = prior;
            let mut retry = false;
            for l in entries.clone() {
                for component in [l.real_component(), l.imaginary_component()] {
                    let square = component * component;
                    retry |= product_needs_retry(component, component, square);
                    diagonal -= square;
                }
            }
            if !retry && diagonal.is_finite() && diagonal >= S::Real::MIN_POSITIVE {
                return Ok(S::from_real(diagonal.sqrt()));
            }
        }
        let mut diagonal = Scaled::from_value(prior)?;
        for l in entries {
            let l = helpers::scaled(l)?;
            diagonal = diagonal
                .difference_of_products(Scaled::from_value(S::Real::ONE)?, l.real(), l.real())?
                .sub(l.imaginary().mul(l.imaginary())?)?;
        }
        if diagonal.is_zero() || diagonal.is_sign_negative() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "Cholesky requires a strictly positive pivot",
            ));
        }
        helpers::project(
            ScaledValue::from_components(diagonal.sqrt()?, Scaled::ZERO),
            Projection::Strict,
        )
    }
}

/// A checked ordinary intermediate. Zero is permitted; subnormal products retry.
#[inline]
pub(crate) fn ordinary_product<S: NumericScalar>(a: S, b: S) -> Option<S> {
    let value = a * b;
    (!product_needs_retry(a, b, value)).then_some(value)
}

#[inline]
pub(crate) fn ordinary_sum<S: NumericScalar>(a: S, b: S) -> Option<S> {
    let value = a + b;
    value.has_finite_components().then_some(value)
}
