//! Values that are vectors over the reals, such as scalars, complex numbers and arrays.
//!
//! An [`AffineValue`] is read as a list of real components. A complex number has
//! two components, real then imaginary; arrays and slices list the components
//! of their elements in order. Two values with the same number of components
//! have compatible shapes, so a complex value can be written to a two-element
//! real array. Components are not ordered or interpreted as coordinates.

use crate::RangeAccumulator;
use crate::backend::try_with_capacity;
use crate::{Complex, Dimensions, NonFiniteOrigin, RealScalar, SciOxideError};
use alloc::vec::Vec;

/// A value that is a vector over the reals, read as a list of real components.
///
/// Interpolation and automatic differentiation use this trait to treat
/// scalars, complex numbers and arrays alike. Implementations must not
/// allocate in `dimension` or `component`, and `component(i)` must succeed for
/// every `i < dimension()`. The component count and order must not change while
/// the value is borrowed.
pub trait AffineValue {
    /// Real scalar type of the components and of the weights.
    type Scale: RealScalar;
    /// Owned, mutable type returned by [`AffineValue::try_to_owned`].
    type Owned: AffineValueMut<Scale = Self::Scale>;
    /// Number of components, when the type fixes it.
    const FIXED_DIMENSION: Option<usize> = None;
    /// Returns the number of real components; zero is valid.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the count does not fit in `usize`.
    fn dimension(&self) -> Result<usize, SciOxideError>;
    /// Returns the real component at `index`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.dimension()`.
    fn component(&self, index: usize) -> Result<Self::Scale, SciOxideError>;
    /// Returns the only component of a one-component value.
    ///
    /// The bits are returned unchanged, including the sign of zero, NaN and
    /// infinities.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the value does not have exactly one component.
    /// - Errors from [`Self::dimension`] and [`Self::component`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::AffineValue;
    ///
    /// assert_eq!(3.0_f64.try_to_scalar()?, 3.0);
    /// assert!([1.0_f64, 2.0].try_to_scalar().is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline(always)]
    fn try_to_scalar(&self) -> Result<Self::Scale, SciOxideError> {
        let dimension = self.dimension()?;
        if dimension != 1 {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(1),
                Dimensions::one(dimension),
            ));
        }
        self.component(0)
    }
    /// Returns an owned copy with the same components in the same order.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if storage for the copy cannot be reserved.
    fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError>;
}

/// An [`AffineValue`] whose components can be overwritten.
///
/// Implementations must visit each of the `dimension()` components exactly
/// once, in the order of [`AffineValue::component`], without changing the shape
/// or allocating.
pub trait AffineValueMut: AffineValue {
    /// Calls `visit` on each real component, in order.
    fn for_each_component_mut(&mut self, visit: impl FnMut(&mut Self::Scale));
}

// Trait-owned identities shared by every `AffineValue` implementation here.

fn index_error(index: usize, length: usize) -> SciOxideError {
    SciOxideError::out_of_bounds(crate::quantity!("affine_component"), index, length)
}

impl<S: RealScalar> AffineValue for S {
    type Scale = S;
    type Owned = S;
    const FIXED_DIMENSION: Option<usize> = Some(1);
    #[inline(always)]
    fn dimension(&self) -> Result<usize, SciOxideError> {
        Ok(1)
    }
    #[inline(always)]
    fn component(&self, index: usize) -> Result<S, SciOxideError> {
        if index == 0 {
            Ok(*self)
        } else {
            Err(index_error(index, 1))
        }
    }
    #[inline(always)]
    fn try_to_scalar(&self) -> Result<S, SciOxideError> {
        Ok(*self)
    }
    fn try_to_owned(&self) -> Result<S, SciOxideError> {
        Ok(*self)
    }
}
impl<S: RealScalar> AffineValueMut for S {
    #[inline(always)]
    fn for_each_component_mut(&mut self, mut visit: impl FnMut(&mut S)) {
        visit(self);
    }
}
impl<S: RealScalar> AffineValue for Complex<S> {
    type Scale = S;
    type Owned = Self;
    const FIXED_DIMENSION: Option<usize> = Some(2);
    fn dimension(&self) -> Result<usize, SciOxideError> {
        Ok(2)
    }
    fn component(&self, index: usize) -> Result<S, SciOxideError> {
        match index {
            0 => Ok(self.re),
            1 => Ok(self.im),
            _ => Err(index_error(index, 2)),
        }
    }
    fn try_to_owned(&self) -> Result<Self, SciOxideError> {
        Ok(*self)
    }
}
impl<S: RealScalar> AffineValueMut for Complex<S> {
    fn for_each_component_mut(&mut self, mut visit: impl FnMut(&mut S)) {
        visit(&mut self.re);
        visit(&mut self.im);
    }
}
impl<V: AffineValue> AffineValue for [V] {
    type Scale = V::Scale;
    type Owned = Vec<V::Owned>;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        match V::FIXED_DIMENSION {
            Some(width) => crate::backend::checked_mul(
                self.len(),
                width,
                crate::quantity!("affine_component_count"),
            ),
            None => self.iter().try_fold(0, |count, value| {
                crate::backend::checked_add(
                    count,
                    value.dimension()?,
                    crate::quantity!("affine_component_count"),
                )
            }),
        }
    }
    fn component(&self, mut index: usize) -> Result<Self::Scale, SciOxideError> {
        let length = self.dimension()?;
        if index >= length {
            return Err(index_error(index, length));
        }
        // Fixed-width elements allow O(1) indexing by division. Variable-width
        // elements need a linear scan over the concatenated components.
        // `checked_div` also covers a zero width.
        if let Some(width) = V::FIXED_DIMENSION
            && let Some(element) = index.checked_div(width)
        {
            return self
                .get(element)
                .ok_or_else(|| index_error(index, length))?
                .component(index % width);
        }
        let original = index;
        for value in self {
            let width = value.dimension()?;
            if index < width {
                return value.component(index);
            }
            index -= width;
        }
        Err(index_error(original, length))
    }
    fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError> {
        let mut output = try_with_capacity(self.len(), crate::quantity!("affine_owned_copy"))?;
        for value in self {
            output.push(value.try_to_owned()?);
        }
        Ok(output)
    }
}
impl<V: AffineValueMut> AffineValueMut for [V] {
    fn for_each_component_mut(&mut self, mut visit: impl FnMut(&mut Self::Scale)) {
        for value in self {
            value.for_each_component_mut(&mut visit);
        }
    }
}
impl<V: AffineValue> AffineValue for Vec<V> {
    type Scale = V::Scale;
    type Owned = Vec<V::Owned>;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        self.as_slice().dimension()
    }
    fn component(&self, index: usize) -> Result<Self::Scale, SciOxideError> {
        self.as_slice().component(index)
    }
    fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError> {
        AffineValue::try_to_owned(self.as_slice())
    }
}
impl<V: AffineValueMut> AffineValueMut for Vec<V> {
    fn for_each_component_mut(&mut self, visit: impl FnMut(&mut Self::Scale)) {
        self.as_mut_slice().for_each_component_mut(visit);
    }
}
impl<V: AffineValueMut + Copy, const N: usize> AffineValue for [V; N] {
    type Scale = V::Scale;
    type Owned = Self;
    const FIXED_DIMENSION: Option<usize> = if N == 0 {
        Some(0)
    } else {
        match V::FIXED_DIMENSION {
            Some(width) => N.checked_mul(width),
            None => None,
        }
    };
    fn dimension(&self) -> Result<usize, SciOxideError> {
        self.as_slice().dimension()
    }
    fn component(&self, index: usize) -> Result<Self::Scale, SciOxideError> {
        self.as_slice().component(index)
    }
    fn try_to_owned(&self) -> Result<Self, SciOxideError> {
        Ok(*self)
    }
}
impl<V: AffineValueMut + Copy, const N: usize> AffineValueMut for [V; N] {
    fn for_each_component_mut(&mut self, visit: impl FnMut(&mut Self::Scale)) {
        self.as_mut_slice().for_each_component_mut(visit);
    }
}

/// Checks that `value` has `dimension` real components.
///
/// # Errors
///
/// - `DimensionMismatch` if the component count differs.
#[inline]
pub fn check_value_shape<V: AffineValue + ?Sized>(
    value: &V,
    dimension: usize,
) -> Result<(), SciOxideError> {
    if value.dimension()? != dimension {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(dimension),
            Dimensions::one(value.dimension()?),
        ));
    }
    Ok(())
}

/// Checks that every real component of `value` is finite.
///
/// # Errors
///
/// - `NonFinite` if a component is NaN or infinite.
/// - Errors from [`AffineValue::dimension`] and [`AffineValue::component`].
#[inline]
pub fn validate_value<V: AffineValue + ?Sized>(value: &V) -> Result<(), SciOxideError> {
    for index in 0..value.dimension()? {
        if !value.component(index)?.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "value components must be finite",
            ));
        }
    }
    Ok(())
}

/// Writes `components` into `output`, in component order.
///
/// On error, `output` is unchanged.
///
/// # Errors
///
/// - `DimensionMismatch` if `output` does not have `components.len()` components.
/// - `NonFinite` if a component is NaN or infinite.
#[inline(always)]
pub fn write_value<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    components: &[S],
    output: &mut O,
) -> Result<(), SciOxideError> {
    check_value_shape(output, components.len())?;
    validate_value(components)?;
    if components.is_empty() {
        return Ok(());
    }
    let mut components = components.iter();
    output.for_each_component_mut(|slot| {
        if let Some(value) = components.next() {
            *slot = *value;
        }
    });
    Ok(())
}

/// Computes the weighted sum `Σ wᵢ vᵢ` of `terms = [(wᵢ, vᵢ), …]` into `output`.
///
/// Each component is accumulated exactly and rounded once, so the result is
/// correctly rounded and does not depend on the order of the terms. An empty
/// `terms` writes zero. `workspace` must have one entry per component of
/// `output`; its contents are unspecified afterward. The cost is
/// O(components × terms) when component access takes constant time, including
/// fixed-width arrays. Variable-width nested slices scan their elements on each
/// access; for `n` one-component nested vectors the cost is O(n² × terms).
/// On error, `output` is unchanged.
///
/// # Errors
///
/// - `DimensionMismatch` if `output`, a term value or `workspace` has a
///   different component count.
/// - `NonFinite` if a weight or a component is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result component exceeds the
///   largest finite value.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, affine_combination_into};
///
/// // 0.75·(2 − 4i) + 0.25·(6 + 8i) = 3 − i
/// let left = Complex::new(2.0_f64, -4.0);
/// let right = Complex::new(6.0_f64, 8.0);
/// let mut output = Complex::new(0.0, 0.0);
/// let mut scratch = [0.0; 2];
/// affine_combination_into(&[(0.75, &left), (0.25, &right)], &mut output, &mut scratch)?;
/// assert_eq!(output, Complex::new(3.0, -1.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn affine_combination_into<S, V, O>(
    terms: &[(S, &V)],
    output: &mut O,
    workspace: &mut [S],
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S> + ?Sized,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    check_value_shape(output, workspace.len())?;
    for &(weight, value) in terms {
        check_value_shape(value, workspace.len())?;
        validate_value(value)?;
        if !weight.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "weight must be finite",
            ));
        }
    }
    // Each component is an exactly accumulated sum of products, rounded once,
    // so the combination does not depend on term order. Results are staged in
    // `workspace` and written only after every component is known to be
    // finite.
    for (index, slot) in workspace.iter_mut().enumerate() {
        let mut sum = RangeAccumulator::new();
        for &(weight, value) in terms {
            sum.add_product(weight, value.component(index)?)?;
        }
        *slot = sum.finish()?;
    }
    write_value(workspace, output)
}
