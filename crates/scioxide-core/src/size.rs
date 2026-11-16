//! Checked size arithmetic and fallible allocation of vectors.
//!
//! Use these helpers for shapes, capacities and workspace lengths computed from
//! user input. Overflow of a size is a `SizeOverflow` error and a failed
//! reservation is an `AllocationFailure`; both differ from numerical errors.

use alloc::vec::Vec;

use crate::{Dimensions, Quantity, SciOxideError};

/// Checks that two one-dimensional lengths are equal.
pub(crate) fn validate_same_len(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(actual),
        ))
    }
}

// Rank-zero shapes describe one scalar.
fn shape_is_empty(dimensions: &[usize]) -> bool {
    dimensions.contains(&0)
}

/// Adds two sizes, reporting overflow instead of wrapping.
///
/// # Errors
///
/// - `SizeOverflow` (naming `quantity`) if the sum exceeds `usize::MAX`.
///
/// ```
/// use scioxide_core::backend::checked_add;
/// assert_eq!(checked_add(2, 3, scioxide_core::quantity!("length"))?, 5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_add(a: usize, b: usize, quantity: Quantity) -> Result<usize, SciOxideError> {
    a.checked_add(b)
        .ok_or_else(|| SciOxideError::size_overflow(quantity))
}

/// Multiplies two sizes, reporting overflow instead of wrapping.
///
/// Zero times any size is zero.
///
/// # Errors
///
/// - `SizeOverflow` (naming `quantity`) if the product exceeds `usize::MAX`.
///
/// ```
/// use scioxide_core::backend::checked_mul;
/// assert_eq!(checked_mul(0, usize::MAX, scioxide_core::quantity!("length"))?, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_mul(a: usize, b: usize, quantity: Quantity) -> Result<usize, SciOxideError> {
    a.checked_mul(b)
        .ok_or_else(|| SciOxideError::size_overflow(quantity))
}

/// Returns the product of all dimensions, reporting overflow instead of wrapping.
///
/// An empty list has product 1. If any dimension is zero the product is 0, even
/// when the other dimensions would overflow.
///
/// # Errors
///
/// - `SizeOverflow` (naming `quantity`) if a partial product exceeds `usize::MAX`.
///
/// ```
/// use scioxide_core::backend::checked_product;
///
/// let quantity = scioxide_core::quantity!("length");
/// assert_eq!(checked_product(&[], quantity)?, 1);
/// assert_eq!(checked_product(&[usize::MAX, usize::MAX, 0], quantity)?, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_product(dimensions: &[usize], quantity: Quantity) -> Result<usize, SciOxideError> {
    // Zero extents are checked first: a shape with any zero extent is empty
    // even when the product of its other extents would overflow.
    if shape_is_empty(dimensions) {
        return Ok(0);
    }
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        checked_mul(product, dimension, quantity)
    })
}

/// Returns the smallest power of two that is at least `n`; zero gives one.
///
/// # Errors
///
/// - `SizeOverflow` (naming `quantity`) if that power of two exceeds `usize::MAX`.
///
/// ```
/// use scioxide_core::backend::checked_next_power_of_two;
/// assert_eq!(checked_next_power_of_two(0, scioxide_core::quantity!("length"))?, 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_next_power_of_two(n: usize, quantity: Quantity) -> Result<usize, SciOxideError> {
    if n == 0 {
        return Ok(1);
    }
    n.checked_next_power_of_two()
        .ok_or_else(|| SciOxideError::size_overflow(quantity))
}

/// Creates an empty vector with room for at least `capacity` elements.
///
/// The capacity is reserved once; the allocator may round it up.
///
/// # Errors
///
/// - `AllocationFailure` (naming `quantity`) if the memory cannot be reserved.
///
/// ```
/// use scioxide_core::backend::try_with_capacity;
/// let values = try_with_capacity::<f64>(4, scioxide_core::quantity!("scratch"))?;
/// assert!(values.is_empty());
/// assert!(values.capacity() >= 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn try_with_capacity<T>(capacity: usize, quantity: Quantity) -> Result<Vec<T>, SciOxideError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| SciOxideError::allocation_failure(quantity, capacity))?;
    Ok(values)
}

/// Reserves room for at least `additional` more elements in `values`.
///
/// The length and elements do not change. On failure the capacity does not
/// change either.
///
/// # Errors
///
/// - `AllocationFailure` (naming `quantity`) if the memory cannot be reserved.
///
/// ```
/// use scioxide_core::backend::try_reserve;
/// let mut values = vec![1.0_f64];
/// try_reserve(&mut values, 4, scioxide_core::quantity!("scratch"))?;
/// assert_eq!(values, [1.0]);
/// assert!(values.capacity() >= 5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn try_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| SciOxideError::allocation_failure(quantity, additional))
}

/// Creates a vector of `len` clones of `value`.
///
/// The memory is reserved before cloning. A panic or allocation inside
/// `T::clone` is not caught.
///
/// # Errors
///
/// - `AllocationFailure` (naming `quantity`) if the memory cannot be reserved.
///
/// ```
/// use scioxide_core::backend::try_filled;
/// assert_eq!(try_filled(3, 0.0_f64, scioxide_core::quantity!("scratch"))?, [0.0; 3]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn try_filled<T: Clone>(
    len: usize,
    value: T,
    quantity: Quantity,
) -> Result<Vec<T>, SciOxideError> {
    let mut values = try_with_capacity(len, quantity)?;
    values.resize(len, value);
    Ok(values)
}

/// Copies a slice into a new vector.
///
/// Element bits are copied unchanged, including signed zeros and NaN; no
/// numerical check is made.
///
/// # Errors
///
/// - `AllocationFailure` (naming `quantity`) if the memory cannot be reserved.
///
/// ```
/// use scioxide_core::backend::try_copy_slice;
/// let copy = try_copy_slice(&[1.0_f64, 2.0], scioxide_core::quantity!("scratch"))?;
/// assert_eq!(copy, [1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn try_copy_slice<T: Copy>(source: &[T], quantity: Quantity) -> Result<Vec<T>, SciOxideError> {
    let mut values = try_with_capacity(source.len(), quantity)?;
    values.extend_from_slice(source);
    Ok(values)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn checked_arithmetic_accepts_boundaries() {
        assert!(matches!(checked_add(2, 3, crate::quantity!("test")), Ok(5)));
        assert!(matches!(
            checked_mul(0, usize::MAX, crate::quantity!("test")),
            Ok(0)
        ));
        assert!(matches!(
            checked_product(&[3, 0, usize::MAX], crate::quantity!("test")),
            Ok(0)
        ));
        assert!(matches!(
            checked_product(&[usize::MAX, 0], crate::quantity!("test")),
            Ok(0)
        ));
        assert!(matches!(
            checked_product(&[], crate::quantity!("test")),
            Ok(1)
        ));
        assert!(shape_is_empty(&[usize::MAX, 0]));
        assert!(!shape_is_empty(&[]));
        assert!(matches!(
            checked_next_power_of_two(0, crate::quantity!("test")),
            Ok(1)
        ));
        assert!(matches!(
            checked_next_power_of_two(5, crate::quantity!("test")),
            Ok(8)
        ));
    }

    #[test]
    fn checked_arithmetic_reports_structural_overflow() {
        for (result, quantity) in [
            (
                checked_add(usize::MAX, 1, crate::quantity!("sum")),
                crate::quantity!("sum"),
            ),
            (
                checked_mul(usize::MAX / 2 + 1, 2, crate::quantity!("product")),
                crate::quantity!("product"),
            ),
            (
                checked_product(&[usize::MAX / 2 + 1, 2], crate::quantity!("shape")),
                crate::quantity!("shape"),
            ),
            (
                checked_next_power_of_two(usize::MAX, crate::quantity!("power")),
                crate::quantity!("power"),
            ),
        ] {
            let error = result.unwrap_err();
            let crate::SciOxideErrorView::SizeOverflow {
                context: _view_context,
            } = error.view()
            else {
                unreachable!("unexpected error kind")
            };

            assert_eq!(_view_context, quantity);
        }
    }

    #[test]
    fn fallible_allocation_reports_failure() {
        let result = try_with_capacity::<u8>(usize::MAX, crate::quantity!("test_allocation"));
        let error = result.unwrap_err();
        let crate::SciOxideErrorView::AllocationFailure {
            context: _view_context,
            requested: _view_requested,
        } = error.view()
        else {
            unreachable!("unexpected error kind")
        };

        assert_eq!(_view_context, crate::quantity!("test_allocation"));
        assert_eq!(_view_requested, usize::MAX);
    }

    #[test]
    fn fallible_filled_vector_supports_zero_length() {
        let values = try_filled::<u8>(0, 1, crate::quantity!("test"));
        assert!(matches!(values, Ok(values) if values.is_empty()));
    }
}
