//! Owned dense vector with contiguous storage.

use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_reserve};
use scioxide_core::{NumericScalar, SciOxideError};

use super::view::{VectorView, VectorViewMut};

/// Owned dense vector of [`NumericScalar`] elements in contiguous storage.
///
/// Element `i` is `as_slice()[i]`. Arithmetic takes a [`VectorView`], which you
/// get from [`view`](Self::view). See [Matrices and views](crate#matrices-and-views).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Vector;
///
/// let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(v.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct Vector<S: NumericScalar> {
    data: Vec<S>,
}

impl<S: NumericScalar> Vector<S> {
    /// Returns the heap bytes held for elements, counting the full capacity of the storage.
    ///
    /// Capacity left over from an earlier longer vector is included. The struct
    /// itself and allocator headers are not.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        scioxide_core::backend::checked_mul(
            self.data.capacity(),
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("retained_bytes"),
        )
    }

    /// Returns the elements as a `Vec<S>` without copying.
    #[must_use]
    pub fn into_vec(self) -> Vec<S> {
        self.data
    }

    /// Returns the vector of length 0 without reserving storage.
    pub const fn empty() -> Self {
        Self { data: Vec::new() }
    }

    /// Returns a vector that takes ownership of `data` without copying or checking it.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Vector;
    ///
    /// let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(v.len(), 3);
    /// ```
    pub const fn from_vec(data: Vec<S>) -> Self {
        Self { data }
    }

    /// Returns a vector of `len` zeros.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Vector;
    ///
    /// let v = Vector::<f64>::zeros(4).unwrap();
    /// assert_eq!(v.len(), 4);
    /// ```
    pub fn zeros(len: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            data: try_filled(
                len,
                S::ZERO,
                scioxide_core::quantity!("dense_vector_storage"),
            )?,
        })
    }

    /// Returns the number of elements.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the vector has no elements.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Borrows the vector as a [`VectorView`] with stride 1.
    pub fn view(&self) -> VectorView<'_, S> {
        VectorView::from_slice(&self.data)
    }

    /// Borrows the vector as a [`VectorViewMut`] with stride 1.
    pub fn view_mut(&mut self) -> VectorViewMut<'_, S> {
        VectorViewMut::from_slice(&mut self.data)
    }

    /// Returns the elements as a slice.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }

    /// Returns the elements as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }

    /// Resizes to `len` elements, filling new elements with `value`.
    ///
    /// The first `min(old, len)` elements keep their values. Shrinking keeps
    /// the capacity.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if additional storage cannot be reserved. The
    ///   vector is then unchanged.
    pub fn try_resize(&mut self, len: usize, value: S) -> Result<(), SciOxideError> {
        if len > self.data.len() {
            let additional = len - self.data.len();
            try_reserve(
                &mut self.data,
                additional,
                scioxide_core::quantity!("dense_vector_resize"),
            )?;
        }
        self.data.resize(len, value);
        Ok(())
    }
    pub(crate) fn capacity(&self) -> usize {
        self.data.capacity()
    }
}

impl<T: NumericScalar + scioxide_core::AffineValueMut> scioxide_core::AffineValue for Vector<T> {
    type Scale = <T as scioxide_core::AffineValue>::Scale;
    type Owned = Self;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        self.as_slice().dimension()
    }
    fn component(&self, index: usize) -> Result<Self::Scale, SciOxideError> {
        self.as_slice().component(index)
    }
    fn try_to_owned(&self) -> Result<Self, SciOxideError> {
        Ok(Self::from_vec(scioxide_core::backend::try_copy_slice(
            self.as_slice(),
            scioxide_core::quantity!("affine_dense_vector"),
        )?))
    }
}
impl<T: NumericScalar + scioxide_core::AffineValueMut> scioxide_core::AffineValueMut for Vector<T> {
    fn for_each_component_mut(&mut self, visit: impl FnMut(&mut Self::Scale)) {
        self.as_mut_slice().for_each_component_mut(visit);
    }
}

impl<S: NumericScalar> crate::workspace::accounting::StorageBytes for Vector<S> {
    fn storage_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        self.data.storage_bytes()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    // ---- Construction ----

    #[test]
    fn test_from_vec_f64() {
        let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(!v.is_empty());
    }

    #[test]
    fn test_from_vec_f32() {
        let v = Vector::<f32>::from_vec(vec![1.0, 2.0]);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_zeros_f64() {
        let v = Vector::<f64>::zeros(5).unwrap();
        assert_eq!(v.len(), 5);
        assert!(v.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_zeros_reports_allocation_failure() {
        assert!(
            matches!(Vector::<f64>::zeros(usize::MAX), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::AllocationFailure { context, requested, .. } if context == scioxide_core::quantity!("dense_vector_storage") && requested == usize::MAX))
        );
    }

    #[test]
    fn test_empty_vector() {
        let v = Vector::<f64>::from_vec(vec![]);
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    // ---- Element access ----

    #[test]
    fn test_get_valid() {
        let v = Vector::<f64>::from_vec(vec![10.0, 20.0, 30.0]);
        assert!((v.view().get(0).unwrap() - 10.0).abs() < f64::EPSILON);
        assert!((v.view().get(2).unwrap() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let v = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (v.view().get(5)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds {
                index: 5,
                size: 2,
                ..
            }
        ));
    }

    #[test]
    fn test_set_valid() {
        let mut v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(v.view_mut().set(1, 99.0).is_ok());
        assert!((v.view().get(1).unwrap() - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut v = Vector::<f64>::from_vec(vec![1.0]);
        assert!(matches!(
            (v.view_mut().set(3, 0.0)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds {
                index: 3,
                size: 1,
                ..
            }
        ));
    }

    // ---- Dot product ----

    #[test]
    fn test_dot_f64() {
        let a = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::<f64>::from_vec(vec![4.0, 5.0, 6.0]);
        let d = a
            .view()
            .dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((d - 32.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_f32() {
        let a = Vector::<f32>::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::<f32>::from_vec(vec![4.0, 5.0, 6.0]);
        let d = a
            .view()
            .dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((d - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dot_compensates_f64_cancellation() {
        let a = Vector::<f64>::from_vec(vec![1e16, 1.0, -1e16]);
        let b = Vector::<f64>::from_vec(vec![1.0; 3]);
        let d = a
            .view()
            .dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new().with_accuracy(scioxide_core::Accuracy::Exact),
                crate::LinalgExecutionContext::unscoped(crate::KernelContext::sequential()),
            )
            .unwrap();
        assert!((d - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_compensates_f32_cancellation() {
        let a = Vector::<f32>::from_vec(vec![1e8, 1.0, -1e8]);
        let b = Vector::<f32>::from_vec(vec![1.0; 3]);
        let d = a
            .view()
            .dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new().with_accuracy(scioxide_core::Accuracy::Exact),
                crate::LinalgExecutionContext::unscoped(crate::KernelContext::sequential()),
            )
            .unwrap();
        assert!((d - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dot_complex_conjugate() {
        let a = Vector::from_vec(vec![Complex::new(1.0_f64, 2.0), Complex::new(3.0, 4.0)]);
        let b = Vector::from_vec(vec![Complex::new(5.0_f64, 6.0), Complex::new(7.0, 8.0)]);
        // dot = a[0]*conj(b[0]) + a[1]*conj(b[1])
        // = (1+2i)*(5-6i) + (3+4i)*(7-8i)
        // = (5+12+(-6+10)i) + (21+32+(-24+28)i)
        // = 17+4i + 53+4i = 70+8i
        let d = crate::VectorView::from(&b)
            .dotc_left_scoped(
                a.view(),
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((d.re - 70.0).abs() < 1e-12);
        assert!((d.im - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_dot_complex_compensates_both_components() {
        let a = Vector::from_vec(vec![
            Complex::new(1e16_f64, -1e16),
            Complex::new(1.0, 2.0),
            Complex::new(-1e16, 1e16),
        ]);
        let b = Vector::from_vec(vec![Complex::new(1.0_f64, 0.0); 3]);
        let d = crate::VectorView::from(&b)
            .dotc_left_scoped(
                a.view(),
                &crate::LinalgExecutionOptions::new().with_accuracy(scioxide_core::Accuracy::Exact),
                crate::LinalgExecutionContext::unscoped(crate::KernelContext::sequential()),
            )
            .unwrap();
        assert!((d.re - 1.0).abs() < f64::EPSILON);
        assert!((d.im - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_complex_f32_compensates_both_components() {
        let a = Vector::from_vec(vec![
            Complex::new(1e8_f32, -1e8),
            Complex::new(1.0, 2.0),
            Complex::new(-1e8, 1e8),
        ]);
        let b = Vector::from_vec(vec![Complex::new(1.0_f32, 0.0); 3]);
        let d = crate::VectorView::from(&b)
            .dotc_left_scoped(
                a.view(),
                &crate::LinalgExecutionOptions::new().with_accuracy(scioxide_core::Accuracy::Exact),
                crate::LinalgExecutionContext::unscoped(crate::KernelContext::sequential()),
            )
            .unwrap();
        assert!((d.re - 1.0).abs() < f32::EPSILON);
        assert!((d.im - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dot_dimension_mismatch() {
        let a = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            (a.view().dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_dot_dimension_mismatch_precedes_value_validation() {
        let a = Vector::<f64>::from_vec(vec![f64::NAN]);
        let b = Vector::<f64>::empty();
        assert!(matches!(
            (a.view().dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_dot_rejects_non_finite_input() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let a = Vector::<f64>::from_vec(vec![value]);
            let b = Vector::<f64>::from_vec(vec![1.0]);
            assert!(matches!(
                (a.view().dotu_scoped(
                    &b,
                    &crate::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                ))
                .as_ref()
                .err()
                .unwrap()
                .view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_dot_reports_finite_product_overflow() {
        let a = Vector::<f64>::from_vec(vec![f64::MAX]);
        let b = Vector::<f64>::from_vec(vec![2.0]);
        assert!(matches!(
            (a.view().dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn test_dot_empty_vectors() {
        let a = Vector::<f64>::from_vec(vec![]);
        let b = Vector::<f64>::from_vec(vec![]);
        let d = a
            .view()
            .dotu_scoped(
                &b,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((d - 0.0).abs() < f64::EPSILON);
    }

    // ---- Norm ----

    #[test]
    fn test_norm2_f64() {
        let v = Vector::<f64>::from_vec(vec![3.0, 4.0]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm2_f32() {
        let v = Vector::<f32>::from_vec(vec![3.0, 4.0]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm2_mixed_magnitudes() {
        let v = Vector::<f64>::from_vec(vec![3e200, 4e200, 1e-200]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n / 5e200 - 1.0).abs() < 4.0 * f64::EPSILON);
    }

    #[test]
    fn test_norm2_preserves_f64_subnormal_values() {
        let value = f64::MIN_POSITIVE / 4.0;
        let expected = value.hypot(value);
        let v = Vector::<f64>::from_vec(vec![value, value]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(n > 0.0);
        assert!((n - expected).abs() <= f64::from_bits(1));
    }

    #[test]
    fn test_norm2_preserves_f32_subnormal_values() {
        let value = f32::MIN_POSITIVE / 4.0;
        let expected = value.hypot(value);
        let v = Vector::<f32>::from_vec(vec![value, value]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(n > 0.0);
        assert!((n - expected).abs() <= f32::from_bits(1));
    }

    #[test]
    fn test_norm2_near_overflow_f64() {
        let value = f64::MAX / 2.0;
        let expected = value.hypot(value);
        let v = Vector::<f64>::from_vec(vec![value, value]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(n.is_finite());
        assert!((n / expected - 1.0).abs() < 4.0 * f64::EPSILON);
    }

    #[test]
    fn test_norm2_near_overflow_f32() {
        let value = f32::MAX / 2.0;
        let expected = value.hypot(value);
        let v = Vector::<f32>::from_vec(vec![value, value]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(n.is_finite());
        assert!((n / expected - 1.0).abs() < 4.0 * f32::EPSILON);
    }

    #[test]
    fn test_norm2_zero_vector() {
        let v = Vector::<f64>::zeros(3).unwrap();
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_norm2_empty() {
        let v = Vector::<f64>::from_vec(vec![]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_norm2_complex() {
        let v = Vector::from_vec(vec![Complex::new(3.0_f64, 4.0)]);
        // |3+4i| = 5, norm = 5
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm2_complex_avoids_component_overflow() {
        let component = f64::MAX / 4.0;
        let element_norm = component.hypot(component);
        let expected = element_norm.hypot(element_norm);
        let v = Vector::from_vec(vec![
            Complex::new(component, component),
            Complex::new(component, -component),
        ]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(n.is_finite());
        assert!((n / expected - 1.0).abs() < 4.0 * f64::EPSILON);
    }

    #[test]
    fn test_norm2_complex_f32_avoids_component_overflow() {
        let component = f32::MAX / 4.0;
        let element_norm = component.hypot(component);
        let expected = element_norm.hypot(element_norm);
        let v = Vector::from_vec(vec![
            Complex::new(component, component),
            Complex::new(component, -component),
        ]);
        let n = v
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(n.is_finite());
        assert!((n / expected - 1.0).abs() < 4.0 * f32::EPSILON);
    }

    #[test]
    fn test_norm2_rejects_non_finite_input() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let v = Vector::<f64>::from_vec(vec![value]);
            assert!(matches!(
                (v.view().norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                ))
                .as_ref()
                .err()
                .unwrap()
                .view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_norm2_reports_unrepresentable_result() {
        let v = Vector::<f64>::from_vec(vec![f64::MAX, f64::MAX]);
        assert!(matches!(
            (v.view().norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }

    // ---- Normalize ----

    #[test]
    fn test_normalize_f64() {
        let v = Vector::<f64>::from_vec(vec![3.0, 4.0]);
        let u = v.view().normalize().unwrap();
        let n = u
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((n - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = Vector::<f64>::zeros(3).unwrap();
        assert!(matches!(
            (v.view().normalize()).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Undefined,
                ..
            }
        ));
    }

    #[test]
    fn test_normalize_smallest_subnormal_value() {
        let v = Vector::<f64>::from_vec(vec![f64::from_bits(1)]);
        let normalized = v.view().normalize().unwrap();
        assert!((normalized.view().get(0).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!(
            (normalized
                .view()
                .norm2_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap()
                - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_normalize_empty() {
        let v = Vector::<f64>::from_vec(vec![]);
        assert!(matches!(
            (v.view().normalize()).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Undefined,
                ..
            }
        ));
    }

    // ---- Cross product ----

    #[test]
    fn test_cross_f64() {
        let a = Vector::<f64>::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Vector::<f64>::from_vec(vec![0.0, 1.0, 0.0]);
        let c = a.view().cross(&b).unwrap();
        assert!((c.view().get(0).unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((c.view().get(1).unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((c.view().get(2).unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cross_anticommutativity() {
        let a = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::<f64>::from_vec(vec![4.0, 5.0, 6.0]);
        let ab = a.view().cross(&b).unwrap();
        let ba = b.view().cross(&a).unwrap();
        let sum = ab.view().try_add(&ba).unwrap();
        for &s in sum.as_slice() {
            assert!(s.abs() < 1e-14);
        }
    }

    #[test]
    fn test_cross_dimension_mismatch() {
        let a = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let b = Vector::<f64>::from_vec(vec![3.0, 4.0, 5.0]);
        assert!(matches!(
            (a.view().cross(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_cross_self_is_zero() {
        let a = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let c = a.view().cross(&a).unwrap();
        for &x in c.as_slice() {
            assert!(x.abs() < 1e-14);
        }
    }

    // ---- Slice access ----

    #[test]
    fn test_as_slice() {
        let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let s = v.as_slice();
        assert_eq!(s.len(), 3);
        assert!((s[0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut v = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        v.as_mut_slice()[0] = 99.0;
        assert!((v.view().get(0).unwrap() - 99.0).abs() < f64::EPSILON);
    }

    // ---- Iterator ----

    #[test]
    fn test_iter() {
        let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let sum: f64 = v.view().iter().fold(0.0, |a, b| a + b);
        assert!((sum - 6.0).abs() < f64::EPSILON);
    }

    // ---- Arithmetic ops ----

    #[test]
    fn test_in_place_blas_primitives() {
        let x = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = Vector::<f64>::from_vec(vec![4.0, 5.0, 6.0]);

        crate::VectorView::from(&x)
            .copy_into(
                crate::CopyPolicy::Preserve,
                &crate::LinalgExecutionOptions::new(),
                (y.view_mut()).reborrow(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(y.as_slice(), &[1.0, 2.0, 3.0]);
        crate::VectorViewMut::scale_in_place(
            2.0,
            &crate::LinalgExecutionOptions::new(),
            y.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(y.as_slice(), &[2.0, 4.0, 6.0]);
        crate::VectorViewMut::axpy_in_place(
            -1.0,
            &x,
            &crate::LinalgExecutionOptions::new(),
            y.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(y.as_slice(), &[1.0, 2.0, 3.0]);
        crate::VectorViewMut::axpby_in_place(
            3.0,
            &x,
            -1.0,
            &crate::LinalgExecutionOptions::new(),
            y.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(y.as_slice(), &[2.0, 4.0, 6.0]);
        y.view_mut().fill(0.5);
        assert_eq!(y.as_slice(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_in_place_dimension_errors_leave_destination_unchanged() {
        let x = Vector::<f32>::from_vec(vec![1.0, 2.0]);
        let mut y = Vector::<f32>::from_vec(vec![7.0, 8.0, 9.0]);

        assert!(matches!(
            (crate::VectorView::from(&x).copy_into(
                crate::CopyPolicy::Preserve,
                &crate::LinalgExecutionOptions::new(),
                (y.view_mut()).reborrow(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(y.as_slice(), &[7.0, 8.0, 9.0]);
        assert!(matches!(
            (crate::VectorViewMut::axpy_in_place(
                1.0,
                &x,
                &crate::LinalgExecutionOptions::new(),
                y.view_mut(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(y.as_slice(), &[7.0, 8.0, 9.0]);
        assert!(matches!(
            (crate::VectorViewMut::axpby_in_place(
                1.0,
                &x,
                1.0,
                &crate::LinalgExecutionOptions::new(),
                y.view_mut(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(y.as_slice(), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_try_add_f64() {
        let a = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let b = Vector::<f64>::from_vec(vec![3.0, 4.0]);
        let c = a.view().try_add(&b).unwrap();
        assert!((c.view().get(0).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((c.view().get(1).unwrap() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_try_add_dimension_mismatch() {
        let a = Vector::<f64>::from_vec(vec![1.0]);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (a.view().try_add(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_try_sub_f64() {
        let a = Vector::<f64>::from_vec(vec![5.0, 3.0]);
        let b = Vector::<f64>::from_vec(vec![1.0, 1.0]);
        let c = a.view().try_sub(&b).unwrap();
        assert!((c.view().get(0).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((c.view().get(1).unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_try_sub_dimension_mismatch() {
        let a = Vector::<f64>::from_vec(vec![1.0]);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (a.view().try_sub(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_neg_f64() {
        let v = Vector::<f64>::from_vec(vec![1.0, -2.0, 3.0]);
        let neg = v.view().try_neg().unwrap();
        assert!((neg.view().get(0).unwrap() - (-1.0)).abs() < f64::EPSILON);
        assert!((neg.view().get(1).unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((neg.view().get(2).unwrap() - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalar_mul_f64() {
        let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = v.view().try_scale(2.0).unwrap();
        assert!((scaled.view().get(0).unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((scaled.view().get(1).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((scaled.view().get(2).unwrap() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalar_mul_complex() {
        let v = Vector::from_vec(vec![Complex::new(1.0_f64, 0.0), Complex::new(0.0, 1.0)]);
        let s = Complex::new(0.0, 1.0);
        let result = v.view().try_scale(s).unwrap();
        // (1+0i)*(0+i) = 0+i
        assert!((result.view().get(0).unwrap().re - 0.0).abs() < 1e-14);
        assert!((result.view().get(0).unwrap().im - 1.0).abs() < 1e-14);
        // (0+i)*(0+i) = -1+0i
        assert!((result.view().get(1).unwrap().re - (-1.0)).abs() < 1e-14);
        assert!((result.view().get(1).unwrap().im - 0.0).abs() < 1e-14);
    }

    #[test]
    fn get_returns_values_at_both_boundaries() {
        let v = Vector::<f64>::from_vec(vec![10.0, 20.0, 30.0]);
        assert!((v.view().get(0).unwrap() - 10.0).abs() < f64::EPSILON);
        assert!((v.view().get(1).unwrap() - 20.0).abs() < f64::EPSILON);
        assert!((v.view().get(2).unwrap() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn get_rejects_out_of_bounds_indices() {
        let v = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        for index in [v.len(), 5, usize::MAX] {
            assert!(matches!(
                v.view().get(index).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::OutOfBounds { .. }
            ));
        }
    }
}
