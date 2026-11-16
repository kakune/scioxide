//! Owned dense matrix with contiguous row-major storage.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_product, try_filled, try_reserve};
use scioxide_core::{NumericScalar, SciOxideError};
use scioxide_sparse::{CscMatrix, CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

use super::view::{MatrixView, MatrixViewMut};

/// Owned dense `rows × cols` matrix of [`NumericScalar`] elements in row-major order.
///
/// Element `(i, j)` is stored at index `i * cols + j` of [`as_slice`](Self::as_slice).
/// Arithmetic and decompositions take a [`MatrixView`], which you get from
/// [`view`](Self::view). See [Matrices and views](crate#matrices-and-views).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
///
/// let m = Matrix::<f64>::identity(3).unwrap();
/// assert!(m.is_square());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct Matrix<S: NumericScalar> {
    rows: usize,
    cols: usize,
    data: Vec<S>,
}

impl<S: NumericScalar> Matrix<S> {
    /// Returns the heap bytes held for elements, counting the full capacity of the storage.
    ///
    /// Capacity left over from an earlier larger shape is included. The struct
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

    /// Returns the `0 × 0` matrix without reserving storage.
    pub const fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }

    /// Converts a CSR sparse matrix to a dense matrix, filling absent entries with zero.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if the dense storage cannot be reserved.
    pub fn from_csr(matrix: &CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let mut dense = Self::zeros(matrix.rows(), matrix.cols())?;
        matrix.to_dense_into(dense.as_mut_slice())?;
        Ok(dense)
    }

    /// Converts a CSC sparse matrix to a dense matrix, filling absent entries with zero.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if the dense storage cannot be reserved.
    pub fn from_csc(matrix: &CscMatrix<S>) -> Result<Self, SciOxideError> {
        let mut dense = Self::zeros(matrix.rows(), matrix.cols())?;
        matrix.to_dense_into(dense.as_mut_slice())?;
        Ok(dense)
    }

    /// Converts this matrix to CSR sparse storage.
    ///
    /// Every element is a candidate entry. `zero_policy` decides whether exact
    /// zeros are stored ([`ExplicitZeroPolicy::Preserve`]) or dropped
    /// ([`ExplicitZeroPolicy::Drop`]).
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the sparse storage cannot be sized or reserved.
    pub fn to_sparse_csr(
        &self,
        zero_policy: ExplicitZeroPolicy,
    ) -> Result<CsrMatrix<S>, SciOxideError> {
        let mut builder = SparseBuilder::with_capacity(self.rows, self.cols, self.data.len())?;
        self.data
            .chunks(self.cols.max(1))
            .enumerate()
            .try_for_each(|(row, values)| {
                values
                    .iter()
                    .copied()
                    .enumerate()
                    .try_for_each(|(col, value)| builder.push(row, col, value))
            })?;
        builder.finish_csr(DuplicatePolicy::Reject, zero_policy)
    }

    /// Returns a `rows × cols` matrix of zeros.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    ///
    /// let m = Matrix::<f64>::zeros(2, 3).unwrap();
    /// assert_eq!(m.rows(), 2);
    /// assert_eq!(m.cols(), 3);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, SciOxideError> {
        let len = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("dense_matrix_element_count"),
        )?;
        Ok(Self {
            rows,
            cols,
            data: try_filled(
                len,
                S::ZERO,
                scioxide_core::quantity!("dense_matrix_storage"),
            )?,
        })
    }

    /// Returns the matrix whose `i`-th row is `rows[i]`.
    ///
    /// The result has `rows.len()` rows and `C` columns; an empty slice gives a
    /// `0 × C` matrix. The entries are copied once and not checked for
    /// finiteness.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    ///
    /// let m = Matrix::from_rows(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    /// assert_eq!((m.rows(), m.cols()), (2, 3));
    /// assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_rows<const C: usize>(rows: &[[S; C]]) -> Result<Self, SciOxideError> {
        Self::from_vec(
            rows.len(),
            C,
            scioxide_core::backend::try_copy_slice(
                rows.as_flattened(),
                scioxide_core::quantity!("dense_matrix_storage"),
            )?,
        )
    }

    /// Returns a `rows × cols` matrix that takes ownership of `data` in row-major order.
    ///
    /// Element `(i, j)` is `data[i * cols + j]`. No element is copied or checked
    /// for finiteness.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `DimensionMismatch` if `data.len() != rows * cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    ///
    /// let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// ```
    pub fn from_vec(rows: usize, cols: usize, data: Vec<S>) -> Result<Self, SciOxideError> {
        let expected = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("dense_matrix_element_count"),
        )?;
        if data.len() != expected {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(expected),
                scioxide_core::Dimensions::one(data.len()),
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// Returns the `n × n` identity matrix.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `n * n` does not fit in `usize`.
    /// - `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    ///
    /// let eye = Matrix::<f64>::identity(3).unwrap();
    /// assert_eq!(eye.rows(), 3);
    /// ```
    pub fn identity(n: usize) -> Result<Self, SciOxideError> {
        let len = checked_product(
            &[n, n],
            scioxide_core::quantity!("identity_matrix_element_count"),
        )?;
        let mut data = try_filled(
            len,
            S::ZERO,
            scioxide_core::quantity!("identity_matrix_storage"),
        )?;
        for i in 0..n {
            data[i * n + i] = S::ONE;
        }
        Ok(Self {
            rows: n,
            cols: n,
            data,
        })
    }

    /// Returns the number of rows.
    #[must_use]
    #[inline]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[must_use]
    #[inline]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns `true` if `rows == cols`.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Borrows the matrix as a [`MatrixView`] with row stride `cols` and column stride 1.
    pub fn view(&self) -> MatrixView<'_, S> {
        MatrixView {
            data: &self.data,
            rows: self.rows,
            cols: self.cols,
            row_stride: self.cols,
            col_stride: 1,
            conjugated: false,
        }
    }

    /// Borrows the matrix as a [`MatrixViewMut`] with row stride `cols` and column stride 1.
    pub fn view_mut(&mut self) -> MatrixViewMut<'_, S> {
        MatrixViewMut {
            data: &mut self.data,
            rows: self.rows,
            cols: self.cols,
            row_stride: self.cols,
            col_stride: 1,
        }
    }

    /// Returns the elements as a row-major slice of length `rows * cols`.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }

    /// Reinterprets the row-major storage as a `rows × cols` matrix without moving elements.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `DimensionMismatch` if `rows * cols` differs from the current element count.
    ///
    /// The matrix is unchanged on error.
    pub fn try_reshape(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        let len = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("dense_matrix_reshape_element_count"),
        )?;
        if len != self.data.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.data.len()),
                scioxide_core::Dimensions::one(len),
            ));
        }
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }

    /// Resizes to `rows × cols` by truncating or zero-extending the row-major storage.
    ///
    /// The first `min(old, new)` elements of [`as_slice`](Self::as_slice) keep
    /// their values, so an element generally moves to a different `(i, j)` when
    /// the column count changes. Use
    /// [`try_resize_preserving_coordinates`](Self::try_resize_preserving_coordinates)
    /// to keep values at their coordinates.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if additional storage cannot be reserved.
    ///
    /// The matrix is unchanged on error.
    #[inline(always)]
    pub fn try_resize_flat(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        // Workspace-oriented resize: the flat storage prefix is kept but the
        // (row, col) meaning of old elements is not. Growth is reserved
        // fallibly first so an allocation failure leaves the shape unchanged.
        let len = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("dense_matrix_flat_resize_element_count"),
        )?;
        if len > self.data.len() {
            let additional = len - self.data.len();
            try_reserve(
                &mut self.data,
                additional,
                scioxide_core::quantity!("dense_matrix_flat_resize"),
            )?;
        }
        // Monomorphize the zero producer with this matrix instead of calling a
        // shared Vec::resize with an arbitrary runtime fill value.
        self.data.resize_with(len, || S::ZERO);
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }

    /// Resizes to `rows × cols`, keeping the value at every `(i, j)` inside both shapes.
    ///
    /// New elements are zero. Unless the shape is unchanged, this builds a new
    /// buffer of `rows * cols` elements.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if the new storage cannot be reserved.
    ///
    /// The matrix is unchanged on error.
    pub fn try_resize_preserving_coordinates(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<(), SciOxideError> {
        if self.rows == rows && self.cols == cols {
            return Ok(());
        }
        let len = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("coordinate_preserving_matrix_resize_element_count"),
        )?;
        // Row starts move when the column count changes, so an in-place
        // resize would scramble coordinates; a fresh buffer is built and only
        // swapped in once allocation has succeeded.
        let mut candidate = try_filled(
            len,
            S::ZERO,
            scioxide_core::quantity!("coordinate_preserving_matrix_resize"),
        )?;
        let preserved_rows = self.rows.min(rows);
        let preserved_cols = self.cols.min(cols);
        for row in 0..preserved_rows {
            let old_start = row * self.cols;
            let new_start = row * cols;
            candidate[new_start..new_start + preserved_cols]
                .copy_from_slice(&self.data[old_start..old_start + preserved_cols]);
        }
        self.data = candidate;
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }

    // Reserve storage for a `rows x cols` shape.
    pub(crate) fn try_reserve_shape(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<(), SciOxideError> {
        let required = checked_product(&[rows, cols], scioxide_core::quantity!("matrix.elements"))?;
        if required > self.data.capacity() {
            let additional = required.saturating_sub(self.data.len());
            try_reserve(
                &mut self.data,
                additional,
                scioxide_core::quantity!("matrix.elements"),
            )?;
        }
        Ok(())
    }

    // Copy `source` into reused storage.
    pub(crate) fn try_copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.try_reserve_shape(source.rows, source.cols)?;
        self.data.clear();
        self.data.extend_from_slice(&source.data);
        self.rows = source.rows;
        self.cols = source.cols;
        Ok(())
    }

    /// Allocated element capacity, used to size factorization byte limits.
    pub(crate) fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns the elements as a mutable row-major slice of length `rows * cols`.
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }

    /// Returns element `(i, j)`; the caller has checked the bounds.
    #[must_use]
    pub(crate) fn get_element(&self, i: usize, j: usize) -> S {
        debug_assert!(i < self.rows);
        debug_assert!(j < self.cols);
        self.data[i * self.cols + j]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Vector;
    use scioxide_core::Complex;
    use scioxide_core::{AdjointOperator, LinearOperator};

    // ---- Construction ----

    #[test]
    fn test_new_zeros() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert!(m.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_constructors_report_structural_size_errors() {
        assert!(
            matches!(Matrix::<f64>::zeros(usize::MAX, 2), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow { context, .. } if context == scioxide_core::quantity!("dense_matrix_element_count")))
        );
        assert!(
            matches!(Matrix::<f64>::from_vec(usize::MAX, 2, Vec::new()), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow { context, .. } if context == scioxide_core::quantity!("dense_matrix_element_count")))
        );
        assert!(
            matches!(Matrix::<f64>::identity(usize::MAX), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow { context, .. } if context == scioxide_core::quantity!("identity_matrix_element_count")))
        );
    }

    #[test]
    fn test_zero_dimension_shape_short_circuits_product() {
        for (rows, cols) in [(0, usize::MAX), (usize::MAX, 0)] {
            let matrix = Matrix::<f64>::zeros(rows, cols).unwrap();
            assert_eq!(matrix.rows(), rows);
            assert_eq!(matrix.cols(), cols);
            assert!(matrix.as_slice().is_empty());

            let from_vec = Matrix::<f64>::from_vec(rows, cols, Vec::new()).unwrap();
            assert_eq!(from_vec.rows(), rows);
            assert_eq!(from_vec.cols(), cols);
            assert!(from_vec.as_slice().is_empty());
        }
    }

    #[test]
    fn test_from_vec_valid() {
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert!((m.view().get(0, 0).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((m.view().get(1, 1).unwrap() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_vec_dimension_mismatch() {
        let r = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            (r).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_identity() {
        let eye = Matrix::<f64>::identity(3).unwrap();
        assert!(eye.is_square());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((eye.view().get(i, j).unwrap() - expected).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_identity_f32() {
        let eye = Matrix::<f32>::identity(2).unwrap();
        assert!((eye.view().get(0, 0).unwrap() - 1.0).abs() < f32::EPSILON);
        assert!((eye.view().get(0, 1).unwrap() - 0.0).abs() < f32::EPSILON);
    }

    // ---- Element access ----

    #[test]
    fn test_get_out_of_bounds_row() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (m.view().get(5, 0)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds {
                index: 5,
                size: 2,
                ..
            }
        ));
    }

    #[test]
    fn test_get_out_of_bounds_col() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (m.view().get(0, 5)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds {
                index: 5,
                size: 3,
                ..
            }
        ));
    }

    #[test]
    fn test_set_and_get() {
        let mut m = Matrix::<f64>::zeros(2, 2).unwrap();
        assert!(m.view_mut().set(0, 1, 42.0).is_ok());
        assert!((m.view().get(0, 1).unwrap() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_out_of_bounds() {
        let mut m = Matrix::<f64>::zeros(2, 2).unwrap();
        assert!(matches!(
            (m.view_mut().set(3, 0, 1.0)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds { .. }
        ));
    }

    // ---- Transpose ----

    #[test]
    fn test_transpose() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.view().transpose().try_to_matrix().unwrap();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert!((t.view().get(0, 0).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((t.view().get(0, 1).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((t.view().get(2, 1).unwrap() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transpose_transpose_roundtrip() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tt = m
            .view()
            .transpose()
            .try_to_matrix()
            .unwrap()
            .view()
            .transpose()
            .try_to_matrix()
            .unwrap();
        assert_eq!(m, tt);
    }

    #[test]
    fn test_conjugate_transpose_real() {
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            m.view().adjoint().try_to_matrix().unwrap(),
            m.view().transpose().try_to_matrix().unwrap()
        );
    }

    #[test]
    fn test_conjugate_transpose_complex() {
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Complex::new(1.0_f64, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 6.0),
                Complex::new(7.0, 8.0),
            ],
        )
        .unwrap();
        let ct = m.view().adjoint().try_to_matrix().unwrap();
        // (0,0) -> conj(1+2i) = 1-2i
        assert!((ct.view().get(0, 0).unwrap().re - 1.0).abs() < 1e-14);
        assert!((ct.view().get(0, 0).unwrap().im - (-2.0)).abs() < 1e-14);
        // (0,1) -> conj(5+6i) = 5-6i
        assert!((ct.view().get(0, 1).unwrap().re - 5.0).abs() < 1e-14);
        assert!((ct.view().get(0, 1).unwrap().im - (-6.0)).abs() < 1e-14);
    }

    // ---- Multiplication ----

    #[test]
    fn test_mul_vec_f64() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Vector::from_vec(vec![1.0, 1.0, 1.0]);
        let y = m.view().mul_vec(&x).unwrap();
        assert!((y.view().get(0).unwrap() - 6.0).abs() < f64::EPSILON);
        assert!((y.view().get(1).unwrap() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mul_vec_dimension_mismatch() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        let x = Vector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (m.view().mul_vec(&x)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_mul_mat_f64() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::<f64>::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.view().mul_mat(&b).unwrap();
        assert!((c.view().get(0, 0).unwrap() - 19.0).abs() < f64::EPSILON);
        assert!((c.view().get(0, 1).unwrap() - 22.0).abs() < f64::EPSILON);
        assert!((c.view().get(1, 0).unwrap() - 43.0).abs() < f64::EPSILON);
        assert!((c.view().get(1, 1).unwrap() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mul_mat_dimension_mismatch() {
        let a = Matrix::<f64>::zeros(2, 3).unwrap();
        let b = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (a.view().mul_mat(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_identity_mul() {
        let eye = Matrix::<f64>::identity(3).unwrap();
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let result = eye.view().mul_mat(&a).unwrap();
        assert_eq!(result, a);
    }

    // ---- Row/Col/Diag ----

    #[test]
    fn test_col() {
        let m = Matrix::<f64>::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = m.view().col(1).unwrap();
        assert!((c.get(0).unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((c.get(1).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((c.get(2).unwrap() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_col_out_of_bounds() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (m.view().col(5)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds {
                index: 5,
                size: 3,
                ..
            }
        ));
    }

    #[test]
    fn test_row() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let r = m.view().row(1).unwrap();
        assert!((r.get(0).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((r.get(1).unwrap() - 5.0).abs() < f64::EPSILON);
        assert!((r.get(2).unwrap() - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_row_out_of_bounds() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (m.view().row(3)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::OutOfBounds {
                index: 3,
                size: 2,
                ..
            }
        ));
    }

    #[test]
    fn test_set_col() {
        let mut m = Matrix::<f64>::zeros(3, 2).unwrap();
        let v = Vector::from_vec(vec![10.0, 20.0, 30.0]);
        assert!(
            m.view_mut()
                .col_mut(0)
                .and_then(|mut column| crate::VectorView::from(&v).copy_into(
                    crate::CopyPolicy::Preserve,
                    &crate::LinalgExecutionOptions::new(),
                    (column).reborrow(),
                    &mut crate::LinalgWorkspace::new(),
                    scioxide_core::ExecutionContext::sequential()
                ))
                .is_ok()
        );
        assert!((m.view().get(0, 0).unwrap() - 10.0).abs() < f64::EPSILON);
        assert!((m.view().get(1, 0).unwrap() - 20.0).abs() < f64::EPSILON);
        assert!((m.view().get(2, 0).unwrap() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_col_dimension_mismatch() {
        let mut m = Matrix::<f64>::zeros(3, 2).unwrap();
        let v = Vector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (m.view_mut()
                .col_mut(0)
                .and_then(|mut column| crate::VectorView::from(&v).copy_into(
                    crate::CopyPolicy::Preserve,
                    &crate::LinalgExecutionOptions::new(),
                    (column).reborrow(),
                    &mut crate::LinalgWorkspace::new(),
                    scioxide_core::ExecutionContext::sequential()
                )))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_diag() {
        let m = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let d = m.view().try_diag().unwrap();
        assert!((d.get(0).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((d.get(1).unwrap() - 5.0).abs() < f64::EPSILON);
        assert!((d.get(2).unwrap() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diag_rectangular() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let d = m.view().try_diag().unwrap();
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn diagonal_view_is_nonallocating_and_shape_resize_names_are_explicit() {
        let mut matrix = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            matrix.view().try_diag().unwrap().iter().collect::<Vec<_>>(),
            [1.0, 5.0]
        );

        matrix.try_reshape(3, 2).unwrap();
        assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let before = matrix.clone();
        assert!(matrix.try_reshape(4, 2).is_err());
        assert_eq!(matrix, before);

        matrix.try_resize_flat(2, 2).unwrap();
        assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        matrix.try_resize_flat(3, 3).unwrap();
        assert_eq!(
            matrix.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn coordinate_preserving_resize_keeps_overlapping_coordinates() {
        let mut matrix = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        matrix.try_resize_preserving_coordinates(3, 2).unwrap();
        assert_eq!(matrix.as_slice(), &[1.0, 2.0, 4.0, 5.0, 0.0, 0.0]);

        matrix
            .try_resize_preserving_coordinates(0, usize::MAX)
            .unwrap();
        assert_eq!((matrix.rows(), matrix.cols()), (0, usize::MAX));
        assert!(matrix.as_slice().is_empty());
    }

    // ---- Rank diagnostics ----

    #[test]
    fn test_rank_identity() {
        let m = Matrix::<f64>::identity(3).unwrap();
        assert_eq!(
            crate::SvdValues::compute(m.view())
                .unwrap()
                .analyze_rank(crate::RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .rank(),
            3
        );
    }

    #[test]
    fn test_rank_singular() {
        // Rank-1 matrix: [[1,2],[2,4]]
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(
            crate::SvdValues::compute(m.view())
                .unwrap()
                .analyze_rank(crate::RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .rank(),
            1
        );
    }

    #[test]
    fn test_rank_policy_is_explicit() {
        let m = Matrix::<f64>::identity(2).unwrap();
        assert_eq!(
            crate::SvdValues::compute(m.view())
                .unwrap()
                .analyze_rank(crate::RankPolicy::Exact)
                .unwrap()
                .rank(),
            2
        );
        let policy =
            crate::RankPolicy::Threshold(scioxide_core::RankThreshold::new(1.0, 0.0).unwrap());
        assert_eq!(
            crate::SvdValues::compute(m.view())
                .unwrap()
                .analyze_rank(policy)
                .unwrap()
                .rank(),
            0
        );
    }

    // ---- LinearOperator trait ----

    #[test]
    fn test_linear_operator_apply() {
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let op: &dyn LinearOperator<f64> = &m.view().try_finite().unwrap();
        let mut output = [0.0; 2];
        let mut workspace = [0.0; 2];
        op.apply_into(&[1.0, 1.0], &mut output, &mut workspace)
            .unwrap();
        assert!((output[0] - 3.0).abs() < f64::EPSILON);
        assert!((output[1] - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_operator_apply_into_matches_dense_kernel() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let op: &dyn LinearOperator<f64> = &m.view().try_finite().unwrap();
        let x = Vector::from_vec(vec![1.0, -1.0, 2.0]);
        let expected = m.view().mul_vec(&x).unwrap();
        let mut output = Vector::from_vec(vec![99.0; 2]);
        let mut workspace = [0.0; 2];

        op.apply_into(x.as_slice(), output.as_mut_slice(), &mut workspace)
            .unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_apply_into_rejects_output_dimension_before_writing() {
        let m = Matrix::<f64>::identity(2).unwrap();
        let x = Vector::from_vec(vec![1.0, 2.0]);
        let mut output = Vector::from_vec(vec![99.0]);
        let mut workspace = [0.0; 2];

        assert!(matches!(
            (m.view().try_finite().unwrap().apply_into(
                x.as_slice(),
                output.as_mut_slice(),
                &mut workspace
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(output.as_slice(), &[99.0]);
    }

    #[test]
    fn test_linear_operator_apply_transpose() {
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let op: &dyn AdjointOperator<f64> = &m.view().try_finite().unwrap();
        let mut output = [0.0; 2];
        let mut workspace = [0.0; 2];
        op.apply_adjoint_into(&[1.0, 1.0], &mut output, &mut workspace)
            .unwrap();
        // A^T = [[1,3],[2,4]], A^T * [1,1] = [4,6]
        assert!((output[0] - 4.0).abs() < f64::EPSILON);
        assert!((output[1] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_complex_apply_transpose_into_is_conjugating() {
        use scioxide_core::Complex;

        let m = Matrix::from_vec(
            2,
            1,
            vec![Complex::new(1.0_f64, 2.0), Complex::new(3.0, -1.0)],
        )
        .unwrap();
        let x = [Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)];
        let mut output = [Complex::new(0.0, 0.0)];
        let mut workspace = [Complex::new(0.0, 0.0)];

        m.view()
            .try_finite()
            .unwrap()
            .apply_adjoint_into(&x, &mut output, &mut workspace)
            .unwrap();

        assert_eq!(output, [Complex::new(1.0, -1.0)]);
    }

    // ---- Arithmetic ops ----

    #[test]
    fn test_try_add_matrices() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::<f64>::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.view().try_add(&b).unwrap();
        assert!((c.view().get(0, 0).unwrap() - 6.0).abs() < f64::EPSILON);
        assert!((c.view().get(1, 1).unwrap() - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_try_add_dimension_mismatch() {
        let a = Matrix::<f64>::zeros(2, 2).unwrap();
        let b = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (a.view().try_add(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_try_sub_matrices() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let b = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = a.view().try_sub(&b).unwrap();
        assert!((c.view().get(0, 0).unwrap() - 4.0).abs() < f64::EPSILON);
        assert!((c.view().get(1, 1).unwrap() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_try_sub_dimension_mismatch() {
        let a = Matrix::<f64>::zeros(2, 2).unwrap();
        let b = Matrix::<f64>::zeros(3, 2).unwrap();
        assert!(matches!(
            (a.view().try_sub(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_neg_matrix() {
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, -2.0, 3.0, -4.0]).unwrap();
        let n = m.view().try_neg().unwrap();
        assert!((n.view().get(0, 0).unwrap() - (-1.0)).abs() < f64::EPSILON);
        assert!((n.view().get(0, 1).unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scalar_mul_matrix() {
        let m = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scaled = m.view().try_scale(2.0).unwrap();
        assert!((scaled.view().get(0, 0).unwrap() - 2.0).abs() < f64::EPSILON);
        assert!((scaled.view().get(1, 1).unwrap() - 8.0).abs() < f64::EPSILON);
    }

    // ---- get_element (pub(crate) fast-path) ----

    #[test]
    fn test_get_element_valid() {
        let m = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!((m.get_element(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((m.get_element(0, 2) - 3.0).abs() < f64::EPSILON);
        assert!((m.get_element(1, 0) - 4.0).abs() < f64::EPSILON);
        assert!((m.get_element(1, 2) - 6.0).abs() < f64::EPSILON);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "i < self.rows")]
    fn test_get_element_out_of_bounds_row() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        let _ = m.get_element(2, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "j < self.cols")]
    fn test_get_element_out_of_bounds_col() {
        let m = Matrix::<f64>::zeros(2, 3).unwrap();
        let _ = m.get_element(0, 3);
    }
}
