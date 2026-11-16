//! Strided vector and matrix views over borrowed slices.
//!
//! Strides are counted in elements. Immutable views may map several elements
//! to the same storage location, for example with a zero stride. Mutable views
//! reject such overlap, so every element has its own location. Constructors
//! check the shape and that the slice covers every addressed element, so
//! element access never goes past the slice.

use alloc::vec::Vec;

use scioxide_core::backend::{
    checked_add, checked_product, checked_strided_span, matrix_layout_is_injective,
    try_with_capacity, validate_layout_backing_len as validate_backing_len,
};
use scioxide_core::{
    AlgebraicScalar, Argument, Dimensions, InvalidInputReason, NonFiniteOrigin, NumericScalar,
    RealScalar, SciOxideError,
};

use super::{Matrix, Vector};

impl<'a, S: NumericScalar> From<&'a Matrix<S>> for MatrixView<'a, S> {
    fn from(matrix: &'a Matrix<S>) -> Self {
        matrix.view()
    }
}

impl<'a, S: NumericScalar> From<&'a mut Matrix<S>> for MatrixView<'a, S> {
    fn from(matrix: &'a mut Matrix<S>) -> Self {
        matrix.view()
    }
}

impl<'a, S: NumericScalar> From<&'a Vector<S>> for VectorView<'a, S> {
    fn from(vector: &'a Vector<S>) -> Self {
        vector.view()
    }
}

impl<'a, S: NumericScalar> From<&'a mut Vector<S>> for VectorView<'a, S> {
    fn from(vector: &'a mut Vector<S>) -> Self {
        vector.view()
    }
}

impl<'a, S: NumericScalar> From<&'a [S]> for VectorView<'a, S> {
    fn from(values: &'a [S]) -> Self {
        Self::from_slice(values)
    }
}

/// A vector is the single column of a `len × 1` matrix.
impl<'a, S: NumericScalar> From<VectorView<'a, S>> for MatrixView<'a, S> {
    fn from(vector: VectorView<'a, S>) -> Self {
        Self {
            data: vector.data,
            rows: vector.len,
            cols: 1,
            row_stride: vector.stride,
            col_stride: 1,
            conjugated: vector.conjugated,
        }
    }
}

/// A mutable vector is the single column of a `len × 1` matrix.
impl<'a, S: NumericScalar> From<VectorViewMut<'a, S>> for MatrixViewMut<'a, S> {
    fn from(vector: VectorViewMut<'a, S>) -> Self {
        Self {
            data: vector.data,
            rows: vector.len,
            cols: 1,
            row_stride: vector.stride,
            col_stride: 1,
        }
    }
}

/// Choice of whether a copy rejects NaN and infinite values.
///
/// Used by [`MatrixView::copy_into`] and [`VectorView::copy_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyPolicy {
    /// Return `NonFinite` if any source element is NaN or infinite.
    RequireFinite,
    /// Copy every value, including NaN and infinity, without checking. Never calls a BLAS provider.
    Preserve,
}

/// Borrowed read-only vector whose elements are `stride` apart in a slice.
///
/// Element `i` is `data[i * stride]`, conjugated if the view has the
/// conjugation flag set (see [`conjugate`](Self::conjugate)). Different
/// elements may share a location, for example with stride 0. Get one from
/// [`Vector::view`], a slice (`From<&[S]>`), or a row, column or diagonal of a
/// [`MatrixView`]. See [Matrices and views](crate#matrices-and-views).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct VectorView<'a, S: NumericScalar> {
    pub(crate) data: &'a [S],
    pub(crate) len: usize,
    pub(crate) stride: usize,
    pub(crate) conjugated: bool,
}

impl<'a, S: NumericScalar> VectorView<'a, S> {
    /// Borrows `data` as a vector with stride 1.
    pub const fn from_slice(data: &'a [S]) -> Self {
        Self {
            data,
            len: data.len(),
            stride: 1,
            conjugated: false,
        }
    }

    /// Borrows `len` elements of `data` that are `stride` elements apart.
    ///
    /// Element `i` is `data[i * stride]`. A zero stride repeats `data[0]`.
    /// `data` may be longer than needed.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `(len − 1) · stride` does not fit in `usize`.
    /// - `DimensionMismatch` if `data` is shorter than `(len − 1) · stride + 1`.
    pub fn from_slice_with_stride(
        data: &'a [S],
        len: usize,
        stride: usize,
    ) -> Result<Self, SciOxideError> {
        validate_backing_len(data.len(), required_vector_span(len, stride)?)?;
        Ok(Self {
            data,
            len,
            stride,
            conjugated: false,
        })
    }

    /// Returns the number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view has no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the distance, in slice elements, between consecutive elements.
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Returns `true` if the elements are consecutive in the slice and not conjugated.
    pub const fn is_contiguous(&self) -> bool {
        // A conjugated view is lazy: the backing slice still holds the
        // unconjugated values, so exposing it as a slice would be wrong.
        self.len == 0 || (!self.conjugated && (self.len == 1 || self.stride == 1))
    }

    /// Returns `true` if reads return the complex conjugate of the stored values.
    pub const fn is_conjugated(&self) -> bool {
        self.conjugated
    }

    /// Returns a view of the same storage that toggles the conjugation flag.
    pub const fn conjugate(self) -> Self {
        Self {
            conjugated: !self.conjugated,
            ..self
        }
    }

    /// Returns the elements as a slice if they are contiguous.
    ///
    /// Returns `None` unless [`is_contiguous`](Self::is_contiguous).
    pub fn as_contiguous_slice(&self) -> Option<&'a [S]> {
        self.is_contiguous().then(|| &self.data[..self.len])
    }

    /// Returns element `index`, conjugated if the view is conjugated.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.len()`.
    pub fn get(&self, index: usize) -> Result<S, SciOxideError> {
        if index >= self.len {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("vector_elements"),
                index,
                self.len,
            ));
        }
        Ok(self.get_element(index))
    }

    /// Returns an iterator over the elements in order, conjugated if the view is conjugated.
    pub fn iter(&self) -> impl Iterator<Item = S> + '_ {
        (0..self.len).map(|index| self.get_element(index))
    }

    /// Copies the elements into a new [`Vector`], applying the conjugation flag.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_to_vector(&self) -> Result<Vector<S>, SciOxideError> {
        let mut data = try_with_capacity(self.len, scioxide_core::quantity!("vector_view_copy"))?;
        data.extend(self.iter());
        Ok(Vector::from_vec(data))
    }

    pub(crate) fn try_finite_dispatched(
        self,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<FiniteVectorView<'a, S>, SciOxideError> {
        if self.stride == 1
            && let Some(maximum) = crate::kernel::arch::maximum(&self.data[..self.len], dispatch)?
        {
            return Ok(FiniteVectorView {
                view: self,
                maximum,
            });
        }
        self.try_finite()
    }

    pub(crate) fn try_finite(self) -> Result<FiniteVectorView<'a, S>, SciOxideError> {
        let maximum =
            self.iter()
                .enumerate()
                .try_fold(S::Real::ZERO, |maximum, (index, value)| {
                    validate_finite_element(value, index)?;
                    Ok::<_, SciOxideError>(
                        maximum
                            .max(value.real_component().abs())
                            .max(value.imaginary_component().abs()),
                    )
                })?;
        Ok(FiniteVectorView {
            view: self,
            maximum,
        })
    }

    #[must_use]
    #[inline]
    pub(crate) fn get_element(&self, index: usize) -> S {
        debug_assert!(index < self.len);
        let value = self.data[index * self.stride];
        if self.conjugated { value.conj() } else { value }
    }
}

/// Read-only vector view whose elements are known to be finite.
///
/// The check stays valid because the storage is borrowed immutably.
/// Conjugation keeps it without a new scan.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub(crate) struct FiniteVectorView<'a, S: NumericScalar> {
    view: VectorView<'a, S>,
    pub(crate) maximum: S::Real,
}

impl<'a, S: NumericScalar> FiniteVectorView<'a, S> {
    /// Wrap a finite view; `maximum` is its largest component magnitude.
    pub(crate) const fn from_validated(view: VectorView<'a, S>, maximum: S::Real) -> Self {
        Self { view, maximum }
    }
    // Successful checked kernels and providers establish finiteness. Measure
    // the current range without repeating their public input checks.
    pub(crate) fn from_workspace(view: VectorView<'a, S>) -> Self {
        let maximum = view.iter().fold(S::Real::ZERO, |maximum, value| {
            maximum.max(crate::kernel::standard::magnitude(value))
        });
        Self { view, maximum }
    }

    /// Return the underlying view.
    pub(crate) const fn as_view(self) -> VectorView<'a, S> {
        self.view
    }

    /// Number of logical elements.
    pub(crate) const fn len(&self) -> usize {
        self.view.len()
    }

    /// Returns the view with the conjugation flag toggled, without a new scan.
    pub(crate) const fn conjugate(self) -> Self {
        Self {
            view: self.view.conjugate(),
            maximum: self.maximum,
        }
    }

    /// Iterate over the elements in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = S> + '_ {
        self.view.iter()
    }
}

/// Borrowed mutable vector whose elements are `stride` apart in a slice.
///
/// Element `i` is `data[i * stride]`. Every element has its own location, so a
/// zero stride is allowed only for length 0 or 1. A mutable view has no
/// conjugation flag. See [Matrices and views](crate#matrices-and-views).
#[derive(Debug)]
#[must_use]
pub struct VectorViewMut<'a, S: NumericScalar> {
    pub(crate) data: &'a mut [S],
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

impl<'a, S: NumericScalar> VectorViewMut<'a, S> {
    /// Borrows `data` as a mutable vector with stride 1.
    pub fn from_slice(data: &'a mut [S]) -> Self {
        Self {
            len: data.len(),
            data,
            stride: 1,
        }
    }

    /// Borrows `len` elements of `data` that are `stride` elements apart, for writing.
    ///
    /// Element `i` is `data[i * stride]`. `data` may be longer than needed.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `(len − 1) · stride` does not fit in `usize`.
    /// - `DimensionMismatch` if `data` is shorter than `(len − 1) · stride + 1`.
    /// - `InvalidInput` if `stride == 0` and `len > 1`, since elements would overlap.
    pub fn from_slice_with_stride(
        data: &'a mut [S],
        len: usize,
        stride: usize,
    ) -> Result<Self, SciOxideError> {
        let required = required_vector_span(len, stride)?;
        validate_backing_len(data.len(), required)?;
        if len > 1 && stride == 0 {
            return Err(overlapping_layout_error(
                scioxide_core::argument!("stride"),
                "a mutable vector view with more than one element requires a nonzero stride",
            ));
        }
        Ok(Self { data, len, stride })
    }

    /// Returns the number of elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the view has no elements.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the distance, in slice elements, between consecutive elements.
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Returns `true` if the elements are consecutive in the slice.
    pub const fn is_contiguous(&self) -> bool {
        self.len <= 1 || self.stride == 1
    }

    /// Returns a read-only view of the same elements.
    pub fn as_view(&self) -> VectorView<'_, S> {
        VectorView {
            data: self.data,
            len: self.len,
            stride: self.stride,
            conjugated: false,
        }
    }

    /// Borrows this view for one call, so that it can be used again afterwards.
    pub fn reborrow(&mut self) -> VectorViewMut<'_, S> {
        VectorViewMut {
            data: self.data,
            len: self.len,
            stride: self.stride,
        }
    }
    /// Returns the elements as a slice if they are contiguous.
    ///
    /// Returns `None` unless [`is_contiguous`](Self::is_contiguous).
    pub fn as_contiguous_slice(&self) -> Option<&[S]> {
        self.is_contiguous().then(|| &self.data[..self.len])
    }

    /// Returns the elements as a mutable slice if they are contiguous.
    ///
    /// Returns `None` unless [`is_contiguous`](Self::is_contiguous).
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [S]> {
        self.is_contiguous().then(|| &mut self.data[..self.len])
    }

    /// Returns element `index`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.len()`.
    pub fn get(&self, index: usize) -> Result<S, SciOxideError> {
        self.as_view().get(index)
    }

    /// Returns a mutable reference to element `index`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.len()`.
    pub fn get_mut(&mut self, index: usize) -> Result<&mut S, SciOxideError> {
        if index >= self.len {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("vector_elements"),
                index,
                self.len,
            ));
        }
        Ok(&mut self.data[index * self.stride])
    }

    /// Sets element `index` to `value`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.len()`.
    pub fn set(&mut self, index: usize, value: S) -> Result<(), SciOxideError> {
        *self.get_mut(index)? = value;
        Ok(())
    }

    /// Returns an iterator over references to the elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &S> {
        // Construction allows stride 0 only for `len <= 1`, where the step is
        // irrelevant; clamping avoids `step_by(0)`, which panics.
        self.data.iter().step_by(self.stride.max(1)).take(self.len)
    }

    /// Returns an iterator over mutable references to the elements in order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut S> {
        self.data
            .iter_mut()
            .step_by(self.stride.max(1))
            .take(self.len)
    }

    /// Sets every element to `value`.
    pub fn fill(&mut self, value: S) {
        self.iter_mut().for_each(|element| *element = value);
    }

    /// Copies the elements into a new [`Vector`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_to_vector(&self) -> Result<Vector<S>, SciOxideError> {
        self.as_view().try_to_vector()
    }

    #[inline]
    pub(crate) fn set_element(&mut self, index: usize, value: S) {
        debug_assert!(index < self.len);
        self.data[index * self.stride] = value;
    }
}

/// Borrowed read-only matrix with independent row and column strides.
///
/// Element `(i, j)` is `data[i * row_stride + j * col_stride]`, conjugated if
/// the view has the conjugation flag set. A row-major matrix has
/// `row_stride = cols` and `col_stride = 1`; a column-major matrix has
/// `row_stride = 1` and `col_stride = rows`. [`transpose`](Self::transpose),
/// [`conjugate`](Self::conjugate), [`adjoint`](Self::adjoint) and
/// [`block`](Self::block) return new views of the same storage without
/// copying. Different elements may share a location.
///
/// Most operations in this crate take a `MatrixView` and check that it is
/// finite. Use [`try_finite`](Self::try_finite) to check once and pass the
/// resulting [`FiniteMatrixView`] to several calls. See
/// [Matrices and views](crate#matrices-and-views).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
///
/// // [[1, 2, 3], [4, 5, 6]] in row-major order.
/// let a = MatrixView::from_slice(2, 3, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert_eq!(a.get(1, 0)?, 4.0);
/// assert_eq!(a.transpose().get(0, 1)?, 4.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct MatrixView<'a, S: NumericScalar> {
    pub(crate) data: &'a [S],
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
    pub(crate) conjugated: bool,
}

impl<'a, S: NumericScalar> MatrixView<'a, S> {
    /// Borrows `data` as a `rows × cols` matrix in row-major order.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `DimensionMismatch` if `data.len() != rows * cols`.
    pub fn from_slice(rows: usize, cols: usize, data: &'a [S]) -> Result<Self, SciOxideError> {
        let expected = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("matrix_view_element_count"),
        )?;
        if data.len() != expected {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(expected),
                Dimensions::one(data.len()),
            ));
        }
        Ok(Self {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
            conjugated: false,
        })
    }

    /// Borrows a `rows × cols` matrix with the given strides.
    ///
    /// Element `(i, j)` is `data[i * row_stride + j * col_stride]`. Elements
    /// may overlap. `data` may be longer than needed.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` or the largest offset does not fit in `usize`.
    /// - `DimensionMismatch` if `data` does not reach the largest offset.
    pub fn from_slice_with_strides(
        rows: usize,
        cols: usize,
        data: &'a [S],
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self, SciOxideError> {
        let required = required_matrix_span(rows, cols, row_stride, col_stride)?;
        validate_backing_len(data.len(), required)?;
        Ok(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
            conjugated: false,
        })
    }

    /// Keep at most the first `count` columns.
    pub(crate) fn take_columns(self, count: usize) -> Self {
        Self {
            cols: self.cols.min(count),
            ..self
        }
    }

    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the distance, in slice elements, between `(i, j)` and `(i + 1, j)`.
    pub const fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the distance, in slice elements, between `(i, j)` and `(i, j + 1)`.
    pub const fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// Returns `true` if `rows == 0` or `cols == 0`.
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Returns `true` if the elements are stored row-major without gaps and not conjugated.
    pub const fn is_contiguous(&self) -> bool {
        self.is_empty()
            || (!self.conjugated
                && matrix_layout_is_contiguous(
                    self.rows,
                    self.cols,
                    self.row_stride,
                    self.col_stride,
                ))
    }

    /// Returns `true` if reads return the complex conjugate of the stored values.
    pub const fn is_conjugated(&self) -> bool {
        self.conjugated
    }

    /// Returns a view of the same storage that toggles the conjugation flag.
    pub const fn conjugate(self) -> Self {
        Self {
            conjugated: !self.conjugated,
            ..self
        }
    }
    /// Returns the elements as a row-major slice if they are contiguous.
    ///
    /// Returns `None` unless [`is_contiguous`](Self::is_contiguous).
    pub fn as_contiguous_slice(&self) -> Option<&'a [S]> {
        self.is_contiguous()
            .then(|| &self.data[..self.rows * self.cols])
    }

    /// Returns element `(row, col)`, conjugated if the view is conjugated.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()` or `col >= self.cols()`.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        self.check_bounds(row, col)?;
        Ok(self.get_element(row, col))
    }

    /// Returns row `row` as a vector view with stride `col_stride`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()`.
    pub fn row(&self, row: usize) -> Result<VectorView<'a, S>, SciOxideError> {
        if row >= self.rows {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("matrix_rows"),
                row,
                self.rows,
            ));
        }
        if self.cols == 0 {
            return Ok(VectorView::from_slice(&self.data[..0]));
        }
        Ok(vector_from_matrix_axis(
            self.data,
            row * self.row_stride,
            self.cols,
            self.col_stride,
            self.conjugated,
        ))
    }

    /// Returns column `col` as a vector view with stride `row_stride`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `col >= self.cols()`.
    pub fn col(&self, col: usize) -> Result<VectorView<'a, S>, SciOxideError> {
        if col >= self.cols {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("matrix_columns"),
                col,
                self.cols,
            ));
        }
        if self.rows == 0 {
            return Ok(VectorView::from_slice(&self.data[..0]));
        }
        Ok(vector_from_matrix_axis(
            self.data,
            col * self.col_stride,
            self.rows,
            self.row_stride,
            self.conjugated,
        ))
    }

    /// Returns the main diagonal as a vector view of length `min(rows, cols)`.
    ///
    /// Its stride is `row_stride + col_stride`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `row_stride + col_stride` does not fit in `usize`.
    pub fn try_diag(&self) -> Result<VectorView<'a, S>, SciOxideError> {
        let len = self.rows.min(self.cols);
        if len == 0 {
            return Ok(VectorView {
                data: &self.data[..0],
                len: 0,
                stride: 0,
                conjugated: self.conjugated,
            });
        }
        // Element (i, i) lives at `i * (row_stride + col_stride)`, and the
        // last diagonal entry is inside the matrix span, so the backing slice
        // needs no re-slicing; only the stride sum can overflow.
        let stride = checked_add(
            self.row_stride,
            self.col_stride,
            scioxide_core::quantity!("matrix_diagonal_stride"),
        )?;
        Ok(VectorView {
            data: self.data,
            len,
            stride,
            conjugated: self.conjugated,
        })
    }

    /// Returns the transpose `Aᵀ` as a view of the same storage, by swapping the strides.
    pub const fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            conjugated: self.conjugated,
        }
    }

    /// Returns the conjugate transpose `Aᴴ` as a view of the same storage.
    ///
    /// For real scalars it equals [`transpose`](Self::transpose).
    pub const fn adjoint(self) -> Self {
        self.transpose().conjugate()
    }

    /// Returns the `rows × cols` block whose top-left element is `(first_row, first_col)`.
    ///
    /// The block keeps the strides and the conjugation flag.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `first_row + rows > self.rows()` or
    ///   `first_col + cols > self.cols()`.
    pub fn block(
        &self,
        first_row: usize,
        first_col: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, SciOxideError> {
        validate_block(self.rows, self.cols, first_row, first_col, rows, cols)?;
        // An empty block may start exactly at the matrix edge, where the
        // computed offset could lie past the backing slice; it gets an empty
        // slice instead. Non-empty blocks are re-based onto their own span so
        // the view's `row * row_stride + col * col_stride` addressing stays
        // relative to the block origin and within the sub-slice.
        if rows == 0 || cols == 0 {
            return Ok(Self {
                data: &self.data[..0],
                rows,
                cols,
                row_stride: self.row_stride,
                col_stride: self.col_stride,
                conjugated: self.conjugated,
            });
        }
        let start = first_row * self.row_stride + first_col * self.col_stride;
        let span = required_matrix_span(rows, cols, self.row_stride, self.col_stride)?;
        Ok(Self {
            data: &self.data[start..start + span],
            rows,
            cols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            conjugated: self.conjugated,
        })
    }

    /// Copies the elements into a new row-major [`Matrix`], applying the conjugation flag.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_to_matrix(&self) -> Result<Matrix<S>, SciOxideError> {
        let len = checked_product(
            &[self.rows, self.cols],
            scioxide_core::quantity!("matrix_view_copy_element_count"),
        )?;
        if len == 0 {
            return Matrix::from_vec(self.rows, self.cols, Vec::new());
        }
        let mut data = try_with_capacity(len, scioxide_core::quantity!("matrix_view_copy"))?;
        for row in 0..self.rows {
            data.extend(self.row(row)?.iter());
        }
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Checks that every element is finite and returns the view as a [`FiniteMatrixView`].
    ///
    /// The check reads each element once. Pass the result to several
    /// operations to avoid repeating it.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if the real or imaginary part of an element is NaN or infinite.
    pub fn try_finite(self) -> Result<FiniteMatrixView<'a, S>, SciOxideError> {
        if self.rows == 0 || self.cols == 0 {
            return Ok(FiniteMatrixView::from_validated(self, S::Real::ZERO));
        }
        let maximum = (0..self.rows)
            .flat_map(|row| (0..self.cols).map(move |col| (row, col)))
            .try_fold(S::Real::ZERO, |maximum, (row, col)| {
                let value = self.get_element(row, col);
                validate_finite_element(value, row * self.cols + col)?;
                Ok::<_, SciOxideError>(
                    maximum
                        .max(value.real_component().abs())
                        .max(value.imaginary_component().abs()),
                )
            })?;
        Ok(FiniteMatrixView {
            view: self,
            maximum,
        })
    }

    pub(crate) fn try_finite_dispatched(
        self,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<FiniteMatrixView<'a, S>, SciOxideError> {
        // Conjugation preserves the component maximum. Both dense storage
        // orders admit the same physical scan, including transpose/adjoint views.
        let contiguous =
            matrix_layout_is_contiguous(self.rows, self.cols, self.row_stride, self.col_stride)
                || matrix_layout_is_contiguous(
                    self.cols,
                    self.rows,
                    self.col_stride,
                    self.row_stride,
                );
        if contiguous
            && let Some(maximum) =
                crate::kernel::arch::maximum(&self.data[..self.rows * self.cols], dispatch)?
        {
            return Ok(FiniteMatrixView {
                view: self,
                maximum,
            });
        }
        self.try_finite()
    }

    // Bounds check reported.
    const fn check_bounds(&self, row: usize, col: usize) -> Result<(), SciOxideError> {
        check_bounds(self.rows, self.cols, row, col)
    }

    #[must_use]
    #[inline]
    pub(crate) fn get_element(&self, row: usize, col: usize) -> S {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        let value = self.data[row * self.row_stride + col * self.col_stride];
        if self.conjugated { value.conj() } else { value }
    }
}

/// Read-only matrix view whose elements are known to be finite.
///
/// Get one from [`MatrixView::try_finite`]. The check stays valid because the
/// storage is borrowed immutably, and [`transpose`](Self::transpose),
/// [`conjugate`](Self::conjugate), [`adjoint`](Self::adjoint) and
/// [`block`](Self::block) keep it without a new scan. It implements
/// [`LinearOperator`](scioxide_core::LinearOperator) and
/// [`AdjointOperator`](scioxide_core::AdjointOperator), so it can be passed to
/// the iterative solvers.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct FiniteMatrixView<'a, S: NumericScalar> {
    view: MatrixView<'a, S>,
    pub(crate) maximum: S::Real,
}

impl<'a, S: NumericScalar> FiniteMatrixView<'a, S> {
    // A successful checked kernel or factorization step established finiteness.
    // Re-measure the current range without re-running public input validation.
    pub(crate) fn from_workspace(view: MatrixView<'a, S>) -> Self {
        if view.is_empty() {
            return Self::from_validated(view, S::Real::ZERO);
        }
        let maximum = crate::kernel::standard::component_maximum(view, None);
        Self { view, maximum }
    }

    // Callers have checked every written component while constructing storage.
    pub(crate) const fn from_validated(view: MatrixView<'a, S>, maximum: S::Real) -> Self {
        Self { view, maximum }
    }

    /// Returns the underlying [`MatrixView`].
    pub const fn as_view(self) -> MatrixView<'a, S> {
        self.view
    }

    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.view.rows()
    }

    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.view.cols()
    }

    /// Returns `true` if `rows == 0` or `cols == 0`.
    pub const fn is_empty(&self) -> bool {
        self.view.is_empty()
    }

    /// Returns the distance, in slice elements, between `(i, j)` and `(i + 1, j)`.
    pub const fn row_stride(&self) -> usize {
        self.view.row_stride()
    }

    /// Returns the distance, in slice elements, between `(i, j)` and `(i, j + 1)`.
    pub const fn col_stride(&self) -> usize {
        self.view.col_stride()
    }

    /// Returns the transpose `Aᵀ` without a new scan.
    pub const fn transpose(self) -> Self {
        Self {
            view: self.view.transpose(),
            maximum: self.maximum,
        }
    }

    /// Returns the view with the conjugation flag toggled, without a new scan.
    pub const fn conjugate(self) -> Self {
        Self {
            view: self.view.conjugate(),
            maximum: self.maximum,
        }
    }

    /// Returns the conjugate transpose `Aᴴ` without a new scan.
    pub const fn adjoint(self) -> Self {
        Self {
            view: self.view.adjoint(),
            maximum: self.maximum,
        }
    }

    /// Returns the `rows × cols` block starting at `(first_row, first_col)`, without a new scan.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `first_row + rows > self.rows()` or
    ///   `first_col + cols > self.cols()`.
    pub fn block(
        &self,
        first_row: usize,
        first_col: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            view: self.view.block(first_row, first_col, rows, cols)?,
            maximum: self.maximum,
        })
    }

    /// Returns element `(row, col)`, conjugated if the view is conjugated.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()` or `col >= self.cols()`.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        self.view.get(row, col)
    }
}

/// Borrowed mutable matrix with independent row and column strides.
///
/// Element `(i, j)` is `data[i * row_stride + j * col_stride]`. Every element
/// has its own location; constructors reject strides that would make two
/// elements overlap. A mutable view has no conjugation flag. Operations write
/// their results into a `MatrixViewMut`. See
/// [Matrices and views](crate#matrices-and-views).
#[derive(Debug)]
#[must_use]
pub struct MatrixViewMut<'a, S: NumericScalar> {
    pub(crate) data: &'a mut [S],
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

impl<'a, S: NumericScalar> MatrixViewMut<'a, S> {
    /// Borrows `data` as a mutable `rows × cols` matrix in row-major order.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `DimensionMismatch` if `data.len() != rows * cols`.
    pub fn from_slice(rows: usize, cols: usize, data: &'a mut [S]) -> Result<Self, SciOxideError> {
        let expected = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("mutable_matrix_view_element_count"),
        )?;
        if data.len() != expected {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(expected),
                Dimensions::one(data.len()),
            ));
        }
        Ok(Self {
            data,
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
        })
    }

    /// Borrows a mutable `rows × cols` matrix with the given strides.
    ///
    /// Element `(i, j)` is `data[i * row_stride + j * col_stride]`. `data` may
    /// be longer than needed.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` or the largest offset does not fit in `usize`.
    /// - `DimensionMismatch` if `data` does not reach the largest offset.
    /// - `InvalidInput` if two different elements would share a location.
    pub fn from_slice_with_strides(
        rows: usize,
        cols: usize,
        data: &'a mut [S],
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self, SciOxideError> {
        let required = required_matrix_span(rows, cols, row_stride, col_stride)?;
        validate_backing_len(data.len(), required)?;
        // Unlike read-only views, a mutable view must give every logical
        // element its own storage cell; otherwise two writes could silently
        // clobber each other and in-place algorithms would be incorrect.
        if !matrix_layout_is_injective(rows, cols, row_stride, col_stride) {
            return Err(overlapping_layout_error(
                scioxide_core::argument!("col_stride"),
                "mutable matrix view strides map distinct logical elements to the same storage",
            ));
        }
        Ok(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the distance, in slice elements, between `(i, j)` and `(i + 1, j)`.
    pub const fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the distance, in slice elements, between `(i, j)` and `(i, j + 1)`.
    pub const fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// Returns `true` if `rows == 0` or `cols == 0`.
    pub const fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Returns `true` if the elements are stored row-major without gaps.
    pub const fn is_contiguous(&self) -> bool {
        matrix_layout_is_contiguous(self.rows, self.cols, self.row_stride, self.col_stride)
    }

    /// Returns a read-only view of the same elements.
    pub fn as_view(&self) -> MatrixView<'_, S> {
        MatrixView {
            data: self.data,
            rows: self.rows,
            cols: self.cols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            conjugated: false,
        }
    }

    /// Borrows this view for one call, so that it can be used again afterwards.
    pub fn reborrow(&mut self) -> MatrixViewMut<'_, S> {
        MatrixViewMut {
            data: self.data,
            rows: self.rows,
            cols: self.cols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
        }
    }

    /// Returns the elements as a row-major slice if they are contiguous.
    ///
    /// Returns `None` unless [`is_contiguous`](Self::is_contiguous).
    pub fn as_contiguous_slice(&self) -> Option<&[S]> {
        self.is_contiguous()
            .then(|| &self.data[..self.rows * self.cols])
    }

    /// Returns the elements as a mutable row-major slice if they are contiguous.
    ///
    /// Returns `None` unless [`is_contiguous`](Self::is_contiguous).
    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [S]> {
        self.is_contiguous()
            .then(|| &mut self.data[..self.rows * self.cols])
    }

    /// Returns element `(row, col)`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()` or `col >= self.cols()`.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        self.as_view().get(row, col)
    }

    /// Returns a mutable reference to element `(row, col)`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()` or `col >= self.cols()`.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Result<&mut S, SciOxideError> {
        self.check_bounds(row, col)?;
        Ok(&mut self.data[row * self.row_stride + col * self.col_stride])
    }

    /// Sets element `(row, col)` to `value`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()` or `col >= self.cols()`.
    pub fn set(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        *self.get_mut(row, col)? = value;
        Ok(())
    }

    /// Returns row `row` as a read-only vector view.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()`.
    pub fn row(&self, row: usize) -> Result<VectorView<'_, S>, SciOxideError> {
        self.as_view().row(row)
    }

    /// Returns column `col` as a read-only vector view.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `col >= self.cols()`.
    pub fn col(&self, col: usize) -> Result<VectorView<'_, S>, SciOxideError> {
        self.as_view().col(col)
    }

    /// Returns row `row` as a mutable vector view.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row >= self.rows()`.
    pub fn row_mut(&mut self, row: usize) -> Result<VectorViewMut<'_, S>, SciOxideError> {
        if row >= self.rows {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("matrix_rows"),
                row,
                self.rows,
            ));
        }
        if self.cols == 0 {
            return Ok(VectorViewMut::from_slice(&mut self.data[..0]));
        }
        Ok(vector_mut_from_matrix_axis(
            self.data,
            row * self.row_stride,
            self.cols,
            self.col_stride,
        ))
    }

    /// Returns column `col` as a mutable vector view.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `col >= self.cols()`.
    pub fn col_mut(&mut self, col: usize) -> Result<VectorViewMut<'_, S>, SciOxideError> {
        if col >= self.cols {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("matrix_columns"),
                col,
                self.cols,
            ));
        }
        if self.rows == 0 {
            return Ok(VectorViewMut::from_slice(&mut self.data[..0]));
        }
        Ok(vector_mut_from_matrix_axis(
            self.data,
            col * self.col_stride,
            self.rows,
            self.row_stride,
        ))
    }

    /// Returns the transpose as a mutable view of the same storage, by swapping the strides.
    pub fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// Return the `rows × cols` block whose top-left element is `(first_row, first_col)`.
    ///
    /// # Errors
    ///
    /// Returns `OutOfBounds` if the block is outside the matrix.
    pub(crate) fn into_block(
        self,
        first_row: usize,
        first_col: usize,
        rows: usize,
        cols: usize,
    ) -> Result<MatrixViewMut<'a, S>, SciOxideError> {
        validate_block(self.rows, self.cols, first_row, first_col, rows, cols)?;
        if rows == 0 || cols == 0 {
            return Ok(MatrixViewMut {
                data: &mut self.data[..0],
                rows,
                cols,
                row_stride: self.row_stride,
                col_stride: self.col_stride,
            });
        }
        let start = first_row * self.row_stride + first_col * self.col_stride;
        let span = required_matrix_span(rows, cols, self.row_stride, self.col_stride)?;
        Ok(MatrixViewMut {
            data: &mut self.data[start..start + span],
            rows,
            cols,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
        })
    }

    /// Returns the `rows × cols` block whose top-left element is `(first_row, first_col)`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `first_row + rows > self.rows()` or
    ///   `first_col + cols > self.cols()`.
    pub fn block_mut(
        &mut self,
        first_row: usize,
        first_col: usize,
        rows: usize,
        cols: usize,
    ) -> Result<MatrixViewMut<'_, S>, SciOxideError> {
        self.reborrow().into_block(first_row, first_col, rows, cols)
    }

    /// Sets every element to `value`.
    pub fn fill(&mut self, value: S) {
        if self.is_empty() {
            return;
        }
        let mut row = 0;
        while row < self.rows {
            let mut col = 0;
            while col < self.cols {
                self.set_element(row, col, value);
                col += 1;
            }
            row += 1;
        }
    }

    /// Copies the elements into a new row-major [`Matrix`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows * cols` does not fit in `usize`.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_to_matrix(&self) -> Result<Matrix<S>, SciOxideError> {
        self.as_view().try_to_matrix()
    }

    // Bounds check reported.
    const fn check_bounds(&self, row: usize, col: usize) -> Result<(), SciOxideError> {
        check_bounds(self.rows, self.cols, row, col)
    }

    #[must_use]
    #[inline]
    pub(crate) fn get_element(&self, row: usize, col: usize) -> S {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        self.data[row * self.row_stride + col * self.col_stride]
    }

    #[inline]
    pub(crate) fn set_element(&mut self, row: usize, col: usize, value: S) {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        self.data[row * self.row_stride + col * self.col_stride] = value;
    }
}

// Reachable vector span.
fn required_vector_span(len: usize, stride: usize) -> Result<usize, SciOxideError> {
    checked_strided_span(&[len], &[stride])
}

fn required_matrix_span(
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
) -> Result<usize, SciOxideError> {
    checked_strided_span(&[rows, cols], &[row_stride, col_stride])
}

fn validate_block(
    source_rows: usize,
    source_cols: usize,
    first_row: usize,
    first_col: usize,
    rows: usize,
    cols: usize,
) -> Result<(), SciOxideError> {
    let rows_fit = first_row <= source_rows && rows <= source_rows - first_row;
    let cols_fit = first_col <= source_cols && cols <= source_cols - first_col;
    if !rows_fit {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("matrix_block_rows"),
            first_row.saturating_add(rows),
            source_rows,
        ));
    }
    if !cols_fit {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("matrix_block_columns"),
            first_col.saturating_add(cols),
            source_cols,
        ));
    }
    Ok(())
}

fn validate_finite_element<S: NumericScalar>(value: S, index: usize) -> Result<(), SciOxideError> {
    if value.has_finite_components() {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            alloc::format!("logical element {index} has a non-finite component"),
        ))
    }
}

fn overlapping_layout_error(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Inconsistent, detail)
}

const fn matrix_layout_is_contiguous(
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
) -> bool {
    // "Contiguous" means dense row-major. Strides along an axis of extent one
    // are never used for addressing, so single-row/column views only need
    // unit stride along their one non-trivial axis.
    if rows == 0 || cols == 0 || (rows == 1 && cols == 1) {
        return true;
    }
    if rows == 1 {
        return col_stride == 1;
    }
    if cols == 1 {
        return row_stride == 1;
    }
    col_stride == 1 && row_stride == cols
}

fn vector_from_matrix_axis<'a, S: NumericScalar>(
    data: &'a [S],
    start: usize,
    len: usize,
    stride: usize,
    conjugated: bool,
) -> VectorView<'a, S> {
    if len == 0 {
        return VectorView {
            data: &data[..0],
            len: 0,
            stride,
            conjugated,
        };
    }
    let end = start + (len - 1) * stride + 1;
    VectorView {
        data: &data[start..end],
        len,
        stride,
        conjugated,
    }
}

fn vector_mut_from_matrix_axis<'a, S: NumericScalar>(
    data: &'a mut [S],
    start: usize,
    len: usize,
    stride: usize,
) -> VectorViewMut<'a, S> {
    if len == 0 {
        return VectorViewMut::from_slice(&mut data[..0]);
    }
    let end = start + (len - 1) * stride + 1;
    VectorViewMut {
        data: &mut data[start..end],
        len,
        stride,
    }
}

impl<T: NumericScalar + scioxide_core::AffineValueMut> scioxide_core::AffineValue
    for VectorView<'_, T>
{
    type Scale = <T as scioxide_core::AffineValue>::Scale;
    type Owned = Vector<T>;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        if self.is_empty() {
            return Ok(0);
        }
        scioxide_core::backend::checked_mul(
            self.len(),
            self.get(0)?.dimension()?,
            scioxide_core::quantity!("affine_vector_view"),
        )
    }
    fn component(&self, index: usize) -> Result<Self::Scale, SciOxideError> {
        let dimension = self.dimension()?;
        if index >= dimension {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("affine_vector_view"),
                index,
                dimension,
            ));
        }
        let width = self.get(0)?.dimension()?;
        self.get(index / width)?.component(index % width)
    }
    fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError> {
        self.try_to_vector()
    }
}
impl<T: NumericScalar + scioxide_core::AffineValueMut> scioxide_core::AffineValue
    for VectorViewMut<'_, T>
{
    type Scale = <T as scioxide_core::AffineValue>::Scale;
    type Owned = Vector<T>;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        self.as_view().dimension()
    }
    fn component(&self, index: usize) -> Result<Self::Scale, SciOxideError> {
        self.as_view().component(index)
    }
    fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError> {
        self.try_to_vector()
    }
}
impl<T: NumericScalar + scioxide_core::AffineValueMut> scioxide_core::AffineValueMut
    for VectorViewMut<'_, T>
{
    fn for_each_component_mut(&mut self, mut visit: impl FnMut(&mut Self::Scale)) {
        self.iter_mut()
            .for_each(|value| value.for_each_component_mut(&mut visit));
    }
}

/// Reject an element index outside a `rows x cols` view.
const fn check_bounds(
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) -> Result<(), SciOxideError> {
    if row >= rows {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("matrix_rows"),
            row,
            rows,
        ));
    }
    if col >= cols {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("matrix_columns"),
            col,
            cols,
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Complex64;
    use std::collections::HashSet;

    #[test]
    fn vector_views_cover_contiguous_and_strided_storage() {
        let data = [1.0_f64, 99.0, 2.0, 99.0, 3.0];
        let contiguous = VectorView::from_slice(&data[..3]);
        let strided = VectorView::from_slice_with_stride(&data, 3, 2).unwrap();

        assert_eq!(contiguous.as_contiguous_slice(), Some(&data[..3]));
        assert_eq!(strided.as_contiguous_slice(), None);
        assert_eq!(strided.iter().collect::<Vec<_>>(), [1.0, 2.0, 3.0]);
        assert_eq!(
            strided.try_to_vector().unwrap().as_slice(),
            &[1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn mutable_vector_view_updates_only_logical_elements() {
        let mut data = [1.0_f64, 99.0, 2.0, 99.0, 3.0];
        let mut view = VectorViewMut::from_slice_with_stride(&mut data, 3, 2).unwrap();
        view.iter_mut().for_each(|value| *value *= 10.0);

        assert_eq!(data, [10.0, 99.0, 20.0, 99.0, 30.0]);
    }

    #[test]
    fn immutable_overlap_is_a_value_but_mutable_overlap_is_structural() {
        let mut data = [7.0_f64];
        let repeated = VectorView::from_slice_with_stride(&data, 3, 0).unwrap();
        assert_eq!(repeated.iter().collect::<Vec<_>>(), [7.0; 3]);

        assert!(matches!(
            (VectorViewMut::from_slice_with_stride(&mut data, 3, 0))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn matrix_views_cover_padded_and_transposed_layouts() {
        let data = [1.0_f64, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0];
        let view = MatrixView::from_slice_with_strides(2, 3, &data, 4, 1).unwrap();
        let transposed = view.transpose();

        assert_eq!(
            view.row(1).unwrap().as_contiguous_slice(),
            Some(&data[4..7])
        );
        assert_eq!(view.col(1).unwrap().iter().collect::<Vec<_>>(), [2.0, 5.0]);
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        assert_eq!(transposed.get(2, 1).unwrap(), 6.0);
        assert_eq!(
            view.try_to_matrix().unwrap().as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn mutable_matrix_rows_columns_and_transpose_share_storage_safely() {
        let mut data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        {
            let mut view = MatrixViewMut::from_slice(2, 3, &mut data).unwrap();
            view.col_mut(1).unwrap().fill(9.0);
            let mut transposed = view.transpose();
            transposed.row_mut(2).unwrap().fill(8.0);
        }

        assert_eq!(data, [1.0, 9.0, 8.0, 4.0, 9.0, 8.0]);
    }

    #[test]
    fn mutable_matrix_rejects_overlapping_strides() {
        let mut data = [0.0_f64; 3];
        assert!(matches!(
            (MatrixViewMut::from_slice_with_strides(2, 2, &mut data, 1, 1))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn mutable_layout_check_matches_distinct_offsets() {
        for rows in 0..5 {
            for cols in 0..5 {
                for row_stride in 0..7 {
                    for col_stride in 0..7 {
                        let mut offsets = HashSet::new();
                        let mut distinct = true;
                        for row in 0..rows {
                            for col in 0..cols {
                                distinct &= offsets.insert(row * row_stride + col * col_stride);
                            }
                        }
                        assert_eq!(
                            matrix_layout_is_injective(rows, cols, row_stride, col_stride,),
                            distinct,
                            "shape={rows}x{cols}, strides=({row_stride}, {col_stride})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn constructors_distinguish_empty_values_from_structural_errors() {
        let empty_rows = MatrixView::<f64>::from_slice(0, usize::MAX, &[]).unwrap();
        assert!(empty_rows.col(usize::MAX - 1).unwrap().is_empty());
        let empty_columns =
            MatrixView::<f64>::from_slice_with_strides(3, 0, &[], usize::MAX, 1).unwrap();
        assert!(empty_columns.row(2).unwrap().is_empty());
        assert!(matches!(
            (MatrixView::<f64>::from_slice_with_strides(2, 2, &[0.0; 2], 2, 1))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert!(matches!(
            (VectorView::<f64>::from_slice_with_stride(&[0.0], usize::MAX, 2))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
    }

    #[test]
    fn conjugate_transpose_and_adjoint_are_composable_zero_copy_views() {
        let data = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, -8.0),
        ];
        let view = MatrixView::from_slice(2, 2, &data).unwrap();
        let adjoint = view.adjoint();

        assert!(adjoint.is_conjugated());
        assert_eq!(adjoint.get(1, 0).unwrap(), Complex64::new(3.0, 4.0));
        assert_eq!(view.conjugate().conjugate().get(0, 1).unwrap(), data[1]);
        assert!(view.conjugate().as_contiguous_slice().is_none());
    }

    #[test]
    fn finite_tokens_reject_non_finite_values_and_preserve_transformations() {
        let finite = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let token = VectorView::from_slice(&finite).try_finite().unwrap();
        assert_eq!(
            token.conjugate().as_view().get(1).unwrap(),
            Complex64::new(3.0, -4.0)
        );

        let non_finite = [Complex64::new(1.0, f64::INFINITY)];
        assert!(matches!(
            (VectorView::from_slice(&non_finite).try_finite())
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn block_views_preserve_layout_and_mutable_alias_proof() {
        let mut data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        {
            let mut matrix = MatrixViewMut::from_slice(3, 3, &mut data).unwrap();
            let mut block = matrix.block_mut(1, 1, 2, 2).unwrap();
            block.fill(0.0);
        }
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 0.0, 0.0]);

        let view = MatrixView::from_slice(3, 3, &data).unwrap();
        assert_eq!(view.block(0, 1, 2, 2).unwrap().get(1, 0).unwrap(), 0.0);
        assert!(view.block(2, 2, 2, 1).is_err());
    }
}

#[cfg(all(test, feature = "std"))]
#[test]
#[allow(clippy::unwrap_used)]
fn finite_operator_reuses_matrix_proof_for_scoped_products() {
    use scioxide_core::{
        AdjointOperator, ExecutionContext, LinearOperator, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let values = alloc::vec![1.0_f64; 257*129];
    let matrix = MatrixView::from_slice(257, 129, &values)
        .unwrap()
        .try_finite()
        .unwrap();
    let input = alloc::vec![1.0;129];
    let mut output = alloc::vec![0.0;257];
    let mut scratch = alloc::vec![0.0;257];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(939),
        scioxide_core::algorithm_id!("test", "finite_operator", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    for _ in 0..4 {
        matrix
            .apply_into_scoped(
                &input,
                &mut output,
                &mut scratch,
                &mut scioxide_core::OperatorContext::new(&mut context),
            )
            .and_then(scioxide_core::Invocation::require)
            .unwrap();
        assert!(output.iter().all(|&value| value == 129.0));
    }
    let mut adjoint = alloc::vec![0.0;129];
    matrix
        .apply_adjoint_into_scoped(
            &output,
            &mut adjoint,
            &mut scratch,
            &mut scioxide_core::OperatorContext::new(&mut context),
        )
        .and_then(scioxide_core::Invocation::require)
        .unwrap();
}
