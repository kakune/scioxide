//! Contiguous matrix regions lent by a prepared workspace.

use crate::{Matrix, MatrixView, MatrixViewMut};
use scioxide_core::{NumericScalar, SciOxideError};

/// Internal contiguous storage used by the same owned and borrowed kernels.
pub(crate) trait MatrixStorage<S: NumericScalar> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn as_slice(&self) -> &[S];
    fn as_mut_slice(&mut self) -> &mut [S];
    fn try_resize_flat(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError>;

    #[inline]
    fn try_view(&self) -> Result<MatrixView<'_, S>, SciOxideError> {
        MatrixView::from_slice(self.rows(), self.cols(), self.as_slice())
    }
    #[inline]
    fn try_view_mut(&mut self) -> Result<MatrixViewMut<'_, S>, SciOxideError> {
        let (rows, cols) = (self.rows(), self.cols());
        MatrixViewMut::from_slice(rows, cols, self.as_mut_slice())
    }
    fn get_element(&self, row: usize, col: usize) -> S {
        self.as_slice()[row * self.cols() + col]
    }
    fn set_element(&mut self, row: usize, col: usize, value: S) {
        let index = row * self.cols() + col;
        self.as_mut_slice()[index] = value;
    }
}

impl<S: NumericScalar> MatrixStorage<S> for Matrix<S> {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn as_slice(&self) -> &[S] {
        self.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [S] {
        self.as_mut_slice()
    }
    #[inline]
    fn try_view(&self) -> Result<MatrixView<'_, S>, SciOxideError> {
        Ok(self.view())
    }
    #[inline]
    fn try_view_mut(&mut self) -> Result<MatrixViewMut<'_, S>, SciOxideError> {
        Ok(self.view_mut())
    }
    fn try_resize_flat(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        self.try_resize_flat(rows, cols)
    }
}

/// A matrix header over one disjoint arena region; never owns or grows storage.
pub(crate) struct ScratchMatrix<'a, S> {
    data: &'a mut [S],
    rows: usize,
    cols: usize,
}

impl<'a, S: NumericScalar> ScratchMatrix<'a, S> {
    #[inline]
    pub(crate) fn new(data: &'a mut [S], rows: usize, cols: usize) -> Result<Self, SciOxideError> {
        let len = scioxide_core::backend::checked_product(
            &[rows, cols],
            scioxide_core::quantity!("matrix_scratch"),
        )?;
        if len > data.len() {
            return Err(SciOxideError::dimension_mismatch([len], [data.len()]));
        }
        Ok(Self { data, rows, cols })
    }
}

impl<S: NumericScalar> MatrixStorage<S> for ScratchMatrix<'_, S> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn as_slice(&self) -> &[S] {
        &self.data[..self.rows * self.cols]
    }
    fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data[..self.rows * self.cols]
    }
    fn try_resize_flat(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        let len = scioxide_core::backend::checked_product(
            &[rows, cols],
            scioxide_core::quantity!("matrix_scratch"),
        )?;
        if len > self.data.len() {
            return Err(SciOxideError::dimension_mismatch([len], [self.data.len()]));
        }
        let previous = self.rows * self.cols;
        if len > previous {
            self.data[previous..len].fill(S::ZERO);
        }
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }
}

/// A variable-length sequence within one reserved, fixed-capacity scratch region.
pub(crate) struct ScratchVec<'a, T> {
    data: &'a mut [T],
    len: usize,
}
impl<'a, T: Copy> ScratchVec<'a, T> {
    #[inline]
    pub(crate) fn new(data: &'a mut [T], len: usize) -> Result<Self, SciOxideError> {
        if len > data.len() {
            return Err(Self::short(len, data.len()));
        }
        Ok(Self { data, len })
    }
    fn short(needed: usize, available: usize) -> SciOxideError {
        SciOxideError::dimension_mismatch([needed], [available])
    }
    #[inline]
    pub(crate) fn resize(&mut self, len: usize, fill: T) -> Result<(), SciOxideError> {
        if len > self.data.len() {
            return Err(Self::short(len, self.data.len()));
        }
        if len > self.len {
            self.data[self.len..len].fill(fill);
        }
        self.len = len;
        Ok(())
    }
    #[inline]
    pub(crate) fn assign(&mut self, source: &[T]) -> Result<(), SciOxideError> {
        if source.len() > self.data.len() {
            return Err(Self::short(source.len(), self.data.len()));
        }
        self.data[..source.len()].copy_from_slice(source);
        self.len = source.len();
        Ok(())
    }
    #[inline]
    pub(crate) fn push(&mut self, value: T) -> Result<(), SciOxideError> {
        let Some(slot) = self.data.get_mut(self.len) else {
            return Err(Self::short(self.len.saturating_add(1), self.data.len()));
        };
        *slot = value;
        self.len += 1;
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }
    pub(crate) fn remove(&mut self, index: usize) {
        self.data.copy_within(index + 1..self.len, index);
        self.len -= 1;
    }
}
impl<T> core::ops::Deref for ScratchVec<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data[..self.len]
    }
}
impl<T> core::ops::DerefMut for ScratchVec<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }
}
