//! Owned dense operator storage with validation retained between updates.

use crate::{FiniteMatrixView, Matrix, MatrixViewMut};
use scioxide_core::{
    AlgebraicScalar, InvalidInputReason, NumericScalar, RealScalar, SciOxideError,
};

/// Stores a row-major matrix and its validated component bound.
///
/// Updating costs O(rows × cols), including one finite-value scan; borrowing an
/// operator costs O(1). Updates and products allocate nothing after construction.
/// The operator uses the same arithmetic and accuracy as [`FiniteMatrixView`].
/// Failed updates invalidate the operator until a complete update succeeds.
pub struct DenseOperatorWorkspace<T: NumericScalar> {
    matrix: Matrix<T>,
    maximum: Option<T::Real>,
}

impl<T: NumericScalar> DenseOperatorWorkspace<T> {
    /// Allocates a zero matrix with O(rows × cols) storage.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure.
    ///
    /// ```
    /// use scioxide_core::LinearOperator;
    /// use scioxide_linalg::backend::dense::DenseOperatorWorkspace;
    /// let mut workspace = DenseOperatorWorkspace::<f64>::new(1, 1)?;
    /// workspace.update(|matrix| matrix.set(0, 0, 2.0))?;
    /// let mut output = [0.0];
    /// workspace.operator()?.apply_into(&[3.0], &mut output, &mut [0.0])?;
    /// assert!((output[0] - 6.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(rows: usize, cols: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            matrix: Matrix::zeros(rows, cols)?,
            maximum: Some(T::Real::ZERO),
        })
    }

    /// Returns the number of matrix rows.
    pub const fn rows(&self) -> usize {
        self.matrix.rows()
    }

    /// Returns the number of matrix columns.
    pub const fn cols(&self) -> usize {
        self.matrix.cols()
    }

    /// Replaces every entry through `fill`, then validates the matrix once.
    ///
    /// Entries initially contain NaNs. No operator is available after an error
    /// or panic; a subsequent complete update restores availability.
    ///
    /// # Errors
    /// Propagates callback errors and rejects non-finite or unwritten entries.
    pub fn update(
        &mut self,
        fill: impl FnOnce(&mut MatrixViewMut<'_, T>) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        self.maximum = None;
        self.matrix.as_mut_slice().fill(T::from_real(T::Real::NAN));
        fill(&mut self.matrix.view_mut())?;
        self.maximum = Some(self.matrix.view().try_finite()?.maximum);
        Ok(())
    }

    /// Borrows the validated operator without scanning matrix entries.
    ///
    /// # Errors
    /// Returns an inconsistent-state error after a failed update.
    pub fn operator(&self) -> Result<FiniteMatrixView<'_, T>, SciOxideError> {
        let maximum = self.maximum.ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "dense operator requires a successful complete update",
            )
        })?;
        Ok(FiniteMatrixView::from_validated(
            self.matrix.view(),
            maximum,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{AdjointOperator, LinearOperator};

    #[test]
    fn updates_refresh_component_bounds_across_scalar_ranges() {
        let mut workspace = DenseOperatorWorkspace::<f64>::new(1, 1).unwrap();
        for (entry, input) in [(1e-300, 1e300), (1e300, 1e-300), (2., 0.5)] {
            workspace.update(|matrix| matrix.set(0, 0, entry)).unwrap();
            let operator = workspace.operator().unwrap();
            let mut out = [0.];
            let mut scratch = [0.];
            operator
                .apply_into(&[input], &mut out, &mut scratch)
                .unwrap();
            assert!((out[0] - 1.).abs() < 1e-14);
            operator
                .apply_adjoint_into(&[input], &mut out, &mut scratch)
                .unwrap();
            assert!((out[0] - 1.).abs() < 1e-14);
        }
    }
}
