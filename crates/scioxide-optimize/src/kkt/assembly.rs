//! Assembly of dense symmetric KKT blocks.
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;

pub(crate) struct Assembly<'a, S: RealScalar + NumericScalar<Real = S>> {
    matrix: &'a mut Matrix<S>,
}

impl<'a, S: RealScalar + NumericScalar<Real = S>> Assembly<'a, S> {
    pub(crate) fn new(matrix: &'a mut Matrix<S>, order: usize) -> Result<Self, SciOxideError> {
        matrix.try_resize_flat(order, order)?;
        matrix
            .as_mut_slice()
            .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        Ok(Self { matrix })
    }
    pub(crate) fn retained(matrix: &'a mut Matrix<S>) -> Self {
        Self { matrix }
    }
    pub(crate) fn symmetric(
        &mut self,
        offset: usize,
        size: usize,
        mut coefficient: impl FnMut(usize, usize) -> Result<S, SciOxideError>,
    ) -> Result<(), SciOxideError> {
        for i in 0..size {
            for j in 0..=i {
                let value = coefficient(i, j)?;
                self.entry(offset + i, offset + j, value)?;
            }
        }
        Ok(())
    }
    pub(crate) fn diagonal(
        &mut self,
        offset: usize,
        size: usize,
        mut coefficient: impl FnMut(usize) -> Result<S, SciOxideError>,
    ) -> Result<(), SciOxideError> {
        for i in 0..size {
            self.entry(offset + i, offset + i, coefficient(i)?)?;
        }
        Ok(())
    }
    pub(crate) fn coupling(
        &mut self,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        mut coefficient: impl FnMut(usize, usize) -> Result<S, SciOxideError>,
    ) -> Result<(), SciOxideError> {
        for i in 0..rows {
            for j in 0..cols {
                self.entry(row + i, col + j, coefficient(i, j)?)?;
            }
        }
        Ok(())
    }
    pub(crate) fn fix(&mut self, coordinate: usize) -> Result<(), SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        for j in 0..self.matrix.rows() {
            self.entry(coordinate, j, zero)?;
        }
        self.entry(
            coordinate,
            coordinate,
            <S as scioxide_core::AlgebraicScalar>::ONE,
        )
    }
    pub(crate) fn shift(&mut self, coordinate: usize, amount: S) -> Result<(), SciOxideError> {
        let value =
            scioxide_core::checked_sum(&[self.matrix.view().get(coordinate, coordinate)?, amount])?;
        self.entry(coordinate, coordinate, value)
    }
    fn entry(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        if !value.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "non-finite KKT coefficient",
            ));
        }
        self.matrix.view_mut().set(row, col, value)?;
        self.matrix.view_mut().set(col, row, value)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn signed_blocks_regularization_and_fixed_coordinates() {
        let mut matrix = Matrix::<f64>::empty();
        let mut assembly = Assembly::new(&mut matrix, 3).unwrap();
        assembly.diagonal(0, 2, |i| Ok(-(i as f64 + 1.0))).unwrap();
        assembly
            .coupling(2, 0, 1, 2, |_, j| Ok(j as f64 + 3.0))
            .unwrap();
        assembly.shift(2, 0.5).unwrap();
        assert_eq!(
            matrix.as_slice(),
            &[-1.0, 0.0, 3.0, 0.0, -2.0, 4.0, 3.0, 4.0, 0.5]
        );
        Assembly::retained(&mut matrix).fix(1).unwrap();
        assert_eq!(
            matrix.as_slice(),
            &[-1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.5]
        );
        Assembly::new(&mut matrix, 0).unwrap();
        assert_eq!(matrix.rows(), 0);
    }
}
