//! Analytic diagonal ODEs and regular constrained pairs used by examples.
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    NumericScalar, RealScalar, SciOxideError, SolveContext, StructuralPatternView,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::{
    AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue, DaeAdjointSystem, DaeSystem,
    OdeAdjointSystem, OdeSystem,
};

pub struct Diagonal<T: NumericScalar> {
    pub parameters: Vec<T::Real>,
}
impl<T: NumericScalar> Diagonal<T> {
    pub fn new(n: usize, rate: T::Real) -> Result<Self, SciOxideError> {
        Ok(Self {
            parameters: scioxide_core::backend::try_filled(
                n,
                rate,
                scioxide_core::quantity!("adjoint_example_parameters"),
            )?,
        })
    }
}
impl<T: NumericScalar> OdeSystem<T::Real, T> for Diagonal<T> {
    fn dimension(&self) -> usize {
        self.parameters.len()
    }
    fn rhs(
        &self,
        _: T::Real,
        y: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for ((out, &y), &p) in out.iter_mut().zip(y).zip(&self.parameters) {
            *out = T::from_real(p) * y;
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: T::Real,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (row, &p) in self.parameters.iter().enumerate() {
            for col in 0..self.parameters.len() {
                out.set(row, col, if row == col { T::from_real(p) } else { T::ZERO })?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: T::Real,
        _: &[T],
        pattern: StructuralPatternView<'_>,
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..pattern.rows() {
            let range = pattern.row_ptr()[row]..pattern.row_ptr()[row + 1];
            for (&col, value) in pattern.col_indices()[range.clone()]
                .iter()
                .zip(&mut out[range])
            {
                *value = if col == row {
                    T::from_real(self.parameters[row])
                } else {
                    T::ZERO
                };
            }
        }
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        time: T::Real,
        _: &[T],
        input: &[T],
        out: &mut [T],
        _: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        OdeSystem::rhs(self, time, input, out, cx).map(Some)
    }
}
impl<T: NumericScalar> OdeAdjointSystem<T> for Diagonal<T> {
    fn parameters(&self) -> &[T::Real] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: T::Real,
        y: &[T],
        seed: &[T],
        state: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, (&y, &seed)) in y.iter().zip(seed).enumerate() {
            state[i] = T::from_real(self.parameters[i]) * seed;
            parameters[i] = (seed.conj() * y).real_component();
        }
        Ok(())
    }
}
pub struct Energy;
impl<T: NumericScalar> AdjointObjective<T> for Energy {
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        _: T::Real,
        y: &[T],
        _: &[T::Real],
        state: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        parameters.fill(T::Real::ZERO);
        state.fill(T::ZERO);
        let mut value = T::Real::ZERO;
        if part == AdjointObjectivePart::Terminal {
            state.copy_from_slice(y);
            value = y.iter().fold(T::Real::ZERO, |sum, &y| sum + y.norm_sqr()) * T::Real::HALF;
        }
        Ok(AdjointObjectiveValue {
            value,
            time_derivative: T::Real::ZERO,
        })
    }
}

/// `x'_i=p_i*x_i`, `z_i=2*x_i`, with one real parameter per pair.
pub struct Pairs<T: NumericScalar> {
    pub parameters: Vec<T::Real>,
}
impl<T: NumericScalar> Pairs<T> {
    pub fn new(pairs: usize, rate: T::Real) -> Result<Self, SciOxideError> {
        Ok(Self {
            parameters: scioxide_core::backend::try_filled(
                pairs,
                rate,
                scioxide_core::quantity!("adjoint_example_pairs"),
            )?,
        })
    }
}
impl<T: NumericScalar> DaeSystem<T> for Pairs<T> {
    fn dimension(&self) -> usize {
        2 * self.parameters.len()
    }
    fn residual(
        &self,
        _: T::Real,
        y: &[T],
        v: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, &p) in self.parameters.iter().enumerate() {
            out[2 * i] = v[2 * i] - T::from_real(p) * y[2 * i];
            out[2 * i + 1] = y[2 * i + 1] - y[2 * i] - y[2 * i];
        }
        Ok(())
    }
    fn derivatives(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        y: &mut MatrixViewMut<'_, T>,
        v: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..self.dimension() {
            for col in 0..self.dimension() {
                y.set(row, col, T::ZERO)?;
                v.set(row, col, T::ZERO)?;
            }
        }
        for (i, &p) in self.parameters.iter().enumerate() {
            y.set(2 * i, 2 * i, -T::from_real(p))?;
            v.set(2 * i, 2 * i, T::ONE)?;
            y.set(2 * i + 1, 2 * i, -(T::ONE + T::ONE))?;
            y.set(2 * i + 1, 2 * i + 1, T::ONE)?;
        }
        Ok(Some(()))
    }
    fn sparse_derivatives(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        pattern: StructuralPatternView<'_>,
        y: &mut [T],
        v: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..pattern.rows() {
            for index in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                let col = pattern.col_indices()[index];
                y[index] = if row == col {
                    if row.is_multiple_of(2) {
                        -T::from_real(self.parameters[row / 2])
                    } else {
                        T::ONE
                    }
                } else if !row.is_multiple_of(2) && col + 1 == row {
                    -(T::ONE + T::ONE)
                } else {
                    T::ZERO
                };
                v[index] = if row == col && row.is_multiple_of(2) {
                    T::ONE
                } else {
                    T::ZERO
                };
            }
        }
        Ok(Some(()))
    }
    fn derivative_product(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        y: &[T],
        v: &[T],
        out: &mut [T],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (i, &p) in self.parameters.iter().enumerate() {
            out[2 * i] = v[2 * i] - T::from_real(p) * y[2 * i];
            out[2 * i + 1] = y[2 * i + 1] - y[2 * i] - y[2 * i];
        }
        Ok(Some(()))
    }
}
impl<T: NumericScalar> DaeAdjointSystem<T> for Pairs<T> {
    fn parameters(&self) -> &[T::Real] {
        &self.parameters
    }
    fn is_replayable(&self) -> bool {
        true
    }
    fn pullback(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        seed: &[T],
        state: &mut [T],
        derivative: &mut [T],
        parameters: &mut [T::Real],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, &p) in self.parameters.iter().enumerate() {
            state[2 * i] = -T::from_real(p) * seed[2 * i] - seed[2 * i + 1] - seed[2 * i + 1];
            state[2 * i + 1] = seed[2 * i + 1];
            derivative[2 * i] = seed[2 * i];
            derivative[2 * i + 1] = T::ZERO;
            parameters[i] = -(seed[2 * i].conj() * y[2 * i]).real_component();
        }
        Ok(())
    }
}
