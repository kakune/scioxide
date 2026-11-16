//! An ODE written as a DAE residual for the implicit adjoint plans.
use super::*;
use crate::DaeSystem;
use core::marker::PhantomData;
use scioxide_core::StructuralPatternView;
use scioxide_linalg::MatrixViewMut;

/// Writes an ODE `y′ = f(t, y, p)` as the DAE residual `F(t, y, y′) = y′ − f(t, y, p)`,
/// so that [`DaeAdjointSolver`] can compute its gradients with BDF, backward Euler or the
/// implicit trapezoidal rule.
///
/// The residual Jacobians are `∂F/∂y = −∂f/∂y` and `∂F/∂y′ = I`. An analytic Jacobian
/// of the ODE system is used when it has one; otherwise the derivative strategy of the
/// DAE settings applies to the residual. For complex states the right-hand side must be
/// holomorphic. An ODE with a mass matrix needs its own [`DaeAdjointSystem`].
pub struct OdeAdjointResidual<'a, T: NumericScalar, P> {
    problem: &'a P,
    scalar: PhantomData<T>,
}
impl<'a, T: NumericScalar, P: OdeAdjointSystem<T>> OdeAdjointResidual<'a, T, P> {
    /// Wraps `problem`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `problem` has a mass matrix or is not replayable.
    pub fn new(problem: &'a P) -> Result<Self, SciOxideError> {
        if problem.mass_matrix_revision().is_some() || !problem.is_replayable() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "implicit ODE adjoints require replayable normalized dynamics",
            ));
        }
        Ok(Self {
            problem,
            scalar: PhantomData,
        })
    }
    fn dense(
        &self,
        time: T::Real,
        state: &[T],
        y: &mut MatrixViewMut<'_, T>,
        v: &mut MatrixViewMut<'_, T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if self.problem.jacobian(time, state, y, context)?.is_none() {
            return Ok(None);
        }
        for row in 0..state.len() {
            for col in 0..state.len() {
                y.set(row, col, -y.get(row, col)?)?;
                v.set(row, col, if row == col { T::ONE } else { T::ZERO })?;
            }
        }
        Ok(Some(()))
    }
    fn sparse(
        &self,
        time: T::Real,
        state: &[T],
        pattern: StructuralPatternView<'_>,
        y: &mut [T],
        v: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let available = self
            .problem
            .sparse_jacobian(time, state, pattern, y, context)?;
        if available.is_none() {
            return Ok(None);
        }
        for row in 0..pattern.rows() {
            for index in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                y[index] = -y[index];
                v[index] = if pattern.col_indices()[index] == row {
                    T::ONE
                } else {
                    T::ZERO
                };
            }
        }
        Ok(Some(()))
    }
}
impl<T: NumericScalar, P: OdeAdjointSystem<T>> DaeSystem<T> for OdeAdjointResidual<'_, T, P> {
    fn step_limit(
        &self,
        time: T::Real,
        state: &[T],
        _: &[T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<T::Real>, SciOxideError> {
        self.problem.step_limit(time, state, context)
    }
    fn dimension(&self) -> usize {
        self.problem.dimension()
    }
    fn residual(
        &self,
        time: T::Real,
        state: &[T],
        derivative: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.problem.rhs(time, state, output, cx)?;
        for (out, &v) in output.iter_mut().zip(derivative) {
            *out = v - *out;
        }
        Ok(())
    }
    fn derivatives(
        &self,
        time: T::Real,
        state: &[T],
        _: &[T],
        y: &mut MatrixViewMut<'_, T>,
        v: &mut MatrixViewMut<'_, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.dense(time, state, y, v, cx)
    }
    fn sparse_derivatives(
        &self,
        time: T::Real,
        state: &[T],
        _: &[T],
        pattern: StructuralPatternView<'_>,
        y: &mut [T],
        v: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.sparse(time, state, pattern, y, v, cx)
    }
    fn derivative_product(
        &self,
        time: T::Real,
        state: &[T],
        _: &[T],
        y: &[T],
        v: &[T],
        output: &mut [T],
        scratch: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if self
            .problem
            .jacobian_product(time, state, y, output, scratch, cx)?
            .is_none()
        {
            return Ok(None);
        }
        for (out, &v) in output.iter_mut().zip(v) {
            *out = v - *out;
        }
        Ok(Some(()))
    }
}
impl<T: NumericScalar, P: OdeAdjointSystem<T>> DaeAdjointSystem<T>
    for OdeAdjointResidual<'_, T, P>
{
    fn parameters(&self) -> &[T::Real] {
        self.problem.parameters()
    }
    fn is_replayable(&self) -> bool {
        self.problem.is_replayable()
    }
    fn pullback_workspace_len(&self) -> usize {
        self.problem.pullback_workspace_len()
    }
    fn pullback(
        &self,
        time: T::Real,
        state: &[T],
        _: &[T],
        seed: &[T],
        y: &mut [T],
        v: &mut [T],
        p: &mut [T::Real],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.problem
            .pullback(time, state, seed, y, p, scratch, context)?;
        for out in y {
            *out = -*out;
        }
        for out in p {
            *out = -*out;
        }
        v.copy_from_slice(seed);
        Ok(())
    }
}
