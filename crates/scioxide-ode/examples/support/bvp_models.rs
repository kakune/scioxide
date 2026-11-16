//! Analytic manufactured BVP used by the example.
use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_ode::{BvpJacobian, BvpSystem};
/// `y''=y`, with exact solution `amplitude*exp(x)` on `[0,1]`.
pub struct Exponential<T: NumericScalar> {
    pub amplitude: T,
}
impl<T: NumericScalar> BvpSystem<T> for Exponential<T> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[1];
        out[1] = y[0];
        Ok(())
    }
    fn boundary(
        &self,
        a: &[T],
        b: &[T],
        _: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = a[0] - self.amplitude;
        out[1] = b[0] - self.amplitude * T::from_real(T::Real::ONE.exp());
        Ok(())
    }
}
impl<T: NumericScalar> BvpJacobian<T> for Exponential<T> {
    fn rhs_jacobian(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        y: &mut [T],
        p: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        y.copy_from_slice(&[T::ZERO, T::ONE, T::ONE, T::ZERO]);
        p.fill(T::ZERO);
        Ok(())
    }
    fn boundary_jacobian(
        &self,
        _: &[T],
        _: &[T],
        _: &[T],
        a: &mut [T],
        b: &mut [T],
        p: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        a.copy_from_slice(&[T::ONE, T::ZERO, T::ZERO, T::ZERO]);
        b.copy_from_slice(&[T::ZERO, T::ZERO, T::ONE, T::ZERO]);
        p.fill(T::ZERO);
        Ok(())
    }
}
