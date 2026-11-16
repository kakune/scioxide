//! Interfaces for nonlinear residual maps `F(x)` and their Jacobian-vector products.
//!
//! Nonlinear solvers in other SciOxide crates call these traits. An
//! implementation checks shapes before it writes, and then writes every output
//! component. Scratch is a buffer you provide whose length the implementation
//! reports and keeps fixed during a solve. The solver has already counted a
//! scoped call (see [solve accounting](crate#solve-accounting)); the
//! implementation charges only nested work. For complex scalars the Jacobian
//! must be complex-linear; write a non-holomorphic map in real coordinates.

use crate::{NumericScalar, SciOxideError, SolveContext};

/// A nonlinear map `F: Tⁿ → Tᵐ` with `n = input_dimension()` and `m = output_dimension()`.
pub trait ResidualOperator<T: NumericScalar> {
    /// Returns the number of input coordinates `n`.
    fn input_dimension(&self) -> usize;
    /// Returns the number of residual components `m`.
    fn output_dimension(&self) -> usize;
    /// Returns the scratch length that [`Self::residual_into`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`, or an error
    ///   describing an invalid configuration of the implementation.
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError>;
    /// Writes `F(point)` to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point`, `output` or `scratch` has the wrong length.
    /// - `DomainError` or a numerical error from evaluating `F`.
    fn residual_into(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError>;
    /// Writes `F(point)` to `output`, charging nested work to `context`.
    ///
    /// The calling solver has already counted this evaluation. The default
    /// calls [`Self::residual_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::residual_into`], and errors from nested work such
    /// as an exhausted budget.
    fn residual_into_scoped(
        &self,
        point: &[T],
        output: &mut [T],
        scratch: &mut [T],
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.residual_into(point, output, scratch)
    }
}

/// Jacobian-vector products `J(x) v` of a [`ResidualOperator`], without forming `J`.
///
/// The products may be analytic or come from automatic differentiation.
pub trait JacobianOperator<T: NumericScalar>: ResidualOperator<T> {
    /// Returns the scratch length that [`Self::jvp_into`] needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`, or an error
    ///   describing an invalid configuration of the implementation.
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError>;
    /// Writes `J(point) · direction` to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a slice has the wrong length.
    /// - `DomainError` or a numerical error from evaluating the product.
    fn jvp_into(
        &self,
        point: &[T],
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError>;
    /// Writes `J(point) · direction` to `output`, charging nested work to `context`.
    ///
    /// The calling solver has already counted this product. The default calls
    /// [`Self::jvp_into`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::jvp_into`], and errors from nested work.
    fn jvp_into_scoped(
        &self,
        point: &[T],
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.jvp_into(point, direction, output, scratch)
    }
}
