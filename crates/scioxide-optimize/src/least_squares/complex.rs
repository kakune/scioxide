//! Complex residuals as a real least-squares problem.
use crate::LeastSquaresProblem;
use core::cell::RefCell;

use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;

/// Turns a residual `r: ℂⁿ → ℂᵐ` into a [`LeastSquaresProblem`] in real
/// coordinates.
///
/// Parameters and residuals are interleaved as
/// `[Re z₀, Im z₀, Re z₁, Im z₁, …]`, so the real problem has `2n` parameters
/// and `2m` residual components. Each complex residual is one observation of
/// the loss, so a robust loss acts on `|rᵢ|`. Every residual has unit weight
/// and unit scale. The residual need not be
/// holomorphic. An optional analytic Jacobian, set with
/// [`Self::with_jacobian`], is the `2m × 2n` real Jacobian with respect to the
/// interleaved coordinates.
///
/// The adapter converts values in the two scratch slices you pass to
/// [`Self::new`] and does not allocate. It is not `Sync`, and a reentrant
/// call from inside the residual callback returns `InvalidInput`.
pub struct ComplexLeastSquares<'a, S: RealScalar + NumericScalar<Real = S>, F> {
    residual: F,
    jacobian: Option<&'a RealJacobian<'a, S>>,
    scratch: RefCell<ComplexScratch<'a, S>>,
    parameters: usize,
    residuals: usize,
}
type RealJacobian<'a, S> = dyn Fn(&[Complex<S>], &mut Matrix<S>) -> Result<(), SciOxideError> + 'a;
struct ComplexScratch<'a, S: RealScalar> {
    x: &'a mut [Complex<S>],
    residual: &'a mut [Complex<S>],
}
impl<'a, S, F> ComplexLeastSquares<'a, S, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[Complex<S>], &mut [Complex<S>]) -> Result<(), SciOxideError>,
{
    /// Creates the adapter. The lengths of `parameter_scratch` and
    /// `residual_scratch` set the complex dimensions `n` and `m`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if `2n` or `2m` overflows `usize`.
    pub fn new(
        residual: F,
        parameter_scratch: &'a mut [Complex<S>],
        residual_scratch: &'a mut [Complex<S>],
    ) -> Result<Self, SciOxideError> {
        let parameters = parameter_scratch.len().checked_mul(2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("complex_parameters"))
        })?;
        let residuals = residual_scratch.len().checked_mul(2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("complex_residuals"))
        })?;
        Ok(Self {
            residual,
            jacobian: None,
            scratch: RefCell::new(ComplexScratch {
                x: parameter_scratch,
                residual: residual_scratch,
            }),
            parameters,
            residuals,
        })
    }
    /// Sets the analytic Jacobian with respect to the interleaved real
    /// coordinates.
    #[must_use]
    pub fn with_jacobian(mut self, jacobian: &'a RealJacobian<'a, S>) -> Self {
        self.jacobian = Some(jacobian);
        self
    }
}
fn borrow_error() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("self"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "callback scratch is already borrowed",
    )
}
fn unpack<S: RealScalar>(input: &[S], output: &mut [Complex<S>]) -> Result<(), SciOxideError> {
    if input.len() != output.len() * 2 {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(output.len() * 2),
            scioxide_core::Dimensions::one(input.len()),
        ));
    }
    for (value, pair) in output.iter_mut().zip(input.chunks_exact(2)) {
        *value = Complex::new(pair[0], pair[1]);
    }
    Ok(())
}
impl<S, F> LeastSquaresProblem<S> for ComplexLeastSquares<'_, S, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[Complex<S>], &mut [Complex<S>]) -> Result<(), SciOxideError>,
{
    fn param_dim(&self) -> usize {
        self.parameters
    }
    fn residual_dim(&self) -> usize {
        self.residuals
    }
    fn residual_group_size(&self) -> usize {
        2
    }
    fn residual_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        if output.len() != self.residuals {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.residuals),
                scioxide_core::Dimensions::one(output.len()),
            ));
        }
        let mut scratch = self.scratch.try_borrow_mut().map_err(|_| borrow_error())?;
        let ComplexScratch {
            x: parameters,
            residual,
        } = &mut *scratch;
        unpack(x, parameters)?;
        (self.residual)(parameters, residual)?;
        for (pair, value) in output.chunks_exact_mut(2).zip(residual.iter()) {
            pair.copy_from_slice(&[value.re, value.im]);
        }
        // The engine checks the start and rejects non-finite trial residuals.
        Ok(())
    }
    fn jacobian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        if x.len() != self.parameters {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.parameters),
                scioxide_core::Dimensions::one(x.len()),
            ));
        }
        if output.rows() != self.residuals || output.cols() != self.parameters {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(self.residuals, self.parameters),
                scioxide_core::Dimensions::two(output.rows(), output.cols()),
            ));
        }
        let Some(callback) = self.jacobian else {
            return Ok(false);
        };
        let mut scratch = self.scratch.try_borrow_mut().map_err(|_| borrow_error())?;
        unpack(x, scratch.x)?;
        callback(scratch.x, output)?;
        Ok(true)
    }
}
