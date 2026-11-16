//! Per-call context for linear-operator applications: solve accounting and an optional dense kernel.

use crate::{ExecutionContext, NumericScalar, SciOxideError, SolveContext};

/// A strided dense matrix passed to a [`DenseOperatorKernel`].
///
/// Entry `(row, column)` is `data[row * strides[0] + column * strides[1]]`.
/// The caller guarantees that every addressed value is finite and that the
/// real and imaginary parts of each are at most `maximum` in magnitude. The
/// kernel must still check that the addressed indices lie within `data`.
#[derive(Debug, Clone, Copy)]
pub struct DenseOperatorInput<'a, S: NumericScalar> {
    /// Scalar storage, possibly with padding between rows or columns.
    pub data: &'a [S],
    /// Number of rows and columns.
    pub shape: [usize; 2],
    /// Row and column strides, in elements.
    pub strides: [usize; 2],
    /// Whether the matrix entries are the complex conjugates of the stored values.
    pub conjugated: bool,
    /// Upper bound on the magnitude of the real and imaginary parts of every entry.
    pub maximum: S::Real,
}

/// A dense matrix-vector kernel that a library such as `scioxide-linalg` lends to operators.
///
/// Implementations must honor the cancellation, deadline and worker limits of
/// the [`ExecutionContext`] they receive, and must not count the operator
/// application again; the caller has already counted it.
pub trait DenseOperatorKernel<S: NumericScalar> {
    /// Returns the processor dispatch that the lending library already selected.
    ///
    /// Operators reuse it for their own kernels instead of detecting CPU
    /// features again. This method must not count work.
    ///
    /// ```
    /// fn backend<S: scioxide_core::NumericScalar>(
    ///     kernel: &dyn scioxide_core::DenseOperatorKernel<S>,
    /// ) -> scioxide_core::backend::KernelBackend {
    ///     kernel.dispatch().backend()
    /// }
    /// ```
    fn dispatch(&self) -> crate::backend::KernelDispatch;

    /// Writes `matrix · input` to `output`.
    ///
    /// `maximum` bounds the magnitude of the real and imaginary parts of
    /// `input`. `output` is solver scratch and may be changed on error.
    ///
    /// # Errors
    ///
    /// Implementations return an error for invalid storage or bounds, and
    /// propagate cancellation, executor and numerical failures.
    fn apply_dense_into(
        &self,
        matrix: DenseOperatorInput<'_, S>,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
}

/// The context of one linear-operator application: the solve's accounting and an optional dense kernel.
///
/// It dereferences to the underlying [`SolveContext`]. When one operator
/// applies another, pass [`Self::reborrow`]; creating or reborrowing a context
/// does not count work or allocate.
///
/// ```
/// fn apply<S: scioxide_core::NumericScalar>(
///     operator: &dyn scioxide_core::LinearOperator<S>,
///     x: &[S],
///     y: &mut [S],
///     scratch: &mut [S],
///     solve: &mut scioxide_core::SolveContext<'_, '_>,
/// ) -> Result<(), scioxide_core::SciOxideError> {
///     let mut context = scioxide_core::OperatorContext::new(solve);
///     operator.apply_into_scoped(x, y, scratch, &mut context)?.require()
/// }
/// ```
pub struct OperatorContext<'borrow, 'state, 'capability, S: NumericScalar> {
    solve: &'borrow mut SolveContext<'state, 'capability>,
    kernels: Option<&'borrow dyn DenseOperatorKernel<S>>,
}

impl<'borrow, 'state, 'capability, S: NumericScalar>
    OperatorContext<'borrow, 'state, 'capability, S>
{
    /// Creates a context without a dense kernel.
    pub fn new(solve: &'borrow mut SolveContext<'state, 'capability>) -> Self {
        Self {
            solve,
            kernels: None,
        }
    }

    /// Creates a context that lends `kernels` to the operators it reaches.
    pub fn with_kernels(
        solve: &'borrow mut SolveContext<'state, 'capability>,
        kernels: &'borrow dyn DenseOperatorKernel<S>,
    ) -> Self {
        Self {
            solve,
            kernels: Some(kernels),
        }
    }

    /// Returns the lent dense kernel, if any.
    pub fn kernels(&self) -> Option<&dyn DenseOperatorKernel<S>> {
        self.kernels
    }

    /// Returns a shorter-lived context with the same accounting and kernel, for a nested application.
    pub fn reborrow(&mut self) -> OperatorContext<'_, 'state, 'capability, S> {
        OperatorContext {
            solve: self.solve,
            kernels: self.kernels,
        }
    }
}

impl<'state, 'capability, S: NumericScalar> core::ops::Deref
    for OperatorContext<'_, 'state, 'capability, S>
{
    type Target = SolveContext<'state, 'capability>;
    fn deref(&self) -> &Self::Target {
        self.solve
    }
}

impl<S: NumericScalar> core::ops::DerefMut for OperatorContext<'_, '_, '_, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.solve
    }
}
