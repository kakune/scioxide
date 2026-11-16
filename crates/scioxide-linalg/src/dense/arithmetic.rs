//! Elementwise arithmetic on dense views: sums, differences, scaling and `axpy`.
//!
//! ```
//! use scioxide_core::ExecutionContext;
//! use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace, Vector, VectorViewMut};
//! let input = Vector::from_vec(vec![1.0_f64, 2.0]);
//! let mut output = [3.0, 4.0];
//! input.view().axpy_into(
//!     2.0,
//!     &[1.0, 1.0][..],
//!     &LinalgExecutionOptions::new(),
//!     VectorViewMut::from_slice(&mut output),
//!     &mut LinalgWorkspace::new(),
//!     ExecutionContext::sequential_native(),
//! )?;
//! assert_eq!(output, [3.0, 5.0]);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```

use super::{Matrix, MatrixView, Vector, VectorView, VectorViewMut};
use crate::{Accuracy, DenseWorkspaceRequest, LinalgExecutionOptions, LinalgWorkspace};
use scioxide_core::{NumericScalar, SciOxideError};

impl<S: NumericScalar> VectorView<'_, S> {
    /// Returns the elementwise sum `x + y`, where `x` is `self` and `y` is `other`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a sum overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_add<'v>(
        &self,
        other: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.combine(other.into(), S::ONE)
    }
    /// Returns the elementwise difference `x − y`, where `x` is `self` and `y` is `other`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a difference overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_sub<'v>(
        &self,
        other: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.combine(other.into(), -S::ONE)
    }
    fn combine(&self, other: VectorView<'_, S>, sign: S) -> Result<Vector<S>, SciOxideError> {
        crate::checked::expect_shape([self.len()], [other.len()])?;
        let mut output = Vector::zeros(self.len())?;
        self.axpby_into(
            S::ONE,
            other,
            sign,
            &LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Returns `−x`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_neg(&self) -> Result<Vector<S>, SciOxideError> {
        self.try_scale(-S::ONE)
    }
    /// Returns `α x`.
    ///
    /// Each entry is computed as by [`VectorViewMut::scale_in_place`] with `Standard`
    /// accuracy, so a product fails only when it is not representable.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or an element is NaN or infinite.
    /// - `NumericalInstability` if a product is not representable.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn try_scale(&self, alpha: S) -> Result<Vector<S>, SciOxideError> {
        let mut output = self.try_to_vector()?;
        VectorViewMut::scale_in_place(
            alpha,
            &LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Writes `α x + y` to `output`, where `x` is `self`.
    ///
    /// This is [`axpby_into`](Self::axpby_into) with `β = 1`.
    ///
    /// # Errors
    ///
    /// As for [`axpby_into`](Self::axpby_into).
    pub fn axpy_into<'v, 'scope, 'capability>(
        &self,
        alpha: S,
        y: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.axpby_into(alpha, y, S::ONE, options, output, workspace, context)
    }
    /// Copies the elements of `self` into `output`.
    ///
    /// The stride and the conjugation flag of `self` are honored. `policy`
    /// decides whether NaN and infinite values are rejected or copied (see
    /// [`CopyPolicy`](crate::CopyPolicy)). The native copy needs no workspace.
    /// With [`CopyPolicy::RequireFinite`](crate::CopyPolicy::RequireFinite) and
    /// a BLAS provider attached, the copy goes through the provider and the
    /// workspace must hold [`DenseWorkspaceRequest::vector`]`(len, options)`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len() != self.len()`.
    /// - `NonFinite` with `RequireFinite` if an element is NaN or infinite.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{CopyPolicy, LinalgExecutionOptions, LinalgWorkspace, VectorView, VectorViewMut};
    /// let mut output = [0.0; 2];
    /// VectorView::from_slice(&[1.0_f64, 2.0]).copy_into(
    ///     CopyPolicy::RequireFinite,
    ///     &LinalgExecutionOptions::new(),
    ///     VectorViewMut::from_slice(&mut output),
    ///     &mut LinalgWorkspace::new(),
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert_eq!(output, [1.0, 2.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn copy_into<'scope, 'capability>(
        &self,
        policy: crate::CopyPolicy,
        options: &LinalgExecutionOptions,
        mut output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        crate::checked::expect_shape([self.len()], [output.len()])?;
        if policy == crate::CopyPolicy::RequireFinite {
            let _ = self.try_finite_dispatched(context.dispatch())?;
        }
        if options.byte_limit().is_some() {
            let _ = workspace.check(&crate::LinalgRequirements::new(options.byte_limit()))?;
        }
        if self.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "blas")]
        if policy == crate::CopyPolicy::RequireFinite
            && let Some(dispatch) = context.blas()
        {
            workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
                &DenseWorkspaceRequest::vector(self.len(), options).for_kernels(context),
            )?)?;
            let candidate = &mut workspace.scalar_mut(crate::kernel::CANDIDATE)[..self.len()];
            if crate::blas::dispatch::copy(
                *self,
                candidate,
                dispatch,
                context.accuracy(),
                crate::kernel::provider_worker_bound::<S>(
                    self.len(),
                    1,
                    1,
                    options.execution_policy(),
                    context,
                )?,
            )? == crate::blas::dispatch::BlasRoute::Provider
            {
                output
                    .iter_mut()
                    .zip(candidate.iter())
                    .for_each(|(out, &value)| *out = value);
                return Ok(());
            }
        }
        if self.stride == 1 && output.stride == 1 && !self.conjugated {
            output.data[..self.len()].copy_from_slice(&self.data[..self.len()]);
        } else {
            output
                .iter_mut()
                .zip(self.iter())
                .for_each(|(out, value)| *out = value);
        }
        Ok(())
    }
}

impl<S: NumericScalar> VectorViewMut<'_, S> {
    /// Overwrites `output` with `α x + output`.
    ///
    /// This is [`axpby_in_place`](Self::axpby_in_place) with `β = 1`.
    ///
    /// # Errors
    ///
    /// As for [`axpby_in_place`](Self::axpby_in_place).
    pub fn axpy_in_place<'v, 'scope, 'capability>(
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        Self::axpby_in_place(alpha, x, S::ONE, options, output, workspace, context)
    }
    /// Overwrites `output` with `α x + β output`.
    ///
    /// The [`Accuracy`](scioxide_core::Accuracy) level in `options` sets the
    /// error bound. The native path computes every result first and writes
    /// only if all are finite, so it needs no workspace. With a BLAS provider
    /// attached, the workspace must hold
    /// [`DenseWorkspaceRequest::in_place_vector`] with provider packing.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() != output.len()`.
    /// - `NonFinite` if `alpha`, `beta` or an element of `x` or `output` is NaN
    ///   or infinite.
    /// - `NumericalInstability` if a result element overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn axpby_in_place<'v, 'scope, 'capability>(
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        crate::checked::expect_shape([output.len()], [x.len()])?;
        crate::checked::validate_input(alpha, "alpha")?;
        crate::checked::validate_input(beta, "beta")?;
        let x = x.try_finite_dispatched(context.dispatch())?;
        let y = output.as_view().try_finite_dispatched(context.dispatch())?;
        let affine = crate::kernel::standard::Affine::new(alpha, x.maximum, beta, y.maximum)?;
        let request =
            DenseWorkspaceRequest::in_place_vector(output.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            if beta == S::ONE {
                let candidate = &mut workspace.scalar_mut(crate::kernel::CANDIDATE)[..output.len()];
                candidate
                    .iter_mut()
                    .zip(y.iter())
                    .for_each(|(out, value)| *out = value);
                if crate::blas::dispatch::axpy(
                    alpha,
                    x.as_view(),
                    candidate,
                    dispatch,
                    context.accuracy(),
                    crate::kernel::provider_worker_bound::<S>(
                        output.len(),
                        1,
                        1,
                        options.execution_policy(),
                        context,
                    )?,
                )? == crate::blas::dispatch::BlasRoute::Provider
                {
                    output
                        .iter_mut()
                        .zip(candidate.iter())
                        .for_each(|(out, &value)| *out = value);
                    return Ok(());
                }
            } else {
                let _ = crate::blas::dispatch::unsupported(dispatch)?;
            }
        }
        if context.accuracy() == Accuracy::Standard {
            if affine.publication != crate::kernel::standard::Publication::Direct {
                for (x, y) in x.iter().zip(y.iter()) {
                    crate::checked::finite_candidate(affine.apply(x, y))?;
                }
            }
            context.execution().check()?;
            output
                .iter_mut()
                .zip(x.iter())
                .for_each(|(out, x)| *out = affine.apply(x, *out));
        } else {
            for (x, y) in x.iter().zip(y.iter()) {
                let _ = super::kernel::axpby_value(alpha, x, beta, y)?;
            }
            context.execution().check()?;
            // The identical operands and deterministic scalar contract make this
            // second pass infallible after the complete validation pass above.
            for (out, x) in output.iter_mut().zip(x.iter()) {
                *out = super::kernel::axpby_value(alpha, x, beta, *out)?;
            }
        }
        Ok(())
    }
    /// Overwrites `output` with `α output`.
    ///
    /// Under `Standard` [`Accuracy`](scioxide_core::Accuracy) each entry is the ordinary
    /// product, recomputed exactly and rounded once when one of its real partial products
    /// overflows or underflows; under `Exact` each component is rounded once. Either way a
    /// product fails only when the exact result is not representable. The native path
    /// computes every product first and writes only if all are finite, so it needs no
    /// workspace. With a BLAS provider attached, the workspace must hold
    /// [`DenseWorkspaceRequest::in_place_vector`] with provider packing.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or an element is NaN or infinite.
    /// - `NumericalInstability` if a product is not representable.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn scale_in_place<'scope, 'capability>(
        alpha: S,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        crate::checked::validate_input(alpha, "scale")?;
        let input = output.as_view().try_finite_dispatched(context.dispatch())?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::in_place_vector(output.len(), options).for_kernels(context),
        )?)?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            let candidate = &mut workspace.scalar_mut(crate::kernel::CANDIDATE)[..output.len()];
            candidate
                .iter_mut()
                .zip(input.iter())
                .for_each(|(out, value)| *out = value);
            if crate::blas::dispatch::scale(
                alpha,
                candidate,
                dispatch,
                context.accuracy(),
                crate::kernel::provider_worker_bound::<S>(
                    output.len(),
                    1,
                    1,
                    options.execution_policy(),
                    context,
                )?,
            )? == crate::blas::dispatch::BlasRoute::Provider
            {
                output
                    .iter_mut()
                    .zip(candidate.iter())
                    .for_each(|(out, &value)| *out = value);
                return Ok(());
            }
        }
        let accuracy = context.accuracy();
        for value in input.iter() {
            let _ = scioxide_core::backend::standard::product(value, alpha, accuracy)?;
        }
        context.execution().check()?;
        for value in output.iter_mut() {
            *value = scioxide_core::backend::standard::product(*value, alpha, accuracy)?;
        }
        Ok(())
    }
}

impl<S: NumericScalar> MatrixView<'_, S> {
    /// Returns the elementwise sum `A + B`, where `A` is `self` and `B` is `other`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shapes differ.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a sum overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be allocated.
    pub fn try_add<'m>(
        &self,
        other: impl Into<MatrixView<'m, S>>,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.zip_entries(other.into(), |a, b| a + b)
    }
    /// Returns the elementwise difference `A − B`, where `A` is `self` and `B` is `other`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shapes differ.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a difference overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be allocated.
    pub fn try_sub<'m>(
        &self,
        other: impl Into<MatrixView<'m, S>>,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.zip_entries(other.into(), |a, b| a - b)
    }
    /// Returns `−A`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be allocated.
    pub fn try_neg(&self) -> Result<Matrix<S>, SciOxideError> {
        self.try_scale(-S::ONE)
    }
    /// Returns `α A`.
    ///
    /// Each entry is computed as by [`VectorViewMut::scale_in_place`] with `Standard`
    /// accuracy, so a product fails only when it is not representable.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or an element is NaN or infinite.
    /// - `NumericalInstability` if a product is not representable.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be allocated.
    pub fn try_scale(&self, alpha: S) -> Result<Matrix<S>, SciOxideError> {
        crate::checked::validate_input(alpha, "scale")?;
        let finite = self.try_finite()?;
        let len = scioxide_core::backend::checked_product(
            &[self.rows(), self.cols()],
            scioxide_core::quantity!("scaled_matrix"),
        )?;
        let mut values = scioxide_core::backend::try_with_capacity(
            len,
            scioxide_core::quantity!("scaled_matrix"),
        )?;
        for i in 0..len {
            values.push(scioxide_core::backend::standard::product(
                finite
                    .as_view()
                    .get_element(i / self.cols(), i % self.cols()),
                alpha,
                Accuracy::Standard,
            )?);
        }
        Matrix::from_vec(self.rows(), self.cols(), values)
    }
    fn zip_entries(
        &self,
        other: MatrixView<'_, S>,
        map: impl Fn(S, S) -> S,
    ) -> Result<Matrix<S>, SciOxideError> {
        if (self.rows(), self.cols()) != (other.rows(), other.cols()) {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows(), self.cols()],
                [other.rows(), other.cols()],
            ));
        }
        let left = self.try_finite()?;
        let right = other.try_finite()?;
        let len = scioxide_core::backend::checked_product(
            &[self.rows(), self.cols()],
            scioxide_core::quantity!("matrix_entries"),
        )?;
        let mut values = scioxide_core::backend::try_with_capacity(
            len,
            scioxide_core::quantity!("matrix_entries"),
        )?;
        for i in 0..len {
            let row = i / self.cols();
            let col = i % self.cols();
            values.push(crate::checked::finite_candidate(map(
                left.as_view().get_element(row, col),
                right.as_view().get_element(row, col),
            ))?);
        }
        Matrix::from_vec(self.rows(), self.cols(), values)
    }
}
