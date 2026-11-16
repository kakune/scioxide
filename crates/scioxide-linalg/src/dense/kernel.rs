//! Products, reductions and copies on dense matrix and vector views.
use scioxide_core::Accuracy;

use scioxide_core::AlgebraicScalar;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{
    AdjointOperator, BlockOperator, Dimensions, DomainReason, ExecutionContext, ExecutionPolicy,
    LinearOperator, NumericScalar, ProductSum, RealScalar, ScaledNorm, SciOxideError,
};

use super::view::{FiniteMatrixView, MatrixView, MatrixViewMut, VectorView, VectorViewMut};
use super::{Matrix, Vector};
use crate::{KernelContext, LinalgWorkspace, kernel};

impl<S: NumericScalar> VectorView<'_, S> {
    /// Computes the unconjugated dot product `xᵀy = Σᵢ xᵢ yᵢ` of `self` and `other`.
    ///
    /// The product uses `Standard` accuracy, whose error bound is stated on
    /// [`Accuracy`]; [`dotu_scoped`](Self::dotu_scoped) with
    /// [`Accuracy::Exact`] in its options rounds the result correctly. Use
    /// [`dotc_left`](Self::dotc_left) for the complex inner product `xᴴy`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if the result overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::VectorView;
    /// let x = VectorView::from_slice(&[1.0_f64, 2.0]);
    /// // 1·1 + 2·2 = 5
    /// assert!((x.dotu(x)? - 5.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn dotu<'v>(&self, other: impl Into<VectorView<'v, S>>) -> Result<S, SciOxideError> {
        self.dotu_scoped(
            other,
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`dotu`](Self::dotu) with `options` in the caller's execution context, which can
    /// select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider or stop the
    /// computation.
    ///
    /// # Errors
    ///
    /// The errors of [`dotu`](Self::dotu), and:
    ///
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn dotu_scoped<'v, 'scope, 'a>(
        &self,
        other: impl Into<VectorView<'v, S>>,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<S, SciOxideError> {
        let other = other.into();
        let context = context.into().with_options(options)?.kernel();
        crate::reduction::dotu_with_context(*self, other, context)
    }

    /// Computes the inner product `xᴴy = Σᵢ conj(xᵢ) yᵢ`, conjugating `self`.
    ///
    /// For real scalars it equals [`dotu`](Self::dotu). Accuracy is as for
    /// [`dotu`](Self::dotu).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if the result overflows.
    pub fn dotc_left<'v>(&self, other: impl Into<VectorView<'v, S>>) -> Result<S, SciOxideError> {
        self.dotc_left_scoped(
            other,
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`dotc_left`](Self::dotc_left) with `options` in the caller's execution context,
    /// which can select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider
    /// or stop the computation.
    ///
    /// # Errors
    ///
    /// The errors of [`dotc_left`](Self::dotc_left), and:
    ///
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn dotc_left_scoped<'v, 'scope, 'a>(
        &self,
        other: impl Into<VectorView<'v, S>>,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<S, SciOxideError> {
        let other = other.into();
        let context = context.into().with_options(options)?.kernel();
        crate::reduction::dot_with_context(*self, other, true, context)
    }

    /// Computes the Euclidean norm `‖x‖₂ = (Σᵢ |xᵢ|²)^(1/2)`.
    ///
    /// The sum of squares is rescaled to avoid overflow and underflow, so the
    /// result is accurate whenever `‖x‖₂` itself is representable. The sum
    /// uses `Standard` [`Accuracy`]; [`norm2_scoped`](Self::norm2_scoped) can
    /// select another level.
    ///
    /// ```
    /// use scioxide_linalg::VectorView;
    /// let x = VectorView::from_slice(&[3.0_f64, 4.0]);
    /// assert!((x.norm2()? - 5.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if the norm overflows.
    pub fn norm2(&self) -> Result<S::Real, SciOxideError> {
        self.norm2_scoped(
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`norm2`](Self::norm2) with `options` in the caller's execution context, which can
    /// select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider or stop the
    /// computation.
    ///
    /// # Errors
    ///
    /// The errors of [`norm2`](Self::norm2), and:
    ///
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn norm2_scoped<'scope, 'a>(
        &self,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<S::Real, SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        crate::reduction::l2_norm_with_context(*self, context)
    }

    /// Returns the unit vector `x / ‖x‖₂`.
    ///
    /// Each element is divided by the scale and the root of the scaled sum of
    /// squares separately, so the result is correct even when `‖x‖₂` overflows.
    ///
    /// # Errors
    ///
    /// - `DomainError` if every element is zero.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a result element is not representable.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn normalize(&self) -> Result<Vector<S>, SciOxideError> {
        let state = normalized_state(*self, ExecutionContext::sequential_native())?;
        let mut data = try_with_capacity(self.len, scioxide_core::quantity!("normalized_vector"))?;
        self.iter()
            .try_for_each(|value| normalized_value(value, state).map(|value| data.push(value)))?;
        Ok(Vector::from_vec(data))
    }

    /// Writes the unit vector `x / ‖x‖₂` to `output`.
    ///
    /// This is the `_into` form of [`normalize`](Self::normalize). The
    /// workspace needs `self.len()` scalars.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len() != self.len()`.
    /// - `DomainError` if every element is zero.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a result element is not representable.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn normalize_into<'scope, 'capability>(
        &self,
        options: &crate::LinalgExecutionOptions,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        crate::checked::expect_shape([self.len], [output.len])?;
        let request = crate::DenseWorkspaceRequest::vector(self.len, options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        validate_vector_scratch(self.len, scratch.len())?;
        let state = normalized_state(*self, context.execution())?;
        self.iter()
            .zip(scratch.iter_mut())
            .try_for_each(|(value, candidate)| {
                normalized_value(value, state).map(|value| *candidate = value)
            })?;
        commit_vector(output, &scratch[..self.len]);
        Ok(())
    }

    /// Returns the cross product `x × y` of two vectors of length 3.
    ///
    /// Each component, such as `x₁y₂ − x₂y₁`, is accumulated exactly and
    /// rounded once.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` unless both vectors have length 3.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a component overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    pub fn cross<'v>(
        &self,
        other: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        let other = other.into();

        validate_cross_dimensions(*self, other)?;
        let mut data = try_with_capacity(3, scioxide_core::quantity!("vector_cross_product"))?;
        cross_candidates(*self, other, |value| data.push(value))?;
        Ok(Vector::from_vec(data))
    }

    /// Writes the cross product `x × y` of two vectors of length 3 to `output`.
    ///
    /// This is the `_into` form of [`cross`](Self::cross). The workspace needs
    /// three scalars.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` unless both inputs and `output` have length 3.
    /// - `NonFinite` if an element is NaN or infinite.
    /// - `NumericalInstability` if a component overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn cross_into<'v, 'scope, 'capability>(
        &self,
        other: impl Into<VectorView<'v, S>>,
        options: &crate::LinalgExecutionOptions,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let other = other.into();

        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        validate_cross_dimensions(*self, other)?;
        crate::checked::expect_shape([3], [output.len])?;
        let request = crate::DenseWorkspaceRequest::vector(3, options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        validate_vector_scratch(3, scratch.len())?;
        let mut next = 0;
        cross_candidates(*self, other, |value| {
            scratch[next] = value;
            next += 1;
        })?;
        commit_vector(output, &scratch[..3]);
        Ok(())
    }

    /// Writes `α x + β y` to `output`, where `x` is `self`.
    ///
    /// The [`Accuracy`] level in `options` sets the error bound. With `Exact`
    /// each element is accumulated exactly and rounded once. The workspace needs
    /// `self.len()` scalars.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `y` or `output` differs in length from `self`.
    /// - `NonFinite` if `alpha`, `beta` or an element is NaN or infinite.
    /// - `NumericalInstability` if a result element overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn axpby_into<'v, 'scope, 'a>(
        &self,
        alpha: S,
        y: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &crate::LinalgExecutionOptions,
        mut output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let y = y.into();

        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        validate_axpby_dimensions(*self, y, output.len)?;
        let request = crate::DenseWorkspaceRequest::vector(self.len, options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        validate_vector_scratch(self.len, scratch.len())?;
        crate::checked::validate_input(alpha, "alpha")?;
        crate::checked::validate_input(beta, "beta")?;
        let context: crate::KernelContext<'_> = context;
        let left = self.try_finite_dispatched(context.dispatch())?;
        let right = y.try_finite_dispatched(context.dispatch())?;
        context.execution().check()?;

        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            // CBLAS AXPY only computes `y += alpha * x`, so only `beta == 1`
            // maps onto it; `y` is staged in scratch because AXPY overwrites
            // its output operand. Other betas still consult the fallback
            // policy so an `Error` policy reports the missing provider route.
            if beta == S::ONE {
                right
                    .iter()
                    .zip(scratch.iter_mut())
                    .for_each(|(value, candidate)| *candidate = value);
                if crate::blas::dispatch::axpy(
                    alpha,
                    left.as_view(),
                    &mut scratch[..self.len],
                    dispatch,
                    context.accuracy(),
                    crate::kernel::provider_worker_bound::<S>(
                        self.len,
                        1,
                        1,
                        options.execution_policy(),
                        context,
                    )?,
                )? == crate::blas::dispatch::BlasRoute::Provider
                {
                    commit_vector(output, &scratch[..self.len]);
                    return Ok(());
                }
            } else {
                let _ = crate::blas::dispatch::unsupported(dispatch)?;
            }
        }

        let standard =
            crate::kernel::standard::Affine::new(alpha, left.maximum, beta, right.maximum)?;
        if context.accuracy() == Accuracy::Standard
            && standard.publication == crate::kernel::standard::Publication::Direct
        {
            // All fallible checks precede this pass. The range proof permits
            // direct publication, including portable and strided output.
            if let (Some(x), Some(y), Some(out)) = (
                left.as_view().as_contiguous_slice(),
                right.as_view().as_contiguous_slice(),
                output.as_contiguous_slice_mut(),
            ) && crate::kernel::arch::affine(alpha, x, beta, y, out, context.dispatch())
            {
                return Ok(());
            }
            for ((x, y), out) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
                *out = standard.apply(x, y);
            }
            return Ok(());
        }
        left.iter()
            .zip(right.iter())
            .zip(scratch.iter_mut())
            .try_for_each(|((x, y), candidate)| {
                *candidate = match context.accuracy() {
                    Accuracy::Exact => axpby_value(alpha, x, beta, y)?,
                    Accuracy::Standard => {
                        let value = standard.apply(x, y);
                        if standard.publication == crate::kernel::standard::Publication::Direct {
                            value
                        } else {
                            crate::checked::finite_candidate(value)?
                        }
                    }
                };
                Ok::<(), SciOxideError>(())
            })?;
        commit_vector(output, &scratch[..self.len]);
        Ok(())
    }
}

pub(super) fn axpby_value<S: NumericScalar>(
    alpha: S,
    x: S,
    beta: S,
    y: S,
) -> Result<S, SciOxideError> {
    let mut sum = ProductSum::new();
    sum.add_product(alpha, x)?;
    sum.add_product(beta, y)?;
    sum.finish()
}

fn normalized_state<S: NumericScalar>(
    values: VectorView<'_, S>,
    context: ExecutionContext<'_>,
) -> Result<ScaledNorm<S::Real>, SciOxideError> {
    let state = crate::reduction::scaled_norm_state_with_context(values, context)?;
    if state.scale() == S::Real::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("self"),
            DomainReason::Undefined,
            "cannot normalize a zero vector",
        ));
    }
    Ok(state)
}

fn normalized_value<S: NumericScalar>(
    value: S,
    state: ScaledNorm<S::Real>,
) -> Result<S, SciOxideError> {
    let reciprocal = (S::Real::ONE / state.scale()) / state.sum_squares().sqrt();
    if reciprocal.is_normal() {
        return crate::checked::finite_candidate(value * S::from_real(reciprocal));
    }
    // The norm is `scale * sqrt(sum_squares)` in LASSQ form: every component
    // is at most `scale` in magnitude and `sum_squares >= 1`. Dividing by the
    // two factors separately keeps intermediates of order one, even when the
    // product `scale * sqrt(...)` itself would overflow.
    let scaled = crate::checked::div(value, S::from_real(state.scale()))?;
    crate::checked::div(scaled, S::from_real(state.sum_squares().sqrt()))
}

fn validate_cross_dimensions<S: NumericScalar>(
    left: VectorView<'_, S>,
    right: VectorView<'_, S>,
) -> Result<(), SciOxideError> {
    if left.len != 3 || right.len != 3 {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(3),
            Dimensions::two(left.len, right.len),
        ));
    }
    Ok(())
}

fn cross_candidates<S: NumericScalar>(
    left: VectorView<'_, S>,
    right: VectorView<'_, S>,
    mut accept: impl FnMut(S),
) -> Result<(), SciOxideError> {
    let left = left.try_finite()?.as_view();
    let right = right.try_finite()?.as_view();
    // Each component is a difference of two products, the classic
    // cancellation case; the exact accumulator rounds it only once.
    [((1, 2), (2, 1)), ((2, 0), (0, 2)), ((0, 1), (1, 0))]
        .into_iter()
        .try_for_each(|(positive, negative)| {
            let mut difference = ProductSum::new();
            difference.add_product(left.get_element(positive.0), right.get_element(positive.1))?;
            difference
                .subtract_product(left.get_element(negative.0), right.get_element(negative.1))?;
            accept(difference.finish()?);
            Ok(())
        })
}

fn validate_axpby_dimensions<S: NumericScalar>(
    x: VectorView<'_, S>,
    y: VectorView<'_, S>,
    output_len: usize,
) -> Result<(), SciOxideError> {
    if x.len != y.len {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(x.len),
            Dimensions::one(y.len),
        ));
    }
    crate::checked::expect_shape([x.len], [output_len])
}

fn validate_vector_scratch(required: usize, actual: usize) -> Result<(), SciOxideError> {
    if actual < required {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(required),
            Dimensions::one(actual),
        ));
    }
    Ok(())
}

fn commit_vector<S: NumericScalar>(mut output: VectorViewMut<'_, S>, candidate: &[S]) {
    output
        .iter_mut()
        .zip(candidate.iter().copied())
        .for_each(|(output, candidate)| *output = candidate);
}

fn product_vector_into<S: NumericScalar>(
    view: MatrixView<'_, S>,
    input: VectorView<'_, S>,
    options: &crate::LinalgExecutionOptions,
    output: VectorViewMut<'_, S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    context.execution().check()?;
    validate_mat_vec_output(view, input, &output)?;
    let requirements = LinalgWorkspace::<S>::workspace_requirements(
        &crate::DenseWorkspaceRequest::vector(view.rows, options).for_kernels(context),
    )?;
    workspace.try_reserve(&requirements)?;
    let scratch = &mut workspace.scalar_mut(kernel::CANDIDATE)[..view.rows];
    kernel::level2::gemv_candidate(view, input, scratch, options.execution_policy(), context)?;

    context.execution().check()?;
    commit_vector(output, scratch);
    Ok(())
}

impl<S: NumericScalar> MatrixView<'_, S> {
    /// Returns the matrix-vector product `A x`, where `A` is `self`.
    ///
    /// This is the owned form of [`mul_vec_into`](Self::mul_vec_into). It runs
    /// sequentially (see [Execution](crate#execution)). Accuracy is as for
    /// [`mul_vec_into`](Self::mul_vec_into).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input.len() != self.cols()`.
    /// - `NonFinite` if an element of `A` or `x` is NaN or infinite.
    /// - `NumericalInstability` if a result element overflows.
    /// - `AllocationFailure` if the scratch or result cannot be allocated.
    pub fn mul_vec<'b>(
        &self,
        input: impl Into<VectorView<'b, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.mul_vec_with_options(input, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`mul_vec`](Self::mul_vec) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`mul_vec`](Self::mul_vec), and:
    ///
    /// - `ResourceLimit` if the scratch exceeds the byte limit in `options`.
    /// - `CapabilityUnavailable` if `options` sets a time limit.
    pub fn mul_vec_with_options<'b>(
        &self,
        input: impl Into<VectorView<'b, S>>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Vector<S>, SciOxideError> {
        let options = options.with_execution_policy(ExecutionPolicy::Sequential);
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let input = input.into();
        validate_mat_vec_input(*self, input)?;
        let mut workspace = LinalgWorkspace::<S>::workspace(
            &crate::DenseWorkspaceRequest::vector(self.rows, &options),
        )?;
        let mut output = Vector::zeros(self.rows)?;
        self.mul_vec_into(input, &options, output.view_mut(), &mut workspace, context)?;
        Ok(output)
    }

    /// Writes the matrix-vector product `A x` to `output`, where `A` is `self`.
    ///
    /// Strides and the conjugation flag of `self` and `input` are honored. With
    /// [`Accuracy::Exact`] in `options` each result element is correctly
    /// rounded. With the default `Standard`, the error bound is stated on
    /// [`Accuracy`]. The workspace needs `self.rows()` scalars.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input.len() != self.cols()` or
    ///   `output.len() != self.rows()`.
    /// - `NonFinite` if an element of `A` or `x` is NaN or infinite.
    /// - `NumericalInstability` if a result element overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn mul_vec_into<'b, 'scope, 'capability>(
        &self,
        input: impl Into<VectorView<'b, S>>,
        options: &crate::LinalgExecutionOptions,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        product_vector_into(*self, input.into(), options, output, workspace, context)
    }

    /// Copies the elements of `self` into `output`.
    ///
    /// Strides and the conjugation flag are honored, so applying
    /// [`transpose`](Self::transpose), [`conjugate`](Self::conjugate) or
    /// [`adjoint`](Self::adjoint) first copies the transformed matrix. `policy`
    /// decides whether NaN and infinite values are rejected or copied (see
    /// [`CopyPolicy`](crate::CopyPolicy)). No workspace is needed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have the shape of `self`.
    /// - `NonFinite` with [`CopyPolicy::RequireFinite`](crate::CopyPolicy::RequireFinite)
    ///   if an element is NaN or infinite.
    /// - `ExecutionStopped` or `CapabilityUnavailable` from the execution context
    ///   (see [Execution](crate#execution)).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{CopyPolicy, LinalgExecutionOptions, MatrixView, MatrixViewMut};
    /// let source = MatrixView::from_slice(1, 2, &[1.0_f64, 2.0])?;
    /// let mut output = [0.0; 2];
    /// source.transpose().copy_into(
    ///     CopyPolicy::RequireFinite,
    ///     &LinalgExecutionOptions::new(),
    ///     MatrixViewMut::from_slice(2, 1, &mut output)?,
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert_eq!(output, [1.0, 2.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn copy_into<'scope, 'capability>(
        &self,
        policy: crate::CopyPolicy,
        options: &crate::LinalgExecutionOptions,
        output: MatrixViewMut<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        kernel::level2::copy_into(*self, output, policy)
    }

    /// Returns the matrix product `A B`, where `A` is `self` and `B` is `other`.
    ///
    /// This is the owned form of [`mul_mat_into`](Self::mul_mat_into). It runs
    /// sequentially (see [Execution](crate#execution)). Strides and the
    /// conjugation flag of both views are honored. Accuracy is as for
    /// [`mul_mat_into`](Self::mul_mat_into).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self.cols() != other.rows()`.
    /// - `NonFinite` if an element of `A` or `B` is NaN or infinite.
    /// - `NumericalInstability` if a result element overflows.
    /// - `SizeOverflow` if the result or scratch size does not fit in `usize`.
    /// - `AllocationFailure` if the scratch or result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// let a = Matrix::<f64>::identity(2)?;
    /// let product = a.view().mul_mat(&a)?;
    /// assert_eq!(product, a);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn mul_mat<'b>(
        &self,
        other: impl Into<MatrixView<'b, S>>,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.mul_mat_with_options(other, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`mul_mat`](Self::mul_mat) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`mul_mat`](Self::mul_mat), and:
    ///
    /// - `ResourceLimit` if the scratch exceeds the byte limit in `options`.
    /// - `CapabilityUnavailable` if `options` sets a time limit.
    pub fn mul_mat_with_options<'b>(
        &self,
        other: impl Into<MatrixView<'b, S>>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = options.with_execution_policy(ExecutionPolicy::Sequential);
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let other = other.into();
        validate_mat_mat_inputs(*self, other)?;
        let request =
            crate::DenseWorkspaceRequest::product(self.rows, self.cols, other.cols, &options);
        let mut workspace = LinalgWorkspace::<S>::workspace(&request)?;
        let mut output = Matrix::zeros(self.rows, other.cols)?;
        self.mul_mat_into(other, &options, output.view_mut(), &mut workspace, context)?;
        Ok(output)
    }

    /// Writes the matrix product `A B` to `output`, where `A` is `self` and `B` is `other`.
    ///
    /// Strides and the conjugation flag of both views are honored. With
    /// [`Accuracy::Exact`] in `options` each result element is correctly
    /// rounded. With the default `Standard`, the error bound is stated on
    /// [`Accuracy`]. With the `blas` feature, a provider attached to `context`
    /// may compute the product when it supports the requested accuracy.
    /// Size the workspace with
    /// [`DenseWorkspaceRequest::product`](crate::DenseWorkspaceRequest::product).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self.cols() != other.rows()` or `output` is not
    ///   `self.rows() × other.cols()`.
    /// - `NonFinite` if an element of `A` or `B` is NaN or infinite.
    /// - `NumericalInstability` if a result element overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped`, `CapabilityUnavailable` or `CapabilityFailure` from the
    ///   execution context or an attached BLAS provider (see [Execution](crate#execution)).
    pub fn mul_mat_into<'b, 'scope, 'capability>(
        &self,
        other: impl Into<MatrixView<'b, S>>,
        options: &crate::LinalgExecutionOptions,
        output: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let other = other.into();
        kernel::validate_dimensions(self.rows, self.cols, other.rows, other.cols, &output)?;
        let request =
            crate::DenseWorkspaceRequest::product(self.rows, self.cols, other.cols, options)
                .for_kernels(context);
        let requirements = LinalgWorkspace::<S>::workspace_requirements(&request)?;
        if self.rows == 0 || other.cols == 0 {
            workspace.try_reserve(&requirements)?;
            return Ok(());
        }
        let left = self.try_finite_dispatched(context.dispatch())?;
        let right = other.try_finite_dispatched(context.dispatch())?;
        workspace.try_reserve(&requirements)?;
        kernel::gemm_bounded(
            left,
            right,
            output,
            kernel::GemmConfig::ordinary(),
            options.execution_policy(),
            workspace,
            context,
        )
    }
}

impl<S: NumericScalar> FiniteMatrixView<'_, S> {
    /// Applies the matrix for a solver that has already checked the input.
    /// The solver handles the output on error.
    pub(crate) fn apply_workspace(
        &self,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let input = VectorView::from_slice(input);
        validate_mat_vec_output(self.as_view(), input, &VectorViewMut::from_slice(output))?;
        crate::checked::validate_input(S::from_real(maximum), "input bound")?;
        if maximum < S::Real::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("maximum"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "negative input bound",
            ));
        }
        kernel::level2::gemv_bounded(
            *self,
            crate::FiniteVectorView::from_validated(input, maximum),
            output,
            context.policy(),
            context,
        )
    }

    fn apply_operator(
        &self,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        context: &scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<(), SciOxideError> {
        if let Some(kernels) = context.kernels() {
            let matrix = self.as_view();
            kernels.apply_dense_into(
                scioxide_core::DenseOperatorInput {
                    data: matrix.data,
                    shape: [matrix.rows, matrix.cols],
                    strides: [matrix.row_stride, matrix.col_stride],
                    conjugated: matrix.conjugated,
                    maximum: self.maximum,
                },
                input,
                maximum,
                output,
                context.execution(),
            )
        } else {
            self.apply_workspace(input, maximum, output, KernelContext::for_operator(context))
        }
    }

    fn apply_operator_checked(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: &scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<(), SciOxideError> {
        validate_mat_vec_output(
            self.as_view(),
            VectorView::from_slice(input),
            &VectorViewMut::from_slice(output),
        )?;
        validate_vector_scratch(self.rows(), scratch.len())?;
        let maximum = input.iter().try_fold(S::Real::ZERO, |bound, &value| {
            crate::checked::validate_input(value, "operator input")?;
            Ok::<_, SciOxideError>(
                bound
                    .max(value.real_component().abs())
                    .max(value.imaginary_component().abs()),
            )
        })?;
        self.apply_operator(input, maximum, &mut scratch[..self.rows()], context)?;
        output.copy_from_slice(&scratch[..self.rows()]);
        Ok(())
    }

    fn apply_finite(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let input = VectorView::from_slice(input);
        validate_mat_vec_output(self.as_view(), input, &VectorViewMut::from_slice(output))?;
        validate_vector_scratch(self.rows(), scratch.len())?;
        let input = input.try_finite()?;
        kernel::level2::gemv_bounded(*self, input, scratch, context.policy(), context)?;
        output.copy_from_slice(&scratch[..self.rows()]);
        Ok(())
    }
}

impl<S: NumericScalar> LinearOperator<S> for FiniteMatrixView<'_, S> {
    fn apply_finite_into_scoped(
        &self,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        validate_vector_scratch(self.rows(), workspace.len())?;
        self.apply_operator(input, maximum, output, context)
            .map(scioxide_core::Invocation::Evaluated)
    }
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows())
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_finite(input, output, workspace, KernelContext::sequential_native())
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_operator_checked(input, output, workspace, context)
            .map(scioxide_core::Invocation::Evaluated)
    }
}

impl<S: NumericScalar> AdjointOperator<S> for FiniteMatrixView<'_, S> {
    fn apply_adjoint_finite_into_scoped(
        &self,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        validate_vector_scratch(self.cols(), workspace.len())?;
        self.adjoint()
            .apply_operator(input, maximum, output, context)
            .map(scioxide_core::Invocation::Evaluated)
    }
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols())
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.adjoint()
            .apply_finite(input, output, workspace, KernelContext::sequential_native())
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.adjoint()
            .apply_operator_checked(input, output, workspace, context)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: NumericScalar> BlockOperator<S> for FiniteMatrixView<'_, S> {}

fn validate_mat_vec_input<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    input: VectorView<'_, S>,
) -> Result<(), SciOxideError> {
    if input.len != matrix.cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(matrix.cols),
            Dimensions::one(input.len),
        ));
    }
    Ok(())
}

fn validate_mat_vec_output<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    input: VectorView<'_, S>,
    output: &VectorViewMut<'_, S>,
) -> Result<(), SciOxideError> {
    validate_mat_vec_input(matrix, input)?;
    crate::checked::expect_shape([matrix.rows], [output.len])
}

fn validate_mat_mat_inputs<S: NumericScalar>(
    left: MatrixView<'_, S>,
    right: MatrixView<'_, S>,
) -> Result<(), SciOxideError> {
    if left.cols != right.rows {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(left.cols),
            Dimensions::one(right.rows),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use scioxide_core::Complex;

    #[test]
    fn complex_conjugation_names_fix_the_selected_operand() {
        let left_data = [Complex::new(1.0_f64, 2.0)];
        let right_data = [Complex::new(3.0_f64, 4.0)];
        let left = VectorView::from_slice(&left_data);
        let right = VectorView::from_slice(&right_data);

        assert_eq!(
            left.dotu_scoped(
                right,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
            Complex::new(-5.0, 10.0)
        );
        assert_eq!(
            left.dotc_left_scoped(
                right,
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
            Complex::new(11.0, -2.0)
        );
        assert_eq!(
            right
                .dotc_left_scoped(
                    left,
                    &crate::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            Complex::new(11.0, 2.0)
        );
    }

    #[test]
    fn range_safe_vector_kernels_stage_before_committing() {
        let cross_left = [0.0_f64, f64::MAX, f64::MAX];
        let cross_right = [0.0_f64, 2.0, 2.0];
        let mut cross_output = [9.0_f64; 3];
        VectorView::from_slice(&cross_left)
            .cross_into(
                VectorView::from_slice(&cross_right),
                &crate::LinalgExecutionOptions::new(),
                VectorViewMut::from_slice(&mut cross_output),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(cross_output, [0.0; 3]);

        let x = [f64::MAX];
        let y = [f64::MAX];
        let mut output = [7.0_f64];
        VectorView::from_slice(&x)
            .axpby_into(
                2.0,
                VectorView::from_slice(&y),
                -2.0,
                &crate::LinalgExecutionOptions::new(),
                VectorViewMut::from_slice(&mut output),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(output, [0.0]);

        let mut unchanged = [7.0_f64];
        let mut no_scratch = [];
        assert!(
            VectorView::from_slice(&x)
                .axpby_into(
                    2.0,
                    VectorView::from_slice(&y),
                    -2.0,
                    &crate::LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut unchanged),
                    &mut crate::LinalgWorkspace::from_scalar_slice(&mut no_scratch),
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(unchanged, [7.0]);
    }

    #[test]
    fn normalization_does_not_materialize_an_unrepresentable_norm() {
        let values = [f64::MAX, f64::MAX];
        let normalized = VectorView::from_slice(&values).normalize().unwrap();
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!((normalized.as_slice()[0] - expected).abs() <= f64::EPSILON);
        assert!((normalized.as_slice()[1] - expected).abs() <= f64::EPSILON);
    }

    #[test]
    fn view_mat_vec_supports_non_contiguous_input_and_output() {
        let matrix_data = [1.0_f64, 2.0, 99.0, 3.0, 4.0];
        let input_data = [5.0, 99.0, 6.0];
        let mut output_data = [0.0, 99.0, 0.0];
        let matrix = MatrixView::from_slice_with_strides(2, 2, &matrix_data, 3, 1).unwrap();
        let input = VectorView::from_slice_with_stride(&input_data, 2, 2).unwrap();
        let output = VectorViewMut::from_slice_with_stride(&mut output_data, 2, 2).unwrap();

        matrix
            .mul_vec_into(
                input,
                &crate::LinalgExecutionOptions::new(),
                output,
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();

        assert_eq!(output_data, [17.0, 99.0, 39.0]);
    }

    #[test]
    fn transpose_and_conjugate_transpose_are_distinct_for_complex_values() {
        let matrix = Matrix::from_vec(
            2,
            2,
            vec![
                Complex::new(1.0_f64, 1.0),
                Complex::new(2.0, -1.0),
                Complex::new(3.0, 2.0),
                Complex::new(4.0, 0.5),
            ],
        )
        .unwrap();
        let input = Vector::from_vec(vec![Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)]);

        let transposed = matrix.view().transpose().mul_vec(input.view()).unwrap();
        let adjoint = matrix.view().adjoint().mul_vec(input.view()).unwrap();

        assert_eq!(
            transposed.as_slice(),
            &[Complex::new(0.0, 5.0), Complex::new(3.5, 2.0)]
        );
        assert_eq!(
            adjoint.as_slice(),
            &[Complex::new(4.0, 1.0), Complex::new(4.5, 6.0)]
        );
    }

    #[test]
    fn blocked_mat_mat_handles_rectangular_complex_inputs() {
        let left = Matrix::from_vec(
            2,
            3,
            vec![
                Complex::new(1.0_f64, 1.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, -1.0),
                Complex::new(3.0, 0.0),
                Complex::new(-1.0, 2.0),
                Complex::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let right = Matrix::from_vec(
            3,
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(2.0, -1.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 2.0),
                Complex::new(-1.0, 1.0),
            ],
        )
        .unwrap();

        let product = left.view().mul_mat(&right).unwrap();

        assert_eq!(
            product.as_slice(),
            &[
                Complex::new(7.0, -1.0),
                Complex::new(2.0, 2.0),
                Complex::new(3.0, 13.0),
                Complex::new(-5.0, 9.0),
            ]
        );
    }

    #[test]
    fn blocked_mat_mat_supports_strided_inputs_and_output() {
        let left_data = [1.0_f64, 2.0, 99.0, 3.0, 4.0];
        let right_data = [5.0_f64, 6.0, 99.0, 7.0, 8.0];
        let mut output_data = [11.0_f64, 99.0, 11.0, 99.0, 11.0, 99.0, 11.0];
        let left = MatrixView::from_slice_with_strides(2, 2, &left_data, 3, 1).unwrap();
        let right = MatrixView::from_slice_with_strides(2, 2, &right_data, 3, 1).unwrap();
        let output = MatrixViewMut::from_slice_with_strides(2, 2, &mut output_data, 4, 2).unwrap();

        left.mul_mat_into(
            right,
            &crate::LinalgExecutionOptions::new(),
            output,
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();

        assert_eq!(output_data, [19.0, 99.0, 22.0, 99.0, 43.0, 99.0, 50.0]);
    }

    #[test]
    fn zero_sized_products_preserve_mathematical_shapes() {
        let left = Matrix::<f64>::zeros(2, 0).unwrap();
        let right = Matrix::<f64>::zeros(0, 3).unwrap();
        let product = left.view().mul_mat(&right).unwrap();

        assert_eq!((product.rows(), product.cols()), (2, 3));
        assert_eq!(product.as_slice(), &[0.0; 6]);

        let empty_rows = Matrix::<f64>::zeros(0, 3).unwrap();
        let input = Vector::from_vec(Vec::new());
        assert_eq!(
            empty_rows
                .view()
                .transpose()
                .mul_vec(input.view())
                .unwrap()
                .as_slice(),
            &[0.0; 3]
        );

        let many_empty_rows = Matrix::<f64>::zeros(usize::MAX, 0).unwrap();
        let no_columns = Matrix::<f64>::zeros(0, 0).unwrap();
        let empty_product = many_empty_rows.view().mul_mat(&no_columns).unwrap();
        assert_eq!(
            (empty_product.rows(), empty_product.cols()),
            (usize::MAX, 0)
        );
        assert!(empty_product.as_slice().is_empty());
    }

    #[test]
    fn structural_errors_leave_outputs_unchanged() {
        let matrix = Matrix::<f64>::zeros(2, 3).unwrap();
        let input = Vector::from_vec(vec![1.0, 2.0]);
        let mut output = Vector::from_vec(vec![9.0, 9.0]);
        assert!(matches!(
            (matrix.view().mul_vec_into(
                &input,
                &crate::LinalgExecutionOptions::new(),
                output.view_mut(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(output.as_slice(), &[9.0, 9.0]);

        let right = Matrix::<f64>::zeros(3, 2).unwrap();
        let mut wrong_output = Matrix::from_vec(1, 1, vec![7.0]).unwrap();
        assert!(matches!(
            (matrix.view().mul_mat_into(
                &right,
                &crate::LinalgExecutionOptions::new(),
                wrong_output.view_mut(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(wrong_output.as_slice(), &[7.0]);
    }

    #[test]
    fn product_shape_overflow_is_structural() {
        let left = Matrix::<f64>::zeros(usize::MAX, 0).unwrap();
        let right = Matrix::<f64>::zeros(0, 2).unwrap();
        assert!(matches!(
            (left.view().mul_mat(&right)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
    }
}
