//! Integrand interface for sensitivity plans.

use super::{SensitivityMode, checked_callback_values};
use scioxide_core::{
    AdjointOperator, ExecutionContext, LinearOperator, NumericScalar, RealScalar, SciOxideError,
};

/// Integrand `f(x, p)` of a [`QuadratureSensitivityPlan`](crate::QuadratureSensitivityPlan),
/// with its parameter Jacobian `J(x, p) = ∂f/∂p` as a linear operator.
///
/// `f` has [`value_dimension`](Self::value_dimension) real components `m` and
/// depends on [`parameter_dimension`](Self::parameter_dimension) real
/// parameters `P`, so `J` is `m × P`, with `x` held fixed. A complex value or
/// parameter counts as its real and imaginary parts, interleaved. At each node,
/// [`evaluate`](Self::evaluate) writes `f(x, p)` and applies `J` once through
/// [`SensitivitySample`]: as `J v` for a JVP or `Jᵀ w` for a VJP. At a moving
/// endpoint it is called for the value only. The integrand must describe the
/// same function at every call.
///
/// `Workspace` is scratch storage of the integrand, such as the storage of an
/// automatic-differentiation tape. You pass one or more workspaces to each
/// call; each worker of a parallel execution uses its own. Parallel execution
/// needs `Self: Sync` and `Workspace: Send`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError, LinearOperator};
/// use scioxide_integrate::{SensitivityIntegrand, SensitivityMode, SensitivitySample};
///
/// // The Jacobian ∂f/∂p = x of f(x, p) = p x, as the 1 × 1 operator v ↦ x v.
/// struct Scale(f64);
/// # impl LinearOperator<f64> for Scale {
/// #     fn rows(&self) -> usize {
/// #         1
/// #     }
/// #     fn cols(&self) -> usize {
/// #         1
/// #     }
/// #     fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
/// #         Ok(0)
/// #     }
/// #     fn apply_into(&self, v: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
/// #         y[0] = self.0 * v[0];
/// #         Ok(())
/// #     }
/// # }
/// struct Linear;
/// impl SensitivityIntegrand<f64> for Linear {
/// #     type Workspace = ();
/// #     fn parameter_dimension(&self) -> usize {
/// #         1
/// #     }
/// #     fn value_dimension(&self) -> usize {
/// #         1
/// #     }
/// #     fn supports(&self, mode: SensitivityMode) -> bool {
/// #         mode == SensitivityMode::Jvp
/// #     }
///     fn evaluate(
///         &self,
///         x: f64,
///         p: &[f64],
///         sample: &mut SensitivitySample<'_, f64>,
///         _: &mut (),
///         _: ExecutionContext<'_>,
///     ) -> Result<(), SciOxideError> {
///         sample.value_mut()[0] = p[0] * x;
///         sample.apply_forward(&Scale(x), &mut [])
///     }
/// }
/// ```
pub trait SensitivityIntegrand<S: RealScalar> {
    /// Scratch storage of the integrand for one worker.
    type Workspace;
    /// Returns the number of real parameters `P`.
    fn parameter_dimension(&self) -> usize;
    /// Returns the number of real value components `m`.
    fn value_dimension(&self) -> usize;
    /// Returns whether the integrand can compute products of the given mode:
    /// `J v` for [`SensitivityMode::Jvp`], `Jᵀ w` for [`SensitivityMode::Vjp`].
    fn supports(&self, mode: SensitivityMode) -> bool;
    /// Writes `f(x, parameters)` to [`sample.value_mut()`](SensitivitySample::value_mut),
    /// then applies the Jacobian at `x` once.
    ///
    /// Use [`SensitivitySample::apply_forward`] with a [`LinearOperator`] if
    /// the integrand supports only JVPs, or [`SensitivitySample::apply`] with
    /// an [`AdjointOperator`] for both modes. Write every value component
    /// before applying the Jacobian. When [`sample.mode()`](SensitivitySample::mode)
    /// is `None`, only the value is needed and both apply methods do nothing.
    /// `execution` carries the cancellation and deadline of the call; use it for
    /// nested work instead of starting threads of your own.
    ///
    /// # Errors
    ///
    /// Returns any error of the integrand; the plan returns it unchanged. A
    /// call that returns `Ok` without applying a requested product makes the
    /// plan fail with `InvalidInput`.
    fn evaluate(
        &self,
        x: S,
        parameters: &[S],
        sample: &mut SensitivitySample<'_, S>,
        workspace: &mut Self::Workspace,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
}

#[derive(Clone, Copy)]
pub(super) enum Product<'a, S> {
    Value,
    Jvp(&'a [S]),
    Vjp(&'a [S]),
}
impl<S> Product<'_, S> {
    pub(super) fn mode(self) -> Option<SensitivityMode> {
        match self {
            Self::Value => None,
            Self::Jvp(_) => Some(SensitivityMode::Jvp),
            Self::Vjp(_) => Some(SensitivityMode::Vjp),
        }
    }
}

/// Value and derivative product of one integrand evaluation, passed to
/// [`SensitivityIntegrand::evaluate`].
///
/// Only the plan creates samples. Write the value with
/// [`value_mut`](Self::value_mut), then apply the Jacobian once with
/// [`apply_forward`](Self::apply_forward) or [`apply`](Self::apply). The
/// plan computes the product from the operator; it never forms a dense
/// Jacobian or uses finite differences.
pub struct SensitivitySample<'a, S> {
    pub(super) value: &'a mut [S],
    pub(super) product: &'a mut [S],
    pub(super) request: Product<'a, S>,
    pub(super) parameters: usize,
    pub(super) attempted: &'a mut bool,
}
impl<S: Copy> SensitivitySample<'_, S> {
    /// Returns the value `f(x, p)` to fill, with one entry per real component.
    pub fn value_mut(&mut self) -> &mut [S] {
        self.value
    }
    /// Returns the requested product, or `None` at a moving endpoint, where
    /// only the value is needed.
    pub fn mode(&self) -> Option<SensitivityMode> {
        self.request.mode()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SensitivitySample<'_, S> {
    fn validate(&self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        crate::kernel::shape(self.value.len(), rows)?;
        crate::kernel::shape(self.parameters, cols)?;
        checked_callback_values(self.value)?;
        if *self.attempted {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "a sample may apply its derivative only once",
            ));
        }
        Ok(())
    }
    /// Computes the Jacobian–vector product `J v` with the operator `J`.
    ///
    /// `operator` is the `m × P` Jacobian at the current node. `scratch` must
    /// hold at least `operator.apply_workspace_len()` values. If only the
    /// value is requested, the call does nothing.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a VJP is requested, the Jacobian was already
    ///   applied, or `scratch` is too short.
    /// - `DimensionMismatch` if `operator` is not `m × P`.
    /// - `NonFinite` if the value has an unwritten, NaN or infinite component,
    ///   or the product has a NaN or infinite component.
    /// - Any error returned by `operator`.
    pub fn apply_forward<O: LinearOperator<S> + ?Sized>(
        &mut self,
        operator: &O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        let seed = match self.request {
            Product::Value => return Ok(()),
            Product::Jvp(seed) => seed,
            Product::Vjp(_) => {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("operator"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "VJP requires an adjoint operator",
                ));
            }
        };
        self.validate(operator.rows(), operator.cols())?;
        let n = operator.apply_workspace_len()?;
        let scratch = scratch.get_mut(..n).ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("scratch"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient operator scratch",
            )
        })?;
        // Marked before the call so that a failing operator is still charged as one
        // derivative application.
        *self.attempted = true;
        operator.apply_into(seed, self.product, scratch)?;
        checked_callback_values(self.product)
    }
    /// Computes the requested product with the operator `J`: `J v` for a JVP
    /// or `Jᵀ w` for a VJP.
    ///
    /// For complex values and parameters, `J` is the real `m × P` Jacobian of
    /// the real and imaginary parts, so its transpose is the adjoint under
    /// `⟨u, v⟩ = Re(conj(u) v)`, also for functions that are not holomorphic.
    /// `scratch` must hold at least the operator's workspace length for the
    /// product.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the Jacobian was already applied or `scratch` is
    ///   too short.
    /// - `DimensionMismatch` if `operator` is not `m × P`.
    /// - `NonFinite` if the value has an unwritten, NaN or infinite component,
    ///   or the product has a NaN or infinite component.
    /// - Any error returned by `operator`.
    pub fn apply<O: AdjointOperator<S> + ?Sized>(
        &mut self,
        operator: &O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        let Product::Vjp(seed) = self.request else {
            return self.apply_forward(operator, scratch);
        };
        self.validate(operator.rows(), operator.cols())?;
        let n = operator.apply_adjoint_workspace_len()?;
        let scratch = scratch.get_mut(..n).ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("scratch"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient adjoint scratch",
            )
        })?;
        *self.attempted = true;
        operator.apply_adjoint_into(seed, self.product, scratch)?;
        checked_callback_values(self.product)
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate<S: RealScalar, I: SensitivityIntegrand<S> + ?Sized>(
    integrand: &I,
    x: S,
    parameters: &[S],
    product: Product<'_, S>,
    buffers: (&mut [S], &mut I::Workspace),
    execution: ExecutionContext<'_>,
    attempted: &mut bool,
) -> Result<(), SciOxideError> {
    // A sample row is laid out as `[value (d) | product]`: for JVP the product has d entries
    // (J seed), for VJP it has one entry per parameter (J^T seed). The value width is
    // recovered from the row length and re-checked against the provider's declared shape.
    let (row, workspace) = buffers;
    let components = match product {
        Product::Value => row.len(),
        Product::Jvp(_) => row.len() / 2,
        Product::Vjp(_) => row.len().checked_sub(parameters.len()).ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("integrand"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "changed provider output shape",
            )
        })?,
    };
    crate::kernel::shape(components, integrand.value_dimension())?;
    crate::kernel::shape(parameters.len(), integrand.parameter_dimension())?;
    row.fill(S::NAN);
    let (value, output) = row.split_at_mut(components);
    let mut sample = SensitivitySample {
        value,
        product: output,
        request: product,
        parameters: parameters.len(),
        attempted,
    };
    integrand.evaluate(x, parameters, &mut sample, workspace, execution)?;
    if product.mode().is_some() && !*sample.attempted {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("integrand"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "provider omitted the requested derivative action",
        ));
    }
    checked_callback_values(row)
}
