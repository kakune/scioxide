//! Bordered systems for singular square operators with known left and right null spaces.

use crate::kernel::CANDIDATE;
use crate::structured::{admit, identity};
use crate::{
    LinalgExecutionContext, LinalgRequirements, LinalgWorkspace, Vector, VectorView, VectorViewMut,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    AdjointOperator, BlockVectorView, ExecutionContext, LinearOperator, NumericScalar, RealScalar,
    SciOxideError, Tolerance,
};

/// Operation of a [`BorderedSystem`] whose workspace size is requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderedSystemWorkspaceRequest {
    /// [`BorderedSystem::prepare_rhs_into`]: `n + 2k` scalars.
    PrepareRhs,
    /// [`BorderedSystem::diagnose_scoped`]: `3n + p` scalars, with `p` the operator workspace.
    Diagnose,
}

/// Treatment of the component `Lᴴ b` of a right-hand side outside the range of `A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityPolicy {
    /// Rejects `b` if `‖Lᴴ b‖₂ > atol + rtol ‖b‖₂` for the compatibility tolerance.
    RequireCompatible,
    /// Replaces `b` by its projection `b − L Lᴴ b` onto the range of `A`.
    Project,
}

/// Options for a [`BorderedSystem`]: two tolerances and the execution options.
///
/// The validation tolerance checks the null-space bases when the system is built. The
/// compatibility tolerance checks right-hand sides under
/// [`CompatibilityPolicy::RequireCompatible`]. Both default to
/// [`Tolerance::from_sqrt_epsilon`], and they are independent. The execution options
/// apply to construction, [`prepare_rhs_into`](BorderedSystem::prepare_rhs_into) and
/// [`diagnose_scoped`](BorderedSystem::diagnose_scoped).
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_linalg::BorderedSystemOptions;
/// let tolerance = Tolerance::new(1e-12_f64, 1e-12)?;
/// let options = BorderedSystemOptions::new()
///     .with_validation_tolerance(tolerance)
///     .with_compatibility_tolerance(tolerance);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BorderedSystemOptions<R: RealScalar> {
    validation_tolerance: Tolerance<R>,
    compatibility_tolerance: Tolerance<R>,
    execution: crate::LinalgExecutionOptions,
}
impl<R: RealScalar> BorderedSystemOptions<R> {
    /// Returns options with both tolerances `Tolerance::from_sqrt_epsilon()`.
    pub fn new() -> Self {
        Self {
            validation_tolerance: Tolerance::from_sqrt_epsilon(),
            compatibility_tolerance: Tolerance::from_sqrt_epsilon(),
            execution: crate::LinalgExecutionOptions::new(),
        }
    }
    /// Returns the execution options.
    pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
        self.execution
    }
    /// Sets the execution options.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }
    /// Sets the tolerance for the orthonormality of the bases and for `‖A R‖` and `‖Aᴴ L‖`.
    pub const fn with_validation_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.validation_tolerance = tolerance;
        self
    }
    /// Sets the tolerance used by [`CompatibilityPolicy::RequireCompatible`].
    pub const fn with_compatibility_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.compatibility_tolerance = tolerance;
        self
    }
    /// Returns the basis validation tolerance.
    pub const fn validation_tolerance(&self) -> Tolerance<R> {
        self.validation_tolerance
    }
    /// Returns the right-hand-side compatibility tolerance.
    pub const fn compatibility_tolerance(&self) -> Tolerance<R> {
        self.compatibility_tolerance
    }
}
impl<R: RealScalar> Default for BorderedSystemOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Residual norms of a solution of a bordered system, evaluated on the original system `A x = b`.
///
/// All norms are Euclidean. `b` is the original right-hand side, `[b̂; c]` the prepared
/// right-hand side and `x` the first `n` entries of the solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BorderedSystemDiagnostics<R: RealScalar> {
    /// Dimension of the supplied subspace, not a numerical rank certificate.
    pub(crate) nullity: usize,
    /// Norm of the original load's left-nullspace component.
    pub(crate) compatibility_defect: R,
    /// Norm of the load change actually applied.
    pub(crate) projection_norm: R,
    /// Norm of `A x - b` for the original load.
    pub(crate) original_residual_norm: R,
    /// Norm of `A x - prepared_b`.
    pub(crate) projected_residual_norm: R,
    /// Norm of `Rᴴ x - gauge_coordinates`.
    pub(crate) gauge_defect: R,
}

impl<R: RealScalar> BorderedSystemDiagnostics<R> {
    /// Returns `k`, the number of basis columns; this is not a computed nullity.
    pub const fn nullity(&self) -> usize {
        self.nullity
    }

    /// Returns `‖Lᴴ b‖₂`, the component of `b` outside the range of `A`.
    pub const fn compatibility_defect(&self) -> R {
        self.compatibility_defect
    }

    /// Returns `‖b − b̂‖₂`, the change made by the projection.
    pub const fn projection_norm(&self) -> R {
        self.projection_norm
    }

    /// Returns `‖A x − b‖₂`.
    pub const fn original_residual_norm(&self) -> R {
        self.original_residual_norm
    }

    /// Returns `‖A x − b̂‖₂`.
    pub const fn projected_residual_norm(&self) -> R {
        self.projected_residual_norm
    }

    /// Returns `‖Rᴴ x − c‖₂`, the defect of the gauge condition.
    pub const fn gauge_defect(&self) -> R {
        self.gauge_defect
    }
}

fn difference<S: NumericScalar>(a: S, b: S) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::ProductSum::new();
    sum.add_product(a, S::ONE)?;
    sum.subtract_product(b, S::ONE)?;
    sum.finish()
}

fn combine_modes<S: NumericScalar>(
    values: &mut [S],
    basis: BlockVectorView<'_, S>,
    coefficients: &[S],
    subtract: bool,
) -> Result<(), SciOxideError> {
    for (row, value) in values.iter_mut().enumerate() {
        let mut sum = scioxide_core::ProductSum::new();
        sum.add_product(*value, S::ONE)?;
        for (column, &coefficient) in coefficients.iter().enumerate() {
            let entry = basis.column(column)?[row];
            if subtract {
                sum.subtract_product(entry, coefficient)?;
            } else {
                sum.add_product(entry, coefficient)?;
            }
        }
        *value = sum.finish()?;
    }
    Ok(())
}

// Bordered gauge rows use one portable Standard reduction, so scoped and
// unscoped operator applications round identically.
fn dot<S: NumericScalar>(a: &[S], b: &[S]) -> Result<S, SciOxideError> {
    crate::reduction::dot_with_context(
        VectorView::from_slice(a),
        VectorView::from_slice(b),
        true,
        crate::KernelContext::new(scioxide_core::ExecutionContext::sequential()),
    )
}
/// Bordered system that turns a singular square system `A x = b` into a nonsingular one.
///
/// `A` is an `n × n` operator with known orthonormal bases `L` of the left null space
/// (`Aᴴ L = 0`) and `R` of the right null space (`A R = 0`), both `n × k`. The system
/// represents the `(n + k) × (n + k)` bordered system
///
/// ```text
/// ⎡ A   L ⎤ ⎡ x ⎤   ⎡ b̂ ⎤
/// ⎣ Rᴴ  0 ⎦ ⎣ λ ⎦ = ⎣ c ⎦
/// ```
///
/// If `L` and `R` span the whole left and right null spaces, the bordered matrix is
/// nonsingular. For a compatible right-hand side (`Lᴴ b̂ = 0`), the solution has `λ = 0`,
/// and `x` solves `A x = b̂` with the gauge condition `Rᴴ x = c`. It does not solve
/// the system. Pass [`augmented`](Self::augmented) to a direct or iterative solver, build
/// the right-hand side with [`prepare_rhs`](Self::prepare_rhs), and check the solution with
/// [`diagnose`](Self::diagnose).
///
/// The system borrows the operator and the bases. Construction checks that each basis is
/// orthonormal and that `‖A rⱼ‖₂` and `‖Aᴴ lⱼ‖₂` are at most `atol + rtol · operator_scale`
/// for the validation tolerance. `operator_scale` is your estimate of `‖A‖`. The system
/// cannot check that the bases are complete; an incomplete basis leaves the bordered
/// matrix singular.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
#[derive(Clone)]
pub struct BorderedSystem<'a, S: NumericScalar> {
    operator: &'a (dyn AdjointOperator<S> + Sync),
    left: BlockVectorView<'a, S>,
    right: BlockVectorView<'a, S>,
    options: BorderedSystemOptions<S::Real>,
}
impl<S: NumericScalar> core::fmt::Debug for BorderedSystem<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BorderedSystem")
            .field("left", &self.left)
            .field("right", &self.right)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl<'a, S: NumericScalar> BorderedSystem<'a, S> {
    /// Checks the null-space bases and returns the system.
    ///
    /// `left` holds `L` and `right` holds `R`, each as `k` columns of length `n`.
    ///
    /// # Errors
    ///
    /// The errors of [`build`](Self::build).
    pub fn new(
        operator: &'a (dyn AdjointOperator<S> + Sync),
        left: BlockVectorView<'a, S>,
        right: BlockVectorView<'a, S>,
        operator_scale: S::Real,
        options: &BorderedSystemOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            operator,
            left,
            right,
            operator_scale,
            options,
            ExecutionContext::sequential_native(),
        )
    }

    /// Checks the null-space bases in the execution context `context` and returns the system.
    ///
    /// Each of the `2k` basis columns counts as one iteration toward the iteration limit of
    /// the options. The system does not keep `context`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the operator is not square, a basis does not have `n` rows,
    ///   or `left` and `right` have different numbers of columns.
    /// - `NonFinite` if `operator_scale` or a basis entry is not finite.
    /// - `InvalidInput` if `k = 0`, `k > n` or `operator_scale < 0`, if a basis is not
    ///   orthonormal within the validation tolerance, or if `‖A rⱼ‖₂` or `‖Aᴴ lⱼ‖₂`
    ///   exceeds it.
    /// - `ConvergenceFailure` if the `2k` columns exceed the iteration limit.
    /// - The errors of the operator, and workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn build<'scope, 'capability>(
        operator: &'a (dyn AdjointOperator<S> + Sync),
        left: BlockVectorView<'a, S>,
        right: BlockVectorView<'a, S>,
        operator_scale: S::Real,
        options: &BorderedSystemOptions<S::Real>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let context = context.into().with_options(&options.execution)?;
        context.execution().check()?;
        let n = operator.rows();
        crate::checked::expect_shape([n], [operator.cols()])?;
        crate::checked::expect_shape([n], [left.rows()])?;
        crate::checked::expect_shape([n], [right.rows()])?;
        crate::checked::expect_shape([left.right_hand_sides()], [right.right_hand_sides()])?;
        let k = left.right_hand_sides();
        checked_add(n, k, scioxide_core::quantity!("augmented_dimension"))?;
        if !operator_scale.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "operator scale must be finite",
            ));
        }
        if k == 0 || k > n || operator_scale < S::Real::ZERO {
            let argument = if operator_scale < S::Real::ZERO {
                scioxide_core::argument!("operator_scale")
            } else {
                scioxide_core::argument!("right")
            };
            return Err(crate::checked::out_of_range(
                argument,
                "invalid nullity or operator scale",
            ));
        }
        for &value in left.as_slice().iter().chain(right.as_slice()) {
            crate::checked::validate_input(value, "bordered basis")?;
        }
        let mut requirements = LinalgRequirements::new(options.execution.byte_limit());
        *requirements.scalar_mut(CANDIDATE) = checked_add(
            n,
            operator
                .apply_workspace_len()?
                .max(operator.apply_adjoint_workspace_len()?),
            scioxide_core::quantity!("bordered_validation"),
        )?;
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&requirements)?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "bordered_build", 1),
                n,
            ),
            options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                let (image, scratch) = crate::workspace::split_mut(
                    &mut workspace.scalar_mut(CANDIDATE)[..requirements.scalar(CANDIDATE)],
                    n,
                )?;
                for (basis, adjoint, argument) in [
                    (left, true, scioxide_core::argument!("left")),
                    (right, false, scioxide_core::argument!("right")),
                ] {
                    for j in 0..k {
                        admit(kernels.execution(), scope)?;
                        let column = basis.column(j)?;
                        for i in 0..=j {
                            let expected = if i == j { S::ONE } else { S::ZERO };
                            let dot = crate::reduction::dot_with_context(
                                VectorView::from_slice(basis.column(i)?),
                                VectorView::from_slice(column),
                                true,
                                kernels,
                            )?;
                            if !options
                                .validation_tolerance
                                .is_residual_ok(difference(dot, expected)?.abs(), S::Real::ONE)?
                            {
                                return Err(crate::checked::out_of_range(
                                    argument,
                                    "basis columns must be orthonormal",
                                ));
                            }
                        }
                        if adjoint {
                            let len = operator.apply_adjoint_workspace_len()?;
                            operator
                                .apply_adjoint_into_scoped(
                                    column,
                                    image,
                                    &mut scratch[..len],
                                    &mut scioxide_core::OperatorContext::with_kernels(
                                        scope, &kernels,
                                    ),
                                )
                                .and_then(scioxide_core::Invocation::require)?;
                        } else {
                            let len = operator.apply_workspace_len()?;
                            operator
                                .apply_into_scoped(
                                    column,
                                    image,
                                    &mut scratch[..len],
                                    &mut scioxide_core::OperatorContext::with_kernels(
                                        scope, &kernels,
                                    ),
                                )
                                .and_then(scioxide_core::Invocation::require)?;
                        }
                        crate::checked::provider_output(image)?;
                        if !options.validation_tolerance.is_residual_ok(
                            crate::reduction::l2_norm_with_context(
                                VectorView::from_slice(image),
                                kernels,
                            )?,
                            operator_scale,
                        )? {
                            return Err(crate::checked::inconsistent(
                                argument,
                                "basis does not annihilate the operator within tolerance",
                            ));
                        }
                    }
                }
                kernels.execution().check()?;
                Ok(())
            },
        )?;
        Ok(Self {
            operator,
            left,
            right,
            options: *options,
        })
    }

    /// Returns the left null-space basis `L` (`n × k`).
    pub const fn left_basis(&self) -> BlockVectorView<'a, S> {
        self.left
    }
    /// Returns the right null-space basis `R` (`n × k`).
    pub const fn right_basis(&self) -> BlockVectorView<'a, S> {
        self.right
    }
    /// Returns the bordered matrix `[A L; Rᴴ 0]` as an operator, without forming it.
    pub fn augmented(&self) -> BorderedSystemOperator<'_, 'a, S> {
        BorderedSystemOperator { plan: self }
    }

    /// Returns the bordered right-hand side `[b̂; c]` of length `n + k`.
    ///
    /// `rhs` is `b` and `gauge_coordinates` is `c`, the prescribed value of `Rᴴ x`. `b̂` is
    /// `b`, or its projection `b − L Lᴴ b` under [`CompatibilityPolicy::Project`].
    ///
    /// # Errors
    ///
    /// The errors of [`prepare_rhs_into`](Self::prepare_rhs_into).
    pub fn prepare_rhs<'rhs, 'gauge>(
        &self,
        rhs: impl Into<VectorView<'rhs, S>>,
        gauge_coordinates: impl Into<VectorView<'gauge, S>>,
        policy: CompatibilityPolicy,
    ) -> Result<Vector<S>, SciOxideError> {
        let mut output = Vector::from_vec(try_filled(
            self.augmented().rows(),
            S::ZERO,
            scioxide_core::quantity!("bordered_rhs"),
        )?);
        self.prepare_rhs_into(
            rhs,
            gauge_coordinates,
            policy,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Returns the workspace size for one operation of the system.
    ///
    /// A buffer you provide can serve as the workspace through
    /// [`LinalgWorkspace::from_scalar_slice`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the options.
    /// - The errors of the operator's workspace-length query.
    pub fn workspace_requirements(
        &self,
        request: BorderedSystemWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let n = self.operator.rows();
        let k = self.left.right_hand_sides();
        let terms = match request {
            BorderedSystemWorkspaceRequest::PrepareRhs => [n, k, k, 0],
            BorderedSystemWorkspaceRequest::Diagnose => {
                [n, n, n, self.operator.apply_workspace_len()?]
            }
        };
        let mut requirements = LinalgRequirements::new(self.options.execution.byte_limit());
        *requirements.scalar_mut(CANDIDATE) =
            terms.into_iter().try_fold(0usize, |total, term| {
                checked_add(total, term, scioxide_core::quantity!("bordered_scratch"))
            })?;
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }
    /// Allocates a workspace sized by [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// The errors of [`workspace_requirements`](Self::workspace_requirements), and
    /// workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn workspace(
        &self,
        request: BorderedSystemWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Writes the right-hand side `[b̂; c]` of [`prepare_rhs`](Self::prepare_rhs) to `output`.
    ///
    /// `output` has `n + k` entries and may be strided. Each of the `k` coefficients `lⱼᴴ b`
    /// counts as one iteration toward the iteration limit of the options.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have `n` entries, `gauge_coordinates` does
    ///   not have `k`, or `output` does not have `n + k`.
    /// - `NonFinite` if an entry of `rhs` or `gauge_coordinates` is not finite.
    /// - `DomainError` under [`CompatibilityPolicy::RequireCompatible`] if
    ///   `‖Lᴴ b‖₂ > atol + rtol ‖b‖₂` for the compatibility tolerance.
    /// - `ConvergenceFailure` if the `k` coefficients exceed the iteration limit.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn prepare_rhs_into<'rhs, 'gauge, 'scope, 'capability>(
        &self,
        rhs: impl Into<VectorView<'rhs, S>>,
        gauge_coordinates: impl Into<VectorView<'gauge, S>>,
        policy: CompatibilityPolicy,
        mut output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&self.options.execution)?;
        context.execution().check()?;
        let rhs = rhs.into();
        let gauge_coordinates = gauge_coordinates.into();
        crate::checked::expect_shape([self.operator.rows()], [rhs.len()])?;
        crate::checked::expect_shape([self.left.right_hand_sides()], [gauge_coordinates.len()])?;
        crate::checked::expect_shape([self.augmented().rows()], [output.len()])?;
        for value in rhs.iter().chain(gauge_coordinates.iter()) {
            crate::checked::validate_input(value, "bordered load")?;
        }
        let requirements =
            self.workspace_requirements(BorderedSystemWorkspaceRequest::PrepareRhs)?;
        workspace.try_reserve(&requirements)?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "bordered_rhs", 1),
                rhs.len(),
            ),
            self.options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                let n = rhs.len();
                let (out, coefficients) = crate::workspace::split_mut(
                    &mut workspace.scalar_mut(CANDIDATE)[..requirements.scalar(CANDIDATE)],
                    output.len(),
                )?;
                for (value, entry) in out[..n].iter_mut().zip(rhs.iter()) {
                    *value = entry;
                }
                let scale = crate::reduction::l2_norm_with_context(rhs, kernels)?;
                for (j, coefficient) in coefficients.iter_mut().enumerate() {
                    admit(kernels.execution(), scope)?;
                    *coefficient = crate::reduction::dot_with_context(
                        VectorView::from_slice(self.left.column(j)?),
                        rhs,
                        true,
                        kernels,
                    )?;
                }
                if policy == CompatibilityPolicy::RequireCompatible
                    && !self.options.compatibility_tolerance.is_residual_ok(
                        crate::reduction::l2_norm_with_context(
                            VectorView::from_slice(coefficients),
                            kernels,
                        )?,
                        scale,
                    )?
                {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("rhs"),
                        scioxide_core::DomainReason::Undefined,
                        "load has an incompatible left-nullspace component",
                    ));
                }
                if policy == CompatibilityPolicy::Project {
                    combine_modes(&mut out[..n], self.left, coefficients, true)?;
                }
                // Gauge coordinates were validated before the epoch.
                for (value, target) in out[n..].iter_mut().zip(gauge_coordinates.iter()) {
                    *value = target;
                }
                kernels.execution().check()?;
                for (i, &value) in out.iter().enumerate() {
                    output.set_element(i, value);
                }
                Ok(())
            },
        )
    }

    /// Evaluates the residual norms of a solution `x` of the bordered system.
    ///
    /// `rhs` is the original `b`, `prepared_rhs` is `[b̂; c]` from
    /// [`prepare_rhs`](Self::prepare_rhs) and `solution` is `x` (length `n`).
    ///
    /// # Errors
    ///
    /// The errors of [`diagnose_scoped`](Self::diagnose_scoped).
    pub fn diagnose<'rhs, 'prepared, 'solution>(
        &self,
        rhs: impl Into<VectorView<'rhs, S>>,
        prepared_rhs: impl Into<VectorView<'prepared, S>>,
        solution: impl Into<VectorView<'solution, S>>,
    ) -> Result<BorderedSystemDiagnostics<S::Real>, SciOxideError> {
        self.diagnose_scoped(
            rhs,
            prepared_rhs,
            solution,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }

    /// Evaluates the residual norms of [`diagnose`](Self::diagnose) in `context`.
    ///
    /// The one application of `A` counts as one iteration toward the iteration limit of the
    /// options.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or `solution` does not have `n` entries, or
    ///   `prepared_rhs` does not have `n + k`.
    /// - `NonFinite` if an input entry or an entry of `A x` is not finite.
    /// - The errors of the operator, and workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn diagnose_scoped<'rhs, 'prepared, 'solution, 'scope, 'capability>(
        &self,
        rhs: impl Into<VectorView<'rhs, S>>,
        prepared_rhs: impl Into<VectorView<'prepared, S>>,
        solution: impl Into<VectorView<'solution, S>>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<BorderedSystemDiagnostics<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&self.options.execution)?;
        context.execution().check()?;
        let (rhs, prepared_rhs, solution) = (rhs.into(), prepared_rhs.into(), solution.into());
        let n = self.operator.rows();
        let k = self.left.right_hand_sides();
        crate::checked::expect_shape([n], [rhs.len()])?;
        crate::checked::expect_shape([n], [solution.len()])?;
        crate::checked::expect_shape([n + k], [prepared_rhs.len()])?;
        for value in rhs.iter().chain(prepared_rhs.iter()).chain(solution.iter()) {
            crate::checked::validate_input(value, "bordered diagnostics input")?;
        }
        let requirements = self.workspace_requirements(BorderedSystemWorkspaceRequest::Diagnose)?;
        workspace.try_reserve(&requirements)?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "bordered_diagnose", 1),
                n,
            ),
            self.options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                let (packed, rest) = crate::workspace::split_mut(
                    &mut workspace.scalar_mut(CANDIDATE)[..requirements.scalar(CANDIDATE)],
                    n,
                )?;
                for (value, entry) in packed.iter_mut().zip(solution.iter()) {
                    *value = entry;
                }
                let (image, rest) = crate::workspace::split_mut(rest, n)?;
                let (residual, work) = crate::workspace::split_mut(rest, n)?;
                admit(kernels.execution(), scope)?;
                self.operator
                    .apply_into_scoped(
                        packed,
                        image,
                        work,
                        &mut scioxide_core::OperatorContext::with_kernels(scope, &kernels),
                    )
                    .and_then(scioxide_core::Invocation::require)?;
                crate::checked::provider_output(image)?;
                for ((r, &a), b) in residual.iter_mut().zip(image.iter()).zip(rhs.iter()) {
                    *r = difference(a, b)?;
                }
                let original_residual_norm = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(residual),
                    kernels,
                )?;
                for ((r, &a), b) in residual
                    .iter_mut()
                    .zip(image.iter())
                    .zip(prepared_rhs.iter())
                {
                    *r = difference(a, b)?;
                }
                let projected_residual_norm = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(residual),
                    kernels,
                )?;
                for ((r, a), b) in residual.iter_mut().zip(rhs.iter()).zip(prepared_rhs.iter()) {
                    *r = difference(a, b)?;
                }
                let projection_norm = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(residual),
                    kernels,
                )?;
                for (j, r) in residual[..k].iter_mut().enumerate() {
                    *r = crate::reduction::dot_with_context(
                        VectorView::from_slice(self.left.column(j)?),
                        rhs,
                        true,
                        kernels,
                    )?;
                }
                let compatibility_defect = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(&residual[..k]),
                    kernels,
                )?;
                for (j, r) in residual[..k].iter_mut().enumerate() {
                    *r = difference(
                        crate::reduction::dot_with_context(
                            VectorView::from_slice(self.right.column(j)?),
                            solution,
                            true,
                            kernels,
                        )?,
                        prepared_rhs.get_element(n + j),
                    )?;
                }
                let gauge_defect = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(&residual[..k]),
                    kernels,
                )?;
                kernels.execution().check()?;
                Ok(BorderedSystemDiagnostics {
                    nullity: k,
                    compatibility_defect,
                    projection_norm,
                    original_residual_norm,
                    projected_residual_norm,
                    gauge_defect,
                })
            },
        )
    }
}

/// The bordered matrix `[A L; Rᴴ 0]` of a [`BorderedSystem`] as a linear operator.
///
/// The operator has order `n + k`. Its adjoint is `[Aᴴ R; Lᴴ 0]`. Each application
/// applies `A` (or `Aᴴ`) once and adds `O(n k)` work for the bordering.
pub struct BorderedSystemOperator<'p, 'a, S: NumericScalar> {
    plan: &'p BorderedSystem<'a, S>,
}
impl<S: NumericScalar> BorderedSystemOperator<'_, '_, S> {
    fn apply_forward(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: Option<&mut scioxide_core::OperatorContext<'_, '_, '_, S>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        crate::checked::expect_shape([self.cols()], [input.len()])?;
        crate::checked::expect_shape([self.rows()], [output.len()])?;
        crate::checked::expect_shape([self.apply_workspace_len()?], [workspace.len()])?;
        // Bordered operator `[A, L; R^H, 0]` for a singular A with declared
        // left/right null bases L, R: it is nonsingular when the bases are
        // exact, turns a consistent singular system into a regular one, and
        // the extra rows fix the gauge `R^H x = c`. The adjoint below is
        // `[A^H, R; L^H, 0]`.
        let n = self.plan.operator.rows();
        let (staged, scratch) = crate::workspace::split_mut(workspace, self.rows())?;
        let outcome = match context {
            Some(context) => self.plan.operator.apply_into_scoped(
                &input[..n],
                &mut staged[..n],
                scratch,
                context,
            ),
            None => self
                .plan
                .operator
                .apply_into(&input[..n], &mut staged[..n], scratch)
                .map(scioxide_core::Invocation::Evaluated),
        }?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
        crate::checked::provider_output(&staged[..n])?;
        combine_modes(&mut staged[..n], self.plan.left, &input[n..], false)?;
        for (j, value) in staged[n..].iter_mut().enumerate() {
            *value = dot(self.plan.right.column(j)?, &input[..n])?;
        }
        output.copy_from_slice(staged);
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
    fn apply_adjoint(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: Option<&mut scioxide_core::OperatorContext<'_, '_, '_, S>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        crate::checked::expect_shape([self.rows()], [input.len()])?;
        crate::checked::expect_shape([self.cols()], [output.len()])?;
        crate::checked::expect_shape([self.apply_adjoint_workspace_len()?], [workspace.len()])?;
        let n = self.plan.operator.rows();
        let (staged, scratch) = crate::workspace::split_mut(workspace, self.rows())?;
        let outcome = match context {
            Some(context) => self.plan.operator.apply_adjoint_into_scoped(
                &input[..n],
                &mut staged[..n],
                scratch,
                context,
            ),
            None => self
                .plan
                .operator
                .apply_adjoint_into(&input[..n], &mut staged[..n], scratch)
                .map(scioxide_core::Invocation::Evaluated),
        }?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
        crate::checked::provider_output(&staged[..n])?;
        combine_modes(&mut staged[..n], self.plan.right, &input[n..], false)?;
        for (j, value) in staged[n..].iter_mut().enumerate() {
            *value = dot(self.plan.left.column(j)?, &input[..n])?;
        }
        output.copy_from_slice(staged);
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
}
impl<S: NumericScalar> LinearOperator<S> for BorderedSystemOperator<'_, '_, S> {
    fn rows(&self) -> usize {
        self.plan.operator.rows() + self.plan.left.right_hand_sides()
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_add(
            self.rows(),
            self.plan.operator.apply_workspace_len()?,
            scioxide_core::quantity!("augmented_operator_scratch"),
        )
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_forward(input, output, workspace, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_forward(input, output, workspace, Some(context))
    }
}

impl<S: NumericScalar> AdjointOperator<S> for BorderedSystemOperator<'_, '_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_add(
            self.rows(),
            self.plan.operator.apply_adjoint_workspace_len()?,
            scioxide_core::quantity!("augmented_adjoint_scratch"),
        )
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_adjoint(input, output, workspace, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_adjoint(input, output, workspace, Some(context))
    }
}
