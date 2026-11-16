//! Complex-step differentiation: the method value, plans, workspaces and kernels.

use crate::nodes::{
    StencilScratch, configure_geometry, geometry_bytes, geometry_sufficient, node_identity,
    reserve_geometry,
};
use crate::options::DifferenceShape;
use crate::stencil::bound_for;
use crate::validation::{breakdown, invalid, length, shape, validate_direction, validate_problem};
use crate::workspace;
use crate::{
    DerivativeRequest, DerivativeSource, DifferentiationReport, DifferentiationResult,
    FiniteDifferenceStencil, SampleFunction, ZeroDirectionPolicy,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{
    Bounds, Capacity, Complex, DifferentiationLocation, ExecutionPolicy, InvalidInputReason,
    NonFiniteOrigin, NumericScalar, NumericalIssue, RangeAccumulator, RealScalar, SciOxideError,
    SolveBudgets, SolveState, checked_float_product, checked_float_quotient,
};
/// The kernel of one complex-step evaluation: the operation, the imaginary
/// step and, for second derivatives and Hessians, one real first-derivative
/// stencil per coordinate. Second derivatives and Hessians apply these real
/// stencils to complex-step gradients and have the usual finite-difference
/// truncation and rounding error.
#[derive(Debug, Clone)]
struct ComplexPlan<S: RealScalar> {
    operation: DifferenceShape,
    inputs: usize,
    step: S,
    outer: Vec<FiniteDifferenceStencil<S>>,
    outputs: usize,
    len: usize,
    max_nodes: usize,
}
/// Complex callback buffers and real scratch for the derivative before it is
/// written.
#[derive(Debug, Clone)]
struct ComplexBuffers<S: RealScalar> {
    point: Vec<Complex<S>>,
    callback: Vec<Complex<S>>,
    samples: Vec<S>,
    staged: Vec<S>,
    base_gradient: Vec<Option<S>>,
    reductions: Vec<[RangeAccumulator<S>; 2]>,
}
impl<S: RealScalar + NumericScalar<Real = S>> ComplexPlan<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    // The callers validate the step and build one outer stencil per coordinate
    // exactly for second derivatives and Hessians.
    fn new(
        kind: DifferenceShape,
        inputs: usize,
        step: S,
        outer: Vec<FiniteDifferenceStencil<S>>,
        outputs: usize,
    ) -> Result<Self, SciOxideError> {
        let len = shape(kind, inputs, outputs)?;
        let max_nodes = outer
            .iter()
            .map(|stencil| stencil.nodes.len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            operation: kind,
            inputs,
            step,
            outer,
            outputs,
            len,
            max_nodes,
        })
    }
    // `relative` stencils are copied into the workspace and rebound at each
    // point; fixed stencils stay in the plan.
    fn requirements(
        &self,
        relative: bool,
        limit: usize,
    ) -> Result<ComplexStepRequirements<'_, S>, SciOxideError> {
        requirements(
            [
                self.inputs,
                self.outputs,
                self.max_nodes,
                self.len,
                if self.operation == DifferenceShape::Hessian {
                    self.inputs
                } else {
                    0
                },
                usize::from(self.max_nodes != 0),
            ],
            if relative { &self.outer } else { &[] },
            limit,
        )
    }
    fn evaluate<F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        method: &ComplexStep<S>,
        relative: bool,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        let mut work = ComplexStepWorkspace::default();
        work.try_reserve(&self.requirements(relative, method.max_workspace_bytes)?)?;
        let mut output = try_filled(
            self.len,
            S::ZERO,
            scioxide_core::quantity!("derivative_output"),
        )?;
        let report = self.evaluate_into(
            method,
            relative,
            x,
            bounds,
            direction,
            function,
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_evaluation(report, output))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller output and borrowed resources remain explicit"
    )]
    fn evaluate_into<'scope, 'capability, F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        method: &ComplexStep<S>,
        relative: bool,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
        output: &mut [S],
        work: &mut ComplexStepWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        length(self.len, output.len())?;
        work.prepare(&self.requirements(relative, method.max_workspace_bytes)?)?;
        let ComplexStepWorkspace {
            data:
                ComplexWork {
                    outer,
                    scratch,
                    buffers,
                },
            state,
        } = work;
        let outer = if relative {
            for (axis, (stencil, template)) in outer.iter_mut().zip(&self.outer).enumerate() {
                stencil.rebind(x[axis], &template.nodes, bound_for(bounds, axis), scratch)?;
            }
            &outer[..self.outer.len()]
        } else {
            self.outer.as_slice()
        };
        let algorithm = if direction.is_some() {
            scioxide_core::algorithm_id!("differentiate", "complex-step-jvp", 1)
        } else {
            scioxide_core::algorithm_id!("differentiate", "complex-step", 1)
        };
        scope.into().run(
            state,
            node_identity(algorithm, self.len),
            method.budgets,
            |cx| {
                self.evaluate_samples(
                    x,
                    outer,
                    direction,
                    function,
                    output,
                    buffers,
                    cx,
                    method.execution,
                )
            },
        )
    }
    /// Evaluates the complex-step samples. A call at the base point checks that
    /// `f(x)` is real; this does not test holomorphy. The output is unchanged on
    /// any error or stop.
    ///
    /// # Errors
    /// Rejects incompatible output/scratch/direction, nonreal base output, callback
    /// failures, lost directional perturbations and unrepresentable arithmetic.
    #[expect(
        clippy::too_many_arguments,
        reason = "numerical inputs, caller-owned staging and execution capabilities stay explicit"
    )]
    #[expect(
        clippy::needless_range_loop,
        reason = "coupled numerical coordinates and mutable scratch require indexed traversal"
    )]
    fn evaluate_samples<F>(
        &self,
        point: &[S],
        outer: &[FiniteDifferenceStencil<S>],
        direction: Option<&[S]>,
        mut function: F,
        output: &mut [S],
        workspace: &mut ComplexBuffers<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
        policy: scioxide_core::ExecutionPolicy,
    ) -> Result<DifferentiationReport, SciOxideError>
    where
        F: SampleFunction<Complex<S>, Complex<S>>,
    {
        let n = point.len();
        let jvp = matches!(self.operation, DifferenceShape::Jvp(_));
        if self.len == 0 {
            let mut report =
                DifferentiationReport::fixed(DerivativeSource::ComplexStep, output.len());
            report.poll(context)?;
            return Ok(report);
        }
        if let Some(direction) = direction {
            let step = self.step;
            for &component in direction {
                let perturbation = step * component;
                if !perturbation.is_finite() || (component != S::ZERO && perturbation == S::ZERO) {
                    return Err(invalid(
                        scioxide_core::argument!("direction"),
                        "nonzero directional imaginary perturbations must survive",
                    ));
                }
            }
        }
        workspace.base_gradient.fill(None);
        let mut report = DifferentiationReport::fixed(DerivativeSource::ComplexStep, output.len());
        for (z, &x) in workspace.point.iter_mut().zip(point) {
            *z = Complex::new(x, S::ZERO);
        }
        let zero_direction = direction.is_some_and(|d| d.iter().all(|v| *v == S::ZERO));
        if zero_direction
            && self.operation == DifferenceShape::Jvp(ZeroDirectionPolicy::MathematicalZero)
        {
            report.completed_entries = output.len();
            if report.poll(context)? {
                output.fill(S::ZERO);
            }
            return Ok(report);
        }
        // f(x + i h) = f(x) + i h f'(x) + O(h^2) for an analytic extension of a
        // real function, so Im f(x + i h) / h gives f' with no subtractive
        // cancellation. The unperturbed call checks the one cheap consequence of
        // the contract that is testable, a real value at a real point.
        if !report.invoke(
            &workspace.point,
            &mut workspace.callback,
            &mut function,
            if jvp {
                DifferentiationLocation::direction()
            } else {
                DifferentiationLocation::base_point()
            },
            context,
            policy,
        )? {
            return Ok(report);
        }
        if workspace
            .callback
            .iter()
            .any(|v| v.imaginary_component() != S::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("function"),
                "analytic extension must be real-valued at the real base point",
            ));
        }
        if zero_direction {
            report.completed_entries = output.len();
            if report.poll(context)? {
                output.fill(S::ZERO);
            }
            return Ok(report);
        }
        if let Some(direction) = direction {
            if !report.iteration(context)? {
                return Ok(report);
            }
            // A single complex step along the direction gives J d = Im f(x + i h d) / h.
            // A component whose imaginary part underflows would silently drop out of
            // the product, so that is reported as a breakdown.
            let h = self.step;
            for (i, &d) in direction.iter().enumerate() {
                let imaginary = checked_float_product(h, d)?;
                if d != S::ZERO && imaginary == S::ZERO {
                    return Err(breakdown("directional imaginary perturbation underflowed"));
                }
                workspace.point[i] = Complex::new(point[i], imaginary);
            }
            if !report.invoke(
                &workspace.point,
                &mut workspace.callback,
                &mut function,
                DifferentiationLocation::direction(),
                context,
                policy,
            )? {
                return Ok(report);
            }
            for (value, z) in workspace.staged.iter_mut().zip(&workspace.callback) {
                *value = complex_quotient(z.imaginary_component(), h)?;
                report.completed_entries += 1;
            }
        } else if outer.is_empty() {
            for i in 0..n {
                if !report.iteration(context)? {
                    return Ok(report);
                }
                workspace.point[i] = Complex::new(point[i], self.step);
                if !report.invoke(
                    &workspace.point,
                    &mut workspace.callback,
                    &mut function,
                    DifferentiationLocation::variable(i),
                    context,
                    policy,
                )? {
                    return Ok(report);
                }
                for (lane, z) in workspace.callback.iter().enumerate() {
                    workspace.staged[lane * n + i] =
                        complex_quotient(z.imaginary_component(), self.step)?;
                    report.completed_entries += 1;
                }
                workspace.point[i] = Complex::new(point[i], S::ZERO);
            }
        } else {
            // Second derivatives: the complex step gives the exact-to-roundoff gradient
            // component df/dx_i at real nodes along axis j, and a real first-derivative
            // stencil along j turns it into d2f/dx_j dx_i. When i == j, both
            // perturbations act on the same coordinate. Every ordered pair is computed,
            // so each mixed partial has two independent estimates, which are averaged
            // below; this also makes the result exactly symmetric.
            for i in 0..n {
                for j in 0..n {
                    if !report.iteration(context)? {
                        return Ok(report);
                    }
                    for (a, &node) in outer[j].nodes.iter().enumerate() {
                        workspace.point[j] = Complex::new(node, S::ZERO);
                        let real = if i == j { node } else { point[i] };
                        workspace.point[i] = Complex::new(real, self.step);
                        if self.operation == DifferenceShape::Hessian
                            && crate::samples::same_bits(node, point[j])
                            && let Some(value) = workspace.base_gradient[i]
                        {
                            if !report.poll(context)? {
                                return Ok(report);
                            }
                            workspace.samples[a] = value;
                            workspace.point[i] = Complex::new(point[i], S::ZERO);
                            continue;
                        }
                        if !report.invoke(
                            &workspace.point,
                            &mut workspace.callback,
                            &mut function,
                            DifferentiationLocation::variables(i, j),
                            context,
                            policy,
                        )? {
                            return Ok(report);
                        }
                        workspace.samples[a] = complex_quotient(
                            workspace.callback[0].imaginary_component(),
                            self.step,
                        )?;
                        if self.operation == DifferenceShape::Hessian
                            && crate::samples::same_bits(node, point[j])
                        {
                            workspace.base_gradient[i] = Some(workspace.samples[a]);
                        }
                        workspace.point[i] = Complex::new(point[i], S::ZERO);
                    }
                    workspace.point[j] = Complex::new(point[j], S::ZERO);
                    let [real, imag] = &mut workspace.reductions[0];
                    workspace.staged[i * n + j] =
                        outer[j].apply_lane_reusing(|a| workspace.samples[a], real, imag)?;
                    report.completed_entries += 1;
                }
            }
            for i in 0..n {
                for j in i + 1..n {
                    let value = scioxide_core::checked_midpoint(
                        workspace.staged[i * n + j],
                        workspace.staged[j * n + i],
                    )?;
                    workspace.staged[i * n + j] = value;
                    workspace.staged[j * n + i] = value;
                }
            }
        }
        if report.poll(context)? {
            output.copy_from_slice(&workspace.staged);
        }
        Ok(report)
    }
}
fn complex_quotient<S: RealScalar>(imaginary: S, step: S) -> Result<S, SciOxideError> {
    let result = checked_float_quotient(imaginary, step)?;
    if imaginary != S::ZERO && (imaginary.abs() < S::MIN_POSITIVE || result == S::ZERO) {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Underflow,
            "complex-step derivative underflowed",
        ));
    }
    Ok(result)
}

/// Computes first derivatives by the complex step `∂f/∂xᵢ = Im f(x + ih eᵢ)/h`
/// for the analytic extension of a real function.
///
/// The callback is the extension `f: ℂⁿ → ℂᵐ` of a real function `ℝⁿ → ℝᵐ`;
/// `eᵢ` is the `i`-th unit vector and `h` the step. It must be holomorphic near
/// every real point where it is sampled and real on the real axis. Only the
/// second property is checked, at the base point `x`: code that uses
/// conjugation, absolute values or the real part, or that branches on the
/// imaginary part, usually breaks the assumption and gives wrong derivatives
/// without an error.
///
/// The formula has no subtractive cancellation. The step must make truncation
/// small while keeping the imaginary response normal: `|h·f′(x)| ≥ MIN_POSITIVE`.
/// A nonzero subnormal response returns `NumericalInstability` (`Underflow`).
/// An exactly zero response remains valid: a true zero derivative cannot be
/// distinguished from a response already rounded to zero inside the callback.
/// The relative truncation error of `Im f(x + ih)/h` is
/// `(h²/6)·|f‴(x)/f′(x)|`. The default step `h = ε` gives `8.2·10⁻³³·|f‴/f′|`
/// for `f64` and `2.4·10⁻¹⁵·|f‴/f′|` for `f32`, below one rounding error of `f`
/// while `|f‴/f′| ≤ 2·10¹⁶` (`f64`) or `5·10⁷` (`f32`), and it keeps `h·f′(x)`
/// normal for `|f′(x)| ≥ MIN_POSITIVE / ε` (about `1e-292` for `f64`).
/// A derivative, gradient or Jacobian is then accurate to
/// the rounding error of evaluating `f`.
///
/// A gradient or Jacobian costs `n + 1` evaluations: one at `x`, where `f` must
/// be real, and one per coordinate. A Jacobian-vector product
/// `J(x) v = Im f(x + ih v)/h` costs two. No error estimate is returned. For
/// second derivatives and Hessians use [`ComplexStepPlan`] or
/// [`RelativeComplexStep`].
///
/// The defaults are `h = ε`, sequential execution, unbounded budgets, no
/// workspace limit and [`ZeroDirectionPolicy::MathematicalZero`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex64;
/// use scioxide_differentiate::ComplexStep;
///
/// // f(x) = x₀ x₁, so ∇f(1, 2) = (2, 1) to rounding error.
/// let f = |z: &[Complex64], y: &mut [Complex64]| {
///     y[0] = z[0] * z[1];
///     Ok(())
/// };
/// let g = ComplexStep::default().gradient(f, &[1.0_f64, 2.0])?.into_values()?;
/// assert_eq!(g, [2.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - W. Squire and G. Trapp, "Using complex variables to estimate derivatives
///   of real functions", *SIAM Rev.* 40(1), 1998, 110–112.
/// - J. R. R. A. Martins, P. Sturdza and J. J. Alonso, "The complex-step
///   derivative approximation", *ACM Trans. Math. Softw.* 29(3), 2003,
///   245–262.
#[derive(Debug, Clone, Copy)]
pub struct ComplexStep<S: RealScalar> {
    step: S,
    execution: ExecutionPolicy,
    budgets: SolveBudgets,
    max_workspace_bytes: usize,
    zero_direction: ZeroDirectionPolicy,
}
impl<S: RealScalar> Default for ComplexStep<S> {
    fn default() -> Self {
        Self {
            step: S::EPSILON,
            execution: ExecutionPolicy::Sequential,
            budgets: SolveBudgets::default(),
            max_workspace_bytes: usize::MAX,
            zero_direction: ZeroDirectionPolicy::MathematicalZero,
        }
    }
}
impl<S: RealScalar> ComplexStep<S> {
    /// Sets the step `h` of every coordinate.
    ///
    /// Evaluation requires `MIN_POSITIVE ≤ h ≤ MAX`; smaller finite steps
    /// return `InvalidInput` (`OutOfRange`). For full precision the response
    /// must satisfy `|h·f′(x)| ≥ MIN_POSITIVE`, or approximately
    /// `|f′(x)| ≥ MIN_POSITIVE / h`. Nonzero subnormal responses return
    /// `NumericalInstability` (`Underflow`); responses rounded to zero inside
    /// the callback remain indistinguishable from true zero derivatives.
    pub const fn with_step(mut self, step: S) -> Self {
        self.step = step;
        self
    }
    /// Sets the execution policy for [`Parallel`](crate::Parallel) callbacks.
    pub const fn with_execution(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
    /// Sets the evaluation and iteration budgets of each evaluation. Limits of
    /// an enclosing solve still apply.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Limits the heap size of the workspace in bytes, including capacity left
    /// from earlier evaluations.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Sets what [`jvp`](Self::jvp) does when every component of the direction
    /// is zero. Other operations require the default.
    pub const fn with_zero_direction(mut self, policy: ZeroDirectionPolicy) -> Self {
        self.zero_direction = policy;
        self
    }
    // `argument` names the caller's method parameter.
    fn validate(&self, argument: scioxide_core::Argument, jvp: bool) -> Result<(), SciOxideError> {
        if !self.step.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "imaginary steps must be finite",
            ));
        }
        if self.step < S::MIN_POSITIVE {
            return Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::OutOfRange,
                "imaginary steps must be at least the smallest positive normal number",
            ));
        }
        if !jvp && self.zero_direction != ZeroDirectionPolicy::MathematicalZero {
            return Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::Inconsistent,
                "a zero-direction policy applies only to Jacobian-vector products",
            ));
        }
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ComplexStep<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    // Validates the settings for one operation; every call runs it.
    fn plan(
        &self,
        kind: DifferenceShape,
        x: &[S],
        direction: Option<&[S]>,
        outputs: usize,
    ) -> Result<ComplexPlan<S>, SciOxideError> {
        self.validate(scioxide_core::argument!("self"), direction.is_some())?;
        validate_problem(x, None, scioxide_core::argument!("x"))?;
        if let Some(direction) = direction {
            validate_direction(x, direction)?;
        }
        ComplexPlan::new(kind, x.len(), self.step, Vec::new(), outputs)
    }
    /// Computes `f′(x) = Im f(x + ih)/h` of a real function `ℝ → ℝ` from its
    /// analytic extension, and returns it with the report.
    ///
    /// It costs two evaluations: `f(x)`, which must be real, and `f(x + ih)`.
    /// The evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient`](Self::gradient) for one coordinate.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Complex64;
    /// use scioxide_differentiate::ComplexStep;
    ///
    /// // d/dx eˣ = 1 at x = 0, to rounding error.
    /// let result = ComplexStep::default().derivative(|z: Complex64| Ok(z.exp()), 0.0)?;
    /// assert!((result.into_values()?[0] - 1.0).abs() <= f64::EPSILON);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative<F>(
        &self,
        mut function: F,
        x: S,
    ) -> Result<DifferentiationResult<S>, SciOxideError>
    where
        F: FnMut(Complex<S>) -> Result<Complex<S>, SciOxideError>,
    {
        self.plan(DifferenceShape::First, &[x], None, 1)?.evaluate(
            self,
            false,
            &[x],
            None,
            None,
            |z: &[Complex<S>], y: &mut [Complex<S>]| -> Result<(), SciOxideError> {
                y[0] = function(z[0])?;
                Ok(())
            },
        )
    }
    /// Computes the gradient `∇f(x)` of a real function `ℝⁿ → ℝ` from its
    /// analytic extension, and returns it with the report.
    ///
    /// The evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into), except those of
    ///   `out` and `workspace`.
    /// - `AllocationFailure` if the workspace or the result cannot be
    ///   allocated.
    pub fn gradient<F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(DifferenceShape::Gradient, x, None, 1)?
            .evaluate(self, false, x, None, None, function)
    }
    /// Computes the gradient `∇f(x)` into `out`, using `workspace` for scratch.
    ///
    /// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) or a
    /// child [`SolveScope`](scioxide_core::SolveScope); [`Parallel`](crate::Parallel)
    /// callbacks run on its executor. With a sufficient workspace the
    /// evaluation does not allocate. `out` is written only if the report has
    /// output available.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Input`) if `x` or the step is NaN or infinite.
    /// - `InvalidInput` (`self`, `OutOfRange`) if the step is less than
    ///   `MIN_POSITIVE`.
    /// - `InvalidInput` (`self`, `Inconsistent`) if the zero-direction policy is
    ///   not the default.
    /// - `InvalidInput` (`function`, `UnsupportedConfiguration`) if `f(x)` has a
    ///   nonzero imaginary part.
    /// - `NumericalInstability` (`Underflow`) if a nonzero imaginary response
    ///   is subnormal or its quotient `Im f/h` rounds to zero.
    /// - `DimensionMismatch` if `out` does not have `n` entries or `workspace`
    ///   is too small.
    /// - `DifferentiationFailure` whose `source()` is the error returned by
    ///   `function` or `NonFinite` (origin `Callback`) for a NaN or infinite
    ///   value of `function`; it records the sample location and the
    ///   evaluation count.
    /// - `ResourceLimit` if the workspace exceeds `max_workspace_bytes`.
    pub fn gradient_into<'scope, 'capability, F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        out: &mut [S],
        workspace: &mut ComplexStepWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Gradient, x, None, 1)?
            .evaluate_into(self, false, x, None, None, function, out, workspace, scope)
    }
    /// Computes the Jacobian of a real function `ℝⁿ → ℝᵐ` with `m = outputs`,
    /// row-major `m × n`, from its analytic extension, and returns it with the
    /// report.
    ///
    /// Entry `i·n + j` is `∂fᵢ/∂xⱼ`.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient`](Self::gradient).
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Complex64;
    /// use scioxide_differentiate::ComplexStep;
    ///
    /// // f(x) = (x₀², x₀ x₁) has the Jacobian [[2x₀, 0], [x₁, x₀]] = [[2, 0], [3, 1]].
    /// let f = |z: &[Complex64], y: &mut [Complex64]| {
    ///     y[0] = z[0] * z[0];
    ///     y[1] = z[0] * z[1];
    ///     Ok(())
    /// };
    /// let j = ComplexStep::default().jacobian(f, &[1.0_f64, 3.0], 2)?.into_values()?;
    /// assert_eq!(j, [2.0, 0.0, 3.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jacobian<F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        outputs: usize,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(DifferenceShape::Jacobian, x, None, outputs)?
            .evaluate(self, false, x, None, None, function)
    }
    /// Computes the Jacobian of a real function `ℝⁿ → ℝᵐ` with `m = outputs`
    /// into `out`, using `workspace` for scratch.
    ///
    /// Arguments are as in [`gradient_into`](Self::gradient_into); `out` has
    /// `m·n` entries.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into).
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
    pub fn jacobian_into<'scope, 'capability, F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        outputs: usize,
        out: &mut [S],
        workspace: &mut ComplexStepWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Jacobian, x, None, outputs)?
            .evaluate_into(self, false, x, None, None, function, out, workspace, scope)
    }
    /// Computes the Jacobian-vector product `J(x) v = Im f(x + ih v)/h` with
    /// `v = direction` and `m = outputs`, and returns it with the report.
    ///
    /// It costs two evaluations: `f(x)`, which must be real, and `f(x + ih v)`.
    /// A zero direction follows [`with_zero_direction`](Self::with_zero_direction).
    ///
    /// # Errors
    ///
    /// - The errors of [`jacobian`](Self::jacobian), except the zero-direction
    ///   rule.
    /// - `DimensionMismatch` if `direction` does not have `n` entries.
    /// - `NonFinite` (origin `Input`) if `direction` contains NaN or infinity.
    /// - `InvalidInput` (`direction`, `UnsupportedConfiguration`) if `h·vᵢ`
    ///   overflows or underflows to zero for a nonzero `vᵢ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Complex64;
    /// use scioxide_differentiate::ComplexStep;
    ///
    /// // f(x) = x₀² + x₁², so J v = 2x₀v₀ + 2x₁v₁ = 22 at x = (1, 2), v = (3, 4).
    /// let f = |z: &[Complex64], y: &mut [Complex64]| {
    ///     y[0] = z[0] * z[0] + z[1] * z[1];
    ///     Ok(())
    /// };
    /// let jv = ComplexStep::default().jvp(f, &[1.0_f64, 2.0], &[3.0, 4.0], 1)?;
    /// assert_eq!(jv.into_values()?, [22.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jvp<F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        direction: &[S],
        outputs: usize,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.plan(
            DifferenceShape::Jvp(self.zero_direction),
            x,
            Some(direction),
            outputs,
        )?
        .evaluate(self, false, x, None, Some(direction), function)
    }
    /// Computes `J(x) v` with `v = direction` into `out`, using `workspace` for
    /// scratch.
    ///
    /// The number of outputs `m` is `out.len()`. Other arguments are as in
    /// [`gradient_into`](Self::gradient_into).
    ///
    /// # Errors
    ///
    /// - The errors of [`jvp`](Self::jvp), except allocation.
    /// - `DimensionMismatch` if `workspace` is too small.
    pub fn jvp_into<'scope, 'capability, F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        direction: &[S],
        out: &mut [S],
        workspace: &mut ComplexStepWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(
            DifferenceShape::Jvp(self.zero_direction),
            x,
            Some(direction),
            out.len(),
        )?
        .evaluate_into(
            self,
            false,
            x,
            None,
            Some(direction),
            function,
            out,
            workspace,
            scope,
        )
    }
}
/// Sizes the buffers of one evaluation; `outer` holds the stencils that the
/// workspace copies and rebinds at each point.
fn requirements<S: RealScalar + NumericScalar<Real = S>>(
    lengths: [usize; 6],
    outer: &[FiniteDifferenceStencil<S>],
    limit: usize,
) -> Result<ComplexStepRequirements<'_, S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut r = ComplexStepRequirements {
        outer,
        nodes: outer
            .iter()
            .map(|stencil| stencil.nodes.len())
            .max()
            .unwrap_or(0),
        lengths,
        bytes: 0,
        limit,
    };
    r.bytes = ComplexWork::default().bytes(Some(&r))?;
    workspace::memory_limit(r.bytes, limit)?;
    Ok(r)
}
/// Workspace size of a complex-step evaluation, returned by
/// [`ComplexStepWorkspace::workspace_requirements`],
/// [`ComplexStepPlan::workspace_requirements`] and
/// [`RelativeComplexStep::workspace_requirements`].
#[derive(Debug)]
pub struct ComplexStepRequirements<'a, S: RealScalar> {
    outer: &'a [FiniteDifferenceStencil<S>],
    nodes: usize,
    lengths: [usize; 6],
    bytes: usize,
    limit: usize,
}
impl<S: RealScalar> ComplexStepRequirements<'_, S> {
    /// Returns the heap size in bytes of a new workspace for the request.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
    fn geometry(&self) -> &[FiniteDifferenceStencil<S>] {
        self.outer
    }
    const fn nodes(&self) -> usize {
        self.nodes
    }
}
#[derive(Debug, Clone)]
struct ComplexWork<S: RealScalar> {
    outer: Vec<FiniteDifferenceStencil<S>>,
    scratch: StencilScratch<S>,
    buffers: ComplexBuffers<S>,
}
/// Scratch storage for [`ComplexStep`], [`ComplexStepPlan`] and
/// [`RelativeComplexStep`].
///
/// Create it with [`workspace`](Self::workspace) for a method value, with
/// `plan.workspace(())` for a plan, or with `Default` followed by
/// [`try_reserve`](Self::try_reserve). One workspace serves every evaluation
/// whose requirements it meets. Evaluation does not grow it.
#[derive(Debug, Clone)]
pub struct ComplexStepWorkspace<S: RealScalar> {
    data: ComplexWork<S>,
    state: SolveState,
}
impl<S: RealScalar> Default for ComplexStepWorkspace<S> {
    fn default() -> Self {
        Self {
            data: ComplexWork::default(),
            state: SolveState::new(),
        }
    }
}
impl<S: RealScalar> Default for ComplexWork<S> {
    fn default() -> Self {
        Self {
            outer: Vec::new(),
            scratch: StencilScratch {
                normalized: Vec::new(),
                ordered: Vec::new(),
                permutation: Vec::new(),
                table: Vec::new(),
            },
            buffers: ComplexBuffers {
                point: Vec::new(),
                callback: Vec::new(),
                samples: Vec::new(),
                staged: Vec::new(),
                base_gradient: Vec::new(),
                reductions: Vec::new(),
            },
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ComplexWork<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn capacities(&self) -> [usize; 6] {
        [
            self.buffers.point.capacity(),
            self.buffers.callback.capacity(),
            self.buffers.samples.capacity(),
            self.buffers.staged.capacity(),
            self.buffers.base_gradient.capacity(),
            self.buffers.reductions.capacity(),
        ]
    }
    fn sufficient(&self, r: &ComplexStepRequirements<'_, S>) -> Result<bool, SciOxideError> {
        let nodes = r.nodes();
        Ok(self
            .capacities()
            .iter()
            .zip(r.lengths)
            .all(|(&c, n)| c >= n)
            && geometry_sufficient(&self.outer, r.geometry())
            && self.scratch.normalized.capacity() >= nodes
            && self.scratch.ordered.capacity() >= nodes
            && self.scratch.permutation.capacity() >= nodes
            && self.scratch.table.capacity()
                >= checked_mul(nodes, 2, scioxide_core::quantity!("fornberg_table"))?)
    }
    fn bytes(&self, r: Option<&ComplexStepRequirements<'_, S>>) -> Result<usize, SciOxideError> {
        let lengths = r.map_or([0; 6], |r| r.lengths);
        let nodes = r.map_or(0, ComplexStepRequirements::nodes);
        let table = checked_mul(nodes, 2, scioxide_core::quantity!("fornberg_table"))?;
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<Complex<S>>(
                    [
                        (self.buffers.point.capacity(), lengths[0]),
                        (self.buffers.callback.capacity(), lengths[1]),
                    ],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<S>(
                    [
                        (self.buffers.samples.capacity(), lengths[2]),
                        (self.buffers.staged.capacity(), lengths[3]),
                    ],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<Option<S>>(
                    [(self.buffers.base_gradient.capacity(), lengths[4])],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<[RangeAccumulator<S>; 2]>(
                    [(self.buffers.reductions.capacity(), lengths[5])],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<S>(
                    [
                        (self.scratch.normalized.capacity(), nodes),
                        (self.scratch.ordered.capacity(), nodes),
                        (self.scratch.table.capacity(), table),
                    ],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<usize>(
                    [(self.scratch.permutation.capacity(), nodes)],
                    workspace::BYTES,
                )?,
                geometry_bytes(&self.outer, r.map(ComplexStepRequirements::geometry))?,
            ],
            workspace::BYTES,
        )
    }
    fn configure(&mut self, r: &ComplexStepRequirements<'_, S>) -> Result<(), SciOxideError> {
        let zero = Complex::new(S::ZERO, S::ZERO);
        self.buffers.point.resize(r.lengths[0], zero);
        self.buffers.callback.resize(r.lengths[1], zero);
        self.buffers.samples.resize(r.lengths[2], S::ZERO);
        self.buffers.staged.resize(r.lengths[3], S::ZERO);
        self.buffers.base_gradient.resize(r.lengths[4], None);
        self.buffers.reductions.resize(
            r.lengths[5],
            [RangeAccumulator::new(), RangeAccumulator::new()],
        );
        configure_geometry(&mut self.outer, r.geometry());
        let nodes = r.nodes();
        self.scratch.normalized.resize(nodes, S::ZERO);
        self.scratch.ordered.resize(nodes, S::ZERO);
        self.scratch.permutation.resize(nodes, 0);
        self.scratch.table.resize(
            checked_mul(nodes, 2, scioxide_core::quantity!("fornberg_table"))?,
            S::ZERO,
        );
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ComplexStepWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Returns the heap size in bytes of the buffers the workspace holds.
    ///
    /// Includes retained capacity from earlier requests and nested buffers.
    /// Performs no allocation.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.data.bytes(None)
    }

    /// Reports whether the workspace is large enough for `r`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace holds more than the method's
    ///   `max_workspace_bytes`.
    pub fn check(&self, r: &ComplexStepRequirements<'_, S>) -> Result<Capacity, SciOxideError> {
        workspace::check(self.data.bytes(None)?, r.limit, || self.data.sufficient(r))
    }

    /// Grows the workspace to meet `r`.
    ///
    /// The byte limit is checked before any buffer grows, counting capacity
    /// left from earlier evaluations. If an allocation fails, buffers grown before it
    /// are kept; call [`check`](Self::check) again before evaluating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace would exceed `max_workspace_bytes`.
    /// - `AllocationFailure` if a buffer cannot grow.
    pub fn try_reserve(&mut self, r: &ComplexStepRequirements<'_, S>) -> Result<(), SciOxideError> {
        let projected = self.data.bytes(Some(r))?;
        workspace::reserve(
            &mut self.data,
            projected,
            r.limit,
            |data| {
                let zero = Complex::new(S::ZERO, S::ZERO);
                workspace::reserve_buffer(
                    &mut data.buffers.point,
                    r.lengths[0],
                    zero,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.buffers.callback,
                    r.lengths[1],
                    zero,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.buffers.samples,
                    r.lengths[2],
                    S::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.buffers.staged,
                    r.lengths[3],
                    S::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.buffers.base_gradient,
                    r.lengths[4],
                    None,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.buffers.reductions,
                    r.lengths[5],
                    [RangeAccumulator::new(), RangeAccumulator::new()],
                    scioxide_core::quantity!("node_workspace"),
                )?;
                reserve_geometry(&mut data.outer, r.geometry())?;
                let nodes = r.nodes();
                workspace::reserve_buffer(
                    &mut data.scratch.normalized,
                    nodes,
                    S::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.scratch.ordered,
                    nodes,
                    S::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.scratch.permutation,
                    nodes,
                    0,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.scratch.table,
                    checked_mul(nodes, 2, scioxide_core::quantity!("fornberg_table"))?,
                    S::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                Ok(())
            },
            |data| data.bytes(None),
        )
    }

    fn prepare(&mut self, r: &ComplexStepRequirements<'_, S>) -> Result<(), SciOxideError> {
        workspace::prepare(self.check(r)?, r.bytes, || self.data.bytes(None))?;
        self.data.configure(r)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ComplexStepWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Returns the workspace size of `request`, without allocating.
    ///
    /// The size depends on the shape and on the method's
    /// [`max_workspace_bytes`](ComplexStep::with_max_workspace_bytes) only.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`request`, `UnsupportedConfiguration`) for a Hessian
    ///   shape, which [`ComplexStep`] does not compute.
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements(
        request: DerivativeRequest<'_, ComplexStep<S>>,
    ) -> Result<ComplexStepRequirements<'static, S>, SciOxideError> {
        let (method, shape) = (request.method(), request.shape());
        if shape.is_hessian() {
            return Err(invalid(
                scioxide_core::argument!("request"),
                "complex-step method values compute no Hessian",
            ));
        }
        let len = shape.output_len()?;
        requirements(
            [shape.inputs(), shape.outputs(), 0, len, 0, 0],
            &[],
            method.max_workspace_bytes,
        )
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace(
        request: DerivativeRequest<'_, ComplexStep<S>>,
    ) -> Result<Self, SciOxideError> {
        let mut work = Self::default();
        work.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(work)
    }
}

/// Computes the Hessian at a fixed point by applying real first-derivative
/// stencils on nodes you choose to complex-step gradients.
///
/// The plan is prepared at `point`, `n = point.len()`, with one node list per
/// coordinate. Construction computes the Fornberg weights of the
/// first-derivative stencil of each coordinate, `O(k²)` operations for `k`
/// nodes; every evaluation reads them. Entry `(i, j)` is the stencil along `j`
/// applied to the complex-step derivative `∂f/∂xᵢ` at the nodes of `j`, and each
/// mixed entry is the mean of its two orders, so the result is exactly
/// symmetric. An evaluation costs `1 + n·Σⱼ kⱼ` calls at most, with bit-identical
/// gradient samples at `point` evaluated once. The result has the truncation
/// and rounding error of the real stencils, not the accuracy of
/// [`ComplexStep`]. A second derivative is the case `n = 1`. For a changing
/// point use [`RelativeComplexStep`]. The callback contract is that of
/// [`ComplexStep`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex64;
/// use scioxide_differentiate::{ComplexStep, ComplexStepPlan};
///
/// // f(x) = x³, so f″(1) = 6; three nodes are exact for the quadratic f′.
/// let nodes = [0.5, 1.0, 1.5];
/// let plan = ComplexStepPlan::new(&[1.0_f64], &[&nodes], ComplexStep::default())?;
/// let f = |z: &[Complex64], y: &mut [Complex64]| {
///     y[0] = z[0] * z[0] * z[0];
///     Ok(())
/// };
/// let h = plan.evaluate(f, None)?.into_values()?;
/// assert!((h[0] - 6.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ComplexStepPlan<S: RealScalar> {
    template: ComplexPlan<S>,
    point: Vec<S>,
    method: ComplexStep<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> ComplexStepPlan<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Prepares the Hessian at `point` with the node list `nodes[i]` of
    /// coordinate `i` and the imaginary step of `method`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Input`) if the step, `point` or a node is NaN or
    ///   infinite.
    /// - `InvalidInput` (`method`, `OutOfRange`) if the step is less than
    ///   `MIN_POSITIVE`.
    /// - `InvalidInput` (`method`, `Inconsistent`) if the zero-direction policy
    ///   is not the default.
    /// - `DimensionMismatch` if `nodes` does not have one list per coordinate.
    /// - `InvalidInput` (`nodes`, `OutOfRange`) if a list has fewer than two
    ///   nodes, and (`nodes`, `Inconsistent`) if two nodes are equal.
    /// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(point: &[S], nodes: &[&[S]], method: ComplexStep<S>) -> Result<Self, SciOxideError> {
        method.validate(scioxide_core::argument!("method"), false)?;
        validate_problem(point, None, scioxide_core::argument!("point"))?;
        length(point.len(), nodes.len())?;
        shape(DifferenceShape::Hessian, point.len(), 1)?;
        let mut outer = scioxide_core::backend::try_with_capacity(
            nodes.len(),
            scioxide_core::quantity!("outer_stencils"),
        )?;
        for (&coordinate, &axis) in point.iter().zip(nodes) {
            outer.push(FiniteDifferenceStencil::prepare(
                coordinate,
                axis,
                1,
                scioxide_core::argument!("nodes"),
            )?);
        }
        Ok(Self {
            template: ComplexPlan::new(
                DifferenceShape::Hessian,
                point.len(),
                method.step,
                outer,
                1,
            )?,
            point: scioxide_core::backend::try_copy_slice(
                point,
                scioxide_core::quantity!("plan_point"),
            )?,
            method,
        })
    }
    /// Returns the number of output entries, `n²`.
    pub const fn output_len(&self) -> usize {
        self.template.len
    }
    /// Returns the workspace size this plan needs, without allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<ComplexStepRequirements<'_, S>, SciOxideError> {
        self.template
            .requirements(false, self.method.max_workspace_bytes)
    }
    /// Allocates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace(&self, request: ()) -> Result<ComplexStepWorkspace<S>, SciOxideError> {
        let mut work = ComplexStepWorkspace::default();
        work.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(work)
    }
    fn validate(&self, bounds: Option<&Bounds<S>>) -> Result<(), SciOxideError> {
        validate_problem(&self.point, bounds, scioxide_core::argument!("bounds"))?;
        for (axis, stencil) in self.template.outer.iter().enumerate() {
            if let Some((lower, upper)) = bound_for(bounds, axis)
                && stencil
                    .nodes
                    .iter()
                    .any(|&node| lower.is_some_and(|v| node < v) || upper.is_some_and(|v| node > v))
            {
                return Err(invalid(
                    scioxide_core::argument!("bounds"),
                    "complex-step nodes must lie in bounds",
                ));
            }
        }
        Ok(())
    }
    /// Computes the Hessian at the prepared point and returns it with the
    /// report.
    ///
    /// `bounds`, when given, must contain the point and every node. The
    /// evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`evaluate_into`](Self::evaluate_into), except those of
    ///   `out` and `workspace`.
    /// - `AllocationFailure` if the workspace or result cannot be allocated.
    pub fn evaluate<F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.validate(bounds)?;
        self.template
            .evaluate(&self.method, false, &self.point, bounds, None, function)
    }
    /// Computes the Hessian at the prepared point into `out`, using
    /// `workspace` for scratch.
    ///
    /// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) or a
    /// child [`SolveScope`](scioxide_core::SolveScope); [`Parallel`](crate::Parallel)
    /// callbacks run on its executor. With a sufficient workspace the
    /// evaluation does not allocate. `out` is written only if the report has
    /// output available.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds` does not have `n` entries, `out` does
    ///   not have `n²` entries or `workspace` is too small.
    /// - `InvalidInput` (`bounds`, `OutOfRange` or `UnsupportedConfiguration`)
    ///   if the point or a node lies outside `bounds`.
    /// - `InvalidInput` (`function`, `UnsupportedConfiguration`) if `f(point)`
    ///   has a nonzero imaginary part.
    /// - `NumericalInstability` if a complex-step quotient underflows or a
    ///   stencil sum leaves the finite range.
    /// - `DifferentiationFailure` whose `source()` is the error returned by
    ///   `function` or `NonFinite` (origin `Callback`) for a NaN or infinite
    ///   value of `function`; it records the sample location and the
    ///   evaluation count.
    /// - `ResourceLimit` if the workspace exceeds `max_workspace_bytes`.
    pub fn evaluate_into<'scope, 'capability, F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        bounds: Option<&Bounds<S>>,
        out: &mut [S],
        workspace: &mut ComplexStepWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate(bounds)?;
        self.template.evaluate_into(
            &self.method,
            false,
            &self.point,
            bounds,
            None,
            function,
            out,
            workspace,
            scope,
        )
    }
}

/// Computes the Hessian at any point by applying real first-derivative
/// stencils on `x + offsets` to complex-step gradients.
///
/// The method value holds one offset list per coordinate, `n = offsets.len()`.
/// At each evaluation the nodes `xᵢ + offsets[i][k]` are formed and the
/// Fornberg weights are recomputed from the displacements that remain after
/// rounding, `O(k²)` operations per coordinate and call for `k` offsets,
/// without allocating. Accuracy, cost and the callback contract are those of
/// [`ComplexStepPlan`]; a second derivative is the case `n = 1`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex64;
/// use scioxide_differentiate::{ComplexStep, RelativeComplexStep};
///
/// // f(x) = x³, so f″(2) = 12; three offsets are exact for the quadratic f′.
/// let offsets = [-0.5, 0.0, 0.5];
/// let method = RelativeComplexStep::new(&[&offsets], ComplexStep::default())?;
/// let f = |z: &[Complex64], y: &mut [Complex64]| {
///     y[0] = z[0] * z[0] * z[0];
///     Ok(())
/// };
/// let h = method.evaluate(f, &[2.0_f64], None)?.into_values()?;
/// assert!((h[0] - 12.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RelativeComplexStep<S: RealScalar> {
    template: ComplexPlan<S>,
    method: ComplexStep<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> RelativeComplexStep<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Binds the offset list `offsets[i]` of coordinate `i` and the imaginary
    /// step of `method`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Input`) if the step or an offset is NaN or
    ///   infinite.
    /// - `InvalidInput` (`method`, `OutOfRange`) if the step is less than
    ///   `MIN_POSITIVE`.
    /// - `InvalidInput` (`method`, `Inconsistent`) if the zero-direction policy
    ///   is not the default.
    /// - `InvalidInput` (`offsets`, `OutOfRange`) if a list has fewer than two
    ///   offsets, and (`offsets`, `Inconsistent`) if two offsets are equal.
    /// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
    /// - `AllocationFailure` if the offsets cannot be stored.
    pub fn new(offsets: &[&[S]], method: ComplexStep<S>) -> Result<Self, SciOxideError> {
        method.validate(scioxide_core::argument!("method"), false)?;
        shape(DifferenceShape::Hessian, offsets.len(), 1)?;
        let mut outer = scioxide_core::backend::try_with_capacity(
            offsets.len(),
            scioxide_core::quantity!("outer_stencils"),
        )?;
        for &axis in offsets {
            outer.push(FiniteDifferenceStencil::prepare(
                S::ZERO,
                axis,
                1,
                scioxide_core::argument!("offsets"),
            )?);
        }
        Ok(Self {
            template: ComplexPlan::new(
                DifferenceShape::Hessian,
                offsets.len(),
                method.step,
                outer,
                1,
            )?,
            method,
        })
    }
    /// Returns the number of output entries, `n²`.
    pub const fn output_len(&self) -> usize {
        self.template.len
    }
    /// Returns the workspace size of an evaluation, without allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<ComplexStepRequirements<'_, S>, SciOxideError> {
        self.template
            .requirements(true, self.method.max_workspace_bytes)
    }
    /// Allocates a workspace for the evaluations of this method value.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace(&self, request: ()) -> Result<ComplexStepWorkspace<S>, SciOxideError> {
        let mut work = ComplexStepWorkspace::default();
        work.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(work)
    }
    fn validate(&self, x: &[S], bounds: Option<&Bounds<S>>) -> Result<(), SciOxideError> {
        validate_problem(x, bounds, scioxide_core::argument!("x"))?;
        length(self.template.inputs, x.len())
    }
    /// Computes the Hessian at `x` and returns it with the report.
    ///
    /// The evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`evaluate_into`](Self::evaluate_into), except those of
    ///   `out` and `workspace`.
    /// - `AllocationFailure` if the workspace or result cannot be allocated.
    pub fn evaluate<F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<S>, SciOxideError> {
        self.validate(x, bounds)?;
        self.template
            .evaluate(&self.method, true, x, bounds, None, function)
    }
    /// Computes the Hessian at `x` into `out`, using `workspace` for scratch.
    ///
    /// The nodes are formed and checked before the first callback call.
    /// `scope` is as in [`ComplexStepPlan::evaluate_into`]. With a sufficient
    /// workspace the evaluation does not allocate. `out` is written only if the
    /// report has output available.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `bounds` does not have `n` entries,
    ///   `out` does not have `n²` entries or `workspace` is too small.
    /// - `NonFinite` (origin `Input`) if `x` contains NaN or infinity.
    /// - `InvalidInput` (`x`, `OutOfRange`) if `x` lies outside `bounds`.
    /// - `InvalidInput` (`x`, `UnsupportedConfiguration`) if a node
    ///   `xᵢ + offsets[i][k]` is not finite, lies outside `bounds`, or rounds
    ///   onto another node.
    /// - The errors of [`ComplexStepPlan::evaluate_into`] from the callback and
    ///   the arithmetic.
    pub fn evaluate_into<'scope, 'capability, F: SampleFunction<Complex<S>, Complex<S>>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        out: &mut [S],
        workspace: &mut ComplexStepWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate(x, bounds)?;
        self.template.evaluate_into(
            &self.method,
            true,
            x,
            bounds,
            None,
            function,
            out,
            workspace,
            scope,
        )
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use scioxide_core::SciOxideErrorKind;

    #[test]
    fn complex_buffers_and_geometry_children_require_each_slot() {
        let nodes = [-0.5, 0.0, 0.5];
        let axes = [nodes.as_slice(); 2];
        let method =
            RelativeComplexStep::new(&axes, ComplexStep::default().with_step(1e-100)).unwrap();
        let r = method.workspace_requirements(()).unwrap();
        for i in 0..12 {
            let mut work = method.workspace(()).unwrap();
            work.data.buffers.point.reserve(1024);
            work.data.buffers.staged.reserve(1024);
            let n = r.nodes();
            match i {
                0 => work.data.buffers.point = Vec::with_capacity(r.lengths[0] - 1),
                1 => work.data.buffers.callback = Vec::with_capacity(r.lengths[1] - 1),
                2 => work.data.buffers.samples = Vec::with_capacity(r.lengths[2] - 1),
                3 => work.data.buffers.staged = Vec::with_capacity(r.lengths[3] - 1),
                4 => work.data.buffers.base_gradient = Vec::with_capacity(r.lengths[4] - 1),
                5 => work.data.buffers.reductions = Vec::with_capacity(r.lengths[5] - 1),
                6 => work.data.outer[0].nodes = Vec::with_capacity(n - 1),
                7 => work.data.outer[0].weights = Vec::with_capacity(n - 1),
                8 => work.data.scratch.normalized = Vec::with_capacity(n - 1),
                9 => work.data.scratch.ordered = Vec::with_capacity(n - 1),
                10 => work.data.scratch.permutation = Vec::with_capacity(n - 1),
                _ => work.data.scratch.table = Vec::with_capacity(n * 2 - 1),
            }
            assert!(work.data.bytes(None).unwrap() > r.bytes());
            assert_eq!(work.check(&r).unwrap(), Capacity::NeedsGrowth);
            assert_eq!(
                work.prepare(&r).unwrap_err().kind(),
                SciOxideErrorKind::DimensionMismatch
            );
        }
    }
}
