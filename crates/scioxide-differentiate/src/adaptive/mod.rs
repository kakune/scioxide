//! Adaptive finite differences with Richardson or Ridders extrapolation.

use crate::options::DifferenceShape;
use crate::outcome::is_external_stop;
use crate::samples::HessianSamples;
use crate::sampling::stop;
use crate::stencil::{
    PositiveMagnitude, Stencil, axis_stencil_with_nominal, bound_for, directional_bounds,
    directional_stencil_for_points, generated_directional_point, mixed_stencil_derivative,
    stencil_derivative, update_minimum, vacant_stencil,
};
use crate::validation::{length, validate_direction, validate_problem};
use crate::workspace;
use crate::{
    DerivativeRequest, DerivativeSource, DifferenceScheme, DifferenceStep, DifferentiationReport,
    DifferentiationResult, DifferentiationTermination, SampleFunction, ZeroDirectionPolicy,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AlgebraicScalar, Bounds, DifferentiationLocation, ErrorTolerance, ExecutionPolicy,
    InvalidInputReason, NormSelection, NumericScalar, NumericalIssue, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, checked_float_product, checked_float_quotient,
    checked_product_quotient, checked_sum,
};
mod colored;
pub use colored::{ColoredJacobianPlan, ColoredJacobianRequirements, ColoredJacobianWorkspace};

/// Selects the extrapolation method of [`AdaptiveDifference`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdaptiveDifferenceMethod {
    /// Halves the step at each level and applies one Richardson extrapolation
    /// step to each pair of successive levels.
    Richardson,
    /// Ridders' method: reduces the step by a factor 1.4 at each level,
    /// extrapolates to increasing order in a Neville tableau, and returns the
    /// entry with the smallest error estimate.
    #[default]
    Ridders,
}

/// Computes derivatives and error estimates by adaptive differences with
/// extrapolation.
///
/// For each entry the method evaluates second-order stencils at up to `L`
/// decreasing steps and extrapolates the results to zero step (see
/// [`AdaptiveDifferenceMethod`]). It stops refining an entry when the error
/// estimate meets the tolerance or extrapolation stops improving, and stops the
/// evaluation if an entry gets no estimate at all. The best estimates and their
/// errors are returned when every entry has one, even if not all converged; see
/// [Results and termination](crate#results-and-termination). Callback values
/// may be real or complex.
///
/// A change of stencil geometry restarts extrapolation and its stagnation
/// history. The best estimate across geometries remains available as a fallback.
///
/// Each level costs two evaluations for each coordinate of a gradient or
/// Jacobian and a Jacobian-vector product. A diagonal Hessian entry costs two
/// evaluations for a central stencil and three for an inward one-sided stencil.
/// A mixed Hessian entry costs up to eight evaluations with one-sided stencils
/// (four when both are central), plus one shared evaluation of `f(x)`. Bit-identical
/// Hessian sample points are evaluated once; the workspace then holds up to
/// `1 + 8nL` axis samples and `16L` mixed samples for `n` inputs.
///
/// The error estimate of each entry is taken from the extrapolation table plus
/// a bound on the propagated rounding error. It is an estimate, not a bound. An
/// entry converges when its estimate is at most `atol + rtol·|D|` on two
/// consecutive levels and the best estimate also meets its tolerance, where
/// `D` is the entry. At least three levels are needed. A smooth function of
/// moderate size typically converges to a
/// relative error near `√ε` (`1.5·10⁻⁸` for `f64`, `3.5·10⁻⁴` for `f32`) or
/// better.
///
/// The first step `0.5 · max(1, |xᵢ|)` is large: near the edge of the domain of
/// `f`, such as `ln x` near zero, pass bounds or set a smaller step with
/// [`with_step`](Self::with_step). Near a bound the stencils become one-sided
/// second-order stencils that point into the domain. For coordinates of very
/// different scales, set an absolute step per coordinate with
/// [`DifferenceStep::absolute_coordinates`].
///
/// The defaults are [`AdaptiveDifferenceMethod::Ridders`], 16 levels, the
/// initial step `0.5 · max(1, |xᵢ|)`, the tolerance `atol = rtol = √ε` in the
/// maximum norm, sequential execution, unbounded budgets, no workspace limit
/// and [`ZeroDirectionPolicy::MathematicalZero`]. Each callback call counts as
/// one function evaluation and each level as one iteration.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::AdaptiveDifference;
///
/// // f(x) = x³, so f′(2) = 12, with an error estimate.
/// let result = AdaptiveDifference::default().derivative(|x: f64| Ok(x * x * x), 2.0)?;
/// assert!(result.report().is_success());
/// let (value, error) = (result.values().unwrap()[0], result.errors().unwrap()[0]);
/// assert!((value - 12.0).abs() < 1e-8 && error < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Even an exact central difference needs three levels to confirm convergence:
///
/// ```
/// use scioxide_differentiate::AdaptiveDifference;
/// let result = AdaptiveDifference::default().derivative(|x: f64| Ok(x * x), 1.0)?;
/// assert!((result.values().unwrap()[0] - 2.0).abs() < 1e-12);
/// assert!(result.report().is_success());
/// assert_eq!(result.report().iterations(), 3);
/// assert_eq!(result.report().evaluations(), 7);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. J. F. Ridders, "Accurate computation of F′(x) and F′(x)F″(x)",
///   *Adv. Eng. Softw.* 4(2), 1982, 75–76.
/// - W. H. Press et al., *Numerical Recipes*, 3rd ed., Cambridge, 2007, §5.7.
#[derive(Debug, Clone)]
pub struct AdaptiveDifference<S: RealScalar> {
    method: AdaptiveDifferenceMethod,
    tolerance: ErrorTolerance<S>,
    levels: usize,
    step: DifferenceStep<S>,
    execution: ExecutionPolicy,
    max_workspace_bytes: usize,
    budgets: SolveBudgets,
    zero_direction: ZeroDirectionPolicy,
}

impl<S: RealScalar> Default for AdaptiveDifference<S> {
    fn default() -> Self {
        Self {
            method: AdaptiveDifferenceMethod::Ridders,
            tolerance: ErrorTolerance::from_tolerance(
                scioxide_core::Tolerance::from_sqrt_epsilon(),
                NormSelection::Infinity,
            ),
            levels: 16,
            step: DifferenceStep::default(),
            execution: ExecutionPolicy::Sequential,
            max_workspace_bytes: usize::MAX,
            budgets: SolveBudgets::unbounded(),
            zero_direction: ZeroDirectionPolicy::MathematicalZero,
        }
    }
}

impl<S: RealScalar> AdaptiveDifference<S> {
    /// Sets the extrapolation method.
    pub fn with_method(mut self, method: AdaptiveDifferenceMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets the tolerance on the estimated error of each entry.
    ///
    /// An entry `D` converges when its error estimate is at most
    /// `atol + rtol·|D|` on two consecutive levels and the best estimate also
    /// meets its tolerance; at least three levels are needed.
    /// A vector absolute tolerance has one value per output
    /// entry, in output order. The norm must be the maximum norm, so that one
    /// inaccurate entry cannot hide in an average; every operation rejects
    /// other norms.
    pub fn with_tolerance(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of step sizes per entry.
    ///
    /// Every operation accepts 3 to 64 levels.
    pub fn with_max_levels(mut self, levels: usize) -> Self {
        self.levels = levels;
        self
    }

    /// Sets the initial step, replacing any earlier step selection. The default
    /// is `0.5 · max(1, |xᵢ|)`.
    pub fn with_step(mut self, step: DifferenceStep<S>) -> Self {
        self.step = step;
        self
    }
    /// Sets the execution policy for [`Parallel`](crate::Parallel) callbacks.
    pub const fn with_execution(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
    /// Limits the heap size of the workspace in bytes, including capacity left
    /// from earlier operations.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Sets the evaluation and iteration budgets of each evaluation. Limits of
    /// an enclosing solve still apply.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Sets what [`jvp`](Self::jvp) does when every component of the direction
    /// is zero. Other operations require the default.
    pub const fn with_zero_direction(mut self, policy: ZeroDirectionPolicy) -> Self {
        self.zero_direction = policy;
        self
    }
    /// Returns the initial step selection.
    pub const fn step(&self) -> &DifferenceStep<S> {
        &self.step
    }
    // Validates levels, norm and step for `inputs` coordinates and `entries`
    // outputs; `argument` names the caller's method parameter.
    pub(super) fn validate_shape(
        &self,
        inputs: usize,
        entries: usize,
        argument: scioxide_core::Argument,
    ) -> Result<(), SciOxideError> {
        if self.tolerance.norm() != NormSelection::Infinity || !(3..=64).contains(&self.levels) {
            return Err(invalid(
                argument,
                "adaptive derivatives require maximum-norm control and 3 through 64 levels",
            ));
        }
        self.step.validate(inputs)?;
        if let Some(absolute) = self.tolerance.absolute().as_vector() {
            length(entries, absolute.len())?;
        }
        Ok(())
    }
    pub(super) fn require_mathematical_zero(
        &self,
        argument: scioxide_core::Argument,
    ) -> Result<(), SciOxideError> {
        if self.zero_direction == ZeroDirectionPolicy::MathematicalZero {
            Ok(())
        } else {
            Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::Inconsistent,
                "a zero-direction policy applies only to Jacobian-vector products",
            ))
        }
    }
    // Validates the settings for one operation; every call runs it.
    fn plan(
        &self,
        kind: DifferenceShape,
        inputs: usize,
        outputs: usize,
    ) -> Result<AdaptivePlan<'_, S>, SciOxideError> {
        if !matches!(kind, DifferenceShape::Jvp(_)) {
            self.require_mathematical_zero(scioxide_core::argument!("self"))?;
        }
        AdaptivePlan::new(
            kind,
            inputs,
            outputs,
            self,
            scioxide_core::argument!("self"),
        )
    }

    /// Computes `f′(x)` of `f: ℝ → ℝ` or `ℝ → ℂ` and its error estimate, and
    /// returns them with the report.
    ///
    /// Central differences at the initial step and up to `L − 1` smaller steps
    /// are extrapolated until the estimated error meets the tolerance. If it
    /// does not converge, the best estimate is still returned and the report
    /// says why (see [Results and termination](crate#results-and-termination)).
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
    /// use scioxide_differentiate::AdaptiveDifference;
    ///
    /// // f(x) = sin x, so f′(0) = 1.
    /// let result = AdaptiveDifference::default().derivative(|x: f64| Ok(x.sin()), 0.0)?;
    /// assert!((result.into_values()?[0] - 1.0).abs() < 1e-10);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative<T, F>(
        &self,
        mut function: F,
        x: S,
    ) -> Result<DifferentiationResult<T>, SciOxideError>
    where
        T: NumericScalar<Real = S>,
        F: FnMut(S) -> Result<T, SciOxideError>,
    {
        self.plan(DifferenceShape::First, 1, 1)?.evaluate(
            &[x],
            None,
            None,
            |p: &[S], y: &mut [T]| -> Result<(), SciOxideError> {
                y[0] = function(p[0])?;
                Ok(())
            },
        )
    }
    /// Computes `f″(x)` of `f: ℝ → ℝ` or `ℝ → ℂ` and its error estimate, and
    /// returns them with the report.
    ///
    /// As [`derivative`](Self::derivative), applied to the central second
    /// difference `(f(x + h) − 2f(x) + f(x − h)) / h²`.
    ///
    /// # Errors
    ///
    /// - The errors of [`derivative`](Self::derivative).
    pub fn second_derivative<T, F>(
        &self,
        mut function: F,
        x: S,
    ) -> Result<DifferentiationResult<T>, SciOxideError>
    where
        T: NumericScalar<Real = S>,
        F: FnMut(S) -> Result<T, SciOxideError>,
    {
        self.plan(DifferenceShape::Second, 1, 1)?.evaluate(
            &[x],
            None,
            None,
            |p: &[S], y: &mut [T]| -> Result<(), SciOxideError> {
                y[0] = function(p[0])?;
                Ok(())
            },
        )
    }
    /// Computes the gradient `∇f(x)` of `f: ℝⁿ → ℝ` (or `ℂ`) and its error
    /// estimates, and returns them with the report.
    ///
    /// `bounds` restrict every sample point. The evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into), except those of
    ///   the output and workspace.
    /// - `AllocationFailure` if the workspace or the result cannot be
    ///   allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::AdaptiveDifference;
    ///
    /// // f(x) = x₀² + x₁², so ∇f(1, 2) = (2, 4).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[0] + x[1] * x[1];
    ///     Ok(())
    /// };
    /// let result = AdaptiveDifference::default().gradient(f, &[1.0, 2.0], None)?;
    /// assert!(result.report().is_success());
    /// let values = result.values().unwrap();
    /// assert!((values[0] - 2.0).abs() < 1e-8 && (values[1] - 4.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn gradient<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.plan(DifferenceShape::Gradient, x.len(), 1)?
            .evaluate(x, bounds, None, function)
    }
    /// Computes the gradient `∇f(x)` into `out` and its error estimates into
    /// `errors`, using `workspace` for scratch.
    ///
    /// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) or a
    /// child [`SolveScope`](scioxide_core::SolveScope); [`Parallel`](crate::Parallel)
    /// callbacks run on its executor. With a sufficient workspace the
    /// evaluation does not allocate. `out` and `errors` are written together,
    /// only if the report has output available.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`self`, `OutOfRange`) if the levels are outside 3 to
    ///   64 or the tolerance norm is not the maximum norm.
    /// - `InvalidInput` (`self`, `Inconsistent`) if the zero-direction policy is
    ///   not the default.
    /// - `DimensionMismatch` if `bounds`, `out`, `errors`, per-coordinate steps
    ///   or a vector absolute tolerance have the wrong length, or `workspace`
    ///   is too small.
    /// - `NonFinite` (origin `Input`) if `x` contains NaN or infinity.
    /// - `InvalidInput` (`x`, `OutOfRange`) if `x` lies outside `bounds`.
    /// - `DifferentiationFailure` whose `source()` is the error returned by
    ///   `function`, `NonFinite` (origin `Callback`) for a NaN or infinite value
    ///   of `function`, or `NumericalInstability` for a sample point or sum
    ///   outside the finite range; it records the sample location and the
    ///   evaluation count.
    /// - `ResourceLimit` if the workspace exceeds `max_workspace_bytes`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    pub fn gradient_into<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Gradient, x.len(), 1)?
            .evaluate_into(x, bounds, None, function, out, errors, workspace, scope)
    }
    /// Computes the Hessian of `f: ℝⁿ → ℝ` (or `ℂ`), row-major `n × n`, and
    /// its error estimates, and returns them with the report.
    ///
    /// Arguments are as in [`gradient`](Self::gradient).
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient`](Self::gradient).
    /// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::AdaptiveDifference;
    ///
    /// // f(x) = x₀² x₁ has the Hessian [[4, 2], [2, 0]] at (1, 2).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[0] * x[1];
    ///     Ok(())
    /// };
    /// let h = AdaptiveDifference::default().hessian(f, &[1.0, 2.0], None)?.into_values()?;
    /// for (value, expected) in h.iter().zip([4.0, 2.0, 2.0, 0.0]) {
    ///     assert!((value - expected).abs() < 1e-7);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn hessian<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.plan(DifferenceShape::Hessian, x.len(), 1)?
            .evaluate(x, bounds, None, function)
    }
    /// Computes the Hessian of `f: ℝⁿ → ℝ` (or `ℂ`) into `out` and its error
    /// estimates into `errors`, using `workspace` for scratch.
    ///
    /// Arguments are as in [`gradient_into`](Self::gradient_into); `out` and
    /// `errors` have `n²` entries.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into).
    /// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    pub fn hessian_into<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Hessian, x.len(), 1)?
            .evaluate_into(x, bounds, None, function, out, errors, workspace, scope)
    }
    /// Computes the Jacobian of `f: ℝⁿ → ℝᵐ` (or `ℂᵐ`) with `m = outputs`,
    /// row-major `m × n`, and its error estimates, and returns them with the
    /// report.
    ///
    /// Entry `i·n + j` is `∂fᵢ/∂xⱼ`. Other arguments are as in
    /// [`gradient`](Self::gradient).
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient`](Self::gradient).
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::AdaptiveDifference;
    ///
    /// // f(x) = (x₀ x₁, x₀ + x₁) has the Jacobian [[2, 1], [1, 1]] at (1, 2).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[1];
    ///     y[1] = x[0] + x[1];
    ///     Ok(())
    /// };
    /// let j = AdaptiveDifference::default().jacobian(f, &[1.0, 2.0], 2, None)?.into_values()?;
    /// for (value, expected) in j.iter().zip([2.0, 1.0, 1.0, 1.0]) {
    ///     assert!((value - expected).abs() < 1e-8);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jacobian<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        x: &[S],
        outputs: usize,
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.plan(DifferenceShape::Jacobian, x.len(), outputs)?
            .evaluate(x, bounds, None, function)
    }
    /// Computes the Jacobian of `f: ℝⁿ → ℝᵐ` (or `ℂᵐ`) with `m = outputs` into
    /// `out` and its error estimates into `errors`, using `workspace` for
    /// scratch.
    ///
    /// Arguments are as in [`gradient_into`](Self::gradient_into); `out` and
    /// `errors` have `m·n` entries.
    ///
    /// # Errors
    ///
    /// - The errors of [`gradient_into`](Self::gradient_into).
    /// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    pub fn jacobian_into<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        x: &[S],
        outputs: usize,
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(DifferenceShape::Jacobian, x.len(), outputs)?
            .evaluate_into(x, bounds, None, function, out, errors, workspace, scope)
    }
    /// Computes the Jacobian-vector product `J(x) v` of `f: ℝⁿ → ℝᵐ` (or `ℂᵐ`)
    /// with `v = direction` and `m = outputs`, and its error estimates, and
    /// returns them with the report.
    ///
    /// Each of the `m` entries is extrapolated along `v` and gets its own error
    /// estimate. The first step along `v` is the smallest initial step of a
    /// coordinate divided by `|vᵢ|`. A zero direction follows
    /// [`with_zero_direction`](Self::with_zero_direction).
    ///
    /// # Errors
    ///
    /// - The errors of [`jacobian`](Self::jacobian), except the zero-direction
    ///   rule.
    /// - `DimensionMismatch` if `direction` does not have `n` entries.
    /// - `NonFinite` (origin `Input`) if `direction` contains NaN or infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::AdaptiveDifference;
    ///
    /// // f(x) = x₀² + x₁, so J v = 2x₀v₀ + v₁ = 5 at x = (1, 2), v = (3, −1).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[0] + x[1];
    ///     Ok(())
    /// };
    /// let result = AdaptiveDifference::default().jvp(f, &[1.0, 2.0], &[3.0, -1.0], 1, None)?;
    /// assert!(result.report().is_success());
    /// assert!((result.values().unwrap()[0] - 5.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jvp<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        x: &[S],
        direction: &[S],
        outputs: usize,
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.plan(DifferenceShape::Jvp(self.zero_direction), x.len(), outputs)?
            .evaluate(x, bounds, Some(direction), function)
    }
    /// Computes `J(x) v` with `v = direction` into `out` and its error
    /// estimates into `errors`, using `workspace` for scratch.
    ///
    /// The number of outputs `m` is `out.len()`. Other arguments are as in
    /// [`jacobian_into`](Self::jacobian_into).
    ///
    /// # Errors
    ///
    /// - The errors of [`jvp`](Self::jvp), except allocation.
    /// - `DimensionMismatch` if `errors` does not have `m` entries or
    ///   `workspace` is too small.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    pub fn jvp_into<'scope, 'capability, T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        x: &[S],
        direction: &[S],
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.plan(
            DifferenceShape::Jvp(self.zero_direction),
            x.len(),
            out.len(),
        )?
        .evaluate_into(
            x,
            bounds,
            Some(direction),
            function,
            out,
            errors,
            workspace,
            scope,
        )
    }
}

/// One operation of an [`AdaptiveDifference`] with its shape, borrowed for a
/// call. Coordinates are real; callback values may be real or complex.
#[derive(Debug, Clone, Copy)]
pub(super) struct AdaptivePlan<'a, S: RealScalar> {
    operation: DifferenceShape,
    inputs: usize,
    outputs: usize,
    len: usize,
    options: &'a AdaptiveDifference<S>,
}

/// Scratch sized for one operation. Construction allocates; evaluation does not.
/// For Hessians it caches axis samples and the mixed samples of the current
/// coordinate pair; the cache is cleared on each call, including after an error
/// or a panic in the callback.
#[derive(Debug, Clone)]
pub(super) struct AdaptiveBuffers<T: NumericScalar> {
    operation: DifferenceShape,
    inputs: usize,
    outputs: usize,
    levels: usize,
    point: Vec<T::Real>,
    previous_point: Vec<T::Real>,
    sample_points: Vec<T::Real>,
    samples: Vec<T>,
    base: Vec<T>,
    table: Vec<T>,
    roundoff: Vec<T::Real>,
    best: Vec<T>,
    best_error: Vec<T::Real>,
    geometry_error: Vec<T::Real>,
    values: Vec<T>,
    errors: Vec<T::Real>,
    arithmetic: Vec<AdaptiveArithmetic<T>>,
}

#[derive(Debug, Clone)]
struct AdaptiveArithmetic<T: NumericScalar> {
    retained: HessianSamples<T>,
    real: scioxide_core::RangeAccumulator<T::Real>,
    imaginary: scioxide_core::RangeAccumulator<T::Real>,
}

// Private helpers report failures.
fn invalid(argument: scioxide_core::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::OutOfRange, detail)
}

impl<'a, S: RealScalar> AdaptivePlan<'a, S> {
    /// Prepare shape and options. Scalar operations require one callback output;
    /// first/second require one input. Zero-dimensional gradients/Jacobians/Hessians
    /// are valid empty outputs and perform no callbacks.
    ///
    /// # Errors
    /// Rejects invalid shapes, tolerance dimensions, or unrepresentable storage sizes.
    pub(super) fn new(
        kind: DifferenceShape,
        inputs: usize,
        outputs: usize,
        options: &'a AdaptiveDifference<S>,
        argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        let len = crate::validation::shape(kind, inputs, outputs)?;
        options.validate_shape(inputs, len, argument)?;
        Ok(Self {
            operation: kind,
            inputs,
            outputs,
            len,
            options,
        })
    }

    fn evaluate<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.validate_inputs(x, bounds, direction)?;
        let mut workspace = AdaptiveDifferenceWorkspace::default();
        workspace.try_reserve(&self.requirements()?)?;
        let mut values = try_filled(
            self.len,
            T::ZERO,
            scioxide_core::quantity!("derivative_output"),
        )?;
        let mut errors = try_filled(self.len, S::ZERO, scioxide_core::quantity!("error_output"))?;
        let report = self.evaluate_validated(
            x,
            bounds,
            direction,
            function,
            &mut values,
            &mut errors,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_estimates(
            report, values, errors,
        ))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    fn evaluate_into<'scope, 'capability, T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: F,
        output: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate_inputs(x, bounds, direction)?;
        self.evaluate_validated(
            x, bounds, direction, function, output, errors, workspace, scope,
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "validated problem data, caller buffers and execution stay explicit"
    )]
    fn evaluate_validated<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        mut function: F,
        output: &mut [T],
        errors: &mut [S],
        workspace: &mut AdaptiveDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        workspace.prepare(self)?;
        self.validate_workspace(output.len(), errors.len(), &workspace.data)?;
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            if direction.is_some() {
                scioxide_core::algorithm_id!("differentiate", "adaptive-jvp", 1)
            } else {
                scioxide_core::algorithm_id!("differentiate", "adaptive", 1)
            },
            self.len as u128,
            0,
        );
        scope.into().run(
            &mut workspace.state,
            identity,
            self.options.budgets,
            |context| {
                self.evaluate_samples(
                    x,
                    bounds,
                    direction,
                    &mut function,
                    output,
                    errors,
                    &mut workspace.data,
                    context,
                )
            },
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    pub(super) fn evaluate_samples<T, F>(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: &mut F,
        output: &mut [T],
        error_output: &mut [S],
        workspace: &mut AdaptiveBuffers<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError>
    where
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    {
        // Both public entry points validate once before entering the solve.
        workspace.arithmetic[0].retained.clear();
        let mut report = DifferentiationReport {
            termination: DifferentiationTermination::Converged,
            source: DerivativeSource::AdaptiveDifference,
            evaluations: 0,
            iterations: 0,
            completed_entries: 0,
            output_len: self.len,
        };
        let zero_direction = direction.is_some_and(|d| d.iter().all(|&v| v == S::ZERO));
        if self.len == 0 {
            if let Some(reason) = stop(context)? {
                report.termination = reason;
            }
            return Ok(report);
        }
        if (!zero_direction
            || self.operation == DifferenceShape::Jvp(ZeroDirectionPolicy::ValidateBase))
            && let Some(stop) = function.evaluate(
                x,
                &mut workspace.base,
                &[true],
                self.inputs,
                self.outputs,
                context,
                &mut report,
                if direction.is_some() {
                    DifferentiationLocation::direction()
                } else {
                    DifferentiationLocation::base_point()
                },
                self.options.execution,
            )?
        {
            report.termination = stop;
            return Ok(report);
        }
        if zero_direction {
            if let Some(reason) = stop(context)? {
                report.termination = reason;
                return Ok(report);
            }
            output.fill(T::ZERO);
            error_output.fill(S::ZERO);
            report.completed_entries = self.len;
            return Ok(report);
        }
        if self.operation == DifferenceShape::Hessian {
            workspace.arithmetic[0]
                .retained
                .insert(x, x, workspace.base[0])?;
        }
        let tasks = if matches!(self.operation, DifferenceShape::Jvp(_)) {
            1
        } else {
            self.inputs
        };
        'entries: for first in 0..tasks {
            let last = if self.operation == DifferenceShape::Hessian {
                self.inputs
            } else {
                first + 1
            };
            for second in first..last {
                let (termination, available) = self.refine(
                    first,
                    second,
                    x,
                    bounds,
                    direction,
                    function,
                    workspace,
                    context,
                    &mut report,
                )?;
                if available {
                    for lane in 0..self.outputs {
                        let index = self.index(first, second, lane);
                        workspace.values[index] = workspace.best[lane];
                        workspace.errors[index] = workspace.best_error[lane];
                        report.completed_entries += 1;
                        if self.operation == DifferenceShape::Hessian && first != second {
                            workspace.values[second * self.inputs + first] = workspace.best[lane];
                            workspace.errors[second * self.inputs + first] =
                                workspace.best_error[lane];
                            report.completed_entries += 1;
                        }
                    }
                }
                // The report keeps the first numerical termination, but an external stop
                // (budget, cancellation, deadline) always wins because it ends all further
                // work. An entry that stopped short but still has a best estimate lets the
                // remaining entries proceed; an entry without any estimate makes a
                // complete output impossible.
                if termination != DifferentiationTermination::Converged
                    && (report.termination == DifferentiationTermination::Converged
                        || is_external_stop(termination))
                {
                    report.termination = termination;
                }
                if is_external_stop(termination) || !available {
                    break 'entries;
                }
            }
        }
        // A final callback can request cancellation on the same level that hits
        // a numerical/refinement limit. Observe that stable publication boundary.
        if let Some(reason) = stop(context)? {
            report.termination = reason;
        }
        if report.output_available() {
            output.copy_from_slice(&workspace.values);
            error_output.copy_from_slice(&workspace.errors);
        }
        Ok(report)
    }

    fn index(&self, first: usize, second: usize, lane: usize) -> usize {
        match self.operation {
            DifferenceShape::Jacobian => lane * self.inputs + first,
            DifferenceShape::Hessian => first * self.inputs + second,
            DifferenceShape::Jvp(_) => lane,
            _ => first,
        }
    }
    fn validate_workspace<T: NumericScalar<Real = S>>(
        &self,
        output: usize,
        errors: usize,
        workspace: &AdaptiveBuffers<T>,
    ) -> Result<(), SciOxideError> {
        length(self.len, output)?;
        length(self.len, errors)?;
        if workspace.operation != self.operation
            || workspace.inputs != self.inputs
            || workspace.outputs != self.outputs
            || workspace.levels != self.options.levels
            || workspace.values.len() != self.len
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace state does not match prepared shape",
            ));
        }
        Ok(())
    }

    fn validate_inputs(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
    ) -> Result<(), SciOxideError> {
        length(self.inputs, x.len())?;
        validate_problem(x, bounds, scioxide_core::argument!("x"))?;
        direction.map_or(Ok(()), |direction| validate_direction(x, direction))
    }
}

/// Preserve a fixed inward/symmetric geometry before rounding, so extrapolation
/// has a valid power series even when a coordinate is close to its boundary.
#[inline]
fn adaptive_axis<S: RealScalar>(
    x: S,
    mut step: S,
    order: usize,
    bound: Option<(Option<S>, Option<S>)>,
) -> Result<Option<Stencil<S>>, SciOxideError> {
    // Finite representability is itself a bound, including when the caller has
    // supplied no mathematical bounds. Keep one-sided offsets representable.
    let (lower, upper) = bound.unwrap_or((None, None));
    let bound = Some((Some(lower.unwrap_or(S::MIN)), Some(upper.unwrap_or(S::MAX))));
    // The one-sided fallback reaches (order + 1) * step, and that offset must
    // stay finite.
    step = step.min(S::MAX / S::try_from_usize_exact(order + 1)?);
    if !step.is_finite() || step <= S::ZERO {
        return Ok(None);
    }
    let mut stencil = vacant_stencil();
    axis_stencil_with_nominal(
        x,
        step,
        DifferenceScheme::Central,
        order,
        bound,
        &mut stencil,
    )
    .map(|available| available.map(|()| stencil))
}

// Symmetric stencils have error expansions in even powers of h only, which
// lets each extrapolation column remove two orders. Rounded offsets are
// never exactly opposite, so they count as symmetric when they agree to
// sqrt(eps) relative to the stencil scale.
fn symmetric<S: RealScalar>(stencil: Stencil<S>) -> bool {
    stencil.len == 3
        && stencil.offsets[0] < S::ZERO
        && (stencil.offsets[0] / stencil.scale + stencil.offsets[2] / stencil.scale).abs()
            <= S::SQRT_EPSILON
}

// Extrapolation assumes that successive levels are the same stencil
// shape scaled by a constant factor. Clipping at a bound or rounding can
// change the shape, and then the tableau must restart.
fn same_geometry<S: RealScalar>(a: Stencil<S>, b: Stencil<S>) -> bool {
    let allowance = S::SQRT_EPSILON * (S::TWO);
    a.len == b.len
        && a.offsets[..a.len]
            .iter()
            .zip(&b.offsets[..b.len])
            .all(|(&x, &y)| (x / a.scale - y / b.scale).abs() <= allowance)
}

fn initial_step<S: RealScalar>(x: S, index: usize, step: &DifferenceStep<S>) -> S {
    step.magnitude(x, index, S::HALF * x.abs().max(S::ONE))
}

impl<S: RealScalar> AdaptivePlan<'_, S> {
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    fn refine<T, F>(
        &self,
        first: usize,
        second: usize,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        direction: Option<&[S]>,
        function: &mut F,
        w: &mut AdaptiveBuffers<T>,
        context: &mut SolveContext<'_, '_>,
        report: &mut DifferentiationReport,
    ) -> Result<(DifferentiationTermination, bool), SciOxideError>
    where
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    {
        use DifferentiationTermination::*;
        let ratio = match self.options.method {
            AdaptiveDifferenceMethod::Richardson => S::TWO,
            AdaptiveDifferenceMethod::Ridders => S::try_from_f64_lossy(1.4)?,
        };
        let mixed = self.operation == DifferenceShape::Hessian && first != second;
        let order = if self.operation == DifferenceShape::Second
            || (self.operation == DifferenceShape::Hessian && !mixed)
        {
            2
        } else {
            1
        };
        let mut h1 = x
            .get(first)
            .map_or(S::ONE, |&v| initial_step(v, first, &self.options.step));
        let mut h2 = x
            .get(second)
            .map_or(S::ONE, |&v| initial_step(v, second, &self.options.step));
        // For a JVP, the initial alpha is the smallest per-coordinate step divided
        // by |d_i|, so no coordinate moves farther than its own nominal step.
        let directional = if let Some(direction) = direction {
            let mut nominal = None;
            for (i, (&value, &component)) in x.iter().zip(direction).enumerate() {
                if component != S::ZERO {
                    update_minimum(
                        &mut nominal,
                        Some(PositiveMagnitude::ratio(
                            initial_step(value, i, &self.options.step),
                            component.abs(),
                        )?),
                    );
                }
            }
            Some((
                nominal.ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("direction"),
                        "nonzero direction required",
                    )
                })?,
                directional_bounds(x, direction, bounds)?,
            ))
        } else {
            None
        };
        let mut directional = directional;
        let mut progress = RefinementProgress::new();
        w.arithmetic[0].retained.clear_pair();
        for _ in 0..self.options.levels {
            if let Some(reason) = stop(context)? {
                return Ok((reason, progress.available));
            }
            if !context.record_iteration()?.is_granted() {
                return Ok((IterationLimit, progress.available));
            }
            report.iterations = checked_add(
                report.iterations,
                1,
                scioxide_core::quantity!("differentiation_iterations"),
            )?;
            let a = if let (Some((nominal, room)), Some(direction)) = (directional, direction) {
                directional_stencil_for_points(
                    nominal,
                    DifferenceScheme::Central,
                    room,
                    x,
                    direction,
                    bounds,
                    (&mut w.point, &mut w.previous_point),
                )?
            } else {
                adaptive_axis(x[first], h1, order, bound_for(bounds, first))?
            };
            let Some(a) = a else {
                return Ok((StepSizeLimit, progress.available));
            };
            let b = if mixed {
                let Some(b) = adaptive_axis(x[second], h2, 1, bound_for(bounds, second))? else {
                    return Ok((StepSizeLimit, progress.available));
                };
                Some(b)
            } else {
                None
            };
            let Some(contraction) = progress.prepare(a, b, ratio) else {
                return Ok((StepSizeLimit, progress.available));
            };
            // Stage the whole level first. The base sample is copied instead of being
            // re-evaluated, and `active` tells the sampler which slots need a callback.
            let count = a.len * b.map_or(1, |b| b.len);
            let mut active = [false; 16];
            let mut aliases = [None; 16];
            for sample in 0..count {
                let (ia, ib) = b.map_or((sample, 0), |b| (sample / b.len, sample % b.len));
                let offset_a = a.offsets[ia];
                let offset_b = b.map_or(S::ZERO, |b| b.offsets[ib]);
                let values = &mut w.samples[sample * self.outputs..(sample + 1) * self.outputs];
                if offset_a == S::ZERO && offset_b == S::ZERO {
                    values.copy_from_slice(&w.base);
                    continue;
                }
                active[sample] = true;
                let point = &mut w.sample_points[sample * self.inputs..(sample + 1) * self.inputs];
                point.copy_from_slice(x);
                if let Some(direction) = direction {
                    if !generated_directional_point(x, direction, offset_a, bounds, point)? {
                        return Ok((StepSizeLimit, progress.available));
                    }
                } else {
                    point[first] = x[first] + offset_a;
                    if mixed {
                        point[second] = x[second] + offset_b;
                    }
                }
                if self.operation == DifferenceShape::Hessian {
                    if let Some(saved) = w.arithmetic[0].retained.find(x, point)? {
                        values[0] = saved;
                        active[sample] = false;
                    } else {
                        for earlier in 0..sample {
                            let current =
                                &w.sample_points[sample * self.inputs..(sample + 1) * self.inputs];
                            let previous = &w.sample_points
                                [earlier * self.inputs..(earlier + 1) * self.inputs];
                            if active[earlier]
                                && current
                                    .iter()
                                    .zip(previous)
                                    .all(|(&a, &b)| crate::samples::same_bits(a, b))
                            {
                                aliases[sample] = Some(earlier);
                                active[sample] = false;
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(reason) = function.evaluate(
                &w.sample_points[..count * self.inputs],
                &mut w.samples[..count * self.outputs],
                &active[..count],
                self.inputs,
                self.outputs,
                context,
                report,
                if direction.is_some() {
                    DifferentiationLocation::direction()
                } else if b.is_some() {
                    DifferentiationLocation::variables(first, second)
                } else {
                    DifferentiationLocation::variable(first)
                },
                self.options.execution,
            )? {
                return Ok((reason, progress.available));
            }
            if self.operation == DifferenceShape::Hessian {
                for sample in 0..count {
                    let start = sample * self.outputs;
                    if let Some(earlier) = aliases[sample] {
                        w.samples.copy_within(
                            earlier * self.outputs..(earlier + 1) * self.outputs,
                            start,
                        );
                    } else if active[sample] {
                        w.arithmetic[0].retained.insert(
                            x,
                            &w.sample_points[sample * self.inputs..(sample + 1) * self.inputs],
                            w.samples[start],
                        )?;
                    }
                }
            }
            let (all_satisfied, improved) = self.update_table(
                first,
                second,
                a,
                b,
                progress.depth,
                contraction,
                progress.available,
                w,
            )?;
            // A callback may have requested cancellation on this same level; report
            // that in preference to the numerical outcome.
            if let Some(reason) = progress.observe(all_satisfied, improved) {
                return Ok((stop(context)?.unwrap_or(reason), progress.available));
            }
            // Use the clipped actual scale to avoid wasting levels on identical
            // boundary stencils. One-sided stencils span len-1 elementary steps.
            h1 = a.scale
                / if a.offsets[0] == S::ZERO {
                    S::try_from_usize_exact(a.len - 1)?
                } else {
                    S::ONE
                }
                / ratio;
            if let Some(b) = b {
                h2 = b.scale
                    / if b.offsets[0] == S::ZERO {
                        S::try_from_usize_exact(b.len - 1)?
                    } else {
                        S::ONE
                    }
                    / ratio;
            }
            if let Some((nominal, room)) = directional {
                directional = Some((
                    nominal.divide(PositiveMagnitude::from_positive(ratio)?)?,
                    room,
                ));
            }
        }
        Ok((RefinementLimit, progress.available))
    }
}

/// Shared refinement history; numerical table storage stays in the workspace.
#[derive(Debug, Clone)]
struct RefinementProgress<S: RealScalar> {
    previous: Option<(Stencil<S>, Option<Stencil<S>>)>,
    depth: usize,
    available: bool,
    confirmations: usize,
    stale: usize,
}
impl<S: RealScalar> RefinementProgress<S> {
    fn new() -> Self {
        Self {
            previous: None,
            depth: 0,
            available: false,
            confirmations: 0,
            stale: 0,
        }
    }
    fn prepare(&mut self, a: Stencil<S>, b: Option<Stencil<S>>, ratio: S) -> Option<S> {
        let mut contraction = ratio;
        if let Some((old_a, old_b)) = self.previous {
            // Clipping at a bound or rounding reproduced the previous stencil, so
            // another level would add no information.
            if a.offsets == old_a.offsets && b.map(|v| v.offsets) == old_b.map(|v| v.offsets) {
                return None;
            }
            // Extrapolate with the ratio that was actually realized, not the nominal
            // one, since clipping and rounding change it. If the geometry changed, or
            // the step failed to shrink, the older columns do not share an asymptotic
            // series with the new level, so the tableau restarts at depth zero.
            contraction = old_a.scale / a.scale;
            let compatible = same_geometry(a, old_a)
                && symmetric(a) == symmetric(old_a)
                && match (b, old_b) {
                    (Some(b), Some(old)) => {
                        same_geometry(b, old)
                            && ((old.scale / b.scale) / contraction - S::ONE).abs()
                                <= S::SQRT_EPSILON
                    }
                    (None, None) => true,
                    _ => false,
                };
            if !compatible || !contraction.is_finite() || contraction <= S::ONE {
                self.depth = 0;
                self.confirmations = 0;
                self.stale = 0;
            }
        }
        self.previous = Some((a, b));
        Some(contraction)
    }
    fn observe(&mut self, satisfied: bool, improved: bool) -> Option<DifferentiationTermination> {
        // An estimate exists after the first extrapolated level. Convergence
        // requires two consecutive satisfied levels so that one lucky agreement is
        // not accepted. Three extrapolated levels in the current geometry without
        // a better error mean roundoff dominates truncation; shrinking only hurts.
        self.available |= self.depth > 0;
        self.confirmations = if satisfied { self.confirmations + 1 } else { 0 };
        self.stale = if improved || self.depth == 0 {
            0
        } else {
            self.stale + 1
        };
        self.depth += 1;
        if self.confirmations >= 2 {
            Some(DifferentiationTermination::Converged)
        } else if self.stale >= 3 {
            Some(DifferentiationTermination::RoundoffLimit)
        } else {
            None
        }
    }
}

fn extrapolate<T: NumericScalar>(
    lower: T,
    previous: T,
    denominator: T::Real,
) -> Result<T, SciOxideError> {
    let component = |left: T::Real, right: T::Real| {
        let ordinary = checked_sum(&[left, -right]).and_then(|correction| {
            checked_float_quotient(correction, denominator)
                .and_then(|correction| checked_sum(&[left, correction]))
        });
        match ordinary {
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                // left + (left-right)/denominator can fit even when the
                // difference cannot. Accumulate its numerator before division.
                let mut numerator = scioxide_core::RangeAccumulator::new();
                numerator.add_product(left, denominator)?;
                numerator.add(left)?;
                numerator.add(-right)?;
                numerator.finish_quotient(denominator)
            }
            result => result,
        }
    };
    Ok(T::from_components(
        component(lower.real_component(), previous.real_component())?,
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            component(lower.imaginary_component(), previous.imaginary_component())?
        },
    ))
}

#[cfg(test)]
#[allow(clippy::panic)]
mod arithmetic_tests {
    use super::*;

    fn floors<S: RealScalar>() {
        let mut stencil = Stencil {
            offsets: [-S::ONE, S::ZERO, S::ONE, S::ZERO],
            weights: [-S::HALF, S::ZERO, S::HALF, S::ZERO],
            scale: S::ONE,
            derivative_order: 1,
            len: 3,
        };
        for scale in [S::MIN_SUBNORMAL, S::MIN_POSITIVE, S::HALF, S::ONE, S::MAX] {
            stencil.scale = scale;
            for value in [S::ZERO, S::MIN_SUBNORMAL, S::MIN_POSITIVE, S::ONE, S::MAX] {
                for imaginary in [S::ZERO, value / S::TWO] {
                    let real = [value; 3];
                    let imaginary = [imaginary; 3];
                    let actual = rounding_floor(stencil, None, &real, &imaginary);
                    let expected = scaled_rounding_floor(stencil, None, &real, &imaginary);
                    match (actual, expected) {
                        (Ok(actual), Ok(expected)) => assert!(
                            (actual - expected).abs()
                                <= (S::EPSILON * expected).max(S::MIN_SUBNORMAL)
                        ),
                        (Err(actual), Err(expected)) => assert_eq!(actual.kind(), expected.kind()),
                        pair => panic!("ordinary and scaled roundoff disagree: {pair:?}"),
                    }
                }
            }
        }
    }
    #[test]
    fn roundoff_floors_preserve_scaled_f32() {
        floors::<f32>();
    }
    #[test]
    fn roundoff_floors_preserve_scaled_f64() {
        floors::<f64>();
    }
    #[test]
    fn extrapolation_preserves_finite_corrected_value_with_overflowing_difference() {
        for maximum in [f32::MAX as f64, f64::MAX] {
            assert_eq!(extrapolate(maximum / 2.0, -maximum, 3.0).unwrap(), maximum);
        }
        assert_eq!(
            extrapolate(f32::MAX / 2.0, -f32::MAX, 3.0).unwrap(),
            f32::MAX
        );
    }
}

// Ridders repeatedly compares table entries through this checked two-term sum.
// Issue #997's ABBA experiment measures the call/Result boundary's cost; keep
// the same arithmetic and overflow checks in the table loop instead.
#[inline(always)]
fn distance<T: NumericScalar>(left: T, right: T) -> Result<T::Real, SciOxideError> {
    let difference = T::from_components(
        checked_sum(&[left.real_component(), -right.real_component()])?,
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            checked_sum(&[left.imaginary_component(), -right.imaginary_component()])?
        },
    );
    let result = difference.abs();
    if !result.is_finite() {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "derivative difference exceeds scalar range",
        ));
    }
    Ok(result)
}
#[expect(
    clippy::too_many_arguments,
    reason = "problem data, caller buffers and execution resources remain explicit"
)]
fn estimate<T: NumericScalar>(
    a: Stencil<T::Real>,
    b: Option<Stencil<T::Real>>,
    samples: &[T],
    lanes: usize,
    lane: usize,
    base: T,
    real_reduction: &mut scioxide_core::RangeAccumulator<T::Real>,
    imaginary_reduction: &mut scioxide_core::RangeAccumulator<T::Real>,
) -> Result<(T, T::Real), SciOxideError> {
    let mut real = [T::Real::ZERO; 16];
    let mut imaginary = real;
    let count = samples.len() / lanes;
    for i in 0..count {
        let value = samples[i * lanes + lane];
        real[i] = value.real_component();
        imaginary[i] = value.imaginary_component();
    }
    let floor = rounding_floor(a, b, &real[..count], &imaginary[..count])?;
    let derivative =
        |values: &[T::Real], base, reduction: &mut scioxide_core::RangeAccumulator<T::Real>| match b
        {
            Some(b) => mixed_stencil_derivative(a, b, values, base, reduction),
            None => stencil_derivative(a, values, base, reduction),
        };
    let value = T::from_components(
        derivative(&real[..count], base.real_component(), real_reduction)?,
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            derivative(
                &imaginary[..count],
                base.imaginary_component(),
                imaginary_reduction,
            )?
        },
    );
    Ok((value, floor))
}

fn rounding_floor<S: RealScalar>(
    a: Stencil<S>,
    b: Option<Stencil<S>>,
    real: &[S],
    imaginary: &[S],
) -> Result<S, SciOxideError> {
    let weight = |i: usize| {
        b.map_or(a.weights[i.min(a.len - 1)].abs(), |b| {
            (a.weights[i / b.len] * b.weights[i % b.len]).abs()
        })
    };
    // A normal IEEE operation commutes with exact power-of-two normalization.
    // With positive terms there is no cancellation. Require normal operands
    // and intermediates, otherwise restart in extended range below. Keep the
    // scaled path's operation order and per-operation rounding unchanged.
    let ordinary = real.iter().zip(imaginary).enumerate().try_fold(
        S::ZERO,
        |sum, (i, (&real, &imaginary))| {
            let weight = weight(i);
            let larger = real.abs().max(imaginary.abs());
            let smaller = real.abs().min(imaginary.abs());
            if weight == S::ZERO || larger == S::ZERO {
                return Some(sum);
            }
            let normal = |value: S| value.is_normal().then_some(value);
            normal(larger)?;
            if smaller != S::ZERO {
                normal(smaller)?;
            }
            let magnitude = normal(larger + smaller)?;
            let factor = normal(normal(weight * S::EPSILON)? * (S::TWO * S::TWO))?;
            let mut term = normal(magnitude * factor)?;
            for _ in 0..a.derivative_order {
                term = normal(term / normal(a.scale)?)?;
            }
            if let Some(b) = b {
                term = normal(term / normal(b.scale)?)?;
            }
            normal(sum + term)
        },
    );
    if let Some(floor) = ordinary {
        return Ok(floor);
    }
    scaled_rounding_floor(a, b, real, imaginary)
}

fn scaled_rounding_floor<S: RealScalar>(
    a: Stencil<S>,
    b: Option<Stencil<S>>,
    real: &[S],
    imaginary: &[S],
) -> Result<S, SciOxideError> {
    // Estimate the rounding noise of the stencil as sum_i 4 eps |w_i| |f_i|,
    // divided by the step scales just as the derivative is. It is accumulated
    // in scaled form because |f_i| / h^order can exceed the scalar range for
    // intermediate terms even when the final floor does not.
    let mut rounding: Option<PositiveMagnitude<S>> = None;
    for (i, (&real, &imaginary)) in real.iter().zip(imaginary).enumerate() {
        let weight = b.map_or(a.weights[i.min(a.len - 1)].abs(), |b| {
            (a.weights[i / b.len] * b.weights[i % b.len]).abs()
        });
        let magnitude = real.abs().max(imaginary.abs());
        if weight > S::ZERO && magnitude > S::ZERO {
            let mut term = PositiveMagnitude::from_positive(magnitude)?;
            let smaller = real.abs().min(imaginary.abs());
            if smaller > S::ZERO {
                term = term.add(PositiveMagnitude::from_positive(smaller)?)?;
            }
            let factor = PositiveMagnitude::from_positive(weight * S::EPSILON * (S::TWO * S::TWO))?;
            term = term.multiply(factor)?;
            for _ in 0..a.derivative_order {
                term = term.divide(PositiveMagnitude::from_positive(a.scale)?)?;
            }
            if let Some(b) = b {
                term = term.divide(PositiveMagnitude::from_positive(b.scale)?)?;
            }
            rounding = Some(match rounding {
                None => term,
                Some(old) => old.add(term)?,
            });
        }
    }
    match rounding {
        None => Ok(S::ZERO),
        Some(term) => {
            let result = term.0.round()?;
            Ok(result.max(S::MIN_POSITIVE * S::EPSILON))
        }
    }
}

impl<S: RealScalar> AdaptivePlan<'_, S> {
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    fn update_table<T: NumericScalar<Real = S>>(
        &self,
        first: usize,
        second: usize,
        a: Stencil<S>,
        b: Option<Stencil<S>>,
        depth: usize,
        contraction: S,
        available: bool,
        w: &mut AdaptiveBuffers<T>,
    ) -> Result<(bool, bool), SciOxideError> {
        let mixed = b.is_some();
        let count = a.len * b.map_or(1, |b| b.len);
        // Every base stencil here is second order, so column k eliminates the
        // h^(2 + inc * (k - 1)) term, with inc = 2 for even-power (symmetric)
        // expansions and inc = 1 otherwise.
        let power_increment = if symmetric(a) && b.is_none_or(symmetric) {
            2
        } else {
            1
        };
        let mut all_satisfied = depth > 0;
        let mut improved = false;
        for lane in 0..self.outputs {
            let arithmetic = &mut w.arithmetic[0];
            let (raw, floor) = estimate(
                a,
                b,
                &w.samples[..count * self.outputs],
                self.outputs,
                lane,
                w.base[lane],
                &mut arithmetic.real,
                &mut arithmetic.imaginary,
            )?;
            // Each lane stores only the latest tableau row and overwrites it in
            // place. `old_*` carries the previous row's entry for column - 1 (the
            // Neville predecessor) before it is overwritten.
            let mut old_lower = w.table[lane];
            let mut old_floor = w.roundoff[lane];
            w.table[lane] = raw;
            w.roundoff[lane] = floor;
            let columns = match self.options.method {
                AdaptiveDifferenceMethod::Richardson => depth.min(1),
                AdaptiveDifferenceMethod::Ridders => depth,
            };
            let mut row_satisfied = false;
            for column in 1..=columns {
                let index = column * self.outputs + lane;
                let old_upper = w.table[index];
                let old_upper_floor = w.roundoff[index];
                let lower = w.table[index - self.outputs];
                let lower_floor = w.roundoff[index - self.outputs];
                let factor = contraction.powi(2 + power_increment * (column as i32 - 1));
                let denominator = factor - S::ONE;
                // Neville/Ridders step: A[i][k] = A[i][k-1] + (A[i][k-1] - A[i-1][k-1]) /
                // (r^p - 1). Roundoff floors propagate through the same linear combination,
                // plus one rounding of the result.
                let value = extrapolate(lower, old_lower, denominator)?;
                let product = lower_floor * factor;
                let quotient = product / denominator;
                let amplified = if product.is_normal() && quotient.is_normal() {
                    quotient
                } else {
                    checked_product_quotient(lower_floor, factor, denominator)?
                };
                let propagated = checked_sum(&[
                    amplified,
                    checked_float_quotient(old_floor, denominator)?,
                    checked_float_product(value.abs(), S::EPSILON)?,
                ])?;
                // Richardson estimates the remaining error from the change of the
                // extrapolant between levels, scaled by the next power ratio. Ridders uses
                // the larger distance to both parents. Adding the roundoff floor keeps the
                // controller from claiming accuracy below the noise level.
                let truncation =
                    if self.options.method == AdaptiveDifferenceMethod::Richardson && depth >= 2 {
                        checked_float_quotient(
                            distance(value, old_upper)?,
                            contraction.powi(2 + power_increment) - S::ONE,
                        )?
                    } else {
                        distance(value, lower)?.max(distance(value, old_lower)?)
                    };
                let error = checked_sum(&[truncation, propagated])?;
                w.table[index] = value;
                w.roundoff[index] = propagated;
                let output_index = self.index(first, second, lane);
                let mut threshold =
                    self.options
                        .tolerance
                        .threshold(output_index, self.len, value.abs())?;
                // A mixed entry is published at both (i, j) and (j, i), so it must meet
                // the tighter of the two componentwise tolerances.
                if mixed {
                    threshold = threshold.min(self.options.tolerance.threshold(
                        second * self.inputs + first,
                        self.len,
                        value.abs(),
                    )?);
                }
                row_satisfied |= error <= threshold;
                // The first extrapolated row starts a new geometry's history.
                // Retain the global best as a fallback without letting its error
                // prevent the restarted table from making progress.
                if (depth == 1 && column == 1) || error < w.geometry_error[lane] {
                    w.geometry_error[lane] = error;
                    improved = true;
                }
                if !available || error < w.best_error[lane] {
                    w.best[lane] = value;
                    w.best_error[lane] = error;
                }
                old_lower = old_upper;
                old_floor = old_upper_floor;
            }
            all_satisfied &= row_satisfied;
            if depth > 0 {
                let index = self.index(first, second, lane);
                let mut threshold =
                    self.options
                        .tolerance
                        .threshold(index, self.len, w.best[lane].abs())?;
                if mixed {
                    threshold = threshold.min(self.options.tolerance.threshold(
                        second * self.inputs + first,
                        self.len,
                        w.best[lane].abs(),
                    )?);
                }
                all_satisfied &= w.best_error[lane] <= threshold;
            }
        }
        Ok((all_satisfied, improved))
    }
}

/// Workspace size of an [`AdaptiveDifference`] operation for callback values of
/// type `T`, returned by [`AdaptiveDifferenceWorkspace::workspace_requirements`].
#[derive(Debug, Clone)]
pub struct AdaptiveDifferenceRequirements<T: NumericScalar> {
    operation: DifferenceShape,
    inputs: usize,
    outputs: usize,
    levels: usize,
    real: [usize; 7],
    values: [usize; 5],
    axis_nodes: usize,
    mixed_nodes: usize,
    bytes: usize,
    limit: usize,
    scalar: core::marker::PhantomData<T>,
}
impl<T: NumericScalar> AdaptiveDifferenceRequirements<T> {
    /// Returns the heap size in bytes of a new workspace for the request.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Scratch storage for the operations of [`AdaptiveDifference`] with callback
/// values of type `T`.
///
/// Create it with [`workspace`](Self::workspace), or with `Default` followed by
/// [`try_reserve`](Self::try_reserve). One workspace serves every operation
/// whose requirements it meets. Evaluation does not grow it.
#[derive(Debug, Clone)]
pub struct AdaptiveDifferenceWorkspace<T: NumericScalar> {
    data: AdaptiveBuffers<T>,
    state: scioxide_core::SolveState,
    required: Option<AdaptiveDifferenceRequirements<T>>,
}
impl<T: NumericScalar> Default for AdaptiveDifferenceWorkspace<T> {
    fn default() -> Self {
        Self {
            data: AdaptiveBuffers::empty(),
            state: scioxide_core::SolveState::new(),
            required: None,
        }
    }
}
impl<T: NumericScalar> AdaptiveDifferenceWorkspace<T> {
    /// Returns the heap size in bytes of the buffers the workspace holds.
    ///
    /// Includes retained capacity from earlier requests and nested buffers.
    /// Performs no allocation.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.data.retained_bytes()
    }

    /// Returns the workspace size of `request`, without allocating.
    ///
    /// The size depends on the shape, the number of levels and the method's
    /// [`max_workspace_bytes`](AdaptiveDifference::with_max_workspace_bytes).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements(
        request: DerivativeRequest<'_, AdaptiveDifference<T::Real>>,
    ) -> Result<AdaptiveDifferenceRequirements<T>, SciOxideError> {
        let (method, shape) = (request.method(), request.shape());
        let operation = shape.difference(method.zero_direction);
        requirements(
            operation,
            shape.inputs(),
            shape.outputs(),
            crate::validation::shape(operation, shape.inputs(), shape.outputs())?,
            method,
        )
    }
    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace(
        request: DerivativeRequest<'_, AdaptiveDifference<T::Real>>,
    ) -> Result<Self, SciOxideError> {
        let mut workspace = Self::default();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Reports whether the workspace is large enough for `required`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace holds more than the plan's
    ///   `max_workspace_bytes`.
    pub fn check(
        &self,
        required: &AdaptiveDifferenceRequirements<T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        self.data.check(required)
    }
    /// Grows the workspace to meet `required`.
    ///
    /// The byte limit is checked before any buffer grows. If an allocation
    /// fails, buffers grown before it are kept; call [`check`](Self::check)
    /// again before evaluating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace would exceed `max_workspace_bytes`.
    /// - `AllocationFailure` if a buffer cannot grow.
    pub fn try_reserve(
        &mut self,
        required: &AdaptiveDifferenceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        self.required = None;
        self.data.reserve(required)?;
        self.required = Some(required.clone());
        Ok(())
    }

    #[inline]
    fn prepare(&mut self, plan: &AdaptivePlan<'_, T::Real>) -> Result<(), SciOxideError> {
        if let Some(r) = &self.required
            && r.operation == plan.operation
            && r.inputs == plan.inputs
            && r.outputs == plan.outputs
            && r.levels == plan.options.levels
            && r.limit == plan.options.max_workspace_bytes
            && self.data.check(r)? == scioxide_core::Capacity::Sufficient
        {
            // Buffer lengths are private and only preparation changes them.
            // Evaluation resets sample validity independently of configuration.
            return Ok(());
        }
        self.prepare_request(plan)
    }
    // Keep immutable sizing and reconfiguration outside repeated admission.
    fn prepare_request(&mut self, plan: &AdaptivePlan<'_, T::Real>) -> Result<(), SciOxideError> {
        let required = plan.requirements::<T>()?;
        workspace::prepare(self.data.check(&required)?, required.bytes, || {
            self.data.retained_bytes()
        })?;
        self.data.configure(&required)?;
        self.required = Some(required);
        Ok(())
    }
}
impl<T: NumericScalar> AdaptiveBuffers<T> {
    fn empty() -> Self {
        Self {
            operation: DifferenceShape::Gradient,
            inputs: 0,
            outputs: 0,
            levels: 0,
            point: Vec::new(),
            previous_point: Vec::new(),
            sample_points: Vec::new(),
            samples: Vec::new(),
            base: Vec::new(),
            table: Vec::new(),
            roundoff: Vec::new(),
            best: Vec::new(),
            best_error: Vec::new(),
            geometry_error: Vec::new(),
            values: Vec::new(),
            errors: Vec::new(),
            arithmetic: Vec::new(),
        }
    }
    fn real_buffers(&self) -> [&Vec<T::Real>; 7] {
        [
            &self.point,
            &self.previous_point,
            &self.sample_points,
            &self.roundoff,
            &self.best_error,
            &self.geometry_error,
            &self.errors,
        ]
    }
    fn value_buffers(&self) -> [&Vec<T>; 5] {
        [
            &self.samples,
            &self.base,
            &self.table,
            &self.best,
            &self.values,
        ]
    }
    fn real_buffers_mut(&mut self) -> [&mut Vec<T::Real>; 7] {
        [
            &mut self.point,
            &mut self.previous_point,
            &mut self.sample_points,
            &mut self.roundoff,
            &mut self.best_error,
            &mut self.geometry_error,
            &mut self.errors,
        ]
    }
    fn value_buffers_mut(&mut self) -> [&mut Vec<T>; 5] {
        [
            &mut self.samples,
            &mut self.base,
            &mut self.table,
            &mut self.best,
            &mut self.values,
        ]
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<T::Real>(
                    self.real_buffers().map(|v| (v.capacity(), 0)),
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<T>(
                    self.value_buffers().map(|v| (v.capacity(), 0)),
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<AdaptiveArithmetic<T>>(
                    [(self.arithmetic.capacity(), 0)],
                    workspace::BYTES,
                )?,
                workspace::try_sum_bytes(
                    self.arithmetic.iter().map(|a| a.retained.retained_bytes()),
                    workspace::BYTES,
                )?,
            ],
            workspace::BYTES,
        )
    }
    fn check(
        &self,
        required: &AdaptiveDifferenceRequirements<T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let retained = self.retained_bytes()?;
        workspace::check(retained, required.limit, || {
            Ok(self
                .real_buffers()
                .into_iter()
                .zip(required.real)
                .all(|(v, n)| v.capacity() >= n)
                && self
                    .value_buffers()
                    .into_iter()
                    .zip(required.values)
                    .all(|(v, n)| v.capacity() >= n)
                && self.arithmetic.first().is_some_and(|a| {
                    a.retained.has_capacity(
                        required.inputs,
                        required.axis_nodes,
                        required.mixed_nodes,
                    )
                }))
        })
    }
    fn projected_bytes(
        &self,
        required: &AdaptiveDifferenceRequirements<T>,
    ) -> Result<usize, SciOxideError> {
        let cache = match self.arithmetic.first() {
            Some(a) => a.retained.projected_bytes(
                required.inputs,
                required.axis_nodes,
                required.mixed_nodes,
            )?,
            None => HessianSamples::<T>::bytes_for(
                required.inputs,
                required.axis_nodes,
                required.mixed_nodes,
            )?,
        };
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<T::Real>(
                    self.real_buffers()
                        .into_iter()
                        .zip(required.real)
                        .map(|(v, n)| (v.capacity(), n)),
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<T>(
                    self.value_buffers()
                        .into_iter()
                        .zip(required.values)
                        .map(|(v, n)| (v.capacity(), n)),
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<AdaptiveArithmetic<T>>(
                    [(self.arithmetic.capacity(), 1)],
                    workspace::BYTES,
                )?,
                cache,
                workspace::try_sum_bytes(
                    self.arithmetic
                        .iter()
                        .skip(1)
                        .map(|a| a.retained.retained_bytes()),
                    workspace::BYTES,
                )?,
            ],
            workspace::BYTES,
        )
    }
    fn reserve(
        &mut self,
        required: &AdaptiveDifferenceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        let bytes = self.projected_bytes(required)?;
        workspace::reserve(
            self,
            bytes,
            required.limit,
            |data| {
                for (v, n) in data.real_buffers_mut().into_iter().zip(required.real) {
                    workspace::reserve_buffer(
                        v,
                        n,
                        T::Real::ZERO,
                        scioxide_core::quantity!("workspace_buffer"),
                    )?;
                }
                for (v, n) in data.value_buffers_mut().into_iter().zip(required.values) {
                    workspace::reserve_buffer(
                        v,
                        n,
                        T::ZERO,
                        scioxide_core::quantity!("workspace_buffer"),
                    )?;
                }
                let additional = usize::from(data.arithmetic.is_empty());
                scioxide_core::backend::try_reserve(
                    &mut data.arithmetic,
                    additional,
                    scioxide_core::quantity!("adaptive_arithmetic"),
                )?;
                if data.arithmetic.is_empty() {
                    data.arithmetic.push(AdaptiveArithmetic {
                        retained: HessianSamples::default(),
                        real: scioxide_core::RangeAccumulator::new(),
                        imaginary: scioxide_core::RangeAccumulator::new(),
                    });
                }
                data.arithmetic[0].retained.reserve(
                    required.inputs,
                    required.axis_nodes,
                    required.mixed_nodes,
                )?;
                data.configure(required)?;
                Ok(())
            },
            |data| data.retained_bytes(),
        )
    }
    fn configure(
        &mut self,
        required: &AdaptiveDifferenceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        for (v, n) in self.real_buffers_mut().into_iter().zip(required.real) {
            v.resize(n, T::Real::ZERO);
        }
        for (v, n) in self.value_buffers_mut().into_iter().zip(required.values) {
            v.resize(n, T::ZERO);
        }
        self.arithmetic[0].retained.configure(
            required.inputs,
            required.axis_nodes,
            required.mixed_nodes,
        )?;
        self.operation = required.operation;
        self.inputs = required.inputs;
        self.outputs = required.outputs;
        self.levels = required.levels;
        Ok(())
    }
}

impl<S: RealScalar> AdaptivePlan<'_, S> {
    pub(super) fn requirements<T: NumericScalar<Real = S>>(
        &self,
    ) -> Result<AdaptiveDifferenceRequirements<T>, SciOxideError> {
        requirements(
            self.operation,
            self.inputs,
            self.outputs,
            self.len,
            self.options,
        )
    }
}
/// Sizes the buffers of one operation with `len` output entries.
fn requirements<S: RealScalar, T: NumericScalar<Real = S>>(
    operation: DifferenceShape,
    inputs: usize,
    outputs: usize,
    len: usize,
    options: &AdaptiveDifference<S>,
) -> Result<AdaptiveDifferenceRequirements<T>, SciOxideError> {
    let table = checked_mul(
        options.levels,
        outputs,
        scioxide_core::quantity!("workspace_bytes"),
    )?;
    let real = [
        inputs,
        if matches!(operation, DifferenceShape::Jvp(_)) {
            inputs
        } else {
            0
        },
        checked_mul(16, inputs, scioxide_core::quantity!("workspace_bytes"))?,
        table,
        outputs,
        outputs,
        len,
    ];
    let values = [
        checked_mul(16, outputs, scioxide_core::quantity!("workspace_bytes"))?,
        outputs,
        table,
        outputs,
        len,
    ];
    let axis_nodes = if operation == DifferenceShape::Hessian {
        checked_mul(
            8,
            options.levels,
            scioxide_core::quantity!("hessian_samples"),
        )?
    } else {
        0
    };
    let mixed_nodes = if operation == DifferenceShape::Hessian {
        checked_mul(
            16,
            options.levels,
            scioxide_core::quantity!("hessian_samples"),
        )?
    } else {
        0
    };
    let bytes = workspace::sum_bytes(
        [
            workspace::buffer_bytes::<S>(real.map(|n| (0, n)), workspace::BYTES)?,
            workspace::buffer_bytes::<T>(values.map(|n| (0, n)), workspace::BYTES)?,
            workspace::buffer_bytes::<AdaptiveArithmetic<T>>([(0, 1)], workspace::BYTES)?,
            HessianSamples::<T>::bytes_for(inputs, axis_nodes, mixed_nodes)?,
        ],
        workspace::BYTES,
    )?;
    workspace::memory_limit(bytes, options.max_workspace_bytes)?;
    Ok(AdaptiveDifferenceRequirements {
        operation,
        inputs,
        outputs,
        levels: options.levels,
        real,
        values,
        axis_nodes,
        mixed_nodes,
        bytes,
        limit: options.max_workspace_bytes,
        scalar: core::marker::PhantomData,
    })
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use scioxide_core::{Capacity, Complex64, SciOxideErrorKind};

    #[test]
    fn geometry_restart_counts_only_current_extrapolated_stalls() {
        let one_sided = adaptive_axis(0.05_f64, 0.1, 1, Some((Some(0.0), None)))
            .unwrap()
            .unwrap();
        let central = adaptive_axis(0.05_f64, 0.02, 1, Some((Some(0.0), None)))
            .unwrap()
            .unwrap();
        let mut progress = RefinementProgress::new();
        assert!(progress.prepare(one_sided, None, 2.0).is_some());
        assert_eq!(progress.observe(false, false), None);
        assert_eq!(progress.observe(true, true), None);
        assert_eq!(progress.observe(false, false), None);
        assert_eq!(progress.observe(true, false), None);
        assert_eq!(progress.stale, 2);
        assert!(progress.prepare(central, None, 2.0).is_some());
        assert_eq!(progress.depth, 0);
        assert_eq!(progress.stale, 0);
        assert_eq!(progress.confirmations, 0);
        assert!(progress.available);
        assert_eq!(progress.observe(false, false), None);
        assert_eq!(progress.stale, 0);
        assert_eq!(progress.observe(false, true), None);
        assert_eq!(progress.observe(false, false), None);
        assert_eq!(progress.observe(false, false), None);
        assert_eq!(
            progress.observe(false, false),
            Some(DifferentiationTermination::RoundoffLimit)
        );
    }

    #[test]
    fn every_adaptive_buffer_is_required_independently_of_aggregate_spare_bytes() {
        let method = AdaptiveDifference::<f64>::default();
        let plans = [
            method.plan(DifferenceShape::Jacobian, 2, 2).unwrap(),
            method.plan(DifferenceShape::Hessian, 2, 1).unwrap(),
            method
                .plan(
                    DifferenceShape::Jvp(ZeroDirectionPolicy::MathematicalZero),
                    2,
                    2,
                )
                .unwrap(),
        ];
        for plan in plans {
            let required = plan.requirements::<Complex64>().unwrap();
            for i in 0..required.real.len() + required.values.len() {
                let n = if i < required.real.len() {
                    required.real[i]
                } else {
                    required.values[i - required.real.len()]
                };
                if n == 0 {
                    continue;
                }
                let mut work = AdaptiveDifferenceWorkspace::default();
                work.try_reserve(&required).unwrap();
                work.data.point.reserve(1024);
                work.data.values.reserve(1024);
                if i < required.real.len() {
                    *work.data.real_buffers_mut()[i] = Vec::with_capacity(n - 1);
                } else {
                    *work.data.value_buffers_mut()[i - required.real.len()] =
                        Vec::with_capacity(n - 1);
                }
                assert!(work.data.retained_bytes().unwrap() > required.bytes());
                assert_eq!(work.check(&required).unwrap(), Capacity::NeedsGrowth);
                assert_eq!(
                    work.prepare(&plan).unwrap_err().kind(),
                    SciOxideErrorKind::DimensionMismatch
                );
            }
            let mut work = AdaptiveDifferenceWorkspace::default();
            work.try_reserve(&required).unwrap();
            work.data.point.reserve(1024);
            work.data.arithmetic.clear();
            assert_eq!(work.check(&required).unwrap(), Capacity::NeedsGrowth);
        }
    }
}
