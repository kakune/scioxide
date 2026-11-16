//! One-shot derivatives: one call that takes the function and the point and
//! returns the derivative.

use crate::{
    AdaptiveDifference, ColoredJacobianPlan, ComplexStep, DerivativeOperation, FiniteDifference,
    NodeDifferenceOptions, PhysicalDifferencePlan,
};
use alloc::vec::Vec;
use scioxide_core::{
    ColoringSchedule, Complex, ExecutionContext, ExtendedInterval, NumericScalar, RealScalar,
    SciOxideError,
};

/// Computes `f′(x)` of `f: ℝ → ℝ` by a central difference with the default
/// step.
///
/// The difference is `(f(x + h) − f(x − h)) / (2h)` with
/// `h = ε^(1/3)·max(1, |x|)` (`6.1·10⁻⁶` for `f64`, `4.9·10⁻³` for `f32` at
/// `|x| ≤ 1`); it costs three evaluations, including `f(x)`. For a smooth
/// function of moderate size the error is of order `ε^(2/3)`: `3.7·10⁻¹¹` for
/// `f64` and `2.4·10⁻⁵` for `f32`. The call allocates nothing. Use
/// [`AdaptiveDifference`] for an error estimate, and [`FiniteDifference`] for
/// bounds, another scheme or step, or a known `f(x)`.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if `x` is NaN or infinite.
/// - `DifferentiationFailure` whose `source()` is the error returned by
///   `function`, `NonFinite` (origin `Callback`) for a NaN or infinite value of
///   `function`, or `NumericalInstability` for a sample point or stencil sum
///   outside the finite range; it records the sample location and the
///   evaluation count.
///
/// # Examples
///
/// ```
/// // f(x) = sin x, so f′(1) = cos 1.
/// let slope = scioxide_differentiate::derivative(|x: f64| Ok(x.sin()), 1.0)?;
/// assert!((slope - 1.0_f64.cos()).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §8.1.
pub fn derivative<S, F>(function: F, x: S) -> Result<S, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    let mut value = S::ZERO;
    FiniteDifference::default()
        .derivative_into(
            function,
            x,
            ExtendedInterval::entire(),
            None,
            &mut value,
            ExecutionContext::sequential_native(),
        )?
        .require_success()?;
    Ok(value)
}

/// Computes `f″(x)` of `f: ℝ → ℝ` by a central second difference with the
/// default step.
///
/// The difference is `(f(x + h) − 2f(x) + f(x − h)) / h²` with
/// `h = ε^(1/4)·max(1, |x|)` (`1.2·10⁻⁴` for `f64`, `1.9·10⁻²` for `f32` at
/// `|x| ≤ 1`); it costs three evaluations. For a smooth function of moderate
/// size the error is of order `√ε`: `1.5·10⁻⁸` for `f64` and `3.5·10⁻⁴` for
/// `f32`. Use [`AdaptiveDifference`] for an error estimate, and
/// [`FiniteDifference`] for bounds, another scheme or step, or a known `f(x)`.
///
/// # Errors
///
/// - The errors of [`derivative`].
/// - `AllocationFailure` if the workspace or the result cannot be allocated.
///
/// # Examples
///
/// ```
/// // f(x) = x³, so f″(2) = 12.
/// let curvature = scioxide_differentiate::second_derivative(|x: f64| Ok(x * x * x), 2.0)?;
/// assert!((curvature - 12.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §8.1.
pub fn second_derivative<S, F>(function: F, x: S) -> Result<S, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    Ok(FiniteDifference::default()
        .second_derivative(function, x)?
        .into_values()?[0])
}

/// Computes the gradient `∇f(x)` of `f: ℝⁿ → ℝ` by central differences with
/// the default step.
///
/// Coordinate `i` uses `hᵢ = ε^(1/3)·max(1, |xᵢ|)` (`6.1·10⁻⁶` for `f64`,
/// `4.9·10⁻³` for `f32` at `|xᵢ| ≤ 1`). It costs `2n + 1` evaluations and needs
/// no bounds. For a smooth function of moderate size the error of each entry
/// is of order `ε^(2/3)`: `3.7·10⁻¹¹` for `f64` and `2.4·10⁻⁵` for `f32`. An
/// empty `x` gives an empty gradient without a call of `function`. Use
/// [`AdaptiveDifference`] for error estimates, and [`FiniteDifference`] for
/// bounds, another scheme or step, or a known `f(x)`.
///
/// # Errors
///
/// - The errors of [`derivative`], for any coordinate of `x`.
/// - `AllocationFailure` if the workspace or the result cannot be allocated.
///
/// # Examples
///
/// ```
/// // f(x) = x₀² + x₁², so ∇f(1, 2) = (2, 4).
/// let g = scioxide_differentiate::gradient(|p: &[f64]| Ok(p[0] * p[0] + p[1] * p[1]), &[1.0, 2.0])?;
/// assert!((g[0] - 2.0).abs() < 1e-9 && (g[1] - 4.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §8.1.
pub fn gradient<S, F>(function: F, x: &[S]) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<S, SciOxideError>,
{
    FiniteDifference::default()
        .gradient(
            |p: &[S], y: &mut [S]| -> Result<(), SciOxideError> {
                y[0] = function(p)?;
                Ok(())
            },
            x,
            None,
            None,
        )?
        .into_values()
}

/// Computes the Hessian of `f: ℝⁿ → ℝ`, row-major `n × n`, by central
/// differences with the default step.
///
/// Coordinate `i` uses `hᵢ = ε^(1/4)·max(1, |xᵢ|)` (`1.2·10⁻⁴` for `f64`,
/// `1.9·10⁻²` for `f32` at `|xᵢ| ≤ 1`). Diagonal entries use the central second
/// difference and mixed entries a product of central first differences;
/// only the upper triangle is evaluated, so the result is exactly symmetric.
/// It costs `1 + 2n²` evaluations and needs no bounds. For a smooth
/// function of moderate size the error of each entry is of order `√ε`:
/// `1.5·10⁻⁸` for `f64` and `3.5·10⁻⁴` for `f32`. Use [`AdaptiveDifference`]
/// for error estimates, and [`FiniteDifference`] for bounds, another scheme or
/// step, or a known `f(x)`.
///
/// # Errors
///
/// - The errors of [`gradient`].
/// - `SizeOverflow` (`derivative_entries`) if `n²` does not fit in `usize`.
///
/// # Examples
///
/// ```
/// // f(x) = x₀² x₁ has the Hessian [[2x₁, 2x₀], [2x₀, 0]] = [[4, 2], [2, 0]] at (1, 2).
/// let h = scioxide_differentiate::hessian(|p: &[f64]| Ok(p[0] * p[0] * p[1]), &[1.0, 2.0])?;
/// for (value, expected) in h.iter().zip([4.0, 2.0, 2.0, 0.0]) {
///     assert!((value - expected).abs() < 1e-6);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §8.1.
pub fn hessian<S, F>(function: F, x: &[S]) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<S, SciOxideError>,
{
    FiniteDifference::default()
        .hessian(
            |p: &[S], y: &mut [S]| -> Result<(), SciOxideError> {
                y[0] = function(p)?;
                Ok(())
            },
            x,
            None,
            None,
        )?
        .into_values()
}

/// Computes the Jacobian of `f: ℝⁿ → ℝᵐ` with `m = outputs`, row-major `m × n`,
/// by central differences with the default step.
///
/// Entry `i·n + j` is `∂fᵢ/∂xⱼ`. The callback writes the `m` values of `f(x)`
/// into its output slice. Coordinate `j` uses `hⱼ = ε^(1/3)·max(1, |xⱼ|)`
/// (`6.1·10⁻⁶` for `f64`, `4.9·10⁻³` for `f32` at `|xⱼ| ≤ 1`). It costs
/// `2n + 1` evaluations and needs no bounds. For a smooth function of moderate
/// size the error of each entry is of order `ε^(2/3)`: `3.7·10⁻¹¹` for `f64`
/// and `2.4·10⁻⁵` for `f32`. Use [`AdaptiveDifference`] for error estimates,
/// and [`FiniteDifference`] for bounds, another scheme or step, or a known
/// `f(x)`.
///
/// # Errors
///
/// - The errors of [`gradient`].
/// - `SizeOverflow` (`derivative_entries`) if `m·n` does not fit in `usize`.
///
/// # Examples
///
/// ```
/// // f(x) = (x₀ x₁, x₀ + x₁) has the Jacobian [[x₁, x₀], [1, 1]] = [[2, 1], [1, 1]].
/// let f = |p: &[f64], y: &mut [f64]| {
///     y[0] = p[0] * p[1];
///     y[1] = p[0] + p[1];
///     Ok(())
/// };
/// let j = scioxide_differentiate::jacobian(f, &[1.0, 2.0], 2)?;
/// for (value, expected) in j.iter().zip([2.0, 1.0, 1.0, 1.0]) {
///     assert!((value - expected).abs() < 1e-9);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §8.1.
pub fn jacobian<S, F>(function: F, x: &[S], outputs: usize) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    FiniteDifference::default()
        .jacobian(function, x, outputs, None, None)?
        .into_values()
}

/// Computes the Jacobian-vector product `J(x) v` of `f: ℝⁿ → ℝᵐ` with
/// `v = direction` and `m = outputs` by a central difference along `v`.
///
/// The step along `v` is `α = ε^(1/3) / maxᵢ(|vᵢ| / max(1, |xᵢ|))`, so that no
/// coordinate moves farther than its own default step. It costs three
/// evaluations for any `n`, and `J v = 0` without a call for `v = 0`. For a
/// smooth function of moderate size the error is of order `ε^(2/3)` relative
/// to the scale of `v`: `3.7·10⁻¹¹` for `f64` and `2.4·10⁻⁵` for `f32`. Use
/// [`AdaptiveDifference`] for error estimates, and [`FiniteDifference`] for
/// bounds, another scheme or step, or a known `f(x)`.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if `x` or `direction` contains NaN or
///   infinity.
/// - `DimensionMismatch` if `direction.len() != x.len()`.
/// - `DifferentiationFailure` whose `source()` is the error returned by
///   `function`, `NonFinite` (origin `Callback`) for a NaN or infinite value of
///   `function`, `NumericalInstability` for a sample point or stencil sum
///   outside the finite range, or `ConvergenceFailure` (`StepSizeLimit`) when
///   no step along `direction` gives distinct points.
/// - `AllocationFailure` if the workspace or the result cannot be allocated.
///
/// # Examples
///
/// ```
/// // f(x) = (x₀ + x₁, 2x₀ − x₁) is linear, so J v = (1, 2) for v = (1, 0).
/// let f = |p: &[f64], y: &mut [f64]| {
///     y[0] = p[0] + p[1];
///     y[1] = 2.0 * p[0] - p[1];
///     Ok(())
/// };
/// let jv = scioxide_differentiate::jvp(f, &[1.0, 2.0], &[1.0, 0.0], 2)?;
/// assert!((jv[0] - 1.0).abs() < 1e-8 && (jv[1] - 2.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn jvp<S, F>(
    function: F,
    x: &[S],
    direction: &[S],
    outputs: usize,
) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    FiniteDifference::default()
        .jvp(function, x, direction, outputs, None, None)?
        .into_values()
}

/// Computes the stored entries of a sparse Jacobian from a column coloring, by
/// adaptive differences.
///
/// `schedule` holds the pattern of `∂fᵢ/∂xⱼ` and a
/// [`ColoringKind::DerivativeColumns`](scioxide_core::ColoringKind::DerivativeColumns)
/// coloring of its columns, such as `scioxide_sparse::derivative_coloring`
/// computes. The result has one value per stored entry, row by row. The
/// callback writes all `m` rows of `f(x)`, values real or complex. Each column
/// is refined by the extrapolation of [`AdaptiveDifference`] with its defaults,
/// until the estimated error of every entry is at most `√ε·(1 + |Jᵢⱼ|)`
/// on two consecutive levels and its best estimate also meets the tolerance.
/// At least three levels are needed (`√ε` is `1.5·10⁻⁸` for `f64`,
/// `3.5·10⁻⁴` for `f32`); the first step
/// `0.5·max(1, |xⱼ|)` must stay in the domain of `f`. It costs at most
/// `1 + 32c` evaluations for `c` colors. Use [`ColoredJacobianPlan`] with an
/// [`AdaptiveDifference`] for bounds, other settings or the error estimates,
/// and [`FiniteDifference::jacobian`] for a dense fixed-step Jacobian.
///
/// # Errors
///
/// - `InvalidInput` (`schedule`, `UnsupportedConfiguration`) if the coloring is
///   not a `DerivativeColumns` coloring.
/// - `NonFinite` (origin `Input`) if `x` contains NaN or infinity.
/// - `DimensionMismatch` if `x.len()` is not the number of columns of the
///   pattern.
/// - `ConvergenceFailure` if an entry ends without meeting the tolerance, with
///   the reason of [`DifferentiationReport::require_success`](crate::DifferentiationReport::require_success).
/// - `DifferentiationFailure` as in [`derivative`].
/// - `AllocationFailure` if the plan, the workspace or the result cannot be
///   allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ColoringKind, ColoringSchedule, StructuralPattern};
///
/// // f(x) = (x₀², x₁²) has a diagonal Jacobian, so both columns share color 0.
/// let pattern = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1])?;
/// let schedule = ColoringSchedule::new(pattern, ColoringKind::DerivativeColumns, vec![0, 0])?;
/// let f = |x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0];
///     y[1] = x[1] * x[1];
///     Ok(())
/// };
/// let values = scioxide_differentiate::sparse_jacobian(f, &[1.0, 2.0], &schedule)?;
/// assert!((values[0] - 2.0).abs() < 1e-8 && (values[1] - 4.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. R. Curtis, M. J. D. Powell and J. K. Reid, "On the estimation of sparse
/// Jacobian matrices", *J. Inst. Math. Appl.* 13, 1974, 117–119.
pub fn sparse_jacobian<S, T, F>(
    function: F,
    x: &[S],
    schedule: &ColoringSchedule,
) -> Result<Vec<T>, SciOxideError>
where
    S: RealScalar,
    T: NumericScalar<Real = S>,
    F: Fn(&[S], &mut [T]) -> Result<(), SciOxideError>,
{
    ColoredJacobianPlan::new(schedule, AdaptiveDifference::default())?
        .evaluate(function, x, None)?
        .into_values()
}

/// Computes `f′(x)` from the values of `f` at the given nodes.
///
/// The derivative is `Σₖ wₖ f(nodesₖ)`, with Fornberg weights `wₖ` that make
/// the formula exact for polynomials of degree below the number of nodes `k`.
/// The nodes need not be equally spaced or contain `x`; for nodes of spacing
/// `h` around `x`, the truncation error is of order `h^(k−1)`. It costs one
/// evaluation per node, and `f` may return real or complex values. Use
/// [`AdaptiveDifference`] for an error estimate,
/// [`RelativeDifference`](crate::RelativeDifference) for nodes that follow a
/// changing point, and [`FiniteDifference`] for a fixed-step difference with
/// bounds.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if `x` or a node is NaN or infinite.
/// - `InvalidInput` (`nodes`, `OutOfRange`) if there are fewer than two nodes.
/// - `InvalidInput` (`nodes`, `Inconsistent`) if two nodes are equal.
/// - `DifferentiationFailure` as in [`derivative`].
/// - `AllocationFailure` if the plan, the workspace or the result cannot be
///   allocated.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::physical_difference_derivative;
///
/// // Three nodes are exact for quadratics: d/dx x² = 1 at x = 0.5.
/// let slope = physical_difference_derivative(|x: f64| Ok(x * x), 0.5, &[0.0, 0.5, 1.0])?;
/// assert!((slope - 1.0_f64).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// B. Fornberg, "Generation of finite difference formulas on arbitrarily
/// spaced grids", *Math. Comp.* 51(184), 1988, 699–706.
pub fn physical_difference_derivative<S, T, F>(
    function: F,
    x: S,
    nodes: &[S],
) -> Result<T, SciOxideError>
where
    S: RealScalar,
    T: NumericScalar<Real = S>,
    F: Fn(S) -> Result<T, SciOxideError>,
{
    Ok(PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[x],
        &[nodes],
        1,
        NodeDifferenceOptions::default(),
    )?
    .evaluate(
        |p: &[S], y: &mut [T]| -> Result<(), SciOxideError> {
            y[0] = function(p[0])?;
            Ok(())
        },
        None,
    )?
    .into_values()?[0])
}

/// Computes `f′(x) = Im f(x + ih)/h` of a real function by the complex step
/// with `h = ε`.
///
/// `function` is the analytic extension of `f`: holomorphic near `x` and real
/// on the real axis (see [`ComplexStep`]). The formula has no subtractive
/// cancellation, so with `h = ε` the truncation error is below one rounding
/// error of `f` and the result is accurate to the rounding error of
/// evaluating `f`, about `ε·|f′(x)|` (`2.2·10⁻¹⁶` for `f64`, `1.2·10⁻⁷` for
/// `f32`). It costs two evaluations: `f(x)`, which must be real, and
/// `f(x + ih)`. Use [`ComplexStep`] for another step, gradients, Jacobians and
/// workspaces. For a function without an analytic extension, use
/// [`AdaptiveDifference`] for an error estimate and [`FiniteDifference`] for
/// bounds, a chosen scheme or step, or a known `f(x)`.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if `x` is NaN or infinite.
/// - `InvalidInput` (`function`, `UnsupportedConfiguration`) if `f(x)` has a
///   nonzero imaginary part.
/// - `NumericalInstability` (`Underflow`) if a nonzero imaginary response is
///   subnormal or its quotient `Im f(x + ih)/h` rounds to zero.
/// - `DifferentiationFailure` whose `source()` is the error returned by
///   `function` or `NonFinite` (origin `Callback`) for a NaN or infinite value
///   of `function`.
/// - `AllocationFailure` if the workspace or the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex64;
///
/// // d/dx x² = 4 at x = 2, to rounding error.
/// let slope = scioxide_differentiate::complex_step_derivative(|z: Complex64| Ok(z * z), 2.0)?;
/// assert_eq!(slope, 4.0);
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
pub fn complex_step_derivative<S, F>(function: F, x: S) -> Result<S, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    F: Fn(Complex<S>) -> Result<Complex<S>, SciOxideError>,
{
    Ok(ComplexStep::default()
        .derivative(function, x)?
        .into_values()?[0])
}
