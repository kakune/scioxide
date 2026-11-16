//! Default solvers for plain problem data.

use scioxide_core::backend::{try_copy_slice, try_with_capacity};
use scioxide_core::{Argument, Bounds, Interval, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, MatrixView};

use crate::result::termination_error;
use crate::{
    Bfgs, BoundedLinearLeastSquares, BoundedObjective, ConstrainedProblem, Constraints,
    DifferentialEvolution, DifferentialEvolutionOptions, FiniteBounds, LbfgsB, LeastSquares,
    LeastSquaresFn, LeastSquaresProblem, LeastSquaresResult, LinearLeastSquaresProblem,
    LinearLeastSquaresResult, Nnls, ObjectiveFn, OptimizeResult, OptimizeTermination, QapDirection,
    QapProblemView, QapResult, ScalarMinimizer, ScalarMinimumResult, ScalarMinimumStart, Sqp,
};

fn nonempty(argument: Argument, length: usize) -> Result<(), SciOxideError> {
    if length == 0 {
        Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Empty,
            "at least one entry is required",
        ))
    } else {
        Ok(())
    }
}

fn optimum<S: RealScalar>(result: OptimizeResult<S>) -> Result<OptimizeResult<S>, SciOxideError> {
    termination_error(
        scioxide_core::argument!("f"),
        result.termination(),
        result.counts(),
        result.algorithm(),
    )
    .map_or(Ok(result), Err)
}

/// Minimizes a smooth real objective from `x0` using BFGS.
///
/// Uses central differences, strong Wolfe line search, 1000 iterations and
/// gradient tolerances `atol = rtol = √ε`, where `ε` is scalar precision.
/// Central differences cost `2n` values per gradient and have error of order
/// `ε^(2/3)` for smooth functions; BFGS uses `O(n²)` storage for `n` variables.
/// The stopping test certifies local stationarity, not a global minimum.
///
/// # Errors
///
/// Returns `InvalidInput` (`Empty`, `x0`) for an empty start, `NonFinite` for
/// non-finite input, the errors of [`Bfgs::minimize`], or the errors described
/// by [`OptimizeResult::require_success`] for an unsuccessful termination.
/// Callback errors propagate unchanged.
///
/// # Examples
///
/// ```
/// let result = scioxide_optimize::minimize(
///     |x: &[f64]| Ok((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2)),
///     &[0.0, 0.0],
/// )?;
/// assert!((result.x()[0] - 1.0).abs() < 1e-6);
/// assert!((result.x()[1] + 2.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Nocedal and Wright, *Numerical Optimization*, 2nd ed., §§6.1, 8.1.
pub fn minimize<S, F>(f: F, x0: &[S]) -> Result<OptimizeResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[S]) -> Result<S, SciOxideError>,
{
    nonempty(scioxide_core::argument!("x0"), x0.len())?;
    crate::validation::finite_input(x0)?;
    optimum(Bfgs::<S>::default().minimize(&ObjectiveFn::new(x0.len(), f), x0)?)
}

/// Minimizes a smooth objective inside `bounds` using L-BFGS-B.
///
/// Uses forward differences and [`crate::LbfgsBOptions::default`]: 1000
/// iterations and gradient tolerances `atol = rtol = √ε`. Storage is linear
/// in the number of variables for the fixed correction history. The projected
/// gradient test measures local stationarity; forward errors are `O(√ε)`.
///
/// # Errors
///
/// Returns `InvalidInput` (`Empty`, `x0`) for an empty start,
/// `DimensionMismatch` for different start and bounds lengths, `InvalidInput`
/// (`OutOfRange`, `x0`) for a start outside the box, and input, callback and
/// numerical errors from [`LbfgsB::minimize`]. Unsuccessful stops follow
/// [`OptimizeResult::require_success`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Bounds;
/// let bounds = Bounds::new(vec![(Some(0.0), Some(1.0)), (None, Some(0.0))])?;
/// let result = scioxide_optimize::minimize_bounded(
///     |x: &[f64]| Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2)),
///     &[0.25, 0.0], &bounds,
/// )?;
/// assert!((result.x()[0] - 1.0).abs() < 1e-6);
/// assert!((result.x()[1] + 1.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. P. Bertsekas, "Projected Newton methods for optimization problems
///   with simple constraints", *SIAM J. Control Optim.* 20(2), 221–246, 1982.
/// - D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large
///   scale optimization", *Math. Program.* 45, 503–528, 1989.
pub fn minimize_bounded<S, F>(
    f: F,
    x0: &[S],
    bounds: &Bounds<S>,
) -> Result<OptimizeResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[S]) -> Result<S, SciOxideError>,
{
    nonempty(scioxide_core::argument!("x0"), x0.len())?;
    let objective = ObjectiveFn::new(x0.len(), f);
    let problem = BoundedObjective::new(&objective, bounds)?;
    optimum(LbfgsB::<S>::default().minimize(&problem, x0)?)
}

/// Minimizes a smooth objective with nonlinear constraints using SQP.
///
/// Uses [`crate::SqpOptions::default`], forward differences, 1000 iterations
/// and `√ε` role tolerances. Each dense quadratic subproblem uses quadratic
/// storage in the variables and constraints. Success requires the scaled KKT
/// tests; forward derivative errors are `O(√ε)` for smooth functions.
///
/// # Errors
///
/// Returns `InvalidInput` (`Empty`, `x0`) for an empty start,
/// `DimensionMismatch` if the constraint dimension differs, and the input,
/// callback and numerical errors of [`Sqp::minimize`]. Unsuccessful stops
/// follow [`OptimizeResult::require_success`].
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{ConstraintFn, Constraints, minimize_constrained};
/// let constraints = Constraints::new(2)?
///     .equality(ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)))?;
/// let result = minimize_constrained(
///     |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]), &[0.0, 0.0], &constraints,
/// )?;
/// assert!(result.x().iter().all(|x| (x - 0.5).abs() < 1e-6));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Nocedal and Wright, *Numerical Optimization*, 2nd ed., chapter 18.
pub fn minimize_constrained<S, F>(
    f: F,
    x0: &[S],
    constraints: &Constraints<S>,
) -> Result<OptimizeResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[S]) -> Result<S, SciOxideError>,
{
    nonempty(scioxide_core::argument!("x0"), x0.len())?;
    let objective = ObjectiveFn::new(x0.len(), f);
    let problem = ConstrainedProblem::new(&objective, constraints)?;
    optimum(Sqp::<S>::default().minimize(&problem, x0)?)
}

/// Searches a finite box for a global minimum using differential evolution.
///
/// Uses [`DifferentialEvolutionOptions::default`] with `seed`, a midpoint start,
/// at most 1000 generations, `max(15n, 4)` members, mutation factor 0.8 and
/// crossover rate 0.9. Returns the best point when the population
/// stagnates or the schedule completes. No status certifies a global optimum
/// or an accuracy bound. Storage is `O(np)` for `n` variables and `p` members.
///
/// # Errors
///
/// Returns `InvalidInput` (`Empty`, `bounds`) for an empty box, `NonFinite` for
/// non-finite endpoints, `InvalidInput` (`OutOfRange`, `bounds`) if `lower ≥ upper`,
/// and the allocation, callback and numerical errors of [`DifferentialEvolution::minimize`].
/// Other stops follow [`OptimizeResult::require_success`].
///
/// # Examples
///
/// ```
/// let result = scioxide_optimize::minimize_global(
///     |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]), &[(-5.0, 5.0); 2], 42,
/// )?;
/// assert!(result.fun() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Storn and Price, *Journal of Global Optimization* 11, 341–359, 1997.
pub fn minimize_global<S, F>(
    f: F,
    bounds: &[(S, S)],
    seed: u64,
) -> Result<OptimizeResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(&[S]) -> Result<S, SciOxideError>,
{
    let finite = FiniteBounds::new(try_copy_slice(bounds, scioxide_core::quantity!("bounds"))?)?;
    let mut entries = try_with_capacity(finite.len(), scioxide_core::quantity!("bounds"))?;
    let mut x0 = try_with_capacity(finite.len(), scioxide_core::quantity!("x0"))?;
    for &(lower, upper) in finite.as_slice() {
        entries.push((Some(lower), Some(upper)));
        x0.push(scioxide_core::checked_midpoint(lower, upper)?);
    }
    let bounds = Bounds::new(entries)?;
    let objective = ObjectiveFn::new(x0.len(), f);
    let problem = BoundedObjective::new(&objective, &bounds)?;
    let result =
        DifferentialEvolution::new(DifferentialEvolutionOptions::default().with_seed(seed))?
            .minimize(&problem, &x0)?;
    match result.termination() {
        OptimizeTermination::PopulationStagnation | OptimizeTermination::SearchBudgetCompleted => {
            Ok(result)
        }
        _ => optimum(result),
    }
}

/// Minimizes a scalar function on `[lower, upper]` using bounded Brent search.
///
/// Uses 1000 iterations, 2000 evaluations and step and function tolerances
/// `atol = rtol = √ε`, where `ε` is scalar precision.
/// Each iteration evaluates at most one new point and uses constant storage.
/// The interval-width stopping test gives accuracy relative to those tolerances
/// for a continuous unimodal objective.
///
/// # Errors
///
/// Returns `NonFinite` for non-finite endpoints, `InvalidInput` (`Inconsistent`,
/// `lower`) for reversed endpoints, and the errors of [`ScalarMinimizer::minimize`].
/// Unsuccessful stops follow [`ScalarMinimumResult::require_success`].
///
/// # Examples
///
/// ```
/// let result = scioxide_optimize::minimize_scalar(|x: f64| Ok((x - 0.25).powi(2)), -1.0, 2.0)?;
/// assert!((result.x() - 0.25).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Brent, *Algorithms for Minimization without Derivatives*, chapter 5, 1973.
pub fn minimize_scalar<S: RealScalar, F>(
    f: F,
    lower: S,
    upper: S,
) -> Result<ScalarMinimumResult<S>, SciOxideError>
where
    F: Fn(S) -> Result<S, SciOxideError>,
{
    let result = ScalarMinimizer::<S>::default().minimize(
        &f,
        ScalarMinimumStart::Bounded(Interval::new(lower, upper)?),
    )?;
    termination_error(
        scioxide_core::argument!("f"),
        result.termination(),
        result.counts(),
        Some(result.algorithm()),
    )
    .map_or(Ok(result), Err)
}

/// Minimizes `½ ∑ rᵢ(x)²` using Levenberg–Marquardt least squares.
///
/// The residual closure writes all `residuals` entries into solver storage.
/// Uses quadratic loss, forward differences and 1000 iterations. Dense storage
/// is `O(mn + n²)` for `m` residuals and `n` parameters; forward derivative
/// errors are `O(√ε)`. See [`crate::LeastSquaresOptions`] for the stopping tests.
///
/// # Errors
///
/// Returns `InvalidInput` (`Empty`, `x0` or `residuals`) for zero dimensions,
/// `NonFinite` for non-finite input and the errors of [`LeastSquares::solve`].
/// Unsuccessful stops follow [`LeastSquaresResult::require_success`].
///
/// # Examples
///
/// ```
/// let result = scioxide_optimize::least_squares(|x: &[f64], r: &mut [f64]| {
///     r[0] = x[0] - 1.0;
///     r[1] = x[1] + 2.0;
///     Ok(())
/// }, &[0.0, 0.0], 2)?;
/// assert!((result.x()[0] - 1.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Moré, "The Levenberg–Marquardt algorithm: implementation and theory", 1978.
pub fn least_squares<S, R>(
    residual: R,
    x0: &[S],
    residuals: usize,
) -> Result<LeastSquaresResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: Fn(&[S], &mut [S]) -> Result<(), SciOxideError>,
{
    nonempty(scioxide_core::argument!("x0"), x0.len())?;
    nonempty(scioxide_core::argument!("residuals"), residuals)?;
    crate::validation::finite_input(x0)?;
    let problem = LeastSquaresFn::new(x0.len(), residuals, residual)?;
    let result = LeastSquares::<S>::default().solve(&problem, x0)?;
    termination_error(
        scioxide_core::argument!("residual"),
        result.termination(),
        result.counts(),
        result.algorithm(),
    )
    .map_or(Ok(result), Err)
}

struct CurveFit<'a, S, M> {
    model: M,
    xdata: &'a [S],
    ydata: &'a [S],
    parameters: usize,
}
impl<S, M> LeastSquaresProblem<S> for CurveFit<'_, S, M>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: Fn(S, &[S]) -> Result<S, SciOxideError>,
{
    fn param_dim(&self) -> usize {
        self.parameters
    }
    fn residual_dim(&self) -> usize {
        self.xdata.len()
    }
    fn residual_into(&self, p: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        for ((value, &x), &y) in output.iter_mut().zip(self.xdata).zip(self.ydata) {
            *value = (self.model)(x, p)? - y;
        }
        Ok(())
    }
}

/// Fits `model(x, p)` to paired observations by unweighted least squares.
///
/// Minimizes `½ ∑ (model(xᵢ, p) − yᵢ)²`, starting at `p0`, with the same
/// Levenberg–Marquardt defaults, cost and accuracy as [`least_squares`].
/// No parameter covariance is computed.
///
/// # Errors
///
/// Returns `DimensionMismatch` for different data lengths, `InvalidInput`
/// (`Empty`, `xdata` or `p0`) for empty data or parameters, `NonFinite` for
/// non-finite data or parameters, and the errors of [`LeastSquares::solve`].
/// Model errors propagate unchanged; unsuccessful stops follow
/// [`LeastSquaresResult::require_success`].
///
/// # Examples
///
/// ```
/// let result = scioxide_optimize::curve_fit(
///     |t: f64, p: &[f64]| Ok(p[0] * t + p[1]),
///     &[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0], &[0.0, 0.0],
/// )?;
/// assert!((result.x()[0] - 2.0).abs() < 1e-6);
/// assert!((result.x()[1] - 1.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Moré, "The Levenberg–Marquardt algorithm: implementation and theory", 1978.
pub fn curve_fit<S, M>(
    model: M,
    xdata: &[S],
    ydata: &[S],
    p0: &[S],
) -> Result<LeastSquaresResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: Fn(S, &[S]) -> Result<S, SciOxideError>,
{
    crate::validation::shape(xdata.len(), ydata.len())?;
    nonempty(scioxide_core::argument!("xdata"), xdata.len())?;
    nonempty(scioxide_core::argument!("p0"), p0.len())?;
    crate::validation::finite_input(xdata)?;
    crate::validation::finite_input(ydata)?;
    let problem = CurveFit {
        model,
        xdata,
        ydata,
        parameters: p0.len(),
    };
    let result = LeastSquares::<S>::default().solve(&problem, p0)?;
    termination_error(
        scioxide_core::argument!("model"),
        result.termination(),
        result.counts(),
        result.algorithm(),
    )
    .map_or(Ok(result), Err)
}

/// Solves `min ½ ‖Ax − b‖²` with `x ≥ 0` using active-set NNLS.
///
/// Uses [`crate::LinearLeastSquaresOptions::default`], dense QR, 1000 outer
/// iterations and `√ε` tolerances. Storage is `O(mn + n²)` for an `m × n`
/// matrix; conditioning determines the attainable forward accuracy.
///
/// # Errors
///
/// Returns `DimensionMismatch` if `b.len() != a.rows()`, `NonFinite` for
/// non-finite data, and the errors of [`Nnls::solve`]. Unsuccessful stops follow
/// [`LinearLeastSquaresResult::require_success`].
///
/// # Examples
///
/// ```
/// let a = scioxide_linalg::Matrix::from_rows(&[[1.0_f64, 0.0], [0.0, 1.0]])?;
/// let result = scioxide_optimize::nnls(&a, &[2.0, -1.0])?;
/// assert!((result.x()[0] - 2.0).abs() < 1e-6);
/// assert!(result.x()[1].abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Lawson and Hanson, *Solving Least Squares Problems*, chapter 23, 1974.
pub fn nnls<S: RealScalar + NumericScalar<Real = S>>(
    a: &Matrix<S>,
    b: &[S],
) -> Result<LinearLeastSquaresResult<S>, SciOxideError> {
    let result = Nnls::<S>::default().solve(&LinearLeastSquaresProblem::from_dense(a, b)?)?;
    termination_error(
        scioxide_core::argument!("a"),
        result.termination(),
        result.counts(),
        result.algorithm(),
    )
    .map_or(Ok(result), Err)
}

/// Solves `min ½ ‖Ax − b‖²` within `bounds` using bounded-variable least squares.
///
/// Uses the same dense QR defaults, cost and conditioning limits as [`nnls`].
/// One-sided and two-sided bounds are accepted.
///
/// # Errors
///
/// Returns `DimensionMismatch` if `b.len() != a.rows()` or the bounds length
/// differs from `a.cols()`, `NonFinite` for non-finite data, and the errors of
/// [`BoundedLinearLeastSquares::solve`]. Unsuccessful stops follow
/// [`LinearLeastSquaresResult::require_success`].
///
/// # Examples
///
/// ```
/// let a = scioxide_linalg::Matrix::from_rows(&[[1.0_f64, 0.0], [0.0, 1.0]])?;
/// let bounds = scioxide_core::Bounds::new(vec![(Some(0.0), Some(1.0)); 2])?;
/// let result = scioxide_optimize::bvls(&a, &[2.0, 2.0], &bounds)?;
/// assert!(result.x().iter().all(|x| (x - 1.0).abs() < 1e-6));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Stark and Parker, *Computational Statistics* 10, 129–141, 1995.
pub fn bvls<S: RealScalar + NumericScalar<Real = S>>(
    a: &Matrix<S>,
    b: &[S],
    bounds: &Bounds<S>,
) -> Result<LinearLeastSquaresResult<S>, SciOxideError> {
    let problem = LinearLeastSquaresProblem::from_dense(a, b)?.with_bounds(bounds)?;
    let result = BoundedLinearLeastSquares::<S>::default().solve(&problem)?;
    termination_error(
        scioxide_core::argument!("a"),
        result.termination(),
        result.counts(),
        result.algorithm(),
    )
    .map_or(Ok(result), Err)
}

/// Approximates a quadratic assignment optimum in the requested direction.
///
/// Optimizes `∑ᵢⱼ aᵢⱼ bₚ₍ᵢ₎ₚ₍ⱼ₎` over permutations `p`. Prepares `O(n²)` data
/// and runs FAQ followed by 2-opt, with 30 FAQ iterations, 100,000 swap checks,
/// step tolerance 0.03, no restarts and seed 42. Completion gives no global
/// optimality or accuracy guarantee. Use [`crate::QapPlan`] to retain partial results.
///
/// # Errors
///
/// Returns `InvalidInput` (`b`) unless the matrices are square of the same order,
/// `NonFinite` for non-finite data, and preparation and numerical errors of
/// [`QapProblemView::prepare`]. Reached method limits are `ConvergenceFailure`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_optimize::{quadratic_assignment, QapDirection};
/// let a = Matrix::from_rows(&[[0.0_f64, 1.0], [1.0, 0.0]])?;
/// let b = Matrix::from_rows(&[[0.0, 2.0], [2.0, 0.0]])?;
/// let result = quadratic_assignment(a.view(), b.view(), QapDirection::Minimize)?;
/// assert!((result.objective().unwrap() - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Vogelstein et al., *PLOS ONE* 10(4), e0121002, 2015.
pub fn quadratic_assignment<S: RealScalar + NumericScalar<Real = S> + Send + Sync>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    direction: QapDirection,
) -> Result<QapResult<S>, SciOxideError> {
    let result = QapProblemView::new(a, b, &[])?
        .prepare()?
        .solve(direction, None)?;
    let counts = scioxide_core::SolveCounts::from_own(
        scioxide_core::EvaluationCounts::from_iterations(result.fw_iterations()),
    );
    termination_error(
        scioxide_core::argument!("a"),
        result.termination(),
        counts,
        None,
    )
    .map_or(Ok(result), Err)
}
