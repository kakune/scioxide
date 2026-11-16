//! Theil–Sen and Siegel slope estimators, and Huber M-estimation of linear models.
use super::{budget_bytes, finite, invalid, number, zeros};
use crate::ranking::TieSums;
use crate::{ConfidenceProbability, QuantileMethod, QuantileProbability};
use crate::{StatisticsOptions, StatsTermination};
use alloc::vec::Vec;

use scioxide_core::{
    Budget, ProblemToken, ProductSum, RealScalar, Scaled, SciOxideError, SolveContext,
    SolveIdentity,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, Matrix, MatrixView, RankPolicy, ThinSvdDecomposition,
    Vector,
};

fn median<S: RealScalar>(values: &mut [S]) -> Result<S, SciOxideError> {
    Ok(crate::empirical::quantile_in_place(
        values,
        QuantileProbability::new(0.5)?,
        QuantileMethod::Linear,
    )?
    .value())
}

/// Returns the median, averaging the two middle order statistics for an even
/// count.
fn scaled_median(values: &mut [Scaled<f64>]) -> Result<Scaled<f64>, SciOxideError> {
    crate::robust::scaled_quantile_by(
        values.len(),
        QuantileProbability::new(0.5)?,
        QuantileMethod::Linear,
        |lower, upper| {
            crate::empirical::select_adjacent_by(values, lower, upper, |a, b| a.cmp_value(*b))
        },
    )?
    .total()
}

/// Returns the `rank`-th smallest value (from zero) and reorders the buffer.
fn scaled_order_statistic(values: &mut [Scaled<f64>], rank: usize) -> Scaled<f64> {
    *values
        .select_nth_unstable_by(rank, |a, b| a.cmp_value(*b))
        .1
}

fn pair_slope<S: RealScalar>(a: S, b: S, c: S, d: S) -> Result<Scaled<f64>, SciOxideError> {
    Scaled::from_value(d.into_f64())?
        .sub(Scaled::from_value(b.into_f64())?)?
        .div(Scaled::from_value(c.into_f64())?.sub(Scaled::from_value(a.into_f64())?)?)
}

/// Returns the median slope from point `i` to every point with a different
/// abscissa.
///
/// The slopes are computed in `f64` when every difference is finite and every
/// quotient is normal or zero, where they equal [`pair_slope`]. Otherwise they
/// are computed in the extended exponent range.
fn point_median_slope<S: RealScalar>(
    x: &[S],
    y: &[S],
    i: usize,
    native: &mut [f64],
    pairs: &mut [Scaled<f64>],
) -> Result<Scaled<f64>, SciOxideError> {
    let (xi, yi) = (x[i].into_f64(), y[i].into_f64());
    let mut count = 0;
    let mut exact = true;
    for (&xj, &yj) in x.iter().zip(y) {
        if xj != x[i] {
            let (dx, dy) = (xj.into_f64() - xi, yj.into_f64() - yi);
            let slope = dy / dx;
            exact &= dx.is_finite()
                && dy.is_finite()
                && (dy == 0.0 || (slope.is_finite() && slope.abs() >= f64::MIN_POSITIVE));
            native[count] = slope;
            count += 1;
        }
    }
    if count == 0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant predictor",
        ));
    }
    if exact {
        return crate::robust::scaled_quantile_by(
            count,
            QuantileProbability::new(0.5)?,
            QuantileMethod::Linear,
            |lower, upper| {
                let [low, high] = crate::empirical::select_adjacent_by(
                    &mut native[..count],
                    lower,
                    upper,
                    f64::total_cmp,
                )?;
                Ok([Scaled::from_value(low)?, Scaled::from_value(high)?])
            },
        )?
        .total();
    }
    let mut count = 0;
    for j in 0..x.len() {
        if x[i] != x[j] {
            pairs[count] = pair_slope(x[i], y[i], x[j], y[j])?;
            count += 1;
        }
    }
    scaled_median(&mut pairs[..count])
}

fn line_intercept<S: RealScalar>(
    x: S,
    y: S,
    slope: Scaled<f64>,
) -> Result<Scaled<f64>, SciOxideError> {
    Scaled::from_value(y.into_f64())?.difference_of_products(
        Scaled::ONE,
        slope,
        Scaled::from_value(x.into_f64())?,
    )
}
/// Selects the rank-based slope estimator of [`robust_slope`].
///
/// # References
///
/// - P. K. Sen, "Estimates of the regression coefficient based on Kendall's
///   tau", *J. Amer. Statist. Assoc.* 63, 1379–1389, 1968.
/// - A. F. Siegel, "Robust regression using repeated medians", *Biometrika*
///   69(1), 242–244, 1982.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustSlopeMethod {
    /// Theil–Sen estimator: the median of the slopes `(yⱼ − yᵢ) / (xⱼ − xᵢ)`
    /// over all pairs with `xᵢ ≠ xⱼ`. Its breakdown point is about 29%. It
    /// also gives Sen's confidence interval for the slope.
    TheilSen,
    /// Siegel's repeated median estimator: `medianᵢ medianⱼ≠ᵢ (yⱼ − yᵢ) /
    /// (xⱼ − xᵢ)`. Its breakdown point is 50%. It gives no confidence interval.
    Siegel,
}
/// Selects how [`robust_slope`] estimates the intercept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RobustIntercept {
    /// For Theil–Sen, `median(y) − β̂₁ median(x)`. For Siegel, the repeated
    /// median `medianᵢ medianⱼ≠ᵢ (yᵢ − bᵢⱼ xᵢ)`, where `bᵢⱼ` is the slope
    /// through points `i` and `j`.
    Separate,
    /// `medianᵢ (yᵢ − β̂₁ xᵢ)`, the median of the residual intercepts.
    #[default]
    Joint,
}
/// Holds the slope and intercept from [`robust_slope`].
///
/// The fields are read-only.
///
/// ```compile_fail
/// fn corrupt(mut fit: scioxide_stats::RobustSlope<f64>) { fit.slope_interval = None; }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RobustSlope<S: RealScalar> {
    slope: S,
    intercept: S,
    slope_interval: Option<scioxide_core::Interval<S>>,
    method: RobustSlopeMethod,
    intercept_method: RobustIntercept,
}
impl<S: RealScalar> RobustSlope<S> {
    /// Returns the slope estimate `β̂₁`.
    pub const fn slope(&self) -> S {
        self.slope
    }
    /// Returns the intercept estimate `β̂₀`.
    pub const fn intercept(&self) -> S {
        self.intercept
    }
    /// Returns Sen's confidence interval for the slope, or `None` for the
    /// Siegel estimator.
    pub const fn slope_interval(&self) -> Option<scioxide_core::Interval<S>> {
        self.slope_interval
    }
    /// Returns the slope estimator.
    pub const fn method(&self) -> RobustSlopeMethod {
        self.method
    }
    /// Returns the intercept estimator.
    pub const fn intercept_method(&self) -> RobustIntercept {
        self.intercept_method
    }
}
/// Holds the scratch storage of [`robust_slope`] for a fixed sample size and
/// estimator.
pub struct RobustSlopeWorkspace<S: RealScalar> {
    pairs: Vec<Scaled<f64>>,
    native: Vec<f64>,
    work: Vec<S>,
    slopes: Vec<Scaled<f64>>,
    intercepts: Vec<Scaled<f64>>,
    method: RobustSlopeMethod,
}
impl<S: RealScalar> RobustSlopeWorkspace<S> {
    /// Allocates a workspace for `n` points.
    ///
    /// Theil–Sen needs storage for `n(n − 1)/2` slopes, Siegel O(n). Only
    /// `options.max_bytes()` applies.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n < 2`.
    /// - `ResourceLimit` if the storage exceeds `options.max_bytes()`.
    pub fn new(
        n: usize,
        method: RobustSlopeMethod,
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        if n < 2 {
            return Err(invalid(
                scioxide_core::argument!("n"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "robust slopes need at least two points",
            ));
        }
        let (pairs, native) = if method == RobustSlopeMethod::TheilSen {
            let pairs = n.checked_mul(n - 1).map(|v| v / 2).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
            (pairs, 0)
        } else {
            (n, n)
        };
        let bytes = n
            .checked_mul(2)
            .and_then(|v| v.checked_add(pairs))
            .and_then(|v| v.checked_mul(core::mem::size_of::<Scaled<f64>>()))
            .and_then(|v| v.checked_add(n.checked_mul(core::mem::size_of::<S>())?))
            .and_then(|v| v.checked_add(native.checked_mul(core::mem::size_of::<f64>())?))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        budget_bytes(bytes, 1, options.max_bytes())?;
        let scaled_zeros = |n| {
            scioxide_core::backend::try_filled(
                n,
                Scaled::ZERO,
                scioxide_core::quantity!("robust_slope_storage"),
            )
        };
        Ok(Self {
            pairs: scaled_zeros(pairs)?,
            native: zeros(native)?,
            work: zeros(n)?,
            slopes: scaled_zeros(n)?,
            intercepts: scaled_zeros(n)?,
            method,
        })
    }
}

/// Options of the Huber M-estimator: the tuning constant, a step tolerance,
/// iteration limits and a byte limit.
///
/// Defaults to tuning 1.345 (95% asymptotic normal efficiency), the default
/// step tolerance, 100 IRLS iterations, 4096 SVD iterations and no byte limit.
///
/// # References
///
/// P. J. Huber, “Robust estimation of a location parameter”, Ann. Math. Statist.
/// 35 (1964), 73–101.
///
/// The estimator resists outliers in the response but not high-leverage points
/// in the design.
#[derive(Clone, Copy, Debug)]
pub struct HuberOptions<S: RealScalar> {
    /// Huber tuning constant `c`, in units of the residual scale.
    tuning: Option<S>,
    /// Tolerance on the change of the coefficients.
    tolerance: scioxide_core::StepTolerance<S>,
    /// Maximum number of weighted least-squares solves, including the first
    /// unweighted one.
    iterations: Budget,
    /// Iteration limit of each SVD.
    factor_iterations: Budget,
    max_bytes: usize,
}
impl<S: RealScalar> Default for HuberOptions<S> {
    fn default() -> Self {
        Self {
            tuning: None,
            tolerance: Default::default(),
            iterations: Budget::ONE.saturating_add(99),
            factor_iterations: Budget::ONE.saturating_add(4095),
            max_bytes: usize::MAX,
        }
    }
}
impl<S: RealScalar> HuberOptions<S> {
    /// Sets the tuning constant `c`. It must be positive and finite.
    pub fn with_tuning(mut self, value: S) -> Self {
        self.tuning = Some(value);
        self
    }
    /// Returns the explicit tuning constant; `None` selects 1.345 (95% normal efficiency).
    pub fn tuning(&self) -> Option<S> {
        self.tuning
    }
    /// Sets the tolerance on the change of the coefficients between iterations.
    pub fn with_tolerance(mut self, value: scioxide_core::StepTolerance<S>) -> Self {
        self.tolerance = value;
        self
    }
    /// Returns the step tolerance.
    pub fn tolerance(&self) -> scioxide_core::StepTolerance<S> {
        self.tolerance
    }
    /// Sets the maximum number of weighted least-squares solves.
    pub fn with_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Returns the maximum number of weighted least-squares solves.
    pub fn iterations(&self) -> Budget {
        self.iterations
    }
    /// Sets the iteration limit of each SVD.
    pub fn with_factor_iterations(mut self, value: Budget) -> Self {
        self.factor_iterations = value;
        self
    }
    /// Returns the iteration limit of each SVD.
    pub fn factor_iterations(&self) -> Budget {
        self.factor_iterations
    }
    /// Sets the byte limit for the workspace and output. The default is
    /// `usize::MAX`.
    pub fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the byte limit.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}
/// Stores the coefficients, residuals, weights and scale of a Huber iterate.
struct HuberStorage<S: RealScalar> {
    /// Coefficients.
    pub coefficients: Vector<S>,
    /// Unweighted residuals.
    pub residuals: Vector<S>,
    /// Huber weights computed from these residuals and scale.
    pub weights: Vector<S>,
    /// Normalized MAD of the residuals.
    pub scale: S,
    /// Number of completed solves.
    pub iterations: usize,
    /// Numerical rank of the last weighted design.
    pub rank: usize,
}
impl<S: RealScalar> HuberStorage<S> {
    /// Allocates storage for `n` rows and `p` columns.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub(crate) fn new(n: usize, p: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            coefficients: Vector::zeros(p)?,
            residuals: Vector::zeros(n)?,
            weights: Vector::zeros(n)?,
            scale: S::ZERO,
            iterations: 0,
            rank: 0,
        })
    }
}
fn copy_huber<S: RealScalar>(out: &mut HuberStorage<S>, source: &HuberStorage<S>) {
    out.coefficients
        .as_mut_slice()
        .copy_from_slice(source.coefficients.as_slice());
    out.residuals
        .as_mut_slice()
        .copy_from_slice(source.residuals.as_slice());
    out.weights
        .as_mut_slice()
        .copy_from_slice(source.weights.as_slice());
    out.scale = source.scale;
    out.iterations = source.iterations;
    out.rank = source.rank;
}
/// Stores the weighted design, its SVD, the current iterate and the last
/// accepted iterate.
struct HuberScratch<S: RealScalar> {
    matrix: Matrix<S>,
    rhs: Vector<S>,
    previous: Vector<S>,
    median: Vec<S>,
    factor: ThinSvdDecomposition<S>,
    svd: LinalgWorkspace<'static, S>,
    stage: HuberStorage<S>,
    accepted: HuberStorage<S>,
}
impl<S: RealScalar> HuberScratch<S> {
    /// Allocates the scratch storage and runs one SVD to size the factors.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `p = 0` or `n < p`, and `ResourceLimit` above
    /// `max_bytes`.
    pub(crate) fn new(n: usize, p: usize, max_bytes: usize) -> Result<Self, SciOxideError> {
        if p == 0 || n < p {
            return Err(invalid(
                scioxide_core::argument!("p"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Huber needs a nonempty tall design",
            ));
        }
        budget_bytes(
            n.checked_mul(p)
                .and_then(|v| v.checked_add(n))
                .and_then(|v| v.checked_add(p))
                .and_then(|v| v.checked_mul(64))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                })?,
            core::mem::size_of::<S>(),
            max_bytes,
        )?;
        let matrix = Matrix::zeros(n, p)?;
        let mut factor = ThinSvdDecomposition::empty();
        let mut svd = LinalgWorkspace::new();
        svd.try_reserve(
            &scioxide_linalg::ThinSvdDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::SvdWorkspaceRequest::new(n, p, &(Default::default())),
            )?,
        )?;
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &matrix,
            &Default::default(),
            &mut factor,
            &mut svd,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(Self {
            matrix,
            rhs: Vector::zeros(n)?,
            previous: Vector::zeros(p)?,
            median: zeros(n)?,
            factor,
            svd,
            stage: HuberStorage::new(n, p)?,
            accepted: HuberStorage::new(n, p)?,
        })
    }
}
/// Returns the solve identity of a Huber fit.
///
/// # Errors
///
/// Does not fail.
fn huber_identity(
    problem: ProblemToken,
    n: usize,
    p: usize,
) -> Result<SolveIdentity, SciOxideError> {
    Ok(SolveIdentity::new(
        problem,
        scioxide_core::algorithm_id!("stats", "huber", 1),
        ((n as u128) << 64) | (p as u128),
        0,
    ))
}
/// Runs the Huber iteration and copies the last accepted iterate to `output`.
///
/// The scale is re-estimated from the residuals at every iteration. An exact
/// fit stops with scale zero. A zero MAD without an exact fit is an error.
///
/// # Errors
///
/// Returns `InvalidInput` for inconsistent shapes, non-finite data, a
/// rank-deficient weighted design or a zero MAD, and the errors of the SVD.
fn huber_attempt<S: RealScalar>(
    design: MatrixView<'_, S>,
    response: &[S],
    options: HuberOptions<S>,
    output: &mut HuberStorage<S>,
    work: &mut HuberScratch<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<StatsTermination, SciOxideError> {
    let tuning = options.tuning.map_or_else(|| number(1.345), Ok)?;
    let (n, p) = (design.rows(), design.cols());
    if n != response.len()
        || work.matrix.rows() != n
        || work.matrix.cols() != p
        || output.coefficients.len() != p
        || output.residuals.len() != n
        || output.weights.len() != n
        || !tuning.is_finite()
        || tuning <= S::ZERO
        || response.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            scioxide_core::argument!("design"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid Huber inputs or workspace",
        ));
    }
    for i in 0..n {
        for j in 0..p {
            finite(design.get(i, j)?)?;
        }
    }
    work.stage.weights.as_mut_slice().fill(S::ONE);
    work.stage.coefficients.as_mut_slice().fill(S::ZERO);
    work.stage.iterations = 0;
    work.accepted.iterations = 0;
    let mut termination = StatsTermination::BudgetExhausted;
    for iteration in 0..options.iterations.get() {
        if let Some(stop) = context.stop_reason()? {
            termination = StatsTermination::Stopped(stop);
            break;
        }
        if !context.record_iteration()?.is_granted() {
            break;
        }
        scioxide_linalg::VectorView::from(&work.stage.coefficients).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (work.previous.view_mut()).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        for (i, &y) in response.iter().enumerate() {
            let w = work.stage.weights.view().get(i)?.sqrt();
            work.rhs.view_mut().set(i, finite(y * w)?)?;
            for j in 0..p {
                work.matrix
                    .view_mut()
                    .set(i, j, finite(design.get(i, j)? * w)?)?;
            }
        }
        if !context
            .record_provider(scioxide_core::EvaluationRole::Factorization)?
            .is_granted()
        {
            break;
        }
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &work.matrix,
            &LinalgExecutionOptions::default().with_iteration_budget(options.factor_iterations),
            &mut work.factor,
            &mut work.svd,
            context.execution(),
        )?;
        let rank = work
            .factor
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
            .rank();
        if rank < p {
            return Err(invalid(
                scioxide_core::argument!("design"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Huber design is rank deficient",
            ));
        }
        if !context
            .record_provider(scioxide_core::EvaluationRole::LinearSolve)?
            .is_granted()
        {
            break;
        }
        work.factor.solve_into(
            &work.rhs,
            RankPolicy::DimensionScaledEpsilon,
            S::ZERO,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            work.stage.coefficients.view_mut(),
            &mut work.svd,
            context.execution(),
        )?;
        for (i, &y) in response.iter().enumerate() {
            let mut sum = ProductSum::new();
            for (j, &b) in work.stage.coefficients.as_slice().iter().enumerate() {
                sum.add_product(design.get(i, j)?, b)?;
            }
            let r = finite(y - sum.finish()?)?;
            work.stage.residuals.view_mut().set(i, r)?;
            work.median[i] = r;
        }
        work.stage.iterations = iteration + 1;
        work.stage.rank = rank;
        let roundoff = number::<S>(32.0)? * S::EPSILON;
        if work
            .stage
            .residuals
            .as_slice()
            .iter()
            .zip(response)
            .all(|(&r, &y)| r.abs() <= roundoff * y.abs() + roundoff * (y - r).abs())
        {
            work.stage.scale = S::ZERO;
            work.stage.weights.as_mut_slice().fill(S::ONE);
            termination = StatsTermination::ExactFit;
            context.commit_accepted(scioxide_core::PartialResultTrust::ResidualChecked)?;
            copy_huber(&mut work.accepted, &work.stage);
            break;
        }
        let center = median(&mut work.median)?;
        for (v, &r) in work.median.iter_mut().zip(work.stage.residuals.as_slice()) {
            *v = (r - center).abs();
        }
        let scale = finite(median(&mut work.median)? / number::<S>(0.6744897501960817)?)?;
        if scale == S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Huber residual MAD is zero without an exact fit",
            ));
        }
        work.stage.scale = scale;
        for (w, &r) in work
            .stage
            .weights
            .as_mut_slice()
            .iter_mut()
            .zip(work.stage.residuals.as_slice())
        {
            *w = if r == S::ZERO {
                S::ONE
            } else {
                (tuning * scale / r.abs()).min(S::ONE)
            };
        }
        let mut change = S::ZERO;
        let mut magnitude = S::ZERO;
        for (&a, &b) in work
            .previous
            .as_slice()
            .iter()
            .zip(work.stage.coefficients.as_slice())
        {
            change = change.max((a - b).abs());
            magnitude = magnitude.max(b.abs());
        }
        context.commit_accepted(scioxide_core::PartialResultTrust::ResidualChecked)?;
        copy_huber(&mut work.accepted, &work.stage);
        if iteration > 0 && options.tolerance.is_satisfied(change, magnitude)? {
            termination = StatsTermination::Converged;
            break;
        }
    }
    if work.accepted.iterations > 0 {
        scioxide_linalg::VectorView::from(&work.accepted.coefficients).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (output.coefficients.view_mut()).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        scioxide_linalg::VectorView::from(&work.accepted.residuals).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (output.residuals.view_mut()).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        scioxide_linalg::VectorView::from(&work.accepted.weights).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (output.weights.view_mut()).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        output.scale = work.accepted.scale;
        output.iterations = work.accepted.iterations;
        output.rank = work.accepted.rank;
    }
    Ok(termination)
}

/// Computes the result of [`huber_location_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`huber_location_with_options`] for the fixed default settings.
pub fn huber_location<S: RealScalar>(samples: &[S]) -> Result<HuberFit<S>, SciOxideError> {
    let fit = huber_location_with_options(samples, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("samples"),
        )
    })?;
    if let Some(error) = fit.termination().one_shot_error(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(fit.report().counts()),
    ) {
        return Err(error);
    }
    Ok(fit)
}
/// Computes Huber's M-estimate of location.
///
/// This is [`PreparedHuber`] with a design of one column of ones, run
/// sequentially. `options.max_bytes()` bounds the design, the workspace and
/// the output together.
///
/// # Errors
///
/// - `InvalidInput` if `samples` is empty or contains a NaN or infinite value.
/// - `InvalidInput` if the MAD of the residuals is zero but the samples are
///   not all equal.
/// - `ResourceLimit` if the storage exceeds `options.max_bytes()`.
/// - `ConvergenceFailure` if the iteration stops before the first iterate.
///
/// # Examples
///
/// ```
/// // The outlier 100 has little influence on the estimate.
/// use scioxide_core::Budget;
/// use scioxide_stats::{HuberOptions, huber_location_with_options};
/// let options = HuberOptions::default().with_iterations(Budget::try_from(50)?);
/// let fit = huber_location_with_options(&[1.0_f64, 2.0, 3.0, 4.0, 100.0], options)?;
/// assert!(fit.coefficients().as_slice()[0] < 4.0);
/// assert!(huber_location_with_options(&[1.0_f64, 2.0, 3.0], options.with_max_bytes(1)).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Huber, "Robust estimation of a location parameter", *Ann. Math.
/// Statist.* 35(1), 73–101, 1964.
pub fn huber_location_with_options<S: RealScalar>(
    samples: &[S],
    options: HuberOptions<S>,
) -> Result<HuberFit<S>, SciOxideError> {
    budget_bytes(samples.len(), core::mem::size_of::<S>(), options.max_bytes)?;
    let design = scioxide_core::backend::try_filled(
        samples.len(),
        S::ONE,
        scioxide_core::quantity!("huber_location_design"),
    )?;
    PreparedHuber::new(MatrixView::from_slice(samples.len(), 1, &design)?, options)?.fit(samples)
}

/// Describes how a [`PreparedHuber`] fit ended.
#[derive(Clone, Copy, Debug)]
pub struct HuberReport {
    termination: StatsTermination,
    counts: scioxide_core::SolveCounts,
    accepted_iterations: usize,
}
impl HuberReport {
    /// Returns the termination status. It is set even when no iterate was
    /// accepted.
    pub const fn termination(self) -> StatsTermination {
        self.termination
    }
    /// Returns the iterations and evaluations of this fit, including nested
    /// solves.
    pub const fn counts(self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of accepted iterations.
    pub const fn accepted_iterations(self) -> usize {
        self.accepted_iterations
    }
}
/// Holds a Huber fit and its report, as returned by [`PreparedHuber::fit`].
///
/// ```compile_fail
/// let _ = scioxide_stats::HuberFit::<f64>::new(8, 2);
/// ```
pub struct HuberFit<S: RealScalar> {
    raw: HuberStorage<S>,
    report: HuberReport,
}
impl<S: RealScalar> HuberFit<S> {
    /// Returns the report.
    pub fn report(&self) -> HuberReport {
        self.report
    }
    /// Returns the termination status.
    pub fn termination(&self) -> StatsTermination {
        self.report.termination
    }
    /// Returns the coefficient estimates `β̂`.
    pub fn coefficients(&self) -> &Vector<S> {
        &self.raw.coefficients
    }
    /// Returns the residuals `y − X β̂`.
    pub fn residuals(&self) -> &Vector<S> {
        &self.raw.residuals
    }
    /// Returns the Huber weights `min(1, c σ̂ / |rᵢ|)` of the final residuals.
    pub fn weights(&self) -> &Vector<S> {
        &self.raw.weights
    }
    /// Returns the residual scale `σ̂ = MAD / 0.6745`, zero for an exact fit.
    pub fn scale(&self) -> S {
        self.raw.scale
    }
    /// Returns the number of weighted least-squares solves.
    pub fn iterations(&self) -> usize {
        self.raw.iterations
    }
    /// Returns the numerical rank of the last weighted design.
    pub fn rank(&self) -> usize {
        self.raw.rank
    }
}
/// Holds the output of [`PreparedHuber::fit_into`].
///
/// Create it with [`PreparedHuber::output`]. It holds no fit until a call
/// accepts an iterate.
pub struct HuberOutput<S: RealScalar> {
    raw: HuberStorage<S>,
    report: Option<HuberReport>,
}
impl<S: RealScalar> HuberOutput<S> {
    /// Returns the last fit written to this output, or `None` if there is none.
    pub fn fit(&self) -> Option<HuberFitView<'_, S>> {
        self.report.map(|report| HuberFitView {
            raw: &self.raw,
            report,
        })
    }
}
/// Borrows a Huber fit and its report from a [`HuberOutput`].
pub struct HuberFitView<'a, S: RealScalar> {
    raw: &'a HuberStorage<S>,
    report: HuberReport,
}
impl<S: RealScalar> HuberFitView<'_, S> {
    /// Returns the report.
    pub fn report(&self) -> HuberReport {
        self.report
    }
    /// Returns the termination status.
    pub fn termination(&self) -> StatsTermination {
        self.report.termination
    }
    /// Returns the coefficient estimates `β̂`.
    pub fn coefficients(&self) -> &Vector<S> {
        &self.raw.coefficients
    }
    /// Returns the residuals `y − X β̂`.
    pub fn residuals(&self) -> &Vector<S> {
        &self.raw.residuals
    }
    /// Returns the Huber weights of the final residuals.
    pub fn weights(&self) -> &Vector<S> {
        &self.raw.weights
    }
    /// Returns the residual scale `σ̂`, zero for an exact fit.
    pub fn scale(&self) -> S {
        self.raw.scale
    }
    /// Returns the number of weighted least-squares solves.
    pub fn iterations(&self) -> usize {
        self.raw.iterations
    }
    /// Returns the numerical rank of the last weighted design.
    pub fn rank(&self) -> usize {
        self.raw.rank
    }
}
/// Borrows the last accepted Huber iterate from a [`HuberWorkspace`].
pub struct HuberAccepted<'a, S: RealScalar> {
    raw: &'a HuberStorage<S>,
}
impl<S: RealScalar> HuberAccepted<'_, S> {
    /// Returns the coefficients.
    pub fn coefficients(&self) -> &Vector<S> {
        &self.raw.coefficients
    }
    /// Returns the residuals.
    pub fn residuals(&self) -> &Vector<S> {
        &self.raw.residuals
    }
    /// Returns the Huber weights.
    pub fn weights(&self) -> &Vector<S> {
        &self.raw.weights
    }
    /// Returns the residual scale.
    pub fn scale(&self) -> S {
        self.raw.scale
    }
    /// Returns the number of weighted least-squares solves.
    pub fn iterations(&self) -> usize {
        self.raw.iterations
    }
    /// Returns the numerical rank of the last weighted design.
    pub fn rank(&self) -> usize {
        self.raw.rank
    }
}
/// Fits a linear model by Huber's M-estimator, using iteratively reweighted
/// least squares (IRLS).
///
/// For an `n × p` design `X` and a response `y`, the estimate minimizes
///
/// ```text
/// Σᵢ ρ_c(rᵢ / σ̂),   ρ_c(u) = u²/2 for |u| ≤ c,   c|u| − c²/2 otherwise,
/// ```
///
/// where `rᵢ = yᵢ − xᵢᵀβ` and `σ̂ = MAD(r) / 0.6745` is the normalized median
/// absolute deviation of the residuals. The first iteration is ordinary least
/// squares. Each later iteration solves a weighted least-squares problem by
/// thin SVD with the weights `wᵢ = min(1, c σ̂ / |rᵢ|)` from the previous
/// residuals, and re-estimates `σ̂`. The iteration stops when the largest
/// coefficient change satisfies the step tolerance, or with
/// `StatsTermination::ExactFit` when all residuals vanish to rounding. Each
/// iteration costs O(np²).
///
/// The model has no implicit intercept: include a column of ones. The design
/// is borrowed and must have at least as many rows as columns and full column
/// rank.
///
/// # References
///
/// - P. J. Huber, "Robust estimation of a location parameter", *Ann. Math.
///   Statist.* 35(1), 73–101, 1964.
/// - P. W. Holland and R. E. Welsch, "Robust regression using iteratively
///   reweighted least-squares", *Comm. Statist. Theory Methods* 6(9),
///   813–827, 1977.
#[derive(Clone)]
pub struct PreparedHuber<'a, S: RealScalar> {
    design: MatrixView<'a, S>,
    options: HuberOptions<S>,
}
/// Holds the scratch storage of [`PreparedHuber::fit_into`] and a record of
/// the last fit.
///
/// Create it with [`PreparedHuber::workspace`].
pub struct HuberWorkspace<S: RealScalar> {
    raw: HuberScratch<S>,
    slot: crate::execution::SolveSlot<usize>,
    retained_bytes: usize,
}
impl<S: RealScalar> HuberWorkspace<S> {
    /// Returns the record of the last fit run with this workspace, including a
    /// fit rejected by input checks.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<usize>> {
        self.slot.last_attempt()
    }
    /// Returns the last accepted iterate of the last fit, if it accepted one.
    pub fn last_accepted(&self) -> Option<HuberAccepted<'_, S>> {
        self.slot
            .last_attempt()
            .and_then(scioxide_core::AttemptRecord::progress)
            .map(|_| HuberAccepted {
                raw: &self.raw.accepted,
            })
    }
}
impl<'a, S: RealScalar> PreparedHuber<'a, S> {
    /// Checks a borrowed design and the options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the design has no columns or fewer rows than
    ///   columns.
    /// - `InvalidInput` if the tuning constant is not positive and finite.
    /// - `NonFinite` if the design contains a NaN, or `NumericalInstability`
    ///   if it contains an infinite entry.
    pub fn new(design: MatrixView<'a, S>, options: HuberOptions<S>) -> Result<Self, SciOxideError> {
        let tuning = options.tuning.map_or_else(|| number(1.345), Ok)?;
        if design.cols() == 0
            || design.rows() < design.cols()
            || !tuning.is_finite()
            || tuning <= S::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("design"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Huber design or tuning",
            ));
        }
        for row in 0..design.rows() {
            for col in 0..design.cols() {
                finite(design.get(row, col)?)?;
            }
        }
        Ok(Self { design, options })
    }
    /// Allocates the workspace for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds `options.max_bytes()`.
    pub fn workspace(&self) -> Result<HuberWorkspace<S>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        let retained_bytes = n
            .checked_mul(p)
            .and_then(|v| v.checked_add(n))
            .and_then(|v| v.checked_add(p))
            .and_then(|v| v.checked_mul(64))
            .and_then(|v| v.checked_mul(core::mem::size_of::<S>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("huber_workspace_bytes"))
            })?;

        Ok(HuberWorkspace {
            raw: HuberScratch::new(
                self.design.rows(),
                self.design.cols(),
                self.options.max_bytes,
            )?,
            slot: Default::default(),
            retained_bytes,
        })
    }
    /// Allocates an empty output for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds `options.max_bytes()`.
    pub fn output(&self) -> Result<HuberOutput<S>, SciOxideError> {
        let n = self.design.rows();
        let p = self.design.cols();
        let count = n
            .checked_mul(2)
            .and_then(|n| n.checked_add(p))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("huber_output"))
            })?;
        crate::error_helpers::limit_bytes(
            count
                .checked_mul(core::mem::size_of::<S>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("huber_output"))
                })?,
            self.options.max_bytes,
        )?;
        Ok(HuberOutput {
            raw: HuberStorage::new(n, p)?,
            report: None,
        })
    }
    /// Fits one response and writes the last accepted iterate to `output`.
    ///
    /// Each iteration counts toward the iteration limit of the options and of
    /// `scope`. When the limit, cancellation or a deadline stops the
    /// iteration, the method returns `Ok` with an unsuccessful termination and
    /// writes the last accepted iterate, if any. See
    /// [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `response`, `output` or `work` does not match the
    ///   design shape.
    /// - `InvalidInput` if `response` contains a NaN or infinite value.
    /// - `InvalidInput` if a weighted design is numerically rank deficient.
    /// - `InvalidInput` if the MAD of the residuals is zero without an exact
    ///   fit.
    /// - `ResourceLimit` if `work` exceeds `options.max_bytes()`.
    /// - The error of the SVD.
    pub fn fit_into<'scope, 'capability>(
        &self,
        response: &[S],
        output: &mut HuberOutput<S>,
        work: &mut HuberWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<HuberReport, SciOxideError> {
        work.slot.reject();
        crate::error_helpers::limit_bytes(work.retained_bytes, self.options.max_bytes)?;
        let (n, p) = (self.design.rows(), self.design.cols());
        if response.len() != n
            || response.iter().any(|v| !v.is_finite())
            || output.raw.coefficients.len() != p
            || output.raw.residuals.len() != n
            || work.raw.matrix.rows() != n
            || work.raw.matrix.cols() != p
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Huber response, output or workspace shape",
            ));
        }
        work.raw.accepted.iterations = 0;
        let accepted = core::cell::Cell::new(0usize);
        let mut slot = core::mem::take(&mut work.slot);
        let result = slot.run(
            scope.into(),
            huber_identity(ProblemToken::new(0), n, p)?,
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Iteration,
                self.options.iterations,
            ),
            |context| {
                let result = huber_attempt(
                    self.design,
                    response,
                    self.options,
                    &mut output.raw,
                    &mut work.raw,
                    context,
                );
                accepted.set(work.raw.accepted.iterations);
                result.map(|termination| HuberReport {
                    termination,
                    counts: context.counts(),
                    accepted_iterations: work.raw.accepted.iterations,
                })
            },
            |result| {
                (
                    match result {
                        Ok(report)
                            if matches!(
                                report.termination,
                                StatsTermination::Converged | StatsTermination::ExactFit
                            ) =>
                        {
                            scioxide_core::AttemptOutcome::Succeeded
                        }
                        Ok(_) => scioxide_core::AttemptOutcome::Stopped,
                        Err(_) => scioxide_core::AttemptOutcome::Failed,
                    },
                    (accepted.get() > 0).then_some(accepted.get()),
                )
            },
        );
        work.slot = slot;
        if let Ok(report) = &result
            && report.accepted_iterations > 0
        {
            output.report = Some(*report);
        }
        result
    }
    /// Fits one response sequentially and returns a new result.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace`], [`Self::output`] and
    ///   [`Self::fit_into`].
    /// - `ConvergenceFailure` if the iteration stops before the first accepted
    ///   iterate.
    pub fn fit(&self, response: &[S]) -> Result<HuberFit<S>, SciOxideError> {
        let mut output = self.output()?;
        let mut work = self.workspace()?;
        let report = self.fit_into(
            response,
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        if report.accepted_iterations == 0 {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::IterationLimit,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts),
                "Huber stopped before a complete iterate",
            ));
        }
        Ok(HuberFit {
            raw: output.raw,
            report,
        })
    }
}

impl<S: RealScalar> RobustSlopeWorkspace<S> {
    /// Fits a line `y = β₀ + β₁ x` by the Theil–Sen or the Siegel repeated median
    /// estimator.
    ///
    /// The estimator is the one `work` was created for (see [`RobustSlopeMethod`]).
    /// Both cost O(n²) time. Medians and order statistics are found by selection,
    /// not by sorting. Slopes and intercepts are computed in an extended exponent
    /// range and rounded to `S` once.
    ///
    /// For Theil–Sen the slope interval has the order statistics of the `N` pair
    /// slopes at ranks `(N − C)/2` and `(N + C)/2 + 1` (from one, rounded to
    /// integers), where `C = z √Var(S)`, `z` the normal quantile for
    /// `confidence`, and `Var(S)` the variance of Kendall's statistic with the
    /// tie correction for both `x` and `y`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` and `y` differ in length, or their length differs
    ///   from the size `work` was created for.
    /// - `InvalidInput` if an observation is NaN or infinite.
    /// - `InvalidInput` if `x` is constant.
    /// - `NumericalInstability` if the slope, intercept or an interval endpoint
    ///   overflows `S`.
    ///
    /// # References
    ///
    /// See [`RobustSlopeMethod`].
    pub fn evaluate(
        &mut self,
        x: &[S],
        y: &[S],
        intercept_method: RobustIntercept,
        confidence: ConfidenceProbability,
    ) -> Result<RobustSlope<S>, SciOxideError> {
        let n = x.len();
        if n != y.len() || n != self.work.len() || x.iter().chain(y).any(|v| !v.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("work"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid robust slope samples or workspace",
            ));
        }
        let (slope, interval) = match self.method {
            RobustSlopeMethod::TheilSen => {
                let mut count = 0;
                for i in 0..n {
                    for j in i + 1..n {
                        if x[j] != x[i] {
                            self.pairs[count] = pair_slope(x[i], y[i], x[j], y[j])?;
                            count += 1;
                        }
                    }
                }
                if count == 0 {
                    return Err(invalid(
                        scioxide_core::argument!("x"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "constant predictor",
                    ));
                }
                let slope = scaled_median(&mut self.pairs[..count])?;
                self.work.copy_from_slice(x);
                let tx = TieSums::of_values(&mut self.work).kendall;
                self.work.copy_from_slice(y);
                let ty = TieSums::of_values(&mut self.work).kendall;
                let nf = n as f64;
                let variance = ((nf * (nf - 1.0) * (2.0 * nf + 5.0) - tx - ty) / 18.0).max(0.0);
                let width = confidence.normal_critical_value::<f64>()? * variance.sqrt();
                let lo = (((count as f64 - width) / 2.0).round() - 1.0)
                    .max(0.0)
                    .min((count - 1) as f64) as usize;
                let hi = ((count as f64 + width) / 2.0)
                    .round()
                    .max(0.0)
                    .min((count - 1) as f64) as usize;
                let pairs = &mut self.pairs[..count];
                let lower = scaled_order_statistic(pairs, lo).round_to()?;
                let upper = scaled_order_statistic(pairs, hi).round_to()?;
                (slope, Some(scioxide_core::Interval::new(lower, upper)?))
            }
            RobustSlopeMethod::Siegel => {
                for i in 0..n {
                    self.slopes[i] =
                        point_median_slope(x, y, i, &mut self.native, &mut self.pairs)?;
                    if intercept_method == RobustIntercept::Separate {
                        let mut count = 0;
                        for j in 0..n {
                            if x[i] != x[j] {
                                self.pairs[count] = line_intercept(
                                    x[i],
                                    y[i],
                                    pair_slope(x[i], y[i], x[j], y[j])?,
                                )?;
                                count += 1;
                            }
                        }
                        self.intercepts[i] = scaled_median(&mut self.pairs[..count])?;
                    }
                }
                (scaled_median(&mut self.slopes)?, None)
            }
        };
        let intercept = if intercept_method == RobustIntercept::Joint {
            for ((v, &a), &b) in self.slopes.iter_mut().zip(x).zip(y) {
                *v = line_intercept(a, b, slope)?;
            }
            scaled_median(&mut self.slopes)?
        } else if self.method == RobustSlopeMethod::Siegel {
            scaled_median(&mut self.intercepts)?
        } else {
            for (target, &value) in self.slopes.iter_mut().zip(x) {
                *target = Scaled::from_value(value.into_f64())?;
            }
            let mx = scaled_median(&mut self.slopes)?;
            for (target, &value) in self.slopes.iter_mut().zip(y) {
                *target = Scaled::from_value(value.into_f64())?;
            }
            scaled_median(&mut self.slopes)?.difference_of_products(Scaled::ONE, slope, mx)?
        };
        Ok(RobustSlope {
            slope: slope.round_to()?,
            intercept: intercept.round_to()?,
            slope_interval: interval,
            method: self.method,
            intercept_method,
        })
    }
}
/// Estimates a robust slope with a joint intercept and 95% confidence.
///
/// # Errors
///
/// Returns the errors of [`robust_slope_with_options`].
pub fn robust_slope<S: RealScalar>(
    x: &[S],
    y: &[S],
    method: RobustSlopeMethod,
) -> Result<RobustSlope<S>, SciOxideError> {
    robust_slope_with_options(
        x,
        y,
        method,
        RobustIntercept::default(),
        ConfidenceProbability::default(),
        StatisticsOptions::default(),
    )
}
/// Estimates a robust slope and intercept using owned workspace.
///
/// Cost and accuracy follow [`RobustSlopeWorkspace::evaluate`].
///
/// # Errors
///
/// Rejects fewer than two points and returns preparation or numerical errors.
pub fn robust_slope_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    method: RobustSlopeMethod,
    intercept_method: RobustIntercept,
    confidence: ConfidenceProbability,
    options: StatisticsOptions,
) -> Result<RobustSlope<S>, SciOxideError> {
    let mut workspace = RobustSlopeWorkspace::new(x.len(), method, options).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["n"], scioxide_core::argument!("x"))
    })?;
    workspace
        .evaluate(x, y, intercept_method, confidence)
        .map_err(|error| {
            crate::error_helpers::one_shot_argument(error, &["work"], scioxide_core::argument!("x"))
        })
}
