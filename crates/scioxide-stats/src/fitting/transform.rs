use super::{
    FitModel, FitObjective, FitRequest, FitResult, FitSolver, finite, fit_scoped, invalid, number,
};
use crate::{MomentOrder, OnlineCovariance, StatisticsOptions};
use alloc::vec::Vec;

use scioxide_core::{Bounds, ExecutionContext, RealScalar, SciOxideError};

/// Selects a family of monotone power transformations `y = ψ(x; λ)`.
///
/// ```text
/// Box–Cox, x > 0:      ψ = (x^λ − 1)/λ                    if λ ≠ 0,   ln x         if λ = 0
/// Yeo–Johnson, x ≥ 0:  ψ = ((x + 1)^λ − 1)/λ              if λ ≠ 0,   ln(1 + x)    if λ = 0
/// Yeo–Johnson, x < 0:  ψ = −((1 − x)^(2−λ) − 1)/(2 − λ)   if λ ≠ 2,   −ln(1 − x)   if λ = 2
/// ```
///
/// No shift parameter is estimated, so Box–Cox needs positive data.
///
/// # Examples
///
/// ```
/// use scioxide_stats::PowerTransform;
///
/// // At λ = 0, Box–Cox is the natural logarithm.
/// let value = PowerTransform::BoxCox.apply(2.0_f64, 0.0)?;
/// assert!((value - 2.0_f64.ln()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. E. P. Box and D. R. Cox, "An analysis of transformations", *Journal of
///   the Royal Statistical Society, Series B* 26 (1964), 211–252.
/// - I.-K. Yeo and R. A. Johnson, "A new family of power transformations to
///   improve normality or symmetry", *Biometrika* 87 (2000), 954–959.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerTransform {
    /// Box–Cox transformation of positive data.
    BoxCox,
    /// Yeo–Johnson transformation of real data, with separate branches for
    /// `x ≥ 0` and `x < 0`.
    YeoJohnson,
}
impl PowerTransform {
    /// Returns `ψ(x; λ)`.
    ///
    /// The power is computed with `expm1`, so the result stays accurate near the
    /// logarithmic case: `λ = 0`, or `λ = 2` for negative Yeo–Johnson arguments.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` or `lambda` is NaN or infinite.
    /// - `InvalidInput` if `x ≤ 0` for Box–Cox.
    /// - `NumericalInstability` if the result overflows.
    pub fn apply<S: RealScalar>(self, x: S, lambda: S) -> Result<S, SciOxideError> {
        crate::error_helpers::input_finite(x)?;
        crate::error_helpers::input_finite(lambda)?;
        if lambda == S::ONE {
            return match self {
                Self::BoxCox if x <= S::ZERO => Err(invalid(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "Box-Cox requires positive samples",
                )),
                Self::BoxCox => finite(x - S::ONE),
                Self::YeoJohnson => Ok(x),
            };
        }
        let (log, power, sign) = match self {
            Self::BoxCox => {
                if x <= S::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("x"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "Box-Cox requires positive samples",
                    ));
                }
                (x.ln(), lambda, S::ONE)
            }
            Self::YeoJohnson => {
                if x >= S::ZERO {
                    (x.ln_1p(), lambda, S::ONE)
                } else {
                    ((-x).ln_1p(), number::<S>(2.0)? - lambda, -S::ONE)
                }
            }
        };
        let z = power * log;
        // Multiplication may underflow at tiny lambda; use exprel's limiting form.
        let value = if z == S::ZERO {
            log
        } else {
            if z > S::ONE {
                let magnitude = (z - power.abs().ln()).exp() - S::ONE / power.abs();
                if power < S::ZERO {
                    -magnitude
                } else {
                    magnitude
                }
            } else {
                z.exp_m1() / power
            }
        };
        finite(sign * value)
    }
    /// Returns `ln |dψ/dx|`, the log Jacobian term of the transformed likelihood.
    ///
    /// It is `(λ − 1) ln x` for Box–Cox, and `(λ − 1) ln(1 + x)` for `x ≥ 0` or
    /// `(1 − λ) ln(1 − x)` for `x < 0` for Yeo–Johnson.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` or `lambda` is NaN or infinite.
    /// - `InvalidInput` if `x ≤ 0` for Box–Cox.
    /// - `NumericalInstability` if the result overflows.
    pub fn log_jacobian<S: RealScalar>(self, x: S, lambda: S) -> Result<S, SciOxideError> {
        crate::error_helpers::input_finite(x)?;
        crate::error_helpers::input_finite(lambda)?;
        let value = match self {
            Self::BoxCox => {
                if x <= S::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("x"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "Box-Cox requires positive samples",
                    ));
                }
                (lambda - S::ONE) * x.ln()
            }
            Self::YeoJohnson => {
                if x >= S::ZERO {
                    (lambda - S::ONE) * x.ln_1p()
                } else {
                    (S::ONE - lambda) * (-x).ln_1p()
                }
            }
        };
        finite(value)
    }
}

/// Applies a power transformation to each sample, writing `ψ(xᵢ; λ)` to `output`.
///
/// `output` and `scratch` have the length of `samples`, which may be empty.
/// `options` sets the execution policy, and `context` supplies the executor.
///
/// # Errors
///
/// - `InvalidInput` if `output` or `scratch` has the wrong length.
/// - The errors of [`PowerTransform::apply`].
/// - `ExecutionStopped` on cancellation or deadline.
#[allow(clippy::too_many_arguments)]
pub fn power_transform_into<S: RealScalar>(
    kind: PowerTransform,
    samples: &[S],
    lambda: S,
    options: StatisticsOptions,
    output: &mut [S],
    scratch: &mut [S],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    crate::error_helpers::input_finite(lambda)?;
    if samples.len() != output.len() || scratch.len() != samples.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "transform buffer shape",
        ));
    }
    let mut workers = [(); 64];
    crate::estimation::execute(
        options,
        context,
        samples.len(),
        1,
        &mut workers,
        scratch,
        |i, _, out| {
            out[0] = kind.apply(samples[i], lambda)?;
            Ok(())
        },
    )?;
    output.copy_from_slice(scratch);
    Ok(())
}
/// Returns the profile log likelihood of `λ` under a normal model for the
/// transformed sample.
///
/// For `n` samples with transformed values `yᵢ = ψ(xᵢ; λ)`,
///
/// ```text
/// ℓ(λ) = −(n/2) ln σ̂²(λ) + Σᵢ ln |dψ/dx (xᵢ; λ)|,   σ̂² = (1/n) Σᵢ (yᵢ − ȳ)²,
/// ```
///
/// omitting the constant `−(n/2)(1 + ln 2π)`. The variance uses divisor `n`.
/// `ln σ̂²` is computed in extended range, so it is accurate even when `σ̂²`
/// underflows. The cost is O(n).
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two samples or the transformed
///   sample is constant.
/// - The errors of [`PowerTransform::apply`] and
///   [`PowerTransform::log_jacobian`].
///
/// # Examples
///
/// ```
/// use scioxide_stats::{PowerTransform, power_transform_llf};
///
/// // λ = 1 shifts the data by 1: σ̂² = 14/9 and the Jacobian term is zero.
/// let llf = power_transform_llf(PowerTransform::BoxCox, 1.0_f64, &[1.0, 2.0, 4.0])?;
/// assert!((llf + 1.5 * (14.0_f64 / 9.0).ln()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn power_transform_llf<S: RealScalar>(
    kind: PowerTransform,
    lambda: S,
    samples: &[S],
) -> Result<S, SciOxideError> {
    crate::error_helpers::input_finite(lambda)?;
    if samples.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("samples"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "profile likelihood needs at least two samples",
        ));
    }
    let first = kind.apply(samples[0], lambda)?;
    let mut state = crate::central::Central::singleton(first)?;
    let mut jac = crate::arithmetic::Compensated::<f64>::zero()
        .added(kind.log_jacobian(samples[0], lambda)?.try_to_f64_exact()?)?;
    for (index, &x) in samples.iter().enumerate().skip(1) {
        state = state.observed(kind.apply(x, lambda)?, index + 1, MomentOrder::Second)?;
        jac = jac.added(kind.log_jacobian(x, lambda)?.try_to_f64_exact()?)?;
    }
    if state.m2()?.is_zero() || state.m2()?.is_sign_negative() {
        return Err(invalid(
            scioxide_core::argument!("samples"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant or unresolved transformed variance",
        ));
    }
    // Query logarithmic variance before rounding: the variance itself need not
    // fit S, and rescaling nearly equal observations would discard differences.
    finite(number::<S>(
        jac.total()? - (samples.len() as f64 / 2.0) * state.m2()?.ln_abs()?,
    )?)
}

/// Least-squares line and correlation of a probability plot.
#[derive(Clone, Copy, Debug)]
pub struct ProbabilityPlotFit<S: RealScalar> {
    /// Slope of the line, in data units per unit of the theoretical quantile.
    pub slope: S,
    /// Intercept of the line, in data units.
    pub intercept: S,
    /// Pearson correlation of the plot, `None` if the observations are constant.
    pub correlation: Option<S>,
}
impl<S: RealScalar> ProbabilityPlotFit<S> {
    /// Computes the line and correlation from accumulated covariance moments.
    pub(crate) fn from_covariance(
        state: crate::OnlineCovariance<S>,
    ) -> Result<Self, SciOxideError> {
        let snapshot = state.snapshot();
        let (slope, intercept) = snapshot.least_squares_line()?;
        Ok(Self {
            slope: slope.round_to()?,
            intercept: intercept.round_to()?,
            correlation: snapshot.correlation()?,
        })
    }
}
/// Probability plot of a sample: sorted observations against theoretical
/// quantiles, with a least-squares line.
///
/// For `n` sorted observations `x₍₁₎ ≤ … ≤ x₍ₙ₎`, the theoretical quantiles are
/// `Q(mᵢ)` at Filliben's plotting positions
///
/// ```text
/// m₁ = 1 − 0.5^(1/n),   mₙ = 0.5^(1/n),   mᵢ = (i − 0.3175)/(n + 0.365) otherwise.
/// ```
///
/// # References
///
/// J. J. Filliben, "The probability plot correlation coefficient test for
/// normality", *Technometrics* 17 (1975), 111–117.
pub struct ProbabilityPlot<S: RealScalar> {
    theoretical: Vec<S>,
    ordered: Vec<S>,
    fit: ProbabilityPlotFit<S>,
}
impl<S: RealScalar> ProbabilityPlot<S> {
    /// Returns the theoretical quantiles `Q(mᵢ)`.
    pub fn theoretical(&self) -> &[S] {
        &self.theoretical
    }
    /// Returns the observations in increasing order.
    pub fn ordered(&self) -> &[S] {
        &self.ordered
    }
    /// Returns the least-squares line of the observations on the theoretical
    /// quantiles, and their correlation.
    pub const fn fit(&self) -> ProbabilityPlotFit<S> {
        self.fit
    }
    /// Returns the theoretical quantiles, the sorted observations and the fit.
    pub fn into_parts(self) -> (Vec<S>, Vec<S>, ProbabilityPlotFit<S>) {
        (self.theoretical, self.ordered, self.fit)
    }
}
fn sorted<S: RealScalar>(samples: &[S]) -> Result<Vec<S>, SciOxideError> {
    if samples.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("samples"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "probability plot needs at least two observations",
        ));
    }
    for &x in samples {
        crate::error_helpers::input_finite(x)?;
    }
    let mut values = scioxide_core::backend::try_copy_slice(
        samples,
        scioxide_core::quantity!("ordered_fit_samples"),
    )?;
    crate::ranking::sort_finite(&mut values);
    Ok(values)
}
fn position<S: RealScalar>(i: usize, n: usize) -> Result<S, SciOxideError> {
    // Filliben 1975 endpoint medians, and interior approximation.
    let n = crate::error_helpers::count::<S>(n)?;
    let tail = number::<S>(0.5)?.ln() / n;
    if i == 0 {
        Ok(-tail.exp_m1())
    } else if crate::error_helpers::count::<S>(i + 1)? == n {
        Ok(tail.exp())
    } else {
        Ok(
            (crate::error_helpers::count::<S>(i + 1)? - number::<S>(0.3175)?)
                / (n + number::<S>(0.365)?),
        )
    }
}
/// Computes the result of [`probplot_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`probplot_with_options`] for the fixed default settings.
pub fn probplot<S: RealScalar>(
    samples: &[S],
    model: &dyn FitModel<S>,
    parameters: &[S],
) -> Result<ProbabilityPlot<S>, SciOxideError> {
    probplot_with_options(samples, model, parameters, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("samples"),
        )
    })
}
/// Computes the probability plot of `samples` against a fitted distribution.
///
/// The theoretical quantiles come from [`FitModel::quantile`] at `parameters`.
/// The cost is one sort, O(n log n), and `n` quantile evaluations.
/// [`StatisticsOptions::max_bytes`] limits the two vectors of the plot.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two samples.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `ResourceLimit` if the plot exceeds the byte limit.
/// - The errors of validating `parameters` and of [`FitModel::quantile`], and an
///   error if a quantile is infinite.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{DistributionFitModel, StatisticsOptions, probplot_with_options};
///
/// // Symmetric data against symmetric normal quantiles lie on a line.
/// let samples = [1.0_f64, -1.0, 0.0];
/// let normal = DistributionFitModel::Normal;
/// let plot = probplot_with_options(&samples, &normal, &[0.0, 1.0], StatisticsOptions::new())?;
/// assert_eq!(plot.ordered(), &[-1.0, 0.0, 1.0]);
/// assert!((plot.fit().correlation.unwrap() - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn probplot_with_options<S: RealScalar>(
    samples: &[S],
    model: &dyn FitModel<S>,
    parameters: &[S],
    options: StatisticsOptions,
) -> Result<ProbabilityPlot<S>, SciOxideError> {
    check_plot_bytes::<S>(samples.len(), options.max_bytes())?;
    model.validate(parameters)?;
    let ordered = sorted(samples)?;
    let mut theoretical = scioxide_core::backend::try_with_capacity(
        samples.len(),
        scioxide_core::quantity!("plot_quantiles"),
    )?;
    let mut state = OnlineCovariance::new(MomentOrder::Second);
    for (i, &y) in ordered.iter().enumerate() {
        let x = finite(model.quantile(parameters, position(i, ordered.len())?)?)?;
        state.observe(x, y)?;
        theoretical.push(x);
    }
    Ok(ProbabilityPlot {
        theoretical,
        ordered,
        fit: ProbabilityPlotFit::from_covariance(state)?,
    })
}
fn check_plot_bytes<S: RealScalar>(n: usize, max: usize) -> Result<(), SciOxideError> {
    let bytes = n
        .checked_mul(2)
        .and_then(|x| x.checked_mul(core::mem::size_of::<S>()))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("probability_plot_bytes"))
        })?;
    crate::error_helpers::limit(bytes, max, scioxide_core::quantity!("transform_bytes"))?;
    Ok(())
}
/// Objective for the probability plot correlation coefficient (PPCC) method:
/// choose the parameters that make the probability plot most linear.
///
/// The objective is `−r(θ)`, where `r(θ)` is the correlation of the
/// [`ProbabilityPlot`] of the sample against the family at parameters `θ`. The
/// sample is sorted once. Minimize it with [`ppcc_max_scoped`].
///
/// # References
///
/// J. J. Filliben, "The probability plot correlation coefficient test for
/// normality", *Technometrics* 17 (1975), 111–117.
pub struct PreparedPpcc<'a, S: RealScalar> {
    ordered: Vec<S>,
    model: &'a dyn FitModel<S>,
    bounds: &'a Bounds<S>,
    integer: &'a [bool],
}
impl<'a, S: RealScalar> PreparedPpcc<'a, S> {
    /// Checks the sample and sorts it.
    ///
    /// `bounds` and `integer` have one entry per model parameter.
    /// [`StatisticsOptions::max_bytes`] limits the sorted copy of the sample.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `bounds` or `integer` has the wrong length.
    /// - `InvalidInput` if there are fewer than two samples.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `ResourceLimit` if the sorted copy exceeds the byte limit.
    pub fn new(
        samples: &[S],
        model: &'a dyn FitModel<S>,
        bounds: &'a Bounds<S>,
        integer: &'a [bool],
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        if model.parameter_count() != bounds.len() || integer.len() != bounds.len() {
            return Err(invalid(
                scioxide_core::argument!("bounds"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "PPCC parameter dimensions",
            ));
        }
        let bytes = samples
            .len()
            .checked_mul(core::mem::size_of::<S>())
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("ppcc_bytes")))?;
        crate::error_helpers::limit(
            bytes,
            options.max_bytes(),
            scioxide_core::quantity!("transform_bytes"),
        )?;
        Ok(Self {
            ordered: sorted(samples)?,
            model,
            bounds,
            integer,
        })
    }
}
impl<S: RealScalar> FitObjective<S> for PreparedPpcc<'_, S> {
    type Workspace = ();
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn dimension(&self) -> usize {
        self.bounds.len()
    }
    fn validate(&self, p: &[S], _workspace: &mut ()) -> Result<(), SciOxideError> {
        super::problem::validate_parameters(p, self.bounds, self.integer)?;
        self.model.validate(p)
    }
    fn value(&self, p: &[S], workspace: &mut ()) -> Result<S, SciOxideError> {
        self.validate(p, workspace)?;
        let mut state = OnlineCovariance::new(MomentOrder::Second);
        for (i, &y) in self.ordered.iter().enumerate() {
            state.observe(self.model.quantile(p, position(i, self.ordered.len())?)?, y)?;
        }
        Ok(-ProbabilityPlotFit::from_covariance(state)?
            .correlation
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("p"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "PPCC of constant data",
                )
            })?)
    }
}
/// Executes [`ppcc_max_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn ppcc_max_with_options<S: RealScalar>(
    problem: &PreparedPpcc<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
    options: StatisticsOptions,
) -> Result<FitResult<S>, SciOxideError> {
    ppcc_max_scoped(
        problem,
        initial,
        solver,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Executes [`ppcc_max_scoped`] with the [one-shot defaults](crate#one-shot-forms)
/// and sequential native capabilities, requiring successful termination.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors, and maps
/// every unsuccessful termination to an error under the one-shot contract.
pub fn ppcc_max<S: RealScalar>(
    problem: &PreparedPpcc<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
) -> Result<FitResult<S>, SciOxideError> {
    let result = ppcc_max_with_options(problem, initial, solver, Default::default())?;
    let report = result.report();
    if let Some(error) = report.termination().one_shot_error(
        scioxide_core::ConvergenceReason::NestedSolver,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts()),
    ) {
        return Err(error);
    }
    Ok(result)
}
/// Finds the parameters that maximize the probability plot correlation, using
/// the given solver.
///
/// Runs [`fit_scoped`] on `problem` from `initial`, without derivatives.
/// [`StatisticsOptions::max_bytes`] limits the parameter vectors of the fit.
///
/// # Errors
///
/// The errors of [`fit_scoped`]. The objective fails with `InvalidInput` if the
/// observations are constant.
pub fn ppcc_max_scoped<'scope, 'capability, S: RealScalar>(
    problem: &PreparedPpcc<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
    options: StatisticsOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<FitResult<S>, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    scope.into().run(
        &mut state,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "ppcc_max", 1),
            0,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            fit_scoped(
                FitRequest {
                    objective: problem,
                    derivatives: None,
                    bounds: problem.bounds,
                    integrality: problem.integer,
                    initial,
                },
                solver,
                options,
                context,
            )
        },
    )
}
/// Selects the criterion for choosing the power transformation parameter `λ`.
#[derive(Clone, Copy, Debug)]
pub enum PowerFitMethod {
    /// Maximizes the profile log likelihood [`power_transform_llf`].
    Likelihood,
    /// Maximizes the correlation between the sorted transformed sample and the
    /// standard normal quantiles at Filliben's plotting positions.
    Pearson,
}
/// Objective for choosing the parameter `λ` of a power transformation.
///
/// The objective is the negated criterion of [`PowerFitMethod`], a function of
/// the single parameter `λ`. Minimize it with [`fit_scoped`], or use
/// [`power_transform_normmax_scoped`].
pub struct PreparedPowerFit<'a, S: RealScalar> {
    samples: &'a [S],
    kind: PowerTransform,
    method: PowerFitMethod,
    ordered: Vec<S>,
    normal: Vec<S>,
}
impl<'a, S: RealScalar> PreparedPowerFit<'a, S> {
    /// Checks the sample and, for [`PowerFitMethod::Pearson`], sorts it and
    /// computes the normal quantiles.
    ///
    /// [`StatisticsOptions::max_bytes`] limits that storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two samples, or a sample is not
    ///   positive for Box–Cox.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    pub fn new(
        samples: &'a [S],
        kind: PowerTransform,
        method: PowerFitMethod,
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        if samples.len() < 2 {
            return Err(invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "power fit_scoped needs at least two samples",
            ));
        }
        for &x in samples {
            crate::error_helpers::input_finite(x)?;
            if kind == PowerTransform::BoxCox && x <= S::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("samples"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "Box-Cox requires positive samples",
                ));
            }
        }
        let (ordered, normal) = if matches!(method, PowerFitMethod::Pearson) {
            let plot = probplot_with_options(
                samples,
                &super::DistributionFitModel::Normal,
                &[S::ZERO, S::ONE],
                options,
            )?;
            (plot.ordered, plot.theoretical)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            samples,
            kind,
            method,
            ordered,
            normal,
        })
    }
}
impl<S: RealScalar> FitObjective<S> for PreparedPowerFit<'_, S> {
    type Workspace = ();
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn dimension(&self) -> usize {
        1
    }
    fn validate(&self, p: &[S], _workspace: &mut ()) -> Result<(), SciOxideError> {
        if p.len() != 1 {
            return Err(invalid(
                scioxide_core::argument!("p"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "power parameter dimension",
            ));
        }
        finite(p[0])?;
        Ok(())
    }
    fn value(&self, p: &[S], workspace: &mut ()) -> Result<S, SciOxideError> {
        self.validate(p, workspace)?;
        match self.method {
            PowerFitMethod::Likelihood => Ok(-power_transform_llf(self.kind, p[0], self.samples)?),
            PowerFitMethod::Pearson => {
                let mut state = OnlineCovariance::new(MomentOrder::Second);
                for (&x, &z) in self.ordered.iter().zip(&self.normal) {
                    state.observe(z, self.kind.apply(x, p[0])?)?;
                }
                Ok(-ProbabilityPlotFit::from_covariance(state)?
                    .correlation
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("p"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "power correlation of constant data",
                        )
                    })?)
            }
        }
    }
}
/// Executes [`power_transform_normmax_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn power_transform_normmax_with_options<S: RealScalar>(
    kind: PowerTransform,
    samples: &[S],
    method: PowerFitMethod,
    bounds: &Bounds<S>,
    initial: S,
    solver: &mut dyn FitSolver<S>,
    options: StatisticsOptions,
) -> Result<FitResult<S>, SciOxideError> {
    power_transform_normmax_scoped(
        kind,
        samples,
        method,
        bounds,
        initial,
        solver,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Executes [`power_transform_normmax_scoped`] with the
/// [one-shot defaults](crate#one-shot-forms) and sequential native capabilities,
/// requiring successful termination.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors, and maps
/// every unsuccessful termination to an error under the one-shot contract.
pub fn power_transform_normmax<S: RealScalar>(
    kind: PowerTransform,
    samples: &[S],
    method: PowerFitMethod,
    bounds: &Bounds<S>,
    initial: S,
    solver: &mut dyn FitSolver<S>,
) -> Result<FitResult<S>, SciOxideError> {
    let result = power_transform_normmax_with_options(
        kind,
        samples,
        method,
        bounds,
        initial,
        solver,
        Default::default(),
    )?;
    let report = result.report();
    if let Some(error) = report.termination().one_shot_error(
        scioxide_core::ConvergenceReason::NestedSolver,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts()),
    ) {
        return Err(error);
    }
    Ok(result)
}
/// Chooses the parameter `λ` of a power transformation with the given solver.
///
/// Prepares a [`PreparedPowerFit`] and runs [`fit_scoped`] from `initial` within
/// `bounds`, without derivatives. [`StatisticsOptions::max_bytes`] limits the
/// preparation and the parameter vectors together.
///
/// # Errors
///
/// - `ResourceLimit` if the memory needed exceeds the byte limit.
/// - The errors of [`PreparedPowerFit::new`] and [`fit_scoped`].
#[allow(clippy::too_many_arguments)]
pub fn power_transform_normmax_scoped<'scope, 'capability, S: RealScalar>(
    kind: PowerTransform,
    samples: &[S],
    method: PowerFitMethod,
    bounds: &Bounds<S>,
    initial: S,
    solver: &mut dyn FitSolver<S>,
    options: StatisticsOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<FitResult<S>, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    scope.into().run(
        &mut state,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "power_transform_normmax", 1),
            0,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            let solver_bytes = core::mem::size_of::<S>().checked_mul(2).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("power_solve_bytes"))
            })?;
            let preparation_limit =
                options
                    .max_bytes()
                    .checked_sub(solver_bytes)
                    .ok_or_else(|| {
                        SciOxideError::resource_limit(
                            scioxide_core::quantity!("power_solve_bytes"),
                            solver_bytes,
                            options.max_bytes(),
                        )
                    })?;
            let problem = PreparedPowerFit::new(
                samples,
                kind,
                method,
                options.with_max_bytes(preparation_limit),
            )?;
            fit_scoped(
                FitRequest {
                    objective: &problem,
                    derivatives: None,
                    bounds,
                    integrality: &[false],
                    initial: &[initial],
                },
                solver,
                options.with_max_bytes(solver_bytes),
                context,
            )
        },
    )
}
