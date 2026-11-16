use super::*;
use crate::error_helpers::{input_finite, limit_bytes};
use crate::{
    BatchView, FitObjective, FitReport, FitRequest, FitSolver, FitWorkspace, MissingPolicy,
    NonFinitePolicy, StatisticsOptions,
};
use alloc::vec::Vec;
use scioxide_core::backend::checked_product_sum;
use scioxide_core::{
    Bounds, ExecutionContext, ExecutionPolicy, RangeAccumulator, ResidualTolerance,
};
use scioxide_linalg::{LinalgWorkspace, LuDecomposition, Matrix};
use scioxide_signal::{IirDomain, TransferFunction};

/// Order `(p, d, q)` of a nonseasonal ARIMA model.
///
/// ARMA(p, q) is the case `d = 0`, and AR(p) and MA(q) are special cases of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArimaOrder {
    /// Number of autoregressive coefficients.
    pub p: usize,
    /// Number of differences.
    pub d: usize,
    /// Number of moving-average coefficients.
    pub q: usize,
}
impl ArimaOrder {
    fn state_len(self) -> Result<usize, SciOxideError> {
        add(add(self.p, self.q)?, self.d)
    }
}
/// Deterministic trend `μₜ` of an ARIMA model, at the zero-based sample index
/// `t` of the original series.
///
/// A trend that `d` differences would remove is rejected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArimaTrend {
    /// No trend, `μₜ = 0`.
    #[default]
    None,
    /// Constant level `μₜ = c`. Requires `d = 0`.
    Constant,
    /// Linear trend `μₜ = b t`. Requires `d ≤ 1`; with `d = 1`, `b` is the drift.
    Linear,
    /// Level and slope `μₜ = c + b t`. Requires `d = 0`.
    ConstantAndLinear,
}
impl ArimaTrend {
    fn len(self) -> usize {
        match self {
            Self::None => 0,
            Self::ConstantAndLinear => 2,
            _ => 1,
        }
    }
    fn check(self, d: usize) -> Result<(), SciOxideError> {
        if (d > 0 && matches!(self, Self::Constant | Self::ConstantAndLinear))
            || (d > 1 && self == Self::Linear)
        {
            Err(invalid(
                scioxide_core::argument!("trend"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "trend is annihilated by differencing",
            ))
        } else {
            Ok(())
        }
    }
    fn value<S: RealScalar>(self, parameters: &[S], t: usize) -> Result<S, SciOxideError> {
        match self {
            Self::None => Ok(S::ZERO),
            Self::Constant => Ok(parameters[0]),
            Self::Linear => finite(parameters[0] * number(t)?),
            Self::ConstantAndLinear => {
                checked_product_sum([(parameters[0], S::ONE), (parameters[1], number(t)?)])
            }
        }
    }
}
/// One-step prediction and innovation of an ARIMA model at one time point.
///
/// All three fields are `None` for the first `d` observations, on which the
/// likelihood is conditioned.
#[derive(Clone, Copy, Debug)]
pub struct ArimaInnovation<S: RealScalar> {
    /// Prediction `x̂ₜ` of `xₜ` from the earlier observations, in the units of
    /// the series and including the trend.
    pub prediction: Option<S>,
    /// Prediction error variance `fₜ`, including the innovation variance `σ²`.
    pub variance: Option<S>,
    /// Innovation `vₜ = xₜ − x̂ₜ`. `None` also at gaps.
    pub residual: Option<S>,
}
impl<S: RealScalar> Default for ArimaInnovation<S> {
    fn default() -> Self {
        Self {
            prediction: None,
            variance: None,
            residual: None,
        }
    }
}
/// Gaussian likelihood of an ARIMA model, from
/// [`PreparedArimaFit::evaluate_into`].
#[derive(Clone, Copy, Debug)]
pub struct ArimaLikelihood<S: RealScalar> {
    /// Negative log likelihood `½ Σ (vₜ²/fₜ + ln fₜ + ln 2π)` over the
    /// innovations, including the constant term.
    pub negative_log_likelihood: S,
    /// Number of innovations in the likelihood: the observations after the
    /// first `d`, without gaps.
    pub observations: usize,
    /// Relative residual of the linear equations solved for the stationary
    /// autocovariances of the ARMA part.
    pub covariance_residual: S,
}
/// Forecast of an ARIMA model at one future time point.
#[derive(Clone, Copy, Debug)]
pub struct ArimaForecast<S: RealScalar> {
    /// Conditional mean, in the units of the series and including the trend.
    pub mean: S,
    /// Conditional variance of the forecast error. It treats the parameters as
    /// known, so it does not include their estimation error.
    pub variance: S,
}
/// Observations: a slice or one record of a view.
// The view keeps its layout inline; boxing it would allocate per series.
#[derive(Clone, Copy)]
#[allow(clippy::large_enum_variant)]
enum Series<'a, S: RealScalar> {
    Slice(&'a [S]),
    Lane {
        view: BatchView<'a, S>,
        origin: usize,
    },
}
impl<S: RealScalar> Series<'_, S> {
    #[inline]
    fn at(self, t: usize) -> S {
        match self {
            Self::Slice(values) => values[t],
            Self::Lane { view, origin } => view.value_at_origin(origin, t, 0),
        }
    }
}

/// Gaussian maximum likelihood problem of an ARIMA(p, d, q) model for one real
/// time series with gaps.
///
/// The model for the series `xₜ` with trend `μₜ` ([`ArimaTrend`]) is
///
/// ```text
/// φ(B) (1 − B)ᵈ (xₜ − μₜ) = θ(B) εₜ,   εₜ ~ N(0, σ²) independent,
/// φ(z) = 1 − φ₁ z − … − φₚ zᵖ,   θ(z) = 1 + θ₁ z + … + θ_q z^q,
/// ```
///
/// where `B` is the backshift operator. The parameter vector is
/// `[φ₁, …, φₚ, θ₁, …, θ_q, trend, σ²]`, where `trend` holds the coefficients
/// of [`ArimaTrend`] (`c`, `b` or both, in that order). The AR part must be
/// stationary and the MA part invertible. The objective rejects parameters
/// whose AR or MA polynomial has a reflection coefficient within `64 ε` of
/// modulus 1, and parameters with `σ² ≤ 0`.
///
/// The likelihood is exact for the stationary ARMA part and conditional on the
/// first `d` observations, which must be present. A Kalman filter on a state
/// space form evaluates it. It starts from the stationary covariance, found
/// from a linear system of order `p + 1`. Each evaluation costs O(n m³) for
/// `n` time points and state dimension `m = p + d + q`. Gaps are skipped by
/// the filter and keep their place on the time grid.
///
/// The missing-value and non-finite policies of the [`StatisticsOptions`] passed
/// at construction decide which observations are gaps. Its memory limit applies
/// to the workspaces, models and fits made from this problem. The filter runs
/// sequentially.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_stats::{ArimaInnovation, ArimaOrder, ArimaTrend, PreparedArimaFit, StatisticsOptions};
///
/// // AR(1) with φ₁ = 0.5 and σ² = 1: the first prediction is the stationary mean 0.
/// let values = [0.5_f64, -0.25, 0.75, 0.0, -0.5];
/// let order = ArimaOrder { p: 1, d: 0, q: 0 };
/// let options = StatisticsOptions::new();
/// let problem = PreparedArimaFit::new(&values, None, order, ArimaTrend::None, options)?;
/// let mut workspace = problem.workspace()?;
/// let mut innovations = [ArimaInnovation::default(); 5];
/// let context = ExecutionContext::sequential();
/// let likelihood = problem.evaluate_into(&[0.5, 1.0], &mut innovations, &mut workspace, context)?;
/// assert_eq!(likelihood.observations, 5);
/// assert_eq!(innovations[0].prediction, Some(0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. H. Jones, "Maximum likelihood fitting of ARMA models to time series with
/// missing observations", *Technometrics* 22 (1980), 389–395.
///
/// J. Durbin and S. J. Koopman, *Time Series Analysis by State Space Methods*,
/// 2nd ed., Oxford University Press, 2012.
#[derive(Clone)]
pub struct PreparedArimaFit<'a, S: RealScalar> {
    series: Series<'a, S>,
    samples: usize,
    validity: Option<&'a [bool]>,
    order: ArimaOrder,
    trend: ArimaTrend,
    options: StatisticsOptions,
}
impl<'a, S: RealScalar> PreparedArimaFit<'a, S>
where
    scioxide_core::Complex<S>: NumericScalar<Real = S>,
{
    /// Checks the series `values` and creates the problem.
    ///
    /// `validity`, if given, marks the present observations. The first `d`
    /// observations condition the likelihood and must be present and finite.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `trend` is not allowed for `order.d`.
    /// - `InvalidInput` if `values` has at most `d` elements, or `validity`
    ///   differs from it in length.
    /// - `InvalidInput` if one of the first `d` observations is missing or not
    ///   finite, or no observation after them is present.
    /// - `InvalidInput` if an observation is missing under
    ///   [`MissingPolicy::Reject`], and `NonFinite` if one is NaN or infinite
    ///   under [`NonFinitePolicy::Reject`].
    /// - `SizeOverflow` if the order is too large.
    pub fn new(
        values: &'a [S],
        validity: Option<&'a [bool]>,
        order: ArimaOrder,
        trend: ArimaTrend,
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            Series::Slice(values),
            values.len(),
            validity,
            order,
            trend,
            options,
        )
    }
    /// Creates the problem for record `lane` of `view`, without copying.
    ///
    /// The sample axis of `view` is time. Records are numbered as in
    /// [Data, weights and missing values](crate#data-weights-and-missing-values),
    /// and the validity mask of `view` marks the present observations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `view` has vector observations or `lane` is not less
    ///   than the number of records.
    /// - The errors of [`Self::new`].
    pub fn from_view(
        view: BatchView<'a, S>,
        lane: usize,
        order: ArimaOrder,
        trend: ArimaTrend,
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        if view.dimensions() != 1 || lane >= view.batches() {
            return Err(invalid(
                scioxide_core::argument!("lane"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "ARIMA lane must select a record of a scalar view",
            ));
        }
        let samples = view.samples();
        Self::prepare(
            Series::Lane {
                view,
                origin: view.batch_origin(lane),
            },
            samples,
            view.validity
                .map(|mask| &mask[lane * samples..(lane + 1) * samples]),
            order,
            trend,
            options,
        )
    }
    fn prepare(
        series: Series<'a, S>,
        samples: usize,
        validity: Option<&'a [bool]>,
        order: ArimaOrder,
        trend: ArimaTrend,
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        order.state_len()?;
        add(add(order.p, order.q)?, add(trend.len(), 1)?)?;
        trend.check(order.d)?;
        if samples <= order.d || validity.is_some_and(|m| m.len() != samples) {
            return Err(invalid(
                scioxide_core::argument!("validity"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ARIMA sample or mask shape",
            ));
        }
        let mut observed = 0;
        for i in 0..samples {
            let x = series.at(i);
            let present = validity.is_none_or(|m| m[i]);
            if present && !x.is_finite() && options.non_finite_policy() == NonFinitePolicy::Reject {
                input_finite(x)?;
            }
            if present && x.is_finite() {
                observed += usize::from(i >= order.d);
            } else if i < order.d {
                return Err(invalid(
                    scioxide_core::argument!("validity"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "ARIMA anchors must be observed",
                ));
            } else if !present && options.missing_policy() == MissingPolicy::Reject {
                return Err(invalid(
                    scioxide_core::argument!("validity"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing observation",
                ));
            }
        }
        if observed == 0 {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no observed innovations",
            ));
        }
        Ok(Self {
            series,
            samples,
            validity,
            order,
            trend,
            options,
        })
    }
    /// Returns the model order.
    pub fn order(&self) -> ArimaOrder {
        self.order
    }
    /// Returns the length of the series, including gaps.
    pub fn sample_count(&self) -> usize {
        self.samples
    }
    /// Returns the options passed at construction.
    pub fn options(&self) -> StatisticsOptions {
        self.options
    }
    /// Returns whether position `t` holds an observation rather than a gap.
    fn observed(&self, t: usize) -> bool {
        self.validity.is_none_or(|m| m[t]) && self.series.at(t).is_finite()
    }
    /// Creates a workspace for [`Self::evaluate_into`] and [`Self::model`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the memory limit of
    ///   [`Self::options`].
    pub fn workspace(&self) -> Result<ArimaWorkspace<S>, SciOxideError> {
        ArimaWorkspace::reserve(self.order, self.samples, self.options.max_bytes())
    }
    /// Computes the negative log likelihood at `parameters` and writes the
    /// one-step predictions and innovations of every time point to `output`.
    ///
    /// `output` has one element per time point, gaps included. The call does
    /// not count toward any solve budget.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len()` differs from
    ///   [`Self::sample_count`].
    /// - `InvalidInput` if `parameters` does not have `p + q + 1` values plus
    ///   one per trend coefficient, or `workspace` was made for a different
    ///   order or a shorter series.
    /// - An error if a parameter is NaN or infinite.
    /// - `InvalidInput` if `σ² ≤ 0`, the AR part is not stationary or the MA
    ///   part is not invertible.
    /// - `InvalidInput` if the stationary covariance cannot be computed
    ///   accurately, and `Singular` if its linear system is singular.
    /// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
    pub fn evaluate_into(
        &self,
        parameters: &[S],
        output: &mut [ArimaInnovation<S>],
        workspace: &mut ArimaWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<ArimaLikelihood<S>, SciOxideError> {
        if output.len() != self.samples {
            return Err(SciOxideError::dimension_mismatch(
                [self.samples],
                [output.len()],
            ));
        }
        let report = self.evaluate(parameters, workspace, context)?;
        context.check()?;
        output.copy_from_slice(&workspace.innovations[..output.len()]);
        Ok(report)
    }
    fn evaluate(
        &self,
        parameters: &[S],
        w: &mut ArimaWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<ArimaLikelihood<S>, SciOxideError> {
        self.validate(parameters, w)?;
        context.check()?;
        let covariance_residual = initialize(
            self.order,
            self.trend,
            parameters,
            |i| self.series.at(i),
            w,
            context,
        )?;
        let trend_start = self.order.p + self.order.q;
        let trend_parameters = &parameters[trend_start..trend_start + self.trend.len()];
        let variance = parameters[parameters.len() - 1];
        w.innovations[..self.order.d].fill(ArimaInnovation::default());
        let mut sum = RangeAccumulator::new();
        let mut observations = 0;
        for t in self.order.d..self.samples {
            context.check()?;
            let trend = self.trend.value(trend_parameters, t)?;
            let observed = if self.observed(t) {
                Some(finite(self.series.at(t) - trend)?)
            } else {
                None
            };
            let (prediction, f, residual) = step(w, variance, observed)?;
            w.innovations[t] = ArimaInnovation {
                prediction: Some(finite(prediction + trend)?),
                variance: Some(f),
                residual,
            };
            if let Some(v) = residual {
                let standardized = finite(v / f.sqrt())?;
                sum.add_product(standardized, standardized)?;
                sum.add(finite(f.ln() + S::LN_2 + S::PI.ln())?)?;
                observations += 1;
            }
        }
        let nll = finite(sum.finish()? / (S::TWO))?;
        Ok(ArimaLikelihood {
            negative_log_likelihood: nll,
            observations,
            covariance_residual,
        })
    }
    /// Returns the model with the given `parameters`, ready for forecasting.
    ///
    /// The model holds the filter state after the last time point of the
    /// series.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_into`], apart from the output length.
    /// - `ResourceLimit` if the model exceeds the memory limit of
    ///   [`Self::options`].
    pub fn model(
        &self,
        parameters: &[S],
        workspace: &mut ArimaWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<ArimaModel<S>, SciOxideError> {
        let max_bytes = self.options.max_bytes();
        limit_bytes(
            ArimaModel::<S>::required_bytes(self.order, self.trend)?,
            max_bytes,
        )?;
        self.evaluate(parameters, workspace, context)?;
        ArimaModel::from_state(self, parameters, workspace, max_bytes)
    }
    /// Estimates the parameters by maximum likelihood with `solver`, starting
    /// from `initial` within `bounds`.
    ///
    /// The solver minimizes the negative log likelihood. No analytic
    /// derivatives are supplied.
    /// After it stops, the likelihood is evaluated once more at its result, and
    /// the model is built from that evaluation. The memory limit of
    /// [`Self::options`] covers the workspace, parameter vectors and model, but
    /// not the solver's own storage.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the memory limit of
    ///   [`Self::options`].
    /// - Errors from `solver`, and from checking `initial`, are returned
    ///   unchanged.
    pub fn fit_scoped<'scope, 'capability>(
        &self,
        initial: &[S],
        bounds: &Bounds<S>,
        solver: &mut dyn FitSolver<S, ArimaWorkspace<S>>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ArimaFitReport<S>, SciOxideError> {
        let mut state = scioxide_core::SolveState::new();
        scope.into().run(
            &mut state,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("stats", "arima", 1),
                0,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |context| {
                let d = self.dimension();
                let model_bytes = ArimaModel::<S>::required_bytes(self.order, self.trend)?;
                let objective_bytes = self.workspace_bytes()?;
                let candidate_bytes = mul(d, core::mem::size_of::<S>())?;
                let bytes = add(
                    add(objective_bytes, mul(2, candidate_bytes)?)?,
                    add(model_bytes, d)?,
                )?;
                limit_bytes(bytes, self.options.max_bytes())?;
                let mut workspace = FitWorkspace::for_objective(
                    self,
                    crate::StatisticsOptions::new()
                        .with_max_bytes(add(objective_bytes, candidate_bytes)?),
                )?;
                let mut output =
                    try_filled(d, S::ZERO, scioxide_core::quantity!("arima_parameters"))?;
                let integer =
                    try_filled(d, false, scioxide_core::quantity!("arima_integer_flags"))?;
                let report = crate::fit_into(
                    FitRequest {
                        objective: self,
                        derivatives: None,
                        bounds,
                        integrality: &integer,
                        initial,
                    },
                    solver,
                    &mut output,
                    &mut workspace,
                    context,
                )?;
                let model = if report.objective().is_some() {
                    Some(ArimaModel::from_state(
                        self,
                        &output,
                        &workspace.objective,
                        model_bytes,
                    )?)
                } else {
                    None
                };
                Ok(ArimaFitReport { report, model })
            },
        )
    }
    /// Executes [`Self::fit_scoped`] with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the scoped form's input, numerical and allocation errors.
    pub fn fit(
        &self,
        initial: &[S],
        bounds: &Bounds<S>,
        solver: &mut dyn FitSolver<S, ArimaWorkspace<S>>,
    ) -> Result<ArimaFitReport<S>, SciOxideError> {
        self.fit_scoped(
            initial,
            bounds,
            solver,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}
impl<S: RealScalar> FitObjective<S> for PreparedArimaFit<'_, S>
where
    scioxide_core::Complex<S>: NumericScalar<Real = S>,
{
    type Workspace = ArimaWorkspace<S>;
    fn dimension(&self) -> usize {
        self.order.p + self.order.q + self.trend.len() + 1
    }
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        ArimaWorkspace::<S>::required_bytes(self.order, self.samples)
    }
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError> {
        ArimaWorkspace::reserve(self.order, self.samples, self.workspace_bytes()?)
    }
    fn validate(&self, parameters: &[S], w: &mut ArimaWorkspace<S>) -> Result<(), SciOxideError> {
        if parameters.len() != self.dimension()
            || w.order != self.order
            || w.innovations.len() < self.samples
        {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ARIMA parameter/workspace shape",
            ));
        }
        for &x in parameters {
            finite(x)?;
        }
        if parameters[parameters.len() - 1] <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "innovation variance must be positive",
            ));
        }
        stable(
            &parameters[..self.order.p],
            false,
            &mut w.reflection,
            &mut w.previous,
        )?;
        stable(
            &parameters[self.order.p..self.order.p + self.order.q],
            true,
            &mut w.reflection,
            &mut w.previous,
        )?;
        Ok(())
    }
    fn value(
        &self,
        parameters: &[S],
        workspace: &mut ArimaWorkspace<S>,
    ) -> Result<S, SciOxideError> {
        Ok(self
            .evaluate(parameters, workspace, ExecutionContext::sequential())?
            .negative_log_likelihood)
    }
}
/// Result of [`PreparedArimaFit::fit`]: the solver report and the fitted model.
///
/// The fields are read through accessors:
///
/// ```compile_fail
/// fn forge<S: scioxide_core::RealScalar>(fit: &mut scioxide_stats::ArimaFitReport<S>) {
///     fit.model = None;
/// }
/// ```
pub struct ArimaFitReport<S: RealScalar> {
    report: FitReport<S>,
    model: Option<ArimaModel<S>>,
}
impl<S: RealScalar> ArimaFitReport<S> {
    /// Returns the solver report with the termination status and the final
    /// negative log likelihood.
    pub fn report(&self) -> &FitReport<S> {
        &self.report
    }
    /// Returns the fitted model. `None` when the fit stopped before the final
    /// likelihood evaluation, for example on cancellation or an exhausted
    /// budget.
    pub fn model(&self) -> Option<&ArimaModel<S>> {
        self.model.as_ref()
    }
    /// Returns the fitted model by value; see [`Self::model`].
    pub fn into_model(self) -> Option<ArimaModel<S>> {
        self.model
    }
}
/// Workspace for likelihood evaluation and forecasting, from
/// [`PreparedArimaFit::workspace`] or [`ArimaModel::workspace`].
///
/// Evaluations do not depend on earlier contents of the workspace.
pub struct ArimaWorkspace<S: RealScalar> {
    order: ArimaOrder,
    reflection: Vec<S>,
    previous: Vec<S>,
    psi: Vec<S>,
    gamma: Vec<S>,
    rhs: Vec<S>,
    equations: Matrix<S>,
    factor: LuDecomposition<S>,
    lu: LinalgWorkspace<'static, S>,
    transition: Vec<S>,
    observation: Vec<S>,
    noise: Vec<S>,
    mean: Vec<S>,
    covariance: Vec<S>,
    gain: Vec<S>,
    prediction: Vec<S>,
    adjusted: Vec<S>,
    product: Vec<S>,
    next: Vec<S>,
    innovations: Vec<ArimaInnovation<S>>,
    forecasts: Vec<ArimaForecast<S>>,
}
impl<S: RealScalar> ArimaWorkspace<S> {
    /// Returns an upper bound on the bytes of a workspace for `order` and
    /// `records` time points, without allocator overhead.
    ///
    /// With state dimension `m = p + d + q` and `s = size_of::<S>() +
    /// size_of::<usize>()`, the bound is
    /// `128 (m² + (p + 1)² + m + q + 1) s` plus the storage of one
    /// [`ArimaInnovation`] and one [`ArimaForecast`] per time point.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `records` is zero.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(order: ArimaOrder, records: usize) -> Result<usize, SciOxideError> {
        if records == 0 {
            return Err(invalid(
                scioxide_core::argument!("records"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty ARIMA workspace",
            ));
        }
        let m = order.state_len()?;
        let p = add(order.p, 1)?;
        let base = add(add(mul(m, m)?, mul(p, p)?)?, add(add(m, order.q)?, 1)?)?;
        add(
            mul(
                mul(128, base)?,
                add(core::mem::size_of::<S>(), core::mem::size_of::<usize>())?,
            )?,
            mul(
                records,
                add(
                    core::mem::size_of::<ArimaInnovation<S>>(),
                    core::mem::size_of::<ArimaForecast<S>>(),
                )?,
            )?,
        )
    }
    /// Allocates all storage, including the LU factorization buffers.
    fn reserve(order: ArimaOrder, records: usize, max_bytes: usize) -> Result<Self, SciOxideError> {
        limit_bytes(Self::required_bytes(order, records)?, max_bytes)?;
        let m = order.state_len()?;
        let size = mul(m, m)?;
        let p = add(order.p, 1)?;
        let equations = Matrix::identity(p)?;
        let mut factor = LuDecomposition::empty();
        let mut lu = LinalgWorkspace::new();
        LuDecomposition::factor_into(
            &equations,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut lu,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        lu.try_reserve(
            &scioxide_linalg::LuDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LuWorkspaceRequest::solve(
                    p,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        Ok(Self {
            order,
            reflection: try_filled(
                order.p.max(order.q),
                S::ZERO,
                scioxide_core::quantity!("arima_reflection"),
            )?,
            previous: try_filled(
                order.p.max(order.q),
                S::ZERO,
                scioxide_core::quantity!("arima_previous"),
            )?,
            psi: try_filled(
                add(order.q, 1)?,
                S::ZERO,
                scioxide_core::quantity!("arima_psi"),
            )?,
            gamma: try_filled(p, S::ZERO, scioxide_core::quantity!("arima_gamma"))?,
            rhs: try_filled(p, S::ZERO, scioxide_core::quantity!("arima_rhs"))?,
            equations,
            factor,
            lu,
            transition: try_filled(size, S::ZERO, scioxide_core::quantity!("arima_transition"))?,
            observation: try_filled(m, S::ZERO, scioxide_core::quantity!("arima_observation"))?,
            noise: try_filled(m, S::ZERO, scioxide_core::quantity!("arima_noise"))?,
            mean: try_filled(m, S::ZERO, scioxide_core::quantity!("arima_mean"))?,
            covariance: try_filled(size, S::ZERO, scioxide_core::quantity!("arima_covariance"))?,
            gain: try_filled(m, S::ZERO, scioxide_core::quantity!("arima_gain"))?,
            prediction: try_filled(m, S::ZERO, scioxide_core::quantity!("arima_prediction"))?,
            adjusted: try_filled(size, S::ZERO, scioxide_core::quantity!("arima_adjusted"))?,
            product: try_filled(size, S::ZERO, scioxide_core::quantity!("arima_product"))?,
            next: try_filled(size, S::ZERO, scioxide_core::quantity!("arima_next"))?,
            innovations: try_filled(
                records,
                ArimaInnovation::default(),
                scioxide_core::quantity!("arima_innovations"),
            )?,
            forecasts: try_filled(
                records,
                ArimaForecast {
                    mean: S::ZERO,
                    variance: S::ZERO,
                },
                scioxide_core::quantity!("arima_forecasts"),
            )?,
        })
    }
}
fn stable<S: RealScalar>(
    a: &[S],
    ma: bool,
    work: &mut [S],
    previous: &mut [S],
) -> Result<(), SciOxideError> {
    for (out, &x) in work.iter_mut().zip(a) {
        *out = if ma { -x } else { x };
    }
    let threshold = S::EPSILON * number::<S>(64)?;
    for k in (1..=a.len()).rev() {
        let reflection = work[k - 1];
        if S::ONE - reflection.abs() <= threshold {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nonstationary, noninvertible or unresolved unit-root boundary",
            ));
        }
        let divisor = (S::ONE - reflection) * (S::ONE + reflection);
        previous[..k].copy_from_slice(&work[..k]);
        for j in 0..k - 1 {
            work[j] = finite((previous[j] + reflection * previous[k - 2 - j]) / divisor)?;
        }
    }
    Ok(())
}
fn initialize<S: RealScalar>(
    order: ArimaOrder,
    trend: ArimaTrend,
    parameters: &[S],
    values: impl Fn(usize) -> S,
    w: &mut ArimaWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<S, SciOxideError> {
    let ArimaOrder { p, d, q } = order;
    let m = order.state_len()?;
    let ar = &parameters[..p];
    let ma = &parameters[p..p + q];
    let variance = parameters[parameters.len() - 1];
    w.psi.fill(S::ZERO);
    w.psi[0] = S::ONE;
    for k in 1..=q {
        w.psi[k] = checked_product_sum(
            core::iter::once((ma[k - 1], S::ONE))
                .chain((1..=p.min(k)).map(|i| (ar[i - 1], w.psi[k - i]))),
        )?;
    }
    w.equations.as_mut_slice().fill(S::ZERO);
    for k in 0..=p {
        w.equations.view_mut().set(k, k, S::ONE)?;
        for i in 1..=p {
            let j = k.abs_diff(i);
            let v = w.equations.view().get(k, j)?;
            w.equations.view_mut().set(k, j, finite(v - ar[i - 1])?)?;
        }
        w.rhs[k] = if k <= q {
            checked_product_sum(
                (k..=q).map(|j| (if j == 0 { S::ONE } else { ma[j - 1] }, w.psi[j - k])),
            )?
        } else {
            S::ZERO
        };
    }
    LuDecomposition::factor_into(
        &w.equations,
        &scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential),
        &mut w.factor,
        &mut w.lu,
        context,
    )?;
    w.factor
        .solve_into(
            scioxide_linalg::VectorView::from_slice(&w.rhs),
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            scioxide_linalg::VectorViewMut::from_slice(&mut w.gamma),
            &mut w.lu,
            context,
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })?;
    let mut residual = S::ZERO;
    let mut scale = S::ZERO;
    for k in 0..=p {
        let a = checked_product_sum(
            (0..=p).map(|j| (w.equations.as_slice()[k * (p + 1) + j], w.gamma[j])),
        )?;
        residual = residual.max((a - w.rhs[k]).abs());
        scale = scale.max(w.rhs[k].abs()).max(a.abs());
    }
    if !ResidualTolerance::<S>::default().is_satisfied(residual, scale)? || w.gamma[0] <= S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("parameters"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "stationary covariance initialization failed verification",
        ));
    }
    let relative = if scale > S::ZERO {
        finite(residual / scale)?
    } else {
        residual
    };
    w.mean.fill(S::ZERO);
    w.covariance.fill(S::ZERO);
    w.transition.fill(S::ZERO);
    w.observation.fill(S::ZERO);
    w.noise.fill(S::ZERO);
    w.observation[..p].copy_from_slice(ar);
    w.observation[p..p + q].copy_from_slice(ma);
    // (1-B)^d z[t] = z[t] - sum_j difference[j-1]*z[t-j].
    let mut binomial = S::ONE;
    for j in 1..=d {
        binomial = finite(binomial * number::<S>(d - j + 1)? / number::<S>(j)?)?;
        w.observation[p + q + j - 1] = if j % 2 == 1 { binomial } else { -binomial };
        w.mean[p + q + j - 1] =
            finite(values(d - j) - trend.value(&parameters[p + q..p + q + trend.len()], d - j)?)?;
    }
    if p > 0 {
        w.transition[..p + q].copy_from_slice(&w.observation[..p + q]);
        w.noise[0] = S::ONE;
        for i in 1..p {
            w.transition[i * m + i - 1] = S::ONE;
        }
    }
    if q > 0 {
        w.noise[p] = S::ONE;
        for i in 1..q {
            w.transition[(p + i) * m + p + i - 1] = S::ONE;
        }
    }
    if d > 0 {
        w.transition[(p + q) * m..(p + q + 1) * m].copy_from_slice(&w.observation);
        w.noise[p + q] = S::ONE;
        for i in 1..d {
            w.transition[(p + q + i) * m + p + q + i - 1] = S::ONE;
        }
    }
    for i in 0..p {
        for j in 0..p {
            w.covariance[i * m + j] = finite(variance * w.gamma[i.abs_diff(j)])?;
        }
    }
    for i in 0..q {
        w.covariance[(p + i) * m + p + i] = variance;
    }
    for i in 0..p {
        for j in i..q {
            let v = finite(variance * w.psi[j - i])?;
            w.covariance[i * m + p + j] = v;
            w.covariance[(p + j) * m + i] = v;
        }
    }
    Ok(relative)
}
// Correlated observation/process noise. Joseph form avoids covariance downdate cancellation.
fn step<S: RealScalar>(
    w: &mut ArimaWorkspace<S>,
    variance: S,
    observation: Option<S>,
) -> Result<(S, S, Option<S>), SciOxideError> {
    let m = w.mean.len();
    let prediction =
        checked_product_sum(w.observation.iter().copied().zip(w.mean.iter().copied()))?;
    for i in 0..m {
        w.gain[i] =
            checked_product_sum((0..m).map(|j| (w.covariance[i * m + j], w.observation[j])))?;
    }
    let f = checked_product_sum(
        core::iter::once((variance, S::ONE))
            .chain(w.observation.iter().copied().zip(w.gain.iter().copied())),
    )?;
    if f <= S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("parameters"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "nonpositive innovation variance",
        ));
    }
    let residual = observation.map(|x| finite(x - prediction)).transpose()?;
    for i in 0..m {
        let cross = checked_product_sum(
            core::iter::once((variance, w.noise[i]))
                .chain((0..m).map(|j| (w.transition[i * m + j], w.gain[j]))),
        )?;
        let k = if residual.is_some() {
            finite(cross / f)?
        } else {
            S::ZERO
        };
        w.prediction[i] = checked_product_sum(
            (0..m)
                .map(|j| (w.transition[i * m + j], w.mean[j]))
                .chain(core::iter::once((k, residual.unwrap_or(S::ZERO)))),
        )?;
        for j in 0..m {
            w.adjusted[i * m + j] = finite(w.transition[i * m + j] - k * w.observation[j])?;
        }
        // Reuse prediction-only RHS storage after PH has been fully consumed below.
        w.next[i] = finite(w.noise[i] - k)?;
    }
    for i in 0..m {
        for j in 0..m {
            w.product[i * m + j] = checked_product_sum(
                (0..m).map(|k| (w.adjusted[i * m + k], w.covariance[k * m + j])),
            )?;
        }
    }
    // Noise loadings must survive writes to the candidate covariance.
    for i in 0..m {
        w.gain[i] = w.next[i];
    }
    for i in 0..m {
        for j in 0..=i {
            let v = checked_product_sum(
                (0..m)
                    .map(|k| (w.product[i * m + k], w.adjusted[j * m + k]))
                    .chain(core::iter::once((variance * w.gain[i], w.gain[j]))),
            )?;
            if i == j && v < S::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("parameters"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "negative propagated covariance",
                ));
            }
            w.next[i * m + j] = v;
            w.next[j * m + i] = v;
        }
    }
    w.mean.copy_from_slice(&w.prediction);
    w.covariance.copy_from_slice(&w.next);
    Ok((prediction, f, residual))
}
/// ARIMA model with fixed parameters and the filter state after the last time
/// point of its series, from [`PreparedArimaFit::model`] or
/// [`PreparedArimaFit::fit`].
///
/// Forecasting does not change the model.
pub struct ArimaModel<S: RealScalar> {
    order: ArimaOrder,
    trend: ArimaTrend,
    parameters: Vec<S>,
    samples: usize,
    max_bytes: usize,
    transition: Vec<S>,
    observation: Vec<S>,
    noise: Vec<S>,
    mean: Vec<S>,
    covariance: Vec<S>,
    transfer: TransferFunction<S>,
}
impl<S: RealScalar> ArimaModel<S>
where
    scioxide_core::Complex<S>: NumericScalar<Real = S>,
{
    /// Returns an upper bound on the bytes needed to build a model, including
    /// temporary storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `trend` is not allowed for `order.d`.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(order: ArimaOrder, trend: ArimaTrend) -> Result<usize, SciOxideError> {
        trend.check(order.d)?;
        let m = order.state_len()?;
        mul(
            add(mul(2, mul(m, m)?)?, add(mul(8, add(m, 2)?)?, trend.len())?)?,
            core::mem::size_of::<S>(),
        )
    }
    fn from_state(
        problem: &PreparedArimaFit<'_, S>,
        parameters: &[S],
        w: &ArimaWorkspace<S>,
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        limit_bytes(
            Self::required_bytes(problem.order, problem.trend)?,
            max_bytes,
        )?;
        let mut numerator = try_filled(
            problem.order.q + 1,
            S::ONE,
            scioxide_core::quantity!("arima_numerator"),
        )?;
        numerator[1..]
            .copy_from_slice(&parameters[problem.order.p..problem.order.p + problem.order.q]);
        let mut denominator = try_filled(
            problem.order.p + 1,
            S::ONE,
            scioxide_core::quantity!("arima_denominator"),
        )?;
        for (a, &phi) in denominator[1..].iter_mut().zip(parameters) {
            *a = -phi;
        }
        let transfer = TransferFunction::new(IirDomain::Digital, &numerator, &denominator)?;
        fn copy<S: RealScalar>(a: &[S]) -> Result<Vec<S>, SciOxideError> {
            let mut v = try_filled(
                a.len(),
                S::ZERO,
                scioxide_core::quantity!("arima_model_arrays"),
            )?;
            v.copy_from_slice(a);
            Ok(v)
        }
        Ok(Self {
            order: problem.order,
            trend: problem.trend,
            parameters: copy(parameters)?,
            samples: problem.samples,
            max_bytes: problem.options.max_bytes(),
            transition: copy(&w.transition)?,
            observation: copy(&w.observation)?,
            noise: copy(&w.noise)?,
            mean: copy(&w.mean)?,
            covariance: copy(&w.covariance)?,
            transfer,
        })
    }
    /// Returns the model order.
    pub fn order(&self) -> ArimaOrder {
        self.order
    }
    /// Returns the trend.
    pub fn trend(&self) -> ArimaTrend {
        self.trend
    }
    /// Returns the length `n` of the series the model was built from. The first
    /// forecast is for time index `n`.
    pub fn sample_count(&self) -> usize {
        self.samples
    }
    /// Returns the parameters `[φ₁, …, φₚ, θ₁, …, θ_q, trend, σ²]`, as described
    /// on [`PreparedArimaFit`].
    pub fn parameters(&self) -> &[S] {
        &self.parameters
    }
    /// Creates a workspace for forecasts of up to `horizon` steps.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `horizon` is zero.
    /// - `ResourceLimit` if the workspace exceeds the memory limit of the
    ///   problem the model came from.
    pub fn workspace(&self, horizon: usize) -> Result<ArimaWorkspace<S>, SciOxideError> {
        ArimaWorkspace::reserve(self.order, horizon, self.max_bytes)
    }
    /// Computes the two-sided spectral density of the ARMA part of the model at
    /// `frequency`.
    ///
    /// The ARMA part is the differenced, detrended series
    /// `(1 − B)ᵈ (xₜ − μₜ)`. With sample rate `fs` and `z = exp(2πi ν / fs)`,
    ///
    /// ```text
    /// S(ν) = (σ² / fs) |θ(z)|² / |φ(z)|²,   −fs/2 ≤ ν ≤ fs/2,
    /// ```
    ///
    /// in variance per unit of frequency. Its integral over `[−fs/2, fs/2]` is the
    /// variance of the ARMA part.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `frequency` or `sample_rate` is not finite,
    ///   `sample_rate ≤ 0`, or `|frequency| > sample_rate / 2`.
    pub fn spectral_density(&self, frequency: S, sample_rate: S) -> Result<S, SciOxideError> {
        if !frequency.is_finite()
            || !sample_rate.is_finite()
            || sample_rate <= S::ZERO
            || frequency.abs() > sample_rate / (S::TWO)
        {
            return Err(invalid(
                scioxide_core::argument!("frequency"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "model spectrum frequency/rate",
            ));
        }
        let response = self
            .transfer
            .response((frequency / sample_rate) * (S::TWO))?;
        let variance = self.parameters[self.parameters.len() - 1];
        let power = checked_product_sum([(response.re, response.re), (response.im, response.im)])?;
        finite((variance / sample_rate) * power)
    }
    /// Forecasts the next horizon observations using owned storage.
    ///
    /// # Errors
    ///
    /// Returns allocation errors and the errors of [`Self::forecast_into`].
    pub fn forecast(&self, horizon: usize) -> Result<Vec<ArimaForecast<S>>, SciOxideError> {
        let mut workspace = self.workspace(horizon)?;
        let mut output = try_filled(
            horizon,
            ArimaForecast {
                mean: S::ZERO,
                variance: S::ZERO,
            },
            scioxide_core::quantity!("arima_forecast"),
        )?;
        self.forecast_into(
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Computes forecasts for the `output.len()` time points after the series.
    ///
    /// `output[h]` is the mean and variance of `x` at time index
    /// `n + h`, given the observations, where `n` is [`Self::sample_count`].
    /// The variance includes the uncertainty of the `d` integrations, so it
    /// grows without bound for `d > 0`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `workspace` was made for a different order or for
    ///   fewer than `output.len()` steps, as by [`Self::workspace`].
    /// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
    pub fn forecast_into(
        &self,
        output: &mut [ArimaForecast<S>],
        workspace: &mut ArimaWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if workspace.order != self.order || workspace.forecasts.len() < output.len() {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "forecast workspace shape",
            ));
        }
        add(self.samples, output.len())?;
        context.check()?;
        workspace.transition.copy_from_slice(&self.transition);
        workspace.observation.copy_from_slice(&self.observation);
        workspace.noise.copy_from_slice(&self.noise);
        workspace.mean.copy_from_slice(&self.mean);
        workspace.covariance.copy_from_slice(&self.covariance);
        let start = self.order.p + self.order.q;
        for i in 0..output.len() {
            context.check()?;
            let (prediction, variance, _) =
                step(workspace, self.parameters[self.parameters.len() - 1], None)?;
            let trend = self.trend.value(
                &self.parameters[start..start + self.trend.len()],
                self.samples + i,
            )?;
            workspace.forecasts[i] = ArimaForecast {
                mean: finite(prediction + trend)?,
                variance,
            };
        }
        context.check()?;
        output.copy_from_slice(&workspace.forecasts[..output.len()]);
        Ok(())
    }
}
