//! Cox proportional hazards regression with time-fixed covariates.

use super::{SurvivalObservation, SurvivalOptions, event_times, invalid, number};
use alloc::vec::Vec;
use scioxide_core::NumericScalar;
use scioxide_core::{
    Bounds, Interval, RankThreshold, RealScalar, Scaled, ScaledAccumulator, SciOxideError,
    SolveContext,
};
use scioxide_linalg::{HermitianEigenOptions, MatrixView};

use crate::{
    ConfidenceProbability, FitDerivativeCapabilities, FitDerivatives, FitInformation, FitObjective,
    FitRequest, FitResult, FitSolver, StatsTermination, fit_information,
};

/// Approximation of the Cox partial likelihood at tied event times.
///
/// At an event time with `d` tied events, event set `D` and risk set `R`, let
/// `wⱼ = exp(xⱼᵀβ)`. The two methods differ in the denominators of the partial
/// likelihood.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoxTies {
    /// Efron's approximation: the `k`-th of the `d` denominators,
    /// `k = 0, …, d − 1`, is `Σ_{j∈R} wⱼ − (k/d) Σ_{j∈D} wⱼ`.
    ///
    /// B. Efron, "The efficiency of Cox's likelihood function for censored
    /// data", *J. Amer. Statist. Assoc.* 72 (1977), 557–565.
    #[default]
    Efron,
    /// Breslow's approximation: all `d` denominators are `Σ_{j∈R} wⱼ`.
    ///
    /// N. E. Breslow, "Covariance analysis of censored survival data",
    /// *Biometrics* 30 (1974), 89–99.
    Breslow,
}

/// Negative log partial likelihood of the Cox proportional hazards model, for
/// use with [`cox_fit_scoped`] or another [`FitSolver`](crate::FitSolver).
///
/// The model gives subject `i` with covariate row `xᵢ` the hazard
/// `h(t | xᵢ) = h₀(t) exp(xᵢᵀβ)`, with an unspecified baseline hazard `h₀`.
/// For each distinct event time `t` with event set `D(t)` and risk set `R(t)`,
/// the Breslow form of the objective is
///
/// ```text
/// ℓ(β) = Σ_t [ |D(t)| log Σ_{j∈R(t)} exp(xⱼᵀβ) − Σ_{i∈D(t)} xᵢᵀβ ]
/// ```
///
/// [`CoxTies`] selects how tied events enter the denominators. Row `i` of
/// `design` holds the covariates of observation `i`. No intercept is added,
/// because the partial likelihood does not identify one. The gradient is the
/// negative score, and the Hessian is the observed information.
///
/// Exponentials are evaluated with an extended exponent range, and covariates
/// enter through differences between subjects. So large linear predictors do
/// not overflow, and adding a constant to a covariate column loses no
/// accuracy. One evaluation of the objective costs O(e n p) operations, for
/// `n` observations, `e` events and `p` covariates. The gradient costs `p`
/// times as much and the Hessian `p²` times as much.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{
///     CoxTies, FitDerivatives, PreparedCox, SurvivalObservation, SurvivalOptions, SurvivalOutcome,
/// };
///
/// let data = [1.0, 2.0, 3.0]
///     .map(|t| SurvivalObservation::new(None, t, SurvivalOutcome::Event))
///     .into_iter()
///     .collect::<Result<Vec<_>, _>>()?;
/// // Covariates 0, 1, 2 shifted by 2³⁰. At β = 0 the information is the sum of
/// // the risk-set variances 2/3 + 1/4 + 0 = 11/12, whatever the shift.
/// let x = [1073741824.0_f64, 1073741825.0, 1073741826.0];
/// let options = SurvivalOptions::new().with_max_bytes(1024);
/// let design = MatrixView::from_slice(3, 1, &x)?;
/// let model = PreparedCox::new(&data, design, CoxTies::Efron, options)?;
/// let mut information = [0.0];
/// model.hessian(&[0.0], &mut information)?;
/// assert!((information[0] - 11.0 / 12.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. R. Cox, "Regression models and life-tables", *J. R. Stat. Soc. Ser. B*
/// 34 (1972), 187–220.
#[derive(Clone)]
pub struct PreparedCox<'a, S: RealScalar> {
    observations: &'a [SurvivalObservation<S>],
    design: MatrixView<'a, S>,
    times: Vec<S>,
    ties: CoxTies,
}

impl<'a, S: RealScalar> PreparedCox<'a, S> {
    /// Checks the data and stores the sorted distinct event times.
    ///
    /// `design` is an `n × p` matrix with one row per observation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `observations` is empty, `design` has no columns, or
    ///   the number of rows of `design` differs from the number of observations.
    /// - `InvalidInput` if a covariate is NaN or infinite.
    /// - `InvalidInput` if no observation has an event.
    /// - `ResourceLimit` if the event times exceed `options.max_bytes()`.
    pub fn new(
        observations: &'a [SurvivalObservation<S>],
        design: MatrixView<'a, S>,
        ties: CoxTies,
        options: SurvivalOptions,
    ) -> Result<Self, SciOxideError> {
        if observations.is_empty() || design.rows() != observations.len() || design.cols() == 0 {
            return Err(invalid(
                scioxide_core::argument!("design"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox observation/design shape",
            ));
        }
        for row in 0..design.rows() {
            for col in 0..design.cols() {
                if !design.get(row, col)?.is_finite() {
                    return Err(invalid(
                        scioxide_core::argument!("design"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "Cox covariates must be finite",
                    ));
                }
            }
        }
        let times = event_times(observations, options.max_bytes())?;
        if times.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("observations"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Cox regression needs at least one event",
            ));
        }
        Ok(Self {
            observations,
            design,
            times,
            ties,
        })
    }

    /// Returns the tie method, which also applies to the baseline hazard and
    /// residuals of [`cox_fit_scoped`].
    pub const fn ties(&self) -> CoxTies {
        self.ties
    }

    /// Returns the number of observations `n`.
    pub const fn observations(&self) -> usize {
        self.observations.len()
    }

    /// Returns the number of covariates `p`.
    pub const fn covariates(&self) -> usize {
        self.design.cols()
    }

    fn covariate(&self, row: usize, col: usize) -> Result<f64, SciOxideError> {
        self.design.get(row, col)?.try_to_f64_exact()
    }

    fn predictor(
        &self,
        row: usize,
        reference: Option<usize>,
        coefficients: &[S],
    ) -> Result<Scaled<f64>, SciOxideError> {
        let mut sum = ScaledAccumulator::ZERO;
        for (col, &coefficient) in coefficients.iter().enumerate() {
            let mut x = Scaled::from_value(self.covariate(row, col)?)?;
            if let Some(reference) = reference {
                x = x.sub(Scaled::from_value(self.covariate(reference, col)?)?)?;
            }
            sum.add_product(Scaled::from_value(coefficient.into_f64())?, x)?;
        }
        sum.total()
    }

    fn validate_coefficients(&self, coefficients: &[S]) -> Result<(), SciOxideError> {
        if coefficients.len() != self.covariates() || coefficients.iter().any(|x| !x.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("coefficients"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox coefficient shape or nonfinite value",
            ));
        }
        Ok(())
    }

    fn group(&self, time: S, coefficients: &[S]) -> Result<Group, SciOxideError> {
        let mut anchor = None;
        let mut deaths = 0;
        for (row, &observation) in self.observations.iter().enumerate() {
            if observation.at_risk(time) {
                if let Some(previous) = anchor {
                    // Compare predictor contrasts directly: subtracting two
                    // rounded raw predictors can erase their ordering.
                    if self
                        .predictor(row, Some(previous), coefficients)?
                        .cmp_value(Scaled::ZERO)
                        .is_gt()
                    {
                        anchor = Some(row);
                    }
                } else {
                    anchor = Some(row);
                }
                deaths += usize::from(observation.event && observation.time == time);
            }
        }
        let anchor = anchor.filter(|_| deaths > 0).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("coefficients"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox event group has no valid risk set",
            )
        })?;
        Ok(Group { anchor, deaths })
    }

    fn contrast(
        &self,
        row: usize,
        anchor: usize,
        col: usize,
    ) -> Result<Scaled<f64>, SciOxideError> {
        Scaled::from_value(self.covariate(row, col)?)?
            .sub(Scaled::from_value(self.covariate(anchor, col)?)?)
    }

    fn weight(
        &self,
        time: S,
        row: usize,
        group: Group,
        step: usize,
        coefficients: &[S],
    ) -> Result<Scaled<f64>, SciOxideError> {
        let observation = self.observations[row];
        if !observation.at_risk(time) {
            return Ok(Scaled::ZERO);
        }
        let weight = exponential(self.predictor(row, Some(group.anchor), coefficients)?)?;
        if self.ties == CoxTies::Efron && observation.event && observation.time == time {
            weight.mul(crate::arithmetic::count_ratio(
                group.deaths - step,
                group.deaths,
            )?)
        } else {
            Ok(weight)
        }
    }

    fn denominator(
        &self,
        time: S,
        group: Group,
        step: usize,
        coefficients: &[S],
    ) -> Result<Scaled<f64>, SciOxideError> {
        let mut sum = ScaledAccumulator::ZERO;
        for row in 0..self.observations.len() {
            sum.add_scaled(self.weight(time, row, group, step, coefficients)?)?;
        }
        let sum = sum.total()?;
        if sum.cmp_value(Scaled::ZERO).is_le() {
            return Err(invalid(
                scioxide_core::argument!("coefficients"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox risk denominator vanishes",
            ));
        }
        Ok(sum)
    }

    fn mean(
        &self,
        time: S,
        group: Group,
        step: usize,
        coefficients: &[S],
        col: usize,
        denominator: Scaled<f64>,
    ) -> Result<ScaledAccumulator<f64>, SciOxideError> {
        let mut sum = ScaledAccumulator::ZERO;
        for row in 0..self.observations.len() {
            sum.add_product(
                self.weight(time, row, group, step, coefficients)?,
                self.contrast(row, group.anchor, col)?,
            )?;
        }
        sum.multiplied(Scaled::ONE.div(denominator)?)
    }

    fn event_mean(
        &self,
        time: S,
        group: Group,
        col: usize,
    ) -> Result<ScaledAccumulator<f64>, SciOxideError> {
        let mut sum = ScaledAccumulator::ZERO;
        for (row, observation) in self.observations.iter().enumerate() {
            if observation.event && observation.time == time {
                sum.add_scaled(self.contrast(row, group.anchor, col)?)?;
            }
        }
        sum.multiplied(crate::arithmetic::count_ratio(1, group.deaths)?)
    }

    // One owner for all likelihood derivatives; scalar projection happens only
    // after all event groups for the requested component have been accumulated.
    fn component(
        &self,
        coefficients: &[S],
        quantity: CoxQuantity,
    ) -> Result<Scaled<f64>, SciOxideError> {
        let mut result = ScaledAccumulator::ZERO;
        for &time in &self.times {
            let group = self.group(time, coefficients)?;
            if matches!(quantity, CoxQuantity::Value) {
                for (row, observation) in self.observations.iter().enumerate() {
                    if observation.event && observation.time == time {
                        result.sub_scaled(self.predictor(
                            row,
                            Some(group.anchor),
                            coefficients,
                        )?)?;
                    }
                }
            }
            for step in 0..group.deaths {
                let denominator = self.denominator(time, group, step, coefficients)?;
                match quantity {
                    CoxQuantity::Value => result.add_value(denominator.ln_abs()?)?,
                    CoxQuantity::Gradient(col) => {
                        let event = self.event_mean(time, group, col)?;
                        let mean = self.mean(time, group, step, coefficients, col, denominator)?;
                        result.add_scaled(crate::arithmetic::center_difference(event, mean)?)?;
                    }
                    CoxQuantity::Hessian(a, b) => {
                        let ma = self.mean(time, group, step, coefficients, a, denominator)?;
                        let mb = self.mean(time, group, step, coefficients, b, denominator)?;
                        let mut covariance = ScaledAccumulator::ZERO;
                        for row in 0..self.observations.len() {
                            let weight = self.weight(time, row, group, step, coefficients)?;
                            if weight.is_zero() {
                                continue;
                            }
                            let da = crate::arithmetic::center_difference(
                                ma,
                                ScaledAccumulator::from_scaled(self.contrast(
                                    row,
                                    group.anchor,
                                    a,
                                )?),
                            )?;
                            let db = crate::arithmetic::center_difference(
                                mb,
                                ScaledAccumulator::from_scaled(self.contrast(
                                    row,
                                    group.anchor,
                                    b,
                                )?),
                            )?;
                            covariance.add_product(weight.mul(da)?, db)?;
                        }
                        result.add_scaled(covariance.total()?.div(denominator)?)?;
                    }
                }
            }
        }
        result.total()
    }
}

#[derive(Clone, Copy)]
struct Group {
    anchor: usize,
    deaths: usize,
}

#[derive(Clone, Copy)]
enum CoxQuantity {
    Value,
    Gradient(usize),
    Hessian(usize, usize),
}

fn exponential(log: Scaled<f64>) -> Result<Scaled<f64>, SciOxideError> {
    // Finite binary64 covariate contrasts are < 2^1025. Even squared
    // contrasts, three usize-sized count factors, and a final subnormal
    // projection cannot make exp(-4096) contribute a representable result.
    if log.cmp_value(Scaled::from_value(-4096.0)?).is_lt() {
        Ok(Scaled::ZERO)
    } else {
        Scaled::from_log(1.0, log.round()?)
    }
}

impl<S: RealScalar> FitObjective<S> for PreparedCox<'_, S> {
    type Workspace = ();

    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }

    fn workspace(&self) -> Result<Self::Workspace, SciOxideError> {
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.covariates()
    }

    fn validate(
        &self,
        parameters: &[S],
        _workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.validate_coefficients(parameters)
    }

    fn value(
        &self,
        parameters: &[S],
        _workspace: &mut Self::Workspace,
    ) -> Result<S, SciOxideError> {
        self.validate_coefficients(parameters)?;
        self.component(parameters, CoxQuantity::Value)?.round_to()
    }
}

impl<S: RealScalar> FitDerivatives<S> for PreparedCox<'_, S> {
    fn capabilities(&self) -> FitDerivativeCapabilities {
        FitDerivativeCapabilities {
            gradient: true,
            hessian: true,
            jacobian: false,
        }
    }

    fn gradient(&self, parameters: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        self.validate_coefficients(parameters)?;
        if output.len() != self.covariates() {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox gradient output shape",
            ));
        }
        for (col, value) in output.iter_mut().enumerate() {
            *value = self
                .component(parameters, CoxQuantity::Gradient(col))?
                .round_to()?;
        }
        Ok(())
    }

    fn hessian(&self, parameters: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        self.validate_coefficients(parameters)?;
        let p = self.covariates();
        if p.checked_mul(p) != Some(output.len()) {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox Hessian output shape",
            ));
        }
        for col in 0..p {
            for other in col..p {
                let value = self
                    .component(parameters, CoxQuantity::Hessian(col, other))?
                    .round_to()?;
                output[col * p + other] = value;
                output[other * p + col] = value;
            }
        }
        Ok(())
    }
}

/// One step of the estimated baseline cumulative hazard of a Cox model.
///
/// The baseline is the hazard of a subject with all covariates zero. At the
/// fitted coefficients, the Breslow estimator adds `d / Σ_{j∈R} exp(xⱼᵀβ)` at
/// each event time with `d` events and risk set `R`. With [`CoxTies::Efron`] it
/// adds the reciprocals of the `d` Efron denominators instead.
#[derive(Clone, Copy, Debug)]
pub struct CoxBaselinePoint<S: RealScalar> {
    /// Distinct event time.
    pub time: S,
    /// Number of events at this time.
    pub events: usize,
    /// Baseline cumulative hazard `H₀(t)` just after the events at this time.
    pub cumulative_hazard: S,
    /// Baseline survival `exp(−H₀(t))`.
    pub survival: S,
}

/// Inference and diagnostics of a converged Cox fit, from [`CoxFit::report`].
///
/// It holds the observed information at the fitted coefficients `β̂`, Wald
/// intervals, hazard ratios, the baseline cumulative hazard and the
/// Schoenfeld residuals.
///
/// The Schoenfeld residual of an event of subject `i` at time `t` is
/// `xᵢ − x̄(t)`, where `x̄(t)` is the risk-set mean of the covariates weighted
/// by `exp(xⱼᵀβ̂)`. For `d` tied events, `x̄(t)` is the average of the `d`
/// weighted means that belong to the [`CoxTies`] denominators. A trend of the
/// residuals in time points to non-proportional hazards. It is a diagnostic,
/// not a test.
///
/// The fields are read through accessors:
///
/// ```compile_fail
/// fn forge<S: scioxide_core::RealScalar>(report: &mut scioxide_stats::CoxReport<S>) {
///     report.coefficient_intervals = None;
/// }
/// ```
///
/// # References
///
/// D. Schoenfeld, "Partial residuals for the proportional hazards regression
/// model", *Biometrika* 69 (1982), 239–241.
///
/// P. M. Grambsch and T. M. Therneau, "Proportional hazards tests and
/// diagnostics based on weighted residuals", *Biometrika* 81 (1994), 515–526.
pub struct CoxReport<S: RealScalar> {
    information: FitInformation<S>,
    coefficient_intervals: Option<Vec<Interval<S>>>,
    hazard_ratios: Vec<S>,
    baseline: Vec<CoxBaselinePoint<S>>,
    /// Observation index of each event, in event-time order.
    schoenfeld_rows: Vec<usize>,
    /// Schoenfeld residuals, one row of `p` values per event, row-major.
    schoenfeld_values: Vec<S>,
    schoenfeld_columns: usize,
}

/// Schoenfeld residual of one event, from [`CoxReport::schoenfeld_residuals`].
#[derive(Clone, Copy, Debug)]
pub struct SchoenfeldResidual<'a, S: RealScalar> {
    observation: usize,
    values: &'a [S],
}
impl<'a, S: RealScalar> SchoenfeldResidual<'a, S> {
    /// Returns the index of the observation with this event.
    pub const fn observation(self) -> usize {
        self.observation
    }
    /// Returns the residual `xᵢ − x̄(t)`, one value per covariate.
    pub const fn values(self) -> &'a [S] {
        self.values
    }
}
impl<S: RealScalar> CoxReport<S> {
    /// Returns the observed information at `β̂`: its rank, condition number,
    /// identifiability and, when identifiable, the covariance of `β̂`.
    pub fn information(&self) -> &FitInformation<S> {
        &self.information
    }
    /// Returns the pointwise Wald intervals `β̂ₖ ± z SE(β̂ₖ)` of the
    /// coefficients, on the log hazard ratio scale. `z` is the normal critical
    /// value of [`CoxInferenceOptions::confidence`]. `None` when the
    /// information gives no covariance.
    pub fn coefficient_intervals(&self) -> Option<&[Interval<S>]> {
        self.coefficient_intervals.as_deref()
    }
    /// Returns the hazard ratios `exp(β̂ₖ)`.
    pub fn hazard_ratios(&self) -> &[S] {
        &self.hazard_ratios
    }
    /// Returns the baseline cumulative hazard and survival at each distinct
    /// event time.
    pub fn baseline(&self) -> &[CoxBaselinePoint<S>] {
        &self.baseline
    }
    /// Returns the Schoenfeld residuals, one per event, in order of event time.
    pub fn schoenfeld_residuals(&self) -> impl ExactSizeIterator<Item = SchoenfeldResidual<'_, S>> {
        self.schoenfeld_rows
            .iter()
            .copied()
            .zip(self.schoenfeld_values.chunks_exact(self.schoenfeld_columns))
            .map(|(observation, values)| SchoenfeldResidual {
                observation,
                values,
            })
    }
}

/// Result of [`cox_fit_scoped`]: the fitted coefficients with the solver report, and
/// the model report when the fit converged.
///
/// The termination status of the fit is in [`FitResult::report`](crate::FitResult::report).
/// The fields are read through accessors:
///
/// ```compile_fail
/// fn forge<S: scioxide_core::RealScalar>(fit: &mut scioxide_stats::CoxFit<S>) {
///     fit.report = None;
/// }
/// ```
pub struct CoxFit<S: RealScalar> {
    fit: FitResult<S>,
    report: Option<CoxReport<S>>,
}
impl<S: RealScalar> CoxFit<S> {
    /// Returns the fitted coefficients `β̂` and the solver report.
    pub fn fit(&self) -> &FitResult<S> {
        &self.fit
    }
    /// Returns the model report, or `None` unless the fit terminated with
    /// [`StatsTermination::Converged`](crate::StatsTermination::Converged).
    pub fn report(&self) -> Option<&CoxReport<S>> {
        self.report.as_ref()
    }
}

/// Options of [`cox_fit_scoped`]: interval coverage, rank decision of the observed
/// information, and memory limits.
#[derive(Clone, Copy, Debug)]
pub struct CoxInferenceOptions<S: RealScalar> {
    /// Two-sided coverage of the Wald intervals.
    confidence: ConfidenceProbability,
    /// Threshold for the numerical rank of the observed information.
    rank_tolerance: RankThreshold<S>,
    /// Options of the eigensolver applied to the observed information.
    eigen_options: HermitianEigenOptions<S>,
    /// Memory limit of the fit, in bytes.
    max_fit_bytes: usize,
    /// Memory limit of the report, in bytes.
    max_report_bytes: usize,
}

impl<S: RealScalar> Default for CoxInferenceOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> CoxInferenceOptions<S> {
    /// Returns 95 % coverage, the default rank threshold and eigensolver
    /// options, and no memory limits.
    pub fn new() -> Self {
        Self {
            confidence: Default::default(),
            rank_tolerance: Default::default(),
            eigen_options: Default::default(),
            max_fit_bytes: usize::MAX,
            max_report_bytes: usize::MAX,
        }
    }
    /// Sets the two-sided coverage of the Wald intervals.
    pub fn with_confidence(mut self, value: ConfidenceProbability) -> Self {
        self.confidence = value;
        self
    }
    /// Returns the coverage of the Wald intervals.
    pub fn confidence(&self) -> ConfidenceProbability {
        self.confidence
    }
    /// Sets the threshold that decides the numerical rank of the observed
    /// information from its eigenvalues.
    pub fn with_rank_tolerance(mut self, value: RankThreshold<S>) -> Self {
        self.rank_tolerance = value;
        self
    }
    /// Returns the rank threshold.
    pub fn rank_tolerance(&self) -> RankThreshold<S> {
        self.rank_tolerance
    }
    /// Sets the options of the eigensolver applied to the observed information.
    pub fn with_eigen_options(mut self, value: HermitianEigenOptions<S>) -> Self {
        self.eigen_options = value;
        self
    }
    /// Returns the eigensolver options.
    pub fn eigen_options(&self) -> HermitianEigenOptions<S> {
        self.eigen_options
    }
    /// Sets the memory limit of the fit in bytes. It covers the parameter
    /// vectors and bounds, but not the solver's own storage.
    pub fn with_max_fit_bytes(mut self, value: usize) -> Self {
        self.max_fit_bytes = value;
        self
    }
    /// Returns the memory limit of the fit in bytes.
    pub fn max_fit_bytes(&self) -> usize {
        self.max_fit_bytes
    }
    /// Sets the memory limit of the report in bytes, including the
    /// eigendecomposition of the observed information.
    pub fn with_max_report_bytes(mut self, value: usize) -> Self {
        self.max_report_bytes = value;
        self
    }
    /// Returns the memory limit of the report in bytes.
    pub fn max_report_bytes(&self) -> usize {
        self.max_report_bytes
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> PreparedCox<'_, S> {
    fn report(
        &self,
        coefficients: &[S],
        bounds: &Bounds<S>,
        integrality: &[bool],
        options: CoxInferenceOptions<S>,
    ) -> Result<CoxReport<S>, SciOxideError> {
        self.validate_coefficients(coefficients)?;
        let p = self.covariates();
        let events = self.observations.iter().filter(|row| row.event).count();
        let square = p.checked_mul(p).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("cox_report_dimensions"))
        })?;
        // The eigensolver and fit_information use several dense p-by-p buffers.
        // This deliberately conservative payload envelope includes them.
        let bytes = square
            .checked_mul(16 * core::mem::size_of::<S>())
            .and_then(|x| {
                x.checked_add(events.checked_mul(p.checked_mul(core::mem::size_of::<S>())?)?)
            })
            .and_then(|x| x.checked_add(events.checked_mul(core::mem::size_of::<usize>())?))
            .and_then(|x| {
                x.checked_add(
                    self.times
                        .len()
                        .checked_mul(core::mem::size_of::<CoxBaselinePoint<S>>())?,
                )
            })
            .and_then(|x| {
                x.checked_add(p.checked_mul(
                    core::mem::size_of::<Interval<S>>()
                        + core::mem::size_of::<S>()
                        + core::mem::size_of::<ScaledAccumulator<f64>>(),
                )?)
            })
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("cox_report_byte_budget"))
            })?;
        crate::error_helpers::limit_bytes(bytes, options.max_report_bytes)?;
        let mut hessian = scioxide_core::backend::try_filled(
            square,
            S::ZERO,
            scioxide_core::quantity!("cox_information"),
        )?;
        self.hessian(coefficients, &mut hessian)?;
        let information = fit_information(
            coefficients,
            bounds,
            integrality,
            &hessian,
            options.rank_tolerance,
            &options.eigen_options,
        )?;
        let z = options.confidence.normal_critical_value::<f64>()?;
        let coefficient_intervals = if let Some(covariance) = information.covariance() {
            let mut intervals = Vec::new();
            intervals.try_reserve_exact(p).map_err(|_| {
                invalid(
                    scioxide_core::argument!("options"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Cox intervals allocation failed",
                )
            })?;
            for (col, &coefficient) in coefficients.iter().enumerate() {
                let standard_error = covariance[col * p + col].try_to_f64_exact()?.sqrt();
                let center = coefficient.try_to_f64_exact()?;
                intervals.push(Interval::new(
                    number(center - z * standard_error)?,
                    number(center + z * standard_error)?,
                )?);
            }
            Some(intervals)
        } else {
            None
        };
        let mut hazard_ratios = Vec::new();
        hazard_ratios.try_reserve_exact(p).map_err(|_| {
            invalid(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox hazard-ratio allocation failed",
            )
        })?;
        for &coefficient in coefficients {
            hazard_ratios.push(number(coefficient.try_to_f64_exact()?.exp())?);
        }
        let mut baseline = Vec::new();
        baseline.try_reserve_exact(self.times.len()).map_err(|_| {
            invalid(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox baseline allocation failed",
            )
        })?;
        let mut schoenfeld_rows = Vec::new();
        schoenfeld_rows.try_reserve_exact(events).map_err(|_| {
            invalid(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Cox residual-index allocation failed",
            )
        })?;
        let mut schoenfeld_values = Vec::new();
        schoenfeld_values
            .try_reserve_exact(events * p)
            .map_err(|_| {
                invalid(
                    scioxide_core::argument!("options"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Cox residual allocation failed",
                )
            })?;
        let mut mean = scioxide_core::backend::try_filled(
            p,
            ScaledAccumulator::ZERO,
            scioxide_core::quantity!("cox_risk_means"),
        )?;
        let mut cumulative = ScaledAccumulator::ZERO;
        for &time in &self.times {
            mean.fill(ScaledAccumulator::ZERO);
            let group = self.group(time, coefficients)?;
            let inverse_shift = exponential(-self.predictor(group.anchor, None, coefficients)?)?;
            for step in 0..group.deaths {
                let denominator = self.denominator(time, group, step, coefficients)?;
                cumulative.add_scaled(inverse_shift.div(denominator)?)?;
                for (col, average) in mean.iter_mut().enumerate() {
                    let contribution = self
                        .mean(time, group, step, coefficients, col, denominator)?
                        .multiplied(crate::arithmetic::count_ratio(1, group.deaths)?)?;
                    for part in contribution.parts() {
                        average.add_scaled(part)?;
                    }
                }
            }
            baseline.push(CoxBaselinePoint {
                time,
                events: group.deaths,
                cumulative_hazard: cumulative.total()?.round_to()?,
                survival: number((-cumulative.total()?).exp()?)?,
            });
            for (row, &observation) in self.observations.iter().enumerate() {
                if observation.event && observation.time == time {
                    schoenfeld_rows.push(row);
                    for (col, &average) in mean.iter().enumerate() {
                        schoenfeld_values.push(
                            crate::arithmetic::center_difference(
                                average,
                                ScaledAccumulator::from_scaled(self.contrast(
                                    row,
                                    group.anchor,
                                    col,
                                )?),
                            )?
                            .round_to()?,
                        );
                    }
                }
            }
        }
        Ok(CoxReport {
            information,
            coefficient_intervals,
            hazard_ratios,
            baseline,
            schoenfeld_rows,
            schoenfeld_values,
            schoenfeld_columns: p,
        })
    }
}

/// Executes [`cox_fit_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn cox_fit_with_options<S: RealScalar + NumericScalar<Real = S>>(
    problem: &PreparedCox<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
    options: CoxInferenceOptions<S>,
) -> Result<CoxFit<S>, SciOxideError> {
    cox_fit_scoped(
        problem,
        initial,
        solver,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Executes [`cox_fit_scoped`] with the [one-shot defaults](crate#one-shot-forms)
/// and sequential native capabilities, requiring successful termination.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors, and maps
/// every unsuccessful termination to an error under the one-shot contract.
pub fn cox_fit<S: RealScalar + NumericScalar<Real = S>>(
    problem: &PreparedCox<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
) -> Result<CoxFit<S>, SciOxideError> {
    let result = cox_fit_with_options(problem, initial, solver, Default::default())?;
    let report = result.fit().report();
    if let Some(error) = report.termination().one_shot_error(
        scioxide_core::ConvergenceReason::NestedSolver,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts()),
    ) {
        return Err(error);
    }
    Ok(result)
}
/// Fits a Cox proportional hazards model by minimizing the negative log partial
/// likelihood of `problem` with `solver`, starting from `initial`.
///
/// The coefficients are unbounded. When the solver reports
/// [`StatsTermination::Converged`](crate::StatsTermination::Converged), the
/// result also holds a [`CoxReport`]. It contains the observed information
/// (the Hessian of [`PreparedCox`] at `β̂`), the Wald intervals from its
/// inverse, the hazard ratios, the baseline cumulative hazard and the
/// Schoenfeld residuals. `scope` supplies the execution context and the
/// [solve accounting](scioxide_core#solve-accounting).
///
/// # Errors
///
/// - `InvalidInput` if `initial` does not have one value per covariate.
/// - `ResourceLimit` if the fit exceeds `options.max_fit_bytes()` or the report
///   exceeds `options.max_report_bytes()`.
/// - `NumericalInstability` if a hazard ratio or interval endpoint overflows
///   the scalar type.
/// - Errors of `solver` and of the eigendecomposition of the observed
///   information are returned unchanged.
pub fn cox_fit_scoped<'scope, 'capability, S: RealScalar + NumericScalar<Real = S>>(
    problem: &PreparedCox<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
    options: CoxInferenceOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<CoxFit<S>, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("stats", "cox-fit", 1),
        ((problem.observations() as u128) << 64) | problem.covariates() as u128,
        0,
    );
    scope.into().run(
        &mut state,
        identity,
        scioxide_core::SolveBudgets::unbounded(),
        |context| cox_fit_attempt(problem, initial, solver, options, context),
    )
}

fn cox_fit_attempt<S: RealScalar + NumericScalar<Real = S>>(
    problem: &PreparedCox<'_, S>,
    initial: &[S],
    solver: &mut dyn FitSolver<S>,
    options: CoxInferenceOptions<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<CoxFit<S>, SciOxideError> {
    let p = problem.covariates();
    if initial.len() != p {
        return Err(invalid(
            scioxide_core::argument!("initial"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Cox initial coefficient shape",
        ));
    }
    let request_bytes = p
        .checked_mul(core::mem::size_of::<(Option<S>, Option<S>)>() + core::mem::size_of::<bool>())
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("cox_fitting_request"))
        })?;
    let fit_bytes = options
        .max_fit_bytes
        .checked_sub(request_bytes)
        .ok_or_else(|| {
            SciOxideError::resource_limit(
                scioxide_core::quantity!("cox_fitting_request_bytes"),
                request_bytes,
                options.max_fit_bytes,
            )
        })?;
    let bounds = Bounds::new(scioxide_core::backend::try_filled(
        p,
        (None, None),
        scioxide_core::quantity!("cox_bounds"),
    )?)?;
    let integrality =
        scioxide_core::backend::try_filled(p, false, scioxide_core::quantity!("cox_integrality"))?;
    let fit = crate::fit_scoped(
        FitRequest {
            objective: problem,
            derivatives: Some(problem),
            bounds: &bounds,
            integrality: &integrality,
            initial,
        },
        solver,
        crate::StatisticsOptions::new().with_max_bytes(fit_bytes),
        context,
    )?;
    let report = if fit.report().termination() == StatsTermination::Converged {
        fit.parameters()
            .as_ref()
            .map(|coefficients| problem.report(coefficients, &bounds, &integrality, options))
            .transpose()?
    } else {
        None
    };
    Ok(CoxFit { fit, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_event_residual_score_and_baseline_identity() -> Result<(), SciOxideError> {
        let data = [
            SurvivalObservation::new(None, 1.0, crate::SurvivalOutcome::Event)?,
            SurvivalObservation::new(None, 1.0, crate::SurvivalOutcome::Event)?,
            SurvivalObservation::new(None, 2.0, crate::SurvivalOutcome::Event)?,
        ];
        let covariates = [0.0, 1.0, 2.0];
        let design = MatrixView::from_slice(3, 1, &covariates)?;
        let bounds = Bounds::new(alloc::vec![(None, None)])?;
        let options = CoxInferenceOptions::new()
            .with_confidence(ConfidenceProbability::default())
            .with_rank_tolerance(RankThreshold::new(0.0, 1e-10)?)
            .with_eigen_options(HermitianEigenOptions::default())
            .with_max_fit_bytes(1024)
            .with_max_report_bytes(4096);
        for (ties, expected_baseline) in
            [(CoxTies::Efron, 11.0 / 6.0), (CoxTies::Breslow, 5.0 / 3.0)]
        {
            let problem = PreparedCox::new(
                &data,
                design,
                ties,
                SurvivalOptions::new().with_max_bytes(1024),
            )?;
            let report = problem.report(&[0.0], &bounds, &[false], options)?;
            let mut gradient = [0.0];
            FitDerivatives::gradient(&problem, &[0.0], &mut gradient)?;
            let residual_sum: f64 = report.schoenfeld_values.iter().sum();
            assert!((residual_sum + gradient[0]).abs() < 1e-12);
            assert!((report.baseline[1].cumulative_hazard - expected_baseline).abs() < 1e-12);
        }
        Ok(())
    }
}
