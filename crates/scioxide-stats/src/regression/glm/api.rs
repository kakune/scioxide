//! Prepared dense and operator GLMs, their outputs, workspaces and batch fits.
use super::dense::{DenseGlm, GlmScratch};
use super::irls::GlmStorage;
use super::operator::{GlmOperatorScratch, OperatorGlm};
use super::{
    GlmCoefficientInference, GlmCovariance, GlmFamily, GlmObjective, GlmOptions, GlmReport,
};
use crate::execution::SolveSlot;
use crate::regression::invalid;
use crate::{
    ConfidenceProbability, FitRequest, FitSolver, FitWorkspace, RegressionWeights,
    StatisticsOptions, StatsTermination,
};
use alloc::vec::Vec;
use scioxide_core::{
    AttemptOutcome, AttemptRecord, Bounds, BudgetKind, ExecutionContext, RealScalar, SciOxideError,
    SolveBudgets, SolveScope,
};
use scioxide_linalg::{Matrix, MatrixView, Vector};

/// Sets the byte limit of a prepared GLM.
///
/// The limit applies to the copied design and row metadata, and to every
/// output and workspace the plan allocates.
#[derive(Clone, Copy, Debug)]
pub struct GlmStorageOptions {
    max_bytes: usize,
}
impl Default for GlmStorageOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl GlmStorageOptions {
    /// Returns options with the byte limit `usize::MAX`.
    pub const fn new() -> Self {
        Self {
            max_bytes: usize::MAX,
        }
    }
    /// Sets the byte limit. Borrowed inputs and allocator overhead do not count
    /// toward it.
    pub const fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes;
        self
    }
    /// Returns the byte limit.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

/// Selects how a GLM fit computes the coefficients.
pub enum GlmSolver<'a, S: RealScalar> {
    /// Iteratively reweighted least squares. Each iteration solves a weighted
    /// least-squares problem by thin SVD for a dense design, or by LSMR for an
    /// operator design.
    Irls,
    /// An optimizer you provide minimizes the [`GlmObjective`] of the model,
    /// with optional bounds and integer constraints. Only dense designs accept
    /// it.
    Injected {
        /// Optimizer, with its own workspace.
        solver: &'a mut dyn FitSolver<S>,
        /// Starting coefficients, finite and within `bounds`.
        initial: &'a [S],
        /// Bounds on the coefficients.
        bounds: &'a Bounds<S>,
        /// Marks the coefficients restricted to integer values.
        integrality: &'a [bool],
    },
}
/// Holds the response, options and solver of one GLM fit.
pub struct GlmInputs<'a, S: RealScalar> {
    response: &'a [S],
    options: GlmOptions<S>,
    solver: GlmSolver<'a, S>,
}
impl<'a, S: RealScalar> GlmInputs<'a, S> {
    /// Returns inputs that fit `response` by IRLS with `options`.
    pub fn new(response: &'a [S], options: GlmOptions<S>) -> Self {
        Self {
            response,
            options,
            solver: GlmSolver::Irls,
        }
    }
    /// Sets the solver. [`PreparedOperatorGlm`] accepts only
    /// [`GlmSolver::Irls`].
    pub fn with_solver(mut self, solver: GlmSolver<'a, S>) -> Self {
        self.solver = solver;
        self
    }
}
/// Holds a GLM fit and the report of the fit that produced it.
///
/// Only a fit creates it. To reuse storage, use a [`GlmOutput`].
///
/// ```compile_fail
/// let _ = scioxide_stats::GlmFit::<f64>::new(4, 1);
/// ```
pub struct GlmFit<S: RealScalar> {
    raw: GlmStorage<S>,
    report: GlmReport,
}
impl<S: RealScalar> GlmFit<S> {
    /// Borrows the fit as a [`GlmFitView`].
    pub fn view(&self) -> GlmFitView<'_, S> {
        GlmFitView {
            raw: &self.raw,
            report: self.report,
        }
    }
    /// Returns the report of the fit.
    pub fn report(&self) -> &GlmReport {
        &self.report
    }
    /// Returns the coefficient estimates `β̂`.
    pub fn coefficients(&self) -> &Vector<S> {
        &self.raw.coefficients
    }
    /// Returns the fitted means `μ̂ᵢ = g⁻¹(η̂ᵢ)`, including rows of zero weight.
    pub fn fitted(&self) -> &Vector<S> {
        &self.raw.fitted
    }
    /// Returns the response residuals `yᵢ − μ̂ᵢ`.
    pub fn residuals(&self) -> &Vector<S> {
        &self.raw.residuals
    }
    /// Returns the linear predictors `η̂ᵢ = xᵢᵀβ̂ + oᵢ`, including the offset
    /// and the logarithm of the exposure.
    pub fn linear_predictor(&self) -> &Vector<S> {
        &self.raw.linear_predictor
    }
    /// Returns the residual deviance `D = Σᵢ wᵢ mᵢ d(yᵢ, μ̂ᵢ)`, where `d` is the
    /// unit deviance of the family, `wᵢ` the row weight and `mᵢ` the number of
    /// binomial trials. It is finite and nonnegative.
    pub fn deviance(&self) -> S {
        self.raw.deviance
    }
    /// Returns the Pearson statistic `X² = Σᵢ wᵢ mᵢ (yᵢ − μ̂ᵢ)² / V(μ̂ᵢ)`.
    pub fn pearson_chi_square(&self) -> S {
        self.raw.pearson_chi_square
    }
    /// Returns the Euclidean norm of the score `∇ℓ(β̂)` at unit dispersion.
    pub fn score_norm(&self) -> S {
        self.raw.score_norm
    }
    /// Returns the Pearson dispersion estimate `X² / (n_eff − p)`, or `None`
    /// for an operator design.
    pub fn dispersion(&self) -> Option<S> {
        self.raw.dispersion
    }
    /// Returns the residual degrees of freedom `n_eff − p`, or `None` for an
    /// operator design.
    ///
    /// The effective sample size `n_eff = Σᵢ fᵢ mᵢ` sums over the rows of
    /// positive weight, where `mᵢ` is the binomial trial count of row `i` (1
    /// without trial counts) and `fᵢ` its frequency weight (1 for other weight
    /// models). A binomial row with trial counts thus contributes its trials,
    /// not one observation.
    pub fn degrees_of_freedom(&self) -> Option<f64> {
        self.raw.degrees_of_freedom
    }
    /// Returns the numerical rank of the weighted design at the last IRLS
    /// iteration, or `None` for an operator design or an injected solver.
    pub fn rank(&self) -> Option<usize> {
        self.raw.rank
    }
}
/// Holds the report of a GLM fit and the fit, if an iterate was accepted.
///
/// ```compile_fail
/// fn forge(report: scioxide_stats::GlmReport) -> scioxide_stats::GlmResult<f64> {
///     scioxide_stats::GlmResult { fit: None, report }
/// }
/// ```
pub struct GlmResult<S: RealScalar> {
    fit: Option<GlmFit<S>>,
    report: GlmReport,
}
impl<S: RealScalar> GlmResult<S> {
    /// Returns the fit, or `None` if no iterate was accepted.
    pub fn fit(&self) -> Option<GlmFitView<'_, S>> {
        self.fit.as_ref().map(GlmFit::view)
    }
    /// Returns the termination status and counts.
    pub fn report(&self) -> &GlmReport {
        &self.report
    }
    /// Splits the result into the fit and the report without copying.
    pub fn into_parts(self) -> (Option<GlmFit<S>>, GlmReport) {
        (self.fit, self.report)
    }
}
/// Holds the output of a GLM `fit_into` call, for reuse across fits.
///
/// It holds no fit until a call accepts an iterate.
pub struct GlmOutput<S: RealScalar> {
    storage: GlmStorage<S>,
    report: Option<GlmReport>,
}
impl<S: RealScalar> GlmOutput<S> {
    /// Returns the last fit written to this output and its report, or `None`
    /// if there is none.
    pub fn fit(&self) -> Option<GlmFitView<'_, S>> {
        self.report.map(|report| GlmFitView {
            raw: &self.storage,
            report,
        })
    }
    /// Allocates an empty output for `n` rows and `p` coefficients.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds `options.max_bytes()`.
    pub fn new(n: usize, p: usize, options: GlmStorageOptions) -> Result<Self, SciOxideError> {
        let bytes = n
            .checked_mul(3)
            .and_then(|n| n.checked_add(p))
            .and_then(|n| n.checked_mul(core::mem::size_of::<S>()))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("glm_output")))?;
        crate::error_helpers::limit_bytes(bytes, options.max_bytes())?;
        Ok(Self {
            storage: GlmStorage::new(n, p)?,
            report: None,
        })
    }
}
/// Borrows a GLM fit and the report of the fit that produced it.
#[derive(Clone, Copy)]
pub struct GlmFitView<'a, S: RealScalar> {
    raw: &'a GlmStorage<S>,
    report: GlmReport,
}
impl<S: RealScalar> GlmFitView<'_, S> {
    /// Returns the report of the fit.
    pub fn report(&self) -> &GlmReport {
        &self.report
    }
    /// Returns the coefficient estimates `β̂`.
    pub fn coefficients(&self) -> &Vector<S> {
        &self.raw.coefficients
    }
    /// Returns the fitted means `μ̂ᵢ`.
    pub fn fitted(&self) -> &Vector<S> {
        &self.raw.fitted
    }
    /// Returns the response residuals `yᵢ − μ̂ᵢ`.
    pub fn residuals(&self) -> &Vector<S> {
        &self.raw.residuals
    }
    /// Returns the linear predictors `η̂ᵢ`, including offsets.
    pub fn linear_predictor(&self) -> &Vector<S> {
        &self.raw.linear_predictor
    }
    /// Returns the residual deviance.
    pub fn deviance(&self) -> S {
        self.raw.deviance
    }
    /// Returns the Pearson chi-square statistic.
    pub fn pearson_chi_square(&self) -> S {
        self.raw.pearson_chi_square
    }
    /// Returns the Euclidean norm of the score.
    pub fn score_norm(&self) -> S {
        self.raw.score_norm
    }
    /// Returns the Pearson dispersion estimate, or `None` for an operator
    /// design.
    pub fn dispersion(&self) -> Option<S> {
        self.raw.dispersion
    }
    /// Returns the residual degrees of freedom `n_eff − p` of
    /// [`GlmFit::degrees_of_freedom`], or `None` for an operator design.
    pub fn degrees_of_freedom(&self) -> Option<f64> {
        self.raw.degrees_of_freedom
    }
    /// Returns the numerical rank of the weighted design, or `None` for an
    /// operator design or an injected solver.
    pub fn rank(&self) -> Option<usize> {
        self.raw.rank
    }
}
/// Borrows the last accepted iterate of the last fit run with a workspace.
///
/// A fit that returned an error can still have accepted an iterate. The error
/// describes the outcome of that fit.
pub struct GlmAccepted<'a, S: RealScalar> {
    raw: &'a GlmStorage<S>,
    iterations: usize,
}
impl<S: RealScalar> GlmAccepted<'_, S> {
    /// Returns the number of accepted iterates.
    pub fn accepted_iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the coefficient estimates `β̂`.
    pub fn coefficients(&self) -> &Vector<S> {
        &self.raw.coefficients
    }
    /// Returns the fitted means `μ̂ᵢ`.
    pub fn fitted(&self) -> &Vector<S> {
        &self.raw.fitted
    }
    /// Returns the response residuals `yᵢ − μ̂ᵢ`.
    pub fn residuals(&self) -> &Vector<S> {
        &self.raw.residuals
    }
    /// Returns the linear predictors `η̂ᵢ`, including offsets.
    pub fn linear_predictor(&self) -> &Vector<S> {
        &self.raw.linear_predictor
    }
    /// Returns the residual deviance.
    pub fn deviance(&self) -> S {
        self.raw.deviance
    }
    /// Returns the Pearson chi-square statistic.
    pub fn pearson_chi_square(&self) -> S {
        self.raw.pearson_chi_square
    }
    /// Returns the Euclidean norm of the score.
    pub fn score_norm(&self) -> S {
        self.raw.score_norm
    }
    /// Returns the Pearson dispersion estimate, or `None` for an operator
    /// design.
    pub fn dispersion(&self) -> Option<S> {
        self.raw.dispersion
    }
    /// Returns the residual degrees of freedom `n_eff − p` of
    /// [`GlmFit::degrees_of_freedom`], or `None` for an operator design.
    pub fn degrees_of_freedom(&self) -> Option<f64> {
        self.raw.degrees_of_freedom
    }
    /// Returns the numerical rank of the weighted design, or `None` for an
    /// operator design or an injected solver.
    pub fn rank(&self) -> Option<usize> {
        self.raw.rank
    }
}

fn workspace_bytes<S: RealScalar>(
    n: usize,
    p: usize,
    provider: Option<usize>,
) -> Result<usize, SciOxideError> {
    let elements = if let Some(scratch) = provider {
        n.checked_mul(10)
            .and_then(|n| p.checked_mul(8).and_then(|p| n.checked_add(p)))
            .and_then(|n| n.checked_add(scratch))
    } else {
        n.checked_mul(p).and_then(|np| {
            n.checked_mul(16)
                .and_then(|n| p.checked_mul(12).and_then(|p| n.checked_add(p)))
                .and_then(|extra| np.checked_add(extra))
        })
    };
    elements
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_mul(core::mem::size_of::<S>()))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("glm_workspace_bytes"))
        })
}
fn classify(report: &Result<GlmReport, SciOxideError>) -> AttemptOutcome {
    match report {
        Ok(report) if report.termination == StatsTermination::Converged => {
            AttemptOutcome::Succeeded
        }
        Ok(_) => AttemptOutcome::Stopped,
        Err(_) => AttemptOutcome::Failed,
    }
}

/// Holds a dense design and its row metadata for fitting a generalized linear
/// model.
///
/// The model for row `i` with covariates `xᵢ` is
///
/// ```text
/// g(μᵢ) = ηᵢ = xᵢᵀβ + oᵢ,   E[yᵢ] = μᵢ,   Var[yᵢ] = φ V(μᵢ) / (wᵢ mᵢ),
/// ```
///
/// where `g` is the canonical link and `V` the variance function of the
/// [`GlmFamily`], `oᵢ` the offset, `wᵢ` the row weight, `mᵢ` the number of
/// binomial trials (one for other families) and `φ` the dispersion. The fit
/// maximizes the likelihood, which is the same as minimizing the deviance
/// `Σᵢ wᵢ mᵢ d(yᵢ, μᵢ)`. The model has no implicit intercept: include a column
/// of ones.
///
/// The default solver is iteratively reweighted least squares (IRLS). It
/// starts from the weighted mean of the response. Each iteration solves a
/// weighted least-squares problem by thin SVD, in O(np²) for an `n × p`
/// design, and halves the step up to 32 times until the deviance does not
/// increase. The fit converges when both the coefficient tolerance and the
/// score tolerance of [`GlmOptions`] hold. Otherwise the report gives the
/// reason it stopped:
///
/// - `StatsTermination::RankDeficient` if a weighted design has numerical rank
///   below `p`.
/// - `StatsTermination::Separated` if a binomial fit diverges toward complete
///   or quasi-complete separation.
/// - `StatsTermination::NoProgress` if no halved step reduces the deviance.
/// - `StatsTermination::BudgetExhausted` or `StatsTermination::Stopped` for an
///   iteration or evaluation limit, cancellation or a deadline.
///
/// [`GlmSolver::Injected`] minimizes the deviance with an optimizer you
/// provide instead. After a converged fit, [`Self::covariance`] and
/// [`Self::coefficient_inference`] give standard errors and Wald tests. For a
/// sparse or matrix-free design, use [`PreparedOperatorGlm`].
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{
///     GlmFamily, GlmInputs, GlmOptions, GlmStorageOptions, PreparedGlm, StatsTermination, Unweighted,
/// };
/// // An intercept-only Gaussian model estimates the mean, 2.5.
/// let design = Matrix::from_vec(4, 1, vec![1.0_f64; 4])?;
/// let plan = PreparedGlm::new(
///     design.view(),
///     GlmFamily::Gaussian,
///     Unweighted,
///     None,
///     None,
///     None,
///     GlmStorageOptions::new(),
/// )?;
/// let result = plan.fit_scoped(
///     GlmInputs::new(&[1., 2., 3., 4.], GlmOptions::default()),
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(result.report().termination, StatsTermination::Converged);
/// assert!((result.fit().unwrap().coefficients().as_slice()[0] - 2.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Reliability weights define no likelihood and do not compile:
///
/// ```compile_fail
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{GlmFamily, GlmStorageOptions, PreparedGlm, Reliability, Weights};
/// let design = Matrix::from_vec(4, 1, vec![1.0_f64; 4]).unwrap();
/// let weights = [1.0, 2.0, 1.0, 2.0];
/// PreparedGlm::new(design.view(), GlmFamily::Gaussian, Weights::<_, Reliability>::new(&weights),
///     None, None, None, GlmStorageOptions::new());
/// ```
#[derive(Clone)]
pub struct PreparedGlm<S: RealScalar> {
    pub(super) raw: DenseGlm<S>,
    options: GlmStorageOptions,
}
/// Holds the scratch storage of [`PreparedGlm::fit_into`] and a record of the
/// last fit.
///
/// Create it with [`PreparedGlm::workspace`].
pub struct GlmWorkspace<S: RealScalar> {
    raw: GlmScratch<S>,
    slot: SolveSlot<usize>,
    retained_bytes: usize,
    injected: FitWorkspace<S>,
}
impl<S: RealScalar> GlmWorkspace<S> {
    /// Returns the record of the last fit run with this workspace, including a
    /// fit rejected by input checks.
    pub fn last_attempt(&self) -> Option<&AttemptRecord<usize>> {
        self.slot.last_attempt()
    }
    /// Returns the last accepted iterate of the last fit, if it accepted one.
    pub fn last_accepted(&self) -> Option<GlmAccepted<'_, S>> {
        self.slot
            .last_attempt()
            .and_then(AttemptRecord::progress)
            .map(|&iterations| GlmAccepted {
                raw: &self.raw.irls.stage,
                iterations,
            })
    }
}
impl<S: RealScalar> PreparedGlm<S> {
    /// Copies a dense design and its row metadata.
    ///
    /// `weights` is [`crate::Unweighted`], [`crate::Frequency`] (integer
    /// replication counts) or [`crate::Analytic`] (relative inverse variances).
    /// `trials` gives the number of trials of each binomial row and defaults
    /// to one. `offsets` gives `oᵢ` and defaults to zero. `exposure` is for the
    /// Poisson family: its logarithm is added to the offset.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the design has no rows or no columns, or fewer rows
    ///   than columns.
    /// - `InvalidInput` if `weights`, `trials`, `offsets` or `exposure` does
    ///   not have one entry per row, `trials` is given for a family other than
    ///   binomial, or `exposure` for a family other than Poisson.
    /// - `InvalidInput` if a weight is negative or not finite, or a frequency
    ///   weight is not an integer.
    /// - `InvalidInput` if a trial count is zero, exceeds 2⁵³, or is not exact
    ///   in `S`.
    /// - `InvalidInput` if an offset is not finite, or an exposure is not
    ///   positive and finite.
    /// - `InvalidInput` if the effective sample size `n_eff` is at most the
    ///   number of columns, or exceeds 2⁵³.
    /// - `NonFinite` or `NumericalInstability` if the design contains a NaN or
    ///   an infinite entry.
    /// - `ResourceLimit` if the copy exceeds `options.max_bytes()`.
    pub fn new<'w>(
        design: MatrixView<'_, S>,
        family: GlmFamily,
        weights: impl RegressionWeights<'w, S>,
        trials: Option<&[u64]>,
        offsets: Option<&[S]>,
        exposure: Option<&[S]>,
        options: GlmStorageOptions,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            raw: DenseGlm::new(
                design,
                family,
                weights.rows(),
                trials,
                offsets,
                exposure,
                options.max_bytes(),
            )?,
            options,
        })
    }
    /// Allocates an empty output for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds the plan's byte limit.
    pub fn output(&self) -> Result<GlmOutput<S>, SciOxideError> {
        let (n, p) = self.shape();
        crate::error_helpers::limit_bytes(
            n.checked_mul(3)
                .and_then(|v| v.checked_add(p))
                .and_then(|v| v.checked_mul(core::mem::size_of::<S>()))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("glm_output"))
                })?,
            self.options.max_bytes(),
        )?;
        GlmOutput::new(n, p, self.options)
    }
    /// Allocates the workspace for [`Self::fit_into`], of size O(np).
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the smaller of
    ///   `options.max_bytes()` and the plan's byte limit.
    pub fn workspace(&self, options: GlmOptions<S>) -> Result<GlmWorkspace<S>, SciOxideError> {
        let max_bytes = options.max_bytes.min(self.options.max_bytes());
        Ok(GlmWorkspace {
            raw: self.raw.workspace(max_bytes)?,
            slot: SolveSlot::default(),
            retained_bytes: workspace_bytes::<S>(self.shape().0, self.shape().1, None)?,
            injected: FitWorkspace::new(
                self.shape().1,
                crate::StatisticsOptions::new().with_max_bytes(max_bytes),
            )?,
        })
    }
    fn shape(&self) -> (usize, usize) {
        (self.raw.design().rows(), self.raw.design().cols())
    }
    /// Fits one response and writes the last accepted iterate to `output`.
    ///
    /// With [`GlmSolver::Irls`] the fit runs IRLS as described on
    /// [`PreparedGlm`]. With [`GlmSolver::Injected`] your optimizer minimizes
    /// the [`GlmObjective`], and its final point is checked and counts as one
    /// accepted iterate. Each iteration counts toward the iteration limit of
    /// `inputs` and of `scope`. A limit, cancellation or deadline ends the fit
    /// with `Ok` and an unsuccessful termination. In that case `output` holds
    /// the last accepted iterate, or is unchanged if there is none. See
    /// [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the response does not have one value per row, or a
    ///   value lies outside the family's domain.
    /// - `InvalidInput` if `output` or `work` does not match the design shape.
    /// - `ResourceLimit` if `work` exceeds the byte limit of `inputs`.
    /// - `InvalidInput` if the final point of an injected solver gives a mean
    ///   outside the family's domain.
    /// - The error of the SVD or of the injected solver.
    pub fn fit_into<'scope, 'capability>(
        &self,
        inputs: GlmInputs<'_, S>,
        output: &mut GlmOutput<S>,
        work: &mut GlmWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<GlmReport, SciOxideError> {
        work.slot.reject();
        let (n, p) = self.shape();
        if inputs.response.len() != n
            || inputs
                .response
                .iter()
                .any(|&value| !self.raw.family().valid_response(value))
            || !output.storage.has_shape(n, p)
            || !work.raw.irls.stage.has_shape(n, p)
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM response or output shape",
            ));
        }
        crate::error_helpers::limit_bytes(work.retained_bytes, inputs.options.max_bytes)?;
        let scope = scope.into();
        let identity = self
            .raw
            .solve_identity(scioxide_core::ProblemToken::new(0))?;
        work.raw.irls.accepted_iterations = 0;
        let accepted = core::cell::Cell::new(0usize);
        let mut slot = core::mem::take(&mut work.slot);
        let result = slot.run(
            scope,
            identity,
            SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, inputs.options.iterations),
            |context| {
                let result = (|| {
                    match inputs.solver {
                        GlmSolver::Irls => self.raw.fit_into(
                            inputs.response,
                            inputs.options,
                            &mut output.storage,
                            &mut work.raw,
                            context,
                        ),
                        GlmSolver::Injected {
                            solver,
                            initial,
                            bounds,
                            integrality,
                        } => {
                            let objective = GlmObjective::new(self, inputs.response)?;
                            let report = crate::fit_into(
                                FitRequest {
                                    objective: &objective,
                                    derivatives: Some(&objective),
                                    initial,
                                    bounds,
                                    integrality,
                                },
                                solver,
                                work.raw.irls.candidate.coefficients.as_mut_slice(),
                                &mut work.injected,
                                &mut *context,
                            )?;
                            let termination = report.termination();
                            let accepted = if report.objective().is_some() {
                                if let Some(stop) = context.stop_reason()? {
                                    return Ok(GlmReport {
                                        termination: StatsTermination::Stopped(stop),
                                        accepted_iterations: 0,
                                        counts: context.counts(),
                                    });
                                }
                                if !context
                                    .record_provider(scioxide_core::EvaluationRole::Function)?
                                    .is_granted()
                                {
                                    return Ok(GlmReport {
                                        termination: StatsTermination::BudgetExhausted,
                                        accepted_iterations: 0,
                                        counts: context.counts(),
                                    });
                                }
                                if !self
                                    .raw
                                    .assess(inputs.response, &mut work.raw.irls.candidate)?
                                {
                                    return Err(invalid(
                                        scioxide_core::argument!("solver"),
                                        scioxide_core::InvalidInputReason::OutOfRange,
                                        "injected GLM candidate outside family domain",
                                    ));
                                }
                                if let Some(stop) = context.stop_reason()? {
                                    return Ok(GlmReport {
                                        termination: StatsTermination::Stopped(stop),
                                        accepted_iterations: 0,
                                        counts: context.counts(),
                                    });
                                }
                                work.raw.irls.candidate.rank = None;
                                // Assess rank only for dense inference; a generic optimizer makes no rank claim.
                                context.commit_accepted(
                                    scioxide_core::PartialResultTrust::ResidualChecked,
                                )?;
                                work.raw.irls.stage.copy_from(&work.raw.irls.candidate)?;
                                work.raw.irls.accepted_iterations = 1;
                                output.storage.copy_from(&work.raw.irls.candidate)?;
                                1
                            } else {
                                0
                            };
                            Ok(GlmReport {
                                termination,
                                accepted_iterations: accepted,
                                counts: context.counts(),
                            })
                        }
                    }
                })();
                accepted.set(work.raw.irls.accepted_iterations);
                result
            },
            |result| {
                (
                    classify(result),
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
    /// Fits one response and returns a new result.
    ///
    /// Allocates the output and the workspace. Use [`Self::fit_into`] to reuse
    /// them.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::output`], [`Self::workspace`] and
    /// [`Self::fit_into`].
    pub fn fit_scoped<'scope, 'capability>(
        &self,
        inputs: GlmInputs<'_, S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<GlmResult<S>, SciOxideError> {
        let mut output = self.output()?;
        let mut work = self.workspace(inputs.options)?;
        let report = self.fit_into(inputs, &mut output, &mut work, scope)?;
        Ok(GlmResult {
            fit: (report.accepted_iterations > 0).then_some(GlmFit {
                raw: output.storage,
                report,
            }),
            report,
        })
    }
    /// Executes [`Self::fit_scoped`] with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the scoped form's input, numerical and allocation errors.
    pub fn fit(&self, inputs: GlmInputs<'_, S>) -> Result<GlmResult<S>, SciOxideError> {
        self.fit_scoped(inputs, scioxide_core::ExecutionContext::sequential_native())
    }

    /// Returns the family, which also determines the link.
    pub fn family(&self) -> GlmFamily {
        self.raw.family()
    }
    /// Returns the copied design.
    pub fn design(&self) -> &Matrix<S> {
        self.raw.design()
    }
    /// Computes the covariance matrix of the coefficients of a converged fit.
    ///
    /// `convention` selects the estimator (see [`GlmCovariance`]). The method
    /// recomputes the fit from its coefficients and checks that it matches
    /// `response`. The sandwich estimators treat each row as one independent
    /// observation. The cost is O(n²p + np²) time and O(np) storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the fit did not converge, or has no rank because it
    ///   comes from an injected solver.
    /// - `InvalidInput` if the weighted design is rank deficient.
    /// - `InvalidInput` if the fit does not match `response`.
    /// - `InvalidInput` for [`GlmCovariance::Hc0`] or [`GlmCovariance::Hc3`]
    ///   with frequency weights other than 0 and 1. Expand the rows instead.
    /// - `InvalidInput` for [`GlmCovariance::Hc3`] if a row has leverage 1.
    pub fn covariance(
        &self,
        response: &[S],
        fit: &GlmFitView<'_, S>,
        convention: GlmCovariance,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.raw
            .covariance(response, fit.raw, &fit.report, convention)
    }
    /// Returns the Wald test and confidence interval for each coefficient of a
    /// converged fit.
    ///
    /// See [`GlmCoefficientInference`] for the definitions.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::covariance`].
    /// - `NumericalInstability` if a standard error or an interval endpoint
    ///   overflows `S`.
    pub fn coefficient_inference(
        &self,
        response: &[S],
        fit: &GlmFitView<'_, S>,
        convention: GlmCovariance,
        confidence: ConfidenceProbability,
    ) -> Result<Vec<GlmCoefficientInference<S>>, SciOxideError> {
        self.raw
            .coefficient_inference(response, fit.raw, &fit.report, convention, confidence)
    }
}

/// Holds a borrowed sparse, dense or matrix-free design and its row metadata
/// for fitting a generalized linear model by IRLS.
///
/// The model and the IRLS iteration are those of [`PreparedGlm`], but each
/// weighted least-squares problem is solved by LSMR from the previous
/// coefficients, using only products with the design and its adjoint. Each
/// LSMR solve runs in a child scope of the fit and is limited by
/// [`GlmOptions::factor_iterations`]. LSMR does not determine the rank, so the
/// fit reports no rank, dispersion or degrees of freedom, and gives no
/// covariance. Injected solvers are not supported.
///
/// The plan is `Clone`, and `Send + Sync` when `O` is `Sync`.
pub struct PreparedOperatorGlm<'a, S: RealScalar, O: ?Sized> {
    raw: OperatorGlm<'a, S, O>,
    options: GlmStorageOptions,
}
impl<S: RealScalar, O: ?Sized> Clone for PreparedOperatorGlm<'_, S, O> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            options: self.options,
        }
    }
}
/// Holds the scratch storage of [`PreparedOperatorGlm::fit_into`] and a record
/// of the last fit.
///
/// Create it with [`PreparedOperatorGlm::workspace`].
pub struct GlmOperatorWorkspace<S: RealScalar> {
    raw: GlmOperatorScratch<S>,
    slot: SolveSlot<usize>,
    retained_bytes: usize,
}
impl<S: RealScalar> GlmOperatorWorkspace<S> {
    /// Returns the record of the last fit run with this workspace, including a
    /// fit rejected by input checks.
    pub fn last_attempt(&self) -> Option<&AttemptRecord<usize>> {
        self.slot.last_attempt()
    }
    /// Returns the last accepted iterate of the last fit, if it accepted one.
    pub fn last_accepted(&self) -> Option<GlmAccepted<'_, S>> {
        self.slot
            .last_attempt()
            .and_then(AttemptRecord::progress)
            .map(|&iterations| GlmAccepted {
                raw: &self.raw.irls.stage,
                iterations,
            })
    }
}
impl<'a, S: RealScalar, O: scioxide_core::AdjointOperator<S> + ?Sized>
    PreparedOperatorGlm<'a, S, O>
{
    /// Borrows a design operator and copies its row metadata.
    ///
    /// `weights`, `trials`, `offsets` and `exposure` are as for
    /// [`PreparedGlm::new`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operator has no rows or no columns.
    /// - `InvalidInput` for invalid row metadata, as for [`PreparedGlm::new`].
    /// - `InvalidInput` if no row has positive weight.
    /// - `ResourceLimit` if the metadata exceeds `options.max_bytes()`.
    pub fn new<'w>(
        design: &'a O,
        family: GlmFamily,
        weights: impl RegressionWeights<'w, S>,
        trials: Option<&[u64]>,
        offsets: Option<&[S]>,
        exposure: Option<&[S]>,
        options: GlmStorageOptions,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            raw: OperatorGlm::new(
                design,
                family,
                weights.rows(),
                trials,
                offsets,
                exposure,
                options.max_bytes(),
            )?,
            options,
        })
    }
    /// Allocates an empty output for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds the plan's byte limit.
    pub fn output(&self) -> Result<GlmOutput<S>, SciOxideError> {
        let (n, p) = self.shape();
        crate::error_helpers::limit_bytes(
            n.checked_mul(3)
                .and_then(|v| v.checked_add(p))
                .and_then(|v| v.checked_mul(core::mem::size_of::<S>()))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("glm_output"))
                })?,
            self.options.max_bytes(),
        )?;
        GlmOutput::new(n, p, self.options)
    }
    /// Allocates the workspace for [`Self::fit_into`], of size O(n + p + s),
    /// where `s` is the scratch length the operator requests.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the smaller of
    ///   `options.max_bytes()` and the plan's byte limit.
    /// - The error of the operator's scratch-length query.
    pub fn workspace(
        &self,
        options: GlmOptions<S>,
    ) -> Result<GlmOperatorWorkspace<S>, SciOxideError> {
        let max_bytes = options.max_bytes.min(self.options.max_bytes());
        Ok(GlmOperatorWorkspace {
            raw: self.raw.workspace(max_bytes)?,
            slot: SolveSlot::default(),
            retained_bytes: workspace_bytes::<S>(
                self.shape().0,
                self.shape().1,
                Some(
                    self.raw
                        .operator
                        .apply_workspace_len()?
                        .max(self.raw.operator.apply_adjoint_workspace_len()?),
                ),
            )?,
        })
    }
    fn shape(&self) -> (usize, usize) {
        (self.raw.operator.rows(), self.raw.operator.cols())
    }
    /// Fits one response by IRLS and writes the last accepted iterate to
    /// `output`.
    ///
    /// Limits, cancellation and deadlines behave as for
    /// [`PreparedGlm::fit_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the response does not have one value per row, or a
    ///   value lies outside the family's domain.
    /// - `InvalidInput` if `output` or `work` does not match the operator shape.
    /// - `ResourceLimit` if `work` exceeds the byte limit of `inputs`.
    /// - `NotImplemented` if the solver is [`GlmSolver::Injected`].
    /// - The error of the operator.
    pub fn fit_into<'scope, 'capability>(
        &self,
        inputs: GlmInputs<'_, S>,
        output: &mut GlmOutput<S>,
        work: &mut GlmOperatorWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<GlmReport, SciOxideError> {
        work.slot.reject();
        let (n, p) = self.shape();
        if inputs.response.len() != n
            || inputs
                .response
                .iter()
                .any(|&value| !self.raw.family.valid_response(value))
            || !output.storage.has_shape(n, p)
            || !work.raw.irls.stage.has_shape(n, p)
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM response or output shape",
            ));
        }
        crate::error_helpers::limit_bytes(work.retained_bytes, inputs.options.max_bytes)?;
        let scope = scope.into();
        if !matches!(inputs.solver, GlmSolver::Irls) {
            return Err(SciOxideError::not_implemented(
                "injected operator GLM solver",
            ));
        }
        let identity = self
            .raw
            .solve_identity(scioxide_core::ProblemToken::new(0))?;
        work.raw.irls.accepted_iterations = 0;
        let accepted = core::cell::Cell::new(0usize);
        let mut slot = core::mem::take(&mut work.slot);
        let result = slot.run(
            scope,
            identity,
            SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, inputs.options.iterations),
            |context| {
                let result = self.raw.fit_into(
                    inputs.response,
                    inputs.options,
                    &mut output.storage,
                    &mut work.raw,
                    context,
                );
                accepted.set(work.raw.irls.accepted_iterations);
                result
            },
            |result| {
                (
                    classify(result),
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
    /// Fits one response and returns a new result.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::output`], [`Self::workspace`] and
    /// [`Self::fit_into`].
    pub fn fit_scoped<'scope, 'capability>(
        &self,
        inputs: GlmInputs<'_, S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<GlmResult<S>, SciOxideError> {
        let mut output = self.output()?;
        let mut work = self.workspace(inputs.options)?;
        let report = self.fit_into(inputs, &mut output, &mut work, scope)?;
        Ok(GlmResult {
            fit: (report.accepted_iterations > 0).then_some(GlmFit {
                raw: output.storage,
                report,
            }),
            report,
        })
    }
    /// Executes [`Self::fit_scoped`] with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the scoped form's input, numerical and allocation errors.
    pub fn fit(&self, inputs: GlmInputs<'_, S>) -> Result<GlmResult<S>, SciOxideError> {
        self.fit_scoped(inputs, scioxide_core::ExecutionContext::sequential_native())
    }
}

/// Holds the inputs of [`glm_batch_into`]: one dense model and several
/// responses fitted by IRLS.
pub struct GlmBatchInputs<'a, S: RealScalar> {
    /// Model shared by all fits.
    pub model: &'a PreparedGlm<S>,
    /// Responses, in the order of the outputs.
    pub responses: &'a [&'a [S]],
    /// Options of each fit.
    pub options: GlmOptions<S>,
    /// Execution policy, and the byte limit of the batch workspace.
    pub batch_options: StatisticsOptions,
}
/// Holds the output and report of one response in [`glm_batch_into`].
pub struct GlmBatchOutput<S: RealScalar> {
    /// Output of the fit. Create it with [`PreparedGlm::output`].
    pub output: GlmOutput<S>,
    /// Report of the fit in the last successful batch call, or `None` before
    /// the first one.
    pub report: Option<GlmReport>,
}
/// Holds one [`GlmWorkspace`] per worker and one staging output per response
/// for [`glm_batch_into`].
///
/// Create it with [`PreparedGlm::batch_workspace`].
pub struct GlmBatchWorkspace<S: RealScalar> {
    workers: Vec<GlmWorkspace<S>>,
    staging: Vec<(GlmOutput<S>, Option<GlmReport>)>,
    retained_bytes: usize,
}
impl<S: RealScalar> PreparedGlm<S> {
    /// Allocates a batch workspace for `jobs` responses and `workers` workers.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the smaller of
    ///   `options.max_bytes()` and the plan's byte limit.
    pub fn batch_workspace(
        &self,
        jobs: usize,
        workers: usize,
        options: GlmOptions<S>,
    ) -> Result<GlmBatchWorkspace<S>, SciOxideError> {
        let workers = if jobs == 0 { 0 } else { workers };
        let worker_bytes = workspace_bytes::<S>(self.shape().0, self.shape().1, None)?
            .checked_add(core::mem::size_of::<GlmWorkspace<S>>())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("glm_batch_bytes"))
            })?;
        let stage_bytes = self
            .shape()
            .0
            .checked_mul(3)
            .and_then(|n| n.checked_add(self.shape().1))
            .and_then(|n| n.checked_mul(core::mem::size_of::<S>()))
            .and_then(|n| n.checked_add(core::mem::size_of::<(GlmOutput<S>, Option<GlmReport>)>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("glm_batch_bytes"))
            })?;
        let retained_bytes = worker_bytes
            .checked_mul(workers)
            .and_then(|n| stage_bytes.checked_mul(jobs).and_then(|m| n.checked_add(m)))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("glm_batch_bytes"))
            })?;
        crate::error_helpers::limit_bytes(
            retained_bytes,
            options.max_bytes.min(self.options.max_bytes()),
        )?;

        let mut work = scioxide_core::backend::try_with_capacity(
            workers,
            scioxide_core::quantity!("glm_batch_workers"),
        )?;
        for _ in 0..workers {
            work.push(self.workspace(options)?);
        }
        let mut staging = scioxide_core::backend::try_with_capacity(
            jobs,
            scioxide_core::quantity!("glm_batch_staging"),
        )?;
        for _ in 0..jobs {
            staging.push((self.output()?, None));
        }
        Ok(GlmBatchWorkspace {
            workers: work,
            staging,
            retained_bytes,
        })
    }
}
/// Fits several responses to one dense GLM by IRLS, in parallel when the
/// execution context allows it.
///
/// `outputs[k]` receives the fit of `inputs.responses[k]`. A fit that ends
/// without an accepted iterate leaves its output unchanged but still sets its
/// report. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Errors
///
/// - `ResourceLimit` if `workspace` exceeds `inputs.batch_options.max_bytes()`.
/// - `InvalidInput` if `outputs`, the responses and the staging outputs of
///   `workspace` differ in number, or a response or output does not match the
///   design shape.
/// - The errors of [`PreparedGlm::fit_into`] for any response.
/// - The error of the executor.
pub fn glm_batch_into<S: RealScalar>(
    inputs: GlmBatchInputs<'_, S>,
    outputs: &mut [GlmBatchOutput<S>],
    workspace: &mut GlmBatchWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    crate::error_helpers::limit_bytes(workspace.retained_bytes, inputs.batch_options.max_bytes())?;

    let model = inputs.model;
    let (n, p) = model.shape();
    if outputs.len() != inputs.responses.len()
        || workspace.staging.len() != outputs.len()
        || inputs.responses.iter().any(|r| r.len() != n)
        || outputs.iter().any(|o| !o.output.storage.has_shape(n, p))
    {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "GLM batch shape",
        ));
    }
    crate::estimation::execute(
        inputs.batch_options,
        context,
        outputs.len(),
        1,
        &mut workspace.workers,
        &mut workspace.staging,
        |job, work, slot| {
            let (out, report) = &mut slot[0];
            *report = Some(model.fit_into(
                GlmInputs::new(inputs.responses[job], inputs.options),
                out,
                work,
                context,
            )?);
            Ok(())
        },
    )?;
    context.check()?;
    for (out, (stage, report)) in outputs.iter_mut().zip(&workspace.staging) {
        if let Some(report) = report {
            if report.accepted_iterations > 0 {
                out.output.storage.copy_from(&stage.storage)?;
                out.output.report = Some(*report);
            }
            out.report = Some(*report);
        }
    }
    Ok(())
}

/// Fits an unweighted generalized linear model with default IRLS options.
///
/// Uses [`PreparedGlm::fit`] and returns the converged model.
///
/// # Errors
///
/// - The errors of [`PreparedGlm::new`] and [`PreparedGlm::fit`].
/// - `Singular` if a weighted design is numerically rank deficient.
/// - `ConvergenceFailure` with `Divergence` if a binomial fit separates, `Stagnation` if no
///   halved step reduces the deviance, and `IterationLimit` if the iteration limit is reached
///   or no iterate is accepted.
pub fn glm_fit<S: RealScalar>(
    design: MatrixView<'_, S>,
    family: GlmFamily,
    response: &[S],
) -> Result<GlmFit<S>, SciOxideError> {
    let result = PreparedGlm::new(
        design,
        family,
        crate::Unweighted,
        None,
        None,
        None,
        GlmStorageOptions::new(),
    )?
    .fit(GlmInputs::new(response, GlmOptions::default()))?;
    let (fit, report) = result.into_parts();
    if let Some(error) = report.termination.one_shot_error(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts),
    ) {
        return Err(error);
    }
    fit.ok_or_else(|| {
        SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            Default::default(),
            "no accepted GLM iterate",
        )
    })
}
