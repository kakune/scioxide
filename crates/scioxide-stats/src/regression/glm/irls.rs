//! Iteratively reweighted least squares for dense and operator GLM designs.
use super::family::nonnegative_deviance;
use super::{GlmFamily, GlmOptions, GlmReport};
use crate::StatsTermination;
use crate::regression::{finite, invalid, number};
use crate::weights::RegressionRows;
use scioxide_core::{
    ProblemToken, ProductSum, RealScalar, Scaled, ScaledAccumulator, SciOxideError, SolveContext,
    SolveIdentity,
};
use scioxide_linalg::Vector;

/// Stores the checked weights, binomial trials and offsets of each row.
#[derive(Clone)]
pub(super) struct GlmRows<S: RealScalar> {
    pub(super) weights: Vector<S>,
    pub(super) trials: Vector<S>,
    pub(super) offsets: Vector<S>,
    /// Effective sample size: the sum of weight times trials for frequency
    /// weights, otherwise the sum of trials over rows of positive weight.
    pub(super) units: f64,
    /// Whether weights are exact replication counts.
    pub(super) frequency: bool,
}
impl<S: RealScalar> GlmRows<S> {
    /// Copies the metadata of `n` rows. The caller has checked the lengths and
    /// the family. `trials` default to one, and the logarithm of the exposure
    /// is added to the offset.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for an invalid weight, trial count, offset or
    /// exposure.
    pub(super) fn new(
        n: usize,
        weights: RegressionRows<'_, S>,
        trials: Option<&[u64]>,
        offsets: Option<&[S]>,
        exposure: Option<&[S]>,
    ) -> Result<Self, SciOxideError> {
        let mut row_weights = Vector::zeros(n)?;
        let mut row_trials = Vector::zeros(n)?;
        let mut row_offsets = Vector::zeros(n)?;
        let mut units = 0.0;
        for i in 0..n {
            let w = weights.weight(i)?;
            let t = trials.map_or(1, |v| v[i]);
            if t == 0 || t > 9_007_199_254_740_992 {
                return Err(invalid(
                    scioxide_core::argument!("trials"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "invalid binomial trial count",
                ));
            }
            let tr = S::try_from_i64_exact(t as i64)?;
            if tr.try_to_f64_exact()? != t as f64 {
                return Err(invalid(
                    scioxide_core::argument!("trials"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "binomial trial count is not exact in the scalar width",
                ));
            }
            let mut offset = offsets.map_or(S::ZERO, |v| v[i]);
            if !offset.is_finite() {
                return Err(invalid(
                    scioxide_core::argument!("offsets"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "non-finite GLM offset",
                ));
            }
            if let Some(values) = exposure {
                let e = values[i];
                if !e.is_finite() || e <= S::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("exposure"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "exposure must be finite and positive",
                    ));
                }
                offset = finite(offset + e.ln())?;
            }
            row_weights.view_mut().set(i, w)?;
            row_trials.view_mut().set(i, tr)?;
            row_offsets.view_mut().set(i, offset)?;
            let count = if weights.is_frequency() {
                w.try_to_f64_exact()? * t as f64
            } else if w > S::ZERO {
                t as f64
            } else {
                0.0
            };
            units = finite(units + count)?;
        }
        Ok(Self {
            weights: row_weights,
            trials: row_trials,
            offsets: row_offsets,
            units,
            frequency: weights.is_frequency(),
        })
    }
    /// Returns the weight of row `i` times its number of trials.
    pub(super) fn weight(&self, i: usize) -> Result<S, SciOxideError> {
        finite(self.weights.view().get(i)? * self.trials.view().get(i)?)
    }
}
/// Returns whether the row metadata has `n` rows and suits `family`.
pub(super) fn rows_match<S: RealScalar>(
    n: usize,
    family: GlmFamily,
    weights: RegressionRows<'_, S>,
    trials: Option<&[u64]>,
    offsets: Option<&[S]>,
    exposure: Option<&[S]>,
) -> bool {
    weights.matches(n)
        && trials.is_none_or(|v| v.len() == n && family == GlmFamily::Binomial)
        && offsets.is_none_or(|v| v.len() == n)
        && exposure.is_none_or(|v| v.len() == n && family == GlmFamily::Poisson)
}
pub(super) fn assess_rows<S: RealScalar>(
    family: GlmFamily,
    rows: &GlmRows<S>,
    response: &[S],
    fit: &mut GlmStorage<S>,
    mut score_rhs: Option<&mut [S]>,
) -> Result<bool, SciOxideError> {
    let mut deviance = ScaledAccumulator::ZERO;
    let mut pearson = ProductSum::new();
    for (i, &y) in response.iter().enumerate() {
        let eta = finite(fit.linear_predictor.view().get(i)? + rows.offsets.view().get(i)?)?;
        let Some((mu, _, variance)) = family.mean(eta) else {
            return Ok(false);
        };
        let r = finite(y - mu)?;
        let weight = rows.weight(i)?;
        fit.linear_predictor.view_mut().set(i, eta)?;
        fit.fitted.view_mut().set(i, mu)?;
        fit.residuals.view_mut().set(i, r)?;
        deviance.add_product(Scaled::from_value(weight)?, family.deviance(y, mu)?)?;
        pearson.add_product(weight, r * r / variance)?;
        if let Some(rhs) = score_rhs.as_deref_mut() {
            rhs[i] = family
                .score(y, mu)?
                .mul(Scaled::from_value(weight)?)?
                .round()?;
        }
    }
    fit.deviance = nonnegative_deviance(deviance.total()?)?.round()?;
    fit.pearson_chi_square = pearson.finish()?.max(S::ZERO);
    Ok(true)
}
fn initial_admission(
    context: &mut SolveContext<'_, '_>,
) -> Result<Option<StatsTermination>, SciOxideError> {
    if let Some(stop) = context.stop_reason()? {
        return Ok(Some(StatsTermination::Stopped(stop)));
    }
    if !context
        .can_consume(scioxide_core::BudgetKind::Iteration, 1)?
        .is_granted()
        || !context
            .can_consume(scioxide_core::BudgetKind::Evaluation, 1)?
            .is_granted()
    {
        return Ok(Some(StatsTermination::BudgetExhausted));
    }
    Ok(None)
}

/// Returns the square root of the IRLS weight of row `i` at `eta`, and the
/// working response multiplied by it.
///
/// # Errors
///
/// Returns `InvalidInput` if the mean leaves the family's domain, and an
/// error if the arithmetic overflows.
pub(super) fn working_row<S: RealScalar>(
    family: GlmFamily,
    rows: &GlmRows<S>,
    i: usize,
    y: S,
    eta: S,
) -> Result<(S, S), SciOxideError> {
    let (mu, derivative, variance) = family.mean(eta).ok_or_else(|| {
        invalid(
            scioxide_core::argument!("response"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "GLM accepted mean left domain",
        )
    })?;
    let root = finite((rows.weight(i)? * derivative * derivative / variance).sqrt())?;
    let z = finite(eta + (y - mu) / derivative - rows.offsets.view().get(i)?)?;
    Ok((root, finite(root * z)?))
}

/// Stores the current, proposed and accepted iterates of IRLS.
pub(super) struct IrlsBuffers<S: RealScalar> {
    previous: Vector<S>,
    proposed: Vector<S>,
    pub(super) stage: GlmStorage<S>,
    pub(super) candidate: GlmStorage<S>,
    pub(super) accepted_iterations: usize,
}
impl<S: RealScalar> IrlsBuffers<S> {
    pub(super) fn new(n: usize, p: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            previous: Vector::zeros(p)?,
            proposed: Vector::zeros(p)?,
            stage: GlmStorage::new(n, p)?,
            candidate: GlmStorage::new(n, p)?,
            accepted_iterations: 0,
        })
    }
}

/// Describes the weighted working problem of one IRLS iteration.
pub(super) struct WorkingProblem<'b, S: RealScalar> {
    initial_eta: Option<S>,
    accepted: &'b Vector<S>,
    /// Coefficients of the last accepted iterate, the starting point of each
    /// solve.
    pub(super) previous: &'b Vector<S>,
    /// Destination of the proposed coefficients.
    pub(super) proposed: &'b mut Vector<S>,
}
impl<S: RealScalar> WorkingProblem<'_, S> {
    /// Returns the linear predictor of row `i`: the initial predictor on the
    /// first iteration, then that of the accepted iterate.
    pub(super) fn eta(&self, i: usize) -> Result<S, SciOxideError> {
        match self.initial_eta {
            Some(eta) => Ok(eta),
            None => self.accepted.view().get(i),
        }
    }
}

/// Result of one weighted least-squares solve.
pub(super) enum Proposal {
    /// The proposed coefficients are ready, with the numerical rank if known.
    Ready(Option<usize>),
    /// The fit ends with this termination before the step search.
    Stop(StatsTermination),
}

/// Solves the weighted problem and evaluates candidates for [`irls`].
pub(super) trait IrlsDesign<S: RealScalar> {
    /// Scratch storage of the design.
    type Scratch;
    /// Returns the family.
    fn family(&self) -> GlmFamily;
    /// Returns the row metadata.
    fn rows(&self) -> &GlmRows<S>;
    /// Solves the weighted working problem into `problem.proposed`.
    fn propose(
        &self,
        response: &[S],
        problem: WorkingProblem<'_, S>,
        options: GlmOptions<S>,
        scratch: &mut Self::Scratch,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Proposal, SciOxideError>;
    /// Evaluates candidate coefficients. Returns `None` when the evaluation
    /// limit is exhausted.
    fn assess(
        &self,
        response: &[S],
        candidate: &mut GlmStorage<S>,
        scratch: &mut Self::Scratch,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<bool>, SciOxideError>;
}

/// Runs IRLS from the weighted mean of the response.
///
/// Each step is halved up to 32 times until the deviance does not increase.
/// The iteration stops when both the coefficient and score tolerances hold, or
/// reports separation for a binomial model whose fitted means reach 0 or 1.
///
/// # Errors
///
/// Returns the errors of the design, the operator, the solve context and the
/// arithmetic.
pub(super) fn irls<S: RealScalar, D: IrlsDesign<S>>(
    design: &D,
    response: &[S],
    options: GlmOptions<S>,
    buffers: &mut IrlsBuffers<S>,
    scratch: &mut D::Scratch,
    context: &mut SolveContext<'_, '_>,
) -> Result<StatsTermination, SciOxideError> {
    if let Some(termination) = initial_admission(context)? {
        return Ok(termination);
    }
    let family = design.family();
    let p = buffers.stage.coefficients.len();
    buffers.stage.coefficients.as_mut_slice().fill(S::ZERO);
    let mut total = ProductSum::new();
    let mut count = ProductSum::new();
    for (i, &y) in response.iter().enumerate() {
        let weight = design.rows().weight(i)?;
        total.add_product(weight, y)?;
        count.add_product(weight, S::ONE)?;
    }
    let mut initial = finite(total.finish()? / count.finish()?)?;
    let edge = number::<S>(4.0)? * S::SQRT_EPSILON;
    initial = match family {
        GlmFamily::Binomial => initial.clamp(edge, S::ONE - edge),
        GlmFamily::Poisson | GlmFamily::Gamma | GlmFamily::InverseGaussian => initial.max(edge),
        GlmFamily::Gaussian => initial,
    };
    let initial_eta = family.link_value(initial)?;
    let mut termination = StatsTermination::BudgetExhausted;
    for iteration in 0..options.iterations.get() {
        if let Some(stop) = context.stop_reason()? {
            termination = StatsTermination::Stopped(stop);
            break;
        }
        if !context.record_iteration()?.is_granted() {
            break;
        }
        buffers
            .previous
            .as_mut_slice()
            .copy_from_slice(buffers.stage.coefficients.as_slice());
        let problem = WorkingProblem {
            initial_eta: (iteration == 0).then_some(initial_eta),
            accepted: &buffers.stage.linear_predictor,
            previous: &buffers.previous,
            proposed: &mut buffers.proposed,
        };
        let rank = match design.propose(response, problem, options, scratch, context)? {
            Proposal::Ready(rank) => rank,
            Proposal::Stop(stop) => {
                termination = stop;
                break;
            }
        };
        let mut step = S::ONE;
        let mut accepted = false;
        let mut exhausted = false;
        for _ in 0..32 {
            for j in 0..p {
                let old = buffers.previous.view().get(j)?;
                let target = buffers.proposed.view().get(j)?;
                buffers
                    .candidate
                    .coefficients
                    .view_mut()
                    .set(j, finite(old + step * (target - old))?)?;
            }
            match design.assess(response, &mut buffers.candidate, scratch, context)? {
                None => {
                    exhausted = true;
                    break;
                }
                Some(true)
                    if iteration == 0
                        || buffers.candidate.deviance
                            <= buffers.stage.deviance
                                + number::<S>(64.0)?
                                    * S::EPSILON
                                    * (S::ONE + buffers.stage.deviance) =>
                {
                    accepted = true;
                    break;
                }
                Some(_) => {}
            }
            step /= number::<S>(2.0)?;
        }
        if exhausted {
            break;
        }
        if !accepted {
            termination = if family == GlmFamily::Binomial {
                StatsTermination::Separated
            } else {
                StatsTermination::NoProgress
            };
            break;
        }
        if let Some(stop) = context.stop_reason()? {
            termination = StatsTermination::Stopped(stop);
            break;
        }
        context.commit_accepted(scioxide_core::PartialResultTrust::ResidualChecked)?;
        buffers.stage.copy_from(&buffers.candidate)?;
        if rank.is_some() {
            buffers.stage.rank = rank;
        }
        buffers.accepted_iterations += 1;
        let mut change = S::ZERO;
        let mut magnitude = S::ZERO;
        for (&old, &new) in buffers
            .previous
            .as_slice()
            .iter()
            .zip(buffers.stage.coefficients.as_slice())
        {
            change = change.max((old - new).abs());
            magnitude = magnitude.max(new.abs());
        }
        if iteration > 0
            && options.tolerance.is_satisfied(change, magnitude)?
            && options
                .score_tolerance
                .is_satisfied(buffers.stage.score_norm, S::ONE + buffers.stage.deviance)?
        {
            termination = StatsTermination::Converged;
            break;
        }
        if family == GlmFamily::Binomial
            && magnitude > number::<S>(32.0)?
            && buffers
                .stage
                .fitted
                .as_slice()
                .iter()
                .any(|&mu| mu < edge || mu > S::ONE - edge)
        {
            termination = StatsTermination::Separated;
            break;
        }
    }
    Ok(termination)
}

/// Stores a GLM fit and its diagnostics.
///
/// It holds a fit only when [`GlmReport::accepted_iterations`] is positive.
/// New storage holds zeros.
pub(super) struct GlmStorage<S: RealScalar> {
    /// Coefficients, one per design column.
    pub(super) coefficients: Vector<S>,
    /// Fitted means, including rows of zero weight.
    pub(super) fitted: Vector<S>,
    /// Response minus fitted mean.
    pub(super) residuals: Vector<S>,
    /// Linear predictors, including offset and exposure.
    pub(super) linear_predictor: Vector<S>,
    /// Weighted residual deviance, finite and nonnegative. It may underflow
    /// gradually.
    pub(super) deviance: S,
    /// Weighted Pearson chi-square statistic.
    pub(super) pearson_chi_square: S,
    /// Euclidean norm of the score.
    pub(super) score_norm: S,
    /// Pearson dispersion estimate, if residual degrees are positive.
    pub(super) dispersion: Option<S>,
    /// Residual degrees of freedom.
    pub(super) degrees_of_freedom: Option<f64>,
    /// Numerical rank of the weighted dense design, or `None` for operators.
    pub(super) rank: Option<usize>,
}

impl<S: RealScalar> GlmStorage<S> {
    /// Allocates storage for `rows` rows and `columns` columns.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub(super) fn new(rows: usize, columns: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            coefficients: Vector::zeros(columns)?,
            fitted: Vector::zeros(rows)?,
            residuals: Vector::zeros(rows)?,
            linear_predictor: Vector::zeros(rows)?,
            deviance: S::ZERO,
            pearson_chi_square: S::ZERO,
            score_norm: S::ZERO,
            dispersion: None,
            degrees_of_freedom: None,
            rank: None,
        })
    }

    pub(super) fn has_shape(&self, rows: usize, columns: usize) -> bool {
        self.coefficients.len() == columns
            && self.fitted.len() == rows
            && self.residuals.len() == rows
            && self.linear_predictor.len() == rows
    }

    pub(super) fn copy_from(&mut self, other: &Self) -> Result<(), SciOxideError> {
        if !self.has_shape(other.fitted.len(), other.coefficients.len()) {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM output dimensions",
            ));
        }
        self.coefficients
            .as_mut_slice()
            .copy_from_slice(other.coefficients.as_slice());
        self.fitted
            .as_mut_slice()
            .copy_from_slice(other.fitted.as_slice());
        self.residuals
            .as_mut_slice()
            .copy_from_slice(other.residuals.as_slice());
        self.linear_predictor
            .as_mut_slice()
            .copy_from_slice(other.linear_predictor.as_slice());
        self.deviance = other.deviance;
        self.pearson_chi_square = other.pearson_chi_square;
        self.score_norm = other.score_norm;
        self.dispersion = other.dispersion;
        self.degrees_of_freedom = other.degrees_of_freedom;
        self.rank = other.rank;
        Ok(())
    }
}

/// Returns the solve identity of a GLM fit.
///
/// # Errors
///
/// Does not fail.
pub(super) fn glm_solve_identity(
    problem: ProblemToken,
    rows: usize,
    columns: usize,
    family: GlmFamily,
) -> Result<SolveIdentity, SciOxideError> {
    let shape = ((rows as u128) << 64) | columns as u128;
    Ok(SolveIdentity::new(
        problem,
        scioxide_core::algorithm_id!("stats", "glm-fit", 1),
        shape,
        family as u128,
    ))
}

pub(super) fn publish_attempt<S: RealScalar>(
    result: Result<StatsTermination, SciOxideError>,
    stage: &GlmStorage<S>,
    output: &mut GlmStorage<S>,
    accepted_iterations: usize,
    context: &SolveContext<'_, '_>,
) -> Result<GlmReport, SciOxideError> {
    let mut termination = match result {
        Ok(termination) => termination,
        Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::ExecutionStopped => {
            match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { reason } => {
                    StatsTermination::Stopped(reason)
                }
                _ => return Err(error),
            }
        }
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::EvaluationLimit
                        | scioxide_core::ConvergenceReason::IterationLimit
                        | scioxide_core::ConvergenceReason::ResourceLimit(_),
                    ..
                }
            ) =>
        {
            StatsTermination::BudgetExhausted
        }
        Err(error) => return Err(error),
    };
    if let Some(stop) = context.stop_reason()? {
        termination = StatsTermination::Stopped(stop);
    }
    if accepted_iterations > 0 {
        output.copy_from(stage)?;
    }
    Ok(GlmReport {
        termination,
        accepted_iterations,
        counts: context.counts(),
    })
}
