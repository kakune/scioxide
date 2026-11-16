//! Dense GLM designs: IRLS by thin SVD, and covariance estimates.
use super::irls::{
    GlmRows, GlmStorage, IrlsBuffers, IrlsDesign, Proposal, WorkingProblem, assess_rows,
    glm_solve_identity, irls, publish_attempt, rows_match, working_row,
};
use super::{GlmCoefficientInference, GlmCovariance, GlmFamily, GlmOptions, GlmReport};
use crate::ConfidenceProbability;
use crate::StatsTermination;
use crate::regression::{budget_bytes, finite, invalid, number};
use crate::weights::RegressionRows;
use alloc::vec::Vec;
use scioxide_core::{
    EvaluationRole, ProblemToken, ProductSum, RealScalar, Scaled, SciOxideError, SolveContext,
    SolveIdentity,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, Matrix, MatrixView, RankPolicy, ThinSvdDecomposition,
    Vector,
};

/// Stores a copy of a dense GLM design and its row metadata.
#[derive(Clone)]
pub(super) struct DenseGlm<S: RealScalar> {
    pub(super) design: Matrix<S>,
    pub(super) family: GlmFamily,
    pub(super) rows: GlmRows<S>,
}

/// Stores the O(np) scratch of a dense IRLS fit.
pub(super) struct GlmScratch<S: RealScalar> {
    pub(super) irls: IrlsBuffers<S>,
    solve: DenseSolve<S>,
}
/// Stores the weighted design and its SVD for one IRLS solve.
pub(super) struct DenseSolve<S: RealScalar> {
    matrix: Matrix<S>,
    rhs: Vector<S>,
    roots: Vector<S>,
    factor: ThinSvdDecomposition<S>,
    svd: LinalgWorkspace<'static, S>,
}

impl<S: RealScalar> DenseGlm<S> {
    /// Copies a dense design and its row metadata.
    ///
    /// `trials` is allowed only for the binomial family and defaults to one.
    /// `exposure` is allowed only for the Poisson family, and its logarithm is
    /// added to the offset.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for an invalid shape or row metadata, or an
    /// effective sample size not above the number of columns, and
    /// `ResourceLimit` above `max_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        design: MatrixView<'_, S>,
        family: GlmFamily,
        weights: RegressionRows<'_, S>,
        trials: Option<&[u64]>,
        offsets: Option<&[S]>,
        exposure: Option<&[S]>,
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        let (n, p) = (design.rows(), design.cols());
        if n == 0 || p == 0 || n < p || !rows_match(n, family, weights, trials, offsets, exposure) {
            return Err(invalid(
                scioxide_core::argument!("design"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid GLM design or row metadata",
            ));
        }
        let elements = n
            .checked_mul(p)
            .and_then(|v| n.checked_mul(3).and_then(|rows| v.checked_add(rows)))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        budget_bytes(elements, core::mem::size_of::<S>(), max_bytes)?;
        let mut owned = Matrix::zeros(n, p)?;
        let rows = GlmRows::new(n, weights, trials, offsets, exposure)?;
        for i in 0..n {
            for j in 0..p {
                owned.view_mut().set(i, j, finite(design.get(i, j)?)?)?;
            }
        }
        if rows.units <= p as f64 || rows.units > 9_007_199_254_740_992.0 {
            return Err(invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM effective sample count must exceed columns",
            ));
        }
        Ok(Self {
            design: owned,
            family,
            rows,
        })
    }

    /// Returns the family.
    pub(super) const fn family(&self) -> GlmFamily {
        self.family
    }

    /// Returns the copied design.
    pub(super) fn design(&self) -> &Matrix<S> {
        &self.design
    }

    /// Returns the solve identity of a fit.
    ///
    /// # Errors
    ///
    /// Does not fail.
    pub(super) fn solve_identity(
        &self,
        problem: ProblemToken,
    ) -> Result<SolveIdentity, SciOxideError> {
        glm_solve_identity(problem, self.design.rows(), self.design.cols(), self.family)
    }

    /// Allocates the scratch of a fit.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` above `max_bytes`.
    pub(super) fn workspace(&self, max_bytes: usize) -> Result<GlmScratch<S>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        let elements = n
            .checked_mul(p)
            .and_then(|v| {
                n.checked_mul(16)
                    .and_then(|rows| p.checked_mul(12).and_then(|cols| rows.checked_add(cols)))
                    .and_then(|extra| v.checked_add(extra))
            })
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        budget_bytes(elements, core::mem::size_of::<S>(), max_bytes)?;
        let matrix = Matrix::zeros(n, p)?;
        let mut factor = ThinSvdDecomposition::empty();
        let mut svd = LinalgWorkspace::new();
        svd.try_reserve(
            &scioxide_linalg::ThinSvdDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::SvdWorkspaceRequest::new(
                    n,
                    p,
                    &(LinalgExecutionOptions::default()),
                ),
            )?,
        )?;
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &matrix,
            &LinalgExecutionOptions::default(),
            &mut factor,
            &mut svd,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(GlmScratch {
            irls: IrlsBuffers::new(n, p)?,
            solve: DenseSolve {
                matrix,
                rhs: Vector::zeros(n)?,
                roots: Vector::zeros(n)?,
                factor,
                svd,
            },
        })
    }

    pub(super) fn row_weight(&self, i: usize) -> Result<S, SciOxideError> {
        self.rows.weight(i)
    }

    pub(super) fn assess(
        &self,
        response: &[S],
        fit: &mut GlmStorage<S>,
    ) -> Result<bool, SciOxideError> {
        let p = self.design.cols();
        for i in 0..response.len() {
            let mut eta = ProductSum::new();
            for j in 0..p {
                eta.add_product(
                    self.design.view().get(i, j)?,
                    fit.coefficients.view().get(j)?,
                )?;
            }
            fit.linear_predictor.view_mut().set(i, eta.finish()?)?;
        }
        if !assess_rows(self.family, &self.rows, response, fit, None)? {
            return Ok(false);
        }
        let mut score_square = ProductSum::new();
        for j in 0..p {
            let mut score = ProductSum::new();
            for (i, &y) in response.iter().enumerate() {
                score.add_product(
                    self.design.view().get(i, j)?,
                    self.family
                        .score(y, fit.fitted.view().get(i)?)?
                        .mul(Scaled::from_value(self.row_weight(i)?)?)?
                        .round()?,
                )?;
            }
            let value = score.finish()?;
            score_square.add_product(value, value)?;
        }
        fit.score_norm = score_square.finish()?.sqrt();
        fit.degrees_of_freedom = Some(self.rows.units - p as f64);
        fit.dispersion = Some(finite(
            fit.pearson_chi_square / number::<S>(self.rows.units - p as f64)?,
        )?);
        Ok(true)
    }

    /// Fits one response and writes the last accepted iterate to `output`.
    ///
    /// The report describes this fit even when `output` is unchanged.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for inconsistent shapes or a response outside the
    /// family's domain, and the errors of IRLS.
    pub(super) fn fit_into(
        &self,
        response: &[S],
        options: GlmOptions<S>,
        output: &mut GlmStorage<S>,
        work: &mut GlmScratch<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<GlmReport, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        if response.len() != n
            || !output.has_shape(n, p)
            || work.solve.matrix.rows() != n
            || work.solve.matrix.cols() != p
            || work.irls.stage.coefficients.len() != p
            || response.iter().any(|&y| !self.family.valid_response(y))
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid GLM response, output, workspace or identity",
            ));
        }
        work.irls.accepted_iterations = 0;
        let result = irls(
            self,
            response,
            options,
            &mut work.irls,
            &mut work.solve,
            context,
        );
        publish_attempt(
            result,
            &work.irls.stage,
            output,
            work.irls.accepted_iterations,
            context,
        )
    }
}

impl<S: RealScalar> IrlsDesign<S> for DenseGlm<S> {
    type Scratch = DenseSolve<S>;
    fn family(&self) -> GlmFamily {
        self.family
    }
    fn rows(&self) -> &GlmRows<S> {
        &self.rows
    }
    fn propose(
        &self,
        response: &[S],
        problem: WorkingProblem<'_, S>,
        options: GlmOptions<S>,
        scratch: &mut DenseSolve<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Proposal, SciOxideError> {
        let p = self.design.cols();
        for (i, &y) in response.iter().enumerate() {
            let (root, rhs) = working_row(self.family, &self.rows, i, y, problem.eta(i)?)?;
            scratch.roots.view_mut().set(i, root)?;
            scratch.rhs.view_mut().set(i, rhs)?;
            for j in 0..p {
                scratch.matrix.view_mut().set(
                    i,
                    j,
                    finite(root * self.design.view().get(i, j)?)?,
                )?;
            }
        }
        if !context
            .record_provider(EvaluationRole::Factorization)?
            .is_granted()
        {
            return Ok(Proposal::Stop(StatsTermination::BudgetExhausted));
        }
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &scratch.matrix,
            &LinalgExecutionOptions::default().with_iteration_budget(options.factor_iterations),
            &mut scratch.factor,
            &mut scratch.svd,
            context.execution(),
        )?;
        let rank = scratch.factor.analyze_rank(options.rank_policy)?.rank();
        if rank < p {
            return Ok(Proposal::Stop(StatsTermination::RankDeficient));
        }
        if !context
            .record_provider(EvaluationRole::LinearSolve)?
            .is_granted()
        {
            return Ok(Proposal::Stop(StatsTermination::BudgetExhausted));
        }
        scratch.factor.solve_into(
            &scratch.rhs,
            options.rank_policy,
            S::ZERO,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            problem.proposed.view_mut(),
            &mut scratch.svd,
            context.execution(),
        )?;
        Ok(Proposal::Ready(Some(rank)))
    }
    fn assess(
        &self,
        response: &[S],
        candidate: &mut GlmStorage<S>,
        _: &mut DenseSolve<S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<bool>, SciOxideError> {
        DenseGlm::assess(self, response, candidate).map(Some)
    }
}

impl<S: RealScalar> DenseGlm<S> {
    /// Computes the model-based or sandwich covariance of the coefficients of a
    /// converged fit described by `report`.
    ///
    /// The sandwich estimators treat each row as one independent observation,
    /// so frequency weights other than 0 and 1 are rejected. The method
    /// recomputes the Fisher information from the fit and needs O(np + p²)
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the fit did not converge, does not match the
    /// response, or is rank deficient, for a sandwich estimator with frequency
    /// weights, and for HC3 at leverage 1.
    pub(super) fn covariance(
        &self,
        response: &[S],
        fit: &GlmStorage<S>,
        report: &GlmReport,
        convention: GlmCovariance,
    ) -> Result<Matrix<S>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        if response.len() != n
            || fit.coefficients.len() != p
            || fit.fitted.len() != n
            || report.termination != StatsTermination::Converged
            || report.accepted_iterations == 0
            || fit.rank != Some(p)
            || (convention != GlmCovariance::Model
                && self.rows.frequency
                && self
                    .rows
                    .weights
                    .as_slice()
                    .iter()
                    .any(|&v| v != S::ZERO && v != S::ONE))
        {
            return Err(invalid(
                scioxide_core::argument!("fit"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM covariance requires a converged full-rank matching fit",
            ));
        }
        let mut verified = GlmStorage::new(n, p)?;
        scioxide_linalg::VectorView::from(&fit.coefficients).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (verified.coefficients.view_mut()).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        if !self.assess(response, &mut verified)? {
            return Err(invalid(
                scioxide_core::argument!("fit"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "GLM inference mean outside family domain",
            ));
        }
        let tolerance = number::<S>(128.0)? * S::EPSILON * (S::ONE + fit.deviance.abs());
        if (verified.deviance - fit.deviance).abs() > tolerance
            || verified
                .fitted
                .as_slice()
                .iter()
                .zip(fit.fitted.as_slice())
                .any(|(&current, &reported)| {
                    (current - reported).abs()
                        > tolerance * (S::ONE + current.abs() + reported.abs())
                })
        {
            return Err(invalid(
                scioxide_core::argument!("fit"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM fit and response do not match",
            ));
        }
        let mut matrix = Matrix::zeros(n, p)?;
        let mut roots = Vector::zeros(n)?;
        for i in 0..n {
            let (_, derivative, variance) = self
                .family
                .mean(verified.linear_predictor.view().get(i)?)
                .ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("fit"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid GLM inference mean",
                    )
                })?;
            let root = finite((self.row_weight(i)? * derivative * derivative / variance).sqrt())?;
            roots.view_mut().set(i, root)?;
            for j in 0..p {
                matrix
                    .view_mut()
                    .set(i, j, finite(root * self.design.view().get(i, j)?)?)?;
            }
        }
        let factor = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)?;
        if factor
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)?
            .rank()
            != p
        {
            return Err(invalid(
                scioxide_core::argument!("fit"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM inference Fisher information is rank deficient",
            ));
        }
        let mut inverse = Matrix::zeros(p, n)?;
        let mut rhs = Vector::zeros(n)?;
        let mut column = Vector::zeros(p)?;
        let mut scratch = ThinSvdDecomposition::<S>::workspace(
            &scioxide_linalg::SvdWorkspaceRequest::solve(n, p, &LinalgExecutionOptions::new()),
        )?;
        for i in 0..n {
            rhs.as_mut_slice().fill(S::ZERO);
            rhs.view_mut().set(i, S::ONE)?;
            factor.solve_into(
                &rhs,
                RankPolicy::DimensionScaledEpsilon,
                S::ZERO,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                column.view_mut(),
                &mut scratch,
                scioxide_core::ExecutionContext::sequential(),
            )?;
            for j in 0..p {
                inverse.view_mut().set(j, i, column.view().get(j)?)?;
            }
        }
        let dispersion = if matches!(self.family, GlmFamily::Binomial | GlmFamily::Poisson) {
            S::ONE
        } else {
            verified.dispersion.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("fit"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "GLM dispersion unavailable",
                )
            })?
        };
        let mut multipliers = Vector::zeros(n)?;
        for i in 0..n {
            let value = if convention == GlmCovariance::Model {
                dispersion
            } else {
                let (_, derivative, _) = self
                    .family
                    .mean(verified.linear_predictor.view().get(i)?)
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("fit"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "invalid GLM robust mean",
                        )
                    })?;
                let q = if roots.view().get(i)? == S::ZERO {
                    S::ZERO
                } else {
                    finite(roots.view().get(i)? * verified.residuals.view().get(i)? / derivative)?
                };
                let adjustment = if convention == GlmCovariance::Hc3 {
                    let mut leverage = ProductSum::new();
                    for j in 0..p {
                        leverage
                            .add_product(matrix.view().get(i, j)?, inverse.view().get(j, i)?)?;
                    }
                    let rest = finite(S::ONE - leverage.finish()?)?;
                    if rest <= number::<S>(32.0)? * S::EPSILON {
                        return Err(invalid(
                            scioxide_core::argument!("convention"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "GLM HC3 undefined at unit leverage",
                        ));
                    }
                    rest
                } else {
                    S::ONE
                };
                let scaled = finite(q / adjustment)?;
                finite(scaled * scaled)?
            };
            multipliers.view_mut().set(i, value)?;
        }
        let mut covariance = Matrix::zeros(p, p)?;
        for j in 0..p {
            for k in j..p {
                let mut sum = ProductSum::new();
                for i in 0..n {
                    sum.add_product(
                        inverse.view().get(j, i)?,
                        finite(inverse.view().get(k, i)? * multipliers.view().get(i)?)?,
                    )?;
                }
                let value = sum.finish()?;
                covariance.view_mut().set(j, k, value)?;
                covariance.view_mut().set(k, j, value)?;
            }
        }
        Ok(covariance)
    }

    /// Returns Wald tests and confidence intervals for each coefficient.
    ///
    /// They use the normal distribution and are asymptotic.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::covariance`], and `NumericalInstability`
    /// if a standard error or endpoint overflows.
    pub(super) fn coefficient_inference(
        &self,
        response: &[S],
        fit: &GlmStorage<S>,
        report: &GlmReport,
        covariance: GlmCovariance,
        confidence: ConfidenceProbability,
    ) -> Result<Vec<GlmCoefficientInference<S>>, SciOxideError> {
        let matrix = self.covariance(response, fit, report, covariance)?;
        let critical = number::<S>(confidence.normal_critical_value::<f64>()?)?;
        let mut output = scioxide_core::backend::try_with_capacity(
            matrix.rows(),
            scioxide_core::quantity!("glm_inference"),
        )?;
        for j in 0..matrix.rows() {
            let variance = matrix.view().get(j, j)?;
            if variance < S::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("covariance"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "negative GLM coefficient variance",
                ));
            }
            let se = finite(variance.sqrt())?;
            let estimate = fit.coefficients.view().get(j)?;
            let p_value = if se == S::ZERO {
                if estimate == S::ZERO { 1.0 } else { 0.0 }
            } else {
                let z = finite(estimate / se)?;
                scioxide_special::erfc(z.abs().try_to_f64_exact()? / core::f64::consts::SQRT_2)?
            };
            let half = finite(critical * se)?;
            output.push(GlmCoefficientInference {
                estimate,
                standard_error: se,
                p_value: number(p_value)?,
                interval: scioxide_core::Interval::new(
                    finite(estimate - half)?,
                    finite(estimate + half)?,
                )?,
                covariance,
            });
        }
        Ok(output)
    }
}
