//! Matrix-free GLM designs fitted by IRLS with LSMR solves.
use super::irls::{
    GlmRows, GlmStorage, IrlsBuffers, IrlsDesign, Proposal, WorkingProblem, assess_rows,
    glm_solve_identity, irls, publish_attempt, rows_match, working_row,
};
use super::{GlmFamily, GlmOptions, GlmReport};
use crate::StatsTermination;
use crate::regression::{budget_bytes, finite, invalid};
use crate::weights::RegressionRows;
use scioxide_core::{
    AdjointOperator, Budget, EvaluationRole, GradientTolerance, LinearOperator, ProblemToken,
    ProductSum, RealScalar, ResidualTolerance, SciOxideError, SolveContext, SolveIdentity,
};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, LsmrSolver, Vector,
};

/// Borrows a sparse, dense or matrix-free GLM design.
///
/// A fit reports coefficients and residual diagnostics, but no rank and no
/// covariance, because LSMR does not determine the rank of the design.
pub(super) struct OperatorGlm<'a, S: RealScalar, O: ?Sized> {
    pub(super) operator: &'a O,
    pub(super) family: GlmFamily,
    rows: GlmRows<S>,
}
impl<S: RealScalar, O: ?Sized> Clone for OperatorGlm<'_, S, O> {
    fn clone(&self) -> Self {
        Self {
            operator: self.operator,
            family: self.family,
            rows: self.rows.clone(),
        }
    }
}

/// Stores the IRLS iterates and the LSMR scratch of an operator fit.
pub(super) struct GlmOperatorScratch<S: RealScalar> {
    pub(super) irls: IrlsBuffers<S>,
    solve: OperatorSolve<S>,
}
/// Stores the weighted working problem and the LSMR and operator scratch of
/// one IRLS solve.
pub(super) struct OperatorSolve<S: RealScalar> {
    roots: Vector<S>,
    rhs: Vector<S>,
    score_rhs: Vector<S>,
    score: Vector<S>,
    provider: Vector<S>,
    krylov: IterativeWorkspace<S>,
}

// Reports errors raised through the operator traits, since trait methods cannot take an
// operation parameter.
struct WeightedGlmOperator<'a, S: RealScalar, O: ?Sized> {
    inner: &'a O,
    roots: &'a [S],
}

impl<S: RealScalar, O: AdjointOperator<S> + ?Sized> WeightedGlmOperator<'_, S, O> {
    fn apply_forward(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        let outcome = match context {
            Some(context) => self.inner.apply_into_scoped(
                input,
                output,
                workspace,
                &mut scioxide_core::OperatorContext::new(context),
            ),
            None => self
                .inner
                .apply_into(input, output, workspace)
                .map(scioxide_core::Invocation::Evaluated),
        }?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
        for (value, &root) in output.iter_mut().zip(self.roots) {
            *value = finite(*value * root)?;
        }
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
    fn apply_adjoint(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        let (scaled, inner) = workspace.split_at_mut(self.rows());
        for ((value, &input), &root) in scaled.iter_mut().zip(input).zip(self.roots) {
            *value = finite(input * root)?;
        }
        match context {
            Some(context) => self.inner.apply_adjoint_into_scoped(
                scaled,
                output,
                inner,
                &mut scioxide_core::OperatorContext::new(context),
            ),
            None => self
                .inner
                .apply_adjoint_into(scaled, output, inner)
                .map(scioxide_core::Invocation::Evaluated),
        }
    }
}
impl<S: RealScalar, O: AdjointOperator<S> + ?Sized> LinearOperator<S>
    for WeightedGlmOperator<'_, S, O>
{
    fn rows(&self) -> usize {
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.inner.apply_workspace_len()
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_forward(input, output, workspace, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_forward(input, output, workspace, Some(context))
    }
}

impl<S: RealScalar, O: AdjointOperator<S> + ?Sized> AdjointOperator<S>
    for WeightedGlmOperator<'_, S, O>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.rows()
            .checked_add(self.inner.apply_adjoint_workspace_len()?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_adjoint(input, output, workspace, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_adjoint(input, output, workspace, Some(context))
    }
}

impl<'a, S: RealScalar, O: AdjointOperator<S> + ?Sized> OperatorGlm<'a, S, O> {
    /// Borrows an operator and copies its row metadata. `trials` and
    /// `exposure` have the same meaning as for a dense design.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for an empty shape, invalid row metadata or no
    /// positive weight, and `ResourceLimit` above `max_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        operator: &'a O,
        family: GlmFamily,
        weights: RegressionRows<'_, S>,
        trials: Option<&[u64]>,
        offsets: Option<&[S]>,
        exposure: Option<&[S]>,
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        let n = operator.rows();
        if n == 0
            || operator.cols() == 0
            || !rows_match(n, family, weights, trials, offsets, exposure)
        {
            return Err(invalid(
                scioxide_core::argument!("operator"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid operator GLM shape or row metadata",
            ));
        }
        budget_bytes(
            n.checked_mul(3).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?,
            core::mem::size_of::<S>(),
            max_bytes,
        )?;
        let rows = GlmRows::new(n, weights, trials, offsets, exposure)?;
        if rows.units == 0.0 {
            return Err(invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "operator GLM needs positive row weight",
            ));
        }
        Ok(Self {
            operator,
            family,
            rows,
        })
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
        glm_solve_identity(
            problem,
            self.operator.rows(),
            self.operator.cols(),
            self.family,
        )
    }

    /// Allocates O(n + p + s) scratch, where `s` is the operator's scratch
    /// length, and an LSMR workspace.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` above `max_bytes`, and the error of the
    /// operator's scratch-length query.
    pub(super) fn workspace(
        &self,
        max_bytes: usize,
    ) -> Result<GlmOperatorScratch<S>, SciOxideError> {
        let (n, p) = (self.operator.rows(), self.operator.cols());
        let provider = self
            .operator
            .apply_workspace_len()?
            .max(self.operator.apply_adjoint_workspace_len()?);
        let elements = n
            .checked_mul(10)
            .and_then(|v| p.checked_mul(8).and_then(|cols| v.checked_add(cols)))
            .and_then(|v| v.checked_add(provider))
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        budget_bytes(elements, core::mem::size_of::<S>(), max_bytes)?;
        let roots = Vector::zeros(n)?;
        let weighted = WeightedGlmOperator {
            inner: self.operator,
            roots: roots.as_slice(),
        };
        let solver = LsmrSolver::new(ResidualTolerance::default(), Budget::try_from(4096)?)
            .with_normal_tolerance(GradientTolerance::default());
        let krylov = solver.workspace(&IterativeWorkspaceRequest::adjoint(&weighted)?)?;
        Ok(GlmOperatorScratch {
            irls: IrlsBuffers::new(n, p)?,
            solve: OperatorSolve {
                roots,
                rhs: Vector::zeros(n)?,
                score_rhs: Vector::zeros(n)?,
                score: Vector::zeros(p)?,
                provider: Vector::zeros(provider)?,
                krylov,
            },
        })
    }

    /// Fits one response and writes the last accepted iterate to `output`.
    ///
    /// Each LSMR solve runs in a child scope of `context`. The fit has no rank
    /// and no covariance.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for inconsistent shapes or a response outside the
    /// family's domain, and the errors of the operator and of IRLS.
    pub(super) fn fit_into(
        &self,
        response: &[S],
        options: GlmOptions<S>,
        output: &mut GlmStorage<S>,
        work: &mut GlmOperatorScratch<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<GlmReport, SciOxideError> {
        let (n, p) = (self.operator.rows(), self.operator.cols());
        if response.len() != n
            || !output.has_shape(n, p)
            || work.solve.roots.len() != n
            || !work.irls.stage.has_shape(n, p)
            || response.iter().any(|&y| !self.family.valid_response(y))
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid operator GLM response, output or identity",
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

impl<S: RealScalar, O: AdjointOperator<S> + ?Sized> IrlsDesign<S> for OperatorGlm<'_, S, O> {
    type Scratch = OperatorSolve<S>;
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
        scratch: &mut OperatorSolve<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Proposal, SciOxideError> {
        for (i, &y) in response.iter().enumerate() {
            let (root, rhs) = working_row(self.family, &self.rows, i, y, problem.eta(i)?)?;
            scratch.roots.view_mut().set(i, root)?;
            scratch.rhs.view_mut().set(i, rhs)?;
        }
        let weighted = WeightedGlmOperator {
            inner: self.operator,
            roots: scratch.roots.as_slice(),
        };
        let solver = LsmrSolver::new(ResidualTolerance::default(), options.factor_iterations)
            .with_normal_tolerance(GradientTolerance::default());
        problem
            .proposed
            .as_mut_slice()
            .copy_from_slice(problem.previous.as_slice());
        // The LSMR correction runs in a child epoch of this fit.
        let report = solver
            .solve_into(
                &weighted,
                &scratch.rhs,
                problem.proposed.view_mut(),
                &mut scratch.krylov,
                &mut *context,
            )?
            .termination();
        Ok(match report {
            LinearSolveTermination::Converged | LinearSolveTermination::LeastSquaresConverged => {
                Proposal::Ready(None)
            }
            LinearSolveTermination::ExecutionStopped(stop) => {
                Proposal::Stop(StatsTermination::Stopped(stop))
            }
            LinearSolveTermination::IterationLimit | LinearSolveTermination::EvaluationLimit => {
                Proposal::Stop(StatsTermination::BudgetExhausted)
            }
            _ => Proposal::Stop(StatsTermination::NoProgress),
        })
    }
    fn assess(
        &self,
        response: &[S],
        fit: &mut GlmStorage<S>,
        scratch: &mut OperatorSolve<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<bool>, SciOxideError> {
        if !context
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            return Ok(None);
        }
        let forward = self.operator.apply_workspace_len()?;
        self.operator
            .apply_into_scoped(
                fit.coefficients.as_slice(),
                fit.linear_predictor.as_mut_slice(),
                &mut scratch.provider.as_mut_slice()[..forward],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        if !assess_rows(
            self.family,
            &self.rows,
            response,
            fit,
            Some(scratch.score_rhs.as_mut_slice()),
        )? {
            return Ok(Some(false));
        }
        if !context
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            return Ok(None);
        }
        let adjoint = self.operator.apply_adjoint_workspace_len()?;
        self.operator
            .apply_adjoint_into_scoped(
                scratch.score_rhs.as_slice(),
                scratch.score.as_mut_slice(),
                &mut scratch.provider.as_mut_slice()[..adjoint],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        fit.score_norm = scratch
            .score
            .as_slice()
            .iter()
            .try_fold(ProductSum::new(), |mut sum, &v| {
                sum.add_product(v, v)?;
                Ok::<_, SciOxideError>(sum)
            })?
            .finish()?
            .sqrt();
        fit.degrees_of_freedom = None;
        fit.dispersion = None;
        fit.rank = None;
        Ok(Some(true))
    }
}
