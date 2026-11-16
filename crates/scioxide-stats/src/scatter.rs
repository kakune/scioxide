//! Huber M-estimator of multivariate location and scatter.
use crate::StatsTermination;
use crate::error_helpers::invalid;
use crate::sample_covariance_with_options;
use alloc::vec::Vec;
use scioxide_core::NumericScalar;

use scioxide_core::{
    Budget, EvaluationRole, ProblemToken, RealScalar, SciOxideError, SolveContext, SolveIdentity,
};
use scioxide_linalg::{
    HermitianEigenDecomposition, HermitianEigenOptions, LinalgWorkspace, Matrix, MatrixView,
};

use crate::error_helpers::count as scalar;

/// Options for [`PreparedHuberScatter`]: the Huber cutoff, the convergence
/// tolerance, the iteration limit and a byte limit.
///
/// The scatter is normalized by the total weight and has no consistency
/// correction for the normal distribution. The estimator does not have a high
/// breakdown point, and the data must have full column rank.
#[derive(Clone, Copy, Debug)]
pub struct HuberScatterOptions<S: RealScalar> {
    /// Mahalanobis distance above which observations are downweighted; positive.
    tuning: S,
    /// Tolerance on the relative change of the estimate; positive.
    tolerance: S,
    /// Maximum number of reweighting iterations.
    iterations: Budget,
    /// Limit on the memory used by the plan, in bytes.
    max_bytes: usize,
}
impl<S: RealScalar> Default for HuberScatterOptions<S> {
    fn default() -> Self {
        Self {
            tuning: S::ONE + S::ONE,
            tolerance: S::SQRT_EPSILON,
            iterations: Budget::ONE.saturating_add(99),
            max_bytes: usize::MAX,
        }
    }
}
impl<S: RealScalar> HuberScatterOptions<S> {
    /// Sets the cutoff `c`: observations with Mahalanobis distance above `c` get
    /// weight `c/d`. It must be finite and positive.
    pub fn with_tuning(mut self, value: S) -> Self {
        self.tuning = value;
        self
    }
    /// Returns the cutoff `c`.
    pub fn tuning(&self) -> S {
        self.tuning
    }
    /// Sets the convergence tolerance on the relative change of the estimate. It
    /// must be finite and positive.
    pub fn with_tolerance(mut self, value: S) -> Self {
        self.tolerance = value;
        self
    }
    /// Returns the convergence tolerance.
    pub fn tolerance(&self) -> S {
        self.tolerance
    }
    /// Sets the maximum number of reweighting iterations.
    pub fn with_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Returns the maximum number of reweighting iterations.
    pub fn iterations(&self) -> Budget {
        self.iterations
    }
    /// Sets the limit, in bytes, on the memory used by the plan.
    pub fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the byte limit.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}
/// Robust location and scatter estimate, with its iteration diagnostics.
pub struct HuberScatterFit<S: RealScalar> {
    /// Weighted location.
    location: Vec<S>,
    /// Symmetric positive definite scatter matrix.
    scatter: Matrix<S>,
    /// Number of completed reweighting iterations.
    iterations: usize,
    /// Relative change of the last iteration, `None` before the first.
    relative_change: Option<S>,
    /// Termination status.
    termination: StatsTermination,
}
/// Returns the solve identity of a Huber scatter fit of an `n × p` matrix.
fn identity(problem: ProblemToken, n: usize, p: usize) -> Result<SolveIdentity, SciOxideError> {
    Ok(SolveIdentity::new(
        problem,
        scioxide_core::algorithm_id!("stats", "huber-scatter", 1),
        ((n as u128) << 64) | (p as u128),
        0,
    ))
}

/// Huber M-estimator of the location and scatter of real multivariate data.
///
/// Rows of the `n × p` matrix `X` are observations `xᵢ`. Starting from the
/// sample mean and covariance, each iteration computes the Mahalanobis distances
/// `dᵢ = √((xᵢ − μ)ᵀ Σ⁻¹ (xᵢ − μ))`, the Huber weights `wᵢ = min(1, c/dᵢ)` with
/// cutoff `c`, and the new estimate
///
/// ```text
/// μ = Σᵢ wᵢ xᵢ / Σᵢ wᵢ,    Σ = Σᵢ wᵢ (xᵢ − μ)(xᵢ − μ)ᵀ / Σᵢ wᵢ.
/// ```
///
/// The iteration converges when the largest change of an entry of `μ` or `Σ`,
/// divided by the largest absolute entry (at least 1), is at most the
/// tolerance. Each iteration costs one symmetric eigendecomposition, O(p³),
/// and O(np²) for the distances and the new scatter.
///
/// The fit reports its status as a [`StatsTermination`](crate::StatsTermination):
/// `Converged`, `BudgetExhausted` when the iteration limit is reached, or
/// `Stopped`. See [Results and termination](crate#results-and-termination).
///
/// # References
///
/// R. A. Maronna, "Robust M-estimators of multivariate location and scatter",
/// *Annals of Statistics* 4 (1976), 51–67.
#[derive(Clone)]
pub struct PreparedHuberScatter<'a, S: RealScalar> {
    data: MatrixView<'a, S>,
    options: HuberScatterOptions<S>,
    initial_location: Vec<S>,
    initial_scatter: Matrix<S>,
}
/// Scratch storage for [`PreparedHuberScatter::fit_into`]: weights, the
/// eigendecomposition and the candidate estimate.
///
/// It also records the most recent call. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct HuberScatterWorkspace<S: RealScalar> {
    stage: HuberScatterFit<S>,
    weights: Vec<S>,
    candidate_location: Vec<S>,
    candidate_scatter: Matrix<S>,
    inverse: Matrix<S>,
    eig: HermitianEigenDecomposition<S>,
    eig_work: LinalgWorkspace<'static, S>,
    slot: crate::execution::SolveSlot<usize>,
}
/// Output storage for [`PreparedHuberScatter::fit_into`].
///
/// It holds a fit only after a call returns `Ok`.
pub struct HuberScatterOutput<S: RealScalar> {
    fit: HuberScatterFit<S>,
    published: bool,
}
impl<S: RealScalar> HuberScatterOutput<S> {
    /// Returns the fit written by the last successful call, or `None` before the first.
    pub fn fit(&self) -> Option<&HuberScatterFit<S>> {
        self.published.then_some(&self.fit)
    }
}
/// Report of one call to [`PreparedHuberScatter::fit_into`].
#[derive(Clone, Copy, Debug)]
pub struct HuberScatterReport<S: RealScalar> {
    termination: StatsTermination,
    iterations: usize,
    relative_change: Option<S>,
    counts: scioxide_core::SolveCounts,
}
impl<S: RealScalar> HuberScatterReport<S> {
    /// Returns the termination status.
    pub fn termination(self) -> StatsTermination {
        self.termination
    }
    /// Returns the number of completed reweighting iterations.
    pub fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the relative change of the last iteration, or `None` if none completed.
    pub fn relative_change(self) -> Option<S> {
        self.relative_change
    }
    /// Returns the iteration and evaluation counts of this call.
    pub fn counts(self) -> scioxide_core::SolveCounts {
        self.counts
    }
}
impl<S: RealScalar> HuberScatterFit<S> {
    /// Returns the robust location `μ`.
    pub fn location(&self) -> &[S] {
        &self.location
    }
    /// Returns the robust scatter matrix `Σ`, symmetric positive definite.
    pub fn scatter(&self) -> &Matrix<S> {
        &self.scatter
    }
    /// Returns the number of completed reweighting iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the relative change of the last iteration, or `None` if the result
    /// is the sample mean and covariance used as the starting point.
    pub fn relative_change(&self) -> Option<S> {
        self.relative_change
    }
    /// Returns the termination status.
    pub fn termination(&self) -> StatsTermination {
        self.termination
    }
}
impl<S: RealScalar> HuberScatterWorkspace<S> {
    /// Returns the record of the most recent call, including one that failed
    /// before its first iteration, or `None` before the first call.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<usize>> {
        self.slot.last_attempt()
    }
    /// Returns the last accepted estimate of the most recent call, or `None` if
    /// that call accepted none.
    pub fn last_accepted(&self) -> Option<HuberScatterAccepted<'_, S>> {
        self.slot
            .last_attempt()
            .and_then(scioxide_core::AttemptRecord::progress)
            .map(|_| HuberScatterAccepted { raw: &self.stage })
    }
}
/// Last accepted estimate of a Huber scatter fit, which may be the starting point.
pub struct HuberScatterAccepted<'a, S: RealScalar> {
    raw: &'a HuberScatterFit<S>,
}
impl<S: RealScalar> HuberScatterAccepted<'_, S> {
    /// Returns the location.
    pub fn location(&self) -> &[S] {
        &self.raw.location
    }
    /// Returns the scatter matrix.
    pub fn scatter(&self) -> &Matrix<S> {
        &self.raw.scatter
    }
    /// Returns the number of completed reweighting iterations.
    pub fn iterations(&self) -> usize {
        self.raw.iterations
    }
    /// Returns the relative change of the last iteration, or `None` if none completed.
    pub fn relative_change(&self) -> Option<S> {
        self.raw.relative_change
    }
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> PreparedHuberScatter<'a, S> {
    /// Validates `data` and `options` and computes the starting sample mean and
    /// covariance.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `data` has no columns or no more rows than columns.
    /// - `InvalidInput` if the cutoff or the tolerance is not finite and positive.
    /// - `NonFinite` if `data` contains NaN or infinity.
    /// - `ResourceLimit` if the memory needed exceeds `options.max_bytes()`.
    pub fn new(
        data: MatrixView<'a, S>,
        options: HuberScatterOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let (n, p) = (data.rows(), data.cols());
        if n <= p
            || p == 0
            || !options.tuning.is_finite()
            || options.tuning <= S::ZERO
            || !options.tolerance.is_finite()
            || options.tolerance <= S::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "scatter shape or controls",
            ));
        }
        let elements = n
            .checked_mul(p)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| {
                p.checked_mul(p)
                    .and_then(|q| q.checked_mul(16))
                    .and_then(|q| v.checked_add(q))
            })
            .and_then(|v| v.checked_add(n))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("scatter_elements"))
            })?;
        crate::error_helpers::limit_bytes(
            elements
                .checked_mul(core::mem::size_of::<S>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("scatter_bytes"))
                })?,
            options.max_bytes,
        )?;
        let (initial_location, initial_scatter) = sample_covariance_with_options(
            data,
            crate::MultivariateOptions::new().with_max_bytes(options.max_bytes),
        )?;
        Ok(Self {
            data,
            options,
            initial_location,
            initial_scatter,
        })
    }
    fn storage(&self) -> Result<HuberScatterFit<S>, SciOxideError> {
        let p = self.data.cols();
        Ok(HuberScatterFit {
            location: scioxide_core::backend::try_filled(
                p,
                S::ZERO,
                scioxide_core::quantity!("scatter_location"),
            )?,
            scatter: Matrix::zeros(p, p)?,
            iterations: 0,
            relative_change: None,
            termination: StatsTermination::BudgetExhausted,
        })
    }
    /// Creates output storage for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if memory cannot be reserved.
    pub fn output(&self) -> Result<HuberScatterOutput<S>, SciOxideError> {
        Ok(HuberScatterOutput {
            fit: self.storage()?,
            published: false,
        })
    }
    /// Creates a workspace for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the sample covariance is numerically singular: its
    ///   smallest eigenvalue is at most `p ε λ_max`, where `λ_max` is the largest.
    /// - The errors of the symmetric eigendecomposition of the sample covariance.
    pub fn workspace(&self) -> Result<HuberScatterWorkspace<S>, SciOxideError> {
        let p = self.data.cols();
        let mut eig = HermitianEigenDecomposition::empty();
        let mut eig_work = LinalgWorkspace::new();
        HermitianEigenDecomposition::decompose_into(
            &self.initial_scatter,
            scioxide_linalg::HermitianInput::Exact,
            &HermitianEigenOptions::default(),
            &mut eig,
            &mut eig_work,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        let high = eig.eigenvalues().iter().copied().fold(S::ZERO, S::max);
        let threshold = high * S::EPSILON * scalar::<S>(p)?;
        if eig.eigenvalues().iter().any(|&value| value <= threshold) {
            return Err(invalid(
                scioxide_core::argument!("data"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "singular robust scatter initialization",
            ));
        }
        Ok(HuberScatterWorkspace {
            stage: self.storage()?,
            weights: scioxide_core::backend::try_filled(
                self.data.rows(),
                S::ONE,
                scioxide_core::quantity!("scatter_weights"),
            )?,
            candidate_location: scioxide_core::backend::try_filled(
                p,
                S::ZERO,
                scioxide_core::quantity!("scatter_candidate"),
            )?,
            candidate_scatter: Matrix::zeros(p, p)?,
            inverse: Matrix::zeros(p, p)?,
            eig,
            eig_work,
            slot: Default::default(),
        })
    }
    /// Runs the reweighting iteration into `output` as a root or nested solve.
    ///
    /// The iteration limit comes from the options. When a limit, cancellation or
    /// deadline ends the iteration, the call returns `Ok` with the last accepted
    /// estimate, which may be the sample mean and covariance, and an unsuccessful
    /// status. See [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output` or `work` was created for data of another shape.
    /// - `InvalidInput` if an iterate becomes numerically singular or yields an
    ///   invalid Mahalanobis distance or total weight.
    /// - `NumericalInstability` (`Breakdown`) if the scatter or the relative change
    ///   overflows.
    /// - The errors of the symmetric eigendecomposition.
    pub fn fit_into<'scope, 'capability>(
        &self,
        output: &mut HuberScatterOutput<S>,
        work: &mut HuberScatterWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<HuberScatterReport<S>, SciOxideError> {
        work.slot.reject();
        let p = self.data.cols();
        if output.fit.location.len() != p
            || output.fit.scatter.rows() != p
            || work.stage.location.len() != p
            || work.weights.len() != self.data.rows()
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "scatter output or workspace shape",
            ));
        }
        work.stage.location.copy_from_slice(&self.initial_location);
        work.stage
            .scatter
            .as_mut_slice()
            .copy_from_slice(self.initial_scatter.as_slice());
        work.stage.iterations = 0;
        work.stage.relative_change = None;
        let accepted = core::cell::Cell::new(0usize);
        let mut slot = core::mem::take(&mut work.slot);
        let result = slot.run(
            scope.into(),
            identity(ProblemToken::new(0), self.data.rows(), p)?,
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Iteration,
                self.options.iterations,
            ),
            |context| {
                let result = scatter_attempt(self.data, self.options, work, context);
                accepted.set(work.stage.iterations);
                result
            },
            |result| {
                (
                    match result {
                        Ok(report) if report.termination == StatsTermination::Converged => {
                            scioxide_core::AttemptOutcome::Succeeded
                        }
                        Ok(_) => scioxide_core::AttemptOutcome::Stopped,
                        Err(_) => scioxide_core::AttemptOutcome::Failed,
                    },
                    Some(accepted.get()),
                )
            },
        );
        work.slot = slot;
        if let Ok(report) = &result {
            output.fit.location.copy_from_slice(&work.stage.location);
            output
                .fit
                .scatter
                .as_mut_slice()
                .copy_from_slice(work.stage.scatter.as_slice());
            output.fit.iterations = report.iterations;
            output.fit.relative_change = report.relative_change;
            output.fit.termination = report.termination;
            output.published = true;
        }
        result
    }
    /// Runs the reweighting iteration and returns the estimate.
    ///
    /// Allocates the output and workspace for one call. Use [`Self::fit_into`] to
    /// reuse them.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::fit_into`].
    pub fn fit_scoped<'scope, 'capability>(
        &self,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<HuberScatterFit<S>, SciOxideError> {
        let mut output = self.output()?;
        let mut ws = self.workspace()?;
        self.fit_into(&mut output, &mut ws, scope)?;
        Ok(output.fit)
    }
    /// Executes [`Self::fit_scoped`] with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the scoped form's input, numerical and allocation errors.
    pub fn fit(&self) -> Result<HuberScatterFit<S>, SciOxideError> {
        self.fit_scoped(scioxide_core::ExecutionContext::sequential_native())
    }
}
fn scatter_attempt<S: RealScalar + NumericScalar<Real = S>>(
    data: MatrixView<'_, S>,
    options: HuberScatterOptions<S>,
    work: &mut HuberScatterWorkspace<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<HuberScatterReport<S>, SciOxideError> {
    let p = data.cols();
    let HuberScatterWorkspace {
        stage,
        weights,
        candidate_location,
        candidate_scatter,
        inverse,
        eig,
        eig_work,
        ..
    } = work;
    let location = &mut stage.location;
    let scatter = &mut stage.scatter;
    let mut iterations = 0;
    let mut relative_change = None;
    let mut termination = StatsTermination::BudgetExhausted;
    for _ in 0..options.iterations.get() {
        if let Some(stop) = context.stop_reason()? {
            termination = StatsTermination::Stopped(stop);
            break;
        }
        if !context.record_iteration()?.is_granted() {
            break;
        }
        if !context
            .record_provider(EvaluationRole::Factorization)?
            .is_granted()
        {
            break;
        }
        HermitianEigenDecomposition::decompose_into(
            &*scatter,
            scioxide_linalg::HermitianInput::Exact,
            &HermitianEigenOptions::default(),
            eig,
            eig_work,
            context.execution(),
        )?;
        let high = eig.eigenvalues().iter().copied().fold(S::ZERO, S::max);
        let low_bound = high * S::EPSILON * scalar::<S>(p)?;
        if eig.eigenvalues().iter().any(|&v| v <= low_bound) {
            return Err(invalid(
                scioxide_core::argument!("data"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "singular robust scatter",
            ));
        }
        for i in 0..p {
            for j in 0..p {
                let mut value = S::ZERO;
                for k in 0..p {
                    value += eig.eigenvectors().view().get(i, k)?
                        * eig.eigenvectors().view().get(j, k)?
                        / eig.eigenvalues()[k];
                }
                inverse.view_mut().set(i, j, value)?;
            }
        }
        let mut total = S::ZERO;
        candidate_location.fill(S::ZERO);
        for (i, w) in weights.iter_mut().enumerate() {
            let mut d2 = S::ZERO;
            for j in 0..p {
                for k in 0..p {
                    d2 += (data.get(i, j)? - location[j])
                        * inverse.view().get(j, k)?
                        * (data.get(i, k)? - location[k]);
                }
            }
            if !d2.is_finite() || d2 < -low_bound {
                return Err(invalid(
                    scioxide_core::argument!("data"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "invalid Mahalanobis distance",
                ));
            }
            let d = d2.max(S::ZERO).sqrt();
            *w = if d <= options.tuning {
                S::ONE
            } else {
                options.tuning / d
            };
            total += *w;
            for (j, m) in candidate_location.iter_mut().enumerate() {
                *m += *w * data.get(i, j)?;
            }
        }
        if !total.is_finite() || total <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("data"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid robust total weight",
            ));
        }
        for m in candidate_location.iter_mut() {
            *m /= total;
        }
        for j in 0..p {
            for k in j..p {
                let mut value = S::ZERO;
                for (i, &w) in weights.iter().enumerate() {
                    value += w
                        * (data.get(i, j)? - candidate_location[j])
                        * (data.get(i, k)? - candidate_location[k]);
                }
                value /= total;
                if !value.is_finite() {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "non-finite robust scatter",
                    ));
                }
                candidate_scatter.view_mut().set(j, k, value)?;
                candidate_scatter.view_mut().set(k, j, value)?;
            }
        }
        let mut change = S::ZERO;
        let mut scale = S::ONE;
        for (&a, &b) in candidate_location.iter().zip(location.iter()) {
            change = change.max((a - b).abs());
            scale = scale.max(a.abs()).max(b.abs());
        }
        for (&a, &b) in candidate_scatter.as_slice().iter().zip(scatter.as_slice()) {
            change = change.max((a - b).abs());
            scale = scale.max(a.abs()).max(b.abs());
        }
        let step_change = change / scale;
        if !step_change.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "non-finite robust convergence metric",
            ));
        }
        context.commit_accepted(scioxide_core::PartialResultTrust::ResidualChecked)?;
        relative_change = Some(step_change);
        location.copy_from_slice(candidate_location);
        scatter
            .as_mut_slice()
            .copy_from_slice(candidate_scatter.as_slice());
        iterations += 1;
        stage.iterations = iterations;
        stage.relative_change = relative_change;
        if step_change <= options.tolerance {
            termination = StatsTermination::Converged;
            break;
        }
    }
    if let Some(stop) = context.stop_reason()? {
        termination = StatsTermination::Stopped(stop);
    }
    Ok(HuberScatterReport {
        termination,
        iterations,
        relative_change,
        counts: context.counts(),
    })
}
