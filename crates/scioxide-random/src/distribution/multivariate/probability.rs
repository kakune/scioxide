//! Box probabilities of multivariate normal and Student t laws by randomized quasi-Monte Carlo.
use super::*;
use crate::float::open_coordinate as open;
#[cfg(test)]
use crate::{Cdf, SeedableRng, Survival};
use crate::{
    InverseSurvival, Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan, Quantile,
    QuasiRng, StudentT,
};
use scioxide_core::{ExecutionContext, ExecutionPolicy, SolveBudgets, SolveScope, Tolerance};
use scioxide_linalg::{HouseholderDecomposition, HouseholderOrientation};

/// Options for [`MultivariateProbabilityPlan`]: tolerance, point counts, limits and scheduling.
///
/// Setters do not validate. `prepare_probability` checks the replicate count,
/// the point count, the evaluation limit and the byte limit. The result does not
/// depend on the execution policy. Any [`Tolerance`] is accepted, including zero.
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_random::MultivariateProbabilityOptions;
/// let options = MultivariateProbabilityOptions::default()
///     .with_points_per_replicate(1024)
///     .with_work_budget(Budget::try_from(8192)?);
/// assert_eq!(options.replicates(), 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MultivariateProbabilityOptions {
    /// Absolute-plus-relative tolerance for the estimated six-standard-error bound.
    tolerance: Tolerance<f64>,
    /// Power-of-two point count per independently randomized Sobol replicate.
    points_per_replicate: usize,
    /// Number of independent randomizations; must be at least two.
    replicates: usize,
    /// Maximum integrand evaluations, including all replicates.
    evaluations: Budget,
    /// Maximum planned payload bytes, excluding allocator metadata.
    max_bytes: usize,
    /// Replicate scheduling.
    execution_policy: ExecutionPolicy,
}
impl Default for MultivariateProbabilityOptions {
    fn default() -> Self {
        Self {
            tolerance: Tolerance::from_sqrt_epsilon(),
            points_per_replicate: 65_536,
            replicates: 8,
            evaluations: Budget::ONE.saturating_add(524_287),
            max_bytes: 64 * 1024 * 1024,
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
}

impl MultivariateProbabilityOptions {
    /// Returns the tolerance on the error estimate: the plan converges when
    /// `estimated_absolute_error ≤ abs_tol + rel_tol · probability`.
    pub const fn tolerance(self) -> Tolerance<f64> {
        self.tolerance
    }
    /// Sets the tolerance.
    pub const fn with_tolerance(mut self, value: Tolerance<f64>) -> Self {
        self.tolerance = value;
        self
    }
    /// Returns the number of Sobol points per replicate, a power of two.
    pub const fn points_per_replicate(self) -> usize {
        self.points_per_replicate
    }
    /// Sets the number of Sobol points per replicate.
    pub const fn with_points_per_replicate(mut self, value: usize) -> Self {
        self.points_per_replicate = value;
        self
    }
    /// Returns the number of independently randomized replicates, at least two.
    pub const fn replicates(self) -> usize {
        self.replicates
    }
    /// Sets the number of replicates.
    pub const fn with_replicates(mut self, value: usize) -> Self {
        self.replicates = value;
        self
    }
    /// Returns the maximum number of integrand evaluations over all replicates.
    pub const fn work_budget(self) -> Budget {
        self.evaluations
    }
    /// Sets the maximum number of integrand evaluations.
    pub const fn with_work_budget(mut self, value: Budget) -> Self {
        self.evaluations = value;
        self
    }
    /// Returns the byte limit on plan and workspace storage, excluding allocator
    /// overhead. Exceeding it is `ResourceLimit`.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
    /// Sets the byte limit.
    pub const fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the scheduling of replicates (default sequential).
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
    /// Sets the scheduling of replicates.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.execution_policy = value;
        self
    }
}
/// Termination status of a box probability evaluation.
/// See [`MultivariateProbabilityPlan`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultivariateProbabilityTermination {
    /// The result is exact, or the error estimate meets the tolerance.
    Converged,
    /// All configured points were evaluated, but the error estimate exceeds the tolerance.
    AccuracyNotMet,
}
/// Box probability with its error estimate.
///
/// The error estimate is statistical, not a bound.
/// See [`MultivariateProbabilityPlan`] for an example.
#[derive(Debug, Clone, Copy)]
pub struct MultivariateProbabilityReport {
    /// Probability, possibly underflowed to zero.
    pub probability: f64,
    /// Natural logarithm of the probability, finite even where `probability` underflows.
    pub log_probability: f64,
    /// Six times the standard error between randomized replicate means.
    /// If every sampled contribution is zero, it is set to 1 and the termination
    /// is `AccuracyNotMet`.
    pub estimated_absolute_error: f64,
    /// Natural logarithm of `estimated_absolute_error`, finite where it underflows.
    /// It is 0 when every sampled contribution is zero.
    pub log_estimated_absolute_error: f64,
    /// Number of integrand evaluations, zero for closed-form cases.
    pub evaluations: usize,
    /// Termination status.
    pub termination: MultivariateProbabilityTermination,
}
impl MultivariateProbabilityReport {
    fn analytic(log_probability: f64) -> Self {
        Self {
            probability: log_probability.exp(),
            log_probability,
            estimated_absolute_error: 0.0,
            log_estimated_absolute_error: f64::NEG_INFINITY,
            evaluations: 0,
            termination: MultivariateProbabilityTermination::Converged,
        }
    }
    fn required(self) -> Result<Self, SciOxideError> {
        if self.termination == MultivariateProbabilityTermination::Converged {
            Ok(self)
        } else {
            Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::EvaluationLimit,
                scioxide_core::ConvergenceDiagnostics::new(),
                "replicate error estimate exceeds tolerance",
            ))
        }
    }
}
/// Plan that computes probabilities of boxes `P(a < X ≤ b)` for a multivariate
/// normal or Student t law.
///
/// Create it with [`MultivariateNormal::prepare_probability`] or
/// [`MultivariateStudentT::prepare_probability`]. The plan uses Genz's
/// separation of variables with variable reordering, integrated by randomized
/// quasi-Monte Carlo: each replicate is an independently scrambled Sobol point
/// set. The error estimate is six standard errors of the replicate means. Rank
/// zero, rank one and diagonal Gaussian factors use closed forms. Unmet
/// accuracy is reported as [`MultivariateProbabilityTermination::AccuracyNotMet`],
/// not as an error. Preparation computes a pivoted lower-trapezoidal factor of
/// the law in O(d² r) and the `R` randomized Sobol plans, so several boxes reuse
/// both.
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_linalg::Matrix;
/// use scioxide_random::{MultivariateNormal, MultivariateProbabilityOptions,
///     MultivariateProbabilityTermination, Philox4x64, SeedableRng};
/// let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0])?;
/// let law = MultivariateNormal::new(&[0.0; 2], &covariance)?;
/// let options = MultivariateProbabilityOptions::default()
///     .with_points_per_replicate(4096)
///     .with_tolerance(Tolerance::new(1e-4, 1e-4)?);
/// let plan = law.prepare_probability(options, &Philox4x64::from_seed(7))?;
/// let report = plan.rectangle(&[f64::NEG_INFINITY; 2], &[0.0; 2])?;
/// // Orthant probability 1/4 + asin(rho) / (2 pi) = 1/3 for rho = 1/2.
/// assert_eq!(report.termination, MultivariateProbabilityTermination::Converged);
/// assert!((report.probability - 1.0 / 3.0).abs() < 1e-4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Genz, "Numerical computation of multivariate normal probabilities",
///   *J. Comput. Graph. Statist.* 1(2), 1992, 141–149.
/// - A. Genz and F. Bretz, "Comparison of methods for the computation of
///   multivariate t probabilities", *J. Comput. Graph. Statist.* 11(4), 2002, 950–971.
#[derive(Debug, Clone)]
pub struct MultivariateProbabilityPlan {
    mean: Vec<f64>,
    factor: Matrix<f64>,
    permutation: Vec<usize>,
    df: DegreesOfFreedom,
    qmc: QmcReplicatePlan,
    options: MultivariateProbabilityOptions,
}
/// Workspace for [`MultivariateProbabilityPlan::rectangle_scoped`].
///
/// Create it with [`MultivariateProbabilityPlan::workspace`]. It fits only the
/// plan that created it, or one of the same shape. See
/// [`MultivariateProbabilityPlan`] for an example.
#[derive(Debug, Clone)]
pub struct MultivariateProbabilityWorkspace {
    factor: Matrix<f64>,
    permutation: Vec<usize>,
    low: Vec<f64>,
    high: Vec<f64>,
    latent: Vec<Vec<f64>>,
    points: Vec<Vec<f64>>,
    means: Vec<f64>,
}
impl MultivariateProbabilityPlan {
    pub(super) fn prepare(
        mean: Vec<f64>,
        factor: Matrix<f64>,
        df: DegreesOfFreedom,
        options: MultivariateProbabilityOptions,
        root: &Philox4x64,
    ) -> Result<Self, SciOxideError> {
        let (d, r) = (factor.rows(), factor.cols());
        let qdim = preflight(d, r, df, options)?;
        // Genz's separation of variables needs a lower-trapezoidal factor L
        // (d x r) with positive diagonal, so X = mu + L Z can be integrated one
        // latent coordinate at a time. A pivoted LQ of the support factor gives
        // it (pivoting keeps rank-deficient laws well ordered). Negating a
        // column of L together with its latent Z_j leaves the law unchanged,
        // which is what the sign normalization relies on.
        let (mut triangular, mut permutation) = (Matrix::zeros(d, r)?, indices(d)?);
        if r > 0 {
            let qr = HouseholderDecomposition::factor(
                &factor,
                HouseholderOrientation::Lq,
                scioxide_linalg::HouseholderPivoting::Column,
            )?;
            let upper = qr.to_triangular_matrix()?;
            permutation.copy_from_slice(qr.permutation());
            for j in 0..r {
                let diagonal = upper.view().get(j, j)?;
                if diagonal == 0.0 {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "retained support factor lost rank during probability preparation",
                    ));
                }
                let sign = if diagonal < 0.0 { -1.0 } else { 1.0 };
                for i in j..d {
                    triangular
                        .view_mut()
                        .set(i, j, upper.view().get(i, j)? * sign)?;
                }
            }
        }
        let qmc = QmcReplicatePlan::new(
            &QmcPlan::sobol(qdim)?,
            root,
            QmcReplicateOptions::default()
                .with_replicates(options.replicates)
                .with_points_per_replicate(options.points_per_replicate)
                .with_execution_policy(options.execution_policy),
        )?;
        Ok(Self {
            mean,
            factor: triangular,
            permutation,
            df,
            qmc,
            options,
        })
    }
    /// Creates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(&self) -> Result<MultivariateProbabilityWorkspace, SciOxideError> {
        let mut latent = scioxide_core::backend::try_with_capacity(
            self.options.replicates,
            scioxide_core::quantity!("probability_latent_work"),
        )?;
        let mut points = scioxide_core::backend::try_with_capacity(
            self.options.replicates,
            scioxide_core::quantity!("probability_qmc_work"),
        )?;
        for plan in self.qmc.plans() {
            latent.push(scioxide_core::backend::try_filled(
                self.factor.cols(),
                0.0,
                scioxide_core::quantity!("probability_latent"),
            )?);
            points.push(scioxide_core::backend::try_filled(
                plan.dim(),
                0.0,
                scioxide_core::quantity!("probability_coordinates"),
            )?);
        }
        Ok(MultivariateProbabilityWorkspace {
            factor: self.factor.view().try_to_matrix()?,
            permutation: indices(self.mean.len())?,
            low: scioxide_core::backend::try_filled(
                self.mean.len(),
                0.0,
                scioxide_core::quantity!("probability_lower_bounds"),
            )?,
            high: scioxide_core::backend::try_filled(
                self.mean.len(),
                0.0,
                scioxide_core::quantity!("probability_upper_bounds"),
            )?,
            latent,
            points,
            means: scioxide_core::backend::try_filled(
                self.options.replicates,
                0.0,
                scioxide_core::quantity!("probability_replicate_means"),
            )?,
        })
    }
    /// Computes `P(lowerᵢ < Xᵢ ≤ upperᵢ for every i)`.
    ///
    /// Infinite bounds are allowed. The evaluation uses exactly the configured
    /// number of points, `R · N` integrand evaluations of O(d r) each for `R`
    /// replicates of `N` points and a factor of rank `r`. It runs on the
    /// calling thread with a new workspace; [`Self::rectangle_scoped`] reuses a
    /// workspace and runs replicates on the executor of a scope. The result does
    /// not depend on the schedule. Unmet accuracy is reported in
    /// [`MultivariateProbabilityReport::termination`], not as an error.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `lower` or `upper` does not have length `d`.
    /// - `InvalidInput` if a bound is NaN or if `lowerᵢ > upperᵢ`.
    /// - `NumericalInstability` if a finite bound minus the mean overflows.
    /// - `AllocationFailure` if the workspace, or the O(R d) replicate cursors and schedule
    ///   storage, cannot be allocated.
    pub fn rectangle(
        &self,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<MultivariateProbabilityReport, SciOxideError> {
        self.rectangle_scoped(
            lower,
            upper,
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )
    }
    /// Computes the box probability of [`Self::rectangle`] in `work`, as one epoch of `scope`.
    ///
    /// `scope` is an [`ExecutionContext`] for a top-level call or the `&mut SolveContext` of an
    /// enclosing solve. Replicates run on the executor of the scope under the execution policy
    /// of the options, and the result does not depend on the schedule, so a sequential scope
    /// gives the report of [`Self::rectangle`] bit for bit. The call observes the cancellation
    /// and deadline of the scope before each integrand evaluation. It calls no caller function,
    /// so it counts no work in the scope. The workspace contents are unspecified after an
    /// error.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::rectangle`].
    /// - `InvalidInput` if `work` belongs to a plan of another shape.
    /// - `ExecutionStopped` if the scope is cancelled or past its deadline, and executor
    ///   errors.
    /// - `AccountingInvariant` if the counts of an enclosing solve are inconsistent.
    pub fn rectangle_scoped<'scope, 'capability>(
        &self,
        lower: &[f64],
        upper: &[f64],
        work: &mut MultivariateProbabilityWorkspace,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<MultivariateProbabilityReport, SciOxideError> {
        crate::law::run_scoped(
            scope,
            scioxide_core::algorithm_id!("random", "MultivariateProbability", 1),
            SolveBudgets::unbounded(),
            |context| self.evaluate(lower, upper, work, context.execution()),
        )
    }
    fn evaluate(
        &self,
        lower: &[f64],
        upper: &[f64],
        work: &mut MultivariateProbabilityWorkspace,
        execution: ExecutionContext<'_>,
    ) -> Result<MultivariateProbabilityReport, SciOxideError> {
        let d = self.mean.len();
        let r = self.factor.cols();
        size(lower.len(), d)?;
        size(upper.len(), d)?;
        if work.low.len() != d
            || work.high.len() != d
            || work.latent.len() != self.options.replicates
            || work.points.len() != self.options.replicates
            || work.means.len() != self.options.replicates
            || work.latent.iter().any(|x| x.len() != r)
            || work
                .points
                .iter()
                .zip(self.qmc.plans())
                .any(|(x, p)| x.len() != p.dim())
        {
            return Err(invalid(
                scioxide_core::argument!("work"),
                "probability workspace belongs to another shape",
            ));
        }
        execution.check()?;
        self.order_for_bounds(lower, upper, work)?;
        for i in 0..d {
            let j = work.permutation[i];
            if lower[j].is_nan() || upper[j].is_nan() || lower[j] > upper[j] {
                return Err(invalid(
                    if lower[j].is_nan() {
                        scioxide_core::argument!("lower")
                    } else {
                        scioxide_core::argument!("upper")
                    },
                    "invalid rectangle bounds",
                ));
            }
            work.low[i] = lower[j] - self.mean[j];
            work.high[i] = upper[j] - self.mean[j];
            if work.low[i].is_nan()
                || work.high[i].is_nan()
                || (lower[j].is_finite() && !work.low[i].is_finite())
                || (upper[j].is_finite() && !work.high[i].is_finite())
            {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "unrepresentable centered probability bounds",
                ));
            }
        }
        if lower.iter().zip(upper).any(|(a, b)| a == b) {
            return Ok(MultivariateProbabilityReport::analytic(f64::NEG_INFINITY));
        }
        if r == 0 {
            return Ok(MultivariateProbabilityReport::analytic(
                if work
                    .low
                    .iter()
                    .zip(&work.high)
                    .all(|(&a, &b)| a < 0.0 && b >= 0.0)
                {
                    0.0
                } else {
                    f64::NEG_INFINITY
                },
            ));
        }
        // Closed forms before Monte Carlo: rank 1 is a single interval
        // probability of the latent variable, and a diagonal full-rank Gaussian
        // factor makes the coordinates independent.
        if r == 1 {
            let (a, b) = self.final_bounds(&work.factor, 0, &[], &work.low, &work.high, 1.0)?;
            let log = if a >= b {
                f64::NEG_INFINITY
            } else {
                match self.df {
                    DegreesOfFreedom::Gaussian => {
                        crate::log_interval_probability(&Normal::<f64>::standard(), a, b)?
                    }
                    DegreesOfFreedom::StudentT(df) => {
                        crate::log_interval_probability(&StudentT::new(df)?, a, b)?
                    }
                }
            };
            return Ok(MultivariateProbabilityReport::analytic(log));
        }
        if matches!(self.df, DegreesOfFreedom::Gaussian)
            && r == d
            && (0..d).all(|i| (0..i).all(|j| work.factor.as_slice()[i * r + j] == 0.0))
        {
            let normal = Normal::<f64>::standard();
            let mut log = 0.0;
            for i in 0..d {
                log += crate::log_interval_probability(
                    &normal,
                    work.low[i] / work.factor.view().get(i, i)?,
                    work.high[i] / work.factor.view().get(i, i)?,
                )?;
            }
            return Ok(MultivariateProbabilityReport::analytic(log));
        }
        // Randomized QMC: each independently scrambled Sobol replicate gives an
        // unbiased estimate, and their spread yields an honest error estimate.
        // Every replicate owns disjoint scratch; results do not depend on the
        // schedule.
        let scratch = work.points.iter_mut().zip(work.latent.iter_mut());
        self.qmc.execute(
            execution,
            scratch,
            &mut work.means,
            |sequence, (point, latent)| {
                let mut logsum = f64::NEG_INFINITY;
                for _ in 0..self.options.points_per_replicate {
                    execution.check()?;
                    sequence.next_point_into(point)?;
                    logsum = logadd(
                        logsum,
                        self.integrand(&work.factor, point, latent, &work.low, &work.high)?,
                    )?;
                }
                Ok(logsum - (self.options.points_per_replicate as f64).ln())
            },
        )?;
        // Replicate means are logs; rescale by the largest before averaging so
        // tiny probabilities do not underflow. The standard error of the mean
        // is widened by a factor 6 to form a conservative error bound.
        let max = work.means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max == f64::NEG_INFINITY {
            return Ok(MultivariateProbabilityReport {
                probability: 0.0,
                log_probability: max,
                estimated_absolute_error: 1.0,
                log_estimated_absolute_error: 0.0,
                evaluations: self.options.replicates * self.options.points_per_replicate,
                termination: MultivariateProbabilityTermination::AccuracyNotMet,
            });
        }
        for mean in &mut work.means {
            *mean = (*mean - max).exp();
        }
        let (average, error) = crate::quasi::replicate_moments(&work.means)?;
        let log_probability = crate::law::log_probability(max + average.ln())?;
        let logerror = max + (6.0 * error).ln();
        let logtolerance = logadd(
            self.options.tolerance.absolute().ln(),
            self.options.tolerance.relative().ln() + log_probability,
        )?;
        Ok(MultivariateProbabilityReport {
            probability: log_probability.exp(),
            log_probability,
            estimated_absolute_error: logerror.exp(),
            log_estimated_absolute_error: logerror,
            evaluations: self.options.replicates * self.options.points_per_replicate,
            termination: if logerror <= logtolerance {
                MultivariateProbabilityTermination::Converged
            } else {
                MultivariateProbabilityTermination::AccuracyNotMet
            },
        })
    }
    fn order_for_bounds(
        &self,
        lower: &[f64],
        upper: &[f64],
        work: &mut MultivariateProbabilityWorkspace,
    ) -> Result<(), SciOxideError> {
        let (d, r) = (self.mean.len(), self.factor.cols());
        // Genz-Bretz variable reordering: integrate the most constraining
        // coordinates (smallest marginal interval probability) first. This
        // concentrates the variation in the leading QMC dimensions and markedly
        // reduces the variance of the estimator.
        let mut order = scioxide_core::backend::try_with_capacity(
            d,
            scioxide_core::quantity!("conditional_ordering"),
        )?;
        let normal = Normal::<f64>::standard();
        for i in 0..d {
            let j = self.permutation[i];
            if lower[j].is_nan() || upper[j].is_nan() || lower[j] > upper[j] {
                return Err(invalid(
                    if lower[j].is_nan() {
                        scioxide_core::argument!("lower")
                    } else {
                        scioxide_core::argument!("upper")
                    },
                    "invalid rectangle bounds",
                ));
            }
            let norm = (0..r).try_fold(0.0_f64, |s, k| {
                Ok::<_, SciOxideError>(s.hypot(self.factor.view().get(i, k)?))
            })?;
            let score = if i < r {
                crate::log_interval_probability(
                    &normal,
                    (lower[j] - self.mean[j]) / norm,
                    (upper[j] - self.mean[j]) / norm,
                )?
            } else {
                0.0
            };
            order.push((i, score));
        }
        order[..r].sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for (i, &(j, _)) in order.iter().enumerate() {
            work.permutation[i] = self.permutation[j];
        }
        if r == 0 || order.iter().enumerate().all(|(i, (j, _))| i == *j) {
            work.factor
                .as_mut_slice()
                .copy_from_slice(self.factor.as_slice());
            return Ok(());
        }
        // Reversing an unpivoted RQ gives LQ in the requested constraint order.
        // Only independent leading rows are permuted; trailing constraints stay explicit.
        let mut reversed = Matrix::zeros(d, r)?;
        for i in 0..d {
            for j in 0..r {
                reversed.view_mut().set(
                    i,
                    j,
                    self.factor.view().get(order[d - 1 - i].0, r - 1 - j)?,
                )?;
            }
        }
        let rq = scioxide_linalg::RqDecomposition::factor(reversed.view())?;
        let upper = rq.r_matrix();
        for j in 0..r {
            let diag = upper.view().get(d - 1 - j, r - 1 - j)?;
            if diag == 0.0 {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "conditional ordering lost support rank",
                ));
            }
            let sign = if diag < 0.0 { -1.0 } else { 1.0 };
            for i in 0..d {
                work.factor
                    .view_mut()
                    .set(i, j, upper.view().get(d - 1 - i, r - 1 - j)? * sign)?;
            }
        }
        Ok(())
    }
    fn bound(x: f64, scale: f64) -> f64 {
        if x.is_infinite() { x } else { x * scale }
    }
    fn final_bounds(
        &self,
        factor: &Matrix<f64>,
        j: usize,
        z: &[f64],
        low: &[f64],
        high: &[f64],
        scale: f64,
    ) -> Result<(f64, f64), SciOxideError> {
        // The last latent coordinate must satisfy all remaining (possibly
        // redundant, for rank-deficient laws) constraints simultaneously, so its
        // interval is the intersection of every row's bound on it. A zero
        // coefficient makes the row a pure feasibility check.
        let mut a = f64::NEG_INFINITY;
        let mut b = f64::INFINITY;
        for i in j..self.mean.len() {
            let mut shift = 0.0;
            for (k, &v) in z.iter().enumerate().take(j) {
                shift += factor.view().get(i, k)? * v;
            }
            let c = factor.view().get(i, j)?;
            let lo = Self::bound(low[i], scale) - shift;
            let hi = Self::bound(high[i], scale) - shift;
            if c == 0.0 {
                if lo >= 0.0 || hi < 0.0 {
                    return Ok((1.0, 0.0));
                }
            } else if c > 0.0 {
                a = a.max(lo / c);
                b = b.min(hi / c);
            } else {
                a = a.max(hi / c);
                b = b.min(lo / c);
            }
        }
        Ok((a, b))
    }
    fn integrand(
        &self,
        factor: &Matrix<f64>,
        point: &[f64],
        z: &mut [f64],
        low: &[f64],
        high: &[f64],
    ) -> Result<f64, SciOxideError> {
        // One Genz integrand evaluation. For the t law the extra QMC coordinate
        // selects the chi scale by inversion, scaling the bounds. Then for each
        // latent j the conditional interval [a, b] is given by the previously
        // fixed z_k; its normal mass multiplies the weight, and z_j is drawn
        // inside [a, b] by inversion of the truncated normal.
        let normal = Normal::<f64>::standard();
        let r = z.len();
        let scale = match self.df {
            DegreesOfFreedom::Gaussian => 1.0,
            DegreesOfFreedom::StudentT(df) => {
                let q = Gamma::new(df * 0.5, 2.0)?.quantile(open(point[r]))?;
                crate::law::positive_result((q / df).sqrt())?
            }
        };
        let mut logweight = 0.0;
        for j in 0..r {
            let (a, b) = if j == r - 1 {
                self.final_bounds(factor, j, z, low, high, scale)?
            } else {
                let shift = (0..j).try_fold(0.0, |s, k| {
                    Ok::<_, SciOxideError>(s + factor.view().get(j, k)? * z[k])
                })?;
                let diagonal = factor.view().get(j, j)?;
                (
                    (Self::bound(low[j], scale) - shift) / diagonal,
                    (Self::bound(high[j], scale) - shift) / diagonal,
                )
            };
            if a >= b {
                return Ok(f64::NEG_INFINITY);
            }
            let mass = crate::log_interval_probability(&normal, a, b)?;
            if mass == f64::NEG_INFINITY {
                return Ok(mass);
            }
            logweight += mass;
            if j + 1 < r {
                // Truncated-normal inversion Phi(a) + u (Phi(b) - Phi(a)),
                // formed in log space from whichever end gives the smaller
                // probability; far tails (below e^-700) use the log-domain
                // inverter because `exp` would underflow.
                let u = open(point[j]);
                let lp = logadd(
                    crate::distribution::special::normal_log(a, false)?,
                    u.ln() + mass,
                )?;
                // The lower target already identifies whether it is the small
                // tail. Form the upper target only when needed, especially for
                // half-lines where the lower log sum has an exact zero term.
                let upper = lp > -core::f64::consts::LN_2;
                let p = if upper {
                    logadd(
                        crate::distribution::special::normal_log(b, true)?,
                        (-u).ln_1p() + mass,
                    )?
                } else {
                    lp
                };
                z[j] = if p > -700.0 {
                    if upper {
                        normal.isf(p.exp())?
                    } else {
                        normal.quantile(p.exp())?
                    }
                } else {
                    if upper {
                        scioxide_special::normal_inv_log_sf(p)?
                    } else {
                        scioxide_special::normal_inv_log_cdf(p)?
                    }
                };
            }
        }
        finite(logweight)
    }
}
pub(super) fn convenience<S: RealScalar>(
    plan: MultivariateProbabilityPlan,
    x: &[S],
    upper: bool,
) -> Result<MultivariateProbabilityReport, SciOxideError> {
    size(x.len(), plan.mean.len())?;
    let x = widened(x)?;
    let other = scioxide_core::backend::try_filled(
        x.len(),
        if upper {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        },
        scioxide_core::quantity!("orthant_bounds"),
    )?;
    let (low, high) = if upper { (&x, &other) } else { (&other, &x) };
    plan.rectangle(low, high)?.required()
}
/// Box probability of a law through a plan prepared with its own options.
pub(super) fn law_rectangle<S: RealScalar>(
    plan: &MultivariateProbabilityPlan,
    lower: &[S],
    upper: &[S],
) -> Result<S, SciOxideError> {
    number(
        plan.rectangle(&widened(lower)?, &widened(upper)?)?
            .required()?
            .probability,
    )
}
/// Bounds widened to `f64`; infinite bounds stay infinite.
fn widened<S: RealScalar>(x: &[S]) -> Result<Vec<f64>, SciOxideError> {
    let mut out = scioxide_core::backend::try_with_capacity(
        x.len(),
        scioxide_core::quantity!("orthant_coordinates"),
    )?;
    out.extend(x.iter().map(|v| v.into_f64()));
    Ok(out)
}
// Convenience queries inherit their law's numerical and storage policy. Explicit
// prepared plans keep their independently configured replicate design.
pub(super) fn law_probability_options<S: RealScalar>(
    law: crate::LawOptions<S>,
) -> Result<MultivariateProbabilityOptions, SciOxideError> {
    let work = law.work_budget().get();
    if work < 2 {
        return Err(crate::law::exhausted());
    }
    let replicates = work.clamp(2, 8);
    let points = 1usize << (work / replicates).ilog2();
    Ok(MultivariateProbabilityOptions::default()
        .with_tolerance(Tolerance::new(
            law.tolerance().absolute().into_f64(),
            law.tolerance().relative().into_f64(),
        )?)
        .with_replicates(replicates)
        .with_points_per_replicate(points)
        .with_work_budget(law.work_budget())
        .with_max_bytes(law.max_bytes())
        .with_execution_policy(law.execution_policy()))
}
// Joint tail capabilities and box probabilities evaluate through a default
// probability plan; `$example` documents `rectangle` for the law.
macro_rules! joint_traits {
    ($law:ident, $(#[$example:meta])* ; $($bound:tt)*) => {
        impl<S: RealScalar $($bound)*> $law<S> {
            /// Returns `P(lowerᵢ < Xᵢ ≤ upperᵢ for every i)`, the probability of a box.
            ///
            /// Infinite bounds are allowed, and `rectangle(&[−∞; d], x, &Philox4x64::from_seed(0))`
            /// equals [`Cdf::cdf`] at `x`. The method prepares a
            /// [`MultivariateProbabilityPlan`] with the tolerance, work limit and byte limit of the
            /// law's options and randomizations drawn from `root`, which does not advance, and
            /// evaluates the box on the calling thread: at most `work_budget` integrand
            /// evaluations of O(d²) each. The estimate meets the tolerance of the options with an
            /// error estimate of six standard errors, a statistical measure, not a bound. A
            /// correlated law of rank 2 or more is integrated by quasi-Monte Carlo and rarely
            /// meets the default tolerance `√ε`; construct it with a looser tolerance, as in the
            /// example. Use [`Self::prepare_probability`] to evaluate several boxes with one plan
            /// or to read an unmet accuracy from the report.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `lower` or `upper` does not have `d` elements.
            /// - `InvalidInput` if a bound is NaN or `lowerᵢ > upperᵢ`.
            /// - `ConvergenceFailure` if the error estimate exceeds the tolerance
            ///   (`EvaluationLimit`), or the work limit allows fewer than two evaluations.
            /// - `NumericalInstability` if a finite bound minus the mean overflows or the
            ///   probability is not representable in `S`.
            /// - `ResourceLimit` if the plan exceeds the byte limit, and `AllocationFailure`.
            $(#[$example])*
            pub fn rectangle(
                &self,
                lower: &[S],
                upper: &[S],
                root: &Philox4x64,
            ) -> Result<S, SciOxideError> {
                let plan = self.prepare_probability(self.joint_options()?, root)?;
                super::probability::law_rectangle(&plan, lower, upper)
            }
        }
        impl<S: RealScalar $($bound)*> Cdf<S, &[S]> for $law<S> {
            fn cdf(&self, x: &[S]) -> Result<S, SciOxideError> {

                let plan = self.prepare_probability(
                    self.joint_options()?,
                    &Philox4x64::from_seed(0),
                )?;
                number(convenience(plan, x, false)?.probability)
            }
        }
        impl<S: RealScalar $($bound)*> LogCdf<S, &[S]> for $law<S> {
            fn log_cdf(&self, x: &[S]) -> Result<S, SciOxideError> {

                let plan = self.prepare_probability(
                    self.joint_options()?,
                    &Philox4x64::from_seed(0),
                )?;
                number(convenience(plan, x, false)?.log_probability)
            }
        }
        impl<S: RealScalar $($bound)*> Survival<S, &[S]> for $law<S> {
            fn sf(&self, x: &[S]) -> Result<S, SciOxideError> {

                let plan = self.prepare_probability(
                    self.joint_options()?,
                    &Philox4x64::from_seed(0),
                )?;
                number(convenience(plan, x, true)?.probability)
            }
        }
        impl<S: RealScalar $($bound)*> LogSurvival<S, &[S]> for $law<S> {
            fn log_sf(&self, x: &[S]) -> Result<S, SciOxideError> {

                let plan = self.prepare_probability(
                    self.joint_options()?,
                    &Philox4x64::from_seed(0),
                )?;
                number(convenience(plan, x, true)?.log_probability)
            }
        }
    };
}
pub(super) use joint_traits;

pub(super) fn preflight(
    d: usize,
    r: usize,
    df: DegreesOfFreedom,
    options: MultivariateProbabilityOptions,
) -> Result<usize, SciOxideError> {
    if options.replicates < 2 || !options.points_per_replicate.is_power_of_two() {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "joint integration requires >=2 replicates and power-of-two point count",
        ));
    }
    let evaluations = options
        .replicates
        .checked_mul(options.points_per_replicate)
        .ok_or_else(|| {
            invalid(
                scioxide_core::argument!("options"),
                "probability evaluation count overflow",
            )
        })?;
    if evaluations > options.evaluations.get() {
        return Err(super::super::count_math::exhausted());
    }
    let qdim = (r + usize::from(matches!(df, DegreesOfFreedom::StudentT(_)))).max(1);
    let bytes = d
        .checked_mul(d)
        .and_then(|v| v.checked_mul(128))
        .and_then(|v| {
            options
                .replicates
                .checked_mul(qdim)
                .and_then(|q| q.checked_mul(2048))
                .and_then(|q| v.checked_add(q))
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("workspace_bytes")))?;
    crate::workspace::admit_bytes(bytes, options.max_bytes)?;
    Ok(qdim)
}

fn indices(n: usize) -> Result<Vec<usize>, SciOxideError> {
    let mut out = scioxide_core::backend::try_with_capacity(
        n,
        scioxide_core::quantity!("probability_permutation"),
    )?;
    out.extend(0..n);
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::LogSurvival;

    fn options() -> MultivariateProbabilityOptions {
        MultivariateProbabilityOptions {
            points_per_replicate: 2048,
            replicates: 8,
            tolerance: Tolerance::new(2e-5, 2e-5).unwrap(),
            ..Default::default()
        }
    }
    #[test]
    fn correlated_gaussian_and_t_orthants() {
        for rho in [-0.7, 0.5] {
            let shape = Matrix::from_vec(2, 2, vec![1.0, rho, rho, 1.0]).unwrap();
            let g = MultivariateNormal::new(&[0.0, 0.0], &shape).unwrap();
            let root = Philox4x64::from_seed(542);
            let plan = g.prepare_probability(options(), &root).unwrap();
            let expected = 0.25 + rho.asin() / (2.0 * core::f64::consts::PI);
            let report = plan.rectangle(&[f64::NEG_INFINITY; 2], &[0.0; 2]).unwrap();
            assert!(
                (report.probability - expected).abs() < 2e-5,
                "{report:?} != {expected}"
            );
            assert_eq!(
                report.termination,
                MultivariateProbabilityTermination::Converged
            );
            let t = MultivariateStudentT::with_options(
                &[0.0, 0.0],
                &shape,
                3.0,
                Default::default(),
                crate::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap();
            let plan = t.prepare_probability(options(), &root).unwrap();
            let report = plan.rectangle(&[f64::NEG_INFINITY; 2], &[0.0; 2]).unwrap();
            assert!(
                (report.probability - expected).abs() < 2e-5,
                "{report:?} != {expected}"
            );
        }
    }
    #[test]
    fn singular_constraints_and_point_mass_boundaries() {
        let shape = Matrix::from_vec(2, 2, vec![1.0, -1.0, -1.0, 1.0]).unwrap();
        let g = MultivariateNormal::new(&[0.0, 0.0], &shape).unwrap();
        assert_eq!(g.cdf(&[0.0, 0.0]).unwrap(), 0.0);
        let shape = Matrix::zeros(2, 2).unwrap();
        let g = MultivariateNormal::new(&[1.0, 2.0], &shape).unwrap();
        assert_eq!(g.cdf(&[1.0, 2.0]).unwrap(), 1.0);
        assert_eq!(g.sf(&[1.0, 2.0]).unwrap(), 0.0);
        let shape =
            Matrix::from_vec(3, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0]).unwrap();
        let g = MultivariateNormal::new(&[0.0; 3], &shape).unwrap();
        let plan = g
            .prepare_probability(options(), &Philox4x64::from_seed(5))
            .unwrap();
        let report = plan.rectangle(&[f64::NEG_INFINITY; 3], &[0.0; 3]).unwrap();
        assert!((report.probability - 0.25).abs() < 2e-5, "{report:?}");
    }
    #[test]
    fn independent_log_tails_and_resources() {
        let g = MultivariateNormal::new(&[0.0; 2], &Matrix::identity(2).unwrap()).unwrap();
        let expected = 2.0 * Normal::new(0.0, 1.0).unwrap().log_sf(40.0).unwrap();
        assert!((g.log_sf(&[40.0, 40.0]).unwrap() - expected).abs() < 1e-10);
        assert_eq!(g.sf(&[40.0, 40.0]).unwrap(), 0.0);
        let mut o = options();
        o.max_bytes = 1;
        assert_eq!(
            g.prepare_probability(o, &Philox4x64::from_seed(1))
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        o = options();
        o.evaluations = Budget::ONE;
        assert!(g.prepare_probability(o, &Philox4x64::from_seed(1)).is_err());
        let plan = g
            .prepare_probability(options(), &Philox4x64::from_seed(1))
            .unwrap();
        assert!(plan.rectangle(&[f64::NAN; 2], &[1.0; 2]).is_err());
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod execution_tests {
    use super::*;
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn, SequentialExecutor};

    struct Reverse;
    impl Executor for Reverse {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            let results = std::thread::scope(|threads| {
                let mut handles = Vec::new();
                for index in (0..plan.worker_count()).rev() {
                    handles.push(threads.spawn(move || {
                        SequentialExecutor.execute(context, plan, &|part, (), scope| {
                            if part.worker_index() == index {
                                operation(part, (), scope)
                            } else {
                                Ok(())
                            }
                        })
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for result in results {
                result?;
            }
            SequentialExecutor.execute(context, plan, &|_, (), _| Ok(()))
        }
    }
    #[test]
    fn executor_reordering_is_reproducible_and_accuracy_failure_explicit() {
        let g = MultivariateNormal::new(
            &[0.0; 2],
            &Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]).unwrap(),
        )
        .unwrap();
        let options = MultivariateProbabilityOptions {
            points_per_replicate: 32,
            replicates: 8,
            tolerance: Tolerance::new(0.0, 0.0).unwrap(),
            ..Default::default()
        };
        let p = g
            .prepare_probability(options, &Philox4x64::from_seed(542))
            .unwrap();
        let serial = p.rectangle(&[f64::NEG_INFINITY; 2], &[0.25, -0.5]).unwrap();
        assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
        let parallel_plan = g
            .prepare_probability(
                options.with_execution_policy(ExecutionPolicy::Parallel(
                    NonZeroUsize::new(4).unwrap(),
                )),
                &Philox4x64::from_seed(542),
            )
            .unwrap();
        let parallel = parallel_plan
            .rectangle_scoped(
                &[f64::NEG_INFINITY; 2],
                &[0.25, -0.5],
                &mut parallel_plan.workspace().unwrap(),
                ExecutionContext::new(&Reverse),
            )
            .unwrap();
        assert_eq!(serial.log_probability, parallel.log_probability);
        assert_eq!(
            serial.log_estimated_absolute_error,
            parallel.log_estimated_absolute_error
        );
        assert_eq!(serial.evaluations, 256);
        assert_eq!(
            serial.termination,
            MultivariateProbabilityTermination::AccuracyNotMet
        );
        let stop = scioxide_core::CancellationToken::new();
        stop.cancel();
        assert!(
            p.rectangle_scoped(
                &[f64::NEG_INFINITY; 2],
                &[0.0; 2],
                &mut p.workspace().unwrap(),
                ExecutionContext::sequential().with_cancellation(&stop),
            )
            .is_err()
        );
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    use scioxide_core::{NumericalIssue, SciOxideErrorKind};
    fn options() -> MultivariateProbabilityOptions {
        MultivariateProbabilityOptions::default()
            .with_points_per_replicate(4)
            .with_replicates(2)
    }
    #[test]
    fn all_zero_integrands_keep_a_finite_conservative_uncertainty() {
        let factor = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let plan = MultivariateProbabilityPlan::prepare(
            vec![0.0; 3],
            factor,
            DegreesOfFreedom::Gaussian,
            options(),
            &Philox4x64::from_seed(1),
        )
        .unwrap();
        let report = plan
            .rectangle(
                &[1.0, 1.0, f64::NEG_INFINITY],
                &[f64::INFINITY, f64::INFINITY, 0.0],
            )
            .unwrap();
        assert_eq!(report.probability, 0.0);
        assert_eq!(report.log_probability, f64::NEG_INFINITY);
        assert_eq!(report.estimated_absolute_error, 1.0);
        assert_eq!(report.log_estimated_absolute_error, 0.0);
        assert_eq!(report.evaluations, 8);
        assert_eq!(
            report.termination,
            MultivariateProbabilityTermination::AccuracyNotMet
        );
        assert_eq!(
            report.required().unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        let exact = plan.rectangle(&[0.0; 3], &[0.0; 3]).unwrap();
        assert_eq!(
            exact.termination,
            MultivariateProbabilityTermination::Converged
        );
        assert_eq!(exact.estimated_absolute_error, 0.0);
    }
    #[test]
    fn lost_support_rank_is_a_breakdown() {
        crate::error_assertions::numerical(
            MultivariateProbabilityPlan::prepare(
                vec![0.0; 2],
                Matrix::zeros(2, 2).unwrap(),
                DegreesOfFreedom::Gaussian,
                options(),
                &Philox4x64::from_seed(1),
            )
            .unwrap_err(),
            NumericalIssue::Breakdown,
        );
        let mut plan = MultivariateProbabilityPlan::prepare(
            vec![0.0; 2],
            Matrix::identity(2).unwrap(),
            DegreesOfFreedom::Gaussian,
            options(),
            &Philox4x64::from_seed(1),
        )
        .unwrap();
        // Simulate two retained rows becoming dependent during numerical reordering.
        plan.factor = Matrix::from_vec(2, 2, vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        crate::error_assertions::numerical(
            plan.order_for_bounds(&[-2.0, -0.1], &[2.0, 0.1], &mut plan.workspace().unwrap())
                .unwrap_err(),
            NumericalIssue::Breakdown,
        );
    }
}
