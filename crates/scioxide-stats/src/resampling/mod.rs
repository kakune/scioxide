//! Bootstrap, jackknife, permutation and Monte Carlo inference for a scalar
//! statistic.
use crate::StatsTermination;
use crate::error_helpers::{callback_finite as finite, invalid};
mod execution;
pub use execution::{
    IndexedReplicates, ReplicateExecution, ReplicateOptions, SequentialReplicates,
};
mod convenience;
mod design;
mod inference;
mod runner;
pub use convenience::{ResamplingOutput, bootstrap, jackknife, monte_carlo_test, permutation_test};

use crate::ConfidenceProbability;
use alloc::vec::Vec;
pub(crate) use runner::NoRandom;
pub use runner::ResamplingWorkspace;
use scioxide_core::{Interval, RealScalar, SciOxideError, Tolerance};
use scioxide_random::{BitGenerator, Rng};

/// Resampling scheme of a bootstrap or jackknife.
///
/// The data come in one or more groups. The bootstrap draws observations with
/// replacement and keeps the size of each group.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, *Bootstrap Methods and Their
/// Application*, Cambridge University Press, 1997, ch. 2–3 and ch. 8 (block
/// designs).
///
/// H. R. Künsch, "The jackknife and the bootstrap for general stationary
/// observations", *Ann. Statist.* 17 (1989), 1217–1241.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapDesign {
    /// Resamples each group independently of the others.
    Independent,
    /// Resamples rows: draws row indices with replacement and uses the same
    /// rows in every group. The groups must have equal lengths. The jackknife
    /// deletes one row from all groups.
    Paired,
    /// Treats the groups as strata with fixed sizes. Each stratum is resampled
    /// independently, as with `Independent`. The statistic should weight the
    /// strata by their fixed sizes.
    Stratified,
    /// Moving block bootstrap of a single series of length `n`.
    ///
    /// The resample concatenates blocks of `length` consecutive observations.
    /// Each block starts at one of the `n − length + 1` positions, drawn
    /// uniformly, and the last block is cut to make the total `n`.
    MovingBlock {
        /// Number of consecutive observations per block, in `1..=n`.
        length: usize,
    },
    /// Circular block bootstrap of a single series of length `n`.
    ///
    /// As `MovingBlock`, but a block may start at any of the `n` positions and
    /// wraps around from the end of the series to its start.
    CircularBlock {
        /// Number of consecutive observations per block, in `1..=n`.
        length: usize,
    },
}
/// Construction of a two-sided bootstrap confidence interval.
///
/// The variants use the observed statistic `θ̂`, the `B` replicate statistics
/// `θ*`, `α = 1 − confidence`, and the Hyndman–Fan type 7 sample quantile
/// `q(p)` of the replicates.
///
/// # References
///
/// B. Efron, "Better bootstrap confidence intervals", *J. Amer. Statist.
/// Assoc.* 82 (1987), 171–185.
///
/// A. C. Davison and D. V. Hinkley, *Bootstrap Methods and Their
/// Application*, Cambridge University Press, 1997, ch. 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Percentile interval `[q(α/2), q(1 − α/2)]`.
    Percentile,
    /// Basic bootstrap interval `[2θ̂ − q(1 − α/2), 2θ̂ − q(α/2)]`.
    Basic,
    /// Bias-corrected and accelerated (BCa) interval `[q(α₁), q(α₂)]`.
    ///
    /// ```text
    /// αᵢ = Φ(z₀ + (z₀ + zᵢ) / (1 − a (z₀ + zᵢ))),   z₁ = Φ⁻¹(α/2), z₂ = Φ⁻¹(1 − α/2)
    /// z₀ = Φ⁻¹((#{θ* < θ̂} + #{θ* = θ̂}/2) / B)
    /// ```
    ///
    /// `Φ` is the standard normal distribution function. The acceleration
    /// `a = Σ Uᵢ³ / (6 (Σ Uᵢ²)^(3/2))` comes from the jackknife influence
    /// values `Uᵢ` of the design, which needs one extra statistic evaluation
    /// per deletion. Block designs are not supported.
    Bca,
}
/// Exchangeability assumption of a permutation test.
///
/// Each variant gives the number of distinct assignments that an exact test
/// enumerates, for `k` groups of lengths `n₁, …, nₖ` and `N = n₁ + … + nₖ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationKind {
    /// Reassigns the pooled observations to groups of the original sizes, as
    /// in an unpaired two-sample or `k`-sample test. The order within a group
    /// does not matter. There are `N! / (n₁! ⋯ nₖ!)` assignments, and at least
    /// two groups are needed.
    Independent,
    /// Permutes the group labels within each row, as in a paired or
    /// repeated-measures test. The groups must have equal length `n`. There
    /// are `(k!)ⁿ` assignments, and at least two groups are needed.
    PairedLabels,
    /// Permutes the rows of each group independently and keeps every
    /// observation in its group. This breaks the pairing between groups, as in
    /// a test of association. The groups must have equal length `n`. There are
    /// `(n!)ᵏ` assignments.
    Pairings,
}
/// Exact enumeration or random sampling of permutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationMode {
    /// Enumerates every distinct assignment once. Their number must not exceed
    /// [`ResamplingOptions::replicates`].
    Exact,
    /// Draws [`ResamplingOptions::replicates`] assignments uniformly with
    /// replacement. The identity assignment can occur.
    Randomized,
}
/// Alternative hypothesis of a permutation or Monte Carlo test.
///
/// [`ResamplingTest`] defines the tail probabilities.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TestAlternative {
    /// Lower-tail test, `p = p₋`: small statistics count against the null
    /// hypothesis.
    Less,
    /// Upper-tail test, `p = p₊`: large statistics count against the null
    /// hypothesis.
    Greater,
    /// Both tails: `p = min(1, 2 min(p₋, p₊))`.
    #[default]
    TwoSided,
}
/// Resampling method and its settings, fixed when the plan is created.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResamplingMethod {
    /// Bootstrap confidence interval.
    Bootstrap {
        /// Resampling scheme.
        design: BootstrapDesign,
        /// Interval construction.
        method: BootstrapMethod,
        /// Nominal two-sided coverage.
        confidence: ConfidenceProbability,
    },
    /// Delete-one jackknife estimates of bias and standard error.
    Jackknife {
        /// `Independent`, `Stratified` or `Paired`. With `Paired`, each deletion
        /// removes one row from all groups. Block designs are not supported.
        design: BootstrapDesign,
    },
    /// Exact or randomized permutation test.
    Permutation {
        /// Exchangeability assumption.
        kind: PermutationKind,
        /// Exact enumeration or random sampling.
        mode: PermutationMode,
        /// Alternative hypothesis.
        alternative: TestAlternative,
    },
    /// Monte Carlo test against samples from a null sampler. Needs a single
    /// group.
    MonteCarlo {
        /// Alternative hypothesis.
        alternative: TestAlternative,
    },
}
/// Configures replicate counts, memory limits and tie tolerance of a [`ResamplingPlan`].
///
/// Defaults to 9999 replicates, one worker slot, unlimited memory and generator
/// calls, and relative tie tolerance `100 ε`. At p = 0.05 the Monte Carlo
/// standard error is about 0.0022, and `(B + 1) × 0.025 = 250`.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, Bootstrap Methods and Their Application
/// (1997), chapters 4–5.
#[derive(Debug, Clone, Copy)]
pub struct ResamplingOptions<S: RealScalar> {
    /// Number of random replicates, or the maximum number of exact assignments
    /// or jackknife deletions.
    replicates: usize,
    /// Maximum number of replicates held in the workspace at once.
    batch_size: usize,
    /// Memory limit of the workspace in bytes.
    max_workspace_bytes: usize,
    /// Maximum number of generator calls per replicate.
    max_rng_calls: usize,
    /// Tolerance for ties with the observed statistic.
    tie_tolerance: Option<Tolerance<S>>,
}

impl<S: RealScalar> Default for ResamplingOptions<S> {
    fn default() -> Self {
        Self {
            replicates: 9999,
            batch_size: 1,
            max_workspace_bytes: usize::MAX,
            max_rng_calls: usize::MAX,
            tie_tolerance: None,
        }
    }
}
impl<S: RealScalar> ResamplingOptions<S> {
    /// Returns the number of random replicates `B`. For exact permutation
    /// tests and the jackknife, it is the largest number of assignments or
    /// deletions the plan accepts.
    pub const fn replicates(&self) -> usize {
        self.replicates
    }
    /// Sets the number of random replicates; see [`Self::replicates`].
    pub const fn with_replicates(mut self, value: usize) -> Self {
        self.replicates = value;
        self
    }
    /// Returns the largest number of replicates held in the workspace at once.
    /// It bounds memory and parallelism. It does not change the random stream
    /// of any replicate.
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
    /// Sets the batch size; see [`Self::batch_size`].
    pub const fn with_batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }
    /// Returns the memory limit of the workspace in bytes. It does not count
    /// allocator overhead or storage owned by the generator or the callbacks.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }
    /// Sets the memory limit of the workspace in bytes.
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
    /// Returns the largest number of generator calls per replicate.
    ///
    /// Each 32-bit or 64-bit draw counts once, and filling bytes counts one
    /// call per 8 bytes. A replicate that reaches the limit ends the run with
    /// [`StatsTermination::BudgetExhausted`].
    pub const fn max_rng_calls(&self) -> usize {
        self.max_rng_calls
    }
    /// Sets the largest number of generator calls per replicate.
    pub const fn with_max_rng_calls(mut self, value: usize) -> Self {
        self.max_rng_calls = value;
        self
    }
    /// Returns the tie tolerance of permutation and Monte Carlo tests.
    ///
    /// A replicate statistic `T*` ties with the observed `T` when
    /// `|T* − T| ≤ atol + rtol |T|`, and then counts in both tails. `None`
    /// means `atol = 0` and `rtol = 100 ε`.
    pub const fn tie_tolerance(&self) -> Option<Tolerance<S>> {
        self.tie_tolerance
    }
    /// Sets the tie tolerance; see [`Self::tie_tolerance`].
    pub const fn with_tie_tolerance(mut self, value: Option<Tolerance<S>>) -> Self {
        self.tie_tolerance = value;
        self
    }
}
/// Indices of the observations in one resample, grouped like the data.
///
/// Indices refer to the concatenation of all groups: group `g` starts at
/// `n₀ + … + n_(g−1)`, where `nⱼ` is the length of group `j`. The statistic maps
/// them to its own data storage.
#[derive(Debug, Clone, Copy)]
pub struct ResampledIndices<'a> {
    pub(crate) indices: &'a [usize],
    pub(crate) offsets: &'a [usize],
}
impl<'a> ResampledIndices<'a> {
    /// Returns all indices, group after group.
    pub const fn indices(self) -> &'a [usize] {
        self.indices
    }
    /// Returns the number of groups.
    pub fn groups(self) -> usize {
        self.offsets.len() - 1
    }
    /// Returns the indices of group `group`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `group` is not less than [`Self::groups`].
    pub fn group(self, group: usize) -> Result<&'a [usize], SciOxideError> {
        if group >= self.groups() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("resampled_group"),
                group,
                self.groups(),
            ));
        }
        Ok(&self.indices[self.offsets[group]..self.offsets[group + 1]])
    }
}
/// Statistic of a resample given by its indices, for
/// [`ResamplingProblem::Indexed`].
pub type IndexStatistic<'a, S> =
    dyn Fn(ResampledIndices<'_>) -> Result<S, SciOxideError> + Sync + 'a;
/// Statistic of a resample given by its indices, with a scratch slice, for
/// [`ResamplingProblem::IndexedWorkspace`].
///
/// Each worker has its own scratch slice of
/// [`ResamplingPlan::statistic_scratch_elements`] elements. Its contents on
/// entry are unspecified, so write every element before reading it.
pub type WorkspaceIndexStatistic<'a, S> =
    dyn Fn(ResampledIndices<'_>, &mut [S]) -> Result<S, SciOxideError> + Sync + 'a;
/// Test statistic of a sample, for [`ResamplingProblem::MonteCarlo`].
pub type SampleStatistic<'a, S> = dyn Fn(&[S]) -> Result<S, SciOxideError> + Sync + 'a;
/// Test statistic of a sample, with a scratch slice, for
/// [`ResamplingProblem::MonteCarloWorkspace`].
///
/// The scratch slice is as for [`WorkspaceIndexStatistic`]: write every element
/// before reading it.
pub type WorkspaceSampleStatistic<'a, S> =
    dyn Fn(&[S], &mut [S]) -> Result<S, SciOxideError> + Sync + 'a;
/// Draws one sample under the null hypothesis into the slice, using only the
/// given generator.
///
/// The sampler must write every element. An element left unwritten or set to
/// NaN or infinity gives a `NonFinite` error.
pub type NullSampler<'a, S> =
    dyn Fn(&mut dyn Rng<S>, &mut [S]) -> Result<(), SciOxideError> + Sync + 'a;
/// Statistic and data of a resampling analysis.
///
/// Callbacks may run on several threads with indexed execution, so they are
/// `Sync`. Their side effects are not undone when an error occurs.
#[derive(Clone, Copy)]
pub enum ResamplingProblem<'a, S: RealScalar> {
    /// Statistic of the observations selected by index, for the bootstrap, the
    /// jackknife and permutation tests.
    Indexed(&'a IndexStatistic<'a, S>),
    /// As `Indexed`, with the scratch slice set by
    /// [`ResamplingPlan::with_statistic_scratch`].
    IndexedWorkspace(&'a WorkspaceIndexStatistic<'a, S>),
    /// Monte Carlo test: `statistic` is applied to `observed` and to samples
    /// from `sampler`.
    MonteCarlo {
        /// Observed sample. Its length must equal the single group length of
        /// the plan, and its values must be finite.
        observed: &'a [S],
        /// Draws one sample under the null hypothesis.
        sampler: &'a NullSampler<'a, S>,
        /// Test statistic.
        statistic: &'a SampleStatistic<'a, S>,
    },
    /// As `MonteCarlo`, with the scratch slice set by
    /// [`ResamplingPlan::with_statistic_scratch`].
    MonteCarloWorkspace {
        /// Observed sample. Its length must equal the single group length of
        /// the plan, and its values must be finite.
        observed: &'a [S],
        /// Draws one sample under the null hypothesis.
        sampler: &'a NullSampler<'a, S>,
        /// Test statistic.
        statistic: &'a WorkspaceSampleStatistic<'a, S>,
    },
}
/// Prepared configuration of a bootstrap, jackknife, permutation test or Monte
/// Carlo test for given group sizes.
///
/// The plan holds the group offsets, exact assignment counts, jackknife deletion
/// counts and tie tolerance, computed in O(k) time for k groups.
/// Create it with [`ResamplingPlan::new`]. Run it with [`ResamplingPlan::evaluate`],
/// which allocates its output and workspace, or use [`ResamplingPlan::evaluate_into`]
/// with a workspace from [`ResamplingPlan::workspace`]. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
#[derive(Clone, Debug)]
pub struct ResamplingPlan<S: RealScalar> {
    pub(crate) statistic_scratch: usize,
    pub(crate) offsets: Vec<usize>,
    pub(crate) method: ResamplingMethod,
    pub(crate) options: ResamplingOptions<S>,
    pub(crate) replicates: usize,
    pub(crate) deletions: usize,
    pub(crate) tolerance: Tolerance<S>,
}
/// Delete-one jackknife estimates of bias and standard error.
///
/// Let `θ̂` be the observed statistic and `θ̂₍g,i₎` the statistic with
/// observation `i` of group `g` deleted, with mean `θ̄_g` over the `n_g`
/// deletions of group `g`. Then
///
/// ```text
/// b = Σ_g (n_g − 1) (θ̄_g − θ̂)
/// SE² = Σ_g ((n_g − 1) / n_g) Σᵢ (θ̂₍g,i₎ − θ̄_g)²
/// ```
///
/// With the `Paired` design, the rows form a single group.
///
/// # References
///
/// B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman
/// & Hall, 1993, ch. 11.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JackknifeEstimate<S: RealScalar> {
    /// Bias estimate `b`.
    pub bias: S,
    /// Bias-corrected estimate `θ̂ − b`.
    pub bias_corrected: S,
    /// Standard error estimate `SE`.
    pub standard_error: S,
}
/// Bootstrap confidence interval and summaries of the replicate distribution.
///
/// The interval is a statistical estimate, not a guaranteed enclosure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapInterval<S: RealScalar> {
    /// Two-sided confidence interval from [`BootstrapMethod`].
    pub bounds: Interval<S>,
    /// Bootstrap standard error: the sample standard deviation `s` of the `B`
    /// replicates, with divisor `B − 1`.
    pub standard_error: S,
    /// Monte Carlo standard error `s / √B` of the bootstrap mean and of
    /// `bias`. It does not apply to the interval endpoints.
    pub mean_monte_carlo_error: S,
    /// Bootstrap bias estimate `θ̄* − θ̂`, the mean of the replicates minus the
    /// observed statistic.
    pub bias: S,
    /// Whether all replicates are equal.
    pub degenerate: bool,
    /// BCa bias correction `z₀`; `None` for the other methods.
    pub bias_correction: Option<f64>,
    /// BCa acceleration `a`; `None` for the other methods.
    pub acceleration: Option<f64>,
}
/// p-value and tail counts of a permutation or Monte Carlo test.
///
/// Let `L` and `G` be the numbers of replicate statistics at most and at least
/// the observed one, counting ties (see [`ResamplingOptions::tie_tolerance`])
/// in both. For `B` replicates, the tail probabilities are
///
/// ```text
/// p₋ = (L + c) / (B + c),   p₊ = (G + c) / (B + c)
/// ```
///
/// with `c = 0` for an exact test and `c = 1` for a randomized one, so that a
/// randomized p-value is never zero. [`TestAlternative`] combines them.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, *Bootstrap Methods and Their
/// Application*, Cambridge University Press, 1997, ch. 4.
///
/// E. B. Wilson, "Probable inference, the law of succession, and statistical
/// inference", *J. Amer. Statist. Assoc.* 22 (1927), 209–212.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResamplingTest {
    pub(crate) p_value: f64,
    pub(crate) less_equal: usize,
    pub(crate) greater_equal: usize,
    pub(crate) exact: bool,
    pub(crate) resolution: f64,
    pub(crate) tail_uncertainty: Option<[Interval<f64>; 2]>,
}
impl ResamplingTest {
    /// Returns the p-value of the alternative hypothesis of the plan.
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }
    /// Returns `L`, the number of replicate statistics at most the observed
    /// one, ties included.
    pub const fn less_equal(&self) -> usize {
        self.less_equal
    }
    /// Returns `G`, the number of replicate statistics at least the observed
    /// one, ties included.
    pub const fn greater_equal(&self) -> usize {
        self.greater_equal
    }
    /// Returns whether all assignments were enumerated.
    pub const fn exact(&self) -> bool {
        self.exact
    }
    /// Returns the step of a one-tailed p-value: `1/N` for `N` enumerated
    /// assignments, `1/(B + 1)` for `B` random replicates.
    pub const fn resolution(&self) -> f64 {
        self.resolution
    }
    /// Returns 95 % Wilson score intervals for the tail probabilities `L/B`
    /// and `G/B`, which measure the Monte Carlo error of a randomized test.
    /// `None` for an exact test.
    pub const fn tail_uncertainty(&self) -> Option<[Interval<f64>; 2]> {
        self.tail_uncertainty
    }
}
/// Inference of a completed resampling analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResamplingInference<S: RealScalar> {
    /// Bootstrap confidence interval.
    Bootstrap(BootstrapInterval<S>),
    /// Jackknife bias and standard error.
    Jackknife(JackknifeEstimate<S>),
    /// Permutation or Monte Carlo test.
    Test(ResamplingTest),
}
/// Result of a resampling analysis.
///
/// It holds the observed statistic, the replicate counts, the termination and,
/// when every replicate completed, the inference. The replicate statistics
/// are in the output slice of [`ResamplingPlan::evaluate_into`] or in
/// [`ResamplingOutput`]. See
/// [Results and termination](crate#results-and-termination).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResamplingResult<S: RealScalar> {
    pub(crate) method: ResamplingMethod,
    pub(crate) observed: Option<S>,
    pub(crate) requested: usize,
    pub(crate) completed: usize,
    pub(crate) termination: StatsTermination,
    pub(crate) inference: Option<ResamplingInference<S>>,
}
impl<S: RealScalar> ResamplingResult<S> {
    /// Returns the method of the plan, with its confidence or alternative.
    pub const fn method(&self) -> ResamplingMethod {
        self.method
    }
    /// Returns the observed statistic `θ̂`, or `None` if the run stopped before
    /// computing it.
    pub const fn observed(&self) -> Option<S> {
        self.observed
    }
    /// Returns the number of replicates to run, [`ResamplingPlan::replicates`].
    pub const fn requested(&self) -> usize {
        self.requested
    }
    /// Returns the number of replicates that completed.
    pub const fn completed(&self) -> usize {
        self.completed
    }
    /// Returns the termination status: [`StatsTermination::Completed`], or
    /// the reason the run stopped early (cancellation, deadline or an exhausted
    /// budget).
    pub const fn termination(&self) -> StatsTermination {
        self.termination
    }
    /// Returns the inference. It is present exactly when the termination is
    /// [`StatsTermination::Completed`].
    pub const fn inference(&self) -> Option<ResamplingInference<S>> {
        self.inference
    }
}
// Private helpers report.
fn breakdown(message: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Breakdown, message)
}
