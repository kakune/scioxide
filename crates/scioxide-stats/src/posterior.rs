//! Posterior summaries and convergence diagnostics for Markov chain Monte Carlo draws.
use crate::error_helpers::invalid;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZeroUsize;
use scioxide_core::ExecutionPolicy;

use scioxide_core::backend::{PartitionSlot, try_filled, try_with_capacity};
use scioxide_core::{
    ExecutionContext, NonFiniteOrigin, NumericalIssue, RealScalar, Scaled, ScaledAccumulator,
    SciOxideError,
};
use scioxide_linalg::MatrixView;

use crate::arithmetic::center_difference;
use crate::central::Central;
use crate::ranking::{finite_cmp, sort_finite, tie_runs};
use crate::{MomentOrder, MomentsSnapshot, OnlineMoments, QuantileMethod, QuantileProbability};

fn required_bytes(chains: usize, draws: usize) -> Result<usize, SciOxideError> {
    // Conservative bound for ranking (split, rank, distances, order, output),
    // chain centers and the retained autocorrelation prefix. Short-lived buffers
    // are released before the next phase; input conversion fits the same bound.
    chains
        .checked_mul(draws)
        .and_then(|count| {
            count.checked_mul(
                3 * core::mem::size_of::<f64>()
                    + core::mem::size_of::<Scaled<f64>>()
                    + core::mem::size_of::<usize>(),
            )
        })
        .and_then(|bytes| {
            chains
                .checked_mul(2)?
                .checked_mul(core::mem::size_of::<ScaledAccumulator<f64>>())?
                .checked_add(bytes)
        })
        .and_then(|bytes| {
            (draws / 2)
                .checked_mul(core::mem::size_of::<f64>())?
                .checked_add(bytes)
        })
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("posterior_workspace"))
        })
}

/// Limits on the working storage and autocorrelation work of [`posterior_summary_with_options`].
///
/// The default has no storage or work limit and uses every available lag.
#[derive(Debug, Clone, Copy)]
pub struct PosteriorOptions {
    /// Working storage limit in bytes.
    max_workspace_bytes: usize,
    /// Largest autocorrelation lag, or zero for every lag.
    max_lag: usize,
    /// Largest number of lag-product evaluations.
    max_lag_work: usize,
}

impl PosteriorOptions {
    /// Returns the limit in bytes on working storage, not counting the input and the result.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }
    /// Sets the limit returned by [`Self::max_workspace_bytes`].
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
    /// Returns the largest autocorrelation lag used by the effective sample size, or zero for
    /// every available lag.
    ///
    /// A cap truncates Geyer's sequence of autocorrelation pairs at that lag. The effective
    /// sample size is still normalized by the full number of split draws.
    pub const fn max_lag(&self) -> usize {
        self.max_lag
    }
    /// Sets the lag returned by [`Self::max_lag`].
    pub const fn with_max_lag(mut self, value: usize) -> Self {
        self.max_lag = value;
        self
    }
    /// Returns the limit on autocorrelation products summed over all diagnostics.
    ///
    /// [`posterior_summary_with_options`] needs `5 · 2M⌊D/2⌋ · (L + 1)` products for `M` chains of `D`
    /// draws and largest lag `L`.
    pub const fn max_lag_work(&self) -> usize {
        self.max_lag_work
    }
    /// Sets the limit returned by [`Self::max_lag_work`].
    pub const fn with_max_lag_work(mut self, value: usize) -> Self {
        self.max_lag_work = value;
        self
    }
}

impl Default for PosteriorOptions {
    fn default() -> Self {
        Self {
            max_workspace_bytes: usize::MAX,
            max_lag: 0,
            max_lag_work: usize::MAX,
        }
    }
}

/// Posterior summary and convergence diagnostics of one real parameter, from
/// [`posterior_summary_with_options`].
///
/// A diagnostic is `None` when it is undefined because the draws it depends on do not vary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosteriorSummary {
    /// Mean of all draws.
    pub mean: f64,
    /// Sample variance of all `N` draws, with divisor `N − 1`.
    pub variance: f64,
    /// The 5%, 50% and 95% quantiles of all draws (Hyndman–Fan type 7).
    pub quantiles: [f64; 3],
    /// Rank-normalized split R̂: the larger of R̂ for the rank-normalized draws and for the
    /// folded draws. It is not clamped below at 1. `None` if neither has within-chain variance.
    pub rhat: Option<f64>,
    /// Bulk effective sample size: the effective sample size of the rank-normalized split
    /// chains.
    pub bulk_ess: Option<f64>,
    /// Tail effective sample size: the smaller effective sample size of the indicators
    /// `x ≤ q₀.₀₅` and `x ≤ q₀.₉₅`.
    pub tail_ess: Option<f64>,
    /// Monte Carlo standard error of the mean, `√(variance / ESS)` with the effective sample
    /// size `ESS` of the untransformed split chains.
    pub mean_mcse: Option<f64>,
    /// Number of chains.
    pub chains: usize,
    /// Number of draws in each chain.
    pub draws_per_chain: usize,
}

fn quantile(sorted: &[f64], p: f64) -> Result<f64, SciOxideError> {
    crate::empirical::quantile_sorted_by(
        sorted.len(),
        QuantileProbability::new(p)?,
        QuantileMethod::Linear,
        |i| sorted[i],
    )
    .map(|q| q.value())
}

fn moments(values: &[f64]) -> Result<MomentsSnapshot<f64>, SciOxideError> {
    Ok(OnlineMoments::from_slice(values, MomentOrder::Second)?.snapshot())
}

fn sample_variance(snapshot: MomentsSnapshot<f64>) -> Result<Scaled<f64>, SciOxideError> {
    snapshot.variance_scaled(1)?.ok_or_else(|| {
        SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "posterior moment requires at least two observations",
        )
    })
}

fn copied(values: &[f64]) -> Result<Vec<f64>, SciOxideError> {
    let mut result = try_with_capacity(
        values.len(),
        scioxide_core::quantity!("posterior_copied_values"),
    )?;
    result.extend_from_slice(values);
    Ok(result)
}

fn rank_normalize(values: &[f64]) -> Result<Vec<f64>, SciOxideError> {
    rank_normalize_by(values.len(), |a, b| finite_cmp(&values[a], &values[b]))
}

fn rank_normalize_by(
    count: usize,
    compare: impl Fn(usize, usize) -> Ordering,
) -> Result<Vec<f64>, SciOxideError> {
    let mut order = try_with_capacity(count, scioxide_core::quantity!("posterior_rank_order"))?;
    order.extend(0..count);
    order.sort_unstable_by(|&a, &b| compare(a, b));
    let mut ranks = try_filled(count, 0.0, scioxide_core::quantity!("posterior_ranks"))?;
    let normal = scioxide_random::Normal::<f64>::new(0.0, 1.0)?;
    for (run, members) in tie_runs(&order, |&a, &b| compare(a, b) == Ordering::Equal) {
        let p = (run.midrank()? - 0.375) / (count as f64 + 0.25);
        let score = scioxide_random::Quantile::quantile(&normal, p)?;
        for &index in members {
            ranks[index] = score;
        }
    }
    Ok(ranks)
}

fn folded_rank(values: &[f64]) -> Result<Vec<f64>, SciOxideError> {
    let mut sorted = copied(values)?;
    sort_finite(&mut sorted);
    let median = Scaled::<f64>::from_value(quantile(&sorted, 0.5)?)?;
    drop(sorted);
    let mut distances = try_with_capacity(
        values.len(),
        scioxide_core::quantity!("posterior_folded_distances"),
    )?;
    for &value in values {
        distances.push(Scaled::<f64>::from_value(value)?.sub(median)?.abs());
    }
    rank_normalize_by(values.len(), |a, b| distances[a].cmp_value(distances[b]))
}

fn split_draws<S: RealScalar>(chains: MatrixView<'_, S>) -> Result<Vec<f64>, SciOxideError> {
    let half = chains.cols() / 2;
    let mut result = try_filled(
        chains.rows() * 2 * half,
        0.0,
        scioxide_core::quantity!("split_posterior_chains"),
    )?;
    for chain in 0..chains.rows() {
        for draw in 0..half {
            result[(2 * chain) * half + draw] = chains.get(chain, draw)?.into_f64();
            result[(2 * chain + 1) * half + draw] =
                chains.get(chain, chains.cols() - half + draw)?.into_f64();
        }
    }
    Ok(result)
}

struct ChainMoments {
    centers: Vec<ScaledAccumulator<f64>>,
    within: Scaled<f64>,
    marginal: Scaled<f64>,
}

fn chain_moments(
    values: &[f64],
    chains: usize,
    draws: usize,
) -> Result<ChainMoments, SciOxideError> {
    let mut centers =
        try_with_capacity(chains, scioxide_core::quantity!("posterior_chain_centers"))?;
    let mut within = ScaledAccumulator::<f64>::ZERO;
    let mut means = OnlineMoments::new(MomentOrder::Second);
    for chain in values.chunks_exact(draws) {
        let snapshot = moments(chain)?;
        centers.push(snapshot.state.mean()?);
        within.add_scaled(sample_variance(snapshot)?)?;
        // Treat each compensated chain center as one observation of the shared
        // moment merger, without projecting away low terms at a common offset.
        let mut center = Central::singleton(snapshot.state.mean()?.round()?)?.extended()?;
        center.mean = snapshot.state.mean()?;
        let center = Central::from_extended(center)?;
        means.merge(&OnlineMoments::from_parts(1, center, MomentOrder::Second))?;
    }
    let within = within
        .total()?
        .div(Scaled::<f64>::from_value(chains as f64)?)?;
    let marginal = within
        .mul(Scaled::<f64>::from_value((draws - 1) as f64)?)?
        .div(Scaled::<f64>::from_value(draws as f64)?)?
        .add(sample_variance(means.snapshot())?)?;
    Ok(ChainMoments {
        centers,
        within,
        marginal,
    })
}

fn split_rhat(values: &[f64], chains: usize, draws: usize) -> Result<Option<f64>, SciOxideError> {
    let moments = chain_moments(values, chains, draws)?;
    if moments.within.is_zero() {
        return Ok(None);
    }
    moments
        .marginal
        .div(moments.within)?
        .sqrt()?
        .round()
        .map(Some)
}

fn ess(
    values: &[f64],
    chains: usize,
    draws: usize,
    max_lag: usize,
) -> Result<Option<f64>, SciOxideError> {
    let moments = chain_moments(values, chains, draws)?;
    if moments.marginal.is_zero() {
        return Ok(None);
    }
    let lag = |t: usize| -> Result<f64, SciOxideError> {
        let mut covariance = ScaledAccumulator::<f64>::ZERO;
        for (chain, &center) in values.chunks_exact(draws).zip(&moments.centers) {
            for (&left, &right) in chain.iter().zip(&chain[t..]) {
                let left = center_difference(center, ScaledAccumulator::from_value(left)?)?;
                let right = center_difference(center, ScaledAccumulator::from_value(right)?)?;
                covariance.add_product(left, right)?;
            }
        }
        let covariance = covariance
            .total()?
            .div(Scaled::<f64>::from_value(draws as f64)?)?
            .div(Scaled::<f64>::from_value(chains as f64)?)?;
        Scaled::<f64>::ONE
            .sub(moments.within.sub(covariance)?.div(moments.marginal)?)?
            .round()
    };
    let limit = max_lag.min(draws - 1);
    let mut rho = try_filled(
        limit + 1,
        0.0,
        scioxide_core::quantity!("posterior_autocorrelations"),
    )?;
    let mut even = 1.0;
    let mut odd = lag(1)?;
    rho[0] = even;
    rho[1] = odd;
    // Finite-sample endpoints follow ArviZ Stats 1.3.0 / Stan's estimator:
    // retain accepted pairs, then one positive even term at half weight. The
    // same reducer runs on a capped prefix without inventing unavailable lags.
    let mut t = 1;
    while t + 3 < rho.len() && even + odd > 0.0 {
        even = lag(t + 1)?;
        odd = lag(t + 2)?;
        if even + odd >= 0.0 {
            rho[t + 1] = even;
            rho[t + 2] = odd;
        }
        t += 2;
    }
    if even > 0.0 {
        rho[t - 1] = even;
    }
    for i in (1..t.saturating_sub(3)).step_by(2) {
        let previous = rho[i - 1] + rho[i];
        if rho[i + 1] + rho[i + 2] > previous {
            rho[i + 1] = previous / 2.0;
            rho[i + 2] = previous / 2.0;
        }
    }
    let mut sum = ScaledAccumulator::<f64>::from_value(-1.0)?;
    for &value in &rho[..t - 1] {
        sum.add_value(2.0 * value)?;
    }
    sum.add_value(rho[t - 1])?;
    let count = values.len() as f64;
    let tau = sum.round()?.max(1.0 / count.log10());
    let estimate = count / tau;
    if !estimate.is_finite() || estimate <= 0.0 {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "posterior effective sample size must be finite and positive",
        ));
    }
    Ok(Some(estimate))
}

/// Computes the result of [`posterior_summary_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`posterior_summary_with_options`] for the fixed default settings.
pub fn posterior_summary<S: RealScalar>(
    chains: MatrixView<'_, S>,
) -> Result<PosteriorSummary, SciOxideError> {
    posterior_summary_with_options(chains, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("chains"),
        )
    })
}
/// Computes the posterior summary and convergence diagnostics of one real parameter.
///
/// Row `i` of `chains` holds the `D` post-warmup draws of chain `i`. The mean, variance and
/// quantiles use all draws. The diagnostics split each chain into its first and last
/// `⌊D/2⌋` draws. For odd `D` they omit the middle draw. Rank normalization replaces each
/// split draw by `Φ⁻¹((r − 3/8)/(S + 1/4))`, where `r` is its average rank among all `S` split
/// draws and `Φ⁻¹` is the standard normal quantile function. Folding replaces a draw `x` by
/// `|x − median|` before rank normalization.
///
/// R̂ is `√(V̂/W)`, where `W` is the mean within-chain variance and
/// `V̂ = (n − 1)/n · W + B/n` adds the variance `B/n` of the chain means for `n` draws per
/// split chain. The effective sample size is `S/τ̂`. The integrated autocorrelation time `τ̂`
/// sums autocorrelations by Geyer's initial monotone sequence estimator, and is bounded below
/// by `1/log₁₀ S`, so antithetic chains can have an effective sample size above `S`. The cost
/// is `O(S L)` for largest lag `L`, and the storage is `O(M D)` for `M` chains.
///
/// These are diagnostics. A good R̂ and a large effective sample size do not prove that the
/// chains have converged.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two chains or fewer than four draws per chain.
/// - `NonFinite` if a draw is NaN or infinite.
/// - `ResourceLimit` if the working storage exceeds [`PosteriorOptions::max_workspace_bytes`].
/// - `ConvergenceFailure` with `ConvergenceReason::EvaluationLimit` if the required
///   autocorrelation work exceeds [`PosteriorOptions::max_lag_work`]. This is checked before
///   any computation.
/// - `NumericalInstability` if an intermediate result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{PosteriorOptions, posterior_summary_with_options};
/// // Two identical chains: the chain means agree, so R̂ = √((n − 1)/n) with n = 2.
/// let draws = [-1.0_f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
/// let summary = posterior_summary_with_options(
///     MatrixView::from_slice(2, 4, &draws)?,
///     PosteriorOptions::default(),
/// )?;
/// assert_eq!(summary.mean, 0.0);
/// assert!((summary.rhat.unwrap_or_default() - 0.5_f64.sqrt()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Vehtari, A. Gelman, D. Simpson, B. Carpenter and P.-C. Bürkner, "Rank-normalization,
///   folding, and localization: An improved R̂ for assessing convergence of MCMC", *Bayesian
///   Analysis* 16 (2021), 667–718.
/// - C. J. Geyer, "Practical Markov chain Monte Carlo", *Statistical Science* 7 (1992),
///   473–483.
pub fn posterior_summary_with_options<S: RealScalar>(
    chains: MatrixView<'_, S>,
    options: PosteriorOptions,
) -> Result<PosteriorSummary, SciOxideError> {
    let m = chains.rows();
    let draws = chains.cols();
    if m < 2 || draws < 4 {
        return Err(invalid(
            scioxide_core::argument!("chains"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "at least two chains and four draws per chain required",
        ));
    }
    let count = m
        .checked_mul(draws)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("posterior_draws")))?;
    let bytes = required_bytes(m, draws)?;
    if bytes > options.max_workspace_bytes {
        return Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("posterior_workspace"),
            bytes,
            options.max_workspace_bytes,
        ));
    }
    let half = draws / 2;
    let split_count = m
        .checked_mul(2)
        .and_then(|v| v.checked_mul(half))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("posterior_split")))?;
    let lag = if options.max_lag == 0 {
        half - 1
    } else {
        options.max_lag.min(half - 1)
    };
    let work = split_count
        .checked_mul(lag + 1)
        .and_then(|v| v.checked_mul(5))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("posterior_lag_work"))
        })?;
    crate::error_helpers::work(
        work,
        options.max_lag_work,
        scioxide_core::BudgetKind::Evaluation,
        "posterior lag-pair evaluations exceed options.max_lag_work",
    )?;
    let mut values = try_filled(count, 0.0, scioxide_core::quantity!("posterior_values"))?;
    for i in 0..m {
        for j in 0..draws {
            let value = chains.get(i, j)?;
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "posterior draw must be finite",
                ));
            }
            values[i * draws + j] = value.into_f64();
        }
    }
    let snapshot = moments(&values)?;
    let mean = snapshot.state.mean()?.round()?;
    let variance_scaled = sample_variance(snapshot)?;
    let variance = variance_scaled.round()?;
    sort_finite(&mut values);
    let quantiles = [
        quantile(&values, 0.05)?,
        quantile(&values, 0.5)?,
        quantile(&values, 0.95)?,
    ];
    drop(values);
    let split = split_draws(chains)?;
    let rank = rank_normalize(&split)?;
    let folded_rank = folded_rank(&split)?;
    let split_chains = m * 2;
    let rhat = match (
        split_rhat(&rank, split_chains, half)?,
        split_rhat(&folded_rank, split_chains, half)?,
    ) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    drop(folded_rank);
    let bulk_ess = ess(&rank, split_chains, half, lag)?;
    drop(rank);
    let mut indicator = copied(&split)?;
    indicator
        .iter_mut()
        .for_each(|x| *x = f64::from(*x <= quantiles[0]));
    let low = ess(&indicator, split_chains, half, lag)?;
    for (output, &value) in indicator.iter_mut().zip(&split) {
        *output = f64::from(value <= quantiles[2]);
    }
    let tail_ess = match (low, ess(&indicator, split_chains, half, lag)?) {
        (Some(a), Some(b)) => Some(a.min(b)),
        _ => None,
    };
    drop(indicator);
    let mean_mcse = ess(&split, split_chains, half, lag)?
        .map(|effective| {
            variance_scaled
                .div(Scaled::<f64>::from_value(effective)?)?
                .sqrt()?
                .round()
        })
        .transpose()?;
    Ok(PosteriorSummary {
        mean,
        variance,
        quantiles,
        rhat,
        bulk_ess,
        tail_ess,
        mean_mcse,
        chains: m,
        draws_per_chain: draws,
    })
}

/// Computes [`posterior_summary_with_options`] for each parameter and writes it to the same position of
/// `output`.
///
/// Parameters run in parallel when the execution context allows it. The result does not
/// depend on the number of workers. On success every entry of `output` is `Some`.
///
/// # Errors
///
/// - `InvalidInput` if `output.len()` differs from `parameters.len()`.
/// - `ResourceLimit` if the working storage of all concurrent parameters exceeds
///   [`PosteriorOptions::max_workspace_bytes`].
/// - `ExecutionStopped` on cancellation.
/// - The first error of [`posterior_summary_with_options`] for any parameter.
pub fn posterior_summaries_into<S: RealScalar>(
    parameters: &[MatrixView<'_, S>],
    options: PosteriorOptions,
    output: &mut [Option<PosteriorSummary>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if parameters.len() != output.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "posterior batch output length",
        ));
    }
    let execution = context.plan(
        ExecutionPolicy::Auto,
        parameters.len(),
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: context.worker_budget(),
        },
    );
    let staging = parameters
        .len()
        .checked_mul(core::mem::size_of::<Option<PosteriorSummary>>())
        .and_then(|v| {
            execution
                .worker_count()
                .checked_mul(core::mem::size_of::<
                    PartitionSlot<&mut [Option<PosteriorSummary>]>,
                >())?
                .checked_add(v)
        })
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("posterior_batch_staging"))
        })?;
    let largest = parameters
        .iter()
        .map(|view| required_bytes(view.rows(), view.cols()))
        .try_fold(0_usize, |current, bytes| {
            Ok::<_, SciOxideError>(current.max(bytes?))
        })?;
    let required = staging
        .checked_add(
            execution
                .worker_count()
                .checked_mul(largest)
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!(
                        "posterior_concurrent_workspace"
                    ))
                })?,
        )
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("posterior_batch_workspace"))
        })?;
    if required > options.max_workspace_bytes {
        return Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("posterior_batch_workspace"),
            required,
            options.max_workspace_bytes,
        ));
    }
    let mut staged = try_filled(
        parameters.len(),
        None,
        scioxide_core::quantity!("posterior_batch_output"),
    )?;
    let mut slots = try_with_capacity(
        execution.worker_count(),
        scioxide_core::quantity!("posterior_batch_slots"),
    )?;
    let mut remaining = staged.as_mut_slice();
    for part in execution.partitions() {
        let (results, tail) = remaining.split_at_mut(part.range().len());
        remaining = tail;
        slots.push(PartitionSlot::new(results));
    }
    context.execute_partitioned(execution, &slots, &|partition, results, _| {
        for (index, result) in partition.range().zip(results) {
            context.check()?;
            *result = Some(posterior_summary_with_options(parameters[index], options)?);
        }
        Ok(())
    })?;
    drop(slots);
    output.copy_from_slice(&staged);
    Ok(())
}
