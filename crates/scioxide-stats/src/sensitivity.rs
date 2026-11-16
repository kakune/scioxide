//! Sobol sensitivity indices by the pick-freeze method.
use crate::StatisticsOptions;
use crate::error_helpers::invalid;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use scioxide_core::backend::{PartitionSlot, try_filled, try_with_capacity};
use scioxide_core::{
    ExecutionContext, NonFiniteOrigin, RealScalar, Scaled, ScaledAccumulator, SciOxideError,
};
use scioxide_random::{BitGenerator, QmcPlan, QmcReplicatePlan, QuasiRng, Rng, SplittableRng};

/// Model outputs on the two sample matrices `A` and `B` of a pick-freeze design and on the
/// matrices `A_B⁽ⁱ⁾`.
///
/// `A` and `B` are independent `n × d` samples of the `d` input coordinates. `A_B⁽ⁱ⁾` is `A`
/// with column `i` taken from `B`. Row `k` of each matrix gives one model output: `a[k] =
/// f(A_k)`, `b[k] = f(B_k)` and `hybrids[k·d + i] = f(A_B⁽ⁱ⁾_k)`. The input coordinates must be
/// independent.
#[derive(Debug, Clone, Copy)]
pub struct SobolEvaluations<'a, S: RealScalar> {
    /// Outputs on the rows of `A`.
    pub a: &'a [S],
    /// Outputs on the rows of `B`.
    pub b: &'a [S],
    /// Outputs on the rows of every `A_B⁽ⁱ⁾`, stored row by row with length `n·d`.
    pub hybrids: &'a [S],
    /// Number `d` of input coordinates.
    pub dimensions: usize,
}

/// Estimated first-order and total-order Sobol indices, one entry per input coordinate.
///
/// The estimates are not clamped, so sampling error can put them outside `[0, 1]`. Standard
/// errors are present only for [`sobol_indices_qmc_scoped`].
#[derive(Debug, Clone, PartialEq)]
pub struct SobolIndices {
    first_order: Vec<f64>,
    total_order: Vec<f64>,
    rows: usize,
    randomized_standard_error: Option<(Vec<f64>, Vec<f64>)>,
}
impl SobolIndices {
    /// Returns the first-order indices `Sᵢ`.
    pub fn first_order(&self) -> &[f64] {
        &self.first_order
    }
    /// Returns the total-order indices `STᵢ`.
    pub fn total_order(&self) -> &[f64] {
        &self.total_order
    }
    /// Returns the number of rows `n` of the design, summed over replicates for
    /// [`sobol_indices_qmc_scoped`].
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the standard errors of the first-order and total-order indices, estimated from
    /// the spread across independently randomized QMC replicates.
    ///
    /// Returns `None` unless the indices come from [`sobol_indices_qmc_scoped`].
    pub fn randomized_standard_error(&self) -> Option<(&[f64], &[f64])> {
        self.randomized_standard_error
            .as_ref()
            .map(|(first, total)| (first.as_slice(), total.as_slice()))
    }
}

fn validate<S: RealScalar>(e: SobolEvaluations<'_, S>) -> Result<(), SciOxideError> {
    if e.dimensions == 0
        || e.a.len() < 4
        || e.b.len() != e.a.len()
        || e.a.len().checked_mul(e.dimensions) != Some(e.hybrids.len())
    {
        return Err(invalid(
            scioxide_core::argument!("e"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "A/B/hybrid shape or sample count",
        ));
    }
    if e.a
        .iter()
        .chain(e.b)
        .chain(e.hybrids)
        .any(|x| !x.is_finite())
    {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "model output must be finite",
        ));
    }
    Ok(())
}

fn calculate<S: RealScalar>(
    e: SobolEvaluations<'_, S>,
    rows: impl Iterator<Item = usize> + Clone,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), SciOxideError> {
    let mut moments = crate::OnlineMoments::<S>::new(crate::MomentOrder::Second);
    for row in rows.clone() {
        moments.observe(e.a[row])?;
        moments.observe(e.b[row])?;
    }
    let state = moments.snapshot().state;
    let variance = state.m2()?;
    if variance.is_zero() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "zero output variance",
        ));
    }
    let mut first = try_filled(
        e.dimensions,
        0.0,
        scioxide_core::quantity!("sobol_first_order"),
    )?;
    let mut total = try_filled(
        e.dimensions,
        0.0,
        scioxide_core::quantity!("sobol_total_order"),
    )?;
    let count = Scaled::from_value(n as f64)?;
    for (dimension, (s, t)) in first.iter_mut().zip(&mut total).enumerate() {
        let mut first_sum = ScaledAccumulator::ZERO;
        let mut total_sum = ScaledAccumulator::ZERO;
        for row in rows.clone() {
            let a = Scaled::from_value(e.a[row].into_f64())?;
            let b = crate::arithmetic::center_difference(
                state.mean()?,
                ScaledAccumulator::from_value(e.b[row].into_f64())?,
            )?;
            let ab = Scaled::from_value(e.hybrids[row * e.dimensions + dimension].into_f64())?;
            let delta = ab.sub(a)?;
            first_sum.add_product(b, delta)?;
            total_sum.add_scaled(delta.square()?)?;
        }
        *s = first_sum.total()?.div(count)?.div(variance)?.round()?;
        *t = total_sum
            .total()?
            .mul(Scaled::HALF)?
            .div(count)?
            .div(variance)?
            .round()?;
    }
    Ok((first, total))
}

/// Computes first-order and total-order Sobol indices from the outputs of a pick-freeze design.
///
/// With `m` and `V` the mean and variance of the `2n` pooled outputs `f(A_k)` and `f(B_k)`,
/// the first-order index uses the Saltelli (2010) estimator and the total-order index uses
/// the Jansen (1999) estimator:
///
/// ```text
/// Sᵢ  = (1/(n V)) Σₖ (f(B_k) − m) (f(A_B⁽ⁱ⁾_k) − f(A_k))
/// STᵢ = (1/(2n V)) Σₖ (f(A_k) − f(A_B⁽ⁱ⁾_k))²
/// ```
///
/// The cost is `O(n d)`. Sums are accumulated with an extended exponent range, so adding a
/// constant to the outputs or scaling them by a nonzero factor changes the estimates only by
/// rounding. The estimators assume independent input coordinates.
///
/// # Errors
///
/// - `InvalidInput` if `dimensions` is zero, `n < 4`, `b.len() ≠ n`, or
///   `hybrids.len() ≠ n·d`.
/// - `NonFinite` if an output is NaN or infinite.
/// - `NumericalInstability` if the pooled output variance is zero.
///
/// # References
///
/// - I. M. Sobol', "Global sensitivity indices for nonlinear mathematical models and their
///   Monte Carlo estimates", *Math. Comput. Simulation* 55 (2001), 271–280.
/// - M. J. W. Jansen, "Analysis of variance designs for model output", *Comput. Phys.
///   Commun.* 117 (1999), 35–43.
/// - A. Saltelli et al., "Variance based sensitivity analysis of model output. Design and
///   estimator for the total sensitivity index", *Comput. Phys. Commun.* 181 (2010), 259–270.
pub fn sobol_indices<S: RealScalar>(
    e: SobolEvaluations<'_, S>,
) -> Result<SobolIndices, SciOxideError> {
    validate(e)?;
    let (first_order, total_order) = calculate(e, 0..e.a.len(), e.a.len())?;
    Ok(SobolIndices {
        first_order,
        total_order,
        rows: e.a.len(),
        randomized_standard_error: None,
    })
}

/// Bootstrap percentile confidence intervals for Sobol indices, from [`sobol_bootstrap_scoped`].
#[derive(Debug, Clone, PartialEq)]
pub struct SobolBootstrap {
    first_order: Vec<(f64, f64)>,
    total_order: Vec<(f64, f64)>,
    confidence: crate::ConfidenceProbability,
    replicates: usize,
}
impl SobolBootstrap {
    /// Returns the `(lower, upper)` interval of each first-order index.
    pub fn first_order(&self) -> &[(f64, f64)] {
        &self.first_order
    }
    /// Returns the `(lower, upper)` interval of each total-order index.
    pub fn total_order(&self) -> &[(f64, f64)] {
        &self.total_order
    }
    /// Returns the confidence level of the intervals.
    pub const fn confidence(&self) -> crate::ConfidenceProbability {
        self.confidence
    }
    /// Returns the number of bootstrap resamples.
    pub const fn replicates(&self) -> usize {
        self.replicates
    }
}

fn sample_index<R: BitGenerator>(rng: &mut R, n: usize) -> Result<usize, SciOxideError> {
    let bound = u64::try_from(n).map_err(|_| {
        invalid(
            scioxide_core::argument!("e"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "sample count exceeds u64",
        )
    })?;
    let threshold = bound.wrapping_neg() % bound;
    for _ in 0..128 {
        let word = rng.next_uniform_u64()?;
        if word >= threshold {
            return Ok((word % bound) as usize);
        }
    }
    Err(invalid(
        scioxide_core::argument!("root"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "bootstrap index rejection budget exhausted",
    ))
}

/// Sobol indices from [`sobol_indices_monte_carlo_scoped`] together with the model outputs they were
/// computed from.
///
/// Pass [`Self::evaluations`] to [`sobol_bootstrap_scoped`] for confidence intervals.
#[derive(Debug, Clone)]
pub struct SobolMonteCarloResult {
    indices: SobolIndices,
    a: Vec<f64>,
    b: Vec<f64>,
    hybrids: Vec<f64>,
}

impl SobolMonteCarloResult {
    /// Returns the first-order and total-order indices.
    pub const fn indices(&self) -> &SobolIndices {
        &self.indices
    }
    /// Returns the model outputs on `A`, `B` and every `A_B⁽ⁱ⁾`.
    pub fn evaluations(&self) -> SobolEvaluations<'_, f64> {
        SobolEvaluations {
            a: &self.a,
            b: &self.b,
            hybrids: &self.hybrids,
            dimensions: self.indices.first_order.len(),
        }
    }
}

struct MonteCarloSlot<'a, S: RealScalar> {
    a_point: Vec<S>,
    b_point: Vec<S>,
    hybrid: Vec<S>,
    a_values: &'a mut [f64],
    b_values: &'a mut [f64],
    hybrid_values: &'a mut [f64],
}

/// Executes [`sobol_indices_monte_carlo_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn sobol_indices_monte_carlo<S, R>(
    dimensions: usize,
    rows: usize,
    root: &R,
    sampler: impl Fn(usize, &mut R) -> Result<S, SciOxideError> + Sync,
    model: impl Fn(&[S]) -> Result<f64, SciOxideError> + Sync,
) -> Result<SobolMonteCarloResult, SciOxideError>
where
    S: RealScalar,
    R: Rng<S> + SplittableRng + Sync,
{
    sobol_indices_monte_carlo_scoped(
        dimensions,
        rows,
        root,
        sampler,
        model,
        Default::default(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Estimates Sobol indices by Monte Carlo sampling of a pick-freeze design.
///
/// Row `k` draws `A_k` and `B_k` from the generator `root.split(k)`. For each coordinate `j`,
/// `sampler(j, rng)` returns one independent draw from the marginal distribution of coordinate
/// `j`. `model` is evaluated at `A_k`, `B_k` and every `A_B⁽ⁱ⁾_k`, so it is called
/// `n(d + 2)` times. The indices are those of [`sobol_indices`]. `root` itself does not
/// advance, and the result does not depend on the execution policy in `options`. The stored
/// outputs take `O(n d)` memory, which counts toward [`StatisticsOptions::max_bytes`].
///
/// # Errors
///
/// - `InvalidInput` if `dimensions` is zero, `rows < 4`, or `rows` does not fit in `u64`.
/// - `NonFinite` if `sampler` or `model` returns a NaN or infinity.
/// - `NumericalInstability` if the output variance is zero.
/// - `SizeOverflow` or `ResourceLimit` if the storage overflows or exceeds the byte limit.
/// - `ExecutionStopped` on cancellation.
/// - Errors returned by `sampler`, `model` or `root.split`.
pub fn sobol_indices_monte_carlo_scoped<'scope, 'capability, S, R>(
    dimensions: usize,
    rows: usize,
    root: &R,
    sampler: impl Fn(usize, &mut R) -> Result<S, SciOxideError> + Sync,
    model: impl Fn(&[S]) -> Result<f64, SciOxideError> + Sync,
    options: StatisticsOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<SobolMonteCarloResult, SciOxideError>
where
    S: RealScalar,
    R: Rng<S> + SplittableRng + Sync,
{
    let mut state = scioxide_core::SolveState::new();
    scope.into().run(
        &mut state,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "sobol_indices_monte_carlo", 1),
            rows as u128,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |solve| {
            let context = solve.execution();

            if dimensions == 0 || rows < 4 || u64::try_from(rows).is_err() {
                return Err(invalid(
                    scioxide_core::argument!("rows"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Monte Carlo dimensions or rows",
                ));
            }
            let output_count = rows
                .checked_mul(dimensions.checked_add(2).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_design"))
                })?)
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_design"))
                })?;
            let execution = context.plan(
                options.execution_policy(),
                rows,
                scioxide_core::PlanLimits {
                    work_per_item: core::num::NonZeroUsize::MIN,
                    auto_threshold: NonZeroUsize::MIN,
                    worker_limit: context.worker_budget(),
                },
            );
            let bytes = output_count
                .checked_mul(core::mem::size_of::<f64>())
                .and_then(|v| {
                    execution
                        .worker_count()
                        .checked_mul(dimensions)?
                        .checked_mul(3)?
                        .checked_mul(core::mem::size_of::<S>())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    execution
                        .worker_count()
                        .checked_mul(core::mem::size_of::<PartitionSlot<MonteCarloSlot<'_, S>>>())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    dimensions
                        .checked_mul(2)?
                        .checked_mul(core::mem::size_of::<f64>())?
                        .checked_add(v)
                })
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!(
                        "sobol_monte_carlo_workspace"
                    ))
                })?;
            crate::error_helpers::limit(
                bytes,
                options.max_bytes(),
                scioxide_core::quantity!("sensitivity_bytes"),
            )?;
            let mut a = try_filled(rows, 0.0, scioxide_core::quantity!("sobol_a_values"))?;
            let mut b = try_filled(rows, 0.0, scioxide_core::quantity!("sobol_b_values"))?;
            let mut hybrids = try_filled(
                rows.checked_mul(dimensions).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_hybrids"))
                })?,
                0.0,
                scioxide_core::quantity!("sobol_hybrids"),
            )?;
            let mut slots = try_with_capacity(
                execution.worker_count(),
                scioxide_core::quantity!("sobol_monte_carlo_slots"),
            )?;
            let mut remaining_a = a.as_mut_slice();
            let mut remaining_b = b.as_mut_slice();
            let mut remaining_hybrids = hybrids.as_mut_slice();
            for partition in execution.partitions() {
                let len = partition.range().len();
                let (a_values, a_rest) = remaining_a.split_at_mut(len);
                let (b_values, b_rest) = remaining_b.split_at_mut(len);
                let (hybrid_values, hybrid_rest) = remaining_hybrids.split_at_mut(len * dimensions);
                remaining_a = a_rest;
                remaining_b = b_rest;
                remaining_hybrids = hybrid_rest;
                slots.push(PartitionSlot::new(MonteCarloSlot {
                    a_point: try_filled(
                        dimensions,
                        S::ZERO,
                        scioxide_core::quantity!("sobol_a_point"),
                    )?,
                    b_point: try_filled(
                        dimensions,
                        S::ZERO,
                        scioxide_core::quantity!("sobol_b_point"),
                    )?,
                    hybrid: try_filled(
                        dimensions,
                        S::ZERO,
                        scioxide_core::quantity!("sobol_hybrid_point"),
                    )?,
                    a_values,
                    b_values,
                    hybrid_values,
                }));
            }
            let evaluations = Some(rows)
                .and_then(|n| dimensions.checked_add(2).and_then(|d| n.checked_mul(d)))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_evaluations"))
                })?;
            solve
                .record_providers(scioxide_core::EvaluationRole::Function, evaluations)?
                .require()?;
            context.execute_partitioned(
                execution,
                &slots[..execution.worker_count()],
                &|partition, mut slot, _| {
                    for (offset, (a_value, b_value)) in slot
                        .a_values
                        .iter_mut()
                        .zip(slot.b_values.iter_mut())
                        .enumerate()
                    {
                        context.check()?;
                        let row = partition.range().start + offset;
                        let mut rng = root.split(row as u64)?;
                        for j in 0..dimensions {
                            slot.a_point[j] = sampler(j, &mut rng)?;
                            slot.b_point[j] = sampler(j, &mut rng)?;
                            if !slot.a_point[j].is_finite() || !slot.b_point[j].is_finite() {
                                return Err(SciOxideError::non_finite(
                                    NonFiniteOrigin::Callback,
                                    "sampler returned nonfinite input",
                                ));
                            }
                        }
                        *a_value = model(&slot.a_point)?;
                        *b_value = model(&slot.b_point)?;
                        slot.hybrid.copy_from_slice(&slot.a_point);
                        for j in 0..dimensions {
                            slot.hybrid[j] = slot.b_point[j];
                            slot.hybrid_values[offset * dimensions + j] = model(&slot.hybrid)?;
                            slot.hybrid[j] = slot.a_point[j];
                        }
                    }
                    Ok(())
                },
            )?;

            drop(slots);
            let indices = sobol_indices(SobolEvaluations {
                a: &a,
                b: &b,
                hybrids: &hybrids,
                dimensions,
            })?;
            Ok(SobolMonteCarloResult {
                indices,
                a,
                b,
                hybrids,
            })
        },
    )
}

/// Executes [`sobol_bootstrap_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn sobol_bootstrap<S: RealScalar, R: BitGenerator + SplittableRng>(
    e: SobolEvaluations<'_, S>,
    root: &R,
    replicates: usize,
    confidence: crate::ConfidenceProbability,
) -> Result<SobolBootstrap, SciOxideError> {
    sobol_bootstrap_scoped(
        e,
        root,
        replicates,
        confidence,
        Default::default(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes bootstrap percentile confidence intervals for Sobol indices.
///
/// Each resample draws `n` row indices with replacement and keeps the outputs of a row on `A`,
/// `B` and every `A_B⁽ⁱ⁾` together. It then recomputes the indices with the estimators of
/// [`sobol_indices`]. The interval for confidence level `c` runs from the `(1 − c)/2` to the
/// `(1 + c)/2` quantile of the resampled indices (Hyndman–Fan type 7). The intervals are valid
/// only for independent rows, so do not use them on the points of one QMC sequence. Resample
/// `r` uses a substream of `root` chosen by `r` alone. Resamples run sequentially at a cost of
/// `O(replicates · n d)`.
///
/// # Errors
///
/// - The conditions of [`sobol_indices`] for `e`.
/// - `InvalidInput` if `replicates < 2`.
/// - `NumericalInstability` if a resample has zero output variance.
/// - `SizeOverflow` or `ResourceLimit` if the storage overflows or exceeds the byte limit of
///   `options`.
/// - `ExecutionStopped` on cancellation.
/// - Errors returned by the generator.
///
/// # References
///
/// B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993,
/// ch. 13.
pub fn sobol_bootstrap_scoped<
    'scope,
    'capability,
    S: RealScalar,
    R: BitGenerator + SplittableRng,
>(
    e: SobolEvaluations<'_, S>,
    root: &R,
    replicates: usize,
    confidence: crate::ConfidenceProbability,
    options: StatisticsOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<SobolBootstrap, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    scope.into().run(
        &mut state,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "sobol_bootstrap", 1),
            replicates as u128,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            let context = context.execution();

            validate(e)?;
            if replicates < 2 {
                return Err(invalid(
                    scioxide_core::argument!("replicates"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "bootstrap requires at least two replicates",
                ));
            }
            let bytes =
                e.a.len()
                    .checked_mul(core::mem::size_of::<usize>())
                    .and_then(|v| {
                        replicates
                            .checked_mul(e.dimensions)?
                            .checked_mul(2)?
                            .checked_mul(core::mem::size_of::<f64>())?
                            .checked_add(v)
                    })
                    .and_then(|v| {
                        e.dimensions
                            .checked_mul(2)?
                            .checked_mul(core::mem::size_of::<f64>())?
                            .checked_add(v)
                    })
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!(
                            "sobol_bootstrap_workspace"
                        ))
                    })?;
            crate::error_helpers::limit(
                bytes,
                options.max_bytes(),
                scioxide_core::quantity!("sensitivity_bytes"),
            )?;
            let mut rows = try_filled(
                e.a.len(),
                0_usize,
                scioxide_core::quantity!("sobol_bootstrap_rows"),
            )?;
            let buffer_len = e.dimensions.checked_mul(replicates).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("sobol_bootstrap_output"))
            })?;
            let mut first = try_filled(
                buffer_len,
                0.0,
                scioxide_core::quantity!("sobol_bootstrap_first"),
            )?;
            let mut total = try_filled(
                buffer_len,
                0.0,
                scioxide_core::quantity!("sobol_bootstrap_total"),
            )?;
            for replicate in 0..replicates {
                context.check()?;
                let mut rng = root.substream(
                    0x0053_4f42_4f4c,
                    u64::try_from(replicate).map_err(|_| {
                        invalid(
                            scioxide_core::argument!("replicates"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "bootstrap replicate index exceeds u64",
                        )
                    })?,
                )?;
                for row in &mut rows {
                    *row = sample_index(&mut rng, e.a.len())?;
                }
                let (s, t) = calculate(e, rows.iter().copied(), rows.len())?;
                for dimension in 0..e.dimensions {
                    first[dimension * replicates + replicate] = s[dimension];
                    total[dimension * replicates + replicate] = t[dimension];
                }
            }
            let tail = (1.0 - confidence.get()) / 2.0;
            let mut first_order = scioxide_core::backend::try_with_capacity(
                e.dimensions,
                scioxide_core::quantity!("sobol_first_bounds"),
            )?;
            let mut total_order = scioxide_core::backend::try_with_capacity(
                e.dimensions,
                scioxide_core::quantity!("sobol_total_bounds"),
            )?;
            for dimension in 0..e.dimensions {
                for (values, bounds) in [
                    (&mut first, &mut first_order),
                    (&mut total, &mut total_order),
                ] {
                    let lane = &mut values[dimension * replicates..(dimension + 1) * replicates];
                    crate::ranking::sort_finite(lane);
                    bounds.push((percentile(lane, tail)?, percentile(lane, 1.0 - tail)?));
                }
            }
            Ok(SobolBootstrap {
                first_order,
                total_order,
                confidence,
                replicates,
            })
        },
    )
}

fn percentile(sorted: &[f64], p: f64) -> Result<f64, SciOxideError> {
    crate::empirical::quantile_sorted_by(
        sorted.len(),
        crate::QuantileProbability::new(p)?,
        crate::QuantileMethod::Linear,
        |i| sorted[i],
    )
    .map(|q| q.value())
}

fn qmc_replica<S: RealScalar>(
    replica: &QmcPlan,
    dimensions: usize,
    points: usize,
    transform: &impl Fn(usize, S) -> Result<S, SciOxideError>,
    model: &impl Fn(&[S]) -> Result<f64, SciOxideError>,
    context: ExecutionContext<'_>,
) -> Result<SobolIndices, SciOxideError> {
    context.check()?;
    let mut sequence = replica.sequence::<S>()?;
    let mut unit = try_filled(
        2 * dimensions,
        S::ZERO,
        scioxide_core::quantity!("sobol_qmc_point"),
    )?;
    let mut a_point = try_filled(dimensions, S::ZERO, scioxide_core::quantity!("sobol_a"))?;
    let mut b_point = try_filled(dimensions, S::ZERO, scioxide_core::quantity!("sobol_b"))?;
    let mut hybrid = try_filled(
        dimensions,
        S::ZERO,
        scioxide_core::quantity!("sobol_hybrid"),
    )?;
    let mut a = try_filled(points, 0.0, scioxide_core::quantity!("sobol_a_values"))?;
    let mut b = try_filled(points, 0.0, scioxide_core::quantity!("sobol_b_values"))?;
    let mut ab = try_filled(
        points.checked_mul(dimensions).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("sobol_hybrid_values"))
        })?,
        0.0,
        scioxide_core::quantity!("sobol_hybrid_values"),
    )?;
    for row in 0..points {
        context.check()?;
        sequence.next_point_into(&mut unit)?;
        for j in 0..dimensions {
            a_point[j] = transform(j, unit[j])?;
            b_point[j] = transform(j, unit[dimensions + j])?;
            if !a_point[j].is_finite() || !b_point[j].is_finite() {
                return Err(invalid(
                    scioxide_core::argument!("transform"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "nonfinite marginal transform",
                ));
            }
        }
        a[row] = model(&a_point)?;
        b[row] = model(&b_point)?;
        hybrid.copy_from_slice(&a_point);
        for j in 0..dimensions {
            hybrid[j] = b_point[j];
            ab[row * dimensions + j] = model(&hybrid)?;
            hybrid[j] = a_point[j];
        }
    }
    sobol_indices(SobolEvaluations {
        a: &a,
        b: &b,
        hybrids: &ab,
        dimensions,
    })
}

/// Executes [`sobol_indices_qmc_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn sobol_indices_qmc<S: RealScalar>(
    plan: &QmcReplicatePlan,
    dimensions: usize,
    points: usize,
    transform: impl Fn(usize, S) -> Result<S, SciOxideError> + Sync,
    model: impl Fn(&[S]) -> Result<f64, SciOxideError> + Sync,
) -> Result<SobolIndices, SciOxideError> {
    sobol_indices_qmc_scoped(
        plan,
        dimensions,
        points,
        transform,
        model,
        Default::default(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Estimates Sobol indices by randomized quasi-Monte Carlo sampling of a pick-freeze design.
///
/// Each randomized replicate in `plan` generates `points` points in `[0, 1)^(2d)`. The first
/// `d` coordinates of a point give a row of `A` and the last `d` give the matching row of `B`.
/// `transform(j, u)` maps a unit coordinate `u` to coordinate `j` of the input, for example by
/// the inverse CDF of its marginal. The indices of each replicate are those of
/// [`sobol_indices`]. The result is their mean over the replicates, and the standard errors
/// are their standard deviation divided by `√R`, where `R` is the number of replicates.
/// `model` is called `R · points · (d + 2)` times. Replicates run under the execution policy
/// of `options`.
///
/// # Errors
///
/// - `InvalidInput` if `dimensions` is zero, `points < 4`, `points` differs from the points
///   per replicate of `plan`, `plan` has fewer than two replicates, or a replicate does not
///   have `2d` dimensions.
/// - `InvalidInput` if `transform` returns a NaN or infinity.
/// - `NonFinite` if `model` returns a NaN or infinity.
/// - `NumericalInstability` if a replicate has zero output variance or the mean or standard
///   error overflows.
/// - `SizeOverflow` or `ResourceLimit` if the storage overflows or exceeds the byte limit.
/// - `ExecutionStopped` on cancellation.
/// - Errors returned by `transform`, `model` or the QMC sequence.
pub fn sobol_indices_qmc_scoped<'scope, 'capability, S: RealScalar>(
    plan: &QmcReplicatePlan,
    dimensions: usize,
    points: usize,
    transform: impl Fn(usize, S) -> Result<S, SciOxideError> + Sync,
    model: impl Fn(&[S]) -> Result<f64, SciOxideError> + Sync,
    options: StatisticsOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<SobolIndices, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    scope.into().run(
        &mut state,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "sobol_indices_qmc", 1),
            plan.plans().len() as u128,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |solve| {
            let context = solve.execution();

            if dimensions == 0
                || points < 4
                || points != plan.points_per_replicate()
                || plan.plans().len() < 2
                || plan
                    .plans()
                    .iter()
                    .any(|p| p.dim() != dimensions.checked_mul(2).unwrap_or(0))
            {
                return Err(invalid(
                    scioxide_core::argument!("points"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "QMC dimension, points or replicate count",
                ));
            }
            let execution = context.plan(
                options.execution_policy(),
                plan.plans().len(),
                scioxide_core::PlanLimits {
                    work_per_item: core::num::NonZeroUsize::MIN,
                    auto_threshold: NonZeroUsize::MIN,
                    worker_limit: context.worker_budget(),
                },
            );
            let per_replica = dimensions
                .checked_add(2)
                .and_then(|v| points.checked_mul(v))
                .and_then(|v| v.checked_mul(core::mem::size_of::<f64>()))
                .and_then(|v| {
                    v.checked_add(
                        dimensions
                            .checked_mul(5)?
                            .checked_mul(core::mem::size_of::<S>())?,
                    )
                })
                .and_then(|v| {
                    dimensions
                        .checked_mul(2)?
                        .checked_mul(core::mem::size_of::<u64>())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    dimensions
                        .checked_mul(2)?
                        .checked_mul(core::mem::size_of::<f64>())?
                        .checked_add(v)
                })
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_qmc_replica"))
                })?;
            let bytes = execution
                .worker_count()
                .checked_mul(per_replica)
                .and_then(|v| {
                    plan.plans()
                        .len()
                        .checked_mul(core::mem::size_of::<Option<SobolIndices>>())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    plan.plans()
                        .len()
                        .checked_mul(core::mem::size_of::<
                            PartitionSlot<(&QmcPlan, &mut Option<SobolIndices>)>,
                        >())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    plan.plans()
                        .len()
                        .checked_mul(core::mem::size_of::<SobolIndices>())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    plan.plans()
                        .len()
                        .checked_mul(dimensions)?
                        .checked_mul(2)?
                        .checked_mul(core::mem::size_of::<f64>())?
                        .checked_add(v)
                })
                .and_then(|v| {
                    dimensions
                        .checked_mul(4)?
                        .checked_mul(core::mem::size_of::<f64>())?
                        .checked_add(v)
                })
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_qmc_workspace"))
                })?;
            crate::error_helpers::limit(
                bytes,
                options.max_bytes(),
                scioxide_core::quantity!("sensitivity_bytes"),
            )?;
            let mut staged = try_filled(
                plan.plans().len(),
                None,
                scioxide_core::quantity!("sobol_qmc_estimates"),
            )?;
            let mut slots = try_with_capacity(
                plan.plans().len(),
                scioxide_core::quantity!("sobol_qmc_slots"),
            )?;
            for (replica, result) in plan.plans().iter().zip(&mut staged) {
                slots.push(PartitionSlot::new((replica, result)));
            }
            let evaluations = points
                .checked_mul(plan.plans().len())
                .and_then(|n| dimensions.checked_add(2).and_then(|d| n.checked_mul(d)))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_evaluations"))
                })?;
            solve
                .record_providers(scioxide_core::EvaluationRole::Function, evaluations)?
                .require()?;
            context.execute(execution, &|partition, (), _| {
                for index in partition.range() {
                    let (replica, result) = slots[index].take().ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("context"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "duplicate QMC replicate assignment",
                        )
                    })?;
                    *result = Some(qmc_replica(
                        replica, dimensions, points, &transform, &model, context,
                    )?);
                }
                Ok(())
            })?;

            drop(slots);
            let mut estimates = try_with_capacity(
                staged.len(),
                scioxide_core::quantity!("sobol_qmc_completed_estimates"),
            )?;
            for result in staged {
                estimates.push(result.ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("context"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "missing QMC result",
                    )
                })?);
            }
            let count = estimates.len() as f64;
            let mut first_order = try_filled(
                dimensions,
                0.0,
                scioxide_core::quantity!("sobol_first_mean"),
            )?;
            let mut total_order = try_filled(
                dimensions,
                0.0,
                scioxide_core::quantity!("sobol_total_mean"),
            )?;
            let mut first_se =
                try_filled(dimensions, 0.0, scioxide_core::quantity!("sobol_first_se"))?;
            let mut total_se =
                try_filled(dimensions, 0.0, scioxide_core::quantity!("sobol_total_se"))?;
            for j in 0..dimensions {
                first_order[j] = estimates.iter().map(|e| e.first_order[j] / count).sum();
                total_order[j] = estimates.iter().map(|e| e.total_order[j] / count).sum();
                first_se[j] = (estimates
                    .iter()
                    .map(|e| (e.first_order[j] - first_order[j]).powi(2))
                    .sum::<f64>()
                    / (count * (count - 1.0)))
                    .sqrt();
                total_se[j] = (estimates
                    .iter()
                    .map(|e| (e.total_order[j] - total_order[j]).powi(2))
                    .sum::<f64>()
                    / (count * (count - 1.0)))
                    .sqrt();
                if !first_order[j].is_finite()
                    || !total_order[j].is_finite()
                    || !first_se[j].is_finite()
                    || !total_se[j].is_finite()
                {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "unrepresentable randomized Sobol result",
                    ));
                }
            }
            Ok(SobolIndices {
                first_order,
                total_order,
                rows: points.checked_mul(estimates.len()).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sobol_qmc_rows"))
                })?,
                randomized_standard_error: Some((first_se, total_se)),
            })
        },
    )
}
