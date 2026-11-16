//! Empirical and estimated CDFs of sample batches evaluated at query points.
use crate::empirical::{ExactMass, estimated_cdf_by};
use crate::error_helpers::invalid;
use crate::estimation::{execute, worker_count};
use crate::ranking::finite_cmp;
use crate::{KernelWeights, ObservationData, QuantileMethod, StatisticsOptions};
use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{ExecutionContext, RealScalar, Scaled, SciOxideError};

/// Probability computed at each query point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawCdfMethod {
    /// Empirical CDF `F̂(x) = P(X ≤ x)`, right-continuous.
    Empirical,
    /// Empirical survival function `P(X > x)`, accumulated from the upper tail.
    Survival,
    /// Estimate of the population CDF from the plotting positions of a quantile method.
    Estimated(QuantileMethod),
}
/// Storage sizes for one CDF computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CdfWorkspaceRequirements {
    /// Observations per worker.
    pub samples: usize,
    /// Number of batches.
    pub batches: usize,
    /// Query points per batch.
    pub queries: usize,
    /// Number of workers.
    pub workers: usize,
}
impl CdfWorkspaceRequirements {
    fn query_order_len(self) -> usize {
        if self.batches == 0 { 0 } else { self.queries }
    }

    /// Returns the heap bytes the workspace needs, excluding allocator overhead and the output.
    ///
    /// # Errors
    /// Returns `SizeOverflow` if the byte count overflows `usize`.
    pub(crate) fn required_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let records =
            crate::estimation::storage_bytes(&[(self.samples, core::mem::size_of::<Entry<S>>())])?;
        let outputs =
            crate::estimation::storage_bytes(&[(self.queries, core::mem::size_of::<S>())])?;
        crate::estimation::storage_bytes(&[
            (self.workers, records),
            (self.workers, core::mem::size_of::<Vec<Entry<S>>>()),
            (self.batches, outputs),
            (self.query_order_len(), core::mem::size_of::<usize>()),
        ])
    }
    /// Computes the storage sizes. All batches use the same query points.
    ///
    /// # Errors
    /// Fails if the observations are not one-dimensional, or the output size overflows.
    pub(crate) fn new<S: RealScalar>(
        view: ObservationData<'_, S>,
        queries: usize,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if view.dimensions() != 1 {
            return Err(invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CDF observations must be scalar",
            ));
        }
        queries
            .checked_mul(view.batches())
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("cdf_output")))?;
        Ok(Self {
            samples: view.samples(),
            batches: view.batches(),
            queries,
            workers: worker_count(options, context, view.batches())?,
        })
    }
}
#[derive(Clone, Copy)]
struct Entry<S> {
    value: S,
    weight: S,
    // Replications on admission, cumulative replications after sorting.
    count: usize,
}
/// Scratch for sorting samples and queries, and buffers for the results.
/// Its size is linear in the observations per worker, the query points and the
/// output length. Zero batches need no storage.
pub(crate) struct RawCdfWorkspace<S: RealScalar> {
    workers: Vec<Vec<Entry<S>>>,
    staged: Vec<S>,
    query_order: Vec<usize>,
}
impl<S: RealScalar> Default for RawCdfWorkspace<S> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            staged: Vec::new(),
            query_order: Vec::new(),
        }
    }
}
impl<S: RealScalar> RawCdfWorkspace<S> {
    pub(crate) fn sufficient(&self, r: CdfWorkspaceRequirements) -> bool {
        self.workers.len() >= r.workers
            && self
                .workers
                .iter()
                .take(r.workers)
                .all(|w| w.len() >= r.samples)
            && self.staged.len() >= r.queries * r.batches
            && self.query_order.len() >= r.query_order_len()
    }
    /// Allocates storage of the given sizes. On error the previous storage is unchanged.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure.
    pub(crate) fn reserve(&mut self, r: CdfWorkspaceRequirements) -> Result<(), SciOxideError> {
        r.required_bytes::<S>()?;
        let size = r
            .queries
            .checked_mul(r.batches)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("cdf_outputs")))?;
        let mut workers = try_with_capacity(r.workers, scioxide_core::quantity!("cdf_workers"))?;
        for _ in 0..r.workers {
            workers.push(try_filled(
                r.samples,
                Entry {
                    value: S::ZERO,
                    weight: S::ZERO,
                    count: 0,
                },
                scioxide_core::quantity!("cdf_worker_records"),
            )?);
        }
        *self = Self {
            workers,
            staged: try_filled(size, S::ZERO, scioxide_core::quantity!("cdf_staged"))?,
            query_order: try_filled(
                r.query_order_len(),
                0,
                scioxide_core::quantity!("cdf_query_order"),
            )?,
        };
        Ok(())
    }
}
/// Evaluates the CDF of every batch at every query point, in batch-major order.
///
/// Frequency weights give the same result as the expanded sample. Mass weights
/// support the empirical CDF, the survival function and the type 1 estimate only.
/// Each batch sorts its observations once. For `n` observations and `q` queries
/// the cost per batch is O(n log n + q log n), or O(n log n + q) with mass
/// weights after one O(q log q) sort of the queries.
///
/// # Errors
/// Fails on a NaN query, a batch with no accepted observations, an estimated CDF
/// other than type 1 with mass weights, a workspace that is too small, invalid
/// input or an execution failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdf_into<S: RealScalar>(
    view: ObservationData<'_, S>,
    queries: &[S],
    method: RawCdfMethod,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    workspace: &mut RawCdfWorkspace<S>,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let r = CdfWorkspaceRequirements::new(view, queries.len(), options, context)?;
    if output.len() != r.batches * r.queries
        || workspace.workers.len() < r.workers
        || workspace
            .workers
            .iter()
            .take(r.workers)
            .any(|w| w.len() < r.samples)
        || workspace.staged.len() < output.len()
        || workspace.query_order.len() < r.query_order_len()
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "CDF storage mismatch",
        ));
    }
    if queries.iter().any(|q| q.is_nan()) {
        return Err(invalid(
            scioxide_core::argument!("queries"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "NaN CDF query",
        ));
    }
    if matches!(view.weights, Some(KernelWeights::Mass(_)))
        && matches!(method,RawCdfMethod::Estimated(m) if m!=QuantileMethod::NearestRank)
    {
        return Err(invalid(
            scioxide_core::argument!("method"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "relative-mass estimated CDF requires type 1",
        ));
    }
    let relative_mass = matches!(view.weights, Some(KernelWeights::Mass(_)));
    let query_order = &mut workspace.query_order[..r.query_order_len()];
    if relative_mass {
        for (index, slot) in query_order.iter_mut().enumerate() {
            *slot = index;
        }
        query_order.sort_unstable_by(|&a, &b| finite_cmp(&queries[a], &queries[b]));
        if method == RawCdfMethod::Survival {
            query_order.reverse();
        }
    }
    execute(
        options,
        context,
        r.batches,
        r.queries,
        &mut workspace.workers,
        &mut workspace.staged,
        |batch, worker, out| {
            let mut len = 0;
            let mut total = ExactMass::zero();
            let mut count = 0usize;
            for i in 0..r.samples {
                let Some((_, n)) =
                    view.admit(batch, i, options, scioxide_core::argument!("view"))?
                else {
                    continue;
                };
                let weight = match view.weights {
                    Some(KernelWeights::Mass(w)) => {
                        let weight = w[batch * r.samples + i];
                        total.try_add_scalar(weight)?;
                        weight
                    }
                    _ => S::ZERO,
                };
                count = count.checked_add(n).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("cdf_replication_count"))
                })?;
                worker[len] = Entry {
                    value: view.value(batch, i, 0),
                    weight,
                    count: n,
                };
                len += 1;
            }
            if len == 0 {
                return Err(invalid(
                    scioxide_core::argument!("view"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "empty retained CDF sample",
                ));
            }
            worker[..len].sort_unstable_by(|a, b| finite_cmp(&a.value, &b.value));
            let entries = &mut worker[..len];
            if relative_mass {
                let denominator = total.to_scaled(0)?;
                let mut numerator = ExactMass::zero();
                let upper = method == RawCdfMethod::Survival;
                let mut cursor = if upper { len } else { 0 };
                // Reverse both the query order and sample traversal for direct
                // upper tails: no rounded probability or mass is subtracted.
                for &index in query_order.iter() {
                    let query = queries[index];
                    if upper {
                        while cursor > 0 && entries[cursor - 1].value > query {
                            cursor -= 1;
                            numerator.try_add_scalar(entries[cursor].weight)?;
                        }
                    } else {
                        while cursor < len && entries[cursor].value <= query {
                            numerator.try_add_scalar(entries[cursor].weight)?;
                            cursor += 1;
                        }
                    }
                    out[index] = numerator.to_scaled(0)?.div(denominator)?.round_to()?;
                }
                return Ok(());
            }
            let mut cumulative = 0;
            for entry in entries.iter_mut() {
                // Admission checked the total, so every positive prefix fits.
                cumulative += entry.count;
                entry.count = cumulative;
            }
            // usize -> f64 rounds identically to ExactMass::count(...).to_scaled(0).
            let denominator = Scaled::<f64>::from_value(count as f64)?;
            for (query, result) in queries.iter().zip(out) {
                let boundary = entries.partition_point(|entry| entry.value <= *query);
                let rank = boundary
                    .checked_sub(1)
                    .map_or(0, |index| entries[index].count);
                *result = match method {
                    RawCdfMethod::Estimated(m) => {
                        estimated_cdf_by(count, rank, *query, m, |index| {
                            entries[entries.partition_point(|entry| entry.count <= index)].value
                        })?
                    }
                    _ => {
                        let numerator = if method == RawCdfMethod::Survival {
                            count - rank
                        } else {
                            rank
                        };
                        Scaled::<f64>::from_value(numerator as f64)?
                            .div(denominator)?
                            .round_to()?
                    }
                };
            }
            Ok(())
        },
    )?;
    output.copy_from_slice(&workspace.staged[..output.len()]);
    Ok(())
}
