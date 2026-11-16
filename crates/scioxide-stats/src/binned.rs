//! Rectilinear bin grids and statistics computed per bin.
mod accumulation;

use crate::arithmetic::Compensated;
use crate::arithmetic::RangeSum;
use crate::central::Central;
use crate::error_helpers::invalid;
use crate::estimation::{execute, worker_count};
use crate::{
    KernelWeights, MomentOrder, ObservationData, QuantileMethod, QuantileProbability,
    StatisticsOptions,
};
use accumulation::{Mass, Mean};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

/// Location of a point relative to a [`BinPlan`] grid.
///
/// A point outside the grid on several axes is classified by the first such
/// axis, so every point falls in exactly one location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinLocation {
    /// Inside the grid, with the row-major bin index (the last axis varies fastest).
    Inside(usize),
    /// Below the first edge of this axis.
    Underflow(usize),
    /// Above the last edge of this axis.
    Overflow(usize),
}

/// Rectilinear grid of bins given by strictly increasing edges on each axis.
///
/// An axis with edges `e₀ < e₁ < … < eₘ` has `m` bins. Bin `i` is the half-open
/// interval `[eᵢ, eᵢ₊₁)`, except the last bin, which also contains `eₘ`. Bins
/// are numbered in row-major order, with the last axis varying fastest. Use the
/// plan with [`binned_statistics`](crate::binned_statistics) or
/// [`binned_statistics_into`](crate::binned_statistics_into).
#[derive(Clone, Debug)]
pub struct BinPlan<S: RealScalar> {
    edges: Vec<BinAxis<S>>,
    bins: usize,
}
#[derive(Clone, Debug)]
struct BinAxis<S> {
    edges: Vec<S>,
    prediction: Option<(f64, f64)>,
}
impl<S: RealScalar> BinAxis<S> {
    fn new(edges: Vec<S>) -> Self {
        let low = edges[0].into_f64();
        let width = edges[edges.len() - 1].into_f64() - low;
        let inverse = (edges.len() - 1) as f64 / width;
        // Enable the predictor only when edges map near consecutive integers.
        // This selects regular grids; correctness never depends on prediction.
        let prediction = (width.is_finite()
            && inverse.is_finite()
            && inverse > 0.0
            && edges.iter().enumerate().all(|(i, edge)| {
                let index = (edge.into_f64() - low) * inverse;
                (index + 0.5) as usize == i
            }))
        .then_some((low, inverse));
        Self { edges, prediction }
    }
    #[inline]
    fn locate(&self, x: S) -> usize {
        if let Some((low, inverse)) = self.prediction {
            let bin = (((x.into_f64() - low) * inverse) as usize).min(self.edges.len() - 2);
            if self.edges[bin] <= x
                && (x < self.edges[bin + 1]
                    || (bin == self.edges.len() - 2 && x == self.edges[bin + 1]))
            {
                return bin;
            }
        }
        self.edges
            .partition_point(|edge| *edge <= x)
            .saturating_sub(1)
            .min(self.edges.len() - 2)
    }
}
impl<S: RealScalar> BinPlan<S> {
    /// Creates a grid from the edges of each axis.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `edges` is empty.
    /// - `InvalidInput` if an axis has fewer than two edges, or its edges are not
    ///   finite and strictly increasing.
    /// - `SizeOverflow` if the total number of bins overflows `usize`.
    pub fn new(edges: &[&[S]]) -> Result<Self, SciOxideError> {
        if edges.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("edges"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "bins need at least one dimension",
            ));
        }
        let mut owned = try_with_capacity(edges.len(), scioxide_core::quantity!("bin_axes"))?;
        let mut bins = 1usize;
        for axis in edges {
            if axis.len() < 2
                || axis.iter().any(|x| !x.is_finite())
                || axis.windows(2).any(|x| x[0] >= x[1])
            {
                return Err(invalid(
                    scioxide_core::argument!("edges"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "edges must be finite and strictly increasing",
                ));
            }
            bins = bins.checked_mul(axis.len() - 1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("bin_product"))
            })?;
            let mut copy = try_with_capacity(axis.len(), scioxide_core::quantity!("bin_edges"))?;
            copy.extend_from_slice(axis);
            owned.push(BinAxis::new(copy));
        }
        Ok(Self { edges: owned, bins })
    }
    /// Creates a grid of equal-width bins from `(low, high, n)` for each axis.
    ///
    /// Each axis has `n` bins on `[low, high]`. The end edges are exactly `low`
    /// and `high`. Interior edge `i` is `low·(n − i)/n + high·i/n`, rounded to `S`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axes` is empty.
    /// - `InvalidInput` if `n = 0`, a bound is not finite, or `low ≥ high`.
    /// - `InvalidInput` if adjacent edges round to the same value in `S`.
    /// - `SizeOverflow` if the total number of bins overflows `usize`.
    pub fn uniform(axes: &[(S, S, usize)]) -> Result<Self, SciOxideError> {
        if axes.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("axes"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "bins need at least one dimension",
            ));
        }
        let mut edges = try_with_capacity(axes.len(), scioxide_core::quantity!("uniform_axes"))?;
        let mut bins = 1usize;
        for &(low, high, n) in axes {
            if n == 0 || !low.is_finite() || !high.is_finite() || low >= high {
                return Err(invalid(
                    scioxide_core::argument!("axes"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "invalid uniform bounds",
                ));
            }
            bins = bins.checked_mul(n).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("bin_product"))
            })?;
            let size = n.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("bin_edges"))
            })?;
            let mut axis = try_with_capacity(size, scioxide_core::quantity!("uniform_edges"))?;
            for i in 0..=n {
                let edge = if i == 0 {
                    low
                } else if i == n {
                    high
                } else {
                    crate::arithmetic::affine_with_fractions(
                        low,
                        high,
                        crate::arithmetic::count_ratio(n - i, n)?,
                        crate::arithmetic::count_ratio(i, n)?,
                    )?
                };
                if axis.last().is_some_and(|previous| *previous >= edge) {
                    return Err(invalid(
                        scioxide_core::argument!("axes"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "uniform edges collapse at this precision",
                    ));
                }
                axis.push(edge);
            }
            edges.push(BinAxis::new(axis));
        }
        Ok(Self { edges, bins })
    }
    /// Returns the number of axes.
    pub fn dimensions(&self) -> usize {
        self.edges.len()
    }
    /// Returns the total number of bins, the product of the bin counts of all axes.
    pub const fn bins(&self) -> usize {
        self.bins
    }
    /// Returns the edges of `axis`, or `None` if `axis ≥ dimensions()`.
    pub fn edges(&self, axis: usize) -> Option<&[S]> {
        self.edges.get(axis).map(|axis| axis.edges.as_slice())
    }
    /// Returns the bin that contains `point`, or the first axis on which it lies outside.
    ///
    /// The last edge of each axis belongs to the last bin.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `point.len()` differs from [`dimensions`](Self::dimensions).
    /// - `InvalidInput` if a coordinate is NaN or infinite.
    pub fn locate(&self, point: &[S]) -> Result<BinLocation, SciOxideError> {
        if point.len() != self.dimensions() {
            return Err(invalid(
                scioxide_core::argument!("point"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "point dimension mismatch",
            ));
        }
        if point.iter().any(|x| !x.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("point"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "non-finite bin coordinate",
            ));
        }
        Ok(self.locate_by(|d| point[d]))
    }
    // Admission or locate has already checked every coordinate, including axes
    // after the first outlier. Geometry and edge inclusion are shared.
    fn locate_by(&self, point: impl Fn(usize) -> S) -> BinLocation {
        let mut index = 0;
        let mut outside = None;
        for (d, geometry) in self.edges.iter().enumerate() {
            let axis = &geometry.edges;
            let x = point(d);
            if x < axis[0] {
                outside = outside.or(Some(BinLocation::Underflow(d)));
            } else if x > axis[axis.len() - 1] {
                outside = outside.or(Some(BinLocation::Overflow(d)));
            } else {
                index = index * (axis.len() - 1) + geometry.locate(x);
            }
        }
        outside.unwrap_or(BinLocation::Inside(index))
    }
    fn volume(&self, mut index: usize) -> Result<Scaled<f64>, SciOxideError> {
        let mut volume = Scaled::<f64>::ONE;
        for geometry in self.edges.iter().rev() {
            let axis = &geometry.edges;
            let i = index % (axis.len() - 1);
            index /= axis.len() - 1;
            volume = volume.mul(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(axis[i + 1]))?.sub(
                    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(axis[i]))?,
                )?,
            )?;
        }
        Ok(volume)
    }
}

/// Statistic computed in each bin by [`binned_statistics`](crate::binned_statistics).
///
/// The mass of an observation is its weight: 1 when unweighted, the
/// replication count for frequency weights, or the weight for mass weights.
/// `Sum` through `Median` summarize a separate value per observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinnedStatistic {
    /// Number of observations, counting replications. With mass weights,
    /// the number of observations with positive weight.
    Count,
    /// Total mass in the bin, not normalized.
    Mass,
    /// Bin mass divided by the mass of all accepted observations, including
    /// those outside the grid.
    RelativeFrequency,
    /// Cumulative mass of the bins up to and including this one, not
    /// normalized. Requires a one-dimensional grid.
    CumulativeFrequency,
    /// Bin mass divided by the total mass inside the grid and by the bin
    /// volume, so the density integrates to 1 over the grid.
    Density,
    /// Weighted sum of the values.
    Sum,
    /// Weighted mean of the values.
    Mean,
    /// Weighted population standard deviation of the values (divisor: the bin mass).
    StandardDeviation,
    /// Smallest value with positive weight.
    Minimum,
    /// Largest value with positive weight.
    Maximum,
    /// Median of the values: Hyndman–Fan type 7 for unweighted data and
    /// frequency weights, and the inverse of the weighted empirical CDF for mass
    /// weights.
    Median,
}
impl BinnedStatistic {
    fn needs_values(self) -> bool {
        matches!(
            self,
            Self::Sum
                | Self::Mean
                | Self::StandardDeviation
                | Self::Minimum
                | Self::Maximum
                | Self::Median
        )
    }
}

/// Result for one bin.
///
/// Statistics that are undefined for an empty bin (mean, standard deviation,
/// minimum, maximum, median) are `None`, never NaN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinEstimate<S> {
    /// Number of observations, counting replications. With mass weights, the
    /// number of observations with positive weight.
    pub count: usize,
    /// Total mass in the bin, or `None` if it overflows `S`.
    pub mass: Option<S>,
    /// Requested statistic. Counts, masses, sums and frequencies of an empty bin are zero.
    pub value: Option<S>,
}
impl<S: RealScalar> Default for BinEstimate<S> {
    fn default() -> Self {
        Self {
            count: 0,
            mass: Some(S::ZERO),
            value: None,
        }
    }
}
/// Totals for one batch of [`binned_statistics`](crate::binned_statistics).
///
/// Each mass is `None` if it overflows `S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinnedDiagnostics<S> {
    /// Mass of all accepted observations, inside and outside the grid.
    pub admitted_mass: Option<S>,
    /// Mass inside the grid.
    pub retained_mass: Option<S>,
    /// Mass of observations located as [`BinLocation::Underflow`].
    pub underflow_mass: Option<S>,
    /// Mass of observations located as [`BinLocation::Overflow`].
    pub overflow_mass: Option<S>,
    /// Number of skipped observations: missing or non-finite ones skipped by
    /// policy, and those with zero weight.
    pub skipped: usize,
}
impl<S: RealScalar> Default for BinnedDiagnostics<S> {
    fn default() -> Self {
        Self {
            admitted_mass: Some(S::ZERO),
            retained_mass: Some(S::ZERO),
            underflow_mass: Some(S::ZERO),
            overflow_mass: Some(S::ZERO),
            skipped: 0,
        }
    }
}
/// Storage sizes for one binned computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BinnedWorkspaceRequirements {
    /// Observations per batch. Only the median needs scratch of this size.
    pub samples: usize,
    /// Statistic the storage is sized for.
    pub statistic: BinnedStatistic,
    /// Bins per batch.
    pub bins: usize,
    /// Number of batches.
    pub batches: usize,
    /// Number of workers.
    pub workers: usize,
}
impl BinnedWorkspaceRequirements {
    /// Returns the heap bytes the workspace needs, excluding allocator overhead and the output.
    ///
    /// # Errors
    /// Returns `SizeOverflow` if the byte count overflows `usize`.
    pub(crate) fn required_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let workers = crate::estimation::storage_bytes(&[
            (
                self.bins,
                core::mem::size_of::<usize>() + core::mem::size_of::<RangeSum>(),
            ),
            (
                self.scratch_bins(BinnedStatistic::Sum),
                core::mem::size_of::<RangeSum>(),
            ),
            (
                self.scratch_bins(BinnedStatistic::Mean),
                core::mem::size_of::<Mean<S>>(),
            ),
            (
                self.scratch_bins(BinnedStatistic::StandardDeviation),
                core::mem::size_of::<Option<Central<S>>>(),
            ),
            (self.extrema_bins(), core::mem::size_of::<Option<S>>()),
            (
                self.median_samples(),
                core::mem::size_of::<Record<S>>()
                    + core::mem::size_of::<(S, u64)>()
                    + core::mem::size_of::<(S, S)>(),
            ),
        ])?;
        let batches = crate::estimation::storage_bytes(&[(
            self.bins,
            core::mem::size_of::<BinEstimate<S>>(),
        )])?;
        crate::estimation::storage_bytes(&[
            (self.workers, workers),
            (self.workers, core::mem::size_of::<Worker<S>>()),
            (self.batches, batches),
            (self.batches, core::mem::size_of::<Batch<S>>()),
        ])
    }
    fn scratch_bins(self, statistic: BinnedStatistic) -> usize {
        if self.statistic == statistic {
            self.bins
        } else {
            0
        }
    }
    fn extrema_bins(self) -> usize {
        if matches!(
            self.statistic,
            BinnedStatistic::Minimum | BinnedStatistic::Maximum
        ) {
            self.bins
        } else {
            0
        }
    }
    fn median_samples(self) -> usize {
        if self.statistic == BinnedStatistic::Median {
            self.samples
        } else {
            0
        }
    }
    /// Computes the storage sizes for this grid, data shape and execution context.
    ///
    /// # Errors
    /// Fails if the grid and data dimensions differ, or the output size overflows.
    pub(crate) fn new<S: RealScalar>(
        plan: &BinPlan<S>,
        view: ObservationData<'_, S>,
        statistic: BinnedStatistic,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if plan.dimensions() != view.dimensions() {
            return Err(invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "bin dimension mismatch",
            ));
        }
        plan.bins.checked_mul(view.batches()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("binned_outputs"))
        })?;
        Ok(Self {
            samples: view.samples(),
            statistic,
            bins: plan.bins,
            batches: view.batches(),
            workers: worker_count(options, context, view.batches())?,
        })
    }
}
#[derive(Clone, Copy)]
struct Record<S> {
    bin: usize,
    value: S,
    count: usize,
    original_mass: S,
}
struct Worker<S: RealScalar> {
    counts: Vec<usize>,
    masses: Vec<RangeSum>,
    sums: Vec<RangeSum>,
    means: Vec<Mean<S>>,
    central: Vec<Option<Central<S>>>,
    extrema: Vec<Option<S>>,
    records: Vec<Record<S>>,
    frequency: Vec<(S, u64)>,
    mass: Vec<(S, S)>,
}
impl<S: RealScalar> Worker<S> {
    #[inline]
    fn bin_mass(&self, bin: usize, relative: bool) -> Mass {
        if relative {
            Mass::relative(self.masses[bin])
        } else {
            Mass::count(self.counts[bin])
        }
    }
}
struct Batch<S> {
    bins: Vec<BinEstimate<S>>,
    diagnostics: BinnedDiagnostics<S>,
}
/// Per-worker bin accumulators and per-batch result buffers.
/// Only the median stores observations.
pub(crate) struct RawBinnedWorkspace<S: RealScalar> {
    workers: Vec<Worker<S>>,
    staged: Vec<Batch<S>>,
    statistic: Option<BinnedStatistic>,
}
impl<S: RealScalar> Default for RawBinnedWorkspace<S> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            staged: Vec::new(),
            statistic: None,
        }
    }
}
impl<S: RealScalar> RawBinnedWorkspace<S> {
    pub(crate) fn sufficient(&self, r: BinnedWorkspaceRequirements) -> bool {
        (r.batches == 0 || self.statistic == Some(r.statistic))
            && self.workers.len() >= r.workers
            && self
                .workers
                .iter()
                .take(r.workers)
                .all(|w| w.records.len() >= r.median_samples() && w.counts.len() >= r.bins)
            && self.staged.len() >= r.batches
            && self
                .staged
                .iter()
                .take(r.batches)
                .all(|b| b.bins.len() >= r.bins)
    }
    /// Allocates storage of the given sizes. On error the previous storage is unchanged.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure.
    pub(crate) fn reserve(&mut self, r: BinnedWorkspaceRequirements) -> Result<(), SciOxideError> {
        r.required_bytes::<S>()?;
        let mut workers = try_with_capacity(r.workers, scioxide_core::quantity!("bin_workers"))?;
        for _ in 0..r.workers {
            workers.push(Worker {
                counts: try_filled(r.bins, 0, scioxide_core::quantity!("binned_counts"))?,
                masses: try_filled(
                    r.bins,
                    RangeSum::ZERO,
                    scioxide_core::quantity!("binned_masses"),
                )?,
                sums: try_filled(
                    r.scratch_bins(BinnedStatistic::Sum),
                    RangeSum::ZERO,
                    scioxide_core::quantity!("binned_sums"),
                )?,
                means: try_filled(
                    r.scratch_bins(BinnedStatistic::Mean),
                    Mean::empty(),
                    scioxide_core::quantity!("binned_means"),
                )?,
                central: try_filled(
                    r.scratch_bins(BinnedStatistic::StandardDeviation),
                    None,
                    scioxide_core::quantity!("binned_moments"),
                )?,
                extrema: try_filled(
                    r.extrema_bins(),
                    None,
                    scioxide_core::quantity!("binned_extrema"),
                )?,
                records: try_filled(
                    r.median_samples(),
                    Record {
                        bin: 0,
                        value: S::ZERO,
                        count: 0,
                        original_mass: S::ZERO,
                    },
                    scioxide_core::quantity!("binned_records"),
                )?,
                frequency: try_filled(
                    r.median_samples(),
                    (S::ZERO, 0),
                    scioxide_core::quantity!("binned_frequency"),
                )?,
                mass: try_filled(
                    r.median_samples(),
                    (S::ZERO, S::ZERO),
                    scioxide_core::quantity!("binned_mass"),
                )?,
            });
        }
        let mut staged = try_with_capacity(r.batches, scioxide_core::quantity!("bin_batches"))?;
        for _ in 0..r.batches {
            staged.push(Batch {
                bins: try_filled(
                    r.bins,
                    BinEstimate::default(),
                    scioxide_core::quantity!("binned_bins"),
                )?,
                diagnostics: BinnedDiagnostics::default(),
            });
        }
        *self = Self {
            workers,
            staged,
            statistic: Some(r.statistic),
        };
        Ok(())
    }
}

/// Computes `statistic` in every bin of every batch into `output` and `diagnostics`.
///
/// `values` is in batch-major, sample-minor order. Observations are accumulated
/// in input order. Only the median sorts the observations of each bin. Scratch
/// is O(workers · bins) for all other statistics.
///
/// # Errors
/// Fails on inconsistent shapes, invalid data, a workspace that is too small or
/// sized for another statistic, a result that overflows `S`, zero normalization
/// mass, or an execution failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn binned_statistics_into<S: RealScalar>(
    plan: &BinPlan<S>,
    view: ObservationData<'_, S>,
    values: Option<&[S]>,
    statistic: BinnedStatistic,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    workspace: &mut RawBinnedWorkspace<S>,
    output: &mut [BinEstimate<S>],
    diagnostics: &mut [BinnedDiagnostics<S>],
) -> Result<(), SciOxideError> {
    let r = BinnedWorkspaceRequirements::new(plan, view, statistic, options, context)?;
    if output.len() != r.bins * r.batches
        || diagnostics.len() != r.batches
        || values.is_some_and(|v| v.len() != r.samples * r.batches)
        || (statistic.needs_values() && values.is_none())
    {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "binned input/output shape mismatch",
        ));
    }
    if statistic == BinnedStatistic::CumulativeFrequency && plan.dimensions() != 1 {
        return Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "cumulative frequency requires one dimension",
        ));
    }
    if (r.batches != 0 && workspace.statistic != Some(statistic))
        || workspace.workers.len() < r.workers
        || workspace
            .workers
            .iter()
            .take(r.workers)
            .any(|w| w.records.len() < r.median_samples() || w.counts.len() < r.bins)
        || workspace.staged.len() < r.batches
        || workspace
            .staged
            .iter()
            .take(r.batches)
            .any(|b| b.bins.len() < r.bins)
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient binned workspace",
        ));
    }
    execute(
        options,
        context,
        r.batches,
        1,
        &mut workspace.workers,
        &mut workspace.staged,
        |batch, worker, staged| {
            let result = &mut staged[0];
            let origin = view.batch_origin(batch);
            worker.counts[..r.bins].fill(0);
            worker.masses[..r.bins].fill(RangeSum::ZERO);
            worker.sums.fill(RangeSum::ZERO);
            worker.means.fill(Mean::empty());
            worker.central.fill(None);
            worker.extrema.fill(None);
            let relative = matches!(view.weights, Some(KernelWeights::Mass(_)));
            let mut total = Mass::zero();
            let mut retained = Mass::zero();
            let mut under = Mass::zero();
            let mut over = Mass::zero();
            let mut len = 0;
            let mut skipped = 0;
            for i in 0..r.samples {
                let coordinate = |d| view.value_at_origin(origin, i, d);
                let Some(count) = view.admit_count_by(
                    batch,
                    i,
                    options,
                    scioxide_core::argument!("view"),
                    coordinate,
                )?
                else {
                    skipped += 1;
                    continue;
                };
                let value = values.map_or(S::ZERO, |v| v[batch * r.samples + i]);
                if statistic.needs_values() && !value.is_finite() {
                    if options.non_finite_policy() == crate::NonFinitePolicy::Skip {
                        skipped += 1;
                        continue;
                    }
                    return Err(invalid(
                        scioxide_core::argument!("values"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "non-finite binned value",
                    ));
                }
                let weight = match view.weights {
                    Some(KernelWeights::Mass(w)) => Some(w[batch * r.samples + i].into_f64()),
                    _ => None,
                };
                total.add(count, weight)?;
                match plan.locate_by(coordinate) {
                    BinLocation::Underflow(_) => under.add(count, weight)?,
                    BinLocation::Overflow(_) => over.add(count, weight)?,
                    BinLocation::Inside(bin) => {
                        retained.add(count, weight)?;
                        let old = worker.bin_mass(bin, relative);
                        if let Some(weight) = weight {
                            worker.masses[bin].add_product(weight, 1.0)?;
                        }
                        worker.counts[bin] =
                            worker.counts[bin].checked_add(count).ok_or_else(|| {
                                SciOxideError::size_overflow(scioxide_core::quantity!("bin_count"))
                            })?;
                        match statistic {
                            BinnedStatistic::Sum => worker.sums[bin]
                                .add_product(value.into_f64(), weight.unwrap_or(count as f64))?,
                            BinnedStatistic::Mean => worker.means[bin].add(
                                value,
                                weight.unwrap_or(count as f64),
                                old,
                                relative,
                            )?,
                            BinnedStatistic::StandardDeviation => {
                                worker.central[bin] = Some(match worker.central[bin] {
                                    None => Central::singleton(value)?,
                                    Some(c) => {
                                        let next =
                                            worker.bin_mass(bin, relative).scaled(relative)?;
                                        c.merged(
                                            Central::singleton(value)?,
                                            crate::arithmetic::Shares::rounded(
                                                old.scaled(relative)?.div(next)?,
                                                Scaled::from_value(weight.unwrap_or(count as f64))?
                                                    .div(next)?,
                                            ),
                                            MomentOrder::Second,
                                        )?
                                    }
                                });
                            }
                            BinnedStatistic::Minimum | BinnedStatistic::Maximum => {
                                worker.extrema[bin] =
                                    Some(worker.extrema[bin].map_or(value, |previous| {
                                        if (statistic == BinnedStatistic::Minimum
                                            && value < previous)
                                            || (statistic == BinnedStatistic::Maximum
                                                && value > previous)
                                        {
                                            value
                                        } else {
                                            previous
                                        }
                                    }));
                            }
                            BinnedStatistic::Median => {
                                worker.records[len] = Record {
                                    bin,
                                    value,
                                    count,
                                    original_mass: match view.weights {
                                        Some(KernelWeights::Mass(w)) => w[batch * r.samples + i],
                                        _ => S::ONE,
                                    },
                                };
                                len += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
            let total_scaled = total.scaled(relative)?;
            let retained_scaled = retained.scaled(relative)?;
            if (statistic == BinnedStatistic::Density && retained_scaled.is_zero())
                || (statistic == BinnedStatistic::RelativeFrequency && total_scaled.is_zero())
            {
                return Err(invalid(
                    scioxide_core::argument!("view"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "zero normalization mass",
                ));
            }
            if statistic == BinnedStatistic::Median {
                worker.records[..len].sort_unstable_by_key(|x| x.bin);
            }
            let mut start = 0;
            let mut prefix = Compensated::<Scaled<f64>>::zero();
            let mut integer_prefix = Mass::zero();
            for (bin, out) in result.bins[..r.bins].iter_mut().enumerate() {
                let count = worker.counts[bin];
                let mass = worker.bin_mass(bin, relative);
                let value = match statistic {
                    BinnedStatistic::Count => Some(
                        scioxide_core::backend::scalar_from_u128_rounded(count as u128)?,
                    ),
                    BinnedStatistic::Mass => Some(mass.requested_value(relative)?),
                    BinnedStatistic::Sum => Some(worker.sums[bin].total()?.round_to()?),
                    BinnedStatistic::RelativeFrequency => {
                        Some(mass.scaled(relative)?.div(total_scaled)?.round_to()?)
                    }
                    BinnedStatistic::CumulativeFrequency => {
                        if relative {
                            prefix = prefix.added(mass.scaled(true)?)?;
                            Some(prefix.total()?.round_to()?)
                        } else {
                            integer_prefix.add(count, None)?;
                            Some(integer_prefix.requested_value(false)?)
                        }
                    }
                    BinnedStatistic::Density => Some(
                        mass.scaled(relative)?
                            .div(retained_scaled)?
                            .div(plan.volume(bin)?)?
                            .round_to()?,
                    ),
                    BinnedStatistic::Mean => worker.means[bin].value(mass, relative)?,
                    BinnedStatistic::StandardDeviation => worker.central[bin]
                        .map(|c| c.m2()?.sqrt()?.round_to())
                        .transpose()?,
                    BinnedStatistic::Minimum | BinnedStatistic::Maximum => worker.extrema[bin],
                    BinnedStatistic::Median => {
                        let end =
                            start + worker.records[start..len].partition_point(|x| x.bin == bin);
                        let records = &worker.records[start..end];
                        start = end;
                        if records.is_empty() {
                            None
                        } else {
                            let p = QuantileProbability::new(0.5)?;
                            Some(if relative {
                                for (out, record) in worker.mass.iter_mut().zip(records) {
                                    *out = (record.value, record.original_mass);
                                }
                                crate::weighted_inverse_ecdf_quantile_in_place(
                                    &mut worker.mass[..records.len()],
                                    p,
                                )?
                                .value()
                            } else {
                                for (out, record) in worker.frequency.iter_mut().zip(records) {
                                    *out = (record.value, record.count as u64);
                                }
                                crate::frequency_quantile_in_place(
                                    &mut worker.frequency[..records.len()],
                                    p,
                                    QuantileMethod::Linear,
                                )?
                                .value()
                            })
                        }
                    }
                };
                *out = BinEstimate {
                    count,
                    mass: mass.value(relative)?,
                    value,
                };
            }
            result.diagnostics = BinnedDiagnostics {
                admitted_mass: total.value(relative)?,
                retained_mass: retained.value(relative)?,
                underflow_mass: under.value(relative)?,
                overflow_mass: over.value(relative)?,
                skipped,
            };
            Ok(())
        },
    )?;
    for (batch, staged) in workspace.staged[..r.batches].iter().enumerate() {
        output[batch * r.bins..(batch + 1) * r.bins].copy_from_slice(&staged.bins[..r.bins]);
        diagnostics[batch] = staged.diagnostics;
    }
    Ok(())
}

/// Result of [`binned_statistics`](crate::binned_statistics) for all batches.
#[derive(Clone, Debug, PartialEq)]
pub struct BinnedResult<S> {
    /// Bin results in batch-major order: the bins of each batch are contiguous,
    /// in row-major bin order.
    pub bins: Vec<BinEstimate<S>>,
    /// Totals for each batch.
    pub diagnostics: Vec<BinnedDiagnostics<S>>,
}

/// Stores counts and strictly increasing edges of a one-dimensional histogram.
#[derive(Clone, Debug)]
pub struct Histogram<S: RealScalar> {
    edges: Vec<S>,
    counts: Vec<usize>,
}
impl<S: RealScalar> Histogram<S> {
    /// Returns the bin edges; the final bin includes its right endpoint.
    pub fn edges(&self) -> &[S] {
        &self.edges
    }
    /// Returns the number of observations in each bin.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }
    /// Returns the edges and counts.
    pub fn into_parts(self) -> (Vec<S>, Vec<usize>) {
        (self.edges, self.counts)
    }
}
/// Counts a sample in equal-width bins spanning its minimum and maximum.
///
/// Uses [`BinPlan::uniform`] and [`crate::binned_statistics`], with exact integer counts.
/// Costs O(n log b + b) time and O(n + b) storage for n values and b bins.
///
/// # Errors
///
/// Rejects empty, non-finite or constant samples, zero bins, and coincident edges.
/// Propagates allocation and size-overflow errors.
///
/// ```
/// let histogram = scioxide_stats::histogram(&[0.0_f64, 0.1, 0.4, 0.6, 1.0], 4)?;
/// assert_eq!(histogram.counts(), &[2, 1, 1, 1]);
/// assert!((histogram.edges()[1] - 0.25).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn histogram<S: RealScalar>(values: &[S], bins: usize) -> Result<Histogram<S>, SciOxideError> {
    if bins == 0 {
        return Err(invalid(
            scioxide_core::argument!("bins"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "histogram needs at least one bin",
        ));
    }
    let low = crate::minimum(values)?;
    let high = crate::maximum(values)?;
    let plan = BinPlan::uniform(&[(low, high, bins)]).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["axes", "edges"],
            scioxide_core::argument!("values"),
        )
    })?;
    let shape = [values.len()];
    let strides = [1];
    let view = crate::BatchView::new(
        values,
        scioxide_core::StridedLayout::new(&shape, &strides)?,
        crate::Axis::Index(0),
    )?;
    let result = crate::binned_statistics(
        &plan,
        view,
        None,
        BinnedStatistic::Count,
        StatisticsOptions::default(),
    )?;
    let mut counts = try_with_capacity(bins, scioxide_core::quantity!("histogram_counts"))?;
    for estimate in result.bins {
        counts.push(estimate.count);
    }
    let mut edges = try_with_capacity(bins + 1, scioxide_core::quantity!("histogram_edges"))?;
    edges.extend_from_slice(plan.edges(0).ok_or_else(|| {
        invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "missing histogram axis",
        )
    })?);
    Ok(Histogram { edges, counts })
}
