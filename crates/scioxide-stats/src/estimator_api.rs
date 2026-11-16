//! Batched empirical CDFs, binned statistics, kernel density estimates,
//! isotonic regression and directional means, with their workspaces.
use crate::error_helpers::limit_bytes;
use crate::{
    BatchView, BinEstimate, BinPlan, BinnedDiagnostics, BinnedResult, BinnedStatistic,
    EmpiricalWeights, IsotonicBatchOutput, IsotonicDirection, IsotonicWeights, KdeOptions,
    QuantileRule, StatisticsOptions, Unweighted,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::{Capacity, ExecutionContext, RealScalar, SciOxideError};

macro_rules! workspace {
    ($workspace:ident, $requirements:ident, $raw_workspace:ident, $raw_requirements:ident) => {
        /// Storage sizes for a workspace, checked against the byte limit
        /// [`StatisticsOptions::max_bytes`].
        ///
        /// See [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
        #[derive(Clone, Copy, Debug)]
        pub struct $requirements<S: RealScalar> {
            raw: crate::$raw_requirements,
            bytes: usize,
            max_bytes: usize,
            scalar: PhantomData<S>,
        }
        impl<S: RealScalar> $requirements<S> {
            fn new(
                raw: crate::$raw_requirements,
                options: StatisticsOptions,
            ) -> Result<Self, SciOxideError> {
                let bytes = raw.required_bytes::<S>()?;
                limit_bytes(bytes, options.max_bytes())?;
                Ok(Self {
                    raw,
                    bytes,
                    max_bytes: options.max_bytes(),
                    scalar: PhantomData,
                })
            }
            /// Returns the heap bytes the workspace needs, excluding the output
            /// and allocator overhead.
            pub const fn required_bytes(self) -> usize {
                self.bytes
            }
        }
        /// Reusable storage for one estimator.
        ///
        /// Create it with `workspace` from the requirements of a
        /// `workspace_requirements` call, or grow an existing one with
        /// `try_reserve`. See
        /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
        pub struct $workspace<S: RealScalar> {
            raw: crate::$raw_workspace<S>,
            retained_bytes: usize,
        }
        impl<S: RealScalar> Default for $workspace<S> {
            fn default() -> Self {
                Self {
                    raw: Default::default(),
                    retained_bytes: 0,
                }
            }
        }
        impl<S: RealScalar> $workspace<S> {
            /// Reports whether the workspace already holds the storage in `req`.
            ///
            /// # Errors
            ///
            /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit)
            ///   if the storage the workspace holds exceeds the byte limit of
            ///   `req`.
            pub fn check(&self, req: &$requirements<S>) -> Result<Capacity, SciOxideError> {
                limit_bytes(self.retained_bytes, req.max_bytes)?;
                Ok(if self.raw.sufficient(req.raw) {
                    Capacity::Sufficient
                } else {
                    Capacity::NeedsGrowth
                })
            }
            /// Grows the workspace to the storage in `req`. A workspace that is
            /// already large enough is kept as it is.
            ///
            /// # Errors
            ///
            /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit)
            ///   if the storage held or required exceeds the byte limit of
            ///   `req`.
            /// - [`AllocationFailure`](scioxide_core::SciOxideErrorKind::AllocationFailure)
            ///   if the allocation fails.
            pub fn try_reserve(&mut self, req: &$requirements<S>) -> Result<(), SciOxideError> {
                if self.check(req)? == Capacity::Sufficient {
                    return Ok(());
                }
                limit_bytes(req.bytes, req.max_bytes)?;
                self.raw.reserve(req.raw)?;
                self.retained_bytes = req.bytes;
                Ok(())
            }
            /// Creates a workspace that holds the storage in `req`.
            ///
            /// # Errors
            ///
            /// Fails as `try_reserve`.
            pub fn workspace(req: &$requirements<S>) -> Result<Self, SciOxideError> {
                let mut ws = Self::default();
                ws.try_reserve(req)?;
                Ok(ws)
            }
        }
    };
}
workspace!(
    CdfWorkspace,
    CdfRequirements,
    RawCdfWorkspace,
    CdfWorkspaceRequirements
);
workspace!(
    BinnedWorkspace,
    BinnedRequirements,
    RawBinnedWorkspace,
    BinnedWorkspaceRequirements
);
workspace!(
    KdeWorkspace,
    KdeRequirements,
    RawKdeWorkspace,
    KdeWorkspaceRequirements
);
workspace!(
    IsotonicWorkspace,
    IsotonicRequirements,
    RawIsotonicBatchWorkspace,
    IsotonicWorkspaceRequirements
);

impl<S: RealScalar> IsotonicWorkspace<S> {
    /// Returns the longest lane that every worker's storage can fit.
    pub fn capacity(&self) -> usize {
        self.raw
            .workers
            .iter()
            .map(crate::RawIsotonicWorkspace::capacity)
            .min()
            .unwrap_or(0)
    }
    /// Returns the requirements for fitting one slice of at most `samples`
    /// observations with [`isotonic_regression_into`].
    ///
    /// # Errors
    ///
    /// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
    ///   storage size overflows `usize`.
    /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
    ///   storage exceeds `options.max_bytes()`.
    pub fn scalar_requirements(
        samples: usize,
        options: StatisticsOptions,
    ) -> Result<IsotonicRequirements<S>, SciOxideError> {
        IsotonicRequirements::new(
            crate::IsotonicWorkspaceRequirements {
                lane_length: samples,
                batches: 1,
                workers: 1,
                slots: samples,
            },
            options,
        )
    }
    /// Returns the requirements for fitting every lane of `view` with
    /// [`isotonic_regression_axes_into`].
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if `view`
    ///   has a coordinate axis.
    /// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
    ///   storage size overflows `usize`.
    /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
    ///   storage exceeds `options.max_bytes()`.
    pub fn workspace_requirements<M: IsotonicWeights>(
        view: BatchView<'_, S, M>,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<IsotonicRequirements<S>, SciOxideError> {
        if view.dimensions() != 1 {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "isotonic observations must be scalar",
            ));
        }
        let policy = crate::IsotonicPolicy {
            direction: IsotonicDirection::Increasing,
            data: options.order(view.axis()),
        };
        IsotonicRequirements::new(
            crate::IsotonicWorkspaceRequirements::new(
                crate::OrderData::from_batch(view),
                policy,
                context,
            )?,
            options,
        )
    }
}

impl<S: RealScalar> IsotonicRequirements<S> {
    /// Returns the maximum number of observations per lane.
    pub const fn lane_length(self) -> usize {
        self.raw.lane_length
    }
    /// Returns the number of lanes, one fit each.
    pub const fn batches(self) -> usize {
        self.raw.batches
    }
    /// Returns the number of fitted values, which is also the number of block
    /// slots.
    pub const fn slots(self) -> usize {
        self.raw.slots
    }
    /// Resolved worker count.
    pub const fn workers(self) -> usize {
        self.raw.workers
    }
}
impl<S: RealScalar> CdfRequirements<S> {
    /// Returns the maximum number of observations per batch.
    pub const fn samples(self) -> usize {
        self.raw.samples
    }
    /// Independent batch count.
    pub const fn batches(self) -> usize {
        self.raw.batches
    }
    /// Returns the number of queries per batch.
    pub const fn queries(self) -> usize {
        self.raw.queries
    }
    /// Resolved worker count.
    pub const fn workers(self) -> usize {
        self.raw.workers
    }
}
impl<S: RealScalar> BinnedRequirements<S> {
    /// Returns the maximum number of observations per batch.
    pub const fn samples(self) -> usize {
        self.raw.samples
    }
    /// Independent batch count.
    pub const fn batches(self) -> usize {
        self.raw.batches
    }
    /// Returns the number of bins per batch.
    pub const fn bins(self) -> usize {
        self.raw.bins
    }
    /// Resolved worker count.
    pub const fn workers(self) -> usize {
        self.raw.workers
    }
}
impl<S: RealScalar> KdeRequirements<S> {
    /// Returns the total number of query points.
    pub const fn queries(self) -> usize {
        self.raw.queries
    }
    /// Returns the number of coordinates per point.
    pub const fn dimensions(self) -> usize {
        self.raw.dimensions
    }
    /// Resolved worker count.
    pub const fn workers(self) -> usize {
        self.raw.workers
    }
}
/// Fits an isotonic regression to every lane of `view`.
///
/// Each lane is fitted as in [`isotonic_regression_with_options`], by the
/// pool-adjacent-violators algorithm in O(n) time. Observations are unweighted
/// or carry masses. `output` receives the fitted values in the layout of
/// `view`, the blocks of each lane, and one record per lane. Lanes are
/// independent and can run in parallel with the same result.
///
/// # Errors
///
/// - [`DimensionMismatch`](scioxide_core::SciOxideErrorKind::DimensionMismatch)
///   if `workspace` does not hold the storage for this call.
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if `view`
///   has a coordinate axis, an output slice has the wrong length, a mass is
///   negative, a nonempty lane has zero total mass, or an observation is
///   missing and the missing-value policy rejects it.
/// - [`NonFinite`](scioxide_core::SciOxideErrorKind::NonFinite) if a mass is NaN
///   or infinite, or the non-finite policy rejects a non-finite observation.
pub fn isotonic_regression_axes_into<S: RealScalar, M: IsotonicWeights>(
    view: BatchView<'_, S, M>,
    direction: IsotonicDirection,
    options: StatisticsOptions,
    output: IsotonicBatchOutput<'_, S>,
    workspace: &mut IsotonicWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = IsotonicWorkspace::workspace_requirements(view, options, context)?;
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.bytes],
            [workspace.retained_bytes],
        ));
    }
    crate::isotonic_axis::isotonic_regression_axes_into(
        crate::OrderData::from_batch(view),
        crate::IsotonicPolicy {
            direction,
            data: options.order(view.axis()),
        },
        context,
        &mut workspace.raw,
        output,
    )
}
/// Fits an isotonic regression to one slice into `output` and `blocks`.
///
/// The fit is defined in [`IsotonicResult`](crate::IsotonicResult) and
/// computed by the pool-adjacent-violators algorithm in O(n) time.
/// `output.len()` must equal `values.len()`, and `blocks` needs at least as
/// many slots. Only the first
/// [`block_count`](crate::IsotonicDiagnostics::block_count) blocks are
/// written. Reserve `workspace` with
/// [`IsotonicWorkspace::scalar_requirements`].
///
/// # Errors
///
/// - [`DimensionMismatch`](scioxide_core::SciOxideErrorKind::DimensionMismatch)
///   if `workspace` was never reserved.
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
///   `output`, `blocks` or the weights have the wrong length, `workspace` is
///   too small, a weight is negative, or a nonempty input has zero total
///   weight.
/// - [`NonFinite`](scioxide_core::SciOxideErrorKind::NonFinite) if a weight is
///   NaN or infinite, or the non-finite policy of `options` rejects a
///   non-finite observation.
/// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if
///   `workspace` holds more than `options.max_bytes()`.
#[allow(clippy::too_many_arguments)]
pub fn isotonic_regression_into<'a, S: RealScalar, W: crate::BatchWeights<'a, S>>(
    values: &[S],
    weights: W,
    direction: IsotonicDirection,
    options: StatisticsOptions,
    output: &mut [S],
    blocks: &mut [crate::IsotonicBlock<S>],
    workspace: &mut IsotonicWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<crate::IsotonicDiagnostics<S>, SciOxideError>
where
    W::Model: IsotonicWeights,
{
    let data = crate::weights::batch_sealed::Sealed::data(weights);
    let weights = match data {
        crate::weights::WeightData::Mass(w) => Some(w),
        _ => None,
    };
    limit_bytes(workspace.retained_bytes, options.max_bytes())?;
    let worker = workspace
        .raw
        .workers
        .first_mut()
        .ok_or_else(|| SciOxideError::dimension_mismatch([1], [0]))?;
    context.check()?;
    crate::isotonic::isotonic_regression_into(
        values,
        weights,
        crate::RawIsotonicOptions {
            direction,
            non_finite: options.non_finite_policy(),
        },
        worker,
        output,
        blocks,
    )
}
/// Fits an isotonic regression to `values` and returns the fit.
///
/// The fit `f̂` minimizes `Σ wᵢ (yᵢ − fᵢ)²` subject to the order constraint
/// `direction`. It is computed by the pool-adjacent-violators algorithm in
/// O(n) time and memory; see [`IsotonicResult`](crate::IsotonicResult).
/// Observations are unweighted or carry masses.
///
/// # Errors
///
/// Fails as [`isotonic_regression_into`], and with
/// [`AllocationFailure`](scioxide_core::SciOxideErrorKind::AllocationFailure) or
/// [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
/// storage cannot be allocated within `options.max_bytes()`.
///
/// # References
///
/// R. E. Barlow, D. J. Bartholomew, J. M. Bremner and H. D. Brunk,
/// *Statistical Inference under Order Restrictions*, Wiley, 1972.
pub fn isotonic_regression_with_options<'a, S: RealScalar, W: crate::BatchWeights<'a, S>>(
    values: &[S],
    weights: W,
    direction: IsotonicDirection,
    options: StatisticsOptions,
) -> Result<crate::IsotonicResult<S>, SciOxideError>
where
    W::Model: IsotonicWeights,
{
    let shape = [values.len()];
    let strides = [1];
    let view = BatchView::new(
        values,
        scioxide_core::StridedLayout::new(&shape, &strides)?,
        crate::Axis::Index(0),
    )?;
    let req = IsotonicWorkspace::workspace_requirements(
        view,
        options,
        ExecutionContext::sequential_native(),
    )?;
    let mut workspace = IsotonicWorkspace::workspace(&req)?;
    let mut fitted = scioxide_core::backend::try_filled(
        values.len(),
        S::ZERO,
        scioxide_core::quantity!("isotonic_output"),
    )?;
    let mut blocks = scioxide_core::backend::try_filled(
        values.len(),
        crate::IsotonicBlock::default(),
        scioxide_core::quantity!("isotonic_blocks"),
    )?;
    let diagnostics = isotonic_regression_into(
        values,
        weights,
        direction,
        options,
        &mut fitted,
        &mut blocks,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    blocks.truncate(diagnostics.block_count);
    Ok(crate::IsotonicResult::from_fit(fitted, blocks, diagnostics))
}

/// Probability returned by [`cdf_with_options`] and [`cdf_into`].
///
/// The weight model `M` of the observations fixes the quantile rules that
/// [`Estimated`](Self::Estimated) accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CdfMethod<M: crate::WeightModel = Unweighted> {
    /// The empirical CDF `P(X ≤ x)`, as in
    /// [`EmpiricalDistribution::cdf`](crate::EmpiricalDistribution::cdf).
    Empirical,
    /// The empirical survival function `P(X > x)`, as in
    /// [`EmpiricalDistribution::survival`](crate::EmpiricalDistribution::survival).
    Survival,
    /// The CDF estimate for a quantile definition, as in
    /// [`EmpiricalDistribution::estimated_cdf`](crate::EmpiricalDistribution::estimated_cdf).
    Estimated(QuantileRule<M>),
}
impl<M: crate::WeightModel> CdfMethod<M> {
    fn kernel(self) -> crate::RawCdfMethod {
        match self {
            Self::Empirical => crate::RawCdfMethod::Empirical,
            Self::Survival => crate::RawCdfMethod::Survival,
            Self::Estimated(rule) => crate::RawCdfMethod::Estimated(rule.method),
        }
    }
}
impl<S: RealScalar> CdfWorkspace<S> {
    /// Returns the requirements for [`cdf_into`] on `view` with `queries`
    /// query points.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if `view`
    ///   has a coordinate axis.
    /// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
    ///   storage size overflows `usize`.
    /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
    ///   storage exceeds `options.max_bytes()`.
    pub fn workspace_requirements<M: EmpiricalWeights>(
        view: BatchView<'_, S, M>,
        queries: usize,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<CdfRequirements<S>, SciOxideError> {
        CdfRequirements::new(
            crate::CdfWorkspaceRequirements::new(
                crate::ObservationData::from_batch(view),
                queries,
                options,
                context,
            )?,
            options,
        )
    }
}
/// Evaluates the empirical CDF, survival function or CDF estimate of every
/// batch of `view` at each of `queries`.
///
/// `output[b · m + j]` receives the value for batch `b` at `queries[j]`, where
/// `m = queries.len()`. The definitions are those of [`CdfMethod`]. Each
/// batch is sorted in O(n log n) time. The queries are then answered by binary
/// search, or with masses by one merged pass over the sorted queries.
///
/// # Errors
///
/// - [`DimensionMismatch`](scioxide_core::SciOxideErrorKind::DimensionMismatch)
///   if `workspace` does not hold the storage for this call.
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if `view`
///   has a coordinate axis, `output.len()` is not `m` times the number of
///   batches, a query is NaN, a mass is negative or non-finite, a batch has
///   no observation with positive weight, or an observation is missing or
///   non-finite and the policy rejects it.
/// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
///   total frequency count of a batch overflows `usize`.
pub fn cdf_into<S: RealScalar, M: EmpiricalWeights>(
    view: BatchView<'_, S, M>,
    queries: &[S],
    method: CdfMethod<M>,
    options: StatisticsOptions,
    output: &mut [S],
    workspace: &mut CdfWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = CdfWorkspace::workspace_requirements(view, queries.len(), options, context)?;
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.bytes],
            [workspace.retained_bytes],
        ));
    }
    crate::cdf::cdf_into(
        crate::ObservationData::from_batch(view),
        queries,
        method.kernel(),
        options,
        context,
        &mut workspace.raw,
        output,
    )
}
/// Computes the result of [`cdf_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`cdf_with_options`] for the fixed default settings.
pub fn cdf<S: RealScalar, M: EmpiricalWeights>(
    view: BatchView<'_, S, M>,
    queries: &[S],
    method: CdfMethod<M>,
) -> Result<Vec<S>, SciOxideError> {
    cdf_with_options(view, queries, method, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("view"),
        )
    })
}
/// Evaluates the empirical CDF, survival function or CDF estimate of every
/// batch of `view` at each of `queries`, and returns the values in
/// batch-major order.
///
/// This runs [`cdf_into`] sequentially with a new workspace.
///
/// # Errors
///
/// Fails as [`cdf_into`] (except for the workspace and output conditions), and
/// with [`AllocationFailure`](scioxide_core::SciOxideErrorKind::AllocationFailure)
/// or [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
/// storage cannot be allocated within `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// use scioxide_core::StridedLayout;
/// use scioxide_stats::{Axis, BatchView, CdfMethod, StatisticsOptions, cdf_with_options};
///
/// let values = [3.0_f64, 1.0, 4.0, 2.0];
/// let view = BatchView::new(&values, StridedLayout::new(&[4], &[1])?, Axis::Index(0))?;
/// let p = cdf_with_options(view, &[2.0, 5.0], CdfMethod::Empirical, StatisticsOptions::new())?;
/// // Two of the four observations are at most 2, and all are at most 5.
/// assert!((p[0] - 0.5).abs() < 1e-15);
/// assert!((p[1] - 1.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn cdf_with_options<S: RealScalar, M: EmpiricalWeights>(
    view: BatchView<'_, S, M>,
    queries: &[S],
    method: CdfMethod<M>,
    options: StatisticsOptions,
) -> Result<Vec<S>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = CdfWorkspace::workspace_requirements(view, queries.len(), options, context)?;
    let mut ws = CdfWorkspace::workspace(&req)?;
    let mut output = scioxide_core::backend::try_filled(
        req.raw.batches * req.raw.queries,
        S::ZERO,
        scioxide_core::quantity!("cdf_output"),
    )?;
    cdf_into(
        view,
        queries,
        method,
        options,
        &mut output,
        &mut ws,
        context,
    )?;
    Ok(output)
}
impl<S: RealScalar> BinnedWorkspace<S> {
    /// Returns the requirements for [`binned_statistics_into`] with `plan`,
    /// `view` and `statistic`. A workspace serves only the statistic it was
    /// reserved for.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
    ///   number of coordinates of `view` differs from the dimension of `plan`.
    /// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
    ///   storage size overflows `usize`.
    /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
    ///   storage exceeds `options.max_bytes()`.
    pub fn workspace_requirements<M: EmpiricalWeights>(
        plan: &BinPlan<S>,
        view: BatchView<'_, S, M>,
        statistic: BinnedStatistic,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<BinnedRequirements<S>, SciOxideError> {
        BinnedRequirements::new(
            crate::BinnedWorkspaceRequirements::new(
                plan,
                crate::ObservationData::from_batch(view),
                statistic,
                options,
                context,
            )?,
            options,
        )
    }
}
/// Computes `statistic` in every bin of `plan` for every batch of `view`.
///
/// The observations of `view` are the points that are binned. `values` holds
/// the quantity summarized by [`BinnedStatistic::Sum`](crate::BinnedStatistic::Sum),
/// `Mean`, `StandardDeviation`, `Minimum`, `Maximum` and `Median`, one per
/// observation in batch-major, sample-minor order. It may be `None` for the
/// statistics that only count or weigh points. `output` receives the bins of
/// each batch contiguously, in row-major bin order, and `diagnostics` one
/// record per batch. Only the median sorts the observations of each bin.
///
/// # Errors
///
/// - [`DimensionMismatch`](scioxide_core::SciOxideErrorKind::DimensionMismatch)
///   if `workspace` does not hold the storage for this call.
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if an
///   output slice or `values` has the wrong length, `values` is `None` for a
///   statistic that needs it, `statistic` is
///   [`CumulativeFrequency`](crate::BinnedStatistic::CumulativeFrequency) on a
///   grid of more than one dimension, the workspace was reserved for another
///   statistic, a weight is invalid, a point or value is missing or
///   non-finite and the policy rejects it, or a batch has zero normalizing
///   mass for `RelativeFrequency` or `Density`.
/// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if a bin
///   count overflows `usize`.
#[allow(clippy::too_many_arguments)]
pub fn binned_statistics_into<S: RealScalar, M: EmpiricalWeights>(
    plan: &BinPlan<S>,
    view: BatchView<'_, S, M>,
    values: Option<&[S]>,
    statistic: BinnedStatistic,
    options: StatisticsOptions,
    output: &mut [BinEstimate<S>],
    diagnostics: &mut [BinnedDiagnostics<S>],
    workspace: &mut BinnedWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = BinnedWorkspace::workspace_requirements(plan, view, statistic, options, context)?;
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.bytes],
            [workspace.retained_bytes],
        ));
    }
    crate::binned::binned_statistics_into(
        plan,
        crate::ObservationData::from_batch(view),
        values,
        statistic,
        options,
        context,
        &mut workspace.raw,
        output,
        diagnostics,
    )
}
/// Computes `statistic` in every bin of `plan` for every batch of `view`, and
/// returns the bins and per-batch totals.
///
/// This runs [`binned_statistics_into`] sequentially with a new workspace.
///
/// # Errors
///
/// Fails as [`binned_statistics_into`] (except for the workspace and output
/// conditions), and with
/// [`AllocationFailure`](scioxide_core::SciOxideErrorKind::AllocationFailure) or
/// [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
/// storage cannot be allocated within `options.max_bytes()`.
pub fn binned_statistics<S: RealScalar, M: EmpiricalWeights>(
    plan: &BinPlan<S>,
    view: BatchView<'_, S, M>,
    values: Option<&[S]>,
    statistic: BinnedStatistic,
    options: StatisticsOptions,
) -> Result<BinnedResult<S>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = BinnedWorkspace::workspace_requirements(plan, view, statistic, options, context)?;
    let mut workspace = BinnedWorkspace::workspace(&req)?;
    let mut estimates = scioxide_core::backend::try_filled(
        req.raw.bins * req.raw.batches,
        BinEstimate::default(),
        scioxide_core::quantity!("binned_output"),
    )?;
    let mut diagnostics = scioxide_core::backend::try_filled(
        req.raw.batches,
        BinnedDiagnostics::default(),
        scioxide_core::quantity!("binned_diagnostics"),
    )?;
    binned_statistics_into(
        plan,
        view,
        values,
        statistic,
        options,
        &mut estimates,
        &mut diagnostics,
        &mut workspace,
        context,
    )?;
    Ok(BinnedResult {
        bins: estimates,
        diagnostics,
    })
}

/// Gaussian kernel density estimate of one batch of scalar or vector
/// observations.
///
/// For observations `xᵢ ∈ ℝᵈ` with weights `wᵢ` and total weight `W`, the
/// density estimate is
///
/// ```text
/// f̂(x) = (1/W) Σᵢ wᵢ φ_H(x − xᵢ),
/// ```
///
/// where `φ_H` is the `d`-variate normal density with mean zero and kernel
/// covariance `H`, chosen by [`KdeBandwidth`](crate::KdeBandwidth). With a
/// support box in [`KdeOptions`], each kernel is truncated to the box and
/// renormalized. Each density evaluation sums over all `n` centers.
///
/// # References
///
/// - D. W. Scott, *Multivariate Density Estimation*, Wiley, 1992.
/// - B. W. Silverman, *Density Estimation for Statistics and Data Analysis*,
///   Chapman & Hall, 1986.
pub struct GaussianKde<S: RealScalar> {
    raw: crate::RawGaussianKde<S>,
}
impl<S: RealScalar> GaussianKde<S> {
    /// Builds the density estimate of batch `batch` of `view`.
    ///
    /// The observations with positive weight are copied as kernel centers,
    /// and the kernel covariance is computed from them or taken from
    /// `options`.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `batch` is out of range, a weight is invalid, an observation is
    ///   missing or non-finite under a rejecting policy, no observation has
    ///   positive weight, an observation lies outside the support box, the
    ///   support box is invalid or used with a correlated kernel, the
    ///   bandwidth is invalid, or a fitted bandwidth has no residual degrees of
    ///   freedom.
    /// - Fails if the kernel covariance is not positive definite.
    pub fn build<M: EmpiricalWeights>(
        view: BatchView<'_, S, M>,
        batch: usize,
        options: KdeOptions<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        crate::RawGaussianKde::prepare(
            crate::ObservationData::from_batch(view),
            batch,
            options,
            context,
        )
        .map(|raw| Self { raw })
    }
    /// Returns the dimension `d` of the observations.
    pub const fn dimensions(&self) -> usize {
        self.raw.dimensions()
    }
    /// Returns the number of kernel centers: the observations with positive
    /// weight, without replication.
    pub fn observations(&self) -> usize {
        self.raw.observations()
    }
    /// Returns the effective sample size: the total count without weights or
    /// with frequency weights, and Kish's `(Σ wᵢ)² / Σ wᵢ²` with masses.
    pub const fn effective_size(&self) -> f64 {
        self.raw.effective_size()
    }
    /// Returns the bandwidth factor `f`, or `None` for an explicit bandwidth.
    pub const fn bandwidth_factor(&self) -> Option<S> {
        self.raw.bandwidth_factor()
    }
    /// Returns the density `f̂(point)`.
    ///
    /// Allocates d scratch values. Outside the support box the density is 0.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `point` does not have `d` coordinates or a coordinate is not finite.
    /// - `AllocationFailure` if query scratch cannot be allocated.
    /// - Fails if the density overflows `S`.
    pub fn pdf(&self, point: &[S]) -> Result<S, SciOxideError> {
        let mut scratch = scioxide_core::backend::try_filled(
            self.dimensions(),
            0.0,
            scioxide_core::quantity!("kde_scratch"),
        )?;
        self.raw.pdf(point, &mut scratch)
    }
    /// Returns the log density `ln f̂(point)`, evaluated directly rather than
    /// as the logarithm of [`pdf`](Self::pdf).
    ///
    /// Outside the support box the result is negative infinity.
    ///
    /// # Errors
    ///
    /// Fails as [`pdf`](Self::pdf).
    pub fn log_pdf(&self, point: &[S]) -> Result<S, SciOxideError> {
        let mut scratch = scioxide_core::backend::try_filled(
            self.dimensions(),
            0.0,
            scioxide_core::quantity!("kde_scratch"),
        )?;
        self.raw.log_pdf(point, &mut scratch)
    }
    /// Returns the probability `∫ f̂(x) dx` over the box
    /// `[lower₁, upper₁] × … × [lower_d, upper_d]`, from differences of the
    /// normal CDF.
    ///
    /// Bounds may be infinite. The kernel covariance must be diagonal.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if a
    ///   bound is NaN, a lower bound exceeds its upper bound, the bounds do not
    ///   have `d` coordinates, or the kernel covariance is not diagonal.
    pub fn integrate_box(&self, lower: &[S], upper: &[S]) -> Result<S, SciOxideError> {
        self.raw.integrate_box(lower, upper)
    }
    /// Returns `∫ f̂(x) ĝ(x) dx` for the density estimate `ĝ` of `other`, in
    /// closed form over all pairs of centers.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
    ///   dimensions differ or either estimate has a support box.
    /// - [`AllocationFailure`](scioxide_core::SciOxideErrorKind::AllocationFailure)
    ///   if scratch storage cannot be allocated.
    pub fn integrate_kde(&self, other: &Self) -> Result<S, SciOxideError> {
        self.raw.integrate_kde(&other.raw)
    }
    /// Returns `∫ f̂(x) φ(x) dx` for the normal density `φ` with mean `mean`
    /// and row-major `d × d` covariance `covariance`, in closed form.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
    ///   estimate has a support box, `mean` or `covariance` has the wrong
    ///   length, `mean` is not finite, or `covariance` is not symmetric.
    /// - Fails if `covariance` is not positive definite.
    pub fn integrate_gaussian(&self, mean: &[S], covariance: &[S]) -> Result<S, SciOxideError> {
        self.raw.integrate_gaussian(mean, covariance)
    }
}
impl<S: RealScalar> KdeWorkspace<S> {
    /// Returns the requirements for [`kde_into`] with `kde` and the query
    /// points `queries`, which carry no weights.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
    ///   query points do not have the dimension of `kde`.
    /// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
    ///   storage size overflows `usize`.
    /// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
    ///   storage exceeds `options.max_bytes()`.
    pub fn workspace_requirements(
        kde: &GaussianKde<S>,
        queries: BatchView<'_, S>,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<KdeRequirements<S>, SciOxideError> {
        KdeRequirements::new(
            crate::KdeWorkspaceRequirements::new(
                &kde.raw,
                crate::ObservationData::from_batch(queries),
                options,
                context,
            )?,
            options,
        )
    }
}
/// Evaluates the density or log density of `kde` at every query point.
///
/// `output` receives one value per query point, in batch-major order. `scale`
/// selects the density or its logarithm. Query points are never skipped.
///
/// # Errors
///
/// - [`DimensionMismatch`](scioxide_core::SciOxideErrorKind::DimensionMismatch)
///   if `workspace` does not hold the storage for this call.
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
///   `output` has the wrong length, or a query point is missing or not
///   finite.
pub fn kde_into<S: RealScalar>(
    kde: &GaussianKde<S>,
    queries: BatchView<'_, S>,
    scale: crate::KdeScale,
    options: StatisticsOptions,
    output: &mut [S],
    workspace: &mut KdeWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = KdeWorkspace::workspace_requirements(kde, queries, options, context)?;
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.bytes],
            [workspace.retained_bytes],
        ));
    }
    crate::kde::kde_into(
        &kde.raw,
        crate::ObservationData::from_batch(queries),
        scale,
        options,
        context,
        &mut workspace.raw,
        output,
    )
}

/// Computes the mean direction and mean resultant length of every batch of
/// vector observations.
///
/// For vectors `xᵢ ∈ ℝᵈ` with weights `wᵢ` and total weight `W`, the mean
/// resultant vector is `R̄ = (1/W) Σᵢ wᵢ uᵢ` with unit vectors
/// `uᵢ = xᵢ / ‖xᵢ‖`. The mean direction is `R̄ / ‖R̄‖`, defined when
/// `‖R̄‖ > 8ε`, and is all zeros otherwise. `normalization` selects whether
/// each vector is normalized or must already have unit norm.
/// `directions` receives `d` components per batch, in batch-major order, and
/// `reports` the resultant length of each batch. Each batch costs O(n d) time.
///
/// # Errors
///
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
///   `workspace` is too small, an output slice has the wrong length, the
///   unit-norm tolerance is negative or not finite, a vector is zero or not of
///   unit norm when unit norm is required, a weight is invalid, a batch has no
///   observation with positive weight, or an observation is missing or
///   non-finite under a rejecting policy.
///
/// # References
///
/// K. V. Mardia and P. E. Jupp, *Directional Statistics*, Wiley, 2000.
#[allow(clippy::too_many_arguments)]
pub fn directional_statistics_into<S: RealScalar, M: EmpiricalWeights>(
    view: BatchView<'_, S, M>,
    normalization: crate::DirectionNormalization<S>,
    options: StatisticsOptions,
    directions: &mut [S],
    reports: &mut [crate::DirectionalSummary<S>],
    workspace: &mut crate::DirectionalWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = crate::DirectionalWorkspace::workspace_requirements(view, options, context)?;
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient directional storage",
        ));
    }
    crate::circular::directional_statistics_into(
        crate::ObservationData::from_batch(view),
        normalization,
        options,
        context,
        workspace,
        directions,
        reports,
    )
}

/// Computes the result of [`frequency_covariance_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`frequency_covariance_with_options`] for the fixed default settings.
pub fn frequency_covariance<
    'a,
    T: scioxide_core::NumericScalar,
    W: crate::BatchWeights<'a, T::Real, Model = crate::Frequency>,
>(
    data: scioxide_linalg::MatrixView<'_, T>,
    weights: W,
) -> Result<(Vec<T>, scioxide_linalg::Matrix<T>), SciOxideError> {
    frequency_covariance_with_options(data, weights, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("data"),
        )
    })
}
/// Computes the mean vector and the unbiased covariance matrix of real or
/// complex data with integer frequency weights.
///
/// Row `i` of `data`, with entries `x_ij`, is an observation repeated `cᵢ`
/// times. With `N = Σ cᵢ ≥ 2`, the results are
///
/// ```text
/// x̄_j = (1/N) Σᵢ cᵢ x_ij,
/// C_jk = (1/(N − 1)) Σᵢ cᵢ conj(x_ij − x̄_j) (x_ik − x̄_k).
/// ```
///
/// Only [`Frequency`](crate::Frequency) weights are accepted.
/// `options.max_bytes()` bounds the storage for the counts, means and
/// covariance. The rank policy of `options` does not apply.
///
/// # Errors
///
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if `data`
///   is empty, the number of counts differs from the number of rows, a count
///   is not exactly representable in the real type, or `N` is less than 2 or
///   exceeds 2^(mantissa digits).
/// - [`ResourceLimit`](scioxide_core::SciOxideErrorKind::ResourceLimit) if the
///   storage exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{Frequency, MultivariateOptions, Weights, frequency_covariance_with_options};
///
/// let data = MatrixView::from_slice(2, 1, &[1.0_f64, 3.0])?;
/// let (mean, covariance) = frequency_covariance_with_options(
///     data,
///     Weights::<_, Frequency>::new(&[1u64, 1]),
///     MultivariateOptions::new(),
/// )?;
/// // Mean 2; sample variance ((1 − 2)² + (3 − 2)²) / (2 − 1) = 2.
/// assert_eq!(mean, [2.0]);
/// assert_eq!(covariance.as_slice(), &[2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// ```compile_fail
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{frequency_covariance_with_options, Analytic, MultivariateOptions, Weights};
/// let data = MatrixView::from_slice(2, 1, &[1.0_f64, 3.0]).unwrap();
/// frequency_covariance_with_options(data, Weights::<_, Analytic>::new(&[1.0, 1.0]), MultivariateOptions::new());
/// ```
pub fn frequency_covariance_with_options<
    'a,
    T: scioxide_core::NumericScalar,
    W: crate::BatchWeights<'a, T::Real, Model = crate::Frequency>,
>(
    data: scioxide_linalg::MatrixView<'_, T>,
    weights: W,
    options: crate::MultivariateOptions<T::Real>,
) -> Result<(Vec<T>, scioxide_linalg::Matrix<T>), SciOxideError> {
    let max_bytes = options.max_bytes();
    let crate::weights::WeightData::Frequency(counts) =
        crate::weights::batch_sealed::Sealed::data(weights)
    else {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid frequency capability",
        ));
    };
    let bytes =
        crate::estimation::storage_bytes(&[(counts.len(), core::mem::size_of::<T::Real>())])?;
    limit_bytes(bytes, max_bytes)?;
    let mut real = scioxide_core::backend::try_with_capacity(
        counts.len(),
        scioxide_core::quantity!("frequency_covariance_weights"),
    )?;
    for &count in counts {
        real.push(crate::weights::frequency_scalar(count)?);
    }
    crate::multivariate::frequency_covariance_with_options(data, &real, max_bytes - bytes)
}

/// Computes the result of [`isotonic_regression_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the configured form's errors for these defaults.
pub fn isotonic_regression<S: RealScalar>(
    values: &[S],
) -> Result<crate::IsotonicResult<S>, SciOxideError> {
    isotonic_regression_with_options(
        values,
        crate::Unweighted,
        IsotonicDirection::Increasing,
        Default::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &[
                "options",
                "weights",
                "left_weights",
                "right_weights",
                "direction",
                "ties",
            ],
            scioxide_core::argument!("values"),
        )
    })
}

impl<S: RealScalar> GaussianKde<S> {
    /// Estimates a univariate density using Scott's bandwidth and full covariance.
    ///
    /// Uses [`Self::with_options`] and the [one-shot defaults](crate#one-shot-forms).
    ///
    /// # Errors
    ///
    /// Rejects non-finite samples, insufficient observations or singular covariance.
    pub fn new(values: &[S]) -> Result<Self, SciOxideError> {
        let shape = [values.len()];
        let strides = [1];
        let view = BatchView::new(
            values,
            scioxide_core::StridedLayout::new(&shape, &strides)?,
            crate::Axis::Index(0),
        )?;
        Self::with_options(view, 0, KdeOptions::default()).map_err(|error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["view", "batch", "options"],
                scioxide_core::argument!("values"),
            )
        })
    }
    /// Prepares a density estimate with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::build`].
    pub fn with_options<M: EmpiricalWeights>(
        view: BatchView<'_, S, M>,
        batch: usize,
        options: KdeOptions<'_, S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(view, batch, options, ExecutionContext::sequential_native())
    }
    /// Evaluates row-major query points, reusing one scratch vector.
    ///
    /// For dimension d, costs O(n d²) per point, with the accuracy of [`Self::pdf`].
    ///
    /// # Errors
    ///
    /// Rejects a length that is not a multiple of d and propagates density and allocation errors.
    pub fn evaluate(&self, points: &[S]) -> Result<Vec<S>, SciOxideError> {
        let d = self.dimensions();
        if !points.len().is_multiple_of(d) {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("points"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "incomplete density query point",
            ));
        }
        let mut scratch =
            scioxide_core::backend::try_filled(d, 0.0, scioxide_core::quantity!("kde_scratch"))?;
        let mut output = scioxide_core::backend::try_with_capacity(
            points.len() / d,
            scioxide_core::quantity!("kde_output"),
        )?;
        for point in points.chunks_exact(d) {
            output.push(self.raw.pdf(point, &mut scratch)?);
        }
        Ok(output)
    }
}
