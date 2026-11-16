//! Moments and covariance of every batch of a strided array.
use crate::arithmetic::Compensated;
use crate::error_helpers::limit_bytes;
use crate::statistics_workspace::{admit, admit_at_origin};
use crate::weights::WeightData;
use crate::{
    Analytic, BatchView, CovarianceSnapshot, Frequency, MergePolicy, MomentsSnapshot,
    OnlineCovariance, OnlineMoments, OnlineWeightedCovariance, OnlineWeightedMoments, Reliability,
    StatisticsOptions, TrimLimits, WeightModel, WeightedCovarianceSnapshot,
    WeightedMomentsSnapshot,
};
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{PartitionSlot, try_filled};
use scioxide_core::{
    Capacity, ExecutionContext, ExecutionPolicy, RealScalar, Scaled, ScaledAccumulator,
    SciOxideError,
};

/// Weight models accepted by weighted moments: [`Frequency`], [`Reliability`] and [`Analytic`].
pub trait MomentWeights: WeightModel {
    /// Weight model reported by the accumulators.
    #[doc(hidden)]
    const SEMANTICS: crate::WeightSemantics;
}
impl MomentWeights for Frequency {
    const SEMANTICS: crate::WeightSemantics = crate::WeightSemantics::Frequency;
}
impl MomentWeights for Reliability {
    const SEMANTICS: crate::WeightSemantics = crate::WeightSemantics::Reliability;
}
impl MomentWeights for Analytic {
    const SEMANTICS: crate::WeightSemantics = crate::WeightSemantics::Analytic;
}

#[derive(Clone, Copy)]
enum State<S: RealScalar> {
    Moments(OnlineMoments<S>),
    Covariance(OnlineCovariance<S>),
    WeightedMoments(crate::weighted::WeightedMoments<S>),
    WeightedCovariance(crate::weighted_covariance::WeightedCovariance<S>),
    Power {
        count: usize,
        sum: Compensated<Scaled<f64>>,
    },
}
impl<S: RealScalar> State<S> {
    fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        match (self, other) {
            (Self::Moments(a), Self::Moments(b)) => a.merge(b),
            (Self::Covariance(a), Self::Covariance(b)) => a.merge(b),
            (Self::WeightedMoments(a), Self::WeightedMoments(b)) => a.merge(b),
            (Self::WeightedCovariance(a), Self::WeightedCovariance(b)) => a.merge(b),
            (Self::Power { count, sum }, Self::Power { count: n, sum: s }) => {
                *count = count.checked_add(*n).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("moment_count"))
                })?;
                *sum = sum.merged(*s)?;
                Ok(())
            }
            _ => Err(crate::error_helpers::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "inconsistent reduction state",
            )),
        }
    }
}
/// Storage sizes that a moment reduction needs.
///
/// Returned by [`ReductionWorkspace::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct ReductionRequirements {
    batches: usize,
    workers: usize,
    max_bytes: usize,
}
impl ReductionRequirements {
    /// Returns the number of batches, one output entry each.
    pub const fn batches(self) -> usize {
        self.batches
    }
    /// Returns the number of partitions per batch in the execution plan.
    pub const fn workers(self) -> usize {
        self.workers
    }
    /// Returns the workspace size in bytes, excluding the output slice.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[
            (self.workers, size_of::<State<S>>()),
            (self.batches, size_of::<State<S>>()),
            (self.batches, size_of::<Option<S>>()),
            (self.batches, size_of::<ScaledAccumulator<f64>>()),
        ])
    }
}
/// Reusable storage for the batch moment reductions.
///
/// It holds the partition states and per-batch results of
/// [`reduce_moments_into`], [`reduce_trimmed_moments_into`],
/// [`reduce_covariance_into`], [`reduce_weighted_moments_into`],
/// [`reduce_weighted_covariance_into`] and [`reduce_moment_into`]. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct ReductionWorkspace<S: RealScalar> {
    workers: Vec<State<S>>,
    stage: Vec<State<S>>,
    values: Vec<Option<S>>,
    centers: Vec<ScaledAccumulator<f64>>,
    reserved: ReductionRequirements,
}
impl<S: RealScalar> Default for ReductionWorkspace<S> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            stage: Vec::new(),
            values: Vec::new(),
            centers: Vec::new(),
            reserved: ReductionRequirements {
                batches: 0,
                workers: 0,
                max_bytes: usize::MAX,
            },
        }
    }
}
fn plan(
    options: StatisticsOptions,
    samples: usize,
    context: ExecutionContext<'_>,
) -> scioxide_core::ExecutionPlan {
    context.plan(
        if options.merge_policy() == MergePolicy::InputOrder {
            ExecutionPolicy::Sequential
        } else {
            options.execution_policy()
        },
        samples,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: core::num::NonZeroUsize::MIN.saturating_add(1023),
            worker_limit: core::num::NonZeroUsize::MIN.saturating_add(63),
        },
    )
}
impl<S: RealScalar> ReductionWorkspace<S> {
    /// Returns the storage needed to reduce `view` with `options` under `context`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `view` has vector observations.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds
    ///   [`StatisticsOptions::max_bytes`].
    pub fn workspace_requirements<M: WeightModel>(
        view: BatchView<'_, S, M>,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<ReductionRequirements, SciOxideError> {
        if view.dimensions() != 1 {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "moment observations must be scalar",
            ));
        }
        let req = ReductionRequirements {
            batches: view.batches(),
            workers: if view.batches() == 0 {
                0
            } else {
                plan(options, view.samples(), context).worker_count()
            },
            max_bytes: options.max_bytes(),
        };
        limit_bytes(req.required_bytes::<S>()?, req.max_bytes)?;
        Ok(req)
    }
    /// Returns whether the reserved storage is large enough for `req`.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit` if the storage already reserved exceeds the byte
    /// limit of `req`.
    pub fn check(&self, req: &ReductionRequirements) -> Result<Capacity, SciOxideError> {
        limit_bytes(self.reserved.required_bytes::<S>()?, req.max_bytes)?;
        Ok(
            if self.reserved.batches >= req.batches && self.reserved.workers >= req.workers {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the storage to satisfy `req`, if needed.
    ///
    /// # Errors
    ///
    /// Returns `ResourceLimit`, `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve(&mut self, req: &ReductionRequirements) -> Result<(), SciOxideError> {
        if self.check(req)? == Capacity::Sufficient {
            return Ok(());
        }
        limit_bytes(req.required_bytes::<S>()?, req.max_bytes)?;
        let initial = State::Moments(OnlineMoments::new(crate::MomentOrder::Second));
        *self = Self {
            workers: try_filled(
                req.workers,
                initial,
                scioxide_core::quantity!("reduction_workers"),
            )?,
            stage: try_filled(
                req.batches,
                initial,
                scioxide_core::quantity!("reduction_stage"),
            )?,
            values: try_filled(
                req.batches,
                None,
                scioxide_core::quantity!("reduction_values"),
            )?,
            centers: try_filled(
                req.batches,
                ScaledAccumulator::ZERO,
                scioxide_core::quantity!("reduction_centers"),
            )?,
            reserved: *req,
        };
        Ok(())
    }
    /// Creates a workspace with storage for `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::try_reserve`].
    pub fn workspace(req: &ReductionRequirements) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
}
fn run<S: RealScalar, M: WeightModel>(
    view: BatchView<'_, S, M>,
    options: StatisticsOptions,
    initial: State<S>,
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
    observe: impl Fn(&mut State<S>, usize, usize, usize) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.batches, req.workers],
            [workspace.reserved.batches, workspace.reserved.workers],
        ));
    }
    let plan = plan(options, view.samples(), context);
    for batch in 0..view.batches() {
        context.check()?;
        let origin = view.batch_origin(batch);
        if options.merge_policy() == MergePolicy::InputOrder {
            let mut state = initial;
            for i in 0..view.samples() {
                if i % 256 == 0 {
                    context.check()?;
                }
                observe(&mut state, batch, i, origin)?;
            }
            workspace.stage[batch] = state;
            continue;
        }
        workspace.workers[..req.workers].fill(initial);
        let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
        for (slot, worker) in slots.iter_mut().zip(&mut workspace.workers[..req.workers]) {
            *slot = PartitionSlot::new(worker);
        }
        context.execute_partitioned(plan, &slots[..req.workers], &|partition, state, _| {
            for i in partition.range() {
                if i % 256 == 0 {
                    context.check()?;
                }
                observe(state, batch, i, origin)?;
            }
            Ok(())
        })?;
        drop(slots);
        let mut state = initial;
        for worker in &workspace.workers[..req.workers] {
            state.merge(worker)?;
        }
        workspace.stage[batch] = state;
    }
    context.check()
}
fn same_shape<S: RealScalar, M: WeightModel>(
    x: BatchView<'_, S, M>,
    y: BatchView<'_, S>,
) -> Result<(), SciOxideError> {
    if x.layout().shape().extents() != y.layout().shape().extents()
        || x.axis() != y.axis()
        || y.dimensions() != 1
    {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("y"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "paired observation geometry differs",
        ));
    }
    Ok(())
}
fn weight<S: RealScalar>(data: WeightData<'_, S>, index: usize) -> Result<S, SciOxideError> {
    match data {
        WeightData::Frequency(w) => crate::weights::frequency_scalar(w[index]),
        WeightData::Reliability(w) | WeightData::Analytic(w) => Ok(w[index]),
        _ => Err(crate::error_helpers::invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid moment weight capability",
        )),
    }
}

// Keep the input-order kernel typed: one ordinary moment state per lane, with
// no variant dispatch or partition arena on the observation path.
fn ordered_moments<S: RealScalar>(
    view: BatchView<'_, S>,
    limits: TrimLimits<S>,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    mut stage: impl FnMut(usize, OnlineMoments<S>),
) -> Result<(), SciOxideError> {
    if view.dimensions() != 1 {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("view"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "moment observations must be scalar",
        ));
    }
    for batch in 0..view.batches() {
        context.check()?;
        let origin = view.batch_origin(batch);
        let mut moments = OnlineMoments::new(options.moment_order());
        for i in 0..view.samples() {
            if i % 256 == 0 {
                context.check()?;
            }
            if let Some(value) = admit_at_origin(view, origin, batch, i, options)?
                && limits.contains(value)
            {
                moments.observe(value)?;
            }
        }
        stage(batch, moments);
    }
    context.check()
}

/// Computes the moments of every batch of `view` and writes one snapshot per batch to `output`.
///
/// Each snapshot holds what an [`OnlineMoments`] accumulator would hold after
/// observing the batch, up to [`StatisticsOptions::moment_order`]. Missing and
/// non-finite observations are handled by the policies in `options`.
/// [`StatisticsOptions::merge_policy`] selects sequential input order or
/// partitions run under `context`. The cost is O(n) per batch of `n`
/// observations.
///
/// # Errors
///
/// - `DimensionMismatch` if `output.len()` differs from the number of
///   batches, or if `workspace` is too small for this call.
/// - The errors of [`ReductionWorkspace::workspace_requirements`].
/// - `InvalidInput` or `NonFinite` for a missing or non-finite observation
///   that `options` rejects.
/// - The errors of [`OnlineMoments::observe`] and [`OnlineMoments::merge`].
/// - The errors of `context`; see
///   [Execution contexts](scioxide_core#execution-contexts).
pub fn reduce_moments_into<S: RealScalar>(
    view: BatchView<'_, S>,
    options: StatisticsOptions,
    output: &mut [MomentsSnapshot<S>],
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    reduce_trimmed_moments_into(
        view,
        TrimLimits::new(None, None)?,
        options,
        output,
        workspace,
        context,
    )
}
/// Computes the moments of the observations inside `limits` for every batch of `view`.
///
/// Observations are first checked against the policies in `options`, then
/// those outside [`TrimLimits`] are left out. Otherwise it works as
/// [`reduce_moments_into`].
///
/// # Errors
///
/// Returns the errors of [`reduce_moments_into`].
pub fn reduce_trimmed_moments_into<S: RealScalar>(
    view: BatchView<'_, S>,
    limits: TrimLimits<S>,
    options: StatisticsOptions,
    output: &mut [MomentsSnapshot<S>],
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != view.batches() {
        return Err(SciOxideError::dimension_mismatch(
            [view.batches()],
            [output.len()],
        ));
    }
    if options.merge_policy() == MergePolicy::InputOrder {
        let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
        if workspace.check(&req)? != Capacity::Sufficient {
            return Err(SciOxideError::dimension_mismatch(
                [req.batches, req.workers],
                [workspace.reserved.batches, workspace.reserved.workers],
            ));
        }
        ordered_moments(view, limits, options, context, |batch, moments| {
            workspace.stage[batch] = State::Moments(moments)
        })?;
    } else {
        run(
            view,
            options,
            State::Moments(OnlineMoments::new(options.moment_order())),
            workspace,
            context,
            |state, batch, i, origin| {
                if let Some(x) = admit_at_origin(view, origin, batch, i, options)?
                    && limits.contains(x)
                    && let State::Moments(a) = state
                {
                    a.observe(x)?;
                }
                Ok(())
            },
        )?;
    }
    for (out, state) in output.iter_mut().zip(&workspace.stage) {
        if let State::Moments(a) = state {
            *out = a.snapshot();
        }
    }
    Ok(())
}
/// Computes the covariance of paired batches of `x` and `y` and writes one snapshot per batch.
///
/// Each snapshot holds what an [`OnlineCovariance`] accumulator would hold
/// after observing the pairs of the batch. A pair is left out if either value
/// is skipped by the policies in `options`.
///
/// # Errors
///
/// - `InvalidInput` if `y` differs from `x` in shape or sample axis, or has
///   vector observations.
/// - The errors of [`reduce_moments_into`], with [`OnlineCovariance::observe`]
///   in place of [`OnlineMoments::observe`].
pub fn reduce_covariance_into<S: RealScalar>(
    x: BatchView<'_, S>,
    y: BatchView<'_, S>,
    options: StatisticsOptions,
    output: &mut [CovarianceSnapshot<S>],
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    same_shape(x, y)?;
    if output.len() != x.batches() {
        return Err(SciOxideError::dimension_mismatch(
            [x.batches()],
            [output.len()],
        ));
    }
    run(
        x,
        options,
        State::Covariance(OnlineCovariance::new(options.moment_order())),
        workspace,
        context,
        |state, batch, i, origin| {
            if let Some(x) = admit_at_origin(x, origin, batch, i, options)?
                && let Some(y) = admit(y, batch, i, options)?
                && let State::Covariance(a) = state
            {
                a.observe(x, y)?;
            }
            Ok(())
        },
    )?;
    for (out, state) in output.iter_mut().zip(&workspace.stage) {
        if let State::Covariance(a) = state {
            *out = a.snapshot();
        }
    }
    Ok(())
}
/// Computes the weighted moments of every batch of `view` and writes one snapshot per batch.
///
/// Each snapshot holds what an [`OnlineWeightedMoments`] accumulator with the
/// weight model `M` would hold after observing the batch.
///
/// # Errors
///
/// - `InvalidInput` if a weight is negative or not finite.
/// - The errors of [`reduce_moments_into`], with
///   [`OnlineWeightedMoments::observe`] in place of [`OnlineMoments::observe`].
pub fn reduce_weighted_moments_into<S: RealScalar, M: MomentWeights>(
    view: BatchView<'_, S, M>,
    options: StatisticsOptions,
    output: &mut [WeightedMomentsSnapshot<S, M>],
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != view.batches() {
        return Err(SciOxideError::dimension_mismatch(
            [view.batches()],
            [output.len()],
        ));
    }
    run(
        view,
        options,
        State::WeightedMoments(crate::weighted::WeightedMoments::new(
            M::SEMANTICS,
            options.moment_order(),
        )),
        workspace,
        context,
        |state, batch, i, origin| {
            if let Some(x) = admit_at_origin(view, origin, batch, i, options)?
                && let State::WeightedMoments(a) = state
            {
                a.observe(x, weight(view.weights, batch * view.samples() + i)?)?;
            }
            Ok(())
        },
    )?;
    for (out, state) in output.iter_mut().zip(&workspace.stage) {
        if let State::WeightedMoments(a) = state {
            *out = a.snapshot();
        }
    }
    Ok(())
}
/// Computes the weighted covariance of paired batches of `x` and `y`, with the weights of `x`.
///
/// Each snapshot holds what an [`OnlineWeightedCovariance`] accumulator with
/// the weight model `M` would hold after observing the pairs of the batch. A
/// pair is left out if either value is skipped by the policies in `options`.
///
/// # Errors
///
/// - `InvalidInput` if `y` differs from `x` in shape or sample axis, has
///   vector observations, or if a weight is negative or not finite.
/// - The errors of [`reduce_moments_into`], with
///   [`OnlineWeightedCovariance::observe`] in place of [`OnlineMoments::observe`].
pub fn reduce_weighted_covariance_into<S: RealScalar, M: MomentWeights>(
    x: BatchView<'_, S, M>,
    y: BatchView<'_, S>,
    options: StatisticsOptions,
    output: &mut [WeightedCovarianceSnapshot<S, M>],
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    same_shape(x, y)?;
    if output.len() != x.batches() {
        return Err(SciOxideError::dimension_mismatch(
            [x.batches()],
            [output.len()],
        ));
    }
    run(
        x,
        options,
        State::WeightedCovariance(crate::weighted_covariance::WeightedCovariance::new(
            M::SEMANTICS,
            options.moment_order(),
        )),
        workspace,
        context,
        |state, batch, i, origin| {
            if let Some(a) = admit_at_origin(x, origin, batch, i, options)?
                && let Some(b) = admit(y, batch, i, options)?
                && let State::WeightedCovariance(c) = state
            {
                c.observe(a, b, weight(x.weights, batch * x.samples() + i)?)?;
            }
            Ok(())
        },
    )?;
    for (out, state) in output.iter_mut().zip(&workspace.stage) {
        if let State::WeightedCovariance(a) = state {
            *out = a.snapshot();
        }
    }
    Ok(())
}

/// Computes the moment of order `k = order` about a center for every batch of `view`.
///
/// For a batch of `n` observations the moment is
///
/// ```text
/// μₖ = Σᵢ (xᵢ − c)ᵏ / n,
/// ```
///
/// where `c` is `center` if given, and otherwise the batch mean. Use
/// `center = Some(0.0)` for raw moments. The output is `None` for an empty
/// batch. Order 0 gives exactly 1, and order 1 about the mean gives exactly 0.
/// About the mean, the reduction makes one pass for the means and one for the
/// powers.
///
/// # Errors
///
/// - `NonFinite` if `center` is NaN or infinite.
/// - The errors of [`reduce_moments_into`].
#[allow(clippy::too_many_arguments)]
pub fn reduce_moment_into<S: RealScalar>(
    view: BatchView<'_, S>,
    order: usize,
    center: Option<S>,
    options: StatisticsOptions,
    output: &mut [Option<S>],
    workspace: &mut ReductionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if center.is_some_and(|x| !x.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "nonfinite moment center",
        ));
    }
    if output.len() != view.batches() {
        return Err(SciOxideError::dimension_mismatch(
            [view.batches()],
            [output.len()],
        ));
    }
    if center.is_none() && order > 1 {
        run(
            view,
            options,
            State::Moments(OnlineMoments::new(crate::MomentOrder::Second)),
            workspace,
            context,
            |state, batch, i, origin| {
                if let Some(x) = admit_at_origin(view, origin, batch, i, options)?
                    && let State::Moments(a) = state
                {
                    a.observe(x)?;
                }
                Ok(())
            },
        )?;
        for (slot, state) in workspace.centers[..view.batches()]
            .iter_mut()
            .zip(&workspace.stage[..view.batches()])
        {
            if let State::Moments(a) = state {
                *slot = if a.snapshot().count() == 0 {
                    ScaledAccumulator::ZERO
                } else {
                    a.state.mean()?
                };
            }
        }
    }
    // Borrow centers separately while the shared partition driver stages powers.
    let mut centers = core::mem::take(&mut workspace.centers);
    let result = run(
        view,
        options,
        State::Power {
            count: 0,
            sum: Compensated::zero(),
        },
        workspace,
        context,
        |state, batch, i, origin| {
            let Some(x) = admit_at_origin(view, origin, batch, i, options)? else {
                return Ok(());
            };
            let term = if order == 0 {
                Scaled::<f64>::ONE
            } else if order == 1 && center.is_none() {
                Scaled::<f64>::ZERO
            } else {
                let origin = if let Some(x) = center {
                    ScaledAccumulator::from_value(x.into_f64())?
                } else {
                    centers[batch]
                };
                crate::arithmetic::integer_power(
                    crate::arithmetic::center_difference(
                        origin,
                        ScaledAccumulator::from_value(x.into_f64())?,
                    )?,
                    order,
                )?
            };
            if let State::Power { count, sum } = state {
                *count = count.checked_add(1).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("moment_count"))
                })?;
                *sum = sum.added(term)?;
            }
            Ok(())
        },
    );
    core::mem::swap(&mut centers, &mut workspace.centers);
    result?;
    for (slot, state) in workspace.values[..view.batches()]
        .iter_mut()
        .zip(&workspace.stage[..view.batches()])
    {
        if let State::Power { count, sum } = state {
            *slot = if *count == 0 {
                None
            } else {
                Some(
                    sum.total()?
                        .div(Scaled::from_value(*count as f64)?)?
                        .round_to()?,
                )
            };
        }
    }
    output.copy_from_slice(&workspace.values[..view.batches()]);
    Ok(())
}

macro_rules! owned_reduction {
    ($name:ident, $configured:ident, $into:ident, $snapshot:ident, $initial:expr, $($extra:ident : $ty:ty),* $(,)?) => {
        /// Computes the covariance of paired batches of `view` and `y`, one snapshot per batch.
        ///
        /// It runs [`reduce_covariance_into`] on the current thread with a new
        /// workspace.
        ///
        /// # Errors
        ///
        /// Returns the errors of [`reduce_covariance_into`] and `AllocationFailure`.
        pub fn $configured<S: RealScalar>(view: BatchView<'_, S>, $($extra: $ty,)* options: StatisticsOptions) -> Result<Vec<$snapshot<S>>, SciOxideError> {
            let context = ExecutionContext::sequential_native();
            let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
            let mut workspace = ReductionWorkspace::workspace(&req)?;
            let mut output = try_filled(view.batches(), ($initial)(options), scioxide_core::quantity!("reduction_output"))?;
            $into(view, $($extra,)* options, &mut output, &mut workspace, context)?;
            Ok(output)
        }
        /// Computes batch covariances with the [one-shot defaults](crate#one-shot-forms).
        ///
        /// # Errors
        ///
        /// Returns the errors of the configured covariance reduction.
        pub fn $name<S: RealScalar>(view: BatchView<'_, S>, $($extra: $ty,)*) -> Result<Vec<$snapshot<S>>, SciOxideError> {
            $configured(view, $($extra,)* StatisticsOptions::default()).map_err(|error| crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("view")))
        }
    };
}
/// Computes the result of [`reduce_moments_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`reduce_moments_with_options`] for the fixed default settings.
pub fn reduce_moments<S: RealScalar>(
    view: BatchView<'_, S>,
) -> Result<Vec<MomentsSnapshot<S>>, SciOxideError> {
    reduce_moments_with_options(view, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("view"),
        )
    })
}
/// Computes the moments of every batch of `view` and returns one snapshot per batch.
///
/// It runs [`reduce_moments_into`] on the current thread with a new workspace.
///
/// # Errors
///
/// Returns the errors of [`reduce_moments_into`] and `AllocationFailure`.
///
/// # Examples
///
/// ```
/// use scioxide_core::StridedLayout;
/// use scioxide_stats::{Axis, BatchView, StatisticsOptions, reduce_moments_with_options};
/// // A 2×3 row-major array; each row is one batch.
/// let values = [1.0_f64, 2.0, 3.0, 10.0, 20.0, 30.0];
/// let layout = StridedLayout::new(&[2, 3], &[3, 1])?;
/// let view = BatchView::new(&values, layout, Axis::Index(1))?;
/// let rows = reduce_moments_with_options(view, StatisticsOptions::new())?;
/// let mean = rows[1].mean()?.unwrap_or(f64::NAN);
/// let variance = rows[1].sample_variance()?.unwrap_or(f64::NAN);
/// // The second row 10, 20, 30 has mean 20 and sample variance 100.
/// assert!((mean - 20.0).abs() < 1e-12 && (variance - 100.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn reduce_moments_with_options<S: RealScalar>(
    view: BatchView<'_, S>,
    options: StatisticsOptions,
) -> Result<Vec<MomentsSnapshot<S>>, SciOxideError> {
    reduce_trimmed_moments_with_options(view, TrimLimits::new(None, None)?, options)
}
/// Computes the result of [`reduce_trimmed_moments_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`reduce_trimmed_moments_with_options`] for the fixed default settings.
pub fn reduce_trimmed_moments<S: RealScalar>(
    view: BatchView<'_, S>,
    limits: TrimLimits<S>,
) -> Result<Vec<MomentsSnapshot<S>>, SciOxideError> {
    reduce_trimmed_moments_with_options(view, limits, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("view"),
        )
    })
}
/// Computes the moments of the observations inside `limits` for every batch of `view`.
///
/// It runs [`reduce_trimmed_moments_into`] on the current thread and returns
/// one snapshot per batch.
///
/// # Errors
///
/// Returns the errors of [`reduce_trimmed_moments_into`] and `AllocationFailure`.
pub fn reduce_trimmed_moments_with_options<S: RealScalar>(
    view: BatchView<'_, S>,
    limits: TrimLimits<S>,
    options: StatisticsOptions,
) -> Result<Vec<MomentsSnapshot<S>>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    if options.merge_policy() == MergePolicy::InputOrder {
        if view.dimensions() != 1 {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "moment observations must be scalar",
            ));
        }
        let mut output = scioxide_core::backend::try_with_capacity(
            view.batches(),
            scioxide_core::quantity!("moments_output"),
        )?;
        ordered_moments(view, limits, options, context, |_, moments| {
            output.push(moments.snapshot())
        })?;
        return Ok(output);
    }
    let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
    let mut workspace = ReductionWorkspace::workspace(&req)?;
    let mut output = try_filled(
        view.batches(),
        OnlineMoments::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("moments_output"),
    )?;
    reduce_trimmed_moments_into(view, limits, options, &mut output, &mut workspace, context)?;
    Ok(output)
}
owned_reduction!(reduce_covariance, reduce_covariance_with_options, reduce_covariance_into, CovarianceSnapshot, |o: StatisticsOptions| OnlineCovariance::<S>::new(o.moment_order()).snapshot(), y: BatchView<'_, S>);

/// Computes the result of [`reduce_weighted_moments_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`reduce_weighted_moments_with_options`] for the fixed default settings.
pub fn reduce_weighted_moments<S: RealScalar, M: MomentWeights>(
    view: BatchView<'_, S, M>,
) -> Result<Vec<WeightedMomentsSnapshot<S, M>>, SciOxideError> {
    reduce_weighted_moments_with_options(view, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("view"),
        )
    })
}
/// Computes the weighted moments of every batch of `view` and returns one snapshot per batch.
///
/// It runs [`reduce_weighted_moments_into`] on the current thread with a new
/// workspace.
///
/// # Errors
///
/// Returns the errors of [`reduce_weighted_moments_into`] and `AllocationFailure`.
pub fn reduce_weighted_moments_with_options<S: RealScalar, M: MomentWeights>(
    view: BatchView<'_, S, M>,
    options: StatisticsOptions,
) -> Result<Vec<WeightedMomentsSnapshot<S, M>>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
    let mut workspace = ReductionWorkspace::workspace(&req)?;
    let mut output = try_filled(
        view.batches(),
        OnlineWeightedMoments::<S, M>::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("weighted_reduction_output"),
    )?;
    reduce_weighted_moments_into(view, options, &mut output, &mut workspace, context)?;
    Ok(output)
}
/// Computes the result of [`reduce_weighted_covariance_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`reduce_weighted_covariance_with_options`] for the fixed default settings.
pub fn reduce_weighted_covariance<S: RealScalar, M: MomentWeights>(
    view: BatchView<'_, S, M>,
    y: BatchView<'_, S>,
) -> Result<Vec<WeightedCovarianceSnapshot<S, M>>, SciOxideError> {
    reduce_weighted_covariance_with_options(view, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("view"),
        )
    })
}
/// Computes the weighted covariance of paired batches, with the weights of `view`.
///
/// It runs [`reduce_weighted_covariance_into`] on the current thread with a new
/// workspace and returns one snapshot per batch.
///
/// # Errors
///
/// Returns the errors of [`reduce_weighted_covariance_into`] and `AllocationFailure`.
pub fn reduce_weighted_covariance_with_options<S: RealScalar, M: MomentWeights>(
    view: BatchView<'_, S, M>,
    y: BatchView<'_, S>,
    options: StatisticsOptions,
) -> Result<Vec<WeightedCovarianceSnapshot<S, M>>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
    let mut workspace = ReductionWorkspace::workspace(&req)?;
    let mut output = try_filled(
        view.batches(),
        OnlineWeightedCovariance::<S, M>::new(options.moment_order()).snapshot(),
        scioxide_core::quantity!("weighted_reduction_output"),
    )?;
    reduce_weighted_covariance_into(view, y, options, &mut output, &mut workspace, context)?;
    Ok(output)
}
/// Computes the result of [`reduce_moment_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`reduce_moment_with_options`] for the fixed default settings.
pub fn reduce_moment<S: RealScalar>(
    view: BatchView<'_, S>,
    order: usize,
    center: Option<S>,
) -> Result<Vec<Option<S>>, SciOxideError> {
    reduce_moment_with_options(view, order, center, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("view"),
        )
    })
}
/// Computes the moment of order `order` about a center for every batch of `view`.
///
/// It runs [`reduce_moment_into`] on the current thread with a new workspace
/// and returns one value per batch.
///
/// # Errors
///
/// Returns the errors of [`reduce_moment_into`] and `AllocationFailure`.
pub fn reduce_moment_with_options<S: RealScalar>(
    view: BatchView<'_, S>,
    order: usize,
    center: Option<S>,
    options: StatisticsOptions,
) -> Result<Vec<Option<S>>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = ReductionWorkspace::workspace_requirements(view, options, context)?;
    let mut workspace = ReductionWorkspace::workspace(&req)?;
    let mut output = try_filled(
        view.batches(),
        None,
        scioxide_core::quantity!("moment_output"),
    )?;
    reduce_moment_into(
        view,
        order,
        center,
        options,
        &mut output,
        &mut workspace,
        context,
    )?;
    Ok(output)
}
