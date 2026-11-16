//! Isotonic regression of every lane along one axis of a strided array.
use crate::error_helpers::invalid;
use crate::order_axis::{logical_index, shape};
use crate::{
    IsotonicBlock, IsotonicDiagnostics, IsotonicDirection, KernelWeights, MissingPolicy, OrderData,
    OrderPolicy, RawIsotonicWorkspace,
};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use scioxide_core::backend::{PartitionSlot, try_filled, try_with_capacity};
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

/// Order constraint together with the axis, data and execution policies.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IsotonicPolicy {
    /// Non-strict order constraint within each lane.
    pub direction: IsotonicDirection,
    /// Axis, missing-value and non-finite policies, and worker limit.
    pub data: OrderPolicy,
}

/// Storage sizes for fitting all lanes. Each worker needs a slice workspace
/// for `lane_length` observations, and the result buffers hold `slots` fitted
/// values, `slots` blocks and `batches` lane records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IsotonicWorkspaceRequirements {
    /// Observations per lane.
    pub lane_length: usize,
    /// Number of lanes, each fitted independently.
    pub batches: usize,
    /// Number of workers, at most 64.
    pub workers: usize,
    /// Total number of elements, `lane_length · batches`, and the number of block slots.
    pub slots: usize,
}
impl IsotonicWorkspaceRequirements {
    pub(crate) fn required_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[
            (
                self.workers,
                core::mem::size_of::<RawIsotonicWorkspace<S>>(),
            ),
            (
                self.workers,
                RawIsotonicWorkspace::<S>::required_bytes(self.lane_length)?,
            ),
            (self.slots, core::mem::size_of::<S>()),
            (self.slots, core::mem::size_of::<IsotonicBlock<S>>()),
            (self.batches, core::mem::size_of::<IsotonicLaneResult<S>>()),
        ])
    }
    /// Computes the storage sizes, with the same worker plan as the fit.
    ///
    /// # Errors
    /// Fails on an invalid axis, a worker limit above 64, frequency weights, or
    /// size overflow.
    pub(crate) fn new<S: RealScalar>(
        view: OrderData<'_, S>,
        options: IsotonicPolicy,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if options.data.workers.get() > 64 {
            return Err(invalid(
                scioxide_core::argument!("options.data.workers"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "worker limit exceeds 64",
            ));
        }
        if matches!(view.weights, Some(KernelWeights::Frequency(_))) {
            return Err(invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "isotonic weights must be relative masses",
            ));
        }
        let (lane_length, batches) = shape(view.layout, options.data.axis)?;
        let slots = lane_length.checked_mul(batches).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("isotonic_slots"))
        })?;
        let plan = context.plan(
            options.data.execution,
            batches,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: options.data.workers,
            },
        );
        Ok(Self {
            lane_length,
            batches,
            workers: plan.worker_count(),
            slots,
        })
    }
}

/// Result for one lane of [`isotonic_regression_axes_into`](crate::isotonic_regression_axes_into).
///
/// The lane's blocks are `blocks[block_offset..block_offset + diagnostics.block_count]`
/// in [`IsotonicBatchOutput::blocks`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotonicLaneResult<S> {
    /// Index of the lane's first block: the lane index times the lane length.
    pub block_offset: usize,
    /// Summary of the lane's fit, including its number of blocks.
    pub diagnostics: IsotonicDiagnostics<S>,
}
impl<S: RealScalar> Default for IsotonicLaneResult<S> {
    fn default() -> Self {
        Self {
            block_offset: 0,
            diagnostics: IsotonicDiagnostics::default(),
        }
    }
}

/// Output buffers for [`isotonic_regression_axes_into`](crate::isotonic_regression_axes_into).
///
/// The lengths must match [`IsotonicRequirements`](crate::IsotonicRequirements).
/// Block slots beyond each lane's block count are not written.
pub struct IsotonicBatchOutput<'a, S> {
    /// Fitted values in row-major order, with the shape of the input. Length: `slots()`.
    pub fitted: &'a mut [S],
    /// Blocks of each lane, in lane order, with `lane_length()` slots per lane.
    /// Length: `slots()`. Block positions are indices within the lane, not
    /// offsets into `fitted`.
    pub blocks: &'a mut [IsotonicBlock<S>],
    /// One record per lane. Length: `batches()`.
    pub lanes: &'a mut [IsotonicLaneResult<S>],
}

/// Worker workspaces and result buffers, O(workers · lane_length + slots + batches)
/// in size. Fitting never grows it.
pub(crate) struct RawIsotonicBatchWorkspace<S: RealScalar> {
    pub(crate) workers: Vec<RawIsotonicWorkspace<S>>,
    fits: Vec<S>,
    blocks: Vec<IsotonicBlock<S>>,
    lanes: Vec<IsotonicLaneResult<S>>,
}
impl<S: RealScalar> Default for RawIsotonicBatchWorkspace<S> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            fits: Vec::new(),
            blocks: Vec::new(),
            lanes: Vec::new(),
        }
    }
}
impl<S: RealScalar> RawIsotonicBatchWorkspace<S> {
    /// Grows the workspace to the given sizes, keeping storage that is large enough.
    ///
    /// # Errors
    /// Fails on inconsistent sizes, size overflow or allocation failure; the
    /// previous storage is then unchanged.
    pub(crate) fn reserve(
        &mut self,
        r: IsotonicWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if r.workers > 64
            || (r.batches > 0 && r.workers == 0)
            || r.lane_length.checked_mul(r.batches) != Some(r.slots)
        {
            return Err(invalid(
                scioxide_core::argument!("r"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "inconsistent isotonic workspace requirements",
            ));
        }
        if self.sufficient(r) {
            return Ok(());
        }
        let mut workers =
            try_with_capacity(r.workers, scioxide_core::quantity!("isotonic_workers"))?;
        for _ in 0..r.workers {
            let mut w = RawIsotonicWorkspace::default();
            w.reserve(r.lane_length)?;
            workers.push(w);
        }
        let next = Self {
            workers,
            fits: try_filled(r.slots, S::ZERO, scioxide_core::quantity!("isotonic_fits"))?,
            blocks: try_filled(
                r.slots,
                IsotonicBlock::default(),
                scioxide_core::quantity!("isotonic_blocks"),
            )?,
            lanes: try_filled(
                r.batches,
                IsotonicLaneResult::default(),
                scioxide_core::quantity!("isotonic_lanes"),
            )?,
        };
        *self = next;
        Ok(())
    }
    pub(crate) fn sufficient(&self, r: IsotonicWorkspaceRequirements) -> bool {
        self.workers.len() >= r.workers
            && self
                .workers
                .iter()
                .take(r.workers)
                .all(|w| w.capacity() >= r.lane_length)
            && self.fits.len() >= r.slots
            && self.blocks.len() >= r.slots
            && self.lanes.len() >= r.batches
    }
}

fn load_lane<S: RealScalar>(
    view: OrderData<'_, S>,
    options: IsotonicPolicy,
    batch: usize,
    w: &mut RawIsotonicWorkspace<S>,
) -> Result<bool, SciOxideError> {
    let len = shape(view.layout, options.data.axis)?.0;
    let mut admit = |position: usize, offset: usize, _logical: usize| {
        let logical = batch * len + position;
        if view.validity.is_some_and(|mask| !mask[logical]) {
            return if options.data.missing == MissingPolicy::Skip {
                w.admit(
                    position,
                    S::ZERO,
                    S::ZERO,
                    options.data.non_finite,
                    scioxide_core::argument!("view"),
                )
            } else {
                Err(invalid(
                    scioxide_core::argument!("view"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing observation",
                ))
            };
        }
        let weight = match view.weights {
            Some(KernelWeights::Mass(weights)) => weights[logical],
            None => S::ONE,
            Some(KernelWeights::Frequency(_)) => {
                return Err(invalid(
                    scioxide_core::argument!("view"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "isotonic weights must be relative masses",
                ));
            }
        };
        w.admit(
            position,
            view.values[offset],
            weight,
            options.data.non_finite,
            scioxide_core::argument!("view"),
        )
    };
    let mut ordinary = true;
    if let Some(axis) = options.data.axis {
        let trailing = view.layout.shape().extents()[axis + 1..].iter().product();
        let len = view.layout.shape().extents()[axis];
        for (position, offset) in view.layout.axis(axis)?.lane(batch)?.offsets().enumerate() {
            ordinary &= admit(
                position,
                offset,
                logical_index(batch, position, len, trailing),
            )?;
        }
    } else {
        for (position, offset) in view.layout.offsets().enumerate() {
            ordinary &= admit(position, offset, position)?;
        }
    }
    Ok(ordinary)
}

/// Fits every lane independently. Each lane gives the same result whether the
/// lanes run sequentially or in parallel. An empty lane has an empty fit, and a
/// nonempty lane needs positive total weight. Missing entries are handled
/// before their values and weights are checked.
///
/// # Errors
/// Fails on invalid shapes or policies, a workspace that is too small, rejected
/// data, a numerical failure, cancellation or an execution failure.
pub(crate) fn isotonic_regression_axes_into<S: RealScalar>(
    view: OrderData<'_, S>,
    options: IsotonicPolicy,
    context: ExecutionContext<'_>,
    workspace: &mut RawIsotonicBatchWorkspace<S>,
    output: IsotonicBatchOutput<'_, S>,
) -> Result<(), SciOxideError> {
    let r = IsotonicWorkspaceRequirements::new(view, options, context)?;
    if output.fitted.len() != r.slots
        || output.blocks.len() != r.slots
        || output.lanes.len() != r.batches
    {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "axis output shape mismatch",
        ));
    }
    if !workspace.sufficient(r) {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient isotonic batch workspace",
        ));
    }
    context.check()?;
    if r.batches == 0 {
        return Ok(());
    }
    let plan = context.plan(
        options.data.execution,
        r.batches,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: options.data.workers,
        },
    );
    let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
    let mut fits_rest = &mut workspace.fits[..r.slots];
    let mut blocks_rest = &mut workspace.blocks[..r.slots];
    let mut lanes_rest = &mut workspace.lanes[..r.batches];
    for (partition, worker) in plan.partitions().zip(workspace.workers.iter_mut()) {
        let range = partition.range();
        let (fits, rest) = fits_rest.split_at_mut(range.len() * r.lane_length);
        fits_rest = rest;
        let (blocks, rest) = blocks_rest.split_at_mut(range.len() * r.lane_length);
        blocks_rest = rest;
        let (lanes, rest) = lanes_rest.split_at_mut(range.len());
        lanes_rest = rest;
        slots[partition.worker_index()] = PartitionSlot::new((worker, fits, blocks, lanes));
    }
    context.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, (worker, fits, blocks, lanes), _| {
            let range = partition.range();

            for (local, batch) in range.enumerate() {
                context.check()?;
                let ordinary = load_lane(view, options, batch, worker)?;
                let diagnostics = worker.fit(
                    r.lane_length,
                    options.direction,
                    ordinary,
                    scioxide_core::argument!("view"),
                )?;
                let start = local * r.lane_length;
                fits[start..start + r.lane_length].copy_from_slice(&worker.fitted[..r.lane_length]);
                blocks[start..start + diagnostics.block_count]
                    .copy_from_slice(&worker.blocks[..diagnostics.block_count]);
                lanes[local] = IsotonicLaneResult {
                    block_offset: batch * r.lane_length,
                    diagnostics,
                };
            }
            Ok(())
        },
    )?;

    context.check()?;
    if let Some(axis) = options.data.axis {
        let trailing = view.layout.shape().extents()[axis + 1..].iter().product();
        for batch in 0..r.batches {
            for position in 0..r.lane_length {
                output.fitted[logical_index(batch, position, r.lane_length, trailing)] =
                    workspace.fits[batch * r.lane_length + position];
            }
        }
    } else {
        output.fitted.copy_from_slice(&workspace.fits[..r.slots]);
    }
    for lane in &workspace.lanes[..r.batches] {
        let start = lane.block_offset;
        let end = start + lane.diagnostics.block_count;
        output.blocks[start..end].copy_from_slice(&workspace.blocks[start..end]);
    }
    output.lanes.copy_from_slice(&workspace.lanes[..r.batches]);
    Ok(())
}
