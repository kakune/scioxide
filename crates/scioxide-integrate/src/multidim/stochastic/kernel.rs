//! Block sampling shared by Monte Carlo evaluation and VEGAS and MISER training.
//!
//! A schedule draws `samples` observations in consecutive blocks of
//! [`SAMPLE_BLOCK`]. Block `b` opens the source at global block `offset + b`, so
//! its uniforms do not depend on which worker evaluates it. Each block is
//! summarized in one sequential pass, and the summaries are merged in block
//! order, so sequential and parallel runs give bit-identical results and counts.

use alloc::vec::Vec;

use scioxide_core::backend::{
    PartitionSlot, PartitionedSolve, try_filled, with_partitioned_solves,
};
use scioxide_core::{
    AffineValue, AlgorithmId, BudgetKind, EvaluationRole, ExecutionContext, ExecutionPolicy,
    Interval, RealScalar, Scaled, SciOxideError, SolveBudgets, SolveContext,
};

use super::{MonteCarloProposal, SAMPLE_BLOCK, Source};
use crate::evaluation::{LIMITS, MAX_WORKERS, dispatch};
use crate::geometry::{IntervalGeometry, checked_volume};
use crate::report::IntegrationTermination;

/// Returns the number of worker lanes for `work` units under `policy` and `execution`.
pub(super) fn lanes(
    policy: ExecutionPolicy,
    work: usize,
    execution: ExecutionContext<'_>,
) -> usize {
    execution.plan(policy, work, LIMITS).worker_count().max(1)
}

/// Summary of one block, stored in a lane.
pub(super) trait Summary: Send {
    /// Resets to the empty summary before a block.
    fn clear(&mut self);
    /// Completes the summary after the last observation of a block.
    fn finish(&mut self) -> Result<(), SciOxideError> {
        Ok(())
    }
}

/// Scratch storage, call counts and summary of one block evaluated by one worker.
#[derive(Debug, Clone)]
pub(super) struct Lane<S: RealScalar, T> {
    uniforms: Vec<S>,
    point: Vec<S>,
    geometry: Vec<IntervalGeometry<S>>,
    draws: usize,
    calls: usize,
    quota: usize,
    attempted: usize,
    observed: usize,
    stop: Option<IntegrationTermination>,
    pub(super) summary: T,
}

impl<S: RealScalar, T> Lane<S, T> {
    /// Allocates a lane for `coordinates` uniforms and point coordinates.
    pub(super) fn allocate(coordinates: usize, summary: T) -> Result<Self, SciOxideError> {
        let unit = IntervalGeometry::prepare_half_open(Interval::new(S::ZERO, S::ONE)?)?
            .ok_or_else(degenerate)?;
        Ok(Self {
            uniforms: try_filled(
                coordinates,
                S::ZERO,
                scioxide_core::quantity!("mc_uniforms"),
            )?,
            point: try_filled(coordinates, S::ZERO, scioxide_core::quantity!("mc_point"))?,
            geometry: try_filled(coordinates, unit, scioxide_core::quantity!("mc_geometry"))?,
            draws: 0,
            calls: 0,
            quota: 0,
            attempted: 0,
            observed: 0,
            stop: None,
            summary,
        })
    }

    /// Allocates `count` lanes with summaries from `summary`.
    pub(super) fn allocate_many(
        count: usize,
        coordinates: usize,
        mut summary: impl FnMut() -> Result<T, SciOxideError>,
    ) -> Result<Vec<Self>, SciOxideError> {
        let mut lanes =
            scioxide_core::backend::try_with_capacity(count, scioxide_core::quantity!("mc_lanes"))?;
        for _ in 0..count {
            lanes.push(Self::allocate(coordinates, summary()?)?);
        }
        Ok(lanes)
    }
}

fn degenerate() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "non-degenerate box produced degenerate geometry",
    )
}

/// Prepares every lane's geometry for `bounds` and returns the box volume.
/// Every bound must have positive width.
pub(super) fn prepare<S: RealScalar, T>(
    lanes: &mut [Lane<S, T>],
    bounds: &[Interval<S>],
) -> Result<Scaled<S>, SciOxideError> {
    for lane in lanes.iter_mut() {
        if lane.geometry.len() < bounds.len() {
            crate::kernel::shape(bounds.len(), lane.geometry.len())?;
        }
        for (geometry, &bound) in lane.geometry.iter_mut().zip(bounds) {
            *geometry = IntervalGeometry::prepare_half_open(bound)?.ok_or_else(degenerate)?;
        }
    }
    match lanes.first() {
        Some(lane) => checked_volume(lane.geometry[..bounds.len()].iter().copied().map(Ok)),
        None => Err(crate::kernel::invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Empty,
            "sampling requires at least one lane",
        )),
    }
}

/// Inputs shared by every block of a schedule.
pub(super) struct Job<'a, Src: ?Sized, P: ?Sized, F: ?Sized> {
    pub(super) source: &'a Src,
    /// `None` for uniform sampling, where uniforms map directly to points of the box.
    pub(super) proposal: Option<&'a P>,
    pub(super) function: &'a F,
    pub(super) dimension: usize,
    pub(super) uniforms: usize,
}

/// Observations of one child: its index, first global block and count.
#[derive(Debug, Clone, Copy)]
pub(super) struct Schedule {
    pub(super) child: usize,
    pub(super) offset: u64,
    pub(super) samples: usize,
}

impl Schedule {
    /// Returns the number of blocks for `samples` observations.
    pub(super) const fn blocks(samples: usize) -> usize {
        samples.div_ceil(SAMPLE_BLOCK)
    }

    /// Returns the schedule of child `child` when every child draws `samples`
    /// observations.
    pub(super) fn uniform(child: usize, samples: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            child,
            offset: offset(child, Self::blocks(samples))?,
            samples,
        })
    }
}

/// Returns the first global block of child `child` when every child spans
/// `blocks` blocks.
pub(super) fn offset(child: usize, blocks: usize) -> Result<u64, SciOxideError> {
    child
        .checked_mul(blocks)
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("sample_blocks")))
}

/// One checked observation, as passed to a summary.
pub(super) struct Draw<'a, S> {
    pub(super) uniforms: &'a [S],
    pub(super) point: &'a [S],
    pub(super) density: S,
}

/// Outcome of one schedule.
#[derive(Debug, Clone, Copy)]
pub(super) struct Progress {
    pub(super) observations: usize,
    pub(super) evaluations: usize,
    pub(super) stop: Option<IntegrationTermination>,
}

/// Returns how many of `requested` observations the budgets allow, and the stop
/// when none are allowed.
fn quota(
    context: &SolveContext<'_, '_>,
    requested: usize,
    cost: usize,
) -> Result<(usize, Option<IntegrationTermination>), SciOxideError> {
    if !context.can_consume(BudgetKind::Sample, 1)?.is_granted() {
        return Ok((
            0,
            Some(IntegrationTermination::BudgetLimit(BudgetKind::Sample)),
        ));
    }
    if let Some(stop) = context.stop_reason()? {
        return Ok((0, Some(IntegrationTermination::ExecutionStopped(stop))));
    }
    let mut count = requested;
    let mut limiting = BudgetKind::Sample;
    for (kind, per_sample) in [
        (BudgetKind::Sample, 1),
        (BudgetKind::Evaluation, cost),
        (BudgetKind::Iteration, 1),
        (BudgetKind::Proposal, 1),
    ] {
        if let Some(remaining) = context.effective_budget_report(kind).remaining() {
            let allowed = remaining / per_sample;
            if allowed < count {
                count = allowed;
                limiting = kind;
            }
        }
    }
    if count == 0 {
        return Ok((0, Some(IntegrationTermination::BudgetLimit(limiting))));
    }
    let evaluations = count.checked_mul(cost).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("sample_providers"))
    })?;
    context.can_consume(BudgetKind::Sample, count)?.require()?;
    context
        .can_consume(BudgetKind::Evaluation, evaluations)?
        .require()?;
    context.can_record_iterations(count)?.require()?;
    context.can_record_proposals(count)?.require()?;
    Ok((count, None))
}

/// Records the observations actually attempted, including a failed last call.
fn record<S: RealScalar, T>(
    context: &mut SolveContext<'_, '_>,
    lane: &Lane<S, T>,
) -> Result<usize, SciOxideError> {
    context.record_iterations(lane.attempted)?.require()?;
    context.record_proposals(lane.attempted)?.require()?;
    context
        .try_consume(BudgetKind::Sample, lane.attempted)?
        .require()?;
    context
        .record_providers(EvaluationRole::Sample, lane.draws)?
        .require()?;
    context
        .record_providers(EvaluationRole::Function, lane.calls)?
        .require()?;
    Ok(lane.calls)
}

/// Evaluates one block, checking for cancellation between observations.
fn observe<S, T, Src, P, F, V, A, const PROPOSAL: bool, const UNIT_BOX: bool>(
    job: &Job<'_, Src, P, F>,
    schedule: Schedule,
    block: usize,
    lane: &mut Lane<S, T>,
    len: usize,
    execution: ExecutionContext<'_>,
    accumulate: &A,
) -> Result<usize, SciOxideError>
where
    S: RealScalar,
    Src: Source<S> + ?Sized,
    P: MonteCarloProposal<S> + ?Sized,
    F: Fn(&[S]) -> Result<V, SciOxideError> + ?Sized,
    A: Fn(&mut T, Draw<'_, S>, &V) -> Result<(), SciOxideError> + ?Sized,
{
    let Lane {
        uniforms,
        point,
        geometry,
        draws,
        calls,
        summary,
        attempted,
        stop,
        ..
    } = lane;
    let uniforms = &mut uniforms[..job.uniforms];
    let point = &mut point[..job.dimension];
    let geometry = &geometry[..job.dimension];
    let proposal = if PROPOSAL { job.proposal } else { None };
    let global = u64::try_from(block)
        .ok()
        .and_then(|block| schedule.offset.checked_add(block))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("sample_blocks")))?;
    let mut state = job.source.open(schedule.child, global)?;
    let first = block * SAMPLE_BLOCK;
    let mut observed = 0;
    for index in first..first + len {
        if let Some(reason) = execution.stop_reason()? {
            *stop = Some(IntegrationTermination::ExecutionStopped(reason));
            break;
        }
        *attempted += 1;
        let admitted = job
            .source
            .fill(&mut state, schedule.child, index, uniforms, draws)?;
        let uniforms = admitted.as_slice();
        let density = if let Some(proposal) = proposal {
            point.fill(S::NAN);
            let density = proposal.sample(admitted, point)?;
            if !density.is_finite() || density <= S::ZERO {
                return Err(super::invalid(
                    scioxide_core::argument!("proposal"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "proposal density must be finite and positive",
                ));
            }
            for (coordinate, geometry) in point.iter_mut().zip(geometry) {
                *coordinate = geometry.half_open_point(*coordinate)?;
            }
            density
        } else if UNIT_BOX {
            // The unit box needs only the established lower-endpoint zero sign.
            for (coordinate, &uniform) in point.iter_mut().zip(uniforms) {
                *coordinate = uniform.abs();
            }
            S::ONE
        } else {
            for ((coordinate, geometry), &uniform) in
                point.iter_mut().zip(geometry).zip(uniforms.iter())
            {
                *coordinate = geometry.half_open_point_admitted(uniform)?;
            }
            S::ONE
        };
        *calls += 1;
        let value = (job.function)(point)?;
        accumulate(
            summary,
            Draw {
                uniforms,
                point,
                density,
            },
            &value,
        )?;
        observed += 1;
    }
    Ok(observed)
}

/// Draws one schedule through `lanes`, merging block summaries in block order.
///
/// Budgets are checked once per wave of blocks, and the attempts are recorded
/// before merging. Cancellation is checked between observations in every lane.
#[allow(clippy::too_many_arguments)]
pub(super) fn sample<S, T, Src, P, F, V, A>(
    job: &Job<'_, Src, P, F>,
    schedule: Schedule,
    lanes: &mut [Lane<S, T>],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
    accumulate: &A,
    fold: &mut dyn FnMut(&T) -> Result<(), SciOxideError>,
) -> Result<Progress, SciOxideError>
where
    S: RealScalar,
    T: Summary,
    Src: Source<S> + ?Sized,
    P: MonteCarloProposal<S> + Sync + ?Sized,
    F: Fn(&[S]) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
    A: Fn(&mut T, Draw<'_, S>, &V) -> Result<(), SciOxideError> + Sync,
{
    let cost = job.uniforms.checked_add(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("sample_providers"))
    })?;
    let blocks = Schedule::blocks(schedule.samples);
    let block_len = |block: usize| SAMPLE_BLOCK.min(schedule.samples - block * SAMPLE_BLOCK);
    let mut progress = Progress {
        observations: 0,
        evaluations: 0,
        stop: None,
    };
    let mut block = 0;
    while block < blocks && progress.stop.is_none() {
        let wave = if policy == ExecutionPolicy::Sequential {
            1
        } else {
            lanes.len().min(blocks - block)
        };
        let requested: usize = (block..block + wave).map(block_len).sum();
        let (mut available, stop) = quota(context, requested, cost)?;
        if available == 0 {
            progress.stop = stop;
            break;
        }
        let admitted = available;
        let mut active = 0;
        for (index, lane) in lanes[..wave].iter_mut().enumerate() {
            if available == 0 {
                break;
            }
            lane.quota = available.min(block_len(block + index));
            available -= lane.quota;
            lane.draws = 0;
            lane.calls = 0;
            lane.attempted = 0;
            lane.observed = 0;
            lane.stop = None;
            lane.summary.clear();
            active += 1;
        }
        let execution = context.execution();
        let outcome = dispatch(&mut lanes[..active], policy, execution, &|index, lane| {
            // Unswitch invariant proposal/geometry choices once per block. Each
            // observation keeps the same source, stop poll and accumulation.
            let observer = if job.proposal.is_some() {
                observe::<S, T, Src, P, F, V, A, true, false>
            } else if lane.geometry[..job.dimension]
                .iter()
                .all(|axis| axis.is_positive_unit())
            {
                observe::<S, T, Src, P, F, V, A, false, true>
            } else {
                observe::<S, T, Src, P, F, V, A, false, false>
            };
            let quota = lane.quota;
            lane.observed = observer(
                job,
                schedule,
                block + index,
                lane,
                quota,
                execution,
                accumulate,
            )?;
            lane.summary.finish()
        });
        for lane in &lanes[..active] {
            progress.evaluations += record(context, lane)?;
        }
        outcome?;
        for lane in &lanes[..active] {
            if lane.observed > 0 {
                fold(&lane.summary)?;
            }
            progress.observations += lane.observed;
            if progress.stop.is_none() {
                progress.stop = lane.stop;
            }
        }
        block += active;
        if progress.stop.is_none() && admitted < requested {
            progress.stop = quota(context, 1, cost)?.1;
        }
    }
    Ok(progress)
}

/// Runs `run` once per child in its own child solve, in index order or across
/// executor partitions. Parent budgets are divided by child index, so every
/// schedule sees the same limits.
#[allow(clippy::type_complexity)]
pub(super) fn each_child<S: RealScalar, T: Send, C: Send>(
    lanes: &mut [Lane<S, T>],
    children: &mut [C],
    solves: &mut [PartitionedSolve],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
    algorithm: AlgorithmId,
    run: &(
         dyn Fn(
        usize,
        &mut Lane<S, T>,
        &mut C,
        &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
             + Sync
     ),
) -> Result<(), SciOxideError> {
    crate::kernel::shape(children.len(), solves.len())?;
    with_partitioned_solves(
        context,
        solves,
        SolveBudgets::unbounded(),
        |execution, batch| {
            let plan = execution.plan(policy, children.len(), LIMITS);
            if lanes.len() < plan.worker_count() {
                crate::kernel::shape(plan.worker_count(), lanes.len())?;
            }
            let mut slots: [PartitionSlot<_>; MAX_WORKERS] =
                core::array::from_fn(|_| PartitionSlot::empty());
            let (mut children, mut batch) = (children, batch);
            for (partition, lane) in plan.partitions().zip(lanes.iter_mut()) {
                let len = partition.range().len();
                let (child_head, child_tail) = core::mem::take(&mut children).split_at_mut(len);
                let (batch_head, batch_tail) = batch.split_at(len)?;
                (children, batch) = (child_tail, batch_tail);
                slots[partition.worker_index()] =
                    PartitionSlot::new((lane, child_head, batch_head));
            }
            execution.execute_partitioned(
                plan,
                &slots[..plan.worker_count()],
                &|partition, (lane, children, batch), _| {
                    for ((index, child), permit) in
                        partition.range().zip(children.iter_mut()).zip(batch)
                    {
                        permit.run(
                            execution,
                            crate::workspace::identity(algorithm, index),
                            |child_context| run(index, lane, child, child_context),
                        )?;
                    }
                    Ok(())
                },
            )
        },
    )
}
