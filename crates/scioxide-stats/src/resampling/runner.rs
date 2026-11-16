//! Execution of resampling plans: observed statistic, replicates and inference.
use super::*;
use core::{mem::size_of, num::NonZeroUsize};
use scioxide_core::backend::{
    PartitionSlot, PartitionedSolve, checked_add, checked_mul, try_filled, try_reserve,
    with_partitioned_solves,
};
use scioxide_core::{
    BudgetKind, EvaluationRole, ExecutionPolicy, PartialResultTrust, ProblemToken, SolveBudgets,
    SolveContext, SolveIdentity,
};
use scioxide_random::{OutputPolicy, SamplingPolicy, SplittableRng, StatePolicy};

pub(super) struct Slot<S: RealScalar> {
    pub indices: Vec<usize>,
    pub offsets: Vec<usize>,
    pub counts: Vec<usize>,
    sample: Vec<S>,
    scratch: Vec<S>,
    value: Option<S>,
    termination: StatsTermination,
}
impl<S: RealScalar> Slot<S> {
    fn selection(&self) -> ResampledIndices<'_> {
        ResampledIndices {
            indices: &self.indices[..self.offsets[self.offsets.len() - 1]],
            offsets: &self.offsets,
        }
    }
}
/// Reusable workspace of [`ResamplingPlan::evaluate_into`], from
/// [`ResamplingPlan::workspace`].
///
/// Its contents after an error or an early stop are unspecified, but it stays
/// usable. See [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct ResamplingWorkspace<S: RealScalar> {
    pub(super) solve: crate::execution::SolveSlot<()>,
    slots: Vec<Slot<S>>,
    values: Vec<S>,
    deletes: Vec<S>,
    sorted: Vec<S>,
    completed: Vec<bool>,
    children: Vec<PartitionedSolve>,
}
impl<S: RealScalar> ResamplingWorkspace<S> {
    /// Returns the diagnostics of the last attempted call.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.solve.last_attempt()
    }
    pub(crate) fn reject_attempt(&mut self) {
        self.solve.reject();
    }
    pub(super) fn principal_values(&self) -> &[S] {
        &self.values
    }
}
impl<S: RealScalar> ResamplingPlan<S> {
    fn sample_len(&self) -> usize {
        if matches!(self.method, ResamplingMethod::MonteCarlo { .. }) {
            self.total()
        } else {
            0
        }
    }
    /// Returns the number of bytes of a workspace for this plan.
    ///
    /// It does not count the plan, the output slice, allocator overhead, or
    /// storage owned by the generator or the callbacks.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        let slots = self.options.batch_size.min(self.jobs());
        let scalar = checked_mul(
            self.replicates,
            if matches!(self.method, ResamplingMethod::Bootstrap { .. }) {
                2
            } else {
                1
            },
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        let scalar = checked_add(
            scalar,
            self.deletions,
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        let mut bytes = checked_mul(
            scalar,
            size_of::<S>(),
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        bytes = checked_add(
            bytes,
            checked_mul(
                self.jobs(),
                size_of::<bool>() + size_of::<PartitionedSolve>(),
                scioxide_core::quantity!("resampling_workspace"),
            )?,
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        let indices = checked_add(
            self.total(),
            checked_add(
                self.groups(),
                self.offsets.len(),
                scioxide_core::quantity!("resampling_workspace"),
            )?,
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        let mut per_slot = checked_add(
            size_of::<Slot<S>>(),
            checked_mul(
                indices,
                size_of::<usize>(),
                scioxide_core::quantity!("resampling_workspace"),
            )?,
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        let sample_len = self.sample_len();
        per_slot = checked_add(
            per_slot,
            checked_mul(
                sample_len,
                size_of::<S>(),
                scioxide_core::quantity!("statistic_scratch"),
            )?,
            scioxide_core::quantity!("statistic_scratch"),
        )?;
        per_slot = checked_add(
            per_slot,
            checked_mul(
                self.statistic_scratch,
                size_of::<S>(),
                scioxide_core::quantity!("statistic_scratch"),
            )?,
            scioxide_core::quantity!("statistic_scratch"),
        )?;
        bytes = checked_add(
            bytes,
            checked_mul(
                slots,
                per_slot,
                scioxide_core::quantity!("resampling_workspace"),
            )?,
            scioxide_core::quantity!("resampling_workspace"),
        )?;
        Ok(bytes)
    }
    /// Creates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if [`Self::workspace_bytes`] exceeds
    ///   [`ResamplingOptions::max_workspace_bytes`].
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace(&self) -> Result<ResamplingWorkspace<S>, SciOxideError> {
        crate::error_helpers::limit(
            self.workspace_bytes()?,
            self.options.max_workspace_bytes,
            scioxide_core::quantity!("runner_bytes"),
        )?;
        let n = self.options.batch_size.min(self.jobs());
        let mut slots = Vec::new();
        try_reserve(&mut slots, n, scioxide_core::quantity!("resampling_slots"))?;
        for _ in 0..n {
            slots.push(Slot {
                indices: try_filled(
                    self.total(),
                    0,
                    scioxide_core::quantity!("resampling_indices"),
                )?,
                offsets: try_filled(
                    self.offsets.len(),
                    0,
                    scioxide_core::quantity!("resampling_offsets"),
                )?,
                counts: try_filled(
                    self.groups(),
                    0,
                    scioxide_core::quantity!("resampling_counts"),
                )?,
                sample: try_filled(
                    self.sample_len(),
                    S::ZERO,
                    scioxide_core::quantity!("null_sample"),
                )?,
                scratch: try_filled(
                    self.statistic_scratch,
                    S::ZERO,
                    scioxide_core::quantity!("statistic_scratch"),
                )?,
                value: None,
                termination: StatsTermination::Completed,
            });
        }
        let mut children = Vec::new();
        try_reserve(
            &mut children,
            self.jobs(),
            scioxide_core::quantity!("replicate_accounting"),
        )?;
        children.resize_with(self.jobs(), PartitionedSolve::default);
        Ok(ResamplingWorkspace {
            solve: Default::default(),
            slots,
            values: try_filled(
                self.replicates,
                S::ZERO,
                scioxide_core::quantity!("replicate_values"),
            )?,
            deletes: try_filled(
                self.deletions,
                S::ZERO,
                scioxide_core::quantity!("jackknife_values"),
            )?,
            sorted: try_filled(
                if matches!(self.method, ResamplingMethod::Bootstrap { .. }) {
                    self.replicates
                } else {
                    0
                },
                S::ZERO,
                scioxide_core::quantity!("sorted_statistics"),
            )?,
            completed: try_filled(
                self.jobs(),
                false,
                scioxide_core::quantity!("completed_replicates"),
            )?,
            children,
        })
    }
    pub(super) fn validate(
        &self,
        problem: ResamplingProblem<'_, S>,
        output: &[S],
        ws: &ResamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        if output.len() != self.replicates
            || ws.values.len() != self.replicates
            || ws.deletes.len() != self.deletions
            || ws.slots.len() != self.options.batch_size.min(self.jobs())
            || ws.completed.len() != self.jobs()
            || ws.slots.iter().any(|s| {
                s.indices.len() != self.total()
                    || s.offsets.len() != self.offsets.len()
                    || s.sample.len() != self.sample_len()
                    || s.scratch.len() != self.statistic_scratch
            })
            || ws.sorted.len()
                != if matches!(self.method, ResamplingMethod::Bootstrap { .. }) {
                    self.replicates
                } else {
                    0
                }
        {
            return Err(invalid(
                scioxide_core::argument!("ws"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "plan/workspace/output shape mismatch",
            ));
        }
        crate::error_helpers::limit(
            self.workspace_bytes()?,
            self.options.max_workspace_bytes,
            scioxide_core::quantity!("resampling_workspace_bytes"),
        )?;
        match (self.method, problem) {
            (
                ResamplingMethod::MonteCarlo { .. },
                ResamplingProblem::MonteCarlo { observed, .. }
                | ResamplingProblem::MonteCarloWorkspace { observed, .. },
            ) => {
                if observed.len() != self.total() {
                    return Err(invalid(
                        scioxide_core::argument!("problem"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "null sample shape mismatch",
                    ));
                }
                observed
                    .iter()
                    .try_for_each(|&x| crate::error_helpers::input_finite(x).map(|_| ()))
            }
            (ResamplingMethod::MonteCarlo { .. }, _)
            | (_, ResamplingProblem::MonteCarlo { .. })
            | (_, ResamplingProblem::MonteCarloWorkspace { .. }) => Err(invalid(
                scioxide_core::argument!("problem"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "method/callback mismatch",
            )),
            _ => Ok(()),
        }
    }
    /// Runs the plan with all replicates drawn in order from `rng`.
    ///
    /// On a returned error, the state policy of `sampling` decides whether
    /// `rng` is restored. An early stop returns `Ok` and leaves `rng` where it
    /// is.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_sequential<R: BitGenerator, P: StatePolicy<R>>(
        &self,
        problem: ResamplingProblem<'_, S>,
        rng: &mut R,
        output: &mut [S],
        ws: &mut ResamplingWorkspace<S>,
        sampling: SamplingPolicy<P>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ResamplingResult<S>, SciOxideError> {
        self.validate(problem, output, ws)?;
        let checkpoint = P::capture(rng)?;
        let result = self.analyze(
            problem,
            output,
            ws,
            sampling.output_policy(),
            context,
            &mut |delete, values, completed, slots, _children, output, ctx| {
                let slot = &mut slots[0];
                for (i, (value, done)) in values.iter_mut().zip(completed).enumerate() {
                    self.job(problem, delete, i, rng, slot, ctx)?;
                    if let Some(v) = slot.value {
                        *value = v;
                        *done = true;
                        if !delete && sampling.output_policy() == OutputPolicy::Incremental {
                            output[i] = v;
                        }
                    }
                    if slot.termination != StatsTermination::Completed {
                        return Ok(slot.termination);
                    }
                }
                Ok(StatsTermination::Completed)
            },
        );
        if result.is_err() {
            P::on_error(rng, checkpoint);
        }
        result
    }
    /// Runs the plan with replicate `i` on stream `first_stream + i` of `root`.
    ///
    /// Budget shares are fixed per replicate before the batches run, so the
    /// result does not depend on the number of workers if the callbacks are
    /// deterministic.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_streams<R: BitGenerator + SplittableRng + Sync>(
        &self,
        problem: ResamplingProblem<'_, S>,
        root: &R,
        first_stream: u64,
        policy: ExecutionPolicy,
        output: &mut [S],
        ws: &mut ResamplingWorkspace<S>,
        publication: OutputPolicy,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ResamplingResult<S>, SciOxideError> {
        self.validate(problem, output, ws)?;
        first_stream
            .checked_add(u64::try_from(self.replicates - 1).map_err(|_| {
                SciOxideError::size_overflow(scioxide_core::quantity!("replicate_stream_id"))
            })?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("replicate_stream_id"))
            })?;
        self.analyze(
            problem,
            output,
            ws,
            publication,
            context,
            &mut |delete, values, completed, slots, children, output, ctx| {
                let algorithm = scioxide_core::algorithm_id!("stats", "Resampling", 1);
                with_partitioned_solves(
                    ctx,
                    children,
                    SolveBudgets::unbounded(),
                    |execution, children| {
                        let batch = slots.len();
                        let mut termination = StatsTermination::Completed;
                        for (batch_index, child_batch) in children.into_chunks(batch)?.enumerate() {
                            let start = batch_index * batch;
                            let active = &mut slots[..child_batch.len()];
                            for slot in active.iter_mut() {
                                slot.value = None;
                                slot.termination = StatsTermination::Completed;
                            }
                            let plan = execution.plan(
                                policy,
                                active.len(),
                                scioxide_core::PlanLimits {
                                    work_per_item: core::num::NonZeroUsize::MIN,
                                    auto_threshold: NonZeroUsize::MIN,
                                    worker_limit: NonZeroUsize::new(64).ok_or_else(|| {
                                        invalid(
                                            scioxide_core::argument!("policy"),
                                            scioxide_core::InvalidInputReason::Inconsistent,
                                            "worker limit",
                                        )
                                    })?,
                                },
                            );
                            let mut partitions: [_; 64] =
                                core::array::from_fn(|_| PartitionSlot::empty());
                            let mut remaining = &mut active[..];
                            let mut remaining_children = child_batch;
                            for partition in plan.partitions() {
                                let (entries, tail) =
                                    remaining.split_at_mut(partition.range().len());
                                remaining = tail;
                                let (children, tail) =
                                    remaining_children.split_at(partition.range().len())?;
                                remaining_children = tail;
                                partitions[partition.worker_index()] =
                                    PartitionSlot::new((entries, children));
                            }
                            let result = execution.execute_partitioned(
                                plan,
                                &partitions[..plan.worker_count()],
                                &|partition, (entries, children), _| {
                                    for (local, (slot, child)) in
                                        partition.range().zip(entries.iter_mut().zip(children))
                                    {
                                        let i = start + local;
                                        let identity = SolveIdentity::new(
                                            ProblemToken::new(i as u128),
                                            algorithm,
                                            0,
                                            0,
                                        );
                                        child.run(execution, identity, |ctx| {
                                            if !delete && self.randomized() {
                                                let mut rng =
                                                    root.split(first_stream + i as u64)?;
                                                self.job(problem, delete, i, &mut rng, slot, ctx)
                                            } else {
                                                self.job(
                                                    problem,
                                                    delete,
                                                    i,
                                                    &mut NoRandom(scioxide_core::argument!(
                                                        "problem"
                                                    )),
                                                    slot,
                                                    ctx,
                                                )
                                            }
                                        })?;
                                    }
                                    Ok(())
                                },
                            );

                            for (i, slot) in active.iter().enumerate() {
                                if let Some(value) = slot.value {
                                    values[start + i] = value;
                                    completed[start + i] = true;
                                    if !delete && publication == OutputPolicy::Incremental {
                                        output[start + i] = value;
                                    }
                                }
                                if termination == StatsTermination::Completed
                                    && slot.termination != StatsTermination::Completed
                                {
                                    termination = slot.termination;
                                }
                            }
                            result?;
                        }
                        Ok(termination)
                    },
                )
            },
        )
    }
    fn randomized(&self) -> bool {
        matches!(
            self.method,
            ResamplingMethod::Bootstrap { .. }
                | ResamplingMethod::MonteCarlo { .. }
                | ResamplingMethod::Permutation {
                    mode: PermutationMode::Randomized,
                    ..
                }
        )
    }
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn analyze(
        &self,
        problem: ResamplingProblem<'_, S>,
        output: &mut [S],
        ws: &mut ResamplingWorkspace<S>,
        publication: OutputPolicy,
        context: &mut SolveContext<'_, '_>,
        run: &mut impl FnMut(
            bool,
            &mut [S],
            &mut [bool],
            &mut [Slot<S>],
            &mut [PartitionedSolve],
            &mut [S],
            &mut SolveContext<'_, '_>,
        ) -> Result<StatsTermination, SciOxideError>,
    ) -> Result<ResamplingResult<S>, SciOxideError> {
        let mut result = ResamplingResult {
            method: self.method,
            observed: None,
            requested: self.replicates,
            completed: 0,
            termination: StatsTermination::Completed,
            inference: None,
        };
        if let Some(stop) = stopped(context)? {
            result.termination = stop;
            return Ok(result);
        }
        self.identity(&mut ws.slots[0]);
        let observed = context.invoke_provider(EvaluationRole::Function, || match problem {
            ResamplingProblem::Indexed(statistic) => statistic(ws.slots[0].selection()),
            ResamplingProblem::IndexedWorkspace(statistic) => {
                let slot = &mut ws.slots[0];
                statistic(
                    ResampledIndices {
                        indices: &slot.indices[..slot.offsets[slot.offsets.len() - 1]],
                        offsets: &slot.offsets,
                    },
                    &mut slot.scratch,
                )
            }
            ResamplingProblem::MonteCarlo {
                observed,
                statistic,
                ..
            } => statistic(observed),
            ResamplingProblem::MonteCarloWorkspace {
                observed,
                statistic,
                ..
            } => statistic(observed, &mut ws.slots[0].scratch),
        })?;
        let scioxide_core::Invocation::Evaluated(observed) = observed else {
            result.termination = StatsTermination::BudgetExhausted;
            return Ok(result);
        };
        result.observed = Some(finite(observed)?);
        ws.completed.fill(false);
        result.termination = run(
            false,
            &mut ws.values,
            &mut ws.completed[..self.replicates],
            &mut ws.slots,
            &mut ws.children[..self.replicates],
            output,
            context,
        )?;
        result.completed = ws.completed[..self.replicates]
            .iter()
            .filter(|&&x| x)
            .count();
        if result.termination != StatsTermination::Completed {
            return Ok(result);
        }
        if result.completed != self.replicates {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "missing replicate callback",
            ));
        }
        if self.deletions > 0 {
            ws.completed.fill(false);
            result.termination = run(
                true,
                &mut ws.deletes,
                &mut ws.completed[..self.deletions],
                &mut ws.slots,
                &mut ws.children[..self.deletions],
                output,
                context,
            )?;
            if result.termination != StatsTermination::Completed {
                return Ok(result);
            }
            if ws.completed[..self.deletions].iter().any(|&x| !x) {
                return Err(invalid(
                    scioxide_core::argument!("problem"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing jackknife callback",
                ));
            }
        }
        if let Some(stop) = stopped(context)? {
            result.termination = stop;
            return Ok(result);
        }
        result.inference = Some(super::inference::infer(
            self,
            observed,
            &ws.values,
            &ws.deletes,
            &mut ws.sorted,
        )?);
        context.commit_accepted(PartialResultTrust::Admissible)?;
        if publication == OutputPolicy::Atomic {
            output.copy_from_slice(&ws.values);
        }
        Ok(result)
    }
    #[allow(clippy::too_many_arguments)]
    fn job(
        &self,
        problem: ResamplingProblem<'_, S>,
        delete: bool,
        index: usize,
        rng: &mut dyn BitGenerator,
        slot: &mut Slot<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        slot.value = None;
        slot.termination = StatsTermination::Completed;
        if let Some(stop) = stopped(context)? {
            slot.termination = stop;
            return Ok(());
        }
        if !context.record_proposal()?.is_granted() {
            slot.termination = StatsTermination::BudgetExhausted;
            return Ok(());
        }
        if delete {
            self.delete(index, slot);
        } else if self.randomized() {
            if !context
                .record_provider(EvaluationRole::Sample)?
                .is_granted()
            {
                slot.termination = StatsTermination::BudgetExhausted;
                return Ok(());
            }
            let mut limited = LimitedRng {
                rng,
                context,
                remaining: self.options.max_rng_calls,
                exhausted: false,
            };
            let sampled = match problem {
                ResamplingProblem::Indexed(_) | ResamplingProblem::IndexedWorkspace(_) => {
                    self.sample(index, &mut limited, slot)
                }
                ResamplingProblem::MonteCarlo { sampler, .. }
                | ResamplingProblem::MonteCarloWorkspace { sampler, .. } => {
                    // Poison scratch so a callback that forgets any output is rejected.
                    slot.sample.fill(S::NAN);
                    sampler(&mut limited, &mut slot.sample)
                }
            };
            let exhausted = limited.exhausted;
            if exhausted {
                slot.termination = StatsTermination::BudgetExhausted;
                return Ok(());
            }
            sampled?;
            if matches!(
                problem,
                ResamplingProblem::MonteCarlo { .. }
                    | ResamplingProblem::MonteCarloWorkspace { .. }
            ) {
                slot.sample
                    .iter()
                    .try_for_each(|&x| finite(x).map(|_| ()))?;
            }
        } else {
            self.sample(index, rng, slot)?;
        }
        if let Some(stop) = stopped(context)? {
            slot.termination = stop;
            return Ok(());
        }
        let value = context.invoke_provider(EvaluationRole::Function, || match problem {
            ResamplingProblem::Indexed(statistic) => statistic(slot.selection()),
            ResamplingProblem::IndexedWorkspace(statistic) => statistic(
                ResampledIndices {
                    indices: &slot.indices[..slot.offsets[slot.offsets.len() - 1]],
                    offsets: &slot.offsets,
                },
                &mut slot.scratch,
            ),
            ResamplingProblem::MonteCarlo { statistic, .. } => statistic(&slot.sample),
            ResamplingProblem::MonteCarloWorkspace { statistic, .. } => {
                statistic(&slot.sample, &mut slot.scratch)
            }
        })?;
        match value {
            scioxide_core::Invocation::Evaluated(value) => slot.value = Some(finite(value)?),
            scioxide_core::Invocation::ExecutionStopped(reason) => {
                slot.value = None;
                slot.termination = StatsTermination::Stopped(reason);
            }
            scioxide_core::Invocation::BudgetExhausted(_) => {
                slot.value = None;
                slot.termination = StatsTermination::BudgetExhausted;
            }
        }
        Ok(())
    }
}
fn stopped(context: &SolveContext<'_, '_>) -> Result<Option<StatsTermination>, SciOxideError> {
    context
        .stop_reason()
        .map(|s| s.map(StatsTermination::Stopped))
}
// Charge each native or uniform-word request once, including failed requests.
// Uniform-word normalization independently bounds its native rejection attempts.
struct LimitedRng<'a, 'b, 'c> {
    rng: &'a mut dyn BitGenerator,
    context: &'a mut SolveContext<'b, 'c>,
    remaining: usize,
    exhausted: bool,
}
impl LimitedRng<'_, '_, '_> {
    fn admit(&mut self) -> Result<(), SciOxideError> {
        if self.remaining == 0
            || !self
                .context
                .try_consume(BudgetKind::Sample, 1)?
                .is_granted()
        {
            self.exhausted = true;
            return Err(invalid(
                scioxide_core::argument!("self.options.max_rng_calls"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "replicate RNG-call budget exhausted",
            ));
        }
        self.remaining -= 1;
        Ok(())
    }
}
impl BitGenerator for LimitedRng<'_, '_, '_> {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.admit()?;
        self.rng.next_u32()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.admit()?;
        self.rng.next_u64()
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.admit()?;
        self.rng.next_uniform_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.admit()?;
        self.rng.next_uniform_u64()
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        for chunk in output.chunks_mut(8) {
            let word = self.next_u64()?.to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        Ok(())
    }
}
/// Generator for deterministic analyses. Every draw returns an `InvalidInput`
/// error.
pub(crate) struct NoRandom(pub(crate) scioxide_core::Argument);
impl BitGenerator for NoRandom {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Err(invalid(
            self.0,
            scioxide_core::InvalidInputReason::Inconsistent,
            "deterministic operation requested RNG",
        ))
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Err(invalid(
            self.0,
            scioxide_core::InvalidInputReason::Inconsistent,
            "deterministic operation requested RNG",
        ))
    }
    fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
        Err(invalid(
            self.0,
            scioxide_core::InvalidInputReason::Inconsistent,
            "deterministic operation requested RNG",
        ))
    }
}
impl<S: RealScalar> scioxide_random::Rng<S> for LimitedRng<'_, '_, '_> {
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        self.next_uniform_u64().map(scioxide_random::decode_u64)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn restricted_uniform_requests_preserve_both_budgets() {
        use scioxide_core::{Budget, ExecutionContext, SolveState};
        use scioxide_random::{Ranlux, Rng, SeedableRng};
        for (parent_limit, call_limit) in [(2, 3), (3, 2)] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Sample, Budget::try_from(parent_limit).unwrap());
            let identity = SolveIdentity::new(
                ProblemToken::new(1331),
                scioxide_core::algorithm_id!("stats", "resampling-test", 1),
                0,
                0,
            );
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                .unwrap();
            let mut rng = Ranlux::from_seed(42);
            let mut expected = rng.clone();
            let mut limited = LimitedRng {
                rng: &mut rng,
                context: &mut context,
                remaining: call_limit,
                exhausted: false,
            };
            assert_eq!(
                limited.next_uniform_u32().unwrap(),
                expected.next_uniform_u32().unwrap()
            );
            let actual: f64 = limited.next_float().unwrap();
            assert_eq!(
                actual.to_bits(),
                scioxide_random::decode_u64::<f64>(expected.next_uniform_u64().unwrap()).to_bits()
            );
            assert!(limited.next_uniform_u64().is_err());
            assert!(limited.exhausted);
            assert_eq!(limited.remaining, call_limit - 2);
            assert_eq!(context.budget_report(BudgetKind::Sample).consumed(), 2);
            assert_eq!(rng.position(), expected.position());
        }
    }
}
