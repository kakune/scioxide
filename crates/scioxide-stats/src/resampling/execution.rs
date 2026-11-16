//! Sequential and indexed random streams for resampling replicates.
use super::*;
use scioxide_core::{ExecutionPolicy, SolveContext};
use scioxide_random::{OutputPolicy, SamplingPolicy, SplittableRng, StatePolicy};

/// Random number source and output policy of a resampling run.
///
/// There are two modes. [`ReplicateOptions::sequential`] draws all replicates
/// in order from one generator. [`ReplicateOptions::indexed`] gives each
/// replicate its own stream split from a root generator, so the replicates can
/// run in parallel and the result does not depend on the number of workers.
///
/// # Examples
///
/// ```
/// use scioxide_random::{OutputPolicy, SamplingPolicy, SeedableRng, SplitMix64};
/// use scioxide_stats::ReplicateOptions;
///
/// let mut rng = SplitMix64::from_seed(7);
/// let sampling = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
/// let options = ReplicateOptions::sequential(&mut rng, sampling);
/// ```
pub struct ReplicateOptions<M = ()> {
    mode: M,
}
/// Sequential mode of [`ReplicateOptions`], from
/// [`ReplicateOptions::sequential`].
pub struct SequentialReplicates<'a, R, P> {
    rng: &'a mut R,
    sampling: SamplingPolicy<P>,
}
/// Indexed-stream mode of [`ReplicateOptions`], from
/// [`ReplicateOptions::indexed`].
pub struct IndexedReplicates<'a, R> {
    root: &'a R,
    first_stream: u64,
    policy: ExecutionPolicy,
    publication: OutputPolicy,
}
impl ReplicateOptions {
    /// Draws all replicates in order from `rng`.
    ///
    /// The output policy of `sampling` sets whether the output slice is written
    /// once at the end or replicate by replicate. Its state policy sets
    /// whether `rng` is restored when an error is returned.
    pub fn sequential<R: BitGenerator, P: StatePolicy<R>>(
        rng: &mut R,
        sampling: SamplingPolicy<P>,
    ) -> ReplicateOptions<SequentialReplicates<'_, R, P>> {
        ReplicateOptions {
            mode: SequentialReplicates { rng, sampling },
        }
    }
    /// Gives replicate `i` the stream `root.split(first_stream + i)`, and runs
    /// the replicates as `policy` allows.
    ///
    /// `root` does not advance, and the result does not depend on the number
    /// of workers. Exact enumeration and the jackknife deletions of BCa use no
    /// stream. `publication` sets whether the output slice is written once at
    /// the end or replicate by replicate. [`ResamplingPlan::evaluate`] returns
    /// `SizeOverflow` if `first_stream + replicates − 1` overflows `u64`.
    pub fn indexed<R: BitGenerator + SplittableRng + Sync>(
        root: &R,
        first_stream: u64,
        policy: ExecutionPolicy,
        publication: OutputPolicy,
    ) -> ReplicateOptions<IndexedReplicates<'_, R>> {
        ReplicateOptions {
            mode: IndexedReplicates {
                root,
                first_stream,
                policy,
                publication,
            },
        }
    }
}
/// Replicate execution mode accepted by [`ResamplingPlan::evaluate`].
///
/// Only the values of [`ReplicateOptions::sequential`] and
/// [`ReplicateOptions::indexed`] implement it.
pub trait ReplicateExecution<S: RealScalar>: sealed::Run<S> {}
impl<S: RealScalar, T: sealed::Run<S>> ReplicateExecution<S> for T {}

mod sealed {
    use super::*;
    pub trait Run<S: RealScalar> {
        fn run(
            self,
            plan: &ResamplingPlan<S>,
            problem: ResamplingProblem<'_, S>,
            output: &mut [S],
            workspace: &mut ResamplingWorkspace<S>,
            context: &mut SolveContext<'_, '_>,
        ) -> Result<ResamplingResult<S>, SciOxideError>;
    }
    impl<S: RealScalar, R: BitGenerator, P: StatePolicy<R>> Run<S>
        for ReplicateOptions<SequentialReplicates<'_, R, P>>
    {
        fn run(
            self,
            plan: &ResamplingPlan<S>,
            problem: ResamplingProblem<'_, S>,
            output: &mut [S],
            workspace: &mut ResamplingWorkspace<S>,
            context: &mut SolveContext<'_, '_>,
        ) -> Result<ResamplingResult<S>, SciOxideError> {
            plan.evaluate_sequential(
                problem,
                self.mode.rng,
                output,
                workspace,
                self.mode.sampling,
                context,
            )
        }
    }
    impl<S: RealScalar, R: BitGenerator + SplittableRng + Sync> Run<S>
        for ReplicateOptions<IndexedReplicates<'_, R>>
    {
        fn run(
            self,
            plan: &ResamplingPlan<S>,
            problem: ResamplingProblem<'_, S>,
            output: &mut [S],
            workspace: &mut ResamplingWorkspace<S>,
            context: &mut SolveContext<'_, '_>,
        ) -> Result<ResamplingResult<S>, SciOxideError> {
            plan.evaluate_streams(
                problem,
                self.mode.root,
                self.mode.first_stream,
                self.mode.policy,
                output,
                workspace,
                self.mode.publication,
                context,
            )
        }
    }
}
impl<S: RealScalar> ResamplingPlan<S> {
    /// Runs the analysis of `problem`: the observed statistic, the replicate
    /// statistics, which go to `output`, and the inference.
    ///
    /// `output` has [`Self::replicates`] elements, and `workspace` comes from
    /// [`Self::workspace`] of this plan. `replicates` selects the random
    /// streams. The child epoch counts statistic calls and generator draws toward
    /// every ancestor shared budget ([solve accounting](scioxide_core#solve-accounting)).
    /// Replicates are limited by this plan; ancestor proposal and iteration limits
    /// do not cap the child epoch.
    /// A budget, cancellation or deadline that stops the run gives `Ok` with
    /// an unsuccessful termination and no inference. With replicate-by-replicate
    /// output, the completed replicates are then already in `output`. For BCa
    /// intervals, the jackknife deletions also call the statistic, but their
    /// values are not written to `output`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output` or `workspace` does not fit the plan.
    /// - `ResourceLimit` if the workspace exceeds
    ///   [`ResamplingOptions::max_workspace_bytes`].
    /// - `InvalidInput` if `problem` does not fit the method, or the observed
    ///   Monte Carlo sample has the wrong length.
    /// - `NonFinite` if the observed sample, a null sample or a statistic is
    ///   NaN or infinite.
    /// - `SizeOverflow` if an indexed stream number overflows `u64`.
    /// - `NumericalInstability` if a BCa interval cannot be formed, for example
    ///   when all jackknife statistics are equal.
    /// - Errors from the callbacks, the generator and the executor are returned
    ///   unchanged.
    pub fn evaluate_into<'scope, 'capability>(
        &self,
        problem: ResamplingProblem<'_, S>,
        replicates: impl ReplicateExecution<S>,
        output: &mut [S],
        workspace: &mut ResamplingWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ResamplingResult<S>, SciOxideError> {
        workspace.solve.reject();
        self.validate(problem, output, workspace)?;
        let mut slot = core::mem::take(&mut workspace.solve);
        let result = slot.run(
            scope.into(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("stats", "resampling", 1),
                self.replicates() as u128,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |context| replicates.run(self, problem, output, workspace, context),
            |result| {
                (
                    match result {
                        Ok(report) if report.termination().is_success() => {
                            scioxide_core::AttemptOutcome::Succeeded
                        }
                        Ok(_) => scioxide_core::AttemptOutcome::Stopped,
                        Err(_) => scioxide_core::AttemptOutcome::Failed,
                    },
                    None,
                )
            },
        );
        workspace.solve = slot;
        result
    }
}
