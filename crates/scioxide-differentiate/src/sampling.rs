//! Callback forms, and the sampling loop that calls them and counts evaluations.

use crate::{DifferentiationReport, DifferentiationTermination};
use core::num::NonZeroUsize;
use scioxide_core::backend::{
    PartitionSlot, PartitionedSolve, checked_add, with_partitioned_solves,
};
use scioxide_core::{
    DifferentiationLocation, EvaluationRole, ExecutionPolicy, ExecutionStopReason, NonFiniteOrigin,
    NumericScalar, RealScalar, SciOxideError, SolveBudgets, SolveContext,
};
/// A function `f: ℝⁿ → ℝᵐ` (or `ℂᵐ`) that a method or plan can sample.
///
/// Three forms implement it, and no others can:
///
/// - a closure `FnMut(&[S], &mut [T]) -> Result<(), SciOxideError>`, called on
///   the calling thread;
/// - [`Contextual`], which also receives the solve context;
/// - [`Parallel`], whose independent samples run on the executor.
///
/// The callback receives the point `x` and must write all `m` values, each
/// finite, into the output slice.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideError;
/// use scioxide_differentiate::{FiniteDifference, SampleFunction};
///
/// fn slope_at_2(f: impl SampleFunction<f64, f64>) -> Result<f64, SciOxideError> {
///     let gradient = FiniteDifference::default().gradient(f, &[2.0], None, None)?;
///     Ok(gradient.into_values()?[0])
/// }
/// // f(x) = x², so f′(2) = 4.
/// let slope = slope_at_2(|x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0];
///     Ok(())
/// })?;
/// assert!((slope - 4.0).abs() < 1e-8);
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait SampleFunction<S: Send + Sync, T: scioxide_core::NumericScalar>: Samples<S, T> {}
impl<S: Send + Sync, T: scioxide_core::NumericScalar, F: Samples<S, T>> SampleFunction<S, T> for F {}

/// Wraps a callback that also receives the [`SolveContext`](scioxide_core::SolveContext)
/// of the evaluation.
///
/// Use it when the callback runs a nested solve or does work that should count
/// toward the same [solve budgets](scioxide_core#solve-accounting), or when it
/// needs the execution context. Calls are made on the calling thread.
///
/// # Examples
///
/// ```
/// use scioxide_core::{SciOxideError, SolveContext};
/// use scioxide_differentiate::{Contextual, FiniteDifference};
///
/// let f = |x: &[f64], y: &mut [f64], context: &mut SolveContext<'_, '_>| {
///     // Count one iteration of work done inside the callback.
///     assert!(context.record_iteration()?.is_granted());
///     y[0] = x[0] * x[0];
///     Ok(())
/// };
/// let result = FiniteDifference::default().gradient(Contextual(f), &[2.0], None, None)?;
/// assert!((result.values().unwrap()[0] - 4.0).abs() < 1e-8); // f′(2) = 4
/// # Ok::<(), SciOxideError>(())
/// ```
pub struct Contextual<F>(
    /// The callback `FnMut(&[S], &mut [T], &mut SolveContext) -> Result<(), SciOxideError>`.
    pub F,
);
/// Wraps a `Fn + Sync` callback whose independent samples may run in parallel.
///
/// Samples are distributed over the executor of the
/// [execution context](scioxide_core#execution-contexts) according to the
/// [`ExecutionPolicy`](scioxide_core::ExecutionPolicy) of the method or plan.
/// With `Sequential`, or with `Auto` when one worker suffices, they run on the
/// calling thread.
/// `Parallel(n)` always uses the executor, even for one sample. `Auto` counts
/// one work unit per sample and uses at most sixteen workers.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, ExecutionPolicy, SciOxideError};
/// use scioxide_differentiate::{DerivativeRequest, DerivativeShape, FiniteDifference};
/// use scioxide_differentiate::{FiniteDifferenceWorkspace, Parallel};
///
/// let method = FiniteDifference::<f64>::default().with_execution(ExecutionPolicy::Auto);
/// let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
/// let (mut workspace, mut slope) = (FiniteDifferenceWorkspace::workspace(request)?, [0.0]);
/// let f = Parallel(|x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0];
///     Ok(())
/// });
/// let context = ExecutionContext::sequential();
/// let report = method.gradient_into(f, &[2.0], None, None, &mut slope, &mut workspace, context)?;
/// assert!(report.output_available() && (slope[0] - 4.0).abs() < 1e-8); // f′(2) = 4
/// # Ok::<(), SciOxideError>(())
/// ```
pub struct Parallel<F>(
    /// The callback `Fn(&[S], &mut [T]) -> Result<(), SciOxideError> + Sync`.
    pub F,
);

// Shared sequential/executor stencil sampling. Partition children own callback budgets.

pub trait Samples<S: Send + Sync, T: NumericScalar> {
    #[expect(
        clippy::too_many_arguments,
        reason = "numerical inputs, caller-owned staging and execution capabilities stay explicit"
    )]
    fn evaluate(
        &mut self,
        points: &[S],
        values: &mut [T],
        active: &[bool],
        inputs: usize,
        outputs: usize,
        context: &mut SolveContext<'_, '_>,
        report: &mut DifferentiationReport,
        location: DifferentiationLocation,
        policy: ExecutionPolicy,
    ) -> Result<Option<DifferentiationTermination>, SciOxideError>;
}

impl<S, T, F> Samples<S, T> for F
where
    S: Send + Sync,
    T: NumericScalar,
    F: FnMut(&[S], &mut [T]) -> Result<(), SciOxideError>,
{
    fn evaluate(
        &mut self,
        points: &[S],
        values: &mut [T],
        active: &[bool],
        inputs: usize,
        outputs: usize,
        context: &mut SolveContext<'_, '_>,
        report: &mut DifferentiationReport,
        location: DifferentiationLocation,
        _policy: ExecutionPolicy,
    ) -> Result<Option<DifferentiationTermination>, SciOxideError> {
        for (index, &run) in active.iter().enumerate() {
            if run
                && let Some(reason) = sample_contextual(
                    &points[index * inputs..(index + 1) * inputs],
                    &mut values[index * outputs..(index + 1) * outputs],
                    &mut |x, y, _| self(x, y),
                    location,
                    context,
                    &mut report.evaluations,
                )?
            {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }
}

impl<S, T, F> Samples<S, T> for Contextual<F>
where
    S: Send + Sync,
    T: NumericScalar,
    F: FnMut(&[S], &mut [T], &mut SolveContext<'_, '_>) -> Result<(), SciOxideError>,
{
    fn evaluate(
        &mut self,
        points: &[S],
        values: &mut [T],
        active: &[bool],
        inputs: usize,
        outputs: usize,
        context: &mut SolveContext<'_, '_>,
        report: &mut DifferentiationReport,
        location: DifferentiationLocation,
        _policy: ExecutionPolicy,
    ) -> Result<Option<DifferentiationTermination>, SciOxideError> {
        for (index, &run) in active.iter().enumerate() {
            if run
                && let Some(reason) = sample_contextual(
                    &points[index * inputs..(index + 1) * inputs],
                    &mut values[index * outputs..(index + 1) * outputs],
                    &mut self.0,
                    location,
                    context,
                    &mut report.evaluations,
                )?
            {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }
}

impl<S, T, F> Samples<S, T> for Parallel<F>
where
    S: Send + Sync,
    T: NumericScalar,
    F: Fn(&[S], &mut [T]) -> Result<(), SciOxideError> + Sync,
{
    fn evaluate(
        &mut self,
        points: &[S],
        values: &mut [T],
        active: &[bool],
        inputs: usize,
        outputs: usize,
        context: &mut SolveContext<'_, '_>,
        report: &mut DifferentiationReport,
        location: DifferentiationLocation,
        policy: ExecutionPolicy,
    ) -> Result<Option<DifferentiationTermination>, SciOxideError> {
        if let Some(reason) = stop(context)? {
            return Ok(Some(reason));
        }
        let count = active.iter().filter(|&&run| run).count();
        if count == 0 {
            return Ok(None);
        }
        let plan = context.execution().plan(
            policy,
            count,
            scioxide_core::PlanLimits {
                work_per_item: NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: NonZeroUsize::MIN.saturating_add(15),
            },
        );
        if !matches!(policy, ExecutionPolicy::Parallel(_)) && plan.worker_count() == 1 {
            // An inline level needs no detached children or partition protocol.
            // Still visit the level after a provider error, as the partitioned
            // path does, and report the first sample's error after counting all
            // admitted attempts. Parallel callbacks cannot start contextual work.
            let mut first_error = None;
            let mut termination = None;
            for (index, &run) in active.iter().enumerate() {
                if !run {
                    continue;
                }
                let mut evaluations = 0;
                let outcome = sample_contextual(
                    &points[index * inputs..(index + 1) * inputs],
                    &mut values[index * outputs..(index + 1) * outputs],
                    &mut |x, y, _| (self.0)(x, y),
                    location,
                    context,
                    &mut evaluations,
                );
                report.evaluations = checked_add(
                    report.evaluations,
                    evaluations,
                    scioxide_core::quantity!("function_evaluations"),
                )?;
                match outcome {
                    Err(error) if first_error.is_none() => first_error = Some(error),
                    Ok(Some(reason)) if termination.is_none() => termination = Some(reason),
                    _ => {}
                }
            }
            if let Some(error) = first_error {
                return Err(SciOxideError::differentiation_failure(
                    location,
                    report.evaluations,
                    error,
                ));
            }
            return Ok(stop(context)?.or(termination));
        }
        // Each active sample runs in its own child solve, which admits and counts
        // its callback against a share of the parent budget. A level has at most
        // 16 samples, so fixed arrays keep dispatch allocation-free.
        let mut children = core::array::from_fn::<_, 16, _>(|_| PartitionedSolve::default());
        let mut outcomes = core::array::from_fn::<_, 16, _>(|_| None);
        let identity = context.identity();
        let dispatched = with_partitioned_solves(
            context,
            &mut children[..count],
            SolveBudgets::unbounded(),
            |execution, children| {
                let mut slots: [_; 16] = core::array::from_fn(|_| None);
                // Split staging into disjoint slices, then pair each active sample
                // with its child permit and outcome. All three have `count` entries;
                // the tuple cannot contain a sample without its resources.
                let mut remaining = values;
                let samples = active.iter().enumerate().filter_map(|(index, &run)| {
                    let (value, tail) = core::mem::take(&mut remaining).split_at_mut(outputs);
                    remaining = tail;
                    run.then_some((index, value))
                });
                let resources = children.into_iter().zip(outcomes.iter_mut());
                for (slot, ((index, value), (child, outcome))) in
                    slots.iter_mut().zip(samples.zip(resources))
                {
                    *slot = Some((index, value, child, outcome));
                }
                let mut remaining = &mut slots[..count];
                let mut partitions = plan.partitions();
                let partition_slots: [_; 16] = core::array::from_fn(|_| {
                    partitions
                        .next()
                        .map_or_else(PartitionSlot::empty, |partition| {
                            let (entries, tail) = core::mem::take(&mut remaining)
                                .split_at_mut(partition.range().len());
                            remaining = tail;
                            PartitionSlot::new(entries)
                        })
                });
                execution.execute_partitioned(
                    plan,
                    &partition_slots[..plan.worker_count()],
                    &|_, entries, scope| {
                        for (sample, value, child, outcome) in
                            entries.iter_mut().filter_map(Option::take)
                        {
                            let mut evaluations = 0;
                            *outcome = Some(child.run(scope, identity, |ctx| {
                                sample_contextual(
                                    &points[sample * inputs..(sample + 1) * inputs],
                                    value,
                                    &mut |x, y, _| (self.0)(x, y),
                                    location,
                                    ctx,
                                    &mut evaluations,
                                )
                            }));
                        }
                        Ok(())
                    },
                )
            },
        );
        for child in &children[..count] {
            report.evaluations = checked_add(
                report.evaluations,
                child.counts().own().provider(EvaluationRole::Function),
                scioxide_core::quantity!("function_evaluations"),
            )?;
        }
        // Every launched callback is accounted for even when the executor fails.
        let execution_stop = match dispatched {
            Ok(()) => None,
            Err(error) => match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
                    Some(match reason {
                        ExecutionStopReason::Cancelled => DifferentiationTermination::Cancelled,
                        ExecutionStopReason::DeadlineReached => {
                            DifferentiationTermination::DeadlineReached
                        }
                        _ => return Err(error),
                    })
                }
                _ => return Err(error),
            },
        };
        let mut termination = execution_stop;
        // Outcomes are scanned in sample order, not completion order, so the first
        // reported termination and error are deterministic under any scheduling.
        // Core checks partition completion before successful dispatch returns.
        // Absence is legitimate only for samples skipped by a stopped executor.
        for outcome in outcomes[..count].iter_mut().filter_map(Option::take) {
            match outcome {
                Err(error) => {
                    return Err(SciOxideError::differentiation_failure(
                        location,
                        report.evaluations,
                        error,
                    ));
                }
                Ok(Some(reason)) if termination.is_none() => termination = Some(reason),
                _ => {}
            }
        }
        Ok(stop(context)?.or(termination))
    }
}

pub(super) fn stop(
    context: &SolveContext<'_, '_>,
) -> Result<Option<DifferentiationTermination>, SciOxideError> {
    match crate::sampling::publish(context, || {})? {
        None => Ok(None),
        Some(ExecutionStopReason::Cancelled) => Ok(Some(DifferentiationTermination::Cancelled)),
        Some(ExecutionStopReason::DeadlineReached) => {
            Ok(Some(DifferentiationTermination::DeadlineReached))
        }
        Some(reason) => Err(SciOxideError::execution_stopped(reason)),
    }
}

impl DifferentiationReport {
    #[inline]
    pub(super) fn poll(
        &mut self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        if let Some(reason) = publish(context, || {})? {
            self.termination = match reason {
                scioxide_core::ExecutionStopReason::Cancelled => {
                    DifferentiationTermination::Cancelled
                }
                scioxide_core::ExecutionStopReason::DeadlineReached => {
                    DifferentiationTermination::DeadlineReached
                }
                reason => return Err(SciOxideError::execution_stopped(reason)),
            };
            return Ok(false);
        }
        Ok(true)
    }
    #[inline]
    pub(super) fn iteration(
        &mut self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        if !self.poll(context)? {
            return Ok(false);
        }
        if !context.record_iteration()?.is_granted() {
            self.termination = DifferentiationTermination::IterationLimit;
            return Ok(false);
        }
        self.iterations = checked_add(
            self.iterations,
            1,
            scioxide_core::quantity!("differentiation_iterations"),
        )?;
        Ok(true)
    }

    #[inline]
    pub(super) fn invoke<X: Send + Sync, T: NumericScalar, F: SampleFunction<X, T>>(
        &mut self,
        point: &[X],
        values: &mut [T],
        function: &mut F,
        location: DifferentiationLocation,
        context: &mut scioxide_core::SolveContext<'_, '_>,
        policy: scioxide_core::ExecutionPolicy,
    ) -> Result<bool, SciOxideError> {
        match function.evaluate(
            point,
            values,
            &[true],
            point.len(),
            values.len(),
            context,
            self,
            location,
            policy,
        )? {
            None => Ok(true),
            Some(stop) => {
                self.termination = stop;
                Ok(false)
            }
        }
    }
}
// One callback admission, poisoning, accounting and failure-provenance owner.
pub(super) fn sample_contextual<X, T: NumericScalar>(
    point: &[X],
    values: &mut [T],
    function: &mut impl FnMut(
        &[X],
        &mut [T],
        &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>,
    location: DifferentiationLocation,
    context: &mut scioxide_core::SolveContext<'_, '_>,
    evaluations: &mut usize,
) -> Result<Option<DifferentiationTermination>, SciOxideError> {
    if let Some(reason) = context.stop_reason()? {
        return Ok(Some(match reason {
            scioxide_core::ExecutionStopReason::Cancelled => DifferentiationTermination::Cancelled,
            scioxide_core::ExecutionStopReason::DeadlineReached => {
                DifferentiationTermination::DeadlineReached
            }
            reason => return Err(SciOxideError::execution_stopped(reason)),
        }));
    }
    if !context
        .record_provider(EvaluationRole::Function)?
        .is_granted()
    {
        return Ok(Some(DifferentiationTermination::EvaluationLimit));
    }
    *evaluations = checked_add(
        *evaluations,
        1,
        scioxide_core::quantity!("function_evaluations"),
    )?;
    values.fill(T::from_real(T::Real::NAN));
    function(point, values, context)
        .map_err(|error| SciOxideError::differentiation_failure(location, *evaluations, error))?;
    if let Some(lane) = values
        .iter()
        .position(|value| !value.has_finite_components())
    {
        return Err(SciOxideError::differentiation_failure(
            location.with_component(lane),
            *evaluations,
            SciOxideError::non_finite(
                NonFiniteOrigin::Callback,
                "callback must overwrite every lane with a finite value",
            ),
        ));
    }
    Ok(None)
}
/// Checks for cancellation and deadlines once more, then writes the output.
/// Applies also to empty results.
pub(super) fn publish(
    context: &scioxide_core::SolveContext<'_, '_>,
    commit: impl FnOnce(),
) -> Result<Option<scioxide_core::ExecutionStopReason>, SciOxideError> {
    let stop = context.stop_reason()?;
    if stop.is_none() {
        commit();
    }
    Ok(stop)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{DerivativeSource, DifferenceScheme};
    use alloc::vec::Vec;
    use scioxide_core::{
        CancellationToken, ExecutionContext, ExecutionFailureReason, ExecutionPlan, Executor,
        PartitionFn, ProblemToken, SciOxideErrorView, SolveIdentity, SolveState,
    };

    struct ReverseExecutor {
        omit: bool,
    }
    impl Executor for ReverseExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::MIN.saturating_add(3)
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            function: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            if !self.omit {
                for partition in plan.partitions().collect::<Vec<_>>().into_iter().rev() {
                    function(partition, (), context)?;
                }
            }
            Ok(())
        }
    }
    fn identity() -> SolveIdentity {
        SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "sampling", 1),
            0,
            0,
        )
    }

    #[test]
    fn inline_levels_preserve_failure_counts_and_stop_before_the_next_sample() {
        let cancellation = CancellationToken::new();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&cancellation),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let mut report = DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(DifferenceScheme::Central),
            3,
        );
        let mut values = [-1.0; 3];
        let error = Parallel(|x: &[f64], y: &mut [f64]| {
            if x[0] == 0.0 {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("x"),
                    scioxide_core::DomainReason::OutsideDomain,
                    "injected failure",
                ));
            }
            y[0] = x[0];
            Ok(())
        })
        .evaluate(
            &[0.0, 1.0, 2.0],
            &mut values,
            &[true; 3],
            1,
            1,
            &mut context,
            &mut report,
            DifferentiationLocation::base_point(),
            ExecutionPolicy::Sequential,
        )
        .unwrap_err();
        assert_eq!(report.evaluations(), 3);
        assert_eq!(values[1..], [1.0, 2.0]);
        assert!(
            matches!(error.view(), SciOxideErrorView::DifferentiationFailure { cause, .. }
            if matches!(cause.view(), SciOxideErrorView::DomainError {  .. }))
        );
        let termination = Parallel(|_: &[f64], y: &mut [f64]| {
            y[0] = 1.0;
            cancellation.cancel();
            Ok(())
        })
        .evaluate(
            &[0.0, 1.0, 2.0],
            &mut values,
            &[true; 3],
            1,
            1,
            &mut context,
            &mut report,
            DifferentiationLocation::base_point(),
            ExecutionPolicy::Sequential,
        )
        .unwrap();
        assert_eq!(termination, Some(DifferentiationTermination::Cancelled));
        assert_eq!(report.evaluations(), 4);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            4
        );
    }

    #[test]
    fn active_samples_keep_their_resources_under_reordered_completion() {
        let executor = ReverseExecutor { omit: false };
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&executor), identity(), &mut budgets)
            .unwrap();
        let mut report = DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(DifferenceScheme::Central),
            4,
        );
        let mut values = [-1.0; 4];
        let termination = Parallel(|point: &[f64], value: &mut [f64]| {
            value[0] = point[0] * point[0];
            Ok(())
        })
        .evaluate(
            &[0.0, 1.0, 2.0, 3.0],
            &mut values,
            &[true, false, true, true],
            1,
            1,
            &mut context,
            &mut report,
            DifferentiationLocation::base_point(),
            ExecutionPolicy::Parallel(NonZeroUsize::MIN.saturating_add(1)),
        )
        .unwrap();
        assert_eq!(termination, None);
        assert_eq!(values, [0.0, -1.0, 4.0, 9.0]);
        assert_eq!(report.evaluations(), 3);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            3
        );

        let error = Parallel(|_: &[f64], _: &mut [f64]| {
            Err(SciOxideError::domain_error(
                scioxide_core::argument!("point"),
                scioxide_core::DomainReason::OutsideDomain,
                "injected callback failure",
            ))
        })
        .evaluate(
            &[0.0, 1.0, 2.0, 3.0],
            &mut values,
            &[true, false, true, true],
            1,
            1,
            &mut context,
            &mut report,
            DifferentiationLocation::base_point(),
            ExecutionPolicy::Parallel(NonZeroUsize::MIN.saturating_add(1)),
        )
        .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::DifferentiationFailure { cause, .. }
            if matches!(cause.view(), SciOxideErrorView::DomainError {  .. }))
        );
        assert_eq!(report.evaluations(), 6);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            6
        );
    }

    #[test]
    fn omitted_partitions_are_executor_failures_and_stops_allow_absent_outcomes() {
        let executor = ReverseExecutor { omit: true };
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&executor), identity(), &mut budgets)
            .unwrap();
        let mut report = DifferentiationReport::fixed(
            DerivativeSource::FiniteDifference(DifferenceScheme::Central),
            2,
        );
        let mut values = [-1.0; 2];
        let error = Parallel(|_: &[f64], _: &mut [f64]| Ok(()))
            .evaluate(
                &[0.0, 1.0],
                &mut values,
                &[true, true],
                1,
                1,
                &mut context,
                &mut report,
                DifferentiationLocation::base_point(),
                ExecutionPolicy::Parallel(NonZeroUsize::MIN),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionFailure {
                reason: ExecutionFailureReason::MissingPartition,
                ..
            }
        ));
        assert_eq!(report.evaluations(), 0);
        assert_eq!(values, [-1.0; 2]);

        let cancellation = CancellationToken::new();
        let mut state = SolveState::new();
        let executor = ReverseExecutor { omit: false };
        let mut context = state
            .begin_solve(
                ExecutionContext::new(&executor).with_cancellation(&cancellation),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let termination = Parallel(|point: &[f64], value: &mut [f64]| {
            value[0] = point[0];
            cancellation.cancel();
            Ok(())
        })
        .evaluate(
            &[0.0, 1.0],
            &mut values,
            &[true, true],
            1,
            1,
            &mut context,
            &mut report,
            DifferentiationLocation::base_point(),
            ExecutionPolicy::Parallel(NonZeroUsize::MIN.saturating_add(1)),
        )
        .unwrap();
        assert_eq!(termination, Some(DifferentiationTermination::Cancelled));
        assert_eq!(report.evaluations(), 1);
    }
}
