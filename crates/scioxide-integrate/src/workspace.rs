//! Workspace requirements, capacity checks and records of the last call.

use scioxide_core::{
    AlgorithmId, AttemptKind, AttemptOutcome, AttemptRecord, Capacity, ConvergenceDiagnostics,
    ConvergenceReason, EvaluationRole, ProblemToken, RealScalar, SciOxideError, SolveBudgets,
    SolveContext, SolveCounts, SolveIdentity, SolveScope, SolveState,
};

use crate::report::{IntegrationErrorComponents, IntegrationReport, IntegrationTermination};

/// Workspace family that produced a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    FixedRule,
    Adaptive,
    Romberg,
    DoubleExponential,
    MonteCarlo,
    Weighted,
    FourierTail,
    Oscillatory,
    Cubature,
    Product,
    Iterated,
    SparseGrid,
    Geometric,
}

/// Storage a plan needs in its workspace.
///
/// A plan returns these from `workspace_requirements`, and only the workspace type
/// of the same plan type accepts them. A workspace serves one number of real value
/// components and, for multidimensional plans, one dimension at a time.
/// `check` returns [`Capacity::NeedsGrowth`] when either differs or when storage is
/// too small. `try_reserve` keeps the larger of the current and requested capacities, so one
/// workspace can serve several plans of the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct IntegrationRequirements {
    family: Family,
    components: usize,
    dimension: usize,
    units: [usize; 2],
}

impl IntegrationRequirements {
    pub(crate) const fn new(family: Family, components: usize, units: [usize; 2]) -> Self {
        Self {
            family,
            components,
            dimension: 0,
            units,
        }
    }

    /// The same requirements for coordinates of `dimension`, served exactly.
    pub(crate) const fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Coordinate dimension served by the workspace; zero for one-dimensional families.
    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of real value components: 1 for real, 2 for complex, and
    /// the length for vector values.
    #[must_use]
    pub const fn components(&self) -> usize {
        self.components
    }

    pub(crate) const fn units(&self) -> [usize; 2] {
        self.units
    }

    /// Classify `requested` against the stored capacity `self`.
    pub(crate) fn capacity(&self, requested: &Self) -> Result<Capacity, SciOxideError> {
        if self.family != requested.family {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("requirements"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "requirements belong to another integration family",
            ));
        }
        let sufficient = self.components == requested.components
            && self.dimension == requested.dimension
            && self
                .units
                .iter()
                .zip(requested.units)
                .all(|(&retained, requested)| retained >= requested);
        Ok(if sufficient {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Requested component count with the larger of both unit capacities.
    pub(crate) fn grown(&self, requested: &Self) -> Self {
        Self {
            family: requested.family,
            components: requested.components,
            dimension: requested.dimension,
            units: core::array::from_fn(|index| self.units[index].max(requested.units[index])),
        }
    }
}

/// Generate the shared requirements and attempt accessors for a workspace with
/// `requirements`, `attempts` fields and a fallible `allocate(&requirements)`.
macro_rules! workspace_protocol {
    ($workspace:ident, $progress:ty) => {
        impl<S: scioxide_core::RealScalar> $workspace<S> {
            /// Returns whether the workspace satisfies `requirements` without growing.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `requirements` come from a different plan type.
            pub fn check(
                &self,
                requirements: &crate::IntegrationRequirements,
            ) -> Result<scioxide_core::Capacity, scioxide_core::SciOxideError> {
                self.requirements.capacity(requirements)
            }

            /// Grows the workspace to satisfy `requirements`.
            ///
            /// The record of the last call is kept. On error, the workspace is unchanged.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `requirements` come from a different plan type.
            /// - `AllocationFailure` or `SizeOverflow` if the storage cannot be reserved.
            pub fn try_reserve(
                &mut self,
                requirements: &crate::IntegrationRequirements,
            ) -> Result<(), scioxide_core::SciOxideError> {
                if self.check(requirements)? == scioxide_core::Capacity::NeedsGrowth {
                    let mut grown = Self::allocate(&self.requirements.grown(requirements))?;
                    core::mem::swap(&mut grown.attempts, &mut self.attempts);
                    *self = grown;
                }
                Ok(())
            }

            /// Returns a record of the last call with this workspace: its outcome, its
            /// evaluation counts and its report, or `None` before the first call.
            #[must_use]
            pub fn last_attempt(&self) -> Option<scioxide_core::AttemptRecord<$progress>> {
                self.attempts.last_attempt()
            }
        }
    };
}
pub(crate) use workspace_protocol;

/// Storage policy for crate-owned immutable attempt progress.
pub(crate) trait AttemptProgress: Copy {
    fn successful(&self) -> bool;
    #[inline]
    fn outcome(&self) -> AttemptOutcome {
        if self.successful() {
            AttemptOutcome::Succeeded
        } else {
            AttemptOutcome::Stopped
        }
    }
    fn store(storage: &mut Option<Self>, current: Option<&Self>) {
        *storage = current.copied();
    }
}

impl AttemptProgress for crate::SeriesReport {
    fn successful(&self) -> bool {
        self.termination().is_success()
    }
}
impl AttemptProgress for crate::SensitivityReport {
    fn successful(&self) -> bool {
        self.termination().is_success()
    }
}
impl AttemptProgress for crate::TrainingReport {
    fn successful(&self) -> bool {
        self.termination().is_success()
    }
}

/// Workspace-owned solve state and the record of its last attempt.
#[derive(Debug)]
pub(crate) struct Attempts<P: AttemptProgress> {
    state: SolveState,
    attempt: Option<Attempt<P>>,
}

/// Counts and epoch live in `SolveState`; this stores only per-call data.
#[derive(Debug, Clone, Copy)]
struct Attempt<P> {
    kind: AttemptKind,
    outcome: AttemptOutcome,
    progress: Option<P>,
}

impl<P: AttemptProgress> Default for Attempts<P> {
    fn default() -> Self {
        Self {
            state: SolveState::new(),
            attempt: None,
        }
    }
}

impl<P: AttemptProgress> Attempts<P> {
    #[inline(always)]
    fn record(&mut self, kind: AttemptKind, outcome: AttemptOutcome, progress: Option<&P>) {
        Self::record_into(&mut self.attempt, kind, outcome, progress);
    }

    #[inline(always)]
    fn record_into(
        storage: &mut Option<Attempt<P>>,
        kind: AttemptKind,
        outcome: AttemptOutcome,
        progress: Option<&P>,
    ) {
        let attempt = storage.get_or_insert(Attempt {
            kind,
            outcome,
            progress: None,
        });
        attempt.kind = kind;
        attempt.outcome = outcome;
        P::store(&mut attempt.progress, progress);
    }
    pub(crate) fn last_attempt(&self) -> Option<AttemptRecord<P>> {
        self.attempt.map(|attempt| {
            AttemptRecord::new(
                self.state.epoch(),
                attempt.kind,
                attempt.outcome,
                self.state.counts(),
                attempt.progress,
            )
        })
    }

    /// Record a validation failure that precedes any epoch, then return it.
    pub(crate) fn reject<T>(&mut self, error: SciOxideError) -> Result<T, SciOxideError> {
        self.reject_kind(AttemptKind::Call, error)
    }

    pub(crate) fn reject_kind<T>(
        &mut self,
        kind: AttemptKind,
        error: SciOxideError,
    ) -> Result<T, SciOxideError> {
        self.record(kind, AttemptOutcome::PreflightRejected, None);
        Err(error)
    }

    /// Run `execute` in a fresh local epoch and record its classified outcome.
    #[inline(always)]
    pub(crate) fn run<'capability>(
        &mut self,
        scope: SolveScope<'_, 'capability>,
        identity: SolveIdentity,
        budgets: SolveBudgets,
        execute: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<P, SciOxideError>,
    ) -> Result<P, SciOxideError> {
        let epoch = self.state.epoch();
        let result = scope.run(&mut self.state, identity, budgets, execute);
        let started = self.state.epoch() != epoch;
        let (outcome, progress) = match (&result, started) {
            (_, false) => (AttemptOutcome::PreflightRejected, None),
            (Ok(value), true) => (value.outcome(), Some(value)),
            (Err(_), true) => (AttemptOutcome::Failed, None),
        };
        self.record(AttemptKind::Call, outcome, progress);
        result
    }
}

impl<P: AttemptProgress> Attempts<P> {
    /// Resume from a validated checkpoint and keep its cumulative counts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resume<'capability>(
        &mut self,
        scope: SolveScope<'_, 'capability>,
        identity: SolveIdentity,
        token: u128,
        checkpoint: scioxide_core::SolveCheckpointMetadata,
        mut budgets: SolveBudgets,
        execute: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<P, SciOxideError>,
    ) -> Result<P, SciOxideError> {
        let mut started = false;
        let result = scope.resume(
            &mut self.state,
            identity,
            token,
            checkpoint,
            &mut budgets,
            |cx| {
                started = true;
                execute(cx)
            },
        );
        let (outcome, progress) = match (&result, started) {
            (_, false) => (AttemptOutcome::PreflightRejected, None),
            (Ok(value), true) => (value.outcome(), Some(value)),
            (Err(_), true) => (AttemptOutcome::Failed, None),
        };
        self.record(AttemptKind::Resume, outcome, progress);
        result
    }
}

/// Kernel outcome before the shell attaches counts.
pub(crate) enum Outcome<S: RealScalar> {
    /// The output holds an estimate over the whole domain.
    Published {
        termination: IntegrationTermination,
        subdivisions: usize,
        errors: Option<IntegrationErrorComponents<S>>,
    },
    /// The method stopped before any complete-domain estimate existed.
    Unpublished(IntegrationTermination),
}

/// Identity of a crate-internal epoch; callers never build one.
pub(crate) const fn identity(algorithm: AlgorithmId, shape: usize) -> SolveIdentity {
    SolveIdentity::new(ProblemToken::new(0), algorithm, shape as u128, 0)
}

impl<S: RealScalar> Attempts<IntegrationReport<S>> {
    /// Run one all-or-nothing integration in its own epoch.
    ///
    /// `Ok` always carries a written estimate; a stop before any estimate becomes
    /// the matching structured error, with the epoch's counts attached.
    #[inline(always)]
    pub(crate) fn integrate<'capability>(
        &mut self,
        scope: SolveScope<'_, 'capability>,
        algorithm: AlgorithmId,
        shape: usize,
        budgets: SolveBudgets,
        kernel: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<Outcome<S>, SciOxideError>,
    ) -> Result<IntegrationReport<S>, SciOxideError> {
        let Self { state, attempt } = self;
        let mut started = false;
        // The scoped operation returns only completion. Store its report once
        // before scope finalization, clearing it if accounting rejects the result.
        let result = scope.run(state, identity(algorithm, shape), budgets, {
            #[inline(always)]
            |context| {
                started = true;
                match kernel(context)? {
                    Outcome::Published {
                        termination,
                        subdivisions,
                        errors,
                    } => {
                        let report = IntegrationReport::new(
                            termination,
                            context.provider_count(EvaluationRole::Function)?,
                            subdivisions,
                            errors,
                        );
                        Self::record_into(
                            attempt,
                            AttemptKind::Call,
                            report.outcome(),
                            Some(&report),
                        );
                        Ok(())
                    }
                    Outcome::Unpublished(termination) => {
                        Err(unpublished(termination, algorithm, context.counts()))
                    }
                }
            }
        });
        if let Err(error) = result {
            let outcome = if started {
                AttemptOutcome::Failed
            } else {
                AttemptOutcome::PreflightRejected
            };
            Self::record_into(attempt, AttemptKind::Call, outcome, None);
            return Err(error);
        }
        attempt
            .as_ref()
            .and_then(|attempt| attempt.progress)
            .ok_or_else(|| {
                crate::kernel::invalid(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "completed integration has no stored report",
                )
            })
    }
}

/// Structured error for a stop that left no estimate over the whole domain.
pub(crate) fn unpublished(
    termination: IntegrationTermination,
    algorithm: AlgorithmId,
    counts: SolveCounts,
) -> SciOxideError {
    let detail = if termination == IntegrationTermination::GeometryFailure {
        "stable geometry produced no representable callback point"
    } else {
        "integration stopped before a complete-domain estimate"
    };
    termination_error(termination, algorithm, counts, detail)
}

/// Requires a successful termination before returning a one-shot answer.
pub(crate) fn require_success(
    termination: IntegrationTermination,
    algorithm: AlgorithmId,
    counts: SolveCounts,
) -> Result<(), SciOxideError> {
    if termination.is_success() {
        Ok(())
    } else {
        Err(termination_error(
            termination,
            algorithm,
            counts,
            "integration ended without meeting its tolerance",
        ))
    }
}
fn termination_error(
    termination: IntegrationTermination,
    algorithm: AlgorithmId,
    counts: SolveCounts,
    detail: &'static str,
) -> SciOxideError {
    let diagnostics = ConvergenceDiagnostics::new()
        .with_algorithm(algorithm)
        .with_counts(counts);
    let reason = match termination {
        IntegrationTermination::ExecutionStopped(reason) => {
            return SciOxideError::execution_stopped(reason);
        }
        IntegrationTermination::GeometryFailure => {
            return SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                detail,
            );
        }
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation) => {
            ConvergenceReason::EvaluationLimit
        }
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Iteration) => {
            ConvergenceReason::IterationLimit
        }
        IntegrationTermination::BudgetLimit(kind) => ConvergenceReason::ResourceLimit(kind),
        IntegrationTermination::RefinementLimit => ConvergenceReason::TruncationLimit,
        IntegrationTermination::Stagnation => ConvergenceReason::Stagnation,
        IntegrationTermination::ProbablyDivergent => ConvergenceReason::Divergence,
        IntegrationTermination::FixedRuleCompleted
        | IntegrationTermination::ToleranceReached
        | IntegrationTermination::HeuristicToleranceReached
        | IntegrationTermination::SampleBudgetCompleted
        | IntegrationTermination::DegenerateDomain => {
            return SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "successful integration withheld its estimate",
            );
        }
    };
    SciOxideError::convergence_failure(reason, diagnostics, detail)
}

/// Apply an optional configured limit to a local allowance.
pub(crate) const fn limit(
    budgets: SolveBudgets,
    kind: scioxide_core::BudgetKind,
    limit: Option<scioxide_core::Budget>,
) -> SolveBudgets {
    match limit {
        Some(limit) => budgets.with_limit(kind, limit),
        None => budgets,
    }
}

/// Check a component tolerance against the value shape before numerical work.
pub(crate) fn check_tolerance<S: RealScalar>(
    tolerance: &scioxide_core::ErrorTolerance<S>,
    components: usize,
) -> Result<(), SciOxideError> {
    match tolerance.absolute().as_vector() {
        Some(absolute) => crate::kernel::shape(components, absolute.len()),
        None => Ok(()),
    }
}

/// Check one complete work unit: stop check, then every touched budget, then charges.
/// Evaluations are charged later, per invocation, by the sampling shell.
#[inline(always)]
pub(crate) fn admit(
    context: &mut SolveContext<'_, '_>,
    evaluations: usize,
    subdivision: bool,
) -> Result<Option<IntegrationTermination>, SciOxideError> {
    if let Some(stop) = context.stop_reason()? {
        return Ok(Some(IntegrationTermination::ExecutionStopped(stop)));
    }
    match context.record_work_unit(evaluations, usize::from(subdivision))? {
        scioxide_core::Admission::Granted(()) => Ok(None),
        scioxide_core::Admission::Denied(report) => {
            Ok(Some(IntegrationTermination::BudgetLimit(report.kind())))
        }
    }
}

#[cfg(test)]
mod one_shot_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use scioxide_core::{BudgetKind, ExecutionStopReason, NumericalIssue, SciOxideErrorView};

    #[test]
    fn every_termination_uses_the_shared_error_classification() {
        let algorithm = scioxide_core::algorithm_id!("integrate", "one-shot-test", 1);
        for termination in [
            IntegrationTermination::FixedRuleCompleted,
            IntegrationTermination::ToleranceReached,
            IntegrationTermination::HeuristicToleranceReached,
            IntegrationTermination::SampleBudgetCompleted,
            IntegrationTermination::DegenerateDomain,
        ] {
            assert!(require_success(termination, algorithm, SolveCounts::new()).is_ok());
        }
        for (termination, reason) in [
            (
                IntegrationTermination::BudgetLimit(BudgetKind::Evaluation),
                ConvergenceReason::EvaluationLimit,
            ),
            (
                IntegrationTermination::BudgetLimit(BudgetKind::Iteration),
                ConvergenceReason::IterationLimit,
            ),
            (
                IntegrationTermination::BudgetLimit(BudgetKind::Subdivision),
                ConvergenceReason::ResourceLimit(BudgetKind::Subdivision),
            ),
            (
                IntegrationTermination::RefinementLimit,
                ConvergenceReason::TruncationLimit,
            ),
            (
                IntegrationTermination::Stagnation,
                ConvergenceReason::Stagnation,
            ),
            (
                IntegrationTermination::ProbablyDivergent,
                ConvergenceReason::Divergence,
            ),
        ] {
            let error = require_success(termination, algorithm, SolveCounts::new()).unwrap_err();
            assert!(
                matches!(error.view(),SciOxideErrorView::ConvergenceFailure {reason:actual,..} if actual==reason)
            );
            assert!(
                matches!(unpublished(termination,algorithm,SolveCounts::new()).view(),SciOxideErrorView::ConvergenceFailure {reason:actual,..} if actual==reason)
            );
        }
        let stop = ExecutionStopReason::Cancelled;
        assert!(
            matches!(require_success(IntegrationTermination::ExecutionStopped(stop),algorithm,SolveCounts::new()).unwrap_err().view(),SciOxideErrorView::ExecutionStopped {reason} if reason==stop)
        );
        assert!(matches!(
            require_success(
                IntegrationTermination::GeometryFailure,
                algorithm,
                SolveCounts::new()
            )
            .unwrap_err()
            .view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Breakdown,
                ..
            }
        ));
    }
}
