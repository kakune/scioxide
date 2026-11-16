//! PDE solve diagnostics and crate-private elliptic options.

use alloc::vec::Vec;
use core::fmt;

use scioxide_core::{
    Budget, BudgetKind, EvaluationCounts, EvaluationRole, ExecutionStopReason, NumericScalar,
    PartialResultTrust, RealScalar, SciOxideError, Tolerance,
};
use scioxide_linalg::LinearSolveTermination;
use scioxide_ode::OdeTermination;

pub(crate) fn default_compatibility_tolerance<S: RealScalar>() -> Tolerance<S> {
    Tolerance::from_sqrt_epsilon()
}

/// Source of the declared discrete nullspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdeNullspaceSource {
    /// Constant fields, detected from the operator: no reaction or advection term, and
    /// only derivative (natural) boundaries or periodic identification.
    StructuralConstant,
    /// Orthonormal reduced-system bases supplied by the problem.
    Supplied,
}

/// Reason a PDE solve stopped.
///
/// Linear convergence maps to `Completed`; iteration exhaustion, stagnation,
/// condition rejection and breakdown map to their corresponding `Linear*` variants.
/// Breakdown retains its recurrence cause. Evaluation and execution stops retain
/// their shared variants. Least-squares convergence remains
/// `LinearLeastSquaresConverged` unless a declared/detected nullspace and independent
/// physical-residual and gauge checks establish a completed elliptic solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PdeTermination {
    /// The steady-state or time-dependent solve completed as requested.
    Completed,
    /// An inner iterative linear solve reached its iteration limit.
    LinearIterationLimit,
    /// The inner solver reached a representational or progress floor.
    LinearStagnation,
    /// The inner solver crossed its configured condition limit.
    LinearConditionLimit,
    /// The inner recurrence failed, retaining its typed cause.
    LinearBreakdown(scioxide_linalg::LinearSolveBreakdown),
    /// Only the normal equations converged; physical equations were not certified.
    LinearLeastSquaresConverged,
    /// Time integration reached its step limit before the requested end time.
    TimeStepLimit,
    /// No smaller admissible representable time step can advance the ODE solve.
    TimeStepSizeLimit,
    /// A callback or provider evaluation budget was exhausted.
    EvaluationLimit,
    /// Explicitly injected cancellation or a deadline stopped the solve.
    ExecutionStopped(ExecutionStopReason),
}

impl PdeTermination {
    /// Returns whether the PDE solve completed successfully.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Completed)
    }

    pub(crate) const fn from_linear(termination: LinearSolveTermination) -> Self {
        match termination {
            LinearSolveTermination::Converged => Self::Completed,
            LinearSolveTermination::LeastSquaresConverged => Self::LinearLeastSquaresConverged,
            LinearSolveTermination::ConditionLimit => Self::LinearConditionLimit,
            LinearSolveTermination::Stagnation => Self::LinearStagnation,
            LinearSolveTermination::Breakdown(cause) => Self::LinearBreakdown(cause),
            LinearSolveTermination::IterationLimit => Self::LinearIterationLimit,
            LinearSolveTermination::EvaluationLimit => Self::EvaluationLimit,
            LinearSolveTermination::ExecutionStopped(reason) => Self::ExecutionStopped(reason),
        }
    }

    /// Classify an unsuccessful or successful ODE stop. Attempted-step budgets
    /// (`Proposal`, `Iteration`) are time-step limits; shared work budgets are
    /// evaluation limits. ODE failures are errors, not terminations.
    pub(crate) fn from_ode(termination: OdeTermination) -> Result<Self, SciOxideError> {
        Ok(match termination {
            OdeTermination::ReachedEnd | OdeTermination::TerminalEvent => Self::Completed,
            OdeTermination::BudgetLimit(BudgetKind::Proposal | BudgetKind::Iteration) => {
                Self::TimeStepLimit
            }
            OdeTermination::BudgetLimit(_) => Self::EvaluationLimit,
            OdeTermination::StepSizeLimit => Self::TimeStepSizeLimit,
            OdeTermination::ExecutionStopped(reason) => Self::ExecutionStopped(reason),
            _ => {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("report"),
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "unrecognized ODE termination",
                ));
            }
        })
    }
}

impl fmt::Display for PdeTermination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => write!(formatter, "Completed"),
            Self::LinearStagnation => write!(formatter, "LinearStagnation"),
            Self::LinearConditionLimit => write!(formatter, "LinearConditionLimit"),
            Self::LinearBreakdown(cause) => write!(formatter, "LinearBreakdown({cause:?})"),
            Self::LinearLeastSquaresConverged => write!(formatter, "LinearLeastSquaresConverged"),
            Self::LinearIterationLimit => write!(formatter, "LinearIterationLimit"),
            Self::TimeStepLimit => write!(formatter, "TimeStepLimit"),
            Self::TimeStepSizeLimit => write!(formatter, "TimeStepSizeLimit"),
            Self::EvaluationLimit => write!(formatter, "EvaluationLimit"),
            Self::ExecutionStopped(reason) => write!(formatter, "ExecutionStopped({reason})"),
        }
    }
}

/// Validated diagnostics and work counters from a PDE solve.
///
/// This metadata is embedded in [`PdeSolveReport`]. A time step is one
/// attempted integrator step, so its count is the sum of accepted, rejected,
/// and callback/numerical-failed attempts.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PdeDiagnostics<S: RealScalar> {
    termination: PdeTermination,
    final_time: S,
    method: Option<&'static str>,
    num_linear_iterations: usize,
    residual_norm: Option<S>,
    linear_residual_report: Option<scioxide_core::ResidualReport<S>>,
    nullspace: Option<scioxide_linalg::BorderedSystemDiagnostics<S>>,
    nullspace_gauge_defect: Option<S>,
    nullspace_source: Option<PdeNullspaceSource>,
    residual_history: Option<Vec<S>>,
    last_step_error_estimate: Option<S>,
    num_time_steps: usize,
    num_accepted_steps: usize,
    num_rejected_steps: usize,
    num_failed_steps: usize,
    num_assemblies: usize,
    num_factorizations: usize,
    num_linear_solves: usize,
    num_rhs_evaluations: usize,
    num_jacobian_evaluations: usize,
    num_nonlinear_iterations: usize,
}

impl<S: RealScalar> PdeDiagnostics<S> {
    /// Independently checked physical linear residual and its explicit norm/scaling.
    /// Direct solves use infinity-norm backward-error scaling.
    pub const fn linear_residual_report(&self) -> Option<scioxide_core::ResidualReport<S>> {
        self.linear_residual_report
    }
    pub(crate) fn with_linear_residual(mut self, report: scioxide_core::ResidualReport<S>) -> Self {
        self.residual_norm = Some(report.raw_norm());
        self.linear_residual_report = Some(report);
        self
    }
    pub(crate) fn with_direct_factorization(mut self) -> Self {
        self.num_factorizations = 1;
        self
    }
    pub(crate) fn with_completed(mut self) -> Self {
        self.termination = PdeTermination::Completed;
        self
    }
    /// Diagnostics of the bordered system's original nullspace constraints.
    /// Use [`Self::nullspace_gauge_defect`] for the requested PDE gauge.
    pub const fn nullspace(&self) -> Option<&scioxide_linalg::BorderedSystemDiagnostics<S>> {
        self.nullspace.as_ref()
    }

    /// Defect of the requested gauge, absent when no gauge was used.
    /// For a weighted-mean gauge this is the absolute mean error.
    pub const fn nullspace_gauge_defect(&self) -> Option<S> {
        self.nullspace_gauge_defect
    }

    pub(crate) fn with_nullspace(
        mut self,
        diagnostics: scioxide_linalg::BorderedSystemDiagnostics<S>,
        source: PdeNullspaceSource,
        gauge_defect: S,
    ) -> Self {
        self.nullspace = Some(diagnostics);
        self.nullspace_gauge_defect = Some(gauge_defect);
        self.nullspace_source = Some(source);
        self
    }

    /// Structural or caller-supplied nullspace provenance.
    pub const fn nullspace_source(&self) -> Option<PdeNullspaceSource> {
        self.nullspace_source
    }

    /// Diagnostics of one elliptic linear solve.
    ///
    /// Inconsistent solver diagnostics reject the caller's `source` argument.
    pub(crate) fn from_linear_solve(
        method: &'static str,
        termination: LinearSolveTermination,
        iterations: usize,
        residual_norm: S,
        residual_history: Option<Vec<S>>,
        source: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        Self::new(
            Self {
                termination: PdeTermination::from_linear(termination),
                final_time: S::ZERO,
                method: Some(method),
                num_linear_iterations: iterations,
                residual_norm: Some(residual_norm),
                linear_residual_report: None,
                nullspace: None,
                nullspace_gauge_defect: None,
                nullspace_source: None,
                residual_history,
                last_step_error_estimate: None,
                num_time_steps: 0,
                num_accepted_steps: 0,
                num_rejected_steps: 0,
                num_failed_steps: 0,
                num_assemblies: 1,
                num_factorizations: 0,
                num_linear_solves: 1,
                num_rhs_evaluations: 0,
                num_jacobian_evaluations: 0,
                num_nonlinear_iterations: 0,
            },
            source,
        )
    }

    /// Diagnostics of a fully constrained elliptic system, validated as `from_linear_solve`.
    #[cfg(test)]
    pub(crate) fn elliptic_without_solve(
        source: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        Self::new(
            Self {
                termination: PdeTermination::Completed,
                final_time: S::ZERO,
                method: None,
                num_linear_iterations: 0,
                residual_norm: None,
                linear_residual_report: None,
                nullspace: None,
                nullspace_gauge_defect: None,
                nullspace_source: None,
                residual_history: None,
                last_step_error_estimate: None,
                num_time_steps: 0,
                num_accepted_steps: 0,
                num_rejected_steps: 0,
                num_failed_steps: 0,
                num_assemblies: 1,
                num_factorizations: 0,
                num_linear_solves: 0,
                num_rhs_evaluations: 0,
                num_jacobian_evaluations: 0,
                num_nonlinear_iterations: 0,
            },
            source,
        )
    }

    /// Diagnostics of an ODE-owned time integration.
    ///
    /// Inconsistent ODE counts reject the caller's `source` argument.
    pub(crate) fn from_ode(
        method: &'static str,
        termination: PdeTermination,
        final_time: S,
        last_step_error_estimate: Option<S>,
        counts: EvaluationCounts,
        source: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        let classified_steps = counts
            .accepted()
            .checked_add(counts.rejected())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "pde_classified_ode_step_count"
                ))
            })?;
        let failed_steps = counts
            .iterations()
            .checked_sub(classified_steps)
            .ok_or_else(|| {
                SciOxideError::invalid_input(
                    source,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "accepted and rejected steps exceed attempted steps",
                )
            })?;
        Self::new(
            Self {
                termination,
                final_time,
                method: Some(method),
                num_linear_iterations: 0,
                residual_norm: None,
                linear_residual_report: None,
                nullspace: None,
                nullspace_gauge_defect: None,
                nullspace_source: None,
                residual_history: None,
                last_step_error_estimate,
                num_time_steps: counts.iterations(),
                num_accepted_steps: counts.accepted(),
                num_rejected_steps: counts.rejected(),
                num_failed_steps: failed_steps,
                num_assemblies: 0,
                num_factorizations: counts.provider(EvaluationRole::Factorization),
                num_linear_solves: counts.provider(EvaluationRole::LinearSolve),
                num_rhs_evaluations: counts.provider(EvaluationRole::RightHandSide),
                num_jacobian_evaluations: counts.provider(EvaluationRole::Jacobian),
                num_nonlinear_iterations: 0,
            },
            source,
        )
    }

    fn new(stats: Self, source: scioxide_core::Argument) -> Result<Self, SciOxideError> {
        stats.validate(source)?;
        Ok(stats)
    }

    fn validate(&self, source: scioxide_core::Argument) -> Result<(), SciOxideError> {
        if !self.final_time.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Output,
                "final time is non-finite",
            ));
        }
        if self.method.is_some_and(str::is_empty) {
            return Err(SciOxideError::invalid_input(
                source,
                scioxide_core::InvalidInputReason::Malformed,
                "method name must not be empty",
            ));
        }
        let classified_steps = self
            .num_accepted_steps
            .checked_add(self.num_rejected_steps)
            .and_then(|count| count.checked_add(self.num_failed_steps))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "pde_accepted_and_rejected_time_step_count"
                ))
            })?;
        if classified_steps != self.num_time_steps {
            return Err(SciOxideError::invalid_input(
                source,
                scioxide_core::InvalidInputReason::Inconsistent,
                alloc::format!(
                    "time-step attempts ({}) must equal accepted ({}), rejected ({}), plus failed ({}) steps",
                    self.num_time_steps,
                    self.num_accepted_steps,
                    self.num_rejected_steps,
                    self.num_failed_steps
                ),
            ));
        }
        if let Some(residual_norm) = self.residual_norm {
            validate_nonnegative_finite(residual_norm, source)?;
        } else if self.num_linear_iterations != 0 || self.residual_history.is_some() {
            return Err(inconsistent_stats(
                source,
                "linear iterations and residual history require a residual norm",
            ));
        }
        if let Some(history) = &self.residual_history {
            let expected_len = self.num_linear_iterations.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "pde_residual_history_length"
                ))
            })?;
            if history.len() != expected_len {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(expected_len),
                    scioxide_core::Dimensions::one(history.len()),
                ));
            }
            history
                .iter()
                .try_for_each(|&residual| validate_nonnegative_finite(residual, source))?;
            if history.last().copied() != self.residual_norm {
                return Err(inconsistent_stats(
                    source,
                    "final residual history sample must equal the reported residual norm",
                ));
            }
        }
        if let Some(estimate) = self.last_step_error_estimate {
            validate_nonnegative_finite(estimate, source)?;
            if self.num_time_steps == 0 {
                return Err(inconsistent_stats(
                    source,
                    "a time-step error estimate requires at least one attempted step",
                ));
            }
        }
        match self.termination {
            PdeTermination::Completed => {}
            PdeTermination::LinearIterationLimit
            | PdeTermination::LinearStagnation
            | PdeTermination::LinearConditionLimit
            | PdeTermination::LinearBreakdown(_)
            | PdeTermination::LinearLeastSquaresConverged => {
                if self.num_time_steps != 0
                    || self.num_linear_solves == 0
                    || self.residual_norm.is_none()
                {
                    return Err(inconsistent_stats(
                        source,
                        "a linear iteration limit requires an elliptic linear solve and residual",
                    ));
                }
            }
            PdeTermination::TimeStepLimit
            | PdeTermination::TimeStepSizeLimit
            | PdeTermination::EvaluationLimit
            | PdeTermination::ExecutionStopped(_) => {}
        }
        Ok(())
    }

    /// Canonical reason the PDE solve stopped.
    #[must_use]
    pub const fn termination(&self) -> PdeTermination {
        self.termination
    }

    /// Actual time reached by the returned values.
    #[must_use]
    pub const fn final_time(&self) -> S {
        self.final_time
    }

    /// Nested numerical method name, if a solve was required.
    #[must_use]
    pub const fn method(&self) -> Option<&'static str> {
        self.method
    }

    /// Number of completed inner linear iterations.
    #[must_use]
    pub const fn num_linear_iterations(&self) -> usize {
        self.num_linear_iterations
    }

    /// Final linear residual norm when applicable.
    #[must_use]
    pub const fn residual_norm(&self) -> Option<S> {
        self.residual_norm
    }

    /// Linear residual history when recording was enabled.
    #[must_use]
    pub fn residual_history(&self) -> Option<&[S]> {
        self.residual_history.as_deref()
    }

    /// Last normalized time-step error estimate when applicable.
    #[must_use]
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }

    /// Number of attempted time steps; zero for elliptic solves.
    #[must_use]
    pub const fn num_time_steps(&self) -> usize {
        self.num_time_steps
    }

    /// Number of accepted time steps.
    #[must_use]
    pub const fn num_accepted_steps(&self) -> usize {
        self.num_accepted_steps
    }

    /// Number of rejected time steps.
    #[must_use]
    pub const fn num_rejected_steps(&self) -> usize {
        self.num_rejected_steps
    }

    /// Number of attempted time steps stopped by a callback or numerical failure.
    #[must_use]
    pub const fn num_failed_steps(&self) -> usize {
        self.num_failed_steps
    }

    /// Number of completed global spatial-system assemblies.
    #[must_use]
    pub const fn num_assemblies(&self) -> usize {
        self.num_assemblies
    }

    /// Number of completed matrix factorizations.
    #[must_use]
    pub const fn num_factorizations(&self) -> usize {
        self.num_factorizations
    }

    /// Number of linear solver invocations.
    #[must_use]
    pub const fn num_linear_solves(&self) -> usize {
        self.num_linear_solves
    }

    /// Number of completed time-integrator right-hand-side evaluations.
    #[must_use]
    pub const fn num_rhs_evaluations(&self) -> usize {
        self.num_rhs_evaluations
    }

    /// Number of completed time-integrator Jacobian evaluations.
    #[must_use]
    pub const fn num_jacobian_evaluations(&self) -> usize {
        self.num_jacobian_evaluations
    }

    /// Number of nested nonlinear iterations.
    #[must_use]
    pub const fn num_nonlinear_iterations(&self) -> usize {
        self.num_nonlinear_iterations
    }
}

/// Termination, diagnostics and trust of one successful steady solve or transient advance.
///
/// Values are published separately: steady solves write caller storage, transient
/// advances update the caller's [`crate::PdeState`] and stream fields to the sink.
/// [`PdeTermination`] is the sole status authority; `Display` renders the method and
/// termination on demand.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PdeSolveReport<S: RealScalar> {
    diagnostics: PdeDiagnostics<S>,
    trust: PartialResultTrust,
}

impl<S: RealScalar> PdeSolveReport<S> {
    pub(crate) fn require_success(&self) -> Result<(), SciOxideError> {
        use scioxide_core::{ConvergenceDiagnostics, ConvergenceReason as Reason};
        let reason = match self.termination() {
            PdeTermination::Completed => return Ok(()),
            PdeTermination::ExecutionStopped(reason) => {
                return Err(SciOxideError::execution_stopped(reason));
            }
            PdeTermination::TimeStepLimit | PdeTermination::LinearIterationLimit => {
                Reason::IterationLimit
            }
            PdeTermination::TimeStepSizeLimit => Reason::StepSizeLimit,
            PdeTermination::EvaluationLimit => Reason::EvaluationLimit,
            PdeTermination::LinearStagnation => Reason::Stagnation,
            PdeTermination::LinearConditionLimit
            | PdeTermination::LinearBreakdown(_)
            | PdeTermination::LinearLeastSquaresConverged => Reason::NestedSolver,
        };
        let d = &self.diagnostics;
        let counts = if matches!(
            self.termination(),
            PdeTermination::TimeStepLimit | PdeTermination::TimeStepSizeLimit
        ) {
            EvaluationCounts::from_iterations(d.num_time_steps()).with_proposals(
                d.num_time_steps(),
                d.num_accepted_steps(),
                d.num_rejected_steps(),
            )
        } else {
            EvaluationCounts::from_iterations(d.num_linear_iterations())
        };
        Err(SciOxideError::convergence_failure(
            reason,
            ConvergenceDiagnostics::new().with_counts(counts.into()),
            "PDE solve did not complete",
        ))
    }

    pub(crate) const fn new(diagnostics: PdeDiagnostics<S>, trust: PartialResultTrust) -> Self {
        Self { diagnostics, trust }
    }

    /// Sole authoritative termination reason.
    pub const fn termination(&self) -> PdeTermination {
        self.diagnostics.termination()
    }

    /// Time of the final accepted state; zero for steady solves.
    pub const fn final_time(&self) -> S {
        self.diagnostics.final_time()
    }

    /// Validated method-specific diagnostics.
    pub const fn diagnostics(&self) -> &PdeDiagnostics<S> {
        &self.diagnostics
    }

    /// Trust established for the published state.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }

    /// Whether the requested solve completed.
    pub const fn is_success(&self) -> bool {
        self.termination().is_success()
    }
}

/// Classify the report of an ODE integration of a [`crate::MethodOfLinesSystem`];
/// step counts include the adapter's provider work. Reconstruct the physical field
/// with [`crate::MethodOfLinesSystem::reconstruct_into`].
impl<S: RealScalar + NumericScalar<Real = S>> TryFrom<&scioxide_ode::OdeSolveReport<S>>
    for PdeSolveReport<S>
{
    type Error = SciOxideError;

    fn try_from(report: &scioxide_ode::OdeSolveReport<S>) -> Result<Self, SciOxideError> {
        let mut counts = scioxide_core::EvaluationCounts::from_iterations(report.attempted_steps())
            .with_proposals(
                report.attempted_steps(),
                report.accepted_steps(),
                report.rejected_steps(),
            );
        let work = report.accounting().total()?;
        for role in [
            EvaluationRole::RightHandSide,
            EvaluationRole::Jacobian,
            EvaluationRole::Factorization,
            EvaluationRole::LinearSolve,
        ] {
            counts = counts.with_provider_count(role, work.provider(role));
        }
        Ok(Self::new(
            PdeDiagnostics::from_ode(
                "MethodOfLines",
                PdeTermination::from_ode(report.termination().ok_or_else(|| {
                    SciOxideError::invalid_input(
                        scioxide_core::argument!("report"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "ODE report has no termination",
                    )
                })?)?,
                report.final_time(),
                report.last_step_error_estimate(),
                counts,
                scioxide_core::argument!("report"),
            )?,
            report.trust(),
        ))
    }
}

impl<S: RealScalar> fmt::Display for PdeSolveReport<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.diagnostics.method() {
            Some(method) => write!(formatter, "{method}: {}", self.termination()),
            None => write!(formatter, "{}", self.termination()),
        }
    }
}

fn validate_nonnegative_finite<S: RealScalar>(
    value: S,
    source: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Output,
            "diagnostic value is non-finite",
        ));
    }
    if value < S::ZERO {
        return Err(SciOxideError::invalid_input(
            source,
            scioxide_core::InvalidInputReason::OutOfRange,
            "diagnostic value must be nonnegative",
        ));
    }
    Ok(())
}

fn inconsistent_stats(source: scioxide_core::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        source,
        scioxide_core::InvalidInputReason::Inconsistent,
        detail,
    )
}

/// Immutable configuration for elliptic PDE kernels.
///
/// Transient plans use [`crate::ParabolicOptions`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub(crate) struct PdeKernelOptions<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) gauge: crate::EllipticGauge<S>,
    pub(crate) compatibility_policy: scioxide_linalg::CompatibilityPolicy,
    pub(crate) compatibility_tolerance: Tolerance<S>,
    tolerance: Tolerance<S>,
    max_linear_iterations: Budget,
}

impl<S: RealScalar + NumericScalar<Real = S>> PdeKernelOptions<S> {
    pub(crate) const fn with_nullspace(
        mut self,
        gauge: crate::EllipticGauge<S>,
        policy: scioxide_linalg::CompatibilityPolicy,
        tolerance: Tolerance<S>,
    ) -> Self {
        self.gauge = gauge;
        self.compatibility_policy = policy;
        self.compatibility_tolerance = tolerance;
        self
    }

    /// Return a copy using `tolerance`.
    pub(crate) const fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Return a copy using `max_linear_iterations`.
    pub(crate) const fn with_max_linear_iterations(
        mut self,
        max_linear_iterations: Budget,
    ) -> Self {
        self.max_linear_iterations = max_linear_iterations;
        self
    }

    /// Configured numerical tolerance.
    pub(crate) const fn tolerance(self) -> Tolerance<S> {
        self.tolerance
    }

    /// Maximum iterations for an elliptic iterative linear solve.
    pub(crate) const fn max_linear_iterations(self) -> Budget {
        self.max_linear_iterations
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for PdeKernelOptions<S> {
    fn default() -> Self {
        Self {
            gauge: crate::EllipticGauge::Unspecified,
            compatibility_policy: scioxide_linalg::CompatibilityPolicy::RequireCompatible,
            compatibility_tolerance: default_compatibility_tolerance(),
            tolerance: Tolerance::from_sqrt_epsilon(),
            max_linear_iterations: Budget::ONE.saturating_add(999),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use scioxide_core::{ProblemToken, SolveBudgets};

    use super::*;

    #[test]
    fn unsuccessful_reports_preserve_reasons_and_counts() {
        use scioxide_core::{ConvergenceReason as Reason, SciOxideErrorView};
        let base = PdeDiagnostics::from_linear_solve(
            "test",
            LinearSolveTermination::Converged,
            7,
            0.0_f64,
            None,
            SOURCE,
        )
        .unwrap();
        let mut report = PdeSolveReport::new(base, PartialResultTrust::DiagnosticOnly);
        assert!(report.require_success().is_ok());
        report.diagnostics.num_time_steps = 3;
        report.diagnostics.num_accepted_steps = 2;
        report.diagnostics.num_rejected_steps = 1;
        for (termination, expected, iterations) in [
            (
                PdeTermination::LinearIterationLimit,
                Reason::IterationLimit,
                7,
            ),
            (PdeTermination::LinearStagnation, Reason::Stagnation, 7),
            (
                PdeTermination::LinearConditionLimit,
                Reason::NestedSolver,
                7,
            ),
            (
                PdeTermination::LinearBreakdown(
                    scioxide_linalg::LinearSolveBreakdown::RecurrenceDenominator,
                ),
                Reason::NestedSolver,
                7,
            ),
            (
                PdeTermination::LinearLeastSquaresConverged,
                Reason::NestedSolver,
                7,
            ),
            (PdeTermination::TimeStepLimit, Reason::IterationLimit, 3),
            (PdeTermination::TimeStepSizeLimit, Reason::StepSizeLimit, 3),
            (PdeTermination::EvaluationLimit, Reason::EvaluationLimit, 7),
        ] {
            report.diagnostics.termination = termination;
            let error = report.require_success().unwrap_err();
            let SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                ..
            } = error.view()
            else {
                unreachable!()
            };
            assert_eq!(reason, expected);
            assert_eq!(diagnostics.counts().own().iterations(), iterations);
            if iterations == 3 {
                assert_eq!(diagnostics.counts().own().accepted(), 2);
                assert_eq!(diagnostics.counts().own().rejected(), 1);
            }
        }
        for stop in [
            ExecutionStopReason::Cancelled,
            ExecutionStopReason::DeadlineReached,
        ] {
            report.diagnostics.termination = PdeTermination::ExecutionStopped(stop);
            assert_eq!(
                report.require_success().unwrap_err(),
                SciOxideError::execution_stopped(stop)
            );
        }
    }

    const SOURCE: scioxide_core::Argument = scioxide_core::argument!("stats");

    #[test]
    fn ode_terminations_map_budget_kinds_to_pde_limits() {
        use scioxide_core::{BudgetKind, ExecutionStopReason};
        for (ode, pde) in [
            (OdeTermination::ReachedEnd, PdeTermination::Completed),
            (OdeTermination::TerminalEvent, PdeTermination::Completed),
            (
                OdeTermination::BudgetLimit(BudgetKind::Proposal),
                PdeTermination::TimeStepLimit,
            ),
            (
                OdeTermination::BudgetLimit(BudgetKind::Iteration),
                PdeTermination::TimeStepLimit,
            ),
            (
                OdeTermination::BudgetLimit(BudgetKind::Evaluation),
                PdeTermination::EvaluationLimit,
            ),
            (
                OdeTermination::BudgetLimit(BudgetKind::Subdivision),
                PdeTermination::EvaluationLimit,
            ),
            (
                OdeTermination::StepSizeLimit,
                PdeTermination::TimeStepSizeLimit,
            ),
            (
                OdeTermination::ExecutionStopped(ExecutionStopReason::Cancelled),
                PdeTermination::ExecutionStopped(ExecutionStopReason::Cancelled),
            ),
        ] {
            assert_eq!(PdeTermination::from_ode(ode).unwrap(), pde, "{ode}");
            assert_eq!(pde.is_success(), ode.is_success());
        }
    }

    #[test]
    fn ode_step_size_exhaustion_preserves_partial_diagnostics() {
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "ode-steps", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                identity,
                &mut budgets,
            )
            .unwrap();
        for rejected in [false, true] {
            if rejected {
                context.record_iteration().unwrap().require().unwrap();
                context.record_rejected().unwrap();
            }
            let diagnostics = PdeDiagnostics::from_ode(
                "DormandPrince5",
                PdeTermination::from_ode(OdeTermination::StepSizeLimit).unwrap(),
                0.0_f64,
                rejected.then_some(2.0),
                context.counts().own(),
                SOURCE,
            )
            .unwrap();
            assert_eq!(diagnostics.termination(), PdeTermination::TimeStepSizeLimit);
            assert!(!diagnostics.termination().is_success());
            assert_eq!(diagnostics.final_time(), 0.0);
            assert_eq!(diagnostics.num_time_steps, usize::from(rejected));
            assert_eq!(diagnostics.num_rejected_steps, usize::from(rejected));
            assert_eq!(diagnostics.num_failed_steps, 0);
        }
    }

    fn completed_stats() -> PdeDiagnostics<f64> {
        PdeDiagnostics::elliptic_without_solve(SOURCE).unwrap()
    }

    #[test]
    fn statistics_require_every_time_step_to_be_classified() {
        let stats = PdeDiagnostics {
            num_time_steps: 3,
            num_accepted_steps: 1,
            num_rejected_steps: 1,
            ..completed_stats()
        };

        assert!(
            matches ! (& (PdeDiagnostics :: new (stats ,  SOURCE)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
    }

    #[test]
    fn linear_statistics_validate_residual_history() {
        let result = PdeDiagnostics::from_linear_solve(
            "test solver",
            LinearSolveTermination::Converged,
            2,
            0.25,
            Some(vec![1.0, 0.25]),
            SOURCE,
        );

        assert!(
            matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: DimensionMismatch { .. }))
        );
    }
}
