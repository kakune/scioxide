//! Results, termination statuses and optimality diagnostics.

use crate::LineSearchFailure;
#[cfg(test)]
use alloc::borrow::ToOwned;
use alloc::string::String;
#[cfg(test)]
use alloc::string::ToString;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use core::fmt;

use scioxide_core::backend::try_reserve;
use scioxide_core::{RealScalar, SciOxideError};

// ---------------------------------------------------------------------------
// Termination reason
// ---------------------------------------------------------------------------

/// Holds the optimality (KKT) residuals of a returned point.
///
/// Each residual is nonnegative and measured in the units of the original,
/// unscaled problem. `None` means that the solver did not compute that
/// residual, not that it is zero. The optimality gap is the absolute
/// difference between the objective and a valid dual or relaxation bound.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OptimalityDiagnostics<S: RealScalar> {
    pub(crate) primal_feasibility: Option<S>,
    pub(crate) stationarity: Option<S>,
    pub(crate) stationarity_norm: scioxide_core::NormSelection,
    pub(crate) dual_feasibility: Option<S>,
    pub(crate) complementarity: Option<S>,
    pub(crate) integrality_violation: Option<S>,
    pub(crate) bound_violation: Option<S>,
    pub(crate) certified_optimality_gap: Option<S>,
}

impl<S: RealScalar> OptimalityDiagnostics<S> {
    /// Returns the primal feasibility residual as an unscaled infinity-norm
    /// [`ResidualReport`](scioxide_core::ResidualReport), or `None` if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value is negative or not finite.
    pub fn primal_feasibility_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.primal_feasibility
            .map(|value| {
                scioxide_core::ResidualReport::unscaled(
                    value,
                    scioxide_core::NormSelection::Infinity,
                )
            })
            .transpose()
    }

    /// Returns the stationarity residual as an unscaled
    /// [`ResidualReport`](scioxide_core::ResidualReport) in the norm of
    /// [`Self::stationarity`], or `None` if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value is negative or not finite.
    pub fn stationarity_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.stationarity
            .map(|value| scioxide_core::ResidualReport::unscaled(value, self.stationarity_norm))
            .transpose()
    }

    /// Returns the dual feasibility residual as an unscaled infinity-norm
    /// [`ResidualReport`](scioxide_core::ResidualReport), or `None` if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value is negative or not finite.
    pub fn dual_feasibility_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.dual_feasibility
            .map(|value| {
                scioxide_core::ResidualReport::unscaled(
                    value,
                    scioxide_core::NormSelection::Infinity,
                )
            })
            .transpose()
    }

    /// Returns the complementarity residual as an unscaled infinity-norm
    /// [`ResidualReport`](scioxide_core::ResidualReport), or `None` if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value is negative or not finite.
    pub fn complementarity_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.complementarity
            .map(|value| {
                scioxide_core::ResidualReport::unscaled(
                    value,
                    scioxide_core::NormSelection::Infinity,
                )
            })
            .transpose()
    }

    /// Returns the integrality violation residual as an unscaled infinity-norm
    /// [`ResidualReport`](scioxide_core::ResidualReport), or `None` if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value is negative or not finite.
    pub fn integrality_violation_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.integrality_violation
            .map(|value| {
                scioxide_core::ResidualReport::unscaled(
                    value,
                    scioxide_core::NormSelection::Infinity,
                )
            })
            .transpose()
    }

    /// Returns the bound violation residual as an unscaled infinity-norm
    /// [`ResidualReport`](scioxide_core::ResidualReport), or `None` if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored value is negative or not finite.
    pub fn bound_violation_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.bound_violation
            .map(|value| {
                scioxide_core::ResidualReport::unscaled(
                    value,
                    scioxide_core::NormSelection::Infinity,
                )
            })
            .transpose()
    }

    /// Returns diagnostics with every residual unavailable.
    pub const fn unavailable() -> Self {
        Self {
            primal_feasibility: None,
            stationarity: None,
            stationarity_norm: scioxide_core::NormSelection::L2,
            dual_feasibility: None,
            complementarity: None,
            integrality_violation: None,
            bound_violation: None,
            certified_optimality_gap: None,
        }
    }

    /// Returns the infinity norm of the constraint violation, if computed.
    pub const fn primal_feasibility(self) -> Option<S> {
        self.primal_feasibility
    }

    /// Returns the norm of the gradient of the objective or Lagrangian, if
    /// computed. LP and QP verification use the infinity norm.
    pub const fn stationarity(self) -> Option<S> {
        self.stationarity
    }

    /// Returns the infinity norm of the multiplier sign violation, if computed.
    pub const fn dual_feasibility(self) -> Option<S> {
        self.dual_feasibility
    }

    /// Returns the infinity norm of the multiplier–slack products, in objective
    /// units, if computed.
    pub const fn complementarity(self) -> Option<S> {
        self.complementarity
    }

    /// Returns the largest distance of an integer variable from an integer, if
    /// computed.
    pub const fn integrality_violation(self) -> Option<S> {
        self.integrality_violation
    }

    /// Returns the largest bound violation, in parameter units, if computed.
    pub const fn bound_violation(self) -> Option<S> {
        self.bound_violation
    }

    /// Returns the absolute gap between the objective and a valid lower bound,
    /// if one is known.
    pub const fn certified_optimality_gap(self) -> Option<S> {
        self.certified_optimality_gap
    }
}

impl<S: RealScalar> Default for OptimalityDiagnostics<S> {
    fn default() -> Self {
        Self::unavailable()
    }
}

/// States why a solver stopped.
///
/// Only [`Converged`](Self::Converged) and [`GradientSmall`](Self::GradientSmall)
/// are successful, as reported by [`Self::is_success`]. Both mean that the
/// method's local stopping tests passed; neither certifies a global minimum.
/// Every other status is unsuccessful, and the result then holds the best
/// accepted point. See [Termination](crate#termination).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptimizeTermination {
    /// A line search found no acceptable step. Running out of a solve budget
    /// during the search is reported by the budget's own status instead.
    LineSearchFailed(LineSearchFailure),
    /// No solve has written to this result yet.
    NotStarted,
    /// The method's stopping tests passed. Successful.
    Converged,
    /// The iteration limit was reached.
    IterationLimit,
    /// The time limit or the deadline of the execution context was reached.
    TimeLimit,
    /// The execution context requested cancellation.
    Cancelled,
    /// The branch-and-bound node limit was reached.
    NodeLimit,
    /// A configured storage limit of a search or relaxation was reached.
    MemoryLimit,
    /// The search depth limit was reached.
    DepthLimit,
    /// The evaluation limit was reached.
    EvaluationLimit,
    /// The limit on proposed trial points was reached.
    ProposalLimit,
    /// The limit on random samples was reached.
    SampleLimit,
    /// A global or heuristic search used its whole configured budget. This is
    /// the normal end of such a search, but it is not success because no
    /// global optimality is claimed.
    SearchBudgetCompleted,
    /// The population of an evolutionary method stopped improving.
    PopulationStagnation,
    /// A sampled gradient and its estimated sampling error are both below the
    /// tolerance. This is a statistical estimate, not a guarantee.
    EstimatedStationarity,
    /// The allowed number of replicated samples could not reduce the noise
    /// enough to decide whether to accept a step.
    NoiseLimit,
    /// The gradient norm fell below its tolerance. Successful.
    GradientSmall,
    /// The step fell below its tolerance before the gradient test passed.
    StepSmall,
    /// The method stopped making progress before a stopping test passed.
    Stagnation,
    /// The change in the objective fell below its tolerance before the
    /// gradient test passed.
    FunctionChangeSmall,
    /// The problem appears infeasible, but no certificate was verified.
    SuspectedInfeasible,
    /// The problem appears unbounded, but no certificate was verified.
    SuspectedUnbounded,
    /// The problem is infeasible, with a verified certificate.
    Infeasible,
    /// The problem is unbounded below, with a verified certificate.
    Unbounded,
}

impl fmt::Display for OptimizeTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotStarted => write!(f, "NotStarted"),
            Self::LineSearchFailed(reason) => write!(f, "LineSearchFailed({reason})"),
            Self::Converged => write!(f, "Converged"),
            Self::IterationLimit => write!(f, "IterationLimit"),
            Self::TimeLimit => write!(f, "TimeLimit"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::NodeLimit => write!(f, "NodeLimit"),
            Self::MemoryLimit => write!(f, "MemoryLimit"),
            Self::DepthLimit => write!(f, "DepthLimit"),
            Self::EvaluationLimit => write!(f, "EvaluationLimit"),
            Self::ProposalLimit => write!(f, "ProposalLimit"),
            Self::SampleLimit => write!(f, "SampleLimit"),
            Self::SearchBudgetCompleted => write!(f, "SearchBudgetCompleted"),
            Self::PopulationStagnation => write!(f, "PopulationStagnation"),
            Self::EstimatedStationarity => write!(f, "EstimatedStationarity"),
            Self::NoiseLimit => write!(f, "NoiseLimit"),
            Self::GradientSmall => write!(f, "GradientSmall"),
            Self::StepSmall => write!(f, "StepSmall"),
            Self::Stagnation => write!(f, "Stagnation"),
            Self::FunctionChangeSmall => write!(f, "FunctionChangeSmall"),
            Self::SuspectedInfeasible => write!(f, "SuspectedInfeasible"),
            Self::SuspectedUnbounded => write!(f, "SuspectedUnbounded"),
            Self::Infeasible => write!(f, "Infeasible"),
            Self::Unbounded => write!(f, "Unbounded"),
        }
    }
}

impl OptimizeTermination {
    /// Returns `true` for [`Converged`](Self::Converged) and
    /// [`GradientSmall`](Self::GradientSmall).
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Converged | Self::GradientSmall)
    }
}

pub(crate) fn termination_error(
    argument: scioxide_core::Argument,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
    algorithm: Option<scioxide_core::AlgorithmId>,
) -> Option<SciOxideError> {
    use OptimizeTermination as T;
    use scioxide_core::{
        BudgetKind, ConvergenceDiagnostics, ConvergenceReason, DomainReason, ExecutionStopReason,
        InvalidInputReason,
    };
    let reason = match termination {
        T::Converged | T::GradientSmall => return None,
        T::Cancelled => {
            return Some(SciOxideError::execution_stopped(
                ExecutionStopReason::Cancelled,
            ));
        }
        T::TimeLimit => {
            return Some(SciOxideError::execution_stopped(
                ExecutionStopReason::DeadlineReached,
            ));
        }
        T::Infeasible => {
            return Some(SciOxideError::domain_error(
                argument,
                DomainReason::Undefined,
                "the problem is infeasible",
            ));
        }
        T::Unbounded => {
            return Some(SciOxideError::domain_error(
                argument,
                DomainReason::Undefined,
                "the objective is unbounded below",
            ));
        }
        T::NotStarted => {
            return Some(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::Inconsistent,
                "the solve has not started",
            ));
        }
        T::IterationLimit | T::SearchBudgetCompleted => ConvergenceReason::IterationLimit,
        T::EvaluationLimit => ConvergenceReason::EvaluationLimit,
        T::ProposalLimit => ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
        T::SampleLimit => ConvergenceReason::ResourceLimit(BudgetKind::Sample),
        T::NodeLimit | T::DepthLimit | T::MemoryLimit => {
            ConvergenceReason::ResourceLimit(BudgetKind::Node)
        }
        T::StepSmall => ConvergenceReason::StepSizeLimit,
        T::LineSearchFailed(_) => ConvergenceReason::NestedSolver,
        T::Stagnation
        | T::FunctionChangeSmall
        | T::PopulationStagnation
        | T::NoiseLimit
        | T::EstimatedStationarity
        | T::SuspectedInfeasible => ConvergenceReason::Stagnation,
        T::SuspectedUnbounded => ConvergenceReason::Divergence,
    };
    let diagnostics = ConvergenceDiagnostics::new().with_counts(counts);
    let diagnostics = match algorithm {
        Some(algorithm) => diagnostics.with_algorithm(algorithm),
        None => diagnostics,
    };
    Some(SciOxideError::convergence_failure(
        reason,
        diagnostics,
        "the solve ended before its stopping test passed",
    ))
}

// ---------------------------------------------------------------------------
// Optimization result
// ---------------------------------------------------------------------------

/// Holds the point, objective value, termination status and counts of a
/// minimization.
///
/// Create one with [`Self::empty`] to pass to `minimize_into`; the solver
/// overwrites it. Check [`Self::termination`] or [`Self::is_success`] before
/// using the point: an unsuccessful result still holds the best accepted point.
#[derive(Debug, Clone)]
#[must_use]
pub struct OptimizeResult<S: RealScalar> {
    pub(crate) method: OptimizeDiagnostics<S>,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) algorithm: Option<scioxide_core::AlgorithmId>,
    pub(crate) trust: scioxide_core::PartialResultTrust,
    /// Approximate minimizer.
    pub(crate) x: Vec<S>,
    /// Objective value at the minimizer, `f(x)`.
    pub(crate) fun: S,
    /// Gradient at the minimizer, if available.
    pub(crate) grad: Option<Vec<S>>,
    /// Reserved gradient storage kept while `grad` is unavailable.
    pub(crate) gradient_storage: Vec<S>,
    /// Number of iterations performed.
    pub(crate) iterations: usize,
    /// Number of function evaluations.
    pub(crate) fun_evals: usize,
    /// Number of gradient evaluations.
    pub(crate) grad_evals: usize,
    /// Reason the iteration terminated.
    pub(crate) termination: OptimizeTermination,
    /// Human-readable status message.
    pub(crate) message: String,
    /// Independent optimality diagnostics.
    pub(crate) diagnostics: OptimalityDiagnostics<S>,
}

/// Holds the diagnostics specific to the method that produced an
/// [`OptimizeResult`].
///
/// Each solver writes exactly one variant, so a result from any solver,
/// including one chosen at run time through [`crate::Optimizer`], carries its
/// method's diagnostics. They describe the run; whether it succeeded is decided
/// only by [`OptimizeResult::termination`].
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub enum OptimizeDiagnostics<S: RealScalar> {
    /// The method has no method-specific diagnostics.
    #[default]
    Unavailable,
    /// Line-search first-order methods: gradient descent, nonlinear CG, BFGS,
    /// L-BFGS, L-BFGS-B, and the penalty and augmented-Lagrangian outer loops.
    FirstOrder(crate::GlobalizationDiagnostics<S>),
    /// Modified-Newton and trust-region model observations.
    Newton(crate::NewtonDiagnostics<S>),
    /// Nelder–Mead simplex geometry and stopping tests.
    NelderMead(crate::NelderMeadDiagnostics<S>),
    /// Powell direction-set rank and stopping tests.
    Powell(crate::PowellDiagnostics<S>),
    /// SQP subproblem observations and quasi-Newton Lagrangian updates.
    Sqp(crate::SqpDiagnostics, crate::GlobalizationDiagnostics<S>),
    /// Nonlinear interior-point state and linear-system observations.
    InteriorPoint(crate::InteriorPointDiagnostics<S>),
    /// COBYLA interpolation and trust-region diagnostics.
    Cobyla(crate::CobylaDiagnostics<S>),
    /// COBYQA quadratic-model and trust-region diagnostics.
    Cobyqa(crate::CobyqaDiagnostics<S>),
}

impl<S: RealScalar> OptimizeResult<S> {
    /// Copy a validated result without changing `self` on an allocation
    /// failure. Adequately reserved targets copy without growth.
    pub(crate) fn try_copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        let gradient_capacity = self
            .grad
            .as_ref()
            .map_or(self.gradient_storage.capacity(), Vec::capacity);
        if self.x.capacity() < source.x.len()
            || source
                .grad
                .as_ref()
                .is_some_and(|gradient| gradient_capacity < gradient.len())
            || self.message.capacity() < source.message.len()
        {
            let mut candidate = Self::empty();
            candidate.try_reserve(source.x.len())?;
            candidate.copy_fields(source);
            *self = candidate;
        } else {
            self.copy_fields(source);
        }
        Ok(())
    }

    fn copy_fields(&mut self, source: &Self) {
        self.x.clear();
        self.x.extend_from_slice(&source.x);
        if let Some(gradient) = source.grad.as_ref() {
            let mut values = self
                .grad
                .take()
                .unwrap_or_else(|| core::mem::take(&mut self.gradient_storage));
            values.clear();
            values.extend_from_slice(gradient);
            self.grad = Some(values);
        } else if let Some(values) = self.grad.take() {
            self.gradient_storage = values;
        }
        self.method = source.method;
        self.counts = source.counts;
        self.algorithm = source.algorithm;
        self.trust = source.trust;
        self.fun = source.fun;
        self.iterations = source.iterations;
        self.fun_evals = source.fun_evals;
        self.grad_evals = source.grad_evals;
        self.termination = source.termination;
        self.message.clear();
        self.message.push_str(&source.message);
        self.diagnostics = source.diagnostics;
    }

    /// Whether the reserved capacity holds a `dimension`-variable result without growth.
    pub(crate) fn fits(&self, dimension: usize) -> bool {
        self.x.capacity() >= dimension
            && self
                .grad
                .as_ref()
                .map_or(self.gradient_storage.capacity(), Vec::capacity)
                >= dimension
            && self.message.capacity() >= 128
    }

    /// Returns the method-specific diagnostics.
    pub const fn method_diagnostics(&self) -> OptimizeDiagnostics<S> {
        self.method
    }
    /// Returns the iteration and evaluation counts of the solve.
    ///
    /// The counts separate the solver's own work from the work of nested solves;
    /// `total()` combines them. See
    /// [Solve accounting](scioxide_core#solve-accounting).
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }

    /// Returns the name of the algorithm that produced the result, or `None`
    /// before the first solve.
    pub const fn algorithm(&self) -> Option<scioxide_core::AlgorithmId> {
        self.algorithm
    }

    /// Returns how far the returned point has been checked: by the method's own
    /// acceptance rules, by an independent residual check, or by a verified
    /// certificate.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }

    /// Returns an empty result with status
    /// [`NotStarted`](OptimizeTermination::NotStarted), to be filled by
    /// `minimize_into`.
    pub fn empty() -> Self {
        Self {
            method: OptimizeDiagnostics::Unavailable,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            x: Vec::new(),
            fun: S::ZERO,
            grad: Some(Vec::new()),
            gradient_storage: Vec::new(),
            iterations: 0,
            fun_evals: 0,
            grad_evals: 0,
            termination: OptimizeTermination::NotStarted,
            message: String::new(),
            diagnostics: OptimalityDiagnostics::unavailable(),
        }
    }

    /// Clear observations while keeping prepared allocation capacities.
    pub(crate) fn reset_for_fit(&mut self) {
        let gradient = self
            .grad
            .take()
            .unwrap_or_else(|| core::mem::take(&mut self.gradient_storage));
        let mut next = Self::empty();
        next.x = core::mem::take(&mut self.x);
        next.x.clear();
        next.message = core::mem::take(&mut self.message);
        next.message.clear();
        next.grad = None;
        next.gradient_storage = gradient;
        next.gradient_storage.clear();
        *self = next;
    }

    /// Reserves storage for a solve with `dimension` variables, so that writing
    /// the result does not allocate.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        if dimension > self.x.capacity() {
            let additional = dimension - self.x.len();
            try_reserve(
                &mut self.x,
                additional,
                scioxide_core::quantity!("optimization_result_point"),
            )?;
        }
        let gradient = self.grad.as_mut().unwrap_or(&mut self.gradient_storage);
        if dimension > gradient.capacity() {
            try_reserve(
                gradient,
                dimension - gradient.len(),
                scioxide_core::quantity!("optimization_result_gradient"),
            )?;
        }
        if self.message.capacity() < 128 {
            self.message
                .try_reserve(128 - self.message.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("optimization_result_message"),
                        128,
                    )
                })?;
        }
        Ok(())
    }

    pub(crate) fn try_reserve_without_gradient(
        &mut self,
        dimension: usize,
    ) -> Result<(), SciOxideError> {
        if dimension > self.x.capacity() {
            let additional = dimension - self.x.len();
            try_reserve(
                &mut self.x,
                additional,
                scioxide_core::quantity!("optimization_result_point"),
            )?;
        }
        if self.message.capacity() < 128 {
            self.message
                .try_reserve(128 - self.message.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("optimization_result_message"),
                        128,
                    )
                })?;
        }
        Ok(())
    }

    /// Returns the best accepted point, the approximate minimizer.
    #[must_use]
    pub fn x(&self) -> &[S] {
        &self.x
    }

    /// Returns the objective value `f(x)` at [`Self::x`].
    #[must_use]
    pub const fn fun(&self) -> S {
        self.fun
    }

    /// Returns the gradient at [`Self::x`], or `None` if the method did not
    /// compute one.
    #[must_use]
    pub fn grad(&self) -> Option<&[S]> {
        self.grad.as_deref()
    }

    /// Returns the number of completed iterations.
    ///
    /// [`Self::counts`] records an iteration when it begins, before the
    /// stopping test, so after a stop by that test `counts().own().iterations()`
    /// can exceed this value by one.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the number of objective evaluations, including those of nested
    /// solves.
    #[must_use]
    pub const fn fun_evals(&self) -> usize {
        self.fun_evals
    }

    /// Returns the number of gradient evaluations, including those of nested
    /// solves.
    #[must_use]
    pub const fn grad_evals(&self) -> usize {
        self.grad_evals
    }

    /// Returns why the solver stopped.
    #[must_use]
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }

    /// Returns a human-readable description of the termination.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns whether [`Self::termination`] is successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Requires the method's stopping test to have passed.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionStopped` for cancellation or a deadline, `DomainError`
    /// for verified infeasibility or unboundedness, `InvalidInput` before a solve,
    /// and `ConvergenceFailure` with counts for every other unsuccessful termination.
    pub fn require_success(&self) -> Result<(), SciOxideError> {
        crate::result::termination_error(
            scioxide_core::argument!("self"),
            self.termination(),
            self.counts(),
            self.algorithm(),
        )
        .map_or(Ok(()), Err)
    }

    /// Returns the optimality residuals of the returned point.
    pub const fn diagnostics(&self) -> OptimalityDiagnostics<S> {
        self.diagnostics
    }

    /// Returns the largest scaled constraint violation, if computed.
    pub const fn primal_feasibility(&self) -> Option<S> {
        self.diagnostics.primal_feasibility()
    }

    /// Returns the Euclidean norm of the gradient of the objective or
    /// Lagrangian, if computed.
    pub const fn stationarity(&self) -> Option<S> {
        self.diagnostics.stationarity()
    }

    /// Returns the infinity norm of the multiplier sign violation, if computed.
    pub const fn dual_feasibility(&self) -> Option<S> {
        self.diagnostics.dual_feasibility()
    }

    /// Returns the infinity norm of the multiplier–slack products, in objective
    /// units, if computed.
    pub const fn complementarity(&self) -> Option<S> {
        self.diagnostics.complementarity()
    }

    /// Returns the largest distance of an integer variable from an integer, if
    /// computed.
    pub const fn integrality_violation(&self) -> Option<S> {
        self.diagnostics.integrality_violation()
    }

    /// Returns the largest scaled bound violation, if computed.
    pub const fn bound_violation(&self) -> Option<S> {
        self.diagnostics.bound_violation()
    }

    /// Returns the absolute gap between the objective and a valid lower bound,
    /// if one is known.
    pub const fn certified_optimality_gap(&self) -> Option<S> {
        self.diagnostics.certified_optimality_gap()
    }

    /// Returns the point, objective value, gradient, iteration count, objective
    /// and gradient evaluation counts, termination, optimality residuals and
    /// message, consuming the result.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Vec<S>,
        S,
        Option<Vec<S>>,
        usize,
        usize,
        usize,
        OptimizeTermination,
        OptimalityDiagnostics<S>,
        String,
    ) {
        (
            self.x,
            self.fun,
            self.grad,
            self.iterations,
            self.fun_evals,
            self.grad_evals,
            self.termination,
            self.diagnostics,
            self.message,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // ---- LineSearchResult ----

    // ---- OptimizeTermination ----

    #[test]
    fn test_optimize_termination_display() {
        assert_eq!(OptimizeTermination::Converged.to_string(), "Converged");
        assert_eq!(
            OptimizeTermination::IterationLimit.to_string(),
            "IterationLimit"
        );
        assert_eq!(
            OptimizeTermination::GradientSmall.to_string(),
            "GradientSmall"
        );
        assert_eq!(OptimizeTermination::StepSmall.to_string(), "StepSmall");
        assert_eq!(OptimizeTermination::Stagnation.to_string(), "Stagnation");
        assert_eq!(OptimizeTermination::TimeLimit.to_string(), "TimeLimit");
        assert_eq!(OptimizeTermination::NodeLimit.to_string(), "NodeLimit");
        assert_eq!(
            OptimizeTermination::SearchBudgetCompleted.to_string(),
            "SearchBudgetCompleted"
        );
        assert_eq!(
            OptimizeTermination::PopulationStagnation.to_string(),
            "PopulationStagnation"
        );
        assert_eq!(
            OptimizeTermination::FunctionChangeSmall.to_string(),
            "FunctionChangeSmall"
        );
        assert_eq!(OptimizeTermination::Infeasible.to_string(), "Infeasible");
        assert_eq!(OptimizeTermination::Unbounded.to_string(), "Unbounded");
    }

    #[test]
    fn test_optimize_termination_eq() {
        assert_eq!(
            OptimizeTermination::Converged,
            OptimizeTermination::Converged
        );
        assert_ne!(
            OptimizeTermination::Converged,
            OptimizeTermination::IterationLimit
        );
    }

    #[test]
    fn test_optimize_termination_is_success() {
        for termination in [
            OptimizeTermination::Converged,
            OptimizeTermination::GradientSmall,
        ] {
            assert!(termination.is_success());
        }
        for termination in [
            OptimizeTermination::IterationLimit,
            OptimizeTermination::TimeLimit,
            OptimizeTermination::NodeLimit,
            OptimizeTermination::EvaluationLimit,
            OptimizeTermination::SearchBudgetCompleted,
            OptimizeTermination::PopulationStagnation,
            OptimizeTermination::StepSmall,
            OptimizeTermination::FunctionChangeSmall,
            OptimizeTermination::Stagnation,
            OptimizeTermination::Infeasible,
            OptimizeTermination::Unbounded,
        ] {
            assert!(!termination.is_success());
        }
    }

    // ---- OptimizeResult ----

    #[test]
    fn test_optimize_result_fields() {
        let r = OptimizeResult {
            method: OptimizeDiagnostics::<f64>::Unavailable,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            gradient_storage: Vec::new(),
            x: vec![1.0_f64, 2.0],
            fun: 5.0,
            grad: Some(vec![2.0, 4.0]),
            iterations: 42,
            fun_evals: 100,
            grad_evals: 50,
            termination: OptimizeTermination::Converged,
            message: "diagnostic text does not define success".to_owned(),
            diagnostics: OptimalityDiagnostics::unavailable(),
        };
        assert!(r.is_success());
        assert_eq!(r.iterations, 42);
        assert_eq!(r.termination, OptimizeTermination::Converged);
    }

    // ---- LeastSquaresResult ----

    #[test]
    fn diagnostic_message_is_not_authoritative() {
        let result = OptimizeResult {
            method: OptimizeDiagnostics::<f64>::Unavailable,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            gradient_storage: Vec::new(),
            x: vec![1.0_f64],
            fun: 1.0,
            grad: None,
            iterations: 1,
            fun_evals: 1,
            grad_evals: 0,
            termination: OptimizeTermination::IterationLimit,
            message: "converged".to_owned(),
            diagnostics: OptimalityDiagnostics::unavailable(),
        };

        assert!(!result.is_success());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod termination_tests {
    use super::*;
    use scioxide_core::{
        BudgetKind, ConvergenceReason as R, DomainReason, ExecutionStopReason, InvalidInputReason,
        SciOxideErrorView,
    };

    #[test]
    fn every_termination_has_one_error_mapping() {
        use OptimizeTermination as T;
        let counts = scioxide_core::SolveCounts::from_own(
            scioxide_core::EvaluationCounts::from_iterations(7),
        );
        let algorithm = scioxide_core::algorithm_id!("optimize", "bfgs", 1);
        let argument = scioxide_core::argument!("self");
        for termination in [T::Converged, T::GradientSmall] {
            assert!(termination_error(argument, termination, counts, Some(algorithm)).is_none());
        }
        for (termination, reason) in [
            (T::IterationLimit, R::IterationLimit),
            (T::SearchBudgetCompleted, R::IterationLimit),
            (T::EvaluationLimit, R::EvaluationLimit),
            (T::ProposalLimit, R::ResourceLimit(BudgetKind::Proposal)),
            (T::SampleLimit, R::ResourceLimit(BudgetKind::Sample)),
            (T::NodeLimit, R::ResourceLimit(BudgetKind::Node)),
            (T::DepthLimit, R::ResourceLimit(BudgetKind::Node)),
            (T::MemoryLimit, R::ResourceLimit(BudgetKind::Node)),
            (T::StepSmall, R::StepSizeLimit),
            (
                T::LineSearchFailed(LineSearchFailure::IterationLimit),
                R::NestedSolver,
            ),
            (T::Stagnation, R::Stagnation),
            (T::FunctionChangeSmall, R::Stagnation),
            (T::PopulationStagnation, R::Stagnation),
            (T::NoiseLimit, R::Stagnation),
            (T::EstimatedStationarity, R::Stagnation),
            (T::SuspectedInfeasible, R::Stagnation),
            (T::SuspectedUnbounded, R::Divergence),
        ] {
            let error = termination_error(argument, termination, counts, Some(algorithm)).unwrap();
            assert!(
                matches!(error.view(),SciOxideErrorView::ConvergenceFailure { reason: actual, diagnostics, .. } if actual==reason && diagnostics.counts()==counts && diagnostics.algorithm()==Some(algorithm))
            );
        }
        for (termination, reason) in [
            (T::Cancelled, ExecutionStopReason::Cancelled),
            (T::TimeLimit, ExecutionStopReason::DeadlineReached),
        ] {
            assert!(
                matches!(termination_error(argument,termination,counts,None).unwrap().view(),SciOxideErrorView::ExecutionStopped {reason: actual,..} if actual==reason)
            );
        }
        for termination in [T::Infeasible, T::Unbounded] {
            assert!(
                matches!(termination_error(argument,termination,counts,None).unwrap().view(),SciOxideErrorView::DomainError {argument: actual,reason: DomainReason::Undefined,..} if actual==argument)
            );
        }
        assert!(
            matches!(termination_error(argument,T::NotStarted,counts,None).unwrap().view(),SciOxideErrorView::InvalidInput {argument: actual,reason: InvalidInputReason::Inconsistent,..} if actual==argument)
        );
    }
}
