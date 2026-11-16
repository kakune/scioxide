//! Mixed-integer result with incumbent, global-bound, gap and search observations.
//!
//! This is separate from [`crate::ProgrammingResult`]: a MIP result is a checked
//! integer incumbent plus a numerical global bound over every open node, not a
//! single LP/QP certificate.
use alloc::string::String;
use alloc::vec::Vec;

use scioxide_core::backend::try_reserve;
use scioxide_core::{RealScalar, SciOxideError};

use crate::{OptimalityDiagnostics, OptimizeTermination};

/// Holds the result of [`BranchAndBound`](crate::BranchAndBound).
///
/// [`Self::termination`] says why the search stopped; `Converged` means the gap
/// test of [`BranchAndBound`](crate::BranchAndBound) passed and `Infeasible`
/// that no integer point exists. When a limit stops the search, the result keeps
/// the best incumbent found. [`Self::has_incumbent`] reports whether [`Self::x`]
/// is an integer point that has been checked against the original constraints
/// and bounds. The bound and gaps are floating-point estimates, not rigorous
/// enclosures.
#[derive(Debug, Clone)]
#[must_use]
pub struct MipResult<S: RealScalar> {
    pub(crate) search: MipSummary<S>,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) algorithm: Option<scioxide_core::AlgorithmId>,
    pub(crate) trust: scioxide_core::PartialResultTrust,

    /// Solution vector.
    pub(crate) x: Vec<S>,
    /// Objective value.
    pub(crate) fun: S,
    /// Number of nodes explored.
    pub(crate) nodes: usize,
    /// Number of iterations performed.
    pub(crate) iterations: usize,
    /// Termination reason.
    pub(crate) termination: OptimizeTermination,
    /// Human-readable status message.
    pub(crate) message: String,
    /// Independent optimality diagnostics.
    pub(crate) diagnostics: OptimalityDiagnostics<S>,
}

/// Numerical search summary; floating-point bounds are not interval certificates.
#[derive(Debug, Clone)]
pub(crate) struct MipSummary<S> {
    pub(crate) has_incumbent: bool,
    pub(crate) lower_bound: Option<S>,
    pub(crate) absolute_gap: Option<S>,
    pub(crate) relative_gap: Option<S>,
    pub(crate) statistics: super::MipStatistics,
}
impl<S> Default for MipSummary<S> {
    fn default() -> Self {
        Self {
            has_incumbent: false,
            lower_bound: None,
            absolute_gap: None,
            relative_gap: None,
            statistics: super::MipStatistics::default(),
        }
    }
}

impl<S: RealScalar> MipResult<S> {
    /// Returns whether [`Self::x`] is a checked integer incumbent.
    pub const fn has_incumbent(&self) -> bool {
        self.search.has_incumbent
    }
    /// Returns the lower bound on the optimal objective over all open nodes, if known.
    pub const fn global_lower_bound(&self) -> Option<S> {
        self.search.lower_bound
    }
    /// Returns the gap `incumbent − bound`, computed in floating point, if both are known.
    pub const fn absolute_gap(&self) -> Option<S> {
        self.search.absolute_gap
    }
    /// Returns the absolute gap divided by `max(1, |incumbent|, |bound|)`, if known.
    pub const fn relative_gap(&self) -> Option<S> {
        self.search.relative_gap
    }
    /// Returns the work counters of the search.
    pub const fn statistics(&self) -> super::MipStatistics {
        self.search.statistics
    }

    /// Returns the work counts of this solve and its nested solves, including failed provider calls.
    ///
    /// See [Solve accounting](scioxide_core#solve-accounting).
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the identifier of the algorithm that produced this result.
    pub const fn algorithm(&self) -> Option<scioxide_core::AlgorithmId> {
        self.algorithm
    }
    /// Returns how far the original-problem diagnostics support the returned point.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }
    /// Creates an empty result without allocating, for use as `output` of a `solve_into` call.
    pub fn empty() -> Self {
        Self {
            search: MipSummary::default(),
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            x: Vec::new(),
            fun: S::ZERO,
            iterations: 0,
            nodes: 0,
            termination: OptimizeTermination::NotStarted,
            message: String::new(),
            diagnostics: OptimalityDiagnostics::unavailable(),
        }
    }
    /// Reserves storage for a point of length `dimension` and the termination message.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated; the result is then unchanged.
    pub fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        let additional = dimension.saturating_sub(self.x.len());
        try_reserve(
            &mut self.x,
            additional,
            scioxide_core::quantity!("programming_result_point"),
        )?;
        if self.message.capacity() < 256 {
            self.message
                .try_reserve(256 - self.message.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("programming_result_message"),
                        256,
                    )
                })?;
        }
        Ok(())
    }

    /// Returns the best integer point found.
    #[must_use]
    pub fn x(&self) -> &[S] {
        &self.x
    }

    /// Returns the objective value at [`Self::x`].
    #[must_use]
    pub const fn fun(&self) -> S {
        self.fun
    }

    /// Returns the number of processed nodes.
    #[must_use]
    pub const fn nodes(&self) -> usize {
        self.nodes
    }

    /// Returns the total number of iterations of the relaxation solves.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the reason the search stopped.
    #[must_use]
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }

    /// Returns a human-readable description of the termination.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns whether [`Self::termination`] is a success.
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

    /// Returns the optimality diagnostics of the returned point.
    pub const fn diagnostics(&self) -> OptimalityDiagnostics<S> {
        self.diagnostics
    }

    /// Returns the largest scaled constraint violation, if computed.
    pub const fn primal_feasibility(&self) -> Option<S> {
        self.diagnostics.primal_feasibility()
    }

    /// Returns the Euclidean norm of the gradient of the objective or Lagrangian, if computed.
    pub const fn stationarity(&self) -> Option<S> {
        self.diagnostics.stationarity()
    }

    /// Returns the infinity norm of the multiplier sign violation, if computed.
    pub const fn dual_feasibility(&self) -> Option<S> {
        self.diagnostics.dual_feasibility()
    }

    /// Returns the infinity norm of the complementarity products in objective units, if computed.
    pub const fn complementarity(&self) -> Option<S> {
        self.diagnostics.complementarity()
    }

    /// Returns the largest distance of an integer variable from an integer, if computed.
    pub const fn integrality_violation(&self) -> Option<S> {
        self.diagnostics.integrality_violation()
    }

    /// Returns the largest scaled variable-bound violation, if computed.
    pub const fn bound_violation(&self) -> Option<S> {
        self.diagnostics.bound_violation()
    }

    /// Returns a rigorous optimality gap, which branch and bound does not compute.
    ///
    /// Use [`Self::absolute_gap`] for the floating-point gap.
    pub const fn certified_optimality_gap(&self) -> Option<S> {
        self.diagnostics.certified_optimality_gap()
    }

    /// Returns the solution, objective, nodes, iterations, termination, diagnostics and message.
    pub fn into_parts(
        self,
    ) -> (
        Vec<S>,
        S,
        usize,
        usize,
        OptimizeTermination,
        OptimalityDiagnostics<S>,
        String,
    ) {
        (
            self.x,
            self.fun,
            self.nodes,
            self.iterations,
            self.termination,
            self.diagnostics,
            self.message,
        )
    }
}

impl<S: RealScalar> MipResult<S> {
    /// Copy `source` into `self`; every allocation precedes the first write.
    pub(crate) fn try_copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        let point = crate::programming::reserved_point(self.x.capacity(), source.x.len())?;
        let message =
            crate::programming::reserved_message(self.message.capacity(), source.message.len())?;
        if let Some(point) = point {
            self.x = point;
        }
        if let Some(message) = message {
            self.message = message;
        }
        self.x.clear();
        self.x.extend_from_slice(&source.x);
        self.message.clear();
        self.message.push_str(&source.message);
        self.search = source.search.clone();
        self.counts = source.counts;
        self.algorithm = source.algorithm;
        self.trust = source.trust;
        self.fun = source.fun;
        self.nodes = source.nodes;
        self.iterations = source.iterations;
        self.termination = source.termination;
        self.diagnostics = source.diagnostics;
        Ok(())
    }
    /// Progress of a written incumbent, absent without one.
    pub(crate) fn progress(&self) -> Option<crate::ProgrammingProgress<S>> {
        self.search
            .has_incumbent
            .then_some(crate::ProgrammingProgress::new(
                self.fun,
                self.iterations,
                self.trust,
            ))
    }
}

impl<S: RealScalar> crate::attempt::Staged for MipResult<S> {
    fn reset(&mut self) {
        self.search = MipSummary::default();
        self.counts = scioxide_core::SolveCounts::new();
        self.algorithm = None;
        self.trust = scioxide_core::PartialResultTrust::DiagnosticOnly;
        self.x.clear();
        self.fun = S::ZERO;
        self.nodes = 0;
        self.iterations = 0;
        self.termination = OptimizeTermination::NotStarted;
        self.message.clear();
        self.diagnostics = OptimalityDiagnostics::unavailable();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.try_copy_from(source)
    }
    fn succeeded(&self) -> bool {
        self.is_success()
    }
}
