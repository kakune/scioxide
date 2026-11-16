//! Result, attempt progress and output staging for LP and QP solvers.
//!
//! [`crate::SimplexSolver`], [`crate::InteriorPointLp`] and [`crate::QpSolver`]
//! return a [`ProgrammingResult`]. Mixed-integer search returns the separate
//! [`crate::MipResult`], which adds the incumbent, global bound, gap and search
//! statistics.

use alloc::string::String;
use alloc::vec::Vec;

use scioxide_core::backend::try_reserve;
use scioxide_core::{
    AttemptKind, PartialResultTrust, RealScalar, SciOxideError, SolveContext, SolveScope,
};

use crate::attempt::{Entry, Shell, Staged};
use crate::verification::OwnedProgrammingCertificate;
use crate::{OptimalityDiagnostics, OptimizeTermination};

/// Holds the result of [`SimplexSolver`](crate::SimplexSolver),
/// [`InteriorPointLp`](crate::InteriorPointLp) or [`QpSolver`](crate::QpSolver).
///
/// [`Self::termination`] says why the solver stopped; only `Converged` is
/// success. `Infeasible` and `Unbounded` are answers backed by a verified
/// certificate, while `SuspectedInfeasible` and `SuspectedUnbounded` lack one.
/// The certificate, available from [`Self::certificate`], is in original
/// coordinates: multipliers for an optimum, Farkas multipliers for
/// infeasibility, or a recession direction for unboundedness.
/// [`ProgrammingVerifier`](crate::ProgrammingVerifier) checks it independently.
/// See [Termination](crate#termination) for the statuses shared by all solvers.
#[derive(Debug, Clone)]
#[must_use]
pub struct ProgrammingResult<S: RealScalar> {
    pub(crate) evidence: Option<OwnedProgrammingCertificate<S>>,
    pub(crate) verification: Option<crate::VerificationReport<S>>,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) algorithm: Option<scioxide_core::AlgorithmId>,
    pub(crate) trust: PartialResultTrust,
    pub(crate) x: Vec<S>,
    pub(crate) fun: S,
    pub(crate) iterations: usize,
    pub(crate) termination: OptimizeTermination,
    pub(crate) message: String,
    pub(crate) diagnostics: OptimalityDiagnostics<S>,
}

impl<S: RealScalar> ProgrammingResult<S> {
    /// Creates a result from the output of an external LP solver.
    ///
    /// The result is unverified: pass it to
    /// [`ProgrammingVerifier`](crate::ProgrammingVerifier) before relying on its
    /// termination or certificate.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `x` is empty, or `x` or `fun` is not finite.
    pub fn from_candidate(
        x: Vec<S>,
        fun: S,
        termination: OptimizeTermination,
        iterations: usize,
        evidence: Option<OwnedProgrammingCertificate<S>>,
    ) -> Result<Self, SciOxideError> {
        if x.is_empty() || !fun.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "expected finite nonempty point and objective",
            ));
        }
        Ok(Self {
            x,
            fun,
            termination,
            iterations,
            evidence,
            ..Self::empty()
        })
    }

    /// Creates an empty result without allocating, for use as `output` of a `solve_into` call.
    pub fn empty() -> Self {
        Self {
            evidence: None,
            verification: None,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: PartialResultTrust::DiagnosticOnly,
            x: Vec::new(),
            fun: S::ZERO,
            iterations: 0,
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

    /// Copy `source` into `self`; every allocation precedes the first write.
    pub(crate) fn try_copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        let evidence = source
            .evidence
            .as_ref()
            .map(OwnedProgrammingCertificate::try_clone)
            .transpose()?;
        let point = reserved_point(self.x.capacity(), source.x.len())?;
        let message = reserved_message(self.message.capacity(), source.message.len())?;
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
        self.evidence = evidence;
        self.verification = source.verification;
        self.counts = source.counts;
        self.algorithm = source.algorithm;
        self.trust = source.trust;
        self.fun = source.fun;
        self.iterations = source.iterations;
        self.termination = source.termination;
        self.diagnostics = source.diagnostics;
        Ok(())
    }

    /// Returns the certificate in original coordinates, if the solver built one.
    pub fn certificate(&self) -> Option<crate::ProgrammingCertificate<'_, S>> {
        self.evidence.as_ref().map(|e| e.borrow(&self.x))
    }
    /// Returns the report of the solver's check of its certificate, if the check completed.
    pub const fn verification_report(&self) -> Option<&crate::VerificationReport<S>> {
        self.verification.as_ref()
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
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
    /// Returns the solution, or the best point found if the solve was unsuccessful.
    #[must_use]
    pub fn x(&self) -> &[S] {
        &self.x
    }
    /// Returns the objective value at [`Self::x`].
    #[must_use]
    pub const fn fun(&self) -> S {
        self.fun
    }
    /// Returns the number of iterations performed.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the reason the solver stopped.
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
    /// Returns the largest constraint violation in original coordinates, if computed.
    pub const fn primal_feasibility(&self) -> Option<S> {
        self.diagnostics.primal_feasibility()
    }
    /// Returns the infinity norm of the gradient of the Lagrangian, if computed.
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
    /// Returns the largest variable-bound violation, if computed.
    pub const fn bound_violation(&self) -> Option<S> {
        self.diagnostics.bound_violation()
    }
    /// Returns a rigorous optimality gap, which LP and QP solvers do not compute.
    ///
    /// The floating-point gap is in [`Self::verification_report`].
    pub const fn certified_optimality_gap(&self) -> Option<S> {
        self.diagnostics.certified_optimality_gap()
    }
    /// Returns the solution, objective, iterations, termination, diagnostics and message.
    pub fn into_parts(
        self,
    ) -> (
        Vec<S>,
        S,
        usize,
        OptimizeTermination,
        OptimalityDiagnostics<S>,
        String,
    ) {
        (
            self.x,
            self.fun,
            self.iterations,
            self.termination,
            self.diagnostics,
            self.message,
        )
    }
    /// Progress of a written result, absent before any solve.
    pub(crate) fn progress(&self) -> Option<ProgrammingProgress<S>> {
        (self.termination != OptimizeTermination::NotStarted).then_some(ProgrammingProgress {
            objective: self.fun,
            iterations: self.iterations,
            trust: self.trust,
        })
    }
}

impl<S: RealScalar> Staged for ProgrammingResult<S> {
    fn reset(&mut self) {
        self.evidence = None;
        self.verification = None;
        self.counts = scioxide_core::SolveCounts::new();
        self.algorithm = None;
        self.trust = PartialResultTrust::DiagnosticOnly;
        self.x.clear();
        self.fun = S::ZERO;
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

/// Fresh point storage when `capacity` cannot hold `len` values.
pub(crate) fn reserved_point<S>(
    capacity: usize,
    len: usize,
) -> Result<Option<Vec<S>>, SciOxideError> {
    Ok(if capacity >= len {
        None
    } else {
        let mut point = Vec::new();
        try_reserve(
            &mut point,
            len,
            scioxide_core::quantity!("programming_result_point"),
        )?;
        Some(point)
    })
}

/// Fresh message storage when `capacity` cannot hold `len` bytes.
pub(crate) fn reserved_message(
    capacity: usize,
    len: usize,
) -> Result<Option<String>, SciOxideError> {
    Ok(if capacity >= len {
        None
    } else {
        let mut message = String::new();
        message.try_reserve(len).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("programming_result_message"),
                len,
            )
        })?;
        Some(message)
    })
}

/// Holds the objective, iteration count and trust of one LP or QP attempt.
///
/// Present only when the attempt wrote a result; absent after an error.
#[derive(Debug, Clone, Copy)]
pub struct ProgrammingProgress<S: RealScalar> {
    objective: S,
    iterations: usize,
    trust: PartialResultTrust,
}

impl<S: RealScalar> ProgrammingProgress<S> {
    pub(crate) const fn new(objective: S, iterations: usize, trust: PartialResultTrust) -> Self {
        Self {
            objective,
            iterations,
            trust,
        }
    }
    /// Returns the objective value at the attempt's point.
    pub const fn objective(self) -> S {
        self.objective
    }
    /// Returns the number of iterations of the attempt.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns how far the diagnostics support the attempt's point.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

/// Borrows the point and progress written by the most recent LP or QP attempt.
#[derive(Debug, Clone, Copy)]
pub struct ProgrammingAccepted<'a, S: RealScalar> {
    point: &'a [S],
    progress: ProgrammingProgress<S>,
}

impl<'a, S: RealScalar> ProgrammingAccepted<'a, S> {
    /// Returns the point in original coordinates.
    pub const fn point(&self) -> &'a [S] {
        self.point
    }
    /// Returns the objective, iteration count and trust at the point.
    pub const fn progress(&self) -> ProgrammingProgress<S> {
        self.progress
    }
}

/// Borrow `point` with the progress of the most recent attempt.
pub(crate) fn last_accepted<'a, S: RealScalar>(
    shell: &Shell<ProgrammingProgress<S>>,
    point: &'a [S],
) -> Option<ProgrammingAccepted<'a, S>> {
    Some(ProgrammingAccepted {
        point,
        progress: *shell.progress()?,
    })
}

/// Run one programming attempt in an epoch of `scope` and copy its staged result.
///
/// `kernel` writes the staged result; `output` changes only when the epoch and
/// the final copy succeed. Progress is recorded only for a written result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<'scope, 'capability, S, R>(
    shell: &mut Shell<ProgrammingProgress<S>>,
    staged: &mut R,
    output: &mut R,
    scope: SolveScope<'scope, 'capability>,
    kind: AttemptKind,
    entry: Entry,
    kernel: impl FnOnce(&mut SolveContext<'_, 'capability>, &mut R) -> Result<(), SciOxideError>,
    progress: impl FnOnce(&R) -> Option<ProgrammingProgress<S>>,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    R: Staged,
{
    staged.reset();
    let (outcome, started) = shell.execute(scope, entry, |context| kernel(context, staged));
    let outcome = outcome.and_then(|()| output.publish(staged));
    let succeeded = outcome.is_ok() && staged.succeeded();
    let progress = if outcome.is_ok() {
        progress(staged)
    } else {
        None
    };
    shell.finish(kind, started, outcome.is_err(), succeeded, progress);
    outcome
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::borrow::ToOwned;
    use alloc::vec;

    #[test]
    fn staged_copy_preserves_every_observation() {
        let source = ProgrammingResult {
            termination: OptimizeTermination::Converged,
            message: "optimal".to_owned(),
            ..ProgrammingResult::from_candidate(
                vec![1.0_f64, 2.0],
                3.0,
                OptimizeTermination::Converged,
                7,
                None,
            )
            .unwrap()
        };
        let mut target = ProgrammingResult::empty();
        target.publish(&source).unwrap();
        assert!(target.is_success());
        assert_eq!(target.x(), &[1.0, 2.0]);
        assert_eq!(target.fun(), 3.0);
        assert_eq!(target.iterations(), 7);
        assert_eq!(target.message(), "optimal");
        target.reset();
        assert_eq!(target.termination(), OptimizeTermination::NotStarted);
        assert!(target.progress().is_none());
    }
}
