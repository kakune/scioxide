//! Termination statuses and results of scalar, complex and system solves.

#[cfg(test)]
use alloc::string::ToString;
use core::fmt;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Vector;

// ---------------------------------------------------------------------------
// Termination reason
// ---------------------------------------------------------------------------

/// Reports why a root-finding solve stopped.
///
/// Every result in this crate carries one `RootTermination`. Only the statuses
/// listed under [`RootTermination::is_success`] mean that the returned point
/// passed the method's acceptance test. Apart from `NotStarted`, every other
/// status comes with the last accepted point, which may still be useful; check
/// its residual before you use it. See [Termination](crate#termination) for how each solver family uses
/// these statuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootTermination {
    /// The result was reserved but no solve has written to it yet.
    NotStarted,
    /// A continuation step accepted a new point and the target is not yet reached.
    Advanced,
    /// Every requested root was found and checked, or a continuation reached its target.
    Complete,
    /// The requested sequence of special-function zeros is empty.
    NoZeros,
    /// The residual at the returned point meets the residual tolerance.
    ResidualTolerance,
    /// An open method's last step was below the step tolerance and the residual
    /// fell below the default relative residual threshold.
    StepTolerance,
    /// The returned sign-change bracket is narrower than the step tolerance.
    BracketTolerance,
    /// A real Muller step would leave the real line; use a complex solver.
    ComplexStepRequired,
    /// The line search or trust region found no acceptable step from the last
    /// accepted point.
    GlobalizationFailure,
    /// The iteration limit was reached.
    IterationLimit,
    /// The iterates stopped making progress before the acceptance test passed.
    Stagnation,
    /// A work budget of this kind was exhausted, in this solve or in an
    /// enclosing one (see [Solve accounting](scioxide_core#solve-accounting)).
    WorkLimit(scioxide_core::BudgetKind),
    /// An inner iterative linear solve did not converge or broke down.
    LinearFailure(scioxide_linalg::LinearSolveTermination),
    /// The continuation history buffer is full.
    HistoryLimit,
    /// Continuation used up its retries or reached the minimum step length.
    StepLimit,
    /// Continuation could not compute a reliable tangent at the minimum step, or
    /// natural-parameter continuation reached a fold.
    TangentFailure,
    /// A computed root failed its check: the residual of the original equation,
    /// separation from other roots, or the requested index or branch.
    Unverified,
    /// A polynomial root computation broke down or left the floating-point range.
    NumericalFailure(scioxide_core::NumericalIssue),
    /// The solve was cancelled through the execution context.
    Cancelled,
    /// The deadline of the execution context passed.
    TimeLimit,
}

impl fmt::Display for RootTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl RootTermination {
    /// Returns `true` for the successful statuses: `ResidualTolerance`,
    /// `StepTolerance`, `BracketTolerance`, `Complete` and `NoZeros`.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(
            self,
            Self::ResidualTolerance
                | Self::StepTolerance
                | Self::BracketTolerance
                | Self::Complete
                | Self::NoZeros
        )
    }

    pub(crate) const fn from_execution(reason: scioxide_core::ExecutionStopReason) -> Self {
        match reason {
            scioxide_core::ExecutionStopReason::DeadlineReached => Self::TimeLimit,
            _ => Self::Cancelled,
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar root result
// ---------------------------------------------------------------------------

/// Holds the result of a real scalar root solve.
///
/// The result stores the approximate root `x`, the value `f(x)`, the
/// [`RootTermination`], the work counts and, for bracketing methods, the final
/// sign-change bracket.
#[derive(Debug, Clone, Copy)]
pub struct RootResult<S: RealScalar> {
    /// Approximate root.
    pub(crate) root: S,
    /// Value of the function at the root, `f(root)`.
    pub(crate) function_value: S,
    pub(crate) termination: RootTermination,
    pub(crate) bracket: Option<crate::Bracket<S>>,
    pub(crate) algorithm: scioxide_core::AlgorithmId,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) residual: scioxide_core::ResidualReport<S>,
    pub(crate) trust: scioxide_core::PartialResultTrust,
}

impl<S: RealScalar> RootResult<S> {
    /// Returns the identifier of the method that produced this result.
    pub const fn algorithm(&self) -> scioxide_core::AlgorithmId {
        self.algorithm
    }
    /// Returns the iterations and callback evaluations of this solve, without
    /// the work of nested solves.
    pub const fn counts(&self) -> scioxide_core::EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this solve and of its nested solves, counted separately.
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of iterations.
    pub const fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns `|f(x)|` at the returned point `x`.
    pub const fn residual_norm(&self) -> S {
        self.residual.raw_norm()
    }
    /// Returns the residual norm in function units and scaled by the residual
    /// reference scale.
    pub const fn residual_report(&self) -> scioxide_core::ResidualReport<S> {
        self.residual
    }
    /// Returns how far the returned point can be trusted. Scalar results are
    /// always `ResidualChecked`: the residual was evaluated at the returned point.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }

    /// Returns the approximate root `x`.
    #[must_use]
    pub const fn point(&self) -> S {
        self.root
    }

    /// Returns the function value at the approximate root.
    #[must_use]
    pub const fn residual(&self) -> S {
        self.function_value
    }

    /// Returns why the solve stopped.
    #[must_use]
    pub const fn termination(&self) -> RootTermination {
        self.termination
    }

    /// Returns the final sign-change bracket of a bracketing method, or `None`
    /// for open methods.
    #[must_use]
    pub const fn bracket(&self) -> Option<crate::Bracket<S>> {
        self.bracket
    }

    /// Returns `true` if the termination status is successful
    /// (see [`RootTermination::is_success`]).
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
}

/// Holds the result of a complex scalar root solve.
#[derive(Debug, Clone, Copy)]
pub struct ComplexRootResult<S: RealScalar> {
    /// Approximate root.
    pub(crate) root: scioxide_core::Complex<S>,
    /// Value of the function at the root, `f(root)`.
    pub(crate) function_value: scioxide_core::Complex<S>,
    pub(crate) termination: RootTermination,
    pub(crate) algorithm: scioxide_core::AlgorithmId,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) residual: scioxide_core::ResidualReport<S>,
    pub(crate) trust: scioxide_core::PartialResultTrust,
}

impl<S: RealScalar> ComplexRootResult<S> {
    /// Returns the identifier of the method that produced this result.
    pub const fn algorithm(&self) -> scioxide_core::AlgorithmId {
        self.algorithm
    }
    /// Returns the iterations and callback evaluations of this solve, without
    /// the work of nested solves.
    pub const fn counts(&self) -> scioxide_core::EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this solve and of its nested solves, counted separately.
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of iterations.
    pub const fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns `|f(z)|` at the returned point `z`.
    pub const fn residual_norm(&self) -> S {
        self.residual.raw_norm()
    }
    /// Returns the residual norm in function units and scaled by the residual
    /// reference scale.
    pub const fn residual_report(&self) -> scioxide_core::ResidualReport<S> {
        self.residual
    }
    /// Returns how far the returned point can be trusted. Scalar results are
    /// always `ResidualChecked`: the residual was evaluated at the returned point.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }

    /// Returns the approximate root `z`.
    #[must_use]
    pub const fn point(&self) -> scioxide_core::Complex<S> {
        self.root
    }

    /// Returns the function value at the approximate root.
    #[must_use]
    pub const fn residual(&self) -> scioxide_core::Complex<S> {
        self.function_value
    }

    /// Returns why the solve stopped.
    #[must_use]
    pub const fn termination(&self) -> RootTermination {
        self.termination
    }

    /// Returns `true` if the termination status is successful
    /// (see [`RootTermination::is_success`]).
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
}

// ---------------------------------------------------------------------------
// System root result
// ---------------------------------------------------------------------------

/// Holds the result of a nonlinear system solve `F(x) = 0`.
///
/// Create one with [`Self::empty`] and size it with [`Self::try_reserve`] before
/// a `solve_into` call; the solve overwrites it. The owned `solve` forms return
/// a filled result.
#[derive(Debug, Clone)]
pub struct SystemRootResult<S: RealScalar + NumericScalar<Real = S>> {
    /// Approximate root.
    pub(crate) root: Vector<S>,
    /// Function value at the root, `F(root)`.
    pub(crate) function_value: Vector<S>,
    /// Work of this solve and of its nested solves.
    pub(crate) counts: scioxide_core::SolveCounts,
    /// Residual norms, absent before the first solve.
    pub(crate) residual: Option<scioxide_core::ResidualReport<S>>,
    /// Why the solve stopped.
    pub(crate) termination: RootTermination,
    /// Method identifier, absent before the first solve.
    pub(crate) algorithm: Option<scioxide_core::AlgorithmId>,
}

impl<S: RealScalar + NumericScalar<Real = S>> SystemRootResult<S> {
    /// Returns the iterations and callback evaluations of the last solve,
    /// without the work of nested solves.
    pub const fn counts(&self) -> scioxide_core::EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of the last solve and of its nested solves, counted
    /// separately.
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the identifier of the method, or `None` before the first solve.
    pub const fn algorithm(&self) -> Option<scioxide_core::AlgorithmId> {
        self.algorithm
    }
    /// Returns the residual norms, or `None` before the first solve.
    pub const fn residual_report(&self) -> Option<scioxide_core::ResidualReport<S>> {
        self.residual
    }
    /// Returns `ResidualChecked` after a solve has evaluated the residual at
    /// the returned point, and `DiagnosticOnly` before.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        if self.residual.is_some() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        }
    }
    /// Creates an empty result with status [`RootTermination::NotStarted`].
    ///
    /// It does not allocate. Size it with [`Self::try_reserve`] before passing it
    /// to `solve_into`.
    pub fn empty() -> Self {
        Self {
            root: Vector::empty(),
            function_value: Vector::empty(),
            counts: scioxide_core::SolveCounts::new(),
            residual: None,
            termination: RootTermination::NotStarted,
            algorithm: None,
        }
    }

    /// Sizes the point and residual vectors for `dimension` unknowns.
    ///
    /// Existing storage is reused when it is large enough. A change of dimension
    /// resets the status to [`RootTermination::NotStarted`]. `solve_into`
    /// requires vectors of exactly the problem dimension and never resizes them.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    /// - `AllocationFailure` if the vectors cannot be grown.
    pub fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        if dimension == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "system dimension must be at least one",
            ));
        }
        if self.root.len() != dimension || self.function_value.len() != dimension {
            self.residual = None;
            self.termination = RootTermination::NotStarted;
            self.algorithm = None;
            self.counts = scioxide_core::SolveCounts::new();
        }
        self.root
            .try_resize(dimension, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        self.function_value
            .try_resize(dimension, <S as scioxide_core::AlgebraicScalar>::ZERO)
    }

    /// Returns the approximate root `x`.
    pub fn point(&self) -> &[S] {
        self.root.as_slice()
    }

    /// Returns the residual `F(x)` at the approximate root.
    pub fn residual(&self) -> &[S] {
        self.function_value.as_slice()
    }

    /// Returns the number of iterations of the last solve.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }

    /// Returns the residual norm `‖F(x)‖₂` in function units, or `None` before
    /// the first solve.
    #[must_use]
    pub fn residual_norm(&self) -> Option<S> {
        self.residual.map(|report| report.raw_norm())
    }

    /// Returns why the solve stopped.
    #[must_use]
    pub const fn termination(&self) -> RootTermination {
        self.termination
    }

    /// Returns `true` if the termination status is successful
    /// (see [`RootTermination::is_success`]).
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for SystemRootResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_termination_display() {
        assert_eq!(
            RootTermination::ResidualTolerance.to_string(),
            "ResidualTolerance"
        );
        assert_eq!(
            RootTermination::IterationLimit.to_string(),
            "IterationLimit"
        );
        assert_eq!(
            RootTermination::BracketTolerance.to_string(),
            "BracketTolerance"
        );
        assert_eq!(RootTermination::Stagnation.to_string(), "Stagnation");
    }

    #[test]
    fn test_root_termination_eq() {
        assert_eq!(
            RootTermination::ResidualTolerance,
            RootTermination::ResidualTolerance
        );
        assert_ne!(RootTermination::Stagnation, RootTermination::IterationLimit);
        assert!(RootTermination::ResidualTolerance.is_success());
        assert!(RootTermination::BracketTolerance.is_success());
        assert!(!RootTermination::Stagnation.is_success());
        assert!(!RootTermination::IterationLimit.is_success());
    }
}
