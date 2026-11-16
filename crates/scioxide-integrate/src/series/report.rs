//! Series termination, error estimates and reports.

use scioxide_core::{BudgetKind, ExecutionStopReason, NumericScalar, RealScalar};

/// Reason a series summation stopped.
///
/// A call that has added at least one term returns `Ok` with its estimate and
/// one of these values. A limit, cancellation or deadline that stops a call
/// before any term was added returns an error instead, as described in
/// [Termination](crate#termination).
///
/// Success means that the method finished as configured, as for integrals.
/// Agreement of successive estimates of Levin's or Wynn's transform does not
/// prove convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesTermination {
    /// The Euler–Maclaurin error bound met the tolerance. The bound holds only
    /// under the assumptions stated on
    /// [`EulerMaclaurinProvider`](crate::EulerMaclaurinProvider).
    BoundedTailReached,
    /// The terms have zero components, so the sum is the empty vector; no
    /// provider is called.
    ExactEmpty,
    /// The error estimate of Levin's or Wynn's transform met the tolerance for
    /// two consecutive new estimates. Convergence is not proved.
    HeuristicToleranceReached,
    /// The call added
    /// [`terms_per_advance`](crate::SeriesOptions::terms_per_advance) terms;
    /// another call to [`SeriesSummation::advance`](crate::SeriesSummation::advance) can
    /// continue with the same state.
    CallLimit,
    /// The state holds [`max_terms`](crate::SeriesOptions::max_terms) terms
    /// without meeting the tolerance.
    TermLimit,
    /// The next term would exceed an evaluation or other limit of the solve
    /// context, such as [`max_evaluations`](crate::SeriesOptions::max_evaluations).
    ContextBudgetLimit(BudgetKind),
    /// Cancellation or a deadline stopped the summation before the next term.
    ExecutionStopped(ExecutionStopReason),
}

impl SeriesTermination {
    /// Returns `true` for `BoundedTailReached`, `ExactEmpty` and `HeuristicToleranceReached`.
    /// Success does not prove convergence or bound the true error.
    pub const fn is_success(self) -> bool {
        matches!(
            self,
            Self::BoundedTailReached | Self::ExactEmpty | Self::HeuristicToleranceReached
        )
    }
}

impl core::fmt::Display for SeriesTermination {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Error estimates of one component of a series estimate.
///
/// Each method fills the components it estimates. A component that is `None`
/// was not estimated; it is not known to be zero. None of the estimates is a
/// bound on the true error, except `tail_bound` plus `provider` under the
/// assumptions of [`EulerMaclaurinProvider`](crate::EulerMaclaurinProvider);
/// see [Error estimates](crate#error-estimates).
///
/// The error that is tested against the tolerance, and used to choose between
/// estimates, is the larger of `acceleration` and `roundoff` for a Levin or
/// Wynn estimate, and `provider + tail_bound + roundoff` for Euler–Maclaurin.
/// When the transform gives no estimate, the value is the partial sum. It is
/// ranked by the larger of `remainder` and `roundoff`, and it never stops the
/// summation.
#[derive(Debug, Clone, Copy)]
pub struct SeriesError<R: RealScalar> {
    /// Largest of the transform's internal change and the last two changes
    /// between successive estimates, divided by `1 − ρ` above rounding level,
    /// where `ρ` is its ratio to the previous value; `None` when `ρ ≥ 1`.
    pub(super) acceleration: Option<R>,
    /// `max (n + 1) |aₙ|` over a full window, for a partial sum.
    pub(super) remainder: Option<R>,
    /// Euler–Maclaurin provider errors, weighted by the Bernoulli coefficients.
    pub(super) provider: Option<R>,
    /// Euler–Maclaurin remainder bound.
    pub(super) tail_bound: Option<R>,
    /// Estimated rounding error.
    pub(super) roundoff: R,
}

impl<R: RealScalar> Default for SeriesError<R> {
    /// Returns error estimates with every optional component `None` and zero
    /// rounding error.
    fn default() -> Self {
        Self {
            acceleration: None,
            remainder: None,
            provider: None,
            tail_bound: None,
            roundoff: R::ZERO,
        }
    }
}

impl<R: RealScalar> SeriesError<R> {
    /// Returns the heuristic error estimate of a Levin or Wynn estimate.
    ///
    /// It is the largest of the change inside the transform and the last two
    /// changes between successive estimates. If this change is above rounding
    /// level and decreases by a ratio `ρ < 1` from one term to the next, it is
    /// divided by `1 − ρ` to account for a geometric tail. It is `None` if the
    /// change does not decrease, or if the value is a partial sum.
    pub const fn acceleration(&self) -> Option<R> {
        self.acceleration
    }
    /// Returns `max (n + 1) |aₙ|` over the last `order + 1` terms, when the
    /// value is a partial sum of Levin or Wynn and `order + 1` terms are
    /// available.
    ///
    /// This is the size of the first omitted terms under Levin's remainder
    /// model, not a bound.
    pub const fn remainder(&self) -> Option<R> {
        self.remainder
    }
    /// Returns the Euler–Maclaurin error from the provider: its error bounds on
    /// the tail integral and the odd derivatives, weighted by `|B₂ⱼ|/(2j)!`.
    pub const fn provider(&self) -> Option<R> {
        self.provider
    }
    /// Returns the Euler–Maclaurin remainder bound
    /// `(|B₂ₚ|/(2p)!) ∫ |f^(2p)(x)| dx`, where `p` is the order.
    pub const fn tail_bound(&self) -> Option<R> {
        self.tail_bound
    }
    /// Returns the estimated rounding error of the value and of the inputs of
    /// the transform.
    pub const fn roundoff(&self) -> R {
        self.roundoff
    }
}

/// Termination and work counts of one series call.
#[derive(Debug, Clone, Copy)]
pub struct SeriesReport {
    pub(super) termination: SeriesTermination,
    pub(super) terms: usize,
    pub(super) provider_calls: usize,
    pub(super) accepted_state: Option<scioxide_core::AcceptedStateMetadata>,
}

impl SeriesReport {
    pub(super) fn has_estimate(&self) -> bool {
        self.terms > 0 || self.termination == SeriesTermination::ExactEmpty
    }
    /// Returns why the summation stopped.
    pub const fn termination(&self) -> SeriesTermination {
        self.termination
    }
    /// Returns the number of terms summed, including those of earlier calls on
    /// the same state.
    pub const fn terms(&self) -> usize {
        self.terms
    }
    /// Returns the number of provider calls in this call, including failed
    /// calls.
    ///
    /// With a parallel execution policy, it can exceed the number of new terms
    /// by up to 31, the terms of a batch that were evaluated after the sum
    /// converged.
    pub const fn provider_calls(&self) -> usize {
        self.provider_calls
    }
    /// Returns the [`AcceptedStateMetadata`](scioxide_core::AcceptedStateMetadata)
    /// of the last term added, or `None` if the state holds no term.
    pub const fn accepted_state(&self) -> Option<scioxide_core::AcceptedStateMetadata> {
        self.accepted_state
    }
}

/// Value, error estimates and report of a scalar series, returned by
/// [`SeriesSummation::sum`](crate::SeriesSummation::sum).
#[derive(Debug, Clone, Copy)]
pub struct SeriesResult<T: NumericScalar> {
    pub(super) value: T,
    pub(super) error: SeriesError<T::Real>,
    pub(super) report: SeriesReport,
}

impl<T: NumericScalar> SeriesResult<T> {
    /// Returns the estimated sum.
    pub const fn value(&self) -> T {
        self.value
    }
    /// Returns the error estimates of the sum.
    pub const fn error(&self) -> &SeriesError<T::Real> {
        &self.error
    }
    /// Returns the termination and work counts.
    pub const fn report(&self) -> &SeriesReport {
        &self.report
    }
}
