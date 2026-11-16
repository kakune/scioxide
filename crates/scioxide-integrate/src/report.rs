//! Termination status, error estimates and the integration report.

use core::fmt;

use scioxide_core::{AffineValue, AlgebraicScalar, BudgetKind, ExecutionStopReason, RealScalar};

/// Reason an integration stopped.
///
/// An `Ok` result always carries an estimate of the whole integral. If a limit,
/// cancellation or deadline stops a method after it has such an estimate, the
/// method returns it with an unsuccessful termination. If no estimate of the whole
/// integral exists yet, the method returns an error instead, as described in
/// [Termination](crate#termination).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum IntegrationTermination {
    /// Cancellation or a deadline stopped the computation.
    ExecutionStopped(ExecutionStopReason),
    /// The next step would exceed an evaluation, subdivision or other limit.
    BudgetLimit(BudgetKind),
    /// A rule of fixed order was applied once; no tolerance was checked.
    FixedRuleCompleted,
    /// The error estimate met the tolerance.
    ToleranceReached,
    /// A heuristic error estimate, such as an observed tail decay, met the tolerance.
    HeuristicToleranceReached,
    /// The maximum refinement level or number of terms was reached before the tolerance.
    RefinementLimit,
    /// No representable evaluation point exists for the next refinement.
    GeometryFailure,
    /// Further refinement cannot change the estimate in floating-point arithmetic.
    Stagnation,
    /// Extrapolation indicates divergence or very slow convergence.
    ProbablyDivergent,
    /// A Monte Carlo method drew its configured number of samples.
    SampleBudgetCompleted,
    /// The domain has zero measure, so the result is exactly zero.
    DegenerateDomain,
}

impl IntegrationTermination {
    /// Returns `true` for `FixedRuleCompleted`, `ToleranceReached`,
    /// `HeuristicToleranceReached`, `SampleBudgetCompleted` and `DegenerateDomain`.
    ///
    /// Success means that the method finished as configured. It does not bound the
    /// true error; see [Error estimates](crate#error-estimates).
    ///
    /// ```
    /// use scioxide_integrate::IntegrationTermination;
    ///
    /// assert!(IntegrationTermination::HeuristicToleranceReached.is_success());
    /// assert!(!IntegrationTermination::Stagnation.is_success());
    /// ```
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(
            self,
            Self::FixedRuleCompleted
                | Self::ToleranceReached
                | Self::HeuristicToleranceReached
                | Self::SampleBudgetCompleted
                | Self::DegenerateDomain
        )
    }
}

impl fmt::Display for IntegrationTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Kind of an error estimate.
///
/// Neither kind is a bound on the true error; see
/// [Error estimates](crate#error-estimates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorReliability {
    /// Estimated from the difference of two rules or refinement levels, or from a rounding-error model.
    Estimated,
    /// Inferred from the observed decay of contributions; parts of the integrand that
    /// were never sampled can make it arbitrarily wrong.
    Heuristic,
}

/// Nonnegative error estimate together with its kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorEstimate<S: RealScalar> {
    pub(crate) magnitude: S,
    pub(crate) reliability: ErrorReliability,
}

impl<S: RealScalar> ErrorEstimate<S> {
    pub(crate) const fn new(magnitude: S, reliability: ErrorReliability) -> Self {
        Self {
            magnitude,
            reliability,
        }
    }

    /// Returns the estimated absolute error.
    #[must_use]
    pub const fn magnitude(&self) -> S {
        self.magnitude
    }

    /// Returns how the estimate was obtained.
    #[must_use]
    pub const fn reliability(&self) -> ErrorReliability {
        self.reliability
    }

    fn max(self, other: Self) -> Self {
        Self {
            magnitude: self.magnitude.max(other.magnitude),
            reliability: if self.reliability == ErrorReliability::Heuristic {
                self.reliability
            } else {
                other.reliability
            },
        }
    }
}

/// State of the truncated tail of a double-exponential sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailStatus {
    /// Too few unit-width shells were completed to truncate the sum.
    Open,
    /// Consecutive shells were negligible, so the sum was truncated. Peaks of the
    /// integrand beyond the explored range are not detected.
    HeuristicallyTruncated,
    /// No further representable nodes exist, or the maximum number of shells was
    /// explored.
    Saturated,
    /// Shell contributions above the tolerance stopped decreasing and do not
    /// change the rounded sum.
    Stagnated,
}

impl TailStatus {
    /// Order used to pick the weakest status over several value components.
    const fn weakness(self) -> u8 {
        match self {
            Self::HeuristicallyTruncated => 0,
            Self::Saturated => 1,
            Self::Stagnated => 2,
            Self::Open => 3,
        }
    }
}

/// Diagnostics of the truncated tail of a double-exponential sum.
///
/// Coordinates are those of the transformed variable `t`, in which the sum runs
/// over nodes `t = kh`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailDiagnostics<S: RealScalar> {
    pub(crate) extent: S,
    pub(crate) shells: usize,
    pub(crate) status: TailStatus,
}

impl<S: RealScalar> TailDiagnostics<S> {
    /// Returns the largest `|t|` that was evaluated.
    #[must_use]
    pub const fn extent(&self) -> S {
        self.extent
    }

    /// Returns the number of completed shells `k ≤ |t| < k + 1` at the final step size.
    #[must_use]
    pub const fn shells(&self) -> usize {
        self.shells
    }

    /// Returns the tail status.
    #[must_use]
    pub const fn status(&self) -> TailStatus {
        self.status
    }

    fn weakest(self, other: Self) -> Self {
        Self {
            extent: self.extent.min(other.extent),
            shells: self.shells.min(other.shells),
            status: if other.status.weakness() > self.status.weakness() {
                other.status
            } else {
                self.status
            },
        }
    }
}

/// Components of a deterministic error estimate.
///
/// Each method fills the components it estimates. A component that is `None`
/// was not estimated; it is not known to be zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrationErrorComponents<S: RealScalar> {
    pub(crate) discretization: Option<ErrorEstimate<S>>,
    pub(crate) roundoff: Option<S>,
    pub(crate) inner: Option<ErrorEstimate<S>>,
    pub(crate) moments: Option<ErrorEstimate<S>>,
    pub(crate) extrapolation: Option<ErrorEstimate<S>>,
    pub(crate) tail: Option<ErrorEstimate<S>>,
    pub(crate) tail_diagnostics: Option<TailDiagnostics<S>>,
}

impl<S: RealScalar> Default for IntegrationErrorComponents<S> {
    /// Returns components with nothing estimated.
    fn default() -> Self {
        Self {
            discretization: None,
            roundoff: None,
            inner: None,
            moments: None,
            extrapolation: None,
            tail: None,
            tail_diagnostics: None,
        }
    }
}

impl<S: RealScalar> IntegrationErrorComponents<S> {
    pub(crate) const fn discretization_only(magnitude: S) -> Self {
        Self {
            discretization: Some(ErrorEstimate::new(magnitude, ErrorReliability::Estimated)),
            roundoff: None,
            inner: None,
            moments: None,
            extrapolation: None,
            tail: None,
            tail_diagnostics: None,
        }
    }

    pub(crate) const fn with_roundoff(mut self, roundoff: S) -> Self {
        self.roundoff = Some(roundoff);
        self
    }

    /// Returns the difference between two embedded rules or two refinement levels.
    #[must_use]
    pub const fn discretization(&self) -> Option<ErrorEstimate<S>> {
        self.discretization
    }

    /// Returns the estimated rounding error of the sum, the level below which the
    /// discretization estimate is not meaningful.
    #[must_use]
    pub const fn roundoff(&self) -> Option<S> {
        self.roundoff
    }

    /// Returns the error of inner integrals, weighted by the outer rule, for
    /// iterated integrals.
    #[must_use]
    pub const fn inner(&self) -> Option<ErrorEstimate<S>> {
        self.inner
    }

    /// Returns the error in the modified moments of a weighted rule.
    #[must_use]
    pub const fn moments(&self) -> Option<ErrorEstimate<S>> {
        self.moments
    }

    /// Returns the error of a sequence extrapolation.
    #[must_use]
    pub const fn extrapolation(&self) -> Option<ErrorEstimate<S>> {
        self.extrapolation
    }

    /// Returns the estimated contribution of a truncated tail.
    #[must_use]
    pub const fn tail(&self) -> Option<ErrorEstimate<S>> {
        self.tail
    }

    /// Returns the tail diagnostics of a double-exponential sum or a Fourier tail.
    #[must_use]
    pub const fn tail_diagnostics(&self) -> Option<TailDiagnostics<S>> {
        self.tail_diagnostics
    }

    /// Returns the combined error estimate
    ///
    /// ```text
    /// max(discretization, roundoff) + inner + moments + extrapolation + tail,
    /// ```
    ///
    /// where missing components other than `discretization` count as zero.
    ///
    /// Returns `None` if there is no discretization estimate, if tail diagnostics
    /// exist without a tail estimate, or if the sum overflows.
    #[must_use]
    pub fn total(&self) -> Option<S> {
        let discretization = self.discretization?.magnitude;
        // Tail diagnostics without a tail magnitude means a truncated domain whose remainder
        // could not be bounded; summing the other parts would understate the true error.
        if self.tail_diagnostics.is_some() && self.tail.is_none() {
            return None;
        }
        // The roundoff floor is the resolution below which the discretization estimate is
        // itself noise, so it bounds that term from below instead of adding to it. Moment,
        // extrapolation and tail errors come from independent approximations and add.
        let total = discretization.max(self.roundoff.unwrap_or(S::ZERO))
            + self.inner.map_or(S::ZERO, |error| error.magnitude)
            + self.moments.map_or(S::ZERO, |error| error.magnitude)
            + self.extrapolation.map_or(S::ZERO, |error| error.magnitude)
            + self.tail.map_or(S::ZERO, |tail| tail.magnitude);
        total.is_finite().then_some(total)
    }

    /// Componentwise maximum over real value components, as reported by
    /// [`IntegrationReport`]. A discretization or tail estimate is kept only if every
    /// component has one, so the maximum never understates a component.
    pub(crate) fn envelope<'a>(components: impl IntoIterator<Item = &'a Self>) -> Option<Self>
    where
        S: 'a,
    {
        fn either<T: Copy>(a: Option<T>, b: Option<T>, max: impl Fn(T, T) -> T) -> Option<T> {
            match (a, b) {
                (Some(a), Some(b)) => Some(max(a, b)),
                (a, b) => a.or(b),
            }
        }
        fn both<T: Copy>(a: Option<T>, b: Option<T>, max: impl Fn(T, T) -> T) -> Option<T> {
            Some(max(a?, b?))
        }
        let merge = |a: Self, b: Self| Self {
            discretization: both(a.discretization, b.discretization, ErrorEstimate::max),
            roundoff: either(a.roundoff, b.roundoff, S::max),
            inner: either(a.inner, b.inner, ErrorEstimate::max),
            moments: either(a.moments, b.moments, ErrorEstimate::max),
            extrapolation: either(a.extrapolation, b.extrapolation, ErrorEstimate::max),
            // A component whose explored tail has no magnitude keeps the envelope unbounded.
            tail: if (a.tail_diagnostics.is_some() && a.tail.is_none())
                || (b.tail_diagnostics.is_some() && b.tail.is_none())
            {
                None
            } else {
                either(a.tail, b.tail, ErrorEstimate::max)
            },
            tail_diagnostics: either(
                a.tail_diagnostics,
                b.tail_diagnostics,
                TailDiagnostics::weakest,
            ),
        };
        let mut components = components.into_iter();
        let first = *components.next()?;
        let Some(&second) = components.next() else {
            return Some(first);
        };
        Some(components.fold(merge(first, second), |a, &b| merge(a, b)))
    }
}

/// Termination, work counts and error estimate of one integration.
///
/// The value is written to the output you pass, or returned next to the report
/// by the methods that return an owned value. For complex and vector values, the
/// error components are the maximum over all real components; the workspace that
/// produced the report gives them per component.
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{FixedRule, GaussKronrod, GaussKronrodRule, IntegrationTermination};
///
/// // ∫₀¹ x² dx = 1/3; the 15-point Kronrod rule integrates x² exactly.
/// let rule = GaussKronrod::new(GaussKronrodRule::G7K15)?;
/// let (value, report) = rule.integrate(|x: f64| Ok(x * x), Interval::new(0.0, 1.0)?)?;
/// assert!((value - 1.0 / 3.0).abs() < 1e-15);
/// assert_eq!(
///     report.termination(),
///     IntegrationTermination::FixedRuleCompleted
/// );
/// assert_eq!(report.evaluations(), 15);
/// assert!(report.error_estimate().is_some());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct IntegrationReport<S: RealScalar> {
    termination: IntegrationTermination,
    evaluations: usize,
    subdivisions: usize,
    errors: Option<IntegrationErrorComponents<S>>,
}

impl<S: RealScalar> IntegrationReport<S> {
    pub(crate) const fn new(
        termination: IntegrationTermination,
        evaluations: usize,
        subdivisions: usize,
        errors: Option<IntegrationErrorComponents<S>>,
    ) -> Self {
        Self {
            termination,
            evaluations,
            subdivisions,
            errors,
        }
    }

    /// Returns the reason the integration stopped.
    pub const fn termination(&self) -> IntegrationTermination {
        self.termination
    }

    /// Returns whether the termination is successful; see
    /// [`IntegrationTermination::is_success`].
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Returns the number of integrand evaluations, including those of inner
    /// integrals.
    #[must_use]
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Returns the number of subdivisions; zero for methods that do not subdivide.
    #[must_use]
    pub const fn subdivisions(&self) -> usize {
        self.subdivisions
    }

    /// Returns the components of the deterministic error estimate.
    ///
    /// Returns `None` for rules without an embedded error estimate and for Monte
    /// Carlo methods, which report a standard error instead.
    #[must_use]
    pub const fn error_components(&self) -> Option<&IntegrationErrorComponents<S>> {
        self.errors.as_ref()
    }

    /// Returns the combined error estimate, [`IntegrationErrorComponents::total`].
    ///
    /// The estimate is not a bound; see [Error estimates](crate#error-estimates).
    #[must_use]
    pub fn error_estimate(&self) -> Option<S> {
        self.errors
            .as_ref()
            .and_then(IntegrationErrorComponents::total)
    }
}

impl<S: RealScalar> crate::workspace::AttemptProgress for IntegrationReport<S> {
    fn successful(&self) -> bool {
        self.is_success()
    }
}

/// Stores an integral value and an estimate of its absolute error.
///
/// The estimate combines the error components of [`IntegrationReport`] over
/// all real components of the value. It is not a bound on the true error.
/// A degenerate domain or fixed rule without an estimate reports zero; another
/// successful computation without a finite estimate reports the largest finite scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Integral<V: AffineValue> {
    value: V,
    error_estimate: V::Scale,
}
impl<V: AffineValue> Integral<V> {
    pub(crate) fn from_report(value: V, report: &IntegrationReport<V::Scale>) -> Self {
        let error_estimate = report.error_estimate().unwrap_or_else(|| {
            if matches!(
                report.termination(),
                IntegrationTermination::DegenerateDomain
                    | IntegrationTermination::FixedRuleCompleted
            ) {
                V::Scale::ZERO
            } else {
                V::Scale::MAX
            }
        });
        Self {
            value,
            error_estimate,
        }
    }
}
impl<V: AffineValue + Copy> Integral<V> {
    /// Returns the integral value.
    #[must_use]
    pub const fn value(&self) -> V {
        self.value
    }
    /// Returns the combined absolute error estimate, which is not a bound.
    #[must_use]
    pub const fn error_estimate(&self) -> V::Scale {
        self.error_estimate
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn estimate(magnitude: f64) -> Option<ErrorEstimate<f64>> {
        Some(ErrorEstimate::new(magnitude, ErrorReliability::Estimated))
    }

    #[test]
    fn total_combines_floor_and_additive_parts_and_rejects_overflow() {
        let mut parts = IntegrationErrorComponents::discretization_only(2.0).with_roundoff(3.0);
        parts.inner = estimate(5.0);
        assert_eq!(parts.total(), Some(8.0));
        parts.inner = estimate(f64::MAX);
        parts.discretization = estimate(f64::MAX);
        assert_eq!(parts.total(), None);
        assert_eq!(IntegrationErrorComponents::<f64>::default().total(), None);
    }

    #[test]
    fn envelope_maximizes_components_and_keeps_unbounded_tails() {
        let mut first = IntegrationErrorComponents::discretization_only(1.0).with_roundoff(4.0);
        first.extrapolation = estimate(2.0);
        let mut second = IntegrationErrorComponents::discretization_only(3.0);
        second.tail = Some(ErrorEstimate::new(0.5, ErrorReliability::Heuristic));
        second.tail_diagnostics = Some(TailDiagnostics {
            extent: 4.0,
            shells: 3,
            status: TailStatus::HeuristicallyTruncated,
        });
        let envelope = IntegrationErrorComponents::envelope([&first, &second]).unwrap();
        assert_eq!(envelope.discretization.unwrap().magnitude(), 3.0);
        assert_eq!(envelope.roundoff, Some(4.0));
        assert_eq!(envelope.extrapolation.unwrap().magnitude(), 2.0);
        assert_eq!(
            envelope.tail.unwrap().reliability(),
            ErrorReliability::Heuristic
        );
        assert_eq!(envelope.total(), Some(6.5));
        let mut open = second;
        open.tail = None;
        assert_eq!(
            IntegrationErrorComponents::envelope([&first, &open])
                .unwrap()
                .tail(),
            None
        );
        let mut missing = first;
        missing.discretization = None;
        assert_eq!(
            IntegrationErrorComponents::envelope([&missing, &second])
                .unwrap()
                .total(),
            None
        );
        assert!(IntegrationErrorComponents::<f64>::envelope([]).is_none());
    }

    #[test]
    fn integral_preserves_estimates_and_distinguishes_unknown_error_from_exact_zero() {
        let estimate = IntegrationReport::new(
            IntegrationTermination::ToleranceReached,
            15,
            0,
            Some(IntegrationErrorComponents::discretization_only(0.25_f64)),
        );
        let answer = Integral::from_report(3.0, &estimate);
        assert_eq!(answer.value(), 3.0);
        assert_eq!(answer.error_estimate(), 0.25);
        for (termination, expected) in [
            (IntegrationTermination::DegenerateDomain, 0.0),
            (IntegrationTermination::FixedRuleCompleted, 0.0),
            (IntegrationTermination::ToleranceReached, f64::MAX),
        ] {
            let report = IntegrationReport::new(termination, 0, 0, None);
            assert_eq!(
                Integral::from_report(0.0_f64, &report).error_estimate(),
                expected
            );
        }
    }
}
