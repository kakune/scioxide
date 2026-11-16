//! Result for fixed-point solves.
use super::*;

/// Reports how the history of a fixed-point solve was used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedPointDiagnostics {
    /// History rank at the last proposal.
    pub(crate) history_rank: usize,
    /// Number of discarded dependent history columns or rejected updates.
    pub(crate) skipped_updates: usize,
    /// Number of history/diagonal resets.
    pub(crate) restarts: usize,
    /// Number of regularized Anderson solves.
    pub(crate) regularized_solves: usize,
}

impl FixedPointDiagnostics {
    /// Returns the number of linearly independent history pairs used in the last
    /// step: the numerical rank of the Anderson least-squares matrix, or the number
    /// of stored Broyden updates.
    pub const fn history_rank(&self) -> usize {
        self.history_rank
    }
    /// Returns the number of skipped updates: rank-deficient Anderson columns,
    /// Broyden updates with a near-zero secant denominator, and Steffensen
    /// coordinates with a near-zero denominator.
    pub const fn skipped_updates(&self) -> usize {
        self.skipped_updates
    }
    /// Returns the number of history restarts and diagonal resets.
    pub const fn restarts(&self) -> usize {
        self.restarts
    }
    /// Returns the number of Anderson least-squares solves that used Tikhonov
    /// regularization.
    pub const fn regularized_solves(&self) -> usize {
        self.regularized_solves
    }
}
/// Holds the point, residual and status of a [`FixedPointSolver`] solve.
///
/// Check [`termination`](Self::termination) before using the point. An
/// unsuccessful result from a budget, cancellation or deadline stop still holds
/// the last accepted point and its residual. Before the first solve the status is
/// [`RootTermination::NotStarted`] and [`residual_report`](Self::residual_report)
/// is `None`.
#[derive(Clone, Debug)]
pub struct FixedPointResult<T: NumericScalar> {
    pub(crate) point: Vector<T>,
    pub(crate) residual: Vector<T>,
    pub(crate) report: Option<ResidualReport<T::Real>>,
    pub(crate) raw_norm: T::Real,
    pub(crate) termination: RootTermination,
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) diagnostics: FixedPointDiagnostics,
}
impl<T: NumericScalar> FixedPointResult<T> {
    /// Creates an empty result for `dimension` unknowns, to pass to
    /// [`FixedPointSolver::solve_into`].
    ///
    /// # Errors
    ///
    /// `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(dimension: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            point: Vector::zeros(dimension)?,
            residual: Vector::zeros(dimension)?,
            report: None,
            raw_norm: T::Real::ZERO,
            termination: RootTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
            diagnostics: FixedPointDiagnostics::default(),
        })
    }
    /// Returns the last accepted point. It is meaningful only when
    /// [`residual_report`](Self::residual_report) is `Some`.
    pub fn point(&self) -> &[T] {
        self.point.as_slice()
    }
    /// Returns the residual at [`point`](Self::point): `F(x)`, or `G(x) − x` for
    /// [`FixedPointInput::Map`].
    pub fn residual(&self) -> &[T] {
        self.residual.as_slice()
    }
    /// Returns the norm used by the stopping test, `‖D⁻¹ f(x)‖₂` with the component
    /// divisors `D`, and its reference: the initial norm, or the scale set by
    /// [`FixedPointOptions::with_residual_scale`].
    pub fn residual_report(&self) -> Option<ResidualReport<T::Real>> {
        self.report
    }
    /// Returns `‖D⁻¹ f(x)‖₂`, the residual norm after the component divisors.
    pub fn scaled_residual_norm(&self) -> Option<T::Real> {
        self.report.map(|report| report.raw_norm())
    }
    /// Returns `‖f(x)‖₂` without divisors, or `None` before the first solve.
    pub fn residual_norm(&self) -> Option<T::Real> {
        self.report.map(|_| self.raw_norm)
    }
    /// Returns why the solve stopped. See [Termination](crate#termination).
    pub fn termination(&self) -> RootTermination {
        self.termination
    }
    /// Returns `true` if the residual passed the residual tolerance.
    pub fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the iterations and evaluations of this solve, without nested work.
    pub fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns this solve's work and the work of nested solves, counted separately
    /// (see [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn accounting(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of iterations.
    pub fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns the history diagnostics.
    pub fn diagnostics(&self) -> FixedPointDiagnostics {
        self.diagnostics
    }
}
