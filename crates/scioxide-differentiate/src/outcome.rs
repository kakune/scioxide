//! Derivative sources, termination statuses and results.

use crate::DifferenceScheme;
use alloc::vec::Vec;
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, EvaluationCounts, EvaluationRole,
    ExecutionStopReason, NumericScalar, NumericalIssue, SciOxideError,
};
/// Names the method that produced a [`DifferentiationReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DerivativeSource {
    /// Adaptive differences ([`AdaptiveDifference`](crate::AdaptiveDifference)).
    AdaptiveDifference,
    /// Sparse Jacobian from a column coloring
    /// ([`ColoredJacobianPlan`](crate::ColoredJacobianPlan)).
    ColoredDifference,
    /// Differences on nodes you choose
    /// ([`PhysicalDifferencePlan`](crate::PhysicalDifferencePlan),
    /// [`RelativeDifference`](crate::RelativeDifference)).
    NodeDifference,
    /// Complex step ([`ComplexStep`](crate::ComplexStep),
    /// [`ComplexStepPlan`](crate::ComplexStepPlan),
    /// [`RelativeComplexStep`](crate::RelativeComplexStep)).
    ComplexStep,
    /// Fixed-step differences with the given scheme
    /// ([`FiniteDifference`](crate::FiniteDifference)).
    FiniteDifference(DifferenceScheme),
}

/// Reports why a differentiation ended.
///
/// Only `Completed` and `Converged` are successes. Whether output was written
/// is reported separately by [`DifferentiationReport::output_available`]; see
/// [Results and termination](crate#results-and-termination).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiationTermination {
    /// A fixed stencil was evaluated. This makes no claim about accuracy.
    Completed,
    /// Every entry met the tolerance on two consecutive levels and its best
    /// estimate also met the tolerance, after at least three levels.
    Converged,
    /// Extrapolation stopped improving, or rounding error in the function
    /// values dominated the estimate.
    RoundoffLimit,
    /// The maximum number of extrapolation levels was reached.
    RefinementLimit,
    /// No smaller step gives distinct finite sample points inside the bounds.
    StepSizeLimit,
    /// The evaluation budget refused another callback call.
    EvaluationLimit,
    /// The iteration budget refused another derivative entry or level.
    IterationLimit,
    /// The cancellation flag was set.
    Cancelled,
    /// The deadline passed.
    DeadlineReached,
}
impl core::fmt::Display for DifferentiationTermination {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Completed => "Completed",
            Self::Converged => "Converged",
            Self::RoundoffLimit => "RoundoffLimit",
            Self::RefinementLimit => "RefinementLimit",
            Self::StepSizeLimit => "StepSizeLimit",
            Self::EvaluationLimit => "EvaluationLimit",
            Self::IterationLimit => "IterationLimit",
            Self::Cancelled => "Cancelled",
            Self::DeadlineReached => "DeadlineReached",
        })
    }
}
impl DifferentiationTermination {
    /// Returns `true` for `Completed` and `Converged`.
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Completed | Self::Converged)
    }
}
pub(super) const fn is_external_stop(reason: DifferentiationTermination) -> bool {
    matches!(
        reason,
        DifferentiationTermination::EvaluationLimit
            | DifferentiationTermination::IterationLimit
            | DifferentiationTermination::Cancelled
            | DifferentiationTermination::DeadlineReached
    )
}

/// Summarizes one evaluation: the method, why it ended, the work done and
/// whether the result was written.
///
/// Read [`output_available`](Self::output_available) before using the output.
/// An error is returned as `Err` instead of a report.
#[derive(Debug, Clone, Copy)]
pub struct DifferentiationReport {
    pub(super) termination: DifferentiationTermination,
    pub(super) source: DerivativeSource,
    pub(super) evaluations: usize,
    pub(super) iterations: usize,
    pub(super) completed_entries: usize,
    pub(super) output_len: usize,
}
impl DifferentiationReport {
    pub(super) fn fixed(source: DerivativeSource, output_len: usize) -> Self {
        Self {
            termination: DifferentiationTermination::Completed,
            source,
            evaluations: 0,
            iterations: 0,
            completed_entries: 0,
            output_len,
        }
    }
    /// Returns the method that produced the result.
    pub const fn source(&self) -> DerivativeSource {
        self.source
    }
    /// Returns why the evaluation ended.
    pub const fn termination(&self) -> DifferentiationTermination {
        self.termination
    }
    /// Returns `true` if a fixed stencil completed or an adaptive evaluation
    /// converged.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the number of callback calls this evaluation made, including a
    /// call that failed. Work of nested solves inside the callback is not
    /// included.
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }
    /// Returns the number of derivative entries or extrapolation levels
    /// attempted.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the number of derivative entries with a finite estimate.
    pub const fn completed_entries(&self) -> usize {
        self.completed_entries
    }
    /// Returns `true` if the result was written to the output.
    ///
    /// Fixed-step, node and complex-step plans write output when they end with
    /// `Completed`. Adaptive plans write output when every entry has an
    /// estimate and no budget, cancellation or deadline stopped the evaluation.
    pub const fn output_available(&self) -> bool {
        matches!(self.termination, DifferentiationTermination::Completed)
            || (self.completed_entries == self.output_len && !is_external_stop(self.termination))
    }
    /// Returns `Ok(())` for `Completed` and `Converged`, and the error that
    /// describes any other termination.
    ///
    /// | Termination | Error |
    /// |---|---|
    /// | `RoundoffLimit` | `ConvergenceFailure` (`Stagnation`) |
    /// | `RefinementLimit` | `ConvergenceFailure` (`TruncationLimit`) |
    /// | `StepSizeLimit` | `ConvergenceFailure` (`StepSizeLimit`) |
    /// | `EvaluationLimit`, `IterationLimit` | `ConvergenceFailure` (`EvaluationLimit`, `IterationLimit`) |
    /// | `Cancelled`, `DeadlineReached` | `ExecutionStopped` with the same reason |
    ///
    /// A `ConvergenceFailure` carries the number of function evaluations in
    /// its diagnostics. The call does not allocate on success.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` or `ExecutionStopped` as in the table.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ConvergenceReason, SciOxideErrorKind};
    /// use scioxide_differentiate::AdaptiveDifference;
    ///
    /// // Three levels cannot meet a tolerance of zero for f(x) = eˣ.
    /// let method = AdaptiveDifference::<f64>::default().with_max_levels(3);
    /// let report = *method.derivative(|x: f64| Ok(x.exp()), 0.0)?.report();
    /// assert!(!report.is_success());
    /// let error = report.require_success().unwrap_err();
    /// assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn require_success(&self) -> Result<(), SciOxideError> {
        let reason = match self.termination {
            DifferentiationTermination::Completed | DifferentiationTermination::Converged => {
                return Ok(());
            }
            DifferentiationTermination::Cancelled => {
                return Err(SciOxideError::execution_stopped(
                    ExecutionStopReason::Cancelled,
                ));
            }
            DifferentiationTermination::DeadlineReached => {
                return Err(SciOxideError::execution_stopped(
                    ExecutionStopReason::DeadlineReached,
                ));
            }
            DifferentiationTermination::RoundoffLimit => ConvergenceReason::Stagnation,
            DifferentiationTermination::RefinementLimit => ConvergenceReason::TruncationLimit,
            DifferentiationTermination::StepSizeLimit => ConvergenceReason::StepSizeLimit,
            DifferentiationTermination::EvaluationLimit => ConvergenceReason::EvaluationLimit,
            DifferentiationTermination::IterationLimit => ConvergenceReason::IterationLimit,
        };
        Err(SciOxideError::convergence_failure(
            reason,
            ConvergenceDiagnostics::new().with_counts(
                EvaluationCounts::new()
                    .with_provider_count(EvaluationRole::Function, self.evaluations)
                    .into(),
            ),
            "differentiation ended without success",
        ))
    }
}

/// Holds the result of an owned evaluation: the report, the derivative values
/// and, for adaptive plans, the error estimates.
///
/// Values are `Some` exactly when [`DifferentiationReport::output_available`]
/// is true. Error estimates are `Some` only for adaptive and colored plans with
/// output available.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::AdaptiveDifference;
///
/// // f(x) = x², so f′(1) = 2.
/// let result = AdaptiveDifference::default().derivative(|x: f64| Ok(x * x), 1.0)?;
/// assert!(result.report().is_success());
/// assert_eq!(result.report().output_available(), result.values().is_some());
/// assert!(result.errors().is_some());
/// let (report, values, _errors) = result.into_parts();
/// let values = values.unwrap();
/// assert_eq!(report.completed_entries(), values.len());
/// assert!((values[0] - 2.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct DifferentiationResult<T: NumericScalar> {
    pub(super) report: DifferentiationReport,
    pub(super) values: Option<Vec<T>>,
    pub(super) errors: Option<Vec<T::Real>>,
}
impl<T: NumericScalar> DifferentiationResult<T> {
    pub(super) fn from_evaluation(report: DifferentiationReport, values: Vec<T>) -> Self {
        Self {
            report,
            values: report.output_available().then_some(values),
            errors: None,
        }
    }
    pub(super) fn from_estimates(
        report: DifferentiationReport,
        values: Vec<T>,
        errors: Vec<T::Real>,
    ) -> Self {
        Self {
            report,
            values: report.output_available().then_some(values),
            errors: report.output_available().then_some(errors),
        }
    }
    /// Returns the report.
    pub const fn report(&self) -> &DifferentiationReport {
        &self.report
    }
    /// Returns the derivative values in row-major order, or `None` if no
    /// output is available.
    pub fn values(&self) -> Option<&[T]> {
        self.values.as_deref()
    }
    /// Returns the absolute error estimates, one per value, or `None` if the
    /// method does not estimate errors or no output is available.
    pub fn errors(&self) -> Option<&[T::Real]> {
        self.errors.as_deref()
    }
    /// Splits the result into the report, the values and the error estimates
    /// without copying.
    #[expect(
        clippy::type_complexity,
        reason = "the consuming result returns its report, values and optional errors directly"
    )]
    pub fn into_parts(self) -> (DifferentiationReport, Option<Vec<T>>, Option<Vec<T::Real>>) {
        (self.report, self.values, self.errors)
    }
    /// Returns the derivative values if the evaluation succeeded, and an error
    /// otherwise.
    ///
    /// To use the best estimates of an adaptive evaluation that stopped early,
    /// read them with [`values`](Self::values) or [`into_parts`](Self::into_parts)
    /// instead. The call does not allocate on success.
    ///
    /// # Errors
    ///
    /// - The errors of [`DifferentiationReport::require_success`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifference;
    ///
    /// // f(x) = x₀² + x₁, so ∇f(1, 2) = (2, 1).
    /// let f = |x: &[f64], y: &mut [f64]| {
    ///     y[0] = x[0] * x[0] + x[1];
    ///     Ok(())
    /// };
    /// let g = FiniteDifference::default().gradient(f, &[1.0, 2.0], None, None)?.into_values()?;
    /// assert!((g[0] - 2.0).abs() < 1e-8 && (g[1] - 1.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn into_values(self) -> Result<Vec<T>, SciOxideError> {
        self.report.require_success()?;
        self.values.ok_or_else(|| {
            SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "successful differentiation withheld its output",
            )
        })
    }
}
