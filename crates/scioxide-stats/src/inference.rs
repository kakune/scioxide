//! Termination status and convention enums shared by several statistical procedures.

use scioxide_core::ExecutionStopReason;

/// Termination status of an iterative fit or a replicate-based procedure.
///
/// Iterative fits (generalized linear models, Huber regression, orthogonal
/// distance regression, scatter estimation, optimal transport, sigma clipping and
/// injected fit solvers) end in `Converged`, `ExactFit` or one of the
/// unsuccessful variants. Monte Carlo and permutation procedures end in
/// `Completed`, `BudgetExhausted` or `Stopped`. Neither `Completed` nor
/// `Converged` is a guaranteed error bound. See
/// [Results and termination](crate#results-and-termination).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatsTermination {
    /// Every requested replicate or permutation was evaluated.
    Completed,
    /// The fit's convergence test passed. Each fit documents its own test.
    Converged,
    /// Every residual is zero up to rounding, so the data are interpolated
    /// exactly and the scale estimate is zero. Huber regression reports this.
    ExactFit,
    /// The iteration, evaluation or replicate limit was reached first, either the
    /// procedure's own limit or the one in the execution context.
    BudgetExhausted,
    /// Cancellation or a deadline in the
    /// [execution context](scioxide_core#execution-contexts) stopped the procedure.
    Stopped(ExecutionStopReason),
    /// The solver stopped without meeting its convergence test for a reason
    /// other than a limit or a stop request.
    NotConverged,
    /// The weighted design matrix of an iteratively reweighted least squares
    /// step has numerical rank below the number of coefficients.
    RankDeficient,
    /// A binomial fit diverges toward complete or quasi-complete separation.
    /// Either the coefficients grow large while fitted probabilities approach 0
    /// or 1, or step halving finds no step that reduces the deviance.
    Separated,
    /// Step halving finds no step that reduces the objective, or the inner
    /// linear solve fails to produce a step.
    NoProgress,
}

impl StatsTermination {
    /// Returns the error of a one-shot form, or `None` after success.
    pub(crate) fn one_shot_error(
        self,
        exhausted: scioxide_core::ConvergenceReason,
        diagnostics: scioxide_core::ConvergenceDiagnostics,
    ) -> Option<scioxide_core::SciOxideError> {
        use scioxide_core::{ConvergenceReason, SciOxideError};
        let reason = match self {
            Self::Completed | Self::Converged | Self::ExactFit => return None,
            Self::Stopped(reason) => return Some(SciOxideError::execution_stopped(reason)),
            Self::RankDeficient => return Some(SciOxideError::singular(None)),
            Self::BudgetExhausted => exhausted,
            Self::NotConverged => ConvergenceReason::NestedSolver,
            Self::NoProgress => ConvergenceReason::Stagnation,
            Self::Separated => ConvergenceReason::Divergence,
        };
        Some(SciOxideError::convergence_failure(
            reason,
            diagnostics,
            "statistical computation did not succeed",
        ))
    }
    /// Returns `true` for `Completed`, `Converged` and `ExactFit`.
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Completed | Self::Converged | Self::ExactFit)
    }

    pub(crate) fn validate_fit(self) -> Result<Self, scioxide_core::SciOxideError> {
        match self {
            Self::Converged | Self::BudgetExhausted | Self::Stopped(_) | Self::NotConverged => {
                Ok(self)
            }
            _ => Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("termination"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "injected fit solver returned an inapplicable termination",
            )),
        }
    }
}

/// Selects whether a normal approximation to a discrete statistic uses a continuity correction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContinuityCorrection {
    /// Apply no correction.
    None,
    /// Reduce the distance between the statistic and its null mean by half a
    /// unit before standardizing, as in Yates's correction.
    #[default]
    HalfUnit,
}

/// Selects whether a kernel density estimate returns the density or its logarithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdeScale {
    /// The density `f̂(x)`.
    Density,
    /// The natural logarithm `ln f̂(x)`.
    LogDensity,
}

/// Selects the normalization of L-moments of order three and above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LMomentNormalization {
    /// Return the L-moment in observation units.
    Raw,
    /// Return the L-moment ratio `τᵣ = λᵣ / λ₂` for orders `r ≥ 3`.
    Standardized,
}

/// Selects whether a finite trimming bound is closed or open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimInclusion {
    /// Include observations equal to the bound.
    Inclusive,
    /// Exclude observations equal to the bound.
    Exclusive,
}

/// Records whether a survival observation ended in an event or in right censoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalOutcome {
    /// The event was observed at the exit time.
    Event,
    /// The subject was right censored at the exit time.
    Censored,
}

/// Selects the variance estimate in the Wald statistic that orders tables in Barnard's test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariancePooling {
    /// Estimate one pooled binomial probability.
    Pooled,
    /// Estimate the two binomial probabilities separately.
    Unpooled,
}

/// Selects the mean subtracted before forming lag products of a time series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeSeriesCentering {
    /// Subtract the mean of the present observations.
    SampleMean,
    /// Assume a known mean of zero and subtract nothing.
    KnownZero,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{
        BudgetKind, ConvergenceDiagnostics, ConvergenceReason, SciOxideErrorKind, SciOxideErrorView,
    };
    #[test]
    fn one_shot_termination_mapping_is_complete() {
        let diagnostics = ConvergenceDiagnostics::new();
        for status in [
            StatsTermination::Completed,
            StatsTermination::Converged,
            StatsTermination::ExactFit,
        ] {
            assert!(
                status
                    .one_shot_error(ConvergenceReason::IterationLimit, diagnostics.clone())
                    .is_none()
            );
        }
        for (status, reason) in [
            (
                StatsTermination::BudgetExhausted,
                ConvergenceReason::IterationLimit,
            ),
            (
                StatsTermination::NotConverged,
                ConvergenceReason::NestedSolver,
            ),
            (StatsTermination::NoProgress, ConvergenceReason::Stagnation),
            (StatsTermination::Separated, ConvergenceReason::Divergence),
        ] {
            let error =
                status.one_shot_error(ConvergenceReason::IterationLimit, diagnostics.clone());
            assert!(
                matches!(error.as_ref().map(|error|error.view()),Some(SciOxideErrorView::ConvergenceFailure {reason:actual,..}) if actual==reason)
            );
        }
        for reason in [
            ConvergenceReason::NestedSolver,
            ConvergenceReason::ResourceLimit(BudgetKind::Sample),
        ] {
            let error =
                StatsTermination::BudgetExhausted.one_shot_error(reason, diagnostics.clone());
            assert!(
                matches!(error.as_ref().map(|error|error.view()),Some(SciOxideErrorView::ConvergenceFailure {reason:actual,..}) if actual==reason)
            );
        }
        assert_eq!(
            StatsTermination::RankDeficient
                .one_shot_error(ConvergenceReason::IterationLimit, diagnostics.clone())
                .map(|e| e.kind()),
            Some(SciOxideErrorKind::Singular)
        );
        assert!(matches!(
            StatsTermination::Stopped(ExecutionStopReason::Cancelled)
                .one_shot_error(ConvergenceReason::IterationLimit, diagnostics)
                .as_ref()
                .map(|e| e.view()),
            Some(SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::Cancelled
            })
        ));
    }
}
