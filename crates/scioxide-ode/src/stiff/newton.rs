//! The single Newton iteration of stiff stages and residual DAEs.
//!
//! Convergence is decided from the scaled correction norm `‖Δₖ‖`, where one is the
//! stopping tolerance κ·tol, and the contraction rate `θₖ = ‖Δₖ‖/‖Δₖ₋₁‖`
//! (Hairer–Wanner, *Solving ODEs II*, IV.8):
//!
//! - an affine stage with an exact native direct factor can converge after one correction;
//! - an exactly zero residual or correction has converged;
//! - a first correction has no rate. It converges iff `‖Δ‖ ≤ 1` and the stage certifies
//!   its linearization: ODE stages need a current linearization and a rate `θ̂ < 0.9`
//!   carried over from an earlier solve of the epoch; DAE
//!   stages, whose iterates must also satisfy the residual blocks, need a current
//!   linearization. Otherwise a second correction measures `θ`; a single-correction
//!   budget accepts `‖Δ‖ ≤ 1`;
//! - `θ ≥ 1` diverges, and a frozen linearization contracting at `θ ≥ 0.9` stagnates: its
//!   estimate `η = θ/(1−θ) ≥ 9` amplifies an error of the measured rate by `1/(1−θ)`.
//!   Either converges only when `‖Δ‖` is within the stage's floor. ODE stages use the
//!   [`rounding_level`] of the iterate, `16·ε` of each component scale, where the rate is
//!   rounding noise; DAE stages use the stopping tolerance, because their residual blocks
//!   then certify the iterate;
//! - with `η = θ/(1−θ)` the iteration converges iff `η·‖Δ‖ ≤ 1`, an estimate of the
//!   remaining error after applying `Δ`;
//! - when the limit is exhausted the failure is an iteration limit; with a frozen
//!   linearization, whose convergence is linear at rate `θ`, a predicted
//!   `θʳ·η·‖Δ‖ > 1` for the `r` remaining corrections is stagnation. Full Newton
//!   converges faster than that prediction, so it only stops at the limit.
//!
//! A converged iteration always applies its final correction, after which a stage may
//! require a further acceptance test (the DAE residual blocks) before the iteration
//! ends. Non-convergence is an outcome, so budget, provider and linear-solve errors
//! stay distinguishable.
use scioxide_core::{ConvergenceReason, RealScalar, SciOxideError};

/// The rounding level `16·ε·scale` of one iterate component: a residual evaluated at
/// the iterate carries rounding errors of the same order, so smaller corrections cannot
/// be resolved.
#[inline]
pub(crate) fn rounding_level<S: RealScalar>(scale: S) -> S {
    let four = S::TWO * S::TWO;
    four * four * S::EPSILON * scale
}

/// Contraction rates at or above this bound do not certify convergence.
fn rate_limit<S: RealScalar>(frozen: bool) -> Result<S, SciOxideError> {
    if frozen {
        S::try_from_f64_lossy(0.9)
    } else {
        Ok(S::ONE)
    }
}

/// Whether a contraction rate `θ̂` carried over from an earlier solve stays below the
/// frozen rate limit, so that it certifies a first correction.
pub(crate) fn contracting<S: RealScalar>(rate: Option<S>) -> bool {
    rate_limit::<S>(true).is_ok_and(|limit| rate.is_some_and(|rate| rate < limit))
}

/// One nonlinear equation `G(x) = 0` with a prepared correction solve.
pub(crate) trait NewtonStage<S: RealScalar> {
    /// The stage's failure type; families that classify denied admissions as values
    /// use [`crate::termination::Fault`].
    type Error: From<SciOxideError>;
    /// Count one iteration and evaluate `G` at the iterate; `true` means exactly zero.
    fn residual(&mut self) -> Result<bool, Self::Error>;
    /// Count and solve the correction for the last residual, relinearizing if required.
    fn correction(&mut self) -> Result<(), Self::Error>;
    /// Weighted correction norm; one equals the stopping tolerance.
    fn norm(&mut self) -> Result<S, Self::Error>;
    /// The norm, in the units of [`norm`](Self::norm), at or below which a correction
    /// converges without a contraction rate. It is called after `norm` and before
    /// `apply`, only when the rate cannot certify convergence.
    fn floor(&mut self) -> Result<S, Self::Error>;
    /// Subtract the correction from the iterate.
    fn apply(&mut self) -> Result<(), Self::Error>;
    /// Whether corrections reuse one frozen linearization (linear convergence).
    fn frozen(&self) -> bool;
    /// Frozen approximate corrections need two norms to estimate the remaining error.
    fn requires_rate(&self) -> bool {
        false
    }
    /// Whether a first correction within tolerance converges without a measured rate.
    fn certifies_first(&self) -> bool {
        !self.requires_rate()
    }
    /// For an affine equation with an exact direct factor, apply the correction
    /// and finalize stored derivatives. `None` keeps ordinary Newton logic;
    /// `Some(false)` has applied the correction and needs another iteration.
    fn affine_correction(&mut self) -> Result<Option<bool>, Self::Error> {
        Ok(None)
    }
    /// Accept a converged iterate; `false` continues iterating within the limit.
    fn accept(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

/// Decision after one correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Progress {
    Converged,
    Continue,
    Failed(ConvergenceReason),
}

/// Classify one correction from the previous norm, its own norm and the corrections left.
/// `frozen` selects the frozen rate limit and the stagnation prediction of linearly
/// converging iterations, `first` whether the stage certifies a first correction and
/// `floor` evaluates the stage floor when a rate fails.
pub(crate) fn assess<S: RealScalar, E: From<SciOxideError>>(
    previous: Option<S>,
    norm: S,
    remaining: usize,
    frozen: bool,
    first: bool,
    floor: impl FnOnce() -> Result<S, E>,
) -> Result<Progress, E> {
    if norm == S::ZERO {
        return Ok(Progress::Converged);
    }
    let Some(previous) = previous else {
        return Ok(if norm <= S::ONE && (first || remaining == 0) {
            Progress::Converged
        } else if remaining == 0 {
            Progress::Failed(ConvergenceReason::IterationLimit)
        } else {
            Progress::Continue
        });
    };
    let rate = norm / previous;
    if rate.is_nan() || rate >= rate_limit(frozen)? {
        return Ok(if norm <= floor()? {
            Progress::Converged
        } else if rate.is_nan() || rate >= S::ONE {
            Progress::Failed(ConvergenceReason::Divergence)
        } else {
            Progress::Failed(ConvergenceReason::Stagnation)
        });
    }
    let remaining_error = rate / (S::ONE - rate) * norm;
    Ok(if remaining_error <= S::ONE {
        Progress::Converged
    } else if remaining == 0 {
        Progress::Failed(ConvergenceReason::IterationLimit)
    } else if frozen && (0..remaining).fold(remaining_error, |error, _| error * rate) > S::ONE {
        Progress::Failed(ConvergenceReason::Stagnation)
    } else {
        Progress::Continue
    })
}

/// Exact work of one Newton solve.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct NewtonReport<S> {
    pub(crate) iterations: usize,
    pub(crate) corrections: usize,
    pub(crate) rate: Option<S>,
}

/// Converged, or failed for a numerical reason with the work spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NewtonOutcome<S> {
    Converged(NewtonReport<S>),
    Failed(ConvergenceReason, NewtonReport<S>),
}

impl<S: RealScalar> NewtonOutcome<S> {
    /// Work spent, whatever the outcome.
    pub(crate) const fn report(self) -> NewtonReport<S> {
        match self {
            Self::Converged(report) | Self::Failed(_, report) => report,
        }
    }

    /// Convert a numerical failure into the recoverable stage error.
    pub(crate) fn converged(self) -> Result<NewtonReport<S>, SciOxideError> {
        match self {
            Self::Converged(report) => Ok(report),
            Self::Failed(reason, report) => Err(SciOxideError::convergence_failure(
                reason,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(
                    scioxide_core::EvaluationCounts::from_iterations(report.iterations).into(),
                ),
                "Newton stage did not converge",
            )),
        }
    }
}

/// Iterate at most `limit` times.
pub(crate) fn iterate<S: RealScalar, N: NewtonStage<S>>(
    stage: &mut N,
    limit: usize,
) -> Result<NewtonOutcome<S>, N::Error> {
    let mut previous = None;
    let mut report = NewtonReport {
        iterations: 0,
        corrections: 0,
        rate: None,
    };
    for iteration in 1..=limit {
        report.iterations = iteration;
        if stage.residual()? {
            return Ok(NewtonOutcome::Converged(report));
        }
        stage.correction()?;
        report.corrections = iteration;
        if let Some(accepted) = stage.affine_correction()? {
            if accepted {
                return Ok(NewtonOutcome::Converged(report));
            }
            previous = None;
            continue;
        }
        let norm = stage.norm()?;
        report.rate = previous.filter(|old| *old > S::ZERO).map(|old| norm / old);
        let frozen = stage.frozen();
        let first = stage.certifies_first();
        match assess(previous, norm, limit - iteration, frozen, first, || {
            stage.floor()
        })? {
            Progress::Converged => {
                stage.apply()?;
                if stage.accept()? {
                    return Ok(NewtonOutcome::Converged(report));
                }
                previous = Some(norm);
            }
            Progress::Continue => {
                stage.apply()?;
                previous = Some(norm);
            }
            Progress::Failed(reason) => return Ok(NewtonOutcome::Failed(reason, report)),
        }
    }
    Ok(NewtonOutcome::Failed(
        ConvergenceReason::IterationLimit,
        report,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Assess with a floor that never certifies a nonzero correction.
    fn plain<S: RealScalar>(
        previous: Option<S>,
        norm: S,
        remaining: usize,
        frozen: bool,
    ) -> Progress {
        assess(previous, norm, remaining, frozen, false, || {
            Ok::<_, SciOxideError>(S::ZERO)
        })
        .unwrap()
    }

    #[test]
    fn first_corrections_need_certification_or_a_single_correction_budget() {
        let first = |norm: f64, remaining, certified| {
            assess(None, norm, remaining, true, certified, || {
                Ok::<_, SciOxideError>(0.0)
            })
            .unwrap()
        };
        assert_eq!(first(0.5, 3, false), Progress::Continue);
        assert_eq!(first(0.5, 3, true), Progress::Converged);
        assert_eq!(first(2.0, 3, true), Progress::Continue);
        assert_eq!(first(1.0, 0, false), Progress::Converged);
        assert_eq!(first(0.0, 3, false), Progress::Converged);
        assert_eq!(
            first(2.0, 0, true),
            Progress::Failed(ConvergenceReason::IterationLimit)
        );
        assert!(!contracting::<f64>(None));
        assert!(contracting(Some(0.0_f64)));
        assert!(contracting(Some(0.89_f64)));
        assert!(!contracting(Some(0.9_f64)));
        assert!(!contracting(Some(f64::NAN)));
    }

    #[test]
    fn contraction_rate_decides_convergence() {
        // θ = 0.1 gives η·‖Δ‖ ≈ 0.56 ≤ 1 although ‖Δ‖ > 1.
        assert_eq!(plain(Some(50.0_f64), 5.0, 5, true), Progress::Converged);
        // θ = 0.8 makes η = 4: a correction within tolerance still continues.
        assert_eq!(plain(Some(1.0_f64), 0.8, 10, true), Progress::Continue);
        assert_eq!(plain(Some(1.0_f32), 0.1, 10, true), Progress::Converged);
    }

    #[test]
    fn growth_diverges_unless_within_the_floor() {
        assert_eq!(
            plain(Some(2.0_f64), 3.0, 5, true),
            Progress::Failed(ConvergenceReason::Divergence)
        );
        assert_eq!(
            plain(Some(2.0_f64), 2.0, 5, false),
            Progress::Failed(ConvergenceReason::Divergence)
        );
        // A tiny correction is not converged by its size alone.
        assert_eq!(
            plain(Some(0.002_f64), 0.002_01, 5, true),
            Progress::Failed(ConvergenceReason::Divergence)
        );
        let rounded = |floor: f64| {
            assess(Some(1e-9_f64), 1.1e-9, 5, true, false, || {
                Ok::<_, SciOxideError>(floor)
            })
            .unwrap()
        };
        assert_eq!(rounded(1.1e-9), Progress::Converged);
        assert_eq!(
            rounded(1e-9),
            Progress::Failed(ConvergenceReason::Divergence)
        );
    }

    #[test]
    fn slow_frozen_contraction_stagnates_or_exhausts_the_limit() {
        // A frozen θ = 0.95 stagnates even though η·‖Δ‖ = 0.019 would claim convergence.
        assert_eq!(
            plain(Some(0.001_f64), 0.000_95, 2, true),
            Progress::Failed(ConvergenceReason::Stagnation)
        );
        assert_eq!(
            plain(Some(100.0_f64), 95.0, 0, true),
            Progress::Failed(ConvergenceReason::Stagnation)
        );
        // θ = 0.8: 0.8²·4·40 > 1 cannot converge within two more corrections.
        assert_eq!(
            plain(Some(50.0_f64), 40.0, 2, true),
            Progress::Failed(ConvergenceReason::Stagnation)
        );
        assert_eq!(
            plain(Some(50.0_f64), 40.0, 0, true),
            Progress::Failed(ConvergenceReason::IterationLimit)
        );
        assert_eq!(plain(Some(100.0_f64), 50.0, 20, true), Progress::Continue);
        // Full Newton is neither limited at θ = 0.95 nor extrapolated linearly.
        assert_eq!(plain(Some(100.0_f64), 95.0, 2, false), Progress::Continue);
        assert_eq!(
            plain(Some(0.001_f64), 0.000_95, 2, false),
            Progress::Converged
        );
    }

    /// Scalar `x² = 2` with an exact or frozen derivative.
    struct Root {
        x: f64,
        g: f64,
        delta: f64,
        slope: Option<f64>,
        applied: usize,
    }
    impl NewtonStage<f64> for Root {
        type Error = SciOxideError;
        fn frozen(&self) -> bool {
            self.slope.is_some()
        }
        fn residual(&mut self) -> Result<bool, SciOxideError> {
            self.g = self.x * self.x - 2.0;
            Ok(self.g == 0.0)
        }
        fn correction(&mut self) -> Result<(), SciOxideError> {
            self.delta = self.g / self.slope.unwrap_or(2.0 * self.x);
            Ok(())
        }
        fn norm(&mut self) -> Result<f64, SciOxideError> {
            Ok(self.delta.abs() / 1e-10)
        }
        fn floor(&mut self) -> Result<f64, SciOxideError> {
            Ok(rounding_level((self.x - self.delta).abs()) / 1e-10)
        }
        fn apply(&mut self) -> Result<(), SciOxideError> {
            self.x -= self.delta;
            self.applied += 1;
            Ok(())
        }
    }

    #[test]
    fn iteration_applies_the_final_correction_and_classifies_failures() {
        let mut exact = Root {
            x: 1.0,
            g: 0.0,
            delta: 0.0,
            slope: None,
            applied: 0,
        };
        let outcome = iterate(&mut exact, 20).unwrap();
        assert!(matches!(outcome, NewtonOutcome::Converged(_)));
        assert_eq!(exact.applied, outcome.report().corrections);
        assert!((exact.x - 2_f64.sqrt()).abs() < 1e-14);

        // A derivative of the wrong sign moves away from the root.
        let mut wrong = Root {
            x: 1.5,
            slope: Some(-1.0),
            ..exact
        };
        assert!(matches!(
            iterate(&mut wrong, 20).unwrap(),
            NewtonOutcome::Failed(ConvergenceReason::Divergence, _)
        ));

        // A frozen slope far too large contracts with θ ≈ 0.99.
        let mut slow = Root {
            x: 1.0,
            slope: Some(300.0),
            ..exact
        };
        assert!(matches!(
            iterate(&mut slow, 20).unwrap(),
            NewtonOutcome::Failed(ConvergenceReason::Stagnation, _)
        ));

        let mut single = Root { x: 1.0, ..exact };
        let outcome = iterate(&mut single, 1).unwrap();
        assert_eq!(
            outcome,
            NewtonOutcome::Failed(
                ConvergenceReason::IterationLimit,
                NewtonReport {
                    iterations: 1,
                    corrections: 1,
                    rate: None
                }
            )
        );
        assert!(outcome.converged().is_err());
    }
}
