//! Reasons why a proof or a refinement stops without a result.
use scioxide_core::ExecutionStopReason;

/// Reason why a proof neither succeeded nor showed the proposal to be wrong.
///
/// The first four reasons are shared by every verifier; `R` holds the reasons
/// specific to one kind of proof, such as [`LinearProofFailure`](crate::LinearProofFailure).
/// An inconclusive proof is an `Ok` outcome in configured forms and an error
/// in one-shot forms. Malformed input and invalid
/// enclosures from your callbacks are errors instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofInconclusive<R> {
    /// A bound or enclosure that the proof needs is not finite.
    Unbounded,
    /// The evaluation budget does not allow another callback evaluation.
    EvaluationLimit,
    /// The iteration budget does not allow another iteration.
    IterationLimit,
    /// Cancellation or a deadline stopped the proof.
    Stopped(ExecutionStopReason),
    /// A test specific to this kind of proof failed.
    Reason(R),
}

/// Reason why a root isolation or quadrature generation cannot subdivide
/// further.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementLimit {
    /// The output is full.
    Capacity,
    /// No floating-point number lies strictly inside the interval to split, or a
    /// disk radius cannot grow further.
    Resolution,
}

use scioxide_core::{
    AlgorithmId, AttemptRecord, BudgetKind, ConvergenceDiagnostics, ConvergenceReason,
    DomainReason, NumericalIssue, SciOxideError,
};

pub(crate) fn diagnostics(
    algorithm: AlgorithmId,
    attempt: Option<&AttemptRecord<()>>,
) -> ConvergenceDiagnostics {
    let diagnostics = ConvergenceDiagnostics::new().with_algorithm(algorithm);
    match attempt {
        Some(a) => diagnostics.with_counts(a.counts()),
        None => diagnostics,
    }
}

pub(crate) fn inconclusive_error<R>(
    reason: ProofInconclusive<R>,
    diagnostics: ConvergenceDiagnostics,
    family: impl FnOnce(R, ConvergenceDiagnostics) -> SciOxideError,
) -> SciOxideError {
    match reason {
        ProofInconclusive::Unbounded => SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "an enclosure that the proof needs is unbounded",
        ),
        ProofInconclusive::EvaluationLimit => SciOxideError::convergence_failure(
            ConvergenceReason::EvaluationLimit,
            diagnostics,
            "the evaluation budget ended the proof",
        ),
        ProofInconclusive::IterationLimit => SciOxideError::convergence_failure(
            ConvergenceReason::IterationLimit,
            diagnostics,
            "the iteration budget ended the proof",
        ),
        ProofInconclusive::Stopped(reason) => SciOxideError::execution_stopped(reason),
        ProofInconclusive::Reason(reason) => family(reason, diagnostics),
    }
}

pub(crate) fn limit_error(
    limit: RefinementLimit,
    resource: BudgetKind,
    diagnostics: ConvergenceDiagnostics,
) -> SciOxideError {
    let (reason, detail) = match limit {
        RefinementLimit::Capacity => (
            ConvergenceReason::ResourceLimit(resource),
            "refinement filled the output capacity",
        ),
        RefinementLimit::Resolution => (
            ConvergenceReason::StepSizeLimit,
            "no representable split point or radius remains",
        ),
    };
    SciOxideError::convergence_failure(reason, diagnostics, detail)
}

pub(crate) fn linear_result(
    status: crate::LinearVerificationStatus,
    diagnostics: ConvergenceDiagnostics,
) -> Result<(), SciOxideError> {
    use crate::{LinearProofFailure as F, LinearVerificationStatus as S};
    match status {
        S::Verified => Ok(()),
        S::Inconclusive(reason) => Err(inconclusive_error(reason, diagnostics, |reason, _| {
            let (issue, detail) = match reason {
                F::RegularityNotProved => (
                    NumericalIssue::IllConditioned,
                    "nonsingularity of the matrix is not proved",
                ),
                F::SeedFailure => (
                    NumericalIssue::Overflow,
                    "the approximate solution or its residual is not finite",
                ),
            };
            SciOxideError::numerical_instability(issue, detail)
        })),
        S::NotRun | S::Rejected(_) => Err(crate::shell::breakdown(
            "generated linear certificate was not verified",
        )),
    }
}

pub(crate) fn root_result(
    status: crate::RootIsolationTermination,
    diagnostics: ConvergenceDiagnostics,
) -> Result<(), SciOxideError> {
    use crate::{RootIsolationTermination as S, RootProofFailure as F};
    match status {
        S::Complete => Ok(()),
        S::Limit(limit) => Err(limit_error(limit, BudgetKind::Node, diagnostics)),
        S::Inconclusive(reason) => Err(inconclusive_error(
            reason,
            diagnostics,
            |reason, diagnostics| match reason {
                F::Inequality => SciOxideError::convergence_failure(
                    ConvergenceReason::Stagnation,
                    diagnostics,
                    "the root tests decide neither a claim nor its opposite",
                ),
                F::Domain => SciOxideError::domain_error(
                    scioxide_core::argument!("f"),
                    DomainReason::Undefined,
                    "an enclosure of f or df is empty or not known to be continuous",
                ),
            },
        )),
        S::Refuted => Err(crate::shell::breakdown(
            "generated root isolation was refuted",
        )),
    }
}

fn quadrature_error(
    reason: crate::QuadratureInconclusive,
    diagnostics: ConvergenceDiagnostics,
) -> SciOxideError {
    inconclusive_error(
        reason,
        diagnostics,
        |crate::QuadratureProofFailure::MissingObligation, _| {
            crate::shell::breakdown("generated quadrature certificate is missing an obligation")
        },
    )
}

pub(crate) fn quadrature_generation_result(
    status: crate::QuadratureTermination,
    diagnostics: ConvergenceDiagnostics,
) -> Result<(), SciOxideError> {
    use crate::QuadratureTermination as S;
    match status {
        S::ToleranceReached => Ok(()),
        S::Limit(limit) => Err(limit_error(limit, BudgetKind::Subdivision, diagnostics)),
        S::Inconclusive(reason) => Err(quadrature_error(reason, diagnostics)),
        S::FixedBounds => Err(SciOxideError::convergence_failure(
            ConvergenceReason::Stagnation,
            diagnostics,
            "the bounds of the remaining panels cannot be reduced",
        )),
    }
}

pub(crate) fn quadrature_verification_result(
    status: crate::QuadratureVerificationStatus,
    diagnostics: ConvergenceDiagnostics,
) -> Result<(), SciOxideError> {
    use crate::QuadratureVerificationStatus as S;
    match status {
        S::Verified => Ok(()),
        S::Inconclusive(reason) => Err(quadrature_error(reason, diagnostics)),
        S::NotRun | S::Rejected(_) => Err(crate::shell::breakdown(
            "generated quadrature certificate was not verified",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use scioxide_core::{ExecutionStopReason, SciOxideErrorView as V};

    fn diagnostics() -> ConvergenceDiagnostics {
        ConvergenceDiagnostics::new().with_algorithm(scioxide_core::algorithm_id!(
            "validated",
            "test",
            1
        ))
    }
    fn numerical(error: SciOxideError, expected: NumericalIssue) {
        assert!(matches!(error.view(), V::NumericalInstability { issue, .. } if issue == expected));
    }
    fn convergence(error: SciOxideError, expected: ConvergenceReason) {
        assert!(matches!(error.view(), V::ConvergenceFailure { reason, .. } if reason == expected));
    }
    #[test]
    fn common_outcomes_preserve_kind_reason_and_argument() {
        let error = |reason| inconclusive_error(reason, diagnostics(), |(), _| unreachable!());
        numerical(
            error(ProofInconclusive::Unbounded),
            NumericalIssue::Overflow,
        );
        convergence(
            error(ProofInconclusive::EvaluationLimit),
            ConvergenceReason::EvaluationLimit,
        );
        convergence(
            error(ProofInconclusive::IterationLimit),
            ConvergenceReason::IterationLimit,
        );
        for reason in [
            ExecutionStopReason::Cancelled,
            ExecutionStopReason::DeadlineReached,
        ] {
            assert!(
                matches!(error(ProofInconclusive::Stopped(reason)).view(), V::ExecutionStopped { reason: found } if found == reason)
            );
        }
        for resource in [BudgetKind::Node, BudgetKind::Subdivision] {
            convergence(
                limit_error(RefinementLimit::Capacity, resource, diagnostics()),
                ConvergenceReason::ResourceLimit(resource),
            );
            convergence(
                limit_error(RefinementLimit::Resolution, resource, diagnostics()),
                ConvergenceReason::StepSizeLimit,
            );
        }
    }
    #[test]
    fn family_outcomes_never_turn_inconclusive_into_success() {
        assert!(linear_result(LinearVerificationStatus::Verified, diagnostics()).is_ok());
        for (reason, issue) in [
            (
                LinearProofFailure::RegularityNotProved,
                NumericalIssue::IllConditioned,
            ),
            (LinearProofFailure::SeedFailure, NumericalIssue::Overflow),
        ] {
            numerical(
                linear_result(
                    LinearVerificationStatus::Inconclusive(ProofInconclusive::Reason(reason)),
                    diagnostics(),
                )
                .unwrap_err(),
                issue,
            );
        }
        for status in [
            LinearVerificationStatus::NotRun,
            LinearVerificationStatus::Rejected(LinearRejection::EnclosureTooNarrow),
        ] {
            numerical(
                linear_result(status, diagnostics()).unwrap_err(),
                NumericalIssue::Breakdown,
            );
        }
        assert!(root_result(RootIsolationTermination::Complete, diagnostics()).is_ok());
        convergence(
            root_result(
                RootIsolationTermination::Inconclusive(ProofInconclusive::Reason(
                    RootProofFailure::Inequality,
                )),
                diagnostics(),
            )
            .unwrap_err(),
            ConvergenceReason::Stagnation,
        );
        let domain = root_result(
            RootIsolationTermination::Inconclusive(ProofInconclusive::Reason(
                RootProofFailure::Domain,
            )),
            diagnostics(),
        )
        .unwrap_err();
        assert!(
            matches!(domain.view(), V::DomainError { argument, reason: DomainReason::Undefined, .. } if argument == scioxide_core::argument!("f"))
        );
        numerical(
            root_result(RootIsolationTermination::Refuted, diagnostics()).unwrap_err(),
            NumericalIssue::Breakdown,
        );
        for limit in [RefinementLimit::Capacity, RefinementLimit::Resolution] {
            assert!(root_result(RootIsolationTermination::Limit(limit), diagnostics()).is_err());
            assert!(
                quadrature_generation_result(QuadratureTermination::Limit(limit), diagnostics())
                    .is_err()
            );
        }
        assert!(
            quadrature_generation_result(QuadratureTermination::ToleranceReached, diagnostics())
                .is_ok()
        );
        convergence(
            quadrature_generation_result(QuadratureTermination::FixedBounds, diagnostics())
                .unwrap_err(),
            ConvergenceReason::Stagnation,
        );
        let missing = ProofInconclusive::Reason(QuadratureProofFailure::MissingObligation);
        numerical(
            quadrature_generation_result(
                QuadratureTermination::Inconclusive(missing),
                diagnostics(),
            )
            .unwrap_err(),
            NumericalIssue::Breakdown,
        );
        assert!(
            quadrature_verification_result(QuadratureVerificationStatus::Verified, diagnostics())
                .is_ok()
        );
        for status in [
            QuadratureVerificationStatus::NotRun,
            QuadratureVerificationStatus::Rejected(QuadratureRejection::LocalBound),
            QuadratureVerificationStatus::Rejected(QuadratureRejection::TotalBound),
            QuadratureVerificationStatus::Inconclusive(missing),
        ] {
            numerical(
                quadrature_verification_result(status, diagnostics()).unwrap_err(),
                NumericalIssue::Breakdown,
            );
        }
    }
}
