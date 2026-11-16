//! External inspection, allocation, and source contracts for shared errors.
#![allow(clippy::unwrap_used)]
use core::error::Error;
use core::fmt::Write;

use scioxide_core::*;

struct Sink;

#[test]
fn replay_mismatch_is_a_typed_convergence_failure() {
    let error = SciOxideError::convergence_failure(
        ConvergenceReason::ReplayMismatch,
        ConvergenceDiagnostics::new(),
        "retained trajectory changed",
    );
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::ReplayMismatch,
            ..
        }
    ));
    assert_eq!(
        ConvergenceReason::ReplayMismatch.to_string(),
        "trajectory replay mismatch"
    );
}

#[test]
fn truncation_failure_preserves_its_reason_and_residual() {
    let residual = ResidualReport::unscaled(0.25_f64, NormSelection::Infinity).unwrap();
    let counts = EvaluationCounts::from_iterations(32);
    let error = SciOxideError::convergence_failure(
        ConvergenceReason::TruncationLimit,
        ConvergenceDiagnostics::new()
            .with_counts(counts.into())
            .with_residual(residual),
        "expansion maximum reached",
    );
    let SciOxideErrorView::ConvergenceFailure {
        reason,
        diagnostics,
        detail,
    } = error.view()
    else {
        unreachable!()
    };
    assert_eq!(reason, ConvergenceReason::TruncationLimit);
    assert_eq!(reason.to_string(), "truncation limit reached");
    assert_eq!(diagnostics.counts().own(), counts);
    assert_eq!(diagnostics.residual(), Some(residual));
    assert_eq!(detail, "expansion maximum reached");
    assert_eq!(error, error.clone());
}
impl core::fmt::Write for Sink {
    fn write_str(&mut self, _: &str) -> core::fmt::Result {
        Ok(())
    }
}

#[test]
fn static_errors_construct_clone_and_format_without_allocation() {
    {
        let errors = [
            SciOxideError::invalid_input(argument!("x"), InvalidInputReason::OutOfRange, "invalid"),
            SciOxideError::domain_error(argument!("x"), DomainReason::OutsideDomain, "negative"),
            SciOxideError::non_finite(NonFiniteOrigin::Input, "nonfinite"),
            SciOxideError::size_overflow(quantity!("shape")),
            SciOxideError::allocation_failure(quantity!("scratch"), 7),
            SciOxideError::dimension_mismatch([2, 3], [4, 5]),
            SciOxideError::singular(Some(2)),
            SciOxideError::scalar_conversion(
                ConversionType::F64,
                ConversionType::Usize,
                ScalarConversionReason::Inexact,
            ),
            SciOxideError::not_implemented("gradient unavailable"),
            SciOxideError::accounting_invariant("ledger", Some(3)),
            SciOxideError::execution_stopped(ExecutionStopReason::Cancelled),
            SciOxideError::capability_unavailable(Capability::Clock),
            SciOxideError::numerical_instability(NumericalIssue::Overflow, "overflow"),
            SciOxideError::resource_limit(quantity!("workspace_bytes"), 128, 64),
        ];
        for error in errors {
            assert_eq!(error, error.clone());
            writeln!(Sink, "{error}").unwrap();
            assert!(error.source().is_none());
        }
    };
}

#[test]
fn owned_details_shapes_and_counts_are_retained() {
    let error = SciOxideError::invalid_input(
        argument!("x"),
        InvalidInputReason::OutOfRange,
        format!("value {}", 7),
    );
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput {
            detail: "value 7",
            ..
        }
    ));
    assert_eq!(error, error.clone());
    let error = SciOxideError::dimension_mismatch(
        Dimensions::from(&[2, 3, 4]),
        Dimensions::from(&[2, 3, 5]),
    );
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch {
            expected: [2, 3, 4],
            actual: [2, 3, 5],
            ..
        }
    ));
    assert_eq!(error, error.clone());
    let counts =
        EvaluationCounts::from_iterations(7).with_provider_count(EvaluationRole::Function, 9);
    let error = SciOxideError::convergence_failure(
        ConvergenceReason::IterationLimit,
        ConvergenceDiagnostics::new().with_counts(counts.into()),
        "limit",
    );
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own() == counts)
    );
}

#[test]
fn differentiation_source_is_the_original_cause() {
    let cause = SciOxideError::singular(None);
    let error = SciOxideError::differentiation_failure(
        DifferentiationLocation::variable(0),
        1,
        cause.clone(),
    );
    let error =
        SciOxideError::differentiation_failure(DifferentiationLocation::variable(1), 2, error);
    let SciOxideErrorView::DifferentiationFailure {
        frames,
        cause: actual,
    } = error.view()
    else {
        unreachable!()
    };
    assert_eq!(actual, &cause);
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].evaluations, 2);
    assert_eq!(frames[1].evaluations, 1);
    assert_eq!(error.source().unwrap().to_string(), cause.to_string());
    assert!(error.source().unwrap().source().is_none());
    let chain = format!("{}: {}", error, error.source().unwrap());
    assert_eq!(chain.matches("differentiation failure").count(), 2);
    assert_eq!(chain.matches("singular matrix").count(), 1);
}

#[test]
fn external_details_preserve_their_storage_and_classification() {
    {
        let error = SciOxideError::external("model rejected input");
        assert_eq!(error.kind(), SciOxideErrorKind::External);
        assert!(matches!(
            error.view(),
            SciOxideErrorView::External {
                detail: "model rejected input"
            }
        ));
        assert_eq!(error.clone(), error);
        writeln!(Sink, "{error}").unwrap();
        assert!(error.source().is_none());
    };
    let detail = format!("model rejected input {}", 7);
    let error = SciOxideError::external(detail.clone());
    assert_eq!(error.kind(), SciOxideErrorKind::External);
    assert!(
        matches!(error.view(), SciOxideErrorView::External { detail: actual } if actual == detail)
    );
    assert_eq!(
        error.to_string(),
        "external failure: model rejected input 7"
    );
    assert_eq!(error.clone(), error);
    assert!(error.source().is_none());
}

#[test]
fn iteration_only_failures_construct_clone_and_format_inline() {
    for iterations in [0, 1, usize::MAX] {
        {
            let counts = EvaluationCounts::from_iterations(iterations).into();
            let error = SciOxideError::convergence_failure(
                ConvergenceReason::IterationLimit,
                ConvergenceDiagnostics::new().with_counts(counts),
                "work exhausted",
            );
            assert_eq!(error, error.clone());
            assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
            let SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                detail,
            } = error.view()
            else {
                unreachable!()
            };
            assert_eq!(reason, ConvergenceReason::IterationLimit);
            assert_eq!(detail, "work exhausted");
            assert_eq!(diagnostics.counts(), counts);
            assert!(diagnostics.algorithm().is_none());
            assert!(diagnostics.residual().is_none());
            assert!(diagnostics.budget().is_none());
            assert!(diagnostics.partial_result().is_none());
            assert_eq!(diagnostics.nested_levels().len(), 0);
            writeln!(Sink, "{error}").unwrap();
        };
    }
}

#[test]
fn compact_convergence_retains_disjoint_nested_counts() {
    let identity = SolveIdentity::new(
        ProblemToken::new(1154),
        algorithm_id!("test", "counts", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(parent.record_iteration().unwrap().is_granted());
    SolveScope::from(&mut parent)
        .run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |child| {
                for _ in 0..3 {
                    assert!(child.record_iteration()?.is_granted());
                }
                Ok(())
            },
        )
        .unwrap();
    let counts = parent.counts();
    {
        let error = SciOxideError::convergence_failure(
            ConvergenceReason::IterationLimit,
            ConvergenceDiagnostics::new().with_counts(counts),
            "work exhausted",
        );
        let SciOxideErrorView::ConvergenceFailure { diagnostics, .. } = error.view() else {
            unreachable!()
        };
        assert_eq!(diagnostics.counts(), counts);
        assert_eq!(diagnostics.counts().own().iterations(), 1);
        assert_eq!(diagnostics.counts().nested().iterations(), 3);
        assert_eq!(diagnostics.counts().total().unwrap().iterations(), 4);
        writeln!(Sink, "{error}").unwrap();
    };
}

#[test]
fn richer_convergence_metadata_and_owned_details_remain_lossless() {
    let algorithm = algorithm_id!("test", "rich", 2);
    let input = ConvergenceDiagnostics::new()
        .with_algorithm(algorithm)
        .with_counts(EvaluationCounts::from_iterations(7).into())
        .with_nested(
            algorithm,
            ConvergenceDiagnostics::new().with_counts(EvaluationCounts::from_iterations(3).into()),
        );
    let error = SciOxideError::convergence_failure(
        ConvergenceReason::IterationLimit,
        input.clone(),
        String::from("owned detail"),
    );
    assert_eq!(error, error.clone());
    let SciOxideErrorView::ConvergenceFailure {
        diagnostics,
        detail,
        ..
    } = error.view()
    else {
        unreachable!()
    };
    assert_eq!(detail, "owned detail");
    assert_eq!(diagnostics.algorithm(), input.algorithm());
    assert_eq!(diagnostics.counts(), input.counts());
    assert_eq!(
        diagnostics.nested_levels().next(),
        input.nested_levels().next()
    );
}
