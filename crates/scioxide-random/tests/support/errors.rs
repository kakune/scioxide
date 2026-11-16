//! Readable assertions for the public structured error contract.
#![allow(dead_code)]
use scioxide_core::{
    Argument, DomainReason, InvalidInputReason, NumericalIssue, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView,
};

pub(crate) fn numerical(error: SciOxideError, expected: NumericalIssue) {
    assert_eq!(
        error.kind(),
        SciOxideErrorKind::NumericalInstability,
        "{error}"
    );
    assert!(
        matches!(error.view(), SciOxideErrorView::NumericalInstability { issue, .. }
        if issue == expected),
        "expected {expected:?}: {error}"
    );
}

pub(crate) fn domain(error: SciOxideError, expected: Argument) {
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError, "{error}");
    assert!(
        matches!(error.view(), SciOxideErrorView::DomainError { argument, reason: DomainReason::OutsideDomain, .. }
        if argument == expected),
        "expected domain of {expected}: {error}"
    );
}

pub(crate) fn dimensions(error: SciOxideError, expected_len: usize, actual_len: usize) {
    assert_eq!(
        error.kind(),
        SciOxideErrorKind::DimensionMismatch,
        "{error}"
    );
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
        if expected == [expected_len] && actual == [actual_len]),
        "expected [{expected_len}], actual [{actual_len}]: {error}"
    );
}

pub(crate) fn invalid(error: SciOxideError, expected: InvalidInputReason) {
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput, "{error}");
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { reason, .. }
        if reason == expected),
        "expected {expected:?}: {error}"
    );
}
