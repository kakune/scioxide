//! Total, allocation-free presentation of externally supplied accounting.
#![allow(clippy::unwrap_used)]

use core::fmt::{self, Write};

use scioxide_core::{
    BudgetKind, ConvergenceDiagnostics, EvaluationCounts, EvaluationRole, SolveBudgets,
};

struct Text {
    bytes: [u8; 2048],
    len: usize,
}
impl Text {
    fn new() -> Self {
        Self {
            bytes: [0; 2048],
            len: 0,
        }
    }
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.len]).unwrap()
    }
}
impl Write for Text {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let end = self.len.checked_add(text.len()).ok_or(fmt::Error)?;
        self.bytes
            .get_mut(self.len..end)
            .ok_or(fmt::Error)?
            .copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

#[test]
fn overflowing_counts_are_presented_without_allocation_or_panic() {
    let counts = EvaluationCounts::new()
        .with_provider_count(EvaluationRole::Function, usize::MAX)
        .with_provider_count(EvaluationRole::Gradient, 1);
    let report = ConvergenceDiagnostics::new().with_counts((counts).into());
    let expected = format!(
        " [own(iterations=0, evaluations=overflow, function={}, gradient=1)]",
        usize::MAX
    );
    assert_eq!(report.to_string(), expected);
    assert_eq!(report.counts().own(), counts);
    assert!(counts.total_providers().is_err());
    {
        let mut text = Text::new();
        write!(text, "{report}").unwrap();
        assert_eq!(text.as_str(), expected);
    };
}

#[test]
fn empty_unbounded_and_nested_text_is_explicit() {
    assert_eq!(ConvergenceDiagnostics::new().to_string(), "");
    let report = ConvergenceDiagnostics::new()
        .with_budget(SolveBudgets::unbounded().report(BudgetKind::Evaluation));
    assert_eq!(report.to_string(), " [budget=evaluation:0/unbounded]");
    let nested = ConvergenceDiagnostics::new()
        .with_nested(scioxide_core::algorithm_id!("test", "inner", 1), report);
    assert_eq!(
        nested.to_string(),
        " [nested=test/inner@1 [budget=evaluation:0/unbounded]]"
    );
    {
        write!(Text::new(), "{nested}").unwrap();
    };
}

#[test]
fn destination_failure_is_propagated() {
    struct Reject;
    impl Write for Reject {
        fn write_str(&mut self, _: &str) -> fmt::Result {
            Err(fmt::Error)
        }
    }
    let report =
        ConvergenceDiagnostics::new().with_counts((EvaluationCounts::from_iterations(1)).into());
    assert!(write!(Reject, "{report}").is_err());
}

#[test]
fn trust_names_have_stable_typed_text() {
    use scioxide_core::{
        ExecutionContext, PartialResultTrust, ProblemToken, SolveIdentity, SolveState,
    };
    for (trust, name) in [
        (PartialResultTrust::DiagnosticOnly, "diagnostic-only"),
        (PartialResultTrust::Admissible, "admissible"),
        (PartialResultTrust::ResidualChecked, "residual-checked"),
        (
            PartialResultTrust::CertificateVerified,
            "certificate-verified",
        ),
    ] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let id = SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("test", "trust", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
            .unwrap();
        context.commit_accepted(trust).unwrap();
        let report =
            ConvergenceDiagnostics::new().with_partial_result(context.accepted_state().unwrap());
        assert_eq!(
            report.to_string(),
            format!(" [accepted_revision=1, trust={name}]")
        );
    }
}
