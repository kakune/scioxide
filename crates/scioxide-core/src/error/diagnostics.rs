//! Diagnostics attached to convergence failures, including nested solvers.
use crate::{
    AcceptedStateMetadata, AccountingIntegrity, AlgorithmId, BudgetReport, EvaluationCounts,
    ResidualReport, SolveCounts,
};
use alloc::vec::Vec;
use core::fmt;

/// Diagnostics reported by a failed iterative process.
#[derive(Debug, Clone, PartialEq, Default)]
#[must_use]
pub struct ConvergenceDiagnostics {
    algorithm: Option<AlgorithmId>,
    counts: SolveCounts,
    budget: Option<BudgetReport>,
    residual: Option<ResidualReport<f64>>,
    partial: Option<AcceptedStateMetadata>,
    nested: Vec<NestedLevel>,
}

impl ConvergenceDiagnostics {
    pub(super) fn iteration_counts(&self) -> Option<(usize, usize)> {
        let own = self.counts.own().iterations();
        let nested = self.counts.nested().iterations();
        (self.algorithm.is_none()
            && self.budget.is_none()
            && self.residual.is_none()
            && self.partial.is_none()
            && self.nested.is_empty()
            && self.counts == SolveCounts::from_iteration_counts(own, nested))
        .then_some((own, nested))
    }

    /// Returns empty diagnostics.
    pub const fn new() -> Self {
        Self {
            algorithm: None,
            counts: SolveCounts::new(),
            budget: None,
            residual: None,
            partial: None,
            nested: Vec::new(),
        }
    }

    /// Returns a copy with the algorithm identity.
    pub const fn with_algorithm(mut self, algorithm: AlgorithmId) -> Self {
        self.algorithm = Some(algorithm);
        self
    }

    /// Returns a copy with the work counts of the solve and its nested solves.
    pub const fn with_counts(mut self, counts: SolveCounts) -> Self {
        self.counts = counts;
        self
    }
    /// Returns a copy with the budget report of the limit that ended the solve.
    pub const fn with_budget(mut self, budget: BudgetReport) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Returns a copy with the final residual report.
    pub const fn with_residual(mut self, residual: ResidualReport<f64>) -> Self {
        self.residual = Some(residual);
        self
    }

    /// Returns a copy with the metadata of the last accepted state.
    pub const fn with_partial_result(mut self, partial: AcceptedStateMetadata) -> Self {
        self.partial = Some(partial);
        self
    }

    /// Appends `inner`, the diagnostics of nested solver `algorithm`, and its own nested levels.
    ///
    /// Levels are kept outermost first.
    pub fn with_nested(mut self, algorithm: AlgorithmId, inner: Self) -> Self {
        self.nested.push(NestedLevel {
            algorithm,
            counts: inner.counts,
            budget: inner.budget,
            residual: inner.residual,
            partial: inner.partial,
        });
        self.nested.extend(inner.nested);
        self
    }
    /// Returns the algorithm identity, if set.
    pub const fn algorithm(&self) -> Option<AlgorithmId> {
        self.algorithm
    }

    /// Returns the work counts of the solve and its nested solves.
    pub const fn counts(&self) -> SolveCounts {
        self.counts
    }

    /// Returns the budget report of the limit that ended the solve, if any.
    pub const fn budget(&self) -> Option<BudgetReport> {
        self.budget
    }

    /// Returns the final residual report, if set.
    pub const fn residual(&self) -> Option<ResidualReport<f64>> {
        self.residual
    }

    /// Returns the metadata of the last accepted state, if set.
    pub const fn partial_result(&self) -> Option<AcceptedStateMetadata> {
        self.partial
    }

    /// Returns the nested solver levels, outermost first.
    pub fn nested_levels(&self) -> impl ExactSizeIterator<Item = &NestedLevel> {
        self.nested.iter()
    }
}

/// Diagnostics of one nested solver level.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedLevel {
    algorithm: AlgorithmId,
    counts: SolveCounts,
    budget: Option<BudgetReport>,
    residual: Option<ResidualReport<f64>>,
    partial: Option<AcceptedStateMetadata>,
}
impl NestedLevel {
    /// Returns the algorithm identity of the nested solver.
    pub const fn algorithm(&self) -> AlgorithmId {
        self.algorithm
    }
    /// Returns the work counts of the solve and its nested solves.
    pub const fn counts(&self) -> SolveCounts {
        self.counts
    }
    /// Returns the budget report of the limit that ended the nested solve, if any.
    pub const fn budget(&self) -> Option<BudgetReport> {
        self.budget
    }
    /// Returns the residual report, if any.
    pub const fn residual(&self) -> Option<ResidualReport<f64>> {
        self.residual
    }
    /// Returns the metadata of the last accepted state, if any.
    pub const fn partial_result(&self) -> Option<AcceptedStateMetadata> {
        self.partial
    }
}

fn display_level(
    formatter: &mut fmt::Formatter<'_>,
    algorithm: Option<AlgorithmId>,
    counts: SolveCounts,
    budget: Option<BudgetReport>,
    residual: Option<ResidualReport<f64>>,
    partial: Option<AcceptedStateMetadata>,
) -> fmt::Result {
    let mut separator = " [";
    if let Some(algorithm) = algorithm {
        write!(formatter, "{separator}algorithm={algorithm}")?;
        separator = ", ";
    }
    if counts.own() != EvaluationCounts::new() {
        write!(formatter, "{separator}own({})", counts.own())?;
        separator = ", ";
    }
    if counts.nested() != EvaluationCounts::new() {
        write!(formatter, "{separator}nested-work({})", counts.nested())?;
        separator = ", ";
    }
    if counts.integrity() == AccountingIntegrity::Invalid {
        write!(formatter, "{separator}accounting=invalid")?;
        separator = ", ";
    }
    if let Some(budget) = budget {
        write!(
            formatter,
            "{separator}budget={}:{}/",
            budget.kind(),
            budget.consumed()
        )?;
        write!(formatter, "{}", budget.limit())?;
        separator = ", ";
    }
    if let Some(residual) = residual {
        write!(
            formatter,
            "{separator}residual(raw={}, scaled={})",
            residual.raw_norm(),
            residual.scaled_norm()
        )?;
        separator = ", ";
    }
    if let Some(partial) = partial {
        write!(
            formatter,
            "{separator}accepted_revision={}, trust={}",
            partial.revision(),
            partial.trust()
        )?;
        separator = ", ";
    }
    if separator == ", " {
        formatter.write_str("]")?;
    }
    Ok(())
}
impl fmt::Display for ConvergenceDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_level(
            f,
            self.algorithm,
            self.counts,
            self.budget,
            self.residual,
            self.partial,
        )?;
        for level in &self.nested {
            write!(f, " [nested={}", level.algorithm)?;
            display_level(
                f,
                None,
                level.counts,
                level.budget,
                level.residual,
                level.partial,
            )?;
            f.write_str("]")?;
        }
        Ok(())
    }
}

/// Borrowed view of the diagnostics of a convergence failure.
///
/// The accessors return the same data as [`ConvergenceDiagnostics`] without
/// building an owned copy.
///
/// ```
/// use scioxide_core::{ConvergenceDiagnostics, ConvergenceReason, SciOxideError, SciOxideErrorView};
///
/// let error = SciOxideError::convergence_failure(
///     ConvergenceReason::IterationLimit,
///     ConvergenceDiagnostics::new(),
///     "iteration limit reached",
/// );
/// if let SciOxideErrorView::ConvergenceFailure { diagnostics, .. } = error.view() {
///     assert_eq!(diagnostics.counts().own().iterations(), 0);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConvergenceDiagnosticsView<'a>(DiagnosticView<'a>);
#[derive(Debug, Clone, Copy)]
enum DiagnosticView<'a> {
    Full(&'a ConvergenceDiagnostics),
    Iterations { own: usize, nested: usize },
}
impl<'a> ConvergenceDiagnosticsView<'a> {
    pub(super) const fn full(value: &'a ConvergenceDiagnostics) -> Self {
        Self(DiagnosticView::Full(value))
    }
    pub(super) const fn iterations(own: usize, nested: usize) -> Self {
        Self(DiagnosticView::Iterations { own, nested })
    }
    /// Returns the algorithm identity, if set.
    pub const fn algorithm(&self) -> Option<AlgorithmId> {
        match self.0 {
            DiagnosticView::Full(d) => d.algorithm(),
            DiagnosticView::Iterations { .. } => None,
        }
    }
    /// Returns the work counts of the solve and its nested solves.
    pub const fn counts(&self) -> SolveCounts {
        match self.0 {
            DiagnosticView::Full(d) => d.counts(),
            DiagnosticView::Iterations { own, nested } => {
                SolveCounts::from_iteration_counts(own, nested)
            }
        }
    }
    /// Returns the budget report of the limit that ended the solve, if any.
    pub const fn budget(&self) -> Option<BudgetReport> {
        match self.0 {
            DiagnosticView::Full(d) => d.budget(),
            DiagnosticView::Iterations { .. } => None,
        }
    }
    /// Returns the final residual report, if set.
    pub const fn residual(&self) -> Option<ResidualReport<f64>> {
        match self.0 {
            DiagnosticView::Full(d) => d.residual(),
            DiagnosticView::Iterations { .. } => None,
        }
    }
    /// Returns the metadata of the last accepted state, if set.
    pub const fn partial_result(&self) -> Option<AcceptedStateMetadata> {
        match self.0 {
            DiagnosticView::Full(d) => d.partial_result(),
            DiagnosticView::Iterations { .. } => None,
        }
    }
    /// Returns the nested solver levels, outermost first.
    pub fn nested_levels(&self) -> impl ExactSizeIterator<Item = &'a NestedLevel> {
        match self.0 {
            DiagnosticView::Full(d) => d.nested.iter(),
            DiagnosticView::Iterations { .. } => [].iter(),
        }
    }
}
impl fmt::Display for ConvergenceDiagnosticsView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            DiagnosticView::Full(d) => d.fmt(f),
            DiagnosticView::Iterations { own, nested } => display_level(
                f,
                None,
                SolveCounts::from_iteration_counts(own, nested),
                None,
                None,
                None,
            ),
        }
    }
}
