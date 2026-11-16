//! Result of checking a workspace against its requirements.

/// Whether a workspace is large enough for a requirement.
///
/// Returned by `check` on growable workspaces; see
/// [plans and workspaces](crate#plans-and-workspaces).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Capacity {
    /// The workspace is large enough; running the plan will not grow it.
    Sufficient,
    /// The workspace must grow before the plan runs.
    NeedsGrowth,
}
