//! Execution contexts for spatial batch operations.

use scioxide_core::{ExecutionContext, SolveContext, SolveScope};

// Each operation owns its epoch and budget. Identity never enters a public API.
pub(crate) fn scoped<'scope, 'capability, T>(
    scope: impl Into<SolveScope<'scope, 'capability>>,
    algorithm: scioxide_core::AlgorithmId,
    budgets: scioxide_core::SolveBudgets,
    run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, scioxide_core::SciOxideError>,
) -> Result<T, scioxide_core::SciOxideError> {
    let scope = scope.into();
    scope.execution().check()?;
    let mut state = scioxide_core::SolveState::new();
    scope.run(
        &mut state,
        scioxide_core::SolveIdentity::new(scioxide_core::ProblemToken::new(0), algorithm, 0, 0),
        budgets,
        |context| {
            let result = run(context)?;
            context.execution().check()?;
            Ok(result)
        },
    )
}

/// Execution context for a spatial batch operation.
///
/// Batch functions accept anything that converts into this type: an
/// [`ExecutionContext`], a [`SolveScope`] or a `&mut` [`SolveContext`]. From a
/// solve context the operation takes its executor, cancellation and
/// deadline; it does not count iterations.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_spatial::SpatialExecutionContext;
/// let mut context = SpatialExecutionContext::from(ExecutionContext::sequential());
/// context.reborrow().execution().check()?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct SpatialExecutionContext<'scope, 'capability> {
    scope: SolveScope<'scope, 'capability>,
}

impl<'capability> SpatialExecutionContext<'_, 'capability> {
    /// Returns a context that runs on the calling thread, without
    /// cancellation or deadline.
    pub fn sequential() -> Self {
        ExecutionContext::sequential().into()
    }

    /// Returns the underlying execution context.
    pub const fn execution(&self) -> ExecutionContext<'capability> {
        self.scope.execution()
    }

    /// Returns a shorter-lived context that can be passed to another call.
    pub fn reborrow(&mut self) -> SpatialExecutionContext<'_, 'capability> {
        self.scope.reborrow().into()
    }
}

impl<'scope, 'capability> From<ExecutionContext<'capability>>
    for SpatialExecutionContext<'scope, 'capability>
{
    fn from(value: ExecutionContext<'capability>) -> Self {
        SolveScope::from(value).into()
    }
}

impl<'scope, 'capability> From<SolveScope<'scope, 'capability>>
    for SpatialExecutionContext<'scope, 'capability>
{
    fn from(scope: SolveScope<'scope, 'capability>) -> Self {
        Self { scope }
    }
}

impl<'scope, 'state, 'capability> From<&'scope mut SolveContext<'state, 'capability>>
    for SpatialExecutionContext<'scope, 'capability>
{
    fn from(parent: &'scope mut SolveContext<'state, 'capability>) -> Self {
        SolveScope::from(parent).into()
    }
}
