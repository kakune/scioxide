//! Execution context for linear-algebra calls, with an optional BLAS provider.

use scioxide_core::backend::KernelDispatch;
use scioxide_core::{Accuracy, ExecutionContext, NumericScalar, SciOxideError};

#[cfg(feature = "blas")]
use crate::BlasDispatch;

/// Execution context for dense kernels and direct solvers.
///
/// Processor dispatch follows the supplied execution context. An optional
/// `BlasDispatch` may be attached.
#[derive(Clone, Copy)]
#[must_use]
pub(crate) struct KernelContext<'a> {
    execution: ExecutionContext<'a>,
    dispatch: KernelDispatch,
    accuracy: Accuracy,
    policy: scioxide_core::ExecutionPolicy,
    #[cfg(feature = "blas")]
    blas: Option<BlasDispatch<'a>>,
}

impl core::fmt::Debug for KernelContext<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = formatter.debug_struct("KernelContext");
        debug.field("execution", &self.execution);
        debug.field("dispatch", &self.dispatch);
        debug.field("accuracy", &self.accuracy);
        debug.field("policy", &self.policy);
        #[cfg(feature = "blas")]
        debug.field("blas", &self.blas);
        debug.finish()
    }
}

impl<'a> KernelContext<'a> {
    /// Wraps an execution context without a BLAS provider.
    pub(crate) fn new(execution: ExecutionContext<'a>) -> Self {
        Self {
            execution,
            dispatch: KernelDispatch::detect(execution.processor()),
            accuracy: Accuracy::Standard,
            policy: scioxide_core::ExecutionPolicy::Auto,
            #[cfg(feature = "blas")]
            blas: None,
        }
    }

    /// Returns the portable sequential context.
    #[cfg(test)]
    pub(crate) fn sequential() -> KernelContext<'static> {
        KernelContext::new(ExecutionContext::sequential())
    }

    /// Returns the sequential context with native processor dispatch.
    pub(crate) fn sequential_native() -> KernelContext<'static> {
        KernelContext::new(ExecutionContext::sequential_native())
    }

    /// Uses the operator context's dispatch, or detects it from its processor.
    /// Operator substitutions and compact products use `Standard` accuracy.
    pub(crate) fn for_operator<S: NumericScalar>(
        context: &scioxide_core::OperatorContext<'_, '_, 'a, S>,
    ) -> Self {
        let execution = context.execution();
        Self {
            execution,
            dispatch: context.kernels().map_or_else(
                || KernelDispatch::detect(execution.processor()),
                scioxide_core::DenseOperatorKernel::dispatch,
            ),
            accuracy: Accuracy::Standard,
            policy: scioxide_core::ExecutionPolicy::Auto,
            #[cfg(feature = "blas")]
            blas: None,
        }
    }

    /// Returns the execution context.
    pub(crate) const fn execution(self) -> ExecutionContext<'a> {
        self.execution
    }

    /// Sets the product accuracy.
    pub(crate) const fn with_accuracy(mut self, accuracy: Accuracy) -> Self {
        self.accuracy = accuracy;
        self
    }

    /// Returns the product accuracy.
    pub(crate) const fn accuracy(self) -> Accuracy {
        self.accuracy
    }

    pub(crate) const fn policy(self) -> scioxide_core::ExecutionPolicy {
        self.policy
    }

    pub(crate) const fn dispatch(self) -> KernelDispatch {
        self.dispatch
    }

    /// Returns the attached BLAS dispatch, if any.
    #[cfg(feature = "blas")]
    pub(crate) const fn blas(self) -> Option<BlasDispatch<'a>> {
        self.blas
    }
}

impl<'capability> KernelContext<'capability> {
    /// Returns a context for a nested solve under `parent`, with the same dispatch and provider.
    pub(crate) fn scoped<'scope, 'state>(
        self,
        parent: &'scope mut scioxide_core::SolveContext<'state, 'capability>,
    ) -> LinalgExecutionContext<'scope, 'capability> {
        LinalgExecutionContext {
            scope: parent.into(),
            dispatch: self.dispatch,
            accuracy: self.accuracy,
            policy: self.policy,
            iteration_budget: None,
            #[cfg(feature = "blas")]
            blas: self.blas,
        }
    }
}

impl<'a> From<ExecutionContext<'a>> for KernelContext<'a> {
    fn from(value: ExecutionContext<'a>) -> Self {
        Self::new(value)
    }
}

/// Execution context passed to the `*_into` forms of linear-algebra operations.
///
/// It wraps a [`SolveScope`](scioxide_core::SolveScope) together with the
/// processor dispatch, detected once, and an optional BLAS provider. Build one
/// with `From`:
///
/// - from an [`ExecutionContext`] for a top-level call, which gets its own
///   solve scope;
/// - from a `&mut` [`SolveContext`](scioxide_core::SolveContext) for a call
///   nested in another solve, whose counts are added to that parent (see
///   [Solve accounting](scioxide_core#solve-accounting)).
///
/// Every `*_into` method takes `impl Into<LinalgExecutionContext>`, so you can
/// pass an `ExecutionContext` directly. Use [`reborrow`](Self::reborrow) to
/// pass one context to several calls. Product accuracy, execution policy and
/// the time limit come from the call's [`LinalgExecutionOptions`](crate::LinalgExecutionOptions).
/// See [Execution](crate#execution).
#[derive(Debug)]
#[must_use]
pub struct LinalgExecutionContext<'scope, 'capability> {
    scope: scioxide_core::SolveScope<'scope, 'capability>,
    dispatch: KernelDispatch,
    accuracy: Accuracy,
    policy: scioxide_core::ExecutionPolicy,
    iteration_budget: Option<scioxide_core::Budget>,
    #[cfg(feature = "blas")]
    blas: Option<BlasDispatch<'capability>>,
}

impl<'scope, 'capability> LinalgExecutionContext<'scope, 'capability> {
    /// Returns a top-level context for `execution`, without a BLAS provider.
    pub fn new(execution: ExecutionContext<'capability>) -> Self {
        execution.into()
    }

    /// Returns a top-level context for [`ExecutionContext::sequential`].
    ///
    /// It runs on the calling thread with portable scalar kernels, no clock,
    /// no cancellation and no BLAS provider.
    pub fn sequential() -> Self {
        // This explicit reference constructor must retain portable dispatch.
        ExecutionContext::sequential().into()
    }

    /// Returns the wrapped execution context.
    pub const fn execution(&self) -> ExecutionContext<'capability> {
        self.scope.execution()
    }

    /// Borrows this context for one call, keeping its dispatch, provider and iteration cap.
    pub fn reborrow(&mut self) -> LinalgExecutionContext<'_, 'capability> {
        LinalgExecutionContext {
            scope: self.scope.reborrow(),
            dispatch: self.dispatch,
            accuracy: self.accuracy,
            policy: self.policy,
            iteration_budget: self.iteration_budget,
            #[cfg(feature = "blas")]
            blas: self.blas,
        }
    }

    /// Caps the iteration count of the iterative solve run with this context.
    ///
    /// The solver runs at most the smaller of `budget` and its own limit.
    /// Calling this twice keeps the smaller cap, and [`reborrow`](Self::reborrow)
    /// keeps it. The cap applies to the solve's own scope only: it does not
    /// change the parent scope's iteration limit, and solves that the solver
    /// runs internally keep their own limits, because nested iterations count
    /// only toward their own scope (see
    /// [Solve accounting](scioxide_core#solve-accounting)).
    ///
    /// ```
    /// use scioxide_core::Budget;
    /// use scioxide_linalg::LinalgExecutionContext;
    /// let mut context = LinalgExecutionContext::sequential()
    ///     .with_iteration_budget(Budget::ONE);
    /// let borrowed = context.reborrow();
    /// assert_eq!(borrowed.execution().check(), Ok(()));
    /// ```
    pub const fn with_iteration_budget(mut self, budget: scioxide_core::Budget) -> Self {
        self.iteration_budget = Some(match self.iteration_budget {
            Some(old) if old.get() < budget.get() => old,
            _ => budget,
        });
        self
    }
    pub(crate) fn cap_iterations(&self, limit: scioxide_core::Budget) -> scioxide_core::Budget {
        self.iteration_budget.map_or(limit, |cap| cap.min(limit))
    }
    /// Attaches a BLAS provider for the kernels that support one.
    #[cfg(feature = "blas")]
    pub const fn with_blas(mut self, blas: BlasDispatch<'capability>) -> Self {
        self.blas = Some(blas);
        self
    }

    /// Returns the attached BLAS dispatch, if any.
    ///
    /// Each kernel decides from the requested accuracy and the provider's
    /// capabilities whether to call the provider.
    #[cfg(feature = "blas")]
    pub const fn blas(&self) -> Option<BlasDispatch<'capability>> {
        self.kernel().blas()
    }

    pub(crate) const fn kernel(&self) -> KernelContext<'capability> {
        KernelContext {
            execution: self.scope.execution(),
            dispatch: self.dispatch,
            accuracy: self.accuracy,
            policy: self.policy,
            #[cfg(feature = "blas")]
            blas: self.blas,
        }
    }

    /// Wraps kernel capabilities in a new top-level scope. Only fixed-cost
    /// kernels, which start no solve scope, use it; nested solver stages use
    /// [`KernelContext::scoped`] so that their work counts toward the parent.
    pub(crate) fn unscoped(kernels: KernelContext<'capability>) -> Self {
        Self {
            scope: kernels.execution().into(),
            dispatch: kernels.dispatch,
            accuracy: kernels.accuracy,
            policy: kernels.policy,
            iteration_budget: None,
            #[cfg(feature = "blas")]
            blas: kernels.blas,
        }
    }

    pub(crate) fn into_scope(self) -> scioxide_core::SolveScope<'scope, 'capability> {
        self.scope
    }

    pub(crate) fn with_options(
        mut self,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        self.accuracy = options.accuracy();
        self.policy = options.execution_policy();
        if let Some(duration) = options.time_limit() {
            self.scope = self.scope.with_time_limit(duration)?;
        }
        Ok(self)
    }
}

impl<'scope, 'capability> From<ExecutionContext<'capability>>
    for LinalgExecutionContext<'scope, 'capability>
{
    fn from(execution: ExecutionContext<'capability>) -> Self {
        scioxide_core::SolveScope::from(execution).into()
    }
}

impl<'scope, 'capability> From<scioxide_core::SolveScope<'scope, 'capability>>
    for LinalgExecutionContext<'scope, 'capability>
{
    fn from(scope: scioxide_core::SolveScope<'scope, 'capability>) -> Self {
        Self {
            dispatch: KernelDispatch::detect(scope.execution().processor()),
            scope,
            accuracy: Accuracy::Standard,
            policy: scioxide_core::ExecutionPolicy::Auto,
            iteration_budget: None,
            #[cfg(feature = "blas")]
            blas: None,
        }
    }
}

impl<'scope, 'state, 'capability> From<&'scope mut scioxide_core::SolveContext<'state, 'capability>>
    for LinalgExecutionContext<'scope, 'capability>
{
    fn from(parent: &'scope mut scioxide_core::SolveContext<'state, 'capability>) -> Self {
        scioxide_core::SolveScope::from(parent).into()
    }
}

impl<S: NumericScalar> scioxide_core::DenseOperatorKernel<S> for KernelContext<'_> {
    fn dispatch(&self) -> KernelDispatch {
        self.dispatch
    }

    fn apply_dense_into(
        &self,
        input_matrix: scioxide_core::DenseOperatorInput<'_, S>,
        input: &[S],
        maximum: S::Real,
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let mut matrix = crate::MatrixView::from_slice_with_strides(
            input_matrix.shape[0],
            input_matrix.shape[1],
            input_matrix.data,
            input_matrix.strides[0],
            input_matrix.strides[1],
        )?;
        if input_matrix.conjugated {
            matrix = matrix.conjugate();
        }
        let matrix = crate::FiniteMatrixView::from_validated(matrix, input_matrix.maximum);
        let context = KernelContext {
            execution,
            dispatch: self.dispatch,
            accuracy: self.accuracy,
            policy: self.policy,
            #[cfg(feature = "blas")]
            blas: self.blas,
        };
        matrix.apply_workspace(input, maximum, output, context)
    }
}
