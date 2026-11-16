//! Line searches: Armijo backtracking and strong Wolfe.
//!
//! A trial point that is not representable, lies outside
//! [`crate::ObjectiveFunction::is_in_domain`] or has a non-finite value is
//! rejected and the step shortens. A search that runs out of trials returns an
//! unsuccessful [`crate::LineSearchResult`]; invalid input and callback errors
//! are returned as errors.

mod backtracking;
mod result;
mod wolfe;

pub use backtracking::BacktrackingLineSearch;
pub use result::{LineSearchFailure, LineSearchResult, LineSearchTermination};
pub use wolfe::StrongWolfeLineSearch;

/// Connects a line search to the solve that runs it.
///
/// To run a search on its own, convert an [`scioxide_core::ExecutionContext`]
/// into a scope. The search then uses that context's cancellation and deadline,
/// and its counts cover that search alone. Inside an optimizer, the scope
/// counts each trial toward the optimizer's evaluation and iteration limits
/// (see [solve accounting](scioxide_core#solve-accounting)).
///
/// A custom [`crate::LineSearch`] calls [`Self::propose`] before it forms each
/// trial point, and [`Self::reject`] after a trial fails the acceptance test.
/// A trial whose callback returns an error is not a rejection.
#[derive(Clone, Copy)]
pub struct LineSearchScope<'a> {
    kind: ScopeKind<'a>,
}

#[derive(Clone, Copy)]
enum ScopeKind<'a> {
    Standalone(scioxide_core::ExecutionContext<'a>),
    Observed(&'a dyn crate::context::Observation),
}

impl<'a> From<scioxide_core::ExecutionContext<'a>> for LineSearchScope<'a> {
    fn from(execution: scioxide_core::ExecutionContext<'a>) -> Self {
        Self {
            kind: ScopeKind::Standalone(execution),
        }
    }
}

impl<'a> LineSearchScope<'a> {
    pub(crate) fn observed(observer: &'a dyn crate::context::Observation) -> Self {
        Self {
            kind: ScopeKind::Observed(observer),
        }
    }

    /// Capabilities of a standalone search, absent inside an optimizer.
    pub(crate) const fn standalone(self) -> Option<scioxide_core::ExecutionContext<'a>> {
        match self.kind {
            ScopeKind::Standalone(execution) => Some(execution),
            ScopeKind::Observed(_) => None,
        }
    }

    pub(crate) fn counts(
        self,
    ) -> Result<scioxide_core::EvaluationCounts, scioxide_core::SciOxideError> {
        match self.kind {
            ScopeKind::Standalone(_) => Ok(scioxide_core::EvaluationCounts::new()),
            ScopeKind::Observed(observer) => observer.counts(),
        }
    }

    pub(crate) fn statistics_since(
        self,
        initial: scioxide_core::EvaluationCounts,
    ) -> Result<(usize, usize, usize), scioxide_core::SciOxideError> {
        let current = self.counts()?;
        use scioxide_core::EvaluationRole as R;
        Ok((
            current.provider(R::Objective) - initial.provider(R::Objective),
            current.provider(R::Gradient) - initial.provider(R::Gradient),
            current.rejected() - initial.rejected(),
        ))
    }

    /// Returns the execution context of the solve that runs the search.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the context of the enclosing solve is already
    /// borrowed.
    pub fn execution(
        self,
    ) -> Result<scioxide_core::ExecutionContext<'a>, scioxide_core::SciOxideError> {
        match self.kind {
            ScopeKind::Standalone(execution) => Ok(execution),
            ScopeKind::Observed(observer) => observer.execution(),
        }
    }

    /// Counts a new trial before the search forms the trial point or calls the objective.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` if the trial would exceed a limit of the
    ///   enclosing solve.
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - `SizeOverflow` if a counter overflows.
    pub fn propose(self) -> Result<(), scioxide_core::SciOxideError> {
        match self.kind {
            ScopeKind::Standalone(_) => Ok(()),
            ScopeKind::Observed(observer) => observer.proposal(),
        }
    }

    /// Counts a completed trial that failed the acceptance test.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the counter overflows.
    pub fn reject(self) -> Result<(), scioxide_core::SciOxideError> {
        match self.kind {
            ScopeKind::Standalone(_) => Ok(()),
            ScopeKind::Observed(observer) => observer.reject(),
        }
    }
}

/// Validate an optional initial-step hint and clamp it to the search limits.
fn initial_step<S: scioxide_core::RealScalar>(
    hint: Option<S>,
    configured: S,
    minimum: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let Some(hint) = hint else {
        return Ok(configured);
    };
    if !hint.is_finite() || hint <= S::ZERO {
        return Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("initial_step"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "initial step hint must be finite and positive",
        ));
    }
    Ok(hint.max(minimum).min(configured))
}

/// Run one standalone search in a local epoch observed by its own session.
pub(crate) fn standalone<T>(
    search: &impl core::fmt::Debug,
    dimension: usize,
    execution: scioxide_core::ExecutionContext<'_>,
    run: impl FnOnce(
        &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<T, scioxide_core::SciOxideError>,
) -> Result<T, scioxide_core::SciOxideError> {
    let identity = crate::context::identity(
        scioxide_core::algorithm_id!("optimize", "StandaloneLineSearch", 1),
        scioxide_core::ProblemToken::new(0),
        dimension,
        crate::context::options_token(search),
    )?;
    let mut state = scioxide_core::SolveState::new();
    scioxide_core::SolveScope::from(execution).run(
        &mut state,
        identity,
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            let session =
                crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
            run(&session)
        },
    )
}

/// Owned standalone searches for unit tests of the built-in strategies.
#[cfg(test)]
pub(crate) mod testing {
    use alloc::vec::Vec;

    use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};
    use scioxide_differentiate::{DerivativeOperation, DerivativeStrategy};

    use crate::{
        DerivativeWorkspace, LineSearch, LineSearchResult, LineSearchTermination, ObjectiveFunction,
    };

    /// A standalone outcome together with its owned trial buffers.
    #[derive(Debug)]
    pub(crate) struct Searched<S: RealScalar> {
        result: LineSearchResult<S>,
        gradient: Vec<S>,
    }

    impl<S: RealScalar> Searched<S> {
        pub(crate) const fn termination(&self) -> LineSearchTermination {
            self.result.termination()
        }
        pub(crate) const fn is_success(&self) -> bool {
            self.result.is_success()
        }
        pub(crate) const fn has_trial(&self) -> bool {
            self.result.has_trial()
        }
        pub(crate) fn step(&self) -> Option<S> {
            self.result.has_trial().then(|| self.result.step())
        }
        pub(crate) fn value(&self) -> Option<S> {
            self.result.has_trial().then(|| self.result.value())
        }
        pub(crate) fn gradient(&self) -> Option<&[S]> {
            self.result
                .gradient_available()
                .then_some(self.gradient.as_slice())
        }
        pub(crate) const fn directional_derivative(&self) -> Option<S> {
            self.result.directional_derivative()
        }
        pub(crate) const fn fun_evals(&self) -> usize {
            self.result.fun_evals()
        }
        pub(crate) const fn grad_evals(&self) -> usize {
            self.result.grad_evals()
        }
        pub(crate) const fn rejected_trials(&self) -> usize {
            self.result.rejected_trials()
        }
    }

    /// Allocate trial buffers and run a standalone sequential search.
    pub(crate) trait Standalone<S: RealScalar + NumericScalar<Real = S>>:
        LineSearch<S>
    {
        fn search<F: ObjectiveFunction<S> + ?Sized>(
            &self,
            f: &F,
            x: &[S],
            f_x: S,
            direction: &[S],
            slope: S,
        ) -> Result<Searched<S>, SciOxideError> {
            self.search_with_strategy(f, x, f_x, direction, slope, DerivativeStrategy::default())
        }

        fn search_with_strategy<F: ObjectiveFunction<S> + ?Sized>(
            &self,
            f: &F,
            x: &[S],
            f_x: S,
            direction: &[S],
            slope: S,
            strategy: DerivativeStrategy,
        ) -> Result<Searched<S>, SciOxideError> {
            let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
            let mut point = alloc::vec![zero; x.len()];
            let mut gradient = alloc::vec![zero; x.len()];
            let mut candidate = alloc::vec![zero; x.len()];
            let mut candidate_gradient = alloc::vec![zero; x.len()];
            let mut derivatives =
                DerivativeWorkspace::new(DerivativeOperation::Gradient, x.len(), 1)?;
            let result = self.search_into(
                f,
                x,
                f_x,
                direction,
                slope,
                strategy,
                &mut point,
                &mut gradient,
                &mut candidate,
                &mut candidate_gradient,
                &mut derivatives,
                ExecutionContext::sequential().into(),
                None,
            )?;
            Ok(Searched { result, gradient })
        }
    }

    impl<S: RealScalar + NumericScalar<Real = S>, L: LineSearch<S>> Standalone<S> for L {}
}
