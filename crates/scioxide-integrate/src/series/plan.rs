//! Series plans: one-call summation and resumable summation.

use super::runner::{Runner, Transform};
use super::workspace::SeriesStorage;
use super::{
    SeriesMethod, SeriesOptions, SeriesProblem, SeriesReport, SeriesRequirements, SeriesResult,
    SeriesState, SeriesTermination, SeriesTerms, SeriesWorkspace, euler, kernel,
};
use scioxide_core::backend::checked_mul;
use scioxide_core::{
    AlgebraicScalar, BudgetKind, ExecutionPolicy, NumericScalar, SciOxideError, SolveContext,
};

/// Sums an infinite series `Σₙ aₙ`, `n = first, first + 1, …`, by a
/// convergence accelerator or by Euler–Maclaurin summation.
///
/// The terms `aₙ` are real, complex or vectors of either; the components of a
/// vector are summed independently. Each partial sum is the correctly rounded
/// sum of its terms (real and imaginary parts separately), so long sums do not
/// accumulate rounding error. After each term the plan forms a new estimate of
/// the sum with its [`SeriesMethod`], and stops as described on
/// [`SeriesTermination`]. Each term costs one provider call, or `order + 3`
/// calls for Euler–Maclaurin. Per term and component, Wynn's epsilon algorithm
/// costs `O(order²)` operations and Levin's u transform `O(order)` when no
/// terms are left out.
///
/// The returned value of each component is the newest estimate, unless its
/// rounding error alone exceeds the error estimate of an earlier estimate; the
/// earlier estimate is then returned.
///
/// [`SeriesSummation::sum`] and [`SeriesSummation::sum_into`] sum the series in one call.
/// [`SeriesSummation::advance`] sums at most
/// [`terms_per_advance`](crate::SeriesOptions::terms_per_advance) new terms
/// per call into a [`SeriesState`], which you can resume later.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{SeriesMethod, SeriesOptions, SeriesSummation, SeriesProblem};
///
/// // Geometric series: Σₙ 2⁻ⁿ = 2.
/// let plan = SeriesSummation::<f64>::new(SeriesMethod::WynnEpsilon, 0, SeriesOptions::default())?;
/// let terms = |n: usize, row: &mut [f64]| {
///     row[0] = 0.5_f64.powi(n as i32);
///     Ok(())
/// };
/// let result = plan.sum(SeriesProblem::Terms(&terms))?;
/// assert!((result.value() - 2.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SeriesSummation<T: NumericScalar> {
    pub(super) method: SeriesMethod,
    pub(super) first: usize,
    pub(super) options: SeriesOptions<T::Real>,
}

impl<T: NumericScalar> SeriesSummation<T> {
    /// Creates a plan that sums the series from index `first` with `method`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `options.order()` is above 32 for `LevinU`, outside
    ///   2 to 32 for `WynnEpsilon`, or above 12 for `EulerMaclaurin`.
    /// - `InvalidInput` if `first + options.max_terms()` overflows `usize`.
    pub fn new(
        method: SeriesMethod,
        first: usize,
        options: SeriesOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let order = options.order().get();
        let maximum = if method == SeriesMethod::EulerMaclaurin {
            euler::MAX_ORDER
        } else {
            32
        };
        for (unsupported, argument) in [
            (
                order > maximum
                    || method == SeriesMethod::WynnEpsilon
                        && order < kernel::FIRST_ACCELERATED_ORDER,
                scioxide_core::argument!("order"),
            ),
            (
                first.checked_add(options.max_terms().get()).is_none(),
                scioxide_core::argument!("max_terms"),
            ),
        ] {
            if unsupported {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "series order or index range is unsupported",
                ));
            }
        }
        Ok(Self {
            method,
            first,
            options,
        })
    }

    /// Returns the options.
    pub const fn options(&self) -> &SeriesOptions<T::Real> {
        &self.options
    }

    /// Returns the identifier of the method, recorded in solve diagnostics.
    pub fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "Series", 1)
    }

    /// Returns the storage requirements for terms with `width` components.
    ///
    /// The requirements do not depend on `max_terms`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        width: usize,
    ) -> Result<SeriesRequirements, SciOxideError> {
        let order = self.options.order().get();
        checked_mul(order + 1, width, scioxide_core::quantity!("series_window"))?;
        checked_mul(
            super::TERM_BATCH,
            width,
            scioxide_core::quantity!("term_batch"),
        )?;
        Ok(SeriesRequirements {
            method: self.method,
            first: self.first,
            width,
            order,
        })
    }

    /// Creates an empty [`SeriesState`] for terms with `width` components, for
    /// use with [`SeriesSummation::advance`].
    ///
    /// The state stores `O(width × order)` values, independent of `max_terms`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn state(&self, width: usize) -> Result<SeriesState<T>, SciOxideError> {
        SeriesState::allocate(self.method, self.first, self.options.order().get(), width)
    }

    /// Creates a workspace for terms with `width` components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, width: usize) -> Result<SeriesWorkspace<T>, SciOxideError> {
        SeriesWorkspace::allocate(&self.workspace_requirements(width)?)
    }

    /// Adds at most [`terms_per_advance`](crate::SeriesOptions::terms_per_advance)
    /// new terms to the partial sum held in `state`, and returns when the sum
    /// converges or a limit is reached.
    ///
    /// The state keeps the partial sum and the last `order + 1` terms, so a
    /// later call with the same state continues from the next term without
    /// evaluating earlier terms again. Any plan with the same method, first
    /// index and order can resume the state. After each term is added, `sink`
    /// receives the state, whose [`value`](SeriesState::value) is the current
    /// estimate. The report counts the terms summed by all calls on the state.
    ///
    /// If the provider or `sink` fails, the state keeps every term added
    /// before the failure, and you can resume it. Calls to `sink` that
    /// already happened are not undone.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `state` was created for a plan with another method,
    ///   first index or order, if `ws` cannot serve `state` (see
    ///   [`SeriesWorkspace`]), or if `problem` does not match the method.
    /// - `InvalidInput` if `state` holds terms but lost its solve checkpoint
    ///   because an earlier call failed while recording its counts; call
    ///   [`SeriesState::reset`] to start again.
    /// - `DimensionMismatch` if the tolerance has per-component absolute
    ///   tolerances of another length than `state`.
    /// - `NonFinite` if a term, a partial sum or an Euler–Maclaurin provider
    ///   value is NaN or infinite.
    /// - `InvalidInput` if an Euler–Maclaurin provider returns a negative or
    ///   non-finite error bound.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the call before the state holds any term.
    /// - Any error returned by the provider or by `sink`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Budget, ExecutionContext};
    /// use scioxide_integrate::{SeriesMethod, SeriesOptions, SeriesSummation, SeriesProblem};
    ///
    /// // Σₙ 2⁻ⁿ = 2 in calls of at most four terms; the second call resumes at term 4.
    /// let options = SeriesOptions::default().with_terms_per_advance(Budget::try_from(4)?);
    /// let plan = SeriesSummation::<f64>::new(SeriesMethod::LevinU, 0, options)?;
    /// let (mut state, mut ws) = (plan.state(1)?, plan.workspace(1)?);
    /// let terms = |n: usize, row: &mut [f64]| {
    ///     row[0] = 0.5_f64.powi(n as i32);
    ///     Ok(())
    /// };
    /// let problem = SeriesProblem::Terms(&terms);
    /// let context = ExecutionContext::sequential_native();
    /// let first = plan.advance(problem, &mut state, |_| Ok(()), &mut ws, context)?;
    /// assert_eq!(first.terms(), 4);
    /// plan.advance(problem, &mut state, |_| Ok(()), &mut ws, context)?;
    /// assert!((state.value().unwrap()[0] - 2.0).abs() < 1e-10);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn advance<'scope, 'capability, F>(
        &self,
        problem: SeriesProblem<'_, T>,
        state: &mut SeriesState<T>,
        mut sink: F,
        ws: &mut SeriesWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<SeriesReport, SciOxideError>
    where
        F: FnMut(&SeriesState<T>) -> Result<(), SciOxideError>,
    {
        self.execute(
            problem,
            state,
            &mut sink,
            &mut ws.storage,
            scope.into(),
            self.options.terms_per_advance().get(),
        )
    }

    /// Sums the series from its first term and writes the estimate to `out`.
    ///
    /// `out` has one entry per component. The error estimates are available
    /// from [`SeriesWorkspace::component_errors`] afterwards. The call sums at
    /// most `max_terms` terms and does not use a [`SeriesState`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` and the workspace width differ.
    /// - The other errors of [`SeriesSummation::advance`], except those about the
    ///   state.
    pub fn sum_into<'scope, 'capability>(
        &self,
        problem: SeriesProblem<'_, T>,
        out: &mut [T],
        ws: &mut SeriesWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<SeriesReport, SciOxideError> {
        let SeriesWorkspace { atomic, storage } = ws;
        if let Err(error) = crate::kernel::shape(storage.output.len(), out.len()) {
            return storage.attempts.reject(error);
        }
        // A sufficient workspace may retain a longer window than this plan uses.
        atomic.reset_as(self.method, self.first, self.options.order().get());
        let result = self.execute(
            problem,
            atomic,
            &mut |_| Ok(()),
            storage,
            scope.into(),
            self.options.max_terms().get(),
        );
        if result.is_ok() {
            out.copy_from_slice(&storage.output);
        }
        result
    }

    /// Sums a scalar real or complex series with a new workspace and a
    /// sequential execution context.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the workspace cannot be
    ///   allocated.
    /// - The errors of [`SeriesSummation::sum_into`].
    pub fn sum(&self, problem: SeriesProblem<'_, T>) -> Result<SeriesResult<T>, SciOxideError> {
        let mut ws = self.workspace(1)?;
        let mut output = [T::ZERO];
        let report = self.sum_into(
            problem,
            &mut output,
            &mut ws,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(SeriesResult {
            value: output[0],
            error: ws.storage.errors[0],
            report,
        })
    }

    /// Matches the problem to the plan's method and checks the state, workspace
    /// and tolerance before any provider call.
    fn preflight<'p>(
        &self,
        problem: SeriesProblem<'p, T>,
        state: &SeriesState<T>,
        storage: &SeriesStorage<T>,
    ) -> Result<(&'p dyn SeriesTerms<T>, Transform<'p, T>), SciOxideError> {
        let width = state.width;
        for (mismatch, argument) in [
            (
                state.method != self.method
                    || state.first != self.first
                    || state.order != self.options.order().get(),
                scioxide_core::argument!("state"),
            ),
            (
                !storage
                    .requirements
                    .serves(&self.workspace_requirements(width)?)?,
                scioxide_core::argument!("workspace"),
            ),
        ] {
            if mismatch {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "series state must match the plan and the workspace must serve it",
                ));
            }
        }
        for index in 0..width {
            let _ = self
                .options
                .tolerance()
                .threshold(index, width, T::Real::ONE)?;
        }
        match (problem, self.method) {
            (SeriesProblem::Terms(terms), SeriesMethod::WynnEpsilon) => {
                Ok((terms, Transform::Wynn))
            }
            (SeriesProblem::Terms(terms), SeriesMethod::LevinU) => Ok((terms, Transform::Levin)),
            (SeriesProblem::EulerMaclaurin(smooth), SeriesMethod::EulerMaclaurin) => {
                Ok((smooth, Transform::Euler(smooth)))
            }
            _ => Err(crate::kernel::invalid(
                scioxide_core::argument!("problem"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "series provider does not match method",
            )),
        }
    }

    fn execute(
        &self,
        problem: SeriesProblem<'_, T>,
        state: &mut SeriesState<T>,
        sink: &mut dyn FnMut(&SeriesState<T>) -> Result<(), SciOxideError>,
        storage: &mut SeriesStorage<T>,
        scope: scioxide_core::SolveScope<'_, '_>,
        quota: usize,
    ) -> Result<SeriesReport, SciOxideError> {
        let checkpoint = state.checkpoint;
        let kind = if checkpoint.is_some() || state.count > 0 {
            scioxide_core::AttemptKind::Resume
        } else {
            scioxide_core::AttemptKind::Call
        };
        if state.count > 0 && checkpoint.is_none() {
            return storage.attempts.reject_kind(
                kind,
                crate::kernel::invalid(
                    scioxide_core::argument!("state"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "series checkpoint is unavailable after an accounting failure",
                ),
            );
        }
        let (terms, transform) = match self.preflight(problem, state, storage) {
            Ok(pair) => pair,
            Err(error) => return storage.attempts.reject_kind(kind, error),
        };
        // Core resume restores cumulative counts from the checkpoint and applies
        // this plan's evaluation limit to them.
        let budgets = crate::workspace::limit(
            scioxide_core::SolveBudgets::unbounded(),
            BudgetKind::Evaluation,
            self.options.max_evaluations(),
        );
        let identity = crate::workspace::identity(self.algorithm_id(), state.width);
        let token = state.count as u128;
        let SeriesStorage {
            attempts,
            scratch,
            batch,
            output,
            errors,
            ..
        } = storage;
        let execute = |context: &mut SolveContext<'_, '_>| {
            let mut runner = Runner {
                plan: self,
                state: &mut *state,
                scratch: &mut *scratch,
                output: &mut output[..],
                errors: &mut errors[..],
                context: &mut *context,
                sink: &mut *sink,
            };
            let result = if self.options.execution_policy() == ExecutionPolicy::Sequential
                || matches!(transform, Transform::Euler(_))
            {
                runner.run(terms, transform, quota, false)
            } else {
                runner.prefetch(terms, transform, batch, quota)
            };
            let saved = context.checkpoint(state.count as u128);
            state.checkpoint = saved.as_ref().ok().copied();
            let report = result?;
            let _ = saved?;
            if !report.has_estimate() {
                let stop = match report.termination {
                    SeriesTermination::ContextBudgetLimit(kind) => {
                        crate::IntegrationTermination::BudgetLimit(kind)
                    }
                    SeriesTermination::ExecutionStopped(reason) => {
                        crate::IntegrationTermination::ExecutionStopped(reason)
                    }
                    _ => crate::IntegrationTermination::RefinementLimit,
                };
                return Err(crate::workspace::unpublished(
                    stop,
                    self.algorithm_id(),
                    context.counts(),
                ));
            }
            Ok(report)
        };
        match checkpoint {
            Some(checkpoint) => {
                attempts.resume(scope, identity, token, checkpoint, budgets, execute)
            }
            None => attempts.run(scope, identity, budgets, execute),
        }
    }
}

/// Sums the infinite series Σₙ term(n), starting at first, using Levin's u transform.
///
/// Defaults to 256 terms, transform order 8, and √ε absolute and relative
/// tolerances. Cost is O(8N) per real component for N terms. Success means the
/// heuristic tolerance was met; it does not prove convergence.
///
/// # Errors
///
/// Rejects an overflowing term range before callbacks. Propagates callback,
/// non-finite-value and allocation errors. A term or call limit is a
/// ConvergenceFailure with TruncationLimit; other stops use the mapping in
/// [Termination](crate#termination).
///
/// ```
/// let sum = scioxide_integrate::sum_series(|n| Ok(2.0_f64.powi(-(n as i32))), 0)?;
/// assert!((sum.value() - 2.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. Levin, Int. J. Comput. Math. B3 (1973), 371–388.
pub fn sum_series<T, F>(term: F, first: usize) -> Result<SeriesResult<T>, SciOxideError>
where
    T: NumericScalar,
    F: Fn(usize) -> Result<T, SciOxideError> + Sync,
{
    let options = SeriesOptions::default();
    if first.checked_add(options.max_terms().get()).is_none() {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("first"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "series term range overflows",
        ));
    }
    let plan = SeriesSummation::new(SeriesMethod::LevinU, first, options)?;
    let mut ws = plan.workspace(1)?;
    let mut output = [T::ZERO];
    let terms = |index, row: &mut [T]| {
        row[0] = term(index)?;
        Ok(())
    };
    let report = plan.sum_into(
        SeriesProblem::Terms(&terms),
        &mut output,
        &mut ws,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    require_sum_success(
        report.termination(),
        plan.algorithm_id(),
        ws.last_attempt()
            .map_or_else(scioxide_core::SolveCounts::new, |attempt| attempt.counts()),
    )?;
    Ok(SeriesResult {
        value: output[0],
        error: ws.storage.errors[0],
        report,
    })
}
fn require_sum_success(
    termination: SeriesTermination,
    algorithm: scioxide_core::AlgorithmId,
    counts: scioxide_core::SolveCounts,
) -> Result<(), SciOxideError> {
    if termination.is_success() {
        return Ok(());
    }
    let stop = match termination {
        SeriesTermination::ContextBudgetLimit(kind) => {
            crate::IntegrationTermination::BudgetLimit(kind)
        }
        SeriesTermination::ExecutionStopped(reason) => {
            crate::IntegrationTermination::ExecutionStopped(reason)
        }
        _ => crate::IntegrationTermination::RefinementLimit,
    };
    crate::workspace::require_success(stop, algorithm, counts)
}

#[cfg(test)]
mod one_shot_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use scioxide_core::{ConvergenceReason, SciOxideErrorView, SolveCounts};
    #[test]
    fn successful_series_and_term_limits_follow_the_status_authority() {
        let algorithm = scioxide_core::algorithm_id!("integrate", "sum-test", 1);
        for termination in [
            SeriesTermination::BoundedTailReached,
            SeriesTermination::ExactEmpty,
            SeriesTermination::HeuristicToleranceReached,
        ] {
            assert!(termination.is_success());
            assert!(require_sum_success(termination, algorithm, SolveCounts::new()).is_ok());
        }
        for termination in [SeriesTermination::TermLimit, SeriesTermination::CallLimit] {
            assert!(!termination.is_success());
            assert!(matches!(
                require_sum_success(termination, algorithm, SolveCounts::new())
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::TruncationLimit,
                    ..
                }
            ));
        }
        for termination in [
            SeriesTermination::ContextBudgetLimit(BudgetKind::Evaluation),
            SeriesTermination::ExecutionStopped(scioxide_core::ExecutionStopReason::Cancelled),
        ] {
            assert!(!termination.is_success());
        }
    }
}
