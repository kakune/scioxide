//! One stiff-stage kernel for implicit Runge–Kutta, BDF/NDF, Rosenbrock-W and residual DAEs.
//!
//! - [`jacobian`] acquires Jacobians in the order of a
//!   [`DerivativeStrategy`]: analytic, automatic, then finite differences, on dense,
//!   sparse and Krylov storage alike.
//! - [`kernel`] owns the linearizations, the mass matrix, the effective factor cache and
//!   the stage solves of every implicit ODE method.
//! - [`linear`] owns the dense, sparse and Krylov linear storage, also used by the DAE
//!   integrator for its residual Jacobians.
//! - [`newton`] is the single Newton iteration; [`stage`] drives it for ODE stages.
//!
//! Injected [`StageLinearSolver`] and [`NonlinearStageSolver`] providers replace the
//! factorization or the whole Newton iteration. Radau keeps its real estimator
//! factor; other kernels prepared for injected solvers reserve no factor storage.
pub(crate) mod accuracy;
pub(crate) mod jacobian;
pub(crate) mod kernel;
pub(crate) mod linear;
pub(crate) mod newton;
pub(crate) mod reuse;
pub(crate) mod sparse;
pub(crate) mod stage;

use crate::OdeStageBackend;
use crate::result::StepResult;
use kernel::{CacheStats, KernelShape};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;

/// Jacobian sources of stiff methods: analytic, then automatic, then forward differences.
pub(crate) const DEFAULT_STRATEGY: DerivativeStrategy =
    DerivativeStrategy::AnalyticOrAutomaticOrForward;

/// Selects how often the Newton iteration of implicit stages recomputes the Jacobian.
///
/// [`BackwardEuler`](crate::BackwardEuler),
/// [`ImplicitTrapezoid`](crate::ImplicitTrapezoid), [`GaussRk`](crate::GaussRk) and
/// [`Sdirk`](crate::Sdirk) accept either mode through `with_newton_mode`.
/// [`RadauIia`](crate::RadauIia) always uses simplified Newton. With an injected
/// stage solver the iteration is always simplified. See
/// [Implicit stages](crate#implicit-stages) for the convergence test.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[must_use]
pub enum NewtonMode {
    /// Reuses one Jacobian and its factorization across iterations and steps.
    ///
    /// The Jacobian is recomputed when the time or a state component moves by more
    /// than half of `max(1, |old|, |new|)`, or when the iteration converges slowly
    /// or fails. This is the default.
    #[default]
    Simplified,
    /// Recomputes the Jacobian and refactors the iteration matrix before every
    /// correction.
    ///
    /// It needs fewer iterations than simplified Newton on strongly nonlinear
    /// problems, at the cost of one Jacobian and one factorization per iteration.
    Full,
}

/// Stage backend, derivative source and Newton policy of a stiff method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StageConfig {
    pub(crate) backend: OdeStageBackend,
    pub(crate) strategy: DerivativeStrategy,
    pub(crate) mode: NewtonMode,
}

impl StageConfig {
    pub(crate) const fn new() -> Self {
        Self {
            backend: OdeStageBackend::Dense,
            strategy: DEFAULT_STRATEGY,
            mode: NewtonMode::Simplified,
        }
    }

    /// Injected providers own their corrections, so they run in simplified mode.
    pub(crate) fn mode_for(&self, provided: bool) -> NewtonMode {
        if provided {
            NewtonMode::Simplified
        } else {
            self.mode
        }
    }

    /// Kernel shape for `stages` simultaneous stage blocks.
    pub(crate) fn shape(&self, stages: usize, provided: bool) -> KernelShape<'_> {
        KernelShape {
            backend: &self.backend,
            layout: kernel::StageLayout::Coupled,
            stages,
            blocks: if self.mode_for(provided) == NewtonMode::Full && stages > 1 {
                stages
            } else {
                1
            },
            provided,
        }
    }

    pub(crate) fn radau_shape(&self, provided: bool) -> KernelShape<'_> {
        KernelShape {
            layout: kernel::StageLayout::Radau,
            ..self.shape(3, provided)
        }
    }
}

/// Nonlinear iteration count of one stiff step, independent of epoch accounting.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StageWork {
    pub(crate) nonlinear_iterations: usize,
}

fn add(
    value: &mut usize,
    count: usize,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    *value = scioxide_core::backend::checked_add(*value, count, quantity)?;
    Ok(())
}

fn delta(after: usize, before: usize) -> Result<usize, SciOxideError> {
    after
        .checked_sub(before)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("counter_delta")))
}

impl StageWork {
    pub(crate) fn add_nonlinear_iteration(&mut self, count: usize) -> Result<(), SciOxideError> {
        add(
            &mut self.nonlinear_iterations,
            count,
            scioxide_core::quantity!("implicit_ode_nonlinear_iteration_counter"),
        )
    }

    /// Add another step's nonlinear work, as for nested startup steps.
    pub(crate) fn absorb<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
        &mut self,
        step: &StepResult<'_, S, T>,
    ) -> Result<(), SciOxideError> {
        self.add_nonlinear_iteration(step.num_nonlinear_iters)
    }

    /// Step result carrying these counters and the cache activity between two snapshots.
    pub(crate) fn step_result<
        'a,
        S: RealScalar + NumericScalar<Real = S>,
        T: NumericScalar<Real = S>,
    >(
        self,
        before: CacheStats,
        after: CacheStats,
    ) -> Result<StepResult<'a, S, T>, SciOxideError> {
        let mut result = StepResult::from_checked_parts(None, None);
        self.record(&mut result, before, after)?;
        Ok(result)
    }

    /// Write the controller-relevant counters and the cache hits between two snapshots
    /// into `result`.
    pub(crate) fn record<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
        self,
        result: &mut StepResult<'_, S, T>,
        before: CacheStats,
        after: CacheStats,
    ) -> Result<(), SciOxideError> {
        result.num_nonlinear_iters = self.nonlinear_iterations;
        result.num_jacobian_cache_hits = delta(after.jacobian_hits, before.jacobian_hits)?;
        result.num_factor_cache_hits = delta(after.factor_hits, before.factor_hits)?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn work_counter_overflow_is_typed_and_transactional() {
        let mut work = StageWork {
            nonlinear_iterations: usize::MAX,
        };
        assert!(matches!(
            work.add_nonlinear_iteration(1).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert_eq!(work.nonlinear_iterations, usize::MAX);
    }
}
