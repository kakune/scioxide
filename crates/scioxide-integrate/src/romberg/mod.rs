//! Romberg integration: Richardson extrapolation of nested trapezoidal rules.

use crate::LevelOptions;

mod kernel;
pub(crate) use kernel::richardson_step;

use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Interval, RealScalar, SciOxideError, SolveBudgets,
    SolveScope,
};

use crate::evaluation::POINT_BLOCK;
use crate::report::{IntegrationErrorComponents, IntegrationReport};
use crate::workspace::{Attempts, Family, IntegrationRequirements, limit, workspace_protocol};

use kernel::{MAX_LEVELS, MIN_CONVERGENCE_LEVEL};

/// Romberg quadrature of `∫ₐᵇ f(x) dx` on a finite interval: Richardson
/// extrapolation of nested trapezoidal rules.
///
/// Level `j` is the trapezoidal rule `T(hⱼ)` with step `hⱼ = (b − a)/2ʲ`. Each
/// level reuses every earlier sample and adds the `2ʲ⁻¹` new midpoints, so with
/// `L` levels the finest grid has `2ᴸ⁻¹ + 1` points. The Romberg tableau
///
/// ```text
/// R(j, 0) = T(hⱼ),   R(j, k) = R(j, k−1) + (R(j, k−1) − R(j−1, k−1)) / (4ᵏ − 1)
/// ```
///
/// eliminates the even powers `h², h⁴, …` of the Euler–Maclaurin error
/// expansion, so the diagonal `R(j, j)` converges rapidly for smooth `f`. The
/// method samples both endpoints, so `f` must be finite there; for endpoint
/// singularities use [`DoubleExponentialIntegrator`](crate::DoubleExponentialIntegrator).
/// [`LevelOptions`] sets the tolerance and the number of levels, between 6 and
/// 30; the default of 15 levels allows up to 16 386 callbacks.
///
/// # Error estimate
///
/// The error estimate of each component is the largest of four times the last
/// changes in the two highest extrapolation orders, the check-point error below,
/// and the rounding-error floor `50 ε A`, where `A` is the trapezoidal estimate
/// of `∫ₐᵇ |f(x)| dx` and `ε` the machine epsilon of `S`.
///
/// After level 4, `f` is sampled once at the fraction `(√5 − 1)/2` of `[a, b]`,
/// which no dyadic grid contains. If that sample differs from the degree-7
/// interpolant of its eight nearest grid samples by more than four times the
/// interpolant's change from degree 5, the error estimate rises to at least four
/// times the difference times `b − a`. This keeps integrands that look smooth
/// on every grid, such as `sin²(2ᵐ π x)`, refining.
///
/// These are estimates, not bounds; see [Error estimates](crate#error-estimates).
///
/// # Termination
///
/// - [`ToleranceReached`](crate::IntegrationTermination::ToleranceReached): from
///   level 5 on (34 callbacks, including the check point), the error estimate
///   meets the tolerance, some sample is nonzero, and the error estimate is at
///   most a quarter of `A`.
/// - [`Stagnation`](crate::IntegrationTermination::Stagnation): the error
///   estimate reached the rounding-error floor without meeting the tolerance,
///   or the grid became finer than the floating-point spacing.
/// - [`RefinementLimit`](crate::IntegrationTermination::RefinementLimit): all
///   levels were used.
/// - [`BudgetLimit`](crate::IntegrationTermination::BudgetLimit) and
///   [`ExecutionStopped`](crate::IntegrationTermination::ExecutionStopped): a
///   limit, cancellation or deadline stopped the next level.
/// - [`DegenerateDomain`](crate::IntegrationTermination::DegenerateDomain): `a = b`;
///   the result is exactly zero and `f` is not called.
///
/// After an unsuccessful stop, each component returns the completed level with
/// the smallest error estimate. Once level 4 is complete, only levels from 4 on
/// are considered. A stop after only the two endpoint samples returns
/// `T(b − a)` without a discretization error estimate.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{LevelOptions, Romberg};
///
/// let romberg = Romberg::<f64>::new(LevelOptions::default())?;
/// let (value, report) = romberg.integrate(|x: f64| Ok(x.exp()), Interval::new(0.0, 1.0)?)?;
/// assert!(report.is_success());
/// // ∫₀¹ eˣ dx = e − 1
/// assert!((value - (1.0_f64.exp() - 1.0)).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - W. Romberg, "Vereinfachte numerische Integration", *Det Kongelige Norske
///   Videnskabers Selskab Forhandlinger* 28, 30–36, 1955.
/// - P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
///   Academic Press, 1984.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Romberg<S: RealScalar> {
    options: LevelOptions<S>,
    factors: kernel::Factors<S>,
}

impl<S: RealScalar> Romberg<S> {
    /// Creates a Romberg integrator and prepares its extrapolation coefficients.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `options` request fewer than 6 or more
    ///   than 30 levels.
    pub fn new(options: LevelOptions<S>) -> Result<Self, SciOxideError> {
        if !(MIN_CONVERGENCE_LEVEL + 1..=MAX_LEVELS).contains(&options.max_levels().get()) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("options.max_levels"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Romberg levels must be in 6..=30",
            ));
        }
        Ok(Self {
            options,
            factors: kernel::Factors::new()?,
        })
    }

    /// Returns the options.
    pub const fn options(&self) -> &LevelOptions<S> {
        &self.options
    }

    /// Returns the identifier of the method, `Romberg`.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "Romberg", 1)
    }

    /// Returns the workspace size for a value with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size cannot be represented.
    pub fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<IntegrationRequirements, SciOxideError> {
        scioxide_core::backend::checked_mul(
            POINT_BLOCK,
            components,
            scioxide_core::quantity!("romberg_rows"),
        )?;
        Ok(IntegrationRequirements::new(
            Family::Romberg,
            components,
            [POINT_BLOCK, 0],
        ))
    }

    /// Creates a workspace for a value with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size cannot be represented.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, components: usize) -> Result<RombergWorkspace<S>, SciOxideError> {
        RombergWorkspace::allocate(&self.workspace_requirements(components)?)
    }

    /// Integrates `f` over `interval` and writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `f` returns a real, complex or vector value with as many real components
    /// as the workspace was created for; see [Integrands](crate#integrands).
    /// Once both endpoints are evaluated, a limit, cancellation or deadline
    /// returns `Ok` with the best estimate so far and an unsuccessful
    /// termination.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a vector absolute tolerance does not
    ///   have one entry per workspace component.
    /// - `ConvergenceFailure` if the evaluation limit is below 2, so the
    ///   endpoints cannot both be evaluated.
    /// - `ExecutionStopped` if cancellation or a deadline stops the call before
    ///   both endpoints are evaluated.
    /// - The errors of `f`, and the integrand errors described in
    ///   [Integrands](crate#integrands).
    #[inline(always)]
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        interval: Interval<S>,
        out: &mut O,
        workspace: &mut RombergWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        let components = workspace.requirements.components();
        crate::execution::preflight(
            &mut workspace.attempts,
            &mut workspace.published_errors,
            out,
            components,
            Some(self.options.tolerance()),
            || Ok(()),
        )?;
        let budgets = limit(
            SolveBudgets::unbounded(),
            scioxide_core::BudgetKind::Evaluation,
            self.options.max_evaluations(),
        );
        let RombergWorkspace {
            attempts,
            state,
            published_errors,
            ..
        } = workspace;
        crate::execution::integrate(
            attempts,
            published_errors,
            scope.into(),
            self.algorithm_id(),
            components,
            budgets,
            {
                #[inline(always)]
                |cx: &mut scioxide_core::SolveContext<'_, '_>| {
                    if components == 1 {
                        state.run::<F, V, O, 1>(&self.options, &self.factors, f, interval, out, cx)
                    } else {
                        state.run::<F, V, O, 0>(&self.options, &self.factors, f, interval, out, cx)
                    }
                }
            },
        )
    }

    /// Integrates `f` over `interval` and returns the value with its report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Creates a workspace for each call and runs on the calling thread with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace`] and [`Self::integrate_into`].
    pub fn integrate<F, V>(
        &self,
        f: F,
        interval: Interval<S>,
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let f = &f;
        crate::execution::owned(
            V::default(),
            |components| self.workspace(components),
            |out, workspace, cx| self.integrate_into(f, interval, out, workspace, cx),
        )
    }
}

/// Reusable storage for [`Romberg`].
///
/// Holds one tableau row per value component and one block of samples, so its
/// size is `O(c)` for `c` value components and does not depend on the number
/// of levels. Create it with [`Romberg::workspace`]; see
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_integrate::{LevelOptions, Romberg};
///
/// let romberg = Romberg::<f64>::new(LevelOptions::default())?;
/// // ∫₀¹ x dx = 1/2 and ∫₀¹ x² dx = 1/3 in one call.
/// let mut workspace = romberg.workspace(2)?;
/// let mut pair = [0.0; 2];
/// let unit = Interval::new(0.0, 1.0)?;
/// let cx = ExecutionContext::sequential_native();
/// let f = |x: f64| Ok([x, x * x]);
/// romberg.integrate_into(&f, unit, &mut pair, &mut workspace, cx)?;
/// assert!((pair[0] - 0.5).abs() < 1e-12 && (pair[1] - 1.0 / 3.0).abs() < 1e-12);
/// let errors = workspace.component_errors().unwrap();
/// assert!(errors[1].total().unwrap() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct RombergWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    state: kernel::State<S>,
    published_errors: bool,
}

impl<S: RealScalar> RombergWorkspace<S> {
    fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let [block, _] = requirements.units();
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            state: kernel::State::allocate(requirements.components(), block)?,
            published_errors: false,
        })
    }

    /// Returns the error estimate of each value component from the last call.
    ///
    /// Returns `None` before the first call, after a call that returned an
    /// error, and after a zero-width interval, which has no error estimate.
    #[must_use]
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        crate::execution::component_errors(self.published_errors, self.state.diagnostics.as_slice())
    }
}

workspace_protocol!(RombergWorkspace, IntegrationReport<S>);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use scioxide_core::SciOxideErrorView;
    use scioxide_core::{Budget, BudgetKind, ErrorTolerance, NormSelection};

    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    use super::*;
    use crate::IntegrationTermination;

    fn exact() -> LevelOptions<f64> {
        LevelOptions::default()
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap())
    }

    fn integrate(
        options: LevelOptions<f64>,
        f: impl Fn(f64) -> Result<f64, SciOxideError> + Sync,
        interval: Interval<f64>,
    ) -> Result<(f64, IntegrationReport<f64>), SciOxideError> {
        Romberg::new(options).unwrap().integrate(&f, interval)
    }

    #[test]
    fn linear_sine_and_exponential_integrals() {
        let unit = Interval::new(0.0, 1.0).unwrap();
        let (value, report) = integrate(LevelOptions::default(), Ok, unit).unwrap();
        assert!((value - 0.5).abs() < 1e-14);
        assert!(report.is_success());
        // 33 grid callbacks and one off-grid witness.
        assert_eq!(report.evaluations(), 34);
        assert_eq!(report.error_estimate(), Some(25.0 * f64::EPSILON));
        let (value, _) = integrate(
            LevelOptions::default(),
            |x| Ok(x.sin()),
            Interval::new(0.0, core::f64::consts::PI).unwrap(),
        )
        .unwrap();
        assert!((value - 2.0).abs() < 1e-10);
        let (value, _) = integrate(LevelOptions::default(), |x| Ok(x.exp()), unit).unwrap();
        assert!((value - (core::f64::consts::E - 1.0)).abs() < 1e-13);
        let (value, _) = Romberg::<f32>::new(LevelOptions::default())
            .unwrap()
            .integrate(
                |x: f32| Ok(x.sin()),
                Interval::new(0.0_f32, core::f32::consts::PI).unwrap(),
            )
            .unwrap();
        assert!((value - 2.0_f32).abs() < 1e-5);
    }

    #[test]
    fn evaluation_budget_stops_before_a_refinement_level() {
        let calls = AtomicUsize::new(0);
        let f = |x: f64| {
            calls.fetch_add(1, Relaxed);
            Ok(x.exp())
        };
        let options = exact().with_max_evaluations(Some(Budget::try_from(4).unwrap()));
        let (_, report) = integrate(options, f, Interval::new(0.0, 1.0).unwrap()).unwrap();
        assert_eq!(report.evaluations(), 3);
        assert_eq!(report.evaluations(), calls.load(Relaxed));
        assert!(report.error_estimate().is_some_and(|error| error > 0.0));
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        let options = exact().with_max_evaluations(Some(Budget::ONE));
        let error = integrate(options, Ok, Interval::new(0.0, 1.0).unwrap()).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }

    #[test]
    fn evaluation_limit_reports_the_diagonal_difference_used_for_selection() {
        let options = exact().with_max_evaluations(Some(Budget::try_from(3).unwrap()));
        let (value, report) =
            integrate(options, |x| Ok(x.exp()), Interval::new(0.0, 1.0).unwrap()).unwrap();
        let initial = 0.5 * (1.0 + 1.0_f64.exp());
        let refined_trapezoid = 0.5 * initial + 0.5 * 0.5_f64.exp();
        let diagonal = refined_trapezoid + (refined_trapezoid - initial) / 3.0;
        let expected_error = 4.0 * (diagonal - initial).abs();
        assert!((value - diagonal).abs() < 1.0e-15);
        assert!((report.error_estimate().unwrap() - expected_error).abs() < 1.0e-15);
        assert_eq!(report.evaluations(), 3);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
    }

    #[test]
    fn collapsed_refinement_grid_reports_stagnation_with_a_real_error() {
        let lower = 1.0_f64;
        let upper = f64::from_bits(lower.to_bits() + 8);
        let options = exact().with_max_levels(Budget::try_from(10).unwrap());
        let (_, report) = integrate(
            options,
            |x| Ok(if x == upper { 1.0 } else { 0.0 }),
            Interval::new(lower, upper).unwrap(),
        )
        .unwrap();
        assert_eq!(report.termination(), IntegrationTermination::Stagnation);
        assert!(report.error_estimate().is_some_and(|error| error > 0.0));
        assert_eq!(report.evaluations(), 9);
    }

    #[test]
    fn callback_failure_and_level_range_are_errors() {
        let calls = AtomicUsize::new(0);
        let error = integrate(
            LevelOptions::default(),
            |_| {
                calls.fetch_add(1, Relaxed);
                Err(crate::kernel::invalid(
                    scioxide_core::argument!("callback"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "intentional callback failure",
                ))
            },
            Interval::new(0.0, 1.0).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(calls.load(Relaxed), 1);
        assert!(
            Romberg::<f64>::new(
                LevelOptions::default().with_max_levels(Budget::try_from(31).unwrap())
            )
            .is_err()
        );
    }
}
