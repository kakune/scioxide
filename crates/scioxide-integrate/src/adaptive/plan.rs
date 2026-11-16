//! Globally adaptive integration plan.

use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Interval, RealScalar, SciOxideError, SolveScope,
};

use super::AdaptiveRule;
use super::kernel::integrate_into;
use super::{AdaptiveWorkspace, SubdivisionOptions};
use crate::report::IntegrationReport;
use crate::workspace::IntegrationRequirements;

/// Globally adaptive quadrature of `∫ₐᵇ f(x) dx` on a finite interval.
///
/// The integrator keeps a set of regions that cover `[a, b]`, each with an
/// integral and an error estimate from the rule selected by [`AdaptiveRule`].
/// Each step bisects the region with the largest error estimate, until the sum
/// of the error estimates meets the tolerance of the [`SubdivisionOptions`].
/// For a value with several real components, each component has its own
/// estimates, and the tolerance norm combines them both to rank regions and to
/// decide convergence. Region estimates use compensated summation; the totals
/// over all regions are summed exactly and rounded once.
///
/// For half-lines, the real line or strong endpoint singularities, use
/// [`DoubleExponentialIntegrator`](crate::DoubleExponentialIntegrator).
///
/// # Error estimate
///
/// For [`AdaptiveRule::GaussKronrod`], the error estimate of a region `R` of
/// length `|R|` is
///
/// ```text
/// E = max(ρ · min(1, (200 |K − G| / ρ)^(3/2)), 50 ε A)
/// ```
///
/// where `K` and `G` are the 15-point Kronrod and 7-point Gauss estimates of
/// `∫_R f`, `ρ` is the Kronrod estimate of `∫_R |f(x) − K/|R|| dx`, `A` is the
/// Kronrod estimate of `∫_R |f(x)| dx`, and `ε` is the machine epsilon of `S`.
/// If `ρ = 0`, the first term is `|K − G|`.
///
/// For [`AdaptiveRule::Simpson`], it is four times the difference between the
/// nine-point composite Boole rule and the five-point Boole rule, and at least
/// `50 ε A`. The first region also samples `f` once at the fraction
/// `(√5 − 1)/2` of `[a, b]`, which no dyadic grid contains. If that sample
/// differs from the degree-7 interpolant of its eight nearest grid samples by
/// more than four times the interpolant's change from degree 5, the error of
/// the region that contains it rises to at least four times the difference
/// times the region length, and all regions are bisected two levels deeper.
/// This keeps integrands that look smooth on every dyadic grid, such as
/// `sin²(2ᵐ π x)`, refining. A Simpson call whose samples are all zero does not
/// report success.
///
/// These are estimates, not bounds; see [Error estimates](crate#error-estimates).
///
/// # Termination
///
/// The [`IntegrationReport`] carries one of these statuses
/// (see [Termination](crate#termination)):
///
/// - [`ToleranceReached`](crate::IntegrationTermination::ToleranceReached): the
///   error estimate meets the tolerance.
/// - [`Stagnation`](crate::IntegrationTermination::Stagnation): the rounding-error
///   floor dominates every region, the regions became too small to bisect, or
///   the bisections stopped making progress. Progress stops after 6 bisections
///   that change the value by at most `10⁻⁵` relative while keeping at least 99 %
///   of the error, or after 20 bisections, past the tenth, that increase the error.
/// - [`BudgetLimit`](crate::IntegrationTermination::BudgetLimit): the
///   subdivision or evaluation limit allows no further bisection.
/// - [`ExecutionStopped`](crate::IntegrationTermination::ExecutionStopped):
///   cancellation or a deadline.
/// - [`ProbablyDivergent`](crate::IntegrationTermination::ProbablyDivergent): only
///   with [`Self::with_extrapolation`].
/// - [`DegenerateDomain`](crate::IntegrationTermination::DegenerateDomain): `a = b`;
///   the result is exactly zero and `f` is not called.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ErrorTolerance, Interval, NormSelection};
/// use scioxide_integrate::{AdaptiveIntegrator, AdaptiveRule, SubdivisionOptions};
///
/// // Absolute tolerances 1e-12 and 1e-8 for the two components, no relative tolerance.
/// let tolerance = ErrorTolerance::try_vector(&[1e-12, 1e-8], 0.0, NormSelection::Infinity)?;
/// let options = SubdivisionOptions::default().with_tolerance(tolerance);
/// let integrator = AdaptiveIntegrator::<f64>::new(AdaptiveRule::GaussKronrod, options)?;
/// let f = |x: f64| Ok([x.sin(), (-x * x).exp()]);
/// let (value, report) = integrator.integrate(&f, Interval::new(0.0, core::f64::consts::PI)?)?;
/// assert!(report.is_success());
/// // ∫₀^π sin x dx = 2
/// assert!((value[0] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer, 1983.
///   <https://doi.org/10.1007/978-3-642-61786-7>
/// - P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
///   Academic Press, 1984.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct AdaptiveIntegrator<S: RealScalar> {
    options: SubdivisionOptions<S>,
    rule: AdaptiveRule,
    extrapolation: bool,
}

impl<S: RealScalar> AdaptiveIntegrator<S> {
    /// Creates an integrator with extrapolation disabled.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `rule` is [`AdaptiveRule::Simpson`]
    ///   and `options` allow fewer than 15 subdivisions.
    /// - `SizeOverflow` if the region storage for `options` cannot be
    ///   represented.
    pub fn new(rule: AdaptiveRule, options: SubdivisionOptions<S>) -> Result<Self, SciOxideError> {
        rule.validate(&options)?;
        Ok(Self {
            options,
            rule,
            extrapolation: false,
        })
    }

    /// Enables or disables extrapolation of the bisection sequence with Wynn's
    /// ε-algorithm.
    ///
    /// For integrands with endpoint or interior singularities the sequence of
    /// bisection totals converges slowly, and extrapolation accelerates it. The
    /// integrator first bisects the larger regions until their error meets the
    /// tolerance, then appends the total to an ε table of at most 50 entries
    /// per component. The error estimate of an extrapolated value is the sum of
    /// its distances to the three preceding extrapolated values, plus the error
    /// of the larger regions. The call succeeds when this estimate meets the
    /// tolerance. It ends with
    /// [`ProbablyDivergent`](crate::IntegrationTermination::ProbablyDivergent)
    /// when the totals keep moving in one direction by similar steps and no
    /// extrapolated value is available.
    ///
    /// Extrapolation is disabled by default and adds no callbacks.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `enabled` is `true` and the rule is
    ///   [`AdaptiveRule::Simpson`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_integrate::{AdaptiveIntegrator, AdaptiveRule, SubdivisionOptions};
    ///
    /// let options = SubdivisionOptions::default();
    /// let plan = AdaptiveIntegrator::<f64>::new(AdaptiveRule::GaussKronrod, options)?
    ///     .with_extrapolation(true)?;
    /// // ∫₀¹ x^(−0.9) dx = 10, with an integrable singularity at x = 0.
    /// let (value, report) = plan.integrate(|x: f64| Ok(x.powf(-0.9)), Interval::new(0.0, 1.0)?)?;
    /// assert!(report.is_success() && (value - 10.0).abs() < 1e-7);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - P. Wynn, "On a device for computing the e_m(S_n) transformation",
    ///   *Math. Tables Aids Comput.* 10, 91–96, 1956.
    /// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
    ///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer,
    ///   1983.
    pub fn with_extrapolation(mut self, enabled: bool) -> Result<Self, SciOxideError> {
        if enabled && self.rule == AdaptiveRule::Simpson {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("extrapolation"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "bisection extrapolation requires the Gauss-Kronrod rule",
            ));
        }
        self.extrapolation = enabled;
        Ok(self)
    }

    /// Returns the rule applied on each region.
    pub const fn rule(&self) -> AdaptiveRule {
        self.rule
    }

    /// Returns whether extrapolation is enabled.
    pub const fn extrapolation_enabled(&self) -> bool {
        self.extrapolation
    }

    /// Returns the options.
    pub const fn options(&self) -> &SubdivisionOptions<S> {
        &self.options
    }

    /// Returns the identifier of the method: `AdaptiveGaussKronrod` or
    /// `AdaptiveSimpson`.
    pub fn algorithm_id(&self) -> AlgorithmId {
        match self.rule {
            AdaptiveRule::GaussKronrod => {
                scioxide_core::algorithm_id!("integrate", "AdaptiveGaussKronrod", 1)
            }
            AdaptiveRule::Simpson => {
                scioxide_core::algorithm_id!("integrate", "AdaptiveSimpson", 1)
            }
        }
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
        self.rule.requirements(&self.options, components)
    }

    /// Creates a workspace for a value with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size cannot be represented.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, components: usize) -> Result<AdaptiveWorkspace<S>, SciOxideError> {
        AdaptiveWorkspace::allocate(&self.workspace_requirements(components)?)
    }

    /// Integrates `f` over `interval` and writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `f` returns a real, complex or vector value with as many real components
    /// as the workspace was created for; see [Integrands](crate#integrands).
    /// Once the first region is evaluated, a limit, cancellation or deadline
    /// returns `Ok` with the estimate so far and an unsuccessful termination.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a vector absolute tolerance does not
    ///   have one entry per workspace component, or if the workspace was sized
    ///   for fewer subdivisions or, for [`AdaptiveRule::Simpson`], for the
    ///   other rule.
    /// - `ConvergenceFailure` if the evaluation limit is smaller than the cost
    ///   of the first region (15 callbacks for Gauss–Kronrod, 10 for Simpson).
    /// - `ExecutionStopped` if cancellation or a deadline stops the call before
    ///   the first region is evaluated.
    /// - `NumericalInstability` (`Breakdown`) if `interval` is too narrow to
    ///   hold distinct quadrature nodes.
    /// - The errors of `f`, and the integrand errors described in
    ///   [Integrands](crate#integrands).
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        interval: Interval<S>,
        out: &mut O,
        workspace: &mut AdaptiveWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        integrate_into::<S, F, V, O>(
            self.rule,
            self.algorithm_id(),
            &self.options,
            self.extrapolation,
            f,
            interval,
            out,
            workspace,
            scope.into(),
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

/// Integrates `∫ₗᵘ f(x) dx` over finite bounds l and u.
///
/// Uses globally adaptive Gauss–Kronrod (G7K15) with Wynn's ε-algorithm
/// extrapolation. Defaults are absolute and relative tolerances √ε in the
/// infinity norm and 50 subdivisions, where ε is the precision of S. Each
/// bisection costs 30 callbacks; storage scales with subdivisions and components.
/// The returned error is an estimate, not a bound.
///
/// # Errors
///
/// Rejects non-finite or reversed bounds, propagates integrand and allocation
/// errors, and returns the [one-shot termination error](crate#termination) when
/// the method does not succeed. Non-finite integrand values are callback errors.
///
/// ```
/// let result = scioxide_integrate::integrate(|x: f64| Ok(x * x), 0.0, 1.0)?;
/// assert!((result.value() - 1.0 / 3.0).abs() < 1e-14);
/// assert!(result.error_estimate() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. Wynn, MTAC 10 (1956), 91–96; R. Piessens et al., QUADPACK (1983).
pub fn integrate<S, F, V>(f: F, lower: S, upper: S) -> Result<crate::Integral<V>, SciOxideError>
where
    S: scioxide_core::RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync,
    V: scioxide_core::AffineValueMut<Scale = S> + Copy + Default,
{
    let interval = scioxide_core::Interval::new(lower, upper)?;
    let plan = AdaptiveIntegrator::new(
        crate::AdaptiveRule::GaussKronrod,
        crate::SubdivisionOptions::default(),
    )?
    .with_extrapolation(true)?;
    let mut value = V::default();
    let mut workspace = plan.workspace(value.dimension()?)?;
    let report = plan.integrate_into(
        &f,
        interval,
        &mut value,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    crate::workspace::require_success(
        report.termination(),
        plan.algorithm_id(),
        workspace
            .last_attempt()
            .map_or_else(scioxide_core::SolveCounts::new, |attempt| attempt.counts()),
    )?;
    Ok(crate::Integral::from_report(value, &report))
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use scioxide_core::{Budget, BudgetKind, SciOxideErrorView};

    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    use super::*;
    use crate::IntegrationTermination;

    fn integrator(options: SubdivisionOptions<f64>) -> AdaptiveIntegrator<f64> {
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options).unwrap()
    }

    #[test]
    fn smooth_and_peaked_integrals_converge() {
        let integrator = integrator(SubdivisionOptions::default());
        let (sine, report) = integrator
            .integrate(
                |x: f64| Ok(x.sin()),
                Interval::new(0.0, core::f64::consts::PI).unwrap(),
            )
            .unwrap();
        assert!((sine - 2.0).abs() < 1.0e-10);
        assert!(report.is_success());

        let peaked = |x: f64| Ok(1.0 / 100.0_f64.mul_add((x - 0.3).powi(2), 1.0));
        let (value, _) = integrator
            .integrate(&peaked, Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        let expected = (7.0_f64.atan() + 3.0_f64.atan()) / 10.0;
        assert!((value - expected).abs() < 1.0e-8);
    }

    #[test]
    fn f32_uses_the_same_scheduler() {
        let integrator = AdaptiveIntegrator::<f32>::new(
            AdaptiveRule::GaussKronrod,
            SubdivisionOptions::default(),
        )
        .unwrap();
        let (value, _) = integrator
            .integrate(
                |x: f32| Ok(x.sin()),
                Interval::new(0.0_f32, core::f32::consts::PI).unwrap(),
            )
            .unwrap();
        assert!((value - 2.0).abs() < 0.01);
    }

    #[test]
    fn split_budget_is_reserved_atomically() {
        let calls = AtomicUsize::new(0);
        let function = |x: f64| {
            calls.fetch_add(1, Relaxed);
            Ok((1000.0 * (x - 0.1)).sin())
        };
        let options =
            SubdivisionOptions::default().with_max_evaluations(Some(Budget::try_from(30).unwrap()));
        let (_, report) = integrator(options)
            .integrate(&function, Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert_eq!(report.evaluations(), 15);
        assert_eq!(report.evaluations(), calls.load(Relaxed));
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(report.error_estimate().unwrap().is_finite());
    }

    #[test]
    fn too_small_initial_budget_publishes_nothing() {
        let options =
            SubdivisionOptions::default().with_max_evaluations(Some(Budget::try_from(10).unwrap()));
        let error = integrator(options)
            .integrate(|x: f64| Ok(x), Interval::new(0.0, 1.0).unwrap())
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
}
