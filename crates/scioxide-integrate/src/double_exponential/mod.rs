//! Double-exponential quadrature: nested trapezoidal sums of the tanh-sinh,
//! exp-sinh and sinh-sinh transforms, with tail truncation by unit shells in `t`.

use crate::LevelOptions;

mod kernel;
mod transform;

use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, BudgetKind, Interval, RealScalar, SciOxideError,
    SolveBudgets, SolveScope,
};

use crate::evaluation::POINT_BLOCK;
use crate::report::{IntegrationErrorComponents, IntegrationReport};
use crate::workspace::{Attempts, Family, IntegrationRequirements, limit, workspace_protocol};

/// A sample point of [`DoubleExponentialIntegrator`], with its distance to the
/// nearest finite endpoint.
///
/// Near an endpoint, the coordinate `x` can round to the endpoint itself while
/// the distance to it is still representable. The integrand receives that
/// distance separately; evaluate an endpoint singularity with it instead of
/// subtracting `x` from the endpoint.
///
/// ```
/// use scioxide_integrate::{
///     DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint, LevelOptions,
/// };
/// let domain = DoubleExponentialDomain::Finite {
///     interval: scioxide_core::Interval::new(0.0, 1.0)?,
///     singular_points: &[],
/// };
/// let de = DoubleExponentialIntegrator::<f64>::new(domain, LevelOptions::default())?;
/// // ∫₀¹ (1 − x)^(−1/2) dx = 2; near x = 1, take 1 − x from the point.
/// let f = |p: DoubleExponentialPoint<f64>| match p.nearest_endpoint() {
///     Some((1.0, d)) => Ok(1.0 / d.sqrt()),
///     _ => Ok(1.0 / (1.0 - p.coordinate()).sqrt()),
/// };
/// let (value, report) = de.integrate(&f)?;
/// assert!(report.is_success() && (value - 2.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleExponentialPoint<S: RealScalar> {
    coordinate: S,
    endpoint: Option<(S, S)>,
}

impl<S: RealScalar> DoubleExponentialPoint<S> {
    /// Returns the coordinate `x`, rounded; it can equal a finite endpoint.
    #[must_use]
    pub const fn coordinate(self) -> S {
        self.coordinate
    }

    /// Returns the nearest finite endpoint and the positive distance to it,
    /// computed without subtracting the rounded coordinate.
    ///
    /// On a finite domain the endpoints are those of the current segment,
    /// including singular points; the segment midpoint returns the lower
    /// endpoint. On a half-line it is the lower bound. On the real line it is
    /// `None`.
    #[must_use]
    pub const fn nearest_endpoint(self) -> Option<(S, S)> {
        self.endpoint
    }
}

/// Integration domain of a [`DoubleExponentialIntegrator`].
///
/// ```
/// use scioxide_integrate::{
///     DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint, LevelOptions,
/// };
/// let domain = DoubleExponentialDomain::Finite {
///     interval: scioxide_core::Interval::new(0.0, 1.0)?,
///     singular_points: &[0.5],
/// };
/// let de = DoubleExponentialIntegrator::<f64>::new(domain, LevelOptions::default())?;
/// // ∫₀¹ |x − 1/2|^(−1/2) dx = 2√2
/// let f = |p: DoubleExponentialPoint<f64>| match p.nearest_endpoint() {
///     Some((0.5, d)) => Ok(1.0 / d.sqrt()),
///     _ => Ok(1.0 / (p.coordinate() - 0.5).abs().sqrt()),
/// };
/// let (value, report) = de.integrate(f)?;
/// assert!(report.is_success());
/// assert!((value - 2.0 * 2.0_f64.sqrt()).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoubleExponentialDomain<'a, S: RealScalar> {
    /// The interval `[a, b]`, split at the given singular points.
    ///
    /// Each segment between consecutive distinct points gets its own tanh-sinh
    /// transform, whose nodes cluster at both segment ends. Points may be given
    /// in any order; repeated points and points equal to `a` or `b` add no
    /// segment. The integral over a zero-width interval is exactly zero.
    Finite {
        /// The interval `[a, b]`.
        interval: Interval<S>,
        /// Points in `[a, b]` where the integrand may be singular.
        singular_points: &'a [S],
    },
    /// The half-line `[lower, ∞)`, with the exp-sinh transform.
    HalfLine {
        /// The finite lower bound.
        lower: S,
    },
    /// The real line `(−∞, ∞)`, with the sinh-sinh transform.
    RealLine,
}

#[derive(Debug, PartialEq)]
struct PreparedDomain<S: RealScalar> {
    kind: DomainKind<S>,
    points: alloc::vec::Vec<S>,
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum DomainKind<S: RealScalar> {
    Finite(Interval<S>),
    HalfLine(S),
    RealLine,
}
impl<S: RealScalar> PreparedDomain<S> {
    fn new(domain: DoubleExponentialDomain<'_, S>) -> Result<Self, SciOxideError> {
        let mut points = alloc::vec::Vec::new();
        let kind = match domain {
            DoubleExponentialDomain::Finite {
                interval,
                singular_points,
            } => {
                points = scioxide_core::backend::try_with_capacity(
                    singular_points.len(),
                    scioxide_core::quantity!("singular_points"),
                )?;
                for &point in singular_points {
                    scioxide_core::validate_value(&point)?;
                    if point < interval.lower() || point > interval.upper() {
                        return Err(crate::kernel::invalid(
                            scioxide_core::argument!("singular_points"),
                            scioxide_core::InvalidInputReason::OutOfRange,
                            "singular point outside interval",
                        ));
                    }
                    if point > interval.lower() && point < interval.upper() {
                        points.push(point);
                    }
                }
                points.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                points.dedup();
                DomainKind::Finite(interval)
            }
            DoubleExponentialDomain::HalfLine { lower } => {
                scioxide_core::validate_value(&lower)?;
                DomainKind::HalfLine(lower)
            }
            DoubleExponentialDomain::RealLine => DomainKind::RealLine,
        };
        Ok(Self { kind, points })
    }
    fn borrowed(&self) -> DoubleExponentialDomain<'_, S> {
        match self.kind {
            DomainKind::Finite(interval) => DoubleExponentialDomain::Finite {
                interval,
                singular_points: &self.points,
            },
            DomainKind::HalfLine(lower) => DoubleExponentialDomain::HalfLine { lower },
            DomainKind::RealLine => DoubleExponentialDomain::RealLine,
        }
    }
}

/// Double-exponential quadrature for endpoint singularities, half-lines and the
/// real line.
///
/// The substitution `x = φ(t)` maps the domain to `t ∈ (−∞, ∞)`:
///
/// ```text
/// [a, b]    tanh-sinh   x = (a + b)/2 + (b − a)/2 · tanh(π/2 · sinh t)
/// [a, ∞)    exp-sinh    x = a + exp(π/2 · sinh t)
/// (−∞, ∞)   sinh-sinh   x = sinh(π/2 · sinh t)
/// ```
///
/// The transformed integrand `f(φ(t)) φ′(t)` decays double exponentially as
/// `|t| → ∞`, and the trapezoidal sum `h Σₖ f(φ(kh)) φ′(kh)` approximates
/// its integral. For `f` analytic in a neighborhood of the open domain, the
/// error decreases like `exp(−c N / log N)` with the number `N` of points, even
/// with algebraic or logarithmic endpoint singularities. The integrand
/// receives a [`DoubleExponentialPoint`], which also gives the distance to the
/// nearest endpoint. Interior singularities must be listed in
/// [`DoubleExponentialDomain::Finite`], so that they become segment endpoints.
///
/// The first level uses the step `h = 1` in `t`; each further level halves `h`
/// and evaluates only the new odd-index points. [`LevelOptions`] sets the
/// tolerance and the number of levels `L`, between 4 and 15 (default 15, a
/// finest step of `2⁻¹⁴`). The sum is truncated at `|t| ≤ 8` at most, or where
/// the transform saturates. Each segment's tails grow outward in unit shells
/// of `t` until the last three shells decrease; the tail estimate is the last
/// shell's mass divided by `1 − r`, where `r` is the larger of the last two
/// shell ratios. With several segments, the tolerance is shared equally
/// between them.
///
/// The plan prepares the nodes once: at most `8 · 2ᴸ⁻¹ + 1` nodes of four
/// scalars each, at most about 4 MiB for `f64` with 15 levels, plus per-segment data
/// in `O(s · 2ᴸ)` for `s` segments. Clones share these tables. Calls evaluate
/// no transcendental functions of `t`.
///
/// # Error estimate
///
/// The error estimate of each component is
///
/// ```text
/// E = max(4 |Q_h − Q_2h|, 50 ε A) + T
/// ```
///
/// where `Q_h` is the sum at step `h`, `A` the sum of `|f|`, `ε` the machine
/// epsilon of `S` and `T` the tail estimate. `T` is heuristic: the method
/// cannot exclude features beyond the sampled range of `t`. See
/// [Error estimates](crate#error-estimates).
///
/// # Termination
///
/// - [`HeuristicToleranceReached`](crate::IntegrationTermination::HeuristicToleranceReached):
///   from level 3 on, `E` meets the tolerance, every tail is truncated, some
///   sample is nonzero, and `4 |Q_h − Q_2h|` is at most a quarter of `A`. An
///   integrand that is zero at every sample never succeeds, because the method
///   cannot tell it from one whose mass lies beyond the grid.
/// - [`Stagnation`](crate::IntegrationTermination::Stagnation): from level 3
///   on, three successive differences `|Q_h − Q_2h|` above the rounding-error
///   floor each fall by less than a factor of 4, the value stops changing, or
///   the differences reach the floor; also when the grid cannot be refined
///   further or a tail does not decay.
/// - [`RefinementLimit`](crate::IntegrationTermination::RefinementLimit): all
///   levels were used.
/// - [`BudgetLimit`](crate::IntegrationTermination::BudgetLimit) and
///   [`ExecutionStopped`](crate::IntegrationTermination::ExecutionStopped): a
///   limit, cancellation or deadline stopped the next level or tail extension.
/// - [`DegenerateDomain`](crate::IntegrationTermination::DegenerateDomain): a
///   zero-width finite interval; the result is exactly zero and `f` is not
///   called.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{
///     DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint, LevelOptions,
/// };
///
/// let domain = DoubleExponentialDomain::RealLine;
/// let integrator = DoubleExponentialIntegrator::<f64>::new(domain, LevelOptions::default())?;
/// // ∫ e^(−x²) dx over the real line = √π
/// let gauss = |point: DoubleExponentialPoint<f64>| Ok((-point.coordinate().powi(2)).exp());
/// let (value, report) = integrator.integrate(&gauss)?;
/// assert!(report.is_success() && (value - core::f64::consts::PI.sqrt()).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - H. Takahasi and M. Mori, "Double exponential formulas for numerical
///   integration", *Publ. RIMS Kyoto Univ.* 9, 721–741, 1974.
///   <https://doi.org/10.2977/prims/1195192451>
/// - M. Mori and M. Sugihara, "The double-exponential transformation in
///   numerical analysis", *J. Comput. Appl. Math.* 127, 287–296, 2001.
///   <https://doi.org/10.1016/S0377-0427(00)00501-X>
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DoubleExponentialIntegrator<S: RealScalar> {
    options: LevelOptions<S>,
    domain: alloc::sync::Arc<PreparedDomain<S>>,
    basis: alloc::sync::Arc<transform::BasisTable<S>>,
    geometry: alloc::sync::Arc<transform::Geometry<S>>,
}

impl<S: RealScalar> DoubleExponentialIntegrator<S> {
    /// Creates an integrator for `domain` and prepares its nodes and weights.
    ///
    /// Singular points are sorted and duplicates removed. Preparation takes
    /// `O(s · 2ᴸ)` time and memory for `s` segments and `L` levels.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `options` request fewer than 4 or more
    ///   than 15 levels, or if a singular point lies outside the interval.
    /// - `NonFinite` if a singular point or the half-line bound is NaN or
    ///   infinite.
    /// - `AllocationFailure` if the node tables cannot be reserved.
    pub fn new(
        domain: DoubleExponentialDomain<'_, S>,
        options: LevelOptions<S>,
    ) -> Result<Self, SciOxideError> {
        if !(kernel::MIN_CONVERGENCE_LEVEL + 1..=kernel::MAX_LEVELS)
            .contains(&options.max_levels().get())
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("options.max_levels"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "double-exponential levels must be in 4..=15",
            ));
        }
        let domain = alloc::sync::Arc::new(PreparedDomain::new(domain)?);
        let levels = options.max_levels().get();
        let basis = alloc::sync::Arc::new(transform::BasisTable::new(levels, domain.borrowed())?);
        let geometry =
            alloc::sync::Arc::new(transform::Geometry::new(domain.borrowed(), &basis, levels)?);
        Ok(Self {
            options,
            basis,
            domain,
            geometry,
        })
    }

    /// Returns the domain, with the interior singular points sorted and
    /// without duplicates.
    pub fn domain(&self) -> DoubleExponentialDomain<'_, S> {
        self.domain.borrowed()
    }

    /// Returns the options.
    pub const fn options(&self) -> &LevelOptions<S> {
        &self.options
    }

    /// Returns the identifier of the method, `DoubleExponential`.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "DoubleExponential", 1)
    }

    /// Returns the workspace size for a value with `components` real
    /// components and this plan's segments.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size cannot be represented.
    pub fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<IntegrationRequirements, SciOxideError> {
        let segments = scioxide_core::backend::checked_add(
            self.domain.points.len(),
            1,
            scioxide_core::quantity!("segments"),
        )?;
        scioxide_core::backend::checked_mul(
            segments.max(POINT_BLOCK),
            components,
            scioxide_core::quantity!("segment_components"),
        )?;
        Ok(IntegrationRequirements::new(
            Family::DoubleExponential,
            components,
            [segments, POINT_BLOCK],
        ))
    }

    /// Creates a workspace for a value with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size cannot be represented.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        components: usize,
    ) -> Result<DoubleExponentialWorkspace<S>, SciOxideError> {
        DoubleExponentialWorkspace::allocate(&self.workspace_requirements(components)?)
    }

    /// Integrates `f` over the domain and writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `f` returns a real, complex or vector value with as many real components
    /// as the workspace was created for; see [Integrands](crate#integrands).
    /// Once the center of every segment is evaluated, a limit, cancellation or
    /// deadline returns `Ok` with the estimate so far and an unsuccessful
    /// termination.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a vector absolute tolerance does not
    ///   have one entry per workspace component, or if the workspace was sized
    ///   for fewer segments.
    /// - `ConvergenceFailure` if the evaluation limit is smaller than the number
    ///   of segments, so not every segment center can be evaluated.
    /// - `ExecutionStopped` if cancellation or a deadline stops the call before
    ///   every segment center is evaluated.
    /// - `NumericalInstability` (`Breakdown`) if a segment is too narrow for its
    ///   transform to produce an interior node.
    /// - The errors of `f`, and the integrand errors described in
    ///   [Integrands](crate#integrands).
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        out: &mut O,
        workspace: &mut DoubleExponentialWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync,
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
            || workspace.state.stage(self.domain()),
        )?;
        let budgets = limit(
            SolveBudgets::unbounded(),
            BudgetKind::Evaluation,
            self.options.max_evaluations(),
        );
        let DoubleExponentialWorkspace {
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
            |cx| state.run(self, f, out, cx),
        )
    }

    /// Integrates `f` over the domain and returns the value with its report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Creates a workspace for each call and runs on the calling thread with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace`] and [`Self::integrate_into`].
    pub fn integrate<F, V>(&self, f: F) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let f = &f;
        crate::execution::owned(
            V::default(),
            |components| self.workspace(components),
            |out, workspace, cx| self.integrate_into(f, out, workspace, cx),
        )
    }
}

/// Reusable storage for [`DoubleExponentialIntegrator`].
///
/// Holds the running sums and tail shells of every segment and value component,
/// and one block of samples. Its size is `O(c · s)` for `c` value components and
/// `s` segments, and does not depend on the number of levels. Create it with
/// [`DoubleExponentialIntegrator::workspace`]; see
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
///
/// ```
/// use scioxide_integrate::{
///     DoubleExponentialDomain, DoubleExponentialIntegrator, LevelOptions, TailStatus,
/// };
///
/// let domain = DoubleExponentialDomain::HalfLine { lower: 0.0 };
/// let integrator = DoubleExponentialIntegrator::<f64>::new(domain, LevelOptions::default())?;
/// let mut workspace = integrator.workspace(1)?;
/// let mut value = 0.0;
/// let f = |p: scioxide_integrate::DoubleExponentialPoint<f64>| Ok((-p.coordinate()).exp());
/// let cx = scioxide_core::ExecutionContext::sequential_native();
/// integrator.integrate_into(&f, &mut value, &mut workspace, cx)?;
/// // ∫₀^∞ e^(−x) dx = 1
/// assert!((value - 1.0).abs() < 1e-8);
/// let tail = workspace.component_errors().unwrap()[0].tail_diagnostics();
/// assert!(tail.is_some_and(|t| t.status() == TailStatus::HeuristicallyTruncated));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct DoubleExponentialWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    state: kernel::State<S>,
    published_errors: bool,
}

impl<S: RealScalar> DoubleExponentialWorkspace<S> {
    fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let [segments, block] = requirements.units();
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            state: kernel::State::allocate(requirements.components(), segments, block)?,
            published_errors: false,
        })
    }

    /// Returns the error estimate of each value component from the last call,
    /// including the tail diagnostics.
    ///
    /// Returns `None` before the first call, after a call that returned an
    /// error, and after a zero-width interval, which has no error estimate.
    #[must_use]
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        crate::execution::component_errors(self.published_errors, self.state.diagnostics.as_slice())
    }
}

workspace_protocol!(DoubleExponentialWorkspace, IntegrationReport<S>);

/// Integrates f over an unbounded [`ExtendedInterval`](scioxide_core::ExtendedInterval).
///
/// Uses exp-sinh on [a, ∞), reflected exp-sinh on (−∞, b] by x = −t,
/// and sinh-sinh on the real line. The integrand must be finite at every finite
/// point, including a finite endpoint. Defaults are √ε absolute and relative
/// tolerances and 15 levels. Each call prepares up to 8·2¹⁴ + 1 nodes of four
/// scalars; use [`DoubleExponentialIntegrator`] to reuse these tables. The
/// heuristic estimate is not an error bound.
///
/// # Errors
///
/// Rejects empty and bounded domains before callbacks. Propagates preparation,
/// integrand and allocation errors, and converts unsuccessful terminations as
/// described in [Termination](crate#termination).
///
/// ```
/// use scioxide_core::ExtendedInterval;
/// let tail = scioxide_integrate::integrate_unbounded(
///     |x: f64| Ok((-x).exp()), ExtendedInterval::new(Some(0.0), None)?,
/// )?;
/// assert!((tail.value() - 1.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. Takahasi and M. Mori, Publ. RIMS 9 (1974), 721–741.
pub fn integrate_unbounded<S, F, V>(
    f: F,
    domain: scioxide_core::ExtendedInterval<S>,
) -> Result<crate::Integral<V>, SciOxideError>
where
    S: scioxide_core::RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync,
    V: scioxide_core::AffineValueMut<Scale = S> + Copy + Default,
{
    let (transform, reflect) = match domain.bounds() {
        Some((Some(lower), None)) => (DoubleExponentialDomain::HalfLine { lower }, false),
        Some((None, Some(upper))) => (DoubleExponentialDomain::HalfLine { lower: -upper }, true),
        Some((None, None)) => (DoubleExponentialDomain::RealLine, false),
        Some((Some(_), Some(_))) => {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("domain"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "a bounded domain is integrated by integrate",
            ));
        }
        None => {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("domain"),
                scioxide_core::InvalidInputReason::Empty,
                "empty integration domain",
            ));
        }
    };
    let plan = DoubleExponentialIntegrator::new(transform, crate::LevelOptions::default())?;
    let integrand = |point: DoubleExponentialPoint<S>| {
        f(if reflect {
            -point.coordinate()
        } else {
            point.coordinate()
        })
    };
    let mut value = V::default();
    let mut workspace = plan.workspace(value.dimension()?)?;
    let report = plan.integrate_into(
        &integrand,
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
    use scioxide_core::SciOxideErrorView;
    use scioxide_core::{Budget, BudgetKind, ErrorTolerance, NormSelection};

    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    use super::*;

    fn integrator(
        domain: DoubleExponentialDomain<'_, f64>,
        options: LevelOptions<f64>,
    ) -> DoubleExponentialIntegrator<f64> {
        DoubleExponentialIntegrator::new(domain, options).unwrap()
    }

    fn finite(lower: f64, upper: f64) -> DoubleExponentialDomain<'static, f64> {
        DoubleExponentialDomain::Finite {
            interval: Interval::new(lower, upper).unwrap(),
            singular_points: &[],
        }
    }

    fn assert_estimate(
        result: Result<(f64, IntegrationReport<f64>), SciOxideError>,
        expected: f64,
        tolerance: f64,
    ) {
        let (value, report) = result.unwrap();
        assert!(report.is_success(), "{report:?}");
        assert!((value - expected).abs() <= tolerance, "{value} {report:?}");
        assert!(
            report
                .error_estimate()
                .is_some_and(|error| error.is_finite() && error >= 0.0),
            "{report:?}"
        );
    }

    #[test]
    fn finite_singular_and_smooth_integrands() {
        let de = integrator(finite(0.0, 1.0), LevelOptions::default());
        assert_estimate(
            de.integrate(|point: DoubleExponentialPoint<f64>| {
                let x = point.coordinate();
                Ok(1.0 / x.sqrt())
            }),
            2.0,
            2.0e-6,
        );
        assert_estimate(
            de.integrate(|point: DoubleExponentialPoint<f64>| {
                let x = point.coordinate();
                Ok(x.ln())
            }),
            -1.0,
            1e-8,
        );
        assert_estimate(
            de.integrate(|point: DoubleExponentialPoint<f64>| {
                let x = point.coordinate();
                Ok(x * x)
            }),
            1.0 / 3.0,
            1e-8,
        );
        let (value, _) = DoubleExponentialIntegrator::<f32>::new(
            DoubleExponentialDomain::Finite {
                interval: Interval::new(0.0_f32, 1.0).unwrap(),
                singular_points: &[],
            },
            LevelOptions::default(),
        )
        .unwrap()
        .integrate(|point: DoubleExponentialPoint<f32>| {
            let x = point.coordinate();
            Ok(x)
        })
        .unwrap();
        assert!((value - 0.5).abs() < 2.0e-4);
    }

    #[test]
    fn improper_domains() {
        let de = integrator(finite(0.0, 1.0), LevelOptions::default());
        let calls = AtomicUsize::new(0);
        let (_, report) = DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::HalfLine { lower: 0.0 },
            de.options().clone(),
        )
        .unwrap()
        .integrate(|point: DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            calls.fetch_add(1, Relaxed);
            Ok((-x).exp())
        })
        .unwrap();
        assert_eq!(report.evaluations(), calls.load(Relaxed));
        assert!(report.evaluations() <= 2_222, "{report:?}");
        assert_estimate(
            DoubleExponentialIntegrator::new(
                DoubleExponentialDomain::HalfLine { lower: 0.0 },
                de.options().clone(),
            )
            .unwrap()
            .integrate(|point: DoubleExponentialPoint<f64>| {
                let x = point.coordinate();
                Ok(1.0 / (1.0 + x).powi(2))
            }),
            1.0,
            1e-8,
        );
        assert_estimate(
            DoubleExponentialIntegrator::new(
                DoubleExponentialDomain::RealLine,
                de.options().clone(),
            )
            .unwrap()
            .integrate(|point: DoubleExponentialPoint<f64>| {
                let x = point.coordinate();
                Ok((-x * x).exp())
            }),
            core::f64::consts::PI.sqrt(),
            1e-8,
        );
        assert_estimate(
            DoubleExponentialIntegrator::new(
                DoubleExponentialDomain::RealLine,
                de.options().clone(),
            )
            .unwrap()
            .integrate(|point: DoubleExponentialPoint<f64>| {
                let x = point.coordinate();
                Ok(1.0 / (1.0 + x * x))
            }),
            core::f64::consts::PI,
            1e-8,
        );
        let (value, report) = DoubleExponentialIntegrator::<f32>::new(
            DoubleExponentialDomain::HalfLine { lower: 0.0 },
            LevelOptions::default(),
        )
        .unwrap()
        .integrate(|point: DoubleExponentialPoint<f32>| {
            let x = point.coordinate();
            Ok((-x).exp())
        })
        .unwrap();
        assert!(report.is_success(), "{report:?}");
        assert!((value - 1.0).abs() < 3.0e-4);
    }

    #[test]
    fn budget_stops_keep_the_centre_estimate() {
        let exact = LevelOptions::default()
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap());
        let calls = AtomicUsize::new(0);
        let function = |point: DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            calls.fetch_add(1, Relaxed);
            Ok(x)
        };
        let (value, report) = integrator(
            finite(0.0, 1.0),
            exact.with_max_evaluations(Some(Budget::try_from(32).unwrap())),
        )
        .integrate(&function)
        .unwrap();
        assert!(report.evaluations() <= 32);
        assert_eq!(report.evaluations(), calls.load(Relaxed));
        assert_eq!(
            report.termination(),
            crate::IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(value.is_finite());
        assert!(report.error_estimate().is_some_and(f64::is_finite));

        calls.store(0, Relaxed);
        let (value, report) = integrator(
            finite(0.0, 1.0),
            LevelOptions::default().with_max_evaluations(Some(Budget::ONE)),
        )
        .integrate(&function)
        .unwrap();
        assert_eq!((calls.load(Relaxed), report.evaluations()), (1, 1));
        assert_eq!(
            report.termination(),
            crate::IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(value.is_finite());
        assert!(report.error_estimate().is_none());
    }

    #[test]
    fn invalid_levels_and_singular_points_are_rejected() {
        let levels = |n| {
            DoubleExponentialIntegrator::<f64>::new(
                finite(0.0, 1.0),
                LevelOptions::default().with_max_levels(Budget::try_from(n).unwrap()),
            )
        };
        // Success needs a completed level three, so three levels could never converge.
        assert!(levels(3).is_err());
        assert!(levels(15).is_ok());
        assert!(levels(16).is_err());
        let (value, report) = levels(4)
            .unwrap()
            .integrate(|point: DoubleExponentialPoint<f64>| Ok(point.coordinate().exp()))
            .unwrap();
        assert!(report.is_success(), "{report:?}");
        assert!((value - (1.0_f64.exp() - 1.0)).abs() < 1e-8);
        let error = DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::Finite {
                interval: Interval::new(0.0, 1.0).unwrap(),
                singular_points: &[2.0],
            },
            LevelOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
    }
}
