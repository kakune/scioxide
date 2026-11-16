//! Iterated adaptive quadrature over domains with variable limits.
use super::sampling::{BatchSampler, ValueSampler};
use crate::adaptive::regional::{
    self, OptionAllowance, Preparation, Prepared, Refinement, Region, RegionalKernel,
    RegionalPolicy, RegionalWorkspace,
};
use crate::geometry::IntervalGeometry;
use crate::kernel::{AffineMap1D, checked_callback_value, checked_intermediate};
use crate::{
    ErrorEstimate, ErrorReliability, IntegrationErrorComponents, IntegrationReport,
    IntegrationTermination, RegionalProgress, SubdivisionOptions,
};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::{RealValueSum, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, ErrorTolerance, EvaluationRole, ExecutionContext, Interval,
    PartialResultTrust, RealScalar, SciOxideError, SolveContext, SolveScope, check_value_shape,
    write_value,
};
// Each axis retains a recursive orchestration frame. Bound depth before allocation
// rather than allowing an extreme caller dimension to overflow the call stack.
const MAX_DIMENSION: usize = 32;
const ALGORITHM: scioxide_core::AlgorithmId =
    scioxide_core::algorithm_id!("integrate", "iterated", 1);

/// Change of variables `x = φ(t)` for [`IteratedDomainPlan`].
///
/// The plan integrates `f(φ(t)) J(t)` over the iterated domain in `t`, where
/// `J` is the value that [`CoordinateMap::map`] returns. For the
/// change-of-variables formula, return `J(t) = |det Dφ(t)|`. The plan uses `J`
/// as given: its error estimates do not include errors of the map or of `J`.
///
/// Closures `Fn(&[S], &mut [S]) -> Result<S, SciOxideError>` implement this
/// trait.
pub trait CoordinateMap<S: RealScalar> {
    /// Writes `φ(t)` for `t = coordinates` to `physical` and returns `J(t)`.
    ///
    /// Both slices have the dimension of the plan. Every coordinate and `J`
    /// must be finite.
    ///
    /// # Errors
    ///
    /// Returns any error of the map, for example where `φ` is undefined. The
    /// integration call fails with that error.
    fn map(&self, coordinates: &[S], physical: &mut [S]) -> Result<S, SciOxideError>;
}
impl<S: RealScalar, F: Fn(&[S], &mut [S]) -> Result<S, SciOxideError>> CoordinateMap<S> for F {
    fn map(&self, coordinates: &[S], physical: &mut [S]) -> Result<S, SciOxideError> {
        self(coordinates, physical)
    }
}
/// The identity map `φ(t) = t` with `J = 1`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityMap;
impl<S: RealScalar> CoordinateMap<S> for IdentityMap {
    fn map(&self, coordinates: &[S], physical: &mut [S]) -> Result<S, SciOxideError> {
        physical.copy_from_slice(coordinates);
        Ok(S::ONE)
    }
}

/// Options of [`IteratedDomainPlan`]: the [`SubdivisionOptions`] of each axis
/// and a limit on rule applications.
///
/// The tolerance applies to the outer integral. The subdivision and evaluation
/// limits count over all axes of a call. By default there is no limit on rule
/// applications.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::{IteratedDomainPlan, IteratedOptions, SubdivisionOptions};
///
/// let max_iterations = Some(Budget::try_from(100)?);
/// let options = IteratedOptions::<f64>::new(SubdivisionOptions::default())
///     .with_max_iterations(max_iterations);
/// let plan = IteratedDomainPlan::new(2, options)?;
/// assert_eq!(plan.options().max_iterations().map(Budget::get), Some(100));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct IteratedOptions<S: RealScalar> {
    subdivision: SubdivisionOptions<S>,
    max_iterations: Option<scioxide_core::Budget>,
}
impl<S: RealScalar> IteratedOptions<S> {
    /// Creates options with the given subdivision options and no limit on rule
    /// applications.
    pub fn new(subdivision: SubdivisionOptions<S>) -> Self {
        Self {
            subdivision,
            max_iterations: None,
        }
    }
    /// Replaces the subdivision options.
    pub fn with_subdivision_options(mut self, options: SubdivisionOptions<S>) -> Self {
        self.subdivision = options;
        self
    }
    /// Sets or removes the limit on G7K15 rule applications per call.
    ///
    /// Applications on every axis count, including those on outer axes, whose
    /// nodes are inner integrals rather than integrand evaluations.
    pub const fn with_max_iterations(mut self, limit: Option<scioxide_core::Budget>) -> Self {
        self.max_iterations = limit;
        self
    }
    /// Returns the subdivision options.
    pub const fn subdivision_options(&self) -> &SubdivisionOptions<S> {
        &self.subdivision
    }
    /// Returns the limit on rule applications, if any.
    pub const fn max_iterations(&self) -> Option<scioxide_core::Budget> {
        self.max_iterations
    }
    fn allowance(&self) -> scioxide_core::SolveBudgets {
        crate::workspace::limit(
            self.subdivision.allowance(),
            scioxide_core::BudgetKind::Iteration,
            self.max_iterations,
        )
    }
}
impl<S: RealScalar> Default for IteratedOptions<S> {
    fn default() -> Self {
        Self::new(SubdivisionOptions::default())
    }
}

/// Iterated adaptive Gauss–Kronrod quadrature over a domain with variable
/// limits.
///
/// Construction rounds and stores one 15-point Gauss–Kronrod table in O(1) work, shared by all axes.
///
/// The plan computes
///
/// ```text
/// I = ∫_{a₀}^{b₀} ∫_{a₁(t₀)}^{b₁(t₀)} … ∫_{a_{d−1}(t′)}^{b_{d−1}(t′)} f(φ(t)) J(t) dt_{d−1} … dt₀,
/// ```
///
/// where `t′ = (t₀, …, t_{d−2})`. The bounds callback returns the finite
/// interval `[a_k, b_k]` of axis `k` given the first `k` coordinates
/// `t₀, …, t_{k−1}`. `φ` and `J` come from a [`CoordinateMap`]
/// ([`IdentityMap`] by default). Axis 0 is outermost.
///
/// Each axis is integrated by adaptive G7K15 quadrature, and each node of an
/// outer rule is itself an inner integral. The cost is therefore at least
/// `15^d` evaluations, so the plan suits low dimensions and domains that
/// [`crate::CubaturePlan`] cannot describe. Breakpoints are fractions of the
/// current interval of an axis.
///
/// Each outer region passes a share of its tolerance to the inner integrals at
/// its nodes, starting at one half and spread uniformly over the axis. When the
/// inner errors dominate a region, the share is halved and the inner integrals
/// are recomputed. The error estimate of the result includes the inner errors
/// (see [Error estimates](crate#error-estimates)).
///
/// # Termination
///
/// An inner integral that stagnates or reaches a refinement limit passes its
/// estimate and error on to the outer rule. A region whose error is within
/// twice its rounding floor, or is dominated by such an inner integral, is not
/// refined further. When no region can be refined, the call reports
/// [`IntegrationTermination::Stagnation`]. An evaluation or iteration limit,
/// cancellation or a deadline inside an inner integral ends the call with an
/// error. If the outer interval has zero width, the result is zero with
/// [`IntegrationTermination::DegenerateDomain`]. See
/// [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{IteratedDomainPlan, IteratedOptions};
///
/// // The triangle 0 ≤ y ≤ x ≤ 1: the limits of y depend on x.
/// let bounds = |x: &[f64]| match x.len() {
///     0 => Interval::new(0.0, 1.0),
///     _ => Interval::new(0.0, x[0]),
/// };
/// let plan = IteratedDomainPlan::<f64>::new(2, IteratedOptions::default())?;
/// let (value, report) = plan.integrate(&|x: &[f64]| Ok(x[1]), &bounds)?;
/// assert!(report.is_success());
/// // ∫₀¹ ∫₀ˣ y dy dx = 1/6
/// assert!((value - 1.0 / 6.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd
///   ed., Academic Press, 1984.
/// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer, 1983.
#[derive(Debug, Clone)]
pub struct IteratedDomainPlan<S: RealScalar> {
    rule: crate::GaussKronrod<S>,
    cuts: super::breakpoints::AxisBreakpoints<S>,
    options: IteratedOptions<S>,
}
impl<S: RealScalar> IteratedDomainPlan<S> {
    /// Prepares a plan for `dimension` axes, at most 32.
    ///
    /// For `dimension = 0` the plan evaluates `f(φ()) J()` once.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `dimension > 32`.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(dimension: usize, options: IteratedOptions<S>) -> Result<Self, SciOxideError> {
        if dimension > MAX_DIMENSION {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "iterated dimension exceeds bounded orchestration depth (32)",
            ));
        }
        Ok(Self {
            rule: crate::GaussKronrod::new(crate::GaussKronrodRule::G7K15)?,
            cuts: super::breakpoints::AxisBreakpoints::new(dimension)?,
            options,
        })
    }
    /// Returns the number of axes `d`.
    pub fn dimension(&self) -> usize {
        self.cuts.dimension()
    }
    /// Replaces the breakpoints of `axis`.
    ///
    /// A breakpoint `s` cuts the current interval `[a, b]` of the axis at
    /// `a + s (b − a)`. An empty slice removes the breakpoints.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the dimension.
    /// - `InvalidInput` (`Inconsistent`) if a fraction is not finite, not in
    ///   the open interval `(0, 1)`, or not strictly greater than the previous
    ///   one.
    /// - `AllocationFailure` if the breakpoints cannot be stored.
    pub fn with_breakpoints(mut self, axis: usize, points: &[S]) -> Result<Self, SciOxideError> {
        self.cuts = self.cuts.with_axis(axis, points)?;
        Ok(self)
    }
    /// Returns the workspace size for an integrand with `components` real
    /// components.
    ///
    /// Each axis holds up to `p + m` regions, where `p` is the largest number of
    /// breakpoint pieces of any axis and `m` is the subdivision limit.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the region count overflows `usize`.
    pub fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<crate::IntegrationRequirements, SciOxideError> {
        let regions = scioxide_core::backend::checked_add(
            self.cuts.max_pieces(),
            self.options.subdivision.max_subdivisions().get(),
            scioxide_core::quantity!("iterated_regions"),
        )?;
        Ok(crate::IntegrationRequirements::new(
            crate::workspace::Family::Iterated,
            components,
            [regions, 0],
        )
        .with_dimension(self.dimension()))
    }
    /// Allocates a workspace for an integrand with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, components: usize) -> Result<IteratedWorkspace<S>, SciOxideError> {
        IteratedWorkspace::allocate(&self.workspace_requirements(components)?)
    }
    /// Returns the options.
    pub const fn options(&self) -> &IteratedOptions<S> {
        &self.options
    }
    /// Returns the identifier of the method, recorded in solve diagnostics.
    pub const fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        ALGORITHM
    }
    /// Integrates `function` over the domain given by `bounds` and `map` and
    /// writes the estimate to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `bounds(t)` returns the interval of axis `k = t.len()` given the outer
    /// coordinates `t = (t₀, …, t_{k−1})`. `scope` is an [`ExecutionContext`] or a solve scope (see
    /// [Execution contexts](scioxide_core#execution-contexts)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a per-component tolerance has a
    ///   different number of components than the workspace.
    /// - `InvalidInput` (`Inconsistent`) if the workspace was made for another
    ///   dimension or for fewer regions, or if `bounds` returns an interval
    ///   with an infinite end.
    /// - The errors of `bounds`, `map` and `function`, and `NonFinite` if `map`
    ///   or `function` returns NaN or infinity (see
    ///   [Integrands](crate#integrands)).
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   a deadline stops the call before the outer integral has an estimate,
    ///   or inside an inner integral.
    /// - `NumericalInstability` (`Breakdown`) if rounding collapses a
    ///   breakpoint piece.
    pub fn integrate_into<'scope, 'capability, B, M, F, V, O>(
        &self,
        function: F,
        bounds: B,
        map: &M,
        out: &mut O,
        ws: &mut IteratedWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        B: Fn(&[S]) -> Result<Interval<S>, SciOxideError>,
        M: CoordinateMap<S> + ?Sized,
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let bounds = &bounds;
        let function = &function;
        ws.published = false;
        if let Err(error) = check_value_shape(out, ws.requirements.components())
            .and_then(|()| {
                crate::workspace::check_tolerance(
                    self.options.subdivision.tolerance(),
                    ws.requirements.components(),
                )
            })
            .and_then(|()| {
                if ws.check(&self.workspace_requirements(ws.requirements.components())?)?
                    == scioxide_core::Capacity::NeedsGrowth
                {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("workspace"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "insufficient iterated storage",
                    ));
                }
                Ok(())
            })
        {
            return ws.attempts.reject(error);
        }
        let mut attempts = core::mem::take(&mut ws.attempts);
        let mut diagnostics = core::mem::take(&mut ws.diagnostics);
        let result = attempts.run(
            scope.into(),
            crate::workspace::identity(ALGORITHM, self.dimension()),
            self.options.allowance(),
            |cx| {
                let progress = self.run(
                    bounds,
                    map,
                    &mut ValueSampler {
                        function,
                        policy: self.options.subdivision.execution_policy(),
                    },
                    self.options.subdivision.tolerance(),
                    out,
                    &mut diagnostics,
                    ws,
                    cx,
                )?;
                if !progress.measured {
                    return Err(crate::workspace::unpublished(
                        progress.termination,
                        ALGORITHM,
                        cx.counts(),
                    ));
                }
                Ok(IntegrationReport::new(
                    progress.termination,
                    progress.num_evals,
                    progress.subdivisions,
                    IntegrationErrorComponents::envelope(diagnostics.iter()),
                ))
            },
        );
        ws.attempts = attempts;
        ws.diagnostics = diagnostics;
        ws.published = result.is_ok();
        result
    }
    /// Integrates a real, complex or fixed-size vector integrand over the domain
    /// given by `bounds`, with the identity map.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Allocates a workspace and runs on
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`IteratedDomainPlan::workspace`] and
    ///   [`IteratedDomainPlan::integrate_into`].
    pub fn integrate<B, F, V>(
        &self,
        function: F,
        bounds: B,
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        B: Fn(&[S]) -> Result<Interval<S>, SciOxideError>,
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let bounds = &bounds;
        let function = &function;
        let mut out = V::default();
        let mut ws = self.workspace(out.dimension()?)?;
        let report = self.integrate_into(
            function,
            bounds,
            &IdentityMap,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok((out, report))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run<B, M, A, O>(
        &self,
        bounds: &B,
        map: &M,
        sampler: &mut A,
        tolerance: &ErrorTolerance<S>,
        output: &mut O,
        errors: &mut [IntegrationErrorComponents<S>],
        ws: &mut IteratedWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<RegionalProgress, SciOxideError>
    where
        B: Fn(&[S]) -> Result<Interval<S>, SciOxideError> + ?Sized,
        M: CoordinateMap<S> + ?Sized,
        A: BatchSampler<S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let c = ws.requirements.components();
        check_value_shape(output, c)?;
        for (mismatch, argument) in [
            (errors.len() != c, scioxide_core::argument!("errors")),
            (
                ws.levels.len() != self.dimension(),
                scioxide_core::argument!("ws"),
            ),
        ] {
            if mismatch {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "workspace or diagnostics shape mismatch",
                ));
            }
        }
        if tolerance
            .absolute()
            .as_vector()
            .is_some_and(|a| a.len() != c)
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("tolerance.absolute"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "component tolerance shape mismatch",
            ));
        }
        for (axis, level) in ws.levels.iter().enumerate() {
            let required = scioxide_core::backend::checked_add(
                self.cuts.pieces(axis),
                self.options.subdivision.max_subdivisions().get(),
                scioxide_core::quantity!("level_regions"),
            )?;
            if level.regional.capacity < required {
                crate::kernel::shape(required, level.regional.capacity)?;
            }
        }
        for (i, target) in ws.targets.iter_mut().enumerate() {
            *target = tolerance.absolute().component(i, c)?;
        }
        let initial_evals = context.counts().own().provider(EvaluationRole::Function);
        let mut report = RegionalProgress {
            termination: IntegrationTermination::ToleranceReached,
            measured: false,
            num_evals: 0,
            subdivisions: 0,
        };
        let mut state = Run {
            plan: self,
            bounds,
            map,
            sampler,
            initial_evals,
            tolerance,
            coordinates: &mut ws.coordinates,
            physical: &mut ws.physical,
            jacobians: &mut ws.jacobians,
            leaf: &mut ws.leaf,
            components: c,
        };
        if let Some(stop) = context.stop_reason()? {
            report.termination = IntegrationTermination::ExecutionStopped(stop);
            return Ok(report);
        }
        if self.dimension() == 0 {
            if let Some(stop) = crate::workspace::admit(context, 1, false)? {
                report.termination = stop;
                return Ok(report);
            }
            ws.values.fill(S::NAN);
            let jacobian = checked_callback_value(map.map(&[], &mut [])?)?;
            state.sampler.sample(&[], &mut ws.values, 0, c, context)?;
            for (v, e) in ws.values.iter_mut().zip(&mut ws.staged_diagnostics) {
                *v = Scaled::from_value(*v)?
                    .mul(Scaled::from_value(jacobian)?)?
                    .round()?;
                *e = IntegrationErrorComponents::discretization_only(S::ZERO);
            }
            report.termination = IntegrationTermination::FixedRuleCompleted;
        } else if c == 0 {
            report.termination = IntegrationTermination::FixedRuleCompleted;
        } else {
            let mut allowance = OptionAllowance::new(report, &self.options.subdivision);
            let complete = state.axis(
                0,
                &mut ws.levels,
                &ws.targets,
                tolerance.relative(),
                context,
                &mut allowance,
            )?;
            report = allowance.report;
            report.num_evals =
                context.counts().own().provider(EvaluationRole::Function) - initial_evals;
            if complete.is_none() {
                return Ok(report);
            }
            let root = &ws.levels[0].regional;
            ws.values.copy_from_slice(&root.values);
            for (e, parts) in ws.staged_diagnostics.iter_mut().zip(&root.parts) {
                *e = IntegrationErrorComponents::discretization_only(parts[1]);
                e.roundoff = Some(parts[2]);
                e.inner = Some(ErrorEstimate {
                    magnitude: parts[3],
                    reliability: ErrorReliability::Estimated,
                });
            }
        }
        report.num_evals =
            context.counts().own().provider(EvaluationRole::Function) - initial_evals;
        context.commit_accepted(PartialResultTrust::Admissible)?;
        write_value(&ws.values, output)?;
        errors.copy_from_slice(&ws.staged_diagnostics);
        report.measured = true;
        Ok(report)
    }
}

/// Workspace of an [`IteratedDomainPlan`]: one region list per axis.
///
/// A workspace is made for one dimension and one number of value components by
/// [`IteratedDomainPlan::workspace`]. It holds `O(d r c)` scalars for `d`
/// axes, `r` regions per axis and `c` components. Any plan of the same
/// dimension and component count can use it if its regions fit.
pub struct IteratedWorkspace<S: RealScalar> {
    requirements: crate::IntegrationRequirements,
    attempts: crate::workspace::Attempts<IntegrationReport<S>>,
    published: bool,
    staged_diagnostics: Vec<IntegrationErrorComponents<S>>,
    levels: Vec<Level<S>>,
    coordinates: Vec<S>,
    physical: Vec<S>,
    jacobians: [S; 15],
    leaf: Vec<S>,
    targets: Vec<S>,
    values: Vec<S>,
    diagnostics: Vec<IntegrationErrorComponents<S>>,
}

struct Level<S: RealScalar> {
    regional: RegionalWorkspace<S>,
    nested: Nested<S>,
}
impl<S: RealScalar> Level<S> {
    fn new(c: usize, capacity: usize) -> Result<Self, SciOxideError> {
        let nodes = checked_mul(15, c, scioxide_core::quantity!("inner_uncertainty"))?;
        Ok(Self {
            regional: RegionalWorkspace::new(c, capacity, 15)?,
            nested: Nested {
                uncertainty: try_filled(
                    nodes,
                    S::ZERO,
                    scioxide_core::quantity!("inner_uncertainty"),
                )?,
                floors: try_filled(nodes, S::ZERO, scioxide_core::quantity!("inner_floors"))?,
                targets: try_filled(c, S::ZERO, scioxide_core::quantity!("level_targets"))?,
                totals: try_filled(
                    c,
                    RealValueSum::new(),
                    scioxide_core::quantity!("level_totals"),
                )?,
                scales: try_filled(c, S::ZERO, scioxide_core::quantity!("level_scales"))?,
                refining: false,
                limited: try_filled(
                    scioxide_core::backend::checked_add(
                        capacity,
                        2,
                        scioxide_core::quantity!("level_slots"),
                    )?,
                    false,
                    scioxide_core::quantity!("level_slots"),
                )?,
            },
        })
    }
}

/// Kernel state of one axis whose rule nodes are inner integrals.
struct Nested<S: RealScalar> {
    /// Reducible inner error `max(discretization - roundoff, 0) + inner`
    /// and the inner arithmetic floor at each rule node of the measured region.
    uncertainty: Vec<S>,
    floors: Vec<S>,
    targets: Vec<S>,
    /// Exact total of the live regional values; its magnitude at the latest
    /// refinement scales the relative tolerance shared with inner axes.
    totals: Vec<RealValueSum<S>>,
    scales: Vec<S>,
    refining: bool,
    /// Regions with an inner solve that stopped short of its target with
    /// reducible error left; a tighter target would repeat it.
    limited: Vec<bool>,
}
impl<S: RealScalar> Nested<S> {
    fn reset(&mut self) {
        self.totals.iter_mut().for_each(RealValueSum::clear);
        self.scales.fill(S::ZERO);
        self.refining = false;
    }
}
struct Run<'a, S: RealScalar, B: ?Sized, M: ?Sized, A> {
    tolerance: &'a ErrorTolerance<S>,
    plan: &'a IteratedDomainPlan<S>,
    bounds: &'a B,
    map: &'a M,
    sampler: &'a mut A,
    initial_evals: usize,
    coordinates: &'a mut [S],
    physical: &'a mut [S],
    jacobians: &'a mut [S; 15],
    leaf: &'a mut [S],
    components: usize,
}
impl<S, B, M, A> Run<'_, S, B, M, A>
where
    S: RealScalar,
    B: Fn(&[S]) -> Result<Interval<S>, SciOxideError> + ?Sized,
    M: CoordinateMap<S> + ?Sized,
    A: BatchSampler<S>,
{
    /// Integrates `axis` to a complete estimate in `levels[0]` and returns its
    /// termination, or `None` after a stop that the outer solve must report.
    ///
    /// Inner axes pass a stagnated or refinement-limited estimate with its
    /// error to the outer rule. Work limits, execution stops and geometry
    /// failures end the whole solve.
    #[allow(clippy::too_many_arguments)]
    fn axis(
        &mut self,
        axis: usize,
        levels: &mut [Level<S>],
        absolute: &[S],
        relative: S,
        context: &mut SolveContext<'_, '_>,
        allowance: &mut OptionAllowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let report = &mut allowance.report;
        if let Some(stop) = context.stop_reason()? {
            report.termination = IntegrationTermination::ExecutionStopped(stop);
            return Ok(None);
        }
        let (level, children) = levels.split_first_mut().ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "missing level storage",
            )
        })?;
        let interval = (self.bounds)(&self.coordinates[..axis])?;
        if !interval.lower().is_finite() || !interval.upper().is_finite() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("bounds"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "variable limits must be finite",
            ));
        }
        let Some(geometry) = IntervalGeometry::prepare(interval)? else {
            level.regional.values.fill(S::ZERO);
            level.regional.parts.fill([S::ZERO; 4]);
            level.regional.errors.fill(S::ZERO);
            level.regional.scales.fill(S::ZERO);
            if axis == 0 {
                report.termination = IntegrationTermination::DegenerateDomain;
                return Ok(Some(IntegrationTermination::DegenerateDomain));
            }
            return Ok(Some(IntegrationTermination::ToleranceReached));
        };
        let initial = self.plan.cuts.pieces(axis);
        for index in 0..initial {
            let Some(piece) = self.plan.cuts.piece(axis, interval, index)? else {
                report.termination = IntegrationTermination::GeometryFailure;
                return Ok(None);
            };
            level.regional.regions[index] = Region::new(piece, index);
        }
        level.nested.reset();
        let tolerance = self.tolerance;
        let norm = |errors: &[S], scales: &[S]| -> Result<S, SciOxideError> {
            if axis == 0 {
                return tolerance.normalized_norm(errors, scales);
            }
            errors.iter().zip(scales).zip(absolute).try_fold(
                S::ZERO,
                |norm, ((&error, &scale), &absolute)| {
                    let threshold = checked_intermediate(absolute + relative * scale)?;
                    let score = if threshold == S::ZERO {
                        if error == S::ZERO { S::ZERO } else { S::MAX }
                    } else {
                        crate::kernel::diagnostic(checked_intermediate(error / threshold))?
                    };
                    Ok(norm.max(score))
                },
            )
        };
        let policy = RegionalPolicy {
            extrapolation: false,
            norm: &norm,
            refine: true,
            offsets: None,
        };
        let mut kernel = AxisKernel {
            run: self,
            axis,
            span: geometry.span()?,
            children,
            absolute,
            relative,
            nested: &mut level.nested,
        };
        let scheduled = regional::schedule(
            initial,
            &mut kernel,
            &policy,
            &mut level.regional,
            context,
            allowance,
        )?;
        let termination = scheduled.termination();
        allowance.report.termination = termination;
        let complete = scheduled.is_measured()
            && (axis == 0
                || matches!(
                    termination,
                    IntegrationTermination::ToleranceReached
                        | IntegrationTermination::Stagnation
                        | IntegrationTermination::RefinementLimit
                ));
        Ok(complete.then_some(termination))
    }
}
struct AxisKernel<'a, 'b, S: RealScalar, B: ?Sized, M: ?Sized, A> {
    run: &'a mut Run<'b, S, B, M, A>,
    axis: usize,
    /// Extent of the complete axis interval, across which inner targets are shared.
    span: Scaled<S>,
    children: &'a mut [Level<S>],
    absolute: &'a [S],
    relative: S,
    nested: &'a mut Nested<S>,
}
impl<S, B, M, A> RegionalKernel<S> for AxisKernel<'_, '_, S, B, M, A>
where
    S: RealScalar,
    B: Fn(&[S]) -> Result<Interval<S>, SciOxideError> + ?Sized,
    M: CoordinateMap<S> + ?Sized,
    A: BatchSampler<S>,
{
    fn midpoint(&self, region: Region<S>) -> Result<Option<S>, SciOxideError> {
        region.geometry.midpoint().map(Some)
    }
    type Prepared = [S; 15];
    type Allowance = OptionAllowance;
    fn prepare(&mut self, region: Region<S>) -> Result<Preparation<Self::Prepared>, SciOxideError> {
        let Some(map) = AffineMap1D::from_interval(region.geometry)? else {
            return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
        };
        let rule = self.run.plan.rule.rule();
        let mut points = [S::ZERO; 15];
        let admitted = rule.points(map, &mut points)?;
        if !admitted
            && !crate::kernel::mapped_geometry(points.iter().copied(), region.geometry, false)
        {
            return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
        }
        let count = if self.children.is_empty() { 15 } else { 0 };
        Ok(Preparation::Ready(Prepared {
            rule: points,
            evaluations: count,
            providers: count,
        }))
    }
    fn refine(
        &mut self,
        region: Region<S>,
        slot: usize,
        estimates: &[[S; 4]],
    ) -> Result<Refinement<S, Self::Prepared>, SciOxideError> {
        if self.children.is_empty() {
            return regional::bisect(self, region);
        }
        // Within twice its arithmetic floor (rule roundoff plus inner floors),
        // neither a tighter inner target nor a split reduces the region's error
        // materially. It keeps its contribution while other regions refine.
        if estimates
            .iter()
            .all(|e| e[1].max(e[2]) + e[3] <= S::TWO * e[2])
        {
            return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
        }
        let refinement = if estimates.iter().any(|e| e[3] > e[1].max(e[2])) {
            if self.nested.limited[slot] {
                return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
            }
            let region = Region {
                factor: region.factor * S::HALF,
                ..region
            };
            if region.factor == S::ZERO {
                return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
            }
            match self.prepare(region)? {
                Preparation::Ready(rule) => Refinement::Tighten(region, rule),
                Preparation::Stop(stop) => Refinement::Stop(stop),
            }
        } else {
            regional::bisect(self, region)?
        };
        if !matches!(refinement, Refinement::Stop(_)) {
            // The live total is the complete estimate before this refinement.
            for ((total, scale), parts) in self
                .nested
                .totals
                .iter_mut()
                .zip(&mut self.nested.scales)
                .zip(estimates)
            {
                *scale = total.finish()?.abs();
                total.add_value(-parts[0])?;
            }
            self.nested.refining = true;
        }
        Ok(refinement)
    }
    fn commit(&mut self, source: usize, destination: usize) {
        self.nested.limited[destination] = self.nested.limited[source];
    }
    fn measure(
        &mut self,
        region: Region<S>,
        points: &Self::Prepared,
        slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        allowance: &mut OptionAllowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let c = self.run.components;
        let axis = self.axis;
        if self.children.is_empty() {
            let d = self.run.coordinates.len();
            for (i, &point) in points.iter().enumerate() {
                self.run.coordinates[axis] = point;
                let row = &mut self.run.physical[i * d..(i + 1) * d];
                row.fill(S::NAN);
                self.run.jacobians[i] =
                    checked_callback_value(self.run.map.map(self.run.coordinates, row)?)?;
                for &v in row.iter() {
                    checked_callback_value(v)?;
                }
            }
            self.run
                .sampler
                .sample(self.run.physical, self.run.leaf, d, c, context)?;
            allowance.report.num_evals =
                context.counts().own().provider(EvaluationRole::Function) - self.run.initial_evals;
            for (i, row) in self.run.leaf.chunks_exact(c).enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    samples[i * c + j] = Scaled::from_value(v)?
                        .mul(Scaled::from_value(self.run.jacobians[i])?)?
                        .round()?;
                }
            }
        } else {
            // Inner axes share the factor of this axis threshold uniformly over
            // its interval. Before a complete estimate supplies the relative
            // scale, the relative share applies to each inner value instead.
            let nested = &mut *self.nested;
            for ((target, &absolute), &scale) in nested
                .targets
                .iter_mut()
                .zip(self.absolute)
                .zip(&nested.scales)
            {
                let threshold = checked_intermediate(absolute + self.relative * scale)?;
                *target = Scaled::from_value(threshold)?
                    .mul(Scaled::from_value(region.factor)?)?
                    .div(self.span)?
                    .round()?;
            }
            let relative = if nested.refining {
                S::ZERO
            } else {
                self.relative * region.factor
            };
            let mut limited = false;
            for (i, &point) in points.iter().enumerate() {
                self.run.coordinates[axis] = point;
                let Some(termination) = self.run.axis(
                    axis + 1,
                    self.children,
                    &nested.targets,
                    relative,
                    context,
                    allowance,
                )?
                else {
                    return Ok(Some(allowance.report.termination));
                };
                let inner = &self.children[0].regional;
                samples[i * c..(i + 1) * c].copy_from_slice(&inner.values);
                for (j, &[_, discretization, roundoff, uncertainty]) in
                    inner.parts.iter().enumerate()
                {
                    let reducible = checked_intermediate(
                        (discretization - roundoff).max(S::ZERO) + uncertainty,
                    )?;
                    nested.floors[i * c + j] = roundoff;
                    nested.uncertainty[i * c + j] = reducible;
                    limited |= termination != IntegrationTermination::ToleranceReached
                        && reducible > S::ZERO;
                }
            }
            nested.limited[slot] = limited;
        }
        let map = AffineMap1D::from_interval(region.geometry)?.ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("region"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "prepared region became empty",
            )
        })?;
        let rule = self.run.plan.rule.rule();
        let nested = !self.children.is_empty();
        for (component, out) in estimates.iter_mut().enumerate() {
            let estimate =
                rule.apply_samples::<0>(map.half_weight()?, |i| Ok(samples[i * c + component]))?;
            *out = [
                estimate.value(),
                estimate.error(),
                estimate.roundoff_floor(),
                S::ZERO,
            ];
            if nested {
                // Inner arithmetic floors join this rule's floor; only the
                // reducible inner uncertainty can call for a tighter target.
                let weight = map.half_weight()?;
                let floor = rule
                    .apply_samples::<0>(weight, |i| Ok(self.nested.floors[i * c + component]))?
                    .value();
                out[2] = checked_intermediate(out[2] + floor)?;
                out[3] = rule
                    .apply_samples::<0>(weight, |i| Ok(self.nested.uncertainty[i * c + component]))?
                    .value();
                self.nested.totals[component].add_value(out[0])?;
            }
        }
        Ok(None)
    }
}

impl<S: RealScalar> IteratedWorkspace<S> {
    fn allocate(requirements: &crate::IntegrationRequirements) -> Result<Self, SciOxideError> {
        let components = requirements.components();
        let dimension = requirements.dimension();
        let capacity = requirements.units()[0];
        let mut levels = try_with_capacity(dimension, scioxide_core::quantity!("levels"))?;
        for _ in 0..dimension {
            levels.push(Level::new(components, capacity)?);
        }
        Ok(Self {
            requirements: *requirements,
            attempts: crate::workspace::Attempts::default(),
            published: false,
            levels,
            coordinates: try_filled(dimension, S::ZERO, scioxide_core::quantity!("coordinates"))?,
            physical: try_filled(
                checked_mul(dimension, 15, scioxide_core::quantity!("physical_points"))?,
                S::ZERO,
                scioxide_core::quantity!("physical_points"),
            )?,
            jacobians: [S::ZERO; 15],
            leaf: try_filled(
                checked_mul(components, 15, scioxide_core::quantity!("leaf_samples"))?,
                S::ZERO,
                scioxide_core::quantity!("leaf_samples"),
            )?,
            targets: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("tolerance_targets"),
            )?,
            values: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_values"),
            )?,
            staged_diagnostics: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("staged_diagnostics"),
            )?,
            diagnostics: try_filled(
                components,
                IntegrationErrorComponents::discretization_only(S::ZERO),
                scioxide_core::quantity!("component_diagnostics"),
            )?,
        })
    }

    /// Returns the error estimate of each real component from the last
    /// successful call, or `None` if the last call failed or none was made.
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.published.then_some(&self.diagnostics)
    }
}
crate::workspace::workspace_protocol!(IteratedWorkspace, IntegrationReport<S>);

/// Integrates f over a d-dimensional domain with nested limits.
///
/// The bounds callback receives the outer coordinates and returns the interval
/// of axis outer.len(); axis 0 is outermost. Uses iterated adaptive G7K15 with
/// √ε absolute and relative tolerances and at most 50 subdivisions in total
/// over all axes.
/// Callback cost can grow exponentially with dimension;
/// the returned estimate combines the errors of inner integrals.
///
/// # Errors
///
/// Returns domain, callback and allocation errors from [`IteratedDomainPlan`]
/// and rejects unsuccessful termination as described in [Termination](crate#termination).
///
/// ```
/// use scioxide_core::Interval;
/// let result = scioxide_integrate::integrate_iterated(
///     |_: &[f64]| Ok(1.0),
///     |outer: &[f64]| Interval::new(0.0, outer.first().copied().unwrap_or(1.0)),
///     2,
/// )?;
/// assert!((result.value() - 0.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. Piessens et al., QUADPACK (1983), nested adaptive quadrature.
pub fn integrate_iterated<S, F, B, V>(
    f: F,
    bounds: B,
    dimension: usize,
) -> Result<crate::Integral<V>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
    B: Fn(&[S]) -> Result<Interval<S>, SciOxideError>,
    V: AffineValueMut<Scale = S> + Copy + Default,
{
    let plan = IteratedDomainPlan::new(dimension, IteratedOptions::default())?;
    let mut value = V::default();
    let mut workspace = plan.workspace(value.dimension()?)?;
    let report = plan.integrate_into(
        f,
        bounds,
        &IdentityMap,
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
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
