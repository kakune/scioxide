//! Adaptive cubature over a hyperrectangle.
//!
//! Dimension one uses the G7K15 regional scheduler of
//! [`crate::AdaptiveIntegrator`]. Dimensions two and above use the Genz–Malik
//! degree-7/5 rule with the same scheduler.

mod genz_malik;

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, ExecutionContext, Interval, PartialResultTrust,
    RealScalar, SciOxideError, SolveScope, check_value_shape, write_value,
};

use super::breakpoints::AxisBreakpoints;
use crate::SubdivisionOptions;
use crate::adaptive::kernel::{PreparedRules, schedule_into};
use crate::adaptive::regional::{Region, RegionalWorkspace};
use crate::report::{IntegrationErrorComponents, IntegrationReport, IntegrationTermination};
use crate::workspace::{
    Attempts, Family, IntegrationRequirements, Outcome, admit, check_tolerance, workspace_protocol,
};
use genz_malik::{GenzMalik, GenzMalikStorage};

/// Integrand evaluations of one G7K15 region in dimension one.
const KRONROD_NODES: usize = 15;

/// Adaptive cubature over a hyperrectangle `[a₁, b₁] × … × [a_d, b_d]`.
///
/// The plan estimates `I = ∫ f(x) dx` over the box and refines the region with
/// the largest error estimate until the [`SubdivisionOptions`] tolerance holds
/// or a limit is reached. The integrand takes a point `x` of length `d` and may
/// return a real, complex or fixed-size vector value (see
/// [Integrands](crate#integrands)).
///
/// - For `d = 1` the plan applies the Gauss–Kronrod G7K15 pair, with 15
///   evaluations per region and 30 per bisection. Without breakpoints it gives
///   the same values, errors and counts as [`crate::AdaptiveIntegrator`] with the
///   same options.
/// - For `d ≥ 2` it applies the degree-7 Genz–Malik rule with an embedded
///   degree-5 rule. A region costs `n(d) = 2^d + 2d² + 2d + 1` evaluations, for
///   example 17 for `d = 2` and 33 for `d = 3`. A split bisects the region along
///   the axis with the largest fourth difference of `f` and evaluates both
///   halves, at a cost of `2 n(d)`. The `2^d` term dominates for large `d`.
/// - For `d = 0` it evaluates the integrand once at the empty point.
///
/// [`CubaturePlan::with_breakpoints`] cuts axes at given fractions. The
/// initial regions are the tensor product of the axis pieces.
///
/// # Error estimates
///
/// For `d ≥ 2` the error estimate of a region is `4 |Q₇ − Q₅|`, where `Q₇`
/// and `Q₅` are the degree-7 and degree-5 estimates of its integral. It is
/// never below the rounding floor `50 ε V Σ|w₇| max|f|`, where `ε` is the
/// machine epsilon, `V` is the volume of the region, `w₇` are the degree-7
/// weights for unit volume and `max|f|` is the largest sampled magnitude.
/// These estimates have [`crate::ErrorReliability::Heuristic`], and
/// convergence is reported as
/// [`IntegrationTermination::HeuristicToleranceReached`]. When the rounding
/// floor alone exceeds the tolerance, the plan stops with
/// [`IntegrationTermination::Stagnation`] after two consecutive splits change
/// the estimate by no more than the rounding floors of their halves. It also
/// stops with `Stagnation` when a region is too narrow to split with distinct
/// nodes. See [Error estimates](crate#error-estimates).
///
/// # Termination
///
/// The subdivision and evaluation limits apply to each call. All initial
/// regions are measured before the first estimate exists. A limit,
/// cancellation or deadline that stops the plan after that returns `Ok` with
/// [`IntegrationTermination::BudgetLimit`] or
/// [`IntegrationTermination::ExecutionStopped`]. A stop before that is an
/// error (see [`CubaturePlan::integrate_into`]). A box with a zero-width axis
/// returns exactly zero with [`IntegrationTermination::DegenerateDomain`] and
/// no evaluations. See [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ErrorTolerance, Interval, NormSelection};
/// use scioxide_integrate::{CubaturePlan, SubdivisionOptions};
///
/// let tolerance = ErrorTolerance::scalar(1e-10, 1e-10, NormSelection::Infinity)?;
/// let options = SubdivisionOptions::default().with_tolerance(tolerance);
/// let square = [Interval::new(0.0, 1.0)?; 2];
/// let plan = CubaturePlan::new(&square, options)?.with_breakpoints(0, &[0.5])?;
/// let (value, report) = plan.integrate(|x: &[f64]| Ok(x[0] * x[1]))?;
/// assert!(report.is_success());
/// // ∫₀¹∫₀¹ x y dx dy = 1/4
/// assert!((value - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. C. Genz and A. A. Malik, "Remarks on algorithm 006: An adaptive
///   algorithm for numerical integration over an N-dimensional rectangular
///   region", *J. Comput. Appl. Math.* 6(4), 295–302, 1980.
///   <https://doi.org/10.1016/0771-050X(80)90039-X>
/// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer, 1983.
#[derive(Debug, Clone)]
#[must_use]
pub struct CubaturePlan<S: RealScalar> {
    bounds: Vec<Interval<S>>,
    breakpoints: AxisBreakpoints<S>,
    options: SubdivisionOptions<S>,
    rule: Option<GenzMalik<S>>,
}

impl<S: RealScalar> CubaturePlan<S> {
    /// Prepares the box `bounds` with no breakpoints.
    ///
    /// The dimension `d` is `bounds.len()`. For `d ≥ 2` the plan stores the
    /// `n(d)` nodes of the Genz–Malik rule, `n(d)·d` scalars.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `2^d` does not fit in `usize`.
    /// - `SizeOverflow` if `2 n(d)`, the node table or the region count of
    ///   [`CubaturePlan::workspace_requirements`] overflows `usize`.
    /// - `AllocationFailure` if the node table cannot be allocated.
    pub fn new(
        bounds: &[Interval<S>],
        options: SubdivisionOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let mut owned = try_with_capacity(bounds.len(), scioxide_core::quantity!("bounds"))?;
        owned.extend_from_slice(bounds);
        let plan = Self {
            breakpoints: AxisBreakpoints::new(bounds.len())?,
            rule: if bounds.len() >= 2 {
                Some(GenzMalik::new(bounds.len())?)
            } else {
                None
            },
            bounds: owned,
            options,
        };
        let _ = plan.workspace_requirements(0)?;
        Ok(plan)
    }

    /// Replaces the breakpoints of `axis`.
    ///
    /// A breakpoint `t` is a fraction of the axis: it cuts `[a, b]` at
    /// `a + t (b − a)`. Use breakpoints to place known kinks or
    /// discontinuities of the integrand on region boundaries. Each axis with
    /// `k` breakpoints has `k + 1` pieces, and the initial regions are all
    /// combinations of the pieces. An empty slice removes the breakpoints.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the dimension.
    /// - `InvalidInput` (`Inconsistent`) if a fraction is not finite, not in
    ///   the open interval `(0, 1)`, or not strictly greater than the previous
    ///   one.
    /// - `SizeOverflow` if the number of initial regions overflows `usize`.
    /// - `AllocationFailure` if the breakpoints cannot be stored.
    pub fn with_breakpoints(mut self, axis: usize, points: &[S]) -> Result<Self, SciOxideError> {
        self.breakpoints = self.breakpoints.with_axis(axis, points)?;
        let _ = self.workspace_requirements(0)?;
        Ok(self)
    }

    /// Returns the bounds of the box, one interval per axis.
    pub fn bounds(&self) -> &[Interval<S>] {
        &self.bounds
    }

    /// Returns the tolerance, limits and execution policy.
    pub const fn options(&self) -> &SubdivisionOptions<S> {
        &self.options
    }

    /// Returns the identifier of the method, recorded in solve diagnostics.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "Cubature", 1)
    }

    /// Returns the workspace size for an integrand with `components` real
    /// components.
    ///
    /// The workspace holds up to `c + m` regions, where `c` is the number of
    /// initial regions and `m` is the subdivision limit. A complex value counts
    /// as two real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the region count overflows `usize`.
    pub fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<IntegrationRequirements, SciOxideError> {
        let regions = checked_add(
            self.breakpoints.cells()?,
            self.options.max_subdivisions().get(),
            scioxide_core::quantity!("cubature_regions"),
        )?;
        // Staging slots of the one-dimensional scheduler must stay representable.
        checked_add(regions, 2, scioxide_core::quantity!("cubature_regions"))?;
        Ok(
            IntegrationRequirements::new(Family::Cubature, components, [regions, 0])
                .with_dimension(self.bounds.len()),
        )
    }

    /// Allocates a workspace for an integrand with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the region count or storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, components: usize) -> Result<CubatureWorkspace<S>, SciOxideError> {
        CubatureWorkspace::allocate(&self.workspace_requirements(components)?)
    }

    /// Integrates `f` over the box and writes the estimate to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// The number of components of `out` must match the workspace. `scope` is
    /// an [`ExecutionContext`] or a solve scope (see
    /// [Execution contexts](scioxide_core#execution-contexts)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a per-component tolerance has a
    ///   different number of components than the workspace, or if the workspace
    ///   was made for another dimension or for fewer regions.
    /// - `ConvergenceFailure` if the evaluation limit is reached before all
    ///   initial regions are measured.
    /// - `ExecutionStopped` if cancellation or a deadline stops the call before
    ///   all initial regions are measured.
    /// - `NumericalInstability` (`Breakdown`) if rounding collapses a
    ///   breakpoint piece or makes two nodes of an initial region coincide.
    /// - The errors of `f`, `NonFinite` if it returns NaN or infinity, and
    ///   `NumericalInstability` on overflow (see [Integrands](crate#integrands)).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, Interval};
    /// use scioxide_integrate::{CubaturePlan, SubdivisionOptions};
    ///
    /// let square = [Interval::new(0.0, 1.0)?; 2];
    /// let plan = CubaturePlan::new(&square, SubdivisionOptions::default())?;
    /// let mut workspace = plan.workspace(2)?;
    /// let mut pair = [0.0; 2];
    /// let f = |x: &[f64]| Ok([x[0], x[1] * x[1]]);
    /// let cx = ExecutionContext::sequential_native();
    /// plan.integrate_into(&f, &mut pair, &mut workspace, cx)?;
    /// // ∫∫ x dx dy = 1/2 and ∫∫ y² dx dy = 1/3 over the unit square.
    /// assert!((pair[0] - 0.5).abs() < 1e-12 && (pair[1] - 1.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        out: &mut O,
        workspace: &mut CubatureWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        workspace.published_errors = false;
        let components = workspace.requirements.components();
        let preflight = check_value_shape(out, components)
            .and_then(|()| check_tolerance(self.options.tolerance(), components))
            .and_then(|()| {
                let required = self.workspace_requirements(components)?;
                crate::kernel::shape(required.dimension(), workspace.requirements.dimension())?;
                let [regions, _] = required.units();
                let [reserved, _] = workspace.requirements.units();
                if regions > reserved {
                    crate::kernel::shape(regions, reserved)
                } else {
                    Ok(())
                }
            });
        if let Err(error) = preflight {
            return workspace.attempts.reject(error);
        }
        let CubatureWorkspace {
            attempts,
            storage,
            errors,
            published_errors,
            ..
        } = workspace;
        let dimension = self.bounds.len();
        let report = attempts.integrate(
            scope.into(),
            self.algorithm_id(),
            dimension,
            self.options.allowance(),
            |cx| {
                if dimension == 0 {
                    // The rank-zero domain has measure one.
                    if let Some(stop) = admit(cx, 1, false)? {
                        return Ok(Outcome::Unpublished(stop));
                    }
                    let CubatureStorage::Point(row) = storage else {
                        return Err(storage_mismatch());
                    };
                    crate::evaluation::sample_nd(
                        f,
                        &[],
                        0,
                        1,
                        components,
                        row,
                        self.options.execution_policy(),
                        cx,
                    )?;
                    cx.commit_accepted(PartialResultTrust::Admissible)?;
                    write_value(row, out)?;
                    return Ok(Outcome::Published {
                        termination: IntegrationTermination::FixedRuleCompleted,
                        subdivisions: 0,
                        errors: None,
                    });
                }
                if components == 0 || self.bounds.iter().any(|b| b.lower() == b.upper()) {
                    cx.commit_accepted(PartialResultTrust::Admissible)?;
                    out.for_each_component_mut(|value| *value = S::ZERO);
                    return Ok(Outcome::Published {
                        termination: if components == 0 {
                            IntegrationTermination::FixedRuleCompleted
                        } else {
                            IntegrationTermination::DegenerateDomain
                        },
                        subdivisions: 0,
                        errors: None,
                    });
                }
                let cells = self.breakpoints.cells()?;
                let piece = |cell: usize, axis: usize| {
                    self.breakpoints.piece(
                        axis,
                        self.bounds[axis],
                        self.breakpoints.digit(cell, axis),
                    )
                };
                match (storage, &self.rule) {
                    (CubatureStorage::GenzMalik(genz_malik), Some(rule)) => {
                        rule.integrate(cells, piece, &self.options, f, genz_malik, errors, out, cx)
                    }
                    (
                        CubatureStorage::Regional {
                            regional,
                            preparation,
                        },
                        None,
                    ) => {
                        for cell in 0..cells {
                            let Some(interval) = piece(cell, 0)? else {
                                return Ok(Outcome::Unpublished(
                                    IntegrationTermination::GeometryFailure,
                                ));
                            };
                            regional.regions[cell] = Region::new(interval, cell);
                        }
                        schedule_into::<S, _, V, O>(
                            crate::AdaptiveRule::GaussKronrod,
                            &self.options,
                            false,
                            &|x: S| f(core::slice::from_ref(&x)),
                            cells,
                            regional,
                            preparation,
                            &mut [],
                            errors,
                            out,
                            cx,
                        )
                    }
                    _ => Err(storage_mismatch()),
                }
            },
        )?;
        *published_errors = report.error_components().is_some();
        Ok(report)
    }

    /// Integrates a real, complex or fixed-size vector integrand over the box.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Allocates a workspace and runs on
    /// [`ExecutionContext::sequential_native`]. Use
    /// [`CubaturePlan::integrate_into`] to reuse a workspace.
    ///
    /// # Errors
    ///
    /// - The errors of [`CubaturePlan::workspace`] and
    ///   [`CubaturePlan::integrate_into`].
    pub fn integrate<F, V>(&self, f: F) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let f = &f;
        let mut out = V::default();
        let mut workspace = self.workspace(out.dimension()?)?;
        let report = self.integrate_into(
            f,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok((out, report))
    }
}

/// Workspace of a [`CubaturePlan`]: the region list and per-component error
/// estimates.
///
/// A workspace is made for one dimension and one number of value components by
/// [`CubaturePlan::workspace`]. It holds `O(r (d + c))` scalars for `r`
/// regions, `d` dimensions and `c` components, plus the evaluations of one
/// region. Any plan of the same dimension and component count can use it if
/// its regions fit; `check` and `try_reserve` compare and grow the region
/// capacity (see [Plans and workspaces](scioxide_core#plans-and-workspaces)).
///
/// # Examples
///
/// ```
/// use scioxide_core::{Capacity, Interval};
/// use scioxide_integrate::{CubaturePlan, SubdivisionOptions};
///
/// let square = [Interval::new(0.0, 1.0)?; 2];
/// let coarse = CubaturePlan::<f64>::new(&square, SubdivisionOptions::default())?;
/// let split = coarse.clone().with_breakpoints(1, &[0.25, 0.5])?;
/// let mut workspace = coarse.workspace(1)?;
/// // The split plan starts with three regions instead of one.
/// let requirements = split.workspace_requirements(1)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct CubatureWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    storage: CubatureStorage<S>,
    errors: Vec<IntegrationErrorComponents<S>>,
    published_errors: bool,
}

impl<S: RealScalar> core::fmt::Debug for CubatureWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CubatureWorkspace")
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar> CubatureWorkspace<S> {
    fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let components = requirements.components();
        let dimension = requirements.dimension();
        let [regions, _] = requirements.units();
        let storage = match dimension {
            0 => CubatureStorage::Point(try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("cubature_point"),
            )?),
            1 => CubatureStorage::Regional {
                regional: RegionalWorkspace::new(components, regions, KRONROD_NODES)?,
                preparation: PreparedRules::new()?,
            },
            _ => CubatureStorage::GenzMalik(GenzMalikStorage::new(dimension, components, regions)?),
        };
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            storage,
            errors: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("cubature_errors"),
            )?,
            published_errors: false,
        })
    }

    /// Returns the error estimate of each real component from the last
    /// successful call.
    ///
    /// Returns `None` if no call has succeeded since the last error, or if the
    /// last call made no error estimate (a degenerate box or `d = 0`).
    #[must_use]
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.published_errors.then_some(self.errors.as_slice())
    }
}

workspace_protocol!(CubatureWorkspace, IntegrationReport<S>);

enum CubatureStorage<S: RealScalar> {
    Point(Vec<S>),
    Regional {
        regional: RegionalWorkspace<S>,
        preparation: PreparedRules<S>,
    },
    GenzMalik(GenzMalikStorage<S>),
}
fn storage_mismatch() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "cubature storage does not match its prepared dimension",
    )
}

/// Integrates f over the box whose lower and upper corners are given.
///
/// Uses adaptive G7K15 in one dimension and Genz–Malik degree 7/5 rules in
/// higher dimensions, with √ε tolerances and at most 50 subdivisions. A rule
/// costs 2ᵈ + 2d² + 2d + 1 callbacks for d ≥ 2. High-dimensional smooth
/// functions can exhaust the default budget; the heuristic estimate is not a bound.
///
/// # Errors
///
/// Rejects mismatched corner lengths, invalid bounds and dimensions at least
/// usize::BITS. Propagates callback and allocation errors and maps unsuccessful
/// terminations as described in [Termination](crate#termination).
///
/// ```
/// let result = scioxide_integrate::integrate_box(
///     |x: &[f64]| Ok(x[0] * x[1]), &[0.0, 0.0], &[1.0, 1.0],
/// )?;
/// assert!((result.value() - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. C. Genz and A. A. Malik, J. Comput. Appl. Math. 6 (1980), 295–302.
pub fn integrate_box<S, F, V>(
    f: F,
    lower: &[S],
    upper: &[S],
) -> Result<crate::Integral<V>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
    V: AffineValueMut<Scale = S> + Copy + Default,
{
    crate::kernel::shape(lower.len(), upper.len())?;
    if lower.len() >= usize::BITS as usize {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("lower"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "cubature dimension exceeds corner-count capacity",
        ));
    }
    let mut bounds =
        scioxide_core::backend::try_with_capacity(lower.len(), scioxide_core::quantity!("bounds"))?;
    for (&a, &b) in lower.iter().zip(upper) {
        bounds.push(scioxide_core::Interval::new(a, b)?);
    }
    let plan = CubaturePlan::new(&bounds, crate::SubdivisionOptions::default())?;
    let mut value = V::default();
    let mut workspace = plan.workspace(value.dimension()?)?;
    let report = plan.integrate_into(
        &f,
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
