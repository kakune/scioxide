//! Derivatives of one-dimensional integrals with respect to parameters.
use crate::kernel::checked_callback_values;
mod provider;
mod sampling;
use crate::adaptive::regional::RegionalPolicy;
use provider::Product;
pub use provider::{SensitivityIntegrand, SensitivitySample};

use crate::adaptive::kronrod::KronrodPartition;
use crate::{
    GaussKronrodRule, IntegrationErrorComponents, IntegrationTermination, RegionalProgress,
    SubdivisionOptions,
};
use alloc::sync::Arc;
use alloc::vec::Vec;
use scioxide_core::RangeAccumulator;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValueMut, BudgetKind, ErrorTolerance, EvaluationRole, ExecutionPolicy, Interval,
    PartialResultTrust, RealScalar, SciOxideError, SolveContext, check_value_shape, write_value,
};

fn norm<S: RealScalar>(
    tolerance: &ErrorTolerance<S>,
    errors: &[S],
    scales: &[S],
) -> Result<S, SciOxideError> {
    crate::kernel::diagnostic(tolerance.normalized_norm(errors, scales))
}
/// Derivative product computed by a [`QuadratureSensitivityPlan`].
///
/// Parameters and values are real coordinates: a complex component counts as
/// its real and imaginary parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivityMode {
    /// Jacobian–vector product `(∂I/∂p) v`, with one entry per value component.
    Jvp,
    /// Vector–Jacobian product `wᵀ (∂I/∂p)`, with one entry per parameter.
    Vjp,
}
/// Partition on which a [`QuadratureSensitivityPlan`] integrates.
///
/// In every case the plan approximates the derivative of the exact integral on
/// that partition; it does not differentiate the refinement decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitivityDiscretization {
    /// One Gauss–Kronrod rule on the whole interval, without a tolerance test.
    Fixed,
    /// Adaptive bisection with the G7K15 rule until the error estimates of the
    /// integral and the derivative meet their tolerances.
    Adaptive,
    /// A partition from an earlier call, reused without subdivision.
    Frozen,
}
/// Error estimates of one component of an integral derivative.
///
/// The estimates assume that the integrand's Jacobian products are exact.
/// They do not include errors of those products, and they are not bounds; see
/// [Error estimates](crate#error-estimates).
#[derive(Debug, Clone, Copy)]
pub struct SensitivityErrorComponents<S: RealScalar> {
    /// Quadrature error estimate of the integral of the Jacobian product.
    quadrature: IntegrationErrorComponents<S>,
    /// Rounding error estimate of the endpoint terms.
    boundary_roundoff: S,
}
impl<S: RealScalar> SensitivityErrorComponents<S> {
    /// Returns the estimated absolute error, the quadrature error plus the
    /// endpoint rounding error, or `None` if the quadrature error was not
    /// estimated or the sum overflows.
    pub fn total(&self) -> Option<S> {
        let total = self.quadrature.total()? + self.boundary_roundoff;
        total.is_finite().then_some(total)
    }
}
/// Termination, tolerance tests and work counts of one call to
/// [`QuadratureSensitivityPlan::jvp_into`] or
/// [`QuadratureSensitivityPlan::vjp_into`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct SensitivityReport {
    /// Why the integration stopped.
    termination: IntegrationTermination,
    /// Integrand evaluations, including endpoint evaluations and failed calls.
    function_evaluations: usize,
    /// Jacobian operator applications, including failed ones.
    derivative_evaluations: usize,
    /// Completed bisections.
    subdivisions: usize,
    /// Whether the integral's error estimate met its tolerance.
    primal_tolerance_met: bool,
    /// Whether the derivative's error estimate, endpoint terms included, met
    /// its tolerance.
    derivative_tolerance_met: bool,
}

/// Computes an integral `I(p) = ∫ₐᵇ f(x, p) dx` over a finite interval and its
/// derivative with respect to the parameters `p`, as a Jacobian–vector or
/// vector–Jacobian product.
///
/// Construction stores the Gauss–Kronrod nodes and weights rounded to S; preparation of this fixed-size table costs O(1).
///
/// The derivative follows from the Leibniz integral rule:
///
/// ```text
/// dI = ∫ₐᵇ J(x, p) dp dx + f(b, p) db − f(a, p) da,
/// ```
///
/// where `J(x, p) = ∂f/∂p` is the `m × P` Jacobian of the integrand with `x`
/// held fixed, `m` is the number of value components and `P` the number of
/// parameters. A JVP ([`jvp_into`](Self::jvp_into)) computes `dI` for a
/// direction `dp = v`, with the endpoint derivatives `da = ∇a·v` and
/// `db = ∇b·v` that you supply. A VJP ([`vjp_into`](Self::vjp_into)) computes
/// `wᵀ ∂I/∂p` for a cotangent `w`, with the endpoint gradients `∇a` and `∇b`.
/// Omit the endpoint terms if the interval does not depend on `p`. The
/// integrand supplies `J` as a linear or adjoint operator through
/// [`SensitivityIntegrand`]; no dense Jacobian is formed.
///
/// The plan differentiates the integral, then approximates both `I` and the
/// integral of `J` with the same Gauss–Kronrod rule and the same nodes. It does
/// not differentiate the choices of an adaptive partition. The
/// [`SensitivityDiscretization`] selects the partition:
///
/// - [`fixed`](Self::fixed) applies one Gauss–Kronrod rule to the whole
///   interval, with no tolerance test.
/// - [`adaptive`](Self::adaptive) bisects subintervals with the adaptive
///   G7K15 rule until the error estimates of both the integral and the
///   derivative meet their tolerances.
/// - [`frozen`](Self::frozen) reuses a partition from an earlier call,
///   without subdivision.
///
/// With `fixed` and `frozen` plans the nodes do not depend on `p`, so the
/// computed integral is a smooth function of `p` for a smooth integrand, and
/// for fixed endpoints the computed derivative is its derivative, up to
/// rounding. This helps optimization and finite-difference checks. An
/// adaptive partition can change between nearby parameter values.
///
/// The interval `[a, b]` is that of the current parameter point; prepare a new
/// plan when it changes. Each node costs one integrand evaluation and one
/// operator application, and each moving endpoint one more evaluation.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// # use scioxide_core::{SciOxideError, LinearOperator};
/// use scioxide_integrate::{
///     GaussKronrodRule, JvpQuery, QuadratureSensitivityPlan, SensitivityOptions,
/// };
/// # use scioxide_integrate::{SensitivityIntegrand, SensitivityMode, SensitivitySample};
/// # struct Scale(f64);
/// # impl LinearOperator<f64> for Scale {
/// #     fn rows(&self) -> usize {
/// #         1
/// #     }
/// #     fn cols(&self) -> usize {
/// #         1
/// #     }
/// #     fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
/// #         Ok(0)
/// #     }
/// #     fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
/// #         y[0] = self.0 * x[0];
/// #         Ok(())
/// #     }
/// # }
/// # struct Linear;
/// # impl SensitivityIntegrand<f64> for Linear {
/// #     type Workspace = ();
/// #     fn parameter_dimension(&self) -> usize {
/// #         1
/// #     }
/// #     fn value_dimension(&self) -> usize {
/// #         1
/// #     }
/// #     fn supports(&self, mode: SensitivityMode) -> bool {
/// #         mode == SensitivityMode::Jvp
/// #     }
/// #     fn evaluate(
/// #         &self,
/// #         x: f64,
/// #         p: &[f64],
/// #         sample: &mut SensitivitySample<'_, f64>,
/// #         _: &mut (),
/// #         _: ExecutionContext<'_>,
/// #     ) -> Result<(), SciOxideError> {
/// #         sample.value_mut()[0] = p[0] * x;
/// #         sample.apply_forward(&Scale(x), &mut [])
/// #     }
/// # }
/// // `Linear` is the integrand f(x, p) = p x of the `SensitivityIntegrand` example.
/// let interval = Interval::new(0.0, 1.0)?;
/// let (rule, options) = (GaussKronrodRule::G7K15, SensitivityOptions::default());
/// let plan = QuadratureSensitivityPlan::fixed(&Linear, interval, rule, options)?;
/// let mut ws = plan.workspace()?;
/// let (mut value, mut derivative) = (0.0, 0.0);
/// let outputs = (&mut value, &mut derivative);
/// let context = ExecutionContext::sequential_native();
/// let query = JvpQuery::new(&[2.0], &[1.0]);
/// plan.jvp_into(&Linear, query, outputs, (&mut ws, &mut [()]), context)?;
/// assert!((value - 1.0).abs() < 1e-12); // I(p) = ∫₀¹ p x dx = p/2 at p = 2
/// assert!((derivative - 0.5).abs() < 1e-12); // dI/dp = ∫₀¹ x dx = 1/2
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
/// Academic Press, 1984.
#[derive(Clone)]
pub struct QuadratureSensitivityPlan<S: RealScalar> {
    quadrature: Arc<KronrodPartition<S>>,
    rule: GaussKronrodRule,
    discretization: SensitivityDiscretization,
    parameters: usize,
    components: usize,
    options: SensitivityOptions<S>,
}
/// Scratch storage of a [`QuadratureSensitivityPlan`], with the error
/// estimates and partition of its last successful call.
///
/// Create it with [`QuadratureSensitivityPlan::workspace`]. Its size grows
/// with the maximum number of subintervals; calls never reallocate it. The
/// integrand's own workspaces are passed to each call separately. After a
/// failed call, [`value_errors`](Self::value_errors),
/// [`derivative_errors`](Self::derivative_errors) and
/// [`QuadratureSensitivityPlan::partition`] are unavailable until the next
/// successful call. See [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct QuadratureSensitivityWorkspace<S: RealScalar> {
    attempts: crate::workspace::Attempts<SensitivityReport>,
    requirements: SensitivityRequirements<S>,
    primal_errors: Vec<IntegrationErrorComponents<S>>,
    product_errors: Vec<SensitivityErrorComponents<S>>,
    regions: crate::adaptive::regional::RegionalWorkspace<S>,
    values: Vec<S>,
    errors: Vec<IntegrationErrorComponents<S>>,
    offsets: Vec<S>,
    floors: Vec<S>,
    endpoints: Vec<S>,
    scales: Vec<S>,
    normalized: Vec<S>,
    captured: bool,
    capacity: usize,
    interval: Interval<S>,
    mode: SensitivityMode,
    nodes: usize,
    components: usize,
    parameters: usize,
}

impl<S: RealScalar> core::fmt::Debug for QuadratureSensitivityPlan<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuadratureSensitivityPlan")
            .field("rule", &self.rule)
            .field("discretization", &self.discretization)
            .field("components", &self.components)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar> core::fmt::Debug for QuadratureSensitivityWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuadratureSensitivityWorkspace")
            .field("requirements", &self.requirements)
            .field("captured", &self.captured)
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar> QuadratureSensitivityPlan<S> {
    /// Creates a plan that applies `rule` once to `interval`.
    ///
    /// The termination is `FixedRuleCompleted`, or `DegenerateDomain` for an
    /// interval of zero length. The report still says whether the error
    /// estimates meet the tolerances.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `integrand` does not support `options.mode()`.
    /// - `InvalidInput` (`OutOfRange`) if `interval` is too short to hold
    ///   distinct nodes of `rule`.
    /// - `DimensionMismatch` if a tolerance has per-component absolute
    ///   tolerances of the wrong length: the number of value components for
    ///   the primal tolerance, and the length of the derivative product for
    ///   the derivative tolerance.
    /// - `SizeOverflow` or `AllocationFailure` if the plan cannot be stored.
    pub fn fixed<I: SensitivityIntegrand<S> + ?Sized>(
        integrand: &I,
        interval: Interval<S>,
        rule: GaussKronrodRule,
        options: SensitivityOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            integrand,
            interval,
            rule,
            SensitivityDiscretization::Fixed,
            options,
        )
    }
    /// Creates a plan that integrates over `interval` with adaptive G7K15
    /// quadrature.
    ///
    /// The plan bisects subintervals until the error estimates of both the
    /// integral and the derivative meet their tolerances, or until a limit
    /// such as [`max_subdivisions`](SensitivityOptions::with_max_subdivisions)
    /// is reached; see [Termination](crate#termination).
    ///
    /// # Errors
    ///
    /// The errors of [`fixed`](Self::fixed), with the G7K15 rule.
    pub fn adaptive<I: SensitivityIntegrand<S> + ?Sized>(
        integrand: &I,
        interval: Interval<S>,
        options: SensitivityOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            integrand,
            interval,
            GaussKronrodRule::G7K15,
            SensitivityDiscretization::Adaptive,
            options,
        )
    }
    /// Creates a plan that applies the rule of `partition` to each of its
    /// subintervals, without subdivision.
    ///
    /// Get a partition from [`partition`](Self::partition) after a successful
    /// call, typically of an adaptive plan. The termination is
    /// `FixedRuleCompleted`.
    ///
    /// # Errors
    ///
    /// The errors of [`fixed`](Self::fixed), checked for every subinterval.
    pub fn frozen<I: SensitivityIntegrand<S> + ?Sized>(
        integrand: &I,
        partition: &SensitivityPartition<S>,
        options: SensitivityOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let mut plan = Self::prepare(
            integrand,
            partition.interval,
            partition.rule,
            SensitivityDiscretization::Frozen,
            options,
        )?;
        plan.quadrature = Arc::new(KronrodPartition::new(
            partition.interval,
            &partition.intervals,
            partition.rule,
        )?);
        Ok(plan)
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare<I: SensitivityIntegrand<S> + ?Sized>(
        integrand: &I,
        interval: Interval<S>,
        rule: GaussKronrodRule,
        discretization: SensitivityDiscretization,
        options: SensitivityOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let mode = options.mode;
        if !integrand.supports(mode) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("integrand"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "provider lacks the requested derivative capability",
            ));
        }
        let components = integrand.value_dimension();
        let parameters = integrand.parameter_dimension();
        let products = if mode == SensitivityMode::Jvp {
            components
        } else {
            parameters
        };
        checked_add(
            components,
            products,
            scioxide_core::quantity!("stacked_components"),
        )?;
        crate::workspace::check_tolerance(options.primal_tolerance(), components)?;
        crate::workspace::check_tolerance(options.derivative_tolerance(), products)?;
        let initial = if interval.lower() == interval.upper() {
            &[][..]
        } else {
            core::slice::from_ref(&interval)
        };
        Ok(Self {
            quadrature: Arc::new(KronrodPartition::new(interval, initial, rule)?),
            rule,
            discretization,
            parameters,
            components,
            options,
        })
    }
    /// Returns the interval of integration.
    pub fn interval(&self) -> Interval<S> {
        self.quadrature.interval()
    }
    /// Returns how the plan chooses its partition.
    pub const fn discretization(&self) -> SensitivityDiscretization {
        self.discretization
    }
    /// Returns the derivative product the plan computes.
    pub const fn mode(&self) -> SensitivityMode {
        self.options.mode
    }
    /// Returns the initial subintervals: the whole interval for `fixed` and
    /// `adaptive` plans, the given partition for `frozen` plans, and none for
    /// an interval of zero length.
    pub fn intervals(&self) -> &[Interval<S>] {
        self.quadrature.segments()
    }
    fn products(&self) -> usize {
        if self.options.mode == SensitivityMode::Jvp {
            self.components
        } else {
            self.parameters
        }
    }
    fn dimension(&self) -> usize {
        self.components + self.products()
    }
    fn capacity(&self) -> Result<usize, SciOxideError> {
        checked_add(
            self.intervals().len(),
            if self.discretization == SensitivityDiscretization::Adaptive {
                self.options.subdivision.max_subdivisions().get()
            } else {
                0
            },
            scioxide_core::quantity!("workspace_regions"),
        )
    }
    /// Returns the storage requirements of the plan's workspace.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the number of subintervals overflows `usize`.
    pub fn workspace_requirements(&self) -> Result<SensitivityRequirements<S>, SciOxideError> {
        Ok(SensitivityRequirements {
            interval: self.interval(),
            mode: self.options.mode,
            nodes: self.rule.rule().evaluation_cost(),
            components: self.components,
            parameters: self.parameters,
            capacity: self.capacity()?,
        })
    }
    /// Creates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn workspace(&self) -> Result<QuadratureSensitivityWorkspace<S>, SciOxideError> {
        QuadratureSensitivityWorkspace::allocate(&self.workspace_requirements()?)
    }
    /// Returns the partition used by the last successful call with
    /// `workspace`, for use with [`frozen`](Self::frozen).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `workspace` belongs to an incompatible plan, or its
    ///   last call failed or did not happen.
    /// - `AllocationFailure` if the partition cannot be stored.
    pub fn partition(
        &self,
        workspace: &QuadratureSensitivityWorkspace<S>,
    ) -> Result<SensitivityPartition<S>, SciOxideError> {
        self.validate_workspace(workspace)?;
        if !workspace.captured {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no complete partition to capture",
            ));
        }
        let mut intervals = try_with_capacity(
            workspace.regions.measured_regions().count(),
            scioxide_core::quantity!("partition"),
        )?;
        intervals.extend(workspace.regions.measured_regions());
        Ok(SensitivityPartition {
            interval: self.interval(),
            intervals,
            rule: self.rule,
        })
    }
    fn validate_workspace(
        &self,
        ws: &QuadratureSensitivityWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        if ws.capacity < self.capacity()? {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient reserved sensitivity regions",
            ));
        }
        if ws.interval.lower() != self.interval().lower()
            || ws.interval.upper() != self.interval().upper()
            || ws.mode != self.options.mode
            || ws.components != self.components
            || ws.parameters != self.parameters
            || ws.nodes != self.rule.rule().evaluation_cost()
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "workspace belongs to an incompatible sensitivity plan",
            ));
        }
        Ok(())
    }

    /// Computes the integral `I` and the Jacobian–vector product
    /// `(∂I/∂p) v` at the parameters of `query`.
    ///
    /// `outputs` receives `I` and the product, each with one entry per value
    /// component. `workspaces` holds the plan's workspace and a pool of
    /// integrand workspaces; a parallel execution policy uses at most one
    /// worker per pool entry, and the result does not depend on the number of
    /// workers. If the termination is unsuccessful, both outputs still hold
    /// complete estimates; see [Termination](crate#termination).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the plan was not prepared for
    ///   [`SensitivityMode::Jvp`], `integrand` does not support it, the
    ///   integrand does not apply its Jacobian exactly once at a node, the
    ///   workspace belongs to an incompatible plan, or the pool is empty.
    /// - `DimensionMismatch` if the parameters, the direction or an output
    ///   has the wrong length, or the integrand's dimensions differ from those
    ///   at preparation.
    /// - `NonFinite` with origin `Input` if parameters, direction or endpoint
    ///   tangents are NaN or infinite; origin `Callback` for non-finite values
    ///   or products returned by the integrand. Invalid query data is rejected
    ///   before a solve epoch starts.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the call before a first estimate of the whole integral,
    ///   for example if `max_evaluations` is below the number of nodes of the
    ///   initial rule plus the endpoint evaluations.
    /// - Any error returned by the integrand or its operator.
    pub fn jvp_into<'scope, 'capability, I, V, D>(
        &self,
        integrand: &I,
        query: JvpQuery<'_, S>,
        outputs: (&mut V, &mut D),
        workspaces: (&mut QuadratureSensitivityWorkspace<S>, &mut [I::Workspace]),
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<SensitivityReport, SciOxideError>
    where
        I: SensitivityIntegrand<S> + Sync + ?Sized,
        I::Workspace: Send,
        V: AffineValueMut<Scale = S> + ?Sized,
        D: AffineValueMut<Scale = S> + ?Sized,
    {
        self.execute(
            integrand,
            query.parameters,
            Product::Jvp(query.direction),
            Boundary::Jvp(query.endpoint_tangent),
            outputs,
            workspaces,
            scope.into(),
        )
    }
    /// Computes the integral `I` and the vector–Jacobian product
    /// `wᵀ (∂I/∂p)` at the parameters of `query`.
    ///
    /// `outputs` receives `I`, with one entry per value component, and the
    /// product, with one entry per parameter. The product uses the real inner
    /// product of the real coordinates, so for complex values it is the
    /// adjoint under `⟨u, v⟩ = Re(conj(u) v)`. Workspaces and termination
    /// are as for [`jvp_into`](Self::jvp_into).
    ///
    /// # Errors
    ///
    /// The errors of [`jvp_into`](Self::jvp_into) with
    /// [`SensitivityMode::Vjp`], the cotangent in place of the direction and
    /// the endpoint gradients in place of the tangents; an endpoint gradient
    /// whose length is not the number of parameters is a `DimensionMismatch`.
    pub fn vjp_into<'scope, 'capability, I, V, D>(
        &self,
        integrand: &I,
        query: VjpQuery<'_, S>,
        outputs: (&mut V, &mut D),
        workspaces: (&mut QuadratureSensitivityWorkspace<S>, &mut [I::Workspace]),
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<SensitivityReport, SciOxideError>
    where
        I: SensitivityIntegrand<S> + Sync + ?Sized,
        I::Workspace: Send,
        V: AffineValueMut<Scale = S> + ?Sized,
        D: AffineValueMut<Scale = S> + ?Sized,
    {
        self.execute(
            integrand,
            query.parameters,
            Product::Vjp(query.cotangent),
            Boundary::Vjp(query.endpoint_gradients),
            outputs,
            workspaces,
            scope.into(),
        )
    }

    /// Validates caller-supplied query data before starting a solve epoch.
    fn validate_query<I: SensitivityIntegrand<S> + ?Sized>(
        &self,
        integrand: &I,
        parameters: &[S],
        product: Product<'_, S>,
        boundary: Boundary<'_, S>,
    ) -> Result<(), SciOxideError> {
        if product.mode() != Some(self.options.mode) || !integrand.supports(self.options.mode) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("integrand"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "unprepared derivative capability",
            ));
        }
        crate::kernel::shape(self.parameters, integrand.parameter_dimension())?;
        crate::kernel::shape(self.components, integrand.value_dimension())?;
        crate::kernel::shape(self.parameters, parameters.len())?;
        scioxide_core::validate_value(parameters)?;
        let (dimension, seed) = match product {
            Product::Jvp(seed) => (self.parameters, seed),
            Product::Vjp(seed) => (self.components, seed),
            Product::Value => {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("query"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing derivative request",
                ));
            }
        };
        crate::kernel::shape(dimension, seed.len())?;
        scioxide_core::validate_value(seed)?;
        boundary.validate(self.parameters)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute<I, V, D>(
        &self,
        integrand: &I,
        parameters: &[S],
        product: Product<'_, S>,
        boundary: Boundary<'_, S>,
        outputs: (&mut V, &mut D),
        workspaces: (&mut QuadratureSensitivityWorkspace<S>, &mut [I::Workspace]),
        scope: scioxide_core::SolveScope<'_, '_>,
    ) -> Result<SensitivityReport, SciOxideError>
    where
        I: SensitivityIntegrand<S> + Sync + ?Sized,
        I::Workspace: Send,
        V: AffineValueMut<Scale = S> + ?Sized,
        D: AffineValueMut<Scale = S> + ?Sized,
    {
        let (ws, providers) = workspaces;
        ws.captured = false;
        if let Err(error) = self
            .validate_workspace(ws)
            .and_then(|()| check_value_shape(outputs.0, self.components))
            .and_then(|()| check_value_shape(outputs.1, self.products()))
            .and_then(|()| self.validate_query(integrand, parameters, product, boundary))
            .and_then(|()| {
                if providers.is_empty() {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("provider_workspaces"),
                        scioxide_core::InvalidInputReason::Empty,
                        "empty provider workspace pool",
                    ));
                }
                Ok(())
            })
        {
            return ws.attempts.reject(error);
        }
        let mut attempts = core::mem::take(&mut ws.attempts);
        let mut primal = core::mem::take(&mut ws.primal_errors);
        let mut derivative = core::mem::take(&mut ws.product_errors);
        let result = attempts.run(
            scope,
            crate::workspace::identity(
                scioxide_core::algorithm_id!("integrate", "Sensitivity", 1),
                self.dimension(),
            ),
            self.options.subdivision.work_allowance(),
            |cx| {
                let report = self.run(
                    product,
                    boundary,
                    outputs.0,
                    outputs.1,
                    &mut primal,
                    &mut derivative,
                    ws,
                    cx,
                    &mut |points, rows, action, context| {
                        sampling::executor(
                            integrand,
                            parameters,
                            action,
                            points,
                            rows,
                            providers,
                            self.options.subdivision.execution_policy(),
                            context,
                        )
                    },
                )?;
                if !ws.captured {
                    return Err(crate::workspace::unpublished(
                        report.termination,
                        scioxide_core::algorithm_id!("integrate", "Sensitivity", 1),
                        cx.counts(),
                    ));
                }
                Ok(report)
            },
        );
        ws.attempts = attempts;
        ws.primal_errors = primal;
        ws.product_errors = derivative;
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn run<V, D>(
        &self,
        product: Product<'_, S>,
        boundary: Boundary<'_, S>,
        value: &mut V,
        derivative: &mut D,
        value_errors: &mut [IntegrationErrorComponents<S>],
        derivative_errors: &mut [SensitivityErrorComponents<S>],
        ws: &mut QuadratureSensitivityWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
        sampler: &mut sampling::Sampler<'_, S>,
    ) -> Result<SensitivityReport, SciOxideError>
    where
        V: AffineValueMut<Scale = S> + ?Sized,
        D: AffineValueMut<Scale = S> + ?Sized,
    {
        ws.captured = false;
        let initial_counts = context.counts();
        let mut report = RegionalProgress {
            termination: IntegrationTermination::FixedRuleCompleted,
            measured: false,
            num_evals: 0,
            subdivisions: 0,
        };
        let finish = |report: RegionalProgress,
                      context: &SolveContext<'_, '_>,
                      primal,
                      derivative| SensitivityReport {
            termination: report.termination,
            function_evaluations: context.counts().own().provider(EvaluationRole::Function)
                - initial_counts.own().provider(EvaluationRole::Function),
            derivative_evaluations: context
                .counts()
                .own()
                .provider(EvaluationRole::LinearOperator)
                - initial_counts
                    .own()
                    .provider(EvaluationRole::LinearOperator),
            subdivisions: report.subdivisions,
            primal_tolerance_met: primal,
            derivative_tolerance_met: derivative,
        };
        if let Some(stop) = context.stop_reason()? {
            report.termination = IntegrationTermination::ExecutionStopped(stop);
            return Ok(finish(report, context, false, false));
        }
        ws.offsets.fill(S::ZERO);
        ws.floors.fill(S::ZERO);
        ws.endpoints.fill(S::ZERO);
        // Leibniz rule: d/dp int_a(p)^b(p) f(x, p) dx = int df/dp dx + f(b) b' - f(a) a'.
        // The boundary terms need one plain value sample at each moving endpoint; endpoints
        // with a zero tangent/gradient are skipped so fixed bounds cost nothing. Each
        // interior node costs two provider records (value + derivative action).
        let needed = if self.components == 0 {
            [false; 2]
        } else {
            boundary.needed()
        };
        let endpoint_cost = needed.iter().filter(|&&v| v).count();
        let initial_nodes = if self.components == 0 {
            0
        } else {
            checked_mul(
                self.intervals().len(),
                self.rule.rule().evaluation_cost(),
                scioxide_core::quantity!("initial_nodes"),
            )?
        };
        let function_cost = checked_add(
            initial_nodes,
            endpoint_cost,
            scioxide_core::quantity!("function_evaluations"),
        )?;
        let provider_cost = checked_add(
            checked_mul(
                initial_nodes,
                2,
                scioxide_core::quantity!("provider_evaluations"),
            )?,
            endpoint_cost,
            scioxide_core::quantity!("provider_evaluations"),
        )?;
        if self
            .options
            .subdivision
            .max_evaluations()
            .is_some_and(|b| b.get() < function_cost)
        {
            report.termination =
                IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation);
            return Ok(finish(report, context, false, false));
        }
        for (kind, amount) in [
            (BudgetKind::Evaluation, provider_cost),
            (BudgetKind::Iteration, usize::from(initial_nodes != 0)),
            (BudgetKind::Proposal, usize::from(initial_nodes != 0)),
        ] {
            if !context.can_consume(kind, amount)?.is_granted() {
                report.termination = IntegrationTermination::BudgetLimit(kind);
                return Ok(finish(report, context, false, false));
            }
        }
        for (i, x) in [self.interval().lower(), self.interval().upper()]
            .into_iter()
            .enumerate()
        {
            if needed[i] {
                report.num_evals += sampler(
                    core::slice::from_ref(&x),
                    &mut ws.endpoints[i * self.components..(i + 1) * self.components],
                    Product::Value,
                    context,
                )?;
            }
        }
        boundary.contribution(
            product,
            self.components,
            &ws.endpoints,
            &mut ws.offsets[self.components..],
            &mut ws.floors[self.components..],
        )?;
        if self.components == 0 {
            ws.values.fill(S::ZERO);
            ws.errors
                .fill(IntegrationErrorComponents::discretization_only(S::ZERO));
            report.measured = true;
        } else {
            // Value and derivative are integrated as one stacked affine value
            // `[value (m) | product]`, sharing every node and the adaptive partition. The
            // refinement norm is the worse of the two tolerance norms, and the boundary terms
            // enter as constant offsets (with their roundoff floors) on the product part.
            let m = self.components;
            let error_norm = |errors: &[S], scales: &[S]| -> Result<S, SciOxideError> {
                Ok(
                    norm(self.options.primal_tolerance(), &errors[..m], &scales[..m])?.max(norm(
                        self.options.derivative_tolerance(),
                        &errors[m..],
                        &scales[m..],
                    )?),
                )
            };
            let policy = RegionalPolicy {
                extrapolation: false,
                norm: &error_norm,
                refine: self.discretization == SensitivityDiscretization::Adaptive,
                offsets: Some((&ws.offsets, &ws.floors)),
            };
            report = self.quadrature.run_controlled(
                &self.options.subdivision,
                &mut ws.values[..],
                &mut ws.errors,
                &mut ws.regions,
                context,
                &mut |points, rows, ctx| sampler(points, rows, product, ctx),
                report.num_evals,
                &policy,
            )?;
        }
        if !report.measured {
            return Ok(finish(report, context, false, false));
        }
        for i in 0..self.dimension() {
            ws.scales[i] = ws.values[i].abs();
            ws.normalized[i] = crate::kernel::checked_intermediate(
                ws.errors[i].total().ok_or_else(|| {
                    crate::kernel::invalid(
                        scioxide_core::argument!("integrand"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "missing quadrature error",
                    )
                })? + ws.floors[i],
            )?;
        }
        let primal_met = norm(
            self.options.primal_tolerance(),
            &ws.normalized[..self.components],
            &ws.scales[..self.components],
        )? <= S::ONE;
        let derivative_met = norm(
            self.options.derivative_tolerance(),
            &ws.normalized[self.components..],
            &ws.scales[self.components..],
        )? <= S::ONE;
        context.commit_accepted(PartialResultTrust::Admissible)?;
        write_value(&ws.values[..self.components], value)?;
        write_value(&ws.values[self.components..], derivative)?;
        value_errors.copy_from_slice(&ws.errors[..self.components]);
        for (i, error) in derivative_errors.iter_mut().enumerate() {
            *error = SensitivityErrorComponents {
                quadrature: ws.errors[self.components + i],
                boundary_roundoff: ws.floors[self.components + i],
            };
        }
        ws.captured = true;
        Ok(finish(report, context, primal_met, derivative_met))
    }
}

#[derive(Clone, Copy)]
enum Boundary<'a, S> {
    Jvp(Option<[S; 2]>),
    Vjp(Option<(&'a [S], &'a [S])>),
}
impl<S: RealScalar> Boundary<'_, S> {
    fn validate(self, p: usize) -> Result<(), SciOxideError> {
        match self {
            Self::Jvp(Some(values)) => scioxide_core::validate_value(&values),
            Self::Vjp(Some((lower, upper))) => {
                crate::kernel::shape(p, lower.len())?;
                crate::kernel::shape(p, upper.len())?;
                scioxide_core::validate_value(lower)?;
                scioxide_core::validate_value(upper)
            }
            _ => Ok(()),
        }
    }
    fn needed(self) -> [bool; 2] {
        match self {
            Self::Jvp(Some(tangent)) => tangent.map(|v| v != S::ZERO),
            Self::Vjp(Some((lower, upper))) => [
                lower.iter().any(|&v| v != S::ZERO),
                upper.iter().any(|&v| v != S::ZERO),
            ],
            _ => [false; 2],
        }
    }
    fn contribution(
        self,
        product: Product<'_, S>,
        m: usize,
        endpoints: &[S],
        values: &mut [S],
        floors: &mut [S],
    ) -> Result<(), SciOxideError> {
        for (i, (value, floor)) in values.iter_mut().zip(floors).enumerate() {
            let mut sum = RangeAccumulator::new();
            let mut absolute = RangeAccumulator::new();
            // JVP: (f(b) db - f(a) da) per value component. VJP: for each parameter i,
            // sum_c w_c (f_c(b) grad_i b - f_c(a) grad_i a) with cotangent w. The absolute
            // sums give the scale for the boundary roundoff floor.
            match (self, product) {
                (Self::Jvp(Some([lower, upper])), _) => {
                    sum.add_product(-lower, endpoints[i])?;
                    sum.add_product(upper, endpoints[m + i])?;
                    absolute.add_product(lower.abs(), endpoints[i].abs())?;
                    absolute.add_product(upper.abs(), endpoints[m + i].abs())?;
                }
                (Self::Vjp(Some((lower, upper))), Product::Vjp(seed)) => {
                    for (c, &cotangent) in seed.iter().enumerate() {
                        sum.add_triple_product(-lower[i], endpoints[c], cotangent)?;
                        sum.add_triple_product(upper[i], endpoints[m + c], cotangent)?;
                        absolute.add_triple_product(
                            lower[i].abs(),
                            endpoints[c].abs(),
                            cotangent.abs(),
                        )?;
                        absolute.add_triple_product(
                            upper[i].abs(),
                            endpoints[m + c].abs(),
                            cotangent.abs(),
                        )?;
                    }
                }
                _ => {}
            }
            *value = sum.finish()?;
            *floor = crate::kernel::diagnostic(
                absolute.finish().and_then(crate::kernel::roundoff_floor),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

impl<S: RealScalar> QuadratureSensitivityWorkspace<S> {
    fn allocate(requirements: &SensitivityRequirements<S>) -> Result<Self, SciOxideError> {
        let d = requirements.components + requirements.products();
        Ok(Self {
            attempts: crate::workspace::Attempts::default(),
            requirements: *requirements,
            primal_errors: try_filled(
                requirements.components,
                IntegrationErrorComponents::discretization_only(S::ZERO),
                scioxide_core::quantity!("primal_errors"),
            )?,
            product_errors: try_filled(
                requirements.products(),
                SensitivityErrorComponents {
                    quadrature: IntegrationErrorComponents::discretization_only(S::ZERO),
                    boundary_roundoff: S::ZERO,
                },
                scioxide_core::quantity!("product_errors"),
            )?,
            regions: crate::adaptive::regional::RegionalWorkspace::new(
                d,
                requirements.capacity,
                requirements.nodes,
            )?,
            values: try_filled(d, S::ZERO, scioxide_core::quantity!("stacked_values"))?,
            errors: try_filled(
                d,
                IntegrationErrorComponents::discretization_only(S::ZERO),
                scioxide_core::quantity!("stacked_errors"),
            )?,
            offsets: try_filled(d, S::ZERO, scioxide_core::quantity!("boundary_offsets"))?,
            floors: try_filled(d, S::ZERO, scioxide_core::quantity!("boundary_floors"))?,
            endpoints: try_filled(
                checked_mul(
                    2,
                    requirements.components,
                    scioxide_core::quantity!("endpoint_samples"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("endpoint_samples"),
            )?,
            scales: try_filled(d, S::ZERO, scioxide_core::quantity!("stacked_scales"))?,
            normalized: try_filled(d, S::ZERO, scioxide_core::quantity!("normalized_errors"))?,
            captured: false,
            capacity: requirements.capacity,
            interval: requirements.interval,
            mode: requirements.mode,
            nodes: requirements.nodes,
            components: requirements.components,
            parameters: requirements.parameters,
        })
    }
}

/// Derivative mode, tolerances, limits and execution policy of a
/// [`QuadratureSensitivityPlan`].
///
/// The integral and the derivative have separate tolerances. The defaults are
/// [`SensitivityMode::Jvp`], the [`SubdivisionOptions`] defaults for the
/// integral, its limits and its execution policy, and the default
/// [`ErrorTolerance`] (`atol = rtol = √ε`, root-mean-square norm) for the
/// derivative.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ErrorTolerance, NormSelection};
/// use scioxide_integrate::{SensitivityMode, SensitivityOptions};
///
/// let tolerance = ErrorTolerance::scalar(1e-10, 1e-10, NormSelection::Infinity)?;
/// let options = SensitivityOptions::<f64>::default()
///     .with_mode(SensitivityMode::Vjp)
///     .with_derivative_tolerance(tolerance);
/// assert_eq!(options.mode(), SensitivityMode::Vjp);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SensitivityOptions<S: RealScalar> {
    subdivision: SubdivisionOptions<S>,
    derivative: ErrorTolerance<S>,
    mode: SensitivityMode,
}
impl<S: RealScalar> Default for SensitivityOptions<S> {
    fn default() -> Self {
        Self {
            subdivision: SubdivisionOptions::default(),
            derivative: ErrorTolerance::default(),
            mode: SensitivityMode::Jvp,
        }
    }
}
impl<S: RealScalar> SensitivityOptions<S> {
    /// Creates options with the integral's tolerance, limits and execution
    /// policy from `subdivision`, and the default derivative tolerance and
    /// mode.
    pub fn new(subdivision: SubdivisionOptions<S>) -> Self {
        Self {
            subdivision,
            derivative: ErrorTolerance::default(),
            mode: SensitivityMode::Jvp,
        }
    }
    /// Returns the integral's tolerance, limits and execution policy.
    pub const fn subdivision_options(&self) -> &SubdivisionOptions<S> {
        &self.subdivision
    }
    /// Returns the tolerance on the integral.
    pub const fn primal_tolerance(&self) -> &ErrorTolerance<S> {
        self.subdivision.tolerance()
    }
    /// Returns the tolerance on the derivative.
    pub const fn derivative_tolerance(&self) -> &ErrorTolerance<S> {
        &self.derivative
    }
    /// Returns the derivative mode.
    pub const fn mode(&self) -> SensitivityMode {
        self.mode
    }
    /// Sets the tolerance on the integral.
    pub fn with_primal_tolerance(mut self, value: ErrorTolerance<S>) -> Self {
        self.subdivision = self.subdivision.with_tolerance(value);
        self
    }
    /// Sets the tolerance on the derivative, endpoint terms included.
    pub fn with_derivative_tolerance(mut self, value: ErrorTolerance<S>) -> Self {
        self.derivative = value;
        self
    }
    /// Sets the derivative mode.
    pub const fn with_mode(mut self, value: SensitivityMode) -> Self {
        self.mode = value;
        self
    }
    /// Sets the maximum number of bisections of an adaptive plan.
    pub fn with_max_subdivisions(mut self, value: scioxide_core::Budget) -> Self {
        self.subdivision = self.subdivision.with_max_subdivisions(value);
        self
    }
    /// Sets or removes the limit on integrand evaluations, endpoint evaluations
    /// included.
    pub fn with_max_evaluations(mut self, value: Option<scioxide_core::Budget>) -> Self {
        self.subdivision = self.subdivision.with_max_evaluations(value);
        self
    }
    /// Sets the execution policy for evaluating the nodes of a rule.
    pub fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.subdivision = self.subdivision.with_execution_policy(value);
        self
    }
}
/// Subintervals and rule of a completed integration, from
/// [`QuadratureSensitivityPlan::partition`], for use with
/// [`QuadratureSensitivityPlan::frozen`].
///
/// A partition does not depend on the derivative mode, so it can be used for
/// both JVP and VJP plans.
#[derive(Debug, Clone)]
pub struct SensitivityPartition<S: RealScalar> {
    interval: Interval<S>,
    intervals: Vec<Interval<S>>,
    rule: GaussKronrodRule,
}
impl<S: RealScalar> SensitivityPartition<S> {
    /// Returns the subintervals, in the order their contributions are summed.
    pub fn intervals(&self) -> &[Interval<S>] {
        &self.intervals
    }
}
/// Parameters, direction and endpoint tangents of a Jacobian–vector product.
#[derive(Debug, Clone, Copy)]
pub struct JvpQuery<'a, S> {
    parameters: &'a [S],
    direction: &'a [S],
    endpoint_tangent: Option<[S; 2]>,
}
impl<'a, S> JvpQuery<'a, S> {
    /// Creates a query for the parameters `parameters` and the direction `v`
    /// (`direction`), with fixed endpoints.
    ///
    /// Both slices have one entry per parameter.
    pub const fn new(parameters: &'a [S], direction: &'a [S]) -> Self {
        Self {
            parameters,
            direction,
            endpoint_tangent: None,
        }
    }
    /// Sets the endpoint tangents `[da, db] = [∇a·v, ∇b·v]` of a moving
    /// interval `[a(p), b(p)]`, or `None` for fixed endpoints.
    ///
    /// A zero tangent costs no integrand evaluation.
    pub fn with_endpoint_tangent(mut self, value: Option<[S; 2]>) -> Self {
        self.endpoint_tangent = value;
        self
    }
}
/// Parameters, cotangent and endpoint gradients of a vector–Jacobian product.
#[derive(Debug, Clone, Copy)]
pub struct VjpQuery<'a, S> {
    parameters: &'a [S],
    cotangent: &'a [S],
    endpoint_gradients: Option<(&'a [S], &'a [S])>,
}
impl<'a, S> VjpQuery<'a, S> {
    /// Creates a query for the parameters `parameters` and the cotangent `w`
    /// (`cotangent`), with fixed endpoints.
    ///
    /// `cotangent` has one entry per value component.
    pub const fn new(parameters: &'a [S], cotangent: &'a [S]) -> Self {
        Self {
            parameters,
            cotangent,
            endpoint_gradients: None,
        }
    }
    /// Sets the gradients `(∇a, ∇b)` of the endpoints of a moving interval
    /// `[a(p), b(p)]`, each with one entry per parameter, or `None` for fixed
    /// endpoints.
    ///
    /// A zero gradient costs no integrand evaluation.
    pub fn with_endpoint_gradients(mut self, value: Option<(&'a [S], &'a [S])>) -> Self {
        self.endpoint_gradients = value;
        self
    }
}
/// Storage requirements of a [`QuadratureSensitivityWorkspace`], from
/// [`QuadratureSensitivityPlan::workspace_requirements`].
///
/// They record the interval, mode, rule size, dimensions and the maximum
/// number of subintervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensitivityRequirements<S: RealScalar> {
    interval: Interval<S>,
    mode: SensitivityMode,
    nodes: usize,
    components: usize,
    parameters: usize,
    capacity: usize,
}
impl<S: RealScalar> SensitivityRequirements<S> {
    fn products(&self) -> usize {
        if self.mode == SensitivityMode::Jvp {
            self.components
        } else {
            self.parameters
        }
    }
}
impl<S: RealScalar> QuadratureSensitivityWorkspace<S> {
    /// Returns whether the workspace serves the requirements `r` without
    /// reallocation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `r` has another interval, mode, rule size or
    ///   dimensions.
    pub fn check(
        &self,
        r: &SensitivityRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let mut expected = *r;
        expected.capacity = self.requirements.capacity;
        if expected != self.requirements {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("requirements"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "incompatible sensitivity preparation",
            ));
        }
        Ok(if self.requirements.capacity >= r.capacity {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    /// Reallocates the workspace, if needed, so that it serves `r`.
    ///
    /// Reallocation keeps the record of the last call but discards its
    /// error estimates and partition. On error, the workspace is unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `r` has another interval, mode, rule size or
    ///   dimensions.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn try_reserve(&mut self, r: &SensitivityRequirements<S>) -> Result<(), SciOxideError> {
        if self.check(r)? == scioxide_core::Capacity::NeedsGrowth {
            let mut grown = Self::allocate(r)?;
            core::mem::swap(&mut self.attempts, &mut grown.attempts);
            *self = grown;
        }
        Ok(())
    }
    /// Returns a record of the last call with this workspace: its outcome, its
    /// evaluation counts and its report, or `None` before the first call.
    pub fn last_attempt(&self) -> Option<scioxide_core::AttemptRecord<SensitivityReport>> {
        self.attempts.last_attempt()
    }
    /// Returns the error estimates of the integral from the last call, or
    /// `None` if it failed.
    pub fn value_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.captured.then_some(&self.primal_errors)
    }
    /// Returns the error estimates of the derivative from the last call, or
    /// `None` if it failed.
    pub fn derivative_errors(&self) -> Option<&[SensitivityErrorComponents<S>]> {
        self.captured.then_some(&self.product_errors)
    }
}
impl<S: RealScalar> SensitivityErrorComponents<S> {
    /// Returns the quadrature error estimate of the integral of the Jacobian
    /// product.
    pub const fn quadrature(&self) -> &IntegrationErrorComponents<S> {
        &self.quadrature
    }
    /// Returns the rounding error estimate of the endpoint terms, zero for
    /// fixed endpoints.
    pub const fn boundary_roundoff(&self) -> S {
        self.boundary_roundoff
    }
}
impl SensitivityReport {
    /// Returns why the integration stopped.
    pub const fn termination(&self) -> IntegrationTermination {
        self.termination
    }
    /// Returns the number of integrand evaluations, including endpoint
    /// evaluations and failed calls.
    pub const fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }
    /// Returns the number of Jacobian operator applications, including failed
    /// ones.
    pub const fn derivative_evaluations(&self) -> usize {
        self.derivative_evaluations
    }
    /// Returns the number of bisections.
    pub const fn subdivisions(&self) -> usize {
        self.subdivisions
    }
    /// Returns whether the error estimate of the integral meets its tolerance.
    pub const fn primal_tolerance_met(&self) -> bool {
        self.primal_tolerance_met
    }
    /// Returns whether the error estimate of the derivative, endpoint rounding
    /// included, meets its tolerance.
    pub const fn derivative_tolerance_met(&self) -> bool {
        self.derivative_tolerance_met
    }
}

impl<S: RealScalar> QuadratureSensitivityPlan<S> {
    /// Returns the options.
    pub const fn options(&self) -> &SensitivityOptions<S> {
        &self.options
    }
    /// Returns the tolerance on the derivative.
    pub const fn derivative_tolerance(&self) -> &ErrorTolerance<S> {
        self.options.derivative_tolerance()
    }
}
