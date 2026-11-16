//! Fixed one-dimensional quadrature rules: the `FixedRule` trait, its workspace
//! and the rules that implement it.

mod apply;
mod clenshaw_curtis;
mod newton_cotes;

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Interval, RealScalar, SciOxideError, SolveScope,
};

use crate::kernel::AffineMap1D;
use crate::report::{IntegrationErrorComponents, IntegrationReport};
use crate::workspace::{Attempts, Family, IntegrationRequirements, workspace_protocol};

pub(crate) use apply::{Geometry, Mapping, apply, apply_kronrod};

pub use clenshaw_curtis::ClenshawCurtis;
pub use newton_cotes::{NewtonCotes, NewtonCotesEndpoints};

/// Affine map of `interval`; a zero-width interval is an error.
pub(crate) fn interval_map<S: RealScalar>(
    interval: Interval<S>,
) -> Result<AffineMap1D<S>, SciOxideError> {
    AffineMap1D::from_interval(interval)?.ok_or_else(degenerate_interval)
}

/// Error for mapping a rule onto a zero-width interval.
pub(crate) fn degenerate_interval() -> SciOxideError {
    crate::kernel::invalid(
        scioxide_core::argument!("interval"),
        scioxide_core::InvalidInputReason::OutOfRange,
        "fixed-rule metadata requires a non-degenerate interval",
    )
}

/// One node of a fixed quadrature rule, mapped to the integration interval.
///
/// A node holds its abscissa `xᵢ`, its weight `wᵢ` in the primary rule and, for
/// rules with an embedded rule, its weight `ŵᵢ` in the embedded rule.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct FixedRuleNode<S: RealScalar> {
    point: S,
    primary_weight: S,
    embedded_weight: Option<S>,
}

impl<S: RealScalar> FixedRuleNode<S> {
    /// Creates a node from its abscissa and weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_integrate::FixedRuleNode;
    ///
    /// let node = FixedRuleNode::new(0.5_f64, 1.0, Some(0.9));
    /// assert_eq!(node.embedded_weight(), Some(0.9));
    /// ```
    pub const fn new(point: S, primary_weight: S, embedded_weight: Option<S>) -> Self {
        Self {
            point,
            primary_weight,
            embedded_weight,
        }
    }

    /// Returns the abscissa `xᵢ` at which the integrand is evaluated.
    pub const fn point(self) -> S {
        self.point
    }

    /// Returns the weight `wᵢ` of the primary rule.
    pub const fn primary_weight(self) -> S {
        self.primary_weight
    }

    /// Returns the weight `ŵᵢ` of the embedded rule, or `None` if the rule has
    /// no embedded rule.
    pub const fn embedded_weight(self) -> Option<S> {
        self.embedded_weight
    }
}

/// Applies a fixed quadrature rule to `∫ₐᵇ f(x) dx` on a finite interval.
///
/// ```text
/// ∫ₐᵇ f(x) dx ≈ Σᵢ wᵢ f(xᵢ),   i = 0, …, n − 1,
/// ```
///
/// where the nodes `xᵢ` and weights `wᵢ` are the rule's reference nodes and
/// weights mapped affinely to `[a, b]`, and `n` is [`Self::node_count`]. The rule
/// calls `f` exactly `n` times. A rule may carry an embedded rule with weights
/// `ŵᵢ` on the same nodes; the difference of the two sums is then the error
/// estimate. In this crate only [`GaussKronrod`](crate::GaussKronrod) has an
/// embedded rule.
///
/// All nodes are mapped and checked before the first call of `f`. They must be
/// strictly monotone and lie in `[a, b]`. A one-node rule may place its node on
/// an endpoint only if [`Self::permits_single_endpoint`] returns `true`. On a
/// zero-width interval the result is exactly zero with termination
/// [`DegenerateDomain`](crate::IntegrationTermination::DegenerateDomain), and `f`
/// is not called. Every other successful call ends with
/// [`FixedRuleCompleted`](crate::IntegrationTermination::FixedRuleCompleted).
///
/// A new rule implements [`Self::algorithm_id`], [`Self::node_count`] and
/// [`Self::map_nodes_into`]. The provided methods integrate real, complex and
/// vector-valued integrands (see [Integrands](crate#integrands)) and require
/// `Self: Sized`. The weighted sums use compensated summation, which is not
/// exact.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, Interval};
/// use scioxide_integrate::{FixedRule, GaussKronrod, GaussKronrodRule};
///
/// let rule = GaussKronrod::new(GaussKronrodRule::G10K21)?;
/// let mut workspace = rule.workspace(2)?;
/// let mut out = Complex::new(0.0_f64, 0.0);
/// let exact = Complex::new(1.0_f64.sin(), 1.0 - 1.0_f64.cos()); // ∫₀¹ e^(ix) dx
/// let report = rule.integrate_into(
///     |x: f64| Ok(Complex::new(x.cos(), x.sin())),
///     Interval::new(0.0, 1.0)?,
///     &mut out,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert!(report.is_success());
/// assert!((out - exact).norm() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait FixedRule<S: RealScalar> {
    /// Returns the identifier of the rule, recorded in solve accounting.
    fn algorithm_id(&self) -> AlgorithmId;

    /// Returns the number of nodes `n`, which is the number of calls of `f`.
    fn node_count(&self) -> usize;

    /// Checks that the rule can integrate with the unit weight `w(x) = 1`.
    ///
    /// The integration methods call this check first, also for a zero-width
    /// interval. The provided implementation accepts every rule.
    ///
    /// # Errors
    ///
    /// The provided implementation does not fail.
    /// [`GaussianRulePlan`](crate::GaussianRulePlan) returns `InvalidInput` if its
    /// weight function is not identically one.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_integrate::{ClenshawCurtis, FixedRule};
    ///
    /// ClenshawCurtis::<f64>::new(8)?.validate_unweighted()?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn validate_unweighted(&self) -> Result<(), SciOxideError> {
        Ok(())
    }

    /// Returns whether a one-node rule may place its node on an endpoint of the
    /// interval.
    ///
    /// The provided implementation returns `false`, so a single interior node
    /// that rounds onto an endpoint is a breakdown.
    fn permits_single_endpoint(&self) -> bool {
        false
    }

    /// Writes the nodes and weights of the rule, mapped to `interval`, to
    /// `destination` in monotone order.
    ///
    /// `destination` must have exactly [`Self::node_count`] entries. The method
    /// does not evaluate an integrand. On error, entries of `destination` may
    /// have changed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `destination.len()` is not [`Self::node_count`].
    /// - `InvalidInput` if `interval` has zero width, or
    ///   [`Self::validate_unweighted`] fails.
    /// - `NumericalInstability` if a mapped node or weight is not representable.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_integrate::{FixedRule, FixedRuleNode, GaussKronrod, GaussKronrodRule};
    ///
    /// let rule = GaussKronrod::<f64>::new(GaussKronrodRule::G7K15)?;
    /// let mut nodes = vec![FixedRuleNode::new(0.0, 0.0, None); rule.node_count()];
    /// rule.map_nodes_into(Interval::new(0.0, 1.0)?, &mut nodes)?;
    /// assert!(
    ///     nodes
    ///         .windows(2)
    ///         .all(|pair| pair[0].point() < pair[1].point())
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn map_nodes_into(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<(), SciOxideError>;

    /// Computes the integral and its error estimate from sampled values, for a
    /// rule with an embedded rule.
    ///
    /// `nodes` are the nodes of the rule mapped to `interval` by
    /// [`Self::map_nodes_into`], each with an embedded weight. `sample(i)` returns
    /// the finite value `f(xᵢ)` of one real component at node `i`; it reads stored
    /// values and does not evaluate the integrand. The integration methods call
    /// this method only for rules with an embedded rule.
    ///
    /// The provided implementation returns `Σ wᵢ f(xᵢ)` with the discretization
    /// error `|Σ wᵢ f(xᵢ) − Σ ŵᵢ f(xᵢ)|` and the roundoff error
    /// `50 ε Σ |wᵢ f(xᵢ)|`, where `ε` is the machine epsilon of `S`.
    /// [`GaussKronrod`](crate::GaussKronrod) replaces it with its own estimate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `nodes.len()` is not [`Self::node_count`].
    /// - `NumericalInstability` if a sum overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_integrate::{FixedRule, FixedRuleNode, GaussKronrod, GaussKronrodRule};
    ///
    /// // ∫₀¹ eˣ dx = e − 1
    /// let rule = GaussKronrod::<f64>::new(GaussKronrodRule::G7K15)?;
    /// let interval = Interval::new(0.0, 1.0)?;
    /// let mut nodes = vec![FixedRuleNode::new(0.0, 0.0, None); rule.node_count()];
    /// rule.map_nodes_into(interval, &mut nodes)?;
    /// let (value, errors) = rule.estimate_embedded(interval, &nodes, &|i| nodes[i].point().exp())?;
    /// assert!((value - (1.0_f64.exp() - 1.0)).abs() < 1e-14);
    /// assert!(errors.total().unwrap() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn estimate_embedded(
        &self,
        interval: Interval<S>,
        nodes: &[FixedRuleNode<S>],
        sample: &dyn Fn(usize) -> S,
    ) -> Result<(S, IntegrationErrorComponents<S>), SciOxideError> {
        let _ = interval;
        crate::kernel::shape(self.node_count(), nodes.len())?;
        apply::embedded_estimate(nodes, sample)
    }

    /// Returns the workspace size for an integrand with `components` real
    /// components (1 for `f64`, 2 for a complex value).
    ///
    /// # Errors
    ///
    /// The provided implementation does not fail.
    fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<IntegrationRequirements, SciOxideError>
    where
        Self: Sized,
    {
        Ok(IntegrationRequirements::new(
            Family::FixedRule,
            components,
            [self.node_count(), 0],
        ))
    }

    /// Creates a workspace for an integrand with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `components · n` overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be reserved.
    fn workspace(&self, components: usize) -> Result<FixedRuleWorkspace<S>, SciOxideError>
    where
        Self: Sized,
    {
        FixedRuleWorkspace::allocate(&self.workspace_requirements(components)?)
    }

    /// Integrates `f` over `interval` and writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `out` must have the component count the workspace was created for. The
    /// report carries an error estimate only for a rule with an embedded rule;
    /// per-component estimates are in [`FixedRuleWorkspace::component_errors`].
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// solve scope whose budgets the `n` evaluations count toward.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` does not have the workspace's component
    ///   count, or the workspace is too small for this rule.
    /// - `InvalidInput` if [`Self::validate_unweighted`] fails, the rule has no
    ///   nodes, or only some nodes have an embedded weight.
    /// - `NonFinite` if a mapped weight is not finite.
    /// - `NumericalInstability` if the mapped nodes are not strictly monotone
    ///   inside `interval`, for example because the interval is too narrow to
    ///   separate them in `S`.
    /// - `ConvergenceFailure` if a budget of `scope` cannot cover `n` evaluations.
    /// - `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    /// - Errors and non-finite values from `f`, as described in
    ///   [Integrands](crate#integrands).
    fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        interval: Interval<S>,
        out: &mut O,
        workspace: &mut FixedRuleWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        Self: Sized,
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        if let Err(error) = self.validate_unweighted() {
            return workspace.reject(error);
        }
        apply(
            self,
            |nodes: &mut [FixedRuleNode<S>]| {
                self.map_nodes_into(interval, nodes)
                    .map(|()| Mapping::Stored(false))
            },
            Geometry::Interval {
                interval,
                single_endpoint: self.permits_single_endpoint(),
            },
            None,
            f,
            out,
            workspace,
            scope.into(),
        )
    }

    /// Integrates `f` over `interval` and returns the value and the report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Creates a workspace for the call and runs on
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace`] and [`Self::integrate_into`].
    fn integrate<F, V>(
        &self,
        f: F,
        interval: Interval<S>,
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        Self: Sized,
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

/// Scratch storage for applying a [`FixedRule`]: mapped nodes, sampled values
/// and per-component error estimates.
///
/// Create it with [`FixedRule::workspace`]. A workspace can be reused for any
/// rule with at most as many nodes and for integrands with the same component
/// count.
#[derive(Debug)]
pub struct FixedRuleWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    nodes: Vec<FixedRuleNode<S>>,
    rows: Vec<S>,
    values: Vec<S>,
    errors: Vec<IntegrationErrorComponents<S>>,
    published_errors: bool,
}

impl<S: RealScalar> FixedRuleWorkspace<S> {
    fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let components = requirements.components();
        let [nodes, _] = requirements.units();
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            nodes: try_filled(
                nodes,
                FixedRuleNode::new(S::ZERO, S::ZERO, None),
                scioxide_core::quantity!("fixed_rule_nodes"),
            )?,
            rows: try_filled(
                scioxide_core::backend::checked_mul(
                    nodes,
                    components,
                    scioxide_core::quantity!("fixed_rule_rows"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("fixed_rule_rows"),
            )?,
            values: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("fixed_rule_values"),
            )?,
            errors: try_filled(
                components,
                IntegrationErrorComponents::discretization_only(S::ZERO),
                scioxide_core::quantity!("fixed_rule_errors"),
            )?,
            published_errors: false,
        })
    }

    /// Returns the error estimate of each real component from the last successful
    /// call, or `None` if that call had no embedded rule or failed.
    #[must_use]
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        crate::execution::component_errors(self.published_errors, self.errors.as_slice())
    }

    /// Records an input check that failed before integration, then returns it.
    pub(crate) fn reject<T>(&mut self, error: SciOxideError) -> Result<T, SciOxideError> {
        self.published_errors = false;
        self.attempts.reject(error)
    }
}

workspace_protocol!(FixedRuleWorkspace, IntegrationReport<S>);

impl<S: RealScalar, R: FixedRule<S> + ?Sized> FixedRule<S> for &R {
    fn algorithm_id(&self) -> AlgorithmId {
        (**self).algorithm_id()
    }
    fn node_count(&self) -> usize {
        (**self).node_count()
    }
    fn validate_unweighted(&self) -> Result<(), SciOxideError> {
        (**self).validate_unweighted()
    }
    fn permits_single_endpoint(&self) -> bool {
        (**self).permits_single_endpoint()
    }
    fn map_nodes_into(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<(), SciOxideError> {
        (**self).map_nodes_into(interval, destination)
    }
    fn estimate_embedded(
        &self,
        interval: Interval<S>,
        nodes: &[FixedRuleNode<S>],
        sample: &dyn Fn(usize) -> S,
    ) -> Result<(S, IntegrationErrorComponents<S>), SciOxideError> {
        (**self).estimate_embedded(interval, nodes, sample)
    }
}
