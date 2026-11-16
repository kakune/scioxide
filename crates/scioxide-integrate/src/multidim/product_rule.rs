//! Tensor-product rules over boxes.
use crate::workspace::{
    Attempts, Family, IntegrationRequirements, Outcome, admit, limit, workspace_protocol,
};
use crate::{
    FixedRule, FixedRuleNode, IntegrationErrorComponents, IntegrationReport, IntegrationTermination,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Budget, BudgetKind, ExecutionContext,
    ExecutionPolicy, Interval, PartialResultTrust, PromotingCompensatedSum, PromotingProduct,
    RealScalar, SciOxideError, SolveBudgets, SolveScope, check_value_shape, write_value,
};

/// Options of [`ProductRule`]: an evaluation limit and an execution policy.
///
/// By default there is no evaluation limit and evaluations run sequentially.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::ProductOptions;
///
/// let options = ProductOptions::default();
/// assert_eq!(options.max_evaluations(), None);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ProductOptions {
    max_evaluations: Option<Budget>,
    policy: ExecutionPolicy,
}
impl ProductOptions {
    /// Sets or removes the limit on integrand evaluations per call.
    ///
    /// A call that needs more than the limit, `n^d` for a rule of `n` nodes in
    /// `d` dimensions, fails with `ConvergenceFailure` before any evaluation.
    pub const fn with_max_evaluations(mut self, value: Option<Budget>) -> Self {
        self.max_evaluations = value;
        self
    }
    /// Sets how the integrand evaluations of one call are distributed over
    /// workers. The result does not depend on the policy.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.policy = value;
        self
    }
    /// Returns the limit on integrand evaluations per call, if any.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }
    /// Returns the execution policy of integrand evaluations.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
}

/// Tensor product of a fixed one-dimensional rule over a box.
///
/// For a rule with nodes `xᵢ` and weights `wᵢ` on each interval `[a_k, b_k]`,
/// the plan computes
///
/// ```text
/// ∫ f(x) dx ≈ Σ_{i₁, …, i_d} w_{i₁} ⋯ w_{i_d} f(x_{i₁}, …, x_{i_d}),
/// ```
///
/// with `n^d` evaluations for `n` nodes in `d` dimensions. Nodes are visited in
/// row-major order: the last axis changes fastest. The rule is any
/// [`FixedRule`], such as a Gauss–Legendre or Gauss–Kronrod rule.
///
/// If the rule has embedded weights, as a Gauss–Kronrod pair does, the error
/// estimate is the difference between the primary and embedded tensor sums,
/// with a rounding floor of `50 ε Σ |w f|` (see
/// [Error estimates](crate#error-estimates)). Otherwise the result has no
/// error estimate. The call reports
/// [`IntegrationTermination::FixedRuleCompleted`], or
/// [`IntegrationTermination::DegenerateDomain`] with a zero result and no
/// evaluations if an axis has zero width. For `d = 0` the plan evaluates the
/// integrand once at the empty point.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_integrate::{GaussKronrod, GaussKronrodRule, ProductOptions, ProductRule};
///
/// let rule = GaussKronrod::new(GaussKronrodRule::G7K15)?;
/// let plan = ProductRule::new(rule, ProductOptions::default());
/// let bounds = [Interval::new(0.0_f64, 1.0)?; 2];
/// let mut workspace = plan.workspace((2, 2))?;
/// let mut value = [0.0; 2];
/// let f = |x: &[f64]| Ok([x[0] * x[1], x[0]]);
/// let cx = ExecutionContext::sequential_native();
/// let report = plan.integrate_into(&f, &bounds, &mut value, &mut workspace, cx)?;
/// assert!(report.is_success());
/// // ∫∫ x y = 1/4 and ∫∫ x = 1/2 over the unit square.
/// assert!((value[0] - 0.25).abs() < 1e-12 && (value[1] - 0.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed.,
/// Academic Press, 1984.
#[derive(Debug, Clone)]
pub struct ProductRule<S: RealScalar, I> {
    integrator: I,
    options: ProductOptions,
    scalar: PhantomData<S>,
}

/// Workspace of a [`ProductRule`]: the mapped nodes of each axis and a block of
/// evaluation points.
///
/// A workspace is made for one dimension and one number of value components by
/// [`ProductRule::workspace`]. It holds `O(n d + 64 (d + c))` scalars for `n`
/// nodes, `d` dimensions and `c` components. Any rule of the same dimension and
/// component count with at most as many nodes can use it.
#[derive(Debug)]
pub struct ProductWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    coordinates: Vec<S>,
    prefixes: Vec<[PromotingProduct<S>; 2]>,
    indices: Vec<usize>,
    axis_offsets: Vec<usize>,
    axis_counts: Vec<usize>,
    nodes: Vec<FixedRuleNode<S>>,
    points: Vec<S>,
    rows: Vec<S>,
    weights: Vec<[PromotingProduct<S>; 2]>,
    sums: Vec<[PromotingCompensatedSum<S>; 3]>,
    candidate: Vec<S>,
    errors: Vec<IntegrationErrorComponents<S>>,
    published: bool,
}
workspace_protocol!(ProductWorkspace, IntegrationReport<S>);
impl<S: RealScalar> ProductWorkspace<S> {
    fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let dimension = requirements.dimension();
        let components = requirements.components();
        let nodes = checked_mul(
            dimension,
            requirements.units()[0],
            scioxide_core::quantity!("tensor_nodes"),
        )?;
        let block = crate::evaluation::POINT_BLOCK;
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            coordinates: try_filled(dimension, S::ZERO, scioxide_core::quantity!("coordinates"))?,
            prefixes: try_filled(
                if dimension > 0 && requirements.units()[0] > 1 {
                    checked_add(dimension, 1, scioxide_core::quantity!("tensor_prefixes"))?
                } else {
                    0
                },
                [PromotingProduct::ONE; 2],
                scioxide_core::quantity!("tensor_prefixes"),
            )?,
            indices: try_filled(dimension, 0, scioxide_core::quantity!("indices"))?,
            axis_offsets: try_with_capacity(dimension, scioxide_core::quantity!("offsets"))?,
            axis_counts: try_with_capacity(dimension, scioxide_core::quantity!("counts"))?,
            nodes: try_with_capacity(nodes, scioxide_core::quantity!("nodes"))?,
            points: try_filled(
                checked_mul(dimension, block, scioxide_core::quantity!("tensor_points"))?,
                S::ZERO,
                scioxide_core::quantity!("tensor_points"),
            )?,
            rows: try_filled(
                checked_mul(components, block, scioxide_core::quantity!("tensor_rows"))?,
                S::ZERO,
                scioxide_core::quantity!("tensor_rows"),
            )?,
            weights: try_filled(
                block,
                [PromotingProduct::ONE; 2],
                scioxide_core::quantity!("tensor_weights"),
            )?,
            sums: try_filled(
                components,
                [PromotingCompensatedSum::ZERO; 3],
                scioxide_core::quantity!("tensor_sums"),
            )?,
            candidate: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("tensor_values"),
            )?,
            errors: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("tensor_errors"),
            )?,
            published: false,
        })
    }
    /// Returns the error estimate of each real component from the last
    /// successful call.
    ///
    /// Returns `None` if the last call failed, or if the rule has no embedded
    /// weights.
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.published.then_some(&self.errors)
    }
}
impl<S: RealScalar, I: FixedRule<S>> ProductRule<S, I> {
    /// Creates the tensor product of `integrator`.
    pub const fn new(integrator: I, options: ProductOptions) -> Self {
        Self {
            integrator,
            options,
            scalar: PhantomData,
        }
    }
    /// Returns the identifier of the method, recorded in solve diagnostics.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "ProductRule", 1)
    }
    /// Returns the options.
    pub const fn options(&self) -> &ProductOptions {
        &self.options
    }
    /// Returns the workspace size for `request = (d, c)`: `d` dimensions and an
    /// integrand with `c` real components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if the rule has no nodes and `d > 0`.
    /// - `SizeOverflow` if `n d` overflows `usize`.
    pub fn workspace_requirements(
        &self,
        request: (usize, usize),
    ) -> Result<IntegrationRequirements, SciOxideError> {
        let (dimension, components) = request;
        let nodes = self.integrator.node_count();
        if nodes == 0 && dimension > 0 {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("rule"),
                scioxide_core::InvalidInputReason::Empty,
                "fixed tensor rule is empty",
            ));
        }
        checked_mul(dimension, nodes, scioxide_core::quantity!("tensor_nodes"))?;
        Ok(
            IntegrationRequirements::new(Family::Product, components, [nodes, 0])
                .with_dimension(dimension),
        )
    }
    /// Allocates a workspace for `request = (d, c)`.
    ///
    /// # Errors
    ///
    /// - The errors of [`ProductRule::workspace_requirements`].
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, request: (usize, usize)) -> Result<ProductWorkspace<S>, SciOxideError> {
        ProductWorkspace::allocate(&self.workspace_requirements(request)?)
    }
    /// Integrates `f` over the box `bounds` and writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `scope` is an [`ExecutionContext`] or a solve scope (see
    /// [Execution contexts](scioxide_core#execution-contexts)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` has a different number of components than
    ///   the workspace.
    /// - `InvalidInput` (`Inconsistent`) if the workspace was made for another
    ///   dimension or for fewer nodes, or if the rule gives embedded weights
    ///   for some nodes but not others.
    /// - The errors of [`ProductRule::workspace_requirements`] and of the
    ///   rule's node mapping.
    /// - `NumericalInstability` (`Breakdown`) if rounding makes mapped nodes of
    ///   an axis coincide or move onto or outside the ends of the interval.
    /// - `SizeOverflow` if `n^d` overflows `usize`.
    /// - `ConvergenceFailure` if [`ProductOptions::max_evaluations`] is less
    ///   than `n^d`, and `ExecutionStopped` on cancellation or a deadline.
    /// - The errors of `f`, and `NonFinite` if it returns NaN or infinity (see
    ///   [Integrands](crate#integrands)).
    #[inline(always)]
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        bounds: &[Interval<S>],
        out: &mut O,
        ws: &mut ProductWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        ws.published = false;
        let components = ws.requirements.components();
        if let Err(error) = check_value_shape(out, components).and_then(|()| {
            let required = self.workspace_requirements((bounds.len(), components))?;
            if ws.check(&required)? == scioxide_core::Capacity::NeedsGrowth {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "insufficient product-rule shape or node capacity",
                ));
            }
            Ok(())
        }) {
            return ws.attempts.reject(error);
        }
        let mut attempts = core::mem::take(&mut ws.attempts);
        let result = attempts.integrate(
            scope.into(),
            self.algorithm_id(),
            bounds.len(),
            limit(
                SolveBudgets::unbounded(),
                BudgetKind::Evaluation,
                self.options.max_evaluations,
            ),
            |cx| {
                let degenerate = bounds.iter().any(|b| b.lower() == b.upper());
                if degenerate || (components == 0 && !bounds.is_empty()) {
                    ws.candidate.fill(S::ZERO);
                    cx.commit_accepted(PartialResultTrust::Admissible)?;
                    write_value(&ws.candidate, out)?;
                    return Ok(Outcome::Published {
                        termination: if degenerate {
                            IntegrationTermination::DegenerateDomain
                        } else {
                            IntegrationTermination::FixedRuleCompleted
                        },
                        subdivisions: 0,
                        errors: None,
                    });
                }
                ws.indices.fill(0);
                ws.axis_offsets.clear();
                ws.axis_counts.clear();
                ws.nodes.clear();
                if !preflight_metadata(&self.integrator, bounds, ws)? {
                    return Ok(Outcome::Unpublished(
                        IntegrationTermination::GeometryFailure,
                    ));
                }
                let count = (0..bounds.len()).try_fold(1usize, |n, _| {
                    checked_mul(
                        n,
                        self.integrator.node_count(),
                        scioxide_core::quantity!("tensor_cardinality"),
                    )
                })?;
                if let Some(stop) = admit(cx, count, false)? {
                    return Ok(Outcome::Unpublished(stop));
                }
                let embedded =
                    !ws.nodes.is_empty() && ws.nodes.iter().all(|n| n.embedded_weight().is_some());
                ws.sums.fill([PromotingCompensatedSum::ZERO; 3]);
                let block = crate::evaluation::POINT_BLOCK;
                let dimension = bounds.len();
                ws.prefixes.fill([PromotingProduct::ONE; 2]);
                let cached = !ws.prefixes.is_empty();
                let mut changed = 0;
                for first in (0..count).step_by(block) {
                    let len = block.min(count - first);
                    for lane in 0..len {
                        let mut weights = if cached {
                            ws.prefixes[changed]
                        } else {
                            [PromotingProduct::ONE; 2]
                        };
                        for axis in changed..dimension {
                            let node = ws.nodes[ws.axis_offsets[axis] + ws.indices[axis]];
                            ws.coordinates[axis] = node.point();
                            weights[0] = weights[0].multiplied(node.primary_weight())?;
                            if let Some(w) = node.embedded_weight() {
                                weights[1] = weights[1].multiplied(w)?;
                            }
                            if cached {
                                ws.prefixes[axis + 1] = weights;
                            }
                        }
                        ws.points[lane * dimension..(lane + 1) * dimension]
                            .copy_from_slice(&ws.coordinates);
                        ws.weights[lane] = weights;
                        changed = advance_indices(&mut ws.indices, &ws.axis_counts);
                        if !cached {
                            changed = 0;
                        }
                    }
                    crate::evaluation::sample_nd(
                        f,
                        &ws.points[..len * dimension],
                        dimension,
                        len,
                        components,
                        &mut ws.rows[..len * components],
                        self.options.policy,
                        cx,
                    )?;
                    let native_weights = ws.weights[..len]
                        .iter()
                        .all(|weights| weights.iter().all(|weight| weight.native().is_some()));
                    if native_weights {
                        let rows = &ws.rows;
                        for (c, sums) in ws.sums.iter_mut().enumerate() {
                            // Every weight was checked native above, so no term is skipped.
                            let terms = |which: usize, absolute: bool| {
                                ws.weights[..len].iter().enumerate().filter_map(
                                    move |(lane, weights)| {
                                        let weight = weights[which].native()?;
                                        let value = rows[lane * components + c];
                                        Some(if absolute {
                                            (weight.abs(), value.abs())
                                        } else {
                                            (weight, value)
                                        })
                                    },
                                )
                            };
                            sums[0]
                                .merge(PromotingCompensatedSum::from_products(terms(0, false))?)?;
                            if embedded {
                                sums[1].merge(PromotingCompensatedSum::from_products(terms(
                                    1, false,
                                ))?)?;
                                sums[2].merge(PromotingCompensatedSum::from_products(terms(
                                    0, true,
                                ))?)?;
                            }
                        }
                    } else {
                        for (lane, weights) in ws.weights[..len].iter().enumerate() {
                            for (c, sums) in ws.sums.iter_mut().enumerate() {
                                let value = ws.rows[lane * components + c];
                                sums[0].add_weighted(weights[0], value)?;
                                if embedded {
                                    sums[2].add_weighted(weights[0].abs(), value.abs())?;
                                    sums[1].add_weighted(weights[1], value)?;
                                }
                            }
                        }
                    }
                }
                for (c, sums) in ws.sums.iter().enumerate() {
                    ws.candidate[c] = sums[0].round()?;
                    if embedded {
                        ws.errors[c] = IntegrationErrorComponents::discretization_only(
                            sums[0].difference(sums[1])?.abs(),
                        )
                        .with_roundoff(crate::kernel::roundoff_floor(crate::kernel::diagnostic(
                            sums[2].round(),
                        )?)?);
                    }
                }
                cx.commit_accepted(PartialResultTrust::Admissible)?;
                write_value(&ws.candidate, out)?;
                ws.published = embedded;
                Ok(Outcome::Published {
                    termination: IntegrationTermination::FixedRuleCompleted,
                    subdivisions: 0,
                    errors: if embedded {
                        IntegrationErrorComponents::envelope(ws.errors.iter())
                    } else {
                        None
                    },
                })
            },
        );
        ws.attempts = attempts;
        result
    }
    /// Integrates a real, complex or fixed-size vector integrand over the box
    /// `bounds`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Allocates a workspace and runs on
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`ProductRule::workspace`] and
    ///   [`ProductRule::integrate_into`].
    pub fn integrate<F, V>(
        &self,
        f: F,
        bounds: &[Interval<S>],
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let f = &f;
        let mut out = V::default();
        let mut ws = self.workspace((bounds.len(), out.dimension()?))?;
        let report = self.integrate_into(
            f,
            bounds,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok((out, report))
    }
}
fn preflight_metadata<S: RealScalar, I: FixedRule<S> + ?Sized>(
    integrator: &I,
    bounds: &[Interval<S>],
    workspace: &mut ProductWorkspace<S>,
) -> Result<bool, SciOxideError> {
    let node_count = integrator.node_count();
    for &bound in bounds {
        workspace.axis_offsets.push(workspace.nodes.len());
        workspace.axis_counts.push(node_count);
        let mut validator = crate::fixed::Geometry::Interval {
            interval: bound,
            single_endpoint: integrator.permits_single_endpoint(),
        }
        .validator(node_count);
        let mut embedded = None;
        let first = workspace.nodes.len();
        if node_count > workspace.nodes.capacity().saturating_sub(first) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "mapped rule exceeds reserved node capacity",
            ));
        }
        workspace.nodes.resize(
            first + node_count,
            FixedRuleNode::new(S::ZERO, S::ZERO, None),
        );
        integrator.map_nodes_into(bound, &mut workspace.nodes[first..])?;
        for node in workspace.nodes[first..].iter().copied() {
            if !node.primary_weight().is_finite()
                || node
                    .embedded_weight()
                    .is_some_and(|weight| !weight.is_finite())
            {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Output,
                    "fixed rule exposed non-finite mapped metadata",
                ));
            }
            if !validator.push(node.point())? {
                return Ok(false);
            }
            let node_embedded = node.embedded_weight().is_some();
            if embedded.is_some_and(|value| value != node_embedded) {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("self.integrator"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "embedded weights must be present for either every node or no node",
                ));
            }
            embedded.get_or_insert(node_embedded);
        }
    }
    Ok(true)
}

// Mixed-radix increment with the last axis fastest (row-major order). After the final
// point it wraps back to all zeros. Return the first changed axis so only the
// changed suffix of the coordinate and weight prefixes needs preparation.
fn advance_indices(indices: &mut [usize], radices: &[usize]) -> usize {
    for axis in (0..indices.len()).rev() {
        indices[axis] += 1;
        if indices[axis] < radices[axis] {
            return axis;
        }
        indices[axis] = 0;
    }
    0
}
