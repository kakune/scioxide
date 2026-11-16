//! Finite differences on nodes you choose: Fornberg weights, the physical-node
//! plan and the relative-offset method value.

use crate::options::DifferenceShape;
use crate::samples::HessianSamples;
use crate::stencil::{bound_for, finish_scaled_steps, fornberg};
use crate::validation::{invalid, length, shape, validate_problem};
use crate::workspace;
use crate::{
    DerivativeOperation, DerivativeSource, DifferentiationReport, DifferentiationResult,
    SampleFunction,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AlgebraicScalar, Bounds, Capacity, DifferentiationLocation, Dimensions, ExecutionPolicy,
    InvalidInputReason, NonFiniteOrigin, NumericScalar, RangeAccumulator, RealScalar,
    SciOxideError, SolveBudgets, SolveState,
};
#[derive(Debug, Clone)]
pub(super) struct StencilScratch<S> {
    pub(super) normalized: Vec<S>,
    pub(super) ordered: Vec<S>,
    pub(super) permutation: Vec<usize>,
    pub(super) table: Vec<S>,
}
impl<S: RealScalar> StencilScratch<S> {
    pub(super) fn new(nodes: usize, order: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            normalized: try_filled(
                nodes,
                S::ZERO,
                scioxide_core::quantity!("stencil_normalized_nodes"),
            )?,
            ordered: try_filled(
                nodes,
                S::ZERO,
                scioxide_core::quantity!("stencil_ordered_nodes"),
            )?,
            permutation: try_filled(
                nodes,
                0,
                scioxide_core::quantity!("stencil_node_permutation"),
            )?,
            table: try_filled(
                checked_mul(
                    nodes,
                    checked_add(order, 1, scioxide_core::quantity!("fornberg_table"))?,
                    scioxide_core::quantity!("fornberg_table"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("fornberg_table"),
            )?,
        })
    }
}

/// Finite-difference weights for the derivative of order `k` at a point `c`
/// from values at arbitrary distinct nodes `x₀, …, xₙ₋₁`.
///
/// ```text
/// f⁽ᵏ⁾(c) ≈ s⁻ᵏ Σⱼ wⱼ f(xⱼ),   s = maxⱼ |xⱼ − c|,
/// ```
///
/// where `wⱼ` are the weights for the normalized displacements `(xⱼ − c)/s`.
/// The normalization keeps the weights of order one for any node spacing. The
/// formula is exact for polynomials of degree below `n`; the nodes may be
/// unequally spaced and lie on one side of `c`. Order zero gives interpolation.
/// Fornberg's recurrence computes the weights in `O(n²(k + 1))` operations;
/// applying them costs `O(n)`. [`weights`](Self::weights) returns the weights
/// `wⱼ/sᵏ` of the formula itself, and [`apply`](Self::apply) forms the sum
/// accurately.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::FiniteDifferenceStencil;
///
/// // Three nodes are exact for quadratics: d/dx x² = 1 at x = 0.5.
/// let stencil = FiniteDifferenceStencil::new(0.5_f64, &[0.0, 0.5, 1.0], 1)?;
/// let derivative = stencil.apply(&[0.0_f64, 0.25, 1.0])?;
/// assert!((derivative - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// B. Fornberg, "Generation of finite difference formulas on arbitrarily spaced
/// grids", *Math. Comp.* 51(184), 1988, 699–706.
#[derive(Debug, Clone)]
pub struct FiniteDifferenceStencil<S: RealScalar> {
    pub(super) coordinate: S,
    pub(super) nodes: Vec<S>,
    pub(super) weights: Vec<S>,
    pub(super) scale: S,
    pub(super) order: usize,
    pub(super) amplification: S,
    pub(super) base_indices: (usize, usize),
}
impl<S: RealScalar> FiniteDifferenceStencil<S> {
    /// Computes the weights for the derivative of order `order` at
    /// `coordinate` from values at `nodes`, in the given node order.
    ///
    /// For a positive order, constant data gives exactly zero.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `coordinate` or a node is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `nodes` is empty or `order` is not
    ///   below the number of nodes.
    /// - `InvalidInput` (`Inconsistent`) if two nodes are equal.
    /// - `NumericalInstability` if the recurrence overflows.
    /// - `AllocationFailure` if the weights cannot be stored.
    pub fn new(coordinate: S, nodes: &[S], order: usize) -> Result<Self, SciOxideError> {
        Self::prepare(coordinate, nodes, order, scioxide_core::argument!("nodes"))
    }
    // Shared preparation; `argument` names the caller's node parameter.
    pub(super) fn prepare(
        coordinate: S,
        nodes: &[S],
        order: usize,
        argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        if !coordinate.is_finite() || nodes.iter().any(|v| !v.is_finite()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "coordinate and nodes must be finite",
            ));
        }
        if nodes.is_empty() || order >= nodes.len() {
            return Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::OutOfRange,
                "require nonempty nodes and derivative order below node count",
            ));
        }
        if nodes
            .iter()
            .enumerate()
            .any(|(i, v)| nodes[..i].contains(v))
        {
            return Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::Inconsistent,
                "physical nodes must be distinct",
            ));
        }
        let mut stencil = Self {
            coordinate,
            nodes: scioxide_core::backend::try_copy_slice(
                nodes,
                scioxide_core::quantity!("stencil_nodes"),
            )?,
            weights: try_filled(
                nodes.len(),
                S::ZERO,
                scioxide_core::quantity!("stencil_weights"),
            )?,
            scale: S::ONE,
            order,
            amplification: S::ZERO,
            base_indices: (0, 0),
        };
        let mut scratch = StencilScratch::new(nodes.len(), order)?;
        stencil.reweight(&mut scratch)?;
        Ok(stencil)
    }
    fn reweight(&mut self, scratch: &mut StencilScratch<S>) -> Result<(), SciOxideError> {
        let nodes = &self.nodes;
        let coordinate = self.coordinate;
        let order = self.order;
        let normalized = &mut scratch.normalized[..nodes.len()];
        for (offset, &node) in normalized.iter_mut().zip(nodes) {
            *offset = scioxide_core::checked_difference(node, coordinate)?;
        }
        let scale = normalized.iter().fold(S::ZERO, |a, b| a.max(b.abs()));
        // All displacements are zero only for a single node at the coordinate,
        // which can only be an order-zero stencil; any nonzero scale would do.
        let scale = if scale == S::ZERO { S::ONE } else { scale };
        for v in normalized.iter_mut() {
            *v /= scale;
        }
        let width = checked_add(order, 1, scioxide_core::quantity!("fornberg_table"))?;
        let table = &mut scratch.table[..checked_mul(
            nodes.len(),
            width,
            scioxide_core::quantity!("fornberg_table"),
        )?];
        // Insert the nearest nodes first to reduce intermediate growth on clustered grids.
        let permutation = &mut scratch.permutation[..nodes.len()];
        for (i, index) in permutation.iter_mut().enumerate() {
            *index = i;
        }
        permutation.sort_unstable_by(|&a, &b| {
            normalized[a]
                .abs()
                .partial_cmp(&normalized[b].abs())
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let ordered = &mut scratch.ordered[..nodes.len()];
        for (v, &index) in ordered.iter_mut().zip(permutation.iter()) {
            *v = normalized[index];
        }
        fornberg(ordered, order, table)?;
        let weights = &mut self.weights;
        for (i, &index) in permutation.iter().enumerate() {
            weights[index] = table[i * width + order];
        }
        // Exact reflection symmetry determines weight parity. Enforce it when
        // present to avoid recurrence roundoff breaking odd/even cancellation.
        if normalized.iter().all(|&v| normalized.contains(&-v)) {
            for i in 0..nodes.len() {
                if let Some(j) = normalized.iter().position(|&v| v == -normalized[i])
                    && j >= i
                {
                    let sign = if order.is_multiple_of(2) {
                        S::ONE
                    } else {
                        -S::ONE
                    };
                    let weight = scioxide_core::checked_midpoint(weights[i], sign * weights[j])?;
                    weights[i] = weight;
                    weights[j] = sign * weight;
                }
            }
        }
        let mut sum = RangeAccumulator::new();
        for weight in weights.iter() {
            sum.add(weight.abs())?;
        }
        // `apply_lane` subtracts the midpoint of the nearest sample and its mirror
        // image. For symmetric stencils this base is itself symmetric, so it keeps
        // the enforced weight parity; without a mirror node it is the nearest
        // sample alone.
        let nearest = permutation[0];
        let reflected = normalized
            .iter()
            .position(|&v| v == -normalized[nearest])
            .unwrap_or(nearest);
        self.scale = scale;
        self.amplification = sum.finish()?;
        self.base_indices = (nearest, reflected);
        Ok(())
    }
    pub(super) fn rebind(
        &mut self,
        coordinate: S,
        offsets: &[S],
        bound: Option<(Option<S>, Option<S>)>,
        scratch: &mut StencilScratch<S>,
    ) -> Result<(), SciOxideError> {
        if self.nodes.len() != offsets.len() {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "relative stencil capacity mismatch",
            ));
        }
        self.coordinate = coordinate;
        for (node, &offset) in self.nodes.iter_mut().zip(offsets) {
            *node = coordinate + offset;
            if !node.is_finite()
                || (offset != S::ZERO && *node == coordinate)
                || bound.is_some_and(|(lower, upper)| {
                    lower.is_some_and(|v| *node < v) || upper.is_some_and(|v| *node > v)
                })
            {
                return Err(invalid(
                    scioxide_core::argument!("x"),
                    "relative nodes must be finite, in bounds, and retain nonzero offsets",
                ));
            }
        }
        if self
            .nodes
            .iter()
            .enumerate()
            .any(|(i, node)| self.nodes[..i].contains(node))
        {
            return Err(invalid(
                scioxide_core::argument!("x"),
                "relative physical nodes collapsed",
            ));
        }
        self.reweight(scratch)
    }
    /// Returns the point `c` at which the derivative is taken.
    pub const fn coordinate(&self) -> S {
        self.coordinate
    }
    /// Returns the nodes, in the order of the values passed to
    /// [`apply`](Self::apply).
    pub fn nodes(&self) -> &[S] {
        &self.nodes
    }
    /// Returns the weights `wⱼ` for the normalized displacements
    /// `(xⱼ − c)/s`.
    pub fn normalized_weights(&self) -> &[S] {
        &self.weights
    }
    /// Returns the scale `s = maxⱼ |xⱼ − c|`; the derivative is the weighted sum
    /// divided by `sᵏ`.
    pub const fn scale(&self) -> S {
        self.scale
    }
    /// Returns the weights `wⱼ/sᵏ` of the formula, so that
    /// `f⁽ᵏ⁾(c) ≈ Σⱼ weights[j]·f(xⱼ)`, in node order.
    ///
    /// Each weight is divided by `sᵏ` once, without an intermediate rounding of
    /// `sᵏ`. A weight that is zero by symmetry, such as the centre weight of a
    /// central first difference, is exactly zero. Use [`apply`](Self::apply)
    /// to form the sum, which subtracts a base value first and accumulates
    /// exactly.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a weight overflows or underflows to zero.
    /// - `AllocationFailure` if the weights cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_differentiate::FiniteDifferenceStencil;
    ///
    /// // The central difference (f(c + h) − f(c − h)) / 2h with h = 0.5.
    /// let stencil = FiniteDifferenceStencil::new(0.0_f64, &[-0.5, 0.0, 0.5], 1)?;
    /// assert_eq!(stencil.weights()?, [-1.0, 0.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn weights(&self) -> Result<Vec<S>, SciOxideError> {
        let mut out = try_filled(
            self.nodes.len(),
            S::ZERO,
            scioxide_core::quantity!("stencil_weights"),
        )?;
        self.weights_into(&mut out)?;
        Ok(out)
    }
    /// Writes the weights `wⱼ/sᵏ` of the formula into `out`, as
    /// [`weights`](Self::weights) without allocating.
    ///
    /// Every weight is computed before the first is written, so `out` is
    /// unchanged on error.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` does not have one entry per node.
    /// - `NumericalInstability` if a weight overflows or underflows to zero.
    pub fn weights_into(&self, out: &mut [S]) -> Result<(), SciOxideError> {
        length(self.nodes.len(), out.len())?;
        let scaled = |weight: S| {
            let mut sum = RangeAccumulator::new();
            sum.add(weight)?;
            finish_scaled_steps(&sum, &[(self.scale, self.order)])
        };
        for &weight in &self.weights {
            scaled(weight)?;
        }
        for (slot, &weight) in out.iter_mut().zip(&self.weights) {
            *slot = scaled(weight)?;
        }
        Ok(())
    }
    /// Returns the derivative order `k`; zero means interpolation.
    pub const fn derivative_order(&self) -> usize {
        self.order
    }
    /// Returns `Σⱼ |wⱼ|`.
    ///
    /// Errors of size `δ` in the values change the derivative by at most
    /// `δ · Σⱼ |wⱼ| / sᵏ`, so this measures how much the stencil amplifies
    /// rounding errors and noise. It is not a bound on the total error.
    pub const fn weight_amplification(&self) -> S {
        self.amplification
    }
    /// Returns `s⁻ᵏ Σⱼ wⱼ values[j]`, the derivative from real or complex values
    /// at the nodes.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values` does not have one entry per node.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `NumericalInstability` if the result overflows or underflows to zero.
    pub fn apply<T: NumericScalar<Real = S>>(&self, values: &[T]) -> Result<T, SciOxideError> {
        if values.len() != self.nodes.len() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.nodes.len()),
                Dimensions::one(values.len()),
            ));
        }
        self.apply_lane_reusing(
            |i| values[i],
            &mut RangeAccumulator::new(),
            &mut RangeAccumulator::new(),
        )
    }
    pub(super) fn apply_lane_reusing<T: NumericScalar<Real = S>>(
        &self,
        value: impl Fn(usize) -> T,
        real: &mut RangeAccumulator<S>,
        imag: &mut RangeAccumulator<S>,
    ) -> Result<T, SciOxideError> {
        // Positive-order weights sum to zero mathematically, but rounded weights do
        // not. Subtracting a base sample makes constant data annihilate exactly and
        // confines weight rounding to the small differences v_i - base. The
        // accumulators are exact, so the extra products cost no accuracy.
        // Interpolation weights sum to one and must not be shifted.
        let base = if self.order > 0 {
            let left = value(self.base_indices.0);
            let right = value(self.base_indices.1);
            T::from_components(
                scioxide_core::checked_midpoint(left.real_component(), right.real_component())?,
                scioxide_core::checked_midpoint(
                    left.imaginary_component(),
                    right.imaginary_component(),
                )?,
            )
        } else {
            T::ZERO
        };
        real.clear();
        imag.clear();
        for (i, &weight) in self.weights.iter().enumerate() {
            let v = value(i);
            real.add_product(weight, v.real_component())?;
            real.subtract_product(weight, base.real_component())?;
            imag.add_product(weight, v.imaginary_component())?;
            imag.subtract_product(weight, base.imaginary_component())?;
        }
        Ok(T::from_components(
            finish_scaled_steps(real, &[(self.scale, self.order)])?,
            finish_scaled_steps(imag, &[(self.scale, self.order)])?,
        ))
    }
}

/// Computes derivatives at a fixed point from function values at nodes you
/// choose.
///
/// The plan is prepared at `point` with one node list per coordinate.
/// Construction computes the [`FiniteDifferenceStencil`] weights of each
/// coordinate and derivative order, `O(k²)` operations for `k` nodes, and every
/// evaluation at `point` reads them; for a changing point use
/// [`RelativeDifference`]. Each entry is exact for polynomials of degree below
/// the number of nodes of its coordinates, and its truncation error is of order
/// `h^(k−d)` for a derivative of order `d` from `k` nodes of spacing `h`.
/// Callback values may be real or complex. An entry along one coordinate costs
/// one evaluation per node, a mixed Hessian entry the product of the node
/// counts; within a Hessian evaluation, bit-identical points are evaluated
/// once. No error estimate is returned.
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::{DerivativeOperation, NodeDifferenceOptions, PhysicalDifferencePlan};
///
/// // Three nodes are exact for quadratics: d/dx x² = 1 at x = 0.5.
/// let (nodes, options) = ([0.0, 0.5, 1.0], NodeDifferenceOptions::default());
/// let plan = PhysicalDifferencePlan::new(DerivativeOperation::First, &[0.5], &[&nodes], 1, options)?;
/// let f = |x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0];
///     Ok(())
/// };
/// let result = plan.evaluate(f, None)?;
/// assert_eq!(result.values(), Some(&[1.0][..]));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// B. Fornberg, "Generation of finite difference formulas on arbitrarily
/// spaced grids", *Math. Comp.* 51(184), 1988, 699–706.
#[derive(Debug, Clone)]
pub struct PhysicalDifferencePlan<S: RealScalar> {
    operation: DifferenceShape,
    point: Vec<S>,
    first: Vec<FiniteDifferenceStencil<S>>,
    second: Vec<FiniteDifferenceStencil<S>>,
    outputs: usize,
    len: usize,
    max_nodes: usize,
    options: NodeDifferenceOptions,
}
/// Callback samples and the staged result, written to the output only when
/// complete.
#[derive(Debug, Clone)]
struct PhysicalBuffers<T: NumericScalar> {
    point: Vec<T::Real>,
    samples: Vec<T>,
    intermediate: Vec<T>,
    staged: Vec<T>,
    retained: HessianSamples<T>,
    reductions: Vec<[RangeAccumulator<T::Real>; 2]>,
}
impl<S: RealScalar> PhysicalDifferencePlan<S> {
    /// Prepares a plan for `operation` at `point`, with one node list per
    /// coordinate in `nodes` and `outputs` callback values.
    ///
    /// For a Jacobian-vector product use
    /// [`AdaptiveDifference::jvp`](crate::AdaptiveDifference::jvp) or
    /// [`ComplexStep::jvp`](crate::ComplexStep::jvp), or differentiate
    /// `t ↦ f(x + t v)` with this plan.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `nodes` does not have one list per coordinate.
    /// - `NonFinite` with origin `Input` if `point` or a node is NaN or infinite.
    /// - `InvalidInput` (`operation`, `UnsupportedConfiguration`) if `operation`
    ///   does not match the number of coordinates and `outputs`.
    /// - `InvalidInput` (`nodes`, `OutOfRange`) if a node list has too few nodes
    ///   for the derivative order (two for first, three for second
    ///   derivatives), and (`nodes`, `Inconsistent`) if two nodes are equal.
    /// - `SizeOverflow` (`derivative_entries`) if the output length does not
    ///   fit in `usize`.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(
        operation: DerivativeOperation,
        point: &[S],
        nodes: &[&[S]],
        outputs: usize,
        options: NodeDifferenceOptions,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(
            DifferenceShape::from(operation),
            point,
            nodes,
            outputs,
            scioxide_core::argument!("nodes"),
            options,
        )
    }
    // Shared preparation;
    pub(super) fn prepare(
        kind: DifferenceShape,
        point: &[S],
        nodes: &[&[S]],
        outputs: usize,
        nodes_argument: scioxide_core::Argument,
        options: NodeDifferenceOptions,
    ) -> Result<Self, SciOxideError> {
        crate::validation::length(point.len(), nodes.len())?;
        validate_problem(point, None, scioxide_core::argument!("point"))?;
        let len = shape(kind, point.len(), outputs)?;
        let mut first = scioxide_core::backend::try_with_capacity(
            point.len(),
            scioxide_core::quantity!("first_derivative_stencils"),
        )?;
        let mut second = scioxide_core::backend::try_with_capacity(
            if matches!(kind, DifferenceShape::Second | DifferenceShape::Hessian) {
                point.len()
            } else {
                0
            },
            scioxide_core::quantity!("second_derivative_stencils"),
        )?;
        let mut max_nodes = 0;
        for (&coordinate, &axis) in point.iter().zip(nodes) {
            first.push(FiniteDifferenceStencil::prepare(
                coordinate,
                axis,
                1,
                nodes_argument,
            )?);
            if matches!(kind, DifferenceShape::Second | DifferenceShape::Hessian) {
                second.push(FiniteDifferenceStencil::prepare(
                    coordinate,
                    axis,
                    2,
                    nodes_argument,
                )?);
            }
            max_nodes = max_nodes.max(axis.len());
        }
        Ok(Self {
            operation: kind,
            point: scioxide_core::backend::try_copy_slice(
                point,
                scioxide_core::quantity!("plan_point"),
            )?,
            first,
            second,
            outputs,
            len,
            max_nodes,
            options,
        })
    }
    /// Returns the number of output entries.
    pub const fn output_len(&self) -> usize {
        self.len
    }
    /// Returns the workspace size this plan needs for callback values of type
    /// `T`, without allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements<T: NumericScalar<Real = S>>(
        &self,
        (): (),
    ) -> Result<PhysicalDifferenceRequirements<T>, SciOxideError> {
        let nodes = if self.operation == DifferenceShape::Hessian {
            checked_mul(
                self.max_nodes,
                self.max_nodes,
                scioxide_core::quantity!("callback_samples"),
            )?
        } else {
            self.max_nodes
        };
        let mut required = PhysicalDifferenceRequirements {
            lengths: [
                self.point.len(),
                checked_mul(
                    nodes,
                    self.outputs,
                    scioxide_core::quantity!("callback_samples"),
                )?,
                self.max_nodes,
                self.len,
                usize::from(self.len != 0),
            ],
            axis_nodes: if self.operation == DifferenceShape::Hessian {
                self.max_nodes
            } else {
                0
            },
            bytes: 0,
            limit: self.options.max_workspace_bytes,
            value: core::marker::PhantomData,
        };
        required.bytes = PhysicalBuffers::<T>::default().bytes(Some(&required))?;
        workspace::memory_limit(required.bytes, required.limit)?;
        Ok(required)
    }

    /// Allocates a workspace for callback values of type `T`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
        request: (),
    ) -> Result<PhysicalDifferenceWorkspace<T>, SciOxideError> {
        let () = request;
        let mut workspace = PhysicalDifferenceWorkspace::default();
        workspace.try_reserve(&self.workspace_requirements(())?)?;
        Ok(workspace)
    }

    fn validate(&self, bounds: Option<&Bounds<S>>) -> Result<(), SciOxideError> {
        validate_problem(&self.point, bounds, scioxide_core::argument!("bounds"))?;
        for (axis, stencil) in self.first.iter().enumerate() {
            if let Some((lower, upper)) = bound_for(bounds, axis)
                && stencil
                    .nodes
                    .iter()
                    .any(|&node| lower.is_some_and(|v| node < v) || upper.is_some_and(|v| node > v))
            {
                return Err(invalid(
                    scioxide_core::argument!("bounds"),
                    "physical nodes must lie within bounds",
                ));
            }
        }
        Ok(())
    }
    /// Computes the derivative at the prepared point and returns it with the
    /// report.
    ///
    /// `bounds`, when given, must contain the point and every node. The
    /// evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`evaluate_into`](Self::evaluate_into), except those of
    ///   `out` and `workspace`.
    /// - `AllocationFailure` if the workspace or result cannot be allocated.
    pub fn evaluate<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.validate(bounds)?;
        let mut workspace = self.workspace(())?;
        let mut output = try_filled(
            self.len,
            T::ZERO,
            scioxide_core::quantity!("derivative_output"),
        )?;
        let report = self.evaluate_validated(
            function,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_evaluation(report, output))
    }
    /// Computes the derivative at the prepared point into `out`, using
    /// `workspace` for scratch.
    ///
    /// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) or a
    /// child [`SolveScope`](scioxide_core::SolveScope); [`Parallel`](crate::Parallel)
    /// callbacks run on its executor. With a sufficient workspace the
    /// evaluation does not allocate. `out` is written only if the report has
    /// output available.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds` does not have one entry per
    ///   coordinate, `out` does not have [`output_len`](Self::output_len)
    ///   entries or `workspace` is too small.
    /// - `InvalidInput` (`bounds`, `OutOfRange` or `UnsupportedConfiguration`)
    ///   if the point or a node lies outside `bounds`.
    /// - `DifferentiationFailure` whose `source()` is the error returned by
    ///   `function`, `NonFinite` (origin `Callback`) for a NaN or infinite value
    ///   of `function`, or `NumericalInstability` for a sum outside the finite
    ///   range; it records the sample location and the evaluation count.
    /// - `ResourceLimit` if the workspace exceeds `max_workspace_bytes`.
    pub fn evaluate_into<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        workspace: &mut PhysicalDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate(bounds)?;
        self.evaluate_validated(function, out, workspace, scope)
    }
    fn evaluate_validated<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        output: &mut [T],
        workspace: &mut PhysicalDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        crate::validation::length(self.len, output.len())?;
        workspace.prepare(&self.workspace_requirements(())?)?;
        let identity = node_identity(
            scioxide_core::algorithm_id!("differentiate", "physical", 1),
            self.len,
        );
        scope.into().run(
            &mut workspace.state,
            identity,
            self.options.budgets,
            |context| {
                self.evaluate_geometry(
                    &self.point,
                    &self.first,
                    &self.second,
                    function,
                    output,
                    &mut workspace.data,
                    context,
                )
            },
        )
    }
    // Shared physical/relative evaluation.
    #[expect(
        clippy::too_many_arguments,
        reason = "numerical inputs, caller-owned staging and execution capabilities stay explicit"
    )]
    fn evaluate_geometry<T: NumericScalar<Real = S>>(
        &self,
        point: &[S],
        first_stencils: &[FiniteDifferenceStencil<S>],
        second_stencils: &[FiniteDifferenceStencil<S>],
        mut function: impl SampleFunction<S, T>,
        output: &mut [T],
        workspace: &mut PhysicalBuffers<T>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        let n = point.len();
        let sample_nodes = if self.operation == DifferenceShape::Hessian {
            checked_mul(
                self.max_nodes,
                self.max_nodes,
                scioxide_core::quantity!("callback_samples"),
            )?
        } else {
            self.max_nodes
        };
        if workspace.staged.len() != self.len
            || workspace.point.len() != n
            || workspace.samples.len()
                < checked_mul(
                    sample_nodes,
                    self.outputs,
                    scioxide_core::quantity!("callback_samples"),
                )?
            || workspace.intermediate.len() < self.max_nodes
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "incompatible workspace state",
            ));
        }
        let mut report =
            DifferentiationReport::fixed(DerivativeSource::NodeDifference, output.len());
        workspace.retained.clear();
        workspace.point.copy_from_slice(point);
        if self.len == 0 {
            report.poll(context)?;
            return Ok(report);
        }
        for i in 0..n {
            let pairs = if self.operation == DifferenceShape::Hessian {
                n - i
            } else {
                1
            };
            for pair in 0..pairs {
                if !report.iteration(context)? {
                    return Ok(report);
                }
                // Hessians visit only j >= i and mirror the result. The diagonal entry
                // (j == i) uses the second-derivative stencil, and mixed entries use the
                // tensor product of the two first-derivative stencils. Samples are stored
                // as [(a * inner + b) * outputs + lane].
                let j = i + pair;
                workspace.retained.clear_pair();
                let mixed = self.operation == DifferenceShape::Hessian && i != j;
                let first = if !mixed
                    && matches!(
                        self.operation,
                        DifferenceShape::Second | DifferenceShape::Hessian
                    ) {
                    &second_stencils[i]
                } else {
                    &first_stencils[i]
                };
                let inner = if mixed {
                    first_stencils[j].nodes.len()
                } else {
                    1
                };
                for (a, &node) in first.nodes.iter().enumerate() {
                    workspace.point[i] = node;
                    for b in 0..inner {
                        if mixed {
                            workspace.point[j] = first_stencils[j].nodes[b];
                        }
                        let start = (a * inner + b) * self.outputs;
                        let retain = self.operation == DifferenceShape::Hessian
                            && (!mixed
                                || crate::samples::same_bits(workspace.point[i], point[i])
                                || crate::samples::same_bits(workspace.point[j], point[j]));
                        if retain
                            && let Some(values) =
                                workspace.retained.find(point, &workspace.point)?
                        {
                            if !report.poll(context)? {
                                return Ok(report);
                            }
                            workspace.samples[start] = values;
                            continue;
                        }
                        if !report.invoke(
                            &workspace.point,
                            &mut workspace.samples[start..start + self.outputs],
                            &mut function,
                            if mixed {
                                DifferentiationLocation::variables(i, j)
                            } else {
                                DifferentiationLocation::variable(i)
                            },
                            context,
                            self.options.execution,
                        )? {
                            return Ok(report);
                        }
                        if retain {
                            workspace.retained.insert(
                                point,
                                &workspace.point,
                                workspace.samples[start],
                            )?;
                        }
                    }
                }
                workspace.point[i] = point[i];
                workspace.point[j] = point[j];
                for lane in 0..self.outputs {
                    let derivative = if mixed {
                        // The tensor weights are separable, so contract along j for every i-node
                        // first and then along i. This costs O(n_i n_j) without forming the
                        // product weights.
                        for a in 0..first.nodes.len() {
                            let [real, imag] = &mut workspace.reductions[0];
                            workspace.intermediate[a] = first_stencils[j].apply_lane_reusing(
                                |b| workspace.samples[(a * inner + b) * self.outputs + lane],
                                real,
                                imag,
                            )?;
                        }
                        let [real, imag] = &mut workspace.reductions[0];
                        first.apply_lane_reusing(|a| workspace.intermediate[a], real, imag)?
                    } else {
                        let [real, imag] = &mut workspace.reductions[0];
                        first.apply_lane_reusing(
                            |a| workspace.samples[a * self.outputs + lane],
                            real,
                            imag,
                        )?
                    };
                    if self.operation == DifferenceShape::Hessian {
                        workspace.staged[i * n + j] = derivative;
                        workspace.staged[j * n + i] = derivative;
                        report.completed_entries += if i == j { 1 } else { 2 };
                    } else {
                        workspace.staged[lane * n + i] = derivative;
                        report.completed_entries += 1;
                    }
                }
            }
        }
        // A stop observed after the last callback still suppresses publication, so
        // the caller's output is either fully updated or untouched.
        if report.poll(context)? {
            output.copy_from_slice(&workspace.staged);
        }
        Ok(report)
    }
}
/// Computes derivatives at any point from function values at `x + offsets`.
///
/// The method value holds one offset list per coordinate. At each evaluation
/// the nodes `xᵢ + offsets[i][k]` are formed and the [`FiniteDifferenceStencil`]
/// weights are recomputed from the displacements that remain after rounding,
/// `O(k²)` operations per coordinate and call for `k` offsets, without
/// allocating. An offset of zero is `xᵢ` itself. Accuracy, cost and callback
/// values are as in [`PhysicalDifferencePlan`].
///
/// # Examples
///
/// ```
/// use scioxide_differentiate::{DerivativeOperation, NodeDifferenceOptions, RelativeDifference};
///
/// // Three offsets are exact for quadratics: d/dx x² = 4 at x = 2.
/// let (offsets, options) = ([-0.5, 0.0, 0.5], NodeDifferenceOptions::default());
/// let method = RelativeDifference::new(DerivativeOperation::First, &[&offsets], 1, options)?;
/// let f = |x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0];
///     Ok(())
/// };
/// let result = method.evaluate(f, &[2.0], None)?;
/// assert_eq!(result.values(), Some(&[4.0][..]));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RelativeDifference<S: RealScalar> {
    template: PhysicalDifferencePlan<S>,
}
/// Node geometry, recurrence scratch, samples and the staged result.
#[derive(Debug, Clone)]
struct RelativeBuffers<T: NumericScalar> {
    first: Vec<FiniteDifferenceStencil<T::Real>>,
    second: Vec<FiniteDifferenceStencil<T::Real>>,
    scratch: StencilScratch<T::Real>,
    samples: PhysicalBuffers<T>,
}
impl<S: RealScalar> RelativeDifference<S> {
    /// Binds `operation`, one offset list per coordinate in `offsets` and
    /// `outputs` callback values.
    ///
    /// # Errors
    ///
    /// - `NonFinite` with origin `Input` if an offset is NaN or infinite.
    /// - `InvalidInput` (`operation`, `UnsupportedConfiguration`) if `operation`
    ///   does not match the number of coordinates and `outputs`.
    /// - `InvalidInput` (`offsets`, `OutOfRange`) if an offset list has too few
    ///   offsets for the derivative order (two for first, three for second
    ///   derivatives), and (`offsets`, `Inconsistent`) if two offsets are
    ///   equal.
    /// - `SizeOverflow` (`derivative_entries`) if the output length does not
    ///   fit in `usize`.
    /// - `AllocationFailure` if the offsets cannot be stored.
    pub fn new(
        operation: DerivativeOperation,
        offsets: &[&[S]],
        outputs: usize,
        options: NodeDifferenceOptions,
    ) -> Result<Self, SciOxideError> {
        let origin = try_filled(
            offsets.len(),
            S::ZERO,
            scioxide_core::quantity!("relative_origin"),
        )?;
        Ok(Self {
            template: PhysicalDifferencePlan::prepare(
                DifferenceShape::from(operation),
                &origin,
                offsets,
                outputs,
                scioxide_core::argument!("offsets"),
                options,
            )?,
        })
    }
    /// Returns the number of output entries.
    pub const fn output_len(&self) -> usize {
        self.template.output_len()
    }
    /// Returns the workspace size of an evaluation with callback values of
    /// type `T`, without allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements<T: NumericScalar<Real = S>>(
        &self,
        (): (),
    ) -> Result<RelativeDifferenceRequirements<'_, T>, SciOxideError> {
        let mut required = RelativeDifferenceRequirements {
            plan: self,
            physical: self.template.workspace_requirements(())?,
            bytes: 0,
        };
        required.bytes = RelativeBuffers::<T>::default().bytes(Some(&required))?;
        workspace::memory_limit(required.bytes, self.template.options.max_workspace_bytes)?;
        Ok(required)
    }

    /// Allocates a workspace for callback values of type `T`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
        request: (),
    ) -> Result<RelativeDifferenceWorkspace<T>, SciOxideError> {
        let () = request;
        let mut workspace = RelativeDifferenceWorkspace::default();
        workspace.try_reserve(&self.workspace_requirements(())?)?;
        Ok(workspace)
    }

    /// Computes the derivative at `x` into `out`, using `workspace` for
    /// scratch.
    ///
    /// `scope` is as in [`PhysicalDifferencePlan::evaluate_into`]. The nodes are
    /// formed and checked before the first callback call. With a sufficient
    /// workspace the evaluation does not allocate. `out` is written only if the
    /// report has output available.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `bounds` does not have one entry per
    ///   coordinate, `out` does not have [`output_len`](Self::output_len)
    ///   entries or `workspace` is too small.
    /// - `NonFinite` (origin `Input`) if `x` contains NaN or infinity.
    /// - `InvalidInput` (`x`, `OutOfRange`) if `x` lies outside `bounds`.
    /// - `InvalidInput` (`x`, `UnsupportedConfiguration`) if a node
    ///   `xᵢ + offsets[i][k]` is not finite, lies outside `bounds`, or rounds
    ///   onto another node.
    /// - The errors of [`PhysicalDifferencePlan::evaluate_into`] from the
    ///   callback and the arithmetic.
    pub fn evaluate_into<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        workspace: &mut RelativeDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate(x, bounds)?;
        self.evaluate_validated(x, bounds, function, out, workspace, scope)
    }

    fn evaluate_validated<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        point: &[S],
        bounds: Option<&Bounds<S>>,
        function: F,
        output: &mut [T],
        workspace: &mut RelativeDifferenceWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        crate::validation::length(self.output_len(), output.len())?;
        workspace.prepare(&self.workspace_requirements(())?)?;
        scope.into().run(
            &mut workspace.state,
            node_identity(
                scioxide_core::algorithm_id!("differentiate", "relative", 1),
                self.output_len(),
            ),
            self.template.options.budgets,
            |context| {
                self.evaluate_geometry(
                    point,
                    bounds,
                    function,
                    output,
                    &mut workspace.data,
                    context,
                )
            },
        )
    }
    /// Forms the nodes at `point` and evaluates under the method's budgets.
    /// Invalid geometry fails before any callback call and leaves `output`
    /// unchanged.
    /// # Errors
    /// Returns point, geometry, workspace, callback or numerical errors.
    fn evaluate_geometry<T: NumericScalar<Real = S>>(
        &self,
        point: &[S],
        bounds: Option<&Bounds<S>>,
        function: impl SampleFunction<S, T>,
        output: &mut [T],
        workspace: &mut RelativeBuffers<T>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        if workspace.first.len() < point.len()
            || workspace.second.len() < self.template.second.len()
            || workspace.scratch.normalized.len() < self.template.max_nodes
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "relative workspace state does not match plan",
            ));
        }
        for (i, &coordinate) in point.iter().enumerate() {
            let bound = bound_for(bounds, i);
            workspace.first[i].rebind(
                coordinate,
                &self.template.first[i].nodes,
                bound,
                &mut workspace.scratch,
            )?;
            if !self.template.second.is_empty() {
                workspace.second[i].rebind(
                    coordinate,
                    &self.template.first[i].nodes,
                    bound,
                    &mut workspace.scratch,
                )?;
            }
        }
        self.template.evaluate_geometry(
            point,
            &workspace.first,
            &workspace.second,
            function,
            output,
            &mut workspace.samples,
            context,
        )
    }
    /// Computes the derivative at `x` and returns it with the report.
    ///
    /// The evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`evaluate_into`](Self::evaluate_into), except those of
    ///   `out` and `workspace`.
    /// - `AllocationFailure` if the workspace or result cannot be allocated.
    pub fn evaluate<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError> {
        self.validate(x, bounds)?;
        let mut workspace = self.workspace(())?;
        let mut output = try_filled(
            self.output_len(),
            T::ZERO,
            scioxide_core::quantity!("derivative_output"),
        )?;
        let report = self.evaluate_validated(
            x,
            bounds,
            function,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_evaluation(report, output))
    }
    fn validate(&self, x: &[S], bounds: Option<&Bounds<S>>) -> Result<(), SciOxideError> {
        length(self.template.point.len(), x.len())?;
        validate_problem(x, bounds, scioxide_core::argument!("x"))
    }
}

// Reusable node-family storage and execution controls.

/// Options of [`PhysicalDifferencePlan`] and [`RelativeDifference`]:
/// execution policy, budgets and workspace limit.
///
/// The defaults are sequential execution, unbounded budgets and no workspace limit.
#[derive(Debug, Clone, Copy)]
pub struct NodeDifferenceOptions {
    pub(super) execution: ExecutionPolicy,
    pub(super) budgets: SolveBudgets,
    pub(super) max_workspace_bytes: usize,
}
impl Default for NodeDifferenceOptions {
    fn default() -> Self {
        Self {
            execution: ExecutionPolicy::Sequential,
            budgets: SolveBudgets::default(),
            max_workspace_bytes: usize::MAX,
        }
    }
}
impl NodeDifferenceOptions {
    /// Sets the execution policy for [`Parallel`](crate::Parallel) callbacks.
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }
    /// Sets the evaluation and iteration budgets of each evaluation. Limits of
    /// an enclosing solve still apply.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Limits the heap size of the workspace in bytes, including capacity left
    /// from earlier plans.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
}

/// Workspace size of a [`PhysicalDifferencePlan`] for callback values of type
/// `T`, returned by [`PhysicalDifferencePlan::workspace_requirements`].
#[derive(Debug, Clone)]
pub struct PhysicalDifferenceRequirements<T: NumericScalar> {
    pub(super) lengths: [usize; 5],
    pub(super) axis_nodes: usize,
    pub(super) bytes: usize,
    pub(super) limit: usize,
    pub(super) value: core::marker::PhantomData<T>,
}
impl<T: NumericScalar> PhysicalDifferenceRequirements<T> {
    /// Returns the heap size in bytes of a new workspace for the plan.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}
/// Scratch storage for a [`PhysicalDifferencePlan`] with callback values of
/// type `T`.
///
/// Create it with `plan.workspace(())`, or with `Default` followed by
/// [`try_reserve`](Self::try_reserve). Evaluation does not grow it.
#[derive(Debug, Clone)]
pub struct PhysicalDifferenceWorkspace<T: NumericScalar> {
    data: PhysicalBuffers<T>,
    pub(super) state: SolveState,
}
impl<T: NumericScalar> Default for PhysicalDifferenceWorkspace<T> {
    fn default() -> Self {
        Self {
            data: PhysicalBuffers::default(),
            state: SolveState::new(),
        }
    }
}
impl<T: NumericScalar> PhysicalDifferenceWorkspace<T> {
    /// Returns the heap size in bytes of the buffers the workspace holds.
    ///
    /// Includes retained capacity from earlier requests and nested buffers.
    /// Performs no allocation.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.data.bytes(None)
    }

    /// Reports whether the workspace is large enough for `required`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace holds more than the plan's
    ///   `max_workspace_bytes`.
    pub fn check(
        &self,
        required: &PhysicalDifferenceRequirements<T>,
    ) -> Result<Capacity, SciOxideError> {
        workspace::check(self.data.bytes(None)?, required.limit, || {
            Ok(self.data.sufficient(required))
        })
    }

    /// Grows the workspace to meet `required`.
    ///
    /// The byte limit is checked before any buffer grows, counting capacity
    /// left from earlier plans. If an allocation fails, buffers grown before it
    /// are kept; call [`check`](Self::check) again before evaluating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace would exceed `max_workspace_bytes`.
    /// - `AllocationFailure` if a buffer cannot grow.
    pub fn try_reserve(
        &mut self,
        required: &PhysicalDifferenceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        let projected = self.data.bytes(Some(required))?;
        workspace::reserve(
            &mut self.data,
            projected,
            required.limit,
            |data| data.reserve(required),
            |data| data.bytes(None),
        )
    }

    pub(super) fn prepare(
        &mut self,
        required: &PhysicalDifferenceRequirements<T>,
    ) -> Result<(), SciOxideError> {
        workspace::prepare(self.check(required)?, required.bytes, || {
            self.data.bytes(None)
        })?;
        self.data.configure(required)
    }
}
impl<T: NumericScalar> Default for PhysicalBuffers<T> {
    fn default() -> Self {
        Self {
            point: Vec::new(),
            samples: Vec::new(),
            intermediate: Vec::new(),
            staged: Vec::new(),
            retained: HessianSamples::default(),
            reductions: Vec::new(),
        }
    }
}
impl<T: NumericScalar> PhysicalBuffers<T> {
    fn capacities(&self) -> [usize; 5] {
        [
            self.point.capacity(),
            self.samples.capacity(),
            self.intermediate.capacity(),
            self.staged.capacity(),
            self.reductions.capacity(),
        ]
    }
    fn sufficient(&self, r: &PhysicalDifferenceRequirements<T>) -> bool {
        self.capacities()
            .iter()
            .zip(r.lengths)
            .all(|(&capacity, len)| capacity >= len)
            && self.retained.has_capacity(r.lengths[0], r.axis_nodes, 0)
    }
    pub(super) fn bytes(
        &self,
        r: Option<&PhysicalDifferenceRequirements<T>>,
    ) -> Result<usize, SciOxideError> {
        let lengths = r.map_or([0; 5], |r| r.lengths);
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<T::Real>(
                    [(self.point.capacity(), lengths[0])],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<T>(
                    [
                        (self.samples.capacity(), lengths[1]),
                        (self.intermediate.capacity(), lengths[2]),
                        (self.staged.capacity(), lengths[3]),
                    ],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<[RangeAccumulator<T::Real>; 2]>(
                    [(self.reductions.capacity(), lengths[4])],
                    workspace::BYTES,
                )?,
                self.retained
                    .projected_bytes(lengths[0], r.map_or(0, |r| r.axis_nodes), 0)?,
            ],
            workspace::BYTES,
        )
    }
    fn reserve(&mut self, r: &PhysicalDifferenceRequirements<T>) -> Result<(), SciOxideError> {
        workspace::reserve_buffer(
            &mut self.point,
            r.lengths[0],
            T::Real::ZERO,
            scioxide_core::quantity!("node_workspace"),
        )?;
        workspace::reserve_buffer(
            &mut self.samples,
            r.lengths[1],
            T::ZERO,
            scioxide_core::quantity!("node_workspace"),
        )?;
        workspace::reserve_buffer(
            &mut self.intermediate,
            r.lengths[2],
            T::ZERO,
            scioxide_core::quantity!("node_workspace"),
        )?;
        workspace::reserve_buffer(
            &mut self.staged,
            r.lengths[3],
            T::ZERO,
            scioxide_core::quantity!("node_workspace"),
        )?;
        workspace::reserve_buffer(
            &mut self.reductions,
            r.lengths[4],
            [RangeAccumulator::new(), RangeAccumulator::new()],
            scioxide_core::quantity!("node_workspace"),
        )?;
        self.retained.reserve(r.lengths[0], r.axis_nodes, 0)
    }
    fn configure(&mut self, r: &PhysicalDifferenceRequirements<T>) -> Result<(), SciOxideError> {
        self.point.resize(r.lengths[0], T::Real::ZERO);
        self.samples.resize(r.lengths[1], T::ZERO);
        self.intermediate.resize(r.lengths[2], T::ZERO);
        self.staged.resize(r.lengths[3], T::ZERO);
        self.reductions.resize(
            r.lengths[4],
            [RangeAccumulator::new(), RangeAccumulator::new()],
        );
        self.retained.configure(r.lengths[0], r.axis_nodes, 0)
    }
}

/// Workspace size of a [`RelativeDifference`] for callback values of type
/// `T`, returned by [`RelativeDifference::workspace_requirements`].
#[derive(Debug)]
pub struct RelativeDifferenceRequirements<'plan, T: NumericScalar> {
    pub(super) plan: &'plan RelativeDifference<T::Real>,
    pub(super) physical: PhysicalDifferenceRequirements<T>,
    pub(super) bytes: usize,
}
impl<T: NumericScalar> RelativeDifferenceRequirements<'_, T> {
    /// Returns the heap size in bytes of a new workspace for the method value.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}
/// Scratch storage for a [`RelativeDifference`] with callback values of
/// type `T`.
///
/// Create it with `method.workspace(())`, or with `Default` followed by
/// [`try_reserve`](Self::try_reserve). Evaluation does not grow it.
#[derive(Debug, Clone)]
pub struct RelativeDifferenceWorkspace<T: NumericScalar> {
    data: RelativeBuffers<T>,
    pub(super) state: SolveState,
}
impl<T: NumericScalar> Default for RelativeDifferenceWorkspace<T> {
    fn default() -> Self {
        Self {
            data: RelativeBuffers::default(),
            state: SolveState::new(),
        }
    }
}
impl<T: NumericScalar> RelativeDifferenceWorkspace<T> {
    /// Returns the heap size in bytes of the buffers the workspace holds.
    ///
    /// Includes retained capacity from earlier requests and nested buffers.
    /// Performs no allocation.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.data.bytes(None)
    }

    /// Reports whether the workspace is large enough for `r`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace holds more than the method's
    ///   `max_workspace_bytes`.
    pub fn check(
        &self,
        r: &RelativeDifferenceRequirements<'_, T>,
    ) -> Result<Capacity, SciOxideError> {
        workspace::check(
            self.data.bytes(None)?,
            r.plan.template.options.max_workspace_bytes,
            || self.data.sufficient(r),
        )
    }

    /// Grows the workspace to meet `r`.
    ///
    /// The byte limit is checked before any buffer grows, counting capacity
    /// left from earlier plans. If an allocation fails, buffers grown before it
    /// are kept; call [`check`](Self::check) again before evaluating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace would exceed `max_workspace_bytes`.
    /// - `AllocationFailure` if a buffer cannot grow.
    pub fn try_reserve(
        &mut self,
        r: &RelativeDifferenceRequirements<'_, T>,
    ) -> Result<(), SciOxideError> {
        let limit = r.plan.template.options.max_workspace_bytes;
        let projected = self.data.bytes(Some(r))?;
        workspace::reserve(
            &mut self.data,
            projected,
            limit,
            |data| {
                reserve_geometry(&mut data.first, &r.plan.template.first)?;
                reserve_geometry(&mut data.second, &r.plan.template.second)?;
                let n = r.plan.template.max_nodes;
                workspace::reserve_buffer(
                    &mut data.scratch.normalized,
                    n,
                    T::Real::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.scratch.ordered,
                    n,
                    T::Real::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.scratch.permutation,
                    n,
                    0,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                workspace::reserve_buffer(
                    &mut data.scratch.table,
                    checked_mul(n, 3, scioxide_core::quantity!("fornberg_table"))?,
                    T::Real::ZERO,
                    scioxide_core::quantity!("node_workspace"),
                )?;
                data.samples.reserve(&r.physical)?;
                Ok(())
            },
            |data| data.bytes(None),
        )
    }

    pub(super) fn prepare(
        &mut self,
        r: &RelativeDifferenceRequirements<'_, T>,
    ) -> Result<(), SciOxideError> {
        workspace::prepare(self.check(r)?, r.bytes, || self.data.bytes(None))?;
        configure_geometry(&mut self.data.first, &r.plan.template.first);
        configure_geometry(&mut self.data.second, &r.plan.template.second);
        let n = r.plan.template.max_nodes;
        self.data.scratch.normalized.resize(n, T::Real::ZERO);
        self.data.scratch.ordered.resize(n, T::Real::ZERO);
        self.data.scratch.permutation.resize(n, 0);
        self.data.scratch.table.resize(
            checked_mul(n, 3, scioxide_core::quantity!("fornberg_table"))?,
            T::Real::ZERO,
        );
        self.data.samples.configure(&r.physical)
    }
}
impl<T: NumericScalar> Default for RelativeBuffers<T> {
    fn default() -> Self {
        Self {
            first: Vec::new(),
            second: Vec::new(),
            scratch: StencilScratch {
                normalized: Vec::new(),
                ordered: Vec::new(),
                permutation: Vec::new(),
                table: Vec::new(),
            },
            samples: PhysicalBuffers::default(),
        }
    }
}
impl<T: NumericScalar> RelativeBuffers<T> {
    fn sufficient(&self, r: &RelativeDifferenceRequirements<'_, T>) -> Result<bool, SciOxideError> {
        let n = r.plan.template.max_nodes;
        Ok(geometry_sufficient(&self.first, &r.plan.template.first)
            && geometry_sufficient(&self.second, &r.plan.template.second)
            && self.scratch.normalized.capacity() >= n
            && self.scratch.ordered.capacity() >= n
            && self.scratch.permutation.capacity() >= n
            && self.scratch.table.capacity()
                >= checked_mul(n, 3, scioxide_core::quantity!("fornberg_table"))?
            && self.samples.sufficient(&r.physical))
    }
    pub(super) fn bytes(
        &self,
        r: Option<&RelativeDifferenceRequirements<'_, T>>,
    ) -> Result<usize, SciOxideError> {
        let nodes = r.map_or(0, |r| r.plan.template.max_nodes);
        let table = checked_mul(nodes, 3, scioxide_core::quantity!("fornberg_table"))?;
        workspace::sum_bytes(
            [
                geometry_bytes(&self.first, r.map(|r| r.plan.template.first.as_slice()))?,
                geometry_bytes(&self.second, r.map(|r| r.plan.template.second.as_slice()))?,
                workspace::buffer_bytes::<T::Real>(
                    [
                        (self.scratch.normalized.capacity(), nodes),
                        (self.scratch.ordered.capacity(), nodes),
                        (self.scratch.table.capacity(), table),
                    ],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<usize>(
                    [(self.scratch.permutation.capacity(), nodes)],
                    workspace::BYTES,
                )?,
                self.samples.bytes(r.map(|r| &r.physical))?,
            ],
            workspace::BYTES,
        )
    }
}
pub(super) fn geometry_sufficient<S: RealScalar>(
    storage: &[FiniteDifferenceStencil<S>],
    template: &[FiniteDifferenceStencil<S>],
) -> bool {
    storage.len() >= template.len()
        && storage.iter().zip(template).all(|(work, stencil)| {
            work.nodes.capacity() >= stencil.nodes.len()
                && work.weights.capacity() >= stencil.weights.len()
        })
}
pub(super) fn geometry_bytes<S: RealScalar>(
    storage: &Vec<FiniteDifferenceStencil<S>>,
    template: Option<&[FiniteDifferenceStencil<S>]>,
) -> Result<usize, SciOxideError> {
    let template = template.unwrap_or(&[]);
    let backing = workspace::buffer_bytes::<FiniteDifferenceStencil<S>>(
        [(storage.capacity(), template.len())],
        workspace::BYTES,
    );
    let children = (0..storage.len().max(template.len())).map(|i| {
        workspace::buffer_bytes::<S>(
            [
                (
                    storage.get(i).map_or(0, |s| s.nodes.capacity()),
                    template.get(i).map_or(0, |s| s.nodes.len()),
                ),
                (
                    storage.get(i).map_or(0, |s| s.weights.capacity()),
                    template.get(i).map_or(0, |s| s.weights.len()),
                ),
            ],
            workspace::BYTES,
        )
    });
    workspace::try_sum_bytes(core::iter::once(backing).chain(children), workspace::BYTES)
}
pub(super) fn reserve_geometry<S: RealScalar>(
    storage: &mut Vec<FiniteDifferenceStencil<S>>,
    template: &[FiniteDifferenceStencil<S>],
) -> Result<(), SciOxideError> {
    scioxide_core::backend::try_reserve(
        storage,
        template.len().saturating_sub(storage.len()),
        scioxide_core::quantity!("reserved_stencils"),
    )?;
    while storage.len() < template.len() {
        storage.push(FiniteDifferenceStencil {
            coordinate: S::ZERO,
            nodes: Vec::new(),
            weights: Vec::new(),
            scale: S::ONE,
            order: 0,
            amplification: S::ZERO,
            base_indices: (0, 0),
        });
    }
    for (work, stencil) in storage.iter_mut().zip(template) {
        workspace::reserve_buffer(
            &mut work.nodes,
            stencil.nodes.len(),
            S::ZERO,
            scioxide_core::quantity!("node_workspace"),
        )?;
        workspace::reserve_buffer(
            &mut work.weights,
            stencil.weights.len(),
            S::ZERO,
            scioxide_core::quantity!("node_workspace"),
        )?;
    }
    configure_geometry(storage, template);
    Ok(())
}
pub(super) fn configure_geometry<S: RealScalar>(
    storage: &mut [FiniteDifferenceStencil<S>],
    template: &[FiniteDifferenceStencil<S>],
) {
    for (work, stencil) in storage.iter_mut().zip(template) {
        work.coordinate = stencil.coordinate;
        work.scale = stencil.scale;
        work.order = stencil.order;
        work.amplification = stencil.amplification;
        work.base_indices = stencil.base_indices;
        work.nodes.resize(stencil.nodes.len(), S::ZERO);
        work.nodes.copy_from_slice(&stencil.nodes);
        work.weights.resize(stencil.weights.len(), S::ZERO);
        work.weights.copy_from_slice(&stencil.weights);
    }
}

pub(super) fn node_identity(
    algorithm: scioxide_core::AlgorithmId,
    len: usize,
) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        algorithm,
        len as u128,
        0,
    )
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use scioxide_core::{Complex64, SciOxideErrorKind};

    #[test]
    fn physical_buffers_require_each_slot_even_with_other_spare_storage() {
        let nodes = [-0.5, 0.0, 0.5];
        let plan = PhysicalDifferencePlan::new(
            DerivativeOperation::Hessian,
            &[0.0; 2],
            &[nodes.as_slice(); 2],
            1,
            Default::default(),
        )
        .unwrap();
        let r = plan.workspace_requirements::<Complex64>(()).unwrap();
        for i in 0..5 {
            let n = r.lengths[i];
            let mut work = plan.workspace::<Complex64>(()).unwrap();
            work.data.point.reserve(1024);
            work.data.samples.reserve(1024);
            match i {
                0 => work.data.point = Vec::with_capacity(n - 1),
                1 => work.data.samples = Vec::with_capacity(n - 1),
                2 => work.data.intermediate = Vec::with_capacity(n - 1),
                3 => work.data.staged = Vec::with_capacity(n - 1),
                _ => work.data.reductions = Vec::with_capacity(n - 1),
            }
            assert!(work.data.bytes(None).unwrap() > r.bytes());
            assert_eq!(work.check(&r).unwrap(), Capacity::NeedsGrowth);
            assert_eq!(
                work.prepare(&r).unwrap_err().kind(),
                SciOxideErrorKind::DimensionMismatch
            );
        }
    }
    #[test]
    fn relative_geometry_children_and_binding_buffers_require_each_slot() {
        let nodes = [-0.5, 0.0, 0.5];
        let plan = RelativeDifference::new(
            DerivativeOperation::Hessian,
            &[nodes.as_slice(); 2],
            1,
            Default::default(),
        )
        .unwrap();
        let r = plan.workspace_requirements::<Complex64>(()).unwrap();
        for i in 0..8 {
            let mut work = plan.workspace::<Complex64>(()).unwrap();
            work.data.samples.point.reserve(1024);
            let n = plan.template.max_nodes;
            match i {
                0 => work.data.first[0].nodes = Vec::with_capacity(n - 1),
                1 => work.data.first[0].weights = Vec::with_capacity(n - 1),
                2 => work.data.second[0].nodes = Vec::with_capacity(n - 1),
                3 => work.data.second[0].weights = Vec::with_capacity(n - 1),
                4 => work.data.scratch.normalized = Vec::with_capacity(n - 1),
                5 => work.data.scratch.ordered = Vec::with_capacity(n - 1),
                6 => work.data.scratch.permutation = Vec::with_capacity(n - 1),
                _ => work.data.scratch.table = Vec::with_capacity(n * 3 - 1),
            }
            assert!(work.data.bytes(None).unwrap() > r.bytes());
            assert_eq!(work.check(&r).unwrap(), Capacity::NeedsGrowth);
            assert_eq!(
                work.prepare(&r).unwrap_err().kind(),
                SciOxideErrorKind::DimensionMismatch
            );
        }
    }
}
