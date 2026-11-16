//! Polynomial interpolation with the barycentric formula of the second kind.
//!
//! [`BarycentricCore`] also evaluates [`crate::FloaterHormannInterpolator`],
//! which uses the same quotient with different weights.

// Mathematical code uses standard variable names (x, y, w, t, n, etc.)
#![allow(clippy::many_single_char_names)]
#![allow(clippy::suspicious_operation_groupings)]

use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError, check_value_shape,
    checked_affine, checked_affine_ratio, checked_midpoint, write_value,
};
use scioxide_poly::backend::barycentric::{ChebyshevPoints, chebyshev_points};
use scioxide_poly::backend::polynomial_basis::barycentric_weights;

use crate::error::{instability, invalid};

use crate::polynomial::antiderivative::{PreparedPolynomialAntiderivative, prepare_from_samples};
use crate::protocol::{
    Differentiable, ExtrapolationPolicy, Interpolant, NodeSeparation, SampleOptions, clamped_axes,
    domain_from_nodes, resolve_coordinate,
};
use crate::validate::validate_nodes;
use crate::values::{QueryWorkspace, flatten, zeroed};

/// Interpolates `n` samples with the polynomial of degree at most `n − 1`, in barycentric form.
///
/// For nodes `x₀ < x₁ < … < xₙ₋₁` with values `yⱼ`, the interpolant is
///
/// ```text
/// p(x) = Σⱼ (wⱼ / (x − xⱼ)) yⱼ  /  Σⱼ (wⱼ / (x − xⱼ)),   wⱼ = 1 / Πₖ≠ⱼ (xⱼ − xₖ),
/// ```
///
/// the barycentric formula of the second kind. The weights `wⱼ` are stored
/// divided by a common factor, which cancels in the quotient. With
/// [`ExtrapolationPolicy::MethodDefined`] the polynomial is evaluated outside
/// `[x₀, xₙ₋₁]`.
///
/// Values are any equally shaped [`AffineValue`] (real or complex scalars,
/// arrays, vectors). Each real component is interpolated separately, and gives
/// the same result as a scalar interpolant of that component.
///
/// [`Self::new`] computes the weights in O(n²) operations. [`Self::from_equispaced`],
/// [`Self::from_chebyshev_roots`] and [`Self::from_chebyshev_lobatto`] use
/// closed-form weights and cost O(n). A value or first-derivative query costs
/// O(n d) for `d` real components. Sums are compensated, and terms are
/// rescaled when they would overflow or underflow. For high degrees, prefer
/// Chebyshev nodes: on equispaced nodes the Lebesgue constant grows
/// exponentially with `n`, and the interpolants of smooth functions can
/// diverge (Runge phenomenon).
///
/// Definite integrals come from [`Self::prepare_antiderivative`].
///
/// # Errors
///
/// Queries fail as described in the [crate docs](crate#errors), and with
/// `NumericalInstability` if no finite value can be formed.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{BarycentricInterpolator, Differentiable, ExtrapolationPolicy, Interpolant};
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![0.0_f64, 1.0, 4.0]; // y = x²
/// let interp = BarycentricInterpolator::new(x.clone(), y)?;
/// let value = interp.evaluate([1.5], ExtrapolationPolicy::Reject)?;
/// assert!((value - 2.25_f64).abs() < 1e-14);
/// let [slope] = interp.gradient([1.5], ExtrapolationPolicy::Reject)?;
/// assert!((slope - 3.0_f64).abs() < 1e-13);
///
/// let complex = vec![
///     Complex::new(0.0, 1.0),
///     Complex::new(1.0, 1.0),
///     Complex::new(4.0, 1.0),
/// ];
/// let interp = BarycentricInterpolator::new(x, complex)?;
/// let value = interp.evaluate([0.5], ExtrapolationPolicy::Reject)?;
/// assert!((value - Complex::new(0.25, 1.0)).norm() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
/// *SIAM Review* 46(3), 2004, 501–517.
#[derive(Debug, Clone)]
#[must_use]
pub struct BarycentricInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    core: BarycentricCore<S>,
    template: V,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> BarycentricInterpolator<S, V> {
    /// Builds the interpolant through the samples `(x[j], y[j])`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// `x` must be strictly increasing. Computing the weights costs O(n²)
    /// operations.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() != y.len()` or the values differ in shape.
    /// - `InvalidInput` if there are no nodes or the nodes are not strictly increasing.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, SampleOptions::new())
    }

    /// Builds the interpolant through the samples `(x[j], y[j])`, validating the
    /// nodes under `options`.
    ///
    /// The nodes `x` must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`]. A query within that separation of a
    /// node returns the node's value. Computing the weights costs O(n²)
    /// operations.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() != y.len()` or the values differ in shape.
    /// - `InvalidInput` if there are no nodes or the nodes are not strictly
    ///   increasing and distinct under the separation of `options`.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let separation = options.node_separation();
        if x.len() != y.len() {
            return Err(crate::error::shape(x.len(), y.len()));
        }
        validate_nodes(&x, 1, separation, scioxide_core::argument!("x"))?;
        let flat = flatten(y, x.len())?;
        let w = barycentric_weights(&x)?;
        Ok(Self {
            core: BarycentricCore::new(x, w, flat.components, flat.dimension, separation)?,
            template: flat.template,
        })
    }

    /// Builds the interpolant on `n = values.len()` equispaced nodes over
    /// `[lower, upper]`.
    ///
    /// For `n = 1` the only node is the midpoint of the interval. For `n ≥ 2` the
    /// nodes are `a + j (b − a) / (n − 1)` with `a = lower` and `b = upper`,
    /// including both endpoints. The closed-form weights `wⱼ = (−1)ʲ C(n − 1, j)`
    /// cost O(n) operations. The nodes must be strictly increasing, as under
    /// [`Self::new`].
    ///
    /// High-degree interpolation on equispaced nodes is ill-conditioned and
    /// subject to the Runge phenomenon. This constructor is faster than
    /// [`Self::new`] but does not improve that conditioning.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lower` or `upper` is NaN or infinite, or a value is not
    ///   finite.
    /// - `InvalidInput` (`Empty`) if `values` is empty, and `InvalidInput` if
    ///   `lower > upper` or the generated nodes are not strictly increasing, as
    ///   for `lower == upper` and `n ≥ 2`.
    /// - `DimensionMismatch` if the values differ in shape.
    /// - `NumericalInstability` if a generated node or weight is not
    ///   representable.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BarycentricInterpolator, Univariate};
    /// // Nodes −1, 0, 1 with the values of f(x) = x².
    /// let parabola = BarycentricInterpolator::from_equispaced(vec![1.0_f64, 0.0, 1.0], -1.0, 1.0)?;
    /// assert_eq!(parabola.nodes(), [-1.0, 0.0, 1.0]);
    /// assert!((parabola.value_at(0.5)? - 0.25).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_equispaced(values: Vec<V>, lower: S, upper: S) -> Result<Self, SciOxideError> {
        Self::from_structured(
            values,
            Interval::new(lower, upper)?,
            StructuredNodeFamily::Equispaced,
        )
    }

    /// Builds the interpolant on the `n = values.len()` Chebyshev roots mapped to
    /// `[lower, upper]`.
    ///
    /// The nodes are the zeros of the Chebyshev polynomial `Tₙ`, mapped affinely
    /// from `[−1, 1]` to `[lower, upper]` and sorted in increasing order. They
    /// lie strictly inside the interval. For `n = 1` the only node is the
    /// midpoint. The closed-form weights cost O(n) operations.
    ///
    /// # Errors
    ///
    /// As [`Self::from_equispaced`].
    pub fn from_chebyshev_roots(values: Vec<V>, lower: S, upper: S) -> Result<Self, SciOxideError> {
        Self::from_structured(
            values,
            Interval::new(lower, upper)?,
            StructuredNodeFamily::ChebyshevRoots,
        )
    }

    /// Builds the interpolant on the `n = values.len()` Chebyshev–Lobatto points
    /// mapped to `[lower, upper]`.
    ///
    /// The nodes are the extrema of `Tₙ₋₁` on `[−1, 1]`, mapped affinely to
    /// `[lower, upper]`, so both endpoints are nodes. For `n = 1` the only node
    /// is the midpoint. The closed-form weights cost O(n) operations.
    ///
    /// # Errors
    ///
    /// As [`Self::from_equispaced`].
    pub fn from_chebyshev_lobatto(
        values: Vec<V>,
        lower: S,
        upper: S,
    ) -> Result<Self, SciOxideError> {
        Self::from_structured(
            values,
            Interval::new(lower, upper)?,
            StructuredNodeFamily::ChebyshevLobatto,
        )
    }

    /// Builds the interpolant on one of the structured node families.
    fn from_structured(
        values: Vec<V>,
        interval: Interval<S>,
        family: StructuredNodeFamily,
    ) -> Result<Self, SciOxideError> {
        if values.is_empty() {
            return Err(crate::error::empty(
                scioxide_core::argument!("values"),
                "structured barycentric construction requires at least one value",
            ));
        }
        let count = values.len();
        let flat = flatten(values, count)?;
        let (x, w) = structured_nodes_and_weights(count, interval, family)?;
        let separation = NodeSeparation::Exact;
        validate_nodes(&x, 1, separation, scioxide_core::argument!("upper"))?;
        Ok(Self {
            core: BarycentricCore::new(x, w, flat.components, flat.dimension, separation)?,
            template: flat.template,
        })
    }

    /// Evaluates the interpolant and its first derivative in one pass over the nodes.
    ///
    /// `value` and `derivative` each hold [`Interpolant::value_dimension`]
    /// components. The cost is the same O(n d) as one value query. Under
    /// [`ExtrapolationPolicy::Clamp`] the derivative is zero strictly outside
    /// the domain.
    ///
    /// # Errors
    ///
    /// - The query errors described in the [crate docs](crate#errors).
    /// - `NumericalInstability` if no finite value can be formed.
    ///
    /// ```
    /// use scioxide_interp::{BarycentricInterpolator, ExtrapolationPolicy, Interpolant};
    /// let interp = BarycentricInterpolator::new(
    ///     vec![0.0_f64, 1.0, 2.0],
    ///     vec![[0.0, 1.0], [1.0, 1.0], [4.0, 1.0]],
    /// )?;
    /// let mut workspace = interp.workspace()?;
    /// let (mut value, mut slope) = ([0.0; 2], [0.0; 2]);
    /// interp.evaluate_with_derivative_into(
    ///     [1.5],
    ///     ExtrapolationPolicy::Reject,
    ///     &mut value,
    ///     &mut slope,
    ///     &mut workspace,
    /// )?;
    /// assert!((value[0] - 2.25).abs() < 1e-14 && (slope[0] - 3.0).abs() < 1e-13);
    /// assert_eq!((value[1], slope[1]), (1.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_with_derivative_into<O, P>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        value_out: &mut O,
        derivative_out: &mut P,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError>
    where
        O: AffineValueMut<Scale = S> + ?Sized,
        P: AffineValueMut<Scale = S> + ?Sized,
    {
        let dimension = self.core.dimension;
        check_value_shape(value_out, dimension)?;
        check_value_shape(derivative_out, dimension)?;
        let staged = workspace.scratch(2 * dimension)?;
        let (values, derivatives) = staged.split_at_mut(dimension);
        self.core
            .stage(point, policy, values, Some(derivatives), false)?;
        write_value(values, value_out)?;
        write_value(derivatives, derivative_out)
    }

    /// Evaluates the interpolant and its first derivative in one pass over the nodes,
    /// returning `(value, derivative)`.
    ///
    /// This is [`Self::evaluate_with_derivative_into`] with newly allocated
    /// outputs and scratch, and returns its results bit for bit.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_with_derivative_into`].
    /// - `AllocationFailure` if the outputs or the scratch cannot be allocated.
    ///
    /// ```
    /// use scioxide_interp::{BarycentricInterpolator, ExtrapolationPolicy};
    /// // y = x² through three nodes.
    /// let interp = BarycentricInterpolator::new(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 4.0])?;
    /// let (value, slope) = interp.evaluate_with_derivative([1.5], ExtrapolationPolicy::Reject)?;
    /// assert!((value - 2.25).abs() < 1e-14 && (slope - 3.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_with_derivative(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<(V::Owned, V::Owned), SciOxideError> {
        let mut value = zeroed(&self.template)?;
        let mut derivative = zeroed(&self.template)?;
        let mut workspace = QueryWorkspace::new(2 * self.core.dimension)?;
        self.evaluate_with_derivative_into(
            point,
            policy,
            &mut value,
            &mut derivative,
            &mut workspace,
        )?;
        Ok((value, derivative))
    }

    /// Returns the barycentric weights `wⱼ`, divided by a common factor.
    ///
    /// The factor cancels in the barycentric quotient and keeps the weights
    /// from overflowing.
    pub fn weights(&self) -> &[S] {
        &self.core.weights
    }

    /// Returns the nodes in increasing order.
    pub fn nodes(&self) -> &[S] {
        &self.core.nodes
    }

    /// Returns the values as real components, node by node (`[node][component]`).
    pub fn values(&self) -> &[S] {
        &self.core.values
    }

    /// Computes the antiderivative of the interpolant, for definite integrals.
    ///
    /// The global polynomial interpolants do not implement
    /// [`crate::Integrable`]; integrate with the returned object instead.
    /// Preparation converts each component to power coefficients of the
    /// antiderivative in a normalized coordinate, in O(n² d) operations and
    /// O(n d) memory. Each integral then costs O(n d).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the coefficients cannot be stored.
    /// - `NumericalInstability` if a coefficient is not representable.
    pub fn prepare_antiderivative(
        &self,
    ) -> Result<PreparedPolynomialAntiderivative<S, V::Owned>, SciOxideError> {
        prepare_from_samples(
            &self.core.nodes,
            &self.core.values,
            self.core.dimension,
            &self.template,
            self.core.domain,
        )
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 1>
    for BarycentricInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 1] {
        [self.core.domain]
    }
    fn value_dimension(&self) -> usize {
        self.core.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(2 * self.core.dimension)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(2 * self.core.dimension)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.core.evaluate_into(point, policy, out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Differentiable<S, 1>
    for BarycentricInterpolator<S, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.core.dimension)?;
        let staged = workspace.scratch(self.core.dimension)?;
        self.core.stage(point, policy, staged, None, true)?;
        write_value(staged, out)
    }
}

// ---------------------------------------------------------------------------
// Shared second-kind barycentric storage and kernel
// ---------------------------------------------------------------------------

/// Nodes, weights and values of a barycentric quotient of the second kind.
///
/// Each real component is evaluated separately, as a scalar interpolant would.
#[derive(Debug, Clone)]
pub(crate) struct BarycentricCore<S: RealScalar> {
    nodes: Vec<S>,
    weights: Vec<S>,
    dimension: usize,
    /// `[node][component]` values.
    values: Vec<S>,
    /// Maximum magnitude of each component.
    scales: Vec<S>,
    domain: Interval<S>,
    separation: NodeSeparation<S>,
}

impl<S: RealScalar> BarycentricCore<S> {
    /// Assemble validated nodes, nonzero weights and node-major values.
    pub(crate) fn new(
        nodes: Vec<S>,
        weights: Vec<S>,
        values: Vec<S>,
        dimension: usize,
        separation: NodeSeparation<S>,
    ) -> Result<Self, SciOxideError> {
        let domain = domain_from_nodes(&nodes)?;
        let mut scales = try_filled(
            dimension,
            S::ZERO,
            scioxide_core::quantity!("barycentric_value_scales"),
        )?;
        for (index, &value) in values.iter().enumerate() {
            let scale = &mut scales[index % dimension];
            *scale = scale.max(value.abs());
        }
        Ok(Self {
            nodes,
            weights,
            dimension,
            values,
            scales,
            domain,
            separation,
        })
    }

    pub(crate) fn nodes(&self) -> &[S] {
        &self.nodes
    }
    pub(crate) fn weights(&self) -> &[S] {
        &self.weights
    }
    pub(crate) fn values(&self) -> &[S] {
        &self.values
    }
    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }
    pub(crate) const fn domain(&self) -> Interval<S> {
        self.domain
    }

    /// Evaluates one value into `out`.
    pub(crate) fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        self.stage(point, policy, staged, None, false)?;
        write_value(staged, out)
    }

    /// Computes every component's value (or derivative, when `derivative_only`),
    /// plus derivatives into `derivatives` when given.
    fn stage(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        primary: &mut [S],
        mut derivatives: Option<&mut [S]>,
        derivative_only: bool,
    ) -> Result<(), SciOxideError> {
        let [clamped] = clamped_axes(policy, [self.domain], point);
        let t = resolve_coordinate(&self.domain, point[0], policy)?;
        if derivative_only && clamped {
            primary.fill(S::ZERO);
            return Ok(());
        }
        let wants_derivative = derivative_only || derivatives.is_some();
        let mode = if wants_derivative && !clamped {
            EvaluationMode::ValueAndDerivative
        } else {
            EvaluationMode::ValueOnly
        };
        let reference = self.locate(t)?;
        for (component, slot) in primary.iter_mut().enumerate() {
            let evaluation = self.evaluate_component(reference, t, component, mode)?;
            let derivative = if clamped {
                S::ZERO
            } else {
                evaluation.derivative
            };
            *slot = if derivative_only {
                derivative
            } else {
                evaluation.value
            };
            if let Some(derivatives) = derivatives.as_deref_mut() {
                derivatives[component] = derivative;
            }
        }
        Ok(())
    }

    /// Nearest or coincident node of a resolved query, shared by all components.
    fn locate(&self, t: S) -> Result<QueryReference, SciOxideError> {
        if self.nodes.len() == 1 {
            return Ok(QueryReference::Singleton);
        }
        query_reference(&self.nodes, t, self.separation)
    }

    fn evaluate_component(
        &self,
        reference: QueryReference,
        t: S,
        component: usize,
        mode: EvaluationMode,
    ) -> Result<BarycentricEvaluation<S>, SciOxideError> {
        let (index, query) = match reference {
            QueryReference::Singleton => {
                return Ok(BarycentricEvaluation {
                    value: self.values[component],
                    derivative: S::ZERO,
                });
            }
            QueryReference::Coincident(index) => (index, self.nodes[index]),
            QueryReference::Distinct(index) => (index, t),
        };
        if self.scales[component] == S::ZERO {
            return Ok(BarycentricEvaluation {
                value: S::ZERO,
                derivative: S::ZERO,
            });
        }
        let values = |index| self.values[index * self.dimension + component];
        let cutoff = S::ONE.scalbn(5) * <S as RealScalar>::EPSILON;
        let result = match mode {
            EvaluationMode::ValueOnly => {
                scioxide_poly::backend::barycentric::evaluate::<S, _, false, true, false>(
                    &self.nodes,
                    &self.weights,
                    values,
                    query,
                    index,
                    cutoff,
                )?
            }
            EvaluationMode::ValueAndDerivative => {
                scioxide_poly::backend::barycentric::evaluate::<S, _, true, true, false>(
                    &self.nodes,
                    &self.weights,
                    values,
                    query,
                    index,
                    cutoff,
                )?
            }
        };
        Ok(BarycentricEvaluation {
            value: result.value,
            derivative: result.derivative,
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuredNodeFamily {
    Equispaced,
    ChebyshevRoots,
    ChebyshevLobatto,
}

/// Structured nodes and normalized weights.
fn structured_nodes_and_weights<S: RealScalar>(
    n: usize,
    interval: Interval<S>,
    family: StructuredNodeFamily,
) -> Result<(Vec<S>, Vec<S>), SciOxideError> {
    // Weight conventions (Berrut & Trefethen, 2004), each up to a common
    // factor that cancels in the barycentric quotient: equispaced
    // w_j = (-1)^j C(n-1, j); Chebyshev points take poly's root and extrema
    // weights. Endpoint families use the interval bounds verbatim.
    let (lower, upper) = (interval.lower(), interval.upper());
    if n == 1 {
        return Ok((
            try_filled(
                1,
                checked_midpoint(lower, upper)?,
                scioxide_core::quantity!("structured_barycentric_singleton_node"),
            )?,
            try_filled(
                1,
                S::ONE,
                scioxide_core::quantity!("structured_barycentric_singleton_weight"),
            )?,
        ));
    }
    if lower == upper {
        return Err(invalid(
            scioxide_core::argument!("upper"),
            "multiple structured nodes require a non-degenerate interval",
        ));
    }
    let points = match family {
        StructuredNodeFamily::Equispaced => {
            let mut nodes =
                try_with_capacity(n, scioxide_core::quantity!("structured_barycentric_nodes"))?;
            for index in 0..n {
                nodes.push(checked_affine_ratio(lower, upper, index, n - 1)?);
            }
            let mut weights = try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("structured_barycentric_weights"),
            )?;
            fill_equispaced_weights(&mut weights)?;
            return Ok((nodes, weights));
        }
        StructuredNodeFamily::ChebyshevRoots => ChebyshevPoints::Roots,
        StructuredNodeFamily::ChebyshevLobatto => ChebyshevPoints::Extrema,
    };
    let (mut nodes, weights) = chebyshev_points::<S>(n, points)?;
    for node in &mut nodes {
        // Map [-1, 1] from the nearer endpoint so both ends keep relative accuracy.
        *node = match *node {
            x if x == -S::ONE => lower,
            x if x == S::ONE => upper,
            x if x < S::ZERO => checked_affine(lower, upper, (S::ONE + x) * S::HALF)?,
            x => checked_affine(upper, lower, (S::ONE - x) * S::HALF)?,
        };
    }
    Ok((nodes, weights))
}

/// Normalized equispaced weights; underflow is reported.
fn fill_equispaced_weights<S: RealScalar>(weights: &mut [S]) -> Result<(), SciOxideError> {
    // Binomial weights C(m, j), m = n - 1, divided by the central (largest)
    // coefficient: start at ±1 in the middle and walk outwards with the ratio
    // recurrences C(m, j+1) = C(m, j) (m-j)/(j+1) and
    // C(m, j-1) = C(m, j) j/(m-j+1). Magnitudes only decrease outwards, so
    // nothing can overflow (C(m, m/2) itself would for large m); the smallest
    // end weights may underflow, which is reported rather than silently
    // producing a zero weight that would drop a node from the interpolant.
    let last = weights.len() - 1;
    let center = last / 2;
    weights[center] = alternating_sign(center);

    for index in center..last {
        let numerator = S::try_from_usize_exact(last - index)?;
        let denominator = S::try_from_usize_exact(index + 1)?;
        let magnitude = weights[index].abs() * numerator / denominator;
        weights[index + 1] = alternating_magnitude(magnitude, weights[index]);
        validate_structured_weight(weights[index + 1])?;
    }
    for index in (1..=center).rev() {
        let numerator = S::try_from_usize_exact(index)?;
        let denominator = S::try_from_usize_exact(last - index + 1)?;
        let magnitude = weights[index].abs() * numerator / denominator;
        weights[index - 1] = alternating_magnitude(magnitude, weights[index]);
        validate_structured_weight(weights[index - 1])?;
    }
    Ok(())
}

fn alternating_magnitude<S: RealScalar>(magnitude: S, previous: S) -> S {
    if previous < S::ZERO {
        magnitude
    } else {
        -magnitude
    }
}

/// Reject a zero or non-finite weight.
fn validate_structured_weight<S: RealScalar>(weight: S) -> Result<(), SciOxideError> {
    if weight == S::ZERO {
        Err(instability(
            scioxide_core::NumericalIssue::Underflow,
            "normalized structured weight underflowed to zero",
        ))
    } else if !weight.is_finite() {
        Err(instability(
            scioxide_core::NumericalIssue::Overflow,
            "normalized structured weight is not finite",
        ))
    } else {
        Ok(())
    }
}

fn alternating_sign<S: RealScalar>(index: usize) -> S {
    if index.is_multiple_of(2) {
        S::ONE
    } else {
        -S::ONE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvaluationMode {
    ValueOnly,
    ValueAndDerivative,
}

#[derive(Debug, Clone, Copy)]
struct BarycentricEvaluation<S> {
    value: S,
    derivative: S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryReference {
    /// A single node: the interpolant is constant.
    Singleton,
    Coincident(usize),
    Distinct(usize),
}

fn query_reference<S: RealScalar>(
    nodes: &[S],
    query: S,
    separation: NodeSeparation<S>,
) -> Result<QueryReference, SciOxideError> {
    // The returned reference is the node nearest to the query: its term
    // dominates near a node, so normalizing by it keeps every other term O(1).
    // With exact separation a lower-bound binary search finds both a
    // coincident node and the nearest neighbour in O(log n). A tolerant policy
    // uses a scale-dependent threshold per node, so it scans linearly.
    if separation == NodeSeparation::Exact {
        let mut lower = 0usize;
        let mut upper = nodes.len();
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if nodes[middle] < query {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        if lower < nodes.len() && nodes[lower] == query {
            return Ok(QueryReference::Coincident(lower));
        }
        return Ok(QueryReference::Distinct(match lower {
            0 => 0,
            index if index == nodes.len() => nodes.len() - 1,
            index => {
                let left_distance = (query - nodes[index - 1]).abs();
                let right_distance = (nodes[index] - query).abs();
                if left_distance <= right_distance {
                    index - 1
                } else {
                    index
                }
            }
        }));
    }

    let mut nearest_index = 0usize;
    let mut nearest_distance = S::INFINITY;
    for (index, &node) in nodes.iter().enumerate() {
        if separation.coincides(query, node)? {
            return Ok(QueryReference::Coincident(index));
        }
        let distance = (query - node).abs();
        if distance < nearest_distance {
            nearest_index = index;
            nearest_distance = distance;
        }
    }
    Ok(QueryReference::Distinct(nearest_index))
}

#[cfg(test)]
fn maximum_magnitude<S: RealScalar>(values: &[S]) -> S {
    values
        .iter()
        .fold(S::ZERO, |scale, value| scale.max(value.abs()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, assert_error};

    use scioxide_core::Complex;

    fn value<S: RealScalar>(interp: &BarycentricInterpolator<S>, t: S) -> Result<S, SciOxideError> {
        interp.evaluate([t], ExtrapolationPolicy::Reject)
    }

    fn derivative<S: RealScalar>(
        interp: &BarycentricInterpolator<S>,
        t: S,
    ) -> Result<S, SciOxideError> {
        interp
            .gradient([t], ExtrapolationPolicy::Reject)
            .map(|[slope]| slope)
    }

    fn fused<S: RealScalar>(
        interp: &BarycentricInterpolator<S>,
        t: S,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(S, S), SciOxideError> {
        let (mut value, mut slope) = (S::ZERO, S::ZERO);
        interp.evaluate_with_derivative_into(
            [t],
            ExtrapolationPolicy::Reject,
            &mut value,
            &mut slope,
            workspace,
        )?;
        Ok((value, slope))
    }

    fn fused_owned<S: RealScalar>(
        interp: &BarycentricInterpolator<S>,
        t: S,
    ) -> Result<(S, S), SciOxideError> {
        fused(interp, t, &mut interp.workspace()?)
    }

    fn integral(interp: &BarycentricInterpolator<f64>, a: f64, b: f64) -> f64 {
        interp
            .prepare_antiderivative()
            .unwrap()
            .integral(a, b, ExtrapolationPolicy::Reject)
            .unwrap()
    }

    type StructuredConstructor =
        fn(Vec<f64>, f64, f64) -> Result<BarycentricInterpolator<f64>, SciOxideError>;

    fn structured_constructors() -> [StructuredConstructor; 3] {
        [
            BarycentricInterpolator::from_equispaced,
            BarycentricInterpolator::from_chebyshev_roots,
            BarycentricInterpolator::from_chebyshev_lobatto,
        ]
    }

    #[allow(clippy::cast_precision_loss)]
    fn equispaced_quadratic_fixture(
        n: usize,
        lower: f64,
        upper: f64,
    ) -> (Vec<f64>, Vec<f64>, f64, f64, f64) {
        let nodes: Vec<f64> = (0..n)
            .map(|index| {
                let unit = index as f64 / (n - 1) as f64;
                lower * (1.0 - unit) + upper * unit
            })
            .collect();
        let values: Vec<f64> = (0..n)
            .map(|index| {
                let coordinate = (2.0 * index as f64 / (n - 1) as f64) - 1.0;
                coordinate.mul_add(coordinate, 2.0 * coordinate) - 1.0
            })
            .collect();
        let unit_query = 0.65_f64;
        let coordinate_query = unit_query.mul_add(2.0, -1.0);
        let query = lower * (1.0 - unit_query) + upper * unit_query;
        let expected_value =
            coordinate_query.mul_add(coordinate_query, 2.0 * coordinate_query) - 1.0;
        let expected_derivative = (2.0 * coordinate_query + 2.0) * 2.0 / (upper - lower);
        (nodes, values, query, expected_value, expected_derivative)
    }

    #[test]
    fn test_new_valid_input() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 2.0];
        let interp = BarycentricInterpolator::new(x, y);
        assert!(interp.is_ok());
    }

    #[test]
    fn test_new_single_point() {
        let interp = BarycentricInterpolator::new(vec![1.0], vec![5.0]).unwrap();
        let v = value(&interp, 1.0).unwrap();
        assert_close(v, 5.0, 5e-15, 5e-15);
    }

    #[test]
    fn test_new_empty_input() {
        let result = BarycentricInterpolator::<f64>::new(vec![], vec![]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::InvalidInput);
        for structured in [
            BarycentricInterpolator::<f64>::from_equispaced(vec![], 0.0, 1.0),
            BarycentricInterpolator::<f64>::from_chebyshev_roots(vec![], 0.0, 1.0),
            BarycentricInterpolator::<f64>::from_chebyshev_lobatto(vec![], 0.0, 1.0),
        ] {
            assert!(matches!(
                assert_error(&structured, scioxide_core::SciOxideErrorKind::InvalidInput).view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: scioxide_core::InvalidInputReason::Empty,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_new_dimension_mismatch() {
        let result = BarycentricInterpolator::new(vec![0.0, 1.0], vec![1.0]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::DimensionMismatch);
        let shapes =
            BarycentricInterpolator::new(vec![0.0_f64, 1.0], vec![vec![1.0], vec![1.0, 2.0]]);
        assert_error(&shapes, scioxide_core::SciOxideErrorKind::DimensionMismatch);
    }

    #[test]
    fn test_new_non_monotone() {
        let result = BarycentricInterpolator::new(vec![0.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_new_nan_input() {
        let result = BarycentricInterpolator::new(vec![0.0, f64::NAN, 2.0], vec![1.0, 2.0, 3.0]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn structured_constructors_preserve_values_and_match_general_construction() {
        let values: Vec<f64> = (0_u16..17)
            .map(|index| (f64::from(index) * 0.25).sin())
            .collect();
        for constructor in structured_constructors() {
            let structured = constructor(values.clone(), -2.0, 3.0).unwrap();
            assert_eq!(structured.values(), values);
            assert!(structured.nodes().windows(2).all(|pair| pair[0] < pair[1]));
            let maximum_weight = maximum_magnitude(structured.weights());
            assert!((maximum_weight - 1.0).abs() < 4e-15);

            let general =
                BarycentricInterpolator::new(structured.nodes().to_vec(), values.clone()).unwrap();
            let common_sign = structured.weights()[0] / general.weights()[0];
            for (&structured_weight, &general_weight) in
                structured.weights().iter().zip(general.weights())
            {
                assert_close(
                    structured_weight,
                    common_sign * general_weight,
                    1e-12,
                    1e-12,
                );
            }
            for fraction in [0.125_f64, 0.375, 0.625, 0.875] {
                let query = structured.nodes()[0] * (1.0 - fraction)
                    + structured.nodes()[structured.nodes().len() - 1] * fraction;
                let structured_result = fused_owned(&structured, query).unwrap();
                let general_result = fused_owned(&general, query).unwrap();
                assert_close(structured_result.0, general_result.0, 1e-10, 1e-10);
                assert_close(structured_result.1, general_result.1, 2.5e-9, 2.5e-9);
            }
            for (&node, &expected) in structured.nodes().iter().zip(&values) {
                assert_eq!(value(&structured, node).unwrap(), expected);
            }
        }
    }

    #[test]
    fn structured_node_family_formulas_are_increasing_and_normalized() {
        let equispaced = BarycentricInterpolator::from_equispaced(vec![0.0; 5], -1.0, 1.0).unwrap();
        assert_eq!(equispaced.nodes(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
        for (&actual, expected) in
            equispaced
                .weights()
                .iter()
                .zip([1.0 / 6.0, -2.0 / 3.0, 1.0, -2.0 / 3.0, 1.0 / 6.0])
        {
            assert_close(actual, expected, 1e-15, 1e-15);
        }

        let roots = BarycentricInterpolator::from_chebyshev_roots(vec![0.0; 4], -1.0, 1.0).unwrap();
        let root_angles =
            [1.0_f64, 3.0, 5.0, 7.0].map(|factor| -(factor * std::f64::consts::PI / 8.0).cos());
        for (&actual, expected) in roots.nodes().iter().zip(root_angles) {
            assert_close(actual, expected, 1e-15, 1e-15);
        }

        let lobatto =
            BarycentricInterpolator::from_chebyshev_lobatto(vec![0.0; 4], -1.0, 1.0).unwrap();
        for (&actual, expected) in lobatto.nodes().iter().zip([-1.0, -0.5, 0.5, 1.0]) {
            assert_close(actual, expected, 1e-15, 1e-15);
        }
        assert_eq!(lobatto.weights(), &[0.5, -1.0, 1.0, -0.5]);
    }

    #[test]
    fn structured_singleton_convention_uses_interval_midpoint() {
        for constructor in structured_constructors() {
            let interpolator = constructor(vec![7.0], -4.0, 10.0).unwrap();
            assert_eq!(interpolator.nodes(), &[3.0]);
            assert_eq!(interpolator.weights(), &[1.0]);
            assert_eq!(interpolator.values(), &[7.0]);
        }

        let singleton =
            BarycentricInterpolator::from_chebyshev_lobatto(vec![2.0], 5.0, 5.0).unwrap();
        assert_eq!(singleton.nodes(), &[5.0]);
    }

    #[test]
    fn structured_constructors_validate_values_and_bounds() {
        let argument = |result: &Result<BarycentricInterpolator<f64>, SciOxideError>| {
            result.as_ref().err().and_then(|error| match error.view() {
                scioxide_core::SciOxideErrorView::InvalidInput { argument, .. } => {
                    Some(argument.as_str())
                }
                _ => None,
            })
        };
        for constructor in structured_constructors() {
            assert_error(
                &constructor(Vec::new(), -1.0, 1.0),
                scioxide_core::SciOxideErrorKind::InvalidInput,
            );
            assert_error(
                &constructor(vec![0.0, f64::NAN], -1.0, 1.0),
                scioxide_core::SciOxideErrorKind::NonFinite,
            );
            assert_error(
                &constructor(vec![0.0], f64::NAN, 1.0),
                scioxide_core::SciOxideErrorKind::NonFinite,
            );
            assert_eq!(
                argument(&constructor(vec![0.0, 1.0], 2.0, 2.0)),
                Some("upper")
            );
            assert_eq!(argument(&constructor(vec![0.0], 1.0, -1.0)), Some("lower"));
        }
    }

    #[test]
    fn structured_constructors_support_f32_and_detect_unrepresentable_weights() {
        let roots =
            BarycentricInterpolator::from_chebyshev_roots(vec![1.0_f32; 17], -1.0, 1.0).unwrap();
        assert!(roots.nodes().windows(2).all(|pair| pair[0] < pair[1]));
        assert!(roots.weights().iter().all(|weight| weight.is_finite()));

        assert_error(
            &BarycentricInterpolator::from_equispaced(vec![0.0_f32; 512], -1.0, 1.0),
            scioxide_core::SciOxideErrorKind::NumericalInstability,
        );
    }

    #[test]
    fn test_interpolate_at_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 2.0, 5.0];
        let interp = BarycentricInterpolator::new(x.clone(), y.clone()).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let v = value(&interp, *xi).unwrap();
            assert_close(v, *yi, 5e-15, 5e-15);
        }
    }

    #[test]
    fn test_interpolate_near_node_uses_nodal_value() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let y = vec![2.0_f64, 4.0, 6.0];
        let tolerance = scioxide_core::Tolerance::new(1e-14, 1e-14).unwrap();
        let interp = BarycentricInterpolator::with_options(
            x,
            y,
            SampleOptions::new().with_node_separation(NodeSeparation::Tolerant(tolerance)),
        )
        .unwrap();
        let t = 4.0_f64.mul_add(f64::EPSILON, 1.0);
        let v = value(&interp, t).unwrap();
        assert_close(v, 2.0, 5e-15, 5e-15);
    }

    #[test]
    fn test_interpolate_linear() {
        // y = 2x + 1
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        let interp = BarycentricInterpolator::new(x, y).unwrap();
        assert_close(value(&interp, 0.5).unwrap(), 2.0, 5e-15, 5e-15);
        assert_close(value(&interp, 1.5).unwrap(), 4.0, 5e-15, 5e-15);
    }

    #[test]
    fn test_interpolate_out_of_domain() {
        let interp = BarycentricInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        assert_error(
            &value(&interp, -0.1),
            scioxide_core::SciOxideErrorKind::DomainError,
        );
        assert_error(
            &value(&interp, 1.1),
            scioxide_core::SciOxideErrorKind::DomainError,
        );
    }

    #[test]
    fn test_derivative_linear() {
        // y = 3x + 2 => y' = 3
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![2.0, 5.0, 8.0];
        let interp = BarycentricInterpolator::new(x, y).unwrap();
        assert_close(derivative(&interp, 0.5).unwrap(), 3.0, 5e-13, 5e-13);
        assert_close(derivative(&interp, 0.0).unwrap(), 3.0, 5e-13, 5e-13);
    }

    #[test]
    fn test_derivative_quadratic() {
        // y = x^2 => y' = 2x
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let interp = BarycentricInterpolator::new(x, y).unwrap();
        assert_close(derivative(&interp, 1.5).unwrap(), 3.0, 5e-11, 5e-11);
    }

    #[test]
    fn clamped_derivatives_vanish_outside_and_values_extend() {
        let interp = BarycentricInterpolator::new(vec![0.0, 1.0], vec![1.0, 3.0]).unwrap();
        let mut workspace = interp.workspace().unwrap();
        let (mut value, mut slope) = (0.0, 0.0);
        interp
            .evaluate_with_derivative_into(
                [2.0],
                ExtrapolationPolicy::Clamp,
                &mut value,
                &mut slope,
                &mut workspace,
            )
            .unwrap();
        assert_eq!((value, slope), (3.0, 0.0));
        assert_eq!(
            interp.gradient([-1.0], ExtrapolationPolicy::Clamp).unwrap(),
            [0.0]
        );
        assert_eq!(
            interp.gradient([1.0], ExtrapolationPolicy::Clamp).unwrap(),
            [2.0]
        );
    }

    #[test]
    fn fused_value_and_derivative_match_separate_calls_without_allocation() {
        let nodes = vec![-1.0_f64, -0.25, 0.5, 1.25, 2.0];
        let values: Vec<f64> = nodes
            .iter()
            .map(|&node| node.mul_add(node, 2.0 * node) - 3.0)
            .collect();
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
        let query = 0.125_f64;
        let expected_value = value(&interpolator, query).unwrap();
        let expected_derivative = derivative(&interpolator, query).unwrap();

        let mut workspace = interpolator.workspace().unwrap();
        let fused = fused(&interpolator, query, &mut workspace).unwrap();
        let value = {
            let mut out = 0.0;
            interpolator
                .evaluate_into(
                    [query],
                    ExtrapolationPolicy::Reject,
                    &mut out,
                    &mut workspace,
                )
                .map(|()| out)
        }
        .unwrap();

        assert_eq!(fused, (expected_value, expected_derivative));
        assert_eq!(value, expected_value);
    }

    #[test]
    fn fused_common_path_reproduces_cubic_value_and_derivative() {
        let nodes = vec![-1.0_f64, -0.25, 0.5, 1.25, 2.0];
        let values = nodes.iter().map(|x| x * x * x - 2.0 * x + 1.0).collect();
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
        let query = 0.125;
        let (value, derivative) = fused_owned(&interpolator, query).unwrap();
        assert_close(
            value,
            query * query * query - 2.0 * query + 1.0,
            16. * f64::EPSILON,
            0.,
        );
        assert_close(
            derivative,
            3.0 * query * query - 2.0,
            32. * f64::EPSILON,
            0.,
        );
    }

    #[test]
    fn nodal_fused_evaluation_is_allocation_free() {
        let nodes = vec![0.0_f64, 1.0, 2.0, 3.0];
        let values: Vec<f64> = nodes.iter().map(|node| node * node).collect();
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
        let mut workspace = interpolator.workspace().unwrap();

        let (value, derivative) = fused(&interpolator, 2.0, &mut workspace).unwrap();

        assert_eq!(value, 4.0);
        assert_close(derivative, 4.0, 5e-14, 5e-14);
    }

    #[test]
    fn exact_policy_does_not_snap_an_adjacent_query() {
        let nodes = vec![1.0_f64, 1.5, 2.0];
        let values = vec![10.0_f64, 20.0, 40.0];
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
        let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);

        let (value, derivative) = fused_owned(&interpolator, adjacent).unwrap();

        assert_ne!(value, 10.0);
        assert!(value > 10.0);
        assert!(derivative.is_finite());
    }

    #[test]
    fn clustered_and_extreme_fixtures_preserve_allocation_free_results() {
        let (nodes, values, query, expected_value, expected_derivative) =
            equispaced_quadratic_fixture(500, -1.0, 1.0);
        let fixtures = [
            (nodes, values, query, expected_value, expected_derivative),
            {
                let (nodes, _, query, _, _) = equispaced_quadratic_fixture(500, 1.0, 1.0 + 1e-8);
                (nodes, vec![1.0; 500], query, 1.0, 0.0)
            },
            {
                let (nodes, _, query, _, _) =
                    equispaced_quadratic_fixture(500, 1e150, 1e150 + 1e140);
                (nodes, vec![1.0; 500], query, 1.0, 0.0)
            },
        ];
        for (nodes, values, query, expected_value, expected_derivative) in fixtures {
            let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();

            let mut workspace = interpolator.workspace().unwrap();
            let (value, derivative) = fused(&interpolator, query, &mut workspace).unwrap();
            assert_close(value, expected_value, 2e-6, 0.);
            assert_close(derivative, expected_derivative, 1e-3, 0.);
        }
    }

    #[test]
    fn fused_evaluation_preserves_extreme_value_scaling() {
        let nodes = vec![-1.0_f64, -0.25, 0.5, 1.0];
        let scale = 1e300_f64;
        let values: Vec<f64> = nodes
            .iter()
            .map(|&node| scale * node.mul_add(2.0, 1.0))
            .collect();
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
        let query = 0.125_f64;

        let (value, derivative) = fused_owned(&interpolator, query).unwrap();

        assert!((value / scale - 1.25).abs() < 2e-15);
        assert!((derivative / scale - 2.0).abs() < 2e-14);
    }

    #[test]
    fn deterministic_polynomial_properties_match_value_and_derivative() {
        fn polynomial_with_derivative(coefficients: &[f64], x: f64) -> (f64, f64) {
            coefficients.iter().rev().fold(
                (0.0_f64, 0.0_f64),
                |(value, derivative), &coefficient| {
                    (value.mul_add(x, coefficient), derivative.mul_add(x, value))
                },
            )
        }

        let mut state = 0x4752_4144_4941_0217_u64;
        for _ in 0..16 {
            let coefficients: Vec<f64> = (0..8)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = (state >> 11) as f64 / ((1_u64 << 53) as f64);
                    unit.mul_add(2.0, -1.0)
                })
                .collect();
            let nodes: Vec<f64> = (0_u16..8)
                .map(|index| -(std::f64::consts::PI * (f64::from(index) + 0.5) / 8.0).cos())
                .collect();
            let values: Vec<f64> = nodes
                .iter()
                .map(|&node| polynomial_with_derivative(&coefficients, node).0)
                .collect();
            let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let query = ((state >> 11) as f64 / ((1_u64 << 53) as f64)).mul_add(1.8, -0.9);
            let expected = polynomial_with_derivative(&coefficients, query);
            let actual = fused_owned(&interpolator, query).unwrap();

            assert_close(actual.0, expected.0, 1.5e-13, 1.5e-13);
            assert_close(actual.1, expected.1, 1e-12, 1e-12);
        }
    }

    #[test]
    fn test_integral_linear() {
        // y = x => integral from 0 to 2 = 2
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let interp = BarycentricInterpolator::new(x, y).unwrap();
        assert_close(integral(&interp, 0.0, 2.0), 2.0, 5e-13, 5e-13);
    }

    #[test]
    fn test_integral_quadratic() {
        // y = x^2 => integral from 0 to 3 = 9
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let interp = BarycentricInterpolator::new(x, y).unwrap();
        assert_close(integral(&interp, 0.0, 3.0), 9.0, 5e-11, 5e-11);
    }

    #[test]
    fn test_integral_reversed_bounds() {
        let interp = BarycentricInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let prepared = interp.prepare_antiderivative().unwrap();
        assert_error(
            &prepared.integral(1.0, 0.0, ExtrapolationPolicy::Reject),
            scioxide_core::SciOxideErrorKind::InvalidInput,
        );
    }

    #[test]
    fn test_f32_interpolation() {
        let x: Vec<f32> = vec![0.0, 1.0, 2.0];
        let y: Vec<f32> = vec![1.0, 3.0, 5.0];
        let interp = BarycentricInterpolator::new(x, y).unwrap();
        assert!((value(&interp, 0.5).unwrap() - 2.0_f32).abs() < 1e-5);
    }

    #[test]
    fn high_degree_chebyshev_nodes_remain_stable_f64() {
        let degree = 128_u16;
        let mut nodes: Vec<f64> = (0..=degree)
            .map(|index| (std::f64::consts::PI * f64::from(index) / f64::from(degree)).cos())
            .collect();
        nodes.reverse();
        let values: Vec<f64> = nodes.iter().map(|node| node.exp()).collect();
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();

        assert!(
            interpolator
                .weights()
                .iter()
                .all(|weight| weight.is_finite() && *weight != 0.0)
        );
        let query = 0.123_456_789_f64;
        assert!((value(&interpolator, query).unwrap() - query.exp()).abs() < 2e-13);
    }

    #[test]
    fn high_degree_chebyshev_nodes_remain_stable_f32() {
        let degree = 64_u16;
        let mut nodes: Vec<f32> = (0..=degree)
            .map(|index| (std::f32::consts::PI * f32::from(index) / f32::from(degree)).cos())
            .collect();
        nodes.reverse();
        let values: Vec<f32> = nodes.iter().map(|node| node.exp()).collect();
        let interpolator = BarycentricInterpolator::new(nodes, values).unwrap();
        let query = 0.234_5_f32;
        assert!((value(&interpolator, query).unwrap() - query.exp()).abs() < 2e-5);
    }

    #[test]
    fn clustered_and_extremely_scaled_nodes_evaluate_finitely() {
        let clustered = vec![1.0_f64, 1.0 + 1e-10, 1.0 + 2e-10, 1.0 + 4e-10];
        let clustered_values: Vec<f64> =
            clustered.iter().map(|node| (node - 1.0) / 1e-10).collect();
        let clustered_interpolator =
            BarycentricInterpolator::new(clustered, clustered_values).unwrap();
        let clustered_query = 1.0 + 1.5e-10;
        let mut workspace = clustered_interpolator.workspace().unwrap();
        let (clustered_value, clustered_derivative) =
            fused(&clustered_interpolator, clustered_query, &mut workspace).unwrap();
        assert!((clustered_value - (clustered_query - 1.0) / 1e-10).abs() < 1e-10);
        assert!((clustered_derivative * 1e-10 - 1.0).abs() < 2e-15);

        let base = 1e150_f64;
        let spacing = 1e140_f64;
        let scaled = vec![
            base,
            base + spacing,
            base + 2.0 * spacing,
            base + 4.0 * spacing,
        ];
        let scaled_values: Vec<f64> = scaled.iter().map(|node| (node - base) / spacing).collect();
        let scaled_interpolator = BarycentricInterpolator::new(scaled, scaled_values).unwrap();
        let scaled_query = base + 1.5 * spacing;
        let mut workspace = scaled_interpolator.workspace().unwrap();
        let (scaled_value, scaled_derivative) =
            fused(&scaled_interpolator, scaled_query, &mut workspace).unwrap();
        assert!(scaled_value.is_finite());
        assert!((scaled_value - (scaled_query - base) / spacing).abs() < 1e-10);
        assert!(scaled_derivative.is_finite());
        assert!((scaled_derivative * spacing - 1.0).abs() < 2e-15);
    }

    #[test]
    fn node_separation_and_extrapolation_are_explicit() {
        let nodes = vec![1.0_f64, 1.0 + 1e-12, 2.0];
        let values = vec![1.0_f64, 1.0, 2.0];
        assert!(BarycentricInterpolator::new(nodes.clone(), values.clone()).is_ok());
        let tolerance = scioxide_core::Tolerance::new(1e-10, 0.0).unwrap();
        assert_error(
            &BarycentricInterpolator::with_options(
                nodes,
                values,
                SampleOptions::new().with_node_separation(NodeSeparation::Tolerant(tolerance)),
            ),
            scioxide_core::SciOxideErrorKind::InvalidInput,
        );

        let linear = BarycentricInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        assert_error(
            &value(&linear, 2.0),
            scioxide_core::SciOxideErrorKind::DomainError,
        );
        assert_eq!(
            linear
                .evaluate([2.0], ExtrapolationPolicy::MethodDefined)
                .unwrap(),
            2.0
        );
    }

    #[test]
    fn affine_components_match_scalar_interpolants_bit_for_bit() {
        // Components with very different magnitudes keep independent scales,
        // including the nodal, direct and extended-range paths.
        let nodes = vec![-1.0_f64, -0.25, 0.5, 1.25, 2.0];
        let first: Vec<f64> = nodes.iter().map(|x| x.exp()).collect();
        let second: Vec<f64> = nodes.iter().map(|x| 1e300 * (x * 3.0).sin()).collect();
        let values: Vec<Complex<f64>> = first
            .iter()
            .zip(&second)
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        let complex = BarycentricInterpolator::new(nodes.clone(), values).unwrap();
        let real = BarycentricInterpolator::new(nodes.clone(), first).unwrap();
        let imaginary = BarycentricInterpolator::new(nodes, second).unwrap();
        for query in [-1.0, -0.6, 0.125, 0.5, 1.9, 2.5] {
            let policy = ExtrapolationPolicy::MethodDefined;
            let value = complex.evaluate([query], policy).unwrap();
            assert_eq!(
                value.re.to_bits(),
                real.evaluate([query], policy).unwrap().to_bits()
            );
            assert_eq!(
                value.im.to_bits(),
                imaginary.evaluate([query], policy).unwrap().to_bits()
            );
            let [slope] = complex.gradient([query], policy).unwrap();
            assert_eq!(
                slope.re.to_bits(),
                real.gradient([query], policy).unwrap()[0].to_bits()
            );
            assert_eq!(
                slope.im.to_bits(),
                imaginary.gradient([query], policy).unwrap()[0].to_bits()
            );
        }
    }
}
