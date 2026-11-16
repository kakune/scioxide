//! The interpolation traits, the extrapolation policy and coordinate helpers.

use alloc::vec::Vec;

use scioxide_core::backend::{absolute_difference_le, try_with_capacity};
use scioxide_core::{AffineValueMut, Capacity, Interval, RealScalar, SciOxideError, Tolerance};

/// Selects how close two neighboring nodes may be.
///
/// Nodes must be strictly increasing under both variants. `Tolerant(t)` also
/// rejects adjacent nodes `xᵢ < xᵢ₊₁` with
///
/// ```text
/// xᵢ₊₁ − xᵢ ≤ atol + rtol · max(|xᵢ|, |xᵢ₊₁|),
/// ```
///
/// where `(atol, rtol)` is the [`Tolerance`] pair. Order is checked first, so a
/// decreasing pair is an ordering error under either variant. Both errors are
/// `InvalidInput` (`Inconsistent`). The constructors that take samples receive
/// the separation through [`SampleOptions`]; their `new` uses `Exact`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_interp::{BilinearInterpolator, NodeSeparation, SampleOptions};
/// let (x, y) = (vec![0.0_f64, 1e-12, 1.0], vec![0.0, 1.0]);
/// assert!(BilinearInterpolator::new(x.clone(), y.clone(), vec![0.0; 6]).is_ok());
/// // A gap of 1e-12 is below the absolute tolerance 1e-9.
/// let tolerant = NodeSeparation::Tolerant(Tolerance::new(1e-9, 0.0)?);
/// let options = SampleOptions::new().with_node_separation(tolerant);
/// assert!(BilinearInterpolator::with_options(x, y, vec![0.0; 6], options).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum NodeSeparation<S: RealScalar> {
    /// Requires `xᵢ < xᵢ₊₁` and nothing more.
    Exact,
    /// Also rejects gaps of at most `atol + rtol · max(|xᵢ|, |xᵢ₊₁|)`.
    Tolerant(Tolerance<S>),
}

impl<S: RealScalar> NodeSeparation<S> {
    pub(crate) fn is_zero_at_scale(&self, value: S, scale: S) -> Result<bool, SciOxideError> {
        let magnitude = value.abs();
        match self {
            Self::Exact => Ok(magnitude == S::ZERO),
            Self::Tolerant(tolerance) => Ok(magnitude <= tolerance.threshold(scale.abs())?),
        }
    }

    pub(crate) fn coincides(&self, left: S, right: S) -> Result<bool, SciOxideError> {
        match self {
            Self::Exact => Ok(left == right),
            Self::Tolerant(tolerance) => {
                // The threshold is evaluated before the equality shortcut so an
                // unrepresentable tolerance is reported consistently, not only
                // for distinct nodes.
                let limit = tolerance.threshold(left.abs().max(right.abs()))?;
                Ok(left == right || absolute_difference_le(left, right, limit))
            }
        }
    }
}

/// Sets how the constructors that take samples validate their nodes.
///
/// Each interpolant built from samples on a line or a rectilinear grid has a
/// `with_options` constructor that takes these options, such as
/// [`CubicSpline::with_options`](crate::CubicSpline::with_options); its `new`
/// uses [`SampleOptions::new`]. The one setting is the [`NodeSeparation`] of
/// adjacent nodes, [`NodeSeparation::Exact`] by default.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_interp::{NodeSeparation, PchipInterpolator, SampleOptions};
/// let exact = SampleOptions::<f64>::default().node_separation();
/// assert_eq!(exact, NodeSeparation::Exact);
/// let tolerant = NodeSeparation::Tolerant(Tolerance::new(0.0, 1e-9)?);
/// let options = SampleOptions::new().with_node_separation(tolerant);
/// // The gap 1e-12 at x = 1 is below the relative tolerance 1e-9.
/// let (x, y) = (vec![0.0_f64, 1.0, 1.0 + 1e-12], vec![0.0, 1.0, 2.0]);
/// assert!(PchipInterpolator::new(x.clone(), y.clone()).is_ok());
/// assert!(PchipInterpolator::with_options(x, y, options).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SampleOptions<S: RealScalar> {
    node_separation: NodeSeparation<S>,
}

impl<S: RealScalar> SampleOptions<S> {
    /// Returns options with [`NodeSeparation::Exact`].
    pub const fn new() -> Self {
        Self {
            node_separation: NodeSeparation::Exact,
        }
    }

    /// Sets how close two adjacent nodes may be.
    pub const fn with_node_separation(mut self, separation: NodeSeparation<S>) -> Self {
        self.node_separation = separation;
        self
    }

    /// Returns how close two adjacent nodes may be.
    pub const fn node_separation(&self) -> NodeSeparation<S> {
        self.node_separation
    }
}

impl<S: RealScalar> Default for SampleOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Selects what a query outside the data domain returns.
///
/// The domain is closed, so points on its boundary are inside. It is the box
/// spanned by the nodes for one-dimensional and grid interpolants, and the
/// convex hull of the sites for triangulation-based interpolants. Every query
/// takes a policy; there is no default. See [Extrapolation](crate#extrapolation)
/// for derivatives and integrals outside the domain.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BilinearInterpolator, ExtrapolationPolicy, Interpolant};
/// // f(x, y) = x on the unit square.
/// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 0.0, 1.0])?;
/// assert!(
///     grid.evaluate([1.5, 0.5], ExtrapolationPolicy::Reject)
///         .is_err()
/// );
/// assert_eq!(grid.evaluate([1.5, 0.5], ExtrapolationPolicy::Clamp)?, 1.0);
/// let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 0.5 };
/// assert_eq!(grid.evaluate([1.5, 0.5], bounded)?, 1.5);
/// assert_eq!(
///     grid.evaluate([1.5, 0.5], ExtrapolationPolicy::MethodDefined)?,
///     1.5
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum ExtrapolationPolicy<S: RealScalar> {
    /// Returns `DomainError` for a query outside the domain.
    Reject,
    /// Evaluates at the nearest point of the domain, so the interpolant is
    /// extended by a constant. Derivatives along a clamped axis are zero.
    Clamp,
    /// Continues the interpolant's representation outside the domain, up to
    /// and including `max_distance` along each axis, and returns `DomainError`
    /// for farther queries.
    BoundedContinue {
        /// Finite, nonnegative distance in coordinate units.
        max_distance: S,
    },
    /// Continues the global formula, or the outermost polynomial piece of a
    /// piecewise interpolant, without a distance limit. A pole or an overflow
    /// outside the domain is still an error.
    MethodDefined,
}

/// Describes where one query lies relative to the domain.
///
/// The report holds the query point, the point the method evaluates (they
/// differ only under [`ExtrapolationPolicy::Clamp`]) and the distance outside
/// the domain along each axis. Distances are nonnegative, and a distance too
/// large for the scalar type is `+∞`. The report describes geometry only; it
/// is not an error estimate.
///
/// [`ExtrapolationPolicy::resolve`] measures against a box of coordinate
/// intervals. [`Interpolant::report`] and
/// [`Interpolant::evaluate_into_with_report`] measure against the method's own
/// domain, for example a convex hull.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_interp::ExtrapolationPolicy;
/// let domain = Interval::new(0.0_f64, 2.0)?;
/// let report = ExtrapolationPolicy::Clamp.resolve([domain], [3.0])?;
/// assert_eq!(report.original(), [3.0]);
/// assert_eq!(report.effective(), [2.0]);
/// assert_eq!(report.outside_distance(), [1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ExtrapolationReport<S: RealScalar, const D: usize> {
    original: [S; D],
    effective: [S; D],
    outside_distance: [S; D],
}

impl<S: RealScalar, const D: usize> ExtrapolationReport<S, D> {
    /// Assemble a report from validated parts.
    pub(crate) const fn new(original: [S; D], effective: [S; D], outside_distance: [S; D]) -> Self {
        Self {
            original,
            effective,
            outside_distance,
        }
    }

    /// Report for an interior query: no projection and zero distance.
    pub(crate) const fn interior(point: [S; D]) -> Self {
        Self::new(point, point, [S::ZERO; D])
    }

    /// Returns the query coordinates.
    #[must_use]
    pub const fn original(&self) -> [S; D] {
        self.original
    }

    /// Returns the coordinates the method evaluates; they differ from
    /// [`Self::original`] only under [`ExtrapolationPolicy::Clamp`].
    #[must_use]
    pub const fn effective(&self) -> [S; D] {
        self.effective
    }

    /// Returns the distance outside the domain along each axis, zero inside.
    #[must_use]
    pub const fn outside_distance(&self) -> [S; D] {
        self.outside_distance
    }
}

impl<S: RealScalar> ExtrapolationPolicy<S> {
    /// Applies the policy to `coordinates` in the box spanned by `domains`.
    ///
    /// Clamping projects each coordinate onto its interval.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `max_distance` of
    ///   [`Self::BoundedContinue`] is negative or not finite.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `DomainError` if the point is outside the box under [`Self::Reject`],
    ///   or farther than `max_distance` under [`Self::BoundedContinue`].
    pub fn resolve<const D: usize>(
        self,
        domains: [Interval<S>; D],
        coordinates: [S; D],
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        self.validate(coordinates)?;
        let boundaries = core::array::from_fn(|axis| {
            let coordinate = coordinates[axis];
            if domains[axis].contains(coordinate) {
                coordinate
            } else {
                coordinate
                    .max(domains[axis].lower())
                    .min(domains[axis].upper())
            }
        });
        self.resolve_projected(coordinates, boundaries)
    }

    /// Checks the policy and the coordinates before the domain is examined.
    pub(crate) fn validate<const D: usize>(self, coordinates: [S; D]) -> Result<(), SciOxideError> {
        if let Self::BoundedContinue { max_distance } = self
            && (!max_distance.is_finite() || max_distance < S::ZERO)
        {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("policy"),
                "continuation distance must be finite and nonnegative",
            ));
        }
        if coordinates.iter().any(|x| !x.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "query coordinates must be finite",
            ));
        }
        Ok(())
    }

    /// Apply the policy to validated coordinates and their geometric projection.
    /// The projection is coordinate-wise for grids and Euclidean for hulls.
    pub(crate) fn resolve_projected<const D: usize>(
        self,
        coordinates: [S; D],
        boundaries: [S; D],
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        let mut report = ExtrapolationReport::interior(coordinates);
        for (axis, (&coordinate, &boundary)) in coordinates.iter().zip(&boundaries).enumerate() {
            if coordinate == boundary {
                continue;
            }
            let distance = (coordinate - boundary).abs();
            report.outside_distance[axis] = distance;
            let rejected = match self {
                Self::Reject => true,
                Self::BoundedContinue { max_distance } => {
                    distance_exceeds(coordinate, boundary, distance, max_distance)
                }
                Self::Clamp | Self::MethodDefined => false,
            };
            if rejected {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("coordinates"),
                    scioxide_core::DomainReason::OutsideDomain,
                    "query exceeds the permitted interpolation domain",
                ));
            }
            if self == Self::Clamp {
                report.effective[axis] = boundary;
            }
        }
        Ok(report)
    }
}

// Error-free two-difference residual distinguishes an exact boundary distance
// from a larger distance that rounds down to the same representable limit.
fn distance_exceeds<S: RealScalar>(coordinate: S, boundary: S, rounded: S, limit: S) -> bool {
    if rounded != limit {
        return rounded > limit;
    }
    let high = coordinate.max(boundary);
    let low = coordinate.min(boundary);
    let virtual_low = high - rounded;
    let virtual_high = rounded + virtual_low;
    let residual = (high - virtual_high) + (virtual_low - low);
    residual > S::ZERO
}

/// Evaluates a function of `D` real coordinates built from data.
///
/// Every interpolant in this crate implements this trait. A value is any
/// [`scioxide_core::AffineValue`]: a real or complex scalar, an array or a
/// vector, with [`Self::value_dimension`] real components.
///
/// [`Self::evaluate_into`] writes into a value you provide and uses a
/// workspace from [`Self::workspace`]. A workspace has a fixed size and never
/// grows. One workspace serves one query at a time, so give each thread its
/// own. [`Self::evaluate`] allocates a workspace and the result on every call;
/// [`Self::evaluate_many`] allocates one workspace for all its points.
/// [`crate::BatchEvaluator`] evaluates many points into caller storage,
/// optionally in parallel. A function of one variable also has the scalar
/// forms of [`Univariate`].
///
/// The generic output parameter means these traits cannot be used as trait
/// objects; write generic code with `I: Interpolant<S, D>`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BilinearInterpolator, ExtrapolationPolicy, Interpolant};
/// // f(x, y) = x + 2y on the unit square.
/// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 2.0, 3.0])?;
/// assert_eq!(grid.evaluate([0.5, 0.5], ExtrapolationPolicy::Reject)?, 1.5);
/// let mut workspace = grid.workspace()?;
/// let mut out = 0.0;
/// grid.evaluate_into(
///     [1.0, 1.0],
///     ExtrapolationPolicy::Reject,
///     &mut out,
///     &mut workspace,
/// )?;
/// assert_eq!(out, 3.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Interpolant<S: RealScalar, const D: usize>: Sync {
    /// The value type returned by the owned forms.
    type Value: AffineValueMut<Scale = S>;
    /// Scratch storage for one query at a time.
    type Workspace: Send;

    /// Returns the closed coordinate interval spanned by the data on each axis.
    fn domains(&self) -> [Interval<S>; D];

    /// Returns the number of real components of one value.
    fn value_dimension(&self) -> usize;

    /// Returns a zero value of this interpolant's value shape.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if a heap-allocated value cannot be created.
    fn zero_value(&self) -> Result<Self::Value, SciOxideError>;

    /// Allocates a workspace large enough for every query on this interpolant.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the workspace cannot be
    ///   allocated.
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError>;

    /// Returns whether `workspace` is large enough for this interpolant.
    fn workspace_capacity(&self, workspace: &Self::Workspace) -> Capacity;

    /// Evaluates the interpolant at `point` and writes the value to `out`.
    ///
    /// `out` must have [`Self::value_dimension`] real components.
    ///
    /// # Errors
    ///
    /// - The coordinate errors of [`ExtrapolationPolicy::resolve`], measured
    ///   against the method's domain.
    /// - `DimensionMismatch` if `out` has the wrong number of components or
    ///   `workspace` is too small.
    /// - The numerical errors documented by the interpolant, such as a pole of a
    ///   rational function.
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;

    /// Returns where `point` lies relative to the method's domain.
    ///
    /// # Errors
    ///
    /// - The coordinate errors of [`Self::evaluate_into`].
    fn report(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        workspace: &mut Self::Workspace,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        let _ = workspace;
        policy.resolve(self.domains(), point)
    }

    /// Evaluates the interpolant at `point` and returns its [`ExtrapolationReport`].
    ///
    /// This gives the same value as [`Self::evaluate_into`] and the same report
    /// as [`Self::report`] in one call.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BilinearInterpolator, ExtrapolationPolicy, Interpolant};
    /// // f(x, y) = x + y on the unit square.
    /// let model =
    ///     BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 1.0, 2.0])?;
    /// let mut workspace = model.workspace()?;
    /// let mut value = 0.0;
    /// // Clamping moves (2, 0.5) to (1, 0.5).
    /// let report = model.evaluate_into_with_report(
    ///     [2.0, 0.5],
    ///     ExtrapolationPolicy::Clamp,
    ///     &mut value,
    ///     &mut workspace,
    /// )?;
    /// assert_eq!(report.effective(), [1.0, 0.5]);
    /// assert_eq!(value, 1.5);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn evaluate_into_with_report<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        let report = self.report(point, policy, workspace)?;
        self.evaluate_into(point, policy, out, workspace)?;
        Ok(report)
    }

    /// Evaluates the interpolant at `point` and returns the value.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_into`].
    fn evaluate(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<Self::Value, SciOxideError> {
        let mut out = self.zero_value()?;
        let mut workspace = self.workspace()?;
        self.evaluate_into(point, policy, &mut out, &mut workspace)?;
        Ok(out)
    }

    /// Evaluates the interpolant at each of `points` and returns the values in
    /// order.
    ///
    /// Each value equals [`Self::evaluate`] at its point bit for bit. The
    /// points are evaluated in order on the calling thread with one workspace,
    /// so the call allocates the result, one workspace and the values.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_into`] at the first point that fails;
    ///   no values are returned.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BilinearInterpolator, ExtrapolationPolicy, Interpolant};
    /// // f(x, y) = x + 2y on the unit square.
    /// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 2.0, 3.0])?;
    /// let points = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
    /// let values = grid.evaluate_many(&points, ExtrapolationPolicy::Reject)?;
    /// assert_eq!(values, [0.0, 1.5, 3.0]);
    /// assert!(
    ///     grid.evaluate_many(&[[2.0, 0.0]], ExtrapolationPolicy::Reject)
    ///         .is_err()
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn evaluate_many(
        &self,
        points: &[[S; D]],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<Vec<Self::Value>, SciOxideError> {
        let mut out = try_with_capacity(points.len(), scioxide_core::quantity!("values"))?;
        let mut workspace = self.workspace()?;
        for &point in points {
            let mut value = self.zero_value()?;
            self.evaluate_into(point, policy, &mut value, &mut workspace)?;
            out.push(value);
        }
        Ok(out)
    }
}

/// Evaluates the gradient of an interpolant.
///
/// The gradient holds `D` values, the partial derivatives `∂f/∂x₁, …, ∂f/∂x_D`
/// in coordinate order, so it has `D · value_dimension()` real components.
/// Derivatives are taken with respect to the coordinates as given, not with
/// respect to a normalized variable. Outside the domain they follow
/// [Extrapolation](crate#extrapolation). At a node or knot where the
/// interpolant is not differentiable, each interpolant documents which piece
/// it uses.
///
/// Bilinear grids are not differentiable:
///
/// ```compile_fail
/// use scioxide_interp::{BilinearInterpolator, Differentiable};
/// fn requires_gradient<I: Differentiable<f64, 2>>(_: &I) {}
/// let grid = BilinearInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0; 4]).unwrap();
/// requires_gradient(&grid);
/// ```
pub trait Differentiable<S: RealScalar, const D: usize>: Interpolant<S, D> {
    /// Evaluates the gradient at `point` and writes it to `out`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Interpolant::evaluate_into`], with
    ///   `D · value_dimension()` output components.
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;

    /// Evaluates the gradient at `point` and returns one value per coordinate.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::gradient_into`].
    fn gradient(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<[Self::Value; D], SciOxideError> {
        let mut out = try_array(|| self.zero_value())?;
        let mut workspace = self.workspace()?;
        self.gradient_into(point, policy, out.as_mut_slice(), &mut workspace)?;
        Ok(out)
    }
}

/// Evaluates the Hessian matrix of an interpolant.
///
/// The Hessian holds the `D × D` values `∂²f/∂xᵢ∂xⱼ` in row-major order, so it
/// has `D · D · value_dimension()` real components. Conventions for
/// coordinates and nonsmooth points are those of [`Differentiable`].
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BicubicInterpolator, ExtrapolationPolicy, TwiceDifferentiable};
/// // f(x, y) = x y has ∂²f/∂x∂y = 1 and ∂²f/∂x² = 0.
/// let axis = vec![0.0_f64, 1.0, 2.0, 3.0];
/// let z: Vec<f64> = axis
///     .iter()
///     .flat_map(|&y| axis.iter().map(move |&x| x * y))
///     .collect();
/// let surface = BicubicInterpolator::new(axis.clone(), axis, z)?;
/// let h = surface.hessian([1.5, 1.5], ExtrapolationPolicy::Reject)?;
/// assert!((h[0][1] - 1.0).abs() < 1e-12 && h[0][0].abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait TwiceDifferentiable<S: RealScalar, const D: usize>: Differentiable<S, D> {
    /// Evaluates the Hessian at `point` and writes it to `out`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Interpolant::evaluate_into`], with
    ///   `D · D · value_dimension()` output components.
    fn hessian_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;

    /// Evaluates the Hessian at `point` and returns it as rows.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::hessian_into`].
    fn hessian(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<[[Self::Value; D]; D], SciOxideError> {
        let mut out: [[Self::Value; D]; D] = try_array(|| try_array(|| self.zero_value()))?;
        let mut workspace = self.workspace()?;
        self.hessian_into(point, policy, out.as_flattened_mut(), &mut workspace)?;
        Ok(out)
    }
}

/// Integrates an interpolant over a box.
///
/// Computes `∫ f(x) dx` over the box `[lower₁, upper₁] × … × [lower_D, upper_D]`
/// in the interpolant's coordinates. The bounds are checked in this
/// order: finite, then `lowerᵢ ≤ upperᵢ` on every axis, then the extrapolation
/// policy. Under [`ExtrapolationPolicy::Clamp`] the part of the box outside the
/// domain contributes the constant boundary value times its length.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{ExtrapolationPolicy, Integrable, PchipInterpolator};
/// // f(x) = x on [0, 2], so ∫₀² f = 2.
/// let line = PchipInterpolator::new(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 2.0])?;
/// assert!((line.integral([0.0], [2.0], ExtrapolationPolicy::Reject)? - 2.0).abs() < 1e-14);
/// // The clamped extension contributes 2 * (3 - 2) beyond the right end.
/// assert!((line.integral([0.0], [3.0], ExtrapolationPolicy::Clamp)? - 4.0).abs() < 1e-14);
/// assert!(
///     line.integral([2.0], [0.0], ExtrapolationPolicy::Reject)
///         .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Integrable<S: RealScalar, const D: usize>: Interpolant<S, D> {
    /// Integrates over `[lower, upper]` and writes the integral to `out`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a bound is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `lowerᵢ > upperᵢ` on some axis.
    /// - The coordinate errors of [`Interpolant::evaluate_into`] for each bound.
    /// - `DimensionMismatch` if `out` has the wrong number of components.
    /// - The numerical errors documented by the interpolant.
    fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        lower: [S; D],
        upper: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;

    /// Integrates over `[lower, upper]` and returns the integral.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::integral_into`].
    fn integral(
        &self,
        lower: [S; D],
        upper: [S; D],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<Self::Value, SciOxideError> {
        let mut out = self.zero_value()?;
        let mut workspace = self.workspace()?;
        self.integral_into(lower, upper, policy, &mut out, &mut workspace)?;
        Ok(out)
    }
}

/// Evaluates an interpolant of one variable at scalar arguments on its domain.
///
/// These are the protocol queries with the point `[x]` and
/// [`ExtrapolationPolicy::Reject`]: [`Self::value_at`] is
/// [`Interpolant::evaluate`], [`Self::values_at`] is
/// [`Interpolant::evaluate_many`], [`Self::derivative_at`] is
/// [`Differentiable::gradient`] and [`Self::integrate`] is
/// [`Integrable::integral`], and each returns the same bits. They are defined
/// on the closed domain `[x₀, xₙ₋₁]` spanned by the nodes and never
/// extrapolate; clamping and continuation take a policy through the protocol
/// forms.
///
/// The trait has one blanket implementation for every [`Interpolant<S, 1>`],
/// so importing it is enough to call its methods on any interpolant of one
/// variable.
///
/// # Errors
///
/// The errors of the wrapped forms under [`ExtrapolationPolicy::Reject`]:
/// `DomainError` outside the domain and `NonFinite` for a NaN or infinite
/// argument.
///
/// # Allocation
///
/// As [`Interpolant::evaluate`]: a scalar method allocates nothing when its
/// scratch fits the inline storage of [`crate::QueryWorkspace`] and the value
/// owns no heap storage, as for a real piecewise cubic. [`Self::values_at`]
/// allocates the result and one workspace.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{CubicSpline, Univariate};
/// // A not-a-knot cubic spline reproduces f(x) = x³ on [0, 4].
/// let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
/// let y: Vec<f64> = x.iter().map(|&t| t * t * t).collect();
/// let spline = CubicSpline::new(x, y)?;
/// assert!((spline.value_at(2.5)? - 15.625).abs() < 1e-12);
/// assert!((spline.derivative_at(2.5)? - 18.75).abs() < 1e-12);
/// assert!((spline.integrate(0.0, 4.0)? - 64.0).abs() < 1e-12);
/// let values = spline.values_at(&[1.0, 2.0])?;
/// assert!((values[0] - 1.0).abs() < 1e-12 && (values[1] - 8.0).abs() < 1e-12);
/// assert!(spline.value_at(5.0).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Univariate<S: RealScalar>: Interpolant<S, 1> {
    /// Evaluates the interpolant at `x`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Interpolant::evaluate`] under
    ///   [`ExtrapolationPolicy::Reject`].
    fn value_at(&self, x: S) -> Result<Self::Value, SciOxideError> {
        Interpolant::<S, 1>::evaluate(self, [x], ExtrapolationPolicy::Reject)
    }

    /// Evaluates the interpolant at each of `xs` and returns the values in
    /// order.
    ///
    /// # Errors
    ///
    /// - The errors of [`Interpolant::evaluate_many`] under
    ///   [`ExtrapolationPolicy::Reject`]: the error at the first point that
    ///   fails, with no values, or `AllocationFailure`.
    fn values_at(&self, xs: &[S]) -> Result<Vec<Self::Value>, SciOxideError> {
        let (points, _) = xs.as_chunks::<1>();
        Interpolant::<S, 1>::evaluate_many(self, points, ExtrapolationPolicy::Reject)
    }

    /// Evaluates the first derivative at `x`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Differentiable::gradient`] under
    ///   [`ExtrapolationPolicy::Reject`].
    fn derivative_at(&self, x: S) -> Result<Self::Value, SciOxideError>
    where
        Self: Differentiable<S, 1>,
    {
        let [slope] = Differentiable::<S, 1>::gradient(self, [x], ExtrapolationPolicy::Reject)?;
        Ok(slope)
    }

    /// Integrates the interpolant over `[lower, upper]`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Integrable::integral`] under
    ///   [`ExtrapolationPolicy::Reject`], among them `InvalidInput`
    ///   (`OutOfRange`) for `lower > upper`.
    fn integrate(&self, lower: S, upper: S) -> Result<Self::Value, SciOxideError>
    where
        Self: Integrable<S, 1>,
    {
        Integrable::<S, 1>::integral(self, [lower], [upper], ExtrapolationPolicy::Reject)
    }
}

impl<S: RealScalar, I: Interpolant<S, 1>> Univariate<S> for I {}

impl<S: RealScalar, T: Interpolant<S, D>, const D: usize> Interpolant<S, D> for &T {
    type Value = T::Value;
    type Workspace = T::Workspace;
    fn domains(&self) -> [Interval<S>; D] {
        (**self).domains()
    }
    fn value_dimension(&self) -> usize {
        (**self).value_dimension()
    }
    fn zero_value(&self) -> Result<Self::Value, SciOxideError> {
        (**self).zero_value()
    }
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError> {
        (**self).workspace()
    }
    fn workspace_capacity(&self, workspace: &Self::Workspace) -> Capacity {
        (**self).workspace_capacity(workspace)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        (**self).evaluate_into(point, policy, out, workspace)
    }
    fn report(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        workspace: &mut Self::Workspace,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        (**self).report(point, policy, workspace)
    }
    fn evaluate_into_with_report<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        (**self).evaluate_into_with_report(point, policy, out, workspace)
    }
}

impl<S: RealScalar, T: Differentiable<S, D>, const D: usize> Differentiable<S, D> for &T {
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        (**self).gradient_into(point, policy, out, workspace)
    }
}

impl<S: RealScalar, T: TwiceDifferentiable<S, D>, const D: usize> TwiceDifferentiable<S, D> for &T {
    fn hessian_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        (**self).hessian_into(point, policy, out, workspace)
    }
}

impl<S: RealScalar, T: Integrable<S, D>, const D: usize> Integrable<S, D> for &T {
    fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        lower: [S; D],
        upper: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        (**self).integral_into(lower, upper, policy, out, workspace)
    }
}

/// Build `[T; N]` without heap scratch or unstable array constructors.
pub(crate) fn try_array<T, const N: usize>(
    mut make: impl FnMut() -> Result<T, SciOxideError>,
) -> Result<[T; N], SciOxideError> {
    let mut error = None;
    let items: [Option<T>; N] = core::array::from_fn(|_| {
        if error.is_some() {
            return None;
        }
        match make() {
            Ok(value) => Some(value),
            Err(failure) => {
                error = Some(failure);
                None
            }
        }
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(items.map(|item| match item {
        Some(value) => value,
        None => unreachable!("the factory succeeded for every array entry"),
    }))
}

/// Axes strictly outside their interval under [`ExtrapolationPolicy::Clamp`],
/// where derivatives vanish. Every other policy differentiates the representation.
pub(crate) fn clamped_axes<S: RealScalar, const D: usize>(
    policy: ExtrapolationPolicy<S>,
    domains: [Interval<S>; D],
    point: [S; D],
) -> [bool; D] {
    core::array::from_fn(|axis| {
        policy == ExtrapolationPolicy::Clamp && !domains[axis].contains(point[axis])
    })
}

/// Checks integration bounds (finite, then ordered, then the policy) and
/// returns the bounds the method integrates between, clamped under `Clamp`.
pub(crate) fn integral_bounds<S: RealScalar>(
    domain: Interval<S>,
    a: S,
    b: S,
    policy: ExtrapolationPolicy<S>,
) -> Result<[S; 2], SciOxideError> {
    if !a.is_finite() || !b.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "integration bounds must be finite",
        ));
    }
    if a > b {
        return Err(crate::error::out_of_range(
            scioxide_core::argument!("lower"),
            "integration lower bound must not exceed the upper bound",
        ));
    }
    Ok([
        resolve_coordinate(&domain, a, policy)?,
        resolve_coordinate(&domain, b, policy)?,
    ])
}

/// Find the interval index `i` such that `x[i] <= t < x[i+1]`.
///
/// For `t == x[n-1]` (the right endpoint), returns `n - 2` so the last
/// interval is used.
///
/// Assumes at least two sorted nodes; exterior points use the outer pieces.
pub(crate) fn find_interval<S: RealScalar>(x: &[S], t: S) -> usize {
    debug_assert!(x.len() >= 2);
    // slice's fixed-depth binary search selects bounds without an unpredictable
    // branch at each node. Clamp continuation to the two outer pieces.
    x.partition_point(|&node| node <= t)
        .saturating_sub(1)
        .min(x.len() - 2)
}

/// Applies the extrapolation policy to one scalar coordinate.
#[inline]
pub(crate) fn resolve_coordinate<S: RealScalar>(
    domain: &Interval<S>,
    x: S,
    extrapolation: ExtrapolationPolicy<S>,
) -> Result<S, SciOxideError> {
    extrapolation.validate([x])?;
    if domain.contains(x) {
        return Ok(x);
    }
    let boundary = x.max(domain.lower()).min(domain.upper());
    match extrapolation {
        ExtrapolationPolicy::Clamp => Ok(boundary),
        ExtrapolationPolicy::MethodDefined => Ok(x),
        _ => Ok(extrapolation
            .resolve_projected([x], [boundary])?
            .effective()[0]),
    }
}

/// Integrates over `[a, b]` with a constant extension outside `domain`:
/// `integral` covers the interior part and `value` gives the boundary values.
///
/// Call after [`integral_bounds`].
pub(crate) fn clamped_integral<S: RealScalar>(
    domain: Interval<S>,
    a: S,
    b: S,
    value: impl Fn(S) -> Result<S, SciOxideError>,
    integral: impl Fn(S, S) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    use crate::checked::add;
    use scioxide_poly::backend::polynomial_basis::linear_multiply_difference;
    if a == b {
        return Ok(S::ZERO);
    }
    // Split `[a, b]` into the interior part (integrated by the method's own
    // kernel) and the exterior parts, where the clamped function equals the
    // boundary value so the integral is `value * length`. `linear_multiply_difference`
    // forms that product without first materializing a possibly overflowing
    // length. The `min`/`max` on the exterior lengths handle intervals lying
    // entirely on one side of the domain.
    let left = a.max(domain.lower()).min(domain.upper());
    let right = b.max(domain.lower()).min(domain.upper());
    let mut total = if left == right {
        S::ZERO
    } else {
        integral(left, right)?
    };
    if a < domain.lower() {
        total = add(
            total,
            linear_multiply_difference(value(domain.lower())?, b.min(domain.lower()), a)?,
        )?;
    }
    if b > domain.upper() {
        total = add(
            total,
            linear_multiply_difference(value(domain.upper())?, b, a.max(domain.upper()))?,
        )?;
    }
    Ok(total)
}

/// Build an `Interval` from the first and last elements of a sorted slice.
pub(crate) fn domain_from_nodes<S: RealScalar>(x: &[S]) -> Result<Interval<S>, SciOxideError> {
    debug_assert!(!x.is_empty());
    Interval::new(x[0], x[x.len() - 1])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideErrorKind;

    fn unit() -> Interval<f64> {
        Interval::new(0.0, 1.0).unwrap()
    }

    #[test]
    fn integral_bounds_check_finite_then_order_then_policy() {
        let domain = unit();
        let kind = |a, b, policy| integral_bounds(domain, a, b, policy).unwrap_err().kind();
        // A reversed non-finite pair reports the non-finite bound first.
        assert_eq!(
            kind(f64::NAN, -1.0, ExtrapolationPolicy::Reject),
            SciOxideErrorKind::NonFinite
        );
        // A reversed pair outside the domain reports the order before the policy.
        assert_eq!(
            kind(3.0, 2.0, ExtrapolationPolicy::Reject),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            kind(0.0, 2.0, ExtrapolationPolicy::Reject),
            SciOxideErrorKind::DomainError
        );
        assert_eq!(
            integral_bounds(domain, -1.0, 2.0, ExtrapolationPolicy::Clamp).unwrap(),
            [0.0, 1.0]
        );
        assert_eq!(
            integral_bounds(domain, 0.25, 0.25, ExtrapolationPolicy::Reject).unwrap(),
            [0.25, 0.25]
        );
    }

    #[test]
    fn clamped_axes_apply_only_to_clamp_strictly_outside() {
        let domains = [unit(), unit()];
        assert_eq!(
            clamped_axes(ExtrapolationPolicy::Clamp, domains, [1.5, 1.0]),
            [true, false]
        );
        assert_eq!(
            clamped_axes(ExtrapolationPolicy::MethodDefined, domains, [1.5, -1.0]),
            [false, false]
        );
    }

    #[test]
    fn clamped_integral_adds_exact_constant_tails() {
        let total =
            clamped_integral(unit(), -1.0, 3.0, |x| Ok(2.0 * x), |a, b| Ok(b * b - a * a)).unwrap();
        // Interior 1, left tail 0 * 1, right tail 2 * 2.
        assert_eq!(total, 5.0);
        assert_eq!(
            clamped_integral(unit(), 2.0, 2.0, |_| unreachable!(), |_, _| unreachable!()).unwrap(),
            0.0
        );
    }

    #[test]
    fn try_array_propagates_the_first_error() {
        let values: [usize; 3] = {
            let mut next = 0;
            try_array(|| {
                next += 1;
                Ok(next)
            })
            .unwrap()
        };
        assert_eq!(values, [1, 2, 3]);
        let failed: Result<[usize; 2], _> = try_array(|| Err(SciOxideError::singular(None)));
        assert!(failed.is_err());
    }

    #[test]
    fn try_array_drops_partial_results_and_skips_empty_factories() {
        use core::cell::Cell;
        struct Entry<'a>(&'a Cell<usize>);
        impl Drop for Entry<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let drops = Cell::new(0);
        let mut calls = 0;
        let result: Result<[Entry<'_>; 4], _> = try_array(|| {
            calls += 1;
            if calls == 3 {
                Err(SciOxideError::singular(None))
            } else {
                Ok(Entry(&drops))
            }
        });
        assert!(result.is_err());
        assert_eq!(calls, 3);
        assert_eq!(drops.get(), 2);
        let empty: [Entry<'_>; 0] = try_array(|| {
            calls += 1;
            Ok(Entry(&drops))
        })
        .unwrap();
        assert!(empty.is_empty());
        assert_eq!(calls, 3);
    }

    #[test]
    fn reports_use_accessors_and_bounded_continuation_includes_its_limit() {
        let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 0.5 };
        let report = bounded.resolve([unit()], [1.5]).unwrap();
        assert_eq!(report.original(), [1.5]);
        assert_eq!(report.effective(), [1.5]);
        assert_eq!(report.outside_distance(), [0.5]);
        assert!(bounded.resolve([unit()], [1.5000001]).is_err());
        let invalid = ExtrapolationPolicy::BoundedContinue { max_distance: -1.0 };
        assert!(invalid.resolve([unit()], [0.5]).is_err());
        assert_eq!(
            ExtrapolationReport::interior([0.25]),
            ExtrapolationPolicy::Reject
                .resolve([unit()], [0.25])
                .unwrap()
        );
    }

    #[test]
    fn find_interval_uses_the_outer_pieces_at_the_ends() {
        assert_eq!(find_interval(&[0.0_f32, 1.0, 2.0], 1.0), 1);
        assert_eq!(find_interval(&[0.0_f64, 1.0, 2.0], 2.0), 1);
        assert_eq!(find_interval(&[0.0_f64, 1.0, 2.0], -1.0), 0);
    }
}
