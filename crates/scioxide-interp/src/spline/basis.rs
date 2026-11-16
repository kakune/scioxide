//! B-spline bases and the local evaluation kernel used by every B-spline model.
//!
//! [`BSplineBasis::span`] finds the nonempty knot span that holds a coordinate,
//! and [`local_basis`] computes the `p + 1` nonzero basis functions there, or
//! one derivative order of them, with the triangular Cox–de Boor recurrence
//! (Piegl and Tiller, algorithm A2.2) and its derivative form (A2.3). A query
//! costs `O(p² + log n)`. When a knot difference overflows, the kernel falls
//! back to [`crate::checked::interval_coordinate`] and
//! [`scioxide_poly::backend::polynomial_basis::linear_difference_ratio`], which
//! rescale to avoid overflow. [`cardinal_basis`] holds the closed forms behind
//! [`crate::cardinal_bspline`].

use crate::error::invalid;
use crate::protocol::NodeSeparation;
use crate::validate::{validate_count, validate_finite, validate_nodes};
use alloc::vec::Vec;
use core::ops::Range;

use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Argument, Interval, RealScalar, SciOxideError};

use crate::checked::{add, div, interval_coordinate, mul, sub};
use crate::protocol::{ExtrapolationPolicy, integral_bounds};
use crate::values::QueryWorkspace;
use scioxide_poly::backend::polynomial_basis::{
    linear_difference_ratio, linear_multiply_difference,
};

/// Selects the polynomial piece used for a one-sided limit at a knot.
///
/// A B-spline is a polynomial between consecutive distinct knots. At a knot of
/// multiplicity `m`, derivatives of order `p − m + 1` and higher can have
/// different one-sided limits. At a coordinate that is not a knot both sides select the
/// same piece.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BSpline, BSplineBasis, ExtrapolationPolicy, KnotSide};
/// // A kink at x = 1: slope 1 on the left piece and -1 on the right piece.
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 2.0, 2.0], 1)?;
/// let spline = BSpline::new([basis], vec![0.0, 1.0, 0.0])?;
/// let policy = ExtrapolationPolicy::Reject;
/// assert_eq!(
///     spline.derivative([1.0], [1], Some([KnotSide::Left]), policy)?,
///     1.0
/// );
/// assert_eq!(
///     spline.derivative([1.0], [1], Some([KnotSide::Right]), policy)?,
///     -1.0
/// );
/// assert!(spline.derivative([1.0], [1], None, policy).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnotSide {
    /// Limit from coordinates smaller than the query.
    Left,
    /// Limit from coordinates larger than the query.
    Right,
}

/// Stores a one-dimensional B-spline basis: a degree and a knot vector.
///
/// A basis of degree `p` (order `p + 1`) over a nondecreasing knot vector `t`
/// of length `n + p + 1` has `n` basis functions `N_{i,p}`, defined by the
/// Cox–de Boor recurrence
///
/// ```text
/// N_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0
/// N_{i,p}(x) = (x − t_i)/(t_{i+p} − t_i) N_{i,p−1}(x)
///            + (t_{i+p+1} − x)/(t_{i+p+1} − t_{i+1}) N_{i+1,p−1}(x)
/// ```
///
/// with `0/0 = 0`. The base domain is the closed interval `[t_p, t_n]`, where
/// the `N_{i,p}` sum to one. Knots may repeat up to `p + 1` times. A knot of
/// multiplicity `m` reduces the continuity there to `C^(p−m)`. A clamped
/// (open) knot vector repeats each end knot `p + 1` times. At a knot,
/// evaluation uses the piece to the right unless a [`KnotSide`] is given.
/// Outside the base domain, continuation extends the outermost nonempty piece.
///
/// A row of basis values or basis integrals costs `O(p² + log n)` operations
/// plus `O(n)` to clear the rest of the row.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BSplineBasis, ExtrapolationPolicy};
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let mut workspace = basis.workspace()?;
/// let mut row = [0.0; 2];
/// basis.evaluate_into(
///     0.25,
///     0,
///     None,
///     ExtrapolationPolicy::Reject,
///     &mut row,
///     &mut workspace,
/// )?;
/// assert_eq!(row, [0.75, 0.25]);
/// basis.integrals_into(
///     0.0,
///     1.0,
///     ExtrapolationPolicy::Reject,
///     &mut row,
///     &mut workspace,
/// )?;
/// assert_eq!(row, [0.5, 0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
///   chapters IX–X.
/// - L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., Springer, 1997,
///   chapter 2.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BSplineBasis<S: RealScalar> {
    knots: Vec<S>,
    /// Knots with the first and last knot repeated once more: the degree
    /// `p + 1` basis of the integral identity (see [`IntegralAxis`]).
    integral_knots: Vec<S>,
    degree: usize,
    controls: usize,
    domain: Interval<S>,
    /// First and last nonempty spans of the base domain.
    spans: [usize; 2],
}

impl<S: RealScalar> BSplineBasis<S> {
    /// Checks a complete knot vector for `degree` and builds the basis.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) for no knots, and (`OutOfRange`) for fewer
    ///   than `2(p + 1)` knots.
    /// - `InvalidInput` (`Inconsistent`) for decreasing knots, a knot of
    ///   multiplicity above `p + 1`, or a base domain of zero width.
    /// - `NonFinite` if a knot is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the knot storage cannot be
    ///   reserved.
    pub fn new(knots: Vec<S>, degree: usize) -> Result<Self, SciOxideError> {
        let controls = validate_knots(&knots, degree)?;
        let mut integral_knots = try_with_capacity(
            checked_add(knots.len(), 2, scioxide_core::quantity!("integral_knots"))?,
            scioxide_core::quantity!("integral_knots"),
        )?;
        integral_knots.push(knots[0]);
        integral_knots.extend_from_slice(&knots);
        integral_knots.push(knots[knots.len() - 1]);
        let domain = Interval::new(knots[degree], knots[controls])?;
        let spans = [
            boundary_span(&knots, knots[degree], true),
            boundary_span(&knots, knots[controls], false),
        ];
        Ok(Self {
            knots,
            integral_knots,
            degree,
            controls,
            domain,
            spans,
        })
    }

    /// Builds the clamped basis of degree `p` on the breakpoints
    /// `b₀ < b₁ < … < b_m`, `m ≥ 1`.
    ///
    /// The knot vector repeats each end breakpoint `p + 1` times and keeps
    /// every interior breakpoint once:
    ///
    /// ```text
    /// t = (b₀, …, b₀, b₁, …, b_{m−1}, b_m, …, b_m).
    /// ```
    ///
    /// The basis has `m + p` functions, each `C^(p−1)` at the interior
    /// breakpoints, and its base domain is `[b₀, b_m]`. At each end only one
    /// basis function is nonzero, so a spline takes the value of its first and
    /// last control there. Construction copies the breakpoints in `O(m + p)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) for no breakpoints, (`OutOfRange`) for one,
    ///   and (`Inconsistent`) for a decreasing or repeated breakpoint.
    /// - `NonFinite` if a breakpoint is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the knot storage cannot be
    ///   reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::BSplineBasis;
    /// let basis = BSplineBasis::clamped(&[0.0_f64, 1.0, 3.0], 2)?;
    /// assert_eq!(basis.knots(), [0.0, 0.0, 0.0, 1.0, 3.0, 3.0, 3.0]);
    /// assert_eq!(basis.control_count(), 4);
    /// assert!(BSplineBasis::clamped(&[0.0_f64, 0.0], 2).is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn clamped(breakpoints: &[S], degree: usize) -> Result<Self, SciOxideError> {
        Self::with_repeated_ends(
            breakpoints,
            2,
            1,
            degree,
            scioxide_core::argument!("breakpoints"),
        )
    }

    /// Builds the not-a-knot basis of odd degree `p = 2k + 1` for interpolation
    /// at the nodes `x₀ < x₁ < … < x_{n−1}`, `n ≥ p + 1`.
    ///
    /// The knot vector repeats each end node `p + 1` times and omits the `k + 1`
    /// nodes next to each end:
    ///
    /// ```text
    /// t = (x₀, …, x₀, x_{k+1}, …, x_{n−k−2}, x_{n−1}, …, x_{n−1}).
    /// ```
    ///
    /// The basis has `n` functions and satisfies the Schoenberg–Whitney
    /// condition at the nodes, so interpolation at the nodes has a unique
    /// solution. For `p = 1` the interpolant is piecewise linear; for `p = 3` it
    /// is the cubic spline with not-a-knot ends, whose third derivative is
    /// continuous at `x₁` and `x_{n−2}`. Construction copies the nodes in
    /// `O(n + p)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`, argument `degree`) for an even degree,
    ///   checked first.
    /// - `InvalidInput` (`Empty`) for no nodes, (`OutOfRange`) for fewer than
    ///   `p + 1`, and (`Inconsistent`) for a decreasing or repeated node.
    /// - `NonFinite` if a node is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the knot storage cannot be
    ///   reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::BSplineBasis;
    /// let basis = BSplineBasis::not_a_knot(&[0.0_f64, 1.0, 2.0, 3.0, 4.0], 3)?;
    /// assert_eq!(basis.knots(), [0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0]);
    /// assert_eq!(basis.control_count(), 5);
    /// assert!(BSplineBasis::not_a_knot(&[0.0_f64, 1.0, 2.0], 2).is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
    /// chapter XIII.
    pub fn not_a_knot(nodes: &[S], degree: usize) -> Result<Self, SciOxideError> {
        if degree.is_multiple_of(2) {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("degree"),
                "a not-a-knot basis needs an odd degree",
            ));
        }
        let minimum = checked_add(degree, 1, scioxide_core::quantity!("not_a_knot_nodes"))?;
        Self::with_repeated_ends(
            nodes,
            minimum,
            degree / 2 + 1,
            degree,
            scioxide_core::argument!("nodes"),
        )
    }

    /// Builds the degree-`degree` basis whose knots repeat each end of `nodes`
    /// `degree + 1` times and keep `nodes[skip..n − skip]` between them.
    ///
    /// `nodes` must be strictly increasing with at least `minimum ≥ 2 · skip`
    /// entries; `argument` names them in errors.
    pub(crate) fn with_repeated_ends(
        nodes: &[S],
        minimum: usize,
        skip: usize,
        degree: usize,
        argument: Argument,
    ) -> Result<Self, SciOxideError> {
        validate_nodes(nodes, minimum, NodeSeparation::Exact, argument)?;
        let quantity = scioxide_core::quantity!("knots");
        let interior = nodes
            .get(skip..nodes.len().saturating_sub(skip))
            .unwrap_or_default();
        let ends = checked_add(degree, 1, quantity)?;
        let mut knots = try_with_capacity(
            checked_add(checked_mul(ends, 2, quantity)?, interior.len(), quantity)?,
            quantity,
        )?;
        if let (Some(&first), Some(&last)) = (nodes.first(), nodes.last()) {
            knots.extend(core::iter::repeat_n(first, ends));
            knots.extend_from_slice(interior);
            knots.extend(core::iter::repeat_n(last, ends));
        }
        Self::new(knots, degree)
    }

    /// Fallible copy for owned results built from a borrowed basis.
    pub(crate) fn try_clone(&self) -> Result<Self, SciOxideError> {
        let mut knots = try_with_capacity(self.knots.len(), scioxide_core::quantity!("knots"))?;
        knots.extend_from_slice(&self.knots);
        let mut integral_knots = try_with_capacity(
            self.integral_knots.len(),
            scioxide_core::quantity!("integral_knots"),
        )?;
        integral_knots.extend_from_slice(&self.integral_knots);
        Ok(Self {
            knots,
            integral_knots,
            degree: self.degree,
            controls: self.controls,
            domain: self.domain,
            spans: self.spans,
        })
    }

    /// Returns the complete knot vector.
    pub fn knots(&self) -> &[S] {
        &self.knots
    }

    /// Returns the polynomial degree `p`.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the number of basis functions, which equals the number of control points.
    pub const fn control_count(&self) -> usize {
        self.controls
    }

    /// Returns the closed base domain `[t_p, t_n]`.
    pub const fn domain(&self) -> Interval<S> {
        self.domain
    }

    /// Allocates scratch for [`Self::evaluate_into`] and [`Self::integrals_into`].
    ///
    /// The size depends only on the degree, so the workspace also serves every
    /// basis of equal or lower degree.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the scratch cannot be reserved.
    pub fn workspace(&self) -> Result<SplineBasisWorkspace<S>, SciOxideError> {
        Ok(SplineBasisWorkspace(QueryWorkspace::new(axis_scratch(
            self.degree,
        )?)?))
    }

    /// Writes the values `N_{i,p}(x)`, or their `order`-th derivatives, of all
    /// basis functions to `out`.
    ///
    /// At most `p + 1` entries are nonzero. `side` selects the polynomial piece
    /// at a knot and defaults to the right piece. Under
    /// [`ExtrapolationPolicy::Clamp`], a derivative row strictly outside the
    /// domain is zero. Orders above the degree give a zero row.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` does not have [`Self::control_count`]
    ///   entries or `workspace` is too small.
    /// - The coordinate errors listed in the [crate docs](crate#errors).
    /// - `NonFinite` if an intermediate value overflows.
    pub fn evaluate_into(
        &self,
        coordinate: S,
        order: usize,
        side: Option<KnotSide>,
        policy: ExtrapolationPolicy<S>,
        out: &mut [S],
        workspace: &mut SplineBasisWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        crate::values::length(out, self.controls)?;
        let row = workspace.0.scratch(self.degree + 1)?;
        let report = policy.resolve([self.domain], [coordinate])?;
        if order > 0 && policy == ExtrapolationPolicy::Clamp && !self.domain.contains(coordinate) {
            out.fill(S::ZERO);
            return Ok(());
        }
        let first = self.local(
            report.effective()[0],
            order,
            side.unwrap_or(KnotSide::Right),
            row,
        )?;
        out.fill(S::ZERO);
        out[first..=first + self.degree].copy_from_slice(row);
        Ok(())
    }

    /// Writes the integrals `∫ N_{i,p}(x) dx` over `[lower, upper]` of all basis
    /// functions to `out`.
    ///
    /// Each integral uses the identity
    /// `∫_{−∞}^x N_{i,p} = (t_{i+p+1} − t_i)/(p + 1) · Σ_{j>i} N_{j,p+1}(x)`
    /// on the knot vector with both end knots repeated once more. Under
    /// [`ExtrapolationPolicy::Clamp`] the constant extension of each basis
    /// function is integrated. Under continuation the outermost polynomial piece
    /// is integrated.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a bound is NaN or infinite, and `InvalidInput` if
    ///   `lower > upper`.
    /// - The coordinate errors listed in the [crate docs](crate#errors).
    /// - `DimensionMismatch` if `out` does not have [`Self::control_count`]
    ///   entries or `workspace` is too small.
    /// - `NonFinite` if an integral overflows.
    ///
    /// # References
    ///
    /// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
    /// chapter X.
    pub fn integrals_into(
        &self,
        lower: S,
        upper: S,
        policy: ExtrapolationPolicy<S>,
        out: &mut [S],
        workspace: &mut SplineBasisWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        crate::values::length(out, self.controls)?;
        let scratch = workspace.0.scratch(axis_scratch(self.degree)?)?;
        let axis = IntegralAxis::prepare(self, lower, upper, policy, scratch)?;
        let scratch = &*scratch;
        let range = axis.range(self);
        // Validate every weight before the first write so failure is transactional.
        for index in range.clone() {
            axis.weight(self, scratch, index)?;
        }
        out.fill(S::ZERO);
        for index in range {
            out[index] = axis.weight(self, scratch, index)?;
        }
        Ok(())
    }

    /// Span used for an effective coordinate and side.
    pub(crate) fn span(&self, coordinate: S, side: KnotSide) -> usize {
        if coordinate <= self.domain.lower() {
            return self.spans[0];
        }
        if coordinate >= self.domain.upper() {
            return self.spans[1];
        }
        interior_span(&self.knots, coordinate, side)
    }

    /// Nonzero basis functions (or their `order`-th derivatives) at an effective
    /// coordinate into `row[..=p]`; returns the index of the first one.
    pub(crate) fn local(
        &self,
        coordinate: S,
        order: usize,
        side: KnotSide,
        row: &mut [S],
    ) -> Result<usize, SciOxideError> {
        let span = self.span(coordinate, side);
        local_basis(&self.knots, self.degree, span, coordinate, order, row)?;
        Ok(span - self.degree)
    }

    /// Number of knots equal to `coordinate`.
    pub(crate) fn multiplicity(&self, coordinate: S) -> usize {
        self.knots.partition_point(|&knot| knot <= coordinate)
            - self.knots.partition_point(|&knot| knot < coordinate)
    }
}

/// Holds scratch for [`BSplineBasis`] value rows and integrals.
///
/// Create it with [`BSplineBasis::workspace`]. Its size depends only on the
/// degree: a workspace for degree `p` serves every basis of degree at most `p`,
/// and a basis of higher degree rejects it with `DimensionMismatch`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BSplineBasis, ExtrapolationPolicy};
/// let quadratic = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2)?;
/// let linear = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let mut workspace = quadratic.workspace()?;
/// let mut row = [0.0; 2];
/// linear.evaluate_into(
///     0.5,
///     1,
///     None,
///     ExtrapolationPolicy::Reject,
///     &mut row,
///     &mut workspace,
/// )?;
/// assert_eq!(row, [-1.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SplineBasisWorkspace<S: RealScalar>(QueryWorkspace<S>);

/// Validate a knot vector and return its control count.
///
/// This is the only knot validator: every B-spline model builds a [`BSplineBasis`].
fn validate_knots<S: RealScalar>(knots: &[S], degree: usize) -> Result<usize, SciOxideError> {
    let order = checked_add(degree, 1, scioxide_core::quantity!("b_spline_order"))?;
    let minimum = checked_mul(order, 2, scioxide_core::quantity!("b_spline_minimum_knots"))?;
    let argument = scioxide_core::argument!("knots");
    validate_count(knots.len(), minimum, argument)?;
    validate_finite(knots)?;
    if knots.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(invalid(argument, "B-spline knots must be nondecreasing"));
    }
    let mut multiplicity = 0usize;
    for (index, &knot) in knots.iter().enumerate() {
        multiplicity = if index > 0 && knots[index - 1] == knot {
            multiplicity + 1
        } else {
            1
        };
        if multiplicity > order {
            return Err(invalid(
                argument,
                "knot multiplicity exceeds degree plus one",
            ));
        }
    }
    let controls = knots.len() - order;
    if knots[degree] >= knots[controls] {
        return Err(invalid(
            argument,
            "B-spline base domain must have positive width",
        ));
    }
    Ok(controls)
}

/// Scratch scalars per axis: integral tails at both bounds (`2(p + 2)`) and
/// clamped boundary rows (`2(p + 1)`); a value row uses the first `p + 1`.
pub(crate) fn axis_scratch(degree: usize) -> Result<usize, SciOxideError> {
    checked_add(
        checked_mul(degree, 4, scioxide_core::quantity!("basis_workspace"))?,
        6,
        scioxide_core::quantity!("basis_workspace"),
    )
}

/// First (`first = true`) or last nonempty span touching `boundary`.
fn boundary_span<S: RealScalar>(knots: &[S], boundary: S, first: bool) -> usize {
    if first {
        knots.partition_point(|&knot| knot <= boundary) - 1
    } else {
        knots.partition_point(|&knot| knot < boundary) - 1
    }
}

/// Span `k` with `t_k <= t < t_{k+1}` (right) or `t_k < t <= t_{k+1}` (left) for
/// a coordinate strictly inside the base domain.
fn interior_span<S: RealScalar>(knots: &[S], coordinate: S, side: KnotSide) -> usize {
    match side {
        KnotSide::Left => knots.partition_point(|&knot| knot < coordinate) - 1,
        KnotSide::Right => knots.partition_point(|&knot| knot <= coordinate) - 1,
    }
}

/// `(t - from) / (to - from)`, rescaled when a raw difference overflows.
#[inline]
fn coordinate<S: RealScalar>(t: S, from: S, to: S) -> Result<S, SciOxideError> {
    let numerator = t - from;
    let denominator = to - from;
    if numerator.is_finite() && denominator.is_finite() {
        div(numerator, denominator)
    } else {
        interval_coordinate(t, from, to)
    }
}

/// Uniform-knot specialization of the local kernel for its centered basis
/// function, degrees zero through three. The caller validates degree and the
/// half-open support. These are the expanded A2.2 triangles on unit knots.
#[inline]
pub(crate) fn cardinal_basis<S: RealScalar>(degree: usize, coordinate: S) -> S {
    let x = coordinate.abs();
    let one_half = S::ONE + S::HALF;
    let six = S::TWO + S::TWO + S::TWO;
    match degree {
        0 => S::ONE,
        1 => S::ONE - x,
        2 if x < S::HALF => (S::ONE - S::HALF * S::HALF) - x * x,
        2 => {
            let tail = one_half - x;
            S::HALF * tail * tail
        }
        _ if x < S::ONE => S::TWO / (S::TWO + S::ONE) + (S::HALF * x - S::ONE) * x * x,
        _ => {
            let tail = S::TWO - x;
            tail * tail * tail / six
        }
    }
}

/// The local-support kernel.
///
/// Writes the `order`-th derivatives of the nonzero basis functions
/// `N_{span-p}, ..., N_{span}` of degree `p = degree` at `t` into `row[..=p]`.
/// `span` must be a nonempty span (`t_span < t_{span+1}`), so every knot
/// difference used below is positive. Orders above the degree give zeros.
///
/// Values of degree `d` follow from degree `d - 1`, and the last `order` levels
/// differentiate instead. Each level updates one row in place; the native
/// triangle shares each quotient between the two child entries:
///
/// ```text
/// N_{i,d}      = α_{i,d} N_{i,d-1} + β_{i+1,d} N_{i+1,d-1}
/// α_{i,d}      = (t - t_i) / (t_{i+d} - t_i),   β_{m,d} = (t_{m+d} - t) / (t_{m+d} - t_m)
/// N^{(k)}_{i,d} = d (N^{(k-1)}_{i,d-1} / (t_{i+d} - t_i) - N^{(k-1)}_{i+1,d-1} / (t_{i+d+1} - t_{i+1}))
/// ```
pub(crate) fn local_basis<S: RealScalar>(
    knots: &[S],
    degree: usize,
    span: usize,
    t: S,
    order: usize,
    row: &mut [S],
) -> Result<(), SciOxideError> {
    // The common degrees are constant in these instantiations, allowing the
    // compiler to unroll the triangle without changing the general algorithm.
    let native = match degree {
        0 => native_basis(knots, 0, span, t, order, row),
        1 => native_basis(knots, 1, span, t, order, row),
        2 => native_basis(knots, 2, span, t, order, row),
        3 => native_basis(knots, 3, span, t, order, row),
        _ => native_basis(knots, degree, span, t, order, row),
    };
    if native {
        return Ok(());
    }
    range_basis(knots, degree, span, t, order, row)
}

/// A2.2 with one quotient for both children of each triangle entry. The last
/// `order` levels apply the derivative recurrence instead. Restart the checked
/// recurrence if a gap, quotient or intermediate loses the native range.
#[inline(always)]
fn native_basis<S: RealScalar>(
    knots: &[S],
    degree: usize,
    span: usize,
    t: S,
    order: usize,
    row: &mut [S],
) -> bool {
    let row = &mut row[..=degree];
    row.fill(S::ZERO);
    if order > degree {
        return true;
    }
    row[0] = S::ONE;
    let base = degree - order;
    let mut scale = S::ZERO;
    for d in 1..=degree {
        let mut saved = S::ZERO;
        let next_scale = scale + S::ONE;
        if next_scale == scale {
            return false;
        }
        scale = next_scale;
        for j in 0..d {
            let lower = knots[span + 1 - d + j];
            let upper = knots[span + 1 + j];
            let gap = upper - lower;
            let quotient = row[j] / gap;
            if !gap.is_finite()
                || !quotient.is_finite()
                || (quotient.abs() < S::MIN_POSITIVE && row[j] != S::ZERO)
            {
                return false;
            }
            if d <= base {
                let left = t - lower;
                let right = upper - t;
                if !left.is_finite() || !right.is_finite() {
                    return false;
                }
                row[j] = saved + right * quotient;
                saved = left * quotient;
            } else {
                row[j] = (saved - quotient) * scale;
                saved = quotient;
            }
            if !row[j].is_finite() || !saved.is_finite() {
                return false;
            }
        }
        row[d] = if d <= base { saved } else { saved * scale };
        if !row[d].is_finite() {
            return false;
        }
    }
    true
}

/// Cox–de Boor recurrence with rescaled differences, used only when the native
/// triangle overflows on a knot gap or a divided intermediate.
fn range_basis<S: RealScalar>(
    knots: &[S],
    degree: usize,
    span: usize,
    t: S,
    order: usize,
    row: &mut [S],
) -> Result<(), SciOxideError> {
    let row = &mut row[..=degree];
    row.fill(S::ZERO);
    if order > degree {
        return Ok(());
    }
    row[0] = S::ONE;
    let base = degree - order;
    for d in 1..=base {
        for j in (0..=d).rev() {
            let i = span + j - d;
            let mut value = S::ZERO;
            if j > 0 {
                value = mul(coordinate(t, knots[i], knots[i + d])?, row[j - 1])?;
            }
            if j < d {
                value = add(
                    value,
                    mul(coordinate(t, knots[i + d + 1], knots[i + 1])?, row[j])?,
                )?;
            }
            row[j] = value;
        }
    }
    for d in base + 1..=degree {
        let scale = S::try_from_usize_exact(d)?;
        for j in (0..=d).rev() {
            let i = span + j - d;
            let mut slope = S::ZERO;
            if j > 0 {
                slope = linear_difference_ratio(row[j - 1], S::ZERO, knots[i + d], knots[i])?;
            }
            if j < d {
                slope = sub(
                    slope,
                    linear_difference_ratio(row[j], S::ZERO, knots[i + d + 1], knots[i + 1])?,
                )?;
            }
            row[j] = mul(scale, slope)?;
        }
    }
    Ok(())
}

/// One axis of a definite integral of basis functions, computed locally.
///
/// Uses `∫_{-∞}^{x} N_{i,p} = (t_{i+p+1} - t_i)/(p + 1) · T_i(x)` with
/// `T_i(x) = Σ_{j>i} N_{j,p+1}(x)` on the knot vector extended by one repeated
/// end knot. Only the `p + 2` nonzero degree-`p + 1` functions at each bound are
/// stored, as suffix sums; `T_i` is one below the active block and zero above it.
/// The width factor is applied with [`linear_multiply_difference`] so ±MAX knots stay finite.
/// Clamp adds `length · N_i(boundary)` for each exterior part.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IntegralAxis<S> {
    /// First active index of the suffix-sum blocks at the effective bounds,
    /// present when the interior part has positive length.
    interior: Option<[usize; 2]>,
    /// `(upper end, lower end)` of the exterior part below the domain.
    below: Option<[S; 2]>,
    /// `(upper end, lower end)` of the exterior part above the domain.
    above: Option<[S; 2]>,
    order: S,
}

impl<S: RealScalar> IntegralAxis<S> {
    /// Placeholder for an axis that has not been prepared.
    pub(crate) const EMPTY: Self = Self {
        interior: None,
        below: None,
        above: None,
        order: S::ONE,
    };

    /// Validate bounds and compute the local blocks into `scratch` (at least
    /// [`axis_scratch`] entries).
    pub(crate) fn prepare(
        basis: &BSplineBasis<S>,
        lower: S,
        upper: S,
        policy: ExtrapolationPolicy<S>,
        scratch: &mut [S],
    ) -> Result<Self, SciOxideError> {
        let [left, right] = integral_bounds(basis.domain, lower, upper, policy)?;
        let p = basis.degree;
        let q = p + 1;
        let (tails, rows) = scratch.split_at_mut(2 * (q + 1));
        let (lower_tail, upper_tail) = tails.split_at_mut(q + 1);
        let interior = if left < right {
            let mut first = [0; 2];
            for ((slot, bound), tail) in first
                .iter_mut()
                .zip([left, right])
                .zip([&mut *lower_tail, &mut *upper_tail])
            {
                // Repeating the outer knots shifts every original span by one
                // and leaves the base domain and its side semantics unchanged.
                let span = basis.span(bound, KnotSide::Right) + 1;
                local_basis(&basis.integral_knots, q, span, bound, 0, tail)?;
                // Suffix sums: tail[k] = Σ_{j > k} N_j.
                let mut running = S::ZERO;
                for value in tail.iter_mut().rev() {
                    let current = *value;
                    *value = running;
                    running = add(running, current)?;
                }
                *slot = span - q;
            }
            Some(first)
        } else {
            None
        };
        let (below_row, above_row) = rows.split_at_mut(q);
        let clamp = policy == ExtrapolationPolicy::Clamp;
        let domain = basis.domain;
        let below = if clamp && lower < domain.lower() {
            local_basis(
                &basis.knots,
                p,
                basis.spans[0],
                domain.lower(),
                0,
                below_row,
            )?;
            Some([upper.min(domain.lower()), lower])
        } else {
            None
        };
        let above = if clamp && upper > domain.upper() {
            local_basis(
                &basis.knots,
                p,
                basis.spans[1],
                domain.upper(),
                0,
                above_row,
            )?;
            Some([upper, lower.max(domain.upper())])
        } else {
            None
        };
        Ok(Self {
            interior,
            below,
            above,
            order: S::try_from_usize_exact(q)?,
        })
    }

    /// Indices with possibly nonzero weights.
    pub(crate) fn range(&self, basis: &BSplineBasis<S>) -> Range<usize> {
        let p = basis.degree;
        let mut start = usize::MAX;
        let mut end = 0;
        if let Some([lower_first, upper_first]) = self.interior {
            start = start.min(lower_first);
            end = end.max(upper_first + p + 1);
        }
        if self.below.is_some() {
            start = start.min(basis.spans[0] - p);
            end = end.max(basis.spans[0] + 1);
        }
        if self.above.is_some() {
            start = start.min(basis.spans[1] - p);
            end = end.max(basis.spans[1] + 1);
        }
        let end = end.min(basis.controls);
        start.min(end)..end
    }

    /// Integral of basis function `index`; `scratch` is the prepared block.
    pub(crate) fn weight(
        &self,
        basis: &BSplineBasis<S>,
        scratch: &[S],
        index: usize,
    ) -> Result<S, SciOxideError> {
        let p = basis.degree;
        let q = p + 1;
        let tail = |block: &[S], first: usize| {
            if index < first {
                S::ONE
            } else {
                block.get(index - first).copied().unwrap_or(S::ZERO)
            }
        };
        let mut total = S::ZERO;
        if let Some([lower_first, upper_first]) = self.interior {
            let difference = sub(
                tail(&scratch[q + 1..2 * (q + 1)], upper_first),
                tail(&scratch[..=q], lower_first),
            )?;
            if difference != S::ZERO {
                total = linear_multiply_difference(
                    div(difference, self.order)?,
                    basis.knots[index + q],
                    basis.knots[index],
                )?;
            }
        }
        let rows = &scratch[2 * (q + 1)..];
        for (part, span, row) in [
            (self.below, basis.spans[0], &rows[..q]),
            (self.above, basis.spans[1], &rows[q..2 * q]),
        ] {
            if let Some([end, start]) = part
                && index + p >= span
                && index <= span
            {
                total = add(
                    total,
                    linear_multiply_difference(row[index + p - span], end, start)?,
                )?;
            }
        }
        crate::checked::output(total)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Independent full-vector Cox–de Boor recursion.
    fn reference(knots: &[f64], i: usize, degree: usize, x: f64) -> f64 {
        if degree == 0 {
            return f64::from(knots[i] <= x && x < knots[i + 1]);
        }
        let left = knots[i + degree] - knots[i];
        let right = knots[i + degree + 1] - knots[i + 1];
        (if left == 0. {
            0.
        } else {
            (x - knots[i]) / left * reference(knots, i, degree - 1, x)
        }) + (if right == 0. {
            0.
        } else {
            (knots[i + degree + 1] - x) / right * reference(knots, i + 1, degree - 1, x)
        })
    }

    #[test]
    fn validator_reports_one_vocabulary() {
        let kind = |knots: Vec<f64>, degree| BSplineBasis::new(knots, degree).unwrap_err().kind();
        use scioxide_core::SciOxideErrorKind::{InvalidInput, NonFinite, SizeOverflow};
        assert_eq!(kind(vec![0., 1.], 1), InvalidInput);
        assert_eq!(kind(vec![0., 0., 2., 1.], 1), InvalidInput);
        assert_eq!(kind(vec![0., 0., 0., 1., 1.], 1), InvalidInput);
        assert_eq!(kind(vec![0., 0., 0., 0.], 1), InvalidInput);
        assert_eq!(kind(vec![0., f64::NAN, 1., 1.], 1), NonFinite);
        assert_eq!(kind(vec![0., 1.], usize::MAX), SizeOverflow);
        let basis = BSplineBasis::new(vec![0., 0., 1., 2., 2.], 1).unwrap();
        assert_eq!(basis.control_count(), 3);
        assert_eq!(basis.domain(), Interval::new(0., 2.).unwrap());
    }

    #[test]
    fn local_rows_match_full_recursion() {
        let knots = [-2., -1., 0., 0.5, 0.5, 1., 2., 3., 4.];
        let degree = 2;
        let basis = BSplineBasis::new(knots.to_vec(), degree).unwrap();
        let mut row = [0.; 3];
        for k in 0..=40 {
            let x = f64::from(k) / 20.;
            let first = basis.local(x, 0, KnotSide::Right, &mut row).unwrap();
            for i in 0..basis.control_count() {
                let expected = if x == 2. {
                    // The right endpoint uses the last nonempty piece.
                    reference(&knots, i, degree, x - 1e-300)
                } else {
                    reference(&knots, i, degree, x)
                };
                let actual = if (first..=first + degree).contains(&i) {
                    row[i - first]
                } else {
                    0.
                };
                assert!((actual - expected).abs() < 1e-14, "x={x} i={i}");
            }
        }
    }

    #[test]
    fn derivative_rows_are_differences_of_lower_degree() {
        let basis = BSplineBasis::new(vec![0., 0., 0., 1., 3., 3., 3.], 2).unwrap();
        let mut row = [0.; 3];
        let first = basis.local(0.5, 1, KnotSide::Right, &mut row).unwrap();
        assert_eq!(first, 0);
        // Closed form on [0, 1] for knots 0,0,0,1,3,3,3.
        let x: f64 = 0.5;
        let expected = [-2. * (1. - x), 2. * (1. - x) - 2. * x / 3., 2. * x / 3.];
        for (a, b) in row.iter().zip(expected) {
            assert!((a - b).abs() < 1e-14);
        }
        basis.local(0.5, 3, KnotSide::Right, &mut row).unwrap();
        assert_eq!(row, [0.; 3]);
    }

    #[test]
    fn native_rows_match_checked_rows_at_all_orders() {
        for degree in 0..=7 {
            let knots: Vec<f64> = (0..=2 * degree + 3).map(|k| k as f64 * 0.375).collect();
            let span = degree + 1;
            for fraction in [-1.5, 0.0, 0.25, 0.5, 1.0, 2.5] {
                let x = knots[span] + fraction * 0.375;
                for order in 0..=degree + 1 {
                    let mut native = vec![0.0; degree + 1];
                    let mut checked = native.clone();
                    local_basis(&knots, degree, span, x, order, &mut native).unwrap();
                    range_basis(&knots, degree, span, x, order, &mut checked).unwrap();
                    for (&actual, &expected) in native.iter().zip(&checked) {
                        assert!(
                            (actual - expected).abs()
                                <= 256.0 * f64::EPSILON * expected.abs().max(1.0),
                            "degree={degree} order={order} x={x}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn divided_intermediates_fall_back_without_losing_basis_values() {
        for width in [f64::from_bits(16), f64::MAX / 2.0] {
            let knots = [0.0, 0.0, width, width];
            let mut row = [0.0; 2];
            local_basis(&knots, 1, 1, width * 0.25, 0, &mut row).unwrap();
            assert_eq!(row, [0.75, 0.25]);
        }
    }

    #[test]
    fn uniform_specialization_matches_the_general_kernel() {
        for degree in 0..=3 {
            let half = (degree + 1) as f64 * 0.5;
            let knots: Vec<f64> = (0..3 * degree + 2)
                .map(|k| k as f64 - degree as f64 - half)
                .collect();
            let basis = BSplineBasis::new(knots, degree).unwrap();
            for k in 0..128 {
                let x = -half + 2.0 * half * f64::from(k) / 128.0;
                let mut row = [0.0; 4];
                let first = basis.local(x, 0, KnotSide::Right, &mut row).unwrap();
                let expected = row[degree - first];
                assert!((cardinal_basis(degree, x) - expected).abs() < 8.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn sides_and_boundaries_select_nonempty_spans() {
        let basis = BSplineBasis::new(vec![0., 0., 1., 1., 2., 3., 3.], 1).unwrap();
        assert_eq!(basis.span(1., KnotSide::Right), 3);
        assert_eq!(basis.span(1., KnotSide::Left), 1);
        assert_eq!(basis.span(-5., KnotSide::Left), 1);
        assert_eq!(basis.span(3., KnotSide::Right), 4);
        assert_eq!(basis.span(0., KnotSide::Left), 1);
        assert_eq!(basis.multiplicity(1.), 2);
        assert_eq!(basis.multiplicity(1.5), 0);
    }

    #[test]
    fn extreme_knots_stay_finite() {
        let basis = BSplineBasis::new(vec![-f64::MAX, -f64::MAX, f64::MAX, f64::MAX], 1).unwrap();
        let mut row = [0.; 2];
        basis.local(0., 0, KnotSide::Right, &mut row).unwrap();
        assert_eq!(row, [0.5, 0.5]);
        basis.local(0., 1, KnotSide::Right, &mut row).unwrap();
        assert_eq!(row[0], -row[1]);
        assert!(row[1] > 0.);
        let mut workspace = basis.workspace().unwrap();
        let mut integrals = [0.; 2];
        basis
            .integrals_into(
                -f64::MAX,
                f64::MAX,
                ExtrapolationPolicy::Reject,
                &mut integrals,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(integrals, [f64::MAX, f64::MAX]);
    }

    #[test]
    fn clamped_integrals_add_constant_extensions() {
        let basis = BSplineBasis::new(vec![0., 0., 1., 1.], 1).unwrap();
        let mut workspace = basis.workspace().unwrap();
        let mut out = [9.; 2];
        basis
            .integrals_into(
                -1.,
                3.,
                ExtrapolationPolicy::Clamp,
                &mut out,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(out, [1.5, 2.5]);
        basis
            .integrals_into(
                -3.,
                -1.,
                ExtrapolationPolicy::Clamp,
                &mut out,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(out, [2., 0.]);
        assert!(
            basis
                .integrals_into(1., 0., ExtrapolationPolicy::Clamp, &mut out, &mut workspace)
                .is_err()
        );
        assert_eq!(out, [2., 0.]);
        let mut short = [0.; 1];
        assert_eq!(
            basis
                .integrals_into(
                    0.,
                    1.,
                    ExtrapolationPolicy::Reject,
                    &mut short,
                    &mut workspace
                )
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }
}
