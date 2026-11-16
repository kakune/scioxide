//! Interpolating cubic splines with natural, clamped, not-a-knot and periodic
//! end conditions.

#![allow(clippy::many_single_char_names)]

use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{AffineValue, RealScalar, SciOxideError};
use scioxide_linalg::{
    CyclicTridiagonalPlan, LinalgExecutionOptions, StructuredOperation, TridiagonalMatrixView,
};

use super::PiecewiseCubic;
use crate::checked::{add, div, intermediate, mul, sub};

use crate::protocol::SampleOptions;
use crate::spline::PreparedSplineAntiderivative;
use crate::validate::validate_nodes;
use crate::values::{flatten, gather, scatter};
use scioxide_linalg::backend::tridiagonal::{CyclicTridiagonalFactor, TridiagonalFactor};
use scioxide_poly::backend::polynomial_basis::{
    linear_difference_ratio, normalized_linear_difference,
};

/// Selects the end conditions of a [`CubicSpline`].
///
/// An interpolating cubic spline on `n` nodes has `n − 2` interior continuity
/// conditions on its second derivative and needs two more. Clamped slopes have
/// the spline's value type, so a vector-valued spline takes one slope vector per
/// end:
///
/// ```
/// use scioxide_interp::{CubicSpline, Differentiable, ExtrapolationPolicy, SplineBoundary};
/// let ends = SplineBoundary::Clamped([1.0_f64, 0.0], [1.0, 0.0]);
/// let spline = CubicSpline::with_boundary(
///     vec![0.0, 1.0, 2.0],
///     vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
///     ends,
/// )?;
/// let [slope] = spline.gradient([0.0], ExtrapolationPolicy::Reject)?;
/// assert_eq!(slope, [1.0, 0.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplineBoundary<V> {
    /// Sets `s″(x₀) = s″(xₙ₋₁) = 0`.
    Natural,
    /// Sets `s′(x₀)` and `s′(xₙ₋₁)` to the given slopes, in units of value per
    /// unit of `x`.
    Clamped(V, V),
    /// Makes `s‴` continuous at `x₁` and `xₙ₋₂`, so the first two and the last
    /// two pieces are each one cubic. Two nodes give a straight line and three
    /// nodes give the interpolating parabola.
    NotAKnot,
    /// Makes `s`, `s′` and `s″` equal at both ends. Each value component needs
    /// `|y₀ − yₙ₋₁| ≤ √ε · max(|y₀|, |yₙ₋₁|, 1)`, where `ε` is the machine
    /// epsilon. The last value is then replaced by the first. Two nodes give a
    /// constant.
    Periodic,
}

/// Interpolates samples by a C² cubic spline.
///
/// The spline `s` is a cubic on each interval `[xᵢ, xᵢ₊₁]`, passes through every
/// sample, and has continuous first and second derivatives. The end conditions
/// come from [`SplineBoundary`].
///
/// Construction solves for the node second derivatives. The system is
/// tridiagonal, or cyclic tridiagonal for periodic ends, and depends only on
/// the nodes. It is factored once in O(n) and solved once for each real
/// component of the values. The nodes are first rescaled by half the width of
/// the domain, so the system does not depend on the magnitude of `x`. Value
/// and derivative queries cost O(log n), and an integral costs O(log n + k)
/// for `k` crossed intervals. For many integrals of one spline,
/// [`Self::prepare_antiderivative`] precomputes the running integral so each
/// integral costs O(log n).
///
/// # Errors
///
/// [`Self::with_options`] lists the construction errors. Queries fail as
/// described in the [crate docs](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_interp::{CubicSpline, ExtrapolationPolicy, Interpolant, SplineBoundary};
/// let x = vec![0.0_f64, 1.0, 2.0, 3.0];
/// let y = vec![0.0, 2.0, 4.0, 6.0];
/// let spline = CubicSpline::with_boundary(x, y, SplineBoundary::Clamped(2.0, 2.0))?;
/// assert!((spline.evaluate([1.5], ExtrapolationPolicy::Reject)? - 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
/// chapter IV.
#[derive(Debug, Clone)]
#[must_use]
pub struct CubicSpline<S: RealScalar, V: AffineValue<Scale = S> = S> {
    inner: PiecewiseCubic<S, V>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> CubicSpline<S, V> {
    /// Builds the not-a-knot spline through `(x[i], y[i])`.
    ///
    /// This is [`Self::with_options`] with [`SplineBoundary::NotAKnot`] and
    /// [`SampleOptions::new`], so the nodes must be strictly increasing. Two
    /// nodes give a straight line and three the interpolating parabola.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes or the nodes are not
    ///   strictly increasing.
    /// - `DimensionMismatch` if `x` and `y` differ in length or the values differ
    ///   in shape.
    /// - `NonFinite` if a node or value is NaN or infinite, or a slope overflows.
    /// - `Singular` if the second-derivative system is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{CubicSpline, Univariate};
    /// // A not-a-knot spline reproduces the cubic f(x) = x³.
    /// let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    /// let y = x.iter().map(|&t| t * t * t).collect();
    /// let spline = CubicSpline::new(x, y)?;
    /// assert!((spline.value_at(2.5)? - 15.625).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_boundary(x, y, SplineBoundary::NotAKnot)
    }

    /// Builds the spline through `(x[i], y[i])` with the end conditions
    /// `boundary`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// must be strictly increasing. Every end condition accepts two or more
    /// nodes.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes, the nodes are not
    ///   strictly increasing, or a periodic spline has a component whose end
    ///   values differ by more than the tolerance in
    ///   [`SplineBoundary::Periodic`].
    /// - `DimensionMismatch` if `x` and `y` differ in length, or the values or
    ///   clamped slopes differ in shape.
    /// - `NonFinite` if a node, value or clamped slope is NaN or infinite, or a
    ///   slope overflows.
    /// - `Singular` if the second-derivative system is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{CubicSpline, SplineBoundary, Univariate};
    /// let (x, y) = (vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 0.0]);
    /// let spline = CubicSpline::with_boundary(x, y, SplineBoundary::Natural)?;
    /// // Natural ends give s″(0) = 0 and s″(1) = −3, so s′(0) = 1 + 3/6.
    /// assert!((spline.derivative_at(0.0)? - 1.5).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_boundary(
        x: Vec<S>,
        y: Vec<V>,
        boundary: SplineBoundary<V>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, boundary, SampleOptions::new())
    }

    /// Builds the spline through `(x[i], y[i])` with the end conditions
    /// `boundary`, validating the nodes under `options`.
    ///
    /// Every end condition accepts two or more nodes. The nodes must be strictly
    /// increasing and distinct under [`SampleOptions::node_separation`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes, the nodes are not
    ///   strictly increasing and distinct under the separation of `options`, or
    ///   a periodic spline has a component whose end values differ by more than
    ///   the tolerance in [`SplineBoundary::Periodic`].
    /// - `DimensionMismatch` if `x` and `y` differ in length, or the values or
    ///   clamped slopes differ in shape.
    /// - `NonFinite` if a node, value or clamped slope is NaN or infinite, or a
    ///   slope overflows.
    /// - `Singular` if the second-derivative system is singular.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        boundary: SplineBoundary<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        validate_nodes(
            &x,
            2,
            options.node_separation(),
            scioxide_core::argument!("x"),
        )?;
        let n = x.len();
        let mut flat = flatten(y, n)?;
        let dimension = flat.dimension;
        if let SplineBoundary::Clamped(first, last) = &boundary {
            for slope in [first, last] {
                scioxide_core::check_value_shape(slope, dimension)?;
                scioxide_core::validate_value(slope)?;
            }
        }
        if matches!(boundary, SplineBoundary::Periodic) {
            for component in 0..dimension {
                let first = flat.components[component];
                let last = flat.components[(n - 1) * dimension + component];
                let scale = first.abs().max(last.abs()).max(S::ONE);
                if normalized_linear_difference(first, last, scale)?.abs() > S::SQRT_EPSILON {
                    return Err(crate::error::out_of_range(
                        scioxide_core::argument!("y"),
                        "Periodic boundary requires matching endpoint components",
                    ));
                }
                flat.components[(n - 1) * dimension + component] = first;
            }
        }
        // Work in u = (x - center)/scale; only differences of u are formed.
        let (_, scale) = crate::checked::interval_normalization(x[0], x[n - 1])?;
        // A representable inverse power of two is exact: multiplication has
        // precisely the same rounding and gradual underflow as division by
        // scale. Other scales retain the original division path.
        let inverse = match scale.decompose() {
            scioxide_core::FloatDecomposition::Finite {
                significand,
                negative: false,
                ..
            } if significand.is_power_of_two() => {
                let inverse = S::ONE / scale;
                (inverse.is_finite() && inverse > S::ZERO).then_some(inverse)
            }
            _ => None,
        };
        let mut h = try_with_capacity(n - 1, scioxide_core::quantity!("normalized_widths"))?;
        if let Some(inverse) = inverse.filter(|_| (x[n - 1] - x[0]).is_finite()) {
            h.extend(x.windows(2).map(|pair| (pair[1] - pair[0]) * inverse));
        } else {
            for pair in x.windows(2) {
                h.push(normalized_linear_difference(pair[1], pair[0], scale)?);
            }
        }
        let system = System::new(&h, &boundary)?;
        let native_geometry = crate::checked::ordinary(scale)
            && h.iter()
                .all(|&v| v > S::ZERO && crate::checked::ordinary(v));
        let mut column = try_filled(
            if dimension > 1 { n } else { 0 },
            S::ZERO,
            scioxide_core::quantity!("spline_column"),
        )?;
        let mut sigma = try_filled(
            n,
            S::ZERO,
            scioxide_core::quantity!("spline_second_derivatives"),
        )?;
        let mut slopes = try_filled(
            if dimension > 1 {
                flat.components.len()
            } else {
                0
            },
            S::ZERO,
            scioxide_core::quantity!("node_slopes"),
        )?;
        for c in 0..dimension {
            let y = if dimension == 1 {
                &flat.components[..]
            } else {
                gather(&flat.components, dimension, c, &mut column);
                &column
            };
            let ends = match &boundary {
                SplineBoundary::Clamped(first, last) => Some([
                    mul(first.component(c)?, scale)?,
                    mul(last.component(c)?, scale)?,
                ]),
                _ => None,
            };
            let native = native_geometry
                && y.iter().all(|&v| crate::checked::ordinary(v))
                && ends.iter().flatten().all(|&v| crate::checked::ordinary(v));
            system.solve_into(&h, y, ends, &mut sigma, native)?;
            node_slopes(&h, y, &mut sigma, scale, native, inverse)?;
            if dimension > 1 {
                scatter(&sigma, dimension, c, &mut slopes);
            }
        }
        let slopes = if dimension == 1 { sigma } else { slopes };
        drop(system);
        drop(h);
        drop(column);
        Ok(Self {
            inner: PiecewiseCubic::new(x, flat.components, slopes, dimension, flat.template)?,
        })
    }

    /// Returns the first derivatives at the nodes, node-major: the slopes of all
    /// components at `x[0]`, then at `x[1]`, and so on.
    pub fn slopes(&self) -> &[S] {
        self.inner.slopes()
    }

    /// Precomputes the antiderivative of the spline for repeated integrals.
    ///
    /// Stores the integral of every piece and their running sums, in O(n · m)
    /// time and memory for `m` value components. Each integral then costs
    /// O(log n) per component.
    ///
    /// ```
    /// use scioxide_interp::{CubicSpline, ExtrapolationPolicy, SplineBoundary};
    /// let spline = CubicSpline::with_boundary(
    ///     vec![0.0_f64, 1.0, 2.0],
    ///     vec![0.0, 1.0, 2.0],
    ///     SplineBoundary::Natural,
    /// )?;
    /// let prepared = spline.prepare_antiderivative()?;
    /// assert!((prepared.integral(0.0, 2.0, ExtrapolationPolicy::Reject)? - 2.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a piece integral or running sum overflows.
    pub fn prepare_antiderivative(
        &self,
    ) -> Result<PreparedSplineAntiderivative<S, V::Owned>, SciOxideError> {
        self.inner.prepare_antiderivative()
    }
}

piecewise_protocol!(CubicSpline);

/// Replaces normalized second derivatives by first derivatives in units of `x`.
/// Saves the final endpoint before the forward sweep overwrites its neighbor.
fn node_slopes<S: RealScalar>(
    h: &[S],
    y: &[S],
    sigma: &mut [S],
    scale: S,
    native: bool,
    inverse: Option<S>,
) -> Result<(), SciOxideError> {
    let n = y.len();
    let three = S::TWO + S::ONE;
    let physical = |value| match inverse {
        Some(inverse) => intermediate(value * inverse),
        None => div(value, scale),
    };
    let last = n - 2;
    if native {
        // Admitted data make secants finite. Overflow in subsequent arithmetic
        // propagates to the final physical slope check; divisors stay finite.
        let dy = (y[last + 1] - y[last]) / h[last];
        let curvature =
            sigma[last] / S::TWO + (sigma[last + 1] / S::TWO + sigma[last + 1] / S::TWO);
        let endpoint = physical(dy + h[last] * curvature / three)?;
        let mut current = sigma[0] / S::TWO;
        for i in 0..n - 1 {
            let next = sigma[i + 1] / S::TWO;
            let dy = (y[i + 1] - y[i]) / h[i];
            let curvature = next + (current + current);
            sigma[i] = physical(dy - h[i] * curvature / three)?;
            current = next;
        }
        sigma[n - 1] = endpoint;
        return Ok(());
    }
    let dy = secant(y[last + 1], y[last], h[last])?;
    let right = div(sigma[last + 1], S::TWO)?;
    let curvature = add(div(sigma[last], S::TWO)?, add(right, right)?)?;
    let endpoint = physical(add(dy, div(mul(h[last], curvature)?, three)?)?)?;
    let mut current = div(sigma[0], S::TWO)?;
    for i in 0..n - 1 {
        let next = div(sigma[i + 1], S::TWO)?;
        let dy = secant(y[i + 1], y[i], h[i])?;
        let curvature = add(next, add(current, current)?)?;
        sigma[i] = physical(sub(dy, div(mul(h[i], curvature)?, three)?)?)?;
        current = next;
    }
    sigma[n - 1] = endpoint;
    Ok(())
}

fn secant<S: RealScalar>(right: S, left: S, width: S) -> Result<S, SciOxideError> {
    linear_difference_ratio(right, left, width, S::ZERO)
}

/// `6 (dy_right - dy_left)` at interior node `i`.
fn interior_rhs<S: RealScalar>(h: &[S], y: &[S], i: usize) -> Result<S, SciOxideError> {
    let six = S::TWO * (S::TWO + S::ONE);
    let right = secant(y[i + 1], y[i], h[i])?;
    let left = secant(y[i], y[i - 1], h[i - 1])?;
    mul(six, sub(right, left)?)
}

type Diagonals<S> = (Vec<S>, Vec<S>, Vec<S>);

/// Assembles only the unknown rows. Normalization bounds each width by two,
/// so these diagonal sums cannot overflow. Natural ends omit both end rows.
fn spline_bands<S: RealScalar>(h: &[S], interior: bool) -> Result<Diagonals<S>, SciOxideError> {
    let n = h.len() + 1;
    let off = if interior { &h[1..n - 2] } else { h };
    let lower = scioxide_core::backend::try_copy_slice(
        off,
        scioxide_core::quantity!("spline_lower_diagonal"),
    )?;
    let upper = scioxide_core::backend::try_copy_slice(
        off,
        scioxide_core::quantity!("spline_upper_diagonal"),
    )?;
    let mut main = try_with_capacity(
        if interior { n - 2 } else { n },
        scioxide_core::quantity!("spline_main_diagonal"),
    )?;
    if !interior {
        main.push(S::ZERO);
    }
    main.extend(h.windows(2).map(|pair| S::TWO * (pair[0] + pair[1])));
    if !interior {
        main.push(S::ZERO);
    }
    Ok((lower, main, upper))
}

/// Second-derivative system factored once for all components.
// Construction-local storage: boxing the rare fallback would add a fallible allocation.
#[allow(clippy::large_enum_variant)]
enum System<S: RealScalar> {
    /// Two-node natural/not-a-knot/periodic: zero second derivatives.
    Linear,
    /// Three-node not-a-knot: constant second derivative.
    Quadratic,
    /// Three-node periodic: a two-by-two cyclic system.
    PeriodicPair,
    /// Interior system with zero end second derivatives.
    Natural(TridiagonalFactor<S>),
    /// Full system with end slopes.
    Clamped(TridiagonalFactor<S>),
    /// Interior system with the not-a-knot ratios `h0/h1` and `h_{n-2}/h_{n-3}`.
    NotAKnot(TridiagonalFactor<S>, [S; 2]),
    /// Cyclic system for `σ_0..σ_{n-2}` with `σ_{n-1} = σ_0`.
    Periodic(CyclicTridiagonalFactor<S>),
    PeriodicFallback(CyclicTridiagonalPlan<S>),
}

impl<S: RealScalar> System<S> {
    fn new<V>(h: &[S], boundary: &SplineBoundary<V>) -> Result<Self, SciOxideError> {
        let n = h.len() + 1;
        let two = S::TWO;
        let options = LinalgExecutionOptions::new();
        match boundary {
            SplineBoundary::Natural | SplineBoundary::NotAKnot | SplineBoundary::Periodic
                if n == 2 =>
            {
                return Ok(Self::Linear);
            }
            SplineBoundary::NotAKnot if n == 3 => return Ok(Self::Quadratic),
            SplineBoundary::Periodic if n == 3 => return Ok(Self::PeriodicPair),
            _ => {}
        }
        let (mut lower, mut main, mut upper) =
            spline_bands(h, matches!(boundary, SplineBoundary::Natural))?;
        match boundary {
            SplineBoundary::Natural => Ok(Self::Natural(TridiagonalFactor::factor(
                lower,
                main,
                upper,
                scioxide_core::ExecutionContext::sequential_native(),
            )?)),
            SplineBoundary::Clamped(..) => {
                main[0] = mul(two, h[0])?;
                main[n - 1] = mul(two, h[n - 2])?;
                Ok(Self::Clamped(TridiagonalFactor::factor(
                    lower,
                    main,
                    upper,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?))
            }
            SplineBoundary::NotAKnot => {
                let first = div(h[0], h[1])?;
                main[1] = add(main[1], mul(h[0], add(S::ONE, first)?)?)?;
                upper[1] = sub(h[1], mul(h[0], first)?)?;
                let last = div(h[n - 2], h[n - 3])?;
                main[n - 2] = add(main[n - 2], mul(h[n - 2], add(S::ONE, last)?)?)?;
                lower[n - 3] = sub(h[n - 3], mul(h[n - 2], last)?)?;
                lower.copy_within(1..n - 2, 0);
                lower.truncate(n - 3);
                upper.copy_within(1..n - 2, 0);
                upper.truncate(n - 3);
                main.copy_within(1..n - 1, 0);
                main.truncate(n - 2);
                Ok(Self::NotAKnot(
                    TridiagonalFactor::factor(
                        lower,
                        main,
                        upper,
                        scioxide_core::ExecutionContext::sequential_native(),
                    )?,
                    [first, last],
                ))
            }
            SplineBoundary::Periodic => {
                let m = n - 1;
                main[0] = mul(two, add(h[m - 1], h[0])?)?;
                lower.truncate(m - 1);
                upper.truncate(m - 1);
                main.truncate(m);
                match CyclicTridiagonalFactor::factor(lower, main, upper, h[m - 1], h[m - 1]) {
                    Ok(factor) => Ok(Self::Periodic(factor)),
                    Err(error)
                        if matches!(
                            error.kind(),
                            scioxide_core::SciOxideErrorKind::Singular
                                | scioxide_core::SciOxideErrorKind::NumericalInstability
                                | scioxide_core::SciOxideErrorKind::NonFinite
                        ) =>
                    {
                        // Rebuild only the failed auxiliary split. The common
                        // path consumes its diagonal buffers without copies.
                        let mut main = try_filled(
                            m,
                            S::ZERO,
                            scioxide_core::quantity!("spline_main_diagonal"),
                        )?;
                        main[0] = mul(two, add(h[m - 1], h[0])?)?;
                        for (i, slot) in main.iter_mut().enumerate().skip(1) {
                            *slot = mul(two, add(h[i - 1], h[i])?)?;
                        }
                        let off = &h[..m - 1];
                        Ok(Self::PeriodicFallback(CyclicTridiagonalPlan::new(
                            TridiagonalMatrixView::new(off, &main, off)?,
                            h[m - 1],
                            h[m - 1],
                            &options,
                        )?))
                    }
                    Err(error) => Err(error),
                }
            }
        }
    }

    /// Normalized second derivatives of one component; `ends` are the clamped
    /// slopes in normalized units.
    fn solve_into(
        &self,
        h: &[S],
        y: &[S],
        ends: Option<[S; 2]>,
        sigma: &mut [S],
        native: bool,
    ) -> Result<(), SciOxideError> {
        let n = y.len();
        let six = S::TWO * (S::TWO + S::ONE);
        sigma[0] = S::ZERO;
        sigma[n - 1] = S::ZERO;
        for (i, slot) in sigma.iter_mut().enumerate().take(n - 1).skip(1) {
            *slot = if native {
                // Six times a difference of admitted secants cannot overflow.
                six * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
            } else {
                interior_rhs(h, y, i)?
            };
        }
        match self {
            Self::Linear => {}
            Self::Quadratic => {
                let curvature = div(sigma[1], mul(S::TWO + S::ONE, add(h[0], h[1])?)?)?;
                sigma.fill(curvature);
            }
            Self::PeriodicPair => {
                let curvature = div(sigma[1], add(h[0], h[1])?)?;
                sigma.copy_from_slice(&[-curvature, curvature, -curvature]);
            }
            Self::Natural(factor) => factor.solve_in_place(
                &mut sigma[1..n - 1],
                scioxide_core::ExecutionContext::sequential_native(),
            )?,
            Self::Clamped(factor) => {
                let [first, last] = ends.unwrap_or([S::ZERO; 2]);
                sigma[0] = mul(six, sub(secant(y[1], y[0], h[0])?, first)?)?;
                sigma[n - 1] = mul(six, sub(last, secant(y[n - 1], y[n - 2], h[n - 2])?)?)?;
                factor
                    .solve_in_place(sigma, scioxide_core::ExecutionContext::sequential_native())?;
            }
            Self::NotAKnot(factor, [first, last]) => {
                factor.solve_in_place(
                    &mut sigma[1..n - 1],
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                sigma[0] = add(sigma[1], mul(*first, sub(sigma[1], sigma[2])?)?)?;
                sigma[n - 1] = add(sigma[n - 2], mul(*last, sub(sigma[n - 2], sigma[n - 3])?)?)?;
            }
            Self::Periodic(_) | Self::PeriodicFallback(_) => {
                sigma[0] = mul(
                    six,
                    sub(secant(y[1], y[0], h[0])?, secant(y[0], y[n - 2], h[n - 2])?)?,
                )?;
                match self {
                    Self::Periodic(factor) => factor.solve_in_place(&mut sigma[..n - 1])?,
                    Self::PeriodicFallback(plan) => {
                        let result = plan.solve(StructuredOperation::Normal, &sigma[..n - 1])?;
                        sigma[..n - 1].copy_from_slice(&result.into_vec());
                    }
                    _ => {}
                }
                sigma[n - 1] = sigma[0];
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocol::{Differentiable, ExtrapolationPolicy, Integrable, Interpolant};
    use crate::test_support::assert_close;

    #[test]
    fn admitted_preparation_matches_checked_rhs_and_slopes() {
        let h = [0.3_f64, 0.7, 1.0];
        let y = [0.0, 1.0, -0.5, 0.0];
        for boundary in [
            SplineBoundary::Natural,
            SplineBoundary::Clamped(0.2, -0.3),
            SplineBoundary::NotAKnot,
            SplineBoundary::Periodic,
        ] {
            let system = System::new(&h, &boundary).unwrap();
            let ends = if matches!(boundary, SplineBoundary::Clamped(..)) {
                Some([0.2, -0.3])
            } else {
                None
            };
            let mut native = [0.0; 4];
            let mut checked = native;
            system.solve_into(&h, &y, ends, &mut native, true).unwrap();
            system
                .solve_into(&h, &y, ends, &mut checked, false)
                .unwrap();
            for (a, b) in native.into_iter().zip(checked) {
                assert_close(a, b, 1e-13, 1e-13);
            }
            let mut a = native;
            let mut b = checked;
            node_slopes(&h, &y, &mut a, 2.0, true, Some(0.5)).unwrap();
            node_slopes(&h, &y, &mut b, 2.0, false, None).unwrap();
            for (a, b) in a.into_iter().zip(b) {
                assert_close(a, b, 1e-13, 1e-13);
            }
        }
    }

    #[test]
    fn boundaries_reproduce_polynomials() {
        let reject = ExtrapolationPolicy::Reject;
        let natural = CubicSpline::with_boundary(
            vec![0., 1., 2., 3.],
            vec![0., 1., 0., 1.],
            SplineBoundary::Natural,
        )
        .unwrap();
        for (x, y) in [(0., 0.), (1., 1.), (2., 0.), (3., 1.)] {
            assert_eq!(natural.evaluate([x], reject).unwrap(), y);
        }
        let x: Vec<f64> = (0..5).map(f64::from).collect();
        let square = CubicSpline::new(x.clone(), x.iter().map(|v| v * v).collect()).unwrap();
        assert_close(square.evaluate([2.5], reject).unwrap(), 6.25, 5e-13, 5e-13);
        let line = CubicSpline::with_boundary(
            vec![0., 1., 2., 3.],
            vec![0., 2., 4., 6.],
            SplineBoundary::Clamped(2., 2.),
        )
        .unwrap();
        assert_close(line.gradient([1.5], reject).unwrap()[0], 2., 5e-13, 5e-13);
        assert_close(line.integral([0.], [3.], reject).unwrap(), 9., 5e-13, 5e-13);
    }

    #[test]
    fn periodic_check_is_per_component() {
        let x = vec![0., 1., 2., 3.];
        assert!(
            CubicSpline::with_boundary(
                x.clone(),
                vec![[0., 1.], [1., 0.], [2., 1.], [0., 1.]],
                SplineBoundary::Periodic
            )
            .is_ok()
        );
        assert!(
            CubicSpline::with_boundary(
                x,
                vec![[0., 1.], [1., 0.], [2., 1.], [0., 2.]],
                SplineBoundary::Periodic
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_inputs() {
        let natural =
            |x: Vec<f64>, y: Vec<f64>| CubicSpline::with_boundary(x, y, SplineBoundary::Natural);
        assert_eq!(
            natural(vec![0., f64::MIN_POSITIVE], vec![-f64::MAX, f64::MAX])
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
        assert!(
            CubicSpline::with_boundary(
                vec![0., 1.],
                vec![0., 1.],
                SplineBoundary::Clamped(f64::INFINITY, 1.)
            )
            .is_err()
        );
        assert!(
            CubicSpline::with_boundary(
                vec![0., 1.],
                vec![vec![0., 1.], vec![1., 1.]],
                SplineBoundary::Clamped(vec![1., 2., 3.], vec![1., 2.])
            )
            .is_err()
        );
    }

    #[test]
    fn extreme_opposite_sign_nodes_do_not_overflow() {
        let spline = CubicSpline::with_boundary(
            vec![-f64::MAX, 0.0, f64::MAX],
            vec![2.0, 2.0, 2.0],
            SplineBoundary::Natural,
        )
        .unwrap();
        let reject = ExtrapolationPolicy::Reject;
        assert_eq!(spline.evaluate([f64::MAX / 2.0], reject).unwrap(), 2.0);
        assert_eq!(spline.gradient([-f64::MAX / 2.0], reject).unwrap(), [0.0]);
    }
}
