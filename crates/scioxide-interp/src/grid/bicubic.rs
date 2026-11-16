//! Bicubic Hermite interpolation on a rectilinear grid.
//!
//! Each cell stores the 16 coefficients of its bicubic polynomial, computed
//! from the corner values and finite-difference estimates of the partial and
//! mixed derivatives.

#![allow(clippy::many_single_char_names, clippy::similar_names)]

use alloc::vec::Vec;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, NonFiniteOrigin, RealScalar, SciOxideError,
};

use super::GridAxes;
use crate::checked::{add, mul, output, sub};
use crate::hermite_basis::power;
use crate::protocol::{
    Differentiable, ExtrapolationPolicy, Interpolant, NodeSeparation, SampleOptions,
    TwiceDifferentiable, clamped_axes,
};
use crate::values::{QueryWorkspace, zeroed};
use scioxide_poly::backend::polynomial_basis::{
    linear_difference_ratio, linear_multiply_difference,
};

/// Interpolates values on a rectilinear grid by bicubic Hermite interpolation.
///
/// The grid has nodes `x[0] < … < x[nx−1]` and `y[0] < … < y[ny−1]`, not
/// necessarily equally spaced. In each cell the interpolant is the bicubic
/// polynomial `Σᵢⱼ aᵢⱼ tⁱ uʲ`, `0 ≤ i, j ≤ 3`, in the local coordinates
/// `t, u ∈ [0, 1]`, that matches the values, the first partial derivatives and
/// the mixed derivative `∂²f/∂x∂y` at the four corners. [`Self::new`] estimates
/// the derivatives at each node from the three-point quadratic through the node
/// and its neighbors, one-sided at the ends of an axis. An axis with two nodes
/// uses its secant. Mixed derivatives use the tensor product of the same
/// stencils. The result is C¹ and reproduces quadratics along each axis with at
/// least three nodes. [`Self::monotone`] chooses the derivatives so the
/// surface is monotone along both axes.
///
/// Construction costs O(nx · ny) time and memory per value component: 16
/// coefficients per cell. A query locates its cell by binary search in
/// O(log nx + log ny) time. Gradients and Hessians differentiate the cell
/// polynomial and are returned in the units of `x` and `y`. Very uneven
/// spacing amplifies errors in the samples.
///
/// # Errors
///
/// [`Self::with_options`] and [`Self::monotone`] list the construction errors.
/// Queries fail as described in the [crate docs](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     BicubicInterpolator, Differentiable, ExtrapolationPolicy, Interpolant, TwiceDifferentiable,
/// };
///
/// let x = vec![0.0_f64, 1.0, 2.0, 3.0];
/// let y = vec![0.0_f64, 1.0, 2.0, 3.0];
/// // z = x + y
/// let mut z = Vec::new();
/// for &yv in &y {
///     for &xv in &x {
///         z.push(xv + yv);
///     }
/// }
/// let interp = BicubicInterpolator::new(x, y, z)?;
/// let policy = ExtrapolationPolicy::Reject;
/// assert!((interp.evaluate([1.5, 1.5], policy)? - 3.0_f64).abs() < 1e-10);
/// let [dx, dy] = interp.gradient([1.5, 1.5], policy)?;
/// assert!((dx - 1.0).abs() < 1e-10 && (dy - 1.0).abs() < 1e-10);
/// assert!(
///     interp
///         .hessian([1.5, 1.5], policy)?
///         .as_flattened()
///         .iter()
///         .all(|h| h.abs() < 1e-10)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct BicubicInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    template: V,
    dimension: usize,
    axes: GridAxes<S>,
    /// Coefficients `aᵢⱼ` of `Σ aᵢⱼ tⁱ uʲ` on the unit cell, 16 per cell.
    coeffs: Vec<[S; 16]>,
    nx: usize,
    native_cells: Vec<bool>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> BicubicInterpolator<S, V> {
    /// Builds the interpolant from the axes `x`, `y` and the grid values `z`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`]: each axis
    /// needs at least two strictly increasing nodes. The values are stored row
    /// by row, with `x` varying fastest: `z[j * nx + i]` is the value at
    /// `(x[i], y[j])`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if an axis has fewer than two nodes, or its nodes are not
    ///   strictly increasing.
    /// - `DimensionMismatch` if `z.len() != nx · ny`, or the values differ in
    ///   shape.
    /// - `NonFinite` if a node or value is NaN or infinite, or a derivative
    ///   estimate or coefficient overflows.
    pub fn new(x: Vec<S>, y: Vec<S>, z: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, z, SampleOptions::new())
    }

    /// Builds the interpolant from the axes `x`, `y` and the grid values `z`,
    /// validating the nodes under `options`.
    ///
    /// Each axis needs at least two strictly increasing nodes that are distinct
    /// under [`SampleOptions::node_separation`]. The values are stored row by
    /// row, with `x` varying fastest: `z[j * nx + i]` is the value at
    /// `(x[i], y[j])`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if an axis has fewer than two nodes, or its nodes are not
    ///   strictly increasing and distinct under the separation of `options`.
    /// - `DimensionMismatch` if `z.len() != nx · ny`, or the values differ in
    ///   shape.
    /// - `NonFinite` if a node or value is NaN or infinite, or a derivative
    ///   estimate or coefficient overflows.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<S>,
        z: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(x, y, z, options.node_separation(), false)
    }

    /// Builds a C¹ bicubic interpolant that is monotone along both axes.
    ///
    /// Every real value component must be monotone along each axis, in the same
    /// direction for all rows (or all columns). The direction is read from the
    /// grid edges, and plateaus are allowed. Inside every closed cell the
    /// surface is then monotone in the same directions, up to rounding, and does
    /// not overshoot its four corner values. The guarantee does not cover
    /// extrapolation beyond the grid.
    ///
    /// The first partial derivatives start from the PCHIP slopes along each grid
    /// line (see [`PchipInterpolator`](crate::PchipInterpolator)). They are then
    /// reduced until sufficient sign conditions on the Bernstein coefficients of
    /// each cell hold, and all mixed derivatives are zero. The surface reproduces
    /// planes but may be flatter than necessary, and it does not reproduce
    /// quadratics. The layout, cost and queries are those of [`Self::new`], and
    /// the nodes must be strictly increasing.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if a component increases along an axis in one place and
    ///   decreases along the same axis in another.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BicubicInterpolator, ExtrapolationPolicy, Interpolant};
    /// // A step along x: the monotone surface stays within [0, 1].
    /// let (x, y) = (vec![0.0_f64, 1.0, 2.0, 3.0], vec![0.0, 1.0]);
    /// let z = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    /// let surface = BicubicInterpolator::monotone(x, y, z)?;
    /// let value = surface.evaluate([0.5, 0.5], ExtrapolationPolicy::Reject)?;
    /// assert!((0.0..=1.0).contains(&value));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn monotone(x: Vec<S>, y: Vec<S>, z: Vec<V>) -> Result<Self, SciOxideError> {
        Self::prepare(x, y, z, NodeSeparation::Exact, true)
    }

    /// Builds the interpolant with estimated or monotone derivatives.
    #[allow(clippy::needless_pass_by_value)]
    fn prepare(
        x: Vec<S>,
        y: Vec<S>,
        z: Vec<V>,
        separation: NodeSeparation<S>,
        monotone: bool,
    ) -> Result<Self, SciOxideError> {
        let axes = GridAxes::new(x, y, separation)?;
        axes.samples(z.len())?;
        let (x, y) = (&axes.x, &axes.y);
        let nx = x.len();
        let ny = y.len();
        let dimension = z[0].dimension()?;
        for value in &z {
            scioxide_core::check_value_shape(value, dimension)?;
            scioxide_core::validate_value(value)?;
        }

        let constraints = if monotone {
            let mut channels = try_with_capacity(
                dimension,
                scioxide_core::quantity!("monotone_derivative_channels"),
            )?;
            let mut values = try_with_capacity(
                z.len(),
                scioxide_core::quantity!("monotone_component_samples"),
            )?;
            for component in 0..dimension {
                values.clear();
                for value in &z {
                    values.push(value.component(component)?);
                }
                channels.push(super::monotone::derivatives(x, y, &values)?);
            }
            Some(channels)
        } else {
            None
        };

        // Stencils produce derivatives in the receiving cell's unit coordinates.
        // No physical derivative or product of physical axis widths is formed.
        let ncells = scioxide_core::backend::checked_mul(
            nx - 1,
            ny - 1,
            scioxide_core::quantity!("grid_cells"),
        )?;
        let mut coeffs = try_with_capacity(
            scioxide_core::backend::checked_mul(
                ncells,
                dimension,
                scioxide_core::quantity!("bicubic_channel_coefficients"),
            )?,
            scioxide_core::quantity!("bicubic_interpolation_coefficients"),
        )?;
        // Each axis stencil depends only on the axis and receiving cell.
        let stencils = |nodes: &[S]| -> Result<Vec<[Stencil<S>; 2]>, SciOxideError> {
            let mut result = try_with_capacity(
                if monotone { 0 } else { nodes.len() - 1 },
                scioxide_core::quantity!("grid_stencils"),
            )?;
            if !monotone {
                for i in 0..nodes.len() - 1 {
                    result.push([Stencil::new(nodes, i, i)?, Stencil::new(nodes, i + 1, i)?]);
                }
            }
            Ok(result)
        };
        let xs = stencils(x)?;
        let ys = stencils(y)?;
        let mut native_cells =
            try_with_capacity(ncells * dimension, scioxide_core::quantity!("grid_ranges"))?;
        for iy in 0..ny - 1 {
            let sy = ys.get(iy);
            for ix in 0..nx - 1 {
                let sx = xs.get(ix);
                // Corners are ordered [00, 10, 01, 11]. The two stencils per axis
                // are for the cell's left/bottom and right/top nodes, both scaled
                // to this cell's width. The monotone path converts its physical
                // slopes to unit-cell slopes by multiplying by the cell width and
                // leaves `fxy` at zero (zero twist), as the monotone limiter assumes.
                for component in 0..dimension {
                    let mut f = [S::ZERO; 4];
                    let mut fx = f;
                    let mut fy = f;
                    let mut fxy = f;
                    for j in 0..2 {
                        for i in 0..2 {
                            let corner = j * 2 + i;
                            let node = (iy + j) * nx + ix + i;
                            f[corner] = z[node].component(component)?;
                            if let Some(channels) = &constraints {
                                let (dx, dy) = &channels[component];
                                fx[corner] =
                                    linear_multiply_difference(dx[node], x[ix + 1], x[ix])?;
                                fy[corner] =
                                    linear_multiply_difference(dy[node], y[iy + 1], y[iy])?;
                            } else if let (Some(sx), Some(sy)) = (&sx, &sy) {
                                let dx = &sx[i];
                                let dy = &sy[j];
                                fx[corner] =
                                    dx.apply(|k| z[(iy + j) * nx + k].component(component))?;
                                fy[corner] =
                                    dy.apply(|k| z[k * nx + ix + i].component(component))?;
                                fxy[corner] = dy.apply(|ky| {
                                    dx.apply(|kx| z[ky * nx + kx].component(component))
                                })?;
                            }
                        }
                    }
                    let coefficients = compute_bicubic_coeffs(&f, &fx, &fy, &fxy)?;
                    if coefficients.iter().any(|v| !v.is_finite()) {
                        return Err(SciOxideError::non_finite(
                            NonFiniteOrigin::Intermediate,
                            "derived bicubic coefficient is non-finite",
                        ));
                    }
                    native_cells.push(
                        axes.native && coefficients.iter().all(|&v| crate::checked::ordinary(v)),
                    );
                    coeffs.push(coefficients);
                }
            }
        }
        let template = z.into_iter().next().ok_or_else(|| {
            SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(1),
                scioxide_core::Dimensions::one(0),
            )
        })?;
        Ok(Self {
            axes,
            native_cells,
            template,
            dimension,
            coeffs,
            nx,
        })
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S>> BicubicInterpolator<S, V> {
    /// One cell partial.
    fn component_at<const DX: usize, const DY: usize>(
        &self,
        cell: (usize, usize, S, S),
        clamp: [bool; 2],
        component: usize,
    ) -> Result<S, SciOxideError> {
        let (ix, iy, t, u) = cell;
        if (DX > 0 && clamp[0]) || (DY > 0 && clamp[1]) {
            return Ok(S::ZERO);
        }
        let index = (iy * (self.nx - 1) + ix) * self.dimension + component;
        let a = &self.coeffs[index];
        if self.native_cells[index] && t >= S::ZERO && t <= S::ONE && u >= S::ZERO && u <= S::ONE {
            let value = evaluate_native::<S, DX, DY>(a, t, u);
            let dx = self.axes.x[ix + 1] - self.axes.x[ix];
            let dy = self.axes.y[iy + 1] - self.axes.y[iy];
            let mut result = value;
            let mut normal = true;
            for width in core::iter::repeat_n(dx, DX).chain(core::iter::repeat_n(dy, DY)) {
                let previous = result;
                result /= width;
                // A tiny intermediate may be enlarged by the next division;
                // use the binary-scaled path rather than publishing lost bits.
                normal &= previous == S::ZERO || result.abs() >= S::MIN_POSITIVE;
            }
            if normal && result.is_finite() {
                return output(result);
            }
        }
        // Differentiate in unit-cell coordinates, then convert: each x (y)
        // derivative divides by the cell width in x (y). First derivatives use
        // `linear_difference_ratio` (value / width without forming the width); second
        // derivatives keep both widths in binary-exponent form so neither
        // dx*dy nor dx^2 has to be representable.
        let value = evaluate::<S, DX, DY>(a, t, u)?;
        let result = if DX + DY == 0 {
            value
        } else if DX + DY == 1 {
            let (lo, hi) = if DX == 1 {
                (self.axes.x[ix], self.axes.x[ix + 1])
            } else {
                (self.axes.y[iy], self.axes.y[iy + 1])
            };
            linear_difference_ratio(value, S::ZERO, hi, lo)?
        } else {
            let dx = binary_width(self.axes.x[ix], self.axes.x[ix + 1])?;
            let dy = binary_width(self.axes.y[iy], self.axes.y[iy + 1])?;
            physical_second(
                value,
                if DX > 0 { dx } else { dy },
                if DY > 0 { dy } else { dx },
            )?
        };
        output(result)
    }
    /// Evaluates the partials selected by `partials` (one `[DX, DY]` pair per
    /// output value) into `workspace`, then copies them to `out`.
    fn stage<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        partials: &[(usize, usize)],
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let width = partials.len() * self.dimension;
        scioxide_core::check_value_shape(out, width)?;
        let staged = workspace.scratch(width)?;
        let cell = self.axes.locate(point, policy)?;
        let clamp = clamped_axes(policy, self.axes.domains, point);
        for (row, &partial) in partials.iter().enumerate() {
            let rows = &mut staged[row * self.dimension..(row + 1) * self.dimension];
            for (component, slot) in rows.iter_mut().enumerate() {
                *slot = match partial {
                    (0, 0) => self.component_at::<0, 0>(cell, clamp, component),
                    (1, 0) => self.component_at::<1, 0>(cell, clamp, component),
                    (0, 1) => self.component_at::<0, 1>(cell, clamp, component),
                    (2, 0) => self.component_at::<2, 0>(cell, clamp, component),
                    (1, 1) => self.component_at::<1, 1>(cell, clamp, component),
                    _ => self.component_at::<0, 2>(cell, clamp, component),
                }?;
            }
        }
        crate::values::commit_finite(staged, out);
        Ok(())
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 2>
    for BicubicInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 2] {
        self.axes.domains
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(scioxide_core::backend::checked_mul(
            self.dimension,
            4,
            scioxide_core::quantity!("bicubic_workspace"),
        )?)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.dimension.saturating_mul(4))
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage(point, policy, &[(0, 0)], out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Differentiable<S, 2>
    for BicubicInterpolator<S, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage(point, policy, &[(1, 0), (0, 1)], out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> TwiceDifferentiable<S, 2>
    for BicubicInterpolator<S, V>
{
    fn hessian_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        // Bicubic polynomials have symmetric mixed partials.
        self.stage(
            point,
            policy,
            &[(2, 0), (1, 1), (1, 1), (0, 2)],
            out,
            workspace,
        )
    }
}

// ---------------------------------------------------------------------------
// Bicubic coefficient computation
// ---------------------------------------------------------------------------

/// Compute the 16 bicubic coefficients from function values and derivatives
/// at the four corners of a unit cell.
///
/// Input arrays are indexed as: [00, 10, 01, 11] (bottom-left, bottom-right,
/// top-left, top-right).
fn compute_bicubic_coeffs<S: RealScalar>(
    f: &[S; 4],
    fx: &[S; 4],
    fy: &[S; 4],
    fxy: &[S; 4],
) -> Result<[S; 16], SciOxideError> {
    // Coefficients A = M C M^T, where C holds the corner constraints
    // [[f, f_y], [f_x, f_xy]] (rows: x = 0, x = 1 values then x-derivatives;
    // columns likewise in y) and M is the cubic Hermite→power transform.
    // Row-major A[i][j] multiplies t^i u^j.
    let constraints = [
        [f[0], f[2], fy[0], fy[2]],
        [f[1], f[3], fy[1], fy[3]],
        [fx[0], fx[2], fxy[0], fxy[2]],
        [fx[1], fx[3], fxy[1], fxy[3]],
    ];
    let hermite = |[p0, p1, d0, d1]: [S; 4]| power([p0, sub(p1, p0)?, d0, d1]);
    let mut transformed = [[S::ZERO; 4]; 4];
    for j in 0..4 {
        let column = hermite(constraints.map(|row| row[j]))?;
        for (row, value) in transformed.iter_mut().zip(column) {
            row[j] = value;
        }
    }
    let mut coefficients = [S::ZERO; 16];
    for (target, row) in coefficients.chunks_exact_mut(4).zip(transformed) {
        target.copy_from_slice(&hermite(row)?);
    }
    Ok(coefficients)
}

/// Three-node quadratic derivative, expressed as two weighted differences.
/// Subtracting values first reproduces constants exactly. On a two-node axis the
/// single difference is the derivative in the unit cell.
struct Stencil<S> {
    first: usize,
    weights: [(S, i32); 2],
    quadratic: bool,
}

impl<S: RealScalar> Stencil<S> {
    /// Build the stencil, reporting failures.
    fn new(nodes: &[S], at: usize, cell: usize) -> Result<Self, SciOxideError> {
        if nodes.len() == 2 {
            return Ok(Self {
                first: 0,
                weights: [(S::ONE, 0), (S::ZERO, 0)],
                quadratic: false,
            });
        }
        // Use the three consecutive nodes centred on `at` where possible and
        // shifted inwards at the boundaries (one-sided quadratic).
        // With h_l = b - a, h_r = c - b and H = c - a, the quadratic's derivative
        // at a, b, c is a combination of the secants (f_b - f_a)/h_l and
        // (f_c - f_b)/h_r with weights given by `blend` in terms of the
        // fractions h_l/H and h_r/H. Multiplying by the receiving cell's width
        // gives the derivative in that cell's unit coordinate.
        let first = at.saturating_sub(1).min(nodes.len() - 3);
        let [a, b, c] = [nodes[first], nodes[first + 1], nodes[first + 2]];
        let left_fraction = linear_difference_ratio(b, a, c, a)?;
        let right_fraction = linear_difference_ratio(c, b, c, a)?;
        // Differentiate the local quadratic. Compute complementary weights
        // independently so a very small neighboring gap is not rounded away.
        let blend = match at - first {
            0 => [add(S::ONE, left_fraction)?, -left_fraction],
            1 => [right_fraction, left_fraction],
            _ => [-right_fraction, add(S::ONE, right_fraction)?],
        };
        let cell_left = nodes[cell];
        let cell_right = nodes[cell + 1];
        let ordinary = || -> Result<[S; 2], SciOxideError> {
            Ok([
                mul(
                    blend[0],
                    linear_difference_ratio(cell_right, cell_left, b, a)?,
                )?,
                mul(
                    blend[1],
                    linear_difference_ratio(cell_right, cell_left, c, b)?,
                )?,
            ])
        };
        let weights = if let Ok(weights) = ordinary()
            && weights.iter().all(|&weight| weight != S::ZERO)
        {
            weights.map(|weight| (weight, 0))
        } else {
            // An extreme spacing ratio may not fit S even though its product
            // with a sample difference does. Retain its binary exponent until
            // application, also recovering weights that would underflow.
            let left = binary_width(a, b)?;
            let right = binary_width(b, c)?;
            let span = binary_width(a, c)?;
            let cell = binary_width(cell_left, cell_right)?;
            let weight = |numerator: (S, i32), denominator: (S, i32)| {
                (
                    (numerator.0 * cell.0) / (span.0 * denominator.0),
                    numerator.1 + cell.1 - span.1 - denominator.1,
                )
            };
            match at - first {
                0 => {
                    let w = weight(left, right);
                    [(blend[0], 0), (-w.0, w.1)]
                }
                1 => [weight(right, left), weight(left, right)],
                _ => {
                    let w = weight(right, left);
                    [(-w.0, w.1), (blend[1], 0)]
                }
            }
        };
        Ok(Self {
            first,
            weights,
            quadratic: true,
        })
    }

    /// Apply the stencil, reporting failures.
    fn apply(&self, value: impl Fn(usize) -> Result<S, SciOxideError>) -> Result<S, SciOxideError> {
        // Exponent-zero weights need exactly the original IEEE operations.
        // Any overflowing intermediate propagates to a nonfinite result, so one
        // final check suffices; retry preserves the original structured error.
        if self.weights.iter().all(|weight| weight.1 == 0) {
            let middle = value(self.first + 1)?;
            let left = (middle - value(self.first)?) * self.weights[0].0;
            let result = if self.quadratic {
                left + (value(self.first + 2)? - middle) * self.weights[1].0
            } else {
                left
            };
            if result.is_finite() {
                return Ok(result);
            }
        }
        let middle = value(self.first + 1)?;
        let left = apply_weight(sub(middle, value(self.first)?)?, self.weights[0])?;
        if self.quadratic {
            add(
                left,
                apply_weight(sub(value(self.first + 2)?, middle)?, self.weights[1])?,
            )
        } else {
            Ok(left)
        }
    }
}

fn apply_weight<S: RealScalar>(value: S, weight: (S, i32)) -> Result<S, SciOxideError> {
    if value == S::ZERO {
        return Ok(value);
    }
    if weight.1 == 0 {
        return mul(value, weight.0);
    }
    let (mantissa, exponent) = binary_split(value)?;
    output((mantissa * weight.0).scalbn(exponent + weight.1))
}

/// Horner evaluation of a partial derivative of the unit-cell polynomial.
#[inline]
fn evaluate<S: RealScalar, const DX: usize, const DY: usize>(
    coefficients: &[S; 16],
    t: S,
    u: S,
) -> Result<S, SciOxideError> {
    (DX..4).rev().try_fold(S::ZERO, |sum, i| {
        let row = (DY..4).rev().try_fold(S::ZERO, |sum, j| {
            let factor = (0..DY).fold(1, |factor, k| factor * (j - k));
            let coefficient = coefficients[4 * i + j];
            let term = if DY == 0 {
                coefficient
            } else {
                mul(coefficient, falling(factor))?
            };
            add(mul(sum, u)?, term)
        })?;
        let factor = (0..DX).fold(1, |factor, k| factor * (i - k));
        let term = if DX == 0 {
            row
        } else {
            mul(row, falling(factor))?
        };
        add(mul(sum, t)?, term)
    })
}

/// Evaluates a partial derivative of the unit-cell polynomial in plain
/// arithmetic. The coefficient bound checked at construction limits Horner rows
/// and differentiated rows to 576 · max|a| on the unit square, well below the
/// overflow threshold.
#[inline]
fn evaluate_native<S: RealScalar, const DX: usize, const DY: usize>(a: &[S; 16], t: S, u: S) -> S {
    (DX..4).rev().fold(S::ZERO, |sum, i| {
        let row = (DY..4).rev().fold(S::ZERO, |sum, j| {
            let factor = (0..DY).fold(1, |f, k| f * (j - k));
            sum * u + a[4 * i + j] * falling(factor)
        });
        let factor = (0..DX).fold(1, |f, k| f * (i - k));
        sum * t + row * falling(factor)
    })
}

/// Exact falling-factorial factor (1, 2, 3 or 6) of a cubic derivative coefficient.
#[inline]
fn falling<S: RealScalar>(factor: usize) -> S {
    let three = S::TWO + S::ONE;
    match factor {
        2 => S::TWO,
        3 => three,
        6 => three * S::TWO,
        _ => S::ONE,
    }
}

/// `(m, e)` with `value = m 2^e`, `e` from core's binary exponent and
/// `1/2 <= |m| < 1`; zero is `(0, 0)`. Factors are kept separate until final
/// rounding, never forming dx*dy or dx².
fn binary_split<S: RealScalar>(value: S) -> Result<(S, i32), SciOxideError> {
    let exponent = scioxide_core::backend::standard::exponent(value)?;
    Ok((value.scalbn(-exponent), exponent))
}

fn binary_width<S: RealScalar>(left: S, right: S) -> Result<(S, i32), SciOxideError> {
    // A width that overflows (opposite-sign extremes) is halved first and the
    // lost factor of two is restored in the exponent.
    let width = right - left;
    if width.is_finite() {
        return binary_split(width);
    }
    let two = S::TWO;
    let (mantissa, exponent) = binary_split(sub(right / two, left / two)?)?;
    Ok((mantissa, exponent + 1))
}

fn physical_second<S: RealScalar>(
    value: S,
    dx: (S, i32),
    dy: (S, i32),
) -> Result<S, SciOxideError> {
    if value == S::ZERO {
        return Ok(value);
    }
    let (mantissa, exponent) = binary_split(value)?;
    output((mantissa / (dx.0 * dy.0)).scalbn(exponent - dx.1 - dy.1))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, assert_error};

    #[test]
    fn native_and_range_safe_cell_partials_agree() {
        let axis = vec![0.0_f64, 0.3, 1.0, 2.0];
        let values = axis
            .iter()
            .flat_map(|y| axis.iter().map(move |x| x * x + x * y - y * y))
            .collect();
        let native = BicubicInterpolator::new(axis.clone(), axis, values).unwrap();
        assert!(native.native_cells.iter().all(|&admitted| admitted));
        let mut fallback = native.clone();
        fallback.axes.native = false;
        fallback.native_cells.fill(false);
        for point in [[0.0, 0.0], [0.1, 0.2], [0.8, 1.7], [2.0, 2.0]] {
            let policy = ExtrapolationPolicy::Reject;
            assert_close(
                native.evaluate(point, policy).unwrap(),
                fallback.evaluate(point, policy).unwrap(),
                1e-13,
                1e-13,
            );
            let a = native.gradient(point, policy).unwrap();
            let b = fallback.gradient(point, policy).unwrap();
            for (a, b) in a.into_iter().zip(b) {
                assert_close(a, b, 1e-12, 1e-12);
            }
            let a = native.hessian(point, policy).unwrap();
            let b = fallback.hessian(point, policy).unwrap();
            for (a, b) in a.into_iter().flatten().zip(b.into_iter().flatten()) {
                assert_close(a, b, 1e-11, 1e-11);
            }
        }
    }

    #[test]
    fn exact_hermite_cubic_and_all_partial_derivatives() {
        // Independent tensor Hermite constraints for f(t,u)=t³u³+2t²u+3u.
        let f = |t: f64, u: f64| t.powi(3) * u.powi(3) + 2.0 * t * t * u + 3.0 * u;
        let fx = |t: f64, u: f64| 3.0 * t * t * u.powi(3) + 4.0 * t * u;
        let fy = |t: f64, u: f64| 3.0 * t.powi(3) * u * u + 2.0 * t * t + 3.0;
        let fxy = |t: f64, u: f64| 9.0 * t * t * u * u + 4.0 * t;
        let corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let a = compute_bicubic_coeffs(
            &corners.map(|(t, u)| f(t, u)),
            &corners.map(|(t, u)| fx(t, u)),
            &corners.map(|(t, u)| fy(t, u)),
            &corners.map(|(t, u)| fxy(t, u)),
        )
        .unwrap();
        let (t, u) = (0.25, 0.75);
        let close = |actual: f64, expected: f64| assert!((actual - expected).abs() < 1e-13);
        close(evaluate::<_, 0, 0>(&a, t, u).unwrap(), f(t, u));
        close(evaluate::<_, 1, 0>(&a, t, u).unwrap(), fx(t, u));
        close(evaluate::<_, 0, 1>(&a, t, u).unwrap(), fy(t, u));
        close(evaluate::<_, 1, 1>(&a, t, u).unwrap(), fxy(t, u));
        close(
            evaluate::<_, 2, 0>(&a, t, u).unwrap(),
            6.0 * t * u.powi(3) + 4.0 * u,
        );
        close(evaluate::<_, 0, 2>(&a, t, u).unwrap(), 6.0 * t.powi(3) * u);
    }

    #[test]
    fn test_new_valid() {
        let x = vec![0.0_f64, 1.0, 2.0, 3.0];
        let y = vec![0.0_f64, 1.0, 2.0, 3.0];
        let z = vec![0.0; 16];
        assert!(BicubicInterpolator::new(x, y, z).is_ok());
    }

    #[test]
    fn test_new_dimension_mismatch() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.0_f64, 1.0, 2.0];
        let z = vec![0.0; 10]; // should be 9
        assert_error(
            &BicubicInterpolator::new(x, y, z),
            scioxide_core::SciOxideErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn rejects_non_finite_derived_derivatives_with_intermediate_origin() {
        let result = BicubicInterpolator::new(
            vec![0.0, f64::MIN_POSITIVE],
            vec![0.0, 1.0],
            vec![-f64::MAX, f64::MAX, -f64::MAX, f64::MAX],
        );

        assert!(matches!(
            assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Intermediate,
                ..
            }
        ));
    }

    #[test]
    fn extreme_grid_coordinates_use_normalized_cell_coordinates() {
        let interp = BicubicInterpolator::new(
            vec![-f64::MAX, f64::MAX],
            vec![-f64::MAX, f64::MAX],
            vec![2.0; 4],
        )
        .unwrap();

        assert_eq!(
            interp
                .evaluate([0.0, 0.0], ExtrapolationPolicy::Reject)
                .unwrap(),
            2.0
        );
        assert_eq!(
            interp
                .gradient([0.0, 0.0], ExtrapolationPolicy::Reject)
                .unwrap(),
            [0.0, 0.0]
        );
    }

    #[test]
    fn test_bicubic_at_nodes() {
        let x = vec![0.0_f64, 1.0, 2.0, 3.0];
        let y = vec![0.0_f64, 1.0, 2.0, 3.0];
        let mut z = Vec::new();
        for &yv in &y {
            for &xv in &x {
                z.push(xv + yv);
            }
        }
        let interp = BicubicInterpolator::new(x.clone(), y.clone(), z.clone()).unwrap();
        for (jy, &yv) in y.iter().enumerate() {
            for (ix, &xv) in x.iter().enumerate() {
                let v = interp
                    .evaluate([xv, yv], ExtrapolationPolicy::Reject)
                    .unwrap();
                assert_close(v, z[jy * 4 + ix], 5e-11, 5e-11);
            }
        }
    }

    #[test]
    fn test_bicubic_planar() {
        // z = x + y should be reproduced exactly
        let x = vec![0.0_f64, 1.0, 2.0, 3.0];
        let y = vec![0.0_f64, 1.0, 2.0, 3.0];
        let mut z = Vec::new();
        for &yv in &y {
            for &xv in &x {
                z.push(xv + yv);
            }
        }
        let interp = BicubicInterpolator::new(x, y, z).unwrap();
        let v = interp
            .evaluate([1.5, 1.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(v, 3.0, 5e-11, 5e-11);
    }

    #[test]
    fn test_gradient_planar() {
        // z = 2x + 3y => gradient = (2, 3)
        let x = vec![0.0_f64, 1.0, 2.0, 3.0];
        let y = vec![0.0_f64, 1.0, 2.0, 3.0];
        let mut z = Vec::new();
        for &yv in &y {
            for &xv in &x {
                z.push(2.0_f64.mul_add(xv, 3.0 * yv));
            }
        }
        let interp = BicubicInterpolator::new(x, y, z).unwrap();
        let [gx, gy] = interp
            .gradient([1.5, 1.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(gx, 2.0, 5e-11, 5e-11);
        assert_close(gy, 3.0, 5e-11, 5e-11);
    }

    #[test]
    fn test_out_of_domain() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.0_f64, 1.0, 2.0];
        let z = vec![0.0; 9];
        let interp = BicubicInterpolator::new(x, y, z).unwrap();
        assert_error(
            &interp.evaluate([3.0, 1.0], ExtrapolationPolicy::Reject),
            scioxide_core::SciOxideErrorKind::DomainError,
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod monotone_evidence {
    use super::*;

    // Independent power-to-Bernstein conversion. Positivity of every adjacent
    // control difference proves the derivative sign on the entire unit cell.
    fn bernstein(a: &[f64; 16]) -> [[f64; 4]; 4] {
        let choose = [
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 2., 1., 0.],
            [1., 3., 3., 1.],
        ];
        core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                (0..=i)
                    .flat_map(|k| {
                        (0..=j).map(move |l| {
                            a[k * 4 + l] * choose[i][k] / choose[3][k] * choose[j][l] / choose[3][l]
                        })
                    })
                    .sum()
            })
        })
    }
    #[test]
    fn all_cell_derivative_bernstein_coefficients_have_the_required_sign() {
        for seed in 0..32 {
            let x = vec![0., 0.125, 0.5, 2., 8.];
            let y = vec![-2., -1.875, 0., 0.25];
            let mut values = vec![0.; x.len() * y.len()];
            // Monotone grids with zeros, flat edges and varying mixed changes.
            for j in 0..y.len() {
                for i in 0..x.len() {
                    let left = if i > 0 {
                        values[j * x.len() + i - 1]
                    } else {
                        0.
                    };
                    let below = if j > 0 {
                        values[(j - 1) * x.len() + i]
                    } else {
                        0.
                    };
                    values[j * x.len() + i] = f64::max(left, below)
                        + f64::from(((i * 7 + j * 11 + seed * 13) % 5) as u32);
                }
            }
            for reverse_x in [false, true] {
                for reverse_y in [false, true] {
                    let z = (0..y.len())
                        .flat_map(|j| {
                            let values = &values;
                            (0..x.len()).map(move |i| {
                                values[(if reverse_y { 3 - j } else { j }) * 5
                                    + if reverse_x { 4 - i } else { i }]
                            })
                        })
                        .collect();
                    let plan = BicubicInterpolator::monotone(x.clone(), y.clone(), z).unwrap();
                    let sx = if reverse_x { -1. } else { 1. };
                    let sy = if reverse_y { -1. } else { 1. };
                    for a in &plan.coeffs {
                        let b = bernstein(a);
                        let tol = 2e-12;
                        for i in 0..3 {
                            for j in 0..4 {
                                assert!(sx * (b[i + 1][j] - b[i][j]) >= -tol, "x: {b:?}");
                                assert!(sy * (b[j][i + 1] - b[j][i]) >= -tol, "y: {b:?}");
                            }
                        }
                    }
                }
            }
        }
    }
}
