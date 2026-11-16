//! Bilinear interpolation on a rectilinear grid.

use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;

use scioxide_core::{AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError};

use super::GridAxes;
use crate::checked::{add, output};
use crate::protocol::{ExtrapolationPolicy, Interpolant, SampleOptions};
use crate::values::{QueryWorkspace, flatten, zeroed};
use scioxide_poly::backend::polynomial_basis::linear_multiply_difference;

/// Interpolates values on a rectilinear grid by bilinear interpolation.
///
/// The grid has nodes `x[0] < … < x[nx−1]` and `y[0] < … < y[ny−1]`, not
/// necessarily equally spaced. In the cell `[xᵢ, xᵢ₊₁] × [yⱼ, yⱼ₊₁]`, with local
/// coordinates `t = (x − xᵢ)/(xᵢ₊₁ − xᵢ)` and `u = (y − yⱼ)/(yⱼ₊₁ − yⱼ)`,
///
/// ```text
/// f(x, y) = (1 − u)((1 − t) z₀₀ + t z₁₀) + u((1 − t) z₀₁ + t z₁₁),
/// ```
///
/// where `zₐᵦ` is the value at `(xᵢ₊ₐ, yⱼ₊ᵦ)`. The interpolant is continuous but
/// not differentiable across cell edges, so this type implements only
/// [`Interpolant`](crate::Interpolant). A query locates its cell by binary
/// search in O(log nx + log ny) time.
///
/// # Errors
///
/// [`Self::with_options`] lists the construction errors. Queries fail as
/// described in the [crate docs](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{BilinearInterpolator, ExtrapolationPolicy, Interpolant};
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![0.0_f64, 1.0, 2.0];
/// // z[j][i] = x_i + y_j
/// let z = vec![
///     0.0, 1.0, 2.0, // y=0
///     1.0, 2.0, 3.0, // y=1
///     2.0, 3.0, 4.0, // y=2
/// ];
/// let interp = BilinearInterpolator::new(x.clone(), y.clone(), z)?;
/// let value = interp.evaluate([0.5, 0.5], ExtrapolationPolicy::Reject)?;
/// assert!((value - 1.0_f64).abs() < 1e-14);
///
/// let complex = vec![Complex::new(0.0, 1.0); 9];
/// let interp = BilinearInterpolator::new(x, y, complex)?;
/// assert_eq!(
///     interp.evaluate([1.5, 0.5], ExtrapolationPolicy::Reject)?,
///     Complex::new(0.0, 1.0)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct BilinearInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    template: V,
    dimension: usize,
    axes: GridAxes<S>,
    /// Values indexed `[j][i][component]`.
    values: Vec<S>,
    native_values: Vec<bool>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> BilinearInterpolator<S, V> {
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
    /// - `NonFinite` if a node or value is NaN or infinite.
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
    /// - `NonFinite` if a node or value is NaN or infinite.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<S>,
        z: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let axes = GridAxes::new(x, y, options.node_separation())?;
        let expected = axes.samples(z.len())?;
        let flat = flatten(z, expected)?;
        let mut native_values =
            try_with_capacity(flat.dimension, scioxide_core::quantity!("grid_ranges"))?;
        for c in 0..flat.dimension {
            native_values.push(
                axes.native
                    && flat
                        .components
                        .iter()
                        .skip(c)
                        .step_by(flat.dimension)
                        .all(|&v| crate::checked::ordinary(v)),
            );
        }
        Ok(Self {
            axes,
            native_values,
            template: flat.template,
            dimension: flat.dimension,
            values: flat.components,
        })
    }

    /// Interpolates one component in a located cell.
    fn interpolate_component(
        &self,
        cell: (usize, usize, S, S),
        component: usize,
    ) -> Result<S, SciOxideError> {
        let (ix, iy, tx, ty) = cell;
        let nx = self.axes.x.len();
        let at = |i: usize, j: usize| self.values[(j * nx + i) * self.dimension + component];
        let (z00, z10, z01, z11) = (
            at(ix, iy),
            at(ix + 1, iy),
            at(ix, iy + 1),
            at(ix + 1, iy + 1),
        );
        // Tensor-product bilinear form as two nested lerps (along x on the
        // lower and upper rows, then along y); `linear_multiply_difference` multiplies by
        // each corner difference without first forming a value that may
        // overflow for opposite-sign extremes.
        if self.native_values[component]
            && tx >= S::ZERO
            && tx <= S::ONE
            && ty >= S::ZERO
            && ty <= S::ONE
        {
            let lower = z00 + tx * (z10 - z00);
            let upper = z01 + tx * (z11 - z01);
            return output(lower + ty * (upper - lower));
        }
        let lower = add(z00, linear_multiply_difference(tx, z10, z00)?)?;
        let upper = add(z01, linear_multiply_difference(tx, z11, z01)?)?;
        output(add(lower, linear_multiply_difference(ty, upper, lower)?)?)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 2>
    for BilinearInterpolator<S, V>
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
        QueryWorkspace::new(self.dimension)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.dimension)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        let cell = self.axes.locate(point, policy)?;
        for (component, slot) in staged.iter_mut().enumerate() {
            *slot = self.interpolate_component(cell, component)?;
        }
        crate::values::commit_finite(staged, out);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, assert_error};

    #[test]
    fn admitted_bilinear_values_match_the_range_kernel() {
        let native = BilinearInterpolator::new(
            vec![0.0_f32, 0.3, 2.0],
            vec![-1.0, 0.5],
            vec![
                [0.0, 3.0],
                [1.0, -2.0],
                [4.0, 5.0],
                [2.0, -1.0],
                [3.0, 2.0],
                [6.0, 7.0],
            ],
        )
        .unwrap();
        assert!(native.native_values.iter().all(|&v| v));
        let mut fallback = native.clone();
        fallback.axes.native = false;
        fallback.native_values.fill(false);
        for point in [[0.0, -1.0], [0.1, 0.0], [1.7, 0.2], [2.0, 0.5]] {
            let a = native.evaluate(point, ExtrapolationPolicy::Reject).unwrap();
            let b = fallback
                .evaluate(point, ExtrapolationPolicy::Reject)
                .unwrap();
            for (a, b) in a.into_iter().zip(b) {
                assert!((a - b).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn test_new_valid() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.0_f64, 1.0];
        let z = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(BilinearInterpolator::new(x, y, z).is_ok());
    }

    #[test]
    fn test_new_dimension_mismatch() {
        let x = vec![0.0_f64, 1.0];
        let y = vec![0.0_f64, 1.0];
        let z = vec![0.0, 1.0, 2.0]; // should be 4
        assert_error(
            &BilinearInterpolator::new(x, y, z),
            scioxide_core::SciOxideErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn test_new_rejects_non_monotone_nodes() {
        let result =
            BilinearInterpolator::new(vec![0.0_f64, 2.0, 1.0], vec![0.0_f64, 1.0], vec![0.0; 6]);

        assert_error(&result, scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_new_rejects_non_finite_values() {
        let result = BilinearInterpolator::new(
            vec![0.0_f64, 1.0],
            vec![0.0_f64, 1.0],
            vec![0.0, 1.0, f64::NAN, 2.0],
        );

        assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_bilinear_at_nodes() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.0_f64, 1.0, 2.0];
        let z = vec![0.0, 1.0, 4.0, 1.0, 2.0, 5.0, 4.0, 5.0, 8.0];
        let interp = BilinearInterpolator::new(x.clone(), y.clone(), z.clone()).unwrap();
        for (jy, &yv) in y.iter().enumerate() {
            for (ix, &xv) in x.iter().enumerate() {
                let v = interp
                    .evaluate([xv, yv], ExtrapolationPolicy::Reject)
                    .unwrap();
                assert_close(v, z[jy * 3 + ix], 5e-15, 5e-15);
            }
        }
    }

    #[test]
    fn test_bilinear_planar() {
        // z = x + y, should be reproduced exactly
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.0_f64, 1.0, 2.0];
        let z = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0];
        let interp = BilinearInterpolator::new(x, y, z).unwrap();
        let v = interp
            .evaluate([0.5, 0.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(v, 1.0, 5e-15, 5e-15);
        let v = interp
            .evaluate([1.5, 0.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(v, 2.0, 5e-15, 5e-15);
    }

    #[test]
    fn test_out_of_domain() {
        let x = vec![0.0_f64, 1.0];
        let y = vec![0.0_f64, 1.0];
        let z = vec![0.0, 1.0, 2.0, 3.0];
        let interp = BilinearInterpolator::new(x, y, z).unwrap();
        assert_error(
            &interp.evaluate([-0.1, 0.5], ExtrapolationPolicy::Reject),
            scioxide_core::SciOxideErrorKind::DomainError,
        );
    }

    #[test]
    fn extreme_grid_coordinates_use_normalized_cell_coordinates() {
        let interp = BilinearInterpolator::new(
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
    }
}
