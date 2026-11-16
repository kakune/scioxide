//! Positive-weight rational tensor B-splines (NURBS).

use crate::validate::{Weights, validate_weights};
use alloc::vec::Vec;

use scioxide_core::backend::scaled::scaled_magnitude_le_tolerance;
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, RealScalar, Scaled, SciOxideError, Tolerance,
};

use super::basis::{BSplineBasis, KnotSide};
use super::bspline::{Located, Tensor, control_count, scratch_len};
use crate::protocol::{Differentiable, ExtrapolationPolicy, Interpolant, TwiceDifferentiable};
use crate::values::{QueryWorkspace, flatten, zeroed};

/// Represents a non-uniform rational B-spline (NURBS) with positive weights,
/// as a curve (`D = 1`) or a tensor-product surface or volume.
///
/// With the tensor basis `B_k` of [`crate::BSpline`], controls `c_k` and
/// weights `w_k > 0`, the model is
///
/// ```text
/// R(x) = N(x) / W(x),   N(x) = Σ_k w_k c_k B_k(x),   W(x) = Σ_k w_k B_k(x).
/// ```
///
/// Scaling all weights by a common factor leaves `R` unchanged, and unit
/// weights give the polynomial B-spline. Weights are divided by their maximum
/// in a wide exponent range, so very large or small weights do not overflow.
/// Gradients and Hessians follow from the quotient rule written in terms of
/// `R`, with subscripts denoting partial derivatives:
///
/// ```text
/// R_a  = (N_a − R W_a) / W
/// R_ab = (N_ab − R_a W_b − R_b W_a − R W_ab) / W
/// ```
///
/// At an interior knot, an ordinary derivative exists if the rational values
/// and derivatives up to the requested order agree in every adjacent tensor
/// piece. A jump in `N` and `W` that cancels in the quotient is allowed. Two
/// pieces agree when each entry differs by at most `64 (2 + Σ_a p_a) ε` times
/// the sum of their first-order rounding-error bounds, where `ε` is the
/// machine epsilon; otherwise the derivative is `DomainError` (`Undefined`).
/// The domain endpoints use the inward piece. Under
/// [`ExtrapolationPolicy::Clamp`](crate::ExtrapolationPolicy::Clamp),
/// derivatives along axes strictly outside the domain are zero.
///
/// A query evaluates the `(p + 1)^D` block of nonzero basis functions. The wide
/// exponent range avoids overflow and underflow, but not the cancellation of
/// an ill-conditioned quotient: results carry the precision of `S`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BSplineBasis, ExtrapolationPolicy, Interpolant, Nurbs};
/// // A quarter of the unit circle as a quadratic rational Bézier curve.
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2)?;
/// let middle = 0.5_f64.sqrt();
/// let curve = Nurbs::new(
///     [basis],
///     vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
///     vec![1.0, middle, 1.0],
/// )?;
/// let [x, y] = curve.evaluate([0.5], ExtrapolationPolicy::Reject)?;
/// assert!((x * x + y * y - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., Springer, 1997,
/// chapter 4.
#[derive(Debug, Clone)]
#[must_use]
pub struct Nurbs<S: RealScalar, V: AffineValue<Scale = S> = S, const D: usize = 1> {
    bases: [BSplineBasis<S>; D],
    /// Factored homogeneous controls `[control][c_0, ..., c_{dim-1}, w]`.
    /// Normalized weights and deferred products keep the wide exponent range.
    homogeneous: Vec<Scaled<S>>,
    dimension: usize,
    template: V,
    scratch: usize,
    scaled_scratch: usize,
}

impl<S: RealScalar, V: AffineValue<Scale = S>, const D: usize> Nurbs<S, V, D> {
    /// Builds a NURBS from one basis per axis, the control values and the
    /// weights, both in row-major multi-index order.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of controls or weights differs from
    ///   the product of the axis control counts, or the controls differ in shape.
    /// - `InvalidInput` (`OutOfRange`) if a weight is not positive or `D = 0`.
    /// - `NonFinite` if a weight or control component is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be reserved.
    pub fn new(
        bases: [BSplineBasis<S>; D],
        controls: Vec<V>,
        weights: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        let count = control_count(&bases)?;
        let flat = flatten(controls, count)?;
        validate_weights(
            &weights,
            count,
            Weights::Positive,
            scioxide_core::argument!("weights"),
        )?;
        let maximum = weights.iter().fold(S::ZERO, |maximum, &w| maximum.max(w));
        let dimension = flat.dimension;
        let width = checked_add(dimension, 1, scioxide_core::quantity!("nurbs_width"))?;
        let mut homogeneous = try_with_capacity(
            checked_mul(
                count,
                width,
                scioxide_core::quantity!("homogeneous_controls"),
            )?,
            scioxide_core::quantity!("homogeneous_controls"),
        )?;
        let maximum = Scaled::from_value(maximum)?;
        for (control, &weight) in weights.iter().enumerate() {
            let normalized = Scaled::from_value(weight)?.div(maximum)?;
            for &component in &flat.components[control * dimension..(control + 1) * dimension] {
                homogeneous.push(Scaled::from_value(component)?);
            }
            homogeneous.push(normalized);
        }
        let dimension_squared = checked_mul(D, D, scioxide_core::quantity!("hessian_entries"))?;
        let quantity = scioxide_core::quantity!("nurbs_workspace");
        // Staged quotient value, gradient and Hessian, plus homogeneous value,
        // gradient and one Hessian entry.
        let quotients = checked_mul(
            checked_add(checked_add(1, D, quantity)?, dimension_squared, quantity)?,
            dimension,
            quantity,
        )?;
        // Two quotient jets and their two rounding-mass jets.
        let stage = checked_add(
            checked_mul(4, quotients, quantity)?,
            checked_mul(checked_add(2, D, quantity)?, width, quantity)?,
            quantity,
        )?;
        let scaled_scratch = checked_add(stage, dimension, quantity)?;
        let scratch = scratch_len(
            &bases,
            checked_mul(dimension_squared, dimension, quantity)?,
            0,
        )?;
        Ok(Self {
            bases,
            homogeneous,
            dimension,
            template: flat.template,
            scratch,
            scaled_scratch,
        })
    }

    /// Returns the axis bases in coordinate order.
    pub const fn bases(&self) -> &[BSplineBasis<S>; D] {
        &self.bases
    }

    /// Evaluates the value (`level 0`), gradient (`1`) or Hessian (`2`) and writes it to `out`.
    fn quotient<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        level: usize,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let dim = self.dimension;
        let width = dim + 1;
        let outputs = [dim, D * dim, D * D * dim][level];
        scioxide_core::check_value_shape(out, outputs)?;
        let (native, scratch) = workspace.scratch_with_scaled(self.scratch, self.scaled_scratch)?;
        let (staged, axes) = native.split_at_mut(D * D * dim);
        let (origin, scratch) = scratch.split_at_mut(dim);
        let jet_len = (1 + D + D * D) * dim;
        let (jet, scratch) = scratch.split_at_mut(jet_len);
        let (other, scratch) = scratch.split_at_mut(jet_len);
        let (mass, scratch) = scratch.split_at_mut(jet_len);
        let (other_mass, scratch) = scratch.split_at_mut(jet_len);
        let tensor = Tensor {
            bases: &self.bases,
            coefficients: &self.homogeneous,
            width,
            centered: None,
            magnitude: false,
            magnitudes: None,
        };
        let located = tensor.locate(point, None, policy)?;
        let reference = tensor.reference(&located, axes)?;
        for (c, slot) in origin.iter_mut().enumerate() {
            *slot = self.homogeneous[reference * width + c];
        }
        let tensor = Tensor {
            centered: Some(origin),
            ..tensor
        };
        let magnitude = Tensor {
            magnitude: true,
            ..tensor
        };
        let mut sides = [KnotSide::Right; D];
        let denominator = self.jet(
            &tensor,
            &located.with_sides(sides),
            level,
            jet,
            scratch,
            axes,
        )?;
        if level > 0 {
            let degrees = self.bases.iter().map(BSplineBasis::degree).sum::<usize>();
            let tolerance = Tolerance::new(
                S::ZERO,
                S::try_from_f64_lossy(64. * (2 + degrees) as f64)? * S::EPSILON,
            )?;
            let mut measured = false;
            let check: [bool; D] = core::array::from_fn(|axis| {
                let basis = &self.bases[axis];
                let domain = basis.domain();
                let t = point[axis];
                t > domain.lower()
                    && t < domain.upper()
                    && basis.multiplicity(t) > basis.degree().saturating_sub(level)
            });
            // A binary counter over only the axes with uncertain continuity.
            while let Some(axis) =
                (0..D).find(|&axis| check[axis] && sides[axis] == KnotSide::Right)
            {
                for side in &mut sides[..axis] {
                    *side = KnotSide::Right;
                }
                sides[axis] = KnotSide::Left;
                if !measured {
                    let right = located.with_sides([KnotSide::Right; D]);
                    self.envelope(
                        &magnitude,
                        &right,
                        level,
                        denominator,
                        jet,
                        mass,
                        scratch,
                        axes,
                    )?;
                    measured = true;
                }
                let piece = located.with_sides(sides);
                let other_denominator = self.jet(&tensor, &piece, level, other, scratch, axes)?;
                self.envelope(
                    &magnitude,
                    &piece,
                    level,
                    other_denominator,
                    other,
                    other_mass,
                    scratch,
                    axes,
                )?;
                let count = (if level == 1 { 1 + D } else { 1 + D + D * D }) * dim;
                for entry in 0..count {
                    if !scaled_magnitude_le_tolerance(
                        jet[entry].sub(other[entry])?,
                        &tolerance,
                        mass[entry].add(other_mass[entry])?,
                    )? {
                        return Err(SciOxideError::domain_error(
                            scioxide_core::argument!("point"),
                            scioxide_core::DomainReason::Undefined,
                            "rational knot-side limits disagree through the requested order",
                        ));
                    }
                }
            }
        }
        let (value, rest) = jet.split_at_mut(dim);
        let (gradient, hessian) = rest.split_at_mut(D * dim);
        let result = match level {
            0 => &*value,
            1 => &*gradient,
            _ => &*hessian,
        };
        for (c, (slot, value)) in staged[..outputs].iter_mut().zip(result).enumerate() {
            *slot = if level == 0 {
                value.add(origin[c])?
            } else {
                *value
            }
            .try_value()?;
        }
        crate::values::commit_finite(&staged[..outputs], out);
        Ok(())
    }
    /// One rational jet in extended range, using a common numerator origin.
    /// Returns the positive denominator `W`.
    fn jet(
        &self,
        tensor: &Tensor<'_, S, D, Scaled<S>>,
        located: &Located<S, D>,
        level: usize,
        jet: &mut [Scaled<S>],
        scratch: &mut [Scaled<S>],
        axes: &mut [S],
    ) -> Result<Scaled<S>, SciOxideError> {
        let dim = self.dimension;
        let width = dim + 1;
        let (value, rest) = jet.split_at_mut(dim);
        let (gradient, hessian) = rest.split_at_mut(D * dim);
        let (homogeneous, rest) = scratch.split_at_mut(width);
        let (homogeneous_gradient, homogeneous_hessian) = rest.split_at_mut(D * width);
        let homogeneous_hessian = &mut homogeneous_hessian[..width];
        tensor.partial(located, [0; D], &mut [], axes, homogeneous)?;
        // Positive weights make W > 0 inside the domain; continuation may drive
        // it to zero or below, which is reported rather than divided through.
        let denominator = homogeneous[dim];
        if denominator.cmp_value(Scaled::ZERO).is_le() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "NURBS denominator is non-positive",
            ));
        }
        for (slot, &numerator) in value.iter_mut().zip(&homogeneous[..dim]) {
            *slot = numerator.div(denominator)?;
        }
        // `value` is the correction to the fixed control origin, often much
        // smaller than R. Differentiate this correction and add the origin only
        // when publishing a value; derivative publication never adds it.
        if level >= 1 {
            for axis in 0..D {
                let mut orders = [0; D];
                orders[axis] = 1;
                let partial = &mut homogeneous_gradient[axis * width..(axis + 1) * width];
                tensor.partial(located, orders, &mut [], axes, partial)?;
                let weight = partial[dim];
                for c in 0..dim {
                    gradient[axis * dim + c] =
                        partial[c].sub(value[c].mul(weight)?)?.div(denominator)?;
                }
            }
        }
        if level == 2 {
            for row in 0..D {
                for column in 0..D {
                    let mut orders = [0; D];
                    orders[row] += 1;
                    orders[column] += 1;
                    tensor.partial(located, orders, &mut [], axes, homogeneous_hessian)?;
                    let row_weight = homogeneous_gradient[row * width + dim];
                    let column_weight = homogeneous_gradient[column * width + dim];
                    for c in 0..dim {
                        let numerator = homogeneous_hessian[c]
                            .sub(gradient[row * dim + c].mul(column_weight)?)?
                            .sub(gradient[column * dim + c].mul(row_weight)?)?
                            .sub(value[c].mul(homogeneous_hessian[dim])?)?;
                        hessian[(row * D + column) * dim + c] = numerator.div(denominator)?;
                    }
                }
            }
        }
        Ok(denominator)
    }

    /// First-order rounding masses of one piece's quotient jet. `magnitude`
    /// contracts `|B| |w (c - origin)|` and `|B| w`; each quotient-rule term
    /// then contributes its operand masses, so an entry's rounding error is a
    /// small multiple of epsilon times its mass, even when the entry is zero.
    #[allow(clippy::too_many_arguments)]
    fn envelope(
        &self,
        magnitude: &Tensor<'_, S, D, Scaled<S>>,
        located: &Located<S, D>,
        level: usize,
        denominator: Scaled<S>,
        jet: &[Scaled<S>],
        mass: &mut [Scaled<S>],
        scratch: &mut [Scaled<S>],
        axes: &mut [S],
    ) -> Result<(), SciOxideError> {
        let dim = self.dimension;
        let width = dim + 1;
        let (value, rest) = jet.split_at(dim);
        let (gradient, hessian) = rest.split_at(D * dim);
        let (value_mass, rest) = mass.split_at_mut(dim);
        let (gradient_mass, hessian_mass) = rest.split_at_mut(D * dim);
        let (homogeneous, rest) = scratch.split_at_mut(width);
        let (homogeneous_gradient, homogeneous_hessian) = rest.split_at_mut(D * width);
        let homogeneous_hessian = &mut homogeneous_hessian[..width];
        magnitude.partial(located, [0; D], &mut [], axes, homogeneous)?;
        let weight = homogeneous[dim];
        // Mass of a value-like entry `x` scaled by the denominator mass.
        let term =
            |x: Scaled<S>, x_mass: Scaled<S>, factor: Scaled<S>| x.abs().add(x_mass)?.mul(factor);
        for c in 0..dim {
            value_mass[c] = homogeneous[c]
                .add(value[c].abs().mul(weight)?)?
                .div(denominator)?;
        }
        for axis in 0..D {
            let mut orders = [0; D];
            orders[axis] = 1;
            let partial = &mut homogeneous_gradient[axis * width..(axis + 1) * width];
            magnitude.partial(located, orders, &mut [], axes, partial)?;
            for c in 0..dim {
                let entry = axis * dim + c;
                gradient_mass[entry] = partial[c]
                    .add(term(value[c], value_mass[c], partial[dim])?)?
                    .add(gradient[entry].abs().mul(weight)?)?
                    .div(denominator)?;
            }
        }
        if level == 2 {
            for row in 0..D {
                for column in 0..D {
                    let mut orders = [0; D];
                    orders[row] += 1;
                    orders[column] += 1;
                    magnitude.partial(located, orders, &mut [], axes, homogeneous_hessian)?;
                    let row_weight = homogeneous_gradient[row * width + dim];
                    let column_weight = homogeneous_gradient[column * width + dim];
                    for c in 0..dim {
                        let (r, k) = (row * dim + c, column * dim + c);
                        let entry = (row * D + column) * dim + c;
                        hessian_mass[entry] = homogeneous_hessian[c]
                            .add(term(gradient[r], gradient_mass[r], column_weight)?)?
                            .add(term(gradient[k], gradient_mass[k], row_weight)?)?
                            .add(term(value[c], value_mass[c], homogeneous_hessian[dim])?)?
                            .add(hessian[entry].abs().mul(weight)?)?
                            .div(denominator)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> Interpolant<S, D>
    for Nurbs<S, V, D>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; D] {
        core::array::from_fn(|axis| self.bases[axis].domain())
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::with_scaled(self.scratch, self.scaled_scratch)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.scaled_capacity(self.scratch, self.scaled_scratch)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.quotient(point, policy, 0, out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> Differentiable<S, D>
    for Nurbs<S, V, D>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.quotient(point, policy, 1, out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> TwiceDifferentiable<S, D>
    for Nurbs<S, V, D>
{
    fn hessian_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.quotient(point, policy, 2, out, workspace)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::BSpline;

    fn linear() -> BSplineBasis<f64> {
        BSplineBasis::new(vec![0., 0., 1., 1.], 1).unwrap()
    }

    #[test]
    fn rational_linear_closed_form() {
        // R(x) = 2x / (1 + x) for controls [0, 1] and weights [1, 2].
        let model = Nurbs::new([linear()], vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        let policy = ExtrapolationPolicy::Reject;
        assert!((model.evaluate([0.25], policy).unwrap() - 0.4).abs() < 1e-15);
        assert!((model.gradient([0.25], policy).unwrap()[0] - 1.28).abs() < 1e-14);
        let second = -4.0 / 1.25_f64.powi(3);
        assert!((model.hessian([0.25], policy).unwrap()[0][0] - second).abs() < 1e-14);
    }

    #[test]
    fn unit_weights_reproduce_the_polynomial_exactly() {
        let polynomial = BSpline::new([linear()], vec![2.0, 6.0]).unwrap();
        let scaled = Nurbs::new([linear()], vec![2.0, 6.0], vec![1e200, 1e200]).unwrap();
        let policy = ExtrapolationPolicy::Reject;
        assert_eq!(
            scaled.evaluate([0.375], policy).unwrap(),
            polynomial.evaluate([0.375], policy).unwrap()
        );
    }

    #[test]
    fn invalid_weights_and_breakdown() {
        assert!(Nurbs::new([linear()], vec![1.0, 2.0], vec![1.0, 0.0]).is_err());
        assert_eq!(
            Nurbs::new([linear()], vec![1.0, 2.0], vec![1.0])
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        let extreme = Nurbs::new(
            [linear()],
            vec![1.0, 2.0],
            vec![f64::MAX, f64::MIN_POSITIVE],
        )
        .unwrap();
        assert_eq!(
            extreme
                .evaluate([1.0], ExtrapolationPolicy::Reject)
                .unwrap(),
            2.0
        );
        // Continuation far outside drives W through zero.
        let model = Nurbs::new([linear()], vec![0.0, 1.0], vec![1.0, 1e-3]).unwrap();
        assert!(
            model
                .evaluate([2000.0], ExtrapolationPolicy::MethodDefined)
                .is_err()
        );
    }
}
