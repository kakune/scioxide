//! Diagonal rational interpolation with the Bulirsch–Stoer algorithm.

#![allow(clippy::many_single_char_names)]

use alloc::vec::Vec;

use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError, check_value_shape,
    write_value,
};

use super::tableau::tableau;
use crate::checked::{intermediate, mul, sub};
use crate::error::instability;
use crate::protocol::{
    ExtrapolationPolicy, Interpolant, NodeSeparation, SampleOptions, domain_from_nodes,
    resolve_coordinate,
};
use crate::validate::validate_nodes;
use crate::values::{QueryWorkspace, flatten, gather, zeroed};
use scioxide_poly::backend::polynomial_basis::linear_difference_ratio;

/// Evaluates the diagonal rational interpolant of `n` samples with the Bulirsch–Stoer algorithm.
///
/// The interpolant is a rational function `p(x) / q(x)` through all samples
/// whose numerator and denominator degrees differ by at most one. Each query
/// builds a Neville-type tableau of rational interpolants on consecutive
/// nodes, starting at the node nearest the query, at a cost of O(n² d)
/// operations for `d` real components. The degrees follow from `n`, so there
/// are no degree parameters. The tableau is nonlinear in the data, and each
/// real component of the values is interpolated separately.
///
/// Rational interpolation suits functions with poles near the real axis,
/// where polynomial interpolation converges slowly. For a rational
/// approximation chosen by its accuracy, see [`crate::aaa`].
///
/// # Errors
///
/// - The query errors described in the [crate docs](crate#errors).
/// - `NumericalInstability` (`Breakdown`) if the interpolant has a pole at
///   the query, or a tableau denominator vanishes there.
/// - `NonFinite` if an intermediate or the result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BulirschStoerInterpolator, ExtrapolationPolicy, Interpolant};
/// let interpolator =
///     BulirschStoerInterpolator::new(vec![0.0_f64, 1.0, 2.0], vec![1.0_f64, 0.5, 1.0 / 3.0])?;
/// assert_eq!(
///     interpolator.evaluate([1.0], ExtrapolationPolicy::Reject)?,
///     0.5
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Stoer and R. Bulirsch, *Introduction to Numerical Analysis*, 3rd ed.,
/// Springer, 2002, §2.2.
#[derive(Debug, Clone)]
#[must_use]
pub struct BulirschStoerInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    x: Vec<S>,
    /// `[node][component]` values.
    y: Vec<S>,
    template: V,
    dimension: usize,
    domain: Interval<S>,
    separation: NodeSeparation<S>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> BulirschStoerInterpolator<S, V> {
    /// Stores the samples `(x[j], y[j])`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// `x` must be strictly increasing and only an exactly zero tableau
    /// denominator counts as zero.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ or the values differ in shape.
    /// - `InvalidInput` if `x` is empty or not strictly increasing.
    /// - `NonFinite` if a node or value is NaN or infinite.
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, SampleOptions::new())
    }

    /// Stores the samples `(x[j], y[j])`, validating the nodes under `options`.
    ///
    /// The nodes `x` must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`]. The separation also decides when a
    /// tableau denominator counts as zero.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ or the values differ in shape.
    /// - `InvalidInput` if `x` is empty, or not strictly increasing and distinct
    ///   under the separation of `options`.
    /// - `NonFinite` if a node or value is NaN or infinite.
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
        Ok(Self {
            domain: domain_from_nodes(&x)?,
            x,
            y: flat.components,
            template: flat.template,
            dimension: flat.dimension,
            separation,
        })
    }

    /// Returns the nodes in increasing order.
    pub fn nodes(&self) -> &[S] {
        &self.x
    }

    /// Returns the values as real components, node by node (`[node][component]`).
    pub fn values(&self) -> &[S] {
        &self.y
    }

    /// Scratch length: values plus two tableau tables.
    fn scratch_len(&self) -> usize {
        self.dimension + 2 * self.x.len()
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 1>
    for BulirschStoerInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 1] {
        [self.domain]
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.scratch_len())
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.scratch_len())
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.dimension)?;
        let scratch = workspace.scratch(self.scratch_len())?;
        let (values, tables) = scratch.split_at_mut(self.dimension);
        let (c, d) = tables.split_at_mut(self.x.len());
        let t = resolve_coordinate(&self.domain, point[0], policy)?;
        for (component, slot) in values.iter_mut().enumerate() {
            gather(&self.y, self.dimension, component, c);
            d.copy_from_slice(c);
            *slot = bulirsch_stoer_eval(&self.x, t, self.separation, c, d)?;
        }
        write_value(values, out)
    }
}

/// Diagonal rational tableau for one component; `c` and `d` hold its samples.
fn bulirsch_stoer_eval<S: RealScalar>(
    x: &[S],
    t: S,
    separation: NodeSeparation<S>,
    c: &mut [S],
    d: &mut [S],
) -> Result<S, SciOxideError> {
    // Diagonal rational tableau (Stoer–Bulirsch; NR `ratint`). With
    // r = (t - x_i) / (t - x_{i+m}) and t1 = r * D_{m-1,i}:
    //   factor  = (C_{m-1,i+1} - D_{m-1,i}) / (t1 - C_{m-1,i+1}),
    //   D_{m,i} = C_{m-1,i+1} * factor,  C_{m,i} = t1 * factor.
    let (value, _) = tableau(x, t, separation, c, d, |c, d, index, order| {
        // Both inputs zero means this entry has already converged (e.g.
        // locally constant data): the new corrections are zero, but the
        // generic formula would evaluate 0/0 when `t1 == c`. NR avoids this
        // with an additive "tiny"; propagating exact zeros is scale-free.
        if c[index + 1] == S::ZERO && d[index] == S::ZERO {
            c[index] = S::ZERO;
            d[index] = S::ZERO;
            return Ok(());
        }
        let ratio = linear_difference_ratio(t, x[index], t, x[index + order])?;
        let t1 = mul(ratio, d[index])?;
        let denominator = sub(t1, c[index + 1])?;
        // A vanishing denominator means the rational interpolant has a pole at
        // (or, under a tolerant policy, numerically indistinguishable from)
        // the query. The test is relative to the operands being subtracted so
        // it is invariant to the scale of the data.
        let denominator_scale = t1.abs().max(c[index + 1].abs());
        if separation.is_zero_at_scale(denominator, denominator_scale)? {
            return Err(instability(
                scioxide_core::NumericalIssue::Breakdown,
                "pole encountered in the diagonal rational tableau",
            ));
        }
        let factor = intermediate((c[index + 1] - d[index]) / denominator)?;
        d[index] = mul(c[index + 1], factor)?;
        c[index] = mul(t1, factor)?;
        Ok(())
    })?;
    Ok(value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    // The crate's unit-test allocator is installed in the barycentric tests.

    #[test]
    fn constructs_and_evaluates_without_degree_parameters() {
        let interpolator =
            BulirschStoerInterpolator::new(vec![0.0, 1.0, 2.0], vec![1.0, 0.5, 1.0 / 3.0]).unwrap();
        assert_eq!(
            interpolator
                .evaluate([1.0], ExtrapolationPolicy::Reject)
                .unwrap(),
            0.5
        );
    }

    #[test]
    fn extreme_opposite_sign_nodes_do_not_create_nan() {
        let interpolator =
            BulirschStoerInterpolator::new(vec![-f64::MAX, f64::MAX], vec![2.0, 2.0]).unwrap();
        let value = interpolator
            .evaluate([0.0], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_eq!(value, 2.0);
    }

    #[test]
    fn poles_policies_and_workspace_reuse() {
        let rational = BulirschStoerInterpolator::new(vec![0.0, 1.0], vec![1.0, -1.0]).unwrap();
        let mut workspace = rational.workspace().unwrap();
        let mut out = 9.0;
        assert!(
            rational
                .evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out, &mut workspace)
                .is_err()
        );
        assert_eq!(out, 9.0);
        {
            rational
                .evaluate_into([-1.0], ExtrapolationPolicy::Clamp, &mut out, &mut workspace)
                .unwrap();
        };
        assert_eq!(out, 1.0);
        let small = BulirschStoerInterpolator::new(vec![0.0], vec![1.0])
            .unwrap()
            .workspace()
            .unwrap();
        let nodes: Vec<f64> = (0..9).map(f64::from).collect();
        let values: Vec<f64> = nodes.iter().map(|x| 1.0 / (1.0 + x)).collect();
        let large = BulirschStoerInterpolator::new(nodes, values).unwrap();
        let mut small = small;
        assert_eq!(
            large
                .evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out, &mut small)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn affine_components_match_scalar_interpolants_bit_for_bit() {
        let x = vec![0.0_f32, 1.0, 2.0, 4.0];
        let first: Vec<f32> = x.iter().map(|x| 1.0 / (1.0 + x)).collect();
        let second: Vec<f32> = x.iter().map(|x| (2.0 + x) / (3.0 + x * x)).collect();
        let values: Vec<[f32; 2]> = first.iter().zip(&second).map(|(&a, &b)| [a, b]).collect();
        let vector = BulirschStoerInterpolator::new(x.clone(), values).unwrap();
        let a = BulirschStoerInterpolator::new(x.clone(), first).unwrap();
        let b = BulirschStoerInterpolator::new(x, second).unwrap();
        for query in [0.5_f32, 1.5, 3.0] {
            let value = vector
                .evaluate([query], ExtrapolationPolicy::Reject)
                .unwrap();
            assert_eq!(
                value[0],
                a.evaluate([query], ExtrapolationPolicy::Reject).unwrap()
            );
            assert_eq!(
                value[1],
                b.evaluate([query], ExtrapolationPolicy::Reject).unwrap()
            );
        }
    }
}
