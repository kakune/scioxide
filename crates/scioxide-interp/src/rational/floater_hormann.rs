//! Floater–Hormann barycentric rational interpolation on increasing real nodes.

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError};

use crate::error::{instability, out_of_range, shape};
use crate::polynomial::barycentric::BarycentricCore;
use crate::protocol::{ExtrapolationPolicy, Interpolant, SampleOptions};
use crate::validate::validate_nodes;
use crate::values::{QueryWorkspace, flatten, zeroed};

/// Interpolates `n` samples with a Floater–Hormann rational function, which has no real poles.
///
/// For a blending degree `d`, the interpolant blends the polynomials that
/// interpolate each run of `d + 1` consecutive nodes:
///
/// ```text
/// r(x) = Σᵢ λᵢ(x) pᵢ(x) / Σᵢ λᵢ(x),   λᵢ(x) = (−1)ⁱ / ((x − xᵢ) ⋯ (x − xᵢ₊d)),
/// ```
///
/// for `i = 0, …, n − 1 − d`. It is evaluated with the barycentric formula of
/// the second kind, with weights
/// `wₖ = (−1)^(k−d) Σᵢ∈Jₖ Πⱼ₌ᵢ..ᵢ₊d, j≠k 1 / |xₖ − xⱼ|`, where
/// `Jₖ = {i : max(0, k − d) ≤ i ≤ min(k, n − 1 − d)}`. The interpolant
/// reproduces polynomials of degree at most `d`, and for `d = n − 1` it is the
/// interpolating polynomial. On equispaced nodes, where polynomial
/// interpolation is ill-conditioned, a small fixed `d` stays well-conditioned
/// as `n` grows. With
/// [`ExtrapolationPolicy::MethodDefined`] the rational function is evaluated
/// outside the node interval.
///
/// Each real component of the values is interpolated separately. The weights
/// cost O(n d²) operations and O(n) memory, and a query costs O(n) per real
/// component.
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
/// use scioxide_interp::{ExtrapolationPolicy, FloaterHormannInterpolator, Interpolant};
///
/// let x = vec![-1.0_f64, -0.7, -0.1, 0.4, 1.0];
/// let y: Vec<f64> = x.iter().map(|x| x * x + 2.0 * x).collect();
/// // Degree two reproduces quadratics.
/// let quadratic = FloaterHormannInterpolator::new(x.clone(), y, 2)?;
/// assert!((quadratic.evaluate([0.5], ExtrapolationPolicy::Reject)? - 1.25).abs() < 1e-13);
///
/// let complex: Vec<_> = x.iter().map(|&x| Complex::new(x, 1.0)).collect();
/// let line = FloaterHormannInterpolator::new(x, complex, 1)?;
/// let value = line.evaluate([0.25], ExtrapolationPolicy::Reject)?;
/// assert!((value - Complex::new(0.25, 1.0)).norm() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. S. Floater and K. Hormann, "Barycentric rational interpolation with no
/// poles and high rates of approximation", *Numer. Math.* 107(2), 2007, 315–331.
#[derive(Debug, Clone)]
#[must_use]
pub struct FloaterHormannInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    core: BarycentricCore<S>,
    template: V,
    degree: usize,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> FloaterHormannInterpolator<S, V> {
    /// Builds the interpolant of blending degree `degree` through `(x[j], y[j])`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// must be strictly increasing. The degree satisfies `0 ≤ degree < x.len()`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ or the values differ in shape.
    /// - `InvalidInput` if `x` is empty or not strictly increasing, or
    ///   `degree ≥ x.len()`.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `NumericalInstability` if a weight is not representable.
    pub fn new(x: Vec<S>, y: Vec<V>, degree: usize) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, degree, SampleOptions::new())
    }

    /// Builds the interpolant of blending degree `degree` through `(x[j], y[j])`,
    /// validating the nodes under `options`.
    ///
    /// The nodes must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`], and `0 ≤ degree < x.len()`. A query
    /// within the separation of a node returns the node's value. The weights
    /// are computed from logarithms of the products, so they do not overflow
    /// for widely spaced or clustered nodes.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ or the values differ in shape.
    /// - `InvalidInput` if `x` is empty, or not strictly increasing and distinct
    ///   under the separation of `options`, or `degree ≥ x.len()`.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `NumericalInstability` if a weight is not representable.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        degree: usize,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let separation = options.node_separation();
        let n = x.len();
        if n != y.len() {
            return Err(shape(n, y.len()));
        }
        validate_nodes(&x, 1, separation, scioxide_core::argument!("x"))?;
        if degree >= n {
            return Err(out_of_range(
                scioxide_core::argument!("degree"),
                "Floater-Hormann degree must be below the node count",
            ));
        }
        let flat = flatten(y, n)?;
        let weights = floater_hormann_weights(&x, degree)?;
        Ok(Self {
            core: BarycentricCore::new(x, weights, flat.components, flat.dimension, separation)?,
            template: flat.template,
            degree,
        })
    }

    /// Returns the blending degree `d`.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the nodes in increasing order.
    pub fn nodes(&self) -> &[S] {
        self.core.nodes()
    }

    /// Returns the barycentric weights `wₖ`, scaled so that the largest has magnitude one.
    pub fn weights(&self) -> &[S] {
        self.core.weights()
    }

    /// Returns the values as real components, node by node (`[node][component]`).
    pub fn values(&self) -> &[S] {
        self.core.values()
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 1>
    for FloaterHormannInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 1] {
        [self.core.domain()]
    }
    fn value_dimension(&self) -> usize {
        self.core.dimension()
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.core.dimension())
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.core.dimension())
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

/// Floater–Hormann weights (2007) normalized to unit maximum magnitude.
fn floater_hormann_weights<S: RealScalar>(
    nodes: &[S],
    degree: usize,
) -> Result<Vec<S>, SciOxideError> {
    let n = nodes.len();
    // w_k = (-1)^(k-d) * sum_{i in J_k} prod_{j=i..i+d, j != k} 1/|x_k - x_j|,
    // J_k = {i : max(0, k-d) <= i <= min(k, n-1-d)}.
    // Every product has exactly `d` factors, so dividing all gaps by one
    // coordinate scale multiplies every weight by the same s^d, which cancels
    // in the barycentric quotient; it merely keeps gaps O(1).
    let coordinate_scale = nodes.iter().fold(S::ZERO, |scale, x| scale.max(x.abs()));
    let coordinate_scale = if coordinate_scale == S::ZERO {
        S::ONE
    } else {
        coordinate_scale
    };
    let mut logs = try_filled(n, S::ZERO, scioxide_core::quantity!("floater_hormann_logs"))?;
    for (k, log) in logs.iter_mut().enumerate() {
        let mut maximum = S::NEG_INFINITY;
        let mut sum = S::ZERO;
        for i in k.saturating_sub(degree)..=k.min(n - 1 - degree) {
            let mut term = S::ZERO;
            for (j, &x) in nodes.iter().enumerate().skip(i).take(degree + 1) {
                if j != k {
                    let gap =
                        scioxide_poly::backend::polynomial_basis::normalized_linear_difference(
                            nodes[k],
                            x,
                            coordinate_scale,
                        )?
                        .abs();
                    if !gap.is_finite() || gap == S::ZERO {
                        return Err(instability(
                            scioxide_core::NumericalIssue::Breakdown,
                            "unrepresentable node separation",
                        ));
                    }
                    term -= gap.ln();
                }
            }
            // Streaming log-sum-exp: `sum` is kept relative to the running
            // maximum log term and rescaled when a larger term appears, so no
            // exponential can overflow and the dominant term is exact.
            if term > maximum {
                sum = sum * (maximum - term).exp() + S::ONE;
                maximum = term;
            } else {
                sum += (term - maximum).exp();
            }
        }
        *log = maximum + sum.ln();
    }
    let max = logs.iter().copied().fold(S::NEG_INFINITY, |a, b| a.max(b));
    // Normalize so the largest weight magnitude is one (another common factor).
    // A weight that underflows to zero would remove its node from the
    // interpolant, so it is an error rather than a silent zero.
    // (-1)^(k-d) has the same parity as (-1)^(k+d).
    let mut weights = logs;
    for (k, w) in weights.iter_mut().enumerate() {
        let magnitude = (*w - max).exp();
        if magnitude == S::ZERO {
            return Err(instability(
                scioxide_core::NumericalIssue::Breakdown,
                "weight underflow",
            ));
        }
        *w = if (degree + k).is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        };
    }
    Ok(weights)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    #[test]
    fn weights_follow_the_closed_form_for_degree_zero_and_one() {
        let nodes = vec![0.0_f64, 1.0, 3.0];
        let berrut = FloaterHormannInterpolator::new(nodes.clone(), vec![0.0; 3], 0).unwrap();
        assert_eq!(berrut.weights(), &[1.0, -1.0, 1.0]);
        let linear = FloaterHormannInterpolator::new(nodes, vec![0.0; 3], 1).unwrap();
        // w = (-1)^(k-1) {1/1, 1/1 + 1/2, 1/2}, normalized by 3/2.
        let expected = [-2.0 / 3.0, 1.0, -1.0 / 3.0];
        for (actual, expected) in linear.weights().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-15);
        }
        assert_eq!(linear.degree(), 1);
    }

    #[test]
    fn invalid_degree_shape_and_nodes_are_rejected() {
        let kind = |result: Result<FloaterHormannInterpolator<f64>, SciOxideError>| {
            result.unwrap_err().kind()
        };
        assert_eq!(
            kind(FloaterHormannInterpolator::new(
                vec![0.0, 1.0],
                vec![1.0],
                0
            )),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(
            kind(FloaterHormannInterpolator::new(
                vec![0.0, 1.0],
                vec![1.0, 2.0],
                2
            )),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            kind(FloaterHormannInterpolator::new(
                vec![0.0, f64::NAN],
                vec![1.0, 2.0],
                1
            )),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn affine_components_match_scalar_interpolants_bit_for_bit() {
        let x = vec![-1.0_f32, -0.5, 0.1, 0.6, 1.0];
        let re: Vec<f32> = x.iter().map(|x| 1.0 / (1.0 + 25.0 * x * x)).collect();
        let im: Vec<f32> = x.iter().map(|x| x.exp()).collect();
        let values: Vec<Complex<f32>> = re
            .iter()
            .zip(&im)
            .map(|(&a, &b)| Complex::new(a, b))
            .collect();
        let complex = FloaterHormannInterpolator::new(x.clone(), values, 2).unwrap();
        let real = FloaterHormannInterpolator::new(x.clone(), re, 2).unwrap();
        let imaginary = FloaterHormannInterpolator::new(x, im, 2).unwrap();
        for query in [-0.9_f32, -0.5, 0.0, 0.33, 1.0] {
            let value = complex
                .evaluate([query], ExtrapolationPolicy::Reject)
                .unwrap();
            assert_eq!(
                value.re,
                real.evaluate([query], ExtrapolationPolicy::Reject).unwrap()
            );
            assert_eq!(
                value.im,
                imaginary
                    .evaluate([query], ExtrapolationPolicy::Reject)
                    .unwrap()
            );
        }
    }
}
