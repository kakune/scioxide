//! Akima and modified Akima piecewise cubic interpolation.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_with_capacity};
use scioxide_core::{AffineValue, RealScalar, SciOxideError};

use super::{PiecewiseCubic, componentwise};
use crate::checked::output;
use crate::protocol::SampleOptions;
use crate::validate::validate_nodes;
use crate::values::flatten;
use scioxide_poly::backend::polynomial_basis::linear_difference_ratio;

/// Selects the slope weights of an [`AkimaInterpolator`].
///
/// With secants `mⱼ = (yⱼ₊₁ − yⱼ)/(xⱼ₊₁ − xⱼ)`, the slope at node `i` is
///
/// ```text
/// tᵢ = (w₁ mᵢ₋₁ + w₂ mᵢ) / (w₁ + w₂),
/// ```
///
/// where the weights `w₁` and `w₂` measure how far the secants on the opposite
/// side of the node are from a straight line. The two variants differ only in
/// these weights. The modified weights give smaller slopes where the data have
/// a flat stretch, as in the example:
///
/// ```
/// use scioxide_interp::{AkimaInterpolator, AkimaMethod};
/// let (x, y) = (
///     vec![0.0_f64, 1.0, 2.0, 3.0, 4.0],
///     vec![0.0, 1.0, 2.0, 5.0, 8.0],
/// );
/// let akima = AkimaInterpolator::new(x.clone(), y.clone())?;
/// let makima = AkimaInterpolator::with_method(x, y, AkimaMethod::Makima)?;
/// assert_eq!(akima.slopes()[2], 2.0);
/// assert!(makima.slopes()[2] < 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AkimaMethod {
    /// Uses `w₁ = |mᵢ₊₁ − mᵢ|` and `w₂ = |mᵢ₋₁ − mᵢ₋₂|` (Akima, 1970).
    Akima,
    /// Uses `w₁ = |mᵢ₊₁ − mᵢ| + |mᵢ₊₁ + mᵢ|/2` and
    /// `w₂ = |mᵢ₋₁ − mᵢ₋₂| + |mᵢ₋₁ + mᵢ₋₂|/2` (modified Akima).
    Makima,
}

/// Interpolates samples by a C¹ piecewise cubic with Akima or modified Akima
/// slopes.
///
/// Each interval carries the cubic Hermite polynomial that matches the values
/// and the node slopes chosen by [`AkimaMethod`]. Each slope depends only on
/// the two secants on either side of its node, so the interpolant is local. At
/// each end, two extra secants continue the secant sequence linearly,
/// `m₋₁ = 2m₀ − m₁` and `m₋₂ = 2m₋₁ − m₀`. If both weights at a node are zero,
/// the slope is the mean of the two adjacent secants. Two samples give a
/// straight line.
///
/// The first derivative at a node equals [`Self::slopes`]. Second derivatives
/// can jump at nodes. Neither variant preserves monotonicity; use
/// [`PchipInterpolator`](crate::PchipInterpolator) for monotone data.
///
/// Slopes are computed separately for each real component of the values.
/// Construction costs O(n) time and memory per component. Value and derivative
/// queries cost O(log n), and an integral costs O(log n + k) for `k` crossed
/// intervals. Very unequal node spacing amplifies errors in the samples.
///
/// # Errors
///
/// [`Self::with_options`] lists the construction errors. Queries fail as
/// described in the [crate docs](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_interp::{AkimaInterpolator, AkimaMethod, ExtrapolationPolicy, Interpolant};
/// let x = vec![0.0_f64, 1.0, 3.0, 4.0, 7.0];
/// let y = vec![0.0, 2.0, 3.0, 3.0, 8.0];
/// let akima = AkimaInterpolator::with_method(x, y, AkimaMethod::Makima)?;
/// assert_eq!(akima.evaluate([3.0], ExtrapolationPolicy::Reject)?, 3.0);
/// assert!((akima.slopes()[0] - 169.0 / 68.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - H. Akima, "A new method of interpolation and smooth curve fitting based on
///   local procedures", *J. ACM* 17(4), 1970, 589–602.
/// - C. Moler, "Makima piecewise cubic interpolation", Cleve's Corner, 2019.
#[derive(Debug, Clone)]
#[must_use]
pub struct AkimaInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    inner: PiecewiseCubic<S, V>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> AkimaInterpolator<S, V> {
    /// Builds the interpolant through `(x[i], y[i])` with Akima's slopes.
    ///
    /// This is [`Self::with_options`] with [`AkimaMethod::Akima`] and
    /// [`SampleOptions::new`], so the nodes `x` must be strictly increasing. All
    /// values `y` must have the same shape.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes or the nodes are not
    ///   strictly increasing.
    /// - `DimensionMismatch` if `x` and `y` differ in length or the values differ
    ///   in shape.
    /// - `NonFinite` if a node or value is NaN or infinite, or a secant or slope
    ///   overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{AkimaInterpolator, Univariate};
    /// let x = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    /// let akima = AkimaInterpolator::new(x, vec![0.0, 1.0, 2.0, 5.0, 8.0])?;
    /// // Both weights at x = 2 vanish: the slope is the mean of the secants 1 and 3.
    /// assert!((akima.derivative_at(2.0)? - 2.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_method(x, y, AkimaMethod::Akima)
    }

    /// Builds the interpolant through `(x[i], y[i])` with the slopes of `method`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// `x` must be strictly increasing. All values `y` must have the same shape.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes or the nodes are not
    ///   strictly increasing.
    /// - `DimensionMismatch` if `x` and `y` differ in length or the values differ
    ///   in shape.
    /// - `NonFinite` if a node or value is NaN or infinite, or a secant or slope
    ///   overflows.
    pub fn with_method(x: Vec<S>, y: Vec<V>, method: AkimaMethod) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, method, SampleOptions::new())
    }

    /// Builds the interpolant through `(x[i], y[i])` with the slopes of `method`,
    /// validating the nodes under `options`.
    ///
    /// The nodes `x` must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`]. All values `y` must have the same
    /// shape.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes, or the nodes are not
    ///   strictly increasing and distinct under the separation of `options`.
    /// - `DimensionMismatch` if `x` and `y` differ in length or the values differ
    ///   in shape.
    /// - `NonFinite` if a node or value is NaN or infinite, or a secant or slope
    ///   overflows.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        method: AkimaMethod,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        validate_nodes(
            &x,
            2,
            options.node_separation(),
            scioxide_core::argument!("x"),
        )?;
        let flat = flatten(y, x.len())?;
        let slopes = componentwise(&x, &flat.components, flat.dimension, |x, y| {
            akima_slopes(x, y, method)
        })?;
        Ok(Self {
            inner: PiecewiseCubic::new(x, flat.components, slopes, flat.dimension, flat.template)?,
        })
    }

    /// Returns the first derivatives at the nodes, node-major: the slopes of all
    /// components at `x[0]`, then at `x[1]`, and so on.
    pub fn slopes(&self) -> &[S] {
        self.inner.slopes()
    }
}

piecewise_protocol!(AkimaInterpolator);

/// Computes the Akima or modified Akima node slopes of one component.
fn akima_slopes<S: RealScalar>(
    x: &[S],
    y: &[S],
    method: AkimaMethod,
) -> Result<Vec<S>, SciOxideError> {
    let n = x.len();
    // Secant layout: two leading virtual slots (filled after the real secants
    // are known), the n - 1 real secants at indices 2..=n, and two trailing
    // virtual secants, n + 3 entries in total. Every node i then sees the
    // window [m_{i-2}, m_{i-1}, m_i, m_{i+1}] = secants[i..i + 4], so
    // `windows(4)` yields exactly one window per node.
    let mut secants = try_with_capacity(
        checked_add(n, 3, scioxide_core::quantity!("akima_secants"))?,
        scioxide_core::quantity!("akima_secants"),
    )?;
    secants.extend([S::ZERO; 2]);
    for (xs, ys) in x.windows(2).zip(y.windows(2)) {
        secants.push(linear_difference_ratio(ys[1], ys[0], xs[1], xs[0])?);
    }
    let mut slopes = try_with_capacity(n, scioxide_core::quantity!("akima_slopes"))?;
    if n == 2 {
        slopes.extend([secants[2]; 2]);
        return Ok(slopes);
    }
    // Akima's endpoint rule: continue the secant sequence linearly,
    // m_virtual = 2*a - b, where `a` is the neighbour secant and `b` the next
    // one inwards. Each pair of virtual secants is built outwards in turn.
    let extend = |a: S, b: S| {
        // Scaling avoids an overflowing 2*a when 2*a-b is representable.
        let scale = a.abs().max(b.abs());
        if scale == S::ZERO {
            return Ok(S::ZERO);
        }
        output(((a / scale - b / scale) + a / scale) * scale)
    };
    secants[1] = extend(secants[2], secants[3])?;
    secants[0] = extend(secants[1], secants[2])?;
    secants.push(extend(secants[n], secants[n - 1])?);
    secants.push(extend(secants[n + 1], secants[n])?);
    let two = S::TWO;
    // Akima slope at node i with local = [m_{i-2}, m_{i-1}, m_i, m_{i+1}]:
    //   t_i = (w_left * m_{i-1} + w_right * m_i) / (w_left + w_right),
    //   w_left = |m_{i+1} - m_i|, w_right = |m_{i-1} - m_{i-2}|,
    // i.e. each adjacent secant is weighted by how "flat" the opposite side is.
    // Makima adds |m_j + m_k| / 2 to each weight, which removes the
    // 0/0 case for equal-but-nonzero neighbouring secants and damps overshoot.
    // Weights are only ratios, so the window is normalized by its largest
    // magnitude first to keep the differences from overflowing.
    let native = secants.iter().all(|&v| crate::checked::ordinary(v));
    for local in secants.windows(4) {
        if native {
            let weight = |a: S, b: S| {
                (a - b).abs()
                    + if method == AkimaMethod::Makima {
                        (a + b).abs() / two
                    } else {
                        S::ZERO
                    }
            };
            let left = weight(local[2], local[3]);
            let right = weight(local[0], local[1]);
            let sum = left + right;
            let value = if local[1] == local[2] {
                local[1]
            } else if sum == S::ZERO {
                local[1] / two + local[2] / two
            } else {
                (left / sum) * local[1] + (right / sum) * local[2]
            };
            slopes.push(output(value)?);
            continue;
        }
        let scale = local.iter().fold(S::ZERO, |m, v| m.max(v.abs()));
        if scale == S::ZERO {
            slopes.push(S::ZERO);
            continue;
        }
        let a = local[0] / scale;
        let b = local[1] / scale;
        let c = local[2] / scale;
        let d = local[3] / scale;
        let weight = |a: S, b: S| {
            (a - b).abs()
                + if method == AkimaMethod::Makima {
                    (a + b).abs() / two
                } else {
                    S::ZERO
                }
        };
        let left = weight(c, d);
        let right = weight(a, b);
        let sum = left + right;
        // Keep the adjacent secants in their original scale. Scaling the
        // result through remote secants would erase small, representable slopes.
        // Equal adjacent secants make any convex combination equal to them, so
        // return that value exactly (this keeps linear data reproduced without
        // rounding). A zero weight sum means both sides are locally linear and
        // Akima's convention is the plain average.
        let value = if local[1] == local[2] {
            local[1]
        } else if sum == S::ZERO {
            local[1] / two + local[2] / two
        } else {
            (left / sum) * local[1] + (right / sum) * local[2]
        };
        slopes.push(output(value)?);
    }
    Ok(slopes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocol::{ExtrapolationPolicy, Interpolant};
    use scioxide_core::Complex;

    #[test]
    fn complex_values_are_componentwise() {
        let x = vec![0.0_f64, 1.0, 3.0, 4.0, 7.0];
        let re = [0.0, 2.0, 3.0, 3.0, 8.0];
        let im = [1.0, -1.0, 0.5, 2.0, 2.0];
        let values = re
            .iter()
            .zip(im)
            .map(|(&a, b)| Complex::new(a, b))
            .collect();
        let complex = AkimaInterpolator::new(x.clone(), values).unwrap();
        let real = AkimaInterpolator::new(x.clone(), re.to_vec()).unwrap();
        let imaginary = AkimaInterpolator::new(x, im.to_vec()).unwrap();
        for t in [0.0, 0.5, 2.25, 6.9] {
            let value = complex.evaluate([t], ExtrapolationPolicy::Reject).unwrap();
            assert_eq!(
                value.re,
                real.evaluate([t], ExtrapolationPolicy::Reject).unwrap()
            );
            assert_eq!(
                value.im,
                imaginary
                    .evaluate([t], ExtrapolationPolicy::Reject)
                    .unwrap()
            );
        }
    }

    #[test]
    fn two_points_give_a_line() {
        let line =
            AkimaInterpolator::with_method(vec![0.0, 2.0], vec![1.0, 5.0], AkimaMethod::Makima)
                .unwrap();
        assert_eq!(line.slopes(), [2.0, 2.0]);
    }
}
