//! Monotone piecewise cubic Hermite interpolation (PCHIP).

#![allow(clippy::many_single_char_names)]

use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{AffineValue, NonFiniteOrigin, RealScalar, SciOxideError};

use super::{PiecewiseCubic, componentwise};
use crate::checked::{div, mul};
use crate::protocol::SampleOptions;
use crate::validate::{validate_finite_origin, validate_nodes};
use crate::values::flatten;
use scioxide_poly::backend::polynomial_basis::linear_difference_ratio;

/// Interpolates samples by a monotone piecewise cubic Hermite polynomial (PCHIP).
///
/// On each interval `[xᵢ, xᵢ₊₁]` the interpolant is the cubic that matches the
/// values `yᵢ`, `yᵢ₊₁` and the node slopes `dᵢ`, `dᵢ₊₁`. With secants
/// `δᵢ = (yᵢ₊₁ − yᵢ)/hᵢ` and widths `hᵢ = xᵢ₊₁ − xᵢ`, an interior slope is zero
/// where `δᵢ₋₁` and `δᵢ` differ in sign or one of them is zero, and otherwise the
/// weighted harmonic mean
///
/// ```text
/// dᵢ = (w₁ + w₂) / (w₁/δᵢ₋₁ + w₂/δᵢ),   w₁ = 2hᵢ + hᵢ₋₁,   w₂ = hᵢ + 2hᵢ₋₁.
/// ```
///
/// End slopes use the one-sided three-point formula, set to zero if its sign
/// differs from the end secant and limited to three times the end secant when
/// the first two secants differ in sign. The result is C¹ and monotone on every
/// interval where the data are monotone. Second derivatives can jump at nodes.
/// Two samples give a straight line.
///
/// Slopes are computed separately for each real component of the values.
/// Construction costs O(n) per component. A value or derivative query locates
/// its interval by binary search in O(log n), and an integral costs
/// O(log n + k) for `k` crossed intervals.
///
/// # Errors
///
/// [`Self::with_options`] lists the construction errors. Queries fail as
/// described in the [crate docs](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     Differentiable, ExtrapolationPolicy, Integrable, Interpolant, PchipInterpolator,
/// };
/// let interp = PchipInterpolator::new(vec![0.0_f64, 1.0, 2.0, 3.0], vec![0.0, 2.0, 4.0, 6.0])?;
/// let policy = ExtrapolationPolicy::Reject;
/// assert!((interp.evaluate([1.5], policy)? - 3.0).abs() < 1e-14);
/// assert!((interp.gradient([1.5], policy)?[0] - 2.0).abs() < 1e-14);
/// assert!((interp.integral([0.0], [3.0], policy)? - 9.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - F. N. Fritsch and R. E. Carlson, "Monotone piecewise cubic interpolation",
///   *SIAM J. Numer. Anal.* 17(2), 1980, 238–246.
/// - F. N. Fritsch and J. Butland, "A method for constructing local monotone
///   piecewise cubic interpolants", *SIAM J. Sci. Stat. Comput.* 5(2), 1984,
///   300–304.
#[derive(Debug, Clone)]
#[must_use]
pub struct PchipInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    inner: PiecewiseCubic<S, V>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> PchipInterpolator<S, V> {
    /// Builds the interpolant through `(x[i], y[i])`.
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
    /// - `NonFinite` if a node or value is NaN or infinite, or a slope overflows.
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, SampleOptions::new())
    }

    /// Builds the interpolant through `(x[i], y[i])`, validating the nodes under
    /// `options`.
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
    /// - `NonFinite` if a node or value is NaN or infinite, or a slope overflows.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        validate_nodes(
            &x,
            2,
            options.node_separation(),
            scioxide_core::argument!("x"),
        )?;
        let flat = flatten(y, x.len())?;
        let slopes = componentwise(&x, &flat.components, flat.dimension, pchip_slopes)?;
        Ok(Self {
            inner: PiecewiseCubic::new(x, flat.components, slopes, flat.dimension, flat.template)?,
        })
    }

    /// Returns the first derivatives `dᵢ` at the nodes, node-major: the slopes of
    /// all components at `x[0]`, then at `x[1]`, and so on.
    pub fn slopes(&self) -> &[S] {
        self.inner.slopes()
    }
}

piecewise_protocol!(PchipInterpolator);

// ---------------------------------------------------------------------------
// Fritsch-Butland slope computation
// ---------------------------------------------------------------------------

/// Computes monotone node slopes by the Fritsch–Butland formula.
pub(crate) fn pchip_slopes<S: RealScalar>(x: &[S], y: &[S]) -> Result<Vec<S>, SciOxideError> {
    let n = x.len();
    let two = S::TWO;

    // Normalize only the two adjacent widths. A remote large node must not
    // underflow both local widths and erase their relative weights.
    let native = crate::checked::ordinary_nodes(x);
    let widths = |left: usize, right: usize| -> Result<(S, S), SciOxideError> {
        if native {
            return Ok((x[left + 1] - x[left], x[right + 1] - x[right]));
        }
        let width = |i: usize| {
            scioxide_core::Scaled::from_value(x[i + 1])?
                .sub(scioxide_core::Scaled::from_value(x[i])?)
        };
        let a = width(left)?;
        let b = width(right)?;
        let scale = if a.abs_le(b) { b } else { a };
        Ok((a.div(scale)?.round()?, b.div(scale)?.round()?))
    };
    let mut delta = try_with_capacity(n - 1, scioxide_core::quantity!("pchip_secant_slopes"))?;
    for i in 0..n - 1 {
        delta.push(linear_difference_ratio(y[i + 1], y[i], x[i + 1], x[i])?);
    }

    let mut d = try_filled(n, S::ZERO, scioxide_core::quantity!("pchip_node_slopes"))?;

    if n == 2 {
        d[0] = delta[0];
        d[1] = delta[0];
        return Ok(d);
    }

    let native_slopes = native && delta.iter().all(|&v| crate::checked::ordinary(v));

    // Interior points: harmonic mean where signs agree, zero otherwise.
    // Weighted harmonic mean (Fritsch–Butland / Brodlie):
    //   d_i = (w1 + w2) / (w1 / delta_{i-1} + w2 / delta_i),
    //   w1 = 2 h_i + h_{i-1},  w2 = h_i + 2 h_{i-1}.
    // A zero slope at a local extremum (sign change or a zero secant) is what
    // prevents overshoot; for same-sign secants the harmonic mean lies between
    // them and is bounded by 3 * min(|delta|), which satisfies the
    // Fritsch–Carlson monotonicity region.
    for i in 1..n - 1 {
        let left_delta = delta[i - 1];
        let right_delta = delta[i];
        if (left_delta > S::ZERO && right_delta > S::ZERO)
            || (left_delta < S::ZERO && right_delta < S::ZERO)
        {
            let (hp, hn) = widths(i - 1, i)?;
            if native_slopes {
                // Admitted widths and secants keep both quotients normal and
                // their same-sign sum finite. This is the original weighted
                // harmonic mean, without its extreme-range rescalings.
                let w1 = two * hn + hp;
                let w2 = hn + two * hp;
                d[i] = (w1 + w2) / (w1 / left_delta + w2 / right_delta);
                continue;
            }
            let width_scale = hp.max(hn);
            let hp = hp / width_scale;
            let hn = hn / width_scale;
            let w1 = two * hn + hp;
            let w2 = hn + two * hp;
            // Scale by the smaller secant: every ratio is bounded by one.
            // Algebraically d = small / sum_k (w_k / W) (small / delta_k), which
            // is the formula above rearranged so no reciprocal of a tiny secant
            // (which could overflow) is ever formed.
            let small = left_delta.abs().min(right_delta.abs());
            let denominator =
                (w1 / (w1 + w2)) * (small / left_delta) + (w2 / (w1 + w2)) * (small / right_delta);
            d[i] = div(small, denominator)?;
        }
        // else d[i] = 0 (already initialized)
    }

    // Limited one-sided three-point endpoint slopes (n >= 3 here).
    let (h1, h2) = widths(0, 1)?;
    d[0] = endpoint_slope(delta[0], delta[1], h1, h2)?;
    let (h1, h2) = widths(n - 2, n - 3)?;
    d[n - 1] = endpoint_slope(delta[n - 2], delta[n - 3], h1, h2)?;

    validate_finite_origin(&d, NonFiniteOrigin::Intermediate)?;
    Ok(d)
}

/// Computes an end slope by the limited one-sided three-point formula.
fn endpoint_slope<S: RealScalar>(d1: S, d2: S, h1: S, h2: S) -> Result<S, SciOxideError> {
    // Non-centred three-point formula (as in Moler's `pchip`):
    //   d = ((2 h1 + h2) d1 - h1 d2) / (h1 + h2) = d1 + h1/(h1 + h2) * (d1 - d2),
    // evaluated with widths and secants normalized by their maxima so neither
    // the weight nor the extrapolated secant difference can overflow.
    // `d1 == 0` also guarantees `scale != 0` below.
    if d1 == S::ZERO {
        return Ok(S::ZERO);
    }
    if [d1, d2, h1, h2].into_iter().all(crate::checked::ordinary) {
        let proposal = d1 + (h1 / (h1 + h2)) * (d1 - d2);
        if (proposal > S::ZERO) != (d1 > S::ZERO) {
            return Ok(S::ZERO);
        }
        let three = S::TWO + S::ONE;
        return Ok(
            if (d1 > S::ZERO) != (d2 > S::ZERO) && proposal.abs() > three * d1.abs() {
                three * d1
            } else {
                proposal
            },
        );
    }
    let hs = h1.max(h2);
    let weight = div(h1 / hs, h1 / hs + h2 / hs)?;
    let scale = d1.abs().max(d2.abs());
    let first = d1 / scale;
    let second = d2 / scale;
    let proposal = first + weight * (first - second);
    // Shape limiters: a slope opposite to the end secant would create an
    // extremum inside the end interval, so use zero; when the secants change
    // sign (data turns at the second node) cap |d| at 3|d1|, the
    // Fritsch–Carlson bound that keeps the end piece monotone.
    if (proposal > S::ZERO) != (d1 > S::ZERO) {
        return Ok(S::ZERO);
    }
    let three = S::TWO + S::ONE;
    if (d1 > S::ZERO) != (d2 > S::ZERO) && proposal.abs() / three > first.abs() {
        mul(three, d1)
    } else {
        mul(proposal, scale)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocol::{Differentiable, ExtrapolationPolicy, Interpolant};

    #[test]
    fn rejects_too_few_points_and_unrepresentable_slopes() {
        assert!(PchipInterpolator::new(vec![0.0], vec![1.0]).is_err());
        let result =
            PchipInterpolator::new(vec![0.0, f64::MIN_POSITIVE], vec![-f64::MAX, f64::MAX]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn monotone_data_stays_monotone() {
        let interp =
            PchipInterpolator::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 2.0, 3.0, 4.0])
                .unwrap();
        let policy = ExtrapolationPolicy::Reject;
        let mut previous = interp.evaluate([0.0], policy).unwrap();
        for k in 1..=40 {
            let value = interp.evaluate([f64::from(k) / 10.0], policy).unwrap();
            assert!(value >= previous);
            previous = value;
        }
        assert!((interp.gradient([1.5], policy).unwrap()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn small_nonzero_secants_retain_linear_slopes() {
        let tiny = f64::EPSILON / 2.0;
        let interp =
            PchipInterpolator::new(vec![0.0, 1.0, 2.0], vec![0.0, tiny, 2.0 * tiny]).unwrap();
        assert!(
            interp
                .slopes()
                .iter()
                .all(|s| (*s - tiny).abs() <= tiny * 1e-14)
        );
    }

    #[test]
    fn vector_slopes_are_componentwise() {
        let x = vec![0.0_f32, 1.0, 3.0, 4.0];
        let vector = PchipInterpolator::new(
            x.clone(),
            vec![[0.0, 1.0], [2.0, -1.0], [3.0, 5.0], [3.0, 0.0]],
        )
        .unwrap();
        let first = PchipInterpolator::new(x.clone(), vec![0.0, 2.0, 3.0, 3.0]).unwrap();
        let second = PchipInterpolator::new(x, vec![1.0, -1.0, 5.0, 0.0]).unwrap();
        for (i, pair) in vector.slopes().chunks_exact(2).enumerate() {
            assert_eq!(pair[0], first.slopes()[i]);
            assert_eq!(pair[1], second.slopes()[i]);
        }
    }
}
