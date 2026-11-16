//! Local polynomial primitives and prefix integrals for splines.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{AffineValue, AffineValueMut, Interval, RealScalar, SciOxideError};

use crate::checked::{add, div, interval_coordinate, mul, output, sub};
use crate::protocol::{ExtrapolationPolicy, clamped_integral, find_interval, integral_bounds};
use crate::values::{QueryWorkspace, zeroed};
use scioxide_poly::backend::polynomial_basis::linear_multiply_difference;

/// Stores the antiderivative of a piecewise-polynomial spline for repeated
/// definite integrals.
///
/// Create it with [`crate::BSpline::prepare_antiderivative`] or
/// [`crate::CubicSpline::prepare_antiderivative`]; the source spline may then be
/// dropped. It stores the primitive of each polynomial piece in local
/// coordinates and the integrals from the first breakpoint to every other
/// breakpoint, `O(m (p + 1) c)` scalars for `m` pieces of degree `p` and `c`
/// real components. The integral over `[a, b]` is the difference of two
/// antiderivative values, at a cost of `O(log m + p)` per component.
///
/// The result is exact up to rounding. The conversion to local powers at high
/// degree, cancellation between prefix sums of similar size, and extrapolation
/// far outside the domain can lose accuracy, and no error bound is computed.
/// Each component equals the antiderivative of the scalar spline of that
/// component.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{CubicSpline, ExtrapolationPolicy, SplineBoundary};
/// let spline = CubicSpline::with_boundary(
///     vec![0.0_f64, 1.0, 2.0],
///     vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
///     SplineBoundary::Natural,
/// )?;
/// let prepared = spline.prepare_antiderivative()?;
/// let mut scratch = [0.0; 2];
/// let mut area = [0.0; 2];
/// prepared.integral_into(
///     0.0,
///     2.0,
///     ExtrapolationPolicy::Reject,
///     &mut area,
///     &mut scratch,
/// )?;
/// assert!((area[0] - 2.0).abs() < 1e-14 && (area[1] - 2.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedSplineAntiderivative<S: RealScalar, V: AffineValue<Scale = S> = S> {
    nodes: Vec<S>,
    /// `[piece][component][power]` primitive coefficients without constant term.
    primitives: Vec<S>,
    terms: usize,
    /// `[node][component]` integrals from the first node.
    prefixes: Vec<S>,
    /// `[end][component]` source values at the domain endpoints.
    endpoints: Vec<S>,
    dimension: usize,
    template: V,
    domain: Interval<S>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> PreparedSplineAntiderivative<S, V> {
    /// Stores primitives (`terms` per piece and component) and their prefix integrals.
    pub(crate) fn new(
        nodes: Vec<S>,
        primitives: Vec<S>,
        terms: usize,
        endpoints: Vec<S>,
        dimension: usize,
        template: V,
    ) -> Result<Self, SciOxideError> {
        let pieces = nodes.len() - 1;
        let mut prefixes = try_filled(
            checked_mul(
                checked_add(pieces, 1, scioxide_core::quantity!("prefixes"))?,
                dimension,
                scioxide_core::quantity!("prefixes"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("prefixes"),
        )?;
        // prefixes[i] = ∫ from nodes[0] to nodes[i], so any run of whole pieces
        // costs one subtraction.
        for (piece, coefficients) in primitives
            .chunks_exact((terms * dimension).max(1))
            .take(if dimension == 0 { 0 } else { pieces })
            .enumerate()
        {
            for (component, primitive) in coefficients.chunks_exact(terms).enumerate() {
                let (left, right) = (nodes[piece], nodes[piece + 1]);
                let whole = integrate_piece(primitive, left, right, left, right)?;
                let at = piece * dimension + component;
                prefixes[at + dimension] = add(prefixes[at], whole)?;
            }
        }
        let domain = Interval::new(nodes[0], nodes[pieces])?;
        Ok(Self {
            nodes,
            primitives,
            terms,
            prefixes,
            endpoints,
            dimension,
            template,
            domain,
        })
    }

    /// Returns the closed base domain of the source spline.
    pub const fn domain(&self) -> Interval<S> {
        self.domain
    }

    /// Returns the number of real components of one integral.
    pub const fn value_dimension(&self) -> usize {
        self.dimension
    }

    /// Writes the definite integral `∫_a^b f(x) dx` to `out`.
    ///
    /// `scratch` needs at least [`Self::value_dimension`] entries. Under
    /// [`ExtrapolationPolicy::Clamp`] the parts outside the domain integrate the
    /// constant endpoint values. Under continuation they integrate the boundary
    /// piece.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a bound is NaN or infinite, and `InvalidInput` if `a > b`.
    /// - The coordinate errors listed in the [crate docs](crate#errors).
    /// - `DimensionMismatch` if `out` has the wrong shape or `scratch` is too short.
    /// - `NonFinite` if the integral overflows.
    pub fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        a: S,
        b: S,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let available = scratch.len();
        let staged = scratch
            .get_mut(..self.dimension)
            .ok_or_else(|| crate::error::shape(self.dimension, available))?;
        integral_bounds(self.domain, a, b, policy)?;
        for (component, slot) in staged.iter_mut().enumerate() {
            *slot = if policy == ExtrapolationPolicy::Clamp {
                // Clamp samples the value only at the two domain endpoints.
                clamped_integral(
                    self.domain,
                    a,
                    b,
                    |x| {
                        let end = usize::from(x != self.nodes[0]);
                        Ok(self.endpoints[end * self.dimension + component])
                    },
                    |a, b| self.component(component, a, b),
                )?
            } else if a == b {
                S::ZERO
            } else {
                self.component(component, a, b)?
            };
        }
        scioxide_core::write_value(staged, out)
    }

    /// Returns the definite integral `∫_a^b f(x) dx` as an owned value.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::integral_into`].
    /// - `AllocationFailure` if the output or scratch cannot be reserved.
    pub fn integral(
        &self,
        a: S,
        b: S,
        policy: ExtrapolationPolicy<S>,
    ) -> Result<V::Owned, SciOxideError> {
        let mut out = zeroed(&self.template)?;
        let mut workspace = QueryWorkspace::new(self.dimension)?;
        self.integral_into(a, b, policy, &mut out, workspace.scratch(self.dimension)?)?;
        Ok(out)
    }

    /// Unclamped integral of one component.
    fn component(&self, component: usize, a: S, b: S) -> Result<S, SciOxideError> {
        let first = find_interval(&self.nodes, a);
        let last = find_interval(&self.nodes, b);
        if first == last {
            return self.piece(first, component, a, b);
        }
        let d = self.dimension;
        let middle = sub(
            self.prefixes[last * d + component],
            self.prefixes[(first + 1) * d + component],
        )?;
        output(add(
            add(
                self.piece(first, component, a, self.nodes[first + 1])?,
                middle,
            )?,
            self.piece(last, component, self.nodes[last], b)?,
        )?)
    }

    /// Integral over part of one piece.
    fn piece(&self, piece: usize, component: usize, a: S, b: S) -> Result<S, SciOxideError> {
        let start = (piece * self.dimension + component) * self.terms;
        integrate_piece(
            &self.primitives[start..start + self.terms],
            self.nodes[piece],
            self.nodes[piece + 1],
            a,
            b,
        )
    }
}

/// Convert local power coefficients to primitive coefficients (constant omitted).
pub(crate) fn primitive<S: RealScalar>(coefficients: &mut [S]) -> Result<(), SciOxideError> {
    for (i, c) in coefficients.iter_mut().enumerate() {
        *c = div(*c, S::try_from_usize_exact(i + 1)?)?;
    }
    Ok(())
}

/// Factored primitive difference over `[a, b]` of the piece on `[left, right]`:
/// the mean is multiplied by the physical width only at the end, which avoids
/// subtracting nearly equal primitive values and permits finite products when
/// the raw width is unrepresentable.
pub(crate) fn integrate_piece<S: RealScalar>(
    coefficients: &[S],
    left: S,
    right: S,
    a: S,
    b: S,
) -> Result<S, SciOxideError> {
    if a == b {
        return Ok(S::ZERO);
    }
    // With s ∈ [0, 1] and P(s) = s R(s), R(s) = Σ c_k s^k, the integral is
    // (b - a) (P(v) - P(u)) / (v - u). Horner for R at u runs with its divided
    // difference dd = (R(v) - R(u))/(v - u) (each step updates dd ← v dd + R(u));
    // the last line applies the identity once more for P = s R.
    let u = interval_coordinate(a, left, right)?;
    let v = interval_coordinate(b, left, right)?;
    let mut value = S::ZERO;
    let mut difference = S::ZERO;
    for &coefficient in coefficients.iter().rev() {
        difference = add(mul(difference, v)?, value)?;
        value = add(mul(value, u)?, coefficient)?;
    }
    let mean = add(mul(difference, v)?, value)?;
    output(linear_multiply_difference(mean, b, a)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_and_policies() {
        // Piece 0: 1 on [0, 1]; piece 1: 2s on [1, 3] (s the local coordinate).
        let prepared = PreparedSplineAntiderivative::new(
            vec![0.0_f64, 1.0, 3.0],
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            vec![1.0, 2.0],
            1,
            0.0,
        )
        .unwrap();
        let policy = ExtrapolationPolicy::Reject;
        assert_eq!(prepared.integral(0.0, 3.0, policy).unwrap(), 3.0);
        assert_eq!(prepared.integral(0.5, 0.5, policy).unwrap(), 0.0);
        assert_eq!(
            prepared
                .integral(-1.0, 4.0, ExtrapolationPolicy::Clamp)
                .unwrap(),
            6.0
        );
        assert!(prepared.integral(1.0, 0.0, policy).is_err());
        assert!(prepared.integral(f64::NAN, 0.0, policy).is_err());
        let mut out = 7.0;
        assert_eq!(
            prepared
                .integral_into(0.0, 1.0, policy, &mut out, &mut [])
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(out, 7.0);
    }
}
