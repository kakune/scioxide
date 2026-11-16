//! Piecewise cubic Hermite interpolation shared by the Akima, PCHIP and
//! cubic-spline interpolants.
//!
//! Each interpolant computes node slopes and passes nodes, values and slopes to
//! [`PiecewiseCubic`]. Values and slopes are stored node-major. Components
//! whose values and slopes are far from the overflow threshold also keep
//! unit-interval power coefficients and are evaluated by Horner's rule; other
//! components use the scaled Hermite form. Every component uses the same
//! scalar kernel, so a vector result equals per-component scalar evaluation.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError};

use crate::checked::{add, interval_coordinate, mul, output, sub};
use crate::hermite_basis;
use crate::protocol::{
    ExtrapolationPolicy, clamped_integral, domain_from_nodes, find_interval, integral_bounds,
    resolve_coordinate,
};
use crate::spline::{PreparedSplineAntiderivative, primitive};
use crate::values::{QueryWorkspace, gather, scatter, zeroed};
use scioxide_poly::backend::polynomial_basis::{
    linear_difference_ratio, linear_multiply_difference,
};

/// Implement the value, derivative and integral protocol for a family that
/// wraps a [`PiecewiseCubic`] in its `inner` field.
macro_rules! piecewise_protocol {
    ($family:ident) => {
        impl<S: scioxide_core::RealScalar, V: scioxide_core::AffineValue<Scale = S> + Sync>
            crate::protocol::Interpolant<S, 1> for $family<S, V>
        {
            type Value = V::Owned;
            type Workspace = crate::values::QueryWorkspace<S>;

            fn domains(&self) -> [scioxide_core::Interval<S>; 1] {
                [self.inner.domain()]
            }
            fn value_dimension(&self) -> usize {
                self.inner.dimension()
            }
            fn zero_value(&self) -> Result<V::Owned, scioxide_core::SciOxideError> {
                self.inner.zero_value()
            }
            fn workspace(
                &self,
            ) -> Result<crate::values::QueryWorkspace<S>, scioxide_core::SciOxideError> {
                self.inner.workspace()
            }
            fn workspace_capacity(
                &self,
                workspace: &crate::values::QueryWorkspace<S>,
            ) -> scioxide_core::Capacity {
                self.inner.capacity(workspace)
            }
            fn evaluate_into<O: scioxide_core::AffineValueMut<Scale = S> + ?Sized>(
                &self,
                point: [S; 1],
                policy: crate::protocol::ExtrapolationPolicy<S>,
                out: &mut O,
                workspace: &mut crate::values::QueryWorkspace<S>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                self.inner.evaluate_into(point[0], policy, out, workspace)
            }
        }

        impl<S: scioxide_core::RealScalar, V: scioxide_core::AffineValue<Scale = S> + Sync>
            crate::protocol::Differentiable<S, 1> for $family<S, V>
        {
            fn gradient_into<O: scioxide_core::AffineValueMut<Scale = S> + ?Sized>(
                &self,
                point: [S; 1],
                policy: crate::protocol::ExtrapolationPolicy<S>,
                out: &mut O,
                workspace: &mut crate::values::QueryWorkspace<S>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                self.inner.derivative_into(point[0], policy, out, workspace)
            }
        }

        impl<S: scioxide_core::RealScalar, V: scioxide_core::AffineValue<Scale = S> + Sync>
            crate::protocol::Integrable<S, 1> for $family<S, V>
        {
            fn integral_into<O: scioxide_core::AffineValueMut<Scale = S> + ?Sized>(
                &self,
                lower: [S; 1],
                upper: [S; 1],
                policy: crate::protocol::ExtrapolationPolicy<S>,
                out: &mut O,
                workspace: &mut crate::values::QueryWorkspace<S>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                self.inner
                    .integral_into(lower[0], upper[0], policy, out, workspace)
            }
        }
    };
}

mod akima;
mod cubic;
mod pchip;

pub use akima::{AkimaInterpolator, AkimaMethod};
pub use cubic::{CubicSpline, SplineBoundary};
pub use pchip::PchipInterpolator;
pub(crate) use pchip::pchip_slopes;

/// Immutable C1 piecewise cubic Hermite interpolant with affine values.
#[derive(Debug, Clone)]
pub(crate) struct PiecewiseCubic<S: RealScalar, V: AffineValue<Scale = S>> {
    x: Vec<S>,
    /// `[node][component]` values.
    values: Vec<S>,
    /// `[node][component]` physical first derivatives.
    slopes: Vec<S>,
    dimension: usize,
    template: V,
    domain: Interval<S>,
    /// Unit-interval power coefficients; read only for components evaluated by Horner's rule.
    powers: Vec<[S; 4]>,
    native_components: Vec<bool>,
    native_coordinates: bool,
}

/// Compute node slopes one value component at a time: gather the component,
/// run the scalar rule `slopes(x, column)`, and scatter the result.
pub(crate) fn componentwise<S: RealScalar>(
    x: &[S],
    values: &[S],
    dimension: usize,
    mut slopes: impl FnMut(&[S], &[S]) -> Result<Vec<S>, SciOxideError>,
) -> Result<Vec<S>, SciOxideError> {
    if dimension == 1 {
        return slopes(x, values);
    }
    let mut column = try_filled(x.len(), S::ZERO, scioxide_core::quantity!("value_column"))?;
    let mut result = try_filled(
        checked_mul(x.len(), dimension, scioxide_core::quantity!("node_slopes"))?,
        S::ZERO,
        scioxide_core::quantity!("node_slopes"),
    )?;
    for component in 0..dimension {
        gather(values, dimension, component, &mut column);
        scatter(&slopes(x, &column)?, dimension, component, &mut result);
    }
    Ok(result)
}

impl<S: RealScalar, V: AffineValue<Scale = S>> PiecewiseCubic<S, V> {
    /// Builds the interpolant from validated finite nodes and node-major values
    /// and slopes.
    pub(crate) fn new(
        x: Vec<S>,
        values: Vec<S>,
        slopes: Vec<S>,
        dimension: usize,
        template: V,
    ) -> Result<Self, SciOxideError> {
        let expected = checked_mul(x.len(), dimension, scioxide_core::quantity!("node_values"))?;
        if values.len() != expected || slopes.len() != expected {
            return Err(SciOxideError::dimension_mismatch(
                [expected, expected],
                [values.len(), slopes.len()],
            ));
        }
        let domain = domain_from_nodes(&x)?;
        let ordinary = crate::checked::ordinary_nodes(&x);
        let mut native_components =
            try_with_capacity(dimension, scioxide_core::quantity!("piecewise_ranges"))?;
        for c in 0..dimension {
            native_components.push(
                ordinary
                    && values
                        .iter()
                        .skip(c)
                        .step_by(dimension)
                        .chain(slopes.iter().skip(c).step_by(dimension))
                        .all(|&v| crate::checked::ordinary(v)),
            );
        }
        let prepared = native_components.iter().any(|&v| v);
        let mut powers = try_with_capacity(
            if prepared {
                (x.len() - 1) * dimension
            } else {
                0
            },
            scioxide_core::quantity!("piecewise_coefficients"),
        )?;
        if prepared {
            for i in 0..x.len() - 1 {
                let h = x[i + 1] - x[i];
                for (c, &native) in native_components.iter().enumerate() {
                    let k = i * dimension + c;
                    let next = k + dimension;
                    let coefficients = if native {
                        // With each operand bounded by sqrt(MAX)/64, every
                        // product is <= MAX/4096 and the four-term Horner
                        // envelope is below MAX/16, so the checked transform
                        // cannot fail here.
                        hermite_basis::power([
                            values[k],
                            values[next] - values[k],
                            h * slopes[k],
                            h * slopes[next],
                        ])?
                    } else {
                        [S::ZERO; 4]
                    };
                    powers.push(coefficients);
                }
            }
        }
        Ok(Self {
            native_coordinates: ordinary,
            native_components,
            powers,
            x,
            values,
            slopes,
            dimension,
            template,
            domain,
        })
    }

    pub(crate) const fn domain(&self) -> Interval<S> {
        self.domain
    }

    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn slopes(&self) -> &[S] {
        &self.slopes
    }

    pub(crate) fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }

    pub(crate) fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.dimension)
    }

    pub(crate) const fn capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.dimension)
    }

    /// `(x0, x1, y0, y1, d0, d1)` of piece `i`, component `c`.
    fn piece(&self, i: usize, c: usize) -> [S; 6] {
        let d = self.dimension;
        [
            self.x[i],
            self.x[i + 1],
            self.values[i * d + c],
            self.values[(i + 1) * d + c],
            self.slopes[i * d + c],
            self.slopes[(i + 1) * d + c],
        ]
    }

    fn value(&self, t: S, c: usize) -> Result<S, SciOxideError> {
        let i = find_interval(&self.x, t);
        self.value_at(i, c, t, self.unit_coordinate(i, t))
    }

    #[inline]
    fn unit_coordinate(&self, i: usize, t: S) -> Option<S> {
        (self.native_coordinates && t >= self.x[i] && t <= self.x[i + 1])
            .then(|| (t - self.x[i]) / (self.x[i + 1] - self.x[i]))
    }

    #[inline]
    fn value_at(&self, i: usize, c: usize, t: S, unit: Option<S>) -> Result<S, SciOxideError> {
        if let Some(u) = unit
            && self.native_components[c]
        {
            if t == self.x[i] {
                return Ok(self.values[i * self.dimension + c]);
            }
            if t == self.x[i + 1] {
                return Ok(self.values[(i + 1) * self.dimension + c]);
            }
            let a = self.powers[i * self.dimension + c];
            // Horner reproduces nonzero constants exactly. Zero constants need
            // a shortcut for their stored sign; test the value first to avoid
            // unpredictable branches on shape-preserving methods' zero slopes.
            if a[0] == S::ZERO && a[1] == S::ZERO && a[2] == S::ZERO && a[3] == S::ZERO {
                return Ok(a[0]);
            }
            return output(((a[3] * u + a[2]) * u + a[1]) * u + a[0]);
        }
        let [x0, x1, y0, y1, d0, d1] = self.piece(i, c);
        output(cubic_hermite_eval(x0, x1, y0, y1, d0, d1, t)?)
    }

    fn integral(&self, a: S, b: S, c: usize) -> Result<S, SciOxideError> {
        // Bounds outside the domain (continuation only) use the outer pieces.
        let first = find_interval(&self.x, a);
        let last = find_interval(&self.x, b);
        let mut total = S::ZERO;
        for i in first..=last {
            let lo = if i == first { a } else { self.x[i] };
            let hi = if i == last { b } else { self.x[i + 1] };
            let [x0, x1, y0, y1, d0, d1] = self.piece(i, c);
            let piece = if self.native_components[c] && lo >= x0 && hi <= x1 {
                let a = self.powers[i * self.dimension + c];
                let u = (lo - x0) / (x1 - x0);
                let v = (hi - x0) / (x1 - x0);
                let three = S::TWO + S::ONE;
                let four = S::TWO + S::TWO;
                // Factored differences of primitive powers preserve short
                // intervals without subtracting nearly equal primitives.
                let mean = a[0]
                    + a[1] * ((u + v) / S::TWO)
                    + a[2] * ((u * u + u * v + v * v) / three)
                    + a[3] * ((u * u * u + u * u * v + u * v * v + v * v * v) / four);
                let value = mean * (hi - lo);
                if value.is_finite() {
                    value
                } else {
                    cubic_hermite_integral(x0, x1, y0, y1, d0, d1, lo, hi)?
                }
            } else {
                cubic_hermite_integral(x0, x1, y0, y1, d0, d1, lo, hi)?
            };
            total = add(total, piece)?;
        }
        output(total)
    }

    pub(crate) fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        t: S,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        let t = resolve_coordinate(&self.domain, t, policy)?;
        let i = find_interval(&self.x, t);
        let unit = self.unit_coordinate(i, t);
        if self.dimension == 1 {
            // A single finite value is its complete transaction; no staging copy.
            let value = self.value_at(i, 0, t, unit)?;
            out.for_each_component_mut(|slot| *slot = value);
            return Ok(());
        }
        for (c, slot) in staged.iter_mut().enumerate() {
            *slot = self.value_at(i, c, t, unit)?;
        }
        crate::values::commit_finite(staged, out);
        Ok(())
    }

    pub(crate) fn derivative_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        t: S,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        let clamped = policy == ExtrapolationPolicy::Clamp && !self.domain.contains(t);
        // Resolve even when clamped so non-finite coordinates are rejected.
        let t = resolve_coordinate(&self.domain, t, policy)?;
        let i = find_interval(&self.x, t);
        let unit = self.unit_coordinate(i, t);
        let calculate = |c| {
            if clamped {
                Ok(S::ZERO)
            } else {
                self.derivative_at(i, c, t, unit)
            }
        };
        if self.dimension == 1 {
            let value = calculate(0)?;
            out.for_each_component_mut(|slot| *slot = value);
            return Ok(());
        }
        for (c, slot) in staged.iter_mut().enumerate() {
            *slot = calculate(c)?;
        }
        crate::values::commit_finite(staged, out);
        Ok(())
    }

    fn derivative_at(&self, i: usize, c: usize, t: S, unit: Option<S>) -> Result<S, SciOxideError> {
        let [x0, x1, y0, y1, d0, d1] = self.piece(i, c);
        if t == x0 {
            return Ok(d0);
        }
        if t == x1 {
            return Ok(d1);
        }
        if y0 == y1 && d0 == S::ZERO && d1 == S::ZERO {
            return Ok(S::ZERO);
        }
        if let Some(u) = unit
            && self.native_components[c]
        {
            let a = self.powers[i * self.dimension + c];
            let three = S::TWO + S::ONE;
            let result = ((three * a[3] * u + S::TWO * a[2]) * u + a[1]) / (x1 - x0);
            if result.is_finite() {
                return Ok(result);
            }
        }
        output(cubic_hermite_deriv(x0, x1, y0, y1, d0, d1, t)?)
    }

    pub(crate) fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        a: S,
        b: S,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        integral_bounds(self.domain, a, b, policy)?;
        let calculate = |c| {
            if policy == ExtrapolationPolicy::Clamp {
                clamped_integral(
                    self.domain,
                    a,
                    b,
                    |x| self.value(x, c),
                    |a, b| self.integral(a, b, c),
                )
            } else {
                self.integral(a, b, c)
            }
        };
        if self.dimension == 1 {
            let value = calculate(0)?;
            out.for_each_component_mut(|slot| *slot = value);
            return Ok(());
        }
        for (c, slot) in staged.iter_mut().enumerate() {
            *slot = calculate(c)?;
        }
        crate::values::commit_finite(staged, out);
        Ok(())
    }

    /// Local primitives of every piece for a [`PreparedSplineAntiderivative`].
    pub(crate) fn prepare_antiderivative(
        &self,
    ) -> Result<PreparedSplineAntiderivative<S, V::Owned>, SciOxideError> {
        let pieces = self.x.len() - 1;
        let mut nodes = try_with_capacity(self.x.len(), scioxide_core::quantity!("nodes"))?;
        nodes.extend_from_slice(&self.x);
        let mut primitives = try_with_capacity(
            checked_mul(
                checked_mul(pieces, 4, scioxide_core::quantity!("primitives"))?,
                self.dimension,
                scioxide_core::quantity!("primitives"),
            )?,
            scioxide_core::quantity!("primitives"),
        )?;
        for i in 0..pieces {
            for c in 0..self.dimension {
                // Power form on s ∈ [0, 1] with H_k = h d_k.
                let [x0, x1, y0, y1, d0, d1] = self.piece(i, c);
                let mut coefficients = hermite_basis::power([
                    y0,
                    sub(y1, y0)?,
                    linear_multiply_difference(d0, x1, x0)?,
                    linear_multiply_difference(d1, x1, x0)?,
                ])?;
                primitive(&mut coefficients)?;
                primitives.extend_from_slice(&coefficients);
            }
        }
        let d = self.dimension;
        let mut endpoints = try_with_capacity(
            checked_mul(2, d, scioxide_core::quantity!("endpoints"))?,
            scioxide_core::quantity!("endpoints"),
        )?;
        endpoints.extend_from_slice(&self.values[..d]);
        endpoints.extend_from_slice(&self.values[pieces * d..]);
        PreparedSplineAntiderivative::new(
            nodes,
            primitives,
            4,
            endpoints,
            d,
            zeroed(&self.template)?,
        )
    }
}

/// Cubic Hermite interpolant on `[x0, x1]` at `t`.
#[cold]
#[inline(never)]
fn cubic_hermite_eval<S: RealScalar>(
    x0: S,
    x1: S,
    y0: S,
    y1: S,
    d0: S,
    d1: S,
    t: S,
) -> Result<S, SciOxideError> {
    // Flat segments and the two nodes return stored data verbatim, so the
    // interpolation property holds bit for bit.
    if t == x0 {
        return Ok(y0);
    }
    if t == x1 {
        return Ok(y1);
    }
    if y0 == y1 && d0 == S::ZERO && d1 == S::ZERO {
        return Ok(y0);
    }
    // p = y0 + h01 (y1 - y0) + h (h10 d0 + h11 d1): eliminating h00 = 1 - h01
    // reproduces constants exactly; `linear_multiply_difference` never forms raw differences.
    let [_, h01, h10, h11] = hermite_basis::basis(interval_coordinate(t, x0, x1)?, 0)?;
    let value_change = linear_multiply_difference(h01, y1, y0)?;
    let left_slope_change = mul(linear_multiply_difference(d0, x1, x0)?, h10)?;
    let right_slope_change = mul(linear_multiply_difference(d1, x1, x0)?, h11)?;
    add(
        add(add(y0, value_change)?, left_slope_change)?,
        right_slope_change,
    )
}

/// Derivative of the cubic Hermite interpolant on `[x0, x1]` at `t`.
#[cold]
#[inline(never)]
fn cubic_hermite_deriv<S: RealScalar>(
    x0: S,
    x1: S,
    y0: S,
    y1: S,
    d0: S,
    d1: S,
    t: S,
) -> Result<S, SciOxideError> {
    // Nodes return the stored slopes exactly.
    if t == x0 {
        return Ok(d0);
    }
    if t == x1 {
        return Ok(d1);
    }
    if y0 == y1 && d0 == S::ZERO && d1 == S::ZERO {
        return Ok(S::ZERO);
    }
    // dp/dx = (1/h) dp/ds: the value term becomes a secant times h01'(s).
    let [_, value_basis, left_slope_basis, right_slope_basis] =
        hermite_basis::basis(interval_coordinate(t, x0, x1)?, 1)?;
    let secant = linear_difference_ratio(y1, y0, x1, x0)?;
    add(
        add(mul(secant, value_basis)?, mul(d0, left_slope_basis)?)?,
        mul(d1, right_slope_basis)?,
    )
}

/// Integral of the cubic Hermite interpolant on `[x0, x1]` over `[a, b]`.
#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn cubic_hermite_integral<S: RealScalar>(
    x0: S,
    x1: S,
    y0: S,
    y1: S,
    d0: S,
    d1: S,
    a: S,
    b: S,
) -> Result<S, SciOxideError> {
    if y0 == y1 && d0 == S::ZERO && d1 == S::ZERO {
        return linear_multiply_difference(y0, b, a);
    }
    let sa = interval_coordinate(a, x0, x1)?;
    let sb = interval_coordinate(b, x0, x1)?;
    // Antiderivatives of the Hermite basis in s (h00 eliminated as in evaluation).
    let eval_at = |s: S| -> Result<S, SciOxideError> {
        let [_, ih01, ih10, ih11] = hermite_basis::basis_integral(s)?;
        let base = mul(y0, s)?;
        let value_change = linear_multiply_difference(ih01, y1, y0)?;
        let left_slope = mul(linear_multiply_difference(d0, x1, x0)?, ih10)?;
        let right_slope = mul(linear_multiply_difference(d1, x1, x0)?, ih11)?;
        add(add(add(base, value_change)?, left_slope)?, right_slope)
    };
    linear_multiply_difference(sub(eval_at(sb)?, eval_at(sa)?)?, x1, x0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn constant_negative_zero_keeps_its_stored_value() {
        let cubic =
            PiecewiseCubic::new(vec![0.0_f64, 1.0], vec![-0.0, 0.0], vec![-0.0, 0.0], 1, 0.0)
                .unwrap();
        let mut fallback = cubic.clone();
        fallback.native_components.fill(false);
        let mut ws = cubic.workspace().unwrap();
        for model in [&cubic, &fallback] {
            for (t, value, slope) in [
                (0.0_f64, -0.0_f64, -0.0_f64),
                (0.5, -0.0, 0.0),
                (1.0, 0.0, 0.0),
            ] {
                let mut out = 1.0;
                model
                    .evaluate_into(t, ExtrapolationPolicy::Reject, &mut out, &mut ws)
                    .unwrap();
                assert_eq!(out.to_bits(), value.to_bits());
                model
                    .derivative_into(t, ExtrapolationPolicy::Reject, &mut out, &mut ws)
                    .unwrap();
                assert_eq!(out.to_bits(), slope.to_bits());
            }
        }
    }

    #[test]
    fn prepared_powers_match_hermite_values_derivatives_and_integrals() {
        let native = PiecewiseCubic::new(
            vec![0.0_f64, 0.5, 2.0],
            vec![1.0, -1.0, 3.0],
            vec![2.0, -0.5, 1.0],
            1,
            0.0,
        )
        .unwrap();
        assert!(native.native_components.iter().all(|&v| v));
        let mut fallback = native.clone();
        fallback.native_components.fill(false);
        let mut a = 0.0;
        let mut b = 0.0;
        let mut wa = native.workspace().unwrap();
        let mut wb = fallback.workspace().unwrap();
        for t in [0.0, 0.125, 0.5, 0.75, 1.9, 2.0] {
            let policy = ExtrapolationPolicy::Reject;
            native.evaluate_into(t, policy, &mut a, &mut wa).unwrap();
            fallback.evaluate_into(t, policy, &mut b, &mut wb).unwrap();
            assert!((a - b).abs() < 2e-14);
            native.derivative_into(t, policy, &mut a, &mut wa).unwrap();
            fallback
                .derivative_into(t, policy, &mut b, &mut wb)
                .unwrap();
            assert!((a - b).abs() < 2e-14);
            native
                .integral_into(0.0, t, policy, &mut a, &mut wa)
                .unwrap();
            fallback
                .integral_into(0.0, t, policy, &mut b, &mut wb)
                .unwrap();
            assert!((a - b).abs() < 2e-14);
        }
        let extreme = PiecewiseCubic::new(
            vec![-f64::MAX, f64::MAX],
            vec![2.0, 2.0],
            vec![0.0, 0.0],
            1,
            0.0,
        )
        .unwrap();
        assert!(extreme.native_components.iter().all(|&v| !v));
    }

    #[test]
    fn kernel_reproduces_nodes_and_rejects_invalid_shape() {
        let cubic = PiecewiseCubic::new(
            vec![0.0_f64, 1.0, 3.0],
            vec![1.0, 10.0, 2.0, 20.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.25, 1.0, 0.0, -1.0],
            2,
            [0.0; 2],
        )
        .unwrap();
        let mut workspace = cubic.workspace().unwrap();
        let mut out = [0.0; 2];
        cubic
            .evaluate_into(1.0, ExtrapolationPolicy::Reject, &mut out, &mut workspace)
            .unwrap();
        assert_eq!(out, [2.0, 20.0]);
        cubic
            .derivative_into(3.0, ExtrapolationPolicy::Reject, &mut out, &mut workspace)
            .unwrap();
        assert_eq!(out, [0.0, -1.0]);
        assert!(PiecewiseCubic::new(vec![0.0_f64, 1.0], vec![0.0], vec![0.0], 1, 0.0).is_err());
    }

    #[test]
    fn prepared_primitives_match_direct_integrals() {
        let cubic = PiecewiseCubic::new(
            vec![0.0_f64, 0.5, 2.0],
            vec![1.0, -1.0, 3.0],
            vec![2.0, -0.5, 1.0],
            1,
            0.0,
        )
        .unwrap();
        let prepared = cubic.prepare_antiderivative().unwrap();
        let mut workspace = cubic.workspace().unwrap();
        let mut direct = 0.0;
        for (a, b) in [(0.0, 2.0), (0.25, 1.5), (-0.5, 2.5)] {
            for policy in [
                ExtrapolationPolicy::MethodDefined,
                ExtrapolationPolicy::Clamp,
            ] {
                cubic
                    .integral_into(a, b, policy, &mut direct, &mut workspace)
                    .unwrap();
                let prepared = prepared.integral(a, b, policy).unwrap();
                assert!((prepared - direct).abs() < 1e-14, "{a} {b} {policy:?}");
            }
        }
    }
}
