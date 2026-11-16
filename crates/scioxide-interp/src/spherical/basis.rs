//! The spherical basis: six polar Hermite modes plus periodic cubic tensor
//! B-splines, its coordinate rules, and its value and area-integral kernels.
//!
//! Modes are ordered `[north, south, north·cos, north·sin, south·cos, south·sin,
//! interior lanes]`, where interior lane `(i, j)` (colatitude control `i`,
//! periodic longitude control `j`) is mode `6 + (i - 2) q + j` for `q` periodic
//! controls. Every kernel visits modes in increasing order, so each real value
//! component is accumulated exactly as a scalar model of that component.

#![allow(
    clippy::needless_range_loop,
    reason = "mode assembly indexes polar and tensor arrays in parallel"
)]

use crate::NodeSeparation;
use crate::validate::{validate_count, validate_nodes};
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Argument, Interval, RealScalar, SciOxideError};

use crate::adaptive::open_cubic;
use crate::checked::intermediate;
use crate::error::invalid;
use crate::protocol::{ExtrapolationPolicy, ExtrapolationReport};
use crate::{BSplineBasis, KnotSide, SplineBasisWorkspace, SplineFitLimits};

/// Validated colatitude and periodic longitude cubic bases.
#[derive(Debug, Clone)]
pub(super) struct SphereBasis<S: RealScalar> {
    theta: BSplineBasis<S>,
    phi: BSplineBasis<S>,
    periodic: usize,
}

impl<S: RealScalar> SphereBasis<S> {
    /// Validate both axes: colatitude breaks span `[0, π]`, at least four
    /// longitude intervals span `[0, 2π]`, and both knot vectors fit `limits`.
    pub(super) fn new(
        theta: &[S],
        phi: &[S],
        limits: &SplineFitLimits,
    ) -> Result<Self, SciOxideError> {
        let (theta_argument, longitude_argument) = (
            scioxide_core::argument!("theta_breaks"),
            scioxide_core::argument!("longitude_breaks"),
        );
        validate_count(theta.len(), 2, theta_argument)?;
        validate_count(phi.len(), 5, longitude_argument)?;
        let knots = theta.len().max(phi.len()).saturating_add(6);
        if knots > limits.max_knots() {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("sphere_knots"),
                knots,
                limits.max_knots(),
            ));
        }
        const SPAN: &str = "sphere breaks must span [0,pi] and [0,2pi]";
        let theta_basis = open_cubic(theta, theta_argument)?;
        validate_nodes(phi, 5, NodeSeparation::Exact, longitude_argument)?;
        if theta[0] != S::ZERO || theta[theta.len() - 1] != S::PI {
            return Err(invalid(theta_argument, SPAN));
        }
        if phi[0] != S::ZERO || phi[phi.len() - 1] != S::PI + S::PI {
            return Err(invalid(longitude_argument, SPAN));
        }
        let theta = theta_basis;
        let n = phi.len() - 1;
        let period = S::PI + S::PI;
        let mut knots =
            try_with_capacity(phi.len() + 6, scioxide_core::quantity!("longitude_knots"))?;
        for &x in &phi[n - 3..n] {
            knots.push(x - period);
        }
        knots.extend_from_slice(phi);
        for &x in &phi[1..4] {
            knots.push(x + period);
        }
        Ok(Self {
            theta,
            phi: BSplineBasis::new(knots, 3)?,
            periodic: n,
        })
    }

    /// Fallible copy for owned models built from a borrowed plan.
    pub(super) fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            theta: self.theta.try_clone()?,
            phi: self.phi.try_clone()?,
            periodic: self.periodic,
        })
    }

    /// Colatitude and extended periodic longitude knot vectors.
    pub(super) fn knots(&self) -> [&[S]; 2] {
        [self.theta.knots(), self.phi.knots()]
    }

    /// Colatitude and longitude base domains, `[0, π]` and `[0, 2π]`.
    pub(super) const fn domains(&self) -> [Interval<S>; 2] {
        [self.theta.domain(), self.phi.domain()]
    }

    /// Interior colatitude lanes.
    pub(super) const fn interior(&self) -> usize {
        self.theta.control_count() - 4
    }

    /// Periodic longitude controls.
    pub(super) const fn periodic(&self) -> usize {
        self.periodic
    }

    /// Coefficient modes.
    pub(super) fn columns(&self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("sphere_columns");
        checked_add(
            6,
            checked_mul(self.interior(), self.periodic, quantity)?,
            quantity,
        )
    }

    /// Scratch of [`Self::integral`]: colatitude integrals, a full longitude
    /// row, folded longitude integrals and one local row.
    pub(super) fn integral_scratch(&self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("sphere_workspace");
        checked_add(
            checked_add(
                self.theta.control_count(),
                self.phi.control_count(),
                quantity,
            )?,
            checked_add(self.periodic, 4, quantity)?,
            quantity,
        )
    }

    /// Fresh basis scratch for [`Self::integral`].
    pub(super) fn basis_workspace(&self) -> Result<SplineBasisWorkspace<S>, SciOxideError> {
        self.theta.workspace()
    }

    /// Visit `(mode, weight)` of the partial derivative `orders` at an effective
    /// point in increasing mode order; `rows` holds eight scalars.
    fn terms(
        &self,
        point: [S; 2],
        orders: [usize; 2],
        rows: &mut [S],
        mut visit: impl FnMut(usize, S) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let h = hermite(point[0], orders[0])?;
        let (cos, sin) = angular(point[1], orders[1]);
        if orders[1] == 0 {
            visit(0, h[0])?;
            visit(1, h[1])?;
        }
        visit(2, h[2] * cos)?;
        visit(3, h[2] * sin)?;
        visit(4, h[3] * cos)?;
        visit(5, h[3] * sin)?;
        let (theta_row, phi_row) = rows[..8].split_at_mut(4);
        let theta_first = self
            .theta
            .local(point[0], orders[0], KnotSide::Right, theta_row)?;
        let phi_first = self
            .phi
            .local(point[1], orders[1], KnotSide::Right, phi_row)?;
        let q = self.periodic;
        // Local longitude controls wrap at most once; visit the wrapped ones
        // (smallest periodic indices) first.
        let split = (q - phi_first).min(4);
        let lanes = self.theta.control_count() - 2;
        for (k, &theta_value) in theta_row.iter().enumerate() {
            let i = theta_first + k;
            if i < 2 || i >= lanes {
                continue;
            }
            let base = 6 + (i - 2) * q;
            for m in (split..4).chain(0..split) {
                visit(base + (phi_first + m) % q, theta_value * phi_row[m])?;
            }
        }
        Ok(())
    }

    /// Evaluate the partial `orders` of mode-major `coefficients` with
    /// `dimension` interleaved components at an effective point into `out`.
    pub(super) fn evaluate(
        &self,
        point: [S; 2],
        orders: [usize; 2],
        coefficients: &[S],
        dimension: usize,
        rows: &mut [S],
        out: &mut [S],
    ) -> Result<(), SciOxideError> {
        let out = &mut out[..dimension];
        out.fill(S::ZERO);
        self.terms(point, orders, rows, |mode, weight| {
            let controls = &coefficients[mode * dimension..(mode + 1) * dimension];
            for (slot, &coefficient) in out.iter_mut().zip(controls) {
                *slot += coefficient * weight;
            }
            Ok(())
        })?;
        out.iter()
            .try_for_each(|&value| crate::checked::output(value).map(drop))
    }

    /// Write the value modes at an effective point, scaled by `scale`, into a
    /// zero-initialized design `row`.
    pub(super) fn design_row(
        &self,
        point: [S; 2],
        scale: S,
        rows: &mut [S],
        row: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.terms(point, [0, 0], rows, |mode, weight| {
            row[mode] = intermediate(weight * scale)?;
            Ok(())
        })
    }

    /// Integrate against the sphere area `sin θ dθ dφ` over colatitudes
    /// `theta` (ordered, within `[0, π]`) and unwrapped ordered longitudes `phi`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn integral(
        &self,
        theta: [S; 2],
        phi: [S; 2],
        coefficients: &[S],
        dimension: usize,
        workspace: &mut SplineBasisWorkspace<S>,
        scratch: &mut [S],
        out: &mut [S],
    ) -> Result<(), SciOxideError> {
        let q = self.periodic;
        let (theta_integrals, rest) = scratch.split_at_mut(self.theta.control_count());
        let (phi_row, rest) = rest.split_at_mut(self.phi.control_count());
        let (phi_integrals, rest) = rest.split_at_mut(q);
        let row = &mut rest[..4];
        theta_integrals.fill(S::ZERO);
        let mut polar = [S::ZERO; 4];
        for span in self.theta.knots().windows(2) {
            let a = theta[0].max(span[0]);
            let b = theta[1].min(span[1]);
            if a >= b {
                continue;
            }
            let mid = scioxide_core::checked_midpoint(a, b)?;
            let half = (b - a) / S::TWO;
            let moments = sine_moments(mid, half)?;
            let mut factorial = S::ONE;
            for order in 0..4 {
                if order > 0 {
                    factorial *= S::try_from_usize_exact(order)?;
                }
                let first = self.theta.local(mid, order, KnotSide::Right, row)?;
                for (slot, &d) in theta_integrals[first..first + 4].iter_mut().zip(&*row) {
                    *slot = intermediate(*slot + d * moments[order] / factorial)?;
                }
                let h = hermite(mid, order)?;
                for i in 0..4 {
                    polar[i] = intermediate(polar[i] + h[i] * moments[order] / factorial)?;
                }
            }
        }
        let width = intermediate(phi[1] - phi[0])?;
        let period = S::PI + S::PI;
        let turns = (width / period).floor();
        let remainder = width % period;
        let start = wrap(phi[0])?;
        phi_integrals.fill(S::ZERO);
        self.phi.integrals_into(
            S::ZERO,
            period,
            ExtrapolationPolicy::Reject,
            phi_row,
            workspace,
        )?;
        for (j, &value) in phi_row.iter().enumerate() {
            phi_integrals[j % q] += turns * value;
        }
        let end = start + remainder;
        for (a, b) in [
            (start, end.min(period)),
            (S::ZERO, (end - period).max(S::ZERO)),
        ] {
            if a == b {
                continue;
            }
            self.phi
                .integrals_into(a, b, ExtrapolationPolicy::Reject, phi_row, workspace)?;
            for (j, &value) in phi_row.iter().enumerate() {
                phi_integrals[j % q] += value;
            }
        }
        let cos_integral = end.sin() - start.sin();
        let sin_integral = start.cos() - end.cos();
        let polar_modes = [
            polar[0] * width,
            polar[1] * width,
            polar[2] * cos_integral,
            polar[2] * sin_integral,
            polar[3] * cos_integral,
            polar[3] * sin_integral,
        ];
        let out = &mut out[..dimension];
        out.fill(S::ZERO);
        let mut accumulate = |mode: usize, weight: S| {
            let controls = &coefficients[mode * dimension..(mode + 1) * dimension];
            for (slot, &coefficient) in out.iter_mut().zip(controls) {
                *slot += coefficient * weight;
            }
        };
        for (mode, &weight) in polar_modes.iter().enumerate() {
            accumulate(mode, weight);
        }
        for i in 2..self.theta.control_count() - 2 {
            for j in 0..q {
                accumulate(6 + (i - 2) * q + j, theta_integrals[i] * phi_integrals[j]);
            }
        }
        out.iter()
            .try_for_each(|&value| crate::checked::output(value).map(drop))
    }
}

/// Angular factors `(dᵏ cos φ, dᵏ sin φ)` of derivative order `order`.
fn angular<S: RealScalar>(phi: S, order: usize) -> (S, S) {
    let c = phi.cos();
    let s = phi.sin();
    match order % 4 {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

/// Polar cubic Hermite factors `[h00, h01, π h10, π h11]` of `u = θ/π` and
/// their `order`-th colatitude derivatives.
fn hermite<S: RealScalar>(theta: S, order: usize) -> Result<[S; 4], SciOxideError> {
    let mut result = crate::hermite_basis::basis(theta / S::PI, order)?;
    for (i, value) in result.iter_mut().enumerate() {
        for _ in 0..order {
            *value /= S::PI;
        }
        if i >= 2 {
            *value *= S::PI;
        }
        *value = intermediate(*value)?;
    }
    Ok(result)
}

/// Polynomial-times-sine moments `∫_{-h}^{h} sⁿ sin(m + s) ds`, `n < 4`, from a
/// truncated Taylor series. `|half| <= π/2`; 32 terms put the truncation below
/// `f64` rounding.
fn sine_moments<S: RealScalar>(mid: S, half: S) -> Result<[S; 4], SciOxideError> {
    let mut result = [S::ZERO; 4];
    let mut coefficient = S::ONE;
    for k in 0..32 {
        if k > 0 {
            coefficient = coefficient * half / S::try_from_usize_exact(k)?;
        }
        let (_, sin) = angular(mid, k);
        let mut power = half;
        for (n, out) in result.iter_mut().enumerate() {
            if (n + k) % 2 == 0 {
                *out = intermediate(
                    *out + S::TWO * power * coefficient * sin / S::try_from_usize_exact(n + k + 1)?,
                )?;
            }
            power *= half;
        }
    }
    Ok(result)
}

/// Wrap a finite longitude into `[0, 2π)`.
pub(super) fn wrap<S: RealScalar>(phi: S) -> Result<S, SciOxideError> {
    if !phi.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "longitude must be finite",
        ));
    }
    let period = S::PI + S::PI;
    // `%` retains precision for large finite longitude without a huge quotient product.
    let r = phi % period;
    let r = if r < S::ZERO { r + period } else { r };
    Ok(if r == period { S::ZERO } else { r })
}

/// Resolve `[colatitude, longitude]`: longitude wraps and the colatitude
/// follows `policy` on `[0, π]`, which no continuation may leave.
pub(super) fn resolve<S: RealScalar>(
    point: [S; 2],
    policy: ExtrapolationPolicy<S>,
    argument: Argument,
) -> Result<ExtrapolationReport<S, 2>, SciOxideError> {
    let longitude = wrap(point[1])?;
    let report = policy.resolve([Interval::new(S::ZERO, S::PI)?], [point[0]])?;
    let theta = report.effective()[0];
    if theta < S::ZERO || theta > S::PI {
        return Err(invalid(
            argument,
            "spherical continuation cannot leave [0,pi]",
        ));
    }
    Ok(ExtrapolationReport::new(
        point,
        [theta, longitude],
        [report.outside_distance()[0], S::ZERO],
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn basis() -> SphereBasis<f64> {
        SphereBasis::new(
            &[0., PI / 2., PI],
            &[0., PI / 2., PI, 1.5 * PI, 2. * PI],
            &SplineFitLimits::new(),
        )
        .unwrap()
    }

    #[test]
    fn modes_are_counted_and_visited_in_order() {
        let basis = basis();
        assert_eq!(basis.interior(), 1);
        assert_eq!(basis.periodic(), 4);
        assert_eq!(basis.columns().unwrap(), 10);
        let mut rows = [0.0; 8];
        let mut last = None;
        let mut sum = 0.0;
        // Longitude 6.0 sits in the last span, so its local controls wrap.
        basis
            .terms([1.0, 6.0], [0, 0], &mut rows, |mode, weight| {
                assert!(last.is_none_or(|previous| mode > previous));
                last = Some(mode);
                if mode >= 6 {
                    sum += weight;
                }
                Ok(())
            })
            .unwrap();
        // Interior lanes form a partition of the interior colatitude weight.
        assert!(sum > 0.0 && sum < 1.0);
    }

    #[test]
    fn hermite_and_angular_factors() {
        assert_eq!(hermite(0.0_f64, 0).unwrap(), [1., 0., 0., 0.]);
        assert_eq!(hermite(PI, 0).unwrap(), [0., 1., 0., 0.]);
        let mid = hermite(PI / 2., 0).unwrap();
        assert!((mid[0] - 0.5).abs() < 1e-15 && (mid[1] - 0.5).abs() < 1e-15);
        // d/dθ of π h10(θ/π) is 1 at the north pole.
        assert!((hermite(0.0_f64, 1).unwrap()[2] - 1.0).abs() < 1e-15);
        assert_eq!(hermite(1.0_f64, 4).unwrap(), [0.0; 4]);
        assert_eq!(angular(0.0_f64, 1), (-0.0, 1.0));
    }

    #[test]
    fn sine_moments_match_closed_forms() {
        // ∫_{-h}^{h} sin(m + s) ds = 2 sin(m) sin(h).
        let (mid, half) = (0.7_f64, 0.4);
        let moments = sine_moments(mid, half).unwrap();
        assert!((moments[0] - 2. * mid.sin() * half.sin()).abs() < 1e-15);
        // ∫ s sin(m + s) ds = 2 cos(m) (sin h - h cos h).
        let expected = 2. * mid.cos() * (half.sin() - half * half.cos());
        assert!((moments[1] - expected).abs() < 1e-15);
    }

    #[test]
    fn coordinates_wrap_and_respect_the_poles() {
        assert!((wrap(-0.5_f64).unwrap() - (2. * PI - 0.5)).abs() < 1e-15);
        assert_eq!(wrap(2. * PI).unwrap(), 0.0);
        assert!(wrap(f64::NAN).is_err());
        let argument = scioxide_core::argument!("point");
        let report = resolve([1.0, 7.0], ExtrapolationPolicy::Reject, argument).unwrap();
        assert!((report.effective()[1] - (7.0 - 2. * PI)).abs() < 1e-15);
        assert_eq!(report.outside_distance(), [0.0, 0.0]);
        let clamped = resolve([-0.25, 0.0], ExtrapolationPolicy::Clamp, argument).unwrap();
        assert_eq!(clamped.effective()[0], 0.0);
        assert_eq!(clamped.outside_distance()[0], 0.25);
        assert!(resolve([-0.25, 0.0], ExtrapolationPolicy::MethodDefined, argument).is_err());
    }

    #[test]
    fn axis_validation_and_knot_limits() {
        let limits = SplineFitLimits::new();
        let phi = [0., PI / 2., PI, 1.5 * PI, 2. * PI];
        assert!(SphereBasis::new(&[0., PI], &phi[..4], &limits).is_err());
        assert!(SphereBasis::new(&[0., 1.0], &phi, &limits).is_err());
        let error = SphereBasis::new(&[0., PI], &phi, &limits.with_max_knots(10)).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
}
