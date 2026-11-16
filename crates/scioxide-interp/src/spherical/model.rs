//! The spherical spline and its query workspace.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError};

use super::basis::{SphereBasis, resolve};
use crate::SplineBasisWorkspace;
use crate::error::invalid;
use crate::protocol::{
    Differentiable, ExtrapolationPolicy, ExtrapolationReport, Integrable, Interpolant,
    TwiceDifferentiable, integral_bounds,
};
use crate::values::zeroed;

/// Represents a cubic spline on the sphere that is continuously differentiable
/// everywhere, including at the poles, with affine values.
///
/// Points are `[θ, φ]` in radians, with colatitude `θ ∈ [0, π]` measured from
/// the north pole and longitude `φ`. With `u = θ/π` and the cubic Hermite basis
/// `h00, h01, h10, h11` on `[0, 1]`, the spline is
///
/// ```text
/// f(θ, φ) = a₁ h00(u) + a₂ h01(u)
///         + π h10(u) (a₃ cos φ + a₄ sin φ) + π h11(u) (a₅ cos φ + a₆ sin φ)
///         + Σ_{i,j} c_ij M_i(θ) P_j(φ),
/// ```
///
/// where `M_i` are the cubic B-splines on the colatitude breakpoints whose
/// value and slope vanish at both poles, and `P_j` are the periodic cubic
/// B-splines on the longitude breakpoints. The first six terms, the polar
/// modes, fix the value `a₁` or `a₂` and the tangent plane at each pole, so
/// `f` has one value and one gradient there whatever the longitude.
///
/// Longitude wraps, so every finite `φ` is inside the domain. Colatitude
/// follows the [`ExtrapolationPolicy`] on `[0, π]`, but no continuation may
/// leave that interval. Under [`ExtrapolationPolicy::Clamp`], colatitude
/// derivatives are zero strictly outside it. Derivatives are partial
/// derivatives with respect to `θ` and `φ`, not surface gradients.
/// [`Interpolant::report`] wraps the longitude and reports only the distance
/// in colatitude.
///
/// Coefficients are stored by mode, `[mode][component]`, with the six polar
/// modes first. Each real value component is evaluated exactly as a scalar
/// spline of that component. A point query costs 16 tensor terms plus six
/// polar terms per component; an integral costs `O(modes · components)`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{
///     Differentiable, ExtrapolationPolicy, Integrable, Interpolant, SphericalSamples,
///     SphericalSplinePlan, SplineFitLimits,
/// };
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (0..=4)
///     .flat_map(|i| (0..8).map(move |j| [PI * i as f64 / 4.0, PI * j as f64 / 4.0]))
///     .collect();
/// // 1 and h01(θ/π) lie in the space spanned by the polar modes alone.
/// let values: Vec<Complex<f64>> = samples
///     .iter()
///     .map(|p| {
///         let u = p[0] / PI;
///         Complex::new(1.0, u * u * (3.0 - 2.0 * u))
///     })
///     .collect();
/// let plan = SphericalSplinePlan::new(
///     &[0.0, PI],
///     &[0.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI],
///     SphericalSamples::Scattered(&samples),
///     None,
///     0.0,
///     SplineFitLimits::new(),
/// )?;
/// let spline = plan.fit(&values)?.into_spline();
/// let policy = ExtrapolationPolicy::Reject;
/// let north = spline.evaluate([0.0, 1.0], policy)?;
/// assert!((north.re - 1.0).abs() < 1e-12 && north.im.abs() < 1e-12);
/// let [_, d_longitude] = spline.gradient([0.0, 1.0], policy)?;
/// assert!(d_longitude.re.abs() + d_longitude.im.abs() < 1e-12);
/// let area = spline.integral([0.0, 0.0], [PI, 2.0 * PI], policy)?;
/// assert!((area.re - 4.0 * PI).abs() < 1e-10 && (area.im - 2.0 * PI).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SphericalSpline<S: RealScalar, V: AffineValue<Scale = S> = S> {
    basis: SphereBasis<S>,
    coefficients: Vec<S>,
    dimension: usize,
    template: V,
    scratch: usize,
}

/// Holds query scratch for a [`SphericalSpline`].
///
/// Create it with [`Interpolant::workspace`]. A workspace serves every
/// spherical spline whose requirement it covers, and a smaller one is rejected
/// with `DimensionMismatch`. It stays usable after a failed query.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, SphericalSamples, SphericalSplinePlan, SplineFitLimits,
/// };
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (0..=4)
///     .flat_map(|i| (0..8).map(move |j| [PI * i as f64 / 4.0, PI * j as f64 / 4.0]))
///     .collect();
/// let plan = SphericalSplinePlan::new(
///     &[0.0, PI],
///     &[0.0, PI / 2.0, PI, 1.5 * PI, 2.0 * PI],
///     SphericalSamples::Scattered(&samples),
///     None,
///     0.0,
///     SplineFitLimits::new(),
/// )?;
/// let spline = plan.fit(&vec![2.0; samples.len()])?.into_spline();
/// let mut workspace = spline.workspace()?;
/// let mut out = 0.0;
/// // The longitude 9 wraps to 9 − 2π.
/// spline.evaluate_into(
///     [1.0, 9.0],
///     ExtrapolationPolicy::Reject,
///     &mut out,
///     &mut workspace,
/// )?;
/// assert!((out - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SphericalSplineWorkspace<S: RealScalar> {
    basis: SplineBasisWorkspace<S>,
    scratch: Vec<S>,
}

impl<S: RealScalar> SphericalSplineWorkspace<S> {
    /// The basis scratch and the first `len` scalars, or `DimensionMismatch`.
    fn split(
        &mut self,
        len: usize,
    ) -> Result<(&mut SplineBasisWorkspace<S>, &mut [S]), SciOxideError> {
        let available = self.scratch.len();
        let scratch = self
            .scratch
            .get_mut(..len)
            .ok_or_else(|| crate::error::shape(len, available))?;
        Ok((&mut self.basis, scratch))
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S>> SphericalSpline<S, V> {
    /// Build a spline from checked mode-major coefficients.
    pub(super) fn from_parts(
        basis: SphereBasis<S>,
        coefficients: Vec<S>,
        dimension: usize,
        template: V,
    ) -> Result<Self, SciOxideError> {
        let quantity = scioxide_core::quantity!("sphere_workspace");
        let expected = checked_mul(basis.columns()?, dimension, quantity)?;
        if coefficients.len() != expected {
            return Err(crate::error::shape(expected, coefficients.len()));
        }
        crate::validate::validate_finite(&coefficients)?;
        let stage = checked_add(checked_mul(4, dimension, quantity)?, 8, quantity)?;
        let integral = checked_add(dimension, basis.integral_scratch()?, quantity)?;
        Ok(Self {
            basis,
            coefficients,
            dimension,
            template,
            scratch: stage.max(integral),
        })
    }

    /// Returns the colatitude knot vector and the longitude knot vector, which
    /// extends the periodic breakpoints by three knots at each end.
    pub fn knots(&self) -> [&[S]; 2] {
        self.basis.knots()
    }

    /// Returns the coefficients by mode, with the real components of each value
    /// interleaved and the six polar modes first.
    pub fn coefficients(&self) -> &[S] {
        &self.coefficients
    }

    /// Writes the partial derivative `∂^(k+l) f / ∂θ^k ∂φ^l` with
    /// `orders = [k, l]` to `out`.
    ///
    /// Each order is at most two, and zero orders give the value.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if an order is above two.
    /// - `DimensionMismatch` if `out` does not have `value_dimension()`
    ///   components or `workspace` is too small.
    /// - The coordinate errors listed in the [crate docs](crate#errors), and
    ///   `InvalidInput` if a continuation policy moves `θ` outside `[0, π]`.
    /// - `NonFinite` if the result overflows.
    pub fn derivative_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        orders: [usize; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        if orders.iter().any(|&order| order > 2) {
            return Err(invalid(
                scioxide_core::argument!("orders"),
                "sphere derivative orders are at most two per axis",
            ));
        }
        self.stage(point, policy, &[orders], out, workspace)
    }

    /// Returns the partial derivative of [`Self::derivative_into`] as an owned
    /// value.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::derivative_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn derivative(
        &self,
        point: [S; 2],
        orders: [usize; 2],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<V::Owned, SciOxideError> {
        let mut out = zeroed(&self.template)?;
        let mut workspace = self.query_workspace()?;
        self.derivative_into(point, orders, policy, &mut out, &mut workspace)?;
        Ok(out)
    }

    fn query_workspace(&self) -> Result<SphericalSplineWorkspace<S>, SciOxideError> {
        Ok(SphericalSplineWorkspace {
            basis: self.basis.basis_workspace()?,
            scratch: try_filled(
                self.scratch,
                S::ZERO,
                scioxide_core::quantity!("sphere_workspace"),
            )?,
        })
    }

    /// Compute the partials `orders` in scratch, then write them to `out`.
    fn stage<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        orders: &[[usize; 2]],
        out: &mut O,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let width = orders.len() * self.dimension;
        scioxide_core::check_value_shape(out, width)?;
        let (_, scratch) = workspace.split(self.scratch)?;
        let report = resolve(point, policy, scioxide_core::argument!("point"))?;
        let effective = report.effective();
        let clamped = policy == ExtrapolationPolicy::Clamp && point[0] != effective[0];
        let (stage, rest) = scratch.split_at_mut(4 * self.dimension);
        for (index, &partial) in orders.iter().enumerate() {
            let chunk = &mut stage[index * self.dimension..(index + 1) * self.dimension];
            if clamped && partial[0] > 0 {
                chunk.fill(S::ZERO);
            } else {
                self.basis.evaluate(
                    effective,
                    partial,
                    &self.coefficients,
                    self.dimension,
                    rest,
                    chunk,
                )?;
            }
        }
        scioxide_core::write_value(&stage[..width], out)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 2> for SphericalSpline<S, V> {
    type Value = V::Owned;
    type Workspace = SphericalSplineWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 2] {
        self.basis.domains()
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<SphericalSplineWorkspace<S>, SciOxideError> {
        self.query_workspace()
    }
    fn workspace_capacity(&self, workspace: &SphericalSplineWorkspace<S>) -> Capacity {
        if workspace.scratch.len() >= self.scratch {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage(point, policy, &[[0, 0]], out, workspace)
    }
    fn report(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, 2>, SciOxideError> {
        let _ = workspace;
        resolve(point, policy, scioxide_core::argument!("point"))
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Differentiable<S, 2>
    for SphericalSpline<S, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage(point, policy, &[[1, 0], [0, 1]], out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> TwiceDifferentiable<S, 2>
    for SphericalSpline<S, V>
{
    fn hessian_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage(
            point,
            policy,
            &[[2, 0], [1, 1], [1, 1], [0, 2]],
            out,
            workspace,
        )
    }
}

/// Integrates against the surface element `sin θ dθ dφ` over the box
/// `[θ₀, θ₁] × [φ₀, φ₁]` given by `lower` and `upper`.
///
/// Colatitude bounds follow the policy on `[0, π]`. Under
/// [`ExtrapolationPolicy::Clamp`] they are clipped to `[0, π]`, since the
/// sphere has no area beyond its poles, and continuation may not leave
/// `[0, π]`. Longitude bounds are not wrapped, so the box may cover several
/// turns. On each colatitude piece, the integrals of `θⁿ sin θ` come from a
/// 32-term Taylor series whose truncation error is below `f64` rounding.
impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Integrable<S, 2> for SphericalSpline<S, V> {
    fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        lower: [S; 2],
        upper: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut SphericalSplineWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let (basis, scratch) = workspace.split(self.scratch)?;
        let theta = integral_bounds(self.basis.domains()[0], lower[0], upper[0], policy)?;
        if theta[0] < S::ZERO || theta[1] > S::PI {
            return Err(invalid(
                scioxide_core::argument!("lower"),
                "spherical continuation cannot leave [0,pi]",
            ));
        }
        if !lower[1].is_finite() || !upper[1].is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "integration bounds must be finite",
            ));
        }
        if lower[1] > upper[1] {
            return Err(crate::error::out_of_range(
                scioxide_core::argument!("lower"),
                "integration lower bound must not exceed the upper bound",
            ));
        }
        let (stage, rest) = scratch.split_at_mut(self.dimension);
        self.basis.integral(
            theta,
            [lower[1], upper[1]],
            &self.coefficients,
            self.dimension,
            basis,
            rest,
            stage,
        )?;
        scioxide_core::write_value(stage, out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::SplineFitLimits;
    use core::f64::consts::PI;
    use scioxide_core::SciOxideErrorKind;

    /// `f = 1 + cos θ + (π/2 h10(θ/π)) cos φ` via explicit modes on the polar-only basis.
    fn model() -> SphericalSpline<f64, [f64; 2]> {
        let basis = SphereBasis::new(
            &[0., PI],
            &[0., PI / 2., PI, 1.5 * PI, 2. * PI],
            &SplineFitLimits::new(),
        )
        .unwrap();
        // Component 0 = 2 h00 + 0 h01 + 0.5 north·cos; component 1 = 1.
        let coefficients = vec![2., 1., 0., 1., 0.5, 0., 0., 0., 0., 0., 0., 0.];
        SphericalSpline::from_parts(basis, coefficients, 2, [0.0; 2]).unwrap()
    }

    #[test]
    fn values_gradients_and_poles() {
        let spline = model();
        let policy = ExtrapolationPolicy::Reject;
        assert_eq!(spline.evaluate([0.0, 1.3], policy).unwrap(), [2.0, 1.0]);
        assert_eq!(spline.evaluate([PI, 0.2], policy).unwrap(), [0.0, 1.0]);
        // Longitude wraps; the seam is continuous.
        let a = spline.evaluate([1.0, 0.25], policy).unwrap();
        let b = spline.evaluate([1.0, 0.25 + 2. * PI], policy).unwrap();
        assert!((a[0] - b[0]).abs() < 1e-14);
        let [_, d_phi] = spline.gradient([1.0, 0.0], policy).unwrap();
        assert!(d_phi[0].abs() < 1e-15 && d_phi[1] == 0.0);
        let hessian = spline.hessian([1.0, 0.4], policy).unwrap();
        assert_eq!(hessian[0][1], hessian[1][0]);
        // Clamped colatitude derivatives vanish outside [0, π].
        let clamped = spline
            .derivative([-0.1, 0.0], [1, 0], ExtrapolationPolicy::Clamp)
            .unwrap();
        assert_eq!(clamped, [0.0, 0.0]);
        assert!(spline.derivative([1.0, 0.0], [3, 0], policy).is_err());
        let report = spline
            .report(
                [-0.1, 7.0],
                ExtrapolationPolicy::Clamp,
                &mut spline.workspace().unwrap(),
            )
            .unwrap();
        assert_eq!(report.outside_distance(), [0.1, 0.0]);
    }

    #[test]
    fn integrals_follow_the_shared_bound_order() {
        let spline = model();
        let policy = ExtrapolationPolicy::Reject;
        // ∫ 1 dA over the sphere is 4π; ∫ h00 dA is 2π by symmetry.
        let whole = spline.integral([0.0, 0.0], [PI, 2. * PI], policy).unwrap();
        assert!((whole[1] - 4. * PI).abs() < 1e-13);
        assert!((whole[0] - 4. * PI).abs() < 1e-13);
        let turns = spline
            .integral([0.0, -1.0], [PI, 4. * PI - 1.0], policy)
            .unwrap();
        assert!((turns[1] - 8. * PI).abs() < 1e-12);
        assert_eq!(
            spline
                .integral([PI, 0.0], [0.0, 1.0], policy)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            spline
                .integral([0.0, f64::NAN], [PI, 1.0], policy)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
        assert!(spline.integral([-0.1, 0.0], [PI, 1.0], policy).is_err());
        let clamped = spline
            .integral([-1.0, 0.0], [PI, 2. * PI], ExtrapolationPolicy::Clamp)
            .unwrap();
        assert!((clamped[1] - 4. * PI).abs() < 1e-13);
    }

    #[test]
    fn undersized_workspace_is_a_dimension_mismatch() {
        let spline = model();
        let mut workspace = spline.workspace().unwrap();
        workspace.scratch.truncate(3);
        assert_eq!(spline.workspace_capacity(&workspace), Capacity::NeedsGrowth);
        let mut out = [7.0; 2];
        let error = spline
            .evaluate_into(
                [1.0, 1.0],
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut workspace,
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(out, [7.0; 2]);
        assert!(SphericalSpline::from_parts(model().basis, vec![0.0; 3], 1, 0.0).is_err());
    }
}
