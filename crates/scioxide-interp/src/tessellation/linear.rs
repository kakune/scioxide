//! Continuous simplex-linear interpolation.

use scioxide_core::backend::checked_mul;
use scioxide_core::{AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError};
use scioxide_spatial::Delaunay;

use super::{Base, Family, ScatteredOptions, ScatteredValues, ScatteredWorkspace};
use crate::checked::intermediate;
use crate::protocol::{Differentiable, ExtrapolationPolicy, ExtrapolationReport, Interpolant};
use crate::values::zeroed;

/// Interpolates scattered data linearly on each simplex of a Delaunay triangulation.
///
/// Inside a simplex with vertices `p₀, …, p_D` and values `v₀, …, v_D`, the
/// interpolant is
///
/// ```text
/// f(x) = Σᵢ λᵢ(x) vᵢ,
/// ```
///
/// where `λᵢ(x)` are the barycentric coordinates of `x` (`Σᵢ λᵢ = 1`,
/// `x = Σᵢ λᵢ pᵢ`). The result is continuous, reproduces affine functions
/// exactly and returns the stored value at a site. The gradient is constant on
/// each simplex; on a face shared by two simplices it is that of the simplex
/// found by point location.
///
/// Outside the convex hull of the sites, `Clamp` evaluates at the nearest hull
/// point, and its gradient is that of the projection: tangent to the nearest
/// hull face, and zero where the nearest hull point is a vertex. Where two
/// projection regions meet, their gradients must agree to rounding; otherwise
/// the gradient is `DomainError` (`Undefined`). `BoundedContinue` and
/// `MethodDefined` extend the affine function of the nearest hull simplex. See
/// [Extrapolation](crate#extrapolation).
///
/// Each query locates the simplex that contains the point. An exterior query
/// also scans every hull facet, so its cost grows with the size of the
/// triangulation.
///
/// A nearly flat simplex (a sliver) may have no usable barycentric frame.
/// Values at points within `32 ε` (relative to the hull extent and the query
/// magnitude) of one of its faces are then interpolated on that face, and the
/// `Clamp` gradient is the face's tangential gradient. Other queries in a
/// sliver fail with `NumericalInstability`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     Differentiable, ExtrapolationPolicy, Interpolant, ScatteredValues, SimplexLinearInterpolator,
/// };
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// // f(x, y) = 1 + 3x + 2y
/// let values = ScatteredValues::from_input_values(&mesh, vec![1.0, 4.0, 3.0, 6.0])?;
/// let model = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values)?;
/// assert!((model.evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)? - 2.75).abs() < 1e-12);
/// let [dx, dy] = model.gradient([0.25, 0.5], ExtrapolationPolicy::Reject)?;
/// assert!((dx - 3.0).abs() < 1e-12 && (dy - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct SimplexLinearInterpolator<
    'a,
    S: RealScalar,
    const D: usize,
    V: AffineValue<Scale = S> = S,
> {
    base: Base<'a, S, V, D>,
}

impl<S: RealScalar, const D: usize, V: AffineValue<Scale = S>> Clone
    for SimplexLinearInterpolator<'_, S, D, V>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: RealScalar, const D: usize, V: AffineValue<Scale = S>> Copy
    for SimplexLinearInterpolator<'_, S, D, V>
{
}

impl<'a, S: RealScalar, const D: usize, V: AffineValue<Scale = S>>
    SimplexLinearInterpolator<'a, S, D, V>
{
    /// Creates the interpolator from a triangulation and the values at its sites.
    ///
    /// The query workspace has no memory limit: this is [`Self::with_options`]
    /// with [`ScatteredOptions::default`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` differs from the dimension of `tessellation`,
    ///   or `values` was built for a different triangulation.
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    pub fn new(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(tessellation, values, ScatteredOptions::default())
    }

    /// Creates the interpolator with a limit on the query workspace from `options`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` differs from the dimension of `tessellation`,
    ///   or `values` was built for a different triangulation.
    /// - `ResourceLimit` if the query workspace exceeds
    ///   [`ScatteredOptions::maximum_workspace_bytes`].
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    pub fn with_options(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
        options: ScatteredOptions,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            base: Base::new(tessellation, values, options, Family::Linear)?,
        })
    }
    fn gradient_piece(
        &self,
        simplex: usize,
        output: &mut [S],
        scratch: &mut super::Scratch<S>,
    ) -> Result<(), SciOxideError> {
        let components = self.base.values.components;
        self.base.tessellation.barycentric_gradients_into(
            simplex,
            &mut scratch.barycentric[..(D + 1) * D],
            &mut scratch.staging[..(D + 1) * D],
        )?;
        let vertices = self.base.tessellation.simplex(simplex)?;
        for (axis, row) in output.chunks_exact_mut(components.max(1)).enumerate() {
            for (component, slot) in row.iter_mut().enumerate() {
                *slot = vertices
                    .iter()
                    .enumerate()
                    .try_fold(S::ZERO, |sum, (local, &site)| {
                        intermediate(
                            scratch.barycentric[local * D + axis]
                                .mul_add(self.base.values.component(site, component), sum),
                        )
                    })?;
            }
        }
        Ok(())
    }
}

impl<S: RealScalar, const D: usize, V: AffineValue<Scale = S> + Sync> Interpolant<S, D>
    for SimplexLinearInterpolator<'_, S, D, V>
{
    type Value = V::Owned;
    type Workspace = ScatteredWorkspace<S>;

    fn domains(&self) -> [Interval<S>; D] {
        self.base.domains
    }
    fn value_dimension(&self) -> usize {
        self.base.values.components
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.base.values.template)
    }
    fn workspace(&self) -> Result<ScatteredWorkspace<S>, SciOxideError> {
        self.base.workspace()
    }
    fn workspace_capacity(&self, workspace: &ScatteredWorkspace<S>) -> Capacity {
        self.base.capacity(workspace)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.evaluate_into_with_report(point, policy, out, workspace)
            .map(|_| ())
    }
    fn evaluate_into_with_report<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        let components = self.base.values.components;
        scioxide_core::check_value_shape(out, components)?;
        self.base.check(workspace)?;
        let scratch = &mut workspace.scratch;
        let resolution = self.base.classify(point, policy, scratch, false)?;
        let staged = &mut workspace.output[..components];
        self.base
            .interpolate_resolved(&resolution, staged, scratch)?;
        scioxide_core::write_value(staged, out)?;
        Ok(resolution.report)
    }
    fn report(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        self.base.check(workspace)?;
        Ok(self
            .base
            .classify(point, policy, &mut workspace.scratch, false)?
            .report)
    }
}

impl<S: RealScalar, const D: usize, V: AffineValue<Scale = S> + Sync> Differentiable<S, D>
    for SimplexLinearInterpolator<'_, S, D, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let components = self.base.values.components;
        let width = checked_mul(D, components, scioxide_core::quantity!("gradient_entries"))?;
        scioxide_core::check_value_shape(out, width)?;
        self.base.check(workspace)?;
        // On a simplex the interpolant is sum_k lambda_k(x) f_k with affine
        // lambda_k, so its gradient is the constant sum_k grad(lambda_k) f_k.
        let scratch = &mut workspace.scratch;
        let resolution = self.base.classify(point, policy, scratch, false)?;
        let staged = &mut workspace.output[..width];
        if policy == ExtrapolationPolicy::Clamp && resolution.hull_face.is_some() {
            self.base.clamp_gradient(
                &resolution,
                staged,
                &mut workspace.comparison[..width],
                scratch,
                |simplex, _, count, out, scratch| {
                    match self.gradient_piece(simplex, out, scratch) {
                        // A sliver's tangential gradient is its hull face's.
                        Err(error) if super::degenerate(&error) => {
                            self.base.face_linear_gradient(count, out, scratch)
                        }
                        result => result,
                    }
                },
            )?;
        } else {
            self.gradient_piece(resolution.simplex, staged, scratch)?;
        }
        scioxide_core::write_value(staged, out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::tests::{options, triangulation};
    use super::*;
    use alloc::vec::Vec;
    use scioxide_core::{Complex, NdView, SciOxideErrorView, StridedLayout};
    use scioxide_spatial::{GeometryLimits, GeometryOptions, PointSet};

    #[test]
    fn reproduces_affine_values_and_gradient() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1.]);
        let values = ScatteredValues::from_input_values(&mesh, vec![1., 4., 3., 6.]).unwrap();
        let model =
            SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &values, options()).unwrap();
        let mut workspace = model.workspace().unwrap();
        let mut output = 0.;
        model
            .evaluate_into(
                [0.25, 0.5],
                ExtrapolationPolicy::Reject,
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert!((output - 2.75).abs() < 1e-12);
        let report = model
            .report([0.25, 0.5], ExtrapolationPolicy::Reject, &mut workspace)
            .unwrap();
        assert_eq!(report.outside_distance(), [0.; 2]);
        let mut gradient = [0.; 2];
        model
            .gradient_into(
                [0.25, 0.5],
                ExtrapolationPolicy::Reject,
                &mut gradient,
                &mut workspace,
            )
            .unwrap();
        assert!((gradient[0] - 3.).abs() < 1e-12);
        assert!((gradient[1] - 2.).abs() < 1e-12);
        // Clamped exterior axes have zero derivatives; continued pieces do not.
        model
            .gradient_into(
                [2., 0.5],
                ExtrapolationPolicy::Clamp,
                &mut gradient,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(gradient[0], 0.);
        assert!((gradient[1] - 2.).abs() < 1e-12);
        model
            .gradient_into(
                [2., 0.5],
                ExtrapolationPolicy::MethodDefined,
                &mut gradient,
                &mut workspace,
            )
            .unwrap();
        assert!((gradient[0] - 3.).abs() < 1e-12);
    }

    #[test]
    fn query_sequences_cross_cells_and_recover_after_errors() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1., 0.5, 0.5]);
        let site_values: Vec<_> = mesh
            .points()
            .chunks_exact(2)
            .map(|point| 1. + 2. * point[0] + 3. * point[1])
            .collect();
        let values = ScatteredValues::from_site_values(&mesh, site_values).unwrap();
        let model =
            SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &values, options()).unwrap();
        let mut workspace = model.workspace().unwrap();
        let mut output = [0.];
        let queries = [
            [0.1, 0.2],
            [0.9, 0.8],
            [0.5, 0.5],
            [0., 0.],
            [1., 0.5],
            [0.2, 0.9],
        ];
        for &point in queries.iter().chain(queries.iter().rev()) {
            model
                .evaluate_into(
                    point,
                    ExtrapolationPolicy::Reject,
                    &mut output,
                    &mut workspace,
                )
                .unwrap();
            assert!((output[0] - (1. + 2. * point[0] + 3. * point[1])).abs() < 1e-12);
            let before = output;
            assert!(
                model
                    .evaluate_into(
                        [2., 2.],
                        ExtrapolationPolicy::Reject,
                        &mut output,
                        &mut workspace
                    )
                    .is_err()
            );
            assert_eq!(output, before);
            let mut wrong = [0.; 2];
            assert!(
                matches!(model.evaluate_into(point, ExtrapolationPolicy::Reject, &mut wrong[..], &mut workspace), Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }))
            );
        }
    }

    #[test]
    fn clamp_projects_to_the_hull() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![10., 20., 30.]).unwrap();
        let linear =
            SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &values, options()).unwrap();
        let mut workspace = linear.workspace().unwrap();
        let report = linear
            .report([1., 1.], ExtrapolationPolicy::Clamp, &mut workspace)
            .unwrap();
        assert!((report.effective()[0] - 0.5).abs() < 1e-12);
        assert!((report.effective()[1] - 0.5).abs() < 1e-12);
        let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 0.1 };
        assert!(linear.evaluate([1., 1.], bounded).is_err());
        assert!(linear.evaluate([0.55, 0.5], bounded).is_ok());
    }

    #[test]
    fn f32_complex_components_equal_scalar_components() {
        let coordinates = [0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let shape = [3, 2];
        let strides = [2, 1];
        let points = PointSet::new(
            NdView::new(&coordinates, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        let geometry = GeometryOptions::new().with_limits(
            GeometryLimits::new()
                .with_max_dimension(2)
                .with_max_points(3)
                .with_max_facets(32)
                .with_max_predicate_terms(10_000)
                .with_max_storage_bytes(1024 * 1024),
        );
        let mesh = Delaunay::with_options(&points, geometry).unwrap();
        let complex = vec![
            Complex::new(1.0_f32, -1.0),
            Complex::new(4.0, 2.0),
            Complex::new(3.0, 0.0),
        ];
        let values = ScatteredValues::from_site_values(&mesh, complex.clone()).unwrap();
        let real = ScatteredValues::from_site_values(&mesh, complex.iter().map(|v| v.re).collect())
            .unwrap();
        let imaginary =
            ScatteredValues::from_site_values(&mesh, complex.iter().map(|v| v.im).collect())
                .unwrap();
        let options = ScatteredOptions::new();
        let model =
            SimplexLinearInterpolator::<_, 2, _>::with_options(&mesh, &values, options).unwrap();
        let real = SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &real, options).unwrap();
        let imaginary =
            SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &imaginary, options).unwrap();
        let query = [0.25, 0.5];
        let value = model.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
        assert_eq!(
            value.re,
            real.evaluate(query, ExtrapolationPolicy::Reject).unwrap()
        );
        assert_eq!(
            value.im,
            imaginary
                .evaluate(query, ExtrapolationPolicy::Reject)
                .unwrap()
        );
        let [dx, dy] = model.gradient(query, ExtrapolationPolicy::Reject).unwrap();
        let [rx, ry] = real.gradient(query, ExtrapolationPolicy::Reject).unwrap();
        assert_eq!((dx.re, dy.re), (rx, ry));
    }
}
