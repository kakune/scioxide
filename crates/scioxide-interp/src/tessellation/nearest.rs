//! Piecewise-constant nearest-site interpolation.

use scioxide_core::backend::checked_add;
use scioxide_core::{AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError};
use scioxide_spatial::{
    Delaunay, KdTree, Neighbor, SearchOptions, SearchWorkspaceRequest, SearchWorkspaceRequirements,
};

use super::{Base, Family, ScatteredOptions, ScatteredValues, ScatteredWorkspace};
use crate::protocol::{ExtrapolationPolicy, ExtrapolationReport, Interpolant};
use crate::values::zeroed;

/// Interpolates scattered data by the value at the nearest site.
///
/// The interpolant is piecewise constant on the Voronoi cells of the sites:
/// `f(x) = vₖ`, where `pₖ` is the site closest to `x` in the Euclidean norm.
/// When rounded distances tie, the site with the lowest index in the
/// triangulation wins.
///
/// The domain is the convex hull of the sites, as for the other
/// triangulation-based interpolators (see [`Interpolant::report`]). Outside
/// it, `Clamp` returns the value nearest to the nearest hull point, while
/// `BoundedContinue` and `MethodDefined` return the value nearest to the query
/// itself. See [Extrapolation](crate#extrapolation).
///
/// The constructor builds a k-d tree over the sites. A query takes expected
/// `O(log n)` time for `n` sites.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, NearestNeighborInterpolator, ScatteredValues,
/// };
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// let values = ScatteredValues::from_input_values(&mesh, vec![10.0, 20.0, 30.0])?;
/// let model = NearestNeighborInterpolator::<_, 2>::new(&mesh, &values)?;
/// assert_eq!(
///     model.evaluate([0.9, 0.05], ExtrapolationPolicy::Reject)?,
///     20.0
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. H. Friedman, J. L. Bentley and R. A. Finkel, "An algorithm for finding
///   best matches in logarithmic expected time", *ACM Trans. Math. Softw.*
///   3(3), 1977, 209–226.
#[derive(Debug)]
#[must_use]
pub struct NearestNeighborInterpolator<
    'a,
    S: RealScalar,
    const D: usize,
    V: AffineValue<Scale = S> = S,
> {
    base: Base<'a, S, V, D>,
    tree: KdTree<'a, S>,
    search: SearchWorkspaceRequirements,
}

impl<S: RealScalar, const D: usize, V: AffineValue<Scale = S>> Clone
    for NearestNeighborInterpolator<'_, S, D, V>
{
    fn clone(&self) -> Self {
        Self {
            base: self.base,
            tree: self.tree.clone(),
            search: self.search,
        }
    }
}

impl<'a, S: RealScalar, const D: usize, V: AffineValue<Scale = S>>
    NearestNeighborInterpolator<'a, S, D, V>
{
    /// Creates the interpolator and builds a k-d tree over the sites.
    ///
    /// The query workspace has no memory limit: this is [`Self::with_options`]
    /// with [`ScatteredOptions::default`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` differs from the dimension of `tessellation`,
    ///   or `values` was built for a different triangulation.
    /// - `SizeOverflow` or `AllocationFailure` if the tree or workspace cannot
    ///   be sized or allocated.
    pub fn new(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(tessellation, values, ScatteredOptions::default())
    }

    /// Creates the interpolator with a limit on the query workspace from `options`,
    /// and builds a k-d tree over the sites.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` differs from the dimension of `tessellation`,
    ///   or `values` was built for a different triangulation.
    /// - `ResourceLimit` if the query workspace, including the k-d tree search
    ///   scratch, exceeds [`ScatteredOptions::maximum_workspace_bytes`].
    /// - `SizeOverflow` or `AllocationFailure` if the tree or workspace cannot
    ///   be sized or allocated.
    pub fn with_options(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
        options: ScatteredOptions,
    ) -> Result<Self, SciOxideError> {
        let base = Base::new(tessellation, values, options, Family::Nearest)?;
        let tree = KdTree::new(&tessellation.point_set())?;
        let search =
            tree.workspace_requirements(SearchWorkspaceRequest::new(1, SearchOptions::new()))?;
        let required = checked_add(
            base.workspace_bytes()?,
            search.bytes::<S>()?,
            scioxide_core::quantity!("scattered_workspace_bytes"),
        )?;
        if required > options.maximum_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("scattered_workspace_bytes"),
                required,
                options.maximum_workspace_bytes,
            ));
        }
        Ok(Self { base, tree, search })
    }

    fn check(&self, workspace: &ScatteredWorkspace<S>) -> Result<(), SciOxideError> {
        self.base.check(workspace)?;
        if workspace.search.check(&self.search)? != Capacity::Sufficient {
            return Err(crate::error::shape(1, 0));
        }
        Ok(())
    }

    fn evaluate_report<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        let components = self.base.values.components;
        scioxide_core::check_value_shape(out, components)?;
        self.check(workspace)?;
        let resolution = self
            .base
            .classify(point, policy, &mut workspace.scratch, false)?;
        let query = resolution.report.effective();
        let site = match self.base.nodal_site(resolution.simplex, &query)? {
            Some(site) => site,
            None => {
                let mut nearest = [Neighbor::default(); 1];
                self.tree.nearest_into(
                    &query,
                    1,
                    SearchOptions::new(),
                    &mut nearest,
                    &mut workspace.search,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                nearest[0].index
            }
        };
        let staged = &mut workspace.output[..components];
        for (component, slot) in staged.iter_mut().enumerate() {
            *slot = self.base.values.component(site, component);
        }
        scioxide_core::write_value(staged, out)?;
        Ok(resolution.report)
    }
}

impl<S: RealScalar, const D: usize, V: AffineValue<Scale = S> + Sync> Interpolant<S, D>
    for NearestNeighborInterpolator<'_, S, D, V>
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
        let mut workspace = self.base.workspace()?;
        workspace.search.try_reserve(&self.search)?;
        Ok(workspace)
    }
    fn workspace_capacity(&self, workspace: &ScatteredWorkspace<S>) -> Capacity {
        if self.check(workspace).is_ok() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.evaluate_report(point, policy, out, workspace)
            .map(|_| ())
    }
    fn evaluate_into_with_report<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        self.evaluate_report(point, policy, out, workspace)
    }
    fn report(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, D>, SciOxideError> {
        self.check(workspace)?;
        Ok(self
            .base
            .classify(point, policy, &mut workspace.scratch, false)?
            .report)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::tests::{options, triangulation};
    use super::*;

    #[test]
    fn search_scratch_is_included_in_byte_and_capacity_checks() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![1., 2., 3.]).unwrap();
        let model =
            NearestNeighborInterpolator::<_, 2>::with_options(&mesh, &values, options()).unwrap();
        let bytes = model.base.workspace_bytes().unwrap() + model.search.bytes::<f64>().unwrap();
        assert!(
            NearestNeighborInterpolator::<_, 2>::with_options(
                &mesh,
                &values,
                ScatteredOptions::new().with_maximum_workspace_bytes(bytes)
            )
            .is_ok()
        );
        let error = NearestNeighborInterpolator::<_, 2>::with_options(
            &mesh,
            &values,
            ScatteredOptions::new().with_maximum_workspace_bytes(bytes - 1),
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
        let mut workspace = model.base.workspace().unwrap();
        assert_eq!(model.workspace_capacity(&workspace), Capacity::NeedsGrowth);
        let mut out = 17.;
        let error = model
            .evaluate_into_with_report(
                [0.25, 0.25],
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut workspace,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(out, 17.);
        workspace.search.try_reserve(&model.search).unwrap();
        assert_eq!(model.workspace_capacity(&workspace), Capacity::Sufficient);
        let _ = model
            .evaluate_into_with_report(
                [0.25, 0.25],
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(out, 1.);
    }

    #[test]
    fn nearest_rejects_exterior_and_breaks_ties_by_index() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![10., 20., 30.]).unwrap();
        let nearest =
            NearestNeighborInterpolator::<_, 2>::with_options(&mesh, &values, options()).unwrap();
        let mut workspace = nearest.workspace().unwrap();
        let mut output = 0.;
        assert!(
            nearest
                .evaluate_into(
                    [1., 1.],
                    ExtrapolationPolicy::Reject,
                    &mut output,
                    &mut workspace
                )
                .is_err()
        );
        assert_eq!(output, 0.);
        nearest
            .evaluate_into(
                [0.5, 0.],
                ExtrapolationPolicy::Reject,
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(output, 10.);
        let report = nearest
            .report([1., 1.], ExtrapolationPolicy::Clamp, &mut workspace)
            .unwrap();
        assert!((report.effective()[0] - 0.5).abs() < 1e-12);
        assert_eq!(
            nearest
                .evaluate([1., 1.], ExtrapolationPolicy::MethodDefined)
                .unwrap(),
            20.
        );
    }
}
