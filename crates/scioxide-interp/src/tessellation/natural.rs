//! Classical two-dimensional Sibson natural-neighbor interpolation.

use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, CompensatedSum, Interval, RealScalar, SciOxideError,
};
use scioxide_spatial::{Delaunay, PredicateSign, incircle, orient2d};

use super::{
    Base, Family, Resolution, ScatteredOptions, ScatteredValues, ScatteredWorkspace, Scratch,
};
use crate::checked::intermediate;
use crate::protocol::{ExtrapolationPolicy, ExtrapolationReport, Interpolant};
use crate::values::zeroed;

/// Interpolates scattered planar data with Sibson's natural-neighbor coordinates.
///
/// The interpolant is
///
/// ```text
/// f(x) = Σᵢ wᵢ(x) vᵢ,   wᵢ(x) = Aᵢ(x) / Σⱼ Aⱼ(x),
/// ```
///
/// where `Aᵢ(x)` is the area that the Voronoi cell of `x` would take from the
/// Voronoi cell of site `pᵢ` if `x` were inserted as a new site. Only the
/// natural neighbors of `x`, the sites whose cells it would touch, have
/// `Aᵢ > 0`. The result is continuous, reproduces affine functions exactly and
/// returns the stored value at a site. On a hull edge it reduces to linear
/// interpolation between the two edge vertices.
///
/// The areas are computed in `f64` from the triangles whose circumcircles
/// contain `x`. After point location, the work is linear in the number of
/// natural neighbors and does not depend on the number of sites.
///
/// Natural-neighbor coordinates are not defined outside the convex hull.
/// There, only [`ExtrapolationPolicy::Clamp`] is accepted, which evaluates at
/// the nearest hull point; every other policy fails with `DomainError`. See
/// [Extrapolation](crate#extrapolation). A query whose computed areas do not
/// have a positive sum fails with `NumericalInstability`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, NaturalNeighborInterpolator, ScatteredValues,
/// };
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.4, 0.35];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// // f(x, y) = 1 + 3x + 2y is reproduced.
/// let values = ScatteredValues::from_input_values(&mesh, vec![1.0, 4.0, 3.0, 6.0, 1.0 + 1.2 + 0.7])?;
/// let model = NaturalNeighborInterpolator::new(&mesh, &values)?;
/// assert!((model.evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)? - 2.75).abs() < 1e-8);
/// assert!(
///     model
///         .evaluate([1.5, 0.5], ExtrapolationPolicy::MethodDefined)
///         .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. Sibson, "A brief description of natural neighbour interpolation", in
///   V. Barnett (ed.), *Interpreting Multivariate Data*, Wiley, 1981, 21–36.
#[derive(Debug)]
#[must_use]
pub struct NaturalNeighborInterpolator<'a, S: RealScalar, V: AffineValue<Scale = S> = S> {
    base: Base<'a, S, V, 2>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> Clone for NaturalNeighborInterpolator<'_, S, V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: RealScalar, V: AffineValue<Scale = S>> Copy for NaturalNeighborInterpolator<'_, S, V> {}

impl<'a, S: RealScalar, V: AffineValue<Scale = S>> NaturalNeighborInterpolator<'a, S, V> {
    /// Creates the interpolator from a planar triangulation and the values at
    /// its sites.
    ///
    /// The query workspace has no memory limit: this is [`Self::with_options`]
    /// with [`ScatteredOptions::default`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `tessellation` is not two-dimensional, or `values`
    ///   was built for a different triangulation.
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
    /// - `InvalidInput` if `tessellation` is not two-dimensional, or `values`
    ///   was built for a different triangulation.
    /// - `ResourceLimit` if the query workspace exceeds
    ///   [`ScatteredOptions::maximum_workspace_bytes`].
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    pub fn with_options(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
        options: ScatteredOptions,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            base: Base::new(tessellation, values, options, Family::Natural)?,
        })
    }

    /// Sibson evaluation at a classified query.
    fn evaluate_sibson(
        &self,
        resolution: &Resolution<S, 2>,
        output: &mut [S],
        scratch: &mut Scratch<S>,
    ) -> Result<(), SciOxideError> {
        let query = resolution.report.effective();
        let mesh = self.base.tessellation;
        let vertices = mesh.simplex(resolution.simplex)?;
        for &site in vertices {
            if mesh.points()[2 * site..2 * site + 2] == query {
                for (component, slot) in output.iter_mut().enumerate() {
                    *slot = self.base.values.component(site, component);
                }
                return Ok(());
            }
        }
        let hull_edge = if let Some(omitted) = resolution.hull_face {
            Some(omitted)
        } else {
            let mut found = None;
            for (omitted, neighbor) in mesh.neighbors(resolution.simplex)?.iter().enumerate() {
                if neighbor.is_none() {
                    let a = site_point(mesh, vertices[(omitted + 1) % 3]);
                    let b = site_point(mesh, vertices[(omitted + 2) % 3]);
                    if orient2d(&a, &b, &query)? == PredicateSign::Zero {
                        found = Some(omitted);
                        break;
                    }
                }
            }
            found
        };
        if let Some(omitted) = hull_edge {
            let a = vertices[(omitted + 1) % 3];
            let b = vertices[(omitted + 2) % 3];
            let pa = site_point(mesh, a);
            let pb = site_point(mesh, b);
            let dx = scioxide_core::Scaled::from_value(pb[0])?
                .sub(scioxide_core::Scaled::from_value(pa[0])?)?
                .abs();
            let dy = scioxide_core::Scaled::from_value(pb[1])?
                .sub(scioxide_core::Scaled::from_value(pa[1])?)?
                .abs();
            let axis = usize::from(dy.cmp_value(dx).is_gt());
            let t = scioxide_poly::backend::polynomial_basis::linear_difference_ratio(
                query[axis],
                pa[axis],
                pb[axis],
                pa[axis],
            )?
            .max(S::ZERO)
            .min(S::ONE);
            for (component, slot) in output.iter_mut().enumerate() {
                *slot = scioxide_core::checked_affine(
                    self.base.values.component(a, component),
                    self.base.values.component(b, component),
                    t,
                )?;
            }
            return Ok(());
        }
        // Two local traversals avoid both site-indexed weights and large
        // area/value products. Independent component sums retain scalar bits.
        let cavity = Cavity::new(mesh, resolution.simplex, query, self.base.domains)?;
        let sums = &mut scratch.natural_sums[..output.len()];
        let scales = &mut scratch.natural_scales[..output.len()];
        sums.fill(CompensatedSum::ZERO);
        scales.fill(0.);
        let mut denominator = CompensatedSum::<f64>::ZERO;
        cavity.areas(|site, area| {
            denominator.add(area)?;
            for (component, scale) in scales.iter_mut().enumerate() {
                *scale = scale.max(
                    self.base
                        .values
                        .component(site, component)
                        .try_to_f64_exact()?
                        .abs(),
                );
            }
            Ok(())
        })?;
        let total = denominator.total()?;
        if total <= 0. {
            return Err(unstable("natural-neighbor areas have nonpositive sum"));
        }
        cavity.areas(|site, area| {
            let weight = area / total;
            for (component, (sum, &scale)) in sums.iter_mut().zip(scales.iter()).enumerate() {
                if scale > 0. {
                    sum.add(
                        weight
                            * (self
                                .base
                                .values
                                .component(site, component)
                                .try_to_f64_exact()?
                                / scale),
                    )?;
                }
            }
            Ok(())
        })?;
        for ((slot, sum), scale) in output.iter_mut().zip(sums).zip(scales) {
            *slot = intermediate(S::try_from_f64_lossy(sum.total()? * *scale)?)?;
        }

        Ok(())
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 2>
    for NaturalNeighborInterpolator<'_, S, V>
{
    type Value = V::Owned;
    type Workspace = ScatteredWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 2] {
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
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.evaluate_into_with_report(point, policy, out, workspace)
            .map(|_| ())
    }
    fn evaluate_into_with_report<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, 2>, SciOxideError> {
        let components = self.base.values.components;
        scioxide_core::check_value_shape(out, components)?;
        self.base.check(workspace)?;
        let scratch = &mut workspace.scratch;
        let resolution = self.base.classify(point, policy, scratch, true)?;
        let staged = &mut workspace.output[..components];
        self.evaluate_sibson(&resolution, staged, scratch)?;
        scioxide_core::write_value(staged, out)?;
        Ok(resolution.report)
    }
    fn report(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<ExtrapolationReport<S, 2>, SciOxideError> {
        self.base.check(workspace)?;
        Ok(self
            .base
            .classify(point, policy, &mut workspace.scratch, true)?
            .report)
    }
}

fn unstable(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::IllConditioned, detail)
}

fn site_point<S: RealScalar>(mesh: &Delaunay<S>, site: usize) -> [S; 2] {
    [mesh.points()[site * 2], mesh.points()[site * 2 + 1]]
}

/// A counterclockwise directed edge, with its triangle on the left.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Edge {
    triangle: usize,
    from: usize,
    to: usize,
}

/// Read-only local traversal. All geometry is query-relative and uniformly scaled.
struct Cavity<'a, S: RealScalar> {
    mesh: &'a Delaunay<S>,
    query: [S; 2],
    scale: f64,
    native: bool,
    first: Edge,
}

impl<'a, S: RealScalar> Cavity<'a, S> {
    fn new(
        mesh: &'a Delaunay<S>,
        simplex: usize,
        query: [S; 2],
        domains: [Interval<S>; 2],
    ) -> Result<Self, SciOxideError> {
        let v = mesh.simplex(simplex)?;
        let orientation = orient2d(
            &site_point(mesh, v[0]),
            &site_point(mesh, v[1]),
            &site_point(mesh, v[2]),
        )?;
        let (from, to) = if orientation == PredicateSign::Positive {
            (v[0], v[1])
        } else {
            (v[1], v[0])
        };
        let mut scale = 0_f64;
        for &site in v {
            for (x, q) in site_point(mesh, site).into_iter().zip(query) {
                scale = scale.max((x.try_to_f64_exact()? - q.try_to_f64_exact()?).abs());
            }
        }
        if !scale.is_finite() {
            scale = query.iter().try_fold(0_f64, |scale, &x| {
                Ok::<_, SciOxideError>(scale.max(x.try_to_f64_exact()?.abs()))
            })?;
            for &site in v {
                for x in site_point(mesh, site) {
                    scale = scale.max(x.try_to_f64_exact()?.abs());
                }
            }
        }
        if scale == 0. {
            return Err(unstable(
                "unrepresentable natural-neighbor coordinate scale",
            ));
        }
        // Domain endpoints bound every cavity site. Admit raw differences once,
        // leaving the ordinary local traversal's original arithmetic intact.
        let native = query
            .into_iter()
            .zip(domains)
            .try_fold(true, |admitted, (q, domain)| {
                let q = q.try_to_f64_exact()?;
                Ok::<_, SciOxideError>(
                    admitted
                        && (domain.lower().try_to_f64_exact()? - q).is_finite()
                        && (domain.upper().try_to_f64_exact()? - q).is_finite(),
                )
            })?;
        let mut cavity = Self {
            mesh,
            query,
            scale,
            native,
            first: Edge {
                triangle: simplex,
                from,
                to,
            },
        };
        // The cavity is a triangulated polygon with no interior site. Its dual
        // is a tree; an oriented edge contour needs no visited set or stack.
        let mut edge = cavity.first;
        let limit = mesh.simplex_count().saturating_mul(3);
        for _ in 0..limit {
            if let Some(next) = cavity.conflict_across(edge)? {
                edge = cavity.turn(Edge {
                    triangle: next,
                    from: edge.to,
                    to: edge.from,
                })?;
            } else {
                cavity.first = edge;
                return Ok(cavity);
            }
        }
        Err(unstable(
            "natural-neighbor cavity traversal did not reach a boundary",
        ))
    }

    fn local(&self, site: usize) -> Result<[f64; 2], SciOxideError> {
        let p = site_point(self.mesh, site);
        if self.native {
            return Ok([
                (p[0].try_to_f64_exact()? - self.query[0].try_to_f64_exact()?) / self.scale,
                (p[1].try_to_f64_exact()? - self.query[1].try_to_f64_exact()?) / self.scale,
            ]);
        }
        Ok([
            scioxide_poly::backend::polynomial_basis::normalized_linear_difference(
                p[0].try_to_f64_exact()?,
                self.query[0].try_to_f64_exact()?,
                self.scale,
            )?,
            scioxide_poly::backend::polynomial_basis::normalized_linear_difference(
                p[1].try_to_f64_exact()?,
                self.query[1].try_to_f64_exact()?,
                self.scale,
            )?,
        ])
    }

    fn turn(&self, edge: Edge) -> Result<Edge, SciOxideError> {
        let third = self
            .mesh
            .simplex(edge.triangle)?
            .iter()
            .copied()
            .find(|&v| v != edge.from && v != edge.to)
            .ok_or_else(|| unstable("invalid natural-neighbor triangle edge"))?;
        Ok(Edge {
            triangle: edge.triangle,
            from: edge.to,
            to: third,
        })
    }

    fn conflict_across(&self, edge: Edge) -> Result<Option<usize>, SciOxideError> {
        let v = self.mesh.simplex(edge.triangle)?;
        let opposite = v
            .iter()
            .position(|&v| v != edge.from && v != edge.to)
            .ok_or_else(|| unstable("invalid natural-neighbor triangle edge"))?;
        let Some(next) = self.mesh.neighbors(edge.triangle)?[opposite] else {
            return Ok(None);
        };
        let w = self.mesh.simplex(next)?;
        let a = site_point(self.mesh, w[0]);
        let b = site_point(self.mesh, w[1]);
        let c = site_point(self.mesh, w[2]);
        let orientation = orient2d(&a, &b, &c)?;
        let circle = incircle(&a, &b, &c, &self.query)?;
        Ok((circle == PredicateSign::Zero || circle == orientation).then_some(next))
    }

    fn old_center(&self, triangle: usize) -> Result<[f64; 2], SciOxideError> {
        let v = self.mesh.simplex(triangle)?;
        let a = self.local(v[0])?;
        let b = self.local(v[1])?;
        let c = self.local(v[2])?;
        let center = circumcenter([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]])?;
        Ok([a[0] + center[0], a[1] + center[1]])
    }

    fn new_center(&self, edge: Edge) -> Result<[f64; 2], SciOxideError> {
        circumcenter(self.local(edge.from)?, self.local(edge.to)?)
    }

    /// Emit one stolen area per boundary vertex in cyclic order. Rotating the
    /// incident cavity fans visits each removed triangle a constant number of times.
    fn areas(
        &self,
        mut emit: impl FnMut(usize, f64) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let mut boundary = self.first;
        let mut previous_new = self.new_center(boundary)?;
        let mut visits = 0usize;
        let limit = self.mesh.simplex_count().saturating_mul(6);
        loop {
            let origin = self.old_center(boundary.triangle)?;
            let mut last = [0.; 2];
            let mut area = CompensatedSum::<f64>::ZERO;
            let mut perimeter = 0_f64;
            let mut magnitude = origin[0].abs().max(origin[1].abs());
            let mut edge = self.turn(boundary)?;
            while let Some(next) = self.conflict_across(edge)? {
                let center = self.old_center(next)?;
                let current = [center[0] - origin[0], center[1] - origin[1]];
                area.add(cross(last, current))?;
                perimeter += (current[0] - last[0]).hypot(current[1] - last[1]);
                magnitude = magnitude.max(center[0].abs()).max(center[1].abs());
                last = current;
                edge = self.turn(Edge {
                    triangle: next,
                    from: edge.to,
                    to: edge.from,
                })?;
                visits += 1;
                if visits > limit {
                    return Err(unstable("natural-neighbor cavity fan did not close"));
                }
            }
            let next_new = self.new_center(edge)?;
            let current = [next_new[0] - origin[0], next_new[1] - origin[1]];
            let previous = [previous_new[0] - origin[0], previous_new[1] - origin[1]];
            area.add(cross(last, current))?;
            area.add(cross(current, previous))?;
            perimeter += (current[0] - last[0]).hypot(current[1] - last[1])
                + (previous[0] - current[0]).hypot(previous[1] - current[1])
                + previous[0].hypot(previous[1]);
            magnitude = magnitude
                .max(next_new[0].abs())
                .max(next_new[1].abs())
                .max(previous_new[0].abs())
                .max(previous_new[1].abs());
            let area = -0.5 * area.total()?;
            // A query adjacent to a site can create overlap slivers smaller
            // than circumcenter rounding. Perturbing polygon vertices by O(eps)
            // changes area by O(eps * coordinate magnitude * perimeter).
            // Project only these rounding-size negatives to the known
            // nonnegative area cone; a larger negative is a breakdown.
            let roundoff = 32. * f64::EPSILON * magnitude * perimeter;
            if area < -roundoff {
                return Err(unstable("natural-neighbor stolen area is negative"));
            }
            emit(boundary.to, area.max(0.))?;
            boundary = edge;
            previous_new = next_new;
            visits += 1;
            if boundary == self.first {
                return Ok(());
            }
            if visits > limit {
                return Err(unstable("natural-neighbor boundary did not close"));
            }
        }
    }
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0].mul_add(b[1], -a[1] * b[0])
}

/// Circumcenter of (0,a,b), without absolute-coordinate squared norms.
fn circumcenter(a: [f64; 2], b: [f64; 2]) -> Result<[f64; 2], SciOxideError> {
    let determinant = 2. * cross(a, b);
    let aa = a[0].mul_add(a[0], a[1] * a[1]);
    let bb = b[0].mul_add(b[0], b[1] * b[1]);
    let center = [
        (aa * b[1] - bb * a[1]) / determinant,
        (bb * a[0] - aa * b[0]) / determinant,
    ];
    if center.iter().any(|x| !x.is_finite()) {
        return Err(unstable("unrepresentable natural-neighbor circumcenter"));
    }
    Ok(center)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::tests::{options, triangulation};
    use super::*;

    #[test]
    fn sibson_coordinates_reproduce_affine_values_and_hull_edges() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1., 0.4, 0.35]);
        let values =
            ScatteredValues::from_input_values(&mesh, vec![1.0, 4.0, 3.0, 6.0, 2.9]).unwrap();
        let model = NaturalNeighborInterpolator::with_options(&mesh, &values, options()).unwrap();
        let mut workspace = model.workspace().unwrap();
        let mut output = [0.];
        model
            .evaluate_into(
                [0.25, 0.5],
                ExtrapolationPolicy::Reject,
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert!((output[0] - 2.75).abs() < 1e-8, "{}", output[0]);
        model
            .evaluate_into(
                [0.5, 0.],
                ExtrapolationPolicy::Reject,
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert!((output[0] - 2.5).abs() < 1e-10);
        let before = output;
        assert!(
            model
                .evaluate_into(
                    [1.1, 0.5],
                    ExtrapolationPolicy::MethodDefined,
                    &mut output,
                    &mut workspace,
                )
                .is_err()
        );
        assert_eq!(output, before);
        let report = model
            .report([1.1, 0.5], ExtrapolationPolicy::Clamp, &mut workspace)
            .unwrap();
        assert!((report.effective()[0] - 1.).abs() < 1e-12);
        model
            .evaluate_into(
                [1.1, 0.5],
                ExtrapolationPolicy::Clamp,
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert!((output[0] - 5.).abs() < 1e-10);
    }
}
