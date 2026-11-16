//! Interpolation of scattered data over a Delaunay triangulation.
//!
//! Each interpolator borrows one [`Delaunay`] triangulation and one
//! [`ScatteredValues`] built for it, evaluates through [`crate::Interpolant`]
//! with a [`ScatteredWorkspace`], and reports coordinates against the convex
//! hull of the sites.

mod clough_tocher;
mod linear;
mod natural;
mod nearest;

pub use clough_tocher::{
    CloughTocherInterpolator, CloughTocherOptions, CloughTocherReport, CloughTocherTermination,
};
pub use linear::SimplexLinearInterpolator;
pub use natural::NaturalNeighborInterpolator;
pub use nearest::NearestNeighborInterpolator;

use alloc::vec::Vec;

use scioxide_core::backend::scaled::scaled_magnitude_le_tolerance;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AffineValue, Capacity, CompensatedSum, ExecutionContext, Interval, NdView, RealScalar, Scaled,
    SciOxideError, Tolerance,
};
use scioxide_linalg::{
    DirectSolveOptions, HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting,
    HouseholderWorkspaceRequest, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace,
    MatrixView, RankPolicy, VectorView, VectorViewMut,
};
use scioxide_spatial::{Delaunay, GeometryWorkspace, MeshToken, PointSet, SearchWorkspace};

use crate::checked::intermediate;
use crate::error::invalid;
use crate::geometry::{bounds, norm};
use crate::protocol::{ExtrapolationPolicy, ExtrapolationReport, Interpolant};
use crate::values::flatten;
use scioxide_poly::backend::polynomial_basis::normalized_linear_difference;

/// Sets the memory limit for the query workspace of a scattered-data interpolator.
///
/// The limit applies to [`SimplexLinearInterpolator`],
/// [`NaturalNeighborInterpolator`], [`NearestNeighborInterpolator`] and
/// [`CloughTocherInterpolator`]. It bounds the bytes of one
/// [`ScatteredWorkspace`], including the point-location scratch of the
/// triangulation. Constructors compare the exact workspace size with the limit
/// and fail with `ResourceLimit` if it is exceeded. The default is unlimited.
///
/// ```
/// use scioxide_interp::ScatteredOptions;
/// let options = ScatteredOptions::new().with_maximum_workspace_bytes(1 << 20);
/// assert_eq!(options.maximum_workspace_bytes(), 1 << 20);
/// assert_eq!(
///     ScatteredOptions::default().maximum_workspace_bytes(),
///     usize::MAX
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct ScatteredOptions {
    maximum_workspace_bytes: usize,
}

impl ScatteredOptions {
    /// Returns options with no workspace limit.
    pub const fn new() -> Self {
        Self {
            maximum_workspace_bytes: usize::MAX,
        }
    }

    /// Sets the maximum size of one query workspace in bytes.
    pub const fn with_maximum_workspace_bytes(mut self, bytes: usize) -> Self {
        self.maximum_workspace_bytes = bytes;
        self
    }

    /// Returns the maximum size of one query workspace in bytes.
    pub const fn maximum_workspace_bytes(&self) -> usize {
        self.maximum_workspace_bytes
    }
}

impl Default for ScatteredOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores the data values at the sites of one Delaunay triangulation.
///
/// Values may be real or complex scalars, arrays or vectors. They are stored
/// site by site, each site's real components contiguous, in the site order of
/// the triangulation. The triangulation merges exactly duplicated input points
/// into one site, so a site may correspond to several inputs. The values are
/// tied to the triangulation they were built for: an interpolator that pairs
/// them with a different or rebuilt triangulation fails with `InvalidInput`.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::ScatteredValues;
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// let values = ScatteredValues::from_input_values(
///     &mesh,
///     vec![
///         Complex::new(1.0, 0.0),
///         Complex::new(2.0, 1.0),
///         Complex::new(3.0, -1.0),
///     ],
/// )?;
/// assert_eq!((values.sites(), values.components()), (3, 2));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ScatteredValues<S: RealScalar, V: AffineValue<Scale = S> = S> {
    token: MeshToken,
    /// `[site][component]` components.
    values: Vec<S>,
    sites: usize,
    components: usize,
    template: V,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> ScatteredValues<S, V> {
    /// Builds values from one value per input point, in the order the points
    /// were given to the triangulation.
    ///
    /// Input points that the triangulation merged into one site must carry
    /// identical values.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of values differs from the number
    ///   of input points, or the values differ in shape.
    /// - `NonFinite` if a value component is NaN or infinite.
    /// - `InvalidInput` if duplicate input points carry different values.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized
    ///   or allocated.
    pub fn from_input_values(
        tessellation: &Delaunay<S>,
        values: Vec<V>,
    ) -> Result<Self, SciOxideError> {
        let mapping = tessellation.input_mapping();
        let flat = flatten(values, mapping.len())?;
        let sites = site_count(tessellation);
        let components = flat.dimension;
        let mut canonical = try_filled(
            checked_mul(
                sites,
                components,
                scioxide_core::quantity!("scattered_values"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("scattered_values"),
        )?;
        canonicalize(
            mapping,
            sites,
            components,
            |input, component| Ok(flat.components[input * components + component]),
            &mut canonical,
        )?;
        Ok(Self {
            token: tessellation.mesh_token().clone(),
            values: canonical,
            sites,
            components,
            template: flat.template,
        })
    }

    /// Builds values from one value per site of the triangulation, in its
    /// site order.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of values differs from the number
    ///   of sites, or the values differ in shape.
    /// - `NonFinite` if a value component is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized
    ///   or allocated.
    pub fn from_site_values(
        tessellation: &Delaunay<S>,
        values: Vec<V>,
    ) -> Result<Self, SciOxideError> {
        let sites = site_count(tessellation);
        let flat = flatten(values, sites)?;
        Ok(Self {
            token: tessellation.mesh_token().clone(),
            values: flat.components,
            sites,
            components: flat.dimension,
            template: flat.template,
        })
    }

    /// Returns the number of sites.
    pub const fn sites(&self) -> usize {
        self.sites
    }

    /// Returns the number of real components of one value.
    pub const fn components(&self) -> usize {
        self.components
    }

    /// Returns all value components, site by site.
    #[must_use]
    pub fn values(&self) -> &[S] {
        &self.values
    }

    fn ensure(&self, tessellation: &Delaunay<S>) -> Result<(), SciOxideError> {
        // Site indices are only meaningful for the exact snapshot they were
        // built against; the mesh token rejects pairing values with another
        // (or rebuilt) tessellation.
        self.token.ensure_compatible(tessellation.mesh_token())
    }

    fn component(&self, site: usize, component: usize) -> S {
        self.values[site * self.components + component]
    }
}

impl<S: RealScalar> ScatteredValues<S, Vec<S>> {
    /// Builds real vector values from a two-dimensional view of shape
    /// `[input point, component]`, in the order the points were given to the
    /// triangulation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the view is not two-dimensional, its first extent
    ///   differs from the number of input points, or duplicate input points
    ///   carry different values.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized
    ///   or allocated.
    ///
    /// ```
    /// use scioxide_core::{NdView, StridedLayout};
    /// use scioxide_interp::ScatteredValues;
    /// use scioxide_spatial::{Delaunay, PointSet};
    /// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
    /// let points = PointSet::from_rows(&coordinates, 2)?;
    /// let mesh = Delaunay::new(&points)?;
    /// let samples = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
    /// let view = NdView::new(&samples, StridedLayout::new(&[3, 2], &[2, 1])?)?;
    /// let values = ScatteredValues::from_input_view(&mesh, view)?;
    /// assert_eq!(values.components(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_input_view(
        tessellation: &Delaunay<S>,
        values: NdView<'_, '_, S>,
    ) -> Result<Self, SciOxideError> {
        let shape = values.layout().shape().extents();
        if shape.len() != 2 || shape[0] != tessellation.input_mapping().len() {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "input value view must have shape [original_site, component]",
            ));
        }
        let sites = site_count(tessellation);
        let components = shape[1];
        let mut canonical = try_filled(
            checked_mul(
                sites,
                components,
                scioxide_core::quantity!("scattered_values"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("scattered_values"),
        )?;
        canonicalize(
            tessellation.input_mapping(),
            sites,
            components,
            |input, component| {
                let entry = *values.get([input, component])?;
                if entry.is_finite() {
                    Ok(entry)
                } else {
                    Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Input,
                        "sample components must be finite",
                    ))
                }
            },
            &mut canonical,
        )?;
        Ok(Self {
            token: tessellation.mesh_token().clone(),
            values: canonical,
            sites,
            components,
            template: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("value_template"),
            )?,
        })
    }
}

/// Number of sites of a tessellation after duplicate merging.
fn site_count<S: RealScalar>(tessellation: &Delaunay<S>) -> usize {
    tessellation.points().len() / tessellation.dimension()
}

/// Scatter input-ordered components to sites, rejecting conflicting duplicates.
fn canonicalize<S: RealScalar>(
    mapping: &[usize],
    sites: usize,
    components: usize,
    mut entry: impl FnMut(usize, usize) -> Result<S, SciOxideError>,
    canonical: &mut [S],
) -> Result<(), SciOxideError> {
    // The tessellation merges exact duplicate input points into one canonical
    // site, and `input_mapping` sends each original input to its site. A second
    // input landing on an already-filled site must agree, since one site cannot
    // carry two values.
    let mut seen = try_filled(
        sites,
        false,
        scioxide_core::quantity!("scattered_duplicate_map"),
    )?;
    for (input, &site) in mapping.iter().enumerate() {
        for component in 0..components {
            let value = entry(input, component)?;
            let slot = &mut canonical[site * components + component];
            if seen[site] && *slot != value {
                return Err(invalid(
                    scioxide_core::argument!("values"),
                    "duplicate sites carry conflicting values",
                ));
            }
            *slot = value;
        }
        seen[site] = true;
    }
    Ok(())
}

/// Interpolates scattered data linearly on the Delaunay simplices of the sites and
/// returns the values at `queries`.
///
/// `points` holds the sites in `D` dimensions and `values` one value per
/// site, in the same order; sites that coincide must carry identical values.
/// The interpolant is that of [`SimplexLinearInterpolator`]: inside a simplex
/// with vertices `p₀, …, p_D` and values `v₀, …, v_D`,
///
/// ```text
/// f(x) = Σᵢ λᵢ(x) vᵢ,
/// ```
///
/// where `λᵢ(x)` are the barycentric coordinates of `x`. It is continuous,
/// reproduces affine functions exactly and returns the stored value at a
/// site.
///
/// The call builds the [`Delaunay`] triangulation of the sites with the
/// default geometry options, then the [`ScatteredValues`] and the
/// interpolator, and evaluates every query with
/// [`ExtrapolationPolicy::Reject`]: the domain is the convex hull of the
/// sites. The triangulation is built on every call; to evaluate several sets
/// of queries, keep a [`SimplexLinearInterpolator`] instead.
///
/// # Errors
///
/// - `NonFinite` if a coordinate, value or query is NaN or infinite.
/// - `Singular` if the sites do not span `D` dimensions, for example when
///   there are fewer than `D + 1` affinely independent sites.
/// - `InvalidInput` if `D` is zero, or coinciding sites carry different
///   values.
/// - `DimensionMismatch` if `values` and `points` differ in length, or the
///   values differ in shape.
/// - `DomainError` if a query lies outside the convex hull of the sites.
/// - `ResourceLimit` if the triangulation exceeds the default geometry
///   limits.
/// - `NumericalInstability` if a query falls in a nearly flat simplex that has
///   no usable barycentric frame.
/// - `AllocationFailure` if storage cannot be reserved.
///
/// # Examples
///
/// ```
/// use scioxide_interp::interpolate_scattered;
/// let points: [[f64; 2]; 5] = [[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.4, 0.35]];
/// // f(x, y) = 1 + 3x + 2y is reproduced.
/// let values = points.iter().map(|[x, y]| 1. + 3. * x + 2. * y).collect();
/// let out = interpolate_scattered(&points, values, &[[0.25, 0.5], [0.9, 0.9]])?;
/// assert!((out[0] - 2.75).abs() < 1e-12 && (out[1] - 5.5).abs() < 1e-12);
/// let outside = interpolate_scattered(&points, vec![0.0; 5], &[[1.5, 0.5]]);
/// assert!(outside.is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn interpolate_scattered<S, V, const D: usize>(
    points: &[[S; D]],
    values: Vec<V>,
    queries: &[[S; D]],
) -> Result<Vec<V::Owned>, SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S> + Sync,
{
    let mesh = Delaunay::new(&PointSet::from_points(points)?)?;
    let values = ScatteredValues::from_input_values(&mesh, values)?;
    SimplexLinearInterpolator::<S, D, V>::new(&mesh, &values)?
        .evaluate_many(queries, ExtrapolationPolicy::Reject)
}

/// Holds the scratch storage for queries of the scattered-data interpolators.
///
/// Create it with [`crate::Interpolant::workspace`]. One workspace can serve
/// several of [`SimplexLinearInterpolator`], [`NaturalNeighborInterpolator`],
/// [`NearestNeighborInterpolator`] and [`CloughTocherInterpolator`] if it is
/// large enough for each, which [`crate::Interpolant::workspace_capacity`]
/// checks. A workspace that is too small is rejected with `DimensionMismatch`.
/// A workspace stays usable after a failed query. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// ```
/// use scioxide_interp::{ExtrapolationPolicy, Interpolant, ScatteredValues, SimplexLinearInterpolator};
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// let values = ScatteredValues::from_input_values(&mesh, vec![1.0, 2.0, 3.0])?;
/// let model = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values)?;
/// let mut workspace = model.workspace()?;
/// let mut out = 0.0;
/// model.evaluate_into(
///     [0.25, 0.25],
///     ExtrapolationPolicy::Reject,
///     &mut out,
///     &mut workspace,
/// )?;
/// assert!((out - 1.75).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct ScatteredWorkspace<S: RealScalar> {
    /// Value or gradient components, copied to the output once per query.
    output: Vec<S>,
    comparison: Vec<S>,
    scratch: Scratch<S>,
    search: SearchWorkspace<S>,
}

/// Geometry and kernel scratch, disjoint from the staged output.
#[derive(Debug)]
struct Scratch<S: RealScalar> {
    geometry: GeometryWorkspace,
    barycentric: Vec<S>,
    staging: Vec<S>,
    projected: Vec<S>,
    candidate: Vec<S>,
    matrix: Vec<f64>,
    projector: Vec<f64>,
    normals: Vec<f64>,
    rhs: Vec<f64>,
    coefficients: Vec<f64>,
    query: Vec<f64>,
    indices: Vec<usize>,
    subset: Vec<usize>,
    projection_qr: Option<ProjectionQr>,
    natural_sums: Vec<CompensatedSum<f64>>,
    natural_scales: Vec<f64>,
    /// Sites and normalized weights of the closest face found by
    /// [`closest_face`]; `face_count` entries are valid.
    face_sites: Vec<usize>,
    face_weights: Vec<f64>,
    face_count: usize,
}

/// Buffer lengths one interpolator requires of a [`ScatteredWorkspace`].
#[derive(Debug, Clone, Copy)]
struct Lengths {
    output: usize,
    simplex: usize,
    axes: usize,
    system: usize,
    derivative: usize,
    indices: usize,
    components: usize,
    /// Closest-face record for degenerate-simplex fallbacks.
    face: usize,
}

/// Select only the query buffers consumed by a family.
#[derive(Debug, Clone, Copy)]
enum Family {
    Nearest,
    Linear,
    Natural,
    CloughTocher,
}

/// Validated binding of a tessellation and its values, shared by every family.
#[derive(Debug)]
struct Base<'a, S: RealScalar, V: AffineValue<Scale = S>, const D: usize> {
    tessellation: &'a Delaunay<S>,
    values: &'a ScatteredValues<S, V>,
    domains: [Interval<S>; D],
    lengths: Lengths,
}

// Manual impls: a borrowed binding is copyable whatever the value type.
impl<S: RealScalar, V: AffineValue<Scale = S>, const D: usize> Clone for Base<'_, S, V, D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: RealScalar, V: AffineValue<Scale = S>, const D: usize> Copy for Base<'_, S, V, D> {}

impl<'a, S: RealScalar, V: AffineValue<Scale = S>, const D: usize> Base<'a, S, V, D> {
    /// Shared constructor validation.
    fn new(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
        options: ScatteredOptions,
        family: Family,
    ) -> Result<Self, SciOxideError> {
        if D == 0 || tessellation.dimension() != D {
            return Err(invalid(
                scioxide_core::argument!("tessellation"),
                "interpolator dimension does not match tessellation",
            ));
        }
        values.ensure(tessellation)?;
        let points = tessellation.points();
        let domains = bounds(
            points
                .chunks_exact(D)
                .map(|point| core::array::from_fn(|axis| point[axis])),
            scioxide_core::argument!("tessellation"),
        )?;
        let quantity = scioxide_core::quantity!("scattered_workspace");
        let lengths = Lengths {
            output: checked_mul(
                if matches!(family, Family::Linear | Family::CloughTocher) {
                    D
                } else {
                    1
                },
                values.components,
                quantity,
            )?,
            simplex: match family {
                Family::Linear => checked_mul(checked_add(D, 1, quantity)?, D, quantity)?,
                Family::CloughTocher => 6,
                Family::Nearest | Family::Natural => 0,
            },
            axes: D,
            system: D,
            derivative: if matches!(family, Family::Linear | Family::CloughTocher) {
                D
            } else {
                0
            },
            indices: checked_add(D, 1, quantity)?,
            components: if matches!(family, Family::Natural) {
                values.components
            } else {
                0
            },
            face: if matches!(family, Family::Linear | Family::CloughTocher) {
                checked_add(D, 1, quantity)?
            } else {
                0
            },
        };
        let base = Self {
            tessellation,
            values,
            domains,
            lengths,
        };
        let required = base.workspace_bytes()?;
        if required > options.maximum_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("scattered_workspace_bytes"),
                required,
                options.maximum_workspace_bytes,
            ));
        }
        Ok(base)
    }

    /// Validated buffer lengths, independent of query and workspace history.
    const fn lengths(&self) -> Lengths {
        self.lengths
    }

    /// Exact query workspace bytes, including geometry scratch.
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("workspace_bytes");
        let lengths = self.lengths();
        let geometry = self.tessellation.workspace_requirements(())?.bytes();
        let scalars = [
            lengths.output,
            if lengths.derivative > 0 {
                lengths.output
            } else {
                0
            },
            lengths.simplex,
            lengths.simplex,
        ]
        .into_iter()
        .chain([lengths.axes, lengths.axes])
        .try_fold(0, |sum, count| checked_add(sum, count, quantity))?;
        let floats = checked_add(
            checked_add(
                checked_mul(lengths.system, lengths.system, quantity)?,
                checked_mul(
                    2,
                    checked_mul(lengths.derivative, lengths.derivative, quantity)?,
                    quantity,
                )?,
                quantity,
            )?,
            checked_add(
                checked_add(
                    checked_mul(lengths.system, 2, quantity)?,
                    lengths.axes,
                    quantity,
                )?,
                lengths.face,
                quantity,
            )?,
            quantity,
        )?;
        let indices = checked_add(
            checked_add(lengths.indices, lengths.axes, quantity)?,
            lengths.face,
            quantity,
        )?;
        [
            geometry,
            checked_mul(scalars, core::mem::size_of::<S>(), quantity)?,
            checked_mul(
                checked_add(
                    floats,
                    checked_mul(3, lengths.components, quantity)?,
                    quantity,
                )?,
                core::mem::size_of::<f64>(),
                quantity,
            )?,
            checked_mul(indices, core::mem::size_of::<usize>(), quantity)?,
            ProjectionQr::bytes(D)?,
        ]
        .into_iter()
        .try_fold(0, |sum, bytes| checked_add(sum, bytes, quantity))
    }

    /// Allocate query scratch from this binding's requirements.
    fn workspace(&self) -> Result<ScatteredWorkspace<S>, SciOxideError> {
        let lengths = self.lengths();
        let quantity = scioxide_core::quantity!("scattered_workspace");
        let mut geometry = GeometryWorkspace::new();
        geometry.try_reserve(&self.tessellation.workspace_requirements(())?)?;
        Ok(ScatteredWorkspace {
            search: SearchWorkspace::new(),
            output: try_filled(lengths.output, S::ZERO, quantity)?,
            comparison: try_filled(
                if lengths.derivative > 0 {
                    lengths.output
                } else {
                    0
                },
                S::ZERO,
                quantity,
            )?,
            scratch: Scratch {
                geometry,
                barycentric: try_filled(lengths.simplex, S::ZERO, quantity)?,
                staging: try_filled(lengths.simplex, S::ZERO, quantity)?,
                projected: try_filled(lengths.axes, S::ZERO, quantity)?,
                candidate: try_filled(lengths.axes, S::ZERO, quantity)?,
                matrix: try_filled(lengths.system * lengths.system, 0.0, quantity)?,
                projector: try_filled(lengths.derivative * lengths.derivative, 0.0, quantity)?,
                normals: try_filled(lengths.derivative * lengths.derivative, 0.0, quantity)?,
                rhs: try_filled(lengths.system, 0.0, quantity)?,
                coefficients: try_filled(lengths.system, 0.0, quantity)?,
                query: try_filled(lengths.axes, 0.0, quantity)?,
                indices: try_filled(lengths.indices, 0, quantity)?,
                subset: try_filled(lengths.axes, 0, quantity)?,
                projection_qr: if D > 3 {
                    Some(ProjectionQr::new(D)?)
                } else {
                    None
                },
                natural_sums: try_filled(lengths.components, CompensatedSum::ZERO, quantity)?,
                natural_scales: try_filled(lengths.components, 0., quantity)?,
                face_sites: try_filled(lengths.face, 0, quantity)?,
                face_weights: try_filled(lengths.face, 0., quantity)?,
                face_count: 0,
            },
        })
    }

    /// First buffer shorter than required, as `(required, available)`.
    fn shortfall(&self, workspace: &ScatteredWorkspace<S>) -> Option<(usize, usize)> {
        let lengths = self.lengths();
        let scratch = &workspace.scratch;
        let geometry = self
            .tessellation
            .workspace_requirements(())
            .and_then(|requirements| scratch.geometry.check(&requirements));
        if !matches!(geometry, Ok(Capacity::Sufficient)) {
            return Some((D, 0));
        }
        [
            (lengths.output, workspace.output.len()),
            (
                if lengths.derivative > 0 {
                    lengths.output
                } else {
                    0
                },
                workspace.comparison.len(),
            ),
            (lengths.simplex, scratch.barycentric.len()),
            (lengths.simplex, scratch.staging.len()),
            (lengths.axes, scratch.projected.len()),
            (lengths.axes, scratch.candidate.len()),
            (lengths.system * lengths.system, scratch.matrix.len()),
            (
                lengths.derivative * lengths.derivative,
                scratch.projector.len(),
            ),
            (
                lengths.derivative * lengths.derivative,
                scratch.normals.len(),
            ),
            (lengths.system, scratch.rhs.len()),
            (lengths.system, scratch.coefficients.len()),
            (lengths.axes, scratch.query.len()),
            (lengths.indices, scratch.indices.len()),
            (lengths.axes, scratch.subset.len()),
            (
                D,
                scratch.projection_qr.as_ref().map_or(3, |qr| qr.dimension),
            ),
            (lengths.components, scratch.natural_sums.len()),
            (lengths.components, scratch.natural_scales.len()),
            (lengths.face, scratch.face_sites.len()),
            (lengths.face, scratch.face_weights.len()),
        ]
        .into_iter()
        .find(|(required, available)| available < required)
    }

    /// Whether `workspace` covers this binding.
    fn capacity(&self, workspace: &ScatteredWorkspace<S>) -> Capacity {
        if self.shortfall(workspace).is_none() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }

    /// Reject an undersized workspace with `DimensionMismatch`.
    fn check(&self, workspace: &ScatteredWorkspace<S>) -> Result<(), SciOxideError> {
        match self.shortfall(workspace) {
            None => Ok(()),
            Some((required, available)) => Err(crate::error::shape(required, available)),
        }
    }

    /// Classify a query against the convex hull.
    fn classify(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        scratch: &mut Scratch<S>,
        natural: bool,
    ) -> Result<Resolution<S, D>, SciOxideError> {
        policy.validate(point)?;
        if let Some(location) = self.tessellation.find_simplex_scoped(
            &point,
            &mut scratch.geometry,
            ExecutionContext::sequential_native(),
        )? {
            return Ok(Resolution {
                report: ExtrapolationReport::interior(point),
                simplex: location.simplex,
                hull_face: None,
            });
        }
        // Exterior query. The domain is the convex hull, not the bounding box:
        // the query is projected to the nearest hull point, and the distance
        // per axis is measured to that point. Sibson coordinates have no
        // exterior continuation, so natural-neighbour queries accept only
        // `Clamp`, which evaluates at the projected point.
        if policy == ExtrapolationPolicy::Reject {
            return Err(outside());
        }
        if natural && policy != ExtrapolationPolicy::Clamp {
            return Err(outside());
        }
        let (simplex, omitted) = self.project_hull(point, scratch)?;
        let mut projected = [S::ZERO; D];
        projected.copy_from_slice(&scratch.projected[..D]);
        Ok(Resolution {
            report: policy.resolve_projected(point, projected)?,
            simplex,
            hull_face: Some(omitted),
        })
    }

    /// Nearest convex-hull simplex.
    fn project_hull(
        &self,
        point: [S; D],
        scratch: &mut Scratch<S>,
    ) -> Result<(usize, usize), SciOxideError> {
        if D >= usize::BITS as usize {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "hull projection dimension exceeds subset index capacity",
            ));
        }
        // Scan every boundary facet (a simplex face with no neighbour) for the
        // closest point to the query. The search runs in f64: it only chooses
        // a hull simplex and a projected point, while interior classification
        // stays with the exact predicates. Equal distances resolve to the
        // lowest simplex index so the result is deterministic.
        let mut best: Option<Scaled<f64>> = None;
        let mut best_simplex = None;
        for (slot, value) in scratch.query.iter_mut().zip(point) {
            *slot = value.try_to_f64_exact()?;
        }
        for simplex in 0..self.tessellation.simplex_count() {
            let vertices = self.tessellation.simplex(simplex)?;
            for (omitted, neighbor) in self.tessellation.neighbors(simplex)?.iter().enumerate() {
                if neighbor.is_some() {
                    continue;
                }
                let mut face = 0;
                for (local, &vertex) in vertices.iter().enumerate() {
                    if local != omitted {
                        scratch.indices[face] = vertex;
                        face += 1;
                    }
                }
                if let Some(distance) = closest_face(
                    self.tessellation.points(),
                    D,
                    face,
                    1..(1 << face),
                    PROJECTION_MARGIN,
                    scratch,
                )? && (best.is_none_or(|best| distance.cmp_value(best).is_lt())
                    || (Some(distance) == best
                        && best_simplex.is_none_or(|old: (usize, usize)| (simplex, omitted) < old)))
                {
                    best = Some(distance);
                    best_simplex = Some((simplex, omitted));
                    scratch.projected[..D].copy_from_slice(&scratch.candidate[..D]);
                }
            }
        }
        best_simplex.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("self"),
                "tessellation has no convex-hull boundary",
            )
        })
    }

    /// Differentiate every incident exterior projection region, comparing the
    /// composed limits before writing any component. Regions agree when each
    /// entry differs by at most `8 * (D + 2)^2 * epsilon` times both regions'
    /// rounding scales: the gradient magnitudes plus `max_i |v_i| * Σ_i |∇λ_i|`
    /// over each incident simplex (its hull face's tangential gradients for a
    /// sliver). The first region's gradient is written.
    fn clamp_gradient(
        &self,
        resolution: &Resolution<S, D>,
        output: &mut [S],
        comparison: &mut [S],
        scratch: &mut Scratch<S>,
        mut gradient: impl FnMut(
            usize,
            [S; D],
            usize,
            &mut [S],
            &mut Scratch<S>,
        ) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let point = resolution.report.original();
        let projected = resolution.report.effective();
        let dimension = S::try_from_usize_exact(D + 2)?;
        let tolerance = Tolerance::new(
            S::ZERO,
            S::try_from_usize_exact(8)? * dimension * dimension * S::EPSILON,
        )?;
        let mut first: Option<(usize, Scaled<S>)> = None;
        // Strict region weights first; a face flat within rounding, whose
        // simplices all reject the query by a rounding margin, is then
        // admitted with the projection's own weight margin.
        // The retry margin matches the scalar precision of the flatness test.
        for margin in [0., 128. * S::EPSILON.try_to_f64_exact()?] {
            if first.is_some() {
                break;
            }
            for simplex in 0..self.tessellation.simplex_count() {
                let vertices = self.tessellation.simplex(simplex)?;
                for (omitted, neighbor) in self.tessellation.neighbors(simplex)?.iter().enumerate()
                {
                    if neighbor.is_some() {
                        continue;
                    }
                    for mask in 1usize..(1usize << D) {
                        let mut face = 0;
                        for (local, &site) in vertices.iter().enumerate() {
                            if local != omitted {
                                scratch.indices[face] = site;
                                face += 1;
                            }
                        }
                        for (slot, x) in scratch.query.iter_mut().zip(point) {
                            *slot = x.try_to_f64_exact()?;
                        }
                        if closest_face(
                            self.tessellation.points(),
                            D,
                            D,
                            mask..mask + 1,
                            margin,
                            scratch,
                        )?
                        .is_none()
                            || !self.coincident(&scratch.candidate[..D], projected)?
                        {
                            continue;
                        }
                        let count = mask.count_ones() as usize;
                        if !self.is_hull_face(count, scratch)? {
                            continue;
                        }
                        self.tangent_projector(count, scratch)?;
                        gradient(simplex, projected, count, comparison, scratch)?;
                        let components = self.values.components;
                        // Store one component's projected gradient in the spare rhs.
                        for component in 0..components {
                            for axis in 0..D {
                                scratch.rhs[axis] = (0..D).try_fold(0_f64, |sum, other| {
                                    Ok::<_, SciOxideError>(
                                        scratch.projector[axis * D + other].mul_add(
                                            comparison[other * components + component]
                                                .try_to_f64_exact()?,
                                            sum,
                                        ),
                                    )
                                })?;
                            }
                            for axis in 0..D {
                                comparison[axis * components + component] =
                                    intermediate(S::try_from_f64_lossy(scratch.rhs[axis])?)?;
                            }
                        }
                        let spread = self.gradient_spread(simplex, count, scratch)?;
                        let Some((first_simplex, first_spread)) = first else {
                            output.copy_from_slice(comparison);
                            first = Some((simplex, spread));
                            continue;
                        };
                        for component in 0..components {
                            let scale = self
                                .value_magnitude(first_simplex, component)?
                                .mul(first_spread)?
                                .add(self.value_magnitude(simplex, component)?.mul(spread)?)?;
                            for axis in 0..D {
                                let entry = axis * components + component;
                                let a = Scaled::from_value(output[entry])?;
                                let b = Scaled::from_value(comparison[entry])?;
                                if !scaled_magnitude_le_tolerance(
                                    a.sub(b)?,
                                    &tolerance,
                                    scale.add(a.abs())?.add(b.abs())?,
                                )? {
                                    return Err(SciOxideError::domain_error(
                                        scioxide_core::argument!("point"),
                                        scioxide_core::DomainReason::Undefined,
                                        "exterior projection gradient limits disagree",
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        if first.is_some() {
            Ok(())
        } else {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "no active hull projection region",
            ))
        }
    }

    /// Whether a region's projection is the global projection up to rounding.
    /// Each face projects in its own centered frame, so equal exact points can
    /// differ by `16 * epsilon` times the coordinate and hull extent scale.
    fn coincident(&self, candidate: &[S], projected: [S; D]) -> Result<bool, SciOxideError> {
        let tolerance = S::try_from_usize_exact(16)? * S::EPSILON;
        for (axis, (&a, b)) in candidate.iter().zip(projected).enumerate() {
            let domain = self.domains[axis];
            let scale = Scaled::from_value(a.abs().max(b.abs()))?.add(
                Scaled::from_value(domain.upper())?.sub(Scaled::from_value(domain.lower())?)?,
            )?;
            let difference = Scaled::from_value(a)?.sub(Scaled::from_value(b)?)?;
            if !difference.abs_le(scale.mul(Scaled::from_value(tolerance)?)?) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `Σ_i |∇λ_i|_1` over one simplex's barycentric gradients.
    /// A sliver's spread is that of the region face whose gradient it uses.
    fn gradient_spread(
        &self,
        simplex: usize,
        count: usize,
        scratch: &mut Scratch<S>,
    ) -> Result<Scaled<S>, SciOxideError> {
        let entries = (D + 1) * D;
        if let Err(error) = self.tessellation.barycentric_gradients_into(
            simplex,
            &mut scratch.barycentric[..entries],
            &mut scratch.staging[..entries],
        ) {
            return if degenerate(&error) {
                self.face_spread(count, scratch)
            } else {
                Err(error)
            };
        }
        scratch.barycentric[..entries]
            .iter()
            .try_fold(Scaled::ZERO, |sum, &x| {
                sum.add(Scaled::from_value(x.abs())?)
            })
    }

    /// Largest magnitude of one value component over a simplex's vertices.
    fn value_magnitude(
        &self,
        simplex: usize,
        component: usize,
    ) -> Result<Scaled<S>, SciOxideError> {
        let largest = self
            .tessellation
            .simplex(simplex)?
            .iter()
            .fold(S::ZERO, |largest, &site| {
                largest.max(self.values.component(site, component).abs())
            });
        Scaled::from_value(largest)
    }

    /// A true polytope face is the intersection of supporting facet planes.
    /// Their normal rank excludes artificial subfaces inside a flat hull patch.
    fn is_hull_face(&self, count: usize, scratch: &mut Scratch<S>) -> Result<bool, SciOxideError> {
        let mut rank = 0;
        let points = self.tessellation.points();
        for simplex in 0..self.tessellation.simplex_count() {
            let vertices = self.tessellation.simplex(simplex)?;
            for (omitted, neighbor) in self.tessellation.neighbors(simplex)?.iter().enumerate() {
                if neighbor.is_some() {
                    continue;
                }
                // The facet's own vertices fix its unit normal (in `rhs`), so a
                // sliver simplex's ill-conditioned apex never enters. Exact
                // predicates decide containment for D <= 3; a facet without a
                // normal (itself thin) cannot support any face and is skipped.
                let mut scale = 0.;
                if D > 3 {
                    match self.facet_normal(vertices, omitted, scratch) {
                        Ok(facet) => scale = facet,
                        Err(error) if degenerate(&error) => continue,
                        Err(error) => return Err(error),
                    }
                }
                let mut contains = true;
                for index in 0..count {
                    let site = scratch.subset[index];
                    let p = &points[site * D..(site + 1) * D];
                    // Exact low-dimensional predicates preserve coplanar seams.
                    let coplanar = if D == 2 {
                        let a = vertices[(omitted + 1) % 3];
                        let b = vertices[(omitted + 2) % 3];
                        scioxide_spatial::orient2d(
                            &[points[a * D], points[a * D + 1]],
                            &[points[b * D], points[b * D + 1]],
                            &[p[0], p[1]],
                        )? == scioxide_spatial::PredicateSign::Zero
                    } else if D == 3 {
                        let ids: [usize; 3] =
                            core::array::from_fn(|i| vertices[(omitted + 1 + i) % 4]);
                        let a =
                            ids.map(|id| [points[id * D], points[id * D + 1], points[id * D + 2]]);
                        scioxide_spatial::orient3d(&a[0], &a[1], &a[2], &[p[0], p[1], p[2]])?
                            == scioxide_spatial::PredicateSign::Zero
                    } else {
                        // Distance from the facet plane in its scaled frame.
                        let anchor = vertices[(omitted + 1) % (D + 1)];
                        let mut offset = 0_f64;
                        for axis in 0..D {
                            offset = normalized_linear_difference(
                                p[axis].try_to_f64_exact()?,
                                points[anchor * D + axis].try_to_f64_exact()?,
                                scale,
                            )?
                            .mul_add(scratch.rhs[axis], offset);
                        }
                        offset.abs() <= 128. * S::EPSILON.try_to_f64_exact()?
                    };
                    if !coplanar {
                        contains = false;
                        break;
                    }
                }
                if !contains {
                    continue;
                }
                if D <= 3 {
                    match self.facet_normal(vertices, omitted, scratch) {
                        Ok(_) => {}
                        Err(error) if degenerate(&error) => continue,
                        Err(error) => return Err(error),
                    }
                }
                for _ in 0..2 {
                    for row in scratch.normals[..rank * D].chunks_exact(D) {
                        let dot = row
                            .iter()
                            .zip(&scratch.rhs)
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                        for (x, &n) in scratch.rhs.iter_mut().zip(row) {
                            *x = (-dot).mul_add(n, *x);
                        }
                    }
                }
                let length = norm(scratch.rhs[..D].iter().copied().map(Ok))?;
                if length > 128. * S::EPSILON.try_to_f64_exact()? {
                    for axis in 0..D {
                        scratch.normals[rank * D + axis] = scratch.rhs[axis] / length;
                    }
                    rank += 1;
                    if rank == D - count + 1 {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Unit normal of the facet of `vertices` omitting `omitted`, written to
    /// `scratch.rhs[..D]`; returns the facet's coordinate scale. Facet edges
    /// are orthonormalized by twice-iterated Gram–Schmidt in `scratch.matrix`;
    /// the normal is the best-conditioned coordinate direction's complement.
    fn facet_normal(
        &self,
        vertices: &[usize],
        omitted: usize,
        scratch: &mut Scratch<S>,
    ) -> Result<f64, SciOxideError> {
        let points = self.tessellation.points();
        let x = |site: usize, axis: usize| points[site * D + axis].try_to_f64_exact();
        let anchor = vertices[(omitted + 1) % (D + 1)];
        let facet = |column: usize| vertices[(omitted + 2 + column) % (D + 1)];
        let mut scale = 0_f64;
        for column in 0..D - 1 {
            for axis in 0..D {
                scale = scale.max((x(facet(column), axis)? - x(anchor, axis)?).abs());
            }
        }
        if !(scale.is_finite() && scale > 0.) {
            return Err(SciOxideError::singular(None));
        }
        let rank_loss = || {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "hull facet lost rank",
            )
        };
        let tangents = &mut scratch.matrix;
        for column in 0..D - 1 {
            for axis in 0..D {
                tangents[column * D + axis] =
                    normalized_linear_difference(x(facet(column), axis)?, x(anchor, axis)?, scale)?;
            }
            orthogonalize(tangents, column, D);
            let length = norm((0..D).map(|axis| Ok(tangents[column * D + axis])))?;
            if length.is_nan() || length <= 128. * f64::EPSILON {
                return Err(rank_loss());
            }
            for axis in 0..D {
                tangents[column * D + axis] /= length;
            }
        }
        // The coordinate direction least represented in the facet.
        let mut best = (0, f64::NEG_INFINITY);
        for axis in 0..D {
            let residual = 1.
                - (0..D - 1)
                    .map(|c| tangents[c * D + axis].powi(2))
                    .sum::<f64>();
            if residual > best.1 {
                best = (axis, residual);
            }
        }
        for axis in 0..D {
            tangents[(D - 1) * D + axis] = if axis == best.0 { 1. } else { 0. };
        }
        orthogonalize(tangents, D - 1, D);
        let length = norm((0..D).map(|axis| Ok(tangents[(D - 1) * D + axis])))?;
        if length.is_nan() || length <= 128. * f64::EPSILON {
            return Err(rank_loss());
        }
        for axis in 0..D {
            scratch.rhs[axis] = tangents[(D - 1) * D + axis] / length;
        }
        Ok(scale)
    }

    /// Complete the supporting-normal frame with tangent directions.
    /// All triangulations of one flat hull face then use identical arithmetic.
    fn tangent_projector(
        &self,
        count: usize,
        scratch: &mut Scratch<S>,
    ) -> Result<(), SciOxideError> {
        scratch.projector.fill(0.);
        if count == 1 {
            return Ok(());
        }
        let rank = D - count + 1;
        if D == 2 {
            let x = -scratch.normals[1];
            let y = scratch.normals[0];
            scratch.projector[..4].copy_from_slice(&[x * x, x * y, x * y, y * y]);
            return Ok(());
        }
        let mut filled = rank;
        for coordinate in 0..D {
            scratch.rhs.fill(0.);
            scratch.rhs[coordinate] = 1.;
            for _ in 0..2 {
                for row in scratch.normals[..filled * D].chunks_exact(D) {
                    let dot = row
                        .iter()
                        .zip(&scratch.rhs)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    for (x, &n) in scratch.rhs.iter_mut().zip(row) {
                        *x = (-dot).mul_add(n, *x);
                    }
                }
            }
            let length = norm(scratch.rhs[..D].iter().copied().map(Ok))?;
            if length <= 128. * f64::EPSILON {
                continue;
            }
            for axis in 0..D {
                scratch.normals[filled * D + axis] = scratch.rhs[axis] / length;
            }
            for row in 0..D {
                for col in 0..D {
                    scratch.projector[row * D + col] = scratch.normals[filled * D + row].mul_add(
                        scratch.normals[filled * D + col],
                        scratch.projector[row * D + col],
                    );
                }
            }
            filled += 1;
            if filled == D {
                return Ok(());
            }
        }
        Err(SciOxideError::singular(None))
    }

    /// Record the proper face of `simplex` closest to `point`. A simplex whose
    /// affine frame is unavailable (a sliver) is evaluated on this face only
    /// when the query lies within `32 * epsilon` of the hull extent plus the
    /// query magnitude from it: the result is then the exact interpolant at a
    /// point that close to the query. Farther queries, including continuation
    /// beyond the hull, keep the frame's `NumericalInstability`.
    pub(super) fn closest_simplex_face(
        &self,
        simplex: usize,
        point: [S; D],
        scratch: &mut Scratch<S>,
    ) -> Result<(), SciOxideError> {
        scratch.indices[..=D].copy_from_slice(self.tessellation.simplex(simplex)?);
        for (slot, x) in scratch.query.iter_mut().zip(point) {
            *slot = x.try_to_f64_exact()?;
        }
        scratch.face_count = 0;
        // Every nonempty proper subset; the full mask is the degenerate simplex.
        let found = closest_face(
            self.tessellation.points(),
            D,
            D + 1,
            1..(1 << (D + 1)) - 1,
            PROJECTION_MARGIN,
            scratch,
        )?;
        let extent = self
            .domains
            .iter()
            .try_fold(Scaled::<f64>::ZERO, |extent, domain| {
                let width = Scaled::from_value(domain.upper().try_to_f64_exact()?)?
                    .sub(Scaled::from_value(domain.lower().try_to_f64_exact()?)?)?;
                Ok::<_, SciOxideError>(if width.abs_cmp(extent).is_gt() {
                    width
                } else {
                    extent
                })
            })?;
        let magnitude = scratch.query[..D].iter().fold(0_f64, |m, x| m.max(x.abs()));
        let limit = extent
            .add(Scaled::from_value(magnitude)?)?
            .mul(Scaled::from_value(32. * S::EPSILON.try_to_f64_exact()?)?)?;
        match found {
            Some(distance) if scratch.face_count > 0 && !distance.cmp_value(limit).is_gt() => {
                Ok(())
            }
            _ => Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "simplex approximation is ill-conditioned beyond rounding of its faces",
            )),
        }
    }

    /// Blend values over the recorded closest face.
    fn face_values(&self, output: &mut [S], scratch: &Scratch<S>) -> Result<(), SciOxideError> {
        let count = scratch.face_count;
        for (component, slot) in output.iter_mut().enumerate() {
            *slot = scratch.face_sites[..count]
                .iter()
                .zip(&scratch.face_weights[..count])
                .try_fold(S::ZERO, |sum, (&site, &weight)| {
                    intermediate(
                        S::try_from_f64_lossy(weight)?
                            .mul_add(self.values.component(site, component), sum),
                    )
                })?;
        }
        Ok(())
    }

    /// Orthonormal frame of the face `scratch.subset[..count]`: its scaled
    /// edge matrix `E = T R` by twice-iterated Gram–Schmidt, with `T` in
    /// `scratch.matrix` (column-major) and upper-triangular `R` in
    /// `scratch.normals` (row-major). Returns the coordinate scale, or `None`
    /// for a vertex.
    fn face_frame(
        &self,
        count: usize,
        scratch: &mut Scratch<S>,
    ) -> Result<Option<f64>, SciOxideError> {
        let columns = count.saturating_sub(1);
        if columns == 0 {
            return Ok(None);
        }
        let points = self.tessellation.points();
        let origin = scratch.subset[0];
        let coordinate = |site: usize, axis: usize| points[site * D + axis].try_to_f64_exact();
        let mut scale = 0_f64;
        for &site in &scratch.subset[1..count] {
            for axis in 0..D {
                scale = scale.max((coordinate(site, axis)? - coordinate(origin, axis)?).abs());
            }
        }
        if !(scale.is_finite() && scale > 0.) {
            return Err(SciOxideError::singular(None));
        }
        let (tangents, factor) = (&mut scratch.matrix, &mut scratch.normals);
        factor[..columns * columns].fill(0.);
        for column in 0..columns {
            let site = scratch.subset[column + 1];
            for axis in 0..D {
                tangents[column * D + axis] = normalized_linear_difference(
                    coordinate(site, axis)?,
                    coordinate(origin, axis)?,
                    scale,
                )?;
            }
            for _ in 0..2 {
                for previous in 0..column {
                    let dot = (0..D)
                        .map(|axis| tangents[previous * D + axis] * tangents[column * D + axis])
                        .sum::<f64>();
                    factor[previous * columns + column] += dot;
                    for axis in 0..D {
                        tangents[column * D + axis] = (-dot)
                            .mul_add(tangents[previous * D + axis], tangents[column * D + axis]);
                    }
                }
            }
            let length = norm((0..D).map(|axis| Ok(tangents[column * D + axis])))?;
            if length.is_nan() || length <= 128. * f64::EPSILON {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::IllConditioned,
                    "hull face lost rank",
                ));
            }
            factor[column * columns + column] = length;
            for axis in 0..D {
                tangents[column * D + axis] /= length;
            }
        }
        Ok(Some(scale))
    }

    /// Solve `R^T a = b` by forward substitution in place in `rhs[..columns]`.
    fn face_solve(columns: usize, factor: &[f64], rhs: &mut [f64]) {
        for row in 0..columns {
            let mut value = rhs[row];
            for previous in 0..row {
                value = (-factor[previous * columns + row]).mul_add(rhs[previous], value);
            }
            rhs[row] = value / factor[row * columns + row];
        }
    }

    /// Tangential gradient of the linear interpolant on the face
    /// `scratch.subset[..count]`, written axis-major like simplex gradients:
    /// `g = T R^{-T} Δv / scale` satisfies `E^T g = Δv` and lies in the face.
    pub(super) fn face_linear_gradient(
        &self,
        count: usize,
        output: &mut [S],
        scratch: &mut Scratch<S>,
    ) -> Result<(), SciOxideError> {
        let components = self.values.components;
        output.fill(S::ZERO);
        let Some(scale) = self.face_frame(count, scratch)? else {
            return Ok(());
        };
        let columns = count - 1;
        let origin = scratch.subset[0];
        for component in 0..components {
            let base = self
                .values
                .component(origin, component)
                .try_to_f64_exact()?;
            for row in 0..columns {
                scratch.rhs[row] = self
                    .values
                    .component(scratch.subset[row + 1], component)
                    .try_to_f64_exact()?
                    - base;
            }
            Self::face_solve(columns, &scratch.normals, &mut scratch.rhs);
            for axis in 0..D {
                let gradient = (0..columns).fold(0_f64, |sum, column| {
                    scratch.rhs[column].mul_add(scratch.matrix[column * D + axis], sum)
                }) / scale;
                output[axis * components + component] =
                    intermediate(S::try_from_f64_lossy(gradient)?)?;
            }
        }
        Ok(())
    }

    /// `Σ_i |∇_F λ_i|_1` of the face's tangential barycentric gradients.
    fn face_spread(
        &self,
        count: usize,
        scratch: &mut Scratch<S>,
    ) -> Result<Scaled<S>, SciOxideError> {
        let Some(scale) = self.face_frame(count, scratch)? else {
            return Ok(Scaled::ZERO);
        };
        let columns = count - 1;
        scratch.coefficients.fill(0.);
        let mut spread = 0_f64;
        for vertex in 0..columns {
            scratch.rhs[..columns].fill(0.);
            scratch.rhs[vertex] = 1.;
            Self::face_solve(columns, &scratch.normals, &mut scratch.rhs);
            for axis in 0..D {
                let gradient = (0..columns).fold(0_f64, |sum, column| {
                    scratch.rhs[column].mul_add(scratch.matrix[column * D + axis], sum)
                }) / scale;
                spread += gradient.abs();
                scratch.coefficients[axis] += gradient;
            }
        }
        // The origin's coordinate gradient is minus the others' sum.
        spread += scratch.coefficients[..D]
            .iter()
            .map(|x| x.abs())
            .sum::<f64>();
        Scaled::from_value(S::try_from_f64_lossy(spread)?)
    }

    fn nodal_site(&self, simplex: usize, point: &[S; D]) -> Result<Option<usize>, SciOxideError> {
        Ok(self
            .tessellation
            .simplex(simplex)?
            .iter()
            .copied()
            .find(|&site| self.tessellation.points()[site * D..(site + 1) * D] == *point))
    }

    /// Simplex-linear evaluation at a classified query.
    fn interpolate_resolved(
        &self,
        resolution: &Resolution<S, D>,
        output: &mut [S],
        scratch: &mut Scratch<S>,
    ) -> Result<(), SciOxideError> {
        // Exact hits on a site return its values verbatim instead of relying
        // on barycentric weights that are one/zero only up to rounding.
        if let Some(site) = self.nodal_site(resolution.simplex, &resolution.report.effective())? {
            for (component, slot) in output.iter_mut().enumerate() {
                *slot = self.values.component(site, component);
            }
            return Ok(());
        }
        if let Err(error) = self.tessellation.barycentric_into(
            resolution.simplex,
            &resolution.report.effective(),
            &mut scratch.barycentric[..D + 1],
            &mut scratch.staging[..D + 1],
        ) {
            if !degenerate(&error) {
                return Err(error);
            }
            self.closest_simplex_face(resolution.simplex, resolution.report.effective(), scratch)?;
            return self.face_values(output, scratch);
        }
        let vertices = self.tessellation.simplex(resolution.simplex)?;
        for (component, slot) in output.iter_mut().enumerate() {
            *slot = vertices
                .iter()
                .zip(&scratch.barycentric[..D + 1])
                .try_fold(S::ZERO, |sum, (&site, &weight)| {
                    intermediate(weight.mul_add(self.values.component(site, component), sum))
                })?;
        }
        Ok(())
    }
}

/// A classified query: its hull report and the simplex used for evaluation.
#[derive(Debug, Clone, Copy)]
struct Resolution<S: RealScalar, const D: usize> {
    report: ExtrapolationReport<S, D>,
    simplex: usize,
    /// Opposite vertex of the selected hull facet for projected exterior points.
    hull_face: Option<usize>,
}

/// Remove from column `column` of the column-major `vectors` (length `d`
/// each) its components along the preceding orthonormal columns, twice.
fn orthogonalize(vectors: &mut [f64], column: usize, d: usize) {
    for _ in 0..2 {
        for previous in 0..column {
            let dot = (0..d)
                .map(|axis| vectors[previous * d + axis] * vectors[column * d + axis])
                .sum::<f64>();
            for axis in 0..d {
                vectors[column * d + axis] =
                    (-dot).mul_add(vectors[previous * d + axis], vectors[column * d + axis]);
            }
        }
    }
}

/// An unavailable affine frame: spatial's documented per-simplex
/// conditioning failure, recovered by evaluating on the closest face.
pub(super) fn degenerate(error: &SciOxideError) -> bool {
    error.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability
}

/// Exterior query rejection.
fn outside() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("point"),
        scioxide_core::DomainReason::OutsideDomain,
        "query lies outside the convex hull or the method has no exterior continuation",
    )
}

/// Reserved QR storage for higher-dimensional face projections only.
#[derive(Debug)]
struct ProjectionQr {
    dimension: usize,
    factor: HouseholderDecomposition<f64>,
    workspace: LinalgWorkspace<'static, f64>,
}

impl ProjectionQr {
    fn requirements(d: usize) -> Result<LinalgRequirements, SciOxideError> {
        let options = LinalgExecutionOptions::new();
        let mut requirements = LinalgRequirements::default();
        for columns in 1..d {
            for request in [
                HouseholderWorkspaceRequest::factor(
                    d,
                    columns,
                    HouseholderOrientation::Qr,
                    &options,
                ),
                HouseholderWorkspaceRequest::solve(
                    d,
                    columns,
                    HouseholderOrientation::Qr,
                    &options,
                ),
            ] {
                requirements = requirements.max(
                    &HouseholderDecomposition::<f64>::workspace_requirements(&request)?,
                );
            }
        }
        Ok(requirements)
    }
    fn bytes(d: usize) -> Result<usize, SciOxideError> {
        if d <= 3 {
            return Ok(0);
        }
        checked_add(
            HouseholderDecomposition::<f64>::factor_storage_bytes(
                d,
                d - 1,
                HouseholderOrientation::Qr,
            )?,
            Self::requirements(d)?.required_bytes::<f64>()?,
            scioxide_core::quantity!("projection_workspace"),
        )
    }
    fn new(dimension: usize) -> Result<Self, SciOxideError> {
        let mut factor = HouseholderDecomposition::empty();
        factor.try_reserve(dimension, dimension - 1, HouseholderOrientation::Qr)?;
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::requirements(dimension)?)?;
        Ok(Self {
            dimension,
            factor,
            workspace,
        })
    }
}

/// Barycentric weight margin that accepts a face projection despite rounding.
const PROJECTION_MARGIN: f64 = 128. * f64::EPSILON;

/// Closest point by active subfaces in a centered, uniformly scaled frame.
/// Faces with a weight below `-margin` are skipped.
/// Segments and triangles use small QR closed forms; larger faces use linalg QR.
fn closest_face<S: RealScalar>(
    points: &[S],
    dimension: usize,
    vertex_count: usize,
    masks: core::ops::Range<usize>,
    margin: f64,
    scratch: &mut Scratch<S>,
) -> Result<Option<Scaled<f64>>, SciOxideError> {
    if vertex_count >= usize::BITS as usize {
        return Err(invalid(
            scioxide_core::argument!("self"),
            "hull projection dimension exceeds subset index capacity",
        ));
    }
    let mut best: Option<Scaled<f64>> = None;
    for mask in masks {
        let count = mask.count_ones() as usize;
        let columns = count - 1;
        let mut selected = 0;
        for local in 0..vertex_count {
            if mask & (1 << local) != 0 {
                scratch.subset[selected] = scratch.indices[local];
                selected += 1;
            }
        }
        let origin = scratch.subset[0] * dimension;
        let mut scale = 0_f64;
        for &site in &scratch.subset[1..count] {
            for axis in 0..dimension {
                scale = scale.max(
                    (points[site * dimension + axis].try_to_f64_exact()?
                        - points[origin + axis].try_to_f64_exact()?)
                    .abs(),
                );
            }
        }
        if !scale.is_finite() {
            scale = scratch.subset[..count]
                .iter()
                .try_fold(0_f64, |scale, &site| {
                    points[site * dimension..(site + 1) * dimension]
                        .iter()
                        .try_fold(scale, |scale, &x| {
                            Ok::<_, SciOxideError>(scale.max(x.try_to_f64_exact()?.abs()))
                        })
                })?;
        }
        if columns == 0 {
            scale = (0..dimension).try_fold(1_f64, |scale, axis| {
                Ok::<_, SciOxideError>(
                    scale
                        .max(scratch.query[axis].abs())
                        .max(points[origin + axis].try_to_f64_exact()?.abs()),
                )
            })?;
        }
        if scale == 0. {
            return Err(SciOxideError::singular(None));
        }
        for axis in 0..dimension {
            scratch.coefficients[axis] = normalized_linear_difference(
                scratch.query[axis],
                points[origin + axis].try_to_f64_exact()?,
                scale,
            )?;
            for column in 0..columns {
                scratch.matrix[axis * columns + column] = normalized_linear_difference(
                    points[scratch.subset[column + 1] * dimension + axis].try_to_f64_exact()?,
                    points[origin + axis].try_to_f64_exact()?,
                    scale,
                )?;
            }
        }
        scratch.rhs[..count].fill(0.);
        if columns == 1 {
            let mut numerator = 0.;
            let mut denominator = 0.;
            for axis in 0..dimension {
                let edge = scratch.matrix[axis];
                numerator = edge.mul_add(scratch.coefficients[axis], numerator);
                denominator = edge.mul_add(edge, denominator);
            }
            scratch.rhs[1] = numerator / denominator;
        } else if columns == 2 && dimension <= 3 {
            // Orthogonalize the two edge columns rather than square the
            // conditioning with a Gram determinant.
            let first_norm = norm((0..dimension).map(|axis| Ok(scratch.matrix[axis * 2])))?;
            let mut cross = 0.;
            let mut first_rhs = 0.;
            for axis in 0..dimension {
                scratch.matrix[axis * 2] /= first_norm;
                cross = scratch.matrix[axis * 2].mul_add(scratch.matrix[axis * 2 + 1], cross);
                first_rhs = scratch.matrix[axis * 2].mul_add(scratch.coefficients[axis], first_rhs);
            }
            for axis in 0..dimension {
                scratch.matrix[axis * 2 + 1] =
                    (-cross).mul_add(scratch.matrix[axis * 2], scratch.matrix[axis * 2 + 1]);
            }
            let orthogonal_norm: f64 =
                norm((0..dimension).map(|axis| Ok(scratch.matrix[axis * 2 + 1])))?;
            if orthogonal_norm == 0. {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::IllConditioned,
                    "hull face projection lost rank",
                ));
            }
            let mut second_rhs = 0.;
            for axis in 0..dimension {
                second_rhs = (scratch.matrix[axis * 2 + 1] / orthogonal_norm)
                    .mul_add(scratch.coefficients[axis], second_rhs);
            }
            scratch.rhs[2] = second_rhs / orthogonal_norm;
            scratch.rhs[1] = (-cross).mul_add(scratch.rhs[2], first_rhs) / first_norm;
        } else if columns > 1 {
            let qr = scratch.projection_qr.as_mut().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "missing higher-dimensional projection QR storage",
                )
            })?;
            HouseholderDecomposition::factor_into(
                MatrixView::from_slice(dimension, columns, &scratch.matrix[..dimension * columns])?,
                HouseholderOrientation::Qr,
                HouseholderPivoting::Column,
                &LinalgExecutionOptions::new(),
                &mut qr.factor,
                &mut qr.workspace,
                ExecutionContext::sequential_native(),
            )?;
            if !qr.factor.analyze_rank(RankPolicy::Exact)?.is_full_rank() {
                return Err(SciOxideError::singular(None));
            }
            let report = qr.factor.solve_into(
                VectorView::from_slice(&scratch.coefficients[..dimension]),
                &DirectSolveOptions::new(RankPolicy::Exact),
                VectorViewMut::from_slice(&mut scratch.rhs[1..count]),
                &mut qr.workspace,
                ExecutionContext::sequential_native(),
            )?;
            if !report.termination().is_success() {
                return Err(SciOxideError::singular(None));
            }
        }
        scratch.rhs[0] = 1. - scratch.rhs[1..count].iter().sum::<f64>();
        if scratch.rhs[..count]
            .iter()
            .any(|&weight| !weight.is_finite() || weight < -margin)
        {
            continue;
        }
        let total: f64 = scratch.rhs[..count].iter().map(|&w| w.max(0.)).sum();
        let mut distance = 0_f64;
        for axis in 0..dimension {
            let anchor = points[origin + axis].try_to_f64_exact()?;
            let mut displacement = 0.;
            for row in 1..count {
                displacement = (scratch.rhs[row].max(0.) / total).mul_add(
                    normalized_linear_difference(
                        points[scratch.subset[row] * dimension + axis].try_to_f64_exact()?,
                        anchor,
                        scale,
                    )?,
                    displacement,
                );
            }
            scratch.coefficients[axis] = Scaled::from_value(anchor)?
                .add(Scaled::from_value(displacement)?.mul(Scaled::from_value(scale)?)?)?
                .try_value()?;
            distance = distance.hypot(
                normalized_linear_difference(scratch.query[axis], anchor, scale)? - displacement,
            );
        }
        let distance = Scaled::from_value(distance)?.mul(Scaled::from_value(scale)?)?;
        if best.is_none_or(|best| distance.cmp_value(best).is_lt()) {
            best = Some(distance);
            for axis in 0..dimension {
                scratch.candidate[axis] = S::try_from_f64_lossy(scratch.coefficients[axis])?;
            }
            if scratch.face_sites.len() >= count {
                scratch.face_sites[..count].copy_from_slice(&scratch.subset[..count]);
                for (slot, &weight) in scratch.face_weights.iter_mut().zip(&scratch.rhs[..count]) {
                    *slot = weight.max(0.) / total;
                }
                scratch.face_count = count;
            }
        }
    }
    Ok(best)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;
    use scioxide_core::{SciOxideErrorView, StridedLayout};
    use scioxide_spatial::{GeometryLimits, GeometryOptions, PointSet};

    /// Query workspace limit used by the family tests.
    pub(crate) fn options() -> ScatteredOptions {
        ScatteredOptions::new().with_maximum_workspace_bytes(8 * 1024 * 1024)
    }

    /// Planar Delaunay snapshot of interleaved coordinates.
    pub(crate) fn triangulation(values: &[f64]) -> Delaunay<f64> {
        let shape = [values.len() / 2, 2];
        let strides = [2, 1];
        let layout = StridedLayout::new(&shape, &strides).unwrap();
        let view = NdView::new(values, layout).unwrap();
        let points = PointSet::new(view).unwrap();
        let geometry = GeometryOptions::new().with_limits(
            GeometryLimits::new()
                .with_max_dimension(3)
                .with_max_points(values.len() / 2)
                .with_max_facets(128)
                .with_max_predicate_terms(100_000)
                .with_max_storage_bytes(8 * 1024 * 1024),
        );
        Delaunay::with_options(&points, geometry).unwrap()
    }

    #[test]
    fn value_bindings_reject_conflicts_shapes_and_other_snapshots() {
        let coordinates = [0., 0., 1., 0., 0., 1., 0., 0.];
        let mesh = triangulation(&coordinates);
        assert!(ScatteredValues::from_input_values(&mesh, vec![1., 2., 3., 4.]).is_err());
        let values = ScatteredValues::from_input_values(&mesh, vec![1., 2., 3., 1.]).unwrap();
        assert_eq!(values.sites(), 3);
        assert!(
            matches!(ScatteredValues::from_site_values(&mesh, vec![1., 2.]), Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }))
        );
        assert!(
            matches!(ScatteredValues::from_site_values(&mesh, vec![1., f64::NAN, 2.]), Err(error) if matches!(error.view(), SciOxideErrorView::NonFinite { .. }))
        );
        let other = triangulation(&coordinates);
        assert!(
            SimplexLinearInterpolator::<_, 2>::with_options(&other, &values, options()).is_err()
        );
        let converted = mesh.triangle_mesh().unwrap();
        assert!(converted.mesh_token().is_compatible_with(mesh.mesh_token()));
        let view_data = [1., 2., 3., 1.];
        let view = NdView::new(&view_data, StridedLayout::new(&[4, 1], &[1, 1]).unwrap()).unwrap();
        let from_view = ScatteredValues::from_input_view(&mesh, view).unwrap();
        assert_eq!(from_view.values(), values.values());
        let cloned = values.clone();
        assert_eq!(cloned.values(), values.values());
    }

    #[test]
    fn workspace_limits_and_capacity_are_checked() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![1., 2., 3., 4.]).unwrap();
        let limited = ScatteredOptions::new().with_maximum_workspace_bytes(16);
        assert!(
            matches!(SimplexLinearInterpolator::<_, 2>::with_options(&mesh, &values, limited), Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
        );
        let base = Base::<f64, f64, 2>::new(&mesh, &values, options(), Family::Linear).unwrap();
        let workspace = base.workspace().unwrap();
        assert_eq!(base.capacity(&workspace), Capacity::Sufficient);
        let small_mesh = triangulation(&[0., 0., 1., 0., 0., 1.]);
        let small_values =
            ScatteredValues::from_site_values(&small_mesh, vec![1., 2., 3.]).unwrap();
        let small = Base::<f64, f64, 2>::new(&small_mesh, &small_values, options(), Family::Linear)
            .unwrap();
        let mut small_workspace = small.workspace().unwrap();
        // Equal dimensions/components need equal scratch regardless of site count.
        assert_eq!(base.capacity(&small_workspace), Capacity::Sufficient);
        small_workspace.output.pop();
        assert_eq!(base.capacity(&small_workspace), Capacity::NeedsGrowth);
        assert!(
            matches!(base.check(&small_workspace), Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. }))
        );
        assert_eq!(small.capacity(&workspace), Capacity::Sufficient);
    }

    #[test]
    fn family_buffers_and_exact_byte_limits() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![[1., 2., 3.]; 4]).unwrap();
        for family in [
            Family::Nearest,
            Family::Linear,
            Family::Natural,
            Family::CloughTocher,
        ] {
            let base = Base::<_, _, 2>::new(&mesh, &values, options(), family).unwrap();
            let bytes = base.workspace_bytes().unwrap();
            assert!(
                Base::<_, _, 2>::new(
                    &mesh,
                    &values,
                    ScatteredOptions::new().with_maximum_workspace_bytes(bytes),
                    family
                )
                .is_ok()
            );
            assert!(matches!(Base::<_, _, 2>::new(&mesh, &values,
                ScatteredOptions::new().with_maximum_workspace_bytes(bytes - 1), family),
                Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. })));
            let workspace = base.workspace().unwrap();
            assert_eq!(
                workspace.output.len(),
                if matches!(family, Family::Linear | Family::CloughTocher) {
                    6
                } else {
                    3
                }
            );
            assert_eq!(
                workspace.scratch.barycentric.len(),
                match family {
                    Family::Linear => 6,
                    Family::CloughTocher => 6,
                    _ => 0,
                }
            );
            assert_eq!(
                workspace.scratch.natural_sums.len(),
                if matches!(family, Family::Natural) {
                    3
                } else {
                    0
                }
            );
            assert_eq!(base.capacity(&workspace), Capacity::Sufficient);
        }
    }
}
