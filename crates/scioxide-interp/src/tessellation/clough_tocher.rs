//! Piecewise-cubic, globally C1 Clough–Tocher interpolation in two dimensions.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Capacity, ConvergenceReason, ExecutionContext,
    Interval, PartialResultTrust, ProblemToken, RealScalar, SciOxideError, SolveBudgets,
    SolveContext, SolveIdentity, SolveScope, SolveState, Tolerance,
};
use scioxide_spatial::Delaunay;

use super::{Base, Family, ScatteredOptions, ScatteredValues, ScatteredWorkspace, Scratch};
use crate::checked::intermediate;
use crate::error::{admit_iteration, invalid, iteration_denied, require_proposal, unconverged};
use crate::geometry::norm;
use crate::protocol::{Differentiable, ExtrapolationPolicy, ExtrapolationReport, Interpolant};
use crate::values::zeroed;
use scioxide_poly::backend::polynomial_basis::normalized_linear_difference;

/// Identity of the vertex-gradient estimation in solve accounting and errors.
const ALGORITHM: AlgorithmId = scioxide_core::algorithm_id!("interp", "clough-tocher", 1);

/// Sets the tolerance, iteration limit and memory limits of a [`CloughTocherInterpolator`].
///
/// The defaults are a gradient tolerance of `√ε` (absolute and relative), at
/// most 400 Gauss–Seidel sweeps, no limit on preparation memory, unbounded
/// budgets and no limit on the query workspace. Construction counts one
/// proposal toward the budgets, then one iteration per sweep; see
/// [solve accounting](scioxide_core#solve-accounting).
///
/// ```
/// use scioxide_core::{Budget, BudgetKind, SolveBudgets, Tolerance};
/// use scioxide_interp::CloughTocherOptions;
/// let options = CloughTocherOptions::<f64>::new()
///     .with_gradient_tolerance(Tolerance::new(1e-12, 1e-12)?)
///     .with_maximum_iterations(64)
///     .with_budgets(
///         SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(32)?),
///     );
/// assert_eq!(options.maximum_iterations(), 64);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CloughTocherOptions<S: RealScalar> {
    scattered: ScatteredOptions,
    gradient_tolerance: Tolerance<S>,
    maximum_iterations: usize,
    maximum_preparation_bytes: usize,
    budgets: SolveBudgets,
}

impl<S: RealScalar> CloughTocherOptions<S> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            scattered: ScatteredOptions::new(),
            gradient_tolerance: Tolerance::from_sqrt_epsilon(),
            maximum_iterations: 400,
            maximum_preparation_bytes: usize::MAX,
            budgets: SolveBudgets::unbounded(),
        }
    }

    /// Sets the query workspace limit shared with the other scattered-data
    /// interpolators.
    pub const fn with_scattered(mut self, scattered: ScatteredOptions) -> Self {
        self.scattered = scattered;
        self
    }

    /// Sets the stopping tolerance of the gradient sweeps.
    ///
    /// A value component converges when the largest change of a vertex
    /// gradient in one sweep, relative to `max(1, |∇f|)`, meets the tolerance.
    /// Gradients are measured in coordinates divided by the larger side of the
    /// bounding box, so the test does not depend on the coordinate units.
    pub const fn with_gradient_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.gradient_tolerance = tolerance;
        self
    }

    /// Sets the maximum number of Gauss–Seidel sweeps per value component,
    /// which must be positive.
    pub const fn with_maximum_iterations(mut self, iterations: usize) -> Self {
        self.maximum_iterations = iterations;
        self
    }

    /// Sets the maximum memory in bytes for the vertex gradients and the
    /// construction scratch.
    pub const fn with_maximum_preparation_bytes(mut self, bytes: usize) -> Self {
        self.maximum_preparation_bytes = bytes;
        self
    }

    /// Sets the proposal and iteration budgets of construction.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Returns the query workspace limit.
    pub const fn scattered(&self) -> ScatteredOptions {
        self.scattered
    }

    /// Returns the stopping tolerance of the gradient sweeps.
    pub const fn gradient_tolerance(&self) -> Tolerance<S> {
        self.gradient_tolerance
    }

    /// Returns the maximum number of sweeps per value component.
    pub const fn maximum_iterations(&self) -> usize {
        self.maximum_iterations
    }

    /// Returns the maximum memory for the vertex gradients and construction
    /// scratch.
    pub const fn maximum_preparation_bytes(&self) -> usize {
        self.maximum_preparation_bytes
    }

    /// Returns the proposal and iteration budgets of construction.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }
}

impl<S: RealScalar> Default for CloughTocherOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Tells why the vertex-gradient iteration of a [`CloughTocherInterpolator`] stopped.
///
/// ```
/// use scioxide_interp::CloughTocherTermination;
/// assert!(CloughTocherTermination::Converged.is_success());
/// assert!(!CloughTocherTermination::BudgetExhausted.is_success());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloughTocherTermination {
    /// Every value component met the gradient tolerance.
    Converged,
    /// The maximum number of sweeps was reached first.
    IterationLimit,
    /// The iteration budget was exhausted after at least one sweep.
    BudgetExhausted,
}

impl CloughTocherTermination {
    /// Returns `true` for [`Self::Converged`].
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Converged)
    }
}

/// Reports the termination status and diagnostics of the vertex-gradient iteration.
///
/// An unconverged estimate still gives a C¹ interpolant that matches the data
/// at the sites; only the vertex gradients are less accurate.
///
/// ```
/// use scioxide_interp::{
///     CloughTocherInterpolator, CloughTocherOptions, CloughTocherTermination, ScatteredValues,
/// };
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// let values = ScatteredValues::from_input_values(&mesh, vec![0.0, 1.0, 2.0, 0.5])?;
/// let model = CloughTocherInterpolator::with_options(
///     &mesh,
///     &values,
///     CloughTocherOptions::new().with_maximum_iterations(1),
/// )?;
/// let report = model.gradient_report();
/// assert_eq!(
///     report.termination(),
///     CloughTocherTermination::IterationLimit
/// );
/// assert_eq!(report.iterations(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CloughTocherReport<S: RealScalar> {
    termination: CloughTocherTermination,
    iterations: usize,
    final_change: S,
}

impl<S: RealScalar> CloughTocherReport<S> {
    /// Returns why the iteration stopped.
    pub const fn termination(&self) -> CloughTocherTermination {
        self.termination
    }

    /// Returns the number of completed sweeps, the largest over the value
    /// components.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the largest relative gradient change in the last completed sweep
    /// of any value component, as defined in
    /// [`CloughTocherOptions::with_gradient_tolerance`].
    pub const fn final_change(&self) -> S {
        self.final_change
    }
}

/// Interpolates scattered planar data with the C¹ piecewise-cubic Clough–Tocher element.
///
/// Each triangle of the Delaunay triangulation is split at its centroid into
/// three subtriangles, and the interpolant is a cubic Bézier patch on each.
/// The patches match the data values and the vertex gradients at the sites and
/// join with continuous first derivatives, so the surface is C¹ on the convex
/// hull. It reproduces affine functions exactly. Each real value component is
/// interpolated separately.
///
/// The vertex gradients are not given; construction estimates them by the
/// global method of Nielson. The gradients minimize an approximation of the
/// total curvature of the cubic curves along all triangle edges. Each site
/// gives a 2×2 linear system coupled to its neighbors, solved by Gauss–Seidel
/// sweeps until the change meets [`CloughTocherOptions::with_gradient_tolerance`].
/// The iteration starts from zero gradients. [`Self::gradient_report`]
/// returns the status.
///
/// Outside the convex hull, `Clamp` evaluates at the nearest hull point, and
/// its gradient is the patch gradient projected onto the nearest hull edge; it
/// is zero where the nearest hull point is a vertex. Where two projection
/// regions meet, their gradients must agree to rounding; otherwise the
/// gradient is `DomainError` (`Undefined`). `BoundedContinue` and
/// `MethodDefined` extend the cubic patch nearest to the query. See
/// [Extrapolation](crate#extrapolation).
///
/// A nearly flat triangle (a sliver) may have no usable barycentric frame.
/// Queries within `32 ε` (relative to the hull extent and the query magnitude)
/// of one of its edges or vertices are then evaluated on that edge, where the
/// element reduces to a cubic Hermite curve, or at that vertex. Other queries
/// in a sliver fail with `NumericalInstability`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_interp::{
///     CloughTocherInterpolator, CloughTocherOptions, Differentiable, ExtrapolationPolicy,
///     Interpolant, ScatteredValues,
/// };
/// use scioxide_spatial::{Delaunay, PointSet};
/// let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let points = PointSet::from_rows(&coordinates, 2)?;
/// let mesh = Delaunay::new(&points)?;
/// // f(x, y) = 1 + 3x + 2y is reproduced exactly.
/// let values = ScatteredValues::from_input_values(&mesh, vec![1.0, 4.0, 3.0, 6.0])?;
/// let options = CloughTocherOptions::new().with_gradient_tolerance(Tolerance::new(1e-12, 1e-12)?);
/// let model = CloughTocherInterpolator::with_options(&mesh, &values, options)?;
/// assert!(model.gradient_report().termination().is_success());
/// assert!((model.evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)? - 2.75).abs() < 1e-10);
/// let [dx, dy] = model.gradient([0.25, 0.5], ExtrapolationPolicy::Reject)?;
/// assert!((dx - 3.0).abs() < 1e-9 && (dy - 2.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. W. Clough and J. L. Tocher, "Finite element stiffness matrices for
///   analysis of plate bending", in *Proc. Conf. Matrix Methods in Structural
///   Mechanics*, Wright-Patterson Air Force Base, 1965, 515–545.
/// - G. M. Nielson, "A method for interpolating scattered data based upon a
///   minimum norm network", *Math. Comp.* 40(161), 1983, 253–271.
#[derive(Debug, Clone)]
#[must_use]
pub struct CloughTocherInterpolator<'a, S: RealScalar, V: AffineValue<Scale = S> = S> {
    base: Base<'a, S, V, 2>,
    /// `[site][component][axis]` vertex gradients.
    gradients: Vec<S>,
    report: CloughTocherReport<S>,
}

/// One triangle's value-independent edge and cross-boundary geometry.
struct TriangleGeometry<S> {
    vertices: [usize; 3],
    edges: [[S; 2]; 3],
    cross_derivatives: [S; 3],
}

impl<'a, S: RealScalar, V: AffineValue<Scale = S>> CloughTocherInterpolator<'a, S, V> {
    /// Creates the interpolator and estimates the vertex gradients on the calling thread.
    ///
    /// This is [`Self::with_options`] with [`CloughTocherOptions::default`]: a
    /// gradient tolerance of `√ε`, at most 400 Gauss–Seidel sweeps and no
    /// memory or budget limits. A gradient estimate that does not converge
    /// is an error here; [`Self::with_options`] and [`Self::build`] return the
    /// interpolator with its [`CloughTocherReport`] whatever the status.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::build`].
    /// - `ConvergenceFailure` (`IterationLimit`) if the sweeps stop before
    ///   meeting the gradient tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{CloughTocherInterpolator, ExtrapolationPolicy, Interpolant, ScatteredValues};
    /// use scioxide_spatial::{Delaunay, PointSet};
    /// let sites: [[f64; 2]; 5] = [[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.4, 0.6]];
    /// let mesh = Delaunay::new(&PointSet::from_points(&sites)?)?;
    /// // f(x, y) = 1 + 3x + 2y
    /// let heights = sites.iter().map(|[x, y]| 1.0 + 3.0 * x + 2.0 * y).collect();
    /// let values = ScatteredValues::from_input_values(&mesh, heights)?;
    /// let model = CloughTocherInterpolator::new(&mesh, &values)?;
    /// let value = model.evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)?;
    /// assert!((value - 2.75).abs() < 1e-6);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(tessellation, values, CloughTocherOptions::default())?
            .require_converged()
    }

    /// Creates the interpolator and estimates the vertex gradients on the calling
    /// thread, as configured by `options`.
    ///
    /// This is [`Self::build`] with [`ExecutionContext::sequential_native`]. If
    /// the sweep limit or the iteration budget ends the iteration, the
    /// interpolator is returned and [`Self::gradient_report`] gives the
    /// [`CloughTocherTermination`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn with_options(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
        options: CloughTocherOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            tessellation,
            values,
            options,
            ExecutionContext::sequential_native(),
        )
    }

    /// Creates the interpolator and estimates the vertex gradients within a
    /// solve scope.
    ///
    /// `scope` is an `ExecutionContext` for a top-level solve or a
    /// `&mut SolveContext` to run as a nested solve; see
    /// [solve accounting](scioxide_core#solve-accounting). Construction counts
    /// one proposal and then one iteration per sweep toward
    /// [`CloughTocherOptions::budgets`]. If the sweep limit or the iteration
    /// budget ends the iteration after at least one sweep, the interpolator is
    /// returned and [`Self::gradient_report`] gives the
    /// [`CloughTocherTermination`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `tessellation` is not two-dimensional, `values` was
    ///   built for a different triangulation, the sweep limit is zero, or a
    ///   triangle edge has zero length.
    /// - `ResourceLimit` if the query workspace or the construction memory
    ///   exceeds its limit in `options`.
    /// - `ConvergenceFailure` with
    ///   [`scioxide_core::ConvergenceReason::IterationLimit`] if the iteration
    ///   budget allows no sweep, or with
    ///   [`scioxide_core::ConvergenceReason::ResourceLimit`] if the proposal
    ///   budget is exhausted.
    /// - `Singular` if the 2×2 gradient system of a site is singular.
    /// - `NonFinite` if the gradient iteration overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - `SizeOverflow` or `AllocationFailure` if storage cannot be sized or
    ///   allocated.
    pub fn build<'scope, 'capability>(
        tessellation: &'a Delaunay<S>,
        values: &'a ScatteredValues<S, V>,
        options: CloughTocherOptions<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let base = Base::new(
            tessellation,
            values,
            options.scattered,
            Family::CloughTocher,
        )?;
        if options.maximum_iterations == 0 {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "Clough-Tocher requires a positive iteration limit",
            ));
        }
        let quantity = scioxide_core::quantity!("clough_tocher_preparation_bytes");
        let entries = checked_mul(
            checked_mul(values.sites, values.components, quantity)?,
            2,
            quantity,
        )?;
        let bytes = checked_add(
            checked_mul(entries, core::mem::size_of::<S>(), quantity)?,
            checked_add(
                checked_mul(values.sites, core::mem::size_of::<usize>(), quantity)?,
                checked_mul(
                    values.components,
                    core::mem::size_of::<bool>() + core::mem::size_of::<S>(),
                    quantity,
                )?,
                quantity,
            )?,
            quantity,
        )?;
        if bytes > options.maximum_preparation_bytes {
            return Err(SciOxideError::resource_limit(
                quantity,
                bytes,
                options.maximum_preparation_bytes,
            ));
        }
        let mut state = SolveState::new();
        scope.into().run(
            &mut state,
            identity(values, &options),
            options.budgets,
            |context| {
                require_proposal(context, "proposal budget denied Clough-Tocher preparation")?;
                let (gradients, report) = estimate_gradients(&base, &options, entries, context)?;
                context.execution().check()?;
                context.commit_accepted(PartialResultTrust::Admissible)?;
                Ok(Self {
                    base,
                    gradients,
                    report,
                })
            },
        )
    }

    /// Returns the termination status and diagnostics of the vertex-gradient
    /// iteration.
    pub const fn gradient_report(&self) -> CloughTocherReport<S> {
        self.report
    }

    /// Keeps a model whose gradient estimate converged; the check of [`Self::new`].
    fn require_converged(self) -> Result<Self, SciOxideError> {
        if self.report.termination().is_success() {
            Ok(self)
        } else {
            Err(unconverged(
                ConvergenceReason::IterationLimit,
                ALGORITHM,
                self.report.iterations(),
                "Clough-Tocher gradient estimation did not converge",
            ))
        }
    }

    /// Value-independent geometry of one classified macro-element.
    fn geometry(
        &self,
        simplex: usize,
        scratch: &mut Scratch<S>,
    ) -> Result<TriangleGeometry<S>, SciOxideError> {
        let tessellation = self.base.tessellation;
        let v = tessellation.simplex(simplex)?;
        let p = tessellation.points();
        let edge =
            |a: usize, b: usize| [p[v[b] * 2] - p[v[a] * 2], p[v[b] * 2 + 1] - p[v[a] * 2 + 1]];
        let edges = [edge(0, 1), edge(1, 2), edge(2, 0)];
        let three = S::try_from_usize_exact(3)?;
        let two = S::TWO;
        // g[k] parameterizes the cross-boundary derivative condition on the
        // edge opposite vertex k. It is derived from the neighbouring
        // triangle's centroid expressed in this triangle's barycentric
        // coordinates, so both sides produce the same edge-normal derivative
        // (C1). Hull edges have no neighbour and use the default -1/2, which
        // corresponds to a linearly varying normal derivative.
        let mut g = [S::ZERO; 3];
        for (local, value) in g.iter_mut().enumerate() {
            if let Some(neighbor) = tessellation.neighbors(simplex)?[local] {
                let vertices = tessellation.simplex(neighbor)?;
                let centroid = [
                    (p[vertices[0] * 2] + p[vertices[1] * 2] + p[vertices[2] * 2]) / three,
                    (p[vertices[0] * 2 + 1] + p[vertices[1] * 2 + 1] + p[vertices[2] * 2 + 1])
                        / three,
                ];
                tessellation.barycentric_into(
                    simplex,
                    &centroid,
                    &mut scratch.barycentric[..3],
                    &mut scratch.staging[..3],
                )?;
                let c = &scratch.barycentric;
                let (a, b) = match local {
                    0 => (c[2], c[1]),
                    1 => (c[0], c[2]),
                    _ => (c[1], c[0]),
                };
                *value = (two * a + b - S::ONE) / (two - three * a - three * b);
            } else {
                *value = -S::ONE / two;
            }
        }
        Ok(TriangleGeometry {
            vertices: [v[0], v[1], v[2]],
            edges,
            cross_derivatives: g,
        })
    }

    /// Clough–Tocher control points of one component on one triangle.
    fn coefficients(
        &self,
        geometry: &TriangleGeometry<S>,
        component: usize,
    ) -> Result<[S; 19], SciOxideError> {
        // Clough–Tocher macro-element:
        // the triangle is split at its centroid into three cubic Bézier
        // sub-triangles. Coefficients c_ijkl are indexed by barycentric
        // exponents of (vertex 1, vertex 2, vertex 3, centroid). Corner and
        // edge-adjacent control points come from vertex values and gradients
        // (C1 at vertices); centroid-adjacent points are averages of their
        // neighbours (C1 across the internal edges); and the mid-edge
        // points c0111, c1011, c1101 enforce C1 across each outer edge.
        let values = self.base.values;
        let v = geometry.vertices;
        let [e12, e23, e31] = geometry.edges;
        let gradient = |local: usize| {
            let offset = (v[local] * values.components + component) * 2;
            [self.gradients[offset], self.gradients[offset + 1]]
        };
        let d1 = gradient(0);
        let d2 = gradient(1);
        let d3 = gradient(2);
        let dot = |a: [S; 2], b: [S; 2]| a[0] * b[0] + a[1] * b[1];
        let f1 = values.component(v[0], component);
        let f2 = values.component(v[1], component);
        let f3 = values.component(v[2], component);
        let three = S::try_from_usize_exact(3)?;
        let two = S::TWO;
        let c3000 = f1;
        let c2100 = (dot(d1, e12) + three * f1) / three;
        let c2010 = (-dot(d1, e31) + three * f1) / three;
        let c0300 = f2;
        let c1200 = (-dot(d2, e12) + three * f2) / three;
        let c0210 = (dot(d2, e23) + three * f2) / three;
        let c0030 = f3;
        let c1020 = (dot(d3, e31) + three * f3) / three;
        let c0120 = (-dot(d3, e23) + three * f3) / three;
        let c2001 = (c2100 + c2010 + c3000) / three;
        let c0201 = (c1200 + c0300 + c0210) / three;
        let c0021 = (c1020 + c0120 + c0030) / three;
        let g = geometry.cross_derivatives;
        let c0111 = (g[0] * (-c0300 + three * c0210 - three * c0120 + c0030)
            + (-c0300 + two * c0210 - c0120 + c0021 + c0201))
            / two;
        let c1011 = (g[1] * (-c0030 + three * c1020 - three * c2010 + c3000)
            + (-c0030 + two * c1020 - c2010 + c2001 + c0021))
            / two;
        let c1101 = (g[2] * (-c3000 + three * c2100 - three * c1200 + c0300)
            + (-c3000 + two * c2100 - c1200 + c2001 + c0201))
            / two;
        let c1002 = (c1101 + c1011 + c2001) / three;
        let c0102 = (c1101 + c0111 + c0201) / three;
        let c0012 = (c1011 + c0111 + c0021) / three;
        let c0003 = (c1002 + c0102 + c0012) / three;
        Ok([
            c3000, c0300, c0030, c0003, c2100, c2010, c2001, c1200, c0210, c0201, c1020, c0120,
            c0021, c1101, c1011, c0111, c1002, c0102, c0012,
        ])
    }

    /// Value or gradient evaluation into `output`, `[axis][component]` for gradients.
    fn evaluate_order(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        output: &mut [S],
        scratch: &mut Scratch<S>,
        gradient: bool,
    ) -> Result<ExtrapolationReport<S, 2>, SciOxideError> {
        let resolution = self.base.classify(point, policy, scratch, false)?;
        let effective = resolution.report.effective();
        match self.evaluate_piece(resolution.simplex, effective, output, scratch, gradient) {
            // A sliver's value is evaluated on its closest edge or vertex.
            Err(error) if !gradient && super::degenerate(&error) => {
                self.base
                    .closest_simplex_face(resolution.simplex, effective, scratch)?;
                let count = scratch.face_count;
                let sites = [scratch.face_sites[0], scratch.face_sites[count.max(2) - 1]];
                self.edge_jet(sites, count, effective, false, output)?;
            }
            result => result?,
        }
        Ok(resolution.report)
    }

    /// Value or tangential gradient of the boundary restriction on a face:
    /// a vertex (`count == 1`) or the cubic Hermite edge `sites[0] -> sites[1]`
    /// whose Bézier points are `f_a`, `f_a + ∇f_a·e/3`, `f_b - ∇f_b·e/3`, `f_b`,
    /// exactly the macro-element's restriction to that edge.
    fn edge_jet(
        &self,
        sites: [usize; 2],
        count: usize,
        point: [S; 2],
        gradient: bool,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        let components = self.base.values.components;
        let [a, b] = sites;
        if count == 1 {
            for (component, slot) in output.iter_mut().enumerate() {
                *slot = if gradient {
                    S::ZERO
                } else {
                    self.base.values.component(a, component)
                };
            }
            return Ok(());
        }
        let p = self.base.tessellation.points();
        let x = |site: usize, axis: usize| p[site * 2 + axis].try_to_f64_exact();
        let scale = (0..2).try_fold(0_f64, |scale, axis| {
            Ok::<_, SciOxideError>(scale.max((x(b, axis)? - x(a, axis)?).abs()))
        })?;
        if !(scale.is_finite() && scale > 0.) {
            return Err(SciOxideError::singular(None));
        }
        let mut edge = [0_f64; 2];
        let mut offset = [0_f64; 2];
        for axis in 0..2 {
            edge[axis] = normalized_linear_difference(x(b, axis)?, x(a, axis)?, scale)?;
            offset[axis] =
                normalized_linear_difference(point[axis].try_to_f64_exact()?, x(a, axis)?, scale)?;
        }
        let length = edge[0].mul_add(edge[0], edge[1] * edge[1]);
        let t = (edge[0].mul_add(offset[0], edge[1] * offset[1]) / length).clamp(0., 1.);
        let s = 1. - t;
        for component in 0..components {
            let along = |site: usize| -> Result<f64, SciOxideError> {
                let offset = (site * components + component) * 2;
                let mut sum = 0_f64;
                for (&slope, &direction) in self.gradients[offset..offset + 2].iter().zip(&edge) {
                    sum = slope.try_to_f64_exact()?.mul_add(direction, sum);
                }
                Ok(sum * scale / 3.)
            };
            let c0 = self
                .base
                .values
                .component(a, component)
                .try_to_f64_exact()?;
            let c3 = self
                .base
                .values
                .component(b, component)
                .try_to_f64_exact()?;
            let c1 = c0 + along(a)?;
            let c2 = c3 - along(b)?;
            if gradient {
                // d/dt of the cubic, mapped to physical coordinates along e.
                let derivative =
                    3. * ((c1 - c0) * s * s + 2. * (c2 - c1) * s * t + (c3 - c2) * t * t);
                for axis in 0..2 {
                    output[axis * components + component] = intermediate(S::try_from_f64_lossy(
                        derivative * edge[axis] / (length * scale),
                    )?)?;
                }
            } else {
                let value =
                    s * s * s * c0 + 3. * s * s * t * c1 + 3. * s * t * t * c2 + t * t * t * c3;
                output[component] = intermediate(S::try_from_f64_lossy(value)?)?;
            }
        }
        Ok(())
    }

    fn evaluate_piece(
        &self,
        simplex: usize,
        query: [S; 2],
        output: &mut [S],
        scratch: &mut Scratch<S>,
        gradient: bool,
    ) -> Result<(), SciOxideError> {
        let components = self.base.values.components;
        self.base.tessellation.barycentric_into(
            simplex,
            &query,
            &mut scratch.barycentric[..3],
            &mut scratch.staging[..3],
        )?;
        let lambda = [
            scratch.barycentric[0],
            scratch.barycentric[1],
            scratch.barycentric[2],
        ];
        // Locate the sub-triangle: it is the one opposite the vertex with the
        // smallest barycentric coordinate. Coordinates relative to
        // (the two other vertices, centroid) are b_k = lambda_k - lambda_min for
        // the vertices and 3 lambda_min for the centroid; the zeroed entry marks
        // the omitted vertex, so terms using it vanish automatically and one
        // 19-term table serves all three sub-triangles.
        let minimum = if lambda[1] < lambda[0] {
            if lambda[2] < lambda[1] { 2 } else { 1 }
        } else if lambda[2] < lambda[0] {
            2
        } else {
            0
        };
        let b = [
            lambda[0] - lambda[minimum],
            lambda[1] - lambda[minimum],
            lambda[2] - lambda[minimum],
            S::try_from_usize_exact(3)? * lambda[minimum],
        ];
        let mut barycentric_gradients = [S::ZERO; 6];
        let mut barycentric_stage = [S::ZERO; 6];
        if gradient {
            self.base.tessellation.barycentric_gradients_into(
                simplex,
                &mut barycentric_gradients,
                &mut barycentric_stage,
            )?;
        }
        let mut coordinate_derivatives = [[S::ZERO; 4]; 2];
        if gradient {
            for (axis, row) in coordinate_derivatives.iter_mut().enumerate() {
                let dmin = barycentric_gradients[minimum * 2 + axis];
                *row = [
                    barycentric_gradients[axis] - dmin,
                    barycentric_gradients[2 + axis] - dmin,
                    barycentric_gradients[4 + axis] - dmin,
                    S::try_from_usize_exact(3)? * dmin,
                ];
            }
        }
        let geometry = self.geometry(simplex, scratch)?;
        let mut basis = [S::ZERO; 19];
        let mut derivatives = [[S::ZERO; 4]; 19];
        for term in 0..19 {
            let mut monomial = S::try_from_usize_exact(FACTORS[term])?;
            for variable in 0..4 {
                for _ in 0..EXPONENTS[term][variable] {
                    monomial = intermediate(monomial * b[variable])?;
                }
            }
            basis[term] = monomial;
            if gradient {
                for variable in 0..4 {
                    let exponent = EXPONENTS[term][variable];
                    if exponent == 0 {
                        continue;
                    }
                    let mut derivative = S::try_from_usize_exact(FACTORS[term] * exponent)?;
                    for other in 0..4 {
                        let power = EXPONENTS[term][other] - usize::from(other == variable);
                        for _ in 0..power {
                            derivative = intermediate(derivative * b[other])?;
                        }
                    }
                    derivatives[term][variable] = derivative;
                }
            }
        }
        for component in 0..components {
            let coefficients = self.coefficients(&geometry, component)?;
            let mut value = S::ZERO;
            let mut db = [S::ZERO; 4];
            for term in 0..19 {
                value = intermediate(coefficients[term].mul_add(basis[term], value))?;
                if gradient {
                    for variable in 0..4 {
                        // Zero exponents were skipped by the original recurrence.
                        if EXPONENTS[term][variable] != 0 {
                            db[variable] = intermediate(
                                coefficients[term]
                                    .mul_add(derivatives[term][variable], db[variable]),
                            )?;
                        }
                    }
                }
            }
            // Chain rule: d/dx = sum_v (dP/db_v)(db_v/dx), where db_v/dx follows
            // from the affine map b = (lambda_k - lambda_min, 3 lambda_min) and
            // the constant barycentric gradients of the simplex.
            if gradient {
                for axis in 0..2 {
                    output[axis * components + component] =
                        (0..4).try_fold(S::ZERO, |sum, index| {
                            intermediate(
                                db[index].mul_add(coordinate_derivatives[axis][index], sum),
                            )
                        })?;
                }
            } else {
                output[component] = value;
            }
        }
        Ok(())
    }
}

/// Private identity of one gradient-estimation epoch.
fn identity<S: RealScalar, V: AffineValue<Scale = S>>(
    values: &ScatteredValues<S, V>,
    options: &CloughTocherOptions<S>,
) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(0),
        ALGORITHM,
        ((values.sites as u128) << 64) | values.components as u128,
        options.maximum_iterations as u128,
    )
}

/// Barycentric exponents of the 19 cubic Bézier control points.
const EXPONENTS: [[usize; 4]; 19] = [
    [3, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 3, 0],
    [0, 0, 0, 3],
    [2, 1, 0, 0],
    [2, 0, 1, 0],
    [2, 0, 0, 1],
    [1, 2, 0, 0],
    [0, 2, 1, 0],
    [0, 2, 0, 1],
    [1, 0, 2, 0],
    [0, 1, 2, 0],
    [0, 0, 2, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 2],
    [0, 1, 0, 2],
    [0, 0, 1, 2],
];

/// Multinomial coefficients 3!/(i! j! k! l!) of the cubic Bernstein basis.
const FACTORS: [usize; 19] = [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 3, 3, 3];

/// Global vertex-gradient estimation by interleaved Gauss–Seidel sweeps.
fn estimate_gradients<S: RealScalar, V: AffineValue<Scale = S>>(
    base: &Base<'_, S, V, 2>,
    options: &CloughTocherOptions<S>,
    entries: usize,
    context: &mut SolveContext<'_, '_>,
) -> Result<(Vec<S>, CloughTocherReport<S>), SciOxideError> {
    // Global vertex-gradient estimation (Nielson 1980): choose gradients minimizing an
    // approximate total curvature of the cubic edge interpolants. Each site's
    // 2x2 normal equations couple it to its edge neighbours and are solved by
    // Gauss–Seidel sweeps. Components are independent, so one sweep visits
    // every unconverged component and each component stops at its own
    // convergence, exactly as if it were solved alone. Gradients are stored
    // `[site][component][axis]` and start at zero.
    let values = base.values;
    let mut gradients = try_filled(
        entries,
        S::ZERO,
        scioxide_core::quantity!("clough_tocher_gradients"),
    )?;
    let mut seen = try_filled(
        values.sites,
        usize::MAX,
        scioxide_core::quantity!("clough_tocher_adjacency"),
    )?;
    let mut converged = try_filled(
        values.components,
        false,
        scioxide_core::quantity!("clough_tocher_convergence"),
    )?;
    let mut last_change = try_filled(
        values.components,
        S::ZERO,
        scioxide_core::quantity!("clough_tocher_convergence"),
    )?;
    let mut iterations = 0;
    let mut termination = CloughTocherTermination::IterationLimit;
    while iterations < options.maximum_iterations {
        if converged.iter().all(|&done| done) {
            break;
        }
        if !admit_iteration(context)? {
            if iterations == 0 {
                return Err(iteration_denied(
                    context,
                    "iteration budget denied the first Clough-Tocher sweep",
                ));
            }
            termination = CloughTocherTermination::BudgetExhausted;
            break;
        }
        for component in 0..values.components {
            if converged[component] {
                continue;
            }
            let change = sweep(base, component, &mut gradients, &mut seen)?;
            last_change[component] = change;
            converged[component] = options.gradient_tolerance.is_residual_ok(change, S::ONE)?;
        }
        iterations += 1;
    }
    if converged.iter().all(|&done| done) {
        termination = CloughTocherTermination::Converged;
    }
    let final_change = last_change
        .iter()
        .fold(S::ZERO, |max, &change| max.max(change));
    Ok((
        gradients,
        CloughTocherReport {
            termination,
            iterations,
            final_change,
        },
    ))
}

/// One Gauss–Seidel sweep of one component; returns its largest normalized change.
fn sweep<S: RealScalar, V: AffineValue<Scale = S>>(
    base: &Base<'_, S, V, 2>,
    component: usize,
    gradients: &mut [S],
    seen: &mut [usize],
) -> Result<S, SciOxideError> {
    let tessellation = base.tessellation;
    let values = base.values;
    let points = tessellation.points();
    let coordinate_scale = intermediate(base.domains.iter().fold(S::ZERO, |scale, domain| {
        scale.max(domain.upper() - domain.lower())
    }))?;
    let mut sweep_change = S::ZERO;
    // One reset per sweep; each site's index is its deduplication generation.
    // Stable ascending incidence preserves the exhaustive scan's edge order.
    seen.fill(usize::MAX);
    for site in 0..values.sites {
        seen[site] = site;
        let mut q00 = S::ZERO;
        let mut q01 = S::ZERO;
        let mut q11 = S::ZERO;
        let mut s0 = S::ZERO;
        let mut s1 = S::ZERO;
        for simplex in tessellation.incident_simplices(site)? {
            let vertices = tessellation.simplex(simplex)?;
            for &other in vertices {
                if seen[other] == site {
                    continue;
                }
                seen[other] = site;
                let ex = intermediate((points[other * 2] - points[site * 2]) / coordinate_scale)?;
                let ey = intermediate(
                    (points[other * 2 + 1] - points[site * 2 + 1]) / coordinate_scale,
                )?;
                let length = norm([Ok(ex), Ok(ey)])?;
                if length == S::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("tessellation"),
                        "Clough-Tocher edge has zero length",
                    ));
                }
                let length3 = intermediate(length * length * length)?;
                let other_offset = (other * values.components + component) * 2;
                let df2 = intermediate(
                    -(ex * (gradients[other_offset] * coordinate_scale)
                        + ey * (gradients[other_offset + 1] * coordinate_scale)),
                )?;
                let forcing = intermediate(
                    S::try_from_usize_exact(6)?
                        * (values.component(site, component) - values.component(other, component))
                        - S::TWO * df2,
                )?;
                let four = S::try_from_usize_exact(4)?;
                q00 = intermediate(q00 + four * ex * ex / length3)?;
                q01 = intermediate(q01 + four * ex * ey / length3)?;
                q11 = intermediate(q11 + four * ey * ey / length3)?;
                s0 = intermediate(s0 + forcing * ex / length3)?;
                s1 = intermediate(s1 + forcing * ey / length3)?;
            }
        }
        // Solve the symmetric 2x2 system by Cramer's rule; the solution r is
        // the negated gradient in this sign convention. The change is measured
        // relative to max(1, |r|) in normalized coordinates, so changing
        // physical coordinate units cannot cause premature convergence.
        let determinant = intermediate(q00 * q11 - q01 * q01)?;
        if determinant == S::ZERO {
            return Err(SciOxideError::singular(None));
        }
        let r0 = intermediate((q11 * s0 - q01 * s1) / determinant)?;
        let r1 = intermediate((-q01 * s0 + q00 * s1) / determinant)?;
        let offset = (site * values.components + component) * 2;
        let change = (gradients[offset] * coordinate_scale + r0)
            .abs()
            .max((gradients[offset + 1] * coordinate_scale + r1).abs())
            / S::ONE.max(r0.abs().max(r1.abs()));
        sweep_change = sweep_change.max(change);
        gradients[offset] = intermediate(-r0 / coordinate_scale)?;
        gradients[offset + 1] = intermediate(-r1 / coordinate_scale)?;
    }
    Ok(sweep_change)
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 2>
    for CloughTocherInterpolator<'_, S, V>
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
        let staged = &mut workspace.output[..components];
        let report = self.evaluate_order(point, policy, staged, &mut workspace.scratch, false)?;
        scioxide_core::write_value(staged, out)?;
        Ok(report)
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
            .classify(point, policy, &mut workspace.scratch, false)?
            .report)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Differentiable<S, 2>
    for CloughTocherInterpolator<'_, S, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 2],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut ScatteredWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let width = checked_mul(
            2,
            self.base.values.components,
            scioxide_core::quantity!("gradient_entries"),
        )?;
        scioxide_core::check_value_shape(out, width)?;
        self.base.check(workspace)?;
        let staged = &mut workspace.output[..width];
        let scratch = &mut workspace.scratch;
        let resolution = self.base.classify(point, policy, scratch, false)?;
        if policy == ExtrapolationPolicy::Clamp && resolution.hull_face.is_some() {
            self.base.clamp_gradient(
                &resolution,
                staged,
                &mut workspace.comparison[..width],
                scratch,
                |simplex, point, count, out, scratch| {
                    match self.evaluate_piece(simplex, point, out, scratch, true) {
                        // A sliver's tangential gradient is its hull edge cubic's.
                        Err(error) if super::degenerate(&error) => {
                            let sites = [scratch.subset[0], scratch.subset[count.max(2) - 1]];
                            self.edge_jet(sites, count, point, true, out)
                        }
                        result => result,
                    }
                },
            )?;
        } else {
            self.evaluate_piece(
                resolution.simplex,
                resolution.report.effective(),
                staged,
                scratch,
                true,
            )?;
        }
        scioxide_core::write_value(staged, out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::tests::{options, triangulation};
    use super::*;
    use scioxide_core::{Budget, SciOxideErrorView};
    use scioxide_core::{BudgetKind, ConvergenceReason};

    fn tight() -> CloughTocherOptions<f64> {
        CloughTocherOptions::new()
            .with_scattered(options())
            .with_gradient_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_maximum_iterations(128)
            .with_maximum_preparation_bytes(1024 * 1024)
    }

    #[test]
    fn preparation_limit_covers_gradients_markers_and_component_state() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![[0., 1.]; 3]).unwrap();
        let bytes = 3 * 2 * 2 * core::mem::size_of::<f64>()
            + 3 * core::mem::size_of::<usize>()
            + 2 * (core::mem::size_of::<f64>() + core::mem::size_of::<bool>());
        assert!(
            CloughTocherInterpolator::with_options(
                &mesh,
                &values,
                tight().with_maximum_preparation_bytes(bytes)
            )
            .is_ok()
        );
        let error = CloughTocherInterpolator::with_options(
            &mesh,
            &values,
            tight().with_maximum_preparation_bytes(bytes - 1),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
    }

    #[test]
    fn new_requires_a_converged_gradient_estimate() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1., 0.4, 0.6]);
        let values = ScatteredValues::from_input_values(&mesh, vec![0., 1., 2., 0.5, -1.]).unwrap();
        // One sweep stops before the default tolerance; `with_options` keeps the model.
        let one = CloughTocherOptions::new().with_maximum_iterations(1);
        let limited = CloughTocherInterpolator::with_options(&mesh, &values, one).unwrap();
        let report = limited.gradient_report();
        assert_eq!(
            report.termination(),
            CloughTocherTermination::IterationLimit
        );
        assert_eq!(report.iterations(), 1);
        let error = limited.require_converged().unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                diagnostics,
                ..
            } if diagnostics.algorithm() == Some(ALGORITHM)
                && diagnostics.counts().own().iterations() == 1
        ));
        // A converged estimate is the model of the configured form, bit for bit.
        let model = CloughTocherInterpolator::new(&mesh, &values).unwrap();
        let configured =
            CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::default())
                .unwrap();
        assert!(model.gradient_report().termination().is_success());
        assert_eq!(model.gradient_report(), configured.gradient_report());
        let bits = |gradients: &[f64]| gradients.iter().map(|g| g.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&model.gradients), bits(&configured.gradients));
    }

    #[test]
    fn reproduces_affine_values_and_gradients() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1.]);
        let values = ScatteredValues::from_input_values(&mesh, vec![1., 4., 3., 6.]).unwrap();
        let model = CloughTocherInterpolator::with_options(&mesh, &values, tight()).unwrap();
        let report = model.gradient_report();
        assert_eq!(report.termination(), CloughTocherTermination::Converged);
        assert!(report.iterations() > 0);
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
        assert!((output[0] - 2.75).abs() < 1e-10);
        let mut gradient = [0.; 2];
        model
            .gradient_into(
                [0.25, 0.5],
                ExtrapolationPolicy::Reject,
                &mut gradient,
                &mut workspace,
            )
            .unwrap();
        assert!((gradient[0] - 3.).abs() < 1e-9);
        assert!((gradient[1] - 2.).abs() < 1e-9);
    }

    #[test]
    fn interleaved_components_match_independent_estimates() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1., 0.4, 0.6]);
        let first = vec![0., 1., 2., 0.5, 3.];
        let second = vec![2., -1., 0.25, 4., 1.];
        let paired: Vec<[f64; 2]> = first.iter().zip(&second).map(|(&a, &b)| [a, b]).collect();
        let options = CloughTocherOptions::new().with_maximum_iterations(400);
        let values = ScatteredValues::from_site_values(&mesh, paired).unwrap();
        let model = CloughTocherInterpolator::with_options(&mesh, &values, options).unwrap();
        let mut iterations = 0;
        for (component, column) in [first, second].into_iter().enumerate() {
            let scalar = ScatteredValues::from_site_values(&mesh, column).unwrap();
            let alone = CloughTocherInterpolator::with_options(&mesh, &scalar, options).unwrap();
            iterations = iterations.max(alone.gradient_report().iterations());
            for query in [[0.2, 0.3], [0.7, 0.9], [0.5, 0.1]] {
                let value = model.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
                let expected = alone.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
                assert_eq!(value[component], expected);
            }
        }
        assert_eq!(model.gradient_report().iterations(), iterations);
    }

    #[test]
    fn limits_and_budgets_report_one_status() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1., 0.4, 0.6]);
        let values = ScatteredValues::from_site_values(&mesh, vec![0., 1., 2., 0.5, 3.]).unwrap();
        let limited = CloughTocherOptions::new().with_maximum_iterations(1);
        let model = CloughTocherInterpolator::with_options(&mesh, &values, limited).unwrap();
        assert_eq!(
            model.gradient_report().termination(),
            CloughTocherTermination::IterationLimit
        );
        assert_eq!(model.gradient_report().iterations(), 1);
        assert!(model.gradient_report().final_change() > 0.);

        let budgeted = CloughTocherOptions::new().with_budgets(
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, Budget::try_from(2).unwrap()),
        );
        let model = CloughTocherInterpolator::with_options(&mesh, &values, budgeted).unwrap();
        assert_eq!(
            model.gradient_report().termination(),
            CloughTocherTermination::BudgetExhausted
        );
        assert_eq!(model.gradient_report().iterations(), 2);
        // The stopped estimate equals the iteration-limited one at the same sweep.
        let same = CloughTocherInterpolator::with_options(
            &mesh,
            &values,
            CloughTocherOptions::new().with_maximum_iterations(2),
        )
        .unwrap();
        assert_eq!(model.gradients, same.gradients);

        assert!(
            matches!(CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new().with_maximum_iterations(0)), Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { .. }))
        );
        assert!(
            matches!(CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new().with_maximum_preparation_bytes(8)), Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
        );
    }

    #[test]
    fn denied_first_sweep_is_an_iteration_limit_failure() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![0., 1., 2., 0.5]).unwrap();
        let options = CloughTocherOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
        let base = Base::new(&mesh, &values, options.scattered(), Family::CloughTocher).unwrap();
        // The budget is spent inside the private preparation epoch before the
        // first sweep, which no public input can arrange.
        let result = SolveScope::from(ExecutionContext::sequential_native()).run(
            &mut SolveState::new(),
            identity(&values, &options),
            options.budgets(),
            |context| {
                assert!(context.record_iteration()?.is_granted());
                estimate_gradients(&base, &options, 8, context)
            },
        );
        let Err(error) = result else {
            unreachable!("a denied first sweep publishes no estimate")
        };
        let SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } = error.view()
        else {
            unreachable!("budget denial is a convergence failure")
        };
        assert_eq!(reason, ConvergenceReason::IterationLimit);
        assert_eq!(
            diagnostics.budget().map(|report| report.kind()),
            Some(BudgetKind::Iteration)
        );
    }

    #[test]
    fn cancellation_publishes_nothing() {
        let mesh = triangulation(&[0., 0., 1., 0., 0., 1., 1., 1.]);
        let values = ScatteredValues::from_site_values(&mesh, vec![0., 1., 2., 0.5]).unwrap();
        let token = scioxide_core::CancellationToken::new();
        token.cancel();
        let result = CloughTocherInterpolator::build(
            &mesh,
            &values,
            CloughTocherOptions::new(),
            ExecutionContext::sequential().with_cancellation(&token),
        );
        assert!(
            matches!(result, Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::ExecutionStopped)
        );
        let owned =
            CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new())
                .unwrap();
        let scoped = CloughTocherInterpolator::build(
            &mesh,
            &values,
            CloughTocherOptions::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(owned.gradients, scoped.gradients);
        assert_eq!(owned.gradient_report(), scoped.gradient_report());
    }
}
