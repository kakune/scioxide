//! Local fits around a center and Gaussian-process prediction.
//!
//! [`LocalModelPlan`] builds a k-d tree over the samples once. Local RBF and
//! polynomial models are fitted from it to the nearest samples of a center,
//! and [`GaussianProcessModel`] fits all samples. The models are evaluated
//! through [`Interpolant`](crate::Interpolant) with a
//! [`LocalPredictionWorkspace`].

mod gaussian;
mod polynomial;
mod rbf;

pub use gaussian::{
    Covariance, CovarianceKernel, GaussianPrediction, GaussianProcessModel, KrigingMean,
};
pub use polynomial::{LocalPolynomialModel, LocalPolynomialOptions};
pub use rbf::{LocalRbfModel, LocalRbfOptions};

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, Argument, Capacity, ExecutionContext, Interval, NdView, NumericScalar,
    RankThreshold, RealScalar, SciOxideError, SolveBudgets, StridedLayout,
};
use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace, Matrix, Vector};
use scioxide_spatial::{KdTree, Neighbor, PointSet, SearchOptions, SearchWorkspaceRequest};

use crate::factor::{Factor, SystemKind, factor, solve_requirements};
use crate::{FitConditionNorm, FitFactorization};

type Result<T> = core::result::Result<T, SciOxideError>;

use crate::checked::intermediate;
use crate::error::invalid;
use crate::geometry::{bounds, distance};
use crate::validate::{validate_count, validate_finite, validate_length};

/// Error detail when the proposal limit stops a local fit.
const PROPOSAL_DENIED: &str = "local preparation proposal budget exhausted";

fn point<S: RealScalar, const D: usize>(points: &PointSet<'_, S>, i: usize) -> Result<[S; D]> {
    let mut p = [S::ZERO; D];
    for (axis, x) in p.iter_mut().enumerate() {
        *x = points.coordinate(i, axis)?;
    }
    Ok(p)
}

/// Resolves a query through the extrapolation policy against the fit's
/// bounding box.
fn resolve<S: RealScalar, const D: usize>(
    domains: [Interval<S>; D],
    point: [S; D],
    policy: crate::ExtrapolationPolicy<S>,
) -> Result<[S; D]> {
    Ok(policy.resolve(domains, point)?.effective())
}

/// Stores sample points and values in the dense layout that
/// [`LocalModelPlan`] and [`GaussianProcessModel`] read.
///
/// Points form an `n × D` array and values an `n × m` array, both row-major,
/// where `m` is the number of real components of one value. A complex value
/// takes two columns, its real and imaginary parts.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::LocalSamples;
/// let samples = LocalSamples::new(
///     &[[0.0_f64], [1.0]],
///     &[Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
/// )?;
/// assert_eq!(samples.points()?.len(), 2);
/// assert_eq!(samples.values()?.layout().shape().extents(), &[2, 2]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LocalSamples<S: RealScalar, const D: usize> {
    coordinates: Vec<S>,
    values: Vec<S>,
    value_shape: [usize; 2],
    value_strides: [usize; 2],
}

impl<S: RealScalar, const D: usize> LocalSamples<S, D> {
    /// Copies the points and the real components of the values.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is zero or `points` is empty.
    /// - `DimensionMismatch` if `points` and `values` differ in length or the
    ///   values differ in shape.
    /// - `NonFinite` if a coordinate or value is NaN or infinite.
    pub fn new<V: AffineValue<Scale = S>>(points: &[[S; D]], values: &[V]) -> Result<Self> {
        if D == 0 {
            return Err(invalid(
                scioxide_core::argument!("points"),
                "coordinate dimension must be positive",
            ));
        }
        validate_count(points.len(), 1, scioxide_core::argument!("points"))?;
        validate_length(points.len(), values.len())?;
        validate_finite(points.as_flattened())?;
        let c = values[0].dimension()?;
        let mut coordinates = try_with_capacity(
            checked_mul(points.len(), D, scioxide_core::quantity!("coordinates"))?,
            scioxide_core::quantity!("coordinates"),
        )?;
        let mut components = try_with_capacity(
            checked_mul(points.len(), c, scioxide_core::quantity!("values"))?,
            scioxide_core::quantity!("values"),
        )?;
        for (p, v) in points.iter().zip(values) {
            scioxide_core::check_value_shape(v, c)?;
            scioxide_core::validate_value(v)?;
            coordinates.extend_from_slice(p);
            for i in 0..c {
                components.push(v.component(i)?);
            }
        }
        Ok(Self {
            coordinates,
            values: components,
            value_shape: [points.len(), c],
            value_strides: [c, 1],
        })
    }

    /// Returns the points as an `n × D` point set.
    ///
    /// # Errors
    ///
    /// Returns the error of the point set if the stored coordinates are rejected,
    /// which does not happen for samples built by [`Self::new`].
    pub fn points(&self) -> Result<PointSet<'_, S>> {
        PointSet::from_rows(&self.coordinates, D)
    }

    /// Returns the values as an `n × m` array indexed `[sample, component]`.
    ///
    /// # Errors
    ///
    /// Returns the error of the array view if the stored layout is rejected,
    /// which does not happen for samples built by [`Self::new`].
    pub fn values(&self) -> Result<NdView<'_, '_, S>> {
        NdView::new(
            &self.values,
            StridedLayout::new(&self.value_shape, &self.value_strides)?,
        )
    }
}

/// Holds the rank, conditioning, size and work limits of local fits and
/// Gaussian-process models.
///
/// The defaults are [`RankThreshold::default`], a smallest reciprocal condition
/// number of `√ε`, dense systems of order at most 1024, no byte limit, no
/// iteration limit for the SVD, and unbounded budgets. Each fit runs as one
/// solve with these budgets and counts one proposal for each system it
/// factors (see [solve accounting](scioxide_core#solve-accounting)).
///
/// ```
/// use scioxide_core::{Budget, BudgetKind, SolveBudgets};
/// use scioxide_interp::LocalModelOptions;
/// let options = LocalModelOptions::<f64>::new()
///     .with_max_system_order(64)
///     .with_maximum_bytes(1 << 20)
///     .with_budgets(
///         SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, Budget::try_from(4)?),
///     );
/// assert_eq!(options.max_system_order(), 64);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LocalModelOptions<S: RealScalar> {
    rank_threshold: RankThreshold<S>,
    minimum_rcond: S,
    max_system_order: usize,
    maximum_bytes: usize,
    factor_iterations: Option<scioxide_core::Budget>,
    budgets: SolveBudgets,
}

impl<S: RealScalar> LocalModelOptions<S> {
    /// Creates the default options.
    pub fn new() -> Self {
        Self {
            rank_threshold: RankThreshold::default(),
            minimum_rcond: S::SQRT_EPSILON,
            max_system_order: 1024,
            maximum_bytes: usize::MAX,
            factor_iterations: None,
            budgets: SolveBudgets::unbounded(),
        }
    }

    /// Sets the tolerance that decides the numerical rank of a system.
    pub const fn with_rank_threshold(mut self, threshold: RankThreshold<S>) -> Self {
        self.rank_threshold = threshold;
        self
    }

    /// Sets the smallest accepted reciprocal condition number, in `[0, 1]`.
    ///
    /// The value is checked when a fit starts.
    pub const fn with_minimum_rcond(mut self, rcond: S) -> Self {
        self.minimum_rcond = rcond;
        self
    }

    /// Sets the largest order of a dense system, which also bounds the number of
    /// polynomial terms.
    pub const fn with_max_system_order(mut self, order: usize) -> Self {
        self.max_system_order = order;
        self
    }

    /// Limits the memory estimate of a fit to `bytes`; see
    /// [`LocalModelPlan::preparation_bytes`].
    pub const fn with_maximum_bytes(mut self, bytes: usize) -> Self {
        self.maximum_bytes = bytes;
        self
    }

    /// Limits the iterations of an SVD that replaces a failed direct
    /// factorization. Cholesky, LU and QR factorizations ignore it.
    pub const fn with_factor_iterations(mut self, budget: scioxide_core::Budget) -> Self {
        self.factor_iterations = Some(budget);
        self
    }

    /// Sets the budgets of each fit; the proposal limit bounds the number of
    /// systems a fit may factor.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Returns the rank tolerance.
    pub const fn rank_threshold(&self) -> RankThreshold<S> {
        self.rank_threshold
    }

    /// Returns the smallest accepted reciprocal condition number.
    pub const fn minimum_rcond(&self) -> S {
        self.minimum_rcond
    }

    /// Returns the largest order of a dense system.
    pub const fn max_system_order(&self) -> usize {
        self.max_system_order
    }

    /// Returns the byte limit of a fit.
    pub const fn maximum_bytes(&self) -> usize {
        self.maximum_bytes
    }

    /// Returns the iteration limit of the SVD, if set.
    pub const fn factor_iterations(&self) -> Option<scioxide_core::Budget> {
        self.factor_iterations
    }

    /// Returns the budgets of each fit.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }

    /// Upper bound on the memory of a fit of a `rows × columns` system with
    /// `components` value columns in `D` dimensions, checked against `options`.
    fn required_bytes<const D: usize>(
        &self,
        rows: usize,
        columns: usize,
        components: usize,
        options: Argument,
    ) -> Result<usize> {
        self.check(rows, columns, options)?;
        if D == 0 || D > self.max_system_order {
            return Err(invalid(
                options,
                "coordinate dimension exceeds system budget",
            ));
        }
        // Count scalar-sized entries for a square dense factor of the larger
        // dimension, copied samples, polynomial exponents and small vectors,
        // then multiply by a generous factor of 128 and by the larger of the
        // scalar/index sizes. This is a deliberately loose upper bound so the
        // budget check can run before any allocation without having to model
        // the shared SVD's internal workspace exactly.
        let n = rows.max(columns);
        let dense = checked_mul(n, n, scioxide_core::quantity!("dense_entries"))?;
        let samples = checked_mul(
            n,
            checked_add(D, components, scioxide_core::quantity!("sample_width"))?,
            scioxide_core::quantity!("sample_entries"),
        )?;
        let terms = checked_mul(columns, D, scioxide_core::quantity!("exponent_entries"))?;
        let entries = checked_add(
            checked_add(
                dense,
                samples,
                scioxide_core::quantity!("preparation_entries"),
            )?,
            checked_add(
                terms,
                checked_add(n, 1, scioxide_core::quantity!("vector_entries"))?,
                scioxide_core::quantity!("preparation_entries"),
            )?,
            scioxide_core::quantity!("preparation_entries"),
        )?;
        checked_mul(
            checked_mul(entries, 128, scioxide_core::quantity!("preparation_bytes"))?,
            checked_add(
                core::mem::size_of::<S>(),
                core::mem::size_of::<usize>(),
                scioxide_core::quantity!("entry_bytes"),
            )?,
            scioxide_core::quantity!("preparation_bytes"),
        )
    }

    /// Fails with `ResourceLimit` when the memory bound exceeds the byte limit.
    fn preflight<const D: usize>(
        &self,
        rows: usize,
        columns: usize,
        components: usize,
        options: Argument,
    ) -> Result<()> {
        let required = self.required_bytes::<D>(rows, columns, components, options)?;
        if required > self.maximum_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("local_preparation_bytes"),
                required,
                self.maximum_bytes,
            ));
        }
        Ok(())
    }

    /// Validates the condition and order limits, naming `options` in errors.
    fn check(&self, rows: usize, cols: usize, options: Argument) -> Result<()> {
        if !self.minimum_rcond.is_finite()
            || self.minimum_rcond < S::ZERO
            || self.minimum_rcond > S::ONE
            || rows.max(cols) > self.max_system_order
            || rows == 0
            || cols == 0
        {
            return Err(invalid(
                options,
                "invalid conditioning policy or dense system order budget exceeded",
            ));
        }
        Ok(())
    }
}

impl<S: RealScalar> Default for LocalModelOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reports the neighborhood, rank and conditioning of a local fit or
/// Gaussian-process model.
///
/// After a Cholesky, LU or QR factorization the reciprocal condition number is
/// an estimate in the 1-norm. After an SVD it is `σ_min/σ_max` in the 2-norm.
/// [`LocalModelOptions::with_minimum_rcond`] applies to the reported value. The
/// numbers describe the system that was solved; they are not error bounds for
/// the predictions.
///
/// ```
/// use scioxide_interp::{
///     LocalModelPlan, LocalPolynomialModel, LocalPolynomialOptions, LocalSamples,
/// };
/// let samples = LocalSamples::new(&[[-1.0_f64], [0.0], [1.0]], &[1.0, 2.0, 3.0])?;
/// let points = samples.points()?;
/// let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values()?)?;
/// let model = LocalPolynomialModel::new(&plan, [0.0], LocalPolynomialOptions::new(1, 3, 3))?;
/// let report = model.fit_report();
/// assert_eq!(
///     (report.neighbors(), report.degree(), report.attempts()),
///     (3, Some(1), 1)
/// );
/// assert!(report.residual() < 1e-12);
/// assert_eq!(
///     report.factorization(),
///     scioxide_interp::FitFactorization::PivotedQr
/// );
/// assert!(!report.used_fallback());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LocalFitReport<S: RealScalar> {
    neighbors: usize,
    degree: Option<usize>,
    rank: usize,
    reciprocal_condition: S,
    residual: S,
    attempts: usize,
    factorization: FitFactorization,
    condition_norm: FitConditionNorm,
}

impl<S: RealScalar> LocalFitReport<S> {
    /// Returns the factorization of the solved system.
    pub const fn factorization(&self) -> FitFactorization {
        self.factorization
    }

    /// Returns whether an SVD replaced the Cholesky, LU or QR factorization.
    pub fn used_fallback(&self) -> bool {
        self.factorization == FitFactorization::Svd
    }

    /// Returns the norm of [`Self::reciprocal_condition`].
    pub const fn condition_norm(&self) -> FitConditionNorm {
        self.condition_norm
    }

    /// Returns the number of samples in the fit.
    pub const fn neighbors(&self) -> usize {
        self.neighbors
    }

    /// Returns the total degree of the polynomial part, or `None` without one.
    pub const fn degree(&self) -> Option<usize> {
        self.degree
    }

    /// Returns the numerical rank, equal to the number of columns of the system.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the reciprocal condition number in [`Self::condition_norm`].
    pub const fn reciprocal_condition(&self) -> S {
        self.reciprocal_condition
    }

    /// Returns the largest absolute residual of the solved system over all real
    /// components.
    ///
    /// For a local polynomial this is the least-squares residual at the samples.
    /// For RBF and Gaussian-process models the system includes the
    /// regularization or noise on its diagonal.
    pub const fn residual(&self) -> S {
        self.residual
    }

    /// Returns the number of neighborhood and degree combinations tried.
    pub const fn attempts(&self) -> usize {
        self.attempts
    }
}

/// Indexes samples in a k-d tree for local fits around chosen centers.
///
/// The plan borrows an `n × D` point set and an `n × m` value array, for example
/// from [`LocalSamples`]. [`LocalRbfModel`] and [`LocalPolynomialModel`] select
/// the nearest samples of a center from it by Euclidean distance, and ties go
/// to the lower sample index. Building the tree costs O(D · n log n). Clones
/// share the borrowed samples.
///
/// ```
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, LocalModelPlan, LocalPolynomialModel, LocalPolynomialOptions,
///     LocalSamples,
/// };
/// let samples = LocalSamples::new(&[[-1.0_f64], [0.0], [1.0], [2.0]], &[1.0, 0.0, 1.0, 4.0])?;
/// let points = samples.points()?;
/// let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values()?)?;
/// let options = LocalPolynomialOptions::new(2, 3, 4);
/// assert!(plan.preparation_bytes(4, 3, &options.numerical())? > 0);
/// let model = LocalPolynomialModel::new(&plan.clone(), [0.5], options)?;
/// assert!((model.evaluate([0.5], ExtrapolationPolicy::Reject)?[0] - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LocalModelPlan<'data, 'layout, S: RealScalar, const D: usize> {
    tree: KdTree<'data, S>,
    values: &'data [S],
    value_layout: StridedLayout<'layout>,
    components: usize,
}

impl<'data, 'layout, S: RealScalar, const D: usize> LocalModelPlan<'data, 'layout, S, D> {
    /// Builds the k-d tree over `points`, with `values` indexed
    /// `[sample, component]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is zero, `points` does not have `D` coordinates,
    ///   `values` is not two-dimensional, or there are no samples.
    /// - `DimensionMismatch` if `points` and `values` have different numbers of
    ///   samples.
    /// - `NonFinite` if a value is NaN or infinite.
    pub fn new(points: &PointSet<'data, S>, values: NdView<'data, 'layout, S>) -> Result<Self> {
        let shape = values.layout().shape();
        if D == 0 || points.dimension() != D || shape.rank() != 2 {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "point/value layout mismatch",
            ));
        }
        validate_count(points.len(), 1, scioxide_core::argument!("points"))?;
        validate_length(points.len(), shape.extents()[0])?;
        validate_finite(values.iter())?;
        Ok(Self {
            tree: KdTree::new(points)?,
            components: shape.extents()[1],
            values: values.storage(),
            value_layout: values.layout(),
        })
    }

    /// Returns the number of real value components `m` per sample.
    #[must_use]
    pub const fn components(&self) -> usize {
        self.components
    }

    /// Returns an upper bound on the memory of a fit of a dense `rows × columns`
    /// system under `options`.
    ///
    /// The bound covers the factorization, coefficients, neighbor search and
    /// polynomial exponents, and deliberately overestimates the SVD storage. It
    /// excludes allocator overhead and the borrowed samples and tree. A local
    /// RBF of `n` neighbors with `t` polynomial terms is a square system of
    /// order `n + t`. A local polynomial fit of `t` terms on at most `k`
    /// neighbors is `k × t`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `rows` or `columns` is zero, the minimum reciprocal
    ///   condition number of `options` is outside `[0, 1]`, or the order exceeds
    ///   [`LocalModelOptions::max_system_order`].
    /// - `SizeOverflow` if the bound overflows `usize`.
    pub fn preparation_bytes(
        &self,
        rows: usize,
        columns: usize,
        options: &LocalModelOptions<S>,
    ) -> Result<usize> {
        options.required_bytes::<D>(
            rows,
            columns,
            self.components,
            scioxide_core::argument!("options"),
        )
    }

    /// Selects the `count` nearest samples, naming `argument` for an invalid count.
    fn select(
        &self,
        query: [S; D],
        count: usize,
        context: ExecutionContext<'_>,
        argument: Argument,
    ) -> Result<Selected<S, D>> {
        if count == 0 || count > self.tree.points().len() {
            return Err(invalid(argument, "neighborhood count outside sample range"));
        }
        let mut workspace = self
            .tree
            .workspace(SearchWorkspaceRequest::new(count, SearchOptions::new()))?;
        let mut neighbors = try_filled(
            count,
            Neighbor::default(),
            scioxide_core::quantity!("neighbors"),
        )?;
        self.tree.nearest_into(
            &query,
            count,
            SearchOptions::new(),
            &mut neighbors,
            &mut workspace,
            context,
        )?;
        let mut points = try_with_capacity(count, scioxide_core::quantity!("neighbor_points"))?;
        let mut values = try_with_capacity(
            checked_mul(
                count,
                self.components,
                scioxide_core::quantity!("neighbor_values"),
            )?,
            scioxide_core::quantity!("neighbor_values"),
        )?;
        let mut indices = try_with_capacity(count, scioxide_core::quantity!("neighbor_indices"))?;
        for neighbor in neighbors {
            points.push(point(self.tree.points(), neighbor.index)?);
            indices.push(neighbor.index);
            for c in 0..self.components {
                values.push(self.values[self.value_layout.offset([neighbor.index, c])?]);
            }
        }
        Ok(Selected {
            points,
            values,
            indices,
            components: self.components,
        })
    }
}

/// Copies of the selected samples.
struct Selected<S: RealScalar, const D: usize> {
    points: Vec<[S; D]>,
    /// Values indexed `[sample][component]`.
    values: Vec<S>,
    indices: Vec<usize>,
    components: usize,
}

/// Factors a system and rejects it as `Singular` if it fails the rank or
/// condition limits.
fn checked_factor<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    kind: SystemKind,
    options: &LocalModelOptions<S>,
    context: ExecutionContext<'_>,
) -> Result<Factor<S>> {
    let mut factor_options = LinalgExecutionOptions::new()
        .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);
    if let Some(budget) = options.factor_iterations {
        factor_options = factor_options.with_iteration_budget(budget);
    }
    let factored = factor(
        matrix,
        kind,
        options.rank_threshold,
        options.minimum_rcond,
        factor_options,
        context,
    )?;
    if factored.rank != matrix.cols() || factored.reciprocal_condition < options.minimum_rcond {
        return Err(SciOxideError::singular(None));
    }
    Ok(factored)
}

/// Holds the scratch of one query of a local model or Gaussian-process model.
///
/// Get one from [`Interpolant::workspace`](crate::Interpolant::workspace) or
/// [`GaussianProcessModel::workspace`]. A workspace serves any model whose
/// systems are no larger than those it was made for; a smaller one is rejected
/// with `DimensionMismatch`.
///
/// ```
/// use scioxide_interp::{
///     ExtrapolationPolicy, Interpolant, LocalModelPlan, LocalPolynomialModel,
///     LocalPolynomialOptions, LocalSamples,
/// };
/// let samples = LocalSamples::new(&[[-1.0_f64], [0.0], [1.0]], &[1.0, 2.0, 3.0])?;
/// let points = samples.points()?;
/// let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values()?)?;
/// let model = LocalPolynomialModel::new(&plan, [0.0], LocalPolynomialOptions::new(1, 3, 3))?;
/// let mut workspace = model.workspace()?;
/// let mut out = [0.0];
/// model.evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out, &mut workspace)?;
/// assert!((out[0] - 2.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LocalPredictionWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    rhs: Vector<S>,
    solution: Vector<S>,
    scratch: LinalgWorkspace<'static, S>,
    basis: Vec<S>,
    output: Vec<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> LocalPredictionWorkspace<S> {
    /// Allocates scratch for `rows × cols` solves and `outputs` result components.
    fn new(rows: usize, cols: usize, outputs: usize) -> Result<Self> {
        Ok(Self {
            rhs: Vector::zeros(rows)?,
            solution: Vector::zeros(cols)?,
            scratch: {
                let mut scratch = LinalgWorkspace::new();
                scratch.try_reserve(&solve_requirements::<S>(rows, cols)?)?;
                scratch
            },
            basis: try_filled(rows.max(cols), S::ZERO, scioxide_core::quantity!("basis"))?,
            output: try_filled(outputs, S::ZERO, scioxide_core::quantity!("output"))?,
        })
    }

    /// Whether the basis and result buffers cover `basis` and `outputs` entries.
    fn covers(&self, basis: usize, outputs: usize) -> Capacity {
        if self.basis.len() >= basis && self.output.len() >= outputs {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        }
    }

    /// Whether the solve buffers also cover a square `n × n` system.
    fn covers_solve(&self, n: usize, outputs: usize) -> Capacity {
        let solve = solve_requirements::<S>(n, n)
            .and_then(|requirements| self.scratch.check(&requirements));
        if self.rhs.len() == n
            && self.solution.len() == n
            && matches!(solve, Ok(Capacity::Sufficient))
        {
            self.covers(n, outputs)
        } else {
            Capacity::NeedsGrowth
        }
    }
}

/// `DimensionMismatch` for an undersized prediction workspace.
fn require(capacity: Capacity, required: usize, available: usize) -> Result<()> {
    if capacity == Capacity::Sufficient {
        Ok(())
    } else {
        Err(crate::error::shape(required, available))
    }
}

fn solve<S: RealScalar + NumericScalar<Real = S>>(
    factor: &Factor<S>,
    w: &mut LocalPredictionWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<()> {
    factor.solve_into(&w.rhs, w.solution.view_mut(), &mut w.scratch, context)
}

/// Solves for every component and returns the largest residual.
fn coefficients<S: RealScalar + NumericScalar<Real = S>, const D: usize>(
    data: &Selected<S, D>,
    matrix: &Matrix<S>,
    factor: &Factor<S>,
    context: ExecutionContext<'_>,
) -> Result<(Vec<S>, S)> {
    // Solve each real value component against the shared factor and record
    // the largest residual |A x - b| over all rows and components. Coefficients
    // are stored component-major (`[component][column]`), matching
    // `evaluate_coefficients`.
    let mut w = LocalPredictionWorkspace::new(matrix.rows(), matrix.cols(), data.components)?;
    let mut result = try_with_capacity(
        checked_mul(
            matrix.cols(),
            data.components,
            scioxide_core::quantity!("coefficients"),
        )?,
        scioxide_core::quantity!("coefficients"),
    )?;
    let mut residual = S::ZERO;
    for c in 0..data.components {
        w.rhs.as_mut_slice().fill(S::ZERO);
        for (i, row) in data.values.chunks_exact(data.components).enumerate() {
            w.rhs.as_mut_slice()[i] = row[c];
        }
        solve(factor, &mut w, context)?;
        for row in 0..matrix.rows() {
            let sum = (0..matrix.cols()).try_fold(S::ZERO, |sum, j| {
                intermediate(sum + matrix.view().get(row, j)? * w.solution.as_slice()[j])
            })?;
            residual = residual.max(intermediate(sum - w.rhs.as_slice()[row])?.abs());
        }
        result.extend_from_slice(w.solution.as_slice());
    }
    Ok((result, residual))
}

/// Applies component-major coefficients to a basis.
fn evaluate_coefficients<S: RealScalar>(
    basis: &[S],
    coefficients: &[S],
    out: &mut [S],
) -> Result<()> {
    if basis.is_empty() {
        out.fill(S::ZERO);
        return Ok(());
    }
    for (value, weights) in out.iter_mut().zip(coefficients.chunks_exact(basis.len())) {
        *value = basis
            .iter()
            .zip(weights)
            .try_fold(S::ZERO, |sum, (&b, &w)| intermediate(sum + b * w))?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideErrorView;

    #[test]
    fn options_and_plan_report_limits() {
        let samples = LocalSamples::new(&[[0.0_f64], [1.0], [2.0]], &[1.0, 2.0, 3.0]).unwrap();
        let points = samples.points().unwrap();
        let plan = LocalModelPlan::<f64, 1>::new(&points, samples.values().unwrap()).unwrap();
        let options = LocalModelOptions::new().with_max_system_order(4);
        assert!(plan.preparation_bytes(3, 2, &options).unwrap() > 0);
        assert!(
            matches!(plan.preparation_bytes(5, 2, &options), Err(error) if matches!(error.view(), SciOxideErrorView::InvalidInput { .. }))
        );
        assert!(
            matches!(options.with_maximum_bytes(1).preflight::<1>(3, 2, 1, scioxide_core::argument!("options")), Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
        );
        assert!(
            options
                .with_minimum_rcond(2.0)
                .check(1, 1, scioxide_core::argument!("options"))
                .is_err()
        );
        let clone = plan.clone();
        assert_eq!(clone.components(), 1);
    }

    #[test]
    fn workspace_capacity_compares_basis_staging_and_solves() {
        let small = LocalPredictionWorkspace::<f64>::new(0, 2, 1).unwrap();
        assert_eq!(small.covers(2, 1), Capacity::Sufficient);
        assert_eq!(small.covers(3, 1), Capacity::NeedsGrowth);
        assert_eq!(small.covers_solve(2, 1), Capacity::NeedsGrowth);
        let square = LocalPredictionWorkspace::<f64>::new(3, 3, 3).unwrap();
        assert_eq!(square.covers_solve(3, 3), Capacity::Sufficient);
        assert_eq!(square.covers_solve(2, 3), Capacity::NeedsGrowth);
        assert!(require(Capacity::NeedsGrowth, 3, 2).is_err());
        let mut out = [9.0];
        evaluate_coefficients::<f64>(&[], &[], &mut out).unwrap();
        assert_eq!(out, [0.0]);
    }
}
