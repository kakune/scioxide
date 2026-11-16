//! Nelder–Mead simplex method.

use crate::attempt::{Accepted, AcceptedState, Entry, Shape};
use crate::context::SolveSession;
use crate::result::{OptimizeResult, OptimizeTermination};
use crate::traits::ObjectiveFunction;
use crate::utils::cast_usize;
use alloc::vec::Vec;
use core::ops::Neg;
use core::time::Duration;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::checked_sum_of_products;
use scioxide_core::{
    AttemptKind, Budget, EvaluationRole, ExecutionContext, ExecutionPolicy, NumericScalar,
    PartialResultTrust, RealScalar, ResidualTolerance, SciOxideError, SolveScope, StepTolerance,
};

use scioxide_core::backend::scaled::scaled_tolerance_threshold;
use scioxide_core::{Scaled, ScaledAccumulator};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, RankPolicy,
    RankReport,
};

use crate::validation::invalid;

/// Sets the stopping tolerances, limits, initial simplex and repair policy of [`NelderMead`].
///
/// With objective reference scale `s_f` and coordinate reference scale `s_x`,
/// the thresholds are `atol + rtol·s_f` for the objective spread and
/// `atol + rtol·s_x` for the simplex diameter. The defaults are the default
/// [`ResidualTolerance`] and [`StepTolerance`], reference scales 1, 1000
/// iterations, 2000 objective evaluations, two repairs, the default
/// [`RankThreshold`](scioxide_core::RankThreshold) and no time limit. The
/// setters do not check their arguments; [`NelderMead::new`] does. See also
/// [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct NelderMeadOptions<S: RealScalar> {
    function_tolerance: ResidualTolerance<S>,
    step_tolerance: StepTolerance<S>,
    max_iterations: Budget,
    time_limit: Option<Duration>,
    coordinate_scale: S,
    objective_scale: S,
    evaluations: Budget,
    repairs: usize,
    rank: scioxide_core::RankThreshold<S>,
    simplex: Option<Matrix<S>>,
}

impl<S: RealScalar> Default for NelderMeadOptions<S> {
    fn default() -> Self {
        Self {
            function_tolerance: ResidualTolerance::default(),
            step_tolerance: StepTolerance::default(),
            max_iterations: Budget::ONE.saturating_add(999),
            time_limit: None,
            coordinate_scale: S::ONE,
            objective_scale: S::ONE,
            evaluations: Budget::ONE.saturating_add(1999),
            repairs: 2,
            rank: scioxide_core::RankThreshold::default(),
            simplex: None,
        }
    }
}

impl<S: RealScalar> NelderMeadOptions<S> {
    /// Sets the tolerance on the objective spread.
    #[must_use]
    pub const fn with_function_tolerance(
        mut self,
        function_tolerance: ResidualTolerance<S>,
    ) -> Self {
        self.function_tolerance = function_tolerance;
        self
    }

    /// Sets the tolerance on the simplex diameter.
    #[must_use]
    pub const fn with_step_tolerance(mut self, tolerance: StepTolerance<S>) -> Self {
        self.step_tolerance = tolerance;
        self
    }

    /// Returns the tolerance on the simplex diameter.
    pub const fn step_tolerance(&self) -> StepTolerance<S> {
        self.step_tolerance
    }

    /// Sets the iteration limit.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets a wall-clock time limit. The execution context must provide a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }

    /// Returns the tolerance on the objective spread.
    pub const fn function_tolerance(&self) -> ResidualTolerance<S> {
        self.function_tolerance
    }

    /// Returns the iteration limit.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }

    /// Returns the time limit, if set.
    #[must_use]
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }
}

impl<S: RealScalar> NelderMeadOptions<S> {
    /// Sets the coordinate and objective reference scales (default 1 each).
    ///
    /// They scale the relative tolerances and the default initial simplex.
    #[must_use]
    pub const fn with_reference_scales(mut self, coordinate: S, objective: S) -> Self {
        self.coordinate_scale = coordinate;
        self.objective_scale = objective;
        self
    }
    /// Returns the coordinate reference scale.
    pub const fn coordinate_scale(&self) -> S {
        self.coordinate_scale
    }
    /// Returns the objective reference scale.
    pub const fn objective_scale(&self) -> S {
        self.objective_scale
    }
    /// Sets the objective evaluation limit, which includes the initial simplex
    /// and repairs.
    pub const fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.evaluations = evaluations;
        self
    }
    /// Returns the objective evaluation limit.
    pub const fn max_evaluations(&self) -> Budget {
        self.evaluations
    }
    /// Sets the largest number of simplex rebuilds; zero disables them. The default is two.
    pub const fn with_max_repairs(mut self, repairs: usize) -> Self {
        self.repairs = repairs;
        self
    }
    /// Returns the largest number of simplex rebuilds in one solve.
    pub const fn max_repairs(&self) -> usize {
        self.repairs
    }
    /// Sets the threshold for the numerical rank of the simplex edges, each
    /// normalized to unit length.
    pub const fn with_rank_threshold(mut self, rank: scioxide_core::RankThreshold<S>) -> Self {
        self.rank = rank;
        self
    }
    /// Returns the threshold for the numerical rank of the simplex edges.
    pub const fn rank_threshold(&self) -> scioxide_core::RankThreshold<S> {
        self.rank
    }
    /// Sets the initial simplex: an `(n + 1) × n` matrix with one vertex per row.
    ///
    /// The first row must equal the start point passed to the solve.
    #[must_use]
    pub fn with_initial_simplex(mut self, simplex: Matrix<S>) -> Self {
        self.simplex = Some(simplex);
        self
    }
    /// Returns the initial simplex, if set.
    pub fn initial_simplex(&self) -> Option<scioxide_linalg::MatrixView<'_, S>> {
        self.simplex.as_ref().map(Matrix::view)
    }
}

/// Describes the numerical rank of the simplex edges at the last check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NelderMeadGeometry {
    /// The simplex has not been checked since it was built or changed.
    Incomplete,
    /// The normalized edges have full numerical rank `n`.
    FullRank,
    /// The normalized edges are numerically rank deficient.
    Degenerate,
}

/// Reports the state of the simplex at the end of a [`NelderMead`] solve.
///
/// These values describe the sampled vertices; they do not prove that the
/// best vertex is a minimum. A value is `None` when it was not computed or is
/// not representable in `S`.
#[derive(Debug, Clone, Copy)]
pub struct NelderMeadDiagnostics<S: RealScalar> {
    initialized_vertices: usize,
    geometry: NelderMeadGeometry,
    rank: Option<RankReport<S>>,
    spread: Option<S>,
    diameter: Option<S>,
    function_threshold: Option<S>,
    step_threshold: Option<S>,
    repairs: usize,
    iterations: usize,
}
impl<S: RealScalar> Default for NelderMeadDiagnostics<S> {
    fn default() -> Self {
        Self {
            initialized_vertices: 0,
            geometry: NelderMeadGeometry::Incomplete,
            rank: None,
            spread: None,
            diameter: None,
            function_threshold: None,
            step_threshold: None,
            repairs: 0,
            iterations: 0,
        }
    }
}
impl<S: RealScalar> NelderMeadDiagnostics<S> {
    /// Returns the number of vertices with a finite objective value.
    pub const fn initialized_vertices(self) -> usize {
        self.initialized_vertices
    }
    /// Returns the rank status of the simplex edges.
    pub const fn geometry(self) -> NelderMeadGeometry {
        self.geometry
    }
    /// Returns the numerical rank of the normalized edges.
    pub const fn rank(self) -> Option<RankReport<S>> {
        self.rank
    }
    /// Returns the objective spread: largest minus smallest vertex value.
    pub const fn objective_spread(self) -> Option<S> {
        self.spread
    }
    /// Returns the simplex diameter: largest ∞-norm distance between two vertices.
    pub const fn diameter(self) -> Option<S> {
        self.diameter
    }
    /// Returns the threshold on the objective spread.
    pub const fn function_threshold(self) -> Option<S> {
        self.function_threshold
    }
    /// Returns the threshold on the simplex diameter.
    pub const fn step_threshold(self) -> Option<S> {
        self.step_threshold
    }
    /// Returns the number of simplex rebuilds.
    pub const fn repairs(self) -> usize {
        self.repairs
    }
    /// Returns the number of completed iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
}

/// Minimizes a function without derivatives by the Nelder–Mead simplex method.
///
/// The method keeps `n + 1` vertices and replaces the worst one by reflection,
/// expansion or contraction through the centroid of the others, or shrinks the
/// simplex toward the best vertex. It uses the dimension-dependent
/// coefficients of Gao and Han: reflection 1, expansion `1 + 2/n`, contraction
/// `3/4 − 1/(2n)` and shrinkage `1 − 1/n`. The default initial simplex adds
/// `0.05 c` to one coordinate of `x0` at a time, where `c` is the coordinate
/// reference scale; [`NelderMeadOptions::with_initial_simplex`] sets it
/// explicitly. When the simplex becomes numerically degenerate, the method
/// rebuilds it around the best vertex, at most
/// [`NelderMeadOptions::max_repairs`] times; after that the solve ends with
/// [`OptimizeTermination::Stagnation`](crate::OptimizeTermination::Stagnation).
///
/// The solve converges when both the objective spread (largest minus smallest
/// vertex value) and the simplex diameter (largest ∞-norm distance between two
/// vertices) are below their thresholds, and the edges of the simplex have full
/// numerical rank. An iteration costs one or two objective evaluations, or
/// `n` for a shrink, plus O(n²) arithmetic. Nelder–Mead suits small,
/// nonsmooth or noisy problems; it can stall at a point that is not a minimum.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError};
/// use scioxide_optimize::{NelderMead, ObjectiveFn, OptimizeResult};
///
/// # fn main() -> Result<(), SciOxideError> {
/// // x₀² + x₁² has its minimum 0 at the origin.
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
/// let solver = NelderMead::<f64>::default();
/// let mut workspace = solver.workspace(2)?;
/// let mut output = OptimizeResult::<f64>::empty();
/// solver.minimize_into(
///     &objective,
///     &[1.0, -1.0],
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!(output.is_success() && output.fun() < 1e-8);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - J. A. Nelder and R. Mead, "A simplex method for function minimization",
///   *Comput. J.* 7(4), 308–313, 1965.
/// - F. Gao and L. Han, "Implementing the Nelder–Mead simplex algorithm with
///   adaptive parameters", *Comput. Optim. Appl.* 51(1), 259–277, 2012.
#[derive(Debug, Clone)]
pub struct NelderMead<S: RealScalar> {
    options: NelderMeadOptions<S>,
}

impl<S: RealScalar> Default for NelderMead<S> {
    fn default() -> Self {
        Self {
            options: NelderMeadOptions::default(),
        }
    }
}
impl<S: RealScalar> NelderMead<S> {
    /// Creates the solver after checking `options`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a reference scale is not positive and finite, or the
    ///   initial simplex is not `(n + 1) × n`.
    /// - `NonFinite` if the initial simplex contains NaN or infinity.
    pub fn new(options: NelderMeadOptions<S>) -> Result<Self, SciOxideError> {
        for (scale, argument) in [
            (
                options.coordinate_scale,
                scioxide_core::argument!("coordinate"),
            ),
            (
                options.objective_scale,
                scioxide_core::argument!("objective"),
            ),
        ] {
            if !scale.is_finite() || scale <= S::ZERO {
                return Err(SciOxideError::invalid_input(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "reference scales must be positive and finite",
                ));
            }
        }
        if let Some(simplex) = &options.simplex {
            if simplex.cols().checked_add(1) != Some(simplex.rows()) {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("simplex"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected n + 1 rows and n columns",
                ));
            }
            crate::utils::validate_finite_origin(
                simplex.as_slice(),
                scioxide_core::NonFiniteOrigin::Input,
            )?;
        }
        Ok(Self { options })
    }

    /// Returns the options.
    pub const fn options(&self) -> &NelderMeadOptions<S> {
        &self.options
    }
}

/// Simplex, ordering, centroid and candidate buffers for Nelder–Mead: O(n²) storage.
#[derive(Debug)]
pub(crate) struct NelderMeadStorage<S: RealScalar> {
    dimension: usize,
    simplex: Vec<S>,
    values: Vec<S>,
    order: Vec<usize>,
    centroid: Vec<S>,
    reflected: Vec<S>,
    candidate: Vec<S>,
    best: Vec<S>,
    best_value: Option<S>,
    iterations: usize,
    edges: Matrix<S>,
    qr: HouseholderDecomposition<S>,
    incremental: scioxide_linalg::QrUpdateDecomposition<S>,
    incremental_work: LinalgWorkspace<'static, S>,
    previous_edges: Matrix<S>,
    edge_delta: Matrix<S>,
    edge_basis: Matrix<S>,
    incremental_valid: bool,
    geometry_updates: usize,
    thresholds: [Scaled<S>; 2],
    qr_work: LinalgWorkspace<'static, S>,
    diagnostics: NelderMeadDiagnostics<S>,
}

impl<S: RealScalar> Default for NelderMeadStorage<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> NelderMeadStorage<S> {
    /// Create empty storage without allocating.
    fn new() -> Self {
        Self {
            dimension: 0,
            simplex: Vec::new(),
            values: Vec::new(),
            order: Vec::new(),
            centroid: Vec::new(),
            reflected: Vec::new(),
            candidate: Vec::new(),
            best: Vec::new(),
            best_value: None,
            iterations: 0,
            edges: Matrix::empty(),
            qr: HouseholderDecomposition::empty(),
            incremental: scioxide_linalg::QrUpdateDecomposition::empty(),
            incremental_work: LinalgWorkspace::new(),
            previous_edges: Matrix::empty(),
            edge_delta: Matrix::empty(),
            edge_basis: Matrix::empty(),
            incremental_valid: false,
            geometry_updates: 0,
            thresholds: [Scaled::ZERO; 2],
            qr_work: LinalgWorkspace::new(),
            diagnostics: NelderMeadDiagnostics::default(),
        }
    }
    fn fits(&self, dimension: usize) -> bool {
        self.dimension == dimension && self.values.len() == dimension.saturating_add(1)
    }
    /// Prepare storage for a new dimension, keeping existing capacity.
    fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        if self.fits(dimension) {
            return Ok(());
        }
        let vertices = checked_add(
            dimension,
            1,
            scioxide_core::quantity!("nelder_mead_vertices"),
        )?;
        let entries = checked_mul(
            vertices,
            dimension,
            scioxide_core::quantity!("nelder_mead_simplex"),
        )?;
        for (buffer, length) in [
            (&mut self.simplex, entries),
            (&mut self.values, vertices),
            (&mut self.centroid, dimension),
            (&mut self.reflected, dimension),
            (&mut self.candidate, dimension),
            (&mut self.best, dimension),
        ] {
            try_reserve(
                buffer,
                length.saturating_sub(buffer.len()),
                scioxide_core::quantity!("nelder_mead_storage"),
            )?;
            buffer.resize(length, S::ZERO);
        }
        let additional = vertices.saturating_sub(self.order.len());
        try_reserve(
            &mut self.order,
            additional,
            scioxide_core::quantity!("nelder_mead_order"),
        )?;
        self.order.resize(vertices, 0);
        self.previous_edges.try_resize_flat(dimension, dimension)?;
        self.edge_delta.try_resize_flat(dimension, 1)?;
        self.edge_basis.try_resize_flat(dimension, 1)?;
        self.edges.try_resize_flat(dimension, dimension)?;
        self.edges.as_mut_slice().fill(S::ZERO);
        if dimension > 0 {
            for _ in 0..2 {
                HouseholderDecomposition::factor_into(
                    &self.edges,
                    HouseholderOrientation::Qr,
                    scioxide_linalg::HouseholderPivoting::Column,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    &mut self.qr,
                    &mut self.qr_work,
                    scioxide_core::ExecutionContext::sequential(),
                )?;
            }
        }
        if dimension > 2 {
            for i in 0..dimension {
                self.edges.as_mut_slice()[i * dimension + i] = S::ONE;
            }
            let options = scioxide_linalg::QrUpdateOptions::new(RankPolicy::DimensionScaledEpsilon);
            for _ in 0..2 {
                scioxide_linalg::QrUpdateDecomposition::factor_into(
                    self.edges.view(),
                    &options,
                    &mut self.incremental,
                    &mut self.incremental_work,
                    ExecutionContext::sequential(),
                )?;
                self.incremental.modify_in_place(
                    scioxide_linalg::QrModification::RankUpdate {
                        u: self.edge_delta.view(),
                        v: self.edge_basis.view(),
                    },
                    &options,
                    &mut self.incremental_work,
                    ExecutionContext::sequential(),
                )?;
            }
        }
        self.dimension = dimension;
        Ok(())
    }
    /// Invalidate geometry after a vertex mutation; thresholds survive.
    fn invalidate_geometry(&mut self) {
        self.diagnostics.geometry = NelderMeadGeometry::Incomplete;
        self.diagnostics.rank = None;
        self.diagnostics.spread = None;
        self.diagnostics.diameter = None;
    }
}

impl<S: RealScalar> Accepted<S> for NelderMeadStorage<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        Some(AcceptedState {
            point: &self.best,
            value: self.best_value?,
            gradient: None,
            iterations: self.iterations,
        })
    }
}

/// Describes the workspace that a [`NelderMead`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NelderMeadWorkspaceRequirements {
    dimension: usize,
}

impl NelderMeadWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
}

impl<S: RealScalar> Shape<NelderMeadStorage<S>> for NelderMeadWorkspaceRequirements {
    fn fits(&self, storage: &NelderMeadStorage<S>) -> bool {
        storage.fits(self.dimension)
    }
    fn prepare(&self, storage: &mut NelderMeadStorage<S>) -> Result<(), SciOxideError> {
        storage.try_reserve(self.dimension)
    }
}

crate::attempt::local_workspace!(
    /// Holds the O(n²) storage and the record of the last call for [`NelderMead`].
    NelderMeadWorkspace,
    NelderMeadStorage<S>,
    NelderMeadWorkspaceRequirements
);

impl<S: RealScalar + NumericScalar<Real = S>> NelderMeadWorkspace<S> {
    /// Returns the simplex of the last call, one vertex per row of length `n`.
    ///
    /// Only the first [`NelderMeadDiagnostics::initialized_vertices`] rows are valid.
    pub fn simplex(&self) -> &[S] {
        &self.storage.simplex
    }
    /// Returns the objective values of the valid vertices of the last call.
    pub fn values(&self) -> &[S] {
        &self.storage.values[..self.storage.diagnostics.initialized_vertices]
    }
}

fn affine_trial<S: RealScalar>(
    center: &[S],
    worst: &[S],
    factor: S,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    output
        .iter_mut()
        .zip(center.iter().zip(worst))
        .try_for_each(|(trial, (&center, &worst))| {
            let (difference, tail) = scioxide_core::backend::two_sum(center, -worst);
            *trial = if difference.is_finite() && tail == S::ZERO {
                crate::utils::trial_component(center, factor, difference)?
            } else {
                checked_sum_of_products(&[center, center, worst], &[S::ONE, factor, -factor])?
            };
            Ok(())
        })
}

fn observe<S, F>(
    f: &F,
    point: &[S],
    best: &mut [S],
    best_value: &mut Option<S>,
    session: &SolveSession<'_, '_, '_>,
    limit: usize,
) -> Result<S, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    if session.provider_count(EvaluationRole::Objective)? >= limit {
        session.set_stopped(OptimizeTermination::EvaluationLimit);
        return Err(session.stop_error(OptimizeTermination::EvaluationLimit)?);
    }
    crate::utils::validate_finite(point)?;
    // A non-finite trial value ranks above every finite vertex, so the
    // simplex contracts or shrinks away from it.
    let value = f.eval(point)?;
    if !value.is_finite() {
        return Ok(S::INFINITY);
    }
    if best_value.is_none_or(|best| value < best) {
        best.copy_from_slice(point);
        *best_value = Some(value);
    }
    Ok(value)
}

fn replace_vertex<S: RealScalar>(
    workspace: &mut NelderMeadStorage<S>,
    row: usize,
    value: S,
    reflection: bool,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.accept(PartialResultTrust::Admissible)?;
    workspace.invalidate_geometry();
    let point = if reflection {
        &workspace.reflected
    } else {
        &workspace.candidate
    };
    workspace.simplex[row * workspace.dimension..(row + 1) * workspace.dimension]
        .copy_from_slice(point);
    workspace.values[row] = value;
    Ok(())
}

fn nm_step<S, F>(
    f: &F,
    options: &NelderMeadOptions<S>,
    workspace: &mut NelderMeadStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    let n = workspace.dimension;
    let best = workspace.order[0];
    let worst = workspace.order[n];
    let second_worst = workspace.order[n - 1];
    let reciprocal = <S as scioxide_core::AlgebraicScalar>::ONE / cast_usize::<S>(n)?;
    let (expansion, contraction, shrink) = coefficients::<S>(n)?;
    if n == 1 {
        workspace.centroid[0] = workspace.simplex[best];
    } else if n == 2
        && (0..n).all(|column| {
            let origin = workspace.simplex[workspace.order[0] * n + column];
            let other = workspace.simplex[workspace.order[1] * n + column];
            let delta = other - origin;
            let half = delta * S::HALF;
            let center = origin + half;
            workspace.centroid[column] = center;
            delta.is_finite()
                && center.is_finite()
                && !scioxide_core::backend::standard::product_needs_retry(delta, S::HALF, half)
        })
    {
        // Same rounded difference/half/sum as the extended-range centroid.
    } else {
        for column in 0..n {
            let origin = Scaled::from_value(workspace.simplex[best * n + column])?;
            let mut sum = ScaledAccumulator::from_scaled(origin);
            for &row in &workspace.order[..n] {
                let difference =
                    Scaled::from_value(workspace.simplex[row * n + column])?.sub(origin)?;
                sum.add_scaled(difference.mul(Scaled::from_value(reciprocal)?)?)?;
            }
            workspace.centroid[column] = sum.round()?;
        }
    }
    session.proposal()?;
    affine_trial(
        &workspace.centroid,
        &workspace.simplex[worst * n..(worst + 1) * n],
        <S as scioxide_core::AlgebraicScalar>::ONE,
        &mut workspace.reflected,
    )?;
    let reflected = observe(
        f,
        &workspace.reflected,
        &mut workspace.best,
        &mut workspace.best_value,
        session,
        options.max_evaluations().get(),
    )?;
    if reflected >= workspace.values[best] && reflected < workspace.values[second_worst] {
        return replace_vertex(workspace, worst, reflected, true, session);
    }
    session.proposal()?;
    if reflected < workspace.values[best] {
        affine_trial(
            &workspace.centroid,
            &workspace.simplex[worst * n..(worst + 1) * n],
            expansion,
            &mut workspace.candidate,
        )?;
        let expanded = observe(
            f,
            &workspace.candidate,
            &mut workspace.best,
            &mut workspace.best_value,
            session,
            options.max_evaluations().get(),
        )?;
        session.reject()?;
        return if expanded < reflected {
            replace_vertex(workspace, worst, expanded, false, session)
        } else {
            replace_vertex(workspace, worst, reflected, true, session)
        };
    }
    let outside = reflected < workspace.values[worst];
    let factor = if outside { contraction } else { -contraction };
    affine_trial(
        &workspace.centroid,
        &workspace.simplex[worst * n..(worst + 1) * n],
        factor,
        &mut workspace.candidate,
    )?;
    let contracted = observe(
        f,
        &workspace.candidate,
        &mut workspace.best,
        &mut workspace.best_value,
        session,
        options.max_evaluations().get(),
    )?;
    session.reject()?;
    if (outside && contracted <= reflected) || (!outside && contracted < workspace.values[worst]) {
        return replace_vertex(workspace, worst, contracted, false, session);
    }
    session.reject()?;
    workspace.incremental_valid = false;
    for index in 1..=n {
        let row = workspace.order[index];
        session.proposal()?;
        // A negative coefficient interpolates between the best and this vertex.
        affine_trial(
            &workspace.simplex[best * n..(best + 1) * n],
            &workspace.simplex[row * n..(row + 1) * n],
            -shrink,
            &mut workspace.candidate,
        )?;
        let value = observe(
            f,
            &workspace.candidate,
            &mut workspace.best,
            &mut workspace.best_value,
            session,
            options.max_evaluations().get(),
        )?;
        replace_vertex(workspace, row, value, false, session)?;
    }
    Ok(())
}

fn coefficients<S: RealScalar>(n: usize) -> Result<(S, S, S), SciOxideError> {
    if n <= 1 {
        return Ok((S::TWO, S::HALF, S::HALF));
    }
    let reciprocal = S::ONE / cast_usize::<S>(n)?;
    Ok((
        S::ONE + S::TWO * reciprocal,
        S::try_from_f64_lossy(0.75)? - S::HALF * reciprocal,
        S::ONE - reciprocal,
    ))
}

fn offset_vertex<S: RealScalar>(
    point: &mut [S],
    column: usize,
    scale: S,
) -> Result<bool, SciOxideError> {
    let origin = point[column];
    let offset =
        Scaled::from_value(scale)?.mul(Scaled::from_value(S::try_from_f64_lossy(0.05)?)?)?;
    for delta in [offset, offset.neg()] {
        // Failure to round a finite extended-range sum means this direction
        // cannot represent the requested offset; try the opposite direction.
        if let Ok(value) = Scaled::from_value(origin)?.add(delta)?.round()
            && value != origin
        {
            point[column] = value;
            return Ok(true);
        }
    }
    Ok(false)
}

fn repair<S, F>(
    f: &F,
    w: &mut NelderMeadStorage<S>,
    options: &NelderMeadOptions<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    w.centroid.copy_from_slice(&w.best);
    w.candidate.copy_from_slice(&w.best);
    let value = w.best_value.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("workspace"),
            "repair requires a validated observation",
        )
    })?;
    replace_vertex(w, 0, value, false, session)?;
    w.diagnostics.initialized_vertices = 1;
    for row in 1..=w.dimension {
        w.candidate.copy_from_slice(&w.centroid);
        if !offset_vertex(&mut w.candidate, row - 1, options.coordinate_scale())? {
            return Ok(false);
        }
        session.proposal()?;
        let value = observe(
            f,
            &w.candidate,
            &mut w.best,
            &mut w.best_value,
            session,
            options.max_evaluations().get(),
        )?;
        replace_vertex(w, row, value, false, session)?;
        w.diagnostics.initialized_vertices = row + 1;
    }
    for (index, entry) in w.order.iter_mut().enumerate() {
        *entry = index;
    }
    Ok(true)
}

fn inspect_geometry<S: RealScalar>(
    w: &mut NelderMeadStorage<S>,
    options: &NelderMeadOptions<S>,
    force: bool,
) -> Result<bool, SciOxideError> {
    w.incremental_valid = false;
    let n = w.dimension;
    let mut diameter = Scaled::ZERO;
    for column in 0..n {
        let (lo, hi) = w
            .simplex
            .chunks_exact(n)
            .map(|row| row[column])
            .fold((w.simplex[column], w.simplex[column]), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            });
        let extent = Scaled::from_value(hi)?.sub(Scaled::from_value(lo)?)?;
        if !extent.abs_le(diameter) {
            diameter = extent;
        }
    }
    let (lo, hi) = w
        .values
        .iter()
        .copied()
        .fold((w.values[0], w.values[0]), |(lo, hi), x| {
            (lo.min(x), hi.max(x))
        });
    // A rejected vertex ranks at +inf, so the simplex has no finite spread.
    let spread = if hi.is_finite() {
        Some(Scaled::from_value(hi)?.sub(Scaled::from_value(lo)?)?)
    } else {
        None
    };
    // Normalize each edge before scalar conversion: opposite finite endpoints
    // can have an unrepresentable raw difference but a perfectly usable direction.
    for column in 0..n {
        let edge_values = &mut w.candidate;
        for (row, edge) in edge_values.iter_mut().enumerate() {
            *edge = w.simplex[(column + 1) * n + row] - w.simplex[row];
        }
        let scale = edge_values
            .iter()
            .fold(S::ZERO, |scale, &x| scale.max(x.abs()));
        if edge_values.iter().all(|x| x.is_finite()) {
            for (row, &edge) in edge_values.iter().enumerate() {
                w.edges.as_mut_slice()[row * n + column] = if scale.is_zero() {
                    S::ZERO
                } else {
                    edge / scale
                };
            }
            continue;
        }
        let mut scale = Scaled::ZERO;
        for row in 0..n {
            let edge = Scaled::from_value(w.simplex[(column + 1) * n + row])?
                .sub(Scaled::from_value(w.simplex[row])?)?;
            if !edge.abs_le(scale) {
                scale = edge.abs();
            }
        }
        for row in 0..n {
            w.edges.as_mut_slice()[row * n + column] = if scale.is_zero() {
                S::ZERO
            } else {
                Scaled::from_value(w.simplex[(column + 1) * n + row])?
                    .sub(Scaled::from_value(w.simplex[row])?)?
                    .div(scale)?
                    .round()?
            };
        }
    }
    let [function_threshold, step_threshold] = w.thresholds;
    let converged = spread.is_some_and(|spread| spread.abs_le(function_threshold))
        && diameter.abs_le(step_threshold);
    w.diagnostics.spread = spread.and_then(|spread| spread.round().ok());
    w.diagnostics.diameter = diameter.round().ok();
    // A determinant bound certifies the common 1/2-D full-rank case. Near
    // the threshold the authoritative pivoted factorization decides instead.
    if !force && !converged && n <= 2 {
        let full = if n == 1 {
            w.edges.as_slice()[0].abs() > options.rank_threshold().threshold(S::ONE)?
        } else {
            let a = w.edges.as_slice();
            let det = a[0] * a[3] - a[1] * a[2];
            // ||A||_2 <= ||A||_F <= 2 for column-max normalized edges.
            det.abs() > S::try_from_usize_exact(64)? * options.rank_threshold().threshold(S::TWO)?
        };
        if full {
            w.diagnostics.geometry = NelderMeadGeometry::FullRank;
            return Ok(false);
        }
    }
    let update_options =
        scioxide_linalg::QrUpdateOptions::new(RankPolicy::Threshold(options.rank_threshold()));
    if !force && !converged && n > 2 && w.incremental_valid && w.geometry_updates < 32 {
        for column in 0..n {
            let mut changed = false;
            for row in 0..n {
                let difference = w.edges.as_slice()[row * n + column]
                    - w.previous_edges.as_slice()[row * n + column];
                w.edge_delta.as_mut_slice()[row] = difference;
                changed |= difference != S::ZERO;
            }
            if changed {
                w.edge_basis.as_mut_slice().fill(S::ZERO);
                w.edge_basis.as_mut_slice()[column] = S::ONE;
                w.incremental.modify_in_place(
                    scioxide_linalg::QrModification::RankUpdate {
                        u: w.edge_delta.view(),
                        v: w.edge_basis.view(),
                    },
                    &update_options,
                    &mut w.incremental_work,
                    ExecutionContext::sequential(),
                )?;
            }
        }
        w.previous_edges
            .as_mut_slice()
            .copy_from_slice(w.edges.as_slice());
        w.geometry_updates += 1;
        if w.incremental.rank_decision().report().is_full_rank() {
            w.diagnostics.geometry = NelderMeadGeometry::FullRank;
            return Ok(false);
        }
    }
    HouseholderDecomposition::factor_into(
        &w.edges,
        HouseholderOrientation::Qr,
        scioxide_linalg::HouseholderPivoting::Column,
        &scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential),
        &mut w.qr,
        &mut w.qr_work,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    let rank =
        w.qr.analyze_rank(RankPolicy::Threshold(options.rank_threshold()))?;
    if n > 2 {
        scioxide_linalg::QrUpdateDecomposition::from_householder_into(
            &w.qr,
            &update_options,
            &mut w.incremental,
            &mut w.incremental_work,
            ExecutionContext::sequential(),
        )?;
        w.previous_edges
            .as_mut_slice()
            .copy_from_slice(w.edges.as_slice());
        w.incremental_valid = true;
        w.geometry_updates = 0;
    }

    w.diagnostics.geometry = if rank.rank() == n {
        NelderMeadGeometry::FullRank
    } else {
        NelderMeadGeometry::Degenerate
    };
    w.diagnostics.rank = Some(rank);
    Ok(converged)
}

impl<S: RealScalar + NumericScalar<Real = S>> NelderMead<S> {
    /// Returns the workspace requirements for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the size of the simplex overflows `usize`.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<NelderMeadWorkspaceRequirements, SciOxideError> {
        let vertices = checked_add(
            dimension,
            1,
            scioxide_core::quantity!("nelder_mead_vertices"),
        )?;
        checked_mul(
            vertices,
            dimension,
            scioxide_core::quantity!("nelder_mead_simplex"),
        )?;
        Ok(NelderMeadWorkspaceRequirements { dimension })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(&self, dimension: usize) -> Result<NelderMeadWorkspace<S>, SciOxideError> {
        let mut workspace = NelderMeadWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has a finite vertex,
    /// the call returns the best vertex with an unsuccessful termination. On
    /// error, `output` is unchanged and [`NelderMeadWorkspace::last_attempt`]
    /// describes the failed call.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `f` has bounds.
    /// - `DimensionMismatch` if `x0.len()` differs from `f.dim()`.
    /// - `NonFinite` if `x0` contains NaN or infinity.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation
    ///   or deadline stops the solve before it has a finite vertex.
    /// - `InvalidInput` if the initial simplex does not have `x0.len()`
    ///   columns or its first row differs from `x0`.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by `f` are returned unchanged.
    pub fn minimize_into<'scope, 'capability, F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut NelderMeadWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::validation::unbounded(f, x0)?;
        if let Some(simplex) = self.options.initial_simplex() {
            if simplex.cols() != x0.len() {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "initial simplex dimension does not match objective",
                ));
            }
            for (column, &value) in x0.iter().enumerate() {
                if simplex.get(0, column)? != value {
                    return Err(invalid(
                        scioxide_core::argument!("x0"),
                        "initial simplex first vertex must equal start",
                    ));
                }
            }
        }
        workspace.try_reserve(&self.workspace_requirements(x0.len())?)?;
        crate::attempt::preflight(&scope, self.options.time_limit())?;
        let entry = Entry::Call {
            identity: crate::context::local_identity(
                scioxide_core::algorithm_id!("optimize", "NelderMead", 1),
                x0.len(),
                &self.options,
            ),
            budgets: super::settings::iteration_budgets(self.options.max_iterations()),
        };
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| self.minimize_in_context(f, x0, staged, storage, context),
        )
    }

    fn minimize_in_context<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut NelderMeadStorage<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        workspace.try_reserve(x0.len())?;
        output.try_reserve_without_gradient(x0.len())?;
        workspace.diagnostics = NelderMeadDiagnostics::default();
        workspace.incremental_valid = false;
        workspace.best_value = None;
        workspace.iterations = 0;
        if let Some(limit) = self.options.time_limit() {
            context.limit_time(limit)?;
        }
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        let outcome = self.minimize_kernel(&session.observe(f), x0, workspace, &session);
        let termination = match outcome {
            Ok(termination) => termination,
            Err(error) => match (session.stopped(), workspace.best_value) {
                (Some(termination), Some(_)) => termination,
                _ => return Err(error),
            },
        };
        if workspace.dimension > 0
            && workspace.diagnostics.geometry != NelderMeadGeometry::Incomplete
        {
            inspect_geometry(workspace, &self.options, true)?;
        }
        let value = workspace.best_value.ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no validated vertex",
            )
        })?;
        crate::utils::commit_optimize_result(
            output,
            &workspace.best,
            value,
            None,
            workspace.iterations,
            termination,
            "Nelder-Mead solve completed",
        )?;
        crate::utils::synchronize_metadata(output, &session, PartialResultTrust::Admissible)?;
        output.method = crate::OptimizeDiagnostics::NelderMead(workspace.diagnostics);
        Ok(())
    }
    fn minimize_kernel<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        workspace: &mut NelderMeadStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let n = x0.len();
        workspace.thresholds = [
            scaled_tolerance_threshold(
                &self.options.function_tolerance().tolerance(),
                Scaled::from_value(self.options.objective_scale())?,
            )?,
            scaled_tolerance_threshold(
                &self.options.step_tolerance().tolerance(),
                Scaled::from_value(self.options.coordinate_scale())?,
            )?,
        ];
        workspace.diagnostics.function_threshold = workspace.thresholds[0].round().ok();
        workspace.diagnostics.step_threshold = workspace.thresholds[1].round().ok();
        // Evaluate the start even if representable initial geometry cannot be built.
        workspace.candidate.copy_from_slice(x0);
        session.proposal()?;
        let value = observe(
            f,
            &workspace.candidate,
            &mut workspace.best,
            &mut workspace.best_value,
            session,
            self.options.max_evaluations().get(),
        )?;
        if workspace.best_value.is_none() {
            // Unlike later trials, the caller's start must be a finite observation.
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "objective returned a non-finite value at the start",
            ));
        }
        replace_vertex(workspace, 0, value, false, session)?;
        workspace.diagnostics.initialized_vertices = 1;
        if n == 0 {
            workspace.diagnostics.geometry = NelderMeadGeometry::FullRank;
            workspace.diagnostics.spread = Some(<S as scioxide_core::AlgebraicScalar>::ZERO);
            workspace.diagnostics.diameter = Some(<S as scioxide_core::AlgebraicScalar>::ZERO);
            return Ok(OptimizeTermination::Converged);
        }
        for row in 1..=n {
            workspace.candidate.copy_from_slice(x0);
            if let Some(simplex) = self.options.initial_simplex() {
                for (column, value) in workspace.candidate.iter_mut().enumerate() {
                    *value = simplex.get(row, column)?;
                }
            } else if !offset_vertex(
                &mut workspace.candidate,
                row - 1,
                self.options.coordinate_scale(),
            )? {
                return Ok(OptimizeTermination::Stagnation);
            }
            session.proposal()?;
            let value = observe(
                f,
                &workspace.candidate,
                &mut workspace.best,
                &mut workspace.best_value,
                session,
                self.options.max_evaluations().get(),
            )?;
            replace_vertex(workspace, row, value, false, session)?;
            workspace.diagnostics.initialized_vertices = row + 1;
        }
        for (index, entry) in workspace.order.iter_mut().enumerate() {
            *entry = index;
        }
        loop {
            let converged = inspect_geometry(workspace, &self.options, false)?;
            if workspace.diagnostics.geometry == NelderMeadGeometry::Degenerate {
                if workspace.diagnostics.repairs >= self.options.max_repairs() {
                    return Ok(OptimizeTermination::Stagnation);
                }
                if let Some(stop) = session.iteration()? {
                    return Ok(stop);
                }
                workspace.diagnostics.repairs += 1;
                if !repair(f, workspace, &self.options, session)? {
                    return Ok(OptimizeTermination::Stagnation);
                }
                continue;
            }
            if converged {
                return Ok(OptimizeTermination::Converged);
            }
            if workspace.iterations == self.options.max_iterations().get() {
                return Ok(OptimizeTermination::IterationLimit);
            }
            if let Some(stop) = session.iteration()? {
                return Ok(stop);
            }
            // Stable in-place permutation; vertex storage never moves during ordering.
            for index in 1..=n {
                let mut position = index;
                while position > 0
                    && workspace.values[workspace.order[position]]
                        < workspace.values[workspace.order[position - 1]]
                {
                    workspace.order.swap(position, position - 1);
                    position -= 1;
                }
            }
            nm_step(f, &self.options, workspace, session)?;
            workspace.iterations += 1;
            workspace.diagnostics.iterations = workspace.iterations;
        }
    }
}

impl<S> NelderMead<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Minimizes the objective with this method's settings and owned workspace.
    ///
    /// Returns the complete result, including unsuccessful termination. Call
    /// `require_success()` to require the stopping test to pass.
    ///
    /// # Errors
    ///
    /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
    pub fn minimize<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(x0.len())?;
        let mut output = OptimizeResult::empty();
        self.minimize_into(
            f,
            x0,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S, F> crate::traits::Optimizer<S, F> for NelderMead<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    fn minimize(&self, f: &F, x0: &[S]) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(f, x0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    #[test]
    fn affine_trials_and_centroids_retain_the_complete_expression() {
        let mut point64 = [0.0];
        let mut point32 = [0.0];
        affine_trial(&[f64::MAX], &[f64::MAX], 2.0, &mut point64).unwrap();
        affine_trial(&[f32::MAX], &[f32::MAX], 2.0, &mut point32).unwrap();
        assert_eq!(point64, [f64::MAX]);
        assert_eq!(point32, [f32::MAX]);
        macro_rules! check_centroid {
            ($scalar:ty) => {{
                let objective = crate::ObjectiveFn::new(8, |_: &[$scalar]| Ok(0.0));
                let solver = NelderMead::new(
                    NelderMeadOptions::default()
                        .with_reference_scales(<$scalar>::MAX / 4.0, 1.0)
                        .with_max_iterations(scioxide_core::Budget::ONE),
                )
                .unwrap();
                let result = solver
                    .minimize(&objective, &[<$scalar>::MAX / 4.0; 8])
                    .unwrap();
                assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
                assert!(result.x().iter().all(|value| value.is_finite()));
            }};
        }
        check_centroid!(f64);
        check_centroid!(f32);
    }

    struct Quadratic;

    impl ObjectiveFunction<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
    }

    struct Rosenbrock;

    impl ObjectiveFunction<f64> for Rosenbrock {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            Ok(a * a + 100.0 * b * b)
        }
    }

    struct NonFiniteObjective;

    impl ObjectiveFunction<f64> for NonFiniteObjective {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(f64::NAN)
        }
    }

    #[test]
    fn test_nelder_mead_quadratic_converges() {
        let solver = NelderMead::<f64>::default();
        let result = solver.minimize(&Quadratic, &[5.0, 3.0]).unwrap();
        // Nelder-Mead converges slowly; relax thresholds
        assert!(result.is_success());
        assert!(result.fun < 1e-6);
        assert!(result.x[0].abs() < 1e-3);
        assert!(result.x[1].abs() < 1e-3);
    }

    #[test]
    fn test_nelder_mead_no_gradient_in_result() {
        let solver = NelderMead::<f64>::default();
        let result = solver.minimize(&Quadratic, &[1.0, 1.0]).unwrap();
        assert!(result.grad.is_none());
        assert_eq!(result.grad_evals, 0);
    }

    #[test]
    fn test_nelder_mead_algorithm_identity() {
        let result = NelderMead::<f64>::default()
            .minimize(&Quadratic, &[1.0, 1.0])
            .unwrap();
        assert_eq!(result.algorithm().unwrap().method(), "NelderMead");
    }

    #[test]
    fn test_nelder_mead_dimension_mismatch() {
        let solver = NelderMead::<f64>::default();
        let result = solver.minimize(&Quadratic, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_nelder_mead_decreases_function() {
        let solver = NelderMead::<f64>::default();
        let x0 = [10.0, 10.0];
        let f0 = Quadratic.eval(&x0).unwrap();
        let result = solver.minimize(&Quadratic, &x0).unwrap();
        assert!(result.fun < f0);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let solver = NelderMead::<f64>::default();
        let result = solver.minimize(&Rosenbrock, &[0.0, 0.0]).unwrap();
        // Nelder-Mead may need more iterations for Rosenbrock; just check it improves
        let f0 = Rosenbrock.eval(&[0.0, 0.0]).unwrap();
        assert!(result.fun < f0);
    }

    #[test]
    fn test_nelder_mead_at_origin() {
        let solver = NelderMead::<f64>::default();
        // Starting at origin tests the zero-perturbation path
        let result = solver.minimize(&Quadratic, &[0.0, 0.0]).unwrap();
        assert!(result.fun < 1e-8);
    }

    #[test]
    fn collapsed_values_do_not_certify_a_noncollapsed_simplex() {
        struct Constant;

        impl ObjectiveFunction<f64> for Constant {
            fn dim(&self) -> usize {
                2
            }

            fn eval(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(1.0)
            }
        }

        let solver = NelderMead::new(
            NelderMeadOptions::default().with_max_iterations(scioxide_core::Budget::ONE),
        )
        .unwrap();
        let result = solver.minimize(&Constant, &[1.0, 1.0]).unwrap();

        assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
        assert!(!result.is_success());
        assert_eq!(result.iterations(), 1);
    }

    #[test]
    fn test_nelder_mead_rejects_non_finite_objective() {
        let solver = NelderMead::<f64>::default();
        let result = solver.minimize(&NonFiniteObjective, &[1.0, 1.0]);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }
    #[test]
    fn incremental_geometry_reconstructs_changed_edges_and_confirms_rank_loss() {
        let mut work = NelderMeadStorage::<f64>::new();
        work.try_reserve(4).unwrap();
        work.thresholds = [Scaled::from_value(1e-12).unwrap(); 2];
        work.values.copy_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        for row in 0..5 {
            for column in 0..4 {
                work.simplex[row * 4 + column] = if row == column + 1 { 1.0 } else { 0.0 };
            }
        }
        let options = NelderMeadOptions::default();
        inspect_geometry(&mut work, &options, true).unwrap();
        for step in 0..70 {
            let column = step % 4;
            work.simplex[(column + 1) * 4 + column] *= 0.98;
            work.simplex[(column + 1) * 4 + (column + 1) % 4] += 0.001;
            inspect_geometry(&mut work, &options, false).unwrap();
            assert_eq!(work.diagnostics.geometry, NelderMeadGeometry::FullRank);
            let q = work.incremental.q_matrix();
            let r = work.incremental.r_matrix();
            for i in 0..4 {
                for j in 0..4 {
                    let value: f64 = (0..4)
                        .map(|k| q.as_slice()[i * 4 + k] * r.as_slice()[k * 4 + j])
                        .sum();
                    let original = work.incremental.permutation()[j];
                    assert!((value - work.edges.as_slice()[i * 4 + original]).abs() < 1e-12);
                }
            }
        }
        work.simplex[4..8].fill(0.0);
        inspect_geometry(&mut work, &options, false).unwrap();
        assert_eq!(work.diagnostics.geometry, NelderMeadGeometry::Degenerate);
        assert_eq!(work.diagnostics.rank.unwrap().rank(), 3);
    }
}
