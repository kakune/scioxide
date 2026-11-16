//! Rigid motions and the best-fit rigid motion between matched points.
use crate::math3::{add, cross, dot, mul, n, product, scale, sub, transpose};
use crate::validation::{finite, input, out_of_range as invalid};
use crate::{
    PointSet, Rotation, SpatialExecutionContext, TransformOptions, TransformWorkspace,
    TransformWorkspaceRequest,
};
use alloc::vec::Vec;
use scioxide_core::{
    Argument, ExecutionContext, ExecutionPolicy, RankThreshold, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveScope, Tolerance,
};
use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace, Matrix, SvdDecomposition};

/// A rigid motion `p ↦ R p + t` of three-dimensional space, with rotation `R`
/// and translation `t`.
///
/// Like [`Rotation`], the map is active and preserves orientation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform<S: RealScalar> {
    rotation: Rotation<S>,
    translation: [S; 3],
}
impl<S: RealScalar> RigidTransform<S> {
    /// Returns the identity map.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// let transform = RigidTransform::<f64>::identity();
    /// assert_eq!(transform.translation(), [0.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn identity() -> Self {
        Self {
            rotation: Rotation::identity(),
            translation: [S::ZERO; 3],
        }
    }
    /// Creates the map `p ↦ R p + t` from `R = rotation` and `t = translation`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `translation` is NaN or infinite.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// let transform = RigidTransform::new(Rotation::<f64>::identity(), [1.0, 2.0, 3.0])?;
    /// assert_eq!(transform.translation(), [1.0, 2.0, 3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(rotation: Rotation<S>, translation: [S; 3]) -> Result<Self, SciOxideError> {
        Ok(Self {
            rotation,
            translation: input(translation)?,
        })
    }
    /// Returns the rotation `R`.
    pub const fn rotation(self) -> Rotation<S> {
        self.rotation
    }
    /// Returns the translation `t`.
    pub const fn translation(self) -> [S; 3] {
        self.translation
    }
    /// Returns `R p + t` for a point `p`.
    ///
    /// Each component includes the translation before rounding the exact
    /// sum of products, so intermediate overflow does not reject a finite
    /// result. Evaluation uses constant storage without allocation.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `point` is NaN or infinite.
    /// - `NumericalInstability` if the result is not representable.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// assert_eq!(
    ///     RigidTransform::<f64>::identity().apply([1.0, 2.0, 3.0])?,
    ///     [1.0, 2.0, 3.0]
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply(self, point: [S; 3]) -> Result<[S; 3], SciOxideError> {
        crate::rotation::apply_affine(&self.rotation.as_matrix(), input(point)?, self.translation)
    }
    /// Returns `R v` for a direction `v`, without the translation.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `direction` is NaN or infinite.
    /// - `NumericalInstability` if the result is not representable.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// let transform = RigidTransform::new(Rotation::<f64>::identity(), [1.0; 3])?;
    /// assert_eq!(transform.apply_direction([2.0; 3])?, [2.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply_direction(self, direction: [S; 3]) -> Result<[S; 3], SciOxideError> {
        self.rotation.apply(direction)
    }
    /// Returns the map that applies `other` first, then `self`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the translation is not representable.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// let transform = RigidTransform::new(Rotation::<f64>::identity(), [1.0; 3])?;
    /// assert_eq!(transform.compose(transform)?.translation(), [2.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compose(self, other: Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.rotation.compose(other.rotation)?,
            self.apply(other.translation)?,
        )
    }
    /// Returns the inverse map `p ↦ Rᵀ (p − t)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the translation is not representable.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// let transform = RigidTransform::new(Rotation::<f64>::identity(), [1.0; 3])?;
    /// assert_eq!(transform.inverse()?.translation(), [-1.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn inverse(self) -> Result<Self, SciOxideError> {
        // (R, t)^-1 = (R^T, -R^T t).
        let r = self.rotation.inverse();
        Self::new(r, r.apply(self.translation.map(|x| -x))?)
    }
    /// Returns the 4×4 homogeneous matrix, row-major, acting on column vectors
    /// `[x, y, z, 1]`.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// assert_eq!(
    ///     RigidTransform::<f64>::identity().as_matrix()[3],
    ///     [0.0, 0.0, 0.0, 1.0]
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_matrix(self) -> [[S; 4]; 4] {
        let r = self.rotation.as_matrix();
        core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                if i == 3 {
                    if j == 3 { S::ONE } else { S::ZERO }
                } else if j == 3 {
                    self.translation[i]
                } else {
                    r[i][j]
                }
            })
        })
    }
    /// Creates a rigid motion from a 4×4 homogeneous matrix, row-major.
    ///
    /// The last row must be `[0, 0, 0, 1]` and the upper-left 3×3 block must be
    /// a rotation within `tolerance`, as in [`Rotation::from_matrix`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the last row is wrong or the 3×3 block is not a
    ///   rotation, for example because it scales, shears or reflects.
    /// - `NonFinite` if an entry is NaN or infinite.
    ///
    /// ```
    /// use scioxide_spatial::{RigidTransform, Rotation};
    /// let matrix = RigidTransform::<f64>::identity().as_matrix();
    /// let transform = RigidTransform::from_matrix(matrix, scioxide_core::Tolerance::from_sqrt_epsilon())?;
    /// assert_eq!(transform.translation(), [0.0; 3]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_matrix(m: [[S; 4]; 4], tolerance: Tolerance<S>) -> Result<Self, SciOxideError> {
        for row in m {
            input(row)?;
        }
        for (i, &x) in m[3].iter().enumerate() {
            if !tolerance
                .is_residual_ok((x - if i == 3 { S::ONE } else { S::ZERO }).abs(), S::ONE)?
            {
                return Err(invalid(
                    scioxide_core::argument!("m"),
                    "invalid homogeneous matrix row",
                ));
            }
        }
        let r = Rotation::from_matrix(
            core::array::from_fn(|i| core::array::from_fn(|j| m[i][j])),
            tolerance,
        )?;
        Self::new(r, [m[0][3], m[1][3], m[2][3]])
    }
    /// Applies the map to every point of a three-dimensional point set and
    /// writes the results to `output`.
    ///
    /// `output` must have one entry per point, and `workspace` must meet a
    /// [`TransformWorkspaceRequest::apply`] request for that count. See the
    /// [crate documentation](crate#execution) for parallel execution and
    /// cancellation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the points are not three-dimensional, `output`
    ///   has the wrong length, or `workspace` is too small.
    /// - `ResourceLimit` if the workspace exceeds its byte limit.
    /// - `NumericalInstability` if a result is not representable.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, };
    /// use scioxide_spatial::{
    ///     PointSet, RigidTransform, TransformOptions, TransformWorkspace, TransformWorkspaceRequest,
    /// };
    /// let points = PointSet::from_points(&[[1.0_f64, 2.0, 3.0]])?;
    /// let options = TransformOptions::new();
    /// let mut workspace = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(1, options))?;
    /// let mut out = [[0.0; 3]];
    /// RigidTransform::identity().apply_batch_into(
    ///     &points,
    ///     options,
    ///     &mut out,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert_eq!(out, [[1.0, 2.0, 3.0]]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply_batch_into<'scope, 'capability>(
        self,
        points: &PointSet<'_, S>,
        options: TransformOptions,
        output: &mut [[S; 3]],
        workspace: &mut TransformWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        apply_batch(
            self.rotation,
            self.translation,
            points,
            options,
            output,
            workspace,
            context.into().execution(),
        )
    }
}
impl<S: RealScalar> Rotation<S> {
    /// Rotates every point of a three-dimensional point set and writes the
    /// results to `output`.
    ///
    /// `output` must have one entry per point, and `workspace` must meet a
    /// [`TransformWorkspaceRequest::apply`] request for that count. See the
    /// [crate documentation](crate#execution) for parallel execution and
    /// cancellation.
    ///
    /// # Errors
    ///
    /// As [`RigidTransform::apply_batch_into`].
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, };
    /// use scioxide_spatial::{
    ///     PointSet, Rotation, TransformOptions, TransformWorkspace, TransformWorkspaceRequest,
    /// };
    /// let points = PointSet::from_points(&[[1.0_f64, 2.0, 3.0]])?;
    /// let options = TransformOptions::new();
    /// let mut workspace = TransformWorkspace::workspace(TransformWorkspaceRequest::apply(1, options))?;
    /// let mut out = [[0.0; 3]];
    /// Rotation::identity().apply_batch_into(
    ///     &points,
    ///     options,
    ///     &mut out,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert_eq!(out, [[1.0, 2.0, 3.0]]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply_batch_into<'scope, 'capability>(
        self,
        points: &PointSet<'_, S>,
        options: TransformOptions,
        output: &mut [[S; 3]],
        workspace: &mut TransformWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        apply_batch(
            self,
            [S::ZERO; 3],
            points,
            options,
            output,
            workspace,
            context.into().execution(),
        )
    }
}

fn apply_batch<S: RealScalar>(
    rotation: Rotation<S>,
    translation: [S; 3],
    points: &PointSet<'_, S>,
    options: TransformOptions,
    output: &mut [[S; 3]],
    workspace: &mut TransformWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if points.dimension() != 3 {
        return Err(SciOxideError::dimension_mismatch([3], [points.dimension()]));
    }
    if output.len() != points.len() {
        return Err(SciOxideError::dimension_mismatch(
            [points.len()],
            [output.len()],
        ));
    }
    workspace.require(TransformWorkspaceRequest::apply(points.len(), options))?;
    let matrix = rotation.as_matrix();
    let staging = &mut workspace.points[..points.len()];
    let policy = options.execution_policy();
    if points.contiguous() {
        apply_rows::<S, true>(&matrix, translation, points, staging, policy, context)?;
    } else {
        apply_rows::<S, false>(&matrix, translation, points, staging, policy, context)?;
    }
    context.check()?;
    output.copy_from_slice(staging);
    Ok(())
}

fn apply_rows<S: RealScalar, const CONTIGUOUS: bool>(
    matrix: &[[S; 3]; 3],
    translation: [S; 3],
    points: &PointSet<'_, S>,
    staging: &mut [[S; 3]],
    policy: ExecutionPolicy,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    // Auto estimate: a 3x3 product plus translation per point.
    crate::batch::dispatch_chunks(
        staging,
        policy,
        context,
        crate::batch::work(18),
        |offset, values| {
            for (block, values) in values.chunks_mut(256).enumerate() {
                context.check()?;
                for (index, value) in values.iter_mut().enumerate() {
                    let p = row::<S, CONTIGUOUS>(points, offset + block * 256 + index);
                    *value = crate::rotation::apply_affine(matrix, p, translation)?;
                }
            }
            Ok(())
        },
    )
}

// Inline both row adapters so the point loop keeps direct coordinate loads.
#[inline(always)]
fn row<S: RealScalar, const CONTIGUOUS: bool>(points: &PointSet<'_, S>, i: usize) -> [S; 3] {
    let row = points.row::<CONTIGUOUS>(i);
    [row.value(0), row.value(1), row.value(2)]
}
/// Options for [`estimate_rigid_transform_scoped`]: the rank threshold and the SVD
/// settings.
///
/// ```
/// use scioxide_core::{ExecutionPolicy, RankThreshold};
/// use scioxide_linalg::LinalgExecutionOptions;
/// use scioxide_spatial::RigidEstimationOptions;
/// let options = RigidEstimationOptions::<f64>::new()
///     .with_rank_threshold(RankThreshold::default())
///     .with_svd(LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential));
/// assert_eq!(
///     options.svd().execution_policy(),
///     ExecutionPolicy::Sequential
/// );
/// let _ = options.rank_threshold();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RigidEstimationOptions<S: RealScalar> {
    /// Threshold applied to the singular values of the scaled covariance.
    rank_threshold: RankThreshold<S>,
    /// SVD iteration and execution options.
    svd: LinalgExecutionOptions,
}
impl<S: RealScalar> Default for RigidEstimationOptions<S> {
    fn default() -> Self {
        Self {
            rank_threshold: RankThreshold::default(),
            svd: LinalgExecutionOptions::new(),
        }
    }
}
impl<S: RealScalar> RigidEstimationOptions<S> {
    /// Returns the default rank threshold and SVD options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the threshold below which a singular value of the covariance
    /// matrix counts as zero.
    pub fn with_rank_threshold(mut self, threshold: RankThreshold<S>) -> Self {
        self.rank_threshold = threshold;
        self
    }
    /// Sets the SVD iteration, execution and storage options.
    pub const fn with_svd(mut self, svd: LinalgExecutionOptions) -> Self {
        self.svd = svd;
        self
    }
    /// Returns the rank threshold.
    pub const fn rank_threshold(self) -> RankThreshold<S> {
        self.rank_threshold
    }
    /// Returns the SVD options.
    pub const fn svd(self) -> LinalgExecutionOptions {
        self.svd
    }
}
/// Result of [`estimate_rigid_transform_scoped`]: the fitted map and fit diagnostics.
#[derive(Debug, Clone)]
pub struct RigidEstimationResult<S: RealScalar> {
    /// The rigid motion that maps the source points onto the target points.
    pub transform: RigidTransform<S>,
    /// Weighted root-mean-square distance `(Σᵢ wᵢ |R pᵢ + t − qᵢ|² / Σᵢ wᵢ)^(1/2)`
    /// between mapped source points `pᵢ` and target points `qᵢ`, in input
    /// units.
    pub rms_residual: S,
    /// Singular values of the cross-covariance matrix, scaled so that its
    /// largest entry has magnitude one.
    pub singular_values: [S; 3],
    /// Numerical rank of the cross-covariance matrix: 3, or 2 for coplanar
    /// points.
    pub rank: usize,
}
/// Computes the rigid motion that best maps matched source points onto target
/// points in the weighted least-squares sense.
///
/// The result minimizes `Σᵢ wᵢ |R pᵢ + t − qᵢ|²` over rotations `R` and
/// translations `t`, where `pᵢ` and `qᵢ` are row `i` of `source` and
/// `target` and `wᵢ` the weights (all one when `weights` is `None`). The
/// method is the Kabsch algorithm: center both sets at their weighted
/// centroids, take the SVD of the 3×3 cross-covariance matrix, and correct the
/// sign so that `R` is a rotation, not a reflection. The cost is O(n) for `n`
/// points. Points with zero weight do not affect the fit.
///
/// With a solve context, the fit and its SVD iterations count toward the
/// enclosing solve, as described in
/// [Solve accounting](scioxide_core#solve-accounting). Cancellation and deadlines
/// are checked every 256 rows.
///
/// # Errors
///
/// - `DimensionMismatch` if the point sets are not three-dimensional, differ
///   in length, or `weights` has the wrong length.
/// - `InvalidInput` if there are fewer than three points, a weight is
///   negative or all weights are zero, or the points are collinear, so that
///   the rotation is not unique.
/// - `NumericalInstability` if an intermediate result is not representable.
/// - Errors of the SVD, and `ExecutionStopped` on cancellation or deadline.
///
/// ```
/// use scioxide_core::{ExecutionContext, };
/// use scioxide_spatial::{PointSet, RigidEstimationOptions, estimate_rigid_transform_scoped};
/// let data = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let points = PointSet::from_rows(&data, 3)?;
/// let fit = estimate_rigid_transform_scoped(
///     &points,
///     &points,
///     None,
///     RigidEstimationOptions::new(),
///     ExecutionContext::sequential(),
/// )?;
/// assert!(fit.rms_residual < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. Kabsch, "A solution for the best rotation to relate two sets of
/// vectors", *Acta Cryst.* A32, 1976, 922–923.
pub fn estimate_rigid_transform_scoped<'scope, 'capability, S: RealScalar>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
    weights: Option<&[S]>,
    options: RigidEstimationOptions<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<RigidEstimationResult<S>, SciOxideError> {
    const SOURCE: Argument = scioxide_core::argument!("source");
    for dimension in [source.dimension(), target.dimension()] {
        if dimension != 3 {
            return Err(SciOxideError::dimension_mismatch([3], [dimension]));
        }
    }
    if source.len() != target.len() {
        return Err(SciOxideError::dimension_mismatch(
            [source.len()],
            [target.len()],
        ));
    }
    if let Some(weights) = weights
        && weights.len() != source.len()
    {
        return Err(SciOxideError::dimension_mismatch(
            [source.len()],
            [weights.len()],
        ));
    }
    if source.len() < 3 {
        return Err(invalid(
            SOURCE,
            "alignment needs at least three matched points",
        ));
    }
    crate::context::scoped(
        scope,
        scioxide_core::algorithm_id!("spatial", "rigid-estimation", 1),
        SolveBudgets::unbounded(),
        |context| {
            // A fit is one local attempt; SVD sweeps have their own local allowance.
            if !context.record_iteration()?.is_granted() {
                return Err(SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::IterationLimit,
                    scioxide_core::ConvergenceDiagnostics::new(),
                    "iteration budget exhausted before fit; no estimate is available",
                ));
            }
            match (source.contiguous(), target.contiguous(), weights) {
                (true, true, Some(weights)) => {
                    estimate_rows::<S, true, true, true>(source, target, weights, options, context)
                }
                (true, true, None) => {
                    estimate_rows::<S, true, true, false>(source, target, &[], options, context)
                }
                (true, false, Some(weights)) => {
                    estimate_rows::<S, true, false, true>(source, target, weights, options, context)
                }
                (true, false, None) => {
                    estimate_rows::<S, true, false, false>(source, target, &[], options, context)
                }
                (false, true, Some(weights)) => {
                    estimate_rows::<S, false, true, true>(source, target, weights, options, context)
                }
                (false, true, None) => {
                    estimate_rows::<S, false, true, false>(source, target, &[], options, context)
                }
                (false, false, Some(weights)) => estimate_rows::<S, false, false, true>(
                    source, target, weights, options, context,
                ),
                (false, false, None) => {
                    estimate_rows::<S, false, false, false>(source, target, &[], options, context)
                }
            }
        },
    )
}

fn estimate_rows<S: RealScalar, const LEFT: bool, const RIGHT: bool, const WEIGHTED: bool>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
    weights: &[S],
    options: RigidEstimationOptions<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<RigidEstimationResult<S>, SciOxideError> {
    let execution = context.execution();
    const SOURCE: Argument = scioxide_core::argument!("source");
    const WEIGHTS: Argument = scioxide_core::argument!("weights");
    // Weights and coordinates are each divided by their maximum, so the
    // weight total and the covariance accumulate O(1) quantities. The result
    // is then independent of the input units until the final rescale of the
    // translation and residual. Zero-weight points are ignored everywhere,
    // including in the coordinate scale.
    let mut weight_scale = S::ONE;
    if WEIGHTED {
        weight_scale = S::ZERO;
        for (i, &w) in weights.iter().enumerate() {
            if i % 256 == 0 {
                execution.check()?;
            }
            input([w])?;
            if w < S::ZERO {
                return Err(invalid(WEIGHTS, "negative alignment weight"));
            }
            weight_scale = weight_scale.max(w);
        }
        if weight_scale == S::ZERO {
            return Err(invalid(WEIGHTS, "zero total alignment weight"));
        }
    }
    let weight = |i: usize| {
        if WEIGHTED {
            weights[i] / weight_scale
        } else {
            S::ONE
        }
    };
    let mut coordinate_scale = S::ZERO;
    let mut total = S::ZERO;
    for i in 0..source.len() {
        if i % 256 == 0 {
            execution.check()?;
        }
        let w = weight(i);
        total += w;
        if w > S::ZERO {
            for x in row::<S, LEFT>(source, i)
                .into_iter()
                .chain(row::<S, RIGHT>(target, i))
            {
                coordinate_scale = coordinate_scale.max(x.abs());
            }
        }
    }
    if coordinate_scale == S::ZERO {
        return Err(invalid(SOURCE, "coincident alignment points"));
    }
    finite([total])?;
    let normalized_weight = |i| weight(i) / total;
    let left_point = |i| row::<S, LEFT>(source, i).map(|x| x / coordinate_scale);
    let right_point = |i| row::<S, RIGHT>(target, i).map(|x| x / coordinate_scale);
    let mut left = [S::ZERO; 3];
    let mut right = left;
    for i in 0..source.len() {
        if i % 256 == 0 {
            execution.check()?;
        }
        let w = normalized_weight(i);
        if w > S::ZERO {
            left = add(left, scale(left_point(i), w));
            right = add(right, scale(right_point(i), w));
        }
    }
    // Kabsch cross-covariance H = sum_i w_i (a_i - a_bar)(b_i - b_bar)^T,
    // accumulated from explicitly centered points. The one-pass
    // sum(w a b^T) - a_bar b_bar^T form would cancel catastrophically
    // for clouds far from the origin.
    let mut h = [[S::ZERO; 3]; 3];
    for i in 0..source.len() {
        if i % 256 == 0 {
            execution.check()?;
        }
        let w = normalized_weight(i);
        if w == S::ZERO {
            continue;
        }
        let a = sub(left_point(i), left);
        let b = sub(right_point(i), right);
        for (j, row) in h.iter_mut().enumerate() {
            for (k, x) in row.iter_mut().enumerate() {
                *x += w * a[j] * b[k];
            }
        }
    }
    // Rank policy must not depend on translation magnitude. Normalize the
    // centered covariance as well as the original coordinate arithmetic.
    let covariance_scale = h
        .iter()
        .flatten()
        .fold(S::ZERO, |largest, &x| largest.max(x.abs()));
    if covariance_scale == S::ZERO {
        return Err(invalid(SOURCE, "alignment covariance is zero"));
    }
    let mut matrix = Matrix::zeros(3, 3)?;
    for (out, &x) in matrix.as_mut_slice().iter_mut().zip(h.iter().flatten()) {
        *out = x / covariance_scale;
    }
    let mut svd = SvdDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    scioxide_linalg::SvdDecomposition::compute_into(
        &matrix,
        &options.svd,
        &mut svd,
        &mut workspace,
        &mut *context,
    )?;
    let values = svd.singular_values();
    let singular_values = [values[0], values[1], values[2]];
    let threshold = options.rank_threshold.threshold(values[0])?;
    // Rank 2 (planar or otherwise degenerate but non-collinear data) still
    // fixes a unique proper rotation, because the third axis follows from
    // the determinant constraint. Collinear data leaves a free spin.
    let rank = values.iter().filter(|&&x| x > threshold).count();
    if rank < 2 {
        return Err(invalid(
            SOURCE,
            "alignment rotation is not uniquely determined",
        ));
    }
    let u = svd.u_matrix();
    let vt = svd.vt_matrix();
    // R = V U^T, read from row-major U and V^T storage.
    let u = core::array::from_fn(|i| core::array::from_fn(|j| u.as_slice()[i * 3 + j]));
    let v = core::array::from_fn(|i| core::array::from_fn(|j| vt.as_slice()[j * 3 + i]));
    let mut r = product(v, transpose(u));
    // A reflection is the unconstrained optimum. The best proper rotation is
    // V diag(1, 1, -1) U^T = V U^T - 2 v3 u3^T. That flip is unique only if
    // the smallest singular value is separated from the middle one;
    // otherwise several proper rotations fit equally well.
    if dot(r[0], cross(r[1], r[2])) < S::ZERO {
        if values[1] - values[2] <= threshold {
            return Err(invalid(SOURCE, "reflected alignment has nonunique optimum"));
        }
        for (i, row) in r.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x -= n::<S>(2) * v[i][2] * u[j][2];
            }
        }
    }
    let rotation = estimated_rotation(r)?;
    let translation_normalized = sub(right, rotation.apply(left)?);
    let transform = RigidTransform::new(
        rotation,
        finite(scale(translation_normalized, coordinate_scale))?,
    )?;
    let matrix = rotation.as_matrix();
    let difference = |i| {
        let predicted = add(mul(matrix, left_point(i)), translation_normalized);
        sub(predicted, right_point(i))
    };
    let mut squared = S::ZERO;
    let mut retry = false;
    for i in 0..source.len() {
        if i % 256 == 0 {
            execution.check()?;
        }
        let w = normalized_weight(i);
        if w == S::ZERO {
            continue;
        }
        for x in difference(i) {
            let term = w * x * x;
            retry |= x != S::ZERO && term < S::MIN_POSITIVE;
            squared += term;
        }
    }
    let residual = if retry || !squared.is_finite() {
        let mut residual = scioxide_core::ScaledNorm::new();
        for i in 0..source.len() {
            if i % 256 == 0 {
                execution.check()?;
            }
            let w = normalized_weight(i);
            if w == S::ZERO {
                continue;
            }
            for x in difference(i) {
                residual.add(x * w.sqrt())?;
            }
        }
        residual.finish()?
    } else {
        squared.sqrt()
    };
    let rms_residual = finite([residual * coordinate_scale])?[0];
    execution.check()?;
    Ok(RigidEstimationResult {
        transform,
        rms_residual,
        singular_values,
        rank,
    })
}

// SVD factors, the 3x3 product and the reflection correction each round in S.
// This allowance checks their orthogonality invariant, not fit accuracy.
const ESTIMATED_ROTATION_EPSILONS: u16 = 128;

fn estimated_rotation<S: RealScalar>(matrix: [[S; 3]; 3]) -> Result<Rotation<S>, SciOxideError> {
    let allowance = n::<S>(ESTIMATED_ROTATION_EPSILONS) * S::EPSILON;
    Rotation::from_matrix(matrix, Tolerance::new(allowance, allowance)?).map_err(|_| {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "estimated rotation failed the proper orthogonality check",
        )
    })
}

impl<S: RealScalar> Rotation<S> {
    /// Transforms every point using default options and sequential native execution.
    /// Allocates `n` output points and a staging workspace; cost and accuracy match [`Self::apply_batch_into`].
    ///
    /// # Errors
    /// As [`Self::apply_batch_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn apply_batch(self, points: &PointSet<'_, S>) -> Result<Vec<[S; 3]>, SciOxideError> {
        let options = TransformOptions::new();
        let mut workspace =
            TransformWorkspace::workspace(TransformWorkspaceRequest::apply(points.len(), options))?;
        let mut output = scioxide_core::backend::try_filled(
            points.len(),
            [S::ZERO; 3],
            scioxide_core::quantity!("transform_output"),
        )?;
        self.apply_batch_into(
            points,
            options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S: RealScalar> RigidTransform<S> {
    /// Transforms every point using default options and sequential native execution.
    /// Allocates `n` output points and a staging workspace; cost and accuracy match [`Self::apply_batch_into`].
    ///
    /// # Errors
    /// As [`Self::apply_batch_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn apply_batch(self, points: &PointSet<'_, S>) -> Result<Vec<[S; 3]>, SciOxideError> {
        let options = TransformOptions::new();
        let mut workspace =
            TransformWorkspace::workspace(TransformWorkspaceRequest::apply(points.len(), options))?;
        let mut output = scioxide_core::backend::try_filled(
            points.len(),
            [S::ZERO; 3],
            scioxide_core::quantity!("transform_output"),
        )?;
        self.apply_batch_into(
            points,
            options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

/// Estimates the least-squares rigid motion with uniform weights and default rank and SVD options.
/// Uses [`estimate_rigid_transform_with_options`] with its storage, cost and accuracy.
///
/// # Errors
/// As [`estimate_rigid_transform_with_options`].
///
/// ```
/// use scioxide_spatial::{PointSet, estimate_rigid_transform};
/// let sites = [[0.0_f64; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let points = PointSet::from_points(&sites)?;
/// let fit = estimate_rigid_transform(&points, &points)?;
/// let moved = fit.transform.apply([1.0, 0.0, 0.0])?;
/// assert!((moved[0] - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn estimate_rigid_transform<S: RealScalar>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
) -> Result<RigidEstimationResult<S>, SciOxideError> {
    estimate_rigid_transform_with_options(source, target, None, RigidEstimationOptions::default())
}
/// Estimates the weighted least-squares rigid motion with sequential native execution.
/// Uses [`estimate_rigid_transform_scoped`] with the same storage, cost and accuracy.
///
/// # Errors
/// As [`estimate_rigid_transform_scoped`].
pub fn estimate_rigid_transform_with_options<S: RealScalar>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
    weights: Option<&[S]>,
    options: RigidEstimationOptions<S>,
) -> Result<RigidEstimationResult<S>, SciOxideError> {
    estimate_rigid_transform_scoped(
        source,
        target,
        weights,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

#[cfg(test)]
mod tests {
    use super::estimated_rotation;
    use scioxide_core::{NumericalIssue, SciOxideErrorView};

    #[test]
    fn failed_estimated_rotation_is_a_numerical_breakdown() {
        for matrix in [
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 2.]],
            [[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
        ] {
            assert!(matches!(
                estimated_rotation(matrix),
                Err(error) if matches!(error.view(), SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Breakdown, ..
                })
            ));
        }
    }
}
