//! BFGS and SR1 secant updates of dense real matrices.
//!
//! The Hessian form updates `B` with the secant pair `(s, y)`; the inverse form
//! updates `H` with the inverse BFGS formula and the dual SR1 formula.

use alloc::vec::Vec;
use scioxide_core::Argument;
use scioxide_core::{ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, LinalgWorkspace, Matrix, MatrixView, MatrixViewMut,
};

/// How a model enters the shared update validation boundary.
pub(crate) enum UpdateSource {
    Caller,
    ScaledIdentity,
    Verified,
    VerifiedLower,
}

/// Selects what a BFGS update does when the curvature condition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureAction {
    /// Skips the update and keeps the previous matrix.
    Skip,
    /// Applies Powell's damping: replaces `y` by a convex combination of `y`
    /// and the model product so that the curvature condition holds.
    Damp,
}

/// Sets the curvature test and the response of a BFGS update.
///
/// In the Hessian form the update accepts the pair `(s, y)` when
/// `sᵀy ≥ θ sᵀB s`, where `θ` is [`Self::threshold`]. The inverse form applies
/// the same test with the roles of `s` and `y` exchanged. The test is
/// dimensionless, so it does not depend on the units of the objective. The
/// default is [`CurvatureAction::Damp`] with `θ = 0.2`.
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §18.3 (damped BFGS updating).
#[derive(Debug, Clone, Copy)]
pub struct CurvaturePolicy<S: RealScalar> {
    action: CurvatureAction,
    threshold: S,
}
impl<S: RealScalar> Default for CurvaturePolicy<S> {
    fn default() -> Self {
        Self {
            action: CurvatureAction::Damp,
            threshold: S::ONE / (S::TWO + S::TWO + S::ONE),
        }
    }
}
impl<S: RealScalar> CurvaturePolicy<S> {
    /// Creates a policy with the given action and threshold `θ ∈ (0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `threshold` is not finite or lies outside
    /// `(0, 1)`.
    pub fn new(action: CurvatureAction, threshold: S) -> Result<Self, SciOxideError> {
        if !threshold.is_finite() || threshold <= S::ZERO || threshold >= S::ONE {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("threshold"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "curvature threshold must lie in (0, 1)",
            ));
        }
        Ok(Self { action, threshold })
    }
    /// Returns the action taken when the curvature test fails.
    pub const fn action(self) -> CurvatureAction {
        self.action
    }
    /// Returns the curvature threshold `θ`.
    pub const fn threshold(self) -> S {
        self.threshold
    }
}

/// Selects whether a matrix approximates the Hessian or its inverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianForm {
    /// Approximate Hessian, satisfying `B s = y`.
    Hessian,
    /// Approximate inverse Hessian, satisfying `H y = s`.
    Inverse,
}

/// Selects the secant update formula.
///
/// BFGS keeps a positive definite matrix positive definite. SR1 does not, and
/// can produce an indefinite matrix.
#[derive(Debug, Clone, Copy)]
pub enum HessianUpdate<S: RealScalar> {
    /// BFGS update with the given curvature policy.
    Bfgs(CurvaturePolicy<S>),
    /// Symmetric rank-one update. It is skipped when the denominator is small
    /// relative to the norms of its factors, with a threshold derived from the
    /// machine epsilon.
    Sr1,
}

/// Gives the reason an update was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateSkipReason {
    /// The curvature test failed.
    Curvature,
    /// The SR1 residual is zero or its denominator is too small relative to the norms.
    SmallDenominator,
    /// The updated matrix is not representable in the scalar type.
    Unrepresentable,
    /// The updated matrix is not positive definite after rounding.
    NotPositiveDefinite,
}

/// Reports the outcome of one update. A skipped update leaves the matrix unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianUpdateReport {
    /// The update used the secant pair as given.
    Updated,
    /// The update used a damped secant pair.
    Damped,
    /// The update was skipped.
    Skipped(UpdateSkipReason),
}

pub(crate) fn scaled_dot<S: RealScalar>(a: &[S], b: &[S]) -> Result<Scaled<S>, SciOxideError> {
    if a.len() != b.len() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(a.len()),
            scioxide_core::Dimensions::one(b.len()),
        ));
    }
    if let Some(value) = ordinary_dot(a, b) {
        return Scaled::from_value(value);
    }
    wide_dot(a, b)
}

// A normal-range certificate for the same Neumaier reduction. Keeping the
// scalar candidate available avoids exponent conversions in common policies.
#[inline]
pub(crate) fn ordinary_dot<S: RealScalar>(a: &[S], b: &[S]) -> Option<S> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = scioxide_core::CompensatedSum::ZERO;
    for (&a, &b) in a.iter().zip(b) {
        let product = a * b;
        if scioxide_core::backend::standard::product_needs_retry(a, b, product)
            || sum.add(product).is_err()
        {
            return None;
        }
    }
    sum.total()
        .ok()
        .filter(|value| value.abs() >= S::MIN_POSITIVE)
}

#[cold]
fn wide_dot<S: RealScalar>(a: &[S], b: &[S]) -> Result<Scaled<S>, SciOxideError> {
    let mut sum = ScaledAccumulator::ZERO;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "non-finite vector",
            ));
        }
        sum.add_product(Scaled::from_value(a)?, Scaled::from_value(b)?)?;
    }
    sum.total()
}

pub(crate) fn positive<S: RealScalar>(value: Scaled<S>) -> bool {
    !value.is_zero() && !value.is_sign_negative()
}

/// Modify `v` so `uᵀv >= threshold * uᵀproduct`, with all comparisons in extended range.
/// The caller stages `v`; a skipped update must not copy it out.
pub(crate) fn damp_pair<S: RealScalar>(
    u: &[S],
    v: &mut [S],
    product: &[S],
    policy: CurvaturePolicy<S>,
) -> Result<HessianUpdateReport, SciOxideError> {
    assess_pair(u, v, product, policy).map(|assessment| assessment.report)
}

struct PairAssessment<S: RealScalar> {
    report: HessianUpdateReport,
    curvature: Scaled<S>,
    model: Scaled<S>,
}

fn assess_pair<S: RealScalar>(
    u: &[S],
    v: &mut [S],
    product: &[S],
    policy: CurvaturePolicy<S>,
) -> Result<PairAssessment<S>, SciOxideError> {
    if let Some((uv, up)) = ordinary_dot(u, v).zip(ordinary_dot(u, product))
        && up > S::ZERO
        && uv >= policy.threshold * up
        && let Some(norms) = ordinary_dot(u, u)
            .zip(ordinary_dot(v, v))
            .and_then(|(a, b)| normal_product(a, b))
    {
        let cutoff = norms.sqrt() * S::EPSILON;
        return Ok(PairAssessment {
            report: if uv > cutoff {
                HessianUpdateReport::Updated
            } else {
                HessianUpdateReport::Skipped(UpdateSkipReason::Curvature)
            },
            curvature: Scaled::from_value(uv)?,
            model: Scaled::from_value(up)?,
        });
    }
    let mut uv = scaled_dot(u, v)?;
    let up = scaled_dot(u, product)?;
    if !positive(up) {
        return Ok(PairAssessment {
            report: HessianUpdateReport::Skipped(UpdateSkipReason::Curvature),
            curvature: uv,
            model: up,
        });
    }
    let bound = up.mul(Scaled::from_value(policy.threshold)?)?;
    let mut report = HessianUpdateReport::Updated;
    if uv.cmp_value(bound) == core::cmp::Ordering::Less {
        if policy.action == CurvatureAction::Skip {
            return Ok(PairAssessment {
                report: HessianUpdateReport::Skipped(UpdateSkipReason::Curvature),
                curvature: uv,
                model: up,
            });
        }
        let theta = up
            .mul(Scaled::from_value(S::ONE - policy.threshold)?)?
            .div(up.sub(uv)?)?;
        let complement = Scaled::from_value(S::ONE)?.sub(theta)?;
        for (v, &p) in v.iter_mut().zip(product) {
            *v = theta
                .mul(Scaled::from_value(*v)?)?
                .add(complement.mul(Scaled::from_value(p)?)?)?
                .round()?;
        }
        report = HessianUpdateReport::Damped;
        uv = scaled_dot(u, v)?;
    }
    // Revalidate the rounded pair using an angular threshold with no unit floor.
    let scale = scaled_dot(u, u)?.mul(scaled_dot(v, v)?)?.sqrt()?;
    let cutoff = scale.mul(Scaled::from_value(S::EPSILON)?)?;
    if !positive(uv) || uv.abs_le(cutoff) {
        return Ok(PairAssessment {
            report: HessianUpdateReport::Skipped(UpdateSkipReason::Curvature),
            curvature: uv,
            model: up,
        });
    }
    Ok(PairAssessment {
        report,
        curvature: uv,
        model: up,
    })
}

// Native multiplication is used only when no intermediate range is lost.
// Otherwise the same rank-update expression is evaluated in extended range.
fn normal_product<S: RealScalar>(a: S, b: S) -> Option<S> {
    let value = a * b;
    (value.is_finite() && (value.abs() >= S::MIN_POSITIVE || a == S::ZERO || b == S::ZERO))
        .then_some(value)
}

fn resolved_sum<S: RealScalar>(a: S, b: S) -> Option<S> {
    let value = a + b;
    let scale = a.abs() + b.abs();
    let two = S::TWO;
    let margin = two * two * two * S::EPSILON * scale;
    (value.is_finite() && scale.is_finite() && (scale == S::ZERO || value.abs() > margin))
        .then_some(value)
}

// A conservative fourth-root frame leaves headroom for three-factor rank
// terms, cross-term addition, and cancellation down to the significand grid.
// Outside it the component-wise range checks below remain authoritative.
fn ordinary_frame<S: RealScalar>(values: &[S]) -> bool {
    let lower = S::MIN_POSITIVE.sqrt().sqrt();
    let upper = S::MAX.sqrt().sqrt();
    values.iter().fold(true, |valid, value| {
        let magnitude = value.abs();
        valid & ((magnitude == S::ZERO) | ((magnitude >= lower) & (magnitude <= upper)))
    })
}

pub(crate) fn trusted_product<S: RealScalar>(
    matrix: &Matrix<S>,
    vector: &[S],
    output: &mut [S],
    scratch: &mut [S],
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.invoke(scioxide_core::EvaluationRole::LinearOperator, || {
        session.execution()?.check()?;
        lower_product(matrix.as_slice(), vector, output, scratch, S::MIN_POSITIVE)?;
        session.execution()?.check()
    })
}

// The dense solver owns the lower triangle. Each stored coefficient is read
// once for the two symmetric contributions; the public updater still owns both
// triangles. Boundary results retry the logical row in compensated wide range.
fn lower_product<S: RealScalar>(
    matrix: &[S],
    vector: &[S],
    output: &mut [S],
    lanes: &mut [S],
    uncertainty: S,
) -> Result<(), SciOxideError> {
    let n = vector.len();
    if n == 0 {
        output.fill(S::ZERO);
        return Ok(());
    }
    if uncertainty == S::MAX {
        for (row, value) in output.iter_mut().enumerate() {
            for (col, coefficient) in lanes[..n].iter_mut().enumerate() {
                *coefficient = matrix[row.max(col) * n + row.min(col)];
            }
            *value = scaled_dot(&lanes[..n], vector)?.round()?;
        }
        return Ok(());
    }
    lanes.fill(S::ZERO);
    for (row, values) in matrix.chunks_exact(n).enumerate() {
        let mut sums = [S::ZERO; 4];
        let x = vector[row];
        let upper = &mut lanes[(row % 4) * n..(row % 4 + 1) * n];
        let mut chunks = values[..row].chunks_exact(4);
        let mut inputs = vector[..row].chunks_exact(4);
        let mut outputs = upper[..row].chunks_exact_mut(4);
        for ((a, b), out) in chunks.by_ref().zip(inputs.by_ref()).zip(outputs.by_ref()) {
            sums[0] = a[0] * b[0] + sums[0];
            out[0] += a[0] * x;
            sums[1] = a[1] * b[1] + sums[1];
            out[1] += a[1] * x;
            sums[2] = a[2] * b[2] + sums[2];
            out[2] += a[2] * x;
            sums[3] = a[3] * b[3] + sums[3];
            out[3] += a[3] * x;
        }
        for (lane, ((&a, &b), out)) in chunks
            .remainder()
            .iter()
            .zip(inputs.remainder())
            .zip(outputs.into_remainder())
            .enumerate()
        {
            sums[lane] = a * b + sums[lane];
            *out += a * x;
        }
        sums[row % 4] = values[row] * x + sums[row % 4];
        for (lane, &sum) in sums.iter().enumerate() {
            lanes[lane * n + row] = sum;
        }
    }
    for (row, out) in output.iter_mut().enumerate() {
        *out = (lanes[row] + lanes[n + row]) + (lanes[2 * n + row] + lanes[3 * n + row]);
    }
    for (row, value) in output.iter_mut().enumerate() {
        if !value.is_finite() || value.abs() <= uncertainty {
            for (col, coefficient) in lanes[..n].iter_mut().enumerate() {
                *coefficient = matrix[row.max(col) * n + row.min(col)];
            }
            *value = scaled_dot(&lanes[..n], vector)?.round()?;
        }
    }
    Ok(())
}

// A positive model satisfies |H_ij| <= max(diag(H)). Guard model/secant
// products at square-root precision before the update amplifies cancellation.
fn model_product_uncertainty<S: RealScalar>(
    matrix: &[S],
    vector: &[S],
) -> Result<S, SciOxideError> {
    let n = vector.len();
    if n == 0 {
        return Ok(S::ZERO);
    }
    // Scalar compensation is inexpensive for cache-resident short models and
    // retains the accuracy needed by stiff merit subproblems. Larger models
    // amortize the vector kernel and retry only unresolved contractions.
    if n <= 32 {
        return Ok(S::MAX);
    }
    let diagonal = (0..n).fold(S::ZERO, |scale, i| scale.max(matrix[i * n + i].abs()));
    let vector_scale = vector
        .iter()
        .fold(S::ZERO, |scale, &value| scale.max(value.abs()));
    let units = S::SQRT_EPSILON * S::try_from_usize_exact(n)?;
    if let Some(bound) =
        normal_product(diagonal, vector_scale).and_then(|p| normal_product(p, units))
    {
        return Ok(bound.max(S::MIN_POSITIVE));
    }
    let bound = Scaled::from_value(diagonal)?
        .mul(Scaled::from_value(vector_scale)?)?
        .mul(Scaled::from_value(units)?)?;
    Ok(bound.round().unwrap_or(S::MAX).max(S::MIN_POSITIVE))
}

/// Applies BFGS and SR1 updates to dense `n × n` matrices.
///
/// The workspace holds O(n²) scratch storage. An update costs O(n²) for SR1.
/// A BFGS update of a matrix you pass in also checks that the result is
/// positive definite, which costs O(n³) when the matrix is not diagonally
/// dominant.
///
/// # References
///
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §6.1–6.2.
#[derive(Debug)]
pub struct HessianUpdateWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    candidate: Matrix<S>,
    lower: bool,
    product: Vec<S>,
    product_lanes: Vec<S>,
    modified: Vec<S>,
    factor: CholeskyDecomposition<S>,
    factor_workspace: LinalgWorkspace<'static, S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for HessianUpdateWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> HessianUpdateWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            candidate: Matrix::empty(),
            lower: false,
            product: Vec::new(),
            product_lanes: Vec::new(),
            modified: Vec::new(),
            factor: CholeskyDecomposition::empty(),
            factor_workspace: LinalgWorkspace::new(),
        }
    }
    /// Reserves storage for `dimension × dimension` matrices.
    ///
    /// On error the workspace is unchanged.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        if self.product.len() == dimension {
            return Ok(());
        }
        let mut next = Self::new();
        next.candidate.try_resize_flat(dimension, dimension)?;
        for buffer in [&mut next.product, &mut next.modified] {
            scioxide_core::backend::try_reserve(
                buffer,
                dimension,
                scioxide_core::quantity!("workspace.update_vectors"),
            )?;
            buffer.resize(dimension, <S as scioxide_core::AlgebraicScalar>::ZERO);
        }
        for i in 0..dimension {
            next.candidate.as_mut_slice()[i * dimension + i] =
                <S as scioxide_core::AlgebraicScalar>::ONE;
        }
        let lanes = scioxide_core::backend::checked_mul(
            4,
            dimension,
            scioxide_core::quantity!("workspace.product_lanes"),
        )?;
        scioxide_core::backend::try_reserve(
            &mut next.product_lanes,
            lanes,
            scioxide_core::quantity!("workspace.product_lanes"),
        )?;
        next.product_lanes.resize(lanes, S::ZERO);
        next.factor.try_reserve(dimension)?;
        next.factor_workspace
            .try_reserve(&CholeskyDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::CholeskyWorkspaceRequest::factor(
                    dimension,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )?)?;
        *self = next;
        Ok(())
    }
    fn validate_pd(
        &mut self,
        execution: ExecutionContext<'_>,
        observer: Option<&dyn crate::context::Observation>,
    ) -> Result<bool, SciOxideError> {
        if self.candidate.view().is_positive_diagonally_dominant()? {
            return Ok(true);
        }
        let result = crate::context::invoke(
            observer,
            scioxide_core::EvaluationRole::Factorization,
            || {
                CholeskyDecomposition::factor_into(
                    &self.candidate,
                    scioxide_linalg::HermitianInput::Exact,
                    CholeskyPivoting::None,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    &mut self.factor,
                    &mut self.factor_workspace,
                    execution,
                )
            },
        );
        crate::dense::positive_definite(result)
    }
    /// Updates `matrix` in place with the secant pair `(s, y)`.
    ///
    /// `matrix` is unchanged when the update is skipped or fails.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` or `InvalidInput` if the shapes of `matrix`, `s`,
    ///   `y` and the workspace do not agree.
    /// - `NonFinite` if `matrix`, `s` or `y` contains NaN or infinity.
    pub fn update(
        &mut self,
        matrix: &mut Matrix<S>,
        s: &[S],
        y: &[S],
        form: HessianForm,
        rule: HessianUpdate<S>,
    ) -> Result<HessianUpdateReport, SciOxideError> {
        self.update_observed(
            matrix,
            s,
            y,
            form,
            rule,
            ExecutionContext::sequential(),
            None,
            scioxide_core::argument!("matrix"),
        )
    }
    /// Writes the update of `source` with the secant pair `(s, y)` to `output`.
    ///
    /// `output` is unchanged when the update is skipped or fails.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output` and `source` differ in shape.
    /// - `DimensionMismatch` or `InvalidInput` if `s`, `y` or the workspace do
    ///   not match `source`.
    /// - `NonFinite` if `source`, `s` or `y` contains NaN or infinity.
    /// - `ExecutionStopped` if `execution` is cancelled or reaches its deadline.
    #[allow(clippy::too_many_arguments)]
    pub fn update_into(
        &mut self,
        source: MatrixView<'_, S>,
        mut output: MatrixViewMut<'_, S>,
        s: &[S],
        y: &[S],
        form: HessianForm,
        rule: HessianUpdate<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<HessianUpdateReport, SciOxideError> {
        if output.rows() != source.rows() || output.cols() != source.cols() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "output shape differs from source",
            ));
        }
        let report = self.prepare(
            source,
            s,
            y,
            form,
            rule,
            execution,
            None,
            UpdateSource::Caller,
            scioxide_core::argument!("source"),
        )?;
        if !matches!(report, HessianUpdateReport::Skipped(_)) {
            for row in 0..output.rows() {
                for col in 0..output.cols() {
                    output.set(
                        row,
                        col,
                        self.candidate.as_slice()[row * output.cols() + col],
                    )?;
                }
            }
        }
        Ok(report)
    }
    pub(crate) fn commit(&mut self, matrix: &mut Matrix<S>, report: HessianUpdateReport) {
        if !matches!(report, HessianUpdateReport::Skipped(_)) {
            if self.lower {
                core::mem::swap(matrix, &mut self.candidate);
            } else {
                matrix
                    .as_mut_slice()
                    .copy_from_slice(self.candidate.as_slice());
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_observed(
        &mut self,
        matrix: &mut Matrix<S>,
        s: &[S],
        y: &[S],
        form: HessianForm,
        rule: HessianUpdate<S>,
        execution: ExecutionContext<'_>,
        observer: Option<&dyn crate::context::Observation>,
        argument: Argument,
    ) -> Result<HessianUpdateReport, SciOxideError> {
        let report = self.prepare(
            matrix.view(),
            s,
            y,
            form,
            rule,
            execution,
            observer,
            UpdateSource::Caller,
            argument,
        )?;
        if !matches!(report, HessianUpdateReport::Skipped(_)) {
            matrix
                .as_mut_slice()
                .copy_from_slice(self.candidate.as_slice());
        }
        Ok(report)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        source: MatrixView<'_, S>,
        s: &[S],
        y: &[S],
        form: HessianForm,
        rule: HessianUpdate<S>,
        execution: ExecutionContext<'_>,
        observer: Option<&dyn crate::context::Observation>,
        source_state: UpdateSource,
        argument: Argument,
    ) -> Result<HessianUpdateReport, SciOxideError> {
        let n = self.product.len();
        if source.rows() != n || source.cols() != n || s.len() != n || y.len() != n {
            return Err(SciOxideError::invalid_input(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "square matrix, vectors, and prepared workspace must have matching dimensions",
            ));
        }
        execution.check()?;
        self.lower = matches!(
            source_state,
            UpdateSource::ScaledIdentity | UpdateSource::VerifiedLower
        );
        crate::utils::validate_finite_origin(s, scioxide_core::NonFiniteOrigin::Input)?;
        crate::utils::validate_finite_origin(y, scioxide_core::NonFiniteOrigin::Input)?;
        let scale = if matches!(source_state, UpdateSource::ScaledIdentity)
            && matches!(rule, HessianUpdate::Bfgs(_))
        {
            let sy = scaled_dot(s, y)?;
            let yy = scaled_dot(y, y)?;
            if positive(sy) && positive(yy) {
                Some(sy.div(yy)?)
            } else {
                None
            }
        } else {
            None
        };
        let scale_value = scale.map(Scaled::round).transpose()?;
        if self.lower {
            if source.as_contiguous_slice().is_none() {
                return Err(SciOxideError::invalid_input(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "solver matrix must be contiguous",
                ));
            }
        } else if matches!(source_state, UpdateSource::Verified)
            && let Some(values) = source.as_contiguous_slice()
        {
            self.candidate.as_mut_slice().copy_from_slice(values);
        } else {
            for row in 0..n {
                for col in 0..n {
                    let value = source.get(row, col)?;
                    if !value.is_finite() {
                        return Err(SciOxideError::non_finite(
                            scioxide_core::NonFiniteOrigin::Input,
                            "non-finite matrix",
                        ));
                    }
                    if value != source.get(col, row)? {
                        return Err(SciOxideError::invalid_input(
                            argument,
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "matrix must be exactly symmetric",
                        ));
                    }
                    self.candidate.as_mut_slice()[row * n + col] = match scale {
                        Some(scale) => scale.mul(Scaled::from_value(value)?)?.round()?,
                        None => value,
                    };
                }
            }
        }
        if matches!(source_state, UpdateSource::Caller)
            && matches!(rule, HessianUpdate::Bfgs(_))
            && !self.validate_pd(execution, observer)?
        {
            return Ok(HessianUpdateReport::Skipped(
                UpdateSkipReason::NotPositiveDefinite,
            ));
        }
        let (u, v) = match form {
            HessianForm::Hessian => (s, y),
            HessianForm::Inverse => (y, s),
        };
        self.modified.copy_from_slice(v);
        let result = crate::context::invoke(
            observer,
            scioxide_core::EvaluationRole::LinearOperator,
            || self.calculate(u, rule, form, source, scale_value, execution),
        );
        let report = match result {
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                ) =>
            {
                return Ok(HessianUpdateReport::Skipped(
                    UpdateSkipReason::Unrepresentable,
                ));
            }
            other => other?,
        };
        if matches!(report, HessianUpdateReport::Skipped(_)) {
            return Ok(report);
        }
        if matches!(rule, HessianUpdate::Bfgs(_)) {
            let valid = if self.lower {
                (0..n).all(|i| self.candidate.as_slice()[i * n + i] > S::ZERO)
            } else {
                self.validate_pd(execution, observer)?
            };
            if !valid {
                return Ok(HessianUpdateReport::Skipped(
                    UpdateSkipReason::NotPositiveDefinite,
                ));
            }
        }
        execution.check()?;
        Ok(report)
    }
    fn calculate(
        &mut self,
        u: &[S],
        rule: HessianUpdate<S>,
        form: HessianForm,
        source: MatrixView<'_, S>,
        scale: Option<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<HessianUpdateReport, SciOxideError> {
        let n = u.len();
        if self.lower {
            if let Some(diagonal) = scale {
                for (product, &value) in self.product.iter_mut().zip(u) {
                    let native = diagonal * value;
                    *product = if diagonal == S::ZERO || value == S::ZERO {
                        S::ZERO
                    } else if native.is_finite() && native.abs() >= S::MIN_POSITIVE {
                        native
                    } else {
                        Scaled::from_value(diagonal)?
                            .mul(Scaled::from_value(value)?)?
                            .round()?
                    };
                }
            } else {
                let original = source.as_contiguous_slice().ok_or_else(|| {
                    SciOxideError::invalid_input(
                        scioxide_core::argument!("source"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "contiguous source required",
                    )
                })?;
                lower_product(
                    original,
                    u,
                    &mut self.product,
                    &mut self.product_lanes,
                    model_product_uncertainty(original, u)?,
                )?;
            }
        } else {
            for (row, product) in self.product.iter_mut().enumerate() {
                *product =
                    scaled_dot(&self.candidate.as_slice()[row * n..(row + 1) * n], u)?.round()?;
            }
        }
        execution.check()?;
        let (denominator, model_denominator, report) = match rule {
            HessianUpdate::Bfgs(policy) => {
                let assessment = assess_pair(u, &mut self.modified, &self.product, policy)?;
                if matches!(assessment.report, HessianUpdateReport::Skipped(_)) {
                    return Ok(assessment.report);
                }
                (assessment.curvature, assessment.model, assessment.report)
            }
            HessianUpdate::Sr1 => {
                for (v, &p) in self.modified.iter_mut().zip(&self.product) {
                    *v = Scaled::from_value(*v)?
                        .sub(Scaled::from_value(p)?)?
                        .round()?;
                }
                let denominator = scaled_dot(u, &self.modified)?;
                let cutoff = scaled_dot(u, u)?
                    .mul(scaled_dot(&self.modified, &self.modified)?)?
                    .sqrt()?
                    .mul(Scaled::from_value(<S as RealScalar>::SQRT_EPSILON)?)?;
                if denominator.is_zero() || denominator.abs_le(cutoff) {
                    return Ok(HessianUpdateReport::Skipped(
                        UpdateSkipReason::SmallDenominator,
                    ));
                }
                (denominator, Scaled::ZERO, HessianUpdateReport::Updated)
            }
        };
        execution.check()?;
        let inverse = matches!(rule, HessianUpdate::Bfgs(_)) && form == HessianForm::Inverse;
        let bfgs = matches!(rule, HessianUpdate::Bfgs(_));
        let fast_inverse = if self.lower && inverse {
            denominator
                .round()
                .ok()
                .zip(model_denominator.round().ok())
                .and_then(|(d, m)| {
                    if d < S::MIN_POSITIVE || m < S::MIN_POSITIVE {
                        return None;
                    }
                    let ratio = m / d;
                    let a = (S::ONE + ratio) / d;
                    let b = S::ONE / d;
                    (ratio.is_finite()
                        && a.is_finite()
                        && b.is_finite()
                        && a >= S::MIN_POSITIVE
                        && b >= S::MIN_POSITIVE)
                        .then_some((a, b))
                })
        } else {
            None
        };
        if let Some((a, b)) = fast_inverse
            && self.lower
            && inverse
            && ordinary_frame(&[a, b])
            && ordinary_frame(&self.modified)
            && ordinary_frame(&self.product)
            && source.as_contiguous_slice().is_some()
        {
            // Admit the coefficient/vector frame once, then update the trusted
            // lower triangle contiguously. Full-matrix callers retain the
            // component-wise checked kernel and admission boundary below.
            let original = source.as_contiguous_slice().ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("source"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "contiguous source required",
                )
            })?;
            for (row, (values, base)) in self
                .candidate
                .as_mut_slice()
                .chunks_exact_mut(n)
                .zip(original.chunks_exact(n))
                .enumerate()
            {
                let vr = self.modified[row];
                let pr = self.product[row];
                for (col, ((value, &vc), &pc)) in values[..=row]
                    .iter_mut()
                    .zip(&self.modified)
                    .zip(&self.product)
                    .enumerate()
                {
                    let source =
                        scale.map_or(
                            base[col],
                            |diagonal| if row == col { diagonal } else { S::ZERO },
                        );
                    let first = (a * vr) * vc;
                    let second = (vr * pc + pr * vc) * b;
                    *value = (source + first) - second;
                }
            }
            let valid = self
                .candidate
                .as_slice()
                .chunks_exact(n)
                .zip(original.chunks_exact(n))
                .enumerate()
                .all(|(row, (values, original))| {
                    values[..=row]
                        .iter()
                        .zip(original)
                        .fold(true, |valid, (&value, &source)| {
                            valid
                                & (value.is_finite()
                                    & ((value.abs() >= S::MIN_POSITIVE)
                                        | ((value == S::ZERO) & (source == S::ZERO))))
                        })
                });
            if valid {
                return Ok(report);
            }
        }
        // The cold kernel also proposes from the immutable accepted source.
        // Only a successful acceptance swaps the candidate into the solver.
        if self.lower {
            let original = source.as_contiguous_slice().ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("source"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "contiguous source required",
                )
            })?;
            for (row, (out, base)) in self
                .candidate
                .as_mut_slice()
                .chunks_exact_mut(n)
                .zip(original.chunks_exact(n))
                .enumerate()
            {
                if let Some(diagonal) = scale {
                    out[..=row].fill(S::ZERO);
                    out[row] = diagonal;
                } else {
                    out[..=row].copy_from_slice(&base[..=row]);
                }
            }
        }
        let one = Scaled::from_value(<S as scioxide_core::AlgebraicScalar>::ONE)?;
        let first_coefficient = if inverse {
            one.add(model_denominator.div(denominator)?)?
                .div(denominator)?
        } else {
            one.div(denominator)?
        };
        let second_coefficient = if inverse {
            one.div(denominator)?
        } else if bfgs {
            one.div(model_denominator)?
        } else {
            Scaled::ZERO
        };
        let native = first_coefficient
            .round()
            .ok()
            .zip(second_coefficient.round().ok())
            .filter(|&(a, b)| {
                NumericScalar::abs(a) >= S::MIN_POSITIVE
                    && (!bfgs || NumericScalar::abs(b) >= S::MIN_POSITIVE)
            });
        for row in 0..n {
            for col in 0..=row {
                let source = self.candidate.as_slice()[row * n + col];
                let vr = self.modified[row];
                let vc = self.modified[col];
                let pr = self.product[row];
                let pc = self.product[col];
                let fast = native.and_then(|(a, b)| {
                    let first = normal_product(normal_product(a, vr)?, vc)?;
                    let second = if inverse {
                        let cross = resolved_sum(normal_product(vr, pc)?, normal_product(pr, vc)?)?;
                        if !cross.is_finite() {
                            return None;
                        }
                        normal_product(b, cross)?
                    } else if bfgs {
                        normal_product(normal_product(b, pr)?, pc)?
                    } else {
                        <S as scioxide_core::AlgebraicScalar>::ZERO
                    };
                    resolved_sum(resolved_sum(source, first)?, -second)
                });
                let rounded = if let Some(value) = fast {
                    value
                } else {
                    let first = first_coefficient
                        .mul(Scaled::from_value(vr)?)?
                        .mul(Scaled::from_value(vc)?)?;
                    let second = if inverse {
                        second_coefficient.mul(
                            Scaled::from_value(vr)?
                                .mul(Scaled::from_value(pc)?)?
                                .add(Scaled::from_value(pr)?.mul(Scaled::from_value(vc)?)?)?,
                        )?
                    } else if bfgs {
                        second_coefficient
                            .mul(Scaled::from_value(pr)?)?
                            .mul(Scaled::from_value(pc)?)?
                    } else {
                        Scaled::ZERO
                    };
                    let mut sum = ScaledAccumulator::from_scaled(Scaled::from_value(source)?);
                    sum.add_scaled(first)?;
                    sum.sub_scaled(second)?;
                    sum.round()?
                };
                self.candidate.as_mut_slice()[row * n + col] = rounded;
                if !self.lower {
                    self.candidate.as_mut_slice()[col * n + row] = rounded;
                }
            }
        }
        Ok(report)
    }
}

/// Gives the reason nonlinear CG restarted with the steepest-descent direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgRestartReason {
    /// Periodic restart after `n` iterations, where `n` is the dimension.
    Periodic,
    /// The Polak–Ribière+ rule truncated a negative `β` to zero.
    NegativeBeta,
    /// A coefficient or direction is not representable in the scalar type.
    Unrepresentable,
    /// The new direction is not a descent direction.
    DescentLoss,
}

/// Reports quasi-Newton update and conjugate-gradient restart counts at the last accepted point.
#[derive(Debug, Clone, Copy)]
pub struct GlobalizationDiagnostics<S: RealScalar> {
    /// Number of undamped updates.
    pub(crate) updates: usize,
    /// Number of damped updates.
    pub(crate) damped_updates: usize,
    /// Number of skipped updates.
    pub(crate) skipped_updates: usize,
    /// Outcome of the most recent update.
    pub(crate) last_update: Option<HessianUpdateReport>,
    /// Number of conjugate-gradient restarts.
    pub(crate) restarts: usize,
    /// Reason for the most recent restart.
    pub(crate) last_restart: Option<CgRestartReason>,
    /// Last Polak–Ribière+ coefficient `β`.
    pub(crate) beta: Option<S>,
    /// `|cos θ|` for consecutive gradients.
    pub(crate) gradient_orthogonality: Option<S>,
}

impl<S: RealScalar> GlobalizationDiagnostics<S> {
    /// Returns the number of undamped updates.
    pub const fn updates(self) -> usize {
        self.updates
    }
    /// Returns the number of damped updates.
    pub const fn damped_updates(self) -> usize {
        self.damped_updates
    }
    /// Returns the number of skipped updates.
    pub const fn skipped_updates(self) -> usize {
        self.skipped_updates
    }
    /// Returns the outcome of the most recent update.
    pub const fn last_update(self) -> Option<HessianUpdateReport> {
        self.last_update
    }
    /// Returns the number of conjugate-gradient restarts.
    pub const fn restarts(self) -> usize {
        self.restarts
    }
    /// Returns the reason for the most recent restart.
    pub const fn last_restart(self) -> Option<CgRestartReason> {
        self.last_restart
    }
    /// Returns the last Polak–Ribière+ coefficient `β`, or `None` before the first accepted step.
    pub const fn beta(self) -> Option<S> {
        self.beta
    }
    /// Returns `|cos θ|` for the angle `θ` between consecutive gradients, or `None` for a zero gradient.
    pub const fn gradient_orthogonality(self) -> Option<S> {
        self.gradient_orthogonality
    }
}
impl<S: RealScalar> Default for GlobalizationDiagnostics<S> {
    fn default() -> Self {
        Self {
            updates: 0,
            damped_updates: 0,
            skipped_updates: 0,
            last_update: None,
            restarts: 0,
            last_restart: None,
            beta: None,
            gradient_orthogonality: None,
        }
    }
}
impl<S: RealScalar> GlobalizationDiagnostics<S> {
    pub(crate) fn record_update(&mut self, report: HessianUpdateReport) {
        self.last_update = Some(report);
        match report {
            HessianUpdateReport::Updated => self.updates += 1,
            HessianUpdateReport::Damped => self.damped_updates += 1,
            HessianUpdateReport::Skipped(_) => self.skipped_updates += 1,
        }
    }
    pub(crate) fn restart(&mut self, reason: CgRestartReason) {
        self.restarts += 1;
        self.last_restart = Some(reason);
    }
}

/// Keep the line-search ABI's scalar slope representable by rescaling only
/// directions whose dot product would overflow or disappear through underflow.
pub(crate) fn descent_slope<S: RealScalar>(
    gradient: &[S],
    direction: &mut [S],
    normalize_extreme: bool,
) -> Result<S, SciOxideError> {
    let slope = scaled_dot(gradient, direction)?;
    let magnitude = direction.iter().fold(S::ZERO, |max, &d| max.max(d.abs()));
    let small = S::SQRT_EPSILON;
    let extreme = normalize_extreme
        && (magnitude > S::ONE / small || (magnitude != S::ZERO && magnitude < small));
    match slope.round() {
        Ok(value) if value != S::ZERO && !extreme => return Ok(value),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) => {}
        Ok(_) => (),
        Err(error) => return Err(error),
    }

    if magnitude == S::ZERO {
        return Ok(S::ZERO);
    }
    let scale = Scaled::from_value(magnitude)?.mul(Scaled::from_value(
        S::try_from_usize_exact(direction.len())?,
    )?)?;
    for d in direction.iter_mut() {
        *d = Scaled::from_value(*d)?.div(scale)?.round()?;
    }
    scaled_dot(gradient, direction)?.round()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn verify<S: RealScalar + NumericScalar<Real = S>>() {
        let cv = |v| S::try_from_f64_lossy(v).unwrap();
        let mut workspace = HessianUpdateWorkspace::new();
        workspace.try_reserve(2).unwrap();
        let mut b = Matrix::from_vec(2, 2, vec![cv(2.), cv(0.), cv(0.), cv(3.)]).unwrap();
        let s = [cv(1.), cv(2.)];
        let y = [cv(3.), cv(5.)];
        let rule = HessianUpdate::Bfgs(CurvaturePolicy::default());
        assert_eq!(
            workspace
                .update(&mut b, &s, &y, HessianForm::Hessian, rule)
                .unwrap(),
            HessianUpdateReport::Updated
        );
        for (row, &expected) in y.iter().enumerate() {
            let actual = scaled_dot(&b.as_slice()[row * 2..row * 2 + 2], &s)
                .unwrap()
                .round()
                .unwrap();
            assert!(NumericScalar::abs(actual - expected) < cv(100.) * <S as RealScalar>::EPSILON);
        }
        assert_eq!(b.as_slice()[1], b.as_slice()[2]);
        let mut h = Matrix::from_vec(2, 2, vec![cv(0.5), cv(0.), cv(0.), cv(1. / 3.)]).unwrap();
        workspace
            .update(&mut h, &s, &y, HessianForm::Inverse, rule)
            .unwrap();
        // Independently verify that inverse-BFGS and Hessian-BFGS are inverses.
        for row in 0..2 {
            for col in 0..2 {
                let actual = b.as_slice()[row * 2] * h.as_slice()[col]
                    + b.as_slice()[row * 2 + 1] * h.as_slice()[2 + col];
                let expected = cv(if row == col { 1. } else { 0. });
                assert!(
                    NumericScalar::abs(actual - expected) < cv(100.) * <S as RealScalar>::EPSILON
                );
            }
        }
        let report = workspace
            .update(&mut b, &s, &[cv(-1.), cv(-2.)], HessianForm::Hessian, rule)
            .unwrap();
        assert_eq!(report, HessianUpdateReport::Damped);
        let before = b.as_slice().to_vec();
        let report = workspace
            .update(
                &mut b,
                &s,
                &[cv(-1.), cv(-2.)],
                HessianForm::Hessian,
                HessianUpdate::Bfgs(CurvaturePolicy::new(CurvatureAction::Skip, cv(0.2)).unwrap()),
            )
            .unwrap();
        assert_eq!(
            report,
            HessianUpdateReport::Skipped(UpdateSkipReason::Curvature)
        );
        assert_eq!(b.as_slice(), before);
    }
    #[test]
    fn bfgs_f64() {
        verify::<f64>();
    }
    #[test]
    fn bfgs_f32() {
        verify::<f32>();
    }
    #[test]
    fn sr1_indefinite_and_skip() {
        let mut workspace = HessianUpdateWorkspace::new();
        workspace.try_reserve(2).unwrap();
        let mut b = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]).unwrap();
        assert_eq!(
            workspace
                .update(
                    &mut b,
                    &[1., 0.],
                    &[-2., 0.],
                    HessianForm::Hessian,
                    HessianUpdate::Sr1
                )
                .unwrap(),
            HessianUpdateReport::Updated
        );
        assert_eq!(b.as_slice(), &[-2., 0., 0., 1.]);
        assert_eq!(
            workspace
                .update(
                    &mut b,
                    &[1., 0.],
                    &[-2., 0.],
                    HessianForm::Hessian,
                    HessianUpdate::Sr1
                )
                .unwrap(),
            HessianUpdateReport::Skipped(UpdateSkipReason::SmallDenominator)
        );
    }
    #[test]
    fn scaled_curvature_does_not_use_absolute_epsilon() {
        for scale in [1e-200, 1., 1e200] {
            let mut workspace = HessianUpdateWorkspace::new();
            workspace.try_reserve(2).unwrap();
            let mut b = Matrix::from_vec(2, 2, vec![scale, 0., 0., scale]).unwrap();
            assert_eq!(
                workspace
                    .update(
                        &mut b,
                        &[1., 0.],
                        &[2. * scale, 0.],
                        HessianForm::Hessian,
                        HessianUpdate::Bfgs(CurvaturePolicy::default())
                    )
                    .unwrap(),
                HessianUpdateReport::Updated
            );
            assert!((b.as_slice()[0] / scale - 2.).abs() < 1e-14);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod transactional_tests {
    use super::*;
    use alloc::vec;
    use core::cell::Cell;

    #[test]
    fn compensated_dot_retains_cancellation_and_extended_range() {
        assert_eq!(
            scaled_dot(&[1e16, 1., -1e16], &[1.; 3])
                .unwrap()
                .round()
                .unwrap(),
            1.
        );
        assert!(
            scaled_dot(&[f64::MAX; 2], &[f64::MAX, -f64::MAX])
                .unwrap()
                .is_zero()
        );
        let tiny = scaled_dot(&[f64::from_bits(1)], &[0.5]).unwrap();
        assert!(!tiny.is_zero());
        assert_eq!(tiny.round().unwrap(), 0.);
        assert!(scaled_dot(&[1.], &[]).is_err());
        assert!(scaled_dot(&[f64::NAN], &[1.]).is_err());
    }

    #[test]
    fn lower_model_product_ignores_upper_storage_and_retries_overflow() {
        let matrix = [2., f64::NAN, f64::NAN, 1., 3., f64::NAN, -0.5, 0.25, 4.];
        let mut output = [0.; 3];
        lower_product(
            &matrix,
            &[1., -2., 3.],
            &mut output,
            &mut [0.; 12],
            f64::MIN_POSITIVE,
        )
        .unwrap();
        for (&a, b) in output.iter().zip([-1.5, -4.25, 11.]) {
            assert!((a - b).abs() < 1e-14);
        }
        let mut output = [1.; 2];
        lower_product(
            &[f64::MAX, f64::NAN, f64::MAX, f64::MAX],
            &[f64::MAX, -f64::MAX],
            &mut output,
            &mut [0.; 8],
            f64::MIN_POSITIVE,
        )
        .unwrap();
        assert_eq!(output, [0.; 2]);
    }

    #[test]
    fn model_product_retries_a_normal_but_cancellation_sensitive_sum() {
        let n = 36;
        let block = [
            1e16,
            0.,
            0.,
            0.,
            1.,
            3.,
            0.,
            0.,
            -1e16,
            0.,
            1e16 + 100.,
            0.,
            2.,
            0.,
            0.,
            3.,
        ];
        let mut matrix = alloc::vec![0.;n*n];
        for row in 0..n {
            matrix[row * n + row] = 1.;
        }
        for row in 0..4 {
            for col in 0..=row {
                matrix[row * n + col] = block[row * 4 + col];
            }
        }
        let mut vector = alloc::vec![0.;n];
        vector[..4].fill(1.);
        let mut product = alloc::vec![0.;n];
        let mut lanes = alloc::vec![0.;4*n];
        lower_product(
            &matrix,
            &vector,
            &mut product,
            &mut lanes,
            f64::MIN_POSITIVE,
        )
        .unwrap();
        assert_eq!(product[0], 2.);
        let guard = model_product_uncertainty(&matrix, &vector).unwrap();
        lower_product(&matrix, &vector, &mut product, &mut lanes, guard).unwrap();
        assert!((product[0] - 3.).abs() < 1e-14);
    }

    struct Admission {
        calls: Cell<usize>,
        deny: usize,
    }
    impl crate::context::Observation for Admission {
        fn execution(&self) -> Result<ExecutionContext<'_>, SciOxideError> {
            Ok(ExecutionContext::sequential())
        }
        fn counts(&self) -> Result<scioxide_core::EvaluationCounts, SciOxideError> {
            Ok(scioxide_core::EvaluationCounts::new())
        }

        fn cache_hit(&self) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn proposal(&self) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn reject(&self) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn admit(&self, _: scioxide_core::EvaluationRole) -> Result<(), SciOxideError> {
            let calls = self.calls.get();
            if calls == self.deny {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("role"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "provider denied",
                ));
            }
            self.calls.set(calls + 1);
            Ok(())
        }
    }
    #[test]
    fn caller_admission_and_candidate_validation_remain_transactional() {
        // SPD but not diagonally dominant: exercises Cholesky fallback, not
        // the sufficient diagonal-dominance shortcut. The exact update is B.
        let mut matrix = Matrix::from_vec(2, 2, vec![1., 2., 2., 5.]).unwrap();
        let original = matrix.as_slice().to_vec();
        let mut workspace = HessianUpdateWorkspace::new();
        workspace.try_reserve(2).unwrap();
        for deny in [0, 1, 2] {
            let admission = Admission {
                calls: Cell::new(0),
                deny,
            };
            let result = workspace.update_observed(
                &mut matrix,
                &[1., 0.],
                &[1., 2.],
                HessianForm::Hessian,
                HessianUpdate::Bfgs(CurvaturePolicy::default()),
                ExecutionContext::sequential(),
                Some(&admission),
                scioxide_core::argument!("matrix"),
            );
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
            ));
            assert_eq!(matrix.as_slice(), original);
            assert_eq!(admission.calls.get(), deny);
        }
        let admission = Admission {
            calls: Cell::new(0),
            deny: 3,
        };
        let report = workspace
            .update_observed(
                &mut matrix,
                &[1., 0.],
                &[1., 2.],
                HessianForm::Hessian,
                HessianUpdate::Bfgs(CurvaturePolicy::default()),
                ExecutionContext::sequential(),
                Some(&admission),
                scioxide_core::argument!("matrix"),
            )
            .unwrap();
        assert_eq!(report, HessianUpdateReport::Updated);
        assert_eq!(admission.calls.get(), 3);
        assert_eq!(matrix.as_slice(), original);
    }
    #[test]
    fn nonfinite_input_and_indefinite_model_are_distinct() {
        let mut matrix = Matrix::from_vec(2, 2, vec![1., 0., 0., -1.]).unwrap();
        let mut workspace = HessianUpdateWorkspace::new();
        workspace.try_reserve(2).unwrap();
        let rule = HessianUpdate::Bfgs(CurvaturePolicy::default());
        assert_eq!(
            workspace
                .update(
                    &mut matrix,
                    &[1., 0.],
                    &[2., 0.],
                    HessianForm::Hessian,
                    rule
                )
                .unwrap(),
            HessianUpdateReport::Skipped(UpdateSkipReason::NotPositiveDefinite)
        );
        assert!(matches!(
            (workspace.update(
                &mut matrix,
                &[f64::NAN, 0.],
                &[2., 0.],
                HessianForm::Hessian,
                rule
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert_eq!(matrix.as_slice(), &[1., 0., 0., -1.]);
    }
}
