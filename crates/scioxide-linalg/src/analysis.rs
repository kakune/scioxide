//! Rank classification and the acceptance checks of direct solves.
//!
//! Factorizations fail only on structural breakdown. The types in this module
//! classify numerical rank and decide, as a separate step, whether a solve is
//! accepted.

use alloc::vec::Vec;

use scioxide_core::checked_float_quotient;
use scioxide_core::{
    EstimateReliability, InvalidInputReason, NonFiniteOrigin, NumericalIssue, RankThreshold,
    RealScalar, ReliableEstimate, ResidualReport, SciOxideError,
};

use crate::checked::dimension_epsilon;

/// Threshold that decides which singular values, eigenvalue moduli or pivot
/// moduli count toward the numerical rank.
///
/// `scale` is the largest of the classified values. A value counts as nonzero
/// when it exceeds the threshold `τ` below.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum RankPolicy<R: RealScalar> {
    /// Counts every value that is not exactly zero (`τ = 0`).
    Exact,
    /// Uses `τ = min(max(m, n) ε, 1/2) · scale` for an `m × n` matrix, where `ε`
    /// is the machine epsilon.
    DimensionScaledEpsilon,
    /// Uses `τ = atol + rtol · scale` from the given [`RankThreshold`].
    Threshold(RankThreshold<R>),
}

impl<R: RealScalar> RankPolicy<R> {
    pub(crate) fn cutoff(self, scale: R, rows: usize, cols: usize) -> Result<R, SciOxideError> {
        if !scale.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "rank scale must be finite",
            ));
        }
        if scale < R::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("policy"),
                InvalidInputReason::OutOfRange,
                "rank scale must be nonnegative",
            ));
        }
        match self {
            Self::Exact => Ok(R::ZERO),
            Self::DimensionScaledEpsilon if scale == R::ZERO => Ok(R::ZERO),
            Self::DimensionScaledEpsilon => {
                crate::checked::finite_candidate(scale * dimension_epsilon(rows.max(cols))?)
            }
            Self::Threshold(threshold) => threshold.threshold(scale),
        }
    }
}

/// Numerical rank of a matrix under a [`RankPolicy`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RankReport<R: RealScalar> {
    rank: usize,
    maximum_rank: usize,
    threshold: R,
    scale: R,
}

impl<R: RealScalar> RankReport<R> {
    pub(crate) const fn from_factor(
        rank: usize,
        maximum_rank: usize,
        threshold: R,
        scale: R,
    ) -> Self {
        Self {
            rank,
            maximum_rank,
            threshold,
            scale,
        }
    }
    /// Returns the numerical rank, the number of values above the threshold.
    pub const fn rank(self) -> usize {
        self.rank
    }

    /// Returns the largest possible rank for the matrix shape, `min(m, n)` for an
    /// `m × n` matrix.
    pub const fn maximum_rank(self) -> usize {
        self.maximum_rank
    }

    /// Returns the absolute threshold `τ` that was applied.
    pub const fn threshold(self) -> R {
        self.threshold
    }

    /// Returns the largest classified value, the scale of the relative threshold.
    ///
    /// Under [`RankPolicy::Exact`], a value beyond the scalar range is reported
    /// as the largest finite value; the rank still counts exact zeros only.
    pub const fn scale(self) -> R {
        self.scale
    }

    /// Returns whether [`Self::rank`] equals [`Self::maximum_rank`].
    pub const fn is_full_rank(self) -> bool {
        self.rank == self.maximum_rank
    }
}

pub(crate) fn analyze_magnitudes<R: RealScalar, I>(
    magnitudes: I,
    maximum_rank: usize,
    rows: usize,
    cols: usize,
    policy: RankPolicy<R>,
) -> Result<RankReport<R>, SciOxideError>
where
    I: IntoIterator<Item = R>,
    I::IntoIter: Clone,
{
    analyze_magnitudes_iter(magnitudes.into_iter(), maximum_rank, rows, cols, policy)
}

/// Classify finite-component structured pivots without projecting their magnitudes first.
pub(crate) fn analyze_scaled_magnitudes<R: RealScalar>(
    magnitudes: impl Iterator<Item = Result<scioxide_core::Scaled<R>, SciOxideError>> + Clone,
    order: usize,
    policy: RankPolicy<R>,
) -> Result<RankReport<R>, SciOxideError> {
    let mut largest = scioxide_core::Scaled::ZERO;
    for value in magnitudes.clone() {
        let value = value?;
        if !value.abs_le(largest) {
            largest = value;
        }
    }
    let scale = if matches!(policy, RankPolicy::Exact)
        && !largest.abs_le(scioxide_core::Scaled::from_value(R::MAX)?)
    {
        R::MAX
    } else {
        largest.round()?
    };
    let threshold = policy.cutoff(scale, order, order)?;
    let cutoff = scioxide_core::Scaled::from_value(threshold)?;
    let mut rank = 0;
    for value in magnitudes {
        if !value?.abs_le(cutoff) {
            rank += 1;
        }
    }
    Ok(RankReport::from_factor(rank, order, threshold, scale))
}

fn analyze_magnitudes_iter<R: RealScalar, I: Iterator<Item = R> + Clone>(
    magnitudes: I,
    maximum_rank: usize,
    rows: usize,
    cols: usize,
    policy: RankPolicy<R>,
) -> Result<RankReport<R>, SciOxideError> {
    let mut scale = R::ZERO;
    for value in magnitudes.clone() {
        if !value.is_finite() || value < R::ZERO {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Intermediate,
                "analyzed magnitudes must be finite and nonnegative",
            ));
        }
        scale = scale.max(value);
    }
    // The cutoff is relative to the largest magnitude (e.g.
    // `max(m, n) eps sigma_max`), so rank decisions are invariant under
    // uniform scaling of the input. `Exact` counts only true zeros.
    let threshold = policy.cutoff(scale, rows, cols)?;
    let rank = match policy {
        RankPolicy::Exact => magnitudes.filter(|value| *value != R::ZERO).count(),
        _ => magnitudes.filter(|value| *value > threshold).count(),
    };
    Ok(RankReport {
        rank,
        maximum_rank,
        threshold,
        scale,
    })
}

pub(crate) fn condition_ratio<R: RealScalar>(smallest: R, largest: R) -> Result<R, SciOxideError> {
    if smallest == R::ZERO || largest == R::ZERO {
        return Ok(R::ZERO);
    }
    // Exact zero is reserved for a structurally zero magnitude. A nonzero
    // ratio that underflows (extreme but finite ill-conditioning) is reported
    // as the smallest representable value `2^(1-p) * MIN_POSITIVE` instead,
    // so callers can still distinguish "singular" from "badly conditioned".
    let ratio = checked_float_quotient(smallest, largest)?;
    if ratio != R::ZERO {
        return Ok(ratio.min(R::ONE));
    }
    let digits = i32::try_from(R::MANTISSA_DIGITS).map_err(|_| {
        SciOxideError::size_overflow(scioxide_core::quantity!("condition_ratio.exponent"))
    })?;
    let exponent = 1_i32.checked_sub(digits).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("condition_ratio.exponent"))
    })?;
    Ok(R::MIN_POSITIVE.scalbn(exponent))
}

/// Estimate of the reciprocal condition number `rcond = 1 / κ(A)`, in `[0, 1]`.
///
/// The norm is stated by the method that returns the estimate; for example,
/// [`LuDecomposition::condition_estimate`](crate::LuDecomposition::condition_estimate)
/// estimates `1 / (‖A‖₁ ‖A⁻¹‖₁)`. Zero means that the matrix is exactly singular.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ConditionEstimate<R: RealScalar> {
    reciprocal: ReliableEstimate<R>,
}

impl<R: RealScalar> ConditionEstimate<R> {
    /// Construct a finite reciprocal condition estimate in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an input error when the estimate lies outside `[0, 1]` or is
    /// non-finite.
    pub(crate) fn new(
        reciprocal: R,
        reliability: EstimateReliability,
    ) -> Result<Self, SciOxideError> {
        validate_reciprocal(reciprocal)?;
        Ok(Self {
            reciprocal: ReliableEstimate::new(reciprocal, reliability)?,
        })
    }

    /// Returns the estimate together with its [`EstimateReliability`].
    pub const fn reciprocal(self) -> ReliableEstimate<R> {
        self.reciprocal
    }

    /// Returns whether the estimate is exactly zero, that is, the matrix is exactly singular.
    pub fn is_singular(self) -> bool {
        self.reciprocal.value() == R::ZERO
    }
}

/// Smallest reciprocal condition estimate that a direct solve accepts, in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ReciprocalConditionThreshold<R: RealScalar>(R);

impl<R: RealScalar> ReciprocalConditionThreshold<R> {
    /// Creates a threshold from `value`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `value` is outside `[0, 1]`.
    pub fn new(value: R) -> Result<Self, SciOxideError> {
        validate_reciprocal(value)?;
        Ok(Self(value))
    }

    /// Returns the threshold value.
    pub const fn get(self) -> R {
        self.0
    }
}

/// Row and column scaling by powers of two, `D_r A D_c`, applied before factoring.
///
/// Powers of two scale exactly. Row `i` is multiplied by `2^row_exponents[i]`
/// and column `j` by `2^column_exponents[j]`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct EquilibrationMetadata {
    pub(crate) row_exponents: Vec<i32>,
    pub(crate) column_exponents: Vec<i32>,
}

impl EquilibrationMetadata {
    pub(crate) const fn new(row_exponents: Vec<i32>, column_exponents: Vec<i32>) -> Self {
        Self {
            row_exponents,
            column_exponents,
        }
    }

    /// Returns the exponents `e_i` of the row scaling `D_r = diag(2^e_i)`.
    pub fn row_exponents(&self) -> &[i32] {
        &self.row_exponents
    }

    /// Returns the exponents `e_j` of the column scaling `D_c = diag(2^e_j)`.
    pub fn column_exponents(&self) -> &[i32] {
        &self.column_exponents
    }

    /// Returns whether every exponent is zero, so that `D_r = D_c = I`.
    pub fn is_identity(&self) -> bool {
        self.row_exponents
            .iter()
            .chain(&self.column_exponents)
            .all(|exponent| *exponent == 0)
    }
}

/// Acceptance checks for a solve with an existing factorization.
///
/// A solve is accepted when the factor has full rank under the [`RankPolicy`]
/// and, if a [`ReciprocalConditionThreshold`] is set, the condition estimate is
/// at least that threshold. The condition number is estimated only when a
/// threshold is set. The residual diagnostic is off by default
/// ([`Estimation::Skip`](crate::Estimation::Skip)). See
/// [Direct solves](crate#direct-solves).
///
/// A solve of an empty system returns the empty solution, but a condition
/// threshold on an empty system is a `DomainError`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{DirectSolveOptions, Estimation, LuDecomposition, Matrix, RankPolicy};
/// let factor = LuDecomposition::factor(&Matrix::identity(2)?)?;
/// let options = DirectSolveOptions::new(RankPolicy::Exact).with_residual(Estimation::Estimate);
/// let result = factor.solve_with_options(&[1.0_f64, 2.0][..], &options)?;
/// assert_eq!(result.solution().unwrap().as_slice(), &[1.0, 2.0]);
/// assert_eq!(result.report().residual().unwrap().scaled_norm(), 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DirectSolveOptions<R: RealScalar> {
    rank_policy: RankPolicy<R>,
    minimum_reciprocal_condition: Option<ReciprocalConditionThreshold<R>>,
    residual: crate::Estimation,
    execution: crate::LinalgExecutionOptions,
}

impl<R: RealScalar> Default for DirectSolveOptions<R> {
    /// Returns options with [`RankPolicy::Exact`], which applies no numerical
    /// rank cutoff, no condition threshold and no residual diagnostic.
    fn default() -> Self {
        Self::new(RankPolicy::Exact)
    }
}

impl<R: RealScalar> DirectSolveOptions<R> {
    /// Creates options with the given rank policy, no condition threshold and no
    /// residual diagnostic.
    pub const fn new(rank_policy: RankPolicy<R>) -> Self {
        Self {
            rank_policy,
            minimum_reciprocal_condition: None,
            residual: crate::Estimation::Skip,
            execution: crate::LinalgExecutionOptions::new(),
        }
    }

    /// Rejects the solve when the reciprocal condition estimate is below `threshold`.
    pub const fn with_minimum_reciprocal_condition(
        mut self,
        threshold: ReciprocalConditionThreshold<R>,
    ) -> Self {
        self.minimum_reciprocal_condition = Some(threshold);
        self
    }

    /// Returns the rank policy.
    pub const fn rank_policy(self) -> RankPolicy<R> {
        self.rank_policy
    }

    /// Returns the condition threshold, if one is set.
    pub const fn minimum_reciprocal_condition(self) -> Option<ReciprocalConditionThreshold<R>> {
        self.minimum_reciprocal_condition
    }

    /// Sets whether the solve computes the normwise backward error of its solution.
    ///
    /// For several right-hand sides the report holds the largest backward error
    /// over the columns.
    pub const fn with_residual(mut self, residual: crate::Estimation) -> Self {
        self.residual = residual;
        self
    }

    /// Returns whether the backward error is computed.
    pub const fn residual(self) -> crate::Estimation {
        self.residual
    }

    /// Sets the execution policy and limits of the solve.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the execution policy and limits of the solve.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }
}

pub(crate) fn empty_residual<R: RealScalar>(
    options: &DirectSolveOptions<R>,
    right_hand_sides: usize,
) -> Result<Option<ResidualReport<R>>, SciOxideError> {
    (options.residual() == crate::Estimation::Estimate && right_hand_sides != 0)
        .then(|| ResidualReport::unscaled(R::ZERO, scioxide_core::NormSelection::Infinity))
        .transpose()
}

pub(crate) fn empty_condition() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("self"),
        scioxide_core::DomainReason::Undefined,
        "empty factor has no condition estimate",
    )
}

pub(crate) fn acceptance<R: RealScalar>(
    rank: RankReport<R>,
    condition: Option<ConditionEstimate<R>>,
    options: &DirectSolveOptions<R>,
) -> DirectSolveReport<R> {
    let termination = if !rank.is_full_rank() {
        DirectSolveTermination::RankDeficient
    } else if options
        .minimum_reciprocal_condition()
        .is_some_and(|threshold| {
            condition.is_some_and(|estimate| estimate.reciprocal().value() < threshold.get())
        })
    {
        DirectSolveTermination::IllConditioned
    } else {
        DirectSolveTermination::Accepted
    };
    DirectSolveReport::new(termination, rank, condition, None)
}

pub(crate) fn retain_largest_residual<R: RealScalar>(
    largest: &mut Option<ResidualReport<R>>,
    candidate: ResidualReport<R>,
) {
    if largest.is_none_or(|prior| candidate.scaled_norm() > prior.scaled_norm()) {
        *largest = Some(candidate);
    }
}

/// Outcome of a direct solve: accepted, or the reason the solution was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DirectSolveTermination {
    /// The solution passed every check.
    Accepted,
    /// The factor does not have full rank under the [`RankPolicy`].
    RankDeficient,
    /// The reciprocal condition estimate is below the threshold.
    IllConditioned,
    /// The residual of the computed solution in the original system exceeds the
    /// requested tolerance.
    ResidualRejected,
    /// The Riccati solution does not make the closed-loop matrix stable.
    ///
    /// When both this check and the residual check fail, this status is reported.
    StabilityRejected,
}

impl DirectSolveTermination {
    /// Returns whether the status is [`Self::Accepted`].
    ///
    /// ```
    /// use scioxide_linalg::DirectSolveTermination;
    /// assert!(DirectSolveTermination::Accepted.is_success());
    /// assert!(!DirectSolveTermination::ResidualRejected.is_success());
    /// ```
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Returns `Ok` for [`Self::Accepted`] and the matching error otherwise.
    ///
    /// The plain solves and the one-shot functions report a rejection with
    /// this error.
    ///
    /// # Errors
    ///
    /// - `Singular` for [`Self::RankDeficient`].
    /// - `NumericalInstability` with [`NumericalIssue::IllConditioned`] for
    ///   [`Self::IllConditioned`] and [`Self::StabilityRejected`].
    /// - `NumericalInstability` with [`NumericalIssue::LossOfSignificance`] for
    ///   [`Self::ResidualRejected`].
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_linalg::DirectSolveTermination;
    /// assert!(DirectSolveTermination::Accepted.require_accepted().is_ok());
    /// let error = DirectSolveTermination::RankDeficient.require_accepted().unwrap_err();
    /// assert_eq!(error.kind(), SciOxideErrorKind::Singular);
    /// ```
    pub fn require_accepted(self) -> Result<(), SciOxideError> {
        match self {
            Self::Accepted => Ok(()),
            Self::RankDeficient => Err(SciOxideError::singular(None)),
            Self::IllConditioned => Err(SciOxideError::numerical_instability(
                NumericalIssue::IllConditioned,
                "reciprocal condition estimate is below the threshold",
            )),
            Self::ResidualRejected => Err(SciOxideError::numerical_instability(
                NumericalIssue::LossOfSignificance,
                "residual of the computed solution exceeds the tolerance",
            )),
            Self::StabilityRejected => Err(SciOxideError::numerical_instability(
                NumericalIssue::IllConditioned,
                "closed-loop matrix is not stable within the tolerance",
            )),
        }
    }
}

/// Status and diagnostics of a direct solve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DirectSolveReport<R: RealScalar> {
    termination: DirectSolveTermination,
    rank: RankReport<R>,
    condition: Option<ConditionEstimate<R>>,
    residual: Option<ResidualReport<R>>,
}

impl<R: RealScalar> DirectSolveReport<R> {
    pub(crate) const fn new(
        termination: DirectSolveTermination,
        rank: RankReport<R>,
        condition: Option<ConditionEstimate<R>>,
        residual: Option<ResidualReport<R>>,
    ) -> Self {
        Self {
            termination,
            rank,
            condition,
            residual,
        }
    }

    /// Returns the outcome of the solve.
    pub const fn termination(self) -> DirectSolveTermination {
        self.termination
    }

    /// Returns the rank classification of the factor.
    pub const fn rank(self) -> RankReport<R> {
        self.rank
    }

    /// Returns the condition estimate of the original matrix, present only when
    /// the options set a condition threshold.
    pub const fn condition(self) -> Option<ConditionEstimate<R>> {
        self.condition
    }

    /// Returns the normwise backward error, present when it was requested and a
    /// solution was computed.
    pub const fn residual(self) -> Option<ResidualReport<R>> {
        self.residual
    }
}

/// Solution of a direct solve together with its [`DirectSolveReport`].
///
/// The solution is available only when the solve was accepted.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DirectSolveResult<T, R: RealScalar> {
    solution: T,
    report: DirectSolveReport<R>,
}

impl<T, R: RealScalar> DirectSolveResult<T, R> {
    pub(crate) const fn new(solution: T, report: DirectSolveReport<R>) -> Self {
        Self { solution, report }
    }

    /// Returns the solution, or `None` if the solve was rejected.
    pub fn solution(&self) -> Option<&T> {
        self.report
            .termination()
            .is_success()
            .then_some(&self.solution)
    }

    /// Returns the solution by value, or `None` if the solve was rejected.
    pub fn into_solution(self) -> Option<T> {
        self.report
            .termination()
            .is_success()
            .then_some(self.solution)
    }

    /// Returns the solution of an accepted solve, or the error of
    /// [`DirectSolveTermination::require_accepted`] for a rejected one.
    ///
    /// # Errors
    ///
    /// `Singular` or `NumericalInstability` as listed on
    /// [`DirectSolveTermination::require_accepted`].
    pub fn into_accepted(self) -> Result<T, SciOxideError> {
        self.report.termination().require_accepted()?;
        Ok(self.solution)
    }

    /// Returns the status and diagnostics.
    pub const fn report(&self) -> &DirectSolveReport<R> {
        &self.report
    }
}

fn validate_reciprocal<R: RealScalar>(reciprocal: R) -> Result<(), SciOxideError> {
    if !reciprocal.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "estimate must be finite",
        ));
    }
    if reciprocal < R::ZERO || reciprocal > R::ONE {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("reciprocal"),
            InvalidInputReason::OutOfRange,
            "estimate must lie in [0, 1]",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_policy_is_explicit_and_scale_aware() {
        let report = analyze_magnitudes(
            [4.0_f64, 4.0 * f64::EPSILON],
            2,
            2,
            2,
            RankPolicy::DimensionScaledEpsilon,
        )
        .unwrap();
        assert_eq!(report.rank(), 1);
        assert_eq!(report.maximum_rank(), 2);

        let exact =
            analyze_magnitudes([4.0_f64, f64::MIN_POSITIVE], 2, 2, 2, RankPolicy::Exact).unwrap();
        assert_eq!(exact.rank(), 2);
    }

    #[test]
    fn condition_ratio_keeps_nonzero_extreme_values_distinct_from_singularity() {
        let f64_ratio = condition_ratio(f64::MIN_POSITIVE, f64::MAX).unwrap();
        assert!(f64_ratio > 0.0);
        assert!(f64_ratio.is_finite());

        let f32_ratio = condition_ratio(f32::MIN_POSITIVE, f32::MAX).unwrap();
        assert!(f32_ratio > 0.0);
        assert!(f32_ratio.is_finite());

        assert_eq!(condition_ratio(0.0_f64, f64::MAX).unwrap(), 0.0);
    }
}
