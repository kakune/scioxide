//! Multivariate normal (Gaussian) distribution.

use super::probability::{convenience, joint_traits, preflight};
use super::{MultivariateProbabilityOptions, MultivariateProbabilityPlan, number, values};
use super::{RowLaw, Scratch, vector_sampling};
use crate::BitGenerator;
use crate::distribution::LawOptions;
use crate::{Cdf, LogCdf, LogSurvival, Philox4x64, SeedableRng, Survival};
use alloc::vec::Vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError, Tolerance};
use scioxide_core::{RangeAccumulator, checked_product_quotient};
use scioxide_linalg::{HermitianEigenDecomposition, Matrix, MatrixView, RankPolicy, Vector};

use crate::distribution::Normal;
use crate::traits::Distribution;
use crate::workspace::SamplingWorkspace;

/// Options for [`MultivariateNormal`]: the rule that sets the numerical rank of the covariance.
///
/// Eigenvalues of the covariance at or below the rank threshold are treated as
/// zero. The covariance is never regularized.
///
/// ```
/// use scioxide_linalg::RankPolicy;
/// use scioxide_random::MultivariateNormalOptions;
/// let options = MultivariateNormalOptions::<f64>::default().with_rank_policy(RankPolicy::Exact);
/// assert_eq!(options.rank_policy(), RankPolicy::Exact);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultivariateNormalOptions<S: RealScalar> {
    /// Covariance eigenvalue threshold, in covariance units.
    rank_policy: RankPolicy<S>,
}
impl<S: RealScalar> Default for MultivariateNormalOptions<S> {
    fn default() -> Self {
        Self {
            rank_policy: RankPolicy::DimensionScaledEpsilon,
        }
    }
}

impl<S: RealScalar> MultivariateNormalOptions<S> {
    /// Returns the rule that sets the covariance rank.
    pub const fn rank_policy(self) -> RankPolicy<S> {
        self.rank_policy
    }
    /// Sets the rank rule. [`MultivariateNormal::new`] validates it.
    pub const fn with_rank_policy(mut self, value: RankPolicy<S>) -> Self {
        self.rank_policy = value;
        self
    }
}

/// Multivariate normal distribution with mean `μ ∈ ℝᵈ` and covariance `Σ`.
///
/// `Σ` may be positive semidefinite, including zero. With the eigendecomposition
/// `Σ = V Λ Vᵀ` truncated to its `r` eigenvalues above the rank threshold, a
/// draw is `x = μ + V Λ^(1/2) z` with `z` a vector of `r` independent standard
/// normals. The support is the affine subspace `μ + span(V)`, and densities are
/// taken with respect to `r`-dimensional Lebesgue measure on it.
///
/// Construction costs O(d³) time and O(d²) storage. Each draw costs O(d r).
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_linalg::{Matrix, Vector};
/// use scioxide_random::{LogDensity, MultivariateNormal, SeedableRng, SplitMix64};
/// let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 2.0])?;
/// let law = MultivariateNormal::new(&[1.0, -1.0], &covariance)?;
/// assert_eq!(law.rank(), 2);
/// // At the mean: -ln(2 pi) - ln(det covariance) / 2, with det = 1.75.
/// let expected = -(2.0 * core::f64::consts::PI).ln() - 0.5 * 1.75_f64.ln();
/// assert!((law.log_pdf(&[1.0, -1.0])? - expected).abs() < 1e-12);
/// let x = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(x.as_slice(), Tolerance::from_machine_epsilon())?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct MultivariateNormal<S: RealScalar + NumericScalar<Real = S>> {
    mean: Vector<S>,
    support: Matrix<S>,
    deviations: Vec<S>,
    factor: Matrix<S>,
    ordinary_factor: bool,
    log_normalizer: S,
    roundoff: S,
    log_pseudo_determinant: S,
    rank_options: MultivariateNormalOptions<S>,
    options: crate::LawOptions<S>,
    standard_normal: Normal<S>,
    dim: usize,
}

impl<S: RealScalar + NumericScalar<Real = S>> MultivariateNormal<S> {
    /// Creates the law from a mean vector and an exactly symmetric positive
    /// semidefinite covariance matrix.
    ///
    /// `covariance` is a `&Matrix` or any, possibly strided, [`MatrixView`];
    /// the mean and the covariance are copied. The covariance is scaled to unit
    /// maximum norm and diagonalized with a symmetric eigensolver. An eigenvalue
    /// `λ` with `λ < −8 d ε λ_max` makes the covariance indefinite. Smaller
    /// negative eigenvalues are rounding errors and count as zero. The rank is
    /// the number of eigenvalues above the threshold of
    /// [`MultivariateNormalOptions::rank_policy`].
    ///
    /// It uses the default [`MultivariateNormalOptions`] and
    /// [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the rank
    /// rule, the tolerance and the work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `d = 0`.
    /// - `DimensionMismatch` if `covariance` is not `d × d` for `d = mean.len()`.
    /// - `NonFinite` if `mean` or `covariance` has a NaN or infinite entry.
    /// - `DomainError` if `covariance` is not exactly symmetric, has a negative
    ///   diagonal entry or is indefinite.
    /// - `NumericalInstability` if scaling the covariance underflows a nonzero entry to zero.
    /// - `ConvergenceFailure` if the eigensolver exhausts the work limit, and
    ///   `ResourceLimit` if the `d × d` storage exceeds the byte limit.
    /// - `AllocationFailure` if the copies cannot be allocated.
    pub fn new<'a>(
        mean: &[S],
        covariance: impl Into<MatrixView<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(
            mean,
            covariance,
            MultivariateNormalOptions::default(),
            crate::LawOptions::default(),
        )
    }

    /// Creates the law of [`Self::new`] with the rank rule of `rank_options` and
    /// the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options<'a>(
        mean: &[S],
        covariance: impl Into<MatrixView<'a, S>>,
        rank_options: MultivariateNormalOptions<S>,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let covariance = covariance.into();
        if covariance.rows() != mean.len() || covariance.cols() != mean.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(mean.len(), mean.len()),
                scioxide_core::Dimensions::two(covariance.rows(), covariance.cols()),
            ));
        }
        options.validate()?;
        options.admit::<S>(mean.len())?;
        options.admit::<S>(
            covariance
                .rows()
                .checked_mul(covariance.cols())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("gaussian_parameters"))
                })?,
        )?;
        Self::from_owned(
            scioxide_linalg::VectorView::from_slice(mean).try_to_vector()?,
            &covariance.try_to_matrix()?,
            rank_options,
            options,
        )
    }

    /// Builds the law from an owned mean and covariance.
    pub(crate) fn from_owned(
        mean: Vector<S>,
        covariance: &Matrix<S>,
        rank_options: MultivariateNormalOptions<S>,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let d = mean.len();
        let square = d.checked_mul(d).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("gaussian_preparation"))
        })?;
        options.admit::<S>(square)?;
        options.steps(square.checked_mul(d).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("gaussian_preparation"))
        })?)?;
        if d == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("mean"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "MultivariateNormal requires dimension > 0",
            ));
        }
        if covariance.rows() != d || covariance.cols() != d {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(d, d),
                scioxide_core::Dimensions::two(covariance.rows(), covariance.cols()),
            ));
        }
        for &value in mean.as_slice().iter().chain(covariance.as_slice()) {
            crate::law::argument(value, "mean or covariance")?;
        }
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        for row in 0..d {
            if covariance.as_slice()[row * d + row] < zero {
                return Err(invalid_covariance());
            }
            for col in 0..row {
                if covariance.as_slice()[row * d + col] != covariance.as_slice()[col * d + row] {
                    return Err(invalid_covariance());
                }
            }
        }
        // A symmetric eigendecomposition (not Cholesky) is used so that
        // semidefinite covariances are supported: the retained eigenpairs span
        // the support, and sqrt(eigenvalue) gives the per-direction deviation.
        // The matrix is first scaled to unit max-norm so extreme magnitudes
        // neither overflow nor underflow inside the eigen solver; the scale is
        // reapplied analytically to deviations and the log-determinant.
        let scale = covariance
            .as_slice()
            .iter()
            .fold(zero, |s, x| s.max(scioxide_core::NumericScalar::abs(*x)));
        let mut normalized = Matrix::zeros(d, d)?;
        if scale != zero {
            for (out, &value) in normalized
                .as_mut_slice()
                .iter_mut()
                .zip(covariance.as_slice())
            {
                *out = value / scale;
                if value != zero && *out == zero {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Underflow,
                        "covariance entry lost during normalization",
                    ));
                }
            }
        }
        let decomposition = HermitianEigenDecomposition::decompose_with_options(
            &normalized,
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::HermitianEigenOptions::default().with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new()
                    .with_iteration_budget(options.work_budget()),
            ),
        )?;
        let eigenvalues = decomposition.eigenvalues();
        let largest = eigenvalues.iter().copied().fold(zero, RealScalar::max);
        // Backward-stable eigen solvers produce eigenvalue errors of order
        // d eps ||C||, so slightly negative eigenvalues are rounding noise and
        // are tolerated (then dropped by the rank cutoff); anything more
        // negative means the input is not positive semidefinite.
        let relative_roundoff = S::try_from_usize_exact(d)? * S::EPSILON;
        let negative_allowance = S::try_from_f64_lossy(8.0)? * relative_roundoff * largest;
        if eigenvalues.iter().any(|&value| value < -negative_allowance) {
            return Err(invalid_covariance());
        }
        let cutoff = match rank_options.rank_policy {
            RankPolicy::Exact => zero,
            RankPolicy::DimensionScaledEpsilon => relative_roundoff * largest,
            RankPolicy::Threshold(threshold) => {
                // Overflow means the absolute threshold exceeds the spectrum.
                let absolute = if scale == zero {
                    zero
                } else {
                    threshold.absolute() / scale
                };
                absolute + threshold.relative() * largest
            }
        };
        let rank = eigenvalues.iter().filter(|&&value| value > cutoff).count();
        let mut support = Matrix::zeros(d, rank)?;
        let mut deviations = scioxide_core::backend::try_with_capacity(
            rank,
            scioxide_core::quantity!("gaussian_spectrum"),
        )?;
        let mut log_determinant = RangeAccumulator::new();
        for (column, (source, &value)) in eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > cutoff)
            .enumerate()
        {
            let deviation = RealScalar::sqrt(scale) * RealScalar::sqrt(value);
            if deviation == zero {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "retained standard deviation is unrepresentable",
                ));
            }
            deviations.push(deviation);
            log_determinant.add(RealScalar::ln(scale))?;
            log_determinant.add(RealScalar::ln(value))?;
            for row in 0..d {
                support.as_mut_slice()[row * rank + column] =
                    decomposition.eigenvectors().as_slice()[row * d + source];
            }
        }
        let log_pseudo_determinant = log_determinant.finish()?;
        let mut normalizer = RangeAccumulator::new();
        normalizer.add_product(-S::HALF, log_pseudo_determinant)?;
        normalizer.add_triple_product(
            -S::HALF,
            S::try_from_usize_exact(rank)?,
            RealScalar::ln(S::TAU),
        )?;
        let mut factor = Matrix::zeros(d, rank)?;
        let mut ordinary_factor = true;
        for (i, coefficient) in factor.as_mut_slice().iter_mut().enumerate() {
            let direction = support.as_slice()[i];
            *coefficient = direction * deviations[i % rank];
            ordinary_factor &= coefficient.is_normal() || direction == zero;
        }
        Ok(Self {
            mean,
            support,
            deviations,
            factor,
            ordinary_factor,
            log_normalizer: normalizer.finish()?,
            roundoff: S::try_from_f64_lossy(8.0)? * relative_roundoff,
            rank_options,
            options,
            log_pseudo_determinant,
            standard_normal: Normal::standard(),
            dim: d,
        })
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> MultivariateNormal<S> {
    /// Returns the numerical rank `r` of the covariance (zero for a point mass).
    pub fn rank(&self) -> usize {
        self.deviations.len()
    }

    pub(crate) fn log_determinant_on_support(&self) -> S {
        self.log_pseudo_determinant
    }

    pub(crate) fn log_squared_distance(
        &self,
        point: &[S],
        tolerance: Tolerance<S>,
    ) -> Result<Option<S>, SciOxideError> {
        // ln of the Mahalanobis distance sum (c_j / sigma_j)^2 on the support,
        // accumulated with `logadd` so that far points (as used by the t law)
        // do not overflow the squared distance.
        let (offset, coordinates, scale) = self.project(point)?;
        if !self.contains_projection(&offset, &coordinates, scale, tolerance)? {
            return Ok(None);
        }
        let mut log = S::NEG_INFINITY;
        for (&coordinate, &deviation) in coordinates.iter().zip(&self.deviations) {
            if coordinate != <S as scioxide_core::AlgebraicScalar>::ZERO
                && scale != <S as scioxide_core::AlgebraicScalar>::ZERO
            {
                let term = S::TWO
                    * (RealScalar::ln(NumericScalar::abs(coordinate)) + RealScalar::ln(scale)
                        - RealScalar::ln(deviation));
                log = crate::law::logadd(log, term)?;
            }
        }
        Ok(Some(log))
    }

    /// Returns the rank-`r` covariance `V Λ Vᵀ` in O(d² r) time.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` or `AllocationFailure` if the `d × d` result cannot be stored.
    /// - `NumericalInstability` if an entry overflows.
    pub fn covariance(&self) -> Result<Matrix<S>, SciOxideError> {
        let factor = self.support_factor()?;
        factor
            .view()
            .mul_mat(&factor.view().transpose().try_to_matrix()?)
    }

    pub(crate) fn support_factor(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut factor = Matrix::zeros(self.dim, self.rank())?;
        for i in 0..self.dim {
            for j in 0..self.rank() {
                factor.view_mut().set(
                    i,
                    j,
                    scioxide_core::checked_float_product(
                        self.support.view().get(i, j)?,
                        self.deviations[j],
                    )?,
                )?;
            }
        }
        Ok(factor)
    }

    /// Returns the orthonormal support directions `V` as columns of a `d × r` matrix.
    pub const fn support_basis(&self) -> &Matrix<S> {
        &self.support
    }

    /// Returns the [`LawOptions`](crate::LawOptions) given at construction.
    pub const fn options(&self) -> crate::LawOptions<S> {
        self.options
    }

    fn joint_options(&self) -> Result<MultivariateProbabilityOptions, SciOxideError> {
        super::probability::law_probability_options(self.options)
    }

    /// Returns the rank options given at construction.
    pub const fn rank_options(&self) -> MultivariateNormalOptions<S> {
        self.rank_options
    }

    /// Tests whether `point` lies on the support `μ + span(V)`.
    ///
    /// The point is on the support when the residual of its projection onto the
    /// support satisfies `‖(I − V Vᵀ)(x − μ)‖∞ ≤ abs_tol + rel_tol ‖x − μ‖∞`. The test
    /// costs O(d r). For a singular law it allocates O(d + r) scratch.
    /// [`Self::evaluate_support`] uses a workspace instead. A full-rank law
    /// contains every finite point of dimension `d`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point.len() ≠ d`.
    /// - `NonFinite` if `point` has a NaN or infinite entry.
    /// - `AllocationFailure` if the scratch cannot be allocated.
    pub fn contains_support(
        &self,
        point: &[S],
        tolerance: Tolerance<S>,
    ) -> Result<bool, SciOxideError> {
        self.validate_output_dimension(point.len())?;
        if self.rank() == self.dim {
            for &value in point {
                crate::law::argument(value, "point")?;
            }
            return Ok(true);
        }
        let (offset, coordinates, scale) = self.project(point)?;
        self.contains_projection(&offset, &coordinates, scale, tolerance)
    }

    /// Computes the log pseudo-density, the log density with respect to
    /// `r`-dimensional Lebesgue measure on the support.
    ///
    /// ```text
    /// log p(x) = −(r/2) ln(2π) − ½ Σᵢ ln λᵢ − ½ ‖Λ^(−1/2) Vᵀ (x − μ)‖²
    /// ```
    ///
    /// where `λᵢ` are the `r` nonzero eigenvalues. Points off the support (as
    /// decided by `tolerance`, see [`Self::contains_support`]) give `−∞`. A
    /// point on the support is projected onto it before evaluation. For rank
    /// zero the value is 0 on the support. The cost is O(d r). For a singular
    /// law this allocates O(d + r) scratch; [`Self::evaluate_log_pseudo_density`]
    /// uses a workspace instead.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ d`.
    /// - `NonFinite` if `x` has a NaN or infinite entry.
    /// - `NumericalInstability` if a finite log pseudo-density overflows.
    pub fn log_pseudo_density(
        &self,
        point: &[S],
        tolerance: Tolerance<S>,
    ) -> Result<S, SciOxideError> {
        if self.rank() == self.dim {
            return self.full_rank_density(point);
        }
        self.validate_point(point)?;
        let mut workspace = self.workspace(
            crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Incremental),
            1,
        )?;
        self.evaluate_log_pseudo_density(point, tolerance, &mut workspace)
    }

    /// Tests support membership as [`Self::contains_support`], using scratch in `workspace`.
    ///
    /// A singular law needs `d + r` elements, the size of a one-sample workspace
    /// from [`Self::workspace`]. The workspace is not grown. A full-rank law
    /// needs no scratch.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::contains_support`].
    /// - `DimensionMismatch` if `workspace` is smaller than a one-sample workspace.
    pub fn evaluate_support(
        &self,
        point: &[S],
        tolerance: Tolerance<S>,
        workspace: &mut SamplingWorkspace<S>,
    ) -> Result<bool, SciOxideError> {
        if self.rank() == self.dim {
            return self.contains_support(point, tolerance);
        }
        let (offset, coordinates, scale) = self.project_workspace(point, workspace)?;
        self.contains_projection(offset, coordinates, scale, tolerance)
    }

    /// Computes the log pseudo-density as [`Self::log_pseudo_density`], using
    /// scratch in `workspace`.
    ///
    /// The workspace needs `d + r` elements for a singular law and is not grown.
    /// Its contents may change, also on error. A workspace serves any law of
    /// suitable size.
    ///
    /// ```
    /// use scioxide_core::Tolerance;
    /// use scioxide_linalg::{Matrix, Vector};
    /// use scioxide_random::{MultivariateNormal, OutputPolicy, SamplingPolicy};
    /// let law = MultivariateNormal::new(&[0.0_f64; 2],
    ///     &Matrix::from_vec(2, 2, vec![1.0; 4])?)?;
    /// let mut workspace = law.workspace(SamplingPolicy::advance_on_error(OutputPolicy::Incremental), 1)?;
    /// let value = law.evaluate_log_pseudo_density(&[1.0, 1.0], Tolerance::from_machine_epsilon(), &mut workspace)?;
    /// assert!(value.is_finite());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    /// # Errors
    ///
    /// - The errors of [`Self::log_pseudo_density`].
    /// - `DimensionMismatch` if `workspace` is smaller than a one-sample workspace.
    pub fn evaluate_log_pseudo_density(
        &self,
        point: &[S],
        tolerance: Tolerance<S>,
        workspace: &mut SamplingWorkspace<S>,
    ) -> Result<S, SciOxideError> {
        if self.rank() == self.dim {
            return self.full_rank_density(point);
        }
        let (offset, coordinates, scale) = self.project_workspace(point, workspace)?;
        if !self.contains_projection(offset, coordinates, scale, tolerance)? {
            return Ok(S::NEG_INFINITY);
        }
        let mut result = self.density_accumulator()?;
        for (&coordinate, &deviation) in coordinates.iter().zip(&self.deviations) {
            let standardized = checked_product_quotient(coordinate, scale, deviation)?;
            result.add_triple_product(-S::HALF, standardized, standardized)?;
        }
        result.finish()
    }

    fn density_accumulator(&self) -> Result<RangeAccumulator<S>, SciOxideError> {
        let mut result = RangeAccumulator::new();
        result.add_product(-S::HALF, self.log_pseudo_determinant)?;
        result.add_triple_product(
            -S::HALF,
            S::try_from_usize_exact(self.rank())?,
            RealScalar::ln(S::TAU),
        )?;
        Ok(result)
    }

    fn full_rank_density(&self, point: &[S]) -> Result<S, SciOxideError> {
        self.validate_point(point)?;
        if let Some(value) = self.ordinary_density(point) {
            return Ok(value);
        }
        // The full-rank fallback streams each projection; support testing needs
        // no coordinates, so even extreme-scale log_pdf calls need no heap.
        let (halved, scale) = self.displacement(point)?;
        let mut result = self.density_accumulator()?;
        for (column, &deviation) in self.deviations.iter().enumerate() {
            let mut coordinate = RangeAccumulator::new();
            for (row, (&x, &mean)) in point.iter().zip(self.mean.as_slice()).enumerate() {
                coordinate.add_product(
                    normalized_offset(x, mean, halved, scale)?,
                    self.support.as_slice()[row * self.rank() + column],
                )?;
            }
            let standardized = checked_product_quotient(coordinate.finish()?, scale, deviation)?;
            result.add_triple_product(-S::HALF, standardized, standardized)?;
        }
        result.finish()
    }

    fn ordinary_density(&self, point: &[S]) -> Option<S> {
        let terms =
            self.deviations
                .iter()
                .enumerate()
                .map(|(column, &deviation)| {
                    let coordinate = ordinary_sum(
                        S::ZERO,
                        point.iter().zip(self.mean.as_slice()).enumerate().map(
                            |(row, (&x, &mean))| {
                                let delta = x - mean;
                                if !delta.is_normal() && delta != S::ZERO {
                                    return None;
                                }
                                ordinary_product(
                                    delta,
                                    self.support.as_slice()[row * self.rank() + column],
                                )
                            },
                        ),
                        self.roundoff,
                    )?;
                    let standardized = coordinate / deviation;
                    if !standardized.is_normal() && coordinate != S::ZERO {
                        return None;
                    }
                    ordinary_product(-S::HALF * standardized, standardized)
                });
        ordinary_sum(self.log_normalizer, terms, self.roundoff)
    }

    fn project_workspace<'a>(
        &self,
        point: &[S],
        workspace: &'a mut SamplingWorkspace<S>,
    ) -> Result<(&'a [S], &'a [S], S), SciOxideError> {
        self.validate_output_dimension(point.len())?;
        let requirements = self.workspace_requirements(
            crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Incremental),
            1,
        )?;
        if workspace.check(&requirements)? == scioxide_core::Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(requirements.elements()),
                scioxide_core::Dimensions::one(workspace.as_slice().len()),
            ));
        }
        workspace.stage(&requirements, S::ZERO)?;
        let (offset, coordinates) = workspace.as_mut_slice().split_at_mut(self.dim);
        let scale = self.project_into(point, offset, coordinates)?;
        Ok((offset, coordinates, scale))
    }

    fn project(&self, point: &[S]) -> Result<(Vec<S>, Vec<S>, S), SciOxideError> {
        self.validate_point(point)?;
        let mut offset = scioxide_core::backend::try_filled(
            self.dim,
            S::ZERO,
            scioxide_core::quantity!("gaussian_offset"),
        )?;
        let mut coordinates = scioxide_core::backend::try_filled(
            self.rank(),
            S::ZERO,
            scioxide_core::quantity!("gaussian_projection"),
        )?;
        let scale = self.project_into(point, &mut offset, &mut coordinates)?;
        Ok((offset, coordinates, scale))
    }

    fn displacement(&self, point: &[S]) -> Result<(bool, S), SciOxideError> {
        for &value in point {
            crate::law::argument(value, "point")?;
        }
        // Subtract first to preserve small displacements from large means.
        let halved = point
            .iter()
            .zip(self.mean.as_slice())
            .any(|(&x, &mean)| !(x - mean).is_finite());
        let mut scale = S::ZERO;
        for (&x, &mean) in point.iter().zip(self.mean.as_slice()) {
            let delta = if halved {
                x * S::HALF - mean * S::HALF
            } else {
                x - mean
            };
            if delta == S::ZERO && x != mean {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "displacement lost during halving",
                ));
            }
            scale = scale.max(NumericScalar::abs(delta));
        }
        Ok((halved, scale))
    }

    fn project_into(
        &self,
        point: &[S],
        offset: &mut [S],
        coordinates: &mut [S],
    ) -> Result<S, SciOxideError> {
        let (halved, scale) = self.displacement(point)?;
        for ((value, &x), &mean) in offset.iter_mut().zip(point).zip(self.mean.as_slice()) {
            *value = normalized_offset(x, mean, halved, scale)?;
        }
        for (column, coordinate) in coordinates.iter_mut().enumerate() {
            let mut sum = RangeAccumulator::new();
            for (row, &value) in offset.iter().enumerate() {
                sum.add_product(value, self.support.as_slice()[row * self.rank() + column])?;
            }
            *coordinate = sum.finish()?;
        }
        Ok(scale)
    }

    fn contains_projection(
        &self,
        offset: &[S],
        coordinates: &[S],
        scale: S,
        tolerance: Tolerance<S>,
    ) -> Result<bool, SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        if self.rank() == self.dim || scale == zero {
            return Ok(true);
        }
        // A point is on the (affine) support when the residual of projecting
        // its offset onto the retained eigenvectors is within tolerance. The
        // offset is normalized by `scale`, so the absolute tolerance is
        // rescaled the same way.
        let magnitude = offset
            .iter()
            .fold(zero, |m, x| m.max(scioxide_core::NumericScalar::abs(*x)));
        let bound = tolerance.absolute() / scale + tolerance.relative() * magnitude;
        for (row, &value) in offset.iter().enumerate() {
            let mut residual = RangeAccumulator::new();
            residual.add(value)?;
            for (col, &coordinate) in coordinates.iter().enumerate() {
                residual.subtract_product(
                    self.support.as_slice()[row * self.rank() + col],
                    coordinate,
                )?;
            }
            if scioxide_core::NumericScalar::abs(residual.finish()?) > bound {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns the dimension `d`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dim
    }

    /// Returns the mean `μ`, the location of the law.
    pub fn location(&self) -> &[S] {
        self.mean.as_slice()
    }

    fn validate_output_dimension(&self, dimension: usize) -> Result<(), SciOxideError> {
        if dimension != self.dim {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dim),
                scioxide_core::Dimensions::one(dimension),
            ));
        }
        Ok(())
    }

    fn validate_point(&self, point: &[S]) -> Result<(), SciOxideError> {
        self.validate_output_dimension(point.len())?;
        point
            .iter()
            .try_for_each(|&value| crate::law::argument(value, "point"))
    }

    pub(crate) fn transform(&self, latent: &[S], sample: &mut [S]) -> Result<(), SciOxideError> {
        if self.rank() == 0 {
            sample.copy_from_slice(self.mean.as_slice());
            return Ok(());
        }
        // x = mu + V diag(sigma) z over the retained eigenpairs. Ordinary rows
        // use cached coefficients; range/cancellation fallbacks retain exact
        // triples so large means and deviations can cancel before rounding.
        for (row, value) in sample.iter_mut().enumerate() {
            if self.ordinary_factor {
                let coefficients =
                    &self.factor.as_slice()[row * self.rank()..(row + 1) * self.rank()];
                if let Some(result) = ordinary_sum(
                    self.mean.as_slice()[row],
                    coefficients
                        .iter()
                        .zip(latent)
                        .map(|(&a, &z)| ordinary_product(a, z)),
                    self.roundoff,
                ) {
                    *value = result;
                    continue;
                }
            }
            let mut sum = RangeAccumulator::new();
            sum.add(self.mean.as_slice()[row])?;
            for (col, (&z, &deviation)) in latent.iter().zip(&self.deviations).enumerate() {
                let direction = self.support.as_slice()[row * self.rank() + col];
                // Exact identities avoid redundant triple-product work for coordinate supports.
                if direction == <S as scioxide_core::AlgebraicScalar>::ZERO {
                    continue;
                }
                if direction == <S as scioxide_core::AlgebraicScalar>::ONE {
                    sum.add_product(deviation, z)?;
                } else if direction == -<S as scioxide_core::AlgebraicScalar>::ONE {
                    sum.subtract_product(deviation, z)?;
                } else {
                    sum.add_triple_product(direction, deviation, z)?;
                }
            }
            *value = sum.finish()?;
        }
        Ok(())
    }
}

// Ordinary reductions retain a Neumaier correction. Subnormal products,
// overflow and cancellation at the precision-scaled error bound use the exact
// kernel instead, including the original unrounded factor products.
#[inline]
fn ordinary_product<S: RealScalar>(a: S, b: S) -> Option<S> {
    let product = a * b;
    (product.is_normal() || (product == S::ZERO && (a == S::ZERO || b == S::ZERO)))
        .then_some(product)
}

#[inline]
fn ordinary_sum<S: RealScalar>(
    initial: S,
    terms: impl Iterator<Item = Option<S>>,
    roundoff: S,
) -> Option<S> {
    let mut sum = initial;
    let mut correction = S::ZERO;
    let mut magnitude = NumericScalar::abs(initial);
    for term in terms {
        let term = term?;
        let next = sum + term;
        correction += if NumericScalar::abs(sum) >= NumericScalar::abs(term) {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
        magnitude += NumericScalar::abs(term);
    }
    let result = sum + correction;
    (result.is_finite()
        && magnitude.is_finite()
        && (magnitude == S::ZERO || NumericScalar::abs(result) > roundoff * magnitude))
        .then_some(result)
}

fn normalized_offset<S: RealScalar>(
    x: S,
    mean: S,
    halved: bool,
    scale: S,
) -> Result<S, SciOxideError> {
    let delta = if halved {
        x * S::HALF - mean * S::HALF
    } else {
        x - mean
    };
    if scale == S::ZERO {
        return Ok(delta);
    }
    let value = delta / scale / if halved { S::HALF } else { S::ONE };
    if delta != S::ZERO && value == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "displacement lost during normalization",
        ));
    }
    Ok(value)
}

fn invalid_covariance() -> SciOxideError {
    crate::law::domain(
        scioxide_core::argument!("covariance"),
        "covariance must be exactly symmetric and positive semidefinite",
    )
}

impl<S: RealScalar + NumericScalar<Real = S>> crate::LogDensity<S, &[S]> for MultivariateNormal<S> {
    fn log_pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        if self.rank() != self.dimension() {
            return Err(crate::law::invalid(
                scioxide_core::argument!("self"),
                "singular Gaussian requires log_pseudo_density",
            ));
        }
        self.log_pseudo_density(x, Tolerance::from_machine_epsilon())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::Density<S, &[S]> for MultivariateNormal<S> {
    fn pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        crate::law::finite(RealScalar::exp(crate::LogDensity::log_pdf(self, x)?))
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::Mean<S, Vec<S>> for MultivariateNormal<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        let mut out = scioxide_core::backend::try_with_capacity(
            self.dimension(),
            scioxide_core::quantity!("gaussian_mean"),
        )?;
        out.extend_from_slice(self.mean.as_slice());
        Ok(out)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::Variance<S, Vec<S>> for MultivariateNormal<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        let c = self.covariance()?;
        let mut out = scioxide_core::backend::try_with_capacity(
            self.dimension(),
            scioxide_core::quantity!("gaussian_variance"),
        )?;
        for i in 0..self.dimension() {
            out.push(c.view().get(i, i)?);
        }
        Ok(out)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> MultivariateNormal<S> {
    /// Creates a plan that computes probabilities `P(a < X ≤ b)` of boxes.
    ///
    /// The plan integrates by randomized quasi-Monte Carlo with
    /// randomizations drawn from `root`. See [`MultivariateProbabilityPlan`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `options` has fewer than two replicates or a point
    ///   count per replicate that is not a power of two.
    /// - `ConvergenceFailure` if the total point count exceeds the evaluation limit of `options`.
    /// - `ResourceLimit` if the plan storage exceeds the byte limit of `options`.
    pub fn prepare_probability(
        &self,
        options: MultivariateProbabilityOptions,
        root: &Philox4x64,
    ) -> Result<MultivariateProbabilityPlan, SciOxideError> {
        preflight(
            self.dimension(),
            self.rank(),
            super::DegreesOfFreedom::Gaussian,
            options,
        )?;
        MultivariateProbabilityPlan::prepare(
            values(self.location())?,
            super::matrix::matrix64(self.support_factor()?.view())?,
            super::DegreesOfFreedom::Gaussian,
            options,
            root,
        )
    }
}

joint_traits!(
    MultivariateNormal,
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Tolerance;
    /// use scioxide_linalg::Matrix;
    /// use scioxide_random::{Cdf, LawOptions, MultivariateNormal, Philox4x64, SeedableRng};
    /// let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0])?;
    /// let options = LawOptions::default().with_tolerance(Tolerance::new(1e-5, 1e-5)?);
    /// let law = MultivariateNormal::with_options(&[0.0; 2], &covariance, Default::default(), options)?;
    /// let root = Philox4x64::from_seed(0);
    /// // Orthant probability 1/4 + asin(rho) / (2 pi) = 1/3 for rho = 1/2.
    /// let orthant = law.rectangle(&[f64::NEG_INFINITY; 2], &[0.0; 2], &root)?;
    /// assert!((orthant - 1.0 / 3.0).abs() < 1e-5);
    /// assert_eq!(orthant, law.cdf(&[0.0; 2])?);
    /// let band = law.rectangle(&[-1.0, f64::NEG_INFINITY], &[1.0, f64::INFINITY], &root)?;
    /// assert!((band - 0.682_689_492_137_086).abs() < 1e-5); // marginal P(-1 < X₁ ≤ 1)
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ;
    +NumericScalar<Real=S>
);
// Rank latent normals are item scratch; a draw is the transform of fresh
// latent normals decoded from the law's charged uniform words.
impl<S: RealScalar + NumericScalar<Real = S>> RowLaw<S, S> for MultivariateNormal<S> {
    fn width(&self) -> usize {
        self.dim
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn scratch(&self) -> (usize, usize) {
        (self.rank(), 0)
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        scratch: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_word_budget().get());
        for value in scratch.items.iter_mut() {
            *value = self.standard_normal.sample(&mut words)?;
        }
        self.transform(scratch.items, out)
    }
}
vector_sampling!(
    MultivariateNormal,
    S,
    <S as scioxide_core::AlgebraicScalar>::ZERO,
    NumericScalar<Real = S>
);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::{BitGenerator, Rng, SeedableRng};
    use crate::workspace::SamplingWorkspace;

    #[test]
    fn transform_includes_mean_before_rounding_overflowing_products() {
        macro_rules! check {
            ($scalar:ty, $cov:expr, $z:expr, $mean:expr) => {{
                let law =
                    MultivariateNormal::new(&[$mean], &Matrix::from_vec(1, 1, vec![$cov]).unwrap())
                        .unwrap();
                let mut output = [0.0 as $scalar];
                law.transform(&[$z], &mut output).unwrap();
                let expected =
                    scioxide_core::checked_mul_add(law.deviations[0], $z, $mean).unwrap();
                assert!((law.deviations[0] * $z).is_infinite());
                assert_eq!(output[0], expected);
            }};
        }
        check!(f32, 3.0e38_f32, 3.0e19_f32, -3.0e38_f32);
        check!(f64, 1.0e308_f64, 2.0e154_f64, -1.0e308_f64);
    }

    #[test]
    fn density_preserves_small_displacement_from_large_mean() {
        let mean = 2.0_f64.powi(100);
        let delta = 2.0_f64.powi(48);
        let law = MultivariateNormal::new(
            &[mean],
            &Matrix::from_vec(1, 1, vec![delta * delta]).unwrap(),
        )
        .unwrap();
        let actual = law
            .log_pseudo_density(&[mean + delta], Tolerance::from_machine_epsilon())
            .unwrap();
        let expected =
            -0.5 * (crate::distribution::special::log_two_pi::<f64>() + 96.0 * 2.0_f64.ln() + 1.0);
        assert!((actual - expected).abs() < 2.0e-14);
    }

    fn identity_2d() -> Matrix<f64> {
        Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap()
    }

    fn diagonal_cov() -> Matrix<f64> {
        Matrix::from_vec(2, 2, vec![4.0, 0.0, 0.0, 9.0]).unwrap()
    }

    struct FirstThenRejectRng {
        raw_draws: usize,
    }

    impl BitGenerator for FirstThenRejectRng {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(self.next_u64().unwrap() as u32)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.raw_draws += 1;
            Ok(if self.raw_draws == 1 {
                0
            } else {
                (((1_u64 << 52) - 1) << 9) | 1
            })
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl Rng<f64> for FirstThenRejectRng {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            Ok(1.0_f64.next_down())
        }
    }

    #[test]
    fn test_new_valid() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let cov = identity_2d();
        assert!(MultivariateNormal::new(mean.as_slice(), &cov).is_ok());
    }

    #[test]
    fn test_new_zero_dimension() {
        let mean = Vector::<f64>::from_vec(vec![]);
        let cov = Matrix::zeros(0, 0).unwrap();
        let error = MultivariateNormal::new(mean.as_slice(), &cov).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_new_dimension_mismatch() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0, 0.0]);
        let cov = identity_2d();
        let error = MultivariateNormal::new(mean.as_slice(), &cov).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn test_new_not_positive_definite() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let cov = Matrix::from_vec(2, 2, vec![-1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(MultivariateNormal::new(mean.as_slice(), &cov).is_err());
    }

    #[test]
    fn test_new_non_finite_entries() {
        let mean = Vector::from_vec(vec![f64::NAN, 0.0_f64]);
        let cov = identity_2d();
        let error = MultivariateNormal::new(mean.as_slice(), &cov).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_dim() {
        let mean = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let cov =
            Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();
        assert_eq!(dist.dimension(), 3);
    }

    #[test]
    fn test_sample_dimension() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let cov = identity_2d();
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let x = dist.sample(&mut rng).unwrap();
        assert_eq!(x.len(), 2);
    }

    #[test]
    fn test_sample_preserves_coordinatewise_normal_sequence() {
        let mean = Vector::from_vec(vec![1.0_f64, -2.0]);
        let cov = diagonal_cov();
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();

        let standard = Normal::new(0.0_f64, 1.0).unwrap();
        let mut expected_rng = SplitMix64::from_seed(17);
        let expected = [
            1.0 + 2.0 * standard.sample(&mut expected_rng).unwrap(),
            -2.0 + 3.0 * standard.sample(&mut expected_rng).unwrap(),
        ];

        let mut actual_rng = SplitMix64::from_seed(17);
        let actual = dist.sample(&mut actual_rng).unwrap();
        assert_eq!(actual.as_slice(), expected.as_slice());
        assert_eq!(actual_rng.position(), expected_rng.position());
    }

    #[test]
    fn test_workspace_and_bulk_paths_match_repeated_samples() {
        let mean = Vector::from_vec(vec![1.0_f64, -2.0]);
        let cov = Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 2.0]).unwrap();
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();
        let policy = crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic);

        let mut allocating_rng = SplitMix64::from_seed(23);
        let expected: Vec<f64> = (0..5)
            .flat_map(|_| dist.sample(&mut allocating_rng).unwrap())
            .collect();

        let mut buffered_rng = SplitMix64::from_seed(23);
        let mut workspace = SamplingWorkspace::new();
        let mut output = [0.0; 2];
        for expected_sample in expected.chunks_exact(2) {
            dist.sample_n_into(&mut buffered_rng, policy, &mut output, &mut workspace)
                .unwrap();
            assert_eq!(&output, expected_sample);
        }
        assert_eq!(buffered_rng.position(), allocating_rng.position());

        let mut bulk_rng = SplitMix64::from_seed(23);
        let mut bulk = vec![0.0; expected.len()];
        dist.sample_n_into(&mut bulk_rng, policy, &mut bulk, &mut workspace)
            .unwrap();
        assert_eq!(bulk, expected);
        assert_eq!(bulk_rng.position(), allocating_rng.position());
    }

    #[test]
    fn test_incomplete_row_error_is_transactional() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let dist = MultivariateNormal::new(mean.as_slice(), &identity_2d()).unwrap();
        let mut workspace = SamplingWorkspace::new();
        let mut output = [7.0, 8.0, 9.0];
        let mut rng = SplitMix64::from_seed(29);
        let position = rng.position();

        let error = dist
            .sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(output, [7.0, 8.0, 9.0]);
        assert_eq!(rng.position(), position);
    }

    #[test]
    fn test_indexed_incomplete_row_is_dimension_mismatch() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let dist = MultivariateNormal::new(mean.as_slice(), &identity_2d()).unwrap();
        let mut output = [7.0, 8.0, 9.0];
        let error = dist
            .sample_indexed_into(
                &crate::Pcg64::from_seed(29),
                &mut output,
                &mut SamplingWorkspace::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(output, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_workspace_is_reusable_across_dimensions() {
        let mean_2d = Vector::from_vec(vec![0.0_f64, 0.0]);
        let dist_2d = MultivariateNormal::new(mean_2d.as_slice(), &identity_2d()).unwrap();
        let mean_3d = Vector::from_vec(vec![0.0_f64; 3]);
        let cov_3d = Matrix::identity(3).unwrap();
        let dist_3d = MultivariateNormal::new(mean_3d.as_slice(), &cov_3d).unwrap();
        let policy = crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic);
        let mut workspace = SamplingWorkspace::new();
        let mut rng = SplitMix64::from_seed(31);
        dist_3d
            .sample_n_into(&mut rng, policy, &mut [0.0; 3], &mut workspace)
            .unwrap();
        let mut output = [7.0, 8.0];
        dist_2d
            .sample_n_into(&mut rng, policy, &mut output, &mut workspace)
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_transform_failure_after_sampling_preserves_output() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let dist = MultivariateNormal::new(mean.as_slice(), &identity_2d()).unwrap();
        let mut rng = FirstThenRejectRng { raw_draws: 0 };
        let mut output = [7.0, 8.0];
        let mut workspace = SamplingWorkspace::new();

        let error = dist
            .sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(output, [7.0, 8.0]);
    }

    #[test]
    fn test_empty_bulk_sample_does_not_advance_rng() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let dist = MultivariateNormal::new(mean.as_slice(), &identity_2d()).unwrap();
        let mut workspace = SamplingWorkspace::new();
        let mut rng = SplitMix64::from_seed(41);
        let position = rng.position();

        dist.sample_n_into(
            &mut rng,
            crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
            &mut [],
            &mut workspace,
        )
        .unwrap();
        assert_eq!(rng.position(), position);
    }

    #[test]
    fn test_bulk_size_overflow_precedes_sampling() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let dist = MultivariateNormal::new(mean.as_slice(), &identity_2d()).unwrap();
        let policy = crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic);
        let error = dist.workspace_requirements(policy, usize::MAX).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::SizeOverflow);
    }

    #[test]
    fn test_scale_aware_covariance_policy_f64() {
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            let mean = Vector::from_vec(vec![0.0, 0.0]);
            let positive_definite = Matrix::from_vec(
                2,
                2,
                vec![4.0 * scale, 2.0 * scale, 2.0 * scale, 3.0 * scale],
            )
            .unwrap();
            assert!(MultivariateNormal::new(mean.as_slice(), &positive_definite).is_ok());

            let mean = Vector::from_vec(vec![0.0, 0.0]);
            let singular = Matrix::from_vec(2, 2, vec![scale, scale, scale, scale]).unwrap();
            assert_eq!(
                MultivariateNormal::new(mean.as_slice(), &singular)
                    .unwrap()
                    .rank(),
                1
            );
        }
    }

    #[test]
    fn test_scale_aware_covariance_policy_f32() {
        for scale in [2.0_f32.powi(-60), 1.0, 2.0_f32.powi(60)] {
            let mean = Vector::from_vec(vec![0.0, 0.0]);
            let positive_definite = Matrix::from_vec(
                2,
                2,
                vec![4.0 * scale, 2.0 * scale, 2.0 * scale, 3.0 * scale],
            )
            .unwrap();
            assert!(MultivariateNormal::new(mean.as_slice(), &positive_definite).is_ok());

            let mean = Vector::from_vec(vec![0.0, 0.0]);
            let singular = Matrix::from_vec(2, 2, vec![scale, scale, scale, scale]).unwrap();
            assert_eq!(
                MultivariateNormal::new(mean.as_slice(), &singular)
                    .unwrap()
                    .rank(),
                1
            );
        }
    }

    #[test]
    fn test_sample_mean_approx() {
        let mean = Vector::from_vec(vec![3.0_f64, -1.0]);
        let cov = identity_2d();
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();
        let mut rng = SplitMix64::from_seed(42);

        let n: i32 = 10_000;
        let mut sum = [0.0_f64; 2];
        for _ in 0..n {
            let x = dist.sample(&mut rng).unwrap();
            sum[0] += x[0];
            sum[1] += x[1];
        }

        let nf = f64::from(n);
        let mean_0 = sum[0] / nf;
        let mean_1 = sum[1] / nf;
        assert!((mean_0 - 3.0).abs() < 0.1, "mean[0]={mean_0}, expected 3.0");
        assert!(
            (mean_1 + 1.0).abs() < 0.1,
            "mean[1]={mean_1}, expected -1.0"
        );
    }

    #[test]
    fn test_sample_variance_approx() {
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let cov = diagonal_cov(); // var = [4, 9]
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();
        let mut rng = SplitMix64::from_seed(42);

        let n: i32 = 10_000;
        let mut sum = [0.0_f64; 2];
        let mut sum_sq = [0.0_f64; 2];
        for _ in 0..n {
            let x = dist.sample(&mut rng).unwrap();
            let x0 = x[0];
            let x1 = x[1];
            sum[0] += x0;
            sum[1] += x1;
            sum_sq[0] += x0 * x0;
            sum_sq[1] += x1 * x1;
        }

        let nf = f64::from(n);
        let mean_0 = sum[0] / nf;
        let mean_1 = sum[1] / nf;
        let var_0 = mean_0.mul_add(-mean_0, sum_sq[0] / nf);
        let var_1 = mean_1.mul_add(-mean_1, sum_sq[1] / nf);
        assert!((var_0 - 4.0).abs() < 0.5, "var[0]={var_0}, expected 4.0");
        assert!((var_1 - 9.0).abs() < 0.5, "var[1]={var_1}, expected 9.0");
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_sample_correlated() {
        // Covariance [[1, 0.8],[0.8, 1]] -> strongly correlated
        let mean = Vector::from_vec(vec![0.0_f64, 0.0]);
        let cov = Matrix::from_vec(2, 2, vec![1.0, 0.8, 0.8, 1.0]).unwrap();
        let dist = MultivariateNormal::new(mean.as_slice(), &cov).unwrap();
        let mut rng = SplitMix64::from_seed(42);

        let n: i32 = 10_000;
        let mut acc_xy = 0.0_f64;
        let mut acc_x = 0.0_f64;
        let mut acc_y = 0.0_f64;
        let mut acc_x2 = 0.0_f64;
        let mut acc_y2 = 0.0_f64;
        for _ in 0..n {
            let s = dist.sample(&mut rng).unwrap();
            let x = s[0];
            let y = s[1];
            acc_x += x;
            acc_y += y;
            acc_x2 += x * x;
            acc_y2 += y * y;
            acc_xy += x * y;
        }

        let nf = f64::from(n);
        let mean_x = acc_x / nf;
        let mean_y = acc_y / nf;
        let cov_xy = mean_x.mul_add(-mean_y, acc_xy / nf);
        let std_x = mean_x.mul_add(-mean_x, acc_x2 / nf).sqrt();
        let std_y = mean_y.mul_add(-mean_y, acc_y2 / nf).sqrt();
        let corr = cov_xy / (std_x * std_y);
        assert!(
            (corr - 0.8).abs() < 0.1,
            "correlation={corr}, expected ~0.8"
        );
    }
}
