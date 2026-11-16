//! Gaussian-process regression (simple and ordinary kriging) with a fixed
//! covariance.

use alloc::vec::Vec;

use crate::factor::{Factor, SystemKind};
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValueMut, Capacity, ExecutionContext, Interval, NdView, NumericScalar, ProblemToken,
    RealScalar, Scaled, SciOxideError, SolveIdentity, SolveScope, SolveState,
};
use scioxide_linalg::Matrix;
use scioxide_spatial::PointSet;

use super::PROPOSAL_DENIED;
use super::{
    LocalFitReport, LocalModelOptions, LocalPredictionWorkspace, Result, Selected, bounds,
    checked_factor, coefficients, evaluate_coefficients, intermediate, invalid, point, require,
    resolve, solve,
};
use crate::error::require_proposal;
use crate::geometry::{extended_distance_parts, norm};
use crate::protocol::{ExtrapolationPolicy, Interpolant};
use crate::radial::{corrected_decay, split_product, squared_decay};
use crate::validate::{
    Weights, validate_count, validate_finite, validate_length, validate_weights,
};

/// Selects the correlation function `ρ(r)` of a [`Covariance`].
///
/// All four are stationary and positive definite in every dimension, with
/// `ρ(0) = 1`. `r` is the distance scaled by the length scales, as defined on
/// [`Covariance`]. The Matérn kernel with smoothness `ν` gives sample paths
/// that are `⌈ν⌉ − 1` times differentiable; the squared exponential gives
/// infinitely differentiable paths.
///
/// ```
/// use scioxide_interp::{Covariance, CovarianceKernel};
/// let covariance = Covariance::new(CovarianceKernel::Matern12, 2.0_f64, [1.0])?;
/// assert!((covariance.covariance([0.0], [1.0])? - 2.0 * (-1.0_f64).exp()).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceKernel {
    /// Squared exponential `ρ(r) = exp(−r²/2)`.
    SquaredExponential,
    /// Matérn with `ν = 1/2`, the exponential `ρ(r) = exp(−r)`.
    Matern12,
    /// Matérn with `ν = 3/2`, `ρ(r) = (1 + √3 r) exp(−√3 r)`.
    Matern32,
    /// Matérn with `ν = 5/2`, `ρ(r) = (1 + √5 r + 5r²/3) exp(−√5 r)`.
    Matern52,
}

/// Defines the covariance function `k(a, b) = σ² ρ(r)` of a
/// [`GaussianProcessModel`].
///
/// `σ²` is the signal variance (a variance, not a standard deviation), `ρ` the
/// [`CovarianceKernel`], and
///
/// ```text
/// r = √(Σⱼ ((aⱼ − bⱼ)/ℓⱼ)²)
/// ```
///
/// the distance scaled by one length scale `ℓⱼ > 0` per axis. The parameters
/// are fixed; they are not estimated from the data.
///
/// ```
/// use scioxide_interp::{Covariance, CovarianceKernel};
/// let covariance = Covariance::new(CovarianceKernel::SquaredExponential, 1.5_f64, [0.5, 2.0])?;
/// assert_eq!(covariance.covariance([1.0, 1.0], [1.0, 1.0])?, 1.5);
/// assert_eq!(covariance.variance(), 1.5);
/// assert!(Covariance::new(CovarianceKernel::Matern32, 0.0_f64, [1.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Covariance<S: RealScalar, const D: usize> {
    kernel: CovarianceKernel,
    variance: S,
    length_scales: [S; D],
}

impl<S: RealScalar, const D: usize> Covariance<S, D> {
    /// Creates the covariance with signal variance `variance` and length scales
    /// `length_scales`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is zero, or `variance` or a length scale is not
    ///   positive.
    /// - `NonFinite` if `variance` or a length scale is NaN or infinite.
    pub fn new(kernel: CovarianceKernel, variance: S, length_scales: [S; D]) -> Result<Self> {
        if D == 0 {
            return Err(invalid(
                scioxide_core::argument!("length_scales"),
                "covariance needs at least one coordinate axis",
            ));
        }
        validate_weights(
            &[variance],
            1,
            Weights::Positive,
            scioxide_core::argument!("variance"),
        )?;
        validate_weights(
            &length_scales,
            D,
            Weights::Positive,
            scioxide_core::argument!("length_scales"),
        )?;
        Ok(Self {
            kernel,
            variance,
            length_scales,
        })
    }

    /// Returns the correlation function.
    pub const fn kernel(&self) -> CovarianceKernel {
        self.kernel
    }

    /// Returns the signal variance `σ²`.
    pub const fn variance(&self) -> S {
        self.variance
    }

    /// Returns the length scales `ℓⱼ`.
    pub const fn length_scales(&self) -> [S; D] {
        self.length_scales
    }

    /// Evaluates `k(a, b)`.
    ///
    /// The scaled distance and the product `σ² ρ(r)` are computed with an
    /// extended exponent range where needed, so the result is accurate whenever
    /// it is representable, even if `r` itself overflows. A tail below the
    /// smallest subnormal returns zero.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a coordinate of `a` or `b` is NaN or infinite.
    pub fn covariance(&self, a: [S; D], b: [S; D]) -> Result<S> {
        validate_finite(a.iter().chain(&b))?;
        self.covariance_validated(a, b)
    }

    /// Evaluates `k(a, b)` for coordinates already checked to be finite.
    #[inline]
    fn covariance_validated(&self, a: [S; D], b: [S; D]) -> Result<S> {
        // Anisotropic scaled distance r = ||(a - b) / length_scales||.
        let distance = norm(
            a.iter()
                .zip(b)
                .zip(self.length_scales)
                .map(|((&x, y), scale)| intermediate(intermediate(x - y)? / scale)),
        );
        if let Ok(r) = distance {
            if r == S::ZERO {
                return Ok(self.variance);
            }
            if let Some(value) = self.ordinary_covariance(r) {
                return Ok(value);
            }
        }
        self.extended_covariance(a, b)
    }

    /// Evaluates `k(a, b)` with an extended exponent range, outside the common
    /// path of matrix assembly.
    #[cold]
    fn extended_covariance(&self, a: [S; D], b: [S; D]) -> Result<S> {
        let [r, radius_correction] = extended_distance_parts(&a, &b, &self.length_scales)?.parts();
        let variance = Scaled::from_value(self.variance)?;
        match self.kernel {
            CovarianceKernel::SquaredExponential => {
                squared_decay(variance, r, radius_correction, true)
            }
            CovarianceKernel::Matern12 => corrected_decay(variance, r, radius_correction),
            CovarianceKernel::Matern32 => {
                let (t, correction) =
                    matern_argument(r, radius_correction, S::try_from_usize_exact(3)?)?;
                corrected_decay(variance.mul(Scaled::ONE.add(t)?)?, t, correction)
            }
            CovarianceKernel::Matern52 => {
                let three = Scaled::from_value(S::try_from_usize_exact(3)?)?;
                let (t, correction) =
                    matern_argument(r, radius_correction, S::try_from_usize_exact(5)?)?;
                let polynomial = Scaled::ONE.add(t)?.add(t.square()?.div(three)?)?;
                corrected_decay(variance.mul(polynomial)?, t, correction)
            }
        }
    }

    /// Evaluates `k` in plain floating-point arithmetic; `None` when the
    /// exponential or the product with `σ²` leaves the normal range.
    #[inline]
    fn ordinary_covariance(&self, r: S) -> Option<S> {
        let (polynomial, exponent) = match self.kernel {
            CovarianceKernel::SquaredExponential => (S::ONE, r * r / S::TWO),
            CovarianceKernel::Matern12 => (S::ONE, r),
            CovarianceKernel::Matern32 => {
                let t = S::try_from_usize_exact(3).ok()?.sqrt() * r;
                (S::ONE + t, t)
            }
            CovarianceKernel::Matern52 => {
                let t = S::try_from_usize_exact(5).ok()?.sqrt() * r;
                (S::ONE + t + t * t / S::try_from_usize_exact(3).ok()?, t)
            }
        };
        let tail = (-exponent).exp();
        let radial = polynomial * tail;
        let value = self.variance * radial;
        // With a normal exp(-t) and a polynomial >= 1, the radial factor
        // cannot underflow. Only the tail and final variance product need
        // admission checks; a subnormal normalized radius rounds exp(-r) to 1.
        (tail.is_normal() && value.is_normal()).then_some(value)
    }
}

/// Splits `√order · r` into a leading value and a first-order residual. Keeping
/// the residuals of the root and the product stops a rounded Matérn constant
/// from moving the point where a large `r` underflows to zero.
fn matern_argument<S: RealScalar>(
    r: Scaled<S>,
    correction: Scaled<S>,
    order: S,
) -> Result<(Scaled<S>, Scaled<S>)> {
    let root = order.sqrt();
    let root_correction = -root.mul_add(root, -order) / (S::TWO * root);
    let (product, residual) = split_product(Scaled::from_value(root)?, r)?;
    Ok((
        product,
        residual
            .add(Scaled::from_value(root_correction)?.mul(r)?)?
            .add(Scaled::from_value(root)?.mul(correction)?)?,
    ))
}

/// Selects the mean `μ` of a [`GaussianProcessModel`], for each real value
/// component separately.
///
/// ```
/// use scioxide_interp::KrigingMean;
/// let means = [0.5_f64, -1.0];
/// let simple = KrigingMean::Constant(&means);
/// assert!(matches!(simple, KrigingMean::Constant(m) if m.len() == 2));
/// assert!(matches!(
///     KrigingMean::<f64>::Ordinary,
///     KrigingMean::Ordinary
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KrigingMean<'a, S: RealScalar> {
    /// Uses a known constant mean per component (simple kriging).
    Constant(&'a [S]),
    /// Estimates an unknown constant mean per component by generalized least
    /// squares, `μ̂ = (1ᵀK⁻¹y)/(1ᵀK⁻¹1)` (ordinary kriging).
    Ordinary,
}

/// Conditions a Gaussian process with a fixed covariance on all samples
/// (kriging).
///
/// The samples `yᵢ` at points `xᵢ`, `i = 1, …, n`, are modeled as a Gaussian
/// process with mean `μ`, covariance `k` with signal variance `σ²` from
/// [`Covariance`], and independent observation noise of variance `sᵢ`. With `K = [k(xᵢ, xⱼ)] + diag(s)` and
/// `k*` the vector of `k(x, xᵢ)`, the prediction at `x` is
///
/// ```text
/// mean(x)   = μ + k*ᵀ K⁻¹ (y − μ1),
/// latent(x) = σ² − k*ᵀ K⁻¹ k* + (1 − 1ᵀK⁻¹k*)² / (1ᵀK⁻¹1),
/// ```
///
/// where the last term of the latent variance is present only for ordinary
/// kriging ([`KrigingMean::Ordinary`]). Each real value component has its own
/// mean but shares `K`. Use [`Self::prediction`] to evaluate the model and
/// [`Self::leave_one_out_into`] for leave-one-out cross-validation.
///
/// `K` is factored once by Cholesky in O(n³) time and O(n²) memory. If the
/// factor fails the rank or condition limit of [`LocalModelOptions`], an SVD
/// is used instead. No jitter is added to the diagonal. A prediction costs
/// O(n²) for the variance.
///
/// # Errors
///
/// [`Self::build`] lists the construction errors.
///
/// ```
/// use scioxide_interp::{
///     Covariance, CovarianceKernel, ExtrapolationPolicy, GaussianProcessModel, Interpolant,
///     KrigingMean, LocalSamples,
/// };
/// let samples = LocalSamples::new(&[[0.0_f64], [1.0], [2.0]], &[1.0, 0.0, 2.0])?;
/// let covariance = Covariance::new(CovarianceKernel::Matern32, 1.0, [1.0])?;
/// let points = samples.points()?;
/// let mean = KrigingMean::Ordinary;
/// let model = GaussianProcessModel::new(&points, samples.values()?, covariance, &[0.0; 3], mean)?;
/// // Components: mean, latent variance, observation variance.
/// let prediction = model.prediction(0.1)?;
/// let mut out = [0.0; 3];
/// let mut workspace = prediction.workspace()?;
/// prediction.evaluate_into([1.0], ExtrapolationPolicy::Reject, &mut out, &mut workspace)?;
/// assert!(out[0].abs() < 1e-10 && out[1].abs() < 1e-10 && (out[2] - 0.1).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
///   Learning*, MIT Press, 2006, chapters 2, 4 and 5.
/// - N. A. C. Cressie, *Statistics for Spatial Data*, rev. ed., Wiley, 1993,
///   chapter 3.
/// - O. Dubrule, "Cross validation of kriging in a unique neighborhood",
///   *Math. Geol.* 15(6), 1983, 687–699.
#[derive(Debug, Clone)]
#[must_use]
pub struct GaussianProcessModel<S: RealScalar + NumericScalar<Real = S>, const D: usize> {
    points: Vec<[S; D]>,
    values: Vec<S>,
    components: usize,
    covariance: Covariance<S, D>,
    factor: Factor<S>,
    coefficients: Vec<S>,
    means: Vec<S>,
    ordinary: bool,
    inverse_ones: Vec<S>,
    ones_sum: S,
    domains: [Interval<S>; D],
    report: LocalFitReport<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> GaussianProcessModel<S, D> {
    /// Builds the model on the calling thread with the default limits.
    ///
    /// This is [`Self::with_options`] with [`LocalModelOptions::default`]:
    /// [`RankThreshold::default`](scioxide_core::RankThreshold::default), a
    /// smallest reciprocal condition number of `√ε`, systems of order at most
    /// 1024 and no memory or budget limits.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn new(
        points: &PointSet<'_, S>,
        values: NdView<'_, '_, S>,
        covariance: Covariance<S, D>,
        noise: &[S],
        mean: KrigingMean<'_, S>,
    ) -> Result<Self> {
        Self::with_options(
            points,
            values,
            covariance,
            noise,
            mean,
            LocalModelOptions::default(),
        )
    }

    /// Builds the model on the calling thread with the rank, conditioning, size and
    /// work limits of `options`.
    ///
    /// Runs [`Self::build`] with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn with_options(
        points: &PointSet<'_, S>,
        values: NdView<'_, '_, S>,
        covariance: Covariance<S, D>,
        noise: &[S],
        mean: KrigingMean<'_, S>,
        options: LocalModelOptions<S>,
    ) -> Result<Self> {
        Self::build(
            points,
            values,
            covariance,
            noise,
            mean,
            options,
            ExecutionContext::sequential_native(),
        )
    }

    /// Builds the model from `points`, `values` and the noise variances
    /// `noise`, in `scope`.
    ///
    /// `values` is indexed `[sample, component]`, and `noise` holds one
    /// observation variance `sᵢ ≥ 0` per sample. `scope` is an
    /// [`ExecutionContext`] or the `&mut SolveContext` of an enclosing solve;
    /// the build counts as one proposal (see
    /// [solve accounting](scioxide_core#solve-accounting)). Shapes, finiteness and
    /// the byte limit are checked before any work starts.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `points` does not have `D` coordinates, `values` is
    ///   not two-dimensional, there are no samples, a noise variance is
    ///   negative, the order exceeds [`LocalModelOptions::max_system_order`],
    ///   or the minimum reciprocal condition number is outside `[0, 1]`.
    /// - `DimensionMismatch` if `values`, `noise` or the constant means of
    ///   [`KrigingMean::Constant`] have the wrong length.
    /// - `NonFinite` if a value, noise variance or mean is NaN or infinite.
    /// - `ResourceLimit` if the memory bound exceeds
    ///   [`LocalModelOptions::maximum_bytes`].
    /// - `DomainError` if the Cholesky factorization finds `K` not positive
    ///   definite.
    /// - `Singular` if `K` fails the rank or condition limit.
    /// - `ConvergenceFailure` with reason `ResourceLimit(Proposal)` if the
    ///   proposal limit is exhausted.
    /// - `ExecutionStopped` on cancellation or a deadline.
    pub fn build<'scope, 'capability>(
        points: &PointSet<'_, S>,
        values: NdView<'_, '_, S>,
        covariance: Covariance<S, D>,
        noise: &[S],
        mean: KrigingMean<'_, S>,
        options: LocalModelOptions<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        let shape = values.layout().shape();
        let n = points.len();
        if D == 0 || points.dimension() != D || shape.rank() != 2 {
            return Err(invalid(
                scioxide_core::argument!("points"),
                "point/value layout mismatch",
            ));
        }
        validate_count(n, 1, scioxide_core::argument!("points"))?;
        validate_length(n, shape.extents()[0])?;
        validate_weights(
            noise,
            n,
            Weights::Nonnegative,
            scioxide_core::argument!("noise"),
        )?;
        validate_finite(values.iter())?;
        let c = shape.extents()[1];
        options.preflight::<D>(n, n, c, scioxide_core::argument!("options"))?;
        if let KrigingMean::Constant(m) = mean {
            validate_length(c, m.len())?;
            validate_finite(m)?;
        }
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "gaussian-process", 1),
            ((n as u128) << 64) | c as u128,
            u128::from(matches!(mean, KrigingMean::Ordinary)),
        );
        let mut state = SolveState::new();
        scope
            .into()
            .run(&mut state, identity, options.budgets, |context| {
                require_proposal(context, PROPOSAL_DENIED)?;
                let mut selected = Selected {
                    points: try_with_capacity(n, scioxide_core::quantity!("points"))?,
                    values: try_with_capacity(
                        checked_mul(n, c, scioxide_core::quantity!("values"))?,
                        scioxide_core::quantity!("values"),
                    )?,
                    indices: Vec::new(),
                    components: c,
                };
                for i in 0..n {
                    selected.points.push(point(points, i)?);
                    for j in 0..c {
                        selected.values.push(*values.get([i, j])?);
                    }
                }
                let mut entries = try_filled(
                    checked_mul(n, n, scioxide_core::quantity!("covariance_entries"))?,
                    S::ZERO,
                    scioxide_core::quantity!("covariance_entries"),
                )?;
                for (i, p) in selected.points.iter().enumerate() {
                    context.execution().check()?;
                    for (j, q) in selected.points.iter().enumerate() {
                        entries[i * n + j] = intermediate(
                            covariance.covariance_validated(*p, *q)?
                                + if i == j { noise[i] } else { S::ZERO },
                        )?;
                    }
                }
                let matrix = Matrix::from_vec(n, n, entries)?;
                let factor = checked_factor(
                    &matrix,
                    SystemKind::Covariance,
                    &options,
                    context.execution(),
                )?;
                // u = K^{-1} 1 and 1^T K^{-1} 1 are retained: they give the
                // ordinary kriging GLS mean mu = (1^T K^{-1} y) / (1^T K^{-1} 1)
                // and the mean-estimation term of the prediction and LOO
                // variances. They are computed for simple kriging too so the
                // positivity test below always runs; for SPD K the sum is
                // positive, so a nonpositive value signals numerical breakdown.
                let mut w = LocalPredictionWorkspace::new(n, n, c)?;
                w.rhs.as_mut_slice().fill(S::ONE);
                solve(&factor, &mut w, context.execution())?;
                let mut inverse_ones =
                    try_with_capacity(n, scioxide_core::quantity!("inverse_ones"))?;
                inverse_ones.extend_from_slice(w.solution.as_slice());
                let ones_sum = inverse_ones
                    .iter()
                    .try_fold(S::ZERO, |a, &b| intermediate(a + b))?;
                if ones_sum <= S::ZERO {
                    return Err(SciOxideError::singular(None));
                }
                let ordinary = matches!(mean, KrigingMean::Ordinary);
                let mut means = try_filled(c, S::ZERO, scioxide_core::quantity!("means"))?;
                for (j, m) in means.iter_mut().enumerate() {
                    *m = match mean {
                        KrigingMean::Constant(values) => values[j],
                        KrigingMean::Ordinary => intermediate(
                            inverse_ones
                                .iter()
                                .enumerate()
                                .try_fold(S::ZERO, |s, (i, &w)| {
                                    intermediate(s + w * selected.values[i * c + j])
                                })?
                                / ones_sum,
                        )?,
                    };
                }
                // Keep the uncentred training values for leave-one-out means,
                // then centre the working copy so the solve gives
                // alpha = K^{-1} (y - mu).
                let mut training = try_with_capacity(
                    selected.values.len(),
                    scioxide_core::quantity!("training_values"),
                )?;
                training.extend_from_slice(&selected.values);
                if c > 0 {
                    for row in selected.values.chunks_exact_mut(c) {
                        for (v, &m) in row.iter_mut().zip(&means) {
                            *v = intermediate(*v - m)?;
                        }
                    }
                }
                let (coefficients, residual) =
                    coefficients(&selected, &matrix, &factor, context.execution())?;
                context.execution().check()?;
                let model = Self {
                    domains: bounds(
                        selected.points.iter().copied(),
                        scioxide_core::argument!("points"),
                    )?,
                    points: selected.points,
                    values: training,
                    components: c,
                    covariance,
                    coefficients,
                    means,
                    ordinary,
                    inverse_ones,
                    ones_sum,
                    report: LocalFitReport {
                        neighbors: n,
                        degree: if ordinary { Some(0) } else { None },
                        rank: n,
                        reciprocal_condition: factor.reciprocal_condition,
                        factorization: factor.method(),
                        condition_norm: factor.condition_norm(),
                        residual,
                        attempts: 1,
                    },
                    factor,
                };
                context.commit_accepted(scioxide_core::PartialResultTrust::Admissible)?;
                Ok(model)
            })
    }

    /// Returns the conditioning of `K` and the residual of the solve.
    pub const fn fit_report(&self) -> LocalFitReport<S> {
        self.report
    }

    /// Returns the covariance function.
    pub const fn covariance(&self) -> Covariance<S, D> {
        self.covariance
    }

    /// Returns the number of real value components per sample.
    #[must_use]
    pub const fn components(&self) -> usize {
        self.components
    }

    /// Allocates the scratch for predictions and leave-one-out queries.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the scratch cannot be
    ///   allocated.
    pub fn workspace(&self) -> Result<LocalPredictionWorkspace<S>> {
        LocalPredictionWorkspace::new(self.points.len(), self.points.len(), self.outputs()?)
    }

    /// Number of components of a prediction: means, latent and observation
    /// variance.
    fn outputs(&self) -> Result<usize> {
        checked_add(
            self.components,
            2,
            scioxide_core::quantity!("prediction_width"),
        )
    }

    /// Rejects a workspace too small for this model's solves.
    fn check(&self, workspace: &LocalPredictionWorkspace<S>) -> Result<()> {
        let n = self.points.len();
        require(
            workspace.covers_solve(n, self.outputs()?),
            n,
            workspace.rhs.len(),
        )
    }

    /// Returns the predictions for new observations with noise variance
    /// `observation_noise`.
    ///
    /// The result is an [`Interpolant`]; see [`GaussianPrediction`] for its
    /// components. Evaluate many points with
    /// [`BatchEvaluator`](crate::BatchEvaluator).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `observation_noise` is negative.
    /// - `NonFinite` if `observation_noise` is NaN or infinite.
    pub fn prediction(&self, observation_noise: S) -> Result<GaussianPrediction<'_, S, D>> {
        validate_weights(
            &[observation_noise],
            1,
            Weights::Nonnegative,
            scioxide_core::argument!("observation_noise"),
        )?;
        Ok(GaussianPrediction {
            model: self,
            noise: observation_noise,
        })
    }

    /// Computes the leave-one-out prediction of sample `index`.
    ///
    /// This is the prediction of observation `yᵢ` from the model built on all
    /// other samples, with the same covariance. With `P = K⁻¹`, or for ordinary
    /// kriging `P = K⁻¹ − uuᵀ/(1ᵀu)` with `u = K⁻¹1`, the closed form is
    ///
    /// ```text
    /// mean = yᵢ − (P y)ᵢ / Pᵢᵢ,   variance = 1 / Pᵢᵢ,
    /// ```
    ///
    /// so no refactorization is needed and the cost is O(n²). The means of all
    /// components are written to `out`, and the observation variance is
    /// returned.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` does not have one entry per component, or
    ///   `workspace` is too small.
    /// - `InvalidInput` if `index` is out of range, or ordinary kriging has
    ///   fewer than two samples.
    /// - `Singular` if `Pᵢᵢ ≤ 0` in floating point.
    pub fn leave_one_out_into(
        &self,
        index: usize,
        out: &mut [S],
        workspace: &mut LocalPredictionWorkspace<S>,
    ) -> Result<S> {
        crate::values::length(out, self.components)?;
        self.check(workspace)?;
        let n = self.points.len();
        if (self.ordinary && n < 2) || index >= n {
            return Err(invalid(
                scioxide_core::argument!("index"),
                "leave-one-out index outside the training samples",
            ));
        }
        // Closed-form LOO (Dubrule 1983; Rasmussen & Williams §5.4.2): with
        // precision P = K^{-1} (for ordinary kriging the projected
        // P = K^{-1} - u u^T / (1^T u), u = K^{-1} 1),
        //   mean_i = y_i - (P y)_i / P_ii,   variance_i = 1 / P_ii.
        // (P y)_i is the retained coefficient alpha_i, and P_ii needs only one
        // solve against e_i, so no refactorization is required.
        workspace.rhs.as_mut_slice().fill(S::ZERO);
        workspace.rhs.as_mut_slice()[index] = S::ONE;
        solve(
            &self.factor,
            workspace,
            ExecutionContext::sequential_native(),
        )?;
        let diagonal = intermediate(
            workspace.solution.as_slice()[index]
                - if self.ordinary {
                    self.inverse_ones[index] * self.inverse_ones[index] / self.ones_sum
                } else {
                    S::ZERO
                },
        )?;
        if diagonal <= S::ZERO {
            return Err(SciOxideError::singular(None));
        }
        let variance = intermediate(S::ONE / diagonal)?;
        let staged = &mut workspace.output[..self.components];
        for (c, v) in staged.iter_mut().enumerate() {
            *v = intermediate(
                self.values[index * self.components + c]
                    - self.coefficients[c * n + index] / diagonal,
            )?;
        }
        out.copy_from_slice(staged);
        Ok(variance)
    }

    /// Computes the means and both variances of one prediction into the
    /// workspace.
    fn predict(
        &self,
        point: [S; D],
        noise: S,
        policy: ExtrapolationPolicy<S>,
        w: &mut LocalPredictionWorkspace<S>,
    ) -> Result<()> {
        let n = self.points.len();
        let c = self.components;
        let point = resolve(self.domains, point, policy)?;
        for (k, q) in w.basis[..n].iter_mut().zip(&self.points) {
            *k = self.covariance.covariance_validated(point, *q)?;
        }
        evaluate_coefficients(&w.basis[..n], &self.coefficients, &mut w.output[..c])?;
        for (v, &m) in w.output[..c].iter_mut().zip(&self.means) {
            *v = intermediate(*v + m)?;
        }
        // Mean: mu + k^T alpha (computed above). Variance:
        //   sigma^2 - k^T K^{-1} k                               (simple),
        //   + (1 - 1^T K^{-1} k)^2 / (1^T K^{-1} 1)              (ordinary),
        // the last term accounting for uncertainty in the estimated mean.
        w.rhs.as_mut_slice().copy_from_slice(&w.basis[..n]);
        solve(&self.factor, w, ExecutionContext::sequential_native())?;
        let reduction = w.basis[..n]
            .iter()
            .zip(w.solution.as_slice())
            .try_fold(S::ZERO, |s, (&k, &v)| intermediate(s + k * v))?;
        let trend = if self.ordinary {
            let ku = w.basis[..n]
                .iter()
                .zip(&self.inverse_ones)
                .try_fold(S::ZERO, |s, (&k, &u)| intermediate(s + k * u))?;
            intermediate(intermediate((S::ONE - ku) * (S::ONE - ku))? / self.ones_sum)?
        } else {
            S::ZERO
        };
        // The subtraction cancels badly near training points, so small
        // negative results are expected roundoff. Tolerate up to
        // 64 eps n times the magnitude of the terms involved and clamp to zero;
        // anything more negative indicates a genuinely inconsistent model.
        let raw = intermediate(self.covariance.variance - reduction + trend)?;
        let allowance = intermediate(
            S::try_from_usize_exact(64)?
                * S::EPSILON
                * S::try_from_usize_exact(n)?
                * intermediate(self.covariance.variance + reduction.abs() + trend.abs())?,
        )?;
        if raw < -allowance {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "conditional variance is materially negative",
            ));
        }
        let latent = raw.max(S::ZERO);
        w.output[c] = latent;
        w.output[c + 1] = intermediate(latent + noise)?;
        Ok(())
    }
}

/// Evaluates the predictions of a [`GaussianProcessModel`] for new
/// observations with a fixed noise variance.
///
/// A value has `m + 2` components for `m` value components: the predictive
/// means of the components, the latent variance of the process at the point
/// (including the uncertainty of an estimated ordinary-kriging mean), and the
/// observation variance, which adds [`Self::observation_noise`]. The
/// variances are those of the model, not bounds on the error. Small negative
/// latent variances from rounding are set to zero.
///
/// # Errors
///
/// Queries fail as described in the [crate docs](crate#errors), and with
/// `NumericalInstability` if the latent variance is negative beyond rounding,
/// which indicates an inconsistent model.
///
/// ```
/// use scioxide_interp::{
///     Covariance, CovarianceKernel, ExtrapolationPolicy, GaussianProcessModel, Interpolant,
///     KrigingMean, LocalSamples,
/// };
/// let samples = LocalSamples::new(&[[0.0_f64], [1.0]], &[1.0, 2.0])?;
/// let covariance = Covariance::new(CovarianceKernel::SquaredExponential, 1.0, [1.0])?;
/// let means = [0.0];
/// let model = GaussianProcessModel::new(
///     &samples.points()?,
///     samples.values()?,
///     covariance,
///     &[0.0; 2],
///     KrigingMean::Constant(&means),
/// )?;
/// let prediction = model.prediction(0.25)?;
/// assert_eq!(prediction.value_dimension(), 3);
/// let value = prediction.evaluate([0.0], ExtrapolationPolicy::Reject)?;
/// let (mean, latent, observed) = (value[0], value[1], value[2]);
/// assert!((mean - 1.0).abs() < 1e-12 && latent.abs() < 1e-12 && (observed - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct GaussianPrediction<'a, S: RealScalar + NumericScalar<Real = S>, const D: usize> {
    model: &'a GaussianProcessModel<S, D>,
    noise: S,
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> GaussianPrediction<'_, S, D> {
    /// Returns the noise variance of a new observation.
    #[must_use]
    pub const fn observation_noise(&self) -> S {
        self.noise
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> Interpolant<S, D>
    for GaussianPrediction<'_, S, D>
{
    type Value = Vec<S>;
    type Workspace = LocalPredictionWorkspace<S>;

    fn domains(&self) -> [Interval<S>; D] {
        self.model.domains
    }
    fn value_dimension(&self) -> usize {
        self.model.components + 2
    }
    fn zero_value(&self) -> Result<Vec<S>> {
        try_filled(
            self.model.outputs()?,
            S::ZERO,
            scioxide_core::quantity!("prediction_value"),
        )
    }
    fn workspace(&self) -> Result<LocalPredictionWorkspace<S>> {
        self.model.workspace()
    }
    fn workspace_capacity(&self, workspace: &LocalPredictionWorkspace<S>) -> Capacity {
        match self.model.outputs() {
            Ok(outputs) => workspace.covers_solve(self.model.points.len(), outputs),
            Err(_) => Capacity::NeedsGrowth,
        }
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut LocalPredictionWorkspace<S>,
    ) -> Result<()> {
        let outputs = self.model.outputs()?;
        scioxide_core::check_value_shape(out, outputs)?;
        self.model.check(workspace)?;
        self.model.predict(point, self.noise, policy, workspace)?;
        scioxide_core::write_value(&workspace.output[..outputs], out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::LocalSamples;
    use super::*;
    use scioxide_core::SciOxideErrorView;

    #[test]
    fn covariance_kernels_are_unit_correlated_at_zero_distance() {
        for kernel in [
            CovarianceKernel::SquaredExponential,
            CovarianceKernel::Matern12,
            CovarianceKernel::Matern32,
            CovarianceKernel::Matern52,
        ] {
            let covariance = Covariance::new(kernel, 2.0_f64, [0.5]).unwrap();
            assert_eq!(covariance.covariance([1.0], [1.0]).unwrap(), 2.0);
            assert!(covariance.covariance([0.0], [1.0]).unwrap() < 2.0);
            assert_eq!(covariance.kernel(), kernel);
        }
        assert!(Covariance::<f64, 1>::new(CovarianceKernel::Matern12, 1.0, [0.0]).is_err());
        assert!(Covariance::<f64, 0>::new(CovarianceKernel::Matern12, 1.0, []).is_err());
    }

    #[test]
    fn preparation_rejects_layouts_and_limits() {
        let samples = LocalSamples::new(&[[0.0_f64], [1.0]], &[1.0, 2.0]).unwrap();
        let points = samples.points().unwrap();
        let covariance = Covariance::new(CovarianceKernel::Matern12, 1.0, [1.0]).unwrap();
        let options = LocalModelOptions::new();
        let fit = |noise: &[f64], mean, options| {
            GaussianProcessModel::<f64, 1>::with_options(
                &points,
                samples.values().unwrap(),
                covariance,
                noise,
                mean,
                options,
            )
        };
        assert!(fit(&[0.0], KrigingMean::Ordinary, options).is_err());
        assert!(fit(&[-1.0, 0.0], KrigingMean::Ordinary, options).is_err());
        assert!(fit(&[0.0, 0.0], KrigingMean::Constant(&[0.0, 1.0]), options).is_err());
        assert!(
            matches!(fit(&[0.0, 0.0], KrigingMean::Ordinary, options.with_maximum_bytes(0)), Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
        );
        let model = fit(&[0.0, 0.0], KrigingMean::Ordinary, options).unwrap();
        assert_eq!(model.fit_report().degree(), Some(0));
        assert_eq!(model.components(), 1);
        assert!(model.prediction(-1.0).is_err());
        assert!(model.prediction(f64::NAN).is_err());
        assert_eq!(model.prediction(0.5).unwrap().observation_noise(), 0.5);
    }
}
