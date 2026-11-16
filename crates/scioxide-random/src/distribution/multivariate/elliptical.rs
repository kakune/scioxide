//! Multivariate Student t distribution.
use super::probability::{convenience, joint_traits, preflight};
use super::*;
use crate::distribution::LawOptions;
use crate::{Cdf, LogCdf, LogSurvival, Philox4x64, SeedableRng, Survival};
use crate::{MultivariateNormal, MultivariateNormalOptions};
use scioxide_core::Tolerance;
use scioxide_linalg::{MatrixView, RankPolicy, Vector};

/// Multivariate Student t distribution with location `μ ∈ ℝᵈ`, positive semidefinite
/// shape matrix `Σ` and `ν > 0` degrees of freedom.
///
/// A draw is `x = μ + y / √(w/ν)`, where `y` is multivariate normal with mean
/// zero and covariance `Σ` and `w` is an independent χ²(ν) variate. The shape is
/// not the covariance: for `ν > 2` the covariance is `ν/(ν − 2) Σ`. For a
/// full-rank `Σ` the density is
///
/// ```text
/// p(x) = Γ((ν+d)/2) / (Γ(ν/2) (νπ)^(d/2) |Σ|^(1/2)) · (1 + q/ν)^(−(ν+d)/2),
/// q = (x − μ)ᵀ Σ⁻¹ (x − μ).
/// ```
///
/// A singular `Σ` of rank `r` gives a law on the affine subspace
/// `μ + range(Σ)`, with density [`Self::log_pseudo_density`] with respect to
/// `r`-dimensional Lebesgue measure. [`Self::gaussian_limit`] creates the limit
/// `ν → ∞`, the multivariate normal law.
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_linalg::Matrix;
/// use scioxide_random::{LogDensity, MultivariateStudentT, SeedableRng, SplitMix64, Variance};
/// let law = MultivariateStudentT::new(
///     &[0.0, 0.0],
///     &Matrix::identity(2)?,
///     5.0,
/// )?;
/// // At the location: Gamma(7/2) / (Gamma(5/2) 5 pi) = 1 / (2 pi).
/// assert!((law.log_pdf(&[0.0, 0.0])? + (2.0 * core::f64::consts::PI).ln()).abs() < 1e-12);
/// assert!(law.variance()?.iter().all(|v| (v - 5.0 / 3.0).abs() < 1e-12));
/// let x = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert!(law.contains_support(&x, Tolerance::from_machine_epsilon())?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateStudentT<S: RealScalar> {
    pub(super) gaussian: MultivariateNormal<f64>,
    pub(super) df: DegreesOfFreedom,
    radial: Option<crate::distribution::chi_squared::ChiSquareKernel<f64>>,
    options: LawOptions<S>,
}
impl<S: RealScalar> MultivariateStudentT<S> {
    /// Creates the law from a location, a positive semidefinite shape matrix and `ν = df`.
    ///
    /// `shape` is a `&Matrix` or any, possibly strided, [`MatrixView`]. A zero
    /// shape gives a point mass at `location`. The rank of `shape` is set as in
    /// [`MultivariateNormal::new`].
    ///
    /// It uses the default [`MultivariateNormalOptions`] and [`LawOptions`];
    /// [`Self::with_options`] sets the rank rule, the tolerance and the work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `d = 0`.
    /// - `DimensionMismatch` if `shape` is not `d × d`.
    /// - `NonFinite` if `df`, `location` or `shape` has a NaN or infinite value.
    /// - `DomainError` if `df ≤ 0`, or if `shape` is not exactly symmetric or is indefinite.
    /// - The factorization and storage errors of [`MultivariateNormal::new`].
    pub fn new<'a>(
        location: &[S],
        shape: impl Into<MatrixView<'a, S>>,
        df: S,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        Self::with_options(
            location,
            shape,
            df,
            MultivariateNormalOptions::default(),
            LawOptions::default(),
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
        location: &[S],
        shape: impl Into<MatrixView<'a, S>>,
        df: S,
        rank_options: MultivariateNormalOptions<S>,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        options.validate()?;
        let df = positive(df, scioxide_core::argument!("df"))?.into_f64();
        Self::prepare(
            location,
            shape.into(),
            DegreesOfFreedom::StudentT(df),
            rank_options,
            options,
        )
    }
    /// Creates the limit `ν → ∞`, the multivariate normal law with mean `location`
    /// and covariance `shape`.
    ///
    /// Singular shapes and point masses follow the same support conventions as
    /// [`Self::new`]. It uses the default options; [`Self::gaussian_limit_with_options`]
    /// sets them.
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_random::{MultivariateStudentT, Variance};
    /// let law = MultivariateStudentT::gaussian_limit(&[0.0_f64], &Matrix::identity(1)?)?;
    /// assert_eq!(law.variance()?, vec![1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `d = 0`.
    /// - `DimensionMismatch` if `shape` is not `d × d`.
    /// - `NonFinite` if `location` or `shape` has a NaN or infinite value.
    /// - `DomainError` if `shape` is not exactly symmetric or is indefinite.
    /// - The factorization and storage errors of [`MultivariateNormal::new`].
    pub fn gaussian_limit<'a>(
        location: &[S],
        shape: impl Into<MatrixView<'a, S>>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        Self::gaussian_limit_with_options(
            location,
            shape,
            MultivariateNormalOptions::default(),
            LawOptions::default(),
        )
    }
    /// Creates the limit of [`Self::gaussian_limit`] with the rank rule of
    /// `rank_options` and the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::gaussian_limit`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn gaussian_limit_with_options<'a>(
        location: &[S],
        shape: impl Into<MatrixView<'a, S>>,
        rank_options: MultivariateNormalOptions<S>,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        options.validate()?;
        Self::prepare(
            location,
            shape.into(),
            DegreesOfFreedom::Gaussian,
            rank_options,
            options,
        )
    }
    fn prepare(
        location: &[S],
        shape: MatrixView<'_, S>,
        df: DegreesOfFreedom,
        rank_options: MultivariateNormalOptions<S>,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        let n = location.len();
        options.steps(
            n.checked_mul(n)
                .and_then(|v| v.checked_mul(n))
                .ok_or_else(|| {
                    invalid(scioxide_core::argument!("shape"), "t factor work overflow")
                })?,
        )?;
        options.admit::<f64>(
            n.checked_mul(n)
                .and_then(|n| n.checked_mul(16))
                .ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("shape"),
                        "t workspace size overflow",
                    )
                })?,
        )?;
        // The t law is built on the (possibly rank-deficient) Gaussian factor of
        // the shape matrix. The rank threshold scales with the dimension and the
        // caller's scalar precision, so f32 inputs are not treated as full rank
        // on the strength of f64 rounding noise.
        let rank_policy = match rank_options.rank_policy() {
            RankPolicy::Exact => RankPolicy::Exact,
            RankPolicy::DimensionScaledEpsilon => RankPolicy::Threshold(
                scioxide_core::RankThreshold::new(0.0, n as f64 * S::EPSILON.into_f64())?,
            ),
            RankPolicy::Threshold(threshold) => {
                RankPolicy::Threshold(scioxide_core::RankThreshold::new(
                    threshold.absolute().into_f64(),
                    threshold.relative().into_f64(),
                )?)
            }
        };
        Ok(Self {
            gaussian: MultivariateNormal::from_owned(
                Vector::from_vec(values(location)?),
                &super::matrix::matrix64(shape)?,
                MultivariateNormalOptions::default().with_rank_policy(rank_policy),
                options.widened()?,
            )?,
            df,
            radial: match df {
                DegreesOfFreedom::StudentT(df) => {
                    Some(crate::distribution::chi_squared::ChiSquareKernel::new(df)?)
                }
                DegreesOfFreedom::Gaussian => None,
            },
            options,
        })
    }
    /// Returns the dimension `d` of a sample.
    pub fn dimension(&self) -> usize {
        self.gaussian.dimension()
    }
    /// Returns the numerical rank `r` of the shape matrix.
    pub fn rank(&self) -> usize {
        self.gaussian.rank()
    }
    fn joint_options(&self) -> Result<MultivariateProbabilityOptions, SciOxideError> {
        super::probability::law_probability_options(self.options)
    }

    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Tests whether `x` lies on the support `μ + range(Σ)`, as
    /// [`MultivariateNormal::contains_support`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ d`.
    /// - `NonFinite` if `x` has a NaN or infinite entry.
    pub fn contains_support(
        &self,
        x: &[S],
        tolerance: Tolerance<S>,
    ) -> Result<bool, SciOxideError> {
        self.gaussian.contains_support(
            &values(x)?,
            Tolerance::new(
                tolerance.absolute().into_f64(),
                tolerance.relative().into_f64(),
            )?,
        )
    }
    /// Computes the log density with respect to `r`-dimensional Lebesgue measure
    /// on the support.
    ///
    /// The value is `−∞` off the support, as decided by `tolerance`. For rank
    /// zero it is 0 on the support. For a full-rank law it equals `log_pdf`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x.len() ≠ d`.
    /// - `NonFinite` if `x` has a NaN or infinite entry.
    /// - `NumericalInstability` if the result is outside the range of `S`.
    pub fn log_pseudo_density(&self, x: &[S], tolerance: Tolerance<S>) -> Result<S, SciOxideError> {
        let x = values(x)?;
        let tolerance = Tolerance::new(
            tolerance.absolute().into_f64(),
            tolerance.relative().into_f64(),
        )?;
        let DegreesOfFreedom::StudentT(df) = self.df else {
            return number(self.gaussian.log_pseudo_density(&x, tolerance)?);
        };
        let Some(logq) = self.gaussian.log_squared_distance(&x, tolerance)? else {
            return Ok(S::NEG_INFINITY);
        };
        let r = self.rank() as f64;
        if r == 0.0 {
            return Ok(S::ZERO);
        }
        let a = 0.5 * df;
        let b = 0.5 * r;
        // beta identity avoids subtracting two large log-gammas.
        let normalizer = scioxide_special::ln_gamma(b)?
            - scioxide_special::ln_beta(a, b)?
            - b * (df.ln() + core::f64::consts::PI.ln())
            - 0.5 * self.gaussian.log_determinant_on_support();
        number(normalizer - (a + b) * logadd(0.0, logq - df.ln())?)
    }
    /// Returns the covariance matrix `ν/(ν − 2) Σ` (or `Σ` in the Gaussian limit).
    ///
    /// # Errors
    ///
    /// - `DomainError` if `ν ≤ 2` and the rank is positive: the covariance does not exist.
    /// - `NumericalInstability` if an entry is outside the range of `S`.
    /// - `ResourceLimit` or `AllocationFailure` if the `d × d` result cannot be stored.
    pub fn covariance(&self) -> Result<Matrix<S>, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        if matches!(self.df, DegreesOfFreedom::StudentT(df) if df <= 2.0) && self.rank() > 0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        let c = self.gaussian.covariance()?;
        let scale = match self.df {
            DegreesOfFreedom::StudentT(df) if self.rank() > 0 => df / (df - 2.0),
            _ => 1.0,
        };
        Matrix::from_vec(
            c.rows(),
            c.cols(),
            converted(c.as_slice().iter().map(|x| x * scale))?,
        )
    }
}
impl<S: RealScalar> LogDensity<S, &[S]> for MultivariateStudentT<S> {
    fn log_pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        if self.rank() != self.dimension() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "singular t requires log_pseudo_density",
            ));
        }
        self.log_pseudo_density(x, Tolerance::from_machine_epsilon())
    }
}
impl<S: RealScalar> Density<S, &[S]> for MultivariateStudentT<S> {
    fn pdf(&self, x: &[S]) -> Result<S, SciOxideError> {
        number(self.log_pdf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for MultivariateStudentT<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        if matches!(self.df, DegreesOfFreedom::StudentT(df) if df <= 1.0) && self.rank() > 0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        converted(self.gaussian.location().iter().copied())
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for MultivariateStudentT<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        if matches!(self.df, DegreesOfFreedom::StudentT(df) if df <= 2.0) && self.rank() > 0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        let c = self.gaussian.covariance()?;
        let scale = match self.df {
            DegreesOfFreedom::StudentT(df) if self.rank() > 0 => df / (df - 2.0),
            _ => 1.0,
        };
        let mut out = scioxide_core::backend::try_with_capacity(
            self.dimension(),
            scioxide_core::quantity!("t_variance"),
        )?;
        for i in 0..self.dimension() {
            out.push(number(c.view().get(i, i)? * scale)?);
        }
        Ok(out)
    }
}
impl<S: RealScalar> RowLaw<S, S> for MultivariateStudentT<S> {
    fn width(&self) -> usize {
        self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    // Binary64 latent coordinates followed by the transformed row.
    fn scratch(&self) -> (usize, usize) {
        (0, self.rank() + self.dimension())
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        scratch: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let (z, x) = scratch.reals.split_at_mut(self.rank());
        let normal = Normal::<f64>::standard();
        // Elliptical construction: X = mu + A Z sqrt(nu / W) with W ~ chi^2_nu.
        // The radial factor is formed in log space so that small nu (where W
        // can underflow) still yields a finite, positive scale. Only `rank`
        // latent normals are needed because A maps from the support subspace.
        let scale = match self.df {
            DegreesOfFreedom::StudentT(df) if self.rank() > 0 => {
                let log_chi_square = self
                    .radial
                    .as_ref()
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("self"),
                            "missing Student t radial kernel",
                        )
                    })?
                    .sample_log(&mut words)?;
                crate::law::positive_draw::<f64>(0.5 * (df.ln() - log_chi_square))?
            }
            _ => 1.0,
        };
        finite(scale)?;
        for v in z.iter_mut() {
            *v = finite(normal.sample(&mut words)? * scale)?;
        }
        self.gaussian.transform(z, x)?;
        for (o, &v) in out.iter_mut().zip(x.iter()) {
            *o = number(v)?;
        }
        Ok(())
    }
}
vector_sampling!(MultivariateStudentT, S, S::ZERO);

impl<S: RealScalar> MultivariateStudentT<S> {
    /// Creates a plan that computes probabilities `P(a < X ≤ b)` of boxes.
    ///
    /// See [`MultivariateProbabilityPlan`].
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
        preflight(self.dimension(), self.rank(), self.df, options)?;
        MultivariateProbabilityPlan::prepare(
            values(self.gaussian.location())?,
            self.gaussian.support_factor()?,
            self.df,
            options,
            root,
        )
    }
}

joint_traits!(
    MultivariateStudentT,
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_random::{MultivariateStudentT, Philox4x64, SeedableRng};
    /// let law = MultivariateStudentT::new(&[0.0; 2], &Matrix::identity(2)?, 4.0)?;
    /// let root = Philox4x64::from_seed(0);
    /// // The law is symmetric about its location, so each quadrant holds a quarter.
    /// let quadrant = law.rectangle(&[0.0; 2], &[f64::INFINITY; 2], &root)?;
    /// assert!((quadrant - 0.25).abs() < 1e-6);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ;
);
