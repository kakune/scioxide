use super::*;
use crate::Entropy;
use crate::distribution::LawOptions;
use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting};

#[derive(Debug, Clone)]
pub(super) struct Spd {
    pub factor: CholeskyDecomposition<f64>,
    pub logdet: f64,
}
impl Spd {
    pub(super) fn new(
        matrix: Matrix<f64>,
        argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        if matrix.rows() == 0 || !matrix.is_square() {
            return Err(invalid(
                argument,
                "positive definite matrix must be nonempty and square",
            ));
        }
        // One unpivoted Cholesky factor serves every later use: it proves
        // positive definiteness, gives ln det = 2 sum ln L_ii without forming
        // the determinant, and is reused for solves and for colouring samples.
        let factor = CholeskyDecomposition::factor(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            CholeskyPivoting::None,
        )?;
        let logdet = factor.slogdet()?.log_abs();
        Ok(Self { factor, logdet })
    }
    fn lower(&self) -> &Matrix<f64> {
        self.factor.l_matrix()
    }
    fn dimension(&self) -> usize {
        self.lower().rows()
    }
    fn entry(&self, i: usize, j: usize) -> Result<f64, SciOxideError> {
        let mut sum = scioxide_core::RangeAccumulator::new();
        for k in 0..=i.min(j) {
            sum.add_product(
                self.lower().view().get(i, k)?,
                self.lower().view().get(j, k)?,
            )?;
        }
        sum.finish()
    }
}
pub(super) fn matrix64<S: RealScalar + NumericScalar<Real = S>>(
    x: MatrixView<'_, S>,
) -> Result<Matrix<f64>, SciOxideError> {
    let mut out = Matrix::zeros(x.rows(), x.cols())?;
    for i in 0..x.rows() {
        for j in 0..x.cols() {
            let v = x.get(i, j)?;
            crate::law::argument(v, "x")?;
            out.view_mut().set(i, j, v.into_f64())?;
        }
    }
    Ok(out)
}
fn multi_gamma(a: f64, p: usize) -> Result<f64, SciOxideError> {
    (0..p).try_fold(
        (p as f64) * ((p - 1) as f64) * 0.25 * core::f64::consts::PI.ln(),
        |s, i| Ok(s + scioxide_special::ln_gamma(a - 0.5 * i as f64)?),
    )
}
fn multi_psi(a: f64, p: usize) -> Result<f64, SciOxideError> {
    (0..p).try_fold(0.0, |s, i| {
        Ok(s + scioxide_special::digamma(a - 0.5 * i as f64)?)
    })
}
fn budget<S: RealScalar>(
    options: LawOptions<S>,
    rows: usize,
    cols: usize,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid(argument, "matrix law shape overflow"))?;
    options.admit::<f64>(
        n.checked_mul(24)
            .ok_or_else(|| invalid(argument, "matrix workspace overflow"))?,
    )?;
    options.steps(
        n.checked_mul(rows.max(cols))
            .ok_or_else(|| invalid(argument, "matrix work overflow"))?,
    )
}
// Bartlett decomposition: B B^T ~ Wishart(I, df) for lower-triangular B with
// B_ii = sqrt(chi^2_{df - i}) and standard normal entries below the diagonal.
// The chi variates are drawn through `Gamma::sample_log` so that tiny
// non-integer degrees do not underflow to a zero pivot.
type ChiKernel = crate::distribution::chi_squared::ChiSquareKernel<f64>;
fn bartlett_kernels(df: f64, p: usize) -> Result<Vec<ChiKernel>, SciOxideError> {
    let mut kernels =
        scioxide_core::backend::try_with_capacity(p, scioxide_core::quantity!("bartlett_kernels"))?;
    for i in 0..p {
        kernels.push(ChiKernel::new(df - i as f64)?);
    }
    Ok(kernels)
}
fn bartlett<R: crate::Rng<f64>>(
    rng: &mut R,
    kernels: &[ChiKernel],
    out: &mut [f64],
) -> Result<(), SciOxideError> {
    let p = kernels.len();
    out.fill(0.0);
    let normal = Normal::<f64>::standard();
    for (i, kernel) in kernels.iter().enumerate() {
        out[i * p + i] = crate::law::positive_draw::<f64>(0.5 * kernel.sample_log(rng)?)?;
        for j in 0..i {
            out[i * p + j] = normal.sample(rng)?;
        }
    }
    Ok(())
}
/// Computes `Y = B⁻¹ Lᵀ`, so that `Yᵀ Y` has the inverse-Wishart law.
fn inverse_factor(scale: &Spd, b: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    let p = scale.dimension();
    for i in 0..p {
        for j in 0..p {
            y[i * p + j] = scale.lower().view().get(j, i)?;
        }
    }
    row_lower_solve(MatrixView::from_slice(p, p, b)?, y, p)
}
fn write<S: RealScalar>(matrix: &[f64], out: &mut [S]) -> Result<(), SciOxideError> {
    for (v, &x) in out.iter_mut().zip(matrix) {
        *v = finite(S::try_from_f64_lossy(x)?)?;
    }
    Ok(())
}
// Support test for SPD-valued laws: exact symmetry, then a successful
// Cholesky factorization. Factorization failures mean "outside the support"
// and are mapped to `None`; any other error still propagates.
fn support_matrix(x: Matrix<f64>) -> Result<Option<Spd>, SciOxideError> {
    for i in 0..x.rows() {
        for j in 0..i {
            if x.view().get(i, j)? != x.view().get(j, i)? {
                return Ok(None);
            }
        }
    }
    match Spd::new(x, scioxide_core::argument!("x")) {
        Ok(f) => Ok(Some(f)),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
                    | scioxide_core::SciOxideErrorView::Singular { .. }
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

macro_rules! wishart_law {
    ($(#[$attr:meta])* $name:ident, $inverse:expr) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $name<S: RealScalar> {
            scale: Spd,
            df: f64,
            kernels: Vec<ChiKernel>,
            normalizer: Result<f64, SciOxideError>,
            options: LawOptions<S>,
        }
        impl<S: RealScalar> $name<S> {
            /// Creates the law with `ν = df` degrees of freedom and a `p × p`
            /// positive definite scale matrix.
            ///
            /// `scale` is a `&Matrix` or any, possibly strided, [`MatrixView`].
            /// It uses the default [`LawOptions`]; [`Self::with_options`] sets
            /// the tolerance and work limits.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `scale` is empty or not square.
            /// - `NonFinite` if `df` or `scale` has a NaN or infinite value.
            /// - `DomainError` if `df ≤ p − 1`.
            /// - The Cholesky factorization error if `scale` is not positive definite.
            /// - `ResourceLimit` or `ConvergenceFailure` if the `p × p` storage or
            ///   work exceeds the [work limits](crate#work-limits).
            pub fn new<'a>(
                df: S,
                scale: impl Into<MatrixView<'a, S>>,
            ) -> Result<Self, SciOxideError>
            where
                S: NumericScalar<Real = S>,
            {
                Self::with_options(df, scale, LawOptions::default())
            }
            /// Creates the law of [`Self::new`] with the tolerance and work
            /// limits of `options`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::new`].
            /// - `InvalidInput` if the tolerance in `options` is out of range.
            pub fn with_options<'a>(
                df: S,
                scale: impl Into<MatrixView<'a, S>>,
                options: LawOptions<S>,
            ) -> Result<Self, SciOxideError>
            where
                S: NumericScalar<Real = S>,
            {
                let scale = scale.into();
                options.validate()?;
                let df = positive(df.into_f64(), scioxide_core::argument!("df"))?;
                if df <= scale.rows().saturating_sub(1) as f64 {
                    return Err(crate::law::domain(
                        scioxide_core::argument!("df"),
                        "Wishart degrees must exceed dimension minus one",
                    ));
                }
                budget(
                    options,
                    scale.rows(),
                    scale.cols(),
                    scioxide_core::argument!("options"),
                )?;
                Ok(Self {
                    scale: Spd::new(matrix64(scale)?, scioxide_core::argument!("scale"))?,
                    df,
                    kernels: bartlett_kernels(df, scale.rows())?,
                    normalizer: multi_gamma(0.5 * df, scale.rows()).map(|g| 0.5 * df * scale.rows() as f64 * core::f64::consts::LN_2 + g),
                    options,
                })
            }
            /// Returns the order `p` of a sample.
            pub fn dimension(&self) -> usize {
                self.scale.dimension()
            }
            /// Returns the [`LawOptions`] given at construction.
            pub fn options(&self) -> LawOptions<S> {
                self.options
            }
        }
        impl<S: RealScalar> RowLaw<S, S> for $name<S> {
            fn width(&self) -> usize {
                self.dimension() * self.dimension()
            }
            fn options(&self) -> LawOptions<S> {
                self.options
            }
            fn retained_bytes(&self) -> Result<usize, SciOxideError> {
                scioxide_core::backend::checked_add(self.scale.factor.retained_bytes()?,
                    crate::workspace::storage_bytes::<ChiKernel>(self.kernels.capacity())?,
                    scioxide_core::quantity!("matrix_parameters"))
            }
            fn scratch(&self) -> (usize, usize) { (0, 2 * self.width()) }
            fn draw<R: BitGenerator + ?Sized>(
                &self, rng: &mut R, out: &mut [S], scratch: Scratch<'_, S>,
            ) -> Result<(), SciOxideError> {
                let mut words = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
                if self.dimension() == 1 {
                    let log_chi = self.kernels[0].sample_log(&mut words)?;
                    out[0] = crate::law::positive_draw::<S>(self.scale.logdet + if $inverse { -log_chi } else { log_chi })?;
                    return Ok(());
                }
                let p = self.dimension();
                let (b, a) = scratch.reals.split_at_mut(p*p);
                bartlett(&mut words, &self.kernels, b)?;
                if $inverse {
                    inverse_factor(&self.scale, b, a)?;
                    let av = MatrixView::from_slice(p,p,a)?;
                    row_product(av.transpose(), av, b)?;
                } else {
                    row_product(self.scale.lower().view(), MatrixView::from_slice(p,p,b)?, a)?;
                    let av = MatrixView::from_slice(p,p,a)?;
                    row_product(av, av.transpose(), b)?;
                }
                write(b, out)?;
                for i in 0..self.dimension() {
                    if out[i * self.dimension() + i] == S::ZERO {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Underflow,
                            "positive matrix diagonal rounds to zero",
                        ));
                    }
                }
                Ok(())
            }
        }
        impl<S: RealScalar> $name<S> {
            /// Tests whether `x` is symmetric positive definite.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `x` is not `p × p`.
            /// - `NonFinite` if `x` has a NaN or infinite entry.
            pub fn contains_support(&self, x: MatrixView<'_, S>) -> Result<bool, SciOxideError>
            where
                S: NumericScalar<Real = S>,
            {
                size(x.rows(), self.dimension())?;
                size(x.cols(), self.dimension())?;
                Ok(support_matrix(matrix64(x)?)?.is_some())
            }
        }
        impl<S: RealScalar + NumericScalar<Real = S>> LogDensity<S, MatrixView<'_, S>>
            for $name<S>
        {
            fn log_pdf(&self, x: MatrixView<'_, S>) -> Result<S, SciOxideError> {
                size(x.rows(), self.dimension())?;
                size(x.cols(), self.dimension())?;
                let Some(x) = support_matrix(matrix64(x)?)? else {
                    return Ok(<S as RealScalar>::NEG_INFINITY);
                };
                // tr(S^{-1} X) (or tr(X^{-1} S) for the inverse law) is taken from
                // a Cholesky solve rather than an explicit inverse.
                let p = self.dimension() as f64;
                let (lower, rhs) = if $inverse { (&x, &self.scale) } else { (&self.scale, &x) };
                let mut solve = rhs.lower().clone();
                row_lower_solve(lower.lower().view(), solve.as_mut_slice(), self.dimension())?;
                let mut squares = scioxide_core::RangeAccumulator::new();
                for &value in solve.as_slice() { squares.add_product(value, value)?; }
                let trace = squares.finish()?;
                let normalizer = self.normalizer.clone()?;
                if $inverse {
                    log_terms(&[
                        0.5 * self.df * self.scale.logdet,
                        -0.5 * (self.df + p + 1.0) * x.logdet,
                        -0.5 * trace,
                        -normalizer,
                    ])
                } else {
                    log_terms(&[
                        0.5 * (self.df - p - 1.0) * x.logdet,
                        -0.5 * trace,
                        -0.5 * self.df * self.scale.logdet,
                        -normalizer,
                    ])
                }
            }
        }
        impl<S: RealScalar + NumericScalar<Real = S>> Density<S, MatrixView<'_, S>> for $name<S> {
            fn pdf(&self, x: MatrixView<'_, S>) -> Result<S, SciOxideError> {
                number(self.log_pdf(x)?.into_f64().exp())
            }
        }
        impl<S: RealScalar> Mean<S, Vec<S>> for $name<S> {
            fn mean(&self) -> Result<Vec<S>, SciOxideError> {
                let d = self.df - self.dimension() as f64 - 1.0;
                if $inverse && d <= 0.0 {
                    return Err(crate::law::undefined_moment(
                        "requested moment does not exist",
                    ));
                }
                let mut out = scioxide_core::backend::try_with_capacity(self.width(), scioxide_core::quantity!("wishart_mean"))?;
                for i in 0..self.dimension() { for j in 0..self.dimension() {
                    let s = self.scale.entry(i,j)?;
                    out.push(number(if $inverse { s / d } else { s * self.df })?);
                } }
                Ok(out)
            }
        }
        impl<S: RealScalar> Variance<S, Vec<S>> for $name<S> {
            fn variance(&self) -> Result<Vec<S>, SciOxideError> {
                let p = self.dimension();
                let d = self.df - p as f64;
                if $inverse && d <= 3.0 {
                    return Err(crate::law::undefined_moment(
                        "requested moment does not exist",
                    ));
                }
                let mut out = scioxide_core::backend::try_with_capacity(
                    self.width(),
                    scioxide_core::quantity!("wishart_variance"),
                )?;
                for i in 0..p {
                    for j in 0..p {
                        let s = self.scale.entry(i, j)?;
                        let a = self.scale.entry(i, i)?
                            * self.scale.entry(j, j)?;
                        out.push(number(if $inverse {
                            ((d + 1.0) * s * s + (d - 1.0) * a)
                                / (d * (d - 1.0).powi(2) * (d - 3.0))
                        } else {
                            self.df * (s * s + a)
                        })?);
                    }
                }
                Ok(out)
            }
        }
        impl<S: RealScalar> Entropy<S> for $name<S> {
            fn entropy(&self) -> Result<S, SciOxideError> {
                let p = self.dimension() as f64;
                let psi = multi_psi(self.df * 0.5, self.dimension())?;
                let logz = self.normalizer.clone()?;
                number(if $inverse {
                    logz - self.df * 0.5 * self.scale.logdet
                        + (self.df + p + 1.0)
                            * 0.5
                            * (self.scale.logdet - p * core::f64::consts::LN_2 - psi)
                        + self.df * p * 0.5
                } else {
                    logz + self.df * 0.5 * self.scale.logdet
                        - (self.df - p - 1.0)
                            * 0.5
                            * (self.scale.logdet + p * core::f64::consts::LN_2 + psi)
                        + self.df * p * 0.5
                })
            }
        }
        vector_sampling!($name, S, S::ZERO);
    };
}
wishart_law!(
    /// Wishart distribution `W_p(ν, Σ)` on `p × p` symmetric positive definite
    /// matrices, with `ν > p − 1` degrees of freedom and positive definite scale `Σ`.
    ///
    /// ```text
    /// p(X) = |X|^((ν−p−1)/2) exp(−tr(Σ⁻¹ X)/2) / (2^(νp/2) |Σ|^(ν/2) Γ_p(ν/2))
    /// ```
    ///
    /// where `Γ_p` is the multivariate gamma function. The mean is `ν Σ`.
    /// Draws use the Bartlett decomposition `X = L B Bᵀ Lᵀ`, where `Σ = L Lᵀ`
    /// and `B` is lower triangular with `Bᵢᵢ² ~ χ²(ν − i + 1)` and standard normal
    /// entries below the diagonal. A draw costs O(p³). A draw whose diagonal
    /// rounds to zero is a `NumericalInstability` error.
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_random::{LogDensity, Mean, SeedableRng, SplitMix64, Wishart};
    /// let identity = Matrix::<f64>::identity(2)?;
    /// let law = Wishart::new(4.0, &identity)?;
    /// // At X = I: -tr(I)/2 - 4 ln 2 - ln Gamma_2(2), with Gamma_2(2) = pi/2.
    /// let expected = -1.0 - (8.0 * core::f64::consts::PI).ln();
    /// assert!((law.log_pdf(identity.view())? - expected).abs() < 1e-12);
    /// assert!((law.mean()?[0] - 4.0).abs() < 1e-12);
    /// // Samples are row-major rows of `shape()`.
    /// let (rows, cols) = law.shape();
    /// let draw = Matrix::from_vec(rows, cols, law.sample(&mut SplitMix64::from_seed(7))?)?;
    /// assert!(law.contains_support(draw.view())?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - P. L. Odell and A. H. Feiveson, "A numerical procedure to generate a
    ///   sample covariance matrix", *J. Amer. Statist. Assoc.* 61, 1966, 199–203.
    Wishart,
    false
);
wishart_law!(
    /// Inverse-Wishart distribution `W_p⁻¹(ν, Ψ)` on `p × p` symmetric positive
    /// definite matrices, with `ν > p − 1` degrees of freedom and positive definite scale `Ψ`.
    ///
    /// `X` has this law when `X⁻¹ ~ W_p(ν, Ψ⁻¹)`. The density is
    ///
    /// ```text
    /// p(X) = |Ψ|^(ν/2) |X|^(−(ν+p+1)/2) exp(−tr(Ψ X⁻¹)/2) / (2^(νp/2) Γ_p(ν/2))
    /// ```
    ///
    /// and the mean is `Ψ / (ν − p − 1)` for `ν > p + 1`. Draws invert a Bartlett
    /// factor, at O(p³) per draw.
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_random::{InverseWishart, LogDensity, Mean, SeedableRng, SplitMix64};
    /// let identity = Matrix::<f64>::identity(2)?;
    /// let law = InverseWishart::new(4.0, &identity)?;
    /// // At X = I: -tr(I)/2 - 4 ln 2 - ln Gamma_2(2), with Gamma_2(2) = pi/2.
    /// let expected = -1.0 - (8.0 * core::f64::consts::PI).ln();
    /// assert!((law.log_pdf(identity.view())? - expected).abs() < 1e-12);
    /// assert!((law.mean()?[0] - 1.0).abs() < 1e-12);
    /// // Samples are row-major rows of `shape()`.
    /// let (rows, cols) = law.shape();
    /// let draw = Matrix::from_vec(rows, cols, law.sample(&mut SplitMix64::from_seed(7))?)?;
    /// assert!(law.contains_support(draw.view())?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    InverseWishart,
    true
);

/// Matrix normal distribution `MN(M, U, V)` on `m × n` matrices, with mean `M`,
/// row covariance `U` (`m × m`) and column covariance `V` (`n × n`).
///
/// `X` has this law when `vec(X) ~ N(vec(M), V ⊗ U)`. Draws are
/// `X = M + A Z Bᵀ`, with Cholesky factors `U = A Aᵀ`, `V = B Bᵀ` and `Z` an
/// `m × n` matrix of independent standard normals. Samples and moments are
/// stored row by row.
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_random::{Entropy, LogDensity, MatrixNormal, SeedableRng, SplitMix64};
/// let mean = Matrix::<f64>::zeros(2, 3)?;
/// let law = MatrixNormal::new(&mean, &Matrix::identity(2)?, &Matrix::identity(3)?)?;
/// let log_two_pi = (2.0 * core::f64::consts::PI).ln();
/// assert!((law.log_pdf(mean.view())? + 3.0 * log_two_pi).abs() < 1e-12);
/// assert!((law.entropy()? - 3.0 * (1.0 + log_two_pi)).abs() < 1e-12);
/// // Samples are row-major rows of `shape()`.
/// let draw = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert_eq!(draw.len(), law.shape().0 * law.shape().1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MatrixNormal<S: RealScalar> {
    mean: Matrix<f64>,
    row: Spd,
    col: Spd,
    options: LawOptions<S>,
}
impl<S: RealScalar> MatrixNormal<S> {
    /// Creates the law from a mean and positive definite row and column covariances.
    ///
    /// Each matrix is a `&Matrix` or any, possibly strided, [`MatrixView`]. It
    /// uses the default [`LawOptions`]; [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `row` or `col` is empty or not square.
    /// - `DimensionMismatch` if `row` is not `m × m` or `col` is not `n × n` for an `m × n` mean.
    /// - The Cholesky factorization error if `row` or `col` is not positive definite.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new<'a>(
        mean: impl Into<MatrixView<'a, S>>,
        row: impl Into<MatrixView<'a, S>>,
        col: impl Into<MatrixView<'a, S>>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        Self::with_options(mean, row, col, LawOptions::default())
    }
    /// Creates the law of [`Self::new`] with the tolerance and work limits of
    /// `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options<'a>(
        mean: impl Into<MatrixView<'a, S>>,
        row: impl Into<MatrixView<'a, S>>,
        col: impl Into<MatrixView<'a, S>>,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        let (mean, row, col) = (mean.into(), row.into(), col.into());
        options.validate()?;
        size(row.rows(), mean.rows())?;
        size(col.rows(), mean.cols())?;
        budget(
            options,
            mean.rows().max(mean.cols()),
            mean.rows().max(mean.cols()),
            scioxide_core::argument!("options"),
        )?;
        Ok(Self {
            mean: matrix64(mean)?,
            row: Spd::new(matrix64(row)?, scioxide_core::argument!("row"))?,
            col: Spd::new(matrix64(col)?, scioxide_core::argument!("col"))?,
            options,
        })
    }
    /// Returns the shape `(m, n)` of a sample.
    pub fn shape(&self) -> (usize, usize) {
        (self.mean.rows(), self.mean.cols())
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn difference(&self, x: Matrix<f64>) -> Result<Matrix<f64>, SciOxideError> {
        size(x.rows(), self.mean.rows())?;
        size(x.cols(), self.mean.cols())?;
        x.view().try_sub(&self.mean)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> LogDensity<S, MatrixView<'_, S>> for MatrixNormal<S> {
    fn log_pdf(&self, x: MatrixView<'_, S>) -> Result<S, SciOxideError> {
        size(x.rows(), self.mean.rows())?;
        size(x.cols(), self.mean.cols())?;
        let a = self.difference(matrix64(x)?)?;
        // tr(V^{-1} A^T U^{-1} A) = <U^{-1} A, A V^{-1}>_F: two Cholesky solves and
        // one Frobenius dot product, with no explicit inverse or n x n product.
        let left = self.row.factor.solve_matrix(&a)?;
        let right = self
            .col
            .factor
            .solve_matrix(&a.view().transpose().try_to_matrix()?)?
            .view()
            .transpose()
            .try_to_matrix()?;
        let mut dot = scioxide_core::RangeAccumulator::new();
        for (&a, &b) in left.as_slice().iter().zip(right.as_slice()) {
            dot.add_product(a, b)?;
        }
        let (m, n) = self.shape();
        number(
            -0.5 * ((m * n) as f64 * crate::distribution::special::log_two_pi::<f64>()
                + n as f64 * self.row.logdet
                + m as f64 * self.col.logdet
                + dot.finish()?),
        )
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Density<S, MatrixView<'_, S>> for MatrixNormal<S> {
    fn pdf(&self, x: MatrixView<'_, S>) -> Result<S, SciOxideError> {
        number(self.log_pdf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for MatrixNormal<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        converted(self.mean.as_slice().iter().copied())
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for MatrixNormal<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        let (m, n) = self.shape();
        let mut out = scioxide_core::backend::try_with_capacity(
            m * n,
            scioxide_core::quantity!("matrix_variance"),
        )?;
        for i in 0..m {
            for j in 0..n {
                out.push(number(self.row.entry(i, i)? * self.col.entry(j, j)?)?);
            }
        }
        Ok(out)
    }
}
impl<S: RealScalar> Entropy<S> for MatrixNormal<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        let (m, n) = self.shape();
        number(
            0.5 * ((m * n) as f64 * (1.0 + crate::distribution::special::log_two_pi::<f64>())
                + n as f64 * self.row.logdet
                + m as f64 * self.col.logdet),
        )
    }
}
impl<S: RealScalar> RowLaw<S, S> for MatrixNormal<S> {
    fn width(&self) -> usize {
        self.mean.as_slice().len()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("matrix_parameters");
        scioxide_core::backend::checked_add(
            self.row.factor.retained_bytes()?,
            scioxide_core::backend::checked_add(
                self.col.factor.retained_bytes()?,
                crate::workspace::storage_bytes::<f64>(self.mean.as_slice().len())?,
                q,
            )?,
            q,
        )
    }
    fn scratch(&self) -> (usize, usize) {
        (0, 2 * self.width())
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        scratch: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let (m, n) = self.shape();
        let (z, left) = scratch.reals.split_at_mut(m * n);
        let normal = Normal::<f64>::standard();
        for v in z.iter_mut() {
            *v = normal.sample(&mut words)?;
        }
        row_product(
            self.row.lower().view(),
            MatrixView::from_slice(m, n, z)?,
            left,
        )?;
        row_product(
            MatrixView::from_slice(m, n, left)?,
            self.col.lower().view().transpose(),
            z,
        )?;
        for (value, mean) in z.iter_mut().zip(self.mean.as_slice()) {
            *value = finite(*value + mean)?;
        }
        write(z, out)
    }
}
vector_sampling!(MatrixNormal, S, S::ZERO);

/// Matrix t distribution on `m × n` matrices with mean `M`, row spread `Σ`
/// (`m × m`), column spread `Ω` (`n × n`) and `ν > 0` degrees of freedom.
///
/// The spreads are not covariance matrices. The density is
///
/// ```text
/// p(X) = Γ_m((ν+m+n−1)/2) / (Γ_m((ν+m−1)/2) π^(mn/2) |Σ|^(n/2) |Ω|^(m/2))
///        · |I_m + Σ⁻¹ (X − M) Ω⁻¹ (X − M)ᵀ|^(−(ν+m+n−1)/2)
/// ```
///
/// where `Γ_m` is the multivariate gamma function. For `m = n = 1` it is a
/// Student t law with `ν` degrees of freedom and scale `√(Σ Ω / ν)`. Draws are
/// `X = M + S^(1/2) Z Bᵀ`, with `S ~ W_m⁻¹(ν + m − 1, Σ)`, `Ω = B Bᵀ` and `Z` an
/// `m × n` matrix of independent standard normals.
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_random::{LogDensity, MatrixT, SeedableRng, SplitMix64, Variance};
/// let mean = Matrix::<f64>::zeros(2, 2)?;
/// let identity = Matrix::identity(2)?;
/// let law = MatrixT::new(&mean, &identity, &identity, 4.0)?;
/// // At the mean: Gamma_2(7/2) / (Gamma_2(5/2) pi^2) = 5 / pi^2.
/// let expected = 5.0_f64.ln() - 2.0 * core::f64::consts::PI.ln();
/// assert!((law.log_pdf(mean.view())? - expected).abs() < 1e-12);
/// assert!(law.variance()?.iter().all(|v| (v - 0.5).abs() < 1e-12));
/// // Samples are row-major rows of `shape()`.
/// let draw = law.sample(&mut SplitMix64::from_seed(7))?;
/// assert_eq!(draw.len(), law.shape().0 * law.shape().1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. K. Gupta and D. K. Nagar, *Matrix Variate Distributions*, Chapman &
/// Hall/CRC, 2000, ch. 4.
#[derive(Debug, Clone)]
pub struct MatrixT<S: RealScalar> {
    base: MatrixNormal<S>,
    df: f64,
    kernels: Vec<ChiKernel>,
    log_normalizer: Result<f64, SciOxideError>,
}
impl<S: RealScalar> MatrixT<S> {
    /// Creates the law from a mean, positive definite spreads and `ν = df`.
    ///
    /// Each matrix is a `&Matrix` or any, possibly strided, [`MatrixView`]. It
    /// uses the default [`LawOptions`]; [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `row` or `col` is empty or not square.
    /// - `DimensionMismatch` if `row` is not `m × m` or `col` is not `n × n` for an `m × n` mean.
    /// - `NonFinite` if `df` is NaN or infinite, and `DomainError` if `df ≤ 0`.
    /// - The Cholesky factorization error if `row` or `col` is not positive definite.
    /// - `ResourceLimit` or `ConvergenceFailure` if storage or work exceeds the
    ///   [work limits](crate#work-limits).
    pub fn new<'a>(
        mean: impl Into<MatrixView<'a, S>>,
        row: impl Into<MatrixView<'a, S>>,
        col: impl Into<MatrixView<'a, S>>,
        df: S,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        Self::with_options(mean, row, col, df, LawOptions::default())
    }
    /// Creates the law of [`Self::new`] with the tolerance and work limits of
    /// `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options<'a>(
        mean: impl Into<MatrixView<'a, S>>,
        row: impl Into<MatrixView<'a, S>>,
        col: impl Into<MatrixView<'a, S>>,
        df: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        options.validate()?;
        let base = MatrixNormal::with_options(mean, row, col, options)?;
        let df = positive(df.into_f64(), scioxide_core::argument!("df"))?;
        let (m, n) = base.shape();
        let b = 0.5 * (df + m as f64 - 1.0);
        let log_normalizer = (0..m)
            .try_fold(0.0, |sum, i| {
                Ok::<_, SciOxideError>(
                    sum + scioxide_special::ln_gamma(0.5 * n as f64)?
                        - scioxide_special::ln_beta(b - 0.5 * i as f64, 0.5 * n as f64)?,
                )
            })
            .map(|ratio| {
                ratio
                    - 0.5 * (m * n) as f64 * core::f64::consts::PI.ln()
                    - 0.5 * n as f64 * base.row.logdet
                    - 0.5 * m as f64 * base.col.logdet
            });
        Ok(Self {
            base,
            df,
            kernels: bartlett_kernels(df + m as f64 - 1.0, m)?,
            log_normalizer,
        })
    }
    /// Returns the shape `(m, n)` of a sample.
    pub fn shape(&self) -> (usize, usize) {
        self.base.shape()
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.base.options
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> LogDensity<S, MatrixView<'_, S>> for MatrixT<S> {
    fn log_pdf(&self, x: MatrixView<'_, S>) -> Result<S, SciOxideError> {
        size(x.rows(), self.base.mean.rows())?;
        size(x.cols(), self.base.mean.cols())?;
        let a = self.base.difference(matrix64(x)?)?;
        let (m, n) = self.shape();
        let mut left = a;
        row_lower_solve(self.base.row.lower().view(), left.as_mut_slice(), n)?;
        let mut z = left.view().transpose().try_to_matrix()?;
        row_lower_solve(self.base.col.lower().view(), z.as_mut_slice(), m)?;
        // ln det(I + Z^T Z), with Z = L_U^{-1} A L_V^{-T} from triangular solves,
        // is summed as ln(1 + lambda_i) over the Gram eigenvalues. `ln_1p` keeps
        // small deviations accurate where det(I + G) would round to 1. The
        // Gram matrix is scaled to unit max-norm for the eigen solver and the
        // scale is restored per eigenvalue; slightly negative eigenvalues from
        // rounding are clamped, larger ones are reported as breakdown.
        let mut gram = z.view().transpose().try_to_matrix()?.view().mul_mat(&z)?;
        let scale = gram.as_slice().iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let logdet = if scale == 0.0 {
            0.0
        } else {
            for x in gram.as_mut_slice() {
                *x /= scale;
            }
            let eigen = scioxide_linalg::HermitianEigenDecomposition::decompose(
                &gram,
                scioxide_linalg::HermitianInput::Projected(
                    scioxide_linalg::HermitianProjectionPolicy::Always,
                ),
            )?;
            eigen.eigenvalues().iter().try_fold(0.0, |sum, &lambda| {
                let lambda = gram_eigenvalue(lambda, m)?;
                Ok::<_, SciOxideError>(sum + (lambda * scale).ln_1p())
            })?
        };
        // The ratio of multivariate gamma functions Gamma_m(a) / Gamma_m(b),
        // a - b = n/2, is assembled from per-dimension Beta terms,
        // Gamma(x + n/2) / Gamma(x) = Gamma(n/2) / B(x, n/2), which avoids
        // subtracting two large log-gamma sums.
        let a = 0.5 * (self.df + m as f64 + n as f64 - 1.0);
        number(self.log_normalizer.clone()? - a * logdet)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Density<S, MatrixView<'_, S>> for MatrixT<S> {
    fn pdf(&self, x: MatrixView<'_, S>) -> Result<S, SciOxideError> {
        number(self.log_pdf(x)?.into_f64().exp())
    }
}
impl<S: RealScalar> Mean<S, Vec<S>> for MatrixT<S> {
    fn mean(&self) -> Result<Vec<S>, SciOxideError> {
        if self.df <= 1.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        self.base.mean()
    }
}
impl<S: RealScalar> Variance<S, Vec<S>> for MatrixT<S> {
    fn variance(&self) -> Result<Vec<S>, SciOxideError> {
        if self.df <= 2.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        converted(
            self.base
                .variance()?
                .iter()
                .map(|v| v.into_f64() / (self.df - 2.0)),
        )
    }
}
impl<S: RealScalar> RowLaw<S, S> for MatrixT<S> {
    fn width(&self) -> usize {
        RowLaw::width(&self.base)
    }
    fn options(&self) -> LawOptions<S> {
        self.base.options
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        scioxide_core::backend::checked_add(
            self.base.retained_bytes()?,
            crate::workspace::storage_bytes::<ChiKernel>(self.kernels.capacity())?,
            scioxide_core::quantity!("matrix_parameters"),
        )
    }
    fn scratch(&self) -> (usize, usize) {
        let (m, n) = self.shape();
        (0, 2 * m * m + 2 * m * n)
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        scratch: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let mut words =
            crate::distribution::words::Words::new(rng, self.options().sample_words.get());
        let (m, n) = self.shape();
        let (b, tail) = scratch.reals.split_at_mut(m * m);
        let (y, tail) = tail.split_at_mut(m * m);
        let (z, left) = tail.split_at_mut(m * n);
        bartlett(&mut words, &self.kernels, b)?;
        inverse_factor(&self.base.row, b, y)?;
        let normal = Normal::<f64>::standard();
        for v in z.iter_mut() {
            *v = normal.sample(&mut words)?;
        }
        // Y^T = L B^-T is already a square root of the inverse-Wishart draw.
        row_product(
            MatrixView::from_slice(m, m, y)?.transpose(),
            MatrixView::from_slice(m, n, z)?,
            left,
        )?;
        row_product(
            MatrixView::from_slice(m, n, left)?,
            self.base.col.lower().view().transpose(),
            z,
        )?;
        for (value, mean) in z.iter_mut().zip(self.base.mean.as_slice()) {
            *value = finite(*value + mean)?;
        }
        write(z, out)
    }
}
vector_sampling!(MatrixT, S, S::ZERO);
impl<S: RealScalar> Wishart<S> {
    /// Returns the shape `(p, p)` of a sample.
    pub fn shape(&self) -> (usize, usize) {
        (self.dimension(), self.dimension())
    }
}
impl<S: RealScalar> InverseWishart<S> {
    /// Returns the shape `(p, p)` of a sample.
    pub fn shape(&self) -> (usize, usize) {
        (self.dimension(), self.dimension())
    }
}

// Project only roundoff-sized negative eigenvalues of a computed Gram matrix.
fn gram_eigenvalue(value: f64, dimension: usize) -> Result<f64, SciOxideError> {
    if value < -64.0 * dimension as f64 * f64::EPSILON {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "matrix-t Gram factor lost positive semidefiniteness",
        ));
    }
    Ok(finite(value)?.max(0.0))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    #[test]
    fn negative_gram_eigenvalues_separate_roundoff_from_breakdown() {
        assert_eq!(gram_eigenvalue(-32.0 * f64::EPSILON, 2).unwrap(), 0.0);
        assert_eq!(gram_eigenvalue(0.5, 2).unwrap(), 0.5);
        crate::error_assertions::numerical(
            gram_eigenvalue(-256.0 * f64::EPSILON, 2).unwrap_err(),
            scioxide_core::NumericalIssue::Breakdown,
        );
    }
}
