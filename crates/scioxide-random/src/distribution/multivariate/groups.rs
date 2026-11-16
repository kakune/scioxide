use super::*;
use crate::distribution::LawOptions;
use scioxide_core::Complex;
use scioxide_linalg::RqDecomposition;

fn group_budget<S: RealScalar>(
    n: usize,
    options: LawOptions<S>,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if n == 0 {
        return Err(invalid(argument, "matrix group dimension must be positive"));
    }
    let square = n
        .checked_mul(n)
        .ok_or_else(|| invalid(argument, "group dimension overflow"))?;
    options.admit::<f64>(
        square
            .checked_mul(32)
            .ok_or_else(|| invalid(argument, "group workspace overflow"))?,
    )?;
    options.steps(
        square
            .checked_mul(n)
            .ok_or_else(|| invalid(argument, "group work overflow"))?,
    )
}
fn orthogonal<R: crate::Rng<f64>>(
    rng: &mut R,
    n: usize,
    special: bool,
) -> Result<Matrix<f64>, SciOxideError> {
    if special && n == 1 {
        return Matrix::identity(1);
    }
    let a = normal_matrix(rng, n, n)?;
    // Mezzadri's (2007) recipe: factor a Ginibre (i.i.d. normal) matrix and fix
    // the sign ambiguity of the factorization by making the triangular
    // diagonal positive. Without this normalization the orthogonal factor is
    // not Haar distributed. With A = R Q, row i of Q absorbs the sign of R_ii.
    let factor = RqDecomposition::factor(a.view())?;
    let mut q = factor.q_matrix().view().try_to_matrix()?;
    for i in 0..n {
        let diagonal = factor.r_matrix().view().get(i, i)?;
        if diagonal == 0.0 {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "Gaussian group factor is numerically singular",
            ));
        }
        let sign = if diagonal < 0.0 { -1.0 } else { 1.0 };
        for j in 0..n {
            let value = q.view().get(i, j)? * sign;
            q.view_mut().set(i, j, value)?;
        }
    }
    // Haar on SO(n): flipping one row maps O(n) \ SO(n) bijectively onto SO(n)
    // and preserves the invariant measure.
    if special
        && scioxide_linalg::LuDecomposition::factor(q.view())
            .and_then(|factor| factor.determinant())?
            < 0.0
    {
        for j in 0..n {
            let value = -q.view().get(n - 1, j)?;
            q.view_mut().set(n - 1, j, value)?;
        }
    }
    Ok(q)
}
macro_rules! orthogonal_law {
    ($(#[$attr:meta])* $name:ident, $special:expr) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $name<S: RealScalar> {
            dimension: usize,
            options: LawOptions<S>,
        }
        impl<S: RealScalar> $name<S> {
            /// Creates the law on `d × d` matrices, `d = dimension`.
            ///
            /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
            /// tolerance and work limits.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `dimension = 0`.
            /// - `ResourceLimit` or `ConvergenceFailure` if the `d × d` storage or O(d³)
            ///   work per draw exceeds the [work limits](crate#work-limits).
            pub fn new(dimension: usize) -> Result<Self, SciOxideError> {
                Self::with_options(dimension, LawOptions::default())
            }

            /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::new`].
            /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
            pub fn with_options(dimension: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
                options.validate()?;
                group_budget(dimension, options, scioxide_core::argument!("dimension"))?;
                Ok(Self { dimension, options })
            }
            /// Returns the order `d` of a sample.
            pub fn dimension(&self) -> usize {
                self.dimension
            }
            /// Returns the [`LawOptions`] given at construction.
            pub fn options(&self) -> LawOptions<S> {
                self.options
            }
        }
        impl<S: RealScalar> RowLaw<S, S> for $name<S> {
            fn width(&self) -> usize {
                self.dimension * self.dimension
            }
            fn options(&self) -> LawOptions<S> {
                self.options
            }
            fn draw<R: BitGenerator + ?Sized>(
                &self,
                rng: &mut R,
                out: &mut [S],
                _: Scratch<'_, S>,
            ) -> Result<(), SciOxideError> {
                let mut words = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
                let q = orthogonal(&mut words, self.dimension, $special)?;
                for (v, &x) in out.iter_mut().zip(q.as_slice()) {
                    *v = number(x)?;
                }
                Ok(())
            }
        }
        vector_sampling!($name, S, S::ZERO);
    };
}
orthogonal_law!(
    /// Haar measure on the orthogonal group O(d) of real `d × d` matrices with `QᵀQ = I`.
    ///
    /// A draw factors a `d × d` matrix of independent standard normals and
    /// normalizes the signs of the triangular factor (Mezzadri 2007). A draw
    /// costs O(d³). Samples are stored row by row.
    ///
    /// ```
    /// use scioxide_random::{OrthogonalGroup, SeedableRng, SplitMix64};
    /// let law = OrthogonalGroup::<f64>::new(3)?;
    /// assert_eq!(law.shape(), (3, 3));
    /// let q = law.sample(&mut SplitMix64::from_seed(7))?;
    /// for i in 0..3 {
    ///     for j in 0..3 {
    ///         let dot: f64 = (0..3).map(|k| q[3 * i + k] * q[3 * j + k]).sum();
    ///         assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
    ///     }
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// F. Mezzadri, "How to generate random matrices from the classical compact
    /// groups", *Notices Amer. Math. Soc.* 54(5), 2007, 592–604.
    OrthogonalGroup,
    false
);
orthogonal_law!(
    /// Haar measure on the special orthogonal group SO(d) of real `d × d` matrices
    /// with `QᵀQ = I` and `det Q = 1`.
    ///
    /// A draw is an [`OrthogonalGroup`] draw with its last row negated when its
    /// determinant is −1. A draw costs O(d³). Samples are stored row by row.
    ///
    /// ```
    /// use scioxide_random::{SeedableRng, SpecialOrthogonalGroup, SplitMix64};
    /// let law = SpecialOrthogonalGroup::<f64>::new(2)?;
    /// let q = law.sample(&mut SplitMix64::from_seed(7))?;
    /// // A plane rotation: unit rows and determinant one.
    /// assert!((q[0] * q[0] + q[1] * q[1] - 1.0).abs() < 1e-12);
    /// assert!((q[0] * q[3] - q[1] * q[2] - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    SpecialOrthogonalGroup,
    true
);

/// Haar measure on the unitary group U(d) of complex `d × d` matrices with `U*U = I`.
///
/// A draw factors a `d × d` matrix of independent standard complex normals and
/// normalizes the phases of the triangular factor (Mezzadri 2007). A draw costs
/// O(d³). Samples are [`Complex`] entries stored row by row.
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64, UnitaryGroup};
/// let law = UnitaryGroup::<f64>::new(2)?;
/// let u = law.sample(&mut SplitMix64::from_seed(7))?;
/// // Rows are orthonormal under the Hermitian inner product.
/// assert!((u[0].norm_sqr() + u[1].norm_sqr() - 1.0).abs() < 1e-12);
/// let inner = u[0] * u[2].conj() + u[1] * u[3].conj();
/// assert!(inner.re.abs() < 1e-12 && inner.im.abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. Mezzadri, "How to generate random matrices from the classical compact
/// groups", *Notices Amer. Math. Soc.* 54(5), 2007, 592–604.
#[derive(Debug, Clone)]
pub struct UnitaryGroup<S: RealScalar> {
    dimension: usize,
    options: LawOptions<S>,
}
impl<S: RealScalar> UnitaryGroup<S> {
    /// Creates the law on `d × d` matrices, `d = dimension`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension = 0`.
    /// - `ResourceLimit` or `ConvergenceFailure` if the `d × d` storage or O(d³)
    ///   work per draw exceeds the [work limits](crate#work-limits).
    pub fn new(dimension: usize) -> Result<Self, SciOxideError> {
        Self::with_options(dimension, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(dimension: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        group_budget(dimension, options, scioxide_core::argument!("dimension"))?;
        Ok(Self { dimension, options })
    }
    /// Returns the order `d` of a sample.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
}
impl<S: RealScalar> RowLaw<S, Complex<S>> for UnitaryGroup<S> {
    fn width(&self) -> usize {
        self.dimension * self.dimension
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [Complex<S>],
        _: Scratch<'_, Complex<S>>,
    ) -> Result<(), SciOxideError> {
        let n = self.dimension;
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let normal = Normal::<f64>::standard();
        let mut a = Matrix::zeros(n, n)?;
        for v in a.as_mut_slice() {
            *v = Complex {
                re: normal.sample(&mut words)?,
                im: normal.sample(&mut words)?,
            };
        }
        // Complex analogue of the orthogonal case: each row of Q is multiplied
        // by the unit phase of R_ii so the factorization is unique and Q is
        // Haar distributed on U(n).
        let factor = RqDecomposition::factor(a.view())?;
        for i in 0..n {
            let d = factor.r_matrix().view().get(i, i)?;
            let magnitude = d.re.hypot(d.im);
            if magnitude == 0.0 {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "Gaussian unitary factor is numerically singular",
                ));
            }
            finite(magnitude)?;
            let phase = Complex {
                re: d.re / magnitude,
                im: d.im / magnitude,
            };
            for j in 0..n {
                let q = phase * factor.q_matrix().view().get(i, j)?;
                out[i * n + j] = Complex {
                    re: number(q.re)?,
                    im: number(q.im)?,
                };
            }
        }
        Ok(())
    }
}
vector_sampling!(
    UnitaryGroup,
    Complex<S>,
    Complex {
        re: S::ZERO,
        im: S::ZERO
    }
);

/// Random `d × d` correlation matrices with prescribed eigenvalues `λ₁, …, λ_d ≥ 0`,
/// `Σ λᵢ = d`.
///
/// A draw forms `Q diag(λ) Qᵀ` with `Q` Haar distributed on O(d) and then
/// applies Givens rotations that keep the spectrum and set each diagonal entry
/// to 1 (Bendel and Mickey 1978, in the numerically stable form of Davies and
/// Higham 2000). The resulting law is not uniform over the correlation matrices
/// with that spectrum. A draw costs O(d³). Samples are stored row by row.
///
/// ```
/// use scioxide_random::{RandomCorrelation, SeedableRng, SplitMix64};
/// let law = RandomCorrelation::new(&[0.5_f64, 1.5])?;
/// let c = law.sample(&mut SplitMix64::from_seed(7))?;
/// // [[1, r], [r, 1]] has eigenvalues 1 +- |r|, so |r| = 0.5.
/// assert!((c[0] - 1.0).abs() < 1e-12 && (c[3] - 1.0).abs() < 1e-12);
/// assert!((c[1].abs() - 0.5).abs() < 1e-12);
/// assert_eq!(c[1], c[2]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. B. Bendel and M. R. Mickey, "Population correlation matrices for
///   sampling experiments", *Commun. Statist. Simul. Comput.* 7(2), 1978, 163–182.
/// - P. I. Davies and N. J. Higham, "Numerically stable generation of correlation
///   matrices and their factors", *BIT* 40(4), 2000, 640–651.
#[derive(Debug, Clone)]
pub struct RandomCorrelation<S: RealScalar> {
    eigenvalues: Vec<f64>,
    options: LawOptions<S>,
}
impl<S: RealScalar> RandomCorrelation<S> {
    /// Creates the law with the given eigenvalues.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `eigenvalues` is empty.
    /// - `NonFinite` if an eigenvalue is NaN or infinite.
    /// - `DomainError` if an eigenvalue is negative, or if `|Σ λᵢ − d| > 8 d ε`.
    /// - `ResourceLimit` or `ConvergenceFailure` if the `d × d` storage or O(d³)
    ///   work per draw exceeds the [work limits](crate#work-limits).
    pub fn new(eigenvalues: &[S]) -> Result<Self, SciOxideError> {
        Self::with_options(eigenvalues, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if `options` is invalid; see [`LawOptions`](crate::LawOptions).
    pub fn with_options(eigenvalues: &[S], options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        group_budget(
            eigenvalues.len(),
            options,
            scioxide_core::argument!("eigenvalues"),
        )?;
        let eigenvalues = values(eigenvalues)?;
        if eigenvalues.iter().any(|x| *x < 0.0) {
            return Err(crate::law::domain(
                scioxide_core::argument!("eigenvalues"),
                "correlation eigenvalues must be nonnegative",
            ));
        }
        let n = eigenvalues.len() as f64;
        let sum = eigenvalues.iter().sum::<f64>();
        if (sum - n).abs() > 8.0 * n * S::EPSILON.into_f64() {
            return Err(crate::law::domain(
                scioxide_core::argument!("eigenvalues"),
                "correlation eigenvalues must sum to dimension",
            ));
        }
        Ok(Self {
            eigenvalues,
            options,
        })
    }
    /// Returns the order `d` of a sample.
    pub fn dimension(&self) -> usize {
        self.eigenvalues.len()
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
}
impl<S: RealScalar> RowLaw<S, S> for RandomCorrelation<S> {
    fn width(&self) -> usize {
        self.dimension() * self.dimension()
    }
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [S],
        _: Scratch<'_, S>,
    ) -> Result<(), SciOxideError> {
        let n = self.dimension();
        let mut words =
            crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let q = orthogonal(&mut words, n, false)?;
        let mut scaled = q.view().try_to_matrix()?;
        for i in 0..n {
            for j in 0..n {
                let value = scaled.view().get(i, j)? * self.eigenvalues[j];
                scaled.view_mut().set(i, j, value)?;
            }
        }
        // Bendel-Mickey algorithm: start from Q diag(lambda) Q^T with Haar Q
        // (trace n, prescribed spectrum), then apply Givens rotations, which
        // preserve the spectrum, to set diagonal entries to 1 one at a time.
        // Each rotation pairs a diagonal entry above 1 with one below 1; the
        // angle solves the quadratic that makes the new (i, i) entry exactly 1.
        let mut a = scaled
            .view()
            .mul_mat(&q.view().transpose().try_to_matrix()?)?;
        for i in 0..n - 1 {
            let ai = a.view().get(i, i)? - 1.0;
            if ai == 0.0 {
                continue;
            }
            let mut opposite = None;
            for j in i + 1..n {
                if (a.view().get(j, j)? - 1.0) * ai < 0.0 {
                    opposite = Some(j);
                    break;
                }
            }
            let Some(j) = opposite else {
                continue;
            };
            let aj = a.view().get(j, j)? - 1.0;
            let cross = a.view().get(i, j)?;
            let (c, s) = correlation_rotation(ai, aj, cross)?;
            for k in 0..n {
                let (x, y) = (a.view().get(i, k)?, a.view().get(j, k)?);
                a.view_mut().set(i, k, c * x - s * y)?;
                a.view_mut().set(j, k, s * x + c * y)?;
            }
            for k in 0..n {
                let (x, y) = (a.view().get(k, i)?, a.view().get(k, j)?);
                a.view_mut().set(k, i, c * x - s * y)?;
                a.view_mut().set(k, j, s * x + c * y)?;
            }
        }
        for i in 0..n {
            if (a.view().get(i, i)? - 1.0).abs() > 128.0 * n as f64 * S::EPSILON.into_f64() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "correlation rotations did not resolve unit diagonal",
                ));
            }
            // Symmetrize to remove the rounding asymmetry left by the rotations.
            for j in 0..n {
                out[i * n + j] = number(0.5 * a.view().get(i, j)? + 0.5 * a.view().get(j, i)?)?;
            }
        }
        Ok(())
    }
}
vector_sampling!(RandomCorrelation, S, S::ZERO);
macro_rules! shape {
    ($name:ident) => {
        impl<S: RealScalar> $name<S> {
            /// Returns the shape `(d, d)` of a sample.
            pub fn shape(&self) -> (usize, usize) {
                (self.dimension(), self.dimension())
            }
        }
    };
}
shape!(OrthogonalGroup);
shape!(SpecialOrthogonalGroup);
shape!(UnitaryGroup);
shape!(RandomCorrelation);

// Bendel-Mickey rotation coefficients, checked before changing the matrix.
fn correlation_rotation(ai: f64, aj: f64, cross: f64) -> Result<(f64, f64), SciOxideError> {
    let root = (cross * cross - ai * aj).max(0.0).sqrt();
    let numerator = cross + root.copysign(cross);
    let norm = numerator.hypot(aj);
    if norm == 0.0 {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "correlation rotation is numerically singular",
        ));
    }
    finite(norm)?;
    Ok((aj / norm, numerator / norm))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    use crate::{SeedableRng, SplitMix64};
    use scioxide_core::NumericalIssue;
    #[test]
    fn correlation_rotation_and_diagonal_failures_are_breakdowns() {
        crate::error_assertions::numerical(
            correlation_rotation(0.0, 0.0, 0.0).unwrap_err(),
            NumericalIssue::Breakdown,
        );
        let (c, s) = correlation_rotation(1.0, -1.0, 0.0).unwrap();
        assert!((c * c + s * s - 1.0).abs() < 4.0 * f64::EPSILON);
        // Simulate a computed spectrum losing its trace invariant after validation.
        let law = RandomCorrelation::<f64> {
            eigenvalues: vec![2.0; 2],
            options: Default::default(),
        };
        crate::error_assertions::numerical(
            law.sample(&mut SplitMix64::from_seed(1)).unwrap_err(),
            NumericalIssue::Breakdown,
        );
    }
}
