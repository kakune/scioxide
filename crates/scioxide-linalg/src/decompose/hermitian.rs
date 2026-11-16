//! How a square matrix defines the Hermitian input of Cholesky, LDLᴴ and
//! Hermitian eigenvalue problems.

use scioxide_core::Triangle;
use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledNorm, SciOxideError, Tolerance,
};

/// How a square matrix defines the Hermitian matrix that Cholesky, LDLᴴ and
/// Hermitian eigenvalue methods work on.
///
/// You always choose the interpretation; none is applied implicitly. With
/// [`HermitianInput::Triangle`] the other triangle is never read, so it may
/// hold any values, including NaN. For real scalars, Hermitian means symmetric.
///
/// Cholesky, `L D Lᴴ` and the dense Hermitian eigenvalue methods report these
/// input errors:
///
/// - `NonFinite` if an entry that is read is NaN or infinite.
/// - `InvalidInput` (`Inconsistent`) under [`HermitianInput::Exact`] if `A ≠ Aᴴ`,
///   which includes a diagonal entry with a nonzero imaginary part.
/// - `InvalidInput` (`OutOfRange`) under [`HermitianInput::Triangle`] if a
///   diagonal entry has a nonzero imaginary part, and under
///   [`HermitianInput::Projected`] if a [`HermitianProjectionPolicy::Within`]
///   tolerance is exceeded.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{HermitianInput, LdlDecomposition, Matrix};
/// let a = Matrix::from_vec(2, 2, vec![0.0_f64, 2.0, 2.0, 0.0])?;
/// let factor = LdlDecomposition::factor(&a, HermitianInput::Exact)?;
/// assert!(factor.projection_report().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum HermitianInput<R: RealScalar> {
    /// Uses `A` as given; it must satisfy `A = Aᴴ` exactly, with a real diagonal.
    Exact,
    /// Reads only this triangle and fills the other by conjugate symmetry. The
    /// diagonal must be real.
    Triangle(Triangle),
    /// Uses the Hermitian part `(A + Aᴴ) / 2`, subject to the given policy.
    Projected(HermitianProjectionPolicy<R>),
}

/// When [`HermitianInput::Projected`] accepts the Hermitian part `(A + Aᴴ) / 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum HermitianProjectionPolicy<R: RealScalar> {
    /// Always uses `(A + Aᴴ) / 2`.
    Always,
    /// Uses `(A + Aᴴ) / 2` only if `‖A − (A + Aᴴ)/2‖_F ≤ atol + rtol ‖A‖_F`;
    /// otherwise the method returns `InvalidInput` (`OutOfRange`).
    Within(Tolerance<R>),
}

/// Size of the change `A − (A + Aᴴ) / 2` made by [`HermitianInput::Projected`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct HermitianProjectionReport<R: RealScalar> {
    absolute_frobenius: R,
    relative_frobenius: R,
}

impl<R: RealScalar> HermitianProjectionReport<R> {
    /// Returns `‖A − (A + Aᴴ)/2‖_F`.
    pub const fn absolute_frobenius(self) -> R {
        self.absolute_frobenius
    }

    /// Returns `‖A − (A + Aᴴ)/2‖_F / max(‖A‖_F, 1)`.
    pub const fn relative_frobenius(self) -> R {
        self.relative_frobenius
    }
}

impl<S: NumericScalar> crate::Matrix<S> {
    /// Replaces the matrix by its Hermitian part `(A + Aᴴ) / 2`.
    ///
    /// Each entry is the midpoint of `a_ij` and `conj(a_ji)`, computed without
    /// overflow. The diagonal keeps only its real part. On error the matrix is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the matrix is not square.
    /// - `NonFinite` if an entry is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = scioxide_linalg::Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 4.0, 3.0])?;
    /// a.project_hermitian_in_place()?;
    /// assert_eq!(a.as_slice(), &[1.0, 3.0, 3.0, 3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn project_hermitian_in_place(&mut self) -> Result<(), SciOxideError> {
        let n = self.rows();
        if self.cols() != n {
            return Err(SciOxideError::dimension_mismatch([n, n], [n, self.cols()]));
        }
        for &value in self.as_slice() {
            crate::checked::validate_input(value, "Hermitian input")?;
        }
        for i in 0..n {
            let diagonal = self.as_slice()[i * n + i];
            self.as_mut_slice()[i * n + i] = S::from_real(diagonal.real_component());
            for j in 0..i {
                let value = projected(self.as_slice()[i * n + j], self.as_slice()[j * n + i])?;
                self.as_mut_slice()[i * n + j] = value;
                self.as_mut_slice()[j * n + i] = value.conj();
            }
        }
        Ok(())
    }
}

fn projected<S: NumericScalar>(original: S, other: S) -> Result<S, SciOxideError> {
    Ok(S::from_components(
        scioxide_core::checked_midpoint(original.real_component(), other.real_component())?,
        scioxide_core::checked_midpoint(
            original.imaginary_component(),
            -other.imaginary_component(),
        )?,
    ))
}

impl<S: RealScalar + NumericScalar<Real = S>> crate::MatrixView<'_, S> {
    /// Tests whether a real symmetric matrix has a positive diagonal and is
    /// strictly diagonally dominant by rows, a sufficient condition for positive
    /// definiteness.
    ///
    /// Row `i` passes when `a_ii > 0` and `r_i + (n + 8) ε (1 + r_i) < 1`, where
    /// `r_i = Σ_{j≠i} |a_ij| / a_ii` and `ε` is the machine epsilon. The margin
    /// covers rounding in `r_i`. `false` is inconclusive: a positive definite
    /// matrix need not be diagonally dominant. O(n²) time, no factorization.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the matrix is not square.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if the matrix is not exactly symmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = scioxide_linalg::MatrixView::from_slice(2, 2, &[2.0_f64, 0.5, 0.5, 1.0])?;
    /// assert!(a.is_positive_diagonally_dominant()?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn is_positive_diagonally_dominant(&self) -> Result<bool, SciOxideError> {
        let n = self.rows();
        if self.cols() != n {
            return Err(SciOxideError::dimension_mismatch([n, n], [n, self.cols()]));
        }
        // Every entry is checked before symmetry, so a NaN is reported as
        // non-finite wherever it is.
        for i in 0..n {
            for j in 0..n {
                crate::checked::validate_input(self.get(i, j)?, "symmetric input")?;
            }
        }
        for i in 0..n {
            for j in 0..i {
                if self.get(i, j)? != self.get(j, i)? {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("matrix"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "matrix must be symmetric",
                    ));
                }
            }
        }
        let size = n.checked_add(8).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("dominance_margin"))
        })?;
        let margin = S::try_from_usize_exact(size)? * S::EPSILON;
        let one = <S as AlgebraicScalar>::ONE;
        let zero = <S as AlgebraicScalar>::ZERO;
        for i in 0..n {
            let diagonal = self.get(i, i)?;
            if diagonal <= zero {
                return Ok(false);
            }
            let mut radius = zero;
            for j in 0..n {
                if i != j {
                    radius += NumericScalar::abs(self.get(i, j)?) / diagonal;
                }
            }
            if !radius.is_finite() || radius + margin * (one + radius) >= one {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Validate and copy the full interpreted matrix before factorization.
pub(crate) fn prepare<S: NumericScalar>(
    input: crate::MatrixView<'_, S>,
    interpretation: HermitianInput<S::Real>,
    output: &mut crate::MatrixViewMut<'_, S>,
    argument: scioxide_core::Argument,
) -> Result<Option<HermitianProjectionReport<S::Real>>, SciOxideError> {
    if stage_tiles(input, interpretation, output) {
        return Ok(None);
    }
    let n = input.rows();
    let mut input_norm = ScaledNorm::new();
    let mut change_norm = ScaledNorm::new();
    for row in 0..n {
        let columns = if matches!(interpretation, HermitianInput::Projected(_)) {
            n
        } else {
            row + 1
        };
        for col in 0..columns {
            let value = match interpretation {
                HermitianInput::Exact => {
                    let lower = input.get_element(row, col);
                    let upper = input.get_element(col, row);
                    crate::checked::validate_input(lower, "Hermitian input")?;
                    crate::checked::validate_input(upper, "Hermitian input")?;
                    if lower != upper.conj() {
                        return Err(SciOxideError::invalid_input(
                            argument,
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "input must be exactly Hermitian",
                        ));
                    }
                    lower
                }
                HermitianInput::Triangle(triangle) => {
                    let value = match triangle {
                        Triangle::Lower => input.get_element(row, col),
                        Triangle::Upper => input.get_element(col, row).conj(),
                    };
                    crate::checked::validate_input(value, "Hermitian input")?;
                    value
                }
                HermitianInput::Projected(_) => {
                    let original = input.get_element(row, col);
                    let other = input.get_element(col, row);
                    crate::checked::validate_input(original, "Hermitian input")?;
                    crate::checked::validate_input(other, "Hermitian input")?;
                    let projected = projected(original, other)?;
                    for (value, image) in [
                        (original.real_component(), projected.real_component()),
                        (
                            original.imaginary_component(),
                            projected.imaginary_component(),
                        ),
                    ] {
                        input_norm.add_magnitude(value.abs())?;
                        change_norm.add_magnitude(crate::checked::sub(value, image)?.abs())?;
                    }
                    projected
                }
            };
            if row == col && value.imaginary_component() != S::Real::ZERO {
                return Err(crate::checked::out_of_range(
                    argument,
                    "Hermitian diagonal must be real",
                ));
            }
            output.set_element(row, col, value);
            if !matches!(interpretation, HermitianInput::Projected(_)) {
                output.set_element(col, row, value.conj());
            }
        }
    }
    let HermitianInput::Projected(policy) = interpretation else {
        return Ok(None);
    };
    // Compare in extended range: the input norm may exceed the scalar range.
    let norm = extended(input_norm)?;
    let absolute = extended(change_norm)?;
    if let HermitianProjectionPolicy::Within(tolerance) = policy {
        let threshold = Scaled::from_value(tolerance.absolute())?
            .add(Scaled::from_value(tolerance.relative())?.mul(norm)?)?;
        if absolute.abs_cmp(threshold).is_gt() {
            return Err(crate::checked::out_of_range(
                argument,
                "projection change exceeds the caller tolerance",
            ));
        }
    }
    let denominator = if norm.abs_cmp(Scaled::ONE).is_lt() {
        Scaled::ONE
    } else {
        norm
    };
    Ok(Some(HermitianProjectionReport {
        absolute_frobenius: absolute.round()?,
        relative_frobenius: absolute.div(denominator)?.round()?,
    }))
}

fn extended<R: RealScalar>(norm: ScaledNorm<R>) -> Result<Scaled<R>, SciOxideError> {
    if norm.scale() == R::ZERO {
        return Ok(Scaled::ZERO);
    }
    Scaled::from_value(norm.scale())?.mul(Scaled::from_value(norm.sum_squares().sqrt())?)
}

/// Stage an exact or triangle interpretation tile by tile, so each entry and
/// its mirror stay in cache. Returns `false` for a projected interpretation or
/// when any entry fails validation; the sequential pass of [`prepare`] then
/// reports the first failure in row order.
fn stage_tiles<S: NumericScalar>(
    input: crate::MatrixView<'_, S>,
    interpretation: HermitianInput<S::Real>,
    output: &mut crate::MatrixViewMut<'_, S>,
) -> bool {
    const TILE: usize = 32;
    if matches!(interpretation, HermitianInput::Projected(_)) {
        return false;
    }
    let n = input.rows();
    // Mirrored entries pass through stack tiles, so every read and write of
    // the matrices runs along rows.
    let mut mirror = [[S::ZERO; TILE]; TILE];
    let mut staged = [[S::ZERO; TILE]; TILE];
    for rows in (0..n).step_by(TILE) {
        let height = TILE.min(n - rows);
        for cols in (0..=rows).step_by(TILE) {
            let width = TILE.min(n - cols);
            // Columns of tile row `i` inside the lower triangle.
            let span = |i: usize| width.min(rows + i + 1 - cols);
            if !matches!(interpretation, HermitianInput::Triangle(Triangle::Lower)) {
                for (j, mirrored) in mirror.iter_mut().enumerate().take(width) {
                    let first = (cols + j).saturating_sub(rows);
                    for (i, entry) in mirrored.iter_mut().enumerate().take(height).skip(first) {
                        *entry = input.get_element(cols + j, rows + i);
                    }
                }
            }
            for (i, row) in staged.iter_mut().enumerate().take(height) {
                for (j, entry) in row.iter_mut().enumerate().take(span(i)) {
                    let (r, c) = (rows + i, cols + j);
                    let value = match interpretation {
                        HermitianInput::Triangle(Triangle::Lower) => input.get_element(r, c),
                        HermitianInput::Triangle(Triangle::Upper) => mirror[j][i].conj(),
                        _ => {
                            let lower = input.get_element(r, c);
                            let upper = mirror[j][i];
                            if !upper.has_finite_components() || lower != upper.conj() {
                                return false;
                            }
                            lower
                        }
                    };
                    if !value.has_finite_components()
                        || (r == c && value.imaginary_component() != S::Real::ZERO)
                    {
                        return false;
                    }
                    *entry = value;
                }
            }
            for (i, row) in staged.iter().enumerate().take(height) {
                for (j, &value) in row.iter().enumerate().take(span(i)) {
                    output.set_element(rows + i, cols + j, value);
                }
            }
            for j in 0..width {
                let first = (cols + j).saturating_sub(rows);
                for (i, row) in staged.iter().enumerate().take(height).skip(first) {
                    output.set_element(cols + j, rows + i, row[j].conj());
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{Complex, SciOxideErrorKind};

    fn entry(row: usize, col: usize) -> Complex<f64> {
        let (low, high) = (row.min(col) as f64, row.max(col) as f64);
        if row == col {
            Complex::new(low + 1.0, 0.0)
        } else if row > col {
            Complex::new(low - high / 3.0, high / 7.0 - low)
        } else {
            Complex::new(low - high / 3.0, low - high / 7.0)
        }
    }

    fn stage(
        input: &[Complex<f64>],
        n: usize,
        interpretation: HermitianInput<f64>,
    ) -> Result<alloc::vec::Vec<Complex<f64>>, SciOxideError> {
        let mut output = alloc::vec![Complex::new(f64::NAN, 0.0); n * n];
        prepare(
            crate::MatrixView::from_slice(n, n, input)?,
            interpretation,
            &mut crate::MatrixViewMut::from_slice(n, n, &mut output)?,
            scioxide_core::argument!("a"),
        )?;
        Ok(output)
    }

    #[test]
    fn tiled_staging_matches_every_interpretation_across_tiles() -> Result<(), SciOxideError> {
        let n = 70;
        let exact: alloc::vec::Vec<_> = (0..n * n).map(|i| entry(i / n, i % n)).collect();
        for (interpretation, poisoned) in [
            (HermitianInput::Exact, None),
            (HermitianInput::Triangle(Triangle::Lower), Some((3, 65))),
            (HermitianInput::Triangle(Triangle::Upper), Some((65, 3))),
        ] {
            let mut input = exact.clone();
            if let Some((row, col)) = poisoned {
                input[row * n + col] = Complex::new(f64::NAN, f64::INFINITY);
            }
            assert_eq!(stage(&input, n, interpretation)?, exact);
        }
        Ok(())
    }

    #[test]
    fn the_first_failure_in_row_order_is_reported() -> Result<(), SciOxideError> {
        let n = 70;
        let exact: alloc::vec::Vec<_> = (0..n * n).map(|i| entry(i / n, i % n)).collect();
        // An asymmetric pair in row 40 precedes a non-finite entry in row 60,
        // although tile order visits the non-finite entry first.
        let mut input = exact.clone();
        input[40 * n + 35] = Complex::new(5.0, 0.0);
        input[60 * n + 2] = Complex::new(f64::NAN, 0.0);
        let error = stage(&input, n, HermitianInput::Exact).err();
        assert_eq!(
            error.map(|e| e.kind()),
            Some(SciOxideErrorKind::InvalidInput)
        );
        let mut input = exact.clone();
        input[50 * n + 50] = Complex::new(1.0, 1.0);
        let error = stage(&input, n, HermitianInput::Triangle(Triangle::Lower)).err();
        assert_eq!(
            error.map(|e| e.kind()),
            Some(SciOxideErrorKind::InvalidInput)
        );
        let mut input = exact;
        input[2 * n + 64] = Complex::new(f64::INFINITY, 0.0);
        let error = stage(&input, n, HermitianInput::Triangle(Triangle::Upper)).err();
        assert_eq!(error.map(|e| e.kind()), Some(SciOxideErrorKind::NonFinite));
        Ok(())
    }
}
