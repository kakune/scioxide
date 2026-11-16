//! Companion and comrade matrices whose eigenvalues are polynomial roots.
use super::*;
use scioxide_linalg::Matrix;

/// Names the basis of a coefficient slice passed to
/// [`polynomial_root_matrix_into`].
///
/// The orthogonal variant borrows a prepared plan, which holds the
/// normalization and the affine map of the interval.
///
/// ```
/// use scioxide_poly::*;
///
/// let basis = PolynomialRootBasisRef::<f64>::Monomial;
/// assert!(basis.is_monomial());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolynomialRootBasisRef<'a, R: RealScalar> {
    /// The powers `1, x, x², …` of the polynomial's variable.
    Monomial,
    /// The members `P₀, P₁, …` of the plan's orthogonal family.
    Orthogonal(&'a crate::OrthogonalBasisPlan<R>),
}

impl<R: RealScalar> PolynomialRootBasisRef<'_, R> {
    /// Returns `true` for the monomial basis.
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// let basis = PolynomialRootBasisRef::<f64>::Monomial;
    /// assert!(basis.is_monomial());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn is_monomial(self) -> bool {
        matches!(self, Self::Monomial)
    }
}

/// Marks the bases whose roots can be computed without conversion to the
/// monomial basis: [`MonomialBasis`] and [`OrthogonalBasis`](crate::OrthogonalBasis).
///
/// The trait is sealed.
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let basis = p.basis().root_basis();
/// assert!(basis.is_monomial());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait PolynomialRootBasis<C: NumericScalar>: PolynomialBasis<C> {
    /// Returns the basis description used to build the root matrix.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0])?;
    /// assert!(p.basis().root_basis().is_monomial());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn root_basis(&self) -> PolynomialRootBasisRef<'_, C::Real>;
}

impl<C: NumericScalar> PolynomialRootBasis<C> for MonomialBasis<C> {
    fn root_basis(&self) -> PolynomialRootBasisRef<'_, C::Real> {
        PolynomialRootBasisRef::Monomial
    }
}

impl<C: NumericScalar> PolynomialRootBasis<C> for crate::OrthogonalBasis<C> {
    fn root_basis(&self) -> PolynomialRootBasisRef<'_, C::Real> {
        PolynomialRootBasisRef::Orthogonal(self.plan())
    }
}

impl<C: NumericScalar, B: PolynomialRootBasis<C>> ApproxPolynomial<C, B> {
    /// Computes the companion or comrade matrix whose eigenvalues are the roots
    /// of this polynomial.
    ///
    /// The degree `n` is the length of the stored coefficients minus one. The
    /// result is an `n × n` matrix in the polynomial's own variable: for an
    /// orthogonal basis on `[a, b]`, its eigenvalues lie in the same coordinate
    /// as `a` and `b`. The layout is described on
    /// [`polynomial_root_matrix_into`]. The entries that depend on the
    /// coefficients `c₀, …, cₙ` are ratios `cᵢ/cₙ`, so scaling all coefficients
    /// by a common factor does not change the matrix. A real ratio is one
    /// correctly rounded division when the quotient is normal or zero; other
    /// ratios are formed in extended exponent range and rounded once. The matrix
    /// is not balanced; the general eigensolver of `scioxide-linalg` balances it
    /// by default ([`GeneralEigenBalancing`](scioxide_linalg::GeneralEigenBalancing)).
    /// Building the matrix costs O(n²) time and memory.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the degree is zero or the leading coefficient is zero.
    /// - `NumericalInstability` if a matrix entry overflows.
    /// - `AllocationFailure` or `SizeOverflow` if the matrix cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // The monic Legendre polynomial of degree 2 mapped to [2, 3] has the roots
    /// // 2.5 ± 0.5/√3, so the trace of its comrade matrix is 5.
    /// let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::Legendre, 2.0_f64, 3.0)?
    ///     .with_normalization(BasisNormalization::Monic);
    /// let p = ApproxPolynomial::orthogonal(vec![0.0_f64, 0.0, 1.0], spec)?;
    /// let matrix = p.root_matrix()?;
    /// assert!((matrix.as_slice()[0] + matrix.as_slice()[3] - 5.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn root_matrix(&self) -> Result<Matrix<C>, SciOxideError> {
        let mut matrix = Matrix::zeros(self.degree(), self.degree())?;
        polynomial_root_matrix_into(
            self.basis().coefficients(),
            self.basis().root_basis(),
            &mut matrix,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(matrix)
    }
}

/// Writes the companion or comrade matrix of `p` into `matrix`.
///
/// `coefficients` holds `c₀, …, cₙ` in ascending order, with `cₙ ≠ 0`, and
/// `matrix` must be `n × n`. For the monomial basis, `p(x) = Σₖ cₖ xᵏ` and the
/// matrix is the companion matrix with ones on the subdiagonal and last column
/// `−cᵢ/cₙ`:
///
/// ```text
/// C[i, i−1] = 1,   C[i, n−1] = −cᵢ/cₙ,   i = 0, …, n−1.
/// ```
///
/// For an orthogonal basis with recurrence `Pₖ₊₁ = (aₖx + bₖ)Pₖ − cₖPₖ₋₁`,
/// column `k` holds the coefficients of
/// `x Pₖ = (Pₖ₊₁ − bₖPₖ + cₖPₖ₋₁)/aₖ` in `P₀, …, Pₙ₋₁`, and the last column
/// replaces `Pₙ` using `p`. This is the comrade matrix. Its eigenvalues are in
/// the polynomial's own variable, not in the reference interval of the family.
///
/// The entries are formed as described on [`ApproxPolynomial::root_matrix`].
/// Every entry is computed before `matrix` is written, so on error `matrix` is
/// unchanged. The call takes O(n²) time to clear the matrix and does not
/// allocate.
///
/// # Errors
///
/// - `InvalidInput` if `coefficients` is empty, has degree zero, or has a zero
///   leading coefficient, or if the orthogonal plan's maximum degree is less
///   than `n`.
/// - `NonFinite` if a coefficient is NaN or infinite.
/// - `DimensionMismatch` if `matrix` is not `n × n`.
/// - `NumericalInstability` if an entry overflows.
/// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::*;
///
/// // x² − 1 has the companion matrix [[0, 1], [1, 0]] (row-major).
/// let mut matrix = scioxide_linalg::Matrix::<f64>::zeros(2, 2)?;
/// let p = [-1.0_f64, 0.0, 1.0];
/// let cx = ExecutionContext::sequential();
/// polynomial_root_matrix_into(&p, PolynomialRootBasisRef::Monomial, &mut matrix, cx)?;
/// assert_eq!(matrix.as_slice(), &[0.0, 1.0, 1.0, 0.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_root_matrix_into<C: NumericScalar>(
    coefficients: &[C],
    basis: PolynomialRootBasisRef<'_, C::Real>,
    matrix: &mut Matrix<C>,
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    if coefficients.is_empty() {
        return Err(crate::util::empty_error(scioxide_core::argument!(
            "coefficients"
        )));
    }
    let degree = coefficients.len().saturating_sub(1);
    if degree == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("coefficients"),
            InvalidInputReason::OutOfRange,
            "a root matrix requires positive degree",
        ));
    }
    if matrix.rows() != degree || matrix.cols() != degree {
        return Err(SciOxideError::dimension_mismatch(
            [degree, degree],
            [matrix.rows(), matrix.cols()],
        ));
    }
    if coefficients[degree] == C::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("coefficients"),
            InvalidInputReason::OutOfRange,
            "leading coefficient must be nonzero",
        ));
    }
    if let PolynomialRootBasisRef::Orthogonal(plan) = basis {
        plan.validate_coefficients(coefficients)?;
    } else {
        validate_slice(coefficients)?;
    }
    // A first traversal forms every fallible entry without writing; the
    // publishing traversal recomputes the same O(degree) entries after the
    // O(degree^2) zero fill, so a failure never reaches the caller's matrix.
    root_matrix_entries(coefficients, basis, |_, _| {})?;
    context.check()?;
    let entries = matrix.as_mut_slice();
    entries.fill(C::ZERO);
    root_matrix_entries(coefficients, basis, |offset, value| entries[offset] = value)
}

/// Visits every nonzero root-matrix entry as `(row-major offset, value)`.
///
/// Callers have validated a nonempty, finite expansion of positive degree with
/// a nonzero leading coefficient. Repeated calls visit identical entries.
fn root_matrix_entries<C: NumericScalar>(
    coefficients: &[C],
    basis: PolynomialRootBasisRef<'_, C::Real>,
    mut visit: impl FnMut(usize, C),
) -> Result<(), SciOxideError> {
    let degree = coefficients.len() - 1;
    let leading = scaled_complex(coefficients[degree])?;
    match basis {
        PolynomialRootBasisRef::Monomial => {
            coefficients[..degree]
                .iter()
                .enumerate()
                .try_for_each(|(row, &coefficient)| {
                    if row > 0 {
                        visit(row * degree + row - 1, C::ONE);
                    }
                    let ratio = match native_ratio(coefficient, coefficients[degree]) {
                        Some(ratio) => C::from_real(ratio),
                        None => round_complex::<C>(scaled_complex(coefficient)?.div(leading)?)?,
                    };
                    visit(row * degree + degree - 1, -ratio);
                    Ok(())
                })
        }
        PolynomialRootBasisRef::Orthogonal(plan) => (0..degree).try_for_each(|column| {
            let raw @ (a, _, _) = plan.root_recurrence(column);
            let [lower, middle, upper] =
                multiplication_column(raw, column > 0, column + 1 < degree)?;
            if column + 1 < degree {
                visit(column * degree + column, C::from_real(middle));
                if column > 0 {
                    visit((column - 1) * degree + column, C::from_real(lower));
                }
                visit((column + 1) * degree + column, C::from_real(upper));
                return Ok(());
            }
            // The last column also eliminates `P_n` from every row.
            let denominator = leading.mul(scaled_complex(C::from_real(a))?)?;
            coefficients[..degree]
                .iter()
                .enumerate()
                .try_for_each(|(row, &coefficient)| {
                    let entry = C::from_real(if row == column {
                        middle
                    } else if row + 1 == column {
                        lower
                    } else {
                        C::Real::ZERO
                    });
                    let value =
                        match native_elimination(entry, coefficient, coefficients[degree], a) {
                            Some(value) => value,
                            None => round_complex::<C>(
                                scaled_complex(entry)?
                                    .sub(scaled_complex(coefficient)?.div(denominator)?)?,
                            )?,
                        };
                    visit(row * degree + column, value);
                    Ok(())
                })
        }),
    }
}

fn multiplication_column<R: RealScalar>(
    raw: (R, R, R),
    previous: bool,
    next: bool,
) -> Result<[R; 3], SciOxideError> {
    use crate::admission::{Extended, Native, RealFactors, candidate, extended};
    use crate::orthogonal::coefficients::multiplication;
    if let Some(values) = candidate(multiplication::<R, Native<RealFactors>>(
        R::ONE,
        raw,
        previous,
    ))? {
        return Ok(values);
    }
    let [lower, middle, upper] = extended(multiplication::<R, Extended>(
        ScaledValue::ONE,
        raw,
        previous,
    ))?;
    Ok([
        if previous { lower.round()? } else { R::ZERO },
        middle.round()?,
        if next { upper.round()? } else { R::ZERO },
    ])
}

// Real coefficients also occur in complex rootfind workspaces. Intermediate
// quotients must stay normal or exact zero: premature subnormal rounding could
// hide a representable result after the recurrence scale is applied. Both
// helpers return `None` to retry in extended range, which then rounds once.
#[inline]
fn native_normal<R: RealScalar>(value: R, source: R) -> Option<R> {
    (value.is_finite()
        && (value.abs() >= R::MIN_POSITIVE || (value == R::ZERO && source == R::ZERO)))
        .then_some(value)
}

/// Returns the correctly rounded real `numerator / denominator` when it is
/// normal or an exact zero.
#[inline]
fn native_ratio<C: NumericScalar>(numerator: C, denominator: C) -> Option<C::Real> {
    if numerator.imaginary_component() != C::Real::ZERO
        || denominator.imaginary_component() != C::Real::ZERO
    {
        return None;
    }
    native_normal(
        numerator.real_component() / denominator.real_component(),
        numerator.real_component(),
    )
}

#[inline]
fn native_elimination<C: NumericScalar>(
    entry: C,
    coefficient: C,
    leading: C,
    a: C::Real,
) -> Option<C> {
    let ratio = native_ratio(coefficient, leading)?;
    let correction = native_normal(ratio / a, ratio)?;
    let value = entry.real_component() - correction;
    value.is_finite().then(|| C::from_real(value))
}
