//! Vandermonde matrices in monomial and orthogonal bases.
use crate::util::{minimum_len, validate_slice};
use crate::{OrthogonalBasisPlan, PolynomialScratchRequirements};
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, MatrixViewMut};

#[derive(Debug, Clone, PartialEq)]
enum Basis<R: RealScalar> {
    Monomial(usize),
    Orthogonal(OrthogonalBasisPlan<R>),
}

/// Builds Vandermonde matrices in the monomial basis or in a prepared
/// orthogonal basis.
///
/// The value holds the basis and its degree; it precomputes nothing beyond the
/// [`OrthogonalBasisPlan`] it is given. [`vandermonde`] is the one-shot form for
/// the monomial basis.
///
/// For points `x₀, …, x_(m−1)` and basis members `Q₀, …, Q_d`, the matrix is
/// the `m × (d + 1)` matrix with entries `V[i, j] = Qⱼ(xᵢ)`. In the monomial
/// basis `Qⱼ(x) = xʲ`; in an orthogonal basis the members, their normalization
/// and the affine map come from the [`OrthogonalBasisPlan`]. Each row is filled
/// by the three-term recurrence of the basis in O(d) time, so a matrix costs
/// O(m d) time. An orthogonal basis is never expanded into powers of `x`.
///
/// ```
/// use scioxide_poly::Vandermonde;
///
/// // Rows [1, x, x²] at x = 0 and x = 2, stored row by row.
/// let matrix = Vandermonde::<f64>::monomial(2)?.matrix(&[0.0_f64, 2.0])?;
/// let expected = [1.0_f64, 0.0, 0.0, 1.0, 2.0, 4.0];
/// assert!(
///     matrix
///         .as_slice()
///         .iter()
///         .zip(expected)
///         .all(|(a, b)| (a - b).abs() < 1e-14)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vandermonde<R: RealScalar> {
    basis: Basis<R>,
}

/// Returns the `m × (degree + 1)` Vandermonde matrix `V[i, j] = xᵢʲ` at the
/// points `x₀, …, x_(m−1)`, one row per point.
///
/// Each row is filled by repeated multiplication in O(degree) operations, so
/// the matrix costs O(m · degree). This is
/// `Vandermonde::monomial(degree)?.matrix(points)`; use [`Vandermonde`] for an
/// orthogonal basis or to write into a strided output.
///
/// # Errors
///
/// - `NonFinite` if a point is NaN or infinite.
/// - `NumericalInstability` if a power overflows.
/// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be stored.
///
/// # Examples
///
/// ```
/// use scioxide_poly::vandermonde;
///
/// // Rows [1, x, x²] at x = 0 and x = 2, stored row by row.
/// let matrix = vandermonde(&[0.0_f64, 2.0], 2)?;
/// let expected = [1.0_f64, 0.0, 0.0, 1.0, 2.0, 4.0];
/// assert!(
///     matrix
///         .as_slice()
///         .iter()
///         .zip(expected)
///         .all(|(a, b)| (a - b).abs() < 1e-14)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn vandermonde<C: NumericScalar>(
    points: &[C],
    degree: usize,
) -> Result<Matrix<C>, SciOxideError> {
    Vandermonde::<C::Real>::monomial(degree)?.matrix(points)
}

impl<R: RealScalar> Vandermonde<R> {
    /// Creates the builder for the columns `1, x, …, x^degree`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `degree + 1` overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::Vandermonde;
    ///
    /// let builder = Vandermonde::<f64>::monomial(2)?;
    /// let matrix = builder.matrix(&[0.0_f64, 2.0])?;
    /// let expected = [1.0_f64, 0.0, 0.0, 1.0, 2.0, 4.0];
    /// assert!(
    ///     matrix
    ///         .as_slice()
    ///         .iter()
    ///         .zip(expected)
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn monomial(degree: usize) -> Result<Self, SciOxideError> {
        checked_add(degree, 1, scioxide_core::quantity!("vandermonde_columns"))?;
        Ok(Self {
            basis: Basis::Monomial(degree),
        })
    }
    /// Creates the builder whose columns are the members of an orthogonal basis,
    /// of degree 0 through `plan.degree()`.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily, Vandermonde};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// // [T₀, T₁, T₂] at x = 0.5 is [1, 0.5, −0.5].
    /// let matrix = Vandermonde::orthogonal(plan).matrix(&[0.5_f64])?;
    /// let expected = [1.0_f64, 0.5, -0.5];
    /// assert!(
    ///     matrix
    ///         .as_slice()
    ///         .iter()
    ///         .zip(expected)
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn orthogonal(plan: OrthogonalBasisPlan<R>) -> Self {
        Self {
            basis: Basis::Orthogonal(plan),
        }
    }
    /// Returns the highest degree `d` of a column; the matrix has `d + 1`
    /// columns.
    ///
    /// ```
    /// use scioxide_poly::Vandermonde;
    ///
    /// let builder = Vandermonde::<f64>::monomial(2)?;
    /// assert_eq!(builder.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn degree(&self) -> usize {
        match &self.basis {
            Basis::Monomial(d) => *d,
            Basis::Orthogonal(p) => p.degree(),
        }
    }
    /// Returns the scratch length that [`matrix_into`](Self::matrix_into) needs
    /// for `points` rows: one entry per matrix element. Longer scratch is
    /// accepted.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `points · (degree() + 1)` overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::Vandermonde;
    ///
    /// let builder = Vandermonde::<f64>::monomial(2)?;
    /// assert_eq!(builder.workspace_requirements(2)?.scratch_len(), 6);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        points: usize,
    ) -> Result<PolynomialScratchRequirements, SciOxideError> {
        Ok(PolynomialScratchRequirements::new(checked_mul(
            points,
            self.degree() + 1,
            scioxide_core::quantity!("vandermonde_elements"),
        )?))
    }
    /// Returns the Vandermonde matrix at `points` as a new
    /// [`Matrix`](scioxide_linalg::Matrix), one row per point.
    ///
    /// An empty `points` gives a matrix with no rows.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a point is NaN or infinite.
    /// - `NumericalInstability` if a basis value overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::Vandermonde;
    ///
    /// let builder = Vandermonde::<f64>::monomial(2)?;
    /// let matrix = builder.matrix(&[0.0_f64, 2.0])?;
    /// let expected = [1.0_f64, 0.0, 0.0, 1.0, 2.0, 4.0];
    /// assert!(
    ///     matrix
    ///         .as_slice()
    ///         .iter()
    ///         .zip(expected)
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn matrix<C: NumericScalar<Real = R>>(
        &self,
        points: &[C],
    ) -> Result<Matrix<C>, SciOxideError> {
        let len = self.workspace_requirements(points.len())?.scratch_len();
        let mut scratch = try_filled(
            len,
            C::ZERO,
            scioxide_core::quantity!("vandermonde_scratch"),
        )?;
        let mut output = Matrix::zeros(points.len(), self.degree() + 1)?;
        self.matrix_into(
            points,
            output.view_mut(),
            &mut scratch,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Writes the Vandermonde matrix at `points` into `output`, which may be a
    /// strided view.
    ///
    /// `output` must be `points.len() × (degree() + 1)`, and `scratch` needs
    /// [`workspace_requirements(points.len())`](Self::workspace_requirements)
    /// entries. The method does not allocate. `context` is checked for
    /// cancellation before each row and before writing. On error `output` is
    /// unchanged; see [Output on error](scioxide_core#output-on-error).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` has the wrong shape or `scratch` is too
    ///   short.
    /// - `NonFinite` if a point is NaN or infinite.
    /// - `NumericalInstability` if a basis value overflows.
    /// - `SizeOverflow` if the element count overflows `usize`.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::Vandermonde;
    ///
    /// let builder = Vandermonde::<f64>::monomial(2)?;
    /// let mut matrix = scioxide_linalg::Matrix::zeros(2, 3)?;
    /// let mut scratch = vec![0.0_f64; builder.workspace_requirements(2)?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// builder.matrix_into(&[0.0_f64, 2.0], matrix.view_mut(), &mut scratch, cx)?;
    /// let expected = [1.0_f64, 0.0, 0.0, 1.0, 2.0, 4.0];
    /// assert!(
    ///     matrix
    ///         .as_slice()
    ///         .iter()
    ///         .zip(expected)
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn matrix_into<C: NumericScalar<Real = R>>(
        &self,
        points: &[C],
        mut output: MatrixViewMut<'_, C>,
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let columns = self.degree() + 1;
        if output.rows() != points.len() || output.cols() != columns {
            return Err(SciOxideError::dimension_mismatch(
                [points.len(), columns],
                [output.rows(), output.cols()],
            ));
        }
        let len = self.workspace_requirements(points.len())?.scratch_len();
        minimum_len(len, scratch.len())?;
        validate_slice(points)?;
        for (&point, row) in points.iter().zip(scratch[..len].chunks_exact_mut(columns)) {
            context.check()?;
            match &self.basis {
                Basis::Monomial(_) => crate::orthogonal::recurrence::members_into(
                    point,
                    R::ONE,
                    |_| (R::ONE, R::ZERO, R::ZERO),
                    row,
                )?,
                Basis::Orthogonal(plan) => crate::orthogonal::recurrence::members_into(
                    point,
                    plan.initial_member(),
                    |k| plan.root_recurrence(k),
                    row,
                )?,
            }
        }
        context.check()?;
        for (index, &value) in scratch[..len].iter().enumerate() {
            // Shape was validated before staging; these indices cannot fail.
            output.set(index / columns, index % columns, value)?;
        }
        Ok(())
    }
}
