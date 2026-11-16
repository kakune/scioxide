//! Basis conversion by Clenshaw's recurrence in coefficient space, with an
//! extended-range retry of the whole conversion.
use super::basis::{
    BasisNormalization, OrthogonalBasis, OrthogonalBasisPlan, OrthogonalBasisSpec, lengths,
};
use crate::PolynomialBasis;
use crate::basis::conversion_result;
use crate::util::{minimum_len, validate_slice};
use crate::{
    ApproxPolynomial, OrthogonalFamily, PolynomialApproximationPolicy, PolynomialConversion,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AlgebraicScalar, LinearOperator, NumericScalar, RealScalar, Scaled, SciOxideError,
};

use crate::admission::{
    Arithmetic, ArithmeticError, ArithmeticResult, Extended, Native, RealFactors, candidate,
    extended,
};

/// Staging entries and real factors are range-checked before each operation.
fn apply_row<C: NumericScalar, A: Arithmetic<C, Projected = C>>(
    operator: &OrthogonalTransform<C::Real>,
    input: &[C],
    row: usize,
) -> ArithmeticResult<C> {
    let mut sum = A::scalar(C::ZERO)?;
    for (column, &coefficient) in input.iter().enumerate().skip(row) {
        let entry = operator.entries[column * operator.rows + row];
        if entry != C::Real::ZERO && coefficient != C::ZERO {
            sum = A::checked(A::add(
                sum,
                A::checked(A::mul(
                    A::checked_scalar(coefficient)?,
                    A::checked_real(entry)?,
                )?)?,
            )?)?;
        }
    }
    A::project(sum)
}

fn term<R: RealScalar>(plan: Option<&OrthogonalBasisPlan<R>>, n: usize) -> (R, R, R) {
    plan.map_or((R::ONE, R::ZERO, R::ZERO), |plan| plan.recurrence[n])
}

pub(super) fn coefficients<C: NumericScalar>(
    input: &[C],
    source: Option<&OrthogonalBasisPlan<C::Real>>,
    target: Option<&OrthogonalBasisPlan<C::Real>>,
) -> Result<Vec<C>, SciOxideError> {
    if source.is_some_and(|source| target.is_some_and(|target| source.spec == target.spec)) {
        return scioxide_core::backend::try_copy_slice(
            input,
            scioxide_core::quantity!("conversion_coefficients"),
        );
    }
    // Power-to-Chebyshev coordinates can span roughly one binary exponent per
    // degree. Center this spread in the scalar's normal range using an exact
    // power-of-two frame. Admission still checks every operation and restarts
    // in extended range if the input's own dynamic range needs more headroom.
    let frame = if source.is_none()
        && target.is_some_and(|p| {
            p.spec.family == OrthogonalFamily::ChebyshevT
                && p.spec.normalization == BasisNormalization::Classical
                && p.spec.shift == C::Real::ZERO
                && p.spec.scale == C::Real::ONE
        }) {
        let limit = Scaled::from_value(C::Real::MAX)?.exponent() - 2;
        let degree = i64::try_from(input.len() - 1).unwrap_or(i64::MAX);
        let desired = (degree / 2).min(limit / 2);
        let shift = (desired - crate::util::coefficient_exponent(input)?).clamp(-limit, limit);
        Scaled::from_parts(C::Real::ONE, shift)?.round()?
    } else {
        C::Real::ONE
    };
    match clenshaw::<C, Native<RealFactors>>(input, source, target, frame) {
        Ok(output) => return Ok(output),
        Err(ArithmeticError::Numerical(error)) => return Err(error),
        Err(ArithmeticError::Retry) => {}
    }
    match clenshaw::<C, Extended>(input, source, target, C::Real::ONE) {
        Ok(output) => Ok(output),
        Err(ArithmeticError::Numerical(error)) => Err(error),
        Err(ArithmeticError::Retry) => {
            unreachable!("extended coefficient arithmetic has no native admission")
        }
    }
}

fn clenshaw<C: NumericScalar, A: Arithmetic<C, Projected = C>>(
    input: &[C],
    source: Option<&OrthogonalBasisPlan<C::Real>>,
    target: Option<&OrthogonalBasisPlan<C::Real>>,
    frame: C::Real,
) -> ArithmeticResult<Vec<C>> {
    let size = input.len();
    let mut next = try_filled(
        size,
        A::scalar(C::ZERO)?,
        scioxide_core::quantity!("conversion_coordinates"),
    )?;
    let mut previous = try_filled(
        size,
        A::scalar(C::ZERO)?,
        scioxide_core::quantity!("conversion_coordinates"),
    )?;
    let mut current = try_filled(
        size,
        A::scalar(C::ZERO)?,
        scioxide_core::quantity!("conversion_coordinates"),
    )?;
    let target_p0 = target.map_or(C::Real::ONE, |p| p.p0);
    let mut next_c = C::Real::ZERO;
    for (k, &coefficient) in input.iter().enumerate().rev() {
        current.fill(A::scalar(C::ZERO)?);
        let (a, b, c) = term(source, k);
        let active = size - k - 1;
        for (j, &value) in next.iter().enumerate().take(active) {
            let value = A::checked(A::mul(value, A::checked_real(a)?)?)?;
            let [lower, middle, upper] =
                super::coefficients::multiplication::<C, A>(value, term(target, j), j > 0)?;
            current[j] = A::checked(A::add(current[j], middle)?)?;
            current[j + 1] = A::checked(A::add(current[j + 1], upper)?)?;
            if j > 0 {
                current[j - 1] = A::checked(A::add(current[j - 1], lower)?)?;
            }
        }
        for ((out, &next), &previous) in current
            .iter_mut()
            .zip(&next)
            .zip(&previous)
            .take(active + 1)
        {
            *out = A::checked(A::sub(
                A::checked(A::add(
                    *out,
                    A::checked(A::mul(next, A::checked_real(b)?)?)?,
                )?)?,
                A::checked(A::mul(previous, A::checked_real(next_c)?)?)?,
            )?)?;
        }
        current[0] = A::checked(A::add(
            current[0],
            A::checked(A::div(
                A::checked(A::mul(
                    A::checked_scalar(coefficient)?,
                    A::checked_real(frame)?,
                )?)?,
                A::checked_real(target_p0)?,
            )?)?,
        )?)?;
        core::mem::swap(&mut previous, &mut next);
        core::mem::swap(&mut next, &mut current);
        next_c = c;
    }
    let factor = Scaled::from_value(source.map_or(C::Real::ONE, |p| p.p0))?
        .div(Scaled::from_value(frame)?)?;
    let mut output = try_with_capacity(size, scioxide_core::quantity!("conversion_coefficients"))?;
    for value in next {
        output.push(A::finish(value, factor)?);
    }
    Ok(output)
}

impl<C: NumericScalar> ApproxPolynomial<C, OrthogonalBasis<C>> {
    /// Converts to another orthogonal basis with an operator from
    /// [`OrthogonalBasisPlan::conversion_operator`].
    ///
    /// The result is the same as [`to_orthogonal`](Self::to_orthogonal),
    /// including the conversion diagnostics, but multiplies by the stored
    /// matrix in O(n²) time instead of building the connection coefficients
    /// again. The operator's source must be this polynomial's basis and its
    /// size must equal the number of coefficients.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the operator has a different number of columns
    ///   than this polynomial has coefficients.
    /// - `InvalidInput` (`Inconsistent`) if the operator's source is a different
    ///   basis, or its destination is the monomial basis.
    /// - `NumericalInstability` if a converted coefficient overflows.
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// let operator = p.basis().plan().conversion_operator(p.basis().plan())?;
    /// let result = p.to_orthogonal_with_operator(&operator)?;
    /// // T₀ + 2T₁ + T₂ = 2x² + 2x at x = 0.5 is 1.5.
    /// assert!((result.polynomial().evaluate(0.5)? - 1.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_orthogonal_with_operator(
        &self,
        operator: &OrthogonalTransform<C::Real>,
    ) -> Result<PolynomialConversion<C, OrthogonalBasis<C>>, SciOxideError> {
        operator.validate_source(
            Some(self.basis().plan.spec),
            self.basis().coefficients.len(),
        )?;
        operator.orthogonal_result(self.basis().coefficients(), self.policy())
    }
    /// Converts to the monomial basis with an operator from
    /// [`OrthogonalBasisPlan::monomial_operator`].
    ///
    /// The result carries the same coefficient-growth diagnostics as
    /// `to_monomial`. Applying the stored matrix costs O(n²) time.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the operator has a different number of columns
    ///   than this polynomial has coefficients.
    /// - `InvalidInput` (`Inconsistent`) if the operator's source is a different
    ///   basis, or its destination is not the monomial basis.
    /// - `NumericalInstability` if a converted coefficient overflows.
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// let operator = p.basis().plan().monomial_operator()?;
    /// let result = p.to_monomial_with_operator(&operator)?;
    /// // T₀ + 2T₁ + T₂ = 2x² + 2x at x = 0.5 is 1.5.
    /// assert!((result.polynomial().evaluate(0.5)? - 1.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_monomial_with_operator(
        &self,
        operator: &OrthogonalTransform<C::Real>,
    ) -> Result<PolynomialConversion<C, crate::MonomialBasis<C>>, SciOxideError> {
        operator.validate_source(
            Some(self.basis().plan.spec),
            self.basis().coefficients.len(),
        )?;
        if operator.target.is_some() {
            return Err(mismatch(
                scioxide_core::argument!("operator"),
                "conversion destination must be monomial",
            ));
        }
        let coefficients = operator.apply(self.basis().coefficients())?;
        conversion_result(
            self.basis().coefficients(),
            ApproxPolynomial::monomial_with_options(coefficients, self.policy())?,
        )
    }
}
impl<C: NumericScalar> ApproxPolynomial<C, crate::MonomialBasis<C>> {
    /// Converts from powers of `x` to an orthogonal basis with an operator from
    /// [`OrthogonalBasisPlan::from_monomial_operator`].
    ///
    /// The result is the same as [`to_orthogonal`](Self::to_orthogonal),
    /// including the conversion diagnostics. Applying the stored matrix costs
    /// O(n²) time.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the operator has a different number of columns
    ///   than this polynomial has coefficients.
    /// - `InvalidInput` (`Inconsistent`) if the operator's source is not the
    ///   monomial basis, or its destination is the monomial basis.
    /// - `NumericalInstability` if a converted coefficient overflows.
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily, PolynomialBasis};
    ///
    /// // 1 + 2x + 3x² = 2.5 T₀ + 2 T₁ + 1.5 T₂, since x² = (T₀ + T₂)/2.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 3.0])?;
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let operator = spec
    ///     .prepare(2, Default::default())?
    ///     .from_monomial_operator()?;
    /// let result = p.to_orthogonal_with_operator(&operator)?;
    /// let c = result.polynomial().basis().coefficients();
    /// assert!(
    ///     c.iter()
    ///         .zip([2.5_f64, 2.0, 1.5])
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_orthogonal_with_operator(
        &self,
        operator: &OrthogonalTransform<C::Real>,
    ) -> Result<PolynomialConversion<C, OrthogonalBasis<C>>, SciOxideError> {
        operator.validate_source(None, self.basis().coefficients().len())?;
        operator.orthogonal_result(self.basis().coefficients(), self.policy())
    }
}
impl<R: RealScalar> OrthogonalTransform<R> {
    fn validate_source(
        &self,
        spec: Option<OrthogonalBasisSpec<R>>,
        count: usize,
    ) -> Result<(), SciOxideError> {
        lengths(self.cols, count)?;
        if self.source != spec {
            return Err(mismatch(
                scioxide_core::argument!("operator"),
                "conversion source basis does not match",
            ));
        }
        Ok(())
    }
    fn orthogonal_result<C: NumericScalar<Real = R>>(
        &self,
        input: &[C],
        policy: PolynomialApproximationPolicy<R>,
    ) -> Result<PolynomialConversion<C, OrthogonalBasis<C>>, SciOxideError> {
        let target = self.target.as_ref().ok_or_else(|| {
            mismatch(
                scioxide_core::argument!("operator"),
                "conversion destination must be orthogonal",
            )
        })?;
        let coefficients = self.apply(input)?;
        let plan = target.resized(coefficients.len() - 1)?;
        conversion_result(
            input,
            ApproxPolynomial::from_basis(OrthogonalBasis::new(coefficients, plan), policy)?,
        )
    }
}

/// Stores the matrix of a change of polynomial basis.
///
/// Column `j` holds the coefficients, in the destination basis, of the degree-`j`
/// member of the source basis. Either basis may be an orthogonal basis or the
/// monomial basis. The matrix is upper triangular, stored column-major in
/// O(n²) memory for `n` coefficients.
///
/// The type implements [`LinearOperator`] for real and complex coefficients.
/// `apply_into` takes O(n²) time, needs
/// [`apply_workspace_len`](LinearOperator::apply_workspace_len) `= n` scratch
/// entries, and leaves the output unchanged on error. It returns
/// `DimensionMismatch` for wrong lengths, `NonFinite` for a NaN or infinite
/// input, and `NumericalInstability` if an output overflows. An operator also
/// converts whole polynomials through `to_orthogonal_with_operator` and
/// `to_monomial_with_operator`, which add the conversion diagnostics.
///
/// ```
/// use scioxide_core::LinearOperator;
/// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
///
/// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
/// let operator = spec.prepare(2, Default::default())?.monomial_operator()?;
/// assert_eq!(LinearOperator::<f64>::rows(&operator), 3);
/// assert_eq!(LinearOperator::<f64>::cols(&operator), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OrthogonalTransform<R: RealScalar> {
    rows: usize,
    cols: usize,
    entries: Vec<R>,
    source: Option<OrthogonalBasisSpec<R>>,
    target: Option<OrthogonalBasisPlan<R>>,
}
impl<R: RealScalar> OrthogonalTransform<R> {
    /// Returns the matrix entries in column-major order.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// let operator = plan.monomial_operator()?;
    /// assert_eq!(operator.entries().len(), 9);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn entries(&self) -> &[R] {
        &self.entries
    }
    /// Multiplies the matrix by `coefficients` and returns the converted
    /// coefficients in a new vector.
    ///
    /// This allocates the output and scratch, then calls `apply_into` of the
    /// [`LinearOperator`] implementation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `coefficients` does not have one entry per
    ///   column.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if a converted coefficient overflows.
    /// - `AllocationFailure` if the output cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let operator = spec.prepare(2, Default::default())?.monomial_operator()?;
    /// // T₂(x) = −1 + 2x².
    /// let result = operator.apply(&[0.0_f64, 0.0, 1.0])?;
    /// assert!(
    ///     result
    ///         .iter()
    ///         .zip([-1.0_f64, 0.0, 2.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply<C: NumericScalar<Real = R>>(
        &self,
        coefficients: &[C],
    ) -> Result<Vec<C>, SciOxideError> {
        let mut output = try_filled(
            self.rows,
            C::ZERO,
            scioxide_core::quantity!("transform_output"),
        )?;
        let mut scratch = try_filled(
            self.rows,
            C::ZERO,
            scioxide_core::quantity!("transform_workspace"),
        )?;
        self.apply_into(coefficients, &mut output, &mut scratch)?;
        Ok(output)
    }
}
impl<C: NumericScalar> LinearOperator<C> for OrthogonalTransform<C::Real> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows)
    }
    fn apply_into(
        &self,
        input: &[C],
        output: &mut [C],
        workspace: &mut [C],
    ) -> Result<(), SciOxideError> {
        lengths(self.cols, input.len())?;
        lengths(self.rows, output.len())?;
        minimum_len(self.rows, workspace.len())?;
        validate_slice(input)?;
        // Rows are independent: only an inadmissible dot product restarts in
        // extended range. Publication remains atomic across every row.
        for (row, out) in workspace[..self.rows].iter_mut().enumerate() {
            *out = match candidate(apply_row::<C, Native<RealFactors>>(self, input, row))? {
                Some(value) => value,
                None => extended(apply_row::<C, Extended>(self, input, row))?,
            };
        }
        output.copy_from_slice(&workspace[..self.rows]);
        Ok(())
    }
}

impl<R: RealScalar> OrthogonalBasisPlan<R> {
    /// Computes the connection coefficients from this basis to `target`, up to
    /// this plan's degree.
    ///
    /// Each column is generated from the three-term recurrence of this basis,
    /// with multiplication by `x` applied as the tridiagonal operator of the
    /// target basis. This takes O(n²) time and storage.
    /// When both bases are the same, the result is the identity.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `target.degree() < self.degree()`.
    /// - `NumericalInstability` if a coefficient overflows even in extended
    ///   range.
    /// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be stored.
    ///
    /// ```
    /// use scioxide_core::LinearOperator;
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// let operator = plan.conversion_operator(&plan)?;
    /// assert_eq!(LinearOperator::<f64>::rows(&operator), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn conversion_operator(
        &self,
        target: &Self,
    ) -> Result<OrthogonalTransform<R>, SciOxideError> {
        if target.degree() < self.degree() {
            return Err(mismatch(
                scioxide_core::argument!("target"),
                "conversion target degree is too small",
            ));
        }
        connection(
            self.degree(),
            self.p0,
            |n| self.recurrence[n],
            target.p0,
            |n| target.recurrence[n],
            Some(self.spec),
            Some(target.resized(self.degree())?),
        )
    }
    /// Computes the matrix that converts monomial coefficients into this basis,
    /// up to the plan's degree.
    ///
    /// Column `j` holds the expansion of `xʲ` in this basis.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a coefficient overflows even in extended
    ///   range.
    /// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let operator = spec
    ///     .prepare(2, Default::default())?
    ///     .from_monomial_operator()?;
    /// assert_eq!(operator.entries().len(), 9);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_monomial_operator(&self) -> Result<OrthogonalTransform<R>, SciOxideError> {
        connection(
            self.degree(),
            R::ONE,
            |_| (R::ONE, R::ZERO, R::ZERO),
            self.p0,
            |n| self.recurrence[n],
            None,
            Some(self.resized(self.degree())?),
        )
    }
    /// Computes the matrix that converts coefficients in this basis into
    /// monomial coefficients, up to the plan's degree.
    ///
    /// Column `j` holds the power-series coefficients of the degree-`j` member
    /// `Qⱼ(x)` in the physical variable `x`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a coefficient overflows even in extended
    ///   range.
    /// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let operator = spec.prepare(2, Default::default())?.monomial_operator()?;
    /// assert_eq!(operator.entries().len(), 9);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn monomial_operator(&self) -> Result<OrthogonalTransform<R>, SciOxideError> {
        connection(
            self.degree(),
            self.p0,
            |n| self.recurrence[n],
            R::ONE,
            |_| (R::ONE, R::ZERO, R::ZERO),
            Some(self.spec),
            None,
        )
    }
}

/// Connection coefficients from the `source` to the `target` recurrence,
/// reporting sizing, allocation and arithmetic failures.
fn connection<R: RealScalar>(
    degree: usize,
    source_p0: R,
    source: impl Fn(usize) -> (R, R, R),
    target_p0: R,
    target: impl Fn(usize) -> (R, R, R),
    source_spec: Option<OrthogonalBasisSpec<R>>,
    target_plan: Option<OrthogonalBasisPlan<R>>,
) -> Result<OrthogonalTransform<R>, SciOxideError> {
    let size = checked_add(degree, 1, scioxide_core::quantity!("connection_order"))?;
    let mut entries = try_filled(
        checked_mul(size, size, scioxide_core::quantity!("connection_entries"))?,
        R::ZERO,
        scioxide_core::quantity!("connection_entries"),
    )?;
    if source_spec.is_some() && source_spec == target_plan.as_ref().map(|plan| plan.spec) {
        for diagonal in 0..size {
            entries[diagonal * size + diagonal] = R::ONE;
        }
        return Ok(OrthogonalTransform {
            rows: size,
            cols: size,
            entries,
            source: source_spec,
            target: target_plan,
        });
    }
    if candidate(connection_kernel::<R, Native>(
        &mut entries,
        size,
        source_p0,
        &source,
        target_p0,
        &target,
    ))?
    .is_none()
    {
        entries.fill(R::ZERO);
        extended(connection_kernel::<R, Extended>(
            &mut entries,
            size,
            source_p0,
            source,
            target_p0,
            target,
        ))?;
    }
    Ok(OrthogonalTransform {
        rows: size,
        cols: size,
        entries,
        source: source_spec,
        target: target_plan,
    })
}

fn connection_kernel<R: RealScalar, A: Arithmetic<R, Projected = R>>(
    entries: &mut [R],
    size: usize,
    source_p0: R,
    source: impl Fn(usize) -> (R, R, R),
    target_p0: R,
    target: impl Fn(usize) -> (R, R, R),
) -> ArithmeticResult<()> {
    let zero = A::scalar(R::ZERO)?;
    let mut previous = try_filled(
        size,
        zero,
        scioxide_core::quantity!("connection_coordinates"),
    )?;
    let mut current = try_filled(
        size,
        zero,
        scioxide_core::quantity!("connection_coordinates"),
    )?;
    let mut next = try_filled(
        size,
        zero,
        scioxide_core::quantity!("connection_coordinates"),
    )?;
    current[0] = A::checked(A::div(
        A::checked_scalar(source_p0)?,
        A::checked_scalar(target_p0)?,
    )?)?;
    entries[0] = A::project(current[0])?;
    for n in 0..size - 1 {
        multiply_coordinate::<R, A>(&current[..n + 1], &mut next, &target)?;
        let (a, b, c) = source(n);
        let (a, b, c) = (
            A::checked_scalar(a)?,
            A::checked_scalar(b)?,
            A::checked_scalar(c)?,
        );
        for (k, out) in next.iter_mut().enumerate().take(n + 2) {
            *out = A::checked(A::sub(
                A::checked(A::add(
                    A::checked(A::mul(*out, a)?)?,
                    A::checked(A::mul(current[k], b)?)?,
                )?)?,
                A::checked(A::mul(previous[k], c)?)?,
            )?)?;
            entries[(n + 1) * size + k] = A::project(*out)?;
        }
        core::mem::swap(&mut previous, &mut current);
        core::mem::swap(&mut current, &mut next);
    }
    Ok(())
}

/// Solve the target recurrence for x Q_k in coefficient space.
fn multiply_coordinate<R: RealScalar, A: Arithmetic<R>>(
    input: &[A::Value],
    output: &mut [A::Value],
    recurrence: impl Fn(usize) -> (R, R, R),
) -> ArithmeticResult<()> {
    output.fill(A::scalar(R::ZERO)?);
    for (k, &value) in input.iter().enumerate() {
        let [lower, middle, upper] =
            super::coefficients::multiplication::<R, A>(value, recurrence(k), k > 0)?;
        output[k] = A::checked(A::add(output[k], middle)?)?;
        if k + 1 < output.len() {
            output[k + 1] = A::checked(A::add(output[k + 1], upper)?)?;
        }
        if k > 0 {
            output[k - 1] = A::checked(A::add(output[k - 1], lower)?)?;
        }
    }
    Ok(())
}

impl<C: NumericScalar> ApproxPolynomial<C, OrthogonalBasis<C>> {
    /// Converts the polynomial to another orthogonal basis.
    ///
    /// The coefficients in the target basis come from Clenshaw's recurrence run
    /// in coefficient space, with multiplication by `x` applied as the
    /// tridiagonal operator of the target basis. No power-series coefficients
    /// are formed. This takes O(n²) time and O(n) storage. The target plan is
    /// prepared through [`degree`](Self::degree) with this basis's execution
    /// options. The returned [`PolynomialConversion`] reports the coefficient
    /// growth and backward-error estimates of the conversion.
    ///
    /// # Errors
    ///
    /// - The errors of [`OrthogonalBasisSpec::prepare`] for `target`.
    /// - `NumericalInstability` if a coefficient overflows even in extended
    ///   range.
    /// - `AllocationFailure` if the coefficients cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// let result = p.to_orthogonal(spec)?;
    /// assert!((result.polynomial().evaluate(0.5)? - 1.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_orthogonal(
        &self,
        target: OrthogonalBasisSpec<C::Real>,
    ) -> Result<PolynomialConversion<C, OrthogonalBasis<C>>, SciOxideError> {
        let target_plan = target.prepare(self.degree(), self.basis().plan.options)?;
        let coefficients = coefficients(
            self.basis().coefficients(),
            Some(&self.basis().plan),
            Some(&target_plan),
        )?;
        let polynomial = Self::from_basis(
            OrthogonalBasis::new(coefficients, target_plan),
            self.policy(),
        )?;
        conversion_result(self.basis().coefficients(), polynomial)
    }
}
impl<C: NumericScalar> ApproxPolynomial<C, crate::MonomialBasis<C>> {
    /// Converts the polynomial from powers of `x` to an orthogonal basis.
    ///
    /// Horner's rule runs in coefficient space, with multiplication by `x`
    /// applied as the tridiagonal operator of the target basis. This takes
    /// O(n²) time and O(n) storage. The target plan is prepared through
    /// [`degree`](Self::degree) with default
    /// [`PolynomialExecutionOptions`](crate::PolynomialExecutionOptions). The
    /// returned [`PolynomialConversion`] reports the coefficient growth of the
    /// conversion.
    ///
    /// # Errors
    ///
    /// - The errors of [`OrthogonalBasisSpec::prepare`] for `target`.
    /// - `NumericalInstability` if a coefficient overflows even in extended
    ///   range.
    /// - `AllocationFailure` if the coefficients cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// // (1 + x)² at x = 0.5 is 2.25.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.to_orthogonal(spec)?;
    /// assert!((result.polynomial().evaluate(0.5)? - 2.25).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_orthogonal(
        &self,
        target: OrthogonalBasisSpec<C::Real>,
    ) -> Result<PolynomialConversion<C, OrthogonalBasis<C>>, SciOxideError> {
        let plan = target.prepare(self.degree(), crate::PolynomialExecutionOptions::new())?;
        let coefficients = coefficients(self.basis().coefficients(), None, Some(&plan))?;
        let polynomial =
            ApproxPolynomial::from_basis(OrthogonalBasis::new(coefficients, plan), self.policy())?;
        conversion_result(self.basis().coefficients(), polynomial)
    }
}

/// A prepared operator or target that contradicts the requested conversion.
fn mismatch(argument: scioxide_core::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        detail,
    )
}
