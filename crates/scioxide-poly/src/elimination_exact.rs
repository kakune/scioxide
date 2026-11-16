//! Exact univariate elimination by the Brown subresultant PRS recurrence.
use crate::{ExactField, ExactPolynomial};
use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{InvalidInputReason, SciOxideError};

/// Holds the subresultant polynomial remainder sequence (PRS) and resultant of
/// two exact polynomials.
///
/// [`polynomials`](Self::polynomials) lists the nonzero PRS elements in
/// descending order of degree. The first two are the inputs, the one of higher
/// degree first. Each later element is a pseudo-remainder of its two
/// predecessors, scaled by Brown's recurrence so that it equals a subresultant,
/// a determinant of a submatrix of the Sylvester matrix. The sequence may skip
/// degrees (an abnormal PRS). The last element is, up to a constant factor, the
/// GCD of the inputs.
///
/// Returned by [`exact_subresultants`].
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// let b = ExactPolynomial::new(vec![Q(2, 1), Q(1, 1)])?;
/// let result = exact_subresultants(&p, &b)?;
/// assert_eq!(result.resultant(), &Q(1, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ExactSubresultants<C: ExactField> {
    polynomials: Vec<ExactPolynomial<C>>,
    scalars: Vec<C>,
    resultant: C,
}
impl<C: ExactField> ExactSubresultants<C> {
    /// Returns the inputs, higher degree first, followed by the nonzero
    /// subresultants in descending order of degree.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let b = ExactPolynomial::new(vec![Q(2, 1), Q(1, 1)])?;
    /// let result = exact_subresultants(&p, &b)?;
    /// assert_eq!(result.polynomials()[0], p);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn polynomials(&self) -> &[ExactPolynomial<C>] {
        &self.polynomials
    }
    /// Returns one scalar subresultant per entry of
    /// [`polynomials`](Self::polynomials).
    ///
    /// The first scalar is 1 by convention, and the second is `lc(b)^(m − n)`,
    /// where `b` is the input of lower degree `n` and `m` is the other degree.
    /// When the last polynomial is a nonzero constant, its scalar is the
    /// resultant of the first two entries, taken in that order.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let b = ExactPolynomial::new(vec![Q(2, 1), Q(1, 1)])?;
    /// let result = exact_subresultants(&p, &b)?;
    /// assert_eq!(result.scalar_subresultants().len(), result.polynomials().len());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn scalar_subresultants(&self) -> &[C] {
        &self.scalars
    }
    /// Returns the resultant `Res(left, right)` in the order the inputs were
    /// passed to [`exact_subresultants`].
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let b = ExactPolynomial::new(vec![Q(2, 1), Q(1, 1)])?;
    /// let result = exact_subresultants(&p, &b)?;
    /// assert_eq!(result.resultant(), &Q(1, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn resultant(&self) -> &C {
        &self.resultant
    }
}

fn power<C: ExactField>(mut base: C, mut n: usize) -> C {
    let mut value = C::one();
    while n != 0 {
        if n & 1 != 0 {
            value = value * base.clone();
        }
        n >>= 1;
        if n != 0 {
            base = base.clone() * base;
        }
    }
    value
}
// Combine the pseudo-remainder and Brown factors before scaling once.
fn pseudo_remainder<C: ExactField>(
    a: &ExactPolynomial<C>,
    b: &ExactPolynomial<C>,
    normalization: C,
) -> Result<ExactPolynomial<C>, SciOxideError> {
    let remainder = a.remainder(b)?;
    if remainder.is_zero() {
        return Ok(remainder);
    }
    // prem(a,b) = lc(b)^(deg(a)-deg(b)+1) * (a mod b).
    let factor =
        power(b.leading_coefficient().clone(), a.degree() - b.degree() + 1) * normalization;
    remainder.scaled(factor)
}

/// Computes the subresultant PRS and the resultant of two exact polynomials.
///
/// For `f` of degree `m` with leading coefficient `a` and roots `αᵢ`, and `g`
/// of degree `n`, the resultant is the determinant of the
/// `(m + n) × (m + n)` Sylvester matrix of `f` and `g`:
///
/// ```text
/// Res(f, g) = aⁿ ∏ᵢ g(αᵢ),    Res(g, f) = (−1)^(mn) Res(f, g).
/// ```
///
/// It is zero exactly when `f` and `g` have a common root. If either input is
/// the zero polynomial, the resultant is 0 and the sequence holds only the
/// nonzero input, if any. For a nonzero constant `g = c`, `Res(f, c) = cᵐ`.
///
/// Uses Brown's subresultant PRS algorithm with exact field division. It takes
/// O(d²) field operations and stores O(d²) coefficients for `d = max(m, n)`.
/// The subresultants keep coefficient growth polynomial in `d` for rational
/// coefficients. The cost of each field operation depends on your coefficient
/// type.
///
/// # Errors
///
/// - `NumericalInstability` (`Breakdown`) if the field division of the
///   coefficient type is inconsistent.
/// - `SizeOverflow` or `AllocationFailure` if the sequence cannot be stored.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// // Res((1 + x)², 2 + x) = (2 + x)² at x = −1, which is 1.
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// let b = ExactPolynomial::new(vec![Q(2, 1), Q(1, 1)])?;
/// let result = exact_subresultants(&p, &b)?;
/// assert_eq!(result.resultant(), &Q(1, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. E. Collins, "Subresultants and reduced polynomial remainder
///   sequences", *J. ACM* 14(1), 1967.
/// - W. S. Brown, "The subresultant PRS algorithm", *ACM Trans. Math. Softw.*
///   4(3), 1978.
pub fn exact_subresultants<C: ExactField>(
    left: &ExactPolynomial<C>,
    right: &ExactPolynomial<C>,
) -> Result<ExactSubresultants<C>, SciOxideError> {
    // The recurrence requires `deg a >= deg b`. Swapping changes the resultant by
    // `(-1)^{deg left * deg right}`, corrected at the end.
    let swapped = left.degree() < right.degree() || left.is_zero();
    let (a, b) = if swapped {
        (right, left)
    } else {
        (left, right)
    };
    let capacity = scioxide_core::backend::checked_add(
        b.degree(),
        3,
        scioxide_core::quantity!("exact_subresultants"),
    )?;
    let mut polynomials =
        try_with_capacity(capacity, scioxide_core::quantity!("exact_subresultants"))?;
    let mut scalars = try_with_capacity(capacity, scioxide_core::quantity!("exact_subresultants"))?;
    if !a.is_zero() {
        polynomials.push(a.try_copy()?);
        scalars.push(C::one());
    }
    if a.is_zero() || b.is_zero() {
        return Ok(ExactSubresultants {
            polynomials,
            scalars,
            resultant: C::zero(),
        });
    }
    let mut previous = a.try_copy()?;
    let mut current = b.try_copy()?;
    let gap = a.degree() - b.degree();
    let mut leading = b.leading_coefficient().clone();
    // Brown's subresultant PRS: `auxiliary` is Brown's `psi` and `-auxiliary` the
    // scalar subresultant attached to each pushed polynomial. Dividing every
    // pseudo-remainder by `-lc(previous) * psi^{drop}` makes each sequence element
    // the actual subresultant (a Sylvester-matrix determinant) rather than a
    // scaled remainder, which keeps coefficient growth in exact rational types
    // polynomial instead of exponential. The first step uses the sign
    // `(-1)^{gap + 1}` in place of that division.
    let mut auxiliary = -power(leading.clone(), gap);
    polynomials.push(b.try_copy()?);
    scalars.push(-auxiliary.clone());
    let sign = if gap % 2 == 0 { -C::one() } else { C::one() };
    let mut next = pseudo_remainder(&previous, &current, sign)?;
    while !next.is_zero() {
        let drop = current.degree() - next.degree();
        previous = current;
        current = next;
        polynomials.push(current.try_copy()?);
        let denominator = -leading * power(auxiliary.clone(), drop);
        next = pseudo_remainder(&previous, &current, C::one() / denominator)?;
        leading = current.leading_coefficient().clone();
        // `psi <- (-lc)^{drop} / psi^{drop - 1}`; the `drop == 1` (normal PRS) case
        // simplifies to `-lc` without the division.
        auxiliary = if drop > 1 {
            power(-leading.clone(), drop) / power(auxiliary, drop - 1)
        } else {
            -leading.clone()
        };
        scalars.push(-auxiliary.clone());
    }
    // The resultant is the scalar subresultant of a final constant remainder;
    // a nonconstant last element is a nontrivial common factor, so it vanishes.
    let mut resultant = if current.degree() == 0 {
        -auxiliary
    } else {
        C::zero()
    };
    if swapped && left.degree() % 2 == 1 && right.degree() % 2 == 1 {
        resultant = -resultant;
    }
    Ok(ExactSubresultants {
        polynomials,
        scalars,
        resultant,
    })
}

/// Computes the discriminant of an exact polynomial.
///
/// For `p` of degree `n ≥ 2` with leading coefficient `a` and roots `αᵢ`,
///
/// ```text
/// disc(p) = a^(2n−2) ∏_{i<j} (αᵢ − αⱼ)² = (−1)^(n(n−1)/2) Res(p, p′) / a,
/// ```
///
/// so `disc(p) = 0` exactly when `p` has a repeated root. For example,
/// `disc(a x² + b x + c) = b² − 4ac`. Nonzero constants and linear polynomials
/// return 1. The resultant uses the Sylvester matrix of the formal degree
/// `n − 1` of `p′`, so the formula also holds in positive characteristic when
/// the leading term of `p′` vanishes. If `p′ = 0`, the result is 0.
///
/// Computes `Res(p, p′)` with [`exact_subresultants`], in O(n²) field
/// operations.
///
/// # Errors
///
/// - `InvalidInput` if `polynomial` is the zero polynomial.
/// - `NumericalInstability` (`Breakdown`) if the field division of the
///   coefficient type is inconsistent.
/// - `SizeOverflow` or `AllocationFailure` if intermediate results cannot be
///   stored.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// // 1 + 2x + x² = (1 + x)² has a double root, so b² − 4ac = 0.
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// assert_eq!(exact_discriminant(&p)?, Q(0, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn exact_discriminant<C: ExactField>(
    polynomial: &ExactPolynomial<C>,
) -> Result<C, SciOxideError> {
    if polynomial.is_zero() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("polynomial"),
            InvalidInputReason::OutOfRange,
            "zero polynomial discriminant is undefined",
        ));
    }
    let n = polynomial.degree();
    if n < 2 {
        return Ok(C::one());
    }
    let derivative = polynomial.derivative()?;
    if derivative.is_zero() {
        return Ok(C::zero());
    }
    let mut value = exact_subresultants(polynomial, &derivative)?.resultant;
    // Res with the formal derivative degree carries the omitted leading powers.
    let missing = n - 1 - derivative.degree();
    if missing == 0 {
        value = value / polynomial.leading_coefficient().clone();
    } else {
        value = value * power(polynomial.leading_coefficient().clone(), missing - 1);
    }
    // `disc(p) = (-1)^{n(n-1)/2} Res(p, p') / lc(p)`; `n(n-1)/2` is odd exactly when
    // `n mod 4` is 2 or 3. The `missing` adjustment above folds this division by
    // `lc(p)` into the leading-coefficient power.
    if n % 4 >= 2 {
        value = -value;
    }
    Ok(value)
}
