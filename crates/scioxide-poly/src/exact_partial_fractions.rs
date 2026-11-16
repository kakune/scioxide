//! Exact partial fractions by the Chinese remainder theorem and `F`-adic
//! expansion.
use crate::fast_exact::power;
use crate::util::invalid;
use crate::{ExactField, ExactPolynomial};
use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{InvalidInputReason, SciOxideError};

// Extended Euclid tracks only the coefficient of a, then normalizes the unit GCD.
// Noninvertible factors are rejected as an inconsistent factorization.
fn inverse_mod<C: ExactField>(
    a: &ExactPolynomial<C>,
    modulus: &ExactPolynomial<C>,
) -> Result<ExactPolynomial<C>, SciOxideError> {
    let (mut old_r, mut r) = (modulus.try_copy()?, a.remainder(modulus)?);
    let (mut old_s, mut s) = (
        ExactPolynomial::constant(C::zero())?,
        ExactPolynomial::constant(C::one())?,
    );
    while !r.is_zero() {
        let (q, next_r) = old_r.div_rem(&r)?;
        let next_s = old_s.try_sub(&q.try_mul(&s)?)?;
        (old_r, r) = (r, next_r);
        (old_s, s) = (s, next_s);
    }
    // At exit `old_s * a = old_r (mod modulus)`. Only a nonzero constant `old_r`
    // (a unit) means `a` is invertible; dividing `old_s` by it yields the inverse,
    // reduced to a canonical representative below `deg(modulus)`. This is also
    // where pairwise coprimality of the supplied factors is actually enforced.
    if old_r.degree() != 0 || old_r.is_zero() {
        return Err(invalid(
            scioxide_core::argument!("factors"),
            InvalidInputReason::Inconsistent,
            "factors must be pairwise coprime",
        ));
    }
    let unit = ExactPolynomial::constant(old_r.leading_coefficient().clone())?;
    old_s.div_rem(&unit)?.0.remainder(modulus)
}

#[derive(Debug, Clone)]
struct Factor<C: ExactField> {
    polynomial: ExactPolynomial<C>,
    multiplicity: usize,
    power: ExactPolynomial<C>,
    inverse: ExactPolynomial<C>,
}

/// Decomposes exact rational functions over a fixed factored denominator.
///
/// For a monic denominator `D = F₁^m₁ ⋯ F_r^m_r` with monic, nonconstant and
/// pairwise coprime factors `Fᵢ`, every numerator `N` has the unique
/// decomposition
///
/// ```text
/// N / D = P + Σᵢ Σ_{k=1}^{mᵢ} A_{i,k} / Fᵢᵏ,    deg A_{i,k} < deg Fᵢ,
/// ```
///
/// where `P` is the polynomial quotient of `N` by `D`. The factors need not be
/// irreducible. Over the rationals with linear factors `x − aᵢ`, the `A_{i,k}`
/// are constants.
///
/// Construction checks the factorization and computes, for each `i`, the
/// inverse of `D / Fᵢ^mᵢ` modulo `Fᵢ^mᵢ` by the extended Euclidean algorithm.
/// This precomputation costs O(r n²) field operations for `r` factors and
/// `n = deg D`, and the plan stores O(n) coefficients.
/// [`decompose`](Self::decompose) then needs one product and one reduction per
/// factor, followed by the `Fᵢ`-adic expansion of the result. Factors keep the
/// order you pass.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
/// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
/// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
/// let denominator = x.try_mul(&x1)?;
/// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
/// let numerator = ExactPolynomial::constant(Q(1, 1))?;
/// let result = plan.decompose(&numerator)?;
/// assert_eq!(result.terms().len(), 2);
/// assert_eq!(
///     result.reconstruct()?,
///     ExactRationalFunction::new(numerator, denominator)?
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
/// Cambridge University Press, 2013, §5.11.
#[derive(Debug, Clone)]
#[must_use]
pub struct ExactPartialFractionPlan<C: ExactField> {
    denominator: ExactPolynomial<C>,
    factors: Vec<Factor<C>>,
}

impl<C: ExactField> ExactPartialFractionPlan<C> {
    /// Creates the decomposition plan for `denominator = Π Fᵢ^mᵢ`, given as
    /// pairs `(Fᵢ, mᵢ)`.
    ///
    /// Pass the monic denominator of an
    /// [`ExactRationalFunction`](crate::ExactRationalFunction). The constant
    /// denominator 1 takes an empty factor list.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `denominator` is zero or not monic,
    ///   a factor is constant or not monic, a multiplicity is 0, the degrees
    ///   `Σ mᵢ deg Fᵢ` differ from `deg denominator`, the product of the factor
    ///   powers differs from `denominator`, or two factors are not coprime.
    /// - `SizeOverflow` if `Σ mᵢ deg Fᵢ` overflows `usize`.
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `AllocationFailure` if intermediate polynomials cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// assert_eq!(result.terms().len(), 2);
    /// assert_eq!(
    ///     result.reconstruct()?,
    ///     ExactRationalFunction::new(numerator, denominator)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        denominator: &ExactPolynomial<C>,
        factors: &[(&ExactPolynomial<C>, usize)],
    ) -> Result<Self, SciOxideError> {
        let factors_argument = scioxide_core::argument!("factors");
        crate::util::validate_factors(
            denominator.degree(),
            !denominator.is_zero() && denominator.leading_coefficient() == &C::one(),
            factors
                .iter()
                .map(|(p, m)| (p.degree(), p.leading_coefficient() == &C::one(), *m)),
        )?;
        let mut prepared =
            try_with_capacity(factors.len(), scioxide_core::quantity!("exact_factors"))?;
        let mut product = ExactPolynomial::constant(C::one())?;
        for (factor, multiplicity) in factors {
            let power = power(factor, *multiplicity)?;
            product = product.try_mul(&power)?;
            prepared.push(Factor {
                polynomial: factor.try_copy()?,
                multiplicity: *multiplicity,
                power,
                inverse: ExactPolynomial::constant(C::zero())?,
            });
        }
        if product != *denominator {
            return Err(invalid(
                factors_argument,
                InvalidInputReason::Inconsistent,
                "factors do not reconstruct the denominator exactly",
            ));
        }
        // Chinese remainder preparation: with `D = prod_j F_j^{m_j}` and complement
        // `C_j = D / F_j^{m_j}`, store `C_j^{-1} mod F_j^{m_j}`. Then
        // `R / D = sum_j A_j / F_j^{m_j}` with `A_j = R C_j^{-1} mod F_j^{m_j}`, so each
        // decomposition needs only one product and one reduction per factor.
        for factor in &mut prepared {
            let (complement, _) = denominator.div_rem(&factor.power)?;
            factor.inverse = inverse_mod(&complement, &factor.power)?;
        }
        Ok(Self {
            denominator: denominator.try_copy()?,
            factors: prepared,
        })
    }

    /// Computes the partial-fraction decomposition of `numerator / D`.
    ///
    /// The numerator may have any degree; the quotient by `D` becomes the
    /// polynomial part. Terms are ordered by factor, in the order passed to
    /// [`new`](Self::new), and by ascending power within each factor.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `SizeOverflow` or `AllocationFailure` if the terms cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// assert_eq!(result.terms().len(), 2);
    /// assert_eq!(
    ///     result.reconstruct()?,
    ///     ExactRationalFunction::new(numerator, denominator)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn decompose(
        &self,
        numerator: &ExactPolynomial<C>,
    ) -> Result<ExactPartialFractions<C>, SciOxideError> {
        // Split off the polynomial part so only a proper fraction `R / D` remains.
        let (polynomial, remainder) = numerator.div_rem(&self.denominator)?;
        let count = self.factors.iter().map(|f| f.multiplicity).sum();

        let mut terms = try_with_capacity(
            count,
            scioxide_core::quantity!("exact_partial_fraction_terms"),
        )?;
        for factor in &self.factors {
            let mut local = remainder
                .try_mul(&factor.inverse)?
                .remainder(&factor.power)?;
            let start = terms.len();
            // `F`-adic expansion `A = sum_k a_k F^k` with `deg a_k < deg F`, so
            // `A / F^m = sum_k a_k / F^{m-k}`. Repeated division by `F` yields the digits
            // lowest first, i.e. for the highest power `m` first; the slice is reversed
            // afterwards to list terms in ascending power.
            for exponent in (1..=factor.multiplicity).rev() {
                let (quotient, coefficient) = local.div_rem(&factor.polynomial)?;
                terms.push(ExactPartialFractionTerm {
                    factor: factor.polynomial.try_copy()?,
                    power: exponent,
                    numerator: coefficient,
                });
                local = quotient;
            }
            terms[start..].reverse();
        }
        Ok(ExactPartialFractions { polynomial, terms })
    }
}

/// Holds one term `A / Fᵏ` of an exact partial-fraction decomposition, with
/// `deg A < deg F`.
///
/// ```
/// use scioxide_poly::*;
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
/// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
/// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
/// let denominator = x.try_mul(&x1)?;
/// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
/// let numerator = ExactPolynomial::constant(Q(1, 1))?;
/// let result = plan.decompose(&numerator)?;
/// let term = &result.terms()[0];
/// assert_eq!(term.power(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExactPartialFractionTerm<C: ExactField> {
    factor: ExactPolynomial<C>,
    power: usize,
    numerator: ExactPolynomial<C>,
}
impl<C: ExactField> ExactPartialFractionTerm<C> {
    /// Returns the monic factor `F`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// let term = &result.terms()[0];
    /// assert_eq!(term.factor(), &x);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn factor(&self) -> &ExactPolynomial<C> {
        &self.factor
    }
    /// Returns the power `k ≥ 1` of `F` in the term's denominator.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// let term = &result.terms()[0];
    /// assert_eq!(term.power(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn power(&self) -> usize {
        self.power
    }
    /// Returns the numerator `A`, of degree below `deg F`.
    ///
    /// For a linear factor `F = x − a` and `k = 1`, it is the residue of the
    /// rational function at `a`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// let term = &result.terms()[0];
    /// assert_eq!(term.numerator().coefficients(), &[Q(1, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn numerator(&self) -> &ExactPolynomial<C> {
        &self.numerator
    }
}

/// Holds an exact partial-fraction decomposition `P + Σ A_{i,k} / Fᵢᵏ`.
///
/// Returned by [`ExactPartialFractionPlan::decompose`].
///
/// ```
/// use scioxide_poly::*;
/// mod doctest_field {
///     include!(concat!(
///         env!("CARGO_MANIFEST_DIR"),
///         "/tests/support/exact_field.rs"
///     ));
/// # }
/// # use doctest_field::Q;
/// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
/// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
/// let denominator = x.try_mul(&x1)?;
/// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
/// let numerator = ExactPolynomial::constant(Q(1, 1))?;
/// let result = plan.decompose(&numerator)?;
/// assert_eq!(result.terms().len(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ExactPartialFractions<C: ExactField> {
    polynomial: ExactPolynomial<C>,
    terms: Vec<ExactPartialFractionTerm<C>>,
}
impl<C: ExactField> ExactPartialFractions<C> {
    /// Returns the polynomial part `P`, the quotient of the numerator by the
    /// denominator.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// assert!(result.polynomial().is_zero());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn polynomial(&self) -> &ExactPolynomial<C> {
        &self.polynomial
    }
    /// Returns the terms, ordered by factor and then by ascending power.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// assert_eq!(result.terms().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn terms(&self) -> &[ExactPartialFractionTerm<C>] {
        &self.terms
    }
    /// Sums the decomposition back into an
    /// [`ExactRationalFunction`](crate::ExactRationalFunction) in lowest terms.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `SizeOverflow` or `AllocationFailure` if intermediate polynomials cannot
    ///   be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// mod doctest_field {
    ///     include!(concat!(
    ///         env!("CARGO_MANIFEST_DIR"),
    ///         "/tests/support/exact_field.rs"
    ///     ));
    /// # }
    /// # use doctest_field::Q;
    /// let x = ExactPolynomial::new(vec![Q(0, 1), Q(1, 1)])?;
    /// let x1 = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let denominator = x.try_mul(&x1)?;
    /// let plan = ExactPartialFractionPlan::new(&denominator, &[(&x, 1), (&x1, 1)])?;
    /// let numerator = ExactPolynomial::constant(Q(1, 1))?;
    /// let result = plan.decompose(&numerator)?;
    /// assert_eq!(
    ///     result.reconstruct()?,
    ///     ExactRationalFunction::new(numerator, denominator)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn reconstruct(&self) -> Result<crate::ExactRationalFunction<C>, SciOxideError> {
        let initial = crate::ExactRationalFunction::new(
            self.polynomial.try_copy()?,
            ExactPolynomial::constant(C::one())?,
        )?;
        self.terms.iter().try_fold(initial, |sum, term| {
            sum.try_add(&crate::ExactRationalFunction::new(
                term.numerator.try_copy()?,
                power(&term.factor, term.power)?,
            )?)
        })
    }
}
