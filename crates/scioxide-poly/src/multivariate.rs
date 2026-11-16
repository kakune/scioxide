//! Sparse multivariate polynomials stored as sorted term lists.
use crate::ExactCoefficient;
use crate::admission::native_value as native_scalar;
use crate::util::invalid;
use crate::util::{round_complex, scaled_complex};
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::backend::{checked_add, try_with_capacity};
use scioxide_core::{Argument, ExecutionContext, InvalidInputReason, NumericScalar, SciOxideError};

mod evaluation;
pub use evaluation::{ExactSparseView, SparseEvaluationPlan};

/// Selects the monomial order that sorts the terms of a sparse polynomial.
///
/// For exponent vectors `α` and `β` of the same length, with total degrees
/// `|α| = Σᵢ αᵢ`, each order decides whether `x^α > x^β`. Sparse polynomials
/// store their terms in ascending order, so the last term is the leading term.
///
/// # Examples
///
/// ```
/// use core::cmp::Ordering;
/// use scioxide_poly::{Monomial, MonomialOrder};
///
/// // x² > y² in lexicographic order with x before y.
/// let a = Monomial::new(&[2, 0])?;
/// let b = Monomial::new(&[0, 2])?;
/// let order = MonomialOrder::Lexicographic;
/// assert_eq!(order.compare_monomials(&a, &b)?, Ordering::Greater);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. A. Cox, J. Little and D. O'Shea, *Ideals, Varieties, and Algorithms*,
/// 4th ed., Springer, 2015, §2.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonomialOrder {
    /// Lexicographic order: `x^α > x^β` if the first nonzero entry of `α − β`
    /// is positive, so variable 0 is the most significant.
    Lexicographic,
    /// Graded lexicographic order: `x^α > x^β` if `|α| > |β|`, or if
    /// `|α| = |β|` and `x^α > x^β` lexicographically.
    GradedLexicographic,
    /// Graded reverse lexicographic order: `x^α > x^β` if `|α| > |β|`, or if
    /// `|α| = |β|` and the last nonzero entry of `α − β` is negative.
    GradedReverseLexicographic,
}
impl MonomialOrder {
    /// Compares two monomials in this order.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` and `b` have different numbers of variables.
    ///
    /// ```
    /// use core::cmp::Ordering;
    /// use scioxide_poly::{Monomial, MonomialOrder};
    ///
    /// // Both have degree 2 and the last nonzero entry of (2, 0) − (0, 2) is
    /// // negative, so x² > y² in graded reverse lexicographic order.
    /// let a = Monomial::new(&[2, 0])?;
    /// let b = Monomial::new(&[0, 2])?;
    /// let order = MonomialOrder::GradedReverseLexicographic;
    /// assert_eq!(order.compare_monomials(&a, &b)?, Ordering::Greater);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compare_monomials(self, a: &Monomial, b: &Monomial) -> Result<Ordering, SciOxideError> {
        if a.exponents.len() != b.exponents.len() {
            return Err(SciOxideError::dimension_mismatch(
                a.exponents.len(),
                b.exponents.len(),
            ));
        }
        Ok(self.compare(a, b))
    }
    #[inline]
    pub(crate) fn compare(self, a: &Monomial, b: &Monomial) -> Ordering {
        match self {
            Self::Lexicographic => a.exponents.cmp(b.exponents),
            Self::GradedLexicographic => a
                .degree
                .cmp(&b.degree)
                .then_with(|| a.exponents.cmp(b.exponents)),
            Self::GradedReverseLexicographic => a
                .degree
                .cmp(&b.degree)
                // Graded reverse lexicographic: at equal total degree, the monomial with the
                // *smaller* exponent in the last differing variable is the larger one, hence
                // the reversed iteration with the operands swapped.
                .then_with(|| b.exponents.iter().rev().cmp(a.exponents.iter().rev())),
        }
    }
}
/// Borrows the exponent vector `α = (α₀, …, αₙ₋₁)` of the monomial
/// `x^α = x₀^α₀ ⋯ xₙ₋₁^αₙ₋₁`, together with its total degree `|α| = Σᵢ αᵢ`.
///
/// ```
/// use scioxide_poly::Monomial;
///
/// // x²y has total degree 3.
/// let m = Monomial::new(&[2, 1])?;
/// assert_eq!(m.degree(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Monomial<'a> {
    exponents: &'a [usize],
    degree: usize,
}
impl<'a> Monomial<'a> {
    /// Returns the monomial with the given exponents. An empty slice is the
    /// constant monomial in zero variables.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the total degree does not fit in `usize`.
    ///
    /// ```
    /// assert_eq!(scioxide_poly::Monomial::new(&[2, 1])?.degree(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(exponents: &'a [usize]) -> Result<Self, SciOxideError> {
        let degree = exponents.iter().try_fold(0, |s, &e| {
            checked_add(s, e, scioxide_core::quantity!("total_degree"))
        })?;
        Ok(Self { exponents, degree })
    }
    /// Returns the exponents in variable order.
    ///
    /// ```
    /// assert_eq!(scioxide_poly::Monomial::new(&[2, 1])?.exponents(), &[2, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn exponents(self) -> &'a [usize] {
        self.exponents
    }
    /// Returns the total degree `|α|`.
    ///
    /// ```
    /// assert_eq!(scioxide_poly::Monomial::new(&[2, 1])?.degree(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn degree(self) -> usize {
        self.degree
    }
}
impl AsRef<[usize]> for Monomial<'_> {
    fn as_ref(&self) -> &[usize] {
        self.exponents
    }
}

/// Limits the number of terms in the intermediate results of a substitution.
///
/// Used by `substitute` and `substitute_into` on [`ApproxSparsePolynomial`] and
/// [`ExactSparsePolynomial`]. Terms are counted after like terms are combined
/// and exact zeros removed. A result over the limit is a `ResourceLimit` error.
///
/// ```
/// use scioxide_poly::SparseExpansionLimits;
///
/// let limits = SparseExpansionLimits::new(16);
/// assert_eq!(limits.max_terms(), 16);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseExpansionLimits {
    max_terms: usize,
}

impl SparseExpansionLimits {
    /// Returns limits that allow at most `max_terms` terms in every
    /// intermediate result. With zero, only the zero polynomial can be
    /// expanded.
    ///
    /// ```
    /// use scioxide_poly::SparseExpansionLimits;
    ///
    /// let limits = SparseExpansionLimits::new(16);
    /// assert_eq!(limits.max_terms(), 16);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new(max_terms: usize) -> Self {
        Self { max_terms }
    }
    /// Returns the largest number of terms allowed in an intermediate result.
    ///
    /// ```
    /// use scioxide_poly::SparseExpansionLimits;
    ///
    /// let limits = SparseExpansionLimits::new(16);
    /// assert_eq!(limits.max_terms(), 16);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn max_terms(self) -> usize {
        self.max_terms
    }
}

impl Default for SparseExpansionLimits {
    /// Returns limits that allow any number of terms (`max_terms = usize::MAX`).
    ///
    /// ```
    /// use scioxide_poly::SparseExpansionLimits;
    ///
    /// assert_eq!(SparseExpansionLimits::default().max_terms(), usize::MAX);
    /// ```
    fn default() -> Self {
        Self::new(usize::MAX)
    }
}

// Exact and approximate arithmetic share the sparse structural operations.
trait SparseArithmetic<C: Clone> {
    fn one() -> C;
    fn add(a: C, b: C) -> Result<C, SciOxideError>;
    fn mul(a: C, b: C) -> Result<C, SciOxideError>;
    fn neg(a: C) -> C;
    fn integer(n: usize) -> Result<C, SciOxideError>;
    fn is_zero(a: &C) -> bool;
    fn validate(a: &C) -> Result<(), SciOxideError>;
    fn native_coefficient(_: &C) -> bool {
        true
    }
}
struct Exact;
impl<C: ExactCoefficient> SparseArithmetic<C> for Exact {
    fn one() -> C {
        C::one()
    }
    fn add(a: C, b: C) -> Result<C, SciOxideError> {
        Ok(a + b)
    }
    fn mul(a: C, b: C) -> Result<C, SciOxideError> {
        Ok(a * b)
    }
    fn neg(a: C) -> C {
        -a
    }
    fn integer(n: usize) -> Result<C, SciOxideError> {
        Ok(C::from_usize(n))
    }
    fn is_zero(a: &C) -> bool {
        a.is_zero()
    }
    fn validate(_: &C) -> Result<(), SciOxideError> {
        Ok(())
    }
}
struct Approx;
impl<C: NumericScalar> SparseArithmetic<C> for Approx {
    fn one() -> C {
        C::ONE
    }
    fn add(a: C, b: C) -> Result<C, SciOxideError> {
        finite_coefficient(a + b)
    }
    fn mul(a: C, b: C) -> Result<C, SciOxideError> {
        // Complex component products can overflow or underflow even when the
        // final components fit. Ordinary ranges need no scaled round trip.
        if !C::IS_REAL && !(native_scalar(a) & native_scalar(b)) {
            return round_complex(scaled_complex(a)?.mul(scaled_complex(b)?)?);
        }
        finite_coefficient(a * b)
    }
    fn neg(a: C) -> C {
        -a
    }
    fn integer(n: usize) -> Result<C, SciOxideError> {
        Ok(C::from_real(
            scioxide_core::RealScalar::try_from_usize_exact(n)?,
        ))
    }
    fn is_zero(a: &C) -> bool {
        *a == C::ZERO
    }
    fn native_coefficient(a: &C) -> bool {
        native_scalar(*a)
    }
    fn validate(a: &C) -> Result<(), SciOxideError> {
        crate::util::finite_input(*a, "coefficient or point must have finite components").map(drop)
    }
}

fn finite_coefficient<C: NumericScalar>(value: C) -> Result<C, SciOxideError> {
    crate::util::finite_arithmetic(value, "sparse coefficient arithmetic overflow or breakdown")
}

// Heap entries borrow operand rows; exponent sums are compared without allocating
// or retaining mutable aliases to a frontier arena.
struct ProductHead<'a> {
    a: Monomial<'a>,
    b: Monomial<'a>,
    degree: usize,
    left: usize,
    right: usize,
    order: MonomialOrder,
    degree_order: bool,
}
impl PartialEq for ProductHead<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ProductHead<'_> {}
impl Ord for ProductHead<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let order = (other.order as u8)
            .cmp(&(self.order as u8))
            .then_with(|| other.degree_order.cmp(&self.degree_order));
        if order != Ordering::Equal {
            return order;
        }
        let lex = || {
            other
                .a
                .exponents
                .iter()
                .zip(other.b.exponents)
                .map(|(a, b)| a + b)
                .cmp(
                    self.a
                        .exponents
                        .iter()
                        .zip(self.b.exponents)
                        .map(|(a, b)| a + b),
                )
        };
        // With at most one active variable, degree determines the complete
        // monomial in every supported order. No exponent sums need comparing.
        let key = if self.degree_order {
            other.degree.cmp(&self.degree)
        } else {
            match self.order {
                MonomialOrder::Lexicographic => lex(),
                MonomialOrder::GradedLexicographic => other.degree.cmp(&self.degree).then_with(lex),
                MonomialOrder::GradedReverseLexicographic => {
                    other.degree.cmp(&self.degree).then_with(|| {
                        self.a
                            .exponents
                            .iter()
                            .zip(self.b.exponents)
                            .rev()
                            .map(|(a, b)| a + b)
                            .cmp(
                                other
                                    .a
                                    .exponents
                                    .iter()
                                    .zip(other.b.exponents)
                                    .rev()
                                    .map(|(a, b)| a + b),
                            )
                    })
                }
            }
        };
        key.then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}
impl PartialOrd for ProductHead<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Borrows the terms of a sparse polynomial in ascending monomial order.
///
/// Each term is a [`Monomial`] and a reference to its coefficient; neither is
/// copied.
///
/// ```
/// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
///
/// // p(x, y) = 3xy².
/// let order = MonomialOrder::Lexicographic;
/// let p = ApproxSparsePolynomial::new(2, order, vec![1, 2], vec![3.0_f64])?;
/// for (m, c) in p.terms() {
///     assert_eq!(m.exponents(), &[1, 2]);
///     assert_eq!(*c, 3.0);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SparseTerms<'a, C> {
    dimension: usize,
    exponents: &'a [usize],
    coefficients: &'a [C],
    degrees: &'a [usize],
}
impl<'a, C> SparseTerms<'a, C> {
    /// Returns the number of terms.
    pub const fn len(&self) -> usize {
        self.coefficients.len()
    }
    /// Returns `true` if there are no terms, that is, for the zero polynomial.
    pub const fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }
    /// Returns an iterator over the terms in ascending monomial order.
    pub fn iter(&self) -> SparseTermIter<'a, C> {
        SparseTermIter {
            terms: SparseTerms {
                dimension: self.dimension,
                exponents: self.exponents,
                coefficients: self.coefficients,
                degrees: self.degrees,
            },
            front: 0,
            back: self.len(),
        }
    }
}
/// Iterates over the terms of a sparse polynomial, from both ends.
///
/// Obtain it with [`SparseTerms::iter`] or `IntoIterator`.
///
/// ```
/// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
///
/// // p(x) = 2 + 3x; the leading term comes last.
/// let order = MonomialOrder::Lexicographic;
/// let (exponents, coefficients) = (vec![0, 1], vec![2.0_f64, 3.0]);
/// let p = ApproxSparsePolynomial::new(1, order, exponents, coefficients)?;
/// let reversed: Vec<f64> = p.terms().iter().rev().map(|(_, c)| *c).collect();
/// assert_eq!(reversed, vec![3.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct SparseTermIter<'a, C> {
    terms: SparseTerms<'a, C>,
    front: usize,
    back: usize,
}
impl<'a, C> SparseTermIter<'a, C> {
    fn term(&self, i: usize) -> (Monomial<'a>, &'a C) {
        (
            Monomial {
                exponents: &self.terms.exponents
                    [i * self.terms.dimension..(i + 1) * self.terms.dimension],
                degree: self.terms.degrees[i],
            },
            &self.terms.coefficients[i],
        )
    }
}
impl<'a, C> Iterator for SparseTermIter<'a, C> {
    type Item = (Monomial<'a>, &'a C);
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let i = self.front;
        self.front += 1;
        Some(self.term(i))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}
impl<'a, C> DoubleEndedIterator for SparseTermIter<'a, C> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(self.term(self.back))
    }
}
impl<C> core::iter::FusedIterator for SparseTermIter<'_, C> {}
impl<C> ExactSizeIterator for SparseTermIter<'_, C> {}
impl<'a, C> IntoIterator for SparseTerms<'a, C> {
    type Item = (Monomial<'a>, &'a C);
    type IntoIter = SparseTermIter<'a, C>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AxisPowers {
    offset: usize,
    bits: usize,
}

#[derive(Clone, Debug, PartialEq)]
// Exponent rows, degrees and coefficients have matching term counts. Rows are
// strictly ascending under `order`, with no duplicate monomials or exact zeros.
struct Data<C> {
    dimension: usize,
    order: MonomialOrder,
    exponents: Vec<usize>,
    coefficients: Vec<C>,
    degrees: Vec<usize>,
    axes: Vec<AxisPowers>,
    native_coefficients: bool,
}
impl<C: Clone> Data<C> {
    fn empty(dimension: usize, order: MonomialOrder) -> Self {
        Self {
            dimension,
            order,
            exponents: Vec::new(),
            coefficients: Vec::new(),
            degrees: Vec::new(),
            axes: Vec::new(),
            native_coefficients: true,
        }
    }
    fn reserve(&mut self, capacity: usize) -> Result<(), SciOxideError> {
        let exponents = scioxide_core::backend::checked_mul(
            capacity,
            self.dimension,
            scioxide_core::quantity!("exponents"),
        )?;
        self.exponents
            .try_reserve_exact(exponents.saturating_sub(self.exponents.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("exponents"), exponents)
            })?;
        self.coefficients
            .try_reserve_exact(capacity.saturating_sub(self.coefficients.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("sparse_terms"),
                    capacity,
                )
            })?;
        self.degrees
            .try_reserve_exact(capacity.saturating_sub(self.degrees.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("total_degree"),
                    capacity,
                )
            })?;
        Ok(())
    }
    fn terms(&self) -> SparseTerms<'_, C> {
        SparseTerms {
            dimension: self.dimension,
            exponents: &self.exponents,
            coefficients: &self.coefficients,
            degrees: &self.degrees,
        }
    }
    #[inline]
    fn monomial(&self, i: usize) -> Monomial<'_> {
        Monomial {
            exponents: &self.exponents[i * self.dimension..(i + 1) * self.dimension],
            degree: self.degrees[i],
        }
    }
    fn finish<A: SparseArithmetic<C>>(mut self) -> Result<Self, SciOxideError> {
        self.native_coefficients = self.coefficients.iter().all(A::native_coefficient);
        self.axes = axis_powers(self.dimension, &self.exponents, self.coefficients.len())?;
        Ok(self)
    }
    fn new<A: SparseArithmetic<C>>(
        dimension: usize,
        order: MonomialOrder,
        exponents: Vec<usize>,
        coefficients: Vec<C>,
    ) -> Result<Self, SciOxideError> {
        let expected = scioxide_core::backend::checked_mul(
            dimension,
            coefficients.len(),
            scioxide_core::quantity!("exponents"),
        )?;
        if exponents.len() != expected {
            return Err(SciOxideError::dimension_mismatch(expected, exponents.len()));
        }
        let mut data = Self::empty(dimension, order);
        data.degrees =
            try_with_capacity(coefficients.len(), scioxide_core::quantity!("total_degree"))?;
        for (i, c) in coefficients.iter().enumerate() {
            A::validate(c)?;
            data.degrees
                .push(Monomial::new(&exponents[i * dimension..(i + 1) * dimension])?.degree);
        }
        data.exponents = exponents;
        data.coefficients = coefficients;
        data.canonicalize::<A>(order, None)
    }
    fn canonicalize<A: SparseArithmetic<C>>(
        &self,
        order: MonomialOrder,
        prefixes: Option<(&[usize], usize, usize)>,
    ) -> Result<Self, SciOxideError> {
        let mut indices = crate::util::try_collect(
            (0..self.coefficients.len()).map(Ok),
            scioxide_core::quantity!("sparse_indices"),
        )?;
        indices.sort_unstable_by(|&i, &j| {
            order
                .compare(&self.monomial(i), &self.monomial(j))
                .then_with(|| i.cmp(&j))
        });
        let mut result = Self::empty(self.dimension, order);
        result.reserve(indices.len())?;
        // Each transition changes the number of nonzero monomials in that source
        // prefix. Separate unsigned counts avoid a signed term-count restriction.
        let mut changes = if let Some((_, count, _)) = prefixes {
            scioxide_core::backend::try_filled(
                count,
                (0usize, 0usize),
                scioxide_core::quantity!("substitution_prefixes"),
            )?
        } else {
            Vec::new()
        };
        let mut previous: Option<usize> = None;
        let mut value: Option<C> = None;
        for i in indices {
            if let Some(j) = previous
                && self.monomial(i) != self.monomial(j)
                && let Some(c) = value.take()
            {
                result.append::<A>(self.monomial(j), c, usize::MAX)?;
            }
            let c = self.coefficients[i].clone();
            let was_nonzero = value.is_some();
            if !A::is_zero(&c) {
                let sum = if let Some(v) = value.take() {
                    A::add(v, c)?
                } else {
                    c
                };
                if !A::is_zero(&sum) {
                    value = Some(sum);
                }
            }
            if let Some((sources, _, _)) = prefixes {
                let delta = &mut changes[sources[i]];
                match (was_nonzero, value.is_some()) {
                    (false, true) => delta.0 += 1,
                    (true, false) => delta.1 += 1,
                    _ => {}
                }
            }
            previous = Some(i);
        }
        if let Some(i) = previous
            && let Some(c) = value
        {
            result.append::<A>(self.monomial(i), c, usize::MAX)?;
        }
        if let Some((_, _, limit)) = prefixes {
            let mut count = 0usize;
            for (added, removed) in changes {
                count =
                    checked_add(count, added, scioxide_core::quantity!("sparse_terms"))? - removed;
                if count > limit {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("sparse_terms"),
                        count,
                        limit,
                    ));
                }
            }
        }
        result.finish::<A>()
    }
    fn append<A: SparseArithmetic<C>>(
        &mut self,
        m: Monomial<'_>,
        c: C,
        limit: usize,
    ) -> Result<(), SciOxideError> {
        if A::is_zero(&c) {
            return Ok(());
        }
        let len = self.coefficients.len();
        if len >= limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("sparse_terms"),
                len.saturating_add(1),
                limit,
            ));
        }
        // Every caller reserves an upper bound before traversal. No storage
        // acquisition is permitted in the term-emission loop.
        debug_assert!(len < self.coefficients.capacity());
        debug_assert!(len < self.degrees.capacity());
        debug_assert!(self.exponents.len() + self.dimension <= self.exponents.capacity());
        self.exponents.extend_from_slice(m.exponents);
        self.degrees.push(m.degree);
        self.coefficients.push(c);
        Ok(())
    }
    fn compatible(&self, b: &Self) -> Result<(), SciOxideError> {
        if self.dimension != b.dimension {
            Err(SciOxideError::dimension_mismatch(
                self.dimension,
                b.dimension,
            ))
        } else if self.order != b.order {
            Err(invalid(
                scioxide_core::argument!("other.order"),
                InvalidInputReason::OutOfRange,
                "polynomial order mismatch",
            ))
        } else {
            Ok(())
        }
    }
    fn add<A: SparseArithmetic<C>>(
        &self,
        b: &Self,
        negative: bool,
        limit: usize,
    ) -> Result<Self, SciOxideError> {
        self.compatible(b)?;
        let mut result = Self::empty(self.dimension, self.order);
        result.reserve(
            checked_add(
                self.coefficients.len(),
                b.coefficients.len(),
                scioxide_core::quantity!("sparse_terms"),
            )?
            .min(limit),
        )?;
        let (mut i, mut j) = (0, 0);
        while i < self.coefficients.len() || j < b.coefficients.len() {
            let cmp = if i == self.coefficients.len() {
                Ordering::Greater
            } else if j == b.coefficients.len() {
                Ordering::Less
            } else {
                self.order.compare(&self.monomial(i), &b.monomial(j))
            };
            match cmp {
                Ordering::Less => {
                    result.append::<A>(self.monomial(i), self.coefficients[i].clone(), limit)?;
                    i += 1;
                }
                Ordering::Greater => {
                    let c = b.coefficients[j].clone();
                    result.append::<A>(
                        b.monomial(j),
                        if negative { A::neg(c) } else { c },
                        limit,
                    )?;
                    j += 1;
                }
                Ordering::Equal => {
                    let c = b.coefficients[j].clone();
                    result.append::<A>(
                        self.monomial(i),
                        A::add(
                            self.coefficients[i].clone(),
                            if negative { A::neg(c) } else { c },
                        )?,
                        limit,
                    )?;
                    i += 1;
                    j += 1;
                }
            }
        }
        result.finish::<A>()
    }
    fn head<'a>(
        &'a self,
        b: &'a Self,
        left: usize,
        right: usize,
        degree_order: bool,
    ) -> Result<ProductHead<'a>, SciOxideError> {
        let a = self.monomial(left);
        let b = b.monomial(right);
        let degree = checked_add(a.degree, b.degree, scioxide_core::quantity!("total_degree"))?;
        Ok(ProductHead {
            a,
            b,
            degree,
            left,
            right,
            order: self.order,
            degree_order,
        })
    }
    fn mul<A: SparseArithmetic<C>>(
        &self,
        b: &Self,
        limit: usize,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        self.compatible(b)?;
        context.check()?;
        let mut result = Self::empty(self.dimension, self.order);
        if self.coefficients.is_empty() || b.coefficients.is_empty() {
            return result.finish::<A>();
        }
        let pairs = self.coefficients.len().saturating_mul(b.coefficients.len());
        let mut box_bound = 1usize;
        let mut active_axes = 0usize;
        for axis in 0..self.dimension {
            let max_a = self
                .terms()
                .iter()
                .map(|(m, _)| m.exponents[axis])
                .max()
                .unwrap_or(0);
            let max_b = b
                .terms()
                .iter()
                .map(|(m, _)| m.exponents[axis])
                .max()
                .unwrap_or(0);
            active_axes += usize::from(max_a != 0 || max_b != 0);
            box_bound = box_bound.saturating_mul(max_a.saturating_add(max_b).saturating_add(1));
        }
        result.reserve(pairs.min(box_bound).min(limit))?;
        let rows_on_left = self.coefficients.len() <= b.coefficients.len();
        let rows = self.coefficients.len().min(b.coefficients.len());
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(rows).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("sparse_product_frontier"),
                rows,
            )
        })?;
        for row in 0..rows {
            context.check()?;
            heap.push(if rows_on_left {
                self.head(b, row, 0, active_axes <= 1)?
            } else {
                self.head(b, 0, row, active_axes <= 1)?
            });
        }
        let mut key = scioxide_core::backend::try_filled(
            self.dimension,
            0,
            scioxide_core::quantity!("exponents"),
        )?;
        let mut degree = 0;
        let mut coefficient: Option<C> = None;
        while let Some(mut head) = heap.peek_mut() {
            context.check()?;
            let equal = head
                .a
                .exponents
                .iter()
                .zip(head.b.exponents)
                .zip(&key)
                .all(|((a, b), k)| a + b == *k);
            if !equal && let Some(c) = coefficient.take() {
                result.append::<A>(
                    Monomial {
                        exponents: &key,
                        degree,
                    },
                    c,
                    limit,
                )?;
            }
            let c = A::mul(
                self.coefficients[head.left].clone(),
                b.coefficients[head.right].clone(),
            )?;
            if !A::is_zero(&c) {
                let sum = if let Some(v) = coefficient.take() {
                    A::add(v, c)?
                } else {
                    c
                };
                if !A::is_zero(&sum) {
                    coefficient = Some(sum);
                }
                for ((k, a), b) in key.iter_mut().zip(head.a.exponents).zip(head.b.exponents) {
                    *k = a + b;
                }
                degree = head.degree;
            }
            let (left, right) = if rows_on_left {
                (head.left, head.right + 1)
            } else {
                (head.left + 1, head.right)
            };
            if left < self.coefficients.len() && right < b.coefficients.len() {
                *head = self.head(b, left, right, active_axes <= 1)?;
            } else {
                alloc::collections::binary_heap::PeekMut::pop(head);
            }
        }
        if let Some(c) = coefficient {
            result.append::<A>(
                Monomial {
                    exponents: &key,
                    degree,
                },
                c,
                limit,
            )?;
        }
        context.check()?;
        result.finish::<A>()
    }
    fn derivative<A: SparseArithmetic<C>>(&self, axis: usize) -> Result<Self, SciOxideError> {
        if axis >= self.dimension {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("polynomial_variables"),
                axis,
                self.dimension,
            ));
        }
        let mut result = Self::empty(self.dimension, self.order);
        result.reserve(self.coefficients.len())?;
        for (m, c) in self.terms() {
            let n = m.exponents[axis];
            if n != 0 {
                let value = A::mul(c.clone(), A::integer(n)?)?;
                if !A::is_zero(&value) {
                    result.append::<A>(m, value, usize::MAX)?;
                    let last = result.exponents.len() - self.dimension;
                    result.exponents[last + axis] -= 1;
                    if let Some(d) = result.degrees.last_mut() {
                        *d -= 1;
                    }
                }
            }
        }
        result.finish::<A>()
    }
    fn permute<A: SparseArithmetic<C>>(
        &self,
        permutation: &[usize],
    ) -> Result<Self, SciOxideError> {
        validate_permutation(
            permutation,
            self.dimension,
            scioxide_core::argument!("permutation"),
        )?;
        let mut data = Self::empty(self.dimension, self.order);
        data.reserve(self.coefficients.len())?;
        for (m, c) in self.terms() {
            data.exponents
                .extend(permutation.iter().map(|&i| m.exponents[i]));
            data.coefficients.push(c.clone());
            data.degrees.push(m.degree);
        }
        data.canonicalize::<A>(self.order, None)
    }
    fn substitute<A: SparseArithmetic<C>>(
        &self,
        replacements: &[&Self],
        target_dimension: usize,
        limits: SparseExpansionLimits,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if replacements.len() != self.dimension {
            return Err(invalid(
                scioxide_core::argument!("replacements"),
                InvalidInputReason::OutOfRange,
                "one replacement per source variable is required",
            ));
        }
        for r in replacements {
            if r.dimension != target_dimension || r.order != self.order {
                return Err(invalid(
                    scioxide_core::argument!("replacements"),
                    InvalidInputReason::OutOfRange,
                    "substitution target dimension or order mismatch",
                ));
            }
        }
        let mut output = Self::empty(target_dimension, self.order);
        let mut sources = Vec::new();
        let mut squares: Vec<Vec<Self>> = crate::util::try_collect(
            replacements.iter().map(|_| Ok(Vec::new())),
            scioxide_core::quantity!("substitution_powers"),
        )?;
        let constant = scioxide_core::backend::try_filled(
            target_dimension,
            0,
            scioxide_core::quantity!("exponents"),
        )?;
        for (source, (m, c)) in self.terms().iter().enumerate() {
            context.check()?;
            let mut term = Self::empty(target_dimension, self.order);
            term.reserve(1)?;
            term.append::<A>(
                Monomial {
                    exponents: &constant,
                    degree: 0,
                },
                c.clone(),
                limits.max_terms,
            )?;
            for ((&e, replacement), powers) in
                m.exponents.iter().zip(replacements).zip(&mut squares)
            {
                let mut exponent = e;
                if exponent == 0 {
                    continue;
                }
                if replacement.coefficients.len() > limits.max_terms {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("sparse_terms"),
                        replacement.coefficients.len(),
                        limits.max_terms,
                    ));
                }
                let mut bit = 0;
                while exponent != 0 {
                    if exponent & 1 != 0 {
                        while powers.len() < bit {
                            let previous = powers.last().unwrap_or(replacement);
                            let square = previous.mul::<A>(previous, limits.max_terms, context)?;
                            powers.try_reserve(1).map_err(|_| {
                                SciOxideError::allocation_failure(
                                    scioxide_core::quantity!("substitution_powers"),
                                    powers.len().saturating_add(1),
                                )
                            })?;
                            powers.push(square);
                        }
                        let factor = if bit == 0 {
                            *replacement
                        } else {
                            &powers[bit - 1]
                        };
                        term = term.mul::<A>(factor, limits.max_terms, context)?;
                    }
                    exponent >>= 1;
                    bit += 1;
                }
            }
            context.check()?;
            let total = checked_add(
                output.coefficients.len(),
                term.coefficients.len(),
                scioxide_core::quantity!("sparse_terms"),
            )?;
            // Geometric collection growth is per expanded polynomial, never per monomial.
            if total > output.coefficients.capacity() {
                output.reserve(total.max(output.coefficients.capacity().saturating_mul(2)))?;
            }
            sources.try_reserve(term.coefficients.len()).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("substitution_sources"),
                    total,
                )
            })?;
            sources.extend(core::iter::repeat_n(source, term.coefficients.len()));
            output.exponents.extend(term.exponents);
            output.degrees.extend(term.degrees);
            output.coefficients.extend(term.coefficients);
        }
        context.check()?;
        let result = output.canonicalize::<A>(
            self.order,
            Some((&sources, self.coefficients.len(), limits.max_terms)),
        )?;
        context.check()?;
        Ok(result)
    }
}

fn axis_powers(
    dimension: usize,
    exponents: &[usize],
    terms: usize,
) -> Result<Vec<AxisPowers>, SciOxideError> {
    let mut axes = try_with_capacity(dimension, scioxide_core::quantity!("sparse_axes"))?;
    let mut offset = 0;
    for axis in 0..dimension {
        let max = (0..terms)
            .map(|i| exponents[i * dimension + axis])
            .max()
            .unwrap_or(0);
        let bits = (usize::BITS - max.leading_zeros()) as usize;
        axes.push(AxisPowers { offset, bits });
        offset = checked_add(offset, bits, scioxide_core::quantity!("sparse_powers"))?;
    }
    Ok(axes)
}
fn power<C: Clone, A: SparseArithmetic<C>>(mut x: C, mut n: usize) -> Result<C, SciOxideError> {
    let mut v = A::one();
    while n != 0 {
        if n & 1 != 0 {
            v = A::mul(v, x.clone())?;
        }
        n >>= 1;
        if n != 0 {
            x = A::mul(x.clone(), x)?;
        }
    }
    Ok(v)
}
/// Checks that `p` is a permutation of `0..n`; `argument` names `p` in the error.
fn validate_permutation(p: &[usize], n: usize, argument: Argument) -> Result<(), SciOxideError> {
    if p.len() != n
        || p.iter()
            .enumerate()
            .any(|(i, &v)| v >= n || p[..i].contains(&v))
    {
        Err(invalid(
            argument,
            InvalidInputReason::OutOfRange,
            "invalid variable permutation",
        ))
    } else {
        Ok(())
    }
}

/// Represents a polynomial in `n` variables with coefficients in a
/// user-defined [`ExactCoefficient`] ring, as a list of nonzero terms.
///
/// The polynomial is `p(x) = Σⱼ cⱼ x^αⱼ`. Its terms are sorted in ascending
/// [`MonomialOrder`], with no repeated monomial and no zero coefficient, so
/// two equal polynomials in the same order have the same terms. The exponent
/// vectors are stored as one flat array of rows of length `n`. All arithmetic
/// is exact; floating-point types do not implement [`ExactCoefficient`], so use
/// [`ApproxSparsePolynomial`] for them.
///
/// Products merge the term products with a binary heap of
/// `min(s, t)` entries for operands of `s` and `t` terms, in
/// `O(s t log min(s, t))` monomial comparisons.
///
/// # Examples
///
/// ```
/// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
///
/// // p(x, y) = 1 + 2x, so p(3, 4) = 7.
/// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
/// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
/// assert_eq!(p.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. C. Johnson, "Sparse polynomial arithmetic", *ACM SIGSAM Bulletin* 8(3),
/// 1974, pp. 63–71.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactSparsePolynomial<C: ExactCoefficient> {
    data: Data<C>,
}
/// Represents a polynomial in `n` variables with real or complex
/// floating-point coefficients, as a list of nonzero terms.
///
/// The polynomial is `p(x) = Σⱼ cⱼ x^αⱼ`. Its terms are sorted in ascending
/// [`MonomialOrder`], with no repeated monomial. Construction and arithmetic
/// remove only coefficients that are exactly zero, and no tolerance removes
/// small ones. The exponent vectors are stored as one flat array of rows of
/// length `n`.
///
/// Coefficient arithmetic is in working precision. A sum or product that
/// overflows is a `NumericalInstability` error; complex products whose parts
/// come close to overflow or underflow are formed in extended-exponent
/// arithmetic. Products merge the term products with a binary heap, as for
/// [`ExactSparsePolynomial`].
///
/// # Examples
///
/// ```
/// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
///
/// // p(x, y) = 1 + 2x, so p(3, 4) = 7.
/// let order = MonomialOrder::GradedLexicographic;
/// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
/// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
/// assert!((p.evaluate(&[3.0_f64, 4.0])? - 7.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ApproxSparsePolynomial<C: NumericScalar> {
    data: Data<C>,
}

macro_rules! common {
    ($name:ident,$bound:ident,$arithmetic:ty) => {
        impl<C: $bound> $name<C> {
            /// Returns the number of variables `n`, including variables that appear in
            /// no term.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// assert_eq!(p.dimension(), 2);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub const fn dimension(&self) -> usize {
                self.data.dimension
            }
            /// Returns the monomial order of the terms.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// assert_eq!(p.order(), MonomialOrder::GradedLexicographic);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub const fn order(&self) -> MonomialOrder {
                self.data.order
            }
            /// Returns the nonzero terms in ascending monomial order.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// assert_eq!(p.terms().len(), 2);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn terms(&self) -> SparseTerms<'_, C> {
                self.data.terms()
            }
            /// Returns the exponent vectors as one flat array, `dimension()` entries
            /// per term, in term order.
            pub fn exponents(&self) -> &[usize] {
                &self.data.exponents
            }
            /// Returns the coefficients in term order.
            pub fn coefficients(&self) -> &[C] {
                &self.data.coefficients
            }
            /// Returns the largest total degree of a term; zero for the zero polynomial.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// assert_eq!(p.degree(), 1);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn degree(&self) -> usize {
                self.data.degrees.iter().copied().max().unwrap_or(0)
            }
            /// Returns `true` for the zero polynomial, which has no terms.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// assert!(!p.is_zero());
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn is_zero(&self) -> bool {
                self.data.coefficients.is_empty()
            }
            /// Returns the sum `self + other`.
            ///
            /// The two term lists are merged in one pass, and terms that cancel exactly
            /// are removed.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if the numbers of variables differ.
            /// - `InvalidInput` if the monomial orders differ.
            /// - `NumericalInstability` if a floating-point coefficient overflows.
            /// - `AllocationFailure` if the result cannot be allocated.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// let result = p.try_add(&p)?;
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 14.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn try_add(&self, other: &Self) -> Result<Self, SciOxideError> {
                self.check(other)?;
                Ok(self.wrap(
                    self.data
                        .add::<$arithmetic>(&other.data, false, usize::MAX)?,
                ))
            }
            /// Returns the difference `self − other`.
            ///
            /// # Errors
            ///
            /// As [`Self::try_add`].
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// let result = p.try_sub(&p)?;
            /// assert!(result.is_zero());
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn try_sub(&self, other: &Self) -> Result<Self, SciOxideError> {
                self.check(other)?;
                Ok(self.wrap(
                    self.data
                        .add::<$arithmetic>(&other.data, true, usize::MAX)?,
                ))
            }
            /// Returns the product `self · other`.
            ///
            /// The term products are merged in ascending order with a binary heap of
            /// `min(s, t)` entries for operands of `s` and `t` terms, so the cost is
            /// `O(s t log min(s, t))` monomial comparisons and `s t` coefficient
            /// products. Like terms are added as they leave the heap. There is no limit
            /// on the number of terms.
            ///
            /// # Errors
            ///
            /// - As [`Self::try_add`].
            /// - `SizeOverflow` if a total degree does not fit in `usize`.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// let result = p.try_mul(&p)?;
            /// // p(3, 4)² = 49.
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 49.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn try_mul(&self, other: &Self) -> Result<Self, SciOxideError> {
                self.check(other)?;
                Ok(self.wrap(self.data.mul::<$arithmetic>(
                    &other.data,
                    usize::MAX,
                    ExecutionContext::sequential_native(),
                )?))
            }
            /// Returns the partial derivative `∂p/∂x_axis`.
            ///
            /// # Errors
            ///
            /// - `OutOfBounds` if `axis` is not below the number of variables.
            /// - `NumericalInstability` if a floating-point coefficient overflows.
            /// - `AllocationFailure` if the result cannot be allocated.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// // ∂p/∂x = 2.
            /// let result = p.derivative(0)?;
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 2.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn derivative(&self, axis: usize) -> Result<Self, SciOxideError> {
                Ok(self.wrap(self.data.derivative::<$arithmetic>(axis)?))
            }
            /// Returns the polynomial with its variables permuted: exponent `i` of each
            /// term of the result is exponent `permutation[i]` of `self`.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `permutation` is not a permutation of `0..n`.
            /// - `AllocationFailure` if the result cannot be allocated.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// // Swapping x and y gives 1 + 2y.
            /// let result = p.permute_variables(&[1, 0])?;
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 9.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn permute_variables(&self, permutation: &[usize]) -> Result<Self, SciOxideError> {
                Ok(self.wrap(self.data.permute::<$arithmetic>(permutation)?))
            }
            /// Returns the same polynomial with its terms sorted in another monomial
            /// order.
            ///
            /// # Errors
            ///
            /// - `AllocationFailure` if the result cannot be allocated.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// let result = p.with_order(MonomialOrder::Lexicographic)?;
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 7.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn with_order(&self, order: MonomialOrder) -> Result<Self, SciOxideError> {
                Ok(self.wrap(self.data.canonicalize::<$arithmetic>(order, None)?))
            }
            /// Returns `p(r₀(y), …, rₙ₋₁(y))`, replacing each variable `xᵢ` of `self` by
            /// the polynomial `replacements[i]` in `target_dimension` variables `y`.
            ///
            /// Each term `c x^α` is expanded as `c ∏ᵢ rᵢ^αᵢ`, with the powers formed by
            /// repeated squaring and kept for later terms. The expansions are then
            /// combined and sorted. No limit applies to the number of terms; use
            /// [`Self::substitute_with_options`] to set one.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `replacements.len()` is not the number of variables of
            ///   `self`, or a replacement has another number of variables than
            ///   `target_dimension` or another monomial order.
            /// - `NumericalInstability` if a floating-point coefficient overflows.
            /// - `SizeOverflow` or `AllocationFailure` if the result does not fit in
            ///   memory.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
            ///
            /// // p(x, y) = 1 + 2x, so p(p, p) = 1 + 2(1 + 2x) = 3 + 4x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let p = ApproxSparsePolynomial::from_terms(2, order, [([0, 0], 1.0_f64), ([1, 0], 2.0)])?;
            /// let result = p.substitute(&[p.clone(), p.clone()], 2)?;
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 15.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn substitute(
                &self,
                replacements: &[Self],
                target_dimension: usize,
            ) -> Result<Self, SciOxideError> {
                self.substitute_with_options(
                    replacements,
                    target_dimension,
                    SparseExpansionLimits::default(),
                )
            }
            /// Computes the substitution of [`Self::substitute`] with every
            /// intermediate result checked against `limits`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::substitute`].
            /// - `ResourceLimit` if an intermediate result has more terms than `limits`
            ///   allows.
            ///
            /// ```
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder, SparseExpansionLimits};
            ///
            /// // p(x, y) = 1 + 2x, so p(p, p) = 3 + 4x has two terms.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let p = ApproxSparsePolynomial::from_terms(2, order, [([0, 0], 1.0_f64), ([1, 0], 2.0)])?;
            /// let replacements = [p.clone(), p.clone()];
            /// let result = p.substitute_with_options(&replacements, 2, SparseExpansionLimits::new(2))?;
            /// assert_eq!(result.coefficients().len(), 2);
            /// assert!(
            ///     p.substitute_with_options(&replacements, 2, SparseExpansionLimits::new(1))
            ///         .is_err()
            /// );
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn substitute_with_options(
                &self,
                replacements: &[Self],
                target_dimension: usize,
                limits: SparseExpansionLimits,
            ) -> Result<Self, SciOxideError> {
                self.substitute_impl(
                    replacements,
                    target_dimension,
                    limits,
                    ExecutionContext::sequential_native(),
                )
            }
            /// Computes the substitution of [`Self::substitute_with_options`] and writes it
            /// to `output`, checking `context` for cancellation and deadlines.
            ///
            /// The method allocates the expansion and the result.
            ///
            /// # Errors
            ///
            /// - As [`Self::substitute_with_options`].
            /// - `ExecutionStopped` on cancellation or deadline.
            ///
            /// ```
            /// use scioxide_core::ExecutionContext;
            /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder, SparseExpansionLimits};
            ///
            /// // p(x, y) = 1 + 2x.
            /// let order = MonomialOrder::GradedLexicographic;
            /// let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0)];
            /// let p = ApproxSparsePolynomial::from_terms(2, order, terms)?;
            /// let limits = SparseExpansionLimits::new(16);
            /// let cx = ExecutionContext::sequential();
            /// let mut result = p.clone();
            /// p.substitute_into(&[p.clone(), p.clone()], 2, limits, &mut result, cx)?;
            /// assert!((result.evaluate(&[3.0_f64, 4.0])? - 15.0).abs() < 1e-12);
            /// # Ok::<(), scioxide_core::SciOxideError>(())
            /// ```
            pub fn substitute_into(
                &self,
                replacements: &[Self],
                target_dimension: usize,
                limits: SparseExpansionLimits,
                output: &mut Self,
                context: ExecutionContext<'_>,
            ) -> Result<(), SciOxideError> {
                context.check()?;
                let staged =
                    self.substitute_impl(replacements, target_dimension, limits, context)?;
                context.check()?;
                *output = staged;
                Ok(())
            }
            fn substitute_impl(
                &self,
                replacements: &[Self],
                target_dimension: usize,
                limits: SparseExpansionLimits,
                context: ExecutionContext<'_>,
            ) -> Result<Self, SciOxideError> {
                context.check()?;
                let refs = crate::util::try_collect(
                    replacements.iter().map(|r| Ok(&r.data)),
                    scioxide_core::quantity!("replacements"),
                )?;
                Ok(self.wrap(self.data.substitute::<$arithmetic>(
                    &refs,
                    target_dimension,
                    limits,
                    context,
                )?))
            }
        }
    };
}
common!(ExactSparsePolynomial, ExactCoefficient, Exact);
common!(ApproxSparsePolynomial, NumericScalar, Approx);
impl<C: ExactCoefficient> ExactSparsePolynomial<C> {
    /// Returns `p(x)` at `point`, computed exactly.
    ///
    /// The terms are summed in ascending order, and each power `xᵢ^αᵢ` is formed
    /// by repeated squaring. See [`Self::as_view`] for many points.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point.len()` is not the number of variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x, so p(3, 4) = 7.
    /// let terms = [([0, 0], Q(1, 1)), ([1, 0], Q(2, 1))];
    /// let p = ExactSparsePolynomial::from_terms(2, MonomialOrder::GradedLexicographic, terms)?;
    /// assert_eq!(p.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: &[C]) -> Result<C, SciOxideError> {
        self.as_view().evaluate(point)
    }
    /// Builds a sparse polynomial from flat exponent rows and coefficients.
    ///
    /// Term `j` has exponents `exponents[j·dimension..(j + 1)·dimension]` and
    /// coefficient `coefficients[j]`, so `exponents.len()` must equal
    /// `dimension · coefficients.len()`. With `dimension = 0` the rows are
    /// empty and the terms are constants. The terms may come in any order:
    /// they are sorted, terms with the same monomial are added in input order,
    /// and zero coefficients are removed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `exponents.len()` is not
    ///   `dimension · coefficients.len()`.
    /// - `SizeOverflow` if a total degree does not fit in `usize`.
    /// - `AllocationFailure` if the polynomial cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    ///
    /// // p(x, y) = 1 + 2x from rows (0, 0) and (1, 0).
    /// let order = MonomialOrder::GradedLexicographic;
    /// let p = ExactSparsePolynomial::new(2, order, vec![0, 0, 1, 0], vec![Q(1, 1), Q(2, 1)])?;
    /// assert_eq!(p.evaluate(&[Q(3, 1), Q(4, 1)])?, Q(7, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        dimension: usize,
        order: MonomialOrder,
        exponents: Vec<usize>,
        coefficients: Vec<C>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            data: Data::new::<Exact>(dimension, order, exponents, coefficients)?,
        })
    }
    fn wrap(&self, data: Data<C>) -> Self {
        Self { data }
    }
    fn check(&self, other: &Self) -> Result<(), SciOxideError> {
        self.data.compatible(&other.data)
    }
}
impl<C: NumericScalar> ApproxSparsePolynomial<C> {
    fn native_coefficients(&self) -> bool {
        self.data.native_coefficients
    }
    fn axis_metadata(&self) -> &[AxisPowers] {
        &self.data.axes
    }
    /// Returns `p(x)` at `point`.
    ///
    /// The powers `xᵢ^(2ᵏ)` are computed once per variable, up to the largest
    /// exponent of that variable, and each term multiplies its coefficient by
    /// the powers in the binary expansion of its exponents. The terms are
    /// summed in ascending order. If a power, product or partial sum comes
    /// close to overflow or underflow, the whole evaluation is repeated in
    /// extended-exponent arithmetic and the result is rounded once. The call
    /// allocates one value per bit of the largest exponent of each variable;
    /// [`Self::evaluation_plan`] evaluates without allocating and at many
    /// points.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point.len()` is not the number of variables.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `NumericalInstability` if the value overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the scratch cannot be sized or
    ///   allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, Monomial, MonomialOrder};
    ///
    /// // p(x) = 3x², so p(2) = 12.
    /// let terms = vec![(Monomial::new(&[2])?, 3.0)];
    /// let p = ApproxSparsePolynomial::from_terms(1, MonomialOrder::Lexicographic, terms)?;
    /// assert!((p.evaluate(&[2.0_f64])? - 12.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: &[C]) -> Result<C, SciOxideError> {
        let plan = self.evaluation_plan(crate::PolynomialExecutionOptions::new())?;
        let mut scratch = scioxide_core::backend::try_filled(
            plan.workspace_requirements(1)?.scratch_len(),
            C::ZERO,
            scioxide_core::quantity!("evaluation_scratch"),
        )?;
        plan.evaluate(point, &mut scratch)
    }
    /// Builds a sparse polynomial from flat exponent rows and coefficients.
    ///
    /// Term `j` has exponents `exponents[j·dimension..(j + 1)·dimension]` and
    /// coefficient `coefficients[j]`, so `exponents.len()` must equal
    /// `dimension · coefficients.len()`. The terms may come in any order: they
    /// are sorted, terms with the same monomial are added in input order, and
    /// coefficients that are exactly zero are removed.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `exponents.len()` is not
    ///   `dimension · coefficients.len()`.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if the sum of like terms overflows.
    /// - `SizeOverflow` if a total degree does not fit in `usize`.
    /// - `AllocationFailure` if the polynomial cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // p(x, y) = 1 + 2x from rows (0, 0) and (1, 0).
    /// let order = MonomialOrder::GradedLexicographic;
    /// let (exponents, coefficients) = (vec![0, 0, 1, 0], vec![1.0_f64, 2.0]);
    /// let p = ApproxSparsePolynomial::new(2, order, exponents, coefficients)?;
    /// assert!((p.evaluate(&[3.0_f64, 4.0])? - 7.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn new(
        dimension: usize,
        order: MonomialOrder,
        exponents: Vec<usize>,
        coefficients: Vec<C>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            data: Data::new::<Approx>(dimension, order, exponents, coefficients)?,
        })
    }
    fn wrap(&self, data: Data<C>) -> Self {
        Self { data }
    }
    fn check(&self, other: &Self) -> Result<(), SciOxideError> {
        self.data.compatible(&other.data)
    }
}

fn collect_terms<C, E: AsRef<[usize]>>(
    dimension: usize,
    terms: impl IntoIterator<Item = (E, C)>,
) -> Result<(Vec<usize>, Vec<C>), SciOxideError> {
    let terms = terms.into_iter();
    let capacity = terms.size_hint().0;
    let mut exponents = try_with_capacity(
        scioxide_core::backend::checked_mul(
            dimension,
            capacity,
            scioxide_core::quantity!("exponents"),
        )?,
        scioxide_core::quantity!("exponents"),
    )?;
    let mut coefficients = try_with_capacity(capacity, scioxide_core::quantity!("sparse_terms"))?;
    for (m, c) in terms {
        let e = m.as_ref();
        if e.len() != dimension {
            return Err(SciOxideError::dimension_mismatch(dimension, e.len()));
        }
        exponents.try_reserve(dimension).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("exponents"),
                exponents.len().saturating_add(dimension),
            )
        })?;
        coefficients.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("sparse_terms"),
                coefficients.len().saturating_add(1),
            )
        })?;
        exponents.extend_from_slice(e);
        coefficients.push(c);
    }
    Ok((exponents, coefficients))
}
impl<C: ExactCoefficient> ExactSparsePolynomial<C> {
    /// Builds a sparse polynomial from `(exponents, coefficient)` pairs.
    ///
    /// The exponents may be arrays, slices or [`Monomial`]s. The pairs are
    /// processed as by [`Self::new`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if an exponent vector does not have `dimension`
    ///   entries.
    /// - `SizeOverflow` if a total degree does not fit in `usize`.
    /// - `AllocationFailure` if the polynomial cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{ExactSparsePolynomial, MonomialOrder};
    /// # mod field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    ///
    /// // p(x) = 3x².
    /// let order = MonomialOrder::Lexicographic;
    /// let p = ExactSparsePolynomial::from_terms(1, order, [([2], field::Q(3, 1))])?;
    /// assert_eq!(p.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_terms<E: AsRef<[usize]>>(
        dimension: usize,
        order: MonomialOrder,
        terms: impl IntoIterator<Item = (E, C)>,
    ) -> Result<Self, SciOxideError> {
        let (exponents, coefficients) = collect_terms(dimension, terms)?;
        Self::new(dimension, order, exponents, coefficients)
    }
}
impl<C: NumericScalar> ApproxSparsePolynomial<C> {
    /// Builds a sparse polynomial from `(exponents, coefficient)` pairs.
    ///
    /// The exponents may be arrays, slices or [`Monomial`]s. The pairs are
    /// processed as by [`Self::new`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if an exponent vector does not have `dimension`
    ///   entries.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if the sum of like terms overflows.
    /// - `SizeOverflow` if a total degree does not fit in `usize`.
    /// - `AllocationFailure` if the polynomial cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{ApproxSparsePolynomial, MonomialOrder};
    ///
    /// // p(x) = 3x².
    /// let order = MonomialOrder::Lexicographic;
    /// let p = ApproxSparsePolynomial::from_terms(1, order, [([2], 3.0_f64)])?;
    /// assert_eq!(p.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_terms<E: AsRef<[usize]>>(
        dimension: usize,
        order: MonomialOrder,
        terms: impl IntoIterator<Item = (E, C)>,
    ) -> Result<Self, SciOxideError> {
        let (exponents, coefficients) = collect_terms(dimension, terms)?;
        Self::new(dimension, order, exponents, coefficients)
    }
}
fn validate_borrowed<C: Clone, A: SparseArithmetic<C>>(
    dimension: usize,
    order: MonomialOrder,
    exponents: &[usize],
    coefficients: &[C],
) -> Result<(), SciOxideError> {
    let expected = scioxide_core::backend::checked_mul(
        dimension,
        coefficients.len(),
        scioxide_core::quantity!("exponents"),
    )?;
    if expected != exponents.len() {
        return Err(SciOxideError::dimension_mismatch(expected, exponents.len()));
    }
    let mut previous = None;
    for (i, c) in coefficients.iter().enumerate() {
        A::validate(c)?;
        let m = Monomial::new(&exponents[i * dimension..(i + 1) * dimension])?;
        if A::is_zero(c) || previous.is_some_and(|p| order.compare(&p, &m) != Ordering::Less) {
            return Err(invalid(
                scioxide_core::argument!("terms"),
                InvalidInputReason::OutOfRange,
                "noncanonical sparse terms",
            ));
        }
        previous = Some(m);
    }
    Ok(())
}
fn validate_approx_terms<C: NumericScalar>(
    dimension: usize,
    order: MonomialOrder,
    exponents: &[usize],
    coefficients: &[C],
) -> Result<(), SciOxideError> {
    validate_borrowed::<C, Approx>(dimension, order, exponents, coefficients)
}
