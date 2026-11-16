//! Exact polynomial and rational-function algebra.
//!
//! SciOxide implements [`ExactCoefficient`] and [`ExactField`] for no type, and
//! never for floating-point types. You implement them for your own integer,
//! rational, finite-field or symbolic coefficient type.

use alloc::vec::Vec;
use core::ops::{Add, Div, Mul, Neg, Sub};

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{InvalidInputReason, SciOxideError};

/// Defines the coefficient ring of exact polynomial algebra.
///
/// The type must form a ring with identity under `+`, `-`, `*` and unary `-`,
/// and `==` must be exact equality. Approximate floating-point
/// equality does not satisfy this, so the exact API cannot be used with `f32`
/// or `f64`:
///
/// ```compile_fail
/// use scioxide_poly::ExactPolynomial;
///
/// assert!(ExactPolynomial::<f64>::new(vec![1.0_f64, 2.0]).is_ok());
/// ```
///
/// With a rational type `Q(numerator, denominator)` that implements this trait:
///
/// ```
/// use scioxide_poly::*;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// // p(x) = 1 + 2x + x², so p(1) = 4.
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// assert_eq!(p.evaluate(Q(1, 1)), Q(4, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait ExactCoefficient:
    Clone
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Returns the additive identity.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// assert_eq!(<Q as ExactCoefficient>::zero(), Q(0, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn zero() -> Self;

    /// Returns the multiplicative identity.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// assert_eq!(<Q as ExactCoefficient>::one(), Q(1, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn one() -> Self;

    /// Returns `value · 1`, the image of a nonnegative integer in the ring.
    ///
    /// [`ExactPolynomial::derivative`] uses it for the factors `k` in `k aₖ`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// assert_eq!(<Q as ExactCoefficient>::from_usize(3), Q(3, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn from_usize(value: usize) -> Self;

    /// Returns whether this coefficient equals zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// assert!(<Q as ExactCoefficient>::is_zero(&Q(0, 1)));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

/// Defines the coefficient field of exact polynomial division.
///
/// `/` is only called with a nonzero divisor and must satisfy `(a / b) * b == a`.
/// Long division by a divisor that is not monic checks this law at every step
/// and returns `NumericalInstability` (`Breakdown`) if it fails.
///
/// ```
/// use scioxide_poly::ExactField;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// fn quotient<F: ExactField>(a: F, b: F) -> F { a / b }
/// let a = Q(3, 2);
/// let b = Q(5, 7);
/// assert_eq!(quotient(a, b) * b, a);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait ExactField: ExactCoefficient + Div<Output = Self> {}

/// Stores a dense univariate polynomial `p(x) = a₀ + a₁x + … + aₙxⁿ` with exact
/// coefficients.
///
/// Coefficients are stored in ascending order of degree, without trailing
/// zeros. The zero polynomial is stored as the single coefficient `0` and has
/// degree 0. Equal polynomials therefore have equal storage, and `==` compares
/// them exactly.
///
/// Addition, subtraction and the derivative take O(n) coefficient operations.
/// Multiplication is schoolbook below 32 coefficients in the shorter operand and
/// Karatsuba above, `O(n^1.585)` operations. Division, GCD and LCM need an
/// [`ExactField`]. For products and shifts into reused storage and for
/// multipoint work, see [`exact_multiply_into`](crate::exact_multiply_into),
/// [`exact_taylor_shift_into`](crate::exact_taylor_shift_into) and
/// [`ExactPolynomialProductTree`](crate::ExactPolynomialProductTree).
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// // (1 + x)² = 1 + 2x + x² divided by 1 + x is 1 + x, remainder 0.
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// let b = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
/// let (q, r) = p.div_rem(&b)?;
/// assert_eq!(q, b);
/// assert!(r.is_zero());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
/// Cambridge University Press, 2013, chapters 2, 3 and 8.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct ExactPolynomial<C: ExactCoefficient> {
    coefficients: Vec<C>,
}

impl<C: ExactCoefficient> ExactPolynomial<C> {
    /// Clones the coefficient storage fallibly, without arithmetic.
    pub(crate) fn try_copy(&self) -> Result<Self, SciOxideError> {
        // Exact coefficients are `Clone`, not `Copy`, so the core slice copy
        // does not apply; reserve fallibly and clone each coefficient.
        let mut coefficients = scioxide_core::backend::try_with_capacity(
            self.coefficients.len(),
            scioxide_core::quantity!("exact_polynomial_copy"),
        )?;
        coefficients.extend_from_slice(&self.coefficients);
        Ok(Self { coefficients })
    }

    /// Scales the coefficients in place, reusing their allocation.
    pub(crate) fn scaled(mut self, factor: C) -> Result<Self, SciOxideError> {
        if factor != C::one() {
            for coefficient in &mut self.coefficients {
                *coefficient = core::mem::replace(coefficient, C::zero()) * factor.clone();
            }
        }
        Self::new(self.coefficients)
    }

    /// Creates a polynomial from coefficients in ascending order of degree.
    ///
    /// Trailing zero coefficients are removed. The vector is reused as storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `coefficients` is empty.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.evaluate(Q(1, 1)), Q(4, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(mut coefficients: Vec<C>) -> Result<Self, SciOxideError> {
        if coefficients.is_empty() {
            return Err(crate::util::empty_error(scioxide_core::argument!(
                "coefficients"
            )));
        }
        // Canonical form: no trailing zeros except the single `[0]` of the zero
        // polynomial. `degree`, `leading_coefficient`, equality and the division
        // loops below all rely on the last stored coefficient being nonzero.
        while coefficients.len() > 1 && coefficients.last().is_some_and(C::is_zero) {
            coefficients.pop();
        }
        Ok(Self { coefficients })
    }

    /// Creates the constant polynomial `p(x) = value`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::constant(Q(3, 1))?;
    /// assert_eq!(p.evaluate(Q(1, 1)), Q(3, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the coefficient cannot be stored.
    pub fn constant(value: C) -> Result<Self, SciOxideError> {
        let coefficients = try_filled(
            1,
            value,
            scioxide_core::quantity!("exact_polynomial_constant"),
        )?;
        Self::new(coefficients)
    }

    /// Returns the coefficients in ascending order of degree.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.coefficients(), &[Q(1, 1), Q(2, 1), Q(1, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coefficients(&self) -> &[C] {
        &self.coefficients
    }

    /// Returns the degree. The zero polynomial has degree 0.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }

    /// Returns whether this is the zero polynomial.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert!(!p.is_zero());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn is_zero(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0].is_zero()
    }

    /// Returns the leading coefficient `aₙ`, which is `0` only for the zero
    /// polynomial.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.leading_coefficient(), &Q(1, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn leading_coefficient(&self) -> &C {
        &self.coefficients[self.coefficients.len() - 1]
    }

    /// Evaluates `p(point)` by Horner's rule, in n multiplications and n
    /// additions.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.evaluate(Q(1, 1)), Q(4, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: C) -> C {
        self.coefficients
            .iter()
            .rev()
            .cloned()
            .fold(C::zero(), |value, coefficient| {
                value * point.clone() + coefficient
            })
    }

    /// Returns the sum `self + other`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.try_add(&p)?;
    /// assert_eq!(result.evaluate(Q(1, 1)), Q(8, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn try_add(&self, other: &Self) -> Result<Self, SciOxideError> {
        let length = self.coefficients.len().max(other.coefficients.len());
        let mut coefficients =
            try_with_capacity(length, scioxide_core::quantity!("exact_polynomial_sum"))?;
        coefficients.extend((0..length).map(|index| {
            self.coefficients
                .get(index)
                .cloned()
                .unwrap_or_else(C::zero)
                + other
                    .coefficients
                    .get(index)
                    .cloned()
                    .unwrap_or_else(C::zero)
        }));
        Self::new(coefficients)
    }

    /// Returns the difference `self − other`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.try_sub(&p)?;
    /// assert_eq!(result.evaluate(Q(1, 1)), Q(0, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn try_sub(&self, other: &Self) -> Result<Self, SciOxideError> {
        let length = self.coefficients.len().max(other.coefficients.len());
        let mut coefficients = try_with_capacity(
            length,
            scioxide_core::quantity!("exact_polynomial_difference"),
        )?;
        coefficients.extend((0..length).map(|index| {
            self.coefficients
                .get(index)
                .cloned()
                .unwrap_or_else(C::zero)
                - other
                    .coefficients
                    .get(index)
                    .cloned()
                    .unwrap_or_else(C::zero)
        }));
        Self::new(coefficients)
    }

    /// Returns the product `self · other`.
    ///
    /// Uses schoolbook multiplication when the shorter operand has fewer than 32
    /// coefficients and Karatsuba's method otherwise.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.try_mul(&p)?;
    /// assert_eq!(result.evaluate(Q(1, 1)), Q(16, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the product length overflows `usize`.
    /// - `AllocationFailure` if the product or its scratch cannot be stored.
    pub fn try_mul(&self, other: &Self) -> Result<Self, SciOxideError> {
        if self.is_zero() || other.is_zero() {
            return Self::constant(C::zero());
        }
        Self::new(crate::fast_exact::multiply(
            self.coefficients(),
            other.coefficients(),
        )?)
    }

    /// Returns the formal derivative `p′(x) = Σₖ k aₖ xᵏ⁻¹`.
    ///
    /// The factor `k` is [`ExactCoefficient::from_usize`]`(k)`, so in a field of
    /// characteristic `c` the terms with `c | k` vanish.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.derivative()?.coefficients(), &[Q(2, 1), Q(2, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn derivative(&self) -> Result<Self, SciOxideError> {
        if self.degree() == 0 {
            return Self::constant(C::zero());
        }
        let mut coefficients = try_with_capacity(
            self.degree(),
            scioxide_core::quantity!("exact_polynomial_derivative"),
        )?;
        coefficients.extend(
            self.coefficients
                .iter()
                .enumerate()
                .skip(1)
                .map(|(degree, coefficient)| coefficient.clone() * C::from_usize(degree)),
        );
        Self::new(coefficients)
    }
}

impl<C: ExactField> ExactPolynomial<C> {
    /// Divides by `divisor`, returning the quotient `q` and remainder `r`.
    ///
    /// `self = q · divisor + r` with `deg r < deg divisor`, or `r = 0`. Long
    /// division takes O((m − n + 1) n) field operations for degrees `m ≥ n`. A
    /// monic divisor needs no coefficient division.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let b = ExactPolynomial::new(vec![Q(1, 1), Q(1, 1)])?;
    /// let (q, r) = p.div_rem(&b)?;
    /// assert_eq!(q, b);
    /// assert!(r.is_zero());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `divisor` is the zero polynomial.
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type violates `(a / b) * b == a`.
    /// - `AllocationFailure` if the quotient or remainder cannot be stored.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), SciOxideError> {
        let quotient_length = self.quotient_length(divisor)?;
        let mut quotient = try_filled(
            quotient_length.max(1),
            C::zero(),
            scioxide_core::quantity!("exact_polynomial_division_quotient"),
        )?;
        let remainder = self.divide(divisor, Some(&mut quotient))?;
        Ok((Self::new(quotient)?, remainder))
    }

    fn quotient_length(&self, divisor: &Self) -> Result<usize, SciOxideError> {
        if divisor.is_zero() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("divisor"),
                InvalidInputReason::OutOfRange,
                "the divisor polynomial must be nonzero",
            ));
        }
        Ok(if self.is_zero() || self.degree() < divisor.degree() {
            0
        } else {
            self.degree() - divisor.degree() + 1
        })
    }

    /// Computes only the remainder, without storing a quotient.
    pub(crate) fn remainder(&self, divisor: &Self) -> Result<Self, SciOxideError> {
        self.divide(divisor, None)
    }

    fn divide(
        &self,
        divisor: &Self,
        mut quotient: Option<&mut [C]>,
    ) -> Result<Self, SciOxideError> {
        let quotient_length = self.quotient_length(divisor)?;
        let mut remainder = self.try_copy()?;
        let monic = divisor.leading_coefficient() == &C::one();
        // Each quotient position is visited once, independently of field laws.
        for degree in (0..quotient_length).rev() {
            let leading_slot = degree + divisor.degree();
            let leading = core::mem::replace(&mut remainder.coefficients[leading_slot], C::zero());
            let factor = if monic {
                leading
            } else {
                let factor = leading.clone() / divisor.leading_coefficient().clone();
                if !(leading - factor.clone() * divisor.leading_coefficient().clone()).is_zero() {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "coefficient division failed to cancel the leading remainder term",
                    ));
                }
                factor
            };
            for (index, coefficient) in divisor
                .coefficients
                .iter()
                .take(divisor.degree())
                .enumerate()
            {
                let destination = &mut remainder.coefficients[degree + index];
                *destination = core::mem::replace(destination, C::zero())
                    - factor.clone() * coefficient.clone();
            }
            if let Some(values) = quotient.as_deref_mut() {
                values[degree] = factor;
            }
        }
        Self::new(remainder.coefficients)
    }

    /// Returns the monic greatest common divisor of `self` and `other`.
    ///
    /// Uses the Euclidean algorithm, then divides by the leading coefficient.
    /// `gcd(p, 0)` is `monic(p)` and `gcd(0, 0) = 0`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.gcd(&p)?;
    /// assert_eq!(result, p);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `AllocationFailure` if an intermediate remainder cannot be stored.
    pub fn gcd(&self, other: &Self) -> Result<Self, SciOxideError> {
        let mut left = self.try_copy()?;
        let mut right = other.try_copy()?;
        // Euclid's algorithm is safe here because coefficients are exact; normalizing
        // to monic makes the result unique (and gcd(0, 0) = 0).
        while !right.is_zero() {
            let remainder = left.remainder(&right)?;
            left = right;
            right = remainder;
        }
        left.monic()
    }

    /// Returns the monic least common multiple of `self` and `other`.
    ///
    /// Computes `(self / gcd) · other`, made monic. If either operand is zero,
    /// the result is zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let result = p.lcm(&p)?;
    /// assert_eq!(result, p);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent, so that the GCD does not divide `self`.
    /// - `SizeOverflow` or `AllocationFailure` if an intermediate result cannot
    ///   be stored.
    pub fn lcm(&self, other: &Self) -> Result<Self, SciOxideError> {
        if self.is_zero() || other.is_zero() {
            return Self::constant(C::zero());
        }
        let gcd = self.gcd(other)?;
        let (quotient, remainder) = self.div_rem(&gcd)?;
        if !remainder.is_zero() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "exact GCD did not divide the operand",
            ));
        }
        quotient.try_mul(other)?.monic()
    }

    /// Returns the polynomial divided by its leading coefficient.
    ///
    /// The zero polynomial stays zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// assert_eq!(p.monic()?.leading_coefficient(), &Q(1, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn monic(&self) -> Result<Self, SciOxideError> {
        if self.is_zero() {
            return Self::constant(C::zero());
        }
        let leading = self.leading_coefficient().clone();
        if leading == C::one() {
            return self.try_copy();
        }
        let mut coefficients = try_with_capacity(
            self.coefficients.len(),
            scioxide_core::quantity!("exact_monic_polynomial_coefficients"),
        )?;
        coefficients.extend(
            self.coefficients
                .iter()
                .cloned()
                .map(|coefficient| coefficient / leading.clone()),
        );
        Self::new(coefficients)
    }
}

/// Stores a rational function `p(x) / q(x)` over an exact coefficient field.
///
/// The numerator and denominator are coprime and the denominator is monic, so
/// equal rational functions have equal storage. The zero function is stored as
/// `0 / 1`. Every operation cancels the GCD of the new numerator and
/// denominator with the Euclidean algorithm.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
/// # use doctest_field::Q;
/// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
/// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
/// assert_eq!(r.evaluate(Q(1, 1))?, Q(4, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct ExactRationalFunction<C: ExactField> {
    numerator: ExactPolynomial<C>,
    denominator: ExactPolynomial<C>,
}

impl<C: ExactField> ExactRationalFunction<C> {
    /// Creates `numerator / denominator` in lowest terms with a monic
    /// denominator.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `denominator` is the zero polynomial.
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `AllocationFailure` if the reduced polynomials cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// assert_eq!(r.evaluate(Q(1, 1))?, Q(4, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        numerator: ExactPolynomial<C>,
        denominator: ExactPolynomial<C>,
    ) -> Result<Self, SciOxideError> {
        if denominator.is_zero() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("denominator"),
                InvalidInputReason::OutOfRange,
                "the denominator polynomial must be nonzero",
            ));
        }
        // Canonical representation: coprime numerator and denominator with a monic
        // denominator, so structurally equal values compare equal. A zero numerator
        // has gcd = monic(denominator) and becomes `0 / 1`. The remainders can only be
        // nonzero if the coefficient type's field arithmetic is inconsistent.
        let gcd = numerator.gcd(&denominator)?;
        let (numerator, nr) = numerator.div_rem(&gcd)?;
        let (denominator, dr) = denominator.div_rem(&gcd)?;
        if !nr.is_zero() || !dr.is_zero() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "the exact GCD must divide both polynomials",
            ));
        }
        Self::normalize_coprime(numerator, denominator)
    }

    // Both inputs are already coprime and the denominator is nonzero.
    fn normalize_coprime(
        mut numerator: ExactPolynomial<C>,
        mut denominator: ExactPolynomial<C>,
    ) -> Result<Self, SciOxideError> {
        let leading = denominator.leading_coefficient().clone();
        if leading != C::one() {
            for coefficient in numerator
                .coefficients
                .iter_mut()
                .chain(&mut denominator.coefficients)
            {
                *coefficient = core::mem::replace(coefficient, C::zero()) / leading.clone();
            }
        }
        Ok(Self {
            numerator: ExactPolynomial::new(numerator.coefficients)?,
            denominator: ExactPolynomial::new(denominator.coefficients)?,
        })
    }

    /// Returns the sum `self + other` in lowest terms.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `SizeOverflow` or `AllocationFailure` if an intermediate product
    ///   cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// let result = r.try_add(&r)?;
    /// assert_eq!(result.evaluate(Q(1, 1))?, Q(8, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_add(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator
                .try_mul(&other.denominator)?
                .try_add(&other.numerator.try_mul(&self.denominator)?)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    /// Returns the difference `self − other` in lowest terms.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `SizeOverflow` or `AllocationFailure` if an intermediate product
    ///   cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// let result = r.try_sub(&r)?;
    /// assert_eq!(result.evaluate(Q(1, 1))?, Q(0, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_sub(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator
                .try_mul(&other.denominator)?
                .try_sub(&other.numerator.try_mul(&self.denominator)?)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    /// Returns the product `self · other` in lowest terms.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `SizeOverflow` or `AllocationFailure` if an intermediate product
    ///   cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// let result = r.try_mul(&r)?;
    /// assert_eq!(result.evaluate(Q(1, 1))?, Q(16, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_mul(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator.try_mul(&other.numerator)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    /// Returns the quotient `self / other` in lowest terms.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `other` is the zero function.
    /// - `NumericalInstability` (`Breakdown`) if the field division of the
    ///   coefficient type is inconsistent.
    /// - `SizeOverflow` or `AllocationFailure` if an intermediate product
    ///   cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// let result = r.try_div(&r)?;
    /// assert_eq!(result.evaluate(Q(1, 1))?, Q(1, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_div(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator.try_mul(&other.denominator)?,
            self.denominator.try_mul(&other.numerator)?,
        )
    }

    /// Returns the reciprocal `q / p`.
    ///
    /// Numerator and denominator are already coprime, so no GCD is computed.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `self` is the zero function.
    /// - `AllocationFailure` if the result cannot be stored.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// assert_eq!(r.reciprocal()?.evaluate(Q(1, 1))?, Q(1, 4));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn reciprocal(&self) -> Result<Self, SciOxideError> {
        if self.numerator.is_zero() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("denominator"),
                InvalidInputReason::OutOfRange,
                "the reciprocal requires a nonzero numerator",
            ));
        }
        Self::normalize_coprime(self.denominator.try_copy()?, self.numerator.try_copy()?)
    }

    /// Returns the numerator `p`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// assert_eq!(r.numerator().coefficients(), &[Q(1, 1), Q(2, 1), Q(1, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn numerator(&self) -> &ExactPolynomial<C> {
        &self.numerator
    }

    /// Returns the monic denominator `q`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// assert_eq!(r.denominator().coefficients(), &[Q(1, 1)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn denominator(&self) -> &ExactPolynomial<C> {
        &self.denominator
    }

    /// Evaluates `p(point) / q(point)` by Horner's rule.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// # mod doctest_field { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/exact_field.rs")); }
    /// # use doctest_field::Q;
    /// let p = ExactPolynomial::new(vec![Q(1, 1), Q(2, 1), Q(1, 1)])?;
    /// let r = ExactRationalFunction::new(p.clone(), ExactPolynomial::constant(Q(1, 1))?)?;
    /// assert_eq!(r.evaluate(Q(1, 1))?, Q(4, 1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `point` is a pole, that is `q(point) = 0`.
    pub fn evaluate(&self, point: C) -> Result<C, SciOxideError> {
        let denominator = self.denominator.evaluate(point.clone());
        if denominator.is_zero() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("point"),
                InvalidInputReason::OutOfRange,
                "evaluation point is a pole",
            ));
        }
        Ok(self.numerator.evaluate(point) / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_field::Q;
    fn q(value: i64) -> Q {
        Q(i128::from(value), 1)
    }
    #[test]
    fn exact_gcd_lcm_and_division_preserve_identities() {
        let common = ExactPolynomial::new(vec![-q(1), q(1)]).unwrap();
        let left = common
            .try_mul(&ExactPolynomial::new(vec![q(2), q(1)]).unwrap())
            .unwrap();
        let right = common
            .try_mul(&ExactPolynomial::new(vec![-q(3), q(1)]).unwrap())
            .unwrap();
        assert_eq!(left.gcd(&right).unwrap(), common);
        let lcm = left.lcm(&right).unwrap();
        assert!(lcm.div_rem(&left).unwrap().1.is_zero());
        assert!(lcm.div_rem(&right).unwrap().1.is_zero());
    }
}
