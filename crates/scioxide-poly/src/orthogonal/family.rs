//! Classical orthogonal polynomial families and their recurrence coefficients.
use super::admission::four_product_values;
use super::{recurrence, zeros};
use crate::admission::{
    Arithmetic, ArithmeticError, ArithmeticResult, Extended, Native, candidate, extended, require,
};
use crate::util::{finite_arithmetic, finite_input};
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::{Argument, RealScalar, Scaled, ScaledAccumulator, SciOxideError};

/// Selects a classical orthogonal polynomial family.
///
/// Each family is orthogonal with respect to a weight `w(x)` on its support and
/// uses the standard normalization of DLMF Table 18.3.1:
///
/// | Variant | Polynomials | Weight `w(x)` | Support |
/// |---|---|---|---|
/// | `ChebyshevT` | `Tₙ`, `Tₙ(cos θ) = cos nθ` | `(1 − x²)^(−1/2)` | `[−1, 1]` |
/// | `ChebyshevU` | `Uₙ`, `Uₙ(1) = n + 1` | `(1 − x²)^(1/2)` | `[−1, 1]` |
/// | `Legendre` | `Pₙ`, `Pₙ(1) = 1` | `1` | `[−1, 1]` |
/// | `Jacobi { alpha, beta }` | `Pₙ^(α,β)` | `(1 − x)^α (1 + x)^β` | `[−1, 1]` |
/// | `Gegenbauer { lambda }` | `Cₙ^(λ)` | `(1 − x²)^(λ − 1/2)` | `[−1, 1]` |
/// | `Laguerre` | `Lₙ` | `e^(−x)` | `[0, ∞)` |
/// | `GeneralizedLaguerre { alpha }` | `Lₙ^(α)` | `x^α e^(−x)` | `[0, ∞)` |
/// | `HermiteH` | physicists' `Hₙ` | `e^(−x²)` | `(−∞, ∞)` |
/// | `HermiteHe` | probabilists' `Heₙ` | `e^(−x²/2)` | `(−∞, ∞)` |
///
/// The parameters must satisfy `α, β > −1` and `λ > −1/2`, `λ ≠ 0`. Every
/// method checks them, and so does [`OrthogonalBasisSpec::new`](crate::OrthogonalBasisSpec::new).
/// The parameters are real; points and coefficients of expansions in a
/// prepared basis may be complex.
///
/// # Examples
///
/// ```
/// use scioxide_poly::OrthogonalFamily;
///
/// // P₂(x) = (3x² − 1)/2, so P₂(0.5) = −0.125.
/// let family = OrthogonalFamily::<f64>::Legendre;
/// assert!((family.evaluate(2, 0.5)? + 0.125).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §18.3 and §18.9. <https://dlmf.nist.gov/18>
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrthogonalFamily<R: RealScalar> {
    /// Chebyshev polynomials of the first kind, `Tₙ(1) = 1`.
    ChebyshevT,
    /// Chebyshev polynomials of the second kind, `Uₙ(1) = n + 1`.
    ChebyshevU,
    /// Legendre polynomials, `Pₙ(1) = 1`.
    Legendre,
    /// Jacobi polynomials `Pₙ^(α,β)`, with `α, β > −1`.
    Jacobi {
        /// Exponent `α` of `1 − x` in the weight.
        alpha: R,
        /// Exponent `β` of `1 + x` in the weight.
        beta: R,
    },
    /// Gegenbauer (ultraspherical) polynomials `Cₙ^(λ)`, with `λ > −1/2` and `λ ≠ 0`.
    Gegenbauer {
        /// Parameter `λ`; the weight is `(1 − x²)^(λ − 1/2)`.
        lambda: R,
    },
    /// Laguerre polynomials `Lₙ`, weight `e^(−x)` on `[0, ∞)`.
    Laguerre,
    /// Generalized Laguerre polynomials `Lₙ^(α)`, with `α > −1`.
    GeneralizedLaguerre {
        /// Exponent `α` of `x` in the weight.
        alpha: R,
    },
    /// Physicists' Hermite polynomials `Hₙ`, weight `e^(−x²)`.
    HermiteH,
    /// Probabilists' Hermite polynomials `Heₙ`, weight `e^(−x²/2)`.
    HermiteHe,
}

impl<R: RealScalar> OrthogonalFamily<R> {
    /// Evaluates the degree-`n` polynomial of this family at a real point `x`.
    ///
    /// `x` may lie outside the support of the weight. Legendre values come from
    /// [`scioxide_special::legendre_p`] and have its accuracy. The other families
    /// run the forward three-term recurrence of
    /// [`recurrence_coefficients`](Self::recurrence_coefficients) in O(n) time. When an
    /// intermediate value would overflow or underflow, the recurrence restarts
    /// with a wider exponent range, so only the final value has to fit the
    /// scalar type.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` or a family parameter is NaN or infinite.
    /// - `DomainError` if a family parameter is outside its range.
    /// - `NumericalInstability` if the value overflows the scalar type or a
    ///   recurrence coefficient is not representable.
    /// - `ScalarConversion` if `n` is not exactly representable in the scalar
    ///   type.
    /// - For Legendre, the errors of [`scioxide_special::legendre_p`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // P₂(x) = (3x² − 1)/2, so P₂(0.5) = −0.125.
    /// assert_eq!(OrthogonalFamily::Legendre.evaluate(2, 0.5_f64)?, -0.125);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline(always)]
    pub fn evaluate(self, n: usize, x: R) -> Result<R, SciOxideError> {
        self.validate()?;
        finite_input(x, "evaluation point must be finite")?;
        // Preserve the identity polynomial's bits, including signed zero.
        if n == 1 && matches!(self, Self::ChebyshevT | Self::HermiteHe) {
            return Ok(x);
        }
        self.member::<0>(n, x)
    }

    /// Evaluates the first derivative of the degree-`n` polynomial at `x`.
    ///
    /// The derivative of the three-term recurrence is run alongside it, in O(n)
    /// time for every family, including Legendre. Only the derivative has to fit
    /// the scalar type.
    ///
    /// # Errors
    ///
    /// The errors of [`evaluate`](Self::evaluate), other than those of
    /// [`scioxide_special::legendre_p`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // P₂′(x) = 3x, so P₂′(0.5) = 1.5.
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// assert!((family.evaluate_derivative(2, 0.5)? - 1.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn evaluate_derivative(self, n: usize, x: R) -> Result<R, SciOxideError> {
        self.validate()?;
        finite_input(x, "evaluation point must be finite")?;
        self.member::<1>(n, x)
    }

    // Dispatch once, so constant-family coefficients specialize outside the hot loop.
    #[inline]
    fn member<const ORDER: usize>(self, n: usize, x: R) -> Result<R, SciOxideError> {
        recurrence::member::<R, ORDER>(n, x, self)
    }

    /// Returns the `n` zeros of the degree-`n` polynomial in ascending order.
    ///
    /// Chebyshev zeros come from the closed forms `cos((2k − 1)π/(2n))` for `Tₙ`
    /// and `cos(kπ/(n + 1))` for `Uₙ`, evaluated as sines, with mirrored pairs
    /// and an exact zero at the center for odd `n`. For the other families the
    /// zeros are the eigenvalues of the symmetric tridiagonal Jacobi matrix of
    /// [`jacobi_recurrence`](Self::jacobi_recurrence) (Golub–Welsch), computed
    /// by the symmetric tridiagonal eigensolver of `scioxide-linalg` with its
    /// default options: deflation relative to the machine epsilon and ascending
    /// order. For other eigensolver options, pass the coefficients of
    /// [`jacobi_recurrence`](Self::jacobi_recurrence) to that solver.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n` is zero.
    /// - The parameter errors of [`evaluate`](Self::evaluate).
    /// - The recurrence errors of [`jacobi_recurrence`](Self::jacobi_recurrence).
    /// - The errors of the symmetric tridiagonal eigenvalue solver in
    ///   `scioxide-linalg`, such as `ConvergenceFailure`.
    /// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // The zeros of P₂(x) = (3x² − 1)/2 are ±1/√3.
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// let roots = family.zeros(2)?;
    /// assert!((roots[0] + (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules",
    /// *Math. Comp.* 23(106), 1969, 221–230.
    pub fn zeros(self, n: usize) -> Result<Vec<R>, SciOxideError> {
        self.validate()?;
        match self {
            Self::ChebyshevT => zeros::chebyshev(n, false),
            Self::ChebyshevU => zeros::chebyshev(n, true),
            _ => zeros::golub_welsch_zeros(n, |k| self.recurrence(k)),
        }
    }

    /// Evaluates the weight `w(x)` of this family (see the table on
    /// [`OrthogonalFamily`]).
    ///
    /// At a finite endpoint the weight takes its limit, which is zero for a
    /// positive exponent. Products of powers are formed as a sum of logarithms,
    /// so large parameters do not overflow intermediate values. A weight below
    /// the smallest subnormal rounds to zero.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` or a family parameter is NaN or infinite.
    /// - `DomainError` if `x` is outside the support, at an endpoint where the
    ///   weight is infinite (a negative exponent, or either endpoint for
    ///   `ChebyshevT`), or if a parameter is outside its range.
    /// - `NumericalInstability` if the weight overflows the scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // The Legendre weight is 1 on [−1, 1].
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// assert!((family.weight(0.5)? - 1.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn weight(self, x: R) -> Result<R, SciOxideError> {
        self.validate()?;
        finite_input(x, "weight point must be finite")?;
        match self {
            Self::Legendre => {
                if x < -R::ONE || x > R::ONE {
                    return Err(point_domain("point is outside [-1, 1]"));
                }
                Ok(R::ONE)
            }
            Self::ChebyshevT | Self::ChebyshevU => {
                let second = matches!(self, Self::ChebyshevU);
                if x < -R::ONE || x > R::ONE || (!second && x.abs() == R::ONE) {
                    return Err(point_domain("point is outside the finite weight domain"));
                }
                let root = ((R::ONE - x) * (R::ONE + x)).sqrt();
                finite_arithmetic(
                    if second { root } else { R::ONE / root },
                    "weight is outside the scalar range",
                )
            }
            Self::Jacobi { alpha, beta } => jacobi_weight(alpha, beta, x),
            Self::Gegenbauer { lambda } => jacobi_weight(lambda - R::HALF, lambda - R::HALF, x),
            Self::Laguerre => laguerre_weight(R::ZERO, x),
            Self::GeneralizedLaguerre { alpha } => laguerre_weight(alpha, x),
            Self::HermiteH | Self::HermiteHe => {
                let square = Scaled::from_value(x)?.mul(Scaled::from_value(x)?)?.neg();
                if matches!(self, Self::HermiteH) {
                    square.exp()
                } else {
                    square.div(Scaled::from_value(R::TWO)?)?.exp()
                }
            }
        }
    }

    /// Returns the coefficients `(aₙ, bₙ, cₙ)` of the three-term recurrence.
    ///
    /// ```text
    /// p₍ₙ₊₁₎(x) = (aₙ x + bₙ) pₙ(x) − cₙ p₍ₙ₋₁₎(x),   p₀ = 1,  p₋₁ = 0,
    /// ```
    ///
    /// where `pₙ` is the degree-`n` polynomial of this family in the standard
    /// normalization. The Jacobi coefficients are DLMF 18.9.2. Since `p₋₁ = 0`,
    /// the value of `c₀` has no effect.
    ///
    /// # Errors
    ///
    /// - The parameter errors of [`evaluate`](Self::evaluate).
    /// - `NumericalInstability` if a coefficient is not representable in the
    ///   scalar type, or a Jacobi denominator is zero.
    /// - `ScalarConversion` if `n` is not exactly representable in the scalar
    ///   type.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // Legendre: (n + 1) P₍ₙ₊₁₎ = (2n + 1) x Pₙ − n P₍ₙ₋₁₎, here with n = 1.
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// let (a, b, c) = family.recurrence_coefficients(1)?;
    /// assert!((a - 1.5).abs() < 1e-13 && b.abs() < 1e-13 && (c - 0.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn recurrence_coefficients(self, n: usize) -> Result<(R, R, R), SciOxideError> {
        self.validate()?;
        self.recurrence(n)
    }

    /// Builds the `order × order` symmetric tridiagonal Jacobi matrix of this
    /// family.
    ///
    /// The matrix represents multiplication by `x` in the orthonormal basis of
    /// the family, so it does not depend on the normalization of the
    /// polynomials. Its eigenvalues are the zeros of the degree-`order`
    /// polynomial. Construction takes O(order) time and storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `order` is zero.
    /// - The parameter errors of [`evaluate`](Self::evaluate).
    /// - `NumericalInstability` if a recurrence coefficient breaks down: a zero
    ///   leading coefficient, or a squared off-diagonal entry that is not
    ///   positive or underflows to zero.
    /// - `SizeOverflow` or `AllocationFailure` if the diagonals cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// let recurrence = family.jacobi_recurrence(3)?;
    /// assert_eq!(recurrence.diagonal().len(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn jacobi_recurrence(
        self,
        order: usize,
    ) -> Result<super::JacobiRecurrence<R>, SciOxideError> {
        self.validate()?;
        zeros::jacobi_recurrence(
            order,
            |k| self.recurrence(k),
            scioxide_core::argument!("order"),
        )
    }

    /// Returns `ln μ₀`, the logarithm of the total mass `μ₀ = ∫ w(x) dx` of
    /// the weight over its support.
    ///
    /// The masses are `π`, `π/2` and `2` for `ChebyshevT`, `ChebyshevU` and
    /// `Legendre`; `2^(α+β+1) B(α + 1, β + 1)` for Jacobi;
    /// `B(1/2, λ + 1/2)` for Gegenbauer; `Γ(α + 1)` for (generalized) Laguerre;
    /// `√π` for `HermiteH` and `√(2π)` for `HermiteHe`. The logarithm is
    /// returned because the mass overflows for large parameters.
    ///
    /// # Errors
    ///
    /// - The parameter errors of [`evaluate`](Self::evaluate).
    /// - The errors of [`scioxide_special::ln_beta`] and
    ///   [`scioxide_special::ln_gamma`], and `NumericalInstability` if the Jacobi
    ///   logarithm is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // ∫₋₁¹ 1 dx = 2.
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// assert!((family.log_weight_mass()?.exp() - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn log_weight_mass(self) -> Result<R, SciOxideError> {
        self.validate()?;
        self.log_mass()
    }

    pub(super) fn validate(self) -> Result<(), SciOxideError> {
        let exponent = |value: R, name: Argument| {
            finite_input(value, "family parameter must be finite")?;
            if value <= -R::ONE {
                Err(SciOxideError::domain_error(
                    name,
                    scioxide_core::DomainReason::OutsideDomain,
                    "weight exponent must exceed -1",
                ))
            } else {
                Ok(())
            }
        };
        match self {
            Self::Jacobi { alpha, beta } => {
                exponent(alpha, scioxide_core::argument!("alpha"))?;
                exponent(beta, scioxide_core::argument!("beta"))
            }
            Self::GeneralizedLaguerre { alpha } => {
                exponent(alpha, scioxide_core::argument!("alpha"))
            }
            Self::Gegenbauer { lambda } => {
                finite_input(lambda, "lambda must be finite")?;
                if lambda <= -R::HALF || lambda == R::ZERO {
                    Err(SciOxideError::domain_error(
                        scioxide_core::argument!("lambda"),
                        scioxide_core::DomainReason::OutsideDomain,
                        "Gegenbauer requires lambda > -1/2 and lambda != 0",
                    ))
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }

    /// Parameter validation belongs to preparation or the public entry point.
    #[inline(always)]
    pub(crate) fn recurrence(self, n: usize) -> Result<(R, R, R), SciOxideError> {
        let k = if matches!(self, Self::ChebyshevT | Self::ChebyshevU) {
            R::ZERO
        } else {
            R::try_from_usize_exact(n)?
        };
        self.recurrence_at(n, k)
    }
    /// The index has already been converted exactly, either by a checked cast
    /// or under the bounded-degree check of the range-check module.
    #[inline(always)]
    pub(super) fn recurrence_at(self, n: usize, k: R) -> Result<(R, R, R), SciOxideError> {
        match self {
            Self::ChebyshevT => return Ok((if n == 0 { R::ONE } else { R::TWO }, R::ZERO, R::ONE)),
            Self::ChebyshevU => return Ok((R::TWO, R::ZERO, R::ONE)),
            _ => {}
        }
        let denominator = k + R::ONE;
        let simple = match self {
            Self::Legendre => Some(((k + k + R::ONE) / denominator, R::ZERO, k / denominator)),
            Self::HermiteH => Some((R::TWO, R::ZERO, R::TWO * k)),
            Self::HermiteHe => Some((R::ONE, R::ZERO, k)),
            Self::Laguerre => Some((
                -R::ONE / denominator,
                (k + k + R::ONE) / denominator,
                k / denominator,
            )),
            _ => None,
        };
        if let Some(coefficients) = simple {
            return Ok(coefficients);
        }
        if let Some(coefficients) =
            candidate(self.parameterized_coefficients::<Native, true, true>(n, k))?
        {
            return Ok(coefficients);
        }
        self.extended_coefficients(n, k)
    }

    /// Factoring the common Laguerre denominator removes two divisions from
    /// every scalar recurrence step, using the same coefficient formula.
    #[inline(always)]
    pub(super) fn laguerre_numerators(self, n: usize, k: R) -> Result<(R, R, R), SciOxideError> {
        extended(self.parameterized_coefficients::<Native, false, false>(n, k))
    }
    /// Callers must hold the bounded-family range check for this degree.
    #[inline(always)]
    pub(super) fn certified_coefficients(self, n: usize, k: R) -> Result<(R, R, R), SciOxideError> {
        extended(self.parameterized_coefficients::<Native, false, true>(n, k))
    }
    #[cold]
    #[inline(never)]
    fn extended_coefficients(self, n: usize, k: R) -> Result<(R, R, R), SciOxideError> {
        extended(self.parameterized_coefficients::<Extended, true, true>(n, k))
    }

    /// DLMF 18.9.2 in either arithmetic representation. The first Jacobi
    /// step removes the general formula's 0/0 singularity explicitly.
    #[inline(always)]
    fn parameterized_coefficients<
        A: Arithmetic<R, Projected = R>,
        const CHECK: bool,
        const NORMALIZED: bool,
    >(
        self,
        n: usize,
        k: R,
    ) -> ArithmeticResult<(R, R, R)> {
        if A::NATIVE && CHECK {
            let admitted = match self {
                Self::Jacobi { alpha, beta } => four_product_values([alpha, beta, k]),
                Self::Gegenbauer { lambda } => four_product_values([lambda, k]),
                Self::GeneralizedLaguerre { alpha } => four_product_values([alpha, k]),
                _ => false,
            };
            require(admitted)?;
        }
        let s = A::scalar;
        let add = A::add;
        let sub = A::sub;
        let mul = A::mul;
        let div = A::div;
        let (zero, one, two, k) = (s(R::ZERO)?, s(R::ONE)?, s(R::TWO)?, s(k)?);
        let denominator = add(k, one)?;
        let (a, b, c) = match self {
            Self::Jacobi { alpha, beta } => {
                let (alpha, beta) = (s(alpha)?, s(beta)?);
                let sum = add(alpha, beta)?;
                if n == 0 {
                    (
                        div(add(sum, two)?, two)?,
                        div(sub(alpha, beta)?, two)?,
                        zero,
                    )
                } else {
                    let q = add(add(k, k)?, sum)?;
                    let divisor = mul(mul(mul(two, denominator)?, add(add(k, sum)?, one)?)?, q)?;
                    if divisor == zero {
                        return Err(if A::NATIVE {
                            ArithmeticError::Retry
                        } else {
                            SciOxideError::numerical_instability(
                                scioxide_core::NumericalIssue::Breakdown,
                                "Jacobi recurrence denominator is zero",
                            )
                            .into()
                        });
                    }
                    (
                        div(mul(mul(add(q, one)?, add(q, two)?)?, q)?, divisor)?,
                        div(mul(mul(add(q, one)?, sub(alpha, beta)?)?, sum)?, divisor)?,
                        div(
                            mul(mul(mul(two, add(k, alpha)?)?, add(k, beta)?)?, add(q, two)?)?,
                            divisor,
                        )?,
                    )
                }
            }
            Self::Gegenbauer { lambda } => gegenbauer_coefficients::<R, A>(s(lambda)?, k, n == 0)?,
            Self::GeneralizedLaguerre { alpha } => {
                let alpha = s(alpha)?;
                (
                    if NORMALIZED {
                        div(sub(zero, one)?, denominator)?
                    } else {
                        sub(zero, one)?
                    },
                    if NORMALIZED {
                        div(add(add(add(k, k)?, alpha)?, one)?, denominator)?
                    } else {
                        add(add(add(k, k)?, alpha)?, one)?
                    },
                    if NORMALIZED {
                        div(add(k, alpha)?, denominator)?
                    } else {
                        add(k, alpha)?
                    },
                )
            }
            _ => unreachable!("unparameterized coefficient formulas returned first"),
        };
        // Admit the complete coefficient triple together, retaining the
        // branchless component checks of the native provider.
        if CHECK {
            require(A::safe(a) & A::safe(b) & A::safe(c))?;
        }
        Ok((A::project(a)?, A::project(b)?, A::project(c)?))
    }
    pub(super) fn log_mass(self) -> Result<R, SciOxideError> {
        let one = R::ONE;
        let two = R::TWO;
        let half = one / two;
        match self {
            // Logarithm of the total weight mass `h_0 = integral of w`; kept in log form
            // (via `ln_beta`/`ln_gamma`) because the masses overflow for large parameters.
            Self::ChebyshevT => Ok(R::PI.ln()),
            Self::ChebyshevU => Ok((R::PI / two).ln()),
            Self::Legendre => Ok(two.ln()),
            Self::Jacobi { alpha, beta } => finite_arithmetic(
                (alpha + beta + one) * two.ln()
                    + scioxide_special::ln_beta(alpha + one, beta + one)?,
                "logarithmic weight mass is outside the scalar range",
            ),
            Self::Gegenbauer { lambda } => scioxide_special::ln_beta(half, lambda + half),
            Self::Laguerre => Ok(R::ZERO),
            Self::GeneralizedLaguerre { alpha } => scioxide_special::ln_gamma(alpha + one),
            Self::HermiteH => Ok(half * R::PI.ln()),
            Self::HermiteHe => Ok(half * (two * R::PI).ln()),
        }
    }
}

/// Pure Gegenbauer coefficient update. Callers either check its outputs or
/// hold the bounded-family range check for the entire degree.
#[inline(always)]
pub(super) fn gegenbauer_coefficients<R: RealScalar, A: Arithmetic<R>>(
    lambda: A::Value,
    k: A::Value,
    first: bool,
) -> ArithmeticResult<(A::Value, A::Value, A::Value)> {
    let one = A::scalar(R::ONE)?;
    let two = A::scalar(R::TWO)?;
    let zero = A::scalar(R::ZERO)?;
    let denominator = A::add(k, one)?;
    let a = A::div(A::mul(two, A::add(k, lambda)?)?, denominator)?;
    // Subtract the integer first so the c_1 coefficient retains tiny lambda.
    let c = if first {
        zero
    } else {
        A::div(A::add(A::sub(k, one)?, A::mul(two, lambda)?)?, denominator)?
    };
    Ok((a, zero, c))
}

fn point_domain(detail: &'static str) -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("x"),
        scioxide_core::DomainReason::OutsideDomain,
        detail,
    )
}
fn jacobi_weight<R: RealScalar>(alpha: R, beta: R, x: R) -> Result<R, SciOxideError> {
    if x < -R::ONE || x > R::ONE {
        return Err(point_domain("point is outside [-1, 1]"));
    }
    if x.abs() == R::ONE {
        let (exponent, other) = if x == R::ONE {
            (alpha, beta)
        } else {
            (beta, alpha)
        };
        if exponent < R::ZERO {
            return Err(point_domain("weight diverges at this endpoint"));
        }
        if exponent > R::ZERO {
            return Ok(R::ZERO);
        }
        return Scaled::from_value(other)?
            .mul(Scaled::from_value(R::TWO.ln())?)?
            .exp();
    }
    let mut logarithm = ScaledAccumulator::ZERO;
    logarithm.add_product(
        Scaled::from_value(alpha)?,
        Scaled::from_value((-x).ln_1p())?,
    )?;
    logarithm.add_product(Scaled::from_value(beta)?, Scaled::from_value(x.ln_1p())?)?;
    logarithm.total()?.exp()
}
fn laguerre_weight<R: RealScalar>(alpha: R, x: R) -> Result<R, SciOxideError> {
    if x < R::ZERO {
        return Err(point_domain("point is outside [0, infinity)"));
    }
    if x == R::ZERO {
        return if alpha < R::ZERO {
            Err(point_domain("weight diverges at zero for negative alpha"))
        } else {
            Ok(if alpha == R::ZERO { R::ONE } else { R::ZERO })
        };
    }
    if alpha == R::ZERO {
        return Ok((-x).exp());
    }
    let mut logarithm = ScaledAccumulator::ZERO;
    logarithm.add_product(Scaled::from_value(alpha)?, Scaled::from_value(x.ln())?)?;
    logarithm.sub_scaled(Scaled::from_value(x)?)?;
    logarithm.total()?.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint_limits<R: RealScalar>() {
        let half = R::HALF;
        for point in [-R::ONE, R::ONE] {
            assert_eq!(OrthogonalFamily::ChebyshevU.weight(point).unwrap(), R::ZERO);
            assert_eq!(OrthogonalFamily::Legendre.weight(point).unwrap(), R::ONE);
            assert_eq!(
                OrthogonalFamily::Jacobi {
                    alpha: R::ZERO,
                    beta: R::ZERO
                }
                .weight(point)
                .unwrap(),
                R::ONE
            );
            assert_eq!(
                OrthogonalFamily::Jacobi {
                    alpha: half,
                    beta: half
                }
                .weight(point)
                .unwrap(),
                R::ZERO
            );
            assert_eq!(
                OrthogonalFamily::Gegenbauer { lambda: half }
                    .weight(point)
                    .unwrap(),
                R::ONE
            );
            for family in [
                OrthogonalFamily::ChebyshevT,
                OrthogonalFamily::Jacobi {
                    alpha: -half,
                    beta: -half,
                },
                OrthogonalFamily::Gegenbauer {
                    lambda: -half / R::TWO,
                },
            ] {
                assert!(matches!(
                    family.weight(point).unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::DomainError { .. }
                ));
            }
        }
        assert_eq!(
            OrthogonalFamily::Jacobi {
                alpha: R::ZERO,
                beta: R::ONE
            }
            .weight(R::ONE)
            .unwrap(),
            R::TWO
        );
        assert_eq!(
            OrthogonalFamily::Jacobi {
                alpha: R::ONE,
                beta: R::ZERO
            }
            .weight(-R::ONE)
            .unwrap(),
            R::TWO
        );
        assert_eq!(
            OrthogonalFamily::GeneralizedLaguerre { alpha: half }
                .weight(R::ZERO)
                .unwrap(),
            R::ZERO
        );
        assert_eq!(
            OrthogonalFamily::GeneralizedLaguerre { alpha: R::ZERO }
                .weight(R::ZERO)
                .unwrap(),
            R::ONE
        );
        assert!(matches!(
            OrthogonalFamily::GeneralizedLaguerre { alpha: -half }
                .weight(R::ZERO)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    #[test]
    fn finite_and_singular_weight_endpoints_in_both_widths() {
        endpoint_limits::<f32>();
        endpoint_limits::<f64>();
    }

    #[test]
    fn every_operation_validates_family_parameters() {
        for family in [
            OrthogonalFamily::Jacobi {
                alpha: f64::NAN,
                beta: 0.0,
            },
            OrthogonalFamily::Jacobi {
                alpha: 0.0,
                beta: -1.0,
            },
            OrthogonalFamily::Gegenbauer { lambda: 0.0 },
            OrthogonalFamily::GeneralizedLaguerre { alpha: -1.0 },
        ] {
            assert!(family.evaluate(0, 0.0).is_err());
            assert!(family.evaluate_derivative(0, 0.0).is_err());
            assert!(family.weight(0.0).is_err());
            assert!(family.zeros(1).is_err());
            assert!(family.recurrence_coefficients(0).is_err());
            assert!(family.jacobi_recurrence(1).is_err());
            assert!(family.log_weight_mass().is_err());
        }
    }

    #[test]
    fn bounded_gegenbauer_parameters_match_checked_coefficients_at_both_edges() {
        fn check<R: RealScalar>() {
            for parameter in [1.0 / 65536.0, 1.5, 16384.0] {
                let family = OrthogonalFamily::Gegenbauer {
                    lambda: R::try_from_f64_exact(parameter).unwrap(),
                };
                for degree in [0, 1, 2, 8] {
                    let point = R::try_from_f64_exact(0.25).unwrap();
                    let expected =
                        recurrence::checked_member::<R, 0>(degree, point, family).unwrap();
                    let value = family.evaluate(degree, point).unwrap();
                    assert!(
                        (value - expected).abs()
                            <= R::try_from_f64_exact(256.0).unwrap()
                                * R::EPSILON
                                * (R::ONE + expected.abs())
                    );
                    let expected =
                        recurrence::checked_member::<R, 1>(degree, point, family).unwrap();
                    let derivative = family.evaluate_derivative(degree, point).unwrap();
                    assert!(
                        (derivative - expected).abs()
                            <= R::try_from_f64_exact(256.0).unwrap()
                                * R::EPSILON
                                * (R::ONE + expected.abs())
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
