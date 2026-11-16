//! Approximate greatest common divisor of two polynomials by rank analysis of
//! the Sylvester matrix.
use crate::PolynomialBasis;
use crate::util::{coefficient_exponent, round_complex, scaled_complex};
use scioxide_core::backend::try_copy_slice;

use scioxide_core::AlgebraicScalar;

use scioxide_core::Scaled;
use scioxide_core::backend::scaled::scaled_magnitude_le_tolerance;
use scioxide_core::backend::try_filled;
use scioxide_core::{Argument, InvalidInputReason, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, RankReport, SvdValues};

use crate::{ApproxPolynomial, MonomialBasis, PolynomialApproximationPolicy};

mod recovery;
use recovery::recover_gcd;

#[cfg(test)]
std::thread_local! {
    static DIVISION_CALLS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

type MonomialPolynomial<C> = ApproxPolynomial<C, MonomialBasis<C>>;
type PolynomialDivision<C> = (MonomialPolynomial<C>, MonomialPolynomial<C>);

/// Holds the diagnostics of [`approximate_gcd`].
///
/// ```
/// use scioxide_poly::*;
/// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let result = approximate_gcd(&a, &b)?;
/// let report = result.report();
/// assert_eq!(report.effective_degree(), 1);
/// assert!(report.backward_error() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ApproximateGcdReport<R: RealScalar> {
    sylvester_rank: Option<RankReport<R>>,
    reciprocal_condition: Option<R>,
    operand_errors: [R; 2],
    effective_degree: usize,
    equilibration_exponents: [i64; 2],
    policy: PolynomialApproximationPolicy<R>,
}

impl<R: RealScalar> ApproximateGcdReport<R> {
    /// Returns the numerical rank of the scaled Sylvester matrix.
    ///
    /// It is `None` when an operand is zero, since no matrix is formed.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// let report = result.report();
    /// assert_eq!(report.sylvester_rank().map(|rank| rank.rank()), Some(2));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn sylvester_rank(self) -> Option<RankReport<R>> {
        self.sylvester_rank
    }

    /// Returns the reciprocal condition number `σ_min/σ_max` of the scaled
    /// Sylvester matrix.
    ///
    /// It does not depend on the rank policy. A value near zero means the
    /// operands are close to having a common factor of higher degree. It is
    /// `None` when an operand is zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// let report = result.report();
    /// let rcond = report.reciprocal_condition();
    /// assert!(rcond.is_some_and(|value| value >= 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn reciprocal_condition(self) -> Option<R> {
        self.reciprocal_condition
    }

    /// Returns the larger of the two [`operand_errors`](Self::operand_errors).
    ///
    /// It measures how far the operands are from being exact multiples of the
    /// computed GCD. It is an estimate, not a bound.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// let report = result.report();
    /// assert!(report.backward_error() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn backward_error(self) -> R {
        self.operand_errors[0].max(self.operand_errors[1])
    }

    /// Returns the relative remainders `‖rᵢ‖∞ / ‖fᵢ‖∞` of dividing each operand
    /// `fᵢ` by the GCD, in the order `[left, right]`.
    ///
    /// `‖·‖∞` is the largest coefficient modulus. A zero operand has error zero.
    /// The same values decide cancellation in
    /// [`ApproxRationalFunction::simplify`](crate::ApproxRationalFunction::simplify).
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let report = approximate_gcd(&a, &a)?.report();
    /// assert_eq!(report.operand_errors(), [0.0, 0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn operand_errors(self) -> [R; 2] {
        self.operand_errors
    }

    /// Returns the degree of the computed GCD.
    ///
    /// For nonzero operands of degrees `m` and `n` it is `m + n − rank`, capped
    /// at `min(m, n)`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// let report = result.report();
    /// assert_eq!(report.effective_degree(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn effective_degree(self) -> usize {
        self.effective_degree
    }

    /// Returns the binary exponents `[e₁, e₂]` by which the operands were scaled.
    ///
    /// Operand `fᵢ` is multiplied by `2^(−eᵢ)` so that its largest coefficient
    /// modulus lies in `[1, 2)`. The result is `[0, 0]` when an operand is zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// let report = result.report();
    /// assert_eq!(report.equilibration_exponents(), [1, 0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn equilibration_exponents(self) -> [i64; 2] {
        self.equilibration_exponents
    }

    /// Returns the approximation policy used for the rank decision and the
    /// remainder tolerance.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let policy = PolynomialApproximationPolicy::<f64>::default();
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// let report = result.report();
    /// assert_eq!(report.policy(), policy);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn policy(self) -> PolynomialApproximationPolicy<R> {
        self.policy
    }
}

/// Holds the result of [`approximate_gcd`]: the GCD, the cofactors, the
/// remainders and a report.
///
/// ```
/// use scioxide_poly::*;
/// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let result = approximate_gcd(&a, &b)?;
/// assert_eq!(result.report().effective_degree(), 1);
/// assert!(result.report().backward_error() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ApproximateGcd<C: NumericScalar> {
    polynomial: ApproxPolynomial<C, MonomialBasis<C>>,
    divisions: [PolynomialDivision<C>; 2],
    accepted: [bool; 2],
    report: ApproximateGcdReport<C::Real>,
}

impl<C: NumericScalar> ApproximateGcd<C> {
    /// Returns the monic GCD, or the zero polynomial if both operands are zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// assert_eq!(result.polynomial().degree(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn polynomial(&self) -> &ApproxPolynomial<C, MonomialBasis<C>> {
        &self.polynomial
    }

    /// Returns the quotients `[left / g, right / g]` of dividing each operand by
    /// the GCD `g`.
    ///
    /// For two zero operands both cofactors are zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &a)?;
    /// assert_eq!(result.cofactors()[0].basis().coefficients(), &[1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn cofactors(&self) -> [&MonomialPolynomial<C>; 2] {
        [&self.divisions[0].0, &self.divisions[1].0]
    }

    /// Returns the remainders of dividing each operand by the GCD, in the order
    /// `[left, right]`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &a)?;
    /// assert!(result.remainders().iter().all(|r| r.is_zero()));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn remainders(&self) -> [&MonomialPolynomial<C>; 2] {
        [&self.divisions[0].1, &self.divisions[1].1]
    }

    pub(crate) fn into_cancellation(
        self,
    ) -> (
        [MonomialPolynomial<C>; 2],
        [bool; 2],
        ApproximateGcdReport<C::Real>,
    ) {
        let [(left, _), (right, _)] = self.divisions;
        ([left, right], self.accepted, self.report)
    }

    /// Returns the diagnostics report.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let result = approximate_gcd(&a, &b)?;
    /// assert_eq!(result.report().effective_degree(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn report(&self) -> ApproximateGcdReport<C::Real> {
        self.report
    }
}

/// Computes an approximate greatest common divisor of two monomial-basis
/// polynomials.
///
/// The degree `k` of the GCD is the nullity of the Sylvester matrix of `left`
/// (degree `m`) and `right` (degree `n`). Each operand is first scaled by a
/// power of two, so its largest coefficient modulus lies in `[1, 2)`. The
/// singular values of the `(m + n) × (m + n)` scaled Sylvester matrix and the
/// rank policy of the operands' [`PolynomialApproximationPolicy`] give the rank
/// `r`, and `k = min(m + n − r, m, n)`. The right singular vector of the
/// subresultant matrix for degree `k` gives starting cofactors. A
/// Gauss–Newton least-squares refinement of the GCD and both cofactors then
/// runs for at most 32 iterations, with at most 16 step halvings each. The
/// degree is not changed during refinement. The GCD is normalized to be monic.
///
/// The cost is dominated by the SVD of the Sylvester matrix, O((m + n)³) time
/// and O((m + n)²) memory. If one operand is zero, the result is the monic
/// form of the other in O(m + n) time, and `gcd(0, 0) = 0`. A rank policy that
/// is too loose can select a degree that is too high; check
/// [`ApproximateGcdReport::backward_error`] and
/// [`ApproximateGcdReport::reciprocal_condition`].
///
/// # Errors
///
/// - `InvalidInput` if the operands have different approximation policies.
/// - `NumericalInstability` if the recovered GCD has a zero leading
///   coefficient, or a scaled value overflows.
/// - `ConvergenceFailure` if an SVD does not converge.
/// - `AllocationFailure` or `SizeOverflow` if the matrices cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
///
/// // gcd((x + 1)², x + 1) = x + 1.
/// let a = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let b = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let result = approximate_gcd(&a, &b)?;
/// assert_eq!(result.report().effective_degree(), 1);
/// assert!(result.report().backward_error() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Z. Zeng, "The numerical greatest common divisor of univariate polynomials",
/// in *Randomization, Relaxation, and Complexity in Polynomial Equation
/// Solving*, Contemp. Math. 556, AMS, 2011, 187–217.
pub fn approximate_gcd<C: NumericScalar>(
    left: &ApproxPolynomial<C, MonomialBasis<C>>,
    right: &ApproxPolynomial<C, MonomialBasis<C>>,
) -> Result<ApproximateGcd<C>, SciOxideError> {
    crate::util::same_policy(
        left.policy(),
        right.policy(),
        scioxide_core::argument!("right.policy"),
    )?;
    let policy = left.policy();
    if left.is_zero() || right.is_zero() {
        let polynomial = if left.is_zero() {
            right.monic()?
        } else {
            left.monic()?
        };
        return finish_with_factor(left, right, polynomial, None, None, [0, 0], policy);
    }

    // The GCD degree is the nullity of the Sylvester matrix. Estimating it from
    // singular values (rather than from when Euclidean remainders "look small")
    // lets the immutable rank policy make the numerical decision in one place.
    let (sylvester, exponents) =
        equilibrated_sylvester(left.basis().coefficients(), right.basis().coefficients())?;
    let spectrum = scioxide_linalg::SvdValues::compute(&sylvester)?;
    finish_approximate_gcd(left, right, spectrum, exponents)
}

/// Recovers the GCD of the degree selected by the rank policy.
fn finish_approximate_gcd<C: NumericScalar>(
    left: &ApproxPolynomial<C, MonomialBasis<C>>,
    right: &ApproxPolynomial<C, MonomialBasis<C>>,
    spectrum: SvdValues<C::Real>,
    exponents: [i64; 2],
) -> Result<ApproximateGcd<C>, SciOxideError> {
    let policy = left.policy();
    let rank = spectrum.analyze_rank(policy.rank_policy())?;
    let matrix_order = left.degree() + right.degree();
    // Nullity = `order - rank`, clamped because a GCD cannot exceed either degree
    // even if the rank policy truncates more aggressively.
    let effective_degree = matrix_order
        .saturating_sub(rank.rank())
        .min(left.degree().min(right.degree()));
    let reciprocal_condition = spectrum.condition_estimate()?.reciprocal().value();
    let polynomial = recover_gcd(left, right, effective_degree)?;
    finish_with_factor(
        left,
        right,
        polynomial,
        Some(rank),
        Some(reciprocal_condition),
        exponents,
        policy,
    )
}

fn finish_with_factor<C: NumericScalar>(
    left: &MonomialPolynomial<C>,
    right: &MonomialPolynomial<C>,
    polynomial: MonomialPolynomial<C>,
    sylvester_rank: Option<RankReport<C::Real>>,
    reciprocal_condition: Option<C::Real>,
    equilibration_exponents: [i64; 2],
    policy: PolynomialApproximationPolicy<C::Real>,
) -> Result<ApproximateGcd<C>, SciOxideError> {
    let divisions = if polynomial.is_zero() {
        let zero = || {
            ApproxPolynomial::monomial_with_options(
                try_filled(1, C::ZERO, scioxide_core::quantity!("zero_gcd_component"))?,
                policy,
            )
        };
        [(zero()?, zero()?), (zero()?, zero()?)]
    } else {
        [
            approximate_division(left, &polynomial, scioxide_core::argument!("divisor"))?,
            approximate_division(right, &polynomial, scioxide_core::argument!("divisor"))?,
        ]
    };
    let (left_error, left_ok) = cancellation_error(left, &divisions[0].1)?;
    let (right_error, right_ok) = cancellation_error(right, &divisions[1].1)?;
    let effective_degree = polynomial.degree();
    Ok(ApproximateGcd {
        polynomial,
        divisions,
        accepted: [left_ok, right_ok],
        report: ApproximateGcdReport {
            sylvester_rank,
            reciprocal_condition,
            operand_errors: [left_error, right_error],
            effective_degree,
            equilibration_exponents,
            policy,
        },
    })
}

/// Builds the Sylvester matrix with each operand scaled by a power of two.
fn equilibrated_sylvester<C: NumericScalar>(
    left: &[C],
    right: &[C],
) -> Result<(Matrix<C>, [i64; 2]), SciOxideError> {
    let left_degree = left.len() - 1;
    let right_degree = right.len() - 1;
    let order = scioxide_core::backend::checked_add(
        left_degree,
        right_degree,
        scioxide_core::quantity!("sylvester_matrix_order"),
    )?;
    if order == 0 {
        return Ok((Matrix::identity(1)?, [0, 0]));
    }
    // Scale each operand's rows by a power of two so its largest coefficient lies
    // in [1, 2). Power-of-two scaling is exact (barring underflow, which
    // `fill_minor` rejects), and it keeps the singular values from being dominated
    // by whichever operand happens to have larger coefficients.
    let left_exponent = coefficient_exponent(left)?;
    let right_exponent = coefficient_exponent(right)?;
    let mut matrix = Matrix::zeros(order, order)?;
    crate::elimination::fill_minor(
        left,
        right,
        0,
        0,
        [left_exponent, right_exponent],
        &mut matrix,
    )?;
    Ok((matrix, [left_exponent, right_exponent]))
}

/// Divides two monomial-basis polynomials by long division.
///
/// A zero divisor is reported against `divisor_argument`.
pub(crate) fn approximate_division<C: NumericScalar>(
    dividend: &MonomialPolynomial<C>,
    divisor: &MonomialPolynomial<C>,
    divisor_argument: Argument,
) -> Result<PolynomialDivision<C>, SciOxideError> {
    #[cfg(test)]
    DIVISION_CALLS.with(|calls| calls.set(calls.get() + 1));
    if divisor.is_zero() {
        return Err(SciOxideError::invalid_input(
            divisor_argument,
            InvalidInputReason::OutOfRange,
            "the divisor must be nonzero",
        ));
    }
    let remainder_quantity = scioxide_core::quantity!("polynomial_division_remainder");
    if dividend.degree() < divisor.degree() {
        return Ok((
            ApproxPolynomial::monomial_with_options(
                try_filled(
                    1,
                    C::ZERO,
                    scioxide_core::quantity!("zero_polynomial_quotient"),
                )?,
                dividend.policy(),
            )?,
            ApproxPolynomial::monomial_with_options(
                try_copy_slice(dividend.basis().coefficients(), remainder_quantity)?,
                dividend.policy(),
            )?,
        ));
    }
    let mut remainder = try_copy_slice(dividend.basis().coefficients(), remainder_quantity)?;
    let mut quotient = try_filled(
        dividend.degree() - divisor.degree() + 1,
        C::ZERO,
        scioxide_core::quantity!("polynomial_division_quotient"),
    )?;
    let mut extended = try_filled(
        crate::division::scratch_len(
            dividend.basis().coefficients().len(),
            divisor.basis().coefficients().len(),
        )?,
        scioxide_core::ScaledValue::ZERO,
        scioxide_core::quantity!("scaled_division_scratch"),
    )?;
    division_into(
        dividend.basis().coefficients(),
        divisor.basis().coefficients(),
        &mut quotient,
        &mut remainder,
        &mut extended,
        || Ok(()),
    )?;
    Ok((
        ApproxPolynomial::monomial_with_options(quotient, dividend.policy())?,
        ApproxPolynomial::monomial_with_options(remainder, dividend.policy())?,
    ))
}

use crate::division::division_into;

/// Returns the relative remainder of one operand and whether it is within the
/// policy tolerance.
fn cancellation_error<C: NumericScalar>(
    original: &MonomialPolynomial<C>,
    remainder: &MonomialPolynomial<C>,
) -> Result<(C::Real, bool), SciOxideError> {
    let norm = |p: &MonomialPolynomial<C>| {
        p.basis()
            .coefficients()
            .iter()
            .try_fold(Scaled::ZERO, |maximum, &value| {
                let magnitude = scaled_complex(value)?.magnitude()?;
                Ok::<_, SciOxideError>(if maximum.abs_le(magnitude) {
                    magnitude
                } else {
                    maximum
                })
            })
    };
    let residual = norm(remainder)?;
    let scale = norm(original)?;
    let accepted = scaled_magnitude_le_tolerance(residual, &original.policy().tolerance(), scale)?;
    let relative = if residual.is_zero() {
        C::Real::ZERO
    } else {
        residual.div(scale)?.round()?
    };
    Ok((relative, accepted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{Complex, Tolerance};
    use scioxide_linalg::RankPolicy;

    #[test]
    fn simplification_divides_each_operand_once() {
        let policy = PolynomialApproximationPolicy::<f64>::default();
        let left = ApproxPolynomial::monomial_with_options(vec![-1., 0., 1.], policy).unwrap();
        let right = ApproxPolynomial::monomial_with_options(vec![-2., 1., 1.], policy).unwrap();
        let rational = crate::ApproxRationalFunction::new(left, right).unwrap();
        DIVISION_CALLS.with(|calls| calls.set(0));
        let result = rational.simplify().unwrap();
        DIVISION_CALLS.with(|calls| assert_eq!(calls.get(), 2));
        assert_eq!(
            result.gcd_report().operand_errors(),
            [result.numerator_error(), result.denominator_error()]
        );
    }

    fn policy(relative: f64) -> PolynomialApproximationPolicy<f64> {
        PolynomialApproximationPolicy::new(
            Tolerance::new(0.0, relative).unwrap(),
            RankPolicy::Threshold(scioxide_core::RankThreshold::new(0.0, relative).unwrap()),
        )
    }

    fn policy_f32(relative: f32) -> PolynomialApproximationPolicy<f32> {
        PolynomialApproximationPolicy::new(
            Tolerance::new(0.0, relative).unwrap(),
            RankPolicy::Threshold(scioxide_core::RankThreshold::new(0.0, relative).unwrap()),
        )
    }

    #[test]
    fn sylvester_rank_finds_scaled_and_noisy_common_factors() {
        let shared =
            ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy(1.0e-8)).unwrap();
        let left = shared
            .try_mul(
                &ApproxPolynomial::monomial_with_options(vec![2.0, 1.0], policy(1.0e-8)).unwrap(),
            )
            .unwrap();
        let mut right_coefficients = shared
            .try_mul(
                &ApproxPolynomial::monomial_with_options(vec![-3.0, 1.0], policy(1.0e-8)).unwrap(),
            )
            .unwrap()
            .basis()
            .coefficients()
            .to_vec();
        right_coefficients[0] += 1.0e-11;
        let right =
            ApproxPolynomial::monomial_with_options(right_coefficients, policy(1.0e-8)).unwrap();
        let result = approximate_gcd(&left, &right).unwrap();
        assert_eq!(result.report().effective_degree(), 1);
        assert_eq!(result.polynomial().degree(), 1);
        assert!(result.report().backward_error() <= 1.0e-8);

        let scaled_left = ApproxPolynomial::monomial_with_options(
            left.basis()
                .coefficients()
                .iter()
                .map(|value| value * 2.0_f64.powi(400))
                .collect(),
            policy(1.0e-8),
        )
        .unwrap();
        let scaled_right = ApproxPolynomial::monomial_with_options(
            right
                .basis()
                .coefficients()
                .iter()
                .map(|value| value * 2.0_f64.powi(-400))
                .collect(),
            policy(1.0e-8),
        )
        .unwrap();
        let scaled = approximate_gcd(&scaled_left, &scaled_right).unwrap();
        assert_eq!(scaled.report().effective_degree(), 1);
        assert_eq!(scaled.polynomial().degree(), 1);
        assert_ne!(scaled.report().equilibration_exponents(), [0, 0]);
    }

    #[test]
    fn rank_policy_separates_near_common_roots() {
        let loose = policy(1.0e-6);
        let strict = policy(1.0e-14);
        let make = |policy, displacement| {
            (
                ApproxPolynomial::from_roots_with_options(&[1.0_f64, -2.0], policy).unwrap(),
                ApproxPolynomial::from_roots_with_options(&[1.0 + displacement, 3.0], policy)
                    .unwrap(),
            )
        };
        let (left, right) = make(loose, 1.0e-9);
        assert_eq!(
            approximate_gcd(&left, &right)
                .unwrap()
                .report()
                .effective_degree(),
            1
        );
        let (left, right) = make(strict, 1.0e-9);
        assert_eq!(
            approximate_gcd(&left, &right)
                .unwrap()
                .report()
                .effective_degree(),
            0
        );
    }

    #[test]
    fn multiple_repeated_common_roots_are_recovered() {
        let policy = policy(1.0e-9);
        let repeated = ApproxPolynomial::from_roots_with_options(&[1.0_f64, 1.0], policy).unwrap();
        let left = repeated
            .try_mul(&ApproxPolynomial::from_roots_with_options(&[-2.0_f64, 4.0], policy).unwrap())
            .unwrap();
        let right = repeated
            .try_mul(&ApproxPolynomial::from_roots_with_options(&[-3.0_f64, 5.0], policy).unwrap())
            .unwrap();

        let result = approximate_gcd(&left, &right).unwrap();
        assert_eq!(result.report().effective_degree(), 2);
        assert_eq!(result.polynomial().degree(), 2);
        assert!(result.report().backward_error() <= 1.0e-9);
        assert!(result.polynomial().evaluate(1.0).unwrap().abs() <= 1.0e-12);
    }

    #[test]
    fn complex_common_factor_uses_the_same_rank_pipeline() {
        let policy = policy(1.0e-10);
        let common = ApproxPolynomial::monomial_with_options(
            vec![Complex::new(-1.0, -2.0), Complex::new(1.0, 0.0)],
            policy,
        )
        .unwrap();
        let left = common
            .try_mul(
                &ApproxPolynomial::monomial_with_options(
                    vec![Complex::new(2.0, 0.0), Complex::new(1.0, 0.0)],
                    policy,
                )
                .unwrap(),
            )
            .unwrap();
        let right = common
            .try_mul(
                &ApproxPolynomial::monomial_with_options(
                    vec![Complex::new(-3.0, 0.0), Complex::new(1.0, 0.0)],
                    policy,
                )
                .unwrap(),
            )
            .unwrap();
        let result = approximate_gcd(&left, &right).unwrap();
        assert_eq!(result.report().effective_degree(), 1);
        assert_eq!(result.polynomial().degree(), 1);
    }

    #[test]
    fn f32_common_factor_uses_equilibrated_rank_analysis() {
        let policy = policy_f32(2.0e-4);
        let common = ApproxPolynomial::monomial_with_options(vec![-0.5_f32, 1.0], policy).unwrap();
        let left = common
            .try_mul(&ApproxPolynomial::monomial_with_options(vec![2.0_f32, 1.0], policy).unwrap())
            .unwrap();
        let right = common
            .try_mul(&ApproxPolynomial::monomial_with_options(vec![-3.0_f32, 1.0], policy).unwrap())
            .unwrap();
        let result = approximate_gcd(&left, &right).unwrap();
        assert_eq!(result.report().effective_degree(), 1);
        assert!(result.report().backward_error() <= 2.0e-4);
        assert_eq!(result.report().policy(), policy);
    }
}
