//! Padé approximation from Taylor coefficients.

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, Interval, NumericScalar, ProblemToken, RealScalar,
    SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, Matrix, ThinSvdDecomposition, Vector,
};
use scioxide_poly::{ApproxPolynomial, ApproxRationalFunction, PolynomialApproximationPolicy};

use super::{RationalDomain, RationalReport, RationalTermination, rational_interpolant};
use crate::checked::intermediate;
use crate::error::instability;
use crate::error::shape;
use crate::protocol::ExtrapolationPolicy;
use crate::validate::validate_finite;

/// Computes the `[m/n]` Padé approximant of a Taylor series, over a real or complex scalar `C`.
///
/// Given the Taylor coefficients `c₀, c₁, …` of `f` at `z₀ = center`, the
/// approximant is `r(z) = p(u) / q(u)` with `u = z − z₀`, `deg p ≤ m`,
/// `deg q ≤ n`, `q(0) = 1` and
///
/// ```text
/// q(u) f(u) − p(u) = O(u^(m+n+1)).
/// ```
///
/// The denominator coefficients solve an `n × n` Toeplitz system, by a
/// rank-truncated SVD so that singular (degenerate) blocks are accepted when
/// the system is consistent; [`Self::denominator_rank`] reports the rank, and
/// common factors of `p` and `q` are kept. The numerator then follows from the
/// first `m + 1` conditions. Construction costs O(n³ + m n) operations and a
/// query O(m + n). Evaluation goes through [`crate::Interpolant`] in the forms
/// of [`crate::BarycentricRational`].
///
/// [`Self::new`] and [`Self::with_options`] run on the calling thread.
/// [`Self::build`] runs in a solve scope and counts one iteration; see
/// [`crate::RationalTermination`].
///
/// # Errors
///
/// - The query errors described in the [crate docs](crate#errors).
/// - The errors of [`ApproxRationalFunction`] evaluation at or near a pole.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_interp::{PadeApproximant, RationalDomain, Univariate};
///
/// let domain = RationalDomain::Real(Interval::new(-1.0_f64, 1.0)?);
/// // The [2/2] Padé approximant of exp(z) at 0.
/// let taylor = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
/// let exp = PadeApproximant::new(&taylor, 2, 2, 0.0, domain)?;
/// let z = 0.5_f64;
/// let expected = (1.0 + z / 2.0 + z * z / 12.0) / (1.0 - z / 2.0 + z * z / 12.0);
/// assert!((exp.value_at(z)? - expected).abs() < 1e-13);
/// assert_eq!(exp.denominator_rank(), 2);
/// assert_eq!(exp.fit_report().iterations(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. A. Baker, Jr. and P. Graves-Morris, *Padé Approximants*, 2nd ed.,
/// Cambridge University Press, 1996, chapter 1.
#[derive(Debug, Clone)]
#[must_use]
pub struct PadeApproximant<C: NumericScalar> {
    rational: ApproxRationalFunction<C>,
    center: C,
    domain: RationalDomain<C::Real>,
    plane: [Interval<C::Real>; 2],
    coefficient_residual: C::Real,
    denominator_rank: usize,
    report: RationalReport<C::Real>,
}

impl<C: NumericScalar> PadeApproximant<C> {
    /// Computes the `[m/n]` approximant on the calling thread, with `m = numerator_degree` and `n = denominator_degree`.
    ///
    /// `coefficients` holds at least `m + n + 1` Taylor coefficients in
    /// ascending order; extra coefficients are ignored. `center` must lie in
    /// `domain`. This is [`Self::with_options`] with
    /// [`PolynomialApproximationPolicy::default`]: rank decisions by
    /// [`RankPolicy::DimensionScaledEpsilon`](scioxide_linalg::RankPolicy::DimensionScaledEpsilon)
    /// and the tolerance `atol = rtol = √ε` for the residual of the Padé
    /// conditions.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn new(
        coefficients: &[C],
        numerator_degree: usize,
        denominator_degree: usize,
        center: C,
        domain: RationalDomain<C::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(
            coefficients,
            numerator_degree,
            denominator_degree,
            center,
            domain,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Computes the `[m/n]` approximant on the calling thread with the rank
    /// decision and residual tolerance of `policy`.
    ///
    /// The arguments are those of [`Self::new`]; this is [`Self::build`] with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::build`].
    pub fn with_options(
        coefficients: &[C],
        numerator_degree: usize,
        denominator_degree: usize,
        center: C,
        domain: RationalDomain<C::Real>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            coefficients,
            numerator_degree,
            denominator_degree,
            center,
            domain,
            policy,
            ExecutionContext::sequential_native(),
        )
    }

    /// Computes the `[m/n]` approximant in a solve scope, with `m = numerator_degree` and `n = denominator_degree`.
    ///
    /// The construction counts one iteration before any numerical work, and
    /// the SVD uses the executor of the scope.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `m + n + 1` overflows `usize`.
    /// - `DimensionMismatch` if there are fewer than `m + n + 1` coefficients.
    /// - `NonFinite` if a coefficient is NaN or infinite, or an intermediate
    ///   overflows.
    /// - `DomainError` if `center` lies outside `domain`.
    /// - `ConvergenceFailure` (`IterationLimit`) if the scope has no iteration
    ///   left.
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - `NumericalInstability` (`Breakdown`) if the Toeplitz system is
    ///   singular and inconsistent, so the residual of the Padé conditions
    ///   exceeds the tolerance of `policy`.
    pub fn build<'scope, 'capability>(
        coefficients: &[C],
        numerator_degree: usize,
        denominator_degree: usize,
        center: C,
        domain: RationalDomain<C::Real>,
        policy: PolynomialApproximationPolicy<C::Real>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let order = numerator_degree
            .checked_add(denominator_degree)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("pade_order")))?;
        if coefficients.len() < order {
            return Err(shape(order, coefficients.len()));
        }
        validate_finite(coefficients)?;
        domain.resolve(
            center,
            ExtrapolationPolicy::Reject,
            scioxide_core::argument!("center"),
        )?;
        let plane = domain.plane()?;
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "pade", 1),
            order as u128,
            ((numerator_degree as u128) << 64) | denominator_degree as u128,
        );
        let (rational, coefficient_residual, denominator_rank, singular_ratio) = scope.into().run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |context| {
                solve(
                    &coefficients[..order],
                    numerator_degree,
                    denominator_degree,
                    policy,
                    context,
                )
            },
        )?;
        Ok(Self {
            rational,
            center,
            domain,
            plane,
            coefficient_residual,
            denominator_rank,
            report: RationalReport::new(
                RationalTermination::Converged,
                1,
                coefficient_residual,
                singular_ratio,
            ),
        })
    }

    /// Returns `p / q` as polynomials in `u = z − center`.
    pub const fn rational(&self) -> &ApproxRationalFunction<C> {
        &self.rational
    }

    /// Returns the expansion point `z₀`.
    pub const fn center(&self) -> C {
        self.center
    }

    /// Returns the approximation domain.
    pub const fn domain(&self) -> RationalDomain<C::Real> {
        self.domain
    }

    /// Returns the largest absolute coefficient of `q f − p` through degree `m + n`.
    pub const fn coefficient_residual(&self) -> C::Real {
        self.coefficient_residual
    }

    /// Returns the numerical rank of the `n × n` denominator system under the rank policy.
    #[must_use]
    pub const fn denominator_rank(&self) -> usize {
        self.denominator_rank
    }

    /// Returns the construction report.
    ///
    /// The status is `Converged` with one iteration. The residual is
    /// [`Self::coefficient_residual`], and the singular-value ratio is that of
    /// the denominator system, or one when `n = 0`.
    pub const fn fit_report(&self) -> RationalReport<C::Real> {
        self.report
    }

    /// Returns the value at a resolved coordinate.
    fn field_value(&self, z: C) -> Result<C, SciOxideError> {
        self.rational.evaluate(intermediate(z - self.center)?)
    }
}

rational_interpolant!(PadeApproximant);

/// Rational function, coefficient residual, denominator rank and singular ratio.
type PadeParts<C> = (
    ApproxRationalFunction<C>,
    <C as NumericScalar>::Real,
    usize,
    <C as NumericScalar>::Real,
);

/// Solves for the Padé coefficients after counting one iteration.
fn solve<C: NumericScalar>(
    coefficients: &[C],
    numerator_degree: usize,
    denominator_degree: usize,
    policy: PolynomialApproximationPolicy<C::Real>,
    context: &mut SolveContext<'_, '_>,
) -> Result<PadeParts<C>, SciOxideError> {
    crate::error::require_iteration(context, "iteration budget denied coefficient construction")?;
    let order = coefficients.len();
    // Padé conditions for type (m, n) = (numerator_degree, denominator_degree):
    //   Q(z) f(z) - P(z) = O(z^(m+n+1)),
    // where the z^k coefficient of Q f is sum_{j=0}^{min(k,n)} q_j c_{k-j}.
    // With the normalization q_0 = 1 the rows k = m+1..=m+n give the n x n
    // linear system sum_{j=1}^{n} q_j c_{k-j} = -c_k for the denominator,
    // after which P is read off from rows k = 0..=m. All coefficients are
    // divided by their maximum magnitude (floored at the smallest normal so
    // an all-zero series stays well defined); the system is homogeneous in
    // that scale, so q is unchanged and p is rescaled on output.
    let scale = coefficients
        .iter()
        .fold(C::Real::ZERO, |s, c| s.max(c.abs()))
        .max(C::Real::MIN_POSITIVE);
    let mut q = try_filled(
        denominator_degree + 1,
        C::ZERO,
        scioxide_core::quantity!("pade_denominator"),
    )?;
    q[0] = C::ONE;
    let (rank, singular_ratio) = if denominator_degree > 0 {
        let mut matrix = Matrix::zeros(denominator_degree, denominator_degree)?;
        let mut rhs = try_filled(
            denominator_degree,
            C::ZERO,
            scioxide_core::quantity!("pade_rhs"),
        )?;
        for (row, b) in rhs.iter_mut().enumerate() {
            let k = numerator_degree + 1 + row;
            *b = -coefficients[k] / C::from_real(scale);
            for col in 0..denominator_degree {
                let j = col + 1;
                matrix.view_mut().set(
                    row,
                    col,
                    if j <= k {
                        coefficients[k - j] / C::from_real(scale)
                    } else {
                        C::ZERO
                    },
                )?;
            }
        }
        let mut svd = ThinSvdDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        ThinSvdDecomposition::compute_into(
            &matrix,
            &LinalgExecutionOptions::default(),
            &mut svd,
            &mut workspace,
            context.execution(),
        )?;
        // The Toeplitz system is singular for degenerate Padé blocks. A
        // rank-truncated SVD solve returns the minimum-norm solution, which
        // is valid whenever the system is consistent; consistency is
        // verified below through the residual of the dropped equations.
        let rank = svd.analyze_rank(policy.rank_policy())?.rank();
        let solution = svd.solve(
            &Vector::from_vec(rhs),
            policy.rank_policy(),
            <_ as AlgebraicScalar>::ZERO,
        )?;
        q[1..].copy_from_slice(solution.as_slice());
        let values = svd.singular_values();
        let ratio = match (values.first(), values.last()) {
            (Some(&high), Some(&low)) if high > C::Real::ZERO => low / high,
            _ => C::Real::ZERO,
        };
        (rank, ratio)
    } else {
        (0, C::Real::ONE)
    };
    // One pass over all m+n+1 coefficient equations: the first m+1 define
    // P, the remaining n should vanish and their largest magnitude is the
    // reported coefficient residual (an inconsistent singular system shows
    // up here rather than in the SVD solve).
    let mut p = try_filled(
        numerator_degree + 1,
        C::ZERO,
        scioxide_core::quantity!("pade_numerator"),
    )?;
    let mut error = C::Real::ZERO;
    for k in 0..order {
        let sum = q
            .iter()
            .enumerate()
            .take(k + 1)
            .try_fold(C::ZERO, |s, (j, &qj)| {
                intermediate(s + qj * (coefficients[k - j] / C::from_real(scale)))
            })?;
        if k <= numerator_degree {
            p[k] = intermediate(sum * C::from_real(scale))?;
        } else {
            error = error.max(sum.abs());
        }
    }
    if !policy.tolerance().is_residual_ok(error * scale, scale)? {
        return Err(instability(
            scioxide_core::NumericalIssue::Breakdown,
            "inconsistent defective Pade coefficient system",
        ));
    }
    let coefficient_residual = error * scale;
    if !coefficient_residual.is_finite() {
        return Err(instability(
            scioxide_core::NumericalIssue::Breakdown,
            "unrepresentable Pade residual",
        ));
    }
    Ok((
        ApproxRationalFunction::new(
            ApproxPolynomial::monomial_with_options(p, policy)?,
            ApproxPolynomial::monomial_with_options(q, policy)?,
        )?,
        coefficient_residual,
        rank,
        singular_ratio,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Interpolant;
    use scioxide_core::{BudgetKind, ConvergenceReason};
    use scioxide_core::{Complex, Tolerance};
    use scioxide_linalg::RankPolicy;

    fn policy() -> PolynomialApproximationPolicy<f64> {
        PolynomialApproximationPolicy::new(
            Tolerance::new(1e-12, 1e-12).unwrap(),
            RankPolicy::DimensionScaledEpsilon,
        )
    }

    #[test]
    fn diagnostics_and_report_share_the_rational_vocabulary() {
        let domain = RationalDomain::Real(Interval::new(-1.0, 1.0).unwrap());
        let pade = PadeApproximant::<f64>::with_options(&[1., 1., 1.], 1, 1, 0., domain, policy())
            .unwrap();
        assert_eq!(pade.denominator_rank(), 1);
        assert!(pade.coefficient_residual() <= 1e-15);
        let report = pade.fit_report();
        assert_eq!(report.termination(), RationalTermination::Converged);
        assert_eq!(report.iterations(), 1);
        assert_eq!(report.max_residual(), pade.coefficient_residual());
        assert_eq!(report.singular_ratio(), 1.0);
        assert!((pade.evaluate([0.2], ExtrapolationPolicy::Reject).unwrap() - 1.25).abs() < 1e-12);
        let constant =
            PadeApproximant::<f64>::with_options(&[2.], 0, 0, 0., domain, policy()).unwrap();
        assert_eq!(constant.fit_report().singular_ratio(), 1.0);
        assert_eq!(constant.denominator_rank(), 0);
    }

    #[test]
    fn a_denied_admission_is_an_iteration_limit_failure_before_any_work() {
        // Public scopes give Padé a positive local allowance, so the denial
        // branch is exercised with a context whose allowance is already spent.
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, scioxide_core::Budget::ONE);
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "pade", 1),
            3,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        assert!(context.record_iteration().unwrap().is_granted());
        let error = solve(&[1.0_f64, 1., 1.], 1, 1, policy(), &mut context).unwrap_err();
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::IterationLimit,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(context.counts().own().iterations(), 1);
    }

    #[test]
    fn complex_pade_uses_both_coordinate_forms() {
        let domain = RationalDomain::Complex {
            real: Interval::new(-1.0, 1.0).unwrap(),
            imaginary: Interval::new(-1.0, 1.0).unwrap(),
        };
        let one = Complex::new(1.0_f64, 0.0);
        let pade = PadeApproximant::with_options(
            &[one, one, one],
            1,
            1,
            Complex::new(0.0, 0.0),
            domain,
            policy(),
        )
        .unwrap();
        let value = pade
            .evaluate([0.0, 0.5], ExtrapolationPolicy::Reject)
            .unwrap();
        let expected = one / (one - Complex::new(0.0, 0.5));
        assert!((value - expected).norm() < 1e-12);
        assert!(
            Interpolant::<f64, 1>::evaluate(&pade, [0.2], ExtrapolationPolicy::Reject).is_err()
        );
    }
}
