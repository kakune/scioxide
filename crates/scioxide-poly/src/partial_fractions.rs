//! Floating-point partial fractions over a given factorization, by solving the
//! equilibrated linear system for the term numerators.
use crate::PolynomialBasis;
use crate::root_diagnostics::polynomial_root_condition;
use crate::util::{bytes_of, minimum_len, round_complex, scaled_complex, try_grow, validate_slice};
use crate::{ApproxPolynomial, MonomialBasis, RootConditionReport};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Range;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::try_copy_slice;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Argument, Capacity, ExecutionContext, NumericScalar, Quantity, RealScalar, SciOxideError,
};
use scioxide_core::{Scaled, ScaledValue};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgRequirements, LinalgWorkspace, Matrix,
    RankReport,
};

type Polynomial<C> = ApproxPolynomial<C, MonomialBasis<C>>;

fn unstable(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::IllConditioned, detail)
}
fn component_max<C: NumericScalar>(value: C) -> C::Real {
    value
        .real_component()
        .abs()
        .max(value.imaginary_component().abs())
}
fn divide<C: NumericScalar>(value: C, scale: C::Real) -> Result<C, SciOxideError> {
    round_complex(scaled_complex(value)?.div(scaled_complex(C::from_real(scale))?)?)
}
fn max_scaled<R: RealScalar>(a: Scaled<R>, b: Scaled<R>) -> Scaled<R> {
    if a.abs_le(b) { b } else { a }
}
fn relative<R: RealScalar>(error: Scaled<R>, scale: Scaled<R>) -> Result<R, SciOxideError> {
    if error.is_zero() {
        Ok(R::ZERO)
    } else {
        error.div(scale)?.round()
    }
}
fn coefficient_error<C: NumericScalar>(a: &[C], b: &[C]) -> Result<C::Real, SciOxideError> {
    let mut error = Scaled::ZERO;
    let mut scale = Scaled::ZERO;
    for (&a, &b) in a.iter().zip(b) {
        error = max_scaled(
            error,
            scaled_complex(a)?.sub(scaled_complex(b)?)?.magnitude()?,
        );
        scale = max_scaled(scale, scaled_complex(a)?.magnitude()?);
        scale = max_scaled(scale, scaled_complex(b)?.magnitude()?);
    }
    relative(error, scale)
}

/// Holds one denominator factor `F` of an [`ApproxPartialFractionPlan`], its
/// multiplicity and, for a linear factor, the conditioning of its pole.
///
/// ```
/// use scioxide_poly::*;
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// let factor = &plan.factors()[0];
/// assert_eq!(factor.multiplicity(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PartialFractionFactor<C: NumericScalar> {
    polynomial: Polynomial<C>,
    multiplicity: usize,
    pole_condition: Option<RootConditionReport<C::Real>>,
}
impl<C: NumericScalar> PartialFractionFactor<C> {
    /// Returns the monic factor `F` as passed to the plan.
    ///
    /// The plan does not test whether `F` is irreducible.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let factor = &plan.factors()[0];
    /// assert_eq!(factor.polynomial(), &x);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn polynomial(&self) -> &Polynomial<C> {
        &self.polynomial
    }
    /// Returns the multiplicity `m ≥ 1` of `F` in the denominator.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let factor = &plan.factors()[0];
    /// assert_eq!(factor.multiplicity(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn multiplicity(&self) -> usize {
        self.multiplicity
    }
    /// Returns the condition of the pole `a` of a linear factor `F = x − a` as a
    /// root of the full denominator.
    ///
    /// `None` for factors of degree 2 or more, which have no single pole.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let factor = &plan.factors()[0];
    /// assert!(factor.pole_condition().is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn pole_condition(&self) -> Option<RootConditionReport<C::Real>> {
        self.pole_condition
    }
}

/// Describes where the numerator of one term `A / Fᵏ` lies in
/// [`ApproxPartialFractions::residues`].
///
/// ```
/// use scioxide_poly::*;
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// let term = &plan.terms()[0];
/// assert_eq!(term.power(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialFractionTerm {
    factor_index: usize,
    power: usize,
    coefficients: Range<usize>,
}
impl PartialFractionTerm {
    /// Returns the index of `F` in [`ApproxPartialFractionPlan::factors`].
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let term = &plan.terms()[0];
    /// assert_eq!(term.factor_index(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn factor_index(&self) -> usize {
        self.factor_index
    }
    /// Returns the power `k ≥ 1` of `F` in the term's denominator.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let term = &plan.terms()[0];
    /// assert_eq!(term.power(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn power(&self) -> usize {
        self.power
    }
    /// Returns the range of [`ApproxPartialFractions::residues`] that holds the
    /// coefficients of `A`, in ascending order of degree.
    ///
    /// The range has length `deg F`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let term = &plan.terms()[0];
    /// assert_eq!(term.coefficients(), 0..1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coefficients(&self) -> Range<usize> {
        self.coefficients.clone()
    }
}

/// Reports the diagnostics of a partial-fraction decomposition.
///
/// All values are floating-point estimates, not bounds.
///
/// ```
/// use scioxide_poly::*;
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// let result = plan.decompose(&[1.0_f64])?;
/// let report = result
///     .report()
///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
/// assert!(report.reconstruction_error() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PartialFractionReport<R: RealScalar> {
    factorization_error: R,
    reconstruction_error: R,
    rank: RankReport<R>,
    reciprocal_condition: Option<R>,
}
impl<R: RealScalar> PartialFractionReport<R> {
    /// Returns `‖D − Π Fᵢ^mᵢ‖∞ / max(‖D‖∞, ‖Π Fᵢ^mᵢ‖∞)` over the coefficients,
    /// the relative error of the factor product as a representation of the
    /// denominator `D`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// let report = result
    ///     .report()
    ///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
    /// assert!(report.factorization_error() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn factorization_error(self) -> R {
        self.factorization_error
    }
    /// Returns the relative residual of the numerator rebuilt from the
    /// polynomial part and the terms.
    ///
    /// It is the largest coefficient error `|N_j − Ñ_j|` divided by the
    /// largest `s_j`, where `s_j` adds `|N_j|` and the magnitudes of all
    /// products that contribute to the rebuilt coefficient `Ñ_j`. Cancellation
    /// in the reconstruction therefore shows as a large value.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// let report = result
    ///     .report()
    ///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
    /// assert!(report.reconstruction_error() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn reconstruction_error(self) -> R {
        self.reconstruction_error
    }
    /// Returns the numerical rank of the equilibrated `n × n` coefficient
    /// system, where `n` is the degree of the denominator.
    ///
    /// The plan accepts only full rank, so this equals `n`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// let report = result
    ///     .report()
    ///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
    /// assert_eq!(report.rank().rank(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn rank(self) -> RankReport<R> {
        self.rank
    }
    /// Returns an estimate of the reciprocal condition number of the
    /// equilibrated coefficient system, from its column-pivoted QR factor.
    ///
    /// `None` for a constant denominator, which has no system to solve.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// let report = result
    ///     .report()
    ///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
    /// assert!(
    ///     report
    ///         .reciprocal_condition()
    ///         .is_some_and(|value| value > 0.0)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn reciprocal_condition(self) -> Option<R> {
        self.reciprocal_condition
    }
}

/// Decomposes floating-point rational functions over a fixed factored
/// denominator.
///
/// For a monic denominator `D = F₁^m₁ ⋯ F_r^m_r` of degree `n` with monic,
/// nonconstant and pairwise coprime factors `Fᵢ`, a numerator `N` is written as
///
/// ```text
/// N / D = P + Σᵢ Σ_{k=1}^{mᵢ} A_{i,k} / Fᵢᵏ,    deg A_{i,k} < deg Fᵢ,
/// ```
///
/// where `P` is the polynomial quotient. For a linear factor `x − a`, the
/// constants `A_{i,k}` are the coefficients of the principal part of the
/// Laurent expansion at the pole `a`, and `A_{i,1}` is the residue. You supply the factorization; to find the
/// poles, compute the roots of `D` first. The factors need not be irreducible,
/// and close poles are never merged.
///
/// Multiplying by `D` gives a square linear system of order `n` for the
/// coefficients of all `A_{i,k}`, whose columns are the shifted cofactors
/// `D / Fᵢᵏ`. Construction forms this system, scales its columns and rows by
/// their largest entries, and factors it by Householder QR with column
/// pivoting, in O(n³) operations and O(n²) storage. The system is singular
/// exactly when two factors share a root, so a rank deficiency under the
/// policy's rank policy rejects nearly common factors. Each
/// [`decompose_into`](Self::decompose_into) then costs
/// O(n² + len(N) · (n + 1)) operations. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// For exact coefficients use
/// [`ExactPartialFractionPlan`](crate::ExactPartialFractionPlan).
///
/// # Examples
///
/// ```
/// use scioxide_poly::*;
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// assert_eq!(plan.denominator().degree(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ApproxPartialFractionPlan<C: NumericScalar> {
    denominator: Polynomial<C>,
    product: Polynomial<C>,
    factors: Vec<PartialFractionFactor<C>>,
    terms: Vec<PartialFractionTerm>,
    matrix: Matrix<C>,
    qr: HouseholderDecomposition<C>,
    column_scales: Vec<C::Real>,
    row_scales: Vec<C::Real>,
    report: PartialFractionReport<C::Real>,
}

impl<C: NumericScalar> ApproxPartialFractionPlan<C> {
    /// Creates the plan for `denominator = Π Fᵢ^mᵢ`, given as pairs
    /// `(Fᵢ, mᵢ)`.
    ///
    /// The factors must carry the denominator's
    /// [`PolynomialApproximationPolicy`](crate::PolynomialApproximationPolicy).
    /// The relative coefficient error of the factor product (see
    /// [`PartialFractionReport::factorization_error`]) must pass the policy's
    /// tolerance as a residual test with scale 1. An empty factor list is valid
    /// only for the denominator 1.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `denominator` is zero or not monic,
    ///   a factor is constant or not monic, a multiplicity is 0, the degrees
    ///   `Σ mᵢ deg Fᵢ` differ from `deg denominator`, or a factor has another
    ///   policy.
    /// - `NumericalInstability` (`IllConditioned`) if the factor product misses
    ///   the denominator by more than the tolerance, or the coefficient system
    ///   is rank deficient because two factors are not numerically coprime.
    /// - `SizeOverflow` or `AllocationFailure` if the coefficient system cannot
    ///   be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// // 1 / (x (x + 1)) = 1/x − 1/(x + 1).
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// assert!((result.residues()[0] - 1.0).abs() < 1e-12);
    /// assert!((result.residues()[1] + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        denominator: &Polynomial<C>,
        factors: &[(&Polynomial<C>, usize)],
    ) -> Result<Self, SciOxideError> {
        crate::util::validate_factors(
            denominator.degree(),
            !denominator.is_zero() && denominator.leading_coefficient() == C::ONE,
            factors
                .iter()
                .map(|(p, m)| (p.degree(), p.leading_coefficient() == C::ONE, *m)),
        )?;
        let policy = denominator.policy();
        factors.iter().try_for_each(|(factor, _)| {
            crate::util::same_policy(policy, factor.policy(), scioxide_core::argument!("factors"))
        })?;
        let n = denominator.degree();
        // Reject an unrepresentable `n x n` coefficient system before any product work.
        scioxide_core::backend::checked_mul(
            n,
            n,
            scioxide_core::quantity!("coefficient_system_entries"),
        )?;
        let mut product = Polynomial::monomial_with_options(
            try_filled(1, C::ONE, scioxide_core::quantity!("factor_product"))?,
            policy,
        )?;
        let mut prepared = try_with_capacity(factors.len(), scioxide_core::quantity!("factors"))?;
        for (p, multiplicity) in factors {
            for _ in 0..*multiplicity {
                product = product.try_mul(p)?;
            }
            // Only a linear factor names a single pole (`-c_0`); its conditioning is that
            // of the root of the *full* denominator, since that is what the caller's data
            // perturbs.
            let pole_condition = if p.degree() == 1 {
                let pole = -p.basis().coefficients()[0];
                Some(polynomial_root_condition(
                    denominator.basis().coefficients(),
                    scioxide_core::Complex::new(pole.real_component(), pole.imaginary_component()),
                )?)
            } else {
                None
            };
            prepared.push(PartialFractionFactor {
                polynomial: Polynomial::monomial_with_options(
                    try_copy_slice(
                        p.basis().coefficients(),
                        scioxide_core::quantity!("factor_coefficients"),
                    )?,
                    policy,
                )?,
                multiplicity: *multiplicity,
                pole_condition,
            });
        }
        let factorization_error = coefficient_error(
            denominator.basis().coefficients(),
            product.basis().coefficients(),
        )?;
        if !policy
            .tolerance()
            .is_residual_ok(factorization_error, C::Real::ONE)?
        {
            return Err(unstable(
                "factor product does not reconstruct denominator within tolerance",
            ));
        }
        // Two-sided max-magnitude equilibration `E = D_r^{-1} A D_c^{-1}` (columns
        // first, then rows) so the QR rank decision is not skewed by complements whose
        // coefficient magnitudes differ widely. `run` applies `D_r` to the right-hand
        // side and `D_c` to the solution. The unscaled `matrix` is retained for the
        // reconstruction residual.
        let (matrix, terms) = coefficient_system(factors, n, policy)?;
        let mut equilibrated = Matrix::from_vec(
            n,
            n,
            try_copy_slice(
                matrix.as_slice(),
                scioxide_core::quantity!("coefficient_system_entries"),
            )?,
        )?;
        let mut column_scales =
            try_filled(n, C::Real::ONE, scioxide_core::quantity!("column_scales"))?;
        for (column, scale) in column_scales.iter_mut().enumerate() {
            *scale = (0..n).fold(C::Real::ZERO, |s, row| {
                s.max(component_max(equilibrated.as_slice()[row * n + column]))
            });
            if *scale == C::Real::ZERO {
                return Err(unstable("zero coefficient-system column"));
            }
            for row in 0..n {
                let value = &mut equilibrated.as_mut_slice()[row * n + column];
                *value = divide(*value, *scale)?;
            }
        }
        let mut row_scales = try_filled(n, C::Real::ONE, scioxide_core::quantity!("row_scales"))?;
        for (row, scale) in row_scales.iter_mut().enumerate() {
            let values = &mut equilibrated.as_mut_slice()[row * n..(row + 1) * n];
            *scale = values
                .iter()
                .copied()
                .fold(C::Real::ZERO, |s, c| s.max(component_max(c)));
            if *scale == C::Real::ZERO {
                return Err(unstable("zero coefficient-system row"));
            }
            for value in values {
                *value = divide(*value, *scale)?;
            }
        }
        let qr = HouseholderDecomposition::factor(
            &equilibrated,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
        )?;
        // The system is nonsingular exactly when the factors are pairwise coprime, so
        // a rank deficiency here is how near-common factors are detected.
        let rank = qr.analyze_rank(policy.rank_policy())?;
        if !rank.is_full_rank() {
            return Err(unstable("factors are noncoprime or numerically unresolved"));
        }
        let reciprocal_condition = (rank.maximum_rank() != 0)
            .then(|| {
                qr.condition_estimate()
                    .map(|estimate| estimate.reciprocal().value())
            })
            .transpose()?;
        Ok(Self {
            denominator: Polynomial::monomial_with_options(
                try_copy_slice(
                    denominator.basis().coefficients(),
                    scioxide_core::quantity!("denominator_coefficients"),
                )?,
                policy,
            )?,
            product,
            factors: prepared,
            terms,
            matrix,
            qr,
            column_scales,
            row_scales,
            report: PartialFractionReport {
                factorization_error,
                reconstruction_error: C::Real::ZERO,
                rank,
                reciprocal_condition,
            },
        })
    }

    /// Returns the denominator `D`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// assert_eq!(plan.denominator().basis().coefficients(), &[0.0, 1.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn denominator(&self) -> &Polynomial<C> {
        &self.denominator
    }
    /// Returns the factors in the order passed to [`new`](Self::new).
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// assert_eq!(plan.factors().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn factors(&self) -> &[PartialFractionFactor<C>] {
        &self.factors
    }
    /// Returns the layout of the terms, ordered by factor and then by ascending
    /// power.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// assert_eq!(plan.terms().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn terms(&self) -> &[PartialFractionTerm] {
        &self.terms
    }
    /// Returns the workspace and output sizes for numerators of at most
    /// `numerator_len` coefficients.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `numerator_len` is 0.
    /// - `SizeOverflow` if a size overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// assert_eq!(plan.workspace_requirements(1)?.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        numerator_len: usize,
    ) -> Result<PartialFractionRequirements<C>, SciOxideError> {
        self.lengths(numerator_len, scioxide_core::argument!("numerator_len"))
    }
    /// Allocates a workspace for numerators of at most `numerator_len`
    /// coefficients.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `numerator_len` is 0.
    /// - `SizeOverflow` or `AllocationFailure` if the workspace cannot be
    ///   allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let workspace = plan.workspace(1)?;
    /// assert_eq!(
    ///     workspace.check(&plan.workspace_requirements(1)?)?,
    ///     scioxide_core::Capacity::Sufficient
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        &self,
        numerator_len: usize,
    ) -> Result<PartialFractionWorkspace<C>, SciOxideError> {
        let mut workspace = PartialFractionWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(numerator_len)?)?;
        Ok(workspace)
    }
    /// Storage for `numerator_len` coefficients; an empty numerator is reported
    /// against `argument`.
    fn lengths(
        &self,
        numerator_len: usize,
        argument: Argument,
    ) -> Result<PartialFractionRequirements<C>, SciOxideError> {
        if numerator_len == 0 {
            return Err(crate::util::empty_error(argument));
        }
        let n = self.denominator.degree();
        let requirements = PartialFractionRequirements {
            // The quotient has `numerator_len - n` coefficients (at least one),
            // which shrinks as the degree grows. Reserving the numerator length
            // keeps every requirement monotone in both plan degree and numerator
            // length, so storage reserved for one plan serves every smaller one.
            quotient: numerator_len,
            remainder: numerator_len.max(n),
            residues: n,
            scaled: crate::division::scratch_len(numerator_len, n + 1)?,
            qr: HouseholderDecomposition::<C>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                    n,
                    n,
                    HouseholderOrientation::Qr,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
            scalar: PhantomData,
        };
        requirements.required_bytes()?;
        Ok(requirements)
    }

    /// Computes the partial-fraction decomposition of `N / D` for the
    /// coefficients of `N` in ascending order of degree.
    ///
    /// Allocates the workspace and output, then calls
    /// [`decompose_into`](Self::decompose_into).
    ///
    /// # Errors
    ///
    /// The errors of [`decompose_into`](Self::decompose_into), and
    /// `AllocationFailure` if the workspace or output cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// assert!((result.residues()[0] - 1.0).abs() < 1e-12);
    /// assert!((result.residues()[1] + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn decompose(&self, numerator: &[C]) -> Result<ApproxPartialFractions<C>, SciOxideError> {
        let requirements = self.lengths(numerator.len(), scioxide_core::argument!("numerator"))?;
        let mut workspace = PartialFractionWorkspace::new();
        let mut output = ApproxPartialFractions::new();
        workspace.try_reserve(&requirements)?;
        output.try_reserve(&requirements)?;
        self.decompose_into(
            numerator,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the partial-fraction decomposition of `N / D` into `output`.
    ///
    /// Divides `N` by the factor product to get `P` and a remainder of degree
    /// below `n`, solves the prepared QR system for the term numerators, and
    /// checks the [`reconstruction_error`](PartialFractionReport::reconstruction_error)
    /// against the policy's tolerance. Reserve the workspace and the output
    /// with [`workspace_requirements`](Self::workspace_requirements) for at
    /// least `numerator.len()` coefficients. On error, `output` is unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `numerator` is empty.
    /// - `NonFinite` if a coefficient of `numerator` is NaN or infinite.
    /// - `DimensionMismatch` if the workspace or the output capacity is too
    ///   small.
    /// - `NumericalInstability` (`IllConditioned`) if the solve does not accept
    ///   full rank or the reconstruction error exceeds the tolerance.
    /// - `NumericalInstability` if an intermediate value overflows.
    /// - `ExecutionStopped` if `context` cancels the computation or its deadline
    ///   passes.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// let mut workspace = plan.workspace(1)?;
    /// let mut result = ApproxPartialFractions::new();
    /// result.try_reserve(&req)?;
    /// plan.decompose_into(
    ///     &[1.0_f64],
    ///     &mut result,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!((result.residues()[0] - 1.0).abs() < 1e-12);
    /// assert!((result.residues()[1] + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn decompose_into(
        &self,
        numerator: &[C],
        output: &mut ApproxPartialFractions<C>,
        workspace: &mut PartialFractionWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let lengths = self.lengths(numerator.len(), scioxide_core::argument!("numerator"))?;
        validate_slice(numerator)?;
        workspace.fits(&lengths)?;
        let n = lengths.residues;
        let quotient_len = numerator.len().saturating_sub(n).max(1);
        minimum_len(quotient_len, output.quotient.capacity())?;
        minimum_len(lengths.residues, output.residues.capacity())?;
        // Divide by the reconstructed `product` rather than the caller's denominator
        // so the polynomial part and the proper remainder are consistent with the
        // factorization that the coefficient system is built from.
        crate::division::division_into(
            numerator,
            self.product.basis().coefficients(),
            &mut workspace.quotient[..quotient_len],
            &mut workspace.remainder,
            &mut workspace.scaled,
            || context.check(),
        )?;
        // Common RHS equilibration keeps triangular arithmetic in scalar range.
        let rhs_scale = workspace.remainder[..n]
            .iter()
            .copied()
            .fold(C::Real::ZERO, |s, c| s.max(component_max(c)))
            // Clamped so an all-zero remainder does not divide by zero.
            .max(C::Real::MIN_POSITIVE);
        for (row, rhs) in workspace.rhs[..n].iter_mut().enumerate() {
            context.check()?;
            *rhs = round_complex(
                scaled_complex(workspace.remainder[row])?
                    .div(scaled_complex(C::from_real(rhs_scale))?)?
                    .div(scaled_complex(C::from_real(self.row_scales[row]))?)?,
            )?;
        }
        context.check()?;
        let solved = self.qr.solve_into(
            &workspace.rhs[..n],
            &(scioxide_linalg::DirectSolveOptions::new(self.denominator.policy().rank_policy()))
                .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()),
            scioxide_linalg::VectorViewMut::from_slice(&mut workspace.solution[..n]),
            &mut workspace.qr,
            context,
        )?;
        if solved.termination() != scioxide_linalg::DirectSolveTermination::Accepted {
            return Err(unstable("coefficient solve did not accept full rank"));
        }
        // Undo equilibration: `x = D_c^{-1} y * rhs_scale`.
        for (column, value) in workspace.solution[..n].iter_mut().enumerate() {
            *value = round_complex(
                scaled_complex(*value)?
                    .mul(scaled_complex(C::from_real(rhs_scale))?)?
                    .div(scaled_complex(C::from_real(self.column_scales[column]))?)?,
            )?;
        }
        let reconstruction_error =
            self.reconstruction_error(numerator, quotient_len, workspace, context)?;
        if !self
            .denominator
            .policy()
            .tolerance()
            .is_residual_ok(reconstruction_error, C::Real::ONE)?
        {
            return Err(unstable("numerator reconstruction exceeds tolerance"));
        }
        context.check()?;
        output.quotient.clear();
        output
            .quotient
            .extend_from_slice(&workspace.quotient[..quotient_len]);
        while output.quotient.len() > 1 && output.quotient.last() == Some(&C::ZERO) {
            output.quotient.pop();
        }
        output.residues.clear();
        output.residues.extend_from_slice(&workspace.solution[..n]);
        output.report = Some(PartialFractionReport {
            reconstruction_error,
            ..self.report
        });
        Ok(())
    }

    fn reconstruction_error(
        &self,
        numerator: &[C],
        quotient_len: usize,
        workspace: &PartialFractionWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<C::Real, SciOxideError> {
        let n = self.denominator.degree();
        let mut max_error = Scaled::ZERO;
        let mut max_scale = Scaled::ZERO;
        // Residual of `numerator = quotient * product + sum_k A_k * complement_k`,
        // coefficient by coefficient. Each coefficient's scale also accumulates the
        // magnitudes of every contributing term, so the relative error measures
        // cancellation instead of being hidden by a small result coefficient.
        for degree in 0..numerator.len().max(n) {
            context.check()?;
            let original = numerator.get(degree).copied().unwrap_or(C::ZERO);
            let mut reconstructed = ScaledValue::ZERO;
            let mut scale = scaled_complex(original)?.magnitude()?;
            for (qdegree, &q) in workspace.quotient[..quotient_len]
                .iter()
                .enumerate()
                .take(degree.saturating_add(1))
                // Only quotient terms whose partner `product` coefficient exists
                // (`degree - qdegree <= n`) contribute.
                .skip(degree.saturating_sub(n))
            {
                if let Some(ddegree) = degree.checked_sub(qdegree)
                    && let Some(&d) = self.product.basis().coefficients().get(ddegree)
                {
                    let term = scaled_complex(q)?.mul(scaled_complex(d)?)?;
                    reconstructed = reconstructed.add(term)?;
                    scale = scale.add(term.magnitude()?)?;
                }
            }
            if degree < n {
                for (column, &residue) in workspace.solution[..n].iter().enumerate() {
                    let term = scaled_complex(self.matrix.as_slice()[degree * n + column])?
                        .mul(scaled_complex(residue)?)?;
                    reconstructed = reconstructed.add(term)?;
                    scale = scale.add(term.magnitude()?)?;
                }
            }
            max_error = max_scaled(
                max_error,
                reconstructed.sub(scaled_complex(original)?)?.magnitude()?,
            );
            max_scale = max_scaled(max_scale, scale);
        }
        relative(max_error, max_scale)
    }
}

fn coefficient_system<C: NumericScalar>(
    factors: &[(&Polynomial<C>, usize)],
    n: usize,
    policy: crate::PolynomialApproximationPolicy<C::Real>,
) -> Result<(Matrix<C>, Vec<PartialFractionTerm>), SciOxideError> {
    let mut matrix = Matrix::zeros(n, n)?;
    let mut terms = try_with_capacity(n, scioxide_core::quantity!("partial_fraction_terms"))?;
    let mut column = 0;
    // Multiplying `R / D = sum_{j,p} A_{j,p} / F_j^p` by `D` gives
    // `R = sum_{j,p} A_{j,p} * F_j^{m_j - p} prod_{k != j} F_k^{m_k}`. Each unknown
    // coefficient `a x^s` of `A_{j,p}` (with `s < deg F_j`) contributes the complement
    // shifted by `s` as one column, giving a square `n x n` system in ascending
    // coefficient rows.
    for (index, (factor, multiplicity)) in factors.iter().enumerate() {
        for power in 1..=*multiplicity {
            // Build complements by multiplication, avoiding unstable root deflation.
            let mut complement = Polynomial::monomial_with_options(
                try_filled(
                    1,
                    C::ONE,
                    scioxide_core::quantity!("complement_coefficients"),
                )?,
                policy,
            )?;
            for (other_index, (other, count)) in factors.iter().enumerate() {
                let count = if index == other_index {
                    count - power
                } else {
                    *count
                };
                for _ in 0..count {
                    complement = complement.try_mul(other)?;
                }
            }
            let start = column;
            for shift in 0..factor.degree() {
                for (row, &value) in complement.basis().coefficients().iter().enumerate() {
                    matrix.as_mut_slice()[(row + shift) * n + column] = value;
                }
                column += 1;
            }
            terms.push(PartialFractionTerm {
                factor_index: index,
                power,
                coefficients: start..column,
            });
        }
    }
    Ok((matrix, terms))
}

/// Gives the workspace and output sizes of an [`ApproxPartialFractionPlan`] for
/// numerators up to a given length.
///
/// The same value reserves both [`PartialFractionWorkspace`] and
/// [`ApproxPartialFractions`]. Returned by
/// [`ApproxPartialFractionPlan::workspace_requirements`].
///
/// ```
/// use scioxide_poly::*;
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// let req = plan.workspace_requirements(1)?;
/// assert_eq!(req.degree(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartialFractionRequirements<C: NumericScalar> {
    quotient: usize,
    remainder: usize,
    residues: usize,
    scaled: usize,
    qr: LinalgRequirements,
    scalar: PhantomData<fn() -> C>,
}
impl<C: NumericScalar> PartialFractionRequirements<C> {
    /// Returns the denominator degree `n`, which is both the number of term
    /// coefficients and the order of the coefficient system.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// assert_eq!(req.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn degree(self) -> usize {
        self.residues
    }
    /// Returns the bytes of workspace storage, excluding the output and `Vec`
    /// headers.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// assert!(req.required_bytes()? >= req.degree() * core::mem::size_of::<f64>());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn required_bytes(self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("workspace_bytes");
        let scalars = checked_add(
            checked_add(self.quotient, self.remainder, quantity)?,
            checked_mul(self.residues, 2, quantity)?,
            quantity,
        )?;
        checked_add(
            checked_add(
                bytes_of::<C>(scalars)?,
                bytes_of::<scioxide_core::ScaledValue<C>>(self.scaled)?,
                quantity,
            )?,
            self.qr.required_bytes::<C>()?,
            quantity,
        )
    }
}

/// Holds the result of [`ApproxPartialFractionPlan::decompose`] or
/// [`ApproxPartialFractionPlan::decompose_into`].
///
/// The term numerators are packed into [`residues`](Self::residues); the
/// plan's [`terms`](ApproxPartialFractionPlan::terms) give the layout.
///
/// ```
/// use scioxide_poly::*;
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// let result = plan.decompose(&[1.0_f64])?;
/// assert!((result.residues()[0] - 1.0).abs() < 1e-12);
/// assert!((result.residues()[1] + 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ApproxPartialFractions<C: NumericScalar> {
    quotient: Vec<C>,
    residues: Vec<C>,
    report: Option<PartialFractionReport<C::Real>>,
}
impl<C: NumericScalar> Default for ApproxPartialFractions<C> {
    fn default() -> Self {
        Self::new()
    }
}
impl<C: NumericScalar> ApproxPartialFractions<C> {
    /// Creates an empty output with no result and no storage.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// let mut workspace = plan.workspace(1)?;
    /// let mut result = ApproxPartialFractions::new();
    /// result.try_reserve(&req)?;
    /// plan.decompose_into(
    ///     &[1.0_f64],
    ///     &mut result,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!((result.residues()[0] - 1.0).abs() < 1e-12);
    /// assert!((result.residues()[1] + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            quotient: Vec::new(),
            residues: Vec::new(),
            report: None,
        }
    }
    /// Reserves output storage for `requirements`, keeping the current result.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be allocated. Capacity grown
    ///   before the failure is kept.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// let mut workspace = plan.workspace(1)?;
    /// let mut result = ApproxPartialFractions::new();
    /// result.try_reserve(&req)?;
    /// plan.decompose_into(
    ///     &[1.0_f64],
    ///     &mut result,
    ///     &mut workspace,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!((result.residues()[0] - 1.0).abs() < 1e-12);
    /// assert!((result.residues()[1] + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &PartialFractionRequirements<C>,
    ) -> Result<(), SciOxideError> {
        reserve(
            &mut self.quotient,
            requirements.quotient,
            scioxide_core::quantity!("quotient_coefficients"),
        )?;
        reserve(
            &mut self.residues,
            requirements.residues,
            scioxide_core::quantity!("residue_coefficients"),
        )
    }
    /// Returns the coefficients of the polynomial part `P` in ascending order
    /// of degree, without trailing zeros except for `P = 0`.
    ///
    /// Empty before the first successful decomposition.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// assert_eq!(result.polynomial(), &[0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn polynomial(&self) -> &[C] {
        &self.quotient
    }
    /// Returns the coefficients of all term numerators `A_{i,k}`, packed in the
    /// order of the plan's [`terms`](ApproxPartialFractionPlan::terms).
    ///
    /// For linear factors these are the Laurent coefficients at each pole.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// assert!(
    ///     result
    ///         .residues()
    ///         .iter()
    ///         .zip([1.0, -1.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-12)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn residues(&self) -> &[C] {
        &self.residues
    }
    /// Returns the diagnostics, or `None` before the first successful
    /// decomposition.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let result = plan.decompose(&[1.0_f64])?;
    /// assert!(
    ///     result
    ///         .report()
    ///         .is_some_and(|report| report.reconstruction_error() < 1e-12)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn report(&self) -> Option<PartialFractionReport<C::Real>> {
        self.report
    }
}

/// Holds the scratch storage of [`ApproxPartialFractionPlan::decompose_into`].
///
/// Create it with [`ApproxPartialFractionPlan::workspace`], or with
/// [`new`](Self::new) and [`try_reserve`](Self::try_reserve). Storage only
/// grows, so a workspace reserved for the largest plan and numerator serves
/// every smaller one. It stays usable after an error.
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext};
/// use scioxide_poly::{
///     ApproxPartialFractionPlan, ApproxPartialFractions, ApproxPolynomial,
///     PartialFractionWorkspace,
/// };
/// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
/// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
/// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
/// let requirements = plan.workspace_requirements(1)?;
/// let mut workspace = PartialFractionWorkspace::new();
/// let mut output = ApproxPartialFractions::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// output.try_reserve(&requirements)?;
/// plan.decompose_into(
///     &[1.0_f64],
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!((output.residues()[0] - 1.0).abs() < 1e-14);
/// assert!((output.residues()[1] + 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PartialFractionWorkspace<C: NumericScalar> {
    quotient: Vec<C>,
    remainder: Vec<C>,
    scaled: Vec<scioxide_core::ScaledValue<C>>,
    rhs: Vec<C>,
    solution: Vec<C>,
    qr: LinalgWorkspace<'static, C>,
}
impl<C: NumericScalar> Default for PartialFractionWorkspace<C> {
    fn default() -> Self {
        Self::new()
    }
}
impl<C: NumericScalar> PartialFractionWorkspace<C> {
    /// Creates an empty workspace with no storage.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// let mut workspace = PartialFractionWorkspace::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            quotient: Vec::new(),
            remainder: Vec::new(),
            scaled: Vec::new(),
            rhs: Vec::new(),
            solution: Vec::new(),
            qr: LinalgWorkspace::new(),
        }
    }
    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count of the QR storage overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// let mut workspace = PartialFractionWorkspace::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn check(
        &self,
        requirements: &PartialFractionRequirements<C>,
    ) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.vectors_fit(requirements).is_ok()
                && self.qr.check(&requirements.qr)? == Capacity::Sufficient
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to at least `requirements`, keeping any larger
    /// storage.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be allocated. Storage grown
    ///   before the failure is kept.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let x = ApproxPolynomial::monomial(vec![0.0_f64, 1.0])?;
    /// let x1 = ApproxPolynomial::monomial(vec![1.0_f64, 1.0])?;
    /// let plan = ApproxPartialFractionPlan::new(&x.try_mul(&x1)?, &[(&x, 1), (&x1, 1)])?;
    /// let req = plan.workspace_requirements(1)?;
    /// let mut workspace = PartialFractionWorkspace::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &PartialFractionRequirements<C>,
    ) -> Result<(), SciOxideError> {
        try_grow(
            &mut self.quotient,
            requirements.quotient,
            C::ZERO,
            scioxide_core::quantity!("quotient_coefficients"),
        )?;
        try_grow(
            &mut self.remainder,
            requirements.remainder,
            C::ZERO,
            scioxide_core::quantity!("remainder_coefficients"),
        )?;
        try_grow(
            &mut self.scaled,
            requirements.scaled,
            scioxide_core::ScaledValue::ZERO,
            scioxide_core::quantity!("scaled_division_scratch"),
        )?;
        try_grow(
            &mut self.rhs,
            requirements.residues,
            C::ZERO,
            scioxide_core::quantity!("partial_fraction_rhs"),
        )?;
        try_grow(
            &mut self.solution,
            requirements.residues,
            C::ZERO,
            scioxide_core::quantity!("partial_fraction_solution"),
        )?;
        self.qr.try_reserve(&requirements.qr)
    }
    /// Checks the workspace before execution. The QR solve would grow an
    /// undersized arena, so it is rejected here, in bytes, before any numerical
    /// work.
    fn fits(&self, requirements: &PartialFractionRequirements<C>) -> Result<(), SciOxideError> {
        self.vectors_fit(requirements)?;
        if self.qr.check(&requirements.qr)? == Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                requirements.qr.required_bytes::<C>()?,
                self.qr.retained_bytes()?,
            ));
        }
        Ok(())
    }
    fn vectors_fit(
        &self,
        requirements: &PartialFractionRequirements<C>,
    ) -> Result<(), SciOxideError> {
        minimum_len(requirements.quotient, self.quotient.len())?;
        minimum_len(requirements.remainder, self.remainder.len())?;
        minimum_len(requirements.scaled, self.scaled.len())?;
        minimum_len(requirements.residues, self.rhs.len())?;
        minimum_len(requirements.residues, self.solution.len())
    }
}
fn reserve<C>(
    values: &mut Vec<C>,
    capacity: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    scioxide_core::backend::try_reserve(values, capacity.saturating_sub(values.len()), quantity)
}
