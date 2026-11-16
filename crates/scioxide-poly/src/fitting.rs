//! Weighted polynomial least squares by a truncated SVD of the column-scaled design.
use crate::util::{bytes_of, coefficient_scale, invalid, minimum_len, try_grow, validate_slice};
use crate::{
    ApproxPolynomial, MonomialBasis, OrthogonalBasis, OrthogonalBasisPlan, OrthogonalBasisSpec,
    PolynomialApproximationPolicy, PolynomialBasis, PolynomialExecutionOptions, Vandermonde,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, standard, try_filled};
use scioxide_core::{
    AlgebraicScalar, Capacity, ExecutionContext, InvalidInputReason, NumericScalar, RealScalar,
    Scaled, ScaledNorm, ScaledValue, SciOxideError,
};
use scioxide_linalg::{
    ConditionEstimate, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix,
    RankReport, SvdWorkspaceRequest, ThinSvdDecomposition, VectorViewMut,
};

mod arithmetic;
use arithmetic::{
    extended_column, extended_residual, extended_rhs, native_columns, native_product,
    native_residual, native_rhs,
};

/// Reports the numerical rank, conditioning and weighted residual of a
/// polynomial least-squares fit.
///
/// Rank and condition refer to the design matrix after row weighting and
/// column scaling, as described on [`PolynomialFitPlan`]. The rank is the
/// number of singular values kept under the policy's rank policy.
///
/// ```
/// use scioxide_poly::PolynomialFitPlan;
///
/// // The line y = 1 + 2x through three points fits exactly.
/// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
/// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
/// let report = result.report();
/// assert_eq!(report.rank().rank(), 2);
/// assert!(report.residual_norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PolynomialFitReport<R: RealScalar> {
    rank: RankReport<R>,
    condition: ConditionEstimate<R>,
    residual_norm: R,
    policy: PolynomialApproximationPolicy<R>,
}
impl<R: RealScalar> PolynomialFitReport<R> {
    /// Returns the numerical rank of the scaled design and the threshold that
    /// determined it.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// let report = result.report();
    /// assert_eq!(report.rank().rank(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn rank(self) -> RankReport<R> {
        self.rank
    }
    /// Returns the 2-norm condition number of the scaled design, `σ_max/σ_min`
    /// over all singular values, as a reciprocal.
    ///
    /// It does not depend on the rank threshold.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// let report = result.report();
    /// assert!(report.condition().reciprocal().value() > 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn condition(self) -> ConditionEstimate<R> {
        self.condition
    }
    /// Returns the weighted residual norm `(Σᵢ wᵢ |p(xᵢ) − yᵢ|²)^(1/2)` of the
    /// fitted polynomial `p`, in the units of the sample values `yᵢ`.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// let report = result.report();
    /// assert!(report.residual_norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn residual_norm(self) -> R {
        self.residual_norm
    }
    /// Returns the approximation policy of the fit, including its rank policy.
    ///
    /// ```
    /// use scioxide_poly::{PolynomialApproximationPolicy, PolynomialFitPlan};
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// let report = result.report();
    /// assert_eq!(report.policy(), PolynomialApproximationPolicy::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn policy(self) -> PolynomialApproximationPolicy<R> {
        self.policy
    }
}

/// Holds a fitted polynomial in the basis of its [`PolynomialFitPlan`] and the
/// [`PolynomialFitReport`] of the fit.
///
/// ```
/// use scioxide_poly::PolynomialFitPlan;
///
/// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
/// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
/// assert_eq!(result.report().rank().rank(), 2);
/// assert!(result.report().residual_norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PolynomialFit<C: NumericScalar, B: PolynomialBasis<C>> {
    polynomial: ApproxPolynomial<C, B>,
    report: PolynomialFitReport<C::Real>,
}
impl<C: NumericScalar, B: PolynomialBasis<C>> PolynomialFit<C, B> {
    /// Returns the fitted polynomial.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// // p(x) = 1 + 2x, so p(0.5) = 2.
    /// assert!((result.polynomial().evaluate(0.5)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn polynomial(&self) -> &ApproxPolynomial<C, B> {
        &self.polynomial
    }
    /// Returns the rank, condition and residual report of the fit.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// assert_eq!(result.report().rank().rank(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn report(&self) -> PolynomialFitReport<C::Real> {
        self.report
    }
    /// Returns the fitted polynomial, consuming `self`.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let p = plan.fit(&[-1.0_f64, 1.0, 3.0])?.into_polynomial();
    /// assert!((p.evaluate(0.5)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn into_polynomial(self) -> ApproxPolynomial<C, B> {
        self.polynomial
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, MonomialBasis<C>> {
    /// Fits a polynomial of degree at most `degree` in powers of `x` to the
    /// samples `(pointsᵢ, valuesᵢ)` by linear least squares.
    ///
    /// The result minimizes `Σᵢ |p(xᵢ) − yᵢ|²` with the method of
    /// [`PolynomialFitPlan`]: a thin singular value decomposition of the
    /// column-scaled Vandermonde matrix, in O(m n · min(m, n)) time for `m`
    /// points and `n = degree + 1` coefficients. Singular values below the rank
    /// threshold of the default policy are dropped, so a rank-deficient or
    /// underdetermined problem gives the minimum-norm solution. The polynomial
    /// is the one `PolynomialFitPlan::monomial(points, degree, None,
    /// PolynomialApproximationPolicy::default())?.fit(values)` returns; use
    /// that plan for weights, another policy, the fit report or repeated fits.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `points` is empty.
    /// - `DimensionMismatch` if `values` and `points` differ in length.
    /// - `NonFinite` if a point or value is NaN or infinite.
    /// - `NumericalInstability` if a Vandermonde entry or a coefficient
    ///   overflows.
    /// - The errors of the thin SVD in `scioxide-linalg`, such as
    ///   `ConvergenceFailure`.
    /// - `SizeOverflow` or `AllocationFailure` if the factors cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// // The data lie on y = 1 + 2x, so the least-squares line is exact.
    /// let p = ApproxPolynomial::fit_monomial(&[-1.0_f64, 0.0, 1.0], &[-1.0, 1.0, 3.0], 1)?;
    /// assert!((p.evaluate(0.5)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn fit_monomial(points: &[C], values: &[C], degree: usize) -> Result<Self, SciOxideError> {
        PolynomialFitPlan::monomial(
            points,
            degree,
            None,
            PolynomialApproximationPolicy::default(),
        )?
        .fit(values)
        .map(PolynomialFit::into_polynomial)
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, OrthogonalBasis<C>> {
    /// Fits an expansion of degree at most `degree` in the orthogonal basis of
    /// `spec` to the samples `(pointsᵢ, valuesᵢ)` by linear least squares.
    ///
    /// The method, cost and treatment of rank deficiency are those of
    /// [`fit_monomial`](ApproxPolynomial::fit_monomial), with the orthogonal
    /// basis values in place of powers, which keeps the design matrix well
    /// conditioned for points spread over the interval of `spec`. The basis is
    /// prepared for `degree` with default execution options, and the polynomial
    /// is the one [`PolynomialFitPlan::orthogonal`] with the default policy
    /// returns; use that plan for weights, another policy, the fit report or
    /// repeated fits.
    ///
    /// # Errors
    ///
    /// The errors of [`fit_monomial`](ApproxPolynomial::fit_monomial) and of
    /// [`OrthogonalBasisSpec::prepare`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// // A Chebyshev line on [−1, 1] through (−1, −1), (0, 1), (1, 3): y = 1 + 2x.
    /// let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::ChebyshevT, -1.0_f64, 1.0)?;
    /// let p = ApproxPolynomial::fit_orthogonal(&[-1.0_f64, 0.0, 1.0], &[-1.0, 1.0, 3.0], 1, spec)?;
    /// assert!((p.evaluate(0.5)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn fit_orthogonal(
        points: &[C],
        values: &[C],
        degree: usize,
        spec: OrthogonalBasisSpec<C::Real>,
    ) -> Result<Self, SciOxideError> {
        PolynomialFitPlan::orthogonal(
            points,
            spec.prepare(degree, PolynomialExecutionOptions::new())?,
            None,
            PolynomialApproximationPolicy::default(),
        )?
        .fit(values)
        .map(PolynomialFit::into_polynomial)
    }
}

/// Fits polynomials of a fixed degree to samples at fixed points by weighted
/// linear least squares.
///
/// For points `xᵢ`, values `yᵢ` and weights `wᵢ ≥ 0`, the fit returns the
/// coefficients `c` of `p(x) = Σⱼ cⱼ Qⱼ(x)` that minimize
///
/// ```text
/// Σᵢ wᵢ |p(xᵢ) − yᵢ|²,
/// ```
///
/// where `Q₀, …, Q_d` are the monomials ([`monomial`](Self::monomial)) or the
/// members of a prepared orthogonal basis ([`orthogonal`](Self::orthogonal)).
/// The weights multiply squared residuals; to weight the residuals themselves
/// by `σᵢ`, pass `wᵢ = σᵢ²`. `None` means unit weights. A zero weight removes
/// its point from the objective.
///
/// Preparation forms the Vandermonde matrix `V` with [`Vandermonde`],
/// multiplies row `i` by `√wᵢ`, scales each column to unit 2-norm and computes
/// a thin singular value decomposition. This costs O(m n · min(m, n)) time for
/// `m` points and `n = d + 1` coefficients. Each fit then costs O((m + n) n):
/// singular values below the threshold of the policy's rank policy are
/// dropped, and the result is the minimum-norm solution in the remaining
/// subspace. This handles
/// rank-deficient and underdetermined problems. Normal equations are never
/// formed, and orthogonal bases are never expanded into powers. Row weighting
/// and column norms are computed with a rescaling that avoids overflow and
/// underflow.
///
/// # Examples
///
/// ```
/// use scioxide_poly::{PolynomialBasis, PolynomialFitPlan};
///
/// // The line y = 1 + 2x through three points fits exactly.
/// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
/// let fit = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
/// let c = fit.polynomial().basis().coefficients();
/// assert!((c[0] - 1.0).abs() < 1e-13 && (c[1] - 2.0).abs() < 1e-13);
/// assert_eq!(fit.report().rank().rank(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Å. Björck, *Numerical Methods for Least Squares Problems*, SIAM, 1996.
#[derive(Debug, Clone)]
pub struct PolynomialFitPlan<C: NumericScalar, B = ()> {
    design: Matrix<C>,
    roots: Vec<C::Real>,
    scales: Vec<Scaled<C::Real>>,
    factor: ThinSvdDecomposition<C>,
    rank: RankReport<C::Real>,
    condition: ConditionEstimate<C::Real>,
    policy: PolynomialApproximationPolicy<C::Real>,
    basis: B,
}

impl<C: NumericScalar> PolynomialFitPlan<C> {
    /// Prepares a fit by a polynomial of degree at most `degree` in powers of
    /// `x`, at `points` with optional `weights`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `points` is empty, a weight is negative, or all
    ///   weights are zero.
    /// - `DimensionMismatch` if `weights` and `points` differ in length.
    /// - `NonFinite` if a point or weight is NaN or infinite.
    /// - `NumericalInstability` if a Vandermonde entry overflows, or a policy
    ///   threshold is not representable.
    /// - The errors of the thin SVD in `scioxide-linalg`, such as
    ///   `ConvergenceFailure`.
    /// - `SizeOverflow` or `AllocationFailure` if the factors cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// assert_eq!(result.report().rank().rank(), 2);
    /// assert!(result.report().residual_norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn monomial(
        points: &[C],
        degree: usize,
        weights: Option<&[C::Real]>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::prepare(points, Vandermonde::monomial(degree)?, weights, policy, ())
    }
    /// Fits the sample `values` and returns the polynomial in the monomial
    /// basis with its report.
    ///
    /// This allocates a workspace and calls [`fit_into`](Self::fit_into).
    ///
    /// # Errors
    ///
    /// The errors of [`fit_into`](Self::fit_into), and `AllocationFailure` if
    /// the workspace or result cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let result = plan.fit(&[-1.0_f64, 1.0, 3.0])?;
    /// assert_eq!(result.report().rank().rank(), 2);
    /// assert!(result.report().residual_norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn fit(&self, values: &[C]) -> Result<PolynomialFit<C, MonomialBasis<C>>, SciOxideError> {
        let (coefficients, report) = self.fit_coefficients(values)?;
        Ok(PolynomialFit {
            polynomial: ApproxPolynomial::monomial_with_options(coefficients, self.policy)?,
            report,
        })
    }
}
impl<C: NumericScalar> PolynomialFitPlan<C, OrthogonalBasisPlan<C::Real>> {
    /// Prepares a fit by an expansion in the orthogonal basis of `basis`, using
    /// all of its members up to `basis.degree()`, at `points` with optional
    /// `weights`.
    ///
    /// Every fitted polynomial carries a copy of `basis`, with its family,
    /// normalization, affine map and execution options.
    ///
    /// # Errors
    ///
    /// The errors of [`monomial`](PolynomialFitPlan::monomial), with
    /// `NumericalInstability` also for a basis value that overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily, PolynomialFitPlan};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let basis = spec.prepare(2, Default::default())?;
    /// let points = [-1.0_f64, 0.0, 1.0];
    /// let plan = PolynomialFitPlan::orthogonal(&points, basis, None, Default::default())?;
    /// // The quadratic through (−1, 2), (0, −1), (1, 2) is 3x² − 1, so p(0) = −1.
    /// let result = plan.fit(&[2.0_f64, -1.0, 2.0])?;
    /// assert_eq!(result.report().rank().rank(), 3);
    /// assert!((result.polynomial().evaluate(0.0)? + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn orthogonal(
        points: &[C],
        basis: OrthogonalBasisPlan<C::Real>,
        weights: Option<&[C::Real]>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        let retained = basis.resized(basis.degree())?;
        Self::prepare(
            points,
            Vandermonde::orthogonal(basis),
            weights,
            policy,
            retained,
        )
    }
    /// Fits the sample `values` and returns the expansion in the prepared
    /// orthogonal basis with its report.
    ///
    /// This allocates a workspace and calls [`fit_into`](Self::fit_into).
    ///
    /// # Errors
    ///
    /// The errors of [`fit_into`](Self::fit_into), and `AllocationFailure` if
    /// the workspace or result cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily, PolynomialFitPlan};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let basis = spec.prepare(2, Default::default())?;
    /// let points = [-1.0_f64, 0.0, 1.0];
    /// let plan = PolynomialFitPlan::orthogonal(&points, basis, None, Default::default())?;
    /// // The quadratic through (−1, 2), (0, −1), (1, 2) is 3x² − 1, so p(0) = −1.
    /// let result = plan.fit(&[2.0_f64, -1.0, 2.0])?;
    /// assert!((result.polynomial().evaluate(0.0)? + 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn fit(&self, values: &[C]) -> Result<PolynomialFit<C, OrthogonalBasis<C>>, SciOxideError> {
        let (coefficients, report) = self.fit_coefficients(values)?;
        Ok(PolynomialFit {
            polynomial: ApproxPolynomial::orthogonal_with_plan(
                coefficients,
                self.basis.resized(self.basis.degree())?,
                self.policy,
            )?,
            report,
        })
    }
}
impl<C: NumericScalar, B> PolynomialFitPlan<C, B> {
    fn prepare(
        points: &[C],
        matrix: Vandermonde<C::Real>,
        weights: Option<&[C::Real]>,
        policy: PolynomialApproximationPolicy<C::Real>,
        basis: B,
    ) -> Result<Self, SciOxideError> {
        if points.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("points"),
                InvalidInputReason::Empty,
                "at least one observation is required",
            ));
        }
        if let Some(w) = weights {
            same_len(points.len(), w.len())?;
            if w.iter().any(|x| !x.is_finite()) {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "weights must be finite",
                ));
            }
            if w.iter().any(|&x| x < C::Real::ZERO) {
                return Err(invalid(
                    scioxide_core::argument!("weights"),
                    InvalidInputReason::OutOfRange,
                    "weights must be nonnegative",
                ));
            }
            if w.iter().all(|&x| x == C::Real::ZERO) {
                return Err(invalid(
                    scioxide_core::argument!("weights"),
                    InvalidInputReason::OutOfRange,
                    "at least one weight must be positive",
                ));
            }
        }
        // Validate policy thresholds even when the design is rank deficient.
        validate_thresholds(policy, C::Real::ONE, 1)?;
        let design = matrix.matrix(points)?;
        let mut weighted = Matrix::zeros(design.rows(), design.cols())?;
        let roots = crate::util::try_collect(
            (0..points.len()).map(|i| Ok(weights.map_or(C::Real::ONE, |w| w[i]).sqrt())),
            scioxide_core::quantity!("fit_row_weights"),
        )?;
        let mut scales = try_filled(
            design.cols(),
            Scaled::ONE,
            scioxide_core::quantity!("fit_column_scales"),
        )?;
        if !native_columns(&design, &roots, &mut weighted, &mut scales)? {
            for (column, scale) in scales.iter_mut().enumerate() {
                *scale = extended_column(&design, &roots, column, &mut weighted)?;
            }
        }
        let factor = ThinSvdDecomposition::compute(&weighted)?;
        let rank = factor.analyze_rank(policy.rank_policy())?;
        let condition = factor.condition_estimate()?;
        Ok(Self {
            design,
            roots,
            scales,
            factor,
            rank,
            condition,
            policy,
            basis,
        })
    }
    /// Returns the workspace size that [`fit_into`](Self::fit_into) needs.
    ///
    /// The request is `()` because the plan fixes all dimensions.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// assert!(plan.workspace_requirements(())?.required_bytes::<f64>()? > 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<PolynomialFitRequirements, SciOxideError> {
        let solve =
            ThinSvdDecomposition::<C>::workspace_requirements(&SvdWorkspaceRequest::solve(
                self.design.rows(),
                self.design.cols(),
                &LinalgExecutionOptions::new(),
            ))?;
        let req = PolynomialFitRequirements {
            rows: self.design.rows(),
            cols: self.design.cols(),
            solve,
        };
        req.required_bytes::<C>()?;
        Ok(req)
    }
    /// Allocates a workspace large enough for [`fit_into`](Self::fit_into).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the workspace cannot be
    ///   allocated.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let workspace = plan.workspace(())?;
    /// assert_eq!(workspace.check(&plan.workspace_requirements(())?)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(&self, (): ()) -> Result<PolynomialFitWorkspace<C>, SciOxideError> {
        let mut workspace = PolynomialFitWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(())?)?;
        Ok(workspace)
    }
    fn fit_coefficients(
        &self,
        values: &[C],
    ) -> Result<(Vec<C>, PolynomialFitReport<C::Real>), SciOxideError> {
        let mut coefficients = try_filled(
            self.design.cols(),
            C::ZERO,
            scioxide_core::quantity!("fit_coefficients"),
        )?;
        let report = self.fit_into(
            values,
            &mut coefficients,
            &mut self.workspace(())?,
            ExecutionContext::sequential_native(),
        )?;
        Ok((coefficients, report))
    }
    /// Fits the sample `values` and writes the coefficients, in ascending
    /// degree, to `coefficients`.
    ///
    /// `values` has one entry per prepared point, and `coefficients` has
    /// `degree + 1` entries. Singular values below the rank threshold are
    /// dropped; under `RankPolicy::Exact` only exact zeros are. No
    /// regularization is applied. With a workspace from
    /// [`workspace`](Self::workspace), the call does not allocate. On error
    /// `coefficients` is unchanged; see
    /// [Output on error](scioxide_core#output-on-error).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values` or `coefficients` has the wrong length,
    ///   or the workspace is too small.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `NumericalInstability` if a coefficient or the residual norm overflows,
    ///   or a policy threshold at the scale of the coefficients is not
    ///   representable.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let mut coefficients = [0.0_f64; 2];
    /// let mut workspace = plan.workspace(())?;
    /// let cx = ExecutionContext::sequential();
    /// let report = plan.fit_into(&[-1.0_f64, 1.0, 3.0], &mut coefficients, &mut workspace, cx)?;
    /// // y = 1 + 2x.
    /// assert!(coefficients.iter().zip([1.0_f64, 2.0]).all(|(a, b)| (a - b).abs() < 1e-12));
    /// assert!(report.residual_norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn fit_into(
        &self,
        values: &[C],
        coefficients: &mut [C],
        workspace: &mut PolynomialFitWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<PolynomialFitReport<C::Real>, SciOxideError> {
        context.check()?;
        same_len(self.design.rows(), values.len())?;
        same_len(self.design.cols(), coefficients.len())?;
        let requirements = self.workspace_requirements(())?;
        workspace.fits(&requirements)?;
        validate_slice(values)?;
        let rhs = &mut workspace.rhs[..values.len()];
        let rhs_scale = match native_rhs(values, &self.roots, rhs) {
            Some(scale) => Scaled::from_value(scale)?,
            None => extended_rhs(values, &self.roots, rhs)?,
        };
        let candidate = &mut workspace.candidate[..coefficients.len()];
        self.factor.solve_into(
            &rhs[..],
            self.policy.rank_policy(),
            C::Real::ZERO,
            &LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(candidate),
            &mut workspace.solve,
            context,
        )?;
        for (value, &scale) in candidate.iter_mut().zip(&self.scales) {
            let ratio = rhs_scale.div(scale)?;
            *value = match ratio
                .round()
                .ok()
                .filter(|ratio| *ratio >= C::Real::MIN_POSITIVE)
                .and_then(|ratio| native_product(*value, ratio))
            {
                Some(value) => value,
                _ => ScaledValue::from_value(*value)?.mul_real(ratio)?.round()?,
            };
        }
        validate_thresholds(self.policy, coefficient_scale(candidate), candidate.len())?;
        let residual_norm =
            match native_residual(&self.design, values, &self.roots, candidate, context)? {
                Some(norm) => norm,
                None => extended_residual(&self.design, values, &self.roots, candidate, context)?,
            };
        let report = PolynomialFitReport {
            rank: self.rank,
            condition: self.condition,
            residual_norm,
            policy: self.policy,
        };
        context.check()?;
        coefficients.copy_from_slice(candidate);
        Ok(report)
    }
}

/// Describes the workspace that a [`PolynomialFitPlan`] needs.
///
/// ```
/// use scioxide_poly::PolynomialFitPlan;
///
/// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
/// let req = plan.workspace_requirements(())?;
/// assert!(req.required_bytes::<f64>()? > 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialFitRequirements {
    rows: usize,
    cols: usize,
    solve: LinalgRequirements,
}
impl PolynomialFitRequirements {
    /// Returns the number of bytes of scalar storage for scalar type `C`, not
    /// counting the headers of the vectors that hold it.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the total overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::PolynomialFitPlan;
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// assert!(plan.workspace_requirements(())?.required_bytes::<f64>()? > 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn required_bytes<C: NumericScalar>(&self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("fit_workspace_bytes");
        checked_add(
            bytes_of::<C>(checked_add(self.rows, self.cols, quantity)?)?,
            self.solve.required_bytes::<C>()?,
            quantity,
        )
    }
}

/// Holds the scratch storage of [`PolynomialFitPlan::fit_into`].
///
/// The workspace grows on request and never shrinks. Once reserved for a plan's
/// requirements, it serves repeated fits without allocation; see
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_poly::{PolynomialFitPlan, PolynomialFitWorkspace};
///
/// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
/// let req = plan.workspace_requirements(())?;
/// let mut workspace = PolynomialFitWorkspace::<f64>::new();
/// workspace.try_reserve(&req)?;
/// assert_eq!(workspace.check(&req)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFitWorkspace<C: NumericScalar> {
    rhs: Vec<C>,
    candidate: Vec<C>,
    solve: LinalgWorkspace<'static, C>,
}
impl<C: NumericScalar> Default for PolynomialFitWorkspace<C> {
    fn default() -> Self {
        Self::new()
    }
}
impl<C: NumericScalar> PolynomialFitWorkspace<C> {
    /// Creates an empty workspace without allocating.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialFitPlan, PolynomialFitWorkspace};
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let req = plan.workspace_requirements(())?;
    /// let mut workspace = PolynomialFitWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            rhs: Vec::new(),
            candidate: Vec::new(),
            solve: LinalgWorkspace::new(),
        }
    }
    /// Reports whether the workspace already satisfies `requirements`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the required size overflows `usize`.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialFitPlan, PolynomialFitWorkspace};
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let req = plan.workspace_requirements(())?;
    /// let mut workspace = PolynomialFitWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn check(
        &self,
        requirements: &PolynomialFitRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.rhs.len() >= requirements.rows
                && self.candidate.len() >= requirements.cols
                && self.solve.check(&requirements.solve)? == Capacity::Sufficient
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to satisfy `requirements`.
    ///
    /// On failure, storage that has already grown is kept.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot grow.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_poly::{PolynomialFitPlan, PolynomialFitWorkspace};
    ///
    /// let plan = PolynomialFitPlan::monomial(&[-1.0_f64, 0.0, 1.0], 1, None, Default::default())?;
    /// let req = plan.workspace_requirements(())?;
    /// let mut workspace = PolynomialFitWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &PolynomialFitRequirements,
    ) -> Result<(), SciOxideError> {
        try_grow(
            &mut self.rhs,
            requirements.rows,
            C::ZERO,
            scioxide_core::quantity!("fit_rhs"),
        )?;
        try_grow(
            &mut self.candidate,
            requirements.cols,
            C::ZERO,
            scioxide_core::quantity!("fit_candidate"),
        )?;
        self.solve.try_reserve(&requirements.solve)
    }
    fn fits(&self, req: &PolynomialFitRequirements) -> Result<(), SciOxideError> {
        minimum_len(req.rows, self.rhs.len())?;
        minimum_len(req.cols, self.candidate.len())?;
        if self.solve.check(&req.solve)? == Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                req.solve.required_bytes::<C>()?,
                self.solve.retained_bytes()?,
            ));
        }
        Ok(())
    }
}
fn same_len(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(expected, actual))
    }
}
/// Checks the policy thresholds without building a polynomial report.
fn validate_thresholds<R: RealScalar>(
    policy: PolynomialApproximationPolicy<R>,
    scale: R,
    dimension: usize,
) -> Result<(), SciOxideError> {
    policy.tolerance().threshold(scale)?;
    crate::basis::rank_threshold(policy.rank_policy(), scale, dimension)?;
    Ok(())
}
