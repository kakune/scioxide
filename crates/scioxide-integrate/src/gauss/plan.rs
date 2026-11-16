//! Gauss, Gauss–Radau and Gauss–Lobatto rules for classical weight functions.

use alloc::sync::Arc;

use scioxide_core::backend::{checked_mul, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Interval, NumericScalar, RealScalar, RelativeTolerance, Scaled,
    ScaledAccumulator, SciOxideError, SolveScope, Tolerance,
};
use scioxide_linalg::{
    HermitianEigenFirstComponents, HermitianEigenOptions, LinalgExecutionOptions,
};
use scioxide_poly::OrthogonalFamily;

use crate::fixed::{FixedRule, FixedRuleNode, FixedRuleWorkspace, Geometry, apply};
use crate::kernel::PreparedRule;
use crate::report::IntegrationReport;

/// Selects which endpoints of `[−1, 1]` an `n`-node Gaussian rule includes as
/// nodes.
///
/// Each prescribed node lowers the degree of exactness by one. Only the
/// Legendre and Jacobi families support prescribed nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianRuleVariant {
    /// Gauss rule: no prescribed node, exact for degree `2n − 1`.
    Gauss,
    /// Gauss–Radau rule with a node at −1, exact for degree `2n − 2`.
    RadauLeft,
    /// Gauss–Radau rule with a node at +1, exact for degree `2n − 2`.
    RadauRight,
    /// Gauss–Lobatto rule with nodes at −1 and +1, exact for degree `2n − 3`;
    /// needs `n ≥ 2`.
    Lobatto,
}

/// Domain on which a classical weight function and its reference rule are
/// defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianDomain {
    /// `[−1, 1]`, for the Legendre, Jacobi, Gegenbauer and Chebyshev families.
    UnitInterval,
    /// `[0, ∞)`, for the Laguerre families.
    NonnegativeRealLine,
    /// `(−∞, ∞)`, for the Hermite families.
    RealLine,
}

/// Options for constructing a [`GaussianRulePlan`]: the deflation tolerance and
/// the limits of the eigenvalue computation.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, Tolerance};
/// use scioxide_integrate::GaussianRuleOptions;
/// use scioxide_linalg::LinalgExecutionOptions;
///
/// let options = GaussianRuleOptions::<f64>::default()
///     .with_deflation_tolerance(Tolerance::from_machine_epsilon())
///     .with_execution_options(LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE));
/// assert_eq!(
///     options.execution_options().iteration_budget(),
///     Some(Budget::ONE)
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct GaussianRuleOptions<S: RealScalar> {
    tolerance: Tolerance<S>,
    execution: LinalgExecutionOptions,
}

impl<S: RealScalar> Default for GaussianRuleOptions<S> {
    /// Returns a relative deflation tolerance of machine epsilon with no
    /// absolute part, and default execution options, which allow at most
    /// `30 max(n, 1)` QL sweeps.
    fn default() -> Self {
        Self {
            tolerance: RelativeTolerance::<S>::default().into(),
            execution: LinalgExecutionOptions::new(),
        }
    }
}

impl<S: RealScalar> GaussianRuleOptions<S> {
    /// Sets the deflation tolerance of the QL iteration, relative to the scaled
    /// Jacobi matrix.
    ///
    /// It is not an error bound on the nodes or weights.
    pub const fn with_deflation_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns the deflation tolerance.
    pub const fn deflation_tolerance(&self) -> Tolerance<S> {
        self.tolerance
    }

    /// Sets the execution options, such as the iteration budget and byte limit,
    /// of the eigenvalue computation.
    ///
    /// Construction runs sequentially without a clock, so a time limit makes
    /// [`GaussianRulePlan::new`] fail with `CapabilityUnavailable`.
    pub const fn with_execution_options(mut self, execution: LinalgExecutionOptions) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the execution options of the eigenvalue computation.
    pub const fn execution_options(&self) -> LinalgExecutionOptions {
        self.execution
    }
}

/// Integration domain of a weighted Gaussian rule.
///
/// The support must match the [`GaussianDomain`] of the family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaussianSupport<S: RealScalar> {
    /// Finite interval `[a, b]` for a family on `[−1, 1]`, reached by the
    /// affine map `x = c + h t` with midpoint `c` and half-width `h`.
    ///
    /// The rule computes `∫ₐᵇ f(x) w(t(x)) dx`, so the weights are scaled by
    /// `h`. The Chebyshev T family is the exception: it computes
    /// `∫ₐᵇ f(x) / √((x − a)(b − x)) dx`, whose weights are not scaled.
    Interval(Interval<S>),
    /// Half-line `[c, ∞)` for a Laguerre family, with weight
    /// `(x − c)^α e^(−(x − c))`. The nodes are shifted by `c`; the weights are
    /// unchanged.
    HalfLine(S),
    /// The real line for a Hermite family; nodes and weights are unchanged.
    RealLine,
}

/// Gaussian quadrature for a classical weight function, with optional
/// prescribed endpoints.
///
/// An `n`-node rule approximates
///
/// ```text
/// ∫ f(t) w(t) dt ≈ Σᵢ wᵢ f(tᵢ),   i = 0, …, n − 1,
/// ```
///
/// where `w` is the weight function of the [`OrthogonalFamily`] on its
/// [`GaussianDomain`], `tᵢ` are the nodes and `wᵢ` the positive weights. The
/// weights sum to the zeroth moment `μ₀ = ∫ w(t) dt`; they are not normalized
/// to a probability measure. The rule is exact for polynomials `f` of degree
/// [`Self::exactness_degree`]. It has no error estimate.
///
/// Construction uses the Golub–Welsch algorithm. The nodes are the eigenvalues
/// of the symmetric tridiagonal Jacobi matrix of the three-term recurrence, and
/// each weight is `μ₀ v₀²`, where `v₀` is the first component of the normalized
/// eigenvector. Implicit QL iteration that tracks only these components costs
/// O(n²) operations and O(n) memory. For Gauss–Radau and Gauss–Lobatto rules,
/// the last row of the Jacobi matrix is modified so that the prescribed
/// endpoints are eigenvalues (Golub, 1973). For the Hermite families, the
/// weights are computed from the recurrence at each node instead, so the small
/// weights of the outer nodes are not lost to deflation. For symmetric weight
/// functions, the nodes and weights are made exactly symmetric. All computation
/// runs in `S`.
///
/// Integrate with the weight function through [`Self::integrate_weighted`] or
/// [`Self::integrate_weighted_into`], on a [`GaussianSupport`] that matches the
/// family. Families whose weight is identically one (Legendre, Jacobi(0, 0)
/// and Gegenbauer(1/2)) also integrate `∫ₐᵇ f(x) dx` through [`FixedRule`].
/// Both forms take their workspace from [`FixedRule::workspace`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{FixedRule, GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant};
/// use scioxide_poly::OrthogonalFamily;
///
/// // A 4-node Gauss–Legendre rule is exact for x²: ∫₀¹ x² dx = 1/3.
/// let rule = GaussianRulePlan::<f64>::gauss_legendre(4)?;
/// let (value, report) = rule.integrate(|x: f64| Ok(x * x), Interval::new(0.0, 1.0)?)?;
/// assert!(report.is_success());
/// assert!((value - 1.0 / 3.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules",
///   *Math. Comp.* 23, 221–230, 1969.
///   <https://doi.org/10.1090/S0025-5718-69-99647-1>
/// - G. H. Golub, "Some modified matrix eigenvalue problems", *SIAM Rev.* 15,
///   318–334, 1973. <https://doi.org/10.1137/1015032>
/// - NIST DLMF, §3.5(v), Gauss quadrature. <https://dlmf.nist.gov/3.5.v>
#[derive(Debug, Clone)]
#[must_use]
pub struct GaussianRulePlan<S: RealScalar> {
    family: OrthogonalFamily<S>,
    variant: GaussianRuleVariant,
    mass: S,
    degree: usize,
    rule: Arc<PreparedRule<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> GaussianRulePlan<S> {
    /// Computes the `node_count`-node rule of the given variant for `family`.
    ///
    /// A prescribed Radau or Lobatto endpoint must agree with its computed
    /// eigenvalue to within `8 n max(τ, ε) ‖T‖∞`, where `τ` is the deflation
    /// tolerance and `T` the modified Jacobi matrix; it is then set exactly to
    /// ±1.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `node_count`) if `node_count` is zero, or
    ///   less than 2 for [`GaussianRuleVariant::Lobatto`].
    /// - `InvalidInput` (argument `family`) if a Radau or Lobatto rule is
    ///   requested for a family other than Legendre and Jacobi.
    /// - `DomainError` or `NonFinite` if the family parameters are outside the
    ///   domain of the weight function, for example a Jacobi exponent `≤ −1`.
    /// - `ConvergenceFailure` if the QL iteration exhausts its iteration budget.
    /// - `CapabilityUnavailable` if `options` sets a time limit.
    /// - `ResourceLimit` if the eigenvalue workspace exceeds the byte limit of
    ///   `options`.
    /// - `NumericalInstability` if `μ₀` or a weight is not positive and finite in
    ///   `S`, the nodes are not strictly ascending in the domain, or a prescribed
    ///   endpoint misses its eigenvalue.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_integrate::{GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant};
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // The 3-node Gauss–Lobatto–Legendre rule is Simpson's rule on [−1, 1].
    /// let plan = GaussianRulePlan::<f64>::with_options(OrthogonalFamily::Legendre, 3, GaussianRuleVariant::Lobatto, GaussianRuleOptions::default())?;
    /// assert_eq!(plan.nodes(), [-1.0, 0.0, 1.0]);
    /// assert_eq!(plan.exactness_degree(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_options(
        family: OrthogonalFamily<S>,
        node_count: usize,
        variant: GaussianRuleVariant,
        options: GaussianRuleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let options = HermitianEigenOptions::default()
            .with_deflation_tolerance(options.tolerance)
            .with_execution_options(options.execution);
        if node_count == 0 || (variant == GaussianRuleVariant::Lobatto && node_count < 2) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("node_count"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Gauss/Radau require n >= 1; Lobatto requires n >= 2",
            ));
        }
        if variant != GaussianRuleVariant::Gauss
            && !matches!(
                family,
                OrthogonalFamily::Legendre | OrthogonalFamily::Jacobi { .. }
            )
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("family"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Radau/Lobatto require the Legendre or Jacobi family",
            ));
        }
        let degree = checked_mul(2, node_count, scioxide_core::quantity!("exactness_degree"))?
            - match variant {
                GaussianRuleVariant::Gauss => 1,
                GaussianRuleVariant::RadauLeft | GaussianRuleVariant::RadauRight => 2,
                GaussianRuleVariant::Lobatto => 3,
            };
        let mass = RealScalar::exp(family.log_weight_mass()?);
        if !mass.is_finite() || mass <= <S as scioxide_core::AlgebraicScalar>::ZERO {
            return Err(breakdown(
                "classical weight mass is outside the positive finite scalar range",
            ));
        }
        // Golub-Welsch: the nodes are the eigenvalues of the symmetric Jacobi matrix built
        // from the three-term recurrence, and each weight is `mass * v_0^2` where v_0 is
        // the first component of the normalized eigenvector. Radau/Lobatto rules modify the
        // last row of the matrix first (Golub 1973) so the prescribed endpoints become
        // eigenvalues. This avoids polynomial root finding and its ill-conditioning.
        let recurrence = family.jacobi_recurrence(node_count)?;
        let (decomposition, recurrence, matrix_norm) = if variant == GaussianRuleVariant::Gauss {
            let decomposition = HermitianEigenFirstComponents::tridiagonal_with_options(
                recurrence.diagonal(),
                recurrence.off_diagonal(),
                &options,
            )?;
            (decomposition, Some(recurrence), S::ONE)
        } else {
            let (mut diagonal, mut off_diagonal) = recurrence.into_parts();
            constrain_endpoints(&mut diagonal, &mut off_diagonal, variant)?;
            let norm = row_sum_norm(&diagonal, &off_diagonal);
            (
                HermitianEigenFirstComponents::tridiagonal_with_options(
                    &diagonal,
                    &off_diagonal,
                    &options,
                )?,
                None,
                norm,
            )
        };
        let (mut nodes, components) = decomposition.into_parts();
        let mut weights = try_with_capacity(node_count, scioxide_core::quantity!("weights"))?;
        for (&node, component) in nodes.iter().zip(components) {
            if let Some(recurrence) = &recurrence
                && matches!(
                    family,
                    OrthogonalFamily::HermiteH | OrthogonalFamily::HermiteHe
                )
            {
                // DLMF 3.5.18 and the Hermite derivative identity:
                // w_i = mass / (n * p_{n-1}(x_i)^2), with p_0 = 1.
                // The shared scaled member preserves tiny, undeflated weights.
                let member = recurrence.last_member_scaled(node)?;
                weights.push(
                    Scaled::from_value(mass)?
                        .div(
                            Scaled::from_value(S::try_from_usize_exact(node_count)?)?
                                .mul(member.square()?)?,
                        )?
                        .round()?,
                );
                continue;
            }
            let component = Scaled::from_value(component)?;
            weights.push(
                Scaled::from_value(mass)?
                    .mul(component)?
                    .mul(component)?
                    .round()?,
            );
        }
        // Preserve exact parity for symmetric measures. This also prevents
        // large odd moments from accumulating asymmetric eigensolver noise.
        if symmetric(family)
            && matches!(
                variant,
                GaussianRuleVariant::Gauss | GaussianRuleVariant::Lobatto
            )
        {
            let two = <S as scioxide_core::AlgebraicScalar>::ONE
                + <S as scioxide_core::AlgebraicScalar>::ONE;
            // Average each mirrored pair instead of copying one side; halving before
            // subtracting/adding keeps the averaging overflow-free for huge nodes/weights.
            for left in 0..node_count / 2 {
                let right = node_count - 1 - left;
                let node = nodes[right] / two - nodes[left] / two;
                let weight = weights[left] / two + weights[right] / two;
                nodes[left] = -node;
                nodes[right] = node;
                weights[left] = weight;
                weights[right] = weight;
            }
            if !node_count.is_multiple_of(2) {
                nodes[node_count / 2] = <S as scioxide_core::AlgebraicScalar>::ZERO;
            }
        }
        // The eigensolver returns prescribed endpoints only up to its backward error,
        // which grows with the order and the modified matrix norm: a computed eigenvalue
        // of a symmetric tridiagonal matrix T is within n·u·‖T‖ of an exact one, where u
        // is the larger of the deflation tolerance and the unit roundoff. Prescribed
        // endpoints are checked against eight times that bound (a larger miss means the
        // modified Jacobi matrix was wrong) and then snapped to exactly -1 / 1, so mapped
        // endpoint nodes land bit-for-bit on the caller's bounds rather than outside them.
        let allowance = options
            .deflation_tolerance()
            .threshold(<S as scioxide_core::AlgebraicScalar>::ONE)?
            .max(<S as RealScalar>::EPSILON)
            * S::try_from_f64_lossy(node_count as f64)?
            * matrix_norm.max(<S as scioxide_core::AlgebraicScalar>::ONE)
            * S::try_from_usize_exact(8)?;
        for (index, endpoint) in [
            (0, -<S as scioxide_core::AlgebraicScalar>::ONE),
            (node_count - 1, <S as scioxide_core::AlgebraicScalar>::ONE),
        ] {
            let prescribed = if endpoint < <S as scioxide_core::AlgebraicScalar>::ZERO {
                matches!(
                    variant,
                    GaussianRuleVariant::RadauLeft | GaussianRuleVariant::Lobatto
                )
            } else {
                matches!(
                    variant,
                    GaussianRuleVariant::RadauRight | GaussianRuleVariant::Lobatto
                )
            };
            if prescribed {
                if scioxide_core::NumericScalar::abs(nodes[index] - endpoint) > allowance {
                    return Err(breakdown(
                        "prescribed endpoint failed its eigenvalue tolerance",
                    ));
                }
                nodes[index] = endpoint;
            }
        }
        validate_nodes(&nodes, domain(family))?;
        let sum = weights
            .iter()
            .try_fold(ScaledAccumulator::ZERO, |mut sum, &weight| {
                if weight <= <S as scioxide_core::AlgebraicScalar>::ZERO || !weight.is_finite() {
                    return Err(breakdown(
                        "quadrature weight is outside the positive finite scalar range",
                    ));
                }
                sum.add_scaled(Scaled::from_value(weight)?)?;
                Ok(sum)
            })?
            .total()?;
        // Rescale so the weights sum to the exact zeroth moment. Squared eigenvector
        // components carry a relative error of a few ulps each; renormalizing makes the
        // rule integrate constants exactly (up to one rounding per weight).
        let normalization = Scaled::from_value(mass)?.div(sum)?;
        weights.iter_mut().try_for_each(|weight| {
            *weight = Scaled::from_value(*weight)?.mul(normalization)?.round()?;
            Ok::<(), SciOxideError>(())
        })?;
        Ok(Self {
            family,
            variant,
            mass,
            degree,
            rule: Arc::new(PreparedRule::new(
                nodes,
                weights,
                scioxide_core::argument!("node_count"),
            )?),
        })
    }
    /// Computes the n-node Gauss rule for a classical weight with default options.
    ///
    /// Uses Golub–Welsch in O(n²) time and O(n) storage, with the precision and
    /// domain of [`Self::with_options`]. The exactness degree is 2n − 1.
    ///
    /// # Errors
    ///
    /// Returns the preparation errors of [`Self::with_options`].
    pub fn new(family: OrthogonalFamily<S>, node_count: usize) -> Result<Self, SciOxideError> {
        Self::with_options(
            family,
            node_count,
            GaussianRuleVariant::Gauss,
            GaussianRuleOptions::default(),
        )
    }
    /// Computes the n-node Gauss–Legendre rule on [−1, 1].
    ///
    /// Uses [`Self::new`] in O(n²) time and O(n) storage. The rule integrates
    /// polynomials through degree 2n − 1 exactly, up to rounding in S.
    ///
    /// # Errors
    ///
    /// Returns the preparation errors of [`Self::with_options`].
    ///
    /// ```
    /// let rule = scioxide_integrate::GaussianRulePlan::<f64>::gauss_legendre(4)?;
    /// assert!((rule.weights().iter().sum::<f64>() - 2.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. H. Golub and J. H. Welsch, Math. Comp. 23 (1969), 221–230.
    pub fn gauss_legendre(node_count: usize) -> Result<Self, SciOxideError> {
        Self::new(OrthogonalFamily::Legendre, node_count)
    }
}

impl<S: RealScalar> GaussianRulePlan<S> {
    /// Returns the orthogonal polynomial family, with its weight parameters.
    pub const fn family(&self) -> OrthogonalFamily<S> {
        self.family
    }
    /// Returns the variant: which endpoints are prescribed nodes.
    pub const fn variant(&self) -> GaussianRuleVariant {
        self.variant
    }
    /// Returns the domain of the weight function and of [`Self::nodes`].
    #[inline(always)]
    pub fn domain(&self) -> GaussianDomain {
        domain(self.family)
    }
    /// Returns the zeroth moment `μ₀ = ∫ w(t) dt` of the weight function over
    /// its domain, which is also the sum of the weights.
    pub const fn zeroth_moment(&self) -> S {
        self.mass
    }
    /// Returns the highest polynomial degree the rule integrates exactly, in
    /// exact arithmetic: `2n − 1`, `2n − 2` or `2n − 3` (see
    /// [`GaussianRuleVariant`]).
    pub const fn exactness_degree(&self) -> usize {
        self.degree
    }
    /// Returns the order `d` of the derivative in the remainder term of the rule,
    /// `exactness_degree() + 1`.
    ///
    /// The remainder is proportional to `f⁽ᵈ⁾(ξ)` for some `ξ` in the domain
    /// when `f` has a continuous `d`-th derivative. The plan does not compute it.
    pub const fn remainder_derivative_order(&self) -> usize {
        self.degree + 1
    }
    /// Returns the nodes `tᵢ` on the [domain](Self::domain), in strictly
    /// ascending order.
    pub fn nodes(&self) -> &[S] {
        self.rule.nodes()
    }
    /// Returns the positive weights `wᵢ`, paired with [`Self::nodes`].
    pub fn weights(&self) -> &[S] {
        self.rule.weights()
    }

    /// Rejects families whose weight is not identically one on `[-1, 1]`; only
    /// Legendre, Jacobi(0, 0) and Gegenbauer(1/2) are accepted.
    #[inline(always)]
    fn require_unit_weight(&self) -> Result<(), SciOxideError> {
        let unit = match self.family {
            OrthogonalFamily::Legendre => true,
            OrthogonalFamily::Jacobi { alpha, beta } => alpha == S::ZERO && beta == S::ZERO,
            OrthogonalFamily::Gegenbauer { lambda } => lambda == S::HALF,
            _ => false,
        };
        if unit {
            Ok(())
        } else {
            Err(crate::kernel::invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "unweighted integration requires a unit-weight family; \
                 use integrate_weighted for classical weights",
            ))
        }
    }

    /// Integrates `f` against the weight function over `support` and writes the
    /// result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// The weight function is mapped to `support` as described on
    /// [`GaussianSupport`]. `out` must have the component count the workspace
    /// was created for; create the workspace with [`FixedRule::workspace`]. The
    /// call evaluates `f` at the `n` nodes. The report has no error estimate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (argument `support`) if `support` does not match the
    ///   [`GaussianDomain`] of the family.
    /// - `NonFinite` if the lower bound of a half-line is NaN or infinite.
    /// - `DimensionMismatch` if `out` does not have the workspace's component
    ///   count, or the workspace is too small for this rule.
    /// - `NumericalInstability` if the mapped nodes are not strictly monotone
    ///   inside `support`, for example because the interval is too narrow.
    /// - `ConvergenceFailure` if a budget of `scope` cannot cover `n` evaluations.
    /// - `ExecutionStopped` if `scope` is cancelled or its deadline passes.
    /// - Errors and non-finite values from `f`, as described in
    ///   [Integrands](crate#integrands).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scioxide_core::{Complex, ExecutionContext};
    /// # use scioxide_integrate::{
    /// #     FixedRule, GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport,
    /// # };
    /// # use scioxide_poly::OrthogonalFamily;
    /// // ∫ (x² + i) e^(−x²) dx over the real line = √π/2 + i √π
    /// let options = GaussianRuleOptions::default();
    /// let family = OrthogonalFamily::HermiteH;
    /// let hermite = GaussianRulePlan::<f64>::with_options(family, 8, GaussianRuleVariant::Gauss, options)?;
    /// let mut workspace = hermite.workspace(2)?;
    /// let mut out = Complex::new(0.0, 0.0);
    /// hermite.integrate_weighted_into(
    ///     |x: f64| Ok(Complex::new(x * x, 1.0)),
    ///     GaussianSupport::RealLine,
    ///     &mut out,
    ///     &mut workspace,
    ///     ExecutionContext::sequential_native(),
    /// )?;
    /// let root_pi = core::f64::consts::PI.sqrt();
    /// assert!((out.re - root_pi / 2.0).abs() < 1e-14 && (out.im - root_pi).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline(always)]
    pub fn integrate_weighted_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        support: GaussianSupport<S>,
        out: &mut O,
        workspace: &mut FixedRuleWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        let geometry = match self.geometry(support) {
            Ok(geometry) => geometry,
            Err(error) => return workspace.reject(error),
        };
        self.integrate_on_support(f, support, geometry, out, workspace, scope.into())
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn integrate_on_support<F, V, O>(
        &self,
        f: &F,
        support: GaussianSupport<S>,
        geometry: Geometry<S>,
        out: &mut O,
        workspace: &mut FixedRuleWorkspace<S>,
        scope: SolveScope<'_, '_>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let native = match support {
            GaussianSupport::Interval(interval) => self
                .rule
                .native_interval(interval, self.family != OrthogonalFamily::ChebyshevT),
            GaussianSupport::HalfLine(lower) => self.rule.native_translation(lower),
            GaussianSupport::RealLine => self.rule.native_identity(),
        };
        apply(
            self,
            |nodes: &mut [FixedRuleNode<S>]| {
                self.weighted_nodes(support, nodes)
                    .map(crate::fixed::Mapping::Stored)
            },
            geometry,
            native,
            f,
            out,
            workspace,
            scope,
        )
    }

    /// Integrates `f` against the weight function over `support` and returns the
    /// value and the report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Creates a workspace for the call and runs on
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - The errors of [`FixedRule::workspace`] and
    ///   [`Self::integrate_weighted_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_integrate::{
    ///     GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport,
    /// };
    /// use scioxide_poly::OrthogonalFamily;
    ///
    /// // ∫₂^∞ (x − 2)² e^(−(x − 2)) dx = Γ(3) = 2
    /// let options = GaussianRuleOptions::default();
    /// let family = OrthogonalFamily::Laguerre;
    /// let laguerre = GaussianRulePlan::<f64>::with_options(family, 6, GaussianRuleVariant::Gauss, options)?;
    /// let f = |x: f64| Ok((x - 2.0).powi(2));
    /// let (value, report) = laguerre.integrate_weighted(&f, GaussianSupport::HalfLine(2.0))?;
    /// assert!((value - 2.0).abs() < 1e-13);
    /// assert_eq!(report.evaluations(), 6);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn integrate_weighted<F, V>(
        &self,
        f: F,
        support: GaussianSupport<S>,
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let f = &f;
        crate::execution::owned(
            V::default(),
            |components| FixedRule::workspace(self, components),
            |out, ws, cx| self.integrate_weighted_into(f, support, out, ws, cx),
        )
    }

    /// Checks that `support` matches the family's domain.
    #[inline(always)]
    fn geometry(&self, support: GaussianSupport<S>) -> Result<Geometry<S>, SciOxideError> {
        let geometry = match (support, self.domain()) {
            (GaussianSupport::Interval(interval), GaussianDomain::UnitInterval) => {
                Geometry::Interval {
                    interval,
                    single_endpoint: FixedRule::permits_single_endpoint(self),
                }
            }
            (GaussianSupport::HalfLine(lower), GaussianDomain::NonnegativeRealLine) => {
                if !lower.is_finite() {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Input,
                        "half-line lower bound must be finite",
                    ));
                }
                Geometry::HalfLine { lower }
            }
            (GaussianSupport::RealLine, GaussianDomain::RealLine) => Geometry::RealLine,
            _ => {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("support"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "support must match the family: an interval for bounded families, \
                     a half-line for Laguerre and the real line for Hermite",
                ));
            }
        };
        Ok(geometry)
    }

    /// Writes the nodes and weights mapped to `support`; returns whether their
    /// geometry is known to be valid.
    #[inline(always)]
    fn weighted_nodes(
        &self,
        support: GaussianSupport<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<bool, SciOxideError> {
        crate::kernel::shape(self.rule.len(), destination.len())?;
        match support {
            GaussianSupport::Interval(interval) => self.rule.map_into(
                crate::fixed::interval_map(interval)?,
                self.family != OrthogonalFamily::ChebyshevT,
                destination,
            ),
            GaussianSupport::HalfLine(lower) => {
                let native = self.rule.admits_translation(lower);
                for (slot, (&node, &weight)) in destination
                    .iter_mut()
                    .zip(self.nodes().iter().zip(self.weights()))
                {
                    let point = node + lower;
                    *slot = FixedRuleNode::new(
                        if point.is_finite() {
                            point
                        } else {
                            Scaled::from_value(node)?
                                .add(Scaled::from_value(lower)?)?
                                .round()?
                        },
                        weight,
                        None,
                    );
                }
                Ok(native)
            }
            GaussianSupport::RealLine => {
                for (slot, (&node, &weight)) in destination
                    .iter_mut()
                    .zip(self.nodes().iter().zip(self.weights()))
                {
                    *slot = FixedRuleNode::new(node, weight, None);
                }
                Ok(true)
            }
        }
    }
}

/// Integrates `∫ₐᵇ f(x) dx` with a rule whose weight function is identically one
/// (Legendre, Jacobi(0, 0) and Gegenbauer(1/2)).
///
/// Every other family fails with `InvalidInput` (argument `self`); use
/// [`GaussianRulePlan::integrate_weighted_into`] for those.
impl<S: RealScalar> FixedRule<S> for GaussianRulePlan<S> {
    fn validate_unweighted(&self) -> Result<(), SciOxideError> {
        self.require_unit_weight()
    }
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "GaussianRulePlan", 1)
    }

    fn node_count(&self) -> usize {
        self.rule.len()
    }

    fn permits_single_endpoint(&self) -> bool {
        matches!(
            self.variant,
            GaussianRuleVariant::RadauLeft | GaussianRuleVariant::RadauRight
        )
    }

    fn map_nodes_into(
        &self,
        interval: Interval<S>,
        destination: &mut [FixedRuleNode<S>],
    ) -> Result<(), SciOxideError> {
        self.require_unit_weight()?;
        self.weighted_nodes(GaussianSupport::Interval(interval), destination)
            .map(|_| ())
    }

    #[inline(always)]
    fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        interval: Interval<S>,
        out: &mut O,
        workspace: &mut FixedRuleWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        if let Err(error) = self.validate_unweighted() {
            return workspace.reject(error);
        }
        self.integrate_on_support(
            f,
            GaussianSupport::Interval(interval),
            Geometry::Interval {
                interval,
                single_endpoint: self.permits_single_endpoint(),
            },
            out,
            workspace,
            scope.into(),
        )
    }
}

fn domain<S: RealScalar>(family: OrthogonalFamily<S>) -> GaussianDomain {
    match family {
        OrthogonalFamily::Laguerre | OrthogonalFamily::GeneralizedLaguerre { .. } => {
            GaussianDomain::NonnegativeRealLine
        }
        OrthogonalFamily::HermiteH | OrthogonalFamily::HermiteHe => GaussianDomain::RealLine,
        _ => GaussianDomain::UnitInterval,
    }
}
fn symmetric<S: RealScalar>(family: OrthogonalFamily<S>) -> bool {
    match family {
        OrthogonalFamily::Jacobi { alpha, beta } => alpha == beta,
        OrthogonalFamily::Laguerre | OrthogonalFamily::GeneralizedLaguerre { .. } => false,
        _ => true,
    }
}
fn validate_nodes<S: RealScalar>(nodes: &[S], domain: GaussianDomain) -> Result<(), SciOxideError> {
    use crate::kernel::{MappedPoints, PointDomain};
    let domain = match domain {
        GaussianDomain::UnitInterval => PointDomain::Interval {
            interval: Interval::new(-S::ONE, S::ONE)?,
            closed: true,
        },
        GaussianDomain::NonnegativeRealLine => PointDomain::HalfLine { lower: S::ZERO },
        GaussianDomain::RealLine => PointDomain::RealLine,
    };
    let mut validator = MappedPoints::new(domain, true);
    if !nodes.iter().all(|&node| validator.push(node) == Ok(true)) {
        return Err(breakdown(
            "nodes are non-finite, unordered, or outside the weight domain",
        ));
    }
    Ok(())
}

// Monic P_(n-2)(x)/P_(n-1)(x), evaluated as a continued fraction.
// With p_k = (x - a_(k-1)) p_(k-1) - b_(k-2)^2 p_(k-2), the ratio r_k = p_(k-1)/p_k obeys
// r_k = 1 / (x - a_(k-1) - b_(k-2)^2 r_(k-1)). Iterating on the ratio avoids forming the
// polynomial values themselves, which overflow or underflow quickly as n grows.
fn endpoint_ratio<S: RealScalar>(
    diagonal: &[S],
    off_diagonal: &[S],
    point: S,
) -> Result<Scaled<S>, SciOxideError> {
    let one = Scaled::from_value(S::ONE)?;
    let mut ratio = one.div(Scaled::from_value(point)?.sub(Scaled::from_value(diagonal[0])?)?)?;
    for index in 1..diagonal.len() - 1 {
        let off = Scaled::from_value(off_diagonal[index - 1])?;
        let denominator = Scaled::from_value(point)?
            .sub(Scaled::from_value(diagonal[index])?)?
            .sub(off.mul(off)?.mul(ratio)?)?;
        ratio = one.div(denominator)?;
    }
    Ok(ratio)
}
/// Gershgorin bound `max_i |a_i| + |b_(i-1)| + |b_i|` on the spectral radius of
/// a symmetric tridiagonal matrix; it saturates to infinity rather than failing.
fn row_sum_norm<S: RealScalar>(diagonal: &[S], off_diagonal: &[S]) -> S {
    let off = |index: Option<usize>| {
        index
            .and_then(|i| off_diagonal.get(i))
            .map_or(S::ZERO, |b| b.abs())
    };
    (0..diagonal.len()).fold(S::ZERO, |norm, i| {
        norm.max(diagonal[i].abs() + off(i.checked_sub(1)) + off(Some(i)))
    })
}

fn constrain_endpoints<S: RealScalar>(
    diagonal: &mut [S],
    off_diagonal: &mut [S],
    variant: GaussianRuleVariant,
) -> Result<(), SciOxideError> {
    if variant == GaussianRuleVariant::Gauss {
        return Ok(());
    }
    // Golub's modification of the Jacobi matrix: for a prescribed node `x0`, choosing
    //     a_n = x0 - b_(n-1)^2 * p_(n-2)(x0) / p_(n-1)(x0)
    // makes `x0` an eigenvalue (Radau). For Lobatto both `a_n` and `b_(n-1)^2` are free and
    // solve the 2x2 linear system that makes -1 and 1 eigenvalues simultaneously:
    //     b_(n-1)^2 = (1 - (-1)) / (r(1) - r(-1)),   a_n = -1 - b_(n-1)^2 r(-1).
    let last = diagonal.len() - 1;
    let left = -S::ONE;
    let right = S::ONE;
    if last == 0 {
        diagonal[0] = if variant == GaussianRuleVariant::RadauLeft {
            left
        } else {
            right
        };
        return Ok(());
    }
    let (point, ratio, beta_squared) = if variant == GaussianRuleVariant::Lobatto {
        let left_ratio = endpoint_ratio(diagonal, off_diagonal, left)?;
        let right_ratio = endpoint_ratio(diagonal, off_diagonal, right)?;
        let beta = Scaled::from_value(right - left)?.div(right_ratio.sub(left_ratio)?)?;
        let rounded = beta.round()?;
        if rounded <= S::ZERO {
            return Err(breakdown("non-positive Lobatto off-diagonal square"));
        }
        off_diagonal[last - 1] = rounded.sqrt();
        (left, left_ratio, beta)
    } else {
        let point = if variant == GaussianRuleVariant::RadauLeft {
            left
        } else {
            right
        };
        let off = Scaled::from_value(off_diagonal[last - 1])?;
        (
            point,
            endpoint_ratio(diagonal, off_diagonal, point)?,
            off.mul(off)?,
        )
    };
    diagonal[last] = Scaled::from_value(point)?
        .sub(beta_squared.mul(ratio)?)?
        .round()?;
    Ok(())
}

fn breakdown(message: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Breakdown, message)
}
