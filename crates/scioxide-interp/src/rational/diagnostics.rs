//! Zeros, poles and residues of barycentric rational functions, and removal of Froissart doublets.

use alloc::vec::Vec;

use scioxide_core::backend::{try_copy_slice, try_filled, try_with_capacity};
use scioxide_core::{
    AlgebraicScalar, Complex, NumericScalar, RealScalar, SciOxideError, Tolerance,
};
use scioxide_linalg::{GeneralEigenOptions, Matrix};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{
    ApproxPolynomial, MonomialBasis, PolynomialApproximationPolicy, RootDiagnostics,
    polynomial_root_diagnostics,
};

use super::{BarycentricRational, Support};
use crate::checked::intermediate;
use crate::error::instability;
use crate::error::invalid;
use crate::protocol::ExtrapolationPolicy;
use crate::validate::{validate_count, validate_finite, validate_length};

/// Describes one pole of a rational function: location, residue and distance to the nearest zero.
///
/// Part of [`RationalDiagnostics`]. A pole is a suspected Froissart doublet
/// if its residue is small and a zero lies close to it; such a pair nearly
/// cancels and is usually an artifact of noise or rounding.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Interval, Tolerance};
/// use scioxide_interp::{BarycentricRational, RationalDomain};
/// use scioxide_linalg::RankPolicy;
/// use scioxide_poly::PolynomialApproximationPolicy;
/// // r(z) = 1/(z − 2) on {−1, 1}: weights b = (3/2, −1/2), numerator aⱼ = bⱼ r(zⱼ).
/// let plan = BarycentricRational::new(
///     vec![-1.0_f64, 1.0],
///     vec![-0.5, 0.5],
///     vec![1.5, -0.5],
///     RationalDomain::Real(Interval::new(-1.0, 1.0)?),
/// )?;
/// let policy = PolynomialApproximationPolicy::new(
///     Tolerance::new(1e-13, 1e-13)?,
///     RankPolicy::DimensionScaledEpsilon,
/// );
/// let report = plan.diagnostics(policy, Tolerance::new(1e-6, 1e-6)?, &Default::default())?;
/// let pole = &report.poles()[0];
/// assert!((pole.location().re - 2.0).abs() < 1e-12);
/// assert!((pole.residue().unwrap().re - 1.0).abs() < 1e-12);
/// assert!(!pole.suspected_doublet());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RationalPole<R: RealScalar> {
    location: Complex<R>,
    residue: Option<Complex<R>>,
    nearest_zero_distance: Option<R>,
    suspected_doublet: bool,
}

impl<R: RealScalar> RationalPole<R> {
    /// Returns the estimated location of the pole.
    pub const fn location(&self) -> Complex<R> {
        self.location
    }
    /// Returns the residue `p(ζ) / q′(ζ)` of a simple pole `ζ`, or `None` if `q′(ζ) = 0`.
    pub const fn residue(&self) -> Option<Complex<R>> {
        self.residue
    }
    /// Returns the distance to the nearest zero, or `None` if the numerator has no roots.
    pub const fn nearest_zero_distance(&self) -> Option<R> {
        self.nearest_zero_distance
    }
    /// Returns whether a small residue and a nearby zero suggest a Froissart doublet.
    ///
    /// Both hold when the distance to the nearest zero is within the doublet
    /// tolerance relative to `max(|ζ|, h)` and `|residue|` is within it
    /// relative to `h · maxⱼ|aⱼ| / maxⱼ|bⱼ|`, where `h` is the coordinate
    /// scale of [`RationalDiagnostics::coordinate_scale`].
    #[must_use]
    pub const fn suspected_doublet(&self) -> bool {
        self.suspected_doublet
    }
}

/// Lists the zeros and poles of a rational function, with residues and root diagnostics.
///
/// Returned by [`BarycentricRational::diagnostics`]. The roots are
/// eigenvalues of companion matrices and are estimates, not verified
/// enclosures.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Interval, Tolerance};
/// use scioxide_interp::{BarycentricRational, RationalDomain};
/// use scioxide_linalg::RankPolicy;
/// use scioxide_poly::PolynomialApproximationPolicy;
/// // r(z) = z on {−1, 1}: weights b = (1, −1), numerator aⱼ = bⱼ r(zⱼ).
/// let plan = BarycentricRational::new(
///     vec![-1.0_f64, 1.0],
///     vec![-1.0, -1.0],
///     vec![1.0, -1.0],
///     RationalDomain::Real(Interval::new(-1.0, 1.0)?),
/// )?;
/// let policy = PolynomialApproximationPolicy::new(
///     Tolerance::new(1e-13, 1e-13)?,
///     RankPolicy::DimensionScaledEpsilon,
/// );
/// let report = plan.diagnostics(policy, Tolerance::new(1e-6, 1e-6)?, &Default::default())?;
/// assert_eq!(report.roots().len(), 1);
/// assert!(report.roots()[0].norm() < 1e-14);
/// assert!(report.poles().is_empty());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RationalDiagnostics<R: RealScalar> {
    roots: Vec<Complex<R>>,
    poles: Vec<RationalPole<R>>,
    numerator_diagnostics: RootDiagnostics<R>,
    denominator_diagnostics: RootDiagnostics<R>,
    trimmed_coefficients: [usize; 2],
    center: Complex<R>,
    coordinate_scale: R,
}

impl<R: RealScalar> RationalDiagnostics<R> {
    /// Returns the zeros of the rational function; empty for the zero function.
    pub fn roots(&self) -> &[Complex<R>] {
        &self.roots
    }
    /// Returns the poles, with residues and doublet classification.
    pub fn poles(&self) -> &[RationalPole<R>] {
        &self.poles
    }
    /// Returns the root diagnostics of the numerator polynomial in the local coordinate `u`.
    pub const fn numerator_diagnostics(&self) -> &RootDiagnostics<R> {
        &self.numerator_diagnostics
    }
    /// Returns the root diagnostics of the denominator polynomial in the local coordinate `u`.
    pub const fn denominator_diagnostics(&self) -> &RootDiagnostics<R> {
        &self.denominator_diagnostics
    }
    /// Returns how many negligible leading coefficients were dropped from the numerator and denominator.
    ///
    /// Coefficients below the tolerance of the polynomial policy are dropped
    /// before computing roots, which avoids spurious roots near infinity. The
    /// rational function itself is unchanged.
    #[must_use]
    pub const fn trimmed_coefficients(&self) -> [usize; 2] {
        self.trimmed_coefficients
    }
    /// Returns the origin `c` of the local coordinate `u = (z − c) / h`, the first support point.
    pub const fn center(&self) -> Complex<R> {
        self.center
    }
    /// Returns the scale `h` of the local coordinate `u = (z − c) / h`, the largest distance from `c` to a support point.
    pub const fn coordinate_scale(&self) -> R {
        self.coordinate_scale
    }
}

/// Holds the result of [`BarycentricRational::cleanup_doublets`]: a refitted approximant and its residuals.
///
/// Compare [`Self::before_residual`] with [`Self::after_residual`] to decide
/// whether to use the new approximant.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Interval, Tolerance};
/// use scioxide_interp::{BarycentricRational, RationalDomain};
/// use scioxide_linalg::RankPolicy;
/// use scioxide_poly::PolynomialApproximationPolicy;
/// let plan = BarycentricRational::new(
///     vec![-1.0_f64, 1.0],
///     vec![-1.0, -1.0],
///     vec![1.0, -1.0],
///     RationalDomain::Real(Interval::new(-1.0, 1.0)?),
/// )?;
/// let policy = PolynomialApproximationPolicy::new(
///     Tolerance::new(1e-13, 1e-13)?,
///     RankPolicy::DimensionScaledEpsilon,
/// );
/// let x = [-1.0, 0.0, 1.0];
/// let cleanup = plan.cleanup_doublets(
///     &x,
///     &x,
///     policy,
///     Tolerance::new(1e-6, 1e-6)?,
///     &Default::default(),
/// )?;
/// assert_eq!(cleanup.removed_supports(), 0);
/// assert!(cleanup.after_residual() <= cleanup.before_residual());
/// assert_eq!(
///     cleanup.approximant().support_points(),
///     plan.support_points()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RationalCleanup<C: NumericScalar> {
    approximant: BarycentricRational<C>,
    removed_supports: usize,
    before_residual: C::Real,
    after_residual: C::Real,
}

impl<C: NumericScalar> RationalCleanup<C> {
    /// Returns the refitted approximant.
    pub const fn approximant(&self) -> &BarycentricRational<C> {
        &self.approximant
    }
    /// Returns the refitted approximant by value.
    pub fn into_approximant(self) -> BarycentricRational<C> {
        self.approximant
    }
    /// Returns the number of support points removed.
    #[must_use]
    pub const fn removed_supports(&self) -> usize {
        self.removed_supports
    }
    /// Returns the maximum residual at the samples before cleanup.
    pub const fn before_residual(&self) -> C::Real {
        self.before_residual
    }
    /// Returns the maximum residual at the samples after cleanup.
    pub const fn after_residual(&self) -> C::Real {
        self.after_residual
    }
}

fn complex<C: NumericScalar>(x: C) -> Complex<C::Real> {
    Complex::new(x.real_component(), x.imaginary_component())
}

type DiagnosticPolynomials<R> = (
    Polynomial<Complex<R>>,
    Polynomial<Complex<R>>,
    Complex<R>,
    R,
);
type Polynomial<C> = ApproxPolynomial<C, MonomialBasis<C>>;

impl<C: NumericScalar> BarycentricRational<C>
where
    Complex<C::Real>: NumericScalar<Real = C::Real>,
{
    /// Computes the zeros, poles and residues of the rational function and flags suspected Froissart doublets.
    ///
    /// The barycentric form is multiplied by `Πⱼ (z − zⱼ)` to give polynomials
    /// `p / q` in the local coordinate `u = (z − z₀) / h`, which puts every
    /// support point in the unit disc. Negligible leading coefficients, under
    /// the tolerance of `policy`, are dropped, and the roots of `p` and `q` are
    /// the eigenvalues of their companion matrices, computed with
    /// `eigen_options`. For `k` support points this costs O(k³) operations.
    /// Roots of large modulus are less accurate. A pole is flagged as a doublet
    /// when both its residue and its distance to the nearest zero are small
    /// under the tolerance `doublet`; see [`RationalPole::suspected_doublet`].
    /// Evaluation of the function is unaffected.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Breakdown`) if the tolerance removes every
    ///   denominator coefficient.
    /// - `NonFinite` if the conversion overflows.
    /// - The errors of the eigenvalue solver.
    pub fn diagnostics(
        &self,
        policy: PolynomialApproximationPolicy<C::Real>,
        doublet: Tolerance<C::Real>,
        eigen_options: &GeneralEigenOptions<C::Real>,
    ) -> Result<RationalDiagnostics<C::Real>, SciOxideError> {
        let support = &self.support;
        let (p, q, center, scale) = diagnostic_polynomials(support, policy)?;
        let (p, p_trim) = trim_diagnostic(p)?;
        let (q, q_trim) = trim_diagnostic(q)?;
        if q.is_zero() {
            return Err(instability(
                scioxide_core::NumericalIssue::Breakdown,
                "diagnostic tolerance removed the denominator",
            ));
        }
        let roots = polynomial_roots(&p, eigen_options)?;
        let poles = polynomial_roots(&q, eigen_options)?;
        let numerator_diagnostics = polynomial_root_diagnostics(p.basis().coefficients(), &roots)?;
        let denominator_diagnostics =
            polynomial_root_diagnostics(q.basis().coefficients(), &poles)?;
        // Residue of a simple pole zeta of p/q is p(zeta)/q'(zeta) in the local
        // coordinate u = (z - center)/scale; since dz = scale du, the residue in
        // original coordinates carries one factor of `scale`. `value_scale`
        // (max|a| / max|b|) estimates the function magnitude so the doublet
        // residue test is relative to the data rather than absolute.
        let derivative = q.derivative()?;
        let value_scale = support
            .numerator
            .iter()
            .fold(C::Real::ZERO, |s, a| s.max(a.abs()))
            / support
                .denominator
                .iter()
                .fold(C::Real::ZERO, |s, b| s.max(b.abs()));
        let mut reports = try_with_capacity(
            poles.len(),
            scioxide_core::quantity!("rational_pole_reports"),
        )?;
        for pole in poles {
            let slope = derivative.evaluate(pole)?;
            let residue = if slope == Complex::new(C::Real::ZERO, C::Real::ZERO) {
                None
            } else {
                Some(intermediate(
                    p.evaluate(pole)? / slope * Complex::new(scale, C::Real::ZERO),
                )?)
            };
            let nearest = roots
                .iter()
                .try_fold(None, |nearest: Option<C::Real>, &z| {
                    let distance = crate::checked::output((z - pole).abs() * scale)?;
                    Ok::<_, SciOxideError>(Some(nearest.map_or(distance, |old| old.min(distance))))
                })?;
            let location = intermediate(center + pole * Complex::new(scale, C::Real::ZERO))?;
            // Froissart doublet: a spurious pole almost cancelled by a nearby
            // zero, typically introduced by noise or rounding. Both conditions
            // are required because either alone also describes genuine poles.
            let suspected_doublet = match (nearest, residue) {
                (Some(distance), Some(residue)) => {
                    distance <= doublet.threshold(location.abs().max(scale))?
                        && residue.abs() <= doublet.threshold(value_scale * scale)?
                }
                _ => false,
            };
            reports.push(RationalPole {
                location,
                residue,
                nearest_zero_distance: nearest,
                suspected_doublet,
            });
        }
        let mut mapped =
            try_with_capacity(roots.len(), scioxide_core::quantity!("rational_roots"))?;
        for root in roots {
            mapped.push(intermediate(
                center + root * Complex::new(scale, C::Real::ZERO),
            )?);
        }
        Ok(RationalDiagnostics {
            roots: mapped,
            poles: reports,
            trimmed_coefficients: [p_trim, q_trim],
            numerator_diagnostics,
            denominator_diagnostics,
            center,
            coordinate_scale: scale,
        })
    }

    /// Removes the support point nearest each suspected Froissart doublet and refits the weights.
    ///
    /// `nodes` and `values` are the original samples, which must include every
    /// support point. After [`Self::diagnostics`], each suspected doublet
    /// removes its nearest support point, and the weights on the remaining
    /// support points are recomputed once from the Loewner matrix, as in an
    /// AAA step, without adding new support points. `self` is unchanged.
    /// Check the residuals in the returned [`RationalCleanup`] before using the
    /// new approximant.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::diagnostics`].
    /// - `InvalidInput` if `nodes` is empty, contains duplicates or omits a
    ///   support point.
    /// - `DimensionMismatch` if `nodes` and `values` differ in length.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `DomainError` if a node lies outside the domain.
    /// - `NumericalInstability` (`Breakdown`) if every support point would be
    ///   removed.
    /// - The errors of the SVD if it fails.
    ///
    /// # References
    ///
    /// Y. Nakatsukasa, O. Sète and L. N. Trefethen, "The AAA algorithm for
    /// rational approximation", *SIAM J. Sci. Comput.* 40(3), 2018, A1494–A1522.
    pub fn cleanup_doublets(
        &self,
        nodes: &[C],
        values: &[C],
        policy: PolynomialApproximationPolicy<C::Real>,
        doublet: Tolerance<C::Real>,
        eigen_options: &GeneralEigenOptions<C::Real>,
    ) -> Result<RationalCleanup<C>, SciOxideError> {
        validate_count(nodes.len(), 1, scioxide_core::argument!("nodes"))?;
        validate_length(nodes.len(), values.len())?;
        validate_finite(values)?;
        for &z in nodes {
            self.domain.resolve(
                z,
                ExtrapolationPolicy::Reject,
                scioxide_core::argument!("nodes"),
            )?;
        }
        let mut indices = scioxide_core::backend::try_with_capacity(
            nodes.len(),
            scioxide_core::quantity!("cleanup_node_indices"),
        )?;
        super::distinct(nodes, &mut indices, "duplicate cleanup samples")?;
        let diagnostics = self.diagnostics(policy, doublet, eigen_options)?;
        let supports = &self.support.nodes;
        // Following the AAA cleanup heuristic (Nakatsukasa, Sète & Trefethen),
        // each suspected doublet is attributed to the support point closest to
        // its pole; those supports are dropped and the weights are refitted
        // once on the remaining supports (no further greedy steps).
        let mut removed = try_filled(
            supports.len(),
            false,
            scioxide_core::quantity!("doublet_cleanup_mask"),
        )?;
        for pole in diagnostics
            .poles
            .iter()
            .filter(|pole| pole.suspected_doublet)
        {
            if let Some(index) = supports
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (complex(**a) - pole.location)
                        .abs()
                        .partial_cmp(&(complex(**b) - pole.location).abs())
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                removed[index] = true;
            }
        }
        let mut support =
            try_with_capacity(supports.len(), scioxide_core::quantity!("cleaned_supports"))?;
        let mut support_values =
            try_with_capacity(supports.len(), scioxide_core::quantity!("cleaned_values"))?;
        for (i, &s) in supports.iter().enumerate() {
            if !removed[i] {
                let index = nodes.iter().position(|&z| z == s).ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("nodes"),
                        "cleanup samples omit a support point",
                    )
                })?;
                support.push(s);
                support_values.push(values[index]);
            }
        }
        let count = support.len();
        if count == 0 {
            return Err(instability(
                scioxide_core::NumericalIssue::Breakdown,
                "doublet cleanup would remove every support",
            ));
        }
        let removed_supports = supports.len() - count;
        let before = sample_error(self, nodes, values)?;
        if removed_supports == 0 {
            return Ok(RationalCleanup {
                approximant: BarycentricRational::from_parts(
                    Support {
                        nodes: try_copy_slice(supports, scioxide_core::quantity!("cleanup_nodes"))?,
                        numerator: try_copy_slice(
                            &self.support.numerator,
                            scioxide_core::quantity!("cleanup_numerator"),
                        )?,
                        denominator: try_copy_slice(
                            &self.support.denominator,
                            scioxide_core::quantity!("cleanup_weights"),
                        )?,
                    },
                    self.domain,
                )?,
                removed_supports,
                before_residual: before,
                after_residual: before,
            });
        }
        let scale = values
            .iter()
            .fold(C::Real::ZERO, |s, v| s.max(v.abs()))
            .max(C::Real::MIN_POSITIVE);
        let mut distance = C::Real::MAX;
        for &z in nodes {
            if !support.contains(&z) {
                for &s in &support {
                    distance = distance.min(intermediate(z - s)?.abs());
                }
            }
        }
        // Same scaled Loewner system as an AAA step: common value and distance
        // factors bound the entries without changing the null vector, rows are
        // padded to at least `count`, and the last right singular vector gives
        // the interpolatory weights.
        let mut matrix = Matrix::zeros(nodes.len().max(count), count)?;
        for (i, (&z, &f)) in nodes.iter().zip(values).enumerate() {
            if support.contains(&z) {
                continue;
            }
            for (j, (&s, &fs)) in support.iter().zip(&support_values).enumerate() {
                matrix.view_mut().set(
                    i,
                    j,
                    intermediate(
                        (f / C::from_real(scale) - fs / C::from_real(scale))
                            * (C::from_real(distance) / intermediate(z - s)?),
                    )?,
                )?;
            }
        }
        let svd = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)?;
        let mut weights = try_with_capacity(count, scioxide_core::quantity!("cleaned_weights"))?;
        let mut numerator =
            try_with_capacity(count, scioxide_core::quantity!("cleaned_numerator"))?;
        for (j, &f) in support_values.iter().enumerate() {
            let weight = svd.vt_matrix().view().get(count - 1, j)?.conj();
            weights.push(weight);
            numerator.push(intermediate(weight * f)?);
        }
        let approximant = BarycentricRational::new(support, numerator, weights, self.domain)?;
        let after = sample_error(&approximant, nodes, values)?;
        Ok(RationalCleanup {
            approximant,
            removed_supports,
            before_residual: before,
            after_residual: after,
        })
    }
}

/// Converts the barycentric form to polynomials in the local coordinate.
fn diagnostic_polynomials<C: NumericScalar>(
    support: &Support<C>,
    policy: PolynomialApproximationPolicy<C::Real>,
) -> Result<DiagnosticPolynomials<C::Real>, SciOxideError>
where
    Complex<C::Real>: NumericScalar<Real = C::Real>,
{
    // Convert the barycentric form to polynomials by multiplying numerator
    // and denominator by l(z) = prod_i (z - z_i):
    //   p(z) = sum_j a_j prod_{i != j} (z - z_i),  q likewise with b_j.
    // Nodes are first mapped to u = (z - z_0)/scale, which places every
    // node in the closed unit disc so the monomial coefficients of the
    // node products stay bounded, and the weights are divided by their
    // common maximum (a factor that cancels in p/q).
    let center = complex(support.nodes[0]);
    let scale = support.nodes.iter().try_fold(C::Real::ZERO, |s, &z| {
        Ok::<_, SciOxideError>(s.max(intermediate(complex(z) - center)?.abs()))
    })?;
    let scale = if scale == C::Real::ZERO {
        C::Real::ONE
    } else {
        scale
    };
    let mut nodes = try_with_capacity(
        support.nodes.len(),
        scioxide_core::quantity!("rational_diagnostic_nodes"),
    )?;
    for &node in &support.nodes {
        nodes.push(intermediate(
            (complex(node) - center) / Complex::new(scale, C::Real::ZERO),
        )?);
    }
    let zero = Complex::new(C::Real::ZERO, C::Real::ZERO);
    let mut p = ApproxPolynomial::monomial_with_options(
        try_filled(1, zero, scioxide_core::quantity!("rational_numerator"))?,
        policy,
    )?;
    let mut q = ApproxPolynomial::monomial_with_options(
        try_filled(1, zero, scioxide_core::quantity!("rational_denominator"))?,
        policy,
    )?;
    let weight_scale = support
        .numerator
        .iter()
        .chain(&support.denominator)
        .fold(C::Real::ZERO, |s, v| s.max(v.abs()));
    let mut omitted = try_with_capacity(
        nodes.len().saturating_sub(1),
        scioxide_core::quantity!("rational_product_roots"),
    )?;
    for j in 0..nodes.len() {
        omitted.clear();
        omitted.extend(
            nodes
                .iter()
                .enumerate()
                .filter_map(|(i, &z)| (i != j).then_some(z)),
        );
        let product = ApproxPolynomial::from_roots_with_options(&omitted, policy)?;
        let a = ApproxPolynomial::monomial_with_options(
            try_filled(
                1,
                complex(support.numerator[j]) / Complex::new(weight_scale, C::Real::ZERO),
                scioxide_core::quantity!("rational_coefficient"),
            )?,
            policy,
        )?;
        let b = ApproxPolynomial::monomial_with_options(
            try_filled(
                1,
                complex(support.denominator[j]) / Complex::new(weight_scale, C::Real::ZERO),
                scioxide_core::quantity!("rational_coefficient"),
            )?,
            policy,
        )?;
        p = p.try_add(&product.try_mul(&a)?)?;
        q = q.try_add(&product.try_mul(&b)?)?;
    }
    if q.is_zero() {
        return Err(instability(
            scioxide_core::NumericalIssue::Breakdown,
            "diagnostic denominator cancelled to zero",
        ));
    }
    Ok((p, q, center, scale))
}

/// Drops negligible leading coefficients.
fn trim_diagnostic<C: NumericScalar>(
    polynomial: Polynomial<C>,
) -> Result<(Polynomial<C>, usize), SciOxideError> {
    let coefficients = polynomial.basis().coefficients();
    let scale = coefficients
        .iter()
        .fold(C::Real::ZERO, |s, c| s.max(c.abs()));
    // Negligible leading (highest-degree) coefficients would put companion
    // eigenvalues near infinity; dropping them lowers the degree used for root
    // extraction only. At least the constant term is always kept.
    let threshold = polynomial.policy().tolerance().threshold(scale)?;
    let mut len = coefficients.len();
    while len > 1 && coefficients[len - 1].abs() <= threshold {
        len -= 1;
    }
    let trimmed = coefficients.len() - len;
    Ok((
        ApproxPolynomial::monomial_with_options(
            try_copy_slice(
                &coefficients[..len],
                scioxide_core::quantity!("trimmed_coefficients"),
            )?,
            polynomial.policy(),
        )?,
        trimmed,
    ))
}

fn polynomial_roots<R: RealScalar>(
    p: &Polynomial<Complex<R>>,
    options: &GeneralEigenOptions<R>,
) -> Result<Vec<Complex<R>>, SciOxideError>
where
    Complex<R>: NumericScalar<Real = R>,
{
    if p.degree() == 0 {
        return Ok(Vec::new());
    }
    Ok(
        scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(
            &p.root_matrix()?,
            options,
        )?
        .into_eigenvalues(),
    )
}

/// Returns the maximum error of `plan` at the samples `nodes`.
fn sample_error<C: NumericScalar>(
    plan: &BarycentricRational<C>,
    nodes: &[C],
    values: &[C],
) -> Result<C::Real, SciOxideError> {
    nodes
        .iter()
        .zip(values)
        .try_fold(C::Real::ZERO, |max, (&z, &f)| {
            let z = plan.domain.resolve(
                z,
                ExtrapolationPolicy::Reject,
                scioxide_core::argument!("nodes"),
            )?;
            let value = plan.support.evaluate(z)?.0;
            Ok(max.max(crate::checked::output(intermediate(value - f)?.abs())?))
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::RationalDomain;
    use scioxide_core::Interval;
    use scioxide_linalg::RankPolicy;

    #[test]
    fn cleanup_rejects_samples_missing_a_support_point() {
        let plan = BarycentricRational::<f64>::new(
            vec![-1., 1.],
            vec![-1., 1.],
            vec![1., 1.],
            RationalDomain::Real(Interval::new(-1., 1.).unwrap()),
        )
        .unwrap();
        let policy = PolynomialApproximationPolicy::new(
            Tolerance::new(1e-13, 1e-13).unwrap(),
            RankPolicy::DimensionScaledEpsilon,
        );
        let tolerance = Tolerance::new(1e-6, 1e-6).unwrap();
        assert!(
            plan.cleanup_doublets(&[0.0], &[0.0], policy, tolerance, &Default::default())
                .is_err()
        );
        assert!(
            plan.cleanup_doublets(&[], &[], policy, tolerance, &Default::default())
                .is_err()
        );
    }
}
