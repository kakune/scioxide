//! Fixed cubature rules on simplices and sphere surfaces.

const ALGORITHM: scioxide_core::AlgorithmId =
    scioxide_core::algorithm_id!("integrate", "geometric-cubature", 1);

use alloc::{sync::Arc, vec::Vec};
use scioxide_core::NumericScalar;

use scioxide_core::Scaled;
use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Budget, BudgetKind, ExecutionContext,
    ExecutionPolicy, InvalidInputReason, NonFiniteOrigin, PartialResultTrust, RealScalar,
    SciOxideError, SolveContext, SolveScope, check_value_shape, write_value,
};
use scioxide_core::{PromotingCompensatedSum, RangeAccumulator};
use scioxide_linalg::{LinalgWorkspace, LuDecomposition, Matrix};
use scioxide_poly::OrthogonalFamily;
use scioxide_spatial::{PredicateSign, SimplexMesh, SimplexPredicateWorkspace};

use crate::{GaussianRulePlan, GaussianRuleVariant, IntegrationReport, IntegrationTermination};

use super::lebedev_data;

fn simplex_volume<S: RealScalar>(dimension: usize) -> Result<S, SciOxideError> {
    (1..=dimension).try_fold(S::ONE, |volume, i| Ok(volume / S::try_from_usize_exact(i)?))
}

fn compositions(
    beta: &mut [usize],
    axis: usize,
    remaining: usize,
    emit: &mut impl FnMut(&[usize]) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    if axis + 1 == beta.len() {
        beta[axis] = remaining;
        return emit(beta);
    }
    for value in 0..=remaining {
        beta[axis] = value;
        compositions(beta, axis + 1, remaining - value, emit)?;
    }
    Ok::<(), SciOxideError>(())
}

struct SimplexData<S: RealScalar> {
    dimension: usize,
    degree: usize,
    reference_measure: S,
    barycentric: Vec<S>,
    weights: Vec<S>,
}

/// Fixed cubature rule on a `d`-simplex.
///
/// The rule is stored on the reference simplex
/// `T = {x ∈ ℝ^d : xᵢ ≥ 0, x₁ + … + x_d ≤ 1}` of volume `1/d!`, with nodes in
/// barycentric coordinates `(λ₀, …, λ_d)`. For a simplex with vertices
/// `v₀, …, v_d` it maps node `j` to `xⱼ = Σₖ λⱼₖ vₖ` and computes
///
/// ```text
/// ∫_S f(x) dx ≈ |det J| Σⱼ wⱼ f(xⱼ),   J = [v₁ − v₀, …, v_d − v₀],
/// ```
///
/// where `wⱼ` are the reference weights. The result is exact, up to rounding,
/// for polynomials of total degree at most [`SimplexRule::exactness_degree`].
/// It has no error estimate.
///
/// - [`SimplexRule::positive`] gives rules with positive weights and nodes
///   strictly inside the simplex, for `d ≤ 3` and degree `≤ 7`.
/// - [`SimplexRule::new`] gives the Grundmann–Möller rule of any dimension and
///   odd degree. Its weights alternate in sign.
///
/// Preparing a rule with `n` nodes takes `O(n d)` time and storage. Mapping it
/// to a simplex takes `O(d³)` for the Jacobian determinant plus `O(n d²)` for
/// the nodes. The exact orientation test needs `(d + 1)!` terms of
/// [`GeometricOptions::predicate_terms`]. A simplex whose vertices are exactly
/// affinely dependent gives zero with
/// [`IntegrationTermination::DegenerateDomain`] and no evaluations.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{GeometricOptions, SimplexRule};
///
/// // Grundmann–Möller rule of degree 3 on a triangle.
/// let rule = SimplexRule::<f64>::new(2, 1, GeometricOptions::default())?;
/// let vertices: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
/// let (value, report) = rule.integrate(&|x: &[f64]| Ok(x[0] * x[1]), &vertices)?;
/// assert!(report.is_success());
/// // ∫ x y over the unit triangle is 1/24.
/// assert!((value - 1.0 / 24.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Grundmann and H. M. Möller, "Invariant integration formulas for the
///   n-simplex by combinatorial methods", *SIAM J. Numer. Anal.* 15(2),
///   282–290, 1978. <https://doi.org/10.1137/0715019>
/// - H. Xiao and Z. Gimbutas, "A numerical algorithm for the construction of
///   efficient quadrature rules in two and higher dimensions", *Comput. Math.
///   Appl.* 59(2), 663–676, 2010. <https://doi.org/10.1016/j.camwa.2009.10.027>
#[derive(Clone)]
#[must_use]
pub struct SimplexRule<S: RealScalar> {
    data: Arc<SimplexData<S>>,
    options: GeometricOptions,
}

impl<S: RealScalar> core::fmt::Debug for SimplexRule<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SimplexRule")
            .field("dimension", &self.dimension())
            .field("degree", &self.exactness_degree())
            .field("nodes", &self.node_count())
            .finish()
    }
}

impl<S: RealScalar> SimplexRule<S> {
    /// Prepares a rule with positive weights and interior nodes, exact to at
    /// least `minimum_degree`.
    ///
    /// Dimensions 1 to 3 and degrees 0 to 7 are supported. Segments use the
    /// Gauss–Legendre rule with the fewest nodes. Triangles and tetrahedra use
    /// the tables of Xiao and Gimbutas, rounded from `f64` to `S`; a triangle
    /// rule of degree 3 is the 6-point rule of degree 4. The node counts are:
    ///
    /// | degree | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    /// |---|---|---|---|---|---|---|---|---|
    /// | segment | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 |
    /// | triangle | 1 | 1 | 3 | 6 | 6 | 7 | 12 | 15 |
    /// | tetrahedron | 1 | 1 | 4 | 6 | 11 | 14 | 23 | 31 |
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `dimension` is not 1, 2 or 3, if
    ///   `minimum_degree > 7`, if `(dimension + 1)!` exceeds
    ///   [`GeometricOptions::predicate_terms`], or if the node count exceeds
    ///   [`GeometricOptions::max_nodes`].
    /// - The errors of [`GaussianRulePlan::new`] for segments.
    /// - `AllocationFailure` if the rule cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_integrate::{GeometricOptions, SimplexRule};
    ///
    /// let rule = SimplexRule::<f64>::positive(3, 4, GeometricOptions::default())?;
    /// assert_eq!(rule.node_count(), 11);
    /// assert!(rule.weights().iter().all(|w| *w > 0.0));
    /// // The reference weights sum to the volume 1/3! of the reference tetrahedron.
    /// assert!((rule.weights().iter().sum::<f64>() - 1.0 / 6.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn positive(
        dimension: usize,
        minimum_degree: usize,
        options: GeometricOptions,
    ) -> Result<Self, SciOxideError> {
        use super::positive_simplex_data as data;
        if !(1..=3).contains(&dimension) || minimum_degree > 7 {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("rule"),
                InvalidInputReason::OutOfRange,
                "positive simplex rules require dimension 1–3 and degree 0–7",
            ));
        }
        let required_terms = (1..=dimension + 1).product::<usize>();
        if required_terms > options.predicate_terms.get() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("predicate_terms"),
                InvalidInputReason::OutOfRange,
                "exact orientation exceeds the prepared predicate term budget",
            ));
        }
        let degree = minimum_degree.max(1);
        let degree = if dimension == 2 && degree == 3 {
            4
        } else {
            degree
        };
        let count = match dimension {
            1 => (degree + 2) / 2,
            2 => [0, 1, 3, 6, 6, 7, 12, 15][degree],
            _ => [0, 1, 4, 6, 11, 14, 23, 31][degree],
        };
        if count > options.max_nodes.get() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("max_nodes"),
                InvalidInputReason::OutOfRange,
                "positive simplex node limit exceeded",
            ));
        }
        let mut barycentric = try_with_capacity(
            count * (dimension + 1),
            scioxide_core::quantity!("barycentric_nodes"),
        )?;
        let mut weights = try_with_capacity(count, scioxide_core::quantity!("weights"))?;
        if dimension == 1 {
            let rule = GaussianRulePlan::<S>::with_options(
                OrthogonalFamily::Legendre,
                count,
                GaussianRuleVariant::Gauss,
                crate::GaussianRuleOptions::default(),
            )?;
            for (&x, &w) in rule.nodes().iter().zip(rule.weights()) {
                let x = (S::ONE + x) * S::HALF;
                barycentric.extend([S::ONE - x, x]);
                weights.push(w * S::HALF);
            }
        } else {
            let entries: &[f64] = match (dimension, degree) {
                (2, 1) => data::D2_1.as_flattened(),
                (2, 2) => data::D2_2.as_flattened(),
                (2, 4) => data::D2_4.as_flattened(),
                (2, 5) => data::D2_5.as_flattened(),
                (2, 6) => data::D2_6.as_flattened(),
                (2, 7) => data::D2_7.as_flattened(),
                (3, 1) => data::D3_1.as_flattened(),
                (3, 2) => data::D3_2.as_flattened(),
                (3, 3) => data::D3_3.as_flattened(),
                (3, 4) => data::D3_4.as_flattened(),
                (3, 5) => data::D3_5.as_flattened(),
                (3, 6) => data::D3_6.as_flattened(),
                _ => data::D3_7.as_flattened(),
            };
            for node in entries.chunks_exact(dimension + 2) {
                for &x in &node[..dimension + 1] {
                    barycentric.push(S::try_from_f64_lossy(x)?);
                }
                weights.push(S::try_from_f64_lossy(node[dimension + 1])?);
            }
        }
        Ok(Self {
            options,
            data: Arc::new(SimplexData {
                dimension,
                degree: if dimension == 1 {
                    2 * count - 1
                } else {
                    degree
                },
                reference_measure: simplex_volume(dimension)?,
                barycentric,
                weights,
            }),
        })
    }

    /// Prepares the Grundmann–Möller rule of index `s` on the `d`-simplex.
    ///
    /// The rule has degree `2s + 1` and `C(d + s + 1, s)` nodes. For
    /// `i = 0, …, s` and every `β ∈ ℕ^(d+1)` with `|β| = s − i`, it has the node
    /// and reference weight
    ///
    /// ```text
    /// λₖ = (2βₖ + 1) / (d + 2s + 1 − 2i),   k = 0, …, d,
    /// wᵢ = (−1)^i (d + 2s + 1 − 2i)^(2s+1) / (2^(2s) i! (d + 2s + 1 − i)!).
    /// ```
    ///
    /// The weights alternate in sign, so rounding errors grow with `s`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `dimension` is zero.
    /// - `InvalidInput` (`OutOfRange`) if `(d + 1)!` exceeds
    ///   [`GeometricOptions::predicate_terms`] or the node count exceeds
    ///   [`GeometricOptions::max_nodes`].
    /// - `SizeOverflow` if the degree, node count or factorial overflows `usize`.
    /// - `NumericalInstability` (`Underflow`) if `1/d!` or a weight rounds to
    ///   zero in `S`.
    /// - `AllocationFailure` if the rule cannot be stored.
    ///
    /// # References
    ///
    /// A. Grundmann and H. M. Möller, "Invariant integration formulas for the
    /// n-simplex by combinatorial methods", *SIAM J. Numer. Anal.* 15(2),
    /// 282–290, 1978. <https://doi.org/10.1137/0715019>
    pub fn new(
        dimension: usize,
        index: usize,
        options: GeometricOptions,
    ) -> Result<Self, SciOxideError> {
        let max_nodes = options.max_nodes;
        if dimension == 0 {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "simplex dimension must be positive",
            ));
        }
        let predicate_order = dimension.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("simplex_rule_size"))
        })?;
        let required_terms = (1..=predicate_order).try_fold(1usize, |terms, factor| {
            checked_mul(terms, factor, scioxide_core::quantity!("predicate_terms"))
        })?;
        if required_terms > options.predicate_terms.get() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("predicate_terms"),
                InvalidInputReason::OutOfRange,
                "exact orientation exceeds the prepared predicate term budget",
            ));
        }
        let degree = checked_mul(index, 2, scioxide_core::quantity!("exactness_degree"))?
            .checked_add(1)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("simplex_rule_size"))
            })?;
        let count = scioxide_core::checked_binomial(
            dimension
                .checked_add(index)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("simplex_rule_size"))
                })?,
            index,
        )?;
        if count > max_nodes.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("max_nodes"),
                InvalidInputReason::OutOfRange,
                "simplex node limit exceeded",
            ));
        }
        let barycentric_count = checked_mul(
            count,
            dimension.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("simplex_rule_size"))
            })?,
            scioxide_core::quantity!("barycentric_nodes"),
        )?;
        let mut barycentric = try_with_capacity(
            barycentric_count,
            scioxide_core::quantity!("barycentric_nodes"),
        )?;
        let mut weights = try_with_capacity(count, scioxide_core::quantity!("weights"))?;
        let mut beta = try_filled(
            dimension + 1,
            0usize,
            scioxide_core::quantity!("composition_parts"),
        )?;
        let reference_measure = simplex_volume::<S>(dimension)?;
        if reference_measure == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "reference simplex measure is unrepresentable",
            ));
        }
        // Grundmann-Moller rule of index s = `index` (degree 2s + 1) on the d-simplex:
        // for i = 0..=s and every composition beta of s - i into d + 1 parts,
        //     node (barycentric) = (2 beta_k + 1) / (d + 2s + 1 - 2i),
        //     w_i = (-1)^i 2^(-2s) (d + 2s + 1 - 2i)^(2s+1) d! / (i! (d + 2s + 1 - i)!) * |T|,
        // with |T| = 1/d! the reference measure. The factors of the weight are interleaved
        // in one loop over j so that large powers and factorials are cancelled as they are
        // built and the intermediate stays in f64 range. Weights alternate in sign, so high
        // indices amplify rounding; `max_nodes` bounds the admissible index.
        for i in 0..=index {
            let denominator = degree
                .checked_add(dimension)
                .and_then(|n| n.checked_sub(2 * i))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("simplex_rule_size"))
                })?;
            let mut weight = if i % 2 == 0 { 1.0 } else { -1.0 };
            let maximum = dimension.max(degree).max(degree + dimension - i);
            let numerator = denominator as f64;
            for j in 1..=maximum {
                if j <= dimension {
                    weight *= j as f64;
                }
                if j <= degree {
                    weight *= numerator;
                }
                if j <= 2 * index {
                    weight /= 2.0;
                }
                if j <= i {
                    weight /= j as f64;
                }
                if j <= degree + dimension - i {
                    weight /= j as f64;
                }
            }
            weight *= reference_measure.try_to_f64_exact()?;
            let weight = S::try_from_f64_lossy(weight)?;
            if weight == S::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "simplex rule weight is unrepresentable",
                ));
            }
            let remaining = index - i;
            compositions(&mut beta, 0, remaining, &mut |composition| {
                for &part in composition {
                    let numerator = part
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(1))
                        .ok_or_else(|| {
                            SciOxideError::size_overflow(scioxide_core::quantity!(
                                "simplex_rule_size"
                            ))
                        })?;
                    barycentric.push(S::try_from_f64_lossy(
                        numerator as f64 / denominator as f64,
                    )?);
                }
                weights.push(weight);
                Ok::<(), SciOxideError>(())
            })?;
        }
        if weights.len() != count || barycentric.len() != barycentric_count {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("index"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "simplex rule cardinality mismatch",
            ));
        }
        Ok(Self {
            options,
            data: Arc::new(SimplexData {
                dimension,
                degree,
                reference_measure,
                barycentric,
                weights,
            }),
        })
    }

    /// Returns the dimension `d` of the simplex.
    pub fn dimension(&self) -> usize {
        self.data.dimension
    }
    /// Returns the highest total degree of polynomials that the rule integrates
    /// exactly in exact arithmetic.
    pub fn exactness_degree(&self) -> usize {
        self.data.degree
    }
    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.data.weights.len()
    }
    /// Returns the volume `1/d!` of the reference simplex.
    pub fn reference_measure(&self) -> S {
        self.data.reference_measure
    }
    /// Returns the barycentric coordinates `(λ₀, …, λ_d)` of the nodes, `d + 1`
    /// values per node, node after node.
    pub fn barycentric_nodes(&self) -> &[S] {
        &self.data.barycentric
    }
    /// Returns the weights on the reference simplex.
    ///
    /// They sum to `1/d!` up to rounding. Grundmann–Möller weights can be
    /// negative.
    pub fn weights(&self) -> &[S] {
        &self.data.weights
    }

    /// Allocates a workspace for an integrand with `components` real components.
    ///
    /// The workspace includes storage for the exact orientation test with
    /// [`GeometricOptions::predicate_terms`] terms and an LU factorization of
    /// the `d × d` Jacobian.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, components: usize) -> Result<GeometricWorkspace<S>, SciOxideError> {
        GeometricWorkspace::allocate(&self.workspace_requirements(components)?)
    }
}

struct SphereData<S: RealScalar> {
    degree: usize,
    nodes: Vec<S>,
    weights: Vec<S>,
}

/// Product rule on a sphere surface: Gauss–Legendre in `cos θ`, trapezoidal in
/// the azimuth `φ`.
///
/// With `p` polar and `m` azimuthal nodes, the rule on the unit sphere has the
/// `p m` nodes and weights
///
/// ```text
/// uⱼₖ = (sin θⱼ cos φₖ, sin θⱼ sin φₖ, cos θⱼ),   wⱼₖ = 2π aⱼ / m,
/// ```
///
/// where `cos θⱼ` and `aⱼ` are the Gauss–Legendre nodes and weights on
/// `[−1, 1]` and `φₖ = 2πk/m`. It integrates spherical polynomials of degree
/// up to `min(2p − 1, m − 1)` exactly. On a sphere of center `c` and radius `r`
/// it computes `∮ f dS ≈ r² Σ wⱼₖ f(c + r uⱼₖ)`. It has no error estimate.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{GeometricOptions, SphereProductRule};
///
/// let rule = SphereProductRule::<f64>::new(4, 8, GeometricOptions::default())?;
/// let (value, report) = rule.integrate(&|x: &[f64]| Ok(x[2] * x[2]), &[0.0; 3], 1.0)?;
/// assert!(report.is_success());
/// // ∮ z² dS over the unit sphere is 4π/3.
/// assert!((value - 4.0 * core::f64::consts::PI / 3.0).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. H. Stroud, *Approximate Calculation of Multiple Integrals*,
/// Prentice-Hall, 1971.
#[derive(Clone)]
#[must_use]
pub struct SphereProductRule<S: RealScalar> {
    data: Arc<SphereData<S>>,
    options: GeometricOptions,
}

impl<S: RealScalar + NumericScalar<Real = S>> SphereProductRule<S> {
    /// Prepares the rule with `polar_nodes` Gauss–Legendre nodes in `cos θ` and
    /// `azimuth_nodes` equally spaced azimuths.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if either count is zero.
    /// - `SizeOverflow` if `polar_nodes · azimuth_nodes` overflows `usize`.
    /// - `InvalidInput` (`OutOfRange`) if the node count exceeds
    ///   [`GeometricOptions::max_nodes`].
    /// - The errors of [`GaussianRulePlan::new`] for the polar nodes.
    /// - `AllocationFailure` if the rule cannot be stored.
    pub fn new(
        polar_nodes: usize,
        azimuth_nodes: usize,
        options: GeometricOptions,
    ) -> Result<Self, SciOxideError> {
        let max_nodes = options.max_nodes;
        for (nodes, argument) in [
            (polar_nodes, scioxide_core::argument!("polar_nodes")),
            (azimuth_nodes, scioxide_core::argument!("azimuth_nodes")),
        ] {
            if nodes == 0 {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "sphere node counts must be positive",
                ));
            }
        }
        let count = checked_mul(
            polar_nodes,
            azimuth_nodes,
            scioxide_core::quantity!("sphere_nodes"),
        )?;
        if count > max_nodes.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("max_nodes"),
                InvalidInputReason::OutOfRange,
                "sphere node limit exceeded",
            ));
        }
        let gaussian = GaussianRulePlan::with_options(
            OrthogonalFamily::Legendre,
            polar_nodes,
            GaussianRuleVariant::Gauss,
            crate::GaussianRuleOptions::default(),
        )?;
        let mut nodes = try_with_capacity(
            checked_mul(count, 3, scioxide_core::quantity!("node_coordinates"))?,
            scioxide_core::quantity!("node_coordinates"),
        )?;
        let mut weights = try_with_capacity(count, scioxide_core::quantity!("weights"))?;
        // With z = cos(theta), the surface element is dz dphi, so Gauss-Legendre in z times
        // the equally spaced (trapezoidal) rule in phi integrates spherical polynomials of
        // degree <= min(2p - 1, n - 1) exactly. The clamp keeps sqrt(1 - z^2) real when
        // rounding pushes z^2 marginally above one.
        let step = S::TAU / S::try_from_usize_exact(azimuth_nodes)?;
        for (&z, &polar_weight) in gaussian.nodes().iter().zip(gaussian.weights()) {
            let radial = (S::ONE - z * z).max(S::ZERO).sqrt();
            for k in 0..azimuth_nodes {
                let phase = step * S::try_from_usize_exact(k)?;
                let (sin, cos) = phase.sin_cos();
                nodes.extend_from_slice(&[radial * cos, radial * sin, z]);
                weights.push(polar_weight * step);
            }
        }
        let polar_degree =
            checked_mul(polar_nodes, 2, scioxide_core::quantity!("exactness_degree"))? - 1;
        Ok(Self {
            options,
            data: Arc::new(SphereData {
                degree: polar_degree.min(azimuth_nodes - 1),
                nodes,
                weights,
            }),
        })
    }
}

impl<S: RealScalar> SphereProductRule<S> {
    /// Returns the highest degree of spherical polynomials that the rule
    /// integrates exactly in exact arithmetic.
    pub fn exactness_degree(&self) -> usize {
        self.data.degree
    }
    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.data.weights.len()
    }
    /// Returns the Cartesian coordinates `(x, y, z)` of the nodes on the unit
    /// sphere, three values per node, node after node.
    pub fn nodes(&self) -> &[S] {
        &self.data.nodes
    }
    /// Returns the weights on the unit sphere. They sum to `4π` up to rounding.
    pub fn weights(&self) -> &[S] {
        &self.data.weights
    }
    /// Allocates a workspace for an integrand with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, components: usize) -> Result<GeometricWorkspace<S>, SciOxideError> {
        GeometricWorkspace::allocate(&self.workspace_requirements(components)?)
    }
}

/// Lebedev–Laikov rule on a sphere surface.
///
/// Lebedev–Laikov rules are invariant under the 48 rotations and reflections of
/// the octahedron and integrate spherical polynomials of their degree exactly
/// with close to the fewest nodes. The 32 tabulated degrees run from 3 (6
/// nodes) to 131 (5810 nodes); see [`LebedevRule::supported_degrees`]. On a
/// sphere of center `c` and radius `r` the rule computes
/// `∮ f dS ≈ r² Σ wⱼ f(c + r uⱼ)`, where `uⱼ` are the nodes on the unit sphere
/// and the weights `wⱼ` sum to `4π`. It has no error estimate.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{GeometricOptions, LebedevRule};
///
/// let rule = LebedevRule::<f64>::new(7, GeometricOptions::default())?;
/// let (value, report) = rule.integrate(&|_: &[f64]| Ok(1.0), &[0.0; 3], 1.0)?;
/// assert!(report.is_success());
/// // The area of the unit sphere is 4π.
/// assert!((value - 4.0 * core::f64::consts::PI).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// V. I. Lebedev and D. N. Laikov, "A quadrature formula for the sphere of the
/// 131st algebraic order of accuracy", *Doklady Mathematics* 59(3), 477–481,
/// 1999.
#[derive(Clone)]
#[must_use]
pub struct LebedevRule<S: RealScalar> {
    data: Arc<SphereData<S>>,
    options: GeometricOptions,
}

impl<S: RealScalar> LebedevRule<S> {
    /// Prepares the rule of the given degree.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `degree` is not one of
    ///   [`LebedevRule::supported_degrees`].
    /// - `InvalidInput` (`OutOfRange`) if the node count exceeds
    ///   [`GeometricOptions::max_nodes`].
    /// - `AllocationFailure` if the rule cannot be stored.
    pub fn new(degree: usize, options: GeometricOptions) -> Result<Self, SciOxideError> {
        let max_nodes = options.max_nodes;
        let index = lebedev_data::ORDERS
            .iter()
            .position(|&order| order == degree)
            .ok_or_else(|| {
                crate::kernel::invalid(
                    scioxide_core::argument!("degree"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "unsupported Lebedev exactness degree",
                )
            })?;
        let count = lebedev_data::COUNTS[index];
        if count > max_nodes.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("max_nodes"),
                InvalidInputReason::OutOfRange,
                "Lebedev node limit exceeded",
            ));
        }
        let mut nodes = try_with_capacity(
            checked_mul(count, 3, scioxide_core::quantity!("node_coordinates"))?,
            scioxide_core::quantity!("node_coordinates"),
        )?;
        let mut weights = try_with_capacity(count, scioxide_core::quantity!("weights"))?;
        for orbit in lebedev_data::RULES[index] {
            expand_orbit(orbit, &mut nodes, &mut weights)?;
        }
        if weights.len() != count {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("degree"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Lebedev orbit cardinality mismatch",
            ));
        }
        Ok(Self {
            options,
            data: Arc::new(SphereData {
                degree,
                nodes,
                weights,
            }),
        })
    }
    /// Returns the supported degrees: every odd degree from 3 to 31, then 35 to
    /// 131 in steps of 6.
    pub const fn supported_degrees() -> &'static [usize; 32] {
        &lebedev_data::ORDERS
    }
    /// Returns the highest degree of spherical polynomials that the rule
    /// integrates exactly in exact arithmetic.
    pub fn exactness_degree(&self) -> usize {
        self.data.degree
    }
    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.data.weights.len()
    }
    /// Returns the Cartesian coordinates `(x, y, z)` of the nodes on the unit
    /// sphere, three values per node, node after node.
    pub fn nodes(&self) -> &[S] {
        &self.data.nodes
    }
    /// Returns the weights on the unit sphere. They sum to `4π` up to rounding.
    pub fn weights(&self) -> &[S] {
        &self.data.weights
    }
    /// Allocates a workspace for an integrand with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, components: usize) -> Result<GeometricWorkspace<S>, SciOxideError> {
        GeometricWorkspace::allocate(&self.workspace_requirements(components)?)
    }
}

fn expand_orbit<S: RealScalar>(
    orbit: &lebedev_data::Orbit,
    nodes: &mut Vec<S>,
    weights: &mut Vec<S>,
) -> Result<(), SciOxideError> {
    // Lebedev-Laikov rules are stored as octahedral orbits: a generator (types 1-6 give
    // 6, 12, 8, 24, 24 and 48 points) is expanded by all coordinate permutations and sign
    // changes. Duplicate permutations (equal coordinates) and sign flips of zero coordinates
    // are skipped so each point appears once. Tabulated weights sum to one over the sphere
    // and are scaled by the unit-sphere area 4 pi.
    let a = orbit.a;
    let b = orbit.b;
    let base = match orbit.kind {
        1 => [1.0, 0.0, 0.0],
        2 => [
            0.0,
            core::f64::consts::FRAC_1_SQRT_2,
            core::f64::consts::FRAC_1_SQRT_2,
        ],
        3 => {
            let v = 1.0 / 3.0_f64.sqrt();
            [v, v, v]
        }
        4 => [a, a, (1.0 - 2.0 * a * a).sqrt()],
        5 => [a, (1.0 - a * a).sqrt(), 0.0],
        6 => [a, b, (1.0 - a * a - b * b).sqrt()],
        _ => {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("degree"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid Lebedev orbit type",
            ));
        }
    };
    if base.iter().any(|x| !x.is_finite()) || !orbit.weight.is_finite() {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("degree"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid Lebedev orbit coefficient",
        ));
    }
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for (p_index, permutation) in permutations.iter().enumerate() {
        let ordered = [
            base[permutation[0]],
            base[permutation[1]],
            base[permutation[2]],
        ];
        if permutations[..p_index]
            .iter()
            .any(|prior| [base[prior[0]], base[prior[1]], base[prior[2]]] == ordered)
        {
            continue;
        }
        for mask in 0..8 {
            if (0..3).any(|axis| ordered[axis] == 0.0 && mask & (1 << axis) != 0) {
                continue;
            }
            for (axis, &value) in ordered.iter().enumerate() {
                let signed = if mask & (1 << axis) == 0 {
                    value
                } else {
                    -value
                };
                nodes.push(S::try_from_f64_lossy(signed)?);
            }
            weights.push(S::try_from_f64_lossy(
                4.0 * core::f64::consts::PI * orbit.weight,
            )?);
        }
    }
    Ok::<(), SciOxideError>(())
}

/// Volume and orientation of a simplex, returned by the simplex mapping
/// methods.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SimplexMeasure<S: RealScalar> {
    /// The `d`-dimensional volume `|det J| / d!`, or zero for a flat simplex.
    pub measure: S,
    /// Sign of `det(v₁ − v₀, …, v_d − v₀)`, computed exactly from the vertex
    /// coordinates.
    pub orientation: PredicateSign,
}

/// Workspace of a [`SimplexRule`], [`SphereProductRule`] or [`LebedevRule`]:
/// mapped nodes and weights, and integrand values.
///
/// A workspace is made for one dimension and one number of value components by
/// a rule's `workspace` method. It holds `O(n (d + c) + d²)` scalars for `n`
/// nodes, `d` dimensions and `c` components. A simplex workspace also holds the
/// exact orientation storage and an LU factorization. Any rule of the same
/// dimension and component count with at most [`Self::node_capacity`] nodes,
/// and for simplices no more predicate terms, can use it.
#[must_use]
pub struct GeometricWorkspace<S: RealScalar> {
    requirements: crate::IntegrationRequirements,
    attempts: crate::workspace::Attempts<IntegrationReport<S>>,
    mapped_nodes: usize,
    points: Vec<S>,
    weights: Vec<S>,
    values: Vec<S>,
    sums: Vec<PromotingCompensatedSum<S>>,
    candidate: Vec<S>,
    matrix: Matrix<S>,
    factor: LuDecomposition<S>,
    lu: LinalgWorkspace<'static, S>,
    predicate: Option<SimplexPredicateWorkspace>,
    mapped: bool,
}

impl<S: RealScalar> core::fmt::Debug for GeometricWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GeometricWorkspace")
            .field("dimension", &self.dimension())
            .field("nodes", &self.node_capacity())
            .field("components", &self.components())
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar> GeometricWorkspace<S> {
    fn allocate(requirements: &crate::IntegrationRequirements) -> Result<Self, SciOxideError> {
        let dimension = requirements.dimension();
        let components = requirements.components();
        let [nodes, terms] = requirements.units();
        let predicate_terms = if terms == 0 {
            None
        } else {
            Some(Budget::try_from(terms)?)
        };
        let point_count = checked_mul(dimension, nodes, scioxide_core::quantity!("mapped_points"))?;
        let value_count = checked_mul(components, nodes, scioxide_core::quantity!("node_values"))?;
        let predicate = predicate_terms
            .map(|budget| SimplexPredicateWorkspace::new(dimension, budget))
            .transpose()?;
        let mut matrix = Matrix::empty();
        let mut factor = LuDecomposition::empty();
        let mut lu = LinalgWorkspace::new();
        if predicate.is_some() {
            matrix.try_resize_flat(dimension, dimension)?;
            for i in 0..dimension {
                matrix.as_mut_slice()[i * dimension + i] = S::ONE;
            }
            lu.try_reserve(
                &scioxide_linalg::LuDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::LuWorkspaceRequest::factor(
                        dimension,
                        &scioxide_linalg::LinalgExecutionOptions::new()
                            .with_execution_policy(ExecutionPolicy::Sequential),
                    ),
                )?,
            )?;
            LuDecomposition::factor_into(
                &matrix,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut factor,
                &mut lu,
                scioxide_core::ExecutionContext::sequential(),
            )?;
        }
        Ok(Self {
            requirements: *requirements,
            attempts: crate::workspace::Attempts::default(),
            mapped_nodes: 0,
            points: try_filled(
                point_count,
                S::ZERO,
                scioxide_core::quantity!("mapped_points"),
            )?,
            weights: try_filled(nodes, S::ZERO, scioxide_core::quantity!("mapped_weights"))?,
            values: try_filled(
                value_count,
                S::ZERO,
                scioxide_core::quantity!("node_values"),
            )?,
            sums: try_filled(
                components,
                PromotingCompensatedSum::ZERO,
                scioxide_core::quantity!("component_sums"),
            )?,
            candidate: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_values"),
            )?,
            matrix,
            factor,
            lu,
            predicate,
            mapped: false,
        })
    }

    /// Returns the dimension of the points: `d` for simplices, 3 for spheres.
    pub const fn dimension(&self) -> usize {
        self.requirements.dimension()
    }
    /// Returns the largest node count the workspace holds.
    pub const fn node_capacity(&self) -> usize {
        self.requirements.units()[0]
    }
    /// Returns the number of real value components.
    pub const fn components(&self) -> usize {
        self.requirements.components()
    }
    /// Returns the mapped nodes of the last successful mapping, `d` coordinates
    /// per node, or `None` if the last mapping failed or none was made.
    ///
    /// Integration calls also map the rule. A flat simplex maps every node to
    /// the origin with zero weight.
    pub fn mapped_points(&self) -> Option<&[S]> {
        self.mapped
            .then_some(&self.points[..self.mapped_nodes * self.dimension()])
    }
    /// Returns the mapped weights of the last successful mapping, or `None` if
    /// the last mapping failed or none was made.
    pub fn mapped_weights(&self) -> Option<&[S]> {
        self.mapped.then_some(&self.weights[..self.mapped_nodes])
    }
}

fn validate_workspace<S: RealScalar>(
    workspace: &mut GeometricWorkspace<S>,
    dimension: usize,
    nodes: usize,
    predicate_terms: usize,
) -> Result<(), SciOxideError> {
    let requested = crate::IntegrationRequirements::new(
        crate::workspace::Family::Geometric,
        workspace.components(),
        [nodes, predicate_terms],
    )
    .with_dimension(dimension);
    if workspace.check(&requested)? == scioxide_core::Capacity::NeedsGrowth {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("workspace"),
            InvalidInputReason::Inconsistent,
            "insufficient geometric shape, node or predicate capacity",
        ));
    }
    workspace.mapped_nodes = nodes;
    Ok(())
}

impl<S: RealScalar> SimplexRule<S> {
    fn prepare_map(
        &self,
        vertices: &[&[S]],
        ws: &mut GeometricWorkspace<S>,
    ) -> Result<SimplexMeasure<S>, SciOxideError> {
        ws.mapped = false;
        let d = self.dimension();
        validate_workspace(ws, d, self.node_count(), self.options.predicate_terms.get())?;
        let predicate = ws.predicate.as_mut().ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("ws"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "simplex predicate workspace required",
            )
        })?;
        // Degeneracy is decided by the exact orientation predicate, not by a rounded
        // determinant, so a flat simplex reliably integrates to exactly zero (without
        // evaluating the integrand) and a nearly flat one is still integrated.
        let orientation = predicate.orient(vertices)?;
        if orientation == PredicateSign::Zero {
            ws.points.fill(S::ZERO);
            ws.weights.fill(S::ZERO);
            ws.mapped = true;
            return Ok(SimplexMeasure {
                measure: S::ZERO,
                orientation,
            });
        }
        for row in 0..d {
            for (axis, &origin) in vertices[0][..d].iter().enumerate() {
                ws.matrix.as_mut_slice()[row * d + axis] =
                    scioxide_core::checked_difference(vertices[row + 1][axis], origin)?;
            }
        }
        // The affine map x = v0 + sum_k lambda_k (v_k - v0) has Jacobian |det(V)| with the
        // edge vectors as rows; the determinant comes back as significand/exponent, so it is
        // kept scaled until multiplied into the (small) reference weights. The workspace
        // pre-factored an identity at construction so this call never allocates.
        LuDecomposition::factor_into(
            &ws.matrix,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut ws.factor,
            &mut ws.lu,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        let determinant = ws.factor.scaled_determinant()?;
        let determinant = determinant.real().abs();
        let measure = determinant
            .mul(Scaled::from_value(self.reference_measure())?)?
            .round()?;
        if measure == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "nonzero simplex measure rounds to zero",
            ));
        }
        // Physical nodes use all d + 1 barycentric weights against the vertices (rather than
        // v0 plus edge vectors), which avoids forming differences for points near a vertex.
        for (node, barycentric) in self.data.barycentric.chunks_exact(d + 1).enumerate() {
            for axis in 0..d {
                let mut sum = RangeAccumulator::<S>::new();
                for (lambda, vertex) in barycentric.iter().zip(vertices) {
                    sum.add_product(*lambda, vertex[axis])?;
                }
                ws.points[node * d + axis] = sum.finish()?;
            }
            ws.weights[node] = determinant
                .mul(Scaled::from_value(self.data.weights[node])?)?
                .try_value()?;
        }
        ws.mapped = true;
        Ok(SimplexMeasure {
            measure,
            orientation,
        })
    }

    /// Maps the rule to the simplex with vertices `vertices` and writes the
    /// nodes and weights to `points` and `weights`.
    ///
    /// `vertices` holds `d + 1` points of length `d`. `points` receives `d`
    /// coordinates per node and must have length `n d`; `weights` receives the
    /// weights `|det J| wⱼ` and must have length `n`. A flat simplex gives zero
    /// points and weights. Use this method to integrate values you compute
    /// yourself, or with [`SimplexRule::integrate_values_into`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `points` or `weights` has the wrong length.
    /// - `InvalidInput` (`Inconsistent`) if `vertices` does not hold `d + 1`
    ///   points of length `d`, or if the workspace was made for another
    ///   dimension, another rule type or fewer nodes or predicate terms.
    /// - `NonFinite` if a vertex coordinate is NaN or infinite.
    /// - `NumericalInstability` if an edge vector, mapped node, volume or weight
    ///   overflows, or a nonzero volume or weight underflows to zero.
    pub fn map_nodes_into(
        &self,
        vertices: &[&[S]],
        points: &mut [S],
        weights: &mut [S],
        ws: &mut GeometricWorkspace<S>,
    ) -> Result<SimplexMeasure<S>, SciOxideError> {
        ws.mapped = false;
        crate::kernel::shape(self.node_count() * self.dimension(), points.len())?;
        crate::kernel::shape(self.node_count(), weights.len())?;
        let measure = self.prepare_map(vertices, ws)?;
        points.copy_from_slice(&ws.points[..points.len()]);
        weights.copy_from_slice(&ws.weights[..weights.len()]);
        Ok(measure)
    }

    /// Maps the rule to cell `cell` of a triangle or tetrahedron mesh, as
    /// [`SimplexRule::map_nodes_into`] does for its vertices.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `D` differs from the rule's
    ///   dimension or is not 2 or 3.
    /// - The errors of [`SimplexMesh::cell_vertices`] for an invalid `cell`.
    /// - The errors of [`SimplexRule::map_nodes_into`].
    pub fn map_mesh_cell_into<const D: usize>(
        &self,
        mesh: &SimplexMesh<S, D>,
        cell: usize,
        points: &mut [S],
        weights: &mut [S],
        ws: &mut GeometricWorkspace<S>,
    ) -> Result<SimplexMeasure<S>, SciOxideError> {
        ws.mapped = false;
        if self.dimension() != D || !(D == 2 || D == 3) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("mesh"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "mesh adapter requires matching triangle or tetrahedron dimension",
            ));
        }
        let ids = mesh.cell_vertices(cell)?;
        let mut vertices: [&[S]; 4] = [&[]; 4];
        for (slot, &id) in vertices.iter_mut().zip(ids) {
            *slot = mesh.points()[id].as_array();
        }
        self.map_nodes_into(&vertices[..D + 1], points, weights, ws)
    }
}

fn sphere_map<S: RealScalar>(
    data: &SphereData<S>,
    center: &[S; 3],
    radius: S,
    ws: &mut GeometricWorkspace<S>,
) -> Result<S, SciOxideError> {
    ws.mapped = false;
    validate_workspace(ws, 3, data.weights.len(), 0)?;
    if center.iter().any(|x| !x.is_finite()) || !radius.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "sphere center and radius must be finite",
        ));
    }
    if radius <= S::ZERO {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("radius"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "sphere radius must be positive",
        ));
    }
    // Surface measure scales with r^2; points are centre + r * unit node, each coordinate
    // formed with a single fused range-safe accumulation.
    let scale = Scaled::from_value(radius)?.mul(Scaled::from_value(radius)?)?;
    let measure = scale
        .mul(Scaled::from_value(S::TAU + S::TAU)?)?
        .try_value()?;
    let native_scale = radius * radius;
    for (i, node) in data.nodes.chunks_exact(3).enumerate() {
        for axis in 0..3 {
            // At a zero center this is a single correctly rounded product,
            // including subnormals; an exact accumulator cannot improve it.
            if center[axis] == S::ZERO {
                ws.points[i * 3 + axis] = radius * node[axis];
                continue;
            }
            let mut sum = RangeAccumulator::<S>::new();
            sum.add(center[axis])?;
            sum.add_product(radius, node[axis])?;
            ws.points[i * 3 + axis] = sum.finish()?;
        }
        let weight = native_scale * data.weights[i];
        ws.weights[i] =
            if native_scale.is_normal() && (data.weights[i] == S::ZERO || weight.is_normal()) {
                weight
            } else {
                scale
                    .mul(Scaled::from_value(data.weights[i])?)?
                    .try_value()?
            };
    }
    ws.mapped = true;
    Ok(measure)
}

macro_rules! sphere_mapping {
    ($rule:ident) => {
        impl<S: RealScalar> $rule<S> {
            /// Maps the rule to the sphere of center `center` and radius `radius`,
            /// writes the nodes and weights to `points` and `weights`, and returns
            /// the surface area `4π r²`.
            ///
            /// `points` receives three coordinates per node and must have length
            /// `3n`; `weights` receives `r² wⱼ` and must have length `n`.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `points` or `weights` has the wrong length.
            /// - `NonFinite` if `center` or `radius` is NaN or infinite.
            /// - `InvalidInput` (`Inconsistent`) if `radius ≤ 0`, or if the
            ///   workspace was made for another dimension or fewer nodes.
            /// - `NumericalInstability` if a node, surface area or weight overflows,
            ///   or a nonzero surface area or weight underflows to zero.
            pub fn map_nodes_into(
                &self,
                center: &[S; 3],
                radius: S,
                points: &mut [S],
                weights: &mut [S],
                ws: &mut GeometricWorkspace<S>,
            ) -> Result<S, SciOxideError> {
                ws.mapped = false;
                crate::kernel::shape(self.node_count() * 3, points.len())?;
                crate::kernel::shape(self.node_count(), weights.len())?;
                let measure = sphere_map(&self.data, center, radius, ws)?;
                points.copy_from_slice(&ws.points[..points.len()]);
                weights.copy_from_slice(&ws.weights[..weights.len()]);
                Ok(measure)
            }
        }
    };
}
sphere_mapping!(SphereProductRule);
sphere_mapping!(LebedevRule);

/// Options of [`SimplexRule`], [`SphereProductRule`] and [`LebedevRule`].
///
/// | option | default |
/// |---|---|
/// | [`Self::max_nodes`] | 1,000,000 |
/// | [`Self::predicate_terms`] | 4096 |
/// | [`Self::max_evaluations`] | `None` (no limit) |
/// | [`Self::execution_policy`] | [`ExecutionPolicy::Sequential`] |
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::{GeometricOptions, LebedevRule};
///
/// let max_nodes = Budget::try_from(100)?;
/// let options = GeometricOptions::default().with_max_nodes(max_nodes);
/// // The degree-15 rule has 86 nodes and the degree-17 rule has 110.
/// assert!(LebedevRule::<f64>::new(15, options).is_ok());
/// assert!(LebedevRule::<f64>::new(17, options).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeometricOptions {
    max_nodes: Budget,
    predicate_terms: Budget,
    max_evaluations: Option<Budget>,
    policy: ExecutionPolicy,
}
impl Default for GeometricOptions {
    fn default() -> Self {
        Self {
            max_nodes: Budget::ONE.saturating_add(999_999),
            predicate_terms: Budget::ONE.saturating_add(4095),
            max_evaluations: None,
            policy: ExecutionPolicy::Sequential,
        }
    }
}
impl GeometricOptions {
    /// Sets the largest node count a rule may have.
    pub const fn with_max_nodes(mut self, value: Budget) -> Self {
        self.max_nodes = value;
        self
    }
    /// Sets the number of terms reserved for the exact orientation test of a
    /// simplex.
    ///
    /// A `d`-simplex needs `(d + 1)!` terms, so the default of 4096 allows
    /// `d ≤ 5`.
    pub const fn with_predicate_terms(mut self, value: Budget) -> Self {
        self.predicate_terms = value;
        self
    }
    /// Sets or removes the limit on integrand evaluations per call.
    ///
    /// A rule with more nodes than the limit fails with `ConvergenceFailure`
    /// before any evaluation.
    pub const fn with_max_evaluations(mut self, value: Option<Budget>) -> Self {
        self.max_evaluations = value;
        self
    }
    /// Sets how the integrand evaluations of one call are distributed over
    /// workers. The result does not depend on the policy.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.policy = value;
        self
    }
    /// Returns the largest node count a rule may have.
    pub const fn max_nodes(&self) -> Budget {
        self.max_nodes
    }
    /// Returns the number of terms reserved for the exact orientation test.
    pub const fn predicate_terms(&self) -> Budget {
        self.predicate_terms
    }
    /// Returns the limit on integrand evaluations per call, if any.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }
    /// Returns the execution policy of integrand evaluations.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
}
crate::workspace::workspace_protocol!(GeometricWorkspace, IntegrationReport<S>);

fn mapped_call<S, O>(
    options: &GeometricOptions,
    out: &mut O,
    ws: &mut GeometricWorkspace<S>,
    scope: SolveScope<'_, '_>,
    sample: impl FnOnce(
        &[S],
        usize,
        usize,
        &mut [S],
        &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>,
    values: Option<&[S]>,
) -> Result<IntegrationReport<S>, SciOxideError>
where
    S: RealScalar,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    if let Err(error) = check_value_shape(out, ws.components()).and_then(|()| match values {
        Some(rows) => {
            crate::kernel::shape(ws.mapped_nodes * ws.components(), rows.len())?;
            scioxide_core::validate_value(rows)
        }
        None => Ok::<(), SciOxideError>(()),
    }) {
        return ws.attempts.reject(error);
    }
    let mut attempts = core::mem::take(&mut ws.attempts);
    let result = attempts.integrate(
        scope,
        ALGORITHM,
        ws.dimension(),
        crate::workspace::limit(
            scioxide_core::SolveBudgets::unbounded(),
            BudgetKind::Evaluation,
            options.max_evaluations,
        ),
        |cx| {
            let degenerate = ws.weights[..ws.mapped_nodes]
                .iter()
                .all(|&weight| weight == S::ZERO);
            let count = if degenerate || ws.components() == 0 || values.is_some() {
                0
            } else {
                ws.mapped_nodes
            };
            if let Some(stop) = crate::workspace::admit(cx, count, false)? {
                return Ok(crate::workspace::Outcome::Unpublished(stop));
            }
            ws.sums.fill(PromotingCompensatedSum::ZERO);
            if !degenerate && ws.components() > 0 {
                let rows = match values {
                    Some(rows) => rows,
                    None => {
                        sample(
                            &ws.points[..ws.mapped_nodes * ws.dimension()],
                            ws.dimension(),
                            ws.components(),
                            &mut ws.values[..ws.mapped_nodes * ws.requirements.components()],
                            cx,
                        )?;
                        &ws.values[..ws.mapped_nodes * ws.components()]
                    }
                };
                for (component, sum) in ws.sums.iter_mut().enumerate() {
                    *sum = PromotingCompensatedSum::from_products(
                        ws.weights[..ws.mapped_nodes]
                            .iter()
                            .zip(rows.chunks_exact(ws.requirements.components()))
                            .map(|(&weight, row)| (weight, row[component])),
                    )?;
                }
            }
            for (sum, value) in ws.sums.iter().zip(&mut ws.candidate) {
                *value = sum.round()?;
            }
            cx.commit_accepted(PartialResultTrust::Admissible)?;
            write_value(&ws.candidate, out)?;
            Ok(crate::workspace::Outcome::Published {
                termination: if degenerate {
                    IntegrationTermination::DegenerateDomain
                } else {
                    IntegrationTermination::FixedRuleCompleted
                },
                subdivisions: 0,
                errors: None,
            })
        },
    );
    ws.attempts = attempts;
    result
}
macro_rules! geometry_protocol {
    ($name:ident,$dimension:expr,$predicate:expr) => {
        impl<S: RealScalar> $name<S> {
            /// Returns the options of the rule.
            pub const fn options(&self) -> &GeometricOptions {
                &self.options
            }
            /// Returns the identifier of the method, recorded in solve diagnostics.
            pub const fn algorithm_id(&self) -> AlgorithmId {
                ALGORITHM
            }
            /// Returns the workspace size for an integrand with `components` real
            /// components. A complex value counts as two real components.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if `n · components` overflows `usize`.
            pub fn workspace_requirements(
                &self,
                components: usize,
            ) -> Result<crate::IntegrationRequirements, SciOxideError> {
                let dimension = ($dimension)(self);
                let nodes = self.node_count();
                checked_mul(
                    nodes,
                    components,
                    scioxide_core::quantity!("geometric_values"),
                )?;
                Ok(crate::IntegrationRequirements::new(
                    crate::workspace::Family::Geometric,
                    components,
                    [
                        nodes,
                        if $predicate {
                            self.options.predicate_terms.get()
                        } else {
                            0
                        },
                    ],
                )
                .with_dimension(dimension))
            }
        }
    };
}
geometry_protocol!(SimplexRule, |s: &SimplexRule<S>| s.dimension(), true);
geometry_protocol!(SphereProductRule, |_: &SphereProductRule<S>| 3, false);
geometry_protocol!(LebedevRule, |_: &LebedevRule<S>| 3, false);
impl<S: RealScalar> SimplexRule<S> {
    /// Integrates `function` over the simplex with vertices `vertices` and
    /// writes the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// The call makes `n` evaluations and reports
    /// [`IntegrationTermination::FixedRuleCompleted`], or
    /// [`IntegrationTermination::DegenerateDomain`] with no evaluations for a
    /// flat simplex. `scope` is an [`ExecutionContext`] or a solve scope (see
    /// [Execution contexts](scioxide_core#execution-contexts)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` has a different number of components than
    ///   the workspace.
    /// - The errors of [`SimplexRule::map_nodes_into`] for `vertices` and the
    ///   workspace.
    /// - `ConvergenceFailure` if [`GeometricOptions::max_evaluations`] is less
    ///   than `n`, and `ExecutionStopped` on cancellation or a deadline.
    /// - The errors of `function`, and `NonFinite` if it returns NaN or
    ///   infinity (see [Integrands](crate#integrands)).
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        function: F,
        vertices: &[&[S]],
        out: &mut O,
        ws: &mut GeometricWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let function = &function;
        if let Err(error) = check_value_shape(out, ws.components())
            .and_then(|()| self.prepare_map(vertices, ws).map(|_| ()))
        {
            return ws.attempts.reject(error);
        }
        mapped_call(
            &self.options,
            out,
            ws,
            scope.into(),
            |points, dimension, components, rows, cx| {
                crate::evaluation::sample_nd(
                    function,
                    points,
                    dimension,
                    points.len() / dimension,
                    components,
                    rows,
                    self.options.policy,
                    cx,
                )
            },
            None,
        )
    }
    /// Integrates precomputed values over the simplex with vertices `vertices`
    /// and writes the result to `out`.
    ///
    /// `values` holds one row of `c` real components per node, in the order of
    /// [`SimplexRule::barycentric_nodes`], so its length is `n c`. Compute the
    /// node positions with [`SimplexRule::map_nodes_into`]. The call makes no
    /// evaluations.
    ///
    /// # Errors
    ///
    /// - The errors of [`SimplexRule::map_nodes_into`] for `vertices` and the
    ///   workspace.
    /// - `DimensionMismatch` if `out` has a different number of components than
    ///   the workspace or `values` has the wrong length.
    /// - `NonFinite` if a value is NaN or infinite.
    pub fn integrate_values_into<'scope, 'capability, O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        vertices: &[&[S]],
        values: &[S],
        out: &mut O,
        ws: &mut GeometricWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError> {
        if let Err(error) = self.prepare_map(vertices, ws) {
            return ws.attempts.reject(error);
        }
        mapped_call(
            &self.options,
            out,
            ws,
            scope.into(),
            |_, _, _, _, _| Ok::<(), SciOxideError>(()),
            Some(values),
        )
    }
    /// Integrates a real, complex or fixed-size vector integrand over the
    /// simplex with vertices `vertices`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Allocates a workspace and runs on
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`SimplexRule::workspace`] and
    ///   [`SimplexRule::integrate_into`].
    pub fn integrate<F, V>(
        &self,
        function: F,
        vertices: &[&[S]],
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let function = &function;
        let mut out = V::default();
        let mut ws = self.workspace(out.dimension()?)?;
        let report = self.integrate_into(
            function,
            vertices,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok((out, report))
    }
    /// Integrates `function` over cell `cell` of a triangle or tetrahedron mesh,
    /// as [`SimplexRule::integrate_into`] does for its vertices.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `D` differs from the rule's
    ///   dimension or is not 2 or 3.
    /// - The errors of [`SimplexMesh::cell_vertices`] for an invalid `cell`.
    /// - The errors of [`SimplexRule::integrate_into`].
    pub fn integrate_mesh_cell_into<'scope, 'capability, const D: usize, F, V, O>(
        &self,
        function: F,
        mesh: &SimplexMesh<S, D>,
        cell: usize,
        out: &mut O,
        ws: &mut GeometricWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let function = &function;
        if self.dimension() != D || !(D == 2 || D == 3) {
            return ws.attempts.reject(crate::kernel::invalid(
                scioxide_core::argument!("mesh"),
                InvalidInputReason::Inconsistent,
                "mesh requires matching triangle or tetrahedron dimension",
            ));
        }
        let ids = match mesh.cell_vertices(cell) {
            Ok(ids) => ids,
            Err(error) => return ws.attempts.reject(error),
        };
        let mut vertices: [&[S]; 4] = [&[]; 4];
        for (slot, &id) in vertices.iter_mut().zip(ids) {
            *slot = mesh.points()[id].as_array();
        }
        self.integrate_into(function, &vertices[..D + 1], out, ws, scope)
    }
}
macro_rules! sphere_api {
    ($name:ident) => {
        impl<S: RealScalar> $name<S> {
            /// Integrates `function` over the sphere of center `center` and radius
            /// `radius` and writes the result to `out`.
            ///
            /// Accepts closures by value; pass `&f` to reuse an integrand.
            ///
            /// The call makes `n` evaluations and reports
            /// [`IntegrationTermination::FixedRuleCompleted`]. `scope` is an
            /// [`ExecutionContext`] or a solve scope (see
            /// [Execution contexts](scioxide_core#execution-contexts)).
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `out` has a different number of components
            ///   than the workspace.
            /// - The errors of `map_nodes_into` for `center`, `radius` and the
            ///   workspace.
            /// - `ConvergenceFailure` if [`GeometricOptions::max_evaluations`] is
            ///   less than `n`, and `ExecutionStopped` on cancellation or a deadline.
            /// - The errors of `function`, and `NonFinite` if it returns NaN or
            ///   infinity (see [Integrands](crate#integrands)).
            pub fn integrate_into<'scope, 'capability, F, V, O>(
                &self,
                function: F,
                center: &[S; 3],
                radius: S,
                out: &mut O,
                ws: &mut GeometricWorkspace<S>,
                scope: impl Into<SolveScope<'scope, 'capability>>,
            ) -> Result<IntegrationReport<S>, SciOxideError>
            where
                F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
                V: AffineValue<Scale = S>,
                O: AffineValueMut<Scale = S> + ?Sized,
            {
                let function = &function;
                if let Err(error) = check_value_shape(out, ws.components())
                    .and_then(|()| sphere_map(&self.data, center, radius, ws).map(|_| ()))
                {
                    return ws.attempts.reject(error);
                }
                mapped_call(
                    &self.options,
                    out,
                    ws,
                    scope.into(),
                    |points, dimension, components, rows, cx| {
                        crate::evaluation::sample_nd(
                            function,
                            points,
                            dimension,
                            points.len() / dimension,
                            components,
                            rows,
                            self.options.policy,
                            cx,
                        )
                    },
                    None,
                )
            }
            /// Integrates a real, complex or fixed-size vector integrand over the
            /// sphere of center `center` and radius `radius`.
            ///
            /// Accepts closures by value; pass `&f` to reuse an integrand.
            ///
            /// Allocates a workspace and runs on
            /// [`ExecutionContext::sequential_native`].
            ///
            /// # Errors
            ///
            /// - The errors of the rule's `workspace` and `integrate_into`
            ///   methods.
            pub fn integrate<F, V>(
                &self,
                function: F,
                center: &[S; 3],
                radius: S,
            ) -> Result<(V, IntegrationReport<S>), SciOxideError>
            where
                F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
                V: AffineValueMut<Scale = S> + Copy + Default,
            {
                let function = &function;
                let mut out = V::default();
                let mut ws = self.workspace(out.dimension()?)?;
                let report = self.integrate_into(
                    function,
                    center,
                    radius,
                    &mut out,
                    &mut ws,
                    ExecutionContext::sequential_native(),
                )?;
                Ok((out, report))
            }
        }
    };
}
sphere_api!(SphereProductRule);
sphere_api!(LebedevRule);
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use scioxide_core::{
        Complex, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };

    fn budget(n: usize) -> Budget {
        Budget::try_from(n).unwrap()
    }
    fn context_with_execution<T>(
        execution: ExecutionContext<'_>,
        budgets: SolveBudgets,
        f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
    ) -> T {
        let mut state = SolveState::new();
        let identity = SolveIdentity::new(
            ProblemToken::new(592),
            scioxide_core::algorithm_id!("integrate", "geometric-test", 1),
            0,
            0,
        );
        let mut budgets = budgets;
        let mut ctx = state
            .begin_solve(execution, identity, &mut budgets)
            .unwrap();
        f(&mut ctx)
    }
    fn context<T>(budgets: SolveBudgets, f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
        context_with_execution(ExecutionContext::sequential(), budgets, f)
    }
    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |a, b| a * b as f64)
    }

    fn positive_simplex_moments<S: RealScalar>() {
        for dimension in 1..=3 {
            for degree in 0..=7 {
                let rule =
                    SimplexRule::<S>::positive(dimension, degree, GeometricOptions::default())
                        .unwrap();
                assert!(rule.exactness_degree() >= degree);
                assert!(rule.weights().iter().all(|&w| w > S::ZERO));
                assert!(
                    rule.barycentric_nodes()
                        .iter()
                        .all(|&x| x > S::ZERO && x < S::ONE)
                );
                let counts = match dimension {
                    1 => [1, 1, 2, 2, 3, 3, 4, 4],
                    2 => [1, 1, 3, 6, 6, 7, 12, 15],
                    _ => [1, 1, 4, 6, 11, 14, 23, 31],
                };
                assert_eq!(rule.node_count(), counts[degree]);
                for total in 0..=rule.exactness_degree() {
                    let mut powers = vec![0; dimension + 1];
                    compositions(&mut powers, 0, total, &mut |powers| {
                        let actual = rule.barycentric_nodes().chunks_exact(dimension + 1)
                            .zip(rule.weights()).fold(S::ZERO, |sum, (bary, &weight)| {
                                sum + bary.iter().zip(powers).fold(weight, |v, (&x, &p)| v * x.powi(p as i32))
                            }).into_f64();
                        let expected = powers.iter().map(|&p| factorial(p)).product::<f64>()
                            / factorial(dimension + total);
                        assert!((actual - expected).abs() < 128.0 * S::EPSILON.into_f64() * expected.max(1e-3),
                            "D={dimension}, degree={degree}, powers={powers:?}, {actual} != {expected}");
                        Ok(())
                    }).unwrap();
                }
            }
        }
    }

    #[test]
    fn positive_simplex_exactness_and_interior_nodes() {
        positive_simplex_moments::<f32>();
        positive_simplex_moments::<f64>();
        for (dimension, degree) in [(0, 2), (4, 2), (2, 8)] {
            assert!(
                SimplexRule::<f64>::positive(dimension, degree, GeometricOptions::default())
                    .is_err()
            );
        }
        assert!(
            SimplexRule::<f64>::positive(
                3,
                4,
                GeometricOptions::default().with_max_nodes(budget(10))
            )
            .is_err()
        );
        assert!(
            SimplexRule::<f64>::new(2, 1, GeometricOptions::default())
                .unwrap()
                .weights()
                .iter()
                .any(|&w| w < 0.0)
        );
    }
    fn moment(d: usize, powers: &[usize]) -> f64 {
        powers.iter().map(|&p| factorial(p)).product::<f64>()
            / factorial(d + powers.iter().sum::<usize>())
    }

    #[test]
    fn simplex_monomials_both_widths_and_general_dimension() {
        for d in [1, 2, 3, 4, 5] {
            let f64_rule = SimplexRule::<f64>::new(
                d,
                1,
                GeometricOptions::default().with_max_nodes(budget(100)),
            )
            .unwrap();
            let f32_rule = SimplexRule::<f32>::new(
                d,
                1,
                GeometricOptions::default().with_max_nodes(budget(100)),
            )
            .unwrap();
            assert_eq!(f64_rule.exactness_degree(), 3);
            for p in 0..=3 {
                for q in 0..=3 - p {
                    let powers = [p, q];
                    let expected = moment(d, &powers[..if d == 1 { 1 } else { 2 }]);
                    let integrate = |bary: &[f64], weights: &[f64]| {
                        weights
                            .iter()
                            .zip(bary.chunks_exact(d + 1))
                            .map(|(&w, node)| {
                                w * node[1].powi(p as i32)
                                    * if d == 1 { 1.0 } else { node[2].powi(q as i32) }
                            })
                            .sum::<f64>()
                    };
                    if d == 1 && q != 0 {
                        continue;
                    }
                    let got = integrate(f64_rule.barycentric_nodes(), f64_rule.weights());
                    assert!(
                        (got - expected).abs() < 2e-12,
                        "d={d} p={p} q={q}: {got} vs {expected}"
                    );
                    let got32 = f32_rule
                        .weights()
                        .iter()
                        .zip(f32_rule.barycentric_nodes().chunks_exact(d + 1))
                        .map(|(&w, node)| {
                            w as f64
                                * (node[1] as f64).powi(p as i32)
                                * if d == 1 {
                                    1.0
                                } else {
                                    (node[2] as f64).powi(q as i32)
                                }
                        })
                        .sum::<f64>();
                    assert!(
                        (got32 - expected).abs() < 3e-6,
                        "f32 d={d} p={p} q={q}: {got32} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn simplex_higher_orders_and_four_dimensional_orientation() {
        for d in [2, 3] {
            for index in 0..=3 {
                let rule = SimplexRule::<f64>::new(
                    d,
                    index,
                    GeometricOptions::default().with_max_nodes(budget(1000)),
                )
                .unwrap();
                for p in 0..=rule.exactness_degree() {
                    for q in 0..=rule.exactness_degree() - p {
                        let expected = moment(d, &[p, q]);
                        let actual = rule
                            .weights()
                            .iter()
                            .zip(rule.barycentric_nodes().chunks_exact(d + 1))
                            .map(|(&w, bary)| w * bary[1].powi(p as i32) * bary[2].powi(q as i32))
                            .sum::<f64>();
                        assert!(
                            (actual - expected).abs() < 2e-11,
                            "d={d} index={index} p={p} q={q}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
        let rule =
            SimplexRule::<f64>::new(4, 1, GeometricOptions::default().with_max_nodes(budget(20)))
                .unwrap();
        let origin = [0.0; 4];
        let e1 = [1.0, 0.0, 0.0, 0.0];
        let e2 = [0.0, 1.0, 0.0, 0.0];
        let e3 = [0.0, 0.0, 1.0, 0.0];
        let e4 = [0.0, 0.0, 0.0, 1.0];
        let forward = rule
            .integrate(|_: &[f64]| Ok(1.0), &[&origin, &e1, &e2, &e3, &e4])
            .unwrap();
        let reverse = rule
            .integrate(|_: &[f64]| Ok(1.0), &[&origin, &e2, &e1, &e3, &e4])
            .unwrap();
        assert!((forward.0 - 1.0 / 24.0).abs() < 1e-14);
        assert_eq!(forward.0, reverse.0);
    }

    #[test]
    fn simplex_mapping_orientation_degeneracy_and_budgets() {
        let rule =
            SimplexRule::<f64>::new(2, 1, GeometricOptions::default().with_max_nodes(budget(8)))
                .unwrap();
        let a = [0.0, 0.0];
        let b = [2.0, 0.0];
        let c = [0.0, 3.0];
        let mut ws = rule.workspace(1).unwrap();
        let mut points = vec![0.0; rule.node_count() * 2];
        let mut weights = vec![0.0; rule.node_count()];
        let measure = rule
            .map_nodes_into(&[&a, &b, &c], &mut points, &mut weights, &mut ws)
            .unwrap();
        assert_eq!(measure.orientation, PredicateSign::Positive);
        assert!((measure.measure - 3.0).abs() < 1e-14);
        let forward = rule
            .integrate(|x: &[f64]| Ok(x[0] + x[1]), &[&a, &b, &c])
            .unwrap();
        let reverse = rule
            .integrate(|x: &[f64]| Ok(x[0] + x[1]), &[&a, &c, &b])
            .unwrap();
        assert!((forward.0 - 5.0).abs() < 1e-13);
        assert!((reverse.0 - 5.0).abs() < 1e-13);
        let degenerate = [1.0, 0.0];
        let zero = rule
            .integrate(
                |_: &[f64]| -> Result<f64, SciOxideError> {
                    Err(crate::kernel::invalid(
                        scioxide_core::argument!("function"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "unexpected callback",
                    ))
                },
                &[&a, &degenerate, &b],
            )
            .unwrap();
        assert_eq!(
            zero.1.termination(),
            IntegrationTermination::DegenerateDomain
        );
        assert_eq!(zero.1.evaluations(), 0);
        assert_eq!(zero.0, 0.0);
        let limited = context(
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(2)),
            |ctx| {
                (|| {
                    let mut value = 0.0_f64;
                    let report = rule.integrate_into(
                        |_: &[f64]| Ok(1.0),
                        &[&a, &b, &c],
                        &mut value,
                        &mut ws,
                        ctx,
                    )?;
                    Ok::<_, SciOxideError>((value, report))
                })()
                .unwrap_err()
            },
        );
        assert!(matches!(
            limited.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
    }

    #[test]
    fn sphere_product_and_all_lebedev_orders() {
        let product = SphereProductRule::<f64>::new(
            4,
            8,
            GeometricOptions::default().with_max_nodes(budget(32)),
        )
        .unwrap();
        assert_eq!(product.exactness_degree(), 7);
        let result = product
            .integrate(|p: &[f64]| Ok((p[2] - 3.0).powi(2)), &[1.0, 2.0, 3.0], 2.0)
            .unwrap();
        assert!((result.0 - 64.0 * core::f64::consts::PI / 3.0).abs() < 1e-11);
        for (&degree, &count) in LebedevRule::<f64>::supported_degrees()
            .iter()
            .zip(lebedev_data::COUNTS.iter())
        {
            let rule = LebedevRule::<f64>::new(
                degree,
                GeometricOptions::default().with_max_nodes(budget(count)),
            )
            .unwrap();
            assert_eq!(rule.node_count(), count);
            let area = rule.weights().iter().sum::<f64>();
            let x2 = rule
                .weights()
                .iter()
                .zip(rule.nodes().chunks_exact(3))
                .map(|(&w, p)| w * p[0] * p[0])
                .sum::<f64>();
            assert!(
                (area - 4.0 * core::f64::consts::PI).abs() < 2e-11,
                "degree={degree} area={area}"
            );
            assert!(
                (x2 - 4.0 * core::f64::consts::PI / 3.0).abs() < 2e-11,
                "degree={degree} x2={x2}"
            );
            let mut zonal = vec![0.0; degree / 2 + 1];
            for (&weight, node) in rule.weights().iter().zip(rule.nodes().chunks_exact(3)) {
                let z = node[2];
                let mut p0 = 1.0;
                let mut p1 = z;
                for l in 2..=degree {
                    let p = ((2 * l - 1) as f64 * z * p1 - (l - 1) as f64 * p0) / l as f64;
                    if l % 2 == 0 {
                        zonal[l / 2] += weight * p;
                    }
                    p0 = p1;
                    p1 = p;
                }
            }
            for (l, value) in zonal.iter().enumerate().skip(1) {
                assert!(
                    value.abs() < 3e-10,
                    "degree={degree} harmonic={} residual={value}",
                    2 * l
                );
            }
        }
    }

    fn associated_legendre(l: usize, m: usize, z: f64) -> f64 {
        let mut diagonal = 1.0;
        let root = (1.0 - z * z).max(0.0).sqrt();
        for order in 1..=m {
            diagonal *= -((2 * order - 1) as f64) * root;
        }
        if l == m {
            return diagonal;
        }
        let mut previous = diagonal;
        let mut current = (2 * m + 1) as f64 * z * diagonal;
        for degree in m + 2..=l {
            let next = ((2 * degree - 1) as f64 * z * current - (degree + m - 1) as f64 * previous)
                / (degree - m) as f64;
            previous = current;
            current = next;
        }
        current
    }

    #[test]
    fn nonzonal_spherical_harmonics_are_orthogonal() {
        for degree in [7, 31, 131] {
            let count = lebedev_data::COUNTS[lebedev_data::ORDERS
                .iter()
                .position(|&d| d == degree)
                .unwrap()];
            let rule = LebedevRule::<f64>::new(
                degree,
                GeometricOptions::default().with_max_nodes(budget(count)),
            )
            .unwrap();
            for m in 1..=3 {
                let harmonic = rule
                    .weights()
                    .iter()
                    .zip(rule.nodes().chunks_exact(3))
                    .map(|(&weight, point)| {
                        let phi = point[1].atan2(point[0]);
                        weight * associated_legendre(degree, m, point[2]) * (m as f64 * phi).cos()
                    })
                    .sum::<f64>();
                assert!(
                    harmonic.abs() < 2e-7,
                    "degree={degree} m={m} residual={harmonic}"
                );
            }
        }
    }

    #[test]
    fn borrowed_value_batch_and_transactional_failure() {
        let rule =
            SimplexRule::<f32>::new(3, 1, GeometricOptions::default().with_max_nodes(budget(12)))
                .unwrap();
        let vertices = [
            [0.0f32; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = vertices.iter().map(|p| p.as_slice()).collect();
        let mut ws = rule.workspace(2).unwrap();
        let mut points = vec![0.0; rule.node_count() * 3];
        let mut weights = vec![0.0; rule.node_count()];
        let _ = rule
            .map_nodes_into(&refs, &mut points, &mut weights, &mut ws)
            .unwrap();
        let values = vec![1.0f32; rule.node_count() * 2];
        let mut output = [0.0f32; 2];
        let _ = context(SolveBudgets::unbounded(), |ctx| {
            rule.integrate_values_into(&refs, &values, &mut output, &mut ws, ctx)
                .unwrap()
        });
        assert!((output[0] - 1.0 / 6.0).abs() < 1e-6);
        assert_eq!(output[0], output[1]);
        output = [7.0, 8.0];
        let sample = [0.0; 2];
        let error = context(SolveBudgets::unbounded(), |ctx| {
            rule.integrate_into(
                |_point: &[_]| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as RealScalar>::NAN
                    });
                    let result: Result<(), SciOxideError> = {
                        Err(crate::kernel::invalid(
                            scioxide_core::argument!("function"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "sample failure",
                        ))
                    };
                    result?;
                    Ok(lane)
                },
                &refs,
                &mut output,
                &mut ws,
                ctx,
            )
        });
        assert!(error.is_err());
        assert_eq!(output, [7.0, 8.0]);
    }

    #[test]
    fn f32_complex_sphere_and_invalid_geometry() {
        let rule =
            LebedevRule::<f32>::new(3, GeometricOptions::default().with_max_nodes(budget(6)))
                .unwrap();
        let mut ws = rule.workspace(2).unwrap();
        let sample = Complex::new(0.0f32, 0.0);
        let mut output = Complex::new(7.0f32, 8.0);
        let _ = context(SolveBudgets::unbounded(), |ctx| {
            rule.integrate_into(
                |_point: &[_]| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as RealScalar>::NAN
                    });
                    let result: Result<(), SciOxideError> = {
                        let value = &mut lane;

                        *value = Complex::new(1.0, 1.0);
                        Ok(())
                    };
                    result?;
                    Ok(lane)
                },
                &[0.0; 3],
                1.0,
                &mut output,
                &mut ws,
                ctx,
            )
            .unwrap()
        });
        assert!((output.re - 4.0 * core::f32::consts::PI).abs() < 2e-6);
        assert_eq!(output.re, output.im);
        output = Complex::new(7.0, 8.0);
        let invalid = context(SolveBudgets::unbounded(), |ctx| {
            rule.integrate_into(
                |_point: &[_]| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as RealScalar>::NAN
                    });
                    let result: Result<(), SciOxideError> = {
                        Err(crate::kernel::invalid(
                            scioxide_core::argument!("function"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "callback should not run",
                        ))
                    };
                    result?;
                    Ok(lane)
                },
                &[0.0; 3],
                0.0,
                &mut output,
                &mut ws,
                ctx,
            )
        });
        assert!(invalid.is_err());
        assert_eq!(output, Complex::new(7.0, 8.0));
    }

    #[test]
    fn lebedev_complex_values_and_executor_agree() {
        let rule =
            LebedevRule::<f64>::new(7, GeometricOptions::default().with_max_nodes(budget(26)))
                .unwrap();
        let center = [0.0, 0.0, 0.0];
        let mut ws = rule.workspace(2).unwrap();
        let sample = Complex::new(0.0, 0.0);
        let mut sequential = Complex::new(0.0, 0.0);
        let _ = context(SolveBudgets::unbounded(), |ctx| {
            rule.integrate_into(
                |point: &[_]| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as RealScalar>::NAN
                    });
                    let result: Result<(), SciOxideError> = {
                        let p = point;
                        let out = &mut lane;

                        *out = Complex::new(p[0] * p[0], p[1] * p[1]);
                        Ok(())
                    };
                    result?;
                    Ok(lane)
                },
                &center,
                1.0,
                &mut sequential,
                &mut ws,
                ctx,
            )
            .unwrap()
        });
        let mut parallel = [0.0; 2];
        let _ = context(SolveBudgets::unbounded(), |ctx| {
            rule.integrate_into(
                |point: &[_]| {
                    let mut lane = [<_ as RealScalar>::NAN; 2];
                    let result: Result<(), SciOxideError> = {
                        let p = point;
                        let out = &mut lane;

                        out[0] = p[0] * p[0];
                        out[1] = p[1] * p[1];
                        Ok(())
                    };
                    result?;
                    Ok(lane)
                },
                &center,
                1.0,
                &mut parallel,
                &mut ws,
                ctx,
            )
            .unwrap()
        });
        assert_eq!([sequential.re, sequential.im], parallel);
        assert!((parallel[0] - 4.0 * core::f64::consts::PI / 3.0).abs() < 2e-14);
        #[cfg(feature = "std")]
        {
            let workers = core::num::NonZeroUsize::new(3).unwrap();
            let pool = scioxide_core::ThreadExecutor::new(workers);
            let mut threaded = [0.0; 2];
            let _ = context_with_execution(
                ExecutionContext::new(&pool),
                SolveBudgets::unbounded(),
                |ctx| {
                    rule.integrate_into(
                        |point: &[_]| {
                            let mut lane = [<_ as RealScalar>::NAN; 2];
                            let result: Result<(), SciOxideError> = {
                                let p = point;
                                let out = &mut lane;

                                out[0] = p[0] * p[0];
                                out[1] = p[1] * p[1];
                                Ok(())
                            };
                            result?;
                            Ok(lane)
                        },
                        &center,
                        1.0,
                        &mut threaded,
                        &mut ws,
                        ctx,
                    )
                    .unwrap()
                },
            );
            assert_eq!(threaded, parallel);
        }
    }
}
