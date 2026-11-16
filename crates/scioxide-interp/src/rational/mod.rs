//! Rational approximation: barycentric rational functions, AAA and AAA–Lawson
//! fits, Floater–Hormann interpolation and Padé approximation.
//!
//! [`BarycentricRational`], [`aaa`] and [`PadeApproximant`] are generic over
//! a real or complex scalar `C`: nodes, weights, coefficients and values are all
//! of type `C`, and complex arguments in a rectangle are allowed.
//! [`FloaterHormannInterpolator`] interpolates on increasing real nodes, like
//! the polynomial interpolants.
//!
//! [`BarycentricRational`] documents the one- and two-coordinate evaluation
//! forms, and [`RationalTermination`] the iteration limits of AAA and Padé.

use alloc::vec::Vec;
use scioxide_core::backend::try_reserve;
use scioxide_core::{
    AlgebraicScalar, Argument, Capacity, Interval, NumericScalar, RealScalar, SciOxideError,
};

use crate::error::{invalid, shape};
use crate::protocol::{ExtrapolationPolicy, resolve_coordinate};
use crate::validate::validate_finite;

mod aaa;
mod diagnostics;
mod floater_hormann;
mod pade;

pub use aaa::{
    AaaOptions, RationalFit, RationalRequest, RationalRequirements, RationalWorkspace, aaa,
    aaa_into, aaa_with_options,
};
pub use diagnostics::{RationalCleanup, RationalDiagnostics, RationalPole};
pub use floater_hormann::FloaterHormannInterpolator;
pub use pade::PadeApproximant;

/// Checks that finite nodes are distinct, without changing the sample order.
///
/// `indices` must already have capacity for `nodes.len()` entries.
pub(super) fn distinct<C: NumericScalar>(
    nodes: &[C],
    indices: &mut Vec<usize>,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    indices.clear();
    indices.extend(0..nodes.len());
    indices.sort_unstable_by(|&left, &right| {
        nodes[left]
            .real_component()
            .partial_cmp(&nodes[right].real_component())
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| {
                nodes[left]
                    .imaginary_component()
                    .partial_cmp(&nodes[right].imaginary_component())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
    });
    if indices
        .windows(2)
        .any(|pair| nodes[pair[0]] == nodes[pair[1]])
    {
        return Err(invalid(scioxide_core::argument!("nodes"), detail));
    }
    Ok(())
}

/// Real components of one value of field `C`.
pub(super) const fn value_dimension<C: NumericScalar>() -> usize {
    if C::IS_REAL { 1 } else { 2 }
}

/// Writes a finite value to `out`, which has [`value_dimension`] components.
pub(super) fn commit<C, O>(value: C, out: &mut O) -> Result<(), SciOxideError>
where
    C: NumericScalar,
    O: scioxide_core::AffineValueMut<Scale = C::Real> + ?Sized,
{
    let components = [value.real_component(), value.imaginary_component()];
    scioxide_core::write_value(&components[..value_dimension::<C>()], out)
}

/// One-coordinate queries need a real-interval domain.
pub(super) fn rectangle_error() -> SciOxideError {
    invalid(
        scioxide_core::argument!("point"),
        "a rectangle domain requires the two-coordinate form",
    )
}

/// Specifies the closed domain of a rational approximant: a real interval or a complex rectangle.
///
/// Support points, sample nodes and expansion centers must lie in the domain,
/// and queries are checked against it under the [`ExtrapolationPolicy`].
/// The domain does not imply an accuracy guarantee: a fit is only measured at
/// the samples you supply.
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_interp::RationalDomain;
/// let real = RationalDomain::Real(Interval::new(-1.0_f64, 1.0)?);
/// let rectangle = RationalDomain::Complex {
///     real: Interval::new(-1.0_f64, 1.0)?,
///     imaginary: Interval::new(-0.5, 0.5)?,
/// };
/// assert!(matches!(real, RationalDomain::Real(_)));
/// assert!(matches!(rectangle, RationalDomain::Complex { .. }));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RationalDomain<R: RealScalar> {
    /// Real coordinates in a closed interval.
    Real(Interval<R>),
    /// Complex coordinates in a rectangle, with independent real and imaginary intervals.
    Complex {
        /// Real component interval.
        real: Interval<R>,
        /// Imaginary component interval.
        imaginary: Interval<R>,
    },
}

impl<R: RealScalar> RationalDomain<R> {
    /// Resolves a coordinate under `policy`; a real domain rejects imaginary parts.
    pub(super) fn resolve<C: NumericScalar<Real = R>>(
        self,
        z: C,
        policy: ExtrapolationPolicy<R>,
        argument: Argument,
    ) -> Result<C, SciOxideError> {
        match self {
            Self::Real(interval) => {
                if z.imaginary_component() != R::ZERO {
                    return Err(invalid(argument, "real domain requires real coordinates"));
                }
                Ok(C::from_real(resolve_coordinate(
                    &interval,
                    z.real_component(),
                    policy,
                )?))
            }
            Self::Complex { real, imaginary } => {
                let effective = policy
                    .resolve(
                        [real, imaginary],
                        [z.real_component(), z.imaginary_component()],
                    )?
                    .effective();
                Ok(C::from_components(effective[0], effective[1]))
            }
        }
    }

    /// Resolves a one-coordinate query on a real-interval domain.
    pub(super) fn resolve_real<C: NumericScalar<Real = R>>(
        self,
        x: R,
        policy: ExtrapolationPolicy<R>,
    ) -> Result<C, SciOxideError> {
        match self {
            Self::Real(interval) => Ok(C::from_real(resolve_coordinate(&interval, x, policy)?)),
            Self::Complex { .. } => Err(rectangle_error()),
        }
    }

    /// Returns the coordinate rectangle of the two-coordinate form.
    pub(super) fn plane(self) -> Result<[Interval<R>; 2], SciOxideError> {
        match self {
            Self::Real(interval) => Ok([interval, Interval::new(R::ZERO, R::ZERO)?]),
            Self::Complex { real, imaginary } => Ok([real, imaginary]),
        }
    }
}

/// Reports why a rational fit stopped.
///
/// Only [`Self::Converged`] counts as success. The other statuses come with a
/// usable approximant whose residual is above the requested tolerance.
///
/// # Iteration limits
///
/// AAA and Padé count iterations in the solve scope they run in (see
/// [solve accounting](scioxide_core#solve-accounting)). An AAA solve counts one
/// iteration per support step and per Lawson step, under the limit from
/// [`AaaOptions::with_budgets`]. Its first approximant is the constant sample
/// mean, so reaching the limit returns the best approximant so far with
/// [`Self::BudgetExhausted`]. A Padé construction counts one iteration before
/// any work and has no approximant before it, so reaching the limit there is a
/// `ConvergenceFailure` (`IterationLimit`) error. Cancellation or a deadline
/// is an `ExecutionStopped` error in both. A parent scope counts the
/// iterations of a nested solve, but its own iteration limit does not apply to
/// it.
///
/// ```
/// use scioxide_interp::RationalTermination;
/// assert!(RationalTermination::Converged.is_success());
/// assert!(!RationalTermination::BudgetExhausted.is_success());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RationalTermination {
    /// The maximum residual at the samples is within the requested tolerance.
    Converged,
    /// The maximum number of support points was reached.
    SupportLimit,
    /// The maximum number of Lawson steps was reached.
    IterationLimit,
    /// Lawson refinement stopped reducing the maximum residual at the samples.
    Stagnation,
    /// The iteration limit of the solve scope was reached.
    BudgetExhausted,
}

impl RationalTermination {
    /// Returns whether the fit met its tolerance, that is, the status is [`Self::Converged`].
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Converged)
    }
}

/// Describes the outcome of a rational fit: status, iterations, residual and conditioning.
///
/// For AAA, the residual is the maximum of `|r(zᵢ) − fᵢ|` over the samples.
/// For Padé, it is the largest residual of the Padé conditions on the Taylor
/// coefficients. The singular-value ratio `σ_min / σ_max` belongs to the
/// least-squares or linear system that produced the returned approximant. A
/// small ratio, or zero for a rank-deficient system, signals ill-conditioning.
/// It is not an error bound.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Interval, Tolerance};
/// use scioxide_interp::{AaaOptions, RationalDomain, RationalTermination, aaa_with_options};
/// let x: Vec<f64> = (0..11).map(|i| -1.0 + f64::from(i) / 5.0).collect();
/// let y: Vec<f64> = x.iter().map(|x| 1.0 / (2.0 + x)).collect();
/// let domain = RationalDomain::Real(Interval::new(-1.0, 1.0)?);
/// let tolerance = Tolerance::new(1e-12, 1e-12)?;
/// let options = AaaOptions::new()
///     .with_max_terms(4)
///     .with_tolerance(tolerance);
/// let report = aaa_with_options(&x, &y, domain, options)?.report().unwrap();
/// assert_eq!(report.termination(), RationalTermination::Converged);
/// assert!(report.max_residual() <= 1e-12 && report.iterations() >= 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RationalReport<S: RealScalar> {
    termination: RationalTermination,
    iterations: usize,
    max_residual: S,
    singular_ratio: S,
}

impl<S: RealScalar> RationalReport<S> {
    pub(super) const fn new(
        termination: RationalTermination,
        iterations: usize,
        max_residual: S,
        singular_ratio: S,
    ) -> Self {
        Self {
            termination,
            iterations,
            max_residual,
            singular_ratio,
        }
    }

    /// Returns the status of the fit.
    #[must_use]
    pub const fn termination(&self) -> RationalTermination {
        self.termination
    }

    /// Returns the number of iterations: AAA support and Lawson steps, or one for Padé.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the maximum residual of the returned approximant.
    #[must_use]
    pub const fn max_residual(&self) -> S {
        self.max_residual
    }

    /// Returns `σ_min / σ_max` of the system that produced the returned approximant.
    #[must_use]
    pub const fn singular_ratio(&self) -> S {
        self.singular_ratio
    }
}

/// Support points with numerator and denominator barycentric coefficients.
#[derive(Debug, Clone)]
pub(super) struct Support<C> {
    pub(super) nodes: Vec<C>,
    pub(super) numerator: Vec<C>,
    pub(super) denominator: Vec<C>,
}

impl<C> Default for Support<C> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            numerator: Vec::new(),
            denominator: Vec::new(),
        }
    }
}

impl<C: NumericScalar> Support<C> {
    pub(super) fn clear(&mut self) {
        self.nodes.clear();
        self.numerator.clear();
        self.denominator.clear();
    }

    /// Returns the number of terms the buffers hold without growing.
    pub(super) fn capacity(&self) -> usize {
        self.nodes
            .capacity()
            .min(self.numerator.capacity())
            .min(self.denominator.capacity())
    }

    /// Grows every buffer to hold `terms`; existing coefficients are unchanged.
    pub(super) fn try_reserve(&mut self, terms: usize) -> Result<(), SciOxideError> {
        for buffer in [&mut self.nodes, &mut self.numerator, &mut self.denominator] {
            try_reserve(
                buffer,
                terms.saturating_sub(buffer.len()),
                scioxide_core::quantity!("rational_support"),
            )?;
        }
        Ok(())
    }

    /// Returns the barycentric quotient and denominator condition at a resolved `z`.
    pub(super) fn evaluate(&self, z: C) -> Result<(C, C::Real), SciOxideError> {
        let reference = self
            .nodes
            .iter()
            .zip(&self.denominator)
            .enumerate()
            .filter(|(_, (_, weight))| **weight != C::ZERO)
            .min_by(|(_, (left, _)), (_, (right, _))| {
                (z - **left)
                    .abs()
                    .partial_cmp(&(z - **right).abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .map_or(0, |(index, _)| index);
        let result = scioxide_poly::backend::barycentric::evaluate::<C, _, false, false, false>(
            &self.nodes,
            &self.denominator,
            |index| self.numerator[index],
            z,
            reference,
            C::Real::ZERO,
        )?;
        Ok((result.value, result.denominator_condition))
    }
}

/// Evaluates a rational function in barycentric form over a real or complex scalar `C`.
///
/// With support points `zⱼ`, numerator coefficients `aⱼ` and denominator
/// weights `bⱼ`,
///
/// ```text
/// r(z) = Σⱼ aⱼ / (z − zⱼ)  /  Σⱼ bⱼ / (z − zⱼ).
/// ```
///
/// If `aⱼ = bⱼ fⱼ` with `bⱼ ≠ 0`, then `r(zⱼ) = fⱼ`. AAA fits produce
/// this form, and so does [`Self::cleanup_doublets`]. AAA–Lawson fits have
/// independent `aⱼ` and need not interpolate. A query costs O(k) for `k`
/// support points and needs no workspace (the workspace type is `()`). Terms
/// are normalized by the term nearest the query and rescaled when they would
/// overflow or underflow. [`Self::denominator_condition`] estimates the
/// cancellation in the denominator at a point, and [`Self::diagnostics`]
/// computes zeros, poles and residues.
///
/// # Evaluation forms
///
/// Evaluation goes through [`crate::Interpolant`], as for
/// [`crate::PadeApproximant`]. Every approximant implements
/// `Interpolant<C::Real, 1>` for real arguments on a real-interval domain. On a
/// rectangle domain, one-coordinate queries fail with `InvalidInput`.
/// Approximants over [`scioxide_core::Complex`]`<S>` also implement
/// `Interpolant<S, 2>` for arguments `[Re z, Im z]`; a real domain is then the
/// rectangle `[a, b] × [0, 0]`. Because both forms apply to a complex
/// approximant, calls without a point argument must name the form, for example
/// `Interpolant::<f64, 2>::domains(&plan)` or
/// `Interpolant::<f64, 1>::workspace(&plan)`, and a [`crate::BatchEvaluator`] over
/// a complex approximant names its dimension (`BatchEvaluator::<f64, _, 1>::new`).
///
/// # Errors
///
/// - The query errors described in the [crate docs](crate#errors).
/// - `InvalidInput` for a one-coordinate query on a rectangle domain.
/// - `NumericalInstability` (`Breakdown`) at a pole, where the denominator
///   vanishes.
/// - `NonFinite` if the result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, Interval};
/// use scioxide_interp::{BarycentricRational, ExtrapolationPolicy, Interpolant, RationalDomain};
///
/// let domain = RationalDomain::Real(Interval::new(-1.0_f64, 1.0)?);
/// // Weights b = (1, −1) and aⱼ = bⱼ r(zⱼ) interpolate r(z) = 2 + z at −1 and 1.
/// let real = BarycentricRational::new(
///     vec![-1.0_f64, 1.0],
///     vec![1.0, -3.0],
///     vec![1.0, -1.0],
///     domain,
/// )?;
/// assert!((real.evaluate([0.5], ExtrapolationPolicy::Reject)? - 2.5).abs() < 1e-15);
/// let condition = real.denominator_condition(0.5, ExtrapolationPolicy::Reject)?;
/// assert!((condition - 1.0).abs() < 1e-14); // both terms have the same sign: no cancellation
///
/// let one = Complex::new(1.0, 0.0);
/// let complex = BarycentricRational::new(
///     vec![-one, one],
///     vec![one, Complex::new(-3.0, 0.0)],
///     vec![one, -one],
///     domain,
/// )?;
/// // A complex approximant has both forms; calls without a point name the form.
/// assert_eq!(Interpolant::<f64, 2>::value_dimension(&complex), 2);
/// let value = complex.evaluate([0.5], ExtrapolationPolicy::Reject)?;
/// assert!((value - Complex::new(2.5, 0.0)).norm() < 1e-15);
/// let off_axis = complex.evaluate([0.5, 0.25], ExtrapolationPolicy::MethodDefined)?;
/// assert!((off_axis - Complex::new(2.5, 0.25)).norm() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct BarycentricRational<C: NumericScalar> {
    support: Support<C>,
    domain: RationalDomain<C::Real>,
    plane: [Interval<C::Real>; 2],
}

impl<C: NumericScalar> BarycentricRational<C> {
    /// Builds the rational function from support points, numerator coefficients and denominator weights.
    ///
    /// `nodes` holds the support points `zⱼ`, `numerator` the coefficients
    /// `aⱼ` and `denominator` the weights `bⱼ`. The support points must be finite, distinct and inside `domain`. The
    /// coefficients must be finite, and at least one denominator weight must
    /// be nonzero. The vectors are stored without copying.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `nodes` is empty, contains duplicates, or every
    ///   denominator weight is zero, or if a real domain gets a complex node.
    /// - `DimensionMismatch` if the three vectors differ in length.
    /// - `NonFinite` if a node or coefficient is NaN or infinite.
    /// - `DomainError` if a node lies outside `domain`.
    pub fn new(
        nodes: Vec<C>,
        numerator: Vec<C>,
        denominator: Vec<C>,
        domain: RationalDomain<C::Real>,
    ) -> Result<Self, SciOxideError> {
        if nodes.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("nodes"),
                "nonempty support required",
            ));
        }
        if nodes.len() != numerator.len() {
            return Err(shape(nodes.len(), numerator.len()));
        }
        if nodes.len() != denominator.len() {
            return Err(shape(nodes.len(), denominator.len()));
        }
        for &node in &nodes {
            domain.resolve(
                node,
                ExtrapolationPolicy::Reject,
                scioxide_core::argument!("nodes"),
            )?;
        }
        let mut indices = scioxide_core::backend::try_with_capacity(
            nodes.len(),
            scioxide_core::quantity!("rational_node_indices"),
        )?;
        distinct(&nodes, &mut indices, "duplicate support points")?;
        validate_finite(&numerator)?;
        validate_finite(&denominator)?;
        if denominator.iter().all(|v| *v == C::ZERO) {
            return Err(invalid(
                scioxide_core::argument!("denominator"),
                "finite coefficients and nonzero denominator required",
            ));
        }
        Self::from_parts(
            Support {
                nodes,
                numerator,
                denominator,
            },
            domain,
        )
    }

    /// Wraps support data that is already validated.
    pub(super) fn from_parts(
        support: Support<C>,
        domain: RationalDomain<C::Real>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self::from_resolved(support, domain, domain.plane()?))
    }

    /// Wraps validated support data with its precomputed query rectangle.
    pub(super) const fn from_resolved(
        support: Support<C>,
        domain: RationalDomain<C::Real>,
        plane: [Interval<C::Real>; 2],
    ) -> Self {
        Self {
            support,
            domain,
            plane,
        }
    }

    /// Returns the support buffers for reuse.
    pub(super) fn into_support(self) -> Support<C> {
        self.support
    }

    pub(super) const fn support(&self) -> &Support<C> {
        &self.support
    }

    /// Returns the support points `zⱼ` in construction order.
    pub fn support_points(&self) -> &[C] {
        &self.support.nodes
    }

    /// Returns the numerator coefficients `aⱼ`, which equal `bⱼ fⱼ` only for an interpolating fit.
    pub fn numerator_weights(&self) -> &[C] {
        &self.support.numerator
    }

    /// Returns the denominator weights `bⱼ`.
    pub fn weights(&self) -> &[C] {
        &self.support.denominator
    }

    /// Returns the approximation domain.
    pub const fn domain(&self) -> RationalDomain<C::Real> {
        self.domain
    }

    /// Estimates the cancellation in the denominator at `z`.
    ///
    /// ```text
    /// κ(z) = Σⱼ |bⱼ / (z − zⱼ)|  /  |Σⱼ bⱼ / (z − zⱼ)|
    /// ```
    ///
    /// `κ ≥ 1`, with `κ = 1` at a support point or when the terms do not cancel.
    /// A large `κ` signals cancellation and a less accurate value. `z` is
    /// checked against the domain under `policy`, as for a query. The estimate
    /// indicates conditioning; it is not an error bound.
    ///
    /// # Errors
    ///
    /// - The query errors described in the [crate docs](crate#errors).
    /// - `NumericalInstability` (`Breakdown`) at a pole.
    /// - `NonFinite` if the arithmetic overflows.
    pub fn denominator_condition(
        &self,
        z: C,
        policy: ExtrapolationPolicy<C::Real>,
    ) -> Result<C::Real, SciOxideError> {
        Ok(self.support.evaluate(self.resolve_value(z, policy)?)?.1)
    }

    /// Resolves a coordinate on the query rectangle.
    fn resolve_value(
        &self,
        z: C,
        policy: ExtrapolationPolicy<C::Real>,
    ) -> Result<C, SciOxideError> {
        let effective = policy
            .resolve(self.plane, [z.real_component(), z.imaginary_component()])?
            .effective();
        Ok(C::from_components(effective[0], effective[1]))
    }

    /// Returns the value at a resolved coordinate.
    fn field_value(&self, z: C) -> Result<C, SciOxideError> {
        Ok(self.support.evaluate(z)?.0)
    }
}

/// `Interpolant` forms shared by the rational plans: `D = 1` for every field on a
/// real-interval domain and `D = 2` for complex fields on the query rectangle.
/// The type provides `domain`, `plane` and `field_value(&self, z: C)`.
macro_rules! rational_interpolant {
    ($family:ident) => {
        impl<C> $crate::protocol::Interpolant<C::Real, 1> for $family<C>
        where
            C: scioxide_core::NumericScalar + scioxide_core::AffineValueMut<Scale = C::Real>,
        {
            type Value = C;
            type Workspace = ();

            fn domains(&self) -> [scioxide_core::Interval<C::Real>; 1] {
                [self.plane[0]]
            }
            fn value_dimension(&self) -> usize {
                $crate::rational::value_dimension::<C>()
            }
            fn zero_value(&self) -> Result<C, scioxide_core::SciOxideError> {
                Ok(<C as scioxide_core::AlgebraicScalar>::ZERO)
            }
            fn workspace(&self) -> Result<(), scioxide_core::SciOxideError> {
                Ok(())
            }
            fn workspace_capacity(&self, _workspace: &()) -> scioxide_core::Capacity {
                scioxide_core::Capacity::Sufficient
            }
            fn evaluate_into<O: scioxide_core::AffineValueMut<Scale = C::Real> + ?Sized>(
                &self,
                point: [C::Real; 1],
                policy: $crate::protocol::ExtrapolationPolicy<C::Real>,
                out: &mut O,
                _workspace: &mut (),
            ) -> Result<(), scioxide_core::SciOxideError> {
                scioxide_core::check_value_shape(out, $crate::rational::value_dimension::<C>())?;
                let z = self.domain.resolve_real::<C>(point[0], policy)?;
                $crate::rational::commit(self.field_value(z)?, out)
            }
            fn report(
                &self,
                point: [C::Real; 1],
                policy: $crate::protocol::ExtrapolationPolicy<C::Real>,
                _workspace: &mut (),
            ) -> Result<
                $crate::protocol::ExtrapolationReport<C::Real, 1>,
                scioxide_core::SciOxideError,
            > {
                match self.domain {
                    $crate::rational::RationalDomain::Real(interval) => {
                        policy.resolve([interval], point)
                    }
                    $crate::rational::RationalDomain::Complex { .. } => {
                        Err($crate::rational::rectangle_error())
                    }
                }
            }
        }

        impl<S: scioxide_core::RealScalar> $crate::protocol::Interpolant<S, 2>
            for $family<scioxide_core::Complex<S>>
        where
            scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
        {
            type Value = scioxide_core::Complex<S>;
            type Workspace = ();

            fn domains(&self) -> [scioxide_core::Interval<S>; 2] {
                self.plane
            }
            fn value_dimension(&self) -> usize {
                2
            }
            fn zero_value(
                &self,
            ) -> Result<scioxide_core::Complex<S>, scioxide_core::SciOxideError> {
                Ok(scioxide_core::Complex::new(S::ZERO, S::ZERO))
            }
            fn workspace(&self) -> Result<(), scioxide_core::SciOxideError> {
                Ok(())
            }
            fn workspace_capacity(&self, _workspace: &()) -> scioxide_core::Capacity {
                scioxide_core::Capacity::Sufficient
            }
            fn evaluate_into<O: scioxide_core::AffineValueMut<Scale = S> + ?Sized>(
                &self,
                point: [S; 2],
                policy: $crate::protocol::ExtrapolationPolicy<S>,
                out: &mut O,
                _workspace: &mut (),
            ) -> Result<(), scioxide_core::SciOxideError> {
                scioxide_core::check_value_shape(out, 2)?;
                let [re, im] = policy.resolve(self.plane, point)?.effective();
                $crate::rational::commit(
                    self.field_value(scioxide_core::Complex::new(re, im))?,
                    out,
                )
            }
        }
    };
}
pub(super) use rational_interpolant;

rational_interpolant!(BarycentricRational);

/// Capacity of `support` against `terms`.
pub(super) fn support_capacity<C: NumericScalar>(support: &Support<C>, terms: usize) -> Capacity {
    if support.capacity() >= terms {
        Capacity::Sufficient
    } else {
        Capacity::NeedsGrowth
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Interpolant;
    use scioxide_core::Complex;

    fn domain() -> RationalDomain<f64> {
        RationalDomain::Real(Interval::new(-1.0, 1.0).unwrap())
    }

    #[test]
    fn unordered_complex_duplicates_and_signed_zero_share_exact_equality() {
        let mut indices = Vec::with_capacity(4);
        let nodes = [
            Complex::new(1.0_f64, 1.0),
            Complex::new(0.0, -0.0),
            Complex::new(-1.0, 1.0),
            Complex::new(-0.0, 0.0),
        ];
        assert!(distinct(&nodes, &mut indices, "duplicate nodes").is_err());
        assert!(
            BarycentricRational::new(
                nodes.to_vec(),
                vec![Complex::new(1.0, 0.0); 4],
                vec![Complex::new(1.0, 0.0); 4],
                RationalDomain::Complex {
                    real: Interval::new(-1.0, 1.0).unwrap(),
                    imaginary: Interval::new(-1.0, 1.0).unwrap(),
                }
            )
            .is_err()
        );
        assert!(distinct(&nodes[..3], &mut indices, "duplicate nodes").is_ok());
    }

    #[test]
    fn real_plans_reject_rectangles_and_complex_plans_use_the_plane() {
        let rectangle = RationalDomain::Complex {
            real: Interval::new(-1.0, 1.0).unwrap(),
            imaginary: Interval::new(-1.0, 1.0).unwrap(),
        };
        let one = Complex::new(1.0_f64, 0.0);
        let plan =
            BarycentricRational::new(vec![-one, one], vec![one, -one], vec![one, -one], rectangle)
                .unwrap();
        let error =
            Interpolant::<f64, 1>::evaluate(&plan, [0.0], ExtrapolationPolicy::Reject).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        assert!(
            Interpolant::<f64, 1>::report(&plan, [0.0], ExtrapolationPolicy::Reject, &mut ())
                .is_err()
        );
        let value = plan
            .evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_eq!(value, one);
        assert_eq!(
            Interpolant::<f64, 2>::domains(&plan),
            [Interval::new(-1.0, 1.0).unwrap(); 2]
        );

        let real =
            BarycentricRational::new(vec![-one, one], vec![one, -one], vec![one, -one], domain())
                .unwrap();
        assert_eq!(
            Interpolant::<f64, 2>::domains(&real)[1],
            Interval::new(0.0, 0.0).unwrap()
        );
        let error = real
            .evaluate([0.0, 0.5], ExtrapolationPolicy::Reject)
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        assert_eq!(
            real.evaluate([0.0, 0.5], ExtrapolationPolicy::Clamp)
                .unwrap(),
            one
        );
    }

    #[test]
    fn output_shapes_are_checked_and_unchanged_on_error() {
        let plan =
            BarycentricRational::<f64>::new(vec![-1., 1.], vec![1., -1.], vec![1., 1.], domain())
                .unwrap();
        let mut out = [7.0; 2];
        let error = plan
            .evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out[..], &mut ())
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        let mut scalar = 7.0;
        assert!(
            plan.evaluate_into([0.0], ExtrapolationPolicy::Reject, &mut scalar, &mut ())
                .is_err()
        );
        assert_eq!(scalar, 7.0);
        assert!(
            plan.denominator_condition(0.0, ExtrapolationPolicy::Reject)
                .is_err()
        );
        assert_eq!(
            plan.denominator_condition(1.0, ExtrapolationPolicy::Reject)
                .unwrap(),
            1.0
        );
    }
}
