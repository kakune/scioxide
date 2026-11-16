//! Prepared classical orthogonal bases on a physical interval.
//!
//! Recurrences and norms follow DLMF 18.3 and 18.9. Connection coefficients
//! are generated in the destination basis using its tridiagonal multiplication
//! operator; evaluation and calculus never expand into powers of `x`.
use crate::util::{finite_input, minimum_len, validate_slice};

use crate::basis::condition_report;
use crate::{
    ApproxPolynomial, OrthogonalFamily, PolynomialApproximationPolicy, PolynomialBasis,
    PolynomialEvaluation,
};
use crate::{PolynomialExecutionOptions, PolynomialScratchRequirements};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::ScaledValue;
use scioxide_core::backend::{checked_add, try_with_capacity};
use scioxide_core::{Argument, ExecutionContext, NumericScalar, RealScalar, SciOxideError};

use super::admission::RecurrenceData;
use super::conversion;
use crate::admission::{CoefficientView, Coefficients};

/// Selects how the members of a prepared orthogonal basis are scaled.
///
/// With the affine map `t = (x − shift)/scale` of [`OrthogonalBasisSpec`], the
/// basis member of degree `k` is `Qₖ(x) = σₖ pₖ(t)`, where `pₖ` is the classical
/// polynomial of the [`OrthogonalFamily`] and `σₖ` is a constant fixed by the
/// normalization.
///
/// ```
/// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
///
/// let (family, normalization) = (OrthogonalFamily::ChebyshevT, BasisNormalization::Classical);
/// let spec = OrthogonalBasisSpec::new(family, normalization, 0.0_f64, 1.0)?;
/// assert_eq!(spec.normalization(), BasisNormalization::Classical);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisNormalization {
    /// `σₖ = 1`: the standard normalization of the family (see
    /// [`OrthogonalFamily`]).
    Classical,
    /// Leading coefficient one in the physical variable `x`.
    Monic,
    /// `∫ Qₖ(x)² w((x − shift)/scale) dx = 1`, with each member keeping the
    /// sign of its classical counterpart.
    Orthonormal,
}

/// Describes an orthogonal basis: a family, a normalization and an affine map
/// from the physical variable.
///
/// The basis polynomials are functions of the physical variable `x` through
/// `t = (x − shift)/scale`, with `scale > 0`. For example, Chebyshev or
/// Legendre polynomials on `[a, b]` use `shift = (a + b)/2` and
/// `scale = (b − a)/2`. Derivatives and antiderivatives of expansions are taken
/// with respect to `x`, and integrals are unweighted unless you supply a
/// weight. Call [`prepare`](Self::prepare) to get an [`OrthogonalBasisPlan`].
///
/// ```
/// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
///
/// let (family, normalization) = (OrthogonalFamily::ChebyshevT, BasisNormalization::Classical);
/// let spec = OrthogonalBasisSpec::new(family, normalization, 0.0_f64, 1.0)?;
/// assert_eq!(spec.family(), OrthogonalFamily::ChebyshevT);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrthogonalBasisSpec<R: RealScalar> {
    pub(super) family: OrthogonalFamily<R>,
    pub(super) normalization: BasisNormalization,
    pub(super) shift: R,
    pub(super) scale: R,
}
impl<R: RealScalar> OrthogonalBasisSpec<R> {
    /// Creates a basis description after checking the family parameters and
    /// the affine map.
    ///
    /// # Errors
    ///
    /// - `DomainError` if a family parameter is outside its range or
    ///   `scale ≤ 0`.
    /// - `NonFinite` if a family parameter, `shift` or `scale` is NaN or
    ///   infinite.
    ///
    /// ```
    /// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let (family, normalization) = (OrthogonalFamily::ChebyshevT, BasisNormalization::Classical);
    /// let spec = OrthogonalBasisSpec::new(family, normalization, 0.0_f64, 1.0)?;
    /// assert_eq!(spec.normalization(), BasisNormalization::Classical);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        family: OrthogonalFamily<R>,
        normalization: BasisNormalization,
        shift: R,
        scale: R,
    ) -> Result<Self, SciOxideError> {
        family.validate()?;
        finite_input(shift, "shift must be finite")?;
        finite_input(scale, "scale must be finite")?;
        if scale <= R::ZERO {
            return Err(domain(
                scioxide_core::argument!("scale"),
                "scale must be positive",
            ));
        }
        Ok(Self {
            family,
            normalization,
            shift,
            scale,
        })
    }
    /// Creates the classical basis of `family` in its own variable: `shift = 0`
    /// and `scale = 1`.
    ///
    /// The members are the polynomials of [`OrthogonalFamily`] with their
    /// standard normalization, on the support listed there.
    ///
    /// # Errors
    ///
    /// - `DomainError` if a family parameter is outside its range.
    /// - `NonFinite` if a family parameter is NaN or infinite.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::HermiteH)?;
    /// assert_eq!((spec.shift(), spec.scale()), (0.0, 1.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn classical(family: OrthogonalFamily<R>) -> Result<Self, SciOxideError> {
        Self::new(family, BasisNormalization::Classical, R::ZERO, R::ONE)
    }
    /// Creates the classical basis of a family on `[−1, 1]`, mapped affinely
    /// onto `[lower, upper]`.
    ///
    /// The map `t = (x − shift)/scale` uses `shift = (lower + upper)/2` and
    /// `scale = (upper − lower)/2`, each rounded once and formed without
    /// intermediate overflow. Only the Chebyshev, Legendre, Jacobi and
    /// Gegenbauer families live on `[−1, 1]`; build Laguerre and Hermite bases
    /// with [`classical`](Self::classical) or [`new`](Self::new). Call
    /// [`with_normalization`](Self::with_normalization) for another
    /// normalization.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lower` or `upper` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`, on `upper`) if `lower ≥ upper`, or if
    ///   `(upper − lower)/2` rounds to zero.
    /// - `InvalidInput` (`Inconsistent`, on `family`) for `Laguerre`,
    ///   `GeneralizedLaguerre`, `HermiteH` and `HermiteHe`.
    /// - `DomainError` or `NonFinite` if a family parameter is outside its range
    ///   or not finite.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// // [0, 2] maps onto [−1, 1] through t = x − 1.
    /// let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::ChebyshevT, 0.0_f64, 2.0)?;
    /// assert_eq!((spec.shift(), spec.scale()), (1.0, 1.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn on_interval(
        family: OrthogonalFamily<R>,
        lower: R,
        upper: R,
    ) -> Result<Self, SciOxideError> {
        crate::basis::interval(lower, upper)?;
        if matches!(
            family,
            OrthogonalFamily::Laguerre
                | OrthogonalFamily::GeneralizedLaguerre { .. }
                | OrthogonalFamily::HermiteH
                | OrthogonalFamily::HermiteHe
        ) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("family"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "the support of the family is not [-1, 1]",
            ));
        }
        let shift = scioxide_core::checked_midpoint(lower, upper)?;
        let scale = scioxide_core::checked_midpoint(upper, -lower)?;
        if scale == R::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("upper"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "the half-width of the interval rounds to zero",
            ));
        }
        Self::new(family, BasisNormalization::Classical, shift, scale)
    }
    /// Returns the same basis with another normalization.
    ///
    /// ```
    /// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::Legendre)?
    ///     .with_normalization(BasisNormalization::Orthonormal);
    /// assert_eq!(spec.normalization(), BasisNormalization::Orthonormal);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn with_normalization(self, normalization: BasisNormalization) -> Self {
        Self {
            normalization,
            ..self
        }
    }
    /// Returns the polynomial family.
    ///
    /// ```
    /// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let (family, normalization) = (OrthogonalFamily::ChebyshevT, BasisNormalization::Classical);
    /// let spec = OrthogonalBasisSpec::new(family, normalization, 0.0_f64, 1.0)?;
    /// assert_eq!(spec.family(), OrthogonalFamily::ChebyshevT);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn family(self) -> OrthogonalFamily<R> {
        self.family
    }
    /// Returns the normalization.
    ///
    /// ```
    /// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let (family, normalization) = (OrthogonalFamily::ChebyshevT, BasisNormalization::Classical);
    /// let spec = OrthogonalBasisSpec::new(family, normalization, 0.0_f64, 1.0)?;
    /// assert_eq!(spec.normalization(), BasisNormalization::Classical);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn normalization(self) -> BasisNormalization {
        self.normalization
    }
    /// Returns `shift`, the physical point that maps to `t = 0`.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// assert_eq!(spec.shift(), 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn shift(self) -> R {
        self.shift
    }
    /// Returns `scale > 0`, the physical length that maps to a unit length in `t`.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// assert_eq!(spec.scale(), 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn scale(self) -> R {
        self.scale
    }
    /// Computes the recurrence coefficients of the basis through `degree`.
    ///
    /// The plan stores one normalized coefficient triple per degree, in
    /// O(`degree`) time and storage. For the orthonormal normalization it also
    /// computes `Q₀ = (μ₀ · scale)^(−1/2)`, where `μ₀` is the mass of the weight
    /// ([`OrthogonalFamily::log_weight_mass`]). `options` sets the execution
    /// policy of batch evaluation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `options` allows more than 64 workers.
    /// - `NumericalInstability` if `Q₀` or a recurrence coefficient overflows,
    ///   underflows to zero or breaks down.
    /// - `SizeOverflow` or `AllocationFailure` if the coefficients cannot be
    ///   stored.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// assert_eq!(plan.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn prepare(
        self,
        degree: usize,
        options: PolynomialExecutionOptions,
    ) -> Result<OrthogonalBasisPlan<R>, SciOxideError> {
        options.validate()?;
        let count = checked_add(degree, 1, scioxide_core::quantity!("recurrence_terms"))?;
        let mut recurrence =
            try_with_capacity(count, scioxide_core::quantity!("recurrence_terms"))?;
        // Orthonormal bases need `P_0 = 1 / sqrt(h_0 * scale)`: the affine map
        // `t = (x - shift) / scale` multiplies the weight's mass by `scale`.
        let p0 = match self.normalization {
            BasisNormalization::Orthonormal => crate::util::finite_arithmetic(
                (-(self.family.log_mass()? + self.scale.ln()) / (R::TWO)).exp(),
                "orthonormal normalization is outside the scalar range",
            )?,
            _ => R::ONE,
        };
        if p0 == R::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "normalization constant underflow",
            ));
        }
        for n in 0..count {
            recurrence.push(self.normalized_step(n)?);
        }
        Ok(OrthogonalBasisPlan {
            spec: self,
            options,
            recurrence: RecurrenceData::new(recurrence),
            p0,
        })
    }
}

/// Holds the recurrence coefficients of an orthogonal basis up to a maximum
/// degree.
///
/// Preparation computes one normalized recurrence-coefficient triple per degree,
/// in O(n) time and storage for degree `n`. A plan is immutable and can evaluate
/// any expansion of at most its degree. Pass it to
/// [`ApproxPolynomial::orthogonal_with_plan`] to own coefficients, or call
/// [`evaluate_with_derivative`](Self::evaluate_with_derivative) and
/// [`evaluate_batch_into`](Self::evaluate_batch_into) on borrowed coefficient
/// slices. Expansions in an orthogonal basis are evaluated by the Clenshaw
/// recurrence in O(n) time without expanding into powers of `x`.
///
/// ```
/// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
///
/// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
/// let plan = spec.prepare(2, Default::default())?;
/// assert_eq!(plan.degree(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. W. Clenshaw, "A note on the summation of Chebyshev series",
/// *Math. Tables Aids Comput.* 9(51), 1955, 118–120.
#[derive(Debug, Clone, PartialEq)]
pub struct OrthogonalBasisPlan<R: RealScalar> {
    pub(super) spec: OrthogonalBasisSpec<R>,
    pub(super) options: PolynomialExecutionOptions,
    pub(super) recurrence: RecurrenceData<R>,
    pub(super) p0: R,
}
impl<R: RealScalar> OrthogonalBasisPlan<R> {
    pub(crate) fn initial_member(&self) -> R {
        self.p0
    }
    pub(crate) fn root_recurrence(&self, index: usize) -> (R, R, R) {
        self.recurrence[index]
    }

    pub(crate) fn root_evaluation<C: NumericScalar<Real = R>>(
        &self,
        coefficients: &[C],
        point: scioxide_core::Complex<R>,
    ) -> Result<crate::horner::ScaledHorner<R>, SciOxideError> {
        self.validate_coefficients(coefficients)?;
        self.root_iteration_evaluation::<_, 1>(coefficients, point)?
            .into_scaled()
    }

    pub(crate) fn root_iteration_evaluation<C: NumericScalar<Real = R>, const ORDER: usize>(
        &self,
        coefficients: &[C],
        point: scioxide_core::Complex<R>,
    ) -> Result<crate::horner::HornerEvaluation<scioxide_core::Complex<R>>, SciOxideError> {
        if coefficients.is_empty() {
            return Err(crate::util::empty_error(scioxide_core::argument!(
                "coefficients"
            )));
        }
        if coefficients.len() > self.recurrence.len() {
            return Err(beyond_prepared_degree());
        }
        super::recurrence::series::<_, ORDER, true, true>(
            coefficients
                .iter()
                .map(|c| scioxide_core::Complex::new(c.real_component(), c.imaginary_component())),
            point,
            self.p0,
            |n| Ok(self.recurrence[n]),
        )
    }
    /// Returns the family, normalization and affine map of the basis.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// assert_eq!(plan.spec(), spec);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn spec(&self) -> OrthogonalBasisSpec<R> {
        self.spec
    }
    /// Returns the execution options used by batch evaluation.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily, PolynomialExecutionOptions};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// assert_eq!(plan.options(), PolynomialExecutionOptions::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn options(&self) -> PolynomialExecutionOptions {
        self.options
    }
    /// Returns the maximum degree of an expansion in this basis.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// assert_eq!(plan.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn degree(&self) -> usize {
        self.recurrence.len() - 1
    }
    /// Evaluates `Σₖ cₖ Qₖ(x)` and its derivative for the coefficients `cₖ` in
    /// ascending degree.
    ///
    /// The Clenshaw recurrence takes O(n) time and no scratch. The result
    /// includes a [condition report](crate::PolynomialConditionReport) of
    /// estimates, not bounds. [`evaluate_batch_into`](Self::evaluate_batch_into)
    /// evaluates the value at many points.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `coefficients` is empty, or has more than
    ///   `degree() + 1` entries.
    /// - `NonFinite` if a coefficient or `x` is NaN or infinite.
    /// - `NumericalInstability` if the value or derivative overflows.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// // T₀ + 2T₁ + T₂ = 2x² + 2x at x = 0.5 is 1.5.
    /// let result = plan.evaluate_with_derivative(&[1.0_f64, 2.0, 1.0], 0.5)?;
    /// assert!((result.value() - 1.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_with_derivative<C: NumericScalar<Real = R>>(
        &self,
        coefficients: &[C],
        point: C,
    ) -> Result<PolynomialEvaluation<C>, SciOxideError> {
        self.validate_coefficients(coefficients)?;
        let (value, derivative, bound) =
            self.evaluate_coefficients(CoefficientView::new(coefficients), point)?;
        Ok(PolynomialEvaluation {
            value,
            derivative,
            condition: condition_report(coefficients, bound, value.abs())?,
        })
    }
    #[inline]
    fn evaluate_coefficients<C: NumericScalar<Real = R>>(
        &self,
        coefficients: CoefficientView<'_, C>,
        point: C,
    ) -> Result<(C, C, R), SciOxideError> {
        super::recurrence::admitted_series::<C, 1, false, true>(
            self.recurrence.series(coefficients),
            point,
            self.p0,
        )?
        .rounded()
    }
    #[inline]
    fn evaluate_lanes<C: NumericScalar<Real = R>, const COUNT: usize>(
        &self,
        coefficients: CoefficientView<'_, C>,
        points: [C; COUNT],
    ) -> Result<Option<super::recurrence::LaneEvaluations<C, COUNT>>, SciOxideError> {
        super::recurrence::series_lanes(self.recurrence.series(coefficients), points, self.p0)
    }

    /// Check the caller's `coefficients` against this plan.
    pub(crate) fn validate_coefficients<C: NumericScalar<Real = R>>(
        &self,
        coefficients: &[C],
    ) -> Result<(), SciOxideError> {
        if coefficients.is_empty() {
            return Err(crate::util::empty_error(scioxide_core::argument!(
                "coefficients"
            )));
        }
        if coefficients.len() > self.recurrence.len() {
            return Err(beyond_prepared_degree());
        }
        validate_slice(coefficients)
    }
    /// Returns the scratch length that
    /// [`evaluate_batch_into`](Self::evaluate_batch_into) needs for `points`
    /// points: one entry per point.
    ///
    /// # Errors
    ///
    /// Never fails; the `Result` keeps the signature of the other
    /// `workspace_requirements` methods.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// assert_eq!(plan.workspace_requirements(3)?.scratch_len(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn workspace_requirements(
        &self,
        points: usize,
    ) -> Result<PolynomialScratchRequirements, SciOxideError> {
        Ok(PolynomialScratchRequirements::new(points))
    }
    /// Evaluates the expansion `Σₖ cₖ Qₖ(x)` at each of `points` and writes the
    /// values to `output`.
    ///
    /// Points are split across the workers allowed by the plan's
    /// [`options`](Self::options) and `context`. No derivatives or condition
    /// reports are computed. `scratch` needs
    /// [`workspace_requirements(points.len())`](Self::workspace_requirements)
    /// entries. On error `output` is unchanged; see
    /// [Output on error](scioxide_core#output-on-error).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `coefficients` is empty or longer than
    ///   `degree() + 1`.
    /// - `DimensionMismatch` if `output` and `points` differ in length, or
    ///   `scratch` is too short.
    /// - `NonFinite` if a coefficient or point is NaN or infinite.
    /// - `NumericalInstability` if a value overflows.
    /// - `ExecutionStopped` on cancellation or deadline, and the errors of the
    ///   executor in `context`.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// let points = [0.0_f64, 1.0];
    /// let mut scratch = vec![0.0_f64; plan.workspace_requirements(points.len())?.scratch_len()];
    /// let mut values = [0.0_f64; 2];
    /// let cx = ExecutionContext::sequential();
    /// plan.evaluate_batch_into(&[0.0_f64, 0.0, 1.0], &points, &mut values, &mut scratch, cx)?;
    /// // T₂(x) = 2x² − 1.
    /// assert!(
    ///     values
    ///         .iter()
    ///         .zip([-1.0_f64, 1.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch_into<C: NumericScalar<Real = R>>(
        &self,
        coefficients: &[C],
        points: &[C],
        output: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.validate_coefficients(coefficients)?;
        lengths(points.len(), output.len())?;
        minimum_len(points.len(), scratch.len())?;
        validate_slice(points)?;
        let admission = self.recurrence.series(CoefficientView::new(coefficients));
        crate::execution::dispatch(
            output,
            scratch,
            0,
            self.options.resolved_policy(),
            self.options.workers(),
            context,
            |row, _| {
                super::recurrence::admitted_series::<C, 0, false, false>(
                    admission,
                    points[row],
                    self.p0,
                )?
                .value()
            },
        )
    }
}

/// Stores the coefficients of an orthogonal expansion together with its
/// [`OrthogonalBasisPlan`].
///
/// This is the basis type of [`ApproxPolynomial::orthogonal`]. Coefficients are
/// in ascending degree, and trailing zeros are kept as given.
///
/// ```
/// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
///
/// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
/// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
/// // T₀ + 2T₁ + T₂ = 2x² + 2x at x = 0.5 is 1.5.
/// assert!((p.evaluate(0.5)? - 1.5).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OrthogonalBasis<C: NumericScalar> {
    pub(super) coefficients: Coefficients<C>,
    pub(super) plan: OrthogonalBasisPlan<C::Real>,
}
impl<C: NumericScalar> OrthogonalBasis<C> {
    pub(super) fn new(coefficients: Vec<C>, plan: OrthogonalBasisPlan<C::Real>) -> Self {
        Self {
            coefficients: Coefficients::new(coefficients),
            plan,
        }
    }
    /// Returns the plan of the basis.
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// assert_eq!(p.basis().plan().degree(), p.degree());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn plan(&self) -> &OrthogonalBasisPlan<C::Real> {
        &self.plan
    }
}
impl<C: NumericScalar> PolynomialBasis<C> for OrthogonalBasis<C> {
    fn coefficients(&self) -> &[C] {
        &self.coefficients
    }
}
impl<C: NumericScalar> crate::basis::private::Prepared<C> for OrthogonalBasis<C> {
    fn root_condition_prepared(
        &self,
        point: C,
        _scratch: &mut [C],
    ) -> Result<crate::RootConditionReport<C::Real>, SciOxideError> {
        let point =
            scioxide_core::Complex::new(point.real_component(), point.imaginary_component());
        let evaluation = self.plan.root_evaluation(&self.coefficients, point)?;
        crate::root_diagnostics::condition_from_scaled(
            evaluation.bound,
            evaluation.value.magnitude()?,
            evaluation.derivative.magnitude()?,
            ScaledValue::from_value(point)?.magnitude()?,
        )
    }
    #[inline]
    fn evaluate_prepared(
        &self,
        point: C,
        _scratch: &mut [C],
    ) -> Result<(C, C, crate::basis::EvaluationEnvelope<C::Real>), SciOxideError> {
        self.plan
            .evaluate_coefficients(self.coefficients.view(), point)
            .map(crate::basis::absolute_envelope)
    }
    fn materialize_monomial(&self) -> Result<Vec<C>, SciOxideError> {
        conversion::coefficients(&self.coefficients, Some(&self.plan), None)
    }
}
// Keep chunk preparation out of the per-point iterator path. The returned
// reports borrow no lane storage; immutable polynomial metadata is reused.
struct OrthogonalEvaluations<'a, C: NumericScalar> {
    polynomial: &'a ApproxPolynomial<C, OrthogonalBasis<C>>,
    points: &'a [C],
    index: usize,
    batch_start: usize,
    batch_end: usize,
    native: Option<super::recurrence::LaneEvaluations<C, 8>>,
    short: bool,
}
impl<C: NumericScalar> OrthogonalEvaluations<'_, C> {
    #[inline(never)]
    fn prepare(&mut self) {
        let polynomial = self.polynomial;
        let plan = &polynomial.basis().plan;
        let width = if self.short { 4 } else { 8 };
        let end = self.index.saturating_add(width).min(self.points.len());
        let points = &self.points[self.index..end];
        let coefficients = polynomial.basis().coefficients.view();
        let candidate = if self.short {
            let lanes = core::array::from_fn(|i| points.get(i).copied().unwrap_or(C::ZERO));
            plan.evaluate_lanes::<C, 4>(coefficients, lanes)
                .map(|candidate| {
                    candidate.map(|values| {
                        core::array::from_fn(|i| {
                            values
                                .get(i)
                                .copied()
                                .unwrap_or((C::ZERO, C::ZERO, C::Real::ZERO))
                        })
                    })
                })
        } else {
            let lanes = core::array::from_fn(|i| points.get(i).copied().unwrap_or(C::ZERO));
            plan.evaluate_lanes::<C, 8>(coefficients, lanes)
        };
        // A discarded native candidate is resolved pointwise, preserving
        // scalar validation and extended-range errors for each requested item.
        self.native = candidate.ok().flatten();
        self.batch_start = self.index;
        self.batch_end = end;
    }
}
impl<C: NumericScalar> Iterator for OrthogonalEvaluations<'_, C> {
    type Item = Result<PolynomialEvaluation<C>, SciOxideError>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let point = *self.points.get(self.index)?;
        if self.index == self.batch_end {
            self.prepare();
        }
        let lane = self.index - self.batch_start;
        self.index += 1;
        Some(if let Some(values) = self.native.as_ref() {
            self.polynomial
                .evaluation_from_state(crate::basis::absolute_envelope(values[lane]))
        } else {
            self.polynomial.evaluate_with_derivative(point)
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.points.len() - self.index;
        (remaining, Some(remaining))
    }
}
impl<C: NumericScalar> ExactSizeIterator for OrthogonalEvaluations<'_, C> {}
impl<C: NumericScalar> core::iter::FusedIterator for OrthogonalEvaluations<'_, C> {}

impl<C: NumericScalar> ApproxPolynomial<C, OrthogonalBasis<C>> {
    /// Returns an iterator that evaluates the polynomial and its derivative at
    /// each of `points`, in order.
    ///
    /// Each item equals the result of
    /// [`evaluate_with_derivative`](Self::evaluate_with_derivative) at that
    /// point. Points are processed in groups of four or eight that share
    /// coefficient loads, which is faster than separate calls. The iterator does
    /// not allocate.
    ///
    /// # Errors
    ///
    /// Each item has the errors of
    /// [`evaluate_with_derivative`](Self::evaluate_with_derivative). An error at
    /// one point does not stop the iterator.
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::Legendre)?;
    /// // P₁(x) = x.
    /// let p = ApproxPolynomial::orthogonal(vec![0.0_f64, 1.0], spec)?;
    /// for (result, expected) in p
    ///     .evaluate_many_with_derivative(&[-0.5_f64, 0.5])
    ///     .zip([-0.5_f64, 0.5])
    /// {
    ///     assert!((result?.value() - expected).abs() < 1e-13);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_many_with_derivative<'a>(
        &'a self,
        points: &'a [C],
    ) -> impl Iterator<Item = Result<PolynomialEvaluation<C>, SciOxideError>> + 'a {
        OrthogonalEvaluations {
            polynomial: self,
            points,
            index: 0,
            batch_start: 0,
            batch_end: 0,
            native: None,
            short: self.basis().coefficients().len() <= 32,
        }
    }

    /// Creates the polynomial `Σₖ cₖ Qₖ(x)` from coefficients in ascending
    /// degree in the basis described by `spec`.
    ///
    /// The constructor prepares the recurrence for degree
    /// `coefficients.len() − 1` with default execution options, in O(n) time and
    /// storage, and uses the default policy; use
    /// [`orthogonal_with_plan`](Self::orthogonal_with_plan) to keep a plan or
    /// choose the policy.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `coefficients` is empty.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - The errors of [`OrthogonalBasisSpec::prepare`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// // T₀ + 2T₁ + T₂ on [0, 2], with t = x − 1: at x = 0.5, t = −0.5 and the
    /// // sum 2t² + 2t is −0.5.
    /// let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::ChebyshevT, 0.0_f64, 2.0)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// assert!((p.evaluate(0.5)? + 0.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn orthogonal(
        coefficients: Vec<C>,
        spec: OrthogonalBasisSpec<C::Real>,
    ) -> Result<Self, SciOxideError> {
        let degree = coefficients
            .len()
            .checked_sub(1)
            .ok_or_else(|| crate::util::empty_error(scioxide_core::argument!("coefficients")))?;
        Self::orthogonal_with_plan(
            coefficients,
            spec.prepare(degree, PolynomialExecutionOptions::new())?,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Creates the polynomial `Σₖ cₖ Qₖ(x)` from coefficients in ascending
    /// degree, a prepared basis and a policy.
    ///
    /// This is the configured form of [`orthogonal`](Self::orthogonal). The
    /// polynomial keeps `plan`, including its maximum degree and execution
    /// options, even when `coefficients` is shorter than `plan.degree() + 1`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `coefficients` is empty (reason `Empty`), or longer
    ///   than `plan.degree() + 1` (reason `Inconsistent`).
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if the tolerance or rank threshold of `policy`
    ///   overflows at the scale of the coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// // One plan of degree 4 serves expansions of every degree up to 4.
    /// let plan = spec.prepare(4, PolynomialExecutionOptions::new())?;
    /// let policy =
    ///     PolynomialApproximationPolicy::default().with_rank_policy(scioxide_linalg::RankPolicy::Exact);
    /// let p = ApproxPolynomial::orthogonal_with_plan(vec![1.0_f64, 2.0, 1.0], plan, policy)?;
    /// // T₀ + 2T₁ + T₂ = 2x² + 2x at x = 0.5 is 1.5.
    /// assert!((p.evaluate(0.5)? - 1.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn orthogonal_with_plan(
        coefficients: Vec<C>,
        plan: OrthogonalBasisPlan<C::Real>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        plan.validate_coefficients(&coefficients)?;
        Self::from_basis(OrthogonalBasis::new(coefficients, plan), policy)
    }
}

pub(super) fn lengths(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(expected, actual))
    }
}
/// Coefficients beyond the prepared maximum degree contradict the plan.
fn beyond_prepared_degree() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("coefficients"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "coefficients must fit the prepared degree",
    )
}
pub(super) fn domain(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::domain_error(argument, scioxide_core::DomainReason::OutsideDomain, detail)
}
