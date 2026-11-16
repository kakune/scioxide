//! Detailed results of special-function evaluations.
//!
//! A kernel produces one [`Evaluation`]: its ordinary value, status and the
//! data its detailed forms need. A plain function takes the value with
//! [`Evaluation::value`]. A detailed entry point forms the error estimates and
//! the scaled and logarithmic forms from the same evaluation, and nowhere else.
//! Both read the same ordinary value, so the plain and detailed values are
//! bit-identical.
//!
//! Every inexact form carries one [`Bound`], derived from the evaluation or
//! measured against reference values. Bounds scale with the value, so a
//! representable result, including a subnormal one, never fails because of
//! its estimate.

use crate::ValueWithError;
use scioxide_core::{
    AlgebraicScalar, AlgorithmId, ConvergenceDiagnostics, ConvergenceReason, EstimateReliability,
    EvaluationCounts, LogValue, NumericScalar, NumericalIssue, RealScalar, Scaled, ScaledValue,
    SciOxideError,
};

/// Reports how a special-function evaluation finished.
///
/// The termination does not say whether the error estimate meets an accuracy
/// target; read the estimate from [`SpecialFunctionResult::value`]. See
/// [detailed results](crate#detailed-results).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum SpecialFunctionTermination {
    /// An exact identity or a tabulated exact value gave the result.
    Exact,
    /// The numerical method met its convergence criterion.
    Converged,
    /// The result is a documented mathematical limit, such as `Γ(0⁺) = +∞` where
    /// the function documents it.
    Limit,
    /// The work budget ran out. The result holds the estimate reached so far;
    /// [`SpecialFunctionResult::try_value`] and the plain function return
    /// `ConvergenceFailure` instead.
    IterationLimit,
}

impl SpecialFunctionTermination {
    /// Returns `true` for `Exact`, `Converged` and `Limit`.
    #[inline]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Exact | Self::Converged | Self::Limit)
    }
}

/// Says whether the result is available as an ordinary scalar value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum ScalarRepresentation {
    /// The value is finite.
    ///
    /// A nonzero result that rounds to zero is an ordinary zero, with the sign
    /// of the result where it is known and an error estimate that covers the
    /// true magnitude. The scaled and logarithmic forms keep the magnitude where
    /// they can represent it.
    Finite,
    /// The result overflows the scalar type. The scaled and logarithmic forms
    /// are available where they can represent it.
    Overflow,
    /// The result is a documented infinite limit, returned as `±∞`.
    InfiniteLimit,
}

/// Holds the value of a special function with error estimates and diagnostics.
///
/// The `*_with_options` and `*_scoped` functions return this type. It contains
/// up to four forms of the result, each with its own error estimate:
///
/// | Form | Accessor | Error units |
/// |---|---|---|
/// | ordinary value `v` | [`value`](Self::value) | units of `v` |
/// | scaled value, each component stored as `m·2ᵉ` with an integer `e` | [`scaled`](Self::scaled) | scaled coordinates |
/// | logarithmic form `v = phase·exp(log_magnitude)` | [`logarithmic`](Self::logarithmic) | natural-log units |
/// | derivative `dv/dx` | [`derivative`](Self::derivative) | units of the derivative |
///
/// A form whose estimate would be meaningless is `None`. The logarithmic form
/// is omitted when the relative error reaches ½. The ordinary value equals,
/// bit for bit, the value the plain function returns. `V` is a real or complex
/// scalar; error estimates are always real.
///
/// # Examples
///
/// ```
/// // Γ(5) = 24, with an error estimate and the termination status.
/// let result = scioxide_special::gamma_with_options(5.0_f64, Default::default())?;
/// assert!(result.termination().is_success());
/// let value = result.value().expect("Γ(5) is representable");
/// assert!((*value.value() - 24.0).abs() <= value.absolute_error() + 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
#[allow(clippy::type_complexity)]
pub struct SpecialFunctionResult<V: NumericScalar> {
    value: Option<ValueWithError<V, V::Real>>,
    scaled: Option<ValueWithError<ScaledValue<V>, V::Real>>,
    logarithmic: Option<ValueWithError<LogValue<V, V::Real>, V::Real>>,
    derivative: Option<ValueWithError<V, V::Real>>,
    algorithm: AlgorithmId,
    counts: EvaluationCounts,
    termination: SpecialFunctionTermination,
    scalar_representation: ScalarRepresentation,
}

#[allow(clippy::type_complexity)]
impl<V: NumericScalar> SpecialFunctionResult<V> {
    /// Returns an exact zero with zero work counts, for initializing buffers.
    ///
    /// No function is evaluated and no derivative is supplied.
    ///
    /// ```
    /// let output = [scioxide_special::SpecialFunctionResult::<f64>::zero(); 4];
    /// assert_eq!(output[0].try_value()?, 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn zero() -> Self {
        let zero = ValueWithError::exact_zero();
        Self::new(
            Some(zero),
            None,
            None,
            None,
            scioxide_core::algorithm_id!("special", "zero", 1),
            EvaluationCounts::new(),
            SpecialFunctionTermination::Exact,
            ScalarRepresentation::Finite,
        )
    }

    /// Converts the forms to another scalar type with the same real type.
    #[inline]
    pub(crate) fn convert<T>(self) -> Result<SpecialFunctionResult<T>, SciOxideError>
    where
        T: NumericScalar<Real = V::Real>,
    {
        let value = self
            .value()
            .copied()
            .map(|v| v.try_map(|v| Ok(crate::dispatch::convert_scalar::<V, T>(v))))
            .transpose()?;
        let scaled = self
            .scaled()
            .copied()
            .map(|v| {
                v.try_map(|s: ScaledValue<V>| {
                    Ok(ScaledValue::<T>::from_components(s.real(), s.imaginary()))
                })
            })
            .transpose()?;
        let logarithmic = self
            .logarithmic()
            .copied()
            .map(|v| {
                v.try_map(|log: LogValue<V, T::Real>| match log.log_magnitude() {
                    Some(magnitude) => LogValue::<T, T::Real>::from_scaled_log(
                        ScaledValue::<T>::from_components(
                            log.scaled_phase().real(),
                            log.scaled_phase().imaginary(),
                        ),
                        magnitude,
                    ),
                    None => Ok(LogValue::ZERO),
                })
            })
            .transpose()?;
        let derivative = self
            .derivative()
            .copied()
            .map(|v| v.try_map(|v| Ok(crate::dispatch::convert_scalar::<V, T>(v))))
            .transpose()?;
        Ok(SpecialFunctionResult::new(
            value,
            scaled,
            logarithmic,
            derivative,
            self.algorithm(),
            self.counts(),
            self.termination(),
            self.scalar_representation(),
        ))
    }

    pub(crate) fn with_counts(mut self, counts: EvaluationCounts) -> Self {
        self.counts = counts;
        self
    }
    /// Returns an error if the evaluation did not complete.
    pub(crate) fn require_completion(&self) -> Result<(), SciOxideError> {
        require_completion(self.termination, self.algorithm, self.counts)
    }
    /// Creates a detailed result from validated forms.
    #[allow(clippy::too_many_arguments)]
    const fn new(
        value: Option<ValueWithError<V, V::Real>>,
        scaled: Option<ValueWithError<ScaledValue<V>, V::Real>>,
        logarithmic: Option<ValueWithError<LogValue<V, V::Real>, V::Real>>,
        derivative: Option<ValueWithError<V, V::Real>>,
        algorithm: AlgorithmId,
        counts: EvaluationCounts,
        termination: SpecialFunctionTermination,
        scalar_representation: ScalarRepresentation,
    ) -> Self {
        Self {
            value,
            scaled,
            logarithmic,
            derivative,
            algorithm,
            counts,
            termination,
            scalar_representation,
        }
    }

    /// Returns the ordinary value and its error estimate, or `None` if the value
    /// overflows. Documented infinite limits are returned as `±∞`.
    pub const fn value(&self) -> Option<&ValueWithError<V, V::Real>> {
        self.value.as_ref()
    }

    /// Returns the value with each component stored as a mantissa and a binary
    /// exponent, if available. This form exists even when the value overflows.
    pub const fn scaled(&self) -> Option<&ValueWithError<ScaledValue<V>, V::Real>> {
        self.scaled.as_ref()
    }

    /// Returns the phase and the natural logarithm of the magnitude, if available.
    pub const fn logarithmic(&self) -> Option<&ValueWithError<LogValue<V, V::Real>, V::Real>> {
        self.logarithmic.as_ref()
    }

    /// Returns the derivative with respect to the argument, if the function
    /// computes it.
    pub const fn derivative(&self) -> Option<&ValueWithError<V, V::Real>> {
        self.derivative.as_ref()
    }

    /// Returns the identifier of the algorithm that produced the result.
    pub const fn algorithm(&self) -> AlgorithmId {
        self.algorithm
    }

    /// Returns the work counts of the evaluation, including failed attempts.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts
    }

    /// Returns how the evaluation finished.
    pub const fn termination(&self) -> SpecialFunctionTermination {
        self.termination
    }

    /// Returns whether the ordinary value is finite, overflows or is an infinite limit.
    pub const fn scalar_representation(&self) -> ScalarRepresentation {
        self.scalar_representation
    }

    /// Returns the ordinary value, as the plain function does.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` (`IterationLimit`) if the termination is
    ///   [`SpecialFunctionTermination::IterationLimit`].
    /// - `NumericalInstability` (`Overflow`) if the value overflows the scalar type.
    #[inline]
    pub fn try_value(self) -> Result<V, SciOxideError> {
        project(
            self.value.map(ValueWithError::into_value),
            self.scalar_representation,
            self.termination,
            self.algorithm,
            self.counts,
        )
    }

    /// Returns the derivative with respect to the argument, without its error estimate.
    ///
    /// Use [`Self::derivative`] when the estimate is needed. A function documents whether its
    /// detailed result carries a derivative.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` (`IterationLimit`) if the termination is
    ///   [`SpecialFunctionTermination::IterationLimit`].
    /// - `NumericalInstability` (`Breakdown`) if the result has no derivative: the function
    ///   publishes none, or the derivative is outside the range of the scalar type or singular
    ///   at the argument.
    ///
    /// # Examples
    ///
    /// ```
    /// // Ai′(1) with Ai(1)
    /// let ai = scioxide_special::airy_ai_with_options(1.0_f64, Default::default())?;
    /// assert!((ai.try_derivative()? + 0.15914744129679328).abs() < 1e-14);
    /// assert!((ai.try_value()? - 0.13529241631288141).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_derivative(self) -> Result<V, SciOxideError> {
        self.require_completion()?;
        self.derivative
            .map(ValueWithError::into_value)
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "the evaluation did not produce an ordinary scalar derivative",
                )
            })
    }
}

/// Returns the ordinary value for plain functions and
/// [`SpecialFunctionResult::try_value`].
// Propagate the caller's successful status through projection. An out-of-line
// call otherwise keeps the large diagnostic payload alive even for value-only
// normal kernels whose only outcomes are Converged and Limit.
#[inline(always)]
fn project<V>(
    value: Option<V>,
    representation: ScalarRepresentation,
    termination: SpecialFunctionTermination,
    algorithm: AlgorithmId,
    counts: EvaluationCounts,
) -> Result<V, SciOxideError> {
    require_completion(termination, algorithm, counts)?;
    if let Some(value) = value {
        return Ok(value);
    }
    Err(projection_error(representation))
}

/// Returns the error for a value that is not available as an ordinary scalar.
pub(crate) fn projection_error(representation: ScalarRepresentation) -> SciOxideError {
    let (issue, detail) = match representation {
        ScalarRepresentation::Overflow => (
            NumericalIssue::Overflow,
            "the result is available only in scaled or logarithmic form",
        ),
        ScalarRepresentation::Finite | ScalarRepresentation::InfiniteLimit => (
            NumericalIssue::Breakdown,
            "the evaluation did not produce an ordinary scalar value",
        ),
    };
    SciOxideError::numerical_instability(issue, detail)
}

/// Returns the error for an exhausted work budget, with its counts.
pub(crate) fn exhausted(diagnostics: ConvergenceDiagnostics) -> SciOxideError {
    SciOxideError::convergence_failure(
        ConvergenceReason::IterationLimit,
        diagnostics,
        "the special-function iteration budget was exhausted",
    )
}

pub(crate) fn exhausted_work(iterations: usize) -> SciOxideError {
    exhausted(
        ConvergenceDiagnostics::new()
            .with_counts(EvaluationCounts::from_iterations(iterations).into()),
    )
}

/// Returns an error with the attempted counts if the evaluation did not complete.
#[inline(always)]
pub(crate) fn require_completion(
    termination: SpecialFunctionTermination,
    _algorithm: AlgorithmId,
    counts: EvaluationCounts,
) -> Result<(), SciOxideError> {
    if termination.is_success() {
        Ok(())
    } else {
        Err(exhausted(
            ConvergenceDiagnostics::new().with_counts(counts.into()),
        ))
    }
}

/// One kernel evaluation, before any detailed representation is formed.
///
/// The ordinary value, its range status, termination, counts and algorithm identity are
/// computed eagerly; `form` holds what the detailed result needs for the error estimates and the
/// scaled, logarithmic and derivative forms. `form` is `None` when no estimate was computed.
#[derive(Clone, Copy)]
#[must_use]
pub(crate) struct Evaluation<V: NumericScalar, F = Form<V>> {
    value: Option<V>,
    representation: ScalarRepresentation,
    termination: SpecialFunctionTermination,
    counts: EvaluationCounts,
    algorithm: AlgorithmId,

    form: Option<F>,
}

/// An optional representation `T` of a `V` value with its error estimate.
pub(crate) type Estimated<T, V> = Option<ValueWithError<T, <V as NumericScalar>::Real>>;

/// The detailed forms of one evaluation.
pub(crate) struct Forms<V: NumericScalar> {
    pub(crate) value: Estimated<V, V>,
    pub(crate) scaled: Estimated<ScaledValue<V>, V>,
    pub(crate) logarithmic: Estimated<LogValue<V, V::Real>, V>,
    pub(crate) derivative: Estimated<V, V>,
}

impl<V: NumericScalar> Forms<V> {
    pub(crate) const NONE: Self = Self {
        value: None,
        scaled: None,
        logarithmic: None,
        derivative: None,
    };
}

/// Forms the detailed forms of a kernel evaluation.
pub(crate) trait Publish<V: NumericScalar> {
    /// Form the error estimates and representations of `value`, the ordinary
    /// projection the plain function returns.
    fn publish(self, value: Option<V>) -> Result<Forms<V>, SciOxideError>;
}

impl<V: NumericScalar> Publish<V> for Forms<V> {
    fn publish(self, _: Option<V>) -> Result<Forms<V>, SciOxideError> {
        Ok(self)
    }
}

impl<V: NumericScalar, F> Evaluation<V, F> {
    /// Assemble an evaluation from its kernel projection and lazy detailed forms.
    /// The detailed result and [`Evaluation::value`] check the same invariants.
    pub(crate) fn from_parts(
        value: Option<V>,
        representation: ScalarRepresentation,
        termination: SpecialFunctionTermination,
        counts: EvaluationCounts,
        algorithm: AlgorithmId,
        form: Option<F>,
    ) -> Self {
        Self {
            value,
            representation,
            termination,
            counts,
            algorithm,
            form,
        }
    }

    pub(crate) fn ordinary(&self) -> Option<V> {
        self.value
    }

    pub(crate) fn representation(&self) -> ScalarRepresentation {
        self.representation
    }

    pub(crate) fn algorithm(&self) -> AlgorithmId {
        self.algorithm
    }

    pub(crate) fn form(&self) -> Option<&F> {
        self.form.as_ref()
    }

    /// Replace a projection after an independently established range-boundary correction.
    pub(crate) fn with_projection(
        mut self,
        value: Option<V>,
        representation: ScalarRepresentation,
    ) -> Self {
        self.value = value;
        self.representation = representation;
        self
    }

    // Validate beside projection so scalar calls can discard lazy diagnostic storage.
    #[inline(always)]
    fn validate(&self) -> Result<(), SciOxideError> {
        if self.termination == SpecialFunctionTermination::IterationLimit
            && self.value.is_none()
            && self.form.is_none()
        {
            return Err(exhausted_work(self.counts.iterations()));
        }
        let valid = match (self.representation, self.value) {
            (ScalarRepresentation::Finite, Some(value)) => value.has_finite_components(),
            (ScalarRepresentation::Overflow, None) => true,
            (ScalarRepresentation::InfiniteLimit, Some(value)) => {
                self.termination == SpecialFunctionTermination::Limit
                    && !value.real_component().is_nan()
                    && !value.imaginary_component().is_nan()
                    && !value.has_finite_components()
            }
            _ => false,
        };
        if !valid {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Output,
                "special-function scalar projection disagrees with its representation",
            ));
        }
        Ok(())
    }

    /// The ordinary scalar a plain function returns.
    ///
    /// # Errors
    ///
    /// Returns the failure [`SpecialFunctionResult::try_value`] reports for
    /// the detailed result.
    // Keep a known successful status visible so scalar calls can discard diagnostics.
    #[inline(always)]
    pub(crate) fn value(self) -> Result<V, SciOxideError> {
        self.validate()?;
        project(
            self.value,
            self.representation,
            self.termination,
            self.algorithm,
            self.counts,
        )
    }

    /// Attempted iteration and provider counts.
    pub(crate) fn counts(&self) -> EvaluationCounts {
        self.counts
    }

    /// Termination status.
    pub(crate) fn termination(&self) -> SpecialFunctionTermination {
        self.termination
    }

    /// Replace the attempted work counts.
    pub(crate) fn with_counts(mut self, counts: EvaluationCounts) -> Self {
        self.counts = counts;
        self
    }

    /// The same evaluation with its form transformed by `map`.
    pub(crate) fn map_form<G>(self, map: impl FnOnce(F) -> G) -> Evaluation<V, G> {
        Evaluation {
            value: self.value,
            representation: self.representation,
            termination: self.termination,
            counts: self.counts,
            algorithm: self.algorithm,

            form: self.form.map(map),
        }
    }

    /// Forms the detailed result.
    ///
    /// # Errors
    ///
    /// Returns a structured error when a representation cannot be formed.
    pub(crate) fn publish(self) -> Result<SpecialFunctionResult<V>, SciOxideError>
    where
        F: Publish<V>,
    {
        self.validate()?;
        let forms = match self.form {
            Some(form) => form.publish(self.value)?,
            None => Forms::NONE,
        };
        if self.termination == SpecialFunctionTermination::IterationLimit
            && forms.value.is_none()
            && forms.scaled.is_none()
            && forms.logarithmic.is_none()
        {
            return Err(exhausted_work(self.counts.iterations()));
        }
        // A relative bound vanishes at zero, although an inexact zero may be
        // the rounding of any magnitude below half a quantum: an inexact zero
        // carries at least one destination quantum.
        let value = match forms.value {
            Some(value)
                if !matches!(
                    self.termination,
                    SpecialFunctionTermination::Exact | SpecialFunctionTermination::Limit
                ) && value.absolute_error() == V::Real::ZERO
                    && value.value().real_component() == V::Real::ZERO
                    && value.value().imaginary_component() == V::Real::ZERO =>
            {
                Some(ValueWithError::new(
                    *value.value(),
                    V::Real::MIN_SUBNORMAL,
                    None,
                    value.reliability(),
                )?)
            }
            value => value,
        };
        Ok(SpecialFunctionResult::new(
            value,
            forms.scaled,
            forms.logarithmic,
            forms.derivative,
            self.algorithm,
            self.counts,
            self.termination,
            self.representation,
        ))
    }
}

/// Detailed forms from one of two kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<V: NumericScalar, A: Publish<V>, B: Publish<V>> Publish<V> for Either<A, B> {
    fn publish(self, value: Option<V>) -> Result<Forms<V>, SciOxideError> {
        match self {
            Self::Left(form) => form.publish(value),
            Self::Right(form) => form.publish(value),
        }
    }
}

/// Error bound of one representation, in that representation's units, and its basis.
///
/// Remainder and rounding models derived from the evaluation are `ModelBased`.
/// Envelopes in ulps measured against reference values are
/// `StatisticallyCalibrated`. Magnitudes without a coverage statement are
/// `Heuristic`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bound<S> {
    pub(crate) absolute: S,
    pub(crate) reliability: EstimateReliability,
}

impl<S: RealScalar> Bound<S> {
    fn new(absolute: S, reliability: EstimateReliability) -> Self {
        Self {
            absolute: if absolute.is_finite() {
                absolute
            } else {
                S::MAX
            },
            reliability,
        }
    }

    /// An exact identity, tabulated value or documented limit.
    pub(crate) fn exact() -> Self {
        Self::new(S::ZERO, EstimateReliability::ModelBased)
    }

    /// A remainder or rounding bound derived from the evaluation performed.
    pub(crate) fn derived(absolute: S) -> Self {
        Self::new(absolute, EstimateReliability::ModelBased)
    }

    /// A magnitude estimate without a calibrated coverage statement.
    pub(crate) fn heuristic(absolute: S) -> Self {
        Self::new(absolute, EstimateReliability::Heuristic)
    }

    /// Absolute envelope measured on the kernel's ill-conditioned reference points.
    pub(crate) fn calibrated_absolute(absolute: S) -> Self {
        Self::new(absolute, EstimateReliability::StatisticallyCalibrated)
    }

    /// An envelope of `ulps` units of `value` (see [`ulp_bound`]), measured
    /// against reference values for the kernel.
    pub(crate) fn envelope(value: S, ulps: impl Into<Ulps>) -> Result<Self, SciOxideError> {
        Ok(Self::new(
            ulp_bound(value, S::try_from_f64_lossy(ulps.into().of::<S>())?),
            EstimateReliability::StatisticallyCalibrated,
        ))
    }

    /// One correctly rounded final operation: half a unit of `value`, with
    /// at least one subnormal quantum to represent a positive allowance.
    pub(crate) fn rounded(value: S) -> Self {
        // Half a subnormal quantum cannot be represented. Round the positive
        // rounding allowance upward instead of publishing an exact result.
        Self::derived(ulp_bound(value, S::HALF).max(S::MIN_SUBNORMAL))
    }

    /// Sum of two contributions, with the weaker reliability.
    pub(crate) fn plus(self, other: Self) -> Self {
        Self::new(
            self.absolute + other.absolute,
            weaker(self.reliability, other.reliability),
        )
    }

    /// Add `quanta` subnormal quanta for a final result of `magnitude` below
    /// the normal range, where rounding is absolute. Normal-range rounding is
    /// relative and part of the bound; skipping the term there also keeps
    /// subnormal operands, slow on common hardware, off the normal path.
    pub(crate) fn with_subnormal_rounding(self, magnitude: S, quanta: S) -> Self {
        if magnitude < S::MIN_POSITIVE {
            self.plus(Self::derived(quanta * S::MIN_SUBNORMAL))
        } else {
            self
        }
    }
}

/// Envelope widths of one kernel, in ulps of `f64` and `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ulps {
    binary64: f64,
    binary32: f64,
}

impl Ulps {
    pub(crate) const fn new(binary64: f64, binary32: f64) -> Self {
        Self { binary64, binary32 }
    }

    fn of<S: RealScalar>(self) -> f64 {
        if S::MANTISSA_DIGITS <= f32::MANTISSA_DIGITS {
            self.binary32
        } else {
            self.binary64
        }
    }
}

impl From<f64> for Ulps {
    /// The same width in both precisions.
    fn from(ulps: f64) -> Self {
        Self::new(ulps, ulps)
    }
}
fn weaker(first: EstimateReliability, second: EstimateReliability) -> EstimateReliability {
    let either = |class| first == class || second == class;
    if either(EstimateReliability::Heuristic) {
        EstimateReliability::Heuristic
    } else if either(EstimateReliability::StatisticallyCalibrated) {
        EstimateReliability::StatisticallyCalibrated
    } else {
        EstimateReliability::ModelBased
    }
}

/// `ulps` units of `|value|·ε`, never below `ulps` subnormal quanta.
///
/// One unit is at least the spacing of the scalars at `value`, so the bound
/// scales with the value and stays representable for zero and subnormal values.
pub(crate) fn ulp_bound<S: RealScalar>(value: S, ulps: S) -> S {
    let unit = value.abs() * S::EPSILON;
    // Subnormal operands are slow on common hardware. Normal-range units never
    // form the floor, and the floor `ulps·η` is shifted on the integer grid.
    if unit >= S::MIN_POSITIVE {
        (ulps * unit).min(S::MAX)
    } else {
        let floor = (ulps * S::MIN_POSITIVE).scalbn(1 - S::MANTISSA_DIGITS.cast_signed());
        if unit == S::ZERO {
            floor
        } else {
            (ulps * unit).max(floor)
        }
    }
}

/// Natural-log error of a magnitude with relative error `relative`, including
/// the rounding of `log_magnitude`. There is none once the error interval may
/// reach zero (`relative ≥ ½`).
#[inline]
pub(crate) fn log_error<S: RealScalar>(relative: S, log_magnitude: S) -> Option<S> {
    (relative < S::HALF).then(|| relative / (S::ONE - relative) + S::EPSILON * log_magnitude.abs())
}

#[inline]
pub(crate) fn relative_error<V, R>(value: V, absolute: R) -> Option<R>
where
    V: NumericScalar<Real = R>,
    R: RealScalar,
{
    let magnitude = value.abs();
    (magnitude != R::ZERO).then(|| (absolute / magnitude).min(R::MAX))
}

/// A relative bound expressed in the coordinates of a scaled value.
/// Rounds the multiplication outward and keeps the relative estimate.
#[inline]
pub(crate) fn scaled_estimate<V: NumericScalar>(
    value: ScaledValue<V>,
    relative: V::Real,
    reliability: EstimateReliability,
) -> Result<ValueWithError<ScaledValue<V>, V::Real>, SciOxideError> {
    let magnitude = value.scaled_magnitude();
    let product = (relative * magnitude).min(V::Real::MAX);
    let absolute =
        if product > V::Real::ZERO || (relative > V::Real::ZERO && magnitude > V::Real::ZERO) {
            product.next_after(V::Real::MAX)
        } else {
            product
        };
    ValueWithError::new(
        value,
        absolute,
        (magnitude != V::Real::ZERO).then_some(relative),
        reliability,
    )
}

/// Narrows a nonnegative finite bound outward, saturating at the target maximum.
/// Positive bounds below the target range keep one subnormal quantum.
#[inline]
pub(crate) fn outward<R: RealScalar>(bound: f64) -> Result<R, SciOxideError> {
    let capped = bound.min(R::MAX.into_f64()).max(0.0);
    let rounded = R::try_from_f64_lossy(capped)?;
    Ok(if rounded.into_f64() < capped {
        rounded.next_after(R::MAX)
    } else {
        rounded
    })
}

/// Converts a widened positive value and keeps its overflow status.
pub(crate) fn positive_projection<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    // An infinite output is representation metadata, rather than a
    // nonfinite input to a checked scalar conversion.
    if value.is_infinite() {
        Ok(S::INFINITY)
    } else {
        S::try_from_f64_lossy(value)
    }
}

/// Scalar projection of a logarithmic evaluation.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Projection<S> {
    /// One correctly rounded `exp(log_magnitude)`.
    Exp,
    /// A separately evaluated magnitude with its own bound.
    Direct(S, Bound<S>),
}

/// The detailed forms of the shared real and complex kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Form<V: NumericScalar> {
    /// Forms evaluated at the target precision.
    Native(Standard<V>),
    /// Forms of an `f64` evaluation, narrowed when the detailed result is formed.
    Narrowed(Narrowed),
}

/// Forms evaluated at one precision.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Standard<V: NumericScalar> {
    /// A finite value with its bound and an optional derivative.
    Finite {
        bound: Bound<V::Real>,
        derivative: Option<(V, Bound<V::Real>)>,
    },
    /// A signed logarithmic magnitude whose scalar projection may be out of
    /// range. `bound` is that projection's bound; `None` derives it from the
    /// logarithmic bound and one rounding of `exp`.
    Logarithmic {
        sign: V::Real,
        log_magnitude: V::Real,
        log_bound: Bound<V::Real>,
        bound: Option<Bound<V::Real>>,
        derivative: Option<(V, Bound<V::Real>)>,
    },
    /// A finite value whose bound is the measured envelope of `ulps` units,
    /// formed only for the detailed result.
    Calibrated { ulps: Ulps },
    /// A documented mathematical limit.
    Limit,
}

/// An `f64` evaluation, narrowed to the target type when the detailed result is formed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Narrowed {
    wide: Standard<f64>,
    value: Option<f64>,
    termination: SpecialFunctionTermination,
}

impl<V: NumericScalar> Evaluation<V> {
    /// A finite evaluated value and its bound.
    #[inline]
    pub(crate) fn finite(
        value: V,
        bound: Bound<V::Real>,
        derivative: Option<(V, Bound<V::Real>)>,
        algorithm: AlgorithmId,
        iterations: usize,
        termination: SpecialFunctionTermination,
    ) -> Self {
        Self {
            value: Some(value),
            representation: ScalarRepresentation::Finite,
            termination,
            counts: EvaluationCounts::from_iterations(iterations),
            algorithm,

            form: Some(Form::Native(Standard::Finite { bound, derivative })),
        }
    }

    /// A finite value whose bound is the measured envelope of `ulps` units.
    /// The plain function never forms the bound.
    #[inline]
    pub(crate) fn calibrated(
        value: V,
        ulps: impl Into<Ulps>,
        algorithm: AlgorithmId,
        iterations: usize,
        termination: SpecialFunctionTermination,
    ) -> Self {
        Self {
            form: Some(Form::Native(Standard::Calibrated { ulps: ulps.into() })),
            ..Self::finite(
                value,
                Bound::exact(),
                None,
                algorithm,
                iterations,
                termination,
            )
        }
    }

    /// An exact algebraic value and derivative supplied by an owning kernel.
    pub(crate) fn exact(
        value: V,
        derivative: Option<V>,
        algorithm: AlgorithmId,
        iterations: usize,
    ) -> Self {
        Self::finite(
            value,
            Bound::exact(),
            derivative.map(|derivative| (derivative, Bound::exact())),
            algorithm,
            iterations,
            SpecialFunctionTermination::Exact,
        )
    }

    /// A documented mathematical limit that is intentionally non-finite in
    /// the ordinary scalar representation.
    pub(crate) fn limit(value: V, algorithm: AlgorithmId) -> Self {
        Self {
            value: Some(value),
            representation: ScalarRepresentation::InfiniteLimit,
            termination: SpecialFunctionTermination::Limit,
            counts: EvaluationCounts::new(),
            algorithm,

            form: Some(Form::Native(Standard::Limit)),
        }
    }

    /// Attach a derivative estimate the same kernel produced to a finite
    /// evaluation.
    pub(crate) fn with_derivative(mut self, derivative: Option<(V, Bound<V::Real>)>) -> Self {
        if let Some(Form::Native(
            Standard::Finite {
                derivative: slot, ..
            }
            | Standard::Logarithmic {
                derivative: slot, ..
            },
        )) = &mut self.form
        {
            *slot = derivative;
        }
        self
    }
}

impl<S: RealScalar> Evaluation<S> {
    /// A signed logarithmic evaluation whose logarithmic form remains
    /// available when its scalar projection is outside the target range.
    /// `log_bound` is in natural-log units.
    pub(crate) fn logarithmic(
        sign: S,
        log_magnitude: S,
        log_bound: Bound<S>,
        projection: Projection<S>,
        algorithm: AlgorithmId,
        iterations: usize,
    ) -> Self {
        let (magnitude, bound) = match projection {
            Projection::Exp => (log_magnitude.exp(), None),
            Projection::Direct(magnitude, bound) => (magnitude, Some(bound)),
        };
        // A magnitude that rounds to zero lies below half the smallest
        // subnormal quantum, so one quantum covers it.
        let bound = if magnitude == S::ZERO {
            Some(Bound::new(S::MIN_SUBNORMAL, log_bound.reliability))
        } else {
            bound
        };
        let (value, representation) = if magnitude.is_finite() {
            (Some(sign * magnitude), ScalarRepresentation::Finite)
        } else {
            (None, ScalarRepresentation::Overflow)
        };
        Self {
            value,
            representation,
            termination: SpecialFunctionTermination::Converged,
            counts: EvaluationCounts::from_iterations(iterations),
            algorithm,

            form: Some(Form::Native(Standard::Logarithmic {
                sign,
                log_magnitude,
                log_bound,
                bound,
                derivative: None,
            })),
        }
    }
}

impl Evaluation<f64> {
    /// Narrows an `f64` evaluation once to the target `S` and keeps its range
    /// status. The detailed forms are narrowed when they are formed.
    /// A nonzero value that rounds to zero is an ordinary signed zero whose
    /// bound includes the narrowing error.
    pub(crate) fn narrow<S: RealScalar>(self) -> Result<Evaluation<S>, SciOxideError> {
        let maximum = S::MAX;
        let overflow_boundary =
            maximum.into_f64() + (maximum - maximum.next_after(S::ZERO)).into_f64() * 0.5;
        let (value, representation) = match self.value {
            Some(wide)
                if wide.is_infinite() && self.termination == SpecialFunctionTermination::Limit =>
            {
                (
                    Some(if wide < 0.0 {
                        S::NEG_INFINITY
                    } else {
                        S::INFINITY
                    }),
                    self.representation,
                )
            }
            Some(wide) if wide.abs() >= overflow_boundary => (None, ScalarRepresentation::Overflow),
            Some(wide) => (Some(S::try_from_f64_lossy(wide)?), self.representation),
            None => (None, self.representation),
        };
        Ok(Evaluation {
            value,
            representation,
            termination: self.termination,
            counts: self.counts,
            algorithm: self.algorithm,

            form: self.form.map(|form| {
                Form::Narrowed(match form {
                    Form::Native(wide) => Narrowed {
                        wide,
                        value: self.value,
                        termination: self.termination,
                    },
                    Form::Narrowed(narrowed) => narrowed,
                })
            }),
        })
    }
}

impl<V: NumericScalar> Publish<V> for Form<V> {
    fn publish(self, value: Option<V>) -> Result<Forms<V>, SciOxideError> {
        match self {
            Self::Native(standard) => standard.publish(value),
            Self::Narrowed(narrowed) => narrowed.publish(value),
        }
    }
}

impl<V: NumericScalar> Publish<V> for Standard<V> {
    fn publish(self, value: Option<V>) -> Result<Forms<V>, SciOxideError> {
        match self {
            Self::Finite { bound, derivative } => {
                let Some(value) = value else {
                    return Ok(Forms::NONE);
                };
                let (ordinary, scaled, logarithmic) = representations(value, bound)?;
                Ok(Forms {
                    value: Some(ordinary),
                    scaled,
                    logarithmic,
                    derivative: derivative
                        .map(|(estimate, bound)| {
                            ValueWithError::new(
                                estimate,
                                bound.absolute,
                                relative_error(estimate, bound.absolute),
                                bound.reliability,
                            )
                        })
                        .transpose()?,
                })
            }
            Self::Logarithmic {
                sign,
                log_magnitude,
                log_bound,
                bound,
                derivative,
            } => {
                let mut forms = logarithmic_forms(value, sign, log_magnitude, log_bound, bound)?;
                forms.derivative = derivative
                    .map(|(estimate, bound)| {
                        ValueWithError::new(
                            estimate,
                            bound.absolute,
                            relative_error(estimate, bound.absolute),
                            bound.reliability,
                        )
                    })
                    .transpose()?;
                Ok(forms)
            }
            Self::Calibrated { ulps } => match value {
                Some(value) => Self::Finite {
                    bound: Bound::envelope(value.abs(), ulps)?,
                    derivative: None,
                }
                .publish(Some(value)),
                None => Ok(Forms::NONE),
            },
            Self::Limit => Ok(Forms {
                value: value
                    .map(|value| {
                        ValueWithError::new(
                            value,
                            V::Real::ZERO,
                            None,
                            EstimateReliability::ModelBased,
                        )
                    })
                    .transpose()?,
                ..Forms::NONE
            }),
        }
    }
}

type Representations<V> = (
    ValueWithError<V, <V as NumericScalar>::Real>,
    Option<ValueWithError<ScaledValue<V>, <V as NumericScalar>::Real>>,
    Option<ValueWithError<LogValue<V, <V as NumericScalar>::Real>, <V as NumericScalar>::Real>>,
);

/// Ordinary, scaled and logarithmic forms of one finite value and its bound.
///
/// The scaled error is in scaled coordinates and the logarithmic error in
/// natural-log units. A form whose error has no meaningful value (a scaled
/// coordinate beyond range, or a log interval reaching zero) is omitted.
fn representations<V: NumericScalar>(
    value: V,
    bound: Bound<V::Real>,
) -> Result<Representations<V>, SciOxideError> {
    let relative = relative_error(value, bound.absolute);
    let ordinary = ValueWithError::new(value, bound.absolute, relative, bound.reliability)?;
    let scaled_value = ScaledValue::from_value(value)?;
    let scaled = scaled_error(bound.absolute, scaled_value.scale_exponent())
        .map(|error| ValueWithError::new(scaled_value, error, relative, bound.reliability))
        .transpose()?;
    // The log form reuses the magnitude logarithm LogValue already forms.
    let logarithmic = match relative {
        Some(relative) if relative >= V::Real::HALF => None,
        Some(relative) => {
            let log = nonzero_log(value)?;
            let magnitude = log
                .log_magnitude()
                .map_or(V::Real::ZERO, |magnitude| magnitude.abs());
            let error = log_error(relative, magnitude).ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "invalid logarithmic error",
                )
            })?;
            Some(ValueWithError::new(log, error, None, bound.reliability)?)
        }
        None if bound.absolute == V::Real::ZERO => Some(ValueWithError::new(
            LogValue::from_value(value)?,
            V::Real::ZERO,
            None,
            bound.reliability,
        )?),
        None => None,
    };
    Ok((ordinary, scaled, logarithmic))
}

/// The log form of a nonzero value: a real value takes its sign as the phase
/// and one logarithm, while a complex value forms its scaled phase.
fn nonzero_log<V: NumericScalar>(value: V) -> Result<LogValue<V, V::Real>, SciOxideError> {
    if V::IS_REAL {
        let real = value.real_component();
        LogValue::from_log(V::from_real(V::Real::ONE.copysign(real)), real.abs().ln())
    } else {
        LogValue::from_value(value)
    }
}

/// An absolute error in the scaled coordinates of a floating-point value at binary
/// exponent `exponent`: an exact power-of-two shift, rounded up by one quantum
/// when it lands below the normal range, and absent when it overflows.
#[inline]
pub(crate) fn scaled_error<R: RealScalar>(error: R, exponent: i64) -> Option<R> {
    // A zero value has exponent zero: the error is unchanged, and no
    // arithmetic touches a subnormal operand.
    if exponent == 0 {
        return Some(error);
    }
    let shifted = error.scalbn(i32::try_from(exponent).ok()?.checked_neg()?);
    // Only a right shift into the subnormal range rounds.
    let shifted = if exponent > 0 && shifted < R::MIN_POSITIVE && error > R::ZERO {
        shifted + R::MIN_SUBNORMAL
    } else {
        shifted
    };
    shifted.is_finite().then_some(shifted)
}

/// Forms of a signed logarithmic magnitude whose ordinary projection `value`
/// may be out of range.
fn logarithmic_forms<V: NumericScalar>(
    value: Option<V>,
    sign: V::Real,
    log_magnitude: V::Real,
    log_bound: Bound<V::Real>,
    bound: Option<Bound<V::Real>>,
) -> Result<Forms<V>, SciOxideError> {
    let logarithmic = ValueWithError::new(
        LogValue::from_log(V::from_real(sign), log_magnitude)?,
        log_bound.absolute,
        None,
        log_bound.reliability,
    )?;
    // Relative magnitude error implied by the logarithmic bound.
    let implied = log_bound.absolute.exp_m1().min(V::Real::MAX);
    let ordinary = value.map(|value| {
        let magnitude = value.abs();
        let bound = bound.unwrap_or_else(|| {
            Bound::new(magnitude * implied, log_bound.reliability).plus(Bound::rounded(magnitude))
        });
        (value, magnitude, bound)
    });
    let scaled = match ordinary {
        Some((value, magnitude, bound)) if magnitude >= V::Real::MIN_POSITIVE => {
            let scaled = ScaledValue::from_value(value)?;
            scaled
                .scale_absolute_error(bound.absolute)
                .ok()
                .map(|error| {
                    ValueWithError::new(
                        scaled,
                        error,
                        relative_error(value, bound.absolute),
                        bound.reliability,
                    )
                })
                .transpose()?
        }
        _ => {
            // The significand carries the logarithmic error and one rounding.
            let scaled =
                ScaledValue::from_components(Scaled::from_log(sign, log_magnitude)?, Scaled::ZERO);
            let relative = (implied + V::Real::EPSILON).min(V::Real::MAX);
            Some(scaled_estimate(scaled, relative, log_bound.reliability)?)
        }
    };
    Ok(Forms {
        value: ordinary
            .map(|(value, _, bound)| {
                ValueWithError::new(
                    value,
                    bound.absolute,
                    relative_error(value, bound.absolute),
                    bound.reliability,
                )
            })
            .transpose()?,
        scaled,
        logarithmic: Some(logarithmic),
        derivative: None,
    })
}

impl<V: NumericScalar> Publish<V> for Narrowed {
    /// Forms the `f64` forms, then narrows every form once.
    /// Narrowing serves real targets; their forms embed in `V`.
    fn publish(self, value: Option<V>) -> Result<Forms<V>, SciOxideError> {
        let wide = self.wide.publish(self.value)?;
        let cast = |value: f64| V::Real::try_from_f64_lossy(value);
        let error = outward::<V::Real>;
        let ordinary = match (value, wide.value) {
            (Some(narrowed), _)
                if !narrowed.has_finite_components()
                    && self.termination == SpecialFunctionTermination::Limit =>
            {
                Some(ValueWithError::new(
                    narrowed,
                    V::Real::ZERO,
                    None,
                    EstimateReliability::ModelBased,
                )?)
            }
            (Some(narrowed), Some(source)) => {
                let bound = error(
                    source.absolute_error()
                        + (narrowed.real_component().into_f64() - *source.value()).abs(),
                )?;
                Some(ValueWithError::new(
                    narrowed,
                    bound,
                    relative_error(narrowed, bound),
                    source.reliability(),
                )?)
            }
            _ => None,
        };
        let scaled = wide
            .scaled
            .map(|source| {
                let old_exponent = source.value().real().exponent();
                let real =
                    Scaled::from_parts(cast(source.value().real().significand())?, old_exponent)?;
                let rescale = if real.exponent() == old_exponent {
                    1.0
                } else {
                    0.5
                };
                let scaled = ScaledValue::<V>::from_components(real, Scaled::ZERO);
                let bound = error(
                    source.absolute_error() * rescale
                        + V::Real::EPSILON.into_f64() * scaled.scaled_magnitude().into_f64(),
                )?;
                ValueWithError::new(
                    scaled,
                    bound,
                    (scaled.scaled_magnitude() != V::Real::ZERO)
                        .then(|| (bound / scaled.scaled_magnitude()).min(V::Real::MAX)),
                    source.reliability(),
                )
            })
            .transpose()?;
        let logarithmic = wide
            .logarithmic
            .map(|source| {
                let magnitude = source.value().log_magnitude().map(cast).transpose()?;
                let phase = V::from_real(cast(source.value().scaled_phase().round()?)?);
                ValueWithError::new(
                    match magnitude {
                        Some(magnitude) => LogValue::from_log(phase, magnitude),
                        None => LogValue::from_value(phase),
                    }?,
                    error(
                        source.absolute_error()
                            + magnitude.map_or(0.0, |value| {
                                value.abs().into_f64() * V::Real::EPSILON.into_f64()
                            }),
                    )?,
                    None,
                    source.reliability(),
                )
            })
            .transpose()?;
        // A derivative beyond the destination range is unavailable.
        let derivative = wide
            .derivative
            .and_then(|source| Some((cast(*source.value()).ok()?, source)))
            .map(|(value, source)| {
                let bound =
                    error(source.absolute_error() + (value.into_f64() - source.value()).abs())?;
                ValueWithError::new(
                    V::from_real(value),
                    bound,
                    relative_error(V::from_real(value), bound),
                    source.reliability(),
                )
            })
            .transpose()?;
        Ok(Forms {
            value: ordinary,
            scaled,
            logarithmic,
            derivative,
        })
    }
}

/// A fixed-accuracy evaluation with its kernel envelope; documented infinite limits are exact.
/// Reported.
#[inline(always)]
pub(crate) fn enveloped<S: RealScalar>(
    value: S,
    ulps: Ulps,
    algorithm: AlgorithmId,
    iterations: usize,
    termination: SpecialFunctionTermination,
) -> Result<Evaluation<S>, SciOxideError> {
    if value.is_infinite() {
        return Ok(Evaluation::limit(value, algorithm));
    }
    if termination.is_success() {
        return Ok(Evaluation::calibrated(
            value,
            ulps,
            algorithm,
            iterations,
            termination,
        ));
    }
    Ok(Evaluation::finite(
        value,
        Bound::heuristic(Bound::envelope(value, ulps)?.absolute),
        None,
        algorithm,
        iterations,
        termination,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    const ALGORITHM: AlgorithmId = scioxide_core::algorithm_id!("gamma", "test", 1);

    fn converged(value: f64, bound: Bound<f64>) -> Evaluation<f64> {
        Evaluation::finite(
            value,
            bound,
            None,
            ALGORITHM,
            1,
            SpecialFunctionTermination::Converged,
        )
    }

    fn same_error<V: core::fmt::Debug>(
        plain: Result<V, SciOxideError>,
        detailed: Result<V, SciOxideError>,
    ) {
        assert_eq!(
            format!("{:?}", plain.unwrap_err()),
            format!("{:?}", detailed.unwrap_err())
        );
    }

    #[test]
    fn plain_and_published_projections_agree() {
        let evaluation = converged(-3.5, Bound::derived(1e-15));
        assert_eq!(
            evaluation.value().unwrap().to_bits(),
            evaluation.publish().unwrap().try_value().unwrap().to_bits()
        );
        let unavailable = Evaluation::<f64>::from_parts(
            None,
            ScalarRepresentation::Finite,
            SpecialFunctionTermination::IterationLimit,
            EvaluationCounts::from_iterations(7),
            ALGORITHM,
            None,
        );
        same_error(
            unavailable.value(),
            unavailable
                .publish()
                .and_then(SpecialFunctionResult::try_value),
        );
        let overflow = Evaluation::logarithmic(
            1.0,
            1000.0,
            Bound::derived(1e-13),
            Projection::Exp,
            ALGORITHM,
            1,
        );
        same_error(overflow.value(), overflow.publish().unwrap().try_value());
    }

    #[test]
    fn narrowing_respects_the_ieee_overflow_boundary_and_rounded_zeros() {
        let maximum = f64::from(f32::MAX);
        let quantum = maximum - f64::from(f32::from_bits(f32::MAX.to_bits() - 1));
        let source = converged(maximum + quantum * 0.25, Bound::derived(0.0));
        assert_eq!(source.narrow::<f32>().unwrap().value().unwrap(), f32::MAX);
        let source = converged(maximum + quantum * 0.5, Bound::derived(0.0));
        let narrowed = source.narrow::<f32>().unwrap();
        assert_eq!(narrowed.representation, ScalarRepresentation::Overflow);
        let published = narrowed.publish().unwrap();
        assert!(published.scaled().is_some() && published.logarithmic().is_some());
        for wide in [1e-100_f64, -1e-100] {
            let narrowed = converged(wide, Bound::rounded(wide))
                .narrow::<f32>()
                .unwrap();
            assert_eq!(narrowed.representation, ScalarRepresentation::Finite);
            let zero = narrowed.value().unwrap();
            assert_eq!(zero.to_bits(), 0.0_f32.copysign(wide as f32).to_bits());
            let published = narrowed.publish().unwrap();
            assert_eq!(published.try_value().unwrap().to_bits(), zero.to_bits());
            assert!(f64::from(published.value().unwrap().absolute_error()) >= wide.abs());
        }
    }

    #[test]
    fn logarithmic_values_below_the_range_publish_signed_zeros() {
        for sign in [1.0_f64, -1.0] {
            let evaluation = Evaluation::logarithmic(
                sign,
                -800.0,
                Bound::derived(1e-13),
                Projection::Exp,
                ALGORITHM,
                1,
            );
            let zero: f64 = evaluation.value().unwrap();
            assert_eq!(zero.to_bits(), 0.0_f64.copysign(sign).to_bits());
            let published = evaluation.publish().unwrap();
            assert_eq!(
                published.scalar_representation(),
                ScalarRepresentation::Finite
            );
            let value = published.value().unwrap();
            assert_eq!(value.value().to_bits(), zero.to_bits());
            assert!(value.absolute_error() >= f64::from_bits(1));
            assert!(published.logarithmic().is_some() && published.scaled().is_some());
        }
    }

    #[test]
    fn finite_results_expose_all_representations() {
        let result = Evaluation::exact(-12.0_f64, Some(3.0), ALGORITHM, 1)
            .publish()
            .unwrap();
        assert_eq!(result.try_value().unwrap(), -12.0);
        assert!(result.scaled().is_some());
        assert!(result.logarithmic().is_some());
        assert!(result.derivative().is_some());
    }

    #[test]
    fn bounds_scale_with_subnormal_values() {
        let tiny = f64::from_bits(3);
        assert_eq!(ulp_bound(tiny, 4.0), 4.0 * f64::from_bits(1));
        assert_eq!(ulp_bound(0.0_f64, 0.5), 0.5 * f64::from_bits(1));
        assert_eq!(ulp_bound(0.0_f64, 64.0), 64.0 * f64::from_bits(1));
        assert_eq!(ulp_bound(0.0_f32, 4.0), 4.0 * f32::from_bits(1));
        // A normal value whose unit is subnormal keeps the larger of both.
        assert_eq!(
            ulp_bound(f64::MIN_POSITIVE * 8.0, 2.0),
            16.0 * f64::from_bits(1)
        );
        assert_eq!(ulp_bound(2.0_f64, 1.0), 2.0 * f64::EPSILON);
        // Scaled coordinates shift exactly, round a subnormal right shift up
        // and are absent on overflow.
        assert_eq!(scaled_error(f64::from_bits(1), 0), Some(f64::from_bits(1)));
        assert_eq!(scaled_error(1.0_f64, -5), Some(32.0));
        assert_eq!(scaled_error(f64::from_bits(1), 10), Some(f64::from_bits(1)));
        assert_eq!(scaled_error(f64::MAX, -10), None);
        let result = converged(tiny, Bound::envelope(tiny, 4.0).unwrap())
            .publish()
            .unwrap();
        let value = result.value().unwrap();
        assert_eq!(value.absolute_error(), 4.0 * f64::from_bits(1));
        assert_eq!(
            value.reliability(),
            EstimateReliability::StatisticallyCalibrated
        );
        assert!(result.scaled().unwrap().absolute_error() <= 4.0);
        // The relative interval reaches zero, so no log estimate is published.
        assert!(result.logarithmic().is_none());
        assert_eq!(result.try_value().unwrap(), tiny);
    }

    #[test]
    fn logarithmic_errors_are_in_log_units() {
        assert_eq!(log_error(0.5_f64, 1.0), None);
        let error = log_error(f64::EPSILON, 700.0).unwrap();
        assert!(error > 700.0 * f64::EPSILON && error < 702.0 * f64::EPSILON);
        let result = converged(1e300, Bound::envelope(1e300, 4.0).unwrap())
            .publish()
            .unwrap();
        let logarithm = result.logarithmic().unwrap();
        assert!(logarithm.absolute_error() < 1e-12);
        assert!(result.value().unwrap().absolute_error() > 1e280);
    }

    #[test]
    fn destination_bounds_round_outward_and_retain_relative_errors() {
        for bound in [
            0.0,
            1.0e-100,
            f64::from(f32::from_bits(1)) * 1.25,
            f64::from(f32::EPSILON) * (1.0 + f64::EPSILON),
            1.0e30,
        ] {
            let narrowed = outward::<f32>(bound).unwrap();
            assert!(f64::from(narrowed) >= bound);
            if bound > 0.0 {
                assert!(narrowed > 0.0);
            }
        }
        let source = converged(
            1.0 + f64::from(f32::EPSILON) * 0.25,
            Bound::derived(1.0e-14),
        );
        let narrow = source.narrow::<f32>().unwrap().publish().unwrap();
        let value = narrow.value().unwrap();
        let required = 1.0e-14 + (f64::from(*value.value()) - source.ordinary().unwrap()).abs();
        assert!(f64::from(value.absolute_error()) >= required);
        assert!(value.relative_error().is_some());
        assert!(narrow.scaled().unwrap().relative_error().is_some());
    }

    #[test]
    fn bounds_combine_with_the_weaker_reliability() {
        let model = Bound::<f64>::derived(1.0);
        let calibrated = Bound::envelope(1.0, 2.0).unwrap();
        let heuristic = Bound::heuristic(1.0);
        assert_eq!(
            model.plus(calibrated).reliability,
            EstimateReliability::StatisticallyCalibrated
        );
        assert_eq!(
            calibrated.plus(heuristic).reliability,
            EstimateReliability::Heuristic
        );
        assert_eq!(model.plus(model).absolute, 2.0);
        assert_eq!(Bound::derived(f64::INFINITY).absolute, f64::MAX);
    }

    #[test]
    fn logarithmic_projection_keeps_subnormal_values() {
        let result = Evaluation::logarithmic(
            1.0,
            -742.0_f64,
            Bound::derived(1024.0 * f64::EPSILON),
            Projection::Exp,
            ALGORITHM,
            1,
        )
        .publish()
        .unwrap();
        let value = result.value().unwrap();
        assert!(*value.value() > 0.0 && *value.value() < f64::MIN_POSITIVE);
        assert!(value.absolute_error() >= 0.5 * f64::from_bits(1));
        assert!(result.scaled().is_some());
    }
}
