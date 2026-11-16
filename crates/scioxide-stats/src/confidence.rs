//! Confidence probabilities, critical values and confidence intervals for a mean.

use scioxide_core::Scaled;
use scioxide_core::checked_float_product;
use scioxide_core::{Interval, NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError};
use scioxide_special::BetaOptions;

/// Nominal probability of a two-sided interval, in the open interval `(0, 1)`.
///
/// The result type that uses it gives its meaning: confidence level for a
/// confidence interval, or posterior mass for a credible interval. The default
/// is 0.95.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[must_use]
pub struct ConfidenceProbability(f64);

impl ConfidenceProbability {
    /// Checks and wraps a probability.
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::NonFinite`] if `probability` is NaN or infinite.
    /// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if `probability` is not in `(0, 1)`.
    pub fn new(probability: f64) -> Result<Self, SciOxideError> {
        if !probability.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "confidence probability must be finite",
            ));
        }
        if !(0.0..1.0).contains(&probability) || probability == 0.0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("probability"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "confidence probability must be strictly inside (0, 1)",
            ));
        }
        Ok(Self(probability))
    }

    /// Returns the probability as an `f64`.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Returns the two-sided standard normal critical value `z = Φ⁻¹((1 + p)/2)`.
    ///
    /// Here `p` is the probability and `Φ` the standard normal distribution
    /// function. The value is computed as `z = √2 erf⁻¹(p)`, or as
    /// `√2 erfc⁻¹(1 − p)` for `p > 1/2`, so `(1 + p)/2` is never formed.
    ///
    /// # Errors
    ///
    /// Returns a conversion or numerical error if `p` or `z` cannot be
    /// represented in `S`.
    pub fn normal_critical_value<S: RealScalar>(self) -> Result<S, SciOxideError> {
        let sqrt_two = S::try_from_f64_lossy(core::f64::consts::SQRT_2)?;
        let inverse = if self.0 <= 0.5 {
            let central_probability = S::try_from_f64_lossy(self.0)?;
            scioxide_special::erfinv(central_probability)?
        } else {
            let complement = S::try_from_f64_lossy(1.0 - self.0)?;
            scioxide_special::erfcinv(complement)?
        };
        let critical = checked_float_product(sqrt_two, inverse)?;
        if !critical.is_finite() || critical < S::ZERO {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "inverse error function did not produce a finite non-negative critical value",
            ));
        }
        Ok(critical)
    }
}

impl Default for ConfidenceProbability {
    fn default() -> Self {
        Self(0.95)
    }
}

/// Distribution of the critical value for a mean interval, and the meaning of its standard error.
///
/// The interval is `x̄ ± c·se`, where `x̄` is the estimate, `se` its standard
/// error and `c` the critical value of the selected distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeanIntervalMethod {
    /// Student's t with `ν` degrees of freedom, for an estimated variance under
    /// a normal model. Requires finite `ν > 0`.
    #[default]
    StudentT,
    /// Standard normal, for a known standard error.
    NormalKnownVariance,
    /// Standard normal, as a large-sample approximation with an estimated standard error.
    NormalAsymptotic,
}

/// Sampling assumptions under which a mean interval has its nominal coverage.
///
/// The assumptions are reported, not tested against the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanIntervalAssumptions {
    /// Independent normal observations with a common unknown variance.
    IndependentNormal,
    /// A normal estimator whose variance you supply.
    KnownNormalVariance,
    /// The central limit theorem with a consistent standard error.
    AsymptoticNormal,
    /// Frequency weights count independent normal observations, not correlated duplicates.
    FrequencyNormal,
    /// Independent normal observations with variance proportional to the inverse weight.
    AnalyticNormal,
    /// Fixed reliability weights with a Student's t approximation based on the
    /// Kish effective sample size. Coverage is approximate.
    EffectiveSampleApproximation,
}

/// Confidence interval for a mean, with the method and assumptions behind it.
///
/// The bounds are a statistical interval with nominal coverage. They are not a
/// guaranteed numerical enclosure.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MeanConfidenceInterval<S: RealScalar> {
    bounds: Interval<S>,
    confidence: ConfidenceProbability,
    method: MeanIntervalMethod,
    degrees_of_freedom: Option<f64>,
    assumptions: MeanIntervalAssumptions,
    estimate: S,
    standard_error: S,
    weight_semantics: Option<crate::WeightSemantics>,
}

impl<S: RealScalar> MeanConfidenceInterval<S> {
    /// Returns the bounds `x̄ ± c·se`, rounded to `S` but not outward.
    pub const fn bounds(self) -> Interval<S> {
        self.bounds
    }
    /// Returns the nominal confidence level.
    pub const fn confidence(self) -> ConfidenceProbability {
        self.confidence
    }
    /// Returns the method that selected the critical value.
    pub const fn method(self) -> MeanIntervalMethod {
        self.method
    }
    /// Returns the sampling assumptions under which the coverage holds.
    pub const fn assumptions(self) -> MeanIntervalAssumptions {
        self.assumptions
    }
    /// Returns the point estimate `x̄`.
    pub const fn estimate(self) -> S {
        self.estimate
    }
    /// Returns the standard error `se`, rounded to `S`.
    ///
    /// An interval constructed with an extended-range error retains that error
    /// through the endpoint calculation even when this accessor rounds to zero.
    pub const fn standard_error(self) -> S {
        self.standard_error
    }
    /// Returns the weight model of the weighted snapshot that produced the interval.
    ///
    /// It is reported for every method, including the asymptotic normal one.
    /// Returns `None` for unweighted snapshots and for intervals built with
    /// [`MeanIntervalPlan::interval`] or [`MeanIntervalPlan::interval_scaled`].
    pub const fn weight_semantics(self) -> Option<crate::WeightSemantics> {
        self.weight_semantics
    }

    pub(crate) fn with_weight_semantics(mut self, semantics: crate::WeightSemantics) -> Self {
        self.weight_semantics = Some(semantics);
        self
    }
    /// Returns the degrees of freedom `ν` of Student's t, or `None` for a normal critical value.
    pub const fn degrees_of_freedom(self) -> Option<f64> {
        self.degrees_of_freedom
    }
    /// Returns whether the rounded endpoints coincide.
    pub fn is_degenerate(self) -> bool {
        self.bounds.lower() == self.bounds.upper()
    }
}

impl ConfidenceProbability {
    /// Computes the result of [`Self::student_t_critical_value_with_options`] with the [one-shot defaults](crate#one-shot-forms).
    ///
    /// Uses the same numerical method, cost and accuracy as the configured form.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::student_t_critical_value_with_options`] for the fixed default settings.
    pub fn student_t_critical_value(self, degrees: f64) -> Result<f64, SciOxideError> {
        self.student_t_critical_value_with_options(degrees, Default::default())
    }
    /// Returns the two-sided Student's t critical value `t = F⁻¹_ν((1 + p)/2)` as an `f64`.
    ///
    /// Here `p` is the probability and `F_ν` the distribution function of
    /// Student's t with `ν = degrees` degrees of freedom.
    ///
    /// For `ν < 4096`, `t` comes from the inverse regularized incomplete beta
    /// function through `P(|T| ≤ t) = I_x(1/2, ν/2)`, `x = t²/(ν + t²)`.
    /// `options` sets its iteration limit and tolerance. The central and
    /// complementary forms are inverted so that `(1 + p)/2` is never formed.
    /// For `ν ≥ 4096`, `t` is the expansion in powers of `1/ν` about the normal
    /// critical value, through `ν⁻⁴`. It needs an iteration limit of at least 4,
    /// and its relative truncation error is below `10⁻¹⁷`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `degrees` is not finite and positive.
    /// - `ConvergenceFailure` if `ν ≥ 4096` and the iteration limit in
    ///   `options` is below 4, or if the inverse incomplete beta function does
    ///   not converge.
    /// - `NumericalInstability` if a component of the critical value underflows.
    ///
    /// # References
    ///
    /// M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*,
    /// 1964, eq. 26.7.5.
    pub fn student_t_critical_value_with_options(
        self,
        degrees: f64,
        options: BetaOptions,
    ) -> Result<f64, SciOxideError> {
        if !degrees.is_finite() || degrees <= 0.0 {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("degrees"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Student-t degrees must be finite and positive",
            ));
        }
        if degrees >= 4096.0 {
            // Abramowitz–Stegun 26.7.5 through nu^-4. At this boundary the
            // binary64 confidence domain has z < 9, so z²/nu < 0.02.
            // This evaluates the Student-t quantile, not the normal interval.
            if options.iteration_budget().get() < 4 {
                return Err(SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::IterationLimit,
                    scioxide_core::ConvergenceDiagnostics::new(),
                    "four correction terms require four work units",
                ));
            }
            let z = self.normal_critical_value::<f64>()?;
            let q = z * z;
            let r = 1.0 / degrees;
            let a = (q + 1.0) / 4.0;
            let b = ((5.0 * q + 16.0) * q + 3.0) / 96.0;
            let c = (((3.0 * q + 19.0) * q + 17.0) * q - 15.0) / 384.0;
            let d = ((((79.0 * q + 776.0) * q + 1482.0) * q - 1920.0) * q - 945.0) / 92160.0;
            return Ok(z * (1.0 + r * (a + r * (b + r * (c + r * d)))));
        }
        let (numerator, denominator) = if self.0 <= 0.5 {
            let x = scioxide_special::beta_inc_inv_log_with_options(
                0.5,
                degrees * 0.5,
                self.0.ln(),
                options,
            )?
            .try_value()?;
            (x, 1.0 - x)
        } else if degrees >= 64.0 {
            // For large degrees, nu/(nu+t²) crowds one. Invert the swapped
            // complementary beta tail to retain the small t²/(nu+t²) coordinate.
            let x = scioxide_special::beta_inc_q_inv_log_with_options(
                0.5,
                degrees * 0.5,
                (-self.0).ln_1p(),
                options,
            )?
            .try_value()?;
            (x, 1.0 - x)
        } else {
            let x = scioxide_special::beta_inc_inv_log_with_options(
                degrees * 0.5,
                0.5,
                (-self.0).ln_1p(),
                options,
            )?
            .try_value()?;
            (1.0 - x, x)
        };
        if numerator <= 0.0 || denominator <= 0.0 {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Underflow,
                "beta inverse lost a nonzero critical-value component",
            ));
        }
        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(degrees))?
            .mul(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(numerator),
            )?)?
            .div(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(denominator),
            )?)?
            .sqrt()?
            .round_to()
    }
}

fn validate_degrees(
    method: MeanIntervalMethod,
    degrees_of_freedom: Option<f64>,
) -> Result<(), SciOxideError> {
    match (method, degrees_of_freedom) {
        (MeanIntervalMethod::StudentT, Some(d)) if d.is_finite() && d > 0.0 => {}
        (MeanIntervalMethod::NormalKnownVariance | MeanIntervalMethod::NormalAsymptotic, None) => {}
        _ => {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("degrees_of_freedom"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Student-t requires finite positive degrees; normal intervals require no degrees",
            ));
        }
    }
    Ok(())
}

/// Two-sided critical value for mean intervals, computed once and reused.
///
/// [`MeanIntervalPlan::new`] evaluates the quantile. [`MeanIntervalPlan::interval`]
/// then builds `x̄ ± c·se` for any estimate, standard error and scalar type
/// without evaluating it again.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ConfidenceProbability, MeanIntervalMethod, MeanIntervalPlan};
/// let plan = MeanIntervalPlan::new(
///     ConfidenceProbability::default(),
///     MeanIntervalMethod::StudentT,
///     Some(9.0),
///     Default::default(),
/// )?;
/// let interval = plan.interval(2.0_f64, 0.1)?;
/// assert!(interval.bounds().lower() < 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MeanIntervalPlan {
    confidence: ConfidenceProbability,
    method: MeanIntervalMethod,
    degrees: Option<f64>,
    critical: f64,
}

impl MeanIntervalPlan {
    /// Computes the critical value for `method` at probability `confidence`.
    ///
    /// [`MeanIntervalMethod::StudentT`] requires `degrees = Some(ν)` with finite
    /// `ν > 0`. The normal methods require `None`. The quantile is evaluated
    /// during construction, even if a later interval has a zero standard error.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `degrees` does not match `method`.
    /// - The errors of [`ConfidenceProbability::student_t_critical_value_with_options`] or
    ///   [`ConfidenceProbability::normal_critical_value`].
    pub fn new(
        confidence: ConfidenceProbability,
        method: MeanIntervalMethod,
        degrees: Option<f64>,
        options: BetaOptions,
    ) -> Result<Self, SciOxideError> {
        validate_degrees(method, degrees)?;
        let critical = match degrees {
            Some(degrees) => confidence.student_t_critical_value_with_options(degrees, options)?,
            None => confidence.normal_critical_value::<f64>()?,
        };
        Ok(Self {
            confidence,
            method,
            degrees,
            critical,
        })
    }

    /// Computes the interval `x̄ ± c·se` with the prepared critical value `c`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `estimate` or `standard_error` is NaN or infinite.
    /// - `InvalidInput` if `standard_error` is negative.
    /// - A numerical error if a bound overflows `S`.
    pub fn interval<S: RealScalar>(
        self,
        estimate: S,
        standard_error: S,
    ) -> Result<MeanConfidenceInterval<S>, SciOxideError> {
        self.interval_scaled(estimate, Scaled::from_value(standard_error)?)
    }

    /// Computes `estimate ± c·standard_error` without rounding the error to `S`
    /// before multiplication by the prepared critical value `c`.
    ///
    /// Uses `O(1)` work and no allocation. Extended-range arithmetic preserves
    /// subnormal margins; endpoints are rounded to `S`, not outward. Accuracy
    /// also depends on the quantile used by [`Self::new`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `estimate` is NaN or infinite.
    /// - `InvalidInput` if `standard_error` is negative.
    /// - A numerical error if the reported standard error or a bound overflows
    ///   `S`, or extended-range arithmetic fails.
    ///
    /// ```
    /// use scioxide_core::Scaled;
    /// use scioxide_stats::{ConfidenceProbability, MeanIntervalMethod, MeanIntervalPlan};
    /// let plan = MeanIntervalPlan::new(
    ///     ConfidenceProbability::default(),
    ///     MeanIntervalMethod::StudentT,
    ///     Some(3.0),
    ///     Default::default(),
    /// )?;
    /// let error = Scaled::from_parts(1.0_f64, -1076)?;
    /// let interval = plan.interval_scaled(0.0, error)?;
    /// let upper = interval.bounds().upper() / f64::from_bits(1);
    /// assert!((upper - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. Casella and R. L. Berger, *Statistical Inference*, 2nd ed.,
    /// Duxbury, 2002, Section 9.2.
    pub fn interval_scaled<S: RealScalar>(
        self,
        estimate: S,
        standard_error: Scaled<S>,
    ) -> Result<MeanConfidenceInterval<S>, SciOxideError> {
        if !estimate.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "estimate must be finite",
            ));
        }
        if standard_error.is_sign_negative() && !standard_error.is_zero() {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("standard_error"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "standard error must be nonnegative",
            ));
        }
        interval_from_critical(
            estimate,
            standard_error,
            self.confidence,
            self.method,
            self.degrees,
            self.critical,
        )
    }
}

fn interval_from_critical<S: RealScalar>(
    estimate: S,
    scaled_error: Scaled<S>,
    confidence: ConfidenceProbability,
    method: MeanIntervalMethod,
    degrees_of_freedom: Option<f64>,
    critical: f64,
) -> Result<MeanConfidenceInterval<S>, SciOxideError> {
    let standard_error = scaled_error.round()?;
    let assumptions = match method {
        MeanIntervalMethod::StudentT => MeanIntervalAssumptions::IndependentNormal,
        MeanIntervalMethod::NormalKnownVariance => MeanIntervalAssumptions::KnownNormalVariance,
        MeanIntervalMethod::NormalAsymptotic => MeanIntervalAssumptions::AsymptoticNormal,
    };
    let native = || {
        if Scaled::from_value(standard_error).ok()? != scaled_error {
            return None;
        }
        let center = estimate.into_f64();
        let margin = if standard_error == S::ZERO {
            0.0
        } else {
            standard_error.into_f64() * critical
        };
        if !margin.is_finite() || standard_error != S::ZERO && margin < f64::MIN_POSITIVE {
            return None;
        }
        let lower = center - margin;
        let upper = center + margin;
        if [lower, upper]
            .into_iter()
            .any(|x| !x.is_finite() || x != 0.0 && x.abs() < f64::MIN_POSITIVE)
        {
            return None;
        }
        let lo = S::try_from_f64_lossy(lower).ok()?;
        let hi = S::try_from_f64_lossy(upper).ok()?;
        if lo == S::ZERO && lower != 0.0 || hi == S::ZERO && upper != 0.0 {
            return None;
        }
        Interval::new(lo, hi).ok()
    };
    let bounds = match native() {
        Some(bounds) => bounds,
        None => {
            let margin = if scaled_error.is_zero() {
                Scaled::<f64>::ZERO
            } else {
                Scaled::<f64>::from_parts(
                    scaled_error.significand().into_f64(),
                    scaled_error.exponent(),
                )?
                .mul(Scaled::<f64>::from_value(critical)?)?
            };
            let center = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(estimate))?;
            Interval::new(
                center.sub(margin)?.round_to()?,
                center.add(margin)?.round_to()?,
            )
        }?,
    };
    Ok(MeanConfidenceInterval {
        bounds,
        confidence,
        method,
        degrees_of_freedom,
        assumptions,
        estimate,
        standard_error,
        weight_semantics: None,
    })
}

pub(crate) fn estimated_interval<S: RealScalar>(
    estimate: S,
    standard_error: S,
    confidence: ConfidenceProbability,
    method: MeanIntervalMethod,
    degrees: f64,
    assumptions: MeanIntervalAssumptions,
    options: BetaOptions,
) -> Result<MeanConfidenceInterval<S>, SciOxideError> {
    if method == MeanIntervalMethod::NormalKnownVariance {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("method"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "estimated variance cannot use known-variance inference",
        ));
    }
    let mut result = crate::MeanIntervalPlan::new(
        confidence,
        method,
        (method == MeanIntervalMethod::StudentT).then_some(degrees),
        options,
    )
    .and_then(|plan| plan.interval(estimate, standard_error))?;
    if method == MeanIntervalMethod::StudentT {
        result.assumptions = assumptions;
    }
    Ok(result)
}
