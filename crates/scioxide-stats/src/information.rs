//! Shannon entropy, Kullback–Leibler divergence and spacing estimators of differential entropy.
use crate::RawDescriptiveWorkspace;
use crate::arithmetic::{Compensated, RangeSum, unit_deviance};
use crate::error_helpers::{invalid, number, sample};
use crate::ranking::sort_finite;
use scioxide_core::Scaled;
use scioxide_core::{Argument, RealScalar, SciOxideError};

/// Treatment of the masses passed to [`entropy_with_options`]. In both cases the masses are
/// divided by their sum before use.
#[derive(Clone, Copy, Debug)]
pub enum MassNormalization<S: RealScalar> {
    /// Accepts any nonnegative masses with a positive sum.
    Normalize,
    /// Requires `|Σ pᵢ − 1| ≤ t` for the given tolerance `t ≥ 0`.
    RequireUnit(S),
}
/// Options for [`entropy_with_options`]: the logarithm base and the mass normalization.
///
/// The default is base `e` (nats) with [`MassNormalization::Normalize`].
#[derive(Clone, Copy, Debug)]
pub struct EntropyOptions<S: RealScalar> {
    /// Logarithm base, finite and greater than one; `e` gives nats and 2 gives bits.
    pub(crate) base: S,
    /// Normalization of the masses.
    pub(crate) normalization: MassNormalization<S>,
}
impl<S: RealScalar> Default for EntropyOptions<S> {
    fn default() -> Self {
        Self {
            base: S::ONE.exp(),
            normalization: MassNormalization::Normalize,
        }
    }
}
/// Entropy or divergence on the extended real line.
///
/// The infinite values are exact mathematical results, not overflow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InformationValue<S: RealScalar> {
    /// A finite value.
    Finite(S),
    /// `+∞`: a Kullback–Leibler divergence where some `pᵢ > 0` has `qᵢ = 0`.
    PositiveInfinity,
    /// `−∞`: a differential-entropy estimate with a zero spacing, caused by tied observations.
    NegativeInfinity,
}
pub(crate) fn log_base<S: RealScalar>(base: S) -> Result<f64, SciOxideError> {
    if !base.is_finite() || base <= S::ONE {
        return Err(invalid(
            scioxide_core::argument!("options.base"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "information base must be finite and greater than one",
        ));
    }
    Ok(base.try_to_f64_exact()?.ln())
}

enum InformationSum {
    Finite(Scaled<f64>),
    SupportMismatch,
}

fn validate_mass<S: RealScalar>(value: S, argument: Argument) -> Result<(), SciOxideError> {
    if !value.is_finite() || value < S::ZERO {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "probability masses must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn scaled_total_mass<S: RealScalar>(
    values: &[S],
    argument: Argument,
) -> Result<RangeSum, SciOxideError> {
    let mut sum = scioxide_core::ScaledAccumulator::ZERO;
    for &value in values {
        validate_mass(value, argument)?;
        sum.add_value(value.into_f64())?;
    }
    Ok(RangeSum::Extended(sum))
}

// Nonzero source exponents in [-256,256] and at most usize::MAX terms
// (at most 64 count bits) bound totals below 2^321 and normalized masses
// above 2^-577. Probability ratios and their weighted logarithms remain
// normal binary64. All f32 masses, including subnormals, fit this frame.
fn admitted_total_mass<S: RealScalar>(
    values: &[S],
    argument: Argument,
) -> Result<RangeSum, SciOxideError> {
    let mut sum = scioxide_core::CompensatedSum::ZERO;
    for &value in values {
        validate_mass(value, argument)?;
        let value = value.into_f64();
        if !crate::arithmetic::native(value, 256) {
            return scaled_total_mass(values, argument);
        }
        sum.add(value)?;
    }
    Ok(RangeSum::Native(sum))
}

fn total_mass<S: RealScalar>(
    values: &[S],
    normalization: MassNormalization<S>,
    argument: Argument,
) -> Result<RangeSum, SciOxideError> {
    if let MassNormalization::RequireUnit(t) = normalization
        && (!t.is_finite() || t < S::ZERO)
    {
        return Err(invalid(
            scioxide_core::argument!("options.normalization"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "mass tolerance must be finite and nonnegative",
        ));
    }
    let total = admitted_total_mass(values, argument)?;
    let sum = total.total()?;
    if sum.is_zero() {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "probability mass sum must be positive",
        ));
    }
    if let MassNormalization::RequireUnit(t) = normalization
        && sum
            .sub(Scaled::<f64>::ONE)?
            .abs_cmp(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(t),
            )?)
            .is_gt()
    {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "probabilities do not sum to one",
        ));
    }
    Ok(total)
}

fn native_entropy<S: RealScalar>(
    p: &[S],
    total: scioxide_core::CompensatedSum<f64>,
) -> Result<f64, SciOxideError> {
    let [hi, lo] = total.parts();
    let total = total.total()?;
    let mut sum = Compensated::<f64>::zero();
    for &value in p {
        if value != S::ZERO {
            let value = value.into_f64();
            let mass = value / total;
            // A dominant probability may round to one. Sterbenz subtraction
            // retains the other masses in the two-part total before log1p.
            let log = if value > 0.5 * total {
                (((hi - value) + lo) / value).ln_1p()
            } else {
                -mass.ln()
            };
            sum = sum.added(mass * log)?;
        }
    }
    sum.total()
}

fn native_relative_entropy<S: RealScalar>(
    p: &[S],
    q: &[S],
    tp: f64,
    tq: f64,
) -> Result<InformationSum, SciOxideError> {
    let mut sum = Compensated::<f64>::zero();
    for (&p, &q) in p.iter().zip(q) {
        if p != S::ZERO && q == S::ZERO {
            return Ok(InformationSum::SupportMismatch);
        }
        // Generalized KL sums p*log(p/q)-p+q, whose linear terms cancel
        // analytically for unit masses. Each term is nonnegative.
        let (mass, reference) = (p.into_f64() / tp, q.into_f64() / tq);
        sum = sum.added(unit_deviance(mass, reference, mass - reference)?)?;
    }
    Ok(InformationSum::Finite(Scaled::from_value(sum.total()?)?))
}

fn scaled_entropy<S: RealScalar>(
    p: &[S],
    q: Option<(&[S], Scaled<f64>)>,
    total: RangeSum,
) -> Result<InformationSum, SciOxideError> {
    let tp = total.total()?;
    let [hi, lo] = total.extended()?.parts();
    let mut sum = Compensated::<Scaled<f64>>::zero();
    for (i, &p) in p.iter().enumerate() {
        let value = Scaled::<f64>::from_value(p.into_f64())?;
        let mass = value.div(tp)?;
        let term = if let Some((q, tq)) = q {
            if p != S::ZERO && q[i] == S::ZERO {
                return Ok(InformationSum::SupportMismatch);
            }
            let reference = Scaled::<f64>::from_value(q[i].into_f64())?.div(tq)?;
            unit_deviance(mass, reference, mass.sub(reference)?)?
        } else if p == S::ZERO {
            continue;
        } else {
            let log = if mass.abs_cmp(Scaled::HALF).is_gt() {
                // The other masses are below one half of the dominant one.
                crate::arithmetic::scaled_ln_1p(hi.sub(value)?.add(lo)?.div(value)?)?
            } else {
                Scaled::from_value(-mass.ln_abs()?)?
            };
            mass.mul(log)?
        };
        sum = sum.added(term)?;
    }
    Ok(InformationSum::Finite(sum.total()?))
}

/// Computes the result of [`entropy_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`entropy_with_options`] for the fixed default settings.
pub fn entropy<S: RealScalar>(p: &[S]) -> Result<InformationValue<S>, SciOxideError> {
    entropy_with_options(p, None, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options", "q"],
            scioxide_core::argument!("p"),
        )
    })
}
/// Computes the Shannon entropy of `p`, or the Kullback–Leibler divergence of
/// `p` from `q` when `q` is given.
///
/// With the masses normalized to `pᵢ ← pᵢ / Σ pⱼ` (and likewise `qᵢ`) and `b` the
/// logarithm base,
///
/// ```text
/// H(p)    = −Σ pᵢ log_b pᵢ
/// D(p‖q)  =  Σ pᵢ log_b (pᵢ / qᵢ)
/// ```
///
/// Terms with `pᵢ = 0` are zero, also where `qᵢ = 0`. The divergence is
/// [`InformationValue::PositiveInfinity`] if some `pᵢ > 0` has `qᵢ = 0`. The cost
/// is O(n). The sums are compensated, and the mass sums may exceed the range of
/// `S`.
///
/// # Errors
///
/// - `InvalidInput` if the base is not finite or not greater than 1.
/// - `InvalidInput` if `q` and `p` have different lengths.
/// - `InvalidInput` if a mass is negative, NaN or infinite.
/// - `InvalidInput` if the masses of `p` or `q` sum to zero, including empty input.
/// - `InvalidInput` with [`MassNormalization::RequireUnit`] if the tolerance is
///   negative or not finite, or a sum differs from 1 by more than the tolerance.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{EntropyOptions, InformationValue, MassNormalization, entropy_with_options};
///
/// // Two equally likely outcomes carry 1 bit; the zero mass adds nothing.
/// let options = EntropyOptions::new()
///     .with_base(2.0_f64)
///     .with_normalization(MassNormalization::Normalize);
/// let InformationValue::Finite(bits) = entropy_with_options(&[1.0, 1.0, 0.0], None, options)? else {
///     unreachable!()
/// };
/// assert!((bits - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed.,
/// Wiley, 2006, §2.1 and §2.3.
pub fn entropy_with_options<S: RealScalar>(
    p: &[S],
    q: Option<&[S]>,
    options: EntropyOptions<S>,
) -> Result<InformationValue<S>, SciOxideError> {
    let base = log_base(options.base)?;
    if q.is_some_and(|q| q.len() != p.len()) {
        return Err(invalid(
            scioxide_core::argument!("q"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "entropy support shape mismatch",
        ));
    }
    let tp = total_mass(p, options.normalization, scioxide_core::argument!("p"))?;
    let tq = q
        .map(|q| total_mass(q, options.normalization, scioxide_core::argument!("q")))
        .transpose()?;
    let result = match (tp, tq, q) {
        (RangeSum::Native(tp), None, None) => {
            InformationSum::Finite(Scaled::from_value(native_entropy(p, tp)?)?)
        }
        (RangeSum::Native(tp), Some(RangeSum::Native(tq)), Some(q)) => {
            native_relative_entropy(p, q, tp.total()?, tq.total()?)?
        }
        _ => scaled_entropy(
            p,
            q.zip(tq)
                .map(|(q, total)| total.total().map(|total| (q, total)))
                .transpose()?,
            tp,
        )?,
    };
    let sum = match result {
        InformationSum::Finite(value) => value,
        InformationSum::SupportMismatch => return Ok(InformationValue::PositiveInfinity),
    };
    let value: S = sum.div(Scaled::<f64>::from_value(base)?)?.round_to()?;
    // Only rounding-scale negative KL drift may be clamped.
    let value = if q.is_some() && value < S::ZERO {
        if value.abs() > S::EPSILON * number(32.)? {
            return Err(invalid(
                scioxide_core::argument!("q"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "negative KL divergence beyond rounding error",
            ));
        }
        S::ZERO
    } else {
        value
    };
    Ok(InformationValue::Finite(value))
}
/// Estimator of the differential entropy `h(f) = −∫ f(x) log f(x) dx` from a
/// sample, based on `m`-spacings of the order statistics `x₍₁₎ ≤ … ≤ x₍ₙ₎`.
///
/// In the formulas, `x₍ⱼ₎ = x₍₁₎` for `j < 1` and `x₍ⱼ₎ = x₍ₙ₎` for `j > n`. They
/// give nats; for base `b` the result is divided by `ln b`.
///
/// # References
///
/// - O. Vasicek, "A test for normality based on sample entropy",
///   *J. R. Stat. Soc. B*, 1976.
/// - B. van Es, "Estimating functionals related to a density by a class of
///   statistics based on spacings", *Scand. J. Stat.*, 1992.
/// - N. Ebrahimi, K. Pflughoeft and E. S. Soofi, "Two measures of sample
///   entropy", *Stat. Probab. Lett.*, 1994.
/// - J. C. Correa, "A new estimator of entropy", *Commun. Stat. Theory
///   Methods*, 1995.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifferentialEntropyMethod {
    /// Selects by sample size: van Es for `n ≤ 10`, Ebrahimi for `n ≤ 1000`,
    /// and Vasicek otherwise.
    Auto,
    /// Vasicek's estimator, `(1/n) Σᵢ ln(n (x₍ᵢ₊ₘ₎ − x₍ᵢ₋ₘ₎) / (2m))`.
    Vasicek,
    /// Van Es's estimator from forward spacings,
    /// `(1/(n−m)) Σᵢ₌₁ⁿ⁻ᵐ ln((n+1)(x₍ᵢ₊ₘ₎ − x₍ᵢ₎) / m) + Σₖ₌ₘⁿ 1/k + ln(m/(n+1))`.
    VanEs,
    /// Ebrahimi's estimator: Vasicek's with `2m` replaced by `cᵢ m`, where
    /// `cᵢ = 1 + (i−1)/m` for `i ≤ m`, `cᵢ = 1 + (n−i)/m` for `i > n−m`, and
    /// `cᵢ = 2` otherwise.
    Ebrahimi,
    /// Correa's estimator, which replaces each spacing with the slope of a
    /// least-squares regression of rank on value over `x₍ᵢ₋ₘ₎, …, x₍ᵢ₊ₘ₎`.
    Correa,
}
/// Options for [`differential_entropy_with_options`]. The estimators do not accept weights.
///
/// The default is [`DifferentialEntropyMethod::Auto`] with the default window
/// and base `e` (nats).
#[derive(Clone, Copy, Debug)]
pub struct DifferentialEntropyOptions<S: RealScalar> {
    /// Estimator.
    pub(crate) method: DifferentialEntropyMethod,
    /// Window `m` with `0 < 2m < n`; `None` uses `m = ⌊√n + 1/2⌋`.
    pub(crate) window: Option<usize>,
    /// Logarithm base, finite and greater than one.
    pub(crate) base: S,
}
impl<S: RealScalar> Default for DifferentialEntropyOptions<S> {
    fn default() -> Self {
        Self {
            method: DifferentialEntropyMethod::Auto,
            window: None,
            base: S::ONE.exp(),
        }
    }
}
/// Sorts `values` into `work` and evaluates the differential-entropy estimator.
pub(crate) fn differential_entropy_sorted<S: RealScalar>(
    values: &[S],
    options: DifferentialEntropyOptions<S>,
    work: &mut [S],
) -> Result<InformationValue<S>, SciOxideError> {
    sample(values, scioxide_core::argument!("values"))?;
    let n = values.len();
    let base = log_base(options.base)?;
    if work.len() < n {
        return Err(invalid(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient entropy workspace",
        ));
    }
    let m = options
        .window
        .unwrap_or_else(|| ((n as f64).sqrt() + 0.5).floor() as usize);
    if m == 0 || m >= n.div_ceil(2) {
        return Err(invalid(
            scioxide_core::argument!("options.window"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "entropy window must satisfy 0<2m<n",
        ));
    }
    work[..n].copy_from_slice(values);
    sort_finite(&mut work[..n]);
    let x = &work[..n];
    let method = if options.method == DifferentialEntropyMethod::Auto {
        if n <= 10 {
            DifferentialEntropyMethod::VanEs
        } else if n <= 1000 {
            DifferentialEntropyMethod::Ebrahimi
        } else {
            DifferentialEntropyMethod::Vasicek
        }
    } else {
        options.method
    };
    let terms = if method == DifferentialEntropyMethod::VanEs {
        n - m
    } else {
        n
    };
    let mut sum = Compensated::<f64>::zero();
    for i in 0..terms {
        let log = match method {
            DifferentialEntropyMethod::VanEs => {
                let dx = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x[i + m]))?
                    .sub(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(x[i]),
                    )?)?;
                if dx.is_zero() {
                    return Ok(InformationValue::NegativeInfinity);
                }
                dx.ln_abs()? + ((n as f64 + 1.) / m as f64).ln()
            }
            DifferentialEntropyMethod::Correa => {
                // Shift by the central observation to protect large common offsets.
                let center = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x[i]))?;
                let width = 2 * m + 1;
                let mut avg = Compensated::<Scaled<f64>>::zero();
                for j in 0..width {
                    let k = i.saturating_add(j).saturating_sub(m).min(n - 1);
                    avg = avg.added(
                        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x[k]))?
                            .sub(center)?,
                    )?;
                }
                let avg = avg
                    .total()?
                    .div(Scaled::<f64>::from_value((width) as f64)?)?;
                let mut numerator = Compensated::<Scaled<f64>>::zero();
                let mut denominator = Compensated::<Scaled<f64>>::zero();
                for j in 0..width {
                    let k = i.saturating_add(j).saturating_sub(m).min(n - 1);
                    let dx = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x[k]))?
                        .sub(center)?
                        .sub(avg)?;
                    numerator = numerator
                        .added(dx.mul(Scaled::<f64>::from_value(j as f64 - m as f64)?)?)?;
                    denominator = denominator.added(dx.square()?)?;
                }
                let num = numerator.total()?;
                let den = denominator.total()?;
                if den.is_zero() {
                    return Ok(InformationValue::NegativeInfinity);
                }
                if num.is_zero() || num.is_sign_negative() {
                    return Err(invalid(
                        scioxide_core::argument!("values"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "degenerate Correa regression",
                    ));
                }
                den.mul(Scaled::<f64>::from_value((n) as f64)?)?
                    .div(num)?
                    .ln_abs()?
            }
            _ => {
                let lo = i.saturating_sub(m);
                let hi = i.saturating_add(m).min(n - 1);
                let dx = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x[hi]))?
                    .sub(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(x[lo]),
                    )?)?;
                if dx.is_zero() {
                    return Ok(InformationValue::NegativeInfinity);
                }
                let width = if method == DifferentialEntropyMethod::Ebrahimi {
                    if i < m {
                        m + i
                    } else if i >= n - m {
                        m + n - 1 - i
                    } else {
                        2 * m
                    }
                } else {
                    2 * m
                };
                dx.ln_abs()? + (n as f64 / width as f64).ln()
            }
        };
        sum = sum.added(log)?;
    }
    let mut value = sum.total()? / terms as f64;
    if method == DifferentialEntropyMethod::VanEs {
        let correction = (m..=n)
            .try_fold(Compensated::<f64>::zero(), |a, k| a.added(1. / k as f64))?
            .total()?;
        value += correction + (m as f64 / (n as f64 + 1.)).ln();
    }
    Ok(InformationValue::Finite(number(value / base)?))
}
/// Computes the result of [`differential_entropy_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`differential_entropy_with_options`] for the fixed default settings.
pub fn differential_entropy<S: RealScalar>(
    values: &[S],
) -> Result<InformationValue<S>, SciOxideError> {
    differential_entropy_with_options(values, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("values"),
        )
    })
}
/// Estimates the differential entropy of a sample with a spacing estimator.
///
/// The estimators are defined on [`DifferentialEntropyMethod`]. The cost is
/// O(n log n) for the sort plus O(n) for the estimate (O(n·m) for Correa's). Tied
/// observations that give a zero spacing make the estimate
/// [`InformationValue::NegativeInfinity`]. Allocates a copy of the sample.
///
/// # Errors
///
/// - `InvalidInput` if `values` is empty.
/// - `NonFinite` if a value is NaN or infinite.
/// - `InvalidInput` if the base is not finite or not greater than 1.
/// - `InvalidInput` unless `0 < 2m < n`. The default window needs `n ≥ 5`.
/// - `InvalidInput` if Correa's local regression has a nonpositive slope.
pub fn differential_entropy_with_options<S: RealScalar>(
    values: &[S],
    options: DifferentialEntropyOptions<S>,
) -> Result<InformationValue<S>, SciOxideError> {
    let mut w = RawDescriptiveWorkspace::default();
    w.reserve(values.len())?;
    w.differential_entropy_with_options(values, options)
}

impl<S: RealScalar> EntropyOptions<S> {
    /// Returns the default options: base `e` and [`MassNormalization::Normalize`].
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the logarithm base. [`entropy_with_options`] checks it.
    pub const fn with_base(mut self, value: S) -> Self {
        self.base = value;
        self
    }
    /// Returns the logarithm base.
    pub const fn base(self) -> S {
        self.base
    }
    /// Sets the mass normalization.
    pub const fn with_normalization(mut self, value: MassNormalization<S>) -> Self {
        self.normalization = value;
        self
    }
    /// Returns the mass normalization.
    pub const fn normalization(self) -> MassNormalization<S> {
        self.normalization
    }
}

impl<S: RealScalar> DifferentialEntropyOptions<S> {
    /// Returns the default options: automatic method, default window and base `e`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the estimator.
    pub const fn with_method(mut self, value: DifferentialEntropyMethod) -> Self {
        self.method = value;
        self
    }
    /// Returns the estimator.
    pub const fn method(self) -> DifferentialEntropyMethod {
        self.method
    }
    /// Sets the window `m`, or `None` for `m = ⌊√n + 1/2⌋`. It is checked
    /// against the sample size when the estimate is computed.
    pub const fn with_window(mut self, value: Option<usize>) -> Self {
        self.window = value;
        self
    }
    /// Returns the window `m`.
    pub const fn window(self) -> Option<usize> {
        self.window
    }
    /// Sets the logarithm base. It is checked when the estimate is computed.
    pub const fn with_base(mut self, value: S) -> Self {
        self.base = value;
        self
    }
    /// Returns the logarithm base.
    pub const fn base(self) -> S {
        self.base
    }
}

/// Computes the result of [`entropy_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the configured form's errors for these defaults.
pub fn relative_entropy<S: RealScalar>(
    p: &[S],
    q: &[S],
) -> Result<InformationValue<S>, SciOxideError> {
    entropy_with_options(p, Some(q), Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &[
                "options",
                "weights",
                "left_weights",
                "right_weights",
                "direction",
                "ties",
            ],
            scioxide_core::argument!("p"),
        )
    })
}
