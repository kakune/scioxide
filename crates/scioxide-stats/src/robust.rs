//! Means, extrema, trimmed means, MAD and L-moments of checked real samples.
use crate::arithmetic::Compensated;
use crate::empirical::select_adjacent_by;
use crate::error_helpers::{check_weights, finite, invalid, mass, number, sample};
use crate::ranking::sort_finite;
use crate::{KernelWeights, MomentOrder, OnlineMoments, QuantileMethod, QuantileProbability};
use scioxide_core::{Argument, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};

/// Which end of a sorted sample a trimmed statistic removes.
///
/// For a trim proportion `p` in `[0, 1)` and `n` observations, `⌊p n⌋`
/// observations are removed from each selected end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimSide {
    /// Removes the smallest observations.
    Lower,
    /// Removes the largest observations.
    Upper,
    /// Removes `⌊p n⌋` observations from each end.
    Both,
}
/// Descriptive statistic of a real sample.
///
/// Estimators that do not accept weights return an error before evaluation.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RobustStatistic<S: RealScalar> {
    /// Sum `Σ wᵢ xᵢ`.
    Sum,
    /// Arithmetic mean, with relative masses or frequencies as weights.
    Mean,
    /// Smallest observation.
    Minimum,
    /// Largest observation.
    Maximum,
    /// Geometric mean of nonnegative values. A zero with positive weight gives zero.
    GeometricMean,
    /// Harmonic mean of nonnegative values. A zero with positive weight gives zero.
    HarmonicMean,
    /// Power mean `(Σ wᵢ xᵢᵖ / Σ wᵢ)^(1/p)` of nonnegative values.
    ///
    /// `p = 0` gives the geometric mean, and `p = ±∞` the maximum or minimum.
    PowerMean(S),
    /// Geometric standard deviation `exp(sd(ln x))`, unweighted, with divisor `n − ddof`.
    GeometricStandardDeviation(usize),
    /// Median absolute deviation from the median, divided by a positive scale.
    MedianAbsoluteDeviation {
        /// Quantile definition used for both medians.
        method: QuantileMethod,
        /// Positive divisor. A divisor of 1 gives the raw MAD.
        scale: S,
    },
    /// Trimmed mean, with `⌊proportion · n⌋` observations removed from each selected end.
    TrimmedMean {
        /// Fraction in `[0, 1)`.
        proportion: S,
        /// Ends to trim.
        side: TrimSide,
    },
    /// Unbiased sample L-moment `lᵣ`, or the ratio `lᵣ / l₂` for `r ≥ 3`.
    LMoment {
        /// Order `r`, from 1 to the sample size.
        order: usize,
        /// Whether to return the ratio `lᵣ / l₂` for `r ≥ 3`.
        standardize: crate::LMomentNormalization,
    },
}
/// Evaluates `statistic` for `RawDescriptiveWorkspace::evaluate`, using the given scratch storage.
pub(crate) fn evaluate<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    statistic: RobustStatistic<S>,
    work: &mut [S],
    deviations: &mut [Scaled<f64>],
) -> Result<S, SciOxideError> {
    if let Some(value) = direct_statistic(values, weights, statistic)? {
        return Ok(value);
    }
    if values.len() > work.len() {
        return Err(invalid(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient descriptive workspace",
        ));
    }
    check_weights(weights, values.len(), scioxide_core::argument!("weights"))?;
    use RobustStatistic as D;
    if weights.is_some() {
        return Err(invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "this estimator accepts unweighted observations only",
        ));
    }
    sample(values, scioxide_core::argument!("values"))?;
    match statistic {
        D::MedianAbsoluteDeviation { method, scale } => {
            if !scale.is_finite() || scale <= S::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("statistic.scale"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "MAD scale must be positive and finite",
                ));
            }
            let center = median(values, method, work)?.into_f64();
            // `median` left a permutation of the sample in `work`.
            let work = &mut work[..values.len()];
            scaled_quantile_by(
                values.len(),
                QuantileProbability::new(0.5)?,
                method,
                |lower, upper| {
                    if let Some(selected) = native_deviations(work, center, lower, upper)? {
                        return Ok(selected);
                    }
                    let center = Scaled::<f64>::from_value(center)?;
                    for (slot, &x) in deviations.iter_mut().zip(values) {
                        *slot = Scaled::<f64>::from_value(x.into_f64())?.sub(center)?.abs();
                    }
                    select_adjacent_by(&mut deviations[..values.len()], lower, upper, |a, b| {
                        a.abs_cmp(*b)
                    })
                },
            )?
            .total()?
            .div(Scaled::<f64>::from_value(scale.into_f64())?)?
            .round_to()
        }
        D::TrimmedMean { proportion, side } => {
            let (lo, hi) = trim_range(
                values.len(),
                proportion,
                side,
                scioxide_core::argument!("statistic.proportion"),
            )?;
            work[..values.len()].copy_from_slice(values);
            sort_finite(&mut work[..values.len()]);
            arithmetic_mean(&work[lo..hi], None, false)
        }
        D::LMoment { order, standardize } => {
            work[..values.len()].copy_from_slice(values);
            sort_finite(&mut work[..values.len()]);
            let result = lmoment(&work[..values.len()], order)?;
            let result = if standardize == crate::LMomentNormalization::Standardized && order >= 3 {
                let l2 = lmoment(&work[..values.len()], 2)?;
                if l2.is_zero() {
                    return Err(invalid(
                        scioxide_core::argument!("values"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "L-moment ratio undefined for zero L2",
                    ));
                }
                result.div(l2)?
            } else {
                result
            };
            result.round_to()
        }
        D::GeometricStandardDeviation(ddof) => {
            for (slot, &x) in work.iter_mut().zip(values) {
                if x <= S::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("values"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "geometric deviation requires positive samples",
                    ));
                }
                *slot = number(crate::arithmetic::log_ratio(x, values[0])?)?
            }
            let m = OnlineMoments::from_slice(&work[..values.len()], MomentOrder::Second)?;
            let sd = m.snapshot().standard_deviation(ddof)?.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("statistic"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "insufficient degrees of freedom",
                )
            })?;
            finite(sd.exp())
        }
        _ => Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid descriptive operation",
        )),
    }
}

/// Writes the trimmed sorted sample for `RawDescriptiveWorkspace::trim_into`, using given scratch.
pub(crate) fn trim_into<S: RealScalar>(
    values: &[S],
    proportion: S,
    side: TrimSide,
    output: &mut [S],
    work: &mut [S],
) -> Result<usize, SciOxideError> {
    sample(values, scioxide_core::argument!("values"))?;
    let (lo, hi) = trim_range(
        values.len(),
        proportion,
        side,
        scioxide_core::argument!("proportion"),
    )?;
    if work.len() < values.len() || output.len() < hi - lo {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient trim storage",
        ));
    }
    work[..values.len()].copy_from_slice(values);
    sort_finite(&mut work[..values.len()]);
    output[..hi - lo].copy_from_slice(&work[lo..hi]);
    Ok(hi - lo)
}
// Pure reductions bypass ordering and transform storage at both entry points.
fn direct_statistic<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    statistic: RobustStatistic<S>,
) -> Result<Option<S>, SciOxideError> {
    use RobustStatistic as D;
    if !matches!(
        statistic,
        D::Mean
            | D::Sum
            | D::GeometricMean
            | D::HarmonicMean
            | D::PowerMean(_)
            | D::Minimum
            | D::Maximum
    ) {
        return Ok(None);
    }
    check_weights(weights, values.len(), scioxide_core::argument!("weights"))?;
    let value = match statistic {
        D::Mean => arithmetic_mean(values, weights, false)?,
        D::Sum => arithmetic_mean(values, weights, true)?,
        D::GeometricMean => power_mean(values, weights, S::ZERO)?,
        D::HarmonicMean => power_mean(values, weights, -S::ONE)?,
        D::PowerMean(p) => power_mean(values, weights, p)?,
        _ => {
            if weights.is_some() {
                return Err(invalid(
                    scioxide_core::argument!("weights"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "this estimator accepts unweighted observations only",
                ));
            }
            sample(values, scioxide_core::argument!("values"))?;
            match statistic {
                D::Minimum => values.iter().copied().fold(values[0], |a, b| a.min(b)),
                D::Maximum => values.iter().copied().fold(values[0], |a, b| a.max(b)),
                _ => return Ok(None),
            }
        }
    };
    Ok(Some(value))
}
/// Returns the Hyndman–Fan quantile of `len` extended-range values from two order statistics.
///
/// `select(lower, upper)` returns the zero-based order statistics `lower` and
/// `upper` that the quantile position needs. Callers can select them in linear
/// time instead of sorting. The interpolation is the same as for a sorted sample.
///
/// # Errors
///
/// Returns position, selection or interpolation errors.
pub(crate) fn scaled_quantile_by(
    len: usize,
    probability: QuantileProbability,
    method: QuantileMethod,
    select: impl FnOnce(usize, usize) -> Result<[Scaled<f64>; 2], SciOxideError>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    // The position reads its ranks from placeholder values; it has no other effect.
    let ranks = core::cell::Cell::new([usize::MAX, 0]);
    crate::empirical::quantile_position(len, probability, method)?.evaluate(|rank| {
        let [lower, upper] = ranks.get();
        ranks.set([lower.min(rank), upper.max(rank)]);
        0.0_f64
    })?;
    let [lower, upper] = ranks.get();
    let [low, high] = select(lower, upper)?;
    crate::empirical::quantile_scaled_sorted_by(len, probability, method, |rank| {
        Ok(if rank == lower { low } else { high })
    })
}
// Select absolute deviations from `center` as binary64 keys of the sample.
// A finite IEEE difference of binary64 values is the extended difference
// rounded once (a subnormal one is exact), and rounding is monotone, so
// finite selected keys are exactly the extended order statistics even when
// unselected keys overflow. `None` requests the extended replay.
fn native_deviations<S: RealScalar>(
    sample: &mut [S],
    center: f64,
    lower: usize,
    upper: usize,
) -> Result<Option<[Scaled<f64>; 2]>, SciOxideError> {
    let deviation = |x: &S| (x.into_f64() - center).abs();
    let selected = select_adjacent_by(sample, lower, upper, |a, b| {
        deviation(a).total_cmp(&deviation(b))
    })?
    .map(|x| deviation(&x));
    if selected.iter().any(|d| !d.is_finite()) {
        return Ok(None);
    }
    Ok(Some([
        Scaled::from_value(selected[0])?,
        Scaled::from_value(selected[1])?,
    ]))
}
fn median<S: RealScalar>(
    values: &[S],
    method: QuantileMethod,
    scratch: &mut [S],
) -> Result<S, SciOxideError> {
    scratch[..values.len()].copy_from_slice(values);
    crate::empirical::quantile_in_place(
        &mut scratch[..values.len()],
        QuantileProbability::new(0.5)?,
        method,
    )
    .map(|q| q.value())
}
fn trim_range<S: RealScalar>(
    n: usize,
    proportion: S,
    side: TrimSide,
    argument: Argument,
) -> Result<(usize, usize), SciOxideError> {
    if !proportion.is_finite() || proportion < S::ZERO || proportion >= S::ONE {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "trim proportion must be in [0,1)",
        ));
    }
    let k = (proportion.try_to_f64_exact()? * n as f64).floor() as usize;
    let lo = if side == TrimSide::Upper { 0 } else { k };
    let hi = n.saturating_sub(if side == TrimSide::Lower { 0 } else { k });
    if lo >= hi {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "trimming leaves no observations",
        ));
    }
    Ok((lo, hi))
}
pub(crate) fn arithmetic_mean<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    sum: bool,
) -> Result<S, SciOxideError> {
    check_weights(weights, values.len(), scioxide_core::argument!("weights"))?;
    if let Some(value) = native_arithmetic_mean(values, weights, sum)? {
        return Ok(value);
    }
    let mut total = Compensated::<Scaled<f64>>::zero();
    let mut count = Compensated::<Scaled<f64>>::zero();
    for (i, &x) in values.iter().enumerate() {
        let w = mass(weights, i, scioxide_core::argument!("weights"))?;
        if w.is_zero() {
            continue;
        }
        if !x.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "nonfinite positive-mass observation",
            ));
        }
        total = total
            .added(Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?.mul(w)?)?;
        count = count.added(w)?;
    }
    let count = count.total()?;
    if count.is_zero() {
        return if sum {
            Ok(S::ZERO)
        } else {
            Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "empty positive-mass sample",
            ))
        };
    }
    let total = if sum {
        total.total()?
    } else {
        total.total()?.div(count)?
    };
    total.round_to()
}
fn power_mean<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    p: S,
) -> Result<S, SciOxideError> {
    if p.is_nan() {
        return Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "power must not be NaN",
        ));
    }
    if let Some(value) = native_power_mean(values, weights, p)? {
        return Ok(value);
    }
    let mut total = Compensated::<Scaled<f64>>::zero();
    let mut low = None;
    let mut high = None;
    let mut zero = false;
    for (i, &x) in values.iter().enumerate() {
        let w = mass(weights, i, scioxide_core::argument!("weights"))?;
        if w.is_zero() {
            continue;
        }
        if !x.is_finite() || x < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "power means require finite nonnegative observations",
            ));
        }
        low = Some(low.map_or(x, |a: S| a.min(x)));
        high = Some(high.map_or(x, |a: S| a.max(x)));
        zero |= x == S::ZERO;
        total = total.added(w)?;
    }
    let total = total.total()?;
    let high = high.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass sample",
        )
    })?;
    let low = low.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass sample",
        )
    })?;
    if low == high {
        return Ok(low);
    }
    if p.is_infinite() {
        return Ok(if p > S::ZERO { high } else { low });
    }
    if high == S::ZERO || (zero && p <= S::ZERO) {
        return Ok(S::ZERO);
    }
    if p == S::ONE {
        return arithmetic_mean(values, weights, false);
    }
    if p == -S::ONE {
        let mut reciprocal = Compensated::<Scaled<f64>>::zero();
        for (i, &x) in values.iter().enumerate() {
            let w = mass(weights, i, scioxide_core::argument!("weights"))?;
            if !w.is_zero() {
                reciprocal = reciprocal.added(w.div(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(x),
                )?)?)?;
            }
        }
        return total.div(reciprocal.total()?)?.round_to();
    }
    if p == S::ZERO {
        let mut logs = Compensated::<Scaled<f64>>::zero();
        for (i, &x) in values.iter().enumerate() {
            let w = mass(weights, i, scioxide_core::argument!("weights"))?;
            if !w.is_zero() {
                logs = logs.added(w.mul(Scaled::<f64>::from_value(
                    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?.ln_abs()?,
                )?)?)?;
            }
        }
        let log: f64 = logs.total()?.div(total)?.round_to()?;
        return number::<S>(
            log.exp()
                .min(high.try_to_f64_exact()?)
                .max(low.try_to_f64_exact()?),
        );
    }
    // Scale to the endpoint at which x^p is largest. Log ratios avoid x/ref underflow.
    let reference = if p > S::ZERO { high } else { low };
    let small = !zero
        && p.abs().try_to_f64_exact()?
            * Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(high))?
                .div(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(low),
                )?)?
                .ln_abs()?
            < 0.5;
    let mut terms = Compensated::<Scaled<f64>>::zero();
    for (i, &x) in values.iter().enumerate() {
        let w = mass(weights, i, scioxide_core::argument!("weights"))?;
        if w.is_zero() || x == S::ZERO {
            continue;
        }
        let delta = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?
            .div(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(reference),
            )?)?
            .ln_abs()?;
        let exponent = p.try_to_f64_exact()? * delta;
        let term = if small {
            exponent.exp_m1()
        } else {
            exponent.exp()
        };
        terms = terms.added(w.mul(Scaled::<f64>::from_value(term)?)?)?;
    }
    let ratio = terms.total()?.div(total)?;
    let log_ratio = if small {
        let v: f64 = ratio.round_to()?;
        v.ln_1p()
    } else {
        ratio.ln_abs()?
    } / p.try_to_f64_exact()?;
    let binary = (log_ratio / core::f64::consts::LN_2).floor();
    if !binary.is_finite() || binary.abs() > f64::from(i32::MAX - 2) {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "power mean scaling exceeds supported exponent range",
        ));
    }
    let correction = Scaled::<f64>::from_parts(
        (log_ratio - binary * core::f64::consts::LN_2).exp(),
        i64::from(binary as i32),
    )?;
    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(reference))?
        .mul(correction)?
        .round_to()
}
// Hosking sample L-moments: average signed linear combinations of order statistics.
// Recurrence computes combinatorial ratios without factorials or integer overflow.
fn lmoment<S: RealScalar>(sorted: &[S], order: usize) -> Result<Scaled<f64>, SciOxideError> {
    let n = sorted.len();
    if order == 0 || order > n {
        return Err(invalid(
            scioxide_core::argument!("statistic.order"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "L-moment order must be in 1..=sample count",
        ));
    }
    let mut result = Compensated::<Scaled<f64>>::zero();
    for (i, &x) in sorted.iter().enumerate() {
        let mut coefficient = Compensated::<Scaled<f64>>::zero();
        let mut choose = Scaled::<f64>::ONE;
        for k in 0..order {
            // C(i, order-1-k) C(n-1-i,k) / C(n-1,order-1).
            let left = order - 1 - k;
            if i >= left && n - 1 - i >= k {
                let mut ratio = Scaled::<f64>::ONE;
                for j in 0..left {
                    ratio = ratio
                        .mul(Scaled::<f64>::from_value((i - j) as f64)?)?
                        .div(Scaled::<f64>::from_value((n - 1 - j) as f64)?)?;
                }
                for j in 0..k {
                    ratio = ratio
                        .mul(Scaled::<f64>::from_value((n - 1 - i - j) as f64)?)?
                        .div(Scaled::<f64>::from_value((n - 1 - left - j) as f64)?)?;
                }
                // C(order-1,k) occurs both in the hypergeometric factor and signed L combination.
                ratio = ratio.mul(choose)?.mul(choose)?;
                if k % 2 != 0 {
                    ratio = Scaled::<f64>::ZERO.sub(ratio)?;
                }
                coefficient = coefficient.added(ratio)?;
            }
            if k + 1 < order {
                choose = choose
                    .mul(Scaled::<f64>::from_value((order - 1 - k) as f64)?)?
                    .div(Scaled::<f64>::from_value((k + 1) as f64)?)?;
            }
        }
        let x = if order > 1 {
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?.sub(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(sorted[0]))?,
            )?
        } else {
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?
        };
        result = result.added(x.mul(coefficient.total()?)?)?;
    }
    result.total()?.div(Scaled::<f64>::from_value((n) as f64)?)
}

// Relative logarithms retain small spreads near a large common observation.
// With source exponents in [-64,64], products, quotients and slice sums
// (at most usize::MAX terms) remain normal binary64. Guard before arithmetic;
// a miss replays the immutable input through the extended implementation.
fn native_arithmetic_mean<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    sum: bool,
) -> Result<Option<S>, SciOxideError> {
    match weights {
        None => crate::numeric_statistics::native_location(values, sum),
        Some(KernelWeights::Mass(w)) => {
            native_arithmetic_mean_by::<S, true>(values, w.iter().map(|&w| w.into_f64()), sum)
        }
        Some(KernelWeights::Frequency(w)) => {
            native_arithmetic_mean_by::<S, true>(values, w.iter().map(|&w| w as f64), sum)
        }
    }
}
fn native_arithmetic_mean_by<S: RealScalar, const WEIGHTED: bool>(
    values: &[S],
    weights: impl Iterator<Item = f64> + Clone,
    sum: bool,
) -> Result<Option<S>, SciOxideError> {
    let native = if WEIGHTED {
        values
            .iter()
            .zip(weights.clone())
            .fold(true, |native, (&x, w)| {
                native
                    & (w >= 0.0)
                    & ((w == 0.0)
                        | (crate::arithmetic::native(x.into_f64(), 64)
                            & crate::arithmetic::native(w, 64)))
            })
    } else {
        crate::arithmetic::native_sources(values.iter().map(|&x| x.into_f64()))
    };
    if !native {
        return Ok(None);
    }
    let (total, mass) = if WEIGHTED {
        let Ok([total, mass]) =
            scioxide_core::compensated_reduce(values.iter().zip(weights).map(|(&x, w)| {
                if w == 0.0 {
                    [0.0; 2]
                } else {
                    [w * x.into_f64(), w]
                }
            }))
        else {
            return Ok(None);
        };
        (total, mass.total()?)
    } else {
        let Ok([total]) = scioxide_core::compensated_reduce(values.iter().map(|&x| [x.into_f64()]))
        else {
            return Ok(None);
        };
        (total, values.len() as f64)
    };
    if mass == 0.0 && !sum {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass sample",
        ));
    }
    number(if sum {
        total.total()?
    } else {
        crate::arithmetic::sum_mean(total.parts(), mass)
    })
    .map(Some)
}
fn native_power_mean<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    p: S,
) -> Result<Option<S>, SciOxideError> {
    match weights {
        None => native_power_mean_by::<S, false>(values, core::iter::repeat(1.0), p),
        Some(KernelWeights::Mass(w)) => {
            native_power_mean_by::<S, true>(values, w.iter().map(|&w| w.into_f64()), p)
        }
        Some(KernelWeights::Frequency(w)) => {
            native_power_mean_by::<S, true>(values, w.iter().map(|&w| w as f64), p)
        }
    }
}
fn native_power_mean_by<S: RealScalar, const WEIGHTED: bool>(
    values: &[S],
    weights: impl Iterator<Item = f64> + Clone,
    p: S,
) -> Result<Option<S>, SciOxideError> {
    let power = p.into_f64();
    let mut low = f64::INFINITY;
    let mut high = 0.0_f64;
    let native = values
        .iter()
        .zip(weights.clone())
        .fold(true, |native, (&x, w)| {
            if w == 0.0 {
                return native;
            }
            let x = x.into_f64();
            low = low.min(x);
            high = high.max(x);
            native
                & (x >= 0.0)
                & (w >= 0.0)
                & crate::arithmetic::native(x, 64)
                & crate::arithmetic::native(w, 64)
        });
    if !native {
        return Ok(None);
    }
    let mass = if WEIGHTED {
        let Ok([mass]) =
            scioxide_core::compensated_reduce(weights.clone().take(values.len()).map(|w| [w]))
        else {
            return Ok(None);
        };
        mass.total()?
    } else {
        values.len() as f64
    };
    if mass == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass sample",
        ));
    }
    if low == high {
        return number(low).map(Some);
    }
    if power.is_infinite() {
        return number(if power > 0.0 { high } else { low }).map(Some);
    }
    if low == 0.0 && power <= 0.0 {
        return Ok(Some(S::ZERO));
    }
    if power == 1.0 {
        return native_arithmetic_mean_by::<S, WEIGHTED>(values, weights, false);
    }
    let reference = if power > 0.0 { high } else { low };
    let log_ratio = |x: f64| {
        let ratio = x / reference;
        if (0.5..=2.0).contains(&ratio) {
            ((x - reference) / reference).ln_1p()
        } else {
            ratio.ln()
        }
    };
    let small = low > 0.0 && power.abs() * (high / low).ln() < 0.5;
    let mut native = true;
    let terms = scioxide_core::compensated_reduce(values.iter().zip(weights).map(|(&x, w)| {
        // The admission pass validated immutable positive-mass observations.
        if w == 0.0 {
            return [0.0];
        }
        let x = x.into_f64();
        let term = if power == 3.0 {
            x * x * x
        } else if power == 2.0 {
            x * x
        } else if power == -1.0 {
            1.0 / x
        } else if power == 0.0 {
            x.ln()
        } else if x == 0.0 {
            0.0
        } else {
            let exponent = power * log_ratio(x);
            // Rejected transformed ranges replay before any scalar publication.
            native &= exponent.is_finite() & (exponent >= -256.0 * core::f64::consts::LN_2);
            if small {
                exponent.exp_m1()
            } else {
                exponent.exp()
            }
        };
        [w * term]
    }));
    let Ok([terms]) = terms else {
        return Ok(None);
    };
    if !native {
        return Ok(None);
    }
    let total = terms.total()?;
    let value = if power == 3.0 {
        (total / mass).cbrt()
    } else if power == 2.0 {
        (total / mass).sqrt()
    } else if power == -1.0 {
        mass / total
    } else if power == 0.0 {
        (total / mass).exp()
    } else {
        let log = if small {
            (total / mass).ln_1p()
        } else {
            (total / mass).ln()
        } / power;
        reference * log.exp()
    };
    if !value.is_finite() || (value == 0.0 && high > 0.0 && low > 0.0) {
        return Ok(None);
    }
    number(value.min(high).max(low)).map(Some)
}
