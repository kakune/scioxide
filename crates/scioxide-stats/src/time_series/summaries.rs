use super::*;
use crate::arithmetic::Arithmetic;
use crate::arithmetic::{center_difference, normalized_deviation, scaled_center, shift};
use crate::error_helpers::{input_finite, limit_bytes};
use crate::{BatchView, MissingPolicy, NonFinitePolicy, StatisticsOptions};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ops::Neg;
use scioxide_core::{AlgebraicScalar, Capacity, Complex, ExecutionContext};
use scioxide_core::{Scaled, ScaledAccumulator, ScaledValue};

type Lagged<T> = ScaledValue<Complex<<T as NumericScalar>::Real>>;

/// Statistic computed at each lag `k = 0, …, max_lag` of a time series.
///
/// With `x̄` the centering of [`TimeSeriesOptions::center`], the sample
/// autocovariance is
///
/// ```text
/// γ̂(k) = (1/D) Σₜ (xₜ − x̄) conj(xₜ₋ₖ − x̄)
/// ```
///
/// where the sum runs over the pairs `(t, t − k)` with both observations
/// present, and the divisor `D` is set by [`LagNormalization`].
///
/// # References
///
/// P. J. Brockwell and R. A. Davis, *Time Series: Theory and Methods*, 2nd ed.,
/// Springer, 1991, ch. 5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeSeriesStatistic {
    /// Autocovariance `γ̂(k)`.
    Covariance,
    /// Autocorrelation `ρ̂(k) = γ̂(k) / γ̂(0)`, with `ρ̂(0) = 1`.
    Acf,
    /// Partial autocorrelation `φ̂ₖₖ`, the last coefficient of the order-`k`
    /// autoregression from the Levinson–Durbin recursion on `ρ̂`. The lag-zero
    /// value is 1.
    Pacf,
}
/// Divisor `D` of the sample autocovariance at each lag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LagNormalization {
    /// Divides every lag by the number of present observations. The resulting
    /// autocovariance sequence is positive semidefinite.
    #[default]
    ObservationCount,
    /// Divides each lag by its number of present pairs. This reduces the bias
    /// at large lags, but the sequence need not be positive semidefinite.
    PairCount,
}
/// Configures the divisor and centering of a time-series summary.
///
/// These choices define the statistic. Missing and non-finite values,
/// execution and the memory limit are set separately in [`StatisticsOptions`].
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesOptions {
    normalization: LagNormalization,
    center: crate::TimeSeriesCentering,
}

impl Default for TimeSeriesOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl TimeSeriesOptions {
    /// Returns options with the divisor
    /// [`LagNormalization::ObservationCount`] and centering at the sample mean.
    pub const fn new() -> Self {
        Self {
            normalization: LagNormalization::ObservationCount,
            center: crate::TimeSeriesCentering::SampleMean,
        }
    }

    /// Returns the divisor of the autocovariance.
    pub const fn normalization(&self) -> LagNormalization {
        self.normalization
    }
    /// Sets the divisor of the autocovariance.
    pub const fn with_normalization(mut self, value: LagNormalization) -> Self {
        self.normalization = value;
        self
    }
    /// Returns the centering: the mean of the present observations, or a known
    /// mean of zero.
    pub const fn center(&self) -> crate::TimeSeriesCentering {
        self.center
    }
    /// Sets the centering.
    pub const fn with_center(mut self, value: crate::TimeSeriesCentering) -> Self {
        self.center = value;
        self
    }
}
/// Value of a time-series statistic at one lag.
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesLag<T: NumericScalar> {
    /// Value of the statistic at this lag.
    pub value: T,
    /// Number of pairs with both observations present at this lag, whatever
    /// the divisor.
    pub pairs: usize,
}
impl<T: NumericScalar> Default for TimeSeriesLag<T> {
    fn default() -> Self {
        Self {
            value: T::ZERO,
            pairs: 0,
        }
    }
}

/// Scratch storage of one worker. Complex records also store their centered
/// values.
struct LagScratch<T: NumericScalar> {
    centered: Vec<Lagged<T>>,
    covariance: Vec<Lagged<T>>,
    coefficients: Vec<T>,
    previous: Vec<T>,
}
impl<T: NumericScalar> LagScratch<T> {
    fn new(samples: usize, lags: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            centered: try_filled(
                samples,
                ScaledValue::ZERO,
                scioxide_core::quantity!("time_series_centered"),
            )?,
            covariance: try_filled(
                lags,
                ScaledValue::ZERO,
                scioxide_core::quantity!("time_series_covariance"),
            )?,
            coefficients: try_filled(
                lags,
                T::ZERO,
                scioxide_core::quantity!("time_series_coefficients"),
            )?,
            previous: try_filled(
                lags,
                T::ZERO,
                scioxide_core::quantity!("time_series_previous"),
            )?,
        })
    }
}
/// Bytes of `workers` scratch buffers and `stage` output lags.
fn payload<T: NumericScalar>(
    samples: usize,
    lags: usize,
    workers: usize,
    stage: usize,
) -> Result<usize, SciOxideError> {
    let worker = crate::estimation::storage_bytes(&[
        (add(samples, lags)?, size_of::<Lagged<T>>()),
        (lags, mul(2, size_of::<T>())?),
        (1, size_of::<LagScratch<T>>()),
    ])?;
    crate::estimation::storage_bytes(&[(workers, worker), (stage, size_of::<TimeSeriesLag<T>>())])
}

/// Workspace size of a time-series summary, from
/// [`TimeSeriesWorkspace::scalar_requirements`] or
/// [`TimeSeriesWorkspace::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct TimeSeriesRequirements<T: NumericScalar> {
    samples: usize,
    lags: usize,
    lanes: usize,
    workers: usize,
    bytes: usize,
    max_bytes: usize,
    scalar: PhantomData<T>,
}
impl<T: NumericScalar> TimeSeriesRequirements<T> {
    fn new(
        samples: usize,
        max_lag: usize,
        lanes: usize,
        workers: usize,
        options: StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        if samples == 0 || max_lag >= samples {
            return Err(invalid(
                scioxide_core::argument!("max_lag"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty record or invalid maximum lag",
            ));
        }
        let lags = add(max_lag, 1)?;
        let bytes = payload::<T>(
            if T::IS_REAL { 0 } else { samples },
            lags,
            workers,
            mul(lanes, lags)?,
        )?;
        limit_bytes(bytes, options.max_bytes())?;
        Ok(Self {
            samples,
            lags,
            lanes,
            workers,
            bytes,
            max_bytes: options.max_bytes(),
            scalar: PhantomData,
        })
    }
    /// Returns the number of bytes the workspace needs, without the output and
    /// allocator overhead. For real data it does not depend on the record
    /// length.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
    /// Returns the record length, including gaps.
    pub const fn samples(self) -> usize {
        self.samples
    }
    /// Returns the number of lags per record, `max_lag + 1`.
    pub const fn lags(self) -> usize {
        self.lags
    }
    /// Returns the number of records.
    pub const fn lanes(self) -> usize {
        self.lanes
    }
    /// Returns the number of workers.
    pub const fn workers(self) -> usize {
        self.workers
    }
}

/// Reusable workspace for [`time_series_summary_into`] and
/// [`time_series_summaries_into`].
///
/// See [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, StridedLayout};
/// use scioxide_stats::{
///     Axis, BatchView, StatisticsOptions, TimeSeriesLag, TimeSeriesOptions, TimeSeriesStatistic,
///     TimeSeriesWorkspace, time_series_summaries_into,
/// };
///
/// // Two records, one per row. Both have lag-1 autocorrelation (5/16) / (5/4) = 1/4.
/// let values = [1.0_f64, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
/// let layout = StridedLayout::new(&[2, 4], &[4, 1])?;
/// let view = BatchView::new(&values, layout, Axis::Index(1))?;
/// let (summary, options) = (TimeSeriesOptions::new(), StatisticsOptions::new());
/// let (acf, context) = (TimeSeriesStatistic::Acf, ExecutionContext::sequential());
/// let req = TimeSeriesWorkspace::workspace_requirements(view, 1, options, context)?;
/// let mut ws = TimeSeriesWorkspace::workspace(&req)?;
/// let mut output = [TimeSeriesLag::default(); 4];
/// time_series_summaries_into(view, acf, 1, summary, options, &mut output, &mut ws, context)?;
/// assert!((output[1].value - 0.25).abs() < 1e-14);
/// assert!((output[3].value - 0.25).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct TimeSeriesWorkspace<T: NumericScalar> {
    workers: Vec<LagScratch<T>>,
    stage: Vec<TimeSeriesLag<T>>,
    samples: usize,
    lags: usize,
}
impl<T: NumericScalar> Default for TimeSeriesWorkspace<T> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            stage: Vec::new(),
            samples: 0,
            lags: 0,
        }
    }
}
impl<T: NumericScalar> TimeSeriesWorkspace<T> {
    /// Returns the workspace size for one record of `samples` observations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `samples` is zero or `max_lag` is not less
    ///   than `samples`.
    /// - `ResourceLimit` if the size exceeds `options.max_bytes()`.
    pub fn scalar_requirements(
        samples: usize,
        max_lag: usize,
        options: StatisticsOptions,
    ) -> Result<TimeSeriesRequirements<T>, SciOxideError> {
        TimeSeriesRequirements::new(samples, max_lag, 1, 1, options)
    }
    /// Returns the workspace size for all records of `view`.
    ///
    /// The sample axis of `view` is time, and each batch is a separate record.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `view` has vector observations.
    /// - `InvalidInput` if the records are empty or `max_lag` is not
    ///   less than their length.
    /// - `ResourceLimit` if the size exceeds `options.max_bytes()`.
    pub fn workspace_requirements(
        view: BatchView<'_, T>,
        max_lag: usize,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<TimeSeriesRequirements<T>, SciOxideError> {
        if view.dimensions() != 1 {
            return Err(invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "time-series observations must be scalar",
            ));
        }
        let workers = crate::estimation::worker_count(options, context, view.batches())?;
        TimeSeriesRequirements::new(view.samples(), max_lag, view.batches(), workers, options)
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        payload::<T>(
            self.samples,
            self.lags,
            self.workers.len(),
            self.stage.len(),
        )
    }
    /// Returns whether the workspace is large enough for `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage the workspace holds exceeds the memory
    ///   limit of `req`.
    pub fn check(&self, req: &TimeSeriesRequirements<T>) -> Result<Capacity, SciOxideError> {
        limit_bytes(self.retained_bytes()?, req.max_bytes)?;
        Ok(
            if self.workers.len() >= req.workers
                && self.lags >= req.lags
                && (T::IS_REAL || self.samples >= req.samples)
                && self.stage.len() >= mul(req.lanes, req.lags)?
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to fit `req`. Larger existing buffers are kept.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the memory limit of
    ///   `req`.
    /// - `AllocationFailure` if memory cannot be reserved. The workspace is then
    ///   unchanged.
    pub fn try_reserve(&mut self, req: &TimeSeriesRequirements<T>) -> Result<(), SciOxideError> {
        if self.check(req)? == Capacity::Sufficient {
            return Ok(());
        }
        let samples = if T::IS_REAL {
            0
        } else {
            self.samples.max(req.samples)
        };
        let lags = self.lags.max(req.lags);
        let count = self.workers.len().max(req.workers);
        let stage = self.stage.len().max(mul(req.lanes, req.lags)?);
        limit_bytes(payload::<T>(samples, lags, count, stage)?, req.max_bytes)?;
        let mut workers = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("time_series_workers"),
        )?;
        for _ in 0..count {
            workers.push(LagScratch::new(samples, lags)?);
        }
        let stage = try_filled(
            stage,
            TimeSeriesLag::default(),
            scioxide_core::quantity!("time_series_stage"),
        )?;
        *self = Self {
            workers,
            stage,
            samples,
            lags,
        };
        if self.check(req)? != Capacity::Sufficient {
            return Err(SciOxideError::allocation_failure(
                scioxide_core::quantity!("time_series_workspace"),
                req.bytes,
            ));
        }
        Ok(())
    }
    /// Creates a workspace that fits `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::try_reserve`].
    pub fn workspace(req: &TimeSeriesRequirements<T>) -> Result<Self, SciOxideError> {
        let mut workspace = Self::default();
        workspace.try_reserve(req)?;
        Ok(workspace)
    }
    /// Returns a `ResourceLimit` error for the first buffer that is too small.
    fn ensure(&self, req: &TimeSeriesRequirements<T>) -> Result<(), SciOxideError> {
        if self.check(req)? == Capacity::Sufficient {
            return Ok(());
        }
        let stage = mul(req.lanes, req.lags)?;
        let (quantity, required, available) = if self.workers.len() < req.workers {
            (
                scioxide_core::quantity!("time_series_workers"),
                req.workers,
                self.workers.len(),
            )
        } else if self.lags < req.lags {
            (
                scioxide_core::quantity!("time_series_lags"),
                req.lags,
                self.lags,
            )
        } else if self.stage.len() < stage {
            (
                scioxide_core::quantity!("time_series_stage"),
                stage,
                self.stage.len(),
            )
        } else {
            (
                scioxide_core::quantity!("time_series_samples"),
                req.samples,
                self.samples,
            )
        };
        Err(SciOxideError::resource_limit(quantity, required, available))
    }
}
fn narrow<T: NumericScalar>(x: Lagged<T>) -> Result<T, SciOxideError> {
    let z = x.round()?;
    Ok(T::from_components(z.re, z.im))
}

/// Checks every position against the missing and non-finite policies and
/// returns the number of present observations. Skipped positions stay gaps.
fn admit<T: NumericScalar>(
    n: usize,
    value: impl Fn(usize) -> T,
    observed: impl Fn(usize) -> bool,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    let mut count = 0;
    for i in 0..n {
        if i % 256 == 0 {
            context.check()?;
        }
        if !observed(i) {
            if options.missing_policy() == MissingPolicy::Reject {
                return Err(invalid(
                    scioxide_core::argument!("validity"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing observation",
                ));
            }
        } else if !value(i).has_finite_components() {
            if options.non_finite_policy() == NonFinitePolicy::Reject {
                input_finite(value(i))?;
            }
        } else {
            count += 1;
        }
    }
    if count == 0 {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "no present observations",
        ));
    }
    Ok(count)
}

// One real lag-product traversal, instantiated over native or scaled arithmetic.
// Admission and centering precede this loop; scalar projection follows normalization.
#[expect(
    clippy::too_many_arguments,
    reason = "shared native/scaled lag traversal"
)]
fn real_products<T: NumericScalar, N: Arithmetic>(
    n: usize,
    present: impl Fn(usize) -> bool,
    deviation: impl Fn(usize) -> Result<N, SciOxideError>,
    mut encode: impl FnMut(usize, usize, N) -> Result<Scaled<f64>, SciOxideError>,
    count: usize,
    summary: TimeSeriesOptions,
    work: &mut LagScratch<T>,
    out: &mut [TimeSeriesLag<T>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    // The staged record holds exactly `max_lag + 1` lags.
    for (lag, staged) in out.iter_mut().enumerate() {
        context.check()?;
        let mut pairs = 0;
        let sum = N::product_sum(
            (lag..n)
                .filter(|&i| present(i) && present(i - lag))
                .map(|i| {
                    if i % 256 == 0 {
                        context.check()?;
                    }
                    pairs += 1;
                    Ok((deviation(i)?, deviation(i - lag)?))
                }),
        )?;
        if pairs == 0 {
            return Err(invalid(
                scioxide_core::argument!("max_lag"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "lag has no observed pairs",
            ));
        }
        let divisor = if summary.normalization == LagNormalization::PairCount {
            pairs
        } else {
            count
        };
        let divisor = N::number(number::<T::Real>(divisor)?.into_f64())?;
        let covariance = encode(lag, pairs, sum.div(divisor)?)?;
        work.covariance[lag] = ScaledValue::from_components(
            Scaled::from_parts(
                T::Real::try_from_f64_lossy(covariance.significand())?,
                covariance.exponent(),
            )?,
            Scaled::ZERO,
        );
        staged.pairs = pairs;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "record accessors, admitted count, lag specification and staging"
)]
fn real_covariance<T: NumericScalar>(
    n: usize,
    value: impl Fn(usize) -> T,
    present: impl Fn(usize) -> bool,
    count: usize,
    summary: TimeSeriesOptions,
    work: &mut LagScratch<T>,
    out: &mut [TimeSeriesLag<T>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let centered = summary.center == crate::TimeSeriesCentering::SampleMean;
    let mut origin = None;
    let mut native = true;
    let mut failure = None;
    let shifted = scioxide_core::compensated_reduce((0..n).scan((), |(), i| {
        if i % 256 == 0
            && let Err(error) = context.check()
        {
            failure = Some(error);
            return None;
        }
        if !present(i) {
            return Some([0.0; 2]);
        }
        let x = value(i).real_component().into_f64();
        if !crate::arithmetic::native(x, 64) {
            native = false;
            return None;
        }
        let origin = *origin.get_or_insert(x);
        Some(if centered { shift(x, origin) } else { [0.0; 2] })
    }));
    if let Some(error) = failure {
        return Err(error);
    }
    if native {
        let divisor = number::<T::Real>(count)?.into_f64();
        let center = if centered {
            crate::arithmetic::shifted_center(origin.unwrap_or(0.0), shifted?, divisor)?
        } else {
            [0.0; 2]
        };
        let mut variance = 0.0;
        return real_products(
            n,
            &present,
            |i| {
                Ok(normalized_deviation(
                    value(i).real_component().into_f64(),
                    center,
                ))
            },
            |lag, pairs, covariance: f64| {
                if lag == 0 {
                    variance = covariance;
                } else if variance > 0.0
                    && covariance.abs()
                        <= crate::arithmetic::PRODUCT_CANCELLATION * variance * count as f64
                            / if summary.normalization == LagNormalization::PairCount {
                                pairs as f64
                            } else {
                                count as f64
                            }
                {
                    // Lag-zero energy bounds the absolute pair-product sum.
                    // Severe cancellation needs the low centered parts and exact
                    // products, rather than a sum of already-rounded products.
                    let real = |i| value(i).real_component().into_f64();
                    let sum = crate::arithmetic::exact_centered_products(
                        (lag..n)
                            .filter(|&i| present(i) && present(i - lag))
                            .map(|i| (real(i), real(i - lag))),
                        [center, center],
                    )?;
                    let divisor = if summary.normalization == LagNormalization::PairCount {
                        pairs
                    } else {
                        count
                    };
                    return Scaled::from_value(sum / number::<T::Real>(divisor)?.into_f64());
                }
                Scaled::from_value(covariance)
            },
            count,
            summary,
            work,
            out,
            context,
        );
    }

    // Restart the complete center in extended range. No native state or rounded
    // scalar covariance is retained, and no sample-sized fallback buffer is needed.
    let scaled = |i: usize| Scaled::from_value(value(i).real_component().into_f64());
    let center = if centered {
        let first = (0..n).find(|&i| present(i)).unwrap_or(0);
        let divisor = Scaled::from_value(number::<T::Real>(count)?.into_f64())?;
        scaled_center(
            scaled(first)?,
            (0..n).filter(|&i| present(i)).map(scaled),
            divisor,
        )?
    } else {
        ScaledAccumulator::ZERO
    };
    real_products(
        n,
        present,
        |i| center_difference(center, ScaledAccumulator::from_scaled(scaled(i)?)),
        |_, _, covariance| Ok(covariance),
        count,
        summary,
        work,
        out,
        context,
    )
}

fn extended<T: NumericScalar>(value: T) -> Result<Lagged<T>, SciOxideError> {
    ScaledValue::from_value(Complex::new(
        value.real_component(),
        value.imaginary_component(),
    ))
}

#[expect(
    clippy::too_many_arguments,
    reason = "record accessors, admitted count, lag specification and staging"
)]
fn covariance<T: NumericScalar>(
    n: usize,
    value: impl Fn(usize) -> T,
    present: impl Fn(usize) -> bool,
    count: usize,
    summary: TimeSeriesOptions,
    work: &mut LagScratch<T>,
    out: &mut [TimeSeriesLag<T>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    if T::IS_REAL {
        return real_covariance(n, value, present, count, summary, work, out, context);
    }
    let mut re = ScaledAccumulator::ZERO;
    let mut im = ScaledAccumulator::ZERO;
    let mut origin = None;
    if summary.center == crate::TimeSeriesCentering::SampleMean {
        for i in (0..n).filter(|&i| present(i)) {
            let x = extended(value(i))?;
            let difference = x.sub(*origin.get_or_insert(x))?;
            re.add_scaled(difference.real())?;
            im.add_scaled(difference.imaginary())?;
        }
    }
    let origin = origin.unwrap_or(ScaledValue::ZERO);
    let divisor = Scaled::from_value(number(count)?)?;
    let mean = if summary.center == crate::TimeSeriesCentering::SampleMean {
        ScaledValue::from_components(re.total()?.div(divisor)?, im.total()?.div(divisor)?)
    } else {
        ScaledValue::ZERO
    };
    for i in 0..n {
        // Keep the mean offset separate: origin + mean need not be
        // representable even when every centered difference is exact.
        work.centered[i] = if present(i) {
            extended(value(i))?.sub(origin)?.sub(mean)?
        } else {
            ScaledValue::ZERO
        };
    }
    for (lag, staged) in out.iter_mut().enumerate() {
        context.check()?;
        let mut real = ScaledAccumulator::ZERO;
        let mut imag = ScaledAccumulator::ZERO;
        let mut pairs = 0;
        for i in lag..n {
            if present(i) && present(i - lag) {
                let y = work.centered[i - lag];
                let product = work.centered[i]
                    .mul(ScaledValue::from_components(y.real(), y.imaginary().neg()))?;
                real.add_scaled(product.real())?;
                imag.add_scaled(product.imaginary())?;
                pairs += 1;
            }
        }
        if pairs == 0 {
            return Err(invalid(
                scioxide_core::argument!("max_lag"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "lag has no observed pairs",
            ));
        }
        let div = if summary.normalization == LagNormalization::PairCount {
            Scaled::from_value(number(pairs)?)?
        } else {
            divisor
        };
        work.covariance[lag] =
            ScaledValue::from_components(real.total()?.div(div)?, imag.total()?.div(div)?);
        staged.pairs = pairs;
    }
    Ok(())
}
fn finish<T: NumericScalar>(
    statistic: TimeSeriesStatistic,
    max_lag: usize,
    w: &mut LagScratch<T>,
    out: &mut [TimeSeriesLag<T>],
) -> Result<(), SciOxideError> {
    if statistic == TimeSeriesStatistic::Covariance {
        for (staged, &c) in out.iter_mut().zip(&w.covariance) {
            staged.value = narrow(c)?;
        }
        return Ok(());
    }
    let variance = w.covariance[0];
    if variance.real().is_zero() || variance.real().is_sign_negative() {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "zero variance has undefined correlation",
        ));
    }
    for (staged, &c) in out.iter_mut().zip(&w.covariance) {
        staged.value = narrow(c.div(variance)?)?;
    }
    out[0].value = T::ONE;
    if statistic == TimeSeriesStatistic::Acf {
        return Ok(());
    }
    let mut error = T::Real::ONE;
    w.coefficients[..=max_lag].fill(T::ZERO);
    for (k, staged) in out.iter_mut().enumerate().skip(1) {
        let mut sum = extended(staged.value)?;
        for j in 1..k {
            sum = sum.sub(extended(w.coefficients[j])?.mul(w.covariance[k - j].div(variance)?)?)?;
        }
        let reflection: T = narrow(sum.div(ScaledValue::from_value(Complex::new(
            error,
            <_ as AlgebraicScalar>::ZERO,
        ))?)?)?;
        let norm = reflection.abs();
        if norm >= T::Real::ONE {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "PACF covariance is not positive definite",
            ));
        }
        w.previous[..=max_lag].copy_from_slice(&w.coefficients[..=max_lag]);
        for j in 1..k {
            w.coefficients[j] = finite(w.previous[j] - reflection * w.previous[k - j].conj())?;
        }
        w.coefficients[k] = reflection;
        staged.value = reflection;
        error = finite(error * (T::Real::ONE - norm) * (T::Real::ONE + norm))?;
        if error <= T::Real::EPSILON {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "PACF prediction variance is numerically unresolved",
            ));
        }
    }
    Ok(())
}
/// Checks, centers and summarizes one record.
#[expect(
    clippy::too_many_arguments,
    reason = "record accessors, request, options, scratch, staging and context"
)]
fn summarize<T: NumericScalar>(
    n: usize,
    value: impl Fn(usize) -> T,
    observed: impl Fn(usize) -> bool,
    statistic: TimeSeriesStatistic,
    summary: TimeSeriesOptions,
    options: StatisticsOptions,
    work: &mut LagScratch<T>,
    out: &mut [TimeSeriesLag<T>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let count = admit(n, &value, &observed, options, context)?;
    if count == n {
        // A complete record has no gaps; skip per-pair presence checks.
        covariance(n, &value, |_| true, count, summary, work, out, context)?;
    } else {
        // After admission every omitted position is masked or skipped as non-finite.
        let present = |i: usize| observed(i) && value(i).has_finite_components();
        covariance(n, &value, present, count, summary, work, out, context)?;
    }
    finish(statistic, out.len() - 1, work, out)
}
/// Computes the autocovariance, autocorrelation or partial autocorrelation of
/// one record at lags `0..=max_lag` into `output`.
///
/// [`TimeSeriesStatistic`] defines the statistics. `output` has one
/// [`TimeSeriesLag`] per lag. `validity`, if given, marks the present
/// observations. Masked observations follow `options.missing_policy()`, and
/// NaN or infinite values follow `options.non_finite_policy()`. Skipped
/// observations stay gaps on the time grid, so they remove pairs at every lag
/// without shifting later observations.
///
/// The cost is O(n L + L²) for `n` observations and `L = max_lag + 1` lags.
/// Real records use compensated summation, and lag sums with severe
/// cancellation are recomputed with exact products. Complex records, and real
/// values outside the normal `f64` exponent range, use extended-range
/// arithmetic.
///
/// # Errors
///
/// - `DimensionMismatch` if `validity` and `values` differ in length, or
///   `output.len()` is not `max_lag + 1`.
/// - `InvalidInput` if `values` is empty or `max_lag` is not less than its
///   length.
/// - `InvalidInput` if a masked observation meets [`MissingPolicy::Reject`], and
///   `NonFinite` if a NaN or infinity meets [`NonFinitePolicy::Reject`].
/// - `InvalidInput` if no observation is present or some lag has no present
///   pair.
/// - `InvalidInput` if the autocorrelation or partial autocorrelation is
///   requested and `γ̂(0) = 0`.
/// - `InvalidInput` if the Levinson–Durbin recursion finds a partial
///   autocorrelation of modulus 1 or more, or a prediction error variance
///   below `ε`.
/// - `ResourceLimit` if `workspace` is too small for this request or holds more
///   than `options.max_bytes()`.
/// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_stats::{
///     StatisticsOptions, TimeSeriesLag, TimeSeriesOptions, TimeSeriesStatistic, TimeSeriesWorkspace,
///     time_series_summary_into,
/// };
///
/// // Lag-1 autocorrelation of 1, 2, 3, 4: (5/16) / (5/4) = 1/4, from 3 pairs.
/// let values = [1.0_f64, 2.0, 3.0, 4.0];
/// let (summary, options) = (TimeSeriesOptions::new(), StatisticsOptions::new());
/// let req = TimeSeriesWorkspace::scalar_requirements(values.len(), 1, options)?;
/// let mut work = TimeSeriesWorkspace::workspace(&req)?;
/// let mut output = [TimeSeriesLag::default(); 2];
/// let (acf, context) = (TimeSeriesStatistic::Acf, ExecutionContext::sequential());
/// time_series_summary_into(&values, None, acf, 1, summary, options, &mut output, &mut work, context)?;
/// assert!((output[1].value - 0.25).abs() < 1e-14);
/// assert_eq!(output[1].pairs, 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[expect(
    clippy::too_many_arguments,
    reason = "record, mask, request, lag specification, options, output, workspace and context"
)]
pub fn time_series_summary_into<T: NumericScalar>(
    values: &[T],
    validity: Option<&[bool]>,
    statistic: TimeSeriesStatistic,
    max_lag: usize,
    summary: TimeSeriesOptions,
    options: StatisticsOptions,
    output: &mut [TimeSeriesLag<T>],
    workspace: &mut TimeSeriesWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if let Some(mask) = validity
        && mask.len() != values.len()
    {
        return Err(SciOxideError::dimension_mismatch(
            [values.len()],
            [mask.len()],
        ));
    }
    let req = TimeSeriesWorkspace::scalar_requirements(values.len(), max_lag, options)?;
    if output.len() != req.lags {
        return Err(SciOxideError::dimension_mismatch(
            [req.lags],
            [output.len()],
        ));
    }
    workspace.ensure(&req)?;
    let stage = &mut workspace.stage[..req.lags];
    summarize(
        values.len(),
        |i| values[i],
        |i| validity.is_none_or(|mask| mask[i]),
        statistic,
        summary,
        options,
        &mut workspace.workers[0],
        stage,
        context,
    )?;
    context.check()?;
    output.copy_from_slice(stage);
    Ok(())
}
/// Computes a time-series statistic of one record and returns it as a vector
/// of `max_lag + 1` lags.
///
/// This is [`time_series_summary_into`] with its own workspace, run
/// sequentially on the calling thread. `options.max_bytes()` also counts the
/// returned vector.
///
/// # Errors
///
/// Returns the errors of [`time_series_summary_into`].
///
/// # Examples
///
/// ```
/// use scioxide_stats::{
///     MissingPolicy, StatisticsOptions, TimeSeriesOptions, TimeSeriesStatistic, time_series_summary_with_options,
/// };
///
/// // The masked value stays a gap: lag 1 keeps only the pairs (0, 1) and (3, 4).
/// let values = [1.0_f64, 2.0, 0.0, 3.0, 4.0];
/// let mask = [true, true, false, true, true];
/// let options = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
/// let (covariance, summary) = (TimeSeriesStatistic::Covariance, TimeSeriesOptions::new());
/// let lags = time_series_summary_with_options(&values, Some(&mask), covariance, 1, summary, options)?;
/// assert_eq!((lags[0].pairs, lags[1].pairs), (4, 2));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn time_series_summary_with_options<T: NumericScalar>(
    values: &[T],
    validity: Option<&[bool]>,
    statistic: TimeSeriesStatistic,
    max_lag: usize,
    summary: TimeSeriesOptions,
    options: StatisticsOptions,
) -> Result<Vec<TimeSeriesLag<T>>, SciOxideError> {
    let req = TimeSeriesWorkspace::<T>::scalar_requirements(values.len(), max_lag, options)?;
    let output_bytes = mul(req.lags, size_of::<TimeSeriesLag<T>>())?;
    limit_bytes(add(req.bytes, output_bytes)?, options.max_bytes())?;
    let mut workspace = TimeSeriesWorkspace::workspace(&req)?;
    let mut output = try_filled(
        req.lags,
        TimeSeriesLag::default(),
        scioxide_core::quantity!("time_series_output"),
    )?;
    time_series_summary_into(
        values,
        validity,
        statistic,
        max_lag,
        summary,
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Computes a time-series statistic of every record of `view` into `output`.
///
/// The sample axis of `view` is time, and each batch is a separate record.
/// `output` holds `max_lag + 1` lags per record, record after record. Each
/// record is treated as in [`time_series_summary_into`]. Records may run in
/// parallel, as the execution policy of `options` and `context` allow; the
/// result does not depend on the number of workers. See
/// [Data, weights and missing values](crate#data-weights-and-missing-values)
/// for the layout of `view` and its validity mask, and [`TimeSeriesWorkspace`]
/// for an example.
///
/// # Errors
///
/// - `InvalidInput` if `view` has vector observations.
/// - `DimensionMismatch` if `output.len()` is not the number of records times
///   `max_lag + 1`.
/// - The errors of [`time_series_summary_into`] for each record.
#[allow(clippy::too_many_arguments)]
pub fn time_series_summaries_into<T: NumericScalar>(
    view: BatchView<'_, T>,
    statistic: TimeSeriesStatistic,
    max_lag: usize,
    summary: TimeSeriesOptions,
    options: StatisticsOptions,
    output: &mut [TimeSeriesLag<T>],
    workspace: &mut TimeSeriesWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = TimeSeriesWorkspace::workspace_requirements(view, max_lag, options, context)?;
    let len = mul(req.lanes, req.lags)?;
    if output.len() != len {
        return Err(SciOxideError::dimension_mismatch([len], [output.len()]));
    }
    workspace.ensure(&req)?;
    let n = view.samples();
    crate::estimation::execute(
        options,
        context,
        req.lanes,
        req.lags,
        &mut workspace.workers,
        &mut workspace.stage[..len],
        |lane, work, out| {
            let origin = view.batch_origin(lane);
            let mask = view.validity.map(|mask| &mask[lane * n..(lane + 1) * n]);
            summarize(
                n,
                |j| view.value_at_origin(origin, j, 0),
                |j| mask.is_none_or(|mask| mask[j]),
                statistic,
                summary,
                options,
                work,
                out,
                context,
            )
        },
    )?;
    context.check()?;
    output.copy_from_slice(&workspace.stage[..len]);
    Ok(())
}

/// Computes a time-series statistic at lags zero through `max_lag`.
///
/// Uses the sample mean and observation-count normalization. Cost and accuracy
/// follow [`time_series_summary_with_options`].
///
/// # Errors
///
/// Returns the configured form's errors for the [one-shot defaults](crate#one-shot-forms).
pub fn time_series_summary<T: NumericScalar>(
    values: &[T],
    statistic: TimeSeriesStatistic,
    max_lag: usize,
) -> Result<Vec<TimeSeriesLag<T>>, SciOxideError> {
    time_series_summary_with_options(
        values,
        None,
        statistic,
        max_lag,
        TimeSeriesOptions::default(),
        StatisticsOptions::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options", "summary", "validity"],
            scioxide_core::argument!("values"),
        )
    })
}
