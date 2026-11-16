//! Z-scores, geometric z-scores, O'Brien's transform and sigma clipping.
use crate::StatsTermination;
use crate::arithmetic::Compensated;
use crate::arithmetic::normalized_deviation as native_deviation;
use crate::arithmetic::{center_difference, promote, scaled_center, shift, shifted_center};
use crate::error_helpers::{invalid, sample};
use crate::{MomentOrder, OnlineMoments};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AlgebraicScalar, Argument, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

/// Reusable workspace for [`Self::standardize_into`], which computes z-scores
/// of real or complex data against a reference sample.
///
/// The mean `x̄` and the sum of squares use compensated summation. The mean is
/// kept as the unevaluated sum of two `f64` values, and the differences
/// `xᵢ − x̄` use both parts. Values with a nonzero
/// component outside `2⁻⁶⁴ ≤ |x| < 2⁶⁵` are handled in extended-exponent
/// arithmetic, so intermediate results do not overflow or underflow.
pub struct StandardizationWorkspace<T: NumericScalar> {
    stage: Vec<T>,
}
impl<T: NumericScalar> Default for StandardizationWorkspace<T> {
    fn default() -> Self {
        Self { stage: Vec::new() }
    }
}
impl<T: NumericScalar> StandardizationWorkspace<T> {
    /// Grows the staging buffer to `output` values. On error the existing
    /// buffer is kept.
    /// # Errors
    /// Returns size-overflow and allocation errors.
    pub(crate) fn reserve(&mut self, output: usize) -> Result<(), SciOxideError> {
        if self.stage.len() < output {
            self.stage = try_filled(
                output,
                T::ZERO,
                scioxide_core::quantity!("standardization_stage"),
            )?;
        }
        Ok(())
    }
    /// Computes the z-scores of `scores` against a reference sample.
    ///
    /// `output[i] = (scores[i] − x̄) / s`, where `x̄` and `s` are the mean and
    /// the standard deviation with divisor `n − ddof` of `reference`, and `n`
    /// is `reference.len()`. For complex data, `s² = Σ |xⱼ − x̄|² / (n − ddof)`,
    /// and the real and imaginary parts of `scores[i] − x̄` are both divided
    /// by `s`. Passing the same slice as `scores` and `reference` gives the
    /// ordinary z-scores of a sample.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output.len()` differs from `scores.len()`, or the
    ///   workspace holds fewer than `scores.len()` values.
    /// - `InvalidInput` if `n ≤ ddof`, the reference has zero standard
    ///   deviation, or a reference value or score is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a z-score overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_stats::{StandardizationWorkspace, StatisticsOptions};
    ///
    /// // Mean 2 and population standard deviation 1 give z-scores −1 and 1.
    /// let req = StandardizationWorkspace::workspace_requirements(2, StatisticsOptions::new())?;
    /// let mut workspace = StandardizationWorkspace::default();
    /// workspace.try_reserve(&req)?;
    /// let mut output = [0.0_f64; 2];
    /// workspace.standardize_into(&[1.0, 3.0], &[1.0, 3.0], 0, &mut output)?;
    /// assert_eq!(output, [-1.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn standardize_into(
        &mut self,
        scores: &[T],
        reference: &[T],
        ddof: usize,
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        if scores.len() != output.len() || self.stage.len() < scores.len() {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "standardization storage mismatch",
            ));
        }
        degrees(reference.len(), ddof)?;
        let locations = Locations::new(reference, scioxide_core::argument!("reference"))?;
        let sd = locations.variance(reference, ddof)?.sqrt()?;
        if sd.is_zero() {
            return Err(invalid(
                scioxide_core::argument!("reference"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "zero reference standard deviation",
            ));
        }
        let stage = &mut self.stage[..scores.len()];
        if let Locations::Native(centers) = &locations
            && native_standardize(scores, centers, sd.round()?, stage)?
        {
            output.copy_from_slice(stage);
            return Ok(());
        }
        let [re, im] = locations.extended()?;
        for (slot, &x) in stage.iter_mut().zip(scores) {
            if !x.has_finite_components() {
                return Err(invalid(
                    scioxide_core::argument!("scores"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "nonfinite score",
                ));
            }
            *slot = T::from_components(
                re.deviation(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(x.real_component()),
                )?)?
                .div(sd)?
                .round_to()?,
                if T::IS_REAL {
                    T::Real::ZERO
                } else {
                    im.deviation(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(x.imaginary_component()),
                    )?)?
                    .div(sd)?
                    .round_to()?
                },
            );
        }
        output.copy_from_slice(&self.stage[..scores.len()]);
        Ok(())
    }
}

/// Workspace size for [`StandardizationWorkspace`], returned by
/// [`StandardizationWorkspace::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct StandardizationRequirements<T: NumericScalar> {
    scalar: core::marker::PhantomData<T>,
    output: usize,
    max_bytes: usize,
}
impl<T: NumericScalar> StandardizationRequirements<T> {
    /// Returns the heap size in bytes of the workspace, not counting the
    /// output slice.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(self) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[(self.output, core::mem::size_of::<T>())])
    }
}
impl<T: NumericScalar> StandardizationWorkspace<T> {
    /// Returns the workspace size for computing `output` z-scores at a time.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of `options`.
    pub fn workspace_requirements(
        output: usize,
        options: crate::StatisticsOptions,
    ) -> Result<StandardizationRequirements<T>, SciOxideError> {
        let req = StandardizationRequirements {
            scalar: core::marker::PhantomData,
            output,
            max_bytes: options.max_bytes(),
        };
        crate::error_helpers::limit_bytes(req.required_bytes()?, req.max_bytes)?;
        Ok(req)
    }
    /// Returns whether the workspace is large enough for `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage the workspace holds exceeds the byte
    ///   limit of `req`.
    pub fn check(
        &self,
        req: &StandardizationRequirements<T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        crate::error_helpers::limit_bytes(
            crate::estimation::storage_bytes(&[(
                self.stage.capacity(),
                core::mem::size_of::<T>(),
            )])?,
            req.max_bytes,
        )?;
        Ok(if self.stage.len() >= req.output {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to satisfy `req`. A workspace that is already large
    /// enough is left as it is.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the held or required storage exceeds the byte
    ///   limit of `req`.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if allocation fails.
    pub fn try_reserve(
        &mut self,
        req: &StandardizationRequirements<T>,
    ) -> Result<(), SciOxideError> {
        if self.check(req)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        crate::error_helpers::limit_bytes(req.required_bytes()?, req.max_bytes)?;
        self.reserve(req.output)
    }
    /// Creates a workspace that satisfies `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::try_reserve`].
    pub fn workspace(req: &StandardizationRequirements<T>) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
}
fn degrees(n: usize, ddof: usize) -> Result<(), SciOxideError> {
    if n <= ddof {
        Err(invalid(
            scioxide_core::argument!("ddof"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient reference degrees of freedom",
        ))
    } else {
        Ok(())
    }
}

fn components<T: NumericScalar>(value: T) -> [f64; 2] {
    [
        value.real_component().into_f64(),
        value.imaginary_component().into_f64(),
    ]
}

// Native bounds apply to source components, not rounded means. Even adjacent
// binary64 differences, their squares and usize-count sums stay normal. Keep
// the offset of a large origin until after subtraction from each observation.
pub(crate) enum Locations {
    // Real/imaginary centers, each retaining an origin and separate offset.
    Native([[f64; 2]; 2]),
    Extended([CenteredLocation; 2]),
}

impl Locations {
    pub(crate) fn new<T: NumericScalar>(
        values: &[T],
        argument: Argument,
    ) -> Result<Self, SciOxideError> {
        let first = values.first().ok_or_else(|| {
            invalid(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty reference",
            )
        })?;
        let origin = components(*first);
        let count = if T::IS_REAL { 1 } else { 2 };
        // Aggregate admission independently of the arithmetic dependency chains.
        let native = values.iter().fold(true, |native, &value| {
            let x = components(value);
            native
                & crate::arithmetic::native(x[0], 64)
                & (T::IS_REAL | crate::arithmetic::native(x[1], 64))
        });
        if native {
            let shifted = if T::IS_REAL {
                scioxide_core::compensated_reduce(
                    values
                        .iter()
                        .map(|&value| shift(components(value)[0], origin[0])),
                )
                .map(|[hi, lo]| {
                    [
                        hi,
                        lo,
                        scioxide_core::CompensatedSum::ZERO,
                        scioxide_core::CompensatedSum::ZERO,
                    ]
                })
            } else {
                scioxide_core::compensated_reduce(values.iter().map(|&value| {
                    let x = components(value);
                    let ([rh, rl], [ih, il]) = (shift(x[0], origin[0]), shift(x[1], origin[1]));
                    [rh, rl, ih, il]
                }))
            };
            if let Ok([rh, rl, ih, il]) = shifted {
                let mut centers = [[0.0; 2]; 2];
                for ((center, origin), parts) in centers
                    .iter_mut()
                    .zip(origin)
                    .zip([[rh, rl], [ih, il]])
                    .take(count)
                {
                    *center = shifted_center(origin, parts, values.len() as f64)?;
                }
                return Ok(Self::Native(centers));
            }
        }
        if values.iter().any(|x| !x.has_finite_components()) {
            return Err(invalid(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "nonfinite reference",
            ));
        }
        let re = centered_location(
            values
                .iter()
                .map(|x| Scaled::from_value(x.real_component().into_f64())),
            values.len(),
            argument,
        )?;
        let im = if T::IS_REAL {
            CenteredLocation(scioxide_core::ScaledAccumulator::ZERO)
        } else {
            centered_location(
                values
                    .iter()
                    .map(|x| Scaled::from_value(x.imaginary_component().into_f64())),
                values.len(),
                argument,
            )?
        };
        Ok(Self::Extended([re, im]))
    }

    pub(crate) fn extended(&self) -> Result<[CenteredLocation; 2], SciOxideError> {
        match self {
            Self::Extended(centers) => Ok(*centers),
            Self::Native(centers) => Ok([
                CenteredLocation(promote(centers[0])?),
                CenteredLocation(promote(centers[1])?),
            ]),
        }
    }

    pub(crate) fn variance<T: NumericScalar>(
        &self,
        values: &[T],
        ddof: usize,
    ) -> Result<Scaled<f64>, SciOxideError> {
        degrees(values.len(), ddof)?;
        match self {
            Self::Native(centers) => {
                let squares = scioxide_core::compensated_reduce(values.iter().map(|&value| {
                    let x = components(value);
                    let re = native_deviation(x[0], centers[0]);
                    let im = if T::IS_REAL {
                        0.0
                    } else {
                        native_deviation(x[1], centers[1])
                    };
                    [re * re + im * im]
                }));
                let Ok([squares]) = squares else {
                    return Self::Extended(self.extended()?).variance(values, ddof);
                };
                Scaled::from_value(squares.total()? / (values.len() - ddof) as f64)
            }
            Self::Extended([re, im]) => {
                let mut squares = Compensated::<Scaled<f64>>::zero();
                for &value in values {
                    let real = re
                        .deviation(Scaled::from_value(value.real_component().into_f64())?)?
                        .square()?;
                    let square = if T::IS_REAL {
                        real
                    } else {
                        real.add(
                            im.deviation(Scaled::from_value(
                                value.imaginary_component().into_f64(),
                            )?)?
                            .square()?,
                        )?
                    };
                    squares = squares.added(square)?;
                }
                squares
                    .total()?
                    .div(Scaled::from_value((values.len() - ddof) as f64)?)
            }
        }
    }
}

// This component kernel is used twice per complex observation. Keep the checked
// projection in the loop instead of passing a large Result through two calls.
#[inline(always)]
fn native_project<S: RealScalar>(
    value: f64,
    center: [f64; 2],
    sd: f64,
) -> Result<S, SciOxideError> {
    let value = native_deviation(value, center) / sd;
    S::try_from_f64_lossy(value).map_err(|_| {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "final component exceeds the IEEE finite range",
        )
    })
}

fn native_standardize<T: NumericScalar>(
    scores: &[T],
    centers: &[[f64; 2]; 2],
    sd: f64,
    stage: &mut [T],
) -> Result<bool, SciOxideError> {
    for (slot, &value) in stage.iter_mut().zip(scores) {
        let x = components(value);
        if !crate::arithmetic::native(x[0], 64)
            || (!T::IS_REAL && !crate::arithmetic::native(x[1], 64))
        {
            return Ok(false);
        }
        *slot = T::from_components(
            native_project(x[0], centers[0], sd)?,
            if T::IS_REAL {
                T::Real::ZERO
            } else {
                native_project(x[1], centers[1], sd)?
            },
        );
    }
    Ok(true)
}

/// A reference mean in extended-exponent arithmetic, kept as two parts.
#[derive(Clone, Copy)]
pub(crate) struct CenteredLocation(scioxide_core::ScaledAccumulator<f64>);
impl CenteredLocation {
    pub(crate) fn value(&self) -> Result<Scaled<f64>, SciOxideError> {
        self.0.total()
    }
    fn deviation(&self, x: Scaled<f64>) -> Result<Scaled<f64>, SciOxideError> {
        center_difference(self.0, scioxide_core::ScaledAccumulator::from_scaled(x))
    }
}
fn centered_location(
    values: impl Iterator<Item = Result<Scaled<f64>, SciOxideError>> + Clone,
    n: usize,
    argument: Argument,
) -> Result<CenteredLocation, SciOxideError> {
    let origin = values.clone().next().ok_or_else(|| {
        invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty reference",
        )
    })??;
    scaled_center(origin, values, Scaled::<f64>::from_value(n as f64)?).map(CenteredLocation)
}
/// Returns the z-scores of `scores` against a reference sample in a new vector.
///
/// The definition and accuracy are those of
/// [`StandardizationWorkspace::standardize_into`]. Pass the same slice as
/// `scores` and `reference` for the ordinary z-scores of a sample.
///
/// # Errors
///
/// - `InvalidInput` if `reference.len() ≤ ddof`, the reference has zero
///   standard deviation, or a reference value or score is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a z-score overflows.
/// - `AllocationFailure` if the output cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_stats::standardize;
///
/// // [1, 2, 3] has mean 2 and sample standard deviation 1 (ddof = 1).
/// let z = standardize(&[1.0_f64, 2.0, 3.0], &[1.0, 2.0, 3.0], 1)?;
/// assert!((z[0] + 1.0).abs() < 1e-15 && z[1].abs() < 1e-15 && (z[2] - 1.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn standardize<T: NumericScalar>(
    scores: &[T],
    reference: &[T],
    ddof: usize,
) -> Result<Vec<T>, SciOxideError> {
    let mut workspace = StandardizationWorkspace::default();
    workspace.reserve(scores.len())?;
    let mut output = try_filled(
        scores.len(),
        T::ZERO,
        scioxide_core::quantity!("standardization_output"),
    )?;
    workspace.standardize_into(scores, reference, ddof, &mut output)?;
    Ok(output)
}
/// Options for sigma clipping with
/// [`sigma_clip_into`](crate::sigma_clip_into).
///
/// A pass keeps the observations `x` with `x̄ − low·s ≤ x ≤ x̄ + high·s`, where
/// `x̄` and `s` are the mean and the population standard deviation of the
/// observations kept so far. The values are checked when clipping runs:
/// `low` and `high` must be finite and nonnegative.
#[derive(Clone, Copy, Debug)]
pub struct SigmaClipOptions<S: RealScalar> {
    scalar: core::marker::PhantomData<S>,
    /// Maximum number of passes.
    pub(crate) max_iterations: usize,
}
/// Result of sigma clipping: the number of kept observations, the number of
/// passes, the last bounds and the termination status.
///
/// When clipping stops before it converges, the kept observations are those
/// after the last completed pass and remain usable.
#[derive(Clone, Copy, Debug)]
pub struct SigmaClipResult<S: RealScalar> {
    retained: usize,
    iterations: usize,
    lower: Option<S>,
    upper: Option<S>,
    termination: StatsTermination,
}
impl<S: RealScalar> SigmaClipResult<S> {
    /// Returns the number of kept observations, which fill the front of the
    /// output in input order.
    pub const fn retained(&self) -> usize {
        self.retained
    }
    /// Returns the number of completed passes.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the lower bound `x̄ − low·s` of the last pass, or `None` if no
    /// pass ran or the bound overflows `S`.
    pub const fn lower(&self) -> Option<S> {
        self.lower
    }
    /// Returns the upper bound `x̄ + high·s` of the last pass, or `None` if no
    /// pass ran or the bound overflows `S`.
    pub const fn upper(&self) -> Option<S> {
        self.upper
    }
    /// Returns why clipping stopped.
    pub const fn termination(&self) -> StatsTermination {
        self.termination
    }
}
/// Geometric z-scores; see `StatisticsWorkspace::geometric_zscore_into`.
pub(crate) fn geometric_zscore_into<S: RealScalar>(
    values: &[S],
    ddof: usize,
    output: &mut [S],
    work: &mut [S],
    stage: &mut [S],
) -> Result<(), SciOxideError> {
    if work.len() < values.len() || output.len() != values.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "geometric zscore storage mismatch",
        ));
    }
    sample(values, scioxide_core::argument!("values"))?;
    let mut moments = OnlineMoments::new(MomentOrder::Second);
    for (slot, &x) in work.iter_mut().zip(values) {
        if x <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "geometric zscore requires positive observations",
            ));
        }
        *slot = crate::error_helpers::number(crate::arithmetic::log_ratio(x, values[0])?)?;
        moments.observe(*slot)?;
    }
    let snapshot = moments.snapshot();
    let mean = snapshot.mean()?.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty logarithmic sample",
        )
    })?;
    let sd = snapshot.standard_deviation(ddof)?.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("ddof"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient degrees of freedom",
        )
    })?;
    if sd == S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "zero logarithmic deviation",
        ));
    }
    for (slot, &x) in stage.iter_mut().zip(&work[..values.len()]) {
        *slot = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?
            .sub(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(mean),
            )?)?
            .div(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(sd),
            )?)?
            .round_to()?;
    }
    output.copy_from_slice(&stage[..values.len()]);
    Ok(())
}

/// O'Brien's transform; see `StatisticsWorkspace::obrien_transform_into`.
pub(crate) fn obrien_transform_into<S: RealScalar>(
    values: &[S],
    output: &mut [S],
    stage: &mut [S],
) -> Result<(), SciOxideError> {
    sample(values, scioxide_core::argument!("values"))?;
    let n = values.len();
    if n < 3 || stage.len() < n || output.len() != n {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "O'Brien requires n>=3 and matching storage",
        ));
    }
    let mean = centered_location(
        values
            .iter()
            .map(|&x| Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))),
        n,
        scioxide_core::argument!("values"),
    )?;
    let mut sumsq = Compensated::<Scaled<f64>>::zero();
    for &x in values {
        sumsq = sumsq.added(
            mean.deviation(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(x),
            )?)?
            .square()?,
        )?;
    }
    let variance = sumsq
        .total()?
        .div(Scaled::<f64>::from_value((n - 1) as f64)?)?;
    let factor = Scaled::<f64>::from_value((n) as f64)?
        .mul(Scaled::<f64>::from_value((n) as f64)?.sub(Scaled::<f64>::from_value(1.5)?)?)?;
    let offset = Scaled::<f64>::from_value((n - 1) as f64)?
        .mul(variance)?
        .mul(Scaled::<f64>::from_value(0.5)?)?;
    let divisor = Scaled::<f64>::from_value((n - 1) as f64)?
        .mul(Scaled::<f64>::from_value((n - 2) as f64)?)?;
    for (slot, &x) in stage.iter_mut().zip(values) {
        *slot = mean
            .deviation(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(x),
            )?)?
            .square()?
            .mul(factor)?
            .sub(offset)?
            .div(divisor)?
            .round_to()?;
    }
    output.copy_from_slice(&stage[..n]);
    Ok(())
}

/// Clips a sample using caller scratch and a live child context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sigma_clip_into<S: RealScalar>(
    values: &[S],
    low: S,
    high: S,
    options: SigmaClipOptions<S>,
    context: &mut SolveContext<'_, '_>,
    output: &mut [S],
    indices: &mut [usize],
    stage: &mut [S],
) -> Result<SigmaClipResult<S>, SciOxideError> {
    sample(values, scioxide_core::argument!("values"))?;
    if stage.len() < values.len() || output.len() < values.len() || indices.len() < values.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient clipping storage",
        ));
    }
    if !low.is_finite() || !high.is_finite() || low < S::ZERO || high < S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "clipping multipliers must be finite and nonnegative",
        ));
    }
    // A staged 0/1 mask preserves exact original indices without converting to S.
    stage[..values.len()].fill(S::ONE);
    let mut report = SigmaClipResult {
        retained: values.len(),
        iterations: 0,
        lower: None,
        upper: None,
        termination: StatsTermination::BudgetExhausted,
    };
    while report.iterations < options.max_iterations {
        if let Some(stop) = context.stop_reason()? {
            report.termination = StatsTermination::Stopped(stop);
            break;
        }
        if !context.record_iteration()?.is_granted() {
            break;
        }
        let mut m = OnlineMoments::new(MomentOrder::Second);
        for (&x, &keep) in values.iter().zip(stage.iter()) {
            if keep == S::ONE {
                m.observe(x)?;
            }
        }
        let s = m.snapshot();
        let mean = s.mean()?.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "clipping removed all observations",
            )
        })?;
        let sd = s.standard_deviation(0)?.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty clipping sample",
            )
        })?;
        let low = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(mean))?.sub(
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(low))?.mul(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(sd))?,
            )?,
        )?;
        let high = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(mean))?.add(
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(high))?.mul(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(sd))?,
            )?,
        )?;
        report.lower = low.try_value_to().ok();
        report.upper = high.try_value_to().ok();
        let before = report.retained;
        for (&x, keep) in values.iter().zip(stage.iter_mut()) {
            if *keep == S::ONE
                && (Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?
                    .sub(low)?
                    .is_sign_negative()
                    || high
                        .sub(Scaled::<f64>::from_value(
                            scioxide_core::RealScalar::into_f64(x),
                        )?)?
                        .is_sign_negative())
            {
                *keep = S::ZERO;
                report.retained -= 1;
            }
        }
        report.iterations += 1;
        if report.retained == 0 {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "clipping removed all observations",
            ));
        }
        if report.retained == before {
            report.termination = StatsTermination::Converged;
            break;
        }
    }
    let retained = values
        .iter()
        .zip(stage.iter())
        .enumerate()
        .filter(|(_, (_, keep))| **keep == S::ONE);
    for (dest, (source, (&x, _))) in retained.enumerate() {
        output[dest] = x;
        indices[dest] = source;
    }
    Ok(report)
}

impl<S: RealScalar> SigmaClipOptions<S> {
    /// Returns a limit of 100 clipping passes.
    pub const fn new() -> Self {
        Self {
            scalar: core::marker::PhantomData,
            max_iterations: 100,
        }
    }

    /// Sets the maximum number of passes.
    pub const fn with_max_iterations(mut self, value: usize) -> Self {
        self.max_iterations = value;
        self
    }
    /// Returns the maximum number of passes.
    pub const fn max_iterations(self) -> usize {
        self.max_iterations
    }
}

impl<S: RealScalar> Default for SigmaClipOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
