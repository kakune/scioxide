use super::*;
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, ExecutionContext};
use scioxide_signal::{SignalLayout, SignalWorkspace, SpectralEstimatePlan, SpectralScaling};

/// One frequency bin of the periodogram from [`periodogram_diagnostics_into`].
///
/// Bins come in the frequency order of the spectral estimate plan.
#[derive(Clone, Copy, Debug)]
pub struct PeriodogramBin<S: RealScalar> {
    /// Frequency in cycles per time unit of the plan's sample rate.
    pub frequency: S,
    /// Power spectral density, in squared sample units per frequency unit.
    pub density: S,
    /// Fraction of the total power in this bin and all bins before it.
    /// `None` when the total power is zero.
    pub cumulative_power: Option<S>,
}
impl<S: RealScalar> Default for PeriodogramBin<S> {
    fn default() -> Self {
        Self {
            frequency: S::ZERO,
            density: S::ZERO,
            cumulative_power: None,
        }
    }
}
/// Summary of a periodogram from [`periodogram_diagnostics_into`].
///
/// These are descriptive values. They do not test for stationarity or for the
/// significance of a peak.
#[derive(Clone, Copy, Debug)]
pub struct PeriodogramDiagnostics<S: RealScalar> {
    /// Total power `Σ density × bin_width` over all bins.
    pub integrated_power: S,
    /// Frequency of the largest density; the first bin wins a tie. `None` when
    /// the total power is zero.
    pub peak_frequency: Option<S>,
    /// Bin spacing `fs / N` in cycles per time unit, for sample rate `fs` and
    /// FFT length `N`.
    pub bin_width: S,
}
/// Reusable workspace for [`periodogram_diagnostics_into`], sized for one
/// spectral estimate plan.
///
/// See [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct PeriodogramWorkspace<T: NumericScalar> {
    signal: SignalWorkspace<T>,
    density: Vec<T::Real>,
    stage: Vec<PeriodogramBin<T::Real>>,
}
impl<T: NumericScalar> PeriodogramWorkspace<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Returns the number of bytes the workspace needs for `plan`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(plan: &SpectralEstimatePlan<T>) -> Result<usize, SciOxideError> {
        add(
            plan.workspace_requirements(SignalLayout::Contiguous(plan.segment_len()))?
                .bytes::<T>()?,
            mul(
                plan.bins(),
                add(
                    core::mem::size_of::<T::Real>(),
                    core::mem::size_of::<PeriodogramBin<T::Real>>(),
                )?,
            )?,
        )
    }
    /// Creates a workspace for `plan`.
    ///
    /// A periodogram needs a complete, finite record on a uniform grid. So
    /// `options` must reject missing and non-finite values; fill or remove gaps
    /// before estimating a spectrum. Only the policies and the memory limit of
    /// `options` are used.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the missing-value or non-finite policy of `options`
    ///   is not `Reject`.
    /// - `ResourceLimit` if the workspace exceeds `options.max_bytes()`.
    pub fn new(
        plan: &SpectralEstimatePlan<T>,
        options: crate::StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        if options.missing_policy() != crate::MissingPolicy::Reject
            || options.non_finite_policy() != crate::NonFinitePolicy::Reject
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "periodograms require a complete finite record",
            ));
        }
        crate::error_helpers::limit_bytes(Self::required_bytes(plan)?, options.max_bytes())?;
        let mut signal = SignalWorkspace::new();
        signal.try_reserve(
            &plan.workspace_requirements(SignalLayout::Contiguous(plan.segment_len()))?,
        )?;
        Ok(Self {
            signal,
            density: try_filled(
                plan.bins(),
                T::Real::ZERO,
                scioxide_core::quantity!("periodogram_density"),
            )?,
            stage: try_filled(
                plan.bins(),
                PeriodogramBin::default(),
                scioxide_core::quantity!("periodogram_stage"),
            )?,
        })
    }
}
/// Computes the periodogram of a uniformly sampled record into `output` and
/// returns its total power and peak frequency.
///
/// `plan` fixes the sample rate, window, detrending and frequency order, and
/// must use [`SpectralScaling::Density`](scioxide_signal::SpectralScaling::Density).
/// `values` must hold `plan.segment_len()` samples with no gaps. Removing
/// missing samples would shift the later ones on the time grid, so fill gaps
/// before the call. `output` receives one [`PeriodogramBin`] per frequency.
///
/// # Errors
///
/// - `InvalidInput` if `values.len()` differs from `plan.segment_len()`,
///   `output.len()` differs from `plan.bins()`, `workspace` was created for a
///   plan with a different number of bins, or `plan` does not use density
///   scaling.
/// - `NonFinite` if `values` contains NaN or infinity.
/// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{SpectralEstimateOptions, SpectralEstimatePlan};
/// use scioxide_stats::{
///     PeriodogramBin, PeriodogramWorkspace, StatisticsOptions, periodogram_diagnostics_into,
/// };
///
/// // An alternating sequence has unit power, all at the Nyquist frequency 1/2.
/// let values = [1.0_f64, -1.0, 1.0, -1.0];
/// let context = ExecutionContext::sequential();
/// let plan = SpectralEstimatePlan::build(&[1.0; 4], SpectralEstimateOptions::new(), context)?;
/// let mut workspace = PeriodogramWorkspace::new(&plan, StatisticsOptions::new())?;
/// let mut bins = vec![PeriodogramBin::default(); plan.bins()];
/// let report = periodogram_diagnostics_into(&plan, &values, &mut bins, &mut workspace, context)?;
/// assert!((report.integrated_power - 1.0).abs() < 1e-12);
/// assert_eq!(report.peak_frequency.map(f64::abs), Some(0.5));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn periodogram_diagnostics_into<T: NumericScalar>(
    plan: &SpectralEstimatePlan<T>,
    values: &[T],
    output: &mut [PeriodogramBin<T::Real>],
    workspace: &mut PeriodogramWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<PeriodogramDiagnostics<T::Real>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    if values.len() != plan.segment_len()
        || output.len() != plan.bins()
        || workspace.density.len() != plan.bins()
        || plan.options().scaling() != SpectralScaling::Density
    {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "periodogram geometry or density scaling",
        ));
    }
    plan.power_into(
        values,
        &mut workspace.density,
        &mut workspace.signal,
        context,
    )?;
    let df = finite(plan.options().sample_rate() / number(plan.fft_len())?)?;
    let mut sum = scioxide_core::ScaledAccumulator::ZERO;
    let mut peak = 0;
    for (i, &power) in workspace.density.iter().enumerate() {
        sum.add_value(power)?;
        if power > workspace.density[peak] {
            peak = i;
        }
    }
    let total = sum.total()?;
    let integrated_power = total.mul(scioxide_core::Scaled::from_value(df)?)?.round()?;
    let mut cumulative = scioxide_core::ScaledAccumulator::ZERO;
    for (i, (out, &power)) in workspace
        .stage
        .iter_mut()
        .zip(&workspace.density)
        .enumerate()
    {
        cumulative.add_value(power)?;
        *out = PeriodogramBin {
            frequency: plan.frequency(i)?,
            density: power,
            cumulative_power: if total.is_zero() {
                None
            } else {
                Some(cumulative.total()?.div(total)?.round()?)
            },
        };
    }
    let report = PeriodogramDiagnostics {
        integrated_power,
        peak_frequency: if total.is_zero() {
            None
        } else {
            Some(plan.frequency(peak)?)
        },
        bin_width: df,
    };
    context.check()?;
    output.copy_from_slice(&workspace.stage);
    Ok(report)
}
