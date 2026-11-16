//! Periodogram, Welch, cross-spectral and coherence estimates.
use crate::axis::{Geometry, publish};
use crate::stft::frame::weighted_product;
use crate::util::{
    capacity, check_axis_shape, empty, exact_len, exact_shape, inconsistent, invalid,
    scalar_from_usize_rounded as scalar, sum_products, unsupported,
};
use crate::util::{inputs, zeros};
use crate::workspace::SignalBuffers;
use crate::{
    DetrendKind, SignalLayout, SignalRequirements, SignalWorkspace, StftOptions, StftPacking,
    StftScaling,
};
use crate::{WindowKind, WindowNormalization, WindowSymmetry};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::standard;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    Argument, Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar,
    RealScalar, SciOxideError,
};
mod lomb;
pub use lomb::{
    LombScargle, LombScargleMean, LombScargleNormalization, LombScargleOptions, lomb_scargle,
};

/// Estimate that one pass over the segments of a record computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpectralStatistic {
    /// Power spectrum estimate `P_xx`, in the real parts.
    Power,
    /// Cross spectrum estimate `P_xy`, from segment estimates `X conj(Y)`.
    Csd,
    /// Magnitude-squared coherence, in the real parts.
    Coherence,
}

/// Units of a power or cross spectral estimate.
///
/// Below, `X` is the DFT of a windowed segment, `w` the window and `fs` the
/// sample rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralScaling {
    /// Power spectral density `|X|² / (fs · Σ w²)`, in squared sample units per
    /// frequency unit.
    Density,
    /// Power spectrum `|X|² / (Σ w)²`, in squared sample units.
    Spectrum,
}
/// Reduction of the segment estimates to one estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralAverage {
    /// Arithmetic mean. Coherence requires it.
    Mean,
    /// Median, divided by the median of unit-mean exponential noise.
    ///
    /// For `K` segments the divisor is `Σᵢ (−1)^(i+1)/i`, `i = 1, …, 2J + 1`,
    /// with `J = ⌊(K − 1)/2⌋`. Real and imaginary parts take their medians
    /// separately. For even `K` the median is the mean of the two middle
    /// values.
    Median,
}
/// Options of a spectral estimate. The window, passed separately, sets the
/// segment length.
///
/// The defaults are no overlap, an FFT length equal to the window length, unit
/// sample rate, [`StftPacking::Full`], [`SpectralScaling::Density`],
/// [`DetrendKind::Constant`], [`SpectralAverage::Mean`] and sequential
/// execution. The setters do not validate. [`SpectralEstimatePlan::new`]
/// checks the overlap against the window and the transform options.
///
/// ```
/// use scioxide_signal::{SpectralAverage, SpectralEstimateOptions, StftPacking};
///
/// let options = SpectralEstimateOptions::new()
///     .with_fft_len(Some(8))
///     .with_sample_rate(2.0_f64)
///     .with_overlap(4)
///     .with_packing(StftPacking::OneSided)
///     .with_average(SpectralAverage::Median)
///     .with_detrend(None);
/// assert_eq!((options.fft_len(), options.overlap()), (Some(8), 4));
/// assert!(options.detrend().is_none());
/// assert_eq!(SpectralEstimateOptions::<f64>::default().fft_len(), None);
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct SpectralEstimateOptions<S: RealScalar> {
    overlap: usize,
    fft_len: Option<usize>,
    sample_rate: S,
    packing: StftPacking,
    scaling: SpectralScaling,
    detrend: Option<DetrendKind>,
    average: SpectralAverage,
    execution_policy: ExecutionPolicy,
}
impl<S: RealScalar> Default for SpectralEstimateOptions<S> {
    fn default() -> Self {
        Self {
            overlap: 0,
            fft_len: None,
            sample_rate: S::ONE,
            packing: StftPacking::Full,
            scaling: SpectralScaling::Density,
            detrend: Some(DetrendKind::Constant),
            average: SpectralAverage::Mean,
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
}
impl<S: RealScalar> SpectralEstimateOptions<S> {
    /// Returns the default options listed at [`SpectralEstimateOptions`].
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the FFT length `n`, in samples, or `None` for the window length.
    ///
    /// It must be at least the window length `m`. Each segment of `m` samples
    /// is zero-padded to `n` samples before its FFT, which samples the
    /// spectrum more finely without changing its resolution.
    pub fn with_fft_len(mut self, fft_len: Option<usize>) -> Self {
        self.fft_len = fft_len;
        self
    }
    /// Sets the sample rate `fs`, in samples per time unit. It must be positive
    /// and finite.
    pub fn with_sample_rate(mut self, sample_rate: S) -> Self {
        self.sample_rate = sample_rate;
        self
    }
    /// Sets the number of samples shared by consecutive segments. It must be
    /// less than the window length.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }
    /// Sets the order of the frequency bins. One-sided packing requires a real
    /// sample type.
    pub fn with_packing(mut self, packing: StftPacking) -> Self {
        self.packing = packing;
        self
    }
    /// Sets the units of the estimate.
    pub fn with_scaling(mut self, scaling: SpectralScaling) -> Self {
        self.scaling = scaling;
        self
    }
    /// Sets the trend removed from each segment before windowing, or `None`
    /// for no detrending.
    pub fn with_detrend(mut self, detrend: Option<DetrendKind>) -> Self {
        self.detrend = detrend;
        self
    }
    /// Sets the reduction across segments. Coherence requires
    /// [`SpectralAverage::Mean`].
    pub fn with_average(mut self, average: SpectralAverage) -> Self {
        self.average = average;
        self
    }
    /// Sets the execution policy of the FFTs. The workers come from the
    /// execution context of each call.
    pub fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }
    /// Returns the number of samples shared by consecutive segments.
    pub fn overlap(self) -> usize {
        self.overlap
    }
    /// Returns the FFT length, or `None` for the window length.
    pub fn fft_len(self) -> Option<usize> {
        self.fft_len
    }
    /// Returns the sample rate, in samples per time unit.
    pub fn sample_rate(self) -> S {
        self.sample_rate
    }
    /// Returns the order of the frequency bins.
    pub fn packing(self) -> StftPacking {
        self.packing
    }
    /// Returns the units of the estimate.
    pub fn scaling(self) -> SpectralScaling {
        self.scaling
    }
    /// Returns the trend removed from each segment, if any.
    pub fn detrend(self) -> Option<DetrendKind> {
        self.detrend
    }
    /// Returns the reduction across segments.
    pub fn average(self) -> SpectralAverage {
        self.average
    }
    /// Returns the execution policy of the FFTs. The default is sequential.
    pub fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
}
/// Periodogram and Welch estimates of power spectra, cross spectra and
/// coherence.
///
/// For a window `w` of length `m` and overlap `o`, a record of `N ≥ m`
/// samples has `K = 1 + ⌊(N − m)/(m − o)⌋` segments of length `m`. They start
/// at sample zero, and trailing samples that do not fill a segment are
/// ignored. Each segment is detrended, multiplied by `w` and transformed with
/// an FFT of length `n ≥ m`, giving `X` for record `x` and `Y` for record `y`.
/// The segment estimates are
///
/// ```text
/// P_xx = |X|² / S,    P_xy = X conj(Y) / S,
/// ```
///
/// with `S = fs · Σ w²` for [`SpectralScaling::Density`] and `S = (Σ w)²`
/// for [`SpectralScaling::Spectrum`], where `fs` is the sample rate. With
/// one-sided packing, every bin except DC and, for even `n`, the Nyquist bin
/// is doubled. [`SpectralAverage`] reduces the `K` segment estimates to one.
/// A record of one segment gives the periodogram. Coherence is
/// `|P_xy|² / (P_xx P_yy)` of mean estimates. The `*_axis_into` forms take
/// every lane along one axis of an array as a record.
///
/// An estimate costs `K` FFTs of length `n` per record. Records are rescaled
/// by powers of two where needed, so estimates do not overflow or underflow
/// prematurely, and coherence does not change when either record is scaled by
/// a power of two.
///
/// # Accuracy
///
/// Each segment spectrum has the normwise FFT error described under
/// [Accuracy](crate#accuracy). The mean of `K` segment estimates adds at most
/// `γ(8K + 32) · M/K + (8K + 32)·η`, where `M` sums, over the segments, the
/// absolute values of the component products that form each estimate.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{SignalLayout, SpectralEstimateOptions, SpectralEstimatePlan};
///
/// // Segments of 4 samples that overlap by 2: an 8-sample record has 3.
/// let options = SpectralEstimateOptions::new().with_overlap(2);
/// let plan = SpectralEstimatePlan::<f64>::new(&[1.0; 4], options)?;
/// let input = [1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.0, -2.0];
/// assert_eq!(plan.segments(input.len())?, 3);
/// let mut ws = plan.workspace(SignalLayout::Contiguous(input.len()))?;
/// let mut power = vec![0.0; plan.bins()];
/// let cx = ExecutionContext::sequential_native();
/// plan.power_into(&input, &mut power, &mut ws, cx)?;
/// assert_eq!(power, plan.power(&input)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - P. D. Welch, "The use of fast Fourier transform for the estimation of
///   power spectra: a method based on time averaging over short, modified
///   periodograms", *IEEE Trans. Audio Electroacoust.* 15(2), 70–73, 1967.
#[derive(Clone, Debug)]
pub struct SpectralEstimatePlan<T: NumericScalar> {
    frame: crate::stft::frame::FramePlan<T>,
    options: SpectralEstimateOptions<T::Real>,
    length: usize,
}
/// Per-call lane geometry, lane length and workspace of an axis estimate.
struct AxisLayout {
    geometry: Geometry,
    n: usize,
    requirements: SignalRequirements,
}
impl<T: NumericScalar> SpectralEstimatePlan<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Prepares the discrete spectra `|X[k]|² / n` and `X[k] conj(Y[k]) / n`
    /// of records of length `n = len`, where `X` and `Y` are unnormalized DFTs.
    ///
    /// The plan uses one rectangular segment of length `n`, no detrending,
    /// unit sample rate and full packing. See [Conventions](crate#conventions).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `len` is zero.
    /// - `AllocationFailure` if the window cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// // A constant record has all its power at DC: |X[0]|² / n = 16 / 4.
    /// let plan = scioxide_signal::SpectralEstimatePlan::<f64>::discrete(4)?;
    /// let power = plan.power(&[1., 1., 1., 1.])?;
    /// for (actual, expected) in power.iter().zip([4., 0., 0., 0.]) {
    ///     assert!((actual - expected).abs() <= 16. * f64::EPSILON);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn discrete(len: usize) -> Result<Self, SciOxideError> {
        let window = scioxide_core::backend::try_filled(
            len,
            T::Real::ONE,
            scioxide_core::quantity!("spectral_window"),
        )?;
        Self::new(&window, SpectralEstimateOptions::new().with_detrend(None))
    }
    /// Prepares estimates with the real window `window` and `options`.
    ///
    /// The window length `m` is the segment length, and consecutive segments
    /// start `m − o` samples apart for overlap `o`. The FFT length is that of
    /// the options, or `m` when the options leave it unset. Preparation plans
    /// one FFT and computes the window normalization in `O(m)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `window` is empty.
    /// - `InvalidInput` (`OutOfRange`) if the overlap is not less than the
    ///   window length, the FFT length exceeds `i64::MAX`, or the sample rate
    ///   is not positive.
    /// - `InvalidInput` (`Inconsistent`) if the FFT length is shorter than the
    ///   window.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if one-sided packing is
    ///   requested for a complex sample type.
    /// - `NonFinite` if the window or the sample rate is NaN or infinite.
    /// - `DomainError` if [`SpectralScaling::Spectrum`] meets a window with zero
    ///   sum, or [`SpectralScaling::Density`] an all-zero window.
    /// - `NumericalInstability` if the scaling factor overflows or underflows.
    pub fn new(
        window: &[T::Real],
        options: SpectralEstimateOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(window, options, ExecutionContext::sequential_native())
    }
    /// Prepares estimates as [`SpectralEstimatePlan::new`] does, planning the
    /// FFTs with the capabilities of `cx`. The plan does not keep `cx`.
    ///
    /// # Errors
    ///
    /// The errors of [`SpectralEstimatePlan::new`].
    pub fn build(
        window: &[T::Real],
        options: SpectralEstimateOptions<T::Real>,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if window.is_empty() {
            return Err(empty(
                scioxide_core::argument!("window"),
                "window must be nonempty",
            ));
        }
        if options.overlap >= window.len() {
            return Err(invalid(
                scioxide_core::argument!("options.overlap"),
                "overlap must be below the window length",
            ));
        }
        let mut converted = zeros::<T>(window.len())?;
        for (out, &w) in converted.iter_mut().zip(window) {
            *out = T::from_real(w);
        }
        let config = StftOptions::new()
            .with_fft_len(options.fft_len)
            .with_sample_rate(options.sample_rate)
            .with_packing(options.packing)
            .with_execution_policy(options.execution_policy)
            .with_scaling(match options.scaling {
                SpectralScaling::Density => StftScaling::Psd,
                SpectralScaling::Spectrum => StftScaling::Magnitude,
            });
        Ok(Self {
            frame: crate::stft::frame::FramePlan::new(
                &converted,
                window.len() - options.overlap,
                config,
                cx,
            )?,
            options,
            length: window.len(),
        })
    }

    /// Returns the options of the plan, as given.
    pub fn options(&self) -> SpectralEstimateOptions<T::Real> {
        self.options
    }
    /// Returns the FFT length `n`: the length set in the options, or the
    /// segment length when the options leave it unset.
    pub fn fft_len(&self) -> usize {
        self.frame.fft_len
    }
    /// Returns the segment length, which is the window length.
    pub fn segment_len(&self) -> usize {
        self.length
    }
    /// Returns the number of frequency bins: the FFT length `n`, or
    /// `⌊n/2⌋ + 1` with one-sided packing.
    pub fn bins(&self) -> usize {
        self.frame.bins()
    }
    /// Returns the frequency of bin `bin`, in cycles per time unit.
    ///
    /// [`StftPacking`] gives the frequency of each bin.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `bin` is not below
    ///   [`SpectralEstimatePlan::bins`].
    /// - `NumericalInstability` (`Overflow`) if the frequency overflows.
    pub fn frequency(&self, bin: usize) -> Result<T::Real, SciOxideError> {
        self.frame.frequency(bin)
    }
    /// Returns the number `K` of complete segments in a record of `n`
    /// samples.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `n` is less than the segment
    ///   length.
    pub fn segments(&self, n: usize) -> Result<usize, SciOxideError> {
        self.segment_count(n, scioxide_core::argument!("n"))
    }
    // `record` names the caller parameter whose length is `n`.
    fn segment_count(&self, n: usize, record: Argument) -> Result<usize, SciOxideError> {
        if n < self.length {
            return Err(inconsistent(
                record,
                "record must be at least one segment long",
            ));
        }
        Ok(1 + (n - self.length) / (self.length - self.options.overlap))
    }
    /// Returns the workspace size for estimates from the records of `layout`.
    ///
    /// [`SignalLayout::Contiguous`] sizes the record forms such as
    /// [`SpectralEstimatePlan::power_into`] for records of that length, and
    /// [`SignalLayout::Axis`] sizes the axis forms such as
    /// [`SpectralEstimatePlan::power_axis_into`] for arrays of that shape. One
    /// size serves every statistic. With [`SpectralAverage::Mean`] the size
    /// of a record does not depend on its number of segments; with
    /// [`SpectralAverage::Median`] it grows linearly with that number. An
    /// axis layout adds two lanes and the staged output of every lane.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the axis of an axis layout is not
    ///   below the rank of its shape.
    /// - `InvalidInput` (`Inconsistent`) if a record is shorter than one
    ///   segment.
    /// - `SizeOverflow` if a size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        layout: SignalLayout<'_>,
    ) -> Result<SignalRequirements, SciOxideError> {
        let record = scioxide_core::argument!("layout");
        match layout {
            SignalLayout::Contiguous(n) => self.requirements(n, record),
            SignalLayout::Axis { shape, axis } => {
                Ok(self.axis_layout(shape, axis, record)?.requirements)
            }
        }
    }
    /// Allocates a workspace of the size from
    /// [`SpectralEstimatePlan::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - The errors of [`SpectralEstimatePlan::workspace_requirements`].
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, layout: SignalLayout<'_>) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(layout)?)
    }
    /// Requirements with short-record errors attributed to `record`.
    fn requirements(
        &self,
        n: usize,
        record: Argument,
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::new(
            self.sample_len()?,
            0,
            self.work_count(n, record)?,
        ))
    }
    /// Lane geometry and workspace of the estimates along `axis` of arrays of
    /// `shape`: two lanes and the record scratch in the values, the staged
    /// output and the record work in the complex buffer. A short axis is
    /// attributed to `record`.
    fn axis_layout(
        &self,
        shape: &[usize],
        axis: usize,
        record: Argument,
    ) -> Result<AxisLayout, SciOxideError> {
        let n = *shape.get(axis).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("axis"),
                "axis must be below the shape rank",
            )
        })?;
        self.segment_count(n, record)?;
        let geometry = Geometry::new(shape, axis, n, self.bins())?;
        let lane = self.requirements(n, record)?;
        let requirements = SignalRequirements::new(
            checked_add(
                checked_mul(2, n, scioxide_core::quantity!("spectral_lanes"))?,
                lane.values_len(),
                scioxide_core::quantity!("spectral_batch_samples"),
            )?,
            0,
            checked_add(
                geometry.output_len,
                lane.complex_len(),
                scioxide_core::quantity!("spectral_batch_work"),
            )?,
        );
        Ok(AxisLayout {
            geometry,
            n,
            requirements,
        })
    }
    fn sample_len(&self) -> Result<usize, SciOxideError> {
        checked_mul(3, self.length, scioxide_core::quantity!("spectral_samples"))
    }
    /// Validate a record and select its exact power-of-two range scale: zero
    /// when its largest component lies in the ordinary FFT frame.
    fn record_exponent(&self, x: &[T]) -> Result<i32, SciOxideError> {
        crate::util::fft_exponent(
            crate::util::input_scale(x.iter().copied())?,
            self.frame.fft_len,
        )
    }
    fn work_count(&self, n: usize, record: Argument) -> Result<usize, SciOxideError> {
        let count = self.segment_count(n, record)?;
        let reductions = match self.options.average {
            SpectralAverage::Mean => 2,
            SpectralAverage::Median => count,
        };
        checked_add(
            checked_mul(
                checked_mul(
                    3,
                    self.bins(),
                    scioxide_core::quantity!("spectral_products"),
                )?,
                reductions,
                scioxide_core::quantity!("spectral_products"),
            )?,
            checked_add(
                checked_mul(2, self.bins(), scioxide_core::quantity!("spectral_frames"))?,
                self.frame.work_len,
                scioxide_core::quantity!("spectral_frames"),
            )?,
            scioxide_core::quantity!("spectral_work"),
        )
    }
    /// Spectrum of one segment `x` divided by `2^record`, and its binary
    /// exponent relative to that record frame and the plan frames.
    fn coefficients(
        &self,
        x: &[T],
        record: i32,
        out: &mut [Complex<T::Real>],
        samples: &mut [T],
        work: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<i32, SciOxideError> {
        let (source, rest) = samples.split_at_mut(self.length);
        let (values, scratch) = rest.split_at_mut(self.length);
        let x = if record == 0 {
            x
        } else {
            for (out, &v) in source.iter_mut().zip(x) {
                *out = standard::scale(v, -record);
            }
            &*source
        };
        if let Some(kind) = self.options.detrend {
            crate::detrend::detrend_staged(x, kind, &[], values, scratch)?;
        } else {
            values.copy_from_slice(x);
        }
        Ok(self
            .frame
            .analyze_frame_scaled(|j| values[j], out, work, context)?
            - self.frame.plan_exponent())
    }
    /// Only requested channels are computed. Mean uses compensated per-bin state;
    /// median keeps individual segment products for componentwise selection.
    /// Products and reductions use the records' range frames; power and CSD
    /// restore them once per bin, coherence keeps them.
    #[allow(clippy::too_many_arguments)]
    fn products<'w>(
        &self,
        x: &[T],
        y: &[T],
        samples: &mut [T],
        work: &'w mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
        record: Argument,
        statistic: SpectralStatistic,
    ) -> Result<&'w mut [Complex<T::Real>], SciOxideError> {
        exact_len(x.len(), y.len())?;
        capacity(self.sample_len()?, samples.len())?;
        capacity(self.work_count(x.len(), record)?, work.len())?;
        let ex = self.record_exponent(x)?;
        let same = core::ptr::eq(x, y);
        let ey = if same { ex } else { self.record_exponent(y)? };
        context.check()?;
        let count = self.segment_count(x.len(), record)?;
        let bins = self.bins();
        let mean = self.options.average == SpectralAverage::Mean;
        let storage = 3 * bins * if mean { 2 } else { count };
        let (products, rest) = work.split_at_mut(storage);
        products.fill(Complex::new(T::Real::ZERO, T::Real::ZERO));
        let (a, rest) = rest.split_at_mut(bins);
        let (b, fft) = rest.split_at_mut(bins);
        let hop = self.length - self.options.overlap;
        let reciprocal = T::Real::ONE / scalar::<T::Real>(count)?;
        let channels = match statistic {
            SpectralStatistic::Power => 0..1,
            SpectralStatistic::Csd => 2..3,
            SpectralStatistic::Coherence => 0..3,
        };
        let product_bins = if T::IS_REAL && self.options.packing != StftPacking::Centered {
            self.frame.fft_len / 2 + 1
        } else {
            bins
        };
        for p in 0..count {
            context.check()?;
            let range = p * hop..p * hop + self.length;
            let alpha = self.coefficients(&x[range.clone()], ex, a, samples, fft, context)?;
            let beta = if same {
                alpha
            } else {
                self.coefficients(&y[range], ey, b, samples, fft, context)?
            };
            for k in 0..product_bins {
                let weight = if self.options.scaling == SpectralScaling::Spectrum
                    && self.options.packing == StftPacking::OneSided
                    && k != 0
                    && !(self.frame.fft_len.is_multiple_of(2) && k == self.frame.fft_len / 2)
                {
                    T::Real::TWO
                } else {
                    T::Real::ONE
                };
                for channel in channels.clone() {
                    let (u, v, exponent) = if same {
                        (a[k], a[k], 2 * alpha)
                    } else {
                        match channel {
                            0 => (a[k], a[k], 2 * alpha),
                            1 => (b[k], b[k], 2 * beta),
                            _ => (a[k], b[k], alpha + beta),
                        }
                    };
                    let factor = weight * if mean { reciprocal } else { T::Real::ONE };
                    let z = weighted_product(u, v, factor, exponent)?;
                    let i = channel * bins + k;
                    if mean {
                        let (sums, corrections) = products.split_at_mut(3 * bins);
                        let add = |sum: &mut T::Real,
                                   correction: &mut T::Real,
                                   value: T::Real|
                         -> Result<(), SciOxideError> {
                            let (next, error) = scioxide_core::backend::two_sum(*sum, value);
                            if next.is_finite() {
                                *sum = next;
                                *correction = crate::util::finite(*correction + error)?;
                            } else {
                                *sum = sum_products(
                                    [(*sum, T::Real::ONE), (value, T::Real::ONE)].into_iter(),
                                )?;
                            }
                            Ok(())
                        };
                        add(&mut sums[i].re, &mut corrections[i].re, z.re)?;
                        add(&mut sums[i].im, &mut corrections[i].im, z.im)?;
                    } else {
                        products[i * count + p] = z;
                    }
                }
            }
        }
        // Restore power and CSD from the record frames; coherence is invariant.
        let plan = self.frame.plan_exponent();
        let restore = |channel: usize| match (statistic, channel) {
            (SpectralStatistic::Coherence, _) => 0,
            (_, 0) => 2 * (ex + plan),
            (_, 1) => 2 * (ey + plan),
            _ => ex + ey + 2 * plan,
        };
        for channel in channels.clone() {
            for k in 0..product_bins {
                let i = channel * bins + k;
                let reduced = if mean {
                    Complex::new(
                        crate::util::finite(products[i].re + products[3 * bins + i].re)?,
                        crate::util::finite(products[i].im + products[3 * bins + i].im)?,
                    )
                } else {
                    reduce(&mut products[i * count..(i + 1) * count])?
                };
                products[i] = standard::restore(reduced, restore(channel))?;
            }
        }
        // Packing may be centered; frame coefficients already use that ordering.
        // Only natural full real spectra admit this conjugate reconstruction.
        if T::IS_REAL && self.options.packing == StftPacking::Full {
            for channel in channels {
                for k in product_bins..bins {
                    let value = products[channel * bins + self.frame.fft_len - k];
                    products[channel * bins + k] = if channel == 2 {
                        value.conj()
                    } else {
                        Complex::new(value.re, T::Real::ZERO)
                    };
                }
            }
        }
        context.check()?;
        Ok(&mut products[..3 * bins])
    }
    /// Computes the power spectrum estimate `P_xx` of `input` and writes it to
    /// `output`.
    ///
    /// A record of one segment gives the periodogram. `output` needs at least
    /// [`SpectralEstimatePlan::bins`] elements, and `ws` the size of
    /// [`SpectralEstimatePlan::workspace_requirements`] for
    /// `SignalLayout::Contiguous(input.len())`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `input` is shorter than one
    ///   segment.
    /// - `DimensionMismatch` if `output` or `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if an estimate overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn power_into(
        &self,
        input: &[T],
        output: &mut [T::Real],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        capacity(self.bins(), output.len())?;
        let SignalBuffers {
            values, complex, ..
        } = ws.buffers(&self.requirements(input.len(), scioxide_core::argument!("input"))?)?;
        let products = self.products(
            input,
            input,
            values,
            complex,
            cx,
            scioxide_core::argument!("input"),
            SpectralStatistic::Power,
        )?;
        for (out, z) in output.iter_mut().zip(&products[..self.bins()]) {
            *out = z.re;
        }
        Ok(())
    }
    /// Computes the cross spectrum estimate `P_xy`, with segment estimates
    /// `X conj(Y) / S`, of `x` and `y` and writes it to `output`.
    ///
    /// `x` and `y` must have equal lengths. `output` needs at least
    /// [`SpectralEstimatePlan::bins`] elements, and `ws` the size of
    /// [`SpectralEstimatePlan::workspace_requirements`] for
    /// `SignalLayout::Contiguous(x.len())`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` and `y` differ in length, or `output` or
    ///   `ws` is too short.
    /// - `InvalidInput` (`Inconsistent`) if `x` is shorter than one segment.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if an estimate overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn csd_into(
        &self,
        x: &[T],
        y: &[T],
        output: &mut [Complex<T::Real>],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        capacity(self.bins(), output.len())?;
        let SignalBuffers {
            values, complex, ..
        } = self.paired_buffers(x, y, ws)?;
        let products = self.products(
            x,
            y,
            values,
            complex,
            cx,
            scioxide_core::argument!("x"),
            SpectralStatistic::Csd,
        )?;
        output[..self.bins()].copy_from_slice(&products[2 * self.bins()..]);
        Ok(())
    }
    /// Computes the magnitude-squared coherence `|P_xy|² / (P_xx P_yy)` of `x`
    /// and `y` and writes it to `output`.
    ///
    /// The three estimates use mean averaging over the same segments. The
    /// values lie in `[0, 1]`: a value above one by at most `64ε`, from
    /// rounding, is set to one. `output` needs at least
    /// [`SpectralEstimatePlan::bins`] elements, and `ws` the size of
    /// [`SpectralEstimatePlan::workspace_requirements`] for
    /// `SignalLayout::Contiguous(x.len())`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the plan uses
    ///   [`SpectralAverage::Median`].
    /// - The errors of [`SpectralEstimatePlan::csd_into`].
    /// - `DomainError` (`Undefined`) if `P_xx` or `P_yy` is zero at a bin.
    /// - `NumericalInstability` (`LossOfSignificance`) if a value exceeds one
    ///   by more than `64ε`.
    pub fn coherence_into(
        &self,
        x: &[T],
        y: &[T],
        output: &mut [T::Real],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.options.average != SpectralAverage::Mean {
            return Err(unsupported(
                scioxide_core::argument!("self"),
                "coherence requires mean averaging",
            ));
        }
        capacity(self.bins(), output.len())?;
        let SignalBuffers {
            values, complex, ..
        } = self.paired_buffers(x, y, ws)?;
        let p = self.products(
            x,
            y,
            values,
            complex,
            cx,
            scioxide_core::argument!("x"),
            SpectralStatistic::Coherence,
        )?;
        coherence_products(
            p,
            self.bins(),
            [scioxide_core::argument!("x"), scioxide_core::argument!("y")],
        )?;
        cx.check()?;
        for (out, z) in output.iter_mut().zip(&p[..self.bins()]) {
            *out = z.re;
        }
        Ok(())
    }
    /// Borrow paired-record buffers after the length check, so a length
    /// mismatch is reported before workspace errors.
    fn paired_buffers<'w>(
        &self,
        x: &[T],
        y: &[T],
        ws: &'w mut SignalWorkspace<T>,
    ) -> Result<SignalBuffers<'w, T>, SciOxideError> {
        exact_len(x.len(), y.len())?;
        ws.buffers(&self.requirements(x.len(), scioxide_core::argument!("x"))?)
    }
    /// Computes the power spectrum estimate `P_xx` of every lane of `input`
    /// along axis `axis` and writes it to `output`.
    ///
    /// Each lane of `n = input.shape[axis]` samples is one record, estimated
    /// as [`SpectralEstimatePlan::power_into`] estimates it. `output` has the
    /// shape of `input` with `n` replaced by [`SpectralEstimatePlan::bins`].
    /// Lanes run in row-major order of the other axes. Both views may be
    /// strided. `ws` needs the size of
    /// [`SpectralEstimatePlan::workspace_requirements`] for
    /// [`SignalLayout::Axis`] with the shape of `input`; with it the call
    /// does not allocate. The estimates are staged in `ws` and written to
    /// `output` after every lane succeeds. The cost is that of
    /// [`SpectralEstimatePlan::power_into`] per lane.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the rank of
    ///   `input`.
    /// - `InvalidInput` (`Inconsistent`) if the lanes are shorter than one
    ///   segment.
    /// - `DimensionMismatch` if `output` does not have the expected shape, or
    ///   `ws` is too short.
    /// - The errors of [`SpectralEstimatePlan::power_into`] for each lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{SignalLayout, SpectralEstimatePlan};
    /// // Two records of four samples, one per row of a 2 × 4 array.
    /// let (s, x) = ([2, 4], [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]);
    /// let layout = StridedLayout::new(&s, &[4, 1])?;
    /// let plan = SpectralEstimatePlan::<f64>::discrete(4)?;
    /// let mut ws = plan.workspace(SignalLayout::Axis { shape: &s, axis: 1 })?;
    /// let mut power = [0.0; 8];
    /// let output = NdViewMut::new(&mut power, layout)?;
    /// let cx = ExecutionContext::sequential_native();
    /// plan.power_axis_into(NdView::new(&x, layout)?, 1, output, &mut ws, cx)?;
    /// // |X|²/n: 16/4 at DC for the constant row, at bin 2 for the alternating one.
    /// assert!((power[0] - 4.0).abs() < 1e-12 && (power[6] - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn power_axis_into(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        mut output: NdViewMut<'_, '_, T::Real>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let staged = self.estimate_axis(
            &input,
            None,
            axis,
            output.layout().shape().extents(),
            ws,
            cx,
            SpectralStatistic::Power,
        )?;
        crate::axis::publish_real(staged, &mut output);
        Ok(())
    }
    /// Computes the cross spectrum estimate `P_xy` of every pair of lanes of
    /// `x` and `y` along axis `axis` and writes it to `output`.
    ///
    /// `x` and `y` have equal shapes, and each pair of lanes of
    /// `n = x.shape[axis]` samples is estimated as
    /// [`SpectralEstimatePlan::csd_into`] estimates it. `output` has the shape
    /// of `x` with `n` replaced by [`SpectralEstimatePlan::bins`]. Lanes,
    /// workspace, staging and cost are those of
    /// [`SpectralEstimatePlan::power_axis_into`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` and `y` differ in shape, `output` does not
    ///   have the expected shape, or `ws` is too short.
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the rank of `x`.
    /// - `InvalidInput` (`Inconsistent`) if the lanes are shorter than one
    ///   segment.
    /// - The errors of [`SpectralEstimatePlan::csd_into`] for each pair of
    ///   lanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{SignalLayout, SpectralEstimatePlan};
    /// // Row 0: the cosine leads the sine by a quarter period; row 1: it lags.
    /// let x = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    /// let y = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0];
    /// let (s, plan) = ([2, 4], SpectralEstimatePlan::<f64>::discrete(4)?);
    /// let layout = StridedLayout::new(&s, &[4, 1])?;
    /// let mut ws = plan.workspace(SignalLayout::Axis { shape: &s, axis: 1 })?;
    /// let mut cross = [Complex::new(0.0, 0.0); 8];
    /// let output = NdViewMut::new(&mut cross, layout)?;
    /// let (x, y) = (NdView::new(&x, layout)?, NdView::new(&y, layout)?);
    /// let cx = ExecutionContext::sequential_native();
    /// plan.csd_axis_into(x, y, 1, output, &mut ws, cx)?;
    /// assert!((cross[1] - Complex::new(0.0, 1.0)).norm() < 1e-12);
    /// assert!((cross[5] - Complex::new(0.0, -1.0)).norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn csd_axis_into(
        &self,
        x: NdView<'_, '_, T>,
        y: NdView<'_, '_, T>,
        axis: usize,
        mut output: NdViewMut<'_, '_, Complex<T::Real>>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let staged = self.estimate_axis(
            &x,
            Some(&y),
            axis,
            output.layout().shape().extents(),
            ws,
            cx,
            SpectralStatistic::Csd,
        )?;
        publish(staged, &mut output);
        Ok(())
    }
    /// Computes the magnitude-squared coherence of every pair of lanes of `x`
    /// and `y` along axis `axis` and writes it to `output`.
    ///
    /// `x` and `y` have equal shapes, and each pair of lanes of
    /// `n = x.shape[axis]` samples is estimated as
    /// [`SpectralEstimatePlan::coherence_into`] estimates it. `output` has the
    /// shape of `x` with `n` replaced by [`SpectralEstimatePlan::bins`].
    /// Lanes, workspace, staging and cost are those of
    /// [`SpectralEstimatePlan::power_axis_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the plan uses
    ///   [`SpectralAverage::Median`].
    /// - The errors of [`SpectralEstimatePlan::csd_axis_into`].
    /// - The errors of [`SpectralEstimatePlan::coherence_into`] for each pair
    ///   of lanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{SignalLayout, SpectralEstimateOptions, SpectralEstimatePlan};
    /// let options = SpectralEstimateOptions::new().with_detrend(None);
    /// let plan = SpectralEstimatePlan::<f64>::new(&[1.0; 4], options)?;
    /// // Two pairs of eight-sample records, one per row; y is a scaled copy of x.
    /// let x: Vec<f64> = (0..16).map(|i| (0.7 * i as f64).sin()).collect();
    /// let y: Vec<f64> = x.iter().map(|v| -2.0 * v).collect();
    /// let (s, layout) = ([2, 8], StridedLayout::new(&[2, 8], &[8, 1])?);
    /// let mut ws = plan.workspace(SignalLayout::Axis { shape: &s, axis: 1 })?;
    /// let mut c = [0.0; 8];
    /// let output = NdViewMut::new(&mut c, StridedLayout::new(&[2, 4], &[4, 1])?)?;
    /// let (x, y) = (NdView::new(&x, layout)?, NdView::new(&y, layout)?);
    /// let cx = ExecutionContext::sequential_native();
    /// plan.coherence_axis_into(x, y, 1, output, &mut ws, cx)?;
    /// assert!(c.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coherence_axis_into(
        &self,
        x: NdView<'_, '_, T>,
        y: NdView<'_, '_, T>,
        axis: usize,
        mut output: NdViewMut<'_, '_, T::Real>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.options.average != SpectralAverage::Mean {
            return Err(unsupported(
                scioxide_core::argument!("self"),
                "coherence requires mean averaging",
            ));
        }
        let staged = self.estimate_axis(
            &x,
            Some(&y),
            axis,
            output.layout().shape().extents(),
            ws,
            cx,
            SpectralStatistic::Coherence,
        )?;
        crate::axis::publish_real(staged, &mut output);
        Ok(())
    }
    /// Estimates `statistic` for every lane of `x`, paired with the lane of
    /// `y` for the cross statistics, and returns the staged complex bins in
    /// the row-major order of the output shape `output`.
    #[allow(clippy::too_many_arguments)]
    fn estimate_axis<'w>(
        &self,
        x: &NdView<'_, '_, T>,
        y: Option<&NdView<'_, '_, T>>,
        axis: usize,
        output: &[usize],
        ws: &'w mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
        statistic: SpectralStatistic,
    ) -> Result<&'w [Complex<T::Real>], SciOxideError> {
        let shape = x.layout().shape().extents();
        if let Some(y) = y {
            exact_shape(shape, y.layout().shape().extents())?;
        }
        let record = if statistic == SpectralStatistic::Power {
            scioxide_core::argument!("input")
        } else {
            scioxide_core::argument!("x")
        };
        let AxisLayout {
            geometry,
            n,
            requirements,
        } = self.axis_layout(shape, axis, record)?;
        let bins = self.bins();
        check_axis_shape(shape, axis, bins, output)?;
        let SignalBuffers {
            values, complex, ..
        } = ws.buffers(&requirements)?;
        cx.check()?;
        let (a, rest) = values.split_at_mut(n);
        let (b, samples) = rest.split_at_mut(n);
        let (staged, work) = complex.split_at_mut(geometry.output_len);
        for batch in 0..geometry.batches {
            geometry.gather(x, a, batch)?;
            let p = if let Some(y) = y {
                geometry.gather(y, b, batch)?;
                self.products(a, b, samples, work, cx, record, statistic)?
            } else {
                // One lane for both records selects the single-transform path.
                self.products(a, a, samples, work, cx, record, statistic)?
            };
            let lane = match statistic {
                SpectralStatistic::Power => &p[..bins],
                SpectralStatistic::Csd => &p[2 * bins..],
                SpectralStatistic::Coherence => {
                    coherence_products(
                        p,
                        bins,
                        [scioxide_core::argument!("x"), scioxide_core::argument!("y")],
                    )?;
                    &p[..bins]
                }
            };
            geometry.scatter(lane, batch, staged)?;
        }
        cx.check()?;
        Ok(staged)
    }
    /// Computes the power spectrum estimate of `input` and returns it.
    ///
    /// # Errors
    ///
    /// - The errors of [`SpectralEstimatePlan::power_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn power(&self, input: &[T]) -> Result<Vec<T::Real>, SciOxideError> {
        let mut out = zeros(self.bins())?;
        let mut ws = SignalWorkspace::reserved(
            &self.requirements(input.len(), scioxide_core::argument!("input"))?,
        )?;
        self.power_into(
            input,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }

    /// Computes the cross spectrum estimate of `x` and `y` and returns it.
    ///
    /// # Errors
    ///
    /// - The errors of [`SpectralEstimatePlan::csd_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn csd(&self, x: &[T], y: &[T]) -> Result<Vec<Complex<T::Real>>, SciOxideError> {
        let mut out = zeros(self.bins())?;
        let mut ws =
            SignalWorkspace::reserved(&self.requirements(x.len(), scioxide_core::argument!("x"))?)?;
        self.csd_into(
            x,
            y,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
    /// Computes the magnitude-squared coherence of `x` and `y` and returns it.
    ///
    /// # Errors
    ///
    /// - The errors of [`SpectralEstimatePlan::coherence_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn coherence(&self, x: &[T], y: &[T]) -> Result<Vec<T::Real>, SciOxideError> {
        let mut out = zeros(self.bins())?;
        let mut ws =
            SignalWorkspace::reserved(&self.requirements(x.len(), scioxide_core::argument!("x"))?)?;
        self.coherence_into(
            x,
            y,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
}
fn reduce<S: RealScalar>(values: &mut [Complex<S>]) -> Result<Complex<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let n = values.len();
    // E[median] for unit-mean exponential samples, including even sample counts.
    let mut bias = S::ONE;
    for j in 1..=(n - 1) / 2 {
        bias = bias + S::ONE / scalar::<S>(2 * j + 1)? - S::ONE / scalar::<S>(2 * j)?;
    }
    values.sort_unstable_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let re = sum_products(
        [
            (values[(n - 1) / 2].re, S::HALF),
            (values[n / 2].re, S::HALF),
        ]
        .into_iter(),
    )?;
    values.sort_unstable_by(|a, b| {
        a.im.partial_cmp(&b.im)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let im = sum_products(
        [
            (values[(n - 1) / 2].im, S::HALF),
            (values[n / 2].im, S::HALF),
        ]
        .into_iter(),
    )?;
    Ok(Complex::new(
        crate::util::finite(re / bias)?,
        crate::util::finite(im / bias)?,
    ))
}

/// Normalize cross products to coherence; zero power is attributed to
/// `records`, the caller parameters supplying x and y. The products may share
/// any per-record binary frames, which cancel in the ratio.
fn coherence_products<S: RealScalar>(
    p: &mut [Complex<S>],
    bins: usize,
    records: [Argument; 2],
) -> Result<(), SciOxideError> {
    let one = S::ONE;
    let allowance = scalar::<S>(64)? * S::EPSILON;
    for k in 0..bins {
        let a = p[k].re;
        let b = p[bins + k].re;
        if a <= S::ZERO {
            return Err(SciOxideError::domain_error(
                records[0],
                scioxide_core::DomainReason::Undefined,
                "coherence is undefined where x has zero power",
            ));
        }
        if b <= S::ZERO {
            return Err(SciOxideError::domain_error(
                records[1],
                scioxide_core::DomainReason::Undefined,
                "coherence is undefined where y has zero power",
            ));
        }
        let ratio = p[2 * bins + k].norm() / a.sqrt() / b.sqrt();
        let value = crate::util::finite(ratio * ratio)?;
        if value > one + allowance {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::LossOfSignificance,
                "coherence exceeds its numerical bound",
            ));
        }
        p[k] = Complex::new(value.min(one), S::ZERO);
    }
    Ok(())
}

/// Computes the discrete power spectrum `|X[k]|² / n` of `input`, where `X` is
/// the unnormalized DFT and `n` the length.
///
/// The bins are in natural DFT order. The result is not a density: the sample
/// rate does not enter. [`SpectralEstimatePlan::discrete`] prepares the same
/// computation for repeated use. The cost is one FFT of length `n`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a value overflows.
/// - `AllocationFailure` if the output or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_signal::power_spectrum;
///
/// // A unit cosine at bin 1 of 4 has |X|² = 4 at bins 1 and 3, so power 4/4.
/// let power = power_spectrum(&[1.0_f64, 0.0, -1.0, 0.0])?;
/// assert!(power.iter().zip([0.0, 1.0, 0.0, 1.0]).all(|(p, e)| (p - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn power_spectrum<T: NumericScalar>(input: &[T]) -> Result<Vec<T::Real>, SciOxideError> {
    SpectralEstimatePlan::discrete(input.len())?.power(input)
}
/// Computes the discrete cross spectrum `A[k] conj(B[k]) / n` of `a` and `b`,
/// where `A` and `B` are unnormalized DFTs and `n` the common length.
///
/// The bins are in natural DFT order. The phase of bin `k` is the phase by
/// which `a` leads `b` at that frequency. The cost is two FFTs of length `n`.
///
/// # Errors
///
/// - `DimensionMismatch` if `a` and `b` differ in length.
/// - `InvalidInput` (`Empty`) if the inputs are empty.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a value overflows.
/// - `AllocationFailure` if the output or workspace cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_signal::cross_spectrum;
///
/// // The cosine leads the sine by a quarter period: bin 1 has phase +π/2.
/// let cosine = [1.0_f64, 0.0, -1.0, 0.0];
/// let sine = [0.0, 1.0, 0.0, -1.0];
/// let cross = cross_spectrum(&cosine, &sine)?;
/// assert!(cross[1].re.abs() < 1e-12 && (cross[1].im - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn cross_spectrum<T: NumericScalar>(
    a: &[T],
    b: &[T],
) -> Result<Vec<Complex<T::Real>>, SciOxideError> {
    crate::util::exact_len(a.len(), b.len())?;
    SpectralEstimatePlan::discrete(a.len())?.csd(a, b)
}
/// Computes the Welch estimate of the power spectral density of `input`.
///
/// With `m = segment_len` and the sample rate `fs = sample_rate`, the record
/// is split into segments of `m` samples that start `m − ⌊m/2⌋` samples apart.
/// The mean of each segment is subtracted, the segment is multiplied by the
/// periodic Hann window `w[j] = sin²(πj/m)`, `j = 0, …, m − 1`, and
/// transformed with an FFT of length `m`, and the segment densities
/// `|X|² / (fs · Σ w²)` are averaged, as described, with their accuracy, at
/// [`SpectralEstimatePlan`]. Real samples give the `⌊m/2⌋ + 1`
/// bins of nonnegative frequency, with every bin except DC and, for even `m`,
/// the Nyquist bin doubled; complex samples give `m` bins in natural DFT
/// order. Bin `k` has the frequency `k·fs/m`, or `(k − m)·fs/m` in the upper
/// half of a complex estimate. The cost is one FFT of length `m` per segment.
///
/// This form fixes three values that depend on the segment length and the
/// sample type: the periodic Hann window, an overlap of `⌊m/2⌋` samples, and
/// one-sided packing for real samples (full packing for complex samples).
/// The periodic window is the form for DFT analysis. With a Hann window, half
/// overlap leaves a correlation of 16.7 % between adjacent segments, so
/// further overlap adds little variance reduction (Harris, §II and Table 1).
/// The other settings are the defaults of [`SpectralEstimateOptions`]: FFT
/// length `m`, density scaling, mean detrending and mean averaging.
/// [`SpectralEstimatePlan`] takes any other window and options.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `segment_len` is zero.
/// - `InvalidInput` (`OutOfRange`) if `segment_len` is `2⁵²` or more, beyond
///   which window coordinates are not exact, or if `sample_rate` is not
///   positive.
/// - `NonFinite` if a sample or `sample_rate` is NaN or infinite.
/// - `InvalidInput` (`Inconsistent`) if `input` is shorter than one segment.
/// - `NumericalInstability` if the scaling factor or an estimate is not
///   representable.
///
/// # Examples
///
/// ```
/// use scioxide_signal::welch;
/// // 12 Hz sampled at 128 Hz: segments of 64 samples have bins 2 Hz apart.
/// let fs = 128.0_f64;
/// let phase = |i: usize| 2.0 * std::f64::consts::PI * 12.0 * i as f64 / fs;
/// let input: Vec<f64> = (0..512).map(|i| phase(i).sin()).collect();
/// let density = welch(&input, fs, 64)?;
/// assert_eq!(density.len(), 33);
/// // The peak lies at bin 6, at 6·128/64 = 12 Hz.
/// assert!(density.iter().all(|&p| p <= density[6]));
/// // The density integrates to the mean power 1/2 of the sinusoid.
/// let power: f64 = density.iter().sum::<f64>() * fs / 64.0;
/// assert!((power - 0.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - F. J. Harris, "On the use of windows for harmonic analysis with the
///   discrete Fourier transform", *Proc. IEEE* 66(1), 51–83, 1978.
/// - P. D. Welch, "The use of fast Fourier transform for the estimation of
///   power spectra: a method based on time averaging over short, modified
///   periodograms", *IEEE Trans. Audio Electroacoust.* 15(2), 70–73, 1967.
pub fn welch<T: NumericScalar>(
    input: &[T],
    sample_rate: T::Real,
    segment_len: usize,
) -> Result<Vec<T::Real>, SciOxideError> {
    const SEGMENT_LEN: Argument = scioxide_core::argument!("segment_len");
    if segment_len == 0 {
        return Err(empty(SEGMENT_LEN, "window must be nonempty"));
    }
    // A periodic window samples `segment_len + 1` points of the symmetric one.
    if let Some(sampled) = segment_len.checked_add(1) {
        crate::windows::count(sampled, SEGMENT_LEN)?;
    }
    if sample_rate.is_finite() && sample_rate <= T::Real::ZERO {
        return Err(invalid(
            scioxide_core::argument!("sample_rate"),
            "sample rate must be positive",
        ));
    }
    let window = crate::windows::window::<T::Real>(
        WindowKind::Hann,
        segment_len,
        WindowSymmetry::Periodic,
        WindowNormalization::Native,
    )?;
    let packing = if T::IS_REAL {
        StftPacking::OneSided
    } else {
        StftPacking::Full
    };
    let options = SpectralEstimateOptions::new()
        .with_sample_rate(sample_rate)
        .with_overlap(segment_len / 2)
        .with_packing(packing);
    SpectralEstimatePlan::<T>::new(&window, options)?.power(input)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn coherence_bound_violation_is_a_numerical_failure() {
        let records = [scioxide_core::argument!("x"), scioxide_core::argument!("y")];
        // |Pxy|^2 = 4 * Pxx * Pyy cannot come from rounding.
        let mut p = [1.0_f64, 1., 2.].map(|v| Complex::new(v, 0.));
        let error = coherence_products(&mut p, 1, records).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::LossOfSignificance,
                ..
            }
        ));
        // Shared record frames cancel: unit coherence at any binary scale.
        let mut p = [-1000, 600, -200].map(|k| Complex::new(1.0_f64.scalbn(k), 0.));
        coherence_products(&mut p, 1, records).unwrap();
        assert_eq!(p[0].re, 1.);
        let mut p = [0.0_f64, 1., 0.].map(|v| Complex::new(v, 0.));
        let error = coherence_products(&mut p, 1, records).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                argument,
                reason: scioxide_core::DomainReason::Undefined,
                ..
            } if argument == records[0]
        ));
    }

    #[test]
    fn weighted_products_replay_only_non_finite_candidates() {
        let (u, v) = (Complex::new(f64::MAX, 0.), Complex::new(2., -1.));
        // The native product overflows; the weighted result is representable.
        let z = weighted_product(u, v, 0.25, 0).unwrap();
        assert_eq!((z.re, z.im), (f64::MAX / 2., f64::MAX / 4.));
        // Ordinary and gradually underflowing products stay native.
        let (a, b) = (Complex::new(0.75, 0.5), Complex::new(-0.5, 0.25));
        let native = (a * b.conj()).scale(0.5);
        assert_eq!(weighted_product(a, b, 0.5, 0).unwrap(), native);
        let tiny = weighted_product(a, b, 0.5, -1070).unwrap();
        assert_eq!(tiny.re, native.re.scalbn(-1070));
        assert_eq!(tiny.im, native.im.scalbn(-1070));
    }
}
