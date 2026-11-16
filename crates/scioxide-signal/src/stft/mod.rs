//! Prepared short-time Fourier analysis and dual-window synthesis.
use crate::util::{
    capacity, empty, exact_len, inconsistent, invalid, scalar_from_usize_rounded, unsupported,
};
use crate::util::{complex, copy, inputs as finite};
use crate::workspace::SignalBuffers;
use crate::{SignalBoundary, SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar, RealScalar,
    SciOxideError, Shape, Tolerance,
};
use scioxide_fft::{FftDirection, FftLayout, FftPlan, FftPlanSpec, Normalization};
mod convenience;
mod dual;
pub(crate) mod frame;
mod stream;
pub use dual::{
    ClosestStftDual, DualScale, OverlapDiagnostics, canonical_stft_dual, check_overlap,
    closest_stft_dual, tight_stft_window,
};

fn zero<S: RealScalar>() -> Complex<S> {
    Complex::new(S::ZERO, S::ZERO)
}

/// Order of the frequency bins in each STFT or spectral-estimate frame.
///
/// For an FFT length `n` and sample rate `fs`, DFT index `k` has frequency
/// `k·fs/n` for `k < ⌈n/2⌉` and `(k − n)·fs/n` otherwise, in cycles per time
/// unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StftPacking {
    /// Natural DFT order `k = 0, 1, …, n − 1`, with the negative frequencies in
    /// the upper half.
    Full,
    /// Ascending frequency, with zero frequency at bin `⌊n/2⌋`.
    Centered,
    /// The `⌊n/2⌋ + 1` bins of nonnegative frequency. Requires a real sample
    /// type.
    OneSided,
}
/// Scaling of STFT coefficients. Synthesis undoes it.
///
/// Below, `w` is the window and `fs` the sample rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StftScaling {
    /// Unnormalized DFT of the windowed frame.
    Raw,
    /// Divides by `|Σⱼ w[j]|`, so a complex exponential of unit amplitude at a
    /// bin frequency has a coefficient of magnitude one.
    Magnitude,
    /// Divides by `√(fs · Σⱼ |w[j]|²)`, so `|X|²` is a power spectral density.
    /// With one-sided packing, every bin except DC and, for even `n`, the
    /// Nyquist bin is also multiplied by `√2`.
    Psd,
}
/// Options of a short-time Fourier transform. The window and the hop are passed
/// to the plan separately.
///
/// The defaults are an FFT length equal to the window length, unit sample
/// rate, [`StftPacking::Full`], [`SignalBoundary::Zero`], [`StftScaling::Raw`],
/// the tolerance [`Tolerance::from_sqrt_epsilon`] and sequential execution.
/// The setters do not validate. [`StftPlan::new`] checks that the FFT length is
/// at least the window length and that the sample rate is positive and finite.
///
/// ```
/// use scioxide_signal::{StftOptions, StftPacking, StftScaling};
///
/// let options = StftOptions::<f64>::new()
///     .with_fft_len(Some(8))
///     .with_packing(StftPacking::OneSided)
///     .with_scaling(StftScaling::Psd)
///     .with_sample_rate(100.0);
/// assert_eq!(options.fft_len(), Some(8));
/// assert_eq!(options.packing(), StftPacking::OneSided);
/// assert_eq!(StftOptions::<f64>::default().fft_len(), None);
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct StftOptions<S: RealScalar> {
    fft_len: Option<usize>,
    sample_rate: S,
    packing: StftPacking,
    boundary: SignalBoundary,
    scaling: StftScaling,
    tolerance: Tolerance<S>,
    execution_policy: ExecutionPolicy,
}
impl<S: RealScalar> Default for StftOptions<S> {
    fn default() -> Self {
        Self {
            fft_len: None,
            sample_rate: S::ONE,
            packing: StftPacking::Full,
            boundary: SignalBoundary::Zero,
            scaling: StftScaling::Raw,
            tolerance: Tolerance::from_sqrt_epsilon(),
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
}
impl<S: RealScalar> StftOptions<S> {
    /// Returns the default options listed at [`StftOptions`].
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the FFT length `n`, in samples, or `None` for the window length.
    ///
    /// It must be at least the window length `m`. Each frame of `m` samples is
    /// zero-padded to `n` samples before its FFT, which samples the spectrum
    /// more finely without changing its resolution.
    pub fn with_fft_len(mut self, fft_len: Option<usize>) -> Self {
        self.fft_len = fft_len;
        self
    }
    /// Sets the sample rate, in samples per time unit. It must be positive and
    /// finite.
    pub fn with_sample_rate(mut self, sample_rate: S) -> Self {
        self.sample_rate = sample_rate;
        self
    }
    /// Sets the order of the frequency bins.
    pub fn with_packing(mut self, packing: StftPacking) -> Self {
        self.packing = packing;
        self
    }
    /// Sets how analysis extends the record beyond its ends.
    ///
    /// Streams require [`SignalBoundary::Zero`]. Synthesis does not use the
    /// extension.
    pub fn with_boundary(mut self, boundary: SignalBoundary) -> Self {
        self.boundary = boundary;
        self
    }
    /// Sets the scaling of the coefficients.
    pub fn with_scaling(mut self, scaling: StftScaling) -> Self {
        self.scaling = scaling;
        self
    }
    /// Sets the tolerance `(atol, rtol)` of the overlap and dual-window checks
    /// and of real reconstruction.
    ///
    /// [`check_overlap`] describes the overlap tests. With `t = atol + rtol`,
    /// synthesis for a real sample type accepts an imaginary part of at most
    /// `t` times its scale. For the DC and Nyquist bins of one-sided packing,
    /// the scale is the largest coefficient component of the frame. For a
    /// reconstructed sample, it is the sum, over the frames that overlap the
    /// sample, of the largest component of each frame's contribution. Scaling
    /// the coefficients by a power of two does not change the outcome.
    pub fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Sets the execution policy of the FFTs. The workers come from the
    /// execution context of each call.
    pub fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
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
    /// Returns how analysis extends the record beyond its ends.
    pub fn boundary(self) -> SignalBoundary {
        self.boundary
    }
    /// Returns the scaling of the coefficients.
    pub fn scaling(self) -> StftScaling {
        self.scaling
    }
    /// Returns the tolerance of the overlap checks and of real reconstruction.
    pub fn tolerance(self) -> Tolerance<S> {
        self.tolerance
    }
    /// Returns the execution policy of the FFTs. The default is sequential.
    pub fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
}
/// Range of STFT frames that cover a finite record.
///
/// With hop `h` and window length `m`, frame `p` covers the samples
/// `p·h − ⌊m/2⌋, …, p·h − ⌊m/2⌋ + m − 1`, so it is centered at sample `p·h`.
/// The range holds every frame whose window meets the record. Frame indices
/// are signed because the first frames start before sample zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StftFrames {
    first: i64,
    count: usize,
    signal_len: usize,
}
impl StftFrames {
    /// Returns the index of the first frame. It is zero or negative.
    pub const fn first(self) -> i64 {
        self.first
    }
    /// Returns the number of frames. An empty record has none.
    pub const fn count(self) -> usize {
        self.count
    }
    /// Returns the length of the record, in samples.
    pub const fn signal_len(self) -> usize {
        self.signal_len
    }
}
/// Operation and record length, or array shape, for
/// [`StftPlan::workspace_requirements`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StftRequest<'a> {
    /// [`StftPlan::analyze_into`] of a record of this length.
    Analysis(usize),
    /// [`StftPlan::synthesize_into`] of a record of this length.
    Synthesis(usize),
    /// [`StftPlan::spectrogram_into`] of a record of this length.
    Spectrogram(usize),
    /// [`StftPlan::cross_spectrogram_into`] of two records of this length.
    CrossSpectrogram(usize),
    /// [`StftPlan::analyze_detrended_into`] of a record of this length.
    DetrendedAnalysis(usize),
    /// [`StftPlan::analyze_axis_into`] and [`StftPlan::synthesize_axis_into`]
    /// of sample arrays of this shape.
    Axis {
        /// Shape of the sample array, one extent per axis.
        shape: &'a [usize],
        /// Sample axis along which the lanes run.
        axis: usize,
    },
}
/// Short-time Fourier transform (STFT) with a fixed window, and its inverse by
/// overlap-add with a dual window.
///
/// For a window `w` of length `m`, hop `h` and FFT length `n ≥ m`, frame `p`
/// holds the coefficients
///
/// ```text
/// X[p, k] = c_k Σⱼ x[p·h − ⌊m/2⌋ + j] conj(w[j]) exp(−2πi jk/n),   j = 0, …, m − 1,
/// ```
///
/// where `x` is the record extended as set by [`StftOptions::with_boundary`]
/// and `c_k` is the factor of [`StftScaling`]. The phase is relative to the
/// start of each frame. [`StftPlan::frames`] lists the frames, and the
/// coefficients are stored frame by frame, [`StftPlan::bins`] per frame, in the
/// order of [`StftPacking`]. Analysis costs one FFT of length `n` per frame.
/// [`StftPlan::analyze_axis_into`] and [`StftPlan::synthesize_axis_into`]
/// transform every lane along one axis of an array.
///
/// Synthesis inverts the FFT of each frame, multiplies it by a dual window `d`
/// and adds the overlapping frames. It reconstructs the record when, for each
/// residue `r = 0, …, h − 1`,
///
/// ```text
/// Σ_q conj(w[r + q·h]) d[r + q·h] = 1.
/// ```
///
/// A dual exists if and only if the window satisfies the nonzero overlap-add
/// (NOLA) condition: no residue has zero energy `Σ_q |w[r + q·h]|²`. When
/// [`check_overlap`] confirms NOLA, [`StftPlan::new`] uses the minimum-norm
/// dual of [`canonical_stft_dual`]. Otherwise the plan only analyzes, and
/// [`StftPlan::dual_window`] returns `None`. [`StftPlan::from_windows`] and
/// [`StftPlan::from_dual`] take a dual window that you supply.
///
/// # Accuracy
///
/// The coefficients have the normwise FFT error described under
/// [Accuracy](crate#accuracy). In synthesis, adding `r` overlapping frames
/// adds at most `γ(r)` times the sum of the absolute frame contributions, plus
/// `r·η`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_signal::{StftOptions, StftPlan, StftRequest};
///
/// // A rectangular window of length 4 with hop 2 satisfies NOLA.
/// let plan = StftPlan::new(&[1.0_f64; 4], 2, StftOptions::new())?;
/// let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mut ws = plan.workspace(StftRequest::Analysis(signal.len()))?;
/// let mut coefficients = vec![Complex::new(0.0, 0.0); plan.output_len(signal.len())?];
/// let cx = ExecutionContext::sequential_native();
/// plan.analyze_into(&signal, &mut coefficients, &mut ws, cx)?;
/// // Synthesis with the dual window restores the record.
/// let restored = plan.synthesize(&coefficients, signal.len())?;
/// assert!(restored.iter().zip(signal).all(|(a, b)| (a - b).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
///   Fourier transform", *IEEE Trans. Acoust., Speech, Signal Process.* 32(2),
///   236–243, 1984.
/// - I. Daubechies, A. Grossmann and Y. Meyer, "Painless nonorthogonal
///   expansions", *J. Math. Phys.* 27(5), 1271–1283, 1986.
#[derive(Clone, Debug)]
pub struct StftPlan<T: NumericScalar> {
    frame: frame::FramePlan<T>,
    dual: Option<Vec<T>>,
    inverse: FftPlan<T::Real>,
    fft_work: usize,
    overlap: OverlapDiagnostics<T::Real>,
}
/// Per-call geometry of an axis transform: lane staging, the frames of one
/// lane of `n` samples, the sample count of the array, the coefficient count
/// of one lane and the workspace.
struct AxisLayout {
    geometry: crate::axis::Geometry,
    frames: StftFrames,
    n: usize,
    samples: usize,
    lane_output: usize,
    requirements: SignalRequirements,
}
impl<T: NumericScalar> StftPlan<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Prepares the STFT of `window` with hop `hop`, the number of samples
    /// between the starts of consecutive frames, and the minimum-norm dual
    /// window when the window and hop satisfy NOLA.
    ///
    /// The FFT length is that of the options, or the window length when the
    /// options leave it unset. Preparation plans the forward and inverse FFTs
    /// and, under NOLA, computes the dual in `O(m)` for window length `m`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `window` is empty.
    /// - `InvalidInput` (`OutOfRange`) if `hop` is zero or exceeds `i64::MAX`,
    ///   the FFT length exceeds `i64::MAX`, or the sample rate is not positive.
    /// - `InvalidInput` (`Inconsistent`) if the FFT length is shorter than the
    ///   window.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if one-sided packing is
    ///   requested for a complex sample type.
    /// - `NonFinite` if the window or the sample rate is NaN or infinite.
    /// - `DomainError` if [`StftScaling::Magnitude`] meets a window with zero
    ///   sum, or [`StftScaling::Psd`] an all-zero window.
    /// - `NumericalInstability` if the scaling factor overflows or underflows,
    ///   or the computed dual window fails its reconstruction check
    ///   (`LossOfSignificance`).
    pub fn new(
        window: &[T],
        hop: usize,
        options: StftOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(window, hop, options, ExecutionContext::sequential_native())
    }
    /// Prepares the plan as [`StftPlan::new`] does, planning the FFTs with the
    /// capabilities of `cx`. The plan does not keep `cx`.
    ///
    /// # Errors
    ///
    /// The errors of [`StftPlan::new`].
    pub fn build(
        window: &[T],
        hop: usize,
        options: StftOptions<T::Real>,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let frame = frame::FramePlan::new(window, hop, options, cx)?;
        let overlap = dual::check_overlap(window, hop, options.tolerance)?;
        let dual = if overlap.nola() {
            Some(dual::canonical_stft_dual(window, hop, options.tolerance)?)
        } else {
            None
        };
        let inverse = FftPlan::build(
            FftPlanSpec::new(
                frame.fft_len,
                FftDirection::Inverse,
                Normalization::Inverse,
                if T::IS_REAL && options.packing == StftPacking::OneSided {
                    FftLayout::Real
                } else {
                    FftLayout::Complex
                },
            )
            .with_execution_policy(options.execution_policy),
            cx,
        )?;
        let fft_work = frame.work_len.max(checked_add(
            checked_mul(2, frame.fft_len, scioxide_core::quantity!("stft_fft_work"))?,
            inverse.workspace_requirements(())?.complex_len(),
            scioxide_core::quantity!("stft_fft_work"),
        )?);
        Ok(Self {
            frame,
            dual,
            inverse,
            fft_work,
            overlap,
        })
    }
    /// Replace the minimum-norm dual with `dual` after checking reconstruction.
    fn replace_dual(mut self, dual: &[T]) -> Result<Self, SciOxideError> {
        if !self.overlap.nola() {
            return Err(inconsistent(
                scioxide_core::argument!("window"),
                "window and hop violate the nonzero overlap-add condition",
            ));
        }
        dual::validate_dual(
            &self.frame.window,
            dual,
            self.frame.hop,
            self.frame.options.tolerance,
            || {
                inconsistent(
                    scioxide_core::argument!("dual"),
                    "dual violates the reconstruction constraint",
                )
            },
        )?;
        self.dual = Some(copy(dual)?);
        Ok(self)
    }
    /// Returns the analysis window.
    pub fn window(&self) -> &[T] {
        &self.frame.window
    }
    /// Returns the synthesis (dual) window, or `None` when the window and hop
    /// violate NOLA and the plan cannot synthesize.
    pub fn dual_window(&self) -> Option<&[T]> {
        self.dual.as_deref()
    }
    /// Returns the options of the plan, as given.
    pub fn options(&self) -> StftOptions<T::Real> {
        self.frame.options
    }
    /// Returns the hop: the number of samples between the starts of
    /// consecutive frames.
    pub fn hop(&self) -> usize {
        self.frame.hop
    }
    /// Returns the FFT length `n`: the length set in the options, or the window
    /// length when the options leave it unset.
    pub fn fft_len(&self) -> usize {
        self.frame.fft_len
    }
    /// Returns the COLA and NOLA diagnostics of the window and hop, as
    /// [`check_overlap`] computes them.
    pub fn overlap(&self) -> OverlapDiagnostics<T::Real> {
        self.overlap
    }
    /// Returns the number of frequency bins per frame: the FFT length `n`, or
    /// `⌊n/2⌋ + 1` with one-sided packing.
    pub fn bins(&self) -> usize {
        self.frame.bins()
    }
    /// Returns the frames whose window meets a record of `signal_len` samples.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a frame index or the frame count does not fit in
    ///   `i64`.
    pub fn frames(&self, signal_len: usize) -> Result<StftFrames, SciOxideError> {
        let m = self.frame.window.len() as i128;
        let h = self.frame.hop as i128;
        let first = (m / 2 - m).div_euclid(h) + 1;
        let end = if signal_len == 0 {
            first
        } else {
            (signal_len as i128 - 1 + m / 2).div_euclid(h) + 1
        };
        if signal_len > i64::MAX as usize
            || end > i64::MAX as i128
            || end - first > i64::MAX as i128
        {
            return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "stft_frames"
            )));
        }
        Ok(StftFrames {
            first: first as i64,
            count: usize::try_from(end - first).map_err(|_| {
                SciOxideError::size_overflow(scioxide_core::quantity!("stft_frames"))
            })?,
            signal_len,
        })
    }

    fn start(&self, frame: i64) -> i128 {
        frame as i128 * self.frame.hop as i128 - (self.frame.window.len() / 2) as i128
    }
    /// Returns the frequency of bin `bin`, in cycles per time unit.
    ///
    /// [`StftPacking`] gives the frequency of each bin.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `bin` is not below [`StftPlan::bins`].
    /// - `NumericalInstability` (`Overflow`) if the frequency overflows.
    pub fn frequency(&self, bin: usize) -> Result<T::Real, SciOxideError> {
        self.frame.frequency(bin)
    }
    /// Returns the center time `p·h / fs` of frame `p`, where `h` is the hop and
    /// `fs` the sample rate.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the time overflows.
    pub fn frame_time(&self, frame: i64) -> Result<T::Real, SciOxideError> {
        let t = scalar_from_usize_rounded::<T::Real>(frame.unsigned_abs() as usize)?
            * scalar_from_usize_rounded::<T::Real>(self.frame.hop)?
            / self.frame.options.sample_rate;
        crate::util::finite(if frame < 0 { -t } else { t })
    }
    fn bin_index(&self, bin: usize) -> usize {
        self.frame.bin_index(bin)
    }
    fn bin_weight(&self, bin: usize) -> T::Real {
        self.frame.bin_weight(bin)
    }
    /// Returns the number of coefficients of a record of `signal_len` samples:
    /// the frame count of [`StftPlan::frames`] times [`StftPlan::bins`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the count does not fit in `usize`.
    pub fn output_len(&self, signal_len: usize) -> Result<usize, SciOxideError> {
        checked_mul(
            self.frames(signal_len)?.count,
            self.bins(),
            scioxide_core::quantity!("stft_output"),
        )
    }

    /// Returns the workspace size of the operation in `request`.
    ///
    /// The size grows linearly with the record length. For
    /// [`StftRequest::Axis`] it also holds the staged samples and coefficients
    /// of the whole array, and one size serves both directions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the axis of [`StftRequest::Axis`] is
    ///   not below the rank of its shape.
    /// - `SizeOverflow` if a size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        request: StftRequest<'_>,
    ) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("stft_workspace");
        let staged = |copies: usize, n: usize| -> Result<usize, SciOxideError> {
            checked_add(
                checked_mul(copies, self.output_len(n)?, quantity)?,
                self.fft_work,
                quantity,
            )
        };
        Ok(match request {
            StftRequest::Analysis(n) | StftRequest::Spectrogram(n) => {
                SignalRequirements::complex(staged(1, n)?)
            }
            StftRequest::CrossSpectrogram(n) => {
                SignalRequirements::complex(checked_add(staged(1, n)?, self.bins(), quantity)?)
            }
            StftRequest::Synthesis(n) => {
                self.frames(n)?;
                SignalRequirements::new(0, n, checked_add(n, self.fft_work, quantity)?)
            }
            StftRequest::DetrendedAnalysis(n) => SignalRequirements::new(
                checked_mul(3, self.frame.window.len(), quantity)?,
                0,
                staged(1, n)?,
            ),
            StftRequest::Axis { shape, axis } => self.axis_layout(shape, axis)?.requirements,
        })
    }
    /// Lane geometry and workspace of the transform along `axis` of sample
    /// arrays of `shape`. The values hold the staged samples and one lane, the
    /// real buffer the reconstruction scales of one lane, and the complex
    /// buffer the staged coefficients, one lane of coefficients, one lane of
    /// samples and the FFT work.
    fn axis_layout(&self, shape: &[usize], axis: usize) -> Result<AxisLayout, SciOxideError> {
        if axis >= shape.len() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                "axis must be below the shape rank",
            ));
        }
        let samples = Shape::new(shape)?.len();
        let n = shape[axis];
        let frames = self.frames(n)?;
        let lane_output = self.output_len(n)?;
        let geometry = crate::axis::Geometry::new(shape, axis, n, lane_output)?;
        let quantity = scioxide_core::quantity!("stft_axis_workspace");
        let requirements = SignalRequirements::new(
            checked_add(samples, n, quantity)?,
            n,
            [geometry.output_len, lane_output, n, self.fft_work]
                .into_iter()
                .try_fold(0, |total, v| checked_add(total, v, quantity))?,
        );
        Ok(AxisLayout {
            geometry,
            frames,
            n,
            samples,
            lane_output,
            requirements,
        })
    }
    /// Require the coefficient shape `coefficients` for sample arrays of
    /// `shape`: `shape[..axis]`, the frame count, the bin count, then
    /// `shape[axis + 1..]`. The check compares extent by extent; only a
    /// mismatch allocates, for the expected shape of its error.
    fn check_coefficient_shape(
        &self,
        shape: &[usize],
        axis: usize,
        frames: usize,
        coefficients: &[usize],
    ) -> Result<(), SciOxideError> {
        let matches = coefficients.len() == shape.len() + 1
            && coefficients[..axis] == shape[..axis]
            && coefficients[axis] == frames
            && coefficients[axis + 1] == self.bins()
            && coefficients[axis + 2..] == shape[axis + 1..];
        if matches {
            return Ok(());
        }
        let expected: Vec<usize> = shape[..axis]
            .iter()
            .copied()
            .chain([frames, self.bins()])
            .chain(shape[axis + 1..].iter().copied())
            .collect();
        Err(SciOxideError::dimension_mismatch(expected, coefficients))
    }

    /// Allocates a workspace of the size from
    /// [`StftPlan::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, request: StftRequest<'_>) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }

    fn sample(&self, input: &[T], index: i128) -> T {
        let n = input.len() as i128;
        if n == 0 {
            return T::ZERO;
        }
        let mapped = if index >= 0 && index < n {
            index
        } else {
            match self.frame.options.boundary {
                SignalBoundary::Zero => return T::ZERO,
                SignalBoundary::Periodic => index.rem_euclid(n),
                SignalBoundary::Reflect => {
                    if n == 1 {
                        0
                    } else {
                        let p = 2 * (n - 1);
                        let k = index.rem_euclid(p);
                        k.min(p - k)
                    }
                }
            }
        };
        input[mapped as usize]
    }
    /// Analyze one frame through the shared forward kernel.
    pub(crate) fn analyze_frame(
        &self,
        sample: impl Fn(usize) -> T,
        output: &mut [Complex<T::Real>],
        work: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.frame.analyze_frame(sample, output, work, context)
    }
    /// Analyze every frame of validated `input` into `staged` (exactly the
    /// coefficient count), using `fft` as frame scratch.
    fn analyze_staged(
        &self,
        input: &[T],
        frames: StftFrames,
        staged: &mut [Complex<T::Real>],
        fft: &mut [Complex<T::Real>],
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        for (i, out) in staged.chunks_exact_mut(self.bins()).enumerate() {
            let start = self.start(frames.first + i as i64);
            self.analyze_frame(|j| self.sample(input, start + j as i128), out, fft, cx)?;
        }
        Ok(())
    }
    /// Overlap-add validated coefficients into `staged` (exactly the record
    /// length) with per-sample reconstruction `scales`, and require a real
    /// reconstruction for real samples.
    fn synthesize_staged(
        &self,
        input: &[Complex<T::Real>],
        frames: StftFrames,
        staged: &mut [Complex<T::Real>],
        scales: &mut [T::Real],
        scratch: &mut [Complex<T::Real>],
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        staged.fill(zero());
        scales.fill(T::Real::ZERO);
        for (i, frame) in input.chunks_exact(self.bins()).enumerate() {
            let start = self.start(frames.first + i as i64);
            let (values, frame_scale) = self.inverse_frame(frame, scratch, cx)?;
            for (j, &v) in values.iter().enumerate() {
                let k = start + j as i128;
                if k >= 0 && k < staged.len() as i128 {
                    overlap_add(
                        &mut staged[k as usize],
                        &mut scales[k as usize],
                        v,
                        frame_scale,
                    )?;
                }
            }
        }
        self.validate_real(staged, scales)
    }
    /// Require a dual window before any inversion.
    fn require_dual(&self) -> Result<(), SciOxideError> {
        if self.dual.is_none() {
            return Err(unsupported(
                scioxide_core::argument!("self"),
                "plan has no dual window for inversion",
            ));
        }
        Ok(())
    }
    /// Invert one frame of the caller's `input`, returning its window-length
    /// contribution and that contribution's largest component magnitude.
    fn inverse_frame<'a>(
        &self,
        input: &[Complex<T::Real>],
        work: &'a mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<FrameContribution<'a, T::Real>, SciOxideError> {
        use scioxide_core::backend::{components_mut, standard};
        let n = self.frame.fft_len;
        let (buffer, rest) = work.split_at_mut(n);
        let (samples, fft) = rest.split_at_mut(n);
        buffer.fill(zero());
        for (k, &z) in input.iter().enumerate() {
            buffer[self.bin_index(k)] = z.unscale(self.bin_weight(k));
        }
        let real_inverse = self.frame.options.packing == StftPacking::OneSided;
        if real_inverse {
            // Endpoint imaginary parts are measured against the frame scale,
            // not their own bin, so the decision is invariant under scaling.
            let frame_scale = crate::util::scale(buffer);
            for k in [0, n / 2] {
                if k == 0 || n.is_multiple_of(2) {
                    if !self.is_real(buffer[k].im, frame_scale)? {
                        return Err(SciOxideError::domain_error(
                            scioxide_core::argument!("input"),
                            scioxide_core::DomainReason::OutsideDomain,
                            "one-sided endpoint bins must be real within tolerance",
                        ));
                    }
                    buffer[k].im = T::Real::ZERO;
                }
            }
        }
        let exponent = crate::util::fft_exponent(crate::util::scale(buffer), n)?;
        if exponent != 0 {
            for z in buffer.iter_mut() {
                *z = standard::scale(*z, -exponent);
            }
        }
        if real_inverse {
            let values = &mut components_mut(samples)[..n];
            self.inverse.execute_into(
                scioxide_fft::FftInput::hermitian(n, &buffer[..n / 2 + 1]),
                &mut *values,
                fft,
                context,
            )?;
            for (z, &v) in buffer.iter_mut().zip(values.iter()) {
                *z = Complex::new(v, T::Real::ZERO);
            }
        } else {
            self.inverse.execute_in_place(buffer, fft, context)?;
        }
        let dual = self.dual.as_deref().ok_or_else(|| {
            unsupported(
                scioxide_core::argument!("self"),
                "plan has no dual window for inversion",
            )
        })?;
        for (z, &d) in buffer.iter_mut().zip(dual) {
            let v = *z * complex(d);
            let factor = T::Real::ONE / self.frame.scale;
            let ordinary = v.scale(factor);
            let ordinary_product =
                v.re.is_normal() || v.im.is_normal() || *z == zero() || d == T::ZERO;
            *z = if exponent == 0
                && ordinary_product
                && ordinary.has_finite_components()
                && factor.is_finite()
                && (ordinary.re.is_normal() || ordinary.re == T::Real::ZERO)
                && (ordinary.im.is_normal() || ordinary.im == T::Real::ZERO)
            {
                ordinary
            } else {
                let value = scioxide_core::ScaledValue::from_value(*z)?
                    .mul(scioxide_core::ScaledValue::from_value(complex(d))?)?;
                let restore = |x: scioxide_core::Scaled<T::Real>| {
                    x.div(scioxide_core::Scaled::from_value(self.frame.scale)?)?
                        .shifted(i64::from(exponent))?
                        .round()
                };
                Complex::new(restore(value.real())?, restore(value.imaginary())?)
            };
        }
        let values = &buffer[..self.frame.window.len()];
        finite(values)?;
        Ok((values, crate::util::scale(values)))
    }
    /// Computes the STFT coefficients of `input` and writes them to `output`,
    /// frame by frame.
    ///
    /// `output` needs at least [`StftPlan::output_len`] elements, and `ws` the
    /// size for [`StftRequest::Analysis`] of `input.len()`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` or `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a coefficient overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn analyze_into(
        &self,
        input: &[T],
        output: &mut [Complex<T::Real>],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let frames = self.frames(input.len())?;
        let n = self.output_len(input.len())?;
        capacity(n, output.len())?;
        let buffers =
            ws.buffers(&self.workspace_requirements(StftRequest::Analysis(input.len()))?)?;
        finite(input)?;
        cx.check()?;
        let (staged, fft) = buffers.complex.split_at_mut(n);
        self.analyze_staged(input, frames, staged, fft, cx)?;
        cx.check()?;
        output[..n].copy_from_slice(staged);
        Ok(())
    }

    /// Computes the STFT coefficients of `input` and returns them, frame by
    /// frame.
    ///
    /// # Errors
    ///
    /// - The errors of [`StftPlan::analyze_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn analyze(&self, input: &[T]) -> Result<Vec<Complex<T::Real>>, SciOxideError> {
        let mut output = try_filled(
            self.output_len(input.len())?,
            zero(),
            scioxide_core::quantity!("stft_output"),
        )?;
        let mut ws = self.workspace(StftRequest::Analysis(input.len()))?;
        self.analyze_into(
            input,
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Reconstructs a record of `output.len()` samples from its STFT
    /// coefficients `input` by overlap-add with the dual window.
    ///
    /// `input` holds the coefficients of every frame of a record of
    /// `output.len()` samples, as [`StftPlan::analyze_into`] writes them. `ws`
    /// needs the size for [`StftRequest::Synthesis`] of `output.len()`. For a
    /// real sample type, the reconstruction must be real within the tolerance
    /// described at [`StftOptions::with_tolerance`], and its imaginary part is
    /// discarded.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the plan has no dual
    ///   window.
    /// - `DimensionMismatch` if `input.len()` differs from
    ///   [`StftPlan::output_len`] of `output.len()`, or `ws` is too short.
    /// - `DomainError` (`OutsideDomain`) if, for a real sample type, a one-sided
    ///   DC or Nyquist coefficient or a reconstructed sample is not real within
    ///   the tolerance.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a sample overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn synthesize_into(
        &self,
        input: &[Complex<T::Real>],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.require_dual()?;
        let frames = self.frames(output.len())?;
        exact_len(self.output_len(output.len())?, input.len())?;
        let buffers =
            ws.buffers(&self.workspace_requirements(StftRequest::Synthesis(output.len()))?)?;
        finite(input)?;
        cx.check()?;
        let (staged, scratch) = buffers.complex.split_at_mut(output.len());
        self.synthesize_staged(input, frames, staged, buffers.real, scratch, cx)?;
        cx.check()?;
        for (slot, z) in output.iter_mut().zip(staged.iter()) {
            *slot = T::from_components(z.re, z.im);
        }
        Ok(())
    }
    /// Computes the STFT of every lane of `input` along axis `axis` and writes
    /// the coefficients to `output`.
    ///
    /// Each lane of `n = input.shape[axis]` samples is one record, transformed
    /// as [`StftPlan::analyze_into`] transforms it. `output` has one axis more
    /// than `input`: the sample axis is replaced by two adjacent axes, the
    /// [`StftPlan::frames`] of a record of `n` samples and then the
    /// [`StftPlan::bins`] of each frame. Lanes run in row-major order of the
    /// other axes, and both views may be strided. `ws` needs the size of
    /// [`StftPlan::workspace_requirements`] for [`StftRequest::Axis`] with the
    /// shape of `input`; with it the call does not allocate. The coefficients
    /// are staged in `ws` and written to `output` after every lane succeeds.
    /// The cost is that of [`StftPlan::analyze_into`] per lane.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the rank of
    ///   `input`.
    /// - `DimensionMismatch` if `output` does not have the coefficient shape,
    ///   or `ws` is too short.
    /// - The errors of [`StftPlan::analyze_into`] for each lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{StftOptions, StftPlan, StftRequest};
    /// let plan = StftPlan::new(&[1.0_f64; 4], 2, StftOptions::new())?;
    /// // Two records of six samples: shape [2, 6] becomes [2, 4 frames, 4 bins].
    /// let x: Vec<f64> = (0..12).map(|i| (0.5 * i as f64).sin()).collect();
    /// let (s, cx) = ([2, 6], ExecutionContext::sequential_native());
    /// let mut ws = plan.workspace(StftRequest::Axis { shape: &s, axis: 1 })?;
    /// let mut coefficients = [Complex::new(0.0, 0.0); 32];
    /// let layout = StridedLayout::new(&[2, 4, 4], &[16, 4, 1])?;
    /// let output = NdViewMut::new(&mut coefficients, layout)?;
    /// let input = NdView::new(&x, StridedLayout::new(&s, &[6, 1])?)?;
    /// plan.analyze_axis_into(input, 1, output, &mut ws, cx)?;
    /// for (a, b) in coefficients[16..].iter().zip(&plan.analyze(&x[6..])?) {
    ///     assert!((*a - *b).norm() < 1e-12);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn analyze_axis_into(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        mut output: NdViewMut<'_, '_, Complex<T::Real>>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let shape = input.layout().shape().extents();
        let AxisLayout {
            geometry,
            frames,
            n,
            lane_output,
            requirements,
            ..
        } = self.axis_layout(shape, axis)?;
        self.check_coefficient_shape(shape, axis, frames.count, output.layout().shape().extents())?;
        let buffers = ws.buffers(&requirements)?;
        cx.check()?;
        let lane = &mut buffers.values[..n];
        let (staged, rest) = buffers.complex.split_at_mut(geometry.output_len);
        let (coefficients, fft) = rest.split_at_mut(lane_output);
        for batch in 0..geometry.batches {
            geometry.gather(&input, lane, batch)?;
            finite(lane)?;
            cx.check()?;
            self.analyze_staged(lane, frames, coefficients, fft, cx)?;
            geometry.scatter(coefficients, batch, staged)?;
        }
        cx.check()?;
        crate::axis::publish(staged, &mut output);
        Ok(())
    }
    /// Reconstructs every lane of `output` along axis `axis` from the STFT
    /// coefficients in `input` by overlap-add with the dual window.
    ///
    /// `axis` names the sample axis of `output`, and `input` has the
    /// coefficient shape that [`StftPlan::analyze_axis_into`] writes for
    /// sample arrays of the shape of `output`. Each lane is reconstructed as
    /// [`StftPlan::synthesize_into`] reconstructs a record. `ws` needs the
    /// size of [`StftPlan::workspace_requirements`] for [`StftRequest::Axis`]
    /// with the shape of `output`; with it the call does not allocate. The
    /// samples are staged in `ws` and written to `output` after every lane
    /// succeeds. The cost is that of [`StftPlan::synthesize_into`] per lane.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the plan has no dual
    ///   window.
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the rank of
    ///   `output`.
    /// - `DimensionMismatch` if `input` does not have the coefficient shape,
    ///   or `ws` is too short.
    /// - The errors of [`StftPlan::synthesize_into`] for each lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{StftOptions, StftPlan, StftRequest};
    /// let plan = StftPlan::new(&[1.0_f64; 4], 2, StftOptions::new())?;
    /// // The coefficients of two six-sample records, one record per row.
    /// let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    /// let coefficients = [plan.analyze(&x[..6])?, plan.analyze(&x[6..])?].concat();
    /// let (s, cx) = ([2, 6], ExecutionContext::sequential_native());
    /// let mut ws = plan.workspace(StftRequest::Axis { shape: &s, axis: 1 })?;
    /// let layout = StridedLayout::new(&[2, 4, 4], &[16, 4, 1])?;
    /// let input = NdView::new(&coefficients, layout)?;
    /// let mut y = [0.0; 12];
    /// let output = NdViewMut::new(&mut y, StridedLayout::new(&s, &[6, 1])?)?;
    /// plan.synthesize_axis_into(input, 1, output, &mut ws, cx)?;
    /// assert!(y.iter().zip(x).all(|(a, b)| (a - b).abs() < 1e-12));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn synthesize_axis_into(
        &self,
        input: NdView<'_, '_, Complex<T::Real>>,
        axis: usize,
        mut output: NdViewMut<'_, '_, T>,
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.require_dual()?;
        let shape = output.layout().shape().extents();
        let AxisLayout {
            geometry,
            frames,
            n,
            samples,
            lane_output,
            requirements,
        } = self.axis_layout(shape, axis)?;
        self.check_coefficient_shape(shape, axis, frames.count, input.layout().shape().extents())?;
        let buffers = ws.buffers(&requirements)?;
        cx.check()?;
        let (staged, lane) = buffers.values.split_at_mut(samples);
        let (data, rest) = buffers.complex.split_at_mut(geometry.output_len);
        for (slot, &v) in data.iter_mut().zip(input.iter()) {
            *slot = v;
        }
        finite(data)?;
        let (coefficients, rest) = rest.split_at_mut(lane_output);
        let (record, scratch) = rest.split_at_mut(n);
        for batch in 0..geometry.batches {
            cx.check()?;
            geometry.gather_dense(data, coefficients, batch)?;
            self.synthesize_staged(coefficients, frames, record, buffers.real, scratch, cx)?;
            for (slot, z) in lane.iter_mut().zip(record.iter()) {
                *slot = T::from_components(z.re, z.im);
            }
            geometry.scatter(lane, batch, staged)?;
        }
        cx.check()?;
        crate::axis::publish(staged, &mut output);
        Ok(())
    }

    /// Whether an imaginary part is negligible against its nonnegative scale:
    /// `abs(imaginary) / scale <= tolerance.absolute() + tolerance.relative()`.
    /// The ratio is dimensionless, so the decision is scale invariant.
    fn is_real(&self, imaginary: T::Real, scale: T::Real) -> Result<bool, SciOxideError> {
        if imaginary == T::Real::ZERO {
            return Ok(true);
        }
        self.frame
            .options
            .tolerance
            .is_residual_ok(imaginary.abs() / scale, T::Real::ONE)
    }
    /// Require a real reconstruction of the caller's `input`; `scales` holds
    /// each sample's sum of overlapping frame contribution scales.
    fn validate_real(
        &self,
        values: &[Complex<T::Real>],
        scales: &[T::Real],
    ) -> Result<(), SciOxideError> {
        finite(values)?;
        if T::IS_REAL {
            for (z, &scale) in values.iter().zip(scales) {
                if !self.is_real(z.im, scale)? {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("result"),
                        scioxide_core::DomainReason::OutsideDomain,
                        "coefficients do not reconstruct a real signal",
                    ));
                }
            }
        }
        Ok(())
    }
    /// Computes the power spectrogram `|X[p, k]|²` of `input` and writes it to
    /// `output`, frame by frame.
    ///
    /// The coefficients `X` carry the plan's scaling, so with
    /// [`StftScaling::Psd`] each frame is a power spectral density, and with
    /// one-sided packing its bins other than DC and Nyquist are doubled.
    /// `output` needs at least [`StftPlan::output_len`] elements, and `ws` the
    /// size for [`StftRequest::Spectrogram`] of `input.len()`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` or `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a power overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn spectrogram_into(
        &self,
        input: &[T],
        output: &mut [T::Real],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let frames = self.frames(input.len())?;
        let n = self.output_len(input.len())?;
        capacity(n, output.len())?;
        let buffers =
            ws.buffers(&self.workspace_requirements(StftRequest::Spectrogram(input.len()))?)?;
        finite(input)?;
        cx.check()?;
        let (coefficients, fft) = buffers.complex.split_at_mut(n);
        for (i, out) in coefficients.chunks_exact_mut(self.bins()).enumerate() {
            let start = self.start(frames.first + i as i64);
            let exponent = self.frame.analyze_frame_scaled(
                |j| self.sample(input, start + j as i128),
                out,
                fft,
                cx,
            )?;
            for z in out.iter_mut() {
                let power = frame::weighted_product(*z, *z, T::Real::ONE, 2 * exponent)?.re;
                *z = Complex::new(power, T::Real::ZERO);
            }
        }
        cx.check()?;
        for (slot, z) in output.iter_mut().zip(coefficients.iter()) {
            *slot = z.re;
        }
        Ok(())
    }
    /// Computes the cross-spectrogram `X[p, k] conj(Y[p, k])` of `x` and `y` and
    /// writes it to `output`, frame by frame.
    ///
    /// `X` and `Y` are the STFT coefficients of `x` and `y`, which must have
    /// equal lengths. `output` needs at least [`StftPlan::output_len`]
    /// elements, and `ws` the size for [`StftRequest::CrossSpectrogram`] of
    /// `x.len()`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` and `y` differ in length, or `output` or
    ///   `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a product overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn cross_spectrogram_into(
        &self,
        x: &[T],
        y: &[T],
        output: &mut [Complex<T::Real>],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        exact_len(x.len(), y.len())?;
        let frames = self.frames(x.len())?;
        let n = self.output_len(x.len())?;
        capacity(n, output.len())?;
        let buffers =
            ws.buffers(&self.workspace_requirements(StftRequest::CrossSpectrogram(x.len()))?)?;
        finite(x)?;
        finite(y)?;
        cx.check()?;
        let (a, rest) = buffers.complex.split_at_mut(n);
        let (b, fft) = rest.split_at_mut(self.bins());
        for (i, out) in a.chunks_exact_mut(self.bins()).enumerate() {
            let start = self.start(frames.first + i as i64);
            let alpha = self.frame.analyze_frame_scaled(
                |j| self.sample(x, start + j as i128),
                out,
                fft,
                cx,
            )?;
            let beta = self.frame.analyze_frame_scaled(
                |j| self.sample(y, start + j as i128),
                b,
                fft,
                cx,
            )?;
            for (u, &v) in out.iter_mut().zip(b.iter()) {
                *u = frame::weighted_product(*u, v, T::Real::ONE, alpha + beta)?;
            }
        }
        cx.check()?;
        output[..n].copy_from_slice(a);
        Ok(())
    }
    /// Computes STFT coefficients after removing a trend from each frame, and
    /// writes them to `output`, frame by frame.
    ///
    /// The `m` samples of each frame, including those of the boundary
    /// extension, are detrended with `kind` before windowing. The trends are
    /// not kept, so synthesis from these coefficients does not restore the
    /// record. `output` needs at least [`StftPlan::output_len`] elements, and
    /// `ws` the size for [`StftRequest::DetrendedAnalysis`] of `input.len()`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` or `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a coefficient overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn analyze_detrended_into(
        &self,
        input: &[T],
        kind: crate::DetrendKind,
        output: &mut [Complex<T::Real>],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let frames = self.frames(input.len())?;
        let n = self.output_len(input.len())?;
        let m = self.frame.window.len();
        capacity(n, output.len())?;
        let SignalBuffers {
            values, complex, ..
        } =
            ws.buffers(&self.workspace_requirements(StftRequest::DetrendedAnalysis(input.len()))?)?;
        finite(input)?;
        cx.check()?;
        let (source, rest) = values.split_at_mut(m);
        let (detrended, scratch) = rest.split_at_mut(m);
        let (staged, fft) = complex.split_at_mut(n);
        for (i, out) in staged.chunks_exact_mut(self.bins()).enumerate() {
            let start = self.start(frames.first + i as i64);
            for (j, v) in source.iter_mut().enumerate() {
                *v = self.sample(input, start + j as i128);
            }
            crate::detrend::detrend_staged(source, kind, &[], detrended, scratch)?;
            self.analyze_frame(|j| detrended[j], out, fft, cx)?;
        }
        cx.check()?;
        output[..n].copy_from_slice(staged);
        Ok(())
    }
}
/// One frame's window-length synthesis contribution and its largest component
/// magnitude.
type FrameContribution<'a, S> = (&'a [Complex<S>], S);
/// Add one frame contribution `value`, whose frame scale is `frame_scale`, to
/// a reconstructed sample and its saturating reconstruction scale.
fn overlap_add<S: RealScalar>(
    sample: &mut Complex<S>,
    scale: &mut S,
    value: Complex<S>,
    frame_scale: S,
) -> Result<(), SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    *sample = crate::util::working_sum_products(
        [
            (*sample, Complex::new(S::ONE, S::ZERO)),
            (value, Complex::new(S::ONE, S::ZERO)),
        ]
        .into_iter(),
    )?;
    *scale = (*scale + frame_scale).min(S::MAX);
    Ok(())
}
pub use stream::{StftAnalysisStream, StftSynthesisStream};
