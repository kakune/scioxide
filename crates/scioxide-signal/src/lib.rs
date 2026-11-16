//! Digital signal processing: filtering, filter design, resampling, spectral
//! analysis, time–frequency transforms and linear time-invariant systems.
//!
//! The crate convolves and correlates signals, designs FIR and IIR filters and
//! applies them to records or streams, and converts sample rates. It estimates
//! spectra with periodograms, averaged periodograms and the Lomb–Scargle
//! periodogram, computes short-time Fourier and discrete wavelet transforms,
//! and forms analytic signals. It also smooths and denoises data, detects peaks,
//! generates windows, tapers and test waveforms, and simulates dense
//! continuous- and discrete-time state-space systems. Samples are real or
//! complex `f32` or `f64`. Transforms use the plans of `scioxide-fft`.
//!
//! # Quick start
//!
//! ```
//! use scioxide_signal::{IirBand, IirDomain, IirFamily, design_iir};
//! // Fourth-order Butterworth lowpass with its cutoff at 0.2 times the Nyquist frequency.
//! let band = IirBand::Lowpass(0.2_f64);
//! let sos = design_iir(4, IirFamily::Butterworth, band, IirDomain::Digital)?.to_sos()?;
//! // The DC gain is 1, so the causal and the zero-phase step responses settle at 1.
//! let y = sos.filter(&[1.0_f64; 200])?;
//! let z = sos.forward_backward(&[1.0_f64; 200])?;
//! assert!((y[199] - 1.0).abs() < 1e-9 && (z[100] - 1.0).abs() < 1e-9);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Task | Entry points | Method |
//! |---|---|---|
//! | Convolution and correlation of two records | [`convolve`], [`correlate`], [`convolve_circular`], [`convolve2d`] | direct sum or FFT, chosen by operation count |
//! | Repeated N-dimensional convolution with one kernel | [`ConvolutionNdPlan`], [`SeparableFir2dPlan`] | direct sum or FFT; two 1-D passes for a separable kernel |
//! | Long or streamed FIR filtering | [`OverlapPlan`], [`OverlapStream`] | FFT overlap-add in fixed blocks |
//! | Design an IIR filter | [`design_iir`], [`select_iir_order`], [`iir_prototype`] | Butterworth, Chebyshev I and II, Bessel and elliptic prototypes, band transformation and bilinear transform |
//! | Notch, peak, comb and auditory filters | [`design_notch_peak`], [`design_comb`], [`design_gammatone`], [`design_real_gammatone`] | closed-form second-order and comb designs; gammatone cascades |
//! | Apply an IIR filter | [`SecondOrderSections::filter`], [`SecondOrderSections::forward_backward`], [`TransferFunction::filter`], [`IirFilter`] | cascaded second-order sections or a transfer function; zero-phase forward–backward filtering |
//! | Design a linear-phase FIR filter | [`design_fir_window`], [`design_fir_least_squares`], [`design_fir_frequency_sampling`], [`design_fir_remez`] | window method, weighted least squares, frequency sampling, Parks–McClellan (Remez exchange) |
//! | Minimum-phase FIR from a linear-phase one | [`minimum_phase`], [`MinimumPhasePlan`] | cepstral (homomorphic) or discrete Hilbert method |
//! | Rational sample-rate conversion | [`resample`], [`PolyphasePlan`], [`PolyphaseStream`] | polyphase FIR filtering with upsampling and downsampling |
//! | Periodic band-limited resampling | [`resample_periodic`], [`FourierResamplePlan`] | zero padding or truncation of the DFT |
//! | Window functions and tapers | [`window`], [`Window`], [`window_metrics`], [`DpssTapers`] | closed-form windows; discrete prolate spheroidal sequences from a tridiagonal eigenproblem |
//! | Power spectrum, Welch estimate, cross spectrum, coherence | [`power_spectrum`], [`cross_spectrum`], [`welch`], [`SpectralEstimatePlan`] | periodogram; averaged windowed periodograms with mean or median |
//! | Spectrum of unevenly sampled data | [`lomb_scargle`], [`LombScargle`] | least-squares fit of a sinusoid at each frequency |
//! | Short-time Fourier transform and its inverse | [`StftPlan`], [`StftAnalysisStream`], [`StftSynthesisStream`] | windowed FFT frames; overlap-add synthesis with a dual window |
//! | Discrete wavelet transform | [`WaveletPlan`], [`Wavelet2dPlan`] | Mallat's pyramid algorithm with Haar, Daubechies and biorthogonal spline filters |
//! | Analytic signal, envelope, Riesz transform | [`hilbert`], [`envelope`], [`AnalyticSignalNdPlan`], [`RieszPlan`], [`monogenic_nd`] | frequency-domain multipliers |
//! | Median, rank, Wiener and Hampel filters | [`median_filter`], [`LocalFilterPlan`], [`LocalFilterStream`] | sliding-window order statistics and local moments |
//! | Smoothing and differentiation of noisy data | [`savitzky_golay`], [`SavitzkyGolayPlan`], [`WhittakerHendersonPlan`] | local polynomial least squares; difference-penalized least squares |
//! | B-spline coefficients, zero-phase recursive filters | [`SplineFilter`], [`SymmetricFilter`] | forward and backward first- or second-order recursions |
//! | Peaks and their prominence and width | [`find_peaks`], [`find_peaks_with_options`], [`peak_properties`], [`RidgePlan`] | local maxima; ridge lines of a Ricker wavelet transform |
//! | State-space systems | [`StateSpace`], [`FrequencyResponsePlan`], [`TimeResponsePlan`], [`StateSpace::place_poles`] | Hessenberg-form frequency responses; zero-order-hold simulation; pole placement by state feedback |
//! | Detrending, phase locking, test signals | [`detrend`], [`vector_strength`], [`waveform`], [`unit_impulse_into`] | least-squares trend removal; closed-form waveforms |
//!
//! # Conventions
//!
//! - Frequencies of filters and systems are in radians per time unit for
//!   continuous-time (analog) systems, evaluated at `s = iω`. For discrete-time
//!   (digital) systems they are multiples of the Nyquist frequency, so `f = 1` is
//!   half the sample rate, evaluated at `z = exp(iπf)`. Transfer functions,
//!   pole–zero–gain forms, second-order sections, FIR designs and state-space
//!   responses all follow this rule, so they agree at the same frequency.
//! - STFT bins, spectral-estimate bins and waveform frequencies are in cycles per
//!   time unit. Lomb–Scargle frequencies are angular, in radians per time unit.
//! - Linear convolution of lengths `m` and `n` has length `m + n − 1`. Circular
//!   convolution requires equal lengths and keeps them.
//! - Correlation is `c[ℓ] = Σₖ a[k] conj(b[k − ℓ])`. The full output lists the
//!   lags `ℓ = −(n − 1), …, m − 1` in ascending order, so lag zero is at index
//!   `n − 1`.
//! - The discrete power spectrum of `x` of length `n` is `|X[k]|² / n` and the
//!   discrete cross spectrum is `A[k] conj(B[k]) / n`, where `X`, `A`, `B` are
//!   unnormalized DFTs. They are not densities: the sample rate does not enter.
//! - [`hilbert`] returns the analytic signal `x + i H[x]`. Its spectrum keeps
//!   the DC bin and, for even lengths, the Nyquist bin, doubles the positive
//!   frequencies and removes the negative ones.
//! - Forward transforms are unnormalized and inverse transforms divide by the
//!   transform length.
//!
//! # Accuracy
//!
//! Let `u = ε/2` be the unit roundoff of the scalar type,
//! `γ(k) = k u / (1 − k u)`, and `η` the smallest positive subnormal number.
//!
//! FIR filtering, polyphase resampling, Savitzky–Golay filtering, direct
//! convolution and correlation, and wavelet filtering form sums of products in
//! working precision. For a sum of `m` products, each real or imaginary
//! component of the result has absolute error at most
//!
//! ```text
//! γ(2m + 8) · M + (2m + 8) · η,
//! ```
//!
//! where `M` is the sum of the absolute values of the products. Across wavelet
//! levels the error of each level propagates through the absolute filter
//! coefficients. These bounds concern the stored filter coefficients; filter
//! design has its own error.
//!
//! FFT-based convolution, analytic and Riesz transforms and Fourier resampling
//! have a normwise error: proportional to `ε` times the operand norms and the
//! logarithm of the transform length, plus underflow. Components much smaller
//! than the largest one can therefore lose all relative accuracy or round to
//! zero. Inputs are rescaled by powers of two when needed, so finite results do
//! not overflow or underflow prematurely. A result that overflows is an error.
//!
//! The error of a filter or system design depends on the conditioning of its
//! poles and zeros. Items with a different bound state it.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: an option is out of range, a band edge or cutoff lies
//!   outside its domain, an input that must not be empty is empty, or a stream
//!   is called after its final chunk (reason `Inconsistent`).
//! - `DimensionMismatch`: lengths or shapes disagree with the plan or with each
//!   other, or an output or workspace is too short.
//! - `NonFinite`: a sample, coefficient or design parameter is NaN or infinite.
//! - `DomainError`: a normalization divides by a zero sum or zero energy,
//!   coherence meets a zero auto-spectrum, the minimum phase of a zero signal is
//!   requested, or a hyperbolic chirp reaches its pole.
//! - `NumericalInstability`: a result overflows, a filter gain underflows, or a
//!   self-check of a computed result fails (for example a dual window or a
//!   pole–zero conversion).
//! - `ResourceLimit`: a workspace exceeds the byte limit set with
//!   [`SignalRequirements::with_max_bytes`].
//! - `ConvergenceFailure`: an order selection, a Remez exchange or a pole
//!   placement of a one-shot form misses its target; see
//!   [Scoped designs](#scoped-designs).
//! - `ExecutionStopped`: a `_scoped` design is cancelled or passes its
//!   deadline.
//!
//! On error, outputs and stream states are unchanged, as described in
//! [Output on error](scioxide_core#output-on-error).
//!
//! # Reusing plans and workspaces
//!
//! Plans follow the [shared rules](scioxide_core#plans-and-workspaces). A plan
//! checks its options and sizes when it is created with `new`, or with
//! `build(…, cx)` when construction needs an
//! [execution context](scioxide_core#execution-contexts). Plans are immutable,
//! `Send + Sync + Clone`, and keep no global caches.
//!
//! [`IirFilter`], [`SymmetricFilter`], [`SplineFilter`] and [`LombScargle`]
//! are method values: they hold validated coefficients or settings, compute
//! nothing at construction and offer the same execution forms as plans.
//! [`Window`] and [`DpssTapers`] hold computed samples.
//!
//! Most plans report their scratch needs as a [`SignalRequirements`] from
//! `workspace_requirements(request)`. The request is `()`, a length, a lane
//! count, a [`SignalLayout`] or a plan-specific value. A [`SignalWorkspace`]
//! reserved with `workspace(request)` or `try_reserve` then serves every call
//! of at most that size. Peak finding, ridge detection, Whittaker smoothing and
//! LTI responses use their own workspace types with the same `try_reserve` and
//! `check` methods.
//!
//! Each operation has a form `verb_into(inputs, output, workspace, cx)` that
//! writes into an output you provide and takes the execution context last. It
//! writes only the prefix of `output` that it produces. Where the result type is
//! unique there is also an owned form `verb(inputs)` that allocates its result
//! and runs with
//! [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
//! The one-shot functions of the method table, such as [`welch`] or
//! [`SecondOrderSections::forward_backward`], take plain data, use the default
//! options and the conventional settings they document, and run the same way.
//!
//! An operation along one axis of an N-dimensional array is the method
//! `verb_axis_into(input, axis, output, workspace, cx)` of the plan that serves
//! single records. It computes shapes and lane counts from the input view on
//! each call, and its workspace request carries the layout
//! ([`SignalLayout::Axis`]) or the lane count.
//! Options select an [`ExecutionPolicy`](scioxide_core::ExecutionPolicy), which
//! defaults to `Sequential`; plans whose only option is the policy take
//! [`SignalExecutionOptions`].
//!
//! # Streams
//!
//! [`OverlapStream`], [`PolyphaseStream`], [`LocalFilterStream`],
//! [`StftAnalysisStream`] and [`StftSynthesisStream`] process a record in
//! chunks, and an [`IirFilter`] carries its delays in an [`IirState`]. A
//! stream is a value: `clone` saves its state and `clone_from` restores it
//! without allocating. Each call passes a [`StreamChunk`].
//! [`StreamChunk::Final`] flushes the pending output and closes the stream
//! until `reset`. A workspace sized by the stream's
//! `workspace_requirements(max)` serves every later call with at most `max`
//! input samples. Calls report their counts in a [`StreamProgress`].
//!
//! # Scoped designs
//!
//! [`design_iir`], [`iir_prototype`], [`select_iir_order`], [`design_fir_remez`],
//! [`design_real_gammatone`], the conversions [`TransferFunction::to_zpk`],
//! [`SecondOrderSections::to_zpk`] and [`StateSpace::to_zpk`], and
//! [`StateSpace::place_poles`] solve for roots, eigenvalues or extremal points
//! iteratively. The plain forms run in a fresh root scope with
//! [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
//! Each has a `_scoped` form, such as [`design_iir_scoped`], that takes an
//! execution context or the [`SolveContext`](scioxide_core::SolveContext) of an
//! enclosing solver as its last argument. It runs as one epoch of that scope,
//! observes its cancellation and deadline, and its work counts toward the
//! scope's limits as described in
//! [Solve accounting](scioxide_core#solve-accounting).
//!
//! [`select_iir_order`], [`design_fir_remez`] and [`StateSpace::place_poles`]
//! return `ConvergenceFailure` when the order selection, the Remez exchange or
//! the placement misses its target. Their `_with_options` and `_scoped` forms
//! return the last result instead, with the outcome in [`IirOrderTermination`],
//! [`FirTermination`] or [`PolePlacementTermination`].
//!
//! # Feature flags
//!
//! - `std` (default): threads and clocks from `scioxide-core`. Without it the
//!   crate is `no_std` and needs `alloc`; the API is the same.
//!
//! # References
//!
//! - A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
//!   3rd ed., Pearson, 2010.
//! - J. H. McClellan, T. W. Parks and L. R. Rabiner, "A computer program for
//!   designing optimum FIR linear phase digital filters", *IEEE Trans. Audio
//!   Electroacoust.* 21(6), 1973.
//! - P. D. Welch, "The use of fast Fourier transform for the estimation of power
//!   spectra", *IEEE Trans. Audio Electroacoust.* 15(2), 1967.
//! - J. D. Scargle, "Studies in astronomical time series analysis. II",
//!   *Astrophys. J.* 263, 1982.
//! - S. G. Mallat, *A Wavelet Tour of Signal Processing*, 3rd ed., Academic
//!   Press, 2009.
//! - F. J. Harris, "On the use of windows for harmonic analysis with the
//!   discrete Fourier transform", *Proc. IEEE* 66(1), 1978.
//! - D. Slepian, "Prolate spheroidal wave functions, Fourier analysis, and
//!   uncertainty — V: The discrete case", *Bell Syst. Tech. J.* 57(5), 1978.
#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg_attr(test, macro_use)]
extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;
mod workspace;
pub use workspace::{
    SignalExecutionOptions, SignalLayout, SignalRequirements, SignalWorkspace, StreamChunk,
};
mod iir;
pub use iir::{
    BesselNormalization, ForwardBackwardOptions, IirBand, IirDesign, IirDomain, IirFamily,
    IirFilter, IirInitialization, IirOrderFamily, IirOrderOptions, IirOrderReport,
    IirOrderTermination, IirPadding, IirRequest, IirState, PoleZeroGain, ResonanceKind,
    SecondOrderSections, TransferFunction, deconvolve, design_comb, design_gammatone, design_iir,
    design_iir_scoped, design_notch_peak, design_real_gammatone, design_real_gammatone_scoped,
    iir_prototype, iir_prototype_scoped, select_iir_order, select_iir_order_scoped,
    select_iir_order_with_options,
};
mod lti;
pub use lti::{
    DiscretizationMethod, FrequencyResponse, FrequencyResponsePlan, FrequencyResponseRequirements,
    FrequencyResponseWorkspace, LtiTime, PolePlacementOptions, PolePlacementReport,
    PolePlacementTermination, StateSpace, StateSpaceView, TimeResponse, TimeResponsePlan,
    TimeResponseWorkspace,
};
mod fir;
mod minimum_phase;
pub use fir::{
    FirBand, FirDesign, FirNormalization, FirSymmetry, FirTermination, RemezOptions,
    design_fir_frequency_sampling, design_fir_least_squares, design_fir_remez,
    design_fir_remez_scoped, design_fir_remez_with_options, design_fir_window,
};
pub use minimum_phase::{
    MinimumPhaseConvention, MinimumPhaseDiagnostics, MinimumPhaseMethod, MinimumPhaseOptions,
    MinimumPhasePlan, SpectralZeroPolicy, minimum_phase, minimum_phase_with_options,
};
mod convolution;
pub use convolution::separable::SeparableFir2dPlan;
mod fourier;
mod fourier_resampling;
mod resampling;
mod streaming;
pub use convolution::nd::{
    ConvolutionExtent, ConvolutionKernel, ConvolutionMethod, ConvolutionNdPlan, ConvolutionOptions,
    ConvolutionSelection, SignalBoundary, SignalProduct,
};
pub use convolution::stream::{OverlapPlan, OverlapStream};
pub use fourier_resampling::{FourierResamplePlan, resample_periodic};
pub use resampling::{PolyphasePlan, PolyphaseStream, ResampleAlignment, resample};
pub use streaming::StreamProgress;
mod dpss;
mod filters;
mod hilbert;
mod multidimensional;
pub use filters::{
    LocalFilter, LocalFilterPlan, LocalFilterStream, estimate_wiener_noise, median_filter,
};
pub use multidimensional::{AnalyticSignalNdPlan, RieszPlan, monogenic_nd, riesz_nd};
mod detrend;
mod phase;
mod waveforms;
pub use detrend::{DetrendKind, detrend, detrend_into};
pub use phase::{VectorStrength, vector_strength, vector_strength_into};
pub use waveforms::{
    ChirpLaw, Waveform, unit_impulse_into, waveform, waveform_grid_into, waveform_into,
};
mod peaks;
mod ridges;
pub use peaks::{
    Peak, PeakOptions, PeakRequirements, PeakWorkspace, find_peaks, find_peaks_into,
    find_peaks_with_options, peak_properties, peak_properties_into,
};
pub use ridges::{RidgeOptions, RidgePeak, RidgePlan, RidgeRequirements, RidgeWorkspace};
mod axis;
mod util;
mod windows;
pub use convolution::{convolve, convolve_circular, convolve2d, correlate};
pub use dpss::{DpssOptions, DpssReport, DpssTapers, DpssTermination};
pub use hilbert::{complex_envelope_into, envelope, hilbert};
pub use windows::{
    Window, WindowKind, WindowMetrics, WindowNormalization, WindowSymmetry, window, window_metrics,
};
mod spline_filters;
pub use spline_filters::{SplineDegree, SplineFilter, SplineRequest, SymmetricFilter};
mod whittaker;
pub use whittaker::{
    SmoothingConditioning, SmoothingDiagnostics, WhittakerHendersonPlan, WhittakerRequirements,
    WhittakerWorkspace,
};
mod stft;
pub use stft::{
    ClosestStftDual, DualScale, OverlapDiagnostics, StftFrames, StftOptions, StftPacking, StftPlan,
    StftRequest, StftScaling, canonical_stft_dual, check_overlap, closest_stft_dual,
    tight_stft_window,
};
pub use stft::{StftAnalysisStream, StftSynthesisStream};

mod spectra;
pub use spectra::{
    LombScargle, LombScargleMean, LombScargleNormalization, LombScargleOptions, SpectralAverage,
    SpectralEstimateOptions, SpectralEstimatePlan, SpectralScaling, cross_spectrum, lomb_scargle,
    power_spectrum, welch,
};
mod savgol;
pub use savgol::{SavitzkyGolayBoundary, SavitzkyGolayOptions, SavitzkyGolayPlan, savitzky_golay};

mod wavelets;
pub use wavelets::{
    Wavelet2dLayout, Wavelet2dPlan, WaveletFamily, WaveletFilters, WaveletLevel, WaveletOptions,
    WaveletPhase, WaveletPlan, WaveletSubband, wavelet_max_level,
};
