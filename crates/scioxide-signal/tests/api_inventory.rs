//! The authoritative scioxide-signal root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test
//! fails when this list and the `pub use` items of `src/lib.rs` differ. Exported
//! plans and streams are asserted `Send + Sync + Clone`, and each family's
//! requirements, workspace and execution forms are pinned by signature.
#![allow(clippy::type_complexity)]
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_signal::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    // Shared workspace protocol, execution options and request vocabulary.
    SignalExecutionOptions, SignalLayout, SignalRequirements, SignalWorkspace, StreamChunk,
    StreamProgress,
    // IIR designs, representations and prepared filtering.
    BesselNormalization, ForwardBackwardOptions, IirBand, IirDesign, IirDomain, IirFamily,
    IirFilter, IirInitialization, IirOrderFamily, IirOrderOptions, IirOrderReport,
    IirOrderTermination, IirPadding, IirRequest, IirState, PoleZeroGain, ResonanceKind,
    SecondOrderSections, TransferFunction, deconvolve, design_comb, design_gammatone, design_iir,
    design_iir_scoped, design_notch_peak, design_real_gammatone, design_real_gammatone_scoped,
    iir_prototype, iir_prototype_scoped, select_iir_order, select_iir_order_scoped,
    select_iir_order_with_options,
    // Dense continuous and discrete LTI systems and prepared responses.
    DiscretizationMethod, FrequencyResponse, FrequencyResponsePlan, FrequencyResponseRequirements,
    FrequencyResponseWorkspace, LtiTime, PolePlacementOptions, PolePlacementReport,
    PolePlacementTermination, StateSpace, StateSpaceView, TimeResponse, TimeResponsePlan,
    TimeResponseWorkspace,
    // FIR designs and minimum-phase reconstruction.
    FirBand, FirDesign, FirNormalization, FirSymmetry, FirTermination, RemezOptions,
    design_fir_frequency_sampling, design_fir_least_squares, design_fir_remez,
    design_fir_remez_scoped, design_fir_remez_with_options, design_fir_window,
    MinimumPhaseConvention, MinimumPhaseDiagnostics, MinimumPhaseMethod, MinimumPhaseOptions,
    MinimumPhasePlan, SpectralZeroPolicy, minimum_phase, minimum_phase_with_options,
    // One convolution engine, overlap streaming and resampling.
    ConvolutionExtent, ConvolutionKernel, ConvolutionMethod, ConvolutionNdPlan, ConvolutionOptions,
    ConvolutionSelection, SignalBoundary, SignalProduct, SeparableFir2dPlan, OverlapPlan,
    OverlapStream, FourierResamplePlan, PolyphasePlan, PolyphaseStream, ResampleAlignment,
    convolve, convolve_circular, convolve2d, correlate, resample, resample_periodic,
    // Local filters, smoothing and spline filters.
    LocalFilter, LocalFilterPlan, LocalFilterStream, estimate_wiener_noise, median_filter,
    SavitzkyGolayBoundary, SavitzkyGolayOptions, SavitzkyGolayPlan, savitzky_golay,
    SplineDegree, SplineFilter, SplineRequest, SymmetricFilter,
    SmoothingConditioning, SmoothingDiagnostics, WhittakerHendersonPlan, WhittakerRequirements,
    WhittakerWorkspace,
    // Features: detrending, waveforms, phase locking, peaks and ridges.
    DetrendKind, detrend, detrend_into, VectorStrength, vector_strength, vector_strength_into,
    ChirpLaw, Waveform, unit_impulse_into, waveform, waveform_grid_into, waveform_into, Peak,
    PeakOptions, PeakRequirements, PeakWorkspace, find_peaks, find_peaks_into,
    find_peaks_with_options, peak_properties, peak_properties_into, RidgeOptions, RidgePeak,
    RidgePlan, RidgeRequirements, RidgeWorkspace,
    // Analytic signals and multidimensional Riesz transforms.
    AnalyticSignalNdPlan, RieszPlan, monogenic_nd, riesz_nd, complex_envelope_into, envelope,
    hilbert,
    // Windows and DPSS tapers.
    DpssOptions, DpssReport, DpssTapers, DpssTermination, Window, WindowKind, WindowMetrics,
    WindowNormalization, WindowSymmetry, window, window_metrics,
    // STFT analysis, synthesis, duals and streams.
    ClosestStftDual, DualScale, OverlapDiagnostics, StftAnalysisStream, StftFrames, StftOptions,
    StftPacking, StftPlan, StftRequest, StftScaling, StftSynthesisStream, canonical_stft_dual,
    check_overlap, closest_stft_dual, tight_stft_window,
    // Spectral estimation and Lomb–Scargle.
    LombScargle, LombScargleMean, LombScargleNormalization, LombScargleOptions, SpectralAverage,
    SpectralEstimateOptions, SpectralEstimatePlan, SpectralScaling, cross_spectrum,
    power_spectrum,
    welch, lomb_scargle,
    // Discrete wavelets.
    Wavelet2dLayout, Wavelet2dPlan, WaveletFamily, WaveletFilters, WaveletLevel, WaveletOptions,
    WaveletPhase, WaveletPlan, WaveletSubband, wavelet_max_level,
}

/// Asserts the plan auto traits for each listed instantiation and
/// records the plan and stream names for the completeness test.
macro_rules! plans {
    ($($name:ident $(<$($argument:tt),*>)?),* $(,)?) => {
        const _: fn() = || {
            fn plan<T: Send + Sync + Clone>() {}
            $(plan::<$name $(<$($argument),*>)?>();)*
        };
        const PLANS: &[&str] = &[$(stringify!($name)),*];
    };
}

plans! {
    AnalyticSignalNdPlan<f64>, ConvolutionNdPlan<f64>, FourierResamplePlan<f64>,
    FrequencyResponsePlan<f64>, LocalFilterPlan<f64>, MinimumPhasePlan<f64>, OverlapPlan<f64>,
    PolyphasePlan<f64>, RidgePlan<f64>, RieszPlan<f64>, SavitzkyGolayPlan<f64>,
    SeparableFir2dPlan<f64>, SpectralEstimatePlan<f64>, StftPlan<f64>, TimeResponsePlan<f64>,
    Wavelet2dPlan<f64>, WaveletPlan<f64>, WhittakerHendersonPlan<f64>,
    // Streams are values: `clone` snapshots and `clone_from` restores.
    LocalFilterStream<'static, f64>, OverlapStream<'static, f64>, PolyphaseStream<'static, f64>,
    StftAnalysisStream<'static, f64>, StftSynthesisStream<'static, f64>,
}

// Method values hold validated settings and products hold computed samples;
// neither is a plan, and both share the plan auto traits.
const _: fn() = || {
    fn value<T: Send + Sync + Clone>() {}
    value::<LombScargle<'static, f64>>();
    value::<SymmetricFilter<f64>>();
    value::<SplineFilter<f64>>();
    value::<IirFilter<f64>>();
    value::<Window<f64>>();
    value::<DpssTapers<f64>>();
};

// Complex-sample plans share the plan auto traits.
const _: fn() = || {
    fn plan<T: Send + Sync + Clone>() {}
    type C = scioxide_core::Complex<f64>;
    plan::<ConvolutionNdPlan<C>>();
    plan::<IirFilter<C>>();
    plan::<MinimumPhasePlan<C>>();
    plan::<OverlapPlan<C>>();
    plan::<PolyphasePlan<C>>();
    plan::<SpectralEstimatePlan<C>>();
    plan::<StftPlan<C>>();
    plan::<WaveletPlan<C>>();
    plan::<WhittakerHendersonPlan<C>>();
    plan::<SignalWorkspace<C>>();
};

/// Asserts `Default` for each listed options type and records the names for
/// the completeness test.
macro_rules! options {
    ($($name:ident $(<$($argument:tt),*>)?),* $(,)?) => {
        const _: fn() = || {
            fn options<T: Default>() {}
            $(options::<$name $(<$($argument),*>)?>();)*
        };
        const OPTIONS: &[&str] = &[$(stringify!($name)),*];
    };
}

options! {
    ConvolutionOptions, DpssOptions, ForwardBackwardOptions<'static, f64>, IirOrderOptions,
    LombScargleOptions<f64>, MinimumPhaseOptions<f64>, PeakOptions<f64>,
    PolePlacementOptions<f64>, RemezOptions<f64>, RidgeOptions<f64>, SavitzkyGolayOptions<f64>,
    SignalExecutionOptions, SpectralEstimateOptions<f64>, StftOptions<f64>, WaveletOptions,
}

#[test]
fn inventory_lists_every_root_export_once() {
    let listed: BTreeSet<_> = INVENTORY.iter().copied().collect();
    assert_eq!(listed.len(), INVENTORY.len(), "duplicate inventory entry");
    let root = root_exports(include_str!("../src/lib.rs"));
    let unlisted: Vec<_> = root.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&root).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "update tests/api_inventory.rs\nunlisted: {unlisted:?}\nstale: {stale:?}"
    );
}

#[test]
fn every_exported_plan_and_stream_is_send_sync_clone() {
    let exported: BTreeSet<_> = INVENTORY
        .iter()
        .copied()
        .filter(|name| name.ends_with("Plan") || name.ends_with("Stream"))
        .collect();
    let asserted: BTreeSet<_> = PLANS.iter().copied().collect();
    assert_eq!(
        exported, asserted,
        "update plans! in tests/api_inventory.rs"
    );
}

#[test]
fn every_exported_options_type_has_a_default() {
    let exported: BTreeSet<_> = INVENTORY
        .iter()
        .copied()
        .filter(|name| name.ends_with("Options"))
        .collect();
    let asserted: BTreeSet<_> = OPTIONS.iter().copied().collect();
    assert_eq!(
        exported, asserted,
        "update options! in tests/api_inventory.rs"
    );
}

#[test]
fn root_exports_read_single_and_braced_statements() {
    let source = "mod a;\npub use a::One;\npub use b::{\n    Two, three,\n};\nuse c::Hidden;\npub(crate) use d::Internal;";
    assert_eq!(
        root_exports(source),
        BTreeSet::from(["One", "Two", "three"])
    );
}

/// Names re-exported by the `pub use` statements of a crate root. Each
/// statement re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("\npub use ")
        .skip(1)
        .filter_map(|statement| statement.split_once(';'))
        .flat_map(|(path, _)| {
            path.rsplit_once("::")
                .map_or(path, |(_, names)| names)
                .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
                .split(',')
        })
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

// One requirements protocol: plans report `SignalRequirements` for a
// plan-specific request, the shared workspace reserves and checks them, and
// non-scalar families keep their own workspace with the same protocol.
const _: fn() = || {
    use scioxide_core::{Capacity, SciOxideError};
    type Req = Result<SignalRequirements, SciOxideError>;
    type Ws = Result<SignalWorkspace<f64>, SciOxideError>;
    let _: fn(&mut SignalWorkspace<f64>, &SignalRequirements) -> Result<(), SciOxideError> =
        SignalWorkspace::try_reserve;
    let _: fn(&SignalWorkspace<f64>, &SignalRequirements) -> Result<Capacity, SciOxideError> =
        SignalWorkspace::check;
    let _: fn(SignalRequirements, usize) -> SignalRequirements = SignalRequirements::with_max_bytes;
    let _: fn(&ConvolutionNdPlan<f64>, ConvolutionKernel) -> Req =
        ConvolutionNdPlan::workspace_requirements;
    let _: fn(&ConvolutionNdPlan<f64>, ConvolutionKernel) -> Ws = ConvolutionNdPlan::workspace;
    let _: fn(&SeparableFir2dPlan<f64>, ()) -> Req = SeparableFir2dPlan::workspace_requirements;
    let _: fn(&OverlapStream<'static, f64>, usize) -> Req = OverlapStream::workspace_requirements;
    let _: fn(&PolyphasePlan<f64>, SignalLayout<'static>) -> Req =
        PolyphasePlan::workspace_requirements;
    let _: fn(&PolyphaseStream<'static, f64>, usize) -> Req =
        PolyphaseStream::workspace_requirements;
    let _: fn(&FourierResamplePlan<f64>, ()) -> Req = FourierResamplePlan::workspace_requirements;
    let _: fn(&IirFilter<f64>, IirRequest<'static>) -> Req = IirFilter::workspace_requirements;
    let _: fn(&SavitzkyGolayPlan<f64>, SignalLayout<'static>) -> Req =
        SavitzkyGolayPlan::workspace_requirements;
    let _: fn(&SymmetricFilter<f64>, SignalLayout<'static>) -> Req =
        SymmetricFilter::workspace_requirements;
    let _: fn(&SplineFilter<f64>, SplineRequest<'static>) -> Req =
        SplineFilter::workspace_requirements;
    let _: fn(&LocalFilterPlan<f64>, usize) -> Req = LocalFilterPlan::workspace_requirements;
    let _: fn(&LocalFilterStream<'static, f64>, usize) -> Req =
        LocalFilterStream::workspace_requirements;
    let _: fn(&StftPlan<f64>, StftRequest<'static>) -> Req = StftPlan::workspace_requirements;
    let _: fn(&StftPlan<f64>, StftRequest<'static>) -> Ws = StftPlan::workspace;
    let _: fn(&StftAnalysisStream<'static, f64>, usize) -> Req =
        StftAnalysisStream::workspace_requirements;
    let _: fn(&StftSynthesisStream<'static, f64>, usize) -> Req =
        StftSynthesisStream::workspace_requirements;
    // Records of any length take a layout; fixed-length records take a lane count.
    let _: fn(&SpectralEstimatePlan<f64>, SignalLayout<'static>) -> Req =
        SpectralEstimatePlan::workspace_requirements;
    let _: fn(&SpectralEstimatePlan<f64>, SignalLayout<'static>) -> Ws =
        SpectralEstimatePlan::workspace;
    let _: fn(&LombScargle<'static, f64>, usize) -> Req = LombScargle::workspace_requirements;
    let _: fn(&LombScargle<'static, f64>, usize) -> Ws = LombScargle::workspace;
    let _: fn(&WaveletPlan<f64>, usize) -> Req = WaveletPlan::workspace_requirements;
    let _: fn(&WaveletPlan<f64>, usize) -> Ws = WaveletPlan::workspace;
    let _: fn(&Wavelet2dPlan<f64>, ()) -> Req = Wavelet2dPlan::workspace_requirements;
    let _: fn(&AnalyticSignalNdPlan<f64>, ()) -> Req = AnalyticSignalNdPlan::workspace_requirements;
    let _: fn(&RieszPlan<f64>, ()) -> Req = RieszPlan::workspace_requirements;
    let _: fn(&MinimumPhasePlan<f64>, ()) -> Req = MinimumPhasePlan::workspace_requirements;
    // Family workspaces for records, indices and banded solver storage.
    let _: fn(usize) -> PeakRequirements = PeakWorkspace::<f64>::workspace_requirements;
    let _: fn(&mut PeakWorkspace<f64>, &PeakRequirements) -> Result<(), SciOxideError> =
        PeakWorkspace::try_reserve;
    let _: fn(&PeakWorkspace<f64>, &PeakRequirements) -> Result<Capacity, SciOxideError> =
        PeakWorkspace::check;
    let _: fn(&RidgePlan<f64>, usize) -> Result<RidgeRequirements, SciOxideError> =
        RidgePlan::workspace_requirements;
    let _: fn(&RidgePlan<f64>, usize) -> Result<RidgeWorkspace<f64>, SciOxideError> =
        RidgePlan::workspace;
    let _: fn(&WhittakerHendersonPlan<f64>, usize) -> Result<WhittakerRequirements, SciOxideError> =
        WhittakerHendersonPlan::workspace_requirements;
    let _: fn(&mut WhittakerWorkspace<f64>, &WhittakerRequirements) -> Result<(), SciOxideError> =
        WhittakerWorkspace::try_reserve;
    let _: fn(&WhittakerWorkspace<f64>, &WhittakerRequirements) -> Result<Capacity, SciOxideError> =
        WhittakerWorkspace::check;
    let _: fn(&TimeResponsePlan<f64>) -> Result<TimeResponseWorkspace<f64>, SciOxideError> =
        TimeResponsePlan::workspace;
};

// One atomic execution form per operation: inputs, output or state, workspace
// and then the execution context.
const _: fn() = || {
    use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Unit = Result<(), SciOxideError>;
    type V = NdView<'static, 'static, f64>;
    type VMut = NdViewMut<'static, 'static, f64>;
    type W = SignalWorkspace<f64>;
    type C = Complex<f64>;
    let _: fn(&ConvolutionNdPlan<f64>, V, VMut, &mut W, Cx) -> Unit =
        ConvolutionNdPlan::convolve_into;
    let _: fn(&ConvolutionNdPlan<f64>, V, V, VMut, &mut W, Cx) -> Unit =
        ConvolutionNdPlan::convolve_pair_into;
    let _: fn(&SeparableFir2dPlan<f64>, V, VMut, &mut W, Cx) -> Unit =
        SeparableFir2dPlan::filter_into;
    let _: fn(
        &mut OverlapStream<'static, f64>,
        &[f64],
        StreamChunk,
        &mut [f64],
        &mut W,
        Cx,
    ) -> Result<StreamProgress, SciOxideError> = OverlapStream::process_into;
    let _: fn(&PolyphasePlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        PolyphasePlan::resample_into;
    let _: fn(&FourierResamplePlan<f64>, V, VMut, &mut W, Cx) -> Unit =
        FourierResamplePlan::resample_into;
    let _: fn(&IirFilter<f64>, &[f64], &mut [f64], &mut IirState<f64>, &mut W, Cx) -> Unit =
        IirFilter::filter_into::<f64>;
    let _: fn(
        &IirFilter<f64>,
        &[f64],
        ForwardBackwardOptions<'static, f64>,
        &mut [f64],
        &mut W,
        Cx,
    ) -> Unit = IirFilter::forward_backward_into::<f64>;
    let _: fn(
        &SavitzkyGolayPlan<f64>,
        &[f64],
        SavitzkyGolayBoundary,
        &mut [f64],
        &mut W,
        Cx,
    ) -> Unit = SavitzkyGolayPlan::filter_into::<f64>;
    let _: fn(
        &WhittakerHendersonPlan<f64>,
        &[f64],
        &mut [f64],
        &mut WhittakerWorkspace<f64>,
        Cx,
    ) -> Result<SmoothingDiagnostics<f64>, SciOxideError> = WhittakerHendersonPlan::smooth_into;
    let _: fn(
        &LocalFilterPlan<f64>,
        &[f64],
        &mut [f64],
        &mut [usize],
        &mut W,
        Cx,
    ) -> Result<usize, SciOxideError> = LocalFilterPlan::filter_into;
    let _: fn(
        &[f64],
        PeakOptions<f64>,
        &mut [Peak<f64>],
        &mut PeakWorkspace<f64>,
        Cx,
    ) -> Result<usize, SciOxideError> = find_peaks_into::<f64>;
    let _: fn(
        &RidgePlan<f64>,
        &[f64],
        &mut [RidgePeak<f64>],
        &mut RidgeWorkspace<f64>,
        Cx,
    ) -> Result<usize, SciOxideError> = RidgePlan::find_peaks_into;
    // Plan-less kernels publish atomically without caller staging.
    let _: fn(&[f64], DetrendKind, &[usize], &mut [f64], Cx) -> Unit = detrend_into::<f64>;
    let _: fn(&[f64], &[f64], &mut [VectorStrength<f64>], Cx) -> Unit = vector_strength_into::<f64>;
    let _: fn(&[f64], &[usize], f64, &mut [Peak<f64>], Cx) -> Unit = peak_properties_into::<f64>;
    let _: fn(&StftPlan<f64>, &[f64], &mut [C], &mut W, Cx) -> Unit = StftPlan::analyze_into;
    let _: fn(&StftPlan<f64>, &[C], &mut [f64], &mut W, Cx) -> Unit = StftPlan::synthesize_into;
    let _: fn(&SpectralEstimatePlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        SpectralEstimatePlan::power_into;
    let _: fn(&LombScargle<'static, f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        LombScargle::power_into;
    let _: fn(&WaveletPlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        WaveletPlan::analyze_into;
    let _: fn(&WaveletPlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        WaveletPlan::synthesize_into;
    let _: fn(&AnalyticSignalNdPlan<f64>, V, NdViewMut<'static, 'static, C>, &mut W, Cx) -> Unit =
        AnalyticSignalNdPlan::analytic_into;
    let _: fn(
        &MinimumPhasePlan<f64>,
        &[f64],
        &mut [f64],
        &mut W,
        Cx,
    ) -> Result<MinimumPhaseDiagnostics<f64>, SciOxideError> = MinimumPhasePlan::reconstruct_into;
    let _: fn(
        &TimeResponsePlan<f64>,
        &[f64],
        &[f64],
        &mut [f64],
        &mut [f64],
        &mut TimeResponseWorkspace<f64>,
        Cx,
    ) -> Unit = TimeResponsePlan::simulate_into;
};

// Designs and conversions that iterate have an owned form, which runs in a
// fresh root scope, and a `_scoped` form, which takes the solve scope last. The
// three designs with a termination status also have a `_with_options` form
// that reports it. Fixed-cost designs take no context.
const _: fn() = || {
    use scioxide_core::{Complex, ExecutionContext, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Band = IirBand<f64>;
    type Family = IirFamily<f64>;
    type Bands<'a> = &'a [FirBand<f64>];
    type Iir = Result<IirDesign<f64>, SciOxideError>;
    type Zpk = Result<PoleZeroGain<f64>, SciOxideError>;
    type Order = Result<IirOrderReport<f64>, SciOxideError>;
    type Sos = Result<SecondOrderSections<f64>, SciOxideError>;
    type Fir = Result<FirDesign<f64>, SciOxideError>;
    type Placement = Result<PolePlacementReport<f64>, SciOxideError>;
    let _: fn(usize, Family, Band, IirDomain) -> Iir = design_iir::<f64>;
    let _: fn(usize, Family, Band, IirDomain, Cx) -> Iir = design_iir_scoped::<f64>;
    let _: fn(usize, Family) -> Zpk = iir_prototype::<f64>;
    let _: fn(usize, Family, Cx) -> Zpk = iir_prototype_scoped::<f64>;
    let _: fn(IirOrderFamily, Band, Band, f64, f64, IirDomain) -> Order = select_iir_order::<f64>;
    let _: fn(IirOrderFamily, Band, Band, f64, f64, IirDomain, IirOrderOptions) -> Order =
        select_iir_order_with_options::<f64>;
    let _: fn(IirOrderFamily, Band, Band, f64, f64, IirDomain, IirOrderOptions, Cx) -> Order =
        select_iir_order_scoped::<f64>;
    let _: fn(usize, f64, f64) -> Sos = design_real_gammatone::<f64>;
    let _: fn(usize, f64, f64, Cx) -> Sos = design_real_gammatone_scoped::<f64>;
    let _: fn(usize, FirSymmetry, Bands<'static>) -> Fir = design_fir_remez::<f64>;
    let _: fn(usize, FirSymmetry, Bands<'static>, RemezOptions<f64>) -> Fir =
        design_fir_remez_with_options::<f64>;
    let _: fn(usize, FirSymmetry, Bands<'static>, RemezOptions<f64>, Cx) -> Fir =
        design_fir_remez_scoped::<f64>;
    let _: fn(usize, FirSymmetry, Bands<'static>) -> Fir = design_fir_least_squares::<f64>;
    let _: fn(&TransferFunction<f64>) -> Zpk = TransferFunction::to_zpk;
    let _: fn(&TransferFunction<f64>, Cx) -> Zpk = TransferFunction::to_zpk_scoped;
    let _: fn(&SecondOrderSections<f64>) -> Zpk = SecondOrderSections::to_zpk;
    let _: fn(&SecondOrderSections<f64>, Cx) -> Zpk = SecondOrderSections::to_zpk_scoped;
    let _: fn(&StateSpace<f64>, usize, usize) -> Zpk = StateSpace::to_zpk;
    let _: fn(&StateSpace<f64>, usize, usize, Cx) -> Zpk = StateSpace::to_zpk_scoped;
    let _: fn(&StateSpace<f64>, &[Complex<f64>]) -> Placement = StateSpace::place_poles;
    let _: fn(&StateSpace<f64>, &[Complex<f64>], PolePlacementOptions<f64>) -> Placement =
        StateSpace::place_poles_with_options;
    let _: fn(&StateSpace<f64>, &[Complex<f64>], PolePlacementOptions<f64>, Cx) -> Placement =
        StateSpace::place_poles_scoped;
};

// One-shot forms of resampling, peaks, detrending, smoothing, local filters,
// phase locking and test signals take plain data and return owned results;
// the peak workspace and the ridge and local-filter plans add owned forms.
const _: fn() = || {
    use scioxide_core::{Complex, SciOxideError};
    type C = Complex<f64>;
    type Values<T> = Result<Vec<T>, SciOxideError>;
    type Peaks = Values<Peak<f64>>;
    let _: fn(&[f64], usize, usize) -> Values<f64> = resample::<f64>;
    let _: fn(&[C], usize, usize) -> Values<C> = resample::<C>;
    let _: fn(&[f64], usize) -> Values<f64> = resample_periodic::<f64>;
    let _: fn(&[C], usize) -> Values<C> = resample_periodic::<C>;
    let _: fn(usize) -> Result<PeakWorkspace<f64>, SciOxideError> = PeakWorkspace::workspace;
    let _: fn(&[f64]) -> Peaks = find_peaks::<f64>;
    let _: fn(&[f64], PeakOptions<f64>) -> Peaks = find_peaks_with_options::<f64>;
    let _: fn(&[f64], &[usize], f64) -> Peaks = peak_properties::<f64>;
    let _: fn(&RidgePlan<f64>, &[f64]) -> Values<RidgePeak<f64>> = RidgePlan::find_peaks;
    let _: fn(&[f64], DetrendKind) -> Values<f64> = detrend::<f64>;
    let _: fn(&[C], DetrendKind) -> Values<C> = detrend::<C>;
    let _: fn(&[f64], usize, usize) -> Values<f64> = savitzky_golay::<f64>;
    let _: fn(&[C], usize, usize) -> Values<C> = savitzky_golay::<C>;
    let _: fn(&[f64], usize) -> Values<f64> = median_filter::<f64>;
    let _: fn(&LocalFilterPlan<f64>, &[f64]) -> Values<f64> = LocalFilterPlan::filter;
    let _: fn(&[f64], f64) -> Result<VectorStrength<f64>, SciOxideError> = vector_strength::<f64>;
    let _: fn(Waveform<'static, f64>, &[f64]) -> Values<f64> = waveform::<f64>;
};

// The remaining execution forms: filter axis variants, streams, spectral
// statistics, spline, wavelet and response families, and plan-less generators.
// Together with the blocks above and below, every public `_into` and
// `process_into` is pinned.
const _: fn() = || {
    use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Unit = Result<(), SciOxideError>;
    type Progress = Result<StreamProgress, SciOxideError>;
    type V = NdView<'static, 'static, f64>;
    type VMut = NdViewMut<'static, 'static, f64>;
    type C = Complex<f64>;
    type W = SignalWorkspace<f64>;
    let _: fn(
        &mut PolyphaseStream<'static, f64>,
        &[f64],
        StreamChunk,
        &mut [f64],
        &mut W,
        Cx,
    ) -> Progress = PolyphaseStream::process_into;
    let _: fn(&PolyphasePlan<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        PolyphasePlan::resample_axis_into;
    let _: fn(&IirFilter<f64>, V, usize, VMut, &mut IirState<f64>, &mut W, Cx) -> Unit =
        IirFilter::filter_axis_into::<f64>;
    let _: fn(
        &IirFilter<f64>,
        V,
        usize,
        ForwardBackwardOptions<'static, f64>,
        VMut,
        &mut W,
        Cx,
    ) -> Unit = IirFilter::forward_backward_axis_into::<f64>;
    let _: fn(&SavitzkyGolayPlan<f64>, V, usize, SavitzkyGolayBoundary, VMut, &mut W, Cx) -> Unit =
        SavitzkyGolayPlan::filter_axis_into::<f64>;
    let _: fn(&SymmetricFilter<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        SymmetricFilter::filter_into::<f64>;
    let _: fn(&SymmetricFilter<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        SymmetricFilter::filter_axis_into::<f64>;
    let _: fn(&SplineFilter<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        SplineFilter::coefficients_into::<f64>;
    let _: fn(&SplineFilter<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        SplineFilter::reconstruct_into::<f64>;
    let _: fn(&SplineFilter<f64>, &[f64], &[f64], f64, f64, &mut [f64], &mut W, Cx) -> Unit =
        SplineFilter::evaluate_into::<f64>;
    let _: fn(&SplineFilter<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        SplineFilter::coefficients_axis_into::<f64>;
    let _: fn(&SplineFilter<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        SplineFilter::reconstruct_axis_into::<f64>;
    let _: fn(
        &WhittakerHendersonPlan<f64>,
        V,
        usize,
        VMut,
        &mut WhittakerWorkspace<f64>,
        Cx,
    ) -> Unit = WhittakerHendersonPlan::smooth_axis_into;
    let _: fn(
        &mut LocalFilterStream<'static, f64>,
        &[f64],
        StreamChunk,
        &mut [f64],
        &mut [usize],
        &mut W,
        Cx,
    ) -> Progress = LocalFilterStream::process_into;
    let _: fn(&StftPlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit = StftPlan::spectrogram_into;
    let _: fn(&StftPlan<f64>, &[f64], &[f64], &mut [C], &mut W, Cx) -> Unit =
        StftPlan::cross_spectrogram_into;
    let _: fn(&StftPlan<f64>, &[f64], DetrendKind, &mut [C], &mut W, Cx) -> Unit =
        StftPlan::analyze_detrended_into;
    let _: fn(
        &mut StftAnalysisStream<'static, f64>,
        &[f64],
        StreamChunk,
        &mut [C],
        &mut W,
        Cx,
    ) -> Progress = StftAnalysisStream::process_into;
    let _: fn(
        &mut StftSynthesisStream<'static, f64>,
        &[C],
        StreamChunk,
        &mut [f64],
        &mut W,
        Cx,
    ) -> Progress = StftSynthesisStream::process_into;
    let _: fn(&SpectralEstimatePlan<f64>, &[f64], &[f64], &mut [C], &mut W, Cx) -> Unit =
        SpectralEstimatePlan::csd_into;
    let _: fn(&SpectralEstimatePlan<f64>, &[f64], &[f64], &mut [f64], &mut W, Cx) -> Unit =
        SpectralEstimatePlan::coherence_into;
    let _: fn(&Wavelet2dPlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        Wavelet2dPlan::analyze_into;
    let _: fn(&Wavelet2dPlan<f64>, &[f64], &mut [f64], &mut W, Cx) -> Unit =
        Wavelet2dPlan::synthesize_into;
    let _: fn(&AnalyticSignalNdPlan<f64>, V, VMut, &mut W, Cx) -> Unit =
        AnalyticSignalNdPlan::envelope_into;
    let _: fn(&RieszPlan<f64>, V, VMut, &mut W, Cx) -> Unit = RieszPlan::riesz_into::<f64>;
    let _: fn(&RieszPlan<f64>, V, VMut, &mut W, Cx) -> Unit = RieszPlan::monogenic_into;
    let _: fn(
        &FrequencyResponsePlan<f64>,
        &[f64],
        &mut [C],
        &mut FrequencyResponseWorkspace<f64>,
        Cx,
    ) -> Unit = FrequencyResponsePlan::frequency_response_into;
    let _: fn(
        &TimeResponsePlan<f64>,
        &mut [f64],
        &mut [f64],
        &mut TimeResponseWorkspace<f64>,
        Cx,
    ) -> Unit = TimeResponsePlan::impulse_response_into;
    let _: fn(
        &TimeResponsePlan<f64>,
        &mut [f64],
        &mut [f64],
        &mut TimeResponseWorkspace<f64>,
        Cx,
    ) -> Unit = TimeResponsePlan::step_response_into;
    let _: fn(Waveform<'static, f64>, &[f64], &mut [f64], Cx) -> Unit = waveform_into::<f64>;
    let _: fn(Waveform<'static, f64>, f64, f64, usize, &mut [f64], Cx) -> Unit =
        waveform_grid_into::<f64>;
    let _: fn(usize, usize, &mut [f64], Cx) -> Unit = unit_impulse_into::<f64>;
    let _: fn(&[C], &mut [f64], Cx) -> Unit = complex_envelope_into::<f64>;
};

// Options hold tuning and have a `Default`; the hop, the window length and the
// polynomial degree are positional, and plans report the resolved FFT length.
const _: fn() = || {
    use scioxide_core::{ExecutionContext, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Stft = Result<StftPlan<f64>, SciOxideError>;
    type Spectral = Result<SpectralEstimatePlan<f64>, SciOxideError>;
    type SavGol = Result<SavitzkyGolayPlan<f64>, SciOxideError>;
    let _: fn(&[f64], usize, StftOptions<f64>) -> Stft = StftPlan::new;
    let _: fn(&[f64], usize, StftOptions<f64>, Cx) -> Stft = StftPlan::build;
    let _: fn(&[f64], usize, StftOptions<f64>) -> Stft = StftPlan::from_dual;
    let _: fn(&[f64], &[f64], usize, StftOptions<f64>) -> Stft = StftPlan::from_windows;
    let _: fn(&StftPlan<f64>) -> usize = StftPlan::hop;
    let _: fn(&StftPlan<f64>) -> usize = StftPlan::fft_len;
    let _: fn() -> StftOptions<f64> = StftOptions::new;
    let _: fn(StftOptions<f64>, Option<usize>) -> StftOptions<f64> = StftOptions::with_fft_len;
    let _: fn(StftOptions<f64>) -> Option<usize> = StftOptions::fft_len;
    let _: fn(&[f64], SpectralEstimateOptions<f64>) -> Spectral = SpectralEstimatePlan::new;
    let _: fn(&[f64], SpectralEstimateOptions<f64>, Cx) -> Spectral = SpectralEstimatePlan::build;
    let _: fn(&SpectralEstimatePlan<f64>) -> usize = SpectralEstimatePlan::fft_len;
    let _: fn() -> SpectralEstimateOptions<f64> = SpectralEstimateOptions::new;
    let _: fn(SpectralEstimateOptions<f64>, Option<usize>) -> SpectralEstimateOptions<f64> =
        SpectralEstimateOptions::with_fft_len;
    let _: fn(SpectralEstimateOptions<f64>, f64) -> SpectralEstimateOptions<f64> =
        SpectralEstimateOptions::with_sample_rate;
    let _: fn(SpectralEstimateOptions<f64>) -> Option<usize> = SpectralEstimateOptions::fft_len;
    let _: fn(usize, usize, SavitzkyGolayOptions<f64>) -> SavGol = SavitzkyGolayPlan::new;
    let _: fn(usize, usize, SavitzkyGolayOptions<f64>, Cx) -> SavGol = SavitzkyGolayPlan::build;
    let _: fn(&SavitzkyGolayPlan<f64>) -> usize = SavitzkyGolayPlan::window_length;
    let _: fn(&SavitzkyGolayPlan<f64>) -> usize = SavitzkyGolayPlan::polynomial_degree;
    const _: SavitzkyGolayOptions<f64> = SavitzkyGolayOptions::new();
};

// Axis forms of the plans and method values: `verb_axis_into(input, axis,
// output, ws, cx)`, with the shapes, lane counts and requirements computed per
// call from the views.
const _: fn() = || {
    use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Unit = Result<(), SciOxideError>;
    type V = NdView<'static, 'static, f64>;
    type VMut = NdViewMut<'static, 'static, f64>;
    type C = Complex<f64>;
    type CV = NdView<'static, 'static, C>;
    type CVMut = NdViewMut<'static, 'static, C>;
    type W = SignalWorkspace<f64>;
    type CW = SignalWorkspace<C>;
    let _: fn(&SpectralEstimatePlan<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        SpectralEstimatePlan::power_axis_into;
    let _: fn(&SpectralEstimatePlan<f64>, V, V, usize, CVMut, &mut W, Cx) -> Unit =
        SpectralEstimatePlan::csd_axis_into;
    let _: fn(&SpectralEstimatePlan<f64>, V, V, usize, VMut, &mut W, Cx) -> Unit =
        SpectralEstimatePlan::coherence_axis_into;
    // Power and coherence of complex samples are real.
    let _: fn(&SpectralEstimatePlan<C>, CV, usize, VMut, &mut CW, Cx) -> Unit =
        SpectralEstimatePlan::power_axis_into;
    let _: fn(&SpectralEstimatePlan<C>, CV, CV, usize, VMut, &mut CW, Cx) -> Unit =
        SpectralEstimatePlan::coherence_axis_into;
    let _: fn(&StftPlan<f64>, V, usize, CVMut, &mut W, Cx) -> Unit = StftPlan::analyze_axis_into;
    let _: fn(&StftPlan<f64>, CV, usize, VMut, &mut W, Cx) -> Unit = StftPlan::synthesize_axis_into;
    let _: fn(&StftPlan<C>, CV, usize, CVMut, &mut CW, Cx) -> Unit = StftPlan::synthesize_axis_into;
    let _: fn(&WaveletPlan<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        WaveletPlan::analyze_axis_into;
    let _: fn(&WaveletPlan<f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        WaveletPlan::synthesize_axis_into;
    let _: fn(&LombScargle<'static, f64>, V, usize, VMut, &mut W, Cx) -> Unit =
        LombScargle::power_axis_into;
};

// One-shot forms take the problem and nothing else: a design converts to
// sections, the representations filter without a prepared value, and the
// spectral estimates and minimum-phase conversion take plain data. Configured
// owned forms take their options last.
const _: fn() = || {
    use scioxide_core::{Complex, SciOxideError};
    type C = Complex<f64>;
    type Samples = Result<Vec<f64>, SciOxideError>;
    type ComplexSamples = Result<Vec<C>, SciOxideError>;
    type Sos = Result<SecondOrderSections<f64>, SciOxideError>;
    type MinimumPhase = Result<(Vec<f64>, MinimumPhaseDiagnostics<f64>), SciOxideError>;
    let _: fn(&IirDesign<f64>) -> Sos = IirDesign::to_sos;
    let _: fn(IirOrderReport<f64>) -> IirFamily<f64> = IirOrderReport::family;
    let _: fn(&IirFilter<f64>, &[f64], ForwardBackwardOptions<'static, f64>) -> Samples =
        IirFilter::forward_backward_with_options::<f64>;
    let _: fn(&SecondOrderSections<f64>, &[f64]) -> Samples = SecondOrderSections::filter::<f64>;
    let _: fn(&SecondOrderSections<f64>, &[C]) -> ComplexSamples = SecondOrderSections::filter::<C>;
    let _: fn(&SecondOrderSections<f64>, &[f64]) -> Samples =
        SecondOrderSections::forward_backward::<f64>;
    let _: fn(&SecondOrderSections<f64>, &[C]) -> ComplexSamples =
        SecondOrderSections::forward_backward::<C>;
    let _: fn(&TransferFunction<f64>, &[f64]) -> Samples = TransferFunction::filter::<f64>;
    let _: fn(&TransferFunction<f64>, &[f64]) -> Samples =
        TransferFunction::forward_backward::<f64>;
    let _: fn(&[f64], f64, usize) -> Samples = welch::<f64>;
    let _: fn(&[C], f64, usize) -> Samples = welch::<C>;
    let _: fn(&[f64], &[f64], &[f64]) -> Samples = lomb_scargle::<f64>;
    let _: fn(&StftPlan<f64>, &[f64]) -> Samples = StftPlan::spectrogram;
    let _: fn(&[f64]) -> Samples = minimum_phase::<f64>;
    let _: fn(&[C]) -> ComplexSamples = minimum_phase::<C>;
    let _: fn(&[f64], MinimumPhaseOptions<f64>) -> MinimumPhase = minimum_phase_with_options::<f64>;
    let _: fn(&FirDesign<f64>) -> Result<PolyphasePlan<f64>, SciOxideError> =
        FirDesign::to_polyphase::<f64>;
};
