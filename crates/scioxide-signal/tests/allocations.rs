#![allow(clippy::unwrap_used)]
#[macro_use]
#[path = "support/views.rs"]
mod views;

use scioxide_core::{Complex, ExecutionContext};

use scioxide_signal::{
    ConvolutionExtent, ConvolutionKernel, ConvolutionMethod, ConvolutionNdPlan, ConvolutionOptions,
    SignalBoundary, SignalProduct, SpectralEstimatePlan,
};

fn signal(len: usize, phase: f64) -> Vec<f64> {
    (0..len).map(|i| (i as f64 * 0.013 + phase).sin()).collect()
}
#[test]
fn prepared_signal_operations_execute() {
    let len = 1024;
    let a = signal(len, 0.2);
    let b = signal(len, -0.4);
    for (extent, boundary) in [
        (ConvolutionExtent::Full, SignalBoundary::Zero),
        (ConvolutionExtent::Circular, SignalBoundary::Periodic),
    ] {
        for product in [SignalProduct::Convolution, SignalProduct::Correlation] {
            let plan = ConvolutionNdPlan::new(
                &[len],
                view!(&b),
                &[0],
                ConvolutionOptions::new()
                    .with_extent(extent)
                    .with_boundary(boundary)
                    .with_product(product)
                    .with_method(ConvolutionMethod::Fft),
            )
            .unwrap();
            let mut output = vec![0.; plan.output_shape()[0]];
            let mut workspace = plan.workspace(ConvolutionKernel::Supplied).unwrap();
            plan.convolve_into(
                view!(&a),
                view_mut!(&mut output),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
            plan.convolve_pair_into(
                view!(&a),
                view!(&b),
                view_mut!(&mut output),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        }
    }
    let plan = SpectralEstimatePlan::discrete(len).unwrap();
    let mut work = plan
        .workspace(scioxide_signal::SignalLayout::Contiguous(len))
        .unwrap();
    let mut power = vec![0.; len];
    let mut cross = vec![Complex::new(0., 0.); len];
    plan.power_into(
        &a,
        &mut power,
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    plan.csd_into(
        &a,
        &b,
        &mut cross,
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
}

#[test]
fn prepared_features_have_no_hidden_allocations() {
    use scioxide_signal::*;
    let input = [0_f64, 0., 1., 0., 0., 0., 0.];
    let mut output = [0.; 7];
    let plan = AnalyticSignalNdPlan::new(&[7], &[0], SignalExecutionOptions::new()).unwrap();
    let mut analytic_work = plan.workspace(()).unwrap();
    plan.envelope_into(
        view!(&input),
        view_mut!(&mut output),
        &mut analytic_work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let native = ExecutionContext::sequential_native();
    let mut peaks = [Peak::default(); 3];
    let mut peak_work = PeakWorkspace::new();
    peak_work
        .try_reserve(&PeakWorkspace::<f64>::workspace_requirements(input.len()))
        .unwrap();
    find_peaks_into(
        &input,
        PeakOptions::default(),
        &mut peaks,
        &mut peak_work,
        native,
    )
    .unwrap();
    let many = [0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0];
    for distance in [3, 0, 3] {
        find_peaks_into(
            &many,
            PeakOptions::default().with_distance(distance),
            &mut peaks,
            &mut peak_work,
            native,
        )
        .unwrap();
    }
    let mut properties = [Peak::default(); 1];
    peak_properties_into(&input, &[2], 0.5, &mut properties, native).unwrap();
    let filter = LocalFilterPlan::new(1, LocalFilter::Hampel { threshold: 3. }).unwrap();
    let mut indices = [0; 7];
    let mut real = filter.workspace(input.len()).unwrap();
    filter
        .filter_into(&input, &mut output, &mut indices, &mut real, native)
        .unwrap();
    let mut stream = filter.stream().unwrap();
    let mut stream_work = stream.workspace(input.len()).unwrap();
    stream
        .process_into(
            &input,
            StreamChunk::Final,
            &mut output,
            &mut indices,
            &mut stream_work,
            native,
        )
        .unwrap();
    detrend_into(&input, DetrendKind::Linear, &[], &mut output, native).unwrap();
    waveform_grid_into(
        Waveform::Sine {
            frequency: 1.,
            phase: 0.,
        },
        0.,
        0.1,
        0,
        &mut output,
        native,
    )
    .unwrap();
    let mut phase = [VectorStrength::default()];
    vector_strength_into(&input, &[1.], &mut phase, native).unwrap();
    let mut ridges = [RidgePeak::default(); 3];
    let ridge_plan = RidgePlan::new(
        &[1., 2.],
        RidgeOptions::default()
            .with_support_radius(3)
            .with_max_distance(1)
            .with_max_gap(0)
            .with_min_length(2)
            .with_min_strength(0.),
    )
    .unwrap();
    let mut ridge_work = ridge_plan.workspace(input.len()).unwrap();
    ridge_plan
        .find_peaks_into(&input, &mut ridges, &mut ridge_work, native)
        .unwrap();
}

#[test]
fn incremental_filters_and_range_fallbacks_execute() {
    use scioxide_signal::{LocalFilter, LocalFilterPlan, estimate_wiener_noise};
    for input in [
        [0., 2., 1., 3., 0., 4., -1., 2., 0.],
        [0., 1e-300, 2e-300, f64::MAX, -f64::MAX, 0., 1., 1., 1.],
    ] {
        for kind in [
            LocalFilter::Median,
            LocalFilter::Rank { rank: 1 },
            LocalFilter::Hampel { threshold: 3. },
            LocalFilter::Wiener {
                noise_variance: 0.1,
            },
        ] {
            let plan = LocalFilterPlan::new(2, kind).unwrap();
            let mut out = [0.; 9];
            let mut indices = [0; 9];
            let native = scioxide_core::ExecutionContext::sequential_native();
            let mut scratch = plan.workspace(input.len()).unwrap();
            plan.filter_into(&input, &mut out, &mut indices, &mut scratch, native)
                .unwrap();
            let mut stream = plan.stream().unwrap();
            let mut scratch = stream.workspace(input.len()).unwrap();
            let snapshot = stream.clone();
            (|| {
                stream.process_into(
                    &input,
                    scioxide_signal::StreamChunk::Final,
                    &mut out,
                    &mut indices,
                    &mut scratch,
                    native,
                )?;
                stream.clone_from(&snapshot);
                Ok::<(), scioxide_core::SciOxideError>(())
            })()
            .unwrap();
        }
        // Extended-range variances may exceed the scalar's range; allocation
        // freedom applies equally to an error returned at final publication.
        let _ = estimate_wiener_noise(&input, 2);
    }
}

/// Prepared paths beyond the family `*_resources` suites: 2-D and supplied-kernel
/// convolution, separable FIR, polyphase and Fourier resampling, Savitzky–Golay,
/// STFT analysis/synthesis/spectrogram/streams, spectral estimates, Lomb–Scargle,
/// wavelets, Riesz, splines, waveforms and envelopes.
#[test]
fn every_prepared_family_executes_without_allocation() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_signal::*;
    let cx = ExecutionContext::sequential_native();
    let shape = [12_usize, 10];
    let strides = [10_usize, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let x: Vec<f64> = (0..120).map(|i| (i as f64 * 0.3).sin()).collect();
    let y: Vec<f64> = (0..120).map(|i| (i as f64 * 0.17).cos()).collect();
    let xc: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, -v)).collect();
    let kernel_shape = [3_usize, 4];
    let kernel_strides = [4_usize, 1];
    let kernel_layout = StridedLayout::new(&kernel_shape, &kernel_strides).unwrap();
    let h: Vec<f64> = (0..12).map(|i| (i as f64 * 0.7).cos()).collect();
    for method in [ConvolutionMethod::Fft, ConvolutionMethod::Direct] {
        let options = ConvolutionOptions::new()
            .with_method(method)
            .with_boundary(SignalBoundary::Reflect)
            .with_extent(ConvolutionExtent::Same);
        let plan = ConvolutionNdPlan::new(
            &shape,
            NdView::new(&h, kernel_layout).unwrap(),
            &[0, 1],
            options,
        )
        .unwrap();
        let mut out = vec![0.0; 120];
        let mut ws = plan.workspace(ConvolutionKernel::Supplied).unwrap();
        plan.convolve_pair_into(
            NdView::new(&x, layout).unwrap(),
            NdView::new(&h, kernel_layout).unwrap(),
            NdViewMut::new(&mut out, layout).unwrap(),
            &mut ws,
            cx,
        )
        .unwrap();
        let separable = SeparableFir2dPlan::new(
            shape,
            &[0.25, 0.5, 0.25],
            &[1.0, -1.0, 0.5, 0.1],
            ConvolutionOptions::new().with_method(method),
        )
        .unwrap();
        let out_shape = separable.output_shape().to_vec();
        let out_strides = [out_shape[1], 1];
        let mut out = vec![0.0; out_shape[0] * out_shape[1]];
        let mut ws = separable.workspace(()).unwrap();
        separable
            .filter_into(
                NdView::new(&x, layout).unwrap(),
                NdViewMut::new(
                    &mut out,
                    StridedLayout::new(&out_shape, &out_strides).unwrap(),
                )
                .unwrap(),
                &mut ws,
                cx,
            )
            .unwrap();
    }
    let polyphase =
        PolyphasePlan::<f64>::rational_default(3, 2, ResampleAlignment::Centered).unwrap();
    let mut ws = polyphase.workspace(SignalLayout::Contiguous(120)).unwrap();
    let mut out = vec![0.0; polyphase.output_len(120).unwrap()];
    polyphase.resample_into(&x, &mut out, &mut ws, cx).unwrap();
    let rows = polyphase.output_len(12).unwrap();
    let (resampled_shape, resampled_strides) = ([rows, 10], [10, 1]);
    let mut ws = polyphase
        .workspace(SignalLayout::Axis {
            shape: &shape,
            axis: 0,
        })
        .unwrap();
    let mut out = vec![0.0; rows * 10];
    polyphase
        .resample_axis_into(
            NdView::new(&x, layout).unwrap(),
            0,
            NdViewMut::new(
                &mut out,
                StridedLayout::new(&resampled_shape, &resampled_strides).unwrap(),
            )
            .unwrap(),
            &mut ws,
            cx,
        )
        .unwrap();
    let fourier =
        FourierResamplePlan::<Complex<f64>>::new(&shape, 1, 16, SignalExecutionOptions::new())
            .unwrap();
    let mut ws = fourier.workspace(()).unwrap();
    let (fourier_shape, fourier_strides) = ([12, 16], [16, 1]);
    let mut out = vec![Complex::new(0.0, 0.0); 12 * 16];
    fourier
        .resample_into(
            NdView::new(&xc, layout).unwrap(),
            NdViewMut::new(
                &mut out,
                StridedLayout::new(&fourier_shape, &fourier_strides).unwrap(),
            )
            .unwrap(),
            &mut ws,
            cx,
        )
        .unwrap();
    let savgol = SavitzkyGolayPlan::new(7, 3, SavitzkyGolayOptions::new()).unwrap();
    let mut ws = savgol
        .workspace::<f64>(SignalLayout::Contiguous(120))
        .unwrap();
    let mut out = vec![0.0; 120];
    for boundary in [
        SavitzkyGolayBoundary::Polynomial,
        SavitzkyGolayBoundary::Extend(SignalBoundary::Reflect),
    ] {
        savgol
            .filter_into(&x, boundary, &mut out, &mut ws, cx)
            .unwrap();
    }
    let window: Vec<f64> = (0..16)
        .map(|i| 0.5 + (i as f64 * 0.3).sin().abs())
        .collect();
    for packing in [StftPacking::Full, StftPacking::OneSided] {
        let plan = StftPlan::<f64>::new(
            &window,
            4,
            StftOptions::new()
                .with_fft_len(Some(20))
                .with_packing(packing)
                .with_scaling(StftScaling::Psd),
        )
        .unwrap();
        let n = plan.output_len(100).unwrap();
        let mut coefficients = vec![Complex::new(0.0, 0.0); n];
        let mut ws = plan.workspace(StftRequest::Analysis(100)).unwrap();
        plan.analyze_into(&x[..100], &mut coefficients, &mut ws, cx)
            .unwrap();
        let mut ws = plan.workspace(StftRequest::Synthesis(100)).unwrap();
        let mut record = vec![0.0; 100];
        plan.synthesize_into(&coefficients, &mut record, &mut ws, cx)
            .unwrap();
        let mut ws = plan.workspace(StftRequest::Spectrogram(100)).unwrap();
        let mut power = vec![0.0; n];
        plan.spectrogram_into(&x[..100], &mut power, &mut ws, cx)
            .unwrap();
        let mut ws = plan.workspace(StftRequest::CrossSpectrogram(100)).unwrap();
        plan.cross_spectrogram_into(&x[..100], &y[..100], &mut coefficients, &mut ws, cx)
            .unwrap();
        let mut ws = plan.workspace(StftRequest::DetrendedAnalysis(100)).unwrap();
        plan.analyze_detrended_into(
            &x[..100],
            DetrendKind::Linear,
            &mut coefficients,
            &mut ws,
            cx,
        )
        .unwrap();
        let mut stream = plan.analysis_stream().unwrap();
        let mut ws = stream.workspace(30).unwrap();
        let mut frames = vec![Complex::new(0.0, 0.0); 1000];
        (|| {
            stream.process_into(&x[..30], StreamChunk::Partial, &mut frames, &mut ws, cx)?;
            stream.process_into(&x[30..60], StreamChunk::Final, &mut frames, &mut ws, cx)
        })()
        .unwrap();
        let full = plan.analyze(&x[..100]).unwrap();
        let mut synthesis = plan.synthesis_stream(100).unwrap();
        let mut ws = synthesis.workspace(full.len()).unwrap();
        synthesis
            .process_into(&full, StreamChunk::Final, &mut record, &mut ws, cx)
            .unwrap();
        // Every lane along axis 1 of the 12 × 10 array is one record.
        let (frames, bins) = (plan.frames(10).unwrap().count(), plan.bins());
        let grid_shape = [12, frames, bins];
        let grid_strides = [frames * bins, bins, 1];
        let grid_layout = StridedLayout::new(&grid_shape, &grid_strides).unwrap();
        let mut grid = vec![Complex::new(0.0, 0.0); 12 * frames * bins];
        let mut lanes = vec![0.0; 120];
        let mut ws = plan
            .workspace(StftRequest::Axis {
                shape: &shape,
                axis: 1,
            })
            .unwrap();
        (|| -> Result<(), scioxide_core::SciOxideError> {
            let output = NdViewMut::new(&mut grid, grid_layout)?;
            plan.analyze_axis_into(NdView::new(&x, layout)?, 1, output, &mut ws, cx)?;
            let output = NdViewMut::new(&mut lanes, layout)?;
            plan.synthesize_axis_into(NdView::new(&grid, grid_layout)?, 1, output, &mut ws, cx)
        })()
        .unwrap();
    }
    let estimate = SpectralEstimatePlan::<f64>::new(
        &window,
        SpectralEstimateOptions::new()
            .with_fft_len(Some(20))
            .with_sample_rate(1.0)
            .with_overlap(8)
            .with_detrend(Some(DetrendKind::Linear)),
    )
    .unwrap();
    let mut ws = estimate.workspace(SignalLayout::Contiguous(100)).unwrap();
    let mut coherence = vec![0.0; estimate.bins()];
    estimate
        .coherence_into(&x[..100], &y[..100], &mut coherence, &mut ws, cx)
        .unwrap();
    // Two records of 60 samples, one per row.
    let (rows, row_strides) = ([2_usize, 60], [60_usize, 1]);
    let rows_layout = StridedLayout::new(&rows, &row_strides).unwrap();
    let estimates = [2_usize, estimate.bins()];
    let estimate_strides = [estimate.bins(), 1];
    let estimate_layout = StridedLayout::new(&estimates, &estimate_strides).unwrap();
    let mut ws = estimate
        .workspace(SignalLayout::Axis {
            shape: &rows,
            axis: 1,
        })
        .unwrap();
    let mut power = vec![0.0; 2 * estimate.bins()];
    let mut cross = vec![Complex::new(0.0, 0.0); 2 * estimate.bins()];
    (|| -> Result<(), scioxide_core::SciOxideError> {
        let (a, b) = (NdView::new(&x, rows_layout)?, NdView::new(&y, rows_layout)?);
        let output = NdViewMut::new(&mut power, estimate_layout)?;
        estimate.power_axis_into(a, 1, output, &mut ws, cx)?;
        let output = NdViewMut::new(&mut cross, estimate_layout)?;
        estimate.csd_axis_into(a, b, 1, output, &mut ws, cx)?;
        let output = NdViewMut::new(&mut power, estimate_layout)?;
        estimate.coherence_axis_into(a, b, 1, output, &mut ws, cx)
    })()
    .unwrap();
    let times: Vec<f64> = (0..100)
        .map(|i| i as f64 + 0.3 * (i as f64).sin())
        .collect();
    let frequencies: Vec<f64> = (1..50).map(|k| k as f64 * 0.05).collect();
    let lomb = LombScargle::new(&times, &frequencies, None, LombScargleOptions::default()).unwrap();
    let mut ws = lomb.workspace(1).unwrap();
    let mut power = vec![0.0; frequencies.len()];
    lomb.power_into(&x[..100], &mut power, &mut ws, cx).unwrap();
    let lomb = LombScargle::new(&times[..10], &frequencies, None, Default::default()).unwrap();
    let mut ws = lomb.workspace(12).unwrap();
    let periodograms = [12_usize, frequencies.len()];
    let periodogram_strides = [frequencies.len(), 1];
    let periodogram_layout = StridedLayout::new(&periodograms, &periodogram_strides).unwrap();
    let mut power = vec![0.0; 12 * frequencies.len()];
    (|| {
        lomb.power_axis_into(
            NdView::new(&x, layout)?,
            1,
            NdViewMut::new(&mut power, periodogram_layout)?,
            &mut ws,
            cx,
        )
    })()
    .unwrap();
    let wavelet = WaveletPlan::<f64>::new(
        WaveletFamily::BiorthogonalSpline(309),
        100,
        2,
        WaveletOptions::default().with_boundary(SignalBoundary::Zero),
    )
    .unwrap();
    let mut ws = wavelet.workspace(1).unwrap();
    let mut coefficients = vec![0.0; wavelet.coefficient_len()];
    wavelet
        .analyze_into(&x[..100], &mut coefficients, &mut ws, cx)
        .unwrap();
    let mut record = vec![0.0; 100];
    wavelet
        .synthesize_into(&coefficients, &mut record, &mut ws, cx)
        .unwrap();
    let wavelet = WaveletPlan::<f64>::new(
        WaveletFamily::Daubechies(4),
        10,
        1,
        WaveletOptions::default().with_boundary(SignalBoundary::Reflect),
    )
    .unwrap();
    let m = wavelet.coefficient_len();
    let (packed_shape, packed_strides) = ([12_usize, m], [m, 1_usize]);
    let packed_layout = StridedLayout::new(&packed_shape, &packed_strides).unwrap();
    let mut packed = vec![0.0; 12 * m];
    let mut lanes = vec![0.0; 120];
    let mut ws = wavelet.workspace(12).unwrap();
    (|| -> Result<(), scioxide_core::SciOxideError> {
        let output = NdViewMut::new(&mut packed, packed_layout)?;
        wavelet.analyze_axis_into(NdView::new(&x, layout)?, 1, output, &mut ws, cx)?;
        let output = NdViewMut::new(&mut lanes, layout)?;
        wavelet.synthesize_axis_into(NdView::new(&packed, packed_layout)?, 1, output, &mut ws, cx)
    })()
    .unwrap();
    let square = [10_usize, 10];
    let square_strides = [10_usize, 1];
    let square_layout = StridedLayout::new(&square, &square_strides).unwrap();
    let (riesz_shape, riesz_strides) = ([2_usize, 10, 10], [100_usize, 10, 1]);
    let riesz = RieszPlan::<f64>::new(&square, &[0, 1], &[1.0, 1.0], SignalExecutionOptions::new())
        .unwrap();
    let mut ws = riesz.workspace(()).unwrap();
    let mut components = vec![0.0; 200];
    riesz
        .riesz_into(
            NdView::new(&x[..100], square_layout).unwrap(),
            NdViewMut::new(
                &mut components,
                StridedLayout::new(&riesz_shape, &riesz_strides).unwrap(),
            )
            .unwrap(),
            &mut ws,
            cx,
        )
        .unwrap();
    let spline =
        SplineFilter::<f64>::new(SplineDegree::Cubic, SignalExecutionOptions::new()).unwrap();
    let mut ws = spline
        .workspace(SplineRequest::Coefficients(SignalLayout::Contiguous(100)))
        .unwrap();
    let mut spline_coefficients = vec![0.0; 100];
    spline
        .coefficients_into(&x[..100], &mut spline_coefficients, &mut ws, cx)
        .unwrap();
    let mut out = vec![0.0; 100];
    waveform_into(
        Waveform::Sawtooth {
            frequency: 0.1,
            phase: 0.0,
            width: 0.25,
        },
        &times,
        &mut out,
        cx,
    )
    .unwrap();
    unit_impulse_into(7, 3, &mut out, cx).unwrap();
    complex_envelope_into(&xc[..100], &mut out, cx).unwrap();
}
