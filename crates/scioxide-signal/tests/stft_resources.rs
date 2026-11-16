//! Transactional dense channels, executor injection, workspace and allocation tests.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;

use scioxide_core::{
    CancellationToken, Capacity, Complex, ExecutionContext, ExecutionPlan, ExecutionPolicy,
    Executor, PartitionFn, SciOxideError, SciOxideErrorKind, SequentialExecutor,
};
use scioxide_signal::*;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

fn z() -> Complex<f64> {
    Complex::new(0., 0.)
}
fn cx() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
struct Counting(AtomicUsize);
impl Executor for Counting {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        SequentialExecutor.execute(context, plan, operation)
    }
}

#[test]
fn failure_preserves_output_and_stream_state() {
    let plan = StftPlan::new(&[1.; 4], 2, StftOptions::new().with_fft_len(Some(4))).unwrap();
    let mut stream = plan.analysis_stream().unwrap();
    let n = stream.output_len(7, StreamChunk::Partial).unwrap();
    let mut out = vec![Complex::new(9., 3.); n];
    let sentinel = out.clone();
    let requirements = stream.workspace_requirements(7).unwrap();
    let mut work = SignalWorkspace::new();
    work.try_reserve(&requirements).unwrap();
    let mut bad = [1.; 7];
    bad[4] = f64::NAN;
    assert_eq!(
        error_kind(stream.process_into(&bad, StreamChunk::Partial, &mut out, &mut work, cx())),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, sentinel);
    assert_eq!(stream.output_len(7, StreamChunk::Partial).unwrap(), n);
    let mut small = SignalWorkspace::new();
    small
        .try_reserve(
            &plan
                .workspace_requirements(StftRequest::Analysis(0))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(small.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    assert_eq!(
        stream
            .process_into(&[1.; 7], StreamChunk::Partial, &mut out, &mut small, cx())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, sentinel);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&cancel);
    assert_eq!(
        error_kind(stream.process_into(
            &[1.; 7],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            context
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, sentinel);
    assert_eq!(stream.output_len(7, StreamChunk::Partial).unwrap(), n);
    stream
        .process_into(&[1.; 7], StreamChunk::Partial, &mut out, &mut work, cx())
        .unwrap();
    let o = StftOptions::new()
        .with_fft_len(Some(4))
        .with_packing(StftPacking::OneSided);
    let plan = StftPlan::new(&[1.; 4], 2, o).unwrap();
    let mut coefficients = plan.analyze(&[1.; 7]).unwrap();
    coefficients[0].im = 2.;
    let mut output = [9.; 7];
    let mut work = plan.workspace(StftRequest::Synthesis(7)).unwrap();
    assert_eq!(
        error_kind(plan.synthesize_into(&coefficients, &mut output, &mut work, cx())),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(output, [9.; 7]);
    assert_eq!(
        error_kind(plan.output_len(usize::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        error_kind(StftPlan::new(
            &[1.; 4],
            0,
            StftOptions::new().with_fft_len(Some(4))
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(StftPlan::new(
            &[1.; 4],
            2,
            StftOptions::new().with_fft_len(Some(3))
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(StftPlan::new(&[Complex::new(1., 0.); 4], 2, o)),
        SciOxideErrorKind::InvalidInput
    );
}
#[test]
fn workspace_requirements_cover_each_request_and_limit_bytes() {
    let plan = StftPlan::new(&[1., 2., 1.], 1, StftOptions::new().with_fft_len(Some(4))).unwrap();
    let n = 9;
    let coefficients = plan.output_len(n).unwrap();
    let analysis = plan
        .workspace_requirements(StftRequest::Analysis(n))
        .unwrap();
    let cross = plan
        .workspace_requirements(StftRequest::CrossSpectrogram(n))
        .unwrap();
    let detrended = plan
        .workspace_requirements(StftRequest::DetrendedAnalysis(n))
        .unwrap();
    assert_eq!(
        plan.workspace_requirements(StftRequest::Spectrogram(n))
            .unwrap(),
        analysis
    );
    assert_eq!(cross.complex_len(), analysis.complex_len() + plan.bins());
    assert_eq!(
        (detrended.values_len(), detrended.complex_len()),
        (9, analysis.complex_len())
    );
    assert_eq!(
        error_kind(plan.workspace_requirements(StftRequest::Synthesis(usize::MAX))),
        SciOxideErrorKind::SizeOverflow
    );
    // An axis request stages the whole sample and coefficient arrays, one
    // lane of each, and serves both directions.
    let axis = plan
        .workspace_requirements(StftRequest::Axis {
            shape: &[3, n],
            axis: 1,
        })
        .unwrap();
    let synthesis = plan
        .workspace_requirements(StftRequest::Synthesis(n))
        .unwrap();
    assert_eq!(
        (axis.values_len(), axis.real_len()),
        (3 * n + n, synthesis.real_len())
    );
    assert_eq!(
        axis.complex_len(),
        3 * coefficients + coefficients + synthesis.complex_len()
    );
    assert_eq!(
        error_kind(plan.workspace_requirements(StftRequest::Axis {
            shape: &[3, n],
            axis: 2,
        })),
        SciOxideErrorKind::InvalidInput
    );
    // One workspace reserved for the largest request serves every smaller one.
    let mut ws = SignalWorkspace::new();
    for request in [
        StftRequest::Analysis(n),
        StftRequest::Synthesis(n),
        StftRequest::Spectrogram(n),
        StftRequest::CrossSpectrogram(n),
        StftRequest::DetrendedAnalysis(n),
    ] {
        ws.try_reserve(&plan.workspace_requirements(request).unwrap())
            .unwrap();
    }
    assert_eq!(
        ws.check(&cross.max(detrended)).unwrap(),
        Capacity::Sufficient
    );
    let x: Vec<_> = (0..n).map(|i| (i as f64 * 0.4).sin()).collect();
    let mut power = vec![0.; coefficients];
    plan.spectrogram_into(&x, &mut power, &mut ws, cx())
        .unwrap();
    let mut product = vec![z(); coefficients];
    plan.cross_spectrogram_into(&x, &x, &mut product, &mut ws, cx())
        .unwrap();
    for (p, c) in power.iter().zip(&product) {
        assert!((p - c.re).abs() < 1e-12 && c.im.abs() < 1e-12);
    }
    let limited = cross.with_max_bytes(cross.bytes::<f64>().unwrap() - 1);
    assert_eq!(
        ws.check(&limited).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        SignalWorkspace::<f64>::new()
            .try_reserve(&limited)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let before = product.clone();
    let mut small = plan.workspace(StftRequest::Analysis(n)).unwrap();
    assert_eq!(
        plan.cross_spectrogram_into(&x, &x, &mut product, &mut small, cx())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(product, before);
}

#[test]
fn injected_fft_executor_matches_sequential() {
    let executor = Counting(AtomicUsize::new(0));
    let context = ExecutionContext::new(&executor);
    let options = StftOptions::new()
        .with_fft_len(Some(131072))
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let window = vec![1.; 65536];
    let plan = StftPlan::build(&window, 32768, options, context).unwrap();
    let input = vec![1.; 3];
    let mut out = vec![z(); plan.output_len(input.len()).unwrap()];
    let mut work = plan.workspace(StftRequest::Analysis(input.len())).unwrap();
    plan.analyze_into(&input, &mut out, &mut work, context)
        .unwrap();
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    let reference = StftPlan::new(
        &window,
        32768,
        options.with_execution_policy(ExecutionPolicy::Sequential),
    )
    .unwrap()
    .analyze(&input)
    .unwrap();
    assert_eq!(out, reference);
}

#[test]
fn progress_counts_buffer_elements() {
    let plan = StftPlan::new(&[1.; 4], 2, StftOptions::new().with_fft_len(Some(4))).unwrap();
    let mut analysis = plan.analysis_stream().unwrap();
    let n = analysis.output_len(7, StreamChunk::Final).unwrap();
    let mut coefficients = vec![z(); n];
    let mut work = SignalWorkspace::new();
    work.try_reserve(&analysis.workspace_requirements(7).unwrap())
        .unwrap();
    let progress = analysis
        .process_into(
            &[1.; 7],
            StreamChunk::Final,
            &mut coefficients,
            &mut work,
            cx(),
        )
        .unwrap();
    assert_eq!(progress.consumed(), 7);
    assert_eq!(progress.produced(), n);
    assert_eq!(progress.first_output_index(), 0);
    let mut synthesis = plan.synthesis_stream(7).unwrap();
    let mut output = [0.; 7];
    work.try_reserve(&synthesis.workspace_requirements(n).unwrap())
        .unwrap();
    let progress = synthesis
        .process_into(
            &coefficients,
            StreamChunk::Final,
            &mut output,
            &mut work,
            cx(),
        )
        .unwrap();
    assert_eq!(progress.consumed(), n);
    assert_eq!(progress.produced(), 7);
    assert_eq!(progress.first_output_index(), 0);
}
#[test]
fn plans_and_streams_are_clone_debug_send_sync() {
    fn shared<T: Clone + core::fmt::Debug + Send + Sync>() {}
    shared::<StftPlan<f64>>();
    shared::<StftPlan<Complex<f32>>>();
    shared::<StftAnalysisStream<'static, f64>>();
    shared::<StftSynthesisStream<'static, Complex<f64>>>();
    shared::<SpectralEstimatePlan<f64>>();
    shared::<SpectralEstimatePlan<Complex<f64>>>();
    shared::<LombScargle<'static, f64>>();
    shared::<LombScargle<'static, f32>>();
    shared::<WaveletPlan<f64>>();
    shared::<WaveletPlan<Complex<f64>>>();
    shared::<Wavelet2dPlan<f32>>();
    shared::<WaveletFilters<f64>>();
    shared::<AnalyticSignalNdPlan<f64>>();
    shared::<RieszPlan<f64>>();
    fn copy<T: Copy>() {}
    copy::<LombScargle<'static, f64>>();
    copy::<StftOptions<f64>>();
    copy::<SpectralEstimateOptions<f64>>();
    copy::<LombScargleOptions<f64>>();
    copy::<WaveletOptions>();
}
