//! Allocation-free lifecycle, executor parity and transactional admission tests.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;

use scioxide_core::{
    Capacity, Complex, ExecutionContext, NdView, NdViewMut, SciOxideErrorKind, StridedLayout,
};
use scioxide_signal::*;

fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

#[test]
fn invalid_sizes_and_stream_requirements() {
    let p = PolyphasePlan::<f64>::fir(&[1.0, 2.0]).unwrap();
    assert_eq!(
        error_kind(p.output_len(usize::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        error_kind(PolyphasePlan::<f64>::rational_default(
            usize::MAX,
            1,
            ResampleAlignment::Causal
        )),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        error_kind(OverlapPlan::new(
            &[1.0, 2.0],
            usize::MAX,
            SignalExecutionOptions::new()
        )),
        SciOxideErrorKind::SizeOverflow
    );
    let stream = p.stream().unwrap();
    assert_eq!(
        error_kind(stream.workspace_requirements(usize::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
    let requirements = stream.workspace_requirements(8).unwrap();
    // Requirements bound every later call of at most eight samples, in any state.
    let mut workspace = SignalWorkspace::<f64>::new();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    assert_eq!(
        workspace
            .try_reserve(&requirements.with_max_bytes(8))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    assert_eq!(
        workspace
            .check(&requirements.with_max_bytes(8))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut stream = stream;
    let mut out = [0.0; 16];
    for chunk in [&[1.0; 8][..], &[2.0; 3], &[], &[3.0; 8]] {
        stream
            .process_into(
                chunk,
                StreamChunk::Partial,
                &mut out,
                &mut workspace,
                native(),
            )
            .unwrap();
    }
    stream
        .process_into(
            &[4.0; 8],
            StreamChunk::Final,
            &mut out,
            &mut workspace,
            native(),
        )
        .unwrap();
}
#[test]
fn final_chunks_flush_close_and_reset_reopens() {
    let p = PolyphasePlan::fir(&[1.0_f64, -1.0]).unwrap();
    let mut stream = p.stream().unwrap();
    let mut work = stream.workspace(4).unwrap();
    let mut out = [99.0; 4];
    let partial = stream
        .process_into(
            &[1.0, 2.0],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            native(),
        )
        .unwrap();
    assert_eq!((partial.consumed(), partial.produced()), (2, 2));
    assert_eq!(&out[..2], &[1.0, 1.0]);
    // The final chunk flushes the FIR tail and closes the stream.
    let last = stream
        .process_into(&[3.0], StreamChunk::Final, &mut out, &mut work, native())
        .unwrap();
    assert_eq!(
        (last.produced(), last.first_output_index()),
        (2, partial.produced())
    );
    assert_eq!(&out[..2], &[1.0, -3.0]);
    let closed = out;
    assert_eq!(
        error_kind(stream.process_into(
            &[1.0],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            native()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(out, closed);
    stream.reset();
    assert_eq!(stream.output_len(1, StreamChunk::Partial).unwrap(), 1);
    // A final empty chunk on a fresh stream produces nothing.
    let overlap = OverlapPlan::new(&[1.0_f64, 1.0], 4, SignalExecutionOptions::new()).unwrap();
    let mut s = overlap.stream().unwrap();
    assert_eq!(s.output_len(0, StreamChunk::Final).unwrap(), 0);
    let mut w = s.workspace(0).unwrap();
    assert_eq!(
        s.process_into(&[], StreamChunk::Final, &mut [], &mut w, native())
            .unwrap()
            .produced(),
        0
    );
    assert_eq!(
        error_kind(s.output_len(0, StreamChunk::Final)),
        SciOxideErrorKind::InvalidInput
    );
}
#[test]
fn cancellation_and_snapshots_preserve_state() {
    use scioxide_core::CancellationToken;
    let p = PolyphasePlan::fir(&[1.0_f64, -1.0]).unwrap();
    let mut stream = p.stream().unwrap();
    let mut out = [99.0; 4];
    let mut work = stream.workspace(2).unwrap();
    stream
        .process_into(
            &[1.0, 2.0],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            native(),
        )
        .unwrap();
    let before = out;
    let snapshot = stream.clone();
    let token = CancellationToken::new();
    token.cancel();
    assert_eq!(
        error_kind(stream.process_into(
            &[3.0, 4.0],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(out, before);
    stream
        .process_into(
            &[3.0, 4.0],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            native(),
        )
        .unwrap();
    assert_eq!(&out[..2], &[1.0, 1.0]);
    let advanced = out;
    stream.clone_from(&snapshot);
    stream
        .process_into(
            &[3.0, 4.0],
            StreamChunk::Partial,
            &mut out,
            &mut work,
            native(),
        )
        .unwrap();
    assert_eq!(out, advanced);
}
#[test]
fn wide_dynamic_range_direct_and_polyphase_preserve_small_products() {
    let p = PolyphasePlan::fir(&[f64::MAX, 1.0]).unwrap();
    let x = [f64::MIN_POSITIVE, 0.0];
    let out = p.resample(&x).unwrap();
    assert!(out[0] > 1.0);
    assert_eq!(out[1], f64::MIN_POSITIVE);
    let tiny = f64::from_bits(1);
    let p = PolyphasePlan::fir(&[1.0]).unwrap();
    assert_eq!(p.resample(&[f64::MAX, tiny]).unwrap(), [f64::MAX, tiny]);
}
#[test]
fn fft_normalization_accepts_components_below_rounding_envelope() {
    let taps = [1.0_f64];
    let input = [f64::MAX, f64::from_bits(1)];
    let p = ConvolutionNdPlan::new(
        &[2],
        NdView::new(&taps, StridedLayout::new(&[1], &[1]).unwrap()).unwrap(),
        &[0],
        ConvolutionOptions::new().with_method(ConvolutionMethod::Fft),
    )
    .unwrap();
    let mut out = [99.0; 2];
    let mut work = p.workspace(ConvolutionKernel::Prepared).unwrap();
    p.convolve_into(
        NdView::new(&input, StridedLayout::new(&[2], &[1]).unwrap()).unwrap(),
        NdViewMut::new(&mut out, StridedLayout::new(&[2], &[1]).unwrap()).unwrap(),
        &mut work,
        native(),
    )
    .unwrap();
    assert_eq!(out, [f64::MAX, 0.0]);
    let p = FourierResamplePlan::<f64>::new(&[2], 0, 3, SignalExecutionOptions::new()).unwrap();
    let mut out = [99.0; 3];
    let mut work = p.workspace(()).unwrap();
    p.resample_into(
        NdView::new(&input, StridedLayout::new(&[2], &[1]).unwrap()).unwrap(),
        NdViewMut::new(&mut out, StridedLayout::new(&[3], &[1]).unwrap()).unwrap(),
        &mut work,
        native(),
    )
    .unwrap();
    // Periodic interpolation of [MAX, 0] at phases 0, 2π/3, 4π/3.
    for (value, expected) in out.iter().zip([1.0, 0.25, 0.25]) {
        assert!((value / f64::MAX - expected).abs() < 32.0 * f64::EPSILON);
    }
    let mut short = SignalWorkspace::new();
    assert_eq!(
        p.resample_into(
            NdView::new(&input, StridedLayout::new(&[2], &[1]).unwrap()).unwrap(),
            NdViewMut::new(&mut out, StridedLayout::new(&[3], &[1]).unwrap()).unwrap(),
            &mut short,
            native(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}
#[test]
fn plans_and_streams_are_shareable_values() {
    fn value<T: Clone + core::fmt::Debug + Send + Sync>() {}
    value::<ConvolutionNdPlan<Complex<f64>>>();
    value::<SeparableFir2dPlan<f64>>();
    value::<OverlapPlan<f64>>();
    value::<OverlapStream<'static, f64>>();
    value::<PolyphasePlan<f32>>();
    value::<PolyphaseStream<'static, f32>>();
    value::<FourierResamplePlan<f64>>();
    value::<IirFilter<f64>>();
    value::<SavitzkyGolayPlan<f64>>();
    value::<SymmetricFilter<f64>>();
    value::<SplineFilter<f64>>();
    value::<WhittakerHendersonPlan<f64>>();
    value::<LocalFilterPlan<f64>>();
    value::<LocalFilterStream<'static, f64>>();
    value::<RidgePlan<f64>>();
    value::<SignalWorkspace<f64>>();
    value::<WhittakerWorkspace<f64>>();
    value::<PeakWorkspace<f64>>();
    value::<RidgeWorkspace<f64>>();
}
