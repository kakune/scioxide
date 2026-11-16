//! Every shared lane adapter retains the exact one-dimensional arithmetic order.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::{error, error_kind};

use scioxide_core::{
    Argument, Complex, ExecutionContext, InvalidInputReason, NdView, NdViewMut, NumericScalar,
    SciOxideError, SciOxideErrorKind, SciOxideErrorView, StridedLayout, argument,
};
use scioxide_signal::*;

/// Bit patterns of a real or complex sample, for exact comparisons.
trait Bits: Copy {
    fn bits(self) -> [u64; 2];
}
impl Bits for f64 {
    fn bits(self) -> [u64; 2] {
        [self.to_bits(), 0]
    }
}
impl Bits for Complex<f64> {
    fn bits(self) -> [u64; 2] {
        [self.re.to_bits(), self.im.to_bits()]
    }
}
#[track_caller]
fn assert_bits<U: Bits>(actual: &[U], expected: &[U]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.bits(), e.bits(), "element {i}");
    }
}
/// Lanes of `data` along `axis` of `layout`, in row-major order of the other
/// axes.
fn lanes<U: Copy>(data: &[U], layout: StridedLayout<'_>, axis: usize) -> Vec<Vec<U>> {
    let lanes = layout.axis(axis).unwrap();
    (0..lanes.len())
        .map(|batch| {
            let lane = lanes.lane(batch).unwrap();
            lane.offsets().map(|i| data[i]).collect()
        })
        .collect()
}
/// Argument and reason of an invalid-input error; `None` for any other outcome.
fn reason(result: Result<(), SciOxideError>) -> Option<(Argument, InvalidInputReason)> {
    match result.unwrap_err().view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    }
}
fn check_lanes(
    input: &NdView<'_, '_, f64>,
    output: &NdView<'_, '_, f64>,
    axis: usize,
    one: impl Fn(&[f64]) -> Vec<f64>,
) {
    let lanes = input.layout().axis(axis).unwrap();
    let targets = output.layout().axis(axis).unwrap();
    for batch in 0..lanes.len() {
        let x: Vec<_> = lanes
            .lane(batch)
            .unwrap()
            .offsets()
            .map(|i| input.storage()[i])
            .collect();
        let expected = one(&x);
        for (i, value) in targets.lane(batch).unwrap().offsets().zip(expected) {
            assert_eq!(
                output.storage()[i].to_bits(),
                value.to_bits(),
                "axis {axis} batch {batch}"
            );
        }
    }
}
#[test]
fn filters_match_lanes_on_all_axes_and_broadcast_input() {
    let shape = [8, 8, 8];
    let input_strides = [128, 16, 2];
    let out_strides = [2, 128, 16];
    let output_layout = StridedLayout::new(&shape, &out_strides).unwrap();
    let options = SignalExecutionOptions::new();
    let symmetric = SymmetricFilter::first_order(0.2_f64, 0.7, options).unwrap();
    let spline = SplineFilter::new(SplineDegree::Cubic, options).unwrap();
    let whittaker = WhittakerHendersonPlan::new(
        8,
        2,
        0.3,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let iir = IirFilter::from_transfer(
        &TransferFunction::new(IirDomain::Digital, &[0.5_f64], &[1., -0.5]).unwrap(),
        options,
    )
    .unwrap();
    let wavelet = WaveletPlan::new(
        WaveletFamily::Haar,
        8,
        1,
        WaveletOptions::default().with_boundary(SignalBoundary::Periodic),
    )
    .unwrap();
    let savgol = SavitzkyGolayPlan::new(3, 1, SavitzkyGolayOptions::new()).unwrap();
    let fb = ForwardBackwardOptions::new()
        .with_padding(IirPadding::Odd(2))
        .with_forward(IirInitialization::SteadyState)
        .with_backward(IirInitialization::SteadyState);
    let sg_boundary = SavitzkyGolayBoundary::Extend(SignalBoundary::Reflect);
    for broadcast in [false, true] {
        let strides = if broadcast { [0, 16, 2] } else { input_strides };
        let input_layout = StridedLayout::new_broadcast(&shape, &strides).unwrap();
        let mut data = vec![f64::NAN; input_layout.required_span()];
        for offset in input_layout.offsets() {
            data[offset] = (offset as f64 * 0.13).sin() + 0.2;
        }
        for axis in 0..3 {
            for family in 0..9 {
                let mut output = vec![43.; output_layout.required_span()];
                let input = NdView::new(&data, input_layout).unwrap();
                let target = NdViewMut::new(&mut output, output_layout).unwrap();
                let seq = ExecutionContext::sequential_native();
                let layout = SignalLayout::Axis {
                    shape: &shape,
                    axis,
                };
                let mut scratch = SignalWorkspace::new();
                for requirements in [
                    symmetric.workspace_requirements(layout),
                    spline.workspace_requirements(SplineRequest::Coefficients(layout)),
                    spline.workspace_requirements(SplineRequest::Reconstruct(layout)),
                    savgol.workspace_requirements(layout),
                ] {
                    scratch.try_reserve(&requirements.unwrap()).unwrap();
                }
                match family {
                    0 => symmetric
                        .filter_axis_into(input, axis, target, &mut scratch, seq)
                        .unwrap(),
                    1 => spline
                        .coefficients_axis_into(input, axis, target, &mut scratch, seq)
                        .unwrap(),
                    2 => spline
                        .reconstruct_axis_into(input, axis, target, &mut scratch, seq)
                        .unwrap(),
                    3 => whittaker
                        .smooth_axis_into(
                            input,
                            axis,
                            target,
                            &mut whittaker.workspace(64).unwrap(),
                            seq,
                        )
                        .unwrap(),
                    4 => iir
                        .filter_axis_into(
                            input,
                            axis,
                            target,
                            &mut IirState::new(&iir, 64).unwrap(),
                            &mut iir.workspace(IirRequest::Filter(layout)).unwrap(),
                            seq,
                        )
                        .unwrap(),
                    5 => iir
                        .forward_backward_axis_into(
                            input,
                            axis,
                            fb,
                            target,
                            &mut iir
                                .workspace(IirRequest::ForwardBackward(layout, fb.padding()))
                                .unwrap(),
                            seq,
                        )
                        .unwrap(),
                    6 => wavelet
                        .analyze_axis_into(
                            input,
                            axis,
                            target,
                            &mut wavelet.workspace(64).unwrap(),
                            seq,
                        )
                        .unwrap(),
                    7 => wavelet
                        .synthesize_axis_into(
                            input,
                            axis,
                            target,
                            &mut wavelet.workspace(64).unwrap(),
                            seq,
                        )
                        .unwrap(),
                    _ => savgol
                        .filter_axis_into(input, axis, sg_boundary, target, &mut scratch, seq)
                        .unwrap(),
                }
                check_lanes(
                    &NdView::new(&data, input_layout).unwrap(),
                    &NdView::new(&output, output_layout).unwrap(),
                    axis,
                    |x| match family {
                        0 => symmetric.filter(x).unwrap(),
                        1 => spline.coefficients(x).unwrap(),
                        2 => spline.reconstruct(x).unwrap(),
                        3 => whittaker.smooth(x).unwrap().0,
                        4 => iir.filter(x).unwrap(),
                        5 => {
                            let mut result = vec![0.; 8];
                            let mut w = iir
                                .workspace(IirRequest::ForwardBackward(
                                    SignalLayout::Contiguous(8),
                                    fb.padding(),
                                ))
                                .unwrap();
                            iir.forward_backward_into(x, fb, &mut result, &mut w, seq)
                                .unwrap();
                            result
                        }
                        6 => wavelet.analyze(x).unwrap(),
                        7 => wavelet.synthesize(x).unwrap(),
                        _ => savgol.filter(x, sg_boundary).unwrap(),
                    },
                );
                assert!(
                    output
                        .iter()
                        .enumerate()
                        .all(|(i, &v)| i % 2 == 0 || v == 43.)
                );
            }
        }
    }
}
#[test]
fn spectra_and_lomb_large_batches_are_exact_and_execute() {
    let n = 64;
    let batches = 4096;
    let shape = [batches, n];
    let strides = [n, 1];
    let lane: Vec<_> = (0..n).map(|i| (i as f64 * 0.37).sin() + 0.25).collect();
    let input: Vec<_> = lane.iter().copied().cycle().take(n * batches).collect();
    let p = SpectralEstimatePlan::discrete(n).unwrap();
    let expected = p.power(&lane).unwrap();
    let mut out = vec![0.; n * batches];
    let mut work = p
        .workspace(SignalLayout::Axis {
            shape: &shape,
            axis: 1,
        })
        .unwrap();
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    p.power_axis_into(
        NdView::new(&input, layout).unwrap(),
        1,
        NdViewMut::new(&mut out, layout).unwrap(),
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    for row in out.chunks(n) {
        assert_bits(row, &expected);
    }
    let times: Vec<_> = (0..n).map(|i| i as f64 * 0.1).collect();
    let frequencies = [0.2, 0.7, 1.4];
    let p = LombScargle::new(&times, &frequencies, None, Default::default()).unwrap();
    let expected = p.power(&lane).unwrap();
    let mut out = vec![0.; batches * 3];
    let mut work = p.workspace(batches).unwrap();
    p.power_axis_into(
        NdView::new(&input, layout).unwrap(),
        1,
        NdViewMut::new(
            &mut out,
            StridedLayout::new(&[batches, 3], &[3, 1]).unwrap(),
        )
        .unwrap(),
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    for row in out.chunks(3) {
        assert_bits(row, &expected);
    }
}

fn column_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 2;
    shape
        .iter()
        .map(|&n| {
            let result = stride;
            stride *= n;
            result
        })
        .collect()
}
fn row_strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    let mut stride = 1;
    for (&n, v) in shape.iter().zip(result.iter_mut()).rev() {
        *v = stride;
        stride *= n;
    }
    result
}
/// The STFT, spectral and wavelet axis forms equal their record forms lane by
/// lane, bit for bit, on every axis of a strided array.
fn transforms_match_each_lane<T: NumericScalar<Real = f64> + Bits>() {
    let shape = [8, 8, 8];
    let input_layout = StridedLayout::new(&shape, &[128, 16, 2]).unwrap();
    let value = |i: usize, a: f64, b: f64| {
        T::from_components((i as f64 * a).sin() + 0.3, (i as f64 * b).cos())
    };
    let mut x = vec![T::from_components(f64::NAN, f64::NAN); input_layout.required_span()];
    let mut y = x.clone();
    for i in input_layout.offsets() {
        x[i] = value(i, 0.13, 0.29);
        y[i] = value(i, 0.41, 0.17);
    }
    let window = [0.2, 0.5, 1., 0.7];
    let stft = StftPlan::new(
        &window.map(T::from_real),
        2,
        StftOptions::new().with_fft_len(Some(5)),
    )
    .unwrap();
    let options = SpectralEstimateOptions::new()
        .with_fft_len(Some(5))
        .with_sample_rate(2.)
        .with_detrend(None)
        .with_overlap(2);
    let spectral = SpectralEstimatePlan::<T>::new(&window, options).unwrap();
    let wavelet = WaveletPlan::<T>::new(
        WaveletFamily::Daubechies(4),
        8,
        1,
        WaveletOptions::default().with_boundary(SignalBoundary::Reflect),
    )
    .unwrap();
    let cx = ExecutionContext::sequential_native();
    let fill = T::from_components(43., 19.);
    for axis in 0..3 {
        let xs = lanes(&x, input_layout, axis);
        let ys = lanes(&y, input_layout, axis);
        let (a, b) = (
            NdView::new(&x, input_layout).unwrap(),
            NdView::new(&y, input_layout).unwrap(),
        );
        // STFT: the sample axis becomes a frame axis and a bin axis.
        let mut grid_shape = shape.to_vec();
        grid_shape[axis] = stft.frames(8).unwrap().count();
        grid_shape.insert(axis + 1, stft.bins());
        let grid_strides = column_strides(&grid_shape);
        let grid_layout = StridedLayout::new(&grid_shape, &grid_strides).unwrap();
        let mut grid = vec![Complex::new(43., 19.); grid_layout.required_span()];
        let mut ws = stft
            .workspace(StftRequest::Axis {
                shape: &shape,
                axis,
            })
            .unwrap();
        let target = NdViewMut::new(&mut grid, grid_layout).unwrap();
        stft.analyze_axis_into(a, axis, target, &mut ws, cx)
            .unwrap();
        let logical: Vec<_> = NdView::new(&grid, grid_layout)
            .unwrap()
            .iter()
            .copied()
            .collect();
        let mut flattened = shape;
        flattened[axis] = stft.output_len(8).unwrap();
        let flat_strides = row_strides(&flattened);
        let flat_layout = StridedLayout::new(&flattened, &flat_strides).unwrap();
        for (lane, x) in lanes(&logical, flat_layout, axis).iter().zip(&xs) {
            assert_bits(lane, &stft.analyze(x).unwrap());
        }
        let sample_strides = column_strides(&shape);
        let sample_layout = StridedLayout::new(&shape, &sample_strides).unwrap();
        let mut restored = vec![fill; sample_layout.required_span()];
        let coefficients = NdView::new(&grid, grid_layout).unwrap();
        let target = NdViewMut::new(&mut restored, sample_layout).unwrap();
        stft.synthesize_axis_into(coefficients, axis, target, &mut ws, cx)
            .unwrap();
        for (lane, x) in lanes(&restored, sample_layout, axis).iter().zip(&xs) {
            let record = stft.synthesize(&stft.analyze(x).unwrap(), 8).unwrap();
            assert_bits(lane, &record);
        }
        // Spectral estimates: the sample axis becomes the frequency bins.
        let mut estimate_shape = shape;
        estimate_shape[axis] = spectral.bins();
        let estimate_strides = column_strides(&estimate_shape);
        let estimate_layout = StridedLayout::new(&estimate_shape, &estimate_strides).unwrap();
        let span = estimate_layout.required_span();
        let (mut power, mut cross, mut coherence) = (
            vec![43.; span],
            vec![Complex::new(43., 19.); span],
            vec![43.; span],
        );
        let mut ws = spectral
            .workspace(SignalLayout::Axis {
                shape: &shape,
                axis,
            })
            .unwrap();
        let target = NdViewMut::new(&mut power, estimate_layout).unwrap();
        spectral
            .power_axis_into(a, axis, target, &mut ws, cx)
            .unwrap();
        let target = NdViewMut::new(&mut cross, estimate_layout).unwrap();
        spectral
            .csd_axis_into(a, b, axis, target, &mut ws, cx)
            .unwrap();
        let target = NdViewMut::new(&mut coherence, estimate_layout).unwrap();
        spectral
            .coherence_axis_into(a, b, axis, target, &mut ws, cx)
            .unwrap();
        let power = lanes(&power, estimate_layout, axis);
        let cross = lanes(&cross, estimate_layout, axis);
        let coherence = lanes(&coherence, estimate_layout, axis);
        for (i, (x, y)) in xs.iter().zip(&ys).enumerate() {
            assert_bits(&power[i], &spectral.power(x).unwrap());
            assert_bits(&cross[i], &spectral.csd(x, y).unwrap());
            assert_bits(&coherence[i], &spectral.coherence(x, y).unwrap());
        }
        // Wavelets: the sample axis becomes the packed coefficients.
        let mut packed_shape = shape;
        packed_shape[axis] = wavelet.coefficient_len();
        let packed_strides = column_strides(&packed_shape);
        let packed_layout = StridedLayout::new(&packed_shape, &packed_strides).unwrap();
        let mut packed = vec![fill; packed_layout.required_span()];
        let mut ws = wavelet.workspace(64).unwrap();
        let target = NdViewMut::new(&mut packed, packed_layout).unwrap();
        wavelet
            .analyze_axis_into(a, axis, target, &mut ws, cx)
            .unwrap();
        for (lane, x) in lanes(&packed, packed_layout, axis).iter().zip(&xs) {
            assert_bits(lane, &wavelet.analyze(x).unwrap());
        }
        let mut restored = vec![fill; sample_layout.required_span()];
        let coefficients = NdView::new(&packed, packed_layout).unwrap();
        let target = NdViewMut::new(&mut restored, sample_layout).unwrap();
        wavelet
            .synthesize_axis_into(coefficients, axis, target, &mut ws, cx)
            .unwrap();
        let packed_lanes = lanes(&packed, packed_layout, axis);
        for (lane, coefficients) in lanes(&restored, sample_layout, axis)
            .iter()
            .zip(&packed_lanes)
        {
            assert_bits(lane, &wavelet.synthesize(coefficients).unwrap());
        }
        // The odd offsets between the strided output elements are never written.
        let untouched = |data: &[T]| {
            data.iter()
                .skip(1)
                .step_by(2)
                .all(|v| v.bits() == fill.bits())
        };
        assert!(untouched(&packed) && untouched(&restored));
    }
}
#[test]
fn transforms_match_each_real_lane() {
    transforms_match_each_lane::<f64>();
}
#[test]
fn transforms_match_each_complex_lane() {
    transforms_match_each_lane::<Complex<f64>>();
}
#[test]
fn empty_batches_accept_supported_layouts_with_reused_storage() {
    let plan = SpectralEstimatePlan::<f64>::discrete(8).unwrap();
    for (shape, axis) in [([0, 8, 4], 1), ([8, 0, 4], 0), ([8, 4, 0], 0)] {
        let strides = [0, 0, 0];
        let mut output_shape = shape;
        output_shape[axis] = plan.bins();
        let mut work = plan
            .workspace(SignalLayout::Axis {
                shape: &shape,
                axis,
            })
            .unwrap();
        plan.power_axis_into(
            NdView::new(&[], StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
            axis,
            NdViewMut::new(
                &mut [],
                StridedLayout::new(&output_shape, &strides).unwrap(),
            )
            .unwrap(),
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    }
}

#[test]
fn all_filter_families_treat_empty_batches_as_no_ops() {
    let shape = [0, 8, 4];
    let strides = [0, 0, 0];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let options = SignalExecutionOptions::new();
    let symmetric = SymmetricFilter::first_order(0.2_f64, 0.7, options).unwrap();
    let spline = SplineFilter::new(SplineDegree::Cubic, options).unwrap();
    let whittaker = WhittakerHendersonPlan::new(
        8,
        2,
        0.3,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let mut wh_work = WhittakerWorkspace::new();
    let iir = IirFilter::from_transfer(
        &TransferFunction::new(IirDomain::Digital, &[0.5_f64], &[1., -0.5]).unwrap(),
        options,
    )
    .unwrap();
    let mut state = IirState::new(&iir, 0).unwrap();
    let mut iir_work = SignalWorkspace::new();
    let mut scratch = SignalWorkspace::new();
    let wavelet = WaveletPlan::<f64>::new(
        WaveletFamily::Haar,
        8,
        1,
        WaveletOptions::default().with_boundary(SignalBoundary::Periodic),
    )
    .unwrap();
    let savgol = SavitzkyGolayPlan::new(3, 1, SavitzkyGolayOptions::new()).unwrap();
    let polyphase = PolyphasePlan::fir(&[0.25_f64, 0.5, 0.25]).unwrap();
    let mut poly_work = polyphase
        .workspace(SignalLayout::Axis {
            shape: &shape,
            axis: 1,
        })
        .unwrap();
    let fb = ForwardBackwardOptions::new()
        .with_padding(IirPadding::Odd(2))
        .with_forward(IirInitialization::SteadyState)
        .with_backward(IirInitialization::SteadyState);
    let mut output = [43.];
    let seq = ExecutionContext::sequential_native();
    (|| -> Result<(), scioxide_core::SciOxideError> {
        symmetric.filter_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, layout)?,
            &mut scratch,
            seq,
        )?;
        spline.coefficients_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, layout)?,
            &mut scratch,
            seq,
        )?;
        spline.reconstruct_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, layout)?,
            &mut scratch,
            seq,
        )?;
        whittaker.smooth_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, layout)?,
            &mut wh_work,
            seq,
        )?;
        iir.filter_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, layout)?,
            &mut state,
            &mut iir_work,
            seq,
        )?;
        iir.forward_backward_axis_into(
            NdView::new(&[], layout)?,
            1,
            fb,
            NdViewMut::new(&mut output, layout)?,
            &mut iir_work,
            seq,
        )?;
        wavelet.analyze_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, layout)?,
            &mut SignalWorkspace::new(),
            seq,
        )?;
        savgol.filter_axis_into(
            NdView::new(&[], layout)?,
            1,
            SavitzkyGolayBoundary::Extend(SignalBoundary::Reflect),
            NdViewMut::new(&mut output, layout)?,
            &mut scratch,
            seq,
        )?;
        polyphase.resample_axis_into(
            NdView::new(&[], layout)?,
            1,
            NdViewMut::new(&mut output, StridedLayout::new(&[0, 10, 4], &strides)?)?,
            &mut poly_work,
            seq,
        )?;
        Ok(())
    })()
    .unwrap();
    assert_eq!(output, [43.]);
    let empty_axis = StridedLayout::new(&[2, 0], &[0, 0]).unwrap();
    assert_eq!(
        error_kind(iir.forward_backward_axis_into(
            NdView::new(&[], empty_axis).unwrap(),
            1,
            fb,
            NdViewMut::new(&mut output, empty_axis).unwrap(),
            &mut iir_work,
            seq
        )),
        SciOxideErrorKind::InvalidInput
    );
}
#[test]
fn axis_forms_reject_shapes_axes_and_workspaces_without_publishing() {
    let cx = ExecutionContext::sequential_native;
    let mismatch = |result: Result<(), SciOxideError>| {
        assert_eq!(error_kind(result), SciOxideErrorKind::DimensionMismatch);
    };
    let out_of_range = Some((argument!("axis"), InvalidInputReason::OutOfRange));
    // Three records of eight samples along axis 1.
    let shape = [3, 8];
    let x: Vec<f64> = (0..24).map(|i| (i as f64 * 0.37).sin() + 0.1).collect();
    let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
    let input = NdView::new(&x, layout).unwrap();
    let short_y = NdView::new(&x, StridedLayout::new(&[3, 7], &[8, 1]).unwrap()).unwrap();

    let options = SpectralEstimateOptions::new().with_overlap(2);
    let spectral = SpectralEstimatePlan::<f64>::new(&[1.; 4], options).unwrap();
    let bins = spectral.bins();
    let (good, bad, strides) = ([3, bins], [3, bins + 1], [bins + 1, 1]);
    let (good, bad) = (
        StridedLayout::new(&good, &strides).unwrap(),
        StridedLayout::new(&bad, &strides).unwrap(),
    );
    let axis_layout = SignalLayout::Axis {
        shape: &shape,
        axis: 1,
    };
    let mut ws = spectral.workspace(axis_layout).unwrap();
    let mut small = spectral.workspace(SignalLayout::Contiguous(8)).unwrap();
    let mut power = vec![7.; 3 * (bins + 1)];
    let mut cross = vec![Complex::new(7., 7.); 3 * (bins + 1)];
    mismatch(spectral.power_axis_into(
        input,
        1,
        NdViewMut::new(&mut power, bad).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(spectral.power_axis_into(
        input,
        1,
        NdViewMut::new(&mut power, good).unwrap(),
        &mut small,
        cx(),
    ));
    mismatch(spectral.csd_axis_into(
        input,
        short_y,
        1,
        NdViewMut::new(&mut cross, good).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(spectral.csd_axis_into(
        input,
        input,
        1,
        NdViewMut::new(&mut cross, bad).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(spectral.coherence_axis_into(
        input,
        input,
        1,
        NdViewMut::new(&mut power, good).unwrap(),
        &mut small,
        cx(),
    ));
    assert_eq!(
        reason(spectral.power_axis_into(
            input,
            2,
            NdViewMut::new(&mut power, good).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    assert_eq!(
        reason(spectral.csd_axis_into(
            input,
            input,
            2,
            NdViewMut::new(&mut cross, good).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    assert_eq!(
        error(spectral.workspace_requirements(SignalLayout::Axis {
            shape: &shape,
            axis: 2
        }))
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    // Lanes shorter than one segment are reported under the record argument.
    let transposed = NdView::new(&x, StridedLayout::new(&[8, 3], &[3, 1]).unwrap()).unwrap();
    let short = Some((argument!("input"), InvalidInputReason::Inconsistent));
    assert_eq!(
        reason(spectral.power_axis_into(
            transposed,
            1,
            NdViewMut::new(&mut power, good).unwrap(),
            &mut ws,
            cx()
        )),
        short
    );
    let short = Some((argument!("x"), InvalidInputReason::Inconsistent));
    assert_eq!(
        reason(spectral.csd_axis_into(
            transposed,
            transposed,
            1,
            NdViewMut::new(&mut cross, good).unwrap(),
            &mut ws,
            cx()
        )),
        short
    );
    assert_eq!(
        reason(spectral.coherence_axis_into(
            transposed,
            transposed,
            1,
            NdViewMut::new(&mut power, good).unwrap(),
            &mut ws,
            cx()
        )),
        short
    );
    let short = Some((argument!("layout"), InvalidInputReason::Inconsistent));
    let requirements = |layout| -> Result<(), SciOxideError> {
        spectral.workspace_requirements(layout).map(|_| ())
    };
    assert_eq!(
        reason(requirements(SignalLayout::Axis {
            shape: &[8, 3],
            axis: 1
        })),
        short
    );
    assert_eq!(reason(requirements(SignalLayout::Contiguous(3))), short);
    // Coherence needs mean averaging.
    let median =
        SpectralEstimatePlan::<f64>::new(&[1.; 4], options.with_average(SpectralAverage::Median))
            .unwrap();
    let mut median_ws = median.workspace(axis_layout).unwrap();
    assert_eq!(
        reason(median.coherence_axis_into(
            input,
            input,
            1,
            NdViewMut::new(&mut power, good).unwrap(),
            &mut median_ws,
            cx()
        )),
        Some((
            argument!("self"),
            InvalidInputReason::UnsupportedConfiguration
        ))
    );
    assert!(power.iter().all(|&v| v == 7.));
    assert!(cross.iter().all(|&v| v == Complex::new(7., 7.)));

    // STFT: the coefficient view is checked extent by extent.
    let stft = StftPlan::new(&[1.0_f64; 4], 2, StftOptions::new()).unwrap();
    let (frames, bins) = (stft.frames(8).unwrap().count(), stft.bins());
    let mut grid = vec![Complex::new(7., 7.); 3 * (frames + 1) * (bins + 1)];
    let mut samples = vec![7.; 24];
    let (grid_shape, grid_strides) = ([3, frames, bins], [frames * bins, bins, 1]);
    let grid_layout = StridedLayout::new(&grid_shape, &grid_strides).unwrap();
    let request = StftRequest::Axis {
        shape: &shape,
        axis: 1,
    };
    let mut ws = stft.workspace(request).unwrap();
    let mut small = stft.workspace(StftRequest::Analysis(8)).unwrap();
    for wrong in [
        [3, frames + 1, bins],
        [3, frames, bins + 1],
        [2, frames, bins],
    ] {
        let wrong_strides = row_strides(&wrong);
        let wrong_layout = StridedLayout::new(&wrong, &wrong_strides).unwrap();
        assert_eq!(
            error_kind(stft.analyze_axis_into(
                input,
                1,
                NdViewMut::new(&mut grid, wrong_layout).unwrap(),
                &mut ws,
                cx()
            )),
            SciOxideErrorKind::DimensionMismatch
        );
    }
    let (flat_shape, flat_strides) = ([3, frames * bins], [frames * bins, 1]);
    let flat = StridedLayout::new(&flat_shape, &flat_strides).unwrap();
    let error = error(stft.analyze_axis_into(
        input,
        1,
        NdViewMut::new(&mut grid, flat).unwrap(),
        &mut ws,
        cx(),
    ));
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { expected, .. } if expected[..] == grid_shape
    ));
    mismatch(stft.analyze_axis_into(
        input,
        1,
        NdViewMut::new(&mut grid, grid_layout).unwrap(),
        &mut small,
        cx(),
    ));
    assert_eq!(
        reason(stft.analyze_axis_into(
            input,
            2,
            NdViewMut::new(&mut grid, grid_layout).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    assert!(grid.iter().all(|&v| v == Complex::new(7., 7.)));
    let coefficients = NdView::new(&grid, grid_layout).unwrap();
    mismatch(
        stft.synthesize_axis_into(
            coefficients,
            1,
            NdViewMut::new(
                &mut samples[..18],
                StridedLayout::new(&[3, 6], &[6, 1]).unwrap(),
            )
            .unwrap(),
            &mut ws,
            cx(),
        ),
    );
    mismatch(stft.synthesize_axis_into(
        coefficients,
        1,
        NdViewMut::new(&mut samples, layout).unwrap(),
        &mut small,
        cx(),
    ));
    assert_eq!(
        reason(stft.synthesize_axis_into(
            coefficients,
            2,
            NdViewMut::new(&mut samples, layout).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    // Without NOLA the plan has no dual window and only analyzes.
    let analysis_only = StftPlan::new(&[1.0_f64, 0., 0., 0.], 2, StftOptions::new()).unwrap();
    assert!(analysis_only.dual_window().is_none());
    let mut analysis_ws = analysis_only.workspace(request).unwrap();
    assert_eq!(
        reason(analysis_only.synthesize_axis_into(
            coefficients,
            1,
            NdViewMut::new(&mut samples, layout).unwrap(),
            &mut analysis_ws,
            cx()
        )),
        Some((
            argument!("self"),
            InvalidInputReason::UnsupportedConfiguration
        ))
    );
    assert!(samples.iter().all(|&v| v == 7.));

    // Wavelets: lane counts come from the shape of the call.
    let wavelet =
        WaveletPlan::<f64>::new(WaveletFamily::Haar, 8, 2, WaveletOptions::default()).unwrap();
    let m = wavelet.coefficient_len();
    let mut packed = vec![7.; 3 * (m + 1)];
    let (packed_shape, wrong_shape, packed_strides) = ([3, m], [3, m + 1], [m + 1, 1]);
    let packed_layout = StridedLayout::new(&packed_shape, &packed_strides).unwrap();
    let wrong_layout = StridedLayout::new(&wrong_shape, &packed_strides).unwrap();
    let mut ws = wavelet.workspace(3).unwrap();
    let mut small = wavelet.workspace(2).unwrap();
    mismatch(wavelet.analyze_axis_into(
        short_y,
        1,
        NdViewMut::new(&mut packed, packed_layout).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(wavelet.analyze_axis_into(
        input,
        1,
        NdViewMut::new(&mut packed, wrong_layout).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(wavelet.analyze_axis_into(
        input,
        1,
        NdViewMut::new(&mut packed, packed_layout).unwrap(),
        &mut small,
        cx(),
    ));
    assert_eq!(
        reason(wavelet.analyze_axis_into(
            input,
            2,
            NdViewMut::new(&mut packed, packed_layout).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    assert!(packed.iter().all(|&v| v == 7.));
    let coefficients = NdView::new(&packed, packed_layout).unwrap();
    mismatch(wavelet.synthesize_axis_into(
        coefficients,
        1,
        NdViewMut::new(&mut samples, layout).unwrap(),
        &mut small,
        cx(),
    ));
    assert_eq!(
        reason(wavelet.synthesize_axis_into(
            coefficients,
            3,
            NdViewMut::new(&mut samples, layout).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    assert!(samples.iter().all(|&v| v == 7.));

    // Lomb–Scargle: the axis holds one value per sample time.
    let times: Vec<f64> = (0..8)
        .map(|i| i as f64 * 0.7 + 0.1 * (i % 3) as f64)
        .collect();
    let frequencies = [0.5, 1.5];
    let lomb = LombScargle::new(&times, &frequencies, None, Default::default()).unwrap();
    let mut periodograms = vec![7.; 9];
    let out = StridedLayout::new(&[3, 2], &[3, 1]).unwrap();
    let wrong = StridedLayout::new(&[3, 3], &[3, 1]).unwrap();
    let mut ws = lomb.workspace(3).unwrap();
    let mut small = lomb.workspace(2).unwrap();
    mismatch(lomb.power_axis_into(
        short_y,
        1,
        NdViewMut::new(&mut periodograms, out).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(lomb.power_axis_into(
        input,
        1,
        NdViewMut::new(&mut periodograms, wrong).unwrap(),
        &mut ws,
        cx(),
    ));
    mismatch(lomb.power_axis_into(
        input,
        1,
        NdViewMut::new(&mut periodograms, out).unwrap(),
        &mut small,
        cx(),
    ));
    assert_eq!(
        reason(lomb.power_axis_into(
            input,
            2,
            NdViewMut::new(&mut periodograms, out).unwrap(),
            &mut ws,
            cx()
        )),
        out_of_range
    );
    assert!(periodograms.iter().all(|&v| v == 7.));
}
