//! Selected-axis Fourier and trigonometric tensors, layouts and numerical limits.
#![expect(clippy::unwrap_used)]
#[path = "support/views.rs"]
mod views;
use views::view;
#[path = "support/numerics.rs"]
mod numerics;
use numerics::{assert_fft_close, scalar, wide};
#[path = "support/real_definitions.rs"]
mod real_definitions;
use real_definitions::{direct_dct, direct_dst};
#[path = "support/execution.rs"]
mod execution;

use scioxide_core::{Complex, NdView, NdViewMut, RealScalar, StridedLayout};
use scioxide_fft::*;

#[test]
fn every_trigonometric_poll_precedes_output_publication() {
    let shape = [4, 8];
    let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
    let input: Vec<f64> = (0..32).map(|i| f64::from(i) / 32.0).collect();
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let plan = RealTransformNdPlan::<f64>::new(
            RealTransformNdPlanSpec::new(
                &shape,
                &[0, 1],
                &[DctType::II.into(); 2],
                direction,
                Normalization::Orthonormal,
            )
            .unwrap(),
        )
        .unwrap();
        for in_place in [false, true] {
            let mut completed = false;
            for limit in 0..64 {
                let cancellation = execution::CancelAfter::new(limit);
                let context =
                    scioxide_core::ExecutionContext::sequential().with_cancellation(&cancellation);
                let mut output = if in_place {
                    input.clone()
                } else {
                    vec![123.0; 32]
                };
                let before = output.clone();
                let mut workspace = plan.workspace(()).unwrap();
                let target = NdViewMut::new(&mut output, layout).unwrap();
                let result = if in_place {
                    plan.execute_in_place(target, &mut workspace, context)
                } else {
                    plan.execute_into(
                        NdView::new(&input, layout).unwrap(),
                        target,
                        &mut workspace,
                        context,
                    )
                };
                if result.is_ok() {
                    completed = true;
                    assert_ne!(output, before);
                    break;
                }
                assert_eq!(
                    result.unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::ExecutionStopped
                );
                assert_eq!(output, before, "poll {limit}, in_place={in_place}");
            }
            assert!(completed);
        }
    }
}

fn layout(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    let mut result = vec![0; shape.len()];
    for (n, s) in shape.iter().zip(result.iter_mut()).rev() {
        *s = stride;
        stride *= n;
    }
    result
}
// Unnormalized N-D DFT oracle: 1-D defining sums applied axis by axis on a
// row-major layout; unselected axes are independent batches.
fn direct_nd(
    input: &[Complex<f64>],
    shape: &[usize],
    axes: &[usize],
    inverse: bool,
) -> Vec<Complex<f64>> {
    let strides = layout(shape);
    axes.iter().fold(input.to_vec(), |values, &axis| {
        let (n, stride) = (shape[axis], strides[axis]);
        let mut next = values.clone();
        for start in (0..values.len()).filter(|index| (index / stride).is_multiple_of(n)) {
            let lane: Vec<_> = (0..n).map(|j| values[start + j * stride]).collect();
            for (k, value) in numerics::direct_dft(&lane, inverse).into_iter().enumerate() {
                next[start + k * stride] = value;
            }
        }
        next
    })
}
fn fourier_cases<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    for (shape, axes) in [
        (vec![2, 3, 5], vec![2, 0]),
        (vec![2, 17], vec![1]),
        (vec![2, 23], vec![1, 0]),
        (vec![2, 3, 4], vec![0, 1, 2]),
        // Hundreds of strided short lanes, including a partial 256-lane block.
        (vec![2, 512], vec![0, 1]),
        (vec![2, 257], vec![1, 0]),
        (vec![1, 257], vec![0, 1]),
        (vec![1, 1], vec![]),
    ] {
        let strides = layout(&shape);
        let n = shape.iter().product();
        let source: Vec<_> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let input: Vec<_> = source
            .iter()
            .map(|z| Complex::new(scalar::<S>(z.re), scalar::<S>(z.im)))
            .collect();
        for norm in [
            Normalization::Forward,
            Normalization::Inverse,
            Normalization::Unscaled,
            Normalization::Orthonormal,
        ] {
            let factor = axes.iter().map(|&a| shape[a]).product::<usize>() as f64;
            for inverse in [false, true] {
                let scale = match (norm, inverse) {
                    (Normalization::Forward, false) | (Normalization::Inverse, true) => {
                        1.0 / factor
                    }
                    (Normalization::Orthonormal, _) => 1.0 / factor.sqrt(),
                    _ => 1.0,
                };
                let expected: Vec<_> = direct_nd(&source, &shape, &axes, inverse)
                    .into_iter()
                    .map(|z| z * scale)
                    .collect();
                let result = if inverse {
                    ifftn(&input, &shape, &axes, norm)
                } else {
                    fftn(&input, &shape, &axes, norm)
                }
                .unwrap();
                assert_fft_close(&result, &expected, n);
            }
        }
        let real: Vec<_> = source.iter().map(|z| scalar::<S>(z.re)).collect();
        if axes.is_empty() {
            continue;
        }
        let real_source: Vec<_> = real.iter().map(|&v| Complex::new(wide(v), 0.0)).collect();
        let direct = direct_nd(&real_source, &shape, &axes, false);
        for norm in [
            Normalization::Forward,
            Normalization::Inverse,
            Normalization::Unscaled,
            Normalization::Orthonormal,
        ] {
            let packed = rfftn(&real, &shape, &axes, norm).unwrap();
            let mut packed_shape = shape.clone();
            let axis = *axes.last().unwrap();
            packed_shape[axis] = shape[axis] / 2 + 1;
            let packed_strides = layout(&packed_shape);
            let factor = axes.iter().map(|&a| shape[a]).product::<usize>() as f64;
            let scale = match norm {
                Normalization::Forward => factor.recip(),
                Normalization::Orthonormal => factor.sqrt().recip(),
                _ => 1.0,
            };
            let expected: Vec<_> = (0..packed.bins().len())
                .map(|index| {
                    let full_index: usize = packed_shape
                        .iter()
                        .zip(&packed_strides)
                        .zip(&strides)
                        .map(|((&extent, &packed_stride), &stride)| {
                            index / packed_stride % extent * stride
                        })
                        .sum();
                    direct[full_index] * scale
                })
                .collect();
            assert_fft_close(packed.bins(), &expected, n);
            let recovered = irfftn(&packed, norm).unwrap();
            let factor = if norm == Normalization::Unscaled {
                axes.iter().map(|&a| shape[a]).product::<usize>() as f64
            } else {
                1.0
            };
            assert_fft_close(
                &recovered
                    .iter()
                    .map(|&v| Complex::new(v, S::ZERO))
                    .collect::<Vec<_>>(),
                &source
                    .iter()
                    .map(|z| Complex::new(z.re * factor, 0.0))
                    .collect::<Vec<_>>(),
                n,
            );
            let hermitian = ihfftn(&real, &shape, &axes, norm).unwrap();
            let recovered = hfftn(&hermitian, norm).unwrap();
            assert_fft_close(
                &recovered
                    .iter()
                    .map(|&v| Complex::new(v, S::ZERO))
                    .collect::<Vec<_>>(),
                &source
                    .iter()
                    .map(|z| Complex::new(z.re * factor, 0.0))
                    .collect::<Vec<_>>(),
                n,
            );
        }
    }
}
#[test]
fn fourier_f32() {
    fourier_cases::<f32>();
}
#[test]
fn fourier_f64() {
    fourier_cases::<f64>();
}

#[test]
fn strided_batch_transaction_and_empty_contracts() {
    let shape = [2, 3, 4];
    let input_strides = [40, 10, 2];
    let output_strides = [50, 12, 2];
    let input: Vec<_> = (0..80)
        .map(|i| Complex::new((i as f64).sin(), 0.0))
        .collect();
    let plan = FftNdPlan::new(
        FftNdPlanSpec::new(
            &shape,
            &[2, 0],
            FftDirection::Forward,
            Normalization::Orthonormal,
            FftLayout::Complex,
        )
        .unwrap(),
    )
    .unwrap();
    let mut scratch = plan.workspace(()).unwrap();
    let sentinel = Complex::new(123.0, 321.0);
    let mut output = vec![sentinel; 90];
    plan.execute_into(
        FftNdInput::Complex(view(&input, &shape, &input_strides)),
        FftNdOutput::Complex(
            NdViewMut::new(
                &mut output,
                StridedLayout::new(&shape, &output_strides).unwrap(),
            )
            .unwrap(),
        ),
        &mut scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    let dense: Vec<_> = view(&input, &shape, &input_strides)
        .iter()
        .copied()
        .collect();
    let expected: Vec<_> = direct_nd(&dense, &shape, &[2, 0], false)
        .into_iter()
        .map(|z| z / (8.0_f64).sqrt())
        .collect();
    let actual: Vec<_> = view(&output, &shape, &output_strides)
        .iter()
        .copied()
        .collect();
    assert_fft_close(&actual, &expected, 24);
    assert_eq!(output[1], sentinel);
    let before = output.clone();
    let short = scratch.len() - 1;
    assert!(
        plan.execute_into(
            FftNdInput::Complex(view(&input, &shape, &input_strides)),
            FftNdOutput::Complex(
                NdViewMut::new(
                    &mut output,
                    StridedLayout::new(&shape, &output_strides).unwrap()
                )
                .unwrap()
            ),
            &mut scratch[..short],
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(before, output);
    let mut bad = input;
    bad[0].re = f64::NAN;
    assert!(
        plan.execute_into(
            FftNdInput::Complex(view(&bad, &shape, &input_strides)),
            FftNdOutput::Complex(
                NdViewMut::new(
                    &mut output,
                    StridedLayout::new(&shape, &output_strides).unwrap()
                )
                .unwrap()
            ),
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(before, output);
    for axes in [&[0, 0][..], &[3][..]] {
        assert!(
            FftNdPlan::<f64>::new(
                FftNdPlanSpec::new(
                    &shape,
                    axes,
                    FftDirection::Forward,
                    Normalization::Inverse,
                    FftLayout::Complex
                )
                .unwrap()
            )
            .is_err()
        );
    }
    assert!(
        FftNdPlan::<f64>::new(
            FftNdPlanSpec::new(
                &[0, 3],
                &[0],
                FftDirection::Forward,
                Normalization::Inverse,
                FftLayout::Complex
            )
            .unwrap()
        )
        .is_err()
    );
    let empty = FftNdPlan::<f64>::new(
        FftNdPlanSpec::new(
            &[0, usize::MAX],
            &[1],
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(empty.workspace_requirements(()).unwrap().complex_len(), 0);
    let empty_shape = [0, usize::MAX];
    let empty_strides = [0, 0];
    let mut empty_out = [];
    empty
        .execute_into(
            FftNdInput::Real(view(&[], &empty_shape, &empty_strides)),
            FftNdOutput::Complex(
                NdViewMut::new(
                    &mut empty_out,
                    StridedLayout::new(empty.packed_shape(), &empty_strides).unwrap(),
                )
                .unwrap(),
            ),
            &mut [],
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
}

#[test]
fn packed_endpoint_planes_are_not_individually_real() {
    let shape = [3, 4];
    let real: Vec<_> = (0..12).map(|v| (v as f64).sin()).collect();
    let mut spectrum = rfftn(&real, &shape, &[0, 1], Normalization::Inverse).unwrap();
    assert!(spectrum.bins()[3].im.abs() > 0.01);
    irfftn(&spectrum, Normalization::Inverse).unwrap();
    spectrum.bins_mut()[3].im += 1.0;
    assert!(irfftn(&spectrum, Normalization::Inverse).is_err());
}

#[test]
fn shifts_and_real_tensor_products() {
    let shape = [3, 4];
    let input: Vec<_> = (0..12).map(|i| i as f64).collect();
    let shifted = fftshift_nd(&input, &shape, &[1, 0]).unwrap();
    assert_eq!(ifftshift_nd(&shifted, &shape, &[0, 1]).unwrap(), input);
    for ty in [DctType::I, DctType::II, DctType::III, DctType::IV] {
        let transformed = dctn(
            &input,
            &shape,
            &[1, 0],
            &[ty, ty],
            Normalization::Orthonormal,
        )
        .unwrap();
        let recovered = idctn(
            &transformed,
            &shape,
            &[1, 0],
            &[ty, ty],
            Normalization::Orthonormal,
        )
        .unwrap();
        for (a, b) in recovered.iter().zip(&input) {
            assert!((a - b).abs() < 1e-11);
        }
        assert!(
            (transformed.iter().map(|v| v * v).sum::<f64>()
                - input.iter().map(|v| v * v).sum::<f64>())
            .abs()
                < 1e-10
        );
    }
    for ty in [DstType::I, DstType::II, DstType::III, DstType::IV] {
        let transformed = dstn(
            &input,
            &shape,
            &[1, 0],
            &[ty, ty],
            Normalization::Orthonormal,
        )
        .unwrap();
        let recovered = idstn(
            &transformed,
            &shape,
            &[1, 0],
            &[ty, ty],
            Normalization::Orthonormal,
        )
        .unwrap();
        for (a, b) in recovered.iter().zip(&input) {
            assert!((a - b).abs() < 1e-11);
        }
    }
}

fn real_tensor_definition(
    input: &[f64],
    shape: &[usize],
    axes: &[usize],
    kinds: &[usize],
    cosine: bool,
    normalization: Normalization,
    inverse: bool,
) -> Vec<f64> {
    let strides = layout(shape);
    axes.iter()
        .zip(kinds)
        .fold(input.to_vec(), |values, (&axis, &kind)| {
            let n = shape[axis];
            let stride = strides[axis];
            let mut next = vec![0.0; values.len()];
            for start in (0..values.len()).filter(|index| (index / stride).is_multiple_of(n)) {
                let lane: Vec<_> = (0..n).map(|j| values[start + j * stride]).collect();
                let transformed = if cosine {
                    direct_dct(
                        &lane,
                        [DctType::I, DctType::II, DctType::III, DctType::IV][kind],
                        normalization,
                        inverse,
                    )
                } else {
                    direct_dst(
                        &lane,
                        [DstType::I, DstType::II, DstType::III, DstType::IV][kind],
                        normalization,
                        inverse,
                    )
                };
                for (j, value) in transformed.into_iter().enumerate() {
                    next[start + j * stride] = value;
                }
            }
            next
        })
}

fn real_tensor_result<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    kinds: &[usize],
    cosine: bool,
    normalization: Normalization,
    inverse: bool,
) -> Vec<S> {
    if cosine {
        let types: Vec<_> = kinds
            .iter()
            .map(|&k| [DctType::I, DctType::II, DctType::III, DctType::IV][k])
            .collect();
        if inverse {
            idctn(input, shape, axes, &types, normalization)
        } else {
            dctn(input, shape, axes, &types, normalization)
        }
    } else {
        let types: Vec<_> = kinds
            .iter()
            .map(|&k| [DstType::I, DstType::II, DstType::III, DstType::IV][k])
            .collect();
        if inverse {
            idstn(input, shape, axes, &types, normalization)
        } else {
            dstn(input, shape, axes, &types, normalization)
        }
    }
    .unwrap()
}

fn real_tensor_close<S: RealScalar>(actual: &[S], expected: &[f64]) {
    let magnitude = expected.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
    let bound = 512.0 * wide(S::EPSILON) * (actual.len() as f64).log2().max(1.0) * magnitude;
    numerics::assert_real_close(actual, expected, |_| bound);
}

fn real_oracles<S: RealScalar>() {
    for (shape, axes) in [
        (vec![3, 4], vec![1, 0]),
        (vec![2, 3, 4], vec![2, 0]),
        (vec![2, 3, 4], vec![2, 1]),
        (vec![2, 1, 3], vec![1]),
    ] {
        let len = shape.iter().product();
        let source: Vec<_> = (0..len)
            .map(|i| ((i * 7 % 23) as f64 - 11.0) / 8.0)
            .collect();
        let input: Vec<_> = source.iter().map(|&v| scalar::<S>(v)).collect();
        for cosine in [true, false] {
            for combination in 0..if axes.len() == 1 { 4 } else { 16 } {
                let kinds: Vec<_> = (0..axes.len())
                    .map(|i| (combination >> (2 * i)) & 3)
                    .collect();
                if cosine
                    && axes
                        .iter()
                        .zip(&kinds)
                        .any(|(&axis, &kind)| shape[axis] == 1 && kind == 0)
                {
                    assert!(
                        dctn(
                            &input,
                            &shape,
                            &axes,
                            &[DctType::I],
                            Normalization::Unscaled
                        )
                        .is_err()
                    );
                    continue;
                }
                for normalization in [
                    Normalization::Unscaled,
                    Normalization::Forward,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                ] {
                    for inverse in [false, true] {
                        let expected = real_tensor_definition(
                            &source,
                            &shape,
                            &axes,
                            &kinds,
                            cosine,
                            normalization,
                            inverse,
                        );
                        let actual = real_tensor_result(
                            &input,
                            &shape,
                            &axes,
                            &kinds,
                            cosine,
                            normalization,
                            inverse,
                        );
                        real_tensor_close(&actual, &expected);
                        // Reorder axes together with their types: the tensor product is unchanged.
                        let reversed_axes: Vec<_> = axes.iter().rev().copied().collect();
                        let reversed_kinds: Vec<_> = kinds.iter().rev().copied().collect();
                        let reordered = real_tensor_result(
                            &input,
                            &shape,
                            &reversed_axes,
                            &reversed_kinds,
                            cosine,
                            normalization,
                            inverse,
                        );
                        real_tensor_close(&reordered, &expected);
                    }
                }
            }
        }
    }
}
#[test]
fn independent_real_definitions_f32() {
    real_oracles::<f32>();
}
#[test]
fn independent_real_definitions_f64() {
    real_oracles::<f64>();
}

fn short_real_axes_many_lanes<S: RealScalar>() {
    for shape in [[2, 512], [2, 257], [1, 257]] {
        let source: Vec<_> = (0..shape.iter().product())
            .map(|i| ((i * 7 % 23) as f64 - 11.0) / 8.0)
            .collect();
        // Padded input and output exercise the outer gather/commit separately
        // from the prepared traversal of the dense candidate.
        let strides = [2 * shape[1], 2];
        let input: Vec<_> = source
            .iter()
            .flat_map(|&v| [scalar::<S>(v), S::NAN])
            .collect();
        for axes in [[0, 1], [1, 0]] {
            for cosine in [true, false] {
                for kind in 1..=3 {
                    let ty: RealTransformKind = if cosine {
                        [DctType::II, DctType::III, DctType::IV][kind - 1].into()
                    } else {
                        [DstType::II, DstType::III, DstType::IV][kind - 1].into()
                    };
                    for normalization in [
                        Normalization::Unscaled,
                        Normalization::Forward,
                        Normalization::Inverse,
                        Normalization::Orthonormal,
                    ] {
                        for direction in [FftDirection::Forward, FftDirection::Inverse] {
                            let expected = real_tensor_definition(
                                &source,
                                &shape,
                                &axes,
                                &[kind; 2],
                                cosine,
                                normalization,
                                direction == FftDirection::Inverse,
                            );
                            let plan = RealTransformNdPlan::new(
                                RealTransformNdPlanSpec::new(
                                    &shape,
                                    &axes,
                                    &[ty; 2],
                                    direction,
                                    normalization,
                                )
                                .unwrap(),
                            )
                            .unwrap();
                            let mut workspace = plan.workspace(()).unwrap();
                            let mut output = vec![scalar::<S>(123.0); input.len()];
                            let layout = StridedLayout::new(&shape, &strides).unwrap();
                            plan.execute_into(
                                NdView::new(&input, layout).unwrap(),
                                NdViewMut::new(&mut output, layout).unwrap(),
                                &mut workspace,
                                scioxide_core::ExecutionContext::sequential(),
                            )
                            .unwrap();
                            let actual: Vec<_> = output.iter().step_by(2).copied().collect();
                            real_tensor_close(&actual, &expected);
                            assert!(
                                output
                                    .iter()
                                    .skip(1)
                                    .step_by(2)
                                    .all(|&v| v == scalar::<S>(123.0))
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn short_real_axes_match_direct_sums_f32() {
    short_real_axes_many_lanes::<f32>();
}

#[test]
fn short_real_axes_match_direct_sums_f64() {
    short_real_axes_many_lanes::<f64>();
}

#[derive(serde::Deserialize)]
struct TensorFixtures {
    cases: Vec<TensorFixture>,
}

#[derive(serde::Deserialize)]
struct TensorFixture {
    width: usize,
    family: String,
    shape: Vec<usize>,
    axes: Vec<usize>,
    kinds: Vec<usize>,
    normalization: String,
    inverse: bool,
    input: Vec<f64>,
    expected: Vec<f64>,
    scipy_norm: String,
    orthogonalize: bool,
    scale: f64,
    scipy_output: Vec<f64>,
}

fn real_tensor_fixtures<S: RealScalar>(width: usize) {
    let fixtures: TensorFixtures =
        serde_json::from_str(include_str!("fixtures/real_transforms_nd.json")).unwrap();
    let cases: Vec<_> = fixtures
        .cases
        .iter()
        .filter(|case| case.width == width)
        .collect();
    assert_eq!(cases.len(), 128);
    for case in cases {
        let normalization = match case.normalization.as_str() {
            "Forward" => Normalization::Forward,
            "Inverse" => Normalization::Inverse,
            "Unscaled" => Normalization::Unscaled,
            "Orthonormal" => Normalization::Orthonormal,
            _ => unreachable!("unknown fixture normalization"),
        };
        let cosine = case.family == "dct";
        let kinds: Vec<_> = case.kinds.iter().map(|k| k - 1).collect();
        let expected_norm = match normalization {
            Normalization::Orthonormal => "ortho",
            Normalization::Forward => "forward",
            Normalization::Unscaled if case.inverse => "forward",
            Normalization::Inverse | Normalization::Unscaled => "backward",
        };
        assert_eq!(case.scipy_norm, expected_norm);
        assert_eq!(
            case.orthogonalize,
            normalization == Normalization::Orthonormal
        );
        let raw_scales = [if cosine { 1.0 } else { 0.5 }, 0.5, 1.0, 0.5];
        let scale: f64 = kinds
            .iter()
            .map(|&kind| {
                let a = raw_scales[kind];
                let b = raw_scales[[0, 2, 1, 3][kind]];
                match normalization {
                    Normalization::Orthonormal => 1.0,
                    Normalization::Forward if !case.inverse => 1.0 / b,
                    Normalization::Inverse if case.inverse => 1.0 / a,
                    _ if case.inverse => b,
                    _ => a,
                }
            })
            .product();
        assert_eq!(case.scale, scale);
        let direct = real_tensor_definition(
            &case.input,
            &case.shape,
            &case.axes,
            &kinds,
            cosine,
            normalization,
            case.inverse,
        );
        let direct: Vec<_> = direct.into_iter().map(scalar::<S>).collect();
        real_tensor_close(&direct, &case.expected);
        let mapped: Vec<_> = case
            .scipy_output
            .iter()
            .map(|&v| scalar::<S>(v * scale))
            .collect();
        real_tensor_close(&mapped, &case.expected);
        let input: Vec<_> = case.input.iter().copied().map(scalar::<S>).collect();
        let owned = real_tensor_result(
            &input,
            &case.shape,
            &case.axes,
            &kinds,
            cosine,
            normalization,
            case.inverse,
        );
        real_tensor_close(&owned, &case.expected);
        let strides = layout(&case.shape);
        let mut prepared = vec![S::ZERO; input.len()];
        let output = NdViewMut::new(
            &mut prepared,
            StridedLayout::new(&case.shape, &strides).unwrap(),
        )
        .unwrap();
        let direction = if case.inverse {
            FftDirection::Inverse
        } else {
            FftDirection::Forward
        };
        let cx = FftPlanningContext::<S>::default().execution();
        if cosine {
            let types: Vec<_> = kinds
                .iter()
                .map(|&k| [DctType::I, DctType::II, DctType::III, DctType::IV][k])
                .collect();
            let plan = RealTransformNdPlan::new(
                RealTransformNdPlanSpec::new(
                    &case.shape,
                    &case.axes,
                    &types.iter().copied().map(Into::into).collect::<Vec<_>>(),
                    direction,
                    normalization,
                )
                .unwrap(),
            )
            .unwrap();
            let mut workspace = plan.workspace(()).unwrap();
            plan.execute_into(
                view(&input, &case.shape, &strides),
                output,
                &mut workspace,
                cx,
            )
            .unwrap();
        } else {
            let types: Vec<_> = kinds
                .iter()
                .map(|&k| [DstType::I, DstType::II, DstType::III, DstType::IV][k])
                .collect();
            let plan = RealTransformNdPlan::new(
                RealTransformNdPlanSpec::new(
                    &case.shape,
                    &case.axes,
                    &types.iter().copied().map(Into::into).collect::<Vec<_>>(),
                    direction,
                    normalization,
                )
                .unwrap(),
            )
            .unwrap();
            let mut workspace = plan.workspace(()).unwrap();
            plan.execute_into(
                view(&input, &case.shape, &strides),
                output,
                &mut workspace,
                cx,
            )
            .unwrap();
        }
        real_tensor_close(&prepared, &case.expected);
    }
}

#[test]
fn real_tensor_scipy_and_defining_sum_fixtures_f32() {
    real_tensor_fixtures::<f32>(32);
}

#[test]
fn real_tensor_scipy_and_defining_sum_fixtures_f64() {
    real_tensor_fixtures::<f64>(64);
}

#[test]
fn packed_forward_matches_full_dft_for_nonlast_axis() {
    let shape = [5, 2, 3];
    let axes = [2, 0];
    let real: Vec<_> = (0..30).map(|i| (i as f64 * 0.8).cos()).collect();
    let full: Vec<_> = real.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let expected = direct_nd(&full, &shape, &axes, false);
    let packed = rfftn(&real, &shape, &axes, Normalization::Inverse).unwrap();
    assert_fft_close(packed.bins(), &expected[..18], 30);
    let hermitian = ihfftn(&real, &shape, &axes, Normalization::Unscaled).unwrap();
    assert_fft_close(
        hermitian.bins(),
        &expected[..18].iter().map(|z| z.conj()).collect::<Vec<_>>(),
        30,
    );
}

#[test]
fn injected_capabilities_and_broadcast_input() {
    use scioxide_core::backend::KernelBackend;
    use scioxide_core::{CancellationToken, ExecutionContext, ExecutionPolicy};
    let shape = [2, 3, 4];
    let strides = layout(&shape);
    let zero = Complex::new(0.0_f64, 0.0);
    let input = vec![Complex::new(1.0, 0.0); 24];
    let spec = FftNdPlanSpec::new(
        &shape,
        &[2, 0],
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    )
    .unwrap();
    let plan = FftNdPlan::build(
        spec.clone()
            .with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut output = vec![zero; 24];
    let mut scratch = plan.workspace(()).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let error = plan
        .execute_into(
            FftNdInput::Complex(view(&input, &shape, &strides)),
            FftNdOutput::Complex(
                NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
            ),
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    assert!(output.iter().all(|&v| v == zero));
    let broadcast = StridedLayout::new_broadcast(&shape, &[0, 0, 0]).unwrap();
    plan.execute_into(
        FftNdInput::Complex(NdView::new(&input[..1], broadcast).unwrap()),
        FftNdOutput::Complex(
            NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        ),
        &mut scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    let expected = direct_nd(&input, &shape, &[2, 0], false);
    assert_fft_close(&output, &expected, 24);
    #[cfg(feature = "std")]
    for backend in numerics::fft_backends(&scioxide_core::StdProcessor) {
        let candidate = FftNdPlan::new(
            spec.clone()
                .with_execution_policy(ExecutionPolicy::Sequential)
                .with_backend_strategy(FftBackendStrategy::Required(backend)),
        )
        .unwrap();
        let mut scratch = candidate.workspace(()).unwrap();
        let mut actual = input.clone();
        candidate
            .execute_in_place(
                NdViewMut::new(&mut actual, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
        assert_fft_close(&actual, &expected, 24);
    }
    #[cfg(not(feature = "std"))]
    let _ = (spec, ExecutionPolicy::Sequential);
}

#[test]
fn bounded_child_deadline_names_the_public_caller() {
    use scioxide_core::{Clock, ClockBinding, Deadline, ExecutionContext, MonotonicInstant};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    struct LaneClock(AtomicU64);
    impl Clock for LaneClock {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_nanos(self.0.fetch_add(1, Ordering::Relaxed)),
            ))
        }
    }

    let shape = [2, 4];
    let strides = layout(&shape);
    let plan = FftNdPlan::<f64>::build(
        FftNdPlanSpec::new(
            &shape,
            &[1, 0],
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Complex,
        )
        .unwrap(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let input = vec![Complex::new(0.25, -0.125); 8];
    let mut scratch = plan.workspace(()).unwrap();
    for in_place in [false, true] {
        let clock = LaneClock(AtomicU64::new(0));
        let binding = ClockBinding::new(&clock);
        // The public entry check passes; the bounded child's check expires.
        let context = ExecutionContext::sequential()
            .with_deadline(Deadline::at(
                &binding,
                MonotonicInstant::from_duration_since_epoch(Duration::from_nanos(1)),
            ))
            .unwrap();
        let mut output = input.clone();
        let target =
            NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
        let result = if in_place {
            plan.execute_in_place(target, &mut scratch, context)
        } else {
            plan.execute_into(
                FftNdInput::Complex(view(&input, &shape, &strides)),
                FftNdOutput::Complex(target),
                &mut scratch,
                context,
            )
        };
        let error = result.unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ExecutionStopped {
                reason: scioxide_core::ExecutionStopReason::DeadlineReached,
                ..
            }
        ));
        assert_eq!(clock.0.load(Ordering::Relaxed), 2);
        assert_eq!(output, input);
    }
}

#[test]
fn short_axis_blocks_check_deadlines_before_publication() {
    use scioxide_core::{Clock, ClockBinding, Deadline, ExecutionContext, MonotonicInstant};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    struct BlockClock(AtomicU64);
    impl Clock for BlockClock {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_nanos(self.0.fetch_add(1, Ordering::Relaxed)),
            ))
        }
    }
    let shape = [2, 514];
    let layout = StridedLayout::new(&shape, &[514, 1]).unwrap();
    let fft = FftNdPlan::<f64>::new(
        FftNdPlanSpec::new(
            &shape,
            &[0, 1],
            FftDirection::Forward,
            Normalization::Orthonormal,
            FftLayout::Complex,
        )
        .unwrap(),
    )
    .unwrap();
    let real = RealTransformNdPlan::<f64>::new(
        RealTransformNdPlanSpec::new(
            &shape,
            &[0, 1],
            &[DctType::IV.into(); 2],
            FftDirection::Forward,
            Normalization::Orthonormal,
        )
        .unwrap(),
    )
    .unwrap();
    let mut fft_work = fft.workspace(()).unwrap();
    let mut real_work = real.workspace(()).unwrap();
    // Entry and first-block checks pass. Stop at the second or third block of
    // the short axis, before the long-axis child or the final publication.
    for stop in [2, 3] {
        for cosine in [false, true] {
            let clock = BlockClock(AtomicU64::new(0));
            let binding = ClockBinding::new(&clock);
            let context = ExecutionContext::sequential()
                .with_deadline(Deadline::at(
                    &binding,
                    MonotonicInstant::from_duration_since_epoch(Duration::from_nanos(stop)),
                ))
                .unwrap();
            let mut complex = vec![Complex::new(0.25, -0.125); 1028];
            let mut values = vec![0.25; 1028];
            let error = if cosine {
                real.execute_in_place(
                    NdViewMut::new(&mut values, layout).unwrap(),
                    &mut real_work,
                    context,
                )
            } else {
                fft.execute_in_place(
                    NdViewMut::new(&mut complex, layout).unwrap(),
                    &mut fft_work,
                    context,
                )
            }
            .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped {
                    reason: scioxide_core::ExecutionStopReason::DeadlineReached,
                    ..
                }
            ));
            assert_eq!(clock.0.load(Ordering::Relaxed), stop + 1);
            assert!(values.iter().all(|&v| v == 0.25));
            assert!(complex.iter().all(|&v| v == Complex::new(0.25, -0.125)));
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_runs_large_axis_and_preserves_values() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, ThreadExecutor};
    use std::num::NonZeroUsize;
    let executor = ThreadExecutor::new(NonZeroUsize::new(2).unwrap());
    let shape = [2, 65536];
    let strides = [65536, 1];
    let input = vec![Complex::new(1.0_f64, 0.0); 131072];
    let spec = FftNdPlanSpec::new(
        &shape,
        &[1],
        FftDirection::Forward,
        Normalization::Forward,
        FftLayout::Complex,
    )
    .unwrap()
    .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let context = ExecutionContext::new(&executor);
    let plan = FftNdPlan::build(spec, context).unwrap();
    let mut scratch = plan.workspace(()).unwrap();
    let mut output = input;
    plan.execute_in_place(
        NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        &mut scratch,
        context,
    )
    .unwrap();
    for (i, z) in output.iter().enumerate() {
        assert!(
            (z.re - if i % 65536 == 0 { 1.0 } else { 0.0 }).abs() < 1e-12 && z.im.abs() < 1e-12
        );
    }
}

fn hermitian_plane_cases<S: RealScalar>() {
    let exponents: &[i32] = if S::MANTISSA_DIGITS == 53 {
        &[-900, 0, 900]
    } else {
        &[-100, 0, 100]
    };
    for n in [5_usize, 6] {
        let shape = [3, n];
        let packed_shape = [3, n / 2 + 1];
        let width = packed_shape[1];
        for scale in exponents
            .iter()
            .map(|&e| S::ONE.scalbn(e))
            .chain([S::MIN_SUBNORMAL.scalbn(4)])
        {
            let allowance = (S::EPSILON.scalbn(3) * scale).max(S::MIN_SUBNORMAL);
            let mut bins = vec![Complex::new(S::ZERO, S::ZERO); 3 * width];
            bins[1].re = scale;
            // These endpoint-plane bins are a legitimate non-real pair.
            bins[width].im = scale * S::HALF;
            bins[2 * width].im = -scale * S::HALF;
            if n.is_multiple_of(2) {
                bins[2 * width - 1].im = scale * S::HALF;
                bins[3 * width - 1].im = -scale * S::HALF;
            } else {
                // No Nyquist constraint applies to the final odd-length plane.
                bins[width - 1].im = scale;
            }
            let original = HermitianSpectrumNd::from_vec(&shape, &[0, 1], bins.clone()).unwrap();
            irfftn(&original, Normalization::Inverse).unwrap();
            hfftn(&original, Normalization::Inverse).unwrap();
            for endpoint in if n.is_multiple_of(2) {
                vec![0, width - 1]
            } else {
                vec![0]
            } {
                for accepted in [true, false] {
                    let mismatch = if accepted {
                        (allowance * S::TWO).next_after(S::ZERO)
                    } else {
                        (allowance * S::TWO).next_after(S::INFINITY)
                    };
                    let mut perturbed = bins.clone();
                    perturbed[width + endpoint].re = mismatch;
                    let spectrum =
                        HermitianSpectrumNd::from_vec(&shape, &[0, 1], perturbed.clone()).unwrap();
                    if accepted {
                        let average = scioxide_core::checked_midpoint(mismatch, S::ZERO).unwrap();
                        perturbed[width + endpoint].re = average;
                        perturbed[2 * width + endpoint].re = average;
                        let projected =
                            HermitianSpectrumNd::from_vec(&shape, &[0, 1], perturbed).unwrap();
                        for (actual, expected) in [
                            (
                                irfftn(&spectrum, Normalization::Inverse).unwrap(),
                                irfftn(&projected, Normalization::Inverse).unwrap(),
                            ),
                            (
                                hfftn(&spectrum, Normalization::Inverse).unwrap(),
                                hfftn(&projected, Normalization::Inverse).unwrap(),
                            ),
                        ] {
                            for (a, b) in actual.iter().zip(&expected) {
                                assert!(
                                    (*a - *b).abs()
                                        <= (S::EPSILON.scalbn(7) * scale).max(S::MIN_SUBNORMAL)
                                );
                            }
                        }
                        assert_eq!(spectrum.bins(), &{
                            let mut input = bins.clone();
                            input[width + endpoint].re = mismatch;
                            input
                        });
                    } else {
                        assert_eq!(
                            irfftn(&spectrum, Normalization::Inverse)
                                .unwrap_err()
                                .kind(),
                            scioxide_core::SciOxideErrorKind::InvalidInput
                        );
                        assert_eq!(
                            hfftn(&spectrum, Normalization::Inverse).unwrap_err().kind(),
                            scioxide_core::SciOxideErrorKind::InvalidInput
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn hermitian_plane_policy_f32() {
    hermitian_plane_cases::<f32>();
}

#[test]
fn hermitian_plane_policy_f64() {
    hermitian_plane_cases::<f64>();
}

#[test]
fn hermitian_batches_validate_before_range_scaling_and_preserve_strided_output() {
    let shape = [2, 3, 5];
    let packed_shape = [2, 3, 3];
    let packed_strides = [18, 6, 2];
    let output_strides = [30, 10, 2];
    let mut input = vec![Complex::new(f64::NAN, f64::NAN); 36];
    for slot in input.iter_mut().step_by(2) {
        *slot = Complex::new(0.0, 0.0);
    }
    input[2].re = 1e-300;
    input[20].re = 1e300;
    // This invalid tiny DC would disappear if checked after global scaling,
    // and must not borrow the large independent batch's allowance.
    input[0].im = 1e-301;
    for (family, direction) in [
        (FftLayout::Real, FftDirection::Inverse),
        (FftLayout::Hermitian, FftDirection::Forward),
    ] {
        let plan = FftNdPlan::<f64>::build(
            FftNdPlanSpec::new(&shape, &[1, 2], direction, Normalization::Inverse, family).unwrap(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let mut work = plan.workspace(()).unwrap();
        let mut output = vec![77.0; 60];
        for invalid in [1e-301, f64::NAN, f64::INFINITY] {
            input[0].im = invalid;
            let error = plan
                .execute_into(
                    FftNdInput::Complex(view(&input, &packed_shape, &packed_strides)),
                    FftNdOutput::Real(
                        NdViewMut::new(
                            &mut output,
                            StridedLayout::new(&shape, &output_strides).unwrap(),
                        )
                        .unwrap(),
                    ),
                    &mut work,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(
                error.kind(),
                if invalid.is_finite() {
                    scioxide_core::SciOxideErrorKind::InvalidInput
                } else {
                    scioxide_core::SciOxideErrorKind::NonFinite
                }
            );
            assert_eq!(output, vec![77.0; 60]);
        }
        input[0].im = 0.0;
        plan.execute_into(
            FftNdInput::Complex(view(&input, &packed_shape, &packed_strides)),
            FftNdOutput::Real(
                NdViewMut::new(
                    &mut output,
                    StridedLayout::new(&shape, &output_strides).unwrap(),
                )
                .unwrap(),
            ),
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(output.iter().skip(1).step_by(2).all(|&gap| gap == 77.0));
    }
}

#[test]
fn hermitian_endpoint_cancellation_and_selected_axis_order() {
    for axes in [&[0, 1][..], &[1, 0][..]] {
        let shape = [3, 5];
        let real: Vec<_> = (0..15)
            .map(|i| {
                if i % 3 == 0 {
                    0.0_f64
                } else {
                    (i as f64).sin()
                }
            })
            .collect();
        let spectrum = rfftn(&real, &shape, axes, Normalization::Inverse).unwrap();
        let actual = irfftn(&spectrum, Normalization::Inverse).unwrap();
        for (a, b) in actual.iter().zip(&real) {
            assert!((a - b).abs() < 1e-13);
        }
    }
    // Conjugate endpoints invert along the first axis to one nonzero lane
    // and two cancellation-dominated lanes; those lanes are not new user inputs.
    let shape = [3, 5];
    let mut bins = vec![Complex::new(0.0_f64, 0.0); 9];
    bins[0].re = 1.0;
    bins[3].re = 1.0;
    bins[6].re = 1.0;
    let spectrum = HermitianSpectrumNd::from_vec(&shape, &[0, 1], bins).unwrap();
    let output = irfftn(&spectrum, Normalization::Inverse).unwrap();
    assert!(output[5..].iter().all(|value| value.abs() < 1e-14));
}

// Compare in each independent batch's frame: an absolute floor of one would
// incorrectly accept a tiny batch that had been flushed entirely to zero.
fn batch_close<S: RealScalar>(actual: &[S], expected: &[S], scale: S)
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let scale = wide(scale);
    let norm = expected
        .iter()
        .map(|&v| (wide(v) / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| ((wide(a) - wide(b)) / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    let tolerance = 8.0 * wide(S::EPSILON) * (expected.len() as f64).log2().max(1.0) * norm;
    assert!(
        error <= tolerance,
        "batch error {error:e} > {tolerance:e}; scale={scale:e}"
    );
}

fn independent_batch_cases<S>()
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let ranges: &[(f64, f64)] = if S::MANTISSA_DIGITS == 53 {
        &[(1e-300, 1e300)]
    } else {
        &[(1e-19, 1e19), (1e-20, 1e19), (1e-35, 1e35)]
    };
    let cx = FftPlanningContext::<S>::default().execution();
    for &(tiny, large) in ranges {
        let scales = [scalar::<S>(tiny), scalar::<S>(large)];
        for n in [4, 5, 8] {
            for axis in [0, 1] {
                let shape = if axis == 1 { [2, n] } else { [n, 2] };
                let strides = layout(&shape);
                let index = |batch, i| {
                    if axis == 1 {
                        batch * n + i
                    } else {
                        i * 2 + batch
                    }
                };
                let mut input = vec![S::ZERO; 2 * n];
                for (batch, scale) in scales.iter().enumerate() {
                    for i in 0..n {
                        input[index(batch, i)] = scalar::<S>((i + 1) as f64) * *scale;
                    }
                }
                let complex: Vec<_> = input
                    .iter()
                    .map(|&v| Complex::new(v, -v * S::HALF))
                    .collect();
                let rows: Vec<Vec<_>> = (0..2)
                    .map(|b| (0..n).map(|i| input[index(b, i)]).collect())
                    .collect();
                let complex_rows: Vec<Vec<_>> = (0..2)
                    .map(|b| (0..n).map(|i| complex[index(b, i)]).collect())
                    .collect();
                for norm in [
                    Normalization::Forward,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                    Normalization::Unscaled,
                ] {
                    for direction in [FftDirection::Forward, FftDirection::Inverse] {
                        let plan = FftNdPlan::new(
                            FftNdPlanSpec::new(
                                &shape,
                                &[axis],
                                direction,
                                norm,
                                FftLayout::Complex,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                        let mut work = plan.workspace(()).unwrap();
                        let sentinel = Complex::new(S::TEN, S::TEN);
                        let mut output = vec![sentinel; 4 * n];
                        let output_strides: Vec<_> = strides.iter().map(|s| 2 * s).collect();
                        plan.execute_into(
                            FftNdInput::Complex(view(&complex, &shape, &strides)),
                            FftNdOutput::Complex(
                                NdViewMut::new(
                                    &mut output,
                                    StridedLayout::new(&shape, &output_strides).unwrap(),
                                )
                                .unwrap(),
                            ),
                            &mut work,
                            cx,
                        )
                        .unwrap();
                        assert!(output.iter().skip(1).step_by(2).all(|&v| v == sentinel));
                        let mut in_place = complex.clone();
                        plan.execute_in_place(
                            NdViewMut::new(
                                &mut in_place,
                                StridedLayout::new(&shape, &strides).unwrap(),
                            )
                            .unwrap(),
                            &mut work,
                            cx,
                        )
                        .unwrap();
                        let allocating = if direction == FftDirection::Forward {
                            fftn(&complex, &shape, &[axis], norm)
                        } else {
                            ifftn(&complex, &shape, &[axis], norm)
                        }
                        .unwrap();
                        for batch in 0..2 {
                            let reference = if direction == FftDirection::Forward {
                                fft(&complex_rows[batch], norm)
                            } else {
                                ifft(&complex_rows[batch], norm)
                            }
                            .unwrap();
                            let expected: Vec<_> =
                                reference.iter().flat_map(|v| [v.re, v.im]).collect();
                            for values in [&in_place, &allocating] {
                                let actual: Vec<_> = (0..n)
                                    .flat_map(|i| {
                                        let v = values[index(batch, i)];
                                        [v.re, v.im]
                                    })
                                    .collect();
                                batch_close(&actual, &expected, scales[batch]);
                            }
                            let actual: Vec<_> = (0..n)
                                .flat_map(|i| {
                                    let v = output[2 * index(batch, i)];
                                    [v.re, v.im]
                                })
                                .collect();
                            batch_close(&actual, &expected, scales[batch]);
                        }
                    }
                    let m = n / 2 + 1;
                    let packed_shape = if axis == 1 { [2, m] } else { [m, 2] };
                    let packed_strides = layout(&packed_shape);
                    let packed_index = |batch, i| {
                        if axis == 1 {
                            batch * m + i
                        } else {
                            i * 2 + batch
                        }
                    };
                    for (family, forward, inverse) in [
                        (
                            FftLayout::Real,
                            FftDirection::Forward,
                            FftDirection::Inverse,
                        ),
                        (
                            FftLayout::Hermitian,
                            FftDirection::Inverse,
                            FftDirection::Forward,
                        ),
                    ] {
                        let result = if family == FftLayout::Real {
                            rfftn(&input, &shape, &[axis], norm)
                        } else {
                            ihfftn(&input, &shape, &[axis], norm)
                        }
                        .unwrap();
                        let plan = FftNdPlan::new(
                            FftNdPlanSpec::new(&shape, &[axis], forward, norm, family).unwrap(),
                        )
                        .unwrap();
                        let mut planned = vec![Complex::new(S::ZERO, S::ZERO); 2 * m];
                        plan.execute_into(
                            FftNdInput::Real(view(&input, &shape, &strides)),
                            FftNdOutput::Complex(
                                NdViewMut::new(
                                    &mut planned,
                                    StridedLayout::new(&packed_shape, &packed_strides).unwrap(),
                                )
                                .unwrap(),
                            ),
                            &mut plan.workspace(()).unwrap(),
                            cx,
                        )
                        .unwrap();
                        for batch in 0..2 {
                            let reference = if family == FftLayout::Real {
                                rfft(&rows[batch], norm)
                            } else {
                                ihfft(&rows[batch], norm)
                            }
                            .unwrap();
                            let expected: Vec<_> =
                                reference.bins().iter().flat_map(|v| [v.re, v.im]).collect();
                            for values in [result.bins(), &planned] {
                                let actual: Vec<_> = (0..m)
                                    .flat_map(|i| {
                                        let v = values[packed_index(batch, i)];
                                        [v.re, v.im]
                                    })
                                    .collect();
                                batch_close(&actual, &expected, scales[batch]);
                            }
                        }
                        let recovered = if family == FftLayout::Real {
                            irfftn(&result, norm)
                        } else {
                            hfftn(&result, norm)
                        }
                        .unwrap();
                        let plan = FftNdPlan::new(
                            FftNdPlanSpec::new(&shape, &[axis], inverse, norm, family).unwrap(),
                        )
                        .unwrap();
                        let mut output = vec![S::ZERO; 2 * n];
                        plan.execute_into(
                            FftNdInput::Complex(view(
                                result.bins(),
                                &packed_shape,
                                &packed_strides,
                            )),
                            FftNdOutput::Real(
                                NdViewMut::new(
                                    &mut output,
                                    StridedLayout::new(&shape, &strides).unwrap(),
                                )
                                .unwrap(),
                            ),
                            &mut plan.workspace(()).unwrap(),
                            cx,
                        )
                        .unwrap();
                        for batch in 0..2 {
                            let spectrum = HermitianSpectrum::from_vec(
                                n,
                                (0..m)
                                    .map(|i| result.bins()[packed_index(batch, i)])
                                    .collect(),
                            )
                            .unwrap();
                            let expected = if family == FftLayout::Real {
                                irfft(&spectrum, norm)
                            } else {
                                hfft(&spectrum, norm)
                            }
                            .unwrap();
                            for values in [&recovered, &output] {
                                let actual: Vec<_> =
                                    (0..n).map(|i| values[index(batch, i)]).collect();
                                batch_close(&actual, &expected, scales[batch]);
                            }
                        }
                    }
                    macro_rules! check_real {
                        ($plan:ident, $spec:ident, $kind:ident, $one:ident, $inverse:ident, $many:ident, $inverse_many:ident) => {
                            for kind in [$kind::I, $kind::II, $kind::III, $kind::IV] {
                                for direction in [FftDirection::Forward, FftDirection::Inverse] {
                                    let plan = $plan::new(
                                        $spec::new(
                                            &shape,
                                            &[axis],
                                            &[kind.into()],
                                            direction,
                                            norm,
                                        )
                                        .unwrap(),
                                    )
                                    .unwrap();
                                    let mut workspace = plan.workspace(()).unwrap();
                                    let mut actual = vec![S::ZERO; 2 * n];
                                    plan.execute_into(
                                        view(&input, &shape, &strides),
                                        NdViewMut::new(
                                            &mut actual,
                                            StridedLayout::new(&shape, &strides).unwrap(),
                                        )
                                        .unwrap(),
                                        &mut workspace,
                                        cx,
                                    )
                                    .unwrap();
                                    let allocating = if direction == FftDirection::Forward {
                                        $many(&input, &shape, &[axis], &[kind], norm)
                                    } else {
                                        $inverse_many(&input, &shape, &[axis], &[kind], norm)
                                    }
                                    .unwrap();
                                    let mut in_place = input.clone();
                                    plan.execute_in_place(
                                        NdViewMut::new(
                                            &mut in_place,
                                            StridedLayout::new(&shape, &strides).unwrap(),
                                        )
                                        .unwrap(),
                                        &mut workspace,
                                        cx,
                                    )
                                    .unwrap();
                                    for batch in 0..2 {
                                        let expected = if direction == FftDirection::Forward {
                                            $one(&rows[batch], kind, norm)
                                        } else {
                                            $inverse(&rows[batch], kind, norm)
                                        }
                                        .unwrap();
                                        for values in [&actual, &allocating, &in_place] {
                                            let batch_values: Vec<_> =
                                                (0..n).map(|i| values[index(batch, i)]).collect();
                                            batch_close(&batch_values, &expected, scales[batch]);
                                        }
                                    }
                                }
                            }
                        };
                    }
                    check_real!(
                        RealTransformNdPlan,
                        RealTransformNdPlanSpec,
                        DctType,
                        dct,
                        idct,
                        dctn,
                        idctn
                    );
                    check_real!(
                        RealTransformNdPlan,
                        RealTransformNdPlanSpec,
                        DstType,
                        dst,
                        idst,
                        dstn,
                        idstn
                    );
                }
            }
        }
    }
}

#[test]
fn independent_batches_keep_their_range_f32() {
    independent_batch_cases::<f32>();
}
#[test]
fn independent_batches_keep_their_range_f64() {
    independent_batch_cases::<f64>();
}

fn multi_axis_batch_ranges<S>(short: usize, long: usize)
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let magnitudes = if S::MANTISSA_DIGITS == 53 {
        [1e-300, 1e300]
    } else {
        [1e-20, 1e19]
    };
    let scales = magnitudes.map(scalar::<S>);
    let shape = [short, 2, long];
    let local_shape = [short, long];
    let len = short * long;
    let rows: Vec<Vec<_>> = scales
        .iter()
        .map(|&scale| {
            (0..len)
                .map(|i| scalar::<S>((i % 7) as f64 - 2.0) * scale)
                .collect()
        })
        .collect();
    let mut input = vec![S::ZERO; 2 * len];
    for (batch, row) in rows.iter().enumerate() {
        for (i, &value) in row.iter().enumerate() {
            input[(i / long * 2 + batch) * long + i % long] = value;
        }
    }
    for axes in [[0, 2], [2, 0]] {
        let local_axes = axes.map(|axis| if axis == 0 { 0 } else { 1 });
        let norm = Normalization::Orthonormal;
        let complex: Vec<_> = input.iter().map(|&v| Complex::new(v, -v)).collect();
        let transformed = fftn(&complex, &shape, &axes, norm).unwrap();
        let cosine = dctn(&input, &shape, &axes, &[DctType::II; 2], norm).unwrap();
        let sine = dstn(&input, &shape, &axes, &[DstType::III; 2], norm).unwrap();
        let packed = rfftn(&input, &shape, &axes, norm).unwrap();
        let recovered = irfftn(&packed, norm).unwrap();
        let hermitian = hfftn(&packed, norm).unwrap();
        for batch in 0..2 {
            let local_complex: Vec<_> = rows[batch].iter().map(|&v| Complex::new(v, -v)).collect();
            let expected = fftn(&local_complex, &local_shape, &local_axes, norm).unwrap();
            let actual: Vec<_> = (0..len)
                .flat_map(|i| {
                    let z = transformed[(i / long * 2 + batch) * long + i % long];
                    [z.re, z.im]
                })
                .collect();
            batch_close(
                &actual,
                &expected
                    .iter()
                    .flat_map(|z| [z.re, z.im])
                    .collect::<Vec<_>>(),
                scales[batch],
            );
            let expected_cosine = dctn(
                &rows[batch],
                &local_shape,
                &local_axes,
                &[DctType::II; 2],
                norm,
            )
            .unwrap();
            let expected_sine = dstn(
                &rows[batch],
                &local_shape,
                &local_axes,
                &[DstType::III; 2],
                norm,
            )
            .unwrap();
            let local_packed = rfftn(&rows[batch], &local_shape, &local_axes, norm).unwrap();
            let expected_hermitian = hfftn(&local_packed, norm).unwrap();
            for (actual, expected) in [
                (&cosine, &expected_cosine),
                (&sine, &expected_sine),
                (&recovered, &rows[batch]),
                (&hermitian, &expected_hermitian),
            ] {
                let actual: Vec<_> = (0..len)
                    .map(|i| actual[(i / long * 2 + batch) * long + i % long])
                    .collect();
                batch_close(&actual, expected, scales[batch]);
            }
        }
    }
}
#[test]
fn multi_axis_independent_batches_f32() {
    multi_axis_batch_ranges::<f32>(3, 5);
    multi_axis_batch_ranges::<f32>(2, 257);
}
#[test]
fn multi_axis_independent_batches_f64() {
    multi_axis_batch_ranges::<f64>(3, 5);
    multi_axis_batch_ranges::<f64>(2, 257);
}

#[test]
fn later_batch_failure_preserves_the_whole_transaction() {
    for n in [2, 4] {
        let shape = [2, n];
        let strides = [n, 1];
        let cx = FftPlanningContext::<f64>::default().execution();
        for invalid in [f64::MAX, f64::NAN] {
            let input: Vec<_> = (1..=n)
                .map(|i| i as f64)
                .chain(core::iter::repeat_n(invalid, n))
                .collect();
            let complex: Vec<_> = input.iter().map(|&v| Complex::new(v, 0.0)).collect();
            let plan = FftNdPlan::new(
                FftNdPlanSpec::new(
                    &shape,
                    &[1],
                    FftDirection::Forward,
                    Normalization::Unscaled,
                    FftLayout::Complex,
                )
                .unwrap(),
            )
            .unwrap();
            let mut output = vec![Complex::new(77.0, -31.0); 4 * n];
            assert!(
                plan.execute_into(
                    FftNdInput::Complex(view(&complex, &shape, &strides)),
                    FftNdOutput::Complex(
                        NdViewMut::new(
                            &mut output,
                            StridedLayout::new(&shape, &[2 * n, 2]).unwrap()
                        )
                        .unwrap()
                    ),
                    &mut plan.workspace(()).unwrap(),
                    cx
                )
                .is_err()
            );
            assert_eq!(output, vec![Complex::new(77.0, -31.0); 4 * n]);
            let plan = RealTransformNdPlan::new(
                RealTransformNdPlanSpec::new(
                    &shape,
                    &[1],
                    &[DctType::II].map(Into::into),
                    scioxide_fft::FftDirection::Forward,
                    Normalization::Unscaled,
                )
                .unwrap(),
            )
            .unwrap();
            let mut output = vec![77.0; 4 * n];
            assert!(
                plan.execute_into(
                    view(&input, &shape, &strides),
                    NdViewMut::new(
                        &mut output,
                        StridedLayout::new(&shape, &[2 * n, 2]).unwrap()
                    )
                    .unwrap(),
                    &mut plan.workspace(()).unwrap(),
                    cx
                )
                .is_err()
            );
            assert_eq!(output, vec![77.0; 4 * n]);
        }
    }
}

#[test]
fn normalized_matrix_transforms_avoid_intermediate_overflow() {
    let shape = [2, 2];
    for (normalization, value) in [
        (Normalization::Forward, f64::MAX),
        (Normalization::Orthonormal, f64::MAX / 2.0),
    ] {
        let real = [value; 4];
        let complex = real.map(|x| Complex::new(x, 0.0));
        let output = fftn(&complex, &shape, &[1, 0], normalization).unwrap();
        let packed = rfftn(&real, &shape, &[1, 0], normalization).unwrap();
        for bins in [&output[..], packed.bins()] {
            assert!(bins.iter().all(|z| z.is_finite()));
            assert!((bins[0].re - f64::MAX).abs() <= 8.0 * f64::EPSILON * f64::MAX);
        }
    }
    let real = [f64::MAX, f64::MAX, f64::MAX, -f64::MAX];
    let complex = real.map(|x| Complex::new(x, 0.0));
    let output = fftn(&complex, &shape, &[1, 0], Normalization::Orthonormal).unwrap();
    let packed = rfftn(&real, &shape, &[1, 0], Normalization::Orthonormal).unwrap();
    for bins in [&output[..], packed.bins()] {
        for (actual, expected) in bins.iter().zip(&complex) {
            assert!((actual.re - expected.re).abs() <= 8.0 * f64::EPSILON * f64::MAX);
            assert_eq!(actual.im, 0.0);
        }
    }
}

#[test]
fn each_nd_shift_matches_numpy_permutation() {
    #[derive(serde::Deserialize)]
    struct Case {
        shape: Vec<usize>,
        axes: Vec<usize>,
        forward: Vec<f64>,
        inverse: Vec<f64>,
    }
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/shifts.json")).unwrap();
    for case in cases {
        let len: usize = case.shape.iter().product();
        let input: Vec<_> = (0..len).map(|i| i as f64).collect();
        assert_eq!(
            fftshift_nd(&input, &case.shape, &case.axes).unwrap(),
            case.forward
        );
        assert_eq!(
            ifftshift_nd(&input, &case.shape, &case.axes).unwrap(),
            case.inverse
        );
    }
}

// Row-major packed subset of a full spectrum, keeping the last selected axis
// up to its Nyquist bin.
fn pack(full: &[Complex<f64>], shape: &[usize], packed_shape: &[usize]) -> Vec<Complex<f64>> {
    let (strides, packed_strides) = (layout(shape), layout(packed_shape));
    (0..packed_shape.iter().product())
        .map(|index| {
            let offset: usize = packed_shape
                .iter()
                .zip(&packed_strides)
                .zip(&strides)
                .map(|((&extent, &packed), &full)| index / packed % extent * full)
                .sum();
            full[offset]
        })
        .collect()
}

// Values followed by one gap each, and the strides of that padded layout.
fn padded<T: Copy>(values: &[T], gap: T, shape: &[usize]) -> (Vec<T>, Vec<usize>) {
    let values = values.iter().flat_map(|&v| [v, gap]).collect();
    (values, layout(shape).iter().map(|&s| 2 * s).collect())
}

fn unpadded<T: Copy + PartialEq + core::fmt::Debug>(values: &[T], gap: T) -> Vec<T> {
    assert!(values.iter().skip(1).step_by(2).all(|&v| v == gap));
    values.iter().step_by(2).copied().collect()
}

fn complex_of<S: RealScalar>(values: &[Complex<f64>]) -> Vec<Complex<S>>
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    values
        .iter()
        .map(|z| Complex::new(scalar::<S>(z.re), scalar::<S>(z.im)))
        .collect()
}

fn short_strided_lanes<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let gap = scalar::<S>(123.0);
    let complex_gap = Complex::new(gap, -gap);
    let cx = scioxide_core::ExecutionContext::sequential();
    for n in [1, 2, 3, 4, 5, 8, 12, 16, 17] {
        // The short axis is strided. 37 lanes leave partial staging blocks
        // and partial four-lane tiles; [3, n, 7] merges runs of seven lanes.
        // Axes [1, 0] make the short axis the packed real axis; one selected
        // axis makes every lane its own batch.
        for (shape, axes) in [
            (vec![n, 37], vec![0, 1]),
            (vec![n, 37], vec![1, 0]),
            (vec![n, 37], vec![0]),
            (vec![3, n, 7], vec![1]),
            (vec![3, n, 7], vec![2, 1, 0]),
        ] {
            let len: usize = shape.iter().product();
            let work = axes.iter().map(|&a| shape[a]).product::<usize>();
            let source: Vec<_> = (0..len)
                .map(|i| Complex::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos() - 0.5))
                .collect();
            let source = complex_of::<S>(&source);
            let exact: Vec<_> = source
                .iter()
                .map(|z| Complex::new(wide(z.re), wide(z.im)))
                .collect();

            // Complex layout, both directions, orthonormal and unscaled.
            let (input, strides) = padded(&source, Complex::new(S::NAN, S::NAN), &shape);
            for inverse in [false, true] {
                for norm in [Normalization::Unscaled, Normalization::Orthonormal] {
                    let scale = if norm == Normalization::Orthonormal {
                        (work as f64).sqrt().recip()
                    } else {
                        1.0
                    };
                    let expected: Vec<_> = direct_nd(&exact, &shape, &axes, inverse)
                        .into_iter()
                        .map(|z| z * scale)
                        .collect();
                    let direction = if inverse {
                        FftDirection::Inverse
                    } else {
                        FftDirection::Forward
                    };
                    let plan = FftNdPlan::<S>::new(
                        FftNdPlanSpec::new(&shape, &axes, direction, norm, FftLayout::Complex)
                            .unwrap(),
                    )
                    .unwrap();
                    let mut output = vec![complex_gap; 2 * len];
                    plan.execute_into(
                        FftNdInput::Complex(view(&input, &shape, &strides)),
                        FftNdOutput::Complex(
                            NdViewMut::new(
                                &mut output,
                                StridedLayout::new(&shape, &strides).unwrap(),
                            )
                            .unwrap(),
                        ),
                        &mut plan.workspace(()).unwrap(),
                        cx,
                    )
                    .unwrap();
                    assert_fft_close(&unpadded(&output, complex_gap), &expected, work);
                }
            }

            // Real and Hermitian layouts against direct sums of a known real
            // signal; inverse normalization scales only inverse directions.
            let real: Vec<_> = source.iter().map(|z| z.re).collect();
            let real_exact: Vec<_> = real.iter().map(|&v| Complex::new(wide(v), 0.0)).collect();
            let mut packed_shape = shape.clone();
            let last = *axes.last().unwrap();
            packed_shape[last] = shape[last] / 2 + 1;
            let packed_len: usize = packed_shape.iter().product();
            let (real_input, _) = padded(&real, S::NAN, &shape);
            let factor = work as f64;
            for hermitian in [false, true] {
                let layout_kind = if hermitian {
                    FftLayout::Hermitian
                } else {
                    FftLayout::Real
                };
                // rfftn is the unscaled forward sum; ihfftn the inverse sum / N.
                let spectrum = pack(
                    &direct_nd(&real_exact, &shape, &axes, hermitian),
                    &shape,
                    &packed_shape,
                );
                let expected: Vec<_> = spectrum
                    .iter()
                    .map(|&z| if hermitian { z / factor } else { z })
                    .collect();
                let (bins_direction, samples_direction) = if hermitian {
                    (FftDirection::Inverse, FftDirection::Forward)
                } else {
                    (FftDirection::Forward, FftDirection::Inverse)
                };
                let forward = FftNdPlan::<S>::new(
                    FftNdPlanSpec::new(
                        &shape,
                        &axes,
                        bins_direction,
                        Normalization::Inverse,
                        layout_kind,
                    )
                    .unwrap(),
                )
                .unwrap();
                let mut bins = vec![complex_gap; 2 * packed_len];
                let packed_strides: Vec<_> = layout(&packed_shape).iter().map(|&s| 2 * s).collect();
                forward
                    .execute_into(
                        FftNdInput::Real(view(&real_input, &shape, &strides)),
                        FftNdOutput::Complex(
                            NdViewMut::new(
                                &mut bins,
                                StridedLayout::new(&packed_shape, &packed_strides).unwrap(),
                            )
                            .unwrap(),
                        ),
                        &mut forward.workspace(()).unwrap(),
                        cx,
                    )
                    .unwrap();
                assert_fft_close(&unpadded(&bins, complex_gap), &expected, work);

                // irfftn of the exact forward sums gives the signal; hfftn of
                // the exact unscaled inverse sums gives N times the signal.
                let (spectrum_input, _) = padded(
                    &complex_of::<S>(&spectrum),
                    Complex::new(S::NAN, S::NAN),
                    &packed_shape,
                );
                let inverse = FftNdPlan::<S>::new(
                    FftNdPlanSpec::new(
                        &shape,
                        &axes,
                        samples_direction,
                        Normalization::Inverse,
                        layout_kind,
                    )
                    .unwrap(),
                )
                .unwrap();
                let mut samples = vec![gap; 2 * len];
                inverse
                    .execute_into(
                        FftNdInput::Complex(view(&spectrum_input, &packed_shape, &packed_strides)),
                        FftNdOutput::Real(
                            NdViewMut::new(
                                &mut samples,
                                StridedLayout::new(&shape, &strides).unwrap(),
                            )
                            .unwrap(),
                        ),
                        &mut inverse.workspace(()).unwrap(),
                        cx,
                    )
                    .unwrap();
                let recovered: Vec<_> = unpadded(&samples, gap)
                    .iter()
                    .map(|&v| Complex::new(v, S::ZERO))
                    .collect();
                let expected: Vec<_> = real_exact
                    .iter()
                    .map(|&z| if hermitian { z * factor } else { z })
                    .collect();
                assert_fft_close(&recovered, &expected, work);
            }

            // DCT/DST types I-IV in both directions (type I needs two points).
            for cosine in [true, false] {
                for kind in 0..4 {
                    if cosine && kind == 0 && axes.iter().any(|&a| shape[a] < 2) {
                        continue;
                    }
                    let kinds = vec![kind; axes.len()];
                    let types: Vec<RealTransformKind> = kinds
                        .iter()
                        .map(|&k| {
                            if cosine {
                                [DctType::I, DctType::II, DctType::III, DctType::IV][k].into()
                            } else {
                                [DstType::I, DstType::II, DstType::III, DstType::IV][k].into()
                            }
                        })
                        .collect();
                    for direction in [FftDirection::Forward, FftDirection::Inverse] {
                        let expected = real_tensor_definition(
                            &real.iter().map(|&v| wide(v)).collect::<Vec<_>>(),
                            &shape,
                            &axes,
                            &kinds,
                            cosine,
                            Normalization::Orthonormal,
                            direction == FftDirection::Inverse,
                        );
                        let plan = RealTransformNdPlan::<S>::new(
                            RealTransformNdPlanSpec::new(
                                &shape,
                                &axes,
                                &types,
                                direction,
                                Normalization::Orthonormal,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                        let mut output = vec![gap; 2 * len];
                        plan.execute_into(
                            view(&real_input, &shape, &strides),
                            NdViewMut::new(
                                &mut output,
                                StridedLayout::new(&shape, &strides).unwrap(),
                            )
                            .unwrap(),
                            &mut plan.workspace(()).unwrap(),
                            cx,
                        )
                        .unwrap();
                        real_tensor_close(&unpadded(&output, gap), &expected);
                    }
                }
            }
        }
    }
}

#[test]
fn short_strided_lanes_match_direct_sums_f32() {
    short_strided_lanes::<f32>();
}

#[test]
fn short_strided_lanes_match_direct_sums_f64() {
    short_strided_lanes::<f64>();
}

// With one selected axis every lane is an independent batch and must equal
// the matching one-dimensional plan exactly, for contiguous and strided lanes.
#[test]
fn single_axis_lanes_equal_one_dimensional_plans() {
    use scioxide_linalg::{VectorView, VectorViewMut};
    let cx = scioxide_core::ExecutionContext::sequential();
    for (shape, axis) in [([7, 12], 0), ([7, 12], 1), ([6, 5], 0), ([6, 5], 1)] {
        let len = shape[0] * shape[1];
        let n = shape[axis];
        let strides = layout(&shape);
        let lanes: Vec<Vec<usize>> = (0..len / n)
            .map(|lane| {
                let start = if axis == 0 { lane } else { lane * n };
                (0..n).map(|i| start + i * strides[axis]).collect()
            })
            .collect();
        // Lane magnitudes span many binades, so each lane has its own range.
        let real: Vec<f64> = (0..len)
            .map(|i| ((i as f64 * 0.73).sin() + 0.25) * 2.0_f64.powi((i % 5) as i32 * 40 - 80))
            .collect();
        let complex: Vec<_> = real.iter().map(|&v| Complex::new(v, -0.5 * v)).collect();
        let packed_shape = {
            let mut p = shape;
            p[axis] = n / 2 + 1;
            p
        };
        let packed_strides = layout(&packed_shape);
        let packed_lanes: Vec<Vec<usize>> = (0..len / n)
            .map(|lane| {
                let start = if axis == 0 { lane } else { lane * (n / 2 + 1) };
                (0..n / 2 + 1)
                    .map(|i| start + i * packed_strides[axis])
                    .collect()
            })
            .collect();
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let norm = Normalization::Orthonormal;
            let plan = FftNdPlan::<f64>::new(
                FftNdPlanSpec::new(&shape, &[axis], direction, norm, FftLayout::Complex).unwrap(),
            )
            .unwrap();
            let mut output = complex.clone();
            plan.execute_into(
                FftNdInput::Complex(view(&complex, &shape, &strides)),
                FftNdOutput::Complex(
                    NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap())
                        .unwrap(),
                ),
                &mut plan.workspace(()).unwrap(),
                cx,
            )
            .unwrap();
            let one = FftPlan::<f64>::new(FftPlanSpec::new(n, direction, norm, FftLayout::Complex))
                .unwrap();
            for lane in &lanes {
                let input: Vec<_> = lane.iter().map(|&i| complex[i]).collect();
                let mut expected = input.clone();
                one.execute_into(
                    &input[..],
                    &mut expected[..],
                    &mut one.workspace(()).unwrap(),
                    cx,
                )
                .unwrap();
                let actual: Vec<_> = lane.iter().map(|&i| output[i]).collect();
                assert_eq!(actual, expected);
            }
        }
        for family in [FftLayout::Real, FftLayout::Hermitian] {
            let norm = Normalization::Inverse;
            let (to_bins, to_samples) = if family == FftLayout::Real {
                (FftDirection::Forward, FftDirection::Inverse)
            } else {
                (FftDirection::Inverse, FftDirection::Forward)
            };
            let forward = FftNdPlan::<f64>::new(
                FftNdPlanSpec::new(&shape, &[axis], to_bins, norm, family).unwrap(),
            )
            .unwrap();
            let mut bins = vec![Complex::new(0.0, 0.0); packed_shape.iter().product()];
            let bins_layout = StridedLayout::new(&packed_shape, &packed_strides).unwrap();
            forward
                .execute_into(
                    FftNdInput::Real(view(&real, &shape, &strides)),
                    FftNdOutput::Complex(NdViewMut::new(&mut bins, bins_layout).unwrap()),
                    &mut forward.workspace(()).unwrap(),
                    cx,
                )
                .unwrap();
            let inverse = FftNdPlan::<f64>::new(
                FftNdPlanSpec::new(&shape, &[axis], to_samples, norm, family).unwrap(),
            )
            .unwrap();
            let mut samples = vec![0.0; len];
            inverse
                .execute_into(
                    FftNdInput::Complex(view(&bins, &packed_shape, &packed_strides)),
                    FftNdOutput::Real(
                        NdViewMut::new(&mut samples, StridedLayout::new(&shape, &strides).unwrap())
                            .unwrap(),
                    ),
                    &mut inverse.workspace(()).unwrap(),
                    cx,
                )
                .unwrap();
            let one_forward =
                FftPlan::<f64>::new(FftPlanSpec::new(n, to_bins, norm, family)).unwrap();
            let one_inverse =
                FftPlan::<f64>::new(FftPlanSpec::new(n, to_samples, norm, family)).unwrap();
            for (lane, packed_lane) in lanes.iter().zip(&packed_lanes) {
                let input: Vec<_> = lane.iter().map(|&i| real[i]).collect();
                let mut expected = vec![Complex::new(0.0, 0.0); n / 2 + 1];
                one_forward
                    .execute_into(
                        FftInput::Real(VectorView::from_slice(&input)),
                        FftOutput::Hermitian {
                            logical_len: n,
                            bins: VectorViewMut::from_slice(&mut expected),
                        },
                        &mut one_forward.workspace(()).unwrap(),
                        cx,
                    )
                    .unwrap();
                let actual: Vec<_> = packed_lane.iter().map(|&i| bins[i]).collect();
                // Equal values; signed zeros of conjugated bins may differ.
                assert_eq!(actual, expected);
                let mut expected = vec![0.0; n];
                one_inverse
                    .execute_into(
                        FftInput::Hermitian {
                            logical_len: n,
                            bins: VectorView::from_slice(&actual),
                        },
                        FftOutput::Real(VectorViewMut::from_slice(&mut expected)),
                        &mut one_inverse.workspace(()).unwrap(),
                        cx,
                    )
                    .unwrap();
                let actual: Vec<_> = lane.iter().map(|&i| samples[i]).collect();
                assert_eq!(actual, expected);
            }
        }
        for kind in [
            RealTransformKind::Dct(DctType::I),
            DctType::II.into(),
            DctType::IV.into(),
            DstType::I.into(),
            DstType::III.into(),
        ] {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let norm = Normalization::Orthonormal;
                let plan = RealTransformNdPlan::<f64>::new(
                    RealTransformNdPlanSpec::new(&shape, &[axis], &[kind], direction, norm)
                        .unwrap(),
                )
                .unwrap();
                let mut output = real.clone();
                plan.execute_into(
                    view(&real, &shape, &strides),
                    NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap())
                        .unwrap(),
                    &mut plan.workspace(()).unwrap(),
                    cx,
                )
                .unwrap();
                let one = RealTransformPlan::<f64>::new(RealTransformPlanSpec::new(
                    n, kind, direction, norm,
                ))
                .unwrap();
                for lane in &lanes {
                    let input: Vec<_> = lane.iter().map(|&i| real[i]).collect();
                    let mut expected = vec![0.0; n];
                    one.execute_into(&input, &mut expected, &mut one.workspace(()).unwrap(), cx)
                        .unwrap();
                    let actual: Vec<_> = lane.iter().map(|&i| output[i]).collect();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

// Nineteen two-point axes exceed f32's prepared headroom, so every lane is
// range-scaled by its child; f64 keeps bounded batch scaling. Both agree.
#[test]
fn trig_lanes_without_prepared_headroom_match_bounded_double_precision() {
    let shape = [2; 19];
    let axes: Vec<_> = (0..19).collect();
    let len = 1 << 19;
    let source: Vec<f64> = (0..len)
        .map(|i| ((i * 7 % 23) as f64 - 11.0) * if i % 3 == 0 { 1e-6 } else { 1.0 })
        .collect();
    let single: Vec<f32> = source.iter().map(|&v| v as f32).collect();
    let exact: Vec<f64> = single.iter().map(|&v| f64::from(v)).collect();
    for types in [[DctType::II; 19], [DctType::IV; 19]] {
        let expected = dctn(&exact, &shape, &axes, &types, Normalization::Orthonormal).unwrap();
        let actual = dctn(&single, &shape, &axes, &types, Normalization::Orthonormal).unwrap();
        real_tensor_close(&actual, &expected);
    }
}

// Single-axis even II/III/IV lanes run in blocks of four: each lane keeps its
// own child call and the one-dimensional arithmetic. Zero lanes, subnormal
// lanes (extended binary scales) and lanes whose restoration cannot be proven
// finite sit inside blocks and partial blocks, along both axes.
#[test]
fn checked_trig_blocks_equal_one_dimensional_plans_bitwise() {
    let cx = scioxide_core::ExecutionContext::sequential();
    for (shape, axis) in [([20, 64], 1), ([64, 9], 0), ([6, 66], 1), ([10, 8], 1)] {
        let len = shape[0] * shape[1];
        let n = shape[axis];
        let strides = layout(&shape);
        let lane_of = |i: usize| if axis == 0 { i % shape[1] } else { i / n };
        let index_of = |i: usize| if axis == 0 { i / shape[1] } else { i % n };
        let values: Vec<f64> = (0..len)
            .map(|i| {
                let base = ((i * 37 % 101) as f64 - 50.0) / 64.0;
                match lane_of(i) % 11 {
                    5 => 0.0,
                    6 => base * 2.0_f64.powi(-1060),
                    9 if index_of(i) == 0 => 2.0_f64.powi(1019),
                    9 => 0.0,
                    _ => base * 2.0_f64.powi((lane_of(i) % 3) as i32 * 30 - 30),
                }
            })
            .collect();
        let lanes = len / n;
        let starts: Vec<usize> = (0..lanes)
            .map(|lane| if axis == 0 { lane } else { lane * n })
            .collect();
        for kind in [
            RealTransformKind::Dct(DctType::II),
            DctType::III.into(),
            DctType::IV.into(),
            DstType::II.into(),
            DstType::III.into(),
            DstType::IV.into(),
        ] {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                for norm in [
                    Normalization::Unscaled,
                    Normalization::Forward,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                ] {
                    let plan = RealTransformNdPlan::<f64>::new(
                        RealTransformNdPlanSpec::new(&shape, &[axis], &[kind], direction, norm)
                            .unwrap(),
                    )
                    .unwrap();
                    let mut output = vec![f64::NAN; len];
                    let result = plan.execute_into(
                        view(&values, &shape, &strides),
                        NdViewMut::new(&mut output, StridedLayout::new(&shape, &strides).unwrap())
                            .unwrap(),
                        &mut plan.workspace(()).unwrap(),
                        cx,
                    );
                    let one = RealTransformPlan::<f64>::new(RealTransformPlanSpec::new(
                        n, kind, direction, norm,
                    ))
                    .unwrap();
                    let mut work = one.workspace(()).unwrap();
                    let expected: Result<Vec<Vec<f64>>, _> = starts
                        .iter()
                        .map(|&start| {
                            let input: Vec<_> =
                                (0..n).map(|i| values[start + i * strides[axis]]).collect();
                            let mut lane = vec![0.0; n];
                            one.execute_into(&input, &mut lane, &mut work, cx)
                                .map(|()| lane)
                        })
                        .collect();
                    match (result, expected) {
                        (Ok(()), Ok(expected)) => {
                            for (&start, expected) in starts.iter().zip(expected) {
                                for (i, expected) in expected.into_iter().enumerate() {
                                    let actual = output[start + i * strides[axis]];
                                    assert_eq!(
                                        actual.to_bits(),
                                        expected.to_bits(),
                                        "{shape:?} axis {axis} {kind:?} {direction:?} {norm:?}"
                                    );
                                }
                            }
                        }
                        (Err(actual), Err(expected)) => {
                            assert_eq!(actual.kind(), expected.kind());
                            assert!(output.iter().all(|v| v.is_nan()));
                        }
                        (actual, expected) => assert_eq!(
                            actual.err().map(|error| error.kind()),
                            expected.err().map(|error| error.kind()),
                            "{shape:?} axis {axis} {kind:?}"
                        ),
                    }
                }
            }
        }
    }
}

#[test]
fn dense_one_shots_check_the_element_count() {
    use scioxide_core::{ExecutionContext, SciOxideErrorKind};
    let (shape, axes, len) = ([3, 4], [1, 0], 12);
    let real: Vec<f64> = (0..len).map(|i| ((i * 5 % 7) as f64 - 3.0) / 4.0).collect();
    let complex: Vec<_> = real.iter().map(|&x| Complex::new(x, 0.5 - x)).collect();
    let norm = Normalization::Orthonormal;
    let (dct_types, dst_types) = ([DctType::II, DctType::IV], [DstType::I, DstType::III]);
    for n in [len - 1, len + 1] {
        let (r, c) = (vec![0.5; n], vec![Complex::new(0.5, 0.0); n]);
        for error in [
            fftn(&c, &shape, &axes, norm).unwrap_err(),
            ifftn(&c, &shape, &axes, norm).unwrap_err(),
            rfftn(&r, &shape, &axes, norm).unwrap_err(),
            ihfftn(&r, &shape, &axes, norm).unwrap_err(),
            dctn(&r, &shape, &axes, &dct_types, norm).unwrap_err(),
            idctn(&r, &shape, &axes, &dct_types, norm).unwrap_err(),
            dstn(&r, &shape, &axes, &dst_types, norm).unwrap_err(),
            idstn(&r, &shape, &axes, &dst_types, norm).unwrap_err(),
            fftshift_nd(&r, &shape, &axes).unwrap_err(),
            ifftshift_nd(&r, &shape, &axes).unwrap_err(),
        ] {
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch, "{n}");
        }
    }
    let (no_real, no_complex): (&[f64], &[Complex<f64>]) = (&[], &[]);
    assert!(fftn(no_complex, &[0, 3], &[1], norm).unwrap().is_empty());
    assert!(
        dctn(no_real, &[0, 3], &[1], &[DctType::II], norm)
            .unwrap()
            .is_empty()
    );
    assert!(fftshift_nd(no_real, &[0, 3], &[1]).unwrap().is_empty());

    let cx = ExecutionContext::sequential_native();
    let layout = StridedLayout::row_major(&shape).unwrap();
    let complex_bits = |z: &[Complex<f64>]| -> Vec<_> {
        z.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
    };
    let real_bits = |x: &[f64]| -> Vec<_> { x.iter().map(|x| x.to_bits()).collect() };
    for (direction, fft_layout) in [
        (FftDirection::Forward, FftLayout::Complex),
        (FftDirection::Inverse, FftLayout::Complex),
        (FftDirection::Forward, FftLayout::Real),
        (FftDirection::Inverse, FftLayout::Hermitian),
    ] {
        let spec = FftNdPlanSpec::new(&shape, &axes, direction, norm, fft_layout).unwrap();
        let plan = FftNdPlan::<f64>::new(spec).unwrap();
        let packed_shape = plan.packed_shape();
        let mut expected = vec![Complex::new(0.0, 0.0); packed_shape.iter().product()];
        let target = NdViewMut::row_major(&mut expected, packed_shape).unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let actual = match (direction, fft_layout) {
            (_, FftLayout::Complex) => {
                let source = NdView::new(&complex, layout).unwrap();
                plan.execute_into(source, target, &mut workspace, cx)
                    .unwrap();
                if direction == FftDirection::Forward {
                    fftn(&complex, &shape, &axes, norm).unwrap()
                } else {
                    ifftn(&complex, &shape, &axes, norm).unwrap()
                }
            }
            _ => {
                let source = NdView::new(&real, layout).unwrap();
                plan.execute_into(source, target, &mut workspace, cx)
                    .unwrap();
                if fft_layout == FftLayout::Real {
                    rfftn(&real, &shape, &axes, norm).unwrap().into_vec()
                } else {
                    ihfftn(&real, &shape, &axes, norm).unwrap().into_vec()
                }
            }
        };
        assert_eq!(complex_bits(&actual), complex_bits(&expected));
    }
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let inverse = direction == FftDirection::Inverse;
        for cosine in [true, false] {
            let kinds: Vec<RealTransformKind> = if cosine {
                dct_types.iter().map(|&t| t.into()).collect()
            } else {
                dst_types.iter().map(|&t| t.into()).collect()
            };
            let spec = RealTransformNdPlanSpec::new(&shape, &axes, &kinds, direction, norm);
            let plan = RealTransformNdPlan::<f64>::new(spec.unwrap()).unwrap();
            let mut expected = vec![0.0; len];
            plan.execute_into(
                NdView::new(&real, layout).unwrap(),
                NdViewMut::new(&mut expected, layout).unwrap(),
                &mut plan.workspace(()).unwrap(),
                cx,
            )
            .unwrap();
            let actual = match (cosine, inverse) {
                (true, false) => dctn(&real, &shape, &axes, &dct_types, norm),
                (true, true) => idctn(&real, &shape, &axes, &dct_types, norm),
                (false, false) => dstn(&real, &shape, &axes, &dst_types, norm),
                (false, true) => idstn(&real, &shape, &axes, &dst_types, norm),
            }
            .unwrap();
            assert_eq!(real_bits(&actual), real_bits(&expected));
        }
    }
    // The shifts have no plan: along the last axis each row is a 1-D shift.
    let rows = |shift: fn(&[f64]) -> Result<Vec<f64>, _>| -> Vec<f64> {
        real.chunks_exact(4)
            .flat_map(|row| shift(row).unwrap())
            .collect()
    };
    assert_eq!(fftshift_nd(&real, &shape, &[1]).unwrap(), rows(fftshift));
    assert_eq!(ifftshift_nd(&real, &shape, &[1]).unwrap(), rows(ifftshift));
}
