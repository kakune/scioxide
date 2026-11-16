//! Batch partitioning must preserve the arithmetic of each independent tensor.
#![expect(clippy::unwrap_used)]
#[path = "support/views.rs"]
mod views;
use views::view;
#[path = "support/numerics.rs"]
mod numerics;
use numerics::{scalar, wide};
use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, NumericScalar, RealScalar};
use scioxide_fft::*;

fn direction(inverse: bool) -> FftDirection {
    if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    }
}
fn fourier<S: RealScalar>(
    shape: &[usize],
    axes: &[usize],
    inverse: bool,
    norm: Normalization,
    layout: FftLayout,
) -> FftNdPlan<S> {
    let spec = FftNdPlanSpec::new(shape, axes, direction(inverse), norm, layout).unwrap();
    FftNdPlan::new(spec).unwrap()
}
/// `fftn` or `ifftn` of a strided array through its plan, in row-major order.
fn complex_plan<S: RealScalar>(
    input: NdView<'_, '_, Complex<S>>,
    axes: &[usize],
    inverse: bool,
    norm: Normalization,
) -> Vec<Complex<S>> {
    let shape = input.layout().shape().extents();
    let plan = fourier::<S>(shape, axes, inverse, norm, FftLayout::Complex);
    let mut output = vec![Complex::new(S::ZERO, S::ZERO); input.layout().shape().len()];
    let target = NdViewMut::row_major(&mut output, shape).unwrap();
    let cx = ExecutionContext::sequential_native();
    plan.execute_into(input, target, &mut plan.workspace(()).unwrap(), cx)
        .unwrap();
    output
}
/// The packed bins of `rfftn` or `ihfftn` of a strided array through its plan.
fn packed_plan<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axes: &[usize],
    hermitian: bool,
    norm: Normalization,
) -> HermitianSpectrumNd<S> {
    let shape = input.layout().shape().extents();
    let layout = if hermitian {
        FftLayout::Hermitian
    } else {
        FftLayout::Real
    };
    let plan = fourier::<S>(shape, axes, hermitian, norm, layout);
    let packed_shape = plan.packed_shape();
    let len = packed_shape.iter().product();
    let mut bins = vec![Complex::new(S::ZERO, S::ZERO); len];
    let target = NdViewMut::row_major(&mut bins, packed_shape).unwrap();
    let cx = ExecutionContext::sequential_native();
    plan.execute_into(input, target, &mut plan.workspace(()).unwrap(), cx)
        .unwrap();
    HermitianSpectrumNd::from_vec(shape, axes, bins).unwrap()
}
/// A DCT or DST of a strided array through its plan, in row-major order.
fn trig_plan<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axes: &[usize],
    kind: RealTransformKind,
    inverse: bool,
    norm: Normalization,
) -> Vec<S> {
    let shape = input.layout().shape().extents();
    let kinds = vec![kind; axes.len()];
    let spec = RealTransformNdPlanSpec::new(shape, axes, &kinds, direction(inverse), norm);
    let plan = RealTransformNdPlan::<S>::new(spec.unwrap()).unwrap();
    let mut output = vec![S::ZERO; input.layout().shape().len()];
    let target = NdViewMut::row_major(&mut output, shape).unwrap();
    let cx = ExecutionContext::sequential_native();
    plan.execute_into(input, target, &mut plan.workspace(()).unwrap(), cx)
        .unwrap();
    output
}

fn real_bits<S: RealScalar>(actual: &[S], expected: &[S]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_eq!(wide(actual).to_bits(), wide(expected).to_bits());
    }
}
fn complex_bits<S: RealScalar>(actual: &[Complex<S>], expected: &[Complex<S>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        real_bits(&[actual.re, actual.im], &[expected.re, expected.im]);
    }
}

/// Two batches of shape `[rows, columns]` along axis 0 of `[2, rows, columns]`,
/// dense and interleaved, through the plans against the one-call functions of
/// each batch. Odd lengths take the per-lane kernels; even ones the
/// lane blocks, where a batch run alone and the whole tensor place the same
/// lane in different blocks.
fn partitions<S: RealScalar + NumericScalar<Real = S>>(exponent: i32, [rows, columns]: [usize; 2])
where
    Complex<S>: NumericScalar<Real = S>,
{
    let size = rows * columns;
    let shape = [2, rows, columns];
    let local = [rows, columns];
    for exponents in [[exponent, -exponent], [0, 0]] {
        let batches: Vec<Vec<S>> = exponents
            .iter()
            .map(|&e| {
                (0..size)
                    .map(|i| {
                        let value = if exponents == [0, 0] {
                            -S::ZERO
                        } else {
                            scalar::<S>(((i * 7 % 15) as f64 - 7.0) / 8.0)
                        };
                        value.scalbn(e)
                    })
                    .collect()
            })
            .collect();
        let complex_batches: Vec<Vec<_>> = batches
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|&x| Complex::new(x, -x / S::TWO))
                    .collect()
            })
            .collect();
        for strided in [false, true] {
            let strides = if strided {
                [1, 2 * columns, 2]
            } else {
                [size, columns, 1]
            };
            let mut real = vec![S::ZERO; 2 * size];
            let mut complex = vec![Complex::new(S::ZERO, S::ZERO); 2 * size];
            for batch in 0..2 {
                for i in 0..size {
                    let offset = batch * strides[0]
                        + (i / columns) * strides[1]
                        + (i % columns) * strides[2];
                    real[offset] = batches[batch][i];
                    complex[offset] = complex_batches[batch][i];
                }
            }
            for axes in [&[1, 2][..], &[2, 1], &[2], &[]] {
                let local_axes: Vec<_> = axes.iter().map(|axis| axis - 1).collect();
                for norm in [
                    Normalization::Unscaled,
                    Normalization::Forward,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                ] {
                    for inverse in [false, true] {
                        let transform = if inverse { ifftn } else { fftn };
                        let actual =
                            complex_plan(view(&complex, &shape, &strides), axes, inverse, norm);
                        let expected: Vec<_> = complex_batches
                            .iter()
                            .flat_map(|batch| transform(batch, &local, &local_axes, norm).unwrap())
                            .collect();
                        complex_bits(&actual, &expected);
                    }
                    if !axes.is_empty() {
                        for hermitian in [false, true] {
                            let forward = if hermitian { ihfftn } else { rfftn };
                            let inverse = if hermitian { hfftn } else { irfftn };
                            let actual =
                                packed_plan(view(&real, &shape, &strides), axes, hermitian, norm);
                            let expected: Vec<_> = batches
                                .iter()
                                .flat_map(|batch| {
                                    forward(batch, &local, &local_axes, norm)
                                        .unwrap()
                                        .into_vec()
                                })
                                .collect();
                            complex_bits(actual.bins(), &expected);
                            let recovered = inverse(&actual, norm).unwrap();
                            let expected: Vec<_> = batches
                                .iter()
                                .flat_map(|batch| {
                                    let packed = forward(batch, &local, &local_axes, norm).unwrap();
                                    inverse(&packed, norm).unwrap()
                                })
                                .collect();
                            real_bits(&recovered, &expected);
                        }
                    }
                    for ty in [DctType::I, DctType::II, DctType::III, DctType::IV] {
                        for inverse in [false, true] {
                            let transform = if inverse { idctn } else { dctn };
                            let source = view(&real, &shape, &strides);
                            let actual = trig_plan(source, axes, ty.into(), inverse, norm);
                            let types = vec![ty; axes.len()];
                            let expected: Vec<_> = batches
                                .iter()
                                .flat_map(|batch| {
                                    transform(batch, &local, &local_axes, &types, norm).unwrap()
                                })
                                .collect();
                            real_bits(&actual, &expected);
                        }
                    }
                    for ty in [DstType::I, DstType::II, DstType::III, DstType::IV] {
                        for inverse in [false, true] {
                            let transform = if inverse { idstn } else { dstn };
                            let source = view(&real, &shape, &strides);
                            let actual = trig_plan(source, axes, ty.into(), inverse, norm);
                            let types = vec![ty; axes.len()];
                            let expected: Vec<_> = batches
                                .iter()
                                .flat_map(|batch| {
                                    transform(batch, &local, &local_axes, &types, norm).unwrap()
                                })
                                .collect();
                            real_bits(&actual, &expected);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn independent_tensor_partitions_preserve_bits_f32() {
    for shape in [[3, 5], [3, 8], [4, 6], [6, 16]] {
        partitions::<f32>(100, shape);
    }
}
#[test]
fn independent_tensor_partitions_preserve_bits_f64() {
    for shape in [[3, 5], [3, 8], [4, 6], [6, 16]] {
        partitions::<f64>(900, shape);
    }
}
