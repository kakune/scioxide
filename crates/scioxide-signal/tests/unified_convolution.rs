//! Independent literal sums exercise the single engine in all scalar domains.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, NumericScalar, RealScalar, SciOxideErrorKind,
    StridedLayout,
};
use scioxide_signal::{
    ConvolutionExtent, ConvolutionKernel, ConvolutionMethod, ConvolutionNdPlan, ConvolutionOptions,
    SignalBoundary, SignalProduct,
};
fn scalar<T: NumericScalar>(re: f64, im: f64) -> T {
    T::from_components(
        T::Real::try_from_f64_lossy(re).unwrap(),
        T::Real::try_from_f64_lossy(im).unwrap(),
    )
}
fn exercise<T: NumericScalar>() {
    for (shape, kernel_shape, axes, strides, ks) in [
        (vec![4], vec![3], vec![0], vec![1], vec![1]),
        (vec![3, 4], vec![2, 3], vec![0, 1], vec![4, 1], vec![3, 1]),
    ] {
        let a: Vec<T> = (0..shape.iter().product())
            .map(|i| scalar((i % 7) as f64 / 4. - 0.5, (i % 3) as f64 / 8.))
            .collect();
        let b: Vec<T> = (0..kernel_shape.iter().product())
            .map(|i| scalar((i % 5) as f64 / 4. - 0.25, -((i % 3) as f64) / 8.))
            .collect();
        for extent in [
            ConvolutionExtent::Full,
            ConvolutionExtent::Same,
            ConvolutionExtent::Valid,
            ConvolutionExtent::Circular,
        ] {
            for product in [SignalProduct::Convolution, SignalProduct::Correlation] {
                let options = ConvolutionOptions::new()
                    .with_extent(extent)
                    .with_product(product)
                    .with_boundary(if extent == ConvolutionExtent::Circular {
                        SignalBoundary::Periodic
                    } else {
                        SignalBoundary::Zero
                    });
                for method in [
                    ConvolutionMethod::Direct,
                    ConvolutionMethod::Fft,
                    ConvolutionMethod::Automatic,
                ] {
                    let plan = ConvolutionNdPlan::new(
                        &shape,
                        NdView::new(&b, StridedLayout::new(&kernel_shape, &ks).unwrap()).unwrap(),
                        &axes,
                        options.with_method(method),
                    )
                    .unwrap();
                    let output_shape = plan.output_shape();
                    let out_stride = if shape.len() == 1 {
                        vec![1]
                    } else {
                        vec![output_shape[1], 1]
                    };
                    let mut expected = vec![T::ZERO; output_shape.iter().product()];
                    for (i, out) in expected.iter_mut().enumerate() {
                        let mut sum = T::ZERO;
                        for (j, &tap) in b.iter().enumerate() {
                            let mut source = 0;
                            let mut valid = true;
                            for axis in 0..shape.len() {
                                let coordinate = (i / out_stride[axis]) % output_shape[axis];
                                let k = (j / ks[axis]) % kernel_shape[axis];
                                let offset = if product == SignalProduct::Correlation {
                                    kernel_shape[axis] - 1 - k
                                } else {
                                    k
                                };
                                let crop = match extent {
                                    ConvolutionExtent::Full => 0,
                                    ConvolutionExtent::Same => (kernel_shape[axis] - 1) / 2,
                                    ConvolutionExtent::Valid => kernel_shape[axis] - 1,
                                    ConvolutionExtent::Circular => {
                                        if product == SignalProduct::Correlation {
                                            kernel_shape[axis] - 1
                                        } else {
                                            0
                                        }
                                    }
                                };
                                let n = coordinate as i128 + crop as i128 - offset as i128;
                                let n = if extent == ConvolutionExtent::Circular {
                                    n.rem_euclid(shape[axis] as i128)
                                } else {
                                    n
                                };
                                if n < 0 || n >= shape[axis] as i128 {
                                    valid = false;
                                    break;
                                }
                                source += n as usize * strides[axis];
                            }
                            if valid {
                                sum = sum
                                    + a[source]
                                        * if product == SignalProduct::Correlation {
                                            tap.conj()
                                        } else {
                                            tap
                                        };
                            }
                        }
                        *out = sum;
                    }
                    let actual = plan
                        .convolve(
                            NdView::new(&a, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
                        )
                        .unwrap();
                    let tolerance = T::Real::try_from_usize_exact(2048).unwrap() * T::Real::EPSILON;
                    for (&x, &y) in actual.iter().zip(&expected) {
                        assert!(
                            (x - y).abs() <= tolerance,
                            "{method:?} {extent:?} {product:?}: {x:?} {y:?}"
                        );
                    }
                    let mut raw = vec![T::ZERO; expected.len()];
                    let mut work = plan.workspace(ConvolutionKernel::Supplied).unwrap();
                    plan.convolve_pair_into(
                        NdView::new(&a, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
                        NdView::new(&b, StridedLayout::new(&kernel_shape, &ks).unwrap()).unwrap(),
                        NdViewMut::new(
                            &mut raw,
                            StridedLayout::new(output_shape, &out_stride).unwrap(),
                        )
                        .unwrap(),
                        &mut work,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                    assert_eq!(raw, actual);
                    raw.fill(scalar(19., 2.));
                    let mut short = plan.workspace(ConvolutionKernel::Prepared).unwrap();
                    assert!(
                        plan.convolve_pair_into(
                            NdView::new(&a, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
                            NdView::new(&b, StridedLayout::new(&kernel_shape, &ks).unwrap())
                                .unwrap(),
                            NdViewMut::new(
                                &mut raw,
                                StridedLayout::new(output_shape, &out_stride).unwrap()
                            )
                            .unwrap(),
                            &mut short,
                            ExecutionContext::sequential_native()
                        )
                        .unwrap_err()
                        .kind()
                            == scioxide_core::SciOxideErrorKind::DimensionMismatch
                    );
                    assert!(raw.iter().all(|&x| x == scalar(19., 2.)));
                }
            }
        }
    }
}
#[test]
fn real32() {
    exercise::<f32>();
}
#[test]
fn real64() {
    exercise::<f64>();
}
#[test]
fn complex32() {
    exercise::<Complex<f32>>();
}
#[test]
fn complex64() {
    exercise::<Complex<f64>>();
}
#[test]
fn direct_circular_cancellation_keeps_representable_results() {
    let result = scioxide_signal::convolve_circular(&[f64::MAX, -f64::MAX], &[2., 2.]).unwrap();
    assert_eq!(result, [0., 0.]);
}

#[test]
fn supplied_kernel_validation_and_numerical_failures_preserve_output() {
    let kernel = [0.5_f64; 32];
    let mut input = [1.; 64];
    let plan = ConvolutionNdPlan::new(
        &[64],
        NdView::new(&kernel, StridedLayout::new(&[32], &[1]).unwrap()).unwrap(),
        &[0],
        ConvolutionOptions::new().with_method(ConvolutionMethod::Fft),
    )
    .unwrap();
    let mut work = plan.workspace(ConvolutionKernel::Supplied).unwrap();
    let mut output = vec![43.; 95];
    let output_layout = StridedLayout::new(&[95], &[1]).unwrap();
    let input_layout = StridedLayout::new(&[64], &[1]).unwrap();
    let error = plan
        .convolve_pair_into(
            NdView::new(&input, input_layout).unwrap(),
            NdView::new(&kernel[..31], StridedLayout::new(&[31], &[1]).unwrap()).unwrap(),
            NdViewMut::new(&mut output, output_layout).unwrap(),
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch {
            expected: [32],
            actual: [31]
        }
    ));
    assert!(output.iter().all(|&v| v == 43.));
    input[63] = f64::NAN;
    assert_eq!(
        error_kind(plan.convolve_pair_into(
            NdView::new(&input, input_layout).unwrap(),
            NdView::new(&kernel, StridedLayout::new(&[32], &[1]).unwrap()).unwrap(),
            NdViewMut::new(&mut output, output_layout).unwrap(),
            &mut work,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert!(output.iter().all(|&v| v == 43.));
}
