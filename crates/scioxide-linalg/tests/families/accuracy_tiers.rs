//! Accuracy tiers, binary range restoration, and deterministic execution.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::Accuracy;
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NumericScalar, NumericalIssue, RealScalar,
    SciOxideErrorView,
};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::{
    LinalgExecutionContext, LinalgWorkspace, Matrix, MatrixSide, MatrixView, MatrixViewMut,
    StructuredOperation, VectorView, VectorViewMut,
};

fn product<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>, accuracy: Accuracy) -> Matrix<S> {
    let mut result = Matrix::zeros(a.rows(), b.cols()).unwrap();
    a.view()
        .mul_mat_into(
            b,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential)
                .with_accuracy(accuracy),
            result.view_mut(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
    result
}

fn family<S: NumericScalar>() {
    let scalar = |i: usize| {
        S::from_components(
            S::Real::try_from_f64_lossy(((i * 37 + 11) % 101) as f64 / 31.0 - 1.5).unwrap(),
            S::Real::try_from_f64_lossy(((i * 13 + 3) % 43) as f64 / 37.0 - 0.5).unwrap(),
        )
    };
    let (m, k, n) = (35, 39, 37);
    let a = Matrix::from_vec(m, k, (0..m * k).map(scalar).collect()).unwrap();
    let b = Matrix::from_vec(k, n, (0..k * n).map(|i| scalar(i + 113)).collect()).unwrap();
    let standard = product(&a, &b, Accuracy::Standard);
    let exact = product(&a, &b, Accuracy::Exact);
    let eps = S::Real::EPSILON.try_to_f64_exact().unwrap();
    // Each complex component contains two real products per inner index.
    let operations = if S::IS_REAL { k } else { 2 * k } as f64;
    let gamma = operations * eps / (1.0 - operations * eps);
    for row in 0..m {
        for col in 0..n {
            let absolute = (0..k)
                .map(|inner| {
                    let x = a.view().get(row, inner).unwrap();
                    let y = b.view().get(inner, col).unwrap();
                    (x.real_component().abs() + x.imaginary_component().abs())
                        .try_to_f64_exact()
                        .unwrap()
                        * (y.real_component().abs() + y.imaginary_component().abs())
                            .try_to_f64_exact()
                            .unwrap()
                })
                .sum::<f64>();
            let error =
                standard.view().get(row, col).unwrap() - exact.view().get(row, col).unwrap();
            for component in [error.real_component(), error.imaginary_component()] {
                assert!(component.abs().try_to_f64_exact().unwrap() <= gamma * absolute);
            }
        }
    }
    let x: Vec<_> = (0..k).map(|i| scalar(i + 7)).collect();
    let mut ordinary = vec![S::ZERO; m];
    let mut reference = ordinary.clone();
    let mut scratch = ordinary.clone();
    for (accuracy, out) in [
        (Accuracy::Standard, &mut ordinary),
        (Accuracy::Exact, &mut reference),
    ] {
        a.view()
            .mul_vec_into(
                VectorView::from_slice(&x),
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                VectorViewMut::from_slice(out),
                &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
                LinalgExecutionContext::sequential(),
            )
            .unwrap();
    }
    for (row, (&actual, &expected)) in ordinary.iter().zip(&reference).enumerate() {
        let bound = (0..k)
            .map(|i| {
                a.view()
                    .get(row, i)
                    .unwrap()
                    .abs()
                    .try_to_f64_exact()
                    .unwrap()
                    * x[i].abs().try_to_f64_exact().unwrap()
            })
            .sum::<f64>();
        assert!((actual - expected).abs().try_to_f64_exact().unwrap() <= 2.0 * gamma * bound);
    }
    let dot = |accuracy| {
        VectorView::from_slice(&x)
            .dotc_left_scoped(
                VectorView::from_slice(&x),
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                LinalgExecutionContext::sequential(),
            )
            .unwrap()
    };
    assert!(
        (dot(Accuracy::Standard) - dot(Accuracy::Exact))
            .abs()
            .try_to_f64_exact()
            .unwrap()
            <= gamma * dot(Accuracy::Exact).abs().try_to_f64_exact().unwrap()
    );
    let norm = |accuracy| {
        VectorView::from_slice(&x)
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                LinalgExecutionContext::sequential(),
            )
            .unwrap()
    };
    assert!(
        (norm(Accuracy::Standard) - norm(Accuracy::Exact))
            .abs()
            .try_to_f64_exact()
            .unwrap()
            <= gamma * norm(Accuracy::Exact).try_to_f64_exact().unwrap()
    );
}

#[test]
fn component_bounds_cover_every_scalar_family_and_tile_tails() {
    family::<f32>();
    family::<f64>();
    family::<Complex<f32>>();
    family::<Complex<f64>>();
}

#[test]
fn wide_ranges_subnormals_and_cancellation_restore_finite_results() {
    for (a, b) in [
        (1.0e300, 1.0e-300),
        (1.0e-300, 1.0e300),
        (f64::from_bits(1), 1.0),
        (f64::MAX, 0.25),
    ] {
        let left = Matrix::from_vec(1, 2, vec![a, -a]).unwrap();
        let right = Matrix::from_vec(2, 1, vec![b, -b]).unwrap();
        let standard = product(&left, &right, Accuracy::Standard)
            .view()
            .get(0, 0)
            .unwrap();
        let exact = product(&left, &right, Accuracy::Exact)
            .view()
            .get(0, 0)
            .unwrap();
        assert!(
            (standard - exact).abs() <= 4.0 * f64::EPSILON * exact.abs() + 2.0 * f64::from_bits(1)
        );
        let x = [a, -a];
        let y = [b, -b];
        let standard = VectorView::from_slice(&x)
            .dotu_scoped(
                VectorView::from_slice(&y),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(
            (standard - exact).abs() <= 4.0 * f64::EPSILON * exact.abs() + 2.0 * f64::from_bits(1)
        );
    }
    for values in [
        [1.0e300, 1.0e-300],
        [f64::from_bits(1), f64::from_bits(1)],
        [f64::MAX / 4.0, -f64::MAX / 4.0],
    ] {
        let view = VectorView::from_slice(&values);
        let expected = view
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                LinalgExecutionContext::sequential(),
            )
            .unwrap();
        assert!(
            (view
                .norm2_scoped(
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap()
                - expected)
                .abs()
                <= 4.0 * f64::EPSILON * expected + f64::from_bits(1)
        );
    }
    let values = [f64::MAX, f64::MAX];
    let signs = [2.0, -2.0];
    assert_eq!(
        VectorView::from_slice(&values)
            .dotu_scoped(
                VectorView::from_slice(&signs),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
        0.0
    );
}

#[test]
fn structural_products_and_triangular_solves_cover_range_extremes() {
    for scale in [1.0, 1.0e300, 1.0e-300] {
        let data = [2.0 * scale, f64::NAN, -0.5 * scale, 3.0 * scale];
        let triangular = MatrixView::from_slice(2, 2, &data).unwrap();
        let initial = [scale, -scale, 2.0 * scale, scale];
        let mut standard = initial;
        let mut exact = initial;
        for (accuracy, output) in [
            (Accuracy::Standard, &mut standard),
            (Accuracy::Exact, &mut exact),
        ] {
            triangular
                .triangular_solve_mat_in_place(
                    MatrixSide::Left,
                    Triangle::Lower,
                    DiagonalKind::Explicit,
                    StructuredOperation::Normal,
                    1.0,
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                    MatrixViewMut::from_slice(2, 2, output).unwrap(),
                    &mut LinalgWorkspace::new(),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
        }
        for (&x, &y) in standard.iter().zip(&exact) {
            assert!((x - y).abs() <= 8.0 * f64::EPSILON * y.abs());
        }
        // Multiplication by the reciprocal input scale has a moderate result.
        let initial = [1.0 / scale, -1.0 / scale, 2.0 / scale, 1.0 / scale];
        let mut standard = initial;
        let mut exact = initial;
        for (accuracy, output) in [
            (Accuracy::Standard, &mut standard),
            (Accuracy::Exact, &mut exact),
        ] {
            triangular
                .triangular_mul_mat_in_place(
                    MatrixSide::Left,
                    Triangle::Lower,
                    DiagonalKind::Explicit,
                    StructuredOperation::Normal,
                    1.0,
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                    MatrixViewMut::from_slice(2, 2, output).unwrap(),
                    &mut LinalgWorkspace::new(),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
        }
        for (&x, &y) in standard.iter().zip(&exact) {
            assert!((x - y).abs() <= 8.0 * f64::EPSILON * y.abs());
        }
    }
    let input = [1.0, 2.0, -3.0, 4.0, 5.0, -6.0];
    let a = MatrixView::from_slice(3, 2, &input).unwrap();
    let mut ordinary = [99.0; 9];
    let mut exact = ordinary;
    for (accuracy, output) in [
        (Accuracy::Standard, &mut ordinary),
        (Accuracy::Exact, &mut exact),
    ] {
        scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
            a,
            Triangle::Lower,
            StructuredOperation::Normal,
            1.0,
            0.0,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential)
                .with_accuracy(accuracy),
            MatrixViewMut::from_slice(3, 3, output).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
    }
    for (&x, &y) in ordinary.iter().zip(&exact) {
        assert!((x - y).abs() <= 4.0 * f64::EPSILON * y.abs());
    }
}

#[test]
fn overflow_and_nonfinite_input_preserve_output_and_provenance() {
    let right = [2.0];
    let mut output = [7.0];
    let mut workspace = LinalgWorkspace::new();
    for input in [f64::MAX, f64::NAN] {
        let values = [input];
        let error = MatrixView::from_slice(1, 1, &values)
            .unwrap()
            .mul_mat_into(
                MatrixView::from_slice(1, 1, &right).unwrap(),
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential)
                    .with_accuracy(Accuracy::Standard),
                MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
                &mut workspace,
                LinalgExecutionContext::sequential(),
            )
            .unwrap_err();
        match error.view() {
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            } => assert!(input.is_finite()),
            SciOxideErrorView::NonFinite { .. } => {
                assert!(input.is_nan());
            }
            error => panic!("unexpected error: {error:?}"),
        }
        assert_eq!(output, [7.0]);
    }
}

#[cfg(feature = "std")]
#[test]
fn standard_is_bitwise_independent_of_one_two_and_eight_workers() {
    use core::num::NonZeroUsize;
    let a = Matrix::from_vec(
        1025,
        35,
        (0..1025 * 35)
            .map(|i| ((i * 17) % 127) as f64 / 51.0 - 1.0)
            .collect(),
    )
    .unwrap();
    let b = Matrix::from_vec(
        35,
        41,
        (0..35 * 41)
            .map(|i| ((i * 13) % 131) as f64 / 49.0 - 1.0)
            .collect(),
    )
    .unwrap();
    static PORTABLE: scioxide_core::PortableProcessor = scioxide_core::PortableProcessor;
    static NATIVE: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
    for processor in [
        &PORTABLE as &dyn scioxide_core::ProcessorCapabilities,
        &NATIVE,
    ] {
        let mut results = Vec::new();
        for workers in [1, 2, 8] {
            let budget = NonZeroUsize::new(workers).unwrap();
            let executor = scioxide_core::ThreadExecutor::new(budget);
            let context = LinalgExecutionContext::new(
                ExecutionContext::new(&executor).with_processor(processor),
            );
            let mut result = Matrix::zeros(a.rows(), b.cols()).unwrap();
            a.view()
                .mul_mat_into(
                    &b,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Parallel(budget)),
                    result.view_mut(),
                    &mut LinalgWorkspace::new(),
                    context,
                )
                .unwrap();
            results.push(
                result
                    .as_slice()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }
}

#[test]
fn prepared_context_does_not_repeat_processor_detection() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct Processor(AtomicUsize);
    impl scioxide_core::ProcessorCapabilities for Processor {
        fn supports(&self, _: scioxide_core::ProcessorFeature) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
    let processor = Processor(AtomicUsize::new(0));
    let mut context =
        LinalgExecutionContext::new(ExecutionContext::sequential().with_processor(&processor));
    processor.0.store(0, Ordering::Relaxed);
    let a = Matrix::from_vec(35, 35, vec![0.5_f64; 35 * 35]).unwrap();
    let mut c = Matrix::zeros(35, 35).unwrap();
    let mut workspace = LinalgWorkspace::new();
    workspace
        .try_reserve(
            &scioxide_linalg::LinalgWorkspace::<f64>::workspace_requirements(
                &scioxide_linalg::DenseWorkspaceRequest::product(
                    35,
                    35,
                    35,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    let x = [1.0; 35];
    let mut y = [0.0; 35];
    for _ in 0..3 {
        a.view()
            .mul_mat_into(
                &a,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                c.view_mut(),
                &mut workspace,
                context.reborrow(),
            )
            .unwrap();
        a.view()
            .mul_vec_into(
                VectorView::from_slice(&x),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                VectorViewMut::from_slice(&mut y),
                &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut [0.0; 35]),
                context.reborrow(),
            )
            .unwrap();
        VectorView::from_slice(&x)
            .dotu_scoped(
                VectorView::from_slice(&x),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                context.reborrow(),
            )
            .unwrap();
    }
    assert_eq!(processor.0.load(Ordering::Relaxed), 0);
}

#[test]
fn widened_and_offset_dots_obey_the_selected_tier() {
    let x = [1.0e20_f32, 1.0, -1.0e20];
    let y = [1.0; 3];
    let x = VectorView::from_slice(&x);
    let y = VectorView::from_slice(&y);
    assert_eq!(
        x.dot_wide_scoped(
            y,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
            LinalgExecutionContext::sequential()
        )
        .unwrap(),
        1.0
    );
    assert_eq!(
        x.dot_plus_scoped(
            y,
            2.0,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
            LinalgExecutionContext::sequential()
        )
        .unwrap(),
        3.0
    );
    let wide = x
        .dot_wide_scoped(
            y,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!((wide - 1.0).abs() <= 3.0 * f64::EPSILON / (1.0 - 3.0 * f64::EPSILON) * 2.0e20);
    let offset = x
        .dot_plus_scoped(
            y,
            2.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!((offset - 3.0).abs() <= 4.0 * f32::EPSILON / (1.0 - 4.0 * f32::EPSILON) * 2.0e20);
    let extreme = [f32::MAX, f32::MAX];
    let weights = [2.0, -2.0];
    assert_eq!(
        VectorView::from_slice(&extreme)
            .dot_plus_scoped(
                VectorView::from_slice(&weights),
                1.0,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
        1.0
    );
}

#[test]
fn standard_products_preserve_the_destination_subnormal_grid() {
    let tiny = f64::from_bits(1);
    let a = Matrix::from_vec(1, 1, vec![tiny]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    assert_eq!(
        product(&a, &b, Accuracy::Standard)
            .view()
            .get(0, 0)
            .unwrap()
            .to_bits(),
        1
    );
    assert_eq!(
        VectorView::from_slice(&[tiny])
            .dotu_scoped(
                VectorView::from_slice(&[1.0]),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap()
            .to_bits(),
        1
    );
    let mut output = [0.0];
    VectorView::from_slice(&[tiny])
        .axpy_into(
            1.0,
            VectorView::from_slice(&[0.0]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut [0.0]),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output[0].to_bits(), 1);
}

#[cfg(feature = "std")]
fn native_affine<R: RealScalar + NumericScalar<Real = R>>() {
    static PROCESSOR: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
    let mut native =
        LinalgExecutionContext::new(ExecutionContext::sequential().with_processor(&PROCESSOR));
    let scalar = |value| R::try_from_f64_lossy(value).unwrap();
    let left: Vec<_> = (0..70)
        .map(|i| scalar(((i * 17) % 47) as f64 / 13.0 - 1.0))
        .collect();
    let right: Vec<_> = (0..70)
        .map(|i| scalar(((i * 29) % 53) as f64 / 19.0 - 1.0))
        .collect();
    let a = scalar(-0.625);
    let b = scalar(1.75);
    let mut output = [scalar(99.0); 69];
    let mut exact = vec![R::ZERO; 67];
    let mut scratch = vec![R::ZERO; 67];
    for (accuracy, out) in [
        (Accuracy::Standard, &mut output[1..68]),
        (Accuracy::Exact, &mut exact[..]),
    ] {
        let context = native.reborrow();
        VectorView::from_slice(&left[1..68])
            .axpby_into(
                a,
                VectorView::from_slice(&right[2..69]),
                b,
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                VectorViewMut::from_slice(out),
                &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
                context,
            )
            .unwrap();
    }
    let gamma = scalar(2.0) * R::EPSILON / (R::ONE - scalar(2.0) * R::EPSILON);
    for (i, (&actual, &expected)) in output[1..68].iter().zip(&exact).enumerate() {
        assert!(
            (actual - expected).abs()
                <= gamma * ((a * left[i + 1]).abs() + (b * right[i + 2]).abs())
        );
    }
    assert_eq!(output[0], scalar(99.0));
    assert_eq!(output[68], scalar(99.0));
}

#[cfg(feature = "std")]
#[test]
fn native_validation_and_affine_updates_cover_unaligned_tails() {
    native_affine::<f32>();
    native_affine::<f64>();
    static PROCESSOR: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
    let mut native =
        LinalgExecutionContext::new(ExecutionContext::sequential().with_processor(&PROCESSOR));
    for length in [1, 3, 4, 7, 8, 15, 16, 33, 67] {
        for position in [0, length - 1] {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut data = vec![1.0; length + 1];
                data[position + 1] = invalid;
                let input = VectorView::from_slice(&data[1..]);
                let error = input
                    .dotu_scoped(
                        input,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        native.reborrow(),
                    )
                    .unwrap_err();
                assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
            }
        }
    }
}

fn sample(index: usize) -> f64 {
    ((index * 37 + 11) % 101) as f64 / 512.0 - 0.1
}

// Evaluate the bound in the operands' original binary scale so neither the
// absolute-product sum nor subtraction overflows before the assertion.
fn assert_scaled_bound(actual: f64, exact: f64, absolute: f64, exponent: i32, terms: usize) {
    let gamma = terms as f64 * f64::EPSILON / (1.0 - terms as f64 * f64::EPSILON);
    let growth = usize::BITS as i32 - terms.max(1).leading_zeros() as i32 + 4;
    let restore = (exponent + growth).max(0);
    let underflow = terms as f64 * f64::from_bits(1).scalbn(restore - exponent);
    let error = (actual.scalbn(-exponent) - exact.scalbn(-exponent)).abs();
    assert!(
        error <= gamma * absolute + underflow,
        "error {error:e} exceeds bound {:e} at scale 2^{exponent}",
        gamma * absolute + underflow
    );
}

#[test]
fn gemm_gemv_and_dot_bounds_cover_scaled_blocked_inputs() {
    let (m, k, n) = (35, 39, 37);
    for (ae, be) in [(997, -997), (-997, 997), (510, 510), (-537, -537)] {
        let a = Matrix::from_vec(m, k, (0..m * k).map(|i| sample(i).scalbn(ae)).collect()).unwrap();
        let b = Matrix::from_vec(
            k,
            n,
            (0..k * n).map(|i| sample(i + 113).scalbn(be)).collect(),
        )
        .unwrap();
        let standard = product(&a, &b, Accuracy::Standard);
        let exact = product(&a, &b, Accuracy::Exact);
        for row in 0..m {
            for col in 0..n {
                let absolute = (0..k)
                    .map(|inner| {
                        sample(row * k + inner).abs() * sample(inner * n + col + 113).abs()
                    })
                    .sum();
                assert_scaled_bound(
                    standard.view().get(row, col).unwrap(),
                    exact.view().get(row, col).unwrap(),
                    absolute,
                    ae + be,
                    k,
                );
            }
        }
        let x: Vec<_> = (0..k).map(|i| sample(i + 7).scalbn(be)).collect();
        let mut standard = vec![0.0; m];
        let mut exact = standard.clone();
        for (tier, out) in [
            (Accuracy::Standard, &mut standard),
            (Accuracy::Exact, &mut exact),
        ] {
            a.view()
                .mul_vec_into(
                    VectorView::from_slice(&x),
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(tier),
                    VectorViewMut::from_slice(out),
                    &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut vec![0.0; m]),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
        }
        for row in 0..m {
            let absolute = (0..k)
                .map(|i| sample(row * k + i).abs() * sample(i + 7).abs())
                .sum();
            assert_scaled_bound(standard[row], exact[row], absolute, ae + be, k);
        }
        let left: Vec<_> = (0..k).map(|i| sample(i).scalbn(ae)).collect();
        let dot = |tier| {
            VectorView::from_slice(&left)
                .dotu_scoped(
                    VectorView::from_slice(&x),
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(tier),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap()
        };
        let absolute = (0..k).map(|i| sample(i).abs() * sample(i + 7).abs()).sum();
        assert_scaled_bound(
            dot(Accuracy::Standard),
            dot(Accuracy::Exact),
            absolute,
            ae + be,
            k,
        );
    }
}

#[test]
fn syrk_and_trmm_bounds_cover_block_boundaries_and_range_extremes() {
    let (n, k) = (35, 39);
    for exponent in [0, 510, -537] {
        let a = Matrix::from_vec(
            n,
            k,
            (0..n * k).map(|i| sample(i).scalbn(exponent)).collect(),
        )
        .unwrap();
        let mut standard = vec![99.0; n * n];
        let mut exact = standard.clone();
        for (tier, out) in [
            (Accuracy::Standard, &mut standard),
            (Accuracy::Exact, &mut exact),
        ] {
            scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
                a.view(),
                Triangle::Lower,
                StructuredOperation::Normal,
                1.0,
                0.0,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential)
                    .with_accuracy(tier),
                MatrixViewMut::from_slice(n, n, out).unwrap(),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential(),
            )
            .unwrap();
        }
        for row in 0..n {
            for col in 0..n {
                if row < col {
                    assert_eq!(standard[row * n + col].to_bits(), 99.0_f64.to_bits());
                    continue;
                }
                let absolute = (0..k)
                    .map(|i| sample(row * k + i).abs() * sample(col * k + i).abs())
                    .sum();
                assert_scaled_bound(
                    standard[row * n + col],
                    exact[row * n + col],
                    absolute,
                    2 * exponent,
                    k,
                );
            }
        }
    }
    for (ae, be) in [(997, -997), (-997, 997), (510, 510), (-537, -537)] {
        let triangular: Vec<_> = (0..n * n)
            .map(|i| {
                if i / n >= i % n {
                    sample(i).scalbn(ae)
                } else {
                    f64::NAN
                }
            })
            .collect();
        let initial: Vec<_> = (0..n * 3).map(|i| sample(i + 7).scalbn(be)).collect();
        let mut standard = initial.clone();
        let mut exact = initial;
        for (tier, out) in [
            (Accuracy::Standard, &mut standard),
            (Accuracy::Exact, &mut exact),
        ] {
            MatrixView::from_slice(n, n, &triangular)
                .unwrap()
                .triangular_mul_mat_in_place(
                    MatrixSide::Left,
                    Triangle::Lower,
                    DiagonalKind::Explicit,
                    StructuredOperation::Normal,
                    1.0,
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(tier),
                    MatrixViewMut::from_slice(n, 3, out).unwrap(),
                    &mut LinalgWorkspace::new(),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
        }
        for row in 0..n {
            for col in 0..3 {
                let absolute = (0..=row)
                    .map(|i| sample(row * n + i).abs() * sample(i * 3 + col + 7).abs())
                    .sum();
                assert_scaled_bound(
                    standard[row * 3 + col],
                    exact[row * 3 + col],
                    absolute,
                    ae + be,
                    n,
                );
            }
        }
    }
}

#[test]
fn trsm_has_small_backward_error_for_scaled_blocked_systems() {
    let n = 67;
    for exponent in [0, 997, -997] {
        for side in [MatrixSide::Left, MatrixSide::Right] {
            let t: Vec<_> = (0..n * n)
                .map(|i| {
                    if i / n == i % n {
                        1.0
                    } else if i / n > i % n {
                        sample(i) / n as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let triangular: Vec<_> = t.iter().map(|v| v.scalbn(exponent)).collect();
            let (rows, cols) = if side == MatrixSide::Left {
                (n, 3)
            } else {
                (3, n)
            };
            let initial: Vec<_> = (0..rows * cols).map(|i| sample(i + 7)).collect();
            let mut output: Vec<_> = initial.iter().map(|v| v.scalbn(exponent)).collect();
            MatrixView::from_slice(n, n, &triangular)
                .unwrap()
                .triangular_solve_mat_in_place(
                    side,
                    Triangle::Lower,
                    DiagonalKind::Explicit,
                    StructuredOperation::Normal,
                    1.0,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_accuracy(Accuracy::Standard),
                    MatrixViewMut::from_slice(rows, cols, &mut output).unwrap(),
                    &mut LinalgWorkspace::new(),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
            let gamma = (n + 1) as f64 * f64::EPSILON / (1.0 - (n + 1) as f64 * f64::EPSILON);
            for row in 0..rows {
                for col in 0..cols {
                    let terms: Vec<_> = (0..n)
                        .map(|i| {
                            if side == MatrixSide::Left {
                                (t[row * n + i], output[i * cols + col])
                            } else {
                                (output[row * cols + i], t[i * n + col])
                            }
                        })
                        .collect();
                    let mut residual = scioxide_core::ProductSum::new();
                    for &(a, b) in &terms {
                        residual.add_product(a, b).unwrap();
                    }
                    residual
                        .add_product(-initial[row * cols + col], 1.0)
                        .unwrap();
                    let absolute = terms.iter().map(|(a, b)| a.abs() * b.abs()).sum::<f64>()
                        + initial[row * cols + col].abs();
                    assert!(residual.finish().unwrap().abs() <= gamma * absolute);
                }
            }
        }
    }
}

#[test]
fn restore_overflow_is_transactional_across_product_families() {
    let mut output = [7.0_f64];
    let mut scratch = [0.0];
    let mut workspace = LinalgWorkspace::new();
    let overflow = |error: scioxide_core::SciOxideError| {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ))
    };
    let values = [f64::MAX];
    let a = MatrixView::from_slice(1, 1, &values).unwrap();
    overflow(
        a.mul_vec_into(
            VectorView::from_slice(&[2.0]),
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Standard),
            VectorViewMut::from_slice(&mut output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            LinalgExecutionContext::sequential(),
        )
        .unwrap_err(),
    );
    assert_eq!(output, [7.0]);
    overflow(
        scioxide_linalg::MatrixViewMut::symmetric_rank_k_in_place(
            a,
            Triangle::Lower,
            StructuredOperation::Normal,
            1.0,
            0.0,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential)
                .with_accuracy(Accuracy::Standard),
            MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
            &mut workspace,
            LinalgExecutionContext::sequential(),
        )
        .unwrap_err(),
    );
    assert_eq!(output, [7.0]);
    overflow(
        a.triangular_mul_mat_in_place(
            MatrixSide::Left,
            Triangle::Lower,
            DiagonalKind::Explicit,
            StructuredOperation::Normal,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Standard),
            MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
            &mut workspace,
            LinalgExecutionContext::sequential(),
        )
        .unwrap_err(),
    );
    assert_eq!(output, [7.0]);
    let divisor = [f64::MIN_POSITIVE];
    overflow(
        MatrixView::from_slice(1, 1, &divisor)
            .unwrap()
            .triangular_solve_mat_in_place(
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Normal,
                1.0,
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Standard),
                MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
                &mut workspace,
                LinalgExecutionContext::sequential(),
            )
            .unwrap_err(),
    );
    assert_eq!(output, [7.0]);
}

#[test]
fn zero_scale_trsm_ignores_nonfinite_factor_and_prior_rhs() {
    let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let factor = MatrixView::from_slice(2, 2, &values).unwrap();
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        for side in [MatrixSide::Left, MatrixSide::Right] {
            let mut output = values;
            factor
                .triangular_solve_mat_in_place(
                    side,
                    Triangle::Lower,
                    DiagonalKind::Explicit,
                    StructuredOperation::Normal,
                    0.0,
                    &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                    MatrixViewMut::from_slice(2, 2, &mut output).unwrap(),
                    &mut LinalgWorkspace::new(),
                    LinalgExecutionContext::sequential(),
                )
                .unwrap();
            assert!(output.iter().all(|value| value.to_bits() == 0));
        }
    }
}

#[cfg(feature = "std")]
fn blocked_reductions<R: RealScalar + NumericScalar<Real = R>>() {
    static PROCESSOR: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
    let mut native =
        LinalgExecutionContext::new(ExecutionContext::sequential().with_processor(&PROCESSOR));
    let scalar = |value| R::try_from_f64_exact(value).unwrap();
    for length in [
        0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 79, 129,
    ] {
        let left: Vec<_> = (0..length + 1)
            .map(|i| scalar(((i * 11) % 19) as f64 / 8.0 - 1.0))
            .collect();
        let right: Vec<_> = (0..length + 1)
            .map(|i| scalar(((i * 7) % 17) as f64 / 8.0 - 1.0))
            .collect();
        let x = VectorView::from_slice(&left[1..]);
        let y = VectorView::from_slice(&right[1..]);
        let exact = x
            .dotu_scoped(
                y,
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                native.reborrow(),
            )
            .unwrap();
        let actual = x
            .dotu_scoped(
                y,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                native.reborrow(),
            )
            .unwrap();
        let absolute = left[1..]
            .iter()
            .zip(&right[1..])
            .fold(R::ZERO, |sum, (&a, &b)| sum + (a * b).abs());
        let tolerance = scalar((length + 1) as f64) * R::EPSILON;
        assert!((actual - exact).abs() <= tolerance * absolute);
        let exact = x
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                native.reborrow(),
            )
            .unwrap();
        let actual = x
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                native.reborrow(),
            )
            .unwrap();
        assert!((actual - exact).abs() <= tolerance * exact);
        if length == 0 {
            continue;
        }
        // The largest component can occupy any unrolled lane or remainder.
        for position in 0..length {
            let mut input = vec![R::ZERO; length + 1];
            input[position + 1] = R::MAX;
            let x = VectorView::from_slice(&input[1..]);
            let mut output = vec![scalar(19.0); length];
            let mut scratch = vec![R::ZERO; length];
            assert!(
                x.axpby_into(
                    R::TWO,
                    x,
                    R::ZERO,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut output),
                    &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
                    native.reborrow()
                )
                .is_err()
            );
            assert!(output.iter().all(|&value| value == scalar(19.0)));
            let norm = x
                .norm2_scoped(
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    native.reborrow(),
                )
                .unwrap();
            assert!((norm / R::MAX - R::ONE).abs() <= scalar(4.0) * R::EPSILON);
        }
    }
    for value in [
        R::MIN_SUBNORMAL,
        R::MIN_POSITIVE,
        R::MIN_POSITIVE.sqrt(),
        R::MAX.sqrt() / scalar(4.0),
    ] {
        let data = [value; 65];
        let input = VectorView::from_slice(&data);
        let actual = input
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                native.reborrow(),
            )
            .unwrap();
        let exact = input
            .norm2_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                native.reborrow(),
            )
            .unwrap();
        assert!((actual - exact).abs() <= scalar(65.0) * R::EPSILON * exact + R::MIN_SUBNORMAL);
    }
}

#[cfg(feature = "std")]
#[test]
fn blocked_reductions_and_range_proofs_cover_all_lanes_and_remainders() {
    blocked_reductions::<f32>();
    blocked_reductions::<f64>();
}

#[test]
fn blocked_gemv_failure_in_last_block_preserves_all_output() {
    for last in [f64::NAN, f64::MAX] {
        let mut data = [1.0; 9 * 33];
        data[9 * 33 - 1] = last;
        let mut output = [17.0; 9];
        let mut scratch = [0.0; 9];
        let error = MatrixView::from_slice(9, 33, &data)
            .unwrap()
            .mul_vec_into(
                VectorView::from_slice(&[2.0; 33]),
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Standard),
                VectorViewMut::from_slice(&mut output),
                &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
                LinalgExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite { .. } | SciOxideErrorView::NumericalInstability { .. }
        ));
        assert_eq!(output, [17.0; 9]);
    }
}

#[test]
fn exact_rank_two_updates_round_a_zero_alpha_scaling_once() {
    // re(beta * c) = 3 * 2^-52 + 2^-103 is representable; ordinary complex
    // multiplication rounds the 1 + ... products first and loses 2^-103.
    let beta = Complex::new(1.0 + f64::EPSILON, 1.0);
    let c = Complex::new(1.0 + 2.0 * f64::EPSILON, 1.0);
    let expected = 3.0 * f64::EPSILON + 2f64.powi(-103);
    let zeros = [Complex::new(0.0, 0.0); 2];
    let mut output = [c];
    MatrixViewMut::symmetric_rank_two_k_in_place(
        MatrixView::from_slice(1, 2, &zeros).unwrap(),
        MatrixView::from_slice(1, 2, &zeros).unwrap(),
        Triangle::Lower,
        StructuredOperation::Normal,
        Complex::new(0.0, 0.0),
        beta,
        &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
        MatrixViewMut::from_slice(1, 1, &mut output).unwrap(),
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output[0].re, expected);
    assert_eq!(output[0].im, 2.0 + 3.0 * f64::EPSILON);
}

#[test]
fn exact_trsm_solves_systems_with_overflowing_unit_frame_products() {
    // x1 = (1e300 - 1e300 * 1e10) / 1e300: the product overflows although
    // the solution is ordinary.
    let triangular = [1.0, 0.0, 1e300, 1e300];
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let mut rhs = [1e10, 1e300];
        MatrixView::from_slice(2, 2, &triangular)
            .unwrap()
            .triangular_solve_mat_in_place(
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Normal,
                1.0,
                &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
                MatrixViewMut::from_slice(2, 1, &mut rhs).unwrap(),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(rhs[0], 1e10);
        assert!((rhs[1] + 9_999_999_999.0).abs() <= 4.0 * f64::EPSILON * 1e10);
    }
}

#[test]
fn standard_products_keep_an_out_of_range_alpha_a_with_an_ordinary_product() {
    // alpha*A = 2^-1200 underflows or 2^1200 overflows, while alpha*A*B = 2^-600
    // or 2^600 is ordinary.
    let n = 4;
    for (accuracy, exponent) in [
        (Accuracy::Standard, -600),
        (Accuracy::Exact, -600),
        (Accuracy::Standard, 600),
        (Accuracy::Exact, 600),
    ] {
        let (alpha, a_entry, b_entry) = (
            2f64.powi(exponent),
            2f64.powi(exponent),
            2f64.powi(-exponent),
        );
        let expected = 2f64.powi(exponent);
        let close = |actual: &[f64]| {
            for &value in actual {
                assert!(
                    (value - expected).abs() <= 8.0 * f64::EPSILON * expected,
                    "{value:e} vs {expected:e}"
                );
            }
        };
        let options = scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential)
            .with_accuracy(accuracy);
        // A = tiny * I keeps every product with B a single term.
        let mut diagonal = vec![0.0; n * n];
        for i in 0..n {
            diagonal[i * n + i] = a_entry;
        }
        let a = MatrixView::from_slice(n, n, &diagonal).unwrap();
        let general = vec![b_entry; n * n];
        let mut output = vec![0.0; n * n];
        a.symmetric_mul_mat_in_place(
            MatrixView::from_slice(n, n, &general).unwrap(),
            MatrixSide::Left,
            Triangle::Lower,
            alpha,
            0.0,
            &options,
            MatrixViewMut::from_slice(n, n, &mut output).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(&output);
        let mut output = vec![b_entry; n * n];
        a.triangular_mul_mat_in_place(
            MatrixSide::Left,
            Triangle::Lower,
            DiagonalKind::Explicit,
            StructuredOperation::Normal,
            alpha,
            &options,
            MatrixViewMut::from_slice(n, n, &mut output).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(&output);
        let x = vec![b_entry; n];
        let mut y = vec![0.0; n];
        a.symmetric_mul_vec_in_place(
            Triangle::Lower,
            alpha,
            VectorView::from_slice(&x),
            0.0,
            &options,
            VectorViewMut::from_slice(&mut y),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(&y);
        let band_values = vec![a_entry; n];
        let band = scioxide_linalg::BandedMatrixView::new(n, n, 0, 0, 1, &band_values).unwrap();
        let mut y = vec![0.0; n];
        band.mul_vec_in_place(
            StructuredOperation::Normal,
            alpha,
            VectorView::from_slice(&x),
            0.0,
            &options,
            VectorViewMut::from_slice(&mut y),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(&y);
    }
}

#[test]
fn standard_products_keep_small_entries_of_a_folded_operand() {
    // alpha * 2^-500 = 2^-1100 underflows although alpha * 2^-500 * 2^500 = 2^-600 is
    // ordinary; the largest entry 2^500 alone does not reveal the loss.
    let power = |exponent: i32| 2f64.powi(exponent);
    let alpha = power(-600);
    let expected = power(-600);
    let close = |value: f64| {
        assert!(
            (value - expected).abs() <= 4.0 * f64::EPSILON * expected,
            "{value:e} vs {expected:e}"
        );
    };
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential)
            .with_accuracy(accuracy);
        // A prior of 2^1020 makes the product evaluate in the unit frame first.
        for prior in [1.0, power(1020)] {
            let a = [power(-500), power(500)];
            let mut c = [prior, 0.0, 0.0, 0.0];
            MatrixViewMut::symmetric_rank_k_in_place(
                MatrixView::from_slice(2, 1, &a).unwrap(),
                Triangle::Upper,
                StructuredOperation::Normal,
                alpha,
                1.0,
                &options,
                MatrixViewMut::from_slice(2, 2, &mut c).unwrap(),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential(),
            )
            .unwrap();
            close(c[1]);
            assert_eq!(c[0], prior);
        }
        let diagonal = [power(-500), 0.0, 0.0, power(500)];
        let a = MatrixView::from_slice(2, 2, &diagonal).unwrap();
        let b = [power(500), 1.0];
        let mut c = [0.0; 2];
        a.symmetric_mul_mat_in_place(
            MatrixView::from_slice(2, 1, &b).unwrap(),
            MatrixSide::Left,
            Triangle::Lower,
            alpha,
            0.0,
            &options,
            MatrixViewMut::from_slice(2, 1, &mut c).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(c[0]);
        let mut x = b;
        a.triangular_mul_mat_in_place(
            MatrixSide::Left,
            Triangle::Lower,
            DiagonalKind::Explicit,
            StructuredOperation::Normal,
            alpha,
            &options,
            MatrixViewMut::from_slice(2, 1, &mut x).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(x[0]);
        let band_values = [power(-500), power(500)];
        let band = scioxide_linalg::BandedMatrixView::new(2, 2, 0, 0, 1, &band_values).unwrap();
        let mut y = [0.0; 2];
        band.mul_vec_in_place(
            StructuredOperation::Normal,
            alpha,
            VectorView::from_slice(&b),
            0.0,
            &options,
            VectorViewMut::from_slice(&mut y),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        close(y[0]);
        // 0.75 * 3 * 2^-1074 rounds on the subnormal grid; the product with 2^1000 is
        // 2.25 * 2^-74 exactly.
        let band_values = [f64::from_bits(3), power(500)];
        let band = scioxide_linalg::BandedMatrixView::new(2, 2, 0, 0, 1, &band_values).unwrap();
        let mut y = [0.0; 2];
        band.mul_vec_in_place(
            StructuredOperation::Normal,
            0.75,
            VectorView::from_slice(&[power(1000), 1.0]),
            0.0,
            &options,
            VectorViewMut::from_slice(&mut y),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(y[0], 2.25 * power(-74));
    }
}

#[test]
fn standard_products_with_unit_alpha_keep_subnormal_entries() {
    // alpha = 1 folds exactly, so a subnormal entry next to 2^500 is kept.
    let power = |exponent: i32| 2f64.powi(exponent);
    let a = [f64::from_bits(3), power(500)];
    let mut c = [0.0; 4];
    MatrixViewMut::symmetric_rank_k_in_place(
        MatrixView::from_slice(2, 1, &a).unwrap(),
        Triangle::Upper,
        StructuredOperation::Normal,
        1.0,
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut c).unwrap(),
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(c[1], 3.0 * power(-574));
}

#[test]
fn standard_rank_two_k_survives_an_overflowing_partial_sum() {
    // A = I, B = [[0, M], [-M, 0]]: (A Bᵀ + B Aᵀ)(1, 0) = M - M, so C(1, 0) = M although
    // the first product plus the prior is 2M.
    let max = f64::MAX;
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [0.0, max, -max, 0.0];
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let mut c = [0.0, 0.0, max, 0.5];
        MatrixViewMut::symmetric_rank_two_k_in_place(
            MatrixView::from_slice(2, 2, &a).unwrap(),
            MatrixView::from_slice(2, 2, &b).unwrap(),
            Triangle::Lower,
            StructuredOperation::Normal,
            1.0,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
            MatrixViewMut::from_slice(2, 2, &mut c).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(c, [0.0, 0.0, max, 0.5]);
        // C(1, 0) = M + M + M is rejected and leaves C unchanged.
        let mut c = [0.0, 0.0, max, 0.5];
        let error = MatrixViewMut::symmetric_rank_two_k_in_place(
            MatrixView::from_slice(2, 2, &a).unwrap(),
            MatrixView::from_slice(2, 2, &[0.0, max, max, 0.0]).unwrap(),
            Triangle::Lower,
            StructuredOperation::Normal,
            1.0,
            1.0,
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
            MatrixViewMut::from_slice(2, 2, &mut c).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(c, [0.0, 0.0, max, 0.5]);
    }
    // alpha = 0: beta * c = (2 * 0.6 M - 0.2 M, -0.4 M - 0.6 M) = (M, -M) is representable
    // although the partial product 1.2 M is not.
    let zeros = [Complex::new(0.0, 0.0)];
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let mut c = [Complex::new(0.6 * max, -0.2 * max)];
        MatrixViewMut::symmetric_rank_two_k_in_place(
            MatrixView::from_slice(1, 1, &zeros).unwrap(),
            MatrixView::from_slice(1, 1, &zeros).unwrap(),
            Triangle::Lower,
            StructuredOperation::Normal,
            Complex::new(0.0, 0.0),
            Complex::new(2.0, -1.0),
            &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
            MatrixViewMut::from_slice(1, 1, &mut c).unwrap(),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
        assert!((c[0].re - max).abs() <= 4.0 * f64::EPSILON * max);
        assert!((c[0].im + max).abs() <= 4.0 * f64::EPSILON * max);
    }
}

#[test]
fn trsm_growth_beyond_the_range_restarts_with_scaled_substitution() {
    // Every |entry| lies in [2^-256, 2^256], yet x = (2^512, 2^1000, -2^1000) and the
    // ordinary residual 2^256 * 2^1000 of the last row overflows.
    let power = |exponent: i32| 2f64.powi(exponent);
    let lower = [
        power(-256),
        0.0,
        0.0,
        -power(232),
        power(-256),
        0.0,
        0.0,
        power(256),
        power(256),
    ];
    let expected = [power(512), power(1000), -power(1000)];
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy);
        let solution = MatrixView::from_slice(3, 3, &lower)
            .unwrap()
            .triangular_solve_mat_with_options(
                MatrixView::from_slice(3, 1, &[power(256), 0.0, 0.0]).unwrap(),
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Normal,
                1.0,
                &options,
            )
            .unwrap();
        for (actual, expected) in solution.as_slice().iter().zip(expected) {
            assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs());
        }
        // X Lᵀ = bᵀ is the same system from the right.
        let mut rhs = [power(256), 0.0, 0.0];
        MatrixView::from_slice(3, 3, &lower)
            .unwrap()
            .triangular_solve_mat_in_place(
                MatrixSide::Right,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Transpose,
                1.0,
                &options,
                MatrixViewMut::from_slice(1, 3, &mut rhs).unwrap(),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential(),
            )
            .unwrap();
        for (actual, expected) in rhs.iter().zip(expected) {
            assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs());
        }
        // A solution beyond the range is still rejected without touching the data.
        let mut rhs = [power(512), 0.0, 0.0];
        let error = MatrixView::from_slice(3, 3, &lower)
            .unwrap()
            .triangular_solve_mat_in_place(
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Normal,
                1.0,
                &options,
                MatrixViewMut::from_slice(3, 1, &mut rhs).unwrap(),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(rhs, [power(512), 0.0, 0.0]);
    }
}

#[test]
fn complex_scaling_fails_only_for_unrepresentable_products() {
    // x * x with |x|² = 1.4 MAX: the partial product Re(x)² overflows, while the result
    // (MAX/√2 · 1.4, ...) ≈ (1.7796e308, 1.7796e308) fits.
    let radius = 1.4f64.sqrt() * f64::MAX.sqrt();
    let angle = core::f64::consts::PI / 8.0;
    let x = Complex::new(radius * angle.cos(), radius * angle.sin());
    let scale = 2f64.powi(-600);
    let reference = {
        let small =
            Complex::new(x.re * scale, x.im * scale) * Complex::new(x.re * scale, x.im * scale);
        Complex::new(small.re * 2f64.powi(1200), small.im * 2f64.powi(1200))
    };
    let close = |value: Complex<f64>| {
        assert!((value.re - reference.re).abs() <= 4.0 * f64::EPSILON * reference.re);
        assert!((value.im - reference.im).abs() <= 4.0 * f64::EPSILON * reference.im);
    };
    close(
        VectorView::from_slice(&[x])
            .try_scale(x)
            .unwrap()
            .as_slice()[0],
    );
    close(
        MatrixView::from_slice(1, 1, &[x])
            .unwrap()
            .try_scale(x)
            .unwrap()
            .as_slice()[0],
    );
    let mut values = [x];
    VectorViewMut::scale_in_place(
        x,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut values),
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    close(values[0]);
    // Exact rounds each component once: Re((1 + e)²) - 1 = 2e + e² keeps e² = 2^-60.
    let e = 2f64.powi(-30);
    let z = Complex::new(1.0 + e, 1.0);
    let mut values = [z];
    VectorViewMut::scale_in_place(
        z,
        &scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
        VectorViewMut::from_slice(&mut values),
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(values[0].re, 2.0 * e + e * e);
    assert_eq!(values[0].im, 2.0 + 2.0 * e);
    // A product beyond the range is rejected and leaves the vector unchanged.
    let mut values = [x, Complex::new(f64::MAX, f64::MAX)];
    let error = VectorViewMut::scale_in_place(
        Complex::new(2.0, 0.0),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut values),
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability { .. }
    ));
    assert_eq!(values, [x, Complex::new(f64::MAX, f64::MAX)]);
}

#[test]
fn exact_blocked_trsm_and_lu_solves_match_on_every_kernel_backend() {
    // Order 70 takes the blocked path; Exact must not depend on the dispatched kernels.
    let n = 70;
    let p = 5;
    let sample = |i: usize, seed: usize| {
        ((i * 2_654_435_761 + seed * 40_503) % 1_000_003) as f64 / 1_000_003.0 - 0.5
    };
    let lower: Vec<f64> = (0..n * n)
        .map(|i| {
            if i % (n + 1) == 0 {
                2.0 + sample(i, 1)
            } else {
                sample(i, 1)
            }
        })
        .collect();
    let rhs: Vec<f64> = (0..n * p).map(|i| sample(i, 2)).collect();
    let options = scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact);
    let solve = |context: ExecutionContext<'static>| {
        let mut x = rhs.clone();
        MatrixView::from_slice(n, n, &lower)
            .unwrap()
            .triangular_solve_mat_in_place(
                MatrixSide::Left,
                Triangle::Lower,
                DiagonalKind::Explicit,
                StructuredOperation::Normal,
                0.75,
                &options,
                MatrixViewMut::from_slice(n, p, &mut x).unwrap(),
                &mut LinalgWorkspace::new(),
                context,
            )
            .unwrap();
        x
    };
    let portable = solve(ExecutionContext::sequential());
    let native = solve(ExecutionContext::sequential_native());
    assert!(
        portable
            .iter()
            .zip(&native)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    // Residual of `L X = 0.75 B`, relative to `|L| |X| + 0.75 |B|`.
    for row in 0..n {
        for col in 0..p {
            let (mut residual, mut scale) =
                (-0.75 * rhs[row * p + col], 0.75 * rhs[row * p + col].abs());
            for k in 0..=row {
                residual += lower[row * n + k] * portable[k * p + col];
                scale += (lower[row * n + k] * portable[k * p + col]).abs();
            }
            assert!(residual.abs() <= 2.0 * n as f64 * f64::EPSILON * scale);
        }
    }
    let general: Vec<f64> = (0..n * n).map(|i| sample(i, 3)).collect();
    let lu = scioxide_linalg::LuDecomposition::factor(&Matrix::from_vec(n, n, general).unwrap())
        .unwrap();
    let solve_options =
        scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact)
            .with_execution_options(options);
    let lu_solve = |context: ExecutionContext<'static>| {
        let mut x = vec![0.0; n * p];
        let _ = lu
            .solve_matrix_into(
                MatrixView::from_slice(n, p, &rhs).unwrap(),
                &solve_options,
                MatrixViewMut::from_slice(n, p, &mut x).unwrap(),
                &mut LinalgWorkspace::new(),
                context,
            )
            .unwrap();
        x
    };
    let portable = lu_solve(ExecutionContext::sequential());
    let native = lu_solve(ExecutionContext::sequential_native());
    assert!(
        portable
            .iter()
            .zip(&native)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
