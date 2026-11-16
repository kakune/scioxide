//! The shared proof must survive normalized products and eigen dispatch.
use super::*;
use crate::{HermitianEigenDecomposition, HermitianEigenOptions, Matrix};
use scioxide_core::ProcessorCapabilities;
use scioxide_core::{InvalidInputReason, ProcessorFeature, SciOxideErrorView};

struct Liar;
impl ProcessorCapabilities for Liar {
    fn supports(&self, _: ProcessorFeature) -> bool {
        true
    }
}

#[test]
fn lying_provider_normalized_and_exact_products() {
    let context = ExecutionContext::sequential().with_processor(&Liar);
    let epsilon = 2.0_f64.powi(-27);
    let fused = (1.0 + epsilon).mul_add(1.0 - epsilon, -1.0);
    let unfused = (1.0 + epsilon) * (1.0 - epsilon) - 1.0;
    assert_ne!(fused, unfused);
    let left = Matrix::from_vec(
        17,
        19,
        (0..17 * 19)
            .map(|i| match i % 19 {
                0 => -1.0,
                1 => 1.0 + epsilon,
                _ => 0.0,
            })
            .collect(),
    )
    .unwrap();
    let right = Matrix::from_vec(
        19,
        33,
        (0..19 * 33)
            .map(|i| match i / 33 {
                0 => 1.0,
                1 => 1.0 - epsilon,
                _ => 0.0,
            })
            .collect(),
    )
    .unwrap();
    let mut actual = Matrix::from_vec(17, 33, alloc::vec![7.0; 17 * 33]).unwrap();
    for backend in [
        None,
        Some(KernelBackend::Scalar),
        Some(KernelBackend::Avx2Fma),
        Some(KernelBackend::Avx512),
        Some(KernelBackend::Neon),
    ] {
        let config = GemmConfig {
            backend,
            ..GemmConfig::ordinary()
        };
        let before = actual.clone();
        let result = (|| -> Result<(), SciOxideError> {
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                actual.view_mut(),
                config,
                ExecutionPolicy::Sequential,
                &mut LinalgWorkspace::new(),
                KernelContext::from(context),
            )
        })();
        if backend.is_none_or(|b| b.is_available(&Liar)) {
            result.unwrap();
            let dispatch = backend.map_or_else(
                || KernelDispatch::detect(&Liar),
                |b| KernelDispatch::for_backend(b, &Liar).unwrap(),
            );
            let uses_fma = arch::select_microkernel::<f64>(dispatch).is_some();
            for row in 0..17 {
                for col in 0..33 {
                    let expected = if uses_fma { fused } else { unfused };
                    assert_eq!(actual.get_element(row, col), expected);
                }
            }
            (|| -> Result<(), SciOxideError> {
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    actual.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut LinalgWorkspace::new(),
                    KernelContext::new(context).with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();
            assert!(actual.as_slice().iter().all(|&value| value == fused));
        } else {
            assert!(matches!(
                result.unwrap_err().view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
            assert_eq!(actual.as_slice(), before.as_slice());
        }
    }
    let values = [1.0, 2.0, 2.0];
    let view = crate::VectorView::from_slice(&values);
    assert_eq!(
        crate::reduction::dotu_with_context(view, view, context).unwrap(),
        9.0
    );
    assert_eq!(
        crate::reduction::l2_norm_with_context(view, context).unwrap(),
        3.0
    );
}

#[test]
fn lying_provider_hermitian_reduction_has_small_residual() {
    let n = 67;
    let matrix = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                if row == col {
                    70. + row as f64
                } else {
                    ((row + col) % 17) as f64 * 0.01
                }
            })
            .collect(),
    )
    .unwrap();
    let mut result = HermitianEigenDecomposition::empty();
    HermitianEigenDecomposition::decompose_into(
        &matrix,
        crate::HermitianInput::Exact,
        &HermitianEigenOptions::default(),
        &mut result,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential().with_processor(&Liar),
    )
    .unwrap();
    let vectors = result.eigenvectors();
    let residual = (0..n)
        .flat_map(|row| (0..n).map(move |col| (row, col)))
        .map(|(row, col)| {
            let product: f64 = (0..n)
                .map(|k| matrix.get_element(row, k) * vectors.get_element(k, col))
                .sum();
            (product - result.eigenvalues()[col] * vectors.get_element(row, col)).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(residual < 1e-10, "eigen residual: {residual}");
}
