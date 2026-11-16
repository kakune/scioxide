//! Dense range, layout, and transaction contract coverage.

#![allow(clippy::expect_used)]

use scioxide_core::{Complex, NdView, NdViewMut, StridedLayout};
use scioxide_linalg::{
    LinalgWorkspace, Matrix, MatrixView, MatrixViewMut, Vector, VectorView, VectorViewMut,
};

fn close(actual: f64, expected: f64) {
    crate::support::comparisons::within(
        actual,
        expected,
        4.0 * f64::EPSILON * expected.abs().max(1.0),
    );
}

#[test]
fn range_safe_kernels_cover_real_and_complex_scalar_families() {
    let f64_left = Vector::from_vec(vec![f64::MAX, f64::MAX]);
    let f64_right = Vector::from_vec(vec![2.0, -2.0]);
    assert_eq!(
        f64_left
            .view()
            .dotu_scoped(
                &f64_right,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .expect("finite cancellation"),
        0.0
    );

    let f32_left = Vector::from_vec(vec![f32::MAX, f32::MAX]);
    let f32_right = Vector::from_vec(vec![2.0, -2.0]);
    assert_eq!(
        f32_left
            .view()
            .dotu_scoped(
                &f32_right,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .expect("finite cancellation"),
        0.0
    );

    let complex_left = Vector::from_vec(vec![
        Complex::new(f64::MAX, f64::MAX),
        Complex::new(f64::MAX, f64::MAX),
    ]);
    let complex_right = Vector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(-1.0, -1.0)]);
    assert_eq!(
        complex_left
            .view()
            .dotu_scoped(
                &complex_right,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .expect("complex finite cancellation"),
        Complex::new(0.0, 0.0)
    );
    let complex32_left = Vector::from_vec(vec![
        Complex::new(f32::MAX, f32::MAX),
        Complex::new(f32::MAX, f32::MAX),
    ]);
    let complex32_right =
        Vector::from_vec(vec![Complex::new(1.0_f32, 1.0), Complex::new(-1.0, -1.0)]);
    assert_eq!(
        complex32_left
            .view()
            .dotu_scoped(
                &complex32_right,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .expect("complex32 finite cancellation"),
        Complex::new(0.0, 0.0)
    );

    let mut gemv_f64 = Vector::from_vec(vec![9.0]);
    Matrix::from_vec(1, 2, vec![f64::MAX, f64::MAX])
        .expect("valid matrix")
        .view()
        .mul_vec_into(
            &f64_right,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            gemv_f64.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe GEMV");
    assert_eq!(gemv_f64.as_slice(), &[0.0]);

    let mut gemv_f32 = Vector::from_vec(vec![9.0]);
    Matrix::from_vec(1, 2, vec![f32::MAX, f32::MAX])
        .expect("valid matrix")
        .view()
        .mul_vec_into(
            &f32_right,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            gemv_f32.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe GEMV");
    assert_eq!(gemv_f32.as_slice(), &[0.0]);

    let mut gemv_complex = Vector::from_vec(vec![Complex::new(9.0, 9.0)]);
    Matrix::from_vec(
        1,
        2,
        vec![
            Complex::new(f64::MAX, f64::MAX),
            Complex::new(f64::MAX, f64::MAX),
        ],
    )
    .expect("valid matrix")
    .view()
    .mul_vec_into(
        &complex_right,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        gemv_complex.view_mut(),
        &mut scioxide_linalg::LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .expect("range-safe complex GEMV");
    assert_eq!(gemv_complex.as_slice(), &[Complex::new(0.0, 0.0)]);
    let mut gemv_complex32 = Vector::from_vec(vec![Complex::new(9.0_f32, 9.0)]);
    Matrix::from_vec(
        1,
        2,
        vec![
            Complex::new(f32::MAX, f32::MAX),
            Complex::new(f32::MAX, f32::MAX),
        ],
    )
    .expect("valid complex32 matrix")
    .view()
    .mul_vec_into(
        &complex32_right,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        gemv_complex32.view_mut(),
        &mut scioxide_linalg::LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .expect("range-safe complex32 GEMV");
    assert_eq!(gemv_complex32.as_slice(), &[Complex::new(0.0, 0.0)]);

    let left = Matrix::from_vec(1, 2, vec![f64::MAX, f64::MAX]).expect("valid matrix");
    let right = Matrix::from_vec(2, 1, vec![2.0, -2.0]).expect("valid matrix");
    assert_eq!(
        left.view()
            .mul_mat(&right)
            .expect("range-safe GEMM")
            .view()
            .get(0, 0)
            .expect("product element"),
        0.0
    );
    let left = Matrix::from_vec(1, 2, vec![f32::MAX, f32::MAX]).expect("valid matrix");
    let right = Matrix::from_vec(2, 1, vec![2.0_f32, -2.0]).expect("valid matrix");
    assert_eq!(
        left.view()
            .mul_mat(&right)
            .expect("range-safe f32 GEMM")
            .view()
            .get(0, 0)
            .expect("product element"),
        0.0
    );
    let left = Matrix::from_vec(
        1,
        2,
        vec![
            Complex::new(f64::MAX, f64::MAX),
            Complex::new(f64::MAX, f64::MAX),
        ],
    )
    .expect("valid complex matrix");
    let right = Matrix::from_vec(2, 1, vec![Complex::new(1.0, 1.0), Complex::new(-1.0, -1.0)])
        .expect("valid complex matrix");
    assert_eq!(
        left.view()
            .mul_mat(&right)
            .expect("range-safe complex GEMM")
            .view()
            .get(0, 0)
            .expect("product element"),
        Complex::new(0.0, 0.0)
    );
    let left = Matrix::from_vec(
        1,
        2,
        vec![
            Complex::new(f32::MAX, f32::MAX),
            Complex::new(f32::MAX, f32::MAX),
        ],
    )
    .expect("valid complex32 matrix");
    let right = Matrix::from_vec(
        2,
        1,
        vec![Complex::new(1.0_f32, 1.0), Complex::new(-1.0, -1.0)],
    )
    .expect("valid complex32 matrix");
    assert_eq!(
        left.view()
            .mul_mat(&right)
            .expect("range-safe complex32 GEMM")
            .view()
            .get(0, 0)
            .expect("product element"),
        Complex::new(0.0, 0.0)
    );

    let cross_left = Vector::from_vec(vec![0.0_f64, f64::MAX, f64::MAX]);
    let cross_right = Vector::from_vec(vec![0.0_f64, 2.0, 2.0]);
    assert_eq!(
        cross_left
            .view()
            .cross(&cross_right)
            .expect("range-safe cross")
            .as_slice(),
        &[0.0; 3]
    );
    let cross_left = Vector::from_vec(vec![0.0_f32, f32::MAX, f32::MAX]);
    let cross_right = Vector::from_vec(vec![0.0_f32, 2.0, 2.0]);
    assert_eq!(
        cross_left
            .view()
            .cross(&cross_right)
            .expect("range-safe cross")
            .as_slice(),
        &[0.0; 3]
    );
    let cross_left = Vector::from_vec(vec![
        Complex::new(0.0_f64, 0.0),
        Complex::new(f64::MAX, f64::MAX),
        Complex::new(f64::MAX, f64::MAX),
    ]);
    let cross_right = Vector::from_vec(vec![
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(1.0, 1.0),
    ]);
    assert_eq!(
        cross_left
            .view()
            .cross(&cross_right)
            .expect("range-safe complex cross")
            .as_slice(),
        &[Complex::new(0.0, 0.0); 3]
    );
    let cross_left = Vector::from_vec(vec![
        Complex::new(0.0_f32, 0.0),
        Complex::new(f32::MAX, f32::MAX),
        Complex::new(f32::MAX, f32::MAX),
    ]);
    let cross_right = Vector::from_vec(vec![
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 1.0),
        Complex::new(1.0, 1.0),
    ]);
    assert_eq!(
        cross_left
            .view()
            .cross(&cross_right)
            .expect("range-safe complex32 cross")
            .as_slice(),
        &[Complex::new(0.0, 0.0); 3]
    );

    let mut affine = Vector::from_vec(vec![9.0_f64]);
    Vector::from_vec(vec![f64::MAX])
        .view()
        .axpby_into(
            2.0,
            &Vector::from_vec(vec![f64::MAX]),
            -2.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe AXPBY");
    assert_eq!(affine.as_slice(), &[0.0]);
    let mut affine = Vector::from_vec(vec![9.0_f32]);
    Vector::from_vec(vec![f32::MAX])
        .view()
        .axpby_into(
            2.0,
            &Vector::from_vec(vec![f32::MAX]),
            -2.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe f32 AXPBY");
    assert_eq!(affine.as_slice(), &[0.0]);
    let mut affine = Vector::from_vec(vec![Complex::new(9.0_f64, 9.0)]);
    Vector::from_vec(vec![Complex::new(f64::MAX, f64::MAX)])
        .view()
        .axpby_into(
            Complex::new(2.0, 0.0),
            &Vector::from_vec(vec![Complex::new(f64::MAX, f64::MAX)]),
            Complex::new(-2.0, 0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe complex AXPBY");
    assert_eq!(affine.as_slice(), &[Complex::new(0.0, 0.0)]);
    let mut affine = Vector::from_vec(vec![Complex::new(9.0_f32, 9.0)]);
    Vector::from_vec(vec![Complex::new(f32::MAX, f32::MAX)])
        .view()
        .axpby_into(
            Complex::new(2.0, 0.0),
            &Vector::from_vec(vec![Complex::new(f32::MAX, f32::MAX)]),
            Complex::new(-2.0, 0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe complex32 AXPBY");
    assert_eq!(affine.as_slice(), &[Complex::new(0.0, 0.0)]);

    let mut affine = Vector::from_vec(vec![-f64::MAX]);
    Vector::from_vec(vec![f64::MAX])
        .view()
        .axpy_into(
            2.0,
            &Vector::from_vec(vec![-f64::MAX]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe f64 AXPY");
    assert_eq!(affine.as_slice(), &[f64::MAX]);
    let mut affine = Vector::from_vec(vec![-f32::MAX]);
    Vector::from_vec(vec![f32::MAX])
        .view()
        .axpy_into(
            2.0,
            &Vector::from_vec(vec![-f32::MAX]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe f32 AXPY");
    assert_eq!(affine.as_slice(), &[f32::MAX]);
    let mut affine = Vector::from_vec(vec![Complex::new(-f64::MAX, -f64::MAX)]);
    Vector::from_vec(vec![Complex::new(f64::MAX, f64::MAX)])
        .view()
        .axpy_into(
            Complex::new(2.0, 0.0),
            &Vector::from_vec(vec![Complex::new(-f64::MAX, -f64::MAX)]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe complex AXPY");
    assert_eq!(affine.as_slice(), &[Complex::new(f64::MAX, f64::MAX)]);
    let mut affine = Vector::from_vec(vec![Complex::new(-f32::MAX, -f32::MAX)]);
    Vector::from_vec(vec![Complex::new(f32::MAX, f32::MAX)])
        .view()
        .axpy_into(
            Complex::new(2.0, 0.0),
            &Vector::from_vec(vec![Complex::new(-f32::MAX, -f32::MAX)]),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            affine.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .expect("range-safe complex32 AXPY");
    assert_eq!(affine.as_slice(), &[Complex::new(f32::MAX, f32::MAX)]);

    let normalized = f64_left.view().normalize().expect("MAX vector normalizes");
    let expected = 1.0 / 2.0_f64.sqrt();
    normalized
        .as_slice()
        .iter()
        .for_each(|&value| close(value, expected));
    let normalized = f32_left
        .view()
        .normalize()
        .expect("f32 MAX vector normalizes");
    normalized
        .as_slice()
        .iter()
        .for_each(|&value| assert!((value - 1.0 / 2.0_f32.sqrt()).abs() <= 4.0 * f32::EPSILON));
    let normalized = Vector::from_vec(vec![
        Complex::new(f64::MAX, 0.0),
        Complex::new(f64::MAX, 0.0),
    ])
    .view()
    .normalize()
    .expect("complex MAX vector normalizes");
    normalized
        .as_slice()
        .iter()
        .for_each(|value| close(value.re, expected));
    let normalized = Vector::from_vec(vec![
        Complex::new(f32::MAX, 0.0),
        Complex::new(f32::MAX, 0.0),
    ])
    .view()
    .normalize()
    .expect("complex32 MAX vector normalizes");
    normalized
        .as_slice()
        .iter()
        .for_each(|value| assert!((value.re - 1.0 / 2.0_f32.sqrt()).abs() <= 4.0 * f32::EPSILON));
}

#[test]
fn transformations_strides_batches_and_permutations_are_semantically_identical() {
    let left = [1.0e200_f64, 3.0, -1.0e200, 5.0];
    let right = [1.0_f64, 7.0, 1.0, 11.0];
    let baseline = VectorView::from_slice(&left)
        .dotu_scoped(
            VectorView::from_slice(&right),
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_accuracy(scioxide_core::Accuracy::Exact),
            scioxide_linalg::LinalgExecutionContext::sequential(),
        )
        .expect("baseline dot");
    for permutation in [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]] {
        let permuted_left = permutation.map(|index| left[index]);
        let permuted_right = permutation.map(|index| right[index]);
        assert_eq!(
            VectorView::from_slice(&permuted_left)
                .dotu_scoped(
                    VectorView::from_slice(&permuted_right),
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_accuracy(scioxide_core::Accuracy::Exact),
                    scioxide_linalg::LinalgExecutionContext::sequential()
                )
                .expect("permuted dot"),
            baseline
        );
    }

    let left_strided = [1.0e200, 99.0, 3.0, 99.0, -1.0e200, 99.0, 5.0];
    let right_strided = [1.0, 99.0, 7.0, 99.0, 1.0, 99.0, 11.0];
    assert_eq!(
        VectorView::from_slice_with_stride(&left_strided, 4, 2)
            .expect("valid stride")
            .dotu_scoped(
                VectorView::from_slice_with_stride(&right_strided, 4, 2).expect("valid stride"),
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_accuracy(scioxide_core::Accuracy::Exact),
                scioxide_linalg::LinalgExecutionContext::sequential()
            )
            .expect("strided dot"),
        baseline
    );

    let complex = [
        Complex::new(1.0_f64, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
        Complex::new(9.0, 10.0),
        Complex::new(11.0, 12.0),
    ];
    let matrix = MatrixView::from_slice(2, 3, &complex).expect("valid matrix");
    let transformed = matrix.adjoint();
    assert_eq!(transformed.rows(), 3);
    assert_eq!(transformed.cols(), 2);
    for row in 0..3 {
        for col in 0..2 {
            assert_eq!(
                transformed.get(row, col).expect("in bounds"),
                matrix.get(col, row).expect("in bounds").conj()
            );
        }
    }

    let batch_storage = [1.0_f64, 2.0, 3.0, 4.0, 99.0, 5.0, 6.0, 7.0, 8.0];
    let layout = StridedLayout::new(&[2, 2, 2], &[5, 2, 1]).expect("valid batch layout");
    let batch = NdView::new(&batch_storage, layout).expect("valid batch backing");
    let matrix_at = |index| {
        let start = batch
            .layout()
            .offset([index, 0, 0])
            .expect("batch in bounds");
        MatrixView::from_slice_with_strides(2, 2, &batch_storage[start..], 2, 1)
            .expect("valid matrix")
    };
    let input = VectorView::from_slice(&[1.0, -1.0]);
    let first = matrix_at(0).mul_vec(input).expect("first GEMV");
    let second = matrix_at(1).mul_vec(input).expect("second GEMV");
    assert_eq!(first.as_slice(), &[-1.0, -1.0]);
    assert_eq!(second.as_slice(), &[-1.0, -1.0]);
}

#[test]
fn invalid_alias_scratch_and_values_never_commit_outputs() {
    let mut overlapping_vector = [0.0_f64; 1];
    assert!(matches!(
        (VectorViewMut::from_slice_with_stride(&mut overlapping_vector, 2, 0))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
            | scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));

    let mut overlapping_matrix = [0.0_f64; 4];
    assert!(matches!(
        (MatrixViewMut::from_slice_with_strides(2, 2, &mut overlapping_matrix, 1, 1))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));

    let matrix = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).expect("valid matrix");
    let input = Vector::from_vec(vec![5.0, 6.0]);
    let mut output = Vector::from_vec(vec![7.0, 8.0]);
    let mut undersized = [0.0];
    assert!(matches!(
        (matrix.view().mul_vec_into(
            &input,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut undersized),
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output.as_slice(), &[7.0, 8.0]);

    let non_finite =
        Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, f64::NAN]).expect("raw storage accepts values");
    let mut scratch = [0.0; 2];
    assert!(matches!(
        (non_finite.view().mul_vec_into(
            &input,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(output.as_slice(), &[7.0, 8.0]);

    let left = Matrix::from_vec(1, 2, vec![f64::MAX, f64::MAX]).expect("valid matrix");
    let right = Matrix::from_vec(2, 1, vec![2.0, 2.0]).expect("valid matrix");
    let mut matrix_output = Matrix::from_vec(1, 1, vec![13.0]).expect("valid output");
    let mut workspace = LinalgWorkspace::new();
    assert!(matches!(
        (left.view().mul_mat_into(
            &right,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            matrix_output.view_mut(),
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ));
    assert_eq!(matrix_output.as_slice(), &[13.0]);

    let mut batches = [0.0_f64; 6];
    assert!(matches!(
        (NdViewMut::new(
            &mut batches,
            StridedLayout::new_broadcast(&[2, 2, 2], &[2, 2, 1]).expect("valid immutable layout")
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn zero_extent_metadata_obeys_f0_without_backing_allocation() {
    let vector = NdView::<f64>::new(
        &[],
        StridedLayout::new(&[usize::MAX, 0], &[usize::MAX, usize::MAX]).expect("empty layout"),
    )
    .expect("empty batched vector");
    assert!(vector.layout().shape().is_empty());

    let matrix =
        MatrixView::<f64>::from_slice_with_strides(0, usize::MAX, &[], usize::MAX, usize::MAX)
            .expect("empty matrix metadata");
    assert!(matrix.is_empty());
    assert_eq!(
        (matrix.transpose().rows(), matrix.transpose().cols()),
        (usize::MAX, 0)
    );

    let owned =
        Matrix::<f64>::from_vec(0, usize::MAX, Vec::new()).expect("owned empty metadata shape");
    assert_eq!((owned.rows(), owned.cols()), (0, usize::MAX));
    let product = owned
        .view()
        .mul_mat(&Matrix::from_vec(usize::MAX, 0, Vec::new()).expect("empty right matrix"))
        .expect("empty product");
    assert_eq!((product.rows(), product.cols()), (0, 0));
    assert!(product.as_slice().is_empty());
}

#[test]
fn level3_shape_errors_name_the_offending_operand() {
    use scioxide_core::{SciOxideErrorView, Triangle};
    use scioxide_linalg::{LinalgExecutionContext, MatrixSide, StructuredOperation};
    let mismatch = |error: scioxide_core::SciOxideError, expected: &[usize], actual: &[usize]| {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::DimensionMismatch { expected: e, actual: a, .. }
                    if e == expected && a == actual
            ),
            "{error:?}"
        );
    };
    let options = scioxide_linalg::LinalgExecutionOptions::new();
    let structured = vec![1.0_f64; 16];
    let general = vec![1.0_f64; 9];
    let mut output = vec![0.0_f64; 9];
    // A 4 x 4 structured operand against a 3 x 3 general operand and output.
    mismatch(
        MatrixView::from_slice(4, 4, &structured)
            .expect("structured view")
            .symmetric_mul_mat_in_place(
                MatrixView::from_slice(3, 3, &general).expect("general view"),
                MatrixSide::Left,
                Triangle::Lower,
                1.0,
                0.0,
                &options,
                MatrixViewMut::from_slice(3, 3, &mut output).expect("output view"),
                &mut LinalgWorkspace::new(),
                LinalgExecutionContext::sequential(),
            )
            .expect_err("structured order mismatch"),
        &[3, 3],
        &[4, 4],
    );
    // `b` must share the stored shape of `a`.
    let a = vec![1.0_f64; 6];
    let b = vec![1.0_f64; 12];
    mismatch(
        MatrixViewMut::symmetric_rank_two_k_in_place(
            MatrixView::from_slice(3, 2, &a).expect("a view"),
            MatrixView::from_slice(3, 4, &b).expect("b view"),
            Triangle::Lower,
            StructuredOperation::Normal,
            1.0,
            0.0,
            &options,
            MatrixViewMut::from_slice(3, 3, &mut output).expect("output view"),
            &mut LinalgWorkspace::new(),
            LinalgExecutionContext::sequential(),
        )
        .expect_err("rank-2k operand mismatch"),
        &[3, 2],
        &[3, 4],
    );
    // A non-real prior Hermitian diagonal names the output parameter.
    let complex = vec![Complex::new(1.0_f64, 0.0); 6];
    let mut prior = vec![Complex::new(1.0_f64, 0.0); 9];
    prior[0] = Complex::new(1.0, 1.0);
    let error = MatrixViewMut::hermitian_rank_two_k_in_place(
        MatrixView::from_slice(3, 2, &complex).expect("a view"),
        MatrixView::from_slice(3, 2, &complex).expect("b view"),
        Triangle::Lower,
        StructuredOperation::Normal,
        Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0),
        &options,
        MatrixViewMut::from_slice(3, 3, &mut prior).expect("output view"),
        &mut LinalgWorkspace::new(),
        LinalgExecutionContext::sequential(),
    )
    .expect_err("non-real diagonal");
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("data")
        ),
        "{error:?}"
    );
}

#[test]
fn from_rows_copies_rows_in_row_major_order() -> Result<(), scioxide_core::SciOxideError> {
    let m = Matrix::from_rows(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    assert_eq!((m.rows(), m.cols()), (2, 3));
    assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(m.view().get(1, 0)?, 4.0);
    let column = Matrix::from_rows(&[[1.0_f64], [2.0], [3.0]])?;
    assert_eq!((column.rows(), column.cols()), (3, 1));
    let complex = Matrix::from_rows(&[[Complex::new(1.0_f64, -1.0), Complex::new(0.0, 2.0)]])?;
    assert_eq!((complex.rows(), complex.cols()), (1, 2));
    assert_eq!(complex.view().get(0, 1)?, Complex::new(0.0, 2.0));
    // The entries are copied without a finiteness check.
    let unchecked = Matrix::from_rows(&[[f64::NAN, f64::INFINITY]])?;
    assert!(unchecked.as_slice()[0].is_nan() && unchecked.as_slice()[1] == f64::INFINITY);
    let empty = Matrix::<f64>::from_rows::<4>(&[])?;
    assert_eq!((empty.rows(), empty.cols()), (0, 4));
    assert!(empty.as_slice().is_empty());
    let no_columns = Matrix::<f64>::from_rows(&[[], []])?;
    assert_eq!((no_columns.rows(), no_columns.cols()), (2, 0));
    Ok(())
}

/// Documented relative error bound `γ_(2p+5) + γ_(8m+8) + γ_(2p+5) γ_(8m+8)` of a Frobenius
/// norm summed as `m` vectors of `p` real components each, where `γ_k = k u / (1 − k u)`.
fn frobenius_bound(m: usize, p: usize) -> f64 {
    let u = f64::EPSILON / 2.0;
    let gamma = |k: usize| k as f64 * u / (1.0 - k as f64 * u);
    let (row, sum) = (gamma(2 * p + 5), gamma(8 * m + 8));
    row + sum + row * sum
}

/// Checks the 1-, ∞- and Frobenius norms of a `2 × 2` view whose rows hold `p` real components.
fn check_norms<S: scioxide_core::NumericScalar<Real = f64>>(
    view: MatrixView<'_, S>,
    p: usize,
    [one, infinity, frobenius]: [f64; 3],
) -> Result<(), scioxide_core::SciOxideError> {
    use scioxide_linalg::MatrixNorm;
    close(view.norm(MatrixNorm::One)?, one);
    close(view.norm(MatrixNorm::Infinity)?, infinity);
    let actual = view.norm(MatrixNorm::Frobenius)?;
    assert!(
        (actual - frobenius).abs() <= frobenius_bound(view.rows(), p) * frobenius,
        "{actual} != {frobenius}"
    );
    Ok(())
}

#[test]
fn matrix_norms_match_hand_computed_values_on_every_layout()
-> Result<(), scioxide_core::SciOxideError> {
    // A = [1 −2; 3 4]: ‖A‖₁ = 6, ‖A‖∞ = 7, ‖A‖_F = √30.
    let frobenius = 30.0_f64.sqrt();
    let a = Matrix::from_rows(&[[1.0_f64, -2.0], [3.0, 4.0]])?;
    check_norms(a.view(), 2, [6.0, 7.0, frobenius])?;
    check_norms(a.view().transpose(), 2, [7.0, 6.0, frobenius])?;
    // Rows 4 apart and columns 2 apart; the NaN padding is never read.
    let n = f64::NAN;
    let strided = [1.0, n, -2.0, n, 3.0, n, 4.0];
    let strided = MatrixView::from_slice_with_strides(2, 2, &strided, 4, 2)?;
    check_norms(strided, 2, [6.0, 7.0, frobenius])?;
    let padded = Matrix::from_rows(&[[n, n, n], [n, 1.0, -2.0], [n, 3.0, 4.0]])?;
    let block = padded.view().block(1, 1, 2, 2)?;
    check_norms(block, 2, [6.0, 7.0, frobenius])?;
    check_norms(block.transpose(), 2, [7.0, 6.0, frobenius])?;

    // (3 + 4i) A has entries of modulus 5 |aᵢⱼ|, stored column-major here.
    let c = |re: f64, im: f64| Complex::new(re, im);
    let complex = [c(3.0, 4.0), c(9.0, 12.0), c(-6.0, -8.0), c(12.0, 16.0)];
    let complex = MatrixView::from_slice_with_strides(2, 2, &complex, 1, 2)?;
    let scaled = [30.0, 35.0, 5.0 * frobenius];
    check_norms(complex, 4, scaled)?;
    check_norms(complex.conjugate(), 4, scaled)?;
    check_norms(complex.adjoint(), 4, [35.0, 30.0, 5.0 * frobenius])?;
    Ok(())
}

#[test]
fn matrix_norms_stay_finite_near_overflow_and_reject_invalid_entries()
-> Result<(), scioxide_core::SciOxideError> {
    use scioxide_core::SciOxideErrorKind;
    use scioxide_linalg::MatrixNorm;
    let kinds = [MatrixNorm::One, MatrixNorm::Infinity, MatrixNorm::Frobenius];
    // Every norm of [q −q; q q] with q = MAX/4 is 2q = MAX/2, although q² overflows.
    let q = f64::MAX / 4.0;
    let large = Matrix::from_rows(&[[q, -q], [q, q]])?;
    check_norms(large.view(), 2, [2.0 * q; 3])?;
    // Moduli √2 q: ‖·‖₁ = 2√2 q, ‖·‖∞ = √2 q, ‖·‖_F = 2q.
    let column = Matrix::from_rows(&[[Complex::new(q, q)], [Complex::new(q, -q)]])?;
    let root = 2.0_f64.sqrt();
    close(column.view().norm(MatrixNorm::One)?, 2.0 * root * q);
    close(column.view().norm(MatrixNorm::Infinity)?, root * q);
    let actual = column.view().norm(MatrixNorm::Frobenius)?;
    assert!((actual - 2.0 * q).abs() <= frobenius_bound(2, 2) * 2.0 * q);
    // A norm above MAX overflows: ‖·‖₁ = 1.5 MAX and ‖·‖_F = 0.75 √2 MAX.
    let huge = Matrix::from_rows(&[[0.75 * f64::MAX], [0.75 * f64::MAX]])?;
    for kind in [MatrixNorm::One, MatrixNorm::Frobenius] {
        assert_eq!(
            huge.view().norm(kind).unwrap_err().kind(),
            SciOxideErrorKind::NumericalInstability
        );
    }
    close(huge.view().norm(MatrixNorm::Infinity)?, 0.75 * f64::MAX);

    for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
        let empty = Matrix::<f64>::zeros(rows, cols)?;
        let complex = Matrix::<Complex<f64>>::zeros(rows, cols)?;
        for kind in kinds {
            assert_eq!(empty.view().norm(kind)?, 0.0);
            assert_eq!(complex.view().norm(kind)?, 0.0);
        }
    }
    for value in [f64::NAN, f64::INFINITY] {
        let invalid = Matrix::from_rows(&[[1.0, value], [3.0, 4.0]])?;
        let complex = Matrix::from_rows(&[[Complex::new(1.0, 0.0), Complex::new(0.0, value)]])?;
        for kind in kinds {
            assert_eq!(
                invalid.view().norm(kind).unwrap_err().kind(),
                SciOxideErrorKind::NonFinite
            );
            assert_eq!(
                complex.view().norm(kind).unwrap_err().kind(),
                SciOxideErrorKind::NonFinite
            );
        }
    }
    Ok(())
}

#[test]
fn matrix_norms_agree_with_operator_norms() -> Result<(), scioxide_core::SciOxideError> {
    use scioxide_linalg::{MatrixNorm, operator_norm};
    let (rows, cols) = (4, 3);
    let entry = |i: usize, j: usize| ((i * 7 + j * 3) % 11) as f64 / 3.0 - 1.7;
    let real = Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|k| entry(k / cols, k % cols))
            .collect(),
    )?;
    let complex = Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|k| Complex::new(entry(k / cols, k % cols), entry(k % cols, k / cols + 1)))
            .collect(),
    )?;
    let allowance = 8.0 * rows.max(cols) as f64 * f64::EPSILON;
    for kind in [MatrixNorm::One, MatrixNorm::Infinity, MatrixNorm::Frobenius] {
        for view in [real.view(), real.view().transpose()] {
            let (actual, reference) = (view.norm(kind)?, operator_norm(&view.try_finite()?, kind)?);
            assert!(
                (actual - reference).abs() <= allowance * reference,
                "{kind:?}"
            );
        }
        for view in [complex.view(), complex.view().adjoint()] {
            let (actual, reference) = (view.norm(kind)?, operator_norm(&view.try_finite()?, kind)?);
            assert!(
                (actual - reference).abs() <= allowance * reference,
                "{kind:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn frobenius_norm_of_empty_and_tall_views_takes_one_pass()
-> Result<(), scioxide_core::SciOxideError> {
    use scioxide_linalg::MatrixNorm;
    // An empty view of any height has norm zero without visiting its rows.
    let empty = MatrixView::<f64>::from_slice(usize::MAX, 0, &[])?;
    assert_eq!(empty.norm(MatrixNorm::Frobenius)?, 0.0);
    // A strided 3000 × 2 view with columns of ones and twos: ‖·‖_F = √(3000 · 5).
    let rows = 3000;
    let data: Vec<f64> = (0..rows * 3)
        .map(|k| match k % 3 {
            0 => 1.0,
            1 => 2.0,
            _ => f64::NAN,
        })
        .collect();
    let tall = MatrixView::from_slice_with_strides(rows, 2, &data, 3, 1)?;
    let expected = (5.0 * rows as f64).sqrt();
    let actual = tall.norm(MatrixNorm::Frobenius)?;
    assert!((actual - expected).abs() <= frobenius_bound(2, 2 * rows) * expected);
    let actual = tall.transpose().norm(MatrixNorm::Frobenius)?;
    assert!((actual - expected).abs() <= frobenius_bound(2, 2 * rows) * expected);
    Ok(())
}
