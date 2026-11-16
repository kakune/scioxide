//! Public contracts for allocation-free dense Level-2 operations.

#![allow(clippy::panic)]

use scioxide_core::Complex;
use scioxide_linalg::{Matrix, MatrixView, MatrixViewMut, Vector, VectorView, VectorViewMut};

fn matrix_f64(rows: usize, cols: usize) -> Matrix<f64> {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|index| (index as f64 - 7.0) / 11.0)
            .collect(),
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn strided_unaligned_views_cover_tails_and_preserve_padding() {
    let rows = 3;
    let cols = 37;
    let mut matrix_storage = vec![-99.0_f32; 1 + rows * 41];
    (0..rows).for_each(|row| {
        (0..cols).for_each(|col| matrix_storage[1 + row * 41 + col] = (row + col) as f32 / 7.0)
    });
    let matrix = MatrixView::from_slice_with_strides(rows, cols, &matrix_storage[1..], 41, 1)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut input_storage = vec![-88.0_f32; 1 + cols * 2];
    (0..cols).for_each(|col| input_storage[1 + col * 2] = (col + 1) as f32 / 13.0);
    let input = VectorView::from_slice_with_stride(&input_storage[1..], cols, 2)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut output_storage = [77.0_f32; 7];
    let output = VectorViewMut::from_slice_with_stride(&mut output_storage[1..], rows, 2)
        .unwrap_or_else(|error| panic!("{error}"));

    matrix
        .mul_vec_into(
            input,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            output,
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));

    (0..rows).for_each(|row| {
        let expected = (0..cols)
            .map(|col| matrix_storage[1 + row * 41 + col] * input_storage[1 + col * 2])
            .sum::<f32>();
        assert!((output_storage[1 + row * 2] - expected).abs() <= 2.0e-4);
    });
    assert_eq!(output_storage[0], 77.0);
    assert_eq!(output_storage[2], 77.0);
    assert_eq!(output_storage[4], 77.0);
    assert_eq!(output_storage[6], 77.0);
}

#[test]
fn rectangular_complex_adjoint_fuses_conjugation() {
    let matrix = Matrix::from_vec(
        2,
        3,
        vec![
            Complex::new(1.0_f64, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(-2.0, 4.0),
            Complex::new(0.5, -3.0),
            Complex::new(2.0, 1.0),
            Complex::new(5.0, -2.0),
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let input = Vector::from_vec(vec![Complex::new(2.0, -1.0), Complex::new(-3.0, 0.5)]);
    let mut transpose = Vector::zeros(3).unwrap_or_else(|error| panic!("{error}"));
    let mut adjoint = Vector::zeros(3).unwrap_or_else(|error| panic!("{error}"));

    matrix
        .view()
        .transpose()
        .mul_vec_into(
            input.view(),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            transpose.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    matrix
        .view()
        .adjoint()
        .mul_vec_into(
            input.view(),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            adjoint.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));

    let expected_adjoint = (0..3)
        .map(|col| {
            (0..2)
                .map(|row| matrix.as_slice()[row * 3 + col].conj() * input.as_slice()[row])
                .sum::<Complex<f64>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(adjoint.as_slice(), expected_adjoint);
    assert_ne!(transpose, adjoint);

    let mut copied = Matrix::zeros(3, 2).unwrap_or_else(|error| panic!("{error}"));
    matrix
        .view()
        .adjoint()
        .copy_into(
            scioxide_linalg::CopyPolicy::RequireFinite,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (copied).view_mut(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        copied.as_slice(),
        &[
            Complex::new(1.0, -2.0),
            Complex::new(0.5, 3.0),
            Complex::new(3.0, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(-2.0, -4.0),
            Complex::new(5.0, 2.0),
        ]
    );
}

#[test]
fn transpose_into_supports_strided_rectangular_views() {
    let source_storage = [1.0_f64, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0];
    let source = MatrixView::from_slice_with_strides(2, 3, &source_storage, 4, 1)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut output_storage = [77.0_f64; 12];
    let output = MatrixViewMut::from_slice_with_strides(3, 2, &mut output_storage, 4, 2)
        .unwrap_or_else(|error| panic!("{error}"));

    source
        .transpose()
        .copy_into(
            scioxide_linalg::CopyPolicy::RequireFinite,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            output,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));

    assert_eq!(
        output_storage,
        [
            1.0, 77.0, 4.0, 77.0, 2.0, 77.0, 5.0, 77.0, 3.0, 77.0, 6.0, 77.0
        ]
    );
}

#[test]
fn empty_and_degenerate_shapes_are_mathematical_values() {
    let no_rows = Matrix::<f64>::zeros(0, 3).unwrap_or_else(|error| panic!("{error}"));
    let empty_input = Vector::from_vec(Vec::new());
    let mut transposed = Vector::from_vec(vec![9.0; 3]);
    no_rows
        .view()
        .transpose()
        .mul_vec_into(
            empty_input.view(),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            transposed.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(transposed.as_slice(), &[0.0; 3]);

    let no_columns = Matrix::<f64>::zeros(3, 0).unwrap_or_else(|error| panic!("{error}"));
    let mut ordinary = Vector::from_vec(vec![9.0; 3]);
    no_columns
        .view()
        .mul_vec_into(
            &empty_input,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ordinary.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(ordinary.as_slice(), &[0.0; 3]);

    let transposed_empty = no_rows
        .view()
        .transpose()
        .try_to_matrix()
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!((transposed_empty.rows(), transposed_empty.cols()), (3, 0));
}

#[test]
fn every_dimension_error_precedes_output_mutation() {
    let matrix = matrix_f64(2, 3);
    let wrong_input = Vector::from_vec(vec![1.0, 2.0]);
    let transposed_input = Vector::from_vec(vec![1.0, 2.0]);
    let mut ordinary = Vector::from_vec(vec![7.0, 8.0]);
    let mut wrong_transposed = Vector::from_vec(vec![7.0, 8.0]);
    let mut wrong_transpose =
        Matrix::from_vec(2, 3, vec![7.0; 6]).unwrap_or_else(|error| panic!("{error}"));

    assert!(matches!(
        (matrix.view().mul_vec_into(
            &wrong_input,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ordinary.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(matches!(
        (matrix.view().transpose().mul_vec_into(
            transposed_input.view(),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            wrong_transposed.view_mut(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(matches!(
        (matrix.view().adjoint().copy_into(
            scioxide_linalg::CopyPolicy::RequireFinite,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (wrong_transpose).view_mut(),
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(ordinary.as_slice(), &[7.0, 8.0]);
    assert_eq!(wrong_transposed.as_slice(), &[7.0, 8.0]);
    assert_eq!(wrong_transpose.as_slice(), &[7.0; 6]);
}
