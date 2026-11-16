#![allow(clippy::unwrap_used)]

use scioxide_core::Triangle;
use scioxide_core::{Complex, NdView, NdViewMut, StridedLayout};
use scioxide_linalg::{
    BandedMatrixView, MatrixView, PackedKind, PackedMatrixView, PackedMatrixViewMut,
    StructuredOperation, VectorView, VectorViewMut,
};

#[test]
fn rectangular_band_transpose_and_padding_are_checked() {
    // A = [[1, 2, 0], [3, 4, 5]], kl = ku = 1, lda = 4.
    let data = [
        0.0_f64,
        1.0,
        3.0,
        f64::NAN,
        2.0,
        4.0,
        0.0,
        f64::NAN,
        5.0,
        0.0,
        0.0,
        f64::NAN,
    ];
    let band = BandedMatrixView::new(2, 3, 1, 1, 4, &data).unwrap();
    let x = [7.0, 11.0, 13.0];
    let mut y = [0.0, 0.0];
    let mut scratch = [0.0; 3];
    band.mul_vec_in_place(
        StructuredOperation::Normal,
        1.0,
        VectorView::from_slice(&x),
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut y),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        scioxide_linalg::LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(y, [29.0, 130.0]);
    let mut transpose = [0.0; 3];
    band.mul_vec_in_place(
        StructuredOperation::Transpose,
        1.0,
        VectorView::from_slice(&y),
        0.0,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut transpose),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        scioxide_linalg::LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(transpose, [419.0, 578.0, 650.0]);
}

#[test]
fn selected_hermitian_triangle_ignores_unused_nan() {
    let a = [
        Complex::new(2.0_f64, 0.0),
        Complex::new(f64::NAN, 0.0),
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 0.0),
    ];
    let matrix = MatrixView::from_slice(2, 2, &a).unwrap();
    let x = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
    let mut y = [Complex::new(0.0, 0.0); 2];
    let mut scratch = y;
    matrix
        .hermitian_mul_vec_in_place(
            Triangle::Lower,
            Complex::new(1.0, 0.0),
            VectorView::from_slice(&x),
            Complex::new(0.0, 0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut y),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            scioxide_linalg::LinalgExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(y, [Complex::new(4.0, -4.0), Complex::new(7.0, 2.0)]);
}

#[test]
fn nonfinite_hermitian_diagonal_is_nonfinite_input() {
    let a = [
        Complex::new(1.0_f64, f64::NAN),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let mut y = [Complex::new(0.0, 0.0); 2];
    let error = MatrixView::from_slice(2, 2, &a)
        .unwrap()
        .hermitian_mul_vec_in_place(
            Triangle::Lower,
            Complex::new(1.0, 0.0),
            VectorView::from_slice(&[Complex::new(1.0, 0.0); 2]),
            Complex::new(0.0, 0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut y),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_linalg::LinalgExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn packed_update_view_preserves_reflection_and_rejects_unselected_write() {
    let mut data = [
        Complex::new(1.0_f64, 0.0),
        Complex::new(2.0, 1.0),
        Complex::new(3.0, 0.0),
    ];
    let mut packed =
        PackedMatrixViewMut::new(2, Triangle::Lower, PackedKind::Hermitian, &mut data).unwrap();
    packed.set(1, 0, Complex::new(4.0, -2.0)).unwrap();
    assert_eq!(packed.get(0, 1).unwrap(), Complex::new(4.0, 2.0));
    assert!(packed.set(0, 1, Complex::new(0.0, 0.0)).is_err());
    assert!(packed.set(1, 1, Complex::new(2.0, 1.0)).is_err());
    let view = PackedMatrixView::new(2, Triangle::Lower, PackedKind::Hermitian, &data).unwrap();
    assert_eq!(view.get(1, 0).unwrap(), Complex::new(4.0, -2.0));
}

#[test]
fn rectangular_rank_one_and_packed_hermitian_updates_match_scalar_reference() {
    let mut dense = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut matrix = scioxide_linalg::MatrixViewMut::from_slice(2, 3, &mut dense).unwrap();
    let x = [2.0, -1.0];
    let y = [3.0, 4.0, -2.0];
    scioxide_linalg::MatrixViewMut::rank_one_update_in_place(
        0.5,
        VectorView::from_slice(&x),
        VectorView::from_slice(&y),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        matrix.reborrow(),
        &mut scioxide_linalg::LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(dense, [4.0, 6.0, 1.0, 2.5, 3.0, 7.0]);

    let mut packed_data = [
        Complex::new(1.0_f64, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(2.0, 0.0),
    ];
    let mut packed =
        PackedMatrixViewMut::new(2, Triangle::Lower, PackedKind::Hermitian, &mut packed_data)
            .unwrap();
    let v = [Complex::new(1.0, 1.0), Complex::new(2.0, -1.0)];
    let mut scratch = [Complex::new(0.0, 0.0); 3];
    scioxide_linalg::PackedMatrixViewMut::hermitian_rank_one_update_in_place(
        Complex::new(0.5, 0.0),
        VectorView::from_slice(&v),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        packed.reborrow(),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        scioxide_linalg::LinalgExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(packed.get(0, 0).unwrap(), Complex::new(2.0, 0.0));
    assert_eq!(packed.get(1, 1).unwrap(), Complex::new(4.5, 0.0));
    assert_eq!(packed.get(1, 0).unwrap(), Complex::new(0.5, -1.5));
}

#[test]
fn batch_slices_share_selected_triangle_semantics() {
    let matrices = [2.0_f64, 0.0, 1.0, 3.0, 4.0, 0.0, 2.0, 5.0];
    let inputs = [1.0_f64, 2.0, 3.0, 4.0];
    let mut outputs = [0.0_f64; 4];
    let a = NdView::new(&matrices, StridedLayout::new(&[2, 4], &[4, 1]).unwrap()).unwrap();
    let matrix_lanes = a.layout().axis(1).unwrap();
    let x = NdView::new(&inputs, StridedLayout::new(&[2, 2], &[2, 1]).unwrap()).unwrap();
    let input_lanes = x.layout().axis(1).unwrap();
    let output_lanes = NdViewMut::new(&mut outputs, StridedLayout::new(&[2, 2], &[2, 1]).unwrap())
        .unwrap()
        .layout()
        .axis(1)
        .unwrap();
    let mut scratch = [0.0_f64; 2];
    for batch in 0..2 {
        let matrix = matrix_lanes.lane(batch).unwrap();
        let input = input_lanes.lane(batch).unwrap();
        let output = output_lanes.lane(batch).unwrap();
        MatrixView::from_slice(
            2,
            2,
            &matrices[matrix.start()..matrix.start() + matrix.len()],
        )
        .unwrap()
        .symmetric_mul_vec_in_place(
            Triangle::Lower,
            1.0,
            VectorView::from_slice(&inputs[input.start()..input.start() + input.len()]),
            0.0,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut outputs[output.start()..output.start() + output.len()]),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            scioxide_linalg::LinalgExecutionContext::sequential(),
        )
        .unwrap();
    }
    assert_eq!(outputs, [4.0, 7.0, 20.0, 26.0]);
}

#[test]
fn rectangular_band_operators_cover_every_scalar_representation() {
    fn run<S: scioxide_core::NumericScalar>() {
        use scioxide_core::{AdjointOperator, AlgebraicScalar, LinearOperator, RealScalar};
        use scioxide_linalg::BandedMatrix;
        let r = |value| S::Real::try_from_f64_exact(value).unwrap();
        let c = |real, imaginary| S::from_components(r(real), r(imaginary));
        let matrix = BandedMatrix::new(
            2,
            3,
            1,
            1,
            4,
            vec![
                S::ZERO,
                c(1.0, 1.0),
                c(3.0, 0.0),
                c(99.0, 0.0),
                c(2.0, -1.0),
                c(4.0, 0.0),
                S::ZERO,
                c(99.0, 0.0),
                c(5.0, 2.0),
                S::ZERO,
                S::ZERO,
                c(99.0, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(
            (matrix.rows(), matrix.cols(), matrix.leading_dimension()),
            (2, 3, 4)
        );
        let view = matrix.view();
        assert_eq!(view.apply_workspace_len().unwrap(), 2);
        assert_eq!(view.apply_adjoint_workspace_len().unwrap(), 3);
        let mut normal = [S::ZERO; 2];
        let mut adjoint = [S::ZERO; 3];
        let mut scratch = [S::ZERO; 3];
        view.apply_into(
            &[c(1.0, 0.0), c(2.0, 0.0), c(-1.0, 0.0)],
            &mut normal,
            &mut scratch,
        )
        .unwrap();
        view.apply_adjoint_into(&[c(1.0, 1.0), c(2.0, -1.0)], &mut adjoint, &mut scratch)
            .unwrap();
        let expected = if S::IS_REAL {
            [c(7.0, 0.0), c(10.0, 0.0), c(10.0, 0.0)]
        } else {
            [c(8.0, -3.0), c(9.0, -1.0), c(8.0, -9.0)]
        };
        for (actual, expected) in normal
            .into_iter()
            .chain(adjoint)
            .zip([c(5.0, -1.0), c(6.0, -2.0)].into_iter().chain(expected))
        {
            assert!(
                (actual - expected).abs()
                    <= r(16.0) * S::Real::EPSILON * expected.abs().max(S::Real::ONE)
            );
        }
        let before = normal;
        assert!(
            view.apply_into(&[S::ONE; 3], &mut normal, &mut scratch[..1])
                .is_err()
        );
        assert_eq!(normal, before);
    }
    run::<f32>();
    run::<f64>();
    run::<Complex<f32>>();
    run::<Complex<f64>>();
}

#[test]
fn rectangular_band_layout_errors_and_empty_shapes_are_explicit() {
    use scioxide_core::{AdjointOperator, LinearOperator, SciOxideErrorView};
    for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
        let band = BandedMatrixView::<f64>::new(rows, cols, 0, 0, 0, &[]).unwrap();
        assert_eq!((band.rows(), band.cols()), (rows, cols));
        let mut output = vec![99.0; rows];
        band.apply_into(&vec![0.0; cols], &mut output, &mut vec![0.0; rows])
            .unwrap();
        assert_eq!(output, vec![0.0; rows]);
        let mut output = vec![99.0; cols];
        band.apply_adjoint_into(&vec![0.0; rows], &mut output, &mut vec![0.0; cols])
            .unwrap();
        assert_eq!(output, vec![0.0; cols]);
    }
    assert!(BandedMatrixView::<f64>::new(0, usize::MAX, 0, 0, 0, &[]).is_ok());
    for (rows, cols, lower, upper, leading) in [
        (2, 3, 2, 0, 3),
        (2, 3, 0, 3, 4),
        (2, 3, 1, 1, 2),
        (0, 3, 0, 0, 1),
    ] {
        assert!(BandedMatrixView::<f64>::new(rows, cols, lower, upper, leading, &[]).is_err());
    }
    let error = BandedMatrixView::<f64>::new(2, 3, 1, 1, 3, &[0.0; 8]).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    for (lower, upper, leading) in [(usize::MAX - 1, usize::MAX - 1, usize::MAX), (0, 0, 2)] {
        let error =
            BandedMatrixView::<f64>::new(usize::MAX, usize::MAX, lower, upper, leading, &[])
                .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::SizeOverflow { .. }
        ));
    }
    let band = BandedMatrixView::new(1, 2, 0, 0, 1, &[f64::NAN, 0.0]).unwrap();
    let mut output = [42.0];
    assert!(
        band.apply_into(&[1.0, 1.0], &mut output, &mut [0.0])
            .is_err()
    );
    assert_eq!(output, [42.0]);
    assert!(band.get(1, 0).is_err());
    assert!(band.get(0, 2).is_err());
}
