#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Complex32, Complex64, ExecutionContext, NumericScalar, RealScalar, SequentialExecutor,
};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::{
    BandedMatrixView, LinalgExecutionContext, MatrixSide, MatrixView, MatrixViewMut, PackedKind,
    PackedMatrixView, StructuredOperation, TriangularPlan, VectorView, VectorViewMut,
};

fn scalar<S: NumericScalar>(value: f64) -> S {
    S::from_real(S::Real::try_from_f64_exact(value).unwrap())
}

fn close<S: NumericScalar>(actual: S, expected: S) {
    let tolerance = if S::Real::MANTISSA_DIGITS <= 24 {
        2e-5
    } else {
        2e-12
    };
    let scale = 1.0 + expected.abs().into_f64();
    assert!((actual - expected).abs().into_f64() <= tolerance * scale);
}

fn exercise<S: NumericScalar>() {
    let executor = SequentialExecutor;
    let context = LinalgExecutionContext::new(ExecutionContext::new(&executor));

    let band_data: [S; 4] = [scalar(0.0), scalar(1.0), scalar(2.0), scalar(3.0)];
    let band = BandedMatrixView::new(2, 2, 0, 1, 2, &band_data).unwrap();
    let x: [S; 2] = [scalar(4.0), scalar(5.0)];
    let mut y: [S; 2] = [scalar(0.0); 2];
    let mut scratch: [S; 2] = [scalar(0.0); 2];
    band.mul_vec_in_place(
        StructuredOperation::Normal,
        scalar(1.0),
        VectorView::from_slice(&x),
        scalar(0.0),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        VectorViewMut::from_slice(&mut y),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
        context,
    )
    .unwrap();
    close(y[0], scalar(14.0));
    close(y[1], scalar(15.0));

    let dense_data: [S; 4] = [scalar(2.0), scalar(999.0), scalar(1.0), scalar(3.0)];
    let dense = MatrixView::from_slice(2, 2, &dense_data).unwrap();
    dense
        .hermitian_mul_vec_in_place(
            Triangle::Lower,
            scalar(1.0),
            VectorView::from_slice(&x),
            scalar(0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut y),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
    close(y[0], scalar(13.0));
    close(y[1], scalar(19.0));

    let a: [S; 2] = [scalar(1.0), scalar(2.0)];
    let b: [S; 2] = [scalar(3.0), scalar(4.0)];
    let mut updated: [S; 4] = [scalar(0.0); 4];
    let mut gemm_workspace = scioxide_linalg::LinalgWorkspace::new();
    let mut matrix_scratch: [S; 4] = [scalar(0.0); 4];
    scioxide_linalg::MatrixViewMut::hermitian_rank_two_update_in_place(
        Triangle::Lower,
        scalar(1.0),
        VectorView::from_slice(&a),
        VectorView::from_slice(&b),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut updated).unwrap(),
        &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut matrix_scratch),
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    close(updated[0], scalar(6.0));
    close(updated[1], scalar(0.0));
    close(updated[2], scalar(10.0));
    close(updated[3], scalar(16.0));

    let general: [S; 4] = [scalar(1.0), scalar(2.0), scalar(3.0), scalar(4.0)];
    let general = MatrixView::from_slice(2, 2, &general).unwrap();
    let mut product: [S; 4] = [scalar(0.0); 4];
    dense
        .symmetric_mul_mat_in_place(
            general,
            MatrixSide::Left,
            Triangle::Lower,
            scalar(1.0),
            scalar(0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            MatrixViewMut::from_slice(2, 2, &mut product).unwrap(),
            &mut gemm_workspace,
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
    for (actual, expected) in product.into_iter().zip([5.0, 8.0, 10.0, 14.0]) {
        close(actual, scalar(expected));
    }

    let a_view = MatrixView::from_slice(2, 1, &a).unwrap();
    let b_view = MatrixView::from_slice(2, 1, &b).unwrap();
    let mut rank_two_k: [S; 4] = [scalar(0.0); 4];
    scioxide_linalg::MatrixViewMut::hermitian_rank_two_k_in_place(
        a_view,
        b_view,
        Triangle::Lower,
        StructuredOperation::Normal,
        scalar(1.0),
        scalar(0.0),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice(2, 2, &mut rank_two_k).unwrap(),
        &mut gemm_workspace,
        LinalgExecutionContext::sequential(),
    )
    .unwrap();
    close(rank_two_k[0], scalar(6.0));
    close(rank_two_k[2], scalar(10.0));
    close(rank_two_k[3], scalar(16.0));

    let packed_data: [S; 3] = [scalar(2.0), scalar(1.0), scalar(3.0)];
    let packed =
        PackedMatrixView::new(2, Triangle::Lower, PackedKind::Triangular, &packed_data).unwrap();
    let triangular = TriangularPlan::from_packed(packed, DiagonalKind::Explicit).unwrap();
    let mut applied: [S; 2] = [scalar(0.0); 2];
    packed
        .mul_vec_in_place(
            StructuredOperation::Normal,
            scalar(1.0),
            VectorView::from_slice(&x),
            scalar(0.0),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            VectorViewMut::from_slice(&mut applied),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            LinalgExecutionContext::sequential(),
        )
        .unwrap();
    close(applied[0], scalar(8.0));
    close(applied[1], scalar(19.0));
    let mut recovered: [S; 2] = [scalar(0.0); 2];
    triangular
        .solve_into(
            StructuredOperation::Normal,
            VectorView::from_slice(&applied),
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice(&mut recovered),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(&mut scratch),
            LinalgExecutionContext::sequential(),
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .unwrap();
    close(recovered[0], x[0]);
    close(recovered[1], x[1]);
}

#[test]
fn all_four_scalar_domains_share_checked_blas_semantics() {
    exercise::<f32>();
    exercise::<f64>();
    exercise::<Complex32>();
    exercise::<Complex64>();
}
