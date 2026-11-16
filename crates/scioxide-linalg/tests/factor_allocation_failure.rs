//! Allocation failures leave published factor and decomposition results unchanged.

#![allow(clippy::unwrap_used)]

use scioxide_core::ExecutionPolicy;
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, HouseholderDecomposition, HouseholderOrientation,
    LinalgExecutionOptions, LuDecomposition, Matrix,
};

#[test]
fn prepared_ldl_changes_block_count_without_allocating() {
    use scioxide_linalg::{LdlDecomposition, LinalgWorkspace};
    let pair = Matrix::from_vec(2, 2, vec![0.0_f64, 2.0, 2.0, 0.0]).unwrap();
    let single = Matrix::from_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let malformed = Matrix::from_vec(2, 2, vec![4.0_f64, 1.0, 2.0, 3.0]).unwrap();
    let mut factor = scioxide_linalg::LdlDecomposition::factor(
        scioxide_linalg::MatrixView::from_slice_with_strides(
            2,
            2,
            &[<f64 as scioxide_core::AlgebraicScalar>::ZERO],
            0,
            0,
        )
        .unwrap(),
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let mut workspace = LinalgWorkspace::new();
    workspace
        .try_reserve(
            &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::factor(
                    2,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    for (matrix, count) in [(&pair, 1), (&single, 2), (&pair, 1)] {
        let result = LdlDecomposition::factor_into(
            matrix.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut factor,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        );
        result.unwrap();
        assert_eq!(factor.pivot_blocks().count(), count);
        assert_eq!(
            factor
                .inertia(scioxide_linalg::RankPolicy::Exact)
                .unwrap()
                .total(),
            2
        );
    }
    let mut cloned = factor.clone();
    let result = LdlDecomposition::factor_into(
        single.view(),
        scioxide_linalg::HermitianInput::Exact,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        &mut cloned,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    );
    result.unwrap();
    assert_eq!(cloned.pivot_blocks().count(), 2);
    let before = factor.clone();
    let result = LdlDecomposition::factor_into(
        malformed.view(),
        scioxide_linalg::HermitianInput::Exact,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        &mut factor,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    );
    assert!(result.is_err());
    assert_eq!(factor, before);
}

#[test]
fn factor_storage_reservation_is_exact_checked_and_allocation_free_to_use() {
    use scioxide_core::Capacity;
    let options = LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
    let context = scioxide_core::ExecutionContext::sequential;
    let square =
        Matrix::from_vec(3, 3, vec![4.0_f64, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 6.0]).unwrap();
    let wide = Matrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0]).unwrap();

    let mut lu = LuDecomposition::<f64>::empty();
    assert_eq!(lu.check(0).unwrap(), Capacity::Sufficient);
    assert_eq!(lu.check(3).unwrap(), Capacity::NeedsGrowth);
    lu.try_reserve(3).unwrap();
    assert_eq!(lu.check(3).unwrap(), Capacity::Sufficient);
    assert_eq!(lu.check(4).unwrap(), Capacity::NeedsGrowth);
    let bytes = LuDecomposition::<f64>::factor_storage_bytes(3).unwrap();
    assert_eq!(lu.retained_bytes().unwrap(), bytes);
    lu.try_reserve(2).unwrap();
    assert_eq!(lu.retained_bytes().unwrap(), bytes);
    let mut lu_work = LuDecomposition::<f64>::workspace(
        &scioxide_linalg::LuWorkspaceRequest::factor(3, &options),
    )
    .unwrap();
    LuDecomposition::factor_into(&square, &options, &mut lu, &mut lu_work, context()).unwrap();
    assert_eq!(lu.retained_bytes().unwrap(), bytes);

    let mut cholesky = CholeskyDecomposition::<f64>::empty();
    cholesky.try_reserve(3).unwrap();
    assert_eq!(cholesky.check(3).unwrap(), Capacity::Sufficient);
    let bytes = CholeskyDecomposition::<f64>::factor_storage_bytes(3).unwrap();
    assert_eq!(cholesky.retained_bytes().unwrap(), bytes);
    let mut cholesky_work = CholeskyDecomposition::<f64>::workspace(
        &scioxide_linalg::CholeskyWorkspaceRequest::factor(3, &options),
    )
    .unwrap();
    CholeskyDecomposition::factor_into(
        &square,
        scioxide_linalg::HermitianInput::Exact,
        CholeskyPivoting::None,
        &options,
        &mut cholesky,
        &mut cholesky_work,
        context(),
    )
    .unwrap();
    assert_eq!(cholesky.retained_bytes().unwrap(), bytes);

    for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
        let mut qr = HouseholderDecomposition::<f64>::empty();
        assert_eq!(qr.check(2, 3, orientation).unwrap(), Capacity::NeedsGrowth);
        qr.try_reserve(2, 3, orientation).unwrap();
        assert_eq!(qr.check(2, 3, orientation).unwrap(), Capacity::Sufficient);
        let bytes =
            HouseholderDecomposition::<f64>::factor_storage_bytes(2, 3, orientation).unwrap();
        assert_eq!(qr.retained_bytes().unwrap(), bytes);
        let mut qr_work = HouseholderDecomposition::<f64>::workspace(
            &scioxide_linalg::HouseholderWorkspaceRequest::factor(2, 3, orientation, &options),
        )
        .unwrap();
        HouseholderDecomposition::factor_into(
            &wide,
            orientation,
            scioxide_linalg::HouseholderPivoting::Column,
            &options,
            &mut qr,
            &mut qr_work,
            context(),
        )
        .unwrap();
        assert_eq!(qr.retained_bytes().unwrap(), bytes);
    }
}
