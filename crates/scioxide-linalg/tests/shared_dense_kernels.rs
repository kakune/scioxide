//! Prepared LDL storage, Hermitian projection, and original-system residuals.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Capacity, Complex, ExecutionContext, SciOxideErrorView, Tolerance};
use scioxide_linalg::{
    HermitianInput, LdlDecomposition, LdlWorkspaceRequest, LinalgExecutionOptions, LinalgWorkspace,
    Matrix,
};

#[test]
fn ldl_reservation_preserves_values_and_admits_first_factor_without_allocation() {
    let mut factor = LdlDecomposition::<f64>::empty();
    assert_eq!(factor.check(3).unwrap(), Capacity::NeedsGrowth);
    factor.try_reserve(3).unwrap();
    assert_eq!(factor.check(3).unwrap(), Capacity::Sufficient);
    assert_eq!(factor.order(), 0);
    assert_eq!(
        factor.retained_bytes().unwrap(),
        LdlDecomposition::<f64>::factor_storage_bytes(3).unwrap()
    );
    let a = Matrix::from_vec(3, 3, vec![2.0, 1.0, 0.0, 1.0, -2.0, 1.0, 0.0, 1.0, 3.0]).unwrap();
    let mut work = LinalgWorkspace::new();
    work.try_reserve(
        &LdlDecomposition::<f64>::workspace_requirements(&LdlWorkspaceRequest::factor(
            3,
            &LinalgExecutionOptions::new(),
        ))
        .unwrap(),
    )
    .unwrap();
    LdlDecomposition::factor_into(
        a.view(),
        HermitianInput::Exact,
        &LinalgExecutionOptions::new(),
        &mut factor,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let before = factor.clone();
    factor.try_reserve(7).unwrap();
    assert_eq!(factor, before);
    assert!(factor.try_reserve(usize::MAX).is_err());
    assert_eq!(factor, before);
}

#[test]
fn dominance_is_only_a_sufficient_positive_definiteness_test() {
    let matrix = |values| Matrix::from_vec(2, 2, values).unwrap();
    assert!(
        matrix(vec![2.0_f64, 0.5, 0.5, 1.0])
            .view()
            .is_positive_diagonally_dominant()
            .unwrap()
    );
    assert!(
        !matrix(vec![1.0, 2.0, 2.0, 5.0])
            .view()
            .is_positive_diagonally_dominant()
            .unwrap()
    );
    assert!(
        !matrix(vec![-1.0, 0.0, 0.0, 1.0])
            .view()
            .is_positive_diagonally_dominant()
            .unwrap()
    );
    assert!(
        matrix(vec![1.0, 0.0, 1.0, 1.0])
            .view()
            .is_positive_diagonally_dominant()
            .is_err()
    );
    assert!(
        matrix(vec![1.0, f64::INFINITY, f64::INFINITY, 1.0])
            .view()
            .is_positive_diagonally_dominant()
            .is_err()
    );
    assert!(
        Matrix::<f64>::empty()
            .view()
            .is_positive_diagonally_dominant()
            .unwrap()
    );
    // A NaN is non-finite whichever triangle holds it.
    for values in [vec![1.0, 0.0, f64::NAN, 1.0], vec![1.0, f64::NAN, 0.0, 1.0]] {
        assert_eq!(
            matrix(values)
                .view()
                .is_positive_diagonally_dominant()
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }
}

#[test]
fn projection_is_range_safe_complex_and_transactional() {
    let mut a = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(1.0, 2.0),
            Complex::new(f64::MAX, 3.0),
            Complex::new(f64::MAX, 1.0),
            Complex::new(4.0, -2.0),
        ],
    )
    .unwrap();
    a.project_hermitian_in_place().unwrap();
    assert_eq!(
        a.as_slice(),
        &[
            Complex::new(1.0, 0.0),
            Complex::new(f64::MAX, 1.0),
            Complex::new(f64::MAX, -1.0),
            Complex::new(4.0, 0.0)
        ]
    );
    let mut bad = Matrix::from_vec(2, 2, vec![1.0, 2.0, f64::INFINITY, 3.0]).unwrap();
    let before = bad.clone();
    assert!(bad.project_hermitian_in_place().is_err());
    assert_eq!(bad, before);
    let mut empty = Matrix::<f64>::empty();
    empty.project_hermitian_in_place().unwrap();
}

#[test]
fn residuals_handle_extended_envelopes_zero_rows_and_bad_solutions() {
    let a = Matrix::from_vec(1, 2, vec![f64::MAX, f64::MAX]).unwrap();
    let tolerance = Tolerance::new(0.0, 1e-12).unwrap();
    assert_eq!(
        a.view()
            .verify_solution(&[0.0], &[2.0, -2.0], tolerance)
            .unwrap(),
        0.0
    );
    let a = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(
        a.view()
            .verify_solution(&[4.0, 0.0], &[2.0, 7.0], tolerance)
            .is_ok()
    );
    let error = a
        .view()
        .verify_solution(&[4.0, 1.0], &[2.0, 7.0], tolerance)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability { .. }
    ));
    assert!(
        a.view()
            .verify_solution(&[4.0, 0.0], &[f64::NAN, 7.0], tolerance)
            .is_err()
    );
    assert!(
        a.view()
            .verify_solution(&[4.0], &[2.0, 7.0], tolerance)
            .is_err()
    );
    assert_eq!(
        Matrix::<f64>::empty()
            .view()
            .verify_solution(&[], &[], tolerance)
            .unwrap(),
        0.0
    );
}

#[test]
fn normwise_checks_and_transactional_residuals_keep_zero_equations_distinct() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let tolerance = Tolerance::new(0.0, 1e-8).unwrap();
    let x = [1.0, 1e-12];
    assert!(
        a.view()
            .verify_solution(&[1.0, 0.0], &x, tolerance)
            .is_err()
    );
    assert!(
        a.view()
            .verify_solution_normwise(&[1.0, 0.0], &x, tolerance)
            .unwrap()
            <= 1e-12
    );
    let mut residual = [99.0; 2];
    let mut scratch = [0.0; 2];
    a.view()
        .residual_into(&[1.0, 0.0], &x, &mut residual, &mut scratch)
        .unwrap();
    assert_eq!(residual, [0.0, -1e-12]);
    residual = [99.0; 2];
    assert!(
        a.view()
            .residual_into(&[1.0, f64::NAN], &x, &mut residual, &mut scratch)
            .is_err()
    );
    assert_eq!(residual, [99.0; 2]);
}
