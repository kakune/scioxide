//! Accepted sparse values are finite; candidate publication is transactional.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{AdjointOperator, LinearOperator, SciOxideError, SciOxideErrorKind};
use scioxide_sparse::{
    CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseValueWorkspace,
};

#[test]
fn every_mutable_format_rejects_partial_and_nonfinite_candidates() {
    let mut builder = SparseBuilder::new(2, 2);
    builder
        .extend(&[0, 0, 1], &[0, 1, 1], &[2.0_f64, 1.0, 3.0])
        .unwrap();
    let mut coo = builder
        .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let mut csr = coo.to_csr().unwrap();
    let mut csc = coo.to_csc().unwrap();
    macro_rules! check {
        ($matrix:expr) => {{
            let matrix = &mut $matrix;
            let before = matrix.clone();
            let mut ws = matrix.workspace(()).unwrap();
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let result = matrix.update_values(
                    |old, next| {
                        next.copy_from_slice(old);
                        next[2] = invalid;
                        Ok::<_, SciOxideError>(())
                    },
                    &mut ws,
                );
                assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
                assert_eq!(*matrix, before);
            }
            let error = matrix
                .update_values(
                    |_, next| {
                        next[0] = 9.;
                        Err::<(), _>(SciOxideError::singular(Some(1)))
                    },
                    &mut ws,
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::Singular);
            assert_eq!(*matrix, before);
            assert_eq!(
                matrix
                    .update_values(
                        |_, next| {
                            next[0] = 9.;
                            Ok::<_, SciOxideError>(())
                        },
                        &mut ws
                    )
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::NonFinite
            );
            assert_eq!(*matrix, before);
            assert_eq!(
                matrix
                    .copy_values_from(&[1., f64::NAN, 2.])
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::NonFinite
            );
            assert_eq!(
                matrix.copy_values_from(&[1.]).unwrap_err().kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(
                matrix.set_value(99, 1.).unwrap_err().kind(),
                SciOxideErrorKind::OutOfBounds
            );
            assert_eq!(*matrix, before);
            let mut called = false;
            let mut empty = SparseValueWorkspace::new();
            assert_eq!(
                matrix
                    .update_values(
                        |_, _| {
                            called = true;
                            Ok::<_, SciOxideError>(())
                        },
                        &mut empty
                    )
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert!(!called);
            let mut calls = 0;
            let report = matrix
                .update_values(
                    |old, next| {
                        calls += 1;
                        for (out, &value) in next.iter_mut().zip(old) {
                            *out = 2. * value;
                        }
                        Ok::<_, SciOxideError>(42)
                    },
                    &mut ws,
                )
                .unwrap();
            assert_eq!((calls, report), (1, 42));
            let mut output = [0.; 2];
            matrix
                .view()
                .apply_into(&[1.; 2], &mut output, &mut [0.; 2])
                .unwrap();
            assert_eq!(output, [6., 6.]);
            matrix
                .view()
                .apply_adjoint_into(&[1.; 2], &mut output, &mut [0.; 2])
                .unwrap();
            assert_eq!(output, [4., 8.]);
        }};
    }
    check!(coo);
    check!(csr);
    check!(csc);
}

#[test]
fn empty_updates_and_unwinding_preserve_the_boundary() {
    let mut empty = CsrMatrix::<f32>::from_raw(0, 0, vec![0], vec![], vec![]).unwrap();
    empty
        .update_values(
            |old, next| {
                assert!(old.is_empty() && next.is_empty());
                Ok::<_, SciOxideError>(())
            },
            &mut empty.workspace(()).unwrap(),
        )
        .unwrap();
    let mut a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64]).unwrap();
    let mut ws = a.workspace(()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = a.update_values(
            |_, next| -> Result<(), SciOxideError> {
                next[0] = 7.;
                panic!("callback stopped")
            },
            &mut ws,
        );
    }));
    assert!(result.is_err());
    assert_eq!(a.values(), &[2.]);
    a.update_values(
        |_, next| {
            next[0] = 3.;
            Ok::<_, SciOxideError>(())
        },
        &mut ws,
    )
    .unwrap();
    assert_eq!(a.values(), &[3.]);
}
