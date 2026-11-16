//! Shared invariant/deflating-subspace admission contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::SciOxideErrorKind;
use scioxide_linalg::{GeneralizedSchurDecomposition, Matrix, SchurDecomposition};

#[test]
fn subspace_accessors_share_errors_and_accept_whole_blocks() {
    let a = Matrix::from_vec(3, 3, vec![1.0_f64, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0]).unwrap();
    let b = Matrix::<f64>::identity(3).unwrap();
    let schur = SchurDecomposition::decompose(&a).unwrap();
    let qz = GeneralizedSchurDecomposition::decompose(&a, &b).unwrap();
    for dim in [1, 4] {
        let expected = schur.invariant_subspace(dim).unwrap_err();
        let reason = if dim == 1 {
            scioxide_core::InvalidInputReason::Inconsistent
        } else {
            scioxide_core::InvalidInputReason::OutOfRange
        };
        assert_eq!(
            invalid_contract(&expected),
            (
                SciOxideErrorKind::InvalidInput,
                Some((scioxide_core::argument!("dimension"), reason))
            )
        );
        assert_eq!(
            invalid_contract(&qz.left_deflating_subspace(dim).unwrap_err()),
            invalid_contract(&expected)
        );
        assert_eq!(
            invalid_contract(&qz.right_deflating_subspace(dim).unwrap_err()),
            invalid_contract(&expected)
        );
    }
    for dim in [0, 2, 3] {
        assert_eq!(schur.invariant_subspace(dim).unwrap().cols(), dim);
        assert_eq!(qz.left_deflating_subspace(dim).unwrap().cols(), dim);
        assert_eq!(qz.right_deflating_subspace(dim).unwrap().cols(), dim);
    }
}

fn invalid_contract(
    error: &scioxide_core::SciOxideError,
) -> (
    SciOxideErrorKind,
    Option<(scioxide_core::Argument, scioxide_core::InvalidInputReason)>,
) {
    let detail = match error.view() {
        scioxide_core::SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    };
    (error.kind(), detail)
}
