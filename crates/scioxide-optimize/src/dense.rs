//! Counted adapters to shared linear algebra.
use crate::context::SolveSession;
use scioxide_core::{EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;

/// A conservative Gershgorin sufficient test on an already symmetric matrix.
/// Positive sums use a 4*n*epsilon margin; inconclusive/range-hazard cases
/// fall back to the Cholesky/spectral path.
pub(crate) fn positive_gershgorin<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
) -> Result<bool, SciOxideError> {
    let n = matrix.rows();
    let margin = S::try_from_usize_exact(n)? * S::EPSILON * S::try_from_f64_lossy(4.)?;
    if margin >= S::HALF {
        return Ok(false);
    }
    for (i, row) in matrix.as_slice().chunks_exact(n.max(1)).enumerate() {
        let mut radius = <S as scioxide_core::AlgebraicScalar>::ZERO;
        for (j, &a) in row.iter().enumerate() {
            if i != j {
                radius += NumericScalar::abs(a);
            }
        }
        let bound = radius / (<S as scioxide_core::AlgebraicScalar>::ONE - margin);
        if !bound.is_finite() || !row[i].is_finite() || row[i] <= bound {
            return Ok(false);
        }
    }
    Ok(true)
}
/// Shared capability boundary for rank-revealing programming and certificate solves.
pub(crate) fn svd_solve_with_policy<S: NumericScalar>(
    matrix: &Matrix<S>,
    rhs: &scioxide_linalg::Vector<S>,
    policy: scioxide_linalg::RankPolicy<S::Real>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<scioxide_linalg::Vector<S>, SciOxideError> {
    let mut factor = scioxide_linalg::SvdDecomposition::empty();
    let mut workspace = scioxide_linalg::LinalgWorkspace::new();
    session.invoke(EvaluationRole::Factorization, || {
        scioxide_linalg::SvdDecomposition::compute_into(
            matrix,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut factor,
            &mut workspace,
            session.execution()?,
        )
    })?;
    session.invoke(EvaluationRole::LinearSolve, || {
        factor.solve(rhs, policy, <_ as scioxide_core::AlgebraicScalar>::ZERO)
    })
}

/// Count one shared dense operator application and keep injected capabilities.
pub(crate) fn matrix_vector_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: scioxide_linalg::MatrixView<'_, S>,
    x: &[S],
    output: &mut [S],
    scratch: &mut [S],
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.invoke(EvaluationRole::LinearOperator, || {
        matrix.mul_vec_into(
            scioxide_linalg::VectorView::from_slice(x),
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_linalg::VectorViewMut::from_slice(output),
            &mut scioxide_linalg::LinalgWorkspace::from_scalar_slice(scratch),
            session.execution()?,
        )
    })
}

pub(crate) fn positive_definite(outcome: Result<(), SciOxideError>) -> Result<bool, SciOxideError> {
    match outcome {
        Ok(()) => Ok(true),
        Err(error)
            if matches!(error.view(), scioxide_core::SciOxideErrorView::DomainError {
                argument,
                reason: scioxide_core::DomainReason::OutsideDomain,
                ..
            } if argument == scioxide_core::argument!("a")) =>
        {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}
