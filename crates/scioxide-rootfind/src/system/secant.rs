//! Explicit `QR` factors of a secant Jacobian model.
//!
//! Powell's hybrid method and Broyden's method keep `B = QR` with an explicit
//! orthogonal `Q`, refactor it from every evaluated Jacobian, and apply rank-one
//! secant corrections by plane rotations. Rotations keep the factors accurate even
//! when a correction dwarfs the model, which forming `B + uvᵀ` densely would not.

use scioxide_core::{ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, Vector};

/// Explicit unpivoted `QR` factors of a freshly evaluated Jacobian.
pub(crate) fn factor_secant_into<'scope, 'capability, S: RealScalar + NumericScalar<Real = S>>(
    jacobian: &Matrix<S>,
    secant: &mut scioxide_linalg::QrUpdateDecomposition<S>,
    secant_workspace: &mut scioxide_linalg::LinalgWorkspace<'static, S>,
    policy: ExecutionPolicy,
    context: impl Into<scioxide_linalg::LinalgExecutionContext<'scope, 'capability>> + Copy,
) -> Result<(), SciOxideError> {
    scioxide_linalg::QrUpdateDecomposition::factor_into(
        jacobian,
        &secant_options(policy),
        secant,
        secant_workspace,
        context,
    )
}

/// Rank-one secant correction `QR ← QR + u vᵀ` by plane rotations, which keeps
/// the factors accurate when the correction dwarfs the model.
pub(crate) fn secant_update_into<'scope, 'capability, S: RealScalar + NumericScalar<Real = S>>(
    secant: &mut scioxide_linalg::QrUpdateDecomposition<S>,
    secant_workspace: &mut scioxide_linalg::LinalgWorkspace<'static, S>,
    left: &Vector<S>,
    right: &Vector<S>,
    policy: ExecutionPolicy,
    context: impl Into<scioxide_linalg::LinalgExecutionContext<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    let n = left.len();
    secant
        .modify_in_place(
            scioxide_linalg::QrModification::RankUpdate {
                u: scioxide_linalg::MatrixView::from_slice(n, 1, left.as_slice())?,
                v: scioxide_linalg::MatrixView::from_slice(n, 1, right.as_slice())?,
            },
            &secant_options(policy),
            secant_workspace,
            context,
        )
        .map(|_| ())
}

/// Rank diagnostics of the secant factors are not used; the exact policy
/// uses the triangular full-rank proof when possible.
fn secant_options<S: RealScalar>(policy: ExecutionPolicy) -> scioxide_linalg::QrUpdateOptions<S> {
    scioxide_linalg::QrUpdateOptions::new(scioxide_linalg::RankPolicy::Exact)
        .with_pivoting(scioxide_linalg::HouseholderPivoting::None)
        .with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(policy),
        )
}

/// Whether every diagonal entry of `R` exceeds `nε` times the largest diagonal
/// magnitude, the dimension-scaled rank policy applied to triangular factors.
pub(crate) fn is_full_rank<S: RealScalar + NumericScalar<Real = S>>(r: &Matrix<S>) -> bool {
    let n = r.rows();
    let diagonal = || (0..n).map(|i| r.as_slice()[i * n + i].abs());
    let largest = diagonal().fold(<S as scioxide_core::AlgebraicScalar>::ZERO, S::max);
    let threshold = S::try_from_usize_exact(n).unwrap_or(S::MAX) * S::EPSILON * largest;
    largest > <S as scioxide_core::AlgebraicScalar>::ZERO
        && diagonal().all(|value| value > threshold)
}
