//! Dense eigenvalue problems: Hermitian, Hermitian-definite, general, Schur and QZ.
//!
//! Hermitian problems return real eigenvalues and optionally orthonormal
//! eigenvectors in the input scalar type. General problems return complex
//! eigenvalues and the requested left or right eigenvectors. Each problem has
//! its own result type, and eigenvalue-only results never form eigenvectors.
//! Inputs are scaled by their largest entry magnitude before reduction and the
//! eigenvalues are scaled back, so deflation does not depend on the units.
//! Tridiagonal inputs for Sturm bisection are scaled only by a power of two that
//! keeps the bounds finite, so their smallest entries keep their precision.

use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, RealScalar};

mod complex_schur;
mod general;
mod general_types;
mod generalized;
mod generalized_types;
mod hermitian;
mod hermitian_types;
pub(crate) mod hessenberg;
mod qz;
pub(crate) mod qz_kernel;
mod qz_reorder;
mod qz_swap;
mod qz_types;
mod real_schur;
mod schur;
mod schur_reorder;
mod schur_swap;
mod schur_types;
mod schur_vectors;
pub(crate) mod support;
mod swap;
mod types;

pub(crate) use general::right_vectors_from_schur;
pub use general_types::{
    GeneralEigenBalanceMetadata, GeneralEigenBalancing, GeneralEigenDecomposition,
    GeneralEigenOptions, GeneralEigenWorkspaceRequest, GeneralEigenvaluesBuffer,
    GeneralEigenvaluesResult, GeneralEigenvectorSelection,
};
pub use generalized_types::*;
pub use hermitian_types::{
    EigenIndexRange, EigenvalueAccuracy, HermitianEigenDecomposition,
    HermitianEigenFirstComponents, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    HermitianEigenvaluesBuffer, HermitianEigenvaluesResult,
};
pub(crate) use qz::{for_each_generalized_value, qz_pencil, qz_values};
pub(crate) use qz_reorder::qz_reorder;
pub use qz_types::*;
pub(crate) use schur::{for_each_schur_block, schur_kernel, schur_kernel_lengths};
pub(crate) use schur_swap::swap as swap_schur_blocks;
pub use schur_types::*;
pub use types::{EigenReport, EigenvalueOrder};

fn normalize_slice<S: NumericScalar>(values: &mut [S], scale: S::Real) {
    if scale == S::Real::ZERO || scale == S::Real::ONE {
        return;
    }
    // Multiplying by a reciprocal is cheaper, but for a subnormal `scale` the
    // reciprocal overflows; fall back to component-wise division then.
    let inverse = S::Real::ONE / scale;
    if inverse.is_finite() {
        let factor = S::from_real(inverse);
        values.iter_mut().for_each(|value| *value = *value * factor);
    } else {
        values
            .iter_mut()
            .for_each(|value| *value = crate::rotation::divide_real(*value, scale));
    }
}

// Absolute deflation tolerances have meaning in normalized matrix units,
// independent of the units in which the input was supplied.
fn hermitian_reduction_scale<R: RealScalar>(input_scale: R) -> R {
    if input_scale == R::ZERO {
        R::ONE
    } else {
        input_scale
    }
}

/// Default QR/QZ step budget of 30 steps per order, unless set explicitly.
fn qr_iteration_budget(
    explicit: Option<scioxide_core::Budget>,
    order: usize,
) -> Result<usize, scioxide_core::SciOxideError> {
    explicit.map_or_else(
        || {
            scioxide_core::backend::checked_mul(
                30,
                order.max(1),
                scioxide_core::quantity!("eigen_iteration_budget"),
            )
        },
        |budget| Ok(budget.get()),
    )
}

pub(crate) fn into_complex<S: NumericScalar>(value: S) -> scioxide_core::Complex<S::Real> {
    scioxide_core::Complex::new(value.real_component(), value.imaginary_component())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn automatic_budgets_grow_with_order_and_preserve_overrides() {
        use scioxide_core::Budget;
        for n in [0_usize, 1, 32, 8192, 32768] {
            let order = n.max(1);
            for explicit in [
                GeneralEigenOptions::<f64>::new().iteration_budget(),
                SchurOptions::<f64>::new().iteration_budget(),
            ] {
                assert_eq!(explicit, None);
                assert_eq!(qr_iteration_budget(explicit, n).unwrap(), 30 * order);
            }
            let options = HermitianEigenOptions::<f64>::default();
            assert_eq!(options.iteration_budget(), None);
            assert_eq!(
                options.resolved_iteration_budget(n, false).unwrap(),
                30 * order
            );
            let levels = match n {
                8192 => 10,
                32768 => 12,
                _ => 1,
            };
            assert_eq!(
                options.resolved_iteration_budget(n, true).unwrap(),
                30 * order * levels
            );
            assert_eq!(
                options
                    .with_eigenvalue_accuracy(EigenvalueAccuracy::Relative)
                    .resolved_iteration_budget(n, false)
                    .unwrap(),
                2182 * order
            );
            assert_eq!(
                HermitianEigenOptions::<f32>::default()
                    .with_eigenvalue_accuracy(EigenvalueAccuracy::Relative)
                    .resolved_iteration_budget(n, true)
                    .unwrap(),
                332 * order
            );
            assert_eq!(
                options
                    .with_execution_options(
                        (options)
                            .execution_options()
                            .with_iteration_budget(Budget::ONE)
                    )
                    .resolved_iteration_budget(n, true)
                    .unwrap(),
                1
            );
        }
        assert!(qr_iteration_budget(None, usize::MAX).is_err());
        assert!(
            HermitianEigenOptions::<f64>::default()
                .resolved_iteration_budget(usize::MAX, true)
                .is_err()
        );
        assert!(
            HermitianEigenOptions::<f64>::default()
                .with_eigenvalue_accuracy(EigenvalueAccuracy::Relative)
                .resolved_iteration_budget(usize::MAX, false)
                .is_err()
        );
        assert_eq!(
            qr_iteration_budget(Some(Budget::ONE), usize::MAX).unwrap(),
            1
        );
    }

    #[test]
    fn tridiagonal_solve_context_records_nested_work_bounded_by_options() {
        use scioxide_core::{
            Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity,
            SolveState,
        };
        let identity = SolveIdentity::new(
            ProblemToken::new(683),
            scioxide_core::algorithm_id!("lame", "spectral", 1),
            0,
            0,
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let diagonal = [1.0_f64, 2.0, 3.0, 4.0];
        let off = [0.3, 0.4, 0.5];
        let options = HermitianEigenOptions::default();
        let expected = HermitianEigenDecomposition::tridiagonal(&diagonal, &off).unwrap();
        let mut actual = HermitianEigenDecomposition::empty();
        let mut workspace = crate::LinalgWorkspace::new();
        // The parent's own one-iteration allowance does not bound the child epoch.
        HermitianEigenDecomposition::tridiagonal_into(
            &diagonal,
            &off,
            &options,
            &mut actual,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(actual.eigenvalues(), expected.eigenvalues());
        assert_eq!(
            actual.eigenvectors().as_slice(),
            expected.eigenvectors().as_slice()
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Succeeded
        );
        assert_eq!(context.counts().own().iterations(), 0);
        let nested = context.counts().nested().iterations();
        assert!(nested > 1);
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .own()
                .iterations(),
            nested
        );
        let before = actual.eigenvalues().to_vec();
        assert!(
            HermitianEigenDecomposition::tridiagonal_into(
                &diagonal,
                &off,
                &options.with_execution_options(
                    (options)
                        .execution_options()
                        .with_iteration_budget(Budget::ONE)
                ),
                &mut actual,
                &mut workspace,
                &mut context,
            )
            .is_err()
        );
        assert_eq!(actual.eigenvalues(), before.as_slice());
        assert_eq!(context.counts().nested().iterations(), nested + 1);
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(context.counts().total().unwrap().iterations(), nested + 1);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Stopped
        );
    }

    #[test]
    fn hermitian_scale_policy_normalizes_every_nonzero_input() {
        assert_eq!(hermitian_reduction_scale(0.0_f64), 1.0);
        assert_eq!(hermitian_reduction_scale(256.0_f64), 256.0);
        assert_eq!(hermitian_reduction_scale(1.0e-150_f64), 1.0e-150);
        assert_eq!(hermitian_reduction_scale(1.0e150_f64), 1.0e150);
    }

    #[test]
    fn normalization_divides_safely_when_the_reciprocal_overflows() {
        let smallest_subnormal = f64::from_bits(1);
        let mut values = [smallest_subnormal, -smallest_subnormal];
        normalize_slice(&mut values, smallest_subnormal);
        assert_eq!(values, [1.0, -1.0]);
    }
}
