//! Direct-factor attempt outcomes, nested diagnostics and caller admission.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;
use scioxide_core::{
    AttemptOutcome, Budget, ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor,
    NumericalIssue, PartitionFn, SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::*;

struct FailingExecutor;
impl Executor for FailingExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        _: ExecutionContext<'_>,
        _: ExecutionPlan,
        _: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "injected numerical provider failure",
        ))
    }
}

fn outcome(workspace: &LinalgWorkspace<'_, f64>, expected: AttemptOutcome) {
    assert_eq!(workspace.last_attempt().unwrap().outcome(), expected);
}

macro_rules! contracts {
    ($name:ident, $factor:ty $(, $setting:expr)*) => {
        mod $name {
            use super::*;
            fn factor_into(
                a: &Matrix<f64>,
                options: &LinalgExecutionOptions,
                output: &mut $factor,
                workspace: &mut LinalgWorkspace<'_, f64>,
                context: ExecutionContext<'_>,
            ) -> Result<(), SciOxideError> {
                <$factor>::factor_into(a, $($setting,)* options, output, workspace, context)
            }

            #[test]
            fn factor_attempts_replace_previous_outcomes() {
                let options = LinalgExecutionOptions::new();
                let mut workspace = LinalgWorkspace::new();
                let mut factor = <$factor>::empty();
                let a = Matrix::from_vec(1, 1, vec![0.5_f64]).unwrap();
                factor_into(&a, &options, &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                let epoch = workspace.last_attempt().unwrap().epoch();
                factor_into(&a, &options, &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                assert_ne!(workspace.last_attempt().unwrap().epoch(), epoch);
                assert_eq!(factor_into(&a, &options.with_byte_limit(1), &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                assert!(workspace.last_attempt().unwrap().epoch().is_none());
                let invalid = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
                let _ = factor_into(&invalid, &options, &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                let n = 160;
                let mut large = Matrix::from_vec(n, n, vec![1.0; n * n]).unwrap();
                for i in 0..n { large.view_mut().set(i, i, n as f64).unwrap(); }
                let parallel = options.with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()));
                let _ = factor_into(&large, &parallel, &mut factor, &mut workspace, ExecutionContext::new(&FailingExecutor)).unwrap_err();
                outcome(&workspace, AttemptOutcome::Failed);
                // Failure did not publish the larger factor.
                assert_eq!(factor.solve(&Vector::from_vec(vec![1.0])).unwrap().as_slice().len(), 1);
            }

            #[test]
            fn solve_attempts_cover_success_rejection_failure_stop_and_empty() {
                let execution = LinalgExecutionOptions::new();
                let options = DirectSolveOptions::new(RankPolicy::Exact);
                let mut workspace = LinalgWorkspace::new();
                let mut factor = <$factor>::empty();
                let a = Matrix::from_vec(1, 1, vec![0.5_f64]).unwrap();
                factor_into(&a, &execution, &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap();
                let mut output = [7.0];
                let rhs = [1.0];
                let _ = factor.solve_into(VectorView::from_slice(&rhs), &options, VectorViewMut::from_slice(&mut output), &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                assert!((output[0] - 2.0).abs() < 1e-12);
                let limited = options.with_execution_options(execution.with_byte_limit(1));
                let _ = factor.solve_into(VectorView::from_slice(&rhs), &limited, VectorViewMut::from_slice(&mut output), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                output[0] = 7.0;
                let _ = factor.solve_into(VectorView::from_slice(&[f64::MAX]), &options, VectorViewMut::from_slice(&mut output), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::Failed);
                assert_eq!(output, [7.0]);
                let rhs = Matrix::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
                let mut output = Matrix::from_vec(1, 2, vec![7.0, 7.0]).unwrap();
                let _ = factor.solve_matrix_into(&rhs, &options, output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                let before = output.clone();
                let _ = factor.solve_matrix_into(&rhs, &limited, output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                assert_eq!(output, before);
                let huge = Matrix::from_vec(1, 2, vec![f64::MAX, 1.0]).unwrap();
                let _ = factor.solve_matrix_into(&huge, &options, output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::Failed);
                assert_eq!(output, before);
                let stopped = options.with_execution_options(execution.with_iteration_budget(Budget::ONE));
                let _ = factor.solve_matrix_into(&rhs, &stopped, output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::Stopped);
                assert_eq!(output, before);
                let rejected = DirectSolveOptions::new(RankPolicy::Threshold(scioxide_core::RankThreshold::new(1.0, 0.0).unwrap()));
                let report = factor.solve_matrix_into(&rhs, &rejected, output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap();
                assert_eq!(report.termination(), DirectSolveTermination::RankDeficient);
                outcome(&workspace, AttemptOutcome::Succeeded);
                assert_eq!(output, before);
                // Nested condition diagnostics leave the outer attempt visible.
                let gated = options.with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.1).unwrap());
                let _ = factor.solve_matrix_into(&rhs, &gated, output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                let empty = <$factor>::empty();
                let mut output = [];
                let _ = empty.solve_into(VectorView::from_slice(&[]), &options, VectorViewMut::from_slice(&mut output), &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                let mut matrix = Matrix::zeros(0, 2).unwrap();
                let _ = empty.solve_matrix_into(MatrixView::from_slice(0, 2, &[]).unwrap(), &options, matrix.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                let _ = empty.solve_into(VectorView::from_slice(&[]), &gated, VectorViewMut::from_slice(&mut []), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                let _ = empty.solve_matrix_into(MatrixView::from_slice(0, 2, &[]).unwrap(), &gated, matrix.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap_err();
                outcome(&workspace, AttemptOutcome::PreflightRejected);
            }

            #[test]
            fn condition_queries_obey_byte_limits_in_fresh_and_warmed_workspaces() {
                let options = LinalgExecutionOptions::new();
                let mut workspace = LinalgWorkspace::new();
                let mut factor = <$factor>::empty();
                factor_into(&Matrix::identity(2).unwrap(), &options, &mut factor, &mut workspace, ExecutionContext::sequential()).unwrap();
                for warmed in [false, true] {
                    let mut workspace = LinalgWorkspace::new();
                    if warmed { let _ = factor.condition_estimate_scoped(&options, &mut workspace, ExecutionContext::sequential()).unwrap(); }
                    assert_eq!(factor.condition_estimate_scoped(&options.with_byte_limit(1), &mut workspace, ExecutionContext::sequential()).unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
                    outcome(&workspace, AttemptOutcome::PreflightRejected);
                    let estimate = factor.condition_estimate_scoped(&options, &mut workspace, ExecutionContext::sequential()).unwrap();
                    assert!((estimate.reciprocal().value() - 1.0).abs() < 1e-12);
                    outcome(&workspace, AttemptOutcome::Succeeded);
                    let bytes = workspace.retained_bytes().unwrap();
                    let _ = factor.condition_estimate_scoped(&options.with_byte_limit(bytes), &mut workspace, ExecutionContext::sequential()).unwrap();
                    assert_eq!(factor.condition_estimate_scoped(&options.with_byte_limit(bytes - 1), &mut workspace, ExecutionContext::sequential()).unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
                }
            }
        }
    };
}
contracts!(lu, LuDecomposition<f64>);
contracts!(
    cholesky,
    CholeskyDecomposition<f64>,
    HermitianInput::Exact,
    CholeskyPivoting::None
);
contracts!(ldl, LdlDecomposition<f64>, HermitianInput::Exact);
contracts!(
    householder,
    HouseholderDecomposition<f64>,
    HouseholderOrientation::Qr,
    HouseholderPivoting::Column
);

macro_rules! inverse_limits {
    ($name:ident, $factor:ty, $request:ident $(, $setting:expr)*) => {
        #[test]
        fn $name() {
            let options = LinalgExecutionOptions::new();
            let a = Matrix::<f64>::identity(2).unwrap();
            let factor = <$factor>::factor_with_options(&a, $($setting,)* &options).unwrap();
            let requirements = <$factor>::workspace_requirements(&$request::solve_matrix(2, 2, &options)).unwrap();
            let bytes = requirements.required_bytes::<f64>().unwrap();
            for warmed in [false, true] {
                let mut workspace = LinalgWorkspace::new();
                if warmed { workspace.try_reserve(&requirements).unwrap(); }
                let mut output = Matrix::from_vec(2, 2, vec![7.0; 4]).unwrap();
                let before = output.clone();
                assert_eq!(factor.inverse_into(&options.with_byte_limit(bytes - 1), output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                assert_eq!(output, before);
                factor.inverse_into(&options.with_byte_limit(bytes), output.view_mut(), &mut workspace, ExecutionContext::sequential()).unwrap();
                outcome(&workspace, AttemptOutcome::Succeeded);
                for (actual, expected) in output.as_slice().iter().zip(a.as_slice()) { assert!((actual - expected).abs() < 1e-12); }
            }
        }
    };
}
inverse_limits!(lu_inverse_limits, LuDecomposition<f64>, LuWorkspaceRequest);
inverse_limits!(
    cholesky_inverse_limits,
    CholeskyDecomposition<f64>,
    CholeskyWorkspaceRequest,
    HermitianInput::Exact,
    CholeskyPivoting::None
);
inverse_limits!(
    ldl_inverse_limits,
    LdlDecomposition<f64>,
    LdlWorkspaceRequest,
    HermitianInput::Exact
);

macro_rules! structured_conditions {
    ($name:ident, $factor:ident, { $($setup:tt)* }) => {
        #[test]
        fn $name() {
            $($setup)*
            let options = LinalgExecutionOptions::new();
            let mut workspace = LinalgWorkspace::new();
            let estimate = $factor
                .condition_estimate_scoped(&options, &mut workspace, ExecutionContext::sequential())
                .unwrap();
            assert!(estimate.reciprocal().value() > 0.1);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
            assert!(attempt.counts().own().iterations() >= 2);
            // Every inverse action admits an iteration; one cannot complete Hager–Higham.
            let single = options.with_iteration_budget(Budget::ONE);
            let error = $factor
                .condition_estimate_scoped(&single, &mut workspace, ExecutionContext::sequential())
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
            outcome(&workspace, AttemptOutcome::Stopped);
            let mut fresh = LinalgWorkspace::new();
            let error = $factor
                .condition_estimate_scoped(&options.with_byte_limit(1), &mut fresh, ExecutionContext::sequential())
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
            outcome(&fresh, AttemptOutcome::PreflightRejected);
            // A gated solve estimates in a child scope of the solve attempt.
            let gated = DirectSolveOptions::new(RankPolicy::Exact)
                .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.1).unwrap());
            let rhs = Matrix::from_vec(3, 1, vec![1.0; 3]).unwrap();
            let mut output = Matrix::from_vec(3, 1, vec![7.0; 3]).unwrap();
            let report = $factor
                .solve_matrix_into(StructuredOperation::Normal, &rhs, &gated, output.view_mut(), &mut workspace, ExecutionContext::sequential())
                .unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
            assert!(attempt.counts().nested().iterations() >= 2);
            // The nested estimate inherits the solve's local allowance.
            let before = output.clone();
            let error = $factor
                .solve_matrix_into(
                    StructuredOperation::Normal,
                    &rhs,
                    &gated.with_execution_options(single),
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
            outcome(&workspace, AttemptOutcome::Stopped);
            assert_eq!(output, before);
        }
    };
}
structured_conditions!(banded_lu_condition_attempts, factor, {
    let a = BandedMatrix::new(3, 3, 0, 0, 1, vec![2.0, 3.0, 4.0]).unwrap();
    let factor = BandedLuDecomposition::factor(&a).unwrap();
});
structured_conditions!(banded_cholesky_condition_attempts, factor, {
    let h = HermitianBandMatrix::new(
        3,
        1,
        scioxide_core::Triangle::Lower,
        vec![4.0, 1.0, 4.0, 1.0, 4.0, 0.0],
    )
    .unwrap();
    let factor = BandedCholeskyDecomposition::factor(&h).unwrap();
});
structured_conditions!(tridiagonal_condition_attempts, factor, {
    let tri = TridiagonalMatrix::new(vec![1.0; 2], vec![4.0; 3], vec![1.0; 2]).unwrap();
    let factor = TridiagonalLuDecomposition::factor(&tri).unwrap();
});
structured_conditions!(triangular_condition_attempts, factor, {
    let upper = Matrix::from_vec(3, 3, vec![4.0, 1.0, 0.5, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0]).unwrap();
    let factor = TriangularPlan::from_dense(
        upper.view(),
        scioxide_core::Triangle::Upper,
        scioxide_core::DiagonalKind::Explicit,
    )
    .unwrap();
});
structured_conditions!(cyclic_condition_attempts, factor, {
    let tri = TridiagonalMatrix::new(vec![-1.0; 2], vec![4.0; 3], vec![-1.0; 2]).unwrap();
    let factor =
        CyclicTridiagonalPlan::new(&tri, -0.5, -0.25, &LinalgExecutionOptions::new()).unwrap();
});

macro_rules! svd_solve_attempts {
    ($name:ident, $factor:ty) => {
        #[test]
        fn $name() {
            let options = LinalgExecutionOptions::new();
            let a = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0]).unwrap();
            let factor = <$factor>::compute(&a).unwrap();
            let solve =
                <$factor>::workspace_requirements(&SvdWorkspaceRequest::solve(2, 2, &options))
                    .unwrap()
                    .required_bytes::<f64>()
                    .unwrap();
            let rhs = [2.0, 1.0];
            for warmed in [false, true] {
                let mut workspace = LinalgWorkspace::new();
                let mut x = [7.0; 2];
                if warmed {
                    factor
                        .solve_into(
                            &rhs[..],
                            RankPolicy::Exact,
                            0.0,
                            &options,
                            VectorViewMut::from_slice(&mut x),
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                    x = [7.0; 2];
                }
                let limited = options.with_byte_limit(solve - 1);
                let error = factor
                    .solve_into(
                        &rhs[..],
                        RankPolicy::Exact,
                        0.0,
                        &limited,
                        VectorViewMut::from_slice(&mut x),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
                outcome(&workspace, AttemptOutcome::PreflightRejected);
                assert_eq!(x, [7.0; 2]);
                factor
                    .solve_into(
                        &rhs[..],
                        RankPolicy::Exact,
                        0.0,
                        &options.with_byte_limit(solve),
                        VectorViewMut::from_slice(&mut x),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                let attempt = workspace.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
                assert_eq!(attempt.counts().own().iterations(), 1);
                assert!(x.iter().all(|&value| (value - 1.0).abs() < 1e-14));
            }
            let mut workspace = LinalgWorkspace::new();
            let mut x = [7.0; 2];
            let clocked = options.with_time_limit(core::time::Duration::from_secs(1));
            let error = factor
                .solve_into(
                    &rhs[..],
                    RankPolicy::Exact,
                    0.0,
                    &clocked,
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
            outcome(&workspace, AttemptOutcome::PreflightRejected);
            // Each pseudoinverse column admits one iteration of the local allowance.
            let mut inverse = Matrix::from_vec(2, 2, vec![7.0; 4]).unwrap();
            let before = inverse.clone();
            let error = factor
                .pseudoinverse_into(
                    RankPolicy::Exact,
                    &options.with_iteration_budget(Budget::ONE),
                    inverse.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
            outcome(&workspace, AttemptOutcome::Stopped);
            assert_eq!(inverse, before);
            factor
                .pseudoinverse_into(
                    RankPolicy::Exact,
                    &options,
                    inverse.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
            assert_eq!(attempt.counts().own().iterations(), 2);
            for (actual, expected) in inverse.as_slice().iter().zip([0.5, 0.0, 0.0, 1.0]) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    };
}
svd_solve_attempts!(full_svd_solve_attempts, SvdDecomposition<f64>);
svd_solve_attempts!(thin_svd_solve_attempts, ThinSvdDecomposition<f64>);

#[test]
fn ldl_inverse_honors_a_sequential_execution_policy() {
    let n = 160;
    let mut a = Matrix::from_vec(n, n, vec![1.0; n * n]).unwrap();
    for i in 0..n {
        a.view_mut().set(i, i, n as f64).unwrap();
    }
    let options = LinalgExecutionOptions::new();
    let factor = LdlDecomposition::factor(&a, HermitianInput::Exact).unwrap();
    let mut inverse = Matrix::zeros(n, n).unwrap();
    // The executor fails every dispatch, so only sequential execution succeeds.
    factor
        .inverse_into(
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            inverse.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::new(&FailingExecutor),
        )
        .unwrap();
    let product = a.view().mul_mat(&inverse).unwrap();
    for (index, &value) in product.as_slice().iter().enumerate() {
        let expected = if index % (n + 1) == 0 { 1.0 } else { 0.0 };
        assert!((value - expected).abs() < 1e-12);
    }
}

#[test]
fn minimum_norm_solves_record_attempt_outcomes() {
    let execution = LinalgExecutionOptions::new();
    let plan = |a: &Matrix<f64>, acceptance: &DirectSolveOptions<f64>| {
        HouseholderDecomposition::factor(a, HouseholderOrientation::Qr, HouseholderPivoting::Column)
            .unwrap()
            .into_minimum_norm(acceptance)
            .unwrap()
    };
    let plain = DirectSolveOptions::new(RankPolicy::Exact);
    let gated =
        plain.with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.1).unwrap());
    let a = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0]).unwrap();
    let mut workspace = LinalgWorkspace::new();
    let mut output = [7.0; 2];
    let _ = plan(&a, &gated)
        .solve_into(
            &[2.0, 1.0][..],
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    outcome(&workspace, AttemptOutcome::Succeeded);
    let limited = plain.with_execution_options(execution.with_byte_limit(1));
    let error = plan(&a, &limited)
        .solve_into(
            &[2.0, 1.0][..],
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    outcome(&workspace, AttemptOutcome::PreflightRejected);
    // An empty factor has no condition estimate; the gate rejects before the epoch.
    for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
        let empty = Matrix::<f64>::zeros(rows, cols).unwrap();
        let mut output = vec![0.0; cols];
        let error = plan(&empty, &gated)
            .solve_into(
                &vec![0.0; rows][..],
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
        outcome(&workspace, AttemptOutcome::PreflightRejected);
    }
}

#[test]
fn csd_rejects_nonunitary_input_before_the_epoch() {
    let x = Matrix::from_vec(2, 2, vec![3.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let mut output = CsdDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    let error = CsdDecomposition::factor_into(
        &x,
        1,
        1,
        &LinalgExecutionOptions::new(),
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    outcome(&workspace, AttemptOutcome::PreflightRejected);
    assert!(workspace.last_attempt().unwrap().epoch().is_none());
}

#[test]
fn householder_unitary_forms_record_attempts() {
    let options = LinalgExecutionOptions::new();
    let a = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]).unwrap();
    let factor = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let mut workspace = LinalgWorkspace::new();
    let mut q = Matrix::zeros(3, 3).unwrap();
    let mut applied = Matrix::zeros(3, 2).unwrap();
    let mut reconstructed = Matrix::zeros(3, 2).unwrap();
    for limited in [false, true] {
        let options = if limited {
            options.with_byte_limit(1)
        } else {
            options
        };
        let expected = if limited {
            AttemptOutcome::PreflightRejected
        } else {
            AttemptOutcome::Succeeded
        };
        let cx = ExecutionContext::sequential;
        let results = [
            factor.q_matrix_into(&options, q.view_mut(), &mut LinalgWorkspace::new(), cx()),
            factor.apply_q_into(
                &a,
                StructuredOperation::Adjoint,
                &options,
                applied.view_mut(),
                &mut LinalgWorkspace::new(),
                cx(),
            ),
            factor.reconstruct_into(&options, reconstructed.view_mut(), &mut workspace, cx()),
        ];
        for result in results {
            assert_eq!(result.is_ok(), !limited);
        }
        outcome(&workspace, expected);
    }
    for (actual, expected) in reconstructed.as_slice().iter().zip(a.as_slice()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    // A wrong output shape is rejected before the epoch begins.
    let error = factor
        .q_matrix_into(
            &options,
            applied.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    outcome(&workspace, AttemptOutcome::PreflightRejected);
}
