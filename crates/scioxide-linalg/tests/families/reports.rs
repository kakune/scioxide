//! Public report ownership, publication boundaries, and work diagnostics.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Budget, CancellationToken, ExecutionContext,
    SciOxideErrorView,
};
use scioxide_linalg::*;

fn owned<T: Send + Sync + 'static>(_: &T) {}

#[test]
fn partial_buffers_expose_only_published_columns_with_original_strides() {
    let mut matrix = Matrix::<f64>::zeros(6, 6).unwrap();
    for (i, value) in [3., 3., 2., 2., 1., 1.].into_iter().enumerate() {
        matrix.view_mut().set(i, i, value).unwrap();
    }
    let operator = matrix.view().try_finite().unwrap();
    let mut pairs = PartialSpectrumBuffer::new(6, 4).unwrap();
    let mut triplets = PartialSvdBuffer::new(6, 6, 4).unwrap();
    let blocks = LobpcgBuffer::<f64>::new(6, 4).unwrap();
    assert!(pairs.eigenvalues().is_empty());
    assert_eq!(
        (pairs.eigenvectors().rows(), pairs.eigenvectors().cols()),
        (6, 0)
    );
    assert!(triplets.singular_values().is_empty());
    assert_eq!(triplets.left_vectors().cols(), 0);
    assert_eq!(triplets.right_vectors().cols(), 0);
    assert!(blocks.eigenvalues().is_empty());
    assert!(blocks.residuals().is_empty());
    assert_eq!(blocks.eigenvectors().cols(), 0);

    let plan = PartialSpectrumSolver::new(4).with_structure(OperatorStructure::Hermitian);
    let mut workspace = plan.workspace(&operator).unwrap();
    let report = plan
        .solve_into(
            &operator,
            &[1.; 6][..],
            &mut pairs,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    owned(&report);
    assert_eq!(report.termination(), SpectralTermination::InvariantSubspace);
    assert_eq!(pairs.eigenvalues().len(), 3);
    assert_eq!(pairs.residuals().len(), 3);
    assert_eq!(
        (
            pairs.eigenvectors().cols(),
            pairs.eigenvectors().row_stride()
        ),
        (3, 4)
    );
    assert!(pairs.eigenvectors().get(0, 3).is_err());

    let plan = PartialSvdSolver::new(4);
    let mut workspace = plan.workspace(&operator).unwrap();
    let report = plan
        .solve_into(
            &operator,
            &[1.; 6],
            &mut triplets,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    owned(&report);
    assert_eq!(report.termination(), SpectralTermination::InvariantSubspace);
    assert_eq!(triplets.singular_values().len(), 3);
    assert_eq!(triplets.residuals().len(), 3);
    for view in [triplets.left_vectors(), triplets.right_vectors()] {
        assert_eq!((view.rows(), view.cols(), view.row_stride()), (6, 3, 4));
        assert!(view.get(0, 3).is_err());
    }
}

#[test]
fn action_reports_outlive_workspace_reuse_and_failed_attempts_have_no_view() {
    let matrix = Matrix::<f64>::identity(2).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let plan = ExponentialAction::new(1.);
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let mut output = [0.; 2];
    assert!(workspace.last_accepted().is_none());
    let report = plan
        .apply_into(
            &operator,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 2], 2, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    owned(&report);
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert_eq!(workspace.last_accepted().unwrap().completed(), &[true]);
    let before = output;
    assert!(
        plan.apply_into(
            &operator,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.], 1, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, before);
    assert!(workspace.last_accepted().is_none());
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let stopped = plan
        .apply_into(
            &operator,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 2], 2, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        stopped.termination(),
        SpectralTermination::ExecutionStopped(_)
    ));
    assert_eq!(workspace.last_accepted().unwrap().progress(), &[0.]);
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert!(report.applications() > 0);
    assert_eq!(stopped.applications(), 0);
}

#[test]
fn low_rank_buffers_slice_both_matrix_axes_and_indices() {
    let matrix = Matrix::from_vec(2, 3, vec![3., 0., 0., 0., 0., 0.]).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let sketch = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 0., 0.]).unwrap();
    let probes = Matrix::<f64>::identity(3).unwrap();
    let plan = RandomizedLowRank::new(2);
    let mut workspace = plan.workspace(&operator, 2).unwrap();
    let mut svd = RandomizedSvdBuffer::new(2, 3, 2).unwrap();
    let mut id = InterpolativeBuffer::new(2, 3, 2).unwrap();
    assert_eq!(
        (svd.u().cols(), svd.vt().rows(), svd.singular_values().len()),
        (0, 0, 0)
    );
    assert_eq!(
        (
            id.columns().cols(),
            id.coefficients().rows(),
            id.column_indices().len()
        ),
        (0, 0, 0)
    );
    let report = plan
        .svd_into(
            &operator,
            sketch.view(),
            probes.view(),
            &mut svd,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    owned(&report);
    assert_eq!(report.rank(), 1);
    assert_eq!(
        (svd.u().cols(), svd.vt().rows(), svd.singular_values().len()),
        (1, 1, 1)
    );
    assert_eq!(svd.u().row_stride(), 2);
    assert!((svd.singular_values()[0] - 3.).abs() < 1e-12);
    let report = plan
        .interpolative_into(
            &operator,
            sketch.view(),
            probes.view(),
            &mut id,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.rank(), 1);
    assert_eq!(
        (
            id.columns().cols(),
            id.coefficients().rows(),
            id.column_indices().len()
        ),
        (1, 1, 1)
    );
    assert!(id.columns().get(0, 1).is_err());
    assert!(id.coefficients().get(1, 0).is_err());
}

#[test]
fn eigen_reports_match_the_published_epoch_and_are_absent_before_it() {
    let matrix =
        Matrix::<f64>::from_vec(3, 3, vec![4., 1., 0.5, 1., 2., 0.25, 0.5, 0.25, 3.]).unwrap();
    let options = HermitianEigenOptions::default();
    let mut values = HermitianEigenvaluesBuffer::empty();
    let mut workspace = LinalgWorkspace::new();
    let report = HermitianEigenvaluesBuffer::compute_into(
        &matrix,
        HermitianInput::Exact,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    owned(&report);
    assert_eq!(report.counts(), workspace.last_attempt().unwrap().counts());
    assert!(report.iterations() > 0);
    let owned_values = HermitianEigenvaluesResult::compute(&matrix, HermitianInput::Exact).unwrap();
    assert_eq!(owned_values.report().counts(), report.counts());
    for (borrowed, owned) in values.eigenvalues().iter().zip(owned_values.eigenvalues()) {
        assert!((borrowed - owned).abs() < 1e-12);
    }
    let mut schur = SchurDecomposition::empty();
    let mut qz = GeneralizedSchurDecomposition::empty();
    assert!(schur.report().is_none());
    assert!(qz.report().is_none());
    let report = SchurDecomposition::decompose_into(
        &matrix,
        &SchurOptions::new(),
        &mut schur,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(report.counts(), schur.report().unwrap().counts());
    let report = GeneralizedSchurDecomposition::decompose_into(
        &matrix,
        &Matrix::identity(3).unwrap(),
        &SchurOptions::new(),
        &mut qz,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(report.counts(), qz.report().unwrap().counts());
    assert_eq!(report.counts(), workspace.last_attempt().unwrap().counts());
}

#[test]
fn matrix_function_budget_failure_preserves_work_counts_and_output() {
    let matrix = Matrix::from_vec(2, 2, vec![1., 0.5, 0., 2.]).unwrap();
    let options = MatrixFunctionOptions::new()
        .with_execution_options(LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE));
    let plan = MatrixFunctionPlan::new(&matrix, MatrixFunction::Exp, &options).unwrap();
    let mut workspace = plan
        .workspace(&MatrixFunctionWorkspaceRequest::Evaluate)
        .unwrap();
    let mut output = Matrix::from_vec(2, 2, vec![42.; 4]).unwrap();
    let error = plan
        .evaluate_into(
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own().iterations() == 1)
    );
    assert_eq!(output.as_slice(), &[42.; 4]);
}

#[test]
fn matrix_equation_budget_failure_carries_admitted_counts() {
    let a = Matrix::from_vec(2, 2, vec![3., 1., 0., 2.]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.]).unwrap();
    let options = MatrixEquationOptions::new()
        .with_execution_options(LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE));
    let plan = SylvesterPlan::new(&a, &b, SylvesterSign::Plus, options).unwrap();
    let rhs = Matrix::from_vec(2, 1, vec![5., 3.]).unwrap();
    let error = plan.solve(&rhs).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure { diagnostics, .. }
        if diagnostics.counts().own().iterations() == 1)
    );
}

#[test]
fn reserving_action_storage_invalidates_views_but_keeps_owned_reports_and_attempts() {
    let matrix = Matrix::<f64>::identity(2).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let input = [1.; 2];
    let mut output = [0.; 2];
    let plan = ExponentialAction::new(1.);
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let report = plan
        .apply_into(
            &operator,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&input, 2, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let epoch = workspace.last_attempt().unwrap().epoch();
    assert!(workspace.last_accepted().is_some());
    let larger = ExponentialAction::new(1.);
    workspace
        .try_reserve(&larger.workspace_requirements(&operator, 2).unwrap())
        .unwrap();
    assert!(workspace.last_accepted().is_none());
    assert_eq!(workspace.last_attempt().unwrap().epoch(), epoch);
    assert_eq!(report.termination(), SpectralTermination::Converged);

    let plan = FunctionAction::default();
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    let report = plan
        .apply_into(
            &operator,
            MatrixFunction::Sqrt,
            BlockVectorView::new(&input, 2, 1).unwrap(),
            BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    owned(&report);
    let epoch = workspace.last_attempt().unwrap().epoch();
    assert!(workspace.last_accepted().is_some());
    let larger = FunctionAction::default();
    workspace
        .try_reserve(&larger.workspace_requirements(&operator, 2).unwrap())
        .unwrap();
    assert!(workspace.last_accepted().is_none());
    assert_eq!(workspace.last_attempt().unwrap().epoch(), epoch);
    assert_eq!(report.termination(), SpectralTermination::Converged);
}

#[test]
fn reordered_copies_and_failed_reorders_preserve_factorization_reports() {
    let matrix = Matrix::from_vec(2, 2, vec![1.0_f64, 0.5, 0., 3.]).unwrap();
    let original = SchurDecomposition::decompose(&matrix).unwrap();
    let counts = original.report().unwrap().counts();
    let options = SchurReorderOptions::new();
    let (mut reordered, report) = original.reorder(&[false, true]).unwrap();
    assert_eq!(report.selected_dimension(), 1);
    assert_eq!(reordered.report().unwrap().counts(), counts);
    let before = reordered.t().clone();
    assert!(
        reordered
            .reorder_in_place(
                &[true],
                &options,
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(reordered.report().unwrap().counts(), counts);
    assert_eq!(reordered.t(), &before);
}

#[test]
fn require_accepted_maps_each_termination_to_its_error() {
    use scioxide_core::{NumericalIssue, SciOxideErrorKind};
    assert!(DirectSolveTermination::Accepted.require_accepted().is_ok());
    let error = DirectSolveTermination::RankDeficient
        .require_accepted()
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::Singular);
    for (termination, expected) in [
        (
            DirectSolveTermination::IllConditioned,
            NumericalIssue::IllConditioned,
        ),
        (
            DirectSolveTermination::ResidualRejected,
            NumericalIssue::LossOfSignificance,
        ),
        (
            DirectSolveTermination::StabilityRejected,
            NumericalIssue::IllConditioned,
        ),
    ] {
        let error = termination.require_accepted().unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
        assert!(
            matches!(error.view(), SciOxideErrorView::NumericalInstability { issue, .. } if issue == expected),
            "{termination:?}: {error:?}"
        );
    }
}

#[test]
fn direct_solve_results_release_only_accepted_solutions() {
    use scioxide_core::{RankThreshold, SciOxideErrorKind};
    // The pivots of the LU factors are 1 and 1/16, and x = (1, 1).
    let a = Matrix::from_rows(&[[1.0_f64, 1.], [1., 1.0625]]).unwrap();
    let b = [2., 2.0625];
    let lu = LuDecomposition::factor(&a).unwrap();
    let accepted = lu
        .solve_with_options(&b[..], &DirectSolveOptions::default())
        .unwrap();
    assert_eq!(
        accepted.report().termination(),
        DirectSolveTermination::Accepted
    );
    let solution = accepted.solution().unwrap().clone();
    assert_eq!(solution.as_slice(), &[1., 1.]);
    assert_eq!(accepted.into_accepted().unwrap(), solution);
    // The pivot 1/16 lies below the cutoff τ = 0 + 1/2 · maxᵢ |uᵢᵢ| = 1/2.
    let policy = RankPolicy::Threshold(RankThreshold::new(0., 0.5).unwrap());
    let rejected = lu
        .solve_with_options(&b[..], &DirectSolveOptions::new(policy))
        .unwrap();
    assert_eq!(
        rejected.report().termination(),
        DirectSolveTermination::RankDeficient
    );
    assert!(rejected.solution().is_none());
    let error = rejected.into_accepted().unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::Singular);
}

#[test]
fn a_condition_threshold_rejects_an_ill_conditioned_solve() {
    use scioxide_core::NumericalIssue;
    // κ₁(A) = (2 + δ)²/δ ≈ 4 · 10¹⁰ for δ = 10⁻¹⁰, far above 1/0.9.
    let a = Matrix::from_rows(&[[1.0_f64, 1.], [1., 1. + 1e-10]]).unwrap();
    let options = DirectSolveOptions::new(RankPolicy::Exact)
        .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.9).unwrap());
    let result = LuDecomposition::factor(&a)
        .unwrap()
        .solve_with_options(&[2., 2.][..], &options)
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::IllConditioned
    );
    let reciprocal = result.report().condition().unwrap().reciprocal().value();
    assert!(reciprocal > 0. && reciprocal < 1e-9, "{reciprocal:e}");
    assert!(result.solution().is_none());
    let error = result.into_accepted().unwrap_err();
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::IllConditioned,
                ..
            }
        ),
        "{error:?}"
    );
    // The identity has reciprocal condition 1 and meets the same threshold.
    let identity = Matrix::<f64>::identity(2).unwrap();
    let accepted = LuDecomposition::factor(&identity)
        .unwrap()
        .solve_with_options(&[2., 2.][..], &options)
        .unwrap();
    assert_eq!(accepted.into_accepted().unwrap().as_slice(), &[2., 2.]);
}

#[test]
fn matrix_equation_results_release_only_accepted_solutions() {
    use scioxide_core::{NumericalIssue, ResidualTolerance};
    let a = Matrix::from_rows(&[[3.0_f64, 1., 0.5], [0.25, 2., 1.], [1., 0.5, 4.]]).unwrap();
    let b = Matrix::from_rows(&[[1.0_f64, 0.3], [0.7, 2.]]).unwrap();
    let c = Matrix::from_rows(&[[1.0_f64, 2.], [3., 4.], [5., 6.]]).unwrap();
    let plan =
        SylvesterPlan::new(&a, &b, SylvesterSign::Plus, MatrixEquationOptions::new()).unwrap();
    let accepted = plan.solve(&c).unwrap();
    assert_eq!(
        accepted.report().termination(),
        DirectSolveTermination::Accepted
    );
    let solution = accepted.solution().unwrap().clone();
    assert_eq!(accepted.into_accepted().unwrap(), solution);
    // With atol = rtol = 0 only an exact residual passes, and the rounded
    // solution leaves a nonzero one.
    let exact =
        MatrixEquationOptions::new().with_tolerance(ResidualTolerance::new(0., 0.).unwrap());
    let rejected = SylvesterPlan::new(&a, &b, SylvesterSign::Plus, exact)
        .unwrap()
        .solve(&c)
        .unwrap();
    let report = rejected.report();
    assert_eq!(
        report.termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert!(report.residual() > 0. && report.backward_error() < 1e-14);
    assert!(rejected.solution().is_none());
    let error = rejected.into_accepted().unwrap_err();
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::LossOfSignificance,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn riccati_results_release_only_accepted_solutions() {
    use scioxide_core::{NumericalIssue, ResidualTolerance};
    // With a = ½ and b = q/3 = r = 1, the CARE x − x² + 3 = 0 has the
    // stabilizing root x = (1 + √13)/2 and K = x.
    let one = Matrix::<f64>::identity(1).unwrap();
    let a = Matrix::from_rows(&[[0.5_f64]]).unwrap();
    let q = Matrix::from_rows(&[[3.0_f64]]).unwrap();
    let problem = RiccatiProblem {
        a: a.view(),
        b: one.view(),
        q: q.view(),
        r: one.view(),
        e: None,
        s: None,
    };
    let accepted = RiccatiPlan::new(problem, RiccatiKind::Continuous, RiccatiOptions::new())
        .unwrap()
        .solve()
        .unwrap();
    assert_eq!(
        accepted.report().termination(),
        DirectSolveTermination::Accepted
    );
    let x = accepted.solution().unwrap().clone();
    let k = accepted.gain().unwrap().clone();
    assert!((x.as_slice()[0] - (1. + 13.0_f64.sqrt()) / 2.).abs() < 1e-14);
    assert_eq!(accepted.into_accepted().unwrap(), (x, k));
    // With atol = rtol = 0 only an exact residual passes, and the rounded
    // solution leaves a nonzero one.
    let exact = RiccatiOptions::new().with_tolerance(ResidualTolerance::new(0., 0.).unwrap());
    let rejected = RiccatiPlan::new(problem, RiccatiKind::Continuous, exact)
        .unwrap()
        .solve()
        .unwrap();
    assert_eq!(
        rejected.report().termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert!(rejected.report().residual() > 0.);
    assert!(rejected.solution().is_none() && rejected.gain().is_none());
    let error = rejected.into_accepted().unwrap_err();
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::LossOfSignificance,
                ..
            }
        ),
        "{error:?}"
    );
}
