#![allow(clippy::unwrap_used)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::{
    Estimation, LinalgWorkspace, Matrix, SchurDecomposition as Schur, SchurOptions,
    SchurReorderOptions, SchurWorkspaceRequest,
};

fn schur<S: NumericScalar>(a: &Matrix<S>) -> Result<Schur<S>, SciOxideError> {
    Schur::decompose(a)
}
fn estimates() -> SchurReorderOptions {
    SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate)
}

fn invariants<S: NumericScalar>(a: &Matrix<S>, s: &Schur<S>, tol: S::Real) {
    let n = a.rows();
    let scale = a
        .as_slice()
        .iter()
        .map(|v| v.abs())
        .fold(S::Real::ONE, S::Real::max);
    for i in 0..n {
        for j in 0..n {
            let aq = (0..n).fold(S::ZERO, |v, k| {
                v + a.view().get(i, k).unwrap() * s.q().view().get(k, j).unwrap()
            });
            let qt = (0..n).fold(S::ZERO, |v, k| {
                v + s.q().view().get(i, k).unwrap() * s.t().view().get(k, j).unwrap()
            });
            assert!(
                (aq - qt).abs() / scale < tol,
                "n={n} ({i},{j}) {:?} {:?}",
                aq,
                qt
            );
            let orth = (0..n).fold(S::ZERO, |v, k| {
                v + s.q().view().get(k, i).unwrap().conj() * s.q().view().get(k, j).unwrap()
            });
            assert!((orth - if i == j { S::ONE } else { S::ZERO }).abs() < tol);
            if i > j + usize::from(S::IS_REAL) {
                assert_eq!(s.t().view().get(i, j).unwrap(), S::ZERO);
            }
        }
    }
}

#[test]
fn analytic_separation_and_transactional_rejection() {
    let options = estimates();
    let a = Matrix::<f64>::from_vec(3, 3, vec![1., 0., 0., 0., 4., 0., 0., 0., 9.]).unwrap();
    let (schur, report) = schur(&a)
        .unwrap()
        .reorder_with_options(&[false, true, false], &options)
        .unwrap();
    assert!((report.separation().unwrap().value() - 3.0_f64).abs() < 1e-13);
    assert!((report.cluster_condition().unwrap().reciprocal().value() - 1.0).abs() < 1e-13);
    invariants(&a, &schur, 1e-13);
    let a = Matrix::<f64>::from_vec(2, 2, vec![1., 1., 0., 1.]).unwrap();
    let mut schur = self::schur(&a).unwrap();
    let before = schur.t().as_slice().to_vec();
    let q_before = schur.q().as_slice().to_vec();
    assert!(
        schur
            .reorder_in_place(
                &[false, true],
                &options,
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(before, schur.t().as_slice());
    assert_eq!(q_before, schur.q().as_slice());
    let (_, report) = schur
        .reorder_with_options(&[true, false], &options)
        .unwrap();
    assert_eq!(report.separation().unwrap().value(), 0.);
    assert_eq!(report.cluster_condition().unwrap().reciprocal().value(), 0.);
}

#[test]
fn nonnormal_analytic_conditioning() {
    // For m=1, the Sylvester operator is [[-3,0],[-6,-8]]. Its inverse
    // one-norm is 7/12. The projector solves R=[-2/3,1/8].
    let a = Matrix::<f64>::from_vec(3, 3, vec![1., 2., 3., 0., 4., 6., 0., 0., 9.]).unwrap();
    let (_, report) = schur(&a)
        .unwrap()
        .reorder_with_options(&[true, false, false], &estimates())
        .unwrap();
    assert!((report.separation().unwrap().value() - 12. / 7.).abs() < 1e-12);
    let expected = 1. / (1. + 4. / 9. + 1. / 64.0_f64).sqrt();
    assert!((report.cluster_condition().unwrap().reciprocal().value() - expected).abs() < 1e-12);
}

#[test]
fn budget_cancellation_masks_and_views() {
    use scioxide_core::{
        AttemptOutcome, Budget, BudgetKind, CancellationToken, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    use scioxide_linalg::MatrixView;
    let a = Matrix::<f64>::from_vec(
        4,
        4,
        vec![
            1., 2., 3., 4., 0., 2., 1., 2., 0., 0., 3., 1., 0., 0., 0., 4.,
        ],
    )
    .unwrap();
    let mut s = schur(&a).unwrap();
    let before = s.t().as_slice().to_vec();
    let options = SchurReorderOptions::default();
    let mut work = Schur::<f64>::workspace(&SchurWorkspaceRequest::reorder(4, &options)).unwrap();
    // The reorder budget belongs to the options; a parent's own iteration
    // allowance is not consumed and the work lands in its `nested()` counts.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(474),
        scioxide_core::algorithm_id!("linalg", "schur", 1),
        4,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let failure = s
        .reorder_in_place(
            &[false, false, false, true],
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_iteration_budget(Budget::ONE),
            ),
            &mut work,
            &mut context,
        )
        .unwrap_err();
    assert!(
        matches!(failure.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own().iterations() == 1)
    );

    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
    let attempt = work.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.counts().own().iterations(), 1);
    assert_eq!(s.t().as_slice(), before);
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        s.reorder_in_place(
            &[false, false, false, true],
            &options,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(s.t().as_slice(), before);
    assert_eq!(
        s.reorder(&[true]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let (_, report) = s.reorder(&[false; 4]).unwrap();
    assert_eq!(report.selected_dimension(), 0);
    let report = s
        .reorder_in_place(&[true; 4], &options, &mut work, &mut context)
        .unwrap();
    assert_eq!(report.selected_dimension(), 4);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    let data = [1., 0., 99., 2., 3., 99.];
    let view = MatrixView::from_slice_with_strides(2, 2, &data, 3, 1).unwrap();
    let schur = Schur::decompose(view).unwrap();
    invariants(
        &Matrix::from_vec(2, 2, vec![1., 0., 2., 3.]).unwrap(),
        &schur,
        1e-13,
    );
}

#[test]
fn decomposition_charges_nested_qr_steps_and_respects_its_budget() {
    use scioxide_core::{
        AttemptOutcome, Budget, BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let a = Matrix::<f64>::from_vec(3, 3, vec![4., -2., 1., 1., 5., -3., 2., 1., -1.]).unwrap();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(475),
        scioxide_core::algorithm_id!("linalg", "schur", 1),
        3,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut work = LinalgWorkspace::new();
    let mut s = Schur::empty();
    Schur::decompose_into(&a, &SchurOptions::new(), &mut s, &mut work, &mut context).unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    let nested = context.counts().nested().iterations();
    assert!(nested > 1);
    assert_eq!(
        work.last_attempt().unwrap().counts().own().iterations(),
        nested
    );
    let before = s.clone();
    assert_eq!(
        Schur::decompose_into(
            &a,
            &SchurOptions::new().with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE)
            ),
            &mut s,
            &mut work,
            &mut context
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert_eq!(s.t().as_slice(), before.t().as_slice());
    assert_eq!(s.q().as_slice(), before.q().as_slice());
}

#[test]
fn workspace_requirements_errors_preserve_factors() {
    let a = Matrix::<f64>::from_vec(3, 3, vec![4., -2., 1., 1., 5., -3., 2., 1., -1.]).unwrap();
    let options = SchurOptions::<f64>::new();
    let mut s = schur(&a).unwrap();
    let before = s.clone();
    let unchanged = |s: &Schur<f64>| {
        s.t().as_slice() == before.t().as_slice()
            && s.q().as_slice() == before.q().as_slice()
            && s.eigenvalues() == before.eigenvalues()
            && s.blocks() == before.blocks()
    };
    let mut work = LinalgWorkspace::new();
    for (options, kind) in [
        (
            options.with_execution_options((options).execution_options().with_byte_limit(1)),
            SciOxideErrorKind::ResourceLimit,
        ),
        (
            options.with_execution_options(
                (options)
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1)),
            ),
            SciOxideErrorKind::CapabilityUnavailable,
        ),
    ] {
        assert_eq!(
            Schur::decompose_into(
                &a,
                &options,
                &mut s,
                &mut work,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            kind
        );
        assert!(unchanged(&s));
        assert!(work.last_attempt().unwrap().epoch().is_none());
    }
    let mut short = [0.0; 4];
    assert_eq!(
        Schur::decompose_into(
            &a,
            &options,
            &mut s,
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(unchanged(&s));
    let mask = [false, false, true];
    for (reorder, kind) in [
        (
            estimates()
                .with_execution_options((estimates()).execution_options().with_byte_limit(1)),
            SciOxideErrorKind::ResourceLimit,
        ),
        (
            estimates().with_execution_options(
                (estimates())
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1)),
            ),
            SciOxideErrorKind::CapabilityUnavailable,
        ),
    ] {
        assert_eq!(
            s.reorder_in_place(&mask, &reorder, &mut work, ExecutionContext::sequential())
                .unwrap_err()
                .kind(),
            kind
        );
        assert!(unchanged(&s));
    }
    assert_eq!(
        s.reorder_in_place(
            &mask,
            &estimates(),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(unchanged(&s));
    assert_eq!(
        Schur::decompose_with_options(
            &a,
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1))
            )
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(
        schur(&Matrix::<f64>::zeros(2, 3).unwrap())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    for request in [
        SchurWorkspaceRequest::new(usize::MAX, &options),
        SchurWorkspaceRequest::reorder(usize::MAX, &estimates()),
    ] {
        assert_eq!(
            Schur::<f64>::workspace_requirements(&request)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
    }
    let skip = Schur::<f64>::workspace_requirements(&SchurWorkspaceRequest::reorder(
        4,
        &SchurReorderOptions::new(),
    ))
    .unwrap();
    let estimate =
        Schur::<f64>::workspace_requirements(&SchurWorkspaceRequest::reorder(4, &estimates()))
            .unwrap();
    assert!(estimate.required_bytes::<f64>().unwrap() > skip.required_bytes::<f64>().unwrap());
}

fn nearly_inseparable<R: RealScalar + NumericScalar<Real = R>>() {
    let one = <R as scioxide_core::AlgebraicScalar>::ONE;
    let zero = <R as scioxide_core::AlgebraicScalar>::ZERO;
    let delta = R::try_from_usize_exact(128).unwrap() * <R as RealScalar>::EPSILON;
    let a = Matrix::from_vec(2, 2, vec![one, one, zero, one + delta]).unwrap();
    let (s, report) = schur(&a)
        .unwrap()
        .reorder_with_options(&[false, true], &estimates())
        .unwrap();
    assert!(report.cluster_condition().unwrap().reciprocal().value() < delta * (one + one));
    assert!(
        (report.separation().unwrap().value() - delta).abs()
            < R::try_from_usize_exact(8).unwrap() * <R as RealScalar>::EPSILON
    );
    invariants(
        &a,
        &s,
        R::try_from_usize_exact(256).unwrap() * <R as RealScalar>::EPSILON,
    );
}
#[test]
fn nearly_inseparable32() {
    nearly_inseparable::<f32>();
}
#[test]
fn nearly_inseparable64() {
    nearly_inseparable::<f64>();
}

#[test]
fn real_complex_selected_subspace_agrees() {
    let a = Matrix::<f64>::from_vec(
        4,
        4,
        vec![
            1., 2., 1., 2., -1., 1., 3., 4., 0., 0., 5., 3., 0., 0., -2., 5.,
        ],
    )
    .unwrap();
    let complex = Matrix::from_vec(
        4,
        4,
        a.as_slice().iter().map(|&v| Complex::new(v, 0.)).collect(),
    )
    .unwrap();
    let real = schur(&a).unwrap();
    let comp = schur(&complex).unwrap();
    let selection: Vec<_> = comp.eigenvalues().iter().map(|z| z.re > 3.).collect();
    let options = estimates();
    let (real, rr) = real
        .reorder_with_options(&[false, false, true, true], &options)
        .unwrap();
    let (comp, cr) = comp.reorder_with_options(&selection, &options).unwrap();
    assert_eq!(rr.selected_dimension(), cr.selected_dimension());
    assert!(
        (rr.cluster_condition().unwrap().reciprocal().value()
            - cr.cluster_condition().unwrap().reciprocal().value())
        .abs()
            < 1e-12
    );
    for i in 0..4 {
        for j in 0..4 {
            let rp = (0..2)
                .map(|k| real.q().view().get(i, k).unwrap() * real.q().view().get(j, k).unwrap())
                .sum::<f64>();
            let cp = (0..2).fold(Complex::new(0., 0.), |sum, k| {
                sum + comp.q().view().get(i, k).unwrap() * comp.q().view().get(j, k).unwrap().conj()
            });
            assert!((cp - Complex::new(rp, 0.)).norm() < 1e-12);
        }
    }
}

#[test]
fn reordering_retains_near_real_schur_pairs() {
    for delta in [1e-6, 1e-10, 1e-14] {
        let a =
            Matrix::from_vec(3, 3, vec![0.5_f64, 0.2, 0.3, 0., 1., -delta, 0., delta, 1.]).unwrap();
        let factor = schur(&a).unwrap();
        let (factor, report) = factor.reorder(&[false, true, false]).unwrap();
        assert_eq!(report.selected_dimension(), 2);
        let pair = &factor.eigenvalues()[..2];
        assert_eq!(pair[0], pair[1].conj());
        assert!(pair[0].im > 0.0);
        assert!((pair[0] - Complex::new(1.0, delta)).norm() <= 16.0 * f64::EPSILON);
        invariants(&a, &factor, 64.0 * f64::EPSILON);
    }
}
