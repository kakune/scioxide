#![allow(clippy::unwrap_used)]
use super::*;
use crate::CsrMatrix;
use crate::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
use crate::{SparseSolveOptions, SparseSolveWorkspace};
#[cfg(feature = "std")]
use core::num::NonZeroUsize;
use scioxide_core::AlgebraicScalar;
#[cfg(feature = "std")]
use scioxide_core::ExecutionPolicy;
use scioxide_core::{Complex, RealScalar};

fn limits() -> SparseQrLimits {
    SparseQrLimits::new(512, 256, 1 << 20)
}
// Root-scope factorization with default options.
fn factored<S: NumericScalar>(
    symbolic: SparseQrSymbolic,
    matrix: SparseView<'_, S>,
) -> Result<SparseQr<S>, SciOxideError> {
    SparseQr::factor(symbolic, matrix)
}
// Root-scope minimum-norm preparation with default options.
fn prepared<S: NumericScalar>(
    factor: SparseQr<S>,
    threshold: RankThreshold<S::Real>,
) -> SparseQrMinimumNormPlan<S> {
    SparseQrMinimumNormPlan::build(
        factor,
        threshold,
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap()
}
fn scalar<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(re).unwrap(),
        S::Real::try_from_f64_lossy(im).unwrap(),
    )
}
fn matrix<S: NumericScalar>(rows: usize, cols: usize, values: &[S]) -> crate::CsrMatrix<S> {
    let mut builder = SparseBuilder::new(rows, cols);
    for (pos, &value) in values.iter().enumerate() {
        if value != S::ZERO {
            builder.push(pos / cols, pos % cols, value).unwrap();
        }
    }
    builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
fn close<S: NumericScalar>(actual: S, expected: S, multiple: f64) {
    let error = (actual - expected).abs().try_to_f64_exact().unwrap();
    let eps = S::Real::EPSILON.try_to_f64_exact().unwrap();
    assert!(
        error <= multiple * eps * (1.0 + expected.abs().try_to_f64_exact().unwrap()),
        "{actual:?} != {expected:?}"
    );
}
fn cases<S: NumericScalar>() {
    let zero = S::ZERO;
    let one = S::ONE;
    let two = scalar::<S>(2., 0.);
    let three = scalar::<S>(3., 0.);
    let tall = matrix(3, 2, &[one, zero, zero, one, one, one]);
    let plan = SparseQrSymbolic::analyze(
        tall.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, tall.view()).unwrap();
    let threshold = RankThreshold::new(
        S::Real::ZERO,
        S::Real::EPSILON * S::Real::try_from_f64_exact(64.).unwrap(),
    )
    .unwrap();
    let (x, report) = prepared(factor, threshold)
        .solve(&[one, two, scalar(4., 0.)])
        .unwrap();
    close(x[0], scalar(4. / 3., 0.), 1024.);
    close(x[1], scalar(7. / 3., 0.), 1024.);
    assert_eq!(report.rank(), 2);
    assert!(
        report
            .normal_residual()
            .raw_norm()
            .try_to_f64_exact()
            .unwrap()
            <= 1024. * S::Real::EPSILON.try_to_f64_exact().unwrap()
    );

    let wide = matrix(2, 3, &[one, zero, one, zero, one, one]);
    let plan = SparseQrSymbolic::analyze(
        wide.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, wide.view()).unwrap();
    let (x, report) = prepared(factor.clone(), threshold)
        .solve(&[one, two])
        .unwrap();
    for (&actual, expected) in x.iter().zip([zero, one, one]) {
        close(actual, expected, 1024.);
    }
    assert_eq!(report.rank(), 2);
    let (adjoint, _) = prepared(factor, threshold)
        .solve_adjoint(&[one, two, scalar(4., 0.)])
        .unwrap();
    close(adjoint[0], scalar(4. / 3., 0.), 1024.);
    close(adjoint[1], scalar(7. / 3., 0.), 1024.);

    let deficient = matrix(
        3,
        2,
        &[one, two, two, scalar(4., 0.), three, scalar(6., 0.)],
    );
    let plan = SparseQrSymbolic::analyze(
        deficient.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, deficient.view()).unwrap();
    let (x, report) = prepared(factor.clone(), threshold)
        .solve(&[one, two, three])
        .unwrap();
    close(x[0], scalar(0.2, 0.), 2048.);
    close(x[1], scalar(0.4, 0.), 2048.);
    assert_eq!(report.rank(), 1);
    let (adjoint, report) = prepared(factor, threshold)
        .solve_adjoint(&[one, zero])
        .unwrap();
    for (&actual, expected) in adjoint.iter().zip([
        scalar(1. / 70., 0.),
        scalar(2. / 70., 0.),
        scalar(3. / 70., 0.),
    ]) {
        close(actual, expected, 2048.);
    }
    assert_eq!(report.rank(), 1);
    assert!(
        report
            .normal_residual()
            .raw_norm()
            .try_to_f64_exact()
            .unwrap()
            <= 2048. * S::Real::EPSILON.try_to_f64_exact().unwrap()
    );
}
#[test]
fn real_and_complex_rectangular_minimum_norm() {
    cases::<f32>();
    cases::<f64>();
    cases::<Complex<f32>>();
    cases::<Complex<f64>>();
}
#[test]
fn complex_adjoint_and_q_reconstruction() {
    let one = Complex::new(1.0_f64, 0.0);
    let zero = Complex::new(0.0, 0.0);
    let a = matrix(
        3,
        2,
        &[
            Complex::new(1., 1.),
            zero,
            zero,
            Complex::new(2., -1.),
            one,
            one,
        ],
    );
    let plan = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, a.view()).unwrap();
    let mut r = [[zero; 2]; 3];
    for (row, col, value) in factor.r_entries() {
        r[row][col] = value;
    }
    for (col, _) in r[0].iter().enumerate() {
        let mut input = [zero; 3];
        for row in 0..3 {
            input[row] = r[row][col];
        }
        let mut result = [zero; 3];
        factor
            .apply_q_into(
                &input,
                SparseOperation::Normal,
                SparseSolveOptions::new(),
                &mut result,
                &mut SparseSolveWorkspace::borrowed(&mut [zero; 3]),
                ExecutionContext::sequential(),
            )
            .unwrap();
        let old_col = factor.column_permutation()[col];
        for (row, &actual) in result.iter().enumerate() {
            let expected = a
                .iter()
                .find(|&(i, j, _)| i == row && j == old_col)
                .map_or(zero, |(_, _, value)| value);
            close(actual, expected, 1024.);
        }
    }
    let x = [Complex::new(1., -1.), Complex::new(2., 1.)];
    let rhs = [
        a.iter()
            .filter(|&(i, _, _)| i == 0)
            .fold(zero, |sum, (_, j, v)| sum + v * x[j]),
        a.iter()
            .filter(|&(i, _, _)| i == 1)
            .fold(zero, |sum, (_, j, v)| sum + v * x[j]),
        a.iter()
            .filter(|&(i, _, _)| i == 2)
            .fold(zero, |sum, (_, j, v)| sum + v * x[j]),
    ];
    let threshold = RankThreshold::new(0., 1e-12).unwrap();
    let (solved, _) = prepared(factor.clone(), threshold).solve(&rhs).unwrap();
    for (&actual, &expected) in solved.iter().zip(&x) {
        close(actual, expected, 1024.);
    }
    let adjoint_rhs = [0, 1].map(|col| {
        a.iter()
            .filter(|&(_, j, _)| j == col)
            .fold(zero, |sum, (i, _, v)| sum + v.conj() * rhs[i])
    });
    let (adjoint, report) = prepared(factor.clone(), threshold)
        .solve_adjoint(&adjoint_rhs)
        .unwrap();
    for (&actual, &expected) in adjoint.iter().zip(&rhs) {
        close(actual, expected, 1024.);
    }
    assert!(report.normal_residual().raw_norm() < 1e-11);
    let transpose_rhs = [0, 1].map(|col| {
        a.iter()
            .filter(|&(_, j, _)| j == col)
            .fold(zero, |sum, (i, _, v)| sum + v * rhs[i].conj())
    });
    let (transpose, _) = prepared(factor, threshold)
        .solve_transpose(&transpose_rhs)
        .unwrap();
    for (&actual, &expected) in transpose.iter().zip(&rhs) {
        close(actual, expected.conj(), 1024.);
    }
}
#[test]
fn fill_budget_and_refactor_rollback() {
    let a = matrix(3, 2, &[1.0_f64, 0., 0., 1., 1., 1.]);
    let plan = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let mut factor = SparseQr::factor(plan.clone(), a.view()).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut changed = a.clone();
    assert_eq!(
        changed.set_value(0, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    let changed = CsrMatrix::from_raw(
        changed.rows(),
        changed.cols(),
        alloc::vec![0; changed.rows()+1],
        alloc::vec![],
        alloc::vec![],
    )
    .unwrap();
    assert!(
        factor
            .refactor(
                changed.view(),
                SparseSolveOptions::new(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    let (x, _) = prepared(factor.clone(), RankThreshold::new(0., 1e-12).unwrap())
        .solve(&[1., 2., 4.])
        .unwrap();
    close(x[0], 4. / 3., 1024.);
    let tight = limits().with_max_factor_entries(1);
    let plan = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::default(),
        tight,
    )
    .unwrap();
    assert!(SparseQr::factor(plan.clone(), a.view()).is_err());
    let mut wrong_workspace = plan.workspace(()).unwrap();
    assert!(
        factor
            .refactor(
                a.view(),
                SparseSolveOptions::new(),
                &mut wrong_workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(factor.column_permutation().len(), 2);
    let mut wrong_solve = SparseSolveWorkspace::new();
    let mut output = [9.; 2];
    assert!(
        prepared(factor.clone(), RankThreshold::new(0., 1e-12).unwrap())
            .solve_into(
                &[1., 2., 4.],
                SparseOperation::Normal,
                SparseSolveOptions::new(),
                &mut output,
                &mut wrong_solve,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [9.; 2]);
}

#[test]
fn zero_extents_and_ordering() {
    let empty_rows = matrix::<f64>(0, 3, &[]);
    let plan = SparseQrSymbolic::analyze(
        empty_rows.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, empty_rows.view()).unwrap();
    let (solution, report) = prepared(factor, RankThreshold::new(0., 0.).unwrap())
        .solve(&[])
        .unwrap();
    assert_eq!(solution, [0.; 3]);
    assert_eq!(report.rank(), 0);
    let empty_cols = matrix::<f64>(3, 0, &[]);
    let plan = SparseQrSymbolic::analyze(
        empty_cols.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, empty_cols.view()).unwrap();
    let (solution, report) = prepared(factor, RankThreshold::new(0., 0.).unwrap())
        .solve(&[1., 2., 3.])
        .unwrap();
    assert!(solution.is_empty());
    close(report.residual().raw_norm(), 14.0_f64.sqrt(), 64.);
    let identity = matrix(2, 2, &[1.0_f64, 0., 0., 1.]);
    let order = Permutation::new(alloc::vec![1, 0]).unwrap();
    let plan = SparseQrSymbolic::analyze(
        identity.structural_pattern(),
        crate::SparseOrdering::Supplied(&order),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, identity.view()).unwrap();
    assert_eq!(factor.column_permutation(), &[1, 0][..]);
}

#[test]
fn caller_threshold_truncates_model_but_reports_original_residual() {
    let a = matrix(2, 2, &[2.0_f64, 0., 0., 1.]);
    let plan = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::default(),
        limits(),
    )
    .unwrap();
    let factor = SparseQr::factor(plan, a.view()).unwrap();
    let (solution, report) = prepared(factor, RankThreshold::new(3., 0.).unwrap())
        .solve(&[2., 1.])
        .unwrap();
    assert_eq!(report.rank(), 0);
    assert_eq!(solution, [0.; 2]);
    close(report.residual().raw_norm(), 5.0_f64.sqrt(), 16.);
    close(report.normal_residual().raw_norm(), 17.0_f64.sqrt(), 16.);
    let tight = limits().with_max_active_entries(1);
    let plan = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::default(),
        tight,
    )
    .unwrap();
    assert!(SparseQr::factor(plan, a.view()).is_err());
}

#[test]
fn fixed_order_rank_detection_continues_past_dependent_columns() {
    // The first column is zero; columns 1 and 2 are dependent; column 3 adds rank.
    let a = matrix(3, 4, &[0., 1., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let factor = factored(symbolic, a.view()).unwrap();
    assert_eq!(factor.column_permutation(), &[0, 1, 2, 3]);
    let plan = prepared(factor, RankThreshold::new(0., 1e-12).unwrap());
    assert_eq!(plan.rank(), 2);
    let (x, report) = plan.solve(&[5., 6., 7.]).unwrap();
    for (actual, expected) in x.into_iter().zip([0., 1., 2., 2.]) {
        close(actual, expected, 1024.);
    }
    close(report.residual().raw_norm(), 7., 1024.);
    close(report.normal_residual().raw_norm(), 0., 1024.);
    let (x, _) = plan.solve_adjoint(&[0., 1., 2., 3.]).unwrap();
    for (actual, expected) in x.into_iter().zip([1., 1., 0.]) {
        close(actual, expected, 1024.);
    }
    let a = matrix(2, 2, &[1., 0., 0., 2.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let plan = prepared(
        factored(symbolic, a.view()).unwrap(),
        RankThreshold::new(1., 0.).unwrap(),
    );
    assert_eq!(plan.rank(), 1); // Equality is rejected, and the later column survives.
    assert_eq!(plan.solve(&[1., 2.]).unwrap().0, [0., 1.]);
}

#[test]
fn disconnected_columns_do_no_trailing_update_work() {
    let a = matrix(
        4,
        4,
        &[
            1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.,
        ],
    );
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let mut workspace = symbolic.workspace(()).unwrap();
    let mut factor = factored(symbolic, a.view()).unwrap();
    factor
        .refactor(
            a.view(),
            SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(workspace.update_visits(), 0);
}

#[test]
fn prepared_solves_reuse_reductions_and_record_preflight_rejection() {
    use scioxide_core::{AttemptOutcome, Budget, BudgetKind, Capacity, SolveBudgets};
    let a = matrix(3, 2, &[1., 0., 0., 1., 1., 1.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let one = SparseSolveOptions::new()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
    let factor = factored(symbolic, a.view()).unwrap();
    let threshold = RankThreshold::new(0., 1e-12).unwrap();
    // Preparation admits one iteration per compressed and secondary pivot.
    assert_eq!(
        SparseQrMinimumNormPlan::build(
            factor.clone(),
            threshold,
            one,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let plan = prepared(factor, threshold);
    let mut workspace = SparseSolveWorkspace::new();
    let requirements = plan.workspace_requirements(1).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let mut output = [19.; 2];
    for _ in 0..3 {
        plan.solve_into(
            &[1., 2., 4.],
            SparseOperation::Normal,
            one,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
        assert_eq!(attempt.counts().own().iterations(), 1);
    }
    output.fill(19.);
    assert!(
        plan.solve_into(
            &[1., f64::NAN, 4.],
            SparseOperation::Normal,
            one,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, [19.; 2]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let mut raw = plan.into_factor();
    let mut workspace = raw.symbolic().workspace(()).unwrap();
    let mut bad = a;
    assert_eq!(
        bad.set_value(0, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    let bad = CsrMatrix::from_raw(
        bad.rows(),
        bad.cols(),
        alloc::vec![0; bad.rows()+1],
        alloc::vec![],
        alloc::vec![],
    )
    .unwrap();
    // A different accepted pattern is rejected before the numeric epoch.
    assert!(
        raw.refactor(
            bad.view(),
            SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
}

#[test]
fn supplied_ordering_changes_r_fill() {
    let n = 10;
    let mut builder = crate::SparseBuilder::new(2 * n - 1, n);
    for col in 0..n {
        builder.push(col, col, 2.).unwrap();
    }
    for col in 1..n {
        builder.push(n + col - 1, 0, 1.).unwrap();
        builder.push(n + col - 1, col, 1.).unwrap();
    }
    let a = builder
        .finish_csr(
            crate::DuplicatePolicy::Reject,
            crate::ExplicitZeroPolicy::Preserve,
        )
        .unwrap();
    let leaves = Permutation::new((1..n).chain(core::iter::once(0)).collect()).unwrap();
    let analyze = |order| {
        SparseQrSymbolic::analyze(
            a.structural_pattern(),
            order,
            SparseQrLimits::new(2000, 1000, 1 << 20),
        )
        .unwrap()
    };
    let natural = factored(analyze(crate::SparseOrdering::Natural), a.view()).unwrap();
    let ordered = factored(analyze(crate::SparseOrdering::Supplied(&leaves)), a.view()).unwrap();
    assert_eq!(ordered.column_permutation(), leaves.new_to_old());
    assert!(ordered.r_entries().count() < natural.r_entries().count());
    assert!(ordered.factor_entries() < natural.factor_entries());
}

#[test]
fn block_solve_and_format_reuse() {
    let a = matrix(2, 3, &[1., 0., 1., 0., 1., 1.]);
    let csc = a.to_csc().unwrap();
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let factor = factored(symbolic, csc.view()).unwrap();
    let threshold = RankThreshold::new(0., 1e-12).unwrap();
    let rhs = [1., 2., 2., 4.];
    // The block Q action matches column-wise vector actions.
    let mut q_block = [19.; 4];
    let mut q_workspace = factor.workspace(2).unwrap();
    factor
        .apply_q_block_into(
            BlockVectorView::new(&rhs, 2, 2).unwrap(),
            SparseOperation::Adjoint,
            SparseSolveOptions::new(),
            &mut BlockVectorViewMut::new(&mut q_block, 2, 2).unwrap(),
            &mut q_workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    for (column, expected) in rhs.chunks_exact(2).zip(q_block.chunks_exact(2)) {
        let mut q_column = [19.; 2];
        factor
            .apply_q_into(
                column,
                SparseOperation::Adjoint,
                SparseSolveOptions::new(),
                &mut q_column,
                &mut q_workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(q_column, expected);
        // Q is orthogonal, so the column norm is preserved.
        close(
            q_column[0].hypot(q_column[1]),
            column[0].hypot(column[1]),
            16.,
        );
    }
    let mut out = [19.; 6];
    let plan = prepared(factor, threshold);
    let mut workspace = plan.workspace(2).unwrap();
    plan.solve_block_into(
        BlockVectorView::new(&rhs, 2, 2).unwrap(),
        SparseOperation::Normal,
        SparseSolveOptions::new(),
        &mut BlockVectorViewMut::new(&mut out, 3, 2).unwrap(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (actual, expected) in out.into_iter().zip([0., 1., 1., 0., 2., 2.]) {
        close(actual, expected, 1024.);
    }
    #[cfg(feature = "std")]
    {
        let workers = NonZeroUsize::new(2).unwrap();
        let executor = scioxide_core::ThreadExecutor::new(workers);
        let options =
            SparseSolveOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
        let before = out;
        plan.solve_block_into(
            BlockVectorView::new(&rhs, 2, 2).unwrap(),
            SparseOperation::Normal,
            options,
            &mut BlockVectorViewMut::new(&mut out, 3, 2).unwrap(),
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        assert_eq!(before, out);
    }
}

#[cfg(feature = "std")]
#[test]
fn incomplete_executor_preserves_block_output() {
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn};
    struct Incomplete;
    impl Executor for Incomplete {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
    }
    let a = matrix(2, 3, &[1., 0., 1., 0., 1., 1.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let factor = factored(symbolic, a.view()).unwrap();
    let options = SparseSolveOptions::new()
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let mut q = [19.; 4];
    assert!(
        factor
            .apply_q_block_into(
                BlockVectorView::new(&[1., 2., 3., 4.], 2, 2).unwrap(),
                SparseOperation::Normal,
                options,
                &mut BlockVectorViewMut::new(&mut q, 2, 2).unwrap(),
                &mut SparseSolveWorkspace::borrowed(&mut [0.; 4]),
                ExecutionContext::new(&Incomplete)
            )
            .is_err()
    );
    assert_eq!(q, [19.; 4]);
    let plan = prepared(factor, RankThreshold::new(0., 1e-12).unwrap());
    let mut workspace = plan.workspace(2).unwrap();
    let mut out = [19.; 6];
    assert!(
        plan.solve_block_into(
            BlockVectorView::new(&[1., 2., 3., 4.], 2, 2).unwrap(),
            SparseOperation::Normal,
            options,
            &mut BlockVectorViewMut::new(&mut out, 3, 2).unwrap(),
            &mut workspace,
            ExecutionContext::new(&Incomplete)
        )
        .is_err()
    );
    assert_eq!(out, [19.; 6]);
}

#[test]
fn numeric_workspace_growth_and_retained_limits_are_checked() {
    use scioxide_core::{Capacity, SciOxideErrorKind};
    let small = matrix(1, 1, &[1.]);
    let small_plan = SparseQrSymbolic::analyze(
        small.structural_pattern(),
        crate::SparseOrdering::Natural,
        SparseQrLimits::new(16, 16, 1 << 20),
    )
    .unwrap();
    let mut workspace = small_plan.workspace::<f64>(()).unwrap();
    let a = matrix(3, 2, &[1., 0., 0., 1., 1., 1.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let requirements = symbolic.workspace_requirements::<f64>(()).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let mut factor = factored(symbolic, a.view()).unwrap();
    factor
        .refactor(
            a.view(),
            SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let tight = SparseQrSymbolic::analyze(
        small.structural_pattern(),
        crate::SparseOrdering::Natural,
        SparseQrLimits::new(
            16,
            16,
            small_plan
                .workspace_requirements::<f64>(())
                .unwrap()
                .total_bytes(),
        ),
    )
    .unwrap();
    assert_eq!(
        workspace
            .check(&tight.workspace_requirements::<f64>(()).unwrap())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let overflow = SparseQrSymbolic::analyze(
        small.structural_pattern(),
        crate::SparseOrdering::Natural,
        SparseQrLimits::new(usize::MAX, 16, usize::MAX),
    )
    .unwrap();
    assert_eq!(
        overflow
            .workspace_requirements::<f64>(())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
}

#[test]
fn aggregate_qr_byte_limit_has_an_exact_admission_boundary() {
    use scioxide_core::SciOxideErrorKind;
    let a = matrix(2, 3, &[1., 0., 1., 0., 1., 1.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let bytes = symbolic
        .workspace_requirements::<f64>(())
        .unwrap()
        .total_bytes();
    for extra in [0, 1] {
        let symbolic = SparseQrSymbolic::analyze(
            a.structural_pattern(),
            crate::SparseOrdering::Natural,
            limits().with_max_workspace_bytes(bytes - extra),
        )
        .unwrap();
        let result = factored(symbolic, a.view());
        if extra == 0 {
            let factor = result.unwrap();
            assert!(factor.workspace(1).is_ok());
            let plan = prepared(factor, RankThreshold::new(0., 1e-12).unwrap());
            assert!(plan.workspace(1).is_ok());
        } else {
            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
        }
    }
}

#[test]
fn cloned_factors_keep_reserved_refactor_capacity() {
    use scioxide_core::Capacity;
    let a = matrix(2, 1, &[3.0_f64, 4.]);
    let b = matrix(2, 1, &[6.0_f64, 8.]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let requirements = symbolic.workspace_requirements::<f64>(()).unwrap();
    let mut workspace = symbolic.workspace(()).unwrap();
    let factor = factored(symbolic, a.view()).unwrap();
    let plan = prepared(factor.clone(), RankThreshold::new(0., 1e-12).unwrap());
    // Each refactor publishes by swapping the copy's storage into the workspace.
    for mut copy in [factor.clone(), plan.clone().into_factor()] {
        for (values, norm) in [(&b, 10.), (&a, 5.)] {
            copy.refactor(
                values.view(),
                SparseSolveOptions::new(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(
                workspace.check(&requirements).unwrap(),
                Capacity::Sufficient
            );
            let diagonal: f64 = copy.r_entries().map(|(_, _, value)| value).sum();
            close(diagonal.abs(), norm, 16.);
        }
    }
    // The originals are independent of their refactored copies.
    let diagonal: f64 = factor.r_entries().map(|(_, _, value)| value).sum();
    close(diagonal.abs(), 5., 16.);
    let (x, _) = plan.solve(&[3., 4.]).unwrap();
    close(x[0], 1., 16.);
}

#[test]
fn refactor_preflight_reports_the_short_workspace_quantity() {
    use scioxide_core::{AttemptOutcome, SciOxideErrorView};
    let a = matrix(3, 2, &[1.0_f64, 0., 0., 1., 1., 1.]);
    let wide = matrix(2, 3, &[1.0_f64, 0., 1., 0., 1., 1.]);
    let analyze = |m: &CsrMatrix<f64>, limits| {
        SparseQrSymbolic::analyze(
            m.structural_pattern(),
            crate::SparseOrdering::Natural,
            limits,
        )
        .unwrap()
    };
    let mut factor = factored(analyze(&a, limits()), a.view()).unwrap();
    let before: alloc::vec::Vec<_> = factor.r_entries().collect();
    for (symbolic, required) in [
        (analyze(&a, limits().with_max_factor_entries(4)), 512),
        (analyze(&a, limits().with_max_active_entries(3)), 256),
        (analyze(&wide, limits()), 2),
    ] {
        let mut workspace = symbolic.workspace(()).unwrap();
        let error = factor
            .refactor(
                a.view(),
                SparseSolveOptions::new(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
                if expected == [required] && actual.len() == 1 && actual[0] != required),
            "{error:?}"
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
    assert!(factor.r_entries().eq(before));
}

#[test]
fn larger_reservations_still_enforce_the_analyzed_active_cap() {
    use scioxide_core::SciOxideErrorKind;
    let a = matrix(3, 2, &[1.0_f64, 0., 0., 1., 1., 1.]);
    let analyze = |limits| {
        SparseQrSymbolic::analyze(
            a.structural_pattern(),
            crate::SparseOrdering::Natural,
            limits,
        )
        .unwrap()
    };
    // Four stored entries exceed a three-entry active cap.
    let tight = limits().with_max_active_entries(3);
    let mut roomy = analyze(limits()).workspace(()).unwrap();
    let error = factored(analyze(tight), a.view()).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    let mut factor = factored(analyze(limits()), a.view()).unwrap();
    factor
        .refactor(
            a.view(),
            SparseSolveOptions::new(),
            &mut roomy,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let mut capped = SparseQr {
        symbolic: analyze(tight),
        data: factor.data.clone(),
    };
    let error = capped
        .refactor(
            a.view(),
            SparseSolveOptions::new(),
            &mut roomy,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
}

#[test]
fn raw_factor_needs_no_representable_column_norms() {
    // Column 1 has two-norm sqrt(2) * 1.5e308 > f64::MAX, yet Q = I and R = A.
    let big = 1.5e308;
    let a = matrix(2, 2, &[1.0_f64, big, 0., big]);
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        crate::SparseOrdering::Natural,
        limits(),
    )
    .unwrap();
    let mut workspace = symbolic.workspace(()).unwrap();
    let mut factor = factored(symbolic, a.view()).unwrap();
    factor
        .refactor(
            a.view(),
            SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(factor.column_permutation(), &[0, 1]);
    let r: alloc::vec::Vec<_> = factor.r_entries().collect();
    assert_eq!(r, [(0, 0, 1.), (0, 1, big), (1, 1, big)]);
    // Q R reproduces each column of A within rounding.
    for (column, expected) in [[1., 0.], [big, big]].into_iter().enumerate() {
        let mut input = [0.; 2];
        for &(row, col, value) in &r {
            if col == column {
                input[row] = value;
            }
        }
        for operation in [SparseOperation::Normal, SparseOperation::Adjoint] {
            let mut output = [0.; 2];
            factor
                .apply_q_into(
                    &input,
                    operation,
                    SparseSolveOptions::new(),
                    &mut output,
                    &mut SparseSolveWorkspace::borrowed(&mut [0.; 2]),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= f64::EPSILON * expected.abs());
            }
        }
    }
}
