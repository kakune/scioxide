//! Original-coordinate tests independent of the sparse elimination kernel.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#[cfg(feature = "std")]
use scioxide_core::SciOxideError;
use scioxide_core::{
    Accuracy, BlockVectorView, BlockVectorViewMut, Complex, ExecutionContext, NumericScalar,
    Permutation, RealScalar,
};
use scioxide_sparse::*;

/// Solves with `solve_into` and returns the solution, for configured options.
macro_rules! solve {
    ($factor:expr, $rhs:expr, $operation:expr, $options:expr) => {{
        let factor = &$factor;
        let rhs = $rhs;
        let mut output = rhs.to_vec();
        factor
            .solve_into(
                &rhs[..],
                $operation,
                $options,
                &mut output,
                &mut factor.workspace(1).unwrap(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .map(|()| output)
    }};
}
fn limits(entries: usize) -> SparseLuLimits {
    SparseLuLimits::new(entries, usize::MAX)
}
fn scalar<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_exact(re).unwrap(),
        S::Real::try_from_f64_exact(im).unwrap(),
    )
}
fn matrix<S: NumericScalar>(n: usize, values: &[S]) -> CsrMatrix<S> {
    let mut b = SparseBuilder::new(n, n);
    for (pos, &value) in values.iter().enumerate() {
        b.push(pos / n, pos % n, value).unwrap();
    }
    b.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
fn rhs<S: NumericScalar>(a: &[S], x: &[S], op: SparseOperation) -> Vec<S> {
    let n = x.len();
    (0..n)
        .map(|i| {
            (0..n).fold(S::ZERO, |sum, j| {
                let value = match op {
                    SparseOperation::Normal => a[i * n + j],
                    SparseOperation::Transpose => a[j * n + i],
                    SparseOperation::Adjoint => a[j * n + i].conj(),
                };
                sum + value * x[j]
            })
        })
        .collect()
}
fn close<S: NumericScalar>(actual: &[S], expected: &[S], multiplier: f64) {
    let epsilon = S::Real::EPSILON.try_to_f64_exact().unwrap();
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            (a - b).abs().try_to_f64_exact().unwrap()
                <= multiplier * epsilon * (1.0 + b.abs().try_to_f64_exact().unwrap()),
            "{a:?} != {b:?}"
        );
    }
}
fn domains<S: NumericScalar>(accuracy: Accuracy) {
    let arithmetic_options = SparseSolveOptions::new().with_accuracy(accuracy);
    let a = [
        scalar::<S>(0., 0.),
        scalar(2., 1.),
        scalar(1., 0.),
        scalar(4., -1.),
        scalar(1., 0.),
        scalar(0., 0.),
        scalar(1., 0.),
        scalar(0., 0.),
        scalar(3., 2.),
    ];
    let csr = matrix(3, &a);
    let q = Permutation::new(vec![2, 0, 1]).unwrap();
    let plan = SparseLuSymbolic::analyze(
        csr.structural_pattern(),
        scioxide_sparse::SparseOrdering::Supplied(&q),
        limits(9),
    )
    .unwrap();
    let lu = SparseLu::factor_scoped(
        plan.clone(),
        csr.view(),
        arithmetic_options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut l = vec![S::ZERO; 9];
    let mut u = l.clone();
    for i in 0..3 {
        l[i * 3 + i] = S::ONE;
    }
    for (i, j, v) in lu.lower_entries() {
        l[i * 3 + j] = v;
    }
    for (i, j, v) in lu.upper_entries() {
        u[i * 3 + j] = v;
    }
    for i in 0..3 {
        for j in 0..3 {
            let reconstructed = (0..3).fold(S::ZERO, |sum, k| sum + l[i * 3 + k] * u[k * 3 + j]);
            close(
                &[reconstructed],
                &[a[lu.row_permutation()[i] * 3 + q.new_to_old()[j]]],
                32.,
            );
        }
    }
    let expected = [scalar::<S>(1., 1.), scalar(-2., 0.), scalar(3., -1.)];
    for op in [
        SparseOperation::Normal,
        SparseOperation::Transpose,
        SparseOperation::Adjoint,
    ] {
        let b = rhs(&a, &expected, op);
        let x = solve!(lu, &b, op, arithmetic_options).unwrap();
        close(&x, &expected, 128.);
        close(&rhs(&a, &x, op), &b, 256.);
        let block: Vec<_> = b.iter().copied().cycle().take(9).collect();
        let mut out = vec![S::ZERO; 9];
        let mut scratch = vec![S::ZERO; lu.workspace_requirements(3).unwrap().scalar_len()];
        lu.solve_block_into(
            BlockVectorView::new(&block, 3, 3).unwrap(),
            op,
            arithmetic_options,
            &mut BlockVectorViewMut::new(&mut out, 3, 3).unwrap(),
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )
        .unwrap();
        for column in out.chunks(3) {
            close(column, &expected, 128.);
        }
        #[cfg(feature = "std")]
        {
            let workers = core::num::NonZeroUsize::new(2).unwrap();
            let pool = scioxide_core::ThreadExecutor::new(workers);
            let sequential = out.clone();
            lu.solve_block_into(
                BlockVectorView::new(&block, 3, 3).unwrap(),
                op,
                arithmetic_options
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers)),
                &mut BlockVectorViewMut::new(&mut out, 3, 3).unwrap(),
                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                ExecutionContext::new(&pool),
            )
            .unwrap();
            assert_eq!(out, sequential);
        }
    }
    let csc = csr.to_csc().unwrap();
    let other = SparseLu::factor_scoped(
        plan,
        csc.view(),
        arithmetic_options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(other.row_permutation(), lu.row_permutation());
    assert_eq!(
        other.lower_entries().collect::<Vec<_>>(),
        lu.lower_entries().collect::<Vec<_>>()
    );
}
#[test]
fn all_scalar_domains() {
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        domains::<f32>(accuracy);
        domains::<f64>(accuracy);
        domains::<Complex<f32>>(accuracy);
        domains::<Complex<f64>>(accuracy);
    }
}

#[test]
fn refactorization_and_failures_are_transactional() {
    let mut a = matrix(2, &[0., 2., 3., 4.]);
    let plan = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        limits(4),
    )
    .unwrap();
    let mut lu = SparseLu::factor(plan.clone(), a.view()).unwrap();
    let mut w = (plan).workspace(()).unwrap();
    lu.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(lu.row_permutation(), &[1, 0]);
    a.copy_values_from(&[5., 2., 0., 4.]).unwrap();
    lu.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(lu.row_permutation(), &[0, 1]);
    assert_eq!(lu.factor_entries(), 3);
    let old = lu.solve(&[7., 4.]).unwrap();
    for values in [
        [0., 0., 0., 0.],
        [f64::NAN, 2., 0., 4.],
        [f64::INFINITY, 2., 0., 4.],
    ] {
        if let Err(error) = a.copy_values_from(&values) {
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
            continue;
        }
        assert!(
            lu.refactor(
                a.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                &mut w,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(lu.solve(&[7., 4.]).unwrap(), old);
    }
    let different = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder.extend(&[0, 1], &[0, 1], &[1., 1.]).and_then(|()| {
            builder.finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        })
    }
    .unwrap();
    assert!(
        lu.refactor(
            different.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    let mut out = [19.; 2];
    assert!(
        lu.solve_into(
            &[1., f64::NAN],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [19.; 2]);
    assert!(
        lu.solve_into(
            &[1., 1.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 3]),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [19.; 2]);
    let mut block_out = [19.; 4];
    assert!(
        lu.solve_block_into(
            BlockVectorView::new(&[1., 1., 1., f64::NAN], 2, 2).unwrap(),
            SparseOperation::Normal,
            Default::default(),
            &mut BlockVectorViewMut::new(&mut block_out, 2, 2).unwrap(),
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 8]),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(block_out, [19.; 4]);
}

#[test]
fn exact_budget_boundaries_empty_and_shapes() {
    let a = matrix(2, &[2., 1., 1., 2.]);
    let pattern = a.structural_pattern();
    let roomy =
        SparseLuSymbolic::analyze(pattern, scioxide_sparse::SparseOrdering::Natural, limits(4))
            .unwrap();
    let bytes = roomy
        .workspace_requirements::<f64>(())
        .unwrap()
        .total_bytes();
    let exact = SparseLuSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Natural,
        SparseLuLimits::new(4, bytes),
    )
    .unwrap();
    SparseLu::<f64>::factor(exact, a.view()).unwrap();
    let short = SparseLuSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Natural,
        SparseLuLimits::new(4, bytes - 1),
    )
    .unwrap();
    assert!(SparseLu::<f64>::factor(short.clone(), a.view()).is_err());
    assert!((short).workspace::<f64>(()).is_err());
    let fill =
        SparseLuSymbolic::analyze(pattern, scioxide_sparse::SparseOrdering::Natural, limits(3))
            .unwrap();
    assert!(SparseLu::factor(fill, a.view()).is_err());
    assert!(
        SparseLuSymbolic::analyze(
            pattern,
            scioxide_sparse::SparseOrdering::Natural,
            SparseLuLimits::new(4, 0)
        )
        .is_err()
    );
    let overflow = SparseLuSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Natural,
        limits(usize::MAX),
    )
    .unwrap();
    assert!(
        matches ! (& (overflow .workspace_requirements::<f64>(())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
    );
    let rectangular = scioxide_core::StructuralPatternView::new(0, 1, &[0], &[]).unwrap();
    assert!(
        SparseLuSymbolic::analyze(
            rectangular,
            scioxide_sparse::SparseOrdering::Natural,
            limits(0)
        )
        .is_err()
    );
    let empty = {
        let mut builder = scioxide_sparse::SparseBuilder::<f64>::new(0, 0);
        builder.extend(&[], &[], &[]).and_then(|()| {
            builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        })
    }
    .unwrap();
    let empty_plan = SparseLuSymbolic::analyze(
        empty.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        limits(0),
    )
    .unwrap();
    let lu = SparseLu::factor(empty_plan, empty.view()).unwrap();
    assert_eq!(lu.factor_entries(), 0);
    assert_eq!(lu.solve(&[]).unwrap(), Vec::<f64>::new());
}

#[test]
fn workspace_admission_matches_capacity_checks() {
    use scioxide_core::{Capacity, SciOxideErrorView};
    // [[2, 1], [1, 2]] with x = [1, 1] gives b = [3, 3].
    let a = matrix(2, &[2., 1., 1., 2.]);
    let analyze = |entries, bytes| {
        SparseLuSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            SparseLuLimits::new(entries, bytes),
        )
        .unwrap()
    };
    let (small, large) = (analyze(4, 1 << 16), analyze(4, 1 << 17));
    let options = SparseSolveOptions::new();
    let scope = ExecutionContext::sequential();
    let mut lu = SparseLu::factor(large.try_clone().unwrap(), a.view()).unwrap();
    // Analyses differing only in their byte ceiling share one reservation.
    let requirements = large.workspace_requirements::<f64>(()).unwrap();
    let mut workspace = small.workspace::<f64>(()).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    workspace.try_reserve(&requirements).unwrap();
    lu.refactor(a.view(), options, &mut workspace, scope)
        .unwrap();
    close(&lu.solve(&[3., 3.]).unwrap(), &[1., 1.], 16.);
    // A different entry reservation is short storage and preserves the factor.
    let mut mismatched = analyze(6, 1 << 16).workspace::<f64>(()).unwrap();
    assert_eq!(
        mismatched.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let error = lu
        .refactor(a.view(), options, &mut mismatched, scope)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch {
            expected: [4],
            actual: [6]
        }
    ));
    close(&lu.solve(&[3., 3.]).unwrap(), &[1., 1.], 16.);
    // The empty system rejects unreserved storage instead of publishing an unshaped factor.
    let empty = CsrMatrix::<f64>::from_raw(0, 0, vec![0], vec![], vec![]).unwrap();
    let plan = SparseLuSymbolic::analyze(
        empty.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        limits(0),
    )
    .unwrap();
    let mut lu = SparseLu::factor(plan, empty.view()).unwrap();
    let mut unreserved = SparseLuWorkspace::<f64>::new();
    let requirements = lu.symbolic().workspace_requirements::<f64>(()).unwrap();
    assert_eq!(
        unreserved.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let error = lu
        .refactor(empty.view(), options, &mut unreserved, scope)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch {
            expected: [1],
            actual: [0]
        }
    ));
    assert_eq!(lu.factor_entries(), 0);
}

#[test]
fn sparse_memory_and_fill_scale_linearly_on_bands() {
    let mut previous = 0;
    for n in [32, 128, 512] {
        let mut b = SparseBuilder::<f64>::new(n, n);
        for i in 0..n {
            b.push(i, i, 4.).unwrap();
            if i + 1 < n {
                b.push(i, i + 1, 1.).unwrap();
                b.push(i + 1, i, -1.).unwrap();
            }
        }
        let a = b
            .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap();
        let plan = SparseLuSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            limits(3 * n - 2),
        )
        .unwrap();
        let lu = SparseLu::factor(plan.clone(), a.view()).unwrap();
        assert_eq!(lu.factor_entries(), 3 * n - 2);
        let bytes = plan
            .workspace_requirements::<f64>(())
            .unwrap()
            .total_bytes();
        assert!(bytes < 328 * n);
        if previous != 0 {
            assert!(bytes < 5 * previous);
        }
        previous = bytes;
    }
}

#[test]
fn extreme_diagonals_and_complex_magnitudes() {
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = SparseSolveOptions::new().with_accuracy(accuracy);
        for scale in [f64::from_bits(1), 1e-300, 1e300, f64::MAX] {
            let a = matrix(2, &[scale, 0., 0., scale]);
            let plan = SparseLuSymbolic::analyze(
                a.structural_pattern(),
                scioxide_sparse::SparseOrdering::Natural,
                limits(2),
            )
            .unwrap();
            let lu = SparseLu::factor_scoped(
                plan.clone(),
                a.view(),
                options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(
                solve!(lu, &[scale, scale], SparseOperation::Normal, options).unwrap(),
                vec![1., 1.]
            );
        }
        let z = Complex::new(f64::MAX, f64::MAX);
        let a = matrix(1, &[z]);
        let plan = SparseLuSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            limits(1),
        )
        .unwrap();
        let lu = SparseLu::factor_scoped(plan, a.view(), options, ExecutionContext::sequential())
            .unwrap();
        close(
            &solve!(lu, &[z], SparseOperation::Normal, options).unwrap(),
            &[Complex::new(1., 0.)],
            8.,
        );
    }
}

#[test]
fn shared_budget_cancellation_and_fill_failure_preserve_epoch() {
    use scioxide_core::{
        Budget, BudgetKind, CancellationToken, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let mut a = matrix(2, &[2., 1., 0., 2.]);
    let plan = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        limits(3),
    )
    .unwrap();
    let mut lu = SparseLu::factor(plan.clone(), a.view()).unwrap();
    let mut w = (plan).workspace(()).unwrap();
    a.set_value(2, 1.).unwrap();
    assert!(
        lu.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(lu.factor_entries(), 3);
    assert_eq!(lu.solve(&[3., 2.]).unwrap(), vec![1., 1.]);
    a.set_value(2, 0.).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        lu.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(486),
        scioxide_core::algorithm_id!("sparse", "LU", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let result = lu.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new()
            .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)),
        &mut w,
        &mut context,
    );
    assert!(
        matches!(&result, Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::ConvergenceFailure),
        "{result:?}"
    );
    assert_eq!(lu.solve(&[3., 2.]).unwrap(), vec![1., 1.]);
    let mut output = [9.; 2];
    assert!(
        lu.solve_into(
            &[3., 2.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
            &mut context
        )
        .is_ok()
    );
    assert_eq!(output, [1.; 2]);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 2);
    lu.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
}

#[cfg(feature = "std")]
#[test]
fn incomplete_executor_never_publishes_block_output() {
    use core::num::NonZeroUsize;
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
    let a = matrix(2, &[2., 1., 0., 2.]);
    let plan = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        limits(3),
    )
    .unwrap();
    let lu = SparseLu::factor(plan, a.view()).unwrap();
    let mut output = [9.; 4];
    let workers = NonZeroUsize::new(2).unwrap();
    assert!(
        lu.solve_block_into(
            BlockVectorView::new(&[3., 2., 3., 2.], 2, 2).unwrap(),
            SparseOperation::Normal,
            SparseSolveOptions::new()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers)),
            &mut BlockVectorViewMut::new(&mut output, 2, 2).unwrap(),
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 8]),
            ExecutionContext::new(&Incomplete)
        )
        .is_err()
    );
    assert_eq!(output, [9.; 4]);
}

#[test]
fn owned_symbolic_lifecycle_reuses_the_existing_kernel() {
    use scioxide_core::ExecutionContext;
    use scioxide_sparse::{
        DuplicatePolicy, ExplicitZeroPolicy, SparseLu, SparseLuLimits, SparseLuSymbolic,
        SparseOperation,
    };
    let matrix = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[4.0_f64, 1., 2., 3.])
            .and_then(|()| {
                builder.finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }
    .unwrap()
    .to_csr()
    .unwrap();
    let symbolic = SparseLuSymbolic::analyze(
        matrix.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        SparseLuLimits::new(4, 4096),
    )
    .unwrap();
    let mut factor = SparseLu::factor(symbolic, matrix.view()).unwrap();
    let mut workspace = (factor.symbolic()).workspace(()).unwrap();
    let mut output = [0.; 2];
    let mut scratch = [0.; 4];
    factor
        .refactor(
            matrix.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    factor
        .solve_into(
            &[5., 5.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.iter().all(|x| (x - 1.).abs() < 1e-12));
    factor
        .solve_into(
            &[6., 4.],
            SparseOperation::Transpose,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.iter().all(|x| (x - 1.).abs() < 1e-12));
}

#[test]
fn factor_epochs_record_the_attempted_prefix_with_batched_or_restricted_admission() {
    use scioxide_core::{
        Budget, BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    for singular in [false, true] {
        for limit in [None, Some(1), Some(3)] {
            let a = matrix(
                3,
                &[
                    1.,
                    0.,
                    0.,
                    0.,
                    if singular { 0. } else { 2. },
                    0.,
                    0.,
                    0.,
                    3.,
                ],
            );
            let symbolic = SparseLuSymbolic::analyze(
                a.structural_pattern(),
                SparseOrdering::Natural,
                limits(32),
            )
            .unwrap();
            let mut workspace = SparseLuWorkspace::prepare(symbolic).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut cx = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    SolveIdentity::new(
                        ProblemToken::new(2368),
                        scioxide_core::algorithm_id!("test", "factor-prefix", 1),
                        3,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            let options = SparseSolveOptions::new().with_budgets(limit.map_or_else(
                SolveBudgets::unbounded,
                |n| {
                    SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Iteration, Budget::try_from(n).unwrap())
                },
            ));
            let result = workspace.factor_scoped(a.view(), options, &mut cx);
            let attempted = if singular { 2 } else { 3 };
            let admitted = limit.unwrap_or(attempted).min(attempted);
            assert_eq!(cx.counts().nested().iterations(), admitted);
            assert_eq!(cx.counts().own().iterations(), 0);
            assert_eq!(result.is_ok(), !singular && admitted == 3);
        }
    }
}
