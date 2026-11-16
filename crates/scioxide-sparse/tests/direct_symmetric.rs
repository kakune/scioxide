//! Original-coordinate tests independent of the sparse elimination kernel.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#[cfg(feature = "std")]
use scioxide_core::SciOxideError;
use scioxide_core::Triangle;
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
fn limits(entries: usize) -> SparseSymmetricLimits {
    SparseSymmetricLimits::new(entries, entries, usize::MAX)
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
fn reconstruct<S: NumericScalar>(
    n: usize,
    entries: impl Iterator<Item = (usize, usize, S)>,
    diagonal: &[S],
    off: Option<&[S]>,
    p: &[usize],
    original: &[S],
) {
    let mut l = vec![S::ZERO; n * n];
    let mut d = vec![S::ZERO; n * n];
    for i in 0..n {
        l[i * n + i] = if off.is_some() { S::ONE } else { diagonal[i] };
        d[i * n + i] = if off.is_some() { diagonal[i] } else { S::ONE };
    }
    if let Some(off) = off {
        for (k, &v) in off.iter().enumerate().take(n.saturating_sub(1)) {
            d[(k + 1) * n + k] = v;
            d[k * n + k + 1] = v.conj();
        }
    }
    for (i, j, v) in entries {
        assert!(i > j);
        l[i * n + j] = v;
    }
    for i in 0..n {
        for j in 0..n {
            let actual = (0..n).fold(S::ZERO, |sum, k| {
                sum + (0..n).fold(S::ZERO, |sum, h| {
                    sum + l[i * n + k] * d[k * n + h] * l[j * n + h].conj()
                })
            });
            close(&[actual], &[original[p[i] * n + p[j]]], 256.);
        }
    }
}
macro_rules! check_solves {
    ($factor:expr,$a:expr,$n:expr,$options:expr) => {{
        let factor = $factor;
        let n = $n;
        let expected: Vec<_> = (0..n).map(|i| scalar::<S>((i + 1) as f64, 0.5)).collect();
        for operation in [
            SparseOperation::Normal,
            SparseOperation::Transpose,
            SparseOperation::Adjoint,
        ] {
            let b = rhs($a, &expected, operation);
            let x = solve!(factor, &b, operation, $options).unwrap();
            close(&x, &expected, 512.);
            close(&rhs($a, &x, operation), &b, 512.);
            let block: Vec<_> = b.iter().copied().cycle().take(3 * n).collect();
            let mut output = vec![S::ZERO; 3 * n];
            let mut scratch = vec![S::ZERO; factor.workspace_requirements(3).unwrap().scalar_len()];
            factor
                .solve_block_into(
                    BlockVectorView::new(&block, n, 3).unwrap(),
                    operation,
                    $options,
                    &mut BlockVectorViewMut::new(&mut output, n, 3).unwrap(),
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            #[cfg(feature = "std")]
            {
                let sequential = output.clone();
                let workers = core::num::NonZeroUsize::new(2).unwrap();
                let pool = scioxide_core::ThreadExecutor::new(workers);
                factor
                    .solve_block_into(
                        BlockVectorView::new(&block, n, 3).unwrap(),
                        operation,
                        $options.with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(
                            workers,
                        )),
                        &mut BlockVectorViewMut::new(&mut output, n, 3).unwrap(),
                        &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                        ExecutionContext::new(&pool),
                    )
                    .unwrap();
                assert_eq!(sequential, output);
            }
            for col in output.chunks(n) {
                close(col, &expected, 512.);
            }
        }
    }};
}
fn domains<S: NumericScalar>(accuracy: Accuracy) {
    let arithmetic_options = SparseSolveOptions::new().with_accuracy(accuracy);
    let n = 6;
    for indefinite in [false, true] {
        // Explicit congruence gives independent inertia, without an eigensolver oracle.
        let mut b = vec![S::ZERO; n * n];
        for i in 0..n {
            b[i * n + i] = scalar(2., 0.);
            if i > 0 {
                b[i * n + i - 1] = scalar(0.5, 0.25);
            }
            if i > 2 {
                b[i * n + i - 3] = scalar(-0.25, 0.5);
            }
        }
        let a: Vec<_> = (0..n * n)
            .map(|pos| {
                let (i, j) = (pos / n, pos % n);
                (0..n).fold(S::ZERO, |sum, k| {
                    sum + b[i * n + k]
                        * scalar(if indefinite && k % 2 == 0 { -1. } else { 1. }, 0.)
                        * b[j * n + k].conj()
                })
            })
            .collect();
        let csr = matrix(n, &a);
        let order = Permutation::new(vec![3, 0, 5, 1, 4, 2]).unwrap();
        let plan = SparseSymmetricSymbolic::analyze(
            csr.structural_pattern(),
            scioxide_sparse::SparseOrdering::Supplied(&order),
            None,
            limits(n * n),
        )
        .unwrap();
        let mut w = (plan).workspace(()).unwrap();
        let mut ldl = SparseLdl::factor_scoped(
            plan.clone(),
            csr.view(),
            arithmetic_options,
            ExecutionContext::sequential(),
        )
        .unwrap();
        reconstruct(
            n,
            ldl.lower_entries(),
            ldl.diagonal(),
            Some(ldl.off_diagonal()),
            ldl.permutation(),
            &a,
        );
        assert_eq!(
            ldl.inertia().unwrap(),
            LdlInertia::new(
                if indefinite { 3 } else { 6 },
                if indefinite { 3 } else { 0 },
                0
            )
            .unwrap()
        );
        check_solves!(&ldl, &a, n, arithmetic_options);
        let csc = csr.to_csc().unwrap();
        ldl.refactor(
            csc.view(),
            arithmetic_options,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
        reconstruct(
            n,
            ldl.lower_entries(),
            ldl.diagonal(),
            Some(ldl.off_diagonal()),
            ldl.permutation(),
            &a,
        );
        if !indefinite {
            let mut chol = SparseCholesky::factor_scoped(
                plan.clone(),
                csr.view(),
                arithmetic_options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            reconstruct(
                n,
                chol.lower_entries(),
                chol.diagonal(),
                None,
                chol.permutation(),
                &a,
            );
            check_solves!(&chol, &a, n, arithmetic_options);
            chol.refactor(
                csc.view(),
                arithmetic_options,
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap();
            reconstruct(
                n,
                chol.lower_entries(),
                chol.diagonal(),
                None,
                chol.permutation(),
                &a,
            );
        }
    }
    let a = [
        S::ZERO,
        scalar::<S>(2., 1.),
        scalar(1., 0.),
        scalar(2., -1.),
        S::ZERO,
        scalar(0., 1.),
        scalar(1., 0.),
        scalar(0., -1.),
        scalar(3., 0.),
    ];
    let csr = matrix(3, &a);
    let plan = SparseSymmetricSymbolic::analyze(
        csr.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(9),
    )
    .unwrap();
    let f = SparseLdl::factor_scoped(
        plan,
        csr.view(),
        arithmetic_options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(f.pivot_blocks().next().unwrap().size(), 2);
    reconstruct(
        3,
        f.lower_entries(),
        f.diagonal(),
        Some(f.off_diagonal()),
        f.permutation(),
        &a,
    );
    check_solves!(&f, &a, 3, arithmetic_options);
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
fn empty_and_rank_loss() {
    let a = matrix::<f64>(0, &[]);
    let plan = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(0),
    )
    .unwrap();
    let f = SparseLdl::factor(plan.clone(), a.view()).unwrap();
    assert_eq!(f.inertia().unwrap(), LdlInertia::default());
    assert!(f.solve(&[]).unwrap().is_empty());
    let c = SparseCholesky::factor(plan, a.view()).unwrap();
    assert!(c.diagonal().is_empty());
    let a = matrix(3, &[2., 0., 0., 0., -1., 0., 0., 0., 0.]);
    let plan = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(9),
    )
    .unwrap();
    let f = SparseLdl::factor(plan.clone(), a.view()).unwrap();
    assert_eq!(f.inertia().unwrap(), LdlInertia::new(1, 1, 1).unwrap());
    let mut out = [42.; 3];
    assert!(
        matches ! (& (f .solve_into(& [1. ; 3], SparseOperation :: Normal, scioxide_sparse::SparseSolveOptions::new(), & mut out, &mut scioxide_sparse::SparseSolveWorkspace::borrowed(& mut [0. ; 6]), ExecutionContext :: sequential ())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: Singular { .. }))
    );
    assert_eq!(out, [42.; 3]);
    assert!(SparseCholesky::factor(plan, a.view()).is_err());
}
#[test]
fn tree_fill_and_sparse_scaling() {
    for n in [1, 16, 512] {
        let mut b = SparseBuilder::new(n, n);
        for i in 0..n {
            b.push(i, i, 4.).unwrap();
            if i > 0 {
                b.push(i, i - 1, -1.).unwrap();
            }
        }
        let a = b
            .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap();
        let plan = SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            Some(Triangle::Lower),
            limits(2 * n),
        )
        .unwrap();
        assert_eq!(plan.cholesky_entries(), n - 1);
        assert_eq!(
            plan.elimination_tree().collect::<Vec<_>>(),
            (0..n)
                .map(|i| (i + 1 < n).then_some(i + 1))
                .collect::<Vec<_>>()
        );
        assert!(
            plan.workspace_requirements::<f64>(())
                .unwrap()
                .total_bytes()
                < 1000 * n
        );
        let f = SparseCholesky::factor(plan.clone(), a.view()).unwrap();
        assert_eq!(f.factor_entries(), n - 1);
        let f = SparseLdl::factor(plan.clone(), a.view()).unwrap();
        assert_eq!(f.factor_entries(), n - 1);
    }
}
#[test]
fn exact_structure_triangles_and_rollback() {
    let mut a = matrix(2, &[4., 1., 1., 3.]);
    let plan = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4),
    )
    .unwrap();
    let mut c = SparseCholesky::factor(plan.clone(), a.view()).unwrap();
    let mut l = SparseLdl::factor(plan.clone(), a.view()).unwrap();
    let mut w = (plan).workspace(()).unwrap();
    let expected = c.solve(&[5., 4.]).unwrap();
    a.set_value(0, -4.).unwrap();
    assert!(
        c.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(expected, c.solve(&[5., 4.]).unwrap());
    l.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(l.inertia().unwrap().negative(), 1);
    a.set_value(1, 2.).unwrap();
    assert!(
        l.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(l.inertia().unwrap().negative(), 1);
    a.set_value(0, 4.).unwrap();
    assert_eq!(
        a.set_value(1, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    let lower = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        Some(Triangle::Lower),
        limits(4),
    )
    .unwrap();
    SparseCholesky::factor(lower, a.view()).unwrap();
    let mut upper = matrix(2, &[4., 1., 1., 3.]);
    assert_eq!(
        upper.set_value(2, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    let p = SparseSymmetricSymbolic::analyze(
        upper.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        Some(Triangle::Upper),
        limits(4),
    )
    .unwrap();
    SparseLdl::factor(p, upper.view()).unwrap();
    let complex = matrix(1, &[Complex::new(1., 1.)]);
    let p = SparseSymmetricSymbolic::analyze(
        complex.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(1),
    )
    .unwrap();
    assert!(SparseLdl::factor(p, complex.view()).is_err());
}
#[test]
fn exact_resource_admission() {
    let a = matrix(2, &[4., 1., 1., 3.]);
    let pattern = a.structural_pattern();
    let plan = SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4),
    )
    .unwrap();
    let complex = CsrMatrix::from_raw(
        a.rows(),
        a.cols(),
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        a.values().iter().map(|&v| Complex::new(v, 0.)).collect(),
    )
    .unwrap();
    let bytes = plan
        .workspace_requirements::<Complex<f64>>(())
        .unwrap()
        .total_bytes();
    for offset in [0, 1] {
        let mut lim = limits(4);
        lim = lim.with_max_workspace_bytes(bytes - offset);
        let p = SparseSymmetricSymbolic::analyze(
            pattern,
            scioxide_sparse::SparseOrdering::Natural,
            None,
            lim,
        )
        .unwrap();
        assert_eq!(
            SparseLdl::<Complex<f64>>::factor(p.clone(), complex.view()).is_ok(),
            offset == 0
        );
        assert_eq!(p.workspace::<Complex<f64>>(()).is_ok(), offset == 0);
    }
    assert!(
        SparseSymmetricSymbolic::analyze(
            pattern,
            scioxide_sparse::SparseOrdering::Natural,
            None,
            limits(0)
        )
        .is_err()
    );
    let mut lim = limits(4);
    lim = lim.with_max_active_entries(2);
    let p = SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Natural,
        None,
        lim,
    )
    .unwrap();
    assert!(SparseLdl::factor(p, a.view()).is_err());
    // An unbounded entry cap does not size the analysis; the factor reservation overflows.
    let p = SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4).with_max_factor_entries(usize::MAX),
    )
    .unwrap();
    let error = SparseLdl::factor(p, a.view()).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}
#[test]
fn analysis_storage_follows_structure_not_entry_limits() {
    // Tridiagonal [[4, 1, 0], [1, 4, 1], [0, 1, 4]] has two strict-lower fill entries.
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0_f64, 1., 1., 4., 1., 1., 4.],
    )
    .unwrap();
    let analyze = |entries, bytes| {
        SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            None,
            SparseSymmetricLimits::new(entries, 8, bytes),
        )
    };
    let symbolic = |plan: &SparseSymmetricSymbolic| {
        plan.workspace_requirements::<f64>(())
            .unwrap()
            .symbolic_bytes()
    };
    let exact = analyze(2, usize::MAX).unwrap();
    assert_eq!(exact.cholesky_entries(), 2);
    // Retained analysis payload is independent of a generous factor-entry cap.
    let generous = analyze(1 << 20, usize::MAX).unwrap();
    assert_eq!(generous.cholesky_entries(), 2);
    assert_eq!(symbolic(&generous), symbolic(&exact));
    // A cap far beyond addressable memory is neither reserved nor charged by analysis.
    let huge = usize::MAX / 64;
    for bytes in [usize::MAX, 1 << 16] {
        let plan = analyze(huge, bytes).unwrap();
        assert_eq!(plan.cholesky_entries(), 2);
        assert!(plan.try_clone().is_ok());
    }
    // The cap still bounds structural fill during analysis.
    let error = analyze(1, usize::MAX).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ResourceLimit {
            required: 2,
            limit: 1,
            ..
        }
    ));
    // Factor limits are admitted when the numeric reservation is created.
    let error = SparseCholesky::factor(analyze(huge, 1 << 16).unwrap(), a.view()).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ResourceLimit { limit: 65_536, .. }
    ));
}
#[test]
fn extreme_scales_and_explicit_threshold() {
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = SparseSolveOptions::new().with_accuracy(accuracy);
        for scale in [1e-300, 1e300] {
            let a = matrix(2, &[0., scale, scale, 0.]);
            let p = SparseSymmetricSymbolic::analyze(
                a.structural_pattern(),
                scioxide_sparse::SparseOrdering::Natural,
                None,
                limits(4),
            )
            .unwrap();
            let f = SparseLdl::factor_scoped(
                p.clone(),
                a.view(),
                options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(f.inertia().unwrap(), LdlInertia::new(1, 1, 0).unwrap());
            close(
                &solve!(f, &[scale, scale], SparseOperation::Normal, options).unwrap(),
                &[1., 1.],
                16.,
            );
            let a = matrix(2, &[scale, 0., 0., scale]);
            let p = SparseSymmetricSymbolic::analyze(
                a.structural_pattern(),
                scioxide_sparse::SparseOrdering::Natural,
                None,
                limits(4),
            )
            .unwrap();
            let f = SparseCholesky::factor_scoped(
                p.clone(),
                a.view(),
                options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            close(
                &solve!(f, &[scale, scale], SparseOperation::Normal, options).unwrap(),
                &[1., 1.],
                16.,
            );
        }
        let a = matrix(2, &[1., 0., 0., -1e-20]);
        let p = SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            None,
            limits(4),
        )
        .unwrap();
        let f =
            SparseLdl::factor_scoped(p, a.view(), options, ExecutionContext::sequential()).unwrap();
        assert_eq!(f.inertia().unwrap().negative(), 1);
        assert_eq!(
            f.inertia_with_threshold(scioxide_core::RankThreshold::new(1e-10, 0.).unwrap(), 1.)
                .unwrap()
                .zero(),
            1
        );
    }
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
    let a = matrix(2, &[2., 1., 1., 2.]);
    let plan = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4),
    )
    .unwrap();
    let lu = SparseLdl::factor(plan, a.view()).unwrap();
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
fn shared_context_cancellation_and_output_transactions() {
    use scioxide_core::{
        Budget, BudgetKind, CancellationToken, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let a = matrix(2, &[4., 1., 1., 3.]);
    let p = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4),
    )
    .unwrap();
    let mut f = SparseLdl::factor(p.clone(), a.view()).unwrap();
    let mut c = SparseCholesky::factor(p.clone(), a.view()).unwrap();
    let mut w = (p).workspace(()).unwrap();
    let identity = SolveIdentity::new(
        ProblemToken::new(487),
        scioxide_core::algorithm_id!("sparse", "symmetric", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(
        matches ! (& (f .refactor(a . view (), scioxide_sparse::SparseSolveOptions::new().with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)), & mut w, & mut context)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ConvergenceFailure { .. }))
    );
    let mut out = [9.; 2];
    assert!(
        f.solve_into(
            &[5., 4.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
            &mut context
        )
        .is_ok()
    );
    assert_eq!(out, [1.; 2]);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 2);
    out.fill(9.);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(
        c.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new().with_budgets(
                SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE)
            ),
            &mut w,
            &mut context
        )
        .is_err()
    );
    let cancel = CancellationToken::new();
    cancel.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&cancel);
    assert!(
        f.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            context
        )
        .is_err()
    );
    assert!(
        c.refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            context
        )
        .is_err()
    );
    assert!(
        f.solve_into(
            &[5., 4.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
            context
        )
        .is_err()
    );
    assert_eq!(out, [9.; 2]);
    assert!(
        f.solve_into(
            &[5., f64::NAN],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [9.; 2]);
    assert!(
        c.solve_into(
            &[5., 4.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 3]),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [9.; 2]);
    close(&f.solve(&[5., 4.]).unwrap(), &[1., 1.], 16.);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    f.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        &mut context,
    )
    .unwrap();
    c.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut w,
        &mut context,
    )
    .unwrap();
    f.solve_into(
        &[5., 4.],
        SparseOperation::Normal,
        scioxide_sparse::SparseSolveOptions::new(),
        &mut out,
        &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
        &mut context,
    )
    .unwrap();
    close(&out, &[1., 1.], 16.);
}

#[test]
fn all_small_graphs_have_independent_symbolic_fill() {
    // Dense boolean graph elimination is independent of elimination-tree reach.
    let n = 5;
    for mask in 0..1024 {
        let mut graph = vec![false; n * n];
        let mut builder = SparseBuilder::new(n, n);
        let mut bit = 0;
        for i in 0..n {
            builder.push(i, i, 8.).unwrap();
            for j in 0..i {
                if mask & (1 << bit) != 0 {
                    graph[i * n + j] = true;
                    graph[j * n + i] = true;
                    builder.push(i, j, -1.).unwrap();
                }
                bit += 1;
            }
        }
        let a = builder
            .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap();
        let p = SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            Some(Triangle::Lower),
            limits(25),
        )
        .unwrap();
        let mut parents = vec![None; n];
        let mut count = 0;
        for k in 0..n {
            let neighbors: Vec<_> = (k + 1..n).filter(|&i| graph[i * n + k]).collect();
            parents[k] = neighbors.first().copied();
            count += neighbors.len();
            for &i in &neighbors {
                for &j in &neighbors {
                    if i != j {
                        graph[i * n + j] = true;
                    }
                }
            }
        }
        assert_eq!(p.elimination_tree().collect::<Vec<_>>(), parents);
        assert_eq!(p.cholesky_entries(), count);
        let f = SparseCholesky::factor(p.clone(), a.view()).unwrap();
        assert_eq!(f.factor_entries(), count);
    }
}

#[test]
fn numeric_pattern_and_workspace_mismatches_preserve_factors() {
    let a = matrix(2, &[4., 1., 1., 3.]);
    let p = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4),
    )
    .unwrap();
    let mut f = SparseLdl::factor(p.clone(), a.view()).unwrap();
    let mut w = (p).workspace(()).unwrap();
    let b = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder.extend(&[0, 1], &[0, 1], &[4., 3.]).and_then(|()| {
            builder.finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        })
    }
    .unwrap();
    assert!(
        f.refactor(
            b.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    let p2 = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(8),
    )
    .unwrap();
    let mut incompatible = (p2).workspace(()).unwrap();
    // A different reserved entry capacity is short storage, not invalid input.
    let error = f
        .refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut incompatible,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch {
            expected: [4],
            actual: [8]
        }
    ));
    close(&f.solve(&[5., 4.]).unwrap(), &[1., 1.], 16.);
    let f = SparseCholesky::<f64>::factor(p, a.view()).unwrap();
    assert_eq!(f.diagonal().len(), 2);
    assert!(f.solve(&[1.; 2]).is_ok());
}

#[test]
fn workspace_admission_matches_capacity_checks() {
    use scioxide_core::{Capacity, SciOxideErrorView};
    // [[4, 1], [1, 3]] with x = [1, 1] gives b = [5, 4].
    let a = matrix(2, &[4., 1., 1., 3.]);
    let analyze = |bytes| {
        SparseSymmetricSymbolic::analyze(
            a.structural_pattern(),
            scioxide_sparse::SparseOrdering::Natural,
            None,
            SparseSymmetricLimits::new(4, 4, bytes),
        )
        .unwrap()
    };
    let (small, large) = (analyze(1 << 16), analyze(1 << 17));
    let options = SparseSolveOptions::new();
    let scope = ExecutionContext::sequential();
    let mut ldl = SparseLdl::factor(large.try_clone().unwrap(), a.view()).unwrap();
    let mut cholesky = SparseCholesky::factor(large.try_clone().unwrap(), a.view()).unwrap();
    // Analyses differing only in their byte ceiling share one reservation.
    let requirements = large.workspace_requirements::<f64>(()).unwrap();
    let mut workspace = small.workspace::<f64>(()).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    workspace.try_reserve(&requirements).unwrap();
    ldl.refactor(a.view(), options, &mut workspace, scope)
        .unwrap();
    cholesky
        .refactor(a.view(), options, &mut workspace, scope)
        .unwrap();
    for solution in [
        ldl.solve(&[5., 4.]).unwrap(),
        cholesky.solve(&[5., 4.]).unwrap(),
    ] {
        close(&solution, &[1., 1.], 16.);
    }
    // Unreserved storage needs growth and is rejected as short, preserving the factor.
    let mut empty = SparseSymmetricWorkspace::<f64>::new();
    assert_eq!(empty.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    let error = ldl
        .refactor(a.view(), options, &mut empty, scope)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch {
            expected: [2],
            actual: [0]
        }
    ));
    close(&ldl.solve(&[5., 4.]).unwrap(), &[1., 1.], 16.);
    // A different active-entry reservation is equally short storage.
    let other = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        SparseSymmetricLimits::new(4, 6, 1 << 16),
    )
    .unwrap();
    let mut mismatched = other.workspace::<f64>(()).unwrap();
    assert_eq!(
        mismatched.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    let error = cholesky
        .refactor(a.view(), options, &mut mismatched, scope)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch {
            expected: [4],
            actual: [6]
        }
    ));
    // Growth repairs it for the requested analysis.
    mismatched.try_reserve(&requirements).unwrap();
    cholesky
        .refactor(a.view(), options, &mut mismatched, scope)
        .unwrap();
}
#[test]
fn pivot_dependent_fill_failure_preserves_epoch() {
    let n = 10;
    let mut builder = SparseBuilder::new(n, n);
    for i in 0..n {
        builder.push(i, i, 40.).unwrap();
        if i + 1 < n {
            builder.push(i, n - 1, 1.).unwrap();
            builder.push(n - 1, i, 1.).unwrap();
        }
    }
    let mut a = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let mut lim = limits(n * n);
    lim = lim.with_max_factor_entries(n - 1);
    let p = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        lim,
    )
    .unwrap();
    assert_eq!(p.cholesky_entries(), n - 1);
    let mut f = SparseLdl::factor(p.clone(), a.view()).unwrap();
    let mut w = (p).workspace(()).unwrap();
    let rhs: Vec<_> = (0..n).map(|i| if i + 1 == n { 49. } else { 41. }).collect();
    let coordinates: Vec<_> = a.view().iter().map(|(i, j, _)| (i, j)).collect();
    for (pos, (i, j)) in coordinates.into_iter().enumerate() {
        if i == j {
            a.set_value(pos, if i + 1 == n { 0. } else { 0.125 })
                .unwrap();
        }
    }
    assert!(
        matches ! (& (f .refactor(a . view (), scioxide_sparse::SparseSolveOptions::new(), & mut w, ExecutionContext :: sequential ())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ResourceLimit { .. }))
    );
    close(&f.solve(&rhs).unwrap(), &vec![1.; n], 16.);
    // Numeric pivots form a dense trailing clique, independently exercising active storage.
    let mut lim = limits(n * n);
    lim = lim.with_max_active_entries(2 * n - 1);
    let p = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        lim,
    )
    .unwrap();
    assert!(
        matches ! (& (SparseLdl::factor(p, a.view())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ResourceLimit { .. }))
    );
}
#[test]
fn all_sparse_storage_views_and_inverse_operators() {
    use scioxide_core::{AdjointOperator, BlockOperator, LinearOperator};
    let a = matrix(2, &[4., 1., 1., 3.]);
    let csc = a.to_csc().unwrap();
    let coo = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[4., 1., 1., 3.])
            .and_then(|()| {
                builder.finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }
    .unwrap();
    let bsr = BsrMatrix::from_sparse(a.view(), 2, 2).unwrap();
    let dia = DiaMatrix::from_sparse(a.view()).unwrap();
    let p = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::Natural,
        None,
        limits(4),
    )
    .unwrap();
    for view in [a.view(), csc.view(), coo.view(), bsr.view(), dia.view()] {
        let f = SparseLdl::factor(p.clone(), view).unwrap();
        let c = SparseCholesky::factor(p.clone(), view).unwrap();
        for operator in [&f as &dyn LinearOperator<f64>, &c] {
            let mut out = [0.; 2];
            let mut scratch = vec![0.; operator.apply_workspace_len().unwrap()];
            operator
                .apply_into(&[5., 4.], &mut out, &mut scratch)
                .unwrap();
            close(&out, &[1., 1.], 16.);
        }
        let mut out = [0.; 2];
        f.apply_adjoint_into(&[5., 4.], &mut out, &mut [0.; 4])
            .unwrap();
        close(&out, &[1., 1.], 16.);
        let mut out = [0.; 4];
        c.apply_block_into(
            BlockVectorView::new(&[5., 4., 10., 8.], 2, 2).unwrap(),
            &mut BlockVectorViewMut::new(&mut out, 2, 2).unwrap(),
            &mut [0.; 8],
        )
        .unwrap();
        close(&out, &[1., 1., 2., 2.], 16.);
        let mut empty = [];
        c.solve_block_into(
            BlockVectorView::new(&[], 2, 0).unwrap(),
            SparseOperation::Normal,
            Default::default(),
            &mut BlockVectorViewMut::new(&mut empty, 2, 0).unwrap(),
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut []),
            ExecutionContext::sequential(),
        )
        .unwrap();
    }
}
