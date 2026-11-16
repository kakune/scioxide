#![allow(clippy::unwrap_used)]
mod amg_support;
use amg_support::*;
use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Budget, BudgetKind, ExecutionContext, LinearOperator,
    Preconditioner, SciOxideError, SolveBudgets,
};
use scioxide_sparse::*;
use std::sync::atomic::{AtomicUsize, Ordering};
#[test]
fn executor_blocks_empty_and_local_budget() {
    let a = grid::<f64>(24, 1, 1.);
    let p = preconditioner(&a, AmgCycleKind::W);
    let rhs: Vec<_> = (0..72).map(|i| (i + 1) as f64 / 72.).collect();
    let mut sequential = vec![0.; 72];
    let mut workspace = p.workspace(3).unwrap();
    p.apply_block_into(
        BlockVectorView::new(&rhs, 24, 3).unwrap(),
        SparseSolveOptions::new(),
        &mut BlockVectorViewMut::new(&mut sequential, 24, 3).unwrap(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for column in 0..3 {
        let mut vector = vec![0.; 24];
        p.apply_into(
            &rhs[24 * column..24 * (column + 1)],
            SparseSolveOptions::new(),
            &mut vector,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(vector, sequential[24 * column..24 * (column + 1)]);
    }
    #[cfg(feature = "std")]
    for worker_count in [1, 2, 3] {
        let workers = core::num::NonZeroUsize::new(worker_count).unwrap();
        let executor = scioxide_core::ThreadExecutor::new(workers);
        let mut out = vec![99.; 72];
        p.apply_block_into(
            BlockVectorView::new(&rhs, 24, 3).unwrap(),
            SparseSolveOptions::new()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers)),
            &mut BlockVectorViewMut::new(&mut out, 24, 3).unwrap(),
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        assert_eq!(out, sequential);
    }
    // One iteration per right-hand side: two columns exceed an allowance of one
    // before any column runs, and the vector form fits it exactly.
    let limited = SparseSolveOptions::new()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
    let mut out = vec![99.; 48];
    let error = p
        .apply_block_into(
            BlockVectorView::new(&rhs[..48], 24, 2).unwrap(),
            limited,
            &mut BlockVectorViewMut::new(&mut out, 24, 2).unwrap(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(out, vec![99.; 48]);
    let mut out = vec![99.; 24];
    p.apply_into(
        &rhs[..24],
        limited,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out, sequential[..24]);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), scioxide_core::AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), 1);
    let empty = grid::<f64>(0, 1, 1.);
    let p = preconditioner(&empty, AmgCycleKind::V);
    p.precondition_into(&[], &mut [], &mut []).unwrap();
}

// Counts every provider query so the test can separate preparation from application.
struct Counting<'a, T: ?Sized> {
    inner: &'a T,
    queries: &'a AtomicUsize,
    applications: &'a AtomicUsize,
}
impl<T: Preconditioner<f64> + ?Sized> Preconditioner<f64> for Counting<'_, T> {
    fn dimension(&self) -> usize {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.inner.dimension()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.inner.precondition_workspace_len()
    }
    fn precondition_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        workspace: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.applications.fetch_add(1, Ordering::Relaxed);
        self.inner.precondition_into(input, output, workspace)
    }
}
struct CountingOperator<'a> {
    inner: &'a (dyn LinearOperator<f64> + Sync),
    queries: &'a AtomicUsize,
    applications: &'a AtomicUsize,
}
impl LinearOperator<f64> for CountingOperator<'_> {
    fn rows(&self) -> usize {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_workspace_len()
    }
    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        workspace: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.applications.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_into(input, output, workspace)
    }
}
#[test]
fn prepared_cycles_never_revalidate_levels_or_requery_providers() {
    let a = grid::<f64>(20, 1, 1.);
    let h = hierarchy(&a);
    let ac = h.coarse_matrix();
    let symbolic = SparseSymmetricSymbolic::analyze(
        ac.structural_pattern(),
        SparseOrdering::default(),
        None,
        coarse_limits(),
    )
    .unwrap();
    let coarse = SparseCholesky::factor(symbolic, ac.view()).unwrap();
    let jacobi: Vec<_> = (0..h.diagnostics().levels())
        .map(|i| JacobiPreconditioner::new(h.matrix(i).unwrap()).unwrap())
        .collect();
    let (queries, applications) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let smoothers: Vec<_> = jacobi
        .iter()
        .map(|inner| Counting {
            inner,
            queries: &queries,
            applications: &applications,
        })
        .collect();
    let operators: Vec<_> = (0..h.diagnostics().levels())
        .map(|i| CountingOperator {
            inner: h.matrix(i).unwrap(),
            queries: &queries,
            applications: &applications,
        })
        .collect();
    let counted_coarse = CountingOperator {
        inner: &coarse,
        queries: &queries,
        applications: &applications,
    };
    let levels: Vec<_> = (0..h.diagnostics().levels())
        .map(|i| {
            let mut level = h.level(i).unwrap();
            level.operator = &operators[i];
            level.smoother = AmgSmootherView::Injected {
                pre: &smoothers[i],
                post: &smoothers[i],
            };
            level
        })
        .collect();
    let cycle = AmgCycle::new(&levels, &counted_coarse, AmgCycleOptions::new()).unwrap();
    let prepared = queries.load(Ordering::Relaxed);
    assert!(prepared > 0);
    let mut workspace = cycle.workspace(1).unwrap();
    let mut scratch = vec![0.; cycle.precondition_workspace_len().unwrap()];
    let mut output = vec![0.; 20];
    queries.store(0, Ordering::Relaxed);
    for _ in 0..4 {
        cycle
            .apply_into(
                &[1.; 20],
                SparseSolveOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        cycle
            .precondition_into(&[1.; 20], &mut output, &mut scratch)
            .unwrap();
    }
    assert_eq!(queries.load(Ordering::Relaxed), 0);
    assert!(applications.load(Ordering::Relaxed) > 0);
}
struct Dense<'a> {
    n: usize,
    values: &'a [f64],
}
impl LinearOperator<f64> for Dense<'_> {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.values[i * self.n..(i + 1) * self.n]
                .iter()
                .zip(x)
                .map(|(&a, &x)| a * x)
                .sum();
        }
        Ok(())
    }
}
#[test]
fn supplied_dense_nonsymmetric_injected_and_failing_providers() {
    let a = Dense {
        n: 2,
        values: &[2., 1., 0., 3.],
    };
    let inverse = Dense {
        n: 2,
        values: &[0.5, -1. / 6., 0., 1. / 3.],
    };
    let unused = AmgSmootherView::Jacobi {
        inverse_diagonal: &[],
        weight: 1.,
    };
    let levels = [AmgLevelView {
        operator: &a,
        prolongation: None,
        restriction: None,
        smoother: unused,
    }];
    let cycle = AmgCycle::new(&levels, &inverse, Default::default()).unwrap();
    let mut output = [99.; 2];
    cycle
        .precondition_into(&[4., 6.], &mut output, &mut [0.; 2])
        .unwrap();
    close(&output, &[1., 2.], 8.);
    struct Bad;
    impl LinearOperator<f64> for Bad {
        fn rows(&self) -> usize {
            2
        }
        fn cols(&self) -> usize {
            2
        }
        fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
            Ok(0)
        }
        fn apply_into(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), scioxide_core::SciOxideError> {
            out.fill(f64::NAN);
            Ok(())
        }
    }
    let cycle = AmgCycle::new(&levels, &Bad, Default::default()).unwrap();
    let before = output;
    assert!(
        cycle
            .precondition_into(&[4., 6.], &mut output, &mut [0.; 2])
            .is_err()
    );
    assert_eq!(output, before);
    assert!(AmgCycle::new(&[], &inverse, Default::default()).is_err());
    let a = grid::<f64>(20, 1, 1.);
    let h = hierarchy(&a);
    let ac = h.coarse_matrix();
    let symbolic = SparseSymmetricSymbolic::analyze(
        ac.structural_pattern(),
        SparseOrdering::default(),
        None,
        coarse_limits(),
    )
    .unwrap();
    let coarse = SparseCholesky::factor(symbolic, ac.view()).unwrap();
    let jacobi: Vec<_> = (0..h.diagnostics().levels())
        .map(|i| JacobiPreconditioner::new(h.matrix(i).unwrap()).unwrap())
        .collect();
    for injected in [false, true] {
        let levels: Vec<_> = (0..h.diagnostics().levels())
            .map(|i| {
                let mut l = h.level(i).unwrap();
                l.smoother = if injected {
                    AmgSmootherView::Injected {
                        pre: &jacobi[i],
                        post: &jacobi[i],
                    }
                } else {
                    AmgSmootherView::GaussSeidel {
                        matrix: h.matrix(i).unwrap(),
                    }
                };
                l
            })
            .collect();
        let c = AmgCycle::new(&levels, &coarse, Default::default()).unwrap();
        let mut out = vec![0.; 20];
        let mut work = vec![0.; c.precondition_workspace_len().unwrap()];
        c.precondition_into(&[1.; 20], &mut out, &mut work).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn v_w_recursion_linearity_and_malformed_transfers() {
    let a = grid::<f64>(16, 1, 1.);
    let h = hierarchy(&a);
    let ac = h.coarse_matrix();
    let symbolic = SparseSymmetricSymbolic::analyze(
        ac.structural_pattern(),
        SparseOrdering::default(),
        None,
        coarse_limits(),
    )
    .unwrap();
    let f = SparseCholesky::factor(symbolic, ac.view()).unwrap();
    let mut levels: Vec<_> = (0..h.diagnostics().levels())
        .map(|i| h.level(i).unwrap())
        .collect();
    // Use a closure operator to count calls while keeping the Sync bound explicit.
    let calls = AtomicUsize::new(0);
    struct Counter<'a> {
        factor: &'a (dyn LinearOperator<f64> + Sync),
        calls: &'a AtomicUsize,
    }
    impl LinearOperator<f64> for Counter<'_> {
        fn rows(&self) -> usize {
            self.factor.rows()
        }
        fn cols(&self) -> usize {
            self.factor.cols()
        }
        fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
            self.factor.apply_workspace_len()
        }
        fn apply_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            work: &mut [f64],
        ) -> Result<(), scioxide_core::SciOxideError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.factor.apply_into(x, out, work)
        }
    }
    let counted = Counter {
        factor: &f,
        calls: &calls,
    };
    for kind in [AmgCycleKind::V, AmgCycleKind::W] {
        let c = AmgCycle::new(&levels, &counted, AmgCycleOptions::new().with_kind(kind)).unwrap();
        let mut out = vec![0.; 16];
        let mut work = vec![0.; c.precondition_workspace_len().unwrap()];
        calls.store(0, Ordering::Relaxed);
        c.precondition_into(&[1.; 16], &mut out, &mut work).unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            if kind == AmgCycleKind::V {
                1
            } else {
                1usize << (levels.len() - 1)
            }
        );
        let first = out.clone();
        c.precondition_into(&[2.; 16], &mut out, &mut work).unwrap();
        close(
            &out,
            &first.iter().map(|x| 2. * x).collect::<Vec<_>>(),
            128.,
        );
    }
    levels[0].restriction = None;
    assert!(AmgCycle::new(&levels, &f, Default::default()).is_err());
}

#[test]
fn setup_admits_the_documented_iterations() {
    // The 3-node chain coarsens to one aggregate. Entry and completion: 2;
    // fine diagonal: 3; four aggregation passes: 12; one aggregate: 1;
    // A T: 3 + 1; smoothing: 3; A P: 3 + 1; R (A P): 1 + 1; coarse diagonal: 1.
    let expected = 2 + 3 + 12 + 1 + 4 + 3 + 4 + 2 + 1;
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![2.0_f64, -1., -1., 2., -1., -1., 2.],
    )
    .unwrap();
    let build = |iterations: usize| {
        AmgHierarchy::build(
            a.view(),
            BlockVectorView::new(&[1.; 3], 3, 1).unwrap(),
            AmgSetupOptions::new().with_coarse_size(1),
            AmgLimits::new(4, 1000, 1 << 20),
            SparseSolveOptions::new().with_budgets(
                SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Iteration, Budget::try_from(iterations).unwrap()),
            ),
            ExecutionContext::sequential(),
        )
    };
    let hierarchy = build(expected).unwrap();
    assert_eq!(hierarchy.diagnostics().levels(), 2);
    let error = build(expected - 1).err().unwrap();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
}
