//! Fallible preparation and allocation-free numerical epochs.
#![allow(clippy::unwrap_used, unsafe_code)]
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_sparse::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
thread_local! {
    static FAIL: Cell<Option<usize>> = const { Cell::new(None) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct Allocator;
// SAFETY: forwards valid allocation layouts to System unless failure is injected.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|c| c.set(c.get() + 1));
        let fail = FAIL.with(|c| match c.get() {
            Some(0) => {
                c.set(None);
                true
            }
            Some(n) => {
                c.set(Some(n - 1));
                false
            }
            None => false,
        });
        if fail {
            std::ptr::null_mut()
        } else {
            // SAFETY: layout is supplied under the GlobalAlloc contract.
            unsafe { System.alloc(layout) }
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: successful allocations above came from System with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn fail_each<T>(mut operation: impl FnMut() -> Result<T, SciOxideError>) {
    let mut successes = 0;
    for position in 0..128 {
        FAIL.with(|c| c.set(Some(position)));
        let result = operation();
        FAIL.with(|c| c.set(None));
        match result {
            Ok(_) => {
                successes += 1;
                break;
            }
            Err(error) => assert!(
                matches ! (& (error) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: AllocationFailure { .. })),
                "{error:?}"
            ),
        }
    }
    assert_eq!(successes, 1);
}
#[test]
fn every_preparation_allocation_is_fallible_and_epochs_execute() {
    let a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[4., 1., 1., 3.])
            .and_then(|()| {
                builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }
    .unwrap();
    let limits = SparseSymmetricLimits::new(4, 4, usize::MAX);
    let pattern = a.structural_pattern();
    fail_each(|| {
        SparseSymmetricSymbolic::analyze(
            pattern,
            scioxide_sparse::SparseOrdering::default(),
            None,
            limits,
        )
    });
    let plan = SparseSymmetricSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::default(),
        None,
        limits,
    )
    .unwrap();
    fail_each(|| SparseLdl::<f64>::factor(plan.try_clone()?, a.view()));
    fail_each(|| (plan).workspace::<f64>(()));
    fail_each(|| SparseCholesky::<f64>::factor(plan.try_clone()?, a.view()));
    let mut chol = SparseCholesky::factor(plan.clone(), a.view()).unwrap();
    let mut lu = SparseLdl::factor(plan.clone(), a.view()).unwrap();
    let mut workspace = (plan).workspace(()).unwrap();
    let mut out = [0.; 2];
    let mut scratch = [0.; 4];
    let indefinite = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[0., 1., 1., 0.])
            .and_then(|()| {
                builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }
    .unwrap();
    let before = CALLS.with(Cell::get);
    for accuracy in [
        scioxide_core::Accuracy::Standard,
        scioxide_core::Accuracy::Exact,
        scioxide_core::Accuracy::Standard,
    ] {
        let execution_options = SparseSolveOptions::new().with_accuracy(accuracy);
        chol.refactor(
            a.view(),
            execution_options,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        chol.solve_into(
            &[5., 4.],
            SparseOperation::Normal,
            execution_options,
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )
        .unwrap();
        lu.refactor(
            a.view(),
            execution_options,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for op in [
            SparseOperation::Normal,
            SparseOperation::Transpose,
            SparseOperation::Adjoint,
        ] {
            lu.solve_into(
                &[5., 5.],
                op,
                execution_options,
                &mut out,
                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                ExecutionContext::sequential(),
            )
            .unwrap();
        }
    }
    for (matrix, count) in [(&indefinite, 1), (&a, 2), (&indefinite, 1)] {
        lu.refactor(
            matrix.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(lu.pivot_blocks().count(), count);
        assert_eq!(lu.inertia().unwrap().total(), 2);
        lu.solve_into(
            &[1., 1.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut out,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )
        .unwrap();
    }
    assert_eq!(CALLS.with(Cell::get), before);
}

#[test]
fn failed_fill_epochs_preserve_the_factor_and_reuse_storage_without_allocation() {
    let n = 10;
    let mut builder = SparseBuilder::<f64>::new(n, n);
    for i in 0..n {
        builder.push(i, i, 40.).unwrap();
        if i + 1 < n {
            builder.push(i, n - 1, 1.).unwrap();
            builder.push(n - 1, i, 1.).unwrap();
        }
    }
    let good = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let bad = CsrMatrix::from_raw(
        n,
        n,
        good.row_ptr().to_vec(),
        good.col_indices().to_vec(),
        good.view()
            .iter()
            .map(|(i, j, value)| {
                if i != j {
                    value
                } else if i + 1 == n {
                    0.
                } else {
                    0.125
                }
            })
            .collect(),
    )
    .unwrap();
    let rhs: Vec<_> = (0..n).map(|i| if i + 1 == n { 49. } else { 41. }).collect();
    for factor_limit in [true, false] {
        let limits = SparseSymmetricLimits::new(
            if factor_limit { n - 1 } else { n * n },
            if factor_limit { n * n } else { 2 * n - 1 },
            usize::MAX,
        );
        let plan = SparseSymmetricSymbolic::analyze(
            good.structural_pattern(),
            SparseOrdering::Natural,
            None,
            limits,
        )
        .unwrap();
        let mut f = SparseLdl::factor(plan.clone(), good.view()).unwrap();
        let mut workspace = (plan).workspace(()).unwrap();
        let permutation = f.permutation().to_vec();
        let diagonal = f.diagonal().to_vec();
        let off = f.off_diagonal().to_vec();
        let entries: Vec<_> = f.lower_entries().collect();
        let blocks: Vec<_> = f.pivot_blocks().collect();
        let inertia = f.inertia().unwrap();
        let mut output = vec![0.; n];
        let mut scratch = vec![0.; f.workspace_requirements(1).unwrap().scalar_len()];
        let before = CALLS.with(Cell::get);
        for accuracy in [
            scioxide_core::Accuracy::Standard,
            scioxide_core::Accuracy::Exact,
            scioxide_core::Accuracy::Standard,
        ] {
            let execution_options = SparseSolveOptions::new().with_accuracy(accuracy);
            let error = f
                .refactor(
                    bad.view(),
                    execution_options,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            let (expected_resource, expected_limit) = if factor_limit {
                (scioxide_core::quantity!("factor_entries"), n - 1)
            } else {
                (scioxide_core::quantity!("active_entries"), 2 * n - 1)
            };
            assert!(matches!(error.view(),
                scioxide_core::SciOxideErrorView::ResourceLimit { resource, required, limit }
                    if resource == expected_resource && required == expected_limit + 1 && limit == expected_limit
            ));
            assert_eq!(f.permutation(), permutation);
            assert_eq!(f.diagonal(), diagonal);
            assert_eq!(f.off_diagonal(), off);
            assert!(f.lower_entries().eq(entries.iter().copied()));
            assert!(f.pivot_blocks().eq(blocks.iter().copied()));
            assert_eq!(f.inertia().unwrap(), inertia);
            f.solve_into(
                &rhs,
                SparseOperation::Normal,
                execution_options,
                &mut output,
                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert!(
                output
                    .iter()
                    .all(|value| (value - 1.).abs() < 32. * f64::EPSILON)
            );
            f.refactor(
                good.view(),
                execution_options,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        }
        assert_eq!(CALLS.with(Cell::get), before);
    }
}
