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
    for position in 0..32 {
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
fn reservations_are_fallible_and_numeric_epochs_execute() {
    let a = CsrMatrix::from_raw(
        4,
        4,
        vec![0, 3, 6, 9, 12],
        vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
        vec![4., 1., 1., 1., 4., 1., 1., 4., 1., 1., 1., 4.],
    )
    .unwrap();
    // The isolated final zero row/column fails at the final block's pivot,
    // after earlier factor rows and scratch have already been populated.
    let bad = CsrMatrix::from_raw(
        4,
        4,
        a.row_ptr().to_vec(),
        a.col_indices().to_vec(),
        a.view()
            .iter()
            .map(|(r, c, v)| if r == 3 || c == 3 { 0. } else { v })
            .collect(),
    )
    .unwrap();
    let limits = IncompleteLimits::new(16, 1_000_000);
    let options = IncompleteOptions::new()
        .with_drop_tolerance(1e-6)
        .with_max_off_diagonal(4);
    for method in [
        IncompleteMethod::Ilu { level: 0 },
        IncompleteMethod::Ilu { level: 1 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ic0,
        IncompleteMethod::Ict,
    ] {
        for b in [1, 2, 4] {
            let pattern = a.structural_pattern();
            fail_each(|| IncompleteSymbolic::analyze(pattern, method, b, limits));
            let plan = IncompleteSymbolic::analyze(pattern, method, b, limits).unwrap();
            fail_each(|| IncompleteFactor::<f64>::factor(plan.try_clone()?, a.view(), options));
            fail_each(|| plan.workspace::<f64>(()));
            // Options are fixed per factor, so each retention cap owns a factor.
            let mut factors: Vec<_> = [0, 4, 1, 4]
                .into_iter()
                .map(|cap| {
                    IncompleteFactor::factor(
                        plan.clone(),
                        a.view(),
                        options.with_max_off_diagonal(cap),
                    )
                    .unwrap()
                })
                .collect();
            let mut w = plan.workspace(()).unwrap();
            let mut out = [0.; 4];
            let mut scratch = [0.; 8];
            let before = CALLS.with(Cell::get);
            for accuracy in [
                scioxide_core::Accuracy::Standard,
                scioxide_core::Accuracy::Exact,
            ] {
                let execution_options = SparseSolveOptions::new().with_accuracy(accuracy);
                for factor in &mut factors {
                    factor
                        .refactor(
                            a.view(),
                            execution_options,
                            &mut w,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                    for op in [
                        SparseOperation::Normal,
                        SparseOperation::Transpose,
                        SparseOperation::Adjoint,
                    ] {
                        factor
                            .solve_into(
                                &[1.; 4],
                                op,
                                execution_options,
                                &mut out,
                                &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                                ExecutionContext::sequential(),
                            )
                            .unwrap();
                    }
                    let previous = out;
                    let diagnostics = factor.diagnostics();
                    assert!(
                        factor
                            .refactor(
                                bad.view(),
                                execution_options,
                                &mut w,
                                ExecutionContext::sequential()
                            )
                            .is_err()
                    );
                    assert_eq!(factor.diagnostics(), diagnostics);
                    factor
                        .solve_into(
                            &[1.; 4],
                            SparseOperation::Adjoint,
                            execution_options,
                            &mut out,
                            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                    assert_eq!(out, previous);
                }
            }
            assert_eq!(CALLS.with(Cell::get), before);
        }
    }
}
