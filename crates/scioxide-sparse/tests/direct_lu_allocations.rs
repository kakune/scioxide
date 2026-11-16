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
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[4., 1., 2., 3.])
            .and_then(|()| {
                builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }
    .unwrap();
    let limits = SparseLuLimits::new(4, usize::MAX);
    let pattern = a.structural_pattern();
    fail_each(|| {
        SparseLuSymbolic::analyze(pattern, scioxide_sparse::SparseOrdering::default(), limits)
    });
    let plan =
        SparseLuSymbolic::analyze(pattern, scioxide_sparse::SparseOrdering::default(), limits)
            .unwrap();
    fail_each(|| SparseLu::<f64>::factor(plan.try_clone()?, a.view()));
    fail_each(|| (plan).workspace::<f64>(()));
    let mut lu = SparseLu::factor(plan.clone(), a.view()).unwrap();
    let mut workspace = (plan).workspace(()).unwrap();
    let mut out = [0.; 2];
    let mut scratch = [0.; 4];
    let before = CALLS.with(Cell::get);
    for accuracy in [
        scioxide_core::Accuracy::Standard,
        scioxide_core::Accuracy::Exact,
        scioxide_core::Accuracy::Standard,
    ] {
        let execution_options = SparseSolveOptions::new().with_accuracy(accuracy);
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
    assert_eq!(CALLS.with(Cell::get), before);
}

#[test]
fn initial_factor_reservation_retries_without_allocation() {
    let a =
        CsrMatrix::from_raw(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![4., 1., 2., 3.]).unwrap();
    let symbolic = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        SparseLuLimits::new(4, 4096),
    )
    .unwrap();
    fail_each(|| SparseLuWorkspace::<f64>::prepare(symbolic.try_clone()?));
    let mut workspace = SparseLuWorkspace::prepare(symbolic).unwrap();
    let mut singular = a.clone();
    singular
        .copy_values_from(&vec![0.; singular.values().len()])
        .unwrap();
    let before = CALLS.with(Cell::get);
    assert!(
        workspace
            .factor_scoped(
                singular.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(workspace.initial_symbolic().is_some());
    let mut factor = workspace
        .factor_scoped(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(workspace.initial_symbolic().is_none());
    factor
        .refactor(
            a.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let mut output = [0.; 2];
    factor
        .solve_into(
            &[5., 5.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut [0.; 4]),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(CALLS.with(Cell::get), before);
    assert_eq!(output, [1., 1.]);
}
