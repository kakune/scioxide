//! Fallible preparation and allocation-free numerical epochs.
#![allow(clippy::unwrap_used, unsafe_code)]
use scioxide_core::{BlockVectorView, ExecutionContext, Preconditioner};
mod amg_support;
use amg_support::*;
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

#[test]
fn allocation_failures_and_zero_allocation_reuse() {
    let a = grid::<f64>(24, 1, 1.);
    let modes = vec![1.; 24];
    let build = || {
        AmgHierarchy::build(
            a.view(),
            BlockVectorView::new(&modes, 24, 1).unwrap(),
            AmgSetupOptions::new().with_coarse_size(4),
            limits(),
            SparseSolveOptions::new(),
            ExecutionContext::sequential(),
        )
    };
    CALLS.with(|c| c.set(0));
    let h = build().unwrap();
    let calls = CALLS.with(Cell::get);
    for position in 0..calls {
        FAIL.with(|c| c.set(Some(position)));
        let result = build();
        FAIL.with(|c| c.set(None));
        assert!(
            matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: AllocationFailure { .. })),
            "allocation {position}"
        );
    }
    let p = AmgPreconditioner::build(
        h,
        AmgCycleOptions::new(),
        coarse_limits(),
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut work = vec![0.; p.precondition_workspace_len().unwrap()];
    let mut out = vec![0.; 24];
    CALLS.with(|c| c.set(0));
    for _ in 0..8 {
        p.precondition_into(&modes, &mut out, &mut work).unwrap();
    }
    assert_eq!(CALLS.with(Cell::get), 0);
}
