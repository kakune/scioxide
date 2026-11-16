//! Observable allocation-failure coverage for the shared fallible helpers.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ptr;

use scioxide_core::backend::{try_filled, try_reserve, try_with_capacity};
use scioxide_core::{CombinationEnumerator, MultisetEnumerator, Permutation, SciOxideError};

thread_local! {
    static FAIL_NEXT_ALLOCATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
}

struct FailingAllocator;

// SAFETY: successful operations delegate their original layouts directly to
// `System`; only an explicitly armed allocation on the current thread returns
// null, as permitted by `GlobalAlloc`.
unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let fail = FAIL_AFTER.with(|counter| match counter.get() {
            Some(0) => {
                counter.set(None);
                true
            }
            Some(n) => {
                counter.set(Some(n - 1));
                false
            }
            None => false,
        });
        if fail || FAIL_NEXT_ALLOCATION.with(|flag| flag.replace(false)) {
            ptr::null_mut()
        } else {
            // SAFETY: `layout` is the valid layout supplied by the caller.
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: this forwards the pointer and layout supplied for a prior
        // successful `System` allocation.
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn is_allocation_failure(error: SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    )
}

// Fails allocation number `after` of `operation`, counting from zero, and
// checks that the failure propagates; with `after` equal to the number of
// allocations the operation succeeds and leaves the counter at zero.
fn fail_after<T>(after: usize, operation: impl Fn() -> Result<T, SciOxideError>) {
    for attempt in 0..=after {
        FAIL_AFTER.with(|counter| counter.set(Some(attempt)));
        let result = operation();
        if attempt < after {
            assert!(result.is_err_and(is_allocation_failure));
            assert!(FAIL_AFTER.with(|counter| counter.get()).is_none());
        } else {
            assert!(result.is_ok());
            assert_eq!(FAIL_AFTER.with(|counter| counter.replace(None)), Some(0));
        }
    }
}

#[test]
fn owned_normalization_and_reductions_propagate_each_allocation_failure() {
    let x = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let view = scioxide_core::NdView::row_major(&x, &[2, 3]).unwrap();
    fail_after(1, || scioxide_core::softmax(&x));
    fail_after(1, || scioxide_core::log_softmax(&x));
    fail_after(3, || scioxide_core::softmax_axis(view, 1));
    fail_after(3, || scioxide_core::log_softmax_axis(view, 1));
    fail_after(1, || scioxide_core::logsumexp_axis(view, 1));
    fail_after(1, || scioxide_core::weighted_logsumexp_axis(view, view, 1));
    fail_after(1, || scioxide_core::checked_stirling_second_kind(10, 3));
}

#[global_allocator]
static ALLOCATOR: FailingAllocator = FailingAllocator;

fn fail_next_allocation() {
    FAIL_NEXT_ALLOCATION.with(|flag| flag.set(true));
}

fn assert_allocation_failure(
    result: Result<(), SciOxideError>,
    expected_context: scioxide_core::Quantity,
) {
    assert!(
        result.is_err_and(|error| matches!(error.view(), scioxide_core::SciOxideErrorView::AllocationFailure {  context, requested: 8 } if context == expected_context))
    );
}

#[test]
fn every_shared_fallible_allocation_helper_propagates_allocator_failure() {
    fail_next_allocation();
    assert_allocation_failure(
        try_with_capacity::<u8>(8, scioxide_core::quantity!("capacity")).map(|_| ()),
        scioxide_core::quantity!("capacity"),
    );

    let mut values = Vec::<u8>::new();
    fail_next_allocation();
    assert_allocation_failure(
        try_reserve(&mut values, 8, scioxide_core::quantity!("reserve")),
        scioxide_core::quantity!("reserve"),
    );

    fail_next_allocation();
    assert_allocation_failure(
        try_filled(8, 1_u8, scioxide_core::quantity!("filled")).map(|_| ()),
        scioxide_core::quantity!("filled"),
    );
}

#[test]
fn reduction_propagates_allocation_failure() {
    let plan = scioxide_core::ExecutionContext::sequential().plan(
        scioxide_core::ExecutionPolicy::Sequential,
        1,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: core::num::NonZeroUsize::MIN,
            worker_limit: core::num::NonZeroUsize::MIN,
        },
    );
    fail_next_allocation();
    let result = plan.reduce(
        core::iter::empty::<(scioxide_core::WorkPartition, ())>(),
        (),
        |(), ()| Ok(()),
    );
    assert!(matches!(result, Err(error) if matches!(error.view(),
        scioxide_core::SciOxideErrorView::AllocationFailure {  requested: 1, .. })));
}

#[test]
fn owned_combinatorial_state_reports_allocator_failure() {
    fail_next_allocation();
    let combination = CombinationEnumerator::first(8, 8);
    assert!(
        matches!(&combination, Err(error) if match error.view() { scioxide_core::SciOxideErrorView::AllocationFailure {  context: _view_context, requested: _view_requested } => _view_context == scioxide_core::quantity!("combination_state"), _ => false })
    );

    fail_next_allocation();
    let multiset = MultisetEnumerator::first(8, 8);
    assert!(
        matches!(&multiset, Err(error) if match error.view() { scioxide_core::SciOxideErrorView::AllocationFailure {  context: _view_context, requested: _view_requested } => _view_context == scioxide_core::quantity!("multiset_state"), _ => false })
    );

    for (after, expected_context) in [
        (0, scioxide_core::quantity!("identity_permutation")),
        (1, scioxide_core::quantity!("inverse_permutation")),
    ] {
        FAIL_AFTER.with(|counter| counter.set(Some(after)));
        let permutation = Permutation::identity(8);
        assert!(matches!(&permutation, Err(error) if matches!(error.view(),
            scioxide_core::SciOxideErrorView::AllocationFailure {  context, requested: 8 }
            if context == expected_context)));
        assert!(FAIL_AFTER.with(|counter| counter.get()).is_none());
    }
}
