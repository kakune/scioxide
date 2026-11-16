//! Fault-injected allocation errors remain typed and leave caller state reusable.
#![allow(clippy::unwrap_used, clippy::panic)]
mod support;

use support::assert_error;

use scioxide_core::{RankThreshold, SciOxideError};
use scioxide_interp::{
    BSpline, BSplineBasis, BarycentricInterpolator, ExtrapolationPolicy, Interpolant,
    NevilleInterpolator, RbfInterpolator, RbfKernel, RbfOptions,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! { static REMAINING: Cell<usize> = const { Cell::new(usize::MAX) }; }
struct FaultAllocator;
fn admitted() -> bool {
    REMAINING.with(|remaining| match remaining.get() {
        usize::MAX => true,
        0 => false,
        count => {
            remaining.set(count - 1);
            true
        }
    })
}
// SAFETY: Successful requests delegate unchanged layouts/pointers to System;
// rejected requests return null as required by GlobalAlloc. Fault state is thread-local.
unsafe impl GlobalAlloc for FaultAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if admitted() {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.alloc(layout) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if admitted() {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.alloc_zeroed(layout) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if admitted() {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.realloc(pointer, layout, size) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from `System` through this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: FaultAllocator = FaultAllocator;
fn fail_after<T>(count: usize, call: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            REMAINING.with(|state| state.set(usize::MAX));
        }
    }
    REMAINING.with(|state| state.set(count));
    let _reset = Reset;
    call()
}

#[test]
fn newton_insertion_faults_every_allocation_without_changing_the_source() {
    use scioxide_interp::NewtonInterpolator;

    let source =
        NewtonInterpolator::new(vec![0.0_f64, 1.0], vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
    let mut workspace = source.workspace().unwrap();
    let mut output = [0.0; 2];
    let mut succeeded = false;
    let mut failures = 0;
    for limit in 0..32 {
        // Both the source and incoming owned value exist before faults begin.
        let value = vec![5.0, 8.0];
        let result = fail_after(limit, || source.insert_point(2.0, value));
        source
            .evaluate_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(source.nodes(), &[0.0, 1.0]);
        assert_eq!(source.coefficients(), &[1.0, 2.0, 1.0, 2.0]);
        assert_eq!(output, [1.5, 3.0]);
        match result {
            Err(error) => {
                assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::AllocationFailure
                );
                failures += 1;
            }
            Ok(extended) => {
                assert_eq!(extended.nodes(), &[0.0, 1.0, 2.0]);
                assert_eq!(
                    extended
                        .evaluate([2.0], ExtrapolationPolicy::Reject)
                        .unwrap(),
                    vec![5.0, 8.0]
                );
                succeeded = true;
                break;
            }
        }
    }
    assert!(failures > 0 && succeeded);
}

#[test]
fn all_method_workspace_reservations_are_transactional() {
    // Tableau families allocate their query workspace once, from the model.
    let nodes: Vec<f64> = (0_u8..12).map(f64::from).collect();
    let neville = NevilleInterpolator::new(nodes.clone(), nodes.clone()).unwrap();
    let rational = scioxide_interp::BulirschStoerInterpolator::new(
        nodes.clone(),
        nodes.iter().map(|x| 1.0 / (1.0 + x)).collect(),
    )
    .unwrap();
    assert_error(
        &fail_after(0, || neville.workspace()),
        scioxide_core::SciOxideErrorKind::AllocationFailure,
    );
    assert_error(
        &fail_after(0, || rational.workspace()),
        scioxide_core::SciOxideErrorKind::AllocationFailure,
    );
    assert!(neville.workspace().is_ok() && rational.workspace().is_ok());
}

#[test]
fn barycentric_construction_reports_each_allocation_failure() {
    let mut failures = 0;
    for limit in 0..16 {
        let nodes = vec![0.0_f64, 1.0, 2.0];
        let values = vec![0.0, 1.0, 4.0];
        match fail_after(limit, || BarycentricInterpolator::new(nodes, values)) {
            Err(ref error)
                if error.kind() == scioxide_core::SciOxideErrorKind::AllocationFailure =>
            {
                failures += 1
            }
            Ok(interpolator) => {
                assert!(
                    (interpolator
                        .evaluate([0.5], ExtrapolationPolicy::Reject)
                        .unwrap()
                        - 0.25)
                        .abs()
                        < 1e-13
                );
                break;
            }
            other => panic!("unexpected construction outcome: {other:?}"),
        }
    }
    assert!(failures >= 3);
}

#[test]
fn failed_knot_insertion_preserves_the_spline() {
    let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1).unwrap();
    let spline = BSpline::new([basis], vec![1.0, 3.0]).unwrap();
    for limit in 0..2 {
        let error = fail_after(limit, || spline.insert_knot(0.5));
        assert_error(&error, scioxide_core::SciOxideErrorKind::AllocationFailure);
        assert_eq!(
            spline.evaluate([0.5], ExtrapolationPolicy::Reject).unwrap(),
            2.0
        );
        assert_eq!(spline.bases()[0].control_count(), 2);
    }
}

#[test]
fn rbf_construction_has_no_infallible_allocation_path() {
    let points = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let values = [1.0, 2.0, 3.0];
    let options = RbfOptions::new().with_rank_threshold(RankThreshold::new(0.0, 1e-12).unwrap());
    let mut succeeded = false;
    for limit in 0..64 {
        // Inputs are owned before fault injection begins.
        let (owned_points, owned_values) = (points.to_vec(), values.to_vec());
        match fail_after(limit, move || {
            RbfInterpolator::with_options(
                owned_points,
                owned_values,
                RbfKernel::Gaussian(1.0),
                options,
            )
        }) {
            Err(ref error)
                if error.kind() == scioxide_core::SciOxideErrorKind::AllocationFailure => {}
            Ok(_) => {
                succeeded = true;
                break;
            }
            other => panic!("unexpected construction outcome: {other:?}"),
        }
    }
    assert!(succeeded);
}

#[test]
fn prepared_spline_calculus_is_fallible_and_queries_execute() {
    use scioxide_interp::{CubicSpline, KnotSide, SplineBoundary};
    let spline = BSpline::new(
        [BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2).unwrap()],
        vec![0.0, 0.5, 1.0, 1.5],
    )
    .unwrap();
    let cubic = CubicSpline::with_boundary(
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.25, 1.0],
        SplineBoundary::Natural,
    )
    .unwrap();
    let preparations: [&dyn Fn() -> Result<
        scioxide_interp::PreparedSplineAntiderivative<f64>,
        SciOxideError,
    >; 2] = [&|| spline.prepare_antiderivative(), &|| {
        cubic.prepare_antiderivative()
    }];
    for prepare in preparations {
        let mut completed = false;
        for count in 0..64 {
            match fail_after(count, prepare) {
                Err(error) => assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::AllocationFailure
                ),
                Ok(prepared) => {
                    let result = fail_after(0, || {
                        prepared.integral(0.125, 0.875, ExtrapolationPolicy::Reject)
                    });
                    assert!(result.unwrap().is_finite());
                    completed = true;
                    break;
                }
            }
        }
        assert!(completed);
    }
    let mut workspace = spline.workspace().unwrap();
    let mut slope = 0.0;
    fail_after(0, || {
        spline.derivative_into(
            [0.5],
            [1],
            Some([KnotSide::Left]),
            ExtrapolationPolicy::Reject,
            &mut slope,
            &mut workspace,
        )
    })
    .unwrap();
    assert!(slope.is_finite());
}

#[test]
fn batch_reservation_failure_preserves_usable_storage() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionContext, ExecutionPolicy, NdView, NdViewMut, StridedLayout};
    use scioxide_interp::{BatchEvaluator, BatchRequest, NewtonInterpolator};
    let model = NewtonInterpolator::new(
        vec![0., 1., 2.],
        vec![vec![1., 2.], vec![2., 4.], vec![3., 6.]],
    )
    .unwrap();
    let policy = ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap());
    let plan = BatchEvaluator::new(&model, policy);
    let mut w = plan.workspace(BatchRequest::values(1)).unwrap();
    let larger = plan
        .workspace_requirements(BatchRequest::values(10).with_reports())
        .unwrap();
    let mut failures = 0;
    for limit in 0..8 {
        match fail_after(limit, || w.try_reserve(&larger)) {
            Err(ref error)
                if error.kind() == scioxide_core::SciOxideErrorKind::AllocationFailure =>
            {
                failures += 1
            }
            Ok(()) => break,
            result => panic!("unexpected reservation outcome: {result:?}"),
        }
        let queries = NdView::new(&[0.5], StridedLayout::new(&[1, 1], &[1, 1]).unwrap()).unwrap();
        let mut out = [0.; 2];
        let mut view =
            NdViewMut::new(&mut out, StridedLayout::new(&[1, 2], &[2, 1]).unwrap()).unwrap();
        plan.evaluate_into(
            &queries,
            ExtrapolationPolicy::Reject,
            &mut view,
            None,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(out, [1.5, 3.]);
    }
    // Staged values and staged reports each fail once before reservation succeeds.
    assert!(failures >= 2);
}

#[test]
fn spline_primitive_preparation_reports_allocation_failures() {
    let spline = BSpline::new(
        [BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3).unwrap()],
        vec![[1.0, 2.0]; 5],
    )
    .unwrap();
    let mut succeeded = false;
    for limit in 0..64 {
        match fail_after(limit, || spline.prepare_antiderivative()) {
            Ok(primitive) => {
                assert_eq!(
                    primitive
                        .integral(0.0, 1.0, ExtrapolationPolicy::Reject)
                        .unwrap(),
                    [1.0, 2.0]
                );
                succeeded = true;
                break;
            }
            Err(error) => {
                assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::AllocationFailure
                );
            }
        }
    }
    assert!(succeeded);
}

#[test]
fn spherical_grouping_preparation_reports_allocation_failures() {
    use scioxide_interp::{SphericalSamples, SphericalSplinePlan, SplineFitLimits};
    let pi = core::f64::consts::PI;
    let samples: Vec<_> = (0..=4)
        .flat_map(|i| (0..=8).map(move |j| [pi * i as f64 / 4.0, pi * j as f64 / 4.0]))
        .collect();
    let mut succeeded = false;
    for limit in 0..256 {
        match fail_after(limit, || {
            SphericalSplinePlan::new(
                &[0.0, pi],
                &[0.0, pi / 2.0, pi, 1.5 * pi, 2.0 * pi],
                SphericalSamples::Scattered(&samples),
                None,
                0.0,
                SplineFitLimits::new(),
            )
        }) {
            Ok(plan) => {
                assert!(
                    plan.fit(&vec![1.0; samples.len()])
                        .unwrap()
                        .report()
                        .residual()
                        < 1e-25
                );
                succeeded = true;
                break;
            }
            Err(error) => {
                assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::AllocationFailure
                );
            }
        }
    }
    assert!(succeeded);
}
