#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};
use errors::error_kind;

use scioxide_core::{
    Complex, ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor, NdView, NdViewMut,
    PartitionFn, SciOxideError, SciOxideErrorKind, SequentialExecutor, StridedLayout,
};
use scioxide_signal::{AnalyticSignalNdPlan, RieszPlan, SignalExecutionOptions};

struct RecordingExecutor {
    calls: AtomicUsize,
    fail_after: usize,
}
impl Executor for RecordingExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if call >= self.fail_after {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("injected_executor"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "fixture failure",
            ));
        }
        SequentialExecutor.execute(context, plan, operation)
    }
}
#[test]
fn prepared_allocations_executor_and_failure_transactions() {
    // Real transforms use half-length complex children; retain parallel coverage.
    let shape = [131_072];
    let strides = [1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let x: Vec<_> = (0..shape[0])
        .map(|i| (core::f64::consts::TAU * i as f64 / shape[0] as f64).cos())
        .collect();
    let executor = RecordingExecutor {
        calls: AtomicUsize::new(0),
        fail_after: usize::MAX,
    };
    let context = ExecutionContext::new(&executor);
    let options = SignalExecutionOptions::new()
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let a = AnalyticSignalNdPlan::build(&shape, &[0], options, context).unwrap();
    let r = RieszPlan::build(&shape, &[0], &[1.], options, context).unwrap();
    let mut aw = a.workspace(()).unwrap();
    let mut rw = r.workspace(()).unwrap();
    let mut z = vec![Complex::new(0., 0.); shape[0]];
    let mut y = vec![0.; shape[0] * 2];
    let vector_shape = [1, shape[0]];
    let mono_shape = [2, shape[0]];
    let vector_strides = [shape[0], 1];
    let vector_layout = StridedLayout::new(&vector_shape, &vector_strides).unwrap();
    let mono_layout = StridedLayout::new(&mono_shape, &vector_strides).unwrap();
    let complex: Vec<_> = x.iter().map(|&v| Complex::new(v, 0.5 * v)).collect();
    let mut complex_out = z.clone();
    {
        a.analytic_into(
            NdView::new(&x, layout).unwrap(),
            NdViewMut::new(&mut z, layout).unwrap(),
            &mut aw,
            context,
        )
        .unwrap();
        r.riesz_into(
            NdView::new(&complex, layout).unwrap(),
            NdViewMut::new(&mut complex_out, vector_layout).unwrap(),
            &mut rw,
            context,
        )
        .unwrap();
        r.monogenic_into(
            NdView::new(&x, layout).unwrap(),
            NdViewMut::new(&mut y, mono_layout).unwrap(),
            &mut rw,
            context,
        )
        .unwrap();
    };
    assert!(executor.calls.load(Ordering::Relaxed) >= 6);
    let sequential = RieszPlan::new(&shape, &[0], &[1.], SignalExecutionOptions::new())
        .unwrap()
        .monogenic(NdView::new(&x, layout).unwrap())
        .unwrap();
    assert!(
        y.iter()
            .zip(&sequential)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    let before = y.clone();
    r.monogenic_into(
        NdView::new(&x, layout).unwrap(),
        NdViewMut::new(&mut y, mono_layout).unwrap(),
        &mut rw,
        context,
    )
    .unwrap();
    assert_eq!(y, before);
    // Forward succeeds; inverse execution fails, after computation has begun.
    let failing = RecordingExecutor {
        calls: AtomicUsize::new(0),
        fail_after: 1,
    };
    assert_eq!(
        error_kind(r.monogenic_into(
            NdView::new(&x, layout).unwrap(),
            NdViewMut::new(&mut y, mono_layout).unwrap(),
            &mut rw,
            ExecutionContext::new(&failing)
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(y, before);
    let before = z.clone();
    let failing = RecordingExecutor {
        calls: AtomicUsize::new(0),
        fail_after: 1,
    };
    assert_eq!(
        error_kind(a.analytic_into(
            NdView::new(&x, layout).unwrap(),
            NdViewMut::new(&mut z, layout).unwrap(),
            &mut aw,
            ExecutionContext::new(&failing)
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(z, before);
}
#[test]
fn workspace_requirements_reserve_and_limit() {
    use scioxide_core::Capacity;
    use scioxide_signal::SignalWorkspace;
    let a =
        AnalyticSignalNdPlan::<f64>::new(&[8, 3], &[0, 1], SignalExecutionOptions::new()).unwrap();
    let r = RieszPlan::new(&[8, 3], &[0, 1], &[1., 2.], SignalExecutionOptions::new()).unwrap();
    let analytic = a.workspace_requirements(()).unwrap();
    let riesz = r.workspace_requirements(()).unwrap();
    assert_eq!((analytic.values_len(), analytic.real_len()), (0, 24));
    assert_eq!(riesz.real_len(), 24);
    // One workspace reserved for both plans serves either.
    let mut ws = SignalWorkspace::new();
    ws.try_reserve(&analytic).unwrap();
    assert_eq!(ws.check(&riesz).unwrap(), Capacity::NeedsGrowth);
    ws.try_reserve(&riesz).unwrap();
    assert_eq!(
        ws.check(&analytic.max(riesz)).unwrap(),
        Capacity::Sufficient
    );
    let x: Vec<_> = (0..24).map(|i| (i as f64 * 0.7).sin()).collect();
    let shape = [8, 3];
    let strides = [3, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let mut z = vec![Complex::new(0., 0.); 24];
    a.analytic_into(
        NdView::new(&x, layout).unwrap(),
        NdViewMut::new(&mut z, layout).unwrap(),
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(z, a.analytic(NdView::new(&x, layout).unwrap()).unwrap());
    let before = z.clone();
    let mut small = SignalWorkspace::new();
    assert_eq!(
        a.analytic_into(
            NdView::new(&x, layout).unwrap(),
            NdViewMut::new(&mut z, layout).unwrap(),
            &mut small,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(z, before);
    assert_eq!(
        ws.check(&riesz.with_max_bytes(riesz.bytes::<f64>().unwrap() - 1))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}
