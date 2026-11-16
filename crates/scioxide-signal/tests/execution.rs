#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#[macro_use]
#[path = "support/views.rs"]
mod views;
use scioxide_core::{
    CancellationToken, ExecutionContext, ExecutionStopReason, SciOxideError, SciOxideErrorView,
};

use scioxide_signal::{
    ConvolutionExtent, ConvolutionKernel, ConvolutionNdPlan, ConvolutionOptions, SignalBoundary,
};
#[test]
fn cancelled_convolution_preserves_output() -> Result<(), SciOxideError> {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&cancellation);
    let kernel = [0.5, -1.];
    let convolution_plan =
        ConvolutionNdPlan::new(&[3], view!(&kernel), &[0], ConvolutionOptions::default())?;
    let mut convolution_output = [43.; 4];
    let mut workspace = convolution_plan.workspace(ConvolutionKernel::Prepared)?;
    let convolution_error = convolution_plan
        .convolve_into(
            view!(&[1., 2., 3.]),
            view_mut!(&mut convolution_output),
            &mut workspace,
            context,
        )
        .expect_err("cancelled convolution must stop");
    assert_cancelled(convolution_error);
    assert_eq!(convolution_output, [43.0; 4]);
    Ok(())
}
fn assert_cancelled(error: SciOxideError) {
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::Cancelled,
            ..
        }
    ));
}

#[test]
fn fft_consumers_use_the_injected_executor() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{
        ExecutionPlan, ExecutionPolicy, Executor, PartitionFn, SequentialExecutor,
    };
    struct CountingExecutor(AtomicUsize);
    impl Executor for CountingExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).expect("two is nonzero")
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            SequentialExecutor.execute(context, plan, operation)
        }
    }
    // Even real FFTs reduce to half-length complex transforms; reach the
    // documented 65,536-complex-element executor threshold.
    let len = 131_072;
    let executor = CountingExecutor(AtomicUsize::new(0));
    let context = ExecutionContext::new(&executor);
    let input: Vec<f64> = (0..len).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut kernel = vec![0.; len];
    kernel[0] = 1.;
    let plan = ConvolutionNdPlan::build(
        &[len],
        view!(&kernel),
        &[0],
        ConvolutionOptions::new()
            .with_extent(ConvolutionExtent::Circular)
            .with_boundary(SignalBoundary::Periodic)
            .with_execution_policy(ExecutionPolicy::Parallel(
                NonZeroUsize::new(2).expect("two"),
            )),
        context,
    )?;
    let mut workspace = plan.workspace(ConvolutionKernel::Prepared)?;
    executor.0.store(0, Ordering::Relaxed);
    let mut output = vec![0.0; len];
    plan.convolve_into(
        view!(&input),
        view_mut!(&mut output),
        &mut workspace,
        context,
    )?;
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    let analytic = scioxide_signal::AnalyticSignalNdPlan::build(
        &[len],
        &[0],
        scioxide_signal::SignalExecutionOptions::new().with_execution_policy(
            ExecutionPolicy::Parallel(NonZeroUsize::new(2).expect("two")),
        ),
        context,
    )?;
    let mut analytic_scratch = analytic.workspace(())?;
    let mut analytic_output = vec![scioxide_core::Complex::new(0., 0.); len];
    executor.0.store(0, Ordering::Relaxed);
    analytic.analytic_into(
        view!(&input),
        view_mut!(&mut analytic_output),
        &mut analytic_scratch,
        context,
    )?;
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    assert!(
        analytic_output
            .iter()
            .zip(&input)
            .all(|(z, x)| (z.re - x).abs() < 1e-10)
    );

    assert!(
        output
            .iter()
            .zip(&input)
            .all(|(a, b)| (a - b).abs() < 1e-10)
    );
    Ok(())
}

#[test]
fn chebyshev_window_build_is_sequential_under_any_executor() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn, SequentialExecutor};
    use scioxide_signal::{Window, WindowKind, WindowNormalization, WindowSymmetry};
    struct CountingExecutor(AtomicUsize);
    impl Executor for CountingExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).expect("two workers")
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            SequentialExecutor.execute(context, plan, operation)
        }
    }
    let executor = CountingExecutor(AtomicUsize::new(0));
    let built = Window::<f64>::build(
        WindowKind::Chebyshev { attenuation: 100.0 },
        65_536,
        WindowSymmetry::Symmetric,
        WindowNormalization::UnitPeak,
        ExecutionContext::new(&executor),
    )?;
    assert_eq!(executor.0.load(Ordering::Relaxed), 0);
    let sequential = Window::<f64>::new(
        WindowKind::Chebyshev { attenuation: 100.0 },
        65_536,
        WindowSymmetry::Symmetric,
        WindowNormalization::UnitPeak,
    )?;
    assert!(
        built
            .samples()
            .iter()
            .zip(sequential.samples())
            .all(|(a, b)| (a - b).abs() < 2e-12)
    );
    Ok(())
}

#[test]
fn analytic_cancellation_preserves_output() -> Result<(), SciOxideError> {
    use scioxide_signal::AnalyticSignalNdPlan;
    let plan =
        AnalyticSignalNdPlan::new(&[8], &[0], scioxide_signal::SignalExecutionOptions::new())?;
    let mut scratch = plan.workspace(())?;
    let mut output = [43.; 8];
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = plan
        .envelope_into(
            view!(&[1.; 8]),
            view_mut!(&mut output),
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .expect_err("cancelled analytic transform");
    assert_cancelled(error);
    assert_eq!(output, [43.; 8]);
    Ok(())
}

#[test]
fn whittaker_solves_stay_within_the_plan_policy() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{
        ExecutionPlan, ExecutionPolicy, Executor, NdView, NdViewMut, PartitionFn,
        SequentialExecutor, StridedLayout,
    };
    use scioxide_signal::{SignalExecutionOptions, WhittakerHendersonPlan};
    struct CountingExecutor(AtomicUsize);
    impl Executor for CountingExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).expect("four workers")
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            SequentialExecutor.execute(context, plan, operation)
        }
    }
    let (n, lanes) = (64, 256);
    let input: Vec<f64> = (0..n * lanes)
        .map(|i| ((i * 37 % 101) as f64 - 50.0) / 25.0)
        .collect();
    let (shape, strides) = ([n, lanes], [1, n]);
    let layout = StridedLayout::new(&shape, &strides)?;
    let mut smoothed = Vec::new();
    for policy in [ExecutionPolicy::Sequential, ExecutionPolicy::Auto] {
        let options = SignalExecutionOptions::new().with_execution_policy(policy);
        let plan = WhittakerHendersonPlan::<f64>::new(n, 2, 10.0, None, options)?;
        let mut workspace = plan.workspace(lanes)?;
        let mut output = vec![0.0; n * lanes];
        let executor = CountingExecutor(AtomicUsize::new(0));
        plan.smooth_axis_into(
            NdView::new(&input, layout)?,
            0,
            NdViewMut::new(&mut output, layout)?,
            &mut workspace,
            ExecutionContext::new(&executor),
        )?;
        let dispatched = executor.0.load(Ordering::Relaxed);
        match policy {
            ExecutionPolicy::Sequential => assert_eq!(dispatched, 0),
            _ => assert!(dispatched > 0),
        }
        smoothed.push(output);
    }
    assert_eq!(smoothed[0], smoothed[1]);
    Ok(())
}
#[test]
fn vector_strength_deadline_after_admission_preserves_all_output() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::time::Duration;
    use scioxide_core::{CapabilityError, Clock, Deadline, MonotonicInstant};
    use scioxide_signal::{VectorStrength, vector_strength_into};
    struct AdvancingClock(AtomicUsize);
    impl Clock for AdvancingClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(self.0.fetch_add(1, Ordering::Relaxed) as u64),
            ))
        }
    }
    let clock = AdvancingClock(AtomicUsize::new(0));
    let binding = scioxide_core::ClockBinding::new(&clock);
    let context = ExecutionContext::sequential()
        .with_deadline(Deadline::at(
            &binding,
            MonotonicInstant::from_duration_since_epoch(Duration::from_secs(3)),
        ))
        .unwrap();
    let mut output = [VectorStrength::<f64>::default(); 3];
    let original = output;
    let error = vector_strength_into(&[0.0, 1.0], &[1.0, 2.0], &mut output, context).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::DeadlineReached,
            ..
        }
    ));
    assert_eq!(output, original);
}
