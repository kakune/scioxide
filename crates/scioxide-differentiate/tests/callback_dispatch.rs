//! Plain callbacks bypass executors; parallel callbacks use injected execution.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{
    ColoringKind, ColoringSchedule, Complex64, ExecutionContext, ExecutionFailureReason,
    ExecutionPlan, ExecutionPolicy, Executor, PartitionFn, ProcessorFeature, SciOxideError,
    SciOxideErrorKind, SolveContext, StructuralPattern,
};
use scioxide_differentiate::*;

struct RejectExecutor(AtomicUsize);
impl Executor for RejectExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn auto_min_work_per_worker(&self) -> NonZeroUsize {
        NonZeroUsize::new(32_768).unwrap()
    }
    fn execute(
        &self,
        _: ExecutionContext<'_>,
        _: ExecutionPlan,
        _: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Err(SciOxideError::execution_failure(
            None,
            ExecutionFailureReason::ProtocolViolation,
            "injected executor failure",
        ))
    }
}
fn real(x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    y[0] = x[0] * x[0];
    Ok(())
}
fn complex(x: &[Complex64], y: &mut [Complex64]) -> Result<(), SciOxideError> {
    y[0] = x[0] * x[0];
    Ok(())
}

#[test]
fn sequential_and_small_auto_levels_bypass_executor_and_conserve_parent_counts() {
    use scioxide_core::{EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    for policy in [ExecutionPolicy::Sequential, ExecutionPolicy::Auto] {
        let executor = RejectExecutor(AtomicUsize::new(0));
        let method = AdaptiveDifference::default().with_execution(policy);
        let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
        let mut work = AdaptiveDifferenceWorkspace::<f64>::workspace(request).unwrap();
        for limit in [1, 2, 3, 13, 100] {
            let identity = SolveIdentity::new(
                ProblemToken::new(957),
                scioxide_core::algorithm_id!("test", "inline-difference", 1),
                0,
                0,
            );
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Evaluation,
                scioxide_core::Budget::try_from(limit).unwrap(),
            );
            let mut parent = state
                .begin_solve(ExecutionContext::new(&executor), identity, &mut budgets)
                .unwrap();
            let mut output = [99.0];
            let mut errors = [99.0];
            let calls = AtomicUsize::new(0);
            let report = method
                .gradient_into(
                    Parallel(|x: &[f64], y: &mut [f64]| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        y[0] = x[0].sin();
                        Ok(())
                    }),
                    &[1.0],
                    None,
                    &mut output,
                    &mut errors,
                    &mut work,
                    &mut parent,
                )
                .unwrap();
            assert_eq!(report.evaluations(), calls.load(Ordering::Relaxed));
            assert_eq!(parent.counts().own().provider(EvaluationRole::Function), 0);
            assert_eq!(
                parent.counts().nested().provider(EvaluationRole::Function),
                report.evaluations()
            );
            assert_eq!(parent.counts().nested().iterations(), report.iterations());
            assert_eq!(report.evaluations(), limit.min(13));
            if limit < 13 {
                assert!(!report.output_available());
                assert_eq!((output, errors), ([99.0], [99.0]));
            } else {
                assert!(report.output_available());
                assert!((output[0] - 1.0_f64.cos()).abs() < 1e-14);
            }
        }
        assert_eq!(executor.0.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn fixed_and_node_families_dispatch_only_parallel_callbacks() {
    let executor = RejectExecutor(AtomicUsize::new(0));
    let policy = ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap());
    let fixed = FiniteDifference::default().with_execution(policy);
    let request = DerivativeRequest::new(&fixed, DerivativeShape::gradient(1));
    let mut fw = FiniteDifferenceWorkspace::workspace(request).unwrap();
    let mut out = [99.0];
    assert!(
        fixed
            .gradient_into(
                real,
                &[1.0],
                None,
                None,
                &mut out,
                &mut fw,
                ExecutionContext::new(&executor)
            )
            .unwrap()
            .output_available()
    );
    assert_eq!(executor.0.load(Ordering::Relaxed), 0);
    out = [99.0];
    assert_eq!(
        fixed
            .gradient_into(
                Parallel(real),
                &[1.0],
                None,
                None,
                &mut out,
                &mut fw,
                ExecutionContext::new(&executor)
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionFailure
    );
    assert_eq!(out, [99.0]);
    let request = DerivativeRequest::new(&fixed, DerivativeShape::jvp(1, 1));
    let mut jw = FiniteDifferenceWorkspace::workspace(request).unwrap();
    assert_eq!(
        fixed
            .jvp_into(
                Parallel(real),
                &[1.0],
                &[1.0],
                None,
                None,
                &mut out,
                &mut jw,
                ExecutionContext::new(&executor)
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionFailure
    );
    assert_eq!(out, [99.0]);
    let physical = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[1.0],
        &[&[0.5, 1.0, 1.5]],
        1,
        NodeDifferenceOptions::default().with_execution(policy),
    )
    .unwrap();
    let mut pw = physical.workspace(()).unwrap();
    assert_eq!(
        physical
            .evaluate_into(
                Parallel(real),
                None,
                &mut out,
                &mut pw,
                ExecutionContext::new(&executor)
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionFailure
    );
    assert_eq!(out, [99.0]);
    let relative = RelativeDifference::new(
        DerivativeOperation::First,
        &[&[-0.5, 0.0, 0.5]],
        1,
        NodeDifferenceOptions::default().with_execution(policy),
    )
    .unwrap();
    let mut rw = relative.workspace(()).unwrap();
    assert_eq!(
        relative
            .evaluate_into(
                Parallel(real),
                &[1.0],
                None,
                &mut out,
                &mut rw,
                ExecutionContext::new(&executor)
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionFailure
    );
    assert_eq!(out, [99.0]);
    assert_eq!(executor.0.load(Ordering::Relaxed), 4);
}

#[test]
fn complex_and_directional_callbacks_use_the_same_executor_protocol() {
    let executor = RejectExecutor(AtomicUsize::new(0));
    let method = ComplexStep::default()
        .with_step(1e-100)
        .with_execution(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
    let mut work = ComplexStepWorkspace::workspace(request).unwrap();
    let mut out = [99.0];
    let report = method
        .gradient_into(
            complex,
            &[1.0],
            &mut out,
            &mut work,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert!(report.output_available());
    assert_eq!(out, [2.0]);
    assert_eq!(executor.0.load(Ordering::Relaxed), 0);
    out = [99.0];
    assert_eq!(
        method
            .gradient_into(
                Parallel(complex),
                &[1.0],
                &mut out,
                &mut work,
                ExecutionContext::new(&executor)
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionFailure
    );
    assert_eq!(out, [99.0]);
    let request = DerivativeRequest::new(&method, DerivativeShape::jvp(1, 1));
    let mut work = ComplexStepWorkspace::workspace(request).unwrap();
    assert_eq!(
        method
            .jvp_into(
                Parallel(complex),
                &[1.0],
                &[1.0],
                &mut out,
                &mut work,
                ExecutionContext::new(&executor)
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ExecutionFailure
    );
    assert_eq!(out, [99.0]);
    assert_eq!(executor.0.load(Ordering::Relaxed), 2);
}

const FEATURES: [ProcessorFeature; 7] = [
    ProcessorFeature::X86Sse2,
    ProcessorFeature::X86Avx2,
    ProcessorFeature::X86Fma,
    ProcessorFeature::X86Avx512F,
    ProcessorFeature::X86Avx512Dq,
    ProcessorFeature::X86Avx512Vl,
    ProcessorFeature::Aarch64Neon,
];

/// The processor features that `context` lets kernels use.
fn processor_features(context: ExecutionContext<'_>) -> [bool; 7] {
    FEATURES.map(|feature| context.processor().supports(feature))
}

#[test]
fn owned_forms_pass_the_native_context_to_contextual_callbacks() {
    let native = processor_features(ExecutionContext::sequential_native());
    let portable = processor_features(ExecutionContext::sequential());
    // With `std` the native context permits the features the processor has.
    #[cfg(all(
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ))]
    assert_ne!(native, portable);
    let calls = &Cell::new(0);
    let real = |expected: [bool; 7]| {
        Contextual(
            move |x: &[f64], y: &mut [f64], cx: &mut SolveContext<'_, '_>| {
                assert_eq!(processor_features(cx.execution()), expected);
                calls.set(calls.get() + 1);
                y.fill(x[0] * x[1]);
                Ok(())
            },
        )
    };
    let complex = |expected: [bool; 7]| {
        Contextual(
            move |z: &[Complex64], y: &mut [Complex64], cx: &mut SolveContext<'_, '_>| {
                assert_eq!(processor_features(cx.execution()), expected);
                calls.set(calls.get() + 1);
                y.fill(z[0] * z[1]);
                Ok(())
            },
        )
    };
    // Each evaluation calls the callback and publishes its output.
    let observed = |evaluate: &dyn Fn() -> DifferentiationReport| {
        let before = calls.get();
        assert!(evaluate().output_available());
        assert!(calls.get() > before);
    };
    let x = [1.0, 2.0];
    let direction = [1.0, -1.0];
    let nodes: [&[f64]; 2] = [&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]];
    let offsets: [&[f64]; 2] = [&[-1.0, 0.0, 1.0]; 2];

    let fixed = FiniteDifference::default();
    observed(&|| {
        *fixed
            .gradient(real(native), &x, None, None)
            .unwrap()
            .report()
    });
    observed(&|| {
        *fixed
            .hessian(real(native), &x, None, None)
            .unwrap()
            .report()
    });
    observed(&|| {
        *fixed
            .jacobian(real(native), &x, 2, None, None)
            .unwrap()
            .report()
    });
    observed(&|| {
        let result = fixed.jvp(real(native), &x, &direction, 2, None, None);
        *result.unwrap().report()
    });
    // An `_into` form passes on the context it receives.
    observed(&|| {
        let request = DerivativeRequest::new(&fixed, DerivativeShape::gradient(2));
        let mut workspace = FiniteDifferenceWorkspace::workspace(request).unwrap();
        let context = ExecutionContext::sequential();
        let (f, out) = (real(portable), &mut [0.0; 2]);
        let report = fixed.gradient_into(f, &x, None, None, out, &mut workspace, context);
        report.unwrap()
    });

    let adaptive = AdaptiveDifference::default();
    observed(&|| *adaptive.gradient(real(native), &x, None).unwrap().report());
    observed(&|| *adaptive.hessian(real(native), &x, None).unwrap().report());
    observed(&|| {
        *adaptive
            .jacobian(real(native), &x, 2, None)
            .unwrap()
            .report()
    });
    observed(&|| {
        let result = adaptive.jvp(real(native), &x, &direction, 2, None);
        *result.unwrap().report()
    });
    let schedule = ColoringSchedule::new(
        StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0],
    )
    .unwrap();
    let colored = ColoredJacobianPlan::new(&schedule, adaptive).unwrap();
    observed(&|| *colored.evaluate(real(native), &x, None).unwrap().report());

    let step = ComplexStep::default();
    observed(&|| *step.gradient(complex(native), &x).unwrap().report());
    observed(&|| *step.jacobian(complex(native), &x, 2).unwrap().report());
    observed(&|| {
        *step
            .jvp(complex(native), &x, &direction, 2)
            .unwrap()
            .report()
    });
    let plan = ComplexStepPlan::new(&x, &nodes, step).unwrap();
    observed(&|| *plan.evaluate(complex(native), None).unwrap().report());
    let relative = RelativeComplexStep::new(&offsets, step).unwrap();
    observed(&|| {
        *relative
            .evaluate(complex(native), &x, None)
            .unwrap()
            .report()
    });

    let options = NodeDifferenceOptions::default();
    let operation = DerivativeOperation::Gradient;
    let physical = PhysicalDifferencePlan::new(operation, &x, &nodes, 1, options).unwrap();
    observed(&|| *physical.evaluate(real(native), None).unwrap().report());
    let relative = RelativeDifference::new(operation, &offsets, 1, options).unwrap();
    observed(&|| *relative.evaluate(real(native), &x, None).unwrap().report());
}
