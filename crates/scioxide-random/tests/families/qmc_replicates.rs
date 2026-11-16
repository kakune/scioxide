#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;
use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor,
    PartitionFn, ProblemToken, RealScalar, SciOxideError, SequentialExecutor, SolveBudgets,
    SolveContext, SolveIdentity, SolveScope, SolveState,
};
use scioxide_random::*;
struct Reverse;
impl Executor for Reverse {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let results = std::thread::scope(|threads| {
            // Launch every worker before joining any of them.
            let mut handles = Vec::with_capacity(plan.worker_count());
            for index in (0..plan.worker_count()).rev() {
                handles.push(threads.spawn(move || {
                    SequentialExecutor.execute(context, plan, &|part, (), scope| {
                        if part.worker_index() == index {
                            operation(part, (), scope)
                        } else {
                            Ok(())
                        }
                    })
                }));
            }
            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                results.push(handle.join().unwrap());
            }
            results
        });
        for result in results {
            result?;
        }
        SequentialExecutor.execute(context, plan, &|_, (), _| Ok(()))
    }
}
fn check<S: RealScalar>() {
    let base = QmcPlan::sobol(3).unwrap();
    let root = Philox4x64::from_key([123, 456]);
    let plan = QmcReplicatePlan::new(
        &base,
        &root,
        QmcReplicateOptions::default()
            .with_replicates(16)
            .with_points_per_replicate(256),
    )
    .unwrap();
    let workers = NonZeroUsize::new(4).unwrap();
    let integrand = |x: &[S]| Ok(x.iter().copied().fold(S::ONE, |p, x| p * x));
    assert_eq!(plan.execution_policy(), ExecutionPolicy::Sequential);
    let sequential = plan.estimate(integrand).unwrap();
    let parallel = QmcReplicatePlan::new(
        &base,
        &root,
        QmcReplicateOptions::default()
            .with_replicates(16)
            .with_points_per_replicate(256)
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    )
    .unwrap()
    .estimate_scoped(integrand, ExecutionContext::new(&Reverse))
    .unwrap();
    assert_eq!(sequential.mean, parallel.mean);
    assert_eq!(sequential.standard_error, parallel.standard_error);
    assert_eq!(sequential.replicate_means, parallel.replicate_means);
    assert!((sequential.mean - 0.125).abs() < 6.0 * sequential.standard_error + 2e-6);
    let same = QmcReplicatePlan::new(
        &base,
        &root,
        QmcReplicateOptions::default()
            .with_replicates(16)
            .with_points_per_replicate(256),
    )
    .unwrap();
    assert_eq!(
        plan.plans()[5].randomization_words(),
        same.plans()[5].randomization_words()
    );
    assert_ne!(
        plan.plans()[0].randomization_words(),
        plan.plans()[1].randomization_words()
    );
    let halton = QmcReplicatePlan::new(
        &QmcPlan::halton(2).unwrap(),
        &root,
        QmcReplicateOptions::default()
            .with_replicates(32)
            .with_points_per_replicate(256),
    )
    .unwrap();
    let estimate = halton.estimate(|x: &[S]| Ok(x[0] * x[1])).unwrap();
    assert!((estimate.mean - 0.25).abs() < 6.0 * estimate.standard_error + 2e-6);
}
#[test]
fn replicated_estimates_both_widths() {
    check::<f32>();
    check::<f64>();
}
#[test]
fn invalid_plans_and_callback_errors() {
    let base = QmcPlan::sobol(2).unwrap();
    let root = Philox4x64::from_key([1, 2]);
    assert!(
        QmcReplicatePlan::new(
            &base,
            &root,
            QmcReplicateOptions::default()
                .with_replicates(1)
                .with_points_per_replicate(256)
        )
        .is_err()
    );
    assert!(
        QmcReplicatePlan::new(
            &base,
            &root,
            QmcReplicateOptions::default()
                .with_replicates(2)
                .with_points_per_replicate(3)
        )
        .is_err()
    );
    let plan = QmcReplicatePlan::new(
        &base,
        &root,
        QmcReplicateOptions::default()
            .with_replicates(2)
            .with_points_per_replicate(4),
    )
    .unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = plan.estimate(|_: &[f64]| Ok(value)).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Callback,
                ..
            }
        ));
    }
}

#[test]
fn constant_extreme_values_survive_mean_reduction() {
    let plan = QmcReplicatePlan::new(
        &QmcPlan::sobol(1).unwrap(),
        &Philox4x64::from_key([1, 2]),
        QmcReplicateOptions::default()
            .with_replicates(4)
            .with_points_per_replicate(256),
    )
    .unwrap();
    for value in [f64::from_bits(1), -f64::from_bits(1), f64::MAX, -f64::MAX] {
        let estimate = plan.estimate(|_: &[f64]| Ok(value)).unwrap();
        assert_eq!(estimate.mean, value);
        assert_eq!(estimate.standard_error, 0.0);
    }
}

#[test]
fn cancelled_replicates_do_not_invoke_the_integrand() {
    let plan = QmcReplicatePlan::new(
        &QmcPlan::sobol(1).unwrap(),
        &Philox4x64::from_key([1, 2]),
        QmcReplicateOptions::default()
            .with_replicates(4)
            .with_points_per_replicate(16),
    )
    .unwrap();
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let error = plan
        .estimate_scoped(
            |_: &[f64]| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(1.0)
            },
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// Runs `body` inside a parent solve whose evaluation limit is `limit`.
fn in_parent<T>(
    limit: usize,
    body: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    SolveScope::from(ExecutionContext::sequential()).run(
        &mut SolveState::new(),
        SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("test", "parent", 1),
            0,
            0,
        ),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(limit)?),
        body,
    )
}

#[test]
fn scoped_estimates_equal_owned_and_count_in_the_parent() {
    let plan = QmcReplicatePlan::new(
        &QmcPlan::sobol(2).unwrap(),
        &Philox4x64::from_key([7, 9]),
        QmcReplicateOptions::default()
            .with_replicates(4)
            .with_points_per_replicate(64),
    )
    .unwrap();
    let integrand = |x: &[f64]| Ok(x[0] * x[0] + x[1]);
    let owned = plan.estimate(integrand).unwrap();
    let scoped = plan
        .estimate_scoped(integrand, ExecutionContext::sequential_native())
        .unwrap();
    assert_eq!(owned.mean.to_bits(), scoped.mean.to_bits());
    assert_eq!(
        owned.standard_error.to_bits(),
        scoped.standard_error.to_bits()
    );
    assert_eq!(owned.replicate_means, scoped.replicate_means);
    // The parent counts the R · N = 256 calls of the child epoch.
    let (nested, before, after) = in_parent(1_000, |parent| {
        let before = parent.budget_report(BudgetKind::Evaluation).consumed();
        let nested = plan.estimate_scoped(integrand, &mut *parent)?;
        let after = parent.budget_report(BudgetKind::Evaluation).consumed();
        assert_eq!(parent.provider_count(EvaluationRole::Function)?, 256);
        Ok((nested, before, after))
    })
    .unwrap();
    assert_eq!(after - before, 256);
    assert_eq!(nested.replicate_means, owned.replicate_means);
    // A parent limit below R · N admits nothing and calls nothing.
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let error = in_parent(1, |parent| {
        plan.estimate_scoped(
            |x: &[f64]| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                integrand(x)
            },
            parent,
        )
    })
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    #[cfg(feature = "std")]
    {
        let workers = NonZeroUsize::new(3).unwrap();
        let parallel = QmcReplicatePlan::new(
            &QmcPlan::sobol(2).unwrap(),
            &Philox4x64::from_key([7, 9]),
            QmcReplicateOptions::default()
                .with_replicates(4)
                .with_points_per_replicate(64)
                .with_execution_policy(ExecutionPolicy::Parallel(workers)),
        )
        .unwrap();
        let executor = scioxide_core::ThreadExecutor::new(workers);
        let threaded = parallel
            .estimate_scoped(integrand, ExecutionContext::new(&executor))
            .unwrap();
        assert_eq!(threaded.replicate_means, owned.replicate_means);
        assert_eq!(threaded.mean.to_bits(), owned.mean.to_bits());
    }
}
