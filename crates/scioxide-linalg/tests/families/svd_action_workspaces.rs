//! Public capacity, allocation and execution contracts for prepared actions.
#![allow(clippy::unwrap_used)]

use crate::support::action_operator::Operator;

use scioxide_core::{Capacity, SciOxideError};
use scioxide_linalg::*;

#[test]
fn randomized_nested_factors_honor_policy_and_exact_reservation() -> Result<(), SciOxideError> {
    use core::{
        num::NonZeroUsize,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use scioxide_core::{ExecutionContext, ExecutionPolicy, Executor};
    struct Counting(AtomicUsize);
    impl Executor for Counting {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            cx: ExecutionContext<'_>,
            plan: scioxide_core::ExecutionPlan,
            operation: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.fetch_max(plan.worker_count(), Ordering::Relaxed);
            scioxide_core::SequentialExecutor.execute(cx, plan, operation)
        }
    }
    let n = 128;
    let width = 96;
    let rank = 4;
    let op = Operator::new(n, 0);
    let mut sketch = Matrix::<f64>::zeros(n, width)?;
    for i in 0..width {
        sketch.view_mut().set(i, i, 1.)?;
    }
    let mut probes = Matrix::zeros(n, 1)?;
    probes.view_mut().set(0, 0, 1.)?;
    for policy in [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
    ] {
        let options = RandomizedLowRank::new(rank).with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(policy),
        );
        let plan = options;
        let bytes = plan
            .workspace_requirements::<f64>(&op, width)?
            .required_bytes();
        let plan =
            options.with_execution_options((options).execution_options().with_byte_limit(bytes));
        let req = plan.workspace_requirements(&op, width)?;
        let mut workspace = RandomizedLowRankWorkspace::new();
        workspace.try_reserve(&req)?;
        let mut svd = RandomizedSvdBuffer::new(n, n, rank)?;
        let mut id = InterpolativeBuffer::new(n, n, rank)?;
        let executor = Counting(AtomicUsize::new(0));
        for _ in 0..2 {
            (|| {
                plan.svd_into(
                    &op,
                    sketch.view(),
                    probes.view(),
                    &mut svd,
                    &mut workspace,
                    ExecutionContext::new(&executor),
                )?;
                plan.interpolative_into(
                    &op,
                    sketch.view(),
                    probes.view(),
                    &mut id,
                    &mut workspace,
                    ExecutionContext::new(&executor),
                )
            })()?;
            assert_eq!(workspace.check(&req)?, Capacity::Sufficient);
        }
        assert!(
            svd.singular_values()
                .iter()
                .all(|v| (*v - 1.).abs() < 1e-12)
        );
        if policy == ExecutionPolicy::Sequential {
            assert!(executor.0.load(Ordering::Relaxed) <= 1);
        } else {
            assert!(executor.0.load(Ordering::Relaxed) > 1);
        }
    }
    Ok(())
}

#[test]
fn randomized_rank_policy_controls_basis_and_both_factorizations() -> Result<(), SciOxideError> {
    use scioxide_linalg::RankPolicy;
    let sketch = Matrix::<f64>::identity(2)?;
    let probes = Matrix::<f64>::identity(2)?;
    let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1e-18])?;
    assert_eq!(
        RandomizedLowRank::<f64>::new(2).rank_policy(),
        RankPolicy::Exact
    );
    for steps in [0, 1, 2] {
        for (policy, expected) in [
            (RankPolicy::Exact, 2),
            (RankPolicy::DimensionScaledEpsilon, 1),
            (
                RankPolicy::Threshold(scioxide_core::RankThreshold::new(1e-10, 0.0)?),
                1,
            ),
        ] {
            let options = RandomizedLowRank::new(2)
                .with_rank_policy(policy)
                .with_power_steps(steps);
            let plan = options;
            let (_, svd) = plan.svd(&a.view().try_finite()?, sketch.view(), probes.view())?;
            let (_, id) =
                plan.interpolative(&a.view().try_finite()?, sketch.view(), probes.view())?;
            assert_eq!(svd.rank(), expected);
            assert_eq!(id.rank(), expected);
        }
    }
    // A nearly dependent sketch exercises basis admission independently of A's spectrum.
    let a = Matrix::<f64>::identity(2)?;
    let sketch = Matrix::from_vec(2, 2, vec![1.0, 1.0, 0.0, 1e-18])?;
    for (policy, expected) in [
        (RankPolicy::Exact, 2),
        (RankPolicy::DimensionScaledEpsilon, 1),
    ] {
        let plan = RandomizedLowRank::new(2).with_rank_policy(policy);
        assert_eq!(
            plan.svd(&a.view().try_finite()?, sketch.view(), probes.view())?
                .1
                .rank(),
            expected
        );
        assert_eq!(
            plan.interpolative(&a.view().try_finite()?, sketch.view(), probes.view())?
                .1
                .rank(),
            expected
        );
    }
    let dependent = Matrix::from_vec(2, 2, vec![1.0, 1.0, 0.0, 0.0])?;
    let plan = RandomizedLowRank::new(2);
    assert_eq!(
        plan.svd(&a.view().try_finite()?, dependent.view(), probes.view())?
            .1
            .rank(),
        1
    );
    assert_eq!(
        plan.interpolative(&a.view().try_finite()?, dependent.view(), probes.view())?
            .1
            .rank(),
        1
    );
    Ok(())
}
