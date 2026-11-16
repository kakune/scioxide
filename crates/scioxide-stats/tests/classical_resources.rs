//! Resource ceilings, transactional output, masks and executor invariance.
#![allow(clippy::unwrap_used)]

use scioxide_core::*;
use scioxide_linalg::Matrix;
use scioxide_random::{OutputPolicy, Philox4x64, SamplingPolicy, SeedableRng, SplitMix64};
use scioxide_stats::*;
use std::num::NonZeroUsize;

fn run<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(546),
        scioxide_core::algorithm_id!("stats", "classical-resource", 1),
        0,
        0,
    );
    f(&mut state.begin_solve(execution, id, &mut budgets).unwrap())
}
struct Reverse;
impl Executor for Reverse {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(3).unwrap()
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        op: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let parts: Vec<_> = plan.partitions().collect();
        for target in parts.iter().rev() {
            SequentialExecutor.execute(ctx, plan, &|p, (), s| {
                if &p == target { op(p, (), s) } else { Ok(()) }
            })?;
        }
        SequentialExecutor.execute(ctx, plan, &|_, (), _| Ok(()))
    }
}

#[test]
fn indexed_mgc_reproducibility_and_two_sample_design() {
    let x = Matrix::from_vec(5, 1, vec![0., 1., 2., 3., 4.]).unwrap();
    let y = Matrix::from_vec(5, 1, vec![0., 1., 4., 9., 16.]).unwrap();
    let mgc = MgcPlan::from_samples(x.view(), y.view(), ClassicalOptions::default()).unwrap();
    let root = Philox4x64::from_seed(546);
    let mut previous = None;
    for batch in [1, 3] {
        let plan = mgc
            .permutation_plan(
                PermutationMode::Randomized,
                ResamplingOptions::default()
                    .with_replicates(39)
                    .with_batch_size(batch),
            )
            .unwrap();
        let mut ws = plan.workspace().unwrap();
        let mut output = vec![0.; plan.replicates()];
        let policy = if batch == 1 {
            ExecutionPolicy::Sequential
        } else {
            ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap())
        };
        let report = run(
            ExecutionContext::new(&Reverse),
            SolveBudgets::unbounded(),
            |ctx| {
                mgc.test_into(
                    &plan,
                    scioxide_stats::ReplicateOptions::indexed(
                        &root,
                        100,
                        policy,
                        scioxide_random::OutputPolicy::Atomic,
                    ),
                    &mut output,
                    &mut ws,
                    ctx,
                )
            },
        )
        .unwrap();
        if let Some((values, p)) = &previous {
            assert_eq!(&output, values);
            assert_eq!(report.test().unwrap().p_value(), *p);
        } else {
            previous = Some((output, report.test().unwrap().p_value()));
        }
    }
    let a = Matrix::from_vec(2, 1, vec![0., 1.]).unwrap();
    let b = Matrix::from_vec(3, 1, vec![2., 3., 4.]).unwrap();
    let mgc = MgcPlan::two_sample(a.view(), b.view(), ClassicalOptions::default()).unwrap();
    let plan = mgc
        .permutation_plan(
            PermutationMode::Exact,
            ResamplingOptions::default()
                .with_replicates(10)
                .with_batch_size(1),
        )
        .unwrap();
    assert_eq!(plan.replicates(), 10);
    let mut out = vec![0.; 10];
    let mut ws = plan.workspace().unwrap();
    let report = run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded(),
        |ctx| {
            mgc.test_into(
                &plan,
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(0),
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                ),
                &mut out,
                &mut ws,
                ctx,
            )
        },
    )
    .unwrap();
    assert_eq!(
        report.test().unwrap().sample_policy(),
        TestSamplePolicy::IndependentGroups
    );
    assert_eq!(report.test().unwrap().retained(), [2, 3]);
}

#[test]
fn absent_pair_members_hide_unobserved_nonfinite_values() {
    let layout = StridedLayout::new(&[4], &[1]).unwrap();
    let mask = [true, false, true, true];
    let a = BatchView::new(&[1., 2., 3., 5.], layout, Axis::Index(0))
        .unwrap()
        .with_validity(&mask)
        .unwrap();
    let b = BatchView::new(&[0., f64::NAN, 1., 1.], layout, Axis::Index(0)).unwrap();
    let mut ws = ClassicalBatchWorkspace::new(
        a,
        Some(b),
        StatisticsOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut out = [None];
    let options = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
    classical_tests_into(
        ClassicalBatchTest::PairedT,
        a,
        Some(b),
        Default::default(),
        options,
        &mut out,
        &mut ws,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out[0].unwrap().retained(), [3, 3]);
    classical_tests_into(
        ClassicalBatchTest::PairedT,
        b,
        Some(a),
        Default::default(),
        options,
        &mut out,
        &mut ws,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out[0].unwrap().retained(), [3, 3]);
}

#[test]
fn exact_work_allowances_are_convergence_budgets_not_resource_limits() {
    let exact = ClassicalOptions::default().with_calibration(TestCalibration::Exact);
    let node_budget = |error: SciOxideError| {
        matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::ResourceLimit(BudgetKind::Node),
                ..
            }
        )
    };
    let x = [1., 2., 3., 4., 5.];
    let y = [6., 7., 8., 9., 10.];
    // C(10, 5) = 252 exact null states.
    assert!(mann_whitney_test_with_options(&x, &y, exact.with_max_work(252)).is_ok());
    assert!(node_budget(
        mann_whitney_test_with_options(&x, &y, exact.with_max_work(251)).unwrap_err()
    ));
    // The exact KS lattice needs n * m = 25 states.
    assert!(ks_two_sample_test_with_options(&x, &y, exact.with_max_work(25)).is_ok());
    assert!(node_budget(
        ks_two_sample_test_with_options(&x, &y, exact.with_max_work(24)).unwrap_err()
    ));
    // Fisher's exact test enumerates its finite tables against the same allowance.
    let counts = [3, 1, 1, 3];
    let table = ContingencyView::new(&counts, 2, 2).unwrap();
    assert!(fisher_exact_test_with_options(table, ClassicalOptions::default()).is_ok());
    assert!(node_budget(
        fisher_exact_test_with_options(table, ClassicalOptions::default().with_max_work(1))
            .unwrap_err()
    ));
    // A byte limit stays a resource limit carrying the required size.
    let error =
        ks_two_sample_test_with_options(&x, &y, exact.with_max_workspace_bytes(8)).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ResourceLimit { required, limit: 8, .. } if required > 8
    ));
}
