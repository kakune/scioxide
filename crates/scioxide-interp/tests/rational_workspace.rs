//! Allocation, transaction, and caller-budget guarantees.
#![allow(clippy::unwrap_used)]
mod support;

use support::assert_error;

use scioxide_core::{
    Budget, BudgetKind, Capacity, ExecutionContext, ExecutionPolicy, Interval, NdView, NdViewMut,
    ProblemToken, SciOxideErrorKind, SequentialExecutor, SolveBudgets, SolveIdentity, SolveState,
    StridedLayout, Tolerance,
};
use scioxide_interp::{
    AaaOptions, BatchEvaluator, BatchRequest, ExtrapolationPolicy as E, Interpolant,
    PadeApproximant, RationalDomain, RationalFit, RationalRequest, RationalRequirements,
    RationalTermination, RationalWorkspace, aaa_into, aaa_with_options,
};

fn domain() -> RationalDomain<f64> {
    RationalDomain::Real(Interval::new(-1., 1.).unwrap())
}
fn value_at(fit: &RationalFit<f64>, z: f64) -> f64 {
    fit.approximant().unwrap().evaluate([z], E::Reject).unwrap()
}
fn termination(fit: &RationalFit<f64>) -> RationalTermination {
    fit.report().unwrap().termination()
}
fn options(max_terms: usize, tolerance: f64) -> AaaOptions<f64> {
    AaaOptions::new()
        .with_max_terms(max_terms)
        .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap())
}
fn requirements(samples: usize, options: AaaOptions<f64>) -> RationalRequirements {
    RationalWorkspace::<f64>::workspace_requirements(RationalRequest::new(samples, options))
        .unwrap()
}
fn workspace(samples: usize, options: AaaOptions<f64>) -> RationalWorkspace<f64> {
    RationalWorkspace::workspace(RationalRequest::new(samples, options)).unwrap()
}

#[test]
fn lying_provider_reserved_workspace_reuse_is_allocation_free() {
    struct Liar;
    impl scioxide_core::ProcessorCapabilities for Liar {
        fn supports(&self, _: scioxide_core::ProcessorFeature) -> bool {
            true
        }
    }
    let nodes: Vec<_> = (0..65).map(|i| -1. + i as f64 / 32.).collect();
    let values: Vec<_> = nodes.iter().map(|x| 1. / (2. + x)).collect();
    let options = options(4, 1e-12);
    let requirements = requirements(nodes.len(), options);
    // Reservation takes no execution context, so any processor policy may solve.
    let mut work = workspace(nodes.len(), options);
    let mut output = RationalFit::new();
    output.try_reserve(&requirements).unwrap();
    let liar = ExecutionContext::sequential().with_processor(&Liar);
    for execution in [liar, ExecutionContext::sequential(), liar] {
        aaa_into(
            &nodes,
            &values,
            domain(),
            options,
            &mut output,
            &mut work,
            execution,
        )
        .unwrap();
        assert_eq!(termination(&output), RationalTermination::Converged);
        assert!((value_at(&output, 0.2) - 1. / 2.2).abs() < 1e-12);
    }
}

#[test]
fn reserved_fits_queries_and_failed_reuse_execute() {
    let nodes: Vec<_> = (0..33).map(|i| -1. + i as f64 / 16.).collect();
    let values: Vec<_> = nodes.iter().map(|x| x.exp()).collect();
    let options = options(4, 1e-12).with_lawson_iterations(3);
    let requirements = requirements(nodes.len(), options);
    let mut output = RationalFit::new();
    output.try_reserve(&requirements).unwrap();
    let mut work = workspace(nodes.len(), options);
    assert_eq!(work.check(&requirements).unwrap(), Capacity::Sufficient);
    let cx = ExecutionContext::sequential_native();
    aaa_into(
        &nodes,
        &values,
        domain(),
        options,
        &mut output,
        &mut work,
        cx,
    )
    .unwrap();
    let history = work.residual_history().to_vec();
    assert!(history.len() > output.report().unwrap().iterations().min(4));
    assert_eq!(
        *history.last().unwrap(),
        output.report().unwrap().max_residual()
    );
    let before = value_at(&output, 0.25);
    let report = output.report();
    let mut bad = nodes.clone();
    bad[1] = bad[0];
    assert!(aaa_into(&bad, &values, domain(), options, &mut output, &mut work, cx).is_err());
    assert_eq!(before, value_at(&output, 0.25));
    assert_eq!(report, output.report());
    aaa_into(
        &nodes,
        &values,
        domain(),
        options,
        &mut output,
        &mut work,
        cx,
    )
    .unwrap();
    // The protocol's batch form evaluates the published approximant.
    let approximant = output.approximant().unwrap();
    let batch = BatchEvaluator::new(approximant, ExecutionPolicy::Sequential);
    let mut batch_work = batch.workspace(BatchRequest::values(3)).unwrap();
    let points = [-0.5, 0., 0.5];
    let queries = NdView::new(&points, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
    let mut storage = [0.; 3];
    {
        let mut out =
            NdViewMut::new(&mut storage, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
        batch.evaluate_into(&queries, E::Reject, &mut out, None, &mut batch_work, cx)
    }
    .unwrap();
    for (value, point) in storage.into_iter().zip(points) {
        assert_eq!(value, approximant.evaluate([point], E::Reject).unwrap());
    }
}

#[test]
fn options_budget_exhaustion_and_executor_equivalence() {
    let nodes = [-1.0_f64, -0.5, 0., 0.5, 1.];
    let values = nodes.map(|x| 1. / (2. + x));
    let options = options(3, 1e-12);
    let limited = options.with_budgets(
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap()),
    );
    let mut output = RationalFit::new();
    output
        .try_reserve(&requirements(nodes.len(), limited))
        .unwrap();
    let mut work = workspace(nodes.len(), limited);
    aaa_into(
        &nodes,
        &values,
        domain(),
        limited,
        &mut output,
        &mut work,
        ExecutionContext::new(&SequentialExecutor),
    )
    .unwrap();
    let report = output.report().unwrap();
    assert_eq!(report.termination(), RationalTermination::BudgetExhausted);
    assert_eq!(report.iterations(), 1);
    assert!(!report.termination().is_success());
    assert_eq!(work.residual_history().len(), 2);

    aaa_into(
        &nodes,
        &values,
        domain(),
        options,
        &mut output,
        &mut work,
        ExecutionContext::new(&SequentialExecutor),
    )
    .unwrap();
    let expected = value_at(&output, 0.25);
    let owned = aaa_with_options(&nodes, &values, domain(), options).unwrap();
    assert_eq!(expected, value_at(&owned, 0.25));
    assert_eq!(output.report(), owned.report());
}

#[test]
fn cancelled_constant_fit_preserves_output_and_smaller_buffers_do_not_grow() {
    use scioxide_core::CancellationToken;
    let x = [-1.0_f64, -0.5, 0., 0.5, 1.];
    let y = [1.0_f64; 5];
    let options = options(5, 1e-12);
    let mut out = RationalFit::new();
    out.try_reserve(&requirements(5, options)).unwrap();
    let mut w = workspace(5, options);
    let token = CancellationToken::new();
    token.cancel();
    assert_error(
        &aaa_into(
            &x,
            &y,
            domain(),
            options,
            &mut out,
            &mut w,
            ExecutionContext::sequential().with_cancellation(&token),
        ),
        scioxide_core::SciOxideErrorKind::ExecutionStopped,
    );
    assert!(out.approximant().is_none() && out.report().is_none());
    let small_options = options.with_max_terms(1);
    let mut small = RationalFit::new();
    small.try_reserve(&requirements(5, small_options)).unwrap();
    aaa_into(
        &x,
        &y,
        domain(),
        small_options,
        &mut small,
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    {
        let error = aaa_into(
            &x,
            &y,
            domain(),
            options,
            &mut small,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    };
    let mut fresh = workspace(5, small_options);
    {
        let error = aaa_into(
            &x,
            &y,
            domain(),
            options,
            &mut out,
            &mut fresh,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    };
    assert!(out.approximant().is_none());
    // Publication swapped the small output's buffers into `w`; reserve again.
    let requirements = requirements(5, options);
    assert_eq!(w.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    w.try_reserve(&requirements).unwrap();
    aaa_into(
        &x,
        &y,
        domain(),
        options,
        &mut out,
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(termination(&out), RationalTermination::Converged);
}

#[test]
fn scoped_constructions_charge_a_parent_and_aaa_budget_stops_are_ok() {
    use scioxide_linalg::RankPolicy;
    use scioxide_poly::PolynomialApproximationPolicy;
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(1e-12, 1e-12).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    );
    // A parent solve observes the nested work of child scopes; iteration
    // allowances stay local to each scope.
    let id = SolveIdentity::new(
        ProblemToken::new(579),
        scioxide_core::algorithm_id!("interp", "budget_parent", 1),
        3,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(ExecutionContext::new(&SequentialExecutor), id, &mut budgets)
        .unwrap();
    for expected in 1..=2 {
        let plan =
            PadeApproximant::<f64>::build(&[1., 1., 1.], 1, 1, 0., domain(), policy, &mut parent)
                .unwrap();
        assert!((plan.evaluate([0.2], E::Reject).unwrap() - 1.25).abs() < 1e-12);
        assert_eq!(plan.fit_report().iterations(), 1);
        assert_eq!(parent.counts().total().unwrap().iterations(), expected);
    }
    // AAA accepts the constant mean before its first admission, so a denied
    // admission is an unsuccessful `Ok` with the accepted approximant.
    let nodes = [-1.0_f64, -0.5, 0., 0.5, 1.];
    let values = nodes.map(|x| 1. / (2. + x));
    let aaa = options(3, 1e-12)
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
    let mut out = RationalFit::new();
    out.try_reserve(&requirements(5, aaa)).unwrap();
    let mut work = workspace(5, aaa);
    aaa_into(
        &nodes,
        &values,
        domain(),
        aaa,
        &mut out,
        &mut work,
        &mut parent,
    )
    .unwrap();
    let report = out.report().unwrap();
    assert_eq!(report.termination(), RationalTermination::BudgetExhausted);
    assert_eq!(report.iterations(), 1);
    assert_eq!(parent.counts().total().unwrap().iterations(), 3);
    assert_eq!(work.residual_history().len(), 2);
    assert_eq!(work.residual_history()[1], report.max_residual());
}

#[test]
fn parallel_policy_reservation_is_explicit_and_allocation_free() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPlan, ExecutionPolicy, Executor, PartitionFn};
    struct TwoWorkerExecutor;
    impl Executor for TwoWorkerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            SequentialExecutor.execute(context, plan, operation)
        }
    }
    let executor = TwoWorkerExecutor;
    let nodes: Vec<_> = (0..65).map(|i| -1. + f64::from(i) / 32.).collect();
    let values: Vec<_> = nodes.iter().map(|x| x.exp()).collect();
    let options = options(6, 1e-10);
    for policy in [
        ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()),
        ExecutionPolicy::Sequential,
    ] {
        let options = options.with_execution_policy(policy);
        let mut out = RationalFit::new();
        out.try_reserve(&requirements(65, options)).unwrap();
        // Requirements are bounded by the policy, so they hold for any executor.
        let mut work = workspace(65, options);
        aaa_into(
            &nodes,
            &values,
            domain(),
            options,
            &mut out,
            &mut work,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        assert_eq!(termination(&out), RationalTermination::Converged);
        assert!((value_at(&out, 0.2) - 0.2_f64.exp()).abs() < 1e-9);
    }
}

#[test]
fn signed_zero_duplicates_are_rejected_with_reused_storage_or_publishing() {
    let options = options(3, 1e-12);
    let nodes = [-1.0, 0.0, 0.5, 1.0];
    let values = [1.0, 2.0, 2.5, 3.0];
    let requirements = requirements(nodes.len(), options);
    let mut output = RationalFit::new();
    output.try_reserve(&requirements).unwrap();
    let mut workspace = workspace(nodes.len(), options);
    aaa_into(
        &nodes,
        &values,
        domain(),
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let before = value_at(&output, 0.25);
    let report = output.report();
    let result = aaa_into(
        &[-1.0, 0.0, -0.0, 1.0],
        &values,
        domain(),
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    );
    assert_error(&result, SciOxideErrorKind::InvalidInput);
    assert_eq!(output.report(), report);
    assert_eq!(value_at(&output, 0.25), before);
}

#[test]
fn extreme_support_queries_remain_allocation_free() {
    use scioxide_interp::BarycentricRational;
    for scale in [1e-300_f64, 1e300] {
        let domain = RationalDomain::Real(Interval::new(-scale, scale).unwrap());
        let plan = BarycentricRational::new(
            vec![-scale, scale],
            vec![1.0, -3.0],
            vec![1.0, -1.0],
            domain,
        )
        .unwrap();
        let point = f64::from_bits(scale.to_bits() - 1);
        let mut value = 0.0;
        plan.evaluate_into([point], E::Reject, &mut value, &mut ())
            .unwrap();
        assert!((value - (2.0 + point / scale)).abs() < 1e-14);
    }
}
