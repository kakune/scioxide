//! Allocation, execution and transactional AMR failure contracts.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;

use scioxide_core::*;
use scioxide_pde::*;
use scioxide_spatial::*;
#[path = "../examples/support/amr.rs"]
mod support;
use support::*;

struct Threads {
    fail: bool,
}
impl Executor for Threads {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, &|partition, (), scope| {
            std::thread::scope(|s| s.spawn(|| operation(partition, (), scope)).join().unwrap())
        })?;
        if self.fail {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("amr_test"),
                InvalidInputReason::Inconsistent,
                "injected executor failure after evaluation",
            ))
        } else {
            Ok(())
        }
    }
}
struct Residual {
    fail: bool,
}
impl AmrResidual<f64, Complex64, 2> for Residual {
    fn volume(&self, _: usize, point: [f64; 2]) -> Result<Complex64, SciOxideError> {
        if self.fail {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("amr_test"),
                InvalidInputReason::Inconsistent,
                "injected residual failure",
            ));
        }
        Ok(Complex64::new(1. + point[0], point[1]))
    }
    fn flux(&self, cell: usize, _: [f64; 2], _: [f64; 2]) -> Result<Complex64, SciOxideError> {
        Ok(Complex64::new(cell as f64, 0.25))
    }
    fn boundary_flux(
        &self,
        _: usize,
        _: [f64; 2],
        _: [f64; 2],
    ) -> Result<Option<Complex64>, SciOxideError> {
        Ok(Some(Complex64::new(0., 0.)))
    }
}
fn execute<T>(
    execution: ExecutionContext<'_>,
    budgets: &mut SolveBudgets,
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(635),
        scioxide_core::algorithm_id!("pde", "amr", 1),
        0,
        0,
    );
    let mut ctx = state.begin_solve(execution, id, budgets)?;
    run(&mut ctx)
}
#[test]
fn estimator_executor_parity_allocation_and_failure_rollback() {
    let mesh = adapt(&tensor::<f64, 2>(2, false).unwrap(), &[0], &[])
        .unwrap()
        .mesh;
    let workers = NonZeroUsize::new(4).unwrap();
    // The scheduling policy belongs to the space; both estimators share one mesh.
    let space =
        AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(limits()))
            .unwrap();
    let parallel_space = AmrSpace::with_options(
        &mesh,
        AmrOrder::Q1,
        AmrOptions::new()
            .with_limits(limits())
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    )
    .unwrap();
    let k = Field::new(&mesh, vec![1.; mesh.cell_count()]).unwrap();
    let estimator = AmrEstimator::with_options(&space, k.view(), limits()).unwrap();
    let parallel = AmrEstimator::with_options(&parallel_space, k.view(), limits()).unwrap();
    let mut work = estimator.workspace().unwrap();
    let mut a = AmrIndicators::new(&mesh).unwrap();
    let mut b = AmrIndicators::new(&mesh).unwrap();
    let provider = Residual { fail: false };
    let sequential_counts = context(|ctx| {
        estimator.estimate_into(&provider, &mut a, &mut work, &mut *ctx)?;
        Ok(ctx.counts())
    })
    .unwrap();
    let parallel_counts = execute(
        ExecutionContext::new(&Threads { fail: false }),
        &mut SolveBudgets::unbounded(),
        |ctx| {
            parallel.estimate_into(&provider, &mut b, &mut work, &mut *ctx)?;
            Ok(ctx.counts())
        },
    )
    .unwrap();
    // Per-item budget shares make the inherited accounting independent of workers.
    assert_eq!(sequential_counts, parallel_counts);
    assert!(
        sequential_counts
            .total()
            .unwrap()
            .total_providers()
            .unwrap()
            > 0
    );
    assert_eq!(a.volume(), b.volume());
    assert_eq!(a.interior(), b.interior());
    assert_eq!(a.boundary(), b.boundary());
    assert_eq!(a.global(), b.global());
    context(|ctx| {
        estimator
            .estimate_into(&provider, &mut a, &mut work, &mut *ctx)
            .unwrap();
        Ok(())
    })
    .unwrap();
    let before = b.squared().collect::<Vec<_>>();
    assert!(
        execute(
            ExecutionContext::new(&Threads { fail: true }),
            &mut SolveBudgets::unbounded(),
            |ctx| parallel.estimate_into(&provider, &mut b, &mut work, ctx)
        )
        .is_err()
    );
    assert_eq!(before, b.squared().collect::<Vec<_>>());
    assert!(
        context(|ctx| estimator.estimate_into(&Residual { fail: true }, &mut b, &mut work, ctx))
            .is_err()
    );
    assert_eq!(before, b.squared().collect::<Vec<_>>());
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(1).unwrap());
    assert!(
        execute(ExecutionContext::sequential(), &mut budgets, |ctx| {
            estimator.estimate_into(&provider, &mut b, &mut work, ctx)
        })
        .is_err()
    );
    assert_eq!(before, b.squared().collect::<Vec<_>>());
}

#[test]
fn transfer_executor_and_prepared_field_operations_execute() {
    let mesh = tensor::<f64, 2>(2, false).unwrap();
    let fine = adapt(&mesh, &[0], &[]).unwrap();
    // The scheduling policy belongs to the transfer's options.
    let map = AmrCellTransfer::new(
        &mesh,
        &fine.mesh,
        &fine.map,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let automatic = AmrCellTransfer::new(
        &mesh,
        &fine.mesh,
        &fine.map,
        AmrOptions::new()
            .with_limits(limits())
            .with_execution_policy(ExecutionPolicy::Auto),
    )
    .unwrap();
    let field =
        Field::<_, Cell, 2>::new(&mesh, vec![Complex64::new(1., -2.); mesh.cell_count()]).unwrap();
    let mut a = vec![Complex64::new(0., 0.); fine.mesh.cell_count()];
    let mut b = a.clone();
    let mut scratch = a.clone();
    map.apply_into(
        field.view(),
        &mut a,
        &mut scratch,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    automatic
        .apply_into(
            field.view(),
            &mut b,
            &mut scratch,
            ExecutionContext::new(&Threads { fail: false }),
        )
        .unwrap();
    assert_eq!(a, b);
    map.apply_into(
        field.view(),
        &mut a,
        &mut scratch,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(
        map.apply_into(
            field.view(),
            &mut b,
            &mut scratch,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .is_err()
    );
    assert_eq!(a, b);
    let space = AmrSpace::with_options(
        &fine.mesh,
        AmrOrder::Q1,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let input = vec![Complex64::new(1., 2.); space.independent_dofs()];
    let mut full = vec![Complex64::new(0., 0.); space.full_dofs()];
    let mut scratch = full.clone();
    space
        .constraints()
        .apply_into(&input, &mut full, &mut scratch)
        .unwrap();
    let mut corrupt = full.clone();
    let hanging = (0..space.full_dofs())
        .find(|i| !space.independent_nodes().contains(i))
        .unwrap();
    corrupt[hanging] = Complex64::new(99., 0.);
    assert!(space.validate_field(&corrupt).is_err());
    {
        space.evaluate(0, [0.3, 0.4], &full).unwrap();
    };
}

#[test]
fn empty_spaces_and_inherited_preparation_budgets() {
    let empty = TriangleMesh::<f64>::new(vec![], vec![]).unwrap();
    let mesh = AdaptiveMesh::from_simplex(&empty, topology_limits()).unwrap();
    for order in [AmrOrder::P1, AmrOrder::P2] {
        let space =
            AmrSpace::with_options(&mesh, order, AmrOptions::new().with_limits(limits())).unwrap();
        assert_eq!(space.full_dofs(), 0);
        let k = Field::new(&mesh, vec![]).unwrap();
        let e = AmrEstimator::with_options(&space, k.view(), limits()).unwrap();
        let mut w = e.workspace().unwrap();
        let mut out = AmrIndicators::new(&mesh).unwrap();
        context(|ctx| e.estimate_into(&Residual { fail: false }, &mut out, &mut w, ctx)).unwrap();
        assert_eq!(out.global(), 0.);
    }
    let mesh = tensor::<f64, 2>(2, false).unwrap();
    let token = mesh.mesh_token().clone();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::try_from(1).unwrap());
    assert!(
        execute(ExecutionContext::sequential(), &mut budgets, |ctx| {
            AmrSpace::build(
                &mesh,
                AmrOrder::Q1,
                AmrOptions::new().with_limits(limits()),
                ctx,
            )
        })
        .is_err()
    );
    assert_eq!(mesh.mesh_token(), &token);
}
#[test]
fn exhausted_ceilings_and_scope_budgets_are_resource_failures() {
    let mesh = tensor::<f64, 2>(2, false).unwrap();
    // Nine Q1 vertices exceed a ceiling of four degrees of freedom.
    let tight = AmrLimits::new(4, 5_000_000, 4096, 500_000_000, 1 << 30);
    let error = AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(tight))
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ResourceLimit {
            required: 9,
            limit: 4,
            ..
        }
    ));
    let starved = AmrLimits::new(100_000, 5_000_000, 4096, 500_000_000, 16);
    let error = AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(starved))
        .err()
        .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    // An exhausted node budget of the enclosing scope is a convergence failure.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
    let error = execute(ExecutionContext::sequential(), &mut budgets, |ctx| {
        AmrSpace::build(
            &mesh,
            AmrOrder::Q1,
            AmrOptions::new().with_limits(limits()),
            &mut *ctx,
        )
    })
    .err()
    .unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::ResourceLimit(BudgetKind::Node),
            ..
        }
    ));
}
