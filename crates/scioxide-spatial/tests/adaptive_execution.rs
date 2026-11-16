//! Failure-safe solve accounting, injected execution, and allocation contracts.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, Complex32, Complex64, ExecutionContext, ExecutionPlan,
    ExecutionPolicy, Executor, PartitionFn, ProblemToken, SciOxideError, SequentialExecutor,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_spatial::*;

fn limits() -> AdaptationLimits {
    AdaptationLimits::new()
        .with_max_depth(10)
        .with_max_cells(2000)
        .with_max_entities(20000)
        .with_max_work(100_000_000)
        .with_max_storage_bytes(128 * 1024 * 1024)
}
fn mesh() -> AdaptiveMesh<f64, 1> {
    let grid = Grid1d::new(
        [CoordinateAxis::uniform(0., 1., 301).unwrap()],
        [AxisTopology::Bounded],
    )
    .unwrap();
    AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap()
}
struct Threads;
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
            std::thread::scope(|threads| {
                threads
                    .spawn(|| operation(partition, (), scope))
                    .join()
                    .unwrap()
            })
        })
    }
}
#[test]
fn injected_gathers_match_all_scalar_domains_and_preserve_output_on_failure() {
    fn check<T: Copy + Send + Sync + PartialEq + core::fmt::Debug>(input: Vec<T>) {
        let mesh = mesh();
        let a = MeshPartition::balanced(&mesh, 4).unwrap();
        let assignments = (0..mesh.cell_count()).map(|i| i % 3).collect::<Vec<_>>();
        let b = MeshPartition::new(&mesh, 3, &assignments).unwrap();
        let map = RepartitionMap::with_options(
            &a,
            &b,
            RepartitionOptions::new().with_execution_policy(ExecutionPolicy::Auto),
        )
        .unwrap();
        let mut output = input.clone();
        let mut scratch = input.clone();
        let mut expected = input.clone();
        map.gather_into(
            &input,
            &mut expected,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        map.gather_into(
            &input,
            &mut output,
            &mut scratch,
            ExecutionContext::new(&Threads),
        )
        .unwrap();
        assert_eq!(output, expected);
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(
            map.gather_into(
                &input,
                &mut output,
                &mut scratch,
                ExecutionContext::new(&Threads).with_cancellation(&cancellation)
            )
            .is_err()
        );
        assert_eq!(output, expected);
        {
            let clone = mesh.clone();
            assert_eq!(clone.mesh_token(), mesh.mesh_token());
            for cell in 0..mesh.cell_count() {
                std::hint::black_box(mesh.cell_vertices(cell).unwrap().count());
                std::hint::black_box(mesh.cell_faces(cell).unwrap().count());
            }
            std::hint::black_box(mesh.interfaces());
            std::hint::black_box(mesh.hanging_entities());
            let _ = std::hint::black_box(FieldView::<_, Cell, 1>::new(&mesh, &input).unwrap());
            std::hint::black_box(a.ghost_cells(0).unwrap().count());
            std::hint::black_box(a.interfaces(0).unwrap().count());
            std::hint::black_box(map.migration_cells(0, 1).unwrap().count());
            map.gather_into(
                &input,
                &mut output,
                &mut scratch,
                ExecutionContext::sequential(),
            )
            .unwrap();
        };
    }
    check((0..300).map(|i| i as f32).collect());
    check((0..300).map(|i| i as f64).collect());
    check((0..300).map(|i| Complex32::new(i as f32, -2.)).collect());
    check((0..300).map(|i| Complex64::new(i as f64, -2.)).collect());
}
#[test]
fn context_limits_and_cancellation_do_not_publish() {
    let grid = Grid1d::new(
        [CoordinateAxis::uniform(0., 1., 3).unwrap()],
        [AxisTopology::Bounded],
    )
    .unwrap();
    let mesh = AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap();
    let identity = SolveIdentity::new(
        ProblemToken::new(570),
        scioxide_core::algorithm_id!("spatial", "adaptive", 1),
        0,
        0,
    );
    let request = AdaptationRequest {
        source: mesh.mesh_token(),
        refine: &[0],
        coarsen: &[],
    };
    for kind in [BudgetKind::Node, BudgetKind::Subdivision] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let accepted = context.accepted_state();
        assert!(
            mesh.adapt_scoped(
                AdaptationRequest {
                    refine: &[0, 1],
                    ..request
                },
                &mut context
            )
            .is_err()
        );
        assert_eq!(context.accepted_state(), accepted);
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    assert!(mesh.adapt_scoped(request, &mut context).is_err());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::new(&Threads), identity, &mut budgets)
        .unwrap();
    let accepted = context.accepted_state();
    let result = mesh.adapt_scoped(request, &mut context).unwrap();
    assert_eq!(context.accepted_state(), accepted);
    assert_eq!(result.mesh.cell_count(), 3);
    assert_eq!(context.counts().own().accepted(), 0);
    assert_eq!(context.counts().nested().accepted(), 1);
}

#[test]
fn identity_adaptation_preserves_scope_checks_and_ancestor_state() {
    let mesh = mesh();
    let request = AdaptationRequest {
        source: mesh.mesh_token(),
        refine: &[],
        coarsen: &[],
    };
    let baseline = mesh
        .adapt_scoped(request, ExecutionContext::sequential())
        .unwrap();
    let identity = SolveIdentity::new(
        ProblemToken::new(1926),
        scioxide_core::algorithm_id!("spatial", "adaptive", 1),
        0,
        0,
    );
    for allowance in [baseline.work - 1, baseline.work] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Node, Budget::try_from(allowance).unwrap())
            .with_limit(BudgetKind::Subdivision, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let accepted = context.accepted_state();
        let result = mesh.adapt_scoped(request, &mut context);
        if allowance < baseline.work {
            assert!(matches!(
                result.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node),
                    ..
                }
            ));
            assert_eq!(context.counts().nested().accepted(), 0);
        } else {
            assert_eq!(result.unwrap().work, baseline.work);
            assert_eq!(context.counts().nested().accepted(), 1);
        }
        assert_eq!(context.accepted_state(), accepted);
        assert_eq!(
            context.budget_report(BudgetKind::Node).consumed(),
            allowance
        );
        assert_eq!(context.budget_report(BudgetKind::Subdivision).consumed(), 0);
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(matches!(
        mesh.adapt_scoped(
            request,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .unwrap_err()
        .view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    let revised = mesh
        .adapt_scoped(
            AdaptationRequest {
                refine: &[0],
                ..request
            },
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(
        revised
            .mesh
            .adapt_scoped(request, ExecutionContext::sequential())
            .is_err()
    );
}

#[test]
fn indexed_work_and_subdivision_budgets_are_exact_and_transactional() {
    let grid = Grid2d::new(
        core::array::from_fn(|_| CoordinateAxis::uniform(0., 1., 3).unwrap()),
        [AxisTopology::Bounded; 2],
    )
    .unwrap();
    let root = AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap();
    let first = root
        .adapt_scoped(
            AdaptationRequest {
                source: root.mesh_token(),
                refine: &[0],
                coarsen: &[],
            },
            ExecutionContext::sequential(),
        )
        .unwrap()
        .mesh;
    let request = AdaptationRequest {
        source: first.mesh_token(),
        refine: first.children(0).unwrap(),
        coarsen: &[],
    };
    let baseline = first
        .adapt_scoped(request, ExecutionContext::sequential())
        .unwrap();
    assert!(baseline.induced_refinements.len() >= 2);
    let work = baseline.work;
    let subdivisions = request.refine.len() + baseline.induced_refinements.len();
    let exact = limits().with_max_work(work);
    let prepare = |limits| -> Result<_, SciOxideError> {
        let grid = Grid2d::new(
            core::array::from_fn(|_| CoordinateAxis::uniform(0., 1., 3).unwrap()),
            [AxisTopology::Bounded; 2],
        )?;
        let root = AdaptiveMesh::from_structured(&CellMesh::new(grid)?, limits)?;
        Ok(root
            .adapt_scoped(
                AdaptationRequest {
                    source: root.mesh_token(),
                    refine: &[0],
                    coarsen: &[],
                },
                ExecutionContext::sequential(),
            )?
            .mesh)
    };
    let run = |limits| -> Result<_, SciOxideError> {
        let first = prepare(limits)?;
        first.adapt_scoped(
            AdaptationRequest {
                source: first.mesh_token(),
                refine: first.children(0)?,
                coarsen: &[],
            },
            ExecutionContext::sequential(),
        )
    };
    assert_eq!(run(exact).unwrap().work, work);
    let identity = SolveIdentity::new(
        ProblemToken::new(1114),
        scioxide_core::algorithm_id!("spatial", "adaptive", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(BudgetKind::Node, Budget::try_from(work).unwrap())
        .with_limit(
            BudgetKind::Subdivision,
            Budget::try_from(subdivisions).unwrap(),
        );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let result = first.adapt_scoped(request, &mut context).unwrap();
    assert_eq!(result.work, work);
    assert_eq!(context.budget_report(BudgetKind::Node).consumed(), work);
    assert_eq!(
        context.budget_report(BudgetKind::Subdivision).consumed(),
        subdivisions
    );
    assert_eq!(result.mesh.interfaces(), baseline.mesh.interfaces());
    assert_eq!(
        result.mesh.hanging_entities(),
        baseline.mesh.hanging_entities()
    );
    assert_eq!(result.induced_refinements, baseline.induced_refinements);
    assert_eq!(result.map.cells(), baseline.map.cells());
    assert_eq!(result.map.vertices(), baseline.map.vertices());
    assert_eq!(result.map.faces(), baseline.map.faces());
    assert_eq!(result.map.edges(), baseline.map.edges());

    for cap in [1, work / 2, work - 1] {
        let restricted = limits().with_max_work(cap);
        assert!(matches!(
            run(restricted).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { .. }
        ));
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::try_from(cap).unwrap());
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let accepted = context.accepted_state();
        let error = first.adapt_scoped(request, &mut context).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node),
                ..
            }
        ));
        assert_eq!(context.accepted_state(), accepted);
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), cap);
    }
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(
        BudgetKind::Subdivision,
        Budget::try_from(subdivisions - 1).unwrap(),
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let accepted = context.accepted_state();
    assert!(first.adapt_scoped(request, &mut context).is_err());
    assert_eq!(context.accepted_state(), accepted);
    for bytes in [0, 128, 4096] {
        assert!(matches!(
            run(limits().with_max_storage_bytes(bytes))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { .. }
        ));
    }
    assert_eq!(first.mesh_token(), request.source);
    assert_eq!(first.cell_count(), 7);
    assert_eq!(run(exact).unwrap().map.edges(), baseline.map.edges());
}

struct StopClock<'a> {
    reads: core::sync::atomic::AtomicUsize,
    after: usize,
    cancellation: Option<&'a CancellationToken>,
}
impl scioxide_core::Clock for StopClock<'_> {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        let reads = self
            .reads
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if reads >= self.after
            && let Some(token) = self.cancellation
        {
            token.cancel();
        }
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_nanos(reads as u64),
        ))
    }
}

#[test]
fn index_and_map_work_poll_stops_before_publication() {
    let mesh = mesh();
    let request = AdaptationRequest {
        source: mesh.mesh_token(),
        refine: mesh.active_cells(),
        coarsen: &[],
    };
    let work = mesh
        .adapt_scoped(request, ExecutionContext::sequential())
        .unwrap()
        .work;
    for after in [work / 4, work - 128] {
        for cancel in [false, true] {
            let token = CancellationToken::new();
            let clock = StopClock {
                reads: core::sync::atomic::AtomicUsize::new(0),
                after,
                cancellation: cancel.then_some(&token),
            };
            let binding = scioxide_core::ClockBinding::new(&clock);
            let deadline = scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_nanos(if cancel { u64::MAX } else { after as u64 }),
            );
            let execution = ExecutionContext::sequential()
                .with_cancellation(&token)
                .with_deadline(scioxide_core::Deadline::at(&binding, deadline))
                .unwrap();
            let identity = SolveIdentity::new(
                ProblemToken::new(1114),
                scioxide_core::algorithm_id!("spatial", "adaptive", 1),
                0,
                0,
            );
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(execution, identity, &mut budgets)
                .unwrap();
            let accepted = context.accepted_state();
            let error = mesh.adapt_scoped(request, &mut context).unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::ExecutionStopped
            );
            assert_eq!(context.accepted_state(), accepted);
            assert!(context.budget_report(BudgetKind::Node).consumed() > 0);
            assert_eq!(mesh.cell_count(), 300);
            assert_eq!(mesh.mesh_token(), request.source);
        }
    }
}

#[test]
fn retained_limits_validate_import_and_survive_revisions_and_partitions() {
    let grid = CellMesh::new(
        Grid1d::new(
            [CoordinateAxis::uniform(0.0_f64, 1.0, 2).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(AdaptationLimits::default(), AdaptationLimits::new());
    let invalid = AdaptationLimits::new().with_max_depth(21);
    assert_eq!(
        AdaptiveMesh::from_structured(&grid, invalid)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    let bounded = limits().with_max_depth(1).with_max_cells(3);
    let mesh = AdaptiveMesh::from_structured(&grid, bounded).unwrap();
    let token = mesh.mesh_token().clone();
    for _ in 0..2 {
        let fine = mesh
            .adapt_scoped(
                AdaptationRequest {
                    source: &token,
                    refine: &[0],
                    coarsen: &[],
                },
                scioxide_core::SolveScope::from(ExecutionContext::sequential()),
            )
            .unwrap()
            .mesh;
        assert_eq!(fine.limits(), bounded);
        let partition = MeshPartition::balanced(&fine, 3).unwrap();
        assert_eq!(partition.limits(), bounded);
        assert!(partition.owned_cells(2).unwrap().is_empty());
        let error = fine
            .adapt_scoped(
                AdaptationRequest {
                    source: fine.mesh_token(),
                    refine: fine.children(0).unwrap(),
                    coarsen: &[],
                },
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit {
            resource, required: 2, limit: 1,
        } if resource == scioxide_core::quantity!("adaptive_depth"))
        );
    }
    assert_eq!(mesh.mesh_token(), &token);
    assert_eq!(mesh.cell_count(), 1);
}

struct FailAfterWork;
impl Executor for FailAfterWork {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, operation)?;
        Err(SciOxideError::external("injected failure after staging"))
    }
}

#[test]
fn gather_borrowed_staging_shapes_scopes_and_executor_failure() {
    let mesh = mesh();
    let source = MeshPartition::balanced(&mesh, 4).unwrap();
    let assignments: Vec<_> = (0..mesh.cell_count()).map(|i| i % 3).collect();
    let destination = MeshPartition::new(&mesh, 3, &assignments).unwrap();
    let options = RepartitionOptions::new().with_execution_policy(ExecutionPolicy::Auto);
    let map = RepartitionMap::with_options(&source, &destination, options).unwrap();
    assert_eq!(map.options().execution_policy(), ExecutionPolicy::Auto);
    let input: Vec<_> = (0..mesh.cell_count()).collect();
    let mut output = vec![usize::MAX; input.len()];
    let mut staging = vec![usize::MAX; input.len() + 3];
    let expected = map.permutation().to_vec();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(1126),
        scioxide_core::algorithm_id!("test", "gather", 1),
        0,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let accepted = parent.accepted_state();
    let counts = parent.counts();
    {
        map.gather_into(&input, &mut output, &mut staging, &mut parent)
            .unwrap();
    };
    assert_eq!(parent.accepted_state(), accepted);
    assert_eq!(parent.counts(), counts);
    assert_eq!(output, expected);
    assert_eq!(&staging[input.len()..], &[usize::MAX; 3]);
    for which in 0..3 {
        output.fill(usize::MAX);
        let n = input.len();
        let error = map
            .gather_into(
                &input[..n - usize::from(which == 0)],
                &mut output[..n - usize::from(which == 1)],
                &mut staging[..n - usize::from(which == 2)],
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert!(output.iter().all(|&v| v == usize::MAX));
    }
    // Execution fails after writing staging, and still cannot publish output.
    staging.fill(usize::MAX);
    assert!(
        map.gather_into(
            &input,
            &mut output,
            &mut staging,
            ExecutionContext::new(&FailAfterWork)
        )
        .is_err()
    );
    assert_eq!(&staging[..input.len()], &expected);
    assert!(output.iter().all(|&v| v == usize::MAX));
    let replacement = MeshPartition::new(&mesh, 3, &assignments).unwrap();
    assert!(map.validate(&source, &replacement).is_err());
    map.gather_into(
        &input,
        &mut output,
        &mut staging,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, expected);
}

#[test]
fn empty_gather_preserves_unused_staging_and_checks_cancellation() {
    let mesh =
        AdaptiveMesh::from_simplex(&TriangleMesh::<f64>::new(vec![], vec![]).unwrap(), limits())
            .unwrap();
    let partition = MeshPartition::balanced(&mesh, 2).unwrap();
    let map = RepartitionMap::new(&partition, &partition).unwrap();
    let mut staging = [42];
    map.gather_into(&[], &mut [], &mut staging, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(staging, [42]);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(
        map.gather_into(
            &[],
            &mut [],
            &mut staging,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert_eq!(staging, [42]);
}
