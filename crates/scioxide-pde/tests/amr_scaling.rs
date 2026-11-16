//! Counted-work bounds for successively refined meshes; no timing assertions.
#![allow(clippy::unwrap_used)]
use scioxide_core::{BudgetKind, ExecutionContext, SciOxideError, SolveContext};
use scioxide_pde::*;
use scioxide_spatial::{AdaptationLimits, AdaptationRequest, AdaptiveMesh, Cell, Field};
#[path = "../examples/support/simplex_fem.rs"]
mod meshes;
#[path = "../examples/support/amr.rs"]
mod support;

fn limits() -> AmrLimits {
    AmrLimits::new(1_000_000, 16_000_000, 128, 2_000_000_000, 2 << 30)
}

fn measured<T>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> (T, usize) {
    support::context(|ctx| {
        let result = run(ctx)?;
        Ok((result, ctx.budget_report(BudgetKind::Node).consumed()))
    })
    .unwrap()
}

fn bound(label: &str, work: usize, cells: usize, factor: usize) {
    let logarithm = (usize::BITS - cells.leading_zeros()) as usize;
    assert!(work > 0, "{label}: no work charged");
    assert!(
        work <= factor * cells * logarithm,
        "{label}: {work} exceeds {factor} n log2(n) at {cells} cells",
    );
}

struct Constant;
impl AmrDiffusionData<f64, f64, 2> for Constant {
    fn source(&self, _: usize, _: [f64; 2]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}

#[test]
fn estimate_mark_coarsen_and_transfer_have_n_log_n_work_bounds() {
    let mesh = AdaptiveMesh::from_simplex(
        &meshes::triangle_grid::<f64>(16).unwrap(),
        AdaptationLimits::new()
            .with_max_cells(200_000)
            .with_max_entities(2_000_000)
            .with_max_work(2_000_000_000)
            .with_max_storage_bytes(2 << 30),
    )
    .unwrap();
    let options = AmrOptions::new()
        .with_limits(limits())
        .with_quadrature_degree(2);
    let mut source = AmrSpace::with_options(&mesh, AmrOrder::P1, options).unwrap();
    let mut previous = None;
    for expected in [2048, 8192, 32_768, 131_072] {
        let changed = source
            .mesh()
            .adapt_scoped(
                AdaptationRequest {
                    source: source.mesh().mesh_token(),
                    refine: source.mesh().active_cells(),
                    coarsen: &[],
                },
                ExecutionContext::sequential(),
            )
            .unwrap();
        let mesh = &changed.mesh;
        assert_eq!(mesh.cell_count(), expected);
        let (destination, preparation) =
            measured(|ctx| AmrSpace::build(mesh, AmrOrder::P1, options, ctx));
        let (_, cell_transfer) =
            measured(|ctx| AmrCellTransfer::build(source.mesh(), mesh, &changed.map, options, ctx));
        let (_, nodal_transfer) = measured(|ctx| {
            AmrNodalTransfer::build(&source, &destination, &changed.map, limits(), ctx)
        });
        let k = Field::<_, Cell, 2>::new(mesh, vec![1.0; expected]).unwrap();
        let tags = support::tags(mesh).unwrap();
        let boundaries = TaggedBoundaries::adaptive(
            mesh,
            &tags,
            1,
            vec![BoundaryCondition::natural(0, BoundaryData::zero())],
        )
        .unwrap();
        let plan = AmrDiffusionPlan::with_options(
            &destination,
            k.view(),
            &boundaries,
            &Constant,
            limits(),
        )
        .unwrap();
        let values = vec![0.0; destination.full_dofs()];
        let residual = plan.residual(&values).unwrap();
        let estimator = AmrEstimator::with_options(&destination, k.view(), limits()).unwrap();
        let mut workspace = estimator.workspace().unwrap();
        let mut indicators = AmrIndicators::new(mesh).unwrap();
        let (_, estimate) = measured(|ctx| {
            estimator.estimate_into(&residual, &mut indicators, &mut workspace, ctx)
        });
        let (_, mark) =
            measured(|ctx| AmrMarking::new(0.5).mark_scoped(mesh, &indicators, limits(), ctx));
        let zeros = AmrIndicators::new(mesh).unwrap();
        let (marks, coarsen) = measured(|ctx| {
            AmrMarking::new(0.5)
                .with_coarsen_threshold(0.0)
                .mark_scoped(mesh, &zeros, limits(), ctx)
        });
        assert!(marks.refine().is_empty());
        assert_eq!(marks.coarsen().len(), source.mesh().cell_count());
        let counts = [
            preparation,
            cell_transfer,
            nodal_transfer,
            estimate,
            mark,
            coarsen,
        ];
        for ((name, factor), count) in [
            ("space", 256),
            ("cell transfer", 32),
            ("nodal transfer", 512),
            ("estimate", 32),
            ("mark", 8),
            ("coarsen", 16),
        ]
        .into_iter()
        .zip(counts)
        {
            bound(name, count, expected, factor);
        }
        if let Some(old) = previous {
            for (count, before) in counts.into_iter().zip(old) {
                assert!(count <= 5 * before, "superlinear work: {before} -> {count}");
            }
        }
        previous = Some(counts);
        drop(residual);
        drop(plan);
        source = destination;
    }
}
