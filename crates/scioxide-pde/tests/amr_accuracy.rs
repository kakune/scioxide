//! Independent analytic error/effectivity and work-accuracy coverage.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, Complex64, ExecutionContext, LinearOperator, ResidualTolerance};
use scioxide_pde::*;
use scioxide_spatial::*;
#[path = "../examples/support/amr.rs"]
mod amr;
#[path = "../examples/support/simplex_fem.rs"]
mod meshes;
#[path = "../examples/support/amr_poisson.rs"]
mod poisson;

#[test]
fn error_driven_refinement_beats_uniform_error_per_dof_and_has_bounded_effectivity() {
    let mesh = AdaptiveMesh::from_simplex(
        &meshes::triangle_grid::<f64>(4).unwrap(),
        amr::topology_limits(),
    )
    .unwrap();
    let mut space = AmrSpace::with_options(
        &mesh,
        AmrOrder::P1,
        AmrOptions::new().with_limits(amr::limits()),
    )
    .unwrap();
    let mut tags = amr::tags(&mesh).unwrap();
    let mut first = 0.;
    let mut last = 0.;
    let mut seed = Vec::new();
    for cycle in 0..6 {
        let (field, indicators) = poisson::solve(&space, &tags, &seed, "amg").unwrap();
        let (_, error) = poisson::errors(&space, &field, 4).unwrap();
        if cycle == 0 {
            first = error;
        }
        last = error;
        let effectivity = indicators.global() / error;
        assert!(
            (2.0..10.0).contains(&effectivity),
            "effectivity {effectivity}"
        );
        if cycle == 5 {
            let (_, higher) = poisson::errors(&space, &field, 5).unwrap();
            assert!((error - higher).abs() < 0.01 * error);
            break;
        }
        let marks = amr::context(|ctx| {
            AmrMarking::new(0.5).mark_scoped(space.mesh(), &indicators, amr::limits(), ctx)
        })
        .unwrap();
        let candidate = space
            .mesh()
            .adapt_scoped(
                marks.request(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        let next_tags = tags
            .transfer_with_options(space.mesh(), &candidate.mesh, &candidate.map, amr::limits())
            .unwrap();
        let next = AmrSpace::with_options(
            &candidate.mesh,
            AmrOrder::P1,
            AmrOptions::new().with_limits(amr::limits()),
        )
        .unwrap();
        let mut next_field = vec![0.; next.full_dofs()];
        {
            let transfer =
                AmrNodalTransfer::with_options(&space, &next, &candidate.map, amr::limits())
                    .unwrap();
            let mut work = transfer.workspace::<f64>().unwrap();
            transfer
                .interpolate_into(
                    &field,
                    &mut next_field,
                    &mut work,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
        }
        space = next;
        tags = next_tags;
        seed = next_field;
    }
    let uniform = AdaptiveMesh::from_simplex(
        &meshes::triangle_grid::<f64>(8).unwrap(),
        amr::topology_limits(),
    )
    .unwrap();
    let uniform = AmrSpace::with_options(
        &uniform,
        AmrOrder::P1,
        AmrOptions::new().with_limits(amr::limits()),
    )
    .unwrap();
    let tags = amr::tags(uniform.mesh()).unwrap();
    let (field, _) = poisson::solve(&uniform, &tags, &[], "amg").unwrap();
    let (_, error) = poisson::errors(&uniform, &field, 4).unwrap();
    assert!(space.independent_dofs() <= uniform.independent_dofs());
    assert!(last < 0.75 * error);
    assert!(last < 0.5 * first);
    println!(
        "adaptive dofs={} energy={last:.9e}; uniform dofs={} energy={error:.9e}",
        space.independent_dofs(),
        uniform.independent_dofs()
    );
}

#[test]
fn borrowed_schwarz_composes_with_existing_gmres_and_matches_amg() {
    let mesh = AdaptiveMesh::from_simplex(
        &meshes::triangle_grid::<f64>(4).unwrap(),
        amr::topology_limits(),
    )
    .unwrap();
    let mesh = amr::adapt(&mesh, &[0, 1, 3], &[]).unwrap().mesh;
    let space = AmrSpace::with_options(
        &mesh,
        AmrOrder::P1,
        AmrOptions::new().with_limits(amr::limits()),
    )
    .unwrap();
    let tags = amr::tags(&mesh).unwrap();
    let (a, _) = poisson::solve(&space, &tags, &[], "amg").unwrap();
    let (b, _) = poisson::solve(&space, &tags, &[], "schwarz").unwrap();
    assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-9));
}

#[test]
fn mixed_refine_coarsen_projection_conserves_independently_integrated_complex_mass() {
    let base = amr::tensor::<f64, 2>(2, false).unwrap();
    let fine = amr::adapt(&base, &[0], &[]).unwrap();
    let next = amr::adapt(&fine.mesh, &[3], &[0]).unwrap();
    let source = AmrSpace::with_options(
        &fine.mesh,
        AmrOrder::Q1,
        AmrOptions::new().with_limits(amr::limits()),
    )
    .unwrap();
    let destination = AmrSpace::with_options(
        &next.mesh,
        AmrOrder::Q1,
        AmrOptions::new().with_limits(amr::limits()),
    )
    .unwrap();
    let independent: Vec<_> = (0..source.independent_dofs())
        .map(|i| Complex64::new((i as f64).sin(), (i as f64 * 0.7).cos()))
        .collect();
    let mut values = vec![Complex64::new(0., 0.); source.full_dofs()];
    let mut scratch = values.clone();
    source
        .constraints()
        .apply_into(&independent, &mut values, &mut scratch)
        .unwrap();
    let transfer =
        AmrNodalTransfer::with_options(&source, &destination, &next.map, amr::limits()).unwrap();
    let mut output = vec![Complex64::new(0., 0.); destination.full_dofs()];
    let mut work = transfer.workspace::<Complex64>().unwrap();
    amr::context(|ctx| {
        transfer.project_into(
            &values,
            ResidualTolerance::new(1e-11, 1e-11).unwrap(),
            &mut |m, b, x, c| amr::solve(m, b, x, c),
            &mut output,
            &mut work,
            ctx,
        )
    })
    .unwrap();
    let integral = |s: &AmrSpace<f64, 2>, v: &[Complex64]| -> Complex64 {
        (0..s.mesh().cell_count())
            .map(|cell| {
                s.cell_dofs(cell)
                    .unwrap()
                    .iter()
                    .map(|&i| v[i])
                    .sum::<Complex64>()
                    * (s.mesh().cell_measure(cell).unwrap() / 4.)
            })
            .sum()
    };
    assert!((integral(&source, &values) - integral(&destination, &output)).norm() < 1e-10);
    destination.validate_field(&output).unwrap();
}

struct Quadratic;
impl<R: scioxide_core::RealScalar, const D: usize> AmrDiffusionData<R, R, D> for Quadratic {
    fn source(&self, _: usize, _: [R; D]) -> Result<R, scioxide_core::SciOxideError> {
        R::try_from_f64_lossy(-2. * D as f64)
    }
}
fn effectivity<
    R: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = R>,
    const D: usize,
>(
    mut mesh: AdaptiveMesh<R, D>,
) {
    let mut previous = None;
    for level in 0..3 {
        let space = AmrSpace::with_options(
            &mesh,
            AmrOrder::P1,
            AmrOptions::new().with_limits(amr::limits()),
        )
        .unwrap();
        let tags = amr::tags(&mesh).unwrap();
        let boundaries = TaggedBoundaries::adaptive(
            &mesh,
            &tags,
            1,
            vec![BoundaryCondition::dirichlet(
                0,
                BoundaryData::time_invariant(|p: BoundaryPoint<R, D>| {
                    let s = p.point.iter().copied().fold(R::ONE, |a, b| a + b);
                    Ok(s * s)
                }),
            )],
        )
        .unwrap();
        let k = Field::<_, Cell, D>::new(&mesh, vec![R::ONE; mesh.cell_count()]).unwrap();
        let plan = AmrDiffusionPlan::with_options(
            &space,
            k.view(),
            &boundaries,
            &Quadratic,
            amr::limits(),
        )
        .unwrap();
        let mut reduced = vec![R::ZERO; space.independent_dofs()];
        amr::context(|ctx| amr::solve(&plan, plan.rhs(), &mut reduced, ctx)).unwrap();
        let mut full = vec![R::ZERO; space.full_dofs()];
        let mut scratch = full.clone();
        space
            .constraints()
            .apply_into(&reduced, &mut full, &mut scratch)
            .unwrap();
        println!(
            "D={D} cells={} full={} independent={} nodal_error={:?}",
            mesh.cell_count(),
            space.full_dofs(),
            space.independent_dofs(),
            full.iter()
                .enumerate()
                .map(|(i, &u)| {
                    let s = space
                        .node_point(i)
                        .unwrap()
                        .iter()
                        .copied()
                        .fold(R::ONE, |a, b| a + b);
                    (u - s * s).abs()
                })
                .fold(R::ZERO, |a, b| a.max(b))
        );
        let rule = scioxide_integrate::SimplexRule::<R>::new(
            D,
            2,
            scioxide_integrate::GeometricOptions::default()
                .with_max_nodes(Budget::try_from(128).unwrap()),
        )
        .unwrap();
        let mut energy = R::ZERO;
        for cell in 0..mesh.cell_count() {
            let vertices = mesh
                .cell_vertices(cell)
                .unwrap()
                .map(|v| *mesh.point(v).unwrap().as_array())
                .collect::<Vec<_>>();
            for (bary, &weight) in rule
                .barycentric_nodes()
                .chunks_exact(D + 1)
                .zip(rule.weights())
            {
                let reference = core::array::from_fn(|a| bary[a + 1]);
                let point: [R; D] = core::array::from_fn(|a| {
                    bary.iter()
                        .zip(&vertices)
                        .fold(R::ZERO, |s, (&w, p)| s + w * p[a])
                });
                let gradient = space.evaluate(cell, reference, &full).unwrap().1;
                let exact = R::try_from_f64_lossy(2.).unwrap()
                    * point.iter().copied().fold(R::ONE, |a, b| a + b);
                let squared = gradient
                    .iter()
                    .fold(R::ZERO, |s, &g| s + (g - exact) * (g - exact));
                let factorial = R::try_from_f64_lossy(if D == 2 { 2. } else { 6. }).unwrap();
                energy += weight * factorial * mesh.cell_measure(cell).unwrap() * squared;
            }
        }
        let error = energy.sqrt().try_to_f64_exact().unwrap();
        let residual = plan.residual(&full).unwrap();
        let estimator = AmrEstimator::with_options(&space, k.view(), amr::limits()).unwrap();
        let mut out = AmrIndicators::new(&mesh).unwrap();
        let mut work = estimator.workspace().unwrap();
        amr::context(|ctx| estimator.estimate_into(&residual, &mut out, &mut work, ctx)).unwrap();
        let ratio = out.global().try_to_f64_exact().unwrap() / error;
        assert!((0.1..20.).contains(&ratio));
        println!("level={level} energy={error} effectivity={ratio}");
        if let Some(old) = previous {
            assert!(error < old);
            if level == 2 {
                assert!(
                    error < old * 0.7,
                    "D={D}, width={}, energy {old} -> {error}",
                    core::mem::size_of::<R>()
                );
            }
        }
        previous = Some(error);
        if level < 2 {
            let marks = mesh.active_cells().to_vec();
            mesh = amr::adapt(&mesh, &marks, &[]).unwrap().mesh;
        }
    }
}
#[test]
fn manufactured_2d_3d_effectivity_both_precisions() {
    effectivity(
        AdaptiveMesh::from_simplex(
            &meshes::triangle_grid::<f32>(2).unwrap(),
            amr::topology_limits(),
        )
        .unwrap(),
    );
    effectivity(
        AdaptiveMesh::from_simplex(
            &meshes::triangle_grid::<f64>(2).unwrap(),
            amr::topology_limits(),
        )
        .unwrap(),
    );
    effectivity(
        AdaptiveMesh::from_simplex(
            &meshes::tetrahedron_grid::<f32>(1).unwrap(),
            amr::topology_limits(),
        )
        .unwrap(),
    );
    effectivity(
        AdaptiveMesh::from_simplex(
            &meshes::tetrahedron_grid::<f64>(1).unwrap(),
            amr::topology_limits(),
        )
        .unwrap(),
    );
}
