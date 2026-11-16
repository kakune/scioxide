//! Analytic AMR topology/space/transfer and elliptic patch coverage.
#![allow(clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AdjointOperator, Complex32, Complex64, ExecutionContext, LinearOperator, NumericScalar,
    RealScalar, ResidualTolerance, SciOxideError,
};
use scioxide_linalg::Matrix;
use scioxide_pde::*;
use scioxide_spatial::*;
#[path = "../examples/support/simplex_fem.rs"]
mod meshes;
#[path = "../examples/support/amr.rs"]
mod support;
use support::*;
fn r<R: RealScalar>(x: f64) -> R {
    R::try_from_f64_lossy(x).unwrap()
}
fn close<V: NumericScalar>(a: V, b: V, factor: f64) {
    let tolerance = V::Real::EPSILON * r(factor) * (V::Real::ONE + a.abs() + b.abs());
    assert!(
        (a - b).abs() <= tolerance,
        "{a:?} != {b:?}; tolerance {tolerance:?}"
    );
}
fn polynomial<V: NumericScalar, const D: usize>(point: [V::Real; D], order: AmrOrder) -> V {
    let sum = point.iter().copied().fold(V::Real::ONE, |a, b| a + b);
    let value = match order {
        AmrOrder::P1 => sum,
        AmrOrder::P2 => sum * sum,
        AmrOrder::Q1 => point
            .iter()
            .copied()
            .fold(V::Real::ONE, |a, b| a * (V::Real::ONE + b)),
    };
    V::from_components(value, -value * r(0.5))
}
fn cell_transfer<V: NumericScalar, const D: usize>(mesh: AdaptiveMesh<V::Real, D>) {
    let initial: Vec<_> = (0..mesh.cell_count())
        .map(|i| V::from_components(r(1. + i as f64), r(-0.3 * i as f64)))
        .collect();
    let field = Field::<V, Cell, D>::new(&mesh, initial).unwrap();
    let changed = adapt(&mesh, &[0], &[]).unwrap();
    let transfer = AmrCellTransfer::new(
        &mesh,
        &changed.mesh,
        &changed.map,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let fine = transfer.field(&changed.mesh, field.view()).unwrap();
    let integral = |mesh: &AdaptiveMesh<V::Real, D>, values: &[V]| {
        values.iter().enumerate().fold(V::ZERO, |s, (i, &v)| {
            s + V::from_real(mesh.cell_measure(i).unwrap()) * v
        })
    };
    close(
        integral(&mesh, field.values()),
        integral(&changed.mesh, fine.values()),
        64.,
    );
    let back = adapt(&changed.mesh, &[], &[0]).unwrap();
    let reverse = AmrCellTransfer::new(
        &changed.mesh,
        &back.mesh,
        &back.map,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let coarse = reverse.field(&back.mesh, fine.view()).unwrap();
    for (&a, &b) in coarse.values().iter().zip(field.values()) {
        close(a, b, 32.);
    }
    let mut out = vec![V::ONE; back.mesh.cell_count()];
    let saved = out.clone();
    let mut scratch = out.clone();
    assert!(
        reverse
            .apply_into(
                field.view(),
                &mut out,
                &mut scratch,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(out, saved);
}
fn nodal<V: NumericScalar, const D: usize>(mesh: AdaptiveMesh<V::Real, D>, order: AmrOrder)
where
    V::Real: NumericScalar<Real = V::Real>,
{
    let coarse =
        AmrSpace::with_options(&mesh, order, AmrOptions::new().with_limits(limits())).unwrap();
    let changed = adapt(&mesh, &[0], &[]).unwrap();
    let fine = AmrSpace::with_options(
        &changed.mesh,
        order,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    if D > 1 && mesh.cell_count() > 1 {
        assert!(fine.independent_dofs() < fine.full_dofs());
    }
    let initial: Vec<_> = (0..coarse.full_dofs())
        .map(|i| polynomial::<V, D>(coarse.node_point(i).unwrap(), order))
        .collect();
    let fields = coarse.field(&initial).unwrap();
    let mut packed = initial.clone();
    coarse.pack_into(fields.view(), &mut packed).unwrap();
    assert_eq!(packed, initial);
    let transfer = AmrNodalTransfer::with_options(&coarse, &fine, &changed.map, limits()).unwrap();
    let mut work = transfer.workspace::<V>().unwrap();
    let mut output = vec![V::ZERO; fine.full_dofs()];
    transfer
        .interpolate_into(
            &initial,
            &mut output,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    fine.validate_field(&output).unwrap();
    for (i, &value) in output.iter().enumerate() {
        close(
            value,
            polynomial::<V, D>(fine.node_point(i).unwrap(), order),
            256.,
        );
    }
    let back = adapt(&changed.mesh, &[], &[0]).unwrap();
    let destination =
        AmrSpace::with_options(&back.mesh, order, AmrOptions::new().with_limits(limits())).unwrap();
    let project = AmrNodalTransfer::with_options(&fine, &destination, &back.map, limits()).unwrap();
    assert!(project.interpolation().is_none());
    let mut work = project.workspace::<V>().unwrap();
    let mut projected = vec![V::ZERO; destination.full_dofs()];
    let tolerance =
        ResidualTolerance::new(V::Real::EPSILON * r(128.), V::Real::EPSILON * r(128.)).unwrap();
    context(|ctx| {
        project.project_into(
            &output,
            tolerance,
            &mut |m, b, x, ctx| solve(m, b, x, ctx),
            &mut projected,
            &mut work,
            ctx,
        )
    })
    .unwrap();
    for (&a, &b) in projected.iter().zip(&initial) {
        close(a, b, 8192.);
    }
    let before = projected.clone();
    assert!(
        context(|ctx| project.project_into(
            &output,
            tolerance,
            &mut |_m, _b, x, _ctx| {
                x.fill(V::ZERO);
                Ok(())
            },
            &mut projected,
            &mut work,
            ctx,
        ))
        .is_err()
    );
    assert_eq!(projected, before);
    // The map's complex adjoint is conjugate-linear in the first pairing argument.
    let p = fine.constraints();
    let x: Vec<_> = fine
        .independent_nodes()
        .iter()
        .map(|&i| polynomial::<V, D>(fine.node_point(i).unwrap(), order))
        .collect();
    let y = output.clone();
    let mut px = y.clone();
    let mut phy = x.clone();
    let mut scratch = y.clone();
    p.apply_into(&x, &mut px, &mut scratch).unwrap();
    p.apply_adjoint_into(&y, &mut phy, &mut scratch[..x.len()])
        .unwrap();
    close(
        px.iter()
            .zip(&y)
            .fold(V::ZERO, |s, (&a, &b)| s + a.conj() * b),
        x.iter()
            .zip(&phy)
            .fold(V::ZERO, |s, (&a, &b)| s + a.conj() * b),
        256.,
    );
}
macro_rules! domains {
    ($name:ident, $v:ty) => {
        #[test]
        fn $name() {
            cell_transfer::<$v, 1>(tensor(2, false).unwrap());
            cell_transfer::<$v, 2>(tensor(2, true).unwrap());
            cell_transfer::<$v, 3>(tensor(2, false).unwrap());
            let triangle =
                AdaptiveMesh::from_simplex(&meshes::triangle_grid(1).unwrap(), topology_limits())
                    .unwrap();
            let tetra = AdaptiveMesh::from_simplex(
                &meshes::tetrahedron_grid(1).unwrap(),
                topology_limits(),
            )
            .unwrap();
            cell_transfer::<$v, 2>(triangle.clone());
            cell_transfer::<$v, 3>(tetra.clone());
            nodal::<$v, 1>(tensor(2, false).unwrap(), AmrOrder::Q1);
            nodal::<$v, 2>(tensor(2, false).unwrap(), AmrOrder::Q1);
            nodal::<$v, 3>(tensor(2, false).unwrap(), AmrOrder::Q1);
            for order in [AmrOrder::P1, AmrOrder::P2] {
                nodal::<$v, 2>(triangle.clone(), order);
                nodal::<$v, 3>(tetra.clone(), order);
            }
        }
    };
}
domains!(real32_transfer_and_constraints, f32);
domains!(real64_transfer_and_constraints, f64);
domains!(complex32_transfer_and_constraints, Complex32);
domains!(complex64_transfer_and_constraints, Complex64);

#[test]
fn multilevel_hanging_constraints_reproduce_polynomials() {
    for order in [AmrOrder::P1, AmrOrder::P2] {
        let mut mesh = AdaptiveMesh::from_simplex(
            &meshes::tetrahedron_grid::<f64>(1).unwrap(),
            topology_limits(),
        )
        .unwrap();
        let mut leaf = 0;
        for _ in 0..3 {
            mesh = adapt(&mesh, &[leaf], &[]).unwrap().mesh;
            leaf = mesh.children(leaf).unwrap()[0];
            let space =
                AmrSpace::with_options(&mesh, order, AmrOptions::new().with_limits(limits()))
                    .unwrap();
            let values: Vec<_> = (0..space.full_dofs())
                .map(|i| polynomial::<Complex64, 3>(space.node_point(i).unwrap(), order))
                .collect();
            space.validate_field(&values).unwrap();
        }
    }
}

struct Patch {
    quadratic: bool,
    mixed: bool,
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AmrDiffusionData<R, V, D>
    for Patch
{
    fn source(&self, _: usize, _: [R; D]) -> Result<V, SciOxideError> {
        let f = if self.quadratic {
            r::<R>(-2. * D as f64)
        } else {
            R::ZERO
        };
        Ok(V::from_components(f, -f * r(0.5)))
    }
}
/// Dirichlet data on tag 0 and, for mixed patches, Robin data on the `x = 1` tag 1.
fn patch_boundaries<V: NumericScalar + Send + Sync + 'static, const D: usize>(
    mesh: &AdaptiveMesh<V::Real, D>,
    data: &Patch,
) -> Result<TaggedBoundaries<V::Real, V, D>, SciOxideError>
where
    V::Real: NumericScalar<Real = V::Real>,
{
    let mixed = data.mixed;
    let order = if data.quadratic {
        AmrOrder::P2
    } else {
        AmrOrder::P1
    };
    let quadratic = data.quadratic;
    let entries: Vec<_> = mesh
        .interfaces()
        .iter()
        .filter(|i| i.right.is_none())
        .map(|i| {
            let right = mesh
                .face_vertices(i.left_face)
                .unwrap()
                .all(|v| mesh.point(v).unwrap().as_array()[0] == r(1.));
            (i.left_face, u32::from(mixed && right))
        })
        .collect();
    let tags = AmrBoundaryTags::new(mesh, &entries)?;
    let mut conditions = vec![BoundaryCondition::dirichlet(
        0,
        BoundaryData::time_invariant(move |p: BoundaryPoint<V::Real, D>| {
            Ok(polynomial::<V, D>(p.point, order))
        }),
    )];
    if mixed {
        conditions.push(BoundaryCondition::robin(
            1,
            V::from_real(r(2.)),
            BoundaryData::time_invariant(move |p: BoundaryPoint<V::Real, D>| {
                let gradient = if quadratic {
                    r::<V::Real>(2.) * p.point.iter().copied().fold(V::Real::ONE, |a, b| a + b)
                } else {
                    V::Real::ONE
                };
                let flux = gradient * p.normal.iter().copied().fold(V::Real::ZERO, |a, b| a + b);
                Ok(V::from_components(flux, -flux * r(0.5))
                    + V::from_real(r(2.)) * polynomial::<V, D>(p.point, order))
            }),
        ));
    }
    TaggedBoundaries::adaptive(mesh, &tags, 1, conditions)
}
fn patch<V: NumericScalar + Send + Sync + 'static, const D: usize>(
    mesh: AdaptiveMesh<V::Real, D>,
    order: AmrOrder,
) where
    V::Real: NumericScalar<Real = V::Real>,
{
    let mesh = adapt(&mesh, &[0], &[]).unwrap().mesh;
    let space =
        AmrSpace::with_options(&mesh, order, AmrOptions::new().with_limits(limits())).unwrap();
    let diffusion = Field::<_, Cell, D>::new(&mesh, vec![V::Real::ONE; mesh.cell_count()]).unwrap();
    let data = Patch {
        quadratic: order == AmrOrder::P2,
        mixed: true,
    };
    let boundaries = patch_boundaries::<V, D>(&mesh, &data).unwrap();
    let plan = AmrDiffusionPlan::<_, V, D>::with_options(
        &space,
        diffusion.view(),
        &boundaries,
        &data,
        limits(),
    )
    .unwrap();
    let csr = plan.assemble_csr().unwrap();
    let dense = Matrix::from_csr(&csr).unwrap();
    let exact: Vec<_> = space
        .independent_nodes()
        .iter()
        .map(|&i| {
            polynomial::<V, D>(
                space.node_point(i).unwrap(),
                if order == AmrOrder::P2 {
                    AmrOrder::P2
                } else {
                    AmrOrder::P1
                },
            )
        })
        .collect();
    let mut expected = vec![V::ZERO; exact.len()];
    let mut actual = expected.clone();
    let mut scratch = vec![V::ZERO; plan.apply_workspace_len().unwrap()];
    plan.apply_into(&exact, &mut actual, &mut scratch).unwrap();
    csr.apply_into(&exact, &mut expected, &mut scratch[..exact.len()])
        .unwrap();
    for ((&a, &b), &rhs) in actual.iter().zip(&expected).zip(plan.rhs()) {
        close(a, b, 128.);
        close(a, rhs, 1024.);
    }
    dense
        .view()
        .try_finite()
        .unwrap()
        .apply_into(&exact, &mut expected, &mut scratch[..exact.len()])
        .unwrap();
    for (&a, &b) in actual.iter().zip(&expected) {
        close(a, b, 128.);
    }
    context(|ctx| solve(&plan, plan.rhs(), &mut actual, ctx)).unwrap();
    for (&a, &b) in actual.iter().zip(&exact) {
        close(a, b, 2048.);
    }
    let mut full = vec![V::ZERO; space.full_dofs()];
    let mut ws = full.clone();
    space
        .constraints()
        .apply_into(&exact, &mut full, &mut ws)
        .unwrap();
    let residual = plan.residual(&full).unwrap();
    let estimator = AmrEstimator::with_options(&space, diffusion.view(), limits()).unwrap();
    let mut indicators = AmrIndicators::new(&mesh).unwrap();
    let mut work = estimator.workspace().unwrap();
    context(|ctx| estimator.estimate_into(&residual, &mut indicators, &mut work, ctx)).unwrap();
    assert!(
        indicators.global() < V::Real::EPSILON * r(32768.),
        "patch estimate {:?}",
        indicators.global()
    );
}
macro_rules! patches {
    ($name:ident, $v:ty) => {
        #[test]
        fn $name() {
            for order in [AmrOrder::P1, AmrOrder::P2] {
                patch::<$v, 2>(
                    AdaptiveMesh::from_simplex(
                        &meshes::triangle_grid(2).unwrap(),
                        topology_limits(),
                    )
                    .unwrap(),
                    order,
                );
                patch::<$v, 3>(
                    AdaptiveMesh::from_simplex(
                        &meshes::tetrahedron_grid(1).unwrap(),
                        topology_limits(),
                    )
                    .unwrap(),
                    order,
                );
            }
            patch::<$v, 1>(tensor(3, false).unwrap(), AmrOrder::Q1);
            patch::<$v, 2>(tensor(2, false).unwrap(), AmrOrder::Q1);
            patch::<$v, 3>(tensor(2, false).unwrap(), AmrOrder::Q1);
        }
    };
}
patches!(real32_elliptic_patch, f32);
patches!(real64_elliptic_patch, f64);
patches!(complex32_elliptic_patch, Complex32);
patches!(complex64_elliptic_patch, Complex64);

#[test]
fn indexed_boundary_residual_validates_faces_and_evaluates_robin_values() {
    let mesh = adapt(&tensor::<f64, 2>(2, false).unwrap(), &[0], &[])
        .unwrap()
        .mesh;
    let space =
        AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(limits()))
            .unwrap();
    let tags = tags(&mesh).unwrap();
    let boundaries = TaggedBoundaries::adaptive(
        &mesh,
        &tags,
        1,
        vec![BoundaryCondition::robin(
            0,
            2.0,
            BoundaryData::time_invariant(|_: BoundaryPoint<f64, 2>| Ok(7.0)),
        )],
    )
    .unwrap();
    let diffusion = Field::<_, Cell, 2>::new(&mesh, vec![1.0; mesh.cell_count()]).unwrap();
    let data = Patch {
        quadratic: false,
        mixed: false,
    };
    let plan =
        AmrDiffusionPlan::with_options(&space, diffusion.view(), &boundaries, &data, limits())
            .unwrap();
    let values: Vec<_> = (0..space.full_dofs())
        .map(|node| {
            let [x, y] = space.node_point(node).unwrap();
            1.0 + x + 2.0 * y
        })
        .collect();
    let residual = plan.residual(&values).unwrap();
    for face in mesh.interfaces() {
        let vertices: Vec<_> = mesh.face_vertices(face.left_face).unwrap().collect();
        let point = core::array::from_fn(|axis| {
            vertices
                .iter()
                .map(|&v| mesh.point(v).unwrap().as_array()[axis])
                .sum::<f64>()
                / vertices.len() as f64
        });
        let value = residual.boundary_flux(face.left_face, point, [0.0; 2]);
        if face.right.is_some() {
            assert!(value.is_err());
        } else {
            close(
                value.unwrap().unwrap(),
                7.0 - 2.0 * (1.0 + point[0] + 2.0 * point[1]),
                64.0,
            );
        }
    }
    assert!(
        residual
            .boundary_flux(mesh.face_count(), [0.0; 2], [0.0; 2])
            .is_err()
    );
}

#[test]
fn marking_is_bulk_deterministic_and_coarsens_only_complete_families() {
    let mesh = tensor::<f64, 1>(4, false).unwrap();
    let indicators = AmrIndicators::from_squared(&mesh, &[1., 4., 4., 1.]).unwrap();
    let marks =
        context(|ctx| AmrMarking::new(0.5).mark_scoped(&mesh, &indicators, limits(), ctx)).unwrap();
    assert_eq!(marks.refine(), &[1, 2]);
    for owned in [
        AmrMarking::new(0.5).mark(&mesh, &indicators).unwrap(),
        AmrMarking::new(0.5)
            .mark_with_options(&mesh, &indicators, limits())
            .unwrap(),
    ] {
        assert_eq!(owned.refine(), marks.refine());
        assert_eq!(owned.coarsen(), marks.coarsen());
    }

    let fine = mesh
        .adapt_scoped(
            marks.request(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap()
        .mesh;
    let zeros = AmrIndicators::from_squared(&fine, &vec![0.; fine.cell_count()]).unwrap();
    let coarsen = context(|ctx| {
        AmrMarking::new(0.5)
            .with_coarsen_threshold(0.)
            .mark_scoped(&fine, &zeros, limits(), ctx)
    })
    .unwrap();
    assert!(coarsen.refine().is_empty());
    assert_eq!(coarsen.coarsen(), &[1, 2]);
    assert!(
        context(|ctx| AmrMarking::new(0.).mark_scoped(&mesh, &indicators, limits(), ctx)).is_err()
    );
    assert!(
        context(|ctx| AmrMarking::new(0.5).mark_scoped(&fine, &indicators, limits(), ctx)).is_err()
    );
}

#[test]
fn coarsening_skips_inactive_children_and_refinement_conflicts() {
    let root = tensor::<f64, 1>(2, false).unwrap();
    let first = adapt(&root, &[0, 1], &[]).unwrap().mesh;
    let child = first.children(0).unwrap()[0];
    let mesh = adapt(&first, &[child], &[]).unwrap().mesh;
    let zeros = AmrIndicators::new(&mesh).unwrap();
    let marks = context(|ctx| {
        AmrMarking::new(0.5)
            .with_coarsen_threshold(0.0)
            .mark_scoped(&mesh, &zeros, limits(), ctx)
    })
    .unwrap();
    assert!(!marks.coarsen().contains(&0));
    assert!(marks.coarsen().contains(&child));
    let values = vec![1.0; first.cell_count()];
    let indicators = AmrIndicators::from_squared(&first, &values).unwrap();
    let marks = context(|ctx| {
        AmrMarking::new(0.25)
            .with_coarsen_threshold(10.0)
            .mark_scoped(&first, &indicators, limits(), ctx)
    })
    .unwrap();
    assert_eq!(marks.refine().len(), 1);
    let parent = first.parent(marks.refine()[0]).unwrap().unwrap();
    assert!(!marks.coarsen().contains(&parent));
    assert_eq!(marks.coarsen().len(), 1);
}

#[test]
fn boundaries_follow_ancestry_and_reject_merging_distinct_tags() {
    let mesh = tensor::<f64, 2>(1, false).unwrap();
    let tags = tags(&mesh).unwrap();
    let fine = adapt(&mesh, &[0], &[]).unwrap();
    let fine_tags = tags
        .transfer_with_options(&mesh, &fine.mesh, &fine.map, limits())
        .unwrap();
    let back = adapt(&fine.mesh, &[], &[0]).unwrap();
    let back_tags = fine_tags
        .transfer_with_options(&fine.mesh, &back.mesh, &back.map, limits())
        .unwrap();
    assert_eq!(tags.view().values(), back_tags.view().values());
    let entries: Vec<_> = fine
        .mesh
        .interfaces()
        .iter()
        .filter(|i| i.right.is_none())
        .enumerate()
        .map(|(n, i)| (i.left_face, n as u32))
        .collect();
    let incompatible = AmrBoundaryTags::new(&fine.mesh, &entries).unwrap();
    assert!(
        incompatible
            .transfer_with_options(&fine.mesh, &back.mesh, &back.map, limits())
            .is_err()
    );
    assert!(AmrBoundaryTags::new(&mesh, &[]).is_err());
}

#[test]
fn invalid_domains_and_limits_fail_before_callbacks() {
    let mesh = tensor::<f64, 2>(1, false).unwrap();
    assert!(
        AmrSpace::with_options(&mesh, AmrOrder::P1, AmrOptions::new().with_limits(limits()))
            .is_err()
    );
    assert!(
        AmrSpace::with_options(
            &tensor::<f64, 2>(2, true).unwrap(),
            AmrOrder::Q1,
            AmrOptions::new().with_limits(limits())
        )
        .is_err()
    );
    let base = limits();
    let (dofs, entries, nodes, work, bytes) = (
        base.max_dofs(),
        base.max_entries(),
        base.max_quadrature_nodes(),
        base.max_work(),
        base.max_storage_bytes(),
    );
    for l in [
        AmrLimits::new(0, entries, nodes, work, bytes),
        AmrLimits::new(dofs, entries, nodes, work, 1),
        AmrLimits::new(dofs, entries, nodes, 0, bytes),
        AmrLimits::new(dofs, entries, 1, work, bytes),
    ] {
        assert!(
            AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(l)).is_err()
        );
    }
    let space =
        AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(limits()))
            .unwrap();
    let k = Field::new(&mesh, vec![-1.]).unwrap();
    let data = Patch {
        quadratic: false,
        mixed: false,
    };
    let boundaries = patch_boundaries::<f64, 2>(&mesh, &data).unwrap();
    assert!(
        AmrDiffusionPlan::<_, f64, 2>::with_options(&space, k.view(), &boundaries, &data, limits())
            .is_err()
    );
    assert!(AmrIndicators::from_squared(&mesh, &[f64::NAN]).is_err());
}

#[test]
fn underintegrated_simplex_mass_rules_are_rejected() {
    let mesh = AdaptiveMesh::from_simplex(
        &meshes::tetrahedron_grid::<f64>(1).unwrap(),
        topology_limits(),
    )
    .unwrap();
    let options = AmrOptions::new().with_limits(limits());
    assert!(
        AmrSpace::with_options(&mesh, AmrOrder::P1, options.with_quadrature_degree(1)).is_err()
    );
    assert!(
        AmrSpace::with_options(&mesh, AmrOrder::P2, options.with_quadrature_degree(3)).is_err()
    );
    assert!(
        AmrSpace::with_options(&mesh, AmrOrder::P2, options.with_quadrature_degree(8)).is_err()
    );
    assert!(AmrSpace::with_options(&mesh, AmrOrder::P2, options.with_quadrature_degree(4)).is_ok());
}
