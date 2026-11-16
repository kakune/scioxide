//! Conservative and constrained transfer pullbacks, with explicit geometry policy.
#![allow(clippy::unwrap_used)]
use scioxide_core::NumericScalar;
use scioxide_core::*;
use scioxide_linalg::Matrix;
use scioxide_pde::*;
use scioxide_spatial::*;
#[path = "../examples/support/amr.rs"]
mod amr;
use amr::*;

#[test]
fn cell_pullback_observes_cancellation_before_publication() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct Stop(AtomicUsize);
    impl Cancellation for Stop {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) != 0
        }
    }
    let mesh = tensor::<f64, 2>(2, false).unwrap();
    let changed = adapt(&mesh, &[0], &[]).unwrap();
    let transfer =
        AmrCellTransfer::new(&mesh, &changed.mesh, &changed.map, AmrOptions::new()).unwrap();
    let seed =
        Field::<f64, Cell, 2>::new(&changed.mesh, vec![1.0; changed.mesh.cell_count()]).unwrap();
    let mut out = vec![7.0; mesh.cell_count()];
    let mut scratch = vec![0.0; mesh.cell_count()];
    let stop = Stop(AtomicUsize::new(0));
    assert!(
        transfer
            .pullback_into(
                mesh.mesh_token(),
                seed.view(),
                &mut out,
                &mut scratch,
                ExecutionContext::sequential().with_cancellation(&stop)
            )
            .is_err()
    );
    assert!(out.iter().all(|&x| x == 7.0));
}
fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn dot<T: NumericScalar>(a: &[T], b: &[T]) -> T::Real {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x.conj() * y).real_component())
        .fold(T::Real::ZERO, |x, y| x + y)
}
fn close<S: RealScalar>(a: S, b: S) {
    assert!(
        (a - b).abs() < r::<S>(4096.) * S::EPSILON * (S::ONE + a.abs() + b.abs()),
        "{a:?} != {b:?}"
    );
}
fn nodal<T: NumericScalar>()
where
    T::Real: NumericScalar<Real = T::Real>,
{
    let mesh = tensor::<T::Real, 2>(2, false).unwrap();
    let changed = adapt(&mesh, &[0], &[]).unwrap();
    let coarse =
        AmrSpace::with_options(&mesh, AmrOrder::Q1, AmrOptions::new().with_limits(limits()))
            .unwrap();
    let fine = AmrSpace::with_options(
        &changed.mesh,
        AmrOrder::Q1,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let map = AmrNodalTransfer::with_options(&coarse, &fine, &changed.map, limits()).unwrap();
    let initial: Vec<T> = (0..coarse.full_dofs())
        .map(|i| {
            let p = coarse.node_point(i).unwrap();
            T::from_components(T::Real::ONE + p[0] + p[1], p[0] - p[1])
        })
        .collect();
    let mut forward = vec![T::ZERO; fine.full_dofs()];
    let mut fw = map.workspace::<T>().unwrap();
    map.interpolate_into(
        &initial,
        &mut forward,
        &mut fw,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let seed: Vec<T> = (0..fine.full_dofs())
        .map(|i| T::from_components(r((i + 1) as f64), r(-0.2)))
        .collect();
    let mut bar = vec![T::ZERO; coarse.full_dofs()];
    let mut bw = map.workspace::<T>().unwrap();
    context(|ctx| {
        map.interpolate_pullback_into(
            mesh.mesh_token(),
            changed.mesh.mesh_token(),
            &seed,
            &mut bar,
            &mut bw,
            ctx,
        )
    })
    .unwrap();
    close(dot(&seed, &forward), dot(&bar, &initial));
    let back = adapt(&changed.mesh, &[], &[0]).unwrap();
    let destination = AmrSpace::with_options(
        &back.mesh,
        AmrOrder::Q1,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let projection =
        AmrNodalTransfer::with_options(&fine, &destination, &back.map, limits()).unwrap();
    let mut projected = vec![T::ZERO; destination.full_dofs()];
    let mut pw = projection.workspace::<T>().unwrap();
    let tol = ResidualTolerance::new(
        r::<T::Real>(64.) * T::Real::EPSILON,
        r::<T::Real>(64.) * T::Real::EPSILON,
    )
    .unwrap();
    context(|ctx| {
        projection.project_into(
            &forward,
            tol,
            &mut |a, b, x, c| solve(a, b, x, c),
            &mut projected,
            &mut pw,
            ctx,
        )
    })
    .unwrap();
    let seed: Vec<T> = (0..destination.full_dofs())
        .map(|i| T::from_components(r((i + 1) as f64), r(0.3)))
        .collect();
    let mut bar = vec![T::ZERO; fine.full_dofs()];
    let mut work = projection.workspace::<T>().unwrap();
    context(|ctx| {
        projection.project_pullback_into(
            changed.mesh.mesh_token(),
            back.mesh.mesh_token(),
            &seed,
            tol,
            &mut |a, b, x, c| solve(a, b, x, c),
            &mut bar,
            &mut work,
            ctx,
        )
    })
    .unwrap();
    close(dot(&seed, &projected), dot(&bar, &forward));
    let saved = bar.clone();
    assert!(
        context(|ctx| projection.project_pullback_into(
            changed.mesh.mesh_token(),
            back.mesh.mesh_token(),
            &seed,
            tol,
            &mut |_, _, x, _| {
                x.fill(T::ZERO);
                Ok(())
            },
            &mut bar,
            &mut work,
            ctx,
        ))
        .is_err()
    );
    assert_eq!(bar, saved);
    assert!(
        context(|ctx| projection.project_pullback_into(
            mesh.mesh_token(),
            back.mesh.mesh_token(),
            &seed,
            tol,
            &mut |a, b, x, c| solve(a, b, x, c),
            &mut bar,
            &mut work,
            ctx,
        ))
        .is_err()
    );
    assert_eq!(bar, saved);
    let cell = AmrCellTransfer::new(
        &mesh,
        &changed.mesh,
        &changed.map,
        AmrOptions::new().with_limits(limits()),
    )
    .unwrap();
    let input = Field::<T, Cell, 2>::new(&mesh, vec![T::ONE; mesh.cell_count()]).unwrap();
    let output = cell.field(&changed.mesh, input.view()).unwrap();
    let cotangent =
        Field::<T, Cell, 2>::new(&changed.mesh, vec![T::ONE; changed.mesh.cell_count()]).unwrap();
    let mut source_bar = vec![T::ZERO; mesh.cell_count()];
    let mut scratch = vec![T::ZERO; mesh.cell_count()];
    cell.pullback_into(
        mesh.mesh_token(),
        cotangent.view(),
        &mut source_bar,
        &mut scratch,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    close(
        dot(cotangent.values(), output.values()),
        dot(&source_bar, input.values()),
    );
}
#[test]
fn constrained_and_conservative_maps() {
    nodal::<f32>();
    nodal::<f64>();
    nodal::<Complex32>();
    nodal::<Complex64>();
}
fn smooth<T: NumericScalar>() {
    let source = tensor::<T::Real, 1>(2, false).unwrap();
    let dest = tensor::<T::Real, 1>(1, false).unwrap();
    let p = r::<T::Real>(0.3);
    let input = [
        T::from_components(r(2.), r(0.5)),
        T::from_components(r(-1.), r(0.2)),
    ];
    // Conservative coarsening for volumes p and 1-p, total volume one.
    let map =
        Matrix::from_vec(1, 2, vec![T::from_real(p), T::from_real(T::Real::ONE - p)]).unwrap();
    let parameter = Matrix::from_vec(1, 1, vec![input[0] - input[1]]).unwrap();
    let seed = [T::from_components(r(0.7), r(0.4))];
    for differentiated in [false, true] {
        let deriv = if differentiated {
            Some(&parameter.view().try_finite().unwrap() as &dyn AdjointOperator<T>)
        } else {
            None
        };
        let map_finite = map.view().try_finite().unwrap();
        let plan = PdeTransferDerivatives::new(
            source.mesh_token(),
            dest.mesh_token(),
            &map_finite,
            deriv,
            1,
        )
        .unwrap();
        let mut work = plan.workspace().unwrap();
        let mut bar = [T::ZERO; 2];
        let mut gp = [T::Real::ZERO; 1];
        context(|ctx| {
            plan.pullback_into(
                source.mesh_token(),
                dest.mesh_token(),
                &seed,
                &mut bar,
                &mut gp,
                &mut work,
                ctx,
            )
        })
        .unwrap();
        close(
            gp[0],
            if differentiated {
                dot(&seed, &[input[0] - input[1]])
            } else {
                T::Real::ZERO
            },
        );
        let direction = [
            T::from_components(r(0.2), r(0.1)),
            T::from_components(r(-0.3), r(0.2)),
        ];
        let dp = [r::<T::Real>(0.4)];
        let mut tangent = [T::ZERO];
        context(|ctx| {
            plan.jvp_into(
                source.mesh_token(),
                dest.mesh_token(),
                &direction,
                &dp,
                &mut tangent,
                &mut work,
                ctx,
            )
        })
        .unwrap();
        close(dot(&seed, &tangent), dot(&bar, &direction) + gp[0] * dp[0]);
        let saved = bar;
        let saved_p = gp;
        assert!(
            context(|ctx| plan.pullback_into(
                dest.mesh_token(),
                source.mesh_token(),
                &seed,
                &mut bar,
                &mut gp,
                &mut work,
                ctx
            ))
            .is_err()
        );
        assert_eq!(bar, saved);
        assert_eq!(gp, saved_p);
    }
    // A parameter operator with the wrong column count is rejected.
    assert!(
        PdeTransferDerivatives::new(
            source.mesh_token(),
            dest.mesh_token(),
            &map.view().try_finite().unwrap(),
            Some(&parameter.view().try_finite().unwrap() as &dyn AdjointOperator<T>),
            2
        )
        .is_err()
    );
}
#[test]
fn parameter_independent_and_dependent_weights() {
    smooth::<f32>();
    smooth::<f64>();
    smooth::<Complex32>();
    smooth::<Complex64>();
}

#[path = "../examples/support/adjoint_epochs.rs"]
mod epochs;
#[test]
fn reverse_two_epochs_includes_transfer_and_initial_data() {
    let (actual, expected) = epochs::run().unwrap();
    assert!((actual - expected).abs() < 1e-12);
}
