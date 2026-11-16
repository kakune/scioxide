#![allow(clippy::unwrap_used)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use mesh_fixture::explicit;
#[cfg(feature = "std")]
use mesh_fixture::periodic;
use scioxide_core::{ResidualTolerance, SciOxideError};
use scioxide_linalg::CompatibilityPolicy;
use scioxide_pde::{
    EllipticGauge, EllipticOptions, EllipticProblem, EllipticSolution, PdeNullspaceSource,
    SpatialDiscretization,
};
use support::{DISCRETIZATIONS, EllipticPde, End, Ends, Plan1d, Steady, TestScalar, source, value};

#[derive(Clone)]
struct Constant<S> {
    reaction: S,
    source: S,
}
impl<S: TestScalar> EllipticPde<S> for Constant<S> {
    fn diffusion(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(self.reaction)
    }
    fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(self.source)
    }
}

fn solve<S: TestScalar>(
    plan: &Plan1d<S>,
    problem: &Constant<S>,
    options: EllipticOptions<S>,
) -> Result<EllipticSolution<S>, SciOxideError> {
    let elliptic = options;
    let source = source(&Steady(problem.clone()));
    with_plan!(plan, |p| EllipticProblem::new(p, &source)
        .solve_with_options(&elliptic))
}

fn qualify<S: TestScalar>() {
    let mesh = explicit(
        [0.0, 0.08, 0.27, 0.53, 0.81, 1.0]
            .map(support::scalar::<S>)
            .to_vec(),
    );
    let flux = |x: f64| value(move |_| Ok(support::scalar(x)));
    let tolerance = ResidualTolerance::new(support::scalar(2e-5), support::scalar(2e-5)).unwrap();
    let options = EllipticOptions::new()
        .with_residual_tolerance(tolerance)
        .with_compatibility(CompatibilityPolicy::RequireCompatible, tolerance);
    for discretization in DISCRETIZATIONS {
        for robin in [false, true] {
            // Outward conormal flux ±1 is the derivative of x for unit diffusion.
            let end = |g: f64| {
                if robin {
                    End::robin(S::zero_value(), flux(g))
                } else {
                    End::natural(flux(g))
                }
            };
            let ends = Ends::new(end(-1.0), end(1.0));
            let problem = Constant {
                reaction: S::zero_value(),
                source: S::zero_value(),
            };
            let plan = discretization
                .plan(&mesh, &Steady(problem.clone()), &ends)
                .unwrap();
            assert!(with_plan!(&plan, |p| p.has_constant_modes()));
            let error = solve(&plan, &problem, EllipticOptions::new()).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ));
            let mut previous = None::<Vec<S>>;
            for mean in [0.0, 2.0] {
                let solution = solve(
                    &plan,
                    &problem,
                    options.with_gauge(EllipticGauge::WeightedMean(support::scalar(mean))),
                )
                .unwrap();
                let report = solution.report();
                assert!(report.is_success());
                let d = report.diagnostics().nullspace().unwrap();
                assert!(support::as_f64(d.original_residual_norm()) < 2e-4);
                assert!(support::as_f64(d.projected_residual_norm()) < 2e-4);
                assert!(
                    support::as_f64(report.diagnostics().nullspace_gauge_defect().unwrap()) < 2e-5
                );
                assert_eq!(
                    report.diagnostics().nullspace_source(),
                    Some(PdeNullspaceSource::StructuralConstant)
                );
                let values = plan.full(solution.values()).unwrap();
                let coordinates = discretization.coordinates(&mesh);
                // Independent exact affine solution and unit gradient energy.
                for (&u, &x) in values.iter().zip(&coordinates) {
                    assert!(support::as_f64(u - (x + support::scalar(mean - 0.5))).abs() < 2e-4);
                }
                for (u, x) in values.windows(2).zip(coordinates.windows(2)) {
                    assert!((support::as_f64((u[1] - u[0]) / (x[1] - x[0])) - 1.0).abs() < 2e-4);
                }
                if let Some(before) = previous {
                    for (&a, &b) in values.iter().zip(&before) {
                        assert!((support::as_f64(a - b) - 2.0).abs() < 2e-4);
                    }
                }
                previous = Some(values);
            }
            let incompatible = Constant {
                reaction: S::zero_value(),
                source: S::one_value(),
            };
            let error = solve(
                &plan,
                &incompatible,
                options.with_gauge(EllipticGauge::Orthogonal),
            )
            .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
            let solution = solve(
                &plan,
                &incompatible,
                options
                    .with_gauge(EllipticGauge::Orthogonal)
                    .with_compatibility(CompatibilityPolicy::Project, tolerance),
            )
            .unwrap();
            let report = solution.report();
            assert!(report.is_success());
            let d = report.diagnostics().nullspace().unwrap();
            assert!(support::as_f64(d.projection_norm()) > 0.1);
            assert!(support::as_f64(d.original_residual_norm() - d.projection_norm()).abs() < 2e-4);
            assert!(support::as_f64(d.projected_residual_norm()) < 2e-4);
        }
        let zero = Ends::new(End::natural(flux(0.0)), End::natural(flux(0.0)));
        for reaction in [1e-12, -1e-12] {
            let problem = Constant {
                reaction: support::scalar(reaction),
                source: S::zero_value(),
            };
            let plan = discretization
                .plan(&mesh, &Steady(problem.clone()), &zero)
                .unwrap();
            let solution = solve(&plan, &problem, EllipticOptions::new()).unwrap();
            assert!(solution.report().is_success());
            assert!(solution.report().diagnostics().nullspace().is_none());
        }
    }
}

#[test]
fn constant_modes_f32() {
    qualify::<f32>();
}

#[test]
fn constant_modes_f64() {
    qualify::<f64>();
}

#[test]
fn signed_reaction_nonconstant_discrete_nullspace() {
    let mesh = explicit(vec![0.0, 1.0]);
    let ends = Ends::new(
        End::natural(value(|_| Ok(0.0))),
        End::natural(value(|_| Ok(0.0))),
    );
    #[derive(Clone)]
    struct Resonant;
    impl EllipticPde<f64> for Resonant {
        fn diffusion(&self, _: &[f64]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
        fn reaction(&self, _: &[f64]) -> Result<f64, SciOxideError> {
            Ok(-12.0)
        }
        fn source(&self, _: &[f64]) -> Result<f64, SciOxideError> {
            Ok(1.0)
        }
    }
    let plan = support::Discretization::Fem
        .plan(&mesh, &Steady(Resonant), &ends)
        .unwrap();
    let basis = [0.5_f64.sqrt(), -0.5_f64.sqrt()];
    let b = scioxide_core::BlockVectorView::new(&basis, 2, 1).unwrap();
    let elliptic = EllipticOptions::new().with_gauge(EllipticGauge::Orthogonal);
    let data = source(&Steady(Resonant));
    let solution = with_plan!(&plan, |p| EllipticProblem::new(p, &data)
        .with_nullspace(b, b)
        .solve_with_options(&elliptic))
    .unwrap();
    let report = solution.report();
    assert!(report.is_success());
    assert_eq!(
        report.diagnostics().nullspace_source(),
        Some(PdeNullspaceSource::Supplied)
    );
    // Independently integrated P1 matrix is [-3 -3; -3 -3], load [1/2; 1/2].
    for &u in solution.values() {
        assert!((u + 1.0 / 12.0).abs() < 1e-12);
    }
}

#[cfg(feature = "std")]
#[test]
fn periodic_executor_matches_sequential() {
    let mesh = periodic(&explicit(vec![0.0, 0.2, 0.5, 0.8, 1.0]));
    let problem = Constant {
        reaction: 0.0,
        source: 0.0,
    };
    let plan = support::Discretization::Fdm
        .plan(&mesh, &Steady(problem.clone()), &Ends::periodic())
        .unwrap();
    let options = EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(3.0));
    let elliptic = options;
    let data = source(&Steady(problem));
    let sequential = with_plan!(&plan, |p| EllipticProblem::new(p, &data)
        .solve_with_options(&elliptic))
    .unwrap();
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let mut parallel = vec![0.0; sequential.values().len()];
    let report = with_plan!(&plan, |p| EllipticProblem::new(p, &data).solve_into(
        &elliptic,
        &mut parallel,
        &mut scioxide_pde::EllipticWorkspace::new(),
        scioxide_core::ExecutionContext::new(&pool)
    ))
    .unwrap();
    assert!(report.is_success());
    assert_eq!(parallel, sequential.values());
    assert!(parallel.iter().all(|&u| (u - 3.0_f64).abs() < 1e-12));
}

#[test]
fn natural_multicomponent_tridiagonal_pins_every_component() {
    use scioxide_pde::{
        BoundaryData, BoundaryKind, ConstantDiffusion, FieldData, FieldPoint, StructuredBoundaries,
        StructuredFdmPlan,
    };
    use scioxide_spatial::{AxisTopology, CoordinateAxis, Grid1d, SegmentMesh};
    let solve = |components: usize, gauge: EllipticGauge<f64>| {
        let axis = CoordinateAxis::uniform(0.0_f64, 1.0, 9).unwrap();
        let mesh = SegmentMesh::new(Grid1d::new([axis], [AxisTopology::Bounded]).unwrap()).unwrap();
        let boundaries = StructuredBoundaries::uniform(
            &mesh,
            components,
            BoundaryKind::Natural,
            BoundaryData::zero(),
        )
        .unwrap();
        let plan = StructuredFdmPlan::new(mesh, &ConstantDiffusion::new(1.0), &boundaries).unwrap();
        let source = FieldData::time_invariant(|p: FieldPoint<f64, 1>| {
            Ok((core::f64::consts::PI * p.point[0]).cos())
        });
        let tolerance = ResidualTolerance::new(1e-8, 1e-8).unwrap();
        let options = EllipticOptions::new()
            .with_gauge(gauge)
            .with_compatibility(CompatibilityPolicy::Project, tolerance);
        EllipticProblem::new(&plan, &source)
            .solve_with_options(&options)
            .unwrap()
            .values()
            .to_vec()
    };
    for gauge in [EllipticGauge::Orthogonal, EllipticGauge::WeightedMean(0.0)] {
        let single = solve(1, gauge);
        let pair = solve(2, gauge);
        assert_eq!(pair.len(), 2 * single.len());
        let scale = single.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.01);
        for block in pair.chunks_exact(single.len()) {
            for (a, b) in block.iter().zip(&single) {
                assert!((a - b).abs() <= 1e-12 * scale, "{a} vs {b}");
            }
        }
    }
}
