//! Range and publication contracts of mapped geometric rules.
#![allow(clippy::unwrap_used)]

use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{ExecutionContext, Interval, NumericalIssue, SciOxideErrorView};
use scioxide_integrate::{
    GeometricOptions, IntegrationTermination, LebedevRule, SimplexRule, SparseGridOptions,
    SparseGridPlan,
};

#[test]
fn cubature_does_not_overflow_the_split_indicator_of_a_large_constant() {
    use scioxide_core::RealScalar;
    use scioxide_integrate::{CubaturePlan, SubdivisionOptions};
    fn check<S: RealScalar + Default>() {
        let amplitude = S::MAX * S::from_f64_ieee(0.75);
        let plan = CubaturePlan::new(
            &[Interval::new(S::ZERO, S::ONE).unwrap(); 2],
            SubdivisionOptions::default(),
        )
        .unwrap();
        let (value, report) = plan.integrate(|_: &[S]| Ok(amplitude)).unwrap();
        assert!(report.is_success());
        assert!((value / amplitude - S::ONE).abs() <= S::from_f64_ieee(128.0) * S::EPSILON);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn double_exponential_keeps_overflowing_absolute_mass_in_extended_range() {
    use scioxide_core::{ErrorTolerance, NormSelection, RealScalar};
    use scioxide_integrate::{
        DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint, LevelOptions,
    };
    fn check<S: RealScalar + Default>() {
        let tolerance = S::MAX * S::EPSILON.sqrt();
        let plan = DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::Finite {
                interval: Interval::new(-S::ONE, S::ONE).unwrap(),
                singular_points: &[],
            },
            LevelOptions::default().with_tolerance(
                ErrorTolerance::scalar(tolerance, S::ZERO, NormSelection::Infinity).unwrap(),
            ),
        )
        .unwrap();
        let amplitude = S::MAX * S::try_from_f64_lossy(0.75).unwrap();
        let (value, report) = plan
            .integrate(|point: DoubleExponentialPoint<S>| {
                Ok(if point.coordinate() < S::ZERO {
                    -amplitude
                } else if point.coordinate() > S::ZERO {
                    amplitude
                } else {
                    S::ZERO
                })
            })
            .unwrap();
        assert!(report.is_success(), "{report:?}");
        assert!(value.abs() <= tolerance);
        assert!(report.error_estimate().unwrap() <= tolerance);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn weighted_rules_keep_finite_results_across_overflowing_spans() {
    use scioxide_core::RealScalar;
    use scioxide_integrate::{FourierWeight, WeightedQuadratureOptions, WeightedQuadraturePlan};
    fn check<S: RealScalar + Default>() {
        let interval = Interval::new(-S::MAX, S::MAX).unwrap();
        let reciprocal = S::ONE / S::MAX;
        let tolerance = S::try_from_usize_exact(128).unwrap() * S::EPSILON;
        for omega in [S::ZERO, reciprocal] {
            let plan = WeightedQuadraturePlan::fourier(
                interval,
                &[],
                omega,
                FourierWeight::Cosine,
                WeightedQuadratureOptions::default(),
            )
            .unwrap();
            let (value, report) = plan.integrate(|_: S| Ok(S::EPSILON)).unwrap();
            assert!(report.is_success());
            let expected = if omega == S::ZERO {
                S::TWO
            } else {
                S::TWO * S::ONE.sin()
            };
            assert!((value / (S::MAX * S::EPSILON) - expected).abs() < tolerance);
        }
        let plan = WeightedQuadraturePlan::principal_value(
            interval,
            S::MAX * S::HALF,
            WeightedQuadratureOptions::default(),
        )
        .unwrap();
        let (value, report) = plan.integrate(|_: S| Ok(S::ONE)).unwrap();
        assert!(report.is_success());
        assert!((value + S::try_from_usize_exact(3).unwrap().ln()).abs() < tolerance);
        let pole = S::EPSILON * S::EPSILON;
        let plan = WeightedQuadraturePlan::principal_value(
            Interval::new(-S::ONE, S::ONE).unwrap(),
            pole,
            WeightedQuadratureOptions::default(),
        )
        .unwrap();
        let (value, report) = plan.integrate(|_: S| Ok(S::ONE)).unwrap();
        assert!(report.is_success());
        assert!((value / (-S::TWO * pole) - S::ONE).abs() < tolerance);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn contour_products_round_after_combining_the_real_products() {
    use scioxide_core::{Complex, RealScalar};
    use scioxide_integrate::{ContourPlan, SubdivisionOptions};
    fn check<S: RealScalar>() {
        let tiny = S::ZERO.next_after(S::ONE);
        let plan = ContourPlan::new(
            Interval::new(S::ZERO, S::ONE).unwrap(),
            &[],
            SubdivisionOptions::default(),
        )
        .unwrap();
        let (value, report) = plan
            .integrate(
                |_: Complex<S>| Ok(Complex::new(tiny, tiny)),
                |t: S| {
                    Ok((
                        Complex::new(t * S::HALF, t * S::HALF),
                        Complex::new(S::HALF, S::HALF),
                    ))
                },
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(value.re, S::ZERO);
        assert_eq!(value.im, tiny);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn sparse_grid_rejects_collapsed_nodes_before_sampling_them() {
    fn check<S: scioxide_core::RealScalar>() {
        let plan = SparseGridPlan::<S>::new(1, SparseGridOptions::default()).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        for steps in [1, 2] {
            let calls = AtomicUsize::new(0);
            let mut value = S::try_from_usize_exact(7).unwrap();
            let bounds = [Interval::new(
                S::ONE,
                S::ONE + S::try_from_usize_exact(steps).unwrap() * S::EPSILON,
            )
            .unwrap()];
            let result = plan.integrate_into(
                |_: &[S]| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(S::ONE)
                },
                &bounds,
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            );
            if steps == 1 {
                assert!(matches!(
                    result.unwrap_err().view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Breakdown,
                        ..
                    }
                ));
                assert_eq!(value, S::try_from_usize_exact(7).unwrap());
                assert_eq!(calls.load(Ordering::Relaxed), 0);
            } else {
                let report = result.unwrap();
                assert_eq!(
                    report.termination(),
                    IntegrationTermination::GeometryFailure
                );
                assert_eq!(calls.load(Ordering::Relaxed), 3);
                assert!(
                    (value - S::TWO * S::EPSILON).abs()
                        <= (S::TWO * S::TWO) * S::EPSILON * S::EPSILON
                );
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn sphere_weight_underflow_is_not_an_exact_degenerate_domain() {
    fn check<S: scioxide_core::RealScalar>() {
        let rule = LebedevRule::<S>::new(3, GeometricOptions::default()).unwrap();
        let mut workspace = rule.workspace(1).unwrap();
        let calls = AtomicUsize::new(0);
        let mut value = S::try_from_usize_exact(7).unwrap();
        let error = rule
            .integrate_into(
                |_: &[S]| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(S::MAX)
                },
                &[S::ZERO; 3],
                S::ZERO.next_after(S::ONE).sqrt() * S::from_f64_ieee(0.125),
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(value, S::try_from_usize_exact(7).unwrap());
        assert!(workspace.mapped_weights().is_none());
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn simplex_weight_underflow_preserves_the_output() {
    fn check<S: scioxide_core::RealScalar>() {
        let rule = SimplexRule::<S>::positive(1, 2, GeometricOptions::default()).unwrap();
        let mut workspace = rule.workspace(1).unwrap();
        let vertices: [&[S]; 2] = [&[S::ZERO], &[S::ZERO.next_after(S::ONE)]];
        let calls = AtomicUsize::new(0);
        let mut value = S::try_from_usize_exact(7).unwrap();
        let error = rule
            .integrate_into(
                |_: &[S]| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(S::MAX)
                },
                &vertices,
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(value, S::try_from_usize_exact(7).unwrap());
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn degenerate_simplex_still_validates_supplied_values() {
    let rule = SimplexRule::<f64>::positive(1, 1, GeometricOptions::default()).unwrap();
    let mut workspace = rule.workspace(1).unwrap();
    let vertices: [&[f64]; 2] = [&[0.0], &[0.0]];
    for sample in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut value = 7.0;
        let error = rule
            .integrate_values_into(
                &vertices,
                &[sample],
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        assert_eq!(value, 7.0);
    }
}

#[test]
fn invalid_mesh_cells_replace_the_previous_attempt_record() {
    let rule = SimplexRule::<f64>::positive(2, 1, GeometricOptions::default()).unwrap();
    let mut workspace = rule.workspace(1).unwrap();
    let vertices: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
    let mut value = 0.0;
    let report = rule
        .integrate_into(
            |_: &[f64]| Ok(1.0),
            &vertices,
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    let before = value;
    let mesh = scioxide_spatial::TriangleMesh::<f64>::new(vec![], vec![]).unwrap();
    assert!(
        rule.integrate_mesh_cell_into(
            |_: &[f64]| Ok(1.0),
            &mesh,
            0,
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(value, before);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(
        attempt.outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    assert!(attempt.epoch().is_none());
    assert!(attempt.progress().is_none());
}

#[test]
fn failed_mapping_shape_clears_mapped_metadata() {
    let rule = SimplexRule::<f64>::positive(1, 1, GeometricOptions::default()).unwrap();
    let mut workspace = rule.workspace(1).unwrap();
    let vertices: [&[f64]; 2] = [&[0.0], &[1.0]];
    let (mut points, mut weights) = ([0.0], [0.0]);
    let measure = rule
        .map_nodes_into(&vertices, &mut points, &mut weights, &mut workspace)
        .unwrap();
    assert!((measure.measure - 1.0).abs() < 1e-14);
    assert!(workspace.mapped_points().is_some());
    assert!(
        rule.map_nodes_into(&vertices, &mut [], &mut weights, &mut workspace)
            .is_err()
    );
    assert!(workspace.mapped_points().is_none());
    assert!(workspace.mapped_weights().is_none());
}
