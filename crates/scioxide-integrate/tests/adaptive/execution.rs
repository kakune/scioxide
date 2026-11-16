//! Shared 1D execution and ordinary regional kernel regressions.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Budget, Complex, ErrorTolerance, ExecutionContext, Interval, NormSelection, RealScalar,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, ContourPlan, DoubleExponentialDomain,
    DoubleExponentialIntegrator, IntegrationTermination, LevelOptions, SubdivisionOptions,
    WeightedQuadratureOptions, WeightedQuadraturePlan,
};

fn breakpoint_parity<S: RealScalar + Default>() {
    let interval = Interval::new(S::ZERO, S::ONE).unwrap();
    let options = SubdivisionOptions::default()
        .with_tolerance(
            ErrorTolerance::scalar(
                S::try_from_f64_lossy(1e-6).unwrap(),
                S::try_from_f64_lossy(1e-6).unwrap(),
                NormSelection::Infinity,
            )
            .unwrap(),
        )
        .with_max_subdivisions(Budget::try_from(256).unwrap());
    for extrapolation in [false, true] {
        let adaptive = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone())
            .unwrap()
            .with_extrapolation(extrapolation)
            .unwrap();
        let breakpoint = WeightedQuadraturePlan::breakpoints(
            interval,
            &[],
            WeightedQuadratureOptions::new(options.clone()),
        )
        .unwrap()
        .with_extrapolation(extrapolation)
        .unwrap();
        let mut aw = adaptive.workspace(2).unwrap();
        let mut bw = breakpoint.workspace(2).unwrap();
        // Smooth, singular, cancelling and oscillatory integrands exercise both
        // the initial-region shortcut and repeated complete split transactions.
        for kind in 0..4 {
            let function = |x: S| {
                let value = match kind {
                    0 => x.exp(),
                    1 => S::ONE / x.sqrt(),
                    2 => x - S::HALF + S::EPSILON,
                    _ => (S::try_from_f64_lossy(100.0).unwrap() * x).cos(),
                };
                Ok([value, -S::TWO * value])
            };
            let mut a = [S::ZERO; 2];
            let mut b = [S::ZERO; 2];
            let ar = adaptive
                .integrate_into(
                    &function,
                    interval,
                    &mut a,
                    &mut aw,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            let br = breakpoint
                .integrate_into(
                    &function,
                    &mut b,
                    &mut bw,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(
                a.map(|x| x.into_f64().to_bits()),
                b.map(|x| x.into_f64().to_bits())
            );
            assert_eq!(ar, br);
            assert_eq!(aw.component_errors(), bw.component_errors());
        }
    }
}
#[test]
fn empty_breakpoint_plan_is_bit_identical_to_adaptive_kronrod() {
    breakpoint_parity::<f32>();
    breakpoint_parity::<f64>();
}

#[test]
fn contour_reports_only_evidence_it_computes() {
    let plan = ContourPlan::new(
        Interval::new(0.0, 1.0).unwrap(),
        &[],
        SubdivisionOptions::default(),
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut value = Complex::new(0.0, 0.0);
    let report = plan
        .integrate_into(
            |z: Complex<f64>| Ok(z * z),
            |t: f64| Ok((Complex::new(t, 0.0), Complex::new(1.0, 0.0))),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((value.re - 1.0 / 3.0).abs() < 1e-14);
    assert_eq!(report.error_components().unwrap().moments(), None);
    assert!(
        workspace
            .component_errors()
            .unwrap()
            .iter()
            .all(|error| error.moments().is_none())
    );
}

#[test]
fn rule_selection_validates_only_supported_features() {
    let simpson =
        AdaptiveIntegrator::<f64>::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
            .unwrap();
    assert_eq!(simpson.rule(), AdaptiveRule::Simpson);
    assert!(simpson.clone().with_extrapolation(true).is_err());
    assert!(
        !simpson
            .with_extrapolation(false)
            .unwrap()
            .extrapolation_enabled()
    );
}

#[test]
fn de_owns_normalized_domain_and_clears_evidence_after_rejection() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let plan = {
        let cuts = vec![1.0, 0.5, 0.5, 0.0];
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::Finite {
                interval,
                singular_points: &cuts,
            },
            LevelOptions::default(),
        )
        .unwrap()
    };
    assert_eq!(
        plan.domain(),
        DoubleExponentialDomain::Finite {
            interval,
            singular_points: &[0.5]
        }
    );
    let mut workspace = plan.workspace(1).unwrap();
    let mut value = 7.0;
    let report = plan
        .integrate_into(
            |p: scioxide_integrate::DoubleExponentialPoint<f64>| Ok(p.coordinate()),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((value - 0.5).abs() < 1e-10);
    assert!(workspace.component_errors().is_some());
    let mut wrong_shape = [7.0; 2];
    assert!(
        plan.integrate_into(
            |_: scioxide_integrate::DoubleExponentialPoint<f64>| Ok(1.0),
            &mut wrong_shape,
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(wrong_shape, [7.0; 2]);
    assert!(workspace.component_errors().is_none());
    let zero = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: Interval::new(0.5, 0.5).unwrap(),
            singular_points: &[],
        },
        LevelOptions::default(),
    )
    .unwrap();
    let (_, report) = zero
        .integrate(|_: scioxide_integrate::DoubleExponentialPoint<f64>| Ok(1.0))
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert_eq!(report.evaluations(), 0);
}
