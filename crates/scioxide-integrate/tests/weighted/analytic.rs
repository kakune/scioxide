//! Analytic oracles and public prepared weighted quadrature contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_integrate::*;
use std::sync::atomic::AtomicUsize as Cell;

macro_rules! precision {
    ($module:ident,$s:ty,$tolerance:expr) => {
        mod $module {
            use super::*;
            fn options() -> SubdivisionOptions<$s> {
                SubdivisionOptions::default()
                    .with_tolerance(
                        ErrorTolerance::scalar($tolerance, $tolerance, NormSelection::Infinity)
                            .unwrap(),
                    )
                    .with_max_subdivisions(Budget::try_from(256).unwrap())
            }
            fn check(result: ($s, scioxide_integrate::IntegrationReport<$s>), expected: $s) {
                let value = result.0;
                assert!(
                    (value - expected).abs() < $tolerance * expected.abs().max(1.0),
                    "value {value}, expected {expected}, status {:?}, errors {:?}",
                    result.1.termination(),
                    result.1.error_components()
                );
                assert!(result.1.is_success(), "{:?}", result);
            }
            #[test]
            fn endpoint_weights() {
                let interval = Interval::new(0.0, 1.0).unwrap();
                for alpha in [-0.99, -0.5, 0.0, 0.5, 3.0] {
                    let weight = EndpointWeight::new(alpha, 0.0, false, false).unwrap();
                    let plan = WeightedQuadraturePlan::endpoint(
                        interval,
                        &[],
                        weight,
                        scioxide_integrate::WeightedQuadratureOptions::new(options()),
                    )
                    .unwrap();
                    check(
                        plan.integrate(|_: $s| Ok(1.0)).unwrap(),
                        1.0 / (alpha + 1.0),
                    );
                }
                for (l, r, expected) in [
                    (true, false, -1.0),
                    (false, true, -1.0),
                    (true, true, 2.0 - <$s>::PI * <$s>::PI / 6.0),
                ] {
                    let plan = WeightedQuadraturePlan::endpoint(
                        interval,
                        &[],
                        EndpointWeight::new(0.0, 0.0, l, r).unwrap(),
                        scioxide_integrate::WeightedQuadratureOptions::new(options()),
                    )
                    .unwrap();
                    check(plan.integrate(|_: $s| Ok(1.0)).unwrap(), expected);
                }
                let plan = WeightedQuadraturePlan::endpoint(
                    Interval::new(2.0, 5.0).unwrap(),
                    &[],
                    EndpointWeight::new(-0.5, 0.0, true, false).unwrap(),
                    scioxide_integrate::WeightedQuadratureOptions::new(options()),
                )
                .unwrap();
                check(
                    plan.integrate(|_: $s| Ok(1.0)).unwrap(),
                    (3.0 as $s).sqrt() * (2.0 * (3.0 as $s).ln() - 4.0),
                );
            }
            #[test]
            fn breakpoints_and_principal_values() {
                let interval = Interval::new(0.0, 1.0).unwrap();
                let plan = WeightedQuadraturePlan::breakpoints(
                    interval,
                    &[0.3, 0.3, 1.0, 0.0],
                    scioxide_integrate::WeightedQuadratureOptions::new(options()),
                )
                .unwrap();
                check(
                    plan.integrate(|x: $s| Ok(if x < 0.3 { 2.0 } else { -1.0 }))
                        .unwrap(),
                    -0.1,
                );
                let plan = WeightedQuadraturePlan::breakpoints(
                    interval,
                    &[0.5],
                    scioxide_integrate::WeightedQuadratureOptions::new(options()),
                )
                .unwrap();
                check(
                    plan.integrate(|x: $s| {
                        assert_ne!(x, 0.5);
                        Ok((x - 0.5).abs().ln())
                    })
                    .unwrap(),
                    -(2.0 as $s).ln() - 1.0,
                );
                for pole in [0.1, 0.5, 0.9, 2.0] {
                    let plan = WeightedQuadraturePlan::principal_value(
                        interval,
                        pole,
                        scioxide_integrate::WeightedQuadratureOptions::new(options()),
                    )
                    .unwrap();
                    let logarithm = ((1.0 - pole) / pole).abs().ln();
                    check(plan.integrate(|_: $s| Ok(1.0)).unwrap(), logarithm);
                    check(
                        plan.integrate(|x: $s| Ok(x)).unwrap(),
                        1.0 + pole * logarithm,
                    );
                }
            }
            #[test]
            fn finite_fourier() {
                let interval = Interval::new(0.0, 1.0).unwrap();
                for omega in [0.0, 0.1, 3.0, 20.0, 50.0, 1000.0, -20.0] {
                    for weight in [FourierWeight::Cosine, FourierWeight::Sine] {
                        let plan = WeightedQuadraturePlan::fourier(
                            interval,
                            &[],
                            omega,
                            weight,
                            scioxide_integrate::WeightedQuadratureOptions::new(options())
                                .with_moment_levels(Budget::try_from(16).unwrap()),
                        )
                        .unwrap();
                        let expected = match weight {
                            FourierWeight::Cosine => {
                                ((1.0 as $s).exp() * (omega.cos() + omega * omega.sin()) - 1.0)
                                    / (1.0 + omega * omega)
                            }
                            FourierWeight::Sine => {
                                ((1.0 as $s).exp() * (omega.sin() - omega * omega.cos()) + omega)
                                    / (1.0 + omega * omega)
                            }
                        };
                        check(plan.integrate(|x: $s| Ok(x.exp())).unwrap(), expected);
                    }
                }
            }
            #[test]
            fn infinite_fourier() {
                for weight in [FourierWeight::Cosine, FourierWeight::Sine] {
                    let plan = FourierTailPlan::new(
                        FourierDomain::Upper { lower: 0.0 },
                        2.0,
                        weight,
                        scioxide_integrate::FourierTailOptions::new(options())
                            .with_max_cycles(Budget::try_from(64).unwrap())
                            .with_moment_levels(Budget::try_from(16).unwrap()),
                    )
                    .unwrap();
                    let result = plan.integrate(|x: $s| Ok((-x).exp())).unwrap();
                    let expected = if weight == FourierWeight::Cosine {
                        0.2
                    } else {
                        0.4
                    };
                    assert!((result.0 - expected).abs() < $tolerance, "{result:?}");
                    assert_eq!(
                        result.1.termination(),
                        IntegrationTermination::HeuristicToleranceReached,
                        "{result:?}"
                    );
                    assert!(result.1.is_success());
                    assert!(result.1.error_components().unwrap().tail().is_some());
                }
                let plan = FourierTailPlan::new(
                    FourierDomain::Upper { lower: 0.0 },
                    1.0,
                    FourierWeight::Sine,
                    scioxide_integrate::FourierTailOptions::new(options())
                        .with_max_cycles(Budget::try_from(128).unwrap())
                        .with_moment_levels(Budget::try_from(16).unwrap()),
                )
                .unwrap();
                let result = plan.integrate(|x: $s| Ok(1.0 / (1.0 + x))).unwrap();
                // generate_weighted.py: sin(1)Ci(1)+cos(1)(π/2-Si(1)); weighted.csv tail.
                assert!(
                    (result.0 - (0.621_449_624_235_813_4_f64 as $s)).abs() < $tolerance,
                    "{result:?}"
                );
                assert_eq!(
                    result.1.termination(),
                    IntegrationTermination::HeuristicToleranceReached,
                    "{result:?}"
                );
            }
        }
    };
}
precision!(single, f32, 3e-4);
precision!(double, f64, 2e-9);

#[test]
fn affine_executor_and_transactions() {
    let plan = WeightedQuadraturePlan::fourier(
        Interval::new(0.0, 1.0).unwrap(),
        &[],
        20.0,
        FourierWeight::Cosine,
        scioxide_integrate::WeightedQuadratureOptions::new(SubdivisionOptions::default())
            .with_moment_levels(Budget::try_from(16).unwrap()),
    )
    .unwrap();
    let mut ws = plan.workspace(2).unwrap();
    let mut output = Complex::new(0.0, 0.0);
    let f = |x: f64| Ok(Complex::new(x.exp(), 2.0 * x.exp()));
    let first = plan
        .integrate_into(
            &f,
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(first.is_success());
    assert_eq!(output.im, 2.0 * output.re);
    let expected = output;
    let error = ws.component_errors().unwrap()[0].total();
    let second = plan
        .integrate_into(
            &f,
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(output, expected);
    assert_eq!(first, second);
    assert_eq!(ws.component_errors().unwrap()[0].total(), error);
    assert!(
        plan.integrate_into(
            |_: f64| -> Result<Complex<f64>, SciOxideError> {
                Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("function"),
                    InvalidInputReason::OutOfRange,
                    "callback failure",
                ))
            },
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(output, expected);
    assert!(ws.component_errors().is_none());
    assert!(ws.last_attempt().unwrap().progress().is_none());
}
#[test]
fn initial_budget_and_invalid_inputs() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let calls = Cell::new(0);
    let options =
        SubdivisionOptions::default().with_max_evaluations(Some(Budget::try_from(29).unwrap()));
    let plan = WeightedQuadraturePlan::breakpoints(
        interval,
        &[0.5],
        scioxide_integrate::WeightedQuadratureOptions::new(options.clone()),
    )
    .unwrap();
    let mut ws = plan.workspace(1).unwrap();
    let mut out = 99.0;
    let error = plan
        .integrate_into(
            |x: f64| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(x)
            },
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
    assert_eq!(out, 99.0);
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
    assert!(ws.last_attempt().unwrap().progress().is_none());
    assert!(EndpointWeight::new(-1.0, 0.0, false, false).is_err());
    assert!(
        WeightedQuadraturePlan::principal_value(
            interval,
            0.0,
            scioxide_integrate::WeightedQuadratureOptions::new(options.clone())
        )
        .is_err()
    );
    assert!(
        WeightedQuadraturePlan::breakpoints(
            interval,
            &[f64::NAN],
            scioxide_integrate::WeightedQuadratureOptions::new(options)
        )
        .is_err()
    );
}
#[test]
fn contour_orientation_and_jacobian() {
    let plan = ContourPlan::new(
        Interval::new(0.0, 2.0 * core::f64::consts::PI).unwrap(),
        &[],
        SubdivisionOptions::default(),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut out = Complex::new(0.0, 0.0);
    for sign in [1.0, -1.0] {
        let report = plan
            .integrate_into(
                |z: Complex<f64>| Ok(z.conj()),
                |t: f64| {
                    let z = Complex::new(t.cos(), sign * t.sin());
                    Ok((z, Complex::new(-t.sin(), sign * t.cos())))
                },
                &mut out,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!(out.re.abs() < 1e-10);
        assert!((out.im - sign * 2.0 * core::f64::consts::PI).abs() < 1e-10);
        assert_eq!(ws.component_errors().unwrap().len(), 2);
    }
}
