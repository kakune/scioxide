//! Independent high-precision fixtures from independent high-precision calculations.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, ErrorTolerance, Interval, NormSelection, RealScalar, SciOxideError};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, CubaturePlan, DoubleExponentialDomain,
    DoubleExponentialIntegrator, FixedRule, GaussKronrod, GaussKronrodRule, IntegrationReport,
    LevelOptions, Romberg, SparseGridOptions, SparseGridPlan, SubdivisionOptions,
};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn tolerance<S: RealScalar>(t: f64) -> ErrorTolerance<S> {
    ErrorTolerance::scalar(s(t), s(t), NormSelection::Infinity).unwrap()
}
fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}

fn integrand<S: RealScalar>(name: &str, x: S, p: S) -> S {
    match name {
        "constant" => S::ONE,
        "polynomial" => x.powi(8),
        "exponential" | "tiny_range" => x.exp(),
        "gaussian" => (-x * x).exp(),
        "lorentzian" | "offcenter_peak" => {
            let width: S = if name == "lorentzian" {
                s(10.0)
            } else {
                s(100.0)
            };
            S::ONE / (S::ONE + (width * (x - p)).powi(2))
        }
        "oscillatory" => (p * x).cos(),
        "damped_oscillatory" => (-x).exp() * (p * x).cos(),
        "periodic" => s::<S>(2.0) / (s::<S>(2.0) + (p * x).sin()),
        "aliased_product" => s::<S>(4.0) * p * p * x * (s::<S>(10.0) * p * x).sin() * (p * x).cos(),
        "kink" => (x - p).abs(),
        "step" => {
            if x < p {
                S::ZERO
            } else {
                S::ONE
            }
        }
        "sqrt" => x.sqrt(),
        "sqrt_log" => {
            if x == S::ZERO {
                S::ZERO
            } else {
                x.sqrt() * x.ln()
            }
        }
        "inverse_sqrt" => S::ONE / x.sqrt(),
        "log" => x.ln(),
        "cancellation" => x - s(0.5) + p,
        "exponential_cancellation" => x.exp() - p,
        "large_range" => (-x / s(1000.0)).exp(),
        "boundary_layer" => (-s::<S>(100.0) * x).exp(),
        "long_interval" => S::ONE / (S::ONE + x).powi(2),
        "hidden_spike" => (-(s::<S>(1e6) * (x - p)).powi(2)).exp(),
        _ => unreachable!("fixture case {name}"),
    }
}

/// Compare in f64 even for f32 results, preserving the oracle's extra digits.
fn check<S: RealScalar>(
    value: S,
    report: &IntegrationReport<S>,
    expected: f64,
    scale: f64,
    requested: Option<f64>,
    label: &str,
) -> Option<String> {
    let actual = (value.into_f64() - expected).abs();
    // Input/callback rounding is proportional to sampled magnitude, not to a
    // nearly cancelled result. This allowance never depends on observed error.
    let rounding = 8.0 * S::EPSILON.into_f64() * scale;
    if !value.is_finite() {
        return Some(format!("{label}: nonfinite value"));
    }
    if let Some(t) = requested
        && report.is_success()
        && actual > t * (1.0 + expected.abs()) + rounding
    {
        return Some(format!(
            "{label}: false success: error={actual:e}, {report:?}"
        ));
    }
    if let Some(estimate) = report.error_estimate() {
        if !estimate.is_finite() || estimate < S::ZERO || actual > estimate.into_f64() + rounding {
            return Some(format!(
                "{label}: error={actual:e}, estimate={estimate:?}, {report:?}"
            ));
        }
    } else if report.is_success() {
        return Some(format!("{label}: missing estimate: {report:?}"));
    }
    None
}

fn one_dimensional<S: RealScalar + Default>(scalar: &str, tolerances: &[f64]) {
    let mut failures = Vec::new();
    let mut successes = [0; 4];
    for row in include_str!("../data/battery_1d.csv")
        .lines()
        .filter(|r| r.starts_with(scalar))
    {
        let fields: Vec<_> = row.split(',').collect();
        let name = fields[1];
        let number = |i: usize| fields[i].parse::<f64>().unwrap();
        let interval = Interval::new(s::<S>(number(2)), s::<S>(number(3))).unwrap();
        let p = s::<S>(number(4));
        let expected = number(5);
        let scale = number(6);
        for &t in tolerances {
            for (method, success) in successes.iter_mut().enumerate() {
                // Simpson and Romberg sample the endpoints. Their nonfinite
                // callback contract is tested separately, not replaced by a cutoff.
                if matches!(name, "inverse_sqrt" | "log") && matches!(method, 1 | 2) {
                    continue;
                }
                let calls = AtomicUsize::new(0);
                let f = |x| {
                    calls.fetch_add(1, Relaxed);
                    Ok::<_, SciOxideError>(integrand(name, x, p))
                };
                let options = SubdivisionOptions::default()
                    .with_tolerance(tolerance(t))
                    .with_max_subdivisions(budget(2048));
                let result =
                    match method {
                        0 => AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options)
                            .unwrap()
                            .integrate(&f, interval),
                        1 => AdaptiveIntegrator::new(AdaptiveRule::Simpson, options)
                            .unwrap()
                            .integrate(&f, interval),
                        2 => Romberg::new(
                            LevelOptions::default()
                                .with_tolerance(tolerance(t))
                                .with_max_levels(budget(16)),
                        )
                        .unwrap()
                        .integrate(&f, interval),
                        _ => DoubleExponentialIntegrator::new(
                            DoubleExponentialDomain::Finite {
                                interval,
                                singular_points: &[],
                            },
                            LevelOptions::default().with_tolerance(tolerance(t)),
                        )
                        .unwrap()
                        .integrate(
                            |point: scioxide_integrate::DoubleExponentialPoint<_>| {
                                f(point.coordinate())
                            },
                        ),
                    };
                let (value, report) = result.unwrap();
                assert_eq!(report.evaluations(), calls.load(Relaxed));
                // No finite global sampling rule can discover arbitrary hidden
                // support. This single explicit case demonstrates that limitation.
                if name == "hidden_spike" {
                    assert!(value.is_finite());
                    if let Some(error) = report.error_estimate() {
                        assert!(error.is_finite() && error >= S::ZERO);
                    }
                    continue;
                }
                if matches!(
                    name,
                    "constant"
                        | "polynomial"
                        | "exponential"
                        | "gaussian"
                        | "lorentzian"
                        | "large_range"
                        | "tiny_range"
                ) {
                    assert!(
                        report.is_success(),
                        "{scalar}/{name}/method={method}/tol={t}: {report:?}"
                    );
                }
                *success += usize::from(report.is_success());
                let label = format!("{scalar}/{name}/method={method}/tol={t}");
                if let Some(failure) = check(value, &report, expected, scale, Some(t), &label) {
                    failures.push(failure);
                }
            }
        }
        for pair in [
            GaussKronrodRule::G7K15,
            GaussKronrodRule::G10K21,
            GaussKronrodRule::G15K31,
            GaussKronrodRule::G20K41,
            GaussKronrodRule::G25K51,
        ] {
            let calls = AtomicUsize::new(0);
            let (value, report) = GaussKronrod::new(pair)
                .unwrap()
                .integrate(
                    |x| {
                        calls.fetch_add(1, Relaxed);
                        Ok::<_, SciOxideError>(integrand(name, x, p))
                    },
                    interval,
                )
                .unwrap();
            assert_eq!(report.evaluations(), calls.load(Relaxed));
            if name != "hidden_spike"
                && let Some(failure) = check(
                    value,
                    &report,
                    expected,
                    scale,
                    None,
                    &format!("{scalar}/{name}/{pair:?}"),
                )
            {
                failures.push(failure);
            }
        }
    }
    assert!(successes.iter().all(|&n| n >= tolerances.len() * 8));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn genz_value<S: RealScalar>(family: &str, x: &[S], difficulty: S) -> S {
    let coefficients = x.iter().enumerate().map(|(j, &x)| {
        (
            x,
            difficulty * s::<S>(f64::from(u32::try_from(j + 1).unwrap())),
            s::<S>(0.31 + 0.16 * f64::from(u32::try_from(j).unwrap())),
        )
    });
    match family {
        "oscillatory" => (s::<S>(0.4 * core::f64::consts::PI)
            + coefficients
                .map(|(x, c, _)| c * x)
                .fold(S::ZERO, |a, b| a + b))
        .cos(),
        "product_peak" => coefficients
            .map(|(x, c, w)| S::ONE / (S::ONE + (c * (x - w)).powi(2)))
            .fold(S::ONE, |a, b| a * b),
        "corner_peak" => (S::ONE
            + coefficients
                .map(|(x, c, _)| c * x)
                .fold(S::ZERO, |a, b| a + b))
        .powi(-i32::try_from(x.len() + 1).unwrap()),
        "gaussian" => (-coefficients
            .map(|(x, c, w)| (c * (x - w)).powi(2))
            .fold(S::ZERO, |a, b| a + b))
        .exp(),
        "continuous" => (-coefficients
            .map(|(x, c, w)| c * (x - w).abs())
            .fold(S::ZERO, |a, b| a + b))
        .exp(),
        "discontinuous" => {
            let mut exponent = S::ZERO;
            for (x, c, w) in coefficients {
                if x > w {
                    return S::ZERO;
                }
                exponent += c * x;
            }
            exponent.exp()
        }
        _ => unreachable!("Genz fixture {family}"),
    }
}

fn multidimensional<S: RealScalar + Default>(scalar: &str, tolerances: &[f64]) {
    let mut failures = Vec::new();
    let mut successes = [0; 2];
    for row in include_str!("../data/genz.csv")
        .lines()
        .filter(|r| r.starts_with(scalar))
    {
        let fields: Vec<_> = row.split(',').collect();
        let family = fields[1];
        let dimension: usize = fields[2].parse().unwrap();
        let difficulty: f64 = fields[3].parse().unwrap();
        let expected: f64 = fields[4].parse().unwrap();
        let scale: f64 = fields[5].parse().unwrap();
        let bounds = vec![Interval::new(S::ZERO, S::ONE).unwrap(); dimension];
        for &t in tolerances {
            for (method, success) in successes.iter_mut().enumerate() {
                let calls = AtomicUsize::new(0);
                let f = |x: &[S]| {
                    calls.fetch_add(1, Relaxed);
                    Ok::<_, SciOxideError>(genz_value(family, x, s(difficulty)))
                };
                let (value, report) = if method == 0 {
                    CubaturePlan::new(
                        &bounds,
                        SubdivisionOptions::default()
                            .with_tolerance(tolerance(t))
                            .with_max_subdivisions(budget(512)),
                    )
                    .unwrap()
                    .integrate(&f)
                    .unwrap()
                } else {
                    SparseGridPlan::new(
                        dimension,
                        SparseGridOptions::default()
                            .with_tolerance(tolerance(t))
                            .with_max_level(8)
                            .with_max_indices(budget(256))
                            .with_max_evaluations(budget(32768)),
                    )
                    .unwrap()
                    .integrate(&f, &bounds)
                    .unwrap()
                };
                assert_eq!(report.evaluations(), calls.load(Relaxed));
                *success += usize::from(report.is_success());
                let label = format!(
                    "{scalar}/{family}/d={dimension}/c={difficulty}/method={method}/tol={t}"
                );
                if let Some(failure) = check(value, &report, expected, scale, Some(t), &label) {
                    failures.push(failure);
                }
            }
        }
    }
    assert!(successes.iter().all(|&n| n >= 4));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn one_dimensional_f64() {
    one_dimensional::<f64>("f64", &[1e-6, 1e-10]);
}
#[test]
fn one_dimensional_f32() {
    one_dimensional::<f32>("f32", &[1e-3, 1e-4]);
}
#[test]
fn genz_f64() {
    multidimensional::<f64>("f64", &[1e-6, 1e-10]);
}
#[test]
fn genz_f32() {
    multidimensional::<f32>("f32", &[1e-3, 1e-4]);
}

#[test]
fn sparse_grid_cannot_discard_a_mixed_contribution_behind_zero_axis_surpluses() {
    // Every pure-axis sample is zero, while mixed descendants are nonzero.
    // ∫(x-.5)²dx=1/12 and ∫(x-.5)²x⁴dx=11/420, giving 19/1680.
    let bounds = [Interval::new(0.0_f64, 1.0).unwrap(); 2];
    let calls = AtomicUsize::new(0);
    let (value, report) = SparseGridPlan::new(
        2,
        SparseGridOptions::default().with_tolerance(tolerance(1e-11)),
    )
    .unwrap()
    .integrate(
        |x: &[f64]| {
            calls.fetch_add(1, Relaxed);
            Ok((x[0] - 0.5).powi(2) * (x[1] - 0.5).powi(2) * (1.0 + x[0].powi(4) + x[1].powi(4)))
        },
        &bounds,
    )
    .unwrap();
    assert!(report.is_success(), "{report:?}");
    assert_eq!(report.evaluations(), calls.load(Relaxed));
    assert!((value - 19.0 / 1680.0).abs() <= 1e-11);

    // Degree-four levels cannot resolve the degree-six factors. Exhausting
    // them must retain an unsuccessful status rather than erase their tail.
    let (value, report) = SparseGridPlan::new(
        2,
        SparseGridOptions::default()
            .with_max_level(2)
            .with_tolerance(tolerance(1e-11)),
    )
    .unwrap()
    .integrate(
        |x: &[f64]| {
            Ok((x[0] - 0.5).powi(2) * (x[1] - 0.5).powi(2) * (1.0 + x[0].powi(4) + x[1].powi(4)))
        },
        &bounds,
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        scioxide_integrate::IntegrationTermination::RefinementLimit
    );
    assert!((value - 19.0 / 1680.0).abs() > 1e-11);
}

fn nonfinite_callbacks<S: RealScalar + Default>() {
    use scioxide_core::{ExecutionContext, NonFiniteOrigin, SciOxideErrorView};
    use scioxide_integrate::{
        EndpointWeight, FourierDomain, FourierTailOptions, FourierTailPlan, FourierWeight,
        WeightedQuadratureOptions, WeightedQuadraturePlan,
    };
    let unit = Interval::new(S::ZERO, S::ONE).unwrap();
    let callback_error = |error: SciOxideError| {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Callback,
                    ..
                }
            ),
            "{error:?}"
        )
    };
    for nonfinite in [S::NAN, S::INFINITY, -S::INFINITY] {
        let f = |_: S| Ok::<_, SciOxideError>(nonfinite);
        for domain in [
            DoubleExponentialDomain::Finite {
                interval: unit,
                singular_points: &[],
            },
            DoubleExponentialDomain::HalfLine { lower: S::ZERO },
            DoubleExponentialDomain::RealLine,
        ] {
            let de = DoubleExponentialIntegrator::new(domain, LevelOptions::default()).unwrap();
            let mut ws = de.workspace(1).unwrap();
            let mut out = s::<S>(7.0);
            callback_error(
                de.integrate_into(
                    |point: scioxide_integrate::DoubleExponentialPoint<_>| f(point.coordinate()),
                    &mut out,
                    &mut ws,
                    ExecutionContext::sequential_native(),
                )
                .unwrap_err(),
            );
            assert_eq!(out, s(7.0));
        }
        let weighted = [
            WeightedQuadraturePlan::breakpoints(
                unit,
                &[s(0.3)],
                WeightedQuadratureOptions::default(),
            )
            .unwrap(),
            WeightedQuadraturePlan::endpoint(
                unit,
                &[],
                EndpointWeight::new(s(-0.5), S::ZERO, false, false).unwrap(),
                WeightedQuadratureOptions::default(),
            )
            .unwrap(),
            WeightedQuadraturePlan::principal_value(
                unit,
                s(0.3),
                WeightedQuadratureOptions::default(),
            )
            .unwrap(),
            WeightedQuadraturePlan::fourier(
                unit,
                &[],
                s(20.0),
                FourierWeight::Cosine,
                WeightedQuadratureOptions::default(),
            )
            .unwrap(),
        ];
        for plan in weighted {
            let mut ws = plan.workspace(1).unwrap();
            let mut out = s::<S>(7.0);
            callback_error(
                plan.integrate_into(&f, &mut out, &mut ws, ExecutionContext::sequential_native())
                    .unwrap_err(),
            );
            assert_eq!(out, s(7.0));
        }
        let tail = FourierTailPlan::new(
            FourierDomain::Upper { lower: S::ZERO },
            S::ONE,
            FourierWeight::Sine,
            FourierTailOptions::default(),
        )
        .unwrap();
        let mut ws = tail.workspace(1).unwrap();
        let mut out = s::<S>(7.0);
        callback_error(
            tail.integrate_into(&f, &mut out, &mut ws, ExecutionContext::sequential_native())
                .unwrap_err(),
        );
        assert_eq!(out, s(7.0));
    }
    // Closed endpoint methods reject singular endpoint values atomically.
    for name in ["inverse_sqrt", "log"] {
        let f = |x| Ok::<_, SciOxideError>(integrand(name, x, S::ZERO));
        callback_error(
            AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::<S>::default())
                .unwrap()
                .integrate(&f, unit)
                .unwrap_err(),
        );
        callback_error(
            Romberg::new(LevelOptions::<S>::default())
                .unwrap()
                .integrate(&f, unit)
                .unwrap_err(),
        );
    }
}

#[test]
fn nonfinite_callbacks_f64() {
    nonfinite_callbacks::<f64>();
}
#[test]
fn nonfinite_callbacks_f32() {
    nonfinite_callbacks::<f32>();
}

fn endpoint_battery<S: RealScalar + Default>(scalar: &str, tolerances: &[f64]) {
    for row in include_str!("../data/endpoint_singularities.csv")
        .lines()
        .filter(|row| row.starts_with(scalar))
    {
        let fields: Vec<_> = row.split(',').collect();
        let number = |index: usize| fields[index].parse::<f64>().unwrap();
        let interval = Interval::new(s::<S>(number(1)), s::<S>(number(2))).unwrap();
        let endpoint = if fields[3] == "1" {
            interval.upper()
        } else {
            interval.lower()
        };
        let exponent = s::<S>(number(4));
        let expected = number(5);
        for &t in tolerances {
            let de = DoubleExponentialIntegrator::new(
                DoubleExponentialDomain::Finite {
                    interval,
                    singular_points: &[],
                },
                LevelOptions::default().with_tolerance(tolerance::<S>(t)),
            )
            .unwrap();
            let (value, report) = de
                .integrate(|point: scioxide_integrate::DoubleExponentialPoint<S>| {
                    let distance = match point.nearest_endpoint() {
                        Some((location, distance)) if location == endpoint => distance,
                        _ => (point.coordinate() - endpoint).abs(),
                    };
                    Ok(distance.powf(-exponent))
                })
                .unwrap();
            assert!(report.error_estimate().is_some(), "{row} {report:?}");
            assert_eq!(
                check(value, &report, expected, expected, Some(t), row),
                None
            );
        }
    }
}

#[test]
fn independent_endpoint_singularities_binary64() {
    endpoint_battery::<f64>("f64", &[1e-6, 1e-10]);
}
#[test]
fn independent_endpoint_singularities_binary32() {
    endpoint_battery::<f32>("f32", &[1e-3, 1e-4]);
}
