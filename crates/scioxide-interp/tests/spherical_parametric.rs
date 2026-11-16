//! Independent analytic fields and resource/ownership contract tests.
#![allow(clippy::unwrap_used)]
mod support;

use scioxide_core::{
    AlgorithmId, Budget, BudgetKind, CancellationToken, Capacity, Complex, ConvergenceReason,
    ExecutionContext, RealScalar, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
    SolveBudgets, Tolerance,
};
use scioxide_interp::{
    AdaptiveSplineOptions, AdaptiveSplineTermination, CurveParameterization, Differentiable,
    ExtrapolationPolicy, Integrable, Interpolant, SphericalSamples, SphericalSpline,
    SphericalSplinePlan, SplineFitLimits, SplineFitTarget, TwiceDifferentiable,
    fit_parametric_spline, fit_parametric_spline_scoped, fit_parametric_spline_with_options,
    fit_spherical_spline, fit_spherical_spline_scoped, fit_spherical_spline_with_options,
};
use support::{assert_close, from_f64};

const REJECT: ExtrapolationPolicy<f64> = ExtrapolationPolicy::Reject;
fn f<R: RealScalar>(x: R) -> f64 {
    x.try_to_f64_exact().unwrap()
}

fn options<R: RealScalar>(target: f64) -> AdaptiveSplineOptions<R> {
    AdaptiveSplineOptions::new()
        .with_target(SplineFitTarget::Residual(from_f64(target)))
        .with_residual_tolerance(Tolerance::new(from_f64(1e-16), from_f64(1e-5)).unwrap())
}
fn fixed<R: RealScalar>() -> AdaptiveSplineOptions<R> {
    AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(R::ZERO))
}
fn limits(f: impl FnOnce(SplineFitLimits) -> SplineFitLimits) -> SplineFitLimits {
    f(SplineFitLimits::new())
}
fn longitude<R: RealScalar>() -> [R; 5] {
    [
        R::ZERO,
        R::PI / from_f64(2.),
        R::PI,
        R::PI * from_f64(1.5),
        R::PI * from_f64(2.),
    ]
}
fn grid<R: RealScalar>(nt: usize, np: usize) -> (Vec<R>, Vec<R>, Vec<[R; 2]>) {
    let theta = (0..=nt)
        .map(|i| R::PI * from_f64::<R>(i as f64 / nt as f64))
        .collect::<Vec<_>>();
    let phi = (0..np)
        .map(|j| R::PI * from_f64::<R>(2. * j as f64 / np as f64))
        .collect::<Vec<_>>();
    let samples = theta
        .iter()
        .flat_map(|&t| phi.iter().map(move |&p| [t, p]))
        .collect();
    (theta, phi, samples)
}
fn exact_field<R: RealScalar>(point: [R; 2]) -> R {
    let u = point[0] / R::PI;
    let h01 = u * u * (from_f64::<R>(3.) - from_f64::<R>(2.) * u);
    let hn = R::PI * u * (R::ONE - u) * (R::ONE - u);
    let hs = R::PI * u * u * (u - R::ONE);
    R::ONE + h01 + hn * point[1].cos() + from_f64::<R>(2.) * hs * point[1].sin()
}
fn exact_plan<R: RealScalar>() -> SphericalSplinePlan<R> {
    let (_, _, samples) = grid::<R>(6, 12);
    SphericalSplinePlan::new(
        &[R::ZERO, R::PI / from_f64(2.), R::PI],
        &longitude::<R>(),
        SphericalSamples::Scattered(&samples),
        None,
        R::ZERO,
        SplineFitLimits::default(),
    )
    .unwrap()
}
/// Evaluate through the blanket implementation for borrowed interpolants.
fn evaluate_borrowed<R: RealScalar, I: Interpolant<R, 2>>(model: I, point: [R; 2]) -> I::Value {
    model.evaluate(point, ExtrapolationPolicy::Reject).unwrap()
}
fn width_tolerance<R: RealScalar>(single: f64, double: f64) -> f64 {
    if core::mem::size_of::<R>() == 4 {
        single
    } else {
        double
    }
}

fn sphere_family<R: RealScalar>() {
    let (_, _, samples) = grid::<R>(6, 12);
    let values = samples.iter().copied().map(exact_field).collect::<Vec<_>>();
    let plan = exact_plan::<R>();
    let fit = plan.fit(&values).unwrap();
    let tolerance = width_tolerance::<R>(4e-5, 2e-12);
    assert!(f(fit.report().residual()) < tolerance);
    let spline = fit.into_spline();
    let reject = ExtrapolationPolicy::Reject;
    let mut w = spline.workspace().unwrap();
    let mut output = R::ZERO;
    for t in [0., 0.13, 0.57, 1.] {
        for p in [-1.2, 0., 0.37, 6.8] {
            let point = [R::PI * from_f64::<R>(t), from_f64(p)];
            spline
                .evaluate_into(point, reject, &mut output, &mut w)
                .unwrap();
            assert_close::<R, R>(output, exact_field(point), tolerance, 0.0);
            // A borrowed model is an interpolant with the same results.
            assert_eq!(evaluate_borrowed(&spline, point), output);
        }
    }
    for order in [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0]] {
        spline
            .derivative_into(
                [R::PI * from_f64(0.43), R::ZERO],
                order,
                reject,
                &mut output,
                &mut w,
            )
            .unwrap();
        let left = output;
        spline
            .derivative_into(
                [R::PI * from_f64(0.43), R::PI * from_f64(2.)],
                order,
                reject,
                &mut output,
                &mut w,
            )
            .unwrap();
        assert_eq!(left, output);
    }
    for pole in [R::ZERO, R::PI] {
        let x = spline.derivative([pole, R::ZERO], [1, 0], reject).unwrap();
        let y = spline
            .derivative([pole, R::PI / from_f64(2.)], [1, 0], reject)
            .unwrap();
        for phi in [0.21, 1.3, 2.7, 5.4] {
            let p = from_f64::<R>(phi);
            let [d_theta, d_phi] = spline.gradient([pole, p], reject).unwrap();
            assert_close::<R, R>(d_theta, x * p.cos() + y * p.sin(), tolerance, 0.0);
            assert_close::<R, R>(d_phi, R::ZERO, tolerance, 0.0);
        }
    }
    // Integral of h01 over sphere is half the sphere area by north/south symmetry;
    // trigonometric terms integrate to zero. Mean field is therefore 3/2.
    spline
        .integral_into(
            [R::ZERO, from_f64(-1.3)],
            [R::PI, from_f64::<R>(-1.3) + R::PI * from_f64(4.)],
            reject,
            &mut output,
            &mut w,
        )
        .unwrap();
    assert_close::<R, R>(output, R::PI * from_f64(12.), tolerance * 20., 0.0);
    // Bounds are ordered per axis; reversed bounds are rejected, output unchanged.
    let whole = output;
    let error = spline
        .integral_into(
            [R::PI, from_f64(-1.3)],
            [R::ZERO, from_f64::<R>(-1.3) + R::PI * from_f64(4.)],
            reject,
            &mut output,
            &mut w,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(output, whole);
    // Derivative and primitive are checked independently by centered differences.
    let p = [from_f64::<R>(1.1), from_f64::<R>(0.7)];
    let h = from_f64::<R>(width_tolerance::<R>(0.002, 1e-5));
    let plus = spline.evaluate([p[0] + h, p[1]], reject).unwrap();
    let minus = spline.evaluate([p[0] - h, p[1]], reject).unwrap();
    let [d_theta, _] = spline.gradient(p, reject).unwrap();
    assert_close::<R, R>(
        d_theta,
        (plus - minus) / (from_f64::<R>(2.) * h),
        width_tolerance::<R>(4e-4, 1e-9),
        0.0,
    );
    let hessian = spline.hessian(p, reject).unwrap();
    assert_eq!(hessian[0][1], hessian[1][0]);
    assert_close::<R, R>(
        hessian[1][1],
        spline.derivative(p, [0, 2], reject).unwrap(),
        0.,
        0.0,
    );
}
#[test]
fn real_sphere_widths() {
    sphere_family::<f32>();
    sphere_family::<f64>();
}

fn curve_family<R: RealScalar>() {
    let parameters = (0..21)
        .map(|i| from_f64::<R>(i as f64 / 20.))
        .collect::<Vec<_>>();
    let values = parameters
        .iter()
        .map(|&t| [t, t * t, t * t * t])
        .collect::<Vec<_>>();
    let tolerance = width_tolerance::<R>(3e-5, 2e-12);
    let reject = ExtrapolationPolicy::Reject;
    let fit = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&parameters),
        None,
        fixed(),
    )
    .unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::FixedPenalty
    );
    let mapped = parameters
        .iter()
        .map(|&t| from_f64::<R>(2.) + from_f64::<R>(7.) * t)
        .collect::<Vec<_>>();
    let other = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&mapped),
        None,
        fixed(),
    )
    .unwrap();
    let mut w = fit.spline().workspace().unwrap();
    let mut ow = other.spline().workspace().unwrap();
    let mut out = [R::ZERO; 3];
    let mut oo = [R::ZERO; 3];
    let t = from_f64::<R>(0.37);
    let u = from_f64::<R>(2.) + from_f64::<R>(7.) * t;
    fit.spline()
        .evaluate_into([t], reject, &mut out, &mut w)
        .unwrap();
    other
        .spline()
        .evaluate_into([u], reject, &mut oo, &mut ow)
        .unwrap();
    for (a, b) in out.iter().zip(oo) {
        assert_close::<R, R>(*a, b, tolerance, 0.0);
    }
    fit.spline()
        .derivative_into([t], [1], None, reject, &mut out, &mut w)
        .unwrap();
    other
        .spline()
        .derivative_into([u], [1], None, reject, &mut oo, &mut ow)
        .unwrap();
    for (a, b) in out.iter().zip(oo) {
        assert_close::<R, R>(*a, b * from_f64(7.), tolerance, 0.0);
    }
    fit.spline()
        .integral_into([R::ZERO], [R::ONE], reject, &mut out, &mut w)
        .unwrap();
    for (a, b) in out.iter().zip([0.5, 1. / 3., 0.25]) {
        assert_close::<R, R>(*a, from_f64(b), tolerance, 0.0);
    }
    let line = parameters
        .iter()
        .map(|&t| [from_f64::<R>(3.) * t, from_f64::<R>(4.) * t])
        .collect::<Vec<_>>();
    let chord = fit_parametric_spline_with_options(
        &line,
        CurveParameterization::ChordLength,
        None,
        fixed(),
    )
    .unwrap();
    for (a, &b) in chord.parameters().iter().zip(&parameters) {
        assert_close::<R, R>(*a, b, tolerance, 0.0);
    }
}
#[test]
fn real_parametric_widths() {
    curve_family::<f32>();
    curve_family::<f64>();
}

fn complex_family<R: RealScalar>() {
    let (_, _, samples) = grid::<R>(5, 8);
    let values = samples
        .iter()
        .map(|&p| Complex::new(exact_field(p), from_f64::<R>(2.) * exact_field(p)))
        .collect::<Vec<_>>();
    let plan = SphericalSplinePlan::new(
        &[R::ZERO, R::PI],
        &longitude::<R>(),
        SphericalSamples::Scattered(&samples),
        None,
        R::ZERO,
        SplineFitLimits::default(),
    )
    .unwrap();
    let fit = plan.fit(&values).unwrap();
    let reject = ExtrapolationPolicy::Reject;
    let point = [from_f64::<R>(0.83), from_f64::<R>(1.7)];
    let out = fit.spline().evaluate(point, reject).unwrap();
    let tolerance = width_tolerance::<R>(4e-5, 2e-12);
    assert_close::<R, R>(out.re, exact_field(point), tolerance, 0.0);
    assert_close::<R, R>(
        out.im,
        from_f64::<R>(2.) * exact_field(point),
        tolerance,
        0.0,
    );
    let parameters = (0..12)
        .map(|i| from_f64::<R>(i as f64 / 11.))
        .collect::<Vec<_>>();
    let values = parameters
        .iter()
        .map(|&t| Complex::new(t, t * t))
        .collect::<Vec<_>>();
    let fit = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&parameters),
        None,
        fixed(),
    )
    .unwrap();
    let out = fit.spline().evaluate([from_f64(0.4)], reject).unwrap();
    assert_close::<R, R>(out.re, from_f64(0.4), tolerance, 0.0);
    assert_close::<R, R>(out.im, from_f64(0.16), tolerance, 0.0);
    let chord = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::ChordLength,
        None,
        fixed(),
    )
    .unwrap();
    assert_eq!(chord.parameters()[0], R::ZERO);
    assert_eq!(*chord.parameters().last().unwrap(), R::ONE);
}
#[test]
fn complex_widths() {
    complex_family::<f32>();
    complex_family::<f64>();
}

fn affine_family<R: RealScalar>() {
    let (_, _, samples) = grid::<R>(6, 12);
    let plan = SphericalSplinePlan::new(
        &[R::ZERO, R::PI / from_f64(2.), R::PI],
        &longitude::<R>(),
        SphericalSamples::Scattered(&samples),
        None,
        from_f64(0.05),
        SplineFitLimits::default(),
    )
    .unwrap();
    let first = samples.iter().copied().map(exact_field).collect::<Vec<R>>();
    let second = samples
        .iter()
        .map(|p| p[0].cos() * p[1].sin())
        .collect::<Vec<R>>();
    let pairs = first
        .iter()
        .zip(&second)
        .map(|(&a, &b)| [a, b])
        .collect::<Vec<_>>();
    let complex = first
        .iter()
        .zip(&second)
        .map(|(&a, &b)| Complex::new(a, b))
        .collect::<Vec<_>>();
    let scalar = [plan.fit(&first).unwrap(), plan.fit(&second).unwrap()];
    let array = plan.fit(&pairs).unwrap();
    let joint = plan.fit(&complex).unwrap();
    // One real design solved per component: bit-equal coefficients.
    assert_eq!(array.spline().coefficients(), joint.spline().coefficients());
    for (mode, pair) in array.spline().coefficients().chunks_exact(2).enumerate() {
        assert_eq!(pair[0], scalar[0].spline().coefficients()[mode]);
        assert_eq!(pair[1], scalar[1].spline().coefficients()[mode]);
    }
    let reject = ExtrapolationPolicy::Reject;
    for point in [
        [from_f64::<R>(0.3), from_f64::<R>(-2.)],
        [R::ZERO, from_f64(1.)],
        [from_f64(2.9), from_f64(4.)],
    ] {
        let value = joint.spline().evaluate(point, reject).unwrap();
        let [a, b] = array.spline().evaluate(point, reject).unwrap();
        assert_eq!([value.re, value.im], [a, b]);
        assert_eq!(a, scalar[0].spline().evaluate(point, reject).unwrap());
        assert_eq!(b, scalar[1].spline().evaluate(point, reject).unwrap());
        let gradient = array.spline().gradient(point, reject).unwrap();
        let hessian = joint.spline().hessian(point, reject).unwrap();
        for (component, model) in scalar.iter().enumerate() {
            let expected = model.spline().gradient(point, reject).unwrap();
            assert_eq!([gradient[0][component], gradient[1][component]], expected);
            let expected = model.spline().hessian(point, reject).unwrap();
            let parts = hessian.map(|row| row.map(|z| [z.re, z.im][component]));
            assert_eq!(parts, expected);
        }
    }
    let lower = [from_f64::<R>(0.2), from_f64::<R>(-1.)];
    let upper = [from_f64::<R>(2.5), from_f64::<R>(3.)];
    let integral = joint.spline().integral(lower, upper, reject).unwrap();
    assert_eq!(
        [integral.re, integral.im],
        [
            scalar[0].spline().integral(lower, upper, reject).unwrap(),
            scalar[1].spline().integral(lower, upper, reject).unwrap()
        ]
    );
    // Curves: penalized fits solve every component against one banded factor.
    let parameters = (0..15)
        .map(|i| from_f64::<R>(i as f64 / 14.))
        .collect::<Vec<_>>();
    let triples = parameters
        .iter()
        .map(|&t| [t, t * t, (from_f64::<R>(3.) * t).sin()])
        .collect::<Vec<_>>();
    let curve = fit_parametric_spline_with_options(
        &triples,
        CurveParameterization::Supplied(&parameters),
        None,
        fixed(),
    )
    .unwrap();
    for component in 0..3 {
        let column = triples.iter().map(|v| v[component]).collect::<Vec<_>>();
        let single = fit_parametric_spline_with_options(
            &column,
            CurveParameterization::Supplied(&parameters),
            None,
            fixed(),
        )
        .unwrap();
        for (mode, &coefficient) in single.spline().coefficients().iter().enumerate() {
            assert_close::<R, R>(
                curve.spline().coefficients()[3 * mode + component],
                coefficient,
                width_tolerance::<R>(1e-6, 1e-14),
                0.0,
            );
        }
    }
}
#[test]
fn affine_values_match_component_fits() {
    affine_family::<f32>();
    affine_family::<f64>();
}

#[test]
fn curve_automatic_knots_and_residual_search() {
    let p = (0..41).map(|i| i as f64 / 40.).collect::<Vec<_>>();
    let values = p.iter().map(|&x| [x, (8. * x).sin()]).collect::<Vec<_>>();
    let fit = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(1e-6),
    )
    .unwrap();
    let report = *fit.report();
    assert_eq!(
        report.termination(),
        AdaptiveSplineTermination::ResidualSatisfied
    );
    assert!(report.termination().is_success());
    assert!(report.refinements() > 0);
    assert!(report.residual() <= 1.00002e-6);
    assert!(report.penalty_iterations() > 0);
    assert_eq!(
        report.knot_counts()[0],
        fit.spline().bases()[0].knots().len()
    );
    let residual = p
        .iter()
        .zip(&values)
        .map(|(&t, v)| {
            let out = fit.spline().evaluate([t], REJECT).unwrap();
            (out[0] - v[0]).powi(2) + (out[1] - v[1]).powi(2)
        })
        .sum::<f64>();
    assert_close::<f64, f64>(residual, report.residual(), 1e-16, 0.0);
    let no_refinement = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(1e-6).with_limits(limits(|l| l.with_max_refinements(0))),
    )
    .unwrap();
    assert_eq!(
        no_refinement.report().termination(),
        AdaptiveSplineTermination::RefinementLimit
    );
    assert!(no_refinement.report().residual() > 1e-6);
    let no_search = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(1e-6).with_limits(limits(|l| l.with_max_penalty_iterations(0))),
    )
    .unwrap();
    assert_eq!(
        no_search.report().termination(),
        AdaptiveSplineTermination::PenaltySearchLimit
    );
    assert!(!no_search.report().termination().is_success());
    assert_eq!(no_search.report().penalty(), 0.);
}

#[test]
fn penalty_search_limit_only_when_the_search_stops_unsatisfied() {
    let p = (0..41).map(|i| i as f64 / 40.).collect::<Vec<_>>();
    let values = p.iter().map(|&x| [x, (8. * x).sin()]).collect::<Vec<_>>();
    let tolerance = Tolerance::new(1e-16, 1e-5)
        .unwrap()
        .threshold(1e-6)
        .unwrap();
    let fit = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(1e-6),
    )
    .unwrap();
    // The unlimited search stopped because its last trial met the target.
    assert!((fit.report().residual() - 1e-6).abs() <= tolerance);
    let needed = fit.report().penalty_iterations();
    assert!(needed > 1);
    let search = |trials: usize| {
        fit_parametric_spline_with_options(
            &values,
            CurveParameterization::Supplied(&p),
            None,
            options(1e-6).with_limits(limits(|l| l.with_max_penalty_iterations(trials))),
        )
        .unwrap()
    };
    // Success on the last allowed trial is not a limited search.
    let exact = search(needed);
    assert_eq!(
        exact.report().termination(),
        AdaptiveSplineTermination::ResidualSatisfied
    );
    assert_eq!(exact.report(), fit.report());
    // One trial fewer exhausts the search without reaching the target.
    let short = search(needed - 1);
    assert_eq!(
        short.report().termination(),
        AdaptiveSplineTermination::PenaltySearchLimit
    );
    assert_eq!(short.report().penalty_iterations(), needed - 1);
    assert!(short.report().residual() <= 1e-6 + tolerance);
    assert!((short.report().residual() - 1e-6).abs() > tolerance);
}

#[test]
fn spherical_harmonic_accuracy_and_adaptation() {
    let (_, _, samples) = grid::<f64>(16, 16);
    let values = samples.iter().map(|p| p[0].cos()).collect::<Vec<_>>();
    let fit = fit_spherical_spline_with_options(
        SphericalSamples::Scattered(&samples),
        &values,
        None,
        options(1e-6).with_limits(limits(|l| l.with_max_penalty_iterations(6))),
    )
    .unwrap();
    // Six penalty trials all exceeded the bound: the retained least-squares fit
    // meets the residual bound, but the smoothing search was cut short.
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::PenaltySearchLimit,
        "{:?}",
        fit.report()
    );
    assert!(fit.report().residual() <= 1e-6);
    assert!(fit.report().refinements() > 0);
    assert_eq!(fit.report().pole_continuity(), Some(1));
    let spline = fit.spline();
    let mut w = spline.workspace().unwrap();
    let mut out = 0.;
    for t in [0.03, 0.18, 0.41, 0.74, 0.97] {
        let point = [t * core::f64::consts::PI, 0.61];
        spline
            .evaluate_into(point, REJECT, &mut out, &mut w)
            .unwrap();
        assert_close::<f64, f64>(out, point[0].cos(), 3e-4, 0.0);
    }
    for row in include_str!("data/spherical_parametric.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cells = row.split('\t').collect::<Vec<_>>();
        let theta = cells[0].parse::<f64>().unwrap();
        let phi = cells[1].parse::<f64>().unwrap();
        spline
            .evaluate_into([theta, phi], REJECT, &mut out, &mut w)
            .unwrap();
        assert_close::<f64, f64>(out, cells[5].parse().unwrap(), 3e-4, 0.0);
        spline
            .derivative_into([theta, phi], [1, 0], REJECT, &mut out, &mut w)
            .unwrap();
        assert_close::<f64, f64>(out, cells[6].parse().unwrap(), 2e-3, 0.0);
    }
    spline
        .integral_into(
            [0., 0.],
            [core::f64::consts::PI, core::f64::consts::TAU],
            REJECT,
            &mut out,
            &mut w,
        )
        .unwrap();
    assert_close::<f64, f64>(out, 0., 2e-5, 0.0);
}

#[test]
fn rectangular_and_scattered_reuse_weights_and_smoothing() {
    let (theta, phi, samples) = grid::<f64>(6, 12);
    let pi = core::f64::consts::PI;
    let tb = [0., pi / 2., pi];
    let pb = longitude::<f64>();
    let values = samples
        .iter()
        .map(|&p| [exact_field(p), 1.])
        .collect::<Vec<_>>();
    let weights = (0..samples.len())
        .map(|i| {
            if i % 13 == 0 {
                0.
            } else {
                1. + i as f64 / 100.
            }
        })
        .collect::<Vec<_>>();
    let grid = SphericalSplinePlan::new(
        &tb,
        &pb,
        SphericalSamples::Rectangular {
            theta: &theta,
            longitude: &phi,
        },
        Some(&weights),
        0.1,
        SplineFitLimits::default(),
    )
    .unwrap();
    let scattered = SphericalSplinePlan::new(
        &tb,
        &pb,
        SphericalSamples::Scattered(&samples),
        Some(&weights),
        0.1,
        SplineFitLimits::default(),
    )
    .unwrap();
    let mut w = grid.workspace(2).unwrap();
    let mut a = vec![0.; 2 * grid.mode_count()];
    let report = grid
        .fit_into(
            &values,
            a.as_mut_slice(),
            &mut w,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let b = scattered.fit(&values).unwrap();
    assert_eq!(a, b.spline().coefficients());
    assert_eq!(&report, b.report());
    assert_eq!(w.residuals().len(), samples.len());
    let mut again = vec![0.; a.len()];
    let repeated = grid
        .fit_into(
            &values,
            again.as_mut_slice(),
            &mut w,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(repeated, report);
    assert_eq!(a, again);
    assert!(report.residual() < 1e-24);
    assert_eq!(
        report.termination(),
        AdaptiveSplineTermination::FixedPenalty
    );
    assert_eq!(grid.penalty(), 0.1);
    assert_eq!(grid.knots()[0].len(), 9);
}

#[test]
fn errors_limits_and_workspace_recovery() {
    let p = [0., 0.25, 0.5, 1.];
    let values = [0., 1., 2., 3.];
    let supplied = CurveParameterization::Supplied(&p);
    assert!(
        fit_parametric_spline_with_options(
            &vec![Vec::<f64>::new(); 4],
            supplied,
            None,
            options(0.)
        )
        .is_err()
    );
    assert!(
        fit_parametric_spline_with_options(
            &values,
            CurveParameterization::Supplied(&[0., 0.5, 0.5, 1.]),
            None,
            options(0.)
        )
        .is_err()
    );
    assert!(
        fit_parametric_spline_with_options(
            &[0., 0., 1., 2.],
            CurveParameterization::ChordLength,
            None,
            options(0.)
        )
        .is_err()
    );
    assert!(
        fit_parametric_spline_with_options(&values, supplied, Some(&[-1.; 4]), options(0.))
            .is_err()
    );
    assert!(fit_parametric_spline_with_options(&values[..3], supplied, None, options(0.)).is_err());
    let plan = exact_plan::<f64>();
    let (_, _, samples) = grid::<f64>(6, 12);
    let values = samples.iter().copied().map(exact_field).collect::<Vec<_>>();
    let mut fitw = plan.workspace(1).unwrap();
    let mut coefficients = vec![0.; plan.mode_count()];
    let error = plan
        .fit_into(
            &values[..1],
            coefficients.as_mut_slice(),
            &mut fitw,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let fit = plan.fit(&values).unwrap();
    let spline = fit.spline();
    let mut w = spline.workspace().unwrap();
    let mut out = 99.;
    assert!(
        spline
            .evaluate_into(
                [-0.1, 0.],
                ExtrapolationPolicy::MethodDefined,
                &mut out,
                &mut w
            )
            .is_err()
    );
    assert!(
        spline
            .evaluate_into([1., f64::NAN], ExtrapolationPolicy::Clamp, &mut out, &mut w)
            .is_err()
    );
    assert_eq!(out, 99.);
    spline
        .derivative_into(
            [-0.1, 0.],
            [1, 0],
            ExtrapolationPolicy::Clamp,
            &mut out,
            &mut w,
        )
        .unwrap();
    assert_eq!(out, 0.);
    spline
        .evaluate_into([1., 0.3], REJECT, &mut out, &mut w)
        .unwrap();
    assert!(out.is_finite());
    let pi = core::f64::consts::PI;
    let ring = (0..10)
        .map(|i| [pi / 2., i as f64 / 10.])
        .collect::<Vec<_>>();
    let error = SphericalSplinePlan::<f64>::new(
        &[0., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&ring),
        None,
        0.,
        SplineFitLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    for (theta, phi, penalty) in [
        (&[0., pi][..], &longitude::<f64>()[..4], 0.),
        (&[0., 1.][..], &longitude::<f64>()[..], 0.),
        (&[0., pi][..], &longitude::<f64>()[..], -1.),
    ] {
        let error = SphericalSplinePlan::<f64>::new(
            theta,
            phi,
            SphericalSamples::Scattered(&samples),
            None,
            penalty,
            SplineFitLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }
}

#[test]
fn max_workspace_bytes_is_a_resource_limit() {
    let pi = core::f64::consts::PI;
    let (_, _, samples) = grid::<f64>(5, 8);
    let curve = fit_parametric_spline_with_options(
        &[0., 1., 2., 3.],
        CurveParameterization::Supplied(&[0., 0.25, 0.5, 1.]),
        None,
        options(0.).with_limits(limits(|l| l.with_max_workspace_bytes(1))),
    )
    .unwrap_err();
    assert_eq!(curve.kind(), SciOxideErrorKind::ResourceLimit);
    let sphere = fit_spherical_spline_with_options(
        SphericalSamples::Scattered(&samples),
        &vec![1.; samples.len()],
        None,
        options(0.).with_limits(limits(|l| l.with_max_workspace_bytes(1 << 20))),
    )
    .unwrap_err();
    assert_eq!(sphere.kind(), SciOxideErrorKind::ResourceLimit);
    let plan = SphericalSplinePlan::<f64>::new(
        &[0., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        None,
        0.,
        limits(|l| l.with_max_workspace_bytes(32 * 1024 * 1024 + 200_000)),
    )
    .unwrap();
    assert_eq!(
        plan.workspace_requirements(256).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut coefficients = vec![0.; 8 * plan.mode_count()];
    let wide = vec![[1.; 8]; samples.len()];
    let mut workspace = plan.workspace(8).unwrap();
    assert!(
        plan.fit_into(
            &wide,
            coefficients.as_mut_slice(),
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .is_ok()
    );
    let wider = vec![vec![1.; 256]; samples.len()];
    let error = plan.fit(&wider).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
}

#[test]
fn owned_and_scoped_forms_agree() {
    let pi = core::f64::consts::PI;
    let (_, _, samples) = grid::<f64>(6, 12);
    let values = samples
        .iter()
        .map(|&p| Complex::new(exact_field(p), p[0].sin()))
        .collect::<Vec<_>>();
    let owned = SphericalSplinePlan::new(
        &[0., pi / 2., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        None,
        0.2,
        SplineFitLimits::default(),
    )
    .unwrap();
    let scoped = SphericalSplinePlan::build(
        &[0., pi / 2., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        None,
        0.2,
        SplineFitLimits::default(),
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    // Owned forms run with sequential native execution; the scoped forms given
    // the same capabilities perform identical arithmetic.
    let cloned = scoped.clone();
    assert_eq!(cloned.knots(), scoped.knots());
    let a = owned.fit(&values).unwrap();
    let b = cloned.fit(&values).unwrap();
    assert_eq!(a.spline().coefficients(), b.spline().coefficients());
    assert_eq!(a.report(), b.report());
    let (_, _, sphere) = grid::<f64>(16, 16);
    let field = sphere.iter().map(|p| p[0].cos()).collect::<Vec<_>>();
    let adaptive = options::<f64>(1e-6).with_limits(limits(|l| l.with_max_penalty_iterations(2)));
    let a = fit_spherical_spline_with_options(
        SphericalSamples::Scattered(&sphere),
        &field,
        None,
        adaptive,
    )
    .unwrap();
    let b = fit_spherical_spline_scoped(
        SphericalSamples::Scattered(&sphere),
        &field,
        None,
        adaptive,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(a.spline().coefficients(), b.spline().coefficients());
    assert_eq!(a.report(), b.report());
    let p = (0..41).map(|i| i as f64 / 40.).collect::<Vec<_>>();
    let curve = p
        .iter()
        .map(|&x| Complex::new(x, (8. * x).sin()))
        .collect::<Vec<_>>();
    let a = fit_parametric_spline_with_options(
        &curve,
        CurveParameterization::ChordLength,
        None,
        options(1e-5),
    )
    .unwrap();
    let b = fit_parametric_spline_scoped(
        &curve,
        CurveParameterization::ChordLength,
        None,
        options(1e-5),
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(a.spline().coefficients(), b.spline().coefficients());
    assert_eq!(a.parameters(), b.parameters());
    assert_eq!(a.report(), b.report());
}

#[test]
fn budgets_and_cancellation_follow_the_solve_contract() {
    let p = (0..41).map(|i| i as f64 / 40.).collect::<Vec<_>>();
    let values = p.iter().map(|&x| [x, (8. * x).sin()]).collect::<Vec<_>>();
    // One admitted trial: the accepted least-squares fit is returned unsuccessfully.
    let budget = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let limited = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(1e-6).with_budgets(budget),
    )
    .unwrap();
    assert_eq!(
        limited.report().termination(),
        AdaptiveSplineTermination::BudgetExhausted
    );
    assert_eq!(limited.report().fit_trials(), 1);
    assert!(limited.report().residual() > 1e-6);
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = || ExecutionContext::sequential().with_cancellation(&token);
    let error = fit_parametric_spline_scoped(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(0.),
        cancelled(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    let (_, _, samples) = grid::<f64>(5, 8);
    let ones = vec![1.; samples.len()];
    let error = fit_spherical_spline_scoped(
        SphericalSamples::Scattered(&samples),
        &ones,
        None,
        options(0.),
        cancelled(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    let pi = core::f64::consts::PI;
    let error = SphericalSplinePlan::build(
        &[0., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        None,
        0.,
        SplineFitLimits::default(),
        cancelled(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    let plan = SphericalSplinePlan::new(
        &[0., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        None,
        0.,
        SplineFitLimits::default(),
    )
    .unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let mut out = vec![5.; plan.mode_count()];
    let error = plan
        .fit_into(&ones, out.as_mut_slice(), &mut workspace, cancelled())
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert!(out.iter().all(|&x| x == 5.));
}

#[test]
fn preflight_rejection_leaves_output_unchanged() {
    let plan = exact_plan::<f64>();
    let (_, _, samples) = grid::<f64>(6, 12);
    let values = samples
        .iter()
        .map(|&p| [exact_field(p), 2.])
        .collect::<Vec<_>>();
    let mut workspace = plan.workspace(2).unwrap();
    let mut out = vec![3.; 2 * plan.mode_count()];
    let mut bad = values.clone();
    bad[4][1] = f64::NAN;
    let mut ragged = values.iter().map(|v| v.to_vec()).collect::<Vec<_>>();
    ragged[2].push(1.);
    let context = ExecutionContext::sequential_native;
    for kind in [
        plan.fit_into(&values[1..], out.as_mut_slice(), &mut workspace, context())
            .unwrap_err()
            .kind(),
        plan.fit_into(&bad, out.as_mut_slice(), &mut workspace, context())
            .unwrap_err()
            .kind(),
        plan.fit_into(&ragged, out.as_mut_slice(), &mut workspace, context())
            .unwrap_err()
            .kind(),
        plan.fit_into(&values, &mut out[1..], &mut workspace, context())
            .unwrap_err()
            .kind(),
        plan.fit_into(
            &values,
            out.as_mut_slice(),
            &mut plan.workspace(1).unwrap(),
            context(),
        )
        .unwrap_err()
        .kind(),
    ] {
        assert!(matches!(
            kind,
            SciOxideErrorKind::DimensionMismatch | SciOxideErrorKind::NonFinite
        ));
        assert!(out.iter().all(|&x| x == 3.));
        assert!(workspace.residuals().is_empty());
    }
    let fit = plan.fit(&values).unwrap();
    let spline = fit.spline();
    let pi = core::f64::consts::PI;
    let small = SphericalSplinePlan::new(
        &[0., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        None,
        0.,
        SplineFitLimits::default(),
    )
    .unwrap()
    .fit(&vec![1.; samples.len()])
    .unwrap()
    .into_spline();
    let mut undersized = small.workspace().unwrap();
    assert_eq!(
        spline.workspace_capacity(&undersized),
        Capacity::NeedsGrowth
    );
    let mut value = [7.; 2];
    let error = spline
        .evaluate_into([1., 1.], REJECT, &mut value, &mut undersized)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let mut w = spline.workspace().unwrap();
    let error = spline
        .gradient_into([1., 1.], REJECT, &mut value, &mut w)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert!(
        spline
            .evaluate_into([4., 1.], REJECT, &mut value, &mut w)
            .is_err()
    );
    assert_eq!(value, [7.; 2]);
}

#[test]
fn conflicting_poles_preserve_irreducible_weighted_residual() {
    let pi = core::f64::consts::PI;
    let (_, _, mut samples) = grid::<f64>(5, 8);
    let mut values = vec![1.; samples.len()];
    samples.push([0., 0.7]);
    values.push(3.);
    let mut weights = vec![1.; samples.len()];
    *weights.last_mut().unwrap() = 2.;
    let plan = SphericalSplinePlan::new(
        &[0., pi],
        &longitude::<f64>(),
        SphericalSamples::Scattered(&samples),
        Some(&weights),
        0.,
        SplineFitLimits::default(),
    )
    .unwrap();
    let fit = plan.fit(&values).unwrap();
    let spline = fit.spline();
    // The fitted pole also participates in the interior objective; verify the
    // reported objective directly, including every original pole observation.
    let residual = samples
        .iter()
        .enumerate()
        .map(|(i, &p)| weights[i] * (spline.evaluate(p, REJECT).unwrap() - values[i]).powi(2))
        .sum::<f64>();
    assert_close::<f64, f64>(residual, fit.report().residual(), 1e-13, 0.0);
    assert!(fit.report().residual() >= 6.4 - 1e-12);
}

#[test]
fn scattered_interpolation_uses_explicit_wide_lq() {
    let samples = (0..13)
        .map(|i| [0.15 + 2.8 * (i as f64 / 12.), ((i * 7) % 13) as f64 * 0.43])
        .collect::<Vec<_>>();
    let values = samples
        .iter()
        .map(|p| p[0].sin() * (2. * p[1]).cos())
        .collect::<Vec<_>>();
    let fit = fit_spherical_spline_with_options(
        SphericalSamples::Scattered(&samples),
        &values,
        None,
        options(0.),
    )
    .unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::ResidualSatisfied,
        "{:?}",
        fit.report()
    );
    assert!(fit.spline().coefficients().len() > samples.len());
    assert!(fit.report().residual() < 1e-16);
}

#[cfg(feature = "std")]
#[test]
fn threaded_batches_are_deterministic_and_transactional() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, NdView, NdViewMut, StridedLayout, ThreadExecutor};
    use scioxide_interp::{BatchEvaluator, BatchRequest};
    let plan = exact_plan::<f64>();
    let (_, _, samples) = grid::<f64>(6, 12);
    let values = samples.iter().copied().map(exact_field).collect::<Vec<_>>();
    let fit = plan.fit(&values).unwrap();
    let workers = NonZeroUsize::new(2).unwrap();
    let batch = BatchEvaluator::new(fit.spline(), ExecutionPolicy::Parallel(workers));
    let query = [0.2, -1., 1.1, 7.3, 2.8, 3.];
    let queries = NdView::new(&query, StridedLayout::new(&[3, 2], &[2, 1]).unwrap()).unwrap();
    let executor = ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&executor);
    let request = BatchRequest::values(3).with_reports();
    let mut workspace = batch.workspace(request).unwrap();
    let mut output = [99.; 3];
    let unit = scioxide_core::Interval::new(0.0, 1.0).unwrap();
    let mut reports = [ExtrapolationPolicy::MethodDefined
        .resolve([unit; 2], [99.; 2])
        .unwrap(); 3];
    {
        let mut out =
            NdViewMut::new(&mut output, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
        batch
            .evaluate_into(
                &queries,
                REJECT,
                &mut out,
                Some(reports.as_mut_slice()),
                &mut workspace,
                context,
            )
            .unwrap();
    }
    for (i, p) in query.chunks_exact(2).enumerate() {
        assert_close::<f64, f64>(output[i], exact_field([p[0], p[1]]), 2e-12, 0.0);
        assert_eq!(
            output[i],
            fit.spline().evaluate([p[0], p[1]], REJECT).unwrap()
        );
        assert_eq!(reports[i].outside_distance(), [0., 0.]);
    }
    assert!((reports[1].effective()[1] - (7.3 - core::f64::consts::TAU)).abs() < 1e-15);
    let original = output;
    let original_reports = reports;
    let bad_query = [0.2, -1., -0.1, 7.3, 2.8, 3.];
    let bad = NdView::new(&bad_query, StridedLayout::new(&[3, 2], &[2, 1]).unwrap()).unwrap();
    let mut out =
        NdViewMut::new(&mut output, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
    assert!(
        batch
            .evaluate_into(
                &bad,
                REJECT,
                &mut out,
                Some(reports.as_mut_slice()),
                &mut workspace,
                context
            )
            .is_err()
    );
    assert_eq!(output, original);
    assert_eq!(reports, original_reports);
}

fn decimal_fixture_family<R: RealScalar>() {
    let (_, _, samples) = grid::<R>(6, 12);
    let values = samples.iter().copied().map(exact_field).collect::<Vec<_>>();
    let fit = exact_plan::<R>().fit(&values).unwrap();
    let spline = fit.spline();
    let reject = ExtrapolationPolicy::Reject;
    let mut w = spline.workspace().unwrap();
    let mut out = R::ZERO;
    let p = (0..21)
        .map(|i| from_f64::<R>(i as f64 / 20.))
        .collect::<Vec<_>>();
    let y = p.iter().map(|&t| t * t * t).collect::<Vec<_>>();
    let curve =
        fit_parametric_spline_with_options(&y, CurveParameterization::Supplied(&p), None, fixed())
            .unwrap();
    let mut cw = curve.spline().workspace().unwrap();
    let tolerance = width_tolerance::<R>(8e-5, 3e-12);
    for row in include_str!("data/spherical_parametric.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cells = row.split('\t').collect::<Vec<_>>();
        let values = cells[..11]
            .iter()
            .map(|s| from_f64::<R>(s.parse::<f64>().unwrap()))
            .collect::<Vec<_>>();
        for (orders, index) in [([0, 0], 2), ([1, 0], 3), ([0, 1], 4)] {
            spline
                .derivative_into([values[0], values[1]], orders, reject, &mut out, &mut w)
                .unwrap();
            assert_close::<R, R>(out, values[index], tolerance, 0.0);
        }
        curve
            .spline()
            .evaluate_into([values[7]], reject, &mut out, &mut cw)
            .unwrap();
        assert_close::<R, R>(out, values[8], tolerance, 0.0);
        curve
            .spline()
            .derivative_into([values[7]], [1], None, reject, &mut out, &mut cw)
            .unwrap();
        assert_close::<R, R>(out, values[9], tolerance, 0.0);
        curve
            .spline()
            .integral_into([R::ZERO], [values[7]], reject, &mut out, &mut cw)
            .unwrap();
        assert_close::<R, R>(out, values[10], tolerance, 0.0);
        let h = cells[2].parse::<f64>().unwrap();
        assert_eq!(h.to_bits(), u64::from_str_radix(cells[12], 16).unwrap());
        assert_eq!(
            (h as f32).to_bits(),
            u32::from_str_radix(cells[11], 16).unwrap()
        );
    }
}
#[test]
fn independent_decimal_reference_widths() {
    decimal_fixture_family::<f32>();
    decimal_fixture_family::<f64>();
}

fn complex_adaptive_family<R: RealScalar>() {
    let p = (0..25)
        .map(|i| from_f64::<R>(i as f64 / 24.))
        .collect::<Vec<_>>();
    let values = p
        .iter()
        .map(|&t| Complex::new(t, t * t * t * t))
        .collect::<Vec<_>>();
    let fit = fit_parametric_spline_with_options(
        &values,
        CurveParameterization::Supplied(&p),
        None,
        options(1e-5).with_limits(limits(|l| l.with_max_penalty_iterations(4))),
    )
    .unwrap();
    // The residual bound holds; four penalty trials do not reach the target.
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::PenaltySearchLimit
    );
    assert!(f(fit.report().residual()) <= 1e-5 * (1. + 1e-5) + 1e-16);
    assert!(fit.report().refinements() > 0);
    let samples = (0..13)
        .map(|i| {
            [
                from_f64::<R>(0.15 + 2.8 * (i as f64 / 12.)),
                from_f64(((i * 7) % 13) as f64 * 0.43),
            ]
        })
        .collect::<Vec<_>>();
    let values = samples
        .iter()
        .map(|p| Complex::new(p[0].sin() * (from_f64::<R>(2.) * p[1]).cos(), p[0].cos()))
        .collect::<Vec<_>>();
    let fit = fit_spherical_spline_with_options(
        SphericalSamples::Scattered(&samples),
        &values,
        None,
        options(0.).with_residual_tolerance(Tolerance::new(from_f64(1e-8), R::ZERO).unwrap()),
    )
    .unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::ResidualSatisfied
    );
    assert!(f(fit.report().residual()) < 1e-8);
}
#[test]
fn adaptive_complex_widths() {
    complex_adaptive_family::<f32>();
    complex_adaptive_family::<f64>();
}

#[test]
fn angular_quadrupole_converges_and_fixed_penalty_changes_fit() {
    let pi = core::f64::consts::PI;
    let (_, _, samples) = grid::<f64>(8, 24);
    // Unnormalized real l=2,m=2 spherical harmonic: x^2-y^2.
    let values = samples
        .iter()
        .map(|p| p[0].sin().powi(2) * (2. * p[1]).cos())
        .collect::<Vec<_>>();
    let fitted = |latitude: usize, longitude: usize, lambda: f64| {
        let theta = (0..=latitude)
            .map(|i| pi * i as f64 / latitude as f64)
            .collect::<Vec<_>>();
        let phi = (0..=longitude)
            .map(|i| 2. * pi * i as f64 / longitude as f64)
            .collect::<Vec<_>>();
        SphericalSplinePlan::new(
            &theta,
            &phi,
            SphericalSamples::Scattered(&samples),
            None,
            lambda,
            SplineFitLimits::default(),
        )
        .unwrap()
        .fit(&values)
        .unwrap()
    };
    let coarse = fitted(4, 4, 0.);
    let fine = fitted(8, 16, 0.);
    let smooth = fitted(8, 16, 1.);
    assert!(smooth.report().residual() > fine.report().residual());
    let error = |model: &SphericalSpline<f64>| {
        let mut w = model.workspace().unwrap();
        let mut out = 0.;
        let mut error = 0_f64;
        for t in [0.17, 0.33, 0.57, 0.83] {
            for p in [0.21, 0.71, 1.3, 2.6, 5.4] {
                let theta = t * pi;
                model
                    .evaluate_into([theta, p], REJECT, &mut out, &mut w)
                    .unwrap();
                error = error.max((out - theta.sin().powi(2) * (2. * p).cos()).abs());
            }
        }
        error
    };
    let coarse_error = error(coarse.spline());
    let fine_error = error(fine.spline());
    assert!(
        fine_error < coarse_error / 3.,
        "coarse={coarse_error}, fine={fine_error}"
    );
    assert!(fine_error < 0.002, "{fine_error}");
    let spline = fine.spline();
    let mut w = spline.workspace().unwrap();
    let mut out = 0.;
    spline
        .integral_into([0., 0.], [pi, 2. * pi], REJECT, &mut out, &mut w)
        .unwrap();
    assert_close::<f64, f64>(out, 0., 1e-11, 0.0);
    // Periodic cubic interior derivatives, with nonzero tensor coefficients.
    for order in [[0, 0], [0, 1], [0, 2], [1, 1]] {
        spline
            .derivative_into([1.1, 1e-7], order, REJECT, &mut out, &mut w)
            .unwrap();
        let left = out;
        spline
            .derivative_into([1.1, 2. * pi - 1e-7], order, REJECT, &mut out, &mut w)
            .unwrap();
        assert_close::<f64, f64>(out, left, 2e-6, 0.0);
    }
}

/// `n` samples of the curve `[t, sin 8t]` at `t = i / (n − 1)`.
fn sine_curve<R: RealScalar>(n: usize) -> Vec<[R; 2]> {
    (0..n)
        .map(|i| i as f64 / (n - 1) as f64)
        .map(|t| [from_f64(t), from_f64((8. * t).sin())])
        .collect()
}
/// `cos θ` on the grid `θᵢ = πi/16`, `φⱼ = πj/8`.
fn cosine_sphere() -> (Vec<[f64; 2]>, Vec<f64>) {
    let pi = core::f64::consts::PI;
    let points: Vec<[f64; 2]> = (0..=16)
        .flat_map(|i| (0..16).map(move |j| [pi * i as f64 / 16., pi * j as f64 / 8.]))
        .collect();
    let heights = points.iter().map(|p| p[0].cos()).collect();
    (points, heights)
}
fn bound<R: RealScalar>(residual: R, tolerance: Tolerance<R>) -> AdaptiveSplineOptions<R> {
    AdaptiveSplineOptions::new()
        .with_target(SplineFitTarget::Residual(residual))
        .with_residual_tolerance(tolerance)
}
fn bits<R: RealScalar>(values: &[R]) -> Vec<u64> {
    values.iter().map(|&v| f(v).to_bits()).collect()
}
fn assert_unconverged(
    error: &SciOxideError,
    expected: ConvergenceReason,
    algorithm: AlgorithmId,
    trials: usize,
) {
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure { reason, diagnostics, .. }
                if reason == expected
                    && diagnostics.algorithm() == Some(algorithm)
                    && diagnostics.counts().own().iterations() == trials
        ),
        "{error:?}"
    );
}

#[test]
fn one_shot_fits_return_the_spline_of_their_configured_forms() {
    fn curve_family<R: RealScalar>(residual: f64, relative: f64) {
        let points = sine_curve::<R>(41);
        let tolerance = Tolerance::new(R::ZERO, from_f64(relative)).unwrap();
        let curve = fit_parametric_spline(&points, from_f64(residual), tolerance).unwrap();
        let chord = CurveParameterization::ChordLength;
        let options = bound(from_f64(residual), tolerance);
        let fit = fit_parametric_spline_with_options(&points, chord, None, options).unwrap();
        assert_eq!(
            fit.report().termination(),
            AdaptiveSplineTermination::ResidualSatisfied
        );
        assert_eq!(
            bits(curve.coefficients()),
            bits(fit.spline().coefficients())
        );
        assert_eq!(
            bits(curve.bases()[0].knots()),
            bits(fit.spline().bases()[0].knots())
        );
    }
    curve_family::<f32>(1e-3, 1e-2);
    curve_family::<f64>(1e-6, 1e-5);

    let (points, heights) = cosine_sphere();
    let tolerance = Tolerance::new(0., 1e-5).unwrap();
    let surface = fit_spherical_spline(&points, &heights, 1e-6, tolerance).unwrap();
    let samples = SphericalSamples::Scattered(&points);
    let fit =
        fit_spherical_spline_with_options(samples, &heights, None, bound(1e-6, tolerance)).unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::ResidualSatisfied
    );
    assert_eq!(
        bits(surface.coefficients()),
        bits(fit.spline().coefficients())
    );
    let value = surface.evaluate([0.6, 1.0], REJECT).unwrap();
    assert!((value - 0.6_f64.cos()).abs() < 3e-4);

    // Complex values on scattered sites in single precision, interpolated.
    let sites: Vec<[f32; 2]> = (0..13)
        .map(|i| [0.15 + 2.8 * (i as f32 / 12.), ((i * 7) % 13) as f32 * 0.43])
        .collect();
    let values: Vec<_> = sites
        .iter()
        .map(|p| Complex::new(p[0].sin() * (2. * p[1]).cos(), p[0].cos()))
        .collect();
    let tolerance = Tolerance::new(1e-8_f32, 0.).unwrap();
    let surface = fit_spherical_spline(&sites, &values, 0., tolerance).unwrap();
    let samples = SphericalSamples::Scattered(&sites);
    let fit =
        fit_spherical_spline_with_options(samples, &values, None, bound(0., tolerance)).unwrap();
    assert!(fit.report().termination().is_success());
    assert_eq!(
        bits(surface.coefficients()),
        bits(fit.spline().coefficients())
    );
}

#[test]
fn one_shot_fits_turn_unsuccessful_searches_into_convergence_failures() {
    let parametric = scioxide_core::algorithm_id!("interp", "parametric-spline", 1);
    let spherical = scioxide_core::algorithm_id!("interp", "spherical-adaptive-spline", 1);
    let chord = CurveParameterization::ChordLength;
    // The default tolerance ε leaves the penalty search a window it does not find.
    let points = sine_curve::<f64>(41);
    let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Residual(1e-6));
    let fit = fit_parametric_spline_with_options(&points, chord, None, options).unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::PenaltySearchLimit
    );
    let epsilon = Tolerance::from_machine_epsilon();
    let error = fit_parametric_spline(&points, 1e-6, epsilon).unwrap_err();
    let trials = fit.report().fit_trials();
    assert_unconverged(
        &error,
        ConvergenceReason::IterationLimit,
        parametric,
        trials,
    );
    // Interpolating 200 samples needs more than 64 knot insertions.
    let points = sine_curve::<f64>(200);
    let tolerance = Tolerance::new(1e-20, 0.).unwrap();
    let fit =
        fit_parametric_spline_with_options(&points, chord, None, bound(0., tolerance)).unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::RefinementLimit
    );
    let error = fit_parametric_spline(&points, 0., tolerance).unwrap_err();
    let trials = fit.report().fit_trials();
    assert_unconverged(
        &error,
        ConvergenceReason::TruncationLimit,
        parametric,
        trials,
    );

    let (points, heights) = cosine_sphere();
    let samples = SphericalSamples::Scattered(&points);
    let fit =
        fit_spherical_spline_with_options(samples, &heights, None, bound(1e-6, epsilon)).unwrap();
    assert_eq!(
        fit.report().termination(),
        AdaptiveSplineTermination::PenaltySearchLimit
    );
    let error = fit_spherical_spline(&points, &heights, 1e-6, epsilon).unwrap_err();
    let trials = fit.report().fit_trials();
    assert_unconverged(&error, ConvergenceReason::IterationLimit, spherical, trials);
}

#[test]
fn one_shot_fits_reject_invalid_bounds_and_samples() {
    let points = sine_curve::<f64>(41);
    let tolerance = Tolerance::new(0., 1e-5).unwrap();
    for residual in [-1e-6, f64::NAN, f64::INFINITY] {
        let error = fit_parametric_spline(&points, residual, tolerance).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }
    let error = fit_parametric_spline(&points[..3], 1e-6, tolerance).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    let mut repeated = points;
    repeated[1] = repeated[0];
    let error = fit_parametric_spline(&repeated, 1e-6, tolerance).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);

    let (points, heights) = cosine_sphere();
    let error = fit_spherical_spline(&points, &heights, -1., tolerance).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    let error = fit_spherical_spline(&points, &heights[1..], 1e-6, tolerance).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let mut outside = points;
    outside[0] = [4.0, 0.0];
    let error = fit_spherical_spline(&outside, &heights, 1e-6, tolerance).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
}
