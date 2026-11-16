#![allow(clippy::unwrap_used, clippy::too_many_arguments)]
use super::*;
use crate::{IntegrationTermination, SubdivisionOptions};

use scioxide_core::{
    Budget, BudgetKind, Complex, ExecutionContext, Interval, NumericScalar, ProblemToken,
    SolveBudgets, SolveContext, SolveIdentity, SolveState, Tolerance,
};
use std::sync::atomic::AtomicUsize as Cell;

fn n<S: RealScalar + Default>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn interval<S: RealScalar + Default>(a: f64, b: f64) -> Interval<S> {
    Interval::new(n(a), n(b)).unwrap()
}
fn options<S: RealScalar + Default>() -> SubdivisionOptions<S> {
    SubdivisionOptions::default().with_tolerance(scioxide_core::ErrorTolerance::from_tolerance(
        Tolerance::new(n::<S>(2e-9).max(n::<S>(4000.0) * S::EPSILON), S::ZERO).unwrap(),
        scioxide_core::NormSelection::Infinity,
    ))
}
fn context<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(595),
        scioxide_core::algorithm_id!("integrate", "oscillatory-test", 1),
        0,
        0,
    );
    f(&mut state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap())
}
fn sequential<T>(f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    context(ExecutionContext::sequential(), SolveBudgets::unbounded(), f)
}
struct Linear {
    sign: f64,
}
impl<S: RealScalar + Default> OscillatoryPhase<S> for Linear {
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError> {
        Ok((n::<S>(self.sign) * x, n(self.sign)))
    }
    fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<S>> {
        Some(self)
    }
}
impl<S: RealScalar + Default> OscillatoryPhaseInverse<S> for Linear {
    fn invert(&self, _: Interval<S>, y: S) -> Result<S, SciOxideError> {
        Ok(n::<S>(self.sign) * y)
    }
}
fn linear_plan<S: RealScalar + Default>(omega: f64, sign: f64) -> OscillatoryGeometry<S> {
    OscillatoryGeometry::new(
        interval(0.0, 1.0),
        n(omega),
        &[OscillatoryRegion::nonstationary(interval(0.0, 1.0), interval(sign, sign)).unwrap()],
    )
    .unwrap()
}
fn linear_cases<S: RealScalar + Default>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let opts = options();
    for sign in [-1.0, 1.0] {
        for omega in [
            -100000.0, -100.0, -1.0, 0.0, 1.0, 20.0, 63.0, 65.0, 100.0, 1000.0, 100000.0,
        ] {
            for weight in [FourierWeight::Sine, FourierWeight::Cosine] {
                let phase = Linear { sign };
                let w = omega * sign;
                let exact = if w == 0.0 {
                    match weight {
                        FourierWeight::Sine => 0.0,
                        FourierWeight::Cosine => 1.0,
                    }
                } else {
                    match weight {
                        FourierWeight::Sine => (1.0 - w.cos()) / w,
                        FourierWeight::Cosine => w.sin() / w,
                    }
                };
                let plans = linear_plan(omega, sign);
                let l = OscillatoryQuadraturePlan::new(
                    OscillatoryMethod::Levin,
                    plans.clone(),
                    weight,
                    crate::OscillatoryOptions::new(opts.clone()),
                )
                .unwrap()
                .integrate(|_| Ok(S::ONE), &phase)
                .unwrap();
                let f = OscillatoryQuadraturePlan::new(
                    OscillatoryMethod::Filon,
                    plans,
                    weight,
                    crate::OscillatoryOptions::new(opts.clone()),
                )
                .unwrap()
                .integrate(|_| Ok(S::ONE), &phase)
                .unwrap();
                let bound = if S::EPSILON.into_f64() > 1e-10 {
                    6e-4
                } else {
                    2e-8
                };
                for (name, result) in [("Levin", l), ("Filon", f)] {
                    assert!(
                        result.1.is_success(),
                        "{name} omega {omega} sign {sign}: {result:?}"
                    );
                    assert!(
                        (result.0.into_f64() - exact).abs() < bound,
                        "{name}: {result:?} vs {exact}"
                    );
                }
            }
        }
    }
}
#[test]
fn linear_f64() {
    linear_cases::<f64>();
}
#[test]
fn linear_f32() {
    linear_cases::<f32>();
}
struct Square;
impl<S: RealScalar + Default> OscillatoryPhase<S> for Square {
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError> {
        Ok((x * x, n::<S>(2.0) * x))
    }
    fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<S>> {
        Some(self)
    }
}
impl<S: RealScalar + Default> OscillatoryPhaseInverse<S> for Square {
    fn invert(&self, i: Interval<S>, y: S) -> Result<S, SciOxideError> {
        Ok(if i.upper() <= S::ZERO {
            -y.sqrt()
        } else {
            y.sqrt()
        })
    }
}
fn nonlinear<S: RealScalar + Default>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let domain = interval(1.0, 2.0);
    for omega in [10.0, 100.0, 1000.0, 100000.0] {
        let phase = OscillatoryGeometry::new(
            domain,
            n(omega),
            &[OscillatoryRegion::nonstationary(domain, interval(2.0, 4.0)).unwrap()],
        )
        .unwrap();
        let opts = options();
        let exact = ((4.0 * omega).sin() - omega.sin()) / omega;
        let amplitude = |x: S| Ok(n::<S>(2.0) * x);
        for result in [
            OscillatoryQuadraturePlan::new(
                OscillatoryMethod::Levin,
                phase.clone(),
                FourierWeight::Cosine,
                crate::OscillatoryOptions::new(opts.clone()),
            )
            .unwrap()
            .integrate(&amplitude, &Square)
            .unwrap(),
            OscillatoryQuadraturePlan::new(
                OscillatoryMethod::Filon,
                phase,
                FourierWeight::Cosine,
                crate::OscillatoryOptions::new(opts.clone()),
            )
            .unwrap()
            .integrate(&amplitude, &Square)
            .unwrap(),
        ] {
            let bound = if S::EPSILON.into_f64() > 1e-10 {
                6e-4
            } else {
                2e-8
            };
            assert!(
                (result.0.into_f64() - exact).abs() < bound,
                "omega {omega}: {result:?} vs {exact}"
            );
        }
    }
}
#[test]
fn nonlinear_f64() {
    nonlinear::<f64>();
}
#[test]
fn nonlinear_f32() {
    nonlinear::<f32>();
}
#[test]
fn declarations_reject_gaps_overlap_zero_derivative_and_nonfinite_frequency() {
    let i = interval::<f64>(0.0, 1.0);
    assert!(OscillatoryRegion::nonstationary(i, interval(-1.0, 1.0)).is_err());
    let r = OscillatoryRegion::nonstationary(interval(0.0, 0.5), interval(1.0, 1.0)).unwrap();
    for regions in [&[][..], &[r][..], &[r, r][..]] {
        assert!(OscillatoryGeometry::new(i, 10.0, regions).is_err());
    }
    assert!(OscillatoryGeometry::new(i, f64::NAN, &[]).is_err());
    assert!(OscillatoryRegion::stationary(interval::<f64>(0.0, 0.0), interval(0.0, 0.0)).is_err());
}

#[test]
fn evaluation_admission_and_callback_failure_preserve_outputs() {
    let limited_options =
        options::<f64>().with_max_evaluations(Some(Budget::try_from(24).unwrap()));
    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        linear_plan(1000.0, 1.0),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(limited_options),
    )
    .unwrap();
    let calls = Cell::new(0);
    let mut ws = plan.workspace(1).unwrap();
    let mut output = -7.0;
    let error = plan
        .integrate_into(
            |_: f64| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(1.0)
            },
            &Linear { sign: 1.0 },
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(output, -7.0);
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        linear_plan(1000.0, 1.0),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(options::<f64>()),
    )
    .unwrap();
    let error = plan
        .integrate_into(
            |_: f64| -> Result<f64, SciOxideError> {
                Err(crate::kernel::invalid(
                    scioxide_core::argument!("amplitude"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "test failure",
                ))
            },
            &Linear { sign: 1.0 },
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert_eq!(output, -7.0);
    assert!(ws.component_errors().is_none());
}

struct Power(usize);
impl<S: RealScalar + Default> OscillatoryPhase<S> for Power {
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError> {
        Ok(if self.0 == 2 {
            (x * x, n::<S>(2.0) * x)
        } else {
            (x * x * x * x, n::<S>(4.0) * x * x * x)
        })
    }
    fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<S>> {
        Some(self)
    }
}
impl<S: RealScalar + Default> OscillatoryPhaseInverse<S> for Power {
    fn invert(&self, i: Interval<S>, y: S) -> Result<S, SciOxideError> {
        let x = if self.0 == 2 {
            y.sqrt()
        } else {
            y.sqrt().sqrt()
        };
        Ok(if i.upper() <= S::ZERO { -x } else { x })
    }
}
fn stationary_plan<S: RealScalar + Default>(
    omega: f64,
    power: usize,
    interior: bool,
    delta: f64,
) -> OscillatoryGeometry<S> {
    let derivative = |x: f64| if power == 2 { 2.0 * x } else { 4.0 * x * x * x };
    let mut positive: Vec<OscillatoryRegion<S>> = Vec::new();
    let mut x = delta;
    while x < 1.0 {
        let y = (2.0 * x).min(1.0);
        positive.push(
            OscillatoryRegion::nonstationary(
                interval(x, y),
                interval(derivative(x), derivative(y)),
            )
            .unwrap(),
        );
        x = y;
    }
    let mut regions = Vec::new();
    if interior {
        for r in positive.iter().rev() {
            regions.push(
                OscillatoryRegion::nonstationary(
                    Interval::new(-r.interval.upper(), -r.interval.lower()).unwrap(),
                    Interval::new(-r.derivative.upper(), -r.derivative.lower()).unwrap(),
                )
                .unwrap(),
            );
        }
    }
    regions.push(
        OscillatoryRegion::stationary(
            interval(if interior { -delta } else { 0.0 }, delta),
            interval(
                if interior { -derivative(delta) } else { 0.0 },
                derivative(delta),
            ),
        )
        .unwrap(),
    );
    regions.extend(positive);
    OscillatoryGeometry::new(
        interval(if interior { -1.0 } else { 0.0 }, 1.0),
        n(omega),
        &regions,
    )
    .unwrap()
}
fn stationary_fixtures<S: RealScalar + Default>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    for line in include_str!("../../tests/data/oscillatory.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split(',').collect();
        let power: usize = fields[0].parse().unwrap();
        let omega: f64 = fields[1].parse().unwrap();
        for interior in [false, true] {
            for (weight, column) in [(FourierWeight::Cosine, 2), (FourierWeight::Sine, 3)] {
                let expected =
                    fields[column].parse::<f64>().unwrap() * if interior { 2.0 } else { 1.0 };
                let geometry = stationary_plan(omega, power, interior, 0.25);
                let opts = options();
                let l = OscillatoryQuadraturePlan::new(
                    OscillatoryMethod::Levin,
                    geometry.clone(),
                    weight,
                    crate::OscillatoryOptions::new(opts.clone()),
                )
                .unwrap()
                .integrate(|_| Ok(S::ONE), &Power(power))
                .unwrap();
                let f = OscillatoryQuadraturePlan::new(
                    OscillatoryMethod::Filon,
                    geometry,
                    weight,
                    crate::OscillatoryOptions::new(opts.clone()),
                )
                .unwrap()
                .integrate(|_| Ok(S::ONE), &Power(power))
                .unwrap();
                for (method, result) in [("Levin", l), ("Filon", f)] {
                    let bound = if S::EPSILON.into_f64() > 1e-10 {
                        6e-4
                    } else {
                        2e-8
                    };
                    assert!(
                        (result.0.into_f64() - expected).abs() < bound,
                        "{method} p{power} w{omega} interior{interior}: {result:?} expected{expected}"
                    );
                    // A provider/arithmetic floor may limit f32 at large phases.
                    assert!(
                        result.1.is_success()
                            || result.1.termination() == IntegrationTermination::Stagnation,
                        "{result:?}"
                    );
                }
            }
        }
    }
}
#[test]
fn stationary_independent_f64() {
    stationary_fixtures::<f64>();
}
#[test]
fn stationary_independent_f32() {
    stationary_fixtures::<f32>();
}
#[test]
fn high_frequency_stationary_asymptotic_and_cancellation() {
    let omega = 1_000_000.0f64;
    let geometry = stationary_plan(omega, 2, true, 1.0 / 1024.0);
    let opts = options::<f64>();
    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        geometry.clone(),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts.clone()),
    )
    .unwrap();
    let filon = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        geometry,
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts),
    )
    .unwrap();
    for r in [
        plan.integrate(|_| Ok(1.0), &Square).unwrap(),
        filon.integrate(|_| Ok(1.0), &Square).unwrap(),
    ] {
        assert!(r.1.is_success(), "{r:?}");
        let leading = (core::f64::consts::PI / 2.0).sqrt();
        assert!((omega.sqrt() * r.0 - leading).abs() < 0.0021, "{r:?}");
        assert!(r.1.evaluations() < 10000, "{r:?}");
    }
    for r in [
        plan.integrate(Ok, &Square).unwrap(),
        filon.integrate(Ok, &Square).unwrap(),
    ] {
        assert!(r.0.abs() < 2e-8, "{r:?}");
    }
}
#[test]
fn context_budget_counts_phase_amplitude_and_factors_exactly() {
    for (levin, needed) in [(true, 54), (false, 73)] {
        let opts = options::<f64>();
        let phase = Linear { sign: 1.0 };
        let l = OscillatoryQuadraturePlan::new(
            OscillatoryMethod::Levin,
            linear_plan(1000.0, 1.0),
            FourierWeight::Cosine,
            crate::OscillatoryOptions::new(opts.clone()),
        )
        .unwrap();
        let f = OscillatoryQuadraturePlan::new(
            OscillatoryMethod::Filon,
            linear_plan(1000.0, 1.0),
            FourierWeight::Cosine,
            crate::OscillatoryOptions::new(opts),
        )
        .unwrap();
        for limit in [needed - 1, needed] {
            let mut ws = if levin {
                l.workspace(1)
            } else {
                f.workspace(1)
            }
            .unwrap();
            let mut output = -77.0;
            let calls = Cell::new(0);
            context(
                ExecutionContext::sequential(),
                SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap()),
                |ctx| {
                    let amplitude = |_: f64| {
                        calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                        Ok(1.0)
                    };
                    let result = if levin {
                        l.integrate_into(&amplitude, &phase, &mut output, &mut ws, &mut *ctx)
                    } else {
                        f.integrate_into(&amplitude, &phase, &mut output, &mut ws, &mut *ctx)
                    };
                    if limit < needed {
                        assert!(result.is_err());
                        assert_eq!(output, -77.0);
                        assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
                        assert!(ws.last_attempt().unwrap().progress().is_none());
                    } else {
                        let report = result.unwrap();
                        assert_eq!(report.evaluations(), 25);
                        assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 25);
                        assert_eq!(
                            ctx.counts().total().unwrap().total_providers().unwrap(),
                            needed
                        );
                    }
                },
            );
        }
    }
}

#[test]
fn inverse_and_derivative_violations_are_transactional() {
    struct Bad {
        inverse: bool,
        nan: bool,
    }
    impl OscillatoryPhase<f64> for Bad {
        fn evaluate(&self, x: f64) -> Result<(f64, f64), SciOxideError> {
            Ok((
                if self.nan { f64::NAN } else { x },
                if self.inverse { 1.0 } else { 0.0 },
            ))
        }
        fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<f64>> {
            Some(self)
        }
    }
    impl OscillatoryPhaseInverse<f64> for Bad {
        fn invert(&self, _: Interval<f64>, y: f64) -> Result<f64, SciOxideError> {
            Ok(if self.inverse { 0.5 } else { y })
        }
    }
    let opts = options::<f64>();

    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        linear_plan(1000.0, 1.0),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts),
    )
    .unwrap();
    let mut ws = plan.workspace(1).unwrap();
    for p in [
        Bad {
            inverse: true,
            nan: false,
        },
        Bad {
            inverse: false,
            nan: false,
        },
        Bad {
            inverse: false,
            nan: true,
        },
    ] {
        let sample = 0.0;
        let mut output = -33.0;
        let result = sequential(|ctx| {
            plan.integrate_into(
                |_point| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        let y = &mut lane;

                        *y = 1.0;
                        Ok::<(), SciOxideError>(())
                    }?;
                    Ok(lane)
                },
                &p,
                &mut output,
                &mut ws,
                &mut *ctx,
            )
        });
        assert!(result.is_err());
        assert_eq!(output, -33.0);
        assert!(ws.component_errors().is_none());
    }
    assert!(
        plan.integrate(|_| Ok(1.0), &Linear { sign: 1.0 })
            .unwrap()
            .1
            .is_success()
    );
}

#[test]
fn zero_width_zero_sine_empty_values_and_panel_limit() {
    let opts = options::<f64>();
    let forbidden =
        |_: f64| -> Result<f64, SciOxideError> { Err(SciOxideError::external("must not sample")) };
    let geometry = OscillatoryGeometry::new(interval(1.0, 1.0), 1000.0, &[]).unwrap();
    let (value, report) = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        geometry,
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts.clone()).with_ordinary_panels(Budget::ONE),
    )
    .unwrap()
    .integrate(&forbidden, &Linear { sign: 1.0 })
    .unwrap();
    assert_eq!(value, 0.0);
    assert_eq!(
        report.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert_eq!(report.evaluations(), 0);
    let (value, report) = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        linear_plan(0.0, 1.0),
        FourierWeight::Sine,
        crate::OscillatoryOptions::new(opts.clone()).with_ordinary_panels(Budget::ONE),
    )
    .unwrap()
    .integrate(&forbidden, &Linear { sign: 1.0 })
    .unwrap();
    assert_eq!(value, 0.0);
    assert!(report.is_success());
    let region = OscillatoryRegion::stationary(interval(0.0, 1.0), interval(-2.0, 2.0)).unwrap();
    let geometry = OscillatoryGeometry::new(interval(0.0, 1.0), 1000.0, &[region]).unwrap();
    let error = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        geometry,
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts).with_ordinary_panels(Budget::ONE),
    )
    .unwrap()
    .integrate(&forbidden, &Square)
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::TruncationLimit,
            ..
        }
    ));
}

#[test]
fn refinement_and_evaluation_stops_keep_complete_measurements() {
    let amplitude = |x: f64| Ok(1.0 / (1.0 + 10000.0 * (x - 0.3) * (x - 0.3)));
    let prepare = |options| {
        OscillatoryQuadraturePlan::new(
            OscillatoryMethod::Filon,
            linear_plan(1000.0, 1.0),
            FourierWeight::Cosine,
            crate::OscillatoryOptions::new(options),
        )
        .unwrap()
    };
    let initial =
        prepare(options::<f64>().with_max_evaluations(Some(Budget::try_from(25).unwrap())))
            .integrate(&amplitude, &Linear { sign: 1.0 })
            .unwrap();
    assert_eq!(
        initial.1.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
    );
    assert!(initial.0.is_finite());
    assert_eq!(initial.1.evaluations(), 25);
    let before_pair =
        prepare(options::<f64>().with_max_evaluations(Some(Budget::try_from(74).unwrap())))
            .integrate(&amplitude, &Linear { sign: 1.0 })
            .unwrap();
    assert_eq!(before_pair.0, initial.0);
    assert_eq!(before_pair.1.evaluations(), 25);
    let capped = prepare(options::<f64>().with_max_subdivisions(Budget::try_from(1).unwrap()))
        .integrate(&amplitude, &Linear { sign: 1.0 })
        .unwrap();
    assert_eq!(
        capped.1.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
    );
    assert!(capped.0.is_finite());
}

#[test]
fn undersized_workspace_empty_views_and_panel_budget_are_explicit() {
    let opts = options::<f64>();
    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        linear_plan(1000.0, 1.0),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts.clone())
            .with_ordinary_panels(Budget::try_from(2).unwrap()),
    )
    .unwrap();
    let small = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        linear_plan(1000.0, 1.0),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts.clone().with_max_subdivisions(Budget::ONE)),
    )
    .unwrap();
    let mut ws = small.workspace(1).unwrap();
    let mut out = -1.0;
    assert!(
        plan.integrate_into(
            |_: f64| Ok(1.0),
            &Linear { sign: 1.0 },
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(out, -1.0);
    let mut ws = plan.workspace(0).unwrap();
    let mut empty: [f64; 0] = [];
    let report = plan
        .integrate_into(
            |_: f64| -> Result<[f64; 0], SciOxideError> {
                Err(SciOxideError::external("must not sample"))
            },
            &Linear { sign: 1.0 },
            &mut empty,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 0);
    assert_eq!(
        report.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    let region = OscillatoryRegion::stationary(interval(0.0, 1.0), interval(0.0, 2.0)).unwrap();
    let phase = OscillatoryGeometry::new(interval(0.0, 1.0), 1000.0, &[region]).unwrap();
    let error = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        phase,
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(opts).with_ordinary_panels(Budget::try_from(2).unwrap()),
    )
    .unwrap()
    .integrate(|_: f64| Ok(1.0), &Square)
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::TruncationLimit,
            ..
        }
    ));
}

/// `exp(scale x)`: relative to its right end the phase is nearly stationary at the
/// left end (`g'(0) / g'(1) = exp(-scale)`), which makes degree-24 collocation
/// numerically rank deficient. Polynomial phases, even with a stationary endpoint,
/// keep a full-rank collocation.
struct Steep(f64);
impl<S: RealScalar + Default> OscillatoryPhase<S> for Steep {
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError> {
        let scale = n::<S>(self.0);
        Ok(((scale * x).exp(), scale * (scale * x).exp()))
    }
}
fn steep_plan<S: RealScalar + Default>(scale: f64, upper: f64) -> OscillatoryQuadraturePlan<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let domain = interval(0.0, 1.0);
    let region = OscillatoryRegion::nonstationary(domain, interval(scale, upper)).unwrap();
    OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        OscillatoryGeometry::new(domain, n(1000.0), &[region]).unwrap(),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(options()),
    )
    .unwrap()
}
#[test]
fn rank_deficient_root_collocations_are_detected() {
    fn check<S: RealScalar + Default>(scale: f64)
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let cosines = crate::weighted::moments::cosines::<S>().unwrap();
        let mut derivatives = [S::ZERO; 25];
        let phase = Steep(scale);
        for (j, derivative) in derivatives.iter_mut().enumerate() {
            let x = n::<S>(0.5) + n::<S>(0.5) * cosines[1][j];
            *derivative = OscillatoryPhase::<S>::evaluate(&phase, x).unwrap().1;
        }
        let phases = [n::<S>(1000.0), n::<S>(1000.0) * n::<S>(scale).exp()];
        let mut workspace = super::levin::LevinWorkspace::<S>::new().unwrap();
        let weights = sequential(|ctx| {
            workspace.weights(&cosines, &derivatives, n(0.5), n(1000.0), phases, ctx)
        })
        .unwrap();
        assert!(weights.is_none());
    }
    check::<f32>(60.0);
    check::<f64>(60.0);
}
#[test]
fn collocation_failures_bisect_instead_of_failing_the_call() {
    fn check<S: RealScalar + Default>(scale: f64, upper: f64)
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let calls = Cell::new(0);
        let (value, report) = steep_plan::<S>(scale, upper)
            .integrate(
                |_: S| {
                    calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    Ok(S::ONE)
                },
                &Steep(scale),
            )
            .unwrap();
        // The root collocation fails, so its amplitude is sampled and the region
        // bisected; the call ends with a measured estimate, successful or not.
        assert!(value.is_finite(), "{report:?}");
        assert!(report.subdivisions() > 0, "{report:?}");
        assert!(calls.load(core::sync::atomic::Ordering::Relaxed) > 25);
    }
    check::<f32>(60.0, 1e28);
    check::<f64>(60.0, 1e28);
}
#[test]
fn phase_providers_are_called_once_per_node() {
    // Levin collocates 25 nodes, Filon interpolates 25 nodes and inverts its 23
    // interior ones, and weak oscillation uses 15 ordinary nodes per panel.
    for (method, omega) in [
        (OscillatoryMethod::Levin, 1000.0),
        (OscillatoryMethod::Filon, 1000.0),
        (OscillatoryMethod::Levin, 20.0),
        (OscillatoryMethod::Filon, 1.0),
    ] {
        let phases = Cell::new(0);
        let inverses = Cell::new(0);
        let phase = (
            |x: f64| {
                phases.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok((x, 1.0))
            },
            |_: Interval<f64>, y: f64| {
                inverses.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(y)
            },
        );
        let plan = OscillatoryQuadraturePlan::new(
            method,
            linear_plan(omega, 1.0),
            FourierWeight::Cosine,
            crate::OscillatoryOptions::new(options()),
        )
        .unwrap();
        let (_, report) = plan
            .integrate(|x: f64| Ok(1.0 / (1.0 + x * x)), &phase)
            .unwrap();
        assert!(report.is_success(), "{report:?}");
        let nodes = report.evaluations();
        let phases = phases.load(core::sync::atomic::Ordering::Relaxed);
        let inverses = inverses.load(core::sync::atomic::Ordering::Relaxed);
        assert_eq!(phases, nodes, "{method:?} at {omega}");
        let phase_aware = omega == 1000.0;
        assert_eq!(nodes % if phase_aware { 25 } else { 15 }, 0);
        let expected = if phase_aware && method == OscillatoryMethod::Filon {
            nodes / 25 * 23
        } else {
            0
        };
        assert_eq!(inverses, expected, "{method:?} at {omega}");
    }
}
#[test]
fn filon_rejects_providers_without_an_inverse_before_callbacks() {
    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        linear_plan(1000.0, 1.0),
        FourierWeight::Cosine,
        crate::OscillatoryOptions::new(options()),
    )
    .unwrap();
    let calls = Cell::new(0);
    let mut ws = plan.workspace(1).unwrap();
    let mut output = -5.0;
    let result = plan.integrate_into(
        |_: f64| Ok(1.0),
        &|x: f64| {
            calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok((x, 1.0))
        },
        &mut output,
        &mut ws,
        ExecutionContext::sequential_native(),
    );
    assert!(matches!(
        result.unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert_eq!(output, -5.0);
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
}
#[test]
fn f32_complex_outputs_use_the_same_component_contract() {
    let opts = options::<f32>();
    let l = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Levin,
        linear_plan(1000.0, 1.0),
        FourierWeight::Sine,
        crate::OscillatoryOptions::new(opts.clone()),
    )
    .unwrap();
    let f = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        linear_plan(1000.0, 1.0),
        FourierWeight::Sine,
        crate::OscillatoryOptions::new(opts),
    )
    .unwrap();
    for levin in [true, false] {
        let mut ws = if levin {
            l.workspace(2).unwrap()
        } else {
            f.workspace(2).unwrap()
        };
        let sample = Complex::new(0.0f32, 0.0);
        let mut output = sample;
        let amplitude = |_: f32, y: &mut Complex<f32>| {
            *y = Complex::new(1.0, -2.0);
            Ok::<(), SciOxideError>(())
        };
        let r = sequential(|ctx| {
            if levin {
                l.integrate_into(
                    |point| {
                        let mut lane = sample;
                        scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                            *v = <_ as scioxide_core::RealScalar>::NAN
                        });
                        {
                            let result: Result<(), SciOxideError> = amplitude(point, &mut lane);
                            result?;
                        }
                        Ok(lane)
                    },
                    &Linear { sign: 1.0 },
                    &mut output,
                    &mut ws,
                    &mut *ctx,
                )
            } else {
                f.integrate_into(
                    |point| {
                        let mut lane = sample;
                        scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                            *v = <_ as scioxide_core::RealScalar>::NAN
                        });
                        {
                            let result: Result<(), SciOxideError> = amplitude(point, &mut lane);
                            result?;
                        }
                        Ok(lane)
                    },
                    &Linear { sign: 1.0 },
                    &mut output,
                    &mut ws,
                    &mut *ctx,
                )
            }
        })
        .unwrap();
        let expected = (1.0 - 1000.0f64.cos()) / 1000.0;
        assert!(r.error_components().is_some());
        assert!((f64::from(output.re) - expected).abs() < 6e-4);
        assert!((f64::from(output.im) + 2.0 * expected).abs() < 6e-4);
    }
}

#[test]
fn workspace_requirements_match_reserved_execution_storage() {
    let plan = OscillatoryQuadraturePlan::new(
        OscillatoryMethod::Filon,
        linear_plan(100.0_f64, 1.0),
        FourierWeight::Cosine,
        OscillatoryOptions::new(SubdivisionOptions::default().with_max_subdivisions(Budget::ONE)),
    )
    .unwrap();
    let requirements = plan.workspace_requirements(2).unwrap();
    let workspace = plan.workspace(2).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    let phase = (|x: f64| Ok((x, 1.0)), |_: Interval<f64>, p: f64| Ok(p));
    let mut workspace = workspace;
    let mut out = [0.0; 2];
    let report = plan
        .integrate_into(
            |_: f64| Ok([1.0, 2.0]),
            &phase,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
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
fn interior_jumps_stay_visible_to_levin_and_filon_at_high_frequency() {
    // Exact ∫ H(x - 0.3) (1 + x) sin(omega x) dx over [0, 1].
    let reference = |omega: f64| {
        let primitive =
            |x: f64| -(1.0 + x) * (omega * x).cos() / omega + (omega * x).sin() / (omega * omega);
        primitive(1.0) - primitive(0.3)
    };
    let jump = |x: f64| Ok(if x < 0.3 { 0.0 } else { 1.0 + x });
    for omega in [1e4, 1e5, 1e6] {
        for method in [OscillatoryMethod::Levin, OscillatoryMethod::Filon] {
            let plan = OscillatoryQuadraturePlan::new(
                method,
                linear_plan::<f64>(omega, 1.0),
                FourierWeight::Sine,
                crate::OscillatoryOptions::default(),
            )
            .unwrap();
            let (value, report) = plan.integrate(&jump, &Linear { sign: 1.0 }).unwrap();
            let expected = reference(omega);
            let error = (value - expected).abs();
            assert!(report.evaluations() > 25, "{method:?} {omega}: {report:?}");
            assert!(
                error <= report.error_estimate().unwrap(),
                "{method:?} {omega}: {value} versus {expected} {report:?}"
            );
            if report.is_success() {
                assert!(error <= 1.49e-8 * (1.0 + expected.abs()), "{report:?}");
            }
        }
    }
}

#[test]
fn resolved_smooth_amplitudes_keep_one_levin_or_filon_region() {
    fn check<S: RealScalar + Default>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        for omega in [200.0, 1e3, 1e5] {
            let w = n::<S>(omega).into_f64();
            let e = core::f64::consts::E;
            let expected = (e * (w * w.sin() + w.cos()) - 1.0) / (1.0 + w * w);
            for method in [OscillatoryMethod::Levin, OscillatoryMethod::Filon] {
                let plan = OscillatoryQuadraturePlan::new(
                    method,
                    linear_plan::<S>(omega, 1.0),
                    FourierWeight::Cosine,
                    crate::OscillatoryOptions::default(),
                )
                .unwrap();
                let (value, report) = plan
                    .integrate(|x: S| Ok(x.exp()), &Linear { sign: 1.0 })
                    .unwrap();
                assert!(report.is_success(), "{method:?} {omega}: {report:?}");
                assert_eq!(report.evaluations(), 25, "{method:?} {omega}: {report:?}");
                let rounding = 64.0 * S::EPSILON.into_f64() * (1.0 + w * S::EPSILON.into_f64());
                assert!(
                    (value.into_f64() - expected).abs()
                        <= report.error_estimate().unwrap().into_f64() + rounding,
                    "{method:?} {omega}: {value:?} versus {expected} {report:?}"
                );
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
