//! Real and complex scoped forms share values, quotas and execution failures.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, CapabilityError, Clock, ClockBinding, Complex,
    ConvergenceReason, Deadline, ExecutionContext, ExecutionStopReason, MonotonicInstant,
    NumericScalar, ProblemToken, RealScalar, SciOxideError, SciOxideErrorView, SolveBudgets,
    SolveIdentity, SolveScope, SolveState,
};
use scioxide_special::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Default)]
struct TickClock(AtomicU64);
impl Clock for TickClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(instant(self.0.fetch_add(1, Ordering::Relaxed)))
    }
}
struct CancellingClock<'a> {
    tick: AtomicU64,
    trigger: u64,
    token: &'a CancellationToken,
}
impl Clock for CancellingClock<'_> {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        let tick = self.tick.fetch_add(1, Ordering::Relaxed);
        if tick == self.trigger {
            self.token.cancel();
        }
        Ok(instant(tick))
    }
}
fn instant(tick: u64) -> MonotonicInstant {
    MonotonicInstant::from_duration_since_epoch(Duration::from_nanos(tick))
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1155),
        scioxide_core::algorithm_id!("special", "scoped_test", 1),
        0,
        0,
    )
}
fn stopped<T: core::fmt::Debug>(result: Result<T, SciOxideError>, reason: ExecutionStopReason) {
    let error = result.unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ExecutionStopped { reason: actual, .. } if actual == reason),
        "{error}"
    );
}
fn exhausted<V: NumericScalar>(result: Result<SpecialFunctionResult<V>, SciOxideError>) {
    match result {
        Ok(report) => assert_eq!(
            report.termination(),
            SpecialFunctionTermination::IterationLimit
        ),
        Err(error) => assert!(
            matches!(
                error.view(),
                SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::IterationLimit,
                    ..
                }
            ),
            "{error}"
        ),
    }
}
/// How a scoped form uses memory and the tail of its work.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// Allocation-free, and every admitted unit is needed for success.
    Strict,
    /// Allocates documented recurrence scratch.
    Allocating,
    /// A separately evaluated derivative is omitted when work runs short.
    OptionalDerivative,
}
fn exercise<V: NumericScalar>(
    name: &str,
    shape: Shape,
    detailed: impl Fn(Budget) -> Result<SpecialFunctionResult<V>, SciOxideError>,
    scoped: impl for<'scope, 'cap> Fn(
        Budget,
        SolveScope<'scope, 'cap>,
    ) -> Result<SpecialFunctionResult<V>, SciOxideError>,
) {
    let ample = Budget::try_from(65_536).unwrap();
    let expected = detailed(ample).unwrap();
    assert!(expected.termination().is_success(), "{name}");
    let used = expected.counts().iterations();
    assert!(used > 1, "{name} must exercise iteration");
    let actual = scoped(ample, ExecutionContext::sequential().into()).unwrap();
    assert_eq!(actual, expected, "{name}");
    let a = actual.try_value().unwrap();
    let e = expected.try_value().unwrap();
    assert_eq!(
        a.real_component().into_f64().to_bits(),
        e.real_component().into_f64().to_bits(),
        "{name}"
    );
    assert_eq!(
        a.imaginary_component().into_f64().to_bits(),
        e.imaginary_component().into_f64().to_bits(),
        "{name}"
    );

    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let actual = scoped(ample, (&mut parent).into()).unwrap();
    assert_eq!(actual, expected, "{name}");
    assert_eq!(parent.counts().own().iterations(), 0, "{name}");
    assert_eq!(parent.counts().nested().iterations(), used, "{name}");
    let before = parent.counts().nested().iterations();
    exhausted(scoped(Budget::ONE, (&mut parent).into()));
    let attempted = parent.counts().nested().iterations() - before;
    assert!(attempted <= 1, "{name}: {attempted}");
    exhausted(detailed(Budget::ONE));

    // The same boundary allowance produces the same report in both forms.
    let exact = Budget::try_from(used).unwrap();
    match (
        scoped(exact, ExecutionContext::sequential().into()),
        detailed(exact),
    ) {
        (Ok(left), Ok(right)) => assert_eq!(left, right, "{name}"),
        (Err(left), Err(right)) => assert_eq!(
            publication::exhausted_counts(&left),
            publication::exhausted_counts(&right),
            "{name}"
        ),
        other => panic!("{name}: inconsistent boundary outcomes {other:?}"),
    }
    let short = Budget::try_from(used - 1).unwrap();
    match scoped(short, ExecutionContext::sequential().into()) {
        Ok(partial) if shape == Shape::OptionalDerivative && partial.termination().is_success() => {
            assert!(partial.derivative().is_none(), "{name}");
            assert_eq!(partial, detailed(short).unwrap(), "{name}");
        }
        partial => exhausted(partial),
    }

    let token = CancellationToken::new();
    token.cancel();
    let execution = ExecutionContext::sequential().with_cancellation(&token);
    stopped(
        scoped(ample, execution.into()),
        ExecutionStopReason::Cancelled,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(execution, identity(), &mut budgets)
        .unwrap();
    stopped(
        scoped(ample, (&mut parent).into()),
        ExecutionStopReason::Cancelled,
    );
    assert_eq!(parent.counts().nested().iterations(), 0, "{name}");

    let clock = TickClock::default();
    let binding = ClockBinding::new(&clock);
    let execution = ExecutionContext::sequential()
        .with_deadline(Deadline::at(&binding, instant(0)))
        .unwrap();
    stopped(
        scoped(ample, execution.into()),
        ExecutionStopReason::DeadlineReached,
    );

    // Discover the check count, then expire halfway through the same computation.
    clock.0.store(0, Ordering::Relaxed);
    let execution = ExecutionContext::sequential()
        .with_deadline(Deadline::at(&binding, instant(u64::MAX)))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(execution, identity(), &mut budgets)
        .unwrap();
    let _ = scoped(ample, (&mut parent).into()).unwrap();
    let checks = clock.0.load(Ordering::Relaxed);
    clock.0.store(0, Ordering::Relaxed);
    let execution = ExecutionContext::sequential()
        .with_deadline(Deadline::at(&binding, instant(checks / 2)))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(execution, identity(), &mut budgets)
        .unwrap();
    stopped(
        scoped(ample, (&mut parent).into()),
        ExecutionStopReason::DeadlineReached,
    );
    let attempted = parent.counts().nested().iterations();
    assert!(
        attempted > 0 && attempted < used,
        "{name}: deadline admitted {attempted}/{used} steps"
    );
    let token = CancellationToken::new();
    let clock = CancellingClock {
        tick: AtomicU64::new(0),
        trigger: checks / 2,
        token: &token,
    };
    let binding = ClockBinding::new(&clock);
    let execution = ExecutionContext::sequential()
        .with_cancellation(&token)
        .with_deadline(Deadline::at(&binding, instant(u64::MAX)))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut parent = state
        .begin_solve(execution, identity(), &mut budgets)
        .unwrap();
    stopped(
        scoped(ample, (&mut parent).into()),
        ExecutionStopReason::Cancelled,
    );
    assert!(parent.counts().nested().iterations() > 0, "{name}");
}

macro_rules! check {
    ($plain:ident, $scoped:ident, $options:ident, $($arg:expr),+) => {
        exercise(stringify!($scoped), Shape::Strict,
            |budget| $plain($($arg,)+ $options::default().with_iteration_budget(budget)),
            |budget, scope| $scoped($($arg,)+ $options::default().with_iteration_budget(budget), scope));
    };
}

macro_rules! check_value_first {
    ($plain:ident, $scoped:ident, $options:ident, $($arg:expr),+) => {
        exercise(stringify!($scoped), Shape::OptionalDerivative,
            |budget| $plain($($arg,)+ $options::default().with_iteration_budget(budget)),
            |budget, scope| $scoped($($arg,)+ $options::default().with_iteration_budget(budget), scope));
    };
}

#[test]
fn prepared_wigner_scopes_account_work_and_stop_mid_sum() {
    use std::cell::RefCell;
    let plan = AngularPlan::new(AngularMomentum::new(20).unwrap(), Default::default()).unwrap();
    let scratch = RefCell::new(plan.workspace().unwrap());
    let momentum = |j| AngularMomentum::new(j).unwrap();
    for kind in [3, 6, 9, 90] {
        let evaluate = |scope: SolveScope<'_, '_>| {
            let mut scratch = scratch.borrow_mut();
            match kind {
                3 => plan.wigner_3j_scoped::<f64>(
                    [4, -6, 2].map(|m| AngularState::new(20, m).unwrap()),
                    &mut scratch,
                    scope,
                ),
                6 => plan.wigner_6j_scoped::<f64>([momentum(20); 6], &mut scratch, scope),
                _ => plan.wigner_9j_scoped::<f64>(
                    [
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        20,
                        if kind == 90 { 0 } else { 20 },
                    ]
                    .map(momentum),
                    &mut scratch,
                    scope,
                ),
            }
        };
        let expected = evaluate(ExecutionContext::sequential().into()).unwrap();
        let clock = TickClock::default();
        let binding = ClockBinding::new(&clock);
        let execution = ExecutionContext::sequential()
            .with_deadline(Deadline::at(&binding, instant(u64::MAX)))
            .unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut parent = state
            .begin_solve(execution, identity(), &mut budgets)
            .unwrap();
        let actual = evaluate((&mut parent).into()).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            parent.counts().nested().iterations(),
            actual.counts().iterations()
        );
        let checks = clock.0.load(Ordering::Relaxed);
        clock.0.store(0, Ordering::Relaxed);
        let execution = ExecutionContext::sequential()
            .with_deadline(Deadline::at(&binding, instant(checks / 2)))
            .unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut parent = state
            .begin_solve(execution, identity(), &mut budgets)
            .unwrap();
        stopped(
            evaluate((&mut parent).into()),
            ExecutionStopReason::DeadlineReached,
        );
        assert!(parent.counts().nested().iterations() > 0);
        assert!(parent.counts().nested().iterations() < expected.counts().iterations());
        let token = CancellationToken::new();
        let clock = CancellingClock {
            tick: AtomicU64::new(0),
            trigger: checks / 2,
            token: &token,
        };
        let binding = ClockBinding::new(&clock);
        let execution = ExecutionContext::sequential()
            .with_cancellation(&token)
            .with_deadline(Deadline::at(&binding, instant(u64::MAX)))
            .unwrap();
        stopped(evaluate(execution.into()), ExecutionStopReason::Cancelled);
        assert_eq!(
            evaluate(ExecutionContext::sequential().into()).unwrap(),
            expected
        );
    }
}

#[test]
fn complex_carlson_and_pi_admit_each_step_with_reused_storage() {
    for scale in [1.0_f64, 2.0_f64.powi(40), 2.0_f64.powi(120)] {
        let x = Complex::new(0.5, 0.25) * scale;
        let y = Complex::new(2.0, -0.5) * scale;
        let z = Complex::new(3.0, 1.0) * scale;
        let p = Complex::new(0.25, -0.125) * scale;
        check!(
            carlson_rc_with_options,
            carlson_rc_scoped,
            EllipticOptions,
            x,
            y
        );
        check!(
            carlson_rf_with_options,
            carlson_rf_scoped,
            EllipticOptions,
            x,
            y,
            z
        );
        check!(
            carlson_rd_with_options,
            carlson_rd_scoped,
            EllipticOptions,
            x,
            y,
            z
        );
        check!(
            carlson_rj_with_options,
            carlson_rj_scoped,
            EllipticOptions,
            x,
            y,
            z,
            p
        );
    }
    check!(
        elliptic_pi_with_options,
        elliptic_pi_scoped,
        EllipticOptions,
        Complex::new(0.25_f64, 0.125),
        Complex::new(0.5, -0.25)
    );
}

macro_rules! family_cases {
    ($a:expr, $b:expr, $x:expr, $m:expr) => {
        check!(
            bessel_j_derivative_with_options,
            bessel_j_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            bessel_y_derivative_with_options,
            bessel_y_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            bessel_i_derivative_with_options,
            bessel_i_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            bessel_k_derivative_with_options,
            bessel_k_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            bessel_i_scaled_derivative_with_options,
            bessel_i_scaled_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            bessel_k_scaled_derivative_with_options,
            bessel_k_scaled_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            hankel_1_derivative_with_options,
            hankel_1_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check!(
            hankel_2_derivative_with_options,
            hankel_2_derivative_scoped,
            BesselOptions,
            $m,
            $x,
            2
        );
        check_value_first!(
            hyp0f1_with_options,
            hyp0f1_scoped,
            HypergeometricOptions,
            $b,
            $x
        );
        check_value_first!(
            hyp1f1_with_options,
            hyp1f1_scoped,
            HypergeometricOptions,
            $a,
            $b,
            $x
        );
        check_value_first!(
            hyp2f1_with_options,
            hyp2f1_scoped,
            HypergeometricOptions,
            $a,
            $b,
            $x,
            $m
        );
        check_value_first!(
            hyp2f1_regularized_with_options,
            hyp2f1_regularized_scoped,
            HypergeometricOptions,
            $a,
            $b,
            $x,
            $m
        );
        check_value_first!(
            hyp2f0_with_options,
            hyp2f0_scoped,
            HypergeometricOptions,
            $a,
            $b,
            -$m
        );
        check_value_first!(
            hyperu_with_options,
            hyperu_scoped,
            HypergeometricOptions,
            $a,
            $m,
            $x
        );
        check!(gamma_p_with_options, gamma_p_scoped, GammaOptions, $a, $x);
        check!(gamma_q_with_options, gamma_q_scoped, GammaOptions, $a, $x);
        check!(
            gamma_lower_with_options,
            gamma_lower_scoped,
            GammaOptions,
            $a,
            $x
        );
        check!(
            gamma_upper_with_options,
            gamma_upper_scoped,
            GammaOptions,
            $a,
            $x
        );
        check!(
            beta_inc_with_options,
            beta_inc_scoped,
            BetaOptions,
            $a,
            $b,
            $m
        );
        check!(
            polygamma_with_options,
            polygamma_scoped,
            GammaOptions,
            2,
            $x
        );
        check!(
            riemann_zeta_with_options,
            riemann_zeta_scoped,
            ZetaOptions,
            $a
        );
        check!(
            dirichlet_eta_with_options,
            dirichlet_eta_scoped,
            ZetaOptions,
            $a
        );
        check!(
            hurwitz_zeta_with_options,
            hurwitz_zeta_scoped,
            ZetaOptions,
            $a,
            $x
        );
        // Real complete K/E are fixed cost and covered by the fixed-scope
        // test below. These non-real inputs still exercise continuation work.
        if $m.imaginary_component().into_f64() != 0.0 {
            check!(
                elliptic_k_with_options,
                elliptic_k_scoped,
                EllipticOptions,
                -$m
            );
            check!(
                elliptic_e_with_options,
                elliptic_e_scoped,
                EllipticOptions,
                -$m
            );
        }
        check!(
            elliptic_f_with_options,
            elliptic_f_scoped,
            EllipticOptions,
            $x,
            $m
        );
        check!(
            elliptic_e_inc_with_options,
            elliptic_e_inc_scoped,
            EllipticOptions,
            $x,
            $m
        );
        check!(
            elliptic_pi_with_options,
            elliptic_pi_scoped,
            EllipticOptions,
            $m,
            -$m
        );
        check!(
            fresnel_c_with_options,
            fresnel_c_scoped,
            IntegralOptions,
            $x
        );
        check!(
            fresnel_s_with_options,
            fresnel_s_scoped,
            IntegralOptions,
            $x
        );
        check!(
            bessel_j_with_options,
            bessel_j_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            bessel_y_with_options,
            bessel_y_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            bessel_i_with_options,
            bessel_i_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            bessel_k_with_options,
            bessel_k_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            bessel_i_scaled_with_options,
            bessel_i_scaled_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            bessel_k_scaled_with_options,
            bessel_k_scaled_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            hankel_1_with_options,
            hankel_1_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            hankel_2_with_options,
            hankel_2_scoped,
            BesselOptions,
            $m,
            $x
        );
        check!(
            spherical_jn_with_options,
            spherical_jn_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_yn_with_options,
            spherical_yn_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_in_with_options,
            spherical_in_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_kn_with_options,
            spherical_kn_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_in_scaled_with_options,
            spherical_in_scaled_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_kn_scaled_with_options,
            spherical_kn_scaled_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_hankel_1_with_options,
            spherical_hankel_1_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
        check!(
            spherical_hankel_2_with_options,
            spherical_hankel_2_scoped,
            SphericalBesselOptions,
            3,
            $x
        );
    };
}

#[test]
fn fixed_complete_elliptic_scopes() {
    macro_rules! check_fixed {
        ($scalar:ty) => {{
            let options = EllipticOptions::default().with_iteration_budget(Budget::ONE);
            for m in [
                0.0 as $scalar,
                -0.0,
                -0.5,
                -1e30,
                -<$scalar>::MAX,
                0.3,
                0.9,
                (1.0 as $scalar).next_down(),
                1.0,
            ] {
                let expected = [
                    elliptic_k_with_options(m, options).unwrap(),
                    elliptic_e_with_options(m, options).unwrap(),
                ];
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
                let mut parent = state
                    .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                    .unwrap();
                let actual = [
                    elliptic_k_scoped(m, options, &mut parent).unwrap(),
                    elliptic_e_scoped(m, options, &mut parent).unwrap(),
                ];
                assert_eq!(actual, expected);
                assert_eq!(parent.counts().nested().iterations(), 0);
                for (report, plain) in actual
                    .into_iter()
                    .zip([elliptic_k(m).unwrap(), elliptic_e(m).unwrap()])
                {
                    assert!(report.termination().is_success());
                    assert_eq!(report.counts().iterations(), 0);
                    assert_eq!(report.try_value().unwrap().to_bits(), plain.to_bits());
                }
                let token = CancellationToken::new();
                token.cancel();
                let execution = ExecutionContext::sequential().with_cancellation(&token);
                stopped(
                    elliptic_k_scoped(m, options, execution),
                    ExecutionStopReason::Cancelled,
                );
                stopped(
                    elliptic_e_scoped(m, options, execution),
                    ExecutionStopReason::Cancelled,
                );
            }
        }};
    }
    check_fixed!(f64);
    check_fixed!(f32);
}

#[test]
fn real_f64_scoped_forms() {
    let a = 2.25_f64;
    let b = 3.25_f64;
    let x = 1.25_f64;
    let m = 0.3_f64;
    family_cases!(a, b, x, m);
}

#[test]
fn negative_complex_zeta_continuation_scoped_forms() {
    for (s, a) in [
        (Complex::new(-20.5, 0.0), Complex::new(10.0, 0.0)),
        (Complex::new(-2.5, 1.0), Complex::new(10.0, 0.0)),
        (Complex::new(-5.5, 0.25), Complex::new(1.25, 0.25)),
    ] {
        exercise(
            "negative complex Hurwitz",
            Shape::Strict,
            |budget| hurwitz_zeta_with_options(s, a, ZetaOptions::new(budget)),
            |budget, scope| hurwitz_zeta_scoped(s, a, ZetaOptions::new(budget), scope),
        );
    }
    let s = Complex::new(-150.0, 1e-10);
    exercise(
        "complex zeta reflection",
        Shape::Strict,
        |budget| riemann_zeta_with_options(s, ZetaOptions::new(budget)),
        |budget, scope| riemann_zeta_scoped(s, ZetaOptions::new(budget), scope),
    );
}

#[test]
fn complex_f64_scoped_forms() {
    let a = Complex::new(2.25_f64, 0.125);
    let b = Complex::new(3.25_f64, 0.125);
    let x = Complex::new(1.25_f64, 0.125);
    let m = Complex::new(0.3_f64, 0.125);
    family_cases!(a, b, x, m);
}

#[test]
fn real_f32_scoped_forms() {
    let a = 2.25_f32;
    let b = 3.25_f32;
    let x = 1.25_f32;
    let m = 0.3_f32;
    family_cases!(a, b, x, m);
}

#[test]
fn complex_f32_scoped_forms() {
    let a = Complex::new(2.25_f32, 0.125);
    let b = Complex::new(3.25_f32, 0.125);
    let x = Complex::new(1.25_f32, 0.125);
    let m = Complex::new(0.3_f32, 0.125);
    family_cases!(a, b, x, m);
}

#[test]
fn real_regime_shortcuts_and_composed_reductions_share_work() {
    macro_rules! same {
        ($detailed:ident, $scoped:ident, $options:ident, $($arg:expr),+) => {{
            let expected = $detailed($($arg,)+ $options::default()).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut parent = state.begin_solve(ExecutionContext::sequential(), identity(), &mut budgets).unwrap();
            let actual = $scoped($($arg,)+ $options::default(), &mut parent).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(parent.counts().nested().iterations(), actual.counts().iterations());
        }};
    }
    for (a, x) in [
        (1.0_f64, 2.0),
        (0.5, 2.0),
        (6.0, 9.0),
        (5.5, 9.0),
        (1000.0, 1000.0),
        (0.125, 0.5),
        (2.25, 30.0),
        (2.25, 0.0),
    ] {
        same!(gamma_p_with_options, gamma_p_scoped, GammaOptions, a, x);
        same!(gamma_q_with_options, gamma_q_scoped, GammaOptions, a, x);
        same!(
            gamma_lower_with_options,
            gamma_lower_scoped,
            GammaOptions,
            a,
            x
        );
        same!(
            gamma_upper_with_options,
            gamma_upper_scoped,
            GammaOptions,
            a,
            x
        );
    }
    for (a, b, x) in [
        (1.0_f64, 1.0, 0.3),
        (2.0, 3.0, 0.3),
        (5.0, 5.0, 0.5),
        (2.25, 3.25, 0.99),
        (2.25, 3.25, 0.0),
        (2.25, 3.25, 1.0),
    ] {
        same!(beta_inc_with_options, beta_inc_scoped, BetaOptions, a, b, x);
    }
    for x in [-2.25_f64, -100000.5, 100.0] {
        same!(polygamma_with_options, polygamma_scoped, GammaOptions, 2, x);
    }
    for x in [-20.25_f64, -0.25, 0.0, 2.0, 30.0] {
        same!(
            riemann_zeta_with_options,
            riemann_zeta_scoped,
            ZetaOptions,
            x
        );
        same!(
            dirichlet_eta_with_options,
            dirichlet_eta_scoped,
            ZetaOptions,
            x
        );
    }
    for s in [-1.25_f64, 0.0, 2.25] {
        same!(
            hurwitz_zeta_with_options,
            hurwitz_zeta_scoped,
            ZetaOptions,
            s,
            1.25
        );
    }
    for m in [-4.0_f64, 0.0, 0.3, 1.0 - 1e-12] {
        same!(
            elliptic_k_with_options,
            elliptic_k_scoped,
            EllipticOptions,
            m
        );
        same!(
            elliptic_e_with_options,
            elliptic_e_scoped,
            EllipticOptions,
            m
        );
        same!(
            elliptic_f_with_options,
            elliptic_f_scoped,
            EllipticOptions,
            7.25,
            m
        );
        same!(
            elliptic_e_inc_with_options,
            elliptic_e_inc_scoped,
            EllipticOptions,
            7.25,
            m
        );
        same!(
            elliptic_pi_with_options,
            elliptic_pi_scoped,
            EllipticOptions,
            0.125,
            m
        );
    }
    for x in [-20.0_f64, -0.0, 0.0, 1.25, 20.0] {
        same!(fresnel_c_with_options, fresnel_c_scoped, IntegralOptions, x);
        same!(fresnel_s_with_options, fresnel_s_scoped, IntegralOptions, x);
    }
}

#[test]
fn prepared_spectral_recurrences_observe_mid_execution_stops() {
    fn check(run: impl for<'a> Fn(ExecutionContext<'a>) -> Result<usize, SciOxideError>) {
        let clock = TickClock::default();
        let binding = ClockBinding::new(&clock);
        let execution = ExecutionContext::sequential()
            .with_deadline(Deadline::at(&binding, instant(u64::MAX)))
            .unwrap();
        assert!(run(execution).unwrap() > 1);
        let checks = clock.0.load(Ordering::Relaxed);
        for limit in [0, checks / 2] {
            clock.0.store(0, Ordering::Relaxed);
            let execution = ExecutionContext::sequential()
                .with_deadline(Deadline::at(&binding, instant(limit)))
                .unwrap();
            stopped(run(execution), ExecutionStopReason::DeadlineReached);
        }
        let token = CancellationToken::new();
        let clock = CancellingClock {
            tick: AtomicU64::new(0),
            trigger: checks / 2,
            token: &token,
        };
        let binding = ClockBinding::new(&clock);
        let execution = ExecutionContext::sequential()
            .with_cancellation(&token)
            .with_deadline(Deadline::at(&binding, instant(u64::MAX)))
            .unwrap();
        stopped(run(execution), ExecutionStopReason::Cancelled);
    }
    let mathieu = MathieuPlan::new(
        3,
        MathieuParity::Even,
        1.0_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let spheroidal = SpheroidalPlan::new(
        SpheroidalKind::Prolate,
        1,
        2,
        1.0_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    check(|scope| {
        mathieu
            .angular_scoped(0.3_f64, scope)
            .map(|r| r.counts().iterations())
    });
    check(|scope| {
        mathieu
            .angular_scoped(Complex::new(0.3_f64, 0.125), scope)
            .map(|r| r.counts().iterations())
    });
    check(|scope| {
        spheroidal
            .angular_scoped(0.3_f64, scope)
            .map(|r| r.counts().iterations())
    });
    check(|scope| {
        spheroidal
            .angular_scoped(Complex::new(0.3_f64, 0.125), scope)
            .map(|r| r.counts().iterations())
    });
}

macro_rules! integral_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        for value in [x(0.75), x(1.5), x(6.5)] {
            check!(
                expint_e2_with_options,
                expint_e2_scoped,
                IntegralOptions,
                value
            );
            check!(
                expint_e2_scaled_with_options,
                expint_e2_scaled_scoped,
                IntegralOptions,
                value
            );
            check!(
                expint_en_with_options,
                expint_en_scoped,
                IntegralOptions,
                5,
                value
            );
            check!(
                expint_en_scaled_with_options,
                expint_en_scaled_scoped,
                IntegralOptions,
                5,
                value
            );
        }
        for value in [x(0.75), x(2.5), x(50.0)] {
            check!(si_with_options, si_scoped, IntegralOptions, value);
            check!(ci_with_options, ci_scoped, IntegralOptions, value);
            check!(shi_with_options, shi_scoped, IntegralOptions, value);
            check!(chi_with_options, chi_scoped, IntegralOptions, value);
            check!(
                shi_scaled_with_options,
                shi_scaled_scoped,
                IntegralOptions,
                value
            );
            check!(
                chi_scaled_with_options,
                chi_scaled_scoped,
                IntegralOptions,
                value
            );
        }
    }};
}

#[test]
fn integral_scoped_forms() {
    integral_cases!(f64);
    integral_cases!(f32);
}

macro_rules! thermal_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        for value in [x(1.5), x(6.5)] {
            check!(debye_1_with_options, debye_1_scoped, DebyeOptions, value);
            check!(debye_2_with_options, debye_2_scoped, DebyeOptions, value);
            check!(debye_3_with_options, debye_3_scoped, DebyeOptions, value);
            check!(debye_4_with_options, debye_4_scoped, DebyeOptions, value);
            check!(debye_5_with_options, debye_5_scoped, DebyeOptions, value);
            check!(debye_6_with_options, debye_6_scoped, DebyeOptions, value);
            check!(
                transport_2_with_options,
                transport_2_scoped,
                TransportOptions,
                value
            );
            check!(
                transport_3_with_options,
                transport_3_scoped,
                TransportOptions,
                value
            );
            check!(
                transport_4_with_options,
                transport_4_scoped,
                TransportOptions,
                value
            );
            check!(
                transport_5_with_options,
                transport_5_scoped,
                TransportOptions,
                value
            );
        }
        for value in [x(-2.5), x(5.0), x(40.0)] {
            check!(
                fermi_dirac_1_with_options,
                fermi_dirac_1_scoped,
                FermiDiracOptions,
                value
            );
            check!(
                fermi_dirac_2_with_options,
                fermi_dirac_2_scoped,
                FermiDiracOptions,
                value
            );
            check!(
                fermi_dirac_mhalf_with_options,
                fermi_dirac_mhalf_scoped,
                FermiDiracOptions,
                value
            );
            check!(
                fermi_dirac_half_with_options,
                fermi_dirac_half_scoped,
                FermiDiracOptions,
                value
            );
            check!(
                fermi_dirac_3half_with_options,
                fermi_dirac_3half_scoped,
                FermiDiracOptions,
                value
            );
            for order in [-5, 3] {
                check!(
                    fermi_dirac_int_with_options,
                    fermi_dirac_int_scoped,
                    FermiDiracOptions,
                    order,
                    value
                );
            }
        }
    }};
}

#[test]
fn thermal_scoped_forms() {
    thermal_cases!(f64);
    thermal_cases!(f32);
}

macro_rules! orthogonal_zeta_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        for value in [x(-0.3), x(0.7), x(1.7)] {
            check!(
                legendre_p_with_options,
                legendre_p_scoped,
                LegendreOptions,
                40,
                value
            );
        }
        for value in [x(-0.3), x(0.7)] {
            check!(
                legendre_q_with_options,
                legendre_q_scoped,
                LegendreOptions,
                40,
                value
            );
        }
        for (s, q) in [(x(2.5), x(1.25)), (x(1.5), x(30.0))] {
            check!(
                hurwitz_zeta_log_with_options,
                hurwitz_zeta_log_scoped,
                ZetaOptions,
                s,
                q
            );
        }
    }};
}

#[test]
fn orthogonal_and_zeta_scoped_forms() {
    orthogonal_zeta_cases!(f64);
    orthogonal_zeta_cases!(f32);
}

macro_rules! probability_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        let (a, b) = (x(2.25), x(3.25));
        check!(
            gamma_log_p_with_options,
            gamma_log_p_scoped,
            GammaOptions,
            a,
            x(1.25)
        );
        check!(
            gamma_log_q_with_options,
            gamma_log_q_scoped,
            GammaOptions,
            a,
            x(1.25)
        );
        check!(
            beta_inc_q_with_options,
            beta_inc_q_scoped,
            BetaOptions,
            a,
            b,
            x(0.3)
        );
        check!(
            beta_inc_log_with_options,
            beta_inc_log_scoped,
            BetaOptions,
            a,
            b,
            x(0.3)
        );
        check!(
            beta_inc_log_q_with_options,
            beta_inc_log_q_scoped,
            BetaOptions,
            a,
            b,
            x(0.3)
        );
        check!(
            gamma_p_inv_with_options,
            gamma_p_inv_scoped,
            GammaOptions,
            a,
            x(0.3)
        );
        check!(
            gamma_q_inv_with_options,
            gamma_q_inv_scoped,
            GammaOptions,
            a,
            x(0.3)
        );
        check!(
            gamma_p_inv_log_with_options,
            gamma_p_inv_log_scoped,
            GammaOptions,
            a,
            x(-1.2)
        );
        check!(
            gamma_q_inv_log_with_options,
            gamma_q_inv_log_scoped,
            GammaOptions,
            a,
            x(-1.2)
        );
        check!(
            beta_inc_inv_with_options,
            beta_inc_inv_scoped,
            BetaOptions,
            a,
            b,
            x(0.3)
        );
        check!(
            beta_inc_q_inv_with_options,
            beta_inc_q_inv_scoped,
            BetaOptions,
            a,
            b,
            x(0.3)
        );
        check!(
            beta_inc_inv_log_with_options,
            beta_inc_inv_log_scoped,
            BetaOptions,
            a,
            b,
            x(-1.2)
        );
        check!(
            beta_inc_q_inv_log_with_options,
            beta_inc_q_inv_log_scoped,
            BetaOptions,
            a,
            b,
            x(-1.2)
        );
        check!(
            normal_inv_log_cdf_with_options,
            normal_inv_log_cdf_scoped,
            ProbabilityOptions,
            x(-50.0)
        );
        check!(
            normal_inv_log_sf_with_options,
            normal_inv_log_sf_scoped,
            ProbabilityOptions,
            x(-50.0)
        );
        for (value, sigma, gamma) in [(x(0.5), x(1.0), x(0.5)), (x(7.5), x(1.0), x(0.25))] {
            check!(
                voigt_profile_with_options,
                voigt_profile_scoped,
                ProbabilityOptions,
                value,
                sigma,
                gamma
            );
            check!(
                voigt_complex_with_options,
                voigt_complex_scoped,
                ProbabilityOptions,
                value,
                sigma,
                gamma
            );
        }
    }};
}

#[test]
fn probability_scoped_forms() {
    probability_cases!(f64);
    probability_cases!(f32);
}

macro_rules! check_allocating {
    ($plain:ident, $scoped:ident, $options:ident, $scalar:ty; $($arg:expr),+) => {
        exercise(stringify!($scoped), Shape::Allocating,
            |budget| $plain::<$scalar>($($arg,)+ $options::default().with_iteration_budget(budget)),
            |budget, scope| $scoped::<$scalar>($($arg,)+ $options::default().with_iteration_budget(budget), scope));
    };
}

macro_rules! combinatorial_cases {
    ($a:expr, $z:expr) => {{
        check!(
            generalized_binomial_with_options,
            generalized_binomial_scoped,
            CombinatorialOptions,
            $z,
            30
        );
    }};
}

#[test]
fn combinatorial_scoped_forms() {
    combinatorial_cases!(2.25_f64, 7.25_f64);
    combinatorial_cases!(2.25_f32, 7.25_f32);
    combinatorial_cases!(Complex::new(2.25_f64, 0.5), Complex::new(7.25_f64, -0.5));
    combinatorial_cases!(Complex::new(2.25_f32, 0.5), Complex::new(7.25_f32, -0.5));
    for (n, k) in [(30, 10), (12, 5)] {
        check_allocating!(
            stirling_first_kind_with_options,
            stirling_first_kind_scoped,
            CombinatorialOptions,
            f64;
            n,
            k
        );
        check_allocating!(
            stirling_second_kind_with_options,
            stirling_second_kind_scoped,
            CombinatorialOptions,
            f32;
            n,
            k
        );
    }
}

macro_rules! bessel_integral_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        for value in [x(1.25), x(20.0), x(40.0)] {
            check!(
                bessel_j0_integral_with_options,
                bessel_j0_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_y0_integral_with_options,
                bessel_y0_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_i0_integral_with_options,
                bessel_i0_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_k0_integral_with_options,
                bessel_k0_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_j0_complement_integral_with_options,
                bessel_j0_complement_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_y0_tail_integral_with_options,
                bessel_y0_tail_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_i0_remainder_integral_with_options,
                bessel_i0_remainder_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                bessel_k0_tail_integral_with_options,
                bessel_k0_tail_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                struve_h0_integral_with_options,
                struve_h0_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                struve_h0_tail_integral_with_options,
                struve_h0_tail_integral_scoped,
                BesselIntegralOptions,
                value
            );
            check!(
                struve_l0_integral_with_options,
                struve_l0_integral_scoped,
                BesselIntegralOptions,
                value
            );
        }
        check!(
            bessel_lambda_with_options,
            bessel_lambda_scoped,
            BesselOptions,
            x(1.5),
            x(2.5)
        );
        for (order, value) in [(x(2.5), x(1.25)), (x(40.0), x(3.0))] {
            check!(
                bessel_k_log_with_options,
                bessel_k_log_scoped,
                BesselOptions,
                order,
                value
            );
        }
        for (a, exponent, order) in [(x(1.25), x(0.5), x(1.5)), (x(2.0), x(0.0), x(0.0))] {
            check!(
                bessel_weighted_integral_with_options,
                bessel_weighted_integral_scoped,
                BesselIntegralOptions,
                a,
                exponent,
                order
            );
        }
    }};
}

#[test]
fn bessel_integral_scoped_forms() {
    bessel_integral_cases!(f64);
    bessel_integral_cases!(f32);
}

macro_rules! kelvin_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        for value in [x(2.5), x(12.0)] {
            check!(ber_with_options, ber_scoped, KelvinOptions, value);
            check!(bei_with_options, bei_scoped, KelvinOptions, value);
            check!(ker_with_options, ker_scoped, KelvinOptions, value);
            check!(kei_with_options, kei_scoped, KelvinOptions, value);
            check!(berp_with_options, berp_scoped, KelvinOptions, value);
            check!(beip_with_options, beip_scoped, KelvinOptions, value);
            check!(kerp_with_options, kerp_scoped, KelvinOptions, value);
            check!(keip_with_options, keip_scoped, KelvinOptions, value);
            let options = KelvinOptions::default();
            let expected = kelvin_with_options(value, options).unwrap();
            let total: usize = expected.iter().map(|r| r.counts().iterations()).sum();
            assert!(total > 1);
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut parent = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            let actual = kelvin_scoped(value, options, &mut parent).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(parent.counts().nested().iterations(), total);
            let short = options.with_iteration_budget(Budget::ONE);
            let error = kelvin_scoped(value, short, ExecutionContext::sequential()).unwrap_err();
            let expected = kelvin_with_options(value, short).unwrap_err();
            assert_eq!(
                publication::exhausted_counts(&error),
                publication::exhausted_counts(&expected)
            );
            let token = CancellationToken::new();
            token.cancel();
            stopped(
                kelvin_scoped(
                    value,
                    options,
                    ExecutionContext::sequential().with_cancellation(&token),
                ),
                ExecutionStopReason::Cancelled,
            );
        }
    }};
}

#[test]
fn kelvin_scoped_forms() {
    kelvin_cases!(f64);
    kelvin_cases!(f32);
}

macro_rules! cylinder_cases {
    ($scalar:ty) => {{
        let x = |value: f64| value as $scalar;
        for (order, value) in [(x(0.75), x(1.25)), (x(3.5), x(4.5))] {
            check!(
                parabolic_d_with_options,
                parabolic_d_scoped,
                ParabolicCylinderOptions,
                order,
                value
            );
            check!(
                parabolic_v_with_options,
                parabolic_v_scoped,
                ParabolicCylinderOptions,
                order,
                value
            );
            check!(
                parabolic_w_with_options,
                parabolic_w_scoped,
                ParabolicCylinderOptions,
                order,
                value
            );
        }
        check!(
            parabolic_d_with_options,
            parabolic_d_scoped,
            ParabolicCylinderOptions,
            x(0.5),
            x(10000.0)
        );
        check!(
            parabolic_v_with_options,
            parabolic_v_scoped,
            ParabolicCylinderOptions,
            x(0.5),
            x(-10000.0)
        );
        check!(
            parabolic_d_complex_with_options,
            parabolic_d_complex_scoped,
            ParabolicCylinderOptions,
            5,
            Complex::new(x(1.25), x(0.5))
        );
        check!(
            legendre_h3d_with_options,
            legendre_h3d_scoped,
            SphericalBesselOptions,
            3,
            x(1.5),
            x(0.75)
        );
        check!(
            riccati_jn_with_options,
            riccati_jn_scoped,
            SphericalBesselOptions,
            5,
            x(2.5)
        );
        check!(
            riccati_yn_with_options,
            riccati_yn_scoped,
            SphericalBesselOptions,
            5,
            x(2.5)
        );
        for (order, value) in [(x(1.5), x(2.5)), (x(0.5), x(30.0))] {
            check!(
                struve_with_options,
                struve_scoped,
                StruveOptions,
                order,
                value
            );
            check!(
                modstruve_with_options,
                modstruve_scoped,
                StruveOptions,
                order,
                value
            );
        }
    }};
}

#[test]
fn cylinder_scoped_forms() {
    cylinder_cases!(f64);
    cylinder_cases!(f32);
}

#[test]
fn special_scoped_forms_accept_real_and_complex_scalars() -> Result<(), scioxide_core::SciOxideError>
{
    macro_rules! both {
        ($function:ident, $options:ident, $($arg:expr),+) => {
            let _ = $function($($arg,)+ $options::default(), ExecutionContext::sequential())?;
            let _ = $function($(Complex::new($arg, 0.0),)+ $options::default(), ExecutionContext::sequential())?;
        };
    }
    both!(gamma_p_scoped, GammaOptions, 2.25_f64, 1.25);
    both!(beta_inc_scoped, BetaOptions, 2.25_f64, 3.25, 0.3);
    both!(hurwitz_zeta_scoped, ZetaOptions, 2.25_f64, 1.25);
    both!(elliptic_f_scoped, EllipticOptions, 1.25_f64, 0.3);
    both!(fresnel_c_scoped, IntegralOptions, 1.25_f64);
    both!(bessel_j_scoped, BesselOptions, 0.3_f64, 1.25);
    both!(hankel_1_scoped, BesselOptions, 0.3_f64, 1.25);
    both!(hyp1f1_scoped, HypergeometricOptions, 0.75_f64, 1.25, 0.5);
    let _ = bessel_j_derivative_scoped(
        0.3_f64,
        1.25,
        2,
        BesselOptions::default(),
        ExecutionContext::sequential(),
    )?;
    let _ = bessel_j_derivative_scoped(
        Complex::new(0.3_f64, 0.0),
        Complex::new(1.25, 0.0),
        2,
        BesselOptions::default(),
        ExecutionContext::sequential(),
    )?;
    let _ = polygamma_scoped(
        2,
        1.25_f64,
        GammaOptions::default(),
        ExecutionContext::sequential(),
    )?;
    let _ = spherical_jn_scoped(
        3,
        1.25_f64,
        SphericalBesselOptions::default(),
        ExecutionContext::sequential(),
    )?;
    let _ = spherical_hankel_1_scoped(
        3,
        1.25_f64,
        SphericalBesselOptions::default(),
        ExecutionContext::sequential(),
    )?;
    Ok(())
}

#[test]
fn plan_less_pfq_scoped_form() {
    fn cases<T: NumericScalar>(a: [T; 3], b: [T; 2], z: T) {
        exercise(
            "hyp_pfq_scoped",
            Shape::Allocating,
            |budget| {
                hyp_pfq_with_options(
                    &a,
                    &b,
                    z,
                    HypergeometricOptions::default().with_iteration_budget(budget),
                )
            },
            |budget, scope| {
                hyp_pfq_scoped(
                    &a,
                    &b,
                    z,
                    HypergeometricOptions::default().with_iteration_budget(budget),
                    scope,
                )
            },
        );
    }
    cases([0.5_f64, 0.75, 1.25], [1.5, 2.0], 0.5);
    cases([0.5_f32, 0.75, 1.25], [1.5, 2.0], 0.5);
    let c = |re: f64, im: f64| Complex::new(re, im);
    cases(
        [c(0.5, 0.25), c(0.75, 0.0), c(1.25, 0.0)],
        [c(1.5, 0.0), c(2.0, 0.0)],
        c(0.25, 0.25),
    );
}
