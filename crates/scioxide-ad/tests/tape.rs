//! Tapes: rules, capacity protocol, failures, products, patterns and allocation.
#![allow(clippy::unwrap_used, clippy::panic, clippy::needless_range_loop)]

mod support;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use scioxide_ad::{
    BatchFunction, BatchOptions, BatchOutputs, BatchPlan, BatchReduction, CheckpointSchedule,
    DerivativeMode, Dual, ExternalRule, JacobianMode, KinkPolicy, Lanes, LoopBody, NonFinitePolicy,
    Real, Tape, TapeOptions, TapeValue, Var, checkpointed_loop, solve_into,
};

use scioxide_core::{
    AdjointOperator, CancellationToken, Capacity, ColoringKind, ColoringSchedule, DomainReason,
    ExecutionContext, InvalidInputReason, LinearOperator, NonFiniteOrigin, RealScalar,
    SciOxideError, SciOxideErrorView,
};

fn cx() -> ExecutionContext<'static> {
    ExecutionContext::sequential()
}

/// Bits with `-0` and `+0` identified: adjoints accumulate from `+0`, so a `-0`
/// derivative comes back as `+0`.
fn bits<S: RealScalar>(x: S) -> u64 {
    if x == S::ZERO { 0 } else { support::bits(x) }
}

fn unary_ops<S: RealScalar, R: Real<S>>(x: R) -> Vec<R> {
    let (s, c) = x.sin_cos();
    vec![
        x.recip(),
        x.sqrt(),
        x.cbrt(),
        x.exp(),
        x.exp2(),
        x.exp_m1(),
        x.ln(),
        x.ln_1p(),
        x.log2(),
        x.log10(),
        x.sin(),
        x.cos(),
        x.tan(),
        x.asin(),
        x.acos(),
        x.atan(),
        x.sinh(),
        x.cosh(),
        x.tanh(),
        x.asinh(),
        x.acosh(),
        x.atanh(),
        x.abs(),
        x.signum(),
        x.floor(),
        x.ceil(),
        x.round(),
        x.trunc(),
        x.fract(),
        x.to_degrees(),
        x.to_radians(),
        x.softplus(),
        x.sigmoid(),
        x.log_sigmoid(),
        x.logit(),
        x.exprel(),
        x.cosm1(),
        x.sinc(),
        x.sincpi(),
        x.sin_pi(),
        x.cos_pi(),
        x.tan_pi(),
        x.sin_degrees(),
        x.cos_degrees(),
        x.tan_degrees(),
        x.powi(3),
        x.powi(-2),
        x.powi(0),
        s,
        c,
        -x,
        x * S::TWO,
        x / S::TWO,
        x + S::TWO,
        x % S::TWO,
    ]
}

fn binary_ops<S: RealScalar, R: Real<S>>(x: R, y: R) -> Vec<R> {
    vec![
        x.powf(y),
        x.hypot(y),
        x.atan2(y),
        x.min(y),
        x.max(y),
        x.log(y),
        x.logaddexp(y),
        x.xlogy(y),
        x.xlog1py(y),
        x.powm1(y),
        x + y,
        x - y,
        x * y,
        x / y,
        x % y,
        x.azmul(y),
        x.clamp(y, y + S::ONE),
    ]
}

fn unary_bits<S: RealScalar>() {
    let points = [0.3, -0.7, 1.7, 2.5, 0.999, 4.0, 1.0, 0.0];
    let count = unary_ops(S::ONE).len();
    let mut tape = Tape::<S>::new(TapeOptions::new());
    for &p in &points {
        let x = S::from_f64_ieee(p);
        let forward = unary_ops(Dual::variable(x));
        for k in 0..count {
            let mut gradient = [S::ZERO];
            tape.gradient_into(|v| Ok(unary_ops(v[0])[k]), &[x], &mut gradient, cx())
                .unwrap();
            assert_eq!(
                bits(gradient[0]),
                bits(forward[k].derivative()),
                "x = {p}, operation {k}: {:?} != {:?}",
                gradient[0],
                forward[k].derivative()
            );
        }
    }
}

fn binary_bits<S: RealScalar>() {
    let points = [(0.3, 1.7), (2.5, -0.7), (1.5, 2.0), (0.8, 0.8), (3.0, 0.25)];
    let count = binary_ops(S::ONE, S::ONE).len();
    let mut tape = Tape::<S>::new(TapeOptions::new());
    let (one, zero) = (S::ONE, S::ZERO);
    for &(a, b) in &points {
        let (x, y) = (S::from_f64_ieee(a), S::from_f64_ieee(b));
        let dx = binary_ops(Dual::new(x, one), Dual::new(y, zero));
        let dy = binary_ops(Dual::new(x, zero), Dual::new(y, one));
        for k in 0..count {
            let mut gradient = [zero; 2];
            tape.gradient_into(
                |v| Ok(binary_ops(v[0], v[1])[k]),
                &[x, y],
                &mut gradient,
                cx(),
            )
            .unwrap();
            assert_eq!(
                bits(gradient[0]),
                bits(dx[k].derivative()),
                "({a}, {b}) op {k} ∂x"
            );
            assert_eq!(
                bits(gradient[1]),
                bits(dy[k].derivative()),
                "({a}, {b}) op {k} ∂y"
            );
        }
    }
}

#[test]
fn reverse_gradients_equal_dual_derivatives_bit_for_bit() {
    unary_bits::<f32>();
    unary_bits::<f64>();
    binary_bits::<f32>();
    binary_bits::<f64>();
}

fn rosenbrock<S: RealScalar, R: Real<S>>(x: &[R]) -> R {
    x.windows(2)
        .map(|w| {
            let a = w[1] - w[0] * w[0];
            let b = -w[0] + S::ONE;
            a * a * S::from_f64_ieee(100.0) + b * b
        })
        .sum()
}

fn broyden<S: RealScalar, R: Real<S>>(x: &[R], y: &mut Vec<R>) {
    let n = x.len();
    for i in 0..n {
        let mut f = (-(x[i] * S::TWO) + S::from_f64_ieee(3.0)) * x[i] + S::ONE;
        if i > 0 {
            f -= x[i - 1];
        }
        if i + 1 < n {
            f -= x[i + 1] * S::TWO;
        }
        y.push(f);
    }
}

#[test]
fn recording_twice_matches_a_reserved_tape_for_scalars_and_lanes() {
    let point: Vec<f64> = (0..40).map(|k| 0.1 * k as f64 - 1.3).collect();
    let mut cold = Tape::<f64>::new(TapeOptions::new());
    let mut first = vec![0.0; point.len()];
    let v1 = cold
        .gradient_into(|x| Ok(rosenbrock(x)), &point, &mut first, cx())
        .unwrap();
    assert!(cold.report().recorded_twice());
    let requirements = cold.requirements();
    let mut warm = Tape::<f64>::new(TapeOptions::new());
    assert_eq!(warm.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    warm.try_reserve(&requirements).unwrap();
    assert_eq!(warm.check(&requirements).unwrap(), Capacity::Sufficient);
    let mut second = vec![0.0; point.len()];
    let v2 = warm
        .gradient_into(|x| Ok(rosenbrock(x)), &point, &mut second, cx())
        .unwrap();
    assert!(!warm.report().recorded_twice());
    assert_eq!(v1.to_bits(), v2.to_bits());
    assert_eq!(first, second);

    let lanes: Vec<Lanes<f64, 4>> = point
        .iter()
        .map(|&p| Lanes::from_fn(|k| p + 0.01 * k as f64))
        .collect();
    let mut cold = Tape::<Lanes<f64, 4>>::new(TapeOptions::new());
    let mut first = vec![Lanes::splat(0.0); lanes.len()];
    cold.gradient_into(|x| Ok(rosenbrock(x)), &lanes, &mut first, cx())
        .unwrap();
    let mut warm = Tape::<Lanes<f64, 4>>::new(TapeOptions::new());
    warm.try_reserve(&cold.requirements()).unwrap();
    let mut second = vec![Lanes::splat(0.0); lanes.len()];
    warm.gradient_into(|x| Ok(rosenbrock(x)), &lanes, &mut second, cx())
        .unwrap();
    assert!(!warm.report().recorded_twice());
    assert_eq!(first, second);
    for lane in 0..4 {
        let scalar: Vec<f64> = lanes.iter().map(|l| l.lane(lane).unwrap()).collect();
        let g = scioxide_ad::grad(|x| Ok(rosenbrock(x)), &scalar).unwrap();
        for (k, g) in g.as_slice().iter().enumerate() {
            assert_eq!(first[k].lane(lane).unwrap().to_bits(), g.to_bits());
        }
    }
}

#[test]
fn byte_limits_report_required_and_allowed_sizes_and_leave_outputs_unchanged() {
    let point = [0.5; 64];
    let mut tape = Tape::<f64>::new(TapeOptions::new().with_byte_limit(Some(256)));
    let mut gradient = [7.0; 64];
    let error = tape
        .gradient_into(|x| Ok(rosenbrock(x)), &point, &mut gradient, cx())
        .unwrap_err();
    match error.view() {
        SciOxideErrorView::ResourceLimit {
            required, limit, ..
        } => {
            assert_eq!(limit, 256);
            assert!(required > 256);
        }
        other => panic!("{other:?}"),
    }
    assert!(gradient.iter().all(|&g| g == 7.0));
    let requirements = scioxide_ad::TapeRequirements::new(1 << 20, 0, 0, 0, 0, 0);
    assert!(matches!(
        tape.try_reserve(&requirements).unwrap_err().view(),
        SciOxideErrorView::ResourceLimit { .. }
    ));
}

/// `y = (x₀ x₁, x₀ + x₁)` with a pullback and a pushforward.
struct Pair;

impl<V: TapeValue> ExternalRule<V> for Pair {
    fn evaluate(&self, x: &[V], y: &mut [V]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[1];
        y[1] = x[0] + x[1];
        Ok(())
    }
    fn pullback(&self, x: &[V], _: &[V], w: &[V], g: &mut [V]) -> Result<(), SciOxideError> {
        g[0] += w[0] * x[1] + w[1];
        g[1] += w[0] * x[0] + w[1];
        Ok(())
    }
    fn pushforward(&self, x: &[V], _: &[V], t: &[V], out: &mut [V]) -> Result<(), SciOxideError> {
        out[0] = t[0] * x[1] + x[0] * t[1];
        out[1] = t[0] + t[1];
        Ok(())
    }
}

/// A rule without a pushforward.
struct Square;

impl ExternalRule<f64> for Square {
    fn evaluate(&self, x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0];
        Ok(())
    }
    fn pullback(
        &self,
        x: &[f64],
        _: &[f64],
        w: &[f64],
        g: &mut [f64],
    ) -> Result<(), SciOxideError> {
        g[0] += 2.0 * w[0] * x[0];
        Ok(())
    }
}

/// Uses every statement form: unary, binary, scalar, linear, n-ary and call.
fn every_form<'t>(
    rule: &Arc<dyn ExternalRule<f64>>,
    x: &[Var<'t, f64>],
    y: &mut Vec<Var<'t, f64>>,
) -> Result<(), SciOxideError> {
    let a = x[0].sin() * x[1] + x[2] / x[0] - x[1] * 2.0 + 3.0 / x[2];
    let b = -x[1] + x[0].hypot(x[2]) + x[0] % 0.7;
    let c = Var::combine(&[a, b], a.value() * b.value(), &[b.value(), a.value()])?;
    let mut outputs = [Var::constant(0.0); 2];
    Var::external(rule, &[c, x[1]], &mut outputs)?;
    let d = Var::select(x[0].lt(x[1]), outputs[0], outputs[1].exp());
    y.extend([a, b, c, d, outputs[1], x[2]]);
    Ok(())
}

#[test]
fn reverse_and_tangent_products_are_adjoint() {
    let rule: Arc<dyn ExternalRule<f64>> = Arc::new(Pair);
    let point = [0.7, 1.3, -0.4];
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut linear = tape
        .linearize(|x, y| every_form(&rule, x, y), &point, cx())
        .unwrap();
    let (m, n) = (linear.outputs(), linear.inputs());
    let v = [0.3, -1.1, 0.6];
    let w: Vec<f64> = (0..m).map(|k| 0.5 - 0.2 * k as f64).collect();
    let mut jv = vec![0.0; m];
    let mut jtw = vec![0.0; n];
    linear.jvp_into(&v, &mut jv, cx()).unwrap();
    linear.vjp_into(&w, &mut jtw, cx()).unwrap();
    let left: f64 = w.iter().zip(&jv).map(|(a, b)| a * b).sum();
    let right: f64 = jtw.iter().zip(&v).map(|(a, b)| a * b).sum();
    assert!(
        (left - right).abs() <= 1e-13 * left.abs().max(1.0),
        "{left} {right}"
    );
    // Both Jacobian modes agree.
    let mut forward = vec![0.0; m * n];
    linear.jacobian_into(&mut forward, cx()).unwrap();
    let mut tape = Tape::<f64>::new(TapeOptions::new().with_jacobian_mode(JacobianMode::Reverse));
    let mut linear = tape
        .linearize(|x, y| every_form(&rule, x, y), &point, cx())
        .unwrap();
    let mut reverse = vec![0.0; m * n];
    linear.jacobian_into(&mut reverse, cx()).unwrap();
    for (a, b) in forward.iter().zip(&reverse) {
        assert!((a - b).abs() <= 1e-14 * a.abs().max(1.0));
    }
}

#[test]
fn a_rule_without_pushforward_has_no_tangent_products() {
    let rule: Arc<dyn ExternalRule<f64>> = Arc::new(Square);
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut linear = tape
        .linearize(
            |x, y| {
                let mut out = [Var::constant(0.0)];
                Var::external(&rule, x, &mut out)?;
                y.push(out[0]);
                Ok(())
            },
            &[3.0],
            cx(),
        )
        .unwrap();
    let mut out = [5.0];
    linear.vjp_into(&[1.0], &mut out, cx()).unwrap();
    assert_eq!(out, [6.0]);
    let error = linear.jvp_into(&[1.0], &mut out, cx()).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::CapabilityUnavailable { .. }
    ));
    assert_eq!(out, [6.0]);
}

#[test]
fn non_finite_policy_reports_input_and_output() {
    let strict = TapeOptions::new().with_non_finite_policy(NonFinitePolicy::Error);
    let mut tape = Tape::<f64>::new(strict);
    let mut gradient = [9.0];
    let origin = |error: SciOxideError| match error.view() {
        SciOxideErrorView::NonFinite { origin, .. } => origin,
        other => panic!("{other:?}"),
    };
    let error = tape
        .gradient_into(|x| Ok(x[0] * 2.0), &[f64::NAN], &mut gradient, cx())
        .unwrap_err();
    assert_eq!(origin(error), NonFiniteOrigin::Input);
    let error = tape
        .gradient_into(|x| Ok(x[0].recip()), &[0.0], &mut gradient, cx())
        .unwrap_err();
    assert_eq!(origin(error), NonFiniteOrigin::Output);
    let error = tape
        .gradient_into(|x| Ok(x[0].sqrt()), &[0.0], &mut gradient, cx())
        .unwrap_err();
    assert_eq!(origin(error), NonFiniteOrigin::Output);
    assert_eq!(gradient, [9.0]);
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    tape.gradient_into(|x| Ok(x[0].sqrt()), &[0.0], &mut gradient, cx())
        .unwrap();
    assert_eq!(gradient, [f64::INFINITY]);
}

#[test]
fn cancellation_stops_a_sweep() {
    let point = vec![0.1; 2000];
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = ExecutionContext::sequential().with_cancellation(&token);
    let mut linear = tape
        .linearize(
            |x, y| {
                y.push(rosenbrock(x));
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let mut out = vec![4.0; 2000];
    let error = linear.vjp_into(&[1.0], &mut out, cancelled).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert!(out.iter().all(|&g| g == 4.0));
    // The tape recovers.
    linear.vjp_into(&[1.0], &mut out, cx()).unwrap();
    let expected = scioxide_ad::grad(|x| Ok(rosenbrock(x)), &point).unwrap();
    assert_eq!(out, expected.as_slice());
}

/// Checks one row of the kink table: the gradient in `Dual` (one seed per input)
/// and on a tape, and the number of hits the tape counts.
macro_rules! kink {
    (|$x:ident| $body:expr, $point:expr => $gradient:expr, $count:expr) => {{
        fn f<R: Real>($x: &[R]) -> R {
            $body
        }
        let (point, expected): (&[f64], &[f64]) = (&$point, &$gradient);
        let row = stringify!($body);
        for (k, &e) in expected.iter().enumerate() {
            let x: Vec<_> = point
                .iter()
                .enumerate()
                .map(|(j, &p)| Dual::new(p, if j == k { 1.0 } else { 0.0 }))
                .collect();
            assert_eq!(f(&x).derivative(), e, "{row} at {point:?}: Dual ∂{k}");
        }
        let mut tape = Tape::<f64>::new(TapeOptions::new());
        let mut g = vec![0.0; point.len()];
        tape.gradient_into(|x| Ok(f(x)), point, &mut g, cx())
            .unwrap();
        assert_eq!(g, expected, "{row} at {point:?}: tape");
        assert_eq!(tape.report().kinks(), $count, "{row} at {point:?}: hits");
    }};
}

static SWEEP_STOP: CancellationToken = CancellationToken::new();
static PULLBACKS: AtomicUsize = AtomicUsize::new(0);

/// The identity, whose pullbacks count themselves and cancel [`SWEEP_STOP`].
struct CancelOnPullback;

impl ExternalRule<f64> for CancelOnPullback {
    fn evaluate(&self, x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = x[0];
        Ok(())
    }
    fn pullback(
        &self,
        _: &[f64],
        _: &[f64],
        w: &[f64],
        g: &mut [f64],
    ) -> Result<(), SciOxideError> {
        PULLBACKS.fetch_add(1, Ordering::Relaxed);
        SWEEP_STOP.cancel();
        g[0] += w[0];
        Ok(())
    }
}

#[test]
fn a_context_cancelled_during_a_sweep_stops_it_within_1024_words() {
    // 100 calls, each after 64 scalar statements of at least one word: a check
    // every 1,024 words lets at most 1024/64 + 1 further pullbacks run.
    const CALLS: usize = 100;
    const SEGMENT: usize = 64;
    let rule: Arc<dyn ExternalRule<f64>> = Arc::new(CancelOnPullback);
    let context = ExecutionContext::sequential().with_cancellation(&SWEEP_STOP);
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut g = [7.0];
    let error = tape
        .gradient_into(
            |x| {
                let mut y = x[0];
                for _ in 0..CALLS {
                    for _ in 0..SEGMENT {
                        y *= 1.0001;
                    }
                    let mut out = [Var::constant(0.0)];
                    Var::external(&rule, &[y], &mut out)?;
                    y = out[0];
                }
                Ok(y)
            },
            &[1.0],
            &mut g,
            context,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(g, [7.0]);
    let after = PULLBACKS.load(Ordering::Relaxed) - 1;
    assert!(
        after <= 1024 / SEGMENT + 1,
        "{after} pullbacks after cancellation"
    );
}

#[test]
fn kink_table_values_and_counts() {
    kink!(|x| x[0].abs(), [0.0] => [0.0], 1);
    kink!(|x| x[0].abs(), [-2.0] => [-1.0], 0);
    kink!(|x| x[0].max(x[1]), [1.0, 1.0] => [0.5, 0.5], 1);
    kink!(|x| x[0].max(x[1]), [1.0, 2.0] => [0.0, 1.0], 0);
    kink!(|x| x[0].max(x[1]), [1.0, f64::NAN] => [1.0, 0.0], 0);
    kink!(|x| x[0].min(x[1]), [1.0, 1.0] => [0.5, 0.5], 1);
    kink!(|x| x[0].min(x[1]), [1.0, 2.0] => [1.0, 0.0], 0);
    kink!(|x| x[0].floor(), [3.0] => [0.0], 1);
    kink!(|x| x[0].floor(), [3.5] => [0.0], 0);
    kink!(|x| x[0].ceil(), [-3.0] => [0.0], 1);
    kink!(|x| x[0].ceil(), [3.2] => [0.0], 0);
    kink!(|x| x[0].round(), [2.5] => [0.0], 1);
    kink!(|x| x[0].round(), [-2.5] => [0.0], 1);
    kink!(|x| x[0].round(), [2.0] => [0.0], 0);
    kink!(|x| x[0].trunc(), [2.0] => [0.0], 1);
    kink!(|x| x[0].trunc(), [0.0] => [0.0], 0);
    kink!(|x| x[0].trunc(), [2.5] => [0.0], 0);
    kink!(|x| x[0].signum(), [0.0] => [0.0], 1);
    kink!(|x| x[0].signum(), [1.0] => [0.0], 0);
    kink!(|x| x[0].fract(), [2.0] => [1.0], 1);
    kink!(|x| x[0].fract(), [0.0] => [1.0], 0);
    kink!(|x| x[0].fract(), [2.5] => [1.0], 0);
    // a % b: partials (1, −trunc(a/b)); a kink where a/b is a non-zero integer, with
    // the midpoint −2.5 of the one-sided partials −3 and −2.
    kink!(|x| x[0] % x[1], [6.0, 2.0] => [1.0, -2.5], 1);
    kink!(|x| x[0] % x[1], [-6.0, 2.0] => [1.0, 2.5], 1);
    kink!(|x| x[0] % x[1], [7.0, 2.0] => [1.0, -3.0], 0);
    kink!(|x| x[0] % x[1], [0.0, 2.0] => [1.0, 0.0], 0);
    kink!(|x| x[0].hypot(x[1]), [0.0, 0.0] => [0.0, 0.0], 1);
    kink!(|x| x[0].hypot(x[1]), [3.0, 4.0] => [0.6, 0.8], 0);
    kink!(|x| x[0].atan2(x[1]), [0.0, 0.0] => [0.0, 0.0], 1);
    kink!(|x| x[0].atan2(x[1]), [0.0, -1.0] => [-1.0, 0.0], 1);
    kink!(|x| x[0].atan2(x[1]), [1.0, 1.0] => [0.5, -0.5], 0);
    // One hit per statement and lane, also with one active operand.
    kink!(|x| x[0].max(R::constant(1.0)), [1.0] => [0.5], 1);
    kink!(|x| x[0].abs() + x[0].max(x[1]), [0.0, 0.0] => [0.5, 0.5], 2);
    // A NaN in both operands gives a NaN derivative.
    let nan = Dual::new(f64::NAN, 5.0);
    assert!(nan.max(nan).derivative().is_nan());
}

#[test]
fn kinks_are_counted_only_on_tapes_and_rejected_under_the_error_policy() {
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut g = [0.0, 0.0];
    // Stable transforms and regime switches are not kinks.
    let regimes = [-40.0, -1.0, -1e-12, 0.0, 1e-12, 0.5, 1.0, 40.0];
    for &p in &regimes {
        tape.gradient_into(
            |x| {
                let y = x[0];
                let stable = [
                    y.softplus(),
                    y.sigmoid(),
                    y.log_sigmoid(),
                    y.exprel(),
                    y.cosm1(),
                    y.sinc(),
                    y.sincpi(),
                    y.sin_pi(),
                    y.cos_pi(),
                    y.tan_pi(),
                    y.sin_degrees(),
                    y.cos_degrees(),
                    y.tan_degrees(),
                    (y * 0.01 + 0.5).logit(),
                ];
                let switches = [
                    y.asinh(),
                    (y * y + 1.0).acosh(),
                    (y * 0.01).atanh(),
                    y.exp_m1(),
                    (y * y).ln_1p(),
                    y.atan(),
                    y.tanh(),
                    y.hypot(x[1]),
                    x[1].atan2(y),
                    y.logaddexp(x[1]),
                    (y * y + 1.0).powf(x[1]),
                ];
                Ok(stable.into_iter().chain(switches).sum())
            },
            &[p, 0.75],
            &mut g,
            cx(),
        )
        .unwrap();
        assert_eq!(tape.report().kinks(), 0, "{p}");
    }
    // Passive kinks are not counted.
    tape.gradient_into(
        |x| Ok(x[0] * Var::constant(0.0).abs()),
        &[1.0, 0.0],
        &mut g,
        cx(),
    )
    .unwrap();
    assert_eq!(tape.report().kinks(), 0);
    // Forward batches have no tape and report no count; reverse batches count.
    struct Abs;
    impl BatchFunction for Abs {
        type Shared<R: Real> = ();
        fn prepare<R: Real>(&self, _: &[R]) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn item<R: Real>(&self, (): &(), a: &[R], _: &[R::Base]) -> Result<R, SciOxideError> {
            Ok(a[0].abs())
        }
    }
    let points = [0.0, 1.0, 0.0, -1.0, 0.0];
    for (mode, kinks) in [
        (DerivativeMode::Forward, None),
        (DerivativeMode::Reverse, Some(3)),
    ] {
        let options = BatchOptions::new().with_derivative_mode(mode);
        let plan = BatchPlan::<4>::new(5, 0, 1, 0, BatchReduction::Sum, options).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut gradients = [0.0; 5];
        let outputs = BatchOutputs::new().with_input_gradient(&mut gradients);
        let none: &[f64] = &[];
        let estimate = plan
            .evaluate_into(&Abs, &[], &points[..], none, outputs, &mut workspace, cx())
            .unwrap();
        assert_eq!(estimate.kinks(), kinks, "{mode:?}");
        assert_eq!(gradients, [0.0, 1.0, 0.0, -1.0, 0.0]);
    }
    let plan = BatchPlan::<4>::new(5, 0, 1, 0, BatchReduction::Sum, BatchOptions::new()).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let none: &[f64] = &[];
    let estimate = plan
        .evaluate_into(
            &Abs,
            &[],
            &points[..],
            none,
            BatchOutputs::new(),
            &mut workspace,
            cx(),
        )
        .unwrap();
    assert_eq!(estimate.kinks(), None);
    // The error policy turns hits into a domain error and leaves outputs unchanged.
    let mut strict = Tape::<f64>::new(TapeOptions::new().with_kink_policy(KinkPolicy::Error));
    let mut g = [8.0];
    let error = strict
        .gradient_into(|x| Ok(x[0].floor()), &[2.0], &mut g, cx())
        .unwrap_err();
    match error.view() {
        SciOxideErrorView::DomainError {
            argument, reason, ..
        } => {
            assert_eq!(argument.as_str(), "point");
            assert_eq!(reason, DomainReason::Undefined);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(g, [8.0]);
    // One hit per lane.
    let mut lanes = Tape::<Lanes<f64, 4>>::new(TapeOptions::new());
    let mut g = [Lanes::splat(0.0)];
    lanes
        .gradient_into(
            |x| Ok(x[0].abs()),
            &[Lanes::from_array([0.0, 1.0, 0.0, -1.0])],
            &mut g,
            cx(),
        )
        .unwrap();
    assert_eq!(lanes.report().kinks(), 2);
    assert_eq!(g[0].to_array(), [0.0, 1.0, 0.0, -1.0]);
}

#[test]
fn truncated_square_roots_have_zero_derivative_and_the_mode_difference_is_pinned() {
    let f = |v: Dual<f64>| v.max(Dual::constant(0.0)).sqrt();
    assert_eq!(
        f(Dual::variable(-1.0)).derivative().to_bits(),
        0.0_f64.to_bits()
    );
    let g = scioxide_ad::grad(|v| Ok(v[0].max(Var::constant(0.0)).sqrt()), &[-1.0_f64]).unwrap();
    assert_eq!(g.as_slice()[0].to_bits(), 0.0_f64.to_bits());
    // An infinite partial times an exactly zero tangent: NaN forward, 0 in a sweep.
    let forward = Dual::new(f64::INFINITY, 1.0) * Dual::new(2.0, 0.0);
    assert!(forward.derivative().is_nan());
    let reverse = scioxide_ad::grad(|x| Ok(x[0] * x[1]), &[f64::INFINITY, 2.0]).unwrap();
    assert_eq!(reverse.as_slice(), [2.0, f64::INFINITY]);
}

#[test]
fn patterns_are_exact_and_sparse_values_equal_dense_ones() {
    let n = 12;
    let point: Vec<f64> = (0..n).map(|k| 0.3 + 0.1 * k as f64).collect();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut linear = tape
        .linearize(
            |x, y| {
                broyden(x, y);
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let pattern = linear.jacobian_pattern().unwrap();
    let view = pattern.view();
    for row in 0..n {
        let columns = &view.col_indices()[view.row_ptr()[row]..view.row_ptr()[row + 1]];
        let expected: Vec<usize> = (row.saturating_sub(1)..(row + 2).min(n)).collect();
        assert_eq!(columns, expected.as_slice(), "row {row}");
    }
    let coloring = scioxide_sparse::derivative_coloring(pattern.view()).unwrap();
    assert_eq!(coloring.color_count(), 3);
    let mut values = vec![0.0; view.col_indices().len()];
    linear
        .sparse_jacobian_into(&coloring, &mut values, cx())
        .unwrap();
    let mut dense = vec![0.0; n * n];
    linear.jacobian_into(&mut dense, cx()).unwrap();
    for row in 0..n {
        for entry in view.row_ptr()[row]..view.row_ptr()[row + 1] {
            let column = view.col_indices()[entry];
            assert_eq!(values[entry].to_bits(), dense[row * n + column].to_bits());
        }
    }
    // A superset pattern (pentadiagonal) gives the same values.
    let mut pointers = vec![0];
    let mut indices = Vec::new();
    for row in 0..n {
        indices.extend(row.saturating_sub(2)..(row + 3).min(n));
        pointers.push(indices.len());
    }
    let superset = scioxide_core::StructuralPattern::new(n, n, pointers, indices).unwrap();
    let coloring = scioxide_sparse::derivative_coloring(superset.view()).unwrap();
    let mut wide = vec![0.0; superset.view().col_indices().len()];
    linear
        .sparse_jacobian_into(&coloring, &mut wide, cx())
        .unwrap();
    let view = superset.view();
    for row in 0..n {
        for entry in view.row_ptr()[row]..view.row_ptr()[row + 1] {
            let column = view.col_indices()[entry];
            assert_eq!(wide[entry].to_bits(), dense[row * n + column].to_bits());
        }
    }

    // The Rosenbrock Hessian is tridiagonal.
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let linear = tape
        .linearize(
            |x, y| {
                y.push(rosenbrock(x));
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let pattern = linear.hessian_pattern().unwrap();
    let view = pattern.view();
    for row in 0..n {
        let columns = &view.col_indices()[view.row_ptr()[row]..view.row_ptr()[row + 1]];
        let expected: Vec<usize> = (row.saturating_sub(1)..(row + 2).min(n)).collect();
        assert_eq!(columns, expected.as_slice(), "row {row}");
    }
    let coloring = scioxide_sparse::derivative_coloring(pattern.view()).unwrap();
    let mut second = Tape::<Dual<f64>>::new(TapeOptions::new());
    let mut values = vec![0.0; view.col_indices().len()];
    second
        .sparse_hessian_into(|x| Ok(rosenbrock(x)), &point, &coloring, &mut values, cx())
        .unwrap();
    let dense = scioxide_ad::hessian(|x| Ok(rosenbrock(x)), &point).unwrap();
    for row in 0..n {
        for entry in view.row_ptr()[row]..view.row_ptr()[row + 1] {
            let column = view.col_indices()[entry];
            assert_eq!(
                values[entry].to_bits(),
                dense.as_slice()[row * n + column].to_bits()
            );
        }
    }
    // The pentadiagonal superset gives the same values.
    let coloring = scioxide_sparse::derivative_coloring(superset.view()).unwrap();
    let view = superset.view();
    let mut wide = vec![0.0; view.col_indices().len()];
    second
        .sparse_hessian_into(|x| Ok(rosenbrock(x)), &point, &coloring, &mut wide, cx())
        .unwrap();
    for row in 0..n {
        for entry in view.row_ptr()[row]..view.row_ptr()[row + 1] {
            let column = view.col_indices()[entry];
            assert_eq!(
                wide[entry].to_bits(),
                dense.as_slice()[row * n + column].to_bits(),
                "({row}, {column})"
            );
        }
    }
}

#[test]
fn one_shot_sparse_derivatives_equal_the_tape_forms() {
    let n = 12;
    let point: Vec<f64> = (0..n).map(|k| 0.3 + 0.1 * k as f64).collect();
    // The Jacobian of the Broyden function: tridiagonal, with the values of the
    // tangent sweeps.
    let j = scioxide_ad::sparse_jacobian(
        |x, y| {
            broyden(x, y);
            Ok(())
        },
        &point,
    )
    .unwrap();
    assert_eq!((j.rows(), j.cols(), j.nnz()), (n, n, 3 * n - 2));
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut linear = tape
        .linearize(
            |x, y| {
                broyden(x, y);
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let mut dense = vec![0.0; n * n];
    linear.jacobian_into(&mut dense, cx()).unwrap();
    for (row, column, value) in j.iter() {
        assert_eq!(value.to_bits(), dense[row * n + column].to_bits());
    }
    // The Rosenbrock Hessian: the pattern on a dual tape equals the one of a
    // linearization, and the values equal `hessian`.
    let mut second = Tape::<Dual<f64>>::default();
    let pattern = second
        .hessian_pattern(|x| Ok(rosenbrock(x)), &point, cx())
        .unwrap();
    let scalar = Tape::<f64>::default()
        .linearize(
            |x, y| {
                y.push(rosenbrock(x));
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap()
        .hessian_pattern()
        .unwrap();
    assert_eq!(pattern, scalar);
    let h = scioxide_ad::sparse_hessian(|x| Ok(rosenbrock(x)), &point).unwrap();
    assert_eq!(h.structural_pattern(), pattern.view());
    let dense = scioxide_ad::hessian(|x| Ok(rosenbrock(x)), &point).unwrap();
    for (row, column, value) in h.iter() {
        assert_eq!(
            value.to_bits(),
            dense.as_slice()[row * n + column].to_bits()
        );
    }
    // No outputs: a 0 × n Jacobian. A non-finite derivative is rejected.
    let empty = scioxide_ad::sparse_jacobian(|_, _| Ok(()), &point).unwrap();
    assert_eq!((empty.rows(), empty.cols(), empty.nnz()), (0, n, 0));
    let error = scioxide_ad::sparse_hessian(|x| Ok(x[0].sqrt() * x[1]), &[0.0, 1.0]).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::NonFinite { .. }),
        "{error}"
    );
    let error = scioxide_ad::sparse_jacobian(
        |x, y| {
            y.push(x[0].sqrt());
            Ok(())
        },
        &[0.0],
    )
    .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::NonFinite { .. }),
        "{error}"
    );
}

#[test]
fn reused_tapes_reserve_what_a_fresh_tape_reserves() {
    let point: Vec<f64> = (0..32).map(|k| 0.05 * k as f64 - 0.4).collect();
    let fresh = |n: usize| {
        let mut tape = Tape::<f64>::default();
        let mut gradient = vec![0.0; n];
        tape.gradient_into(|x| Ok(rosenbrock(x)), &point[..n], &mut gradient, cx())
            .unwrap();
        (tape.reserved_bytes(), tape.requirements())
    };
    let mut reused = Tape::<f64>::default();
    for n in 1..=32 {
        let mut gradient = vec![0.0; n];
        reused
            .gradient_into(|x| Ok(rosenbrock(x)), &point[..n], &mut gradient, cx())
            .unwrap();
        let (bytes, requirements) = fresh(n);
        // One input per point value and one output.
        assert_eq!(requirements.variables(), n + 1, "n = {n}");
        assert!(reused.requirements().variables() <= n + 1, "n = {n}");
        assert!(
            reused.reserved_bytes() <= 2 * bytes,
            "n = {n}: {} reused, {bytes} fresh",
            reused.reserved_bytes()
        );
    }
    // A limit that admits a fresh recording admits it after smaller ones.
    let (bytes, _) = fresh(32);
    let mut limited = Tape::<f64>::new(TapeOptions::new().with_byte_limit(Some(bytes)));
    for n in [4, 8, 16, 32] {
        let mut gradient = vec![0.0; n];
        limited
            .gradient_into(|x| Ok(rosenbrock(x)), &point[..n], &mut gradient, cx())
            .unwrap();
    }
}

#[test]
fn operators_match_the_linearization() {
    let n = 8;
    let point: Vec<f64> = (0..n).map(|k| 0.2 * k as f64 - 0.5).collect();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut linear = tape
        .linearize(
            |x, y| {
                broyden(x, y);
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let v: Vec<f64> = (0..n).map(|k| 1.0 / (1.0 + k as f64)).collect();
    let mut expected = vec![0.0; n];
    linear.jvp_into(&v, &mut expected, cx()).unwrap();
    let mut adjoint_expected = vec![0.0; n];
    linear.vjp_into(&v, &mut adjoint_expected, cx()).unwrap();
    let operator = linear.jacobian_operator().unwrap();
    let mut workspace = vec![0.0; operator.apply_workspace_len().unwrap()];
    let mut out = vec![0.0; n];
    operator.apply_into(&v, &mut out, &mut workspace).unwrap();
    assert_eq!(out, expected);
    operator
        .apply_adjoint_into(&v, &mut out, &mut workspace)
        .unwrap();
    assert_eq!(out, adjoint_expected);

    let mut second = Tape::<Dual<f64>>::new(TapeOptions::new());
    let mut hv = vec![0.0; n];
    second
        .hvp_into(|x| Ok(rosenbrock(x)), &point, &v, &mut hv, cx())
        .unwrap();
    let operator = second
        .hessian_operator(|x| Ok(rosenbrock(x)), &point, cx())
        .unwrap();
    let mut out = vec![0.0; n];
    operator.apply_into(&v, &mut out, &mut []).unwrap();
    assert_eq!(out, hv);
}

#[test]
fn warm_products_execute() {
    let n = 32;
    let point: Vec<f64> = (0..n).map(|k| 0.05 * k as f64).collect();
    let v: Vec<f64> = (0..n).map(|k| 1.0 - 0.01 * k as f64).collect();
    let mut gradient = vec![0.0; n];
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    tape.gradient_into(|x| Ok(rosenbrock(x)), &point, &mut gradient, cx())
        .unwrap();
    tape.gradient_into(|x| Ok(rosenbrock(x)), &point, &mut gradient, cx())
        .unwrap();
    let mut out = vec![0.0; n];
    let mut jacobian = vec![0.0; n * n];
    {
        let mut linear = tape
            .linearize(
                |x, y| {
                    broyden(x, y);
                    Ok(())
                },
                &point,
                cx(),
            )
            .unwrap();
        linear.jacobian_into(&mut jacobian, cx()).unwrap();
    }
    {
        let mut linear = tape
            .linearize(
                |x, y| {
                    broyden(x, y);
                    Ok(())
                },
                &point,
                cx(),
            )
            .unwrap();
        linear.vjp_into(&v, &mut out, cx()).unwrap();
        linear.jvp_into(&v, &mut out, cx()).unwrap();
        linear.jacobian_into(&mut jacobian, cx()).unwrap();
        let operator = linear.jacobian_operator().unwrap();
        let mut workspace = [0.0; 1024];
        let len = operator.apply_workspace_len().unwrap();
        operator
            .apply_into(&v, &mut out, &mut workspace[..len])
            .unwrap();
        operator
            .apply_adjoint_into(&v, &mut out, &mut workspace[..len])
            .unwrap();
    };
    let mut second = Tape::<Dual<f64>>::new(TapeOptions::new());
    second
        .hvp_into(|x| Ok(rosenbrock(x)), &point, &v, &mut out, cx())
        .unwrap();
    second
        .hvp_into(|x| Ok(rosenbrock(x)), &point, &v, &mut out, cx())
        .unwrap();
    let operator = second
        .hessian_operator(|x| Ok(rosenbrock(x)), &point, cx())
        .unwrap();
    operator.apply_into(&v, &mut out, &mut []).unwrap();
}

#[test]
fn lane_selections_record_per_lane_partials() {
    let mut tape = Tape::<Lanes<f64, 4>>::new(TapeOptions::new());
    let x = Lanes::from_array([1.0, -2.0, 3.0, -4.0]);
    let mut g = [Lanes::splat(0.0); 2];
    let value = tape
        .gradient_into(
            |v| {
                Ok(Var::select(
                    v[0].gt(Var::constant(0.0)),
                    v[0] * v[1],
                    v[1] * 3.0,
                ))
            },
            &[x, Lanes::splat(2.0)],
            &mut g,
            cx(),
        )
        .unwrap();
    assert_eq!(value.to_array(), [2.0, 6.0, 6.0, 6.0]);
    assert_eq!(g[0].to_array(), [2.0, 0.0, 2.0, 0.0]);
    assert_eq!(g[1].to_array(), [1.0, 3.0, 3.0, 3.0]);
    // Gathers pick table entries lane by lane and differentiate them.
    let table = [Lanes::splat(10.0), Lanes::splat(20.0), Lanes::splat(30.0)];
    let mut g = [Lanes::splat(0.0); 4];
    let mut point = vec![Lanes::from_array([0.2, 1.7, 2.5, 0.9])];
    point.extend(table);
    tape.gradient_into(
        |v| Var::gather(&v[1..], v[0], |key| key as usize).map(|y| y * 2.0),
        &point,
        &mut g,
        cx(),
    )
    .unwrap();
    assert_eq!(g[0].to_array(), [0.0; 4]);
    assert_eq!(g[1].to_array(), [2.0, 0.0, 0.0, 2.0]);
    assert_eq!(g[2].to_array(), [0.0, 2.0, 0.0, 0.0]);
    assert_eq!(g[3].to_array(), [0.0, 0.0, 2.0, 0.0]);
}

/// Asserts `|aₖ − bₖ| ≤ 1e-13·max(1, |bₖ|)` entrywise.
fn assert_close(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (k, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-13 * b.abs().max(1.0),
            "{context}: entry {k}: {a} != {b}"
        );
    }
}

fn to_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn is_stopped(error: &SciOxideError) -> bool {
    matches!(error.view(), SciOxideErrorView::ExecutionStopped { .. })
}

fn is_resource_limit(error: &SciOxideError) -> bool {
    matches!(error.view(), SciOxideErrorView::ResourceLimit { .. })
}

/// The error a rule or a function returns on purpose.
fn rejection() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("x"),
        DomainReason::Undefined,
        "rejected on purpose",
    )
}

/// `y = x²` with both products; the pullback returns [`rejection`] while `fail` is
/// set and panics once after `panic` is set.
struct Flaky {
    fail: AtomicBool,
    panic: AtomicBool,
}

impl Flaky {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fail: AtomicBool::new(false),
            panic: AtomicBool::new(false),
        })
    }
}

impl ExternalRule<f64> for Flaky {
    fn evaluate(&self, x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0];
        Ok(())
    }
    fn pullback(
        &self,
        x: &[f64],
        _: &[f64],
        w: &[f64],
        g: &mut [f64],
    ) -> Result<(), SciOxideError> {
        if self.panic.swap(false, Ordering::SeqCst) {
            panic!("the pullback panics once");
        }
        if self.fail.load(Ordering::SeqCst) {
            return Err(rejection());
        }
        g[0] += 2.0 * w[0] * x[0];
        Ok(())
    }
    fn pushforward(
        &self,
        x: &[f64],
        _: &[f64],
        t: &[f64],
        out: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = 2.0 * x[0] * t[0];
        Ok(())
    }
}

/// About 3,000 statement words around one call of `rule`: a reverse sweep that
/// stops at the call, or after 1,024 words, leaves adjoints in the slots below.
fn long<'t>(
    rule: &Arc<dyn ExternalRule<f64>>,
    x: &[Var<'t, f64>],
) -> Result<Var<'t, f64>, SciOxideError> {
    let mut a = x[0];
    for _ in 0..500 {
        a = a.sin() * x[1] + 0.5;
    }
    let mut squared = [Var::constant(0.0)];
    Var::external(rule, &[a], &mut squared)?;
    let mut b = squared[0] + a;
    for _ in 0..500 {
        b = b.cos() * x[1] + a;
    }
    Ok(b)
}

#[test]
fn stale_adjoints_of_tangent_failed_and_cancelled_sweeps_do_not_reach_later_gradients() {
    let flaky = Flaky::new();
    let rule: Arc<dyn ExternalRule<f64>> = Arc::clone(&flaky) as Arc<dyn ExternalRule<f64>>;
    let point = [0.7, 0.9];
    let mut reference = [0.0; 2];
    Tape::<f64>::default()
        .gradient_into(|x| long(&rule, x), &point, &mut reference, cx())
        .unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = ExecutionContext::sequential().with_cancellation(&token);
    for dirt in [
        "tangent sweep",
        "failed reverse sweep",
        "cancelled reverse sweep",
    ] {
        let mut tape = Tape::<f64>::default();
        let mut g = [0.0; 2];
        tape.gradient_into(|x| long(&rule, x), &point, &mut g, cx())
            .unwrap();
        assert!(tape.report().words() > 2048, "{}", tape.report().words());
        let mut untouched = [3.0; 2];
        match dirt {
            "tangent sweep" => {
                let mut linear = tape
                    .linearize(
                        |x, y| {
                            y.push(long(&rule, x)?);
                            Ok(())
                        },
                        &point,
                        cx(),
                    )
                    .unwrap();
                let mut tangent = [0.0];
                linear.jvp_into(&[1.0, 1.0], &mut tangent, cx()).unwrap();
            }
            "failed reverse sweep" => {
                flaky.fail.store(true, Ordering::SeqCst);
                let error = tape
                    .gradient_into(|x| long(&rule, x), &point, &mut untouched, cx())
                    .unwrap_err();
                flaky.fail.store(false, Ordering::SeqCst);
                assert_eq!(error, rejection());
            }
            _ => {
                let mut linear = tape
                    .linearize(
                        |x, y| {
                            y.push(long(&rule, x)?);
                            Ok(())
                        },
                        &point,
                        cx(),
                    )
                    .unwrap();
                let error = linear
                    .vjp_into(&[1.0], &mut untouched, cancelled)
                    .unwrap_err();
                assert!(is_stopped(&error), "{error:?}");
            }
        }
        assert_eq!(untouched, [3.0; 2], "{dirt}");
        let mut short = [0.0; 2];
        tape.gradient_into(|x| Ok(x[0] * x[1]), &[1.5, -2.0], &mut short, cx())
            .unwrap();
        assert_eq!(to_bits(&short), to_bits(&[-2.0, 1.5]), "{dirt}: {short:?}");
        tape.gradient_into(|x| long(&rule, x), &point, &mut g, cx())
            .unwrap();
        assert_eq!(
            to_bits(&g),
            to_bits(&reference),
            "after a {dirt}: {g:?} != {reference:?}"
        );
    }
}

#[test]
fn a_pullback_that_panics_leaves_no_stale_adjoints() {
    let flaky = Flaky::new();
    let rule: Arc<dyn ExternalRule<f64>> = Arc::clone(&flaky) as Arc<dyn ExternalRule<f64>>;
    let point = [0.7, 0.9];
    let mut tape = Tape::<f64>::default();
    let mut before = [0.0; 2];
    tape.gradient_into(|x| long(&rule, x), &point, &mut before, cx())
        .unwrap();
    flaky.panic.store(true, Ordering::SeqCst);
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let mut g = [0.0; 2];
        tape.gradient_into(|x| long(&rule, x), &point, &mut g, cx())
    }));
    assert!(caught.is_err());
    assert!(!flaky.panic.load(Ordering::SeqCst));
    let mut after = [0.0; 2];
    tape.gradient_into(|x| long(&rule, x), &point, &mut after, cx())
        .unwrap();
    assert_eq!(to_bits(&after), to_bits(&before), "{after:?} != {before:?}");
}

fn five_sines<'t>(x: &[Var<'t, f64>], y: &mut Vec<Var<'t, f64>>) -> Result<(), SciOxideError> {
    for k in 0..5 {
        y.push(x[k % 4].sin());
    }
    Ok(())
}

/// `s ← sin(s)·x₁`, `steps` times from `s = x₀`.
fn sine_chain<'t>(x: &[Var<'t, f64>], steps: usize) -> Var<'t, f64> {
    let mut s = x[0];
    for _ in 0..steps {
        s = s.sin() * x[1];
    }
    s
}

#[test]
fn a_warm_tape_is_sufficient_for_the_requirements_of_its_last_recording() {
    let point = [0.5; 4];
    let mut tape = Tape::<f64>::default();
    let mut g = [0.0; 4];
    // A longer scalar recording first, so the five outputs fit without growth.
    tape.gradient_into(|x| Ok(sine_chain(x, 50)), &point, &mut g, cx())
        .unwrap();
    tape.linearize(five_sines, &point, cx()).unwrap();
    assert!(!tape.report().recorded_twice());
    let requirements = tape.requirements();
    assert_eq!(requirements.variables(), 4 + 5);
    assert_eq!(tape.check(&requirements).unwrap(), Capacity::Sufficient);
    {
        tape.linearize(five_sines, &point, cx()).unwrap();
    };
    assert_eq!(tape.check(&requirements).unwrap(), Capacity::Sufficient);
}

/// `yₖ = 2·xₖ`.
fn doubled<'t>(x: &[Var<'t, f64>], y: &mut Vec<Var<'t, f64>>) -> Result<(), SciOxideError> {
    y.extend(x.iter().map(|&v| v * 2.0));
    Ok(())
}

fn broyden_outputs<'t>(x: &[Var<'t, f64>], y: &mut Vec<Var<'t, f64>>) -> Result<(), SciOxideError> {
    broyden(x, y);
    Ok(())
}

#[test]
fn byte_limits_bound_variables_staging_and_every_admitted_transform() {
    // The input variables count against the limit before they are reserved.
    let limit = 64 * 1024;
    let mut tape = Tape::<f64>::new(TapeOptions::new().with_byte_limit(Some(limit)));
    let point = vec![1.0; 100_000];
    let mut gradient = vec![7.0; 100_000];
    let error = tape
        .gradient_into(|x| Ok(x[0]), &point, &mut gradient, cx())
        .unwrap_err();
    match error.view() {
        SciOxideErrorView::ResourceLimit {
            required,
            limit: reported,
            ..
        } => {
            assert_eq!(reported, limit);
            assert!(required > limit, "{required}");
        }
        other => panic!("{other:?}"),
    }
    assert!(tape.reserved_bytes() <= limit, "{}", tape.reserved_bytes());
    assert!(gradient.iter().all(|&g| g == 7.0));

    // The staging of a Jacobian counts against a limit that admits the recording.
    let n = 300;
    let point = vec![1.0; n];
    let mut unlimited = Tape::<f64>::default();
    unlimited.linearize(doubled, &point, cx()).unwrap();
    let limit = unlimited.reserved_bytes() + 4096;
    assert!(n * n * size_of::<f64>() > limit);
    let mut tape = Tape::<f64>::new(TapeOptions::new().with_byte_limit(Some(limit)));
    let mut jacobian = vec![7.0; n * n];
    let mut tangent = vec![0.0; n];
    {
        let mut linear = tape.linearize(doubled, &point, cx()).unwrap();
        let error = linear.jacobian_into(&mut jacobian, cx()).unwrap_err();
        assert!(is_resource_limit(&error), "{error:?}");
        linear.jvp_into(&point, &mut tangent, cx()).unwrap();
    }
    assert!(
        tape.reserved_bytes() <= limit,
        "{} > {limit}",
        tape.reserved_bytes()
    );
    assert!(jacobian.iter().all(|&j| j == 7.0));
    assert_close(&tangent, &vec![2.0; n], "J·1");

    // A transform that is admitted leaves the tape within the limit; a rejected one
    // too, except for the seeds and staging that second-order products grow before
    // their recording is admitted (`TapeOptions::with_byte_limit`).
    let point: Vec<f64> = (0..24).map(|k| 0.1 * k as f64 - 1.0).collect();
    let n = point.len();
    let pattern = Tape::<f64>::default()
        .linearize(
            |x, y| {
                y.push(rosenbrock(x));
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap()
        .hessian_pattern()
        .unwrap();
    let coloring = scioxide_sparse::derivative_coloring(pattern.view()).unwrap();
    let entries = pattern.view().col_indices().len();
    let (mut admitted, mut rejected) = (0, 0);
    let mut limit = 256;
    while limit < 1 << 22 {
        // `bounded`: whether a rejected transform must also stay within the limit.
        let mut tally = |name: &str, outcome: Result<(), SciOxideError>, bytes: usize, bounded| {
            let ok = outcome.is_ok();
            match outcome {
                Ok(()) => admitted += 1,
                Err(error) => {
                    assert!(
                        is_resource_limit(&error),
                        "limit {limit}, {name}: {error:?}"
                    );
                    rejected += 1;
                }
            }
            if ok || bounded {
                assert!(bytes <= limit, "limit {limit}: {bytes} bytes after {name}");
            }
        };
        let options = TapeOptions::new().with_byte_limit(Some(limit));
        let mut tape = Tape::<f64>::new(options);
        let mut gradient = vec![0.0; n];
        let mut values = vec![0.0; n];
        let mut jacobian = vec![0.0; n * n];
        let outcome = tape
            .gradient_into(|x| Ok(rosenbrock(x)), &point, &mut gradient, cx())
            .map(|_| ());
        tally("gradient_into", outcome, tape.reserved_bytes(), true);
        let outcome = tape
            .linearize(broyden_outputs, &point, cx())
            .and_then(|mut linear| {
                linear.vjp_into(&point, &mut gradient, cx())?;
                linear.jvp_into(&point, &mut values, cx())?;
                linear.jacobian_into(&mut jacobian, cx())
            });
        tally(
            "linearize and products",
            outcome,
            tape.reserved_bytes(),
            true,
        );
        let mut second = Tape::<Dual<f64>>::new(options);
        let outcome = second
            .hvp_into(|x| Ok(rosenbrock(x)), &point, &point, &mut values, cx())
            .map(|_| ());
        tally("hvp_into", outcome, second.reserved_bytes(), false);
        let mut second = Tape::<Dual<f64>>::new(options);
        let mut hessian = vec![0.0; entries];
        let outcome = second.sparse_hessian_into(
            |x| Ok(rosenbrock(x)),
            &point,
            &coloring,
            &mut hessian,
            cx(),
        );
        tally(
            "sparse_hessian_into",
            outcome,
            second.reserved_bytes(),
            false,
        );
        limit = limit * 5 / 4;
    }
    assert!(admitted > 0 && rejected > 0, "{admitted} {rejected}");
}

#[test]
fn cancellation_counts_argument_words_and_stops_short_transforms_before_they_start() {
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = ExecutionContext::sequential().with_cancellation(&token);
    // Eight chained statements of 600 arguments: about 4,800 words in eight
    // statements, fewer than 1,024 statements in all.
    let partials = [1.0; 600];
    let mut tape = Tape::<f64>::default();
    let mut linear = tape
        .linearize(
            |x, y| {
                let mut arguments: Vec<_> = (0..partials.len()).map(|k| x[k % 2]).collect();
                for _ in 0..8 {
                    arguments[0] = Var::combine(&arguments, 1.0, &partials)?;
                }
                y.push(arguments[0]);
                Ok(())
            },
            &[0.5, 0.25],
            cx(),
        )
        .unwrap();
    let mut gradient = [7.0; 2];
    let error = linear
        .vjp_into(&[1.0], &mut gradient, cancelled)
        .unwrap_err();
    assert!(is_stopped(&error), "{error:?}");
    assert_eq!(gradient, [7.0; 2]);
    let mut tangent = [7.0];
    let error = linear
        .jvp_into(&[1.0, 1.0], &mut tangent, cancelled)
        .unwrap_err();
    assert!(is_stopped(&error), "{error:?}");
    assert_eq!(tangent, [7.0]);
    // The first statement takes x₀ 300 and x₁ 300 times, every later one x₀ 299 and
    // x₁ 300 times besides the previous statement.
    linear.vjp_into(&[1.0], &mut gradient, cx()).unwrap();
    assert_close(
        &gradient,
        &[300.0 + 7.0 * 299.0, 300.0 + 7.0 * 300.0],
        "gradient",
    );

    // Short transforms check the context before recording.
    let mut gradient = [7.0; 2];
    let error = Tape::<f64>::default()
        .gradient_into(|x| Ok(x[0] * x[1]), &[1.0, 2.0], &mut gradient, cancelled)
        .unwrap_err();
    assert!(is_stopped(&error), "{error:?}");
    assert_eq!(gradient, [7.0; 2]);
    let mut product = [7.0; 2];
    let error = Tape::<Dual<f64>>::default()
        .hvp_into(
            |x| Ok(x[0] * x[1]),
            &[1.0, 2.0],
            &[1.0, 0.0],
            &mut product,
            cancelled,
        )
        .unwrap_err();
    assert!(is_stopped(&error), "{error:?}");
    assert_eq!(product, [7.0; 2]);
}

#[test]
fn failed_recordings_keep_the_requirements_of_the_last_completed_one() {
    let point = [0.5, 0.9];
    let mut tape = Tape::<f64>::default();
    let mut reference = [0.0; 2];
    tape.gradient_into(|x| Ok(sine_chain(x, 100)), &point, &mut reference, cx())
        .unwrap();
    let requirements = tape.requirements();
    let words = tape.report().words();
    let mut g = [5.0; 2];
    // Within and beyond the storage of the last recording.
    for steps in [10, 1000] {
        let error = tape
            .gradient_into(
                |x| {
                    let _ = sine_chain(x, steps);
                    Err(rejection())
                },
                &point,
                &mut g,
                cx(),
            )
            .unwrap_err();
        assert_eq!(error, rejection());
        assert_eq!(tape.requirements(), requirements, "{steps} steps");
        assert_eq!(tape.report().words(), words, "{steps} steps");
    }
    let error = tape
        .linearize(
            |x, y| {
                y.extend_from_slice(x);
                y.extend_from_slice(x);
                Err(rejection())
            },
            &point,
            cx(),
        )
        .unwrap_err();
    assert_eq!(error, rejection());
    assert_eq!(tape.requirements(), requirements);
    assert_eq!(g, [5.0; 2]);
    tape.gradient_into(|x| Ok(sine_chain(x, 100)), &point, &mut g, cx())
        .unwrap();
    assert_eq!(to_bits(&g), to_bits(&reference));
}

/// `2·x₀·x₁ + 3·x₀·x₂`: its Hessian has entries (0, 1), (0, 2), (1, 0) and (2, 0).
fn bilinear<R: Real>(x: &[R]) -> R {
    x[0] * x[1] * 2.0 + x[0] * x[2] * 3.0
}

fn assert_coloring_rejected(error: &SciOxideError) {
    match error.view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => {
            assert_eq!(argument.as_str(), "coloring");
            assert_eq!(reason, InvalidInputReason::UnsupportedConfiguration);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn direct_recovery_rejects_distance_one_colorings() {
    let point = [1.0, 2.0, 3.0];
    let pattern = Tape::<f64>::default()
        .linearize(
            |x, y| {
                y.push(bilinear(x));
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap()
        .hessian_pattern()
        .unwrap();
    assert_eq!(pattern.view().row_ptr(), [0, 2, 3, 4]);
    assert_eq!(pattern.view().col_indices(), [1, 2, 0, 0]);
    // Columns 1 and 2 are not adjacent, so a distance-1 coloring joins them, but
    // they share row 0 and their entries there would add up.
    let distance_one =
        ColoringSchedule::new(pattern.clone(), ColoringKind::DistanceOne, vec![0, 1, 1]).unwrap();
    let mut second = Tape::<Dual<f64>>::default();
    let mut values = [7.0; 4];
    let error = second
        .sparse_hessian_into(
            |x| Ok(bilinear(x)),
            &point,
            &distance_one,
            &mut values,
            cx(),
        )
        .unwrap_err();
    assert_coloring_rejected(&error);
    assert_eq!(values, [7.0; 4]);
    let coloring = scioxide_sparse::derivative_coloring(pattern.view()).unwrap();
    second
        .sparse_hessian_into(|x| Ok(bilinear(x)), &point, &coloring, &mut values, cx())
        .unwrap();
    assert_close(&values, &[2.0, 3.0, 2.0, 3.0], "Hessian");

    // y = (2·x₁ + 3·x₂, x₁, x₂): a square Jacobian pattern of the same shape.
    let mut tape = Tape::<f64>::default();
    let mut linear = tape
        .linearize(
            |x, y| {
                y.push(x[1] * 2.0 + x[2] * 3.0);
                y.push(x[1]);
                y.push(x[2]);
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let pattern = linear.jacobian_pattern().unwrap();
    assert_eq!(pattern.view().row_ptr(), [0, 2, 3, 4]);
    assert_eq!(pattern.view().col_indices(), [1, 2, 1, 2]);
    let distance_one =
        ColoringSchedule::new(pattern.clone(), ColoringKind::DistanceOne, vec![0, 1, 1]).unwrap();
    let mut values = [7.0; 4];
    let error = linear
        .sparse_jacobian_into(&distance_one, &mut values, cx())
        .unwrap_err();
    assert_coloring_rejected(&error);
    assert_eq!(values, [7.0; 4]);
    let coloring = scioxide_sparse::derivative_coloring(pattern.view()).unwrap();
    linear
        .sparse_jacobian_into(&coloring, &mut values, cx())
        .unwrap();
    assert_close(&values, &[2.0, 3.0, 1.0, 1.0], "Jacobian");
}

/// `s ← s·p`, counting its steps.
struct CountedGrowth {
    steps: AtomicUsize,
}

impl LoopBody for CountedGrowth {
    fn step<R: Real>(&self, _: usize, s: &mut [R], p: &[R]) -> Result<(), SciOxideError> {
        self.steps.fetch_add(1, Ordering::Relaxed);
        s[0] *= p[0];
        Ok(())
    }
}

#[test]
fn calls_whose_output_adjoints_are_zero_are_not_pulled_back() {
    let body = Arc::new(CountedGrowth {
        steps: AtomicUsize::new(0),
    });
    let schedule = CheckpointSchedule::binomial(100, 2).unwrap();
    let point = [1.0, 1.001, 0.5];
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut linear = tape
        .linearize(
            |x, y| {
                let mut s = [x[0]];
                checkpointed_loop(&body, 100, &mut s, &x[1..2], &schedule)?;
                y.push(s[0]);
                y.push(x[2] * x[2]);
                Ok(())
            },
            &point,
            cx(),
        )
        .unwrap();
    let recorded = body.steps.load(Ordering::Relaxed);
    let mut g = [9.0; 3];
    linear.vjp_into(&[0.0, 1.0], &mut g, cx()).unwrap();
    assert_eq!(
        body.steps.load(Ordering::Relaxed) - recorded,
        0,
        "steps run by a pullback with a zero cotangent"
    );
    assert_close(&g, &[0.0, 0.0, 1.0], "zero cotangent on the loop");
    // A non-zero cotangent runs the body: s = x₀·x₁¹⁰⁰.
    linear.vjp_into(&[1.0, 0.0], &mut g, cx()).unwrap();
    assert!(body.steps.load(Ordering::Relaxed) > recorded);
    let p: f64 = 1.001;
    assert_close(&g, &[p.powi(100), 100.0 * p.powi(99), 0.0], "loop gradient");
}

#[test]
fn linearization_products_check_cancellation_before_every_sweep() {
    // A 3-output recording of a few words: no sweep reaches 1,024 words.
    let token = CancellationToken::new();
    token.cancel();
    let stopped = cx().with_cancellation(&token);
    let mut tape = Tape::<f64>::default();
    let mut linear = tape
        .linearize(
            |x, y| {
                y.extend([x[0] * x[1], x[1].sin(), x[0] + x[2]]);
                Ok(())
            },
            &[0.5, 1.5, -2.0],
            cx(),
        )
        .unwrap();
    let stop = |error: SciOxideError| {
        assert!(
            matches!(error.view(), SciOxideErrorView::ExecutionStopped { .. }),
            "{error}"
        );
    };
    let mut out = [7.0; 9];
    stop(linear.jacobian_into(&mut out, stopped).unwrap_err());
    assert!(out.iter().all(|&v| v == 7.0));
    let mut gradient = [7.0; 3];
    stop(
        linear
            .vjp_into(&[1.0, 0.0, 0.0], &mut gradient, stopped)
            .unwrap_err(),
    );
    let mut tangent = [7.0; 3];
    stop(
        linear
            .jvp_into(&[1.0, 0.0, 0.0], &mut tangent, stopped)
            .unwrap_err(),
    );
    assert!(gradient.iter().chain(&tangent).all(|&v| v == 7.0));
    // The same products succeed without a stop request.
    linear.jacobian_into(&mut out, cx()).unwrap();
    assert_eq!(out[0], 1.5);
}

#[test]
fn call_pullbacks_keep_nonzero_derivative_components_of_zero_adjoints() {
    struct Identity;
    impl<V: TapeValue> ExternalRule<V> for Identity {
        fn evaluate(&self, x: &[V], y: &mut [V]) -> Result<(), SciOxideError> {
            y.copy_from_slice(x);
            Ok(())
        }
        fn pullback(&self, _: &[V], _: &[V], w: &[V], g: &mut [V]) -> Result<(), SciOxideError> {
            for (g, &w) in g.iter_mut().zip(w) {
                *g += w;
            }
            Ok(())
        }
    }
    fn check<V: TapeValue + PartialEq>(seed: V) {
        let rule: Arc<dyn ExternalRule<V>> = Arc::new(Identity);
        let mut tape = Tape::<V>::default();
        let mut linear = tape
            .linearize(
                |x, y| {
                    y.resize(1, Var::zero());
                    Var::external(&rule, x, y)
                },
                &[V::one()],
                cx(),
            )
            .unwrap();
        let mut out = [V::zero()];
        linear.vjp_into(&[seed], &mut out, cx()).unwrap();
        assert_eq!(out[0], seed);
    }
    check(Dual::new(0.0_f64, 1.0));
    check(Dual::new(
        Lanes::<f32, 2>::splat(0.0),
        Lanes::from_array([1.0, 0.0]),
    ));
    check(scioxide_ad::MultiDual::new(0.0_f64, [0.0, 1.0]));
    check(scioxide_ad::Jet::new([0.0_f64, 0.0, 1.0]));
    check(Dual::new(Dual::new(0.0_f64, 0.0), Dual::new(0.0, 1.0)));

    let mut tape = Tape::<Dual<f64>>::default();
    let mut out = [19.0];
    tape.hvp_into(
        |x| {
            let mut y = [Var::zero()];
            solve_into(&[Var::one()], 1, x, 1, &mut y)?;
            Ok(y[0] * y[0])
        },
        &[0.0],
        &[1.0],
        &mut out,
        cx(),
    )
    .unwrap();
    assert!((out[0] - 2.0).abs() < 1e-14);
}

#[test]
fn empty_products_and_short_hvp_observe_cancellation() {
    let token = CancellationToken::new();
    token.cancel();
    let stopped = cx().with_cancellation(&token);
    for mode in [
        JacobianMode::Auto,
        JacobianMode::Forward,
        JacobianMode::Reverse,
    ] {
        for (n, m) in [(0, 0), (0, 1), (1, 0)] {
            let mut tape = Tape::<f64>::new(TapeOptions::new().with_jacobian_mode(mode));
            let point = vec![1.0; n];
            let mut linear = tape
                .linearize(
                    |_, y| {
                        y.resize(m, Var::constant(1.0));
                        Ok(())
                    },
                    &point,
                    cx(),
                )
                .unwrap();
            assert!(matches!(
                linear.jacobian_into(&mut [], stopped).unwrap_err().view(),
                SciOxideErrorView::ExecutionStopped { .. }
            ));
            let pattern = linear.jacobian_pattern().unwrap();
            let coloring = scioxide_sparse::derivative_coloring(pattern.view()).unwrap();
            assert!(matches!(
                linear
                    .sparse_jacobian_into(&coloring, &mut [], stopped)
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::ExecutionStopped { .. }
            ));
        }
    }
    let token = CancellationToken::new();
    let mut tape = Tape::<Dual<f64>>::default();
    let mut product = [19.0];
    let result = tape.hvp_into(
        |x| {
            token.cancel();
            Ok(x[0] * x[0])
        },
        &[1.0],
        &[1.0],
        &mut product,
        cx().with_cancellation(&token),
    );
    assert!(matches!(
        result.unwrap_err().view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(product, [19.0]);
    tape.hvp_into(|x| Ok(x[0] * x[0]), &[1.0], &[1.0], &mut product, cx())
        .unwrap();
    assert!((product[0] - 2.0).abs() < 1e-14);
}

#[test]
fn scalar_maps_record_their_second_derivative_when_forward_types_wrap_vars() {
    let rule = |x: f64| [x * x * x, 3.0 * x * x, 6.0 * x];
    let mut tape = Tape::<f64>::default();
    for mode in 0..3 {
        let mut g = [19.0];
        tape.gradient_into(
            |x| {
                Ok(match mode {
                    0 => Dual::variable(x[0]).map_scalar(rule).derivative(),
                    1 => scioxide_ad::MultiDual::<_, 2>::variable(x[0], 1)?
                        .map_scalar(rule)
                        .gradient()[1],
                    _ => scioxide_ad::Jet::<_, 2>::variable(x[0])
                        .map_scalar(rule)
                        .coefficients()[1],
                })
            },
            &[2.0],
            &mut g,
            cx(),
        )
        .unwrap();
        assert!((g[0] - 12.0).abs() < 1e-14);
    }
    let g = scioxide_ad::grad(|x| Ok(Dual::variable(x[0]).sigmoid().derivative()), &[0.4]).unwrap();
    let e = (-0.4_f64).exp();
    let expected = -0.2_f64.tanh() * e / (1.0 + e).powi(2);
    assert!((g.as_slice()[0] - expected).abs() < 1e-14);
}

#[test]
fn hessian_operator_restores_its_tape_after_a_panicking_callback() {
    let should_panic = AtomicBool::new(false);
    let mut tape = Tape::<Dual<f64>>::default();
    let operator = tape
        .hessian_operator(
            |x| {
                assert!(
                    !should_panic.swap(false, Ordering::Relaxed),
                    "callback panic"
                );
                Ok(x[0] * x[0])
            },
            &[1.0],
            cx(),
        )
        .unwrap();
    let mut out = [19.0];
    should_panic.store(true, Ordering::Relaxed);
    assert!(
        catch_unwind(AssertUnwindSafe(|| operator.apply_into(
            &[1.0],
            &mut out,
            &mut []
        )))
        .is_err()
    );
    assert_eq!(out, [19.0]);
    operator.apply_into(&[1.0], &mut out, &mut []).unwrap();
    assert!((out[0] - 2.0).abs() < 1e-14);
}

#[test]
fn an_empty_hessian_still_validates_the_function() {
    let error =
        scioxide_ad::hessian::<f64, _>(|_| Err(SciOxideError::singular(Some(0))), &[]).unwrap_err();
    assert!(matches!(error.view(), SciOxideErrorView::Singular { .. }));
    let h = scioxide_ad::hessian::<f64, _>(|_| Ok(Var::constant(2.0)), &[]).unwrap();
    assert_eq!((h.rows(), h.cols()), (0, 0));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn hessian_size_overflow_precedes_callback() {
    let point = vec![0.0_f64; 65_536];
    let error = scioxide_ad::hessian(
        |_| panic!("an overflowing Hessian must fail before recording"),
        &point,
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn infinite_rounding_arguments_are_not_integer_kinks() {
    let mut tape = Tape::<f64>::new(TapeOptions::new().with_kink_policy(KinkPolicy::Error));
    for point in [f64::NEG_INFINITY, f64::INFINITY] {
        for operation in 0..4 {
            let mut gradient = [19.0];
            tape.gradient_into(
                |x| {
                    Ok(match operation {
                        0 => x[0].floor(),
                        1 => x[0].ceil(),
                        2 => x[0].trunc(),
                        _ => x[0].fract(),
                    })
                },
                &[point],
                &mut gradient,
                cx(),
            )
            .unwrap();
            assert_eq!(tape.report().kinks(), 0);
        }
    }
}
