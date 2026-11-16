//! Analytic Taylor series, mixed coefficients by nesting, and the `jet` transform.
#![allow(clippy::unwrap_used, clippy::panic)]

mod support;

use scioxide_ad::{Dual, Jet, Lanes, Real, jet};
use scioxide_core::{RealScalar, SciOxideErrorKind};
use support::{bits, rows};

fn series<S: RealScalar, const N: usize>(coefficients: [f64; N]) -> Jet<S, N> {
    Jet::new(coefficients.map(S::from_f64_ieee))
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "c{k}: {a} != {e}");
    }
}

#[test]
fn integer_powers_of_polynomials_are_exact() {
    // (2 + 3t + 5t²)³, expanded by integer arithmetic.
    let expected = [8.0, 36.0, 114.0, 207.0, 285.0, 225.0, 125.0];
    let x = series::<f64, 7>([2.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(x.powi(3).coefficients(), expected);
    let x = series::<f32, 7>([2.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(x.powi(3).coefficients(), expected.map(|c| c as f32));
    assert_eq!(
        x.powf(Jet::constant(3.0)).coefficients()[..3],
        [8.0, 36.0, 114.0]
    );
}

#[test]
fn high_orders_do_not_form_factorials() {
    // 1/(1 + t) has coefficients (−1)ᵏ, also where k! overflows.
    let alternating = |k: usize| if k.is_multiple_of(2) { 1.0 } else { -1.0 };
    let c = Jet::<f64, 181>::variable(1.0).recip().coefficients();
    assert!(c.iter().enumerate().all(|(k, &v)| v == alternating(k)));
    let c = Jet::<f32, 41>::variable(1.0).recip().coefficients();
    assert!(c.iter().all(|v| v.abs() == 1.0));
}

#[test]
fn elementary_coefficients_match_closed_series() {
    let t = |x: f64| Jet::<f64, 7>::variable(x);
    let (e, f) = (1.0 / 6.0, 1.0 / 120.0);
    for (name, actual, expected) in [
        (
            "exp",
            t(0.0).exp(),
            [1.0, 1.0, 0.5, e, 1.0 / 24.0, f, 1.0 / 720.0],
        ),
        (
            "exp_m1",
            t(0.0).exp_m1(),
            [0.0, 1.0, 0.5, e, 1.0 / 24.0, f, 1.0 / 720.0],
        ),
        (
            "ln",
            t(1.0).ln(),
            [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25, 0.2, -e],
        ),
        (
            "ln_1p",
            t(0.0).ln_1p(),
            [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25, 0.2, -e],
        ),
        ("sin", t(0.0).sin(), [0.0, 1.0, 0.0, -e, 0.0, f, 0.0]),
        (
            "cos",
            t(0.0).cos(),
            [1.0, 0.0, -0.5, 0.0, 1.0 / 24.0, 0.0, -1.0 / 720.0],
        ),
        (
            "tan",
            t(0.0).tan(),
            [0.0, 1.0, 0.0, 1.0 / 3.0, 0.0, 2.0 / 15.0, 0.0],
        ),
        ("sinh", t(0.0).sinh(), [0.0, 1.0, 0.0, e, 0.0, f, 0.0]),
        (
            "cosh",
            t(0.0).cosh(),
            [1.0, 0.0, 0.5, 0.0, 1.0 / 24.0, 0.0, 1.0 / 720.0],
        ),
        (
            "tanh",
            t(0.0).tanh(),
            [0.0, 1.0, 0.0, -1.0 / 3.0, 0.0, 2.0 / 15.0, 0.0],
        ),
        (
            "asin",
            t(0.0).asin(),
            [0.0, 1.0, 0.0, e, 0.0, 3.0 / 40.0, 0.0],
        ),
        (
            "atan",
            t(0.0).atan(),
            [0.0, 1.0, 0.0, -1.0 / 3.0, 0.0, 0.2, 0.0],
        ),
        (
            "asinh",
            t(0.0).asinh(),
            [0.0, 1.0, 0.0, -e, 0.0, 3.0 / 40.0, 0.0],
        ),
        (
            "atanh",
            t(0.0).atanh(),
            [0.0, 1.0, 0.0, 1.0 / 3.0, 0.0, 0.2, 0.0],
        ),
        (
            "sigmoid",
            t(0.0).sigmoid(),
            [0.5, 0.25, 0.0, -1.0 / 48.0, 0.0, 1.0 / 480.0, 0.0],
        ),
        (
            "softplus",
            t(0.0).softplus(),
            [2f64.ln(), 0.5, 0.125, 0.0, -1.0 / 192.0, 0.0, 1.0 / 2880.0],
        ),
        (
            "sqrt",
            t(1.0).sqrt(),
            [
                1.0,
                0.5,
                -0.125,
                0.0625,
                -5.0 / 128.0,
                7.0 / 256.0,
                -21.0 / 1024.0,
            ],
        ),
        (
            "cbrt",
            t(1.0).cbrt(),
            [
                1.0,
                1.0 / 3.0,
                -1.0 / 9.0,
                5.0 / 81.0,
                -10.0 / 243.0,
                22.0 / 729.0,
                -154.0 / 6561.0,
            ],
        ),
        (
            "exp2",
            t(0.0).exp2(),
            core::array::from_fn(|k| 2f64.ln().powi(k as i32) / (1..=k).product::<usize>() as f64),
        ),
    ] {
        for (k, (a, b)) in actual.coefficients().into_iter().zip(expected).enumerate() {
            assert!((a - b).abs() < 1e-14, "{name} c{k}: {a} != {b}");
        }
    }
}

#[test]
fn second_coefficients_are_halved_second_derivatives() {
    let y = Jet::<f64, 3>::variable(2.0).powi(3);
    assert_eq!(y.coefficients(), [8.0, 12.0, 6.0]);
    let y = Jet::<f64, 3>::variable(1.0).exp();
    let e = 1f64.exp();
    assert_close(&y.coefficients(), &[e, e, e / 2.0], 1e-15);
    let y = Jet::<f64, 3>::variable(0.5).sin();
    let (s, c) = 0.5_f64.sin_cos();
    assert_close(&y.coefficients(), &[s, c, -s / 2.0], 1e-16);
}

fn signed_zero_function<R: Real>(name: &str, x: R) -> R {
    match name {
        "tanh" => x.tanh(),
        "atan" => x.atan(),
        "asin" => x.asin(),
        "asinh" => x.asinh(),
        "atanh" => x.atanh(),
        "cbrt" => x.cbrt(),
        "sin_pi" => x.sin_pi(),
        "sin_degrees" => x.sin_degrees(),
        "sigmoid" => x.sigmoid(),
        "softplus" => x.softplus(),
        "log_sigmoid" => x.log_sigmoid(),
        _ => panic!("unknown function {name}"),
    }
}

#[test]
fn shortest_jets_keep_the_primal_and_the_first_order_rule() {
    assert_eq!(Jet::<f64, 1>::variable(0.0).abs().coefficients(), [0.0]);
    assert_eq!(
        Jet::<f64, 2>::variable(0.0).tanh().coefficients(),
        [0.0, 1.0]
    );
    let c = Jet::<f64, 3>::variable(1e100).powi(-2).coefficients();
    assert!((c[0] / 1e-200 - 1.0).abs() < 1e-14);
    assert!((c[1] / -2e-300 - 1.0).abs() < 1e-14);
    assert_eq!(c[2], 0.0);
    for name in [
        "tanh",
        "atan",
        "asin",
        "asinh",
        "atanh",
        "cbrt",
        "sin_pi",
        "sin_degrees",
        "sigmoid",
        "softplus",
        "log_sigmoid",
    ] {
        for value in [0.0_f64, -0.0] {
            let primal = signed_zero_function(name, value).to_bits();
            let short = signed_zero_function(name, Jet::<f64, 1>::variable(value));
            let long = signed_zero_function(name, Jet::<f64, 9>::variable(value));
            assert_eq!(short.coefficients()[0].to_bits(), primal, "{name}");
            assert_eq!(long.coefficients()[0].to_bits(), primal, "{name}");
        }
    }
}

#[test]
fn integer_powers_at_a_zero_base_use_repeated_products() {
    let t = Jet::<f64, 3>::variable(0.0);
    assert_eq!(t.powi(0).coefficients(), [1.0, 0.0, 0.0]);
    assert_eq!(t.powi(2).coefficients(), [0.0, 0.0, 1.0]);
    // p(p − 1)/2 at −1, exactly representable in binary64.
    let c = Jet::<f64, 3>::variable(-1.0).powi(i32::MIN).coefficients();
    assert_eq!(c, [1.0, 2_147_483_648.0, 2_305_843_010_287_435_776.0]);
    let c = Jet::<f64, 5>::variable(0.0)
        .powf(Jet::constant(2.0))
        .coefficients();
    assert_eq!(c, [0.0, 0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn constant_exponents_admit_negative_and_zero_bases() {
    // (−2 + t)³ = −8 + 12t − 6t² + t³.
    let c = Jet::<f64, 5>::variable(-2.0)
        .powf(Jet::constant(3.0))
        .coefficients();
    assert_close(&c, &[-8.0, 12.0, -6.0, 1.0, 0.0], 1e-14);
    // t^2.5: zero below the exponent, NaN from it on.
    let c = Jet::<f64, 5>::variable(0.0)
        .powf(Jet::constant(2.5))
        .coefficients();
    assert_eq!(c[..3], [0.0, 0.0, 0.0]);
    assert!(c[3].is_nan() && c[4].is_nan());
    // A varying exponent needs ln x.
    let c = Jet::<f64, 3>::variable(-2.0)
        .powf(Jet::variable(3.0))
        .coefficients();
    assert_eq!(c[0], -8.0);
    assert!(c[1].is_nan() && c[2].is_nan());
    // x^y along (1, 1) at (1.5, 0.5): exp(y ln x) expanded independently.
    let x = Jet::<f64, 4>::variable(1.5);
    let y = Jet::<f64, 4>::variable(0.5);
    let reference = (y * x.ln()).exp().coefficients();
    assert_close(&x.powf(y).coefficients(), &reference, 1e-14);
    assert_close(&x.powm1(y).coefficients()[1..], &reference[1..], 1e-14);
}

#[test]
fn masked_logaddexp_takes_the_coefficients_of_the_larger_operand() {
    let inf = f64::INFINITY;
    for (x0, y0, expected) in [
        (-inf, 0.0, [0.0, 7.0, 11.0]),
        (0.0, -inf, [0.0, 3.0, 5.0]),
        (-inf, -inf, [-inf, 7.0, 11.0]),
    ] {
        let z = Jet::new([x0, 3.0, 5.0]).logaddexp(Jet::new([y0, 7.0, 11.0]));
        assert_eq!(z.coefficients(), expected);
    }
    // logaddexp(t, t) = t + ln 2.
    let t = Jet::<f64, 4>::variable(40.0);
    let expected = [40.0 + 2f64.ln(), 1.0, 0.0, 0.0];
    assert_close(&t.logaddexp(t).coefficients(), &expected, 1e-14);
}

#[test]
fn kinks_apply_the_first_order_rule_to_every_coefficient() {
    let x = Jet::new([-2.0, 1.0, 3.0]);
    assert_eq!(x.abs().coefficients(), [2.0, -1.0, -3.0]);
    assert_eq!(
        Jet::new([0.0, 1.0, 3.0]).abs().coefficients(),
        [0.0, 0.0, 0.0]
    );
    let y = Jet::new([-2.0, 3.0, -1.0]);
    assert_eq!(x.max(y).coefficients(), [-2.0, 2.0, 1.0]);
    assert_eq!(x.min(Jet::constant(0.0)).coefficients(), [-2.0, 1.0, 3.0]);
    let floor = Jet::new([2.5, 1.0, 1.0]).floor();
    assert_eq!(floor.coefficients(), [2.0, 0.0, 0.0]);
    assert_eq!(
        Jet::new([2.5, 1.0, 1.0]).fract().coefficients(),
        [0.5, 1.0, 1.0]
    );
    let clamped = Jet::new([3.0, 1.0, 1.0]).clamp(Jet::constant(0.0), Jet::new([1.0, 2.0, 4.0]));
    assert_eq!(clamped.coefficients(), [1.0, 2.0, 4.0]);
}

#[test]
fn map_scalar_and_combine_are_exact_to_their_orders() {
    let x = Jet::new([0.3_f64, 1.0, 0.5, 2.0]);
    let (s, c) = 0.3_f64.sin_cos();
    let y = x.map_scalar(|v| [v.sin(), v.cos(), -v.sin()]);
    let c2 = c * 0.5 - s * 0.5;
    assert_eq!(y.coefficients()[..3], [s, c, c2]);
    assert!(y.coefficients()[3].is_nan());
    for z in [x.exprel(), x.sinc(), x.sincpi()] {
        assert!(z.coefficients()[..3].iter().all(|c| c.is_finite()));
        assert!(z.coefficients()[3].is_nan());
    }
    let a = Jet::variable(0.5_f64);
    let b = Jet::new([2.0, 3.0, 0.0, 0.0]);
    let z = Jet::combine(&[a, b], 7.0, &[2.0, -1.0]).unwrap();
    assert_eq!(z.coefficients()[..2], [7.0, -1.0]);
    assert!(z.coefficients()[2..].iter().all(|c| c.is_nan()));
    let error = Jet::combine(&[a, b], 7.0, &[2.0]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let angle = a.atan2(b);
    assert_eq!(angle.coefficients()[0], 0.5_f64.atan2(2.0));
    assert!((angle.coefficients()[1] - (2.0 - 0.5 * 3.0) / 4.25).abs() < 1e-15);
    assert!(angle.coefficients()[2].is_nan());
}

#[test]
fn sigmoid_first_coefficients_keep_saturated_derivatives() {
    fn check<S: RealScalar>(tail: f64) {
        for value in [0.0, -0.0, 0.25, -0.25, 40.0, -40.0, tail, -tail] {
            let x = S::from_f64_ieee(value);
            let dual = Dual::variable(x).sigmoid().derivative();
            let jet = Jet::<S, 3>::variable(x).sigmoid();
            assert!(dual > S::ZERO);
            assert_eq!(bits(jet.coefficients()[1]), bits(dual));
        }
    }
    check::<f32>(100.0);
    check::<f64>(740.0);
}

fn logit_midpoint<S: RealScalar>(width: usize) {
    // Near x = 1/2 the curvature of logit is tiny; c₂ keeps it.
    for (row, fields) in rows(include_str!("data/stable_derivatives.txt")) {
        if fields.len() != 6 || fields[0] != width.to_string() || fields[1] != "logit" {
            continue;
        }
        let x = support::decode::<S>(fields[2], width);
        if !(0.49..=0.51).contains(&x.into_f64()) {
            continue;
        }
        let second = support::decode::<S>(fields[5], width);
        let c = Jet::<S, 3>::variable(x).logit().coefficients();
        support::close(c[2] * S::TWO, second, 64, row);
    }
}

#[test]
fn logit_keeps_small_curvature_near_one_half() {
    logit_midpoint::<f32>(32);
    logit_midpoint::<f64>(64);
}

#[test]
fn nested_jets_give_mixed_coefficients() {
    fn check<S: RealScalar>() {
        // x²y³ at (2, 3) with x varying in the outer and y in the inner variable.
        let x = Jet::<Jet<S, 3>, 3>::new([
            series([2.0, 0.0, 0.0]),
            series([1.0, 0.0, 0.0]),
            Jet::zero(),
        ]);
        let y = Jet::<Jet<S, 3>, 3>::lift(series([3.0, 1.0, 0.0]));
        let c = (x.powi(2) * y.powi(3))
            .coefficients()
            .map(Jet::coefficients);
        let expected = [
            [108.0, 108.0, 36.0],
            [108.0, 108.0, 36.0],
            [27.0, 27.0, 9.0],
        ];
        assert_eq!(c, expected.map(|row| row.map(S::from_f64_ieee)));
    }
    check::<f32>();
    check::<f64>();
    // x² with x = 1 + 2t + 3s + 5ts + 7t² + 11t²s: the coefficient of t²s is
    // 2·11 + 2·2·5 + 2·3·7 = 84.
    let x = Jet::<Jet<f64, 2>, 3>::new([
        Jet::new([1.0, 3.0]),
        Jet::new([2.0, 5.0]),
        Jet::new([7.0, 11.0]),
    ]);
    assert_eq!((x * x).coefficients()[2].coefficients()[1], 84.0);
}

#[test]
fn deep_mixed_orders_avoid_factorials() {
    // 1/(1 + t + s): the coefficient of t¹⁸⁰s is −181.
    let x = Jet::<Jet<f64, 2>, 181>::new(core::array::from_fn(|k| match k {
        0 => Jet::variable(1.0),
        1 => Jet::one(),
        _ => Jet::zero(),
    }));
    assert_eq!(x.recip().coefficients()[180].coefficients()[1], -181.0);
    // exp((s₁ + … + s₆)/2) at 0: the coefficient of s₁⋯s₆ is 2⁻⁶.
    let x: Six = (0..6).map(variable6).sum();
    let all = (x * 0.5).exp().coefficients()[1].coefficients()[1].coefficients()[1];
    let all = all.coefficients()[1].coefficients()[1].coefficients()[1];
    assert!((all - 0.015625).abs() <= 128.0 * f64::EPSILON * 0.015625);
}

/// Six nested jets of length 2, one per variable.
type Six = Jet<Jet<Jet<Jet<Jet<Jet<f64, 2>, 2>, 2>, 2>, 2>, 2>;

/// The variable of nesting depth `depth` (0 outermost) in six nested jets.
fn variable6(depth: usize) -> Six {
    fn level<V: Real<f64>>(seeded: bool, inner: V) -> Jet<V, 2> {
        if seeded {
            Jet::new([inner, V::one()])
        } else {
            Jet::lift(inner)
        }
    }
    let l5 = level(depth == 5, 0.0);
    let l4 = level(depth == 4, l5);
    let l3 = level(depth == 3, l4);
    let l2 = level(depth == 2, l3);
    let l1 = level(depth == 1, l2);
    level(depth == 0, l1)
}

#[test]
fn jets_over_lanes_expand_every_lane() {
    // Lane kernels may differ from the scalar functions in the last ulps.
    let x = Lanes::<f64, 4>::from_array([0.25, 1.5, -0.75, 3.0]);
    let jet = Jet::<Lanes<f64, 4>, 5>::variable(x);
    let y = (jet.sin() * jet.sqrt().exp() + jet.powi(3)).coefficients();
    for lane in 0..4 {
        let t = Jet::<f64, 5>::variable(x.lane(lane).unwrap());
        let scalar = (t.sin() * t.sqrt().exp() + t.powi(3)).coefficients();
        for (wide, single) in y.iter().zip(scalar) {
            let wide = wide.lane(lane).unwrap();
            assert!(
                (wide - single).abs() <= 1e-14 * single.abs() || (wide.is_nan() && single.is_nan())
            );
        }
    }
}

#[test]
fn jet_expands_along_a_line() {
    // f(x, y) = x·eʸ along (1, 0) + t(1, 1): (1 + t)eᵗ = Σ (k + 1)/k! tᵏ.
    let c: [f64; 6] = jet(|x| Ok(x[0] * x[1].exp()), &[1.0, 0.0], &[1.0, 1.0]).unwrap();
    assert_close(
        &c,
        &[1.0, 2.0, 1.5, 4.0 / 6.0, 5.0 / 24.0, 6.0 / 120.0],
        1e-15,
    );
    let error = jet::<f64, _, 3>(|x| Ok(x[0]), &[1.0], &[]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let empty: [f32; 2] = jet(
        |x| Ok(x.iter().copied().sum::<Jet<f32, 2>>() + 2.0),
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(empty, [2.0, 0.0]);
    let single: [f64; 1] = jet(|x| Ok(x[0].exp()), &[0.0], &[1.0]).unwrap();
    assert_eq!(single, [1.0]);
    let failure = jet::<f64, _, 2>(
        |_| Err(scioxide_core::SciOxideError::singular(None)),
        &[1.0],
        &[1.0],
    );
    assert_eq!(failure.unwrap_err().kind(), SciOxideErrorKind::Singular);
}
