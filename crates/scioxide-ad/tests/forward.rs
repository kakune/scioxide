//! Forward types agree with `Dual`: `MultiDual` gradients bit for bit, `Jet` first
//! coefficients up to the sign of zero and second coefficients within 8 ulps of
//! nested duals; plus the `MultiDual` contract and `jvp`.
#![allow(clippy::unwrap_used, clippy::panic)]

mod support;

use scioxide_ad::{
    ComplexElementary, Dual, Jet, KinkPolicy, Lanes, MultiDual, Real, Tape, TapeOptions, jvp,
};
use scioxide_core::{Complex, ExecutionContext, SciOxideErrorView};
use scioxide_core::{RealScalar, SciOxideErrorKind};
use support::{bits, close, moderate};

/// Every unary method, integer power and scalar operand.
const UNARY: [&str; 59] = [
    "recip",
    "sqrt",
    "cbrt",
    "exp",
    "exp2",
    "exp_m1",
    "ln",
    "ln_1p",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "abs",
    "signum",
    "floor",
    "ceil",
    "round",
    "trunc",
    "fract",
    "to_degrees",
    "to_radians",
    "softplus",
    "sigmoid",
    "log_sigmoid",
    "logit",
    "exprel",
    "cosm1",
    "sinc",
    "sincpi",
    "sin_pi",
    "cos_pi",
    "tan_pi",
    "sin_degrees",
    "cos_degrees",
    "tan_degrees",
    "sin_cos.0",
    "sin_cos.1",
    "powi3",
    "powi-2",
    "powi0",
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "rem",
    "square",
    "inverse",
    "azmul",
];

fn unary<S: RealScalar, R: Real<S>>(name: &str, x: R) -> R {
    match name {
        "recip" => x.recip(),
        "sqrt" => x.sqrt(),
        "cbrt" => x.cbrt(),
        "exp" => x.exp(),
        "exp2" => x.exp2(),
        "exp_m1" => x.exp_m1(),
        "ln" => x.ln(),
        "ln_1p" => x.ln_1p(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "asinh" => x.asinh(),
        "acosh" => x.acosh(),
        "atanh" => x.atanh(),
        "abs" => x.abs(),
        "signum" => x.signum(),
        "floor" => x.floor(),
        "ceil" => x.ceil(),
        "round" => x.round(),
        "trunc" => x.trunc(),
        "fract" => x.fract(),
        "to_degrees" => x.to_degrees(),
        "to_radians" => x.to_radians(),
        "softplus" => x.softplus(),
        "sigmoid" => x.sigmoid(),
        "log_sigmoid" => x.log_sigmoid(),
        "logit" => x.logit(),
        "exprel" => x.exprel(),
        "cosm1" => x.cosm1(),
        "sinc" => x.sinc(),
        "sincpi" => x.sincpi(),
        "sin_pi" => x.sin_pi(),
        "cos_pi" => x.cos_pi(),
        "tan_pi" => x.tan_pi(),
        "sin_degrees" => x.sin_degrees(),
        "cos_degrees" => x.cos_degrees(),
        "tan_degrees" => x.tan_degrees(),
        "sin_cos.0" => x.sin_cos().0,
        "sin_cos.1" => x.sin_cos().1,
        "powi3" => x.powi(3),
        "powi-2" => x.powi(-2),
        "powi0" => x.powi(0),
        "neg" => -x,
        "add" => x + S::TWO,
        "sub" => x - S::TWO,
        "mul" => x * S::TWO,
        "div" => x / S::TWO,
        "rem" => x % S::TWO,
        "square" => x * x,
        "inverse" => R::one() / x,
        "azmul" => x.azmul(x),
        _ => panic!("unknown unary {name}"),
    }
}

/// Every binary method, operator, reduction and selection.
const BINARY: [&str; 21] = [
    "powf",
    "hypot",
    "atan2",
    "min",
    "max",
    "log",
    "logaddexp",
    "xlogy",
    "xlog1py",
    "powm1",
    "add",
    "sub",
    "mul",
    "div",
    "rem",
    "azmul",
    "clamp",
    "clamped",
    "sum",
    "product",
    "select",
];

fn binary<S: RealScalar, R: Real<S>>(name: &str, x: R, y: R) -> R {
    match name {
        "powf" => x.powf(y),
        "hypot" => x.hypot(y),
        "atan2" => x.atan2(y),
        "min" => x.min(y),
        "max" => x.max(y),
        "log" => x.log(y),
        "logaddexp" => x.logaddexp(y),
        "xlogy" => x.xlogy(y),
        "xlog1py" => x.xlog1py(y),
        "powm1" => x.powm1(y),
        "add" => x + y,
        "sub" => x - y,
        "mul" => x * y,
        "div" => x / y,
        "rem" => x % y,
        "azmul" => x.azmul(y),
        "clamp" => x.clamp(y, y + S::ONE),
        "clamped" => y.clamp(x - S::ONE, x),
        "sum" => [x, y, x * y].into_iter().sum(),
        "product" => [x, y, x].into_iter().product(),
        "select" => R::select(x.lt(y), x, y),
        _ => panic!("unknown binary {name}"),
    }
}

/// Fixed points away from the cancellation of the nested-dual second derivatives
/// (`(1 − x)(1 + x)` differentiates to `(1 − x) − (1 + x)`, which cancels near 0).
const POINTS: [f64; 9] = [0.3, 0.75, 1.5, 2.5, -0.6, -1.7, 0.0, 4.0, 0.5];

const PAIRS: [(f64, f64); 9] = [
    (0.4, 0.7),
    (1.3, 2.5),
    (2.5, 0.6),
    (-0.6, 1.5),
    (0.7, 0.7),
    (3.0, -2.0),
    (0.0, 1.0),
    (2.0, 0.0),
    (-2.0, 3.0),
];

/// Equal values, with `+0 = −0` and every NaN equal.
fn same<S: RealScalar>(a: S, b: S) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

fn seed2<S: RealScalar>(x: S, a: S, b: S) -> Dual<Dual<S>> {
    Dual::new(Dual::new(x, a), Dual::new(b, S::ZERO))
}

fn unary_agreement<S: RealScalar>() {
    for name in UNARY {
        for x in POINTS.map(S::from_f64_ieee) {
            let context = format!("{name}({x:?})");
            let value = unary(name, x);
            let dual = unary(name, Dual::variable(x));
            let multi = unary(name, MultiDual::<S, 2>::variable(x, 0).unwrap());
            let jet = unary(name, Jet::<S, 3>::variable(x));
            assert_eq!(bits(dual.value()), bits(value), "{context}");
            assert_eq!(bits(multi.value()), bits(value), "{context}");
            assert_eq!(bits(jet.coefficients()[0]), bits(value), "{context}");
            let derivative = dual.derivative();
            assert_eq!(bits(multi.gradient()[0]), bits(derivative), "{context}");
            let unseeded = unary(name, Dual::new(x, S::ZERO)).derivative();
            assert_eq!(bits(multi.gradient()[1]), bits(unseeded), "{context}");
            // In 1/x at 0 the infinite partial of the constant numerator meets its
            // zero tangent: NaN with the IEEE products of jets.
            let first = jet.coefficients()[1];
            if !(name == "inverse" && x == S::ZERO) {
                assert!(
                    same(first, derivative),
                    "{context}: {first:?} != {derivative:?}"
                );
            }
            let second = unary(name, seed2(x, S::ONE, S::ONE))
                .derivative()
                .derivative();
            if moderate(&[x, value, derivative, second]) {
                close(jet.coefficients()[2] * S::TWO, second, 8, &context);
            }
        }
    }
}

#[test]
fn unary_first_derivatives_agree_across_forward_types() {
    unary_agreement::<f32>();
    unary_agreement::<f64>();
}

fn binary_agreement<S: RealScalar>() {
    let (zero, one) = (S::ZERO, S::ONE);
    for name in BINARY {
        for (x, y) in PAIRS.map(|(x, y)| (S::from_f64_ieee(x), S::from_f64_ieee(y))) {
            let context = format!("{name}({x:?}, {y:?})");
            let value = binary(name, x, y);
            let dual = |a, b| binary(name, Dual::new(x, a), Dual::new(y, b)).derivative();
            let partials = [dual(one, zero), dual(zero, one)];
            let multi = binary(
                name,
                MultiDual::new(x, [one, zero]),
                MultiDual::new(y, [zero, one]),
            );
            assert_eq!(bits(multi.value()), bits(value), "{context}");
            for (actual, expected) in multi.gradient().into_iter().zip(partials) {
                assert_eq!(bits(actual), bits(expected), "{context}");
            }
            // Along an axis; along (1, 1) the second derivative
            // f_xx + 2f_xy + f_yy cancels and only the value is compared.
            for (a, b) in [(one, zero), (zero, one), (one, one)] {
                let jet = binary(name, Jet::new([x, a, zero]), Jet::new([y, b, zero]));
                assert_eq!(bits(jet.coefficients()[0]), bits(value), "{context}");
                if a == b {
                    continue;
                }
                let first = dual(a, b);
                // Jets use IEEE products: an infinite or NaN partial of the
                // unseeded operand times its zero tangent is NaN, where `Dual`
                // drops the term.
                let unseeded = if a == zero { partials[0] } else { partials[1] };
                if !unseeded.is_finite() && !first.is_nan() {
                    continue;
                }
                let actual = jet.coefficients()[1];
                assert!(same(actual, first), "{context}: {actual:?} != {first:?}");
                let second = binary(name, seed2(x, a, a), seed2(y, b, b))
                    .derivative()
                    .derivative();
                // atan2 is first order; a varying exponent at a zero base is not
                // analytic, and its jet coefficients above c₁ are NaN.
                let singular = matches!(name, "powf" | "powm1") && x == zero && b != zero;
                if name != "atan2" && !singular && moderate(&[x, y, value, first, second]) {
                    let context = format!("{context} along ({a:?}, {b:?})");
                    close(jet.coefficients()[2] * S::TWO, second, 8, &context);
                }
            }
        }
    }
}

#[test]
fn binary_first_derivatives_agree_across_forward_types() {
    binary_agreement::<f32>();
    binary_agreement::<f64>();
}

macro_rules! scalar_left {
    ($S:ty) => {{
        let x: $S = 0.4;
        let d = Dual::variable(x);
        let m = MultiDual::<$S, 1>::variable(x, 0).unwrap();
        let j = Jet::<$S, 2>::variable(x);
        for (dual, multi, jet) in [
            (2.0 + d, 2.0 + m, 2.0 + j),
            (2.0 - d, 2.0 - m, 2.0 - j),
            (2.0 * d, 2.0 * m, 2.0 * j),
            (2.0 / d, 2.0 / m, 2.0 / j),
            (1.1 % d, 1.1 % m, 1.1 % j),
        ] {
            assert_eq!(multi.value().to_bits(), dual.value().to_bits());
            assert_eq!(multi.gradient()[0].to_bits(), dual.derivative().to_bits());
            assert_eq!(jet.coefficients()[0].to_bits(), dual.value().to_bits());
            assert_eq!(jet.coefficients()[1], dual.derivative());
        }
    }};
}

#[test]
fn scalars_on_the_left_agree_across_forward_types() {
    scalar_left!(f32);
    scalar_left!(f64);
}

#[test]
fn multidual_carries_a_gradient() {
    // f = x₀x₁ + x₁x₂ + x₀x₂ at (1, 2, 3): ∇f = (5, 4, 3).
    let x: Vec<MultiDual<f64, 3>> = [1.0, 2.0, 3.0]
        .iter()
        .enumerate()
        .map(|(i, &v)| MultiDual::variable(v, i).unwrap())
        .collect();
    let f = x[0] * x[1] + x[1] * x[2] + x[0] * x[2];
    assert_eq!(f.value(), 11.0);
    assert_eq!(f.gradient(), [5.0, 4.0, 3.0]);
    assert_eq!(core::mem::size_of::<MultiDual<f64, 3>>(), 32);
    assert_eq!(core::mem::size_of::<Jet<f64, 3>>(), 24);
}

#[test]
fn variable_rejects_a_direction_past_the_gradient() {
    let error = MultiDual::<f64, 2>::variable(1.0, 2).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
    assert!(MultiDual::<f64, 0>::variable(1.0, 0).is_err());
    assert_eq!(
        MultiDual::<f64, 2>::variable(1.0, 1).unwrap().gradient(),
        [0.0, 1.0]
    );
}

#[test]
fn variables_seed_every_direction_like_variable() {
    let values = [1.5, -0.25, 3.0];
    let all = MultiDual::<f64, 3>::variables(values);
    for (i, x) in all.iter().enumerate() {
        let one = MultiDual::variable(values[i], i).unwrap();
        assert_eq!(x.value().to_bits(), one.value().to_bits());
        assert_eq!(
            x.gradient().map(f64::to_bits),
            one.gradient().map(f64::to_bits)
        );
    }
    // f = x₀·exp(x₁)·x₂: ∇f = (e^{x₁}x₂, x₀e^{x₁}x₂, x₀e^{x₁}).
    let [x, y, z] = all;
    let g = (x * y.exp() * z).gradient();
    let e = (-0.25_f64).exp();
    close(g[0], e * 3.0, 2, "∂x₀");
    close(g[1], 1.5 * e * 3.0, 2, "∂x₁");
    close(g[2], 1.5 * e, 2, "∂x₂");
    assert_eq!(MultiDual::<f64, 0>::variables([]), []);
}

#[test]
fn nested_multiduals_give_hessian_vector_products() {
    // f = x₀²x₁ at (2, 3): ∇f = (12, 4), H = [[6, 4], [4, 0]], H·(1, −1) = (2, 4).
    let v = [1.0, -1.0];
    let x: Vec<Dual<MultiDual<f64, 2>>> = [2.0, 3.0]
        .iter()
        .zip(v)
        .enumerate()
        .map(|(i, (&p, d))| Dual::new(MultiDual::variable(p, i).unwrap(), MultiDual::constant(d)))
        .collect();
    let f = x[0] * x[0] * x[1];
    assert_eq!(f.value().gradient(), [12.0, 4.0]);
    assert_eq!(f.derivative().gradient(), [2.0, 4.0]);
}

fn reductions<S: RealScalar>() {
    let empty: MultiDual<S, 2> = core::iter::empty().sum();
    assert_eq!(bits(empty.value()), bits(S::NEG_ZERO));
    let empty: MultiDual<S, 2> = core::iter::empty().product();
    assert_eq!(empty, MultiDual::one());
    let x = MultiDual::<S, 2>::variable(S::TWO, 0).unwrap();
    let y = MultiDual::variable(S::TWO + S::ONE, 1).unwrap();
    let sum: MultiDual<S, 2> = [x, y].into_iter().sum();
    let product: MultiDual<S, 2> = [x, y].into_iter().product();
    assert_eq!(sum, x + y);
    assert_eq!(product, x * y);
    let empty: Jet<S, 3> = core::iter::empty().sum();
    assert_eq!(bits(empty.coefficients()[0]), bits(S::NEG_ZERO));
    let empty: Jet<S, 3> = core::iter::empty().product();
    assert_eq!(empty, Jet::one());
    let t = Jet::<S, 3>::variable(S::TWO);
    let product: Jet<S, 3> = [t, t].into_iter().product();
    assert_eq!(product, t * t);
}

#[test]
fn reductions_start_from_negative_zero_and_one() {
    reductions::<f32>();
    reductions::<f64>();
}

fn remainder<S: RealScalar>() {
    let four = S::TWO * S::TWO;
    let half = S::HALF;
    // ∂(a % b) = (1, −trunc(a/b)); at the kink a/b = q ∈ ℤ \ {0} the partial in b is
    // the midpoint −(q − ½·sign q) of the one-sided −q and −(q − sign q).
    for (a, b, partial) in [
        (four, S::TWO, -(S::TWO - half)),
        (-four, S::TWO, S::TWO - half),
        (four + S::ONE, S::TWO, -S::TWO),
        (S::ONE, four, S::ZERO),
    ] {
        let dual = Dual::new(a, S::ONE) % Dual::new(b, S::TWO);
        let multi = MultiDual::new(a, [S::ONE, S::ZERO]) % MultiDual::new(b, [S::ZERO, S::ONE]);
        let jet = Jet::new([a, S::ONE, S::ONE]) % Jet::new([b, S::TWO, S::ZERO]);
        assert_eq!(dual.value(), a % b);
        assert_eq!(dual.derivative(), S::ONE + partial * S::TWO);
        assert_eq!(multi.gradient(), [S::ONE, partial]);
        assert_eq!(
            jet.coefficients(),
            [a % b, S::ONE + partial * S::TWO, S::ONE]
        );
    }
}

#[test]
fn remainder_differentiates_with_the_truncated_quotient() {
    remainder::<f32>();
    remainder::<f64>();
}

#[test]
fn kinks_take_the_midpoint_of_the_one_sided_derivatives() {
    let x = MultiDual::<f64, 2>::new(0.0, [1.0, -2.0]);
    assert_eq!(x.abs().gradient(), [0.0, 0.0]);
    let (a, b) = (
        MultiDual::new(1.0, [1.0, 0.0]),
        MultiDual::new(1.0, [0.0, 1.0]),
    );
    assert_eq!(a.max(b).gradient(), [0.5, 0.5]);
    assert_eq!(a.min(b).gradient(), [0.5, 0.5]);
    let nan = MultiDual::new(f64::NAN, [0.0, 1.0]);
    assert_eq!(a.max(nan).gradient(), [1.0, 0.0]);
    let clamped = MultiDual::new(3.0, [1.0, 0.0]).clamp(MultiDual::constant(0.0), b);
    assert_eq!((clamped.value(), clamped.gradient()), (1.0, [0.0, 1.0]));
    assert!(a.clamp(MultiDual::constant(2.0), b).value().is_nan());
}

#[test]
fn clamped_square_root_has_a_zero_derivative_below_zero() {
    let v = -1.0_f64;
    let dual = Dual::variable(v).max(Dual::zero()).sqrt();
    let multi = MultiDual::<f64, 1>::variable(v, 0)
        .unwrap()
        .max(MultiDual::zero())
        .sqrt();
    assert_eq!(dual.derivative().to_bits(), 0.0_f64.to_bits());
    assert_eq!(multi.gradient()[0].to_bits(), 0.0_f64.to_bits());
    // Jets use IEEE products: the infinite partial of √ at 0 times a zero
    // coefficient is NaN.
    let jet = Jet::<f64, 2>::variable(v).max(Jet::zero()).sqrt();
    assert!(jet.coefficients()[1].is_nan());
}

#[test]
fn combine_map_scalar_and_gather_apply_per_component() {
    let x = MultiDual::<f64, 2>::new(0.3, [1.0, 2.0]);
    let y = MultiDual::new(1.5, [0.0, -1.0]);
    assert_eq!(x.map_scalar(|v| [v.sin(), v.cos(), -v.sin()]), x.sin());
    let z = MultiDual::combine(&[x, y], 7.0, &[2.0, 3.0]).unwrap();
    assert_eq!((z.value(), z.gradient()), (7.0, [2.0, 1.0]));
    let error = MultiDual::combine(&[x, y], 7.0, &[2.0]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let table = [x, y, x * y];
    let picked = MultiDual::gather(&table, MultiDual::constant(2.0), |k| k as usize).unwrap();
    assert_eq!(picked, x * y);
    let error = MultiDual::gather(&table, x, |_| 3).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
}

#[test]
fn selection_and_finiteness_see_every_component() {
    let x = MultiDual::<f64, 2>::new(1.0, [1.0, f64::INFINITY]);
    let y = MultiDual::new(2.0, [0.0, 3.0]);
    assert!(!x.is_finite());
    assert!(y.is_finite());
    assert!(x.lt(y) && x.le(y) && !x.gt(y) && !x.ge(y));
    assert_eq!(MultiDual::select(false, x, y), y);
    assert_eq!(MultiDual::<f64, 2>::constant(4.0).gradient(), [0.0, 0.0]);
    assert_eq!(MultiDual::<f64, 2>::LEVEL, 1);
    assert_eq!(Dual::<MultiDual<f64, 2>>::LEVEL, 2);
}

#[test]
fn jvp_returns_the_directional_derivative_of_every_output() {
    let tangent = jvp(
        |x, out| {
            out.push(x[0] * x[1]);
            out.push(x[0].exp() + x[1].sin());
            Ok(())
        },
        &[0.5_f64, 2.0],
        &[1.0, -1.0],
    )
    .unwrap();
    let expected = [2.0 - 0.5, 0.5_f64.exp() - 2.0_f64.cos()];
    for (actual, expected) in tangent.as_slice().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-15);
    }
    let error = jvp(|_, _| Ok(()), &[1.0_f64], &[1.0, 2.0]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let empty = jvp(
        |x: &[Dual<f32>], out| {
            out.push(x.iter().copied().sum::<Dual<f32>>() + 1.0);
            Ok(())
        },
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(empty.as_slice(), &[0.0]);
    let failure = jvp(
        |_, _| Err(scioxide_core::SciOxideError::singular(None)),
        &[1.0_f64],
        &[1.0],
    );
    assert_eq!(failure.unwrap_err().kind(), SciOxideErrorKind::Singular);
}

#[test]
fn clamp_takes_its_value_from_the_primal_in_every_type() {
    let (nan, zero) = (f64::NAN, -0.0_f64);
    assert!(nan.clamp(0.0, 1.0).is_nan());
    assert!(
        Real::clamp(
            Lanes::<f64, 2>::splat(nan),
            Lanes::splat(0.0),
            Lanes::splat(1.0)
        )
        .lane(0)
        .unwrap()
        .is_nan()
    );
    assert!(
        Dual::new(nan, 1.0)
            .clamp(Dual::constant(0.0), Dual::constant(1.0))
            .value()
            .is_nan()
    );
    let multi = MultiDual::<f64, 2>::new(nan, [1.0, 0.0]);
    assert!(
        multi
            .clamp(MultiDual::constant(0.0), MultiDual::constant(1.0))
            .value()
            .is_nan()
    );
    let jet = Jet::<f64, 3>::variable(nan).clamp(Jet::constant(0.0), Jet::constant(1.0));
    assert!(jet.coefficients()[0].is_nan());
    // −0 inside [0, 1] keeps its sign, as for `f64`.
    let expected = zero.clamp(0.0, 1.0).to_bits();
    assert_eq!(
        Dual::variable(zero)
            .clamp(Dual::constant(0.0), Dual::constant(1.0))
            .value()
            .to_bits(),
        expected
    );
    let jet = Jet::<f64, 3>::variable(zero).clamp(Jet::constant(0.0), Jet::constant(1.0));
    assert_eq!(jet.coefficients()[0].to_bits(), expected);
    // Inside the interval the derivative is 1, outside 0, at a bound ½.
    for (x, d) in [(0.5, 1.0), (2.0, 0.0), (1.0, 0.5)] {
        assert_eq!(
            Dual::variable(x)
                .clamp(Dual::constant(0.0), Dual::constant(1.0))
                .derivative(),
            d
        );
    }
    let mut tape = Tape::<f64>::default();
    let mut g = [0.0];
    let value = tape
        .gradient_into(
            |x| Ok(x[0].clamp(Real::constant(0.0), Real::constant(1.0))),
            &[nan],
            &mut g,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(value.is_nan());
}

#[test]
fn remainder_uses_the_quotient_of_its_value() {
    // 1.0 % 0.1 = 0.0999…95: the quotient is 9, although fl(1.0/0.1) = 10.
    let d = (Dual::constant(1.0) % Dual::variable(0.1)).derivative();
    assert_eq!(d, -9.0);
    let jet = Jet::<f64, 2>::constant(1.0) % Jet::variable(0.1);
    assert_eq!(jet.coefficients()[1], -9.0);
    let cx = ExecutionContext::sequential();
    let mut strict = Tape::<f64>::new(TapeOptions::new().with_kink_policy(KinkPolicy::Error));
    let mut g = [0.0; 2];
    strict
        .gradient_into(|x| Ok(x[0] % x[1]), &[1.0, 0.1], &mut g, cx)
        .unwrap();
    assert_eq!(g, [1.0, -9.0]);
    // 1.0/0.25 = 4 is a kink: the midpoint −3.5 under the midpoint policy.
    let error = strict
        .gradient_into(|x| Ok(x[0] % x[1]), &[1.0, 0.25], &mut g, cx)
        .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::DomainError { .. }),
        "{error}"
    );
    let mut tape = Tape::<f64>::default();
    tape.gradient_into(|x| Ok(x[0] % x[1]), &[1.0, 0.25], &mut g, cx)
        .unwrap();
    assert_eq!(g, [1.0, -3.5]);
}

#[test]
fn round_has_kinks_only_at_half_integers() {
    let cx = ExecutionContext::sequential();
    let mut tape = Tape::<f64>::default();
    let mut g = [0.0];
    for (x, kinks) in [
        (2.5, 1),
        (2f64.powi(51) + 0.5, 1),
        (3.0, 0),
        (2f64.powi(52) + 1.0, 0),
        (1e17, 0),
    ] {
        tape.gradient_into(|x| Ok(x[0].round() + x[0]), &[x], &mut g, cx)
            .unwrap();
        assert_eq!(tape.report().kinks(), kinks, "x = {x}");
    }
}

#[test]
fn power_and_logaddexp_partials_meet_the_accuracy_bound() {
    // References from mpmath at 600 bits.
    for (x, y, reference) in [
        (1e-150, -0.001, -1.4125375446227543e147),
        (3e100, -0.001, -2.6448535171523398e-104),
        (3e-100, -0.1, -2.9865281994692114e108),
    ] {
        let d = Dual::variable(x).powf(Dual::constant(y)).derivative();
        close(d, reference, 2, "powf");
        let j = Jet::<f64, 2>::variable(x)
            .powf(Jet::constant(y))
            .coefficients()[1];
        close(j, reference, 2, "powf jet");
    }
    let d = Dual::variable(0.1)
        .logaddexp(Dual::constant(300.1))
        .derivative();
    close(d, 5.148200222411896e-131, 2, "logaddexp x");
    let d = Dual::constant(0.1)
        .logaddexp(Dual::variable(-300.1))
        .derivative();
    close(d, 4.214989845091526e-131, 2, "logaddexp y");
}

#[test]
fn xlogy_is_flat_along_a_zero_factor() {
    assert_eq!(
        Dual::constant(0.0).xlogy(Dual::variable(0.0)).derivative(),
        0.0
    );
    assert_eq!(
        Dual::constant(0.0)
            .xlog1py(Dual::variable(-1.0))
            .derivative(),
        0.0
    );
    // Off the singular point the mixed partial 1/y survives nesting.
    let x = Dual::new(Dual::new(0.0, 1.0), Dual::constant(0.0));
    let y = Dual::new(Dual::constant(2.0), Dual::constant(1.0));
    assert_eq!(x.xlogy(y).derivative().derivative(), 0.5);
}

#[test]
fn scalars_take_the_remainder_of_lanes_and_logit_has_a_domain() {
    let r = 7.0 % Lanes::<f64, 2>::from_fn(|k| [2.0, 4.0][k]);
    assert_eq!((r.lane(0), r.lane(1)), (Some(1.0), Some(3.0)));
    let r = 7.0_f32 % Lanes::<f32, 2>::splat(2.0);
    assert_eq!(r.lane(1), Some(1.0));
    for x in [2.0, 0.0, 1.0, -0.5] {
        assert!(Dual::variable(x).logit().derivative().is_nan(), "x = {x}");
    }
    close(
        Dual::variable(0.25).logit().derivative(),
        16.0 / 3.0,
        2,
        "logit",
    );
}

#[test]
fn complex_logarithm_and_inverse_keep_relative_accuracy() {
    for (re, im) in [
        (1.0, 1e-10),
        (0.6, 0.8),
        (1.0, 1e-5),
        (0.3, 1.2),
        (3.0, 4.0),
    ] {
        let expected = Complex::new(re, im).ln();
        let got = ComplexElementary::ln(Complex::new(Dual::variable(re), Dual::constant(im)));
        close(got.re.value(), expected.re, 2, "ln re");
        close(got.im.value(), expected.im, 2, "ln im");
        // ∂ Re ln z / ∂x = x/|z|².
        close(got.re.derivative(), re / (re * re + im * im), 4, "ln re'");
    }
    let big = ComplexElementary::inv(Complex::new(1e308, 1e308));
    let expected = Complex::new(1e308_f64, 1e308).inv();
    close(big.re, expected.re, 2, "inv re");
    close(big.im, expected.im, 2, "inv im");
}
