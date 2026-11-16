//! Every operation on `W` lanes equals the one-lane evaluation bit for bit.
#![allow(clippy::unwrap_used)]

mod support;

use scioxide_ad::{Dual, Jet, Lanes, MultiDual, Real};
use scioxide_core::RealScalar;

/// Every `Real` operation of two operands.
fn operations<S: RealScalar, R: Real<S>>(x: R, y: R) -> Vec<R> {
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
        x.powi(3),
        x.powi(-2),
        x.clamp(y, y + S::ONE),
        s,
        c,
        x.azmul(y),
        x + y,
        x - y,
        x * y,
        x / y,
        x % y,
        -x,
        x + S::TWO,
        x - S::TWO,
        x * S::TWO,
        x / S::TWO,
        x % S::TWO,
        R::select(x.lt(y), x, y),
        [x, y, x * y].into_iter().sum(),
        [x, y].into_iter().product(),
    ]
}

const VALUES: [f64; 18] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.5,
    1e-300,
    1e300,
    f64::INFINITY,
    -f64::INFINITY,
    f64::NAN,
    0.3,
    -2.7,
    7.0,
    100.0,
    1e-8,
    0.999,
    -0.25,
];

fn value<S: RealScalar>(index: usize) -> S {
    S::from_f64_ieee(VALUES[index % VALUES.len()])
}

/// Flattened lanes of the results for lanes `offset..offset + W`, through `build`
/// and `flatten`.
fn evaluate<S: RealScalar, R: Real<S>, const W: usize>(
    offset: usize,
    build: impl Fn(Lanes<S, W>, Lanes<S, W>) -> (R, R),
    flatten: impl Fn(R) -> Vec<Lanes<S, W>>,
) -> Vec<Vec<S>> {
    let x = Lanes::from_fn(|lane| value(offset + lane));
    let y = Lanes::from_fn(|lane| value(3 * (offset + lane) + 1));
    let (x, y) = build(x, y);
    operations(x, y)
        .into_iter()
        .map(|result| {
            flatten(result)
                .into_iter()
                .flat_map(Lanes::to_array)
                .collect()
        })
        .collect()
}

/// Compares `W` lanes with one lane at a time, bit for bit, in every component.
fn compare<S: RealScalar, const W: usize>(
    name: &str,
    offset: usize,
    lane: usize,
    wide: &[Vec<S>],
    single: &[Vec<S>],
) {
    for (operation, (wide, single)) in wide.iter().zip(single).enumerate() {
        for (component, single) in single.iter().enumerate() {
            let wide = wide[component * W + lane];
            assert_eq!(
                support::bits(wide),
                support::bits(*single),
                "{name} W={W} offset={offset} lane={lane} operation={operation} \
                 component={component}: {wide:?} != {single:?}"
            );
        }
    }
}

/// Evaluates the composition `$Type<S, W>` on `W` lanes and on each lane alone.
/// `$build` seeds the operands from lane packs and `$flatten` lists the
/// components of a result; both are written once and typed for either width.
macro_rules! composition {
    ($name:literal, $Type:ident, $offset:expr, $build:expr, $flatten:expr) => {{
        let wide = evaluate::<S, $Type<S, W>, W>($offset, $build, $flatten);
        for lane in 0..W {
            let single = evaluate::<S, $Type<S, 1>, 1>($offset + lane, $build, $flatten);
            compare::<S, W>($name, $offset, lane, &wide, &single);
        }
    }};
}

type Plain<S, const W: usize> = Lanes<S, W>;
type Tangent<S, const W: usize> = Dual<Lanes<S, W>>;
type Second<S, const W: usize> = Dual<Dual<Lanes<S, W>>>;
type Gradient<S, const W: usize> = MultiDual<Lanes<S, W>, 2>;
type GradientTangent<S, const W: usize> = Dual<MultiDual<Lanes<S, W>, 2>>;
type Taylor<S, const W: usize> = Jet<Lanes<S, W>, 4>;
type Mixed<S, const W: usize> = Jet<Jet<Lanes<S, W>, 2>, 3>;

fn check<S: RealScalar, const W: usize>() {
    for offset in 0..VALUES.len() {
        composition!("lanes", Plain, offset, |x, y| (x, y), |r| vec![r]);
        composition!(
            "dual",
            Tangent,
            offset,
            |x, y| {
                (
                    Dual::new(x, Lanes::splat(S::ONE)),
                    Dual::new(y, Lanes::splat(S::HALF)),
                )
            },
            |r| vec![r.value(), r.derivative()]
        );
        composition!(
            "dual2",
            Second,
            offset,
            |x, y| {
                let one = Lanes::splat(S::ONE);
                let zero = Lanes::splat(S::ZERO);
                (
                    Dual::new(Dual::new(x, one), Dual::new(one, zero)),
                    Dual::new(Dual::new(y, zero), Dual::new(one, zero)),
                )
            },
            |r| {
                vec![
                    r.value().value(),
                    r.value().derivative(),
                    r.derivative().value(),
                    r.derivative().derivative(),
                ]
            }
        );
        composition!(
            "multidual",
            Gradient,
            offset,
            |x, y| {
                let (one, half) = (Lanes::splat(S::ONE), Lanes::splat(S::HALF));
                (
                    MultiDual::new(x, [one, half]),
                    MultiDual::new(y, [Lanes::splat(S::ZERO), one]),
                )
            },
            |r| {
                let [d0, d1] = r.gradient();
                vec![r.value(), d0, d1]
            }
        );
        composition!(
            "dual-multidual",
            GradientTangent,
            offset,
            |x, y| {
                let (zero, one) = (Lanes::splat(S::ZERO), Lanes::splat(S::ONE));
                let half = Lanes::splat(S::HALF);
                (
                    Dual::new(
                        MultiDual::new(x, [one, zero]),
                        MultiDual::new(half, [zero; 2]),
                    ),
                    Dual::new(
                        MultiDual::new(y, [zero, one]),
                        MultiDual::new(one, [zero; 2]),
                    ),
                )
            },
            |r| {
                let (value, derivative) = (r.value(), r.derivative());
                let ([a, b], [c, d]) = (value.gradient(), derivative.gradient());
                vec![value.value(), a, b, derivative.value(), c, d]
            }
        );
        composition!(
            "jet",
            Taylor,
            offset,
            |x, y| {
                let (zero, one) = (Lanes::splat(S::ZERO), Lanes::splat(S::ONE));
                let half = Lanes::splat(S::HALF);
                (
                    Jet::new([x, one, half, zero]),
                    Jet::new([y, zero, one, half]),
                )
            },
            |r| r.coefficients().to_vec()
        );
        composition!(
            "jet-jet",
            Mixed,
            offset,
            |x, y| {
                let (zero, one) = (Lanes::splat(S::ZERO), Lanes::splat(S::ONE));
                let half = Lanes::splat(S::HALF);
                (
                    Jet::new([Jet::new([x, one]), Jet::new([one, zero]), Jet::zero()]),
                    Jet::new([
                        Jet::new([y, zero]),
                        Jet::new([half, one]),
                        Jet::new([one, zero]),
                    ]),
                )
            },
            |r| {
                r.coefficients()
                    .into_iter()
                    .flat_map(Jet::coefficients)
                    .collect()
            }
        );
    }
}

#[test]
fn lanes_equal_one_lane_for_every_width() {
    // Unoptimized frames of the nested compositions exceed the default test stack.
    std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(|| {
            check::<f64, 1>();
            check::<f64, 2>();
            check::<f64, 4>();
            check::<f64, 8>();
            check::<f64, 16>();
            check::<f32, 1>();
            check::<f32, 2>();
            check::<f32, 4>();
            check::<f32, 8>();
            check::<f32, 16>();
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn lanes_outside_the_kernels_equal_the_scalar_function() {
    for index in 0..VALUES.len() {
        let x = value::<f64>(index);
        let y = value::<f64>(3 * index + 1);
        let scalar = operations(x, y);
        let lanes = operations(Lanes::<f64, 4>::splat(x), Lanes::splat(y));
        // Indices of exp, exp_m1, ln, ln_1p, sin, cos, tanh, powf and their users
        // (sin_cos, the composite pow-based operations) may differ in the last ulps.
        let kernel = [3, 5, 6, 7, 10, 11, 18, 45, 58, 59];
        for (operation, (s, l)) in scalar.iter().zip(&lanes).enumerate() {
            if kernel.contains(&operation) {
                continue;
            }
            for lane in l.to_array() {
                assert_eq!(
                    support::bits(lane),
                    support::bits(*s),
                    "operation {operation}"
                );
            }
        }
    }
}
