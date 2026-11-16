//! Normalized Taylor recurrences in the arithmetic of the coefficient type.
//!
//! A series `x(t) = Σₖ xₖ tᵏ` is an array `[V; N]`. Every function returns `c₀` as
//! the operation on `V` and `c₁` as the first partial of [`crate::rules`] times
//! `x₁` with an IEEE product (see the exceptions in [`crate::Jet`]); coefficients above
//! follow the recurrences of Griewank and Walther (2008, ch. 13). Products are IEEE
//! products with a separate multiply and add, and every sum runs in index order.
//! Euler-weighted recurrences multiply `xⱼ` by the integer `j` and divide the sum
//! by `k` once.

use core::ops::{Mul, RangeInclusive};

use scioxide_core::RealScalar;

use crate::real::sealed::{self, Token};
use crate::rules;
use crate::stable::{self, Stable};
use crate::{BaseReal, Mask, Real};

/// The integer `k` as a scalar.
#[inline(always)]
fn int<S: RealScalar>(k: usize) -> S {
    S::from_f64_ieee(k as f64)
}

/// The series of the constant `value`.
#[inline(always)]
pub(super) fn lift<S: RealScalar, V: Real<S>, const N: usize>(value: V) -> [V; N] {
    let mut series = [V::zero(); N];
    series[0] = value;
    series
}

/// The series `[value, NaN, …]` of a value whose higher coefficients are not formed.
#[inline(always)]
fn first_order<S: RealScalar, V: Real<S>, const N: usize>(value: V, first: V) -> [V; N] {
    let mut series = [V::constant(S::NAN); N];
    series[0] = value;
    if N > 1 {
        series[1] = first;
    }
    series
}

/// Sums in index order.
#[inline(always)]
fn sum<S: RealScalar, V: Real<S>>(mut terms: impl Iterator<Item = V>) -> V {
    match terms.next() {
        Some(first) => terms.fold(first, |total, term| total + term),
        None => V::zero(),
    }
}

/// `Σⱼ aⱼ b_{k−j}` over `range`.
#[inline(always)]
fn convolution<S: RealScalar, V: Real<S>, const N: usize>(
    a: &[V; N],
    b: &[V; N],
    k: usize,
    range: RangeInclusive<usize>,
) -> V {
    sum(range.map(|j| a[j] * b[k - j]))
}

/// The Cauchy product coefficient `Σ_{j=0}^{k} aⱼ b_{k−j}`.
#[inline(always)]
fn cauchy<S: RealScalar, V: Real<S>, const N: usize>(a: &[V; N], b: &[V; N], k: usize) -> V {
    convolution(a, b, k, 0..=k)
}

/// `Σ_{j=1}^{k} aⱼ b_{k−j}`.
#[inline(always)]
fn tail<S: RealScalar, V: Real<S>, const N: usize>(a: &[V; N], b: &[V; N], k: usize) -> V {
    convolution(a, b, k, 1..=k)
}

/// `Σ_{j=1}^{k−1} aⱼ b_{k−j}`, both indices at least 1.
#[inline(always)]
fn inner<S: RealScalar, V: Real<S>, const N: usize>(a: &[V; N], b: &[V; N], k: usize) -> V {
    convolution(a, b, k, 1..=k.saturating_sub(1))
}

/// `j·xⱼ`, the coefficients of `t·dx/dt`.
#[inline(always)]
fn degrees<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    core::array::from_fn(|j| x[j] * int::<S>(j))
}

/// Picks `if_true` where `mask` holds; evaluates only the forms some lane needs.
#[inline(always)]
fn choose<S: RealScalar, V: Real<S>, const N: usize>(
    mask: V::Mask,
    if_true: impl FnOnce() -> [V; N],
    if_false: impl FnOnce() -> [V; N],
) -> [V; N] {
    if mask.all() {
        if_true()
    } else if !mask.any() {
        if_false()
    } else {
        let (yes, no) = (if_true(), if_false());
        core::array::from_fn(|k| V::select(mask, yes[k], no[k]))
    }
}

use crate::support::is_zero;

/// Lanes in which the base value is ±0.
#[inline(always)]
fn zero_base<S: RealScalar, V: Real<S>>(value: V) -> V::Mask {
    value.ge(V::zero()) & value.le(V::zero())
}

/// `[f, f′, f″]` of a stable transform at `x₀` when `V` is a base type. There the
/// first coefficient uses the derivative of the `map_scalar` rule, evaluated in
/// `f64` and rounded once, as `Dual` does; above the base the recurrences seed
/// themselves in `V` arithmetic, exact to every order.
#[inline(always)]
fn base_rule<S: RealScalar, V: Real<S>>(x0: V, function: Stable) -> Option<[V; 3]> {
    (V::LEVEL == 0)
        .then(|| <V as sealed::Map<S>>::map_all(Token(()), x0, &mut |v| stable::rule(function, v)))
}

// ---- Arithmetic.

pub(super) fn mul<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    core::array::from_fn(|k| match k {
        0 => x[0] * y[0],
        1 => y[0] * x[1] + x[0] * y[1],
        _ => cauchy(x, y, k),
    })
}

/// From `x = q·y`: `qₖ = (xₖ − Σ_{j=1}^{k} yⱼ q_{k−j})/y₀`.
pub(super) fn div<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    let q0 = x[0] / y[0];
    let (left, right) = rules::div_partials(y[0], q0);
    let mut q = lift(q0);
    for k in 1..N {
        q[k] = if k == 1 {
            left * x[1] + right * y[1]
        } else {
            (x[k] - tail(y, &q, k)) / y[0]
        };
    }
    q
}

/// The absolute-zero Cauchy product.
pub(super) fn azmul<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    core::array::from_fn(|k| sum((0..=k).map(|j| x[j].azmul(y[k - j]))))
}

/// A first-order kink rule applied to every coefficient: `zₖ = l·xₖ + r·yₖ` with
/// absolute-zero products, so a partial of 0 or 1 selects.
fn kink<S: RealScalar, V: Real<S>, const N: usize>(
    value: V,
    (left, right): (V, V),
    x: &[V; N],
    y: &[V; N],
) -> [V; N] {
    core::array::from_fn(|k| match k {
        0 => value,
        _ => left.azmul(x[k]) + right.azmul(y[k]),
    })
}

/// `a % b = a − trunc(a₀/b₀)·b` above `c₀`.
pub(super) fn rem<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    let r = x[0] % y[0];
    kink(r, rules::rem_partials(x[0], y[0], r), x, y)
}

pub(super) fn extremum<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    y: &[V; N],
    larger: bool,
) -> [V; N] {
    let (x0, y0) = (x[0], y[0]);
    let value = if larger { x0.max(y0) } else { x0.min(y0) };
    kink(value, rules::extremum_partials(x0, y0, larger), x, y)
}

pub(super) fn abs<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let partial = rules::abs_partial(x[0]);
    core::array::from_fn(|k| match k {
        0 => x[0].abs(),
        _ => partial.azmul(x[k]),
    })
}

/// `x·c` for a constant factor `c` above `c₀`.
pub(super) fn linear<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    value: V,
    factor: S,
) -> [V; N] {
    core::array::from_fn(|k| match k {
        0 => value,
        _ => x[k] * factor,
    })
}

// ---- Exponentials and logarithms.

/// `y′ = σ·y·x′`: `yₖ = σ Σ_{j=1}^{k} j xⱼ y_{k−j} / k`.
fn exponential<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    value: V,
    first: V,
    scale: S,
) -> [V; N] {
    let dx = degrees(x);
    let mut y = lift(value);
    for k in 1..N {
        y[k] = if k == 1 {
            first * x[1]
        } else {
            tail(&dx, &y, k) * scale / int::<S>(k)
        };
    }
    y
}

pub(super) fn exp<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let y0 = x[0].exp();
    exponential(x, y0, y0, S::ONE)
}

pub(super) fn exp2<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let y0 = x[0].exp2();
    exponential(x, y0, rules::exp2_partial(y0), S::LN_2)
}

/// The `exp` recurrence with its own `c₀`.
pub(super) fn exp_m1<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let mut y = exp(x);
    y[0] = x[0].exp_m1();
    y
}

/// `u·y′ = σ·x′` for a logarithm: `yₖ = (σxₖ − Σ_{j=1}^{k−1} j yⱼ x_{k−j}/k)/u₀`.
fn logarithm<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    value: V,
    first: V,
    denominator: V,
    scale: S,
) -> [V; N] {
    let mut y = lift(value);
    let mut dy = [V::zero(); N];
    for k in 1..N {
        y[k] = if k == 1 {
            first * x[1]
        } else {
            (x[k] * scale - inner(&dy, x, k) / int::<S>(k)) / denominator
        };
        dy[k] = y[k] * int::<S>(k);
    }
    y
}

pub(super) fn ln<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    logarithm(x, x0.ln(), rules::ln_partial(x0), x0, S::ONE)
}

pub(super) fn ln_1p<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    logarithm(x, x0.ln_1p(), rules::ln_1p_partial(x0), x0 + S::ONE, S::ONE)
}

pub(super) fn log2<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    logarithm(x, x0.log2(), rules::log2_partial(x0), x0, S::LOG2_E)
}

pub(super) fn log10<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    logarithm(x, x0.log10(), rules::log10_partial(x0), x0, S::LOG10_E)
}

/// From `x·y = 1`: `yₖ = −Σ_{j=1}^{k} xⱼ y_{k−j}/x₀`.
pub(super) fn recip<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let y0 = x[0].recip();
    let first = rules::recip_partial(y0);
    let mut y = lift(y0);
    for k in 1..N {
        y[k] = if k == 1 {
            first * x[1]
        } else {
            -(tail(x, &y, k) / x[0])
        };
    }
    y
}

/// From `y² = x`: `yₖ = (xₖ − Σ_{j=1}^{k−1} yⱼ y_{k−j})/(2y₀)`.
fn square_root<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y0: V) -> [V; N] {
    let first = rules::sqrt_partial(y0);
    let mut y = lift(y0);
    for k in 1..N {
        y[k] = if k == 1 {
            first * x[1]
        } else {
            (x[k] - inner(&y, &y, k)) * S::HALF / y0
        };
    }
    y
}

pub(super) fn sqrt<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    square_root(x, x[0].sqrt())
}

/// Miller's recurrence for `y = xᵖ` from `x·y′ = p·y·x′`:
/// `yₖ = Σ_{j=1}^{k} xⱼ y_{k−j} w(j, k) / (d·k) / x₀` with `w(j, k) = d·(p·j − (k − j))`.
fn power<S: RealScalar, V: Real<S> + Mul<W, Output = V>, W: Copy, const N: usize>(
    x: &[V; N],
    value: V,
    first: V,
    denominator: usize,
    weight: impl Fn(usize, usize) -> W,
) -> [V; N] {
    let mut y = lift(value);
    for k in 1..N {
        y[k] = if k == 1 {
            first * x[1]
        } else {
            sum((1..=k).map(|j| (x[j] * y[k - j]) * weight(j, k)))
                / int::<S>(denominator * k)
                / x[0]
        };
    }
    y
}

/// The integer weight `a·j − b·(k − j)` of `p = a/b`, exact in `f64`.
#[inline(always)]
fn rational_weight<S: RealScalar>(a: f64, b: f64, j: usize, k: usize) -> S {
    S::from_f64_ieee(a * j as f64 - b * (k - j) as f64)
}

pub(super) fn cbrt<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let y0 = x[0].cbrt();
    power(x, y0, rules::cbrt_partial(y0), 3, |j, k| {
        rational_weight::<S>(1.0, 3.0, j, k)
    })
}

/// Binary powering of the whole series; a negative exponent powers `1/x`.
fn repeated_product<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    exponent: i32,
) -> [V; N] {
    let mut base = if exponent < 0 { recip(x) } else { *x };
    let mut result = lift(V::one());
    let mut remaining = exponent.unsigned_abs();
    while remaining != 0 {
        if remaining & 1 == 1 {
            result = core::array::from_fn(|k| cauchy(&result, &base, k));
        }
        remaining >>= 1;
        if remaining != 0 {
            base = core::array::from_fn(|k| cauchy(&base, &base, k));
        }
    }
    result
}

/// Miller's recurrence away from zero; repeated products of the series at a zero
/// base, where the recurrence would divide by `x₀`.
pub(super) fn powi<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], exponent: i32) -> [V; N] {
    let x0 = x[0];
    let value = x0.powi(exponent);
    let first = rules::powi_partial(x0, exponent);
    let n = f64::from(exponent);
    choose(
        zero_base(x0),
        || {
            let mut y = repeated_product(x, exponent);
            y[0] = value;
            if N > 1 {
                y[1] = first * x[1];
            }
            y
        },
        || {
            power(x, value, first, 1, |j, k| {
                rational_weight::<S>(n, 1.0, j, k)
            })
        },
    )
}

/// Replaces the coefficients above `c₁` by those of `powi(n)` in the lanes of
/// `lanes` whose constant exponent `y₀ = n` is an integer in `[0, i32::MAX]`.
fn integer_powers<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    y0: V,
    lanes: V::Mask,
    z: &mut [V; N],
) {
    let constant = is_zero(y0 - V::passive(y0.base()));
    let exponents = y0.base();
    let index = <V::Base as BaseReal<S>>::from_fn(int::<S>);
    for lane in 0..V::LANES {
        let Some(n) = exponents.lane(lane) else {
            continue;
        };
        let position = <V::Base as Real<S>>::constant(int::<S>(lane));
        let mask = lanes & constant & index.ge(position) & index.le(position);
        let integral = n >= S::ZERO && n == n.trunc() && n.into_f64() <= f64::from(i32::MAX);
        if integral && mask.any() {
            let powers = powi(x, n.into_f64() as i32);
            for (coefficient, &power) in z.iter_mut().zip(&powers).skip(2) {
                *coefficient = V::select(mask, power, *coefficient);
            }
        }
    }
}

/// `xʸ` with the coefficients of `powf`, `c₀ = value`. Where every coefficient of
/// the exponent above `c₀` is zero, Miller's recurrence with `p = y₀`, so negative
/// bases are allowed; at a zero base the series of `powi` for an integer `y₀ ≥ 0`,
/// otherwise coefficient `k ≥ 2` is 0 for `k < y₀` and NaN from `y₀` on.
/// Elsewhere `exp(h)` with `h = y·ln x`.
fn power_function<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    y: &[V; N],
    value: V,
) -> [V; N] {
    let (x0, y0) = (x[0], y[0]);
    let z0 = x0.powf(y0);
    let left = rules::powf_left(x0, y0, z0);
    let fixed = y[1..]
        .iter()
        .fold(V::Mask::splat(true), |fixed, &c| fixed & is_zero(c));
    let constant = || {
        let mut z = power(x, z0, left, 1, |j, k| y0 * int::<S>(j) - int::<S>(k - j));
        let zero = zero_base(x0);
        if zero.any() {
            let flat = is_zero(x0);
            for (k, coefficient) in z.iter_mut().enumerate().skip(2) {
                let below = V::constant(int::<S>(k)).lt(y0);
                let limit = V::select(below, V::zero(), V::constant(S::NAN));
                *coefficient = V::select(flat, limit, *coefficient);
            }
            integer_powers(x, y0, zero, &mut z);
        }
        z
    };
    let variable = || {
        let right = rules::powf_right(x0, z0);
        let log = ln(x);
        let h: [V; N] = core::array::from_fn(|k| cauchy(y, &log, k));
        let dh = degrees(&h);
        let mut z = lift(z0);
        for k in 1..N {
            z[k] = if k == 1 {
                left * x[1] + right * y[1]
            } else {
                tail(&dh, &z, k) / int::<S>(k)
            };
        }
        z
    };
    let mut z = choose(fixed, constant, variable);
    z[0] = value;
    z
}

pub(super) fn powf<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    power_function(x, y, x[0].powf(y[0]))
}

/// The `powf` recurrence with its own `c₀`.
pub(super) fn powm1<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    power_function(x, y, x[0].powm1(y[0]))
}

/// `y′ = g·x′` with the series `g` of the derivative: `yₖ = Σ_{j=1}^{k} j xⱼ g_{k−j}/k`.
fn integral<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    value: V,
    derivative: &[V; N],
) -> [V; N] {
    let dx = degrees(x);
    core::array::from_fn(|k| match k {
        0 => value,
        1 => derivative[0] * x[1],
        _ => tail(&dx, derivative, k) / int::<S>(k),
    })
}

/// `σ(v)` from `σ′ = D·v′`, `D′ = u·σ′` and `u = 1 − 2σ`, `u′ = −2σ′`, seeded with
/// `σ₀`, `D₀ = σ′(v₀)` and `u₀`; no coefficient subtracts a rounded saturated value.
fn logistic<S: RealScalar, V: Real<S>, const N: usize>(
    v: &[V; N],
    sigma0: V,
    d0: V,
    u0: V,
) -> [V; N] {
    let dv = degrees(v);
    let mut sigma = lift(sigma0);
    let mut dsigma = [V::zero(); N];
    let mut d = lift(d0);
    let mut u = lift(u0);
    for k in 1..N {
        sigma[k] = if k == 1 {
            d0 * v[1]
        } else {
            tail(&dv, &d, k) / int::<S>(k)
        };
        dsigma[k] = sigma[k] * int::<S>(k);
        d[k] = tail(&dsigma, &u, k) / int::<S>(k);
        u[k] = sigma[k] * -S::TWO;
    }
    sigma
}

/// `σ′(v₀)` from `σ(v₀)·σ(−v₀)` above the base.
#[inline(always)]
fn logistic_slope<S: RealScalar, V: Real<S>>(sigma0: V, v0: V) -> V {
    sigma0 * (-v0).sigmoid()
}

pub(super) fn sigmoid<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    let value = x0.sigmoid();
    let d0 = match base_rule(x0, Stable::Sigmoid) {
        Some([_, first, _]) => first,
        None => logistic_slope(value, x0),
    };
    logistic(x, value, d0, -(x0 * S::HALF).tanh())
}

pub(super) fn softplus<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    let (sigma0, d0) = match base_rule(x0, Stable::Softplus) {
        Some([_, first, second]) => (first, second),
        None => {
            let sigma0 = x0.sigmoid();
            (sigma0, logistic_slope(sigma0, x0))
        }
    };
    let sigma = logistic(x, sigma0, d0, -(x0 * S::HALF).tanh());
    integral(x, x0.softplus(), &sigma)
}

/// `ln σ(x)` with `y′ = σ(−x)·x′`.
pub(super) fn log_sigmoid<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    let (sigma0, d0) = match base_rule(x0, Stable::LogSigmoid) {
        Some([_, first, second]) => (first, -second),
        None => {
            let sigma0 = (-x0).sigmoid();
            (sigma0, logistic_slope(sigma0, -x0))
        }
    };
    let negated = x.map(|c| -c);
    let sigma = logistic(&negated, sigma0, d0, (x0 * S::HALF).tanh());
    integral(x, x0.log_sigmoid(), &sigma)
}

/// `q·y′ = x′` with `q = x(1 − x)`, whose coefficients use `1 − 2x₀` directly.
pub(super) fn logit<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    let q0 = x0 * (-x0 + S::ONE);
    let first = match base_rule(x0, Stable::Logit) {
        Some([_, first, _]) => first,
        None => q0.recip(),
    };
    let center = -(x0 * S::TWO) + S::ONE;
    let mut q = lift(q0);
    let mut y = lift(x0.logit());
    let mut dy = [V::zero(); N];
    for k in 1..N {
        q[k] = center * x[k] - inner(x, x, k);
        y[k] = if k == 1 {
            first * x[1]
        } else {
            (x[k] - inner(&dy, &q, k) / int::<S>(k)) / q0
        };
        dy[k] = y[k] * int::<S>(k);
    }
    y
}

/// `logaddexp(x, y) = hi + softplus(lo − hi)` with `hi` the operand of larger
/// `c₀`; where `x₀ − y₀` is not finite, the coefficients of the larger operand.
pub(super) fn logaddexp<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    y: &[V; N],
) -> [V; N] {
    let (x0, y0) = (x[0], y[0]);
    let larger = x0.gt(y0);
    let high: [V; N] = core::array::from_fn(|k| V::select(larger, x[k], y[k]));
    let shifted = || {
        let low: [V; N] = core::array::from_fn(|k| V::select(larger, y[k], x[k]));
        let d: [V; N] = core::array::from_fn(|k| low[k] - high[k]);
        let d0 = d[0];
        let u0 = -(d0 * S::HALF).tanh();
        let (sigma0, slope) = match base_rule(d0, Stable::Sigmoid) {
            Some([value, first, _]) => (value, first),
            None => {
                let sigma0 = d0.sigmoid();
                (sigma0, logistic_slope(sigma0, d0))
            }
        };
        let sigma = logistic(&d, sigma0, slope, u0);
        let dd = degrees(&d);
        core::array::from_fn(|k| match k {
            0 => high[0],
            _ => high[k] + tail(&dd, &sigma, k) / int::<S>(k),
        })
    };
    let mut z = choose((x0 - y0).base().is_finite(), shifted, || high);
    z[0] = x0.logaddexp(y0);
    if N > 1 {
        let (left, right) = rules::logaddexp_partials(x0, y0);
        z[1] = left * x[1] + right * y[1];
    }
    z
}

/// `x·L` with `L` the series of a logarithm of `y`.
fn times_logarithm<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    y: &[V; N],
    log: &[V; N],
    value: V,
    (left, right): (V, V),
) -> [V; N] {
    core::array::from_fn(|k| match k {
        0 => value,
        1 => left * x[1] + right * y[1],
        _ => cauchy(x, log, k),
    })
}

pub(super) fn xlogy<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    let (x0, y0) = (x[0], y[0]);
    let partials = rules::xlogy_partials(x0, y0);
    times_logarithm(x, y, &ln(y), x0.xlogy(y0), partials)
}

pub(super) fn xlog1py<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    let (x0, y0) = (x[0], y[0]);
    let partials = rules::xlog1py_partials(x0, y0);
    times_logarithm(x, y, &ln_1p(y), x0.xlog1py(y0), partials)
}

/// `log_b x = ln x / ln b` as a quotient of logarithm series.
pub(super) fn log<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], b: &[V; N]) -> [V; N] {
    let (x0, b0) = (x[0], b[0]);
    let value = x0.log(b0);
    let (left, right) = rules::log_partials(x0, b0, value);
    let (lx, lb) = (ln(x), ln(b));
    let mut q = lift(value);
    for k in 1..N {
        q[k] = if k == 1 {
            left * x[1] + right * b[1]
        } else {
            (lx[k] - tail(&lb, &q, k)) / lb[0]
        };
    }
    q
}

// ---- Trigonometric and hyperbolic functions.

/// The coupled pair `s′ = σ·c·x′`, `c′ = ∓σ·s·x′` (`−` circular, `+` hyperbolic).
fn coupled<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    (s0, c0): (V, V),
    (s_first, c_first): (V, V),
    scale: S,
    hyperbolic: bool,
) -> ([V; N], [V; N]) {
    let dx = degrees(x);
    let mut s = lift(s0);
    let mut c = lift(c0);
    for k in 1..N {
        if k == 1 {
            s[1] = s_first * x[1];
            c[1] = c_first * x[1];
        } else {
            let ds = tail(&dx, &c, k) * scale / int::<S>(k);
            let dc = tail(&dx, &s, k) * scale / int::<S>(k);
            s[k] = ds;
            c[k] = if hyperbolic { dc } else { -dc };
        }
    }
    (s, c)
}

pub(super) fn sin_cos<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> ([V; N], [V; N]) {
    let (s0, c0) = x[0].sin_cos();
    coupled(x, (s0, c0), (c0, -s0), S::ONE, false)
}

/// The `cos` recurrence with its own `c₀`.
pub(super) fn cosm1<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    let (s0, c0) = x0.sin_cos();
    let first = base_rule(x0, Stable::Cosm1).map_or(-s0, |[_, first, _]| first);
    let (_, mut c) = coupled(x, (s0, c0), (c0, first), S::ONE, false);
    c[0] = x0.cosm1();
    c
}

pub(super) fn sinh_cosh<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> ([V; N], [V; N]) {
    let (s0, c0) = (x[0].sinh(), x[0].cosh());
    coupled(x, (s0, c0), (c0, s0), S::ONE, true)
}

/// The scale of the degree-based functions, `π/180`.
#[inline(always)]
fn degree<S: RealScalar>() -> S {
    S::from_f64_ieee(core::f64::consts::PI / 180.0)
}

/// `sin(σx)` (`cosine = false`) or `cos(σx)` with exact reduction, `σ = π` or
/// `π/180`, from the pair `(s₀, c₀)` of the reduced functions.
fn scaled_trigonometric<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    (s0, c0): (V, V),
    function: Stable,
    scale: S,
    cosine: bool,
) -> [V; N] {
    let x0 = x[0];
    let rule = base_rule(x0, function).map(|[_, first, _]| first);
    let s_first = c0 * scale;
    let c_first = -(s0 * scale);
    if cosine {
        let first = rule.unwrap_or(c_first);
        coupled(x, (s0, c0), (s_first, first), scale, false).1
    } else {
        let first = rule.unwrap_or(s_first);
        coupled(x, (s0, c0), (first, c_first), scale, false).0
    }
}

pub(super) fn sin_pi<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let pair = (x[0].sin_pi(), x[0].cos_pi());
    scaled_trigonometric(x, pair, Stable::SinPi, S::PI, false)
}

pub(super) fn cos_pi<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let pair = (x[0].sin_pi(), x[0].cos_pi());
    scaled_trigonometric(x, pair, Stable::CosPi, S::PI, true)
}

pub(super) fn sin_degrees<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let pair = (x[0].sin_degrees(), x[0].cos_degrees());
    scaled_trigonometric(x, pair, Stable::SinDegrees, degree(), false)
}

pub(super) fn cos_degrees<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let pair = (x[0].sin_degrees(), x[0].cos_degrees());
    scaled_trigonometric(x, pair, Stable::CosDegrees, degree(), true)
}

/// `y′ = w·x′` with the auxiliary series `w = w₀ + σ(y² − y₀²)`: `σ = 1` for `tan`
/// (`w = 1 + y²`), `σ = −1` for `tanh` (`w = 1 − y²`), `σ = π` and `π/180` for the
/// scaled tangents.
fn tangent<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    value: V,
    w0: V,
    scale: S,
) -> [V; N] {
    let dx = degrees(x);
    let mut y = lift(value);
    let mut w = lift(w0);
    for k in 1..N {
        y[k] = if k == 1 {
            w0 * x[1]
        } else {
            tail(&dx, &w, k) / int::<S>(k)
        };
        w[k] = cauchy(&y, &y, k) * scale;
    }
    y
}

pub(super) fn tan<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let y0 = x[0].tan();
    tangent(x, y0, rules::tan_partial(y0), S::ONE)
}

pub(super) fn tanh<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    let x0 = x[0];
    let y0 = x0.tanh();
    tangent(x, y0, rules::tanh_partial(x0, y0), -S::ONE)
}

fn scaled_tangent<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    value: V,
    function: Stable,
    scale: S,
) -> [V; N] {
    let w0 = base_rule(x[0], function)
        .map_or_else(|| rules::tan_partial(value) * scale, |[_, first, _]| first);
    tangent(x, value, w0, scale)
}

pub(super) fn tan_pi<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    scaled_tangent(x, x[0].tan_pi(), Stable::TanPi, S::PI)
}

pub(super) fn tan_degrees<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N]) -> [V; N] {
    scaled_tangent(x, x[0].tan_degrees(), Stable::TanDegrees, degree())
}

// ---- Inverse functions.

/// An inverse function solved from `u·y′ = ±x′`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Inverse {
    Asin,
    Acos,
    Atan,
    Asinh,
    Acosh,
    Atanh,
}

/// `u·y′ = ±x′` with `u = √(1 − x²)` (`asin`, `acos`), `1 + x²` (`atan`),
/// `√(1 + x²)` (`asinh`), `√(x² − 1)` (`acosh`) or `1 − x²` (`atanh`). The series
/// are divided by the constant `m = max(|x₀|, 1)` so that the quadratic stays in
/// range: `ũ·y′ = ±(x/m)′` for roots and `ũ·y′ = ±(x/m²)′` otherwise, with `ũ = u/m`
/// or `u/m²`.
pub(super) fn inverse<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    function: Inverse,
) -> [V; N] {
    use Inverse::{Acos, Acosh, Asin, Asinh, Atan, Atanh};
    let x0 = x[0];
    let (value, first) = match function {
        Asin => (x0.asin(), rules::asin_partial(x0)),
        Acos => (x0.acos(), rules::acos_partial(x0)),
        Atan => (x0.atan(), rules::atan_partial(x0)),
        Asinh => (x0.asinh(), rules::asinh_partial(x0)),
        Acosh => (x0.acosh(), rules::acosh_partial(x0)),
        Atanh => (x0.atanh(), rules::atanh_partial(x0)),
    };
    let scale = V::passive(x0.base().abs().max(<V::Base as Real<S>>::one()));
    let scaled = x.map(|c| c / scale);
    let negative = matches!(function, Asin | Acos | Atanh);
    let quadratic0 = match function {
        Asin | Acos | Atanh => ((-x0 + S::ONE) / scale) * ((x0 + S::ONE) / scale),
        Acosh => ((x0 - S::ONE) / scale) * ((x0 + S::ONE) / scale),
        Atan | Asinh => {
            let r = scale.recip();
            scaled[0] * scaled[0] + r * r
        }
    };
    let quadratic: [V; N] = core::array::from_fn(|k| match k {
        0 => quadratic0,
        _ if negative => -cauchy(&scaled, &scaled, k),
        _ => cauchy(&scaled, &scaled, k),
    });
    let root = !matches!(function, Atan | Atanh);
    let u = if root {
        square_root(&quadratic, quadratic0.sqrt())
    } else {
        quadratic
    };
    let mut y = lift(value);
    let mut dy = [V::zero(); N];
    for k in 1..N {
        y[k] = if k == 1 {
            first * x[1]
        } else {
            let numerator = if root { scaled[k] } else { scaled[k] / scale };
            let numerator = if function == Acos {
                -numerator
            } else {
                numerator
            };
            (numerator - inner(&dy, &u, k) / int::<S>(k)) / u[0]
        };
        dy[k] = y[k] * int::<S>(k);
    }
    y
}

// ---- Two-argument functions.

/// `h = hypot(x, y)` in polar form: `h′ = u·x′ + v·y′` with the direction cosines
/// `u = x/h`, `v = y/h`, which turn at `θ′ = (u·y′ − v·x′)/h`: `u′ = −v·θ′`,
/// `v′ = u·θ′`. Unlike `h² = x² + y²`, whose recurrence subtracts `h₁²` from
/// `x₁² + y₁²`, no step cancels when `(x′, y′)` is nearly radial. `(u₀, v₀)` are
/// the partials of the rule, `(0, 0)` at the origin, where the coefficients above
/// `c₁` are NaN.
pub(super) fn hypot<S: RealScalar, V: Real<S>, const N: usize>(x: &[V; N], y: &[V; N]) -> [V; N] {
    let (x0, y0) = (x[0], y[0]);
    let h0 = x0.hypot(y0);
    let (u0, v0) = rules::hypot_partials(x0, y0, h0);
    let (dx, dy) = (degrees(x), degrees(y));
    let mut h = lift(h0);
    let mut u = lift(u0);
    let mut v = lift(v0);
    let mut turn = [V::zero(); N];
    for k in 1..N {
        // θ′ coefficient k − 1 from h·θ′ = u·y′ − v·x′.
        let m = k - 1;
        let rate = sum((1..=k).map(|j| u[k - j] * dy[j] - v[k - j] * dx[j]));
        turn[m] = (rate - convolution(&h, &turn, m, 1..=m)) / h0;
        h[k] = if k == 1 {
            u0 * x[1] + v0 * y[1]
        } else {
            sum((1..=k).map(|j| dx[j] * u[k - j] + dy[j] * v[k - j])) / int::<S>(k)
        };
        u[k] = -(cauchy(&v, &turn, m) / int::<S>(k));
        v[k] = cauchy(&u, &turn, m) / int::<S>(k);
    }
    h
}

/// `atan2` to first order; the coefficients above `c₁` are NaN.
pub(super) fn atan2<S: RealScalar, V: Real<S>, const N: usize>(a: &[V; N], b: &[V; N]) -> [V; N] {
    let (y0, x0) = (a[0], b[0]);
    let (left, right) = rules::atan2_partials(y0, x0);
    let first = if N > 1 {
        left * a[1] + right * b[1]
    } else {
        V::zero()
    };
    first_order(y0.atan2(x0), first)
}

/// The value `value` with first coefficient `Σₖ partials[k]·c₁(arguments[k])`;
/// the coefficients above `c₁` are NaN.
pub(super) fn combine<S: RealScalar, V: Real<S>, const N: usize>(
    arguments: impl Iterator<Item = [V; N]>,
    value: V,
    partials: &[V],
) -> [V; N] {
    let first = if N > 1 {
        arguments
            .zip(partials)
            .fold(V::constant(S::NEG_ZERO), |sum, (argument, &partial)| {
                sum + partial * argument[1]
            })
    } else {
        V::zero()
    };
    first_order(value, first)
}

/// `[f(x), f′(x), f″(x)]` from `[f₀, f₁, f₂]` at `x₀`: `f(x) = [f₀, f₁x₁,
/// f₁x₂ + ½f₂x₁², NaN, …]`, `f′(x) = [f₁, f₂x₁, NaN, …]`, `f″(x) = [f₂, NaN, …]`.
pub(super) fn map_all<S: RealScalar, V: Real<S>, const N: usize>(
    x: &[V; N],
    [f0, f1, f2]: [V; 3],
) -> [[V; N]; 3] {
    let nan = V::constant(S::NAN);
    let mut f = [nan; N];
    let mut d = [nan; N];
    let mut dd = [nan; N];
    f[0] = f0;
    d[0] = f1;
    dd[0] = f2;
    if N > 1 {
        f[1] = f1 * x[1];
        d[1] = f2 * x[1];
    }
    if N > 2 {
        f[2] = f1 * x[2] + f2 * (x[1] * x[1]) * S::HALF;
    }
    [f, d, dd]
}
