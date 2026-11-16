//! Chain rules written once for every first-order wrapper.
//!
//! [`Chain`] builds a result from a value and its partials, all in `Primal`
//! arithmetic; `Dual`, `MultiDual` and `Var` implement it, and the functions
//! below are their elementary methods. Each nontrivial partial is also a function
//! of its own (`*_partial`, `*_partials`), which `Jet` uses for its first
//! coefficient. Regime switches evaluate both forms and choose with `select` on
//! base comparisons. Kink rules select components and never multiply.

use scioxide_core::RealScalar;

use crate::Real;
use crate::stable::{self, Stable};

/// How a partial is applied to the incoming derivative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Product {
    /// IEEE products (`×`, scalar factors and the smooth exponential family).
    Ieee,
    /// Absolute-zero products ([`Real::azmul`]).
    Absolute,
}

/// How partials depend on active values, used by sparsity patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// Partials depend on the active arguments.
    Nonlinear,
    /// Partials do not depend on active values.
    Linear,
    /// Each partial is the other argument (a product).
    Bilinear,
}

/// Builds results of elementary operations from values and partials.
pub(crate) trait Chain<S: RealScalar>: Real<S> {
    /// Returns whether `self` carries derivatives (always true for forward types).
    fn is_active(self) -> bool;

    /// Returns `f(self)` with value `value` and partial `partial`.
    fn unary(
        self,
        value: Self::Primal,
        partial: Self::Primal,
        product: Product,
        class: Class,
    ) -> Self;

    /// Returns `f(self, other)` with value `value` and partials `left`, `right`.
    fn binary(
        self,
        other: Self,
        value: Self::Primal,
        left: Self::Primal,
        right: Self::Primal,
        product: Product,
        class: Class,
    ) -> Self;

    /// Counts the lanes of `hits` as kink hits when `self` is active on a tape.
    fn kinks(self, hits: Self::Mask);

    /// Counts the lanes of `hits` once when `self` or `other` is active on a tape.
    fn pair_kinks(self, other: Self, hits: Self::Mask);
}

type P<S, R> = <R as Real<S>>::Primal;

#[inline(always)]
fn constant<S: RealScalar, V: Real<S>>(value: f64) -> V {
    V::constant(S::from_f64_ieee(value))
}

#[inline(always)]
fn unary<S: RealScalar, R: Chain<S>>(
    x: R,
    value: P<S, R>,
    partial: impl FnOnce() -> P<S, R>,
    product: Product,
    class: Class,
) -> R {
    if x.is_active() {
        x.unary(value, partial(), product, class)
    } else {
        R::lift(value)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn binary<S: RealScalar, R: Chain<S>>(
    x: R,
    y: R,
    value: P<S, R>,
    partials: impl FnOnce() -> (P<S, R>, P<S, R>),
    product: Product,
    class: Class,
) -> R {
    if x.is_active() || y.is_active() {
        let (left, right) = partials();
        x.binary(y, value, left, right, product, class)
    } else {
        R::lift(value)
    }
}

/// Returns `x` with every component replaced by zero derivative and value `value`.
#[inline(always)]
fn flat<S: RealScalar, R: Chain<S>>(value: P<S, R>) -> R {
    R::lift(value)
}

use Class::{Linear, Nonlinear};
use Product::{Absolute, Ieee};

/// `d(1/x)/dx = −y²` with `y = 1/x`.
#[inline(always)]
pub(crate) fn recip_partial<S: RealScalar, V: Real<S>>(y: V) -> V {
    -(y * y)
}

#[inline(always)]
pub(crate) fn recip<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let y = x.primal().recip();
    unary(x, y, || recip_partial(y), Absolute, Nonlinear)
}

/// `d√x/dx = 0.5/y` with `y = √x`.
#[inline(always)]
pub(crate) fn sqrt_partial<S: RealScalar, V: Real<S>>(y: V) -> V {
    constant::<S, V>(0.5) / y
}

#[inline(always)]
pub(crate) fn sqrt<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let y = x.primal().sqrt();
    unary(x, y, || sqrt_partial(y), Absolute, Nonlinear)
}

/// `d∛x/dx = (1/y)/(3y)` with `y = ∛x`.
#[inline(always)]
pub(crate) fn cbrt_partial<S: RealScalar, V: Real<S>>(y: V) -> V {
    y.recip() / (y * S::from_f64_ieee(3.0))
}

#[inline(always)]
pub(crate) fn cbrt<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let y = x.primal().cbrt();
    unary(x, y, || cbrt_partial(y), Absolute, Nonlinear)
}

#[inline(always)]
pub(crate) fn exp<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let y = x.primal().exp();
    unary(x, y, || y, Ieee, Nonlinear)
}

/// `d2ˣ/dx = y ln 2` with `y = 2ˣ`.
#[inline(always)]
pub(crate) fn exp2_partial<S: RealScalar, V: Real<S>>(y: V) -> V {
    y * S::LN_2
}

#[inline(always)]
pub(crate) fn exp2<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let y = x.primal().exp2();
    unary(x, y, || exp2_partial(y), Ieee, Nonlinear)
}

#[inline(always)]
pub(crate) fn exp_m1<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    let y = p.exp_m1();
    // eˣ itself keeps relative accuracy where y + 1 cancels (x ≪ 0).
    unary(x, y, || p.exp(), Ieee, Nonlinear)
}

/// `d ln x/dx = 1/x`.
#[inline(always)]
pub(crate) fn ln_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    p.recip()
}

#[inline(always)]
pub(crate) fn ln<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.ln(), || ln_partial(p), Absolute, Nonlinear)
}

/// `d ln(1 + x)/dx = 1/(1 + x)`.
#[inline(always)]
pub(crate) fn ln_1p_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    (p + S::ONE).recip()
}

#[inline(always)]
pub(crate) fn ln_1p<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.ln_1p(), || ln_1p_partial(p), Absolute, Nonlinear)
}

/// `d log₂ x/dx = (1/x) log₂ e`.
#[inline(always)]
pub(crate) fn log2_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    p.recip() * S::LOG2_E
}

#[inline(always)]
pub(crate) fn log2<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.log2(), || log2_partial(p), Absolute, Nonlinear)
}

/// `d log₁₀ x/dx = log₁₀ e / x`.
#[inline(always)]
pub(crate) fn log10_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    V::constant(S::LOG10_E) / p
}

#[inline(always)]
pub(crate) fn log10<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.log10(), || log10_partial(p), Absolute, Nonlinear)
}

#[inline(always)]
pub(crate) fn sin<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.sin(), || p.cos(), Ieee, Nonlinear)
}

#[inline(always)]
pub(crate) fn cos<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.cos(), || -p.sin(), Ieee, Nonlinear)
}

#[inline(always)]
pub(crate) fn sin_cos<S: RealScalar, R: Chain<S>>(x: R) -> (R, R) {
    let p = x.primal();
    let (s, c) = p.sin_cos();
    (
        unary(x, s, || c, Ieee, Nonlinear),
        unary(x, c, || -s, Ieee, Nonlinear),
    )
}

/// `d tan x/dx = 1 + y²` with `y = tan x`.
#[inline(always)]
pub(crate) fn tan_partial<S: RealScalar, V: Real<S>>(y: V) -> V {
    y * y + S::ONE
}

#[inline(always)]
pub(crate) fn tan<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let y = x.primal().tan();
    unary(x, y, || tan_partial(y), Absolute, Nonlinear)
}

fn one_minus_square<S: RealScalar, V: Real<S>>(p: V) -> V {
    (-p + S::ONE) * (p + S::ONE)
}

/// `d arcsin x/dx = 1/√((1 − x)(1 + x))`.
#[inline(always)]
pub(crate) fn asin_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    one_minus_square::<S, V>(p).sqrt().recip()
}

#[inline(always)]
pub(crate) fn asin<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.asin(), || asin_partial(p), Absolute, Nonlinear)
}

/// `d arccos x/dx = −1/√((1 − x)(1 + x))`.
#[inline(always)]
pub(crate) fn acos_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    -asin_partial::<S, V>(p)
}

#[inline(always)]
pub(crate) fn acos<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.acos(), || acos_partial(p), Absolute, Nonlinear)
}

/// `d arctan x/dx = 1/(1 + x²)`; for `|x| > 1`, `r²/(1 + r²)` with `r = 1/x`.
#[inline(always)]
pub(crate) fn atan_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    let near = (p * p + S::ONE).recip();
    let r = p.recip();
    let r2 = r * r;
    let far = r2 / (r2 + S::ONE);
    V::select(p.abs().gt(V::one()), far, near)
}

#[inline(always)]
pub(crate) fn atan<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.atan(), || atan_partial(p), Ieee, Nonlinear)
}

#[inline(always)]
pub(crate) fn sinh<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.sinh(), || p.cosh(), Ieee, Nonlinear)
}

#[inline(always)]
pub(crate) fn cosh<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.cosh(), || p.sinh(), Ieee, Nonlinear)
}

/// `d tanh x/dx = 1 − y²` with `y = tanh x`; for `|x| > 1`, `s²` with
/// `s = 2e/(1 + e²)` and `e = e^{−|x|}`.
#[inline(always)]
pub(crate) fn tanh_partial<S: RealScalar, V: Real<S>>(p: V, y: V) -> V {
    let near = -(y * y) + S::ONE;
    // In the tails 1 − y² cancels; 2e/(1 + e²) with e = e^{−|x|} does not.
    let e = (-p.abs()).exp();
    let s = e * S::TWO / (e * e + S::ONE);
    V::select(p.abs().gt(V::one()), s * s, near)
}

#[inline(always)]
pub(crate) fn tanh<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    let y = p.tanh();
    unary(x, y, || tanh_partial(p, y), Ieee, Nonlinear)
}

/// `d arsinh x/dx = 1/√(x² + 1)`; for `|x| > 1`, `1/hypot(x, 1)`.
#[inline(always)]
pub(crate) fn asinh_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    let near = (p * p + S::ONE).sqrt().recip();
    let far = p.hypot(V::one()).recip();
    V::select(p.abs().gt(V::one()), far, near)
}

#[inline(always)]
pub(crate) fn asinh<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.asinh(), || asinh_partial(p), Ieee, Nonlinear)
}

/// `d arcosh x/dx = 1/√((x − 1)(x + 1))`; for `x > 2`, `1/(√(x − 1)·√(x + 1))`.
#[inline(always)]
pub(crate) fn acosh_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    let near = ((p - S::ONE) * (p + S::ONE)).sqrt().recip();
    let far = (p - S::ONE).sqrt().recip() / (p + S::ONE).sqrt();
    V::select(p.gt(V::constant(S::TWO)), far, near)
}

#[inline(always)]
pub(crate) fn acosh<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.acosh(), || acosh_partial(p), Absolute, Nonlinear)
}

/// `d artanh x/dx = 1/((1 − x)(1 + x))`.
#[inline(always)]
pub(crate) fn atanh_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    one_minus_square::<S, V>(p).recip()
}

#[inline(always)]
pub(crate) fn atanh<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(x, p.atanh(), || atanh_partial(p), Absolute, Nonlinear)
}

#[inline(always)]
pub(crate) fn to_degrees<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(
        x,
        p.to_degrees(),
        || <P<S, R>>::constant(S::ONE.to_degrees()),
        Ieee,
        Linear,
    )
}

#[inline(always)]
pub(crate) fn to_radians<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    unary(
        x,
        p.to_radians(),
        || <P<S, R>>::constant(S::ONE.to_radians()),
        Ieee,
        Linear,
    )
}

/// `dxⁿ/dx = n xⁿ⁻¹`, and 0 for `n = 0`; at `n = i32::MIN`, `xⁿ⁻¹ = xⁿ/x`.
#[inline(always)]
pub(crate) fn powi_partial<S: RealScalar, V: Real<S>>(p: V, exponent: i32) -> V {
    match (exponent, exponent.checked_sub(1)) {
        (0, _) => V::zero(),
        (_, Some(lower)) => p.powi(lower) * S::from_f64_ieee(f64::from(exponent)),
        (_, None) => p.powi(exponent) / p * S::from_f64_ieee(f64::from(exponent)),
    }
}

#[inline(always)]
pub(crate) fn powi<S: RealScalar, R: Chain<S>>(x: R, exponent: i32) -> R {
    let p = x.primal();
    unary(
        x,
        p.powi(exponent),
        || powi_partial(p, exponent),
        Absolute,
        Nonlinear,
    )
}

/// The stable transforms as `map_scalar` rules.
#[inline(always)]
pub(crate) fn stable<S: RealScalar, R: Real<S>>(x: R, function: Stable) -> R {
    x.map_scalar(|v| stable::rule(function, v))
}

// ---- Kinks.

/// `d|x|/dx = sign x`, and 0 at the kink `x = 0`.
#[inline(always)]
pub(crate) fn abs_partial<S: RealScalar, V: Real<S>>(p: V) -> V {
    let zero = V::zero();
    V::select(p.gt(zero), V::one(), V::select(p.lt(zero), -V::one(), zero))
}

#[inline(always)]
pub(crate) fn abs<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    let zero = <P<S, R>>::zero();
    x.kinks(p.ge(zero) & p.le(zero));
    unary(x, p.abs(), || abs_partial(p), Absolute, Linear)
}

/// Rounding functions: zero derivative, counting the given kink lanes.
fn piecewise_constant<S: RealScalar, R: Chain<S>>(x: R, value: P<S, R>, hits: R::Mask) -> R {
    x.kinks(hits);
    flat(value)
}

fn is_integer<S: RealScalar, B: Real<S>>(b: B) -> B::Mask {
    b.is_finite() & b.floor().ge(b) & b.floor().le(b)
}

fn is_nonzero_integer<S: RealScalar, B: Real<S>>(b: B) -> B::Mask {
    is_integer::<S, B>(b) & !(b.ge(B::zero()) & b.le(B::zero()))
}

#[inline(always)]
pub(crate) fn floor<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    piecewise_constant(x, p.floor(), is_integer::<S, _>(p.base()))
}

#[inline(always)]
pub(crate) fn ceil<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    piecewise_constant(x, p.ceil(), is_integer::<S, _>(p.base()))
}

#[inline(always)]
pub(crate) fn round<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    // Half-integers only: above 2⁵² every value is an integer and `x − ½` rounds
    // to one.
    let b = p.base();
    let hits = is_integer::<S, _>(b - S::HALF) & !is_integer::<S, _>(b);
    piecewise_constant(x, p.round(), hits)
}

#[inline(always)]
pub(crate) fn trunc<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    piecewise_constant(x, p.trunc(), is_nonzero_integer::<S, _>(p.base()))
}

#[inline(always)]
pub(crate) fn signum<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    let b = p.base();
    piecewise_constant(x, p.signum(), b.ge(R::Base::zero()) & b.le(R::Base::zero()))
}

#[inline(always)]
pub(crate) fn fract<S: RealScalar, R: Chain<S>>(x: R) -> R {
    let p = x.primal();
    x.kinks(is_nonzero_integer::<S, _>(p.base()));
    unary(x, p.fract(), <P<S, R>>::one, Ieee, Linear)
}

/// Where `a % b` has a kink: `r = a % b` is zero and `a` is not, so `a/b` is a
/// non-zero integer.
#[inline(always)]
pub(crate) fn rem_kink<S: RealScalar, V: Real<S>>(x: V, r: V) -> V::Mask {
    let (x, r) = (x.base(), r.base());
    let zero = V::Base::zero();
    r.ge(zero) & r.le(zero) & !(x.ge(zero) & x.le(zero))
}

/// `∂(a % b) = (1, −n)` with `r = a % b` and `n = (a − r)/b` the integer quotient
/// that `%` used, taken from base values; `fl(a/b)` can round to the next integer.
/// At a kink, `a/b = n ≠ 0`, the one-sided partials in `b` are `−n` and
/// `−(n − sign n)`, and the midpoint `−(n − ½·sign n)` is returned.
#[inline(always)]
pub(crate) fn rem_partials<S: RealScalar, V: Real<S>>(x: V, y: V, r: V) -> (V, V) {
    let n = ((x.base() - r.base()) / y.base()).round();
    let midpoint = V::passive(n - n.signum() * S::HALF);
    let n = V::passive(n);
    (V::one(), V::select(rem_kink::<S, V>(x, r), -midpoint, -n))
}

#[inline(always)]
pub(crate) fn rem<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    let r = x % y;
    a.pair_kinks(b, rem_kink::<S, _>(x, r));
    binary(a, b, r, || rem_partials(x, y, r), Absolute, Linear)
}

/// The partials of `min` (`larger = false`) or `max` (`larger = true`): 1 for the
/// returned operand, ½ each at a tie, 0 for a NaN operand and NaN if both are.
#[inline(always)]
pub(crate) fn extremum_partials<S: RealScalar, V: Real<S>>(x: V, y: V, larger: bool) -> (V, V) {
    let one = V::one();
    let zero = V::zero();
    let half = V::constant(S::HALF);
    let nan = V::constant(S::NAN);
    let tie = x.ge(y) & x.le(y);
    let left_wins = if larger { x.gt(y) } else { x.lt(y) };
    let right_wins = if larger { x.lt(y) } else { x.gt(y) };
    let x_nan = !(x.ge(x));
    let y_nan = !(y.ge(y));
    // Ordered pair: one side wins or they tie. Otherwise a NaN operand is ignored.
    let left = V::select(
        left_wins,
        one,
        V::select(
            right_wins,
            zero,
            V::select(
                tie,
                half,
                V::select(x_nan & y_nan, nan, V::select(y_nan, one, zero)),
            ),
        ),
    );
    let right = V::select(
        right_wins,
        one,
        V::select(
            left_wins,
            zero,
            V::select(
                tie,
                half,
                V::select(x_nan & y_nan, nan, V::select(x_nan, one, zero)),
            ),
        ),
    );
    (left, right)
}

/// `min` (`larger = false`) or `max` (`larger = true`).
#[inline(always)]
pub(crate) fn extremum<S: RealScalar, R: Chain<S>>(a: R, b: R, larger: bool) -> R {
    let (x, y) = (a.primal(), b.primal());
    let value = if larger { x.max(y) } else { x.min(y) };
    extremum_valued(a, b, larger, value)
}

/// [`extremum`] with the given value.
#[inline(always)]
fn extremum_valued<S: RealScalar, R: Chain<S>>(a: R, b: R, larger: bool, value: P<S, R>) -> R {
    let (x, y) = (a.primal(), b.primal());
    let tie = x.ge(y) & x.le(y);
    a.pair_kinks(b, tie);
    binary(
        a,
        b,
        value,
        || extremum_partials(x, y, larger),
        Absolute,
        Linear,
    )
}

#[inline(always)]
pub(crate) fn clamp<S: RealScalar, R: Chain<S>>(x: R, minimum: R, maximum: R) -> R {
    let valid = minimum.le(maximum);
    // The value is the clamp of the primal, so a NaN `x` stays NaN and `−0` keeps its
    // sign; the partials are those of `min(max(x, minimum), maximum)`.
    let value = x.primal().clamp(minimum.primal(), maximum.primal());
    let clamped = extremum_valued(extremum(x, minimum, true), maximum, false, value);
    let base = x.primal().base();
    let nan = !base.ge(base);
    R::select(valid, R::select(nan, x, clamped), R::constant(S::NAN))
}

// ---- Binary smooth functions.

#[inline(always)]
pub(crate) fn mul<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    binary(a, b, x * y, || (y, x), Ieee, Class::Bilinear)
}

/// `∂(a/b) = (1/b, −q/b)` with `q = a/b`.
#[inline(always)]
pub(crate) fn div_partials<S: RealScalar, V: Real<S>>(y: V, q: V) -> (V, V) {
    (y.recip(), -(q / y))
}

#[inline(always)]
pub(crate) fn div<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    let q = x / y;
    binary(a, b, q, || div_partials(y, q), Absolute, Nonlinear)
}

/// `∂aᵇ/∂a = azmul(b, aᵇ⁻¹)`, given `z = aᵇ`. Where `a ≠ 0` and `z` is finite and
/// non-zero, `aᵇ⁻¹` is computed as `z/a`: `aᵇ⁻¹` itself carries the rounding of
/// `b − 1` amplified by `|ln a|`.
#[inline(always)]
pub(crate) fn powf_left<S: RealScalar, V: Real<S>>(x: V, y: V, z: V) -> V {
    let (xb, zb) = (x.base(), z.base());
    let zero = V::Base::zero();
    let nonzero = |v: V::Base| !(v.ge(zero) & v.le(zero));
    let ratio = nonzero(xb) & nonzero(zb) & zb.is_finite();
    V::select(ratio, y.azmul(z / x), y.azmul(x.powf(y - S::ONE)))
}

/// `∂aᵇ/∂b = azmul(z, ln a)` with `z = aᵇ`.
#[inline(always)]
pub(crate) fn powf_right<S: RealScalar, V: Real<S>>(x: V, z: V) -> V {
    z.azmul(x.ln())
}

#[inline(always)]
pub(crate) fn powf<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    let z = x.powf(y);
    let partials = || {
        let right = if b.is_active() {
            powf_right(x, z)
        } else {
            <P<S, R>>::zero()
        };
        (powf_left(x, y, z), right)
    };
    binary(a, b, z, partials, Absolute, Nonlinear)
}

#[inline(always)]
pub(crate) fn powm1<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    let partials = || {
        let z = x.powf(y);
        let right = if b.is_active() {
            powf_right(x, z)
        } else {
            <P<S, R>>::zero()
        };
        (powf_left(x, y, z), right)
    };
    binary(a, b, x.powm1(y), partials, Absolute, Nonlinear)
}

fn is_infinite<S: RealScalar, V: Real<S>>(x: V) -> V::Mask {
    let base = x.base();
    !base.is_finite() & base.ge(base)
}

fn origin<S: RealScalar, V: Real<S>>(x: V, y: V) -> V::Mask {
    let zero = V::zero();
    x.ge(zero) & x.le(zero) & y.ge(zero) & y.le(zero)
}

/// `∂hypot(a, b) = (a/h, b/h)`, evaluated on `(a, b)/max(|a|, |b|)`; `(0, 0)` at
/// the origin.
#[inline(always)]
pub(crate) fn hypot_partials<S: RealScalar, V: Real<S>>(x: V, y: V, h: V) -> (V, V) {
    let zero = V::zero();
    let at_origin = origin::<S, V>(x, y);
    let scale = x.abs().max(y.abs());
    let (u, v) = (x / scale, y / scale);
    let n = u.hypot(v);
    let scaled = scale.base().is_finite() & scale.gt(zero);
    let left = V::select(scaled, u / n, x / h);
    let right = V::select(scaled, v / n, y / h);
    (
        V::select(at_origin, zero, left),
        V::select(at_origin, zero, right),
    )
}

#[inline(always)]
pub(crate) fn hypot<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    let h = x.hypot(y);
    let at_origin = origin::<S, P<S, R>>(x, y);
    a.pair_kinks(b, at_origin);
    binary(a, b, h, || hypot_partials(x, y, h), Absolute, Nonlinear)
}

/// `∂atan2(a, b) = (b′, −a′)/((a′² + b′²)·m)` with `m = max(|a|, |b|)`,
/// `a′ = a/m`, `b′ = b/m`; `(0, 0)` at the origin and where exactly one operand
/// is infinite.
#[inline(always)]
pub(crate) fn atan2_partials<S: RealScalar, V: Real<S>>(y: V, x: V) -> (V, V) {
    let zero = V::zero();
    let scale = y.abs().max(x.abs());
    let (u, v) = (y / scale, x / scale);
    let q = u * u + v * v;
    let left = (v / q) / scale;
    let right = -((u / q) / scale);
    // Exactly one infinite operand: the angle is locally constant.
    let (y_infinite, x_infinite) = (is_infinite::<S, V>(y), is_infinite::<S, V>(x));
    let one_infinite = (y_infinite & !x_infinite) | (x_infinite & !y_infinite);
    let flat = origin::<S, V>(x, y) | one_infinite;
    (V::select(flat, zero, left), V::select(flat, zero, right))
}

#[inline(always)]
pub(crate) fn atan2<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (y, x) = (a.primal(), b.primal());
    let zero = <P<S, R>>::zero();
    let at_origin = origin::<S, P<S, R>>(x, y);
    let cut = y.ge(zero) & y.le(zero) & x.lt(zero);
    a.pair_kinks(b, at_origin | cut);
    binary(
        a,
        b,
        y.atan2(x),
        || atan2_partials(y, x),
        Absolute,
        Nonlinear,
    )
}

/// `∂log_b x = (1/(x ln b), −y/(b ln b))` with `y = log_b x`.
#[inline(always)]
pub(crate) fn log_partials<S: RealScalar, V: Real<S>>(x: V, base: V, y: V) -> (V, V) {
    let ln_base = base.ln();
    ((x * ln_base).recip(), -(y / (base * ln_base)))
}

#[inline(always)]
pub(crate) fn log<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, base) = (a.primal(), b.primal());
    let y = x.log(base);
    binary(a, b, y, || log_partials(x, base, y), Absolute, Nonlinear)
}

/// `∂logaddexp(a, b) = (σ(a − b), σ(b − a))`; where `a − b` is not finite, 1 for
/// the larger argument and 0 for the other.
#[inline(always)]
pub(crate) fn logaddexp_partials<S: RealScalar, V: Real<S>>(x: V, y: V) -> (V, V) {
    let delta = x - y;
    // The rounding error `e` of `x − y` (TwoSum) enters through
    // `σ(δ + e) ≈ σ(δ)(1 + σ(−δ)·e)`; uncorrected it is amplified by `|δ|`.
    let back = delta - x;
    let error = (x - (delta - back)) + (-y - back);
    let (p, q) = (delta.sigmoid(), (-delta).sigmoid());
    let one = V::one();
    let zero = V::zero();
    let finite = delta.base().is_finite() & error.base().is_finite();
    let larger = x.gt(y);
    (
        V::select(finite, p + p * q * error, V::select(larger, one, zero)),
        V::select(finite, q - q * p * error, V::select(larger, zero, one)),
    )
}

#[inline(always)]
pub(crate) fn logaddexp<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    binary(
        a,
        b,
        x.logaddexp(y),
        || logaddexp_partials(x, y),
        Absolute,
        Nonlinear,
    )
}

/// Whether both base values are zero.
fn both_zero<S: RealScalar, V: Real<S>>(x: V, y: V) -> V::Mask {
    let zero = V::Base::zero();
    let (x, y) = (x.base(), y.base());
    x.ge(zero) & x.le(zero) & y.ge(zero) & y.le(zero)
}

/// `∂(a ln b) = (ln b, a/b)`; the function is 0 on the line `a = 0`, so `∂/∂b` is
/// 0 at `a = b = 0` instead of `0/0`.
#[inline(always)]
pub(crate) fn xlogy_partials<S: RealScalar, V: Real<S>>(x: V, y: V) -> (V, V) {
    (y.ln(), V::select(both_zero::<S, V>(x, y), V::zero(), x / y))
}

#[inline(always)]
pub(crate) fn xlogy<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    binary(
        a,
        b,
        x.xlogy(y),
        || xlogy_partials(x, y),
        Absolute,
        Nonlinear,
    )
}

/// `∂(a ln(1 + b)) = (ln(1 + b), a/(1 + b))`, with `∂/∂b = 0` at `a = 0`, `b = −1`.
#[inline(always)]
pub(crate) fn xlog1py_partials<S: RealScalar, V: Real<S>>(x: V, y: V) -> (V, V) {
    let shifted = y + S::ONE;
    (
        y.ln_1p(),
        V::select(both_zero::<S, V>(x, shifted), V::zero(), x / shifted),
    )
}

#[inline(always)]
pub(crate) fn xlog1py<S: RealScalar, R: Chain<S>>(a: R, b: R) -> R {
    let (x, y) = (a.primal(), b.primal());
    binary(
        a,
        b,
        x.xlog1py(y),
        || xlog1py_partials(x, y),
        Absolute,
        Nonlinear,
    )
}
