//! `f64` real Bessel kernels for nonnegative order and positive argument.
//!
//! - Scaled I₀ uses dedicated Chebyshev expansions (`bessel_i0`).
//! - J (x <= 2) and other I (below the modified expansion) use the defining series.
//!   The prefix (x/2)^nu/Gamma(1+nu) comes from `pow` and the two-term
//!   reciprocal-gamma series; terms above 2^-10 of the sum are two-term, the
//!   geometric remainder `f64`.
//! - Y and K at 0 < x <= 2 use Temme's series in `f64` with two-term
//!   leading seeds and a first-order error estimate; past its calibrated
//!   envelope (near zeros of Y, and K near x = 2) the two-term Temme series
//!   runs instead.
//! - J and Y at x > 2 use Steed's method: `f64` CF1 below the turning
//!   region, otherwise two-term CF1 evaluated backward from past the turning
//!   point (deeper near zeros of J_nu); CF2 evaluated backward in `f64` with
//!   a two-term final level; two-term order recurrences and homogeneous
//!   normalization. Near zeros of Y_nu, CF1 restarts deeper and CF2 runs in
//!   two-term arithmetic.
//! - K at x > 2 uses Steed's CF2 for K and forward recurrence.
//! - Hankel expansions for J/Y and I/K are accepted only when their terms
//!   reach relative `f64` accuracy before diverging; the J/Y phase is
//!   reduced in two-term arithmetic so the small factor near a zero stays
//!   relative, and the I/K prefactors are two-term.
//!
//! Other integer orders run the same kernels at u = 0, where the reciprocal-gamma
//! factor is exactly one and series denominators are exact. Two-term
//! arithmetic (`bessel_base`) is confined to the cancellation regions named
//! above and to prefixes outside the `f64` range.
use crate::bessel_airy::bessel_base as wide;
use crate::bessel_airy::bessel_base::Wide as D;
use crate::support::Work;
use crate::support::gamma_series::{reciprocal_gamma, reciprocal_gamma_parts};
use scioxide_core::{RealScalar, SciOxideError};

const EPS: f64 = f64::EPSILON;
/// Series terms below this fraction of the sum do not change the result.
const SERIES_TAIL: f64 = 1.0 / 1_152_921_504_606_846_976.0;
/// Series terms below this fraction of the sum need only `f64` accuracy.
const SMALL_TERM: f64 = 1.0 / 1024.0;
/// Order offsets from an integer below which Temme uses the integer limits.
const TEMME_INTEGER: f64 = 1.0e-30;
/// Accepted Temme/recurrence error estimate, in units of the result's epsilon.
const TEMME_ENVELOPE: f64 = 32.0;
/// Turning-point units past which CF1's `f64` tail starts: Ai there is
/// 2^-8 (shallow) or 2^-40 (deep) of its value at the turning point.
const SHALLOW_MARGIN: f64 = 4.5;
const DEEP_MARGIN: f64 = 12.0;
/// |J_(nu+1)/J_nu| beyond which J_nu is near enough a zero for the shallow
/// tail's damped rounding to matter.
const NEAR_ZERO: f64 = 1024.0;
/// Largest Y_nu modulus-to-value ratio accepted with the mixed-precision CF2,
/// whose eps/16 rounding reaches Y_nu amplified by that ratio.
const CF2_AMPLIFICATION: f64 = 16.0;
/// Beyond this argument the I series takes the log-space two-term path, whose
/// ordinary value cannot overflow before the final projection.
const SERIES_ARGUMENT: f64 = 600.0;
/// Argument below which Hankel expansions are not attempted.
const HANKEL_ARGUMENT: f64 = 20.0;
/// Argument from which J/Y Hankel expansions are attempted at every order
/// with nu^2 <= 4x. Below it only orders whose leading Q term is at most 1/32
/// qualify: rounding near a zero scales with |Q|, and Steed's two-term CF1 is
/// exact there.
const HANKEL_GENERAL: f64 = 32.0;
/// Below this argument Steed is cheaper than compensated Hankel sums.
const COMPENSATED_HANKEL_ARGUMENT: f64 = 256.0;
/// Phases beyond this argument use exact-reduction sine/cosine instead of a
/// two-term multiple of pi/2.
const REDUCED_PHASE: f64 = 67_108_864.0;
use crate::support::constants::FRAC_1_PI as FRAC_1_PI_WIDE;
use crate::support::constants::FRAC_PI_2 as HALF_PI_WIDE;
use crate::support::constants::PI as PI_WIDE;
use crate::support::constants::TAU as TAU_WIDE;

/// Bessel J_nu, nu >= -1/2 and x > 0.
pub(super) fn j(order: f64, x: f64, w: &mut Work) -> Result<f64, SciOxideError> {
    if order == 0.0 || order == 1.0 {
        return super::bessel_low::value(order, x, false, w);
    }
    if order >= 1000.0 && order > 2.0 * x.sqrt() {
        return Ok(super::bessel_uniform::evaluate(order, x, false, w)?.value);
    }
    if x <= 2.0 {
        return match series(order, x, false, false, w)? {
            Some(value) => Ok(value),
            None => wide::power(order, x, false, false, w),
        };
    }
    if let Some((value, _)) = hankel(order, x, w)? {
        return Ok(value);
    }
    steed(order, x, false, w)
}

/// Bessel Y_nu, nu >= -1/2 and x > 0.
pub(super) fn y(order: f64, x: f64, w: &mut Work) -> Result<f64, SciOxideError> {
    if order == 0.0 || order == 1.0 {
        return super::bessel_low::value(order, x, true, w);
    }
    if order >= 1000.0 && order > 2.0 * x.sqrt() {
        return Ok(super::bessel_uniform::evaluate(order, x, true, w)?.value);
    }
    if x <= 2.0 {
        return y_small(order, x, w);
    }
    if let Some((_, value)) = hankel(order, x, w)? {
        return Ok(value);
    }
    steed(order, x, true, w)
}

/// Compensated oscillatory evaluation when a low-order rational loses digits.
pub(super) fn low_cancellation(
    order: f64,
    x: f64,
    second: bool,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    if let Some((j, y)) = compensated_hankel(order, x, w)? {
        return Ok(if second { y } else { j });
    }
    steed(order, x, second, w)
}

/// Modified Bessel I_nu, 0 <= nu < 32 and x > 0, optionally times exp(-x).
pub(super) fn i(order: f64, x: f64, scaled: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    if order == 0.0 && scaled {
        return super::bessel_i0::scaled(x, w);
    }
    if x > HANKEL_ARGUMENT
        && let Some(value) = modified_hankel(order, x, false, w)?
    {
        return Ok(if scaled {
            value.value()
        } else if x < 700.0 {
            wide::scale(value, x.exp()).value()
        } else {
            // Split the growth so the scaled value survives until the final product.
            let half = (0.5 * x).exp();
            value.value() * half * half
        });
    }
    match series(order, x, true, scaled, w)? {
        Some(value) => Ok(value),
        None => wide::power(order, x, true, scaled, w),
    }
}

/// Modified Bessel K_nu, 0 <= nu < 32 and x > 0, optionally times exp(x).
pub(super) fn k(order: f64, x: f64, scaled: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    w.try_step()?;
    let index = order.round();
    if index > w.remaining as f64 {
        w.complete = false;
        w.check()?;
    }
    let n = index.try_to_usize_exact()?;
    let u = order - index;
    let (mut previous, mut current) = if x <= 2.0 {
        let seed = temme(u, x, true, w)?;
        let (value, next) = if seed.within(TEMME_ENVELOPE) {
            (seed.value, seed.next)
        } else {
            wide::temme(u, x, true, w)?
        };
        let scale = x.exp();
        (value * scale, next * scale)
    } else if x < 40.0 {
        k_fraction(u, x, w)?
    } else {
        match (
            modified_hankel(u, x, true, w)?,
            modified_hankel(u + 1.0, x, true, w)?,
        ) {
            (Some(value), Some(next)) => (value.value(), next.value()),
            _ => k_fraction(u, x, w)?,
        }
    };
    for k in 1..=n {
        w.try_step()?;
        let next = 2.0 * ((order - (n - k) as f64) / x) * current + previous;
        previous = current;
        current = next;
    }
    if scaled {
        return Ok(previous);
    }
    if previous <= 0.0 {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "scaled K kernel produced a non-positive value",
        ));
    }
    let half = (-0.5 * x).exp();
    Ok(previous * half * half)
}

/// (x/2)^nu / Gamma(1+nu) for nu = n + u, |u| <= 1/2, or None outside
/// the comfortable normal range of `f64`.
fn prefix(order: f64, x: f64, w: &mut Work) -> Result<Option<D>, SciOxideError> {
    w.try_step()?;
    let index = order.round().max(0.0);
    let power = (0.5 * x).powf(order);
    if index > 160.0 || x < 1.0e-300 || !(power > 1.0e-290 && power < 1.0e290) {
        return Ok(None);
    }
    let n = index.try_to_usize_exact()?;
    let u = order - index;
    // At integer orders 1/Gamma(1+u) is exactly one.
    let numerator = if u == 0.0 {
        D::from(power)
    } else {
        wide::scale(reciprocal_gamma(u), power)
    };
    // Gamma(1+nu) = Gamma(1+u) prod (u+k); each factor nu-(n-k) is exact.
    let value = if n == 0 {
        numerator
    } else {
        let product = (1..=n).fold(D::from(1.0), |product, k| {
            wide::scale(product, order - (n - k) as f64)
        });
        wide::quotient(numerator, product)
    };
    Ok((value.hi > 1.0e-290 && value.hi < 1.0e290).then_some(value))
}

/// J_nu (`modified == false`, x <= 2) or I_nu by the defining series,
/// optionally times exp(-x); None when the prefix leaves the `f64` range.
fn series(
    order: f64,
    x: f64,
    modified: bool,
    scaled: bool,
    w: &mut Work,
) -> Result<Option<f64>, SciOxideError> {
    if x > SERIES_ARGUMENT {
        return Ok(None);
    }
    let Some(prefix) = prefix(order, x, w)? else {
        return Ok(None);
    };
    let factor = wide::scale(wide::two_product(x, x), if modified { 0.25 } else { -0.25 });
    let shift = D::from(order);
    let integer = order == order.round() && order < 1.0e6;
    let mut term = D::from(1.0);
    // Leading sum with the exact addition errors and low parts compensated.
    let (mut sum, mut compensation) = (1.0_f64, 0.0_f64);
    let mut k = 0.0;
    // Two-term terms while they carry more than SMALL_TERM of the sum.
    loop {
        w.try_step()?;
        k += 1.0;
        // The ratio does not depend on the running term, keeping its division
        // off the dependent chain.
        // At integer orders k (nu + k) is an exact binary64 integer.
        let denominator = if integer {
            D::from(k * (order + k))
        } else {
            wide::scale(wide::sum(shift, D::from(k)), k)
        };
        let ratio = wide::quotient(factor, denominator);
        term = wide::product(term, ratio);
        let next = sum + term.hi;
        let shifted = next - sum;
        compensation += (sum - (next - shifted)) + (term.hi - shifted) + term.lo;
        sum = next;
        if ratio.hi.abs() < 1.0 && term.hi.abs() <= SMALL_TERM * sum.abs() {
            break;
        }
    }
    // The geometrically decreasing remainder tolerates binary64 terms.
    let mut tail = term.hi;
    while tail.abs() > SERIES_TAIL * sum.abs() {
        w.try_step()?;
        k += 1.0;
        tail *= factor.hi / (k * (order + k));
        compensation += tail;
    }
    let value = wide::product(prefix, wide::sum(D::from(sum), D::from(compensation)));
    Ok(Some(if scaled {
        wide::scale(value, (-x).exp()).value()
    } else {
        value.value()
    }))
}

/// Temme seeds with first-order absolute error estimates.
struct Seed {
    value: f64,
    next: f64,
    error: f64,
    next_error: f64,
}

impl Seed {
    fn within(&self, envelope: f64) -> bool {
        self.error <= envelope * EPS * self.value.abs()
            && self.next_error <= envelope * EPS * self.next.abs()
    }
}

/// Y_u, Y_(u+1) or K_u, K_(u+1) for |u| <= 1/2 and 0 < x <= 2 (Temme 1975).
///
/// p_0 = Gamma(1+u) (x/2)^-u / 2 and q_0 = Gamma(1-u) (x/2)^u / 2 (times 2/pi
/// for Y) carry two terms: for small x they are the second seed, which every
/// growing order inherits.
fn temme(u: f64, x: f64, modified: bool, w: &mut Work) -> Result<Seed, SciOxideError> {
    w.try_step()?;
    // Below this the order's offset from an integer changes no rounded digit,
    // and the u = 0 limits avoid subnormal intermediates.
    let u = if u.abs() < TEMME_INTEGER { 0.0 } else { u };
    // g1 = (1/Gamma(1-u) - 1/Gamma(1+u))/(2u) = -O, g2 = (1/Gamma(1-u) + 1/Gamma(1+u))/2 = E.
    let (even, odd) = reciprocal_gamma_parts(u);
    let (g1, g2) = (-odd.value(), even.value());
    let a = if x < f64::MIN_POSITIVE {
        x.ln() - core::f64::consts::LN_2
    } else {
        (0.5 * x).ln()
    };
    let sigma = -u * a;
    let growth = sigma.exp_m1();
    let cosh = 0.5 * (growth + 1.0 + 1.0 / (growth + 1.0));
    let sinhc = if sigma == 0.0 {
        1.0
    } else {
        growth / sigma * 0.5 * (1.0 + 1.0 / (growth + 1.0))
    };
    let plus = wide::sum(even, wide::scale(odd, u));
    let minus = wide::difference(even, wide::scale(odd, u));
    let sinc = plus.value() * minus.value();
    let lead = g1 * cosh;
    let logarithmic = g2 * a * sinhc;
    let (scale, half_scale) = if modified {
        (1.0, D::from(0.5))
    } else {
        (core::f64::consts::FRAC_2_PI, FRAC_1_PI_WIDE)
    };
    let mut f = scale * (lead - logarithmic) / sinc;
    // Rounding in f's leading cancellation is carried into every later f_k.
    let f_error = scale * (lead.abs() + logarithmic.abs()) / sinc;
    let power = if x < f64::MIN_POSITIVE {
        (u * a).exp()
    } else {
        (0.5 * x).powf(u)
    };
    let p0 = wide::quotient(half_scale, wide::scale(plus, power));
    let q0 = wide::scale(wide::quotient(half_scale, minus), power);
    let (mut p, mut q) = (p0.value(), q0.value());
    // 2 sin(pi u/2)^2/u, zero at the integer limit.
    let e = if modified || u == 0.0 {
        0.0
    } else {
        let sine = (core::f64::consts::FRAC_PI_2 * u).sin();
        2.0 * sine * sine / u
    };
    let mut sum = e * q + f;
    let mut rest1 = 0.0;
    let mut magnitude = f_error + (e * q).abs();
    let mut magnitude1 = 0.0;
    let mut coefficient = 1.0;
    let factor = 0.25 * x * x * if modified { 1.0 } else { -1.0 };
    for k in 1..usize::MAX {
        w.try_step()?;
        let k = k as f64;
        // Reciprocals keep the divisions off the dependent chains.
        let (lower, upper) = (1.0 / (k - u), 1.0 / (k + u));
        f = (k * f + p + q) * (lower * upper);
        p *= lower;
        q *= upper;
        let g = e * q + f;
        let h = p - k * g;
        coefficient *= factor / k;
        let term = coefficient * g;
        let term1 = coefficient * h;
        sum += term;
        rest1 += term1;
        magnitude += term.abs();
        magnitude1 += term1.abs();
        if term.abs() <= 0.25 * EPS * magnitude
            && term1.abs() <= 0.25 * EPS * (magnitude1 + p0.hi.abs())
        {
            break;
        }
    }
    let sign = if modified { 1.0 } else { -1.0 };
    let numerator = wide::scale(wide::sum(p0, D::from(rest1)), sign * 2.0);
    let leading = numerator.hi / x;
    let next = if leading.is_infinite() {
        leading
    } else {
        wide::quotient(numerator, D::from(x)).value()
    };
    // f_error propagates into f_k with weight 1/k!, so its total weight is
    // bounded by the I-type sum of the coefficients, below 2.3 at x = 2.
    Ok(Seed {
        value: sign * sum,
        next,
        error: EPS * (4.0 * magnitude + 2.3 * f_error),
        next_error: EPS
            * (((p0.hi.abs() + 4.0 * magnitude1 + 1.6 * f_error) / x) * 2.0 + next.abs()),
    })
}

/// Y_nu at 0 < x <= 2 from Temme seeds and two-term forward recurrence. The
/// two-term Temme path runs where the first-order estimate exceeds the
/// envelope, that is near zeros of Y_u or Y_(u+1).
fn y_small(order: f64, x: f64, w: &mut Work) -> Result<f64, SciOxideError> {
    w.try_step()?;
    let index = order.round().max(0.0);
    if index > w.remaining as f64 {
        w.complete = false;
        w.check()?;
    }
    let n = index.try_to_usize_exact()?;
    let u = order - index;
    let seed = temme(u, x, false, w)?;
    let (value, error) = wide::forward_y(
        u,
        x,
        n,
        (D::from(seed.value), seed.error),
        (D::from(seed.next), seed.next_error),
        w,
    )?;
    if error <= TEMME_ENVELOPE * EPS * value.abs() {
        Ok(value)
    } else {
        wide::y_small(order, x, w)
    }
}

/// J_(nu+1)/J_nu where x <= nu: J_nu is recessive and zero-free there, so the
/// CF1 denominators stay positive and `f64` updates do not amplify.
fn monotone_ratio(order: f64, x: f64, w: &mut Work) -> Result<f64, SciOxideError> {
    let mut c = 1e-150;
    let mut d = 0.0;
    let mut f = 1e-150;
    for k in 1..usize::MAX {
        w.try_step()?;
        let b = 2.0 * (order + k as f64) / x;
        c = b - 1.0 / c;
        d = (b - d).recip();
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= EPS {
            break;
        }
    }
    Ok(-f)
}

/// J_(nu+1)/J_nu and the sign of J_nu where x > nu. CF1 is evaluated backward
/// from `margin` turning-point units (x/2)^(1/3) past the turning point; the
/// oscillatory levels below it amplify roundoff and run in two-term
/// arithmetic. The `f64` tail ratio reaches J_(nu+1)/J_nu damped by
/// (J_K/J_(nu+1))^2 but amplified by the inverse of J_nu near its zeros.
fn oscillatory_ratio(
    order: f64,
    x: f64,
    margin: f64,
    w: &mut Work,
) -> Result<(D, f64), SciOxideError> {
    let levels = (x - order + margin * (0.5 * x).cbrt()).ceil();
    let tail = monotone_ratio(order + levels, x, w)?;
    let step = wide::quotient(D::from(2.0), D::from(x));
    let mut ratio = D::from(tail);
    // J_nu changes sign once per negative ratio below the positive tail.
    let mut negative = false;
    for k in (1..=levels.try_to_usize_exact()?).rev() {
        w.try_step()?;
        let b = wide::product(wide::sum(D::from(order), D::from(k as f64)), step);
        ratio = wide::quotient(D::from(1.0), wide::difference(b, ratio));
        negative ^= ratio.hi < 0.0;
    }
    Ok((ratio, if negative { -1.0 } else { 1.0 }))
}

/// J_(nu+1)/J_nu and the sign of J_nu, with CF1 restarted deeper wherever the
/// shallow ratio shows J_nu near a zero (`deep` forces the restart). Below the
/// turning region J_nu is recessive and zero-free, and `f64` CF1 suffices.
fn ratio(order: f64, x: f64, deep: bool, w: &mut Work) -> Result<(D, f64), SciOxideError> {
    if x + SHALLOW_MARGIN * (0.5 * x).cbrt() <= order {
        return Ok((D::from(monotone_ratio(order, x, w)?), 1.0));
    }
    if !deep {
        let shallow = oscillatory_ratio(order, x, SHALLOW_MARGIN, w)?;
        if shallow.0.hi.abs() <= NEAR_ZERO {
            return Ok(shallow);
        }
    }
    oscillatory_ratio(order, x, DEEP_MARGIN, w)
}

/// p + iq = H'_u/H_u for x > 2 (Steed's CF2). The fraction is evaluated
/// backward from a depth past which its tail moves neither p nor q by more than
/// eps/32 for any |u| <= 1/2: backward evaluation damps rounding, whereas
/// Lentz's running product accumulates one rounding per level. The final level
/// carries two terms, since q = 1 + O(1/x^2) would otherwise round to half an
/// ulp of one; p and q are then accurate to about eps/16.
fn fraction(u: f64, x: f64, w: &mut Work) -> Result<(D, D), SciOxideError> {
    // The depth is bounded by 56 at x = 2, so it is charged as one unit.
    w.try_step()?;
    let depth = (14.0 + 84.0 / x).ceil() as usize;
    let (mut tail_r, mut tail_i) = (0.0_f64, 0.0_f64);
    for k in (2..=depth).rev() {
        let half = k as f64 - 0.5;
        let real = 2.0 * x + tail_r;
        let imaginary = 2.0 * k as f64 + tail_i;
        let scale = (half - u) * (half + u) / (real * real + imaginary * imaginary);
        (tail_r, tail_i) = (scale * real, -scale * imaginary);
    }
    // p + iq = -1/(2x) + i + (i/x) (1/4 - u^2)/(2(x + i) + t_2).
    let real = wide::sum(D::from(2.0 * x), D::from(tail_r));
    let imaginary = wide::sum(D::from(2.0), D::from(tail_i));
    let numerator = wide::product(
        wide::difference(D::from(0.5), D::from(u)),
        wide::sum(D::from(0.5), D::from(u)),
    );
    let norm = wide::sum(
        wide::product(real, real),
        wide::product(imaginary, imaginary),
    );
    let scale = wide::quotient(numerator, wide::scale(norm, x));
    Ok((
        wide::difference(
            wide::product(scale, imaginary),
            wide::quotient(D::from(0.5), D::from(x)),
        ),
        wide::sum(wide::product(scale, real), D::from(1.0)),
    ))
}

/// J_nu or Y_nu for x > 2 by Steed's method, preserving recessive J.
fn steed(order: f64, x: f64, second: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    w.try_step()?;
    let index = order.round().max(0.0);
    if index > w.remaining as f64 {
        w.complete = false;
        w.check()?;
    }
    let n = index.try_to_usize_exact()?;
    let fractions = fraction(order - index, x, w)?;
    let (j, value) = normalized(
        order,
        x,
        n,
        second,
        ratio(order, x, false, w)?,
        fractions,
        w,
    )?;
    // CF2 rounding reaches Y_nu amplified by the modulus over |Y_nu|; near
    // zeros of Y_nu, CF1 restarts deeper and CF2 runs in two-term arithmetic.
    if !second || j.hypot(value) <= CF2_AMPLIFICATION * value.abs() {
        return Ok(if second { value } else { j });
    }
    let fractions = wide::jy_fraction(D::from(order - index), D::from(x), w)?;
    Ok(normalized(order, x, n, true, ratio(order, x, true, w)?, fractions, w)?.1)
}

/// J_nu and, when `second`, Y_nu from CF1's J_(nu+1)/J_nu and CF2's p + iq at
/// u = nu - n: backward recurrence to u, homogeneous normalization (avoiding
/// division by J_u near its zeros) and two-term forward recurrence for Y.
fn normalized(
    order: f64,
    x: f64,
    n: usize,
    second: bool,
    (ratio, sign): (D, f64),
    (p, q): (D, D),
    w: &mut Work,
) -> Result<(f64, f64), SciOxideError> {
    let xd = D::from(x);
    let u = order - n as f64;
    let recurrence_step = wide::quotient(D::from(2.0), xd);
    let mut current = D::from(sign);
    let mut upper = wide::scale(ratio, sign);
    let mut target = current;
    for k in (1..=n).rev() {
        w.try_step()?;
        let coefficient = 2.0 * (order - (n - k) as f64) / x;
        if current.hi.abs() > 1e150 / coefficient.abs().max(1.0) {
            current = wide::scale(current, 1e-150);
            upper = wide::scale(upper, 1e-150);
            target = wide::scale(target, 1e-150);
        }
        let lower = if order >= 32.0 && !second {
            // Only amplitude is recovered from this backward pass at large
            // order; CF1 fixes J_nu's phase.
            D::from(scioxide_core::fast_mul_add(
                coefficient,
                current.hi,
                -upper.hi,
            ))
        } else {
            wide::difference(
                wide::product(
                    wide::scale(recurrence_step, order - (n - k) as f64),
                    current,
                ),
                upper,
            )
        };
        upper = current;
        current = lower;
    }
    let shift = wide::quotient(D::from(u), xd);
    let delta = wide::sum(wide::product(wide::difference(p, shift), current), upper);
    let ratio = wide::quotient(delta, q);
    let normalization = ((2.0 / core::f64::consts::PI) / x / q.value()).sqrt()
        / current.value().hypot(ratio.value());
    let j = target.value() * normalization;
    if !second {
        return Ok((j, 0.0));
    }
    let y = wide::scale(ratio, normalization);
    let y1 = wide::difference(
        wide::product(wide::difference(shift, p), y),
        wide::scale(wide::product(q, current), normalization),
    );
    Ok((j, wide::forward_y(u, x, n, (y, 0.0), (y1, 0.0), w)?.0))
}

/// exp(x) K_u and exp(x) K_(u+1), |u| <= 1/2, x > 2 (Steed's CF2 for K).
fn k_fraction(u: f64, x: f64, w: &mut Work) -> Result<(f64, f64), SciOxideError> {
    w.try_step()?;
    let mut a = u * u - 0.25;
    let mut b = 2.0 * (x + 1.0);
    let mut d = 1.0 / b;
    let mut f = d;
    let mut delta = d;
    let mut previous = 0.0;
    let mut current = 1.0;
    let mut coefficient = -a;
    let mut qsum = coefficient;
    let mut sum = 1.0 + qsum * delta;
    for k in 2..usize::MAX {
        w.try_step()?;
        a -= 2.0 * (k - 1) as f64;
        b += 2.0;
        d = (b + a * d).recip();
        delta *= b * d - 1.0;
        f += delta;
        let q = (previous - (b - 2.0) * current) / a;
        previous = current;
        current = q;
        coefficient *= -a / k as f64;
        qsum += coefficient * q;
        let term = qsum * delta;
        sum += term;
        if q != 0.0 && q.abs() < EPS {
            coefficient *= q;
            previous /= q;
            current = 1.0;
        }
        if term.abs() < EPS * sum.abs() {
            break;
        }
    }
    let k = (core::f64::consts::PI / 2.0).sqrt() / x.sqrt() / sum;
    Ok((k, k * (1.0 + (0.5 + u + (u * u - 0.25) * f) / x)))
}

/// (J_nu, Y_nu) from the Hankel expansion (DLMF 10.17.3-4), or None unless
/// P and Q both reach relative `f64` accuracy before the terms diverge.
fn hankel(order: f64, x: f64, w: &mut Work) -> Result<Option<(f64, f64)>, SciOxideError> {
    let Some((p, q)) = hankel_coefficients(order, x, w)? else {
        return if x < COMPENSATED_HANKEL_ARGUMENT {
            Ok(None)
        } else {
            compensated_hankel(order, x, w)
        };
    };
    let (sine, cosine) = hankel_phase(order, x)?;
    let j = p * cosine - q * sine;
    let y = p * sine + q * cosine;
    // Rounded P/Q coefficients lose relative digits when the final phase
    // nearly cancels. Recompute that region before publishing a 64-ULP bound.
    if x >= COMPENSATED_HANKEL_ARGUMENT
        && j.abs().min(y.abs()) < (p.abs() + q.abs()) / 32.0
        && let Some(pair) = compensated_hankel(order, x, w)?
    {
        return Ok(Some(pair));
    }
    let amplitude = (core::f64::consts::FRAC_2_PI / x).sqrt();
    Ok(Some((amplitude * j, amplitude * y)))
}

/// The leading term can exceed one even though later terms converge rapidly.
/// At nu^2 <= 16x its exponential majorant is about exp(8); two-word sums
/// keep ample guard digits through that initial growth and cancellation.
fn compensated_hankel(
    order: f64,
    x: f64,
    w: &mut Work,
) -> Result<Option<(f64, f64)>, SciOxideError> {
    if x < COMPENSATED_HANKEL_ARGUMENT || order > 4.0 * x.sqrt() {
        return Ok(None);
    }
    let twice = D::from(order).mul_f64(2.0);
    let (mut p, mut q, mut term) = (D::from(1.0), D::ZERO, D::from(1.0));
    let (mut previous, mut magnitude) = (1.0, 1.0);
    let mut decreasing = false;
    for k in 1usize.. {
        w.try_step()?;
        let odd = D::from((2 * k - 1) as f64);
        term = term * (twice - odd) * (twice + odd) / D::from(x).mul_f64(8.0 * k as f64);
        let size = term.hi.abs();
        if !size.is_finite() || decreasing && size > previous {
            return Ok(None);
        }
        decreasing |= size < previous;
        let signed = if (k / 2).is_multiple_of(2) {
            term
        } else {
            -term
        };
        if k.is_multiple_of(2) {
            p = p + signed;
        } else {
            q = q + signed;
        }
        magnitude += size;
        if size <= EPS * EPS * (p.hi.abs() + q.hi.abs()) {
            // Reject an ill-conditioned sum rather than publishing its rounding.
            if 32.0 * EPS * magnitude > p.hi.abs() + q.hi.abs() {
                return Ok(None);
            }
            // The order phase is periodic in four; reduce before multiplying
            // by pi so large orders never lose their quarter-turn offset.
            let angle = D::from(x) - (D::from(order % 4.0).mul_f64(0.5) + D::from(0.25)) * PI_WIDE;
            let (sine, cosine) =
                crate::support::wide::sin_cos_with_step(angle, &mut || w.try_step())?;
            let amplitude = crate::support::wide::square_root(D::from(2.0) / PI_WIDE / D::from(x));
            return Ok(Some((
                (amplitude * (p * cosine - q * sine)).value(),
                (amplitude * (p * sine + q * cosine)).value(),
            )));
        }
        previous = size;
    }
    Err(SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "Hankel expansion exhausted representable indices",
    ))
}

/// Hankel pair at the exact argument 2 sqrt(s), without rounding its phase.
pub(super) fn sqrt_hankel(
    order: D,
    s: f64,
    w: &mut Work,
) -> Result<Option<[(f64, f64); 2]>, SciOxideError> {
    let x = 2.0 * s.sqrt();
    let Some((p, q)) = hankel_coefficients(order.hi, x, w)? else {
        return Ok(None);
    };
    // Reserve the fixed restoring-root and reciprocal-reduction work before
    // evaluating it; the allowance is independent of the argument exponent.
    w.steps(8192)?;
    let phase =
        super::phase::sqrt_oscillatory_phase(s) - (order.mul_f64(0.5) + D::from(0.25)) * PI_WIDE;
    let (sine, cosine) = phase.hi.sin_cos();
    let (sine, cosine) = (sine + phase.lo * cosine, cosine - phase.lo * sine);
    let amplitude = (core::f64::consts::FRAC_2_PI / x).sqrt();
    let phase_error = 64.0 * EPS * EPS * (1.0 + order.hi.abs()) * amplitude * (p.abs() + q.abs());
    Ok(Some([
        (
            amplitude * (p * cosine - q * sine),
            64.0 * EPS * amplitude * ((p * cosine).abs() + (q * sine).abs()) + phase_error,
        ),
        (
            amplitude * (p * sine + q * cosine),
            64.0 * EPS * amplitude * ((p * sine).abs() + (q * cosine).abs()) + phase_error,
        ),
    ]))
}

fn hankel_coefficients(
    order: f64,
    x: f64,
    w: &mut Work,
) -> Result<Option<(f64, f64)>, SciOxideError> {
    let mu = 4.0 * order * order;
    if x < HANKEL_ARGUMENT || mu > 16.0 * x || x < HANKEL_GENERAL && (mu - 1.0).abs() > 0.25 * x {
        return Ok(None);
    }
    w.try_step()?;
    let twice = 2.0 * order;
    let (mut p, mut q, mut term) = (1.0_f64, 0.0_f64, 1.0_f64);
    let mut magnitude = 1.0;
    let mut leading = 0.0_f64;
    let mut previous = f64::INFINITY;
    for k in 1..usize::MAX {
        w.try_step()?;
        let odd = (2 * k - 1) as f64;
        term *= (twice - odd) * (twice + odd) / (8.0 * k as f64 * x);
        if term.abs() > previous {
            return Ok(None);
        }
        if k == 1 {
            leading = term.abs().min(1.0);
        }
        let signed = if (k / 2) % 2 == 0 { term } else { -term };
        if k % 2 == 0 {
            p += signed;
        } else {
            q += signed;
        }
        magnitude += term.abs();
        // Q, the small factor at a zero, must itself be relatively accurate.
        if term == 0.0 || term.abs() <= 0.125 * EPS * leading {
            break;
        }
        previous = term.abs();
    }
    if magnitude > 4.0 * (p.abs() + q.abs()) {
        return Ok(None);
    }
    Ok(Some((p, q)))
}

/// sin and cos of x - (nu/2 + 1/4) pi.
pub(super) fn hankel_phase(order: f64, x: f64) -> Result<(f64, f64), SciOxideError> {
    if x < REDUCED_PHASE {
        let phase = wide::difference(
            D::from(x),
            wide::product(wide::sum(D::from(0.5 * order), D::from(0.25)), PI_WIDE),
        );
        let quadrant = (phase.hi / core::f64::consts::FRAC_PI_2).round();
        let reduced = wide::sum(
            wide::difference(phase, wide::scale(HALF_PI_WIDE, quadrant)),
            D::ZERO,
        );
        let (sine, cosine) = reduced.hi.sin_cos();
        let (sine, cosine) = (sine + reduced.lo * cosine, cosine - reduced.lo * sine);
        return Ok(match (quadrant as i64).rem_euclid(4) {
            0 => (sine, cosine),
            1 => (cosine, -sine),
            2 => (-sine, -cosine),
            _ => (-cosine, sine),
        });
    }
    let (sine_x, cosine_x) = x.sin_cos();
    let root = core::f64::consts::SQRT_2;
    let quarter_cosine = (cosine_x + sine_x) / root;
    let quarter_sine = (sine_x - cosine_x) / root;
    let (order_sine, order_cosine) = scioxide_core::sin_cos_pi(0.5 * order)?;
    Ok((
        quarter_sine * order_cosine - quarter_cosine * order_sine,
        quarter_cosine * order_cosine + quarter_sine * order_sine,
    ))
}

/// exp(-x) I_nu(x) (`second == false`) or exp(x) K_nu(x) from DLMF 10.40.1-2,
/// or None unless the terms reach `f64` accuracy before diverging.
fn modified_hankel(
    order: f64,
    x: f64,
    second: bool,
    w: &mut Work,
) -> Result<Option<D>, SciOxideError> {
    w.try_step()?;
    let twice = 2.0 * order;
    // The corrections to the leading one are summed apart, so the sum keeps
    // their rounding at the scale of the corrections.
    let (mut rest, mut term, mut magnitude) = (0.0_f64, 1.0_f64, 1.0_f64);
    let mut previous = f64::INFINITY;
    for k in 1..usize::MAX {
        w.try_step()?;
        let odd = (2 * k - 1) as f64;
        term *= (twice - odd) * (twice + odd) / (8.0 * k as f64 * x);
        if !second {
            term = -term;
        }
        if term.abs() > previous {
            return Ok(None);
        }
        rest += term;
        magnitude += term.abs();
        if term == 0.0 || term.abs() <= 0.125 * EPS * (1.0 + rest).abs() {
            break;
        }
        previous = term.abs();
    }
    let sum = wide::sum(D::from(1.0), D::from(rest));
    if magnitude > 2.0 * sum.hi.abs() {
        return Ok(None);
    }
    // (pi/(2x))^(1/2) for K and (2 pi x)^(-1/2) for I from two-term radicands.
    Ok(Some(if second {
        wide::product(
            sum,
            wide::square_root(wide::quotient(HALF_PI_WIDE, D::from(x))),
        )
    } else {
        wide::quotient(sum, wide::square_root(wide::scale(TAU_WIDE, x)))
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn work() -> Work<'static, 'static, 'static> {
        Work::new(1 << 20)
    }

    fn ulps(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs() / f64::EPSILON
    }

    #[test]
    fn series_matches_the_two_term_log_space_series() {
        for order in [-0.375, 0.0, 0.25, 1.0, 2.5, 7.0, 13.75, 20.0] {
            for x in [0.01, 0.3, 1.0, 1.9, 2.0] {
                let fast = series(order, x, false, false, &mut work())
                    .unwrap()
                    .unwrap();
                let wide = wide::power(order, x, false, false, &mut work()).unwrap();
                assert!(ulps(fast, wide) <= 2.0, "J {order} {x}: {fast:e} {wide:e}");
            }
            for x in [0.01, 1.0, 5.0, 12.5, 20.0] {
                for scaled in [false, true] {
                    let fast = series(order.abs(), x, true, scaled, &mut work())
                        .unwrap()
                        .unwrap();
                    let wide = wide::power(order.abs(), x, true, scaled, &mut work()).unwrap();
                    assert!(ulps(fast, wide) <= 2.0, "I {order} {x}: {fast:e} {wide:e}");
                }
            }
        }
    }

    #[test]
    fn series_defers_out_of_range_prefixes_to_the_log_space_series() {
        // (x/2)^nu underflows binary64, and Gamma(1+nu) overflows it.
        assert!(
            series(150.0, 1.0e-3, false, false, &mut work())
                .unwrap()
                .is_none()
        );
        assert!(
            series(175.0, 1.0, true, false, &mut work())
                .unwrap()
                .is_none()
        );
        assert!(
            series(0.0, 700.0, true, true, &mut work())
                .unwrap()
                .is_none()
        );
        let value = j(150.0, 1.0e-3, &mut work()).unwrap();
        assert_eq!(
            value,
            wide::power(150.0, 1.0e-3, false, false, &mut work()).unwrap()
        );
    }

    #[test]
    fn temme_seeds_agree_with_the_two_term_series_inside_the_envelope() {
        let mut accepted = 0;
        for u in [-0.5, -0.31, 0.0, 1.0e-40, 0.125, 0.5] {
            for x in [1.0e-3, 0.4, 0.89, 1.3, 2.0] {
                for modified in [false, true] {
                    let seed = temme(u, x, modified, &mut work()).unwrap();
                    let (value, next) = wide::temme(u, x, modified, &mut work()).unwrap();
                    assert!(
                        (seed.next - next).abs() <= seed.next_error + f64::EPSILON * next.abs()
                    );
                    if seed.within(TEMME_ENVELOPE) {
                        accepted += 1;
                        assert!(
                            ulps(seed.value, value) <= TEMME_ENVELOPE,
                            "{u} {x} {modified}"
                        );
                        assert!(
                            ulps(seed.next, next) <= TEMME_ENVELOPE,
                            "{u} {x} {modified}"
                        );
                    }
                }
            }
        }
        assert!(accepted > 30);
    }

    #[test]
    fn mixed_precision_fraction_matches_two_term_fraction() {
        for u in [-0.5, -0.2, 0.0, 0.3, 0.5] {
            for x in [2.0, 2.5, 4.0, 9.0, 31.0, 200.0] {
                let (p, q) = fraction(u, x, &mut work()).unwrap();
                let (pw, qw) = wide::jy_fraction(D::from(u), D::from(x), &mut work()).unwrap();
                let ep = ((p - pw).value() / pw.value()).abs() / f64::EPSILON;
                let eq = ((q - qw).value() / qw.value()).abs() / f64::EPSILON;
                assert!(ep <= 0.0625 && eq <= 0.0625, "{u} {x}");
            }
        }
    }

    #[test]
    fn cf1_restarts_deeper_next_to_a_zero() {
        // One ulp from j_(1,2); J_1 = 2.825339409478929e-17 (mpmath, 160 digits).
        let x = f64::from_bits(0x401c_0ff5_f3b4_7250);
        let shallow = oscillatory_ratio(1.0, x, SHALLOW_MARGIN, &mut work()).unwrap();
        let deep = ratio(1.0, x, false, &mut work()).unwrap();
        assert!(shallow.0.hi.abs() > NEAR_ZERO);
        assert!(
            ulps(
                steed(1.0, x, false, &mut work()).unwrap(),
                2.825_339_409_478_929e-17
            ) <= 1.0
        );
        assert!(((deep.0 - shallow.0).value() / deep.0.value()).abs() > 1.0e-12);
        // Away from zeros both depths agree to two-term accuracy.
        let shallow = oscillatory_ratio(0.5, 13.0, SHALLOW_MARGIN, &mut work()).unwrap();
        let deep = oscillatory_ratio(0.5, 13.0, DEEP_MARGIN, &mut work()).unwrap();
        // The shallow tail is damped to about 2^-16 of a binary64 rounding.
        assert!(((deep.0 - shallow.0).value() / deep.0.value()).abs() < 1.0e-19);
        assert_eq!(deep.1, shallow.1);
    }

    #[test]
    fn hankel_expansions_select_their_valid_region() {
        assert!(hankel(0.0, 19.5, &mut work()).unwrap().is_none());
        assert!(hankel(10.0, 30.0, &mut work()).unwrap().is_none());
        for (order, x) in [
            (0.0, 21.0),
            (0.25, 24.5),
            (3.0, 40.0),
            (1.0, 1.0e7),
            (0.5, 1.0e9),
        ] {
            let (jv, yv) = hankel(order, x, &mut work()).unwrap().unwrap();
            if x < 1.0e6 {
                assert!(ulps(jv, steed(order, x, false, &mut work()).unwrap()) <= 8.0);
                assert!(ulps(yv, steed(order, x, true, &mut work()).unwrap()) <= 8.0);
            } else {
                // J_(1/2) and Y_(1/2) are elementary.
                let amplitude = (core::f64::consts::FRAC_2_PI / x).sqrt();
                if order == 0.5 {
                    assert!((jv - amplitude * x.sin()).abs() <= 4.0 * f64::EPSILON * amplitude);
                    assert!((yv + amplitude * x.cos()).abs() <= 4.0 * f64::EPSILON * amplitude);
                }
            }
        }
        assert!(
            modified_hankel(25.0, 21.0, false, &mut work())
                .unwrap()
                .is_none()
        );
        for x in [21.0, 30.0, 45.0] {
            let expansion = modified_hankel(0.75, x, false, &mut work())
                .unwrap()
                .unwrap();
            let series = series(0.75, x, true, true, &mut work()).unwrap().unwrap();
            assert!(ulps(expansion.value(), series) <= 2.0, "{x}");
        }
    }

    #[test]
    fn integer_orders_continue_into_their_neighbors() {
        for x in [0.5, 1.9, 3.0, 8.0, 15.0, 27.0, 60.0] {
            for n in [0.0_f64, 1.0, 2.0, 5.0] {
                for kernel in [j, y] {
                    let center = kernel(n, x, &mut work()).unwrap();
                    let neighbor = kernel(n.next_up(), x, &mut work()).unwrap();
                    assert!(
                        (center - neighbor).abs() <= 16.0 * f64::EPSILON * center.abs().max(0.1)
                    );
                }
                let center = k(n, x, true, &mut work()).unwrap();
                let neighbor = k(n.next_up(), x, true, &mut work()).unwrap();
                assert!(ulps(neighbor, center) <= 16.0);
                let center = i(n, x, true, &mut work()).unwrap();
                let neighbor = i(n.next_up(), x, true, &mut work()).unwrap();
                assert!(ulps(neighbor, center) <= 16.0);
            }
        }
    }

    #[test]
    fn y_temme_fallback_keeps_the_zero_of_y0() {
        // The first zero of Y_0 is 0.8935769662791675; Y_0(0.9) from mpmath.
        let value = y(0.0, 0.9, &mut work()).unwrap();
        assert!(ulps(value, 5.628_306_635_205_558e-3) <= 4.0, "{value:e}");
    }

    #[test]
    fn two_term_products_survive_the_top_of_the_range() {
        let product = wide::two_product(1.5e300, 1.0e-10);
        assert_eq!(product.hi, 1.5e300 * 1.0e-10);
        assert!(product.lo.is_finite());
        assert!(product.lo.abs() <= 1.0e-16 * product.hi);
    }
}
