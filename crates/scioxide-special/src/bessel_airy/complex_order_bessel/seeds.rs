//! Adjacent modified Bessel solutions in their stable principal sectors.
//!
//! Temme's small-argument series and Steed's CF2 seed K at reduced order;
//! forward recurrence grows K, while backward recurrence grows the minimal I
//! solution and the Wronskian supplies its normalization (DLMF 10.28.2).
//! Large orders take Debye's K expansion (DLMF 10.41.4) when it converges.
//! Only adjacent values are kept. All estimates have one exponential scale.

use super::*;
use crate::support::{ONE, gamma_core, reciprocal_gamma::RECIPROCAL_GAMMA};

const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy)]
pub(super) struct Pair {
    pub value: C,
    pub next: C,
    pub error: f64,
    pub next_error: f64,
    pub scale: C,
}

impl Pair {
    pub(super) fn estimate(self, nu: C, z: Jet, shift: C, modified_i: bool) -> Estimate {
        let factor = nu / z.v;
        let next = if modified_i { self.next } else { -self.next };
        let derivative = factor * self.value + next;
        Estimate::from(crate::support::JetData {
            y: Jet {
                v: self.value,
                d: derivative * z.d,
            },
            scale: Jet::constant(self.scale + shift),
            error: self.error,
            derivative_error: Some(
                z.d.norm()
                    * (factor.norm() * self.error
                        + self.next_error
                        + 8.0 * EPS * (factor.norm() * self.value.norm() + self.next.norm())),
            ),
            method: algorithm_component!("complex_bessel_adjacent_recurrence"),
        })
    }

    fn rescale(&mut self, maximum: f64) {
        let size = self.value.norm().max(self.next.norm());
        if size > maximum {
            self.value /= size;
            self.next /= size;
            self.error /= size;
            self.next_error /= size;
            self.scale += real(size.ln());
        }
    }
}

fn log_gamma(z: C, work: &mut Work) -> Result<C, SciOxideError> {
    gamma_core::complex_log_gamma(z, &mut || work.try_step()).map(|(value, _)| value)
}

/// Defining I series; the gamma/power prefactor stays logarithmic.
pub(super) fn i_series(nu: C, z: C, work: &mut Work) -> Result<Pair, SciOxideError> {
    let scale = nu * (z.ln() - real(core::f64::consts::LN_2)) - log_gamma(nu + ONE, work)?;
    let square = z * z * 0.25;
    let (mut term, mut next_term) = (ONE, ONE);
    let (mut sum, mut next_sum) = (ONE, ONE);
    let (mut magnitude, mut next_magnitude) = (1.0, 1.0);
    for k in 1usize.. {
        work.try_step()?;
        let n = k as f64;
        term *= square / ((nu + real(n)) * n);
        next_term *= square / ((nu + real(n + 1.0)) * n);
        sum += term;
        next_sum += next_term;
        magnitude += term.norm();
        next_magnitude += next_term.norm();
        crate::support::finite(sum, breakdown)?;
        crate::support::finite(next_sum, breakdown)?;
        if term.norm() <= EPS * sum.norm() && next_term.norm() <= EPS * next_sum.norm() {
            let factor = z / ((nu + ONE) * 2.0);
            let rounding = 16.0 * EPS * (n + 1.0 + scale.norm());
            return Ok(Pair {
                value: sum,
                next: next_sum * factor,
                error: 2.0 * term.norm() + rounding * magnitude,
                next_error: factor.norm() * (2.0 * next_term.norm() + rounding * next_magnitude),
                scale,
            });
        }
    }
    Err(breakdown())
}

/// Temme's two seeds, including the removable singularity at u=0.
fn temme(u: C, z: C, work: &mut Work) -> Result<Pair, SciOxideError> {
    let log = z.ln() - real(core::f64::consts::LN_2);
    let sigma = -u * log;
    let (mut f, mut p, mut q, initial_error) = if u.norm() <= 0.5 {
        // Even and odd parts of 1/Gamma(1+u), evaluated without subtraction.
        let (mut even, mut odd) = (ZERO, ZERO);
        for (index, &(hi, lo)) in RECIPROCAL_GAMMA.iter().enumerate().rev() {
            work.try_step()?;
            if index % 2 == 0 {
                even = even * u * u + real(hi + lo);
            } else {
                odd = odd * u * u + real(hi + lo);
            }
        }
        let plus = even + u * odd;
        let minus = even - u * odd;
        // The omitted sigma^8/9! term is below 3e-30 for |sigma|<1e-3.
        // This also evaluates the exact removable limit without dividing by zero.
        let sinhc = if sigma.norm() < 1e-3 {
            let square = sigma * sigma;
            ONE + square * (real(1.0 / 6.0) + square * (real(1.0 / 120.0) + square / 5040.0))
        } else {
            sigma.sinh() / sigma
        };
        let a = -odd * sigma.cosh();
        let b = even * log * sinhc;
        let product = plus * minus;
        (
            (a - b) / product,
            sigma.exp() / (plus * 2.0),
            (-sigma).exp() / (minus * 2.0),
            32.0 * EPS * (a.norm() + b.norm()) / product.norm(),
        )
    } else {
        let p = (log_gamma(ONE + u, work)? + sigma).exp() * 0.5;
        let q = (log_gamma(ONE - u, work)? - sigma).exp() * 0.5;
        (
            (p - q) / u,
            p,
            q,
            32.0 * EPS * (p.norm() + q.norm()) * (1.0 + sigma.norm()) / u.norm(),
        )
    };
    let p0 = p;
    let (mut sum, mut rest) = (f, ZERO);
    let (mut magnitude, mut next_magnitude) = (f.norm(), p.norm());
    let mut coefficient = ONE;
    let square = z * z * 0.25;
    for k in 1usize.. {
        work.try_step()?;
        let n = real(k as f64);
        f = (n * f + p + q) / ((n - u) * (n + u));
        p /= n - u;
        q /= n + u;
        coefficient *= square / n;
        let term = coefficient * f;
        let next_term = coefficient * (p - n * f);
        sum += term;
        rest += next_term;
        magnitude += term.norm();
        next_magnitude += next_term.norm();
        crate::support::finite(sum, breakdown)?;
        crate::support::finite(rest, breakdown)?;
        if term.norm() <= EPS * magnitude && next_term.norm() <= EPS * next_magnitude {
            let numerator = (p0 + rest) * 2.0;
            let error = 2.0 * term.norm() + 32.0 * EPS * magnitude + 3.0 * initial_error;
            let numerator_error =
                4.0 * next_term.norm() + 64.0 * EPS * next_magnitude + 4.0 * initial_error;
            // A finite K_u can have an unrepresentable K_(u+1), hence an
            // overflowing derivative. Normalize before the division by z so
            // the value and its scaled form survive independently.
            let radius = z.norm();
            if numerator.norm() > 1e100 * radius || sum.norm() > 1e100 {
                let next_log = numerator.norm().ln() - radius.ln();
                let scale = next_log.max(sum.norm().ln()) - 1e100_f64.ln();
                let encode = |v: C, shift: f64| {
                    if v == ZERO {
                        ZERO
                    } else {
                        v / v.norm() * (v.norm().ln() - shift).exp()
                    }
                };
                let value = encode(sum, scale);
                let next = encode(numerator, scale + radius.ln()) * (z / radius).conj();
                let rounding = 16.0 * EPS * (1.0 + scale.abs() + radius.ln().abs());
                return Ok(Pair {
                    value,
                    next,
                    error: (error.ln() - scale).exp() + rounding * value.norm(),
                    next_error: (numerator_error.ln() - scale - radius.ln()).exp()
                        + rounding * next.norm(),
                    scale: real(scale),
                });
            }
            let next = numerator / z;
            crate::support::finite(next, breakdown)?;
            return Ok(Pair {
                value: sum,
                next,
                error,
                next_error: numerator_error / radius,
                scale: ZERO,
            });
        }
    }
    Err(breakdown())
}

/// Steed's CF2 and its normalization sum, continued through |arg(z)|<=3pi/4.
fn fraction(u: C, z: C, work: &mut Work) -> Result<Pair, SciOxideError> {
    work.try_step()?;
    let a0 = u * u - real(0.25);
    let amplitude = real((PI / 2.0).sqrt()) / z.sqrt();
    if a0 == ZERO {
        let next = amplitude * (ONE + (u + real(0.5)) / z);
        return Ok(Pair {
            value: amplitude,
            next,
            error: 8.0 * EPS * amplitude.norm(),
            next_error: 16.0 * EPS * next.norm(),
            scale: -z,
        });
    }
    let mut a = a0;
    let mut b = (z + ONE) * 2.0;
    let mut d = ONE / b;
    let (mut f, mut delta) = (d, d);
    let (mut previous, mut current) = (ZERO, ONE);
    let mut coefficient = -a;
    let mut qsum = coefficient;
    let mut sum = ONE + qsum * delta;
    let mut magnitude = 1.0 + (qsum * delta).norm();
    let mut quiet = 0;
    for k in 2usize.. {
        work.try_step()?;
        a -= real(2.0 * (k - 1) as f64);
        b += real(2.0);
        d = ONE / (b + a * d);
        delta *= b * d - ONE;
        f += delta;
        let q = (previous - (b - real(2.0)) * current) / a;
        previous = current;
        current = q;
        coefficient *= -a / (k as f64);
        qsum += coefficient * q;
        let term = qsum * delta;
        sum += term;
        magnitude += term.norm();
        crate::support::finite(sum, breakdown)?;
        crate::support::finite(f, breakdown)?;
        if q != ZERO && q.norm() < EPS {
            coefficient *= q;
            previous /= q;
            current = ONE;
        }
        if term.norm() <= EPS * sum.norm() && delta.norm() <= EPS * f.norm() {
            quiet += 1;
        } else {
            quiet = 0;
        }
        if quiet >= 3 {
            let value = amplitude / sum;
            let factor = ONE + (real(0.5) + u + a0 * f) / z;
            let relative = 4.0 * term.norm() / sum.norm()
                + 16.0 * EPS * (k as f64 + 1.0) * (1.0 + magnitude / sum.norm());
            let error = relative * value.norm();
            return Ok(Pair {
                value,
                next: value * factor,
                error,
                next_error: factor.norm() * error
                    + value.norm()
                        * (a0 / z).norm()
                        * (4.0 * delta.norm() + 16.0 * EPS * k as f64 * f.norm()),
                scale: -z,
            });
        }
    }
    Err(breakdown())
}

/// Hankel expansion, accepted only before its least term starts growing.
/// K's sector |arg(z)|<3pi/2 includes both principal cut lips.
pub(super) fn asymptotic(nu: C, z: C, work: &mut Work) -> Result<Option<Pair>, SciOxideError> {
    let radius = z.norm();
    let leading = (nu + real(0.5)).norm() * (nu - real(0.5)).norm();
    // At moderate radii CF2 is cheaper than a speculative expansion. Keep its
    // dispatch; beyond 256 allow initial growth bounded by exp(2), then require
    // the computed tail to converge. This cutoff affects cost, not acceptance.
    if radius <= 20.0 + leading && (radius < 256.0 || leading > 4.0 * radius) {
        return Ok(None);
    }
    let mut terms = [ONE; 2];
    let mut sums = terms;
    let mut magnitudes = [1.0; 2];
    let mut previous = [f64::INFINITY; 2];
    for k in 1usize.. {
        work.try_step()?;
        let odd = (2 * k - 1) as f64;
        for i in 0..2 {
            let order = nu + real(i as f64);
            terms[i] *=
                ((order * 2.0 - real(odd)) * (order * 2.0 + real(odd)) / z) / (8.0 * k as f64);
            let size = terms[i].norm();
            if !size.is_finite() || size > previous[i] {
                return Ok(None);
            }
            sums[i] += terms[i];
            magnitudes[i] += size;
            previous[i] = size;
        }
        if (0..2).all(|i| previous[i] <= EPS * sums[i].norm()) {
            let factor = real((PI / 2.0).sqrt()) / z.sqrt();
            return Ok(Some(Pair {
                value: sums[0] * factor,
                next: sums[1] * factor,
                error: factor.norm() * (4.0 * previous[0] + 32.0 * EPS * magnitudes[0]),
                next_error: factor.norm() * (4.0 * previous[1] + 32.0 * EPS * magnitudes[1]),
                scale: -z,
            }));
        }
    }
    Err(breakdown())
}

/// Smallest order for which Debye's expansion is attempted. Below it CF2 plus
/// a forward recurrence of about `|nu|` steps costs less than two expansions
/// of up to sixteen polynomials. This bound affects cost, not acceptance.
const DEBYE_ORDER: f64 = 64.0;

/// One Debye value of `K_nu(z)`, DLMF 10.41.4 with `t = z/nu`:
/// `sqrt(pi/(2 nu q)) exp(-nu eta) sum (-1)^k U_k(1/q)/nu^k`,
/// `q = sqrt(1+t^2)`, `eta = q + ln(t/(1+q))`.
///
/// Returns `(value, correction, relative error)` with
/// `K = value * exp(-z - correction)`. The large `z` stays exact in the
/// caller's scale; `nu eta - z` is formed without cancelling it. Beyond the
/// turning circle `nu q = z s` and `nu eta - z = nu (s-1)/r - nu log1p(r+s-1)`
/// with `r = nu/z`, `s = sqrt(1+r^2)`; inside it `q` is principal. Both
/// branches continue the right-half-plane expansion analytically.
/// None when the computed terms fail to reach rounding level while
/// decreasing (near the turning points `z = +-i nu`).
fn debye_value(nu: C, z: C, work: &mut Work) -> Result<Option<(C, C, f64)>, SciOxideError> {
    work.try_step()?;
    let (correction, order_q, magnitude) = if z.norm() >= nu.norm() {
        let r = nu / z;
        let s = (ONE + r * r).sqrt();
        let excess = r * r / (ONE + s);
        let quotient = nu * excess / r;
        let logarithm = scioxide_core::xlog1py(nu, r + excess)?;
        (
            quotient - logarithm,
            z * s,
            quotient.norm() + logarithm.norm(),
        )
    } else {
        let t = z / nu;
        let q = (ONE + t * t).sqrt();
        let logarithm = nu * (t / (ONE + q)).ln();
        (
            nu * q + logarithm - z,
            nu * q,
            (nu * q).norm() + logarithm.norm() + z.norm(),
        )
    };
    let q = order_q / nu;
    let square = ONE / (q * q);
    let step = -ONE / order_q;
    if !square.is_finite() || !step.is_finite() || !correction.is_finite() {
        return Ok(None);
    }
    let (mut sum, mut total) = (ONE, 1.0);
    let mut power = ONE;
    let mut previous = f64::INFINITY;
    let mut quiet = 0;
    for (k, coefficients) in crate::bessel_airy::bessel_debye::DEBYE
        .iter()
        .enumerate()
        .skip(1)
    {
        // U_k(p) = p^k P_k(p^2); its absolute Horner sum bounds rounding.
        let (mut polynomial, mut bound) = (ZERO, 0.0);
        for &coefficient in coefficients[k..].iter().step_by(2).rev() {
            work.try_step()?;
            polynomial = polynomial * square + real(coefficient);
            bound = bound * square.norm() + coefficient.abs();
        }
        power *= step;
        let term = polynomial * power;
        let size = term.norm();
        // An asymptotic expansion is used only while its terms decrease.
        if !size.is_finite() || (k > 2 && size > previous) {
            return Ok(None);
        }
        sum += term;
        total += bound * power.norm();
        previous = size;
        quiet = if size <= EPS * sum.norm() {
            quiet + 1
        } else {
            0
        };
        if quiet == 2 {
            let prefactor = real((PI / 2.0).sqrt()) / order_q.sqrt();
            let relative = 4.0 * size / sum.norm()
                + 16.0 * EPS * (k as f64 + 1.0) * total / sum.norm()
                + 8.0 * EPS * (magnitude + 1.0);
            return Ok(Some((prefactor * sum, correction, relative)));
        }
    }
    Ok(None)
}

/// Whether `z` lies inside the turning circle `|z| = |nu|` (with one order of
/// margin for the adjacent value) of an order large enough for Debye.
pub(super) fn turning(nu: C, z: C) -> bool {
    let square = nu.norm_sqr();
    square >= DEBYE_ORDER * DEBYE_ORDER && {
        let radius = square.sqrt() + 1.0;
        z.norm_sqr() <= radius * radius
    }
}

/// Debye's large-order expansion for both adjacent K orders. It is used in
/// the closed right half-plane and, beyond the turning circle, through
/// `|arg z| <= 3pi/4`; inside the circle the left half-plane lies across the
/// Stokes curves from `z = +-i nu`. Its cost depends on neither the order nor
/// the argument.
#[inline(never)]
pub(super) fn debye(nu: C, z: C, work: &mut Work) -> Result<Option<Pair>, SciOxideError> {
    if nu.norm_sqr() < DEBYE_ORDER * DEBYE_ORDER
        || !(z.re >= 0.0 || (-z.re <= z.im.abs() && !turning(nu, z)))
    {
        return Ok(None);
    }
    let Some((value, correction, relative)) = debye_value(nu, z, work)? else {
        return Ok(None);
    };
    let Some((next, next_correction, next_relative)) = debye_value(nu + ONE, z, work)? else {
        return Ok(None);
    };
    // The phase stays in the value: adding it to a large |z| would round it.
    // A larger real part moves into the scale, keeping the value near unit
    // size so connections can project it against the other solution (and
    // Miller's reciprocal normalization) without underflow.
    let folded = if correction.re.abs() > 64.0 {
        correction.re
    } else {
        0.0
    };
    let factor = (C::new(folded, 0.0) - correction).exp();
    let value = value * factor;
    let next = next * factor * (correction - next_correction).exp();
    crate::support::finite(value, breakdown)?;
    crate::support::finite(next, breakdown)?;
    let scale = -z - real(folded);
    let rounding = if folded == 0.0 {
        0.0
    } else {
        4.0 * EPS * (z.re.abs() + folded.abs())
    };
    Ok(Some(Pair {
        value,
        next,
        error: (relative + rounding) * value.norm(),
        next_error: (next_relative + rounding) * next.norm(),
        scale,
    }))
}

/// K is even in its order; callers supply Re(nu)>=0.
pub(super) fn k_pair(nu: C, z: C, work: &mut Work) -> Result<Pair, SciOxideError> {
    if let Some(pair) = asymptotic(nu, z, work)? {
        return Ok(pair);
    }
    if nu.norm_sqr() >= DEBYE_ORDER * DEBYE_ORDER
        && let Some(pair) = debye(nu, z, work)?
    {
        return Ok(pair);
    }
    let shifts = nu.re.round();
    let u = nu - real(shifts);
    let mut pair = if z.norm() <= 2.0 {
        temme(u, z, work)?
    } else {
        fraction(u, z, work)?
    };
    let mut k = 0.0;
    while k < shifts {
        work.try_step()?;
        let factor = (u + real(k + 1.0)) * 2.0 / z;
        pair.rescale(1e100 / factor.norm().max(1.0));
        let next = pair.value + factor * pair.next;
        let error = pair.error
            + factor.norm() * pair.next_error
            + 8.0 * EPS * (pair.value.norm() + factor.norm() * pair.next.norm());
        pair.value = pair.next;
        pair.next = next;
        pair.error = pair.next_error;
        pair.next_error = error;
        crate::support::finite(pair.next, breakdown)?;
        k += 1.0;
    }
    Ok(pair)
}

/// Finite Miller approximants, normalized by the adjacent K Wronskian.
/// Modified Lentz evaluation of DLMF 10.33.1 extends the same backward
/// recurrence without repeating its prefix. Homogeneous normalization avoids
/// multiplying an unbounded ratio by a vanishing solution near an I zero.
/// None requests the independent K connection when the Wronskian cancels.
pub(super) fn i_pair(nu: C, z: C, k: Pair, work: &mut Work) -> Result<Option<Pair>, SciOxideError> {
    // Sixteen orders are an initial guard, not an accuracy or iteration limit.
    let mut checkpoint = 16usize;
    // Squaring the reciprocal of this floor remains inside binary64 range.
    let mut ratio = real(1e-150);
    let mut c = ratio;
    let mut c_square = c.norm_sqr();
    let mut d = ZERO;
    let mut quiet = 0;
    let mut previous: Option<(C, C)> = None;
    for depth in 1usize.. {
        work.try_step()?;
        let coefficient = (nu + real(depth as f64)) * 2.0 / z;
        d += coefficient;
        let mut d_square = d.norm_sqr();
        if d_square < 1e-300 {
            d = real(1e-150);
            d_square = d.norm_sqr();
        }
        // The floor makes these squared denominators normal. Reuse them for
        // Cartesian reciprocals; only an overflowing square needs core's
        // full-range complex division. CF2 uses the same normalization.
        let inverse = |value: C, square: f64| {
            if square.is_finite() {
                value.conj().unscale(square)
            } else {
                ONE / value
            }
        };
        c = coefficient + inverse(c, c_square);
        c_square = c.norm_sqr();
        if c_square < 1e-300 {
            c = real(1e-150);
            c_square = c.norm_sqr();
        }
        d = inverse(d, d_square);
        let change = c * d;
        ratio *= change;
        if (change - ONE).norm_sqr() <= 64.0 * EPS * EPS {
            quiet += 1;
        } else {
            quiet = 0;
        }
        crate::support::finite(ratio, breakdown)?;
        if depth != checkpoint {
            continue;
        }
        checkpoint = checkpoint.saturating_add(16);
        // Three quiet increments reject isolated coincidences. Comparing two
        // normalized checkpoints also checks the finite-tail truncation.
        if quiet < 3 {
            continue;
        }
        let (a, b) = if ratio.norm() > 1.0 {
            (ONE, ONE / ratio)
        } else {
            (ratio, ONE)
        };
        let denominator = b * k.next + a * k.value;
        let condition = (b.norm() * k.next.norm() + a.norm() * k.value.norm()) / denominator.norm();
        // Beyond four bits of cancellation, normalize independently. Both
        // K arguments stay inside CF2's |arg(z)| <= 3pi/4 sector here.
        if condition > 16.0 && z.re <= z.im.abs() {
            return Ok(None);
        }
        let factor = ONE / z / denominator;
        let value = b * factor;
        let next = a * factor;
        crate::support::finite(value, breakdown)?;
        crate::support::finite(next, breakdown)?;
        if let Some((old, old_next)) = previous {
            let difference = (value - old).norm().max((next - old_next).norm());
            if difference
                <= 16.0 * EPS * (depth as f64 + 1.0) * condition * value.norm().max(next.norm())
            {
                let relative = (b.norm() * k.next_error + a.norm() * k.error) / denominator.norm()
                    + 32.0 * EPS * depth as f64 * condition;
                return Ok(Some(Pair {
                    value,
                    next,
                    error: 4.0 * difference + relative * value.norm(),
                    next_error: 4.0 * difference + relative * next.norm(),
                    scale: -k.scale,
                }));
            }
        }
        previous = Some((value, next));
    }
    Err(breakdown())
}
