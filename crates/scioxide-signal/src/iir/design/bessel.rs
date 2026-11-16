//! Delay-normalized Bessel prototype poles in `f64`.
//!
//! The poles are the reciprocals `1/x` of the zeros of the ordinary Bessel
//! polynomial `y_n(x) = sum_k (n+k)!/(k!(n-k)!) (x/2)^k`. Its coefficient form
//! loses about half the digits by order 25, so the zeros are found by
//! Aberth–Ehrlich iteration on an evaluation that avoids that cancellation:
//! with `w = -1/x` (`Re w > 0` for stable poles) and `v = n + 1/2`, continuing
//! `K_v(1/x)` into the right half-plane (DLMF 10.34.2) gives
//!
//! `y_n(x) = (1 - e^(-2w)) I_v(w)/I_(1/2)(w) + (-1)^n e^(-2w) K_v(w)/K_(1/2)(w)`,
//!
//! whose two terms are each computed to working relative accuracy: the
//! modified-Bessel ratio `K_v/K_(1/2)` by its forward recurrence (dominant) and
//! `I_v/I_(1/2)` as a product of continued-fraction ratios (minimal). At a zero
//! they cancel only against each other, so zeros are accurate to a few units
//! of `epsilon` at every order.
use super::*;
type C = Complex<f64>;

/// Aberth sweeps allowed; Campos–Calderón starts converge in at most about 25
/// sweeps through order 300.
const SWEEPS: usize = 128;

fn real(x: f64) -> C {
    Complex::new(x, 0.0)
}
fn count(value: usize) -> Result<f64, SciOxideError> {
    scalar::<f64>(value)
}

/// `y_n(x)` and `y_n'(x)` by the cancellation-free modified-Bessel form.
fn bessel_y(n: usize, x: C) -> Result<(C, C), SciOxideError> {
    let one = real(1.0);
    // 1/w = -x, so the recurrences need no division by w.
    let inv = -x;
    let w = check(one / inv)?;
    // rho_K(m) = K_(m+1/2)(w)/K_(1/2)(w): rho_(m+1) = rho_(m-1) + (2m+1)/w rho_m.
    let (mut k0, mut dk0) = (one, real(0.0));
    let (mut k1, mut dk1) = (one + inv, -(inv * inv));
    for m in 1..n {
        let t = inv.scale(count(2 * m + 1)?);
        (k0, dk0, k1, dk1) = (
            k1,
            dk1,
            check(k0 + t * k1)?,
            check(dk0 + t * dk1 - t * inv * k1)?,
        );
    }
    // r(m) = I_(m+1/2)(w)/I_(m-1/2)(w) = 1/((2m+1)/w + r(m+1)), started well
    // above the order and the argument, where the minimal solution is negligible.
    // Every pole has |w| < n + 1, so a far larger iterate has diverged.
    let limit = n.saturating_mul(8).saturating_add(64);
    let radius = w
        .norm()
        .ceil()
        .try_to_usize_exact()
        .ok()
        .filter(|&radius| radius <= limit)
        .ok_or_else(|| failure("Bessel iterate left the zero region"))?;
    let start = n
        .checked_mul(2)
        .and_then(|value| value.checked_add(radius.checked_mul(2)?))
        .and_then(|value| value.checked_add(64))
        .ok_or_else(|| failure("Bessel iterate left the zero region"))?;
    let (mut ratio, mut rho_i, mut top) = (real(0.0), one, one);
    for m in (1..=start).rev() {
        ratio = check(one / (inv.scale(count(2 * m + 1)?) + ratio))?;
        if m == n {
            top = ratio;
        }
        if m <= n {
            rho_i = check(rho_i * ratio)?;
        }
    }
    let e = check((w.scale(-2.0)).exp())?;
    let coth = check((one + e) / (one - e))?;
    // d log rho_I / dw = I_(v-1)/I_v - v/w - (coth w - 1/(2w)).
    let drho_i = check(rho_i * (one / top - inv.scale(count(n)? + 0.5) - coth + inv.scale(0.5)))?;
    let sign = real(if n.is_multiple_of(2) { 1.0 } else { -1.0 });
    let value = check((one - e) * rho_i + sign * e * k1)?;
    let dw = check(
        e.scale(2.0) * rho_i + (one - e) * drho_i - sign * e.scale(2.0) * k1 + sign * e * dk1,
    )?;
    Ok((value, check(dw / (x * x))?))
}

/// Campos–Calderón (2011) approximations to the zeros of `y_n`, ordered so that
/// entries `k` and `n - 1 - k` are conjugate.
fn starts(n: usize) -> Result<Vec<C>, SciOxideError> {
    let mut out = zeros(n)?;
    if n == 1 {
        out[0] = real(-1.0);
        return Ok(out);
    }
    let nf = count(n)?;
    let poly = |c: &[f64]| c.iter().rev().fold(0.0, |acc, &v| acc * nf + v);
    let s = poly(&[0., 0., 2., 0., -3., 1.]);
    let b3 = poly(&[16., -8.]) / s;
    let b2 = poly(&[-24., -12., 12.]) / s;
    let b1 = poly(&[8., 24., -12., -2.]) / s;
    let b0 = poly(&[0., -6., 0., 5., -1.]) / s;
    let r = poly(&[0., 0., 2., 1.]);
    let a1 = poly(&[-6., -6.]) / r;
    let a2 = 6.0 / r;
    for (k, slot) in out.iter_mut().enumerate() {
        let k = count(k + 1)?;
        *slot = check(Complex::new(
            a1 * k + a2 * k * k,
            b0 + b1 * k + b2 * k * k + b3 * k * k * k,
        ))?;
    }
    Ok(out)
}

/// Poles of the order-`n` Bessel prototype with unit low-frequency group delay.
///
/// Each Aberth sweep is one scope-local iteration. The result is certified
/// independently of the iteration: every final Newton correction is at most
/// `32 epsilon` relative to its zero, conjugate partners agree to the same
/// tolerance before they are made exact conjugates, and the zeros satisfy the
/// coefficient identity `sum(x) = -1` to `32 epsilon sum(|x|)`.
pub(super) fn delay_poles(
    n: usize,
    ctx: &mut Option<&mut SolveContext<'_, '_>>,
) -> Result<Vec<C>, SciOxideError> {
    let one = real(1.0);
    let tolerance = 32.0 * f64::EPSILON;
    let mut x = starts(n)?;
    let mut correction = zeros::<C>(n)?;
    let mut converged = false;
    for _ in 0..SWEEPS {
        step(ctx)?;
        let mut largest = 0.0_f64;
        for (k, delta) in correction.iter_mut().enumerate() {
            let zero = x[k];
            let (value, slope) = bessel_y(n, zero)?;
            let newton = check(value / slope)?;
            let mut repulsion = real(0.0);
            for (j, &other) in x.iter().enumerate() {
                if j != k {
                    repulsion = check(repulsion + one / (zero - other))?;
                }
            }
            *delta = check(newton / (one - newton * repulsion))?;
            largest = largest.max(delta.norm() / zero.norm());
        }
        for (zero, delta) in x.iter_mut().zip(&correction) {
            *zero -= *delta;
        }
        if largest <= 4.0 * f64::EPSILON {
            converged = true;
            break;
        }
    }
    let mut sum = real(0.0);
    let mut magnitude = 0.0;
    for (k, &zero) in x.iter().enumerate() {
        let (value, slope) = bessel_y(n, zero)?;
        let partner = x[n - 1 - k].conj();
        if !converged
            || (value / slope).norm() > tolerance * zero.norm()
            || (zero - partner).norm() > tolerance * zero.norm()
        {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::IterationLimit,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(
                    ctx.as_ref()
                        .map_or_else(scioxide_core::SolveCounts::new, |c| c.counts()),
                ),
                "Bessel pole iteration did not converge",
            ));
        }
        sum += zero;
        magnitude += zero.norm();
    }
    if (sum + one).norm() > tolerance * magnitude {
        return Err(failure("Bessel poles failed the coefficient identity"));
    }
    let mut poles = zeros::<C>(n)?;
    for (k, pole) in poles.iter_mut().enumerate() {
        // Exact conjugate pairs; the middle zero of an odd order becomes real.
        let symmetric = (x[k] + x[n - 1 - k].conj()).scale(0.5);
        *pole = check(one / symmetric)?;
    }
    Ok(poles)
}
