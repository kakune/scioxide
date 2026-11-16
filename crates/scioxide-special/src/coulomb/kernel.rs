//! Scaled Coulomb solutions. DLMF 33.2, 33.6, 33.8, 33.11 and 33.23.
//! Real logarithmic derivatives use independent continuant and Lentz evaluators
//! of DLMF 33.8.1--3, normalized by the Wronskian.
use crate::support::COULOMB_ERRORS as ERRORS;
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::wide::Wide;
use crate::support::{C, ONE, ZERO, real};
use scioxide_core::AlgorithmComponent;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(unused_imports)]
// Required in standalone no_std; a std-linked consumer has inherent f64 methods.
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

#[derive(Clone, Copy)]
pub(super) struct State {
    pub y: C,
    pub d: C,
    pub log: f64,
    pub log_lo: f64,
    pub error: f64,
}
impl State {
    fn normalize(&mut self) -> Result<(), SciOxideError> {
        let magnitude = self.y.norm().max(self.d.norm());
        if !magnitude.is_finite() || !self.error.is_finite() || !self.log.is_finite() {
            return Err(ERRORS.unstable());
        }
        if magnitude > 1e100 || (magnitude > 0.0 && magnitude < 1e-100) {
            self.y = self.y.unscale(magnitude);
            self.d = self.d.unscale(magnitude);
            self.error /= magnitude;
            self.log += magnitude.ln();
        }
        Ok(())
    }
    pub(super) fn estimate(self, derivative: bool, method: AlgorithmComponent) -> Estimate {
        if (self.log != 0.0 || self.log_lo != 0.0) && self.y.im == 0.0 && self.d.im == 0.0 {
            use crate::support::wide::Wide as D;
            let log = D {
                hi: self.log,
                lo: self.log_lo,
            };
            let original = if derivative { self.d.re } else { self.y.re };
            let value = crate::support::wide::project(D::from(original), log);
            if value.is_finite() && (value != 0.0 || original == 0.0) {
                let d = if derivative {
                    0.0
                } else {
                    crate::support::wide::project(D::from(self.d.re), log)
                };
                let available = d.is_finite() && (d != 0.0 || derivative || self.d.re == 0.0);
                let error = crate::support::wide::project(D::from(self.error), log)
                    .abs()
                    .min(f64::MAX);
                return Estimate::from(crate::support::JetData {
                    y: Jet {
                        v: real(value),
                        d: real(if d.is_finite() { d } else { 0.0 }),
                    },
                    scale: Jet::constant(ZERO),
                    error,
                    derivative_error: Some(if available { error } else { f64::INFINITY }),
                    method,
                });
            }
        }
        Estimate::from(crate::support::JetData {
            y: Jet {
                v: if derivative { self.d } else { self.y },
                d: if derivative { ZERO } else { self.d },
            },
            scale: Jet::constant(real(self.log)),
            error: self.error,
            derivative_error: Some(self.error),
            method,
        })
    }
    pub(super) fn conjugate(mut self) -> Self {
        self.y = self.y.conj();
        self.d = self.d.conj();
        self
    }
    /// Stable order recurrence: F downwards, G upwards (DLMF 33.4).
    pub(super) fn advance(
        self,
        l: f64,
        eta: f64,
        z: C,
        up: bool,
        w: &mut Work<'_, '_, '_>,
    ) -> Result<Self, SciOxideError> {
        w.try_step()?;
        let k = if up { l + 1.0 } else { l };
        let s = real(k) / z + real(eta / k);
        let r = (eta / k).hypot(1.0);
        let direction = if up { -1.0 } else { 1.0 };
        let (y, d) = if z.im == 0.0 && z.re > 0.0 && self.y.im == 0.0 && self.d.im == 0.0 {
            use crate::support::wide::Wide as D;
            let k = if up {
                D::from(l) + D::from(1.0)
            } else {
                D::from(l)
            };
            let ratio = D::from(eta) / k;
            let s = k / D::from(z.re) + ratio;
            let square = D::from(1.0) + ratio * ratio;
            let root = square.hi.sqrt();
            let r = D::from(root) + (square - D::from(root) * D::from(root)).div_f64(2.0 * root);
            let y = (s * D::from(self.y.re) + D::from(self.d.re).mul_f64(direction)) / r;
            let d = (s * y - r * D::from(self.y.re)).mul_f64(direction);
            (real(y.value()), real(d.value()))
        } else {
            let y = (s * self.y + real(direction) * self.d).unscale(r);
            (y, real(direction) * (s * y - real(r) * self.y))
        };
        // Stable recurrence transports correlated seed normalization error.
        // Add local cancellation separately; this is a model, not an interval.
        let growth = (y.norm() + d.norm()) / (self.y.norm() + self.d.norm()).max(1e-300);
        let rounding = 16.0
            * f64::EPSILON
            * ((s.norm() * self.y.norm() + self.d.norm()) / r * (1.0 + s.norm())
                + r * self.y.norm());
        let mut next = Self {
            y,
            d,
            log: self.log,
            log_lo: self.log_lo,
            error: self.error * growth + rounding,
        };
        next.normalize()?;
        Ok(next)
    }
}

#[derive(Clone, Copy)]
pub(super) struct Normalization {
    pub log: Wide,
    pub phase: C,
    /// Absolute log-normalization and phase uncertainty, before projection.
    pub error: f64,
}
impl Normalization {
    pub(super) fn estimate(self) -> Estimate {
        State {
            y: ONE,
            d: ZERO,
            log: self.log.hi,
            log_lo: self.log.lo,
            error: self.error + 8.0 * f64::EPSILON,
        }
        .estimate(
            false,
            scioxide_core::algorithm_component!("wide_coulomb_normalization"),
        )
    }
}

pub(super) fn normalization(
    l: f64,
    eta: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Normalization, SciOxideError> {
    super::real::normalization(l, eta, w)
}

/// Frobenius seed, normalized before any exponentially small factor is formed.
fn regular_seed(l: f64, eta: f64, z: C, w: &mut Work<'_, '_, '_>) -> Result<State, SciOxideError> {
    let norm = normalization(l, eta, w)?;
    let mut previous = ZERO;
    let mut term = ONE;
    let mut sum = ONE;
    let mut derivative = real(l + 1.0);
    let mut magnitude = 1.0;
    let mut small = 0;
    for n in 1usize.. {
        w.try_step()?;
        let nf = n as f64;
        let next =
            (real(2.0 * eta) * z * term - z * z * previous).unscale(nf * (nf + 2.0 * l + 1.0));
        previous = term;
        term = next;
        sum += term;
        derivative += term * real(nf + l + 1.0);
        magnitude += term.norm() * (nf + l + 1.0).max(1.0);
        if term.norm() * (nf + l + 1.0) < f64::EPSILON * sum.norm() {
            small += 1;
        } else {
            small = 0;
        }
        if small == 3 {
            break;
        }
        if !sum.is_finite() {
            return Err(ERRORS.unstable());
        }
    }
    let logarithm = z.ln();
    let phase = C::new(0.0, (l + 1.0) * logarithm.im).exp();
    let mut s = State {
        y: phase * sum,
        d: phase * derivative / z,
        log: norm.log.hi + (l + 1.0) * logarithm.re,
        log_lo: norm.log.lo,
        error: (norm.error + 16.0 * f64::EPSILON) * magnitude * (1.0 + 1.0 / z.norm()),
    };
    s.normalize()?;
    Ok(s)
}

/// Taylor continuation of z² y'' + (z² - 2 eta z - l(l+1)) y = 0.
fn propagate(
    l: f64,
    eta: f64,
    mut center: C,
    target: C,
    mut s: State,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    while center != target {
        w.try_step()?;
        let potential = ONE - real(2.0 * eta) / center - real(l * (l + 1.0)) / (center * center);
        let radius = (0.3 * center.norm()).min(8.0 / (1.0 + potential.norm().sqrt()));
        let delta = target - center;
        let last_step = delta.norm() <= radius;
        let h = if last_step {
            delta
        } else {
            delta * real(radius / delta.norm())
        };
        if center + h == center {
            return Err(ERRORS.unstable());
        }
        let h2 = h * h;
        let h3 = h2 * h;
        let h4 = h2 * h2;
        let c2 = center * center;
        let p = c2 - real(2.0 * eta) * center - real(l * (l + 1.0));
        let q = real(2.0) * center - real(2.0 * eta);
        let mut terms = [ZERO, ZERO, s.y, h * s.d];
        let mut y = s.y + h * s.d;
        let mut d = s.d;
        let mut magnitude = y.norm() + d.norm();
        let mut small = 0;
        for n in 0usize.. {
            w.try_step()?;
            let nf = n as f64;
            let term = -(real(2.0 * nf * (nf + 1.0)) * center * h * terms[3]
                + (real(nf * (nf - 1.0)) + p) * h2 * terms[2]
                + q * h3 * terms[1]
                + h4 * terms[0])
                / (c2 * real((nf + 2.0) * (nf + 1.0)));
            let dt = term * real(nf + 2.0) / h;
            y += term;
            d += dt;
            let size = term.norm() + dt.norm();
            magnitude += size;
            if size <= f64::EPSILON * (y.norm() + d.norm()).max(1e-300) {
                small += 1;
            } else {
                small = 0;
            }
            if small >= 3 {
                break;
            }
            if !y.is_finite() || !d.is_finite() {
                return Err(ERRORS.unstable());
            }
            terms = [terms[1], terms[2], terms[3], term];
        }
        let growth = (y.norm() + d.norm()) / (s.y.norm() + s.d.norm()).max(1e-300);
        s.error = s.error * growth + 8.0 * f64::EPSILON * magnitude;
        s.y = y;
        s.d = d;
        s.normalize()?;
        center = if last_step { target } else { center + h };
    }
    Ok(s)
}

/// Combine independently scaled states without materializing exponential factors.
fn combine(a: State, ca: C, b: State, cb: C) -> Result<State, SciOxideError> {
    let scale = (a.log + a.log_lo).max(b.log + b.log_lo);
    let sa = (a.log - scale + a.log_lo).exp();
    let sb = (b.log - scale + b.log_lo).exp();
    let mut result = State {
        y: ca * a.y * real(sa) + cb * b.y * real(sb),
        d: ca * a.d * real(sa) + cb * b.d * real(sb),
        log: scale,
        log_lo: 0.0,
        error: ca.norm() * sa * a.error
            + cb.norm() * sb * b.error
            + 16.0
                * f64::EPSILON
                * (ca.norm() * sa * (a.y.norm() + a.d.norm())
                    + cb.norm() * sb * (b.y.norm() + b.d.norm())),
    };
    result.normalize()?;
    Ok(result)
}

fn rotate(mut s: State, phase: C, log: f64, reverse: bool) -> State {
    s.y *= phase;
    s.d *= if reverse { -phase } else { phase };
    s.log += log;
    s.error += 16.0 * f64::EPSILON * (1.0 + log.abs()) * (s.y.norm() + s.d.norm());
    s
}

/// H^sign and its radial derivative (DLMF 33.11.1). Both signs are used
/// only in the right half-plane, where their asymptotic sectors overlap.
fn asymptotic(
    l: f64,
    eta: f64,
    z: C,
    sign: f64,
    norm: Normalization,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<State>, SciOxideError> {
    let inv = C::new(0.0, -0.5 * sign) / z;
    let Some(expansion) = crate::hypergeometric_family::hypergeometric_kernel::asymptotic(
        Jet::constant(C::new(l + 1.0, sign * eta)),
        Jet::constant(C::new(-l, sign * eta)),
        Jet {
            v: inv,
            d: -inv / z,
        },
        w,
    )?
    else {
        return Ok(None);
    };
    let logz = z.ln();
    let logarithm = core::f64::consts::LN_2 + logz.re;
    let product = -eta * logarithm;
    let residual = (-eta).mul_add(logarithm, -product);
    let carrier = C::new(z.re.cos(), sign * z.re.sin())
        * C::new(product.cos(), sign * product.sin())
        * C::new(residual.cos(), sign * residual.sin())
        * if sign > 0.0 {
            norm.phase
        } else {
            norm.phase.conj()
        }
        * C::new(
            scioxide_core::cos_pi(-l / 2.0)?,
            sign * scioxide_core::sin_pi(-l / 2.0)?,
        );
    let y = expansion.y.v * carrier;
    let d = (expansion.y.d + C::new(0.0, sign) * (ONE - real(eta) / z) * expansion.y.v) * carrier;
    let mut state = State {
        y,
        d,
        log: sign * (-z.im + eta * logz.im),
        log_lo: 0.0,
        error: expansion.error * (1.0 + (ONE - real(eta) / z).norm())
            + (norm.error + 32.0 * f64::EPSILON * (1.0 + product.abs())) * (y.norm() + d.norm()),
    };
    state.normalize()?;
    Ok(Some(state))
}

pub(super) fn regular(
    l: f64,
    eta: f64,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    if z.im.is_sign_negative() {
        return regular(l, eta, z.conj(), w).map(State::conjugate);
    }
    if z.re < 0.0 {
        let f = regular(l, -eta, -z, w)?;
        // DLMF 33.13: continue the regular origin power on the upper lip.
        return Ok(rotate(
            f,
            C::new(-scioxide_core::cos_pi(l)?, -scioxide_core::sin_pi(l)?),
            -core::f64::consts::PI * eta,
            true,
        ));
    }
    if z.im == 0.0 {
        let anchor = eta + eta.hypot(l + 0.5) + 1.0;
        return if z.re >= anchor {
            super::real::waves(l, eta, z.re, w).map(|pair| pair.0)
        } else {
            super::real::regular(l, eta, z.re, w)
        };
    }
    if z.norm() > 8.0 + l.abs() + eta.abs() {
        let norm = normalization(l, eta, w)?;
        if let Some(hp) = asymptotic(l, eta, z, 1.0, norm, w)?
            && let Some(hm) = asymptotic(l, eta, z, -1.0, norm, w)?
        {
            return combine(hp, C::new(0.0, -0.5), hm, C::new(0.0, 0.5));
        }
        // Real Steed supplies a normalized seed at the target's real part.
        // Only the bounded transition region needs Taylor continuation.
        let x = z.re.max(1.0);
        let seed = regular(l, eta, real(x), w)?;
        return propagate(l, eta, real(x), z, seed, w);
    }
    let radius = z.norm().min(0.125 / (1.0 + eta.abs()));
    let start = z * real(radius / z.norm());
    let seed = regular_seed(l, eta, start, w)?;
    propagate(l, eta, start, z, seed, w)
}

/// Complex CF2 for H+'/H+ (DLMF 33.8.2). Evaluated in its recessive
/// half-plane; conjugation supplies H- in the lower half-plane.
fn logarithmic_derivative(
    l: f64,
    eta: f64,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<(C, f64), SciOxideError> {
    let i = C::new(0.0, 1.0);
    let mut a = C::new(-eta * eta - l * (l + 1.0), eta);
    let mut b = real(2.0) * (z - real(eta) + i);
    let mut d = ONE / b;
    let mut term = i * a * d / z;
    let mut sum = i * (ONE - real(eta) / z);
    let mut magnitude = sum.norm();
    let mut small = 0;
    for k in 1usize.. {
        w.try_step()?;
        sum += term;
        magnitude += term.norm();
        if term.norm() <= 0.25 * f64::EPSILON * sum.norm() {
            small += 1;
        } else {
            small = 0;
        }
        if small == 3 {
            return Ok((sum, 32.0 * f64::EPSILON * (k as f64) * magnitude));
        }
        a += C::new(2.0 * k as f64, 2.0 * eta);
        b += real(2.0) * i;
        let t = a * d;
        d = ONE / (b + t);
        term = -term * t * d;
        if !sum.is_finite() || !term.is_finite() {
            return Err(ERRORS.unstable());
        }
    }
    Err(ERRORS.unstable())
}

/// Recover the independently normalized recessive wave from W(F,H)=1.
fn recessive(
    l: f64,
    eta: f64,
    z: C,
    f: State,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    if z.im.is_sign_negative() {
        return recessive(l, eta, z.conj(), f.conjugate(), w).map(State::conjugate);
    }
    if z.norm() < 1.0 {
        // CF2 slows near zero. Continue the dominant G inward from a unit
        // real anchor rather than the old quadratic asymptotic anchor.
        let g = super::real::waves(l, eta, 1.0, w)?.1;
        let g = propagate(l, eta, real(1.0), z, g, w)?;
        return combine(g, ONE, f, C::new(0.0, 1.0));
    }
    let (h, h_error) = logarithmic_derivative(l, eta, z, w)?;
    let denominator = f.d - h * f.y;
    let magnitude = denominator.norm();
    let error = (f.error * (1.0 + h.norm())
        + h_error * f.y.norm()
        + 16.0 * f64::EPSILON * (f.d.norm() + h.norm() * f.y.norm()))
        / magnitude;
    if magnitude == 0.0 || !error.is_finite() {
        return Err(ERRORS.unstable());
    }
    let y = ONE / denominator;
    let mut state = State {
        y,
        d: h * y,
        log: -f.log,
        log_lo: -f.log_lo,
        error: error * (y.norm() + (h * y).norm()) + h_error * y.norm(),
    };
    state.normalize()?;
    Ok(state)
}

/// Irregular wave from an already computed F. The reflection formula uses
/// the recessive H explicitly, so its tiny value is never a difference of F/G.
pub(super) fn irregular(
    l: f64,
    eta: f64,
    z: C,
    f: State,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    if z.im.is_sign_negative() {
        return irregular(l, eta, z.conj(), f.conjugate(), w).map(State::conjugate);
    }
    if z.im == 0.0 && z.re > 0.0 {
        return super::real::waves(l, eta, z.re, w).map(|pair| pair.1);
    }
    let h = if z.re < 0.0 {
        let phase = C::new(-scioxide_core::cos_pi(l)?, scioxide_core::sin_pi(l)?);
        let fw = rotate(f, phase, core::f64::consts::PI * eta, true);
        let hw = recessive(l, -eta, -z, fw, w)?;
        rotate(
            hw,
            C::new(scioxide_core::cos_pi(l)?, -scioxide_core::sin_pi(l)?),
            core::f64::consts::PI * eta,
            true,
        )
    } else {
        recessive(l, eta, z, f, w)?
    };
    combine(h, ONE, f, C::new(0.0, -1.0))
}

pub(super) fn waves(
    l: f64,
    eta: f64,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<(State, State), SciOxideError> {
    if z.im == 0.0 && z.re > 0.0 {
        return super::real::waves(l, eta, z.re, w);
    }
    let f = regular(l, eta, z, w)?;
    let g = irregular(l, eta, z, f, w)?;
    Ok((f, g))
}
