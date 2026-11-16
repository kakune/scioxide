//! Poisson mixtures of incomplete gamma and beta tails by stable recurrences.
//!
//! Benton and Krishnamoorthy (2003, Comput. Statist. Data Anal. 43:249) sum
//! `sum_j w(k0 + j) G(s0 + j)` with Poisson-type weights
//! `w(k) = e^-mu mu^k / Gamma(k + 1)` from one special-function tail: shape
//! steps of the incomplete gamma or beta tail differ by the leading term
//! `d(s)`, the lower tail by `P(s) = P(s + 1) + d(s)` and the upper tail by
//! `Q(s + 1) = Q(s) + d(s)`. Both recurrences here only add positive terms: the
//! lower tail starts above the Poisson window and walks down, the upper tail
//! starts below it and walks up, so tiny tails keep their relative accuracy.
//! Terms are log-concave in `j`, so each walk stops once the geometric bound
//! on the rest falls below an eighth of the tolerance. Values are scaled to
//! the starting term and rescaled before they could overflow; one work unit is
//! charged per term, besides the starting tail's own work.
use super::math::{Work, beta_log};
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "dependency feature unification can enable inherent f64 math"
)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

/// One family of shape-indexed tails `G(s)`.
#[derive(Clone, Copy)]
pub(super) enum Family {
    /// `P(s, y)`/`Q(s, y)` with `d(s) = y^s e^-y / Gamma(s + 1)`.
    Gamma { y: f64 },
    /// `I_z(s, b)` and its complement, from `ln z` and `ln(1 - z)`, with
    /// `d(s) = z^s (1 - z)^b / (s B(s, b))`.
    Beta { b: f64, lz: f64, ly: f64 },
}

impl Family {
    /// `ln d(s)`.
    fn log_term(self, s: f64) -> Result<f64, SciOxideError> {
        Ok(match self {
            Self::Gamma { y } => s * y.ln() - y - scioxide_special::ln_gamma(s + 1.0)?,
            Self::Beta { b, lz, ly } => {
                s * lz + b * ly + scioxide_special::ln_gamma(s + b)?
                    - scioxide_special::ln_gamma(s + 1.0)?
                    - scioxide_special::ln_gamma(b)?
            }
        })
    }
    /// A bound on `d(s + 1) / d(s)` for every `s >= shape`; the ratio is
    /// monotone in `s` toward `y / (s + 1)` or `z`.
    fn ratio_bound(self, shape: f64) -> f64 {
        match self {
            Self::Gamma { .. } => self.ratio(shape),
            Self::Beta { b, lz, .. } => lz.exp() * ((shape + b) / (shape + 1.0)).max(1.0),
        }
    }
    /// `d(s + 1) / d(s)`.
    fn ratio(self, s: f64) -> f64 {
        match self {
            Self::Gamma { y } => y / (s + 1.0),
            Self::Beta { b, lz, .. } => lz.exp() * (s + b) / (s + 1.0),
        }
    }
    /// `ln G(s)` for the requested tail.
    fn log_tail(self, s: f64, upper: bool, w: &mut Work) -> Result<f64, SciOxideError> {
        match self {
            Self::Gamma { y } => w.special(|budget| {
                let options = scioxide_special::GammaOptions::new(budget);
                if upper {
                    scioxide_special::gamma_log_q_with_options(s, y, options)
                } else {
                    scioxide_special::gamma_log_p_with_options(s, y, options)
                }
            }),
            Self::Beta { b, lz, ly } => {
                if upper {
                    beta_log(b, s, ly, lz, w)
                } else {
                    beta_log(s, b, lz, ly, w)
                }
            }
        }
    }
}

/// Poisson-type weights `w(offset + j)` of a mixture with mean `mu > 0`.
#[derive(Clone, Copy)]
pub(super) struct Weights {
    pub(super) mu: f64,
    /// Index offset `k0`: zero for Poisson laws, one half for the
    /// half-integer weights of the noncentral t.
    pub(super) offset: f64,
}

impl Weights {
    fn log_weight(self, j: usize) -> Result<f64, SciOxideError> {
        let k = self.offset + j as f64;
        if self.offset == 0.0 {
            Ok(crate::distribution::discrete_math::poisson_log_mass(
                k, self.mu,
            ))
        } else {
            // The Poisson mass form tabulates integer factorials only.
            Ok(k * self.mu.ln() - self.mu - scioxide_special::ln_gamma(k + 1.0)?)
        }
    }
    /// `w(k0 + j + 1) / w(k0 + j)`.
    fn up(self, j: usize) -> f64 {
        self.mu / (self.offset + j as f64 + 1.0)
    }
    /// The Poisson index window outside which the weights carry at most
    /// `limit` of the weight at the mode, by the geometric weight ratios.
    fn window(self, limit: f64) -> (usize, usize) {
        let mode = (self.mu - self.offset).max(0.0).floor() as usize;
        let mut top = mode;
        let mut next = self.up(top);
        while next / (1.0 - self.up(top + 1)).max(f64::MIN_POSITIVE) > limit {
            top += 1;
            next *= self.up(top);
        }
        let mut bottom = mode;
        let mut below = if mode == 0 {
            0.0
        } else {
            1.0 / self.up(mode - 1)
        };
        while bottom > 0 && below / (1.0 - 1.0 / self.up(bottom - 1)).max(f64::MIN_POSITIVE) > limit
        {
            bottom -= 1;
            below /= if bottom == 0 {
                f64::INFINITY
            } else {
                self.up(bottom - 1)
            };
        }
        (bottom, top)
    }
}

/// Running positive sum in units of `2^scale` times its starting term.
struct Scaled {
    sum: f64,
    scale: i32,
}

impl Scaled {
    /// Rescale the state before any component could overflow.
    fn rescale(&mut self, values: &mut [f64]) {
        if self.sum > 2.0_f64.powi(400) {
            let factor = 2.0_f64.powi(-400);
            self.sum *= factor;
            values.iter_mut().for_each(|value| *value *= factor);
            self.scale += 400;
        }
    }
    fn log(&self) -> f64 {
        self.sum.ln() + f64::from(self.scale) * core::f64::consts::LN_2
    }
}

/// Whether a log-concave walk has passed its peak and the rest of its terms,
/// bounded geometrically by the current ratio, is negligible.
fn finished(term: f64, previous: f64, sum: f64, tolerance: f64) -> bool {
    if term == 0.0 {
        return true;
    }
    let ratio = term / previous;
    ratio < 1.0 && term * ratio / (1.0 - ratio) <= tolerance * sum
}

/// `ln sum_j w(k0 + j) G(s0 + j)` for the lower (`P`, `I`) or upper tail, or
/// `None` when the recurrence state leaves the binary64 range (callers then
/// sum the mixture term by term).
pub(super) fn log_tail(
    weights: Weights,
    shape: f64,
    family: Family,
    upper: bool,
    w: &mut Work,
) -> Result<Option<f64>, SciOxideError> {
    let tolerance = 0.125 * w.tolerance;
    // A lower tail whose next shape step cannot add more than the tolerance is
    // its first term: G(s+1)/G(s) <= rho/(1-rho) for d-ratios below rho, so
    // every later term is at most mu rho/(1-rho) times the previous one.
    let rho = family.ratio_bound(shape);
    let growth = weights.up(0) * rho / (1.0 - rho);
    let (bottom, top) =
        if !upper && rho < 1.0 && growth < 1.0 && growth / (1.0 - growth) <= tolerance {
            (0, 0)
        } else {
            weights.window(0.0625 * w.tolerance)
        };
    let start = if upper { bottom } else { top };
    w.charge(1)?;
    let log_tail = family.log_tail(shape + start as f64, upper, w)?;
    let log_start = log_tail + weights.log_weight(start)?;
    if log_tail == f64::NEG_INFINITY {
        return Ok(Some(f64::NEG_INFINITY));
    }
    // t = w(k0+j) G(s0+j) and f = w(k0+j) d(s0+j') with j' = j - 1 below the
    // start (lower walk) and j' = j above it (upper walk), both relative to the
    // starting term.
    let mut state = Scaled { sum: 1.0, scale: 0 };
    let mut term = 1.0;
    if upper {
        let mut adjacent = (family.log_term(shape + start as f64)? - log_tail).exp();
        let mut j = start;
        loop {
            w.charge(1)?;
            let factor = weights.up(j);
            let next = factor * (term + adjacent);
            adjacent *= factor * family.ratio(shape + j as f64);
            j += 1;
            state.sum += next;
            let done = finished(next, term, state.sum, tolerance) || !state.sum.is_finite();
            term = next;
            let mut values = [term, adjacent];
            state.rescale(&mut values);
            [term, adjacent] = values;
            if done {
                break;
            }
        }
    } else if start > 0 {
        let mut adjacent = (family.log_term(shape + start as f64 - 1.0)? - log_tail).exp();
        for j in (1..=start).rev() {
            w.charge(1)?;
            let factor = 1.0 / weights.up(j - 1);
            let next = factor * (term + adjacent);
            if j > 1 {
                adjacent *= factor / family.ratio(shape + j as f64 - 2.0);
            }
            state.sum += next;
            let done = finished(next, term, state.sum, tolerance) || !state.sum.is_finite();
            term = next;
            let mut values = [term, adjacent];
            state.rescale(&mut values);
            [term, adjacent] = values;
            if done {
                break;
            }
        }
    }
    let total = log_start + state.log();
    if !(state.sum.is_finite() && total.is_finite()) {
        return Ok(None);
    }
    w.error = w.error.max(w.tolerance);
    Ok(Some(total))
}

/// `ln sum_j w(k0 + j) h(s0 + j)` for the component densities: `h(s) = d(s - 1)`
/// for gamma (the density in `y`) and `h(s) = s d(s)` for beta (`x` times the
/// density in `z`, times `z (1 - z)`), from a start near the peak term.
pub(super) fn log_density(
    weights: Weights,
    shape: f64,
    family: Family,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    let tolerance = 0.125 * w.tolerance;
    let mu = weights.mu;
    let k0 = weights.offset;
    // Unit term ratio mu rho(s) = (k0 + j + 1)(s0 + j) (gamma: rho = y; beta:
    // rho = z (s + b), a linear correction) locates the peak.
    let (linear, constant) = match family {
        Family::Gamma { y } => (k0 + 1.0 + shape, (k0 + 1.0) * shape - mu * y),
        Family::Beta { b, lz, .. } => {
            let z = lz.exp();
            (
                k0 + 1.0 + shape - mu * z,
                (k0 + 1.0) * shape - mu * z * (shape + b),
            )
        }
    };
    let root = 0.5 * (-linear + (linear * linear - 4.0 * constant).max(0.0).sqrt());
    let start = if root.is_finite() && root > 0.0 {
        root.min(4.0e15) as usize
    } else {
        0
    };
    let h = |s: f64| match family {
        Family::Gamma { .. } => family.log_term(s - 1.0),
        Family::Beta { .. } => Ok(s.ln() + family.log_term(s)?),
    };
    // h(s + 1) / h(s): gamma y / s, beta z (s + b) / s.
    let ratio = |s: f64| match family {
        Family::Gamma { y } => y / s,
        Family::Beta { b, lz, .. } => lz.exp() * (s + b) / s,
    };
    w.charge(1)?;
    let log_start = weights.log_weight(start)? + h(shape + start as f64)?;
    if log_start == f64::NEG_INFINITY {
        return Ok(log_start);
    }
    let mut state = Scaled { sum: 1.0, scale: 0 };
    let mut term = 1.0;
    let mut j = start;
    loop {
        w.charge(1)?;
        let next = term * weights.up(j) * ratio(shape + j as f64);
        j += 1;
        state.sum += next;
        let done = finished(next, term, state.sum, tolerance) || !state.sum.is_finite();
        term = next;
        let mut values = [term];
        state.rescale(&mut values);
        [term] = values;
        if done {
            break;
        }
    }
    // The downward walk shares the scale of the upward one.
    let mut term = 2.0_f64.powi(-state.scale);
    let mut j = start;
    while j > 0 {
        w.charge(1)?;
        j -= 1;
        let next = term / (weights.up(j) * ratio(shape + j as f64));
        state.sum += next;
        let done = finished(next, term, state.sum, tolerance) || !state.sum.is_finite();
        term = next;
        let mut values = [term];
        state.rescale(&mut values);
        [term] = values;
        if done {
            break;
        }
    }
    if !state.sum.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "Poisson-mixture density recurrence is not finite",
        ));
    }
    w.error = w.error.max(w.tolerance);
    Ok(log_start + state.log())
}

/// `ln d(s)` of one family member, the central law's density factor.
pub(super) fn central_log_term(family: Family, s: f64) -> Result<f64, SciOxideError> {
    family.log_term(s)
}

/// `ln G(s)` of one central member, charged to `w`.
pub(super) fn central_log_tail(
    family: Family,
    shape: f64,
    upper: bool,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    family.log_tail(shape, upper, w)
}
