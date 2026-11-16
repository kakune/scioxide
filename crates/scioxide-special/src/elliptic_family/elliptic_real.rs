//! `f64` Carlson duplication (DLMF 19.36), before extended-range conversion.
//! Fifth-degree corrections stop at relative deviations below 0.001; the
//! omitted sixth-degree terms are below `f64` rounding at that threshold.
use crate::elliptic_family::elliptic_kernel::Companion;
use crate::elliptic_family::elliptic_kernel::Family;
use crate::support::Work;
#[allow(unused_imports)] // Feature unification can supply the native methods.
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

/// RJ forms cubic intermediates, including its nested RC arguments. Reserving
/// over 120 exponent bits beyond three times the input exponent keeps those products,
/// sums and reciprocals normal. Zeros are accepted by the public domain check.
pub(crate) fn ordinary(a: &[f64]) -> bool {
    const LOW: f64 = f64::from_bits((1023 - 300) << 52);
    const HIGH: f64 = f64::from_bits((1023 + 300) << 52);
    a.iter()
        .all(|x| *x == 0.0 || (LOW..=HIGH).contains(&x.abs()))
}

pub(crate) struct Pair {
    pub rf: f64,
    pub companion: f64,
    pub relative: f64,
}

/// One real duplication state; nested RJ/RC work shares the caller's allowance.
pub(crate) fn duplication(
    mut x: f64,
    mut y: f64,
    mut z: f64,
    kind: Companion<f64>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Pair, SciOxideError> {
    let start = work.used;
    let mut p = match kind {
        Companion::Rj(p) => p,
        _ => z,
    };
    let mut sum = 0.0;
    let mut sum_error = 0.0;
    let mut factor = 1.0;
    loop {
        work.try_step()?;
        let mean = (x + y + z) / 3.0;
        let dx = (mean - x) / mean;
        let dy = (mean - y) / mean;
        let dz = (mean - z) / mean;
        let mut delta = dx.abs().max(dy.abs()).max(dz.abs());
        let cm = match kind {
            Companion::None => mean,
            Companion::Rd => (x + y + 3.0 * z) * 0.2,
            Companion::Rj(_) => (x + y + z + 2.0 * p) * 0.2,
        };
        let (cx, cy, cz, cp) = if matches!(kind, Companion::None) {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let cx = (cm - x) / cm;
            let cy = (cm - y) / cm;
            let cz = (cm - z) / cm;
            let cp = (cm - p) / cm;
            delta = delta
                .max(cx.abs())
                .max(cy.abs())
                .max(cz.abs())
                .max(cp.abs());
            (cx, cy, cz, cp)
        };
        if delta < 0.001 {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            let rf = (1.0 + e2 * (-0.1 + e2 / 24.0 - e3 * (3.0 / 44.0)) + e3 / 14.0) / mean.sqrt();
            let companion = if matches!(kind, Companion::None) {
                0.0
            } else {
                let correction = if matches!(kind, Companion::Rd) {
                    let e2 = cx * cy - 6.0 * cz * cz;
                    let e3 = (3.0 * cx * cy - 8.0 * cz * cz) * cz;
                    let e4 = 3.0 * (cx * cy - cz * cz) * cz * cz;
                    let e5 = cx * cy * cz * cz * cz;
                    1.0 + (e2 * (-3.0 / 14.0 + e2 * (9.0 / 88.0) - e3 * (9.0 / 52.0)) + e3 / 6.0
                        - e4 * (3.0 / 22.0)
                        + e5 * (3.0 / 26.0))
                } else {
                    let ea = cx * (cy + cz) + cy * cz;
                    let eb = cx * cy * cz;
                    let ec = cp * cp;
                    let ed = ea - 3.0 * ec;
                    let ee = eb + 2.0 * cp * (ea - ec);
                    1.0 + (ed * (-3.0 / 14.0 + ed * (9.0 / 88.0) - ee * (9.0 / 52.0))
                        + eb * (1.0 / 6.0 + cp * (-3.0 / 11.0 + cp * (3.0 / 26.0)))
                        + cp * ea * (1.0 / 3.0 - cp * (3.0 / 22.0))
                        - cp * ec / 3.0)
                };
                3.0_f64.mul_add(
                    sum,
                    3.0 * sum_error + factor * correction / (cm * cm.sqrt()),
                )
            };
            if !rf.is_finite() || !companion.is_finite() {
                return Err(crate::support::ELLIPTIC_ERRORS.unstable());
            }
            return Ok(Pair {
                rf,
                companion,
                relative: 8.0 * (work.used - start + 1) as f64 * f64::EPSILON + 4.0 * delta.powi(6),
            });
        }
        let sx = x.sqrt();
        let sy = y.sqrt();
        let sz = z.sqrt();
        let lambda = sx.mul_add(sy, sx.mul_add(sz, sy * sz));
        let term = match kind {
            Companion::None => 0.0,
            Companion::Rd => factor / (sz * (z + lambda)),
            Companion::Rj(_) => {
                let alpha = p.mul_add(sx + sy + sz, sx * sy * sz);
                let pl = p + lambda;
                let beta = p * pl * pl;
                let term =
                    factor * duplication(alpha * alpha, beta, beta, Companion::None, work)?.rf;
                p = pl * 0.25;
                term
            }
        };
        let next_sum = sum + term;
        sum_error += if sum >= term {
            (sum - next_sum) + term
        } else {
            (term - next_sum) + sum
        };
        sum = next_sum;
        factor *= 0.25;
        x = (x + lambda) * 0.25;
        y = (y + lambda) * 0.25;
        z = (z + lambda) * 0.25;
        if matches!(kind, Companion::Rd) {
            p = z;
        }
    }
}

/// Validated real Carlson inputs, including real principal values. The returned
/// absolute error includes conditioning of the RG and principal-RJ sums.
pub(crate) fn evaluate(
    family: Family,
    mut a: [f64; 4],
    work: &mut Work<'_, '_, '_>,
) -> Result<(f64, f64), SciOxideError> {
    let [x, y, z, p] = a;
    let pair = match family {
        Family::Rf => duplication(x, y, z, Companion::None, work)?,
        Family::Rd => duplication(x, y, z, Companion::Rd, work)?,
        Family::Rj if p > 0.0 => duplication(x, y, z, Companion::Rj(p), work)?,
        Family::Rc if y > 0.0 => duplication(x, y, y, Companion::None, work)?,
        Family::Rc => {
            work.try_step()?;
            if x == 0.0 {
                return Ok((0.0, 0.0));
            }
            let t = x - y;
            let pair = duplication(t, -y, -y, Companion::None, work)?;
            let value = pair.rf * (x / t).sqrt();
            return Ok((value, value.abs() * (pair.relative + 8.0 * f64::EPSILON)));
        }
        Family::Rg => {
            a[..3].sort_by(f64::total_cmp);
            let [x, z, y, _] = a;
            if y == 0.0 {
                return Ok((0.0, 0.0));
            }
            if z == 0.0 {
                let value = 0.5 * y.sqrt();
                return Ok((value, f64::EPSILON * value));
            }
            let pair = duplication(x, y, z, Companion::Rd, work)?;
            let first = z * pair.rf;
            let second = -(x - z) * (y - z) * pair.companion / 3.0;
            let third = x.sqrt() * y.sqrt() / z.sqrt();
            let value = 0.5 * (first + second + third);
            return Ok((
                value,
                0.5 * (first.abs() + second.abs() + third.abs())
                    * (pair.relative + 8.0 * f64::EPSILON),
            ));
        }
        Family::Rj => {
            work.try_step()?;
            a[..3].sort_by(f64::total_cmp);
            let [x, y, z, n] = a;
            let q = -n;
            let denominator = z + q;
            let w = (z - x) / denominator;
            let p = w * y + (1.0 - w) * z;
            let pair = duplication(x, y, z, Companion::Rj(p), work)?;
            let pq = p * q;
            let s = x * y + pq;
            let rc = duplication(s, pq, pq, Companion::None, work)?;
            let first = (p - z) * pair.companion;
            let second = -3.0 * pair.rf;
            let third = 3.0 * (x * y * z / s).sqrt() * rc.rf;
            return Ok((
                (first + second + third) / denominator,
                (first.abs() + second.abs() + third.abs()) / denominator
                    * (pair.relative + rc.relative + 16.0 * f64::EPSILON),
            ));
        }
    };
    let value = if matches!(family, Family::Rd | Family::Rj) {
        pair.companion
    } else {
        pair.rf
    };
    Ok((value, value.abs() * pair.relative))
}
