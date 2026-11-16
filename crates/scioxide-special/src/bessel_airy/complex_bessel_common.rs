//! Shared complex Bessel selectors, branch-aware rotations and derivative weights.

use crate::support::C;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "required when elementary methods use the no_std scalar trait"
)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    J,
    Y,
    I,
    K,
    H1,
    H2,
}

/// Rotate a Hankel argument to K's coordinate while keeping the original phase.
/// On the imaginary axis either real zero has phase ±π/2: H1 reaches K's lower
/// negative-real lip and H2 its upper lip. A zero's sign must not select a
/// different continuation at these ordinary points of the Hankel domain.
pub(crate) fn hankel_argument(z: C, first: bool) -> C {
    let real = if z.re == 0.0 { 0.0 } else { z.re };
    if first {
        C::new(z.im, -real)
    } else {
        C::new(-z.im, real)
    }
}

/// Normalized binomial weights, visited outwards from their mode to avoid an underflowed
/// endpoint seed. Callers count the three log-Gamma seed evaluations and each yielded term
/// toward the work budget before evaluating it.
pub(crate) fn derivative_terms(
    kind: Kind,
    degree: u32,
) -> Result<impl Iterator<Item = (f64, f64)>, SciOxideError> {
    let mode = degree / 2;
    let (a, _) = crate::gamma_beta::gamma::gamma_log_parts(f64::from(degree) + 1.0)?;
    let (b, _) = crate::gamma_beta::gamma::gamma_log_parts(f64::from(mode) + 1.0)?;
    let (c, _) = crate::gamma_beta::gamma::gamma_log_parts(f64::from(degree - mode) + 1.0)?;
    let weight = (a - b - c - f64::from(degree) * core::f64::consts::LN_2).exp();
    let mut left = weight;
    let mut right = weight;
    let tails = (1..=degree - mode).flat_map(move |distance| {
        let l = if distance <= mode {
            let k = mode - distance;
            left *= f64::from(k + 1) / f64::from(degree - k);
            Some((k, left))
        } else {
            None
        };
        let k = mode + distance;
        right *= f64::from(degree - k + 1) / f64::from(k);
        [l, Some((k, right))].into_iter().flatten()
    });
    Ok(core::iter::once((mode, weight))
        .chain(tails)
        .map(move |(k, weight)| {
            let negative = match kind {
                Kind::I => false,
                Kind::K => !degree.is_multiple_of(2),
                _ => !k.is_multiple_of(2),
            };
            (
                2.0 * (f64::from(k) - f64::from(mode)) - f64::from(degree % 2),
                if negative { -weight } else { weight },
            )
        }))
}
