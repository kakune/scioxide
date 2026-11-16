//! Secular equation roots, rank-one deflation and Löwner weights.
//! Callers supply pole distances in their own coordinates and do all scratch,
//! iteration and cancellation accounting. Inputs are validated by the callers.
use crate::kernel::standard::exponent;
use scioxide_core::{NumericalIssue, RealScalar, SciOxideError};

pub(crate) fn breakdown() -> SciOxideError {
    SciOxideError::numerical_instability(
        NumericalIssue::LossOfSignificance,
        "secular pole separation could not be represented",
    )
}

/// Root of `1 + rho * sum(z² / (d - lambda))`, with sorted distinct poles.
///
/// Two-pole rational interpolation matches the value and the derivatives of
/// the left and right sums (R.-C. Li, "Solving secular equations stably and
/// efficiently", 1994). A pole-relative shift avoids cancellation in the
/// subsequent Löwner reconstruction. The rounding bound stops refinement when
/// evaluating the secular sum cannot resolve another step. A bracket protects
/// the rational step and its Newton fallback. The caller counts every iteration.
pub(crate) fn rank_one_root<R: RealScalar>(
    poles: &[R],
    weights: &[R],
    root: usize,
    rho: R,
    mut admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(R, R), SciOxideError> {
    let n = poles.len();
    if n == 1 {
        admit()?;
        let shift = rho * weights[0] * weights[0];
        return if shift > R::ZERO && shift.is_finite() {
            Ok((poles[0], shift))
        } else {
            Err(breakdown())
        };
    }
    let last = root + 1 == n;
    let split = if last { n - 2 } else { root };
    let extent = if last {
        rho * weights.iter().fold(R::ZERO, |sum, &z| sum + z * z)
    } else {
        poles[root + 1] - poles[root]
    };
    if extent <= R::ZERO || !extent.is_finite() {
        return Err(breakdown());
    }
    // Sum each sign separately, small terms first, and accumulate the absolute
    // partial sums to bound rounding without an n-dependent stopping heuristic.
    let evaluate = |origin: R, shift: R| {
        let sum = |range: core::ops::Range<usize>, reverse: bool| {
            let (mut value, mut derivative, mut error) = (R::ZERO, R::ZERO, R::ZERO);
            for offset in 0..range.len() {
                let i = if reverse {
                    range.end - 1 - offset
                } else {
                    range.start + offset
                };
                let distance = (poles[i] - origin) - shift;
                let ratio = weights[i] / distance;
                value += rho * weights[i] * ratio;
                derivative += rho * ratio * ratio;
                error += value.abs();
            }
            (value, derivative, error)
        };
        let (left, dl, el) = sum(0..split + 1, false);
        let (right, dr, er) = sum(split + 1..n, true);
        let value = (R::ONE + left) + right;
        let error = el
            + er
            + R::TWO.scalbn(2) * (left.abs() + right.abs() + R::ONE)
            + shift.abs() * (dl + dr);
        (value, dl, dr, error)
    };
    rational_root(
        (
            poles[root],
            if last { None } else { Some(poles[root + 1]) },
            extent,
        ),
        [poles[split], poles[split + 1]],
        Coordinates::Eigenvalue,
        evaluate,
        admit,
    )
}

/// Root of the singular-value secular equation, keeping `d − σ` and `d + σ`
/// separately, without squaring the poles or rounding a small shift into its origin.
pub(crate) fn singular_root<R: RealScalar>(
    poles: &[R],
    weights: &[R],
    active: &[usize],
    root: usize,
    admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(R, R), SciOxideError> {
    let n = active.len();
    let left = poles[active[root]];
    let right = active.get(root + 1).map(|&i| poles[i]);
    let norm = active
        .iter()
        .fold(R::ZERO, |norm, &i| norm.hypot(weights[i]));
    let extent = right.map_or_else(|| norm * (norm / (left.hypot(norm) + left)), |r| r - left);
    if n == 1 {
        let mut admit = admit;
        admit()?;
        return if extent > R::ZERO && extent.is_finite() {
            Ok((left, extent))
        } else {
            Err(breakdown())
        };
    }
    let split = if right.is_none() { n - 2 } else { root };
    let evaluate = |origin: R, shift: R| {
        let sum = |range: core::ops::Range<usize>, reverse: bool| {
            let (mut value, mut derivative, mut error) = (R::ZERO, R::ZERO, R::ZERO);
            for offset in 0..range.len() {
                let i = active[if reverse {
                    range.end - 1 - offset
                } else {
                    range.start + offset
                }];
                let minus = (poles[i] - origin) - shift;
                let plus = (poles[i] + origin) + shift;
                let ratio = (weights[i] / minus) / plus;
                value += weights[i] * ratio;
                derivative += ratio * ratio;
                error += value.abs();
            }
            (value, derivative, error)
        };
        let (left, dl, el) = sum(0..split + 1, false);
        let (right, dr, er) = sum(split + 1..n, true);
        let value = (R::ONE + left) + right;
        let error = el
            + er
            + R::TWO.scalbn(2) * (left.abs() + right.abs() + R::ONE)
            + shift.abs() * R::TWO * (origin + shift) * (dl + dr);
        (value, dl, dr, error)
    };
    rational_root(
        (left, right, extent),
        [poles[active[split]], poles[active[split + 1]]],
        Coordinates::SingularValue,
        evaluate,
        admit,
    )
}

#[derive(Clone, Copy)]
enum Coordinates {
    Eigenvalue,
    SingularValue,
}
impl Coordinates {
    fn distance<R: RealScalar>(self, pole: R, origin: R, shift: R) -> R {
        let difference = (pole - origin) - shift;
        match self {
            Self::Eigenvalue => difference,
            Self::SingularValue => difference * ((pole + origin) + shift),
        }
    }
    fn slope<R: RealScalar>(self, value: R) -> R {
        match self {
            Self::Eigenvalue => R::ONE,
            Self::SingularValue => R::TWO * value,
        }
    }
    fn step<R: RealScalar>(self, value: R, correction: R) -> R {
        match self {
            Self::Eigenvalue => correction,
            Self::SingularValue => {
                let updated = if correction >= R::ZERO {
                    value.hypot(correction.sqrt())
                } else {
                    value * (R::ONE + (correction / value) / value).max(R::ZERO).sqrt()
                };
                correction / (value + updated)
            }
        }
    }
}

/// Shared two-pole rational iteration with bracket and Newton safeguards.
fn rational_root<R: RealScalar>(
    (left, right, extent): (R, Option<R>, R),
    neighbors: [R; 2],
    coordinates: Coordinates,
    evaluate: impl Fn(R, R) -> (R, R, R, R),
    mut admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(R, R), SciOxideError> {
    if extent <= R::ZERO || !extent.is_finite() {
        return Err(breakdown());
    }
    let last = right.is_none();
    let halfway = extent * R::HALF;
    let use_right = right.is_some() && evaluate(left, halfway).0 < R::ZERO;
    let origin = if use_right {
        right.unwrap_or(left)
    } else {
        left
    };
    let (mut lower, mut upper) = if use_right {
        (-halfway, R::ZERO)
    } else {
        (R::ZERO, if right.is_none() { extent } else { halfway })
    };
    let mut shift = lower * R::HALF + upper * R::HALF;
    loop {
        admit()?;
        let (value, dl, dr, error) = evaluate(origin, shift);
        let derivative = dl + dr;
        let slope = coordinates.slope(origin + shift) * derivative;
        if value.is_nan() || derivative.is_nan() {
            return Err(breakdown());
        }
        if value.is_finite() && error.is_finite() && value.abs() <= R::EPSILON * error {
            return Ok((origin, shift));
        }
        if value < R::ZERO {
            lower = shift;
        } else {
            upper = shift;
        }
        if upper - lower <= R::TWO * R::EPSILON * shift.abs() {
            return Ok((origin, shift));
        }
        let left = coordinates.distance(neighbors[0], origin, shift);
        let right = coordinates.distance(neighbors[1], origin, shift);
        // Scale the quadratic before its discriminant: roots close to a pole
        // can have large derivatives even though the eigensystem is bounded.
        let a = (left + right) * value - left * right * derivative;
        let b = left * right * value;
        let c = value - left * dl - right * dr;
        let scale = a.abs().max(b.abs()).max(c.abs());
        let (a, b, c) = (a / scale, b / scale, c / scale);
        let discriminant = (a * a - R::TWO.scalbn(1) * b * c).abs().sqrt();
        let mut step = if c == R::ZERO {
            b / a
        } else if last {
            if a >= R::ZERO {
                (a + discriminant) / (R::TWO * c)
            } else {
                R::TWO * b / (a - discriminant)
            }
        } else if a <= R::ZERO {
            (a - discriminant) / (R::TWO * c)
        } else {
            R::TWO * b / (a + discriminant)
        };
        step = coordinates.step(origin + shift, step);
        if !step.is_finite() || (step >= R::ZERO) == (value >= R::ZERO) {
            step = -value / slope;
        }
        let proposed = shift + step;
        let next = if proposed.is_finite() && proposed > lower && proposed < upper {
            proposed
        } else {
            lower * R::HALF + upper * R::HALF
        };
        if next == shift || next == lower || next == upper {
            return Ok((origin, shift));
        }
        shift = next;
    }
}

/// Gu-Eisenstat weight magnitude `sqrt(prod(numerators) / prod(denominators))`.
///
/// Each product is a plain floating-point product rescaled by a power of two
/// whenever it leaves a quarter of the exponent range, so neither overflows or
/// underflows and no logarithm is formed. A zero or non-finite factor, or an
/// unrepresentable weight, means the pole separation was not representable.
pub(crate) fn weight<R: RealScalar>(
    numerators: impl IntoIterator<Item = R>,
    denominators: impl IntoIterator<Item = R>,
) -> Result<R, SciOxideError> {
    let (numerator, up) = product(numerators)?;
    let (denominator, down) = product(denominators)?;
    let exponent = up - down;
    let value = (numerator / denominator)
        .scalbn(exponent.rem_euclid(2))
        .sqrt()
        .scalbn(exponent.div_euclid(2));
    if value.is_finite() && value != R::ZERO {
        Ok(value)
    } else {
        Err(breakdown())
    }
}

/// Product of factor magnitudes as a mantissa and a binary exponent.
fn product<R: RealScalar>(factors: impl IntoIterator<Item = R>) -> Result<(R, i32), SciOxideError> {
    let bound = R::ONE.scalbn(exponent(R::MAX)? / 4);
    factors
        .into_iter()
        .try_fold((R::ONE, 0), |(mantissa, scale), factor| {
            if factor == R::ZERO || !factor.is_finite() {
                return Err(breakdown());
            }
            let mantissa = mantissa * factor.abs();
            Ok(if mantissa > bound || mantissa * bound < R::ONE {
                let shift = exponent(mantissa)?;
                (mantissa.scalbn(-shift), scale + shift)
            } else {
                (mantissa, scale)
            })
        })
}

/// Deflate tiny weights and neighboring sorted poles, rotating the callers'
/// bases together. The returned active indices stay sorted.
pub(crate) fn deflate<R: RealScalar>(
    poles: &[R],
    weights: &mut [R],
    active: &mut [usize],
    weight_scale: R,
    threshold: R,
    mut rotate: impl FnMut(usize, usize, R, R) -> Result<(), SciOxideError>,
) -> Result<usize, SciOxideError> {
    let mut count = 0;
    for index in 0..poles.len() {
        if weight_scale * weights[index].abs() <= threshold {
            weights[index] = R::ZERO;
            continue;
        }
        if count > 0 {
            let previous = active[count - 1];
            if (poles[index] - poles[previous]).abs() <= threshold {
                let radius = weights[previous].hypot(weights[index]);
                rotate(
                    previous,
                    index,
                    weights[previous] / radius,
                    weights[index] / radius,
                )?;
                weights[previous] = radius;
                weights[index] = R::ZERO;
                continue;
            }
        }
        active[count] = index;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn rational_roots_preserve_interlacing_and_rank_one_eigenvectors() {
        for (poles, weights) in [
            (alloc::vec![2.0], alloc::vec![0.75]),
            (alloc::vec![-1.0, 3.0], alloc::vec![0.6, -0.8]),
            (
                alloc::vec![1.0, 1.0 + 1e-12, 1.0 + 2e-12, 2.0],
                alloc::vec![0.5; 4],
            ),
            (
                alloc::vec![-1.0, 0.0, 1e-100, 1e-20, 1.0],
                alloc::vec![1e-8, 0.5, -0.5, 0.5, 0.5],
            ),
        ] {
            for rho in [1e-12, 0.25, 10.0] {
                let mut roots = Vec::new();
                for i in 0..poles.len() {
                    let mut iterations = 0;
                    let (origin, shift) = rank_one_root(&poles, &weights, i, rho, || {
                        iterations += 1;
                        if iterations > 128 {
                            return Err(breakdown());
                        }
                        Ok(())
                    })
                    .unwrap();
                    assert!(origin + shift >= poles[i]);
                    if let Some(&next) = poles.get(i + 1) {
                        assert!(origin + shift <= next);
                    }
                    roots.push((origin, shift));
                }
                // Reconstruct weights from the shifted roots, independently
                // check the original rank-one matrix and the complete basis.
                let z: Vec<f64> = poles
                    .iter()
                    .enumerate()
                    .map(|(i, &pole)| {
                        weight(
                            roots.iter().map(|&(origin, shift)| (origin - pole) + shift),
                            core::iter::once(rho).chain(
                                poles
                                    .iter()
                                    .enumerate()
                                    .filter(|&(j, _)| j != i)
                                    .map(|(_, &p)| p - pole),
                            ),
                        )
                        .unwrap()
                        .copysign(weights[i])
                    })
                    .collect();
                let vectors: Vec<Vec<f64>> = roots
                    .iter()
                    .map(|&(origin, shift)| {
                        let mut v: Vec<_> = poles
                            .iter()
                            .zip(&z)
                            .map(|(&p, &z)| z / ((p - origin) - shift))
                            .collect();
                        let scale = v.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
                        v.iter_mut().for_each(|x| *x /= scale);
                        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                        v.iter_mut().for_each(|x| *x /= norm);
                        v
                    })
                    .collect();
                let bound = 100.0 * poles.len() as f64 * f64::EPSILON;
                let scale = poles.iter().fold(rho, |s, p| s.max(p.abs()));
                for (i, v) in vectors.iter().enumerate() {
                    let zv = weights.iter().zip(v).map(|(z, x)| z * x).sum::<f64>();
                    for j in 0..poles.len() {
                        let residual =
                            ((poles[j] - roots[i].0) - roots[i].1) * v[j] + rho * weights[j] * zv;
                        assert!(
                            residual.abs() <= bound * scale,
                            "{poles:?} rho={rho} i={i} residual={residual}"
                        );
                    }
                    for (j, w) in vectors.iter().enumerate() {
                        let dot = v.iter().zip(w).map(|(x, y)| x * y).sum::<f64>();
                        assert!((dot - f64::from(i == j)).abs() <= bound);
                    }
                }
            }
        }
    }

    #[test]
    fn rational_roots_propagate_iteration_stops_and_unrepresentable_extents() {
        let stopped = rank_one_root(&[0.0, 1.0], &[0.6, 0.8], 0, 1.0, || Err(breakdown()));
        assert!(stopped.is_err());
        assert!(rank_one_root(&[0.0, 1.0], &[0.6, 0.8], 1, f64::INFINITY, || Ok(())).is_err());
    }
}
