//! Modified Chebyshev moments of algebraic–logarithmic, Cauchy and Fourier weights
//! (Piessens & Branders; Piessens et al. 1983).
use super::number;
use crate::kernel::checked_intermediate;
use scioxide_core::{Interval, RealScalar, SciOxideError};

#[derive(Clone, Copy)]
pub(crate) struct Moments<S: RealScalar> {
    pub(crate) values: [S; 25],
    pub(crate) errors: [S; 25],
    pub(crate) native: bool,
    pub(crate) mass: [S; 2],
}

impl<S: RealScalar> Moments<S> {
    pub(crate) fn new(values: [S; 25], errors: [S; 25]) -> Self {
        Self {
            native: Self::admitted(&values, &errors),
            mass: Self::masses(&values),
            values,
            errors,
        }
    }

    /// `[sum |m_k|, sum_(k >= 1) |m_k|]`, the latter in four native lanes.
    #[inline]
    fn masses(values: &[S; 25]) -> [S; 2] {
        let [higher] = scioxide_core::backend::four_lane_sums(24, |i| [values[i + 1].abs()]);
        [values[0].abs() + higher, higher]
    }

    fn refresh(&mut self) {
        self.native = Self::admitted(&self.values, &self.errors);
        self.mass = Self::masses(&self.values);
    }

    #[inline]
    fn admitted(values: &[S; 25], errors: &[S; 25]) -> bool {
        let range = scioxide_core::backend::ShortReductionRange::new(values.len());
        range.admits_all(values) & range.admits_all(errors)
    }
}

pub(crate) struct EndpointMoments<S: RealScalar> {
    pub(crate) power: [S; 25],
    pub(crate) log: [S; 25],
}

// Unit-width canonical moments (1/2) int_{-1}^{1} w_k(t) s^alpha [ln s] dt with
// s = (1 + t)/2, so a region of width w scales them by w^(alpha + 1) and its
// logarithm splits as ln(w) + ln(s). Index zero is the ordinary mass. Higher
// indices integrate T_k(t)-T_k(-1). Recurring these differences directly removes
// the common 1/(alpha+1) singularity before rounding, crucial when the regular
// amplitude vanishes. Every moment is O(1/(alpha + 1)): no power of two is formed,
// so large exponents neither overflow nor underflow here.
pub(super) fn endpoint<S: RealScalar>(alpha: S) -> Result<EndpointMoments<S>, SciOxideError> {
    let one = S::ONE;
    let two = number::<S>(2)?;
    let a1 = alpha + one;
    let a2 = alpha + two;
    // Unit-width normalization: the mass factor of every moment is one.
    let mass = one;
    let mut power = [S::ZERO; 25];
    let mut log = power;
    power[0] = mass / a1;
    power[1] = two * mass / a2;
    log[0] = -power[0] / a1;
    log[1] = -two * mass / (a2 * a2);
    for k in 2..25 {
        let n = number::<S>(k)?;
        let previous = n - one;
        let denominator = previous * (n + a1);
        let sign = if k % 2 == 0 { one } else { -one };
        power[k] =
            -(mass * (one - sign * (one - two * n)) + n * (n - a2) * power[k - 1]) / denominator;
        log[k] =
            -(n * (n - a2) * log[k - 1] - n * power[k - 1] + previous * power[k]) / denominator;
    }
    for &v in power.iter().chain(&log) {
        checked_intermediate(v)?;
    }
    Ok(EndpointMoments { power, log })
}

// Principal-value moments m_k = PV int_{-1}^{1} T_k(t) / (t - c) dt for a canonical pole
// |c| <= 1, or ordinary moments for an exterior pole (QUADPACK DQC25C). m_0 is the log
// ratio of the distances to the endpoints (the caller passes it from physical distances,
// which is scale-invariant and avoids forming 1 - c and 1 + c, so a pole rounded onto
// c = +-1 keeps its exact logarithm). Writing T_k = 2t T_(k-1) - T_(k-2) and
// t = (t - c) + c gives m_k = 2c m_(k-1) - m_(k-2) + 2 int T_(k-1); that integral vanishes
// when k-1 is odd and equals -2/((k-1)^2 - 1) when k-1 is even, hence the correction for
// odd k. For |c| <= 1 the recurrence's roots lie on the unit circle; an exterior pole's
// moments are its minimal solution, so rounding grows by the dominant root
// g = |c| + sqrt(c^2 - 1) per step and the error model carries (g^k - 1)(1 + |m_0|).
pub(super) fn cauchy<S: RealScalar>(pole: S, mass: S) -> Result<Moments<S>, SciOxideError> {
    // (k - 1)^2 - 1 for the odd orders k = 3, 5, ..., 23; the exact quotients
    // 4 / ((k - 1)^2 - 1) fold to constants.
    const DENOMINATORS: [f64; 11] = [
        3.0, 15.0, 35.0, 63.0, 99.0, 143.0, 195.0, 255.0, 323.0, 399.0, 483.0,
    ];
    let corrections: [S; 11] =
        core::array::from_fn(|j| S::from_f64_ieee(4.0) / S::from_f64_ieee(DENOMINATORS[j]));
    let mut values = [S::ZERO; 25];
    let two = S::TWO;
    let twice = two * pole;
    values[0] = mass;
    values[1] = two + pole * values[0];
    // The recurrence is one dependent chain; keeping its two previous terms
    // in registers keeps stores off that chain.
    let (mut before, mut last) = (values[0], values[1]);
    for k in 2..25 {
        let mut next = twice * last - before;
        if k % 2 == 1 {
            next -= corrections[(k - 3) / 2];
        }
        values[k] = next;
        (before, last) = (last, next);
    }
    let growth = (pole.abs() + (pole * pole - S::ONE).max(S::ZERO).sqrt()).max(S::ONE);
    let hundred = S::from_f64_ieee(100.0) * S::EPSILON;
    let spread = S::ONE + mass.abs();
    let mut errors = [S::ZERO; 25];
    let mut finite = true;
    if growth == S::ONE {
        // Without growth the amplified term vanishes exactly.
        for (error, &v) in errors.iter_mut().zip(&values) {
            finite &= v.is_finite();
            *error = hundred * (S::ONE + v.abs());
        }
    } else {
        let mut amplification = S::ONE;
        for (error, &v) in errors.iter_mut().zip(&values) {
            finite &= v.is_finite();
            *error = hundred * ((S::ONE + v.abs()) + (amplification - S::ONE) * spread);
            amplification *= growth;
        }
    }
    if !finite {
        for &v in &values {
            checked_intermediate(v)?;
        }
    }
    Ok(Moments::new(values, errors))
}

fn bounded<S: RealScalar>(values: [S; 25], multiplier: S) -> Result<Moments<S>, SciOxideError> {
    let mut errors = [S::ZERO; 25];
    for (error, &v) in errors.iter_mut().zip(&values) {
        checked_intermediate(v)?;
        *error = multiplier * S::EPSILON * (S::ONE + v.abs());
    }
    Ok(Moments::new(values, errors))
}

// Canonical Fourier moments int_{-1}^{1} T_k(t) cos(p t) dt (even k) and
// int_{-1}^{1} T_k(t) sin(p t) dt (odd k); the other parity combinations vanish by symmetry.
// Above the order, forward recursion is stable. Below it, the Jacobi-Anger expansion
// evaluates them analytically from Bessel coefficients, without trigonometry or quadrature.
pub(crate) fn fourier<S: RealScalar>(phase: S) -> Result<Moments<S>, SciOxideError> {
    let p = phase.abs();
    if p <= number::<S>(24)? {
        return jacobi_anger(p);
    }
    // Closed forms seed the first even/odd moments; the three-term recurrences (in steps of
    // two, one per parity) follow Piessens' derivation used by QUADPACK DQC25F.
    let two = number::<S>(2)?;
    let four = number::<S>(4)?;
    let p2 = p * p;
    let sin = p.sin();
    let cos = p.cos();
    let mut v = [S::ZERO; 25];
    v[0] = two * sin / p;
    v[2] = (number::<S>(8)? * cos + (two * p2 - number::<S>(8)?) * sin / p) / p2;
    v[4] = (number::<S>(32)? * (p2 - number::<S>(12)?) * cos
        + two * ((p2 - number::<S>(80)?) * p2 + number::<S>(192)?) * sin / p)
        / (p2 * p2);
    let ac = number::<S>(8)? * cos;
    let a_s = number::<S>(24)? * p * sin;
    for k in (6..25).step_by(2) {
        let n = number::<S>(k - 2)?;
        let n2 = n * n;
        v[k] = ((n2 - four) * (two * (p2 + two - two * n2) * v[k - 2] - ac) + a_s
            - p2 * (n + S::ONE) * (n + two) * v[k - 4])
            / (p2 * (n - S::ONE) * (n - two));
    }
    v[1] = two * (sin - p * cos) / p2;
    v[3] = (number::<S>(18)? - number::<S>(48)? / p2) * sin / p2
        + (-two + number::<S>(48)? / p2) * cos / p;
    let ac = -number::<S>(24)? * p * cos;
    let a_s = -number::<S>(8)? * sin;
    for k in (5..25).step_by(2) {
        let n = number::<S>(k - 2)?;
        let n2 = n * n;
        v[k] = ((n2 - four) * (two * (p2 + two - two * n2) * v[k - 2] + a_s) + ac
            - p2 * (n + S::ONE) * (n + two) * v[k - 4])
            / (p2 * (n - S::ONE) * (n - two));
    }
    bounded(v, number::<S>(100)?)
}

/// Bessel orders kept for normalized phases `p <= 24`.
const BESSEL_ORDERS: usize = 80;

/// `J_n(p)` for `n <= m` and `0 <= p <= 24`, by Miller's backward recurrence.
///
/// The start `m = 2 ceil((p + 6 p^(1/3) + 36) / 2)` lies 16 orders beyond the last
/// order whose coefficient exceeds binary64 rounding, so the arbitrary start is
/// forgotten. Growing values are rescaled, which only flushes negligible high
/// orders. The magnitude is normalized by the positive identity
/// `J_0^2 + 2 sum J_n^2 = 1` and the sign by `J_0 + 2 sum J_2k = 1`. Below
/// `sqrt(eps)`, `J_0 = 1` and `J_1 = p / 2` are exact to working precision.
fn bessel<S: RealScalar>(p: S) -> Result<([S; BESSEL_ORDERS], usize), SciOxideError> {
    let two = number::<S>(2)?;
    let mut j = [S::ZERO; BESSEL_ORDERS];
    let m = 2
        * ((p + number::<S>(6)? * p.cbrt() + number::<S>(36)?) / two)
            .ceil()
            .try_to_usize_exact()?;
    if m + 1 >= BESSEL_ORDERS {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("phase"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Bessel recurrence order exceeds its storage",
        ));
    }
    if p < S::EPSILON.sqrt() {
        j[0] = S::ONE;
        j[1] = p / two;
        return Ok((j, m));
    }
    let big = S::MAX.sqrt();
    j[m] = S::ONE;
    for n in (1..=m).rev() {
        let previous = (two * number::<S>(n)? / p) * j[n] - j[n + 1];
        j[n - 1] = previous;
        if previous.abs() > big {
            for value in &mut j[n - 1..=m] {
                *value /= big;
            }
        }
    }
    let largest = j[..=m]
        .iter()
        .fold(S::ZERO, |largest, &value| largest.max(value.abs()));
    let mut linear = j[0];
    let mut squares = (j[0] / largest) * (j[0] / largest);
    for (n, &value) in j[..=m].iter().enumerate().skip(1) {
        let unit = value / largest;
        squares += two * unit * unit;
        if n % 2 == 0 {
            linear += two * value;
        }
    }
    let scale = linear.signum() / (largest * squares.sqrt());
    for value in &mut j[..=m] {
        *value *= scale;
    }
    for &value in &j[..=m] {
        checked_intermediate(value)?;
    }
    Ok((j, m))
}

/// Unit-interval Fourier moments from the Jacobi–Anger expansions
/// `cos(p t) = J_0 + 2 sum (-1)^m J_2m T_2m` and `sin(p t) = 2 sum (-1)^m J_2m+1 T_2m+1`,
/// with `int T_k T_n = 1 / (1 - (k + n)^2) + 1 / (1 - (k - n)^2)` for `k + n` even.
/// Each moment's error is `64 eps (|v_k| + sum |c_n w_kn|)`, which covers the
/// rounding of the recurrence, its normalization and the cancelling sum.
fn jacobi_anger<S: RealScalar>(p: S) -> Result<Moments<S>, SciOxideError> {
    let (j, m) = bessel(p)?;
    let mut integrals = [S::ZERO; BESSEL_ORDERS + 25];
    for (s, integral) in integrals.iter_mut().enumerate().step_by(2) {
        let s = number::<S>(s)?;
        *integral = S::ONE / (S::ONE - s * s);
    }
    let two = number::<S>(2)?;
    let mut values = [S::ZERO; 25];
    let mut errors = [S::ZERO; 25];
    for (k, (value, error)) in values.iter_mut().zip(&mut errors).enumerate() {
        let mut mass = S::ZERO;
        for n in (k % 2..=m).step_by(2) {
            let coefficient = match n {
                0 => j[0],
                _ if (n / 2) % 2 == 0 => two * j[n],
                _ => -two * j[n],
            };
            let term = coefficient * (integrals[k + n] + integrals[k.abs_diff(n)]);
            *value += term;
            mass += term.abs();
        }
        checked_intermediate(*value)?;
        *error = number::<S>(64)? * S::EPSILON * (value.abs() + mass);
    }
    Ok(Moments::new(values, errors))
}

// Exact quadrantal entries avoid spurious cancellation at endpoints.
// table[k][j] = cos(k j pi / 24) = T_k(t_j) at the Chebyshev extrema t_j = cos(j pi / 24);
// reducing k*j modulo 48 (a full period) keeps the argument small and exact.
pub(crate) fn cosines<S: RealScalar>() -> Result<[[S; 25]; 25], SciOxideError> {
    let mut table = [[S::ZERO; 25]; 25];
    for (k, row) in table.iter_mut().enumerate() {
        for (j, c) in row.iter_mut().enumerate() {
            *c = match (k * j) % 48 {
                0 => S::ONE,
                24 => -S::ONE,
                12 | 36 => S::ZERO,
                phase => (S::PI * number::<S>(phase)? / number::<S>(24)?).cos(),
            };
        }
    }
    Ok(table)
}

pub(crate) fn fourier_on_interval<S: RealScalar>(
    omega: S,
    weight: super::FourierWeight,
    interval: Interval<S>,
) -> Result<Moments<S>, SciOxideError> {
    let half = interval.width()? / number::<S>(2)?;
    let stored = fourier(checked_intermediate(omega.abs() * half)?)?;
    orient_fourier(stored, omega, weight, half, interval.midpoint()?)
}

/// Unit-interval moments mapped to `[middle - half, middle + half]`.
pub(crate) fn orient_fourier<S: RealScalar>(
    moments: Moments<S>,
    omega: S,
    weight: super::FourierWeight,
    half: S,
    middle: S,
) -> Result<Moments<S>, SciOxideError> {
    let (factors, floor) = fourier_orientation(omega, weight, half, middle)?;
    orient(moments, factors, floor)
}

/// Even/odd factors and phase floor mapping unit-interval Fourier moments to
/// `[middle - half, middle + half]`.
///
/// With `x = m + h t`, `cos(omega x) = cos(omega m) cos(omega h t) - sin(omega m)
/// sin(omega h t)`, and only the parity-matching unit-interval moment survives for
/// each order; the integral also picks up the Jacobian `h`. A negative omega flips
/// the sign of the odd (sine) part. The center phase `omega*m` carries about
/// `eps |omega m|` absolute error, so the floor `4 eps |omega m| h` multiplies each
/// unit-interval moment's magnitude.
pub(crate) fn fourier_orientation<S: RealScalar>(
    omega: S,
    weight: super::FourierWeight,
    half: S,
    middle: S,
) -> Result<([S; 2], S), SciOxideError> {
    let phase = checked_intermediate(omega * middle)?;
    let (sin, cos) = (phase.sin(), phase.cos());
    let sign = if omega < S::ZERO { -S::ONE } else { S::ONE };
    let factors = match weight {
        super::FourierWeight::Cosine => [half * cos, half * (-sign * sin)],
        super::FourierWeight::Sine => [half * sin, half * (sign * cos)],
    };
    let floor = number::<S>(4)? * S::EPSILON * phase.abs() * half;
    Ok((factors, floor))
}

/// Oriented moments `factor_(k mod 2) m_k` of [`fourier_orientation`], each with
/// error `|factor| e_k + floor |m_k|` plus its own rounding floor.
#[inline]
pub(crate) fn orient<S: RealScalar>(
    canonical: Moments<S>,
    factors: [S; 2],
    floor: S,
) -> Result<Moments<S>, SciOxideError> {
    let fifty = number::<S>(50)? * S::EPSILON;
    let mut moments = canonical;
    let mut regular = true;
    for (k, (value, error)) in moments
        .values
        .iter_mut()
        .zip(&mut moments.errors)
        .enumerate()
    {
        let factor = factors[k % 2];
        let phase_floor = floor * value.abs();
        *value *= factor;
        let rounding = fifty * value.abs();
        *error = *error * factor.abs() + phase_floor + rounding;
        // A rounding floor below the normal range takes the extended floor.
        regular &= value.is_finite()
            & error.is_finite()
            & (rounding >= S::MIN_POSITIVE || *value == S::ZERO);
    }
    if !regular {
        // Replay element by element: report the first nonfinite value and give
        // a subnormal value its extended-range rounding floor.
        for (k, ((value, error), (&stored, &stored_error))) in moments
            .values
            .iter_mut()
            .zip(&mut moments.errors)
            .zip(canonical.values.iter().zip(&canonical.errors))
            .enumerate()
        {
            let factor = factors[k % 2];
            *value = checked_intermediate(stored * factor)?;
            *error = checked_intermediate(
                stored_error * factor.abs()
                    + floor * stored.abs()
                    + crate::kernel::roundoff_floor(*value)?,
            )?;
        }
    }
    moments.refresh();
    Ok(moments)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kernel::AffineMap1D;

    /// Independent quadrature reference: G7K15 on 32 equal panels of [-1, 1],
    /// accumulated exactly.
    fn quadrature<S: RealScalar>(p: S) -> [S; 25] {
        let rule = *crate::GaussKronrod::new(crate::GaussKronrodRule::G7K15)
            .unwrap()
            .rule();
        let mut sums: [scioxide_core::RangeAccumulator<S>; 25] =
            core::array::from_fn(|_| scioxide_core::RangeAccumulator::new());
        let panels = 32;
        let two = number::<S>(2).unwrap();
        for panel in 0..panels {
            let a = -S::ONE + two * number::<S>(panel).unwrap() / number::<S>(panels).unwrap();
            let b = -S::ONE + two * number::<S>(panel + 1).unwrap() / number::<S>(panels).unwrap();
            let map = AffineMap1D::from_interval(Interval::new(a, b).unwrap())
                .unwrap()
                .unwrap();
            let mut points = [S::ZERO; 15];
            rule.points(map, &mut points).unwrap();
            let mut samples = [[S::ZERO; 25]; 15];
            for (&x, sample) in points.iter().zip(&mut samples) {
                let (sin, cos) = ((p * x).sin(), (p * x).cos());
                let (mut previous, mut current) = (S::ONE, x);
                sample[0] = cos;
                sample[1] = x * sin;
                for (k, value) in sample.iter_mut().enumerate().skip(2) {
                    let next = two * x * current - previous;
                    *value = next * if k % 2 == 0 { cos } else { sin };
                    previous = current;
                    current = next;
                }
            }
            for (k, sum) in sums.iter_mut().enumerate() {
                sum.add(
                    rule.apply_samples::<0>(map.half_weight().unwrap(), |i| Ok(samples[i][k]))
                        .unwrap()
                        .value(),
                )
                .unwrap();
            }
        }
        sums.map(|sum| sum.finish().unwrap())
    }

    fn analytic_moments_match_closed_forms_and_quadrature<S: RealScalar>(slack: f64) {
        let s = |x: f64| S::try_from_f64_lossy(x).unwrap();
        for p in [0.0, 1e-9, 1e-3, 0.5, 1.0, 2.0, 3.7, 9.5, 16.25, 23.9, 24.0] {
            let p = s(p);
            let moments = fourier(p).unwrap();
            let reference = quadrature(p);
            for (k, ((&value, &error), &reference)) in moments
                .values
                .iter()
                .zip(&moments.errors)
                .zip(&reference)
                .enumerate()
            {
                assert!(
                    (value - reference).abs() <= error + s(slack) * S::EPSILON,
                    "p={p:?} k={k}: {value:?} versus {reference:?}"
                );
            }
            if p > S::ZERO {
                let (sin, cos) = (p.sin(), p.cos());
                let closed = [
                    s(2.0) * sin / p,
                    s(2.0) * (sin - p * cos) / (p * p),
                    (s(8.0) * cos + (s(2.0) * p * p - s(8.0)) * sin / p) / (p * p),
                ];
                // The closed forms cancel catastrophically for small p.
                if p >= S::ONE {
                    for (k, closed) in closed.into_iter().enumerate() {
                        assert!(
                            (moments.values[k] - closed).abs()
                                <= moments.errors[k] + s(64.0) * S::EPSILON,
                            "p={p:?} k={k}"
                        );
                    }
                }
            } else {
                for k in 0..25 {
                    let n = s(k as f64);
                    let exact = if k % 2 == 0 {
                        s(2.0) / (S::ONE - n * n)
                    } else {
                        S::ZERO
                    };
                    assert_eq!(moments.values[k], exact);
                }
            }
        }
        // The analytic and recurrence regimes agree across their boundary.
        let below = fourier(s(24.0)).unwrap();
        let above = fourier(s(24.0).next_after(S::INFINITY)).unwrap();
        for k in 0..25 {
            assert!(
                (below.values[k] - above.values[k]).abs()
                    <= below.errors[k] + above.errors[k] + s(256.0) * S::EPSILON,
                "k={k}"
            );
        }
    }

    #[test]
    fn analytic_fourier_moments_match_independent_references() {
        analytic_moments_match_closed_forms_and_quadrature::<f32>(32.0);
        analytic_moments_match_closed_forms_and_quadrature::<f64>(8.0);
    }
}
