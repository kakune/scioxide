//! Bidiagonal singular values by the differential qd algorithm with shifts (dqds).
//!
//! Computes all singular values of a real bidiagonal matrix to high relative
//! accuracy (Fernando & Parlett 1994; Parlett & Marques 2000). See
//! `THIRD_PARTY_NOTICES.md` for attribution.
//! Four real lanes per value hold q, q', e, e'. A rejected transform only
//! overwrites the inactive lanes. Shifts are accumulated with compensation.
//! Exactly split input blocks are scaled separately; diagonal values never
//! need squaring. Blocks that cannot be squared without overflow or underflow
//! use the zero-shift qd map in square-root form until a split lets the
//! remaining blocks return to shifted dqds.

use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, RealScalar, SciOxideError, SolveContext,
};

use crate::kernel::standard::exponent;

pub(super) const SWEEPS_PER_VALUE: usize = 100;

struct Iteration<'a, 'b, 'c> {
    used: usize,
    budget: usize,
    solve: &'a mut SolveContext<'b, 'c>,
}

impl Iteration<'_, '_, '_> {
    fn admit<R: RealScalar>(&mut self, residual: impl FnOnce() -> R) -> Result<(), SciOxideError> {
        if self.used >= self.budget {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::IterationLimit,
                ConvergenceDiagnostics::new()
                    .with_counts(scioxide_core::EvaluationCounts::from_iterations(self.used).into())
                    .with_residual(scioxide_core::ResidualReport::unscaled(
                        residual().try_to_f64_exact().unwrap_or(f64::INFINITY),
                        scioxide_core::NormSelection::Infinity,
                    )?),
                "SVD bidiagonal dqds iteration exhausted its budget",
            ));
        }
        crate::solve_support::admit_iteration(
            Some(self.solve.execution()),
            Some(self.solve),
            "no complete SVD candidate",
        )?;
        self.used += 1;
        Ok(())
    }
}

pub(super) fn solve<R: RealScalar>(
    diagonal: &mut [R],
    off: &mut [R],
    work: &mut [R],
    budget: usize,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let mut iteration = Iteration {
        used: 0,
        budget,
        solve,
    };
    let mut first = 0;
    while first < diagonal.len() {
        iteration.solve.execution().check()?;
        let mut end = first + 1;
        while end < diagonal.len() && off[end - 1] != R::ZERO {
            end += 1;
        }
        let d = &mut diagonal[first..end];
        let e = &mut off[first..end];
        for value in d.iter_mut().chain(e.iter_mut()) {
            *value = value.abs();
        }
        match d.len() {
            1 => {}
            2 => (d[0], d[1]) = pair(d[0], e[0], d[1]),
            n => {
                let maximum = d
                    .iter()
                    .chain(e.iter())
                    .copied()
                    .fold(R::ZERO, RealScalar::max);
                // LASQ1's sqrt(epsilon / safe_minimum) frame, rounded down to
                // an exact power of two, leaves headroom for qd additions.
                let target = (-exponent(R::MIN_POSITIVE)? - R::MANTISSA_DIGITS as i32) / 2;
                let shift = target - exponent(maximum)?;
                let z = &mut work[..4 * n];
                z.fill(R::ZERO);
                let mut normal = true;
                for (i, (&d, &e)) in d.iter().zip(e.iter()).enumerate() {
                    for (lane, value) in [(0, d), (2, e)] {
                        let scaled = value.scalbn(shift);
                        let square = scaled * scaled;
                        normal &= value == R::ZERO || square >= R::MIN_POSITIVE;
                        z[4 * i + lane] = square;
                    }
                }
                if normal && qd_values(z, &mut iteration, -shift)? {
                    for (value, lanes) in d.iter_mut().zip(z.chunks_exact(4)) {
                        *value = lanes[0].sqrt().scalbn(-shift);
                    }
                } else {
                    split_root_qd(d, e, &mut iteration)?;
                    // Recovery has isolated at least one value or subblock.
                    // Reassess each new block in its own frame instead of
                    // forcing a remaining cluster through zero-shift qd.
                    continue;
                }
            }
        }
        e.fill(R::ZERO);
        first = end;
    }
    crate::checked::validate_values(
        diagonal,
        scioxide_core::NonFiniteOrigin::Output,
        "bidiagonal singular value is not finite",
    )?;
    Ok(())
}

// Stable singular values of [f g; 0 h]. The determinant supplies the smaller
// value without subtracting nearly equal eigenvalues or squaring tiny inputs.
fn pair<R: RealScalar>(f: R, g: R, h: R) -> (R, R) {
    let scale = f.max(g).max(h);
    if scale == R::ZERO {
        return (R::ZERO, R::ZERO);
    }
    let a = f / scale;
    let b = h / scale;
    let c = g / scale;
    let norm = ((a + b).hypot(c) + (a - b).hypot(c)) / R::TWO;
    (scale * norm, f.min(h) * ((f.max(h) / scale) / norm))
}

fn tolerance<R: RealScalar>() -> R {
    // LASQ's relative deflation constant; squared comparisons are used on qd.
    (R::TEN * R::TEN) * R::EPSILON
}

fn reverse<R: RealScalar>(z: &mut [R], first: usize, end: usize) {
    for i in 0..(end - first) / 2 {
        for p in 0..2 {
            z.swap(4 * (first + i) + p, 4 * (end - 1 - i) + p);
        }
    }
    for i in 0..(end - first - 1) / 2 {
        for p in 0..2 {
            z.swap(4 * (first + i) + 2 + p, 4 * (end - 2 - i) + 2 + p);
        }
    }
}

#[derive(Clone, Copy)]
struct Transform<R> {
    underflow: bool,
    minimum: R,
    minimum1: R,
    minimum2: R,
    last: R,
    last1: R,
    last2: R,
}

impl<R: RealScalar> Transform<R> {
    fn zero() -> Self {
        Self {
            underflow: false,
            minimum: R::ZERO,
            minimum1: R::ZERO,
            minimum2: R::ZERO,
            last: R::ZERO,
            last1: R::ZERO,
            last2: R::ZERO,
        }
    }
}

// DLASQ5/6 in a common ping-pong loop. The guarded zero-shift transform
// changes the division order whenever the ordinary ratio could leave range.
// Deflation-aware range classification is only needed after the ordinary
// transform detects underflow; keep that work out of the common recurrence.
fn transform<R: RealScalar, const DEFLATE: bool>(
    z: &mut [R],
    first: usize,
    end: usize,
    p: usize,
    tau: R,
    threshold: R,
    safe: bool,
) -> Transform<R> {
    let out = 1 - p;
    let mut d = z[4 * first + p] - tau;
    let mut result = Transform {
        minimum: d,
        ..Transform::zero()
    };
    let mut emin = z[4 * (first + 1) + p];
    for i in first..end - 1 {
        if i == end - 3 {
            result.last2 = d;
            result.minimum2 = result.minimum;
        }
        if i == end - 2 {
            result.last1 = d;
            result.minimum1 = result.minimum;
        }
        let e = z[4 * i + 2 + p];
        let next = z[4 * (i + 1) + p];
        let previous = d;
        let q = d + e;
        z[4 * i + out] = q;
        let new_e;
        if safe && q == R::ZERO {
            new_e = R::ZERO;
            d = next;
            result.minimum = d;
            emin = R::ZERO;
        } else if i < end - 3
            && (!safe || (R::MIN_POSITIVE * next < q && R::MIN_POSITIVE * q < next))
        {
            let ratio = next / q;
            new_e = e * ratio;
            d = d * ratio - tau;
        } else {
            new_e = next * (e / q);
            d = next * (d / q) - tau;
        }
        // LASQ3 can deflate using the preceding coupling e and the new q.
        // A tiny new_e is ordinary convergence when that test already
        // succeeds; abandoning dqds here discards a nearly converged value.
        // A tiny d still requires unsquared recovery unless its contribution
        // is negligible relative to the accumulated shift (threshold).
        if q > R::ZERO && next > R::ZERO {
            if DEFLATE {
                let tol = tolerance::<R>();
                let tol2 = tol * tol;
                result.underflow |= new_e < R::MIN_POSITIVE && e > tol2 * q;
                result.underflow |= tau == R::ZERO
                    && previous > R::ZERO
                    && d < R::MIN_POSITIVE
                    && !(threshold > R::ZERO && d <= threshold);
            } else {
                result.underflow |= e > R::ZERO && new_e < R::MIN_POSITIVE;
                result.underflow |= tau == R::ZERO && previous > R::ZERO && d < R::MIN_POSITIVE;
            }
        }
        if tau == R::ZERO && !safe && i < end - 3 && d < threshold {
            d = R::ZERO;
        }
        z[4 * i + 2 + out] = new_e;
        if !d.is_finite() || !new_e.is_finite() {
            result.minimum = R::NAN;
            return result;
        }
        result.minimum = result.minimum.min(d);
        emin = emin.min(new_e);
    }
    result.last = d;
    z[4 * (end - 1) + out] = d;
    z[4 * (end - 1) + 2 + out] = emin;
    result
}

struct Shift<R> {
    tau: R,
    fraction: R,
    kind: i32,
}

impl<R: RealScalar> Shift<R> {
    fn zero() -> Self {
        Self {
            tau: R::ZERO,
            fraction: R::ZERO,
            kind: 0,
        }
    }

    // LASQ4's trailing gap and Rayleigh-residual estimates. Early exits retain
    // the previous conservative shift, as in LASQ4; positivity is checked by
    // the subsequent transform before any shift is accepted.
    #[expect(
        clippy::too_many_lines,
        reason = "the numerical recurrence keeps its coupled state updates together"
    )]
    fn choose(
        &mut self,
        z: &[R],
        first: usize,
        end: usize,
        p: usize,
        old_end: usize,
        d: Transform<R>,
    ) -> Result<(), SciOxideError> {
        let quarter = R::ONE / (R::TWO * R::TWO);
        let third = R::try_from_f64_lossy(0.333)?;
        let half = R::ONE / R::TWO;
        let cnst1 = R::try_from_f64_lossy(0.563)?;
        let cnst2 = R::try_from_f64_lossy(1.01)?;
        let cnst3 = R::try_from_f64_lossy(1.05)?;
        let hundred = R::TEN * R::TEN;
        let q = |i: usize| z[4 * i + p];
        let e = |i: usize| z[4 * i + 2 + p];
        let n = end - 1;
        if d.minimum <= R::ZERO {
            self.tau = -d.minimum;
            self.kind = -1;
            return Ok(());
        }
        let mut s;
        match old_end - end {
            0 => {
                if d.minimum == d.last || d.minimum == d.last1 {
                    let b1 = q(n).sqrt() * e(n - 1).sqrt();
                    let b2 = q(n - 1).sqrt() * e(n - 2).sqrt();
                    let mut a2 = q(n - 1) + e(n - 1);
                    if d.minimum == d.last && d.minimum1 == d.last1 {
                        let gap2 = d.minimum2 - a2 - d.minimum2 * quarter;
                        let gap1 = if gap2 > R::ZERO && gap2 > b2 {
                            a2 - d.last - (b2 / gap2) * b2
                        } else {
                            a2 - d.last - (b1 + b2)
                        };
                        if gap1 > R::ZERO && gap1 > b1 {
                            s = (d.last - (b1 / gap1) * b1).max(half * d.minimum);
                            self.kind = -2;
                        } else {
                            s = (d.last - b1).max(R::ZERO);
                            if a2 > b1 + b2 {
                                s = s.min(a2 - (b1 + b2));
                            }
                            s = s.max(third * d.minimum);
                            self.kind = -3;
                        }
                    } else {
                        self.kind = -4;
                        s = quarter * d.minimum;
                        let (gamma, mut b2, stop) = if d.minimum == d.last {
                            a2 = R::ZERO;
                            if e(n - 1) > q(n - 1) {
                                return Ok(());
                            }
                            (d.last, e(n - 1) / q(n - 1), n - 1)
                        } else {
                            if z[4 * (n - 1) + 3 - p] > z[4 * n + 1 - p] {
                                return Ok(());
                            }
                            a2 = z[4 * (n - 1) + 3 - p] / z[4 * n + 1 - p];
                            if e(n - 2) > q(n - 2) {
                                return Ok(());
                            }
                            (d.last1, e(n - 2) / q(n - 2), n - 2)
                        };
                        a2 += b2;
                        for i in (first..stop).rev() {
                            if b2 == R::ZERO {
                                break;
                            }
                            let prior = b2;
                            if e(i) > q(i) {
                                return Ok(());
                            }
                            b2 *= e(i) / q(i);
                            a2 += b2;
                            if hundred * b2.max(prior) < a2 || cnst1 < a2 {
                                break;
                            }
                        }
                        a2 = cnst3 * a2;
                        if a2 < cnst1 {
                            s = gamma * (R::ONE - a2.sqrt()) / (R::ONE + a2);
                        }
                    }
                } else if d.minimum == d.last2 {
                    self.kind = -5;
                    s = quarter * d.minimum;
                    let b1 = z[4 * n + 1 - p];
                    let mut b2 = z[4 * (n - 1) + 1 - p];
                    if z[4 * (n - 2) + 3 - p] > b2 || z[4 * (n - 1) + 3 - p] > b1 {
                        return Ok(());
                    }
                    let mut a2 =
                        (z[4 * (n - 2) + 3 - p] / b2) * (R::ONE + z[4 * (n - 1) + 3 - p] / b1);
                    if n - first > 2 {
                        b2 = e(n - 3) / q(n - 3);
                        a2 += b2;
                        for i in (first..n - 3).rev() {
                            if b2 == R::ZERO {
                                break;
                            }
                            let prior = b2;
                            if e(i) > q(i) {
                                return Ok(());
                            }
                            b2 *= e(i) / q(i);
                            a2 += b2;
                            if hundred * b2.max(prior) < a2 || cnst1 < a2 {
                                break;
                            }
                        }
                        a2 = cnst3 * a2;
                    }
                    if a2 < cnst1 {
                        s = d.last2 * (R::ONE - a2.sqrt()) / (R::ONE + a2);
                    }
                } else {
                    self.fraction = match self.kind {
                        -6 => self.fraction + third * (R::ONE - self.fraction),
                        -18 => quarter * third,
                        _ => quarter,
                    };
                    s = self.fraction * d.minimum;
                    self.kind = -6;
                }
            }
            count @ (1 | 2) => {
                let (minimum, guided) = if count == 1 {
                    (d.minimum1, d.minimum1 == d.last1 && d.minimum2 == d.last2)
                } else {
                    (
                        d.minimum2,
                        d.minimum2 == d.last2 && R::TWO * e(n - 1) < q(n - 1),
                    )
                };
                if guided {
                    self.kind = if count == 1 { -7 } else { -10 };
                    s = third * minimum;
                    if e(n - 1) > q(n - 1) {
                        return Ok(());
                    }
                    let mut term = e(n - 1) / q(n - 1);
                    let mut sum = term;
                    for i in (first..n - 1).rev() {
                        if term == R::ZERO {
                            break;
                        }
                        let prior = term;
                        if e(i) > q(i) {
                            return Ok(());
                        }
                        term *= e(i) / q(i);
                        sum += term;
                        let tail = if count == 1 { term.max(prior) } else { term };
                        if hundred * tail < sum {
                            break;
                        }
                    }
                    let b2 = (cnst3 * sum).sqrt();
                    let a2 = minimum / (R::ONE + b2 * b2);
                    let gap = if count == 1 {
                        half * d.minimum2 - a2
                    } else {
                        q(n - 1) + e(n - 2) - q(n - 2).sqrt() * e(n - 2).sqrt() - a2
                    };
                    if gap > R::ZERO && gap > b2 * a2 {
                        s = s.max(a2 * (R::ONE - cnst2 * a2 * (b2 / gap) * b2));
                    } else {
                        s = s.max(a2 * (R::ONE - cnst2 * b2));
                        if count == 1 {
                            self.kind = -8;
                        }
                    }
                } else {
                    s = quarter * minimum;
                    if count == 1 && d.minimum1 == d.last1 {
                        s = half * minimum;
                    }
                    self.kind = if count == 1 { -9 } else { -11 };
                }
            }
            _ => {
                s = R::ZERO;
                self.kind = -12;
            }
        }
        self.tau = s;
        Ok(())
    }
}

// False means the squared representation left normal range. The caller still
// owns the original bidiagonal and recovers without resetting the budget.
// Keep the large LASQ iteration separate from block range recovery.
#[inline(never)]
fn qd_values<R: RealScalar>(
    z: &mut [R],
    iteration: &mut Iteration<'_, '_, '_>,
    scale: i32,
) -> Result<bool, SciOxideError> {
    let order = z.len() / 4;
    let tol = tolerance::<R>();
    let tol2 = tol * tol;
    let bias = (R::TWO + R::ONE) / R::TWO;
    if bias * z[0] < z[4 * (order - 1)] {
        reverse(z, 0, order);
    }
    // Two initial dqd passes with Li's relative split test.
    for p in 0..2 {
        let mut d = z[4 * (order - 1) + p];
        for i in (0..order - 1).rev() {
            if z[4 * i + 2 + p] <= tol2 * d {
                z[4 * i + 2 + p] = R::ZERO;
                d = z[4 * i + p];
            } else {
                d = z[4 * i + p] * (d / (d + z[4 * i + 2 + p]));
            }
        }
        iteration.admit(|| residual(z, 0, order, p, scale))?;
        if transform::<R, false>(z, 0, order, p, R::ZERO, R::ZERO, true).underflow {
            iteration.admit(|| residual(z, 0, order, p, scale))?;
            if transform::<R, true>(z, 0, order, p, R::ZERO, R::ZERO, true).underflow {
                return Ok(false);
            }
        }
    }
    let mut end = order;
    let mut state = Transform::zero();
    let mut shift = Shift::zero();
    while end > 0 {
        let mut sigma = if end == order {
            R::ZERO
        } else {
            -z[4 * (end - 1) + 2]
        };
        let mut correction = R::ZERO;
        let mut first = end - 1;
        let mut qmin = z[4 * first];
        let mut qmax = qmin;
        let mut emax = R::ZERO;
        while first > 0 && z[4 * (first - 1) + 2] > R::ZERO {
            if qmin >= (R::TWO * R::TWO) * emax {
                qmin = qmin.min(z[4 * first]);
                emax = emax.max(z[4 * (first - 1) + 2]);
            }
            qmax = qmax.max(z[4 * (first - 1)] + z[4 * (first - 1) + 2]);
            first -= 1;
        }
        state.minimum = -(qmin - R::TWO * qmin.sqrt() * emax.sqrt()).max(R::ZERO);
        let mut p = 0;
        while first < end {
            iteration.solve.execution().check()?;
            let old_end = end;
            // LASQ3: test both ping/pong couplings before dropping one or two
            // trailing values; the other lane is the preceding transform.
            while end > first {
                let n = end - 1;
                if end - first == 1
                    || z[4 * (n - 1) + 2 + p] <= tol2 * (sigma + z[4 * n + p])
                    || z[4 * (n - 1) + 3 - p] <= tol2 * z[4 * (n - 1) + p]
                {
                    z[4 * n] = z[4 * n + p] + sigma;
                    end -= 1;
                } else if end - first == 2
                    || z[4 * (n - 2) + 2 + p] <= tol2 * sigma
                    || z[4 * (n - 2) + 3 - p] <= tol2 * z[4 * (n - 2) + p]
                {
                    let (hi, lo) = qd_pair(
                        z[4 * (n - 1) + p],
                        z[4 * (n - 1) + 2 + p],
                        z[4 * n + p],
                        tol2,
                    );
                    z[4 * (n - 1)] = hi + sigma;
                    z[4 * n] = lo + sigma;
                    end -= 2;
                } else {
                    break;
                }
            }
            if end == first {
                break;
            }
            if (state.minimum <= R::ZERO || end < old_end)
                && bias * z[4 * first + p] < z[4 * (end - 1) + p]
            {
                reverse(z, first, end);
                state.minimum = R::ZERO;
                state.minimum2 = state.minimum2.min(z[4 * (end - 1) + 2 + p]);
                for lane in 2..4 {
                    z[4 * (end - 1) + lane] = z[4 * (end - 1) + lane]
                        .min(z[4 * first + lane])
                        .min(z[4 * (first + 1) + lane]);
                }
                qmax = qmax.max(z[4 * first + p]).max(z[4 * (first + 1) + p]);
            }
            shift.choose(z, first, end, p, old_end, state)?;
            let mut safe = false;
            loop {
                iteration.admit(|| residual(z, first, end, p, scale))?;
                let threshold = R::EPSILON * (sigma + shift.tau);
                if shift.tau < threshold / R::TWO {
                    shift.tau = R::ZERO;
                }
                state = transform::<R, false>(z, first, end, p, shift.tau, threshold, safe);
                if state.underflow {
                    iteration.admit(|| residual(z, first, end, p, scale))?;
                    // The recurrence and its minima are identical; only the
                    // range classification changes on this exceptional retry.
                    if transform::<R, true>(z, first, end, p, shift.tau, threshold, safe).underflow
                    {
                        return Ok(false);
                    }
                }
                if state.minimum >= R::ZERO && state.minimum1 >= R::ZERO {
                    break;
                }
                if state.minimum < R::ZERO
                    && state.minimum1 > R::ZERO
                    && z[4 * (end - 2) + 3 - p] < tol * (sigma + state.last1)
                    && state.last.abs() < tol * sigma
                {
                    z[4 * (end - 1) + 1 - p] = R::ZERO;
                    state.minimum = R::ZERO;
                    break;
                }
                if safe {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Overflow,
                        "SVD safe qd transform broke down",
                    ));
                }
                if state.minimum < R::ZERO {
                    if shift.kind < -22 {
                        shift.tau = R::ZERO;
                    } else if state.minimum1 > R::ZERO {
                        shift.tau = (shift.tau + state.minimum) * (R::ONE - R::TWO * R::EPSILON);
                        shift.kind -= 11;
                    } else {
                        shift.tau /= R::TWO * R::TWO;
                        shift.kind -= 12;
                    }
                } else if shift.tau != R::ZERO {
                    shift.tau = R::ZERO;
                } else {
                    safe = true;
                }
            }
            let total;
            if shift.tau < sigma {
                correction += shift.tau;
                total = sigma + correction;
                correction -= total - sigma;
            } else {
                total = sigma + shift.tau;
                correction = sigma - (total - shift.tau) + correction;
            }
            sigma = total;
            p = 1 - p;
            if p == 0
                && end - first >= 4
                && (z[4 * end - 1] <= tol2 * qmax || z[4 * end - 2] <= tol2 * sigma)
            {
                let mut split = first;
                qmax = z[4 * first];
                let mut emin = z[4 * first + 2];
                let mut old_emin = z[4 * first + 3];
                for i in first..end - 3 {
                    if z[4 * i + 3] <= tol2 * z[4 * i] || z[4 * i + 2] <= tol2 * sigma {
                        z[4 * i + 2] = -sigma;
                        split = i + 1;
                        qmax = R::ZERO;
                        emin = z[4 * (i + 1) + 2];
                        old_emin = z[4 * (i + 1) + 3];
                    } else {
                        qmax = qmax.max(z[4 * (i + 1)]);
                        emin = emin.min(z[4 * i + 2]);
                        old_emin = old_emin.min(z[4 * i + 3]);
                    }
                }
                z[4 * end - 2] = emin;
                z[4 * end - 1] = old_emin;
                first = split;
            }
        }
    }
    Ok(true)
}

fn residual<R: RealScalar>(z: &[R], first: usize, end: usize, p: usize, scale: i32) -> R {
    z[4 * first..4 * (end - 1)]
        .chunks_exact(4)
        .map(|lanes| lanes[2 + p].abs().sqrt().scalbn(scale))
        .fold(R::ZERO, RealScalar::max)
}

fn qd_pair<R: RealScalar>(a: R, e: R, b: R, tol2: R) -> (R, R) {
    let hi = a.max(b);
    let lo = a.min(b);
    let t = ((hi - lo) + e) / R::TWO;
    if e <= lo * tol2 || t == R::ZERO {
        return (hi, lo);
    }
    let s = lo * (e / t);
    let denominator = if s <= t {
        t * (R::ONE + (R::ONE + s / t).sqrt())
    } else {
        t + t.sqrt() * (t + s).sqrt()
    };
    let upper = hi + (lo * (e / denominator) + e);
    (upper, lo * (hi / upper))
}

// Work in the unsquared representation only until a new exact split or
// relative deflation permits the caller to reconsider the block ranges.
fn split_root_qd<R: RealScalar>(
    d: &mut [R],
    e: &mut [R],
    iteration: &mut Iteration<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let end = d.len();
    let tol = tolerance::<R>();
    loop {
        iteration.solve.execution().check()?;
        if e[..end - 1].contains(&R::ZERO) {
            return Ok(());
        }
        // Demmel--Kahan's inverse-bidiagonal lower bound (xBDSQR). Local
        // diagonal magnitudes alone need not resolve the smallest value of
        // a strongly coupled block. No absolute safe-minimum floor is added.
        let mut mu = d[0];
        let mut minimum = mu;
        for i in 0..end - 1 {
            mu = d[i + 1] * (mu / (mu + e[i]));
            minimum = minimum.min(mu);
        }
        let lower_bound = minimum / crate::checked::rounded_count::<R>(end)?.sqrt();
        if e[end - 2] <= tol * lower_bound {
            e[end - 2] = R::ZERO;
            return Ok(());
        }
        if d[0] < d[end - 1] {
            d.reverse();
            e[..end - 1].reverse();
        }
        iteration.admit(|| e[..end - 1].iter().copied().fold(R::ZERO, RealScalar::max))?;
        let mut carry = d[0];
        for i in 0..end - 1 {
            let q = carry.hypot(e[i]);
            let next = d[i + 1];
            if q == R::ZERO {
                e[i] = R::ZERO;
                carry = next;
            } else {
                e[i] = next * (e[i] / q);
                carry = next * (carry / q);
            }
            d[i] = q;
        }
        d[end - 1] = carry;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};

    fn values<R: RealScalar>(d: &mut [R], e: &mut [R]) {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "dqds", 1),
            0,
            0,
        );
        let mut scope = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        solve(
            d,
            e,
            &mut vec![R::ZERO; 4 * d.len()],
            SWEEPS_PER_VALUE * d.len(),
            &mut scope,
        )
        .unwrap();
        d.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    }

    use crate::bidiagonal_fixtures as bidiagonal;

    fn fixtures<R: RealScalar>() {
        for case in bidiagonal::cases::<R>() {
            let mut d = case.diagonal;
            let mut e = case.off_diagonal;
            e.push(R::ZERO);
            let expected = case.expected;
            values(&mut d, &mut e);
            let bound = crate::checked::rounded_count::<R>(128 * d.len()).unwrap() * R::EPSILON;
            for (&actual, reference) in d.iter().zip(expected) {
                assert!(
                    (actual - reference).abs() <= bound * reference + R::MIN_SUBNORMAL,
                    "case {}: {actual:?} != {reference:?}",
                    case.name
                );
            }
            assert!(e.iter().all(|&value| value == R::ZERO));
        }
    }

    #[test]
    fn independent_graded_bidiagonal_references() {
        fixtures::<f32>();
        fixtures::<f64>();
    }

    fn agreement<R: RealScalar>() {
        for n in [3, 4, 5, 8, 16, 31, 32, 33, 65] {
            for seed in 0..8 {
                let real = |x| R::try_from_f64_lossy(x).unwrap();
                let mut d: Vec<_> = (0..n)
                    .map(|i| {
                        real(match seed {
                            0 => 1.0,
                            1 => 1.0 + (i % 3) as f64 * 1e-6,
                            _ => 0.25 + ((i * 17 + seed * 11) % 31) as f64 / 16.0,
                        })
                    })
                    .collect();
                let mut e: Vec<_> = (0..n)
                    .map(|i| {
                        if i + 1 == n || (seed == 3 && i % 5 == 0) {
                            R::ZERO
                        } else {
                            real(((i * 7 + seed * 3) % 13) as f64 / 8.0)
                        }
                    })
                    .collect();
                let a = crate::Matrix::from_vec(
                    n,
                    n,
                    (0..n * n)
                        .map(|i| {
                            let (row, col) = (i / n, i % n);
                            if row == col {
                                d[row]
                            } else if row + 1 == col {
                                e[row]
                            } else {
                                R::ZERO
                            }
                        })
                        .collect(),
                )
                .unwrap();
                let reference = crate::ThinSvdDecomposition::compute(&a).unwrap();
                values(&mut d, &mut e);
                let bound = real(128.0 * n as f64) * R::EPSILON * reference.singular_values()[0];
                for (&actual, &expected) in d.iter().zip(reference.singular_values()) {
                    assert!(
                        (actual - expected).abs() <= bound,
                        "n={n} seed={seed}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn clustered_and_split_blocks_agree_with_vector_jobs() {
        agreement::<f32>();
        agreement::<f64>();
    }
}
