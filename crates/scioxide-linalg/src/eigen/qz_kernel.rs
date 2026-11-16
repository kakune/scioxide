//! Hessenberg–triangular reduction and implicit real/complex QZ iteration
//! (Moler and Stewart, 1973).
//!
//! B is first triangularized by Householder QR with `Q^H` applied to A
//! ([`hessenberg::triangularize`]), in width-32 compact-WY panels with
//! Standard level-3 updates above order 128. The Givens Hessenberg–triangular
//! reduction and the QZ sweeps then apply plain rotations and short
//! reflectors from the shared rotation and reflector modules to the
//! structurally nonzero rows and columns only. A sweep applies its
//! transformations at once only near the bulge; the updates of rows above
//! it and of Q and Z are replayed after the sweep row by row, which keeps
//! the plain arithmetic row-contiguous in row-major storage. Every operand is
//! a normalized workspace value, so no sweep intermediate can overflow; the
//! factors are checked for finiteness once, after the iteration. Shifts and
//! 2x2 pencil eigenvalues use extended range.

use super::hessenberg::{self, Packing};
use crate::kernel::frames::{extended, scalar};
use crate::rotation::Rotation;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::take_mut;
use core::ops::{Neg, Range};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, ExecutionContext, ExecutionPolicy, NonFiniteOrigin,
    NumericScalar, RealScalar, SciOxideError, SolveContext, Tolerance,
};
use scioxide_core::{Scaled, ScaledValue};

/// A dense pencil `(S, T)` staged in workspace slices, with optional
/// accumulated left (`Q`) and right (`Z`) transformations.
pub(crate) struct Pencil<'a, S: NumericScalar> {
    pub(crate) s: ScratchMatrix<'a, S>,
    pub(crate) t: ScratchMatrix<'a, S>,
    pub(crate) q: Option<ScratchMatrix<'a, S>>,
    pub(crate) z: Option<ScratchMatrix<'a, S>>,
}

/// Apply `G` to rows `k, k + 1` of S over `s_columns` and of T over
/// `t_columns`, and accumulate `Q := Q G^H`.
pub(super) fn left<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    k: usize,
    g: Rotation<S>,
    s_columns: Range<usize>,
    t_columns: Range<usize>,
) -> Result<(), SciOxideError> {
    let n = p.s.rows();
    g.rotate_rows(p.s.as_mut_slice(), n, k, k + 1, s_columns)?;
    g.rotate_rows(p.t.as_mut_slice(), n, k, k + 1, t_columns)?;
    if let Some(q) = p.q.as_mut() {
        g.conjugate()
            .rotate_columns(q.as_mut_slice(), n, k, k + 1, 0..n);
    }

    Ok(())
}

/// Apply `G^H` to columns `k, k + 1` of S over `s_rows` and of T over
/// `t_rows`, and accumulate `Z := Z G^H`.
fn right<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    k: usize,
    g: Rotation<S>,
    s_rows: Range<usize>,
    t_rows: Range<usize>,
) {
    let applied = g.conjugate();
    let n = p.s.rows();
    applied.rotate_columns(p.s.as_mut_slice(), n, k, k + 1, s_rows);
    applied.rotate_columns(p.t.as_mut_slice(), n, k, k + 1, t_rows);
    accumulate_z(p, k, applied);
}

/// `Z := Z G'` on columns `k, k + 1` for a rotation `G'` as applied to columns.
fn accumulate_z<S: NumericScalar>(p: &mut Pencil<'_, S>, k: usize, applied: Rotation<S>) {
    let n = p.s.rows();
    if let Some(z) = p.z.as_mut() {
        applied.rotate_columns(z.as_mut_slice(), n, k, k + 1, 0..n);
    }
}

/// Which pencil factor a right rotation annihilates an entry of.
#[derive(Clone, Copy)]
enum Factor {
    S,
    T,
}

/// Zero `S(row, col)` or `T(row, col)` with a right rotation on columns
/// `col, col + 1` of S over `s_rows` and of T over `t_rows`, returning the
/// rotation as applied to columns for Z.
fn clear<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    factor: Factor,
    (row, col): (usize, usize),
    s_rows: Range<usize>,
    t_rows: Range<usize>,
) -> Result<Rotation<S>, SciOxideError> {
    let n = p.s.rows();
    let m = match factor {
        Factor::S => &p.s,
        Factor::T => &p.t,
    };
    let (g, _) = Rotation::generate(
        m.get_element(row, col + 1).conj(),
        m.get_element(row, col).conj(),
    )?;
    // The right rotation `[c, -conj(s)]` acts on columns as `[c, -s]`.
    let applied = Rotation { c: g.c, s: -g.s };
    applied.rotate_columns(p.s.as_mut_slice(), n, col, col + 1, s_rows);
    applied.rotate_columns(p.t.as_mut_slice(), n, col, col + 1, t_rows);
    match factor {
        Factor::S => &mut p.s,
        Factor::T => &mut p.t,
    }
    .as_mut_slice()[row * n + col] = S::ZERO;
    Ok(applied)
}

/// Scalars per step of [`Deferred`]: three right rotations `(c, s)` and a
/// left rotation `(c, s)` or reflector `(v0, v1, v2, tau)`.
const STEP: usize = 10;

/// Transformations of one sweep over `lo..=hi` whose updates away from the
/// bulge are replayed after the chase.
///
/// A right rotation of step `k` is applied at once only to the rows below
/// `k`, which later steps still read; rows `0..=k` of S and T and every row
/// of Z take it on replay. Q takes every left transform on replay. Replay
/// walks chunks of rows step by step: each row's window stays in cache while
/// independent rows hide the dependence chain within a row.
struct Deferred<'d, S: NumericScalar> {
    steps: &'d mut [S],
    lo: usize,
    hi: usize,
    double: bool,
}

impl<S: NumericScalar> Deferred<'_, S> {
    /// Right rotations of step `k`, on columns `k, k + 1, k` (a full double
    /// step) or `k` alone.
    fn pairs(&self, k: usize) -> &'static [usize] {
        if self.double && self.hi - k + 1 >= 3 {
            &[0, 1, 0]
        } else {
            &[0]
        }
    }

    fn step(&mut self, k: usize) -> &mut [S] {
        let offset = STEP * (k - self.lo);
        &mut self.steps[offset..offset + STEP]
    }

    fn rotation(values: &[S]) -> Rotation<S> {
        Rotation {
            c: values[0].real_component(),
            s: values[1],
        }
    }

    /// Record the `index`-th right rotation of step `k`, as applied to columns.
    fn right(&mut self, k: usize, index: usize, applied: Rotation<S>) {
        let step = self.step(k);
        step[2 * index] = S::from_real(applied.c);
        step[2 * index + 1] = applied.s;
    }

    /// Record the left transform of step `k` for `Q := Q G^H` or `Q := Q H`.
    fn left(&mut self, k: usize, values: [S; 4]) {
        self.step(k)[6..].copy_from_slice(&values);
    }

    fn replay(&self, p: &mut Pencil<'_, S>) {
        let n = p.s.rows();
        self.rights(p.s.as_mut_slice(), n, 0..self.hi + 1, true);
        self.rights(p.t.as_mut_slice(), n, 0..self.hi + 1, true);
        if let Some(z) = p.z.as_mut() {
            self.rights(z.as_mut_slice(), n, 0..n, false);
        }
        if let Some(q) = p.q.as_mut() {
            self.lefts(q.as_mut_slice(), n);
        }
    }

    /// Right rotations on `rows`; with `triangular`, row `r` only takes
    /// those of steps `k >= r`.
    fn rights(&self, data: &mut [S], n: usize, rows: Range<usize>, triangular: bool) {
        crate::reflector::replay(
            data,
            n,
            rows,
            self.lo..self.hi,
            triangular,
            move |data, k, active| {
                let step = &self.steps[STEP * (k - self.lo)..];
                for (index, &offset) in self.pairs(k).iter().enumerate() {
                    Self::rotation(&step[2 * index..]).rotate_columns(
                        data,
                        n,
                        k + offset,
                        k + offset + 1,
                        active.clone(),
                    );
                }
            },
        );
    }

    fn lefts(&self, q: &mut [S], n: usize) {
        crate::reflector::replay(q, n, 0..n, self.lo..self.hi, false, move |q, k, rows| {
            let left = &self.steps[STEP * (k - self.lo) + 6..STEP * (k - self.lo) + STEP];
            if self.double {
                let d = 3.min(self.hi - k + 1);
                crate::reflector::reflect_columns(&left[..d], left[3], q, n, k, rows);
            } else {
                Self::rotation(left).rotate_columns(q, n, k, k + 1, rows);
            }
        });
    }
}

/// Scratch of an order-`n` pencil: the Householder front end, a work row
/// and the deferred transformations of a sweep.
pub(crate) fn scratch_len<S: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<usize, SciOxideError> {
    super::support::total(&[
        hessenberg::Scratch::<S>::len(n, policy)?,
        n,
        super::support::times(STEP, n)?,
    ])
}

/// Generalized Schur form of a normalized pencil: Hessenberg–triangular
/// reduction then QZ iteration, returning the work units charged to `solve`.
pub(super) fn generalized_schur<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    mut scratch: &mut [S],
    (tolerance, budget, policy): (Tolerance<S::Real>, usize, ExecutionPolicy),
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let n = p.s.rows();
    let mut front = hessenberg::Scratch::carve(&mut scratch, n, policy)?;
    let row = take_mut(&mut scratch, n)?;
    let sweep = take_mut(&mut scratch, super::support::times(STEP, n)?)?;
    reduce(p, &mut front, row, policy, solve.execution())?;
    iterate(p, sweep, tolerance, budget, solve)
}

/// Reduce a normalized pencil to Hessenberg–triangular form, forming Q from
/// the QR front end and accumulating the Givens rotations into Q and Z.
fn reduce<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    front: &mut hessenberg::Scratch<'_, S>,
    row: &mut [S],
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = p.s.rows();
    hessenberg::triangularize(&mut p.t, &mut p.s, front, row, policy, execution)?;
    if let Some(q) = p.q.as_mut() {
        hessenberg::form_q(&p.t, Packing::Triangular, front, q, policy, execution)?;
    }
    hessenberg::clear_reflectors(&mut p.t, Packing::Triangular);
    // xGGHRD: each left rotation zeroes an entry of A below its subdiagonal
    // and fills `T(row, row - 1)`; the paired right rotation removes it. The
    // right rotation acts on columns `row - 1, row > col`, so it never
    // disturbs the zeros already created in A's earlier columns.
    for col in 0..n.saturating_sub(2) {
        execution.check()?;
        for row in (col + 2..n).rev() {
            let (g, r) =
                Rotation::generate(p.s.get_element(row - 1, col), p.s.get_element(row, col))?;
            p.s.as_mut_slice()[(row - 1) * n + col] = r;
            p.s.as_mut_slice()[row * n + col] = S::ZERO;
            left(p, row - 1, g, col + 1..n, row - 1..n)?;
            let applied = clear(p, Factor::T, (row, row - 1), 0..n, 0..row + 1)?;
            accumulate_z(p, row - 1, applied);
        }
    }
    Ok(())
}

type Root<R> = (
    ScaledValue<scioxide_core::Complex<R>>,
    ScaledValue<scioxide_core::Complex<R>>,
);
/// Stable homogeneous quadratic roots of det(A - lambda B), without inversion.
///
pub(super) fn roots<S: NumericScalar, M: MatrixStorage<S>>(
    a: &M,
    b: &M,
    k: usize,
) -> Result<[Root<S::Real>; 2], SciOxideError> {
    // Projective eigenvalues `(alpha, beta)` (lambda = alpha / beta) of the
    // 2x2 pencil at k: roots of `det(A - lambda B) = c2 lambda^2 + c1 lambda
    // + c0` with `c2 = det(B)`. The larger-magnitude `q = -(c1 +- d)/2`
    // avoids cancellation, and the roots are `q / c2` and `c0 / q` (Vieta),
    // returned as unreduced pairs so a zero `c2` (infinite eigenvalue) needs
    // no division. Extended range keeps the products from overflowing.
    let a00 = extended(a.get_element(k, k))?;
    let a01 = extended(a.get_element(k, k + 1))?;
    let a10 = extended(a.get_element(k + 1, k))?;
    let a11 = extended(a.get_element(k + 1, k + 1))?;
    let b00 = extended(b.get_element(k, k))?;
    let b01 = extended(b.get_element(k, k + 1))?;
    let b11 = extended(b.get_element(k + 1, k + 1))?;
    let c2 = b00.mul(b11)?;
    let c1 = a10.mul(b01)?.sub(a00.mul(b11)?)?.sub(a11.mul(b00)?)?;
    let c0 = a00.mul(a11)?.sub(a01.mul(a10)?)?;
    let one = ScaledValue::from_value(scioxide_core::Complex::new(
        S::Real::ONE,
        <_ as scioxide_core::AlgebraicScalar>::ZERO,
    ))?;
    if c2.is_zero() {
        if c1.is_zero() {
            return Ok([(one, ScaledValue::ZERO), (c0, ScaledValue::ZERO)]);
        }
        return Ok([(c0.neg(), c1), (one, ScaledValue::ZERO)]);
    }
    let two = one.add(one)?;
    let four = two.add(two)?;
    let d = c1.mul(c1)?.sub(four.mul(c2)?.mul(c0)?)?.sqrt()?;
    if S::IS_REAL && !d.imaginary().is_zero() {
        // A real quadratic's nonreal roots share one denominator and are
        // exact conjugates. Vieta's independent division rounds them differently.
        let alpha = c1.neg().add(d)?.div(two)?;
        let conjugate = ScaledValue::from_components(alpha.real(), alpha.imaginary().neg());
        return Ok([(alpha, c2), (conjugate, c2)]);
    }
    let plus = c1.add(d)?;
    let minus = c1.sub(d)?;
    let q = if plus.magnitude()?.abs_le(minus.magnitude()?) {
        minus
    } else {
        plus
    }
    .neg()
    .div(two)?;
    if q.is_zero() {
        return Ok([(ScaledValue::ZERO, one); 2]);
    }
    Ok([(q, c2), (c0, q)])
}

fn normalized_vector<S: NumericScalar>(
    v: [ScaledValue<scioxide_core::Complex<S::Real>>; 3],
) -> Result<[S; 3], SciOxideError> {
    let mut scale = Scaled::ZERO;
    for x in v {
        let mag = x.magnitude()?;
        if !mag.abs_le(scale) {
            scale = mag;
        }
    }
    if scale.is_zero() {
        return Ok([S::ZERO; 3]);
    }
    let denominator = ScaledValue::from_components(scale, Scaled::ZERO);
    let mut out = [S::ZERO; 3];
    for (dst, x) in out.iter_mut().zip(v) {
        *dst = scalar(x.div(denominator)?)?;
    }
    Ok(out)
}
fn single<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    (lo, hi): (usize, usize),
    shift: Root<S::Real>,
    sweep: &mut [S],
) -> Result<(), SciOxideError> {
    // Implicit single-shift QZ step with projective shift `alpha / beta`: the
    // first left rotation is built from the first column of
    // `beta A - alpha B` (scaled to O(1)); the bulge is then chased with
    // left rotations on A and right rotations restoring triangular B.
    let (alpha, beta) = shift;
    let v = normalized_vector::<S>([
        beta.mul(extended(p.s.get_element(lo, lo))?)?
            .sub(alpha.mul(extended(p.t.get_element(lo, lo))?)?)?,
        beta.mul(extended(p.s.get_element(lo + 1, lo))?)?,
        ScaledValue::ZERO,
    ])?;
    let n = p.s.rows();
    let mut deferred = Deferred {
        steps: sweep,
        lo,
        hi,
        double: false,
    };
    for k in lo..hi {
        let (a, b) = if k == lo {
            (v[0], v[1])
        } else {
            (p.s.get_element(k, k - 1), p.s.get_element(k + 1, k - 1))
        };
        let (g, _) = Rotation::generate(a, b)?;
        g.rotate_rows(
            p.s.as_mut_slice(),
            n,
            k,
            k + 1,
            k.saturating_sub(1).max(lo)..n,
        )?;
        g.rotate_rows(p.t.as_mut_slice(), n, k, k + 1, k..n)?;
        let q = g.conjugate();
        deferred.left(k, [S::from_real(q.c), q.s, S::ZERO, S::ZERO]);
        if k > lo {
            p.s.as_mut_slice()[(k + 1) * n + k - 1] = S::ZERO;
        }
        // The bulge reaches row `k + 2` of columns `k, k + 1` of S; rows
        // `0..=k` take the rotation on replay.
        let applied = clear(
            p,
            Factor::T,
            (k + 1, k),
            k + 1..(k + 2).min(hi) + 1,
            k + 1..k + 2,
        )?;
        deferred.right(k, 0, applied);
    }
    deferred.replay(p);
    Ok(())
}

/// Three-entry Francis polynomial vector in extended arithmetic. This computes
/// only the leading entries of the implicit polynomial, never a matrix inverse.
// First column of the Francis double-shift polynomial for `A B^{-1}`, using the
// two eigenvalues of the trailing 2x2 pencil as implicit shifts, in the
// xHGEQZ form that only needs the diagonal of B (and two of its
// superdiagonals) instead of an explicit inverse.
fn double_vector<S: NumericScalar>(
    p: &Pencil<'_, S>,
    l: usize,
    h: usize,
) -> Result<[S; 3], SciOxideError> {
    let a = |i, j| extended(p.s.get_element(i, j));
    let b = |i, j| extended(p.t.get_element(i, j));
    let ad11 = a(h - 1, h - 1)?.div(b(h - 1, h - 1)?)?;
    let ad21 = a(h, h - 1)?.div(b(h - 1, h - 1)?)?;
    let ad12 = a(h - 1, h)?.div(b(h, h)?)?;
    let ad22 = a(h, h)?.div(b(h, h)?)?;
    let u12 = b(h - 1, h)?.div(b(h, h)?)?;
    let a11 = a(l, l)?.div(b(l, l)?)?;
    let a21 = a(l + 1, l)?.div(b(l, l)?)?;
    let a12 = a(l, l + 1)?.div(b(l + 1, l + 1)?)?;
    let a22 = a(l + 1, l + 1)?.div(b(l + 1, l + 1)?)?;
    let a32 = a(l + 2, l + 1)?.div(b(l + 1, l + 1)?)?;
    let u = b(l, l + 1)?.div(b(l + 1, l + 1)?)?;
    let v0 = ad11
        .sub(a11)?
        .mul(ad22.sub(a11)?)?
        .sub(ad12.mul(ad21)?)?
        .add(ad21.mul(u12)?.mul(a11)?)?
        .add(a12.sub(a11.mul(u)?)?.mul(a21)?)?;
    let v1 = a22
        .sub(a11)?
        .sub(a21.mul(u)?)?
        .sub(ad11.sub(a11)?)?
        .sub(ad22.sub(a11)?)?
        .add(ad21.mul(u12)?)?
        .mul(a21)?;
    normalized_vector([v0, v1, a32.mul(a21)?])
}

fn double<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    (lo, hi): (usize, usize),
    sweep: &mut [S],
) -> Result<(), SciOxideError> {
    let initial = double_vector(p, lo, hi)?;
    let n = p.s.rows();
    let mut deferred = Deferred {
        steps: sweep,
        lo,
        hi,
        double: true,
    };
    for k in lo..hi {
        let d = 3.min(hi - k + 1);
        let v = if k == lo {
            initial
        } else {
            [
                p.s.get_element(k, k - 1),
                p.s.get_element(k + 1, k - 1),
                if d == 3 {
                    p.s.get_element(k + 2, k - 1)
                } else {
                    S::ZERO
                },
            ]
        };
        let mut v = v;
        let h = crate::reflector::generate(crate::VectorViewMut::from_slice(&mut v[..d]))?;
        let first = k.saturating_sub(1).max(lo);
        crate::reflector::reflect_rows(&v[..d], h.tau.conj(), p.s.as_mut_slice(), n, k, first..n)?;
        crate::reflector::reflect_rows(&v[..d], h.tau.conj(), p.t.as_mut_slice(), n, k, k..n)?;
        deferred.left(k, [v[0], v[1], v[2], h.tau]);
        if k > lo {
            for i in 1..d {
                p.s.as_mut_slice()[(k + i) * n + k - 1] = S::ZERO;
            }
        }
        // Local RQ restores triangular B without a potentially singular solve.
        // The resulting width-three A bulge (through row `k + 3`) is chased by
        // the next reflector; rows `0..=k` take the rotations on replay.
        let mut index = 0;
        for row in (k + 1..k + d).rev() {
            for col in k..row {
                let applied = clear(
                    p,
                    Factor::T,
                    (row, col),
                    k + 1..(k + 3).min(hi) + 1,
                    k + 1..k + d,
                )?;
                deferred.right(k, index, applied);
                index += 1;
            }
        }
    }
    deferred.replay(p);
    Ok(())
}
pub(super) fn canonical_pair<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    k: usize,
) -> Result<(), SciOxideError> {
    // Orthogonalize the columns of the real triangular 2x2 B, then QR.
    let a = p.t.get_element(k, k).real_component();
    let b = p.t.get_element(k, k + 1).real_component();
    let d = p.t.get_element(k + 1, k + 1).real_component();
    let scale = a.abs().max(b.abs()).max(d.abs());
    if scale == S::Real::ZERO {
        return Ok(());
    }
    let a = a / scale;
    let b = b / scale;
    let d = d / scale;
    let two = S::Real::TWO;
    let delta = b * b + d * d - a * a;
    let cross = two * a * b;
    if cross != S::Real::ZERO {
        let (c, s) = crate::rotation::jacobi(delta, -cross);
        right(
            p,
            k,
            Rotation {
                c,
                s: S::from_real(s),
            },
            0..k + 2,
            0..k + 2,
        );
    }
    let (g, _) = Rotation::generate(p.t.get_element(k, k), p.t.get_element(k + 1, k))?;
    let n = p.s.rows();
    left(p, k, g, k..n, k..n)?;
    p.t.as_mut_slice()[(k + 1) * n + k] = S::ZERO;
    // The dot product was annihilated by the Jacobi rotation above.
    p.t.as_mut_slice()[k * n + k + 1] = S::ZERO;
    Ok(())
}

fn iterate<S: NumericScalar>(
    p: &mut Pencil<'_, S>,
    sweep: &mut [S],
    tol: Tolerance<S::Real>,
    budget: usize,
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let n = p.s.rows();
    let mut end = n;
    let mut work = 0;
    let mut since = 0;
    while end > 0 {
        solve.execution().check()?;
        let hi = end - 1;
        let mut lo = hi;
        while lo > 0 {
            let local = p.s.get_element(lo - 1, lo - 1).abs() + p.s.get_element(lo, lo).abs();
            if p.s.get_element(lo, lo - 1).abs() <= tol.absolute() + tol.relative() * local {
                p.s.as_mut_slice()[lo * n + lo - 1] = S::ZERO;
                break;
            }
            lo -= 1;
        }
        if lo == hi {
            end -= 1;
            since = 0;
            continue;
        }
        if work >= budget {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::IterationLimit,
                ConvergenceDiagnostics::new().with_counts(solve.counts()),
                "QZ work budget exhausted",
            ));
        }
        crate::solve_support::admit_iteration(
            Some(solve.execution()),
            Some(&mut *solve),
            "solve iteration budget exhausted",
        )?;
        work += 1;
        since += 1;
        // Exact zeros represent infinity; small nonzero beta is not a rank cutoff.
        // A zero on B's diagonal is chased to the bottom of the block (or
        // deflated at the top directly) with rotations, as in xHGEQZ, so the
        // infinite eigenvalue splits off without dividing by that zero.
        if let Some(k) = (lo..=hi).find(|&k| p.t.get_element(k, k) == S::ZERO) {
            if k == lo {
                let (g, _) = Rotation::generate(p.s.get_element(k, k), p.s.get_element(k + 1, k))?;
                left(p, k, g, k..n, k..n)?;
                p.s.as_mut_slice()[(k + 1) * n + k] = S::ZERO;
            } else {
                for j in k..hi {
                    let (g, _) = Rotation::generate(
                        p.t.get_element(j, j + 1),
                        p.t.get_element(j + 1, j + 1),
                    )?;
                    left(p, j, g, j - 1..n, j..n)?;
                    p.t.as_mut_slice()[(j + 1) * n + j + 1] = S::ZERO;
                    let applied = clear(p, Factor::S, (j + 1, j - 1), 0..j + 2, 0..j + 2)?;
                    accumulate_z(p, j - 1, applied);
                }
                let applied = clear(p, Factor::S, (hi, hi - 1), 0..hi + 1, 0..hi + 1)?;
                accumulate_z(p, hi - 1, applied);
            }
            continue;
        }
        // Shift strategy: real pencils whose trailing 2x2 has a complex pair
        // use the double shift (real arithmetic, pair kept as a 2x2 block);
        // otherwise a single shift with the trailing root closer to the
        // bottom eigenvalue (a Wilkinson-type choice). Every 10th sweep
        // without deflation uses an ad hoc exceptional shift instead.
        let r = roots(&p.s, &p.t, hi - 1)?;
        let conjugate = S::IS_REAL && !r[0].0.div(r[0].1)?.imaginary().is_zero();
        if conjugate && hi == lo + 1 {
            canonical_pair(p, lo)?;
            end -= 2;
            since = 0;
            continue;
        }
        if since % 10 == 0 {
            let a = extended(p.s.get_element(hi, hi))?.add(ScaledValue::from_value(
                scioxide_core::Complex::new(
                    p.s.get_element(hi, hi - 1).abs(),
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                ),
            )?)?;
            single(p, (lo, hi), (a, extended(p.t.get_element(hi, hi))?), sweep)?;
        } else if conjugate {
            double(p, (lo, hi), sweep)?;
        } else {
            let bottom = extended(p.s.get_element(hi, hi))?;
            let beta = extended(p.t.get_element(hi, hi))?;
            let distance = |r: Root<S::Real>| -> Result<Scaled<S::Real>, SciOxideError> {
                r.0.mul(beta)?
                    .sub(bottom.mul(r.1)?)?
                    .magnitude()?
                    .div(r.1.magnitude()?)
            };
            let shift = if distance(r[0])?.abs_le(distance(r[1])?) {
                r[0]
            } else {
                r[1]
            };
            // A real root may carry a complex projective denominator; remove it.
            let shift = if S::IS_REAL {
                (
                    shift.0.div(shift.1)?,
                    ScaledValue::from_value(scioxide_core::Complex::new(
                        S::Real::ONE,
                        <_ as scioxide_core::AlgebraicScalar>::ZERO,
                    ))?,
                )
            } else {
                shift
            };
            single(p, (lo, hi), shift, sweep)?;
        }
    }
    // Normalize so every nonzero diagonal entry of T (a beta) is real and
    // nonnegative: scaling column j of S, T and Z by a unit phase is a
    // unitary right transformation and preserves `A = Q S Z^H`.
    for j in 0..n {
        let beta = p.t.get_element(j, j);
        let mag = beta.abs();
        if mag != S::Real::ZERO {
            let phase = crate::rotation::divide_real(beta.conj(), mag);
            for m in [Some(&mut p.s), Some(&mut p.t), p.z.as_mut()]
                .into_iter()
                .flatten()
            {
                for i in 0..n {
                    m.as_mut_slice()[i * n + j] = m.get_element(i, j) * phase;
                }
            }
            p.t.as_mut_slice()[j * n + j] = S::from_real(mag);
        }
    }
    let finite = [Some(&p.s), Some(&p.t), p.q.as_ref(), p.z.as_ref()]
        .into_iter()
        .flatten()
        .all(|m| m.as_slice().iter().all(|v| v.has_finite_components()));
    if finite {
        Ok(work)
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "generalized Schur candidate is not finite",
        ))
    }
}
