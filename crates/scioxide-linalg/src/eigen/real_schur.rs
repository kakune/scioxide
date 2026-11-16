//! Francis double-shift Hessenberg QR and scaled 2×2 block standardization.
//!
//! Bulge reflectors reuse the narrow Householder arithmetic in their known
//! unit 1-norm frame. Values-only jobs standardize converged pairs locally;
//! active blocks through order 32 apply right updates immediately. Larger
//! blocks and vector jobs keep deferred updates and full Schur similarities.
use scioxide_core::{
    Complex, ConvergenceDiagnostics, ConvergenceReason, RealScalar, SciOxideError, SolveContext,
    Tolerance,
};

/// Standardize a real 2×2 block: a rotation equalizes the diagonal, which
/// avoids cancellation near the transition from a conjugate pair to two real
/// eigenvalues.
///
pub(super) fn standardize_pair<R: RealScalar>(
    t: &mut [R],
    n: usize,
    k: usize,
    mut q: Option<&mut [R]>,
) -> Result<(), SciOxideError> {
    let zero = R::ZERO;
    let one = R::ONE;
    let two = R::TWO;
    let a = t[k * n + k];
    let b = t[k * n + k + 1];
    let c = t[(k + 1) * n + k];
    let d = t[(k + 1) * n + k + 1];
    if c == zero {
        return Ok(());
    }
    if b == zero {
        rotate(t, n, k, zero, one, q)?;
        t[(k + 1) * n + k] = zero;
        return Ok(());
    }
    // Standardization of `[a b; c d]` by a plane
    // similarity. With `p = (a - d)/2`, a discriminant `p^2 + b c` clearly
    // above rounding level means two real eigenvalues: rotate by the
    // eigenvector direction `(z, c)` (z chosen without cancellation) to make
    // the block upper triangular. Otherwise rotate so the diagonal entries
    // become equal (their mean); the pair is then complex iff `b c < 0`, and
    // any remaining real case (same-sign or zero off-diagonals) is
    // triangularized with one more rotation. Entries are scaled by their
    // maximum before squaring.
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    let an = a / scale;
    let bn = b / scale;
    let cn = c / scale;
    let dn = d / scale;
    let p = (an - dn) / two;
    let discriminant = p * p + bn * cn;
    let four = two + two;
    if discriminant > four * <R as RealScalar>::EPSILON {
        let root = discriminant.sqrt();
        let z = if p < zero { p - root } else { p + root };
        let norm = z.hypot(cn);
        rotate(t, n, k, z / norm, cn / norm, q)?;
        t[(k + 1) * n + k] = zero;
    } else {
        let sigma = bn + cn;
        let delta = an - dn;
        let tau = sigma.hypot(delta);
        if tau > zero {
            let cs = ((one + sigma.abs() / tau) / two).sqrt();
            let sign = if sigma < zero { -one } else { one };
            let sn = -(delta / tau) / (two * cs) * sign;
            rotate(t, n, k, cs, sn, q.as_deref_mut())?;
        }
        let mean = t[k * n + k] / two + t[(k + 1) * n + k + 1] / two;
        t[k * n + k] = mean;
        t[(k + 1) * n + k + 1] = mean;
        let b = t[k * n + k + 1];
        let c = t[(k + 1) * n + k];
        if c != zero && b != zero && (b > zero) == (c > zero) {
            let rb = b.abs().sqrt();
            let rc = c.abs().sqrt();
            let norm = rb.hypot(rc);
            rotate(t, n, k, rb / norm, rc / norm, q)?;
            t[(k + 1) * n + k] = zero;
        } else if b == zero && c != zero {
            rotate(t, n, k, zero, one, q)?;
            t[(k + 1) * n + k] = zero;
        }
    }
    crate::checked::finite_candidate(t[k * n + k])?;
    crate::checked::finite_candidate(t[(k + 1) * n + k + 1])?;
    Ok(())
}

fn rotate<R: RealScalar>(
    t: &mut [R],
    n: usize,
    k: usize,
    c: R,
    s: R,
    q: Option<&mut [R]>,
) -> Result<(), SciOxideError> {
    let rotation = crate::rotation::Rotation { c, s };
    if n == 2 && q.is_none() && t.iter().all(|x| x.abs() <= R::MAX / (R::TWO + R::TWO)) {
        // A local, range-bounded pair needs only four entries. Reuse the
        // sweep kernels instead of constructing three general matrix views.
        // The guard also keeps standalone extreme Schur restandardization
        // on the range-safe application below.
        rotation.rotate_columns(t, 2, 0, 1, 0..2);
        rotation.rotate_rows(t, 2, 0, 1, 0..2)?;
        return t
            .iter()
            .try_for_each(|&x| crate::checked::finite_candidate(x).map(|_| ()));
    }
    rotation.apply(
        crate::MatrixViewMut::from_slice(n, n, t)?.transpose(),
        k,
        k + 1,
        0..n,
    )?;
    rotation.apply(crate::MatrixViewMut::from_slice(n, n, t)?, k, k + 1, 0..n)?;
    if let Some(q) = q {
        rotation.apply(
            crate::MatrixViewMut::from_slice(n, n, q)?.transpose(),
            k,
            k + 1,
            0..n,
        )?;
    }
    Ok(())
}

/// Implicit real double-shift QR on an upper Hessenberg matrix. The real path
/// keeps converged 2×2 blocks and extracts their roots in complex form.
///
pub(super) fn real_hessenberg_eigenvalues<R: RealScalar>(
    problem: HessenbergProblem<'_, R>,
    order: usize,
    tolerance: Tolerance<R>,
    iteration_budget: usize,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let HessenbergProblem {
        matrix,
        mut values,
        mut vectors,
        work,
    } = problem;

    if order == 0 {
        return Ok(());
    }

    // Francis double-shift QR in the EISPACK `hqr`/`hqr2` formulation. The
    // double shift uses the eigenvalues of the trailing 2x2 block implicitly
    // through `x`, `y` and `w = h(end,end-1) h(end-1,end)`, so complex
    // conjugate shifts never require complex arithmetic. `shift` accumulates
    // explicit exceptional shifts subtracted from the diagonal and is added
    // back when eigenvalues deflate. The constants 0.75, 0.4375 and 0.964 are
    // the classical EISPACK/Numerical Recipes exceptional-shift values used
    // to break the cycling that plain Francis shifts can fall into.
    let zero = R::ZERO;
    let two = R::TWO;
    let three_quarters = R::try_from_f64_exact(0.75)?;
    let seven_sixteenths = R::try_from_f64_exact(0.4375)?;
    let near_one = R::try_from_f64_lossy(0.964)?;
    let abs_tolerance = tolerance.absolute();
    let rel_tolerance = tolerance.relative();
    let norm = matrix
        .chunks_exact(order)
        .enumerate()
        .fold(zero, |sum, (row, data)| {
            data[row.saturating_sub(1)..]
                .iter()
                .fold(sum, |sum, value| sum + value.abs())
        });
    let mut end = order - 1;
    let mut shift = zero;
    let mut iteration = 0_usize;
    let mut steps = 0_usize;

    loop {
        solve.execution().check()?;
        // Find the top of the active unreduced block: a subdiagonal entry is
        // negligible relative to its two diagonal neighbours (or the
        // Hessenberg norm when both are zero).
        let mut start = end;
        while start > 0 {
            let scale =
                matrix[(start - 1) * order + start - 1].abs() + matrix[start * order + start].abs();
            let scale = if scale == zero { norm } else { scale };
            if matrix[start * order + start - 1].abs() <= abs_tolerance + rel_tolerance * scale {
                matrix[start * order + start - 1] = zero;
                break;
            }
            start -= 1;
        }

        let mut x = matrix[end * order + end];
        if start == end {
            matrix[end * order + end] = x + shift;
            if let Some(values) = values.as_deref_mut() {
                values[end] = Complex::new(x + shift, zero);
            }
            if end == 0 {
                return Ok(());
            }
            end -= 1;
            iteration = 0;
            continue;
        }

        let mut y = matrix[(end - 1) * order + end - 1];
        let mut w = matrix[end * order + end - 1] * matrix[(end - 1) * order + end];
        // A deflated 2x2 block is put in standard Schur form (real pair
        // split, or equal diagonals with `b c < 0` for a complex pair), from
        // which the conjugate eigenvalues are `a +- i sqrt(|b|) sqrt(|c|)`;
        // taking the roots separately avoids overflow in `|b c|`.
        if start + 1 == end {
            matrix[(end - 1) * order + end - 1] = y + shift;
            matrix[end * order + end] = x + shift;
            let pair = if vectors.is_some() {
                standardize_pair(matrix, order, end - 1, vectors.as_deref_mut())?;
                [
                    matrix[(end - 1) * order + end - 1],
                    matrix[(end - 1) * order + end],
                    matrix[end * order + end - 1],
                    matrix[end * order + end],
                ]
            } else {
                // Eigenvalues never read the off-block couplings again. Keep
                // precisely the same stable canonicalization, on four entries.
                let mut pair = [
                    y + shift,
                    matrix[(end - 1) * order + end],
                    matrix[end * order + end - 1],
                    x + shift,
                ];
                standardize_pair(&mut pair, 2, 0, None)?;
                pair
            };
            let [first, b, c, second] = pair;
            let imaginary = c.abs().sqrt() * b.abs().sqrt();
            if let Some(values) = values.as_deref_mut() {
                values[end - 1] = Complex::new(first, imaginary);
                values[end] =
                    Complex::new(second, if imaginary == zero { zero } else { -imaginary });
            }
            if end <= 1 {
                return Ok(());
            }
            end -= 2;
            iteration = 0;
            continue;
        }

        if steps >= iteration_budget {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::IterationLimit,
                ConvergenceDiagnostics::new()
                    .with_counts((scioxide_core::EvaluationCounts::from_iterations(steps)).into()),
                "general real Schur iteration exhausted its budget",
            ));
        }
        crate::solve_support::admit_iteration(
            Some(solve.execution()),
            Some(&mut *solve),
            "solve iteration budget exhausted",
        )?;
        steps += 1;

        if !matches!(iteration, 10 | 30) {
            // Use the trailing eigenvalue nearest the last
            // diagonal twice when the shifts are real. The quotient avoids
            // cancellation in that root; complex shifts keep the full pair.
            let shifts = trailing_shifts(x, y, w);
            (x, y, w) = (shifts.x, shifts.y, shifts.w);
        }
        if iteration == 10 {
            shift += x;
            for diagonal in 0..=end {
                let position = diagonal * order + diagonal;
                matrix[position] -= x;
            }
            let s = matrix[end * order + end - 1].abs() + matrix[(end - 1) * order + end - 2].abs();
            x = three_quarters * s;
            y = x;
            w = -seven_sixteenths * s * s;
        }
        if iteration == 30 {
            let mut s = (y - x) / two;
            s = s * s + w;
            if s > zero {
                s = s.sqrt();
                if y < x {
                    s = -s;
                }
                let denominator = (y - x) / two + s;
                if denominator != zero {
                    s = x - w / denominator;
                    for diagonal in 0..=end {
                        let position = diagonal * order + diagonal;
                        matrix[position] -= s;
                    }
                    shift += s;
                    x = near_one;
                    y = near_one;
                    w = near_one;
                }
            }
        }
        iteration += 1;

        francis_double_shift_sweep(
            matrix,
            vectors.as_deref_mut(),
            ActiveHessenbergBlock { order, start, end },
            DoubleShift { x, y, w },
            tolerance,
            work,
        )?;
    }
}

pub(super) struct HessenbergProblem<'a, R: RealScalar> {
    pub(super) matrix: &'a mut [R],
    pub(super) values: Option<&'a mut [Complex<R>]>,
    pub(super) vectors: Option<&'a mut [R]>,
    /// [`SWEEP`] reals per order for the deferred reflectors of a sweep.
    pub(super) work: &'a mut [R],
}

/// Reals per order of [`HessenbergProblem::work`].
pub(super) const SWEEP: usize = 4;

struct ActiveHessenbergBlock {
    order: usize,
    start: usize,
    end: usize,
}
struct DoubleShift<R> {
    x: R,
    y: R,
    w: R,
}

fn trailing_shifts<R: RealScalar>(x: R, y: R, w: R) -> DoubleShift<R> {
    let difference = (y - x) / R::TWO;
    let discriminant = difference * difference + w;
    if discriminant > R::ZERO {
        let denominator = difference + discriminant.sqrt().copysign(difference);
        let root = x - w / denominator;
        DoubleShift {
            x: root,
            y: root,
            w: R::ZERO,
        }
    } else {
        DoubleShift { x, y, w }
    }
}

/// Normalize a nonzero bulge. A representable reciprocal replaces three
/// divisions; subnormal scales keep the division so the reciprocal cannot
/// overflow. Both the split search and chase use this same bounded frame.
#[inline]
fn normalize_bulge<R: RealScalar>(p: R, q: R, r: R, scale: R) -> [R; 3] {
    let inverse = R::ONE / scale;
    if inverse.is_finite() {
        [p * inverse, q * inverse, r * inverse]
    } else {
        [p / scale, q / scale, r / scale]
    }
}

/// One implicit double-shift bulge chase over `start..=end`.
///
/// Left reflectors update their three rows at once. A right reflector at
/// column `k` is applied at once only to rows `k + 1..=k + 3`, which later
/// reflectors of the sweep still read; its updates of rows `0..=k` and of
/// the vectors are recorded in `deferred` and replayed after the chase one
/// contiguous row at a time, in the original order of each row. Without
/// vectors only the active block is updated: eigenvalues never read the
/// Schur form's off-block parts. Active values-only blocks through order 32
/// update all required rows immediately instead of recording a replay.
fn francis_double_shift_sweep<R: RealScalar>(
    matrix: &mut [R],
    vectors: Option<&mut [R]>,
    block: ActiveHessenbergBlock,
    shifts: DoubleShift<R>,
    tolerance: scioxide_core::Tolerance<R>,
    deferred: &mut [R],
) -> Result<(), SciOxideError> {
    let ActiveHessenbergBlock { order, start, end } = block;
    let DoubleShift { x, y, w } = shifts;
    let zero = R::ZERO;
    let abs_tolerance = tolerance.absolute();
    let rel_tolerance = tolerance.relative();
    // Look for two consecutive small subdiagonals (EISPACK's `m` search):
    // if the first column of the double-shift polynomial, started at
    // `bulge`, would already be decoupled from row `bulge - 1` to working
    // accuracy, the sweep can start there instead of at `start`.
    let mut bulge = end - 2;
    while bulge > start {
        let z = matrix[bulge * order + bulge];
        let r = x - z;
        let s = y - z;
        let denominator = matrix[(bulge + 1) * order + bulge];
        if denominator == zero {
            bulge -= 1;
            continue;
        }
        let p = (r * s - w) / denominator + matrix[bulge * order + bulge + 1];
        let q = matrix[(bulge + 1) * order + bulge + 1] - z - r - s;
        let r = matrix[(bulge + 2) * order + bulge + 1];
        let scale = p.abs() + q.abs() + r.abs();
        if scale != zero {
            let [p, q, r] = normalize_bulge(p, q, r, scale);
            let previous_scale = matrix[(bulge - 1) * order + bulge - 1].abs()
                + z.abs()
                + matrix[(bulge + 1) * order + bulge + 1].abs();
            if matrix[bulge * order + bulge - 1].abs() * (q.abs() + r.abs())
                <= (abs_tolerance + rel_tolerance * previous_scale)
                    * p.abs()
                    * (matrix[(bulge - 1) * order + bulge - 1].abs() + z.abs())
            {
                break;
            }
        }
        bulge -= 1;
    }

    for row in (bulge + 2)..=end {
        matrix[row * order + row - 2] = zero;
        if row > bulge + 2 {
            matrix[row * order + row - 3] = zero;
        }
    }

    let full = vectors.is_some();
    // At small active orders, recording and replaying the upper-row updates
    // costs more than their arithmetic. All updates fit in the same cache;
    // apply them immediately, preserving each row's reflector order.
    let immediate = !full && end - start < 32;
    let last_column = if full { order } else { end + 1 };
    let deferred = &mut deferred[..SWEEP * (end - bulge)];
    if !immediate {
        deferred.fill(zero);
    }
    // Bulge chase with 3x3 Householder reflectors (2x2 at the last step).
    // The first reflector is built from the first column of
    // `(H - s1 I)(H - s2 I)`; later ones re-zero the bulge that the
    // previous reflector pushed one position down. `(p, q, r)` is scaled
    // by its 1-norm before forming the reflector to avoid overflow. A
    // skipped step keeps its zero record, which replays as the identity.
    for column in bulge..end {
        let not_last = column + 1 < end;
        let (p, q, r, scale) = if column == bulge {
            let z = matrix[column * order + column];
            let r0 = x - z;
            let s0 = y - z;
            let denominator = matrix[(column + 1) * order + column];
            if denominator == zero {
                continue;
            }
            let p0 = (r0 * s0 - w) / denominator + matrix[column * order + column + 1];
            let q0 = matrix[(column + 1) * order + column + 1] - z - r0 - s0;
            let r0 = if not_last {
                matrix[(column + 2) * order + column + 1]
            } else {
                zero
            };
            let scale = p0.abs() + q0.abs() + r0.abs();
            (p0, q0, r0, scale)
        } else {
            let p0 = matrix[column * order + column - 1];
            let q0 = matrix[(column + 1) * order + column - 1];
            let r0 = if not_last {
                matrix[(column + 2) * order + column - 1]
            } else {
                zero
            };
            let scale = p0.abs() + q0.abs() + r0.abs();
            (p0, q0, r0, scale)
        };
        if scale == zero {
            continue;
        }
        let d = if not_last { 3 } else { 2 };
        let mut v = normalize_bulge(p, q, r, scale);
        let h = crate::reflector::generate_bounded_real(&mut v[..d])?;
        if column != bulge {
            matrix[column * order + column - 1] = h.beta * scale;
        } else if start != bulge && h.tau != zero {
            let position = column * order + column - 1;
            matrix[position] = -matrix[position];
        }
        crate::reflector::reflect_rows(&v[..d], h.tau, matrix, order, column, column..last_column)?;
        crate::reflector::reflect_columns(
            &v[..d],
            h.tau,
            matrix,
            order,
            column,
            (if immediate { start } else { column + 1 })..end.min(column + 3) + 1,
        );
        if !immediate {
            deferred[SWEEP * (column - bulge)..SWEEP * (column - bulge + 1)]
                .copy_from_slice(&[v[0], v[1], v[2], h.tau]);
        }
    }

    // A matrix row takes only reflectors at or above its diagonal; every
    // vector row takes all reflectors. Skipped steps retain zero tau.
    let first_row = if full { 0 } else { start };
    let deferred = &*deferred;
    for (data, rows, triangular) in core::iter::once((&mut *matrix, first_row..end + 1, true))
        .filter(|_| !immediate)
        .chain(vectors.map(|v| (v, 0..order, false)))
    {
        crate::reflector::replay(
            data,
            order,
            rows,
            bulge..end,
            triangular,
            move |data, column, rows| {
                let record = &deferred[SWEEP * (column - bulge)..SWEEP * (column - bulge + 1)];
                let d = 3.min(end - column + 1);
                crate::reflector::reflect_columns(
                    &record[..d],
                    record[3],
                    data,
                    order,
                    column,
                    rows,
                );
            },
        );
    }
    // Only the second and third subdiagonals were touched by the chase.
    // Clear their rounding residue without revisiting the zero lower triangle.
    for row in bulge + 2..=end {
        matrix[row * order + row - 2] = zero;
        if row > bulge + 2 {
            matrix[row * order + row - 3] = zero;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn shift_roots<R: RealScalar>() {
        let real = |x| R::try_from_f64_exact(x).unwrap();
        for (x, y, w) in [(4., 1., 1.), (1., 4., 1.), (2., 2., 0.25), (4., 1., -1.)] {
            let (x, y, w) = (real(x), real(y), real(w));
            let shifts = trailing_shifts(x, y, w);
            let root = shifts.x;
            let bound = real(64.) * R::EPSILON;
            assert_eq!(shifts.y, root);
            assert_eq!(shifts.w, R::ZERO);
            assert!(((root - x) * (root - y) - w).abs() <= bound);
            let other = x + y - root;
            assert!((root - x).abs() <= (other - x).abs() + bound);
        }
        let complex = trailing_shifts(real(2.), real(2.), -R::ONE);
        assert_eq!(complex.x + complex.y, real(4.));
        assert_eq!(complex.x * complex.y - complex.w, real(5.));
    }

    #[test]
    fn real_shifts_repeat_the_nearest_root_and_complex_pairs_keep_their_polynomial() {
        shift_roots::<f32>();
        shift_roots::<f64>();
    }

    fn pair_ranges<R: RealScalar>() {
        let real = |x| R::try_from_f64_exact(x).unwrap();
        for scale in [
            R::MIN_POSITIVE,
            R::ONE,
            R::MAX / real(8.),
            R::MAX / real(4.),
        ] {
            let mut pair = [scale + scale, scale, scale, scale + scale];
            standardize_pair(&mut pair, 2, 0, None).unwrap();
            assert!(pair.iter().all(|x| x.is_finite()));
            assert_eq!(pair[2], R::ZERO);
            let bound = real(32.) * R::EPSILON;
            assert!((pair[0] / scale - real(3.)).abs() <= bound);
            assert!((pair[3] / scale - R::ONE).abs() <= bound);
        }
    }

    #[test]
    fn local_pair_canonicalization_preserves_extreme_finite_spectra() {
        pair_ranges::<f32>();
        pair_ranges::<f64>();
    }

    fn bulge_frames<R: RealScalar>() {
        let tiny = R::MIN_POSITIVE.scalbn(1 - R::MANTISSA_DIGITS as i32);
        for magnitude in [
            tiny,
            R::MIN_POSITIVE,
            R::ONE,
            R::MAX / R::try_from_f64_exact(4.).unwrap(),
        ] {
            let scale = magnitude + magnitude + magnitude;
            let actual = normalize_bulge(magnitude, -magnitude, magnitude, scale);
            let expected = magnitude / scale;
            let bound = R::try_from_f64_exact(32.).unwrap() * R::EPSILON;
            for (value, reference) in actual.into_iter().zip([expected, -expected, expected]) {
                assert!(value.is_finite() && (value - reference).abs() <= bound);
            }
        }
    }

    #[test]
    fn bulge_normalization_retains_tiny_and_large_frames() {
        bulge_frames::<f32>();
        bulge_frames::<f64>();
    }

    fn sweep_invariants<R: RealScalar>() {
        let real = |x| R::try_from_f64_exact(x).unwrap();
        for n in [3, 8, 16, 31, 32, 33, 34, 35, 64] {
            let mut h = vec![R::ZERO; n * n];
            for row in 0..n {
                for column in row.saturating_sub(1)..n {
                    h[row * n + column] =
                        real(((row * 7 + column * 11 + 3) % 17) as f64 - 8.) / real(16.);
                }
            }
            let (start, end) = if n == 3 { (0, n - 1) } else { (1, n - 2) };
            if start > 0 {
                h[start * n + start - 1] = R::ZERO;
            }
            if end + 1 < n {
                h[(end + 1) * n + end] = R::ZERO;
            }
            let original = h.clone();
            let mut values_only = h.clone();
            let mut q = vec![R::ZERO; n * n];
            for j in 0..n {
                q[j * n + j] = R::ONE;
            }
            let mut work = vec![R::ZERO; SWEEP * n];
            let shifts = || DoubleShift {
                x: original[end * n + end],
                y: original[(end - 1) * n + end - 1],
                w: original[end * n + end - 1] * original[(end - 1) * n + end],
            };
            francis_double_shift_sweep(
                &mut values_only,
                None,
                ActiveHessenbergBlock {
                    order: n,
                    start,
                    end,
                },
                shifts(),
                Tolerance::new(R::ZERO, R::EPSILON).unwrap(),
                &mut work,
            )
            .unwrap();
            francis_double_shift_sweep(
                &mut h,
                Some(&mut q),
                ActiveHessenbergBlock {
                    order: n,
                    start,
                    end,
                },
                shifts(),
                Tolerance::new(R::ZERO, R::EPSILON).unwrap(),
                &mut work,
            )
            .unwrap();
            let bound = real(256. * n as f64) * R::EPSILON;
            for row in 0..n {
                for column in 0..n {
                    let left = (0..n).fold(R::ZERO, |sum, k| {
                        sum + original[row * n + k] * q[k * n + column]
                    });
                    let right =
                        (0..n).fold(R::ZERO, |sum, k| sum + q[row * n + k] * h[k * n + column]);
                    assert!((left - right).abs() <= bound, "n={n}, ({row}, {column})");
                    let inner =
                        (0..n).fold(R::ZERO, |sum, k| sum + q[k * n + row] * q[k * n + column]);
                    let expected = if row == column { R::ONE } else { R::ZERO };
                    assert!((inner - expected).abs() <= bound);
                    if row > column + 1 {
                        assert_eq!(h[row * n + column], R::ZERO);
                        assert_eq!(values_only[row * n + column], R::ZERO);
                    }
                    if (start..=end).contains(&row) && (start..=end).contains(&column) {
                        assert!(
                            (h[row * n + column] - values_only[row * n + column]).abs() <= bound
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bulge_chases_preserve_hessenberg_similarity_and_orthogonality() {
        sweep_invariants::<f32>();
        sweep_invariants::<f64>();
    }
}
