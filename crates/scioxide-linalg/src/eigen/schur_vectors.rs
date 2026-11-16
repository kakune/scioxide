//! Eigenvectors of a Schur factor in its working scalar, by triangular substitution.
//!
//! `t` is a row-major order-`n` Schur factor in normalized units: upper
//! triangular for complex scalars, or standardized quasi-triangular for real
//! scalars, where a nonzero subdiagonal marks a 2x2 block `[[a, b], [c, a]]`
//! with `b c < 0` and eigenvalues `a ± i w`, `w = sqrt|b| sqrt|c|`. Vectors
//! are written one per row of `x`: a 1x1 block's vector in its own row, and
//! the vector `re + i im` of a 2x2 block's eigenvalue `a + i w` in rows `k`
//! (re) and `k + 1` (im); the conjugate eigenvalue's vector is its conjugate.
//!
//! Right vectors solve `T x = λ x` by back substitution whose dots run over
//! contiguous rows of T. Left vectors `u^H T = λ u^H` solve `w^T T = λ w^T`
//! for `w = conj(u)` by forward substitution whose updates run over
//! contiguous rows of T. Diagonal 1x1 and 2x2 systems use complete pivoting:
//! a pivot below `minimum` is raised to `minimum` when its
//! right-hand side exceeds `minimum` and otherwise gives a zero component,
//! so repeated eigenvalues of decoupled blocks keep independent vectors. A
//! partial vector whose entries exceed `1 / eps^2` is rescaled by a power of
//! two, so no intermediate overflows; callers normalize the vectors.

use scioxide_core::backend::KernelDispatch;
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, RealScalar, SciOxideError};

use crate::kernel::{arch, standard};
use crate::rotation::scale_real;

/// Eigenvector side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Side {
    /// `T x = λ x`.
    Right,
    /// `u^H T = λ u^H`.
    Left,
}

/// Size of the diagonal block of `t` starting at `k`: 2 for a real 2x2 block.
pub(super) fn block_size<W: NumericScalar>(t: &[W], n: usize, k: usize) -> usize {
    if W::IS_REAL && k + 1 < n && t[(k + 1) * n + k] != W::ZERO {
        2
    } else {
        1
    }
}

/// Write the `side` eigenvectors of the Schur factor `t` into the rows of `x`
/// (`n x n`). `work` holds at least `2 n` elements.
///
/// # Errors
/// Propagates dispatched kernel failures.
pub(super) fn eigenvectors<W: NumericScalar>(
    t: &[W],
    n: usize,
    side: Side,
    minimum: W::Real,
    x: &mut [W],
    work: &mut [W],
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    let (first, second) = crate::workspace::split_mut(work, n)?;
    let mut k = 0;
    while k < n {
        let size = block_size(t, n, k);
        let (head, tail) = crate::workspace::split_mut(x, (k + 1) * n)?;
        let mut vector = Vector {
            re: &mut head[k * n..],
            im: (size == 2).then(|| &mut tail[..n]),
        };
        let solver = Solver {
            t,
            n,
            k,
            size,
            lambda: eigenvalue(t, n, k, size),
            minimum,
        };
        match side {
            Side::Right => solver.right(&mut vector, dispatch)?,
            Side::Left => solver.left(
                &mut vector,
                Vector {
                    re: &mut *first,
                    im: (size == 2).then_some(&mut *second),
                },
            )?,
        }
        k += size;
    }
    Ok(())
}

/// Eigenvalue of the block at `k`; a 2x2 block gives its positive member.
fn eigenvalue<W: NumericScalar>(t: &[W], n: usize, k: usize, size: usize) -> Complex<W::Real> {
    let diagonal = crate::eigen::into_complex(t[k * n + k]);
    if size == 1 {
        return diagonal;
    }
    let (b, c) = (t[k * n + k + 1].abs(), t[(k + 1) * n + k].abs());
    Complex::new(diagonal.re, b.sqrt() * c.sqrt())
}

/// One eigenvector: a single row, or the real and imaginary rows of a real
/// conjugate pair.
struct Vector<'v, W> {
    re: &'v mut [W],
    im: Option<&'v mut [W]>,
}

impl<W: NumericScalar> Vector<'_, W> {
    fn get(&self, index: usize) -> Complex<W::Real> {
        let value = crate::eigen::into_complex(self.re[index]);
        match &self.im {
            Some(im) => Complex::new(value.re, im[index].real_component()),
            None => value,
        }
    }

    fn set(&mut self, index: usize, value: Complex<W::Real>) {
        match &mut self.im {
            Some(im) => {
                self.re[index] = W::from_real(value.re);
                im[index] = W::from_real(value.im);
            }
            None => self.re[index] = W::from_components(value.re, value.im),
        }
    }

    fn fill_zero(&mut self, range: core::ops::Range<usize>) {
        self.re[range.clone()].fill(W::ZERO);
        if let Some(im) = &mut self.im {
            im[range].fill(W::ZERO);
        }
    }

    fn scale(&mut self, range: core::ops::Range<usize>, factor: W::Real) {
        self.re[range.clone()]
            .iter_mut()
            .for_each(|value| *value = scale_real(*value, factor));
        if let Some(im) = &mut self.im {
            im[range]
                .iter_mut()
                .for_each(|value| *value = scale_real(*value, factor));
        }
    }

    /// `sum_j row[j] v[j]` over `range`, as a complex number.
    fn dot(
        &self,
        row: &[W],
        range: core::ops::Range<usize>,
        dispatch: KernelDispatch,
    ) -> Result<Complex<W::Real>, SciOxideError> {
        let re = crate::eigen::into_complex(dot(&self.re[range.clone()], row, dispatch)?);
        Ok(match &self.im {
            Some(im) => Complex::new(re.re, dot(&im[range], row, dispatch)?.real_component()),
            None => re,
        })
    }

    /// `v[range] += coefficient * row[range]`.
    fn axpy(&mut self, coefficient: Complex<W::Real>, row: &[W], range: core::ops::Range<usize>) {
        let axpy = |target: &mut [W], coefficient: W| {
            target
                .iter_mut()
                .zip(row)
                .for_each(|(value, &entry)| *value = *value + coefficient * entry);
        };
        match &mut self.im {
            Some(im) => {
                axpy(&mut self.re[range.clone()], W::from_real(coefficient.re));
                axpy(&mut im[range], W::from_real(coefficient.im));
            }
            None => axpy(
                &mut self.re[range],
                W::from_components(coefficient.re, coefficient.im),
            ),
        }
    }

    fn conjugate(&mut self) {
        match &mut self.im {
            Some(im) => im.iter_mut().for_each(|value| *value = -*value),
            None => self.re.iter_mut().for_each(|value| *value = value.conj()),
        }
    }
}

fn dot<W: NumericScalar>(
    left: &[W],
    right: &[W],
    dispatch: KernelDispatch,
) -> Result<W, SciOxideError> {
    Ok(arch::dot(left, right, dispatch)?
        .unwrap_or_else(|| standard::dot(left.iter().copied().zip(right.iter().copied()))))
}

/// Substitution for the eigenvector of the block at `k`.
struct Solver<'t, W: NumericScalar> {
    t: &'t [W],
    n: usize,
    k: usize,
    size: usize,
    lambda: Complex<W::Real>,
    minimum: W::Real,
}

impl<W: NumericScalar> Solver<'_, W> {
    fn at(&self, row: usize, col: usize) -> Complex<W::Real> {
        crate::eigen::into_complex(self.t[row * self.n + col])
    }

    fn row(&self, row: usize, range: core::ops::Range<usize>) -> &[W] {
        &self.t[row * self.n + range.start..row * self.n + range.end]
    }

    /// Shifted diagonal block at `start`, transposed for left vectors.
    fn shifted(&self, start: usize, len: usize, transpose: bool) -> [[Complex<W::Real>; 2]; 2] {
        let zero = Complex::new(W::Real::ZERO, W::Real::ZERO);
        let mut block = [[zero; 2]; 2];
        for (i, row) in block.iter_mut().enumerate().take(len) {
            for (j, value) in row.iter_mut().enumerate().take(len) {
                let (r, c) = if transpose { (j, i) } else { (i, j) };
                *value = self.at(start + r, start + c);
            }
            row[i] -= self.lambda;
        }
        block
    }

    /// Back substitution for `T x = λ x`, writing rows in `0..=top`.
    fn right(
        &self,
        vector: &mut Vector<'_, W>,
        dispatch: KernelDispatch,
    ) -> Result<(), SciOxideError> {
        let (n, k) = (self.n, self.k);
        let top = k + self.size - 1;
        vector.fill_zero(0..n);
        self.start(vector, false);
        let mut i = k;
        while i > 0 {
            let len = if W::IS_REAL && i >= 2 && self.t[(i - 1) * n + i - 2] != W::ZERO {
                2
            } else {
                1
            };
            let lo = i - len;
            let mut rhs = [Complex::new(W::Real::ZERO, W::Real::ZERO); 2];
            for (offset, value) in rhs.iter_mut().enumerate().take(len) {
                *value = -vector.dot(self.row(lo + offset, i..top + 1), i..top + 1, dispatch)?;
            }
            let solution = solve(self.shifted(lo, len, false), rhs, len, self.minimum);
            solution
                .iter()
                .take(len)
                .enumerate()
                .for_each(|(offset, &value)| vector.set(lo + offset, value));
            guard(vector, &solution[..len], lo..top + 1, None)?;
            i = lo;
        }
        Ok(())
    }

    /// Forward substitution for `w^T T = λ w^T`, then `u = conj(w)`.
    fn left(
        &self,
        vector: &mut Vector<'_, W>,
        mut sums: Vector<'_, W>,
    ) -> Result<(), SciOxideError> {
        let (n, k) = (self.n, self.k);
        let top = k + self.size - 1;
        vector.fill_zero(0..n);
        sums.fill_zero(0..n);
        self.start(vector, true);
        for row in k..=top {
            sums.axpy(vector.get(row), self.row(row, top + 1..n), top + 1..n);
        }
        let mut i = top + 1;
        while i < n {
            let len = block_size(self.t, n, i);
            let mut rhs = [Complex::new(W::Real::ZERO, W::Real::ZERO); 2];
            for (offset, value) in rhs.iter_mut().enumerate().take(len) {
                *value = -sums.get(i + offset);
            }
            let solution = solve(self.shifted(i, len, true), rhs, len, self.minimum);
            solution
                .iter()
                .take(len)
                .enumerate()
                .for_each(|(offset, &value)| vector.set(i + offset, value));
            guard(
                vector,
                &solution[..len],
                k..i + len,
                Some((&mut sums, i + len..n)),
            )?;
            for row in i..i + len {
                sums.axpy(vector.get(row), self.row(row, i + len..n), i + len..n);
            }
            i += len;
        }
        vector.conjugate();
        Ok(())
    }

    /// Eigenvector of the diagonal block itself, the start of the substitution.
    fn start(&self, vector: &mut Vector<'_, W>, left: bool) {
        let k = self.k;
        let one = Complex::new(W::Real::ONE, W::Real::ZERO);
        if self.size == 1 {
            vector.set(k, one);
            return;
        }
        // Right: (B - λ) x = 0 for B = [[a, b], [c, a]], λ = a + i w. Left:
        // (B^T - λ) w = 0. Normalize by the larger off-diagonal entry.
        let (b, c) = (self.at(k, k + 1).re, self.at(k + 1, k).re);
        let (b, c) = if left { (c, b) } else { (b, c) };
        let w = self.lambda.im;
        let zero = W::Real::ZERO;
        if b.abs() >= c.abs() {
            vector.set(k, one);
            vector.set(k + 1, Complex::new(zero, w / b));
        } else {
            vector.set(k, Complex::new(-w / c, zero));
            vector.set(k + 1, Complex::new(zero, W::Real::ONE));
        }
    }
}

/// Rescale `vector[range]` (and `sums[rest]`) by a power of two when a new
/// component exceeds `1 / eps^2`.
fn guard<W: NumericScalar>(
    vector: &mut Vector<'_, W>,
    solution: &[Complex<W::Real>],
    range: core::ops::Range<usize>,
    sums: Option<(&mut Vector<'_, W>, core::ops::Range<usize>)>,
) -> Result<(), SciOxideError> {
    let big = W::Real::ONE / (W::Real::EPSILON * W::Real::EPSILON);
    let largest = solution
        .iter()
        .fold(W::Real::ZERO, |largest, value| largest.max(value.norm()));
    if largest > big {
        let factor = W::Real::ONE.scalbn(-standard::exponent(largest)?);
        vector.scale(range, factor);
        if let Some((sums, rest)) = sums {
            sums.scale(rest, factor);
        }
    }
    Ok(())
}

/// Solve the leading `len x len` system with complete pivoting and the
/// `minimum` pivot floor.
fn solve<R: RealScalar>(
    matrix: [[Complex<R>; 2]; 2],
    rhs: [Complex<R>; 2],
    len: usize,
    minimum: R,
) -> [Complex<R>; 2] {
    let zero = Complex::new(R::ZERO, R::ZERO);
    if len == 1 {
        return [divide(matrix[0][0], rhs[0], minimum), zero];
    }
    let (p, q) = (0..2)
        .flat_map(|row| (0..2).map(move |col| (row, col)))
        .fold((0, 0), |(p, q), (row, col)| {
            if matrix[row][col].norm() > matrix[p][q].norm() {
                (row, col)
            } else {
                (p, q)
            }
        });
    let pivot = matrix[p][q];
    if pivot.norm() < minimum {
        return rhs.map(|value| divide(zero, value, minimum));
    }
    let (p2, q2) = (1 - p, 1 - q);
    let multiplier = matrix[p2][q] / pivot;
    let second = divide(
        matrix[p2][q2] - multiplier * matrix[p][q2],
        rhs[p2] - multiplier * rhs[p],
        minimum,
    );
    let mut solution = [zero; 2];
    solution[q] = (rhs[p] - matrix[p][q2] * second) / pivot;
    solution[q2] = second;
    solution
}

/// `rhs / pivot`, raising a pivot below `minimum` to `minimum` when `rhs`
/// exceeds it and returning zero otherwise.
fn divide<R: RealScalar>(pivot: Complex<R>, rhs: Complex<R>, minimum: R) -> Complex<R> {
    if pivot.norm() >= minimum {
        rhs / pivot
    } else if rhs.norm() > minimum {
        rhs / Complex::new(minimum, R::ZERO)
    } else {
        Complex::new(R::ZERO, R::ZERO)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn dispatch() -> KernelDispatch {
        crate::KernelContext::new(crate::kernel::tests::test_context()).dispatch()
    }

    /// Complex eigenvector columns with their eigenvalues.
    type Columns<R> = (Vec<Vec<Complex<R>>>, Vec<Complex<R>>);

    /// Complex columns `(vectors, eigenvalues)` of `eigenvectors` output.
    fn expand<W: NumericScalar>(t: &[W], x: &[W], n: usize) -> Columns<W::Real> {
        let mut columns = Vec::new();
        let mut values = Vec::new();
        let mut k = 0;
        while k < n {
            let size = block_size(t, n, k);
            let lambda = eigenvalue(t, n, k, size);
            let column: Vec<_> = (0..n)
                .map(|i| {
                    let re = crate::eigen::into_complex(x[k * n + i]);
                    if size == 2 {
                        Complex::new(re.re, x[(k + 1) * n + i].real_component())
                    } else {
                        re
                    }
                })
                .collect();
            if size == 2 {
                columns.push(column.iter().map(|v| v.conj()).collect());
                values.push(lambda.conj());
            }
            columns.push(column);
            values.push(lambda);
            k += size;
        }
        (columns, values)
    }

    /// Largest `|T x - λ x| / |x|` (right) or `|u^H T - λ u^H| / |u|` (left).
    fn residual<W: NumericScalar>(t: &[W], n: usize, side: Side) -> W::Real
    where
        Complex<W::Real>: NumericScalar<Real = W::Real>,
    {
        let mut x = vec![W::ZERO; n * n];
        let mut work = vec![W::ZERO; 2 * n];
        eigenvectors(t, n, side, minimum::<W>(), &mut x, &mut work, dispatch()).unwrap();
        let (columns, values) = expand(t, &x, n);
        let at = |i: usize, j: usize| crate::eigen::into_complex(t[i * n + j]);
        let zero = Complex::new(W::Real::ZERO, W::Real::ZERO);
        columns
            .iter()
            .zip(&values)
            .map(|(v, &lambda)| {
                let norm = v.iter().fold(W::Real::ZERO, |m, z| m.max(z.norm()));
                assert!(norm > W::Real::ZERO);
                (0..n)
                    .map(|i| match side {
                        Side::Right => {
                            (0..n).fold(zero, |sum, j| sum + at(i, j) * v[j]) - lambda * v[i]
                        }
                        Side::Left => {
                            (0..n).fold(zero, |sum, j| sum + v[j].conj() * at(j, i))
                                - lambda * v[i].conj()
                        }
                    })
                    .fold(W::Real::ZERO, |m, z| m.max(z.norm()))
                    / norm
            })
            .fold(W::Real::ZERO, W::Real::max)
    }

    fn minimum<W: NumericScalar>() -> W::Real {
        W::Real::EPSILON
    }

    fn cast<R: RealScalar>(value: f64) -> R {
        R::try_from_f64_lossy(value).unwrap()
    }

    /// Standardized quasi-triangular T: 1x1 blocks, conjugate pairs on both
    /// sides of 1x1 blocks, and a coupled upper part.
    fn quasi_triangular<R: RealScalar>() -> (Vec<R>, usize) {
        let n = 8;
        let mut t = vec![R::ZERO; n * n];
        let diagonal = [0.75, 0.5, 0.5, -1.0, 2.0, 2.0, -0.25, -0.6];
        for (i, &value) in diagonal.iter().enumerate() {
            t[i * n + i] = cast(value);
            for j in i + 1..n {
                t[i * n + j] = cast(((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.4);
            }
        }
        for (k, b, c) in [(1, 2.0, -0.5), (4, 0.25, -3.0)] {
            t[k * n + k + 1] = cast(b);
            t[(k + 1) * n + k] = cast(c);
        }
        (t, n)
    }

    #[test]
    fn real_quasi_triangular_vectors_satisfy_both_sides() {
        let (t, n) = quasi_triangular::<f64>();
        for side in [Side::Right, Side::Left] {
            let residual = residual(&t, n, side);
            assert!(residual < 64.0 * f64::EPSILON, "{side:?}: {residual}");
        }
        let (t, n) = quasi_triangular::<f32>();
        for side in [Side::Right, Side::Left] {
            assert!(residual(&t, n, side) < 64.0 * f32::EPSILON);
        }
    }

    #[test]
    fn pair_start_vectors_annihilate_the_shifted_block() {
        // Both normalization branches: |b| >= |c| and |b| < |c|.
        for (b, c) in [(2.0, -0.5), (0.5, -2.0), (-3.0, 0.75), (-0.75, 3.0)] {
            let t = [1.0_f64, b, c, 1.0];
            for side in [Side::Right, Side::Left] {
                assert!(
                    residual(&t, 2, side) < 8.0 * f64::EPSILON,
                    "{b} {c} {side:?}"
                );
            }
        }
    }

    #[test]
    fn complex_triangular_vectors_satisfy_both_sides() {
        let n = 5;
        let t: Vec<_> = (0..n * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                if i > j {
                    Complex::new(0.0_f64, 0.0)
                } else if i == j {
                    Complex::new(i as f64 * 0.5 - 1.0, 0.25 * (i % 2) as f64)
                } else {
                    Complex::new(0.3 - 0.1 * j as f64, 0.2 * i as f64 - 0.15)
                }
            })
            .collect();
        for side in [Side::Right, Side::Left] {
            assert!(residual(&t, n, side) < 32.0 * f64::EPSILON);
        }
    }

    #[test]
    fn repeated_eigenvalues_of_decoupled_blocks_stay_independent() {
        // diag(1, 1) gives e_0 and e_1 rather than a perturbed multiple.
        let t = [1.0_f64, 0.0, 0.0, 1.0];
        let mut x = [0.0; 4];
        let mut work = [0.0; 4];
        eigenvectors(
            &t,
            2,
            Side::Right,
            f64::EPSILON,
            &mut x,
            &mut work,
            dispatch(),
        )
        .unwrap();
        assert_eq!(x, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn growth_is_rescaled_before_overflow() {
        // A nearly defective, strongly non-normal T: each substitution step
        // divides by the floor, so unscaled components would overflow.
        let n = 48;
        let mut t = vec![0.0_f64; n * n];
        for i in 0..n {
            t[i * n + i] = 1.0;
            for j in i + 1..n {
                t[i * n + j] = 1.0;
            }
        }
        for side in [Side::Right, Side::Left] {
            let mut x = vec![0.0; n * n];
            let mut work = vec![0.0; 2 * n];
            eigenvectors(&t, n, side, f64::EPSILON, &mut x, &mut work, dispatch()).unwrap();
            assert!(x.iter().all(|value| value.is_finite()));
            assert!(x.iter().any(|value| *value != 0.0));
        }
    }
}
