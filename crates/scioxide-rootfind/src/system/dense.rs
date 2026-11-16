//! Dense row and column product kernels used by the system solvers.
//!
//! Every matrix is square, row-major and `n × n` with `n = input.len()`. The
//! kernels fail only through an explicit extended-range recomputation, do not
//! allocate, and run inside hot loops whose callers have already checked ranges
//! and finiteness. Each call site keeps its rounding by
//! choosing its accumulation step: [`plain`] rounds the product and the sum,
//! [`fused`] contracts them where the target has a hardware fused multiply-add.
//! Changing a site's step would change solver trajectories.
//!
//! These loops deliberately do not use `scioxide_linalg` matrix-vector products,
//! which require a workspace and an execution context and re-check finiteness
//! on every call.
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, Vector};

/// Accumulation step `sum + a·b` with two roundings.
#[inline(always)]
pub(crate) fn plain<S: RealScalar>(sum: S, a: S, b: S) -> S {
    sum + a * b
}

/// Accumulation step `sum + a·b` in [`scioxide_core::fast_mul_add`] form: one
/// fused instruction with hardware FMA, a separate multiply and add otherwise,
/// never an out-of-line library FMA.
///
/// Ordinary model kernels may contract products and additions: their native
/// range screen and 8n-epsilon envelope enclose either rounding sequence.
#[inline(always)]
pub(crate) fn fused<S: RealScalar>(sum: S, a: S, b: S) -> S {
    scioxide_core::fast_mul_add(a, b, sum)
}

/// `Σₖ row[k]·input[k]` accumulated in index order by `step`.
#[inline(always)]
pub(crate) fn row_dot<S: RealScalar>(row: &[S], input: &[S], step: impl Fn(S, S, S) -> S) -> S {
    row.iter()
        .zip(input)
        .fold(S::ZERO, |sum, (&a, &b)| step(sum, a, b))
}

/// `output = M·input`, each row accumulated in index order by `step`.
///
/// Four independent row sums overlap without changing any row's order.
pub(crate) fn rows_into<S: RealScalar>(
    matrix: &[S],
    input: &[S],
    output: &mut [S],
    step: impl Fn(S, S, S) -> S + Copy,
) {
    let n = input.len();
    for (rows, targets) in matrix.chunks(4 * n).zip(output.chunks_mut(4)) {
        let mut sums = [S::ZERO; 4];
        for (k, &value) in input.iter().enumerate() {
            for (sum, row) in sums.iter_mut().zip(rows.chunks_exact(n)) {
                *sum = step(*sum, row[k], value);
            }
        }
        targets.copy_from_slice(&sums[..targets.len()]);
    }
}

/// `output = D⁻¹Rᵀ·input` for the upper triangle of `R`, with plain rounding.
///
/// Each column `j` sums `R(i, j)·input[i]` over `i ≤ j` in row order. With
/// column `divisors` `D`, each entry is divided before its product: `R(:, j)/Dⱼ`
/// is unitless, so the sum keeps the input's scale near the overflow limit.
pub(crate) fn upper_transpose_into<S: RealScalar>(
    r: &[S],
    input: &[S],
    divisors: Option<&[S]>,
    output: &mut [S],
) {
    let n = input.len();
    for (j, target) in output.iter_mut().enumerate() {
        let column = (0..=j).map(|i| (r[i * n + j], input[i]));
        *target = match divisors {
            None => column.fold(S::ZERO, |sum, (a, b)| plain(sum, a, b)),
            Some(divisors) => column.fold(S::ZERO, |sum, (a, b)| plain(sum, a / divisors[j], b)),
        };
    }
}

/// `output = Mᵀ·input`, each column accumulated in row order by `step`.
pub(crate) fn columns_into<S: RealScalar>(
    matrix: &[S],
    input: &[S],
    output: &mut [S],
    step: impl Fn(S, S, S) -> S,
) {
    output.fill(S::ZERO);
    for (row, &value) in matrix.chunks_exact(input.len()).zip(input) {
        for (target, &entry) in output.iter_mut().zip(row) {
            *target = step(*target, entry, value);
        }
    }
}

/// Checked rows of `M·input` through the core standard-accuracy product sum.
fn replayed_rows_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &[S],
    input: &[S],
    output: &mut [S],
) -> Result<(), SciOxideError> {
    for (row, target) in matrix.chunks_exact(input.len()).zip(output) {
        *target = scioxide_core::backend::standard::sum_products(
            row.iter().copied().zip(input.iter().copied()),
            scioxide_core::Accuracy::Standard,
        )?;
    }
    Ok(())
}

/// Jacobian product `J·d` with exceptional-range replay of every row.
pub(crate) fn jacobian_product_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    direction: &Vector<S>,
    output: &mut Vector<S>,
) -> Result<(), SciOxideError> {
    replayed_rows_into(
        matrix.as_slice(),
        direction.as_slice(),
        output.as_mut_slice(),
    )
}

/// `Qᵀv` (`transpose`) or `Qv` for the explicit square orthogonal factor.
pub(crate) fn orthogonal_product_into<S: RealScalar + NumericScalar<Real = S>>(
    q: &Matrix<S>,
    transpose: bool,
    vector: &Vector<S>,
    output: &mut Vector<S>,
) {
    if transpose {
        columns_into(
            q.as_slice(),
            vector.as_slice(),
            output.as_mut_slice(),
            plain,
        );
    } else {
        rows_into(
            q.as_slice(),
            vector.as_slice(),
            output.as_mut_slice(),
            plain,
        );
    }
}

/// `Rv` for the upper-triangular factor.
pub(crate) fn triangular_product_into<S: RealScalar + NumericScalar<Real = S>>(
    r: &Matrix<S>,
    vector: &Vector<S>,
    output: &mut Vector<S>,
) {
    let n = vector.len();
    for (i, (row, target)) in r
        .as_slice()
        .chunks_exact(n)
        .zip(output.as_mut_slice())
        .enumerate()
    {
        *target = row_dot(&row[i..], &vector.as_slice()[i..], plain);
    }
}

/// Gauss–Newton solution of `Rx = QᵀF`, replacing a zero diagonal entry of `R`
/// by `ε` times the largest magnitude in its column (or by `ε` for a zero column).
pub(crate) fn gauss_newton_into<S: RealScalar + NumericScalar<Real = S>>(
    r: &Matrix<S>,
    qtf: &Vector<S>,
    solution: &mut Vector<S>,
) {
    let n = qtf.len();
    let r = r.as_slice();
    let zero = <S as AlgebraicScalar>::ZERO;
    let solution = solution.as_mut_slice();
    for j in (0..n).rev() {
        let sum = (j + 1..n).fold(zero, |sum, i| sum + r[j * n + i] * solution[i]);
        let mut pivot = r[j * n + j];
        if pivot == zero {
            pivot = (0..=j).fold(zero, |largest, i| largest.max(r[i * n + j].abs())) * S::EPSILON;
            if pivot == zero {
                pivot = S::EPSILON;
            }
        }
        solution[j] = (qtf.as_slice()[j] - sum) / pivot;
    }
}

/// Normalized normal equations `−(A/s)ᵀb` and `(A/s)ᵀ(A/s)` with `s = scale`.
///
/// Scaling `A` by its largest magnitude keeps `AᵀA` from overflowing.
pub(crate) fn normal_equations_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    residual: &Vector<S>,
    scale: S,
    right_hand_side: &mut Vector<S>,
    model: &mut Matrix<S>,
) {
    let n = residual.len();
    let rows = || matrix.as_slice().chunks_exact(n);
    columns_into(
        matrix.as_slice(),
        residual.as_slice(),
        right_hand_side.as_mut_slice(),
        |sum, entry, value| sum + entry / scale * value,
    );
    right_hand_side
        .as_mut_slice()
        .iter_mut()
        .for_each(|value| *value = -*value);
    let zero = <S as AlgebraicScalar>::ZERO;
    for (i, model_row) in model.as_mut_slice().chunks_exact_mut(n).enumerate() {
        for (j, target) in model_row.iter_mut().enumerate() {
            *target = rows().fold(zero, |sum, row| sum + row[i] / scale * (row[j] / scale));
        }
    }
}

/// Largest absolute row sum.
pub(crate) fn infinity_norm<S: RealScalar + NumericScalar<Real = S>>(matrix: &Matrix<S>) -> S {
    matrix
        .as_slice()
        .chunks_exact(matrix.cols())
        .map(|row| {
            row.iter()
                .fold(<S as AlgebraicScalar>::ZERO, |sum, &x| sum + x.abs())
        })
        .fold(<S as AlgebraicScalar>::ZERO, S::max)
}

/// Model product `M·input` screened by the normwise bound `norm·‖input‖∞`.
///
/// Ordinary frames use [`fused`] four-row blocks and [`plain`] remainder rows;
/// exceptional frames replay every row in extended range.
pub(crate) fn product_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    input: &[S],
    norm: S,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let maximum = input
        .iter()
        .fold(<S as AlgebraicScalar>::ZERO, |m, &x| m.max(x.abs()));
    let bound = norm * maximum;
    if bound >= S::MIN_POSITIVE.sqrt() && bound <= S::MAX.sqrt() {
        let n = input.len();
        let blocked = 4 * (output.len() / 4);
        let (matrix, tail) = matrix.as_slice().split_at(blocked * n);
        let (output, remainder) = output.split_at_mut(blocked);
        rows_into(matrix, input, output, fused);
        for (row, target) in tail.chunks_exact(n).zip(remainder) {
            *target = row_dot(row, input, plain);
        }
        Ok(())
    } else {
        replayed_rows_into(matrix.as_slice(), input, output)
    }
}

/// Transposed model product `Mᵀ·input` screened by `n·norm·‖input‖∞`.
///
/// Ordinary frames accumulate columns with [`fused`]; exceptional frames replay
/// every column in extended range.
pub(crate) fn transpose_product_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    input: &[S],
    norm: S,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let n = input.len();
    let count = S::try_from_usize_exact(n).unwrap_or(S::MAX);
    let maximum = input
        .iter()
        .fold(<S as AlgebraicScalar>::ZERO, |m, &x| m.max(x.abs()));
    let bound = (norm * count) * maximum;
    if bound >= S::MIN_POSITIVE.sqrt() && bound <= S::MAX.sqrt() {
        columns_into(matrix.as_slice(), input, output, fused);
    } else {
        for (j, out) in output.iter_mut().enumerate() {
            *out = scioxide_core::backend::standard::sum_products(
                (0..n).map(|i| (matrix.as_slice()[i * n + j], input[i])),
                scioxide_core::Accuracy::Standard,
            )?;
        }
    }
    Ok(())
}

/// A native product and its absolute-product bound, used together for inverse
/// certification. Exceptional components replay the original row in extended range.
pub(crate) fn product_bound_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    input: &[S],
    output: &mut [S],
    bounds: &mut [S],
) -> Result<(), SciOxideError> {
    let n = input.len();
    for ((rows, targets), bound_targets) in matrix
        .as_slice()
        .chunks(4 * n)
        .zip(output.chunks_mut(4))
        .zip(bounds.chunks_mut(4))
    {
        let mut sums = [<S as AlgebraicScalar>::ZERO; 4];
        let mut absolute = [<S as AlgebraicScalar>::ZERO; 4];
        for (k, &value) in input.iter().enumerate() {
            for ((sum, bound), row) in sums.iter_mut().zip(&mut absolute).zip(rows.chunks_exact(n))
            {
                let product = row[k] * value;
                *sum = fused(*sum, row[k], value);
                *bound += product.abs();
            }
        }
        for (((out, bound), &sum), (&abs, row)) in targets
            .iter_mut()
            .zip(bound_targets)
            .zip(&sums)
            .zip(absolute.iter().zip(rows.chunks_exact(n)))
        {
            *bound = abs;
            *out = if sum.is_finite() && (abs.is_normal() || abs == <S as AlgebraicScalar>::ZERO) {
                sum
            } else {
                scioxide_core::backend::standard::sum_products(
                    row.iter().copied().zip(input.iter().copied()),
                    scioxide_core::Accuracy::Standard,
                )?
            };
        }
    }
    Ok(())
}

/// Rank-one update `output = M + left·rightᵀ` and the updated norm bound.
///
/// When a normwise bound shows that no entry can overflow, the whole outer
/// product is formed in native arithmetic. The caller's condition bound limits
/// the growth of entry rounding, and inverse steps are checked separately
/// against the stored forward matrix. Otherwise every entry is recomputed in
/// extended range before either matrix is written. `TRACK` also reports
/// whether any entry changed.
pub(crate) fn rank_one_into<S: RealScalar + NumericScalar<Real = S>, const TRACK: bool>(
    matrix: &Matrix<S>,
    left: &Vector<S>,
    right: &Vector<S>,
    old_norm: S,
    rounding: S,
    output: &mut Matrix<S>,
) -> Result<(bool, S), SciOxideError> {
    let n = right.len();
    let zero = <S as AlgebraicScalar>::ZERO;
    let left_max = left.as_slice().iter().fold(zero, |m, &x| m.max(x.abs()));
    let right_max = right.as_slice().iter().fold(zero, |m, &x| m.max(x.abs()));
    let product_bound = left_max * right_max;
    let ordinary = old_norm >= S::MIN_POSITIVE.sqrt()
        && old_norm <= S::MAX.sqrt()
        && product_bound >= S::MIN_POSITIVE.sqrt()
        && product_bound <= S::MAX.sqrt();
    let mut changed = false;
    if ordinary {
        changed = rank_one_plain::<S, TRACK>(
            matrix.as_slice(),
            left.as_slice(),
            right.as_slice(),
            output.as_mut_slice(),
        );
        let row_norm = right.as_slice().iter().fold(zero, |sum, &x| sum + x.abs());
        return Ok((changed, (old_norm + left_max * row_norm) * rounding));
    }
    let mut norm = zero;
    for ((row, target), &u) in matrix
        .as_slice()
        .chunks_exact(n)
        .zip(output.as_mut_slice().chunks_exact_mut(n))
        .zip(left.as_slice())
    {
        let mut sum = zero;
        for ((out, &old), &v) in target.iter_mut().zip(row).zip(right.as_slice()) {
            *out = crate::nonlinear_math::affine(old, u, v)?;
            if TRACK {
                changed |= *out != old;
            }
            sum += out.abs();
        }
        norm = norm.max(sum);
    }
    Ok((changed, norm))
}

/// Flat disjoint slices expose independent columns to the native vectorizer.
#[inline(never)]
fn rank_one_plain<S: RealScalar + NumericScalar<Real = S>, const TRACK: bool>(
    matrix: &[S],
    left: &[S],
    right: &[S],
    output: &mut [S],
) -> bool {
    let n = right.len();
    let mut changed = false;
    for ((row, target), &u) in matrix
        .chunks_exact(n)
        .zip(output.chunks_exact_mut(n))
        .zip(left)
    {
        for ((out, &old), &v) in target.iter_mut().zip(row).zip(right) {
            *out = fused(old, u, v);
            if TRACK {
                changed |= *out != old;
            }
        }
    }
    changed
}
