//! Band LU and Cholesky kernels that work in workspace panels.
//!
//! Wide bands are factored in column-major panels. LU gathers U12 into
//! contiguous rows, applies the panel pivots to the trailing columns, and
//! restores the elimination-order L multipliers before copying the factor
//! out. Cholesky updates its lower panel in place. Matrix–matrix products
//! update the trailing band through transposed strided views. A running bound
//! skips rescans while values stay well below overflow; values near the
//! limits of the exponent range use the scaled retries of the substitution
//! kernels. Contiguous updates keep the order of products for each entry,
//! although the sign of an exact zero can differ. Narrow bands are factored
//! column by column.
use super::super::direct::{admit, factor_requirements};
use super::super::storage::{BandedMatrixView, HermitianBandMatrixView};
use crate::kernel::{self, GemmConfig, GemmOp, TrailingBound, WorkspaceGemmProblem, standard};
use crate::workspace::ScalarRegion;
use crate::{
    DenseWorkspaceRequest, FiniteMatrixView, KernelContext, LinalgExecutionOptions,
    LinalgRequirements, LinalgWorkspace, MatrixView, MatrixViewMut,
};
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError,
    SolveContext, Triangle,
};

/// Bands with fewer subdiagonals factor column by column.
pub(super) const BLOCKED: usize = 16;
/// Narrow LU panels limit scalar pivot and U12 work while keeping the trailing
/// products blocked. Cholesky uses wider panels without pivot interchanges.
const LU_BLOCK: usize = 16;
const CHOLESKY_BLOCK: usize = 32;

/// Panel width, panel rows and active columns of a blocked factor; width one
/// selects the column-by-column kernels.
fn geometry(n: usize, lower: usize, upper: usize, block: usize) -> (usize, usize, usize) {
    if lower.min(n) < BLOCKED {
        return (1, 0, 0);
    }
    let b = block.min(lower).min(n);
    (
        b,
        n.min(lower.saturating_add(b)),
        n.min(lower.saturating_add(upper).saturating_add(b)),
    )
}

/// Band LU candidate, column-major panel, row-major U12 block, interchange
/// column and trailing-product scratch.
pub(super) fn lu_requirements<S: NumericScalar>(
    factors: usize,
    n: usize,
    lower: usize,
    upper: usize,
    limit: Option<usize>,
) -> Result<LinalgRequirements, SciOxideError> {
    let (b, rows, cols) = geometry(n, lower, upper, LU_BLOCK);
    blocked_requirements::<S>(
        factor_requirements::<S>(factors, n, limit)?,
        (b, rows, cols),
        &[(rows, b), (b, cols.saturating_sub(b)), (rows, 1)],
    )
}

/// Band Cholesky candidate, column-major panel and trailing-product scratch.
pub(super) fn cholesky_requirements<S: NumericScalar>(
    factors: usize,
    n: usize,
    bandwidth: usize,
    limit: Option<usize>,
) -> Result<LinalgRequirements, SciOxideError> {
    let (b, rows, cols) = geometry(n, bandwidth, 0, CHOLESKY_BLOCK);
    blocked_requirements::<S>(
        factor_requirements::<S>(factors, 0, limit)?,
        (b, rows, cols),
        &[(rows, b)],
    )
}

fn blocked_requirements<S: NumericScalar>(
    mut requirements: LinalgRequirements,
    (b, rows, cols): (usize, usize, usize),
    scratch: &[(usize, usize)],
) -> Result<LinalgRequirements, SciOxideError> {
    if b > 1 {
        let q = scioxide_core::quantity!("band_panel");
        let direct = requirements.scalar_mut(ScalarRegion::Direct);
        *direct = scratch.iter().try_fold(*direct, |total, &(rows, cols)| {
            scioxide_core::backend::checked_add(
                total,
                scioxide_core::backend::checked_product(&[rows, cols], q)?,
                q,
            )
        })?;
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        // Compact band updates multiply the transposed trailing rectangle so
        // output columns are contiguous. Reserve that same operand geometry.
        requirements = requirements.max(&kernel::product_requirements::<S>(
            &DenseWorkspaceRequest::product(cols - b, b, rows - b, &options),
        )?);
        requirements.required_bytes::<S>()?;
    }
    Ok(requirements)
}

#[inline]
fn larger<S: NumericScalar>(a: S, b: S) -> Result<bool, SciOxideError> {
    let (am, bm) = (a.abs(), b.abs());
    if S::IS_REAL || (am.is_finite() && bm.is_finite()) {
        return Ok(am > bm);
    }
    Ok(scioxide_core::ScaledValue::from_value(a)?
        .magnitude()?
        .abs_cmp(scioxide_core::ScaledValue::from_value(b)?.magnitude()?)
        .is_gt())
}

/// The multiplier `a / pivot` of an elimination step with a nonzero pivot.
///
/// Partial pivoting keeps its modulus at most 1, so the quotient cannot
/// overflow. A quotient below the normal range rounds to a subnormal number
/// or to zero, as in ordinary arithmetic; it is not an error.
#[inline]
pub(in crate::structured) fn multiplier<S: NumericScalar>(
    a: S,
    pivot: S,
) -> Result<S, SciOxideError> {
    crate::checked::div(a, pivot)
}

#[inline]
fn update<S: NumericScalar>(a: S, b: S, c: S) -> Result<S, SciOxideError> {
    let value = a - b * c;
    if value.has_finite_components() {
        Ok(value)
    } else {
        standard::substitute_row(a, S::ONE, [(b, c)].into_iter())
    }
}

// Ordinary panel reductions keep a scalar multiply/add in the common path;
// Standard's shared retry protects overflow and division of tiny residuals.
#[inline(always)]
fn row<S: NumericScalar>(
    prior: S,
    pivot: S,
    products: impl ExactSizeIterator<Item = (S, S)> + Clone,
) -> Result<S, SciOxideError> {
    let residual = products.clone().fold(prior, |sum, (a, x)| sum - a * x);
    standard::finish_substitution(prior, pivot, residual, products, None)
}

/// Factor `a` into the compact candidate `data`, returning the largest
/// component magnitude of U. `maximum` bounds the input components.
#[expect(
    clippy::too_many_arguments,
    reason = "the factorization keeps its input bound, candidate, scratch and controls explicit"
)]
pub(super) fn factor_lu<S: NumericScalar>(
    a: BandedMatrixView<'_, S>,
    maximum: S::Real,
    data: &mut [S],
    pivots: &mut [usize],
    scratch: &mut [S],
    gemm: &mut LinalgWorkspace<'_, S>,
    execution: ExecutionContext<'_>,
    scope: &mut SolveContext<'_, '_>,
) -> Result<S::Real, SciOxideError> {
    let context = KernelContext::new(execution);
    let (n, kl, ku) = (a.rows(), a.lower(), a.upper());
    let kv = kl + ku;
    let width = 2 * kl + ku + 1;
    for (j, column) in data.chunks_exact_mut(width).enumerate() {
        let source = a.column(j);
        let destination = kv - j.min(ku);
        column[..destination].fill(S::ZERO);
        column[destination..destination + source.len()].copy_from_slice(source);
        column[destination + source.len()..].fill(S::ZERO);
    }
    let (block, rows, cols) = geometry(n, kl, ku, LU_BLOCK);
    if block == 1 {
        lu_columns(
            data,
            pivots,
            [n, kl, ku],
            execution,
            scope,
            LuArithmetic {
                is_zero: |value| value == S::ZERO,
                larger: larger::<S>,
                quotient: multiplier::<S>,
                update: update::<S>,
            },
        )?;
        return self::maximum(data, width, kv + 1, context);
    }
    let (panel, rest) = crate::workspace::split_mut(scratch, rows * block)?;
    let (top, interchanged) = crate::workspace::split_mut(rest, block * (cols - block))?;
    let mut bound = TrailingBound::new(maximum);
    let mut upper_bound = S::Real::ZERO;
    let mut last = 0;
    for start in (0..n).step_by(block) {
        let b = block.min(n - start);
        let r = rows.min(n - start);
        // Column-major panel: column `j` holds rows `0..r` of band column
        // `start + j`, where row `i` is slot `kv + i - j`.
        let panel = &mut panel[..r * b];
        for (j, values) in panel.chunks_exact_mut(r).enumerate() {
            let end = (j + kl + 1).min(r);
            let offset = (start + j) * width + kv - j;
            values[..end].copy_from_slice(&data[offset..offset + end]);
            values[end..].fill(S::ZERO);
        }
        // Partial pivoting bounds multipliers by one in modulus. Conservative
        // fourfold component growth permits ordinary contiguous panel updates.
        // The trailing bound covers these input columns before panel work.
        let panel_maximum = bound
            .prior()
            .unwrap_or_else(|| standard::slice_maximum(panel));
        let bounded = panel_maximum <= S::Real::MAX.scalbn(-(2 * b as i32 + 3));
        // Conventional dense panel pivots also exchange preceding panel L
        // columns. Undo just those exchanges after the Schur update, restoring
        // the elimination-order multipliers used by band substitutions.
        for j in 0..b {
            admit(execution, scope)?;
            let end = (j + kl + 1).min(r);
            let column = &panel[j * r..(j + 1) * r];
            let mut pivot = j;
            for i in j + 1..end {
                if larger(column[i], column[pivot])? {
                    pivot = i;
                }
            }
            if column[pivot] == S::ZERO {
                return Err(SciOxideError::singular(Some(start + j)));
            }
            pivots[start + j] = start + pivot;
            last = last.max((start + pivot + ku).min(n - 1));
            if pivot != j {
                panel
                    .chunks_exact_mut(r)
                    .for_each(|values| values.swap(j, pivot));
            }
            let (left, right) = panel.split_at_mut((j + 1) * r);
            let multipliers = &mut left[j * r..];
            let diagonal = multipliers[j];
            let right = &mut right[..(b - j - 1) * r];
            if bounded {
                for value in &mut multipliers[j + 1..end] {
                    *value = multiplier(*value, diagonal)?;
                }
                let multipliers = &multipliers[j + 1..end];
                for target in right.chunks_exact_mut(r) {
                    let u = target[j];
                    target[j + 1..end]
                        .iter_mut()
                        .zip(multipliers)
                        .for_each(|(a, &m)| *a = *a - m * u);
                }
            } else {
                // Exceptional ranges keep the row order of their retries.
                for i in j + 1..end {
                    let value = multiplier(multipliers[i], diagonal)?;
                    multipliers[i] = value;
                    for target in right.chunks_exact_mut(r) {
                        target[i] = update(target[i], value, target[j])?;
                    }
                }
            }
        }
        let active = last + 1 - start;
        let u = active - b;
        let swapped = pivots[start..start + b]
            .iter()
            .enumerate()
            .any(|(j, &p)| p != start + j);
        // Row-major U12 block: row `i` holds panel row `i` of columns `b..active`.
        let top = &mut top[..b * u];
        if swapped {
            for col in b..active {
                let offset = (start + col) * width + kv - col;
                // A column scratch preserves every original coefficient while
                // replaying the panel's row interchanges in compact storage.
                let column = &mut interchanged[..r];
                column.fill(S::ZERO);
                let first = col.saturating_sub(kv);
                let end = (col + kl + 1).min(r);
                column[first..end].copy_from_slice(&data[offset + first..offset + end]);
                for j in 0..b {
                    column.swap(j, pivots[start + j] - start);
                }
                for i in 0..b {
                    top[i * u + col - b] = column[i];
                }
                data[offset + b..offset + r].copy_from_slice(&column[b..]);
            }
        } else if u > 0 {
            for (i, target) in top.chunks_exact_mut(u).enumerate() {
                let offset = (start + b) * width + kv + i - b;
                target.iter_mut().enumerate().for_each(|(col, value)| {
                    *value = data[offset + col * (width - 1)];
                });
            }
        }
        // U12 = L11^-1 A12 with unit pivots. Bounded rows grow at most
        // fourfold per step, so contiguous row updates repeat the checked
        // substitution exactly; other ranges substitute entry by entry.
        if bound
            .prior()
            .unwrap_or_else(|| standard::slice_maximum(top))
            > S::Real::MAX.scalbn(-(2 * b as i32 + 3))
        {
            for col in 0..u {
                for j in 0..b {
                    top[j * u + col] = row(
                        top[j * u + col],
                        S::ONE,
                        (0..j).map(|p| (panel[p * r + j], top[p * u + col])),
                    )?;
                }
            }
        } else if u > 0 {
            for j in 1..b {
                let (solved, rest) = top.split_at_mut(j * u);
                let target = &mut rest[..u];
                for (p, source) in solved.chunks_exact(u).enumerate() {
                    let l = panel[p * r + j];
                    target
                        .iter_mut()
                        .zip(source)
                        .for_each(|(x, &y)| *x = *x - l * y);
                }
            }
        }
        let upper_maximum = standard::slice_maximum(top);
        if r > b && u > 0 {
            kernel::gemm_workspace_with_context(
                WorkspaceGemmProblem {
                    left: FiniteMatrixView::from_validated(
                        MatrixView::from_slice(b, u, top)?,
                        upper_maximum,
                    ),
                    right: FiniteMatrixView::from_validated(
                        MatrixView::from_slice_with_strides(r - b, b, &panel[b..], 1, r)?,
                        S::Real::TWO,
                    ),
                    // The trailing rectangle stays within the expanded band.
                    // Updating its transpose makes output columns contiguous
                    // for native packed leaves, without copying the band.
                    output: MatrixViewMut::from_slice_with_strides(
                        u,
                        r - b,
                        &mut data[(start + b) * width + kv..],
                        width - 1,
                        1,
                    )?,
                    config: GemmConfig {
                        alpha: -S::ONE,
                        beta: S::ONE,
                        left_op: GemmOp::Transpose,
                        right_op: GemmOp::Transpose,
                        prior: bound.prior(),
                        ..GemmConfig::ordinary()
                    },
                },
                ExecutionPolicy::Sequential,
                gemm,
                context,
            )?;
            bound = bound.update(b, S::Real::TWO, upper_maximum);
        }
        for j in (0..b).rev() {
            let pivot = pivots[start + j] - start;
            if pivot != j {
                for values in panel[..j * r].chunks_exact_mut(r) {
                    values.swap(j, pivot);
                }
            }
        }
        for (j, values) in panel.chunks_exact(r).enumerate() {
            let end = (j + kl + 1).min(r);
            let offset = (start + j) * width + kv - j;
            data[offset..offset + end].copy_from_slice(&values[..end]);
        }
        for col in b..active {
            let offset = (start + col) * width + kv - col;
            for i in col.saturating_sub(kv)..b {
                data[offset + i] = top[i * u + col - b];
            }
        }
        // The block's own columns are final: measure their U while cached.
        upper_bound = upper_bound.max(self::maximum(
            &data[start * width..(start + b) * width],
            width,
            kv + 1,
            context,
        )?);
    }
    Ok(upper_bound)
}

pub(in crate::structured) struct LuArithmetic<V> {
    pub(in crate::structured) is_zero: fn(V) -> bool,
    pub(in crate::structured) larger: fn(V, V) -> Result<bool, SciOxideError>,
    pub(in crate::structured) quotient: fn(V, V) -> Result<V, SciOxideError>,
    pub(in crate::structured) update: fn(V, V, V) -> Result<V, SciOxideError>,
}

// Share the pivoting and compact-fill geometry with extended-range cyclic factors.
#[inline]
pub(in crate::structured) fn lu_columns<V: Copy>(
    data: &mut [V],
    pivots: &mut [usize],
    [n, kl, ku]: [usize; 3],
    execution: ExecutionContext<'_>,
    scope: &mut SolveContext<'_, '_>,
    arithmetic: LuArithmetic<V>,
) -> Result<(), SciOxideError> {
    let kv = kl + ku;
    let width = 2 * kl + ku + 1;
    let mut last = 0;
    for j in 0..n {
        admit(execution, scope)?;
        let end = (j + kl + 1).min(n);
        let mut pivot = j;
        for i in j + 1..end {
            if (arithmetic.larger)(
                data[j * width + kv + i - j],
                data[j * width + kv + pivot - j],
            )? {
                pivot = i;
            }
        }
        pivots[j] = pivot;
        if (arithmetic.is_zero)(data[j * width + kv + pivot - j]) {
            return Err(SciOxideError::singular(Some(j)));
        }
        last = last.max((pivot + ku).min(n - 1));
        if pivot != j {
            for c in j..=last {
                data.swap(c * width + kv + j - c, c * width + kv + pivot - c);
            }
        }
        let diagonal = data[j * width + kv];
        for i in j + 1..end {
            data[j * width + kv + i - j] =
                (arithmetic.quotient)(data[j * width + kv + i - j], diagonal)?;
        }
        for c in j + 1..=last {
            let u = data[c * width + kv + j - c];
            for i in j + 1..end {
                let index = c * width + kv + i - c;
                data[index] = (arithmetic.update)(data[index], data[j * width + kv + i - j], u)?;
            }
        }
    }
    Ok(())
}

fn positive_root<S: NumericScalar>(
    value: S,
    prior: S,
    mut products: impl Iterator<Item = (S, S)>,
) -> Result<S, SciOxideError> {
    let real = value.real_component();
    if value.has_finite_components() && real >= S::Real::MIN_POSITIVE {
        return Ok(S::from_real(real.sqrt()));
    }
    // A Schur diagonal can fall below the scalar grid while its square root
    // is representable. Retry before taking the root, just as substitution
    // retries a tiny residual before dividing by a small pivot.
    let diagonal = products
        .try_fold(
            scioxide_core::ScaledValue::from_value(prior)?,
            |sum, (a, b)| {
                sum.sub(
                    scioxide_core::ScaledValue::from_value(a)?
                        .mul(scioxide_core::ScaledValue::from_value(b)?)?,
                )
            },
        )?
        .real();
    if diagonal.is_zero() || diagonal.is_sign_negative() {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("a"),
            scioxide_core::DomainReason::OutsideDomain,
            "matrix is not positive definite",
        ));
    }
    scioxide_core::ScaledValue::<S>::from_components(diagonal.sqrt()?, scioxide_core::Scaled::ZERO)
        .round()
}

// Constant width specializes common narrow bands while preserving one arithmetic
// order, pivot admission rule and extended-range retry for every bandwidth.
fn cholesky_columns<S: NumericScalar, const BANDWIDTH: usize>(
    a: HermitianBandMatrixView<'_, S>,
    data: &mut [S],
    execution: ExecutionContext<'_>,
    scope: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let n = a.order();
    let k = if BANDWIDTH == 0 {
        a.bandwidth()
    } else {
        BANDWIDTH
    };
    let width = k + 1;
    let upper = a.triangle() == Triangle::Upper;
    let source = a.as_slice();
    // Every active factor entry is written before it is read. Only the last
    // columns contain padding that must be initialized independently.
    for j in n.saturating_sub(k)..n {
        data[j * width + (n - j).min(width)..(j + 1) * width].fill(S::ZERO);
    }
    for start in (0..n).step_by(32) {
        let end = (start + 32).min(n);
        let batched = scope
            .can_record_iterations(end - start)
            .is_ok_and(|admission| admission.is_granted());
        execution.check()?;
        let mut attempted = 0;
        // These pivots invoke no numerical providers. Preflight checks the whole
        // block, then even an arithmetic/stop error retires exactly its attempts.
        let outcome: Result<(), SciOxideError> = (|| {
            for j in start..end {
                if !batched {
                    admit(execution, scope)?;
                }
                attempted += 1;
                let products = (j.saturating_sub(k)..j).map(|c| {
                    let v = data[c * width + j - c];
                    (v, v.conj())
                });
                let prior = source[j * width + if upper { k } else { 0 }];
                let value = products.clone().fold(prior, |sum, (a, b)| sum - a * b);
                let diagonal = positive_root(value, prior, products)?;
                data[j * width] = diagonal;
                for i in j + 1..(j + k + 1).min(n) {
                    let prior = if upper {
                        source[i * width + k + j - i].conj()
                    } else {
                        source[j * width + i - j]
                    };
                    let products = (i.saturating_sub(k)..j)
                        .map(|c| (data[c * width + i - c], data[c * width + j - c].conj()));
                    let value = row(prior, diagonal, products)?;
                    data[j * width + i - j] = value;
                }
            }
            Ok(())
        })();
        if batched {
            scope.record_iterations(attempted)?.require()?;
        }
        outcome?;
    }
    Ok(())
}

/// Two-subdiagonal real Cholesky substitutions. Fixed arrays expose the short
/// reductions to the compiler; all exceptional rows use the shared Standard retry.
pub(super) fn solve_real_cholesky_two<S: NumericScalar>(
    data: &[S],
    x: &mut [S],
) -> Result<(), SciOxideError> {
    // bandwidth == 2 guarantees n >= 3 and three stored entries per column.
    let n = x.len();
    let (first, second) = (&data[..3], &data[3..6]);
    x[0] = row(x[0], first[0], core::iter::empty())?;
    x[1] = row(x[1], second[0], [(first[1], x[0])].into_iter())?;
    let (mut older, mut recent) = (x[0], x[1]);
    let (mut far_coefficient, mut near_coefficient, mut next_far) =
        (first[2], second[1], second[2]);
    for (slot, column) in x[2..].iter_mut().zip(data[6..].chunks_exact(3)) {
        let value = row(
            *slot,
            column[0],
            [(far_coefficient, older), (near_coefficient, recent)].into_iter(),
        )?;
        *slot = value;
        older = recent;
        recent = value;
        far_coefficient = next_far;
        near_coefficient = column[1];
        next_far = column[2];
    }
    let (last, preceding) = (
        &data[data.len() - 3..],
        &data[data.len() - 6..data.len() - 3],
    );
    x[n - 1] = row(x[n - 1], last[0], core::iter::empty())?;
    x[n - 2] = row(
        x[n - 2],
        preceding[0],
        [(preceding[1], x[n - 1])].into_iter(),
    )?;
    let (mut next, mut farther) = (x[n - 2], x[n - 1]);
    for (slot, column) in x[..n - 2]
        .iter_mut()
        .rev()
        .zip(data[..data.len() - 6].chunks_exact(3).rev())
    {
        let value = row(
            *slot,
            column[0],
            [(column[1], next), (column[2], farther)].into_iter(),
        )?;
        *slot = value;
        farther = next;
        next = value;
    }
    Ok(())
}

/// Factor `a` into the compact lower candidate `data`. `maximum` bounds the
/// input components.
pub(super) fn factor_cholesky<S: NumericScalar>(
    a: HermitianBandMatrixView<'_, S>,
    maximum: S::Real,
    data: &mut [S],
    scratch: &mut [S],
    gemm: &mut LinalgWorkspace<'_, S>,
    execution: ExecutionContext<'_>,
    scope: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let context = KernelContext::new(execution);
    let (n, k) = (a.order(), a.bandwidth());
    let width = k + 1;
    let (block, rows, _) = geometry(n, k, 0, CHOLESKY_BLOCK);
    if block == 1 {
        return if k == 2 {
            cholesky_columns::<S, 2>(a, data, execution, scope)
        } else {
            cholesky_columns::<S, 0>(a, data, execution, scope)
        };
    }
    for (j, column) in data.chunks_exact_mut(width).enumerate() {
        let mut slots = column.iter_mut();
        slots
            .by_ref()
            .zip(a.lower_column(j))
            .for_each(|(slot, value)| *slot = value);
        slots.for_each(|slot| *slot = S::ZERO);
    }
    let (panel, _) = crate::workspace::split_mut(scratch, rows * block)?;
    let mut bound = TrailingBound::new(maximum);
    for start in (0..n).step_by(block) {
        let b = block.min(n - start);
        let r = rows.min(n - start);
        // Column-major panel: column `j` holds rows `j..r` of band column
        // `start + j`, where row `i` is slot `i - j`.
        let panel = &mut panel[..r * b];
        for (j, values) in panel.chunks_exact_mut(r).enumerate() {
            let end = (j + k + 1).min(r);
            let offset = (start + j) * width;
            values[..j].fill(S::ZERO);
            values[j..end].copy_from_slice(&data[offset..offset + end - j]);
            values[end..].fill(S::ZERO);
        }
        let mut lower_maximum = S::Real::ZERO;
        for j in 0..b {
            admit(execution, scope)?;
            let global_j = start + j;
            let end = (j + k + 1).min(r);
            let (done, rest) = panel.split_at_mut(j * r);
            let (column, later) = rest.split_at_mut(r);
            // Right-looking updates subtract the same products in the same
            // order as a left-looking row fold. A previous block may already
            // have rounded a tiny Schur entry, so exceptional entries are
            // reconstructed from the original matrix and all completed columns.
            let completed = |i: usize, c: usize| {
                if c < start {
                    data[c * width + start + i - c]
                } else {
                    done[(c - start) * r + i]
                }
            };
            let diagonal = positive_root(
                column[j],
                a.element(global_j, global_j),
                (global_j.saturating_sub(k)..global_j).map(|c| {
                    let v = completed(j, c);
                    (v, v.conj())
                }),
            )?;
            column[j] = diagonal;
            for (i, value) in column.iter_mut().enumerate().take(end).skip(j + 1) {
                let residual = *value;
                *value = match standard::ordinary_quotient(residual, diagonal) {
                    Some(quotient) => quotient,
                    None => standard::finish_substitution(
                        a.element(start + i, global_j),
                        diagonal,
                        residual,
                        ((start + i).saturating_sub(k)..global_j)
                            .map(|c| (completed(i, c), completed(j, c).conj())),
                        None,
                    )?,
                };
            }
            if end > b {
                lower_maximum = lower_maximum.max(standard::slice_maximum(&column[b..end]));
            }
            for (c, target) in (j + 1..).zip(later.chunks_exact_mut(r)) {
                let factor = column[c].conj();
                target[c..end]
                    .iter_mut()
                    .zip(&column[c..end])
                    .for_each(|(a, &l)| *a = *a - l * factor);
            }
        }
        if r > b {
            let lower = FiniteMatrixView::from_validated(
                MatrixView::from_slice_with_strides(r - b, b, &panel[b..], 1, r)?,
                lower_maximum,
            );
            kernel::gemm_workspace_with_context(
                WorkspaceGemmProblem {
                    left: lower.adjoint().transpose(),
                    right: lower,
                    output: MatrixViewMut::from_slice_with_strides(
                        r - b,
                        r - b,
                        &mut data[(start + b) * width..],
                        width - 1,
                        1,
                    )?,
                    config: GemmConfig {
                        alpha: -S::ONE,
                        beta: S::ONE,
                        right_op: GemmOp::Transpose,
                        triangle: Some(Triangle::Upper),
                        prior: bound.prior(),
                        ..GemmConfig::ordinary()
                    },
                },
                ExecutionPolicy::Sequential,
                gemm,
                context,
            )?;
            bound = bound.update(b, lower_maximum, lower_maximum);
            for i in b..r {
                let index = (start + i) * width;
                data[index] = S::from_real(data[index].real_component());
            }
        }
        for (j, values) in panel.chunks_exact(r).enumerate() {
            let end = (j + k + 1).min(r);
            let offset = (start + j) * width;
            data[offset..offset + end - j].copy_from_slice(&values[j..end]);
        }
    }
    Ok(())
}

/// Largest magnitude in the stored columns before the factor is copied out:
/// only U for LU, all lower-factor slots for Cholesky. Padding is zero.
pub(super) fn maximum<S: NumericScalar>(
    data: &[S],
    stride: usize,
    retained: usize,
    context: KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    let bound = |values: &[S]| -> Result<S::Real, SciOxideError> {
        if let Some(maximum) = kernel::arch::maximum(values, context.dispatch())? {
            return Ok(maximum);
        }
        values.iter().try_fold(S::Real::ZERO, |maximum, &value| {
            crate::checked::finite_candidate(value)?;
            let magnitude = standard::magnitude(value);
            Ok(if magnitude > maximum {
                magnitude
            } else {
                maximum
            })
        })
    };
    if retained == stride {
        return bound(data);
    }
    data.chunks_exact(stride)
        .try_fold(S::Real::ZERO, |maximum, column| {
            Ok(maximum.max(bound(&column[..retained])?))
        })
}
