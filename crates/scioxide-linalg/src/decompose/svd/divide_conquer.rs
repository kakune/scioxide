//! Bidiagonal divide and conquer by central-row splitting (Gu & Eisenstat 1995).
//!
//! Rectangular children keep their right null vector. A merge is the SVD of
//! a diagonal matrix with its zero row replaced by the coupling row. Secular
//! vectors are formed on both sides; no Gram matrix or B*v/sigma is used.
//! All recursion uses slices of prepared output and reusable merge scratch.

use super::bidiagonal_qr::bidiagonal_qr;
use super::typed::set_identity;
use crate::kernel::{
    GemmConfig, LinalgWorkspace, WorkspaceGemmProblem, gemm_workspace_with_context,
};
use crate::rotation::Rotation;
use crate::secular::breakdown;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::{KernelContext, MatrixView, MatrixViewMut};
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, ExecutionPolicy, RealScalar, SciOxideError,
};

/// QR leaf order inside the divide-and-conquer recursion. The top-level
/// driver separately avoids recursion for small cores through order 64.
pub(super) const LEAF: usize = 32;

pub(super) struct Workspace<'a, R: RealScalar> {
    pub(super) u: ScratchMatrix<'a, R>,
    pub(super) vt: ScratchMatrix<'a, R>,
    left: ScratchMatrix<'a, R>,
    right: ScratchMatrix<'a, R>,
    left_coefficients: ScratchMatrix<'a, R>,
    right_coefficients: ScratchMatrix<'a, R>,
    leaf_u: ScratchMatrix<'a, R>,
    leaf_vt: ScratchMatrix<'a, R>,
    poles: &'a mut [R],
    weights: &'a mut [R],
    origins: &'a mut [R],
    shifts: &'a mut [R],
    active: &'a mut [usize],
    gemm: LinalgWorkspace<'a, R>,
}

#[derive(Clone, Copy)]
pub(super) struct Layout {
    pub(super) reals: usize,
    pub(super) indices: usize,
    gemm: crate::LinalgRequirements,
}
fn shapes(n: usize) -> [[usize; 2]; 8] {
    [
        [n, n],
        [n, n],
        [n, n],
        [n, n],
        [n, n],
        [n, n],
        [LEAF, LEAF],
        [LEAF + 1, LEAF + 1],
    ]
}
pub(super) fn layout<R: RealScalar>(
    n: usize,
    options: &crate::LinalgExecutionOptions,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
) -> Result<Layout, SciOxideError> {
    let mut request =
        crate::DenseWorkspaceRequest::product(n, n, n, options).with_worker_bound(workers);
    if matches!(packing, crate::workspace::Packing::Provider) {
        request = request.with_provider_packing();
    }
    let gemm = LinalgWorkspace::<R>::workspace_requirements(&request)?;
    let size = |shape: &[usize]| {
        scioxide_core::backend::checked_product(
            shape,
            scioxide_core::quantity!("svd_secular_scratch"),
        )
    };
    let add = |a: usize, b| {
        a.checked_add(b).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("svd_secular_scratch"))
        })
    };
    let reals = shapes(n)
        .into_iter()
        .try_fold(add(size(&[4, n])?, gemm.scalar_len()?)?, |sum, shape| {
            add(sum, size(&shape)?)
        })?;
    Ok(Layout {
        reals,
        indices: n,
        gemm,
    })
}

impl<'a, R: RealScalar> Workspace<'a, R> {
    pub(super) fn new(
        mut reals: &'a mut [R],
        indices: &'a mut [usize],
        n: usize,
        layout: Layout,
    ) -> Result<Self, SciOxideError> {
        let mut take = |length| crate::workspace::take_mut(&mut reals, length);
        let [
            u,
            vt,
            left,
            right,
            left_coefficients,
            right_coefficients,
            leaf_u,
            leaf_vt,
        ] = shapes(n).map(|[rows, cols]| ScratchMatrix::new(take(rows * cols)?, rows, cols));
        let poles = take(n)?;
        let weights = take(n)?;
        let origins = take(n)?;
        let shifts = take(n)?;
        let storage = take(layout.gemm.scalar_len()?)?;
        let gemm = LinalgWorkspace::from_slices(
            crate::LinalgSlices {
                scalars: storage,
                ..crate::LinalgSlices::default()
            },
            &layout.gemm,
        )?;
        Ok(Self {
            u: u?,
            vt: vt?,
            left: left?,
            right: right?,
            left_coefficients: left_coefficients?,
            right_coefficients: right_coefficients?,
            leaf_u: leaf_u?,
            leaf_vt: leaf_vt?,
            poles,
            weights,
            origins,
            shifts,
            active: &mut indices[..n],
            gemm,
        })
    }

    pub(super) fn solve<'capability>(
        &mut self,
        diagonal: &mut [R],
        off: &mut [R],
        budget: usize,
        policy: ExecutionPolicy,
        context: KernelContext<'capability>,
        solve: &mut scioxide_core::SolveContext<'_, 'capability>,
    ) -> Result<(), SciOxideError> {
        set_identity(&mut self.u);
        set_identity(&mut self.vt);
        self.range(
            diagonal,
            off,
            0,
            false,
            &mut Iteration {
                solve,
                budget,
                policy,
                context,
            },
        )
    }

    fn range(
        &mut self,
        diagonal: &mut [R],
        off: &mut [R],
        offset: usize,
        extra: bool,
        iteration: &mut Iteration<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        iteration.context.execution().check()?;
        let n = diagonal.len();
        if n <= LEAF {
            return self.leaf(diagonal, off, offset, extra, iteration);
        }
        let middle = n / 2;
        let alpha = diagonal[middle];
        let beta = off[middle];
        let (left, right) = crate::workspace::split_mut(diagonal, middle)?;
        let (left_off, right_off) = crate::workspace::split_mut(off, middle)?;
        self.range(left, left_off, offset, true, iteration)?;
        self.range(
            &mut right[1..],
            &mut right_off[1..],
            offset + middle + 1,
            extra,
            iteration,
        )?;
        self.merge(diagonal, offset, extra, alpha, beta, iteration)?;
        off.fill(R::ZERO);
        Ok(())
    }

    fn leaf(
        &mut self,
        diagonal: &mut [R],
        off: &mut [R],
        offset: usize,
        extra: bool,
        iteration: &mut Iteration<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = diagonal.len();
        let m = n + usize::from(extra);
        self.leaf_u.try_resize_flat(n, n)?;
        self.leaf_vt.try_resize_flat(m, m)?;
        set_identity(&mut self.leaf_u);
        set_identity(&mut self.leaf_vt);
        if extra {
            // Right rotations eliminate the extra column, leaving a square
            // lower bidiagonal; left rotations restore upper orientation.
            for i in 0..n {
                let (rotation, radius) = crate::rotation::Rotation::generate(diagonal[i], off[i])?;
                diagonal[i] = radius;
                if i + 1 < n {
                    off[i] = rotation.s * diagonal[i + 1];
                    diagonal[i + 1] = rotation.c * diagonal[i + 1];
                } else {
                    off[i] = R::ZERO;
                }
                rotation.rotate_rows(self.leaf_vt.as_mut_slice(), m, i, i + 1, 0..m)?;
            }
            for i in 0..n.saturating_sub(1) {
                let (rotation, radius) = crate::rotation::Rotation::generate(diagonal[i], off[i])?;
                diagonal[i] = radius;
                off[i] = rotation.s * diagonal[i + 1];
                diagonal[i + 1] = rotation.c * diagonal[i + 1];
                rotation.rotate_columns(self.leaf_u.as_mut_slice(), n, i, i + 1, 0..n);
            }
        }
        // Leaves share the merge allowance: each receives only what earlier leaves left.
        let remaining = iteration
            .budget
            .saturating_sub(iteration.solve.counts().own().iterations());
        bidiagonal_qr(
            diagonal,
            off,
            &mut self.leaf_u,
            &mut self.leaf_vt,
            remaining,
            iteration.solve,
        )?;
        let stride = self.u.cols();
        for i in 0..n {
            self.u.as_mut_slice()
                [(offset + i) * stride + offset..(offset + i) * stride + offset + n]
                .copy_from_slice(&self.leaf_u.as_slice()[i * n..(i + 1) * n]);
        }
        for i in 0..m {
            self.vt.as_mut_slice()
                [(offset + i) * stride + offset..(offset + i) * stride + offset + m]
                .copy_from_slice(&self.leaf_vt.as_slice()[i * m..(i + 1) * m]);
        }
        Ok(())
    }

    fn merge(
        &mut self,
        diagonal: &mut [R],
        offset: usize,
        extra: bool,
        alpha: R,
        beta: R,
        iteration: &mut Iteration<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = diagonal.len();
        let m = n + usize::from(extra);
        let middle = n / 2;

        self.left.try_resize_flat(n, n)?;
        self.right.try_resize_flat(m, m)?;
        // The central zero pole is always first; remaining modes are sorted.
        for j in 0..n {
            let source = if j == 0 {
                middle
            } else if j <= middle {
                j - 1
            } else {
                j
            };
            self.poles[j] = if j == 0 { R::ZERO } else { diagonal[source] };
            self.weights[j] = if source <= middle {
                alpha * self.vt.get_element(offset + source, offset + middle)
            } else {
                beta * self.vt.get_element(offset + source, offset + middle + 1)
            };
            for i in 0..n {
                self.left.as_mut_slice()[i * n + j] = if j == 0 {
                    if i == middle { R::ONE } else { R::ZERO }
                } else {
                    self.u.get_element(offset + i, offset + source)
                };
            }
            for i in 0..m {
                self.right.as_mut_slice()[j * m + i] =
                    self.vt.get_element(offset + source, offset + i);
            }
        }
        if extra {
            for i in 0..m {
                self.right.as_mut_slice()[n * m + i] = self.vt.get_element(offset + n, offset + i);
            }
            let tail = beta * self.vt.get_element(offset + n, offset + middle + 1);
            let radius = self.weights[0].hypot(tail);
            if radius != R::ZERO {
                Rotation {
                    c: self.weights[0] / radius,
                    s: tail / radius,
                }
                .rotate_rows(self.right.as_mut_slice(), m, 0, n, 0..m)?;
                self.weights[0] = radius;
            }
        }
        for i in 1..n {
            let next = (i..n)
                .min_by(|&a, &b| {
                    self.poles[a]
                        .partial_cmp(&self.poles[b])
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .unwrap_or(i);
            self.poles.swap(i, next);
            self.weights.swap(i, next);
            crate::reflector::swap_columns(self.left.try_view_mut()?, i, next);
            crate::reflector::swap_columns(self.right.try_view_mut()?.transpose(), i, next);
        }
        // Normalize each merge in binary coordinates, including graded children.
        let maximum = self.poles[..n]
            .iter()
            .chain(&self.weights[..n])
            .fold(R::ZERO, |a, b| a.max(b.abs()));
        let exponent = crate::kernel::standard::exponent(maximum)?;
        for x in self.poles[..n].iter_mut().chain(&mut self.weights[..n]) {
            *x = x.scalbn(-exponent);
        }
        let scale = self.poles[..n]
            .iter()
            .chain(&self.weights[..n])
            .fold(R::ZERO, |a, b| a.max(b.abs()));
        let threshold = scale * R::EPSILON * R::try_from_f64_lossy(8.0)?;
        // Zero child modes couple only on the right. Rotating the central left
        // vector here would mix the replaced row with a genuine left null mode.
        for i in 1..n {
            if self.poles[i] <= threshold && self.weights[i].abs() > threshold {
                self.poles[i] = R::ZERO;
                let radius = self.weights[0].hypot(self.weights[i]);
                if radius != R::ZERO {
                    Rotation {
                        c: self.weights[0] / radius,
                        s: self.weights[i] / radius,
                    }
                    .rotate_rows(self.right.as_mut_slice(), m, 0, i, 0..m)?;
                    self.weights[0] = radius;
                }
                self.weights[i] = R::ZERO;
            }
        }
        let isolated = self.weights[1..n].iter().all(|z| z.abs() <= threshold);
        let central = self.weights[0] != R::ZERO && (isolated || self.weights[0].abs() > threshold);
        if !central {
            self.weights[0] = R::ZERO;
        }
        let begin = usize::from(central);
        self.active[0] = 0;
        let count = crate::secular::deflate(
            &self.poles[1..n],
            &mut self.weights[1..n],
            &mut self.active[begin..n],
            R::ONE,
            threshold,
            |a, b, c, s| {
                Rotation { c, s }.rotate_columns(self.left.as_mut_slice(), n, a + 1, b + 1, 0..n);
                Rotation { c, s }.rotate_rows(self.right.as_mut_slice(), m, a + 1, b + 1, 0..m)?;
                Ok(())
            },
        )?;
        for index in &mut self.active[begin..begin + count] {
            *index += 1;
        }
        let count = count + begin;
        for root in 0..count {
            let (origin, shift) = if count == 1 && central {
                // An isolated coupling is a scalar SVD, including subnormals.
                (R::ZERO, self.weights[0].abs())
            } else {
                crate::secular::singular_root(
                    &self.poles[..n],
                    &self.weights[..n],
                    &self.active[..count],
                    root,
                    || iteration.admit(),
                )?
            };
            self.origins[root] = origin;
            self.shifts[root] = shift;
        }
        self.correct_weights(count)?;
        // Deflated modes bypass GEMM. The central zero mode has a nontrivial
        // left null vector when the coupling row contains other components.
        let stride = self.u.cols();
        let mut destination = count;
        for i in 0..n {
            if self.weights[i] != R::ZERO {
                continue;
            }
            diagonal[destination] = self.poles[i].scalbn(exponent);
            for row in 0..n {
                self.u.as_mut_slice()[(offset + row) * stride + offset + destination] =
                    self.left.get_element(row, i);
            }
            if i == 0 && count > 0 {
                let norm = self.active[..count].iter().fold(R::ONE, |norm, &j| {
                    norm.hypot(self.weights[j] / self.poles[j])
                });
                for row in 0..n {
                    let value = self.active[..count].iter().fold(
                        self.left.get_element(row, 0) / norm,
                        |value, &j| {
                            value
                                - self.left.get_element(row, j)
                                    * (self.weights[j] / self.poles[j] / norm)
                        },
                    );
                    self.u.as_mut_slice()[(offset + row) * stride + offset + destination] = value;
                }
            }
            for col in 0..m {
                self.vt.as_mut_slice()[(offset + destination) * stride + offset + col] =
                    self.right.get_element(i, col);
            }
            destination += 1;
        }
        if extra {
            for col in 0..m {
                self.vt.as_mut_slice()[(offset + n) * stride + offset + col] =
                    self.right.get_element(n, col);
            }
        }
        if count == 0 {
            return Ok(());
        }
        let left_count = count + usize::from(!central);
        self.left_coefficients.try_resize_flat(left_count, count)?;
        self.right_coefficients.try_resize_flat(count, count)?;
        for (root, value) in diagonal.iter_mut().enumerate().take(count) {
            *value = (self.origins[root] + self.shifts[root]).scalbn(exponent);
            if count == 1 && central {
                self.left_coefficients.as_mut_slice()[0] = R::ONE;
                self.right_coefficients.as_mut_slice()[0] = self.weights[0].signum();
                continue;
            }
            let mut right_norm = R::ZERO;
            let mut left_norm = R::ONE;
            self.left_coefficients.as_mut_slice()[root] = -R::ONE;
            for (j, &pole) in self.active[..count].iter().enumerate() {
                let minus = (self.poles[pole] - self.origins[root]) - self.shifts[root];
                let plus = (self.poles[pole] + self.origins[root]) + self.shifts[root];
                let value = self.weights[pole] / minus / plus;
                self.right_coefficients.as_mut_slice()[root * count + j] = value;
                right_norm = right_norm.hypot(value);
                if pole != 0 {
                    let component = self.poles[pole] * value;
                    self.left_coefficients.as_mut_slice()
                        [(j + usize::from(!central)) * count + root] = component;
                    left_norm = left_norm.hypot(component);
                }
            }
            if !right_norm.is_finite() || !left_norm.is_finite() || right_norm == R::ZERO {
                return Err(breakdown());
            }
            for j in 0..count {
                self.right_coefficients.as_mut_slice()[root * count + j] =
                    self.right_coefficients.as_slice()[root * count + j] / right_norm;
            }
            for j in 0..left_count {
                self.left_coefficients.as_mut_slice()[j * count + root] =
                    self.left_coefficients.as_slice()[j * count + root] / left_norm;
            }
        }
        // Pack only surviving modes. Forward compaction cannot overwrite an
        // unread source: indices are ascending and each source follows its target.
        for row in 0..n {
            if !central {
                self.left.as_mut_slice()[row * left_count] = self.left.as_slice()[row * n];
            }
            for (j, &source) in self.active[..count].iter().enumerate() {
                self.left.as_mut_slice()[row * left_count + j + usize::from(!central)] =
                    self.left.as_slice()[row * n + source];
            }
        }
        for (j, &source) in self.active[..count].iter().enumerate() {
            self.right
                .as_mut_slice()
                .copy_within(source * m..(source + 1) * m, j * m);
        }
        gemm_workspace_with_context(
            WorkspaceGemmProblem {
                left: crate::FiniteMatrixView::from_workspace(MatrixView::from_slice(
                    n,
                    left_count,
                    &self.left.as_slice()[..n * left_count],
                )?),
                right: crate::FiniteMatrixView::from_workspace(self.left_coefficients.try_view()?),
                output: MatrixViewMut::from_slice_with_strides(
                    n,
                    count,
                    &mut self.u.as_mut_slice()[offset * stride + offset..],
                    stride,
                    1,
                )?,
                config: GemmConfig::ordinary(),
            },
            iteration.policy,
            &mut self.gemm,
            iteration.context,
        )?;
        gemm_workspace_with_context(
            WorkspaceGemmProblem {
                left: crate::FiniteMatrixView::from_workspace(self.right_coefficients.try_view()?),
                right: crate::FiniteMatrixView::from_workspace(MatrixView::from_slice(
                    count,
                    m,
                    &self.right.as_slice()[..count * m],
                )?),
                output: MatrixViewMut::from_slice_with_strides(
                    count,
                    m,
                    &mut self.vt.as_mut_slice()[offset * stride + offset..],
                    stride,
                    1,
                )?,
                config: GemmConfig::ordinary(),
            },
            iteration.policy,
            &mut self.gemm,
            iteration.context,
        )
    }

    fn correct_weights(&mut self, count: usize) -> Result<(), SciOxideError> {
        for &i in &self.active[..count] {
            let pole = self.poles[i];
            let magnitude = crate::secular::weight(
                (0..count).flat_map(|root| {
                    [
                        (self.origins[root] - pole) + self.shifts[root],
                        (self.origins[root] + pole) + self.shifts[root],
                    ]
                }),
                self.active[..count]
                    .iter()
                    .filter(|&&j| j != i)
                    .flat_map(|&j| [pole - self.poles[j], pole + self.poles[j]]),
            )?;
            self.weights[i] = magnitude.copysign(self.weights[i]);
        }
        Ok(())
    }
}

struct Iteration<'scope, 'state, 'capability> {
    solve: &'scope mut scioxide_core::SolveContext<'state, 'capability>,
    budget: usize,
    policy: ExecutionPolicy,
    context: KernelContext<'capability>,
}
impl Iteration<'_, '_, '_> {
    fn admit(&mut self) -> Result<(), SciOxideError> {
        self.solve.execution().check()?;
        if self.solve.counts().own().iterations() >= self.budget {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::IterationLimit,
                ConvergenceDiagnostics::new().with_counts(self.solve.counts()),
                "SVD divide-and-conquer exhausted its shared iteration budget",
            ));
        }
        crate::solve_support::admit_iteration(
            Some(self.solve.execution()),
            Some(self.solve),
            "no complete SVD candidate",
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use crate::Matrix;
    use alloc::{format, vec::Vec};

    use super::*;

    fn compare<R: RealScalar>(n: usize, fixture: usize) -> usize {
        let real = |x| R::try_from_f64_lossy(x).unwrap();
        let diagonal: Vec<R> = (0..n)
            .map(|i| match fixture {
                0 => real(1.0 + (i % 3) as f64 * 32.0) * R::EPSILON + R::ONE,
                1 => R::ONE.scalbn(-(i as i32)),
                2 if i == n / 2 - 1 || i % 13 == 0 => R::ZERO,
                3 => R::ZERO,
                _ => real(1.0 + ((i * 17 + 3) % 31) as f64 / 31.0),
            })
            .collect();
        let off: Vec<R> = (0..n)
            .map(|i| {
                if i + 1 == n || fixture == 3 || (fixture == 4 && i % 7 == 0) {
                    R::ZERO
                } else if fixture == 0 {
                    real(64.0) * R::EPSILON
                } else if fixture == 1 {
                    diagonal[i] * real(-0.125)
                } else {
                    real(0.25)
                }
            })
            .collect();
        check(&diagonal, &off, &[], &format!("fixture {fixture}"))
    }

    fn check<R: RealScalar>(diagonal: &[R], off: &[R], reference: &[R], label: &str) -> usize {
        let n = diagonal.len();
        let real = |x| R::try_from_f64_lossy(x).unwrap();
        let mut qr = diagonal.to_vec();
        let mut qr_off = off.to_vec();
        let mut qr_u_data = Matrix::<R>::identity(n).unwrap();
        let mut qr_vt_data = Matrix::<R>::identity(n).unwrap();
        let mut qr_u = ScratchMatrix::new(qr_u_data.as_mut_slice(), n, n).unwrap();
        let mut qr_vt = ScratchMatrix::new(qr_vt_data.as_mut_slice(), n, n).unwrap();
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "bidiagonal_svd", 1),
            n as u128,
            0,
        );
        let mut qr_state = scioxide_core::SolveState::new();
        let mut qr_budgets = scioxide_core::SolveBudgets::unbounded();
        let mut qr_scope = qr_state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                identity,
                &mut qr_budgets,
            )
            .unwrap();
        bidiagonal_qr(
            &mut qr,
            &mut qr_off,
            &mut qr_u,
            &mut qr_vt,
            1000 * n,
            &mut qr_scope,
        )
        .unwrap();
        let mut dc = diagonal.to_vec();
        let mut dc_off = off.to_vec();
        let layout = layout::<R>(
            n,
            &crate::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            core::num::NonZeroUsize::MIN,
            crate::workspace::Packing::Native,
        )
        .unwrap();
        let mut reals = alloc::vec![R::ZERO;layout.reals];
        let mut indices = alloc::vec![0;layout.indices];
        let mut workspace = Workspace::new(&mut reals, &mut indices, n, layout).unwrap();
        let mut dc_state = scioxide_core::SolveState::new();
        let mut dc_budgets = scioxide_core::SolveBudgets::unbounded();
        let mut dc_scope = dc_state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                identity,
                &mut dc_budgets,
            )
            .unwrap();
        workspace
            .solve(
                &mut dc,
                &mut dc_off,
                1000 * n,
                ExecutionPolicy::Sequential,
                KernelContext::sequential(),
                &mut dc_scope,
            )
            .unwrap();
        let bound = real(512.0 * n as f64) * R::EPSILON;
        let scale = diagonal
            .iter()
            .chain(off)
            .fold(R::ONE, |a, b| a.max(b.abs()));
        for (u, vt, values) in [(&workspace.u, &workspace.vt, &dc), (&qr_u, &qr_vt, &qr)] {
            let mut error = R::ZERO;
            let mut left_error = R::ZERO;
            let mut right_error = R::ZERO;
            for row in 0..n {
                for col in 0..n {
                    let actual = (0..n).fold(R::ZERO, |sum, k| {
                        sum + u.get_element(row, k) * values[k] * vt.get_element(k, col)
                    });
                    let expected = if row == col {
                        diagonal[row]
                    } else if col == row + 1 {
                        off[row]
                    } else {
                        R::ZERO
                    };
                    error = error.hypot((actual - expected) / scale);
                    let identity = if row == col { R::ONE } else { R::ZERO };
                    let l = (0..n).fold(R::ZERO, |sum, k| {
                        sum + u.get_element(k, row) * u.get_element(k, col)
                    });
                    let r = (0..n).fold(R::ZERO, |sum, k| {
                        sum + vt.get_element(row, k) * vt.get_element(col, k)
                    });
                    left_error = left_error.hypot(l - identity);
                    right_error = right_error.hypot(r - identity);
                }
            }
            assert!(error <= bound, "n={n} {label} reconstruction={error:?}");
            assert!(left_error <= bound, "n={n} {label} left={left_error:?}");
            assert!(right_error <= bound, "n={n} {label} right={right_error:?}");
        }
        qr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            qr.iter()
                .zip(&dc)
                .all(|(a, b)| (*a - *b).abs() <= bound * scale),
            "n={n} {label} spectrum"
        );
        if !reference.is_empty() {
            // Weyl's bound: absolute singular-value error is bounded by the
            // normwise backward perturbation, including at multiple/zero values.
            for values in [&qr, &dc] {
                for (&actual, &expected) in values.iter().rev().zip(reference) {
                    assert!(
                        (actual - expected).abs() <= bound * scale,
                        "{label}: {actual:?} != {expected:?}"
                    );
                }
            }
        }
        dc_scope.counts().own().iterations()
    }

    #[test]
    fn independent_vector_path_references() {
        fn fixtures<R: RealScalar>() {
            for case in crate::bidiagonal_fixtures::cases::<R>() {
                let mut off = case.off_diagonal;
                off.push(R::ZERO);
                check(&case.diagonal, &off, &case.expected, case.name);
            }
        }
        fixtures::<f32>();
        fixtures::<f64>();
    }

    #[test]
    fn rational_svd_reduces_iteration_count() {
        let iterations = compare::<f64>(300, 5);
        assert!(
            iterations < 23_519,
            "rational iterations={iterations}; Newton baseline=23519"
        );
    }

    #[test]
    fn divide_conquer_agrees_with_qr_on_clustered_graded_and_deficient_bidiagonals() {
        for n in [31, 32, 33, 64, 65, 129] {
            for fixture in 0..5 {
                compare::<f64>(n, fixture);
                compare::<f32>(n, fixture);
            }
        }
    }
}
