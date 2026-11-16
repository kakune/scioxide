//! Blocked lower-triangle tridiagonalization panels; their scratch also serves
//! compact-WY back-transformation.
use super::*;
use crate::dense::{MatrixView, MatrixViewMut};
use crate::kernel::{GemmConfig, GemmOp, Triangle};
use crate::kernel::{level2::gemv_bounded, standard};
use crate::{FiniteMatrixView, FiniteVectorView, VectorView};

pub(super) const WIDTH: usize = 32;
const _: () = assert!(WIDTH >= crate::reflector::BLOCK_WIDTH);
const STRIDE: usize = 2 * WIDTH;

/// Panel scratch lent by the job's direct arena.
pub(super) struct Scratch<'a, S: NumericScalar> {
    panel: ScratchMatrix<'a, S>,
    swapped: ScratchMatrix<'a, S>,
    triangular: ScratchMatrix<'a, S>,
    first: ScratchMatrix<'a, S>,
}

impl<'a, S: NumericScalar> Scratch<'a, S> {
    /// Scalar elements of the panels of an order-`n` job.
    pub(super) fn len(n: usize) -> Result<usize, SciOxideError> {
        [[n, STRIDE], [n, STRIDE], [WIDTH, WIDTH], [WIDTH, n]]
            .into_iter()
            .try_fold(0_usize, |total, shape| {
                scioxide_core::backend::checked_product(
                    &shape,
                    scioxide_core::quantity!("hermitian_panel_scratch"),
                )?
                .checked_add(total)
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!(
                        "hermitian_panel_scratch"
                    ))
                })
            })
    }

    /// Dense products issued by reduction and, for vector jobs, back-transformation.
    pub(super) const fn products(n: usize, vectors: bool) -> [(usize, usize, usize, bool); 4] {
        let back = if vectors { n } else { 0 };
        [
            (n, STRIDE, n, false),
            (WIDTH, back, back, false),
            (WIDTH, WIDTH, back, true),
            (back, WIDTH, back, false),
        ]
    }

    pub(super) fn new(
        scalars: &'a mut [S],
        n: usize,
    ) -> Result<(Self, &'a mut [S]), SciOxideError> {
        let (panel, scalars) = crate::workspace::split_mut(scalars, n * STRIDE)?;
        let (swapped, scalars) = crate::workspace::split_mut(scalars, n * STRIDE)?;
        let (triangular, scalars) = crate::workspace::split_mut(scalars, WIDTH * WIDTH)?;
        let (first, scalars) = crate::workspace::split_mut(scalars, WIDTH * n)?;
        Ok((
            Self {
                panel: ScratchMatrix::new(panel, n, STRIDE)?,
                swapped: ScratchMatrix::new(swapped, n, STRIDE)?,
                triangular: ScratchMatrix::new(triangular, WIDTH, WIDTH)?,
                first: ScratchMatrix::new(first, WIDTH, n)?,
            },
            scalars,
        ))
    }

    /// Lend the panels to compact-WY eigenvector back-transformation.
    pub(super) fn lend<'s, 'g>(
        &'s mut self,
        gemm: &'s mut LinalgWorkspace<'g, S>,
        policy: ExecutionPolicy,
        context: KernelContext<'s>,
    ) -> crate::eigen::hessenberg::Blocks<'s, 'a, 'g, S> {
        crate::eigen::hessenberg::Blocks {
            wy: crate::reflector::CompactWy {
                vectors: &mut self.panel,
                triangular: &mut self.triangular,
                work: &mut self.first,
            },
            gemm,
            execution: crate::kernel::KernelExecution { policy, context },
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn reduce<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    beta: &mut [S],
    vector: &mut [S],
    work: &mut [S],
    scratch: &mut Scratch<'_, S>,
    gemm: &mut LinalgWorkspace<'_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let n = matrix.rows();
    scratch.first.try_resize_flat(WIDTH, n)?;
    for first in (0..beta.len()).step_by(WIDTH) {
        context.execution().check()?;
        let last = (first + WIDTH).min(beta.len());
        let stride = 2 * (last - first);
        scratch.panel.try_resize_flat(n, stride)?;
        scratch.swapped.try_resize_flat(n, stride)?;
        scratch.panel.as_mut_slice().fill(S::ZERO);
        scratch.swapped.as_mut_slice().fill(S::ZERO);
        // Reflectors occupy completed lower columns. The active upper triangle
        // is free: mirror it once per panel so every subsequent A*v streams
        // contiguous rows through the dispatched GEMV. Materializing a column
        // only changes entries outside the next trailing view, preserving this
        // range proof for the whole panel.
        let mut matrix_bound = S::Real::ZERO;
        for row in first..n {
            for col in first..=row {
                let value = crate::checked::finite_candidate(matrix.get_element(row, col))?;
                matrix_bound = matrix_bound.max(standard::magnitude(value));
                if row != col {
                    matrix.as_mut_slice()[col * n + row] = value.conj();
                }
            }
        }
        let mut panel_bound = S::Real::ZERO;
        let mut coefficients = [S::ZERO; STRIDE];
        for (column, tau) in beta.iter_mut().enumerate().take(last).skip(first) {
            let local = column - first;
            // Materialize this panel column after all prior reflectors in the
            // panel. The trailing block remains unmodified until panel commit.
            if local != 0 {
                for (index, value) in coefficients[..2 * local].iter_mut().enumerate() {
                    *value = scratch.swapped.get_element(column, index).conj();
                }
                gemv_bounded(
                    FiniteMatrixView::from_validated(
                        scratch
                            .panel
                            .try_view()?
                            .block(column, 0, n - column, 2 * local)?,
                        panel_bound,
                    ),
                    FiniteVectorView::from_validated(
                        VectorView::from_slice(&coefficients[..2 * local]),
                        panel_bound,
                    ),
                    scratch.first.as_mut_slice(),
                    policy,
                    context,
                )?;
                for (row, &correction) in (column..n).zip(scratch.first.as_slice()) {
                    matrix.as_mut_slice()[row * n + column] =
                        matrix.get_element(row, column) - correction;
                }
            }
            let start = column + 1;
            let len = n - start;
            for (index, value) in vector[..len].iter_mut().enumerate() {
                *value = matrix.get_element(start + index, column);
            }
            let h =
                crate::reflector::generate(crate::VectorViewMut::from_slice(&mut vector[..len]))?;
            *tau = h.tau;
            let coefficient = h.tau;
            let alpha = S::from_real(h.beta);
            let v = FiniteVectorView::from_workspace(VectorView::from_slice(&vector[..len]));
            gemv_bounded(
                FiniteMatrixView::from_validated(
                    matrix.try_view()?.block(start, start, len, len)?,
                    matrix_bound,
                ),
                v,
                &mut work[..len],
                policy,
                context,
            )?;
            if local != 0 {
                gemv_bounded(
                    FiniteMatrixView::from_validated(
                        scratch.panel.try_view()?.block(start, 0, len, 2 * local)?,
                        panel_bound,
                    )
                    .adjoint(),
                    v,
                    &mut coefficients[..2 * local],
                    policy,
                    context,
                )?;
                gemv_bounded(
                    FiniteMatrixView::from_validated(
                        scratch
                            .swapped
                            .try_view()?
                            .block(start, 0, len, 2 * local)?,
                        panel_bound,
                    ),
                    FiniteVectorView::from_workspace(VectorView::from_slice(
                        &coefficients[..2 * local],
                    )),
                    scratch.first.as_mut_slice(),
                    policy,
                    context,
                )?;
                for (value, &correction) in work[..len].iter_mut().zip(scratch.first.as_slice()) {
                    *value = *value - correction;
                }
            }
            work[..len]
                .iter_mut()
                .for_each(|value| *value = *value * coefficient);
            let dot = (0..len).fold(S::ZERO, |sum, index| {
                sum + vector[index].conj() * work[index]
            });
            let correction = dot * (-tau.conj() * S::from_real(S::Real::HALF));
            for index in 0..len {
                let row = start + index;
                let w = crate::checked::finite_candidate(work[index] + correction * vector[index])?;
                panel_bound = panel_bound
                    .max(standard::magnitude(vector[index]))
                    .max(standard::magnitude(w));
                scratch.panel.as_mut_slice()[row * stride + 2 * local] = vector[index];
                scratch.panel.as_mut_slice()[row * stride + 2 * local + 1] = w;
                scratch.swapped.as_mut_slice()[row * stride + 2 * local] = w;
                scratch.swapped.as_mut_slice()[row * stride + 2 * local + 1] = vector[index];
                matrix.as_mut_slice()[row * n + column] =
                    if index == 0 { alpha } else { vector[index] };
            }
        }
        let trailing = n - last;
        // [V W] [W V]^H is one HER2K update with a normwise backward-error bound
        // for A - V W^H - W V^H, with no allocation after reservation.
        let left = MatrixView::from_slice_with_strides(
            trailing,
            stride,
            &scratch.panel.as_slice()[last * stride..],
            stride,
            1,
        )?;
        let right = MatrixView::from_slice_with_strides(
            trailing,
            stride,
            &scratch.swapped.as_slice()[last * stride..],
            stride,
            1,
        )?;
        let output = MatrixViewMut::from_slice_with_strides(
            trailing,
            trailing,
            &mut matrix.as_mut_slice()[last * n + last..],
            n,
            1,
        )?;
        crate::kernel::gemm_workspace_with_context(
            crate::kernel::WorkspaceGemmProblem {
                left: FiniteMatrixView::from_validated(left, panel_bound),
                right: FiniteMatrixView::from_validated(right, panel_bound),
                output,
                config: GemmConfig {
                    alpha: -S::ONE,
                    beta: S::ONE,
                    right_op: GemmOp::Adjoint,
                    triangle: Some(Triangle::Lower),
                    ..GemmConfig::ordinary()
                },
            },
            policy,
            gemm,
            context,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    fn check<S: NumericScalar>() {
        let real = |value| S::Real::try_from_f64_lossy(value).unwrap();
        for n in [64, 65, 66, 96, 97] {
            let matrix = crate::Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|index| {
                        let (row, col) = (index / n, index % n);
                        S::from_components(
                            real(if row == col {
                                4.0 + row as f64 / n as f64
                            } else {
                                ((row + col) % 7) as f64 * 0.01
                            }),
                            real(if S::IS_REAL {
                                0.0
                            } else {
                                (row as f64 - col as f64) * 0.001
                            }),
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let options = HermitianEigenOptions::default();
            let mut vectors = HermitianEigenDecomposition::empty();
            let mut values = HermitianEigenvaluesBuffer::empty();
            let mut workspace = LinalgWorkspace::new();
            for _ in 0..2 {
                HermitianEigenDecomposition::decompose_into(
                    &matrix,
                    crate::HermitianInput::Exact,
                    &options,
                    &mut vectors,
                    &mut workspace,
                    crate::kernel::tests::test_context(),
                )
                .unwrap();
                HermitianEigenvaluesBuffer::compute_into(
                    &matrix,
                    crate::HermitianInput::Exact,
                    &options,
                    &mut values,
                    &mut workspace,
                    crate::kernel::tests::test_context(),
                )
                .unwrap();
                let bound = real(100.0 * n as f64) * <S::Real as RealScalar>::EPSILON;
                for (actual, expected) in values.eigenvalues().iter().zip(vectors.eigenvalues()) {
                    assert!((*actual - *expected).abs() <= bound * real(8.0));
                }
                let q = vectors.eigenvectors();
                for row in 0..n {
                    for col in 0..n {
                        let product = (0..n).fold(S::ZERO, |sum, inner| {
                            sum + matrix.get_element(row, inner) * q.get_element(inner, col)
                        });
                        let residual = product
                            - q.get_element(row, col) * S::from_real(vectors.eigenvalues()[col]);
                        assert!(residual.abs() <= bound * real(8.0), "n={n} residual");
                        let dot = (0..n).fold(S::ZERO, |sum, inner| {
                            sum + q.get_element(inner, row).conj() * q.get_element(inner, col)
                        });
                        let expected = if row == col { S::ONE } else { S::ZERO };
                        assert!((dot - expected).abs() <= bound, "n={n} orthogonality");
                    }
                }
            }
        }
    }

    #[test]
    fn panel_boundaries_and_reused_scratch_cover_real_and_complex_scalars() {
        check::<f32>();
        check::<f64>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
    }
}
