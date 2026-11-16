//! Householder QR factorization `A[:, p] = Q R`, optionally column-pivoted.
//!
//! `Q` is `m × m` unitary, `R` is `m × n` upper trapezoidal and `p` is the
//! column permutation. The reflectors are stored in packed form: the
//! subdiagonal entries of column `k` hold `v[k+1:]` (with implicit `v[0] = 1`),
//! and the `tau` factors are stored separately. A column-packed copy of the
//! reflector vectors is kept for repeated vector solves. Reflector panels are
//! applied in compact WY form with GEMM and TRMM. Works for tall, square, wide
//! and empty matrices. O(m n min(m, n)) time, O(m n) memory.

use crate::workspace::{RealRegion, ScalarRegion};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;

use scioxide_core::{ExecutionPolicy, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};

use crate::KernelContext;
use crate::dense::Matrix;
use crate::kernel::{GemmOp, LinalgWorkspace};

use super::util::{copy_block, matrix_block_mut, reserve_vec, transfer_panel};
use crate::analysis::{RankPolicy, RankReport, analyze_magnitudes};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};

/// Geometry for a private packed-QR kernel invocation.
#[derive(Clone, Copy)]
pub(super) enum QrRequest {
    Factor {
        rows: usize,
        cols: usize,
    },
    Explicit {
        rows: usize,
        cols: usize,
    },
    ApplyMatrix {
        rows: usize,
        cols: usize,
        rhs: usize,
    },
    SolveMatrix {
        rows: usize,
        cols: usize,
        rhs: usize,
    },
    Vector {
        rows: usize,
    },
}

impl QrRequest {
    fn shapes(self) -> [[usize; 2]; 7] {
        match self {
            Self::Factor { rows, cols } | Self::Explicit { rows, cols } => {
                let columns = if matches!(self, Self::Explicit { .. }) {
                    rows.max(cols)
                } else {
                    cols
                };
                let block = rows.min(cols).min(crate::reflector::BLOCK_WIDTH);
                [
                    [rows, cols],
                    [rows, block],
                    [block, block],
                    [block, columns],
                    [block, cols],
                    [0, 0],
                    [rows, block],
                ]
            }
            Self::ApplyMatrix { rows, cols, rhs } | Self::SolveMatrix { rows, cols, rhs } => {
                let block = rows.min(cols).min(crate::reflector::BLOCK_WIDTH);
                [
                    [0, 0],
                    [rows, block],
                    [block, block],
                    [block, rhs],
                    [0, 0],
                    [rows, rhs],
                    [0, 0],
                ]
            }
            Self::Vector { .. } => [[0, 0]; 7],
        }
    }
    fn lengths(self) -> (usize, usize, usize) {
        match self {
            Self::Factor { rows, cols } | Self::Explicit { rows, cols } => {
                (rows.min(cols), cols, 0)
            }
            Self::ApplyMatrix { .. } | Self::SolveMatrix { .. } => (0, 0, 0),
            Self::Vector { rows } => (0, 0, rows),
        }
    }
}

pub(super) fn requirements<S: NumericScalar>(
    request: QrRequest,
    options: &crate::LinalgExecutionOptions,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
) -> Result<crate::LinalgRequirements, SciOxideError> {
    let mut result = crate::LinalgRequirements::new(options.byte_limit());
    let shapes = request.shapes();
    let (tau, cols, vector) = request.lengths();
    let add = |left: usize, right| {
        left.checked_add(right)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("qr_scratch")))
    };
    let size = |shape: &[usize]| {
        scioxide_core::backend::checked_product(shape, scioxide_core::quantity!("qr_scratch"))
    };
    *result.scalar_mut(ScalarRegion::Direct) = shapes
        .into_iter()
        .try_fold(add(tau, vector)?, |total, shape| add(total, size(&shape)?))?;
    result.indices = size(&[cols, 2])?;
    *result.real_mut(RealRegion::Direct) = size(&[cols, 2])?;
    if !matches!(request, QrRequest::Vector { .. }) {
        let rows = shapes[1][0];
        let block = shapes[1][1];
        let columns = shapes[3][1];
        for (m, k, n, structured) in [
            (block, rows, columns, false),
            (rows, block, columns, false),
            (block, block, columns, true),
        ] {
            let dense = if structured {
                crate::DenseWorkspaceRequest::structured(m, k, n, options)
            } else {
                crate::DenseWorkspaceRequest::product(m, k, n, options)
            };
            let dense = dense
                .with_product_accuracy(scioxide_core::Accuracy::Standard)
                .with_worker_bound(workers);
            let dense = if matches!(packing, crate::workspace::Packing::Provider) {
                dense.with_provider_packing()
            } else {
                dense
            };
            result = result.max(&crate::kernel::product_requirements::<S>(&dense)?);
        }
    }
    if let QrRequest::SolveMatrix { cols, rhs, .. } = request {
        // Applying Q uses only WY panels. The subsequent triangular solve also
        // needs the complete triangular operand when an injected BLAS packs it.
        let dense = crate::DenseWorkspaceRequest::triangular_solve(cols, cols, rhs, options)
            .with_product_accuracy(scioxide_core::Accuracy::Standard)
            .with_worker_bound(workers);
        let dense = if matches!(packing, crate::workspace::Packing::Provider) {
            dense.with_provider_packing()
        } else {
            dense
        };
        result = result.max(&crate::kernel::product_requirements::<S>(&dense)?);
    }
    result.required_bytes::<S>()?;
    Ok(result)
}

pub(super) fn reserve<S: NumericScalar>(
    workspace: &mut LinalgWorkspace<'_, S>,
    request: QrRequest,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let packing = crate::workspace::Packing::Native;
    #[cfg(feature = "blas")]
    let packing = if context.blas().is_some() {
        crate::workspace::Packing::Provider
    } else {
        packing
    };
    workspace.try_reserve(&requirements::<S>(
        request,
        &crate::LinalgExecutionOptions::new().with_execution_policy(policy),
        context.execution().worker_budget(),
        packing,
    )?)
}

struct QrScratch<'a, 'storage, S: NumericScalar> {
    candidate: ScratchMatrix<'a, S>,
    tau: &'a mut [S],
    permutation: &'a mut [usize],
    v_block: ScratchMatrix<'a, S>,
    t_block: ScratchMatrix<'a, S>,
    work_block: ScratchMatrix<'a, S>,
    panel_update: ScratchMatrix<'a, S>,
    norms: ColumnNorms<'a, S::Real>,
    solve_buffer: ScratchMatrix<'a, S>,
    /// Contiguous row-major copy of the factorization panel and the rows below it.
    panel: ScratchMatrix<'a, S>,
    vector_candidate: &'a mut [S],
    gemm: &'a mut LinalgWorkspace<'storage, S>,
}

fn with_scratch<S: NumericScalar, T>(
    workspace: &mut LinalgWorkspace<'_, S>,
    request: QrRequest,
    work: impl FnOnce(&mut QrScratch<'_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.with_direct(|mut scalars, reals, indices, gemm| {
        let shapes = request.shapes();
        let [
            candidate,
            v_block,
            t_block,
            work_block,
            panel_update,
            solve_buffer,
            panel,
        ] = shapes.map(|[rows, cols]| {
            let (values, rest) =
                crate::workspace::split_mut(core::mem::take(&mut scalars), rows * cols)?;
            scalars = rest;
            ScratchMatrix::new(values, rows, cols)
        });
        let (candidate, v_block, t_block, work_block, panel_update, solve_buffer, panel) = (
            candidate?,
            v_block?,
            t_block?,
            work_block?,
            panel_update?,
            solve_buffer?,
            panel?,
        );
        let (tau_len, cols, vector_len) = request.lengths();
        let (tau, vector) = crate::workspace::split_mut(scalars, tau_len)?;
        let (permutation, tags) = crate::workspace::split_mut(indices, cols)?;
        let mut scratch = QrScratch {
            candidate,
            tau,
            permutation,
            v_block,
            t_block,
            work_block,
            panel_update,
            norms: ColumnNorms {
                values: &mut reals[..2 * cols],
                tags: &mut tags[..cols],
            },
            solve_buffer,
            panel,
            vector_candidate: &mut vector[..vector_len],
            gemm,
        };
        work(&mut scratch)
    })
}

fn factor_candidate<S: NumericScalar>(
    matrix: crate::FiniteMatrixView<'_, S>,
    workspace: &mut QrScratch<'_, '_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
    pivoting: crate::HouseholderPivoting,
) -> Result<(S::Real, scioxide_core::Scaled<S::Real>), SciOxideError> {
    let (rows, cols) = (matrix.rows(), matrix.cols());
    workspace.candidate.try_resize_flat(rows, cols)?;
    copy_block(matrix.as_view(), &mut workspace.candidate)?;
    workspace
        .permutation
        .iter_mut()
        .enumerate()
        .for_each(|(column, slot)| *slot = column);

    let one_norm = match pivoting {
        crate::HouseholderPivoting::None => {
            let norm = crate::diagnostics::matrix_one_norm(workspace.candidate.try_view()?)?;
            factor_unpivoted(
                UnpivotedQr {
                    packed: &mut workspace.candidate,
                    tau: workspace.tau,
                    panel: &mut workspace.panel,
                    blocks: crate::reflector::CompactWy {
                        vectors: &mut workspace.v_block,
                        triangular: &mut workspace.t_block,
                        work: &mut workspace.work_block,
                    },
                },
                matrix.maximum,
                policy,
                workspace.gemm,
                context,
            )?;
            norm
        }
        crate::HouseholderPivoting::Column => factor_pivoted(workspace, policy, context)?,
    };
    let maximum = validate_factor_candidate(&workspace.candidate, workspace.tau)?;
    context.execution().check()?;
    Ok((maximum, one_norm))
}

/// Form explicit factors in scratch reserved for QrRequest::Explicit by the
/// owning RQ or QR-update operation, without repeating its preparation.
pub(super) fn factor_explicit<S: NumericScalar>(
    matrix: crate::FiniteMatrixView<'_, S>,
    pivoting: crate::HouseholderPivoting,
    mut q: crate::MatrixViewMut<'_, S>,
    mut r: crate::MatrixViewMut<'_, S>,
    permutation: Option<&mut [usize]>,
    workspace: &mut LinalgWorkspace<'_, S>,
    execution: crate::kernel::KernelExecution<KernelContext<'_>>,
) -> Result<(), SciOxideError> {
    let crate::kernel::KernelExecution { policy, context } = execution;

    let (rows, cols) = (matrix.rows(), matrix.cols());
    if q.rows() != rows || q.cols() != rows {
        return Err(SciOxideError::dimension_mismatch(
            [rows, rows],
            [q.rows(), q.cols()],
        ));
    }
    if r.rows() != rows || r.cols() != cols {
        return Err(SciOxideError::dimension_mismatch(
            [rows, cols],
            [r.rows(), r.cols()],
        ));
    }
    if let Some(p) = &permutation
        && p.len() != cols
    {
        return Err(SciOxideError::dimension_mismatch([cols], [p.len()]));
    }
    let context = context.with_accuracy(scioxide_core::Accuracy::Standard);
    let request = QrRequest::Explicit { rows, cols };
    with_scratch(workspace, request, |scratch| {
        let _ = factor_candidate(matrix, scratch, policy, context, pivoting)?;
        for row in 0..rows {
            for col in 0..rows {
                q.set_element(row, col, if row == col { S::ONE } else { S::ZERO });
            }
        }
        crate::reflector::apply_blocked(
            crate::reflector::Packed {
                factors: scratch.candidate.try_view()?,
                tau: scratch.tau,
                offset: 0,
            },
            crate::reflector::Target::Matrix(crate::reflector::Product::Q),
            q.reborrow(),
            crate::reflector::CompactWy {
                vectors: &mut scratch.v_block,
                triangular: &mut scratch.t_block,
                work: &mut scratch.work_block,
            },
            scratch.gemm,
            crate::kernel::KernelExecution { policy, context },
        )?;
        for row in 0..rows {
            for col in 0..cols {
                r.set_element(
                    row,
                    col,
                    if row <= col {
                        scratch.candidate.get_element(row, col)
                    } else {
                        S::ZERO
                    },
                );
            }
        }
        if let Some(p) = permutation {
            p.copy_from_slice(scratch.permutation);
        }
        context.execution().check()
    })
}

struct ColumnNorms<'a, R> {
    values: &'a mut [R],
    tags: &'a mut [usize],
}
impl<R: Copy> ColumnNorms<'_, R> {
    fn get(&self, column: usize) -> ColumnNorm<R> {
        if self.tags[column] == 0 {
            ColumnNorm::Recompute
        } else {
            ColumnNorm::Reliable {
                partial: self.values[2 * column],
                reference: self.values[2 * column + 1],
            }
        }
    }
    fn set(&mut self, column: usize, norm: ColumnNorm<R>) {
        match norm {
            ColumnNorm::Recompute => self.tags[column] = 0,
            ColumnNorm::Reliable { partial, reference } => {
                self.values[2 * column] = partial;
                self.values[2 * column + 1] = reference;
                self.tags[column] = 1;
            }
        }
    }
    fn swap(&mut self, left: usize, right: usize) {
        self.tags.swap(left, right);
        self.values.swap(2 * left, 2 * right);
        self.values.swap(2 * left + 1, 2 * right + 1);
    }
    fn from(&self, start: usize) -> impl Iterator<Item = ColumnNorm<R>> + '_ {
        (start..self.tags.len()).map(|column| self.get(column))
    }
}

/// Internal blocked Householder QR engine.
///
/// Stores the factorization in packed form: the upper triangle (including
/// diagonal) contains `R`, and the strict lower triangle stores the
/// Householder reflector vectors `v[1:]` (with implicit leading 1). The
/// `tau` coefficients are stored in a separate vector.
///
#[derive(Debug, Clone)]
#[must_use]
pub(crate) struct QrDecomposition<S: NumericScalar> {
    /// Packed storage: R on/above diagonal, Householder v below diagonal.
    packed: Matrix<S>,
    reflectors: Vec<S>,
    maximum: S::Real,
    /// Beta coefficients for each Householder reflector.
    tau: Vec<S>,
    /// Number of rows of the original matrix.
    rows: usize,
    /// Number of columns of the original matrix.
    cols: usize,
    /// Column permutation satisfying `A[:, permutation] = Q * R`.
    permutation: Vec<usize>,
}

impl<S: NumericScalar> QrDecomposition<S> {
    pub(super) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        super::util::storage_bytes([
            (self.packed.capacity(), core::mem::size_of::<S>()),
            (self.reflectors.capacity(), core::mem::size_of::<S>()),
            (self.tau.capacity(), core::mem::size_of::<S>()),
            (self.permutation.capacity(), core::mem::size_of::<usize>()),
        ])
    }
    /// Create an empty QR factorization for reuse.
    pub(crate) const fn empty() -> Self {
        Self {
            packed: Matrix::empty(),
            reflectors: Vec::new(),
            maximum: S::Real::ZERO,
            tau: Vec::new(),
            rows: 0,
            cols: 0,
            permutation: Vec::new(),
        }
    }

    pub(super) fn factor_with_pivoting(
        matrix: crate::FiniteMatrixView<'_, S>,
        factor: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
        pivoting: crate::HouseholderPivoting,
    ) -> Result<scioxide_core::Scaled<S::Real>, SciOxideError> {
        let context = context.with_accuracy(scioxide_core::Accuracy::Standard);
        let rows = matrix.rows();
        let cols = matrix.cols();
        context.execution().check()?;
        let order = rows.min(cols);
        reserve(workspace, QrRequest::Factor { rows, cols }, policy, context)?;
        factor.packed.try_reserve_shape(rows, cols)?;
        reserve_vec(
            &mut factor.reflectors,
            Self::reflector_len(rows, cols)?,
            scioxide_core::quantity!("factor.reflectors"),
        )?;
        reserve_vec(
            &mut factor.tau,
            order,
            scioxide_core::quantity!("factor.tau"),
        )?;
        reserve_vec(
            &mut factor.permutation,
            cols,
            scioxide_core::quantity!("factor.permutation"),
        )?;

        with_scratch(workspace, QrRequest::Factor { rows, cols }, |workspace| {
            let (maximum, one_norm) =
                factor_candidate(matrix, workspace, policy, context, pivoting)?;
            context.execution().check()?;
            factor.packed.try_resize_flat(rows, cols)?;
            factor
                .packed
                .as_mut_slice()
                .copy_from_slice(workspace.candidate.as_slice());
            gather_reflectors(
                workspace.candidate.as_slice(),
                rows,
                cols,
                &mut factor.reflectors,
            );
            factor.tau.clear();
            factor.tau.extend_from_slice(workspace.tau);
            factor.permutation.clear();
            factor.permutation.extend_from_slice(workspace.permutation);
            factor.maximum = maximum;
            factor.rows = rows;
            factor.cols = cols;
            Ok(one_norm)
        })
    }

    pub(crate) fn triangular_diagonal(&self) -> crate::VectorView<'_, S> {
        let len = self.rows.min(self.cols);
        if len == 0 {
            return crate::VectorView::from_slice(&[]);
        }
        // The packed factor owns a validated rectangular layout. Its diagonal
        // is a subset of that layout, with no reflector entries exposed.
        crate::VectorView {
            data: self.packed.as_slice(),
            len,
            stride: self.cols + 1,
            conjugated: false,
        }
    }

    /// Extract `R` into an `m × n` output matrix.
    ///
    /// # Errors
    ///
    /// Returns [`scioxide_core::SciOxideErrorKind::DimensionMismatch`] unless `output` is `m x n`.
    pub(crate) fn r_matrix_into(&self, output: &mut Matrix<S>) -> Result<(), SciOxideError> {
        validate_exact_matrix(output, self.rows, self.cols)?;
        output.as_mut_slice().fill(S::ZERO);
        for row in 0..self.rows {
            for col in row..self.cols {
                output.as_mut_slice()[row * self.cols + col] = self.packed.get_element(row, col);
            }
        }
        Ok(())
    }

    /// Classify the factor diagonal under an explicit rank policy.
    ///
    pub(crate) fn analyze_rank(
        &self,
        policy: RankPolicy<S::Real>,
    ) -> Result<RankReport<S::Real>, SciOxideError> {
        let order = self.rows.min(self.cols);
        analyze_magnitudes(
            (0..order).map(|index| self.packed.get_element(index, index).abs()),
            order,
            self.rows,
            self.cols,
            policy,
        )
    }

    /// Apply the full-column-rank pseudoinverse in ordinary workspace values.
    pub(super) fn inverse_ordinary(
        &self,
        values: &mut [S],
        work: &mut [S],
        adjoint: bool,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<(), SciOxideError> {
        let work = &mut work[..self.rows];
        if adjoint {
            work.fill(S::ZERO);
            for (i, &source) in self.permutation.iter().enumerate() {
                work[i] = values[source];
            }
        } else {
            work.copy_from_slice(&values[..self.rows]);
            self.apply_q_ordinary(work, true, dispatch)?;
        }
        crate::kernel::triangular::solve(
            crate::kernel::TriangularOperand {
                view: crate::kernel::triangular::Storage::Dense(self.packed.view()),
                triangle: scioxide_core::Triangle::Upper,
                op: if adjoint {
                    GemmOp::Adjoint
                } else {
                    GemmOp::None
                },
                diagonal: scioxide_core::DiagonalKind::Explicit,
            },
            self.maximum,
            &mut work[..self.cols],
            self.cols.saturating_sub(1),
            dispatch,
        )?;
        values.fill(S::ZERO);
        if adjoint {
            self.apply_q_ordinary(work, false, dispatch)?;
            values[..self.rows].copy_from_slice(work);
        } else {
            for (i, &destination) in self.permutation.iter().enumerate() {
                values[destination] = work[i];
            }
        }
        Ok(())
    }

    /// Prepared column-packed reflectors make each inverse application a
    /// sequence of contiguous dispatched dots and vector updates.
    fn apply_q_ordinary(
        &self,
        values: &mut [S],
        adjoint: bool,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<(), SciOxideError> {
        for step in 0..self.tau.len() {
            let col = if adjoint {
                step
            } else {
                self.tau.len() - 1 - step
            };
            let tau = if adjoint {
                self.tau[col].conj()
            } else {
                self.tau[col]
            };
            if tau == S::ZERO {
                continue;
            }
            let start = col * self.rows - col * col.saturating_sub(1) / 2;
            let v = &self.reflectors[start..start + self.rows - col];
            let x = &mut values[col..self.rows];
            let dot = if S::IS_REAL
                && let Some(dot) = crate::kernel::arch::dot(v, x, dispatch)?
            {
                dot
            } else {
                crate::kernel::standard::dot(v.iter().zip(x.iter()).map(|(&v, &x)| (v.conj(), x)))
            };
            let dot = dot * tau;
            for (x, &v) in x.iter_mut().zip(v) {
                *x = *x - v * dot;
            }
        }
        crate::checked::validate_values(
            values,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "Householder inverse action exceeded its ordinary range",
        )
    }

    /// Apply the full-column-rank pseudoinverse or its adjoint in extended range.
    pub(super) fn inverse_scaled(
        &self,
        rhs: &mut [scioxide_core::ScaledValue<S>],
        work: &mut [scioxide_core::ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        let work = &mut work[..self.rows];
        if adjoint {
            work.fill(scioxide_core::ScaledValue::ZERO);
            for (i, &original) in self.permutation.iter().enumerate() {
                work[i] = rhs[original];
            }
            crate::diagnostics::matrix_substitute(
                self.packed.view(),
                scioxide_core::Triangle::Upper,
                scioxide_core::DiagonalKind::Explicit,
                true,
                &mut work[..self.cols],
            )?;
            self.apply_q_scaled(work, false)?;
            rhs.fill(scioxide_core::ScaledValue::ZERO);
            rhs[..self.rows].copy_from_slice(work);
        } else {
            work.copy_from_slice(&rhs[..self.rows]);
            self.apply_q_scaled(work, true)?;
            crate::diagnostics::matrix_substitute(
                self.packed.view(),
                scioxide_core::Triangle::Upper,
                scioxide_core::DiagonalKind::Explicit,
                false,
                &mut work[..self.cols],
            )?;
            rhs.fill(scioxide_core::ScaledValue::ZERO);
            for (i, &original) in self.permutation.iter().enumerate() {
                rhs[original] = work[i];
            }
        }
        Ok(())
    }

    pub(super) fn apply_q_scaled(
        &self,
        values: &mut [scioxide_core::ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::ScaledValue;
        for step in 0..self.tau.len() {
            let i = if adjoint {
                step
            } else {
                self.tau.len() - 1 - step
            };
            let tau = if adjoint {
                self.tau[i].conj()
            } else {
                self.tau[i]
            };
            if tau == S::ZERO {
                continue;
            }
            let mut dot = values[i];
            for (row, &value) in values.iter().enumerate().take(self.rows).skip(i + 1) {
                dot = dot.add(
                    ScaledValue::from_value(self.packed.get_element(row, i).conj())?.mul(value)?,
                )?;
            }
            dot = dot.mul(ScaledValue::from_value(tau)?)?;
            values[i] = values[i].sub(dot)?;
            for (row, value) in values.iter_mut().enumerate().take(self.rows).skip(i + 1) {
                *value = value
                    .sub(ScaledValue::from_value(self.packed.get_element(row, i))?.mul(dot)?)?;
            }
        }
        Ok(())
    }

    /// Column permutation with invariant `A[:, permutation] = Q * R`.
    #[must_use]
    pub(crate) fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    fn apply_solve_buffer(
        &self,
        workspace: &mut QrScratch<'_, '_, S>,
        product: crate::reflector::Product,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let QrScratch {
            solve_buffer,
            v_block,
            t_block,
            work_block,
            gemm,
            ..
        } = workspace;
        crate::reflector::apply_blocked(
            crate::reflector::Packed {
                factors: self.packed.try_view()?,
                tau: &self.tau,
                offset: 0,
            },
            crate::reflector::Target::Matrix(product),
            solve_buffer.try_view_mut()?,
            crate::reflector::CompactWy {
                vectors: v_block,
                triangular: t_block,
                work: work_block,
            },
            gemm,
            crate::kernel::KernelExecution { policy, context },
        )
    }

    pub(crate) fn apply_qh_vector_in_place(&self, output: &mut [S]) -> Result<(), SciOxideError> {
        for column in 0..self.tau.len() {
            let vector = crate::VectorView::from_slice_with_stride(
                &self.packed.as_slice()[column * (self.cols + 1)..],
                self.rows - column,
                self.cols,
            )?;
            crate::reflector::apply(
                crate::FiniteVectorView::from_workspace(vector),
                self.tau[column].conj(),
                crate::MatrixViewMut::from_slice(
                    self.rows - column,
                    1,
                    &mut output[column..self.rows],
                )?,
            )?;
        }
        Ok(())
    }

    pub(super) fn packed_factors(&self) -> crate::FiniteMatrixView<'_, S> {
        crate::FiniteMatrixView::from_validated(self.packed.view(), self.maximum)
    }

    pub(super) fn solve_view(
        &self,
        rhs: crate::FiniteVectorView<'_, S>,
        output: crate::VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.execution().check()?;
        if rhs.len() != self.rows || output.len() != self.cols {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.cols],
                [rhs.len(), output.len()],
            ));
        }
        self.validate_solve_factor()?;
        let request = QrRequest::Vector { rows: self.rows };
        reserve(workspace, request, ExecutionPolicy::Sequential, context)?;
        with_scratch(workspace, request, |scratch| {
            scratch
                .vector_candidate
                .iter_mut()
                .zip(rhs.iter())
                .for_each(|(out, value)| *out = value);
            if crate::kernel::standard::product_scales(S::Real::TWO, rhs.maximum, self.rows)?
                == (0, 0)
            {
                self.apply_q_ordinary(scratch.vector_candidate, true, context.dispatch())?;
            } else {
                self.apply_qh_vector_in_place(scratch.vector_candidate)?;
            }
            crate::kernel::triangular::solve(
                crate::kernel::TriangularOperand {
                    view: crate::kernel::triangular::Storage::Dense(
                        self.packed.view().block(0, 0, self.cols, self.cols)?,
                    ),
                    triangle: scioxide_core::Triangle::Upper,
                    op: GemmOp::None,
                    diagonal: scioxide_core::DiagonalKind::Explicit,
                },
                self.maximum,
                &mut scratch.vector_candidate[..self.cols],
                self.cols.saturating_sub(1),
                context.dispatch(),
            )?;
            crate::checked::validate_values(
                scratch.vector_candidate,
                scioxide_core::NonFiniteOrigin::Intermediate,
                "computed QR value is not finite",
            )?;
            context.execution().check()?;
            for (column, &original) in self.permutation.iter().enumerate() {
                output.data[original * output.stride] = scratch.vector_candidate[column];
            }
            Ok(())
        })
    }

    pub(super) fn solve_matrix_view(
        &self,
        rhs: crate::FiniteMatrixView<'_, S>,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.execution().check()?;
        if rhs.rows() != self.rows || output.rows() != self.cols || rhs.cols() != output.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [self.cols, rhs.cols()],
                [output.rows(), output.cols()],
            ));
        }
        self.validate_solve_factor()?;
        let request = QrRequest::SolveMatrix {
            rows: self.rows,
            cols: self.cols,
            rhs: rhs.cols(),
        };
        reserve(workspace, request, policy, context)?;
        with_scratch(workspace, request, |scratch| {
            copy_block(rhs.as_view(), &mut scratch.solve_buffer)?;
            self.apply_solve_buffer(scratch, crate::reflector::Product::Adjoint, policy, context)?;
            crate::kernel::level3::trsm_bounded(
                crate::kernel::TriangularOperand {
                    view: self.packed_factors().block(0, 0, self.cols, self.cols)?,
                    triangle: scioxide_core::Triangle::Upper,
                    op: GemmOp::None,
                    diagonal: scioxide_core::DiagonalKind::Explicit,
                },
                matrix_block_mut(&mut scratch.solve_buffer, 0..self.cols, 0..rhs.cols())?,
                crate::MatrixSide::Left,
                S::ONE,
                scratch.gemm,
                context,
            )?;
            crate::checked::validate_values(
                scratch.solve_buffer.as_slice(),
                scioxide_core::NonFiniteOrigin::Intermediate,
                "computed QR value is not finite",
            )?;
            context.execution().check()?;
            for (row, &original) in self.permutation.iter().enumerate() {
                for col in 0..rhs.cols() {
                    output.set_element(original, col, scratch.solve_buffer.get_element(row, col));
                }
            }
            Ok(())
        })
    }

    pub(super) fn q_view(
        &self,
        mut output: crate::MatrixViewMut<'_, S>,
        adjoint: bool,
        workspace: &mut LinalgWorkspace<'_, S>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.execution().check()?;
        if output.rows() != self.rows || output.cols() != self.rows {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.rows],
                [output.rows(), output.cols()],
            ));
        }
        let request = QrRequest::ApplyMatrix {
            rows: self.rows,
            cols: self.cols,
            rhs: self.rows,
        };
        reserve(workspace, request, policy, context)?;
        with_scratch(workspace, request, |scratch| {
            scratch.solve_buffer.as_mut_slice().fill(S::ZERO);
            for i in 0..self.rows {
                scratch.solve_buffer.as_mut_slice()[i * self.rows + i] = S::ONE;
            }
            self.apply_solve_buffer(scratch, crate::reflector::Product::Q, policy, context)?;
            crate::checked::validate_values(
                scratch.solve_buffer.as_slice(),
                scioxide_core::NonFiniteOrigin::Intermediate,
                "computed QR value is not finite",
            )?;
            context.execution().check()?;
            for row in 0..self.rows {
                for col in 0..self.rows {
                    let value = if adjoint {
                        scratch.solve_buffer.get_element(col, row).conj()
                    } else {
                        scratch.solve_buffer.get_element(row, col)
                    };
                    output.set_element(row, col, value);
                }
            }
            Ok(())
        })
    }

    pub(super) fn apply_view(
        &self,
        input: crate::FiniteMatrixView<'_, S>,
        mut output: crate::MatrixViewMut<'_, S>,
        adjoint: bool,
        workspace: &mut LinalgWorkspace<'_, S>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.execution().check()?;
        if input.rows() != self.rows || output.rows() != self.rows || input.cols() != output.cols()
        {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, input.cols()],
                [output.rows(), output.cols()],
            ));
        }
        if output.is_empty() {
            return Ok(());
        }
        let request = QrRequest::ApplyMatrix {
            rows: self.rows,
            cols: self.cols,
            rhs: input.cols(),
        };
        reserve(workspace, request, policy, context)?;
        with_scratch(workspace, request, |scratch| {
            copy_block(input.as_view(), &mut scratch.solve_buffer)?;
            self.apply_solve_buffer(
                scratch,
                if adjoint {
                    crate::reflector::Product::Adjoint
                } else {
                    crate::reflector::Product::Q
                },
                policy,
                context,
            )?;
            crate::checked::validate_values(
                scratch.solve_buffer.as_slice(),
                scioxide_core::NonFiniteOrigin::Intermediate,
                "computed QR value is not finite",
            )?;
            context.execution().check()?;
            for row in 0..self.rows {
                for col in 0..input.cols() {
                    output.set_element(row, col, scratch.solve_buffer.get_element(row, col));
                }
            }
            Ok(())
        })
    }

    pub(super) fn apply_q_vector_in_place(&self, output: &mut [S]) -> Result<(), SciOxideError> {
        for column in (0..self.tau.len()).rev() {
            let vector = crate::VectorView::from_slice_with_stride(
                &self.packed.as_slice()[column * (self.cols + 1)..],
                self.rows - column,
                self.cols,
            )?;
            crate::reflector::apply(
                crate::FiniteVectorView::from_workspace(vector),
                self.tau[column],
                crate::MatrixViewMut::from_slice(
                    self.rows - column,
                    1,
                    &mut output[column..self.rows],
                )?,
            )?;
        }
        Ok(())
    }

    fn validate_solve_factor(&self) -> Result<(), SciOxideError> {
        // The direct solve is least squares through `R^{-1} Q^H b`, which is
        // only defined for full column rank: a wide factor or an exactly zero
        // diagonal of R is reported as singular before any work is done.
        if self.rows < self.cols
            || (0..self.cols)
                .any(|column| self.packed.get_element(column, column).abs() == S::Real::ZERO)
        {
            return Err(scioxide_core::SciOxideError::singular(None));
        }
        Ok(())
    }
}

fn validate_factor_candidate<S: NumericScalar>(
    packed: &impl MatrixStorage<S>,
    tau: &[S],
) -> Result<S::Real, SciOxideError> {
    if tau
        .iter()
        .any(|coefficient| !coefficient.has_finite_components())
    {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "QR factor metadata is not finite",
        ));
    }
    // The permutation is constructed exclusively by swapping the identity.
    crate::kernel::standard::finite_slice_maximum(packed.as_slice())
}

fn validate_exact_matrix<S: NumericScalar>(
    output: &Matrix<S>,
    rows: usize,
    cols: usize,
) -> Result<(), SciOxideError> {
    if output.rows() != rows || output.cols() != cols {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(rows, cols),
            scioxide_core::Dimensions::two(output.rows(), output.cols()),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum ColumnNorm<R> {
    Reliable { partial: R, reference: R },
    Recompute,
}

impl<R: RealScalar> ColumnNorm<R> {
    // xLAQPS / LAPACK Working Note 176: cancellation in the squared norm
    // downdate must end the panel before another pivot can use that estimate.
    fn downdated(self, exposed: R, tolerance: R) -> Self {
        match self {
            Self::Reliable { partial, reference } if partial != R::ZERO => {
                let ratio = exposed / partial;
                let remaining = ((R::ONE + ratio) * (R::ONE - ratio)).max(R::ZERO);
                let reliability = partial / reference;
                if remaining * reliability * reliability <= tolerance {
                    Self::Recompute
                } else {
                    Self::Reliable {
                        partial: partial * remaining.sqrt(),
                        reference,
                    }
                }
            }
            state => state,
        }
    }
}

fn column_norm<S: NumericScalar>(
    matrix: &impl MatrixStorage<S>,
    start: usize,
    col: usize,
) -> Result<S::Real, SciOxideError> {
    if start == matrix.rows() {
        return Ok(S::Real::ZERO);
    }
    let values = crate::VectorView::from_slice_with_stride(
        &matrix.as_slice()[start * matrix.cols() + col..],
        matrix.rows() - start,
        matrix.cols(),
    )?;
    crate::kernel::standard::norm_workspace(values.iter())
}

/// Gather once before the generator's normalization passes and scatter once.
/// The explicit vector remains available to the panel's disjoint updates.
/// Replace `reflectors` with the column-packed vectors `[1, v_j]` of a packed
/// row-major `rows x cols` factor. Blocks of columns are read along rows, so
/// each row segment is visited once.
fn gather_reflectors<S: NumericScalar>(
    packed: &[S],
    rows: usize,
    cols: usize,
    reflectors: &mut Vec<S>,
) {
    const BLOCK: usize = 32;
    let order = rows.min(cols);
    // Column `j` starts after the `rows - k` entries of every earlier column `k`.
    let offset = |j: usize| j * rows - j * j.saturating_sub(1) / 2;
    reflectors.clear();
    reflectors.resize(offset(order), S::ZERO);
    for first in (0..order).step_by(BLOCK) {
        let last = (first + BLOCK).min(order);
        (first..last).for_each(|j| reflectors[offset(j)] = S::ONE);
        for row in first + 1..rows {
            let values = &packed[row * cols..row * cols + last.min(row)];
            for (j, &value) in values.iter().enumerate().skip(first) {
                reflectors[offset(j) + row - j] = value;
            }
        }
    }
}

fn generate_column<S: NumericScalar>(
    data: &mut [S],
    stride: usize,
    rows: usize,
    column: usize,
    vector: &mut impl MatrixStorage<S>,
    narrow: bool,
) -> Result<crate::reflector::Reflector<S>, SciOxideError> {
    vector.try_resize_flat(rows - column, 1)?;
    let mut maximum = S::Real::ZERO;
    for (value, row) in vector.as_mut_slice().iter_mut().zip(column..rows) {
        *value = data[row * stride + column];
        if narrow && S::IS_REAL {
            maximum = maximum.max(value.abs());
        }
    }
    let h = if narrow {
        crate::reflector::generate_narrow(vector.as_mut_slice(), maximum)?
    } else {
        crate::reflector::generate(crate::VectorViewMut::from_slice(vector.as_mut_slice()))?
    };
    for (&value, row) in vector.as_slice().iter().zip(column..rows) {
        data[row * stride + column] = value;
    }
    Ok(h)
}

/// Borrowed storage for the common packed unpivoted QR kernel. The caller
/// copies in the input and copies out the result; no factor object is allocated.
pub(crate) struct UnpivotedQr<'a, S, M> {
    pub(crate) packed: &'a mut M,
    pub(crate) tau: &'a mut [S],
    pub(crate) panel: &'a mut M,
    pub(crate) blocks: crate::reflector::CompactWy<'a, M>,
}

pub(crate) fn factor_unpivoted<S: NumericScalar, M: MatrixStorage<S>>(
    scratch: UnpivotedQr<'_, S, M>,
    input_maximum: S::Real,
    policy: ExecutionPolicy,
    gemm: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let UnpivotedQr {
        packed,
        tau,
        panel,
        blocks,
    } = scratch;
    let crate::reflector::CompactWy {
        vectors,
        triangular,
        work,
    } = blocks;
    let (m, n) = (packed.rows(), packed.cols());
    let order = m.min(n);
    // Narrow panels reduce level-two work on tall inputs. Reflector replay
    // still groups the resulting columns at the common compact-WY width.
    let cap = if order != 0 && m / order >= 2 {
        8
    } else {
        crate::reflector::BLOCK_WIDTH
    };
    let width = crate::kernel::panel_width::<S>(context, cap);
    // Householder updates preserve column 2-norms, so no trailing component
    // exceeds sqrt(2m) max|A|; twice that covers rounding.
    let bound = input_maximum
        * S::Real::ONE.scalbn((usize::BITS - m.leading_zeros() + 1).div_ceil(2) as i32 + 1);
    let bound = (bound < S::Real::MAX.sqrt()).then_some(bound);
    for start in (0..order).step_by(width) {
        context.execution().check()?;
        let end = (start + width).min(order);
        let (panel_width, rows) = (end - start, m - start);
        // Factor the panel and the rows below it in a contiguous row-major buffer.
        panel.try_resize_flat(rows, panel_width)?;
        transfer_panel(
            packed.as_mut_slice(),
            n,
            start..end,
            panel.as_mut_slice(),
            true,
        );
        let mut maximum = crate::FiniteMatrixView::from_workspace(panel.try_view()?).maximum;
        for k in 0..panel_width {
            // Reuse the guarded native norm/reflector path for narrow tall
            // designs; extreme columns retain the general binary-scaled path.
            let h = generate_column(
                panel.as_mut_slice(),
                panel_width,
                rows,
                k,
                &mut *vectors,
                n <= 8 && S::IS_REAL,
            )?;
            tau[start + k] = h.tau;
            maximum = crate::reflector::apply_panel(
                crate::FiniteVectorView::from_validated(
                    crate::VectorView::from_slice(vectors.as_slice()),
                    S::Real::TWO,
                ),
                h.tau.conj(),
                matrix_block_mut(&mut *panel, k..rows, k + 1..panel_width)?,
                maximum,
                &mut *work,
            )?;
            panel.as_mut_slice()[k * (panel_width + 1)] = S::from_real(h.beta);
        }
        if end < n {
            crate::reflector::pack(
                panel.try_view()?,
                &tau[start..end],
                0,
                panel_width,
                0,
                crate::reflector::Direction::Forward,
                &mut *vectors,
                &mut *triangular,
            )?;
            crate::reflector::apply_block(
                &*vectors,
                &*triangular,
                matrix_block_mut(&mut *packed, start..m, end..n)?,
                bound,
                None,
                GemmOp::Adjoint,
                crate::reflector::Direction::Forward,
                &mut *work,
                policy,
                gemm,
                context,
            )?;
        }
        transfer_panel(
            packed.as_mut_slice(),
            n,
            start..end,
            panel.as_mut_slice(),
            false,
        );
    }
    Ok(())
}

/// Blocked column-pivoted QR panel (Quintana-Ortí, Sun & Bischof, 1998):
/// A_trailing = A_stored - V W. Only the pivot column and the newly exposed R
/// row are formed before the panel's level-3 update.
fn factor_pivoted<S: NumericScalar>(
    w: &mut QrScratch<'_, '_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<scioxide_core::Scaled<S::Real>, SciOxideError> {
    use crate::kernel::{GemmConfig, WorkspaceGemmProblem, gemm_workspace_with_context, standard};
    let (m, n) = (w.candidate.rows(), w.candidate.cols());
    let width = crate::kernel::panel_width::<S>(context, 32);
    let tol3z = S::Real::EPSILON.sqrt();
    let mut one_norm = scioxide_core::Scaled::ZERO;
    for col in 0..n {
        let mut sum = S::Real::ZERO;
        // Reuse the required initial column scan for original-matrix rcond₁
        // metadata; neither another matrix pass nor extra storage is needed.
        let column = w.candidate.try_view()?.col(col)?;
        let norm = if m == 0 {
            S::Real::ZERO
        } else if n <= 8 && S::IS_REAL {
            let mut maximum = S::Real::ZERO;
            let squares = crate::kernel::standard::dot(
                column
                    .iter()
                    .inspect(|value| {
                        let magnitude = value.abs();
                        sum += magnitude;
                        maximum = maximum.max(magnitude);
                    })
                    .map(|x| (x, x)),
            );
            if standard::product_scales(maximum, maximum, m)? == (0, 0) {
                crate::checked::finite_candidate(squares.real_component().sqrt())?
            } else {
                crate::kernel::standard::norm_workspace(column.iter())?
            }
        } else {
            crate::kernel::standard::norm_workspace(
                column.iter().inspect(|value| sum += value.abs()),
            )?
        };
        let sum = if sum.is_finite() {
            scioxide_core::Scaled::from_value(sum)?
        } else {
            crate::diagnostics::matrix_one_norm(w.candidate.try_view()?.block(0, col, m, 1)?)?
        };
        if !sum.abs_le(one_norm) {
            one_norm = sum;
        }
        w.norms.set(
            col,
            ColumnNorm::Reliable {
                partial: norm,
                reference: norm,
            },
        );
    }
    if n <= 8 {
        factor_pivoted_narrow(w, context)?;
        return Ok(one_norm);
    }
    let mut start = 0;
    while start < m.min(n) {
        context.execution().check()?;
        let limit = (start + width).min(m.min(n));
        w.panel_update.try_resize_flat(limit - start, n)?;
        w.panel_update.as_mut_slice().fill(S::ZERO);
        let mut end = start;
        for k in start..limit {
            let mut pivot = k;
            let mut largest = S::Real::ZERO;
            for j in k..n {
                if let ColumnNorm::Reliable { partial, .. } = w.norms.get(j)
                    && partial > largest
                {
                    largest = partial;
                    pivot = j;
                }
            }
            if pivot != k {
                crate::reflector::swap_columns(w.candidate.try_view_mut()?, k, pivot);
                crate::reflector::swap_columns(w.panel_update.try_view_mut()?, k, pivot);
                w.permutation.swap(k, pivot);
                w.norms.swap(k, pivot);
            }
            let local = k - start;
            for i in k..m {
                let value = (start..k).fold(w.candidate.get_element(i, k), |value, p| {
                    -w.candidate.get_element(i, p) * w.panel_update.get_element(p - start, k)
                        + value
                });
                w.candidate.as_mut_slice()[i * n + k] = crate::checked::finite_candidate(value)?;
            }
            let h = generate_column(w.candidate.as_mut_slice(), n, m, k, &mut w.v_block, false)?;
            w.tau[k] = h.tau;
            // v_k^H V_previous, shared by every trailing-column projection.
            w.work_block.try_resize_flat(1, local)?;
            w.work_block.as_mut_slice().fill(S::ZERO);
            for i in k..m {
                let coefficient = w.v_block.as_slice()[i - k].conj();
                w.work_block
                    .as_mut_slice()
                    .iter_mut()
                    .zip(&w.candidate.as_slice()[i * n + start..i * n + k])
                    .for_each(|(dot, &value)| *dot = coefficient * value + *dot);
            }
            let maximum = w
                .norms
                .from(k + 1)
                .fold(S::Real::ZERO, |m, norm| match norm {
                    ColumnNorm::Reliable { reference, .. } => m.max(reference),
                    ColumnNorm::Recompute => m,
                });
            match standard::product_scales(S::Real::TWO, maximum, m - k)? {
                (0, 0) => {
                    for i in k..m {
                        let coefficient = w.v_block.as_slice()[i - k].conj();
                        let input = &w.candidate.as_slice()[i * n + k + 1..(i + 1) * n];
                        let output =
                            &mut w.panel_update.as_mut_slice()[local * n + k + 1..(local + 1) * n];
                        output
                            .iter_mut()
                            .zip(input)
                            .for_each(|(y, &x)| *y = coefficient * x + *y);
                    }
                    let (previous, current) =
                        crate::workspace::split_mut(w.panel_update.as_mut_slice(), local * n)?;
                    let output = &mut current[k + 1..n];
                    for p in 0..local {
                        let coefficient = -w.work_block.as_slice()[p];
                        output
                            .iter_mut()
                            .zip(&previous[p * n + k + 1..(p + 1) * n])
                            .for_each(|(y, &x)| *y = coefficient * x + *y);
                    }
                    for value in output {
                        *value = crate::checked::finite_candidate(h.tau.conj() * *value)?;
                    }
                }
                _ => {
                    for j in k + 1..n {
                        let reference = match w.norms.get(j) {
                            ColumnNorm::Reliable { reference, .. } => reference,
                            ColumnNorm::Recompute => {
                                unreachable!("panel stops before using an unreliable norm")
                            }
                        };
                        let (_, e) = standard::product_scales(S::Real::TWO, reference, m - k)?;
                        let dot = (k..m).fold(S::ZERO, |sum, i| {
                            w.candidate.get_element(i, k).conj()
                                * standard::scale(w.candidate.get_element(i, j), -e)
                                + sum
                        });
                        let dot = (0..local).fold(dot, |sum, p| {
                            -w.work_block.as_slice()[p]
                                * standard::scale(w.panel_update.get_element(p, j), -e)
                                + sum
                        });
                        w.panel_update.as_mut_slice()[local * n + j] =
                            standard::restore(h.tau.conj() * dot, e)?;
                    }
                }
            }
            let (leading, trailing) = crate::workspace::split_mut(
                &mut w.candidate.as_mut_slice()[k * n..(k + 1) * n],
                k + 1,
            )?;
            for (p, &coefficient) in leading[start..k].iter().enumerate() {
                trailing
                    .iter_mut()
                    .zip(&w.panel_update.as_slice()[p * n + k + 1..(p + 1) * n])
                    .for_each(|(value, &update)| *value = -coefficient * update + *value);
            }
            let mut recompute = false;
            for j in k + 1..n {
                let value = w.candidate.get_element(k, j) - w.panel_update.get_element(local, j);
                w.candidate.as_mut_slice()[k * n + j] = crate::checked::finite_candidate(value)?;
                let norm = w.norms.get(j).downdated(value.abs(), tol3z);
                recompute |= matches!(norm, ColumnNorm::Recompute);
                w.norms.set(j, norm);
            }
            w.candidate.as_mut_slice()[k * (n + 1)] = S::from_real(h.beta);
            end = k + 1;
            if recompute {
                break;
            }
        }
        if end < m && end < n {
            let count = end - start;
            w.v_block.try_resize_flat(m - end, count)?;
            for i in end..m {
                for p in start..end {
                    w.v_block.as_mut_slice()[(i - end) * count + p - start] =
                        w.candidate.get_element(i, p);
                }
            }
            gemm_workspace_with_context(
                WorkspaceGemmProblem {
                    left: crate::FiniteMatrixView::from_workspace(w.v_block.try_view()?),
                    right: crate::FiniteMatrixView::from_workspace(super::util::matrix_block(
                        &w.panel_update,
                        0..count,
                        end..n,
                    )?),
                    output: matrix_block_mut(&mut w.candidate, end..m, end..n)?,
                    config: GemmConfig {
                        alpha: -S::ONE,
                        beta: S::ONE,
                        ..GemmConfig::ordinary()
                    },
                },
                policy,
                w.gemm,
                context.with_accuracy(scioxide_core::Accuracy::Standard),
            )?;
        }
        for j in end..n {
            if matches!(w.norms.get(j), ColumnNorm::Recompute) {
                let norm = column_norm(&w.candidate, end, j)?;
                w.norms.set(
                    j,
                    ColumnNorm::Reliable {
                        partial: norm,
                        reference: norm,
                    },
                );
            }
        }
        start = end;
    }
    Ok(one_norm)
}

/// Unblocked column-pivoted QR for narrow matrices, without deferred low-rank
/// updates. Norm downdates use the same safeguard and recomputation rule as
/// the blocked panel.
fn factor_pivoted_narrow<S: NumericScalar>(
    w: &mut QrScratch<'_, '_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let (m, n) = (w.candidate.rows(), w.candidate.cols());
    let tolerance = S::Real::EPSILON.sqrt();
    for k in 0..m.min(n) {
        context.execution().check()?;
        let mut pivot = k;
        let mut largest = S::Real::ZERO;
        for j in k..n {
            if let ColumnNorm::Reliable { partial, .. } = w.norms.get(j)
                && partial > largest
            {
                largest = partial;
                pivot = j;
            }
        }
        if pivot != k {
            crate::reflector::swap_columns(w.candidate.try_view_mut()?, k, pivot);
            w.permutation.swap(k, pivot);
            w.norms.swap(k, pivot);
        }
        let h = generate_column(w.candidate.as_mut_slice(), n, m, k, &mut w.v_block, true)?;
        w.tau[k] = h.tau;
        let maximum = w
            .norms
            .from(k + 1)
            .fold(S::Real::ZERO, |bound, norm| match norm {
                ColumnNorm::Reliable { reference, .. } => bound.max(reference),
                ColumnNorm::Recompute => bound,
            });
        crate::reflector::apply_panel(
            crate::FiniteVectorView::from_validated(
                crate::VectorView::from_slice(w.v_block.as_slice()),
                S::Real::TWO,
            ),
            h.tau.conj(),
            matrix_block_mut(&mut w.candidate, k..m, k + 1..n)?,
            maximum,
            &mut w.work_block,
        )?;
        for j in k + 1..n {
            let norm = w
                .norms
                .get(j)
                .downdated(w.candidate.get_element(k, j).abs(), tolerance);
            let norm = if matches!(norm, ColumnNorm::Recompute) {
                let norm = column_norm(&w.candidate, k + 1, j)?;
                ColumnNorm::Reliable {
                    partial: norm,
                    reference: norm,
                }
            } else {
                norm
            };
            w.norms.set(j, norm);
        }
        w.candidate.as_mut_slice()[k * (n + 1)] = S::from_real(h.beta);
    }
    Ok(())
}

impl<S: NumericScalar> QrDecomposition<S> {
    pub(crate) fn reflector_len(rows: usize, cols: usize) -> Result<usize, SciOxideError> {
        let count = rows.min(cols);
        let rectangle = scioxide_core::backend::checked_product(
            &[rows, count],
            scioxide_core::quantity!("factor.reflectors"),
        )?;
        let triangle = scioxide_core::backend::checked_product(
            &[count, count.saturating_sub(1)],
            scioxide_core::quantity!("factor.reflectors"),
        )? / 2;
        Ok(rectangle - triangle)
    }

    pub(crate) fn action_storage(
        &self,
        plan: &mut crate::workspace::accounting::StoragePlan,
        rows: usize,
        cols: usize,
    ) -> Result<(), SciOxideError> {
        plan.matrix(&self.packed, rows, cols)?;
        plan.buffer(&self.reflectors, Self::reflector_len(rows, cols)?)?;
        plan.buffer(&self.tau, rows.min(cols))?;
        plan.buffer(&self.permutation, cols)
    }

    pub(super) fn reserve_factor(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        self.packed.try_reserve_shape(rows, cols)?;
        reserve_vec(
            &mut self.reflectors,
            Self::reflector_len(rows, cols)?,
            scioxide_core::quantity!("factor.reflectors"),
        )?;
        let additional = rows.min(cols).saturating_sub(self.tau.len());
        scioxide_core::backend::try_reserve(
            &mut self.tau,
            additional,
            scioxide_core::quantity!("factor.tau"),
        )?;
        let additional = cols.saturating_sub(self.permutation.len());
        scioxide_core::backend::try_reserve(
            &mut self.permutation,
            additional,
            scioxide_core::quantity!("factor.permutation"),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{ColumnNorm, gather_reflectors};
    use alloc::vec::Vec;

    #[test]
    fn gathered_reflectors_match_column_order_for_every_shape() {
        for (rows, cols) in [(70, 45), (45, 70), (33, 33), (1, 5), (5, 1), (0, 3)] {
            let packed: Vec<f64> = (0..rows * cols).map(|index| index as f64 + 0.5).collect();
            let expected: Vec<f64> = (0..rows.min(cols))
                .flat_map(|col| {
                    core::iter::once(1.0)
                        .chain((col + 1..rows).map(|row| packed[row * cols + col]))
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut reflectors = Vec::new();
            gather_reflectors(&packed, rows, cols, &mut reflectors);
            assert_eq!(reflectors, expected, "{rows}x{cols}");
        }
    }

    #[test]
    fn partial_norms_keep_the_reference_until_the_safeguard_fails() {
        let tolerance = f64::EPSILON.sqrt();
        let norm = ColumnNorm::Reliable {
            partial: 0.5,
            reference: 1.0,
        };
        let (partial, reference) = match norm.downdated(0.4, tolerance) {
            ColumnNorm::Reliable { partial, reference } => Some((partial, reference)),
            ColumnNorm::Recompute => None,
        }
        .unwrap();
        assert!((partial - 0.3).abs() < 1e-15);
        assert_eq!(reference, 1.0);
        assert!(matches!(
            norm.downdated(0.5, tolerance),
            ColumnNorm::Recompute
        ));
        assert!(matches!(
            ColumnNorm::Reliable {
                partial: 0.0,
                reference: 0.0
            }
            .downdated(0.0, tolerance),
            ColumnNorm::Reliable {
                partial: 0.0,
                reference: 0.0
            }
        ));
    }
}
