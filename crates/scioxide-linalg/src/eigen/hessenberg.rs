//! Householder Hessenberg reduction `A = Q H Q^H` for real and complex scalars,
//! and the Householder QR front end of pencil reductions.
//!
//! Reflector `H(j) = I - tau_j v_j v_j^H` has its unit at row `j + 1` and its
//! tail below it in column `j`, the packed layout Hermitian tridiagonalization
//! shares; QR reflectors start on the diagonal ([`Packing`]). Orders above
//! [`BLOCKED_ORDER`] reduce width-32 panels with plain panel arithmetic,
//! Standard trailing products and compact-WY left updates until the trailing
//! order reaches the crossover; unblocked Householder steps reduce the rest.
//! QR blocks the same orders with width-32 panels. The path depends only
//! on the order. Products run on native kernels in scratch owned by the
//! reduction, sized from order and execution policy. Inputs are normalized
//! finite candidates; reflector generation rescales by powers of two only.

use crate::kernel::{
    GemmConfig, GemmOp, KernelExecution, LinalgWorkspace, WorkspaceGemmProblem,
    gemm_workspace_with_context, standard,
};
use crate::reflector::{self, CompactWy, Direction, Packed, Product, Target};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::take_mut;
use crate::{
    DenseWorkspaceRequest, FiniteMatrixView, FiniteVectorView, KernelContext,
    LinalgExecutionOptions, LinalgRequirements, LinalgSlices, MatrixViewMut, VectorView,
    VectorViewMut,
};
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NonFiniteOrigin, NumericScalar, SciOxideError,
};

use super::support::{times, total};

/// Orders above this crossover reduce blocked panels.
pub(super) const BLOCKED_ORDER: usize = 128;
/// Panel width of the blocked reduction; its scratch also holds the
/// compact-WY groups of Q formation.
const WIDTH: usize = 32;
const _: () = assert!(WIDTH >= reflector::BLOCK_WIDTH);

/// Owned preparation adapter for version-coordinated numerical components.
pub(crate) fn prepare<S: NumericScalar>(
    input: crate::MatrixView<'_, S>,
    execution: ExecutionContext<'_>,
) -> Result<(crate::Matrix<S>, crate::Matrix<S>), SciOxideError> {
    let n = input.rows();
    if input.cols() != n {
        return Err(SciOxideError::dimension_mismatch([n, n], [n, input.cols()]));
    }
    execution.check()?;
    let _ = input.try_finite()?;
    let mut h = crate::Matrix::zeros(n, n)?;
    for row in 0..n {
        for col in 0..n {
            h.as_mut_slice()[row * n + col] = input.get(row, col)?;
        }
    }
    let scale = standard::slice_maximum(h.as_slice());
    super::normalize_slice(h.as_mut_slice(), scale);
    let mut q = crate::Matrix::zeros(n, n)?;
    let policy = ExecutionPolicy::Sequential;
    let mut arena = scioxide_core::backend::try_filled(
        Scratch::<S>::len(n, policy)?,
        S::ZERO,
        scioxide_core::quantity!("hessenberg_preparation"),
    )?;
    let mut scratch = Scratch::carve(&mut arena.as_mut_slice(), n, policy)?;
    reduce(&mut h, &mut scratch, policy, execution)?;
    form_q(
        &h,
        Packing::Hessenberg,
        &mut scratch,
        &mut q,
        policy,
        execution,
    )?;
    clear_reflectors(&mut h, Packing::Hessenberg);
    for value in h.as_mut_slice() {
        *value = crate::checked::finite_candidate(*value * S::from_real(scale))?;
    }
    execution.check()?;
    Ok((h, q))
}

/// Reduction scratch carved from one arena of `S`.
pub(super) struct Scratch<'a, S: NumericScalar> {
    tau: &'a mut [S],
    vector: &'a mut [S],
    panels: Option<Panels<'a, S>>,
}

/// Panel products, present exactly for orders above [`BLOCKED_ORDER`].
struct Panels<'a, S: NumericScalar> {
    y: ScratchMatrix<'a, S>,
    vectors: ScratchMatrix<'a, S>,
    triangular: ScratchMatrix<'a, S>,
    work: ScratchMatrix<'a, S>,
    gemm: LinalgWorkspace<'a, S>,
}

/// Compact-WY scratch lent to [`apply_q`] by a blocked caller.
pub(super) struct Blocks<'s, 'a, 'g, S: NumericScalar> {
    pub(super) wy: CompactWy<'s, ScratchMatrix<'a, S>>,
    pub(super) gemm: &'s mut LinalgWorkspace<'g, S>,
    pub(super) execution: KernelExecution<KernelContext<'s>>,
}

/// Layout of packed column reflectors: `H(j)` has its unit at row `j + offset`
/// and its tail below it in column `j`.
#[derive(Clone, Copy)]
pub(super) enum Packing {
    /// Subdiagonal reflectors of a Hessenberg reduction, offset 1.
    Hessenberg,
    /// Diagonal reflectors of a QR factorization, offset 0.
    Triangular,
}

impl Packing {
    const fn offset(self) -> usize {
        match self {
            Self::Hessenberg => 1,
            Self::Triangular => 0,
        }
    }

    /// Reflectors of an order-`n` factor.
    const fn count(self, n: usize) -> usize {
        n.saturating_sub(1 + self.offset())
    }
}

fn products<S: NumericScalar>(
    n: usize,
    policy: ExecutionPolicy,
) -> Result<LinalgRequirements, SciOxideError> {
    let options = LinalgExecutionOptions::new().with_execution_policy(policy);
    [
        DenseWorkspaceRequest::product(n, n, WIDTH, &options),
        DenseWorkspaceRequest::product(n, WIDTH, n, &options),
        DenseWorkspaceRequest::product(WIDTH, n, n, &options),
        DenseWorkspaceRequest::structured(WIDTH, WIDTH, n, &options),
    ]
    .iter()
    .try_fold(LinalgRequirements::default(), |requirements, request| {
        Ok(requirements.max(&crate::kernel::product_requirements::<S>(request)?))
    })
}

impl<'a, S: NumericScalar> Scratch<'a, S> {
    /// Arena elements of an order-`n` reduction whose products follow `policy`.
    pub(super) fn len(n: usize, policy: ExecutionPolicy) -> Result<usize, SciOxideError> {
        let base = times(2, n)?;
        if n <= BLOCKED_ORDER {
            return Ok(base);
        }
        let panel = times(n, WIDTH)?;
        total(&[
            base,
            panel,
            panel,
            WIDTH * WIDTH,
            panel,
            products::<S>(n, policy)?.scalar_len()?,
        ])
    }

    /// Carve the scratch of [`Self::len`] from the front of `rest`.
    pub(super) fn carve(
        rest: &mut &'a mut [S],
        n: usize,
        policy: ExecutionPolicy,
    ) -> Result<Self, SciOxideError> {
        let tau = take_mut(rest, n)?;
        let vector = take_mut(rest, n)?;
        let panels = if n > BLOCKED_ORDER {
            let products = products::<S>(n, policy)?;
            let mut matrix = |rows: usize, cols: usize| {
                ScratchMatrix::new(take_mut(rest, rows * cols)?, rows, cols)
            };
            let (y, vectors, triangular, work) = (
                matrix(n, WIDTH)?,
                matrix(n, WIDTH)?,
                matrix(WIDTH, WIDTH)?,
                matrix(WIDTH, n)?,
            );
            let gemm = LinalgWorkspace::from_slices(
                LinalgSlices {
                    scalars: take_mut(rest, products.scalar_len()?)?,
                    ..LinalgSlices::default()
                },
                &products,
            )?;
            Some(Panels {
                y,
                vectors,
                triangular,
                work,
                gemm,
            })
        } else {
            None
        };
        Ok(Self {
            tau,
            vector,
            panels,
        })
    }
}

/// Reduce a finite square candidate to upper Hessenberg form in place.
///
/// Reflector tails are stored below the subdiagonal; [`form_q`] reads them and
/// [`clear_reflectors`] removes them. Non-finite candidates are reported as
/// intermediate failures.
pub(super) fn reduce<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    scratch: &mut Scratch<'_, S>,
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = matrix.rows();
    let mut first = 0;
    if let Some(panels) = scratch.panels.as_mut() {
        let context = KernelContext::new(execution);
        while n - first > BLOCKED_ORDER {
            execution.check()?;
            panel(
                matrix,
                first,
                &mut scratch.tau[first..first + WIDTH],
                scratch.vector,
                panels,
                policy,
                context,
            )?;
            first += WIDTH;
        }
    }
    execution.check()?;
    for column in first..n.saturating_sub(2) {
        scratch.tau[column] = elementary(matrix, column, scratch.vector)?;
    }
    if matrix
        .as_slice()
        .iter()
        .all(|value| value.has_finite_components())
    {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "Hessenberg candidate is not finite",
        ))
    }
}

/// Overwrite `q` with the unitary factor of packed reflectors: the factor of
/// a [`reduce`]d matrix or of a [`triangularize`]d one.
pub(super) fn form_q<S: NumericScalar>(
    packed: &impl MatrixStorage<S>,
    packing: Packing,
    scratch: &mut Scratch<'_, S>,
    q: &mut impl MatrixStorage<S>,
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = packed.rows();
    q.as_mut_slice().fill(S::ZERO);
    (0..n).for_each(|index| q.as_mut_slice()[index * n + index] = S::ONE);
    let blocks = scratch.panels.as_mut().map(|panels| Blocks {
        wy: CompactWy {
            vectors: &mut panels.vectors,
            triangular: &mut panels.triangular,
            work: &mut panels.work,
        },
        gemm: &mut panels.gemm,
        execution: KernelExecution {
            policy,
            context: KernelContext::new(execution),
        },
    });
    apply(
        packed,
        &scratch.tau[..packing.count(n)],
        q.try_view_mut()?,
        Target::Identity,
        packing,
        blocks,
    )
}

/// Left-multiply `output` by `Q = H(0) ... H(m-1)` of packed subdiagonal
/// reflectors. `blocks` selects width-32 compact-WY products;
/// `None` applies elementary reflectors.
pub(super) fn apply_q<S: NumericScalar>(
    packed: &impl MatrixStorage<S>,
    tau: &[S],
    output: MatrixViewMut<'_, S>,
    blocks: Option<Blocks<'_, '_, '_, S>>,
) -> Result<(), SciOxideError> {
    apply(
        packed,
        tau,
        output,
        Target::Matrix(Product::Q),
        Packing::Hessenberg,
        blocks,
    )
}

/// Zero the stored reflector tails, leaving the Hessenberg or triangular factor.
pub(super) fn clear_reflectors<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    packing: Packing,
) {
    let n = matrix.cols();
    if n == 0 {
        return;
    }
    let offset = packing.offset();
    matrix
        .as_mut_slice()
        .chunks_exact_mut(n)
        .enumerate()
        .skip(offset + 1)
        .for_each(|(row, values)| values[..row - offset].fill(S::ZERO));
}

/// Householder QR of a finite square candidate that also applies `Q^H` to
/// `companion`, as the front end of a pencil reduction.
///
/// Reflector tails are stored below the diagonal of `matrix` in the
/// [`Packing::Triangular`] layout for [`form_q`]. Orders above
/// [`BLOCKED_ORDER`] factor width-32 panels and apply their compact-WY block
/// reflectors to the trailing columns and to `companion` with Standard
/// level-3 products; smaller orders apply each reflector to contiguous rows.
/// `row` is a work row of at least `n` elements.
pub(super) fn triangularize<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    companion: &mut impl MatrixStorage<S>,
    scratch: &mut Scratch<'_, S>,
    row: &mut [S],
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = matrix.rows();
    let count = Packing::Triangular.count(n);
    let context = KernelContext::new(execution);
    let width = if scratch.panels.is_some() {
        WIDTH
    } else {
        count.max(1)
    };
    let mut work = ScratchMatrix::new(row, 1, n)?;
    for start in (0..count).step_by(width) {
        let end = (start + width).min(count);
        for column in start..end {
            execution.check()?;
            let len = n - column;
            let h = reflector::generate(VectorViewMut::from_slice_with_stride(
                &mut matrix.as_mut_slice()[column * (n + 1)..],
                len,
                n,
            )?)?;
            scratch.tau[column] = h.tau;
            let vector = &mut scratch.vector[..len];
            vector
                .iter_mut()
                .enumerate()
                .for_each(|(index, value)| *value = matrix.get_element(column + index, column));
            matrix.as_mut_slice()[column * (n + 1)] = S::from_real(h.beta);
            let v = FiniteVectorView::from_workspace(VectorView::from_slice(vector));
            // Inside a panel only its own columns are updated; unblocked
            // reflectors also reach the trailing columns and the companion.
            let last = if scratch.panels.is_some() { end } else { n };
            let maximum = FiniteMatrixView::from_workspace(matrix.try_view()?.block(
                column,
                column + 1,
                len,
                last - column - 1,
            )?)
            .maximum;
            reflector::apply_panel(
                v,
                h.tau.conj(),
                matrix
                    .try_view_mut()?
                    .block_mut(column, column + 1, len, last - column - 1)?,
                maximum,
                &mut work,
            )?;
            if scratch.panels.is_none() {
                let maximum = FiniteMatrixView::from_workspace(
                    companion.try_view()?.block(column, 0, len, n)?,
                )
                .maximum;
                reflector::apply_panel(
                    v,
                    h.tau.conj(),
                    companion.try_view_mut()?.block_mut(column, 0, len, n)?,
                    maximum,
                    &mut work,
                )?;
            }
        }
        if let Some(panels) = scratch.panels.as_mut() {
            reflector::pack(
                matrix.try_view()?,
                &scratch.tau[..count],
                start,
                end,
                0,
                Direction::Forward,
                &mut panels.vectors,
                &mut panels.triangular,
            )?;
            for output in [
                matrix
                    .try_view_mut()?
                    .block_mut(start, end, n - start, n - end)?,
                companion
                    .try_view_mut()?
                    .block_mut(start, 0, n - start, n)?,
            ] {
                reflector::apply_block(
                    &panels.vectors,
                    &panels.triangular,
                    output,
                    None,
                    None,
                    GemmOp::Adjoint,
                    Direction::Forward,
                    &mut panels.work,
                    policy,
                    &mut panels.gemm,
                    context,
                )?;
            }
        }
    }
    if matrix
        .as_slice()
        .iter()
        .chain(companion.as_slice())
        .all(|value| value.has_finite_components())
    {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "triangular pencil candidate is not finite",
        ))
    }
}

/// Left-multiply `output` by `Q` of `packing`-laid reflectors: compact-WY
/// groups with `blocks`, elementary reflectors otherwise.
fn apply<S: NumericScalar>(
    packed: &impl MatrixStorage<S>,
    tau: &[S],
    mut output: MatrixViewMut<'_, S>,
    target: Target,
    packing: Packing,
    blocks: Option<Blocks<'_, '_, '_, S>>,
) -> Result<(), SciOxideError> {
    let offset = packing.offset();
    if let Some(Blocks {
        wy,
        gemm,
        execution,
    }) = blocks
    {
        return reflector::apply_blocked(
            Packed {
                factors: packed.try_view()?,
                tau,
                offset,
            },
            target,
            output,
            wy,
            gemm,
            execution,
        );
    }
    let n = packed.rows();
    let cols = output.cols();
    (0..tau.len()).rev().try_for_each(|step| {
        let start = step + offset;
        let vector = VectorView::from_slice_with_stride(
            &packed.as_slice()[start * n + step..],
            n - start,
            n,
        )?;
        let first = target.first_column(start);
        reflector::apply(
            FiniteVectorView::from_workspace(vector),
            tau[step],
            output.block_mut(start, first, n - start, cols - first)?,
        )
    })
}

/// One unblocked reduction step: `A := H^H A H` for the reflector of `column`.
fn elementary<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    column: usize,
    vector: &mut [S],
) -> Result<S, SciOxideError> {
    let n = matrix.rows();
    let start = column + 1;
    let len = n - start;
    let vector = &mut vector[..len];
    vector
        .iter_mut()
        .enumerate()
        .for_each(|(index, value)| *value = matrix.get_element(start + index, column));
    let h = reflector::generate(VectorViewMut::from_slice(vector))?;
    let v = FiniteVectorView::from_workspace(VectorView::from_slice(vector));
    reflector::apply(
        v,
        h.tau.conj(),
        matrix.try_view_mut()?.block_mut(start, start, len, len)?,
    )?;
    reflector::apply(
        v.conjugate(),
        h.tau,
        matrix
            .try_view_mut()?
            .block_mut(0, start, n, len)?
            .transpose(),
    )?;
    let slice = matrix.as_mut_slice();
    slice[start * n + column] = S::from_real(h.beta);
    for (index, &value) in vector.iter().enumerate().skip(1) {
        slice[(start + index) * n + column] = value;
    }
    Ok(h.tau)
}

/// Reduce the width-32 panel starting at column `first`, then apply its block
/// reflector to the trailing matrix.
///
/// With `top = first + 1`, the panel forms `V` (rows `top..`, explicit unit
/// diagonal), upper triangular `T` and `Y = A V T` of the panel-start matrix.
fn panel<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    first: usize,
    tau: &mut [S],
    vector: &mut [S],
    panels: &mut Panels<'_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let n = matrix.rows();
    let top = first + 1;
    let rows = n - top;
    let trailing = first + WIDTH;
    let Panels {
        y,
        vectors,
        triangular,
        work,
        gemm,
    } = panels;
    y.try_resize_flat(n, WIDTH)?;
    vectors.try_resize_flat(rows, WIDTH)?;
    triangular.try_resize_flat(WIDTH, WIDTH)?;
    vectors.as_mut_slice().fill(S::ZERO);
    triangular.as_mut_slice().fill(S::ZERO);
    for (index, tau) in tau.iter_mut().enumerate() {
        let column = first + index;
        if index > 0 {
            update_panel_column(matrix, column, index, top, y, vectors, triangular, vector);
        }
        let h = {
            let slice = matrix.as_mut_slice();
            reflector::generate(VectorViewMut::from_slice_with_stride(
                &mut slice[(column + 1) * n + column..],
                n - column - 1,
                n,
            )?)?
        };
        *tau = h.tau;
        for row in index..rows {
            vectors.as_mut_slice()[row * WIDTH + index] = matrix.get_element(top + row, column);
        }
        matrix.as_mut_slice()[(column + 1) * n + column] = S::from_real(h.beta);
        panel_products(
            matrix, column, index, top, h.tau, y, vectors, triangular, vector, context,
        )?;
    }

    // Rows above the reflectors: Y[..top] = A[..top, top..] V T.
    gemm_workspace_with_context(
        WorkspaceGemmProblem {
            left: FiniteMatrixView::from_workspace(matrix.try_view()?.block(0, top, top, rows)?),
            right: FiniteMatrixView::from_workspace(vectors.try_view()?),
            output: y.try_view_mut()?.block_mut(0, 0, top, WIDTH)?,
            config: GemmConfig::ordinary(),
        },
        policy,
        gemm,
        context,
    )?;
    for row in 0..top {
        let values = &mut y.as_mut_slice()[row * WIDTH..(row + 1) * WIDTH];
        for col in (0..WIDTH).rev() {
            values[col] = (0..=col).fold(S::ZERO, |sum, inner| {
                sum + values[inner] * triangular.get_element(inner, col)
            });
        }
    }

    // A := A H on the trailing columns, then on the panel columns above the reflectors.
    gemm_workspace_with_context(
        WorkspaceGemmProblem {
            left: FiniteMatrixView::from_workspace(y.try_view()?),
            right: FiniteMatrixView::from_workspace(vectors.try_view()?.block(
                WIDTH - 1,
                0,
                n - trailing,
                WIDTH,
            )?),
            output: matrix
                .try_view_mut()?
                .block_mut(0, trailing, n, n - trailing)?,
            config: GemmConfig {
                alpha: -S::ONE,
                beta: S::ONE,
                right_op: GemmOp::Adjoint,
                ..GemmConfig::ordinary()
            },
        },
        policy,
        gemm,
        context,
    )?;
    for offset in 0..WIDTH - 1 {
        for row in 0..top {
            let correction = (0..=offset).fold(S::ZERO, |sum, inner| {
                sum + y.get_element(row, inner) * vectors.get_element(offset, inner).conj()
            });
            let slice = matrix.as_mut_slice();
            slice[row * n + top + offset] = slice[row * n + top + offset] - correction;
        }
    }

    // A := H^H A on the trailing rows and columns (xLARFB).
    reflector::apply_block(
        vectors,
        triangular,
        matrix
            .try_view_mut()?
            .block_mut(top, trailing, rows, n - trailing)?,
        None,
        None,
        GemmOp::Adjoint,
        Direction::Forward,
        work,
        policy,
        gemm,
        context,
    )
}

/// Bring panel column `column` up to date with the panel's earlier reflectors:
/// `b -= Y V(column, ..)^H`, then `b := (I - V T V^H)^H b` on rows `top..`.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn update_panel_column<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    column: usize,
    index: usize,
    top: usize,
    y: &ScratchMatrix<'_, S>,
    vectors: &ScratchMatrix<'_, S>,
    triangular: &ScratchMatrix<'_, S>,
    work: &mut [S],
) {
    let n = matrix.rows();
    let rows = n - top;
    let source = &vectors.as_slice()[(column - top) * WIDTH..][..index];
    for row in top..n {
        let correction = y.as_slice()[row * WIDTH..][..index]
            .iter()
            .zip(source)
            .fold(S::ZERO, |sum, (&y, &v)| sum + y * v.conj());
        let slice = matrix.as_mut_slice();
        slice[row * n + column] = slice[row * n + column] - correction;
    }
    let w = &mut work[..index];
    w.fill(S::ZERO);
    for row in 0..rows {
        let value = matrix.get_element(top + row, column);
        w.iter_mut()
            .zip(&vectors.as_slice()[row * WIDTH..][..index])
            .for_each(|(sum, &v)| *sum = *sum + v.conj() * value);
    }
    for col in (0..index).rev() {
        w[col] = (0..=col).fold(S::ZERO, |sum, inner| {
            sum + triangular.get_element(inner, col).conj() * w[inner]
        });
    }
    for row in 0..rows {
        let correction = vectors.as_slice()[row * WIDTH..][..index]
            .iter()
            .zip(w.iter())
            .fold(S::ZERO, |sum, (&v, &w)| sum + v * w);
        let slice = matrix.as_mut_slice();
        slice[(top + row) * n + column] = slice[(top + row) * n + column] - correction;
    }
}

/// Form `Y[top.., index]` and `T[..=index, index]` for the new reflector `v`.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn panel_products<S: NumericScalar>(
    matrix: &impl MatrixStorage<S>,
    column: usize,
    index: usize,
    top: usize,
    tau: S,
    y: &mut ScratchMatrix<'_, S>,
    vectors: &ScratchMatrix<'_, S>,
    triangular: &mut ScratchMatrix<'_, S>,
    work: &mut [S],
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let n = matrix.rows();
    let len = n - column - 1;
    let v = &mut work[..len];
    v.iter_mut()
        .enumerate()
        .for_each(|(offset, value)| *value = vectors.get_element(index + offset, index));
    // Y[:, index] = A[:, column + 1..] v over the panel-start trailing columns.
    for row in top..n {
        let a = &matrix.as_slice()[row * n + column + 1..(row + 1) * n];
        y.as_mut_slice()[row * WIDTH + index] = crate::kernel::arch::dot(a, v, context.dispatch())?
            .unwrap_or_else(|| standard::dot(a.iter().copied().zip(v.iter().copied())));
    }
    // t = V[column + 1.., ..index]^H v; vectors are zero above their unit.
    let t = triangular.as_mut_slice();
    (0..index).for_each(|col| t[col * WIDTH + index] = S::ZERO);
    for (offset, &value) in v.iter().enumerate() {
        let row = &vectors.as_slice()[(index + offset) * WIDTH..][..index];
        for (col, &entry) in row.iter().enumerate() {
            t[col * WIDTH + index] = t[col * WIDTH + index] + entry.conj() * value;
        }
    }
    for row in top..n {
        let values = &mut y.as_mut_slice()[row * WIDTH..][..=index];
        let correction = (0..index).fold(S::ZERO, |sum, col| {
            sum + values[col] * t[col * WIDTH + index]
        });
        values[index] = (values[index] - correction) * tau;
    }
    (0..index).for_each(|col| t[col * WIDTH + index] = -tau * t[col * WIDTH + index]);
    for row in 0..index {
        t[row * WIDTH + index] = (row..index).fold(S::ZERO, |sum, inner| {
            sum + t[row * WIDTH + inner] * t[inner * WIDTH + index]
        });
    }
    t[index * WIDTH + index] = tau;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Matrix;
    use alloc::vec::Vec;
    use scioxide_core::{AlgebraicScalar, Complex, RealScalar};

    /// Full-rank pseudo-random entries in [-1/2, 1/2), so Q is well determined.
    fn fixture<S: NumericScalar>(n: usize, scale: S::Real) -> Matrix<S> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            S::Real::try_from_f64_lossy((state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5).unwrap()
        };
        Matrix::from_vec(
            n,
            n,
            (0..n * n)
                .map(|_| {
                    let (re, im) = (next(), next());
                    S::from_components(
                        re * scale,
                        if S::IS_REAL {
                            S::Real::ZERO
                        } else {
                            im * scale
                        },
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    /// Reduced H (cleared) and Q, blocked exactly when `blocked`.
    fn factor<S: NumericScalar>(matrix: &Matrix<S>, blocked: bool) -> (Matrix<S>, Matrix<S>) {
        let n = matrix.rows();
        let policy = ExecutionPolicy::Sequential;
        let mut arena = alloc::vec![S::ZERO; Scratch::<S>::len(n, policy).unwrap()];
        let mut rest = arena.as_mut_slice();
        let mut scratch = Scratch::carve(&mut rest, n, policy).unwrap();
        assert_eq!(scratch.panels.is_some(), n > BLOCKED_ORDER);
        if !blocked {
            scratch.panels = None;
        }
        let execution = crate::kernel::tests::test_context();
        let mut h = matrix.clone();
        reduce(&mut h, &mut scratch, policy, execution).unwrap();
        let mut q = Matrix::zeros(n, n).unwrap();
        form_q(
            &h,
            Packing::Hessenberg,
            &mut scratch,
            &mut q,
            policy,
            execution,
        )
        .unwrap();
        clear_reflectors(&mut h, Packing::Hessenberg);
        (h, q)
    }

    fn check<S: NumericScalar>(n: usize, scale: S::Real) {
        let a = fixture::<S>(n, scale);
        let (h, q) = factor(&a, true);
        let epsilon = S::Real::EPSILON;
        let bound = S::Real::try_from_f64_lossy(100.0 * n.max(1) as f64).unwrap() * epsilon;
        let norm = a
            .as_slice()
            .iter()
            .fold(S::Real::ZERO, |m, v| m.max(standard::magnitude(*v)));
        let mut orthogonality = S::Real::ZERO;
        let mut residual = S::Real::ZERO;
        let mut qh = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                qh.push((0..n).fold(S::ZERO, |s, k| {
                    s + q.get_element(row, k) * h.get_element(k, col)
                }));
            }
        }
        for row in 0..n {
            for col in 0..n {
                if row > col + 1 {
                    assert_eq!(h.get_element(row, col), S::ZERO);
                }
                let gram = (0..n).fold(S::ZERO, |s, k| {
                    s + q.get_element(k, row).conj() * q.get_element(k, col)
                });
                let identity = if row == col { S::ONE } else { S::ZERO };
                orthogonality = orthogonality.max(standard::magnitude(gram - identity));
                let product = (0..n).fold(S::ZERO, |s, k| {
                    s + qh[row * n + k] * q.get_element(col, k).conj()
                });
                residual = residual.max(standard::magnitude(product - a.get_element(row, col)));
            }
        }
        assert!(
            orthogonality <= bound,
            "n={n} orthogonality {orthogonality:?}"
        );
        assert!(
            residual <= bound * norm,
            "n={n} residual {residual:?} norm {norm:?}"
        );
        if n > BLOCKED_ORDER {
            let (unblocked_h, unblocked_q) = factor(&a, false);
            let difference = |x: &Matrix<S>, y: &Matrix<S>| {
                x.as_slice()
                    .iter()
                    .zip(y.as_slice())
                    .fold(S::Real::ZERO, |m, (a, b)| {
                        m.max(standard::magnitude(*a - *b))
                    })
            };
            assert!(
                difference(&h, &unblocked_h) <= bound * norm,
                "n={n} blocked H"
            );
            assert!(difference(&q, &unblocked_q) <= bound, "n={n} blocked Q");
        }
    }

    #[test]
    fn blocked_and_unblocked_reductions_are_unitary_similarities() {
        // Unblocked, the crossover, one panel plus tail, and two panels plus tail.
        for n in [0, 1, 2, 5, BLOCKED_ORDER, BLOCKED_ORDER + 1, 161] {
            check::<f64>(n, 1.0);
            check::<Complex<f64>>(n, 1.0);
        }
        for scale in [1.0e-200, 1.0e200] {
            check::<f64>(161, scale);
            check::<Complex<f64>>(161, scale);
        }
        check::<f32>(161, 1.0e-20);
        check::<Complex<f32>>(161, 1.0e20);
    }

    /// `(R, Q^H A, Q)` of the QR front end, blocked exactly when `blocked`.
    fn triangular<S: NumericScalar>(
        b: &Matrix<S>,
        a: &Matrix<S>,
        blocked: bool,
    ) -> (Matrix<S>, Matrix<S>, Matrix<S>) {
        let n = b.rows();
        let policy = ExecutionPolicy::Sequential;
        let mut arena = alloc::vec![S::ZERO; Scratch::<S>::len(n, policy).unwrap()];
        let mut rest = arena.as_mut_slice();
        let mut scratch = Scratch::carve(&mut rest, n, policy).unwrap();
        if !blocked {
            scratch.panels = None;
        }
        let mut row = alloc::vec![S::ZERO; n];
        let execution = crate::kernel::tests::test_context();
        let (mut r, mut companion) = (b.clone(), a.clone());
        triangularize(
            &mut r,
            &mut companion,
            &mut scratch,
            &mut row,
            policy,
            execution,
        )
        .unwrap();
        let mut q = Matrix::zeros(n, n).unwrap();
        form_q(
            &r,
            Packing::Triangular,
            &mut scratch,
            &mut q,
            policy,
            execution,
        )
        .unwrap();
        clear_reflectors(&mut r, Packing::Triangular);
        (r, companion, q)
    }

    #[test]
    fn qr_front_end_is_a_unitary_triangularization() {
        fn check<S: NumericScalar>(n: usize) {
            let b = fixture::<S>(n, S::Real::ONE);
            let a = fixture::<S>(n, S::Real::TWO)
                .view()
                .transpose()
                .try_to_matrix()
                .unwrap();
            let bound =
                S::Real::try_from_f64_lossy(100.0 * n.max(1) as f64).unwrap() * S::Real::EPSILON;
            let product = |left: &Matrix<S>, right: &Matrix<S>, row: usize, col: usize| {
                (0..n).fold(S::ZERO, |sum, k| {
                    sum + left.get_element(row, k) * right.get_element(k, col)
                })
            };
            let (r, companion, q) = triangular(&b, &a, true);
            for row in 0..n {
                for col in 0..n {
                    if row > col {
                        assert_eq!(r.get_element(row, col), S::ZERO);
                    }
                    let gram = (0..n).fold(S::ZERO, |sum, k| {
                        sum + q.get_element(k, row).conj() * q.get_element(k, col)
                    });
                    let identity = if row == col { S::ONE } else { S::ZERO };
                    assert!(standard::magnitude(gram - identity) <= bound, "n={n} Q");
                    let rebuilt = product(&q, &r, row, col) - b.get_element(row, col);
                    assert!(standard::magnitude(rebuilt) <= bound, "n={n} QR");
                    let restored = product(&q, &companion, row, col) - a.get_element(row, col);
                    assert!(
                        standard::magnitude(restored) <= bound * S::Real::TWO,
                        "n={n} companion"
                    );
                }
            }
            if n > BLOCKED_ORDER {
                let (unblocked, _, unblocked_q) = triangular(&b, &a, false);
                let difference = |x: &Matrix<S>, y: &Matrix<S>| {
                    x.as_slice()
                        .iter()
                        .zip(y.as_slice())
                        .fold(S::Real::ZERO, |m, (a, b)| {
                            m.max(standard::magnitude(*a - *b))
                        })
                };
                assert!(difference(&r, &unblocked) <= bound, "n={n} blocked R");
                assert!(difference(&q, &unblocked_q) <= bound, "n={n} blocked Q");
            }
        }
        for n in [0, 1, 2, 5, BLOCKED_ORDER, BLOCKED_ORDER + 1, 161] {
            check::<f64>(n);
            check::<Complex<f64>>(n);
        }
        check::<f32>(40);
    }

    #[test]
    fn extreme_finite_input_returns_typed_errors() {
        let mut arena = [0.0_f64; 6];
        let mut rest = &mut arena[..];
        let mut scratch = Scratch::carve(&mut rest, 3, ExecutionPolicy::Sequential).unwrap();
        let mut real = Matrix::from_vec(3, 3, alloc::vec![f64::MAX; 9]).unwrap();
        let error = reduce(
            &mut real,
            &mut scratch,
            ExecutionPolicy::Sequential,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        let extreme = Complex::new(f64::MAX, 0.0);
        let mut arena = [Complex::new(0.0, 0.0); 6];
        let mut rest = &mut arena[..];
        let mut scratch = Scratch::carve(&mut rest, 3, ExecutionPolicy::Sequential).unwrap();
        let mut complex = Matrix::from_vec(3, 3, alloc::vec![extreme; 9]).unwrap();
        let error = reduce(
            &mut complex,
            &mut scratch,
            ExecutionPolicy::Sequential,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }
}
