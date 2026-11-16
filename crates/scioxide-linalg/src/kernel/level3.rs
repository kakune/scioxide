//! Crate-private blocked level-3 operations built on the packed GEMM substrate.

struct TriangularScale<S> {
    side: MatrixSide,
    alpha: S,
}
pub(crate) struct RankUpdate<S> {
    pub(crate) triangle: Triangle,
    pub(crate) op: GemmOp,
    pub(crate) alpha: S,
    pub(crate) beta: S,
}
struct RankProduct<S> {
    triangle: Triangle,
    kind: super::RankKind,
    left_op: GemmOp,
    right_op: GemmOp,
    alpha: S,
    beta: S,
}

#[derive(Clone, Copy)]
struct Substitution {
    op: GemmOp,
    diagonal: DiagonalKind,
}

struct PreparedTriangular<R: scioxide_core::RealScalar> {
    action: TriangularAction,
    inputs: TriangularInputs<R>,
    blocks: DiagonalBlocks,
}

#[cfg(feature = "blas")]
use super::CANDIDATE;
use scioxide_core::{AlgebraicScalar, RealScalar};
use scioxide_core::{Dimensions, ExecutionPolicy, NumericScalar, SciOxideError};

use super::standard;
use crate::dense::{MatrixView, MatrixViewMut};
use crate::{Accuracy, KernelContext, MatrixSide};
use scioxide_core::DiagonalKind;

use super::{
    GemmConfig, GemmOp, GemmPlan, LinalgWorkspace, PackingPanel, SMALL_GEMM_DIMENSION, Triangle,
    apply_packed_block, flip_triangle, operated_get, pack_a, pack_b,
};

// Diagonal block order for blocked TRSM. It must not exceed any `KernelSpec::kc`
// (minimum 64, for complex scalars on the scalar backend): the trailing update
// packs the whole block's shared range into a single `kc`-deep panel, so there
// is no inner k-loop in the packed updates.
const TRSM_BLOCK_SIZE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiagonalBlocks {
    Scalar,
    Schur,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TriangularAction {
    Multiply,
    Solve,
}

/// Solve an upper Schur system in staging storage that the caller commits.
/// Real 2x2 diagonal blocks stay intact and use the shared complete-pivoting
/// small solver, including its exact-consistency policy. Scalar triangles use
/// ordinary TRSM. No provider receives a quasi-triangular operand.
pub(crate) fn schur_trsm_bounded<S: NumericScalar>(
    triangular: crate::FiniteMatrixView<'_, S>,
    rhs: MatrixViewMut<'_, S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let a = triangular.as_view();
    let context = context.with_accuracy(Accuracy::Standard);
    validate_trsm_dimensions(a, &rhs, MatrixSide::Left)?;
    if !(1..a.rows).any(|i| a.get_element(i, i - 1) != S::ZERO) {
        return trsm_bounded(
            crate::kernel::TriangularOperand {
                view: triangular,
                triangle: Triangle::Upper,
                op: GemmOp::None,
                diagonal: DiagonalKind::Explicit,
            },
            rhs,
            MatrixSide::Left,
            S::ONE,
            workspace,
            context,
        )
        .map(|_| ());
    }
    let rhs_maximum = crate::FiniteMatrixView::from_workspace(rhs.as_view()).maximum;
    let maximum = triangular.maximum.max(rhs_maximum);
    let range = bounded_triangular_range(a, DiagonalKind::Explicit, maximum)?;
    let inputs = TriangularInputs {
        range,
        matrix_maximum: triangular.maximum,
        matrix_minimum: S::Real::ZERO,
        rhs_maximum,
    };
    triangular_prepared(
        PreparedTriangular {
            action: TriangularAction::Solve,
            inputs,
            blocks: DiagonalBlocks::Schur,
        },
        crate::kernel::TriangularOperand {
            view: a,
            triangle: Triangle::Upper,
            op: GemmOp::None,
            diagonal: DiagonalKind::Explicit,
        },
        rhs,
        None,
        crate::kernel::TriangularUpdate {
            side: MatrixSide::Left,
            alpha: S::ONE,
            policy: ExecutionPolicy::Sequential,
        },
        workspace,
        context,
    )
    .map(|_| ())
}

pub(crate) fn syrk_with_context<'a, S: NumericScalar>(
    input: MatrixView<'_, S>,
    output: MatrixViewMut<'_, S>,
    update: crate::kernel::level3::RankUpdate<S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    let crate::kernel::level3::RankUpdate {
        triangle,
        op,
        alpha,
        beta,
    } = update;

    let (left_op, right_op) = match op {
        GemmOp::None => (GemmOp::None, GemmOp::Transpose),
        GemmOp::Transpose => (GemmOp::Transpose, GemmOp::None),
        GemmOp::Adjoint => {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("operation"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "SYRK accepts only no-transpose or transpose",
            ));
        }
    };
    rank_k(
        input,
        output,
        RankProduct {
            triangle,
            kind: super::RankKind::Symmetric,
            left_op,
            right_op,
            alpha,
            beta,
        },
        policy,
        workspace,
        context,
    )
}

pub(crate) fn herk_with_context<'a, S: NumericScalar>(
    input: MatrixView<'_, S>,
    output: MatrixViewMut<'_, S>,
    update: crate::kernel::level3::RankUpdate<S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    let crate::kernel::level3::RankUpdate {
        triangle,
        op,
        alpha,
        beta,
    } = update;

    let (left_op, right_op) = match op {
        GemmOp::None => (GemmOp::None, GemmOp::Adjoint),
        GemmOp::Adjoint => (GemmOp::Adjoint, GemmOp::None),
        GemmOp::Transpose => {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("operation"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "HERK accepts only no-transpose or adjoint",
            ));
        }
    };
    rank_k(
        input,
        output,
        RankProduct {
            triangle,
            kind: super::RankKind::Hermitian,
            left_op,
            right_op,
            alpha,
            beta,
        },
        policy,
        workspace,
        context,
    )
}

fn rank_k<'a, S: NumericScalar>(
    input: MatrixView<'_, S>,
    mut output: MatrixViewMut<'_, S>,
    product: RankProduct<S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    let RankProduct {
        triangle,
        kind,
        left_op,
        right_op,
        alpha,
        beta,
    } = product;

    let real_diagonal = alpha.imaginary_component() == S::Real::ZERO
        && beta.imaginary_component() == S::Real::ZERO
        && (beta == S::ZERO
            || (0..output.rows.min(output.cols))
                .all(|i| output.get_element(i, i).imaginary_component() == S::Real::ZERO));
    let context = context.into();
    let config = GemmConfig {
        left_op,
        right_op,
        alpha,
        beta,
        backend: None,
        triangle: Some(triangle),
        packed: false,
        prior: None,
        #[cfg(feature = "blas")]
        rank: Some(kind),
    };
    let left = super::operated_view(input, left_op);
    let right = super::operated_view(input, right_op);
    super::validate_dimensions(left.rows, left.cols, right.rows, right.cols, &output)?;
    crate::checked::validate_input(alpha, "alpha")?;
    crate::checked::validate_input(beta, "beta")?;
    context.execution().check()?;
    scioxide_core::backend::checked_product(
        &[left.rows, left.cols, right.cols, 2],
        scioxide_core::quantity!("gemm_work"),
    )?;
    if left.rows == 0 || right.cols == 0 {
        return Ok(());
    }
    let finite = input.try_finite_dispatched(context.dispatch())?;
    let operated = |op| match op {
        GemmOp::None => finite,
        GemmOp::Transpose => finite.transpose(),
        GemmOp::Adjoint => finite.adjoint(),
    };
    super::gemm_bounded(
        operated(left_op),
        operated(right_op),
        output.reborrow(),
        config,
        policy,
        workspace,
        context,
    )?;
    if real_diagonal && kind == super::RankKind::Hermitian {
        for index in 0..output.rows.min(output.cols) {
            let value = output.get_element(index, index);
            output.set_element(index, index, S::from_real(value.real_component()));
        }
    }
    Ok(())
}

pub(crate) fn triangular_with_context<'a, S: NumericScalar>(
    action: TriangularAction,
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    right_hand_side: MatrixViewMut<'_, S>,
    update: crate::kernel::TriangularUpdate<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    triangular_checked(
        action,
        operand,
        (right_hand_side, None),
        update,
        workspace,
        context.into(),
    )
}

/// Solves into `staged`, a copy of `original` that the caller commits. A substitution whose
/// growth leaves the ordinary range restarts from `original` in the binary-scaled
/// substitution instead of rejecting a representable solution.
pub(crate) fn triangular_solve_staged<'a, S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    staged: MatrixViewMut<'_, S>,
    original: MatrixView<'_, S>,
    update: crate::kernel::TriangularUpdate<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    triangular_checked(
        TriangularAction::Solve,
        operand,
        (staged, Some(original)),
        update,
        workspace,
        context.into(),
    )
}

fn triangular_checked<S: NumericScalar>(
    action: TriangularAction,
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    (right_hand_side, original): (MatrixViewMut<'_, S>, Option<MatrixView<'_, S>>),
    update: crate::kernel::TriangularUpdate<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;
    let crate::kernel::TriangularUpdate {
        side,
        alpha,
        policy,
    } = update;

    validate_trsm_dimensions(view, &right_hand_side, side)?;
    crate::checked::validate_input(alpha, "scale")?;
    context.execution().check()?;
    if right_hand_side.rows == 0 || right_hand_side.cols == 0 {
        return Ok(());
    }
    let inputs = prepare_triangular_inputs(view, &right_hand_side, triangle, op, diagonal, alpha)?;
    triangular_prepared(
        PreparedTriangular {
            action,
            inputs,
            blocks: DiagonalBlocks::Scalar,
        },
        crate::kernel::TriangularOperand {
            view,
            triangle,
            op,
            diagonal,
        },
        right_hand_side,
        original,
        crate::kernel::TriangularUpdate {
            side,
            alpha,
            policy,
        },
        workspace,
        context,
    )
    .map(|_| ())
}

/// Solve inside a validated composite workspace, without public input scans,
/// and return the largest component magnitude of the finite solution.
pub(crate) fn trsm_bounded<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<crate::FiniteMatrixView<'_, S>>,
    mut right_hand_side: MatrixViewMut<'_, S>,
    side: MatrixSide,
    alpha: S,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    validate_trsm_dimensions(view.as_view(), &right_hand_side, side)?;
    let rhs_maximum = standard::component_maximum(right_hand_side.as_view(), None);
    let maximum = rhs_maximum
        .max(view.maximum)
        .max(standard::magnitude(alpha));
    crate::checked::finite_candidate(S::from_real(maximum))?;
    let range = bounded_triangular_range(view.as_view(), diagonal, maximum)?;
    let inputs = TriangularInputs {
        range,
        matrix_maximum: view.maximum,
        matrix_minimum: S::Real::ZERO,
        rhs_maximum,
    };
    let solved = triangular_prepared(
        PreparedTriangular {
            action: TriangularAction::Solve,
            inputs,
            blocks: DiagonalBlocks::Scalar,
        },
        crate::kernel::TriangularOperand {
            view: view.as_view(),
            triangle,
            op,
            diagonal,
        },
        right_hand_side.reborrow(),
        None,
        crate::kernel::TriangularUpdate {
            side,
            alpha,
            policy: ExecutionPolicy::Sequential,
        },
        workspace,
        context,
    )?;
    Ok(solved.unwrap_or_else(|| standard::component_maximum(right_hand_side.as_view(), None)))
}

fn bounded_triangular_range<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    diagonal: DiagonalKind,
    maximum: S::Real,
) -> Result<TriangularRange, SciOxideError> {
    let limit = standard::exponent(S::Real::MAX)? / 4;
    let small_diagonal = diagonal == DiagonalKind::Explicit
        && (0..triangular.rows).any(|i| {
            standard::magnitude(triangular.get_element(i, i)) < S::Real::ONE.scalbn(-limit)
        });
    if maximum > S::Real::ONE.scalbn(limit) || small_diagonal {
        Ok(TriangularRange::Scaled)
    } else {
        Ok(TriangularRange::Ordinary)
    }
}

/// Runs a prepared triangular product or solve on `right_hand_side`.
///
/// When `original` holds the unmodified right-hand side, a solve whose ordinary
/// substitution leaves the range restarts from it in the binary-scaled substitution,
/// so growth inside the substitution does not reject a representable solution.
fn triangular_prepared<S: NumericScalar>(
    prepared: PreparedTriangular<S::Real>,
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    mut right_hand_side: MatrixViewMut<'_, S>,
    original: Option<MatrixView<'_, S>>,
    update: crate::kernel::TriangularUpdate<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<Option<S::Real>, SciOxideError> {
    let PreparedTriangular {
        action,
        inputs,
        blocks,
    } = prepared;

    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;
    let crate::kernel::TriangularUpdate {
        side,
        alpha,
        policy,
    } = update;

    #[cfg(not(feature = "blas"))]
    let _ = policy;
    let range = inputs.range;
    context.execution().check()?;

    let order = view.rows;
    if right_hand_side.rows == 0 || right_hand_side.cols == 0 {
        return Ok(Some(S::Real::ZERO));
    }
    // BLAS semantics: `alpha == 0` defines X = 0 without reading T, so a
    // singular factor is not an error on this path.
    if alpha == S::ZERO {
        right_hand_side.fill(S::ZERO);
        return Ok(Some(S::Real::ZERO));
    }
    // Singularity is rejected before any mutation, so the in-place solve
    // below cannot fail halfway through on a zero pivot.
    if action == TriangularAction::Solve && blocks == DiagonalBlocks::Scalar {
        validate_diagonal(view, op, diagonal)?;
    }

    #[cfg(feature = "blas")]
    if blocks == DiagonalBlocks::Scalar
        && let Some(dispatch) = context.blas()
    {
        let candidate_len = scioxide_core::backend::checked_product(
            &[right_hand_side.rows, right_hand_side.cols],
            scioxide_core::quantity!("trsm_candidate_elements"),
        )?;
        workspace.require_candidate(candidate_len)?;
        workspace.require_packed(
            scioxide_core::backend::checked_product(
                &[order, order],
                scioxide_core::quantity!("provider_triangular_operand"),
            )?,
            0,
        )?;
        let capabilities = dispatch.capabilities()?;
        let layout = crate::blas::dispatch::preferred_layout(&capabilities);
        for row in 0..right_hand_side.rows {
            for col in 0..right_hand_side.cols {
                workspace.scalar_mut(CANDIDATE)[crate::blas::dispatch::matrix_index(
                    layout,
                    right_hand_side.rows,
                    right_hand_side.cols,
                    row,
                    col,
                )] = right_hand_side.as_view().get_element(row, col);
            }
        }
        let crate::workspace::ProductParts {
            packed_a: packed,
            candidate,
            ..
        } = workspace.product_parts()?;
        if crate::blas::dispatch::triangular(crate::blas::dispatch::BlasTriangular {
            accuracy: context.accuracy(),
            triangular: view,
            op,
            packed,
            candidate: &mut candidate[..candidate_len],
            rows: right_hand_side.rows,
            cols: right_hand_side.cols,
            side: match side {
                MatrixSide::Left => crate::MatrixSide::Left,
                MatrixSide::Right => crate::MatrixSide::Right,
            },
            triangle: (triangle),
            diagonal: (diagonal),
            alpha,
            solve: action == TriangularAction::Solve,
            dispatch,
            capabilities: &capabilities,
            workers: super::provider_worker_bound::<S>(
                right_hand_side.rows,
                order,
                right_hand_side.cols,
                policy,
                context,
            )?,
        })? == crate::blas::dispatch::BlasRoute::Provider
        {
            for row in 0..right_hand_side.rows {
                for col in 0..right_hand_side.cols {
                    let value = workspace.scalar_mut(CANDIDATE)
                        [crate::blas::dispatch::matrix_index(
                            layout,
                            right_hand_side.rows,
                            right_hand_side.cols,
                            row,
                            col,
                        )];
                    right_hand_side.set_element(row, col, value);
                }
            }
            return Ok(None);
        }
    }

    if action == TriangularAction::Multiply {
        return triangular_multiply(
            crate::kernel::TriangularOperand {
                view,
                triangle,
                op,
                diagonal,
            },
            right_hand_side,
            crate::kernel::TriangularUpdate {
                side,
                alpha,
                policy,
            },
            inputs,
            workspace,
            context,
        )
        .map(|()| None);
    }

    if blocks == DiagonalBlocks::Schur && range == TriangularRange::Scaled {
        solve_schur_diagonal(view, &mut right_hand_side, 0, order)?;
        return check_solution(&right_hand_side).map(Some);
    }
    // Both tiers take the range-safe substitution at exponent extremes, where an
    // ordinary or exactly accumulated residual could leave the range although the
    // solution fits.
    if range == TriangularRange::Scaled {
        return scaled_solve(
            crate::kernel::TriangularOperand {
                view,
                triangle,
                op,
                diagonal,
            },
            &mut right_hand_side,
            side,
            alpha,
        )
        .map(|()| None);
    }
    let operand = crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    };
    match ordinary_solve(
        operand,
        &mut right_hand_side,
        (side, alpha),
        blocks,
        workspace,
        context,
    ) {
        Err(error)
            if blocks == DiagonalBlocks::Scalar
                && error.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability =>
        {
            let Some(original) = original else {
                return Err(error);
            };
            for row in 0..right_hand_side.rows {
                for col in 0..right_hand_side.cols {
                    right_hand_side.set_element(row, col, original.get_element(row, col));
                }
            }
            scaled_solve(operand, &mut right_hand_side, side, alpha).map(|()| None)
        }
        result => result,
    }
}

/// Ordinary-range substitution: small systems solve directly, larger ones by blocked
/// right-looking TRSM with packed updates. Growth inside the substitution can still leave
/// the range; the result is then rejected.
fn ordinary_solve<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    right_hand_side: &mut MatrixViewMut<'_, S>,
    (side, alpha): (MatrixSide, S),
    blocks: DiagonalBlocks,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<Option<S::Real>, SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;
    let order = view.rows;
    let mut right_hand_side = right_hand_side.reborrow();
    if context.accuracy() == Accuracy::Exact && blocks == DiagonalBlocks::Scalar {
        exact_solve(operand, &mut right_hand_side, side, alpha)?;
        return check_solution(&right_hand_side).map(Some);
    }
    let effective_triangle = match op {
        GemmOp::None => triangle,
        GemmOp::Transpose | GemmOp::Adjoint => flip_triangle(triangle),
    };
    if order <= SMALL_GEMM_DIMENSION
        && right_hand_side.rows <= SMALL_GEMM_DIMENSION
        && right_hand_side.cols <= SMALL_GEMM_DIMENSION
    {
        scale_matrix(&mut right_hand_side, alpha);
        if blocks == DiagonalBlocks::Schur {
            solve_schur_diagonal(view, &mut right_hand_side, 0, order)?;
            return check_solution(&right_hand_side).map(Some);
        }
        match (side, effective_triangle) {
            (MatrixSide::Left, triangle) => solve_left_diagonal(
                crate::kernel::TriangularOperand {
                    view,
                    triangle,
                    op,
                    diagonal,
                },
                &mut right_hand_side,
                0..order,
            )?,
            (MatrixSide::Right, triangle) => solve_right_diagonal(
                crate::kernel::TriangularOperand {
                    view,
                    triangle,
                    op,
                    diagonal,
                },
                &mut right_hand_side,
                0..order,
            )?,
        }
        return check_solution(&right_hand_side).map(Some);
    }

    // Blocked right-looking TRSM: solve one diagonal block with substitution,
    // then apply its contribution to all remaining rows/columns as a packed
    // GEMM update. The update depth is at most one diagonal block, which is
    // why the plan is sized with `shared = min(order, TRSM_BLOCK_SIZE)`.
    let dispatch = context.dispatch();
    let shared = order.min(TRSM_BLOCK_SIZE);
    let plan = GemmPlan::new::<S>(
        right_hand_side.rows,
        shared,
        right_hand_side.cols,
        ExecutionPolicy::Sequential,
        context.execution(),
        dispatch,
    )?;
    workspace.require_gemm(&plan)?;
    scale_matrix(&mut right_hand_side, alpha);

    match (side, effective_triangle) {
        (MatrixSide::Left, Triangle::Lower) => {
            left_lower(view, &mut right_hand_side, op, diagonal, &plan, workspace)?
        }
        (MatrixSide::Left, Triangle::Upper) => left_upper(
            view,
            &mut right_hand_side,
            Substitution { op, diagonal },
            &plan,
            workspace,
            blocks,
        )?,
        (MatrixSide::Right, Triangle::Lower) => {
            right_lower(view, &mut right_hand_side, op, diagonal, &plan, workspace)?
        }
        (MatrixSide::Right, Triangle::Upper) => {
            right_upper(view, &mut right_hand_side, op, diagonal, &plan, workspace)?
        }
    }
    check_solution(&right_hand_side).map(Some)
}

/// Reject a non-finite solution and return its largest component magnitude.
fn check_solution<S: NumericScalar>(rhs: &MatrixViewMut<'_, S>) -> Result<S::Real, SciOxideError> {
    standard::finite_component_maximum(rhs.as_view())
}

/// Binary-scaled substitution for exponent extremes. Each row restores only
/// its final quotient; the inner reduction uses ordinary finite arithmetic.
fn scaled_solve<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    rhs: &mut MatrixViewMut<'_, S>,
    side: MatrixSide,
    alpha: S,
) -> Result<(), SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    let n = view.rows;
    let triangle = if op == GemmOp::None {
        triangle
    } else {
        flip_triangle(triangle)
    };
    let forward = matches!(
        (side, triangle),
        (MatrixSide::Left, Triangle::Lower) | (MatrixSide::Right, Triangle::Upper)
    );
    let vectors = match side {
        MatrixSide::Left => rhs.cols,
        MatrixSide::Right => rhs.rows,
    };
    let ae = standard::exponent(standard::magnitude(alpha))?;
    let alpha = standard::scale(alpha, -ae);
    for step in 0..n {
        let index = if forward { step } else { n - 1 - step };
        let known = if forward { 0..index } else { index + 1..n };
        let coefficient = |inner| match side {
            MatrixSide::Left => operated_get(view, op, index, inner),
            MatrixSide::Right => operated_get(view, op, inner, index),
        };
        let maximum = known.clone().fold(S::Real::ZERO, |bound, inner| {
            bound.max(standard::magnitude(coefficient(inner)))
        });
        let te = standard::exponent(maximum)?;
        let divisor = if diagonal == DiagonalKind::Unit {
            S::ONE
        } else {
            operated_get(view, op, index, index)
        };
        let de = standard::exponent(standard::magnitude(divisor))?;
        let divisor = standard::scale(divisor, -de);
        let growth = usize::BITS as i32 - known.len().max(1).leading_zeros() as i32 + 3;
        for vector in 0..vectors {
            let position = |inner| match side {
                MatrixSide::Left => (inner, vector),
                MatrixSide::Right => (vector, inner),
            };
            let (row, col) = position(index);
            let prior = rhs.get_element(row, col);
            let be = standard::exponent(standard::magnitude(prior))?;
            let xmax = known.clone().fold(S::Real::ZERO, |bound, inner| {
                let (r, c) = position(inner);
                if coefficient(inner) == S::ZERO {
                    bound
                } else {
                    bound.max(standard::magnitude(rhs.get_element(r, c)))
                }
            });
            let xe = standard::exponent(xmax)?;
            let restore = if maximum == S::Real::ZERO {
                ae + be + 3
            } else {
                (ae + be + 3).max(te + xe + growth)
            };
            let mut residual =
                standard::scale(alpha * standard::scale(prior, -be), ae + be - restore);
            for inner in known.clone() {
                if coefficient(inner) == S::ZERO {
                    continue;
                }
                let (r, c) = position(inner);
                let left = standard::scale(coefficient(inner), -te);
                let right = standard::scale(
                    standard::scale(rhs.get_element(r, c), -xe),
                    te + xe - restore,
                );
                residual = residual - left * right;
            }
            let value = standard::restore(residual / divisor, restore - de)?;
            rhs.set_element(row, col, value);
        }
    }
    Ok(())
}

/// `Exact` substitution, the same on every platform. Each residual
/// `α bᵢ − Σⱼ tᵢⱼ xⱼ` over all solved entries is accumulated exactly and rounded once,
/// then divided by the diagonal entry with one more rounding.
fn exact_solve<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    rhs: &mut MatrixViewMut<'_, S>,
    side: MatrixSide,
    alpha: S,
) -> Result<(), SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    let n = view.rows;
    let triangle = if op == GemmOp::None {
        triangle
    } else {
        flip_triangle(triangle)
    };
    let forward = matches!(
        (side, triangle),
        (MatrixSide::Left, Triangle::Lower) | (MatrixSide::Right, Triangle::Upper)
    );
    let vectors = match side {
        MatrixSide::Left => rhs.cols,
        MatrixSide::Right => rhs.rows,
    };
    for step in 0..n {
        let index = if forward { step } else { n - 1 - step };
        let known = if forward { 0..index } else { index + 1..n };
        let coefficient = |inner| match side {
            MatrixSide::Left => operated_get(view, op, index, inner),
            MatrixSide::Right => operated_get(view, op, inner, index),
        };
        for vector in 0..vectors {
            let position = |inner| match side {
                MatrixSide::Left => (inner, vector),
                MatrixSide::Right => (vector, inner),
            };
            let (row, col) = position(index);
            let mut residual = scioxide_core::ProductSum::new();
            residual.add_product(alpha, rhs.get_element(row, col))?;
            for inner in known.clone() {
                let value = coefficient(inner);
                if value != S::ZERO {
                    let (r, c) = position(inner);
                    residual.add_product(-value, rhs.get_element(r, c))?;
                }
            }
            let residual = residual.finish()?;
            let value = if diagonal == DiagonalKind::Unit {
                residual
            } else {
                crate::checked::div(residual, operated_get(view, op, index, index))?
            };
            rhs.set_element(row, col, value);
        }
    }
    Ok(())
}

fn validate_trsm_dimensions<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    right_hand_side: &MatrixViewMut<'_, S>,
    side: MatrixSide,
) -> Result<(), SciOxideError> {
    if triangular.rows != triangular.cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(triangular.rows, triangular.rows),
            Dimensions::two(triangular.rows, triangular.cols),
        ));
    }
    let actual = match side {
        MatrixSide::Left => right_hand_side.rows,
        MatrixSide::Right => right_hand_side.cols,
    };
    if actual != triangular.rows {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(triangular.rows),
            Dimensions::one(actual),
        ));
    }
    Ok(())
}

fn validate_diagonal<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    op: GemmOp,
    diagonal: DiagonalKind,
) -> Result<(), SciOxideError> {
    if diagonal == DiagonalKind::Unit {
        return Ok(());
    }
    (0..triangular.rows).try_for_each(|index| {
        let value = operated_get(triangular, op, index, index);
        if value == S::ZERO {
            return Err(scioxide_core::SciOxideError::singular(None));
        }
        Ok(())
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TriangularRange {
    Ordinary,
    Scaled,
}

/// One input scan supplies substitution range and product scaling bounds.
struct TriangularInputs<R: RealScalar> {
    range: TriangularRange,
    matrix_maximum: R,
    /// Smallest nonzero component magnitude of the stored triangle without a unit diagonal,
    /// zero when there is none. Only products read it, to decide whether folding `alpha`
    /// into the triangle needs a binary frame.
    matrix_minimum: R,
    rhs_maximum: R,
}

fn prepare_triangular_inputs<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    right_hand_side: &MatrixViewMut<'_, S>,
    triangle: Triangle,
    op: GemmOp,
    diagonal: DiagonalKind,
    alpha: S,
) -> Result<TriangularInputs<S::Real>, SciOxideError> {
    if alpha == S::ZERO {
        return Ok(TriangularInputs {
            range: TriangularRange::Ordinary,
            matrix_maximum: S::Real::ZERO,
            matrix_minimum: S::Real::ZERO,
            rhs_maximum: S::Real::ZERO,
        });
    }
    let effective_triangle = if op == GemmOp::None {
        triangle
    } else {
        flip_triangle(triangle)
    };
    let mut entries = (0..triangular.rows).flat_map(|row| {
        (0..triangular.cols)
            .filter(move |&col| {
                super::triangle_contains(effective_triangle, row, col)
                    && !(diagonal == DiagonalKind::Unit && row == col)
            })
            .map(move |col| operated_get(triangular, op, row, col))
    });
    let mut rhs = (0..right_hand_side.rows).flat_map(|row| {
        (0..right_hand_side.cols).map(move |col| right_hand_side.get_element(row, col))
    });
    let bound = |(maximum, minimum): (S::Real, S::Real), value: S| {
        crate::checked::validate_input(value, "triangular entry")?;
        let magnitude = standard::magnitude(value);
        Ok::<_, SciOxideError>((
            maximum.max(magnitude),
            if magnitude == S::Real::ZERO {
                minimum
            } else {
                minimum.min(magnitude)
            },
        ))
    };
    let initial = (S::Real::ZERO, S::Real::MAX);
    let (matrix_maximum, matrix_minimum) = entries.try_fold(initial, bound)?;
    let (rhs_maximum, rhs_minimum) = rhs.try_fold(initial, bound)?;
    let (maximum, minimum) = bound(
        (
            matrix_maximum.max(rhs_maximum),
            matrix_minimum.min(rhs_minimum),
        ),
        alpha,
    )?;
    let limit = standard::exponent(S::Real::MAX)? / 4;
    Ok(TriangularInputs {
        range: if maximum > S::Real::ONE.scalbn(limit) || minimum < S::Real::ONE.scalbn(-limit) {
            TriangularRange::Scaled
        } else {
            TriangularRange::Ordinary
        },
        matrix_maximum: if diagonal == DiagonalKind::Unit {
            matrix_maximum.max(S::Real::ONE)
        } else {
            matrix_maximum
        },
        matrix_minimum: if matrix_maximum == S::Real::ZERO {
            S::Real::ZERO
        } else {
            matrix_minimum
        },
        rhs_maximum,
    })
}

fn scale_matrix<S: NumericScalar>(matrix: &mut MatrixViewMut<'_, S>, alpha: S) {
    if alpha == S::ONE {
        return;
    }
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            let value = alpha * matrix.get_element(row, col);
            matrix.set_element(row, col, value);
        }
    }
}

fn left_lower<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    op: GemmOp,
    diagonal: DiagonalKind,
    plan: &GemmPlan,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let mut start = 0;
    while start < triangular.rows {
        let end = (start + TRSM_BLOCK_SIZE).min(triangular.rows);
        solve_left_diagonal(
            crate::kernel::TriangularOperand {
                view: triangular,
                triangle: Triangle::Lower,
                op,
                diagonal,
            },
            rhs,
            start..end,
        )?;
        if end < triangular.rows {
            packed_left_update(
                crate::kernel::MatrixOperand {
                    view: triangular,
                    op,
                },
                rhs,
                end..triangular.rows,
                start..end,
                plan,
                workspace,
            )?;
        }
        start = end;
    }
    Ok(())
}

fn left_upper<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    substitution: Substitution,
    plan: &GemmPlan,
    workspace: &mut LinalgWorkspace<'_, S>,
    blocks: DiagonalBlocks,
) -> Result<(), SciOxideError> {
    let Substitution { op, diagonal } = substitution;

    let mut end = triangular.rows;
    while end > 0 {
        let mut start = end.saturating_sub(TRSM_BLOCK_SIZE);
        // Shrink the panel rather than exceeding reserved packing depth.
        if blocks == DiagonalBlocks::Schur
            && start > 0
            && triangular.get_element(start, start - 1) != S::ZERO
        {
            start += 1;
        }
        if blocks == DiagonalBlocks::Schur {
            solve_schur_diagonal(triangular, rhs, start, end)?;
        } else {
            solve_left_diagonal(
                crate::kernel::TriangularOperand {
                    view: triangular,
                    triangle: Triangle::Upper,
                    op,
                    diagonal,
                },
                rhs,
                start..end,
            )?;
        }
        if start > 0 {
            packed_left_update(
                crate::kernel::MatrixOperand {
                    view: triangular,
                    op,
                },
                rhs,
                0..start,
                start..end,
                plan,
                workspace,
            )?;
        }
        end = start;
    }
    Ok(())
}

/// Diagonal Schur panel substitution in extended range. Off-panel updates
/// belong to the same packed TRSM driver as scalar triangles.
fn solve_schur_diagonal<S: NumericScalar>(
    a: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    start: usize,
    end: usize,
) -> Result<(), SciOxideError> {
    use crate::kernel::frames::{extended, scalar};
    use crate::kernel::small::{Singularity, small};
    use scioxide_core::ScaledValue;
    let mut last = end;
    while last > start {
        let first = if last - start >= 2 && a.get_element(last - 1, last - 2) != S::ZERO {
            last - 2
        } else {
            last - 1
        };
        let d = last - first;
        if d == 1 && a.get_element(first, first) == S::ZERO {
            return Err(SciOxideError::singular(None));
        }
        for col in 0..rhs.cols {
            let mut coefficients = [ScaledValue::ZERO; 4];
            let mut values = [ScaledValue::ZERO; 2];
            for row in 0..d {
                let mut value = extended(rhs.get_element(first + row, col))?;
                for k in last..end {
                    value = value.sub(
                        extended(a.get_element(first + row, k))?
                            .mul(extended(rhs.get_element(k, col))?)?,
                    )?;
                }
                values[row] = value;
                for j in 0..d {
                    coefficients[row * 2 + j] = extended(a.get_element(first + row, first + j))?;
                }
            }
            if d == 1 {
                values[0] = values[0].div(coefficients[0])?;
            } else if !small(&mut coefficients, &mut values, 2, Singularity::Consistent)? {
                return Err(SciOxideError::singular(None));
            }
            for (row, &value) in values[..d].iter().enumerate() {
                rhs.set_element(first + row, col, scalar(value)?);
            }
        }
        last = first;
    }
    Ok(())
}

fn solve_left_diagonal<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    rhs: &mut MatrixViewMut<'_, S>,
    block: core::ops::Range<usize>,
) -> Result<(), SciOxideError> {
    let core::ops::Range { start, end } = block;

    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    match triangle {
        Triangle::Lower => {
            let mut row = start;
            while row < end {
                solve_left_row(view, rhs, op, diagonal, start..row, row)?;
                row += 1;
            }
        }
        Triangle::Upper => {
            let mut row = end;
            while row > start {
                row -= 1;
                solve_left_row(view, rhs, op, diagonal, row + 1..end, row)?;
            }
        }
    }
    Ok(())
}

fn solve_left_row<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    op: GemmOp,
    diagonal: DiagonalKind,
    known: core::ops::Range<usize>,
    row: usize,
) -> Result<(), SciOxideError> {
    let diagonal_value = operated_get(triangular, op, row, row);
    let unit = diagonal == DiagonalKind::Unit;
    if rhs.col_stride == 1 {
        // Apply each solved row across adjacent right-hand sides. Keeping the
        // reduction order per entry preserves reproducibility while exposing
        // independent columns to the compiler's vectorizer.
        // Finite separated subtraction cannot create a new negative zero
        // from a row without one. Skip neutral real zero coefficients only
        // when their IEEE zero signs are unobservable in this target row.
        let skip_zeros = S::IS_REAL
            && !known.is_empty()
            && rhs.data[row * rhs.row_stride..][..rhs.cols]
                .iter()
                .all(|&value| value != S::ZERO || !value.real_component().is_sign_negative());
        for inner in known {
            let coefficient = operated_get(triangular, op, row, inner);
            if skip_zeros && coefficient == S::ZERO {
                continue;
            }
            let (target, source) = if inner < row {
                let (before, after) = crate::workspace::split_mut(rhs.data, row * rhs.row_stride)?;
                (
                    &mut after[..rhs.cols],
                    &before[inner * rhs.row_stride..][..rhs.cols],
                )
            } else {
                let (before, after) =
                    crate::workspace::split_mut(rhs.data, inner * rhs.row_stride)?;
                (
                    &mut before[row * rhs.row_stride..][..rhs.cols],
                    &after[..rhs.cols],
                )
            };
            for (value, &solved) in target.iter_mut().zip(source) {
                *value = *value - coefficient * solved;
            }
        }
        if !unit {
            for value in &mut rhs.data[row * rhs.row_stride..][..rhs.cols] {
                *value = *value / diagonal_value;
            }
        }
        return Ok(());
    }
    let mut col = 0;
    while col < rhs.cols {
        let value = known
            .clone()
            .fold(rhs.get_element(row, col), |value, inner| {
                value - operated_get(triangular, op, row, inner) * rhs.get_element(inner, col)
            });
        let value = if unit { value } else { value / diagonal_value };
        rhs.set_element(row, col, value);
        col += 1;
    }
    Ok(())
}

fn right_upper<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    op: GemmOp,
    diagonal: DiagonalKind,
    plan: &GemmPlan,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let mut start = 0;
    while start < triangular.rows {
        let end = (start + TRSM_BLOCK_SIZE).min(triangular.rows);
        solve_right_diagonal(
            crate::kernel::TriangularOperand {
                view: triangular,
                triangle: Triangle::Upper,
                op,
                diagonal,
            },
            rhs,
            start..end,
        )?;
        if end < triangular.rows {
            packed_right_update(
                crate::kernel::MatrixOperand {
                    view: triangular,
                    op,
                },
                rhs,
                0..rhs.rows,
                end..triangular.rows,
                start..end,
                plan,
                workspace,
            )?;
        }
        start = end;
    }
    Ok(())
}

fn right_lower<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    op: GemmOp,
    diagonal: DiagonalKind,
    plan: &GemmPlan,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let mut end = triangular.rows;
    while end > 0 {
        let start = end.saturating_sub(TRSM_BLOCK_SIZE);
        solve_right_diagonal(
            crate::kernel::TriangularOperand {
                view: triangular,
                triangle: Triangle::Lower,
                op,
                diagonal,
            },
            rhs,
            start..end,
        )?;
        if start > 0 {
            packed_right_update(
                crate::kernel::MatrixOperand {
                    view: triangular,
                    op,
                },
                rhs,
                0..rhs.rows,
                0..start,
                start..end,
                plan,
                workspace,
            )?;
        }
        end = start;
    }
    Ok(())
}

fn solve_right_diagonal<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    rhs: &mut MatrixViewMut<'_, S>,
    block: core::ops::Range<usize>,
) -> Result<(), SciOxideError> {
    let core::ops::Range { start, end } = block;

    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    if triangle == Triangle::Upper && rhs.col_stride == 1 && end - start <= RIGHT_UPPER_WIDTH {
        return solve_right_upper_rows(view, rhs, op, diagonal, start..end);
    }
    // Complete one row while its known values remain contiguous and hot.
    for row in 0..rhs.rows {
        match triangle {
            Triangle::Upper => {
                let mut col = start;
                while col < end {
                    solve_right_entry(
                        view,
                        rhs,
                        row,
                        Substitution { op, diagonal },
                        start..col,
                        col,
                    )?;
                    col += 1;
                }
            }
            Triangle::Lower => {
                let mut col = end;
                while col > start {
                    col -= 1;
                    solve_right_entry(
                        view,
                        rhs,
                        row,
                        Substitution { op, diagonal },
                        col + 1..end,
                        col,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Rows substituted together by [`solve_right_upper_rows`].
const ROW_LANES: usize = 8;

/// Widest diagonal block of [`solve_right_upper_rows`]: blocked solves and the
/// direct small-problem path.
const RIGHT_UPPER_WIDTH: usize = if TRSM_BLOCK_SIZE > SMALL_GEMM_DIMENSION {
    TRSM_BLOCK_SIZE
} else {
    SMALL_GEMM_DIMENSION
};

/// Standard `X U = B` on one upper diagonal block for contiguous rows.
///
/// The block's operated coefficients are gathered once. Rows run the
/// column-oriented (AXPY) substitution: solved `x_j` is subtracted from every
/// later entry in ascending `j`, and each entry is divided by its diagonal
/// when reached. Every entry sees exactly the subtractions and the division
/// of the entrywise fold in the same order, so results are bitwise identical.
/// Groups of [`ROW_LANES`] rows are transposed into a stack tile so that each
/// step updates independent rows in vector lanes.
fn solve_right_upper_rows<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    op: GemmOp,
    diagonal: DiagonalKind,
    block: core::ops::Range<usize>,
) -> Result<(), SciOxideError> {
    const WIDTH: usize = RIGHT_UPPER_WIDTH;
    let width = block.len();
    let mut coefficients = [S::ZERO; WIDTH * WIDTH];
    for (j, row) in coefficients.chunks_exact_mut(WIDTH).take(width).enumerate() {
        for (k, coefficient) in row.iter_mut().enumerate().take(width).skip(j) {
            *coefficient = operated_get(triangular, op, block.start + j, block.start + k);
        }
    }
    let coefficients = &coefficients;
    let unit = diagonal == DiagonalKind::Unit;
    let row_stride = rhs.row_stride;
    let entries = |row: usize| row * row_stride + block.start..row * row_stride + block.end;
    let grouped = rhs.rows - rhs.rows % ROW_LANES;
    let mut tile = [[S::ZERO; ROW_LANES]; WIDTH];
    for first in (0..grouped).step_by(ROW_LANES) {
        for lane in 0..ROW_LANES {
            for (column, &value) in tile.iter_mut().zip(&rhs.data[entries(first + lane)]) {
                column[lane] = value;
            }
        }
        for j in 0..width {
            let row = &coefficients[j * WIDTH..(j + 1) * WIDTH];
            let (solved, later) = tile.split_at_mut(j + 1);
            let solved = &mut solved[j];
            if !unit {
                solved.iter_mut().for_each(|value| *value = *value / row[j]);
            }
            for (column, &coefficient) in later[..width - j - 1].iter_mut().zip(&row[j + 1..]) {
                column
                    .iter_mut()
                    .zip(solved.iter())
                    .for_each(|(value, &solved)| *value = *value - solved * coefficient);
            }
        }
        for lane in 0..ROW_LANES {
            for (value, column) in rhs.data[entries(first + lane)].iter_mut().zip(&tile) {
                *value = column[lane];
            }
        }
    }
    for row in grouped..rhs.rows {
        let values = &mut rhs.data[entries(row)];
        for j in 0..width {
            let row = &coefficients[j * WIDTH..(j + 1) * WIDTH];
            if !unit {
                values[j] = values[j] / row[j];
            }
            let (solved, later) = values.split_at_mut(j + 1);
            let solved = solved[j];
            later
                .iter_mut()
                .zip(&row[j + 1..width])
                .for_each(|(value, &coefficient)| *value = *value - solved * coefficient);
        }
    }
    Ok(())
}

fn solve_right_entry<S: NumericScalar>(
    triangular: MatrixView<'_, S>,
    rhs: &mut MatrixViewMut<'_, S>,
    row: usize,
    substitution: Substitution,
    known: core::ops::Range<usize>,
    col: usize,
) -> Result<(), SciOxideError> {
    let Substitution { op, diagonal } = substitution;

    let diagonal_value = operated_get(triangular, op, col, col);
    let unit = diagonal == DiagonalKind::Unit;
    let value = known.fold(rhs.get_element(row, col), |value, inner| {
        value - rhs.get_element(row, inner) * operated_get(triangular, op, inner, col)
    });
    let value = if unit { value } else { value / diagonal_value };
    rhs.set_element(row, col, value);
    Ok(())
}

fn packed_left_update<S: NumericScalar>(
    operand: crate::kernel::MatrixOperand<MatrixView<'_, S>>,
    rhs: &mut MatrixViewMut<'_, S>,
    rows: core::ops::Range<usize>,
    shared: core::ops::Range<usize>,
    plan: &GemmPlan,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let crate::kernel::MatrixOperand { view, op } = operand;

    let core::ops::Range {
        start: shared_start,
        end: shared_end,
    } = shared;

    let core::ops::Range {
        start: row_start,
        end: row_end,
    } = rows;

    // `B[rows, :] -= T[rows, shared] * B[shared, :]`. The solved rows
    // `shared_start..shared_end` are disjoint from the updated rows, so packing
    // them once per column block and then overwriting the target rows in place
    // is safe. The minus sign is folded into packed A (SIMD leaves ignore
    // alpha), and `beta = 1` accumulates into the existing right-hand side.
    let spec = plan.spec;
    let shared = shared_end - shared_start;
    let microkernel = super::arch::select_microkernel::<S>(plan.dispatch);
    let crate::workspace::ProductParts {
        packed_a: a,
        packed_b: b,
        ..
    } = workspace.product_parts()?;
    let packed_a = &mut a[..plan.packed_a_per_worker];
    let packed_b = &mut b[..plan.packed_b_per_worker];
    let mut col = 0;
    while col < rhs.cols {
        let cols = (rhs.cols - col).min(spec.nc);
        pack_b(
            rhs.as_view(),
            GemmOp::None,
            PackingPanel::new(col, shared_start, cols, shared, spec.nr),
            packed_b,
        );
        let mut row = row_start;
        while row < row_end {
            let rows = (row_end - row).min(spec.mc);
            pack_a(
                view,
                op,
                PackingPanel::new(row, shared_start, rows, shared, spec.mr),
                -S::ONE,
                packed_a,
            );
            apply_packed_block(
                packed_a,
                packed_b,
                rhs,
                crate::kernel::PackedBlock {
                    row_start: row,
                    col_start: col,
                    rows,
                    cols,
                    shared,
                    global_row_start: row,
                },
                crate::kernel::PackedKernel {
                    beta: S::ONE,
                    spec,
                    microkernel,
                    triangle: None,
                },
            );
            row += spec.mc;
        }
        col += spec.nc;
    }
    Ok(())
}

fn packed_right_update<S: NumericScalar>(
    operand: crate::kernel::MatrixOperand<MatrixView<'_, S>>,
    rhs: &mut MatrixViewMut<'_, S>,
    rows: core::ops::Range<usize>,
    cols: core::ops::Range<usize>,
    shared: core::ops::Range<usize>,
    plan: &GemmPlan,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let crate::kernel::MatrixOperand { view, op } = operand;

    let core::ops::Range {
        start: shared_start,
        end: shared_end,
    } = shared;

    let core::ops::Range {
        start: col_start,
        end: col_end,
    } = cols;

    let core::ops::Range {
        start: row_start,
        end: row_end,
    } = rows;

    // `B[:, cols] -= B[:, shared] * T[shared, cols]`: the transpose-side
    // analogue of `packed_left_update`. Here B supplies the A operand and the
    // solved columns are disjoint from the updated ones.
    let spec = plan.spec;
    let shared = shared_end - shared_start;
    let microkernel = super::arch::select_microkernel::<S>(plan.dispatch);
    let crate::workspace::ProductParts {
        packed_a: a,
        packed_b: b,
        ..
    } = workspace.product_parts()?;
    let packed_a = &mut a[..plan.packed_a_per_worker];
    let packed_b = &mut b[..plan.packed_b_per_worker];
    let mut col = col_start;
    while col < col_end {
        let cols = (col_end - col).min(spec.nc);
        pack_b(
            view,
            op,
            PackingPanel::new(col, shared_start, cols, shared, spec.nr),
            packed_b,
        );
        let mut row = row_start;
        while row < row_end {
            let rows = (row_end - row).min(spec.mc);
            pack_a(
                rhs.as_view(),
                GemmOp::None,
                PackingPanel::new(row, shared_start, rows, shared, spec.mr),
                -S::ONE,
                packed_a,
            );
            apply_packed_block(
                packed_a,
                packed_b,
                rhs,
                crate::kernel::PackedBlock {
                    row_start: row,
                    col_start: col,
                    rows,
                    cols,
                    shared,
                    global_row_start: row,
                },
                crate::kernel::PackedKernel {
                    beta: S::ONE,
                    spec,
                    microkernel,
                    triangle: None,
                },
            );
            row += spec.mc;
        }
        col += spec.nc;
    }
    Ok(())
}

/// Accumulate only the stored triangle; no materialized triangle or full GEMM.
fn triangular_multiply<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    mut b: MatrixViewMut<'_, S>,
    update: crate::kernel::TriangularUpdate<S>,
    inputs: TriangularInputs<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;
    let crate::kernel::TriangularUpdate {
        side,
        alpha,
        policy,
    } = update;

    let n = view.rows();
    let effective = if op == GemmOp::None {
        triangle
    } else {
        flip_triangle(triangle)
    };
    let entry = |row, col| {
        if row == col && diagonal == DiagonalKind::Unit {
            S::ONE
        } else {
            operated_get(view, op, row, col)
        }
    };
    let frame = standard::AffineProduct::new(
        alpha,
        (inputs.matrix_maximum, inputs.matrix_minimum),
        inputs.rhs_maximum,
        n,
        S::ZERO,
        S::Real::ZERO,
    )?;
    let len = scioxide_core::backend::checked_product(
        &[b.rows(), b.cols()],
        scioxide_core::quantity!("triangular_product_output"),
    )?;
    workspace.require_candidate(len)?;
    let candidate = &mut workspace.scalar_mut(super::CANDIDATE)[..len];
    let plan = GemmPlan::new::<S>(
        b.rows(),
        n,
        b.cols(),
        policy,
        context.execution(),
        context.dispatch(),
    )?;
    let workers = core::num::NonZeroUsize::new(plan.execution.worker_count())
        .unwrap_or(core::num::NonZeroUsize::MIN);
    let policy = if workers.get() == 1 {
        ExecutionPolicy::Sequential
    } else {
        ExecutionPolicy::Parallel(workers)
    };
    let rhs = b.as_view();
    crate::kernel::execution::execute_chunks(
        candidate,
        b.cols(),
        policy,
        context.execution(),
        |first_row, values| {
            if context.accuracy() == Accuracy::Standard && frame.is_unscaled() {
                let rows = values.len() / rhs.cols();
                return packed_triangular_product(
                    crate::kernel::TriangularOperand {
                        view,
                        triangle: effective,
                        op,
                        diagonal,
                    },
                    rhs,
                    MatrixViewMut::from_slice(rows, rhs.cols(), values)?,
                    first_row,
                    TriangularScale { side, alpha },
                    plan.spec,
                    context,
                );
            }

            for (local_row, row_values) in values.chunks_exact_mut(rhs.cols()).enumerate() {
                let row = first_row + local_row;
                context.execution().check()?;
                for (col, output) in row_values.iter_mut().enumerate() {
                    let range = match (side, effective) {
                        (MatrixSide::Left, Triangle::Lower) => 0..row + 1,
                        (MatrixSide::Left, Triangle::Upper) => row..n,
                        (MatrixSide::Right, Triangle::Lower) => col..n,
                        (MatrixSide::Right, Triangle::Upper) => 0..col + 1,
                    };
                    let products = range.map(|k| match side {
                        MatrixSide::Left => (entry(row, k), rhs.get_element(k, col)),
                        MatrixSide::Right => (entry(k, col), rhs.get_element(row, k)),
                    });
                    *output = if context.accuracy() == Accuracy::Standard {
                        frame.apply(S::ZERO, products)?
                    } else {
                        let mut sum = scioxide_core::ProductSum::new();
                        for (view, b) in products {
                            sum.add_triple_product(alpha, view, b)?;
                        }
                        sum.finish()?
                    };
                }
            }
            Ok(())
        },
    )?;
    for row in 0..b.rows() {
        for col in 0..b.cols() {
            b.set_element(row, col, candidate[row * b.cols() + col]);
        }
    }
    Ok(())
}

/// TRMM uses the same packed leaves as blocked TRSM. Bounded stack panels
/// cover the largest microtile (six rows, 32 columns); diagonal panels alone
/// mask triangular entries. No whole triangle is materialized.
fn packed_triangular_product<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
    rhs: MatrixView<'_, S>,
    mut output: MatrixViewMut<'_, S>,
    first_row: usize,
    scale: TriangularScale<S>,
    spec: super::KernelSpec,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let TriangularScale { side, alpha } = scale;

    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    // A 32-deep panel amortizes leaf admission, within every packed kernel kc.
    const DEPTH: usize = SMALL_GEMM_DIMENSION;
    let mut a = [S::ZERO; DEPTH * 8];
    let mut b = [S::ZERO; DEPTH * 32];
    let b_stride = super::PanelSide::Right.depth_stride::<S>(spec.nr);
    debug_assert!(spec.mr <= 8 && b_stride <= 32);
    let microkernel = super::arch::select_microkernel::<S>(context.dispatch());
    let n = view.rows();
    output.fill(S::ZERO);
    let value = |row, col| {
        if row == col && diagonal == DiagonalKind::Unit {
            S::ONE
        } else {
            operated_get(view, op, row, col)
        }
    };
    for col in (0..output.cols()).step_by(spec.nr) {
        let cols = spec.nr.min(output.cols() - col);
        for shared in (0..n).step_by(DEPTH) {
            context.execution().check()?;
            let mut begin = shared;
            let mut end = shared + (n - shared).min(DEPTH);
            match (side, triangle) {
                (MatrixSide::Left, Triangle::Lower) => end = end.min(first_row + output.rows()),
                (MatrixSide::Left, Triangle::Upper) => begin = begin.max(first_row),
                (MatrixSide::Right, Triangle::Lower) => begin = begin.max(col),
                (MatrixSide::Right, Triangle::Upper) => end = end.min(col + cols),
            }
            if begin >= end {
                continue;
            }
            let depth = end - begin;
            if side == MatrixSide::Left {
                pack_b(
                    rhs,
                    GemmOp::None,
                    PackingPanel::new(col, begin, cols, depth, spec.nr),
                    &mut b,
                );
            } else {
                super::pack_entries(
                    &mut b,
                    super::PanelSide::Right,
                    spec.nr,
                    depth,
                    |inner, lane| {
                        let k = begin + inner;
                        if lane < cols && super::triangle_contains(triangle, k, col + lane) {
                            alpha * value(k, col + lane)
                        } else {
                            S::ZERO
                        }
                    },
                );
            }
            for row in (0..output.rows()).step_by(spec.mr) {
                let rows = spec.mr.min(output.rows() - row);
                let global_row = first_row + row;
                let (start, stop) = if side == MatrixSide::Left {
                    match triangle {
                        Triangle::Lower => (begin, end.min(global_row + rows)),
                        Triangle::Upper => (begin.max(global_row), end),
                    }
                } else {
                    (begin, end)
                };
                if start >= stop {
                    continue;
                }
                let depth = stop - start;
                if side == MatrixSide::Left {
                    super::pack_entries(
                        &mut a,
                        super::PanelSide::Left,
                        spec.mr,
                        depth,
                        |inner, lane| {
                            let k = start + inner;
                            if lane < rows
                                && super::triangle_contains(triangle, global_row + lane, k)
                            {
                                alpha * value(global_row + lane, k)
                            } else {
                                S::ZERO
                            }
                        },
                    );
                } else {
                    pack_a(
                        rhs,
                        GemmOp::None,
                        PackingPanel::new(global_row, start, rows, depth, spec.mr),
                        S::ONE,
                        &mut a,
                    );
                }
                let b_start = (start - begin) * b_stride;
                apply_packed_block(
                    &a[..depth * spec.mr],
                    &b[b_start..b_start + depth * b_stride],
                    &mut output,
                    crate::kernel::PackedBlock {
                        row_start: row,
                        col_start: col,
                        rows,
                        cols,
                        shared: depth,
                        global_row_start: global_row,
                    },
                    crate::kernel::PackedKernel {
                        beta: S::ONE,
                        spec,
                        microkernel,
                        triangle: None,
                    },
                );
            }
        }
    }
    check_solution(&output).map(|_| ())
}

#[cfg(test)]
mod tests {
    use scioxide_core::ExecutionContext;
    use std::num::NonZeroUsize;

    use scioxide_core::Complex;

    use super::*;
    use crate::dense::Matrix;

    type C64 = Complex<f64>;

    fn triangular_with_context<'a, S: NumericScalar>(
        action: TriangularAction,
        operand: crate::kernel::TriangularOperand<MatrixView<'_, S>>,
        rhs: MatrixViewMut<'_, S>,
        update: crate::kernel::TriangularUpdate<S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<KernelContext<'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into();
        let options = crate::LinalgExecutionOptions::new()
            .with_accuracy(context.accuracy())
            .with_execution_policy(update.policy);
        let request = crate::DenseWorkspaceRequest::triangular_solve(
            rhs.rows(),
            operand.view.rows(),
            rhs.cols(),
            &options,
        )
        .for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        super::triangular_with_context(action, operand, rhs, update, workspace, context)
    }

    fn prepared_rank<'a, S: NumericScalar>(
        input: MatrixView<'_, S>,
        output: MatrixViewMut<'_, S>,
        update: RankUpdate<S>,
        policy: ExecutionPolicy,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<KernelContext<'a>>,
        kind: super::super::RankKind,
    ) -> Result<(), SciOxideError> {
        let context = context.into();
        let left = super::super::operated_view(input, update.op);
        crate::kernel::tests::prepare_product(
            workspace,
            left.rows(),
            left.cols(),
            left.rows(),
            policy,
            context,
        )?;
        match kind {
            super::super::RankKind::Symmetric => {
                super::syrk_with_context(input, output, update, policy, workspace, context)
            }
            super::super::RankKind::Hermitian => {
                super::herk_with_context(input, output, update, policy, workspace, context)
            }
        }
    }

    fn schur_range_cases<S: NumericScalar>() {
        for scale in [
            S::Real::MIN_SUBNORMAL,
            S::Real::ONE,
            S::Real::MAX * S::Real::HALF,
        ] {
            let v = S::from_real(scale);
            let a = Matrix::from_vec(2, 2, alloc::vec![v, v, -v, v]).unwrap();
            let mut rhs = Matrix::from_vec(2, 1, alloc::vec![v + v, S::ZERO]).unwrap();
            schur_trsm_bounded(
                crate::FiniteMatrixView::from_workspace(a.view()),
                rhs.view_mut(),
                &mut LinalgWorkspace::new(),
                KernelContext::sequential(),
            )
            .unwrap();
            for &value in rhs.as_slice() {
                assert!(
                    (value - S::ONE).abs()
                        <= S::Real::try_from_f64_exact(32.).unwrap() * S::Real::EPSILON
                );
            }
        }
        // An invertible pair can have two zero scalar pivots.
        let a = Matrix::from_vec(2, 2, alloc::vec![S::ZERO, S::ONE, -S::ONE, S::ZERO]).unwrap();
        let mut rhs = Matrix::from_vec(2, 1, alloc::vec![S::ONE, -S::ONE]).unwrap();
        schur_trsm_bounded(
            crate::FiniteMatrixView::from_workspace(a.view()),
            rhs.view_mut(),
            &mut LinalgWorkspace::new(),
            KernelContext::sequential(),
        )
        .unwrap();
        assert!(
            rhs.as_slice()
                .iter()
                .all(|&x| (x - S::ONE).abs() <= S::Real::EPSILON)
        );

        let a = Matrix::from_vec(2, 2, alloc::vec![S::ONE; 4]).unwrap();
        let mut rhs = Matrix::from_vec(2, 1, alloc::vec![S::ONE; 2]).unwrap();
        schur_trsm_bounded(
            crate::FiniteMatrixView::from_workspace(a.view()),
            rhs.view_mut(),
            &mut LinalgWorkspace::new(),
            KernelContext::sequential(),
        )
        .unwrap();
        assert!((rhs.as_slice()[0] + rhs.as_slice()[1] - S::ONE).abs() <= S::Real::EPSILON);
        rhs.as_mut_slice().copy_from_slice(&[S::ONE, S::ZERO]);
        assert!(
            schur_trsm_bounded(
                crate::FiniteMatrixView::from_workspace(a.view()),
                rhs.view_mut(),
                &mut LinalgWorkspace::new(),
                KernelContext::sequential()
            )
            .is_err()
        );
    }

    #[test]
    fn schur_trsm_preserves_range_and_pair_singularity_semantics() {
        schur_range_cases::<f32>();
        schur_range_cases::<f64>();
        schur_range_cases::<Complex<f32>>();
        schur_range_cases::<Complex<f64>>();
    }

    fn schur_update<S: NumericScalar>(
        left: crate::FiniteMatrixView<'_, S>,
        right: crate::FiniteMatrixView<'_, S>,
        output: MatrixViewMut<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        crate::kernel::tests::prepare_product(
            &mut workspace,
            left.rows(),
            left.cols(),
            right.rows(),
            ExecutionPolicy::Sequential,
            execution,
        )?;
        crate::kernel::gemm_workspace_with_context(
            crate::kernel::WorkspaceGemmProblem {
                left,
                right,
                output,
                config: GemmConfig {
                    right_op: GemmOp::Adjoint,
                    alpha: -S::ONE,
                    beta: S::ONE,
                    triangle: Some(Triangle::Lower),
                    ..GemmConfig::ordinary()
                },
            },
            ExecutionPolicy::Sequential,
            &mut workspace,
            execution,
        )
    }

    #[test]
    fn rank_updates_restore_cancellation_and_subnormals() {
        // A transient sum exceeds MAX, while the final candidate is finite.
        let left = crate::Matrix::from_vec(1, 2, alloc::vec![-1e308_f64, 1e308]).unwrap();
        let right = crate::Matrix::from_vec(1, 2, alloc::vec![1.0, 1.0]).unwrap();
        let mut output = crate::Matrix::from_vec(1, 1, alloc::vec![1e308]).unwrap();
        schur_update(
            crate::FiniteMatrixView::from_workspace(left.view()),
            crate::FiniteMatrixView::from_workspace(right.view()),
            output.view_mut(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert!((output.get_element(0, 0) / 1e308 - 1.0).abs() < 1e-15);
        let left = crate::Matrix::<f64>::zeros(1, 1).unwrap();
        let right = crate::Matrix::from_vec(1, 1, alloc::vec![1e308_f64]).unwrap();
        let mut output = crate::Matrix::from_vec(1, 1, alloc::vec![f64::from_bits(1)]).unwrap();
        schur_update(
            crate::FiniteMatrixView::from_workspace(left.view()),
            crate::FiniteMatrixView::from_workspace(right.view()),
            output.view_mut(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output.get_element(0, 0), f64::from_bits(1));
        // MAX * MIN_POSITIVE is ordinary in the unit frame; normalizing both
        // factors by their maxima used to flush it and leave the entry at 8.
        let left = crate::Matrix::from_vec(1, 2, alloc::vec![f64::MAX, f64::MIN_POSITIVE]).unwrap();
        let right =
            crate::Matrix::from_vec(1, 2, alloc::vec![f64::MIN_POSITIVE, f64::MAX]).unwrap();
        let mut output = crate::Matrix::from_vec(1, 1, alloc::vec![8.0_f64]).unwrap();
        schur_update(
            crate::FiniteMatrixView::from_workspace(left.view()),
            crate::FiniteMatrixView::from_workspace(right.view()),
            output.view_mut(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        // The exact entry is 2^-50; ordinary subtraction stays within
        // gamma_2 * (|8| + sum |l r|) <= 2^-48 of it.
        let entry: f64 = output.get_element(0, 0);
        assert!((entry - 2.0_f64.powi(-50)).abs() <= 2.0_f64.powi(-48));
    }

    #[test]
    fn shared_triangular_product_rejects_nonfinite_input_without_mutation() {
        let factor = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
        let mut rhs = Matrix::from_vec(1, 1, vec![7.0]).unwrap();
        let error = crate::kernel::matrix::triangular_product(
            factor.view(),
            &mut rhs.view_mut(),
            crate::kernel::matrix::TriangularProduct {
                side: MatrixSide::Left,
                triangle: scioxide_core::Triangle::Lower,
                diagonal: DiagonalKind::Explicit,
                operation: crate::StructuredOperation::Normal,
                alpha: 1.0,
            },
            &crate::LinalgExecutionOptions::new(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential().into(),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        assert_eq!(rhs.as_slice(), &[7.0]);
    }

    fn triangular(order: usize, triangle: Triangle, diagonal: DiagonalKind) -> Matrix<C64> {
        let values = (0..order * order)
            .map(|index| {
                let row = index / order;
                let col = index % order;
                if !super::super::triangle_contains(triangle, row, col) {
                    return C64::new(31.0, -29.0);
                }
                if row == col {
                    return if diagonal == DiagonalKind::Unit {
                        C64::new(f64::NAN, f64::NAN)
                    } else {
                        C64::new(2.0 + row as f64 / order as f64, 0.125)
                    };
                }
                let real = ((row * 7 + col * 11 + 3) % 17) as f64 - 8.0;
                let imaginary = ((row * 13 + col * 5 + 1) % 19) as f64 - 9.0;
                C64::new(real / 41.0, imaginary / 53.0)
            })
            .collect();
        Matrix::from_vec(order, order, values).unwrap()
    }

    fn expected_solution(rows: usize, cols: usize) -> Matrix<C64> {
        let values = (0..rows * cols)
            .map(|index| {
                let real = ((index * 11 + 7) % 23) as f64 - 11.0;
                let imaginary = ((index * 5 + 2) % 13) as f64 - 6.0;
                C64::new(real / 17.0, imaginary / 19.0)
            })
            .collect();
        Matrix::from_vec(rows, cols, values).unwrap()
    }

    fn effective_entry(
        matrix: MatrixView<'_, C64>,
        stored_triangle: Triangle,
        op: GemmOp,
        diagonal: DiagonalKind,
        row: usize,
        col: usize,
    ) -> C64 {
        if row == col && diagonal == DiagonalKind::Unit {
            return C64::new(1.0, 0.0);
        }
        let effective_triangle = match op {
            GemmOp::None => stored_triangle,
            GemmOp::Transpose | GemmOp::Adjoint => flip_triangle(stored_triangle),
        };
        if !super::super::triangle_contains(effective_triangle, row, col) {
            return C64::new(0.0, 0.0);
        }
        operated_get(matrix, op, row, col)
    }

    fn right_hand_side(
        triangular: MatrixView<'_, C64>,
        solution: MatrixView<'_, C64>,
        side: MatrixSide,
        stored_triangle: Triangle,
        op: GemmOp,
        diagonal: DiagonalKind,
        alpha: C64,
    ) -> Matrix<C64> {
        let mut values = vec![C64::new(0.0, 0.0); solution.rows * solution.cols];
        match side {
            MatrixSide::Left => {
                for row in 0..solution.rows {
                    for col in 0..solution.cols {
                        values[row * solution.cols + col] = (0..solution.rows)
                            .map(|inner| {
                                effective_entry(
                                    triangular,
                                    stored_triangle,
                                    op,
                                    diagonal,
                                    row,
                                    inner,
                                ) * solution.get_element(inner, col)
                            })
                            .fold(C64::new(0.0, 0.0), |sum, value| sum + value)
                            / alpha;
                    }
                }
            }
            MatrixSide::Right => {
                for row in 0..solution.rows {
                    for col in 0..solution.cols {
                        values[row * solution.cols + col] = (0..solution.cols)
                            .map(|inner| {
                                solution.get_element(row, inner)
                                    * effective_entry(
                                        triangular,
                                        stored_triangle,
                                        op,
                                        diagonal,
                                        inner,
                                        col,
                                    )
                            })
                            .fold(C64::new(0.0, 0.0), |sum, value| sum + value)
                            / alpha;
                    }
                }
            }
        }
        Matrix::from_vec(solution.rows, solution.cols, values).unwrap()
    }

    fn assert_complex_close(actual: &Matrix<C64>, expected: &Matrix<C64>, tolerance: f64) {
        actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .for_each(|(&actual, &expected)| {
                assert!(
                    (actual - expected).norm() <= tolerance * (1.0 + expected.norm()),
                    "actual {actual:?}, expected {expected:?}"
                );
            });
    }

    #[test]
    fn blocked_trsm_matches_constructed_solutions_for_all_variants() {
        let order = 70;
        let alpha = C64::new(0.75, -0.25);
        for side in [MatrixSide::Left, MatrixSide::Right] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                for op in [GemmOp::None, GemmOp::Transpose, GemmOp::Adjoint] {
                    for diagonal in [DiagonalKind::Unit, DiagonalKind::Explicit] {
                        let triangular = triangular(order, triangle, diagonal);
                        let expected = match side {
                            MatrixSide::Left => expected_solution(order, 5),
                            MatrixSide::Right => expected_solution(5, order),
                        };
                        let mut actual = right_hand_side(
                            triangular.view(),
                            expected.view(),
                            side,
                            triangle,
                            op,
                            diagonal,
                            alpha,
                        );
                        let mut workspace = LinalgWorkspace::new();
                        triangular_with_context(
                            TriangularAction::Solve,
                            crate::kernel::TriangularOperand {
                                view: triangular.view(),
                                triangle,
                                op,
                                diagonal,
                            },
                            actual.view_mut(),
                            crate::kernel::TriangularUpdate {
                                side,
                                alpha,
                                policy: scioxide_core::ExecutionPolicy::Auto,
                            },
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                        assert_complex_close(&actual, &expected, 2.0e-11);
                    }
                }
            }
        }
    }

    #[test]
    fn small_trsm_matches_constructed_solutions_without_packed_storage() {
        let order = 8;
        let alpha = C64::new(0.75, -0.25);
        for side in [MatrixSide::Left, MatrixSide::Right] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                for op in [GemmOp::None, GemmOp::Transpose, GemmOp::Adjoint] {
                    for diagonal in [DiagonalKind::Unit, DiagonalKind::Explicit] {
                        let triangular = triangular(order, triangle, diagonal);
                        let expected = match side {
                            MatrixSide::Left => expected_solution(order, 5),
                            MatrixSide::Right => expected_solution(5, order),
                        };
                        let mut actual = right_hand_side(
                            triangular.view(),
                            expected.view(),
                            side,
                            triangle,
                            op,
                            diagonal,
                            alpha,
                        );
                        let mut workspace = LinalgWorkspace::new();
                        triangular_with_context(
                            TriangularAction::Solve,
                            crate::kernel::TriangularOperand {
                                view: triangular.view(),
                                triangle,
                                op,
                                diagonal,
                            },
                            actual.view_mut(),
                            crate::kernel::TriangularUpdate {
                                side,
                                alpha,
                                policy: scioxide_core::ExecutionPolicy::Auto,
                            },
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                        assert_complex_close(&actual, &expected, 2.0e-11);
                        assert_eq!(workspace.scalar_mut(crate::kernel::PACKED_A).len(), 0);
                        assert_eq!(workspace.scalar_mut(crate::kernel::PACKED_B).len(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_preflight_failure_preserves_output() {
        let mut triangular = triangular(70, Triangle::Lower, DiagonalKind::Explicit);
        triangular
            .view_mut()
            .set(69, 69, C64::new(0.0, 0.0))
            .unwrap();
        let mut rhs = expected_solution(70, 3);
        let original = rhs.clone();
        let mut workspace = LinalgWorkspace::new();

        assert!(matches!(
            (triangular_with_context(
                TriangularAction::Solve,
                crate::kernel::TriangularOperand {
                    view: triangular.view(),
                    triangle: Triangle::Lower,
                    op: GemmOp::None,
                    diagonal: DiagonalKind::Explicit
                },
                rhs.view_mut(),
                crate::kernel::TriangularUpdate {
                    side: MatrixSide::Left,
                    alpha: C64::new(1.0, 0.0),
                    policy: scioxide_core::ExecutionPolicy::Auto
                },
                &mut workspace,
                ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
        assert_eq!(rhs.as_slice(), original.as_slice());
    }

    #[test]
    fn syrk_updates_only_selected_triangle() {
        let rows = 67;
        let shared = 37;
        let input = Matrix::from_vec(
            rows,
            shared,
            (0..rows * shared)
                .map(|index| ((index * 17 + 5) % 31) as f64 / 23.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let initial = Matrix::from_vec(
            rows,
            rows,
            (0..rows * rows)
                .map(|index| ((index * 7 + 3) % 29) as f64 / 19.0)
                .collect(),
        )
        .unwrap();
        let mut actual = initial.clone();
        let mut workspace = LinalgWorkspace::new();
        prepared_rank(
            input.view(),
            actual.view_mut(),
            crate::kernel::level3::RankUpdate {
                triangle: Triangle::Upper,
                op: GemmOp::None,
                alpha: 0.75,
                beta: -0.25,
            },
            ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
            &mut workspace,
            ExecutionContext::sequential(),
            super::super::RankKind::Symmetric,
        )
        .unwrap();

        for row in 0..rows {
            for col in 0..rows {
                if row <= col {
                    let product = (0..shared)
                        .map(|inner| input.get_element(row, inner) * input.get_element(col, inner))
                        .sum::<f64>();
                    let expected = 0.75 * product - 0.25 * initial.get_element(row, col);
                    assert!((actual.get_element(row, col) - expected).abs() <= 2.0e-12);
                } else {
                    assert_eq!(actual.get_element(row, col), initial.get_element(row, col));
                }
            }
        }
    }

    #[test]
    fn herk_applies_adjoint_and_preserves_unselected_triangle() {
        let rows = 65;
        let shared = 35;
        let input = expected_solution(rows, shared);
        let initial = expected_solution(rows, rows);
        let mut actual = initial.clone();
        let mut workspace = LinalgWorkspace::new();
        let alpha = C64::new(0.5, 0.125);
        let beta = C64::new(-0.25, 0.0);
        prepared_rank(
            input.view(),
            actual.view_mut(),
            crate::kernel::level3::RankUpdate {
                triangle: Triangle::Lower,
                op: GemmOp::None,
                alpha,
                beta,
            },
            ExecutionPolicy::Sequential,
            &mut workspace,
            ExecutionContext::sequential(),
            super::super::RankKind::Hermitian,
        )
        .unwrap();

        for row in 0..rows {
            for col in 0..rows {
                if row >= col {
                    let product = (0..shared)
                        .map(|inner| {
                            input.get_element(row, inner) * input.get_element(col, inner).conj()
                        })
                        .fold(C64::new(0.0, 0.0), |sum, value| sum + value);
                    let expected = alpha * product + beta * initial.get_element(row, col);
                    assert!(
                        (actual.get_element(row, col) - expected).norm()
                            <= 3.0e-12 * (1.0 + expected.norm())
                    );
                } else {
                    assert_eq!(actual.get_element(row, col), initial.get_element(row, col));
                }
            }
        }
    }
    #[test]
    fn zero_triangular_coefficients_preserve_observable_zero_signs() {
        fn check<S: RealScalar + NumericScalar<Real = S>>() {
            for negative_coefficient in [false, true] {
                let mut triangular = crate::Matrix::<S>::identity(2).unwrap();
                triangular.as_mut_slice()[1] = if negative_coefficient {
                    -S::ZERO
                } else {
                    S::ZERO
                };
                for negative_target in [false, true] {
                    let zero = if negative_target { -S::ZERO } else { S::ZERO };
                    let mut rhs =
                        crate::Matrix::from_vec(2, 2, alloc::vec![zero, zero, S::ONE, -S::ONE])
                            .unwrap();
                    solve_left_row(
                        triangular.view(),
                        &mut rhs.view_mut(),
                        GemmOp::None,
                        DiagonalKind::Unit,
                        1..2,
                        0,
                    )
                    .unwrap();
                    assert_eq!(rhs.as_slice()[0], S::ZERO);
                    assert_eq!(rhs.as_slice()[1], S::ZERO);
                    assert_eq!(
                        rhs.as_slice()[0].is_sign_negative(),
                        negative_target && !negative_coefficient
                    );
                    assert_eq!(
                        rhs.as_slice()[1].is_sign_negative(),
                        negative_target && negative_coefficient
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
