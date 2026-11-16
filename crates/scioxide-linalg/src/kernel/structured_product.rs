//! Structural operands prepared in reusable storage for the common product driver.

pub(crate) struct ProductShape {
    pub(crate) rows: usize,
    pub(crate) shared: usize,
    pub(crate) cols: usize,
}
pub(crate) struct RankTwoUpdate<S> {
    pub(crate) kind: RankKind,
    pub(crate) alpha: S,
    pub(crate) beta: S,
    pub(crate) triangle: Triangle,
}

use super::*;
use scioxide_core::{AlgebraicScalar, RealScalar};

pub(crate) fn with_candidate<S: NumericScalar>(
    workspace: &mut LinalgWorkspace<'_, S>,
    len: usize,
    work: impl FnOnce(&mut [S], &mut LinalgWorkspace<'_, S>) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let mut req = crate::LinalgRequirements::default();
    *req.scalar_mut(STRUCTURED_OUTPUT) = len;
    workspace.require_capacity(&req)?;
    workspace.with_regions([STRUCTURED_OUTPUT], |[candidate], workspace| {
        work(&mut candidate[..len], workspace)
    })
}

pub(crate) fn product<S: NumericScalar>(
    shape: crate::kernel::structured_product::ProductShape,
    left: impl Fn(usize, usize) -> S,
    right: impl Fn(usize, usize) -> S,
    output: MatrixViewMut<'_, S>,
    config: GemmConfig<S>,
    execution: crate::kernel::KernelExecution<KernelContext<'_>>,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let crate::kernel::structured_product::ProductShape { rows, shared, cols } = shape;
    let crate::kernel::KernelExecution { policy, context } = execution;

    let mut requirements = crate::LinalgRequirements::default();
    *requirements.scalar_mut(STRUCTURED_A) =
        checked_product(&[rows, shared], scioxide_core::quantity!("structured_left"))?;
    *requirements.scalar_mut(STRUCTURED_B) = checked_product(
        &[shared, cols],
        scioxide_core::quantity!("structured_right"),
    )?;
    workspace.require_capacity(&requirements)?;
    workspace.with_regions([STRUCTURED_A, STRUCTURED_B], |[a, b], workspace| {
        let left = prepare(rows, shared, left, a)?;
        let right = prepare(shared, cols, right, b)?;
        gemm_bounded(left, right, output, config, policy, workspace, context)
    })
}

fn prepare<S: NumericScalar>(
    rows: usize,
    cols: usize,
    entry: impl Fn(usize, usize) -> S,
    storage: &mut [S],
) -> Result<FiniteMatrixView<'_, S>, SciOxideError> {
    let len = checked_product(
        &[rows, cols],
        scioxide_core::quantity!("structured_product_operand"),
    )?;
    if storage.len() < len {
        return Err(SciOxideError::dimension_mismatch([len], [storage.len()]));
    }
    let mut maximum = S::Real::ZERO;
    for row in 0..rows {
        for col in 0..cols {
            let value = entry(row, col);
            crate::checked::validate_input(value, "structured product operand")?;
            maximum = maximum.max(standard::magnitude(value));
            storage[row * cols + col] = value;
        }
    }
    Ok(FiniteMatrixView::from_validated(
        MatrixView::from_slice(rows, cols, &storage[..len])?,
        maximum,
    ))
}

/// Stages the prior `C` of a rank-two update: the selected triangle when `beta` is nonzero,
/// zero elsewhere.
fn stage_prior<S: NumericScalar>(
    candidate: &mut [S],
    prior: MatrixView<'_, S>,
    triangle: Triangle,
    beta: S,
) {
    let n = prior.rows();
    for row in 0..n {
        for col in 0..n {
            candidate[row * n + col] = if triangle_contains(triangle, row, col) && beta != S::ZERO {
                prior.get_element(row, col)
            } else {
                S::ZERO
            };
        }
    }
}

/// Computes `alpha op(A) op(B)ᴴ + conj(alpha) op(B) op(A)ᴴ + beta C` (Hermitian) or the
/// transposed form (symmetric) on one triangle of `candidate`, staged here from `prior`.
///
/// `Exact` shares one exact accumulator through both contributions. `Standard` runs two
/// bounded products; when their intermediate sum leaves the range although the result may not,
/// both run again with `alpha` and `beta` scaled by a common power of two `2^-s` chosen so no
/// partial sum can overflow, and the result is restored by `2^s`.
pub(crate) fn rank_two<S: NumericScalar>(
    operands: Option<(FiniteMatrixView<'_, S>, FiniteMatrixView<'_, S>)>,
    update: crate::kernel::structured_product::RankTwoUpdate<S>,
    prior: MatrixView<'_, S>,
    candidate: &mut [S],
    execution: crate::kernel::KernelExecution<KernelContext<'_>>,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<(), SciOxideError> {
    let crate::kernel::structured_product::RankTwoUpdate {
        kind,
        alpha,
        beta,
        triangle,
    } = update;

    let crate::kernel::KernelExecution { policy, context } = execution;
    let n = prior.rows();
    let candidate = &mut candidate[..n * n];
    stage_prior(candidate, prior, triangle, beta);

    let Some((a, b)) = operands else {
        for row in 0..n {
            for col in 0..n {
                if triangle_contains(triangle, row, col) {
                    let index = row * n + col;
                    // `Standard` retries a complex `beta * c` whose partial products leave the
                    // range; `Exact` rounds it once, as GEMM and SYMM do.
                    candidate[index] = if beta == S::ZERO {
                        S::ZERO
                    } else {
                        scioxide_core::backend::standard::product(
                            beta,
                            candidate[index],
                            context.accuracy(),
                        )?
                    };
                }
            }
        }
        return Ok(());
    };
    let ah = if kind == RankKind::Hermitian {
        a.adjoint()
    } else {
        a.transpose()
    };
    let bh = if kind == RankKind::Hermitian {
        b.adjoint()
    } else {
        b.transpose()
    };
    let second_alpha = if kind == RankKind::Hermitian {
        alpha.conj()
    } else {
        alpha
    };
    let config = GemmConfig {
        alpha,
        beta,
        triangle: Some(triangle),
        ..GemmConfig::ordinary()
    };
    if context.accuracy() == Accuracy::Exact {
        let layout =
            workspace.exact_layout(a.rows(), a.cols(), a.rows(), policy, context.execution())?;
        let crate::workspace::ProductParts {
            tiles,
            reals,
            accumulators,
            ..
        } = workspace.product_parts()?;
        let mut exact = exact::Workspace {
            tiles,
            reals,
            accumulators,
            layout,
        };
        return exact.multiply(
            crate::kernel::exact::ExactProduct {
                left: a.as_view(),
                right: bh.as_view(),
                config,
                addend: Some((b.as_view(), ah.as_view(), second_alpha)),
            },
            candidate,
            policy,
            context.execution(),
            context.dispatch(),
        );
    }
    let mut passes = |candidate: &mut [S], shift: i32| {
        gemm_bounded(
            a,
            bh,
            MatrixViewMut::from_slice(n, n, candidate)?,
            GemmConfig {
                alpha: standard::scale(alpha, -shift),
                beta: standard::scale(beta, -shift),
                ..config
            },
            policy,
            workspace,
            context,
        )?;
        gemm_bounded(
            b,
            ah,
            MatrixViewMut::from_slice(n, n, candidate)?,
            GemmConfig {
                alpha: standard::scale(second_alpha, -shift),
                beta: S::ONE,
                ..config
            },
            policy,
            workspace,
            context,
        )
    };
    match passes(candidate, 0) {
        Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability => {
            // Each partial sum is bounded by `|alpha A Bᴴ| + |alpha B Aᴴ| + |beta C|`, so
            // `2^(bound + 2)` exceeds every intermediate value.
            let growth = usize::BITS as i32 - a.cols().max(1).leading_zeros() as i32 + 4;
            let product = standard::exponent(a.maximum)?
                + standard::exponent(b.maximum)?
                + standard::exponent(standard::magnitude(alpha))?
                + growth;
            let prior_bound = if beta == S::ZERO {
                0
            } else {
                standard::exponent(standard::component_maximum(prior, Some(triangle)))?
                    + standard::exponent(standard::magnitude(beta))?
                    + 3
            };
            let shift = product.max(prior_bound) + 3 - standard::exponent(S::Real::MAX)?;
            if shift <= 0 {
                return Err(error);
            }
            stage_prior(candidate, prior, triangle, beta);
            passes(candidate, shift)?;
            for row in 0..n {
                for col in 0..n {
                    if triangle_contains(triangle, row, col) {
                        candidate[row * n + col] =
                            standard::restore(candidate[row * n + col], shift)?;
                    }
                }
            }
            Ok(())
        }
        other => other,
    }
}
