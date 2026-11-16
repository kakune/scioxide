//! Pure eligibility planning and typed provider invocation.

use crate::MatrixSide;
use core::num::NonZeroUsize;
use scioxide_core::{AlgebraicScalar, RealScalar};

use scioxide_core::backend::{Scalar, ScalarKind, ScalarSlice, ScalarSliceMut, ScalarValue};
use scioxide_core::{Capability, NonFiniteOrigin, NumericScalar, SciOxideError};

use scioxide_core::DiagonalKind;

use scioxide_core::Triangle;

use crate::blas::{
    BlasBandMatrix, BlasCall, BlasCapabilities, BlasDispatch, BlasFallbackPolicy, BlasIntegerWidth,
    BlasLayout, BlasMatrix, BlasMatrixMut, BlasOperation, BlasPackedMatrix, BlasPackedMatrixMut,
    BlasStructuredMatrix, BlasStructuredMatrixMut, BlasThreading, BlasTransform, BlasVector,
    BlasVectorMut,
};
use crate::dense::{MatrixView, VectorView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlasRoute {
    Provider,
    Reference,
}

#[derive(Debug, Clone, Copy)]
struct MatrixEncoding {
    rows: usize,
    cols: usize,
    layout: BlasLayout,
    leading_dimension: usize,
    transform: BlasTransform,
}

pub(crate) struct BlasGemm<'a, S: NumericScalar> {
    pub(crate) left: MatrixView<'a, S>,
    pub(crate) right: MatrixView<'a, S>,
    pub(crate) alpha: S,
    pub(crate) beta: S,
    pub(crate) triangle: Option<Triangle>,
    pub(crate) packed_left: &'a mut [S],
    pub(crate) packed_right: &'a mut [S],
    pub(crate) candidate: &'a mut [S],
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) capabilities: &'a BlasCapabilities,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn gemm<S: NumericScalar>(request: BlasGemm<'_, S>) -> Result<BlasRoute, SciOxideError> {
    let BlasGemm {
        left,
        right,
        alpha,
        beta,
        triangle,
        packed_left,
        packed_right,
        candidate,
        dispatch,
        capabilities,
        accuracy,
        workers,
    } = request;
    if left.rows == 0 || right.cols == 0 || left.cols == 0 {
        return Ok(BlasRoute::Reference);
    }
    let layout = preferred_layout(capabilities);

    let Some((a, left_transform)) =
        prepare_matrix(left, packed_left, capabilities, layout, |_| true)
    else {
        return unsupported(dispatch);
    };
    let Some((b, right_transform)) =
        prepare_matrix(right, packed_right, capabilities, layout, |_| true)
    else {
        return unsupported(dispatch);
    };
    let output = BlasMatrixMut::new(
        candidate,
        left.rows,
        right.cols,
        leading_dimension(layout, left.rows, right.cols),
        layout,
    );
    // A masked product publishes only its selected triangle, so only that
    // triangle is validated; the provider may leave anything elsewhere.
    provide(
        &mut BlasCall::Gemm {
            alpha,
            a,
            b,
            left_transform,
            right_transform,
            beta,
            c: output,
        },
        capabilities,
        triangle,
        dispatch,
        accuracy,
        workers,
    )
}

pub(crate) struct BlasRankK<'a, S: NumericScalar> {
    pub(crate) input: MatrixView<'a, S>,
    pub(crate) order: usize,
    pub(crate) alpha: S,
    pub(crate) beta: S,
    pub(crate) triangle: Triangle,
    pub(crate) hermitian: bool,
    pub(crate) packed_input: &'a mut [S],
    pub(crate) candidate: &'a mut [S],
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) capabilities: &'a BlasCapabilities,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn rank_k<S: NumericScalar>(
    request: BlasRankK<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let BlasRankK {
        input,
        order,
        alpha,
        beta,
        triangle,
        hermitian,
        packed_input,
        candidate,
        dispatch,
        capabilities,
        accuracy,
        workers,
    } = request;
    if order == 0 || input.cols == 0 {
        return Ok(BlasRoute::Reference);
    }
    let layout = preferred_layout(capabilities);

    let Some((a, transform)) =
        prepare_matrix(input, packed_input, capabilities, layout, |transform| {
            transform == BlasTransform::None
                || transform
                    == if hermitian {
                        BlasTransform::Adjoint
                    } else {
                        BlasTransform::Transpose
                    }
        })
    else {
        return unsupported(dispatch);
    };
    let output = BlasMatrixMut::new(candidate, order, order, order.max(1), layout);
    let mut call = if hermitian {
        BlasCall::Herk {
            transform,
            triangle,
            alpha: alpha.real_component(),
            a,
            beta: beta.real_component(),
            c: output,
        }
    } else {
        BlasCall::Syrk {
            transform,
            triangle,
            alpha,
            a,
            beta,
            c: output,
        }
    };
    provide(&mut call, capabilities, None, dispatch, accuracy, workers)
}

pub(crate) struct BlasTriangular<'a, S: NumericScalar> {
    pub(crate) triangular: MatrixView<'a, S>,
    pub(crate) op: crate::kernel::GemmOp,
    pub(crate) packed: &'a mut [S],
    pub(crate) candidate: &'a mut [S],
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) side: MatrixSide,
    pub(crate) triangle: Triangle,
    pub(crate) diagonal: DiagonalKind,
    pub(crate) alpha: S,
    pub(crate) solve: bool,
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) capabilities: &'a BlasCapabilities,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn triangular<S: NumericScalar>(
    request: BlasTriangular<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let BlasTriangular {
        triangular,
        op,
        packed,
        candidate,
        rows,
        cols,
        side,
        triangle,
        diagonal,
        alpha,
        solve,
        dispatch,
        capabilities,
        accuracy,
        workers,
    } = request;
    if rows == 0 || cols == 0 {
        return Ok(BlasRoute::Reference);
    }
    let layout = preferred_layout(capabilities);
    let operated = crate::kernel::operated_view(triangular, op);
    let Some((a, transform)) = prepare_matrix(operated, packed, capabilities, layout, |_| true)
    else {
        return unsupported(dispatch);
    };

    // Uplo describes the encoded storage, before the call-level transform.
    let logical_transpose = op != crate::kernel::GemmOp::None;
    let encoded_transpose = matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint);
    let triangle = if logical_transpose != encoded_transpose {
        crate::kernel::flip_triangle(triangle)
    } else {
        triangle
    };
    let b = BlasMatrixMut::new(
        candidate,
        rows,
        cols,
        leading_dimension(layout, rows, cols),
        layout,
    );
    let mut call = if solve {
        BlasCall::Trsm {
            transform,
            side,
            triangle,
            diagonal,
            alpha,
            a,
            b,
        }
    } else {
        BlasCall::Trmm {
            transform,
            side,
            triangle,
            diagonal,
            alpha,
            a,
            b,
        }
    };
    provide(&mut call, capabilities, None, dispatch, accuracy, workers)
}

pub(crate) fn gemv<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    input: VectorView<'_, S>,
    candidate: &mut [S],
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    let capabilities = dispatch.capabilities()?;
    // GEMV has only one matrix, so either supported layout can represent it.
    let Some(encoding) = [BlasLayout::RowMajor, BlasLayout::ColumnMajor]
        .into_iter()
        .find_map(|layout| direct_matrix_encoding(matrix, &capabilities, layout, |_| true))
    else {
        return unsupported(dispatch);
    };

    let matrix = BlasMatrix::new(
        matrix.data,
        encoding.rows,
        encoding.cols,
        encoding.leading_dimension,
        encoding.layout,
    );
    let input = vector(input);
    // GEMV is issued with `beta = 0`, but the output is still initialized so
    // stale scratch can never reach the result through a provider that reads y.
    candidate.fill(S::ZERO);
    let output = BlasVectorMut::new(candidate, candidate.len(), 1);
    provide(
        &mut BlasCall::Gemv {
            transform: encoding.transform,
            alpha: S::ONE,
            a: matrix,
            x: input,
            beta: S::ZERO,
            y: output,
        },
        &capabilities,
        None,
        dispatch,
        accuracy,
        workers,
    )
}

pub(crate) fn dot<S: NumericScalar>(
    left: VectorView<'_, S>,
    right: VectorView<'_, S>,
    conjugate_left: bool,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<Option<S>, SciOxideError> {
    let mut result = S::ZERO;
    let x = vector(left);
    let y = vector(right);
    let mut call = if conjugate_left {
        BlasCall::Dotc {
            x,
            y,
            result: &mut result,
        }
    } else {
        BlasCall::Dotu {
            x,
            y,
            result: &mut result,
        }
    };
    let route = invoke(&mut call, dispatch, accuracy, workers)?;
    Ok((route == BlasRoute::Provider).then_some(result))
}

pub(crate) fn norm2<S: NumericScalar>(
    values: VectorView<'_, S>,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<Option<S::Real>, SciOxideError> {
    let mut result = S::Real::ZERO;
    let route = invoke(
        &mut BlasCall::Norm2 {
            x: vector(values),
            result: &mut result,
        },
        dispatch,
        accuracy,
        workers,
    )?;
    Ok((route == BlasRoute::Provider).then_some(result))
}

pub(crate) fn axpy<S: NumericScalar>(
    alpha: S,
    input: VectorView<'_, S>,
    candidate: &mut [S],
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    invoke(
        &mut BlasCall::Axpy {
            alpha,
            x: vector(input),
            y: vector_mut(candidate),
        },
        dispatch,
        accuracy,
        workers,
    )
}

pub(crate) fn copy<S: NumericScalar>(
    input: VectorView<'_, S>,
    candidate: &mut [S],
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    invoke(
        &mut BlasCall::Copy {
            x: vector(input),
            y: vector_mut(candidate),
        },
        dispatch,
        accuracy,
        workers,
    )
}

pub(crate) fn scale<S: NumericScalar>(
    alpha: S,
    candidate: &mut [S],
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    invoke(
        &mut BlasCall::Scale {
            alpha,
            x: vector_mut(candidate),
        },
        dispatch,
        accuracy,
        workers,
    )
}

fn base_eligible<S: NumericScalar>(
    capabilities: &BlasCapabilities,
    operation: BlasOperation,
) -> bool {
    capabilities.operation_ulp(S::kind(), operation).is_some()
}

pub(crate) fn asum<S: NumericScalar>(
    values: VectorView<'_, S>,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<Option<S::Real>, SciOxideError> {
    let mut result = S::Real::ZERO;
    let route = invoke(
        &mut BlasCall::Asum {
            x: vector(values),
            result: &mut result,
        },
        dispatch,
        accuracy,
        workers,
    )?;
    Ok((route == BlasRoute::Provider).then_some(result))
}

pub(crate) fn iamax<S: NumericScalar>(
    values: VectorView<'_, S>,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<Option<usize>, SciOxideError> {
    let mut result = 0;
    let route = invoke(
        &mut BlasCall::Iamax {
            x: vector(values),
            result: &mut result,
        },
        dispatch,
        accuracy,
        workers,
    )?;
    Ok((route == BlasRoute::Provider).then_some(result))
}

/// Execute a validated Level 2 product against an explicitly encoded matrix.
pub(crate) struct StructuredMatVec<'a, S: NumericScalar> {
    pub(crate) operation: BlasOperation,
    pub(crate) transform: BlasTransform,
    pub(crate) matrix: BlasStructuredMatrix<'a, S>,
}

pub(crate) struct MatVecRequest<'a, S: NumericScalar> {
    pub(crate) request: StructuredMatVec<'a, S>,
    pub(crate) alpha: S,
    pub(crate) x: VectorView<'a, S>,
    pub(crate) beta: S,
    pub(crate) candidate: &'a mut [S],
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn structured_matvec<S: NumericScalar>(
    request: MatVecRequest<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let MatVecRequest {
        request,
        alpha,
        x,
        beta,
        candidate,
        dispatch,
        accuracy,
        workers,
    } = request;
    let x = vector(x);
    let y = vector_mut(candidate);
    let mut call = match (request.operation, request.matrix) {
        (
            BlasOperation::Gbmv,
            BlasStructuredMatrix::Band {
                matrix: a,
                triangle: None,
            },
        ) => BlasCall::Gbmv {
            transform: request.transform,
            alpha,
            a,
            x,
            beta,
            y,
        },
        (
            BlasOperation::Symv,
            BlasStructuredMatrix::Dense {
                matrix: a,
                triangle,
            },
        ) => BlasCall::Symv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        },
        (
            BlasOperation::Hemv,
            BlasStructuredMatrix::Dense {
                matrix: a,
                triangle,
            },
        ) => BlasCall::Hemv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        },
        (
            BlasOperation::Sbmv,
            BlasStructuredMatrix::Band {
                matrix: a,
                triangle: Some(triangle),
            },
        ) => BlasCall::Sbmv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        },
        (
            BlasOperation::Hbmv,
            BlasStructuredMatrix::Band {
                matrix: a,
                triangle: Some(triangle),
            },
        ) => BlasCall::Hbmv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        },
        (BlasOperation::Spmv, BlasStructuredMatrix::Packed(a)) => BlasCall::Spmv {
            alpha,
            a,
            x,
            beta,
            y,
        },
        (BlasOperation::Hpmv, BlasStructuredMatrix::Packed(a)) => BlasCall::Hpmv {
            alpha,
            a,
            x,
            beta,
            y,
        },
        _ => return unsupported(dispatch),
    };
    invoke(&mut call, dispatch, accuracy, workers)
}

pub(crate) struct RankUpdateRequest<'a, S: NumericScalar> {
    pub(crate) blas_operation: BlasOperation,
    pub(crate) triangle: Triangle,
    pub(crate) alpha: S,
    pub(crate) x: VectorView<'a, S>,
    pub(crate) y: Option<VectorView<'a, S>>,
    pub(crate) matrix: BlasStructuredMatrixMut<'a, S>,
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) capabilities: &'a BlasCapabilities,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn rank_update<S: NumericScalar>(
    request: RankUpdateRequest<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let RankUpdateRequest {
        blas_operation,
        triangle,
        alpha,
        x,
        y,
        matrix,
        dispatch,
        capabilities,
        accuracy,
        workers,
    } = request;
    let x = vector(x);
    let y = y.map(vector);
    let mut call = match (blas_operation, matrix, y) {
        (BlasOperation::Geru, BlasStructuredMatrixMut::Dense(a), Some(y)) => {
            BlasCall::Geru { alpha, x, y, a }
        }
        (BlasOperation::Gerc, BlasStructuredMatrixMut::Dense(a), Some(y)) => {
            BlasCall::Gerc { alpha, x, y, a }
        }
        (BlasOperation::Syr, BlasStructuredMatrixMut::Dense(a), None) => BlasCall::Syr {
            triangle,
            alpha,
            x,
            a,
        },
        (BlasOperation::Syr2, BlasStructuredMatrixMut::Dense(a), Some(y)) => BlasCall::Syr2 {
            triangle,
            alpha,
            x,
            y,
            a,
        },
        (BlasOperation::Her, BlasStructuredMatrixMut::Dense(a), None) => BlasCall::Her {
            triangle,
            alpha: alpha.real_component(),
            x,
            a,
        },
        (BlasOperation::Her2, BlasStructuredMatrixMut::Dense(a), Some(y)) => BlasCall::Her2 {
            triangle,
            alpha,
            x,
            y,
            a,
        },
        (BlasOperation::Spr, BlasStructuredMatrixMut::Packed(a), None) => {
            BlasCall::Spr { alpha, x, a }
        }
        (BlasOperation::Spr2, BlasStructuredMatrixMut::Packed(a), Some(y)) => {
            BlasCall::Spr2 { alpha, x, y, a }
        }
        (BlasOperation::Hpr, BlasStructuredMatrixMut::Packed(a), None) => BlasCall::Hpr {
            alpha: alpha.real_component(),
            x,
            a,
        },
        (BlasOperation::Hpr2, BlasStructuredMatrixMut::Packed(a), Some(y)) => {
            BlasCall::Hpr2 { alpha, x, y, a }
        }
        _ => return unsupported(dispatch),
    };
    provide(&mut call, capabilities, None, dispatch, accuracy, workers)
}

#[derive(Clone, Copy)]
pub(crate) enum Level3Kind {
    SelfAdjoint { side: MatrixSide },
    RankTwoK { transform: BlasTransform },
}

pub(crate) struct Level3Request<'a, S: NumericScalar> {
    pub(crate) kind: Level3Kind,
    pub(crate) blas_operation: BlasOperation,
    pub(crate) triangle: Triangle,
    pub(crate) alpha: S,
    pub(crate) a: MatrixView<'a, S>,
    pub(crate) b: MatrixView<'a, S>,
    pub(crate) beta: S,
    pub(crate) candidate: &'a mut [S],
    pub(crate) packed_a: &'a mut [S],
    pub(crate) packed_b: &'a mut [S],
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) capabilities: &'a BlasCapabilities,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn level3<S: NumericScalar>(
    request: Level3Request<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let Level3Request {
        kind,
        blas_operation,
        triangle,
        alpha,
        a,
        b,
        beta,
        candidate,
        packed_a,
        packed_b,
        rows,
        cols,
        dispatch,
        capabilities,
        accuracy,
        workers,
    } = request;
    let layout = preferred_layout(capabilities);
    let (a, b) = match kind {
        Level3Kind::SelfAdjoint { .. } => (a, b),
        Level3Kind::RankTwoK { transform } => (operated(a, transform), operated(b, transform)),
    };
    if rows == 0 || cols == 0 || matches!(kind, Level3Kind::RankTwoK { .. }) && a.cols == 0 {
        return Ok(BlasRoute::Reference);
    }

    // SYMM/HEMM have no transform argument; rank-2k has one shared transform.
    // Materialize logical inputs when they cannot both use NoTrans.
    let Some((a, _)) = prepare_matrix(a, packed_a, capabilities, layout, |op| {
        op == BlasTransform::None
    }) else {
        return unsupported(dispatch);
    };
    let Some((b, _)) = prepare_matrix(b, packed_b, capabilities, layout, |op| {
        op == BlasTransform::None
    }) else {
        return unsupported(dispatch);
    };
    let c = BlasMatrixMut::new(
        candidate,
        rows,
        cols,
        leading_dimension(layout, rows, cols),
        layout,
    );
    let mut call = match (kind, blas_operation) {
        (Level3Kind::SelfAdjoint { side }, BlasOperation::Symm) => BlasCall::Symm {
            side,
            triangle,
            alpha,
            a,
            b,
            beta,
            c,
        },
        (Level3Kind::SelfAdjoint { side }, BlasOperation::Hemm) => BlasCall::Hemm {
            side,
            triangle,
            alpha,
            a,
            b,
            beta,
            c,
        },
        (Level3Kind::RankTwoK { .. }, BlasOperation::Syr2k) => BlasCall::Syr2k {
            triangle,
            transform: BlasTransform::None,
            alpha,
            a,
            b,
            beta,
            c,
        },
        (Level3Kind::RankTwoK { .. }, BlasOperation::Her2k) => BlasCall::Her2k {
            triangle,
            transform: BlasTransform::None,
            alpha,
            a,
            b,
            beta: beta.real_component(),
            c,
        },
        _ => return unsupported(dispatch),
    };
    provide(&mut call, capabilities, None, dispatch, accuracy, workers)
}

pub(crate) struct TriangularVectorRequest<'a, S: NumericScalar> {
    pub(crate) blas_operation: BlasOperation,
    pub(crate) transform: BlasTransform,
    pub(crate) diagonal: DiagonalKind,
    pub(crate) a: BlasStructuredMatrix<'a, S>,
    pub(crate) candidate: &'a mut [S],
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn triangular_vector<S: NumericScalar>(
    request: TriangularVectorRequest<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let TriangularVectorRequest {
        blas_operation,
        transform,
        diagonal,
        a,
        candidate,
        dispatch,
        accuracy,
        workers,
    } = request;
    let x = vector_mut(candidate);
    let mut call = match (blas_operation, a) {
        (
            BlasOperation::Trsv,
            BlasStructuredMatrix::Dense {
                matrix: a,
                triangle,
            },
        ) => BlasCall::Trsv {
            triangle,
            transform,
            diagonal,
            a,
            x,
        },
        (
            BlasOperation::Tbsv,
            BlasStructuredMatrix::Band {
                matrix: a,
                triangle: Some(triangle),
            },
        ) => BlasCall::Tbsv {
            triangle,
            transform,
            diagonal,
            a,
            x,
        },
        (BlasOperation::Tpsv, BlasStructuredMatrix::Packed(a)) => BlasCall::Tpsv {
            transform,
            diagonal,
            a,
            x,
        },
        _ => return unsupported(dispatch),
    };
    invoke(&mut call, dispatch, accuracy, workers)
}

pub(crate) struct RotationRequest<'a, S: scioxide_core::RealScalar> {
    pub(crate) blas_operation: BlasOperation,
    pub(crate) x: &'a mut [S],
    pub(crate) y: &'a mut [S],
    pub(crate) cosine: S,
    pub(crate) sine: S,
    pub(crate) parameters: Option<[S; 5]>,
    pub(crate) dispatch: BlasDispatch<'a>,
    pub(crate) accuracy: scioxide_core::Accuracy,
    pub(crate) workers: NonZeroUsize,
}

pub(crate) fn rotate<S: scioxide_core::RealScalar>(
    request: RotationRequest<'_, S>,
) -> Result<BlasRoute, SciOxideError> {
    let RotationRequest {
        blas_operation,
        x,
        y,
        cosine,
        sine,
        parameters,
        dispatch,
        accuracy,
        workers,
    } = request;
    let x = vector_mut(x);
    let y = vector_mut(y);
    let mut call = match blas_operation {
        BlasOperation::Rot => BlasCall::Rot { x, y, cosine, sine },
        BlasOperation::Rotm => BlasCall::Rotm {
            x,
            y,
            parameters: parameters.ok_or_else(|| {
                provider_contract_failure(dispatch, "missing rotation parameters")
            })?,
        },
        _ => return unsupported(dispatch),
    };
    invoke::<S>(&mut call, dispatch, accuracy, workers)
}

pub(crate) fn rotation_generator<R: scioxide_core::RealScalar>(
    a: R,
    b: R,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<Option<crate::GivensRotationResult<R>>, SciOxideError> {
    let (mut r, mut z, mut cosine, mut sine) = (a, b, R::ZERO, R::ZERO);
    let route = invoke::<R>(
        &mut BlasCall::Rotg {
            a: &mut r,
            b: &mut z,
            cosine: &mut cosine,
            sine: &mut sine,
        },
        dispatch,
        accuracy,
        workers,
    )?;
    if route == BlasRoute::Reference {
        return Ok(None);
    }
    Ok(Some(crate::GivensRotationResult {
        r,
        z,
        rotation: crate::GivensRotation::new(cosine, sine)?,
    }))
}

type ModifiedGeneratorResult<R> = Option<(R, R, R, [R; 5])>;

pub(crate) fn modified_rotation_generator<R: scioxide_core::RealScalar>(
    d1: R,
    d2: R,
    x1: R,
    y1: R,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<ModifiedGeneratorResult<R>, SciOxideError> {
    let (mut d1, mut d2, mut x1) = (d1, d2, x1);
    let mut parameters = [R::ZERO; 5];
    let route = invoke::<R>(
        &mut BlasCall::Rotmg {
            d1: &mut d1,
            d2: &mut d2,
            x1: &mut x1,
            y1,
            parameters: &mut parameters,
        },
        dispatch,
        accuracy,
        workers,
    )?;
    Ok((route == BlasRoute::Provider).then_some((d1, d2, x1, parameters)))
}

pub(crate) fn wide_dot(
    left: VectorView<'_, f32>,
    right: VectorView<'_, f32>,
    offset: Option<f32>,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<Option<f64>, SciOxideError> {
    let (x, y) = (vector(left), vector(right));
    if let Some(offset) = offset {
        let mut result = 0.0_f32;
        let route = invoke::<f32>(
            &mut BlasCall::OffsetDot {
                x,
                y,
                offset,
                result: &mut result,
            },
            dispatch,
            accuracy,
            workers,
        )?;
        Ok((route == BlasRoute::Provider).then_some(f64::from(result)))
    } else {
        let mut result = 0.0_f64;
        let route = invoke::<f32>(
            &mut BlasCall::WideDot {
                x,
                y,
                result: &mut result,
            },
            dispatch,
            accuracy,
            workers,
        )?;
        Ok((route == BlasRoute::Provider).then_some(result))
    }
}

/// Row-major when the provider supports it, otherwise column-major.
pub(crate) fn preferred_layout(capabilities: &BlasCapabilities) -> BlasLayout {
    if capabilities.supports_layout(BlasLayout::RowMajor) {
        BlasLayout::RowMajor
    } else {
        BlasLayout::ColumnMajor
    }
}

pub(crate) fn matrix_index(
    layout: BlasLayout,
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) -> usize {
    match layout {
        BlasLayout::RowMajor => row * cols + col,
        BlasLayout::ColumnMajor => col * rows + row,
    }
}

pub(crate) fn leading_dimension(layout: BlasLayout, rows: usize, cols: usize) -> usize {
    match layout {
        BlasLayout::RowMajor => cols.max(1),
        BlasLayout::ColumnMajor => rows.max(1),
    }
}

fn direct_matrix_encoding<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    capabilities: &BlasCapabilities,
    layout: BlasLayout,
    allowed: impl Fn(BlasTransform) -> bool,
) -> Option<MatrixEncoding> {
    if !capabilities.supports_layout(layout) {
        return None;
    }
    let conjugated = matrix.conjugated && !S::IS_REAL;
    let normal = if conjugated {
        BlasTransform::Conjugate
    } else {
        BlasTransform::None
    };
    let transposed = if conjugated {
        BlasTransform::Adjoint
    } else {
        BlasTransform::Transpose
    };
    let (minor, major) = match layout {
        BlasLayout::RowMajor => (matrix.col_stride, matrix.row_stride),
        BlasLayout::ColumnMajor => (matrix.row_stride, matrix.col_stride),
    };
    let (major_extent, minor_extent) = match layout {
        BlasLayout::RowMajor => (matrix.rows, matrix.cols),
        BlasLayout::ColumnMajor => (matrix.cols, matrix.rows),
    };
    let encodings = [
        ((minor_extent <= 1 || minor == 1) && (major_extent <= 1 || major >= minor_extent.max(1)))
            .then_some(MatrixEncoding {
                rows: matrix.rows,
                cols: matrix.cols,
                layout,
                leading_dimension: major.max(minor_extent.max(1)),
                transform: normal,
            }),
        ((major_extent <= 1 || major == 1) && (minor_extent <= 1 || minor >= major_extent.max(1)))
            .then_some(MatrixEncoding {
                rows: matrix.cols,
                cols: matrix.rows,
                layout,
                leading_dimension: minor.max(major_extent.max(1)),
                transform: transposed,
            }),
    ];
    // CBLAS has no conjugate-without-transpose matrix argument.
    encodings.into_iter().flatten().find(|encoding| {
        encoding.transform != BlasTransform::Conjugate
            && allowed(encoding.transform)
            && capabilities.supports_transform(encoding.transform)
    })
}

fn matrix_operand<S: NumericScalar>(values: &[S], encoding: MatrixEncoding) -> BlasMatrix<'_, S> {
    BlasMatrix::new(
        values,
        encoding.rows,
        encoding.cols,
        encoding.leading_dimension,
        encoding.layout,
    )
}

fn prepare_matrix<'a, S: NumericScalar>(
    matrix: MatrixView<'a, S>,
    packed: &'a mut [S],
    capabilities: &BlasCapabilities,
    layout: BlasLayout,
    allowed: impl Fn(BlasTransform) -> bool,
) -> Option<(BlasMatrix<'a, S>, BlasTransform)> {
    if let Some(encoding) =
        direct_matrix_encoding(matrix, capabilities, layout, &allowed).filter(|encoding| {
            aligned(matrix.data, capabilities.minimum_alignment())
                && integer_width_supported(
                    capabilities,
                    &[encoding.rows, encoding.cols, encoding.leading_dimension],
                )
        })
    {
        return Some((matrix_operand(matrix.data, encoding), encoding.transform));
    }
    let len = matrix.rows.checked_mul(matrix.cols)?;
    let ld = leading_dimension(layout, matrix.rows, matrix.cols);
    if packed.len() < len
        || !capabilities.supports_layout(layout)
        || !capabilities.supports_transform(BlasTransform::None)
        || !allowed(BlasTransform::None)
        || !aligned(packed, capabilities.minimum_alignment())
        || !integer_width_supported(capabilities, &[matrix.rows, matrix.cols, ld])
    {
        return None;
    }
    for row in 0..matrix.rows {
        for col in 0..matrix.cols {
            packed[matrix_index(layout, matrix.rows, matrix.cols, row, col)] =
                matrix.get_element(row, col);
        }
    }
    Some((
        BlasMatrix::new(&packed[..len], matrix.rows, matrix.cols, ld, layout),
        BlasTransform::None,
    ))
}

fn aligned<S>(values: &[S], alignment: usize) -> bool {
    values.is_empty() || (values.as_ptr() as usize).is_multiple_of(alignment)
}

// CBLAS dimensions, strides and leading dimensions are signed integers of the
// provider's ABI width; any value above that maximum would be truncated.
fn integer_width_supported(capabilities: &BlasCapabilities, values: &[usize]) -> bool {
    let maximum = match capabilities.integer_width() {
        BlasIntegerWidth::Lp64 => i32::MAX as usize,
        BlasIntegerWidth::Ilp64 => i64::MAX as usize,
    };
    values.iter().all(|&value| value <= maximum)
}

fn provider_workers(capabilities: &BlasCapabilities, workers: NonZeroUsize) -> NonZeroUsize {
    match capabilities.threading() {
        BlasThreading::SingleThreaded => NonZeroUsize::MIN,
        BlasThreading::Budgeted => workers,
    }
}

pub(crate) fn unsupported(dispatch: BlasDispatch<'_>) -> Result<BlasRoute, SciOxideError> {
    match dispatch.fallback() {
        BlasFallbackPolicy::Reference => Ok(BlasRoute::Reference),
        BlasFallbackPolicy::Error => Err(SciOxideError::capability_unavailable(Capability::Blas)),
    }
}

fn provider_contract_failure(dispatch: BlasDispatch<'_>, detail: &'static str) -> SciOxideError {
    SciOxideError::capability_failure(
        Capability::Blas,
        alloc::format!("{}: {detail}", dispatch.provider().name()),
    )
}

/// A non-finite provider result is reported with callback origin before the
/// output is written, because SciOxide cannot tell provider overflow from a
/// provider defect.
fn provider_non_finite_failure(dispatch: BlasDispatch<'_>) -> SciOxideError {
    SciOxideError::non_finite(
        NonFiniteOrigin::Callback,
        alloc::format!(
            "{} returned a non-finite result",
            dispatch.provider().name()
        ),
    )
}

fn provider_error(
    dispatch: BlasDispatch<'_>,
    error: scioxide_core::CapabilityError,
) -> SciOxideError {
    SciOxideError::capability_failure(
        Capability::Blas,
        alloc::format!("{}: {}", dispatch.provider().name(), error.detail()),
    )
}

fn operated<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    transform: BlasTransform,
) -> MatrixView<'_, S> {
    match transform {
        BlasTransform::None => matrix,
        BlasTransform::Transpose => matrix.transpose(),
        BlasTransform::Adjoint => matrix.adjoint(),
        BlasTransform::Conjugate => matrix.conjugate(),
    }
}

/// Conjugation of a real vector is the identity, so it needs no capability.
fn vector<S: NumericScalar>(view: VectorView<'_, S>) -> BlasVector<'_, S> {
    BlasVector::new(
        view.data,
        view.len,
        view.stride,
        view.conjugated && !S::IS_REAL,
    )
}
fn vector_mut<S>(values: &mut [S]) -> BlasVectorMut<'_, S> {
    BlasVectorMut::new(values, values.len(), 1)
}
pub(crate) fn swap<S: NumericScalar>(
    x: &mut [S],
    y: &mut [S],
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    invoke(
        &mut BlasCall::Swap {
            x: vector_mut(x),
            y: vector_mut(y),
        },
        dispatch,
        accuracy,
        workers,
    )
}

fn project_call<'a, S: NumericScalar, T>(
    call: &'a mut BlasCall<'_, S>,
) -> Result<BlasCall<'a, T>, SciOxideError>
where
    T: NumericScalar + TryFrom<ScalarValue, Error = SciOxideError>,
    T::Real: TryFrom<ScalarValue, Error = SciOxideError>,
    for<'b> &'b [T]: TryFrom<ScalarSlice<'b>, Error = SciOxideError>,
    for<'b> &'b mut [T]: TryFrom<ScalarSliceMut<'b>, Error = SciOxideError>,
    for<'b> &'b [T::Real]: TryFrom<ScalarSlice<'b>, Error = SciOxideError>,
    for<'b> &'b mut [T::Real]: TryFrom<ScalarSliceMut<'b>, Error = SciOxideError>,
{
    // The same exact slice conversion preserves every descriptor's metadata.
    // These local macros share operand plumbing without another scalar dispatch.
    macro_rules! operand {
        (BlasVector, $input:ident, $element:ty) => {
            BlasVector {
                values: <$element>::slice($input.values).try_into()?,
                len: $input.len,
                stride: $input.stride,
                conjugated: $input.conjugated,
            }
        };
        (BlasVectorMut, $input:ident, $element:ty) => {
            BlasVectorMut {
                values: <$element>::slice_mut(&mut *$input.values).try_into()?,
                len: $input.len,
                stride: $input.stride,
            }
        };
        (BlasMatrix, $input:ident, $element:ty) => {
            BlasMatrix {
                values: <$element>::slice($input.values).try_into()?,
                rows: $input.rows,
                cols: $input.cols,
                leading_dimension: $input.leading_dimension,
                layout: $input.layout,
            }
        };
        (BlasMatrixMut, $input:ident, $element:ty) => {
            BlasMatrixMut {
                values: <$element>::slice_mut(&mut *$input.values).try_into()?,
                rows: $input.rows,
                cols: $input.cols,
                leading_dimension: $input.leading_dimension,
                layout: $input.layout,
            }
        };
        (BlasBandMatrix, $input:ident, $element:ty) => {
            BlasBandMatrix {
                values: <$element>::slice($input.values).try_into()?,
                rows: $input.rows,
                cols: $input.cols,
                lower: $input.lower,
                upper: $input.upper,
                leading_dimension: $input.leading_dimension,
            }
        };
        (BlasPackedMatrix, $input:ident, $element:ty) => {
            BlasPackedMatrix {
                values: <$element>::slice($input.values).try_into()?,
                order: $input.order,
                triangle: $input.triangle,
            }
        };
        (BlasPackedMatrixMut, $input:ident, $element:ty) => {
            BlasPackedMatrixMut {
                values: <$element>::slice_mut(&mut *$input.values).try_into()?,
                order: $input.order,
                triangle: $input.triangle,
            }
        };
    }
    macro_rules! scalar_result {
        ($input:ident, $source:ty, $target:ty) => {{
            // from_mut supplies exactly one element; exact projections preserve length.
            let values: &mut [$target] =
                <$source>::slice_mut(core::slice::from_mut(&mut **$input)).try_into()?;
            &mut values[0]
        }};
    }
    Ok(match call {
        BlasCall::Copy { x, y } => BlasCall::Copy {
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Scale { alpha, x } => BlasCall::Scale {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVectorMut, x, S),
        },
        BlasCall::Axpy { alpha, x, y } => BlasCall::Axpy {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Dotu { x, y, result } => BlasCall::Dotu {
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            result: scalar_result!(result, S, T),
        },
        BlasCall::Dotc { x, y, result } => BlasCall::Dotc {
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            result: scalar_result!(result, S, T),
        },
        BlasCall::Norm2 { x, result } => BlasCall::Norm2 {
            x: operand!(BlasVector, x, S),
            result: scalar_result!(result, S::Real, T::Real),
        },
        BlasCall::Asum { x, result } => BlasCall::Asum {
            x: operand!(BlasVector, x, S),
            result: scalar_result!(result, S::Real, T::Real),
        },
        BlasCall::Iamax { x, result } => BlasCall::Iamax {
            x: operand!(BlasVector, x, S),
            result,
        },
        BlasCall::Gemv {
            transform,
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Gemv {
            transform: *transform,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Gemm {
            left_transform,
            right_transform,
            alpha,
            a,
            b,
            beta,
            c,
        } => BlasCall::Gemm {
            left_transform: *left_transform,
            right_transform: *right_transform,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrix, b, S),
            beta: S::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
        BlasCall::Syrk {
            transform,
            triangle,
            alpha,
            a,
            beta,
            c,
        } => BlasCall::Syrk {
            transform: *transform,
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            beta: S::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
        BlasCall::Herk {
            transform,
            triangle,
            alpha,
            a,
            beta,
            c,
        } => BlasCall::Herk {
            transform: *transform,
            triangle: *triangle,
            alpha: S::Real::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            beta: S::Real::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
        BlasCall::Trmm {
            transform,
            side,
            triangle,
            diagonal,
            alpha,
            a,
            b,
        } => BlasCall::Trmm {
            transform: *transform,
            side: *side,
            triangle: *triangle,
            diagonal: *diagonal,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrixMut, b, S),
        },
        BlasCall::Trsm {
            transform,
            side,
            triangle,
            diagonal,
            alpha,
            a,
            b,
        } => BlasCall::Trsm {
            transform: *transform,
            side: *side,
            triangle: *triangle,
            diagonal: *diagonal,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrixMut, b, S),
        },
        BlasCall::Swap { x, y } => BlasCall::Swap {
            x: operand!(BlasVectorMut, x, S),
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Rot { x, y, cosine, sine } => BlasCall::Rot {
            x: operand!(BlasVectorMut, x, S::Real),
            y: operand!(BlasVectorMut, y, S::Real),
            cosine: S::Real::value(*cosine).try_into()?,
            sine: S::Real::value(*sine).try_into()?,
        },
        BlasCall::Rotm { x, y, parameters } => BlasCall::Rotm {
            x: operand!(BlasVectorMut, x, S::Real),
            y: operand!(BlasVectorMut, y, S::Real),
            parameters: [
                S::Real::value(parameters[0]).try_into()?,
                S::Real::value(parameters[1]).try_into()?,
                S::Real::value(parameters[2]).try_into()?,
                S::Real::value(parameters[3]).try_into()?,
                S::Real::value(parameters[4]).try_into()?,
            ],
        },
        BlasCall::Rotg { a, b, cosine, sine } => BlasCall::Rotg {
            a: scalar_result!(a, S::Real, T::Real),
            b: scalar_result!(b, S::Real, T::Real),
            cosine: scalar_result!(cosine, S::Real, T::Real),
            sine: scalar_result!(sine, S::Real, T::Real),
        },
        BlasCall::Rotmg {
            d1,
            d2,
            x1,
            y1,
            parameters,
        } => BlasCall::Rotmg {
            d1: scalar_result!(d1, S::Real, T::Real),
            d2: scalar_result!(d2, S::Real, T::Real),
            x1: scalar_result!(x1, S::Real, T::Real),
            y1: S::Real::value(*y1).try_into()?,
            parameters: {
                let values: &mut [T::Real] = S::Real::slice_mut(&mut parameters[..]).try_into()?;
                let len = values.len();
                values
                    .try_into()
                    .map_err(|_| SciOxideError::dimension_mismatch([5], [len]))?
            },
        },
        BlasCall::WideDot { x, y, result } => BlasCall::WideDot {
            x: *x,
            y: *y,
            result,
        },
        BlasCall::OffsetDot {
            x,
            y,
            offset,
            result,
        } => BlasCall::OffsetDot {
            x: *x,
            y: *y,
            offset: *offset,
            result,
        },
        BlasCall::Gbmv {
            transform,
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Gbmv {
            transform: *transform,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasBandMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Symv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Symv {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Hemv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Hemv {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Sbmv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Sbmv {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasBandMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Hbmv {
            triangle,
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Hbmv {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasBandMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Spmv {
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Spmv {
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasPackedMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Hpmv {
            alpha,
            a,
            x,
            beta,
            y,
        } => BlasCall::Hpmv {
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasPackedMatrix, a, S),
            x: operand!(BlasVector, x, S),
            beta: S::value(*beta).try_into()?,
            y: operand!(BlasVectorMut, y, S),
        },
        BlasCall::Trsv {
            triangle,
            transform,
            diagonal,
            a,
            x,
        } => BlasCall::Trsv {
            triangle: *triangle,
            transform: *transform,
            diagonal: *diagonal,
            a: operand!(BlasMatrix, a, S),
            x: operand!(BlasVectorMut, x, S),
        },
        BlasCall::Tbsv {
            triangle,
            transform,
            diagonal,
            a,
            x,
        } => BlasCall::Tbsv {
            triangle: *triangle,
            transform: *transform,
            diagonal: *diagonal,
            a: operand!(BlasBandMatrix, a, S),
            x: operand!(BlasVectorMut, x, S),
        },
        BlasCall::Tpsv {
            transform,
            diagonal,
            a,
            x,
        } => BlasCall::Tpsv {
            transform: *transform,
            diagonal: *diagonal,
            a: operand!(BlasPackedMatrix, a, S),
            x: operand!(BlasVectorMut, x, S),
        },
        BlasCall::Geru { alpha, x, y, a } => BlasCall::Geru {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            a: operand!(BlasMatrixMut, a, S),
        },
        BlasCall::Gerc { alpha, x, y, a } => BlasCall::Gerc {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            a: operand!(BlasMatrixMut, a, S),
        },
        BlasCall::Syr {
            triangle,
            alpha,
            x,
            a,
        } => BlasCall::Syr {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            a: operand!(BlasMatrixMut, a, S),
        },
        BlasCall::Syr2 {
            triangle,
            alpha,
            x,
            y,
            a,
        } => BlasCall::Syr2 {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            a: operand!(BlasMatrixMut, a, S),
        },
        BlasCall::Her {
            triangle,
            alpha,
            x,
            a,
        } => BlasCall::Her {
            triangle: *triangle,
            alpha: S::Real::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            a: operand!(BlasMatrixMut, a, S),
        },
        BlasCall::Her2 {
            triangle,
            alpha,
            x,
            y,
            a,
        } => BlasCall::Her2 {
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            a: operand!(BlasMatrixMut, a, S),
        },
        BlasCall::Spr { alpha, x, a } => BlasCall::Spr {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            a: operand!(BlasPackedMatrixMut, a, S),
        },
        BlasCall::Spr2 { alpha, x, y, a } => BlasCall::Spr2 {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            a: operand!(BlasPackedMatrixMut, a, S),
        },
        BlasCall::Hpr { alpha, x, a } => BlasCall::Hpr {
            alpha: S::Real::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            a: operand!(BlasPackedMatrixMut, a, S),
        },
        BlasCall::Hpr2 { alpha, x, y, a } => BlasCall::Hpr2 {
            alpha: S::value(*alpha).try_into()?,
            x: operand!(BlasVector, x, S),
            y: operand!(BlasVector, y, S),
            a: operand!(BlasPackedMatrixMut, a, S),
        },
        BlasCall::Symm {
            side,
            triangle,
            alpha,
            a,
            b,
            beta,
            c,
        } => BlasCall::Symm {
            side: *side,
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrix, b, S),
            beta: S::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
        BlasCall::Hemm {
            side,
            triangle,
            alpha,
            a,
            b,
            beta,
            c,
        } => BlasCall::Hemm {
            side: *side,
            triangle: *triangle,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrix, b, S),
            beta: S::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
        BlasCall::Syr2k {
            triangle,
            transform,
            alpha,
            a,
            b,
            beta,
            c,
        } => BlasCall::Syr2k {
            triangle: *triangle,
            transform: *transform,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrix, b, S),
            beta: S::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
        BlasCall::Her2k {
            triangle,
            transform,
            alpha,
            a,
            b,
            beta,
            c,
        } => BlasCall::Her2k {
            triangle: *triangle,
            transform: *transform,
            alpha: S::value(*alpha).try_into()?,
            a: operand!(BlasMatrix, a, S),
            b: operand!(BlasMatrix, b, S),
            beta: S::Real::value(*beta).try_into()?,
            c: operand!(BlasMatrixMut, c, S),
        },
    })
}

/// Structural and capability validation shared by every typed operand.
struct OperandValidator<'a> {
    caps: &'a BlasCapabilities,
    layout: Option<BlasLayout>,
}
impl OperandValidator<'_> {
    fn dimensions(&self, values: &[usize]) -> bool {
        integer_width_supported(self.caps, values)
    }
    fn layout(&mut self, layout: BlasLayout) -> bool {
        let valid =
            self.caps.supports_layout(layout) && self.layout.is_none_or(|prior| prior == layout);
        self.layout = Some(layout);
        valid
    }
    fn vector<S>(&self, values: &[S], len: usize, stride: usize, conjugated: bool) -> bool {
        let span = if len == 0 {
            Some(0)
        } else {
            (len - 1).checked_mul(stride).and_then(|n| n.checked_add(1))
        };
        self.dimensions(&[len, stride])
            && (len == 0 || stride > 0)
            && span.is_some_and(|span| span <= values.len())
            && aligned(values, self.caps.minimum_alignment())
            && self.caps.supports_transform(if conjugated {
                BlasTransform::Conjugate
            } else {
                BlasTransform::None
            })
    }
    fn matrix<S>(
        &mut self,
        values: &[S],
        rows: usize,
        cols: usize,
        ld: usize,
        layout: BlasLayout,
    ) -> bool {
        let (major, minor) = match layout {
            BlasLayout::RowMajor => (rows, cols),
            BlasLayout::ColumnMajor => (cols, rows),
        };
        let span = if rows == 0 || cols == 0 {
            Some(0)
        } else {
            (major - 1)
                .checked_mul(ld)
                .and_then(|n| n.checked_add(minor))
        };
        self.layout(layout)
            && self.dimensions(&[rows, cols, ld])
            && ld >= minor.max(1)
            && span.is_some_and(|span| span <= values.len())
            && aligned(values, self.caps.minimum_alignment())
    }
    fn band<S>(&mut self, a: &BlasBandMatrix<'_, S>) -> bool {
        let bandwidth = a.lower.checked_add(a.upper).and_then(|n| n.checked_add(1));
        // CBLAS addresses `(cols - 1) * ld + lower + upper + 1` values, so a
        // triangle sliced from a wider band need not hold `cols * ld`.
        let span = if a.cols == 0 {
            Some(0)
        } else {
            bandwidth.and_then(|n| {
                (a.cols - 1)
                    .checked_mul(a.leading_dimension)
                    .and_then(|m| m.checked_add(n))
            })
        };
        self.layout(BlasLayout::ColumnMajor)
            && self.dimensions(&[a.rows, a.cols, a.lower, a.upper, a.leading_dimension])
            && bandwidth.is_some_and(|n| a.leading_dimension >= n)
            && span.is_some_and(|n| n <= a.values.len())
            && aligned(a.values, self.caps.minimum_alignment())
    }
    fn packed<S>(&mut self, values: &[S], order: usize) -> bool {
        self.layout(BlasLayout::ColumnMajor)
            && self.dimensions(&[order])
            && order
                .checked_add(1)
                .and_then(|n| n.checked_mul(order))
                .map(|n| n / 2)
                .is_some_and(|n| n <= values.len())
            && aligned(values, self.caps.minimum_alignment())
    }
}
/// Whether a call has an empty extent: empty vectors, empty products and
/// zero-inner-dimension updates run natively without querying the provider.
fn empty_work<S: NumericScalar>(call: &BlasCall<'_, S>) -> bool {
    match call {
        BlasCall::Rotg { .. } | BlasCall::Rotmg { .. } => false,
        BlasCall::Scale { x, .. } => x.len == 0,
        BlasCall::Norm2 { x, .. } | BlasCall::Asum { x, .. } | BlasCall::Iamax { x, .. } => {
            x.len == 0
        }
        BlasCall::Copy { x, y } | BlasCall::Axpy { x, y, .. } => x.len == 0 || y.len == 0,
        BlasCall::Dotu { x, y, .. }
        | BlasCall::Dotc { x, y, .. }
        | BlasCall::Geru { x, y, .. }
        | BlasCall::Gerc { x, y, .. }
        | BlasCall::Syr2 { x, y, .. }
        | BlasCall::Her2 { x, y, .. }
        | BlasCall::Spr2 { x, y, .. }
        | BlasCall::Hpr2 { x, y, .. } => x.len == 0 || y.len == 0,
        BlasCall::WideDot { x, y, .. } | BlasCall::OffsetDot { x, y, .. } => {
            x.len == 0 || y.len == 0
        }
        BlasCall::Swap { x, y } => x.len == 0 || y.len == 0,
        BlasCall::Rot { x, y, .. } | BlasCall::Rotm { x, y, .. } => x.len == 0 || y.len == 0,
        BlasCall::Syr { x, .. }
        | BlasCall::Her { x, .. }
        | BlasCall::Spr { x, .. }
        | BlasCall::Hpr { x, .. } => x.len == 0,
        BlasCall::Gemv { x, y, .. }
        | BlasCall::Gbmv { x, y, .. }
        | BlasCall::Symv { x, y, .. }
        | BlasCall::Hemv { x, y, .. }
        | BlasCall::Sbmv { x, y, .. }
        | BlasCall::Hbmv { x, y, .. }
        | BlasCall::Spmv { x, y, .. }
        | BlasCall::Hpmv { x, y, .. } => x.len == 0 || y.len == 0,
        BlasCall::Trsv { x, .. } | BlasCall::Tbsv { x, .. } | BlasCall::Tpsv { x, .. } => {
            x.len == 0
        }
        BlasCall::Gemm { a, b, c, .. }
        | BlasCall::Symm { a, b, c, .. }
        | BlasCall::Hemm { a, b, c, .. }
        | BlasCall::Syr2k { a, b, c, .. }
        | BlasCall::Her2k { a, b, c, .. } => {
            [a.rows, a.cols, b.rows, b.cols, c.rows, c.cols].contains(&0)
        }
        BlasCall::Syrk { a, c, .. } | BlasCall::Herk { a, c, .. } => {
            [a.rows, a.cols, c.rows, c.cols].contains(&0)
        }
        BlasCall::Trmm { a, b, .. } | BlasCall::Trsm { a, b, .. } => {
            [a.rows, a.cols, b.rows, b.cols].contains(&0)
        }
    }
}

fn eligible_call<S: NumericScalar>(call: &BlasCall<'_, S>, caps: &BlasCapabilities) -> bool {
    if !base_eligible::<S>(caps, call.operation()) {
        return false;
    }
    let mut validate = OperandValidator { caps, layout: None };
    match call {
        BlasCall::Copy { x, y, .. } | BlasCall::Axpy { x, y, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (x.len == y.len)
        }
        BlasCall::Scale { x, .. } => validate.vector(x.values, x.len, x.stride, false),
        BlasCall::Dotu { x, y, .. } | BlasCall::Dotc { x, y, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, y.conjugated))
                && (x.len == y.len)
        }
        BlasCall::Norm2 { x, .. } | BlasCall::Asum { x, .. } | BlasCall::Iamax { x, .. } => {
            validate.vector(x.values, x.len, x.stride, x.conjugated)
        }
        BlasCall::Gemv {
            transform, a, x, y, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (operated_shape(a.rows, a.cols, *transform) == (y.len, x.len))
        }
        BlasCall::Gemm {
            left_transform,
            right_transform,
            a,
            b,
            c,
            ..
        } => {
            (caps.supports_transform(*left_transform)
                && *left_transform != BlasTransform::Conjugate)
                && (caps.supports_transform(*right_transform)
                    && *right_transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(b.values, b.rows, b.cols, b.leading_dimension, b.layout))
                && (validate.matrix(c.values, c.rows, c.cols, c.leading_dimension, c.layout))
                && (operated_shape(a.rows, a.cols, *left_transform).0 == c.rows)
                && (operated_shape(b.rows, b.cols, *right_transform).1 == c.cols)
                && (operated_shape(a.rows, a.cols, *left_transform).1
                    == operated_shape(b.rows, b.cols, *right_transform).0)
        }
        BlasCall::Syrk {
            transform, a, c, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(c.values, c.rows, c.cols, c.leading_dimension, c.layout))
                && (c.rows == c.cols && operated_shape(a.rows, a.cols, *transform).0 == c.rows)
                && (matches!(transform, BlasTransform::None | BlasTransform::Transpose))
        }
        BlasCall::Herk {
            transform, a, c, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(c.values, c.rows, c.cols, c.leading_dimension, c.layout))
                && (c.rows == c.cols && operated_shape(a.rows, a.cols, *transform).0 == c.rows)
                && (matches!(transform, BlasTransform::None | BlasTransform::Adjoint))
        }
        BlasCall::Trmm {
            transform,
            a,
            b,
            side,
            ..
        }
        | BlasCall::Trsm {
            transform,
            a,
            b,
            side,
            ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(b.values, b.rows, b.cols, b.leading_dimension, b.layout))
                && (a.rows == a.cols
                    && a.rows
                        == if *side == MatrixSide::Left {
                            b.rows
                        } else {
                            b.cols
                        })
        }
        BlasCall::Swap { x, y, .. } => {
            (validate.vector(x.values, x.len, x.stride, false))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (x.len == y.len)
        }
        BlasCall::Rot { x, y, .. } | BlasCall::Rotm { x, y, .. } => {
            (validate.vector(x.values, x.len, x.stride, false))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (x.len == y.len)
        }
        BlasCall::Rotg { .. } | BlasCall::Rotmg { .. } => true,
        BlasCall::WideDot { x, y, .. } | BlasCall::OffsetDot { x, y, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, y.conjugated))
                && (x.len == y.len)
        }
        BlasCall::Gbmv {
            transform, a, x, y, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.band(a))
                && (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (operated_shape(a.rows, a.cols, *transform) == (y.len, x.len))
        }
        BlasCall::Symv { a, x, y, .. } | BlasCall::Hemv { a, x, y, .. } => {
            (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (a.rows == a.cols && a.rows == x.len && x.len == y.len)
        }
        BlasCall::Sbmv { a, x, y, .. } | BlasCall::Hbmv { a, x, y, .. } => {
            (validate.band(a))
                && (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (a.rows == a.cols && a.rows == x.len && x.len == y.len)
        }
        BlasCall::Spmv { a, x, y, .. } | BlasCall::Hpmv { a, x, y, .. } => {
            (validate.packed(a.values, a.order))
                && (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, false))
                && (a.order == x.len && x.len == y.len)
        }
        BlasCall::Trsv {
            transform, a, x, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.vector(x.values, x.len, x.stride, false))
                && (a.rows == a.cols && a.rows == x.len)
        }
        BlasCall::Tbsv {
            transform, a, x, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.band(a))
                && (validate.vector(x.values, x.len, x.stride, false))
                && (a.rows == a.cols && a.rows == x.len)
        }
        BlasCall::Tpsv {
            transform, a, x, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.packed(a.values, a.order))
                && (validate.vector(x.values, x.len, x.stride, false))
                && (a.order == x.len)
        }
        BlasCall::Geru { x, y, a, .. } | BlasCall::Gerc { x, y, a, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, y.conjugated))
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && ((a.rows, a.cols) == (x.len, y.len))
        }
        BlasCall::Syr { x, a, .. } | BlasCall::Her { x, a, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (a.rows == a.cols && a.rows == x.len)
        }
        BlasCall::Syr2 { x, y, a, .. } | BlasCall::Her2 { x, y, a, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, y.conjugated))
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (a.rows == a.cols && a.rows == x.len)
                && (x.len == y.len)
        }
        BlasCall::Spr { x, a, .. } | BlasCall::Hpr { x, a, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.packed(a.values, a.order))
                && (a.order == x.len)
        }
        BlasCall::Spr2 { x, y, a, .. } | BlasCall::Hpr2 { x, y, a, .. } => {
            (validate.vector(x.values, x.len, x.stride, x.conjugated))
                && (validate.vector(y.values, y.len, y.stride, y.conjugated))
                && (validate.packed(a.values, a.order))
                && (a.order == x.len)
                && (x.len == y.len)
        }
        BlasCall::Symm { a, b, c, side, .. } | BlasCall::Hemm { a, b, c, side, .. } => {
            (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(b.values, b.rows, b.cols, b.leading_dimension, b.layout))
                && (validate.matrix(c.values, c.rows, c.cols, c.leading_dimension, c.layout))
                && (a.rows == a.cols
                    && a.rows
                        == if *side == MatrixSide::Left {
                            b.rows
                        } else {
                            b.cols
                        })
                && ((b.rows, b.cols) == (c.rows, c.cols))
        }
        BlasCall::Syr2k {
            transform, a, b, c, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(b.values, b.rows, b.cols, b.leading_dimension, b.layout))
                && (validate.matrix(c.values, c.rows, c.cols, c.leading_dimension, c.layout))
                && (c.rows == c.cols && operated_shape(a.rows, a.cols, *transform).0 == c.rows)
                && (matches!(transform, BlasTransform::None | BlasTransform::Transpose))
                && ((a.rows, a.cols) == (b.rows, b.cols))
        }
        BlasCall::Her2k {
            transform, a, b, c, ..
        } => {
            (caps.supports_transform(*transform) && *transform != BlasTransform::Conjugate)
                && (validate.matrix(a.values, a.rows, a.cols, a.leading_dimension, a.layout))
                && (validate.matrix(b.values, b.rows, b.cols, b.leading_dimension, b.layout))
                && (validate.matrix(c.values, c.rows, c.cols, c.leading_dimension, c.layout))
                && (c.rows == c.cols && operated_shape(a.rows, a.cols, *transform).0 == c.rows)
                && (matches!(transform, BlasTransform::None | BlasTransform::Adjoint))
                && ((a.rows, a.cols) == (b.rows, b.cols))
        }
    }
}
fn operated_shape(rows: usize, cols: usize, transform: BlasTransform) -> (usize, usize) {
    if matches!(transform, BlasTransform::Transpose | BlasTransform::Adjoint) {
        (cols, rows)
    } else {
        (rows, cols)
    }
}
fn finite_vector<S: NumericScalar>(values: &[S], len: usize, stride: usize) -> bool {
    values
        .iter()
        .step_by(stride.max(1))
        .take(len)
        .all(|v| v.has_finite_components())
}
fn finite_matrix<S: NumericScalar>(
    a: &BlasMatrixMut<'_, S>,
    triangle: Option<Triangle>,
    hermitian: bool,
    dispatch: BlasDispatch<'_>,
) -> Result<(), SciOxideError> {
    for row in 0..a.rows {
        for col in 0..a.cols {
            if triangle
                .is_some_and(|triangle| !crate::kernel::triangle_contains(triangle, row, col))
            {
                continue;
            }
            let value = a.values[matrix_index(a.layout, a.rows, a.cols, row, col)];
            if !value.has_finite_components() {
                return Err(provider_non_finite_failure(dispatch));
            }
            if hermitian && row == col && value.imaginary_component() != S::Real::ZERO {
                return Err(provider_contract_failure(
                    dispatch,
                    "provider returned a non-real Hermitian diagonal",
                ));
            }
        }
    }
    Ok(())
}
fn validate_results<S: NumericScalar>(
    call: &BlasCall<'_, S>,
    gemm_mask: Option<Triangle>,
    dispatch: BlasDispatch<'_>,
) -> Result<(), SciOxideError> {
    let finite = match call {
        BlasCall::Copy { y, .. } | BlasCall::Axpy { y, .. } => {
            finite_vector(y.values, y.len, y.stride)
        }
        BlasCall::Scale { x, .. }
        | BlasCall::Trsv { x, .. }
        | BlasCall::Tbsv { x, .. }
        | BlasCall::Tpsv { x, .. } => finite_vector(x.values, x.len, x.stride),
        BlasCall::Dotu { result, .. } | BlasCall::Dotc { result, .. } => {
            result.has_finite_components()
        }
        BlasCall::Norm2 { result, .. } | BlasCall::Asum { result, .. } => {
            if result.has_finite_components() && **result < S::Real::ZERO {
                return Err(provider_contract_failure(
                    dispatch,
                    "provider returned a negative reduction",
                ));
            }
            result.has_finite_components()
        }
        BlasCall::Iamax { result, x, .. } => {
            if **result >= x.len {
                return Err(provider_contract_failure(
                    dispatch,
                    "provider returned an out-of-range maximum index",
                ));
            }
            true
        }
        BlasCall::Gemv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Gemm { c, .. } => {
            finite_matrix(c, gemm_mask, false, dispatch)?;
            true
        }
        BlasCall::Symm { c, .. } | BlasCall::Hemm { c, .. } => {
            finite_matrix(c, None, false, dispatch)?;
            true
        }
        BlasCall::Syrk { c, triangle, .. } | BlasCall::Syr2k { c, triangle, .. } => {
            finite_matrix(c, Some(*triangle), false, dispatch)?;
            true
        }
        BlasCall::Herk { c, triangle, .. } | BlasCall::Her2k { c, triangle, .. } => {
            finite_matrix(c, Some(*triangle), true, dispatch)?;
            true
        }
        BlasCall::Trmm { b, .. } | BlasCall::Trsm { b, .. } => {
            finite_matrix(b, None, false, dispatch)?;
            true
        }
        BlasCall::Swap { x, y, .. } => {
            finite_vector(x.values, x.len, x.stride) && finite_vector(y.values, y.len, y.stride)
        }
        BlasCall::Rot { x, y, .. } | BlasCall::Rotm { x, y, .. } => {
            finite_vector(x.values, x.len, x.stride) && finite_vector(y.values, y.len, y.stride)
        }
        BlasCall::Rotg {
            a, b, cosine, sine, ..
        } => {
            a.has_finite_components()
                && b.has_finite_components()
                && cosine.has_finite_components()
                && sine.has_finite_components()
        }
        BlasCall::Rotmg {
            d1,
            d2,
            x1,
            parameters,
            ..
        } => {
            if parameters[0].has_finite_components()
                && ![-S::Real::TWO, -S::Real::ONE, S::Real::ZERO, S::Real::ONE]
                    .contains(&parameters[0])
            {
                return Err(provider_contract_failure(
                    dispatch,
                    "provider returned an invalid modified-rotation flag",
                ));
            }
            d1.has_finite_components()
                && d2.has_finite_components()
                && x1.has_finite_components()
                && parameters.iter().all(|v| v.has_finite_components())
        }
        BlasCall::WideDot { result, .. } => result.has_finite_components(),
        BlasCall::OffsetDot { result, .. } => result.has_finite_components(),
        BlasCall::Gbmv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Symv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Hemv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Sbmv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Hbmv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Spmv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Hpmv { y, .. } => finite_vector(y.values, y.len, y.stride),
        BlasCall::Geru { a, .. } | BlasCall::Gerc { a, .. } => {
            finite_matrix(a, None, false, dispatch)?;
            true
        }
        BlasCall::Syr { a, triangle, .. } | BlasCall::Syr2 { a, triangle, .. } => {
            finite_matrix(a, Some(*triangle), false, dispatch)?;
            true
        }
        BlasCall::Her { a, triangle, .. } | BlasCall::Her2 { a, triangle, .. } => {
            finite_matrix(a, Some(*triangle), true, dispatch)?;
            true
        }
        BlasCall::Spr { a, .. } | BlasCall::Spr2 { a, .. } => {
            a.values.iter().all(|v| v.has_finite_components())
        }
        BlasCall::Hpr { a, .. } | BlasCall::Hpr2 { a, .. } => {
            for i in 0..a.order {
                let index = match a.triangle {
                    Triangle::Upper => i * (i + 1) / 2 + i,
                    Triangle::Lower => i * a.order - i * i.saturating_sub(1) / 2,
                };
                if !a.values[index].has_finite_components() {
                    return Err(provider_non_finite_failure(dispatch));
                }
                if a.values[index].imaginary_component() != S::Real::ZERO {
                    return Err(provider_contract_failure(
                        dispatch,
                        "provider returned a non-real Hermitian diagonal",
                    ));
                }
            }
            a.values.iter().all(|v| v.has_finite_components())
        }
    };
    if finite {
        Ok(())
    } else {
        Err(provider_non_finite_failure(dispatch))
    }
}
/// Query capabilities once, then invoke and check a call that writes every output entry.
fn invoke<S: NumericScalar>(
    call: &mut BlasCall<'_, S>,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    if empty_work(call) {
        return Ok(BlasRoute::Reference);
    }
    provide(
        call,
        &dispatch.capabilities()?,
        None,
        dispatch,
        accuracy,
        workers,
    )
}

/// Invoke an eligible call against queried capabilities and validate its result.
/// `gemm_mask` restricts GEMM checks to the triangle a masked product writes.
/// Zero work runs natively under every fallback policy.
fn provide<S: NumericScalar>(
    call: &mut BlasCall<'_, S>,
    capabilities: &BlasCapabilities,
    gemm_mask: Option<Triangle>,
    dispatch: BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    workers: NonZeroUsize,
) -> Result<BlasRoute, SciOxideError> {
    if empty_work(call) {
        return Ok(BlasRoute::Reference);
    }
    if !eligible_call(call, capabilities)
        || (accuracy == scioxide_core::Accuracy::Exact
            && capabilities
                .operation_ulp(S::kind(), call.operation())
                .is_some_and(|ulp| ulp.maximum() != 0))
    {
        return unsupported(dispatch);
    }
    let workers = provider_workers(capabilities, workers);
    let outcome = match S::kind() {
        ScalarKind::F32 => dispatch
            .provider()
            .execute_f32(project_call::<S, f32>(call)?, workers),
        ScalarKind::F64 => dispatch
            .provider()
            .execute_f64(project_call::<S, f64>(call)?, workers),
        ScalarKind::Complex32 => dispatch
            .provider()
            .execute_complex32(project_call::<S, scioxide_core::Complex32>(call)?, workers),
        ScalarKind::Complex64 => dispatch
            .provider()
            .execute_complex64(project_call::<S, scioxide_core::Complex64>(call)?, workers),
    };
    outcome.map_err(|error| provider_error(dispatch, error))?;
    validate_results(call, gemm_mask, dispatch)?;
    Ok(BlasRoute::Provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlasProvider, BlasUlp};
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{CapabilityError, Complex32, Complex64, SciOxideErrorKind};
    struct Probe(AtomicUsize);
    macro_rules! reject {
        ($name:ident, $scalar:ty) => {
            fn $name(
                &self,
                _: BlasCall<'_, $scalar>,
                _: NonZeroUsize,
            ) -> Result<(), CapabilityError> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Err(CapabilityError::new("ineligible call reached provider"))
            }
        };
    }
    impl BlasProvider for Probe {
        fn name(&self) -> &'static str {
            "operand-validation-probe"
        }
        fn capabilities(&self) -> BlasCapabilities {
            BlasCapabilities::empty(BlasIntegerWidth::Lp64)
                .with_layout(BlasLayout::RowMajor)
                .with_transform(BlasTransform::None)
                .with_operation(ScalarKind::F64, BlasOperation::Gemv, BlasUlp::new(0))
                .unwrap()
        }
        reject!(execute_f32, f32);
        reject!(execute_f64, f64);
        reject!(execute_complex32, Complex32);
        reject!(execute_complex64, Complex64);
    }
    #[test]
    fn gemv_candidate_length_is_checked_before_projection_and_invocation() {
        let provider = Probe(AtomicUsize::new(0));
        for fallback in [BlasFallbackPolicy::Reference, BlasFallbackPolicy::Error] {
            let mut candidate = [7.0_f64];
            let mut call = BlasCall::Gemv {
                transform: BlasTransform::None,
                alpha: 1.0,
                a: BlasMatrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2, 2, BlasLayout::RowMajor),
                x: BlasVector::new(&[1.0, 2.0], 2, 1, false),
                beta: 0.0,
                y: BlasVectorMut::new(&mut candidate, 1, 1),
            };
            let result = invoke(
                &mut call,
                BlasDispatch::new(&provider, fallback),
                scioxide_core::Accuracy::Standard,
                NonZeroUsize::MIN,
            );
            match fallback {
                BlasFallbackPolicy::Reference => assert_eq!(result.unwrap(), BlasRoute::Reference),
                BlasFallbackPolicy::Error => assert_eq!(
                    result.unwrap_err().kind(),
                    SciOxideErrorKind::CapabilityUnavailable
                ),
            }
            assert_eq!(candidate, [7.0]);
            assert_eq!(provider.0.load(Ordering::Relaxed), 0);
        }
    }
}
